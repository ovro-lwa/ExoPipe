"""serve --tier snapshot (review pages 8779-8782): gate -> copy -> localize -> gate-copy -> carry -> test -> start -> hub ->
verify. Port of snapshot_serve.sh (+ the carry, HOLD rule and served test of snapshot_rechain_batch.sh:367-429).

gate       (unit snapshot_<name>_gate)  the serve-time exclusion gate on the chain run (G17; snapshot_serve.sh:26-27): the
           in-place gate (local source: core.legacy.run; source on the chain host: ssh with site PYTHON 'py36') and, for a
           local source, core.exclusions on the same rows (two readers must agree). Parity: core.exclusions only (the
           in-place gate writes its check file into the source run) and nothing for a remote source (the copy gate binds).
copy       (unit snapshot_<name>_copy)  G20 free space on the review disk >= SNAPSHOT_COPY_MIN_FREE_FACTOR x source size,
           checked before anything is copied or stopped (owed, ledger 1208); rsync -a minus SNAPSHOT_COPY_EXCLUDE
           (snapshot_serve.sh:28); G22 rsync status + `rsync -anc` byte check (class review_copy).
localize   (unit snapshot_<name>_localize)  snapshot_localize_tiers.sh DST in place - serve owns it in the window
           (decision 6, group 7) - must end LOCALIZE-TIERS-OK (G21).
gate-copy  (unit snapshot_<name>_gate_copy)  the gate again on the copy, right before any server is replaced
           (snapshot_serve.sh:47-48): in-place gate + core.exclusions, agreeing (class exclusion_check).
carry      (unit snapshot_<name>_carry)  --carry-from (default 'served'): G18 (no served run, a journal already in DST,
           no CARRY report), snapshot_carry_judgements.py OLD DST in place, the HOLD rule in process (G19: a missed
           non-FP verdict holds the serve; carry_missed.csv written either way).
test, start, hub, verify: lib/serve/flow (expected sources = queue length, snapshot_rechain_batch.sh:419-426).
"""
import os
import subprocess

from exopipe.phase3.core.record import NoInput
from exopipe.phase3.lib.serve import exclusion_gate, flow, journals, scripts

STAGES = ('gate', 'copy', 'localize', 'gate-copy', 'carry', 'test', 'start', 'hub', 'verify')


def _S():
    from exopipe.phase3.config import CFG
    return CFG.serve


def split_source(src):
    """'host:/path' -> ('host', '/path'); '/path' -> (None, '/path')."""
    if src and ':' in src and not src.startswith('/'):
        h, p = src.split(':', 1)
        return h, p
    return None, src


def source_bytes(host, path):
    if host is None:
        tot = 0
        for dp, dn, fn in os.walk(path):
            dn[:] = [d for d in dn if d not in _S().SNAPSHOT_COPY_EXCLUDE]
            for f in fn:
                try:
                    tot += os.lstat(os.path.join(dp, f)).st_size
                except OSError:
                    pass
        return tot
    from exopipe.phase3.config import CFG
    ex = ' '.join(f'--exclude={x}' for x in _S().SNAPSHOT_COPY_EXCLUDE)
    p = subprocess.run(['ssh', '-n', '-o', 'BatchMode=yes', host, f'du -sb {ex} {path}'], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, env=dict(os.environ, **CFG.runner.SSH_PREFIX_ENV), timeout=600)
    return int(p.stdout.split()[0]) if p.returncode == 0 else None


def _check_file(plan, what, scan):
    p = os.path.join(plan.out, '_records', 'serve', 'gates', f'{plan.run_name}_{what}.txt')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        f.write(('EXCLUSION-GATE PASS ' if not scan.hits else 'EXCLUSION-GATE REFUSED ') + f'{plan.run_name} ({what}): '
                + scan.summary() + '\n')
    return p


def gate_stage(run, plan):
    host, src = split_source(plan.source)
    with run.unit(f'snapshot_{plan.run_name}_gate', stage='gate', tier='snapshot', source=plan.source) as u:
        flow.excl_note(u)
        if not plan.source:
            u.refuse('no source: give --source [HOST:]PATH of the chain run (or set site CHAIN_HOST)')
        if host is None:
            if not os.path.isdir(src):
                u.refuse(f'G17: the chain run {src} does not exist')
            scan = exclusion_gate.scan_snapshot_run(src)
            if plan.parity:
                u.note('parity: core.exclusions only on the source (the in-place gate writes into the source run)')
            else:
                rc, out, refusals = exclusion_gate.run_inplace_gate(src, unit=u)
                ok, detail = exclusion_gate.agree(scan, refusals, rc)
                u.check('G17_two_readers_agree', ok, detail=detail)
                if not ok:
                    u.refuse(f'G17 serve-time exclusion gate: {detail}')
            if scan.hits:
                u.refuse(f'G17 serve-time exclusion gate REFUSED {plan.run_name}: {scan.summary()} - serve stopped, servers '
                         f'untouched (re-chain those cells first)')
            u.add_output(_check_file(plan, 'source', scan), 'exclusion_check', run=src)
        else:
            if plan.parity:
                raise NoInput(f'parity: the source gate on {host} is not run (it writes into the chain run); the gate on '
                              f'the copy binds')
            rc, out, _ = exclusion_gate.run_inplace_gate(src, host=host, unit=u)
            u.check('G17_inplace_gate_chain_host', rc == 0, value=rc, detail=scripts.last_line(out))
            if rc != 0:
                u.refuse(f'G17 serve-time exclusion gate REFUSED {plan.run_name} on {host} (rc {rc}): {scripts.last_line(out)}'
                         f' - serve stopped, servers untouched')
            p = os.path.join(plan.out, '_records', 'serve', 'gates', f'{plan.run_name}_source.txt')
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, 'w').write(scripts.last_line(out) + '\n')
            u.add_output(p, 'exclusion_check')


def copy_stage(run, plan):
    S = _S()
    host, src = split_source(plan.source)
    DST = plan.U
    with run.unit(f'snapshot_{plan.run_name}_copy', stage='copy', tier='snapshot', source=plan.source) as u:
        need = source_bytes(host, src)
        if need is None:
            u.refuse(f'G20 review-disk space: the size of {plan.source} could not be read')
        os.makedirs(os.path.dirname(DST), exist_ok=True)
        st = os.statvfs(os.path.dirname(DST))
        free = st.f_bavail * st.f_frsize
        u.check('G20_free_space', free >= S.SNAPSHOT_COPY_MIN_FREE_FACTOR * need, value=free,
                detail=f'{free / 1e9:.1f} GB free, source {need / 1e9:.2f} GB')
        if free < S.SNAPSHOT_COPY_MIN_FREE_FACTOR * need:
            u.refuse(f'G20 review-disk space: {free / 1e9:.1f} GB free under {os.path.dirname(DST)} < '
                     f'{S.SNAPSHOT_COPY_MIN_FREE_FACTOR} x {need / 1e9:.2f} GB - nothing copied, servers untouched '
                     f'(ledger 1208)')
        ex = []
        for x in S.SNAPSHOT_COPY_EXCLUDE:
            ex += ['--exclude', x]
        from exopipe.phase3.config import CFG
        p = subprocess.run(['rsync', '-a'] + ex + [plan.source.rstrip('/') + '/', DST + '/'], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, env=dict(os.environ, **CFG.runner.SSH_PREFIX_ENV),
                           timeout=S.LEGACY_TIMEOUT_S)
        if p.returncode != 0:
            u.refuse(f'G22 snapshot copy: rsync rc {p.returncode}: {scripts.last_line(p.stdout.decode("utf-8", "replace"))}')
        u.add_output(DST, 'review_copy', source=plan.source)


def localize_stage(run, plan):
    DST = plan.U
    with run.unit(f'snapshot_{plan.run_name}_localize', stage='localize', tier='snapshot') as u:
        if not _S().SNAPSHOT_LOCALIZE:
            raise NoInput('serve.SNAPSHOT_LOCALIZE is False')
        rc, out = scripts.run_shell('snapshot_localize', [DST], unit=u,
                                    log=os.path.join(plan.out, '_records', 'serve', 'logs', f'localize_{plan.run_name}.log'))
        ok = rc == 0 and any(l.startswith('LOCALIZE-TIERS-OK') for l in out.splitlines())
        u.check('G21_LOCALIZE_TIERS_OK', ok, value=rc, detail=scripts.last_line(out))
        if not ok:
            u.refuse(f'G21 snapshot localization failed (rc {rc}): {scripts.last_line(out)} - serve stopped, servers untouched')
        for T in ('10s', '1min'):
            u.add_output(os.path.join(DST, f'source_catalogs_loc_{T}', 'localizations.csv'), 'csv_table', min_rows=1)


def gate_copy_stage(run, plan):
    DST = plan.U
    with run.unit(f'snapshot_{plan.run_name}_gate_copy', stage='gate-copy', tier='snapshot') as u:
        flow.excl_note(u)
        scan = exclusion_gate.scan_snapshot_run(DST)
        rc, out, refusals = exclusion_gate.run_inplace_gate(DST, unit=u)
        ok, detail = exclusion_gate.agree(scan, refusals, rc)
        u.check('G17_two_readers_agree', ok, detail=detail)
        if not ok:
            u.refuse(f'G17 serve-time exclusion gate on the copy: {detail}')
        if scan.hits or rc != 0:
            u.refuse(f'G17 serve-time exclusion gate REFUSED {plan.run_name} on the review copy: {scan.summary()} - serve '
                     f'stopped, servers untouched (see {DST}/serve_exclusion_refusals.csv)')
        u.add_output(os.path.join(DST, 'serve_exclusion_check.txt'), 'exclusion_check', run=DST)


def carry_stage(run, plan, args):
    spec = getattr(args, 'carry_from', None) or 'served'
    if spec == 'none':
        return
    DST = plan.U
    with run.unit(f'snapshot_{plan.run_name}_carry', stage='carry', tier='snapshot') as u:
        old = journals.resolve_carry_from(plan, spec)
        u.note(f'carry from the run served on {plan.all_pages[0].port_cfg}: {old}')
        if not os.path.exists(os.path.join(old, 'judgements', 'judgements.jsonl')):
            u.refuse(f'G18 carry preconditions: no journal at {old}/judgements/judgements.jsonl')
        cr = journals.snapshot_carry(old, DST, unit=u)
        u.note(cr)
        notfp, nmiss = journals.hold_check(old, DST)
        u.meta['carry'] = {'old': old, 'report': cr, 'missed': nmiss, 'not_fp': notfp}
        j = os.path.join(DST, 'judgements', 'judgements.jsonl')
        if notfp:
            u.refuse(f'G19 HOLD: {len(notfp)} judged source(s) with a verdict other than FP would disappear: '
                     f'{"; ".join(notfp[:10])} - servers stay on {os.path.basename(old)}; full list '
                     f'{DST}/judgements/carry_missed.csv')
        u.note(f'carry: {nmiss} missed verdict(s), all FP verdicts - serving; list {DST}/judgements/carry_missed.csv')
        u.add_output(j, 'judgement_journal')


def run(args, run, plan, stages):
    state = flow.new_state()
    if 'gate' in stages:
        gate_stage(run, plan)
    if 'copy' in stages:
        copy_stage(run, plan)
    if 'localize' in stages:
        localize_stage(run, plan)
    if 'gate-copy' in stages:
        gate_copy_stage(run, plan)
    if 'carry' in stages:
        carry_stage(run, plan, args)
    if 'test' in stages:
        flow.test_stage(run, plan, state, f'snapshot_{plan.run_name}_test', probe=False)
    elif 'start' in stages:
        exp, refused = flow.expected_from_record(run, f'snapshot_{plan.run_name}_test', 'start')
        state['expected'].update(exp)
        state['refused'].update(refused)
    if 'start' in stages:
        if 'gate-copy' not in stages:
            flow.require_done(run, f'snapshot_{plan.run_name}_gate_copy', 'start (the gate on the copy)')
        flow.start_stage(run, plan, state)
    if 'hub' in stages and not getattr(args, 'no_hub', False):
        flow.hub_stage(run, plan, state)
    if 'verify' in stages:
        flow.verify_stage(run, plan)


def describe(plan, stages, args):
    host, src = split_source(plan.source)
    print(f'  chain run (source): {plan.source or "(unset)"}' + ('' if host or (src and os.path.isdir(src)) else
                                                                   '   [ABSENT locally]'))
    print(f'  review copy (DST):  {plan.U}' + ('   [exists]' if plan.U and os.path.isdir(plan.U) else ''))
    if host is None and src and os.path.isdir(src):
        scan = exclusion_gate.scan_snapshot_run(src)
        print(f'  G17 on the source (core.exclusions, read now): {scan.summary()}' + (' - WOULD REFUSE' if scan.hits else ''))
    elif plan.U_src and os.path.isdir(plan.U_src):
        scan = exclusion_gate.scan_snapshot_run(plan.U_src)
        print(f'  G17 on the served copy {plan.U_src} (core.exclusions, read now): {scan.summary()}'
              + (' - WOULD REFUSE' if scan.hits else ''))
    print(f'  carry from: {getattr(args, "carry_from", None) or "served"} (run on {plan.all_pages[0].port_cfg}: '
          f'{journals.served_run_of_port(plan.all_pages[0].port_cfg)})')
