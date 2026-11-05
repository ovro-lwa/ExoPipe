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

Before every stage (snapshot_serve.sh, exopipe_ops aa100bf): G29 THE SERVE HOLD - while serve.SNAPSHOT_SERVE_HOLD exists (any
file; its text says why) unit snapshot_<name>_hold refuses and no snapshot run is served (nothing gated, copied or served; no
--waive lifts it); without the file there is no hold unit.
Then (snapshot_serve.sh): G27 GREGG'S GO ON A COMBINED VIEW (unit snapshot_<name>_go; refused = nothing gated, copied or
served):
  card2 / re-judging (exopipe_ops 35aad1a)  a run whose combined_view_summary.json holds "card2" or "rejudge" (built with Card 2
           per frame / re-judging; the source, and an existing review copy) is served only when serve.SNAPSHOT_CARD2_REJUDGE_APPROVED
           is non-empty and names it (Gregg's go on the counts, R-GREGG-20261104-DEEP-OMISSION-SHORT-TIERS-ANSWERS);
  restore-only (exopipe_ops 8ff3fc8)  every tag of a combined view's 'batches' resolves to exactly one member run beside it
           (serve.SNAPSHOT_VIEW_MEMBER_GLOB) with a readable chain_summary.json; a member that ran the restore-only 3-D gate
           (gate3d_restore_only) needs its test against the run it replaced passed (serve.SNAPSHOT_RESTORE_DIFF status PASS), and
           a view holding any needs serve.SNAPSHOT_RESTORE_ONLY_APPROVED naming it (Gregg's go on the wave totals,
           R-GREGG-20261104-EVERYTHING-AS-RECOMMENDED). A combined view from a remote source refuses (production serves a view from
           its local path on the review host, its member runs beside it).
A name counts only as a whole token of the go file (production's grep: a substring). A run with neither (a batch run) gets no go
unit (production checks nothing there).
"""
import glob
import json
import os
import re
import shlex
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


GO_KEYS = re.compile(r'"(card2|rejudge)"')   # snapshot_serve.sh (35aad1a): grep -qE '"(card2|rejudge)"' combined_view_summary.json
GO_SUMMARY = 'combined_view_summary.json'
_ABSENT = '__EXOPIPE3_NO_SUMMARY__'


def _summary_text(host, path):
    """The text of <path>/combined_view_summary.json, None when the run has none (a remote source over ssh). OSError when it
    cannot be read: G27 then refuses (fail closed)."""
    f = os.path.join(path, GO_SUMMARY)
    if host is None:
        if not os.path.lexists(f):
            return None
        with open(f, errors='replace') as fh:
            return fh.read()
    from exopipe.phase3.config import CFG
    q = shlex.quote(f)
    p = subprocess.run(['ssh', '-n', '-o', 'BatchMode=yes', host, f'if [ -e {q} ]; then cat {q}; else echo {_ABSENT}; fi'],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ, **CFG.runner.SSH_PREFIX_ENV),
                       timeout=600)
    out = p.stdout.decode('utf-8', 'replace')
    if p.returncode != 0:
        raise OSError(f'ssh {host} rc {p.returncode}: {scripts.last_line(out)}')
    return None if out.strip() == _ABSENT else out


def names_in(text, name):
    """True when ``name`` stands in ``text`` as a whole token (not inside a longer run name)."""
    return re.search(r'(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])' % re.escape(name), text) is not None


def card2_rejudge_go(plan):
    """G27 (module docstring): None when no run of this serve holds the keys; else (refusal reason or '', detail) - '' = the go
    names every one of its names (the served name and the source's directory name)."""
    host, src = split_source(plan.source)
    runs = [(host, src)] if src else []
    if plan.U and os.path.isdir(plan.U) and not (host is None and src and os.path.abspath(plan.U) == os.path.abspath(src)):
        runs.append((None, plan.U))            # an existing review copy (a serve of its later stages only)
    ap = _S().SNAPSHOT_CARD2_REJUDGE_APPROVED
    hit = None
    for h, pth in runs:
        try:
            t = _summary_text(h, pth)
        except OSError as e:
            return (f'G27 REFUSING: the combined-view summary of {(h + ":") if h else ""}{pth} cannot be read ({e}) - nothing '
                    f'served, servers untouched', '')
        if t is not None and GO_KEYS.search(t):
            hit = (h, pth)
            break
    if hit is None:
        return None
    names = sorted({plan.run_name} | {os.path.basename(p.rstrip('/')) for _h, p in runs})
    text = _go_text(ap)
    missing = [n for n in names if not names_in(text, n)]
    where = f'{(hit[0] + ":") if hit[0] else ""}{hit[1]}/{GO_SUMMARY}'
    if missing:
        return (f"G27 REFUSING: {missing[0]} applies Card 2 and re-judging ({where}) - Gregg's go on the counts is required "
                f"first ({ap} naming the view{'' if text else ': absent or empty'}) - nothing served, servers untouched", where)
    return '', where


def _go_text(path):
    try:
        return open(path, errors='replace').read() if os.path.getsize(path) > 0 else ''
    except OSError:
        return ''


def restore_only_go(plan):
    """G27, the restore-only wave's go (module docstring): None when no run of this serve is a combined view; else (refusal
    reason or '', detail, members used, restore-only members)."""
    S = _S()
    host, src = split_source(plan.source)
    if host is not None:
        try:
            t = _summary_text(host, src)
        except OSError as e:
            return (f'G27 REFUSING: the combined-view summary of {host}:{src} cannot be read ({e}) - nothing served, servers '
                    f'untouched', '', [], 0)
        if t is not None:
            return (f'G27 REFUSING: {plan.run_name} is a combined view on {host}: serve a view from its local path on the review '
                    f'host (its member runs beside it, snapshot_serve.sh) - nothing served, servers untouched',
                    f'{host}:{src}/{GO_SUMMARY}', [], 0)
    runs = [src] if (host is None and src) else []
    if plan.U and os.path.isdir(plan.U) and not (runs and os.path.abspath(plan.U) == os.path.abspath(runs[0])):
        runs.append(plan.U)
    for pth in runs:
        f = os.path.join(pth, GO_SUMMARY)
        if not os.path.isfile(f):                      # production: [ -f "$1/combined_view_summary.json" ]
            continue
        name = os.path.basename(pth.rstrip('/'))
        try:
            batches = json.load(open(f))['batches']
        except Exception as e:                         # production: the check heredoc fails -> refused
            return (f'G27 REFUSING: the restore-only check of {name} could not run ({type(e).__name__}: {e}) - nothing served, '
                    f'servers untouched', f, [], 0)
        R = os.path.dirname(os.path.abspath(pth))
        n, bad, used = 0, [], []
        for x in batches:
            c = sorted(glob.glob(os.path.join(R, S.SNAPSHOT_VIEW_MEMBER_GLOB.format(tag=x))))
            if len(c) != 1:
                bad.append('%s: %d run directories' % (x, len(c)))
                continue
            try:
                ro = 'gate3d_restore_only' in json.load(open(os.path.join(c[0], 'chain_summary.json')))
            except Exception as e:
                bad.append('%s: chain summary unreadable (%s)' % (x, e))
                continue
            used.append(c[0])
            if ro:
                n += 1
                p = os.path.join(c[0], S.SNAPSHOT_RESTORE_DIFF)
                try:
                    st = json.load(open(p)).get('status') if os.path.exists(p) else f'no {S.SNAPSHOT_RESTORE_DIFF}'
                except Exception as e:
                    st = f'{S.SNAPSHOT_RESTORE_DIFF} unreadable ({e})'
                if st != 'PASS':
                    bad.append('%s: %s' % (x, str(st)[:120]))
        if bad:
            return (f'G27 REFUSING: {name}: {"; ".join(bad[:5])} - nothing served, servers untouched', f, used, n)
        if n:
            ap = S.SNAPSHOT_RESTORE_ONLY_APPROVED
            text = _go_text(ap)
            missing = [m for m in sorted({plan.run_name, name}) if not names_in(text, m)]
            if missing:
                return (f"G27 REFUSING: {missing[0]} holds {n} restore-only re-chain runs - Gregg's go on the wave totals is "
                        f"required first ({ap} naming the view{'' if text else ': absent or empty'}) - nothing served, servers "
                        f"untouched", f, used, n)
        return ('', f'{f}: {len(batches)} member runs, {n} restore-only' + (' (every test PASS; the go names the view)' if n
                                                                              else ''), used, n)
    return None


def serve_hold():
    """G29 (module docstring): None when serve.SNAPSHOT_SERVE_HOLD does not exist, else (path, its first 400 bytes as text) -
    production's `[ -e $HOLDF ]` and `head -c 400 $HOLDF`."""
    p = _S().SNAPSHOT_SERVE_HOLD
    if not (p and os.path.exists(p)):
        return None
    try:
        with open(p, 'rb') as f:
            why = f.read(400).decode('utf-8', 'replace')
    except OSError as e:
        why = f'(unreadable: {e})'
    return p, why


def hold_unit(run, plan):
    """G29 as a unit, always refused, when the serve hold exists; None (no unit) otherwise."""
    h = serve_hold()
    if h is None:
        return None
    p, why = h
    with run.unit(f'snapshot_{plan.run_name}_hold', stage='hold', tier='snapshot', source=plan.source) as u:
        u.add_input(p)
        u.refuse(f'G29 SERVE HELD {plan.run_name}: {why.strip() or "(empty)"} ({p}) - nothing served, servers untouched')
    return u


def go_unit(run, plan):
    """G27 as a unit when the serve's run holds the Card 2 / re-judging keys or is a combined view (refused or done), None
    otherwise (no unit)."""
    S = _S()
    c2 = card2_rejudge_go(plan)
    ro = restore_only_go(plan)
    if c2 is None and ro is None:
        return None
    lines, names = [], sorted({plan.run_name} | ({os.path.basename(split_source(plan.source)[1].rstrip('/'))}
                                                  if plan.source else set()))
    with run.unit(f'snapshot_{plan.run_name}_go', stage='go', tier='snapshot', source=plan.source) as u:
        if c2 is not None:
            why, where = c2
            u.check('G27_card2_rejudge_go', not why, detail=why or f'{where}: the go names {plan.run_name} '
                                                                       f'({S.SNAPSHOT_CARD2_REJUDGE_APPROVED})')
            if why:
                u.refuse(why)
            u.note(f'G27: {where} holds Card 2 / re-judging; Gregg\'s go names the view ({S.SNAPSHOT_CARD2_REJUDGE_APPROVED})')
            u.add_input(S.SNAPSHOT_CARD2_REJUDGE_APPROVED)
            lines.append(f'card2/rejudge {where}')
            lines += [f'go {S.SNAPSHOT_CARD2_REJUDGE_APPROVED} names {n}' for n in names]
        if ro is not None:
            why, detail, used, n_ro = ro
            u.check('G27_restore_only_go', not why, detail=why or detail)
            if why:
                u.refuse(why)
            u.note(f'G27: {detail}')
            for m in used:
                for fn in ('chain_summary.json', S.SNAPSHOT_RESTORE_DIFF):
                    if os.path.exists(os.path.join(m, fn)):
                        u.add_input(os.path.join(m, fn))
            lines.append(f'restore-only {detail}')
            if n_ro:
                u.add_input(S.SNAPSHOT_RESTORE_ONLY_APPROVED)
                lines += [f'go {S.SNAPSHOT_RESTORE_ONLY_APPROVED} names {n}' for n in names]
        p = os.path.join(plan.out, '_records', 'serve', 'gates', f'{plan.run_name}_go.txt')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w') as f:
            f.write(f'SERVE-GO PASS {plan.run_name} (G27)\n' + ''.join(x + '\n' for x in lines))
        u.add_output(p, 'serve_go')
    return u


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
    return u


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
    return u


def localize_stage(run, plan, env=None):
    """snapshot_localize_tiers.sh on plan.U (G21). ``env``: the script's extra environment (the combined view's
    serve.SNAPVIEW_LOC_ENV, comb_build_20261102.sh:82; none for a batch serve, snapshot_serve.sh:46), plus PYTHONHASHSEED =
    serve.SNAPSHOT_FIGURE_HASHSEED: the figure heredoc's pool inherits its interpreter's string-hash seed, which decides the
    stacking of overlapping markers in the astrometry figure (V_combview review F10: 64 of 4,130 figure v2 PNGs differed between
    two runs on byte-identical tables) - fixed, the figures are a function of their inputs."""
    DST = plan.U
    S = _S()
    env = dict(env or {})
    if S.SNAPSHOT_FIGURE_HASHSEED not in (None, ''):
        env['PYTHONHASHSEED'] = str(S.SNAPSHOT_FIGURE_HASHSEED)
    with run.unit(f'snapshot_{plan.run_name}_localize', stage='localize', tier='snapshot') as u:
        if not S.SNAPSHOT_LOCALIZE:
            raise NoInput('serve.SNAPSHOT_LOCALIZE is False')
        u.note(f'localization environment: {env}')
        logd = os.path.join(plan.out, '_records', 'serve', 'logs')
        os.makedirs(logd, exist_ok=True)
        rc, out = scripts.run_shell('snapshot_localize', [DST], env=env, unit=u,
                                    log=os.path.join(logd, f'localize_{plan.run_name}.log'))
        ok = rc == 0 and any(l.startswith('LOCALIZE-TIERS-OK') for l in out.splitlines())
        u.check('G21_LOCALIZE_TIERS_OK', ok, value=rc, detail=scripts.last_line(out))
        if not ok:
            u.refuse(f'G21 snapshot localization failed (rc {rc}): {scripts.last_line(out)} - serve stopped, servers untouched')
        for T in ('10s', '1min'):
            u.add_output(os.path.join(DST, f'source_catalogs_loc_{T}', 'localizations.csv'), 'csv_table', min_rows=1)
    return u


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
    return u


def carry_stage(run, plan, args):
    spec = getattr(args, 'carry_from', None) or 'served'
    if spec == 'none':
        return None
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
    return u


def run(args, run, plan, stages):
    state = flow.new_state()
    # a refused stage stops the later ones (flow.halts): snapshot_serve.sh exits at a refused gate (:27 exit 5), localization
    # (:46 exit 4), gate on the copy (:48 exit 5), carry (:38 exit 3; the batch driver's HOLD, snapshot_rechain_batch.sh:406)
    # - nothing copied, no server replaced. A recorded no-input (the parity gate on a remote source, localization off) does
    # not stop.
    # G29 then G27 before every stage (snapshot_serve.sh checks both before the gate, the copy or any server; exit 7 there)
    if flow.halts(hold_unit(run, plan), 'hold', need_done=True):
        return
    go = go_unit(run, plan)
    if go is not None and flow.halts(go, 'go', need_done=True):
        return
    if 'gate' in stages and flow.halts(gate_stage(run, plan), 'gate'):
        return
    if 'copy' in stages and flow.halts(copy_stage(run, plan), 'copy', need_done=True):
        return
    if 'localize' in stages and flow.halts(localize_stage(run, plan), 'localize'):
        return
    if 'gate-copy' in stages and flow.halts(gate_copy_stage(run, plan), 'gate-copy', need_done=True):
        return
    if 'carry' in stages and flow.halts(carry_stage(run, plan, args), 'carry'):
        return
    if 'test' in stages:
        if flow.halts(flow.test_stage(run, plan, state, f'snapshot_{plan.run_name}_test', probe=False), 'test'):
            return
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
    h = serve_hold()
    print('  G29 serve hold: ' + (f'HELD - WOULD REFUSE ({h[0]}: {h[1].strip()[:200] or "(empty)"})' if h else
                                  f'none ({_S().SNAPSHOT_SERVE_HOLD} absent)'))
    go = card2_rejudge_go(plan)
    if go is not None:
        print(f'  G27 Card 2 / re-judging go: {go[1] or plan.source} - ' + (f'WOULD REFUSE ({go[0]})' if go[0] else
                                                                           'the go names the view'))
    go = restore_only_go(plan)
    if go is not None:
        print(f'  G27 restore-only go: ' + (f'WOULD REFUSE ({go[0]})' if go[0] else go[1]))
    if host is None and src and os.path.isdir(src):
        scan = exclusion_gate.scan_snapshot_run(src)
        print(f'  G17 on the source (core.exclusions, read now): {scan.summary()}' + (' - WOULD REFUSE' if scan.hits else ''))
    elif plan.U_src and os.path.isdir(plan.U_src):
        scan = exclusion_gate.scan_snapshot_run(plan.U_src)
        print(f'  G17 on the served copy {plan.U_src} (core.exclusions, read now): {scan.summary()}'
              + (' - WOULD REFUSE' if scan.hits else ''))
    print(f'  carry from: {getattr(args, "carry_from", None) or "served"} (run on {plan.all_pages[0].port_cfg}: '
          f'{journals.served_run_of_port(plan.all_pages[0].port_cfg)})')
