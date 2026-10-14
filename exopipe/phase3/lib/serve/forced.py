"""serve --tier snapshot --family forced: the snapshot forced-photometry pages 8786 / 8785 / 8775 / 8776, each serving its
own run Run_SNAPFORCED_<S><T>_<TAG> (serve owns these ports: refactor-chat decision 2026-10-13). Port of
snapshot_forced_serve_20261007.sh, modes 'switch' (default) and 'refresh-siblings' (--forced-refresh-siblings).

switch, one unit per page (snapshot_forced_serve_20261007.sh:60-101):
  1. the run ended FT-READY (last line of <OPS_ROOT>/snapshot_forced_pages_run_<run>.log), else refused - old page kept;
  2. it holds its queue, catalogue view, localization offsets and association table, else refused;
  3. the AD Leo check (test_snapshot_forced_reserve.py adleo --absent-ok <run>, site PYTHON 'py36'; Gregg 2026-10-08), else
     refused;
  4. the serve-time exclusion gate on the run: report only (gate key 'snapshot-forced' is not in EXCLUSION_GATE_TIERS;
     production has no gate on these pages);
  5. the port's old server: anything but a review server refuses; the same run already served = nothing to do (no-input);
     stopped by exact pid (FORCED_STOP_TIMEOUT_S);
  6. Gregg's judgements carried (snapshot_carry_judgements.py OLD NEW, after the old server exited) when the old journal is
     not empty; a refused carry restarts the old server with its own argv / cwd / environment and refuses the page;
  7. the new server with --sibling links to what each other forced port serves NOW; HTTP 200 within the snapshot readiness
     time, else the old server is restarted and the page refused;
  8. record (class served_page; expected sources = the server's own App on the files); the hub button
     (<S> <10-sec|1-min>-forced: port, run, generation, served, live, note=) in the hub unit.
refresh-siblings: every forced page of a <TAG> run whose --sibling links differ from what the other ports serve now is
  restarted (exact pid) with the current links; judgements stay (journal on disk).
Parity: each run is byte-copied (minus judgements/ and server logs) to <out>/snapshot/<run> and served from there on the
offset ports; the carry source is the production server on the PRODUCTION port.
"""
import os
import subprocess

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.serve import common, exclusion_gate, flow, journals, procs, scripts, servers

STAGES = ('switch', 'hub', 'verify')


def _CFG():
    from exopipe.phase3.config import CFG
    return CFG


def ready(name):
    S = _CFG().serve
    p = os.path.join(_CFG().paths.OPS_ROOT, S.FORCED_READY_LOG.format(name=name))
    try:
        with open(p) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
    except OSError:
        return False, f'no log {p}'
    last = lines[-1] if lines else ''
    return S.FORCED_READY_MARK in last, last[29:200] if len(last) > 29 else last


def missing_files(pg, D):
    return [f for f in (x.format(S=pg.stokes) for x in _CFG().serve.FORCED_REQUIRED) if not os.path.exists(os.path.join(D, f))]


def _old_server(pg):
    """(info, environ) of the review server on the page's port, or (None, None). Refuses a non-review listener."""
    for pid in procs.listeners(pg.port):
        info = procs.proc_info(pid)
        if not procs.is_review_server(info):
            raise Refusal(f'G13: port {pg.port} is held by pid {pid} ({(info or {}).get("comm")}), not a review server - '
                          f'old page kept')
        return info, procs.environ(pid)
    return None, None


def _restore(old, env, port, notes):
    run = procs.argv_value(old['argv'], '--run') or '/tmp'
    log = os.path.join(run, _CFG().serve.FORCED_RESTORE_LOG.format(port=port))
    p = procs.start(old['argv'], old['cwd'], env, log)
    notes.append(f'old server restarted on {port}: pid {p.pid} (log {log})')


def _parity_copy(pg):
    os.makedirs(os.path.dirname(pg.root), exist_ok=True)
    p = subprocess.run(['rsync', '-a', '--exclude', 'judgements', '--exclude', 'classify_*.log',
                        pg.root_src.rstrip('/') + '/', pg.root + '/'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise Refusal(f'parity copy of {pg.root_src} failed: {p.stdout.decode()[-300:]}')


def switch_page(run, plan, pg, state, utc):
    S = _CFG().serve
    with run.unit(pg.unit, stage='switch', tier='snapshot', family='forced', page=pg.key, port=pg.port) as u:
        flow.excl_note(u)
        ok, last = ready(pg.run_name)
        u.check('FT_READY', ok, detail=last)
        if not ok:
            u.refuse(f'{pg.run_name} has not ended {S.FORCED_READY_MARK} ({last}) - old page kept')
        miss = missing_files(pg, pg.root_src)
        if miss:
            u.refuse(f'{pg.root_src} lacks {", ".join(miss)} (queue, catalogue, localization offsets or association table) '
                     f'- old page kept')
        if plan.parity:
            _parity_copy(pg)
        D = pg.root
        res = scripts.run('forced_adleo', list(S.FORCED_ADLEO_CHECK[1]) + [D], python='py36', unit=u, refusal_codes=(1, 2))
        u.check('AD_Leo_check', res.ok, detail=scripts.last_line(res.tail))
        if not res.ok:
            u.refuse(f'AD Leo check ({scripts.last_line(res.tail)}) - old page kept')
        scan = exclusion_gate.scan_snapshot_run(D)
        u.note(f'G17 serve-time exclusion gate (report only for the forced pages): {scan.summary()}')
        if scan.hits and exclusion_gate.refuses('snapshot', 'forced'):
            u.refuse(f'G17 serve-time exclusion gate: {scan.summary()}')
        old, oenv = _old_server(pg)
        notes = []
        if old is not None:
            orun = procs.argv_value(old['argv'], '--run')
            if os.path.realpath(orun or '') == os.path.realpath(D):
                raise NoInput(f'{pg.port} already serves {pg.run_name} - nothing to do')
            if plan.parity and not servers._under(orun, plan.out):
                u.refuse(f'G23 parity guard: port {pg.port} is held by {orun}, not a run of {plan.out}')
        # the expected index sources (the server's own App on the new run's files; no journal read) BEFORE the old server is
        # stopped: a probe that fails refuses with the old page still served (production restores the old server on every
        # failure after its stop, snapshot_forced_serve_20261007.sh:78-90)
        exp = servers.app_probe(plan, pg, D, unit=u)
        if old is not None:
            notes.append(procs.stop_exact(old, timeout=S.FORCED_STOP_TIMEOUT_S, poll=1.0))
            oj = os.path.join(orun, 'judgements', 'judgements.jsonl')
            if journals.journal_nonempty(oj):
                try:
                    cr = journals.snapshot_carry(orun, D, unit=u)
                    notes.append(f'judgements carried on {pg.port}: {cr}')
                except Refusal as e:
                    _restore(old, oenv, pg.port, notes)
                    for n in notes:
                        u.note(n)
                    u.refuse(f'judgement carry {orun} -> {D} failed ({e}); the old server is restarted on {pg.port}')
            else:
                notes.append(f'judgements on {pg.port}: none in {orun} (no journal) - nothing to carry')
        p, argv, cwd, log = servers.start_page(plan, pg, utc, D)
        okr, detail = procs.wait_ready(pg.port, p, S.READY_TIMEOUT_S['snapshot'], S.READY_POLL_S, log)
        if not okr:
            if old is not None:
                _restore(old, oenv, pg.port, notes)
            for n in notes:
                u.note(n)
            u.refuse(f'G16 the new server did not answer 200 ({detail}; log {log})')
        notes.append(f'FORCED-RESERVED-PAGE {pg.port} {pg.stokes}/{pg.kind} {pg.run_name} siblings '
                     f'{argv.count("--sibling")} ({detail})')
        for n in notes:
            u.note(n)
        rec = servers._record(plan, pg, p.pid, argv, cwd, plan.server_env_extra(), D, exp, log)
        path = servers.write_record(pg.record_path(), rec)
        u.add_output(path, 'served_page')
        j = os.path.join(D, 'judgements', 'judgements.jsonl')
        if os.path.exists(j):
            u.add_output(j, 'judgement_journal')
        state['served'][pg.key] = (path, rec)


def refresh_siblings(run, plan, state, utc):
    tag = plan.forced_tag
    for pg in plan.pages:
        with run.unit(f'{pg.unit}_siblings', stage='refresh-siblings', tier='snapshot', family='forced', port=pg.port) as u:
            old, oenv = _old_server(pg)
            if old is None:
                raise NoInput(f'{pg.port}: no review server')
            D = procs.argv_value(old['argv'], '--run') or ''
            if tag and not D.endswith(f'_{tag}'):
                raise NoInput(f'{pg.port} serves {D}, not a {tag} run')
            have = sorted(old['argv'][i + 1] for i, a in enumerate(old['argv']) if a == '--sibling')
            want_args = plan.forced_siblings(pg)
            want = sorted(want_args[1::2])
            if have == want:
                raise NoInput(f'siblings of {pg.port} current')
            if plan.parity and not servers._under(D, plan.out):
                u.refuse(f'G23 parity guard: port {pg.port} is held by {D}, not a run of {plan.out}')
            exp = servers.app_probe(plan, pg, D, unit=u)       # before the stop: a failed probe keeps the page served
            u.note(procs.stop_exact(old, timeout=_CFG().serve.FORCED_STOP_TIMEOUT_S, poll=1.0))
            p, argv, cwd, log = servers.start_page(plan, pg, utc, D)
            okr, detail = procs.wait_ready(pg.port, p, _CFG().serve.READY_TIMEOUT_S['snapshot'], 5.0, log)
            if not okr:
                u.refuse(f'refresh {pg.port}: did not answer 200 ({detail})')
            u.note(f'FORCED-SIBLINGS-REFRESHED {pg.port} ({D}): {want}')
            rec = servers._record(plan, pg, p.pid, argv, cwd, plan.server_env_extra(), D, exp, log)
            u.add_output(servers.write_record(pg.record_path(), rec), 'served_page')
            state['served'][pg.key] = (pg.record_path(), rec)


def run(args, run, plan, stages):
    state = flow.new_state()
    utc = common.utc_text()
    if getattr(args, 'forced_refresh_siblings', False):
        refresh_siblings(run, plan, state, utc)
    elif 'switch' in stages:
        for pg in plan.pages:
            switch_page(run, plan, pg, state, utc)
    if 'hub' in stages and not getattr(args, 'no_hub', False) and state['served']:
        flow.hub_stage(run, plan, state, utc=utc)
    if 'verify' in stages:
        flow.verify_stage(run, plan)


def describe(plan, stages, args):
    for pg in plan.pages:
        ok, last = ready(pg.run_name)
        miss = missing_files(pg, pg.root_src) if os.path.isdir(pg.root_src) else ['(run absent)']
        print(f'  {pg.key}: run {pg.root_src}  FT-READY {"yes" if ok else "NO (" + last + ")"}  '
              f'files {"ok" if not miss else "MISSING " + ", ".join(miss)}')
