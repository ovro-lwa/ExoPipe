"""Stage implementations shared by the three tiers: test (G1, G17, App probe), start (one unit per page), twins, hub,
verify; the record chain between stages run in separate invocations; the dry-run plan printer.

State passed between stages of one invocation (``state`` dict):
  state['refused'][page.key]  reason a page cannot be served (an earlier stage refused its queue or view)
  state['expected'][page.key] {'n', 'kind', ...} expected index sources (test stage)
  state['served'][page.key]   (record path, record) of the page served in this invocation
"""
import json
import os
import sys

from exopipe.phase3.core import record as recmod
from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.serve import common, exclusion_gate, hub, servers


def _S():
    from exopipe.phase3.config import CFG
    return CFG.serve


def new_state():
    return {'refused': {}, 'expected': {}, 'served': {}}


def halts(u, what, need_done=False):
    """True when the unit ``u`` of an earlier stage of this invocation stops the later stages: it ended refused or error
    (with need_done: anything but done). A unit's `with` block records a Refusal and swallows it, so without this check
    a refused gate was followed by the next stage - a sub-LST build swapped in after its views were refused (G11;
    sublst_cumulative.sh:72-73 exits before the swap), deep pages restarted on the previous serve's views after G2 refused
    the new ones (p3s3_serve_stokes.sh:29 exits), a snapshot run copied after the exclusion gate refused it
    (snapshot_serve.sh:27 exit 5). The refused unit gives the run its exit code (2)."""
    if u is None:
        return False
    if u.status in ('refused', 'error') or (need_done and u.status != 'done'):
        print(f'serve: stage {what} ended {u.status} ({(u.reason or "")[:300]}) - the later stages of this invocation '
              f'are not run', file=sys.stderr)
        return True
    return False


def excl_note(u):
    try:
        fp = exclusion_gate.fresh_exclusions().fingerprint()
        u.note(f'exclusions: {fp}')
        u.meta['exclusions'] = fp
    except Refusal as e:
        raise Refusal(f'G17 serve-time exclusion gate: {e}')


# ---- record chain ---------------------------------------------------------------------------------------------------
def require_done(run, unit, what):
    """The record of ``unit`` (an earlier invocation) must be done, written from the same config, with every input
    unchanged (DESIGN 11.5); returns the record. Refuses (G28) otherwise."""
    rec = recmod.load_record(run.out_root, 'serve', unit)
    if rec is None:
        raise Refusal(f'G28 inputs current: {what} needs the record {unit} (run that stage first, or in this invocation)')
    if rec.get('status') != 'done':
        raise Refusal(f'G28 inputs current: record {unit} has status {rec.get("status")} ({rec.get("reason")})')
    for p, fp in (rec.get('inputs') or {}).items():
        if not recmod.same_fingerprint(fp, recmod.fingerprint(p)):
            raise Refusal(f'G28 inputs current: {p} changed since {unit} was recorded - run the {what} stages again')
    return rec


# ---- test -----------------------------------------------------------------------------------------------------------
def test_stage(run, plan, state, unit_name, root=None, views_for_gate=None, probe=True):
    """G1 (non-empty queue per page), G17 (serve-time exclusion gate), the content tests of each page's queue and view
    again, and the App probe (expected index sources, 0 unexplained drops) of every selected page."""
    U = root or plan.U
    with run.unit(unit_name, stage='test', tier=plan.tier, run=plan.run_name, parity=plan.parity) as u:
        excl_note(u)
        dirs, queues = [], []
        for pg in plan.pages:
            if pg.key in state['refused']:
                continue
            q, cat = pg.at('queue', U), pg.at('catalog', U)
            if not os.path.exists(q) or not common.has_j_line(q):
                state['refused'][pg.key] = f'G1 empty queue: {q} has no line starting J - not served'
                u.check('G1_nonempty_queue', True, value=0, detail=f'page {pg.key}: refused (empty queue)', path=q)
                u.note(state['refused'][pg.key])
                continue
            u.add_input(q)
            for f in sorted(os.listdir(cat)) if os.path.isdir(cat) else []:
                if f.endswith('.csv'):
                    u.add_input(os.path.join(cat, f))
            dirs.append(cat)
            queues.append(q)
            if plan.tier != 'snapshot':
                u.add_output(q, 'served_queue', tier=plan.tier)
                u.add_output(cat, 'served_view', tier=plan.tier)
        # G17: the serve-time exclusion gate (refuses for serve.EXCLUSION_GATE_TIERS, reports for the others)
        if plan.tier == 'snapshot':
            scan = exclusion_gate.scan_snapshot_run(U)
        else:
            scan = exclusion_gate.scan_tier_views(U, views_for_gate or dirs, queues)
        refuses = exclusion_gate.refuses(plan.tier, plan.family)
        u.meta['exclusion_gate'] = {'hits': len(scan.hits), 'per_key': {' '.join(k): n for k, n in scan.per_key().items()},
                                    'refuses': refuses, 'key': exclusion_gate.gate_key(plan.tier, plan.family)}
        if scan.hits and refuses:
            u.refuse(f'G17 serve-time exclusion gate: {scan.summary()} - NOT served (re-chain those cells first)')
        u.note(f'G17 serve-time exclusion gate ({"refusing" if refuses else "report only"}): {scan.summary()}')
        # the expected index sources of each page
        exp = {}
        for pg in plan.pages:
            if pg.key in state['refused']:
                continue
            if plan.tier == 'snapshot':
                exp[pg.key] = {'n': servers.queue_length(pg.at('queue', U)), 'kind': 'queue length'}
            elif probe:
                r = servers.app_probe(plan, pg, U, unit=u)
                exp[pg.key] = r
                if r.get('n_unexplained'):
                    state['refused'][pg.key] = (f'App probe: {r["n_unexplained"]} queue ids the server cannot resolve '
                                                f'(404), e.g. {r["unexplained"][:3]} - not served')
                    u.note(state['refused'][pg.key])
                u.check('app_probe_unexplained_drops', not r.get('n_unexplained'), value=r.get('n_unexplained'),
                        detail=f'page {pg.key}: {r["n"]} sources, {r.get("n_explained", 0)} explained drops')
        state['expected'].update(exp)
        u.meta['expected'] = exp
        u.meta['refused_pages'] = dict(state['refused'])
    return u


def expected_from_record(run, unit_name, what):
    rec = require_done(run, unit_name, what)
    return (rec.get('meta') or {}).get('expected') or {}, (rec.get('meta') or {}).get('refused_pages') or {}


# ---- start / twins ----------------------------------------------------------------------------------------------------
def start_stage(run, plan, state, root=None, utc=None):
    """One unit per selected page: replace its server and record it (class served_page)."""
    utc = utc or common.utc_text()
    for pg in plan.pages:
        with run.unit(pg.unit, stage='start', tier=plan.tier, page=pg.key, port=pg.port, parity=plan.parity) as u:
            if pg.key in state['refused']:
                why = state['refused'][pg.key]
                if why.startswith('NO-INPUT: '):          # e.g. no multi-integration source: the page is not served
                    raise recmod.NoInput(why[len('NO-INPUT: '):])     # (sublst_cumulative.sh:80, 'continue', not a refusal)
                u.refuse(why)
            exp = state['expected'].get(pg.key)
            if exp is None:
                u.refuse(f'page {pg.key}: no expected source count (the test stage did not pass for it)')
            notes = []
            q = pg.at('queue', root or plan.U)
            u.add_input(q)
            path, rec = servers.replace(plan, pg, utc, root=root, expected=exp, notes=notes)
            for n in notes:
                u.note(n)
            u.add_output(path, 'served_page')
            state['served'][pg.key] = (path, rec)
            common.inc_log(root or plan.U, f'serving {pg.key} on {pg.port}: {len(common.queue_ids(q))} ids (exopipe3 serve)')


def twins_stage(run, plan, state, root=None):
    for pg in plan.pages:
        if not pg.twin:
            continue
        with run.unit(f'{pg.unit}_twin', stage='twins', tier=plan.tier, page=pg.key, port=pg.twin) as u:
            if pg.key not in state['served']:
                rec = recmod.load_record(run.out_root, 'serve', pg.unit)
                if not rec or rec.get('status') != 'done':
                    u.refuse(f'twin of {pg.key}: its judging page was not served (G15)')
            notes = []
            judge_rec = state['served'].get(pg.key, (None, {}))[1]
            exp = {'n': judge_rec.get('expected_sources'), 'kind': judge_rec.get('expected_kind')} if judge_rec else \
                state['expected'].get(pg.key)
            path, rec = servers.start_twin(plan, pg, root=root, notes=notes, expected=exp)
            for n in notes:
                u.note(n)
            u.add_output(path, 'served_page')


# ---- hub --------------------------------------------------------------------------------------------------------------
def hub_stage(run, plan, state, remove=(), utc=None):
    utc = utc or common.utc_text()
    with run.unit(f'{plan.tier}_{plan.run_name}_hub', stage='hub', tier=plan.tier, parity=plan.parity) as u:
        entries, tunnel = [], []
        for pg in plan.pages:
            if pg.key in state['served'] or _served_record_done(run, pg):
                e = plan.hub_entry(pg, utc)
                if e is None:
                    tunnel.append(pg.port)
                else:
                    entries.append(e)
        if not entries and not tunnel:
            u.refuse('hub: no page of this run was served (nothing to register)')
        reg, index, planned = hub.update(plan, entries, remove=remove, tunnel=tunnel, log_path=plan.log_path, unit=u)
        u.meta['planned_sets'] = planned
        expected = [(s, t, dict(kv.split('=', 1) for kv in kvs)) for s, t, kvs in entries]
        u.add_output(index or reg, 'hub_page', expected=expected)
        if plan.parity:
            u.note('parity: the hub page is planned (registry.json written under --out; build_review_hub.py hard-codes '
                   'its directory, spec 12.5)')
        u.add_output(hub.inspection_hub(plan, log_path=plan.log_path, unit=u), 'hub_page')
        for n in hub.ensure_http_servers(plan):
            u.note(n)


def _served_record_done(run, pg):
    rec = recmod.load_record(run.out_root, 'serve', pg.unit)
    return bool(rec and rec.get('status') == 'done')


# ---- verify -----------------------------------------------------------------------------------------------------------
def verify_stage(run, plan):
    """Re-test every selected page's served_page record (and its twin's) now: live, same_process, sources_listed."""
    with run.unit(f'{plan.tier}_{plan.run_name}_verify', stage='verify', tier=plan.tier) as u:
        n = 0
        for pg in plan.pages:
            for port in [pg.port] + ([pg.twin] if pg.twin else []):
                p = pg.record_path(port)
                if os.path.exists(p):
                    u.add_output(p, 'served_page')
                    n += 1
                else:
                    u.note(f'no served_page record for {pg.key} on {port}')
        if not n:
            raise recmod.NoInput('verify: no served_page record of the selected pages under --out')


# ---- dry-run plan ---------------------------------------------------------------------------------------------------
def describe_port(port):
    from exopipe.phase3.lib.serve import procs
    pids = procs.listeners(port)
    if not pids:
        return 'free'
    out = []
    for pid in pids:
        info = procs.proc_info(pid) or {}
        out.append(f'pid {pid} {"review server" if procs.is_review_server(info) else (info.get("comm") or "?")} '
                   f'run={procs.argv_value(info.get("argv", []), "--run")}')
    return '; '.join(out)


def print_plan_header(plan, stages):
    print(f'PLAN exopipe3 serve --tier {plan.tier}: run {plan.run_name} '
          f'({"PARITY into " + plan.out if plan.parity else "PRODUCTION"}); stages {", ".join(stages)}')
    if plan.U_src:
        print(f'  served run (read): {plan.U_src}' + ('' if os.path.isdir(plan.U_src) else '   [ABSENT]'))
        print(f'  writes under:      {plan.U}')
    print(f'  hub:               {plan.hub_dir}   inspection hub: {plan.inspection_hub_dir}')
    if plan.tier == 'sublst':
        print(f'  LSTs:              {plan.lsts_text or "(none found)"}')
    if plan.tier == 'snapshot' and plan.family == 'review':
        print(f'  source:            {plan.source or "(unset: give --source or set site CHAIN_HOST)"}   label: '
              f'{plan.snap_label or "(unset: --label)"}')
    print(f'  server env:        {json.dumps(plan.server_env_extra())}')
    for g, m in plan.gates:
        print(f'  WOULD REFUSE {g}: {m}')
