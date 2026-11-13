"""Stage implementations shared by the three tiers: test (G1, G17, App probe), start (the server check, the ONE registration of
the pages on the two-port server, then one unit per page: its build state and its served_page record, then the warm-up of the
inspection pages built before), hub, verify; the record chain between stages run in separate invocations; the dry-run plan
printer.

State passed between stages of one invocation (``state`` dict):
  state['refused'][page.key]  reason a page cannot be served (an earlier stage refused its queue or view)
  state['expected'][page.key] {'n', 'kind', ...} expected index sources (test stage)
  state['served'][page.key]   (record path, record) of the page registered and built in this invocation
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
    """The exclusion list read now: its fingerprint in the unit, and the file an INPUT of the unit, so a later stage of
    another invocation that requires this record (require_done, G28) refuses once data_decisions.csv changed (review
    2026-10-14 M1: a test stage passed, an exclusion added, a start in a separate invocation served the page)."""
    try:
        ex = exclusion_gate.fresh_exclusions()
        fp = ex.fingerprint()
        u.note(f'exclusions: {fp}')
        u.meta['exclusions'] = fp
        if getattr(ex, 'path', None) and os.path.exists(ex.path):
            u.add_input(ex.path)
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
def mchain_check(pg, U):
    """(refusal or None, notes) for the MULTIPLES CHAIN of page ``pg`` (serve.PAGES mchain_queue / mchain_catalog: the
    server's --mchain-queue / --mchain-catalog-dir, exopipe_ops 4c42f91, Gregg 2026-10-27), as production serves it:
      - the --mchain-queue FILE is absent: refused - the server reads it at start and dies (classify_transients.read_queue,
        :871), after its old server was stopped;
      - the --mchain-catalog-dir is absent: served, with a note - the server's loader skips it (load_sources :719-729);
      - whatever the tier's multiples page became in this serve, the page is served with the chain on disk: production
        starts the singles server and sets its hub entry before p3s3_serve_multiples.sh runs (p3s3_serve_stokes.sh:85, :91,
        :94; p3s3_pass.sh:222 ignores its exit status), and the two-port backend rebuilds every page from disk - no fork:
        the previous chain; the fork content test failed: the chain it refused (written at p3s3_serve_multiples.sh:83-86,
        before the test at :136-137); an empty queue: an empty chain."""
    mq, mc = pg.at('mchain_queue', U), pg.at('mchain_catalog', U)
    if mq and not os.path.isfile(mq):
        return (f'multiples chain: --mchain-queue {mq} is absent - page {pg.key} not served (its server dies reading it at '
                f'start, classify_transients.read_queue)'), []
    notes = []
    if mc and not os.path.isdir(mc):
        notes.append(f'multiples chain: --mchain-catalog-dir {mc} is absent - page {pg.key} served without the chain\'s '
                     f'own view (the server\'s loader skips it)')
    return None, notes


def test_stage(run, plan, state, unit_name, root=None, probe=True):
    """G1 (non-empty queue per page), G17 (serve-time exclusion gate), the content tests of each page's queue and view
    again, and the App probe (expected index sources, 0 unexplained drops) of every selected page."""
    U = root or plan.U
    with run.unit(unit_name, stage='test', tier=plan.tier, run=plan.run_name, parity=plan.parity) as u:
        excl_note(u)
        gated = []
        for pg in plan.pages:
            if pg.key in state['refused']:
                continue
            q, cat = pg.at('queue', U), pg.at('catalog', U)
            if not os.path.exists(q) or not common.has_j_line(q):
                state['refused'][pg.key] = f'G1 empty queue: {q} has no line starting J - not served'
                u.check('G1_nonempty_queue', True, value=0, detail=f'page {pg.key}: refused (empty queue)', path=q)
                u.note(state['refused'][pg.key])
                continue
            why, notes = mchain_check(pg, U)
            for n in notes:
                u.note(n)
            if why:
                state['refused'][pg.key] = why
                u.note(why)
                continue
            u.add_input(q)
            for d in (cat, pg.at('mchain_catalog', U)):
                for f in sorted(os.listdir(d)) if d and os.path.isdir(d) else []:
                    if f.endswith('.csv'):
                        u.add_input(os.path.join(d, f))
            if pg.at('mchain_queue', U):
                u.add_input(pg.at('mchain_queue', U))
            gated.append(pg)
            if plan.tier != 'snapshot':
                u.add_output(q, 'served_queue', tier=plan.tier)
                u.add_output(cat, 'served_view', tier=plan.tier)
        # G17: the serve-time exclusion gate (refuses for serve.EXCLUSION_GATE_TIERS, reports for the others)
        refuses = exclusion_gate.refuses(plan.tier, plan.family)
        gkey = exclusion_gate.gate_key(plan.tier, plan.family)
        if plan.tier == 'snapshot':
            scan = exclusion_gate.scan_snapshot_run(U)
            u.meta['exclusion_gate'] = {'hits': len(scan.hits), 'refuses': refuses, 'key': gkey,
                                        'per_key': {' '.join(k): n for k, n in scan.per_key().items()}}
            if scan.hits and refuses:
                u.refuse(f'G17 serve-time exclusion gate: {scan.summary()} - NOT served (re-chain those cells first)')
            u.note(f'G17 serve-time exclusion gate ({"refusing" if refuses else "report only"}): {scan.summary()}')
        else:
            # deep / sub-LST, PER PAGE (Gregg 2026-10-14, R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS): the rows
            # the page's own server shows (exclusion_gate.scan_page); a deep page holding one on an excluded night is not
            # served - its current server keeps running - and the other deep pages are. Sub-LST: any refused page refuses
            # the swap of the whole build (G11), so no sub-LST page is replaced (its servers share the run directory)
            pages = {}
            for pg in gated:
                scan = exclusion_gate.scan_server_page(pg, U, strict=refuses, tier=plan.tier)
                pages[pg.key] = {'hits': len(scan.hits), 'files': scan.n_files, 'rows': scan.n_rows,
                                 'per_key': {' '.join(k): n for k, n in scan.per_key().items()}}
                if scan.hits and refuses:
                    state['refused'][pg.key] = (f'G17 serve-time exclusion gate: {scan.summary()} - NOT served (its pass must '
                                                f'apply today\'s exclusions first; the page\'s current server keeps running)')
                    u.note(state['refused'][pg.key])
                else:
                    u.note(f'G17 serve-time exclusion gate ({"refusing" if refuses else "report only"}), page {pg.key}: '
                           f'{scan.summary()}')
            u.meta['exclusion_gate'] = {'hits': sum(v['hits'] for v in pages.values()), 'refuses': refuses, 'key': gkey,
                                        'pages': pages}
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


# ---- start: the ONE registration and each page's build ------------------------------------------------------------------
def start_checks(plan, pg, state, U):
    """(refusal or None, expected sources) of a page before the registration: an earlier stage's refusal, the expected
    sources of the test stage, and for deep G17 again (review 2026-10-14 M1: the exclusions or the page folders may have
    changed since the test stage; sub-LST decides before its swap, sublst_build.carry_swap_stage)."""
    if pg.key in state['refused']:
        return state['refused'][pg.key], None
    exp = state['expected'].get(pg.key)
    if exp is None:
        return f'page {pg.key}: no expected source count (the test stage did not pass for it)', None
    if (plan.tier == 'deep' and pg.program == 'classify' and pg.family == 'review'
            and exclusion_gate.refuses(plan.tier, plan.family)):
        scan = exclusion_gate.scan_server_page(pg, U)
        if scan.hits:
            return (f'G17 serve-time exclusion gate at start: {scan.summary()} - NOT registered (the page\'s registry entry '
                    f'and its build stay as they are)'), None
    return None, exp


def start_stage(run, plan, state, root=None, utc=None):
    """REGISTER the selected pages on the two-port server (production since 2026-10-22: no review page has a server of its
    own): each page's checks (start_checks), then ONE registration of every page that passed and each page's build
    (register_pages). A page refused here keeps its registry entry."""
    from exopipe.phase3.lib.serve import register
    utc = utc or common.utc_text()
    U = root or plan.U

    def prepare(u):
        items, refused = [], {}
        for pg in plan.pages:
            why, exp = start_checks(plan, pg, state, U)
            if why is not None:
                refused[pg.key] = why
                u.note(f'page {pg.key} not registered: {why}')
                continue
            inputs = [x for x in (pg.at('queue', U), pg.at('mchain_queue', U)) if x]
            items.append((pg, U, register.page_entry(plan, pg, utc, root), exp, inputs))
        return items, refused
    state['served'].update(register_pages(run, plan, prepare, f'{plan.tier}_{plan.run_name}_register'))


def register_pages(run, plan, prepare, unit_name, stage='start', inc_log=True):
    """THE ONE REGISTRATION of a serve (lib/serve/register.py) and each registered page's build.
    Unit ``unit_name`` (class page_registration): the two-port server first (register.check_server, G16 / G13: it answers,
    reports the build state, reads this registry, runs with the configured publish-hold option) - before ``prepare(u)`` ->
    (items, refused) does anything (a judgement carry): items [(page, run dir, entry, expected, inputs)] of the pages that
    passed their checks, refused {page key: reason}; then the registration checked by the server's loader (nothing written on a
    refusal), the publish marker of every registered run removed (register.lift_hold, whatever was refused: production removes
    it right before its one registration, deep_reregister_after_publish_20261031.py:79-81), and ONE register.write: one registry
    change. Then one unit per page (page.unit): a refused page refuses (or ends no-input) with its reason; a registered page
    waits on its build state (register.wait_built) and writes its served_page record. Then unit <unit_name>_warm: every
    inspection page the server had built before the registration is requested once, the registered pages' first (the warm-up
    of register_page.py:76-87 / warm_pages.py; record class csv_table). Returns {page key: (record path, record)}."""
    from exopipe.phase3.lib.serve import register
    d, ports = register.registry_dir(plan), register.listeners(plan)
    got = {'items': [], 'refused': {}, 'res': None, 'inspection': [], 'prepared': False}
    with run.unit(unit_name, stage=stage, tier=plan.tier, family=plan.family, registry=d, listeners=list(ports),
                  parity=plan.parity) as u:
        register.check_server(ports, d)                 # before anything changes (a carry, a marker, the registry)
        items, refused = prepare(u)
        got['items'], got['refused'], got['prepared'] = items, refused, True
        if not items:
            raise recmod.NoInput('no page passed its checks - nothing registered (the registry is unchanged)')
        prep = register.prepare([it[2] for it in items], d)
        st = register.check_server(ports, d)            # again, right before the write: and what it has built now
        got['inspection'] = register.built_inspection(st)
        holds = []
        for r in dict.fromkeys(it[1] for it in items):
            lifted = register.lift_hold(r)
            holds.append({'run': r, 'lifted': lifted})
            if lifted:
                u.note(f'publish marker of {r} removed right before the registration ({lifted})')
        res = register.write(prep)
        res['record'] = register.registration_record(plan.out, unit_name, res, holds)
        got['res'] = res
        u.add_output(res['record'], 'page_registration')
        u.note(f'ONE registration of {len(items)} page(s) in {res["registry"]}: {", ".join(e["id"] for e in res["entries"])} '
               f'(registry state {res["previous_state"]} -> {res["state"]})')
    why = None if u.status == 'done' else f'not registered: {u.name} ended {u.status} ({(u.reason or "")[:300]})'
    served = {}
    for pg in plan.pages:
        if not got['prepared']:                          # refused before the pages were checked (the server check)
            with run.unit(pg.unit, stage=stage, tier=plan.tier, page=pg.key, port=pg.port, parity=plan.parity) as pu:
                pu.refuse(why or 'not registered')
            continue
        if pg.key in got['refused']:
            w = str(got['refused'][pg.key])
            with run.unit(pg.unit, stage=stage, tier=plan.tier, page=pg.key, port=pg.port, parity=plan.parity) as pu:
                if w.startswith('NO-INPUT: '):          # e.g. no multi-integration source: the page is not served
                    raise recmod.NoInput(w[len('NO-INPUT: '):])     # (sublst_cumulative.sh:80, 'continue', not a refusal)
                pu.refuse(w)
            continue
        it = next((x for x in got['items'] if x[0].key == pg.key), None)
        if it is None:
            continue
        _, root, entry, exp, inputs = it
        with run.unit(pg.unit, stage=stage, tier=plan.tier, page=pg.key, port=pg.port, id=entry['id'],
                      parity=plan.parity) as pu:
            for x in inputs:
                pu.add_input(x)
            if why or got['res'] is None:
                pu.refuse(why or 'not registered')
            res = got['res']
            merged = res['entries'][[e['id'] for e in res['entries']].index(entry['id'])]
            build = register.wait_built(ports[0], merged, res['t'], d=d, log=pu.note)
            rec = servers.registered_record(plan, pg, merged, res, build, root, exp)
            path = servers.write_record(pg.record_path(), rec)
            pu.add_output(path, 'served_page')
            pu.note(f'page {pg.key} registered as {merged["id"]} and built from it ({build.get("build_s")} s; waited '
                    f'{build["waited_s"]} s: {" -> ".join(build["states"])}): {rec["url"]}')
            served[pg.key] = (path, rec)
            if inc_log:
                q = pg.at('queue', root)
                common.inc_log(root, f'registered {merged["id"]} (page {pg.key}) on the two-port server: '
                                     f'{len(common.queue_ids(q)) if q and os.path.exists(q) else 0} ids (exopipe3 serve)')
    if got['res'] is not None:
        warm_inspection(run, plan, got['res'], got['inspection'], f'{unit_name}_warm', stage)
    return served


def warm_inspection(run, plan, res, built, unit_name, stage):
    """The warm-up of the INSPECTION pages (register_page.py:76-87 -> warm_pages.py, which requested every page built before a
    registry change once after it, the registration's own pages first): each inspection page the server had built before the
    registration, requested once on the inspection listener, the registered pages' first, then the others; their answers in
    <out>/_records/serve/registrations/<unit>.csv (class csv_table). A registered page's inspection build was dropped with
    its entry and is built by this request; another page's answers at once and is rebuilt by the server."""
    from exopipe.phase3.lib.serve import register
    ours = [e['id'] for e in res['entries']]
    order = [p for p in ours if p in built] + [p for p in built if p not in ours]
    with run.unit(unit_name, stage=stage, tier=plan.tier, pages=order) as u:
        if not order:
            raise recmod.NoInput('no inspection page was built before the registration - nothing to warm')
        port = register.listeners(plan)[1]
        rows = []
        for pid in order:
            st, sec = register.warm(port, pid)
            rows.append({'page': pid, 'listener': port, 'status': st if st is not None else 'no answer', 'seconds': sec,
                         'registered_now': int(pid in ours)})
            u.note(f'{pid}:inspection requested once: {rows[-1]["status"]} in {sec} s')
        path = os.path.join(os.path.dirname(res['record']), os.path.basename(res['record'])[:-len('.json')] + '_warm.csv')
        import csv
        with open(path + '.part', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['page', 'listener', 'status', 'seconds', 'registered_now'])
            w.writeheader()
            w.writerows(rows)
        os.replace(path + '.part', path)
        u.add_output(path, 'csv_table', required_columns=('page', 'listener', 'status', 'seconds'), min_rows=1)


# ---- hub --------------------------------------------------------------------------------------------------------------
def hub_stage(run, plan, state, remove=(), utc=None):
    utc = utc or common.utc_text()
    with run.unit(f'{plan.tier}_{plan.run_name}_hub', stage='hub', tier=plan.tier, parity=plan.parity) as u:
        entries, tunnel = [], []
        for pg in plan.pages:
            if pg.key in state['served'] or _served_record_done(run, pg) or _hub_entry_unserved(run, plan, pg, state):
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
            u.note('parity: the hub registry and page are written under --out by lib/serve/hub (build_review_hub.py '
                   'hard-codes its directory; hub.render is its page, exopipe_ops 4c42f91)')
        u.note(hub.SERVED_NOTE)


def _served_record_done(run, pg):
    rec = recmod.load_record(run.out_root, 'serve', pg.unit)
    return bool(rec and rec.get('status') == 'done')


def _hub_entry_unserved(run, plan, pg, state):
    """True for a sub-LST page not served for NO-INPUT only (a multiples page with no multi-integration source): its hub
    entry is set all the same, as production's 10-minute two-port path sets all four entries after the registration, the
    multiples ones whether or not their page was registered (sublst_cumulative.sh:298-299; sublst_register_pages.py skips
    such a page). Registry only: the hub page shows no multiples button (exopipe_ops 4c42f91)."""
    if plan.tier != 'sublst' or pg.family != 'review':
        return False
    why = state['refused'].get(pg.key)
    if why is not None:
        return str(why).startswith('NO-INPUT: ')
    rec = recmod.load_record(run.out_root, 'serve', pg.unit)      # served in an earlier invocation of this serve
    return bool(rec and rec.get('status') == 'no-input')


# ---- verify -----------------------------------------------------------------------------------------------------------
def verify_stage(run, plan):
    """Re-test every selected page's served_page record now: live, same_process, sources_listed."""
    with run.unit(f'{plan.tier}_{plan.run_name}_verify', stage='verify', tier=plan.tier) as u:
        n = 0
        for pg in plan.pages:
            p = pg.record_path()
            if os.path.exists(p):
                u.add_output(p, 'served_page')
                n += 1
            else:
                u.note(f'no served_page record for {pg.key} ({pg.port})')
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


def describe_page(plan, pg):
    """What the plan's two-port registry holds for the page now (read only)."""
    from exopipe.phase3.lib.serve import register
    e = register.registered_entry(register.registry_dir(plan), f'p{pg.port}')
    if e is None:
        return 'not registered'
    a = e.get('argv') or []
    run = a[a.index('--run') + 1] if '--run' in a and a.index('--run') + 1 < len(a) else '?'
    return f'registered, --run {run}' + ('; greenlit (inspection listener)' if e.get('greenlit') else '')


def print_plan_header(plan, stages):
    from exopipe.phase3.lib.serve import register
    print(f'PLAN exopipe3 serve --tier {plan.tier}: run {plan.run_name} '
          f'({"PARITY into " + plan.out if plan.parity else "PRODUCTION"}); stages {", ".join(stages)}')
    if plan.U_src:
        print(f'  served run (read): {plan.U_src}' + ('' if os.path.isdir(plan.U_src) else '   [ABSENT]'))
        print(f'  writes under:      {plan.U}')
        hf = register.hold_file(plan.U_src)
        if os.path.lexists(hf):
            print(f'  publish hold:      {hf} stands (lifted right before the run\'s registration)')
    d, ports = register.registry_dir(plan), register.listeners(plan)
    st = register.server_status(ports[0], timeout=10)
    print(f'  pages registered in {register.files(d)[0]} (one registration); two-port server on {ports[0]} / {ports[1]}: '
          + ('not answering - nothing would be registered (serve never starts one)' if st is None else
             f'answering, reads {st.get("registry")}'))
    print(f'  hub:               {plan.hub_dir} (served by the two-port server on both listeners)')
    if plan.tier == 'sublst':
        print(f'  LSTs:              {plan.lsts_text or "(none found)"}')
    if plan.tier == 'snapshot' and plan.family == 'review':
        print(f'  source:            {plan.source or "(unset: give --source or set site CHAIN_HOST)"}   label: '
              f'{plan.snap_label or "(unset: --label)"}')
    print(f'  server env:        {json.dumps(plan.server_env_extra())}')
    for g, m in plan.gates:
        print(f'  WOULD REFUSE {g}: {m}')
