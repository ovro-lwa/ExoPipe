"""serve --tier snapshot --family forced: the snapshot forced-photometry pages p8786 / p8785 / p8775 / p8776, each serving its
own run Run_SNAPFORCED_<S><T>_<TAG>, REGISTERED on the two-port server (serve owns these pages: refactor-chat decision
2026-10-13). Port of snapshot_forced_gen5_serve_20261026.sh - the 2026-10-07 serve step (snapshot_forced_serve_20261007.sh) on
the two-port server, whose per-page servers were retired 2026-10-22 - and of the 10-07 step's 'refresh-siblings' mode.

switch: unit snapshot_forced_<TAG>_register (flow.register_pages): the two-port server is checked first (nothing is carried
  for a registration that cannot happen), then per requested page (gen5:31-45; a page that fails a check keeps the run its
  entry serves):
  1. the run is ready (ready(): an exopipe3 forced family-C run by its verify record - done, servable, for this review copy;
     a production page run by the last line FT-READY of <OPS_ROOT>/snapshot_forced_pages_run_<run>.log);
  2. it holds its queue, catalogue view, localization offsets and association table;
  3. the AD Leo check (test_snapshot_forced_reserve.py adleo --absent-ok <run>, site PYTHON 'py36'; Gregg 2026-10-08 "don't
     label AD Leo unless you have a metric defined and measured");
  4. the serve-time exclusion gate on the run: refuses the page under the ruled value (gate key 'snapshot-forced' in
     EXCLUSION_GATE_TIERS: Gregg 2026-10-14 R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS); production has no
     gate on these pages (the parity file: report only);
  5. THE PER-PANEL QA BLOCK (page_context): scintillation_index.py RUN --tier snapshot, then page_context_snapshot.py RUN
     --served-from VIEW in place on the run the page will serve, VIEW the transient-search view the production registry's
     snapshot pages serve (paired_view, serve.FORCED_PAGE_CONTEXT_VIEW_IDS; production ran both by hand); its files present and
     content tests snapshot_forced_page_context passed, else refused (production's page code falls back silently when
     frame_counts.json is absent, review_filters.py:262-271);
  6. the expected index sources (the server's own App on the run's files: servers.app_probe);
  7. Gregg's judgements carried (snapshot_carry_judgements.py OLD NEW) from the run the page's entry serves now (gen5 cur_run;
     a parity serve: the PRODUCTION registry's entry, read only) when its journal is not empty and the new run has none (a
     journal already in the new run: carry skipped, gen5:40); a page already serving the run is registered again, nothing
     carried (gen5:38).
  Then ALL pages that passed are registered in ONE registration (one registry change), each with --sibling links to what every
  other forced page serves AFTER the switch - its new run where it passed, else the run its entry serves now (gen5:47-56;
  plan.forced_siblings) - and one unit per page waits on its build state (gen5:66-69 waited up to 60 min for the run name in
  the title) and records it (class served_page); the title and hub generation in the words of serve.FORCED_GENERATIONS for
  the run's tag (gen5: HWJ + 10 pc, generation-5 cutouts; gen6: HWJ + 10 pc + LPT, generation-6 cutouts from the rescaled
  cubes; another tag is refused, G26); the inspection pages built before requested once; the hub buttons
  (<S> <10-sec|1-min>-forced) in the hub unit.
refresh-siblings: the pages of a <TAG> run whose --sibling links differ from what the other pages serve now are registered
  again in one registration, with the current links and a new served time; judgements stay (journal on disk).
Parity: each run is byte-copied (minus judgements/ and server logs) to <out>/snapshot/<run> and registered in <out>/two_port
(ids p<port + offset>).
"""
import os
import subprocess

from exopipe.phase3.core import validate
from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.serve import common, exclusion_gate, flow, journals, register, scripts, servers

STAGES = ('switch', 'hub', 'verify')


def _CFG():
    from exopipe.phase3.config import CFG
    return CFG


def ready(name, review=None):
    """(ok, text): run ``name`` is ready to serve. A run built by exopipe3 forced --tier snapshot (family C) by its verify
    record (serve.FORCED_READY_RECORD, step forced, under forced_snapshot.WORK_ROOT): done, servable (associated, not
    restricted to an hour or cells), and naming ``review`` (the run directory the page serves) as its review copy - a record
    in any other state refuses, whatever log exists. A run without such a record (production's page runs): the last line of
    its log serve.FORCED_READY_LOG under paths.OPS_ROOT holds FORCED_READY_MARK."""
    from exopipe.phase3.core import record as recmod
    C = _CFG()
    S = C.serve
    unit = S.FORCED_READY_RECORD.format(name=name)
    rec = recmod.load_record(C.forced_snapshot.WORK_ROOT, 'forced', unit)
    if rec is not None:
        rp = recmod.record_path(C.forced_snapshot.WORK_ROOT, 'forced', unit)
        meta = rec.get('meta') or {}
        if rec.get('status') != 'done':
            return False, f'exopipe3 forced record {rp}: {rec.get("status")} ({(rec.get("reason") or "")[:160]})'
        if not meta.get('servable') or meta.get('partial'):
            return False, (f'exopipe3 forced record {rp}: done but not servable (servable {meta.get("servable")}, partial '
                           f'{meta.get("partial")}: an hour or cell restriction, or no association)')
        if review and os.path.realpath(meta.get('review') or '') != os.path.realpath(review):
            return False, f'exopipe3 forced record {rp} verified the review copy {meta.get("review")}, the page serves {review}'
        return True, f'exopipe3 forced record {rp} done: {(meta.get("verify_line") or "")[:160]}'
    p = os.path.join(C.paths.OPS_ROOT, S.FORCED_READY_LOG.format(name=name))
    try:
        with open(p) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
    except OSError:
        return False, f'no record {unit} under {C.forced_snapshot.WORK_ROOT} and no log {p}'
    last = lines[-1] if lines else ''
    return S.FORCED_READY_MARK in last, last[29:200] if len(last) > 29 else last


def missing_files(pg, D):
    return [f for f in (x.format(S=pg.stokes) for x in _CFG().serve.FORCED_REQUIRED) if not os.path.exists(os.path.join(D, f))]


def paired_view(d=None):
    """The transient-search view the forced pages are paired with (page_context_snapshot.py --served-from VIEW): the one run the
    registry entries serve.FORCED_PAGE_CONTEXT_VIEW_IDS (the snapshot 10-s and 1-minute pages) serve, in the registry directory
    ``d`` (default the production one serve.TWO_PORT_DIR, read only; a parity serve pairs its copies with the production view,
    read in place). Raises Refusal when an entry is missing, when they name different runs, or when the run has no search of its
    own (no chain_summary.json or combined_view_summary.json: the tool refuses it too)."""
    S = _CFG().serve
    runs = {}
    for pid in S.FORCED_PAGE_CONTEXT_VIEW_IDS:
        a = (register.registered_entry(d or S.TWO_PORT_DIR, pid) or {}).get('argv') or []
        i = a.index('--run') + 1 if '--run' in a else len(a)
        runs[pid] = a[i] if i < len(a) else None
    names = sorted({os.path.realpath(r) for r in runs.values() if r})
    if len(names) != 1 or not all(runs.values()):
        raise Refusal(f'no single transient-search view to pair the forced pages with: registry entries {runs} - old entries '
                      f'kept')
    view = names[0]
    if not any(os.path.exists(os.path.join(view, x)) for x in ('chain_summary.json', 'combined_view_summary.json')):
        raise Refusal(f'the paired view {view} has no chain_summary.json or combined_view_summary.json - old entries kept')
    return view


def page_context(u, plan, pg, D, view):
    """THE PER-PANEL QA BLOCK of a forced page, on the run it will serve (``D``; a parity serve: the copy), before it is
    registered: production's two hand steps on its forced runs, in its form and order (2026-11-13/14; the counts repointed by
    DONE-20261120-GEN5-FORCED-PAGES-COUNTS-REPOINTED-TO-SNAPCOMB-20261116; no production driver runs them):
      1. scintillation_index.py D --tier serve.SNAPSHOT_FILTER_TIER (cwd serve.SNAPSHOT_STEP_CWD; a parity serve: its own --cache)
         - D/scintillation_index.{csv,json}, the hours from the forced record (scintillation_index.forced_hours): the panel text's
         scintillation line and the page's scintillation slider;
      2. page_context_snapshot.py D --served-from VIEW --workers serve.SNAPSHOT_PAGE_CONTEXT_WORKERS (cwd the exopipe_ops root) -
         each panel's <png>.page.json (the transient-search view, global noise V / I, served detections I / V, scintillation,
         beam) and D/frame_counts.json naming VIEW, whose frame_counts_<S>_<tier>.csv.gz the n_i_epoch / n_v_epoch sliders read
         (R-GREGG-20261114-NEPOCH-SLIDERS-PER-FRAME).
    Then serve.FORCED_PAGE_CONTEXT_REQUIRED must exist and every test of classes scintillation_index and
    snapshot_forced_page_context (view = VIEW) pass, else Refusal: the page keeps its entry. This refusal is the gate production
    lacks: its page code falls back to the superseded counts with no refusal when frame_counts.json is absent
    (review_filters.py:262-271). Returns the notes."""
    from exopipe.phase3.lib.serve.snapshot_copy import NOHUGE, _parity_cache
    S = _CFG().serve
    notes = []
    args = [D, '--tier', S.SNAPSHOT_FILTER_TIER]
    cache = _parity_cache(plan, 'scintillation_table_index.json')
    if cache:
        args += ['--cache', cache]
    pctx = [D, '--served-from', view, '--workers', S.SNAPSHOT_PAGE_CONTEXT_WORKERS]
    for name, a, cwd in (('scintillation_index', args, S.SNAPSHOT_STEP_CWD), ('page_context_snapshot', pctx, None)):
        log = os.path.join(D, f'{name}.log')
        res = scripts.run(name, a, log=log, unit=u, env=NOHUGE, cwd=cwd)
        if not res.ok:
            raise Refusal(f'{scripts.SCRIPTS[name][0]} on {D}: {res.reason or scripts.last_line(res.tail)} (see {log}) - old '
                          f'entry kept')
        notes.append(f'page {pg.key}: {name}: {scripts.last_line(res.tail)[:200]}')
    miss = [f for f in S.FORCED_PAGE_CONTEXT_REQUIRED if not os.path.exists(os.path.join(D, f))]
    if miss:
        raise Refusal(f'{D} lacks {", ".join(miss)} after the page context - old entry kept')
    sci = os.path.join(D, 'scintillation_index.csv')
    for path, cls, ctx in ((sci, 'scintillation_index', dict(tier=S.SNAPSHOT_FILTER_TIER)),
                           (D, 'snapshot_forced_page_context', dict(view=view))):
        bad = [t for t in validate.run(cls, path, **ctx) if not t['passed']]
        if bad:      # this page only: the registration unit's own record would refuse every page
            raise Refusal('page context: ' + '; '.join(f'{t["name"]} ({t["detail"][:300]})' for t in bad[:3]) +
                          ' - old entry kept')
    for path, cls, ctx in ((sci, 'scintillation_index', dict(tier=S.SNAPSHOT_FILTER_TIER)),
                           (D, 'snapshot_forced_page_context', dict(view=view))):
        u.add_output(path, cls, **ctx)
    return notes


def _parity_copy(pg):
    os.makedirs(os.path.dirname(pg.root), exist_ok=True)
    p = subprocess.run(['rsync', '-a', '--exclude', 'judgements', '--exclude', 'classify_*.log',
                        pg.root_src.rstrip('/') + '/', pg.root + '/'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise Refusal(f'parity copy of {pg.root_src} failed: {p.stdout.decode()[-300:]}')


def check_page(u, plan, pg):
    """Checks 1-7 of the switch for ``pg`` inside the registration unit ``u`` (the in-place calls recorded there). Returns
    (run dir, expected sources, notes); raises Refusal (the page keeps its entry)."""
    S = _CFG().serve
    notes = []
    ok, last = ready(pg.run_name, pg.root_src)
    if not ok:
        raise Refusal(f'{pg.run_name} is not ready to serve ({last}) - old entry kept')
    miss = missing_files(pg, pg.root_src)
    if miss:
        raise Refusal(f'{pg.root_src} lacks {", ".join(miss)} (queue, catalogue, localization offsets or association table) '
                      f'- old entry kept')
    if plan.parity:
        _parity_copy(pg)
    D = pg.root
    res = scripts.run('forced_adleo', list(S.FORCED_ADLEO_CHECK[1]) + [D], python='py36', unit=u, refusal_codes=(1, 2))
    if not res.ok:
        raise Refusal(f'AD Leo check ({scripts.last_line(res.tail)}) - old entry kept')
    scan = exclusion_gate.scan_snapshot_run(D)
    refuses = exclusion_gate.refuses('snapshot', 'forced')
    notes.append(f'page {pg.key}: G17 serve-time exclusion gate ({"refusing" if refuses else "report only"} for the forced '
                 f'pages): {scan.summary()}')
    if scan.hits and refuses:
        raise Refusal(f'G17 serve-time exclusion gate: {scan.summary()} - old entry kept')
    notes += page_context(u, plan, pg, D, plan.forced_view)     # the per-panel QA block, before the App reads frame_counts.json
    exp = servers.app_probe(plan, pg, D, unit=u)        # before anything changes: a failed probe keeps the old entry
    now = plan.registered_run(pg)
    if now and os.path.realpath(now) == os.path.realpath(D):
        notes.append(f'page {pg.key}: {pg.port} already serves {pg.run_name} - registered again, nothing carried')
        return D, exp, notes
    old = journals.served_run_of_port(pg.port_cfg)     # the carry source: the production registry's entry (read only)
    oj = os.path.join(old, 'judgements', 'judgements.jsonl') if old else None
    if old and journals.journal_nonempty(oj):
        if journals.journal_nonempty(os.path.join(D, 'judgements', 'judgements.jsonl')):
            notes.append(f'page {pg.key}: journal already in {D} - carry skipped (live judgements kept)')
        else:
            cr = journals.snapshot_carry(old, D, unit=u)      # a refused carry refuses the page (gen5:42)
            notes.append(f'page {pg.key}: judgements carried from {old}: {cr}')
    else:
        notes.append(f'page {pg.key}: judgements: none in {old or "(no entry)"} - nothing to carry')
    return D, exp, notes


def switch(run, plan, state, utc):
    def prepare(u):
        flow.excl_note(u)
        passed, refused = [], {}
        try:
            plan.forced_view = paired_view()
        except Refusal as e:                      # no view to pair with: no page gets its QA block, none is registered
            for pg in plan.pages:
                refused[pg.key] = str(e)
                u.note(f'page {pg.key} REFUSED: {e}')
            return [], refused
        u.note(f'paired transient-search view (page_context_snapshot.py --served-from): {plan.forced_view}')
        for pg in plan.pages:
            try:
                D, exp, notes = check_page(u, plan, pg)
            except Refusal as e:
                refused[pg.key] = str(e)
                u.note(f'page {pg.key} REFUSED: {e}')
                continue
            for n in notes:
                u.note(n)
            j = os.path.join(D, 'judgements', 'judgements.jsonl')
            if os.path.exists(j):                 # the journal's content tests refuse this page only, never the others
                bad = [t for t in validate.run('judgement_journal', j) if not t['passed']]
                if bad:
                    refused[pg.key] = (f'judgement journal {j}: ' + '; '.join(f'{t["name"]} ({t["detail"]})' for t in bad[:3])
                                       + ' - old entry kept')
                    u.note(f'page {pg.key} REFUSED: {refused[pg.key]}')
                    continue
                u.add_output(j, 'judgement_journal')
            passed.append((pg, D, exp))
        plan.forced_after = {pg.key: D for pg, D, _ in passed}      # what each switched page serves after the switch
        items = [(pg, D, register.page_entry(plan, pg, utc, D), exp, []) for pg, D, exp in passed]
        for pg, D, e, _, _ in items:
            u.note(f'FORCED-SWITCH {e["id"]} {pg.stokes}/{pg.kind} {pg.run_name} siblings {e["argv"].count("--sibling")}')
        return items, refused
    state['served'].update(flow.register_pages(run, plan, prepare, _unit_name(plan), stage='switch', inc_log=False))


def refresh_siblings(run, plan, state, utc):
    tag = plan.forced_tag

    def prepare(u):
        items, refused = [], {}
        for pg in plan.pages:
            D = plan.registered_run(pg)
            if not D:
                refused[pg.key] = f'NO-INPUT: {pg.port}: no registered page'
                continue
            if tag and not D.rstrip('/').endswith(f'_{tag}'):
                refused[pg.key] = f'NO-INPUT: {pg.port} serves {D}, not a {tag} run'
                continue
            e = register.registered_entry(register.registry_dir(plan), f'p{pg.port}') or {}
            a = e.get('argv') or []
            have = sorted(a[i + 1] for i, x in enumerate(a) if x == '--sibling' and i + 1 < len(a))
            want = sorted(plan.forced_siblings(pg)[1::2])
            if have == want:
                refused[pg.key] = f'NO-INPUT: siblings of {pg.port} current'
                continue
            try:
                exp = servers.app_probe(plan, pg, D, unit=u)
            except Refusal as x:
                refused[pg.key] = str(x)
                continue
            u.note(f'FORCED-SIBLINGS-REFRESHED p{pg.port} ({D}): {want}')
            items.append((pg, D, register.page_entry(plan, pg, utc, D), exp, []))
        return items, refused
    state['served'].update(flow.register_pages(run, plan, prepare, _unit_name(plan) + '_siblings', stage='refresh-siblings',
                                               inc_log=False))


def _unit_name(plan):
    return f'snapshot_forced_{plan.forced_tag or "current"}_register'


def run(args, run, plan, stages):
    state = flow.new_state()
    utc = common.utc_text()
    if getattr(args, 'forced_refresh_siblings', False):
        refresh_siblings(run, plan, state, utc)
    elif 'switch' in stages:
        switch(run, plan, state, utc)
    if 'hub' in stages and not getattr(args, 'no_hub', False) and state['served']:
        flow.hub_stage(run, plan, state, utc=utc)
    if 'verify' in stages:
        flow.verify_stage(run, plan)


def describe(plan, stages, args):
    """The dry run: per page its run, readiness, required files, what it serves now, and the title and hub button text the
    switch would register (G26 names a tag without serve.FORCED_GENERATIONS text). Writes nothing."""
    utc = common.utc_text()
    if 'switch' in stages and not getattr(args, 'forced_refresh_siblings', False):
        try:
            print(f'  page context (scintillation_index.py, then page_context_snapshot.py --served-from): paired view '
                  f'{paired_view()}')
        except Refusal as e:
            print(f'  page context: REFUSED: {e}')
    for pg in plan.pages:
        ok, last = ready(pg.run_name, pg.root_src)
        miss = missing_files(pg, pg.root_src) if os.path.isdir(pg.root_src) else ['(run absent)']
        print(f'  {pg.key}: run {pg.root_src}  ready {"yes (" + last + ")" if ok else "NO (" + last + ")"}  '
              f'files {"ok" if not miss else "MISSING " + ", ".join(miss)}  serves now: {plan.registered_run(pg) or "-"}')
        if getattr(args, 'forced_refresh_siblings', False) or not plan.forced_tag:
            continue
        try:
            print(f'    title: {plan.title(pg, utc)}')
            h = plan.hub_entry(pg, utc)
            if h:
                print(f'    hub {h[0]} {h[1]}: {dict(kv.split("=", 1) for kv in h[2]).get("generation")}')
        except Refusal as e:
            print(f'    REFUSED: {e}')
