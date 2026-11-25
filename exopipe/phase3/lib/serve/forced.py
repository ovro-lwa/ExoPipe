"""serve --tier snapshot --family forced: the snapshot forced-photometry pages p8786 / p8785 / p8775 / p8776, each serving its
own run Run_SNAPFORCED_<S><T>_<TAG>, REGISTERED on the two-port server (serve owns these pages: refactor-chat decision
2026-10-13). Port of snapshot_forced_gen5_serve_20261026.sh - the 2026-10-07 serve step (snapshot_forced_serve_20261007.sh) on
the two-port server, whose per-page servers were retired 2026-10-22 - and of the 10-07 step's 'refresh-siblings' mode.

switch: unit snapshot_forced_<TAG>_register (flow.register_pages): the two-port server is checked first (nothing is carried
  for a registration that cannot happen), then per requested page (gen5:31-45; a page that fails a check keeps the run its
  entry serves; check_page), first what only reads - the tag has serve.FORCED_GENERATIONS text (G26), and no registry entry
  serves the run already (a served run is refused, never rewritten in place; gen5:38 registered it again) - then:
  1. the run is ready (ready(): an exopipe3 forced family-C run by its verify record - done, servable, for this review copy;
     a production page run by the last line FT-READY of <OPS_ROOT>/snapshot_forced_pages_run_<run>.log);
  2. it holds its queue, catalogue view, localization offsets and association table;
  3. the AD Leo check (test_snapshot_forced_reserve.py adleo --absent-ok <run>, site PYTHON 'py36'; Gregg 2026-10-08 "don't
     label AD Leo unless you have a metric defined and measured");
  4. the serve-time exclusion gate on the run: refuses the page under the ruled value (gate key 'snapshot-forced' in
     EXCLUSION_GATE_TIERS: Gregg 2026-10-14 R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS); production has no
     gate on these pages (the parity file: report only);
  5. the expected index sources (the server's own App on the run's queue and catalogue: servers.app_probe), before anything
     changes the run;
  6. THE PAGE INPUTS (page_context; the per-panel QA block, row 15): the four steps production's page runner takes before
     FT-READY (snapshot_forced_pages_run_20261007.sh, exopipe_ops bd07263), in its order, each refusing, in place on the run the
     page will serve: scintillation_index.py RUN --tier snapshot, bright_source_filter_precompute.py RUN --tier forced,
     page_context_snapshot.py RUN --served-from VIEW (VIEW the transient-search view the production registry's snapshot pages
     serve: paired_view, serve.FORCED_PAGE_CONTEXT_VIEW_IDS), tess/tess_coverage.py --run RUN --from-candidates; then their
     files present and their content tests passed (each run once, in the registration unit; a failure refuses this page only);
  7. Gregg's judgements carried (snapshot_carry_judgements.py OLD NEW) from the run the page's entry serves now (gen5 cur_run;
     a parity serve: the PRODUCTION registry's entry, read only) when its journal is not empty and the new run has none (a
     journal already in the new run: carry skipped, gen5:40).
  Then ALL pages that passed are registered in ONE registration (one registry change), each with --sibling links to what every
  other forced page serves AFTER the switch - its new run where it passed, else the run its entry serves now (gen5:47-56;
  plan.siblings) - and one unit per page waits on its build state (gen5:66-69 waited up to 60 min for the run name in
  the title) and records it (class served_page); the title and hub generation in the words of serve.FORCED_GENERATIONS for
  the run's tag (gen5: HWJ + 10 pc, generation-5 cutouts; gen6: HWJ + 10 pc + LPT, generation-6 cutouts from the rescaled
  cubes; another tag is refused, G26, for that page only); the inspection pages built before requested once; the hub buttons
  (<S> <10-sec|1-min>-forced) in the hub unit, each naming the run its page's entry serves (plan.hub_entry).
refresh-siblings: the pages of a <TAG> run whose --sibling links differ from what the other pages serve now are registered
  again in one registration, with the current links and a new served time; judgements stay (journal on disk).
Parity: each run is byte-copied (minus judgements/ and server logs) to <out>/snapshot/<run> and registered in <out>/two_port
(ids p<port + offset>).
"""
import os
import subprocess

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.serve import common, exclusion_gate, flow, journals, register, scripts, servers

STAGES = ('switch', 'hub', 'verify')


def _CFG():
    from exopipe.phase3.config import CFG
    return CFG


def forced_outs(review=None):
    """Where the family-C run serving ``review`` keeps its records: forced_snapshot.WORK_ROOT (a production run) and, for a
    review copy <out>/<forced_snapshot.REVIEW_SUBDIR_UNDER_OUT>/<name> (tier_snapshot.Ctx: a run under --out), that <out>."""
    fs = _CFG().forced_snapshot
    outs = [os.path.abspath(fs.WORK_ROOT)]
    p = os.path.dirname(os.path.abspath(review.rstrip('/'))) if review else None
    if p and os.path.basename(p) == fs.REVIEW_SUBDIR_UNDER_OUT:
        outs.append(os.path.dirname(p))
    return list(dict.fromkeys(outs))


def ready(name, review=None):
    """(ok, text): run ``name`` is ready to serve. A run built by exopipe3 forced --tier snapshot (family C) by its verify
    record (serve.FORCED_READY_RECORD, step forced) under the run's own --out (forced_outs: forced_snapshot.WORK_ROOT, or the
    <out> of a review copy <out>/review/<name>; the record naming this review copy first): done, servable (associated, not
    restricted to an hour or cells), and naming ``review`` (the run directory the page serves) as its review copy - a record
    in any other state refuses, whatever log exists. A run without such a record (production's page runs): the last line of
    its log serve.FORCED_READY_LOG under paths.OPS_ROOT holds FORCED_READY_MARK."""
    from exopipe.phase3.core import record as recmod
    C = _CFG()
    S = C.serve
    unit = S.FORCED_READY_RECORD.format(name=name)
    found = [(o, r) for o in forced_outs(review) for r in [recmod.load_record(o, 'forced', unit)] if r is not None]
    found.sort(key=lambda x: not (review and os.path.realpath((x[1].get('meta') or {}).get('review') or '')
                                  == os.path.realpath(review)))      # the record of this review copy first
    if found:
        out, rec = found[0]
        rp = recmod.record_path(out, 'forced', unit)
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
        return False, f'no record {unit} under {" or ".join(forced_outs(review))} and no log {p}'
    last = lines[-1] if lines else ''
    return S.FORCED_READY_MARK in last, last[29:200] if len(last) > 29 else last


def missing_files(pg, D):
    return [f for f in (x.format(S=pg.stokes) for x in _CFG().serve.FORCED_REQUIRED) if not os.path.exists(os.path.join(D, f))]


def paired_view(d=None):
    """The transient-search view the forced pages are paired with (page_context_snapshot.py --served-from VIEW): the one run the
    registry entries serve.FORCED_PAGE_CONTEXT_VIEW_IDS (the snapshot 10-s and 1-minute pages) serve, in the registry directory
    ``d`` (default the production one serve.TWO_PORT_DIR, read only; a parity serve pairs its copies with the production view,
    read in place). Raises Refusal when an entry is missing, when they name different runs, when the run has no search of its
    own (no chain_summary.json or combined_view_summary.json: the tool refuses it too) or no frame_counts.json of its own
    (production's runner: "the served snapshot view has no frame_counts.json - page inputs not built"). Production's runner
    takes the view from FT_SERVED_VIEW, else entry p8986 alone."""
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
    fc = os.path.join(view, 'frame_counts.json')
    if not (os.path.isfile(fc) and os.path.getsize(fc)):       # production's runner refuses here too (bd07263), before any step
        raise Refusal(f'the paired view {view} has no frame_counts.json - page inputs not built, old entries kept')
    return view


def page_context(u, plan, pg, D, view):
    """THE PAGE INPUTS of a forced page, on the run it will serve (``D``; a parity serve: the copy), before it is registered: the
    four steps production's page runner takes before FT-READY (snapshot_forced_pages_run_20261007.sh, exopipe_ops bd07263; by
    hand on the generation-5 runs: the bright-source filter and scintillation 2026-11-12/14, the page context and frame counts
    11-14/20, TESS 11-15), in its order, each from serve.SNAPSHOT_STEP_CWD as the runner (cd /tmp):
      1. scintillation_index.py D --tier serve.SNAPSHOT_FILTER_TIER (a parity serve: its own --cache) - D/scintillation_index.
         {csv,json}, the run's measured hours from its stage-1 record: the panel text's scintillation line and the page's slider;
      2. bright_source_filter_precompute.py D --tier serve.FORCED_BRIGHT_TIER --workers serve.FORCED_BRIGHT_WORKERS -
         D/bright_source_filter.npz (the 2-D bright-source panel; the cutout generation's flux scale);
      3. page_context_snapshot.py D --served-from VIEW --workers serve.FORCED_PAGE_CONTEXT_WORKERS - each panel's <png>.page.json
         and D/frame_counts.json naming VIEW, whose frame_counts_<S>_<tier>.csv.gz the n_i_epoch / n_v_epoch sliders read
         (R-GREGG-20261114-NEPOCH-SLIDERS-PER-FRAME; production's review_filters.py refuses a forced page without it since
         exopipe_ops 5889086);
      4. tess/tess_coverage.py --run D --from-candidates - D/tess_coverage.csv, one row per cand_id of the page's
         candidates/*/*/detections.csv - through lib/serve/tess.coverage, the one TESS helper of the deep, snapshot and 10-minute
         serves: the site PYTHON[pages_deep.TESS_PYTHON], the reader's arguments (--stokes from pages_deep.TESS_STOKES, --no-refresh
         in a parity run or with pages_deep.TESS_REFRESH_IN_READER False, the table then reaching now), and the written / left-out
         counts from this call's own summary line, which content test tess_rows (class tess_coverage) holds the rows to, against
         the run's master_catalog.csv (on a forced run the page's cand_ids: equal on all 8 forced runs checked 2026-11-24).
    What only reads comes first (the TESS interpreter, the orbit table). Steps 1-3 through snapshot_copy._step (the snapshot
    serve's tool-run helper; a failure raises Refusal for this page), step 4 through tess.coverage (the same). Then
    serve.FORCED_PAGE_CONTEXT_REQUIRED must exist and every content test pass, each run once through add_output (page_output,
    page_tests) - classes tess_coverage, scintillation_index, bright_source_filter (tier forced, the page's Stokes and cadence)
    and snapshot_forced_page_context (view = VIEW: the pointer and every panel's sidecar from it) - else Refusal: the page keeps
    its entry (production's runner tests the same files: every sidecar names VIEW, frame_counts.json names VIEW, the filter
    files not empty)."""
    from exopipe.phase3.lib.pages import common as pcommon, tess_orbits
    from exopipe.phase3.lib.serve import tess
    from exopipe.phase3.lib.serve.snapshot_copy import NOHUGE, _parity_cache, _step
    C = _CFG()
    S, TD = C.serve, C.pages_deep
    parity = bool(getattr(plan, 'parity', False))
    if not pcommon.interpreter_or_none(TD.TESS_PYTHON):
        raise Refusal(f"TESS coverage: the site PYTHON[{TD.TESS_PYTHON!r}] interpreter is not on this host - old entry kept")
    table = tess.orbit_table()
    why = tess_orbits.reader_args(table, parity=parity)[1]
    if why:
        raise Refusal(f'TESS coverage: the orbit table {table} {why} - the reader runs with --no-refresh and the refresh step '
                      f'(python -m exopipe.phase3.lib.pages.tess_orbits) has not kept it current - old entry kept')
    sci = [D, '--tier', S.SNAPSHOT_FILTER_TIER]
    cache = _parity_cache(plan, 'scintillation_table_index.json')
    if cache:
        sci += ['--cache', cache]
    steps = (('scintillation_index', sci, NOHUGE, 'default'),
             ('bright_source_filter', [D, '--tier', S.FORCED_BRIGHT_TIER, '--workers', S.FORCED_BRIGHT_WORKERS], NOHUGE, 'default'),
             ('page_context_snapshot', [D, '--served-from', view, '--workers', S.FORCED_PAGE_CONTEXT_WORKERS], NOHUGE, 'default'))
    for name, a, env, py in steps:
        _step(u, name, a, os.path.join(D, f'{name}.log'), cwd=S.SNAPSHOT_STEP_CWD, env=env, python=py, what=f'(page {pg.key}) ',
              where=D, after='old entry kept')
    n0 = len(u.tests)
    written, left_out = tess.coverage(u, D, ['--from-candidates'], os.path.join(D, 'tess_coverage.log'), cwd=S.SNAPSHOT_STEP_CWD,
                                      after=f'page {pg.key}: old entry kept', parity=parity)
    u.meta.setdefault('tess_coverage_pages', {})[pg.key] = dict(written=written, left_out=left_out)
    page_tests(u, pg, n0, os.path.join(D, 'tess_coverage.csv'), 'tess_coverage')
    miss = [f for f in S.FORCED_PAGE_CONTEXT_REQUIRED if not os.path.exists(os.path.join(D, f))]
    if miss:
        raise Refusal(f'{D} lacks {", ".join(miss)} after the page inputs - old entry kept')
    page_output(u, pg, os.path.join(D, 'scintillation_index.csv'), 'scintillation_index', tier=S.SNAPSHOT_FILTER_TIER)
    page_output(u, pg, os.path.join(D, 'bright_source_filter.npz'), 'bright_source_filter', tier=S.FORCED_BRIGHT_TIER,
                stokes=pg.stokes, cadence=pg.kind)
    page_output(u, pg, D, 'snapshot_forced_page_context', view=view)


# ---- the family's words: its PLAN_HOOK (plan.family_hook) ----------------------------------------------------------------
def forced_generation(page, run_name):
    """(samples text, cutout generation text) of a forced page serving ``run_name`` = Run_SNAPFORCED_<S><T>_<tag>
    (serve.FORCED_RUN_NAME): the first two fields of serve.FORCED_GENERATIONS[tag]. G26 refuses a run of another name or a tag without an entry -
    the title and the hub button name the samples and the generation the page shows (generation 6 adds LPT)."""
    S = _CFG().serve
    pre = S.FORCED_RUN_NAME.format(S=page.stokes, T=page.kind, tag='')
    tag = run_name[len(pre):] if run_name.startswith(pre) else None
    if tag not in S.FORCED_GENERATIONS:
        raise Refusal(f'G26 forced page {page.key}: run {run_name!r} - no serve.FORCED_GENERATIONS entry for its tag {tag!r} '
                      f'(the samples and cutout generation its title and hub button name; known: '
                      f'{", ".join(sorted(S.FORCED_GENERATIONS))})')
    return tuple(S.FORCED_GENERATIONS[tag][:2])


class ForcedHook:
    """PLAN_HOOK of the snapshot forced pages (one run per page): snapshot_forced_serve_20261007.sh:49-51 --run --queue
    --catalog-dir --port --loc-offsets --default-fp-reason, then --sibling per other page of the family (what it serves AFTER
    this serve: the run this serve registers for it, plan.forced_after, else its registry entry's run;
    snapshot_forced_gen5_serve_20261026.sh:47-56, '<port>:<run>/source_catalogs_<S>:<label>'), --title serve.FORCED_TITLE;
    environment serve.SERVER_ENV[ENV_KEY]; hub words serve.FORCED_HUB_GENERATION. lib/serve/forced_tier.ForcedTierHook builds
    on it."""
    ENV_KEY = 'snapshot-forced'

    def code(self, plan, page):
        return _CFG().serve.SERVER_ARGV0[plan.tier].format(ops=plan.ops)

    def env(self, plan, page):
        return _CFG().serve.SERVER_ENV[self.ENV_KEY]

    def title(self, plan, page, utc, root=None):
        S = _CFG().serve
        run_name = os.path.basename(os.path.normpath(root)) if root else page.run_name
        samples, gen = forced_generation(page, run_name)
        return S.FORCED_TITLE.format(S=page.stokes, what=S.SNAPSHOT_WHAT.get(page.kind, ''), run_name=run_name, utc=utc,
                                     samples=samples, generation=gen)

    def sibling_kind(self, o):
        """The <T> of a sibling's link label (serve.FORCED_SIBLING_LABEL)."""
        return '10-s' if o.kind == '10s' else '1-min'

    def siblings(self, plan, page, root=None):
        S = _CFG().serve
        out = []
        for o in plan.all_pages:
            if o.family != page.family or o.key == page.key:
                continue
            run = plan.forced_after.get(o.key) or plan.registered_run(o)
            if run:
                lab = S.FORCED_SIBLING_LABEL.format(S=o.stokes, T=self.sibling_kind(o))
                out += ['--sibling', f'{o.port}:{o.at("catalog", run)}:{lab}']
        return out

    def argv(self, plan, page, utc, root=None):
        S = _CFG().serve
        U = root or page.root
        argv = [plan.page_python(page), self.code(plan, page), '--run', U, '--queue', page.at('queue', U), '--catalog-dir',
                page.at('catalog', U), '--port', str(page.port), '--loc-offsets', '--default-fp-reason',
                S.DEFAULT_FP_REASON['snapshot']]
        argv += self.siblings(plan, page, U) + ['--title', self.title(plan, page, utc, U)]
        if plan.parity:
            argv += ['--lists-dir', plan.lists_dir]
        return argv

    def hub_words(self, plan, page, root=None):
        """From ``root``, else the run its entry in this plan's registry serves (what the button points at after a switch or a
        sibling refresh), else the plan's name; G26 refuses a tag without serve.FORCED_GENERATIONS text."""
        S = _CFG().serve
        root = root or plan.registered_run(page)
        run_name = os.path.basename(os.path.normpath(root)) if root else page.run_name
        samples, g = forced_generation(page, run_name)
        return run_name, S.FORCED_HUB_GENERATION.format(S=page.stokes, kind=page.kind, samples=samples, generation=g)


PLAN_HOOK = ForcedHook()


def page_output(u, pg, path, cls, what=None, **ctx):
    """u.add_output(path, cls, **ctx) in the registration unit - each content test runs once - with a failure refusing page
    ``pg`` only: the output is recorded as a refused item (core.record refuse_item, so the unit's status ignores its failed
    tests and the other pages go on), then Refusal (the page keeps its entry)."""
    n0 = len(u.tests)
    u.add_output(path, cls, **ctx)
    page_tests(u, pg, n0, path, cls, what)


def page_tests(u, pg, n0, path, cls, what=None):
    """The tests recorded in ``u`` since its n0-th (the content tests of ``path``, recorded by page_output or by a shared helper
    such as tess.coverage): a failure refuses page ``pg`` only, as page_output says."""
    bad = [t for t in u.tests[n0:] if not t['passed'] and not t.get('waived')]
    if bad:
        why = '; '.join(f'{t["name"]} ({str(t.get("detail") or t.get("value") or "")[:300]})' for t in bad[:3])
        u.refuse_item(path, f'page {pg.key}: {why}', gate=cls)
        raise Refusal(f'{what or cls}: {why} - old entry kept')


def _parity_copy(pg):
    os.makedirs(os.path.dirname(pg.root), exist_ok=True)
    p = subprocess.run(['rsync', '-a', '--exclude', 'judgements', '--exclude', 'classify_*.log',
                        pg.root_src.rstrip('/') + '/', pg.root + '/'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise Refusal(f'parity copy of {pg.root_src} failed: {p.stdout.decode()[-300:]}')


def check_page(u, plan, pg):
    """Checks 1-7 of the switch for ``pg`` inside the registration unit ``u`` (the in-place calls recorded there). Returns
    (run dir, expected sources, notes); raises Refusal (the page keeps its entry). Everything that only reads comes before
    anything changes: the tag's generation text (G26), readiness, the files, and whether a registry entry serves the run
    already - a served run is refused (never rewritten in place: the page inputs write its sidecars, scintillation table,
    bright-source filter, frame_counts.json and TESS table while its page reads them; a deliberate departure from gen5:38, which registered it again) - then the
    parity copy, AD Leo, the exclusion gate and the App probe (it reads the queue and catalogue, which the page inputs leave
    alone), and only then the page inputs and the judgement carry."""
    S = _CFG().serve
    notes = []
    forced_generation(pg, os.path.basename(os.path.normpath(pg.root_src)))      # G26, before anything changes
    ok, last = ready(pg.run_name, pg.root_src)
    if not ok:
        raise Refusal(f'{pg.run_name} is not ready to serve ({last}) - old entry kept')
    miss = missing_files(pg, pg.root_src)
    if miss:
        raise Refusal(f'{pg.root_src} lacks {", ".join(miss)} (queue, catalogue, localization offsets or association table) '
                      f'- old entry kept')
    D = pg.root
    on = served_by(plan, D)
    if on:
        raise Refusal(f'{D} is served already ({", ".join(on)}): the page inputs would rewrite a served run in place - '
                      f'old entry kept (a new run name serves a new build)')
    if plan.parity:
        _parity_copy(pg)
    res = scripts.run('forced_adleo', list(S.FORCED_ADLEO_CHECK[1]) + [D], python='py36', unit=u, refusal_codes=(1, 2))
    if not res.ok:
        raise Refusal(f'AD Leo check ({scripts.last_line(res.tail)}) - old entry kept')
    scan = exclusion_gate.scan_snapshot_run(D)
    refuses = exclusion_gate.refuses('snapshot', 'forced')
    notes.append(f'page {pg.key}: G17 serve-time exclusion gate ({"refusing" if refuses else "report only"} for the forced '
                 f'pages): {scan.summary()}')
    if scan.hits and refuses:
        raise Refusal(f'G17 serve-time exclusion gate: {scan.summary()} - old entry kept')
    exp = servers.app_probe(plan, pg, D, unit=u)        # before anything changes: a failed probe keeps the old entry
    page_context(u, plan, pg, D, plan.forced_view)      # the page inputs, on a run no page serves
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


def served_by(plan, D):
    """The ids of the entries of the plan's registry (register.registry_dir) whose --run is the run directory ``D``."""
    import json
    try:
        with open(register.files(register.registry_dir(plan))[0]) as f:
            pages = json.load(f).get('pages', [])
    except OSError:
        return []
    except ValueError as x:
        raise Refusal(f'the registry of {register.registry_dir(plan)} is unreadable ({x}) - old entry kept')
    want = os.path.realpath(D)
    out = []
    for e in pages:
        a = e.get('argv') or []
        i = a.index('--run') + 1 if '--run' in a else len(a)
        if i < len(a) and os.path.realpath(a[i]) == want:
            out.append(e.get('id'))
    return out


def switch(run, plan, state, utc):
    def prepare(u):
        flow.excl_note(u)
        passed, refused = [], {}
        try:
            plan.forced_view = paired_view()
        except Refusal as e:                      # no view to pair with: no page gets its inputs, none is registered
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
                try:
                    page_output(u, pg, j, 'judgement_journal', what=f'judgement journal {j}')
                except Refusal as e:
                    refused[pg.key] = str(e)
                    u.note(f'page {pg.key} REFUSED: {e}')
                    continue
            passed.append((pg, D, exp))
        while True:     # a page whose entry refuses leaves the others, their sibling links recomputed without it
            plan.forced_after = {pg.key: D for pg, D, _ in passed}      # what each switched page serves after the switch
            items, bad = [], None
            for pg, D, exp in passed:
                try:
                    items.append((pg, D, register.page_entry(plan, pg, utc, D), exp, []))
                except Refusal as e:
                    bad = (pg, e)
                    break
            if bad is None:
                break
            refused[bad[0].key] = str(bad[1])
            u.note(f'page {bad[0].key} REFUSED: {bad[1]}')
            passed = [x for x in passed if x[0] is not bad[0]]
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
            want = sorted(plan.siblings(pg)[1::2])
            if have == want:
                refused[pg.key] = f'NO-INPUT: siblings of {pg.port} current'
                continue
            try:
                exp = servers.app_probe(plan, pg, D, unit=u)
                e = register.page_entry(plan, pg, utc, D)      # G26 refuses this page only
            except Refusal as x:
                refused[pg.key] = str(x)
                continue
            u.note(f'FORCED-SIBLINGS-REFRESHED p{pg.port} ({D}): {want}')
            items.append((pg, D, e, exp, []))
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
            print(f'    title: {plan.title(pg, utc, pg.root_src)}')
            h = plan.hub_entry(pg, utc, pg.root_src)
            if h:
                print(f'    hub {h[0]} {h[1]}: {dict(kv.split("=", 1) for kv in h[2]).get("generation")}')
        except Refusal as e:
            print(f'    REFUSED: {e}')
