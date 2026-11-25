"""serve --tier snapshot --family forced-tier: the 10-minute and deep forced-photometry pages p9120 (V 10-minute), p9121 (I
10-minute), p9122 (V deep), p9123 (I deep), each serving its run Run_SNAPFORCED_<S><10min|deep>_gen<N> (forced_tier_pages.
RUN_NAME) under paths.SNAPSHOT_REVIEW_ROOT, REGISTERED on the two-port server (registered by hand 2026-10-27 08:04:50 UTC, ledger
HWJ10PC-FORCED-10MIN-DEEP-FULL-CHAIN-FITS-LOC-ASSOC-REREGISTERED-20261027). Family T (exopipe3 forced --stages tier_runs, lib/
forced/tier_pages.py) builds each run as <run>.new beside the served one; this family INSTALLS it - production's `swap` /
`swap-stage` (tier_forced_page_20261027.py:1192-1206, :1256-1277) - with the page inputs production added by hand afterwards,
then registers the four pages once and sets the hub buttons <S> 10-min-forced / 1-hour-forced.

install: unit snapshot_forced-tier_install (flow.register_pages): the two-port server is checked first, then per requested page
  (a page that fails a check keeps its entry and its run; the others go on), first what only reads:
  1. the new build is ready: <root>/<run>.new exists and family T's record T_runs (under forced_tier_pages.SITE_ROOT, or the
     --out of a run whose review root is <out>/forced_snapshot.REVIEW_SUBDIR_UNDER_OUT) is done, names this review root, and
     wrote <run>.new/queue_<S>.txt (class forced_tier_run); that content test is run again now on the .new tree, with the
     record's site (every card served once as a byte copy, the radio positions, queue = catalogue = association);
  2. the tree holds what the page reads (serve.FORCED_TIER_REQUIRED);
  3. no earlier install of the run is pending (<root>/serve.FORCED_TIER_STAGE exists: an interrupted install, refused until it
     is looked at) and the kept name serve.FORCED_TIER_KEEP is free;
  4. the serve-time exclusion gate on the new tree (lib/serve/exclusion_gate.scan_forced_tier_run, every CSV keyed per night,
     read now; gate key 'snapshot-forced-tier': refusing);
  then the new tree is STAGED - renamed to <root>/serve.FORCED_TIER_STAGE/<run>, so it carries the run's own name, which the
  page-input tools and the page read (scintillation_index.forced_run) - and on it:
  5. the title and hub words (G26: the run's queue header names its samples, Stokes and tier; the run name its generation);
  6. the expected index sources (the server's own App on the staged queue and catalogue: servers.app_probe);
  7. THE PAGE INPUTS (page_inputs; production's interim pushes, ledger PUSH-20261112-FORCED-PAGES-FILTER-FILES and
     PREAMBLE-20261115-TESS-COVERAGE-INTERIM-SERVED-RUNS), each refusing, then their content tests;
  8. the exclusion gate again, on every file the page will serve.
  A failure in 5-8 renames the staged tree back to <run>.new (nothing installed; the page inputs written into it are rebuilt
  by the next install). Then THE INSTALL (production's swap-stage): the served run's judgements/ copied into the staged run
  (rsync -a --update), the served run renamed serve.FORCED_TIER_KEEP, the staged run renamed into its place, the kept run's
  judgements/ copied again (a label written in between); a first install (no served run) only renames. The kept run stays.
  Then ALL installed pages are registered in ONE registration (one registry change; the two-port server rebuilds them from
  it), each with --sibling links to its three sibling pages (plan.forced_siblings), the title in serve.FORCED_TIER_TITLE words
  with the served time; one unit per page waits on its build and records it (class served_page).
hub: the buttons <S> 10-min-forced / 1-hour-forced (plan.hub_entry, serve.FORCED_TIER_HUB_GENERATION).
verify: the served_page records re-tested.
A requested page without a <run>.new has nothing to install: NO-INPUT (nothing registered for it; no done record).
Parity (--out outside the production roots): the pages install into <out>/forced_snapshot.REVIEW_SUBDIR_UNDER_OUT - where family
T writes under the same --out - and register in <out>/two_port; the tool caches under serve.SERVE_PARITY_CACHE.
"""
import os
import subprocess

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.serve import common, exclusion_gate, flow, register, servers
from exopipe.phase3.lib.serve.forced import page_output, page_tests

STAGES = ('install', 'hub', 'verify')
RECORD_UNIT = 'T_runs'            # lib/forced/tier_pages.UNITS['tier_runs']


def _CFG():
    from exopipe.phase3.config import CFG
    return CFG


def install_root(pg):
    """The directory of the served run and of its new build <run>.new: paths.SNAPSHOT_REVIEW_ROOT (a parity serve: <out>/
    forced_snapshot.REVIEW_SUBDIR_UNDER_OUT)."""
    return os.path.dirname(os.path.abspath(pg.root))


def new_tree(pg):
    return os.path.abspath(pg.root) + _CFG().forced_tier_pages.NEW_SUFFIX


def stage_dir(pg):
    return os.path.join(install_root(pg), _CFG().serve.FORCED_TIER_STAGE.format(run=pg.run_name))


def keep_path(pg, utc):
    return os.path.join(install_root(pg), _CFG().serve.FORCED_TIER_KEEP.format(run=pg.run_name, utc=utc))


def record_outs(root):
    """Where family T keeps the T_runs record of a build into review root ``root``: its production --out forced_tier_pages.
    SITE_ROOT, and the --out of a run under --out (review root <out>/forced_snapshot.REVIEW_SUBDIR_UNDER_OUT)."""
    C = _CFG()
    outs = [os.path.abspath(C.forced_tier_pages.SITE_ROOT)]
    if os.path.basename(os.path.abspath(root)) == C.forced_snapshot.REVIEW_SUBDIR_UNDER_OUT:
        outs.append(os.path.dirname(os.path.abspath(root)))
    return list(dict.fromkeys(outs))


def ready(pg):
    """(ok, text, site): the page's <run>.new is a family T build ready to install - the T_runs record (record_outs) is done,
    names this review root, and wrote <run>.new/queue_<S>.txt (class forced_tier_run) with every test of it passed. ``site``
    is the record's site (the content test's input)."""
    from exopipe.phase3.core import record as recmod
    root, new = install_root(pg), new_tree(pg)
    if not os.path.isdir(new):
        return False, f'NO-INPUT: no new build {new} to install', None
    q = os.path.join(new, f'queue_{pg.stokes}.txt')
    tried = []
    for out in record_outs(root):
        rec = recmod.load_record(out, 'forced', RECORD_UNIT)
        rp = recmod.record_path(out, 'forced', RECORD_UNIT)
        if rec is None:
            tried.append(f'no {rp}')
            continue
        meta = rec.get('meta') or {}
        if os.path.realpath(meta.get('review_root') or '') != os.path.realpath(root):
            tried.append(f'{rp} built into {meta.get("review_root")}, not {root}')
            continue
        if rec.get('status') != 'done':
            return False, f'family T record {rp}: {rec.get("status")} ({(rec.get("reason") or "")[:160]})', None
        outs = [o for o in rec.get('outputs') or [] if o.get('class') == 'forced_tier_run'
                and os.path.realpath(o.get('path') or '') == os.path.realpath(q)]
        bad = [t for t in rec.get('tests') or [] if os.path.realpath(t.get('path') or '') == os.path.realpath(q)
               and not t.get('passed') and not t.get('waived')]
        if not outs or bad:
            return False, (f'family T record {rp} done but ' + ('it wrote no ' + q if not outs else
                                                                 f'{len(bad)} test(s) of {q} failed')), None
        if not meta.get('site'):
            return False, f'family T record {rp} names no site', None
        return True, f'family T record {rp} done: {(meta.get("verify_line") or "")[:160]}', meta['site']
    return False, f'no family T record of a build into {root} ({"; ".join(tried)})', None


def missing_files(pg, D):
    return [f for f in (x.format(S=pg.stokes) for x in _CFG().serve.FORCED_TIER_REQUIRED)
            if not os.path.exists(os.path.join(D, f))]


def _gate(u, plan, pg, D, when):
    scan = exclusion_gate.scan_forced_tier_run(D)
    refuses = exclusion_gate.refuses('snapshot', 'forced-tier')
    u.note(f'page {pg.key}: G17 serve-time exclusion gate {when} ({"refusing" if refuses else "report only"}): '
           f'{scan.summary()}')
    if scan.hits and refuses:
        raise Refusal(f'G17 serve-time exclusion gate {when}: {scan.summary()} - nothing installed')


def page_inputs(u, plan, pg, D):
    """THE PAGE INPUTS of a 10-minute / deep forced run, on the staged run ``D`` (it carries the run's own name: the tools
    recognize a forced run by it), in production's order, each from serve.SNAPSHOT_STEP_CWD with numpy's huge-page advice off:
      1. scintillation_index.py D --tier serve.FORCED_TIER_SCINT_TIER[kind] (tenmin | deep; a parity serve: its own --cache) -
         D/scintillation_index.{csv,json}: the run's measured hours from the tier files (the page's scintillation slider);
      2. bright_source_filter_precompute.py D --tier serve.FORCED_BRIGHT_TIER --parents <paths.TRANSIENTS_ROOT>/
         serve.FORCED_TIER_PARENTS[kind] (the search run whose Masters the event planes subtracted; the tool refuses another)
         --workers serve.FORCED_BRIGHT_WORKERS - D/bright_source_filter.npz (the 2-D bright-source panel);
      3. 10 minutes: tess/frame_times_tenmin.py --run D --image-root <paths.SUBLST_ROOT>/serve.FORCED_TIER_TENMIN_TREE (--reuse
         the served run's frame_times.csv when not empty) - D/frame_times.csv, the windows of the served events (lib/serve/
         tess.tenmin_frame_times, content test against the served candidates);
      4. tess/tess_coverage.py --run D --from-candidates (deep: --frame-times <paths.TRANSIENTS_ROOT>/
         serve.FORCED_TIER_DEEP_FRAME_TIMES) through lib/serve/tess.coverage - D/tess_coverage.csv, its rows held to the served
         events (candidates/*/*/detections.csv; the run has no master_catalog.csv).
    Then serve.FORCED_TIER_INPUTS_REQUIRED must exist and the content tests pass (scintillation_index of the tier,
    bright_source_filter tier forced with the page's Stokes and cadence), each once in the unit; a failure refuses this page
    only (Refusal)."""
    from exopipe.phase3.lib.pages import common as pcommon, tess_orbits
    from exopipe.phase3.lib.serve import tess
    from exopipe.phase3.lib.serve.snapshot_copy import NOHUGE, _parity_cache, _step
    C = _CFG()
    S, P, TD = C.serve, C.paths, C.pages_deep
    parity = bool(getattr(plan, 'parity', False))
    after = 'nothing installed (the served run stays)'
    if not pcommon.interpreter_or_none(TD.TESS_PYTHON):
        raise Refusal(f"TESS coverage: the site PYTHON[{TD.TESS_PYTHON!r}] interpreter is not on this host - {after}")
    why = tess_orbits.reader_args(tess.orbit_table(), parity=parity)[1]
    if why:
        raise Refusal(f'TESS coverage: the orbit table {tess.orbit_table()} {why} - the reader runs with --no-refresh and the '
                      f'refresh step (python -m exopipe.phase3.lib.pages.tess_orbits) has not kept it current - {after}')
    sci_tier = S.FORCED_TIER_SCINT_TIER[pg.kind]
    sci = [D, '--tier', sci_tier]
    cache = _parity_cache(plan, 'scintillation_table_index.json')
    if cache:
        sci += ['--cache', cache]
    parents = os.path.join(P.TRANSIENTS_ROOT, S.FORCED_TIER_PARENTS[pg.kind])
    steps = (('scintillation_index', sci),
             ('bright_source_filter', [D, '--tier', S.FORCED_BRIGHT_TIER, '--parents', parents,
                                       '--workers', S.FORCED_BRIGHT_WORKERS]))
    for name, a in steps:
        _step(u, name, a, os.path.join(D, f'{name}.log'), cwd=S.SNAPSHOT_STEP_CWD, env=NOHUGE, python='default',
              what=f'(page {pg.key}) ', where=D, after=after)
    tess_args = ['--from-candidates']
    if pg.kind == '10min':
        ft = os.path.join(D, 'frame_times.csv')
        n0 = len(u.tests)
        try:
            tess.tenmin_frame_times(u, D, os.path.join(P.SUBLST_ROOT, S.FORCED_TIER_TENMIN_TREE),
                                    os.path.join(pg.root, 'frame_times.csv'), os.path.join(D, 'frame_times_tenmin.log'),
                                    after=f'page {pg.key}: {after}', cwd=S.SNAPSHOT_STEP_CWD, served='candidates')
        except Refusal:
            page_tests(u, pg, n0, ft, 'tenmin_frame_times')
            raise
    else:
        tess_args += ['--frame-times', os.path.join(P.TRANSIENTS_ROOT, S.FORCED_TIER_DEEP_FRAME_TIMES)]
    n0 = len(u.tests)
    try:
        written, left_out = tess.coverage(u, D, tess_args, os.path.join(D, 'tess_coverage.log'), cwd=S.SNAPSHOT_STEP_CWD,
                                          after=f'page {pg.key}: {after}', parity=parity, served='candidates')
    except Refusal:
        page_tests(u, pg, n0, os.path.join(D, 'tess_coverage.csv'), 'tess_coverage')
        raise
    u.meta.setdefault('tess_coverage_pages', {})[pg.key] = dict(written=written, left_out=left_out)
    page_tests(u, pg, n0, os.path.join(D, 'tess_coverage.csv'), 'tess_coverage')
    req = list(S.FORCED_TIER_INPUTS_REQUIRED) + (['frame_times.csv'] if pg.kind == '10min' else [])
    miss = [f for f in req if not os.path.exists(os.path.join(D, f))]
    if miss:
        raise Refusal(f'{D} lacks {", ".join(miss)} after the page inputs - {after}')
    page_output(u, pg, os.path.join(D, 'scintillation_index.csv'), 'scintillation_index', tier=sci_tier)
    page_output(u, pg, os.path.join(D, 'bright_source_filter.npz'), 'bright_source_filter', tier=S.FORCED_BRIGHT_TIER,
                stokes=pg.stokes, cadence=pg.kind)


def _sync_judgements(src, dst):
    """production's swap-stage sync (tier_forced_page_20261027.py:1269-1270): rsync -a --update src/judgements/ dst/judgements/,
    when src has one."""
    j = os.path.join(src, 'judgements')
    if not os.path.isdir(j):
        return 'none'
    p = subprocess.run(['rsync', '-a', '--update', j + '/', os.path.join(dst, 'judgements') + '/'], stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise Refusal(f'judgements copy {j}/ -> {dst}/judgements/ failed (rsync exit {p.returncode}: '
                      f'{p.stdout.decode(errors="replace")[-300:]})')
    return f'{sum(len(f) for _r, _d, f in os.walk(j))} file(s)'


def install(u, pg, D, keep):
    """The swap (production's swap-stage): the served run's judgements into the staged run ``D``, the served run renamed
    ``keep``, ``D`` renamed into its place, the kept run's judgements again; the stage directory removed. A first install (no
    served run) only renames. A failure before the served run moved leaves it served (Refusal); a failure of the rename into
    place puts the served run back."""
    live = os.path.abspath(pg.root)
    first = not os.path.lexists(live)
    if not first:
        n = _sync_judgements(live, D)
        u.note(f'page {pg.key}: judgements of the served run copied into the staged run: {n}')
        os.rename(live, keep)
        try:
            os.rename(D, live)
        except OSError as e:
            os.rename(keep, live)
            raise Refusal(f'install of {D} as {live} failed ({e}) - the served run put back')
        n = _sync_judgements(keep, live)        # a label written between the copy and the rename
        u.note(f'page {pg.key}: INSTALLED {live} (the previous run kept as {keep}; judgements copied again: {n})')
    else:
        os.rename(D, live)
        u.note(f'page {pg.key}: INSTALLED {live} (first install: no served run)')
    os.rmdir(os.path.dirname(D))
    return None if first else keep


def unstage(pg, D):
    """A refused install: the staged tree back to <run>.new, the stage directory removed (the page inputs written into it are
    rebuilt by the next install)."""
    if os.path.isdir(D):
        os.rename(D, new_tree(pg))
    try:
        os.rmdir(os.path.dirname(D))
    except OSError:
        pass


def check_page(u, plan, pg, keep_utc):
    """Checks 1-8 and the install of ``pg`` inside the registration unit ``u``. Returns (installed run, expected sources,
    notes, kept run); raises Refusal (the page keeps its entry and its run) - 'NO-INPUT: ...' when there is no new build."""
    from exopipe.phase3.lib.serve.plan import forced_tier_words
    notes = []
    ok, text, site = ready(pg)
    if not ok:
        raise Refusal(text if text.startswith('NO-INPUT: ') else f'{pg.run_name}.new is not ready to install ({text}) - '
                                                                 f'nothing installed')
    notes.append(f'page {pg.key}: {text}')
    new = new_tree(pg)
    page_output(u, pg, os.path.join(new, f'queue_{pg.stokes}.txt'), 'forced_tier_run', site=site, tier=pg.kind,
                stokes=pg.stokes)
    miss = missing_files(pg, new)
    if miss:
        raise Refusal(f'{new} lacks {", ".join(miss)} - nothing installed')
    sd, keep = stage_dir(pg), keep_path(pg, keep_utc)
    if os.path.lexists(sd):
        raise Refusal(f'{sd} exists: an earlier install of {pg.run_name} did not finish - look at it (its run back to '
                      f'{os.path.basename(new)}, or installed by hand) before installing again')
    if os.path.lexists(keep):
        raise Refusal(f'the kept name {keep} exists - nothing installed')
    _gate(u, plan, pg, new, 'on the new build')
    os.mkdir(sd)
    D = os.path.join(sd, pg.run_name)
    os.rename(new, D)
    try:
        forced_tier_words(pg, D)                      # G26: the title and hub words from the run itself
        exp = servers.app_probe(plan, pg, D, unit=u)
        page_inputs(u, plan, pg, D)
        _gate(u, plan, pg, D, 'on the staged run after its page inputs')
        j = os.path.join(D, 'judgements', 'judgements.jsonl')
        if os.path.exists(j):
            page_output(u, pg, j, 'judgement_journal', what=f'judgement journal {j}')
    except BaseException:
        unstage(pg, D)
        raise
    kept = install(u, pg, D, keep)
    j = os.path.join(pg.root, 'judgements', 'judgements.jsonl')
    if os.path.exists(j):
        u.add_output(j, 'judgement_journal')       # the installed journal (after the second copy)
    return os.path.abspath(pg.root), exp, notes, kept


def install_stage(run, plan, state, utc):
    keep_utc = common.stamp() + 'Z'          # the kept run's name: serve.FORCED_TIER_KEEP

    def prepare(u):
        flow.excl_note(u, 'snapshot')
        passed, refused = [], {}
        for pg in plan.pages:
            try:
                D, exp, notes, kept = check_page(u, plan, pg, keep_utc)
            except Refusal as e:
                refused[pg.key] = str(e)
                u.note(f'page {pg.key} REFUSED: {e}')
                continue
            for n in notes:
                u.note(n)
            u.meta.setdefault('installed', {})[pg.key] = dict(run=D, kept=kept)
            passed.append((pg, D, exp))
        plan.forced_after = {pg.key: D for pg, D, _ in passed}
        items = []
        for pg, D, exp in passed:
            try:
                items.append((pg, D, register.page_entry(plan, pg, utc, D), exp, []))
            except Refusal as e:          # the run is installed; its entry keeps the served time it had (the server rebuilds
                refused[pg.key] = str(e)  # it at the next registry write)
                u.note(f'page {pg.key}: installed, entry REFUSED: {e}')
        for pg, D, e, _, _ in items:
            u.note(f'FORCED-TIER-INSTALL {e["id"]} {pg.stokes}/{pg.kind} {pg.run_name} siblings {e["argv"].count("--sibling")}')
        return items, refused
    state['served'].update(flow.register_pages(run, plan, prepare, f'snapshot_{plan.family}_install', stage='install',
                                               inc_log=False))


def run(args, run, plan, stages):
    state = flow.new_state()
    utc = common.utc_text()
    if 'install' in stages:
        install_stage(run, plan, state, utc)
    if 'hub' in stages and not getattr(args, 'no_hub', False) and state['served']:
        flow.hub_stage(run, plan, state, utc=utc)
    if 'verify' in stages:
        flow.verify_stage(run, plan)


def describe(plan, stages, args):
    """The dry run: per page the served run, its new build and readiness, the files, the kept name, the exclusion gate on the
    new build, and the title and hub button text the install would register. Writes nothing."""
    from exopipe.phase3.lib.serve.plan import forced_tier_words
    utc = common.utc_text()
    for pg in plan.pages:
        new = new_tree(pg)
        ok, text, _site = ready(pg)
        print(f'  {pg.key}: served run {pg.root} ({"present" if os.path.isdir(pg.root) else "absent"}); new build {new}: '
              f'{"ready (" + text + ")" if ok else text}; registry serves {plan.registered_run(pg) or "-"}')
        if not os.path.isdir(new):
            continue
        miss = missing_files(pg, new)
        print(f'    files {"ok" if not miss else "MISSING " + ", ".join(miss)}; stage {stage_dir(pg)}'
              f'{" EXISTS (refused)" if os.path.lexists(stage_dir(pg)) else ""}; the served run would be kept as '
              f'{keep_path(pg, "<UTC>")}')
        try:
            scan = exclusion_gate.scan_forced_tier_run(new)
            print(f'    G17 exclusion gate on the new build: {scan.summary()}'
                  + (' - REFUSED' if scan.hits and exclusion_gate.refuses('snapshot', 'forced-tier') else ''))
            samples, gen = forced_tier_words(pg, new, name=pg.run_name)
            print(f'    title words: samples {samples}, generation {gen}')
        except Refusal as e:
            print(f'    REFUSED: {e}')
    print('  hub: ' + ('skipped (--no-hub)' if getattr(args, 'no_hub', False) else
                       'the buttons <S> 10-min-forced / 1-hour-forced of the installed pages' if 'hub' in stages else 'not run'))
