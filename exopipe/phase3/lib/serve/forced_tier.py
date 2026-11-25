"""serve --tier snapshot --family forced-tier: the 10-minute and deep forced-photometry pages p9120 (V 10-minute), p9121 (I
10-minute), p9122 (V deep), p9123 (I deep), each serving its run Run_SNAPFORCED_<S><10min|deep>_gen<N> (forced_tier_pages.
RUN_NAME) under paths.SNAPSHOT_REVIEW_ROOT, REGISTERED on the two-port server (registered by hand 2026-10-27 08:04:50 UTC, ledger
HWJ10PC-FORCED-10MIN-DEEP-FULL-CHAIN-FITS-LOC-ASSOC-REREGISTERED-20261027). Family T (exopipe3 forced --stages tier_runs, lib/
forced/tier_pages.py) builds each run as <run>.new beside the served one; this family INSTALLS it - production's `swap` /
`swap-stage` (tier_forced_page_20261027.py:1192-1206, :1256-1277) - with the page inputs production added by hand afterwards,
then registers the four pages once and sets the hub buttons <S> 10-min-forced / 1-hour-forced.

install: unit snapshot_forced-tier_install (flow.register_pages with this module's Install): the two-port server is checked
  first, then per requested page (a page that fails a check keeps its entry and its run; the others go on):
  0. an INSTALL RECORD beside the run (serve.FORCED_TIER_INSTALL_RECORD: an earlier serve installed the run and did not
     register it) - the page RESUMES from the recorded step (POST_SWAP_STEPS) without a <run>.new: the kept run's judgements/
     copied again, the installed journal re-tested, then what its entry needs on the installed run (its files, the title words,
     the App probe, the exclusion gate) and the registration below;
  otherwise, first what only reads:
  1. the new build is ready: <root>/<run>.new exists and family T's record T_runs (under forced_tier_pages.SITE_ROOT, or the
     --out of a run whose review root is <out>/forced_snapshot.REVIEW_SUBDIR_UNDER_OUT) is done, names this review root, and
     wrote <run>.new/queue_<S>.txt (class forced_tier_run); that content test is run again now on the .new tree, with the
     record's site (every card served once as a byte copy, the radio positions, queue = catalogue = association);
  2. the tree holds what the page reads (serve.FORCED_TIER_REQUIRED);
  3. no earlier install of the run is pending (<root>/serve.FORCED_TIER_STAGE exists: an interrupted install, refused until it
     is looked at; checked first, before readiness) and the kept name serve.FORCED_TIER_KEEP is free;
  4. the serve-time exclusion gate on the new tree (lib/serve/exclusion_gate.scan_forced_tier_run, every CSV keyed per night,
     read now; gate key 'snapshot-forced-tier': refusing);
  then the new tree is STAGED - renamed to <root>/serve.FORCED_TIER_STAGE/<run>, so it carries the run's own name, which the
  page-input tools and the page read (scintillation_index.forced_run) - and on it:
  5. the title and hub words (G26: the run's queue header names its samples, Stokes and tier; the run name its generation);
  6. the expected index sources (the server's own App on the staged queue and catalogue: servers.app_probe);
  7. THE PAGE INPUTS (page_inputs; production's interim pushes, ledger PUSH-20261112-FORCED-PAGES-FILTER-FILES and
     PREAMBLE-20261115-TESS-COVERAGE-INTERIM-SERVED-RUNS), each refusing, then their content tests;
  8. the exclusion gate again, on every file the page will serve;
  9. the FIRST JUDGEMENTS SYNC (production's swap-stage, tier_forced_page_20261027.py:1269-1270: the served run's judgements/
     into the staged run, rsync -a --update) and the content test of the synced journal (class judgement_journal).
  Any failure from the staging on - an OSError included - puts the staged tree back to <run>.new, removes the stage directory
  and refuses this page only (nothing installed; the page inputs written into it are rebuilt by the next install).
  Then THE REGISTRATION of all the pages that passed, inside the registry lock: its own checks first (the server's loader on
  the registration, the server check: a refusal there leaves every served run as it was and every new build back as
  <run>.new), then per page THE SWAP (the served run renamed serve.FORCED_TIER_KEEP, the staged run renamed into its place; a
  failed rename puts the served run back and refuses the page; a first install - no served run - only renames). From here the
  run is INSTALLED: its install record is written and the stage directory removed; then the SECOND JUDGEMENTS SYNC (the kept
  run's judgements/ again: a label written in between) and the journal re-tested; then ONE register.write of the pages, each
  with --sibling links to its three sibling pages (ForcedTierHook.siblings) and the title in serve.FORCED_TIER_TITLE words
  (from the staged build, plan.title_from) with the served time; then the install records are removed. A failure after the
  swap (the second sync, the re-test, the write) never refuses an installed run: the record keeps the step and its cause, the
  page's unit ends 'installed, not registered: <step>' (flow.InstalledNotRegistered, status error) and the next serve resumes
  from it (0). The kept run stays. One unit per registered page waits on its build and records it (class served_page).
hub: the buttons <S> 10-min-forced / 1-hour-forced (plan.hub_entry, serve.FORCED_TIER_HUB_GENERATION).
verify: the served_page records re-tested.
A requested page without a <run>.new or an install record has nothing to install: NO-INPUT (nothing registered for it; no done
record).
Parity (--out outside the production roots): the pages install into <out>/forced_snapshot.REVIEW_SUBDIR_UNDER_OUT - where family
T writes under the same --out - and register in <out>/two_port; the tool caches under serve.SERVE_PARITY_CACHE.
"""
import json
import os
import re
import subprocess

from exopipe.phase3.core.record import Refusal, atomic_write_json, utcnow
from exopipe.phase3.lib.serve import common, exclusion_gate, flow, register, servers
from exopipe.phase3.lib.serve.forced import ForcedHook, page_output, page_tests

STAGES = ('install', 'hub', 'verify')
RECORD_UNIT = 'T_runs'            # lib/forced/tier_pages.UNITS['tier_runs']
# the steps after the swap, in order: the install record's 'step' is the first one not done
POST_SWAP_STEPS = ('second judgements sync', 'journal re-test', 'register')
NIR = flow.INSTALLED_NOT_REGISTERED


def _CFG():
    from exopipe.phase3.config import CFG
    return CFG


# ---- the family's words: its PLAN_HOOK (plan.family_hook) ----------------------------------------------------------------
def forced_tier_words(page, D, name=None):
    """(samples text, generation number) of a 10-minute / deep forced page serving run directory ``D`` (named ``name``, default
    its own name: a new build <run>.new is read before it is installed): the samples from the run's own queue header
    (serve.FORCED_TIER_QUEUE_HEADER, tier_forced_page_20261027.py samples_short()), the generation from its name
    forced_tier_pages.RUN_NAME (..._gen<N>). G26 refuses a run of another name, a queue without that header, or a header naming
    another Stokes or tier than the page's - the title and the hub button name what the run holds."""
    C = _CFG()
    name = name or os.path.basename(os.path.normpath(D))
    if name != C.forced_tier_pages.RUN_NAME.format(S=page.stokes, tier=page.kind):
        raise Refusal(f'G26 forced page {page.key}: run {name!r} is not forced_tier_pages.RUN_NAME for Stokes {page.stokes}, '
                      f'{page.kind}')
    m = re.search(r'_gen(\d+)$', name)
    q = page.at('queue', D)
    try:
        with open(q) as f:
            first = f.readline().rstrip('\n')
    except OSError as e:
        raise Refusal(f'G26 forced page {page.key}: no queue {q} ({e})')
    h = re.match(C.serve.FORCED_TIER_QUEUE_HEADER, first)
    if not (m and h and h['S'] == page.stokes and h['tier'] == page.kind):
        raise Refusal(f'G26 forced page {page.key}: {q} begins {first[:120]!r}, not the family T queue header of Stokes '
                      f'{page.stokes}, {page.kind} tier (serve.FORCED_TIER_QUEUE_HEADER)')
    return h['samples'], m.group(1)


class ForcedTierHook(ForcedHook):
    """PLAN_HOOK of the 10-minute and deep forced pages: the snapshot forced pages' argv (ForcedHook; the live entries
    p9120-p9123 the same) with the sibling labels serve.FORCED_TIER_SIBLING_T, the title serve.FORCED_TIER_TITLE and hub words
    serve.FORCED_TIER_HUB_GENERATION from the run's own queue header and name (forced_tier_words; a staged build's before its
    swap, plan.title_from), environment serve.SERVER_ENV['snapshot-forced-tier']."""
    ENV_KEY = 'snapshot-forced-tier'

    def sibling_kind(self, o):
        return _CFG().serve.FORCED_TIER_SIBLING_T[o.kind]

    def title(self, plan, page, utc, root=None):
        S = _CFG().serve
        D = root or page.root
        samples, _gen = forced_tier_words(page, plan.title_from.get(page.key) or D)
        return S.FORCED_TIER_TITLE.format(S=page.stokes, what=S.FORCED_TIER_WHAT[page.kind], samples=samples, utc=utc,
                                          run_name=os.path.basename(os.path.normpath(D)))

    def hub_words(self, plan, page, root=None):
        S = _CFG().serve
        D = root or plan.registered_run(page) or page.root
        samples, g = forced_tier_words(page, plan.title_from.get(page.key) or D)
        return (os.path.basename(os.path.normpath(D)),
                S.FORCED_TIER_HUB_GENERATION.format(S=page.stokes, kind=page.kind, samples=samples, gen=g))


PLAN_HOOK = ForcedTierHook()


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


# ---- the install record (serve.FORCED_TIER_INSTALL_RECORD) -----------------------------------------------------------------
def install_record_path(pg):
    return os.path.join(install_root(pg), _CFG().serve.FORCED_TIER_INSTALL_RECORD.format(run=pg.run_name))


def read_install_record(pg):
    """The install record of ``pg`` (its run installed and not yet registered), or None. One that cannot be read ends the page
    installed and not registered (it is looked at, never overwritten)."""
    p = install_record_path(pg)
    if not os.path.lexists(p):
        return None
    try:
        with open(p) as f:
            rec = json.load(f)
        if not isinstance(rec, dict):
            raise ValueError('not a JSON object')
    except (OSError, ValueError) as e:
        raise Refusal(f'{NIR}install record ({p} cannot be read: {e}) - look at it; it is not overwritten')
    return rec


def _write_record(pg, rec):
    rec['updated'] = utcnow()
    atomic_write_json(install_record_path(pg), rec, durable=True)


def _set_step(pg, rec, step):
    rec.update(step=step, failed=None)
    _write_record(pg, rec)


def _not_registered(pg, rec, step, e):
    """``step`` and the failure ``e`` written into the install record of ``pg``; returns the page's 'installed, not registered'
    reason."""
    cause = str(e) if isinstance(e, Refusal) else f'{type(e).__name__}: {e}'
    rec.update(step=step, failed=cause[:2000])
    p = install_record_path(pg)
    try:
        _write_record(pg, rec)
        where = f'its install record {p} names the step and the cause: the next serve resumes from it'
    except OSError as we:
        sd = stage_dir(pg)
        where = (f'its install record {p} could NOT be written ({we}); ' +
                 (f'{sd} stands: the next install refuses until it is looked at' if os.path.lexists(sd) else
                  'nothing else marks the install - look at it'))
    return (f'{NIR}{step} ({cause[:600]}) - {pg.root} is installed (the previous run kept as '
            f'{rec.get("kept") or "none: a first install"}); {where}')


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


def _gate(u, plan, pg, D, when, after='nothing installed'):
    scan = exclusion_gate.scan_forced_tier_run(D)
    refuses = exclusion_gate.refuses('snapshot', 'forced-tier')
    u.note(f'page {pg.key}: G17 serve-time exclusion gate {when} ({"refusing" if refuses else "report only"}): '
           f'{scan.summary()}')
    if scan.hits and refuses:
        raise Refusal(f'G17 serve-time exclusion gate {when}: {scan.summary()} - {after}')


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


def journal_test(u, pg, run_dir, when):
    """The content test of the run's judgement journal judgements/judgements.jsonl (class judgement_journal), when it has one; a
    failure refuses the page (forced.page_output). Returns the journal's path or None."""
    j = os.path.join(run_dir, 'judgements', 'judgements.jsonl')
    if not os.path.exists(j):
        return None
    page_output(u, pg, j, 'judgement_journal', what=f'judgement journal {j} ({when})')
    return j


def unstage(pg, D):
    """Before the swap: the staged tree back to <run>.new and the stage directory removed (the page inputs written into it are
    rebuilt by the next install). Never raises; returns what it did."""
    new, sd = new_tree(pg), os.path.dirname(D)
    out = []
    if os.path.isdir(D):
        try:
            os.rename(D, new)
            out.append(f'the new build put back as {new}')
        except OSError as e:
            return (f'the new build could NOT be put back ({e}): it is at {D} and {sd} stays (the next install refuses until it '
                    f'is looked at)')
    try:
        os.rmdir(sd)
        out.append(f'{sd} removed')
    except FileNotFoundError:
        pass
    except OSError as e:
        out.append(f'{sd} could not be removed ({e})')
    return '; '.join(out) or 'nothing was staged'


class _Stuck(Refusal):
    """A swap whose put-back failed: the served run is at its kept name; nothing is moved further (the staged tree and the stage
    directory stay, so the next install refuses until it is looked at)."""


def swap(u, pg, D, keep):
    """THE SWAP (production's swap-stage renames): the served run renamed ``keep``, the staged run ``D`` renamed into its place;
    a first install (no served run) only renames ``D``. Returns the kept run (None for a first install). A failed rename puts
    the served run back and refuses (Refusal: nothing installed)."""
    live = os.path.abspath(pg.root)
    if not os.path.lexists(live):
        try:
            os.rename(D, live)
        except OSError as e:
            raise Refusal(f'the install of {D} as {live} failed ({e}) - no served run') from e
        u.note(f'page {pg.key}: INSTALLED {live} (first install: no served run)')
        return None
    try:
        os.rename(live, keep)
    except OSError as e:
        raise Refusal(f'renaming the served run {live} to {keep} failed ({e}) - the served run stays') from e
    try:
        os.rename(D, live)
    except OSError as e:
        try:
            os.rename(keep, live)
        except OSError as e2:
            raise _Stuck(f'the install of {D} as {live} failed ({e}) and the served run could NOT be put back ({e2}): it is at '
                         f'{keep}; {D} and its stage directory stay - look at them') from e
        raise Refusal(f'the install of {D} as {live} failed ({e}) - the served run put back') from e
    u.note(f'page {pg.key}: INSTALLED {live} (the previous run kept as {keep})')
    return keep


def after_swap(u, pg, rec, when='after the swap'):
    """The steps after the swap but the registration (POST_SWAP_STEPS), from the record's step, each written into the install
    record once done: the kept run's judgements/ copied into the installed run again (a label written since the first sync)
    and the installed journal re-tested."""
    live = os.path.abspath(pg.root)
    if rec['step'] == POST_SWAP_STEPS[0]:
        kept = rec.get('kept')
        if not kept or not os.path.isdir(kept):
            raise Refusal(f'the kept run {kept!r} is not a directory - the judgements written since the first sync are there')
        n = _sync_judgements(kept, live)
        u.note(f'page {pg.key}: second judgements sync of the kept run {kept} into the installed run ({when}): {n}')
        _set_step(pg, rec, POST_SWAP_STEPS[1])
    if rec['step'] == POST_SWAP_STEPS[1]:
        journal_test(u, pg, live, f'the installed run after the second judgements sync, {when}')
        _set_step(pg, rec, POST_SWAP_STEPS[2])


def resume(u, plan, pg, rec, inst):
    """A page an earlier serve installed and did not register (its install record ``rec``): the steps after the swap from the
    recorded one (after_swap), then what its entry needs on the installed run - its files, the title words, the App probe, the
    exclusion gate - and it goes to the registration. A failure keeps the record (its step and cause) and ends the page
    'installed, not registered' (Refusal with flow.INSTALLED_NOT_REGISTERED: never a refusal of the page)."""
    S = _CFG().serve
    live = os.path.abspath(pg.root)
    notes = [f'page {pg.key}: RESUMING the install recorded in {install_record_path(pg)} (installed {rec.get("installed")}, '
             f'previous run kept as {rec.get("kept")}, step {rec.get("step")!r}'
             + (f', stopped by: {str(rec["failed"])[:300]}' if rec.get('failed') else '') + ')']
    if os.path.isdir(new_tree(pg)):
        notes.append(f'page {pg.key}: {new_tree(pg)} waits for the next install (this serve registers the installed run)')
    try:
        if os.path.realpath(rec.get('run') or '') != os.path.realpath(live) or not os.path.isdir(live):
            raise Refusal(f'the record names the run {rec.get("run")!r}; {live} is '
                          f'{"a directory" if os.path.isdir(live) else "absent"} - look at it')
        if rec.get('step') not in POST_SWAP_STEPS:
            raise Refusal(f'the record names no step of {POST_SWAP_STEPS}: {rec.get("step")!r} - look at it')
        after_swap(u, pg, rec, when='resumed')
        req = list(S.FORCED_TIER_INPUTS_REQUIRED) + (['frame_times.csv'] if pg.kind == '10min' else [])
        miss = missing_files(pg, live) + [f for f in req if not os.path.exists(os.path.join(live, f))]
        if miss:
            raise Refusal(f'{live} lacks {", ".join(miss)}')
        forced_tier_words(pg, live)                    # G26
        exp = servers.app_probe(plan, pg, live, unit=u)
        _gate(u, plan, pg, live, 'on the installed run (resumed)', after='not registered')
    except (Refusal, OSError) as e:
        raise Refusal(_not_registered(pg, rec, rec.get('step') if rec.get('step') in POST_SWAP_STEPS else 'install record', e)) \
            from e
    inst.installed[pg.key] = rec
    return live, exp, notes


def check_page(u, plan, pg, inst):
    """Checks 0-9 of ``pg`` (module docstring) inside the registration unit ``u``: the page resumed from its install record, or
    its new build staged (``inst.staged``) with its first judgements sync done. Returns (served run, expected sources, notes);
    raises Refusal (the page keeps its entry and its run) - 'NO-INPUT: ...' when there is no new build, or
    flow.INSTALLED_NOT_REGISTERED + ... when a resumed page fails."""
    rec = read_install_record(pg)
    if rec is not None:
        return resume(u, plan, pg, rec, inst)
    notes = []
    sd, keep = stage_dir(pg), keep_path(pg, inst.keep_utc)
    if os.path.lexists(sd):              # first: whatever else stands, an install that did not finish is looked at
        raise Refusal(f'{sd} exists: an earlier install of {pg.run_name} did not finish - look at it (its run back to '
                      f'{os.path.basename(new_tree(pg))}, or installed by hand) before installing again')
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
    if os.path.lexists(keep):
        raise Refusal(f'the kept name {keep} exists - nothing installed')
    _gate(u, plan, pg, new, 'on the new build')
    D = os.path.join(sd, pg.run_name)
    try:
        os.mkdir(sd)
    except OSError as e:
        raise Refusal(f'staging {new}: creating {sd} failed ({e}) - nothing installed') from e
    try:
        os.rename(new, D)
    except OSError as e:
        raise Refusal(f'staging {new} as {D} failed ({e}); {unstage(pg, D)} - nothing installed') from e
    try:
        forced_tier_words(pg, D)                      # G26: the title and hub words from the run itself
        exp = servers.app_probe(plan, pg, D, unit=u)
        page_inputs(u, plan, pg, D)
        _gate(u, plan, pg, D, 'on the staged run after its page inputs')
        live = os.path.abspath(pg.root)
        n = _sync_judgements(live, D) if os.path.lexists(live) else 'none (a first install: no served run)'
        notes.append(f'page {pg.key}: first judgements sync of the served run into the staged run: {n}')
        journal_test(u, pg, D, 'the staged run after the first judgements sync')
    except (Refusal, OSError) as e:
        why = str(e) if isinstance(e, Refusal) else f'{type(e).__name__} on the staged run: {e}'
        raise Refusal(f'{why}; {unstage(pg, D)} - nothing installed') from e
    except BaseException:
        unstage(pg, D)
        raise
    inst.staged[pg.key] = D
    plan.title_from[pg.key] = D                        # its entry's title words: from the staged build, before its swap
    return os.path.abspath(pg.root), exp, notes


class Install:
    """flow.register_pages' ``install`` hook of this family. A page is 'staged' (its <run>.install/<run>, put back on any
    failure before its swap) or 'installed' (swapped in this serve, or resumed from its install record, which stands until the
    registration is written)."""

    def __init__(self, plan, keep_utc):
        self.plan, self.keep_utc = plan, keep_utc
        self.pages = {}         # page key -> page
        self.staged = {}        # page key -> its staged run directory
        self.installed = {}     # page key -> its install record (to register)
        self.dropped = {}       # page key -> its 'installed, not registered' reason (a step after its swap failed)

    def unstage(self, u, pg, why):
        D = self.staged.pop(pg.key)
        self.plan.title_from.pop(pg.key, None)
        back = unstage(pg, D)
        u.note(f'page {pg.key}, before its swap ({why}): {back}')
        return back

    def drop(self, u, pg, e):
        """A page whose entry refused after its checks (G26): a staged one is put back; a resumed one stays installed."""
        if pg.key in self.staged:
            why = f'{e}; {self.unstage(u, pg, "its entry refused")} - nothing installed'
            u.note(f'page {pg.key} REFUSED: {why}')
            return why
        why = _not_registered(pg, self.installed.pop(pg.key), POST_SWAP_STEPS[2], e)
        u.note(f'page {pg.key}: {why}')
        return why

    def swap(self, u, items):
        """Inside the registry lock, after the registration's own checks: per staged page the swap, its install record, the
        stage directory removed and the steps after the swap. Returns (items still to register, refused {key: reason})."""
        keep, refused = [], {}
        for it in items:
            pg = it[0]
            if pg.key not in self.staged:                # resumed: installed by an earlier serve
                keep.append(it)
                continue
            D = self.staged[pg.key]
            try:
                kept = swap(u, pg, D, keep_path(pg, self.keep_utc))
            except _Stuck as e:
                self.staged.pop(pg.key)
                refused[pg.key] = str(e)
                u.note(f'page {pg.key} REFUSED: {e}')
                continue
            except Refusal as e:
                refused[pg.key] = f'{e}; {self.unstage(u, pg, "its swap failed")} - nothing installed'
                u.note(f'page {pg.key} REFUSED: {refused[pg.key]}')
                continue
            self.staged.pop(pg.key)
            self.plan.title_from.pop(pg.key, None)
            rec = dict(run=os.path.abspath(pg.root), installed=utcnow(), kept=kept, unit=u.name, failed=None,
                       step=POST_SWAP_STEPS[0] if kept else POST_SWAP_STEPS[1])
            u.meta.setdefault('installed', {})[pg.key] = dict(run=rec['run'], kept=kept, record=install_record_path(pg))
            try:
                _write_record(pg, rec)
                os.rmdir(stage_dir(pg))
                after_swap(u, pg, rec)
            except (Refusal, OSError) as e:
                self.dropped[pg.key] = refused[pg.key] = _not_registered(pg, rec, rec['step'], e)
                u.note(f'page {pg.key}: {refused[pg.key]}')
                continue
            self.installed[pg.key] = rec
            keep.append(it)
        return keep, refused

    def failed(self, u, exc):
        """A failure from prepare on (flow): every page still staged is put back (nothing of it swapped); returns {page key:
        reason} of the pages installed and not registered (their records name the step 'register' and the cause)."""
        for key in list(self.staged):
            self.unstage(u, self.pages[key], f'{type(exc).__name__}: {str(exc)[:200]}')
        out = dict(self.dropped)
        for key, rec in list(self.installed.items()):
            out[key] = _not_registered(self.pages[key], rec, POST_SWAP_STEPS[2], exc)
        self.installed.clear()
        return out

    def registered(self, u, keys):
        """After the write: the install records of the registered pages removed."""
        for key in keys:
            if self.installed.pop(key, None) is None:
                continue
            p = install_record_path(self.pages[key])
            try:
                os.remove(p)
                u.note(f'page {key}: registered - its install record {p} removed')
            except OSError as e:
                u.note(f'page {key}: registered; its install record {p} could not be removed ({e}): the next serve registers '
                       f'it again')

    def pending(self, pg):
        """The 'installed, not registered' reason of a page an earlier serve installed (its install record), or None."""
        try:
            rec = read_install_record(pg)
        except Refusal as e:
            return str(e)
        if rec is None:
            return None
        return (f'{NIR}{rec.get("step")} (recorded {rec.get("updated")}: {str(rec.get("failed") or "interrupted")[:300]}) - '
                f'{pg.root} is installed; its install record {install_record_path(pg)} stands')


def install_stage(run, plan, state, utc):
    inst = Install(plan, common.stamp() + 'Z')     # the kept run's name: serve.FORCED_TIER_KEEP

    def prepare(u):
        flow.excl_note(u, 'snapshot')
        passed, refused = [], {}
        for pg in plan.pages:
            inst.pages[pg.key] = pg
            try:
                root, exp, notes = check_page(u, plan, pg, inst)
            except (Refusal, OSError) as e:          # a page's own failure: this page only
                why = str(e) if isinstance(e, Refusal) else f'{type(e).__name__}: {e} - nothing installed'
                refused[pg.key] = why
                u.note(f'page {pg.key}: {why}' if why.startswith(NIR) else f'page {pg.key} REFUSED: {why}')
                continue
            for n in notes:
                u.note(n)
            passed.append((pg, root, exp))
        plan.forced_after = {pg.key: root for pg, root, _ in passed}
        items = []
        for pg, root, exp in passed:
            try:
                items.append((pg, root, register.page_entry(plan, pg, utc, root), exp, []))
            except Refusal as e:
                refused[pg.key] = inst.drop(u, pg, e)
        for pg, _r, e, _x, _i in items:
            u.note(f'FORCED-TIER-INSTALL {e["id"]} {pg.stokes}/{pg.kind} {pg.run_name} siblings {e["argv"].count("--sibling")}'
                   + (' (resumed)' if pg.key in inst.installed else ''))
        return items, refused
    state['served'].update(flow.register_pages(run, plan, prepare, f'snapshot_{plan.family}_install', stage='install',
                                               inc_log=False, install=inst))


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
    for pg in plan.pages:
        new = new_tree(pg)
        ok, text, _site = ready(pg)
        print(f'  {pg.key}: served run {pg.root} ({"present" if os.path.isdir(pg.root) else "absent"}); new build {new}: '
              f'{"ready (" + text + ")" if ok else text}; registry serves {plan.registered_run(pg) or "-"}')
        pend = Install(plan, '').pending(pg)
        if pend:
            print(f'    {pend}: this serve RESUMES it (no <run>.new needed)')
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
