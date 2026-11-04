"""`exopipe3 forced --tier snapshot --stages wide_jobs,wide_search,wide_test,wide_publish` (family W) and `--stages activity`
(family N): the two inputs of the astrometry figure v2 that analysis scripts made - the forced targets' wide-area burst test
(<run>/forced_wide_v3.csv, astrometry_assoc.V3_MAP) and the night-activity table (astrometry_assoc.night_act). Spec
docs/specs/forced_wide.md. Campaign stages (no per-hour form), named only (never a default), one family per invocation.

Family W. --out = the work root W (default forced_wide.WORK_ROOT); records W/_records/forced/W_*.json.
  wide_jobs     unit W_jobs: the jobs of the four forced runs (forced_wide.RUNS under --wide-runs, default the served runs in
                paths.SNAPSHOT_REVIEW_ROOT) -> W/jobs.json (forced_wide_jobs) and the runs' forced events -> W/events.csv
                (forced_wide_events). --wide-jobs FILE[,FILE] adopts job lists the analysis wrote (forced_wide_jobs.py
                jobs.json, v3 jobs_ext.json) instead of selecting (a replay; recorded).
  wide_search   unit W_search_<S>_<identity> per job: the burst search of each band on the cutouts (--wide-cutout-root, default
                photometry.CUTOUT_ROOT) -> W/bursts/<S>_<identity>_<band>.npz (forced_wide_bursts), a pool of at most
                forced_wide_search.MAX_WORKERS processes (--workers), each cutout read after the recorders' I/O guard allows it.
  wide_test     unit W_test: the v3 analysis of every job (pool: every job of the Stokes) -> W/results_v3.csv
                (forced_wide_results) and W/candidates_v3.csv (forced_wide_candidates). The bursts: this root's done
                W_search units, or --wide-bursts DIR (every file content-tested as an input).
  wide_publish  unit W_publish_<run> per run: <review root>/<run>/forced_wide_v3.csv (forced_wide_published): the results of
                that (Stokes, cadence) with each named source's position (named_radec). The review root: --snap-review-root,
                else <out>/forced_snapshot.REVIEW_SUBDIR_UNDER_OUT under --out, else paths.SNAPSHOT_REVIEW_ROOT (production).
                Written as <file>.part, content-tested, then renamed over the served file (one atomic replace).
  --wide-only S:IDENTITY[,..] (IDENTITY '*' = every job of that Stokes): wide_search searches, wide_test analyzes only those
  (the test's pool stays every job; its results are recorded partial and wide_publish refuses them).
Family N. --out = night_activity.OUT_ROOT by default (where the figure reads).
  activity      unit N_activity: the listed detections of the served snapshot 10-s Stokes I view (--na-detections, default
                night_activity.DETECTIONS) -> W/night_activity_I10s.csv (night_activity_bands) and
                W/night_activity_nights_I10s.csv (night_activity_nights).
Gates (exit 2, the gate named in the record): W-FAMILY (stages of two families, a W/N stage with --epochs, --cells or a family-C
flag), W-ENV (a switch of forced_wide.ENV_FORBIDDEN set in the caller environment), W-HOST (a run or the review root node-local
to another host), W-CATDIR (the radio catalogues the in-place association module reads are not forced_assoc.RADIO_CATDIR),
W-TARGETS (build_targets refused: a listed target missing from its list), W-EVENTS (a forced event read twice with different
values), W-RECORD (no done record of the previous stage), W-WORKERS (more than MAX_WORKERS processes), W-CUTROOT (the cutout root
absent or a retired generation), W-NOCUT (a band without a cutout directory: item refusal), W-FILE (a cutout raised, FILE_ERRORS
'refuse': item refusal), W-POOL (a job's band without its done search / burst file, or a burst file failing its test), W-PARTIAL
(publishing a partial test), W-NAMED (a named source not found around its target), W-RUNDIR (the run directory absent at the
review root), W-ROOT (a production review root while paths.PRODUCTION_WRITES is False), N-INPUT (input columns missing, or rows
of another Stokes or tier than night_activity.STOKES / TIER). A failing content test refuses its unit. --dry-run prints the plan
and writes nothing. The INJECTED-RUN GUARD (core/injected.py) covers the families' flags: an injected run's --snap-review-root
must be scratch, and an ordinary run refuses a --wide-runs / --wide-jobs / --wide-cutout-root / --wide-bursts / --na-detections
source under an injected run."""
import hashlib
import json
import os
import shutil
import time

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.forced import common as K

WIDE_STAGES = ('wide_jobs', 'wide_search', 'wide_test', 'wide_publish')
ACTIVITY_STAGES = ('activity',)
STEP = 'forced'


def add_arguments(p):
    g = p.add_argument_group('snapshot tier, family W (the forced targets\' wide-area burst test: --stages wide_jobs,wide_search,'
                             'wide_test,wide_publish) and family N (the night-activity table: --stages activity)')
    g.add_argument('--wide-runs', metavar='DIR', help='wide_jobs: the directory holding the four forced runs '
                                                     '(forced_wide.RUN_NAME; default paths.SNAPSHOT_REVIEW_ROOT, the served runs)')
    g.add_argument('--wide-jobs', metavar='FILE[,FILE]', help='wide_jobs: adopt job lists the analysis wrote (forced_wide_jobs.py '
                                                             'jobs.json, then v3 jobs_ext.json) instead of selecting them (a replay)')
    g.add_argument('--wide-cutout-root', metavar='DIR', help='wide_search: the cutout generation searched (default '
                                                             'photometry.CUTOUT_ROOT, the served one)')
    g.add_argument('--wide-bursts', metavar='DIR', help='wide_test: read every job\'s burst files from DIR instead of this root\'s '
                                                        'wide_search records (each file content-tested as an input)')
    g.add_argument('--wide-only', metavar='S:IDENTITY[,..]', help='wide_search / wide_test: only these jobs (IDENTITY * = every job '
                                                                 'of that Stokes); a test of a subset is partial, never published')
    g.add_argument('--na-detections', metavar='FILE', help='activity: the listed detections of the served snapshot 10-s Stokes I '
                                                          'view (default night_activity.DETECTIONS)')


def family_of(stage):
    if stage in WIDE_STAGES:
        return 'W'
    if stage in ACTIVITY_STAGES:
        return 'N'
    return 'C'


def family(args):
    """('C' | 'W' | 'N', stages) of a snapshot invocation; refuses stages of two families (W-FAMILY)."""
    stages = K.selected_stages(args, 'snapshot')
    fams = sorted({family_of(s) for s in stages})
    if len(fams) > 1:
        raise Refusal('W-FAMILY one family per invocation: %s are stages of families %s (event panels C, wide-area test W, '
                      'night activity N write into different roots)' % (','.join(stages), '/'.join(fams)))
    return fams[0], stages


def default_out(args, fam):
    c = K.cfg()
    return c.forced_wide.WORK_ROOT if fam == 'W' else c.night_activity.OUT_ROOT


def run(args, run, fam, stages):
    gate_args(args, fam)
    if fam == 'N':
        stage_activity(args, run)
        return None
    c = WCtx(args, run)
    K.plan('family W (wide-area burst test): stages %s; work root %s; runs %s; cutouts %s; review root %s%s'
           % (','.join(stages), c.W, c.runs_root, c.cutout_root, c.review_root,
              '; only %s' % ','.join('%s:%s' % x for x in sorted(c.only)) if c.only else ''))
    for st in stages:
        STAGE_FN[st](c)
    return None


def gate_args(args, fam):
    """W-FAMILY: the W / N stages are campaign stages over whole runs - no per-hour form, no cells, no family-C run flags."""
    bad = []
    if getattr(args, 'epochs', None):
        bad.append('--epochs (no per-hour form: the test pools every night)')
    if getattr(args, 'cells', None):
        bad.append('--cells')
    for flag, attr in (('--snap-name', 'snap_name'), ('--snap-cadence', 'snap_cadence'), ('--stokes', 'stokes')):
        if getattr(args, attr, None):
            bad.append(flag + (' (use --wide-only S:*)' if attr == 'stokes' else ''))
    if fam == 'N':
        for flag, attr in (('--wide-runs', 'wide_runs'), ('--wide-jobs', 'wide_jobs'), ('--wide-cutout-root', 'wide_cutout_root'),
                           ('--wide-bursts', 'wide_bursts'), ('--wide-only', 'wide_only')):
            if getattr(args, attr, None):
                bad.append(flag)
    elif getattr(args, 'na_detections', None):
        bad.append('--na-detections')
    if bad:
        raise Refusal('W-FAMILY %s: not a flag of family %s' % (', '.join(bad), fam))


def gate_env():
    """W-ENV: switches the in-place modules (or the analysis scripts) would read from the caller environment."""
    bad = [v for v in K.cfg().forced_wide.ENV_FORBIDDEN if v in os.environ]
    if bad:
        raise Refusal('W-ENV the caller environment sets %s: the in-place modules would read it (the stage passes its roots '
                      'as arguments); unset it' % ', '.join('%s=%s' % (v, os.environ[v]) for v in bad))


def parse_only(text):
    out = set()
    for x in (text or '').split(','):
        x = x.strip()
        if not x:
            continue
        if ':' not in x or x.split(':', 1)[0] not in ('I', 'V'):
            raise ValueError('--wide-only takes S:IDENTITY items (S = I or V; IDENTITY * = every job of S), got %r' % x)
        out.add(tuple(x.split(':', 1)))
    return out


def selected(job, only):
    return not only or (job['stokes'], job['identity']) in only or (job['stokes'], '*') in only


def _sha1_file(p):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


class WCtx:
    def __init__(self, args, run):
        from exopipe.phase3.lib.forced import wide_jobs as WJ
        self.args, self.run = args, run
        cfg = K.cfg()
        self.cfg = cfg
        self.w, self.ws, self.wt = cfg.forced_wide, cfg.forced_wide_search, cfg.forced_wide_test
        self.dry = run.dry_run
        self.W = os.path.abspath(run.out_root)
        self.runs_root = os.path.abspath(args.wide_runs or cfg.paths.SNAPSHOT_REVIEW_ROOT)
        self.runs = WJ.run_dirs(self.runs_root)
        self.cutout_root = os.path.abspath(args.wide_cutout_root or cfg.photometry.CUTOUT_ROOT)
        if args.out:
            rr = args.snap_review_root or os.path.join(self.W, cfg.forced_snapshot.REVIEW_SUBDIR_UNDER_OUT)
        else:
            rr = args.snap_review_root or cfg.paths.SNAPSHOT_REVIEW_ROOT
        self.review_root = os.path.abspath(rr)
        self.jobs_path = os.path.join(self.W, self.w.JOBS)
        self.events_path = os.path.join(self.W, self.w.EVENTS)
        self.bursts = os.path.join(self.W, self.w.BURSTS_DIR)
        self.results = os.path.join(self.W, self.w.RESULTS)
        self.candidates = os.path.join(self.W, self.w.CANDIDATES)
        self.only = parse_only(args.wide_only)

    def jobs_record(self):
        """The done W_jobs record and its two outputs (refuses W-RECORD)."""
        try:
            rec = K.load_done(self.W, STEP, 'W_jobs', 'the jobs (wide_jobs)')
        except Refusal as e:
            raise Refusal('W-RECORD %s' % e)
        jp = K.outputs_of(rec, 'forced_wide_jobs')
        ep = K.outputs_of(rec, 'forced_wide_events')
        if not jp or not ep:
            raise Refusal('W-RECORD the W_jobs record lists no jobs / events output')
        return rec, jp[0], ep[0]


def search_unit(job):
    return 'W_search_%s_%s' % (job['stokes'], job['identity'])


# ---------------------------------------------------------------------------------------------------------------------
# wide_jobs
# ---------------------------------------------------------------------------------------------------------------------

def stage_jobs(c):
    from exopipe.phase3.lib.forced import wide_jobs as WJ
    name = 'W_jobs'
    adopt = [p for p in (c.args.wide_jobs or '').split(',') if p]
    K.plan('[wide_jobs] %s; events of %s (forced_wide.EVENT_ORDER %s; merged identities forced_wide.TARGET_MEMBER %s) -> %s, %s'
           % ('ADOPT the job lists %s' % ','.join(adopt) if adopt else 'select the jobs', ', '.join(r[3] for r in c.runs),
              c.w.EVENT_ORDER, c.w.TARGET_MEMBER, c.jobs_path, c.events_path))
    if c.dry:
        for _lb, st, _cad, d in c.runs:
            h = K.host_of(d)
            if h and h != K.hostname():
                state = 'node-local on %s (this is %s): W-HOST refuses' % (h, K.hostname())
            else:
                state = 'found' if os.path.isdir(d) else 'NOT FOUND (refused)'
            K.plan('[wide_jobs]   run %s: %s' % (d, state))
        return
    if K.resumed(c.run, name, None, tier='snapshot', stage='wide_jobs', family='W'):
        return
    with c.run.unit(name, tier='snapshot', stage='wide_jobs', family='W', runs=[r[3] for r in c.runs],
                    event_order=c.w.EVENT_ORDER, target_member=c.w.TARGET_MEMBER, adopted=adopt) as u:
        gate_env()
        for _lb, st, _cad, d in c.runs:
            try:
                K.require_host(d, 'forced run')
            except Refusal as e:
                raise Refusal('W-HOST %s' % e)
        fa = WJ.FA()
        if os.path.realpath(fa.CATDIR) != os.path.realpath(c.cfg.forced_assoc.RADIO_CATDIR):
            u.refuse('W-CATDIR snapshot_forced_assoc reads the radio catalogues from %s, not forced_assoc.RADIO_CATDIR %s'
                     % (fa.CATDIR, c.cfg.forced_assoc.RADIO_CATDIR))
        for _lb, st, _cad, d in c.runs:
            for p in WJ.run_inputs(d, st):
                u.add_input(p)
        for fn in sorted(fa.FILES.values()):
            u.add_input(os.path.join(fa.CATDIR, fn))
        for fn in c.cfg.forced.SAMPLE_FILES.values():
            u.add_input(os.path.join(c.cfg.paths.CLV15_ROOT, fn))
        bad = WJ.event_conflicts(c.runs)
        if bad:
            u.refuse('W-EVENTS %d forced event(s) read twice with different values (build tree vs served tree), e.g. %s'
                     % (len(bad), bad[:3]))
        notes = []
        if adopt:
            for p in adopt:
                u.add_input(p)
            base, ext = WJ.adopt_jobs(adopt)
            amb = []
        else:
            try:
                base, ext, amb = WJ.build_jobs(c.runs, log=notes.append)
            except SystemExit as e:              # build_targets: 'REFUSING: ... is not in the current ... list'
                u.refuse('W-TARGETS snapshot_forced_assoc.build_targets: %s' % e)
        for n in notes:
            u.note(n)
        if not base and not ext:
            raise NoInput('no forced target qualifies for the wide-area test in %s' % ', '.join(r[3] for r in c.runs))
        events = WJ.event_table(c.runs)
        u.plan(c.jobs_path, 'forced_wide_jobs')
        u.plan(c.events_path, 'forced_wide_events')
        os.makedirs(c.W, exist_ok=True)
        WJ.write_jobs(c.jobs_path, base, ext, c.runs, adopted=[{'path': p, 'sha1': _sha1_file(p)} for p in adopt])
        WJ.write_events(c.events_path, events)
        u.meta.update(n_base=len(base), n_ext=len(ext), n_ambiguous_rows=len(amb), n_events=len(events),
                      n_band_units=sum(len(j['bands']) for j in base + ext),
                      jobs_by_stokes={s: sum(1 for j in base + ext if j['stokes'] == s) for s in 'IV'})
        u.add_output(c.jobs_path, 'forced_wide_jobs')
        u.add_output(c.events_path, 'forced_wide_events', jobs=c.jobs_path)
        print('W_jobs: %d base + %d extension jobs, %d band units, %d events' % (len(base), len(ext), u.meta['n_band_units'],
                                                                                len(events)), flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# wide_search
# ---------------------------------------------------------------------------------------------------------------------

def gate_cutout_root(c):
    retired = {os.path.realpath(r) for r in c.cfg.photometry.RETIRED_CUTOUT_ROOTS}
    if os.path.realpath(c.cutout_root) in retired:
        raise Refusal('W-CUTROOT %s is a retired cutout generation (photometry.RETIRED_CUTOUT_ROOTS): the forced photometry '
                      'measured photometry.CUTOUT_ROOT' % c.cutout_root)
    if not os.path.isdir(c.cutout_root):
        raise Refusal('W-CUTROOT the cutout root %s is not readable on %s' % (c.cutout_root, K.hostname()))


def _jobs_planned(c):
    """The jobs a plan / dry run can list: from this root's jobs.json when it exists, else none (wide_jobs of this invocation)."""
    from exopipe.phase3.lib.forced import wide_jobs as WJ
    if os.path.exists(c.jobs_path):
        return WJ.all_jobs(WJ.load_jobs(c.jobs_path))
    return None


def stage_search(c):
    from exopipe.phase3.lib.forced import wide_jobs as WJ
    from exopipe.phase3.lib.forced import wide_search as WS
    workers = c.args.workers or c.ws.WORKERS
    K.plan('[wide_search] bursts of every job band in %s -> %s/<S>_<identity>_<band>.npz; %d processes (max %d); I/O guard %s'
           % (c.cutout_root, c.bursts, workers, c.ws.MAX_WORKERS, c.ws.IO_GUARD_LOG or 'off'))
    if c.dry:
        jobs = _jobs_planned(c)
        if jobs is None:
            K.plan('[wide_search]   the jobs come from wide_jobs (no %s yet)' % c.jobs_path)
            return
        jobs = [j for j in jobs if selected(j, c.only)]
        nf = 0
        for j in jobs:
            for b in j['bands']:
                d, files = WS.cutout_files(c.cutout_root, j, b)
                nf += len(files)
                if d is None:
                    K.plan('[wide_search]   %s %s: NO cutout directory (W-NOCUT)' % (search_unit(j), b))
        K.plan('[wide_search]   %d jobs, %d band units, %d cutout files' % (len(jobs), sum(len(j['bands']) for j in jobs), nf))
        if workers > c.ws.MAX_WORKERS:
            K.plan('[wide_search]   W-WORKERS %d processes > %d: refused' % (workers, c.ws.MAX_WORKERS))
        return
    _rec, jobs_path, _ev = c.jobs_record()
    jobs = [j for j in WJ.all_jobs(WJ.load_jobs(jobs_path)) if selected(j, c.only)]
    if not jobs:
        raise NoInput('no job selected (--wide-only %s)' % c.args.wide_only)
    if workers > c.ws.MAX_WORKERS:
        raise Refusal('W-WORKERS %d processes > forced_wide_search.MAX_WORKERS %d (Lustre reads yield to the recorders)'
                      % (workers, c.ws.MAX_WORKERS))
    if c.ws.FILE_ERRORS not in ('refuse', 'drop'):
        raise Refusal('forced_wide_search.FILE_ERRORS %r: refuse or drop' % c.ws.FILE_ERRORS)
    gate_env()
    gate_cutout_root(c)
    # the in-place modules, imported here so the records name their code (the workers inherit them; their caches are installed
    # in the workers only: wide_search.eval_mod)
    for m in ('snapshot_forced_eval', 'phase3_snapshot_search', 'phase3_tier_rules', 'phase3_snapshot_astro'):
        WJ.inplace(m)
    todo = [j for j in jobs if not K.resumed(c.run, search_unit(j), None, tier='snapshot', stage='wide_search', family='W')]
    if not todo:
        return
    stage = os.path.join(c.bursts, '_staging_%s' % c.run.run_id)
    os.makedirs(stage, exist_ok=True)
    tasks = [(j, b) for j in todo for b in j['bands']]
    print('wide_search: %d jobs, %d (Stokes, target, band) units, %d processes' % (len(todo), len(tasks), workers), flush=True)
    t0 = time.time()
    from multiprocessing import get_context
    try:
        with get_context('fork').Pool(workers, initializer=WS.pool_worker_init) as pool:
            pending = {(search_unit(j), b): pool.apply_async(WS.search_band, ((j, b, c.cutout_root,
                                                                               os.path.join(stage, WS.npz_name(j, b))),))
                       for j, b in tasks}
            for k, j in enumerate(todo, 1):
                try:
                    search_job(c, j, pending, stage, jobs_path)
                except Exception as e:  # noqa: BLE001 - the unit's record says error; the stage goes on (exit 1 at the end)
                    print('  %s: ERROR %s: %s' % (search_unit(j), type(e).__name__, str(e)[:300]), flush=True)
                print('  [%d/%d] %s (%.0f s)' % (k, len(todo), search_unit(j), time.time() - t0), flush=True)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def search_job(c, job, pending, stage, jobs_path):
    """One unit W_search_<S>_<identity>: wait for its bands' results, gate them, place the npz files (os.replace)."""
    from exopipe.phase3.lib.forced import wide_search as WS
    from exopipe.phase3.core import validate
    name = search_unit(job)
    with c.run.unit(name, tier='snapshot', stage='wide_search', family='W', stokes=job['stokes'], identity=job['identity'],
                    bands=list(job['bands']), cutout_root=c.cutout_root, file_errors=c.ws.FILE_ERRORS) as u:
        u.add_input(jobs_path)
        u.add_input(c.cfg.paths.DATA_DECISIONS)
        os.makedirs(c.bursts, exist_ok=True)
        nights, inputs, per_band = set(), set(), {}
        for b in job['bands']:
            final = os.path.join(c.bursts, WS.npz_name(job, b))
            u.plan(final, 'forced_wide_bursts')
            res = pending[(name, b)].get()
            part = res['path']
            per_band[b] = {k: res.get(k) for k in ('n_files', 'n_peaks', 'statuses', 'seconds', 'waited')}
            if res.get('missing'):
                u.refuse_item(final, 'W-NOCUT %s: no cutout directory for %s in %s' % (b, job['identity'], c.cutout_root),
                              gate='W-NOCUT')
                continue
            u.add_input(res['cutout_dir'])
            per_band[b]['listing_sha1'] = hashlib.sha1(json.dumps(res['listing']).encode()).hexdigest()[:12]
            if res['errors']:
                per_band[b]['errors'] = res['errors'][:20]
                if c.ws.FILE_ERRORS == 'refuse':
                    os.remove(part)
                    u.refuse_item(final, 'W-FILE %d cutout file(s) of %s raised (forced_wide_search.FILE_ERRORS refuse), e.g. %s'
                                  % (len(res['errors']), b, res['errors'][0]), gate='W-FILE')
                    continue
            bad = [t for t in validate.run('forced_wide_bursts', part, stokes=job['stokes'], identity=job['identity'], band=b)
                   if not t['passed']]
            if bad:
                os.remove(part)
                u.refuse_item(final, 'content test of the staged file failed: %s' % '; '.join(
                    '%s (%s)' % (t['name'], str(t['detail'])[:200]) for t in bad), gate='forced_wide_bursts')
                continue
            os.replace(part, final)
            nights.update(res['nights'])
            inputs.update(res.get('inputs') or ())
            u.add_output(final, 'forced_wide_bursts', stokes=job['stokes'], identity=job['identity'], band=b)
        for p in sorted(inputs):
            u.add_input(p)
        u.set_admitted(nights)
        u.meta['bands_detail'] = per_band


# ---------------------------------------------------------------------------------------------------------------------
# wide_test
# ---------------------------------------------------------------------------------------------------------------------

def pool_files(c, jobs):
    """{(S, identity, band): npz path} of every job band - from --wide-bursts DIR, else this root's done W_search records
    (refuses W-POOL naming what is missing)."""
    from exopipe.phase3.lib.forced import wide_search as WS
    out, missing = {}, []
    for j in jobs:
        if c.args.wide_bursts:
            for b in j['bands']:
                p = os.path.join(os.path.abspath(c.args.wide_bursts), WS.npz_name(j, b))
                if os.path.exists(p):
                    out[(j['stokes'], j['identity'], b)] = p
                else:
                    missing.append(p)
            continue
        try:
            rec = K.load_done(c.W, STEP, search_unit(j), 'the burst search of %s' % search_unit(j))
        except Refusal as e:
            missing.append(str(e))
            continue
        have = {os.path.basename(p): p for p in K.outputs_of(rec, 'forced_wide_bursts')}
        for b in j['bands']:
            p = have.get(WS.npz_name(j, b))
            if p and os.path.exists(p):
                out[(j['stokes'], j['identity'], b)] = p
            else:
                missing.append('%s %s (refused or absent in the record)' % (search_unit(j), b))
    if missing:
        raise Refusal('W-POOL %d job band(s) without a burst file: the v3 pools every job of a Stokes, e.g. %s'
                      % (len(missing), '; '.join(missing[:4])))
    dirs = {os.path.dirname(p) for p in out.values()}
    if len(dirs) != 1:
        raise Refusal('W-POOL the burst files lie in %d directories (%s): the analysis reads one' % (len(dirs), sorted(dirs)[:3]))
    return out, dirs.pop()


def stage_test(c):
    from exopipe.phase3.lib.forced import wide_jobs as WJ
    from exopipe.phase3.lib.forced import wide_v3 as V3
    from exopipe.phase3.core import validate
    name = 'W_test'
    workers = c.args.workers or c.wt.WORKERS
    K.plan('[wide_test] the v3 analysis of every job (bursts %s) -> %s, %s; %d processes%s'
           % (c.args.wide_bursts or c.bursts, c.results, c.candidates, workers,
              '; PARTIAL (--wide-only)' if c.only else ''))
    if c.dry:
        return
    if K.resumed(c.run, name, None, tier='snapshot', stage='wide_test', family='W'):
        return
    with c.run.unit(name, tier='snapshot', stage='wide_test', family='W', partial=bool(c.only),
                    only=sorted('%s:%s' % x for x in c.only), bursts=c.args.wide_bursts or c.bursts, workers=workers) as u:
        gate_env()
        _rec, jobs_path, events_path = c.jobs_record()
        u.add_input(jobs_path)
        u.add_input(events_path)
        jobs = WJ.all_jobs(WJ.load_jobs(jobs_path))
        files, bdir = pool_files(c, jobs)
        for (st, ident, b), p in sorted(files.items()):
            if c.args.wide_bursts:
                bad = [t for t in validate.run('forced_wide_bursts', p, stokes=st, identity=ident, band=b) if not t['passed']]
                if bad:
                    u.refuse('W-POOL the burst file %s fails its content test: %s' % (p, bad[0]['detail']))
            u.add_input(p)
        events = WJ.load_events(events_path)
        only = {(j['stokes'], j['identity']) for j in jobs if selected(j, c.only)} if c.only else None
        if only is not None and not only:
            raise NoInput('no job selected (--wide-only %s)' % c.args.wide_only)
        WJ.inplace('phase3_snapshot_astro')
        rows, crows = V3.analyze_all(jobs, events, bdir, workers=workers, only=only, log=u.note)
        if not rows:
            raise NoInput('no job has forced events in its bands')
        u.plan(c.results, 'forced_wide_results')
        u.plan(c.candidates, 'forced_wide_candidates')
        os.makedirs(c.W, exist_ok=True)
        V3.write_rows(c.results, rows)
        V3.write_rows(c.candidates, crows)
        counts = {}
        for r in rows:
            k = '%s %s %s' % (r['stokes'], r['tier'], r['verdict'])
            counts[k] = counts.get(k, 0) + 1
        u.meta.update(n_rows=len(rows), n_candidate_rows=len(crows), verdicts=dict(sorted(counts.items())),
                      n_jobs=len(jobs), n_analyzed=len(only) if only else len(jobs))
        u.add_output(c.results, 'forced_wide_results')
        u.add_output(c.candidates, 'forced_wide_candidates', results=c.results)
        for k, v in sorted(counts.items()):
            print('  %s: %d' % (k, v), flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# wide_publish
# ---------------------------------------------------------------------------------------------------------------------

def stage_publish(c):
    from exopipe.phase3.lib.forced import wide_jobs as WJ
    from exopipe.phase3.lib.forced import wide_v3 as V3
    from exopipe.phase3.core import validate
    K.plan('[wide_publish] %s into each run under %s: %s' % (c.w.PUBLISHED, c.review_root,
                                                           ', '.join(c.w.RUN_NAME.format(S=s, cadence=t) for s, t in c.w.RUNS)))
    if c.dry:
        h = K.host_of(c.review_root)
        K.plan('[wide_publish]   review root %s%s' % (c.review_root, (' node-local on %s (this is %s)' % (h, K.hostname()))
                                                     if h and h != K.hostname() else ''))
        return
    for st, cad in c.w.RUNS:
        rname = c.w.RUN_NAME.format(S=st, cadence=cad)
        name = 'W_publish_%s' % rname
        D = os.path.join(c.review_root, rname)
        final = os.path.join(D, c.w.PUBLISHED)
        if K.resumed(c.run, name, None, tier='snapshot', stage='wide_publish', family='W'):
            continue
        with c.run.unit(name, tier='snapshot', stage='wide_publish', family='W', stokes=st, cadence=cad, run=D) as u:
            try:
                rec = K.load_done(c.W, STEP, 'W_test', 'the v3 analysis (wide_test)')
            except Refusal as e:
                raise Refusal('W-RECORD %s' % e)
            if (rec.get('meta') or {}).get('partial'):
                u.refuse('W-PARTIAL the v3 analysis analyzed a subset (--wide-only %s): a partial test is never published'
                         % ','.join(rec['meta'].get('only') or []))
            res = K.outputs_of(rec, 'forced_wide_results')
            if not res:
                u.refuse('W-RECORD the W_test record lists no results')
            _jr, jobs_path, _ev = c.jobs_record()
            try:
                K.refuse_production_target(D, 'review run')
            except Refusal as e:
                raise Refusal('W-ROOT %s' % e)
            h = K.host_of(D)
            if h and h != K.hostname():
                u.refuse('W-HOST the run %s is node-local on %s, this is %s: run wide_publish there' % (D, h, K.hostname()))
            if not os.path.isdir(D):
                u.refuse('W-RUNDIR no run %s (publish_v3_to_runs.py refused the same)' % D)
            u.add_input(res[0])
            u.add_input(jobs_path)
            rows, cols = V3.read_rows(res[0])
            pos = {(j['stokes'], j['identity']): j['T'] for j in WJ.all_jobs(WJ.load_jobs(jobs_path))}
            try:
                rows = V3.named_radec(rows, pos)
            except V3.NamedNotFound as e:
                u.refuse('W-NAMED %s' % e)
            keys = cols + ['named_radec']
            rr = [r for r in rows if r['stokes'] == st and r['tier'] == cad]
            u.plan(final, 'forced_wide_published')
            part = final + '.part'
            V3.write_rows_keys(part, rr, keys)
            ctx = dict(results=res[0], jobs=jobs_path, stokes=st, tier=cad)
            bad = [t for t in validate.run('forced_wide_published', part, **ctx) if not t['passed']]
            if bad:
                os.remove(part)
                u.refuse('content test of %s failed: %s' % (part, '; '.join('%s (%s)' % (t['name'], str(t['detail'])[:200])
                                                                           for t in bad)))
            prev = _sha1_file(final) if os.path.exists(final) else None
            os.replace(part, final)
            u.meta.update(n_rows=len(rr), replaced_sha1=prev)
            u.add_output(final, 'forced_wide_published', **ctx)
            print('%s %s: %d rows -> %s' % (st, cad, len(rr), final), flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# activity (family N)
# ---------------------------------------------------------------------------------------------------------------------

def stage_activity(args, run):
    from exopipe.phase3.lib.forced import night_activity as NA
    c = K.cfg().night_activity
    W = os.path.abspath(run.out_root)
    src = os.path.abspath(args.na_detections or c.DETECTIONS)
    bands_p, nights_p = os.path.join(W, c.FILES[0]), os.path.join(W, c.FILES[1])
    name = 'N_activity'
    K.plan('[activity] the listed detections %s (Stokes %s, %s) -> %s, %s' % (src, c.STOKES, c.TIER, bands_p, nights_p))
    if run.dry_run:
        K.plan('[activity]   input %s' % ('found' if os.path.exists(src) else 'NOT FOUND (refused)'))
        return
    if K.resumed(run, name, None, tier='snapshot', stage='activity', family='N'):
        return
    with run.unit(name, tier='snapshot', stage='activity', family='N', detections=src) as u:
        try:
            K.require_host(src, 'the listed detections')
        except Refusal as e:
            raise Refusal('N-INPUT %s' % e)
        u.add_input(src)
        try:
            cnt, nights, n_rows, n_unique, wrong = NA.read_detections(src, c)
        except KeyError as e:
            u.refuse('N-INPUT %s' % (e.args[0] if e.args else e))
        if wrong:
            u.refuse('N-INPUT rows of another Stokes or tier than %s %s in %s, e.g. row %d: %s %s (the table is the served '
                     '10-s Stokes I view\'s)' % (c.STOKES, c.TIER, src, wrong[0][0], wrong[0][1], wrong[0][2]))
        if not cnt:
            raise NoInput('no detection in %s' % src)
        rows, nrows = NA.build(cnt, nights, c)
        u.plan(bands_p, 'night_activity_bands')
        u.plan(nights_p, 'night_activity_nights')
        os.makedirs(W, exist_ok=True)
        NA.write_table(bands_p, rows, NA.BAND_COLUMNS)
        NA.write_table(nights_p, nrows, NA.NIGHT_COLUMNS)
        u.meta.update(n_input_rows=n_rows, n_detections=n_unique, n_band_rows=len(rows), n_night_rows=len(nrows),
                      n_lsts=len(nights))
        u.add_output(bands_p, 'night_activity_bands', n_detections=n_unique)
        u.add_output(nights_p, 'night_activity_nights', bands=bands_p)
        print('N_activity: %d detections; %d band rows; %d night rows' % (n_unique, len(rows), len(nrows)), flush=True)


STAGE_FN = {'wide_jobs': stage_jobs, 'wide_search': stage_search, 'wide_test': stage_test, 'wide_publish': stage_publish}
