"""exopipe3 masters --tier sublst - the 10-minute (sub-LST) tier's masters: per-LST inputs, bin admission (the stack job
plan), window stacks per night, window masters + leave-one-out residuals, parent catalogues.

Stages, in this order (default --stages = masters.DEFAULT_STAGES['sublst'] = prep,admit,stack,loo):
  prep      unit prep_<LST>        md5-verified copies of the frozen generation-4 masters + PB, the live deep weights.csv and
                                   residuals_provisional.csv, the footprint and Run_P3S4 parent catalogues, roots_<LST>.csv
                                   (port of exopipe_ops/sublst_prep_lst.sh; the /data07 and /fast sources are lwacalim07-local)
  admit     unit admit_<KEY>       bin admission: the stack job plan (port of the heredoc of sublst_run_lst_nocat.sh:82-186)
                                   -> <out>/_plans/sublst_jobs_<KEY>.txt and sublst_jobs_<KEY>_deepflow.txt, with every job left
                                   out and its rule in the record. KEY = <LST>, or <LST>_<night> when one night is selected.
  stack     unit stack_<LST>_<night>_<sb>   clv15/phase3_sublst_stack.py in place, Stokes I then V (two invocations, B-76 (b)):
                                   six 10-minute window stacks of the night, flux_dc measured, deep flux scale baked in; then
                                   unit test-windows_<LST>: production content test (a) on <out>/<LST> (class sublst_window_sample)
  loo       unit loo_<LST>_<sb>    clv15/phase3_sublst_loo.py in place, I and V: window masters (moving Jupiter mask) and the
                                   arm-A leave-one-out residuals (per-night rule, Q11)
  sources   unit sources_<LST>     parent catalogues on the LST's masters: production's conformance check (exopipe_ops/
                                   gate_inputs.parents, in place) and exopipe_ops/sublst_parent_catalogs.py when stale
  wideband is NOT a masters stage: frames --tier sublst --stages wideband (refactor decision 6; content test (b) is frames' too).

Items: --lsts 09h (prep, loo, sources; admit/stack over the LST's nights); --cells <LST>_<night>[_<subband>] (core.cells
night_cell: stack and admit restricted to those nights / subbands); --epochs <LST>_<date> when the step declares it;
--sublst-worklist FILE (production 'LST NIGHT' lines). Default nights of an LST = the cube-root cells holding a COMPLETE
MANIFEST.txt (the production worklist rule). --nights, --exclude-cells <LST>_<night>, --scopes (subbands), --stokes (default
I,V) narrow the selection.

Inputs (read only): snapshot cubes <paths.SNAPSHOT_CUBES_ROOT>/<LST>_<night>/<cell>_<sb>_{I,V}_flowcube.fits + MANIFEST.txt, the
deep-flow cubes, the frozen deep records <paths.SUBLST_ROOT>/<FROZEN_RECORDS_DIR>/<LST>/<sb>/weights.csv, the frozen gen-4
masters (masters_snapshot.FROZEN_ROOT), the tier-rule pins (md5 checked), data_decisions.csv (core.exclusions and the co-add
reader must agree), the solar audit (night rule), the PB library; prep: paths.COADD_SB_ROOT, the footprint catalogues.
Outputs (production layout under --out, default paths.SUBLST_ROOT; [product classes], tests in lib/masters/tests_sublst.py):
  coadd_<LST>/<LST>_<band>/..., catalog_inputs/..., parent_catalogs_<LST>/master_R075_* (prep) [sublst_input_copy];
  catalog_inputs/roots_<LST>.csv [sublst_input_manifest];  _plans/sublst_jobs_<KEY>[_deepflow].txt [sublst_job_plan];
  <LST>/<cell>/<sb>/<S>/sublst_<cell>_<sb>_<S>_w<k>.fits [sublst_window], _nframes.fits [sublst_window_nframes], _pb.fits
  [sublst_window_pb], admission_<cell>_<sb>_<S>.csv [sublst_admission], <LST>/<cell>/<sb>/I/fluxdc_<cell>_<sb>.json
  [sublst_fluxdc_record]; <LST>_masters/<sb>/<S>/sublst_master_<LST>_<sb>_<S>_w<k>.fits [sublst_window_master] + _weights.csv
  [csv_table]; <LST>/<cell>/<sb>/<S>/..._w<k>_resid_loo.fits [sublst_loo_residual]; parent_catalogs_<LST>/master_R075_<LST>_
  <scope>.gaul.csv (sources) [master_catalog]; the <LST> directory [sublst_window_sample]. Engine logs: <out>/<LST>/logs/ with
  the production names. Run records: <out>/_records/masters/<unit>.json.

Failure modes (exit 2 = refused, the gate named in the record): earlier products of a unit under --out without --fresh
--supersede TAG (G-S01); a cell without a cube MANIFEST.txt (G-S02); the stacker or the loo engine refusing (G-S04, G-S06; a refused Stokes is an item refusal of the unit); loo before every planned
stack job is done (G-S05; masters_sublst.LOO_ADMITS_REFUSED_STACKS); a missing prep source or a copy whose md5 differs (G-S08);
a tier-rule pin missing or altered (G-S09); stale parent catalogues that cannot be rebuilt here (G-S10:
sublst_parent_catalogs.py writes only to its literal root, so a rebuild runs only with --out = paths.SUBLST_ROOT); config that
differs from an in-place literal it mirrors; core.exclusions and the co-add reader disagreeing (G-M01); a site PYTHON key
missing. Inherited P3_* variables are removed from the process before the in-place imports (noted in the records). Exit 3:
nothing to do - no cell, or every job of the plan left out by rule (G-S03: the reasons are in the admit record; production
goes on with 0 jobs); exit 1: an engine error (no completion line and no REFUSED line).

Per-hour vs intermittent (DESIGN 12): the window masters (loo) and parents (sources) are INTERMITTENT (rebuilt when the nights
on disk grow). The night's window stacks are the per-hour work: frames --tier sublst calls stack_night() below (ONE wrapper,
ONE config section masters_sublst - refactor decision 6); `masters --tier sublst --cells <LST>_<night> --stages admit,stack`
runs the same code.

INTERFACE imported by frames --tier sublst (keep stable; documented contract):
  stack_night(run, lst, night, sublst_root=None, subbands=None, stokes=('I', 'V'), workers=None, fresh=False, supersede=None,
              resume=False) -> {'admit_unit', 'stack_units', 'jobs', 'left_out', 'would_refuse'}
      the night's admit unit (admit_<LST>_<night>, plan files under <run.out_root>/_plans/) and one stack unit per admitted
      subband (stack_<LST>_<night>_<sb>, Stokes I then V), products under <sublst_root or run.out_root>/<LST>/<cell>/, records
      under the CALLER's step (run.step). In --dry-run it prints the plan and writes nothing.
  admit_plan(lst, cells, subbands=None) -> (jobs on the main cubes, jobs on the deep-flow cubes, left-out reasons)
      jobs are 'cell subband Stokes' strings in production's job-list format.
  stack_unit_name(lst, night, sb), stack_products(root, lst, cell, sb, S, k=None), PRODUCT_CLASSES.
"""
import csv
import glob
import hashlib
import os
import re
import shutil
import threading

from exopipe.phase3.core import record
from exopipe.phase3.core.record import NoInput, Refusal

TIER = 'sublst'
SECTION = 'masters_sublst'
OWNED_ELSEWHERE = {'wideband': 'frames --tier sublst --stages wideband (refactor decision 6; crosscheck 1.1 #3: the wideband '
                               'windows and the set-B wideband residuals are frame products)'}
PRODUCT_CLASSES = {'prep': ('sublst_input_copy', 'sublst_input_manifest'), 'admit': ('sublst_job_plan',),
                   'stack': ('sublst_window', 'sublst_window_nframes', 'sublst_window_pb', 'sublst_admission',
                             'sublst_fluxdc_record', 'sublst_window_sample'),
                   'loo': ('sublst_window_master', 'csv_table', 'sublst_loo_residual'), 'sources': ('master_catalog',)}
_PRINT_LOCK = threading.Lock()

# in-place code: (legacy root name, file) - resolved with core.env.legacy_root
SCRIPTS = {'stack': ('clv15', 'phase3_sublst_stack.py'), 'loo': ('clv15', 'phase3_sublst_loo.py'),
           'tier_rules': ('clv15', 'phase3_tier_rules.py'), 'jupmask': ('clv15', 'phase3_jupiter_mask.py'),
           'parentcat': ('exopipe_ops', 'sublst_parent_catalogs.py'),
           'test_windows': ('exopipe_ops', 'sublst_reserve_test_windows.py'),
           'nocat': ('exopipe_ops', 'sublst_run_lst_nocat.sh'), 'gate_inputs': ('exopipe_ops', 'gate_inputs.py')}


# =====================================================================================================================
# config, roots, in-place code
# =====================================================================================================================

def cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _sec():
    return getattr(cfg(), SECTION)


def subbands():
    return tuple(cfg().instrument.ALL_SUBBANDS)


def sublst_root():
    return cfg().paths.SUBLST_ROOT


def cube_root():
    return cfg().paths.SNAPSHOT_CUBES_ROOT


def deepflow_root():
    return os.path.join(sublst_root(), _sec().DEEPFLOW_CUBES_DIR)


def frozen_records():
    return os.path.join(sublst_root(), _sec().FROZEN_RECORDS_DIR)


def frozen_weights(lst, sb):
    return os.path.join(frozen_records(), lst, sb, 'weights.csv')


def frozen_masters():
    return cfg().masters_snapshot.FROZEN_ROOT


def master_path(lst, sb):
    """The frozen gen-4 I Taper R0 master the cubes were flowed to (phase3_sublst_stack.MASTER_PATH)."""
    return os.path.join(frozen_masters(), f'{lst}_{sb}', 'coadd_final', f'I_deep_Taper_R0_{sb}.fits')


def prep_parents_dir():
    return os.path.join(cfg().paths.TRANSIENTS_ROOT, _sec().PREP_PARENTS_RUN, 'parent_catalogs')


def script(short):
    """Path of an in-place script under its legacy code root (core.env.legacy_root)."""
    from exopipe.phase3.core import env
    rootname, name = SCRIPTS[short]
    try:
        p = os.path.join(env.legacy_root(rootname), name)
    except KeyError as e:
        raise Refusal(f'in-place code {name}: {e} (refactor window, DESIGN 11.1)')
    if not os.path.isfile(p):
        raise Refusal(f'in-place code {p} not found')
    return p


_ENGINE_PYTHON = {'stack': 'sublst-stack', 'loo': 'sublst-loo', 'sources': 'sources'}


def python_key(engine):
    """The site PYTHON key of an in-place engine: masters.PYTHON_SITE_KEY (the step owner's section) for stack, loo and
    sources; masters_sublst.TEST_WINDOWS_PYTHON for production content test (a)."""
    if engine == 'test_windows':
        return _sec().TEST_WINDOWS_PYTHON
    return cfg().masters.PYTHON_SITE_KEY[_ENGINE_PYTHON[engine]]


def interpreter(engine):
    """The site interpreter of an in-place engine; refuses when the site file lacks it."""
    from exopipe.phase3.core import legacy
    return legacy.interpreter(python_key(engine))


def say(text):
    with _PRINT_LOCK:
        print(text, flush=True)


# (label, config value getter, in-place file, regex on its source text, converter): literal == config, else refuse
MIRRORS = [
    ('masters_sublst.BEAM_AREA_MAX', lambda s: s.BEAM_AREA_MAX, 'stack', r'^BEAM_AREA_MAX = ([0-9.]+)', float),
    ('masters_sublst.NOISE_K', lambda s: s.NOISE_K, 'stack', r'^NOISE_K = ([0-9.]+)', float),
    ('masters_sublst.BLANK_FINITE_MIN', lambda s: s.BLANK_FINITE_MIN, 'stack', r'^BLANK_FINITE_MIN = ([0-9.]+)', float),
    ('masters_sublst.INTEGRATION_S', lambda s: s.INTEGRATION_S, 'stack', r'^INTEGRATION_S = ([0-9.]+)', float),
    ('masters_sublst.QA3_ADMIT_FLOOR', lambda s: s.QA3_ADMIT_FLOOR, 'stack', r'dc <= ([0-9.]+):', float),
    ('masters_sublst.MIN_USABLE_FRAMES', lambda s: s.MIN_USABLE_FRAMES, 'stack', r'if base_ok\.sum\(\) < (\d+):', int),
    ('masters_sublst.PRODUCT_CARD', lambda s: s.PRODUCT_CARD, 'stack', r'hh\["PRODUCT"\] = "([^"]+)"', str),
    ('masters_sublst.WINDOW_MIN', lambda s: s.WINDOW_MIN, 'stack', r'"--window-min", type=float, default=([0-9.]+)\)', float),
    ('frozen records (paths.SUBLST_ROOT + FROZEN_RECORDS_DIR)', lambda s: frozen_records(), 'stack',
     r'^FLUX_SCALE_CSV = "([^"{]+)/\{lst\}', str),
    ('frozen records (paths.SUBLST_ROOT + FROZEN_RECORDS_DIR)', lambda s: frozen_records(), 'loo',
     r'^WEIGHTS_CSV = "([^"{]+)/\{lst\}', str),
    ('frozen records (paths.SUBLST_ROOT + FROZEN_RECORDS_DIR)', lambda s: frozen_records(), 'tier_rules',
     r'^COADD_FROZEN = "([^"]+)"', str),
    ('masters_snapshot.FROZEN_ROOT', lambda s: frozen_masters(), 'stack', r'^MASTER_PATH = "([^"{]+)/\{lst\}', str),
    ('masters_sublst.N_WINDOWS', lambda s: s.N_WINDOWS, 'loo', r'for k in range\((\d+)\):', int),
    ('masters_sublst.LOO_MIN_WEIGHTED_NIGHTS', lambda s: s.LOO_MIN_WEIGHTED_NIGHTS, 'loo',
     r'if sum\(1 for x in w if x > 0\) < (\d+) or', int),
    ('masters_sublst.PER_NIGHT_MIN', lambda s: s.PER_NIGHT_MIN, 'loo', r'if n_loo < (\d+):', int),
    ('masters_sublst.WINDOW_COADDOP', lambda s: s.WINDOW_COADDOP, 'loo', r'mh\["COADDOP"\] = \("([^"]+)"', str),
    ('masters_sublst.JUPITER_MASK_RADIUS_BEAMS', lambda s: s.JUPITER_MASK_RADIUS_BEAMS, 'jupmask',
     r'^RADIUS_BEAMS = ([0-9.]+)', float),
    ('masters_sublst.TIER_RULE4_RANGE', lambda s: tuple(s.TIER_RULE4_RANGE), 'tier_rules', r'^FLUX_SCALE_RANGE = \(([^)]*)\)',
     lambda t: tuple(float(x) for x in t.split(','))),
    ('masters_sublst.MEMBERSHIP_VETO_PIN_MD5', lambda s: s.MEMBERSHIP_VETO_PIN_MD5, 'tier_rules',
     r'^MEMBERSHIP_VETO_PIN_MD5 = "([0-9a-f]+)"', str),
    ('masters_sublst.DEEP_SEARCH_PIN', lambda s: s.DEEP_SEARCH_PIN, 'tier_rules', r'^DEEP_SEARCH_PIN = "([^"]+)"', str),
    ('masters_sublst.DEEP_SEARCH_PIN_MD5', lambda s: s.DEEP_SEARCH_PIN_MD5, 'tier_rules',
     r'^DEEP_SEARCH_PIN_MD5 = "([0-9a-f]+)"', str),
]


def mirror_violations():
    """['<label> = v but <file> uses w'] for every mirror whose in-place literal differs from config (or is not found)."""
    s = _sec()
    bad, texts = [], {}
    for label, get, short, rx, conv in MIRRORS:
        path = script(short)
        if path not in texts:
            texts[path] = open(path).read()
        m = re.search(rx, texts[path], re.M)
        if not m:
            bad.append(f'{label}: literal not found in {path} (pattern {rx!r}) - the in-place code changed')
            continue
        have, want = conv(m.group(1)), get(s)
        if have != want:
            bad.append(f'{label} = {want!r} but {os.path.basename(path)} uses {have!r}')
    nocat = open(script('nocat')).read()
    m = re.search(r'^SBS="([^"]+)"', nocat, re.M)
    if not m or tuple(m.group(1).split()) != subbands():
        bad.append(f'instrument.ALL_SUBBANDS {subbands()} != the subband list SBS of sublst_run_lst_nocat.sh')
    return bad


# =====================================================================================================================
# shared machinery (also used by tier_snapshot)
# =====================================================================================================================

class Opts:
    """Run options a unit needs (from the CLI args, or given by an importing step such as frames)."""

    def __init__(self, fresh=False, supersede=None, resume=False, workers=None, sublst_jobs=None, sublst_deepflow_jobs=None):
        self.fresh, self.supersede, self.resume, self.workers = bool(fresh), supersede, bool(resume), workers
        self.sublst_jobs, self.sublst_deepflow_jobs = sublst_jobs, sublst_deepflow_jobs

    @classmethod
    def from_args(cls, args):
        return cls(fresh=getattr(args, 'fresh', False), supersede=getattr(args, 'supersede', None),
                   resume=getattr(args, 'resume', False), workers=getattr(args, 'workers', None),
                   sublst_jobs=getattr(args, 'sublst_jobs', None),
                   sublst_deepflow_jobs=getattr(args, 'sublst_deepflow_jobs', None))


def resolve_stages(args, tier):
    """The stages to run in the tier's fixed order: the dispatcher's args.stages_resolved, else --stages / --from / --until."""
    done = getattr(args, 'stages_resolved', None)
    if done:
        return list(done)
    c = cfg().masters
    order = tuple(c.STAGES[tier])
    req = getattr(args, 'stages', None)
    sel = [x.strip() for x in req.split(',') if x.strip()] if isinstance(req, str) else list(req or c.DEFAULT_STAGES[tier])
    frm, until = getattr(args, 'from_stage', None), getattr(args, 'until_stage', None)
    unknown = [x for x in sel + [y for y in (frm, until) if y] if x not in order]
    if unknown:
        raise SystemExit(f'unknown stage(s) {unknown} for --tier {tier}; stages in order: {", ".join(order)}')
    if frm or until:
        lo = order.index(frm) if frm else min(order.index(x) for x in sel)
        hi = order.index(until) if until else max(order.index(x) for x in sel)
        sel = list(order[lo:hi + 1])
    return [x for x in order if x in sel]


def skip_unit(run, name, why):
    """A unit --resume keeps: listed in this run as skipped-valid; its done record on disk is left as it is."""
    u = record.UnitRecord(run, name, {})
    u.status, u.reason = 'skipped-valid', why
    run.units.append(u)
    say(f'  {name}: skipped (--resume: {why})')
    return u


def resume_skip(run, opts, name, inputs, admitted):
    if not opts.resume or opts.fresh:
        return False
    ok, why = run.resume_check(name, inputs=inputs, admitted=admitted)
    if ok:
        skip_unit(run, name, why)
    return ok


def _nonempty(p):
    if os.path.isfile(p) or os.path.islink(p):
        return True
    return os.path.isdir(p) and any(True for _ in os.scandir(p))


def earlier_products(paths):
    return [p for p in paths if _nonempty(p)]


def clear_earlier(run, opts, u, paths, what, gate='G-S01'):
    """Earlier products of this unit under the product root: refuse, or with --fresh --supersede TAG move them to
    <run.out_root>/_superseded/<TAG>/<same relative path> with a sha1 manifest (replaces 'move them aside first', nocat:62-68)."""
    old = earlier_products(paths)
    if not old:
        return []
    if not (opts.fresh and opts.supersede):
        u.refuse(f'{gate}: {what}: products of an earlier run (e.g. {old[0]}) - pass --fresh --supersede TAG to move them '
                 f'aside (never silently re-used)')
    if run.dry_run:
        return old
    dest = os.path.join(run.out_root, '_superseded', opts.supersede)
    rows = []
    for p in old:
        rel_top = os.path.relpath(p, run.out_root)
        if rel_top.startswith('..'):
            u.refuse(f'{gate}: {p} is outside the run root {run.out_root}: not moved')
        files = [p] if not os.path.isdir(p) else [os.path.join(dp, f) for dp, _, fs in os.walk(p) for f in fs]
        for f in files:
            h = hashlib.sha1()
            with open(f, 'rb') as fh:
                for blk in iter(lambda: fh.read(1 << 22), b''):
                    h.update(blk)
            rows.append((os.path.relpath(f, run.out_root), h.hexdigest(), os.path.getsize(f)))
        os.makedirs(os.path.dirname(os.path.join(dest, rel_top)), exist_ok=True)
        shutil.move(p, os.path.join(dest, rel_top))
    man = os.path.join(dest, 'MANIFEST.csv')
    new = not os.path.exists(man)
    with open(man, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['path', 'sha1', 'bytes', 'unit', 'utc'])
        for r in rows:
            w.writerow(list(r) + [u.name, record.utcnow()])
    u.note(f'superseded {len(rows)} file(s) of an earlier run to {dest} (sha1 manifest {man})')
    return old


_CLEARED_P3 = []


def p3_switches():
    """P3_* variables still set in THIS process: the in-process admit predictor and parent check import clv15 / exopipe_ops
    code that reads them at import (P3_COADD_RUN, P3_SOLAR_AUDIT, P3_PARENT_DIR, ...); production's drivers never set them."""
    return sorted(k for k in os.environ if k.startswith('P3_'))


def clear_p3_env():
    """Remove inherited P3_* variables from this process before any in-place module is imported (core.legacy.run does the
    same for children; the deep tier does it for its in-process co-add). The names removed are noted in the admit and
    sources records."""
    from exopipe.phase3.lib.masters import deep_inplace
    gone = sorted(deep_inplace.clear_inherited_p3())
    _CLEARED_P3.extend(k for k in gone if k not in _CLEARED_P3)
    if gone:
        say(f'removed inherited environment variables {gone} (the in-place modules read them; production never sets them)')
    return gone


# a refusal line of the engines: 'REFUSED: ...' (sys.exit), 'RuntimeError: REFUSING: ...' / 'REFUSED (...' (tier rules,
# snapshot search) - never the loo's progress lines 'REFUSED-WINDOW' / 'REFUSED-RESIDUAL' / 'REFUSED-NIGHT'
_REFUSED_RE = re.compile(r'^(?:\w+Error: )?REFUS(?:ED|ING)(?::| \()')


def outcome(rc, lines, success_re):
    """('done', completion line) | ('refused', reason) | ('error', reason) of an in-place engine run. done = rc 0 and the
    engine's own completion line (production's test, sublst_stack_pass.sh:23); refused = rc != 0 and the engine printed a
    refusal line - 'REFUSED: ...' (sys.exit), 'RuntimeError: REFUSING: ...' or 'REFUSED (...' (the tier rules, the snapshot
    search; production's FAILED line, :24); the last such line is the reason. Anything else is an error (retried). Stricter
    than core.legacy's res.refused on purpose: the loo prints 'REFUSED-WINDOW' / 'REFUSED-RESIDUAL' progress lines in
    normal output, so a crash or a kill after them is an error, not a refusal."""
    ok = [l for l in lines if re.match(success_re, l)]
    refusals = [l.strip() for l in lines if _REFUSED_RE.match(l.strip())]
    tail = [l for l in lines if l.strip() and 'warn' not in l.lower() and not l.startswith('===== ')]
    last = tail[-1].strip() if tail else ''
    if rc == 0 and ok:
        return 'done', ok[-1]
    if rc != 0 and refusals:
        return 'refused', refusals[-1][:600]
    return 'error', f'exit {rc}; ' + (f'completion line present but exit {rc}' if ok else f'last line: {last[:400]}')


def run_engine(u, short, engine, args, log, success_re, env=None, cwd=None, unbuffered=False):
    """One in-place engine run through core.legacy.run (the one child runner: site interpreter, P3_* cleared, parity guard,
    child code fingerprint into the unit record). The log is truncated first, as production's '> $L 2>&1'. unbuffered =
    production's `python -u` (stacker, loo: sublst_stack_pass.sh:22, nocat:231), given as PYTHONUNBUFFERED=1 because
    core.legacy.run takes no interpreter flags: the log keeps the engine's chronological order. Returns (status, reason,
    lines) per outcome()."""
    from exopipe.phase3.core import legacy
    os.makedirs(os.path.dirname(log), exist_ok=True)
    open(log, 'w').close()
    env = dict(env or {}, **({'PYTHONUNBUFFERED': '1'} if unbuffered else {}))
    res = legacy.run(script(short), args, python=python_key(engine), env=env, cwd=cwd, log=log, unit=u)
    with open(log, errors='replace') as f:
        lines = f.read().splitlines()
    st, why = outcome(res.rc, lines, success_re)
    return st, why, lines


def md5_file(path, bs=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(bs), b''):
            h.update(b)
    return h.hexdigest()


def pool_map(fn, items, workers):
    """fn(item) for every item, ``workers`` at a time (threads: every unit's work is a child process). A unit's error is in
    its record already; it is printed and the others go on."""
    from concurrent.futures import ThreadPoolExecutor
    if not items:
        return
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        for f in [ex.submit(fn, it) for it in items]:
            try:
                f.result()
            except Exception as e:
                say(f'  unit error: {type(e).__name__}: {str(e)[:300]}')


def exclusion_readers_agree(lst, scope, nights):
    """G-M01: core.exclusions and the co-add's own reader (the reader the stacker and the admit heredoc use) agree on every
    (LST, scope, night), the exclusion file paths and the 18 MHz carve-out agree; (core Exclusions, co-add exclusion set).
    The masters step's one implementation: lib/masters/deep_inplace.exclusion_readers_agree."""
    from exopipe.phase3.lib.masters import deep_inplace
    return deep_inplace.exclusion_readers_agree(lst, scope, nights)


# =====================================================================================================================
# selection
# =====================================================================================================================

def read_worklist(path):
    """Production worklist: 'LST NIGHT' lines, '#' comments (nocat:47 awk '!/^#/ && NF==2')."""
    out = []
    for line in open(path):
        f = line.split()
        if line.startswith('#') or len(f) != 2:
            continue
        out.append(f'{f[0]}_{f[1]}')
    return out


def cube_cells(lst):
    """Cells of the cube root for an LST holding a COMPLETE MANIFEST.txt (how production makes its worklists,
    sublst_newlst_queue_20261010.sh)."""
    out = []
    for d in sorted(glob.glob(os.path.join(cube_root(), f'{lst}_????-??-??'))):
        try:
            if any(l.startswith('COMPLETE') for l in open(os.path.join(d, 'MANIFEST.txt'))):
                out.append(os.path.basename(d))
        except OSError:
            continue
    return out


class Selection:
    """What a run covers: LSTs, the cells (nights) of each LST, per-cell subband limits, subbands, Stokes."""

    def __init__(self):
        self.lsts, self.cells, self.explicit, self.cell_scopes = [], {}, set(), {}
        self.nights, self.subbands, self.stokes = None, list(subbands()), ['I', 'V']
        self.excluded = set()      # night cells left out (--exclude-cells <LST>_<night>)

    def add_cell(self, lst, night, scope=None):
        cell = f'{lst}_{night}'
        if lst not in self.lsts:
            self.lsts.append(lst)
        self.explicit.add(lst)
        if cell not in self.cells.setdefault(lst, []):
            self.cells[lst].append(cell)
        if scope:
            if self.cell_scopes.get(cell) != set():
                self.cell_scopes.setdefault(cell, set()).add(scope)
        else:
            self.cell_scopes[cell] = set()

    @classmethod
    def from_args(cls, args):
        from exopipe.phase3.core import cells as CL
        sel = cls()
        toks = []
        spec = getattr(args, 'cells', None)
        if spec:
            universe = sorted(d for d in os.listdir(cube_root())) if any(c in spec for c in '*?[') else None
            toks += CL.expand(spec, universe=universe, kind='sub-LST night cell')
        for e in CL.expand_epochs(getattr(args, 'epochs', None)) if getattr(args, 'epochs', None) else []:
            toks.append(e)
        if getattr(args, 'sublst_worklist', None):
            toks += read_worklist(args.sublst_worklist)
        for t in toks:
            try:
                lst, night, scope = CL.parse_night_cell(t)
            except ValueError as e:
                raise SystemExit(f'masters --tier sublst: {e}')
            if scope and scope not in subbands():
                raise SystemExit(f'masters --tier sublst: {t}: the sub-LST masters stages take subbands; wideband groups '
                                 f'are {OWNED_ELSEWHERE["wideband"]}')
            sel.add_cell(lst, night, scope)
        for lst in CL.expand(getattr(args, 'lsts', None), universe=CL.LSTS, kind='LST'):
            try:
                CL.parse_lst(lst)
            except ValueError as e:
                raise SystemExit(str(e))
            if lst not in sel.lsts:
                sel.lsts.append(lst)
        sel.lsts.sort()
        if getattr(args, 'nights', None):
            sel.nights = set(CL.expand(args.nights))
        for t_ in CL.expand(getattr(args, 'exclude_cells', None)) if getattr(args, 'exclude_cells', None) else []:
            try:
                lst_, night_, _ = CL.parse_night_cell(t_)
            except ValueError as e:
                raise SystemExit(f'masters --tier sublst --exclude-cells: {e}')
            sel.excluded.add(f'{lst_}_{night_}')
        sc = CL.expand(getattr(args, 'scopes', None)) if getattr(args, 'scopes', None) else []
        bad = [x for x in sc if x not in subbands()]
        if bad:
            raise SystemExit(f'--scopes {bad}: the sub-LST masters stages take subbands (wideband groups are frames)')
        sel.subbands = [x for x in subbands() if not sc or x in sc]
        st = getattr(args, 'stokes', None)
        sel.stokes = [x for x in ('I', 'V') if not st or x in [y.strip() for y in str(st).split(',')]]
        return sel

    @classmethod
    def one_night(cls, lst, night, sbs=None, stokes=('I', 'V')):
        sel = cls()
        sel.add_cell(lst, night)
        sel.subbands = [x for x in subbands() if not sbs or x in sbs]
        sel.stokes = [x for x in ('I', 'V') if x in stokes]
        return sel

    def cells_of(self, lst):
        cs = list(self.cells.get(lst, [])) if lst in self.explicit else cube_cells(lst)
        if self.nights is not None:
            cs = [c for c in cs if c.split('_', 1)[1] in self.nights]
        return sorted(c for c in cs if c not in self.excluded)

    def scope_ok(self, cell, sb):
        named = self.cell_scopes.get(cell)
        return not named or sb in named

    def admit_key(self, lst):
        """<LST> for an LST-wide plan (worklist / cube root / several nights), <LST>_<night> for a one-night plan."""
        cs = self.cells_of(lst)
        return f'{lst}_{cs[0].split("_", 1)[1]}' if lst in self.explicit and len(cs) == 1 else lst


# =====================================================================================================================
# stage: prep  (port of exopipe_ops/sublst_prep_lst.sh)
# =====================================================================================================================

def prep_pairs(lst, root):
    """[(source, destination, kind)] of the prep copies in the script's order (kind: master | weights | residuals | footprint
    | parent), the notes (footprint fallbacks) and the roots_<LST>.csv rows. A missing footprint source is 'A | B'."""
    s = _sec()
    coadd_sb = cfg().paths.COADD_SB_ROOT
    pairs, notes, roots = [], [], []
    for b in tuple(subbands()) + tuple(s.PREP_WIDEBAND):
        F = os.path.join(frozen_masters(), f'{lst}_{b}', 'coadd_final')
        S = os.path.join(coadd_sb, f'{lst}_{b}', 'coadd_final')
        D = os.path.join(root, f'coadd_{lst}', f'{lst}_{b}', 'coadd_final')
        for f in s.PREP_FILES:
            f = f.format(b=b)
            pairs.append((os.path.join(F, f), os.path.join(D, f), 'master'))
        pairs.append((os.path.join(S, 'weights.csv'), os.path.join(D, 'weights.csv'), 'weights'))
        pairs.append((os.path.join(coadd_sb, f'{lst}_{b}', 'residuals_provisional.csv'),
                      os.path.join(root, f'coadd_{lst}', f'{lst}_{b}', 'residuals_provisional.csv'), 'residuals'))
        roots.append(f'{lst}_{b},,{os.path.join(root, f"coadd_{lst}")}')
        if b in s.PREP_WIDEBAND:
            continue
        name = f'master_R075_{lst}_{b}.gaul.csv'
        F1, F2 = os.path.join(s.PREP_FOOTPRINT_DIR, name), os.path.join(prep_parents_dir(), name)
        src = F1 if os.path.isfile(F1) else (F2 if os.path.isfile(F2) else None)
        if src == F2:
            notes.append(f'footprint catalogue {lst} {b}: no August file - using the generation-4 parent catalogue {F2}')
        pairs.append((src or F1 + ' | ' + F2, os.path.join(root, 'catalog_inputs', f'bdsf_catalogs_aug_{lst}', name),
                      'footprint'))
    for p in sorted(glob.glob(os.path.join(prep_parents_dir(), f'master_R075_{lst}_*'))):
        pairs.append((p, os.path.join(root, f'parent_catalogs_{lst}', os.path.basename(p)), 'parent'))
    return pairs, notes, roots


def _locality(path):
    """'' or a note naming the host whose node-local root holds ``path`` (site NODE_LOCAL_ROOTS)."""
    import socket
    host = socket.gethostname().split('.')[0]
    for h, roots in (cfg().site.NODE_LOCAL_ROOTS or {}).items():
        for r in roots:
            if path.startswith(r.rstrip('/') + '/') and h != host:
                return f' (node-local to {h}: run this unit there)'
    if path.startswith(('/fast/', '/data0')):
        return f' (a node-local path; this is {host})'
    return ''


def cpv(src, dst):
    """The prep copy rule (sublst_prep_lst.sh:18-19): a destination already holding an identical file is left untouched
    (a catalogue run may be reading it); else copy with mode and times (cp -p) to <dst>.part, verify md5, rename.
    Returns (md5, 'kept' | 'copied'); a failed md5 refuses (G-S08)."""
    ms = md5_file(src)
    if os.path.isfile(dst) and md5_file(dst) == ms:
        return ms, 'kept'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    part = dst + '.part'
    shutil.copy2(src, part)
    if md5_file(part) != ms:
        os.remove(part)
        raise Refusal(f'G-S08: REFUSING: copy/md5 failed {src}')
    os.replace(part, dst)
    return ms, 'copied'


def stage_prep(run, opts, sel, root):
    s = _sec()
    bad = 0
    for lst in sel.lsts:
        name = f'prep_{lst}'
        pairs, notes, roots = prep_pairs(lst, root)
        missing = [(src, kind) for src, _, kind in pairs if ' | ' in src or not os.path.isfile(src)]
        nparent = sum(1 for p in pairs if p[2] == 'parent')
        if run.dry_run:
            say(f'[sublst prep {lst}] {len(pairs)} copies -> {root} ({sum(1 for p in pairs if os.path.isfile(p[1]))} '
                f'destinations exist; identical ones are kept); {nparent} parent files (>= {s.PREP_MIN_PARENTS} needed)')
            for n in notes:
                say(f'    {n}')
            for src, kind in missing[:6]:
                say(f'    WOULD REFUSE (G-S08): missing {kind} source {src}{_locality(src.split(" | ")[0])}')
            if len(missing) > 6:
                say(f'    ... {len(missing) - 6} more missing sources')
            if nparent < s.PREP_MIN_PARENTS:
                say(f'    WOULD REFUSE (G-S08): only {nparent} parent catalogue files for {lst}')
            bad += bool(missing) or nparent < s.PREP_MIN_PARENTS
            continue
        if resume_skip(run, opts, name, [p[0] for p in pairs if os.path.isfile(p[0])], None):
            continue
        with run.unit(name, lst=lst, stage='prep') as u:
            if missing:
                src, kind = missing[0]
                u.refuse(f'G-S08: REFUSING: missing {kind} source {src}{_locality(src.split(" | ")[0])} '
                         f'({len(missing)} missing in all; nothing copied)')
            if nparent < s.PREP_MIN_PARENTS:
                u.refuse(f'G-S08: REFUSING: only {nparent} parent catalogue files for {lst} in {prep_parents_dir()}')
            for n in notes:
                u.note(n)
            roots_csv = os.path.join(root, 'catalog_inputs', f'roots_{lst}.csv')
            for _, dst, _ in pairs:
                u.plan(dst, 'sublst_input_copy')
            u.plan(roots_csv, 'sublst_input_manifest')
            nk = nc = 0
            for src, dst, kind in pairs:
                u.add_input(src)
                ms, what = cpv(src, dst)
                nk += what == 'kept'
                nc += what == 'copied'
                u.add_output(dst, 'sublst_input_copy', source=src, source_md5=ms)
            os.makedirs(os.path.dirname(roots_csv), exist_ok=True)
            with open(roots_csv + '.new', 'w') as f:
                f.write(''.join(r + '\n' for r in roots))
            os.replace(roots_csv + '.new', roots_csv)
            u.add_output(roots_csv, 'sublst_input_manifest', lst=lst)
            u.note(f'PREP-DONE {lst}: {len(subbands())} subbands + wideband [{" ".join(s.PREP_WIDEBAND)}] x '
                   f'{len(s.PREP_FILES) + 1} co-add files + residuals_provisional.csv, {nparent} parent catalogue files, '
                   f'{len(subbands())} footprint catalogues, roots_{lst}.csv (all md5-verified; {nc} copied, {nk} kept)')
            say(f'  {name}: {nc} copied, {nk} identical kept')
    return bad


# =====================================================================================================================
# stage: admit  (port of the job-list heredoc, sublst_run_lst_nocat.sh:82-186)
# =====================================================================================================================

def admit_plan(lst, cells, subbands_=None):
    """(jobs on the main cube root, jobs on the deep-flow cubes, left-out reasons) for the cells of one LST: the heredoc of
    sublst_run_lst_nocat.sh:82-186 line for line. Changes, all value-identical: roots and floors from config (mirror-checked
    against the stacker); the co-add's exclusion reader from its moved home (lib/masters/coadd); BEAM_AREA_MAX from config
    instead of `from phase3_sublst_stack import` (mirror-checked; avoids the stacker's sys.path insertion of the production
    package). clv15 phase3_snapshot_cube, phase3_snapshot_search.night_verdict and phase3_tier_rules.flux_scale_failed are
    imported IN PLACE (legacy roots on sys.path through core.env.legacy)."""
    import calendar
    import time as _time
    import numpy as np
    from astropy.io import fits
    from exopipe.phase3.core import env
    env.legacy()
    import phase3_snapshot_cube as CB
    from phase3_snapshot_search import night_verdict
    from phase3_tier_rules import flux_scale_failed
    from exopipe.phase3.lib.masters.coadd import load_exclusions, is_excluded
    s = _sec()
    excl = load_exclusions()
    CUBES, DEEPFLOW = cube_root(), deepflow_root()
    INTEGRATION_S = s.INTEGRATION_S
    BEAM_AREA_MAX = s.BEAM_AREA_MAX

    def cube_path(root, c, sb):
        return os.path.join(root, c, "%s_%s_I_flowcube.fits" % (c, sb))

    def cube_flowed(root, c, sb):
        p = cube_path(root, c, sb)
        try:
            if not any(l.startswith("COMPLETE") for l in open(os.path.join(root, c, "MANIFEST.txt"))):
                return False
            h = fits.getheader(p, 0)
        except OSError:
            return False
        return bool(h.get("FLOWED")) and not h.get("NOMASTER")

    def usable_frames(root, c, sb, night):
        def frames_of(path):
            cb = CB.open_cube(path)
            ts = list(cb.timestamps)
            H = [fits.Header.fromstring(x if isinstance(x, str) else x.decode(), sep="\n") for x in cb._t["header"]]
            cb.close()
            return ts, H

        def mid(h, ts_):
            d = str(h.get("DATE-OBS", "") or "")
            try:
                t0 = calendar.timegm(_time.strptime(d[:19], "%Y-%m-%dT%H:%M:%S")) + (float("0" + d[19:]) if len(d) > 19 else 0.0)
            except ValueError:
                t0 = calendar.timegm(_time.strptime(ts_, "%Y%m%d_%H%M%S"))
            return t0 + INTEGRATION_S / 2.0
        ts, H = frames_of(cube_path(root, c, sb))
        tk = [round(mid(h, x), 1) for h, x in zip(H, ts)]
        area = np.array([float(h.get("BMAJ", np.nan)) * float(h.get("BMIN", np.nan)) for h in H])
        blank = np.array([float(h.get("WSCNVIS", 1) or 0) == 0 for h in H])
        vpath = cube_path(root, c, sb).replace("_I_flowcube.fits", "_V_flowcube.fits")
        area_o = np.full(len(ts), np.nan)
        if os.path.exists(vpath):
            tv, HV = frames_of(vpath)
            ao = {round(mid(h, x), 1): float(h.get("BMAJ", np.nan)) * float(h.get("BMIN", np.nan)) for h, x in zip(HV, tv)}
            area_o = np.array([ao.get(k, np.nan) for k in tk])
        beam_bad = np.zeros(len(ts), dtype=bool)
        for ar in (area, area_o):
            g = np.isfinite(ar)
            if g.any():
                beam_bad |= (ar / np.median(ar[g])) > BEAM_AREA_MAX
        dark = np.array([night_verdict(_time.strftime("%Y%m%d_%H%M%S", _time.gmtime(int(np.floor(k)))), lst, night, sb) == ""
                         for k in tk])
        return int(np.sum(dark & ~blank & ~beam_bad)), len(ts)

    SBS = list(subbands_) if subbands_ else list(subbands())
    out, outd, drop = [], [], []
    for c in cells:
        night = c.split("_", 1)[1]
        for sb in SBS:
            if is_excluded(lst, sb, night, excl):
                drop.append("%s %s (data_decisions)" % (c, sb))
                continue
            wp = frozen_weights(lst, sb)                    # frozen copy (ruling B, 2026-10-08)
            dc = None
            try:
                for r in csv.DictReader(open(wp)):
                    if r.get("date") == night:
                        try:
                            dc = float(r["qa3_scale"])
                        except (KeyError, ValueError, TypeError):
                            dc = None
            except OSError:
                dc = None
            if dc is None or not (dc == dc) or dc <= s.QA3_ADMIT_FLOOR:
                drop.append("%s %s (no usable qa3_scale in the deep co-add record)" % (c, sb))
                continue
            fsf = flux_scale_failed(lst, night, sb)
            if fsf:
                drop.append("%s %s (%s)" % (c, sb, fsf))
                continue
            root = CUBES
            if not cube_flowed(CUBES, c, sb):
                if cube_flowed(DEEPFLOW, c, sb):
                    root = DEEPFLOW
                elif os.path.exists(cube_path(CUBES, c, sb)):
                    drop.append("%s %s (cube holds unflowed frames - no Master when cubed - and no deep-Master re-flow)"
                                % (c, sb))
                    continue
            if os.path.exists(cube_path(root, c, sb)):
                nu, nt = usable_frames(root, c, sb, night)
                if nu < s.MIN_USABLE_FRAMES:
                    drop.append("%s %s (admission rule: %d of %d frames pass the blank, beam and night gates)"
                                % (c, sb, nu, nt))
                    continue
            (out if root == CUBES else outd).extend("%s %s %s" % (c, sb, S) for S in ("I", "V"))
    return out, outd, drop


def plan_paths(run, key):
    d = os.path.join(run.out_root, '_plans')
    return os.path.join(d, f'sublst_jobs_{key}.txt'), os.path.join(d, f'sublst_jobs_{key}_deepflow.txt')


def pins_ok():
    """G-S09: the tier-rule pins exist and carry their md5 (the in-place tier rules would refuse mid-run; checked first)."""
    s = _sec()
    pins = [(os.path.join(frozen_records(), 'membership_vetoes_pinned.csv'), s.MEMBERSHIP_VETO_PIN_MD5),
            (s.DEEP_SEARCH_PIN, s.DEEP_SEARCH_PIN_MD5)]
    bad = []
    for p, want in pins:
        if not os.path.isfile(p):
            bad.append(f'pin {p} missing')
        elif md5_file(p) != want:
            bad.append(f'pin {p} md5 {md5_file(p)} != {want}')
    return [p for p, _ in pins], bad


def _read_jobs(path):
    return [' '.join(l.split()[:3]) for l in open(path) if l.strip() and not l.startswith('#')]


def _compare_plan(opts, main, deep, cells, u=None):
    """With --sublst-jobs (production's job list) given to admit: report the set difference over the planned cells
    (parity, spec 10 (i))."""
    if not opts.sublst_jobs:
        return None
    ref = _read_jobs(opts.sublst_jobs) + (_read_jobs(opts.sublst_deepflow_jobs) if opts.sublst_deepflow_jobs else [])
    a, b = set(main + deep), {j for j in ref if j.split()[0] in set(cells)}
    msg = f'plan vs {opts.sublst_jobs}: ' + ('identical job sets' if a == b else
                                             f'+{len(a - b)} -{len(b - a)} (e.g. +{sorted(a - b)[:3]} -{sorted(b - a)[:3]})')
    say('    ' + msg)
    if u is not None:
        u.note(msg)
    return a == b


def stage_admit(run, opts, sel):
    bad = 0
    env_set = p3_switches()
    for lst in sel.lsts:
        key = sel.admit_key(lst)
        name = f'admit_{key}'
        cells = sel.cells_of(lst)
        nomani = [c for c in cells if not os.path.isfile(os.path.join(cube_root(), c, 'MANIFEST.txt'))]
        pins, pin_bad = pins_ok()
        jf, jdf = plan_paths(run, key)
        problems = ([f'G-S02: no cube MANIFEST.txt for {nomani[:5]} under {cube_root()}'] if nomani else []) + \
                   [f'G-S09: {b}' for b in pin_bad] + \
                   ([f'P3_* variables set in this process: {env_set} (the in-place modules read them)'] if env_set else [])
        if run.dry_run:
            say(f'[sublst admit {key}] {len(cells)} cells x {len(sel.subbands)} subbands -> {jf}')
            for p in problems:
                say(f'    WOULD REFUSE: {p}')
            if not cells:
                say('    no cells (NO-INPUT)')
            if problems or not cells:
                bad += 1
                continue
            main, deep, drop = admit_plan(lst, cells, sel.subbands)
            say("    job list: %d jobs (+ %d on the deep-Master re-flow cubes); left out: %s"
                % (len(main), len(deep), "; ".join(drop) or "none"))
            _compare_plan(opts, main, deep, cells)
            if not (main or deep):
                say('    NO-INPUT (G-S03): every (cell, subband) left out by rule - nothing to stack (exit 3)')
            continue
        with run.unit(name, lst=lst, stage='admit', cells=cells) as u:
            if not cells:
                raise NoInput(f'no cells for {lst} (no --cells / --epochs / --sublst-worklist and no cube-root cell with a '
                              f'COMPLETE MANIFEST.txt under {cube_root()})')
            if problems:
                u.refuse('; '.join(problems))
            if _CLEARED_P3:
                u.note(f'inherited environment variables removed before the in-place imports: {_CLEARED_P3}')
            nights = [c.split('_', 1)[1] for c in cells]
            ex = None
            for sb in sel.subbands:
                ex, _ = exclusion_readers_agree(lst, sb, nights)
            u.set_admitted([f'{n}/{sb}' for n in nights for sb in sel.subbands if not ex.is_excluded(lst, sb, n)])
            for p in pins + [frozen_weights(lst, sb) for sb in sel.subbands] + [cfg().paths.DATA_DECISIONS]:
                if os.path.exists(p):
                    u.add_input(p)
            for c in cells:
                u.add_input(os.path.join(cube_root(), c, 'MANIFEST.txt'))
            main, deep, drop = admit_plan(lst, cells, sel.subbands)
            summary = ("job list: %d jobs (+ %d on the deep-Master re-flow cubes); left out: %s"
                       % (len(main), len(deep), "; ".join(drop) or "none"))
            say(f'  {name}: {summary[:300]}')
            u.note(summary)
            u.meta.update(left_out=drop, subbands=list(sel.subbands))
            if not (main or deep):          # production goes on with 0 jobs (nocat:210 'na -ge NJ'): nothing to do, not a refusal
                raise NoInput(f'G-S03: every (cell, subband) of {key} left out by rule: ' + '; '.join(drop[:10]))
            os.makedirs(os.path.dirname(jf), exist_ok=True)
            with open(jf + '.part', 'w') as f:
                f.write("\n".join(main) + "\n")
            os.replace(jf + '.part', jf)
            with open(jdf + '.part', 'w') as f:
                f.write("".join(x + "\n" for x in deep))
            os.replace(jdf + '.part', jdf)
            u.meta['plan_sha1'] = {p: hashlib.sha1(open(p, 'rb').read()).hexdigest() for p in (jf, jdf)}
            u.add_output(jf, 'sublst_job_plan', lst=lst)
            u.add_output(jdf, 'sublst_job_plan', lst=lst)
            _compare_plan(opts, main, deep, cells, u)
    return bad


# =====================================================================================================================
# stage: stack  (clv15/phase3_sublst_stack.py in place; sublst_stack_pass.sh + nocat:188-221 ported)
# =====================================================================================================================

def stack_unit_name(lst, night, sb):
    return f'stack_{lst}_{night}_{sb}'


def stack_products(root, lst, cell, sb, S, k=None):
    """Paths of a stack job's products under the sub-LST product root (phase3_sublst_stack.py:143-145, 440-460)."""
    d = os.path.join(root, lst, cell, sb, S)
    out = {'dir': d, 'admission': os.path.join(d, f'admission_{cell}_{sb}_{S}.csv'),
           'fluxdc': os.path.join(root, lst, cell, sb, 'I', f'fluxdc_{cell}_{sb}.json'),
           'log': os.path.join(root, lst, 'logs', f'{cell}_{sb}_{S}.log')}
    if k is not None:
        base = os.path.join(d, f'sublst_{cell}_{sb}_{S}_w{k}')
        out.update(window=base + '.fits', nframes=base + '_nframes.fits', pb=base + '_pb.fits')
    return out


def load_plan(run, opts, lst, key, planned_now=False):
    """The job plan: [(cell, sb, [Stokes], cube root)], the plan files, how it was found. From the admit record's outputs
    (DESIGN 11.5: a stage reads only the files listed in the previous stage's record; their sha1 must be unchanged), or from
    --sublst-jobs / --sublst-deepflow-jobs (production job lists; recorded with fingerprints). (None, [], why) in a dry run
    whose admit stage is planned in the same invocation."""
    files = []
    if opts.sublst_jobs:
        files = [(opts.sublst_jobs, cube_root())]
        if opts.sublst_deepflow_jobs:
            files.append((opts.sublst_deepflow_jobs, deepflow_root()))
        how = 'job list file(s) given with --sublst-jobs'
    else:
        rec = record.load_record(run.out_root, run.step, f'admit_{key}')
        if rec is not None and rec.get('status') == 'no-input':
            return [], [], f'admit record no-input: {rec.get("reason", "")[:300]}'
        if rec is None or rec.get('status') != 'done':
            if run.dry_run and planned_now:
                return None, [], "planned by this invocation's admit stage"
            raise Refusal(f'no done admit record admit_{key} under {run.out_root} (status '
                          f'{None if rec is None else rec.get("status")}) - run --stages admit first or give --sublst-jobs')
        sha = (rec.get('meta') or {}).get('plan_sha1') or {}
        for o in rec.get('outputs', []):
            if o.get('class') != 'sublst_job_plan':
                continue
            p = o['path']
            if not os.path.isfile(p) or hashlib.sha1(open(p, 'rb').read()).hexdigest() != sha.get(p):
                raise Refusal(f'the admit plan {p} changed or vanished since its record (sha1) - re-run --stages admit')
            files.append((p, deepflow_root() if p.endswith('_deepflow.txt') else cube_root()))
        how = f'admit record {record.record_path(run.out_root, run.step, f"admit_{key}")}'
    jobs, order = {}, []
    for path, root in files:
        for line in _read_jobs(path):
            c, sb, S = line.split()
            if not c.startswith(lst + '_'):
                continue
            if (c, sb) not in jobs:
                jobs[(c, sb)] = ([], root)
                order.append((c, sb))
            if S not in jobs[(c, sb)][0]:
                jobs[(c, sb)][0].append(S)
    return [(c, sb, sorted(jobs[(c, sb)][0]), jobs[(c, sb)][1]) for c, sb in order], [p for p, _ in files], how


def stack_inputs(lst, cell, sb, croot):
    cubes = [os.path.join(croot, cell, f'{cell}_{sb}_{S}_flowcube.fits') for S in ('I', 'V')]
    return cubes, master_path(lst, sb), frozen_weights(lst, sb)


def _stack_unit(run, opts, root, lst, cell, sb, stokes, croot, plan_files):
    from exopipe.phase3.core import exclusions
    s = _sec()
    night = cell.split('_', 1)[1]
    name = stack_unit_name(lst, night, sb)
    odir = os.path.join(root, lst, cell, sb)
    cubes, master, weights = stack_inputs(lst, cell, sb, croot)
    pins, _ = pins_ok()
    inputs = cubes + [master, weights, script('stack')] + pins + list(plan_files)
    if resume_skip(run, opts, name, [p for p in inputs if os.path.exists(p)], [f'{night}/{sb}']):
        return
    with run.unit(name, lst=lst, cell=cell, subband=sb, stokes=stokes, cube_root=croot, stage='stack') as u:
        miss = [p for p in cubes + [master, weights] if not os.path.exists(p)]
        if miss:
            u.refuse(f'G-S04: REFUSED: missing input {miss[0]} ({len(miss)} missing)')
        if exclusions.load().is_excluded(lst, sb, night):
            u.refuse(f'G-S04: REFUSED: {lst} {sb} {night} is excluded in data_decisions.csv (core.exclusions)')
        u.set_admitted([f'{night}/{sb}'])
        for p in inputs:
            u.add_input(p)
        clear_earlier(run, opts, u, [odir], f'{cell} {sb}')
        for S in stokes:
            u.plan(stack_products(root, lst, cell, sb, S)['admission'], 'sublst_admission')
        if 'I' in stokes:
            u.plan(stack_products(root, lst, cell, sb, 'I')['fluxdc'], 'sublst_fluxdc_record')
        env = {'SUBLST_FLUXDC_WAIT_S': s.FLUXDC_WAIT_S}
        for S in ('I', 'V'):
            if S not in stokes:
                continue
            P = stack_products(root, lst, cell, sb, S)
            st, why, _ = run_engine(u, 'stack', 'stack', ['--cube-root', croot, '--cell', cell, '--subband', sb, '--stokes',
                                                         S, '--out', os.path.join(root, lst)],
                                    P['log'], rf'^SUBLST {re.escape(cell)} {sb} {S} ', env=env, unbuffered=True)
            say(f'  {name} {S}: {st}: {why[:160]}')
            if st == 'error':
                raise RuntimeError(f'stacker {cell} {sb} {S}: {why} (log {P["log"]})')
            if st == 'refused':
                u.refuse_item(P['admission'], why, gate='G-S04 stacker')
                if S == 'I':
                    u.refuse_item(P['fluxdc'], why, gate='G-S04 stacker')
                continue
            nwin = int(re.search(r'windows=(\d+)', why).group(1))
            wins = sorted(glob.glob(os.path.join(P['dir'], f'sublst_{cell}_{sb}_{S}_w[0-9].fits')))
            u.check(f'stack.windows_written_{S}', len(wins) == nwin, value=len(wins),
                    detail=f'{len(wins)} window files, engine line: {why[:200]}')
            for w in wins:
                u.add_output(w, 'sublst_window', master=master)
                u.add_output(w[:-5] + '_nframes.fits', 'sublst_window_nframes')
                u.add_output(w[:-5] + '_pb.fits', 'sublst_window_pb')
            u.add_output(P['admission'], 'sublst_admission')
            if S == 'I':
                u.add_output(P['fluxdc'], 'sublst_fluxdc_record', master=master)


def stage_stack(run, opts, sel, root, planned_now=False, lst_test=True):
    s = _sec()
    bad = 0
    for lst in sel.lsts:
        key = sel.admit_key(lst)
        try:
            jobs, files, how = load_plan(run, opts, lst, key, planned_now)
        except Refusal as e:
            if run.dry_run:
                say(f'[sublst stack {lst}] WOULD REFUSE: {e}')
                bad += 1
                continue
            with run.unit(f'stack_{key}', lst=lst, stage='stack') as u:
                u.refuse(str(e))
            continue
        if jobs is None:
            say(f'[sublst stack {lst}] jobs: {how}')
            continue
        want = set(sel.cells_of(lst)) if (lst in sel.explicit or sel.nights is not None or sel.excluded) else None
        units = [(c, sb, [S for S in st if S in sel.stokes], cr) for c, sb, st, cr in jobs
                 if (want is None or c in want) and sb in sel.subbands and sel.scope_ok(c, sb)]
        units = [x for x in units if x[2]]
        if not units and not run.dry_run:
            with run.unit(f'stack_{key}', lst=lst, stage='stack') as u:
                raise NoInput(f'no stack job of the plan in this selection ({how})')
            continue
        if run.dry_run:
            say(f'[sublst stack {lst}] {len(units)} units (of {len(jobs)} planned (cell, subband) jobs; {how})')
            try:
                py = interpreter('stack')
            except Refusal as e:
                py = None
                say(f'    WOULD REFUSE every unit: {e}')
                bad += 1
            n_miss = n_old = 0
            for i, (c, sb, st, cr) in enumerate(units):
                cubes, master, weights = stack_inputs(lst, c, sb, cr)
                miss = [p for p in cubes + [master, weights] if not os.path.exists(p)]
                old = earlier_products([os.path.join(root, lst, c, sb)])
                n_miss += bool(miss)
                n_old += bool(old) and not (opts.fresh and opts.supersede)
                if i < 3:
                    say(f'    {stack_unit_name(lst, c.split("_", 1)[1], sb)} Stokes {",".join(st)}: cube root {cr}; '
                        f'{"missing " + str(miss) if miss else "inputs found"}'
                        + (f'; earlier products {old[0]}' if old else ''))
                    say(f'      {py or "<site PYTHON missing>"} -u {script("stack")} --cube-root {cr} --cell {c} '
                        f'--subband {sb} --stokes {st[0]} --out {os.path.join(root, lst)}')
            if len(units) > 3:
                say(f'    ... {len(units) - 3} more units')
            if n_miss:
                say(f'    WOULD REFUSE (G-S04): {n_miss} unit(s) with a missing cube / master / frozen record')
            if n_old:
                say(f'    WOULD REFUSE (G-S01): {n_old} unit(s) with earlier products under {root} (no --fresh --supersede)')
            bad += bool(n_miss or n_old)
            continue
        pool_map(lambda x: _stack_unit(run, opts, root, lst, x[0], x[1], x[2], x[3], files), units,
                 opts.workers or s.PARALLEL['stack'])
        if lst_test and any(u.name.startswith(f'stack_{lst}_') and u.status == 'done' for u in run.units):
            lst_test_unit(run, lst, root, files)
    return bad


def lst_test_unit(run, lst, root, plan_files=()):
    """Production content test (a) on <root>/<LST> (class sublst_window_sample): every window's header census and windows =
    frames x stated factors on a seeded sample, re-derived from the cubes (sublst_reserve_test_windows.py in place)."""
    deepflow = [p for p in plan_files if str(p).endswith('_deepflow.txt')]
    with run.unit(f'test-windows_{lst}', lst=lst, stage='test-windows') as u:
        u.add_input(script('test_windows'))
        u.add_output(os.path.join(root, lst), 'sublst_window_sample', lst=lst,
                     deepflow_jobs=deepflow[0] if deepflow else None)
        say(f'  test-windows_{lst}: ' + '; '.join(f"{t['name']} {'PASS' if t['passed'] else 'FAIL'} "
                                                  f"{str(t.get('detail'))[:160]}" for t in u.tests))


# =====================================================================================================================
# per-hour interface (frames --tier sublst imports this; keep the signature stable)
# =====================================================================================================================

def stack_night(run, lst, night, sublst_root=None, subbands=None, stokes=('I', 'V'), workers=None, fresh=False,
                supersede=None, resume=False):
    """The window stacks of ONE night of one LST hour (DESIGN 12.1 per-hour work): the night's admit unit
    (admit_<LST>_<night>: the job-list rule of nocat:82-186, plan files under <run.out_root>/_plans/) and one stack unit per
    admitted subband (stack_<LST>_<night>_<sb>: clv15/phase3_sublst_stack.py in place, Stokes I then V). Products go to
    <sublst_root or run.out_root>/<LST>/<LST>_<night>/<sb>/<S>/ (the production layout), records under the caller's step.

    run          the caller's record.Run (its step names the records; its dry_run prints the plan and writes nothing)
    lst, night   '09h', '2024-12-29'
    sublst_root  the sub-LST product root (default run.out_root); must lie under run.out_root (supersede moves)
    subbands     subset of instrument.ALL_SUBBANDS (default all); stokes ('I', 'V') or one of them
    workers      stack units at once (default masters_sublst.PARALLEL['stack'])
    fresh, supersede, resume   as the common flags (--fresh --supersede TAG moves earlier products; --resume by record)
    Returns {'admit_unit', 'stack_units', 'jobs', 'left_out', 'would_refuse'} ('jobs'/'left_out' from the admit record;
    'would_refuse' counts dry-run refusals)."""
    from exopipe.phase3.core import cells as CL
    CL.parse_epoch(f'{lst}_{night}')
    clear_p3_env()
    opts = Opts(fresh=fresh, supersede=supersede, resume=resume, workers=workers)
    sel = Selection.one_night(lst, night, subbands, stokes)
    root = sublst_root or run.out_root
    bad = stage_admit(run, opts, sel)
    bad += stage_stack(run, opts, sel, root, planned_now=True, lst_test=False)
    key = sel.admit_key(lst)
    rec = None if run.dry_run else record.load_record(run.out_root, run.step, f'admit_{key}')
    jobs = []
    if rec and rec.get('status') == 'done':
        for o in rec.get('outputs', []):
            if o.get('class') == 'sublst_job_plan' and os.path.isfile(o['path']):
                jobs += _read_jobs(o['path'])
    return {'admit_unit': f'admit_{key}',
            'stack_units': [u.name for u in run.units if u.name.startswith(f'stack_{lst}_{night}_')],
            'jobs': jobs, 'left_out': ((rec or {}).get('meta') or {}).get('left_out', []), 'would_refuse': bad}


# =====================================================================================================================
# stage: loo  (clv15/phase3_sublst_loo.py in place; nocat:225-234)
# =====================================================================================================================

def stack_gate(run, lst, sb, jobs):
    """G-S05: every planned stack job of this subband has a done stack record (or a refused one when
    masters_sublst.LOO_ADMITS_REFUSED_STACKS) - production stops the LST otherwise (nocat:208-218)."""
    ok_status = ('done', 'refused') if _sec().LOO_ADMITS_REFUSED_STACKS else ('done',)
    bad, refused = [], []
    for c, jsb, _, _ in jobs:
        if jsb != sb:
            continue
        rec = record.load_record(run.out_root, run.step, stack_unit_name(lst, c.split('_', 1)[1], sb))
        status = rec.get('status') if rec else 'missing'
        if status == 'refused':
            refused.append(c)
        if status not in ok_status:
            bad.append(f'{c} {sb} ({status})')
    return bad, refused


def _loo_unit(run, opts, root, lst, sb, stokes, jobs, plan_files):
    name = f'loo_{lst}_{sb}'
    sroot = os.path.join(root, lst)
    mroot = os.path.join(root, f'{lst}_masters')
    weights = frozen_weights(lst, sb)
    pins, _ = pins_ok()
    stacks = sorted(p for S in stokes for p in glob.glob(os.path.join(sroot, f'{lst}_*', sb, S,
                                                                      f'sublst_{lst}_*_{sb}_{S}_w[0-9].fits')))
    nights = sorted({c.split('_', 1)[1] for c, jsb, _, _ in jobs if jsb == sb})
    inputs = stacks + [weights, script('loo'), script('jupmask')] + pins + list(plan_files)
    if resume_skip(run, opts, name, [p for p in inputs if os.path.exists(p)], [f'{n}/{sb}' for n in nights]):
        return
    with run.unit(name, lst=lst, subband=sb, stokes=stokes, stage='loo') as u:
        bad, refused = stack_gate(run, lst, sb, jobs)
        if bad:
            u.refuse(f'G-S05: {len(bad)} planned stack job(s) of {lst} {sb} not complete: {bad[:6]} (loo starts only when '
                     f'every job of the plan is done; masters_sublst.LOO_ADMITS_REFUSED_STACKS='
                     f'{_sec().LOO_ADMITS_REFUSED_STACKS})')
        if refused:
            u.note(f'stack jobs refused (admitted by LOO_ADMITS_REFUSED_STACKS): {refused}')
        n_adm = len(glob.glob(os.path.join(sroot, f'{lst}_*', sb, '[IV]', f'admission_{lst}_*_{sb}_*.csv')))
        n_jobs = sum(len(st) for c, jsb, st, _ in jobs if jsb == sb and c not in refused)
        u.check('loo.admission_files', n_adm >= n_jobs, value=n_adm,
                detail=f'{n_adm} admission files for {n_jobs} done (night, Stokes) jobs (nocat:210)')
        from exopipe.phase3.core import exclusions
        ex = exclusions.load()
        u.set_admitted([f'{n}/{sb}' for n in nights if not ex.is_excluded(lst, sb, n)])
        for p in inputs:
            u.add_input(p)
        if not stacks:
            raise NoInput(f'no window stacks of {lst} {sb} under {sroot}')
        clear_earlier(run, opts, u, [os.path.join(mroot, sb)] +
                      sorted(glob.glob(os.path.join(sroot, f'{lst}_*', sb, '[IV]', '*_resid_loo.fits'))), f'{lst} {sb} loo')
        for S in stokes:
            log = os.path.join(sroot, 'logs', f'loo_{lst}_{sb}_{S}.log')
            st, why, lines = run_engine(u, 'loo', 'loo', ['--root', sroot, '--master-root', mroot, '--lst', lst,
                                                          '--subband', sb, '--stokes', S],
                                        log, rf'^SUBLST-LOO {lst} {sb} {S} windows=', unbuffered=True)
            say(f'  {name} {S}: {st}: {why[:160]}')
            for l in lines:
                if l.startswith(('REFUSED-WINDOW', 'REFUSED-RESIDUAL', 'REFUSED-NIGHT', 'STALE-RESIDUAL')):
                    u.note(f'{S}: {l[:300]}')
            if st == 'error':
                raise RuntimeError(f'loo {lst} {sb} {S}: {why} (log {log})')
            if st == 'refused':
                u.refuse_item(os.path.join(mroot, sb, S), why, gate='G-S06 loo')
                continue
            m = re.search(r'windows=(\d+) residuals=(\d+) refused_night=(\d+)', why)
            nwin, nres = int(m.group(1)), int(m.group(2))
            masters = sorted(glob.glob(os.path.join(mroot, sb, S, f'sublst_master_{lst}_{sb}_{S}_w[0-9].fits')))
            resid = sorted(glob.glob(os.path.join(sroot, f'{lst}_*', sb, S, f'sublst_{lst}_*_{sb}_{S}_w[0-9]_resid_loo.fits')))
            u.check(f'loo.masters_written_{S}', len(masters) == nwin, value=len(masters), detail=why[:200])
            u.check(f'loo.residuals_written_{S}', len(resid) == nres, value=len(resid), detail=why[:200])
            for p in masters:
                u.add_output(p, 'sublst_window_master', lst=lst, subband=sb)
                u.add_output(p[:-5] + '_weights.csv', 'csv_table', required_columns=('night', 'weight'), min_rows=2)
            for p in resid:
                u.add_output(p, 'sublst_loo_residual', master_root=mroot)


def stage_loo(run, opts, sel, root, planned_now=False, stacks_now=False):
    """The loo engine builds each window Master from EVERY night's stack on disk (it globs <LST>_*), so G-S05 gates on the
    LST's plan (admit_<LST>, or --sublst-jobs), never on a one-night plan."""
    bad = 0
    for lst in sel.lsts:
        key = lst
        try:
            jobs, files, how = load_plan(run, opts, lst, key, planned_now and sel.admit_key(lst) == lst)
        except Refusal as e:
            if run.dry_run:
                say(f'[sublst loo {lst}] WOULD REFUSE: {e}')
                bad += 1
                continue
            with run.unit(f'loo_{lst}', lst=lst, stage='loo') as u:
                u.refuse(str(e))
            continue
        if run.dry_run:
            say(f'[sublst loo {lst}] {len(sel.subbands)} units (one per subband, Stokes {",".join(sel.stokes)}); plan: {how}')
            try:
                say(f'    {interpreter("loo")} -u {script("loo")} --root {os.path.join(root, lst)} --master-root '
                    f'{os.path.join(root, lst + "_masters")} --lst {lst} --subband <sb> --stokes <S>')
            except Refusal as e:
                say(f'    WOULD REFUSE every unit: {e}')
                bad += 1
            if jobs is not None and stacks_now:
                say(f'    G-S05 checked when the unit starts: the stack records come from this invocation\'s stack stage')
            elif jobs is not None:
                for sb in sel.subbands:
                    gate, _ = stack_gate(run, lst, sb, jobs)
                    nst = len(glob.glob(os.path.join(root, lst, f'{lst}_*', sb, '[IV]', f'sublst_{lst}_*_w[0-9].fits')))
                    bad += bool(gate)
                    say(f'    loo_{lst}_{sb}: {nst} stacks under {root}; ' +
                        (f'WOULD REFUSE (G-S05): {len(gate)} stack job(s) without a done record, e.g. {gate[:2]}'
                         if gate else 'stack records complete'))
            continue
        pool_map(lambda sb: _loo_unit(run, opts, root, lst, sb, sel.stokes, jobs, files), list(sel.subbands),
                 opts.workers or _sec().PARALLEL['loo'])
    return bad


# =====================================================================================================================
# stage: sources  (pcheck of sublst_catalog_publish.sh:51-65 in place; sublst_parent_catalogs.py when stale)
# =====================================================================================================================

_GI_LOCK = threading.Lock()


def parents_check(lst, scope, pdir, coadd):
    """'' or the refusal of production's own conformance gate (exopipe_ops/gate_inputs.parents, in place) for one scope,
    pointed at this run's parent and coadd directories (the pcheck heredoc sets P3_PARENT_DIR / P3_COADD_ROOT, catpub:53)."""
    from exopipe.phase3.core import env
    env.legacy()
    with _GI_LOCK:
        import gate_inputs as G
        G.PARENT_DIR, G.COADD = pdir, coadd
        G._PAR.clear()
        G._MD5.clear()
        try:
            G.parents(lst, scope)
            return ''
        except Exception as e:
            return f'{type(e).__name__}: {e}'


def stage_sources(run, opts, sel, root, planned_now=False):
    bad = 0
    for lst in sel.lsts:
        name = f'sources_{lst}'
        coadd = os.path.join(root, f'coadd_{lst}')
        pdir = os.path.join(root, f'parent_catalogs_{lst}')
        scopes = sorted(os.path.basename(d)[len(lst) + 1:] for d in glob.glob(os.path.join(coadd, f'{lst}_*')))
        env_set = p3_switches()
        stale = {} if env_set else {sc: parents_check(lst, sc, pdir, coadd) for sc in scopes}
        stale = {k: v for k, v in stale.items() if v}
        rebuild_ok = os.path.realpath(root) == os.path.realpath(sublst_root())
        nproc = cfg().masters_sources.SUBLST_NPROC
        if run.dry_run:
            say(f'[sublst sources {lst}] {len(scopes)} scopes under {coadd}; parent check: {len(stale)} refused'
                + (f' ({", ".join(stale)})' if stale else ''))
            if env_set:
                say(f'    WOULD REFUSE: P3_* variables set in this process: {env_set}')
            if not scopes:
                say('    no coadd_<LST> scopes yet: ' + ("written by this invocation's prep stage" if planned_now
                                                          else 'NO-INPUT - run --stages prep first'))
            if stale and not rebuild_ok:
                say(f'    WOULD REFUSE (G-S10): rebuild needed but sublst_parent_catalogs.py writes only to {sublst_root()}')
            elif stale:
                say(f'      {interpreter("sources")} {script("parentcat")} {lst} {nproc}  (cwd {cfg().paths.OPS_ROOT})')
            bad += bool(env_set) or (not scopes and not planned_now) or (bool(stale) and not rebuild_ok)
            continue
        with run.unit(name, lst=lst, stage='sources') as u:
            if env_set:
                u.refuse(f'P3_* variables set in this process: {env_set} (gate_inputs reads them at import)')
            if not scopes:
                raise NoInput(f'no scopes under {coadd} (run --stages prep first)')
            masters = {sc: os.path.join(coadd, f'{lst}_{sc}', 'coadd_final', f'I_deep_Taper_R075_{sc}.fits') for sc in scopes}
            for p in masters.values():
                u.add_input(p)
            if _CLEARED_P3:
                u.note(f'inherited environment variables removed before the in-place imports: {_CLEARED_P3}')
            u.note(f'PARENT-CHECK {lst}: {len(scopes)} subbands, {len(stale)} refused'
                   + (' (' + ' '.join(stale) + ')' if stale else ''))
            if stale:
                if not rebuild_ok:
                    u.refuse(f'G-S10: parent catalogues of {lst} refused by the conformance check '
                             f'({"; ".join(list(stale.values())[:3])}); the rebuild (sublst_parent_catalogs.py) writes only '
                             f'to its literal root {sublst_root()} (and scratch /fast/gh/main/phase3), so it runs only with '
                             f'--out {sublst_root()} (after cutover; an absorption-time edit otherwise)')
                u.add_input(script('parentcat'))
                log = os.path.join(root, lst, 'logs', f'parentcat_{lst}.log')
                st, why, lines = run_engine(u, 'parentcat', 'sources', [lst, nproc], log, rf'^PARENTCAT-DONE {lst}:',
                                            cwd=cfg().paths.OPS_ROOT)
                for l in lines:
                    if l.startswith('PARENTCAT ') and 'NO-MASTER' in l:
                        u.note(l)
                fin = [l for l in lines if l.startswith('PARENTCAT-')]
                if fin and fin[-1].startswith('PARENTCAT-REFUSED'):
                    u.refuse(f'G-S10: {fin[-1]}')
                if st != 'done':
                    raise RuntimeError(f'sublst_parent_catalogs.py {lst}: {why} (log {log})')
                u.note(why)
                stale = {sc: parents_check(lst, sc, pdir, coadd) for sc in scopes}
                stale = {k: v for k, v in stale.items() if v}
                if stale:
                    u.refuse(f'G-S10: parent catalogues of {lst} still refused after the rebuild: {stale}')
            for sc in scopes:
                u.add_output(os.path.join(pdir, f'master_R075_{lst}_{sc}.gaul.csv'), 'master_catalog', master=masters[sc])
            say(f'  {name}: {len(scopes)} parent catalogues current')
    return bad


# =====================================================================================================================
# the tier interface (steps/masters.py dispatches here)
# =====================================================================================================================

def add_arguments(p):
    g = p.add_argument_group('tier sublst (masters --tier sublst)')
    g.add_argument('--sublst-worklist', metavar='FILE',
                   help="production worklist: 'LST NIGHT' lines (# comments); its cells are the nights of each LST")
    g.add_argument('--sublst-jobs', metavar='FILE',
                   help="stack/loo: take the job plan from FILE (production sublst_jobs_<LST>.txt) instead of this run's admit "
                        "record (recorded with its fingerprint); admit: compare the computed plan with FILE")
    g.add_argument('--sublst-deepflow-jobs', metavar='FILE',
                   help='with --sublst-jobs: the jobs that read the deep-Master re-flow cubes (sublst_jobs_<LST>_deepflow.txt)')


def default_out(args):
    return sublst_root()


def run(args, run):
    """Run the selected sub-LST stages. Returns the exit code in --dry-run (0 = no gate would refuse, 2 = some would);
    None otherwise (the CLI derives the exit code from the unit records)."""
    stages = resolve_stages(args, TIER)
    owned = [x for x in stages if x in OWNED_ELSEWHERE]
    if owned:
        raise SystemExit('masters --tier sublst: ' + '; '.join(f'stage {x} is {OWNED_ELSEWHERE[x]}' for x in owned))
    opts = Opts.from_args(args)
    sel = Selection.from_args(args)
    root = run.out_root
    clear_p3_env()
    say(f'masters --tier sublst: stages {",".join(stages)}; LSTs {",".join(sel.lsts) or "-"}; out {root}'
        + ('; DRY RUN (nothing written)' if run.dry_run else ''))
    viol = mirror_violations()
    if viol:
        msg = 'config differs from the in-place code it mirrors: ' + '; '.join(viol)
        if run.dry_run:
            say(f'WOULD REFUSE: {msg}')
            return record.EXIT_REFUSED
        raise Refusal(msg)
    if not sel.lsts:
        raise SystemExit('masters --tier sublst: give --lsts, --cells <LST>_<night>[_<subband>] or --sublst-worklist')
    bad = 0
    for st in stages:
        if st == 'prep':
            bad += stage_prep(run, opts, sel, root)
        elif st == 'admit':
            bad += stage_admit(run, opts, sel)
        elif st == 'stack':
            bad += stage_stack(run, opts, sel, root, planned_now='admit' in stages)
        elif st == 'loo':
            bad += stage_loo(run, opts, sel, root, planned_now='admit' in stages, stacks_now='stack' in stages)
        elif st == 'sources':
            bad += stage_sources(run, opts, sel, root, planned_now='prep' in stages)
    if run.dry_run:
        return record.EXIT_REFUSED if bad else record.EXIT_DONE
    return None
