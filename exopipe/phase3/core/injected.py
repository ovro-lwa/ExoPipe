"""Injected runs write no production product - structurally (plan/CUTOVER_READINESS.md, 'INJECTED RUNS WRITE NO PRODUCTION
PRODUCT, STRUCTURALLY').

The incident (ledger INCIDENT-INJECTED-SEARCH-WROTE-PRODUCTION-PHOTOMETRY-CUTOUTS-HS-PSC-EVENT-UNREPRODUCIBLE-20261024): the
snapshot search's --inject completeness runs of 2026-09-13/14 (04h 2024-12-29) wrote production photometry cutouts into
/lustre/pipeline/exopipe/phase3/photometry (the engine's ROOT_DEFAULT), and the forced pages 8775/8776/8785/8786 served one
carrying a synthetic 15-sigma source (HS Psc 05:48:42). Production's interim guard (clv15 721cf07): P3_SNAP_INJECTED=1 makes
write_cutouts raise. The port: "an injected or completeness run carries one flag that every writer checks. It refuses any output
root that is not scratch: archive tree, photometry root, served run, data_decisions."

ONE FLAG: the environment variable EXOPIPE3_INJECTED, whose value says why the run is injected, in every process of the run.
`exopipe3 <step>` sets it when the step runs with --inject MANIFEST (search) or --injected (any step whose inputs carry synthetic
sources: a completeness battery, a step run on an injected run's products), and a process started by an injected run inherits it
(sticky: no flag clears it). `exopipe3 run` is never an injected run itself: `run submit --injected` forwards the flag to every
step command on every node; `run submit` refuses a flag it only inherits (the steps on other nodes would not get it), and inside
an injected run's processes it cannot run at all (its job records live under runner.JOBS_ROOT, which the guard refuses).
core.legacy.run passes the flag to every in-place child, with P3_SNAP_INJECTED=1 set after the caller's env (its P3_* clearing
would drop it). P3_SNAP_INJECTED binds only an engine that honors it: production's run_search resets it from its own --inject
argument (clv15 721cf07) and regen_cutouts always removes it, so for --injected runs it does not reach the cutout writer - the
cutout protection is P3_SNAP_CUTOUTS=0, a cutout root under --out (checked) and the write guard.

What the flag does (cli calls begin() before the step runs):
  1. paths.PRODUCTION_WRITES is False in the process (begin(); in a child, the config load): every writer that consults it - the
     --out guard, archive G2, quarantine, install, calibrate tables, the snapshot cutout root, the deep search's archive copies,
     the flow locks - writes under the run's own --out or refuses, after cutover too.
  2. The run's --out must be given (every step's default --out is a production root) and, like --work, lie under no refused
     root (refused_roots(): paths.PRODUCTION_ROOTS, every other root of the paths section - the archive tree, the Phase-1 and
     Phase-2 data, the working trees, the served-run roots, ops state, clv15 with data_decisions.csv and the fix ledger, the
     production package's reference tables -, every section's own production roots (PRODUCTION_ROOTS, EXTRA_PRODUCTION_ROOTS,
     OUT_ROOT, PRODUCTION_OUT, PHOT_ROOT, CUTOUT_ROOT, TIER_FORCED_ROOTS: the photometry roots, the served page roots) and the
     step's default --out). Else exit 2.
  3. The write guard of core/guard.py, labelled INJECTED-RUN GUARD, in this process and every Python process it starts: a write
     anywhere but --out, --work, the node's scratch root, /tmp and /dev/shm raises InjectedWriteRefused (a PermissionError)
     before anything is written.
  4. Each production writer checks the flag by name before it writes (check() / refuse(), a Refusal - exit 2 - naming the
     writer; it also covers what the write guard cannot see: a cp / rsync / shell child, a C library's own I/O):
       archive tree   lib/archive Ctx.guard_writes (every apply: push, sync, quarantine, retire, rollback, supersede)
       photometry root lib/search/tier_snapshot cutout_root (P3_SNAP_CUTOUT_ROOT under --out, checked) with P3_SNAP_CUTOUTS=0
                      for the engines: no cutout is cut from a cube carrying synthetic sources; `photometry` checks every
                      root its cutouts / cutmeasure stages write, flag or default (lib/photometry/steps_cutouts
                      gate_write_roots)
       served run     `serve` and `jupiter serve` refuse an injected run outright (servers, registry, hub, journals, restarts,
                      parity ports)
       data_decisions check() refuses paths.DATA_DECISIONS (and the fix ledger, the run-pin file) whatever else is allowed
     and every DESTINATION flag of a step (DEST_FLAGS: a staging root, a review root, a place target, an emitted list, the
     archive roots ...) must be scratch before the step starts (check_flags; a remote [HOST:]PATH never is) - the write guard
     lets os.makedirs pass on an existing directory and cannot see the rsync / cp that follows.
  5. Records: run and unit records carry 'injected' (core/record.injected(rec)), and the run writes a start record
     <out>/_records/_injected/<run_id>.json before anything else (core/record.mark_injected). An ORDINARY run then refuses
     (check_flags) an --out holding one - an ordinary --resume never adopts an injected run's units, and core/record.
     resume_check never adopts across the injected / ordinary line - and every SOURCE flag naming another run's products
     under one (SOURCE_FLAGS: serve --source / --carry-from, archive --attempt / --staging-root / --workdir, install --gen /
     --rebuild-root, pages --from-run, search --replay-attempt, photometry --src-run / --cutout-root / --stage1, forced
     --deep-eval-run / --wide-runs / --wide-jobs (each file of its comma list) / --wide-cutout-root / --wide-bursts /
     --na-detections, catalog --inc-source run: / --omit-runs (each run of its comma list or @FILE, a bare name under
     paths.SNAPSHOT_REVIEW_ROOT, as the deep-omission stage reads it) / --omit-refusals, calibrate --served-run; a source on
     another host is checked over ssh).

Python 3.6+; imports nothing from exopipe at module level.
"""
import os

ENV = 'EXOPIPE3_INJECTED'
ENGINE_ENV = {'P3_SNAP_INJECTED': '1'}      # clv15 721cf07: write_cutouts raises under it (run_search resets it: see above)
LABEL = 'INJECTED-RUN GUARD'
LOG_NAME = '_injected_guard_refusals.log'   # under <out>/_records/
# config keys of a section that name production roots (scalar, tuple or dict of paths); 'paths' is taken whole
ROOT_KEYS = ('PRODUCTION_ROOTS', 'EXTRA_PRODUCTION_ROOTS', 'OUT_ROOT', 'PRODUCTION_OUT', 'PHOT_ROOT', 'CUTOUT_ROOT',
             'TIER_FORCED_ROOTS')
# files of the paths section whose directory is refused too (Gregg's rulings and exclusions: never written by any run)
RULING_FILES = ('DATA_DECISIONS', 'FIX_LEDGER', 'PHASE1_RUN_PINS')
_ALWAYS_SCRATCH = ('/tmp', '/dev/shm', '/dev/null')     # never a refused root (core/guard.DEFAULT_ALLOWED)


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _refusal(msg):
    from exopipe.phase3.core.record import Refusal
    return Refusal(msg)


# ---- the flag ------------------------------------------------------------------------------------------------------------

def reason(environ=None):
    """Why this process belongs to an injected run (the value of EXOPIPE3_INJECTED), or '' for an ordinary run."""
    return str((os.environ if environ is None else environ).get(ENV) or '').strip()


def active(environ=None):
    return bool(reason(environ))


def requested(args, environ=None):
    """The injected-run reason of a step invocation: the inherited one (sticky), else '--inject <MANIFEST>' (search's
    injection), else '--injected'; '' for an ordinary run."""
    r = reason(environ)
    if r:
        return r
    if getattr(args, 'inject', None):
        return f'--inject {args.inject}'
    if getattr(args, 'injected', False):
        return '--injected'
    return ''


def mark(why, environ=None):
    """Make ``environ`` (default os.environ) an injected run's: EXOPIPE3_INJECTED (an existing reason is kept - sticky) and the
    in-place engines' switch P3_SNAP_INJECTED=1. Returns the reason in force."""
    env = os.environ if environ is None else environ
    if not reason(env):
        env[ENV] = str(why or '').strip() or 'injected'
    env.update(ENGINE_ENV)
    return env[ENV]


def child_env():
    """The variables a child of an injected run must carry (core.legacy.run applies them after clearing P3_*), else {}."""
    return dict({ENV: reason()}, **ENGINE_ENV) if active() else {}


def engine_env():
    """P3_SNAP_INJECTED=1 for an in-place engine's environment in an injected run, else {}."""
    return dict(ENGINE_ENV) if active() else {}


def force_no_production_writes(cfg=None):
    """paths.PRODUCTION_WRITES = False in an injected run's process (it never writes production products, also after cutover).
    Returns True when the flag is active. A protected key: set here directly, never through an override."""
    if not active():
        return False
    C = cfg or _cfg()
    C.paths.PRODUCTION_WRITES = False
    return True


# ---- roots -----------------------------------------------------------------------------------------------------------------

def _paths_in(v):
    if isinstance(v, str):
        if v.startswith('/'):
            yield v.split('{', 1)[0].rstrip('/') if '{' in v else v     # a template: the fixed part before its first field
    elif isinstance(v, (list, tuple, set, frozenset)):
        for x in v:
            yield from _paths_in(x)
    elif isinstance(v, dict):
        for x in v.values():
            yield from _paths_in(x)


def _norm(p):
    return os.path.realpath(p).rstrip('/') or '/'


def _scratch_like(p):
    """True for a root that is (or is under, or holds) the always-scratch roots - never refused (a '{run}' template cut to '/'
    or a step whose default --out is /tmp must not refuse /tmp)."""
    r = _norm(p)
    for s in _ALWAYS_SCRATCH:
        s = _norm(s)
        if r == s or r.startswith(s + '/') or s.startswith(r.rstrip('/') + '/'):
            return True
    return False


_ROOTS_MEMO = {}         # the configured root strings -> their real paths (realpath costs a stat per component on Lustre)


def refused_roots(cfg=None, extra=()):
    """Every root an injected run never writes under - see the module docstring, item 2. ``extra``: more roots (the step's
    default --out). Real paths, de-duplicated, in config order."""
    C = cfg or _cfg()
    out = []
    for k, v in C.paths.items():
        if k in ('PRODUCTION_WRITES', 'PINNED_INPUTS', 'ARCHIVE_RUN'):
            continue
        out.extend(_paths_in(v))
        if k in RULING_FILES and isinstance(v, str):
            out.append(os.path.dirname(v))
    for name, sec in C.sections().items():
        if name in ('paths', 'site'):
            continue
        for k in ROOT_KEYS:
            out.extend(_paths_in(getattr(sec, k, None)))
    out.extend(_paths_in(list(extra or ())))
    key = tuple(out)
    if key not in _ROOTS_MEMO:
        _ROOTS_MEMO[key] = tuple(dict.fromkeys(_norm(p) for p in out if p and not _scratch_like(p)))
    return _ROOTS_MEMO[key]


def refused_root_of(path, cfg=None, extra=()):
    """The refused root that holds ``path`` (realpath; the most specific one, e.g. the served-run root rather than the tree
    above it), or None. paths.DATA_DECISIONS, the fix ledger and the run-pin file are refused by name whatever their
    directory."""
    C = cfg or _cfg()
    real = _norm(path)
    for k in RULING_FILES:
        f = getattr(C.paths, k, None)
        if f and real == _norm(f):
            return f
    hits = [r for r in refused_roots(C, extra) if real == r or real.startswith(r.rstrip('/') + '/')]
    return max(hits, key=len) if hits else None


# ---- the writers' checks ---------------------------------------------------------------------------------------------------

def check(path, writer):
    """Refuse (Refusal, exit 2) when this process belongs to an injected run and ``writer`` is about to write ``path``
    outside scratch: under a refused root, or - while the write guard is installed - outside its allowed roots. Returns the
    path (so a writer can wrap the root it computes)."""
    if not active() or path is None:
        return path
    where = refused_root_of(path)
    if where is not None:
        where = f'under the production root {where}'
    else:
        from exopipe.phase3.core import guard
        if guard.active() and not guard.is_allowed(os.fspath(path)):
            where = f'outside the run\'s scratch roots ({", ".join(guard.allowed_roots())})'
    if where:
        raise _refusal(f'{LABEL}: {writer} would write {path} {where}; this run is injected ({reason()}) and writes no '
                       f'production product (CUTOVER_READINESS; INCIDENT-INJECTED-SEARCH-WROTE-PRODUCTION-PHOTOMETRY-'
                       f'CUTOUTS-HS-PSC-EVENT-UNREPRODUCIBLE-20261024)')
    return path


def check_all(paths, writer):
    """check() for every path of ``paths`` (None entries skipped)."""
    for p in paths or ():
        check(p, writer)
    return paths


def refuse(writer, why):
    """Refuse a writer that has no scratch form at all in an injected run (e.g. `serve`: an injected run is never served)."""
    if active():
        raise _refusal(f'{LABEL}: {writer} refused - {why}; this run is injected ({reason()})')


# ---- the steps' destination and source flags (argparse dest names), checked by cli before the step runs --------------------
# DESTINATIONS: where a step writes outside --out / --work, often through a writer the write guard cannot see (rsync, cp, a
# shell script; os.makedirs of an existing directory is no write). In an injected run each value must be scratch.
DEST_FLAGS = {
    'verify': ('emit_failing',),                                   # the failing list
    'status': ('emit', 'list_out'),                                # item lists
    'followup': ('page_dir',),                                     # the periodicity page
    'photometry': ('stage_root', 'cutout_root', 'tier_out'),       # deep stage: os.makedirs + rsync -aL of the inputs; the
                                                                   # cutout generation the cutouts stage writes (a SOURCE of
                                                                   # measure / cutmeasure: _DEST_WHEN_STAGES); the tier forced
                                                                   # root cutmeasure writes (review F6)
    'archive': ('archive_root', 'ship_root', 'phot_root', 'quar_root', 'qdir', 'stores'),   # applies: moves, copies, rewrites
    'frames': ('snap_stage_root',),                                # snapshot staging: copies + funpack
    'flow': ('snap_stage_root',),                                  # snapshot flow: <staging>/<sb>/*_flowed.fits
    'forced': ('deep_panels_run', 'snap_review_root', 'snap_masters_root'),   # page run, review root, master link tree
    'masters': ('deep_installed_root',),                           # build-stats: its record beside the installed cell tree
    'pages': ('sublst_place_target',),                             # place: os.makedirs + rsync -a --delete
    'jupiter': ('qdir',),                                          # quarantine: a name inside the site cell (an absolute
}                                                                  # path is checked; a relative one stays inside the cell)
_IN_CELL = {('jupiter', 'qdir')}                                   # flags whose relative value is joined to a cell under --out
# a flag that is a destination only when its step runs one of these stages, named in --stages (photometry --cutout-root is what
# the cutouts stage writes and what measure / cutmeasure read; a cutouts stage runs only when named, photometry.DEFAULT_STAGES)
_DEST_WHEN_STAGES = {('photometry', 'cutout_root'): ('cutouts',)}
# SOURCES: another run's products a step reads (a run directory, an attempt, a generation). An ordinary run refuses one that
# lies under an injected run's start record (core/record.injected_runs): its data carry synthetic sources.
SOURCE_FLAGS = {
    'serve': ('source', 'carry_from'),
    'archive': ('attempt', 'staging_root', 'workdir'),
    'install': ('gen', 'rebuild_root'),
    'pages': ('from_run',),
    'search': ('replay_attempt',),
    'photometry': ('src_run', 'cutout_root', 'stage1'),
    'forced': ('deep_eval_run', 'wide_runs', 'wide_jobs', 'wide_cutout_root', 'wide_bursts', 'na_detections'),   # W / N inputs
    'catalog': ('inc_source', 'omit_runs', 'omit_refusals'),   # omit_*: the deep-omission stage (lib/catalog/deep_omission.py) -
                                                               # the served batch runs it counts and the refusal record it reads
    'calibrate': ('served_run',),
}
_KEYWORDS = ('published', 'served', 'none', 'snapshot', 'search')     # flag values that name no path
_RUN_NAME_FLAGS = {'deep_panels_run': 'TRANSIENTS_ROOT',             # a bare name resolves under this paths root (as the step
                   'omit_runs': 'SNAPSHOT_REVIEW_ROOT'}              # resolves it: deep_omission._runs_arg)
_LIST_FLAGS = ('stores', 'omit_runs', 'wide_jobs')   # a comma list of paths (archive --stores: files; catalog --omit-runs: runs;
                                                     # forced --wide-jobs: job lists)
_AT_FILE_FLAGS = ('omit_runs',)                  # '@FILE': the paths FILE lists, one per line ('#' starts a comment)


def _at_file(flag, v):
    """The paths an '@FILE' value lists (deep_omission._runs_arg's reading); an unreadable file refuses - its runs cannot be
    shown not to be an injected run's."""
    try:
        with open(v[1:]) as f:
            return [ln.split('#', 1)[0].strip() for ln in f]
    except OSError as e:
        raise _refusal(f'{LABEL}: --{flag.replace("_", "-")} {v}: the list cannot be read ({e}) - its runs cannot be shown not '
                       f'to be an injected run\'s')


def _values(flag, v):
    if v is None:
        return []
    vals = [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]
    if flag in _AT_FILE_FLAGS and any(x.strip().startswith('@') for x in vals):
        vals = [y for x in vals for y in (_at_file(flag, x.strip()) if x.strip().startswith('@') else [x])]
    elif flag in _LIST_FLAGS:
        vals = [x for y in vals for x in y.split(',')]
    return [x.strip() for x in vals if x.strip() and x.strip() not in _KEYWORDS]


def split_host(v):
    """(host or None, path) of '[HOST:]PATH': a host is a first component that holds ':' and does not start with '/'."""
    head = v.split('/', 1)[0]
    if ':' in head and not v.startswith('/'):
        host, _, path = v.partition(':')
        return host, path
    return None, v


def _this_host(host):
    import socket
    return host in ('localhost', '127.0.0.1') or host.split('.')[0] == socket.gethostname().split('.')[0]


def _fixed(path):
    """The fixed part of a path template: up to the directory holding its first '{field}' ('/g/{cell}' -> '/g')."""
    if '{' not in path:
        return path
    head = path.split('{', 1)[0]
    return head.rstrip('/') if head.endswith('/') else os.path.dirname(head)


def _remote_injected(host, path):
    """core/record.injected_runs of PATH on HOST, over one ssh call: [(root, start record)]. A host that cannot be reached
    refuses - the source cannot be shown not to be an injected run's."""
    import shlex
    import subprocess
    from exopipe.phase3.core import record
    cands, p = [], os.path.normpath(path)
    while True:
        cands.append(p)
        if os.path.dirname(p) == p:
            break
        p = os.path.dirname(p)
    cmd = ('for d in ' + ' '.join(shlex.quote(c) for c in cands) + '; do f=$(ls -1 "$d/_records/' + record.INJECTED_DIR
           + '" 2>/dev/null | head -n 1); [ -n "$f" ] && echo "$d|$f"; done; echo EXOPIPE3_CHECKED')
    try:
        r = subprocess.run(['ssh', '-n', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, cmd],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise _refusal(f'{LABEL}: cannot tell whether {host}:{path} holds an injected run\'s products ({type(e).__name__}: '
                       f'{e}) - refused')
    if 'EXOPIPE3_CHECKED' not in r.stdout:
        raise _refusal(f'{LABEL}: cannot tell whether {host}:{path} holds an injected run\'s products (ssh rc {r.returncode}: '
                       f'{r.stderr.strip()[-200:]}) - refused')
    return [tuple(x.split('|', 1)) for x in r.stdout.splitlines() if '|' in x]


def check_flags(step, args, out, writes=True):
    """The flag checks of one step invocation (cli, before the step runs; Refusal = exit 2):
    - an INJECTED run: every destination flag of the step (DEST_FLAGS; one of _DEST_WHEN_STAGES only when its --stages names
      a stage that writes it) must be scratch - check(); a remote [HOST:]PATH never is;
    - an ORDINARY run: an --out holding an injected run's start record (core/record.injected_runs, the --out or an ancestor)
      refuses when the step writes; every source flag (SOURCE_FLAGS) naming a path under one refuses (a source on another
      host is checked over ssh)."""
    from exopipe.phase3.core import record
    if active():
        named = {s.strip() for s in str(getattr(args, 'stages', None) or '').split(',') if s.strip()}
        for flag in DEST_FLAGS.get(step, ()):
            when = _DEST_WHEN_STAGES.get((step, flag))
            if when and not named & set(when):
                continue
            what = f'{step} --{flag.replace("_", "-")}'
            for v in _values(flag, getattr(args, flag, None)):
                if (step, flag) in _IN_CELL and not os.path.isabs(v):
                    continue
                host, path = split_host(v)
                if host:
                    raise _refusal(f'{LABEL}: {what} {v} is on another host - an injected run writes nowhere but its scratch '
                                   f'roots; this run is injected ({reason()})')
                if flag in _RUN_NAME_FLAGS and not os.path.isabs(path):
                    path = os.path.join(getattr(_cfg().paths, _RUN_NAME_FLAGS[flag]), path)
                check(_fixed(path), what)
        return
    if writes and out:
        hit = record.injected_runs(out)
        if hit:
            root, f, why = hit[0]
            raise _refusal(f'{LABEL}: --out {out} holds an injected run\'s products ({f}: {why}) - an ordinary run never writes '
                           f'into, reads from or resumes an injected run\'s --out; give another --out, or run with --injected')
    for flag in SOURCE_FLAGS.get(step, ()):
        what = f'{step} --{flag.replace("_", "-")}'
        for v in _values(flag, getattr(args, flag, None)):
            if v.startswith('run:'):
                v = v[4:]
            host, path = split_host(v)
            if not host and flag in _RUN_NAME_FLAGS and not os.path.isabs(path):
                path = os.path.join(getattr(_cfg().paths, _RUN_NAME_FLAGS[flag]), path)
            path = _fixed(path)
            if host and not _this_host(host):
                hit = [(h, f, 'injected') for h, f in _remote_injected(host, path)]
            else:
                hit = record.injected_runs(path)
            if hit:
                root, f, why = hit[0]
                raise _refusal(f'{LABEL}: {what} {v}: {root} holds an injected run\'s products ({f}: {why}) - their data carry '
                               f'synthetic sources, and an ordinary run never reads them (core/injected.py)')


# ---- the run ----------------------------------------------------------------------------------------------------------------

def begin(why, out, explicit_out=True, work=None, default_out=None, writes=True):
    """Make this process an injected run before its step runs (cli): the flag (sticky), paths.PRODUCTION_WRITES False, --out
    and --work checked against refused_roots (with the step's default --out), and the write guard installed here and exported
    to every Python child. Raises Refusal on a missing or non-scratch --out / --work. Returns the guard's allowed roots."""
    why = mark(why)
    C = _cfg()
    C.paths.PRODUCTION_WRITES = False
    if writes and not explicit_out:
        raise _refusal(f'{LABEL}: an injected run ({why}) needs an explicit --out on scratch (a step\'s default --out is '
                       f'where it writes its production products: {default_out})')
    extra = () if (default_out is None or _scratch_like(default_out)) else (default_out,)
    for flag, d in (('--out', out), ('--work', work)):
        if d:
            root = refused_root_of(d, C, extra)
            if root is not None:
                raise _refusal(f'{LABEL}: {flag} {d} is under {root}, which an injected run ({why}) never writes '
                               f'(core/injected.refused_roots) - give a scratch {flag}')
    from exopipe.phase3.core import fitsio
    try:
        scratch = fitsio.scratch_root(None)
    except Exception:                  # noqa: BLE001 - no scratch root on this host: not allowed, nothing else changes
        scratch = None
    shared = [x for x in list(_ALWAYS_SCRATCH) + [os.environ.get('TMPDIR'), scratch, '/'] if x]
    if writes and out and _norm(out) in {_norm(x) for x in shared}:
        raise _refusal(f'{LABEL}: --out {out} is a shared scratch root - an injected run marks its whole --out as injected '
                       f'(core/record.mark_injected), so it needs a directory of its own below it')
    roots = [out] + ([work] if work else [])
    if not work and scratch and refused_root_of(scratch, C) is None:   # node-local scratch (core/fitsio.scratch_dir) is
        roots.append(scratch)                                          # temporary, never a product
    from exopipe.phase3.core import guard
    log = os.path.join(out, '_records', LOG_NAME) if out else None
    allowed = guard.install(roots, log=log, label=LABEL)
    guard.export_env(roots, log=log)
    return allowed
