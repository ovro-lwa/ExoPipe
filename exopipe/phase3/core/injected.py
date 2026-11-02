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
(sticky: no flag clears it; `run submit` forwards --injected to every step command, on every node). core.legacy.run passes it to
every in-place child together with the engines' own switch P3_SNAP_INJECTED=1, which its P3_* clearing would otherwise drop.

What the flag does (cli calls begin() before the step runs):
  1. paths.PRODUCTION_WRITES is False in the process (begin(); in a child, the config load): every writer that consults it - the
     --out guard, archive G2, quarantine, install, calibrate tables, the snapshot cutout root, the deep search's archive copies,
     the flow locks - writes under the run's own --out or refuses, after cutover too.
  2. The run's --out must be given (every step's default --out is a production root) and, like --work, lie under no refused
     root (refused_roots(): paths.PRODUCTION_ROOTS, every other root of the paths section - the archive tree, the Phase-1 and
     Phase-2 data, the working trees, the served-run roots, ops state, clv15 with data_decisions.csv and the fix ledger, the
     production package's reference tables -, every section's own production roots (PRODUCTION_ROOTS, EXTRA_PRODUCTION_ROOTS,
     OUT_ROOT, PRODUCTION_OUT, PHOT_ROOT: the photometry roots, the served page roots) and the step's default --out). Else exit 2.
  3. The write guard of core/guard.py, labelled INJECTED-RUN GUARD, in this process and every Python process it starts: a write
     anywhere but --out, --work, the node's scratch root, /tmp and /dev/shm raises InjectedWriteRefused (a PermissionError)
     before anything is written.
  4. Each production writer checks the flag by name before it writes (check() / refuse(), a Refusal - exit 2 - naming the
     writer; it also covers what the write guard cannot see: a cp / rsync / shell child, a C library's own I/O):
       archive tree   lib/archive Ctx.guard_writes (every apply: push, sync, quarantine, retire, rollback, supersede)
       photometry root lib/search/tier_snapshot cutout_root (P3_SNAP_CUTOUT_ROOT; the engines also get P3_SNAP_CUTOUTS=0 and
                      P3_SNAP_INJECTED=1: no cutout is cut from a cube carrying synthetic sources)
       served run     `serve` refuses an injected run outright (pages, servers, registry, hub, journals, restarts)
       data_decisions check() refuses paths.DATA_DECISIONS (and the fix ledger, the run-pin file) whatever else is allowed
  5. Run and unit records carry 'injected' (core/record.injected(rec)), so a consumer can refuse an injected run's products.

Python 3.6+; imports nothing from exopipe at module level.
"""
import os

ENV = 'EXOPIPE3_INJECTED'
ENGINE_ENV = {'P3_SNAP_INJECTED': '1'}      # clv15 721cf07: phase3_snapshot_search writes no cutout, write_cutouts raises
LABEL = 'INJECTED-RUN GUARD'
LOG_NAME = '_injected_guard_refusals.log'   # under <out>/_records/
# config keys of a section that name production roots (scalar, tuple or dict of paths); 'paths' is taken whole
ROOT_KEYS = ('PRODUCTION_ROOTS', 'EXTRA_PRODUCTION_ROOTS', 'OUT_ROOT', 'PRODUCTION_OUT', 'PHOT_ROOT')
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
    roots = [out] + ([work] if work else [])
    if not work:                       # node-local scratch (core/fitsio.scratch_dir) is temporary, never a product
        from exopipe.phase3.core import fitsio
        try:
            scratch = fitsio.scratch_root(None)
        except Exception:              # noqa: BLE001 - no scratch root on this host: not allowed, nothing else changes
            scratch = None
        if scratch and refused_root_of(scratch, C) is None:
            roots.append(scratch)
    from exopipe.phase3.core import guard
    log = os.path.join(out, '_records', LOG_NAME) if out else None
    allowed = guard.install(roots, log=log, label=LABEL)
    guard.export_env(roots, log=log)
    return allowed
