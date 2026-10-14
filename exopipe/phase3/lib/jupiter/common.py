"""Shared machinery of `exopipe3 jupiter`: actions and stages, the Jupiter-cell grammar and window, product / scratch roots
and the write guard, step-level gates (J1 window, J2 exclusions, J12 environment, J13/J14 production writes, pinned inputs),
the in-place call (core.legacy.run + lib/jupiter/_inplace.py), free-core pinning per launch, run.log lines, supersede.
"""
import csv
import hashlib
import json
import os
import shutil
import time
import uuid

from exopipe.phase3.core.record import Refusal

DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_inplace.py')

ACTIONS = {
    'measure': ('extract', 'fit', 'assign', 'test'),
    'v2': ('geometry', 'masters', 'prepare', 'provenance', 'anchor', 'refcheck', 'metrics', 'render', 'summary'),
    'occurrence': ('fetch', 'validate', 'digitize', 'maps', 'figures', 'lookup', 'test'),
    'ephem': ('fetch',),
    'flag-backfill': ('window', 'run', 'verify'),
    'serve': ('phase1', 'v2'),
    'quarantine': ('renders',),
}
# stages run when neither --stages nor --from/--until is given. flag-backfill: verify needs every shard finished, so it is
# run on its own (the old launcher's --verify mode); v2: refcheck re-makes inputs that were copied by hand (explicit only).
DEFAULT_STAGES = {
    'flag-backfill': ('window', 'run'),
    'v2': ('geometry', 'masters', 'prepare', 'provenance', 'anchor', 'metrics', 'render', 'summary'),
}
# actions whose unit of work is a Jupiter cell (--cells required)
CELL_ACTIONS = ('measure', 'v2', 'quarantine')
PINNED_INPUT_KEYS = ('jupiter.SUBLST_ROOT', 'jupiter.CUBES_ROOT', 'jupiter.EPHEM_DIR', 'jupiter.DEEP_WEIGHTS')


def C():
    from exopipe.phase3.config import CFG
    return CFG


def host():
    """This host's short name (core.env.host: the key of every per-host site table)."""
    from exopipe.phase3.core import env
    return env.host()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


# ---- stages ---------------------------------------------------------------------------------------------------------
def select_stages(action, args):
    """The stages to run, in the action's order (--stages, or --from/--until, or the action's default)."""
    order = ACTIONS[action]
    if args.stages:
        want = [s.strip() for s in args.stages.split(',') if s.strip()]
        bad = [s for s in want if s not in order]
        if bad:
            raise Refusal(f'unknown stage(s) {bad} for jupiter {action}; stages: {", ".join(order)}')
        if args.stage_from or args.stage_until:
            raise Refusal('give --stages or --from/--until, not both')
        return [s for s in order if s in want]
    if args.stage_from or args.stage_until:
        a = order.index(args.stage_from) if args.stage_from else 0
        b = order.index(args.stage_until) if args.stage_until else len(order) - 1
        for name in (args.stage_from, args.stage_until):
            if name and name not in order:
                raise Refusal(f'unknown stage {name!r} for jupiter {action}; stages: {", ".join(order)}')
        if a > b:
            raise Refusal(f'--from {args.stage_from} comes after --until {args.stage_until}')
        return list(order[a:b + 1])
    return list(DEFAULT_STAGES.get(action, order))


# ---- the Jupiter window and cells -----------------------------------------------------------------------------------
def window_rows(path=None):
    """[(cell, row)] of the Jupiter window CSV (date,lst,utc_start,utc_end,jup_el_mid,...), every row."""
    p = path or C().jupiter.WINDOW_CSV
    if not os.path.exists(p):
        raise Refusal(f'Jupiter window table {p} not found (jupiter.WINDOW_CSV)')
    out = []
    with open(p, newline='') as f:
        for r in csv.DictReader(f):
            out.append((f"{r['lst'].strip()}_{r['date'].strip()}", r))
    return out


def window_universe():
    """{cell: jup_el_mid} of the window cells with Jupiter >= MIN_EL_DEG at mid-hour (the --cells pattern universe)."""
    J = C().jupiter
    out = {}
    for cell, r in window_rows():
        try:
            el = float(r['jup_el_mid'])
        except (KeyError, ValueError):
            continue
        if el >= J.MIN_EL_DEG:
            out[cell] = el
    return out


def select_cells(args, run):
    """The Jupiter cells of --cells (literals, patterns over the window, @file, record:/JOB: queries) minus --exclude-cells.
    Gate J1: a literal outside the window (jup_el_mid >= MIN_EL_DEG) refuses unless --waive J1 --ruling."""
    from exopipe.phase3.core import cells as core_cells
    if not args.cells:
        raise Refusal('give --cells (Jupiter cells <LST>_<YYYY-MM-DD>, e.g. 05h_2024-12-29; patterns over the Jupiter window)')
    uni = window_universe()
    items = core_cells.expand(args.cells, universe=sorted(uni), kind='Jupiter cell')
    if getattr(args, 'exclude_cells', None):
        drop = set(core_cells.expand(args.exclude_cells, universe=sorted(uni), kind='Jupiter cell'))
        items = [c for c in items if c not in drop]
    for c in items:
        core_cells.parse_epoch(c)
    outside = [c for c in items if c not in uni]
    if outside and not run.waived('J1'):
        raise Refusal(f'J1 Jupiter window: {", ".join(outside[:8])} not in {C().jupiter.WINDOW_CSV} with Jupiter >= '
                      f'{C().jupiter.MIN_EL_DEG:g} deg at mid-hour (Gregg 2026-10-04, JUPITER-PROGRAM-PHASE0-GO); '
                      f'waive with --waive J1 --ruling "..."')
    return items


# ---- roots ----------------------------------------------------------------------------------------------------------
class Roots:
    """Product roots of one run (the production layout under --out, no renames) and the node-local scratch."""

    def __init__(self, args, out, action, dry_run=False):
        cfg = C()
        J = cfg.jupiter
        self.out = os.path.abspath(out)
        self.production = cfg.production_root_of(self.out) is not None
        self.products = os.path.join(self.out, J.PRODUCTS)
        self.logs = os.path.join(self.out, J.PRODUCT_LOGS)
        self.v2 = os.path.join(self.out, J.V2)
        self.flagged = os.path.join(self.out, J.FLAGGED)
        self.ephem_out = os.path.join(self.out, J.EPHEM_SUBDIR)
        self.flag_state = (cfg.jupiter_flag.STATE_DIR if self.production
                           else os.path.join(self.out, cfg.jupiter_flag.STATE_SUBDIR))
        self.work = None
        self.work_problem = None
        if args.work:
            self.work = os.path.abspath(args.work)
        elif self.production:
            self.work = {'measure': cfg.jupiter_measure.CROP_ROOT}.get(action, cfg.jupiter_v2.SCRATCH_ROOT)
        else:
            from exopipe.phase3.core import fitsio
            try:
                self.work = os.path.join(fitsio.scratch_root(None), 'jupiter')
            except Refusal as e:
                self.work_problem = str(e)
        # the write guard of every in-place child: the in-place modules hard-code production write roots, so a run into
        # any other --out executes them under core/guard.py rooted at --out and the scratch (DESIGN 11.1).
        self.guard = None if self.production else [r for r in (self.out, self.work) if r]

    def need_work(self):
        if not self.work:
            raise Refusal(self.work_problem or 'no node-local scratch root: pass --work /fast/...')
        if self.work.startswith('/lustre'):
            raise Refusal(f'scratch root {self.work} is on Lustre - scratch must be node-local (/fast or /dataNN)')
        return self.work

    def describe(self):
        return (f'out {self.out} ({"PRODUCTION" if self.production else "non-production"}); scratch {self.work or "-"}; '
                f'child write guard {"off (production run)" if self.guard is None else ":".join(self.guard)}')


# ---- step-level gates -----------------------------------------------------------------------------------------------
def check_environment():
    """J12: parameters that used to come from the environment must not be set in the caller's environment."""
    bad = []
    for name in C().jupiter.ENV_FORBIDDEN:
        if name.endswith('_'):
            bad += sorted(k for k in os.environ if k.startswith(name))
        elif name in os.environ:
            bad.append(name)
    if bad:
        raise Refusal(f'J12 caller environment: {", ".join(bad)} set - these are config keys or flags now (B-50); unset them')


def check_pinned_overrides(roots):
    """A pinned-input root may be overridden only into a non-production --out (parity against the pre-bake reserve)."""
    if not roots.production:
        return
    bad = sorted({o[0].split('[')[0] for o in C().overrides if o[0].split('[')[0] in PINNED_INPUT_KEYS})
    if bad:
        raise Refusal(f'override of the pinned input root(s) {", ".join(bad)} into the production root {roots.out} refused '
                      f'(allowed only into a non-production --out)')


def exclusion_gate(lst, date, scopes):
    """J2: the scopes of (lst, date) that data_decisions.csv excludes today (core.exclusions; a missing list refuses)."""
    from exopipe.phase3.core import exclusions
    ex = exclusions.load()
    return [s for s in scopes if ex.is_excluded(lst, s, date)], ex.fingerprint()


# ---- in-place calls -------------------------------------------------------------------------------------------------
class InPlaceResult:
    def __init__(self, kind, reason, rc, log, result, legacy):
        self.kind, self.reason, self.rc, self.log, self.result, self.legacy = kind, reason, rc, log, result, legacy

    @property
    def ok(self):
        return self.kind == 'ok'


def interpreter_key(stage_key):
    """The site PYTHON key for a stage ('measure.extract', 'v2.masters.select', 'occurrence', ...)."""
    table = C().jupiter.INTERPRETER
    k = stage_key
    while k:
        if k in table:
            return table[k]
        k = k.rpartition('.')[0]
    head = stage_key.split('.')[0]
    return table.get(f'{head}.default', 'default')


def in_place(u, module, call='main', args=(), kwargs=None, argv=None, globals_=None, python='default', log=None, cwd=None,
             guard=None, env=None, test_stage=False, timeout=None, hash_roots=()):
    """Run ``module.call(*args)`` unchanged in a child (core.legacy.run on lib/jupiter/_inplace.py). Every source file the
    child loaded from the in-place roots becomes an input of ``u`` (code identity: a changed file makes --resume refuse)."""
    from exopipe.phase3.core import legacy
    cfg = C()
    if cwd:
        os.makedirs(cwd, exist_ok=True)
    rdir = cwd or (guard[0] if guard else '/tmp')
    os.makedirs(rdir, exist_ok=True)
    result_path = os.path.join(rdir, f'.jupiter_inplace_{uuid.uuid4().hex[:12]}.json')
    roots = list(hash_roots) + [cfg.jupiter.V2_CODE_ROOT, cfg.jupiter.PHASE0_CODE_ROOT] + list(cfg.site.LEGACY_CODE_ROOTS)
    dargs = ['--module', module, '--call', call, '--args', json.dumps(list(args)), '--kwargs', json.dumps(kwargs or {}),
             '--argv', json.dumps([str(a) for a in (argv or [])]), '--globals', json.dumps(globals_ or {}),
             '--hash-roots', json.dumps(roots), '--result', result_path]
    if test_stage:
        dargs.append('--test-stage')
    child_env = dict(cfg.jupiter.ENV)
    child_env['PYTHONPATH'] = cfg.jupiter.PRODUCTION_EXOPIPE_ROOT
    child_env.update(env or {})
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
    res = legacy.run(DRIVER, dargs, python=python, env=child_env, cwd=cwd, log=log,
                     parity_root=os.pathsep.join(guard) if guard else None, timeout=timeout, unit=u,
                     refusal_codes=(2,), refusal_text=('REFUSED', 'REFUSING'))
    try:
        with open(result_path) as f:
            result = json.load(f)
        os.remove(result_path)
    except (OSError, ValueError):
        result = {'kind': 'error', 'reason': f'the in-place driver wrote no result (exit {res.rc}); log tail: '
                                             + res.tail[-600:].replace('\n', ' | ')}
    if u is not None:
        for path in sorted(result.get('code_sha256') or {}):
            u.add_input(path)
        u.meta.setdefault('inplace', []).append({'module': module, 'call': call, 'kind': result.get('kind'),
                                                 'globals': sorted(globals_ or {}), 'wall_s': result.get('wall_s'),
                                                 'log': log})
    return InPlaceResult(result.get('kind', 'error'), result.get('reason', ''), res.rc, log, result, res)


_CASA = {}


def check_casa(python_key, work):
    """J8: the fit interpreter imports casatasks (a failed import turns every imfit into an unconverged fit)."""
    import subprocess
    from exopipe.phase3.core import legacy
    if python_key in _CASA:
        ok, msg = _CASA[python_key]
    else:
        py = legacy.interpreter(python_key)
        d = os.path.join(work or '/tmp', f'.casa_check_{os.getpid()}')
        os.makedirs(d, exist_ok=True)
        try:
            p = subprocess.run([py, '-c', f'import {C().jupiter.CASA_IMPORT}'], cwd=d, capture_output=True, text=True,
                               timeout=300)
            ok, msg = p.returncode == 0, (p.stderr or p.stdout).strip()[-300:]
        except Exception as e:      # noqa: BLE001
            ok, msg = False, str(e)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        _CASA[python_key] = (ok, msg)
    if not ok:
        raise Refusal(f'J8 CASA: {legacy.interpreter(python_key)} cannot import {C().jupiter.CASA_IMPORT} ({msg}) - every '
                      f'imfit would end unconverged')


def pin_free_cores(args):
    """--cores auto: read the recorder-free cores again before every engine launch (the launchers ran node_free_cores.sh
    per stage) and pin this thread; its children inherit the affinity."""
    import subprocess
    if getattr(args, 'cores', 'auto') != 'auto':
        return None
    try:
        txt = subprocess.run(C().runner.FREE_CORES_CMD, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:      # noqa: BLE001
        txt = ''
    if not txt:
        raise Refusal('could not read the recorder-free cores (runner.FREE_CORES_CMD); pass --cores LIST or --cores none')
    cpus = set()
    for part in txt.split(','):
        a, _, b = part.partition('-')
        cpus.update(range(int(a), int(b or a) + 1))
    os.sched_setaffinity(0, cpus)
    return txt


def runlog(logs_dir, line):
    """A line of products/logs/run.log in the launcher's format ('$(date -u) ...', jupiter_phase1_run.sh:15-20)."""
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, 'run.log'), 'a') as f:
        f.write(time.strftime('%a %b %e %H:%M:%S UTC %Y', time.gmtime()) + ' ' + line + '\n')


def supersede(paths, base_dir, tag):
    """Move ``paths`` aside to <base_dir>/_superseded/<tag>/ with MANIFEST.csv (name, sha256, bytes, original path); refuses
    when that directory exists (one tag, one move). Returns the directory."""
    if not tag:
        raise Refusal('--fresh over existing outputs needs --supersede TAG (they are moved aside by manifest, never deleted)')
    dest = os.path.join(base_dir, '_superseded', tag)
    if os.path.exists(dest):
        raise Refusal(f'{dest} exists - choose another --supersede TAG')
    os.makedirs(dest)
    with open(os.path.join(dest, 'MANIFEST.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'sha256', 'bytes', 'original_path', 'moved_utc'])
        for p in paths:
            w.writerow([os.path.basename(p), sha256(p), os.path.getsize(p), os.path.abspath(p), utc()])
            os.replace(p, os.path.join(dest, os.path.basename(p)))
    return dest


def short(paths, n=4):
    paths = list(paths)
    return ', '.join(os.path.basename(p) for p in paths[:n]) + (f' (+{len(paths) - n})' if len(paths) > n else '')
