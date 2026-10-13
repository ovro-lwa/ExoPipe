"""Shared machinery of `exopipe3 pages`: legacy code roots (core.env.legacy_root + --legacy-root), the step environment
for in-place calls (core.legacy.run is the ONE child runner - crosscheck decision 1), the server's panel-name pattern and
the deep page code identity read from source, CSV helpers, the run-directory lease (gate P9), exclusion bookkeeping,
record-chain helpers (gate P2) and stage/set selection.

Nothing here imports a production module into the step's process: the review server's PNG_RE and every other production
constant are read from source text (ast), never executed.
"""
import ast
import contextlib
import csv
import errno
import hashlib
import json
import os
import re
import time

csv.field_size_limit(10 ** 9)

ROOT_NAMES = {'ops': 'exopipe_ops', 'clv15': 'clv15'}     # short name -> core.env.legacy_root name


def CFG():
    from exopipe.phase3.config import CFG as _C
    return _C


def Refusal(msg):
    from exopipe.phase3.core.record import Refusal as _R
    return _R(msg)


def NoInput(msg):
    from exopipe.phase3.core.record import NoInput as _N
    return _N(msg)


def package_root():
    """The directory holding the `exopipe` package this step runs from (the refactor tree during the window)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


# ---------------------------------------------------------------------------------------------------------------------
# legacy code roots (site LEGACY_CODE_ROOTS via core.env.legacy_root; --legacy-root for a frozen tree or a test stub)
# ---------------------------------------------------------------------------------------------------------------------

def legacy_roots(overrides=()):
    """{'ops': path, 'clv15': path} from core.env.legacy_root, overridden by --legacy-root NAME=PATH (NAME ops or clv15)."""
    from exopipe.phase3.core import env
    roots = {}
    for short, name in ROOT_NAMES.items():
        try:
            roots[short] = env.legacy_root(name).rstrip('/')
        except KeyError:
            pass
    for o in overrides or ():
        if '=' not in o:
            raise ValueError(f'--legacy-root expects NAME=PATH (NAME ops or clv15), got {o!r}')
        k, v = o.split('=', 1)
        k = k.strip()
        if k not in ROOT_NAMES:
            raise ValueError(f'--legacy-root {k!r}: NAME must be ops or clv15')
        roots[k] = os.path.abspath(v.strip()).rstrip('/')
    return roots


def overridden(roots):
    """The roots that differ from the site's (a frozen tree / a stub). core.legacy.run fingerprints child modules under the
    SITE roots only, so for an overridden root only the script itself is in the child fingerprint (recorded in the unit)."""
    site = legacy_roots()
    return {k: v for k, v in roots.items() if site.get(k) != v}


def need_root(roots, name):
    if name not in roots:
        raise Refusal(f'no legacy code root {name!r}: add {ROOT_NAMES.get(name, name)} to LEGACY_CODE_ROOTS in the site file '
                      f'($EXOPIPE3_SITE) or pass --legacy-root {name}=PATH')
    if not os.path.isdir(roots[name]):
        raise Refusal(f'legacy code root {name}={roots[name]} is not a directory')
    return roots[name]


def legacy_script(roots, name, rel):
    p = os.path.join(need_root(roots, name), rel)
    if not os.path.isfile(p):
        raise Refusal(f'in-place script {p} not found (legacy root {name}={roots[name]})')
    return p


def interpreter_or_none(key):
    """The site PYTHON[key] interpreter path, or None when unset / not on this host (the dry run reports it)."""
    py = (CFG().site.PYTHON or {}).get(key)
    return py if py and os.path.exists(py) else None


# ---------------------------------------------------------------------------------------------------------------------
# the step environment of in-place calls
# ---------------------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def scrubbed_environ():
    """For the duration of the step: remove inherited variables of pages.LEGACY_ENV_STRIP_PREFIXES from os.environ (core/
    legacy.run clears P3_* only; P3S_LABEL, P3S3_*, P3S_MERGE_*, P3T_*, SUBLST_*, SNAP_* would change products) and this
    package's root from PYTHONPATH (the in-place code imports the PRODUCTION exopipe package, as production runs it).
    Restored on exit. Every variable a stage needs is passed explicitly to core.legacy.run."""
    saved = dict(os.environ)
    prefixes = tuple(CFG().pages.LEGACY_ENV_STRIP_PREFIXES)
    removed = sorted(k for k in os.environ if k.startswith(prefixes))
    for k in removed:
        del os.environ[k]
    pkg = os.path.realpath(package_root())
    pp = [p for p in (os.environ.get('PYTHONPATH') or '').split(os.pathsep) if p]
    keep = [p for p in pp if os.path.realpath(p) != pkg]
    if keep != pp:
        if keep:
            os.environ['PYTHONPATH'] = os.pathsep.join(keep)
        else:
            os.environ.pop('PYTHONPATH', None)
    try:
        yield removed
    finally:
        os.environ.clear()
        os.environ.update(saved)


def lrun(u, script, args, python='default', env=None, cwd=None, log=None, timeout=None, refusal_codes=(2,),
         refusal_text=('REFUSED', 'REFUSING', 'REFUSE:')):
    """core.legacy.run with the unit; the variables the step passed and the cwd are recorded beside core's call entry."""
    from exopipe.phase3.core import legacy
    res = legacy.run(script, args, python=python, env=env or {}, cwd=cwd, log=log, timeout=timeout,
                     refusal_codes=refusal_codes, refusal_text=refusal_text, unit=u)
    # the script itself: core's child fingerprint misses it (in the child, __main__.__file__ is None at exit under
    # Python 3.8 - reported to the refactor chat) and covers only modules under the SITE roots (not a --legacy-root override)
    u.meta.setdefault('legacy_code', {})[os.path.abspath(script)] = sha1_file(script)
    for c in reversed(u.meta.get('legacy_calls') or []):
        if c.get('argv') == res.argv and 'env_passed' not in c:
            c['env_passed'] = dict(env or {})
            c['cwd'] = cwd
            break
    return res


def legacy_code_unchanged(rec):
    """(True, '') when every legacy file recorded in a unit record (core.legacy.run's child fingerprint) still has its
    sha1; else (False, reason). core.record.resume_check covers only the parent's modules."""
    for p, h in ((rec.get('meta') or {}).get('legacy_code') or {}).items():
        if not os.path.isfile(p):
            return False, f'legacy code gone: {p}'
        if sha1_file(p) != h:
            return False, f'legacy code changed: {p}'
    return True, ''


def sha1_file(p, n=12):
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()[:n]


# ---------------------------------------------------------------------------------------------------------------------
# production constants read from source (never executed)
# ---------------------------------------------------------------------------------------------------------------------

_AST_CACHE = {}


def module_ast(path):
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _AST_CACHE:
        with open(path, encoding='utf-8', errors='replace') as f:
            _AST_CACHE[key] = ast.parse(f.read(), path)
    return _AST_CACHE[key]


def module_constant(path, name):
    """The literal assigned to module-level ``name`` in ``path`` (ast.literal_eval of the right-hand side); for
    ``name = re.compile(<literal>, ...)`` the pattern literal. KeyError when absent; ValueError when not a literal."""
    for node in module_ast(path).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            v = node.value
            if isinstance(v, ast.Call) and getattr(v.func, 'attr', None) == 'compile' and v.args:
                return ast.literal_eval(v.args[0])
            return ast.literal_eval(v)
    raise KeyError(f'{name} is not a module-level literal of {path}')


_PNG_RE = {}


def server_png_re(roots=None):
    """classify_transients.PNG_RE (the review server's panel-name pattern), read from the in-place server source."""
    roots = roots if roots is not None else legacy_roots()
    p = os.path.join(need_root(roots, 'ops'), 'classify_transients.py')
    if p not in _PNG_RE:
        _PNG_RE[p] = re.compile(module_constant(p, 'PNG_RE'))
    return _PNG_RE[p]


def build_i14_path(roots):
    return os.path.join(need_root(roots, 'ops'), 'build_i14_pages_std7.py')


def page_code(roots):
    """The 'code' a deep page sidecar carries: sha1(build_i14 + std7_panel + p3s3_astrom)[:16] (build_i14:819-820), the two
    modules taken from the directory build_i14 puts first on its sys.path (build_i14:27)."""
    b = build_i14_path(roots)
    mod_dir = os.path.dirname(b)
    for node in module_ast(b).body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, 'attr', None) == 'insert' and len(node.value.args) == 2):
            try:
                mod_dir = ast.literal_eval(node.value.args[1])
            except ValueError:
                pass
            break
    blob = b''.join(open(p, 'rb').read() for p in (os.path.abspath(b), os.path.join(mod_dir, 'std7_panel.py'),
                                                       os.path.join(mod_dir, 'p3s3_astrom.py')))
    return hashlib.sha1(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------------------------------------------------

def read_csv(path):
    """(header, rows) of a CSV file (rows as dicts)."""
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        return list(rd.fieldnames or []), rows


def n_rows(path):
    try:
        with open(path, newline='') as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def digest_without(path, column):
    """sha1 of a CSV's content with one column removed (header and rows): the identity of everything except ``column``
    (the survivors' review_dir before and after the merge)."""
    h = hashlib.sha1()
    with open(path, newline='') as f:
        rd = csv.reader(f)
        hdr = next(rd, [])
        idx = hdr.index(column) if column in hdr else None
        for row in [hdr] + list(rd):
            if idx is not None and idx < len(row):
                row = row[:idx] + row[idx + 1:]
            h.update(('\x1f'.join(row) + '\n').encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# lease (gate P9): one exopipe3 pages writer per run directory
# ---------------------------------------------------------------------------------------------------------------------

class Lease:
    """flock on <out>/_records/pages/<run>.lease for the whole invocation; a second `exopipe3 pages` on the same run
    directory waits pages.LEASE_WAIT_S and then refuses (P9; ledger 1457). The production launchers do not take it."""

    def __init__(self, out_root, run_dir, wait_s=0):
        self.path = os.path.join(out_root, '_records', 'pages', os.path.basename(run_dir.rstrip('/')) + '.lease')
        self.run_dir, self.wait_s, self.fh = run_dir, wait_s, None

    def __enter__(self):
        import fcntl
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fh = open(self.path, 'a+')
        t_end = time.time() + max(0, float(self.wait_s or 0))
        while True:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                    self.fh.close()
                    raise Refusal(f'P9: cannot take the run-directory lease {self.path} ({e})')
                if time.time() >= t_end:
                    self.fh.seek(0)
                    holder = self.fh.read().strip()
                    self.fh.close()
                    raise Refusal(f'P9: another exopipe3 pages job holds the lease of {self.run_dir} ({self.path}: '
                                  f'{holder or "holder unknown"}) - one writer per run directory (ledger 1457)')
                time.sleep(2)
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(f'{os.uname().nodename} pid {os.getpid()} since {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}\n')
        self.fh.flush()
        return self

    def __exit__(self, *exc):
        import fcntl
        try:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
        finally:
            self.fh.close()
        return False


# ---------------------------------------------------------------------------------------------------------------------
# exclusions (pages render the catalogue as given; the exclusion state is part of each unit's identity)
# ---------------------------------------------------------------------------------------------------------------------

def exclusion_bookkeeping(u, keys):
    """Record the exclusion list as an input (an exclusion change invalidates the unit under --resume) and the admitted
    nights of the (lst, scope, date) keys the unit rendered; rows on excluded keys are counted (the serve-time exclusion gate
    is serve's; spec section 5)."""
    from exopipe.phase3.core import exclusions
    ex = exclusions.load()
    u.add_input(ex.path)
    admitted, n_ex, ex_keys = set(), 0, set()
    for lst, scope, date in keys:
        d = (date or '')[:10]
        if not (lst and scope and d):
            continue
        if ex.is_excluded(lst, scope, d):
            n_ex += 1
            ex_keys.add((lst, scope, d))
        else:
            admitted.add(d)
    u.set_admitted(admitted)
    u.meta['exclusions'] = dict(ex.fingerprint(), rows_on_excluded=n_ex,
                                excluded_keys=sorted('_'.join(k) for k in ex_keys)[:50])
    if n_ex:
        u.note(f'{n_ex} rendered rows lie on {len(ex_keys)} now-excluded (LST, band, night) keys - rendered as the '
               f'catalogue gives them; serve refuses them at serve time')


# ---------------------------------------------------------------------------------------------------------------------
# fingerprints shared by several units; record chain (P2); resume
# ---------------------------------------------------------------------------------------------------------------------

class FPCache:
    def __init__(self):
        self._c = {}

    def get(self, path):
        from exopipe.phase3.core.record import fingerprint
        key = os.path.abspath(path)
        if key not in self._c:
            fp = fingerprint(path)
            if os.path.islink(path):
                try:
                    fp['target'] = os.readlink(path)
                except OSError:
                    pass
            self._c[key] = fp
        return dict(self._c[key])

    def add(self, u, paths):
        for p in paths:
            u.inputs[os.path.abspath(p)] = self.get(p)


def output_fps(paths):
    """{path: fingerprint} of outputs, kept in the unit meta so the next stage can check it reads exactly them."""
    from exopipe.phase3.core.record import fingerprint
    return {os.path.abspath(p): fingerprint(p) for p in paths if os.path.exists(p)}


def prerequisite(run, done_now, unit, paths=None):
    """Gate P2 within the step: the unit ``unit`` of an earlier stage is done in this invocation (``done_now``) or has a done
    record whose recorded output fingerprints still match the files (pages.REQUIRE_STAGE_RECORDS). Returns a provenance
    string ('no-input' for a recorded NO-INPUT); raises Refusal otherwise."""
    from exopipe.phase3.core import record
    if unit in done_now:
        return done_now[unit]
    if not CFG().pages.REQUIRE_STAGE_RECORDS:
        return 'not checked (pages.REQUIRE_STAGE_RECORDS False)'
    rec = record.load_record(run.out_root, 'pages', unit)
    if rec is None:
        raise Refusal(f'P2: no record of {unit} under {run.out_root}/_records/pages - run that stage first (a stage reads '
                      f'only what a done record of its previous stage lists; DESIGN 11.5)')
    if rec.get('status') == 'no-input':
        return 'no-input'
    if rec.get('status') != 'done':
        raise Refusal(f'P2: the record of {unit} says {rec.get("status")} ({str(rec.get("reason", ""))[:120]})')
    fps = (rec.get('meta') or {}).get('output_fp') or {}
    for p in (paths or fps):
        p = os.path.abspath(p)
        if p not in fps:
            raise Refusal(f'P2: {p} is not an output of the record of {unit}')
        if not record.same_fingerprint(fps[p], record.fingerprint(p)):
            raise Refusal(f'P2: {p} changed after the record of {unit} was written ({rec.get("finished")}) - re-run that stage')
    return f'record {rec.get("finished")}'


def resume_skip(run, name, enabled=None):
    """(True, reason) when --resume (or the per-hour --epochs mode) may keep unit ``name``: core resume_check (record done,
    inputs, admitted nights, step code, config unchanged), the legacy code the unit ran unchanged, and every output passing
    its content tests NOW with the test context the unit recorded (meta 'test_ctx': the code roots it ran, its tier and
    layout - core's retest has no context, so a page drawn by a --legacy-root tree would always look stale)."""
    from exopipe.phase3.core import record, validate
    if not (run.resume if enabled is None else enabled) or run.fresh:
        return False, ''
    ok, why = run.resume_check(name, inputs=None, admitted=None, retest=False)
    if not ok:
        return False, why
    rec = record.load_record(run.out_root, 'pages', name) or {}
    ok2, why2 = legacy_code_unchanged(rec)
    if not ok2:
        return False, why2
    ctx = (rec.get('meta') or {}).get('test_ctx') or {}
    for o in rec.get('outputs', []):
        if not os.path.exists(o['path']):
            return False, f"output missing: {o['path']}"
        for r in validate.run(o['class'], o['path'], **ctx):
            if not r['passed']:
                return False, f"output fails {r['name']}: {o['path']}"
    return True, 'record done; inputs, admitted nights, code (step and in-place), config unchanged; outputs pass now'


def mark_skipped(run, name, why):
    u = run.unit(name)
    u.status, u.reason = 'skipped-valid', why
    return u


def jdump(obj):
    return json.dumps(obj, sort_keys=True, default=str)


# ---------------------------------------------------------------------------------------------------------------------
# stage, set and epoch selection (every tier)
# ---------------------------------------------------------------------------------------------------------------------

def select_stages(args, tier):
    """The stages to run, in the tier's fixed order: --stages (on-request stages only when named), then --from / --until."""
    C = CFG().pages
    order = list(C.STAGES[tier])
    allowed = order + [s for s in C.ON_REQUEST_STAGES.get(tier, ()) if s not in order]
    if getattr(args, 'stages', None):
        want = [s.strip() for s in args.stages.split(',') if s.strip()]
        unknown = [s for s in want if s not in allowed]
        if unknown:
            raise ValueError(f'--stages: unknown stage(s) {unknown} for tier {tier}; stages: {", ".join(allowed)}')
        stages = [s for s in allowed if s in want]
    else:
        stages = order
    for flag, attr in (('--from', 'from_stage'), ('--until', 'until_stage')):
        v = getattr(args, attr, None)
        if v:
            if v not in stages:
                raise ValueError(f'{flag} {v}: not among the selected stages {stages}')
            i = stages.index(v)
            stages = stages[i:] if attr == 'from_stage' else stages[:i + 1]
    return stages


def select_sets(args, tier):
    """The page sets (deep, sub-LST): pages.PAGE_SETS[tier] narrowed by --sets and --stokes, in the configured order."""
    C = CFG().pages
    sets = list(C.PAGE_SETS.get(tier, ()))
    if getattr(args, 'sets', None):
        want = [s.strip() for s in args.sets.split(',') if s.strip()]
        unknown = [s for s in want if s not in sets]
        if unknown:
            raise ValueError(f'--sets: unknown set(s) {unknown} for tier {tier}; sets: {", ".join(sets)}')
        sets = [s for s in sets if s in want]
    if getattr(args, 'stokes', None):
        st = {s.strip() for s in args.stokes.split(',') if s.strip()}
        bad = st - {'I', 'V'}
        if bad:
            raise ValueError(f'--stokes: {sorted(bad)} (I, V or I,V)')
        sets = [s for s in sets if s[0] in st]
    return sets


def epochs_of(args):
    """The per-hour epoch items of the common --epochs selector (core.cells) as [(lst, date)]."""
    spec = getattr(args, 'epochs', None)
    if not spec:
        return []
    from exopipe.phase3.core import cells
    return [cells.parse_epoch(e) for e in cells.expand_epochs(spec)]
