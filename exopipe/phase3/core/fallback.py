"""B-30 (a), Gregg 2026-10-13 (R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32): no silent fallbacks.

A fallback SITE is a place where code substitutes a default for missing, unreadable or degenerate input and carries on
without saying so (coadd.get_midpoint_time returned 2025-01-01T00:00 for an image with no readable time; every
time-dependent gate downstream then worked on a wrong time). Each site names a key of the config section ``fallbacks``
(config/shared.py; ``<SITE>_FALLBACK`` - the section, not the suffix, makes it a fallback key: other sections hold
ordinary tunables named *_FALLBACK):

    'refuse'    (ruled)       the site raises FallbackRefused naming itself and the input;
    'fallback'  (production)  the site keeps production behaviour and is COUNTED in the unit record.

    from exopipe.phase3.core import fallback
    ...
    except Exception:
        fallback.site('fallbacks.IMAGE_TIME_FALLBACK', f'{path}: no DATE-OBS / DATE-BEG / MJD-OBS')
        return Time('2025-01-01T00:00:00', scale='utc')          # production behaviour, reached only under 'fallback'

A refusal is STICKY: it is also recorded here, and the unit that was running refuses at its end even when code between
the site and the unit caught the exception (the moved production modules hold many broad ``except Exception``).
Children: a step process exports its policies (EXOPIPE3_FALLBACK_POLICY, with EXOPIPE3_FALLBACK_POLICY_PID = its pid -
it keeps reading config itself) and its pid as the spool owner (EXOPIPE3_FALLBACK_SPOOL_OWNER), so every descendant uses
the step's policies. A child started by core.legacy.run (EXOPIPE3_FALLBACK_OUT set) owns its events and dumps them at exit
(core/parity_sitecustomize/sitecustomize.py; one file per process, so its own descendants' dumps never overwrite it); a
forked worker or a descendant started any other way writes each event to the owner's spool at once.
Design: exopipe_ops/refactor_20261012/plan/wave2/B30_DESIGN.md.

A MEMORY MAP OF A LUSTRE FILE refused in a child (R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE; the gate is in
core/parity_sitecustomize/sitecustomize.py) is a refusal of this same state, under the key MAP_KEY, so it is sticky in the
same way: the unit that was running refuses even when the child caught the refusal and exited 0. The gate writes one line
per refused map to the file $EXOPIPE3_MAP_REFUSAL_LOG names; two collectors read those lines into this state:
- one child call (core.legacy.run, serve's run_shell): map_log_new() gives the call its own file, map_collect() reads it
  and the gate's lines in the call's own output (a descendant that dropped the variable, or an append that failed, still
  wrote its line to stderr), removes the file and records each line once;
- the step itself (export_env, cli.main): its own file for every other descendant (plain subprocesses), read at each
  mark / since / take / outside / dump - attributed to the thread that reads it, as a forked worker's spooled event.
A long-lived server (serve's procs.start, jupiter serve) is never a unit: its environment drops the variable, and its
refusals are in its own log (stderr) - the request that asked for the map fails.
"""
import collections
import json
import os
import re
import tempfile
import threading

try:
    from exopipe.phase3.core.record import Refusal
except ImportError:             # loaded by path in a child whose `exopipe` is the production package (core/wrap.py)
    class Refusal(Exception):
        """Stand-in: the parent reads this process's dump, never the exception object."""

POLICIES = ('refuse', 'fallback')
SECTION = 'fallbacks'
SUFFIX = '_FALLBACK'
# State of this process. EVENTS = [[thread id, key, kind ('count' | 'refuse'), n, detail, taken]] in order; a unit takes
# the events of its own thread since it started (threaded units each get theirs), the run records what no unit took.
# COUNTS / REFUSALS are the same events as totals (for readers that want them). One lock: sites can run in threads.
EVENTS = []
COUNTS = collections.Counter()
REFUSALS = []
_LOCK = threading.Lock()
# The process whose units record. A FORKED worker (multiprocessing fork pool) sees its parent's pid here: its memory dies
# with it and it skips atexit, so it writes each event to a spool file at once, which the parent drains (mark / since /
# take / dump). A process started fresh (core.legacy.run) is its own owner and sends its state back through
# EXOPIPE3_FALLBACK_OUT instead.
_OUT = os.environ.get('EXOPIPE3_FALLBACK_OUT')
_OWNER = os.getpid() if _OUT else int(os.environ.get('EXOPIPE3_FALLBACK_SPOOL_OWNER') or os.getpid())
_ENV = None
# NO-LUSTRE-MAP (module docstring). MAP_RULING_ID is the gate's own (core/parity_sitecustomize/sitecustomize.py
# _LUSTRE_RULING_ID; tests/test_no_mmap.py checks the two are equal): a gate line is 'NO-LUSTRE-MAP: <file> pid <pid> <id>'.
MAP_KEY = 'gates.NO_LUSTRE_MAP'
MAP_ENV = 'EXOPIPE3_MAP_REFUSAL_LOG'
MAP_RULING_ID = 'R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE'
_MAP_LINE = re.compile(r'NO-LUSTRE-MAP: .+ pid \d+ ' + re.escape(MAP_RULING_ID))
_MAP_STEP = {}                  # this step's own collector: {'pid', 'path', 'off'} (export_env)
_MAP_LOCK = threading.Lock()


class FallbackRefused(Refusal):
    """A fallback site under policy 'refuse'."""


def _env_policies():
    """The policies a parent passed down; {} in the process that exported them (it reads config, overrides included)."""
    global _ENV
    if _ENV is None:
        raw = os.environ.get('EXOPIPE3_FALLBACK_POLICY')
        mine = os.environ.get('EXOPIPE3_FALLBACK_POLICY_PID') == str(os.getpid())
        _ENV = json.loads(raw) if (raw and not mine) else {}
    return _ENV


def policy(key):
    """'refuse' or 'fallback' for ``fallbacks.<SITE>_FALLBACK``: the parent's JSON in a child, else config."""
    sec, _, name = key.partition('.')
    if sec != SECTION or not name.endswith(SUFFIX):
        raise KeyError(f'fallback key {key!r}: a fallback site names a key {SECTION}.<SITE>{SUFFIX} (config/shared.py)')
    env = _env_policies()
    if key in env:
        v = env[key]
    else:
        from exopipe.phase3.config import CFG
        v = getattr(getattr(CFG, sec), name)
    if v not in POLICIES:
        raise Refusal(f'{key} = {v!r}: one of {POLICIES}')
    return v


def _spool_dir():
    return os.path.join(tempfile.gettempdir(), f'exopipe3_fallback_spool_{_OWNER}')


def _record(kind, key, n, detail, tid=None, spool=True):
    tid = threading.get_ident() if tid is None else tid
    with _LOCK:
        EVENTS.append([tid, key, kind, int(n), str(detail), False])
        if kind == 'count':
            COUNTS[key] += int(n)
        else:
            REFUSALS.append((key, str(detail)))
    if spool and os.getpid() != _OWNER:          # a forked worker: write the event out now
        try:
            d = _spool_dir()
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f'{os.getpid()}.jsonl'), 'a') as f:
                f.write(json.dumps([kind, key, int(n), str(detail)]) + '\n')
        except OSError:
            pass


def _drain():
    """Absorb the spooled events of this process's forked workers (owner only) and the new lines of this step's own map
    refusal file; attributed to the draining thread."""
    _drain_map_step()
    if os.getpid() != _OWNER:
        return
    d = _spool_dir()
    if not os.path.isdir(d):
        return
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        try:
            with open(p) as f:
                lines = f.read().split('\n')
            os.remove(p)
        except OSError:
            continue
        for ln in lines:
            try:
                kind, key, n, detail = json.loads(ln)
            except ValueError:                   # empty, or a worker killed mid-write
                continue
            _record(kind, key, n, detail, spool=False)
    try:
        os.rmdir(d)
    except OSError:
        pass


def site(key, detail, n=1):
    """A fallback site was reached. 'fallback': count ``n`` and return (the caller keeps production behaviour);
    'refuse': record and raise FallbackRefused. An invalid policy value refuses too, recorded first, so that a caller's
    broad except cannot hide it (the step refuses it at start - check_values; a child can still be handed one)."""
    try:
        pol = policy(key)
    except Refusal as e:
        _record('refuse', key, 1, f'{detail} [{e}]')
        raise FallbackRefused(f'B-30 fallback site {key}: {detail} - REFUSED: its policy is invalid ({e})')
    if pol == 'fallback':
        _record('count', key, n, detail)
        return
    _record('refuse', key, 1, detail)
    raise FallbackRefused(f'B-30 fallback site {key}: {detail} - REFUSED (no silent fallbacks, Gregg 2026-10-13; '
                          f"{key}='fallback' keeps the production behaviour, counted)")


def mark():
    """The state to compare against later (a unit or a run takes it when it starts)."""
    _drain()
    with _LOCK:
        return len(EVENTS), threading.get_ident()


def _select(m, take, any_thread=False, untaken_only=False):
    _drain()
    n0, tid = m
    refs, counts = [], collections.Counter()
    with _LOCK:
        for e in EVENTS[n0:]:
            if (not any_thread and e[0] != tid) or (untaken_only and e[5]):
                continue
            if take:
                e[5] = True
            if e[2] == 'count':
                counts[e[1]] += e[3]
            else:
                refs.append((e[1], e[4]))
    return refs, dict(counts)


def since(m):
    """(refusals, counts) of this thread recorded after mark ``m``."""
    return _select(m, take=False)


def take(m):
    """since(m), marking those events as taken (a unit at its end: what the run's record must not count again)."""
    return _select(m, take=True)


def outside(m):
    """(refusals, counts) after mark ``m``, any thread, that no unit took (a run at its end)."""
    return _select(m, take=False, any_thread=True, untaken_only=True)


def absorb(counts, refusals):
    """Add a child's counts and refusals to this process's state (core.legacy.run), so the running unit records them."""
    for k, n in (counts or {}).items():
        _record('count', k, n, 'in a child process', spool=False)
    for r in refusals or ():
        _record('refuse', r[0], 1, r[1], spool=False)


def refusal_text(key, detail):
    """How a unit or a child call names one refusal of this state: the gate's own line for a refused map (MAP_KEY), else
    'B-30 fallback site <key>: <detail>'."""
    return str(detail) if key == MAP_KEY else f'B-30 fallback site {key}: {detail}'


def map_lines(lines):
    """The gate's own lines among ``lines`` (text), each stripped of its line end."""
    return [ln.rstrip('\r\n') for ln in lines if _MAP_LINE.fullmatch(ln.rstrip('\r\n'))]


def map_log_new(prefix='exopipe3_maprefusal_'):
    """A new empty file for the map refusal lines of one child call (the caller puts it in the child's MAP_ENV). The gate
    appends only to a file that exists, so a descendant that outlives the call never recreates it once collected."""
    fd, path = tempfile.mkstemp(prefix=prefix, suffix='.log')
    os.close(fd)
    return path


def map_collect(path, output=(), seen=()):
    """The map refusals of one child call: the lines in its file ``path`` (removed here) and the gate's lines in the call's
    own ``output`` (lines of text), each refused map once (a line in both counts once; a line only in the output counts,
    unless it is among ``seen``: lines recorded another way - the child's own nested calls', from its fallback dump - that
    the child may have printed); each is recorded in this process's state under MAP_KEY - sticky: the running unit refuses
    (a forked worker spools it to the owner) - and the list is returned in order."""
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except OSError:
        data = b''
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    got = [ln for ln in data.decode('utf-8', 'replace').splitlines() if ln.strip()]
    left = collections.Counter(got) + collections.Counter(seen)
    extra = []
    for ln in map_lines(output or ()):
        if left[ln] > 0:
            left[ln] -= 1
        else:
            extra.append(ln)
    got += extra
    for ln in got:
        _record('refuse', MAP_KEY, 1, ln)
    return got


def _map_step_open():
    """This step's own map refusal file (export_env): every descendant not started through a collector writes there."""
    if _MAP_STEP.get('pid') == os.getpid():
        return
    import atexit
    path = map_log_new(prefix=f'exopipe3_maprefusal_step_{os.getpid()}_')
    with _MAP_LOCK:
        _MAP_STEP.update(pid=os.getpid(), path=path, off=0)
    os.environ[MAP_ENV] = path
    atexit.register(_map_step_close, os.getpid(), path)


def _map_step_close(pid, path):
    if os.getpid() == pid:
        try:
            os.remove(path)
        except OSError:
            pass


def _drain_map_step():
    """Record the complete lines this step's map refusal file got since the last read (the step process only)."""
    with _MAP_LOCK:
        if _MAP_STEP.get('pid') != os.getpid():
            return
        try:
            with open(_MAP_STEP['path'], 'rb') as f:
                f.seek(_MAP_STEP['off'])
                data = f.read()
        except OSError:
            return
        end = data.rfind(b'\n') + 1                 # complete lines only: a line being written is read next time
        _MAP_STEP['off'] += end
    for ln in data[:end].decode('utf-8', 'replace').splitlines():
        if ln.strip():
            _record('refuse', MAP_KEY, 1, ln, spool=False)


def all_keys(cfg=None):
    """{'fallbacks.KEY': value} of every fallback key (the keys of the config section ``fallbacks``)."""
    if cfg is None:
        from exopipe.phase3.config import CFG as cfg
    return {f'{SECTION}.{k}': v for k, v in getattr(cfg, SECTION).items()}


def export_env():
    """A step process (cli.main, after its overrides): every descendant - legacy.run children, forked workers, plain
    subprocesses - uses this process's policies, and a descendant that does not dump its own state spools it here; a
    descendant's refused map of a Lustre file lands in this step's own map refusal file (MAP_ENV), read into this state."""
    os.environ['EXOPIPE3_FALLBACK_POLICY'] = json.dumps(all_keys(), sort_keys=True)
    os.environ['EXOPIPE3_FALLBACK_POLICY_PID'] = str(os.getpid())
    os.environ['EXOPIPE3_FALLBACK_SPOOL_OWNER'] = str(os.getpid())
    _map_step_open()


def child_env():
    """The environment entries a child needs: every fallback key's policy as JSON - the policies this process was handed
    when it did not export them itself (a legacy.run inside a child passes the step's on, never the child's own config
    defaults), else this process's config, overrides included."""
    raw = os.environ.get('EXOPIPE3_FALLBACK_POLICY')
    if raw and os.environ.get('EXOPIPE3_FALLBACK_POLICY_PID') != str(os.getpid()):
        return {'EXOPIPE3_FALLBACK_POLICY': raw}
    return {'EXOPIPE3_FALLBACK_POLICY': json.dumps(all_keys(), sort_keys=True)}


def check_values(cfg=None):
    """[problems]: each fallback key whose value is not one of POLICIES (cli.main refuses the step on any: a misspelt
    policy would otherwise surface only where a site is reached)."""
    return [f'{k} = {v!r} (one of {POLICIES})' for k, v in sorted(all_keys(cfg).items()) if v not in POLICIES]


def dump(path):
    """Write this process's counts and refusals (a child, at exit) - its forked workers' spooled events included - to
    <path>.<pid> (a descendant that inherited the same EXOPIPE3_FALLBACK_OUT writes its own file)."""
    _drain()
    out = f'{path}.{os.getpid()}'
    with _LOCK:
        d = {'counts': dict(COUNTS), 'refusals': list(REFUSALS)}
    with open(out + '.tmp', 'w') as f:
        json.dump(d, f)
    os.replace(out + '.tmp', out)


def load(path, remove=True):
    """(counts, refusals) a child and its descendants wrote (<path>.<pid> files, and <path> itself); ({}, []) when none
    imported this module. The files are removed after reading (remove=True)."""
    import glob as _glob
    counts, refs = collections.Counter(), []
    for p in [path] + sorted(_glob.glob(glob_escape(path) + '.*')):
        if p.endswith('.tmp') or not os.path.isfile(p):
            continue
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        counts.update({k: int(n) for k, n in (d.get('counts') or {}).items()})
        refs.extend(tuple(r) for r in d.get('refusals') or [])
        if remove:
            try:
                os.remove(p)
            except OSError:
                pass
    return dict(counts), refs


def glob_escape(p):
    import glob as _glob
    return _glob.escape(p)


# ---- comparing a moved module with its production original ----------------------------------------------------------

def _is_site_stmt(n):
    import ast
    if isinstance(n, ast.ImportFrom) and n.module == 'exopipe.phase3.core' and [a.name for a in n.names] == ['fallback']:
        return True
    if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
        f = n.value.func
        return isinstance(f, ast.Attribute) and f.attr == 'site' and isinstance(f.value, ast.Name) and f.value.id == 'fallback'
    return False


# calls a site's `if` test or arguments may make (no side effects); anything else keeps the statement - a difference
_PURE_FUNCS = {'len', 'any', 'all', 'int', 'float', 'str', 'repr', 'sorted', 'min', 'max', 'abs', 'isinstance', 'hasattr',
               'getattr', 'bool', 'round', 'sum', 'set', 'list', 'tuple', 'dict', 'enumerate', 'zip', 'range', 'type'}
_PURE_METHODS = {'get', 'exists', 'isfile', 'isdir', 'isfinite', 'isnan', 'any', 'all', 'sum', 'abs', 'size', 'basename',
                 'dirname', 'join', 'items', 'keys', 'values', 'exc_info', 'isot', 'strip', 'startswith', 'endswith',
                 'lower', 'upper', 'format', 'split'}


def _pure(node, extra=()):
    import ast
    for n in ast.walk(node):
        if isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await, ast.Lambda)) or type(n).__name__ == 'NamedExpr':
            return False
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and (f.id in _PURE_FUNCS or f.id in extra):
                continue
            if isinstance(f, ast.Attribute) and f.attr in _PURE_METHODS:
                continue
            return False
    return True


def site_lines(text, pure_calls=()):
    """Line numbers (1-based) of the B-30 site statements of a module: `from exopipe.phase3.core import fallback`,
    `fallback.site(...)` with pure arguments, an `if` (no else) with a pure test whose body is only those, and an `else:`
    branch that is only those (its `else:` line, comment allowed, included). Each must stand alone on its lines (no `;`).
    ``pure_calls``: module functions a test or argument may call besides the builtins / methods allowed (a path helper).
    A statement that fails any check stays - the comparison then shows a difference (fail-safe)."""
    import ast
    import textwrap
    lines = text.split('\n')
    out = set()

    def alone(st):
        seg = textwrap.dedent('\n'.join(lines[st.lineno - 1:st.end_lineno]))
        try:
            return len(ast.parse(seg).body) == 1
        except SyntaxError:
            return False

    def is_site(st):
        if not _is_site_stmt(st) or not alone(st):
            return False
        if isinstance(st, ast.Expr):
            c = st.value
            return all(_pure(x, pure_calls) for x in c.args) and all(_pure(k.value, pure_calls) for k in c.keywords)
        return True

    def only_sites(stmts):
        return bool(stmts) and all(is_site(x) or (isinstance(x, ast.If) and not x.orelse and _pure(x.test, pure_calls)
                                                  and only_sites(x.body)) for x in stmts)

    for node in ast.walk(ast.parse(text)):
        for field in ('body', 'orelse', 'finalbody'):
            stmts = getattr(node, field, None)
            if not (isinstance(stmts, list) and stmts and isinstance(stmts[0], ast.stmt)):
                continue
            for st in stmts:
                if is_site(st) or (isinstance(st, ast.If) and not st.orelse and _pure(st.test, pure_calls)
                                   and only_sites(st.body) and alone(st)):
                    out.update(range(st.lineno, st.end_lineno + 1))
        if isinstance(node, ast.If) and node.orelse and only_sites(node.orelse) and not isinstance(node.orelse[0], ast.If):
            k = node.orelse[0].lineno - 1                     # the `else:` line, when it stands alone
            if lines[k - 1].split('#')[0].strip() == 'else:':
                out.add(k)
                for st in node.orelse:
                    out.update(range(st.lineno, st.end_lineno + 1))
    return out


def strip_site_lines(text, pure_calls=()):
    """The module text without its B-30 site lines (site_lines): what is left is the production text."""
    drop = site_lines(text, pure_calls)
    return '\n'.join(x for i, x in enumerate(text.split('\n'), 1) if i not in drop)


def same_code_modulo_sites(production_text, branch_text, pure_calls=()):
    """True when the branch module is the production module plus B-30 site lines only: byte equality once they are
    removed (a comment-only or any other edit of production still differs - the gates that use this stay byte-strict)."""
    return production_text == strip_site_lines(branch_text, pure_calls)
