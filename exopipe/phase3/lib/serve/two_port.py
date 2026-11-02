"""The two-port review server (Gregg 2026-10-17, INTERFACE_REQUIREMENTS section 1 "TWO PORTS FOR EVERYTHING"; design:
refactor_20261012/plan/serve/TWO_PORT_SERVER_DESIGN.md, "Implementation approach").

Two listeners: REVIEW (every page, with the classification interface; Gregg only) and INSPECTION (the same pages without it,
greenlit entries only, every write refused but a page's per-person review lists - INSPECTION_WRITES, Gregg 2026-10-27). A page is published by adding it to the registry (registry.json); the server
re-reads the registry when it changes - no new port, no restart.

One backend process per CODE TREE (the directory of a page's classify_transients.py: the live exopipe_ops, each frozen
10-minute tree): pages of different trees import helper modules of the same name (review_filters, source_localization ...)
that differ, and one process shares sys.modules by name - the first tree to import a helper served it to every page (found
2026-10-18: a frozen 19h tree's review_filters served the deep pages without that morning's filters). The listeners' process
(the front) builds no page: it serves the hub, the static sites and the refusals, and forwards every review-page request to
its tree's backend over a Unix socket; a backend is started on first use, restarted when it died and stopped when its tree
leaves the registry. GET /_status reports every backend's tree and, per built page, the md5 of its code and of every helper
module of its tree it imported.

The review-page code is NOT forked. Each page is today's classify_transients.py, unchanged, hosted as:
- its own module instance (a private module name per page and listener), built by the module's own main() with the page's
  argv and ThreadingHTTPServer stubbed, so every page keeps its own module globals (INSPECT, LOC_DIR, LOCOFF, App attributes);
- its own environment, as a per-thread overlay on os.environ (ThreadEnv): the pages differ today in P3_DEEP_XTIER,
  P3_TENMIN_XTIER, P3_XTIER_FRAME, P3_XTIER_SUBBAND_STACK_PREFIXES, P3_LOC_COMMON_LINKS and P3_QUEUE_BAND_ALIAS;
- its own Handler: each request runs through the page module's Handler class on a fake socket whose server_port is the
  page's legacy port (the filter cookie stays per page), so routing, inspection refusals and redirects are production's;
- with its responses rewritten (rewrite_body / rewrite_location): the page's absolute routes gain /p/<page_id>, the URL of
  every registered legacy port becomes that page's path, the two hub URLs become the listener's own '/'.
Writes (POST) are serialized per labels directory, since pages share journals.
At a registry change every built page is rebuilt, one at a time, in its backend's rebuild thread; until its rebuild is done the
page built before answers requests (STALE_HEADER says so), and only a GET carrying FRESH_HEADER waits for the rebuilt page. A page
whose own entry changed is dropped at once and built on its next request, as before. A build also builds the page's indices (its
App.prime), so its first source page is as fast as the next (WEB SPEED step 2, 2026-10-30).
A registry change is a change of the file's inode, size OR modification time (registry_state): Lustre keeps whole seconds, so by
the time alone a second write within the same second looked unchanged and its page answered 404 (B6A p9024, 2026-10-22; FOLD_IN
row 10-22 22:0x: production's register_page.py leaves the time strictly later, this server no longer depends on it).
THE PUBLISH HOLD - an OPTION (serve --publish-hold; off by default = production: a page rebuilt while its run holds the marker
shows the marker's text as a banner, exopipe_ops 4a39c6b, and nothing else changes; not ruled, an amendment option; FOLD_IN row
10-30 23:3x): with it, while <run>/HOLD_FILE (PUBLISH_IN_PROGRESS.txt, the marker production's deep publish writes and its
re-registration removes) exists, the built pages of that run are not rebuilt at a registry change: the page built before keeps
answering (STALE_HEADER and HELD_HEADER say so), a fresh GET included, until the hold is lifted - then its next request or the
next registry change queues its rebuild. A page whose own entry changed is dropped and built as before.
GET /_status reports the BUILD STATE protocol (BUILD_STATE) and the hold option, and per built page what it was built from
(entry: the md5 of its build_key), when its build started, whether it is stale or held, the builds under way, the rebuild
queue, the builds finished and the last build failure per page - what exopipe3 serve waits on after it registers a page
(lib/serve/register.wait_built; a server whose /_status lacks BUILD_STATE is refused before anything is registered).
"""
import collections
import contextlib
import email.message
import fcntl
import hashlib
import html
import http.client
import importlib.util
import io
import json
import os
import re
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
from collections.abc import MutableMapping

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_RUNTIME = os.environ.get('XDG_RUNTIME_DIR') or '/tmp'
SOCK_DIR = os.path.join(_RUNTIME, 'two_port')          # backends' sockets and logs (default)
LOCK_DIR = os.path.join(_RUNTIME, 'two_port_locks')    # one lock file per labels directory, never inside it
STATUS_PATH = '/_twoport_status'                        # a backend's own report
LISTENER_HEADER = 'X-Two-Port-Listener'                 # set by the front: 'review' or 'inspection'
FRESH_HEADER = 'X-Two-Port-Fresh'                       # '1': wait for the page rebuilt since the last registry change
STALE_HEADER = 'X-Two-Port-Stale'                       # a response from a page older than the registry: its build time
HELD_HEADER = 'X-Two-Port-Held'                         # ... kept on that build by its run's publish hold: the hold file
HOLD_FILE = 'PUBLISH_IN_PROGRESS.txt'                   # a run's publish hold (the publish marker, classify_transients.py:1242)
BUILD_STATE = 1                                         # the /_status build-state protocol (entry, build_start, errors, ...)
SOCK_PATH_MAX = 107                                     # bytes of a Unix socket path (sun_path 108 with its NUL)

# ---------------------------------------------------------------------------------------------------------------------------
# the per-thread environment
# ---------------------------------------------------------------------------------------------------------------------------


class ThreadEnv(MutableMapping):
    """os.environ for a process hosting many pages: reads consult the calling thread's page environment (use(env)), else
    the process environment. Writes go to the same mapping a read would consult. Threads started while an overlay is active
    inherit it (install patches threading.Thread)."""

    def __init__(self, base):
        self._base = dict(base)
        self._tl = threading.local()

    def _cur(self):
        env = getattr(self._tl, 'env', None)
        return self._base if env is None else env

    def current_overlay(self):
        return getattr(self._tl, 'env', None)

    @contextlib.contextmanager
    def use(self, env):
        prev = getattr(self._tl, 'env', None)
        self._tl.env = env
        try:
            yield
        finally:
            self._tl.env = prev

    def __getitem__(self, k):
        return self._cur()[k]

    def __setitem__(self, k, v):
        self._cur()[k] = v

    def __delitem__(self, k):
        del self._cur()[k]

    def __iter__(self):
        return iter(dict(self._cur()))

    def __len__(self):
        return len(self._cur())

    def copy(self):
        return dict(self._cur())


_ENV = None
_ORIG_THREAD_INIT = None


def install_env():
    """Replace os.environ by a ThreadEnv (once per process) and make new threads inherit their creator's overlay."""
    global _ENV, _ORIG_THREAD_INIT
    if _ENV is not None:
        return _ENV
    _ENV = ThreadEnv(os.environ)
    os.environ = _ENV
    _ORIG_THREAD_INIT = threading.Thread.__init__

    def _init(self, *a, **k):
        _ORIG_THREAD_INIT(self, *a, **k)
        ov = _ENV.current_overlay()
        if ov is not None:
            run = self.run

            def _run():
                with _ENV.use(ov):
                    run()
            self.run = _run
    threading.Thread.__init__ = _init
    return _ENV


def page_env(base, overrides):
    """The environment a page ran under: the process environment without any P3_* variable, plus the page's own."""
    env = {k: v for k, v in base.items() if not k.startswith('P3_')}
    env.update(overrides or {})
    return env


# ---------------------------------------------------------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------------------------------------------------------

REQUIRED = ('id', 'kind')
KINDS = ('review', 'static')


class RegistryError(RuntimeError):
    pass


def load_registry(path):
    """{page_id: entry}. An entry: id, kind ('review' | 'static'), and for a review page code (the classify_transients.py it
    runs), argv (its arguments as today, without --port and --inspect), env (its P3_* variables), legacy_port (the port it was
    served on: cookie key and link mapping), greenlit (shown on the inspection listener), title; for a static site root (a
    directory) and greenlit. Refuses a malformed registry rather than serving part of it."""
    with open(path) as f:
        reg = json.load(f)
    return check_registry(reg, path)


def registry_state(st):
    """(inode, size, modification time in ns) of a registry file's os.stat result. Any of them changed = the registry
    changed: the time alone misses a second write within the same second on Lustre (whole seconds); every writer replaces
    the file (a new inode: register_page.py, registry_merge.py + mv, lib/serve/register.py)."""
    return (st.st_ino, st.st_size, st.st_mtime_ns)


def read_registry(path):
    """(registry_state, {page_id: entry}) of one open of the file: the state is the state of the bytes read."""
    with open(path) as f:
        state = registry_state(os.fstat(f.fileno()))
        reg = json.load(f)
    return state, check_registry(reg, path)


def check_registry(reg, path):
    """load_registry's checks on a parsed registry ({'pages': [...]}); ``path`` names it in the refusals."""
    out = collections.OrderedDict()
    for e in reg.get('pages', []):
        missing = [k for k in REQUIRED if not e.get(k)]
        if missing:
            raise RegistryError(f'{path}: an entry lacks {missing}: {e}')
        if e['kind'] not in KINDS:
            raise RegistryError(f'{path}: {e["id"]}: kind {e["kind"]!r} not in {KINDS}')
        if not re.match(r'^[A-Za-z0-9_.-]+$', e['id']):
            raise RegistryError(f'{path}: page id {e["id"]!r}: letters, digits, _ . - only')
        if e['id'] in out:
            raise RegistryError(f'{path}: page id {e["id"]!r} twice')
        if e['kind'] == 'review':
            for k in ('code', 'argv'):
                if not e.get(k):
                    raise RegistryError(f'{path}: {e["id"]}: a review page needs {k}')
            if any(a in ('--port', '--inspect', '--host') for a in e['argv']):
                raise RegistryError(f'{path}: {e["id"]}: argv carries --port/--inspect/--host (the server sets them)')
        else:
            if not e.get('root') or not os.path.isdir(e['root']):
                raise RegistryError(f'{path}: {e["id"]}: static root {e.get("root")!r} is not a directory')
        out[e['id']] = e
    ports = collections.Counter(e.get(k) for e in out.values() for k in ('legacy_port', 'inspection_legacy_port') if e.get(k))
    dup = [p for p, n in ports.items() if n > 1]
    if dup:
        raise RegistryError(f'{path}: legacy ports used twice: {dup}')
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# response rewriting
# ---------------------------------------------------------------------------------------------------------------------------

ROUTES = ('source/', 'band/', 'img', 'metrics.json', 'label', 'list')
HUB_URLS = ('http://localhost:8765/', 'http://localhost:8795/')
# Gregg 2026-10-27: the per-person review lists ("Add to Prioritized transient", "Add to Visibility Data Required") are written from
# the inspection listener as well - a page's POST /list and POST /list/user; every other write stays refused there
INSPECTION_WRITES = ('/list', '/list/user')


def inspection_write_ok(method, path):
    m = re.match(r'^/p/[A-Za-z0-9_.-]+(/[^?]*)$', path.split('?', 1)[0])
    return method == 'POST' and m is not None and m.group(1) in INSPECTION_WRITES


def _route_re():
    # an absolute path of one of the page's routes inside a quoted attribute or string: "/source/..", '/img?..', "/label";
    # a route that is a bare name must end there ('/img' not '/imgs/...')
    alt = '|'.join(re.escape(r) if r.endswith('/') else re.escape(r) + r'(?=[?"\'#/])' for r in ROUTES)
    return re.compile(r'(["\'])/(' + alt + r')')


_ROUTE_RE = _route_re()
_ROOT_RE = re.compile(r'(href=|action=)(["\'])/(["\'?#])')          # href="/", action='/', href="/?x" - the page's index
_PORT_RE = re.compile(r'http://(?:localhost|127\.0\.0\.1):(\d+)/')


# the filters script keys its cookie and its saved settings by the browser's port (review_filters.py: "var PORT = location.port
# || '80', LS = 'filters_v1', COOK = 'flt_' + PORT"); under one listener every page would share them and the server, which
# reads flt_<legacy port>, would never see a page's filters - so they are keyed by the page's legacy port instead
JS_PORT = "location.port || '80'"
JS_LS = "LS = 'filters_v1'"


class RewriteError(RuntimeError):
    pass


def rewrite_body(text, base, port_map, hub='/', legacy_port=None):
    """The page's HTML as served under base ('/p/<page_id>'): absolute paths of its routes and its index gain base, the URL of
    every registered legacy port becomes that page's path, the hub URLs become hub, and the filters script is keyed by the
    page's legacy port. Nothing else changes. Refuses a filters script it cannot key (the page code changed under it)."""
    if legacy_port is not None and 'flt_' in text:
        if JS_PORT not in text or JS_LS not in text:
            raise RewriteError(f'{base}: the filters script no longer reads {JS_PORT!r} / {JS_LS!r}: cannot key it per page')
        text = text.replace(JS_PORT, f"'{int(legacy_port)}'").replace(JS_LS, f"LS = 'filters_v1_{int(legacy_port)}'")
    def _port(m):
        pid = port_map.get(int(m.group(1)))
        return f'/p/{pid}/' if pid else m.group(0)
    text = _PORT_RE.sub(_port, text)
    text = _ROUTE_RE.sub(lambda m: f'{m.group(1)}{base}/{m.group(2)}', text)
    text = _ROOT_RE.sub(lambda m: f'{m.group(1)}{m.group(2)}{base}/{m.group(3)}', text)
    for h in HUB_URLS:                          # last: the hub's '/' is not the page's index
        text = text.replace(h, hub)
    return text


def rewrite_location(loc, base, port_map, hub='/'):
    if loc.startswith('/'):
        return base + loc
    return rewrite_body(loc, base, port_map, hub)


def rewrite_links(text, port_map, hub='/'):
    """A static page's HTML: only the URLs of registered legacy ports and the hub URLs change (its own links are relative)."""
    for h in HUB_URLS:
        text = text.replace(h, hub)
    return _PORT_RE.sub(lambda m: f'/p/{port_map[int(m.group(1))]}/' if int(m.group(1)) in port_map else m.group(0), text)


# ---------------------------------------------------------------------------------------------------------------------------
# pages: one module instance each, built by its own main()
# ---------------------------------------------------------------------------------------------------------------------------


class _Stop(Exception):
    pass


class _NoServer:
    def __init__(self, *a, **k):
        raise _Stop()


_BUILD_LOCK = threading.Lock()         # sys.argv and sys.path are process-wide during a build


def _port_for(entry, inspect):
    """The port the page ran on today: the inspection twin's port for the inspection listener when it had one."""
    return int((entry.get('inspection_legacy_port') if inspect else None) or entry.get('legacy_port') or 0)


def _argv_for(entry, inspect):
    """The page's argv; for the inspection listener with the inspection twin's title (added when the review server has none)."""
    a = list(entry['argv'])
    if inspect and entry.get('inspection_title'):
        if '--title' in a:
            a[a.index('--title') + 1] = entry['inspection_title']
        else:
            a += ['--title', entry['inspection_title']]
    return a


def build_key(entry):
    """What of a registry entry its build depends on: all of it but the hub's title and the inspection listener's greenlight
    (the server's own business)."""
    return json.dumps({k: v for k, v in (entry or {}).items() if k not in ('title', 'greenlit')}, sort_keys=True)


def entry_md5(entry):
    """The identity of what a page is built from: the md5 of its build_key (reported per page by /_status)."""
    return hashlib.md5(build_key(entry).encode()).hexdigest()


def hold_path(entry):
    """<run>/HOLD_FILE of a review page's entry (its --run), or None for an entry without --run."""
    a = (entry or {}).get('argv') or []
    i = a.index('--run') + 1 if '--run' in a else len(a)
    return os.path.join(a[i], HOLD_FILE) if i < len(a) else None


def hold_of(entry):
    """The page's publish hold: hold_path when the file exists (followed: a link to a lifted hold is no hold), else None."""
    p = hold_path(entry)
    return p if p and os.path.exists(p) else None


class Page:
    """A built page: its module instance, its App, its environment and the lock of its labels directory."""

    def __init__(self, entry, inspect, mod, env, write_lock):
        self.entry, self.inspect, self.mod, self.env, self.write_lock = entry, inspect, mod, env, write_lock
        self.id = entry['id']
        self.base = f'/p/{self.id}'
        self.legacy_port = _port_for(entry, inspect)
        self.built = time.time()
        self.last_used = self.built
        self.build_s = self.prime_s = None
        self.build_start = None               # when its build started (Store._build): the build read the files after it
        self.gen = None                       # the registry generation its build started at (Store._build)


_BUILDS = iter(range(1, 1 << 62))            # every build's module name is its own: a rebuilt page and the one it replaces coexist


def build_page(entry, inspect, base_env, write_lock):
    """Build one page exactly as its process starts today: module instance, the page's environment, main() with its argv."""
    env = page_env(base_env, entry.get('env'))
    code = entry['code']
    name = f'_twoport_{entry["id"].replace("-", "_").replace(".", "_")}_{"i" if inspect else "r"}_{next(_BUILDS)}'
    with _BUILD_LOCK, _ENV.use(env):
        spec = importlib.util.spec_from_file_location(name, code)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        cdir = os.path.dirname(os.path.abspath(code))
        added = cdir not in sys.path
        if added:
            sys.path.insert(0, cdir)
        saved = sys.argv
        ok = False
        try:
            try:
                spec.loader.exec_module(mod)
            except SystemExit as exc:            # ... nor one that refuses on import: a build failure, never the end of a thread
                raise RuntimeError(f'{entry["id"]}: the page refused to load: {exc}') from None
            mod.ThreadingHTTPServer = _NoServer
            sys.argv = [os.path.basename(code)] + _argv_for(entry, inspect) + (['--inspect'] if inspect else []) + \
                ['--port', str(_port_for(entry, inspect) or 1)]
            try:
                mod.main()
                raise RuntimeError(f'{entry["id"]}: main() returned without reaching its server')
            except _Stop:
                pass
            except SystemExit as exc:            # a page that refuses to start (sys.exit('REFUSING: ...')) never exits the server
                raise RuntimeError(f'{entry["id"]}: the page refused to start: {exc}') from None
            if getattr(getattr(mod, 'Handler', None), 'app', None) is None:
                raise RuntimeError(f'{entry["id"]}: built, but its Handler has no app')
            ok = True
        finally:
            sys.argv = saved
            if added:
                try:
                    sys.path.remove(cdir)
                except ValueError:
                    pass
            if not ok:                           # a failed build leaves no module behind (every build's name is its own)
                sys.modules.pop(name, None)
    return Page(entry, inspect, mod, env, write_lock)


def prime_page(pg):
    """The indices a page builds on its first source request, built now (the page's App.prime, when its code has one), under the
    page's environment as a request runs (WEB SPEED step 2, 2026-10-30: the 1-minute combined Stokes V view spent 40 s of its first
    source page in the sibling and rejected-source indices). A failure is reported and the page still serves: the index is then
    built on first use, as before."""
    fn = getattr(getattr(pg.mod.Handler, 'app', None), 'prime', None)
    if fn is None:
        return
    t = time.time()
    try:
        with _ENV.use(pg.env):
            fn()
    except (Exception, SystemExit) as exc:                 # noqa: BLE001 - the page serves without its indices built
        print(f'two-port: {pg.id}: prime failed ({type(exc).__name__}: {exc}); its indices build on first use', flush=True)
    pg.prime_s = round(time.time() - t, 1)


# ---------------------------------------------------------------------------------------------------------------------------
# one request through the page's own Handler
# ---------------------------------------------------------------------------------------------------------------------------


class _FakeServer:
    def __init__(self, port):
        self.server_port = port
        self.server_address = ('127.0.0.1', port)


def run_request(page, method, path, headers, body=b''):
    """(status, [(header, value)], body bytes) of the page module's Handler for one request, on a fake socket."""
    H = page.mod.Handler
    h = H.__new__(H)
    h.server = _FakeServer(page.legacy_port)
    h.client_address = ('127.0.0.1', 0)
    h.command, h.path, h.request_version = method, path, 'HTTP/1.1'
    h.requestline = f'{method} {path} HTTP/1.1'
    h.close_connection = True
    msg = email.message.Message()
    for k, v in (headers or {}).items():
        msg[k] = v
    if body and 'Content-Length' not in msg:
        msg['Content-Length'] = str(len(body))
    h.headers = msg
    h.rfile, h.wfile = io.BytesIO(body), io.BytesIO()
    with _ENV.use(page.env):
        getattr(h, 'do_' + method)()
        if hasattr(h, '_headers_buffer') and h._headers_buffer:
            h.flush_headers()
    raw = h.wfile.getvalue()
    head, _, payload = raw.partition(b'\r\n\r\n')
    lines = head.decode('iso-8859-1').split('\r\n')
    status = int(lines[0].split()[1])
    hdrs = [tuple(x.split(': ', 1)) for x in lines[1:] if ': ' in x]
    return status, hdrs, payload


# ---------------------------------------------------------------------------------------------------------------------------
# code trees
# ---------------------------------------------------------------------------------------------------------------------------


def tree_of(entry):
    """A review page's code tree: the directory of its code, symlinks resolved."""
    return os.path.dirname(os.path.realpath(entry['code']))


def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


_PY_ROOTS = tuple(sorted({os.path.realpath(x) + os.sep for x in (sys.prefix, sys.base_prefix, sys.exec_prefix)}))


def project_modules():
    """{name: realpath} of every module in sys.modules loaded from a .py file outside the Python installation - the project
    code a page imported (its tree's helpers, clv15, the exopipe package) - the page modules themselves excluded."""
    out = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None) or ''
        if not f or name.startswith('_twoport_') or name == '__main__':
            continue
        rf = os.path.realpath(f)
        if rf.startswith(_PY_ROOTS) or not rf.endswith('.py'):
            continue
        out[name] = rf
    return out


@contextlib.contextmanager
def post_lock(thread_lock, labels_dir):
    """POSTs to one labels directory, one at a time across every backend: the in-process lock, then a file lock named by the
    directory in LOCK_DIR."""
    os.makedirs(LOCK_DIR, exist_ok=True)
    p = os.path.join(LOCK_DIR, hashlib.md5(labels_dir.encode()).hexdigest() + '.lock')
    with thread_lock:
        with open(p, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------------------------------------------------------
# the page store
# ---------------------------------------------------------------------------------------------------------------------------


class Store:
    """Pages built on first request and kept while used. A page requested within keep_s is never evicted; past max_pages the
    least recently used page goes. At a registry change every built page is rebuilt; until its rebuild is done, the page built
    before answers its requests, unless its own entry changed (reload)."""

    REBUILD_GRACE_S = 30.0                    # the rebuild thread starts this long after a registry change (see _rebuild_loop)
    REBUILD_QUIET_S = 10.0                    # ... and this long after the last build a request waited for

    def __init__(self, registry_path, base_env=None, max_pages=80, keep_s=1800, tree=None, builds=True, publish_hold=False):
        install_env()
        self.publish_hold = bool(publish_hold)                 # the publish hold option (off: production, a banner only)
        self.tree = os.path.realpath(tree) if tree else None   # a backend builds the pages of this code tree only
        self.builds = builds                                   # the front builds none
        self.code_report = {}                 # (page_id, inspect) -> {'code': [path, md5], 'helpers': {name: [path, md5]}}
        self._mod_md5 = {}                    # id(module) -> (path, md5) when first seen: the bytes it was imported from
        self.registry_path = registry_path
        self.base_env = dict(base_env if base_env is not None else _ENV._base)
        self.max_pages, self.keep_s = max_pages, keep_s
        self.lock = threading.RLock()
        self.pages = {}                       # (page_id, inspect) -> Page
        self.building = {}                    # (page_id, inspect) -> threading.Event
        self.write_locks = collections.defaultdict(threading.Lock)
        self.errors = {}                      # (page_id, inspect) -> text of the last build failure
        self.error_info = {}                  # (page_id, inspect) -> {'t', 'entry'}: when, and the entry_md5 it was built from
        self.gen = 0                          # the registry generation: one more at every registry change
        self.stale = set()                    # built pages older than the registry: they answer requests until rebuilt
        self.held = set()                     # stale pages kept on their build by their run's publish hold (hold_of)
        self.builds_done = 0                  # build_page calls finished (built or failed): the builds' progress
        self.rebuild_q = collections.deque()  # stale pages, rebuilt one at a time by the rebuild thread, most recently used first
        self._rebuilder = None
        self.reload_t = 0.0                   # when the registry last changed
        self.fg_pending = 0                   # builds a request is waiting for (get): the rebuild thread lets them go first
        self.fg_last_t = 0.0                  # when the last of them ended
        self.reg, self.reg_mtime, self.reg_state = collections.OrderedDict(), None, None
        self.reload()

    def reload(self):
        """Re-read the registry when it changed. Every built page is rebuilt: a page builds its cross-tier and sibling links from
        the other runs' catalogues when it starts, so a run published or replaced must reach the pages already built (the
        2026-10-17 parity found live pages days old without the 10-minute LSTs published since).
        Until its rebuild is done, the page built before answers requests - a POST too: it goes to the page that showed its form,
        so labels are written in the order they are sent - and only a GET carrying FRESH_HEADER waits for the rebuilt page (WEB
        SPEED step 2, Gregg 2026-10-30 "proceed on all steps": a registry change dropped every page, and the large ones then took
        minutes on their next request - the 10-s combined Stokes I view 394 s). The rebuilds run one at a time in this process's
        rebuild thread, the most recently used page first, after the pages requests are waiting for (_rebuild_loop). A page
        whose OWN entry changed (build_key: a new run, title, code or environment), left the registry or no longer belongs to
        this tree is dropped now: its next request builds it, as before, so nobody is shown the page an entry replaced (a
        serve's identity check after it registers a page reads the new page).
        A change is a change of the file's inode, size or modification time (registry_state; one stat per request, the file
        read only when it changed). A stale page whose run holds the publish hold (hold_of) is not queued: it keeps its build
        while the hold stands (self.held)."""
        if registry_state(os.stat(self.registry_path)) == self.reg_state:
            return
        state, reg = read_registry(self.registry_path)
        with self.lock:
            self.gen += 1
            self.reload_t = time.time()
            for key in sorted(self.pages, key=lambda k: -self.pages[k].last_used):
                e = reg.get(key[0])
                if e is None or e['kind'] != 'review' or (self.tree is not None and tree_of(e) != self.tree) \
                        or build_key(e) != build_key(self.pages[key].entry):
                    self._drop(key)
                else:
                    self.stale.add(key)
                    if not self._held(key, e):
                        self._queue(key)
            # the page code's helper modules (review_filters and the like, imported once and shared by every page) are reloaded
            # too, so new page code reaches the rebuilt pages: a deploy is a registry change (touch registry.json). A page built
            # before keeps the helper objects it imported until it is replaced.
            dirs = {os.path.dirname(os.path.realpath(e['code'])) + os.sep for e in reg.values() if e['kind'] == 'review'}
            for name, mod in list(sys.modules.items()):
                f = getattr(mod, '__file__', None) or ''
                if f and any(os.path.realpath(f).startswith(d) for d in dirs) and not name.startswith('exopipe.'):
                    sys.modules.pop(name, None)
            importlib.invalidate_caches()
            self._mod_md5.clear()
            self.reg, self.reg_mtime, self.reg_state = reg, state[2] / 1e9, state

    def _held(self, key, entry):
        """True when the stale page ``key`` is held now by its run's publish hold (the option on and the file there: it is
        looked at now); keeps self.held current (under self.lock)."""
        if self.publish_hold and hold_of(entry):
            self.held.add(key)
            return True
        self.held.discard(key)
        return False

    def _drop(self, key):
        """Forget a built page (under self.lock)."""
        pg = self.pages.pop(key, None)
        if pg is not None:
            sys.modules.pop(pg.mod.__name__, None)
        self.stale.discard(key)
        self.held.discard(key)
        self.code_report.pop(key, None)

    def _queue(self, key):
        """Queue a stale page for the rebuild thread and start the thread when none runs (under self.lock)."""
        if key not in self.rebuild_q:
            self.rebuild_q.append(key)
        if self._rebuilder is None or not self._rebuilder.is_alive():
            self._rebuilder = threading.Thread(target=self._rebuild_loop, name='two-port-rebuild', daemon=True)
            self._rebuilder.start()

    def _rebuild_loop(self):
        """Rebuild the queued stale pages one at a time. A build holds the process's build lock, so the pages requests are waiting
        for go first: the thread starts REBUILD_GRACE_S after a registry change (a registration's new or changed pages - dropped,
        not stale - are requested by its warm-up meanwhile, one after the other) and REBUILD_QUIET_S after the last build a
        request waited for, and never starts a rebuild while one is pending. A request that comes while a rebuild runs waits for
        the rest of that one build."""
        while True:
            while True:
                with self.lock:
                    now = time.time()
                    wait = max(self.reload_t + self.REBUILD_GRACE_S - now, self.fg_last_t + self.REBUILD_QUIET_S - now,
                               0.2 if self.fg_pending else 0.0)
                if wait <= 0:
                    break
                time.sleep(min(wait, 1.0))
            with self.lock:
                if not self.rebuild_q:
                    self._rebuilder = None
                    return
                key = self.rebuild_q.popleft()
                if key not in self.stale or key not in self.pages or key in self.building:
                    continue                  # rebuilt meanwhile (a request that waited for it), dropped, or being built now
                if self._held(key, self.pages[key].entry):
                    continue                  # its run's publish hold was set after it was queued: it keeps its build
                ev = self.building[key] = threading.Event()
            try:
                self._build(key, ev)
            except (Exception, SystemExit) as exc:             # noqa: BLE001 - recorded in self.errors; the next request retries
                print(f'two-port: rebuild of {key[0]}{":inspection" if key[1] else ""} failed: {exc}', flush=True)

    def port_map(self):
        out = {}
        for e in self.reg.values():
            for k in ('legacy_port', 'inspection_legacy_port'):
                if e.get(k):
                    out[int(e[k])] = e['id']
        return out

    def entry(self, pid):
        return self.reg.get(pid)

    def labels_dir(self, entry):
        a = entry.get('argv') or []
        if '--labels-dir' in a:
            return os.path.realpath(a[a.index('--labels-dir') + 1])
        run = a[a.index('--run') + 1] if '--run' in a else ''
        return os.path.realpath(os.path.join(run, 'judgements'))

    def get(self, pid, inspect, fresh=False):
        """The page to answer a request: its current build; a page built before the last registry change answers too unless
        fresh (a GET carrying FRESH_HEADER), which waits for the rebuilt page. A page held by its run's publish hold answers
        from its build, fresh or not; once the hold is lifted its rebuild is queued here."""
        key = (pid, bool(inspect))
        e = self.reg.get(pid)
        if e is None or e['kind'] != 'review':
            return None
        if not self.builds:
            raise RuntimeError(f'{pid}: the front builds no page (a backend per code tree does)')
        if self.tree is not None and tree_of(e) != self.tree:
            raise RuntimeError(f'{pid}: code tree {tree_of(e)} is not this backend\'s ({self.tree})')
        while True:
            with self.lock:
                pg = self.pages.get(key)
                held = pg is not None and key in self.stale and self._held(key, pg.entry)
                if pg is not None and (held or not (fresh and key in self.stale)):
                    pg.last_used = time.time()
                    if key in self.stale and not held and key not in self.building and key not in self.rebuild_q:
                        self._queue(key)                       # never left stale: its rebuild is queued
                    return pg
                ev = self.building.get(key)
                if ev is None:
                    ev = self.building[key] = threading.Event()
                    mine = True
                    self.fg_pending += 1
                else:
                    mine = False
            if not mine:
                ev.wait()
                if key in self.errors and key not in self.pages:
                    raise RuntimeError(self.errors[key])
                continue
            try:
                pg = self._build(key, ev)
            finally:
                with self.lock:
                    self.fg_pending -= 1
                    self.fg_last_t = time.time()
            pg.last_used = time.time()
            return pg

    def _build(self, key, ev):
        """Build the page of key - its building event ev is this thread's - with its indices (prime_page), from the registry
        entry of now; it then replaces the page built before, if any. A failed build drops the page built before, so the next
        request builds it again and shows the failure, as a first build does. A build whose entry changed while it ran is given
        to the request that asked for it and to nobody else."""
        pid, inspect = key
        pg = e = None
        t = time.time()
        try:
            with self.lock:
                e, gen = self.reg.get(pid), self.gen
            if e is None or e['kind'] != 'review' or (self.tree is not None and tree_of(e) != self.tree):
                raise RuntimeError(f'{pid}: no longer a review page of this tree in the registry')
            t = time.time()
            try:
                pg = build_page(e, inspect, self.base_env, self.write_locks[self.labels_dir(e)])
            finally:
                with self.lock:
                    self.builds_done += 1             # a build of the process finished (its page's report may wait)
            pg.build_s, pg.gen, pg.build_start = round(time.time() - t, 1), gen, t
            prime_page(pg)
            report = self._report(e)
            with self.lock:
                cur = self.reg.get(pid)
                if cur is None or cur['kind'] != 'review' or build_key(cur) != build_key(e):
                    self.errors.pop(key, None)                 # its waiters build the new entry; no failure of an older one
                    self.error_info.pop(key, None)
                    sys.modules.pop(pg.mod.__name__, None)     # the asking request holds the page; nothing else keeps it
                    return pg
                old = self.pages.get(key)
                if old is not None:                            # a rebuild keeps the page's own last use (eviction, rebuild order)
                    sys.modules.pop(old.mod.__name__, None)
                    pg.last_used = old.last_used
                self.pages[key] = pg
                self.code_report[key] = report
                self.errors.pop(key, None)
                self.error_info.pop(key, None)
                if gen == self.gen:
                    self.stale.discard(key)
                    self.held.discard(key)
                else:                                          # the registry changed again while this page was built
                    self.stale.add(key)
                    self._queue(key)
                self._evict()
            return pg
        except Exception as exc:                               # one page's failure never takes down the process
            with self.lock:
                self.errors[key] = f'{pid}: {type(exc).__name__}: {exc}'
                self.error_info[key] = {'t': t, 'entry': entry_md5(e) if e else None}
                self._drop(key)
                if pg is not None and self.pages.get(key) is not pg:
                    sys.modules.pop(pg.mod.__name__, None)
            raise
        finally:
            with self.lock:
                self.building.pop(key, None)
            ev.set()

    def _report(self, e):
        """The code a page was built from: its own file and every project module in this process (project_modules), each with
        the md5 of the bytes it was imported from (recorded when the module is first seen)."""
        helpers = {}
        with _BUILD_LOCK:
            for name, path in sorted(project_modules().items()):
                mod = sys.modules.get(name)
                if mod is None:
                    continue
                if id(mod) not in self._mod_md5:
                    self._mod_md5[id(mod)] = (path, md5_file(path))
                helpers[name] = list(self._mod_md5[id(mod)])
        return {'code': [os.path.realpath(e['code']), md5_file(e['code'])], 'helpers': helpers}

    def _evict(self):
        now = time.time()
        if len(self.pages) <= self.max_pages:
            return
        for key, pg in sorted(self.pages.items(), key=lambda kv: kv[1].last_used):
            if len(self.pages) <= self.max_pages:
                break
            if now - pg.last_used > self.keep_s:
                self._drop(key)


# ---------------------------------------------------------------------------------------------------------------------------
# the listeners
# ---------------------------------------------------------------------------------------------------------------------------


def serve_request(store, inspect, method, path, headers, body=b''):
    """(status, headers, body) for one request to a listener: '/' the hub, /p/<id>/... a review page, /s/<id>/... a static
    site. The inspection listener serves greenlit entries only and refuses every write."""
    store.reload()
    if path == '/' or path.startswith('/?'):
        hub = store.entry('hub_inspection' if inspect else 'hub')
        if hub is not None and hub['kind'] == 'static':          # today's hub site, its port links rewritten
            return static_file(hub, '/', store.port_map())
        return 200, [('Content-Type', 'text/html; charset=utf-8')], hub_page(store, inspect).encode()
    m = re.match(r'^/(p|s)/([A-Za-z0-9_.-]+)(/.*)?$', path.split('?', 1)[0])
    if not m:
        # every other path is a file of the listener's hub site (the pages, reports and folders the plain server on the hub
        # port served beside index.html: data_processing.html, nights/, periodicity_*/ ...), its port links rewritten
        hub = store.entry('hub_inspection' if inspect else 'hub')
        if hub is not None and hub['kind'] == 'static':
            return static_file(hub, path, store.port_map())
        return 404, [('Content-Type', 'text/plain')], b'not found'
    kind, pid, rest = m.group(1), m.group(2), m.group(3) or '/'
    e = store.entry(pid)
    if e is None or (inspect and not e.get('greenlit')):
        return 404, [('Content-Type', 'text/plain')], b'unknown page'
    if inspect and method != 'GET' and not inspection_write_ok(method, path):
        return 403, [('Content-Type', 'text/plain; charset=utf-8')], b'inspection mode: this view is read-only and records nothing'
    query = path[len(f'/{kind}/{pid}') + len(rest):] if '?' in path else ''
    if kind == 's':
        return static_file(e, rest, store.port_map()) if e['kind'] == 'static' else (404, [('Content-Type', 'text/plain')], b'unknown page')
    if e['kind'] != 'review':
        return 404, [('Content-Type', 'text/plain')], b'unknown page'
    fresh = method == 'GET' and any(k.lower() == FRESH_HEADER.lower() and v.strip() == '1' for k, v in (headers or {}).items())
    try:
        pg = store.get(pid, inspect, fresh=fresh)
    except Exception as exc:                                    # the page's error page, the process carries on
        return 500, [('Content-Type', 'text/plain; charset=utf-8')], f'page {pid} could not be built: {exc}'.encode()
    sub = rest + (('?' + path.split('?', 1)[1]) if '?' in path else '')
    stale_built = pg.built if pg.gen is not None and pg.gen != store.gen else None    # a page older than the registry
    held = hold_path(pg.entry) if stale_built is not None and (pid, bool(inspect)) in store.held else None
    lock = post_lock(pg.write_lock, store.labels_dir(pg.entry)) if method == 'POST' else contextlib.nullcontext()
    with lock:
        status, hdrs, payload = run_request(pg, method, sub, headers, body)
    pm = store.port_map()
    out = []
    for k, v in hdrs:
        if k.lower() == 'location':
            v = rewrite_location(v, pg.base, pm)
        if k.lower() == 'content-length':
            continue
        out.append((k, v))
    if stale_built is not None:
        out.append((STALE_HEADER, time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(stale_built))))
    if held:
        out.append((HELD_HEADER, held))
    ctype = dict((k.lower(), v) for k, v in hdrs).get('content-type', '')
    if ctype.startswith('text/html'):
        payload = rewrite_body(payload.decode('utf-8'), pg.base, pm, legacy_port=pg.legacy_port).encode('utf-8')
    return status, out, payload


def static_file(entry, rest, port_map=None):
    root = os.path.realpath(entry['root'])
    rel = urllib.parse.unquote(rest.split('?', 1)[0]).lstrip('/') or '.'      # the root: its index.html, else its listing
    # The path as written must stay inside the site ('..' never leaves it); a symlink the site itself holds is then followed, as
    # the plain hub server (python -m http.server) did. The data-processing page's QA images and movies are such links into Lustre
    # (review_hub/qa_assets/*) and answered 404 from the move to two ports until 2026-10-27 (Gregg: that page on both listeners).
    lexical = os.path.normpath(os.path.join(root, rel))
    if not (lexical == root or lexical.startswith(root + os.sep)):
        return 404, [('Content-Type', 'text/plain')], b'not found'
    full = os.path.realpath(lexical)
    if os.path.isdir(full):
        if not os.path.isfile(os.path.join(full, 'index.html')):    # a folder without an index: listed, as the plain server did
            names = sorted(os.listdir(full))
            rows = ''.join(f'<li><a href="{urllib.parse.quote(n)}{"/" if os.path.isdir(os.path.join(full, n)) else ""}">'
                           f'{html.escape(n)}</a></li>' for n in names)
            body = f'<!doctype html><html><body><h1>{html.escape(rel)}</h1><ul>{rows}</ul></body></html>'
            return 200, [('Content-Type', 'text/html; charset=utf-8')], body.encode()
        full = os.path.join(full, 'index.html')
    if not os.path.isfile(full):
        return 404, [('Content-Type', 'text/plain')], b'not found'
    ext = os.path.splitext(full)[1].lower()
    ctype = {'.html': 'text/html; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg', '.css': 'text/css',
             '.js': 'application/javascript', '.json': 'application/json', '.csv': 'text/csv', '.txt': 'text/plain',
             '.svg': 'image/svg+xml', '.gif': 'image/gif', '.pdf': 'application/pdf'}.get(ext, 'application/octet-stream')
    with open(full, 'rb') as f:
        data = f.read()
    if ext == '.html' and port_map is not None:
        data = rewrite_links(data.decode('utf-8', errors='replace'), port_map).encode('utf-8')
    return 200, [('Content-Type', ctype)], data


def hub_page(store, inspect):
    """A plain index of the registry (the 2026-09-12 hub layout is rendered by the hub builder from the same registry)."""
    rows = []
    for e in store.reg.values():
        if inspect and not e.get('greenlit'):
            continue
        href = f'/{"p" if e["kind"] == "review" else "s"}/{e["id"]}/'
        rows.append(f'<li><a href="{href}">{html.escape(e.get("title") or e["id"])}</a></li>')
    title = 'Inspection hub' if inspect else 'Review hub'
    return f'<!doctype html><html><head><title>{title}</title></head><body><h1>{title}</h1><ul>{"".join(rows)}</ul></body></html>'


def make_handler(fn):
    """A request handler class answering every GET and POST with fn(method, path, headers, body) -> (status, headers, body)."""
    from http.server import BaseHTTPRequestHandler

    class Router(BaseHTTPRequestHandler):
        def _go(self, method):
            n = int(self.headers.get('Content-Length') or 0)
            body = self.rfile.read(n) if n else b''
            hdrs = {k: v for k, v in self.headers.items()}
            status, out, payload = fn(method, self.path, hdrs, body)
            self.send_response(status)
            for k, v in out:
                self.send_header(k, v)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self._go('GET')

        def do_POST(self):
            self._go('POST')

        def address_string(self):           # a Unix-socket client has no address
            return 'local'

        def log_message(self, fmt, *args):
            pass
    return Router


# ---------------------------------------------------------------------------------------------------------------------------
# a backend: the pages of one code tree, on a Unix socket
# ---------------------------------------------------------------------------------------------------------------------------


def backend_status(store):
    """The backend's report: per built page the code it was built from ('pages'); per built page its build and index times and
    whether it is older than the registry ('built': stale pages answer requests until their rebuild, queued in 'rebuild_queue')
    or held by its run's publish hold ('held'). The BUILD STATE of a page (lib/serve/register.wait_built): 'built' also names
    what the page was built from ('entry': entry_md5) and when its build started ('build_start', epoch s); 'building' = the
    builds under way or waiting for the build lock; 'errors' = the last failed build per page (its text, the entry it was
    built from, when it started); 'builds_done' = the builds finished so far (a page's own completion can wait behind the
    next build: its report takes the build lock)."""
    def _k(pid, insp):
        return f'{pid}{":inspection" if insp else ""}'
    with store.lock:
        pages = {_k(*key): store.code_report.get(key) for key in store.pages}
        built = {_k(*key): {'built': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(pg.built)), 'build_s': pg.build_s,
                            'prime_s': pg.prime_s, 'stale': key in store.stale, 'held': key in store.held,
                            'entry': entry_md5(pg.entry), 'build_start': pg.build_start} for key, pg in store.pages.items()}
        queue = [_k(*key) for key in store.rebuild_q]
        building = sorted(_k(*key) for key in store.building)
        errors = {_k(*key): dict(store.error_info.get(key) or {}, error=text) for key, text in store.errors.items()}
        held = sorted(_k(*key) for key in store.held)
        state = list(store.reg_state) if store.reg_state else None
        done = store.builds_done
    return {'tree': store.tree, 'pid': os.getpid(), 'build_state': BUILD_STATE, 'publish_hold': store.publish_hold,
            'registry_mtime': store.reg_mtime, 'registry_state': state, 'gen': store.gen, 'pages': pages, 'built': built,
            'rebuild_queue': queue, 'building': building, 'held': held, 'errors': errors, 'builds_done': done}


def backend_request(store, method, path, headers, body=b''):
    """One forwarded request: the listener it came to is in LISTENER_HEADER (set by the front, never by a browser)."""
    if path == STATUS_PATH:
        try:
            store.reload()                       # the report is of the registry of now (a request would load it as well)
        except Exception:                        # noqa: BLE001 - an unreadable registry fails the page requests, not the report
            pass
        return 200, [('Content-Type', 'application/json')], json.dumps(backend_status(store), indent=1).encode()
    who = next((v for k, v in headers.items() if k.lower() == LISTENER_HEADER.lower()), None)
    inspect = who != 'review'                                     # anything but the front's 'review' is the read-only listener
    hdrs = {k: v for k, v in headers.items() if k.lower() != LISTENER_HEADER.lower()}
    return serve_request(store, inspect, method, path, hdrs, body)


class _UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def serve_backend(registry_path, tree, socket_path, publish_hold=False):
    """Serve the pages of one code tree on a Unix socket until killed."""
    # the page code imports production's exopipe package (classify_transients: from exopipe.common import phase2_config); this
    # process was started from the refactor's tree, whose package is also named exopipe - unbound here, so the page code
    # resolves the name exactly as its own process does today
    for name in [n for n in sys.modules if n == 'exopipe' or n.startswith('exopipe.')]:
        sys.modules.pop(name, None)
    sys.path[:] = [x for x in sys.path if os.path.realpath(x or '.') != os.path.realpath(REPO_ROOT)]
    importlib.invalidate_caches()
    store = Store(registry_path, tree=tree, publish_hold=publish_hold)
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass
    srv = _UnixHTTPServer(socket_path, make_handler(lambda m, p, h, b: backend_request(store, m, p, h, b)))
    os.chmod(socket_path, 0o600)
    print(f'two-port backend {os.getpid()}: code tree {store.tree} on {socket_path}', flush=True)
    srv.serve_forever()


class _UDSConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout=900):
        super().__init__('localhost', timeout=timeout)
        self._path = path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self._path)
        self.sock = s


def _uds_get(path, url, timeout=30):
    c = _UDSConnection(path, timeout=timeout)
    try:
        c.request('GET', url)
        r = c.getresponse()
        return r.status, r.read()
    finally:
        c.close()


# ---------------------------------------------------------------------------------------------------------------------------
# the front: the two listeners, one backend per code tree
# ---------------------------------------------------------------------------------------------------------------------------


class Backends:
    """One backend process per code tree: started on first use, restarted when it died, stopped when its tree leaves the
    registry. Each inherits the front's environment, CPU affinity and niceness."""

    def __init__(self, registry_path, sock_dir=None, python=None, start_timeout=120, publish_hold=False):
        self.registry_path = os.path.abspath(registry_path)
        self.sock_dir = sock_dir or SOCK_DIR
        self.publish_hold = bool(publish_hold)
        longest = self.sock_for('')
        if len(longest.encode()) > SOCK_PATH_MAX:      # a backend could never bind it: every page would answer 502
            raise ValueError(f'socket directory {self.sock_dir} is too long: {longest} has {len(longest.encode())} bytes, a Unix '
                             f'socket path at most {SOCK_PATH_MAX} - give a shorter --sock-dir')
        os.makedirs(self.sock_dir, exist_ok=True)
        self.python = python or sys.executable
        self.start_timeout = start_timeout
        self.procs = {}                                   # tree -> (Popen, socket path)
        self.locks = collections.defaultdict(threading.Lock)

    def sock_for(self, tree):
        return os.path.join(self.sock_dir, 'tp_' + hashlib.md5(tree.encode()).hexdigest()[:16] + '.sock')

    def get(self, tree):
        """The socket of the tree's running backend, starting it when needed."""
        with self.locks[tree]:
            cur = self.procs.get(tree)
            if cur is not None and cur[0].poll() is None:
                return cur[1]
            path = self.sock_for(tree)
            logp = path[:-5] + '.log'
            env = dict(_ENV._base) if _ENV is not None else dict(os.environ)
            with open(logp, 'a') as log:
                proc = subprocess.Popen([self.python, '-m', 'exopipe.phase3.lib.serve.two_port', 'backend', self.registry_path,
                                         '--tree', tree, '--socket', path] + (['--publish-hold'] if self.publish_hold else []),
                                        cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            t0 = time.time()
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'the backend of {tree} exited with {proc.returncode} (log {logp})')
                if os.path.exists(path):
                    try:
                        st, _ = _uds_get(path, STATUS_PATH, timeout=5)
                        if st == 200:
                            break
                    except OSError:
                        pass
                if time.time() - t0 > self.start_timeout:
                    proc.kill()
                    raise RuntimeError(f'the backend of {tree} did not answer within {self.start_timeout} s (log {logp})')
                time.sleep(0.1)
            self.procs[tree] = (proc, path)
            return path

    def stop(self, tree):
        with self.locks[tree]:
            cur = self.procs.pop(tree, None)
            if cur is None:
                return
            cur[0].terminate()
            try:
                cur[0].wait(10)
            except subprocess.TimeoutExpired:
                cur[0].kill()
                cur[0].wait()

    def sync(self, trees):
        """Stop the backends of trees that left the registry."""
        for tree in [t for t in list(self.procs) if t not in trees]:
            self.stop(tree)

    def stop_all(self):
        for tree in list(self.procs):
            self.stop(tree)

    def forward(self, tree, inspect, method, path, headers, body=b''):
        sock = self.get(tree)
        hdrs = {k: v for k, v in headers.items()
                if k.lower() not in (LISTENER_HEADER.lower(), 'content-length', 'connection', 'host', 'transfer-encoding')}
        hdrs[LISTENER_HEADER] = 'inspection' if inspect else 'review'
        c = _UDSConnection(sock)
        try:
            c.request(method, path, body=body or None, headers=hdrs)
            r = c.getresponse()
            payload = r.read()
            out = [(k, v) for k, v in r.getheaders() if k.lower() not in ('content-length', 'connection', 'date', 'server')]
            return r.status, out, payload
        finally:
            c.close()

    def status(self):
        out = []
        for tree, (proc, path) in sorted(self.procs.items()):
            try:
                st, b = _uds_get(path, STATUS_PATH)
                out.append(json.loads(b) if st == 200 else {'tree': tree, 'error': f'HTTP {st}'})
            except Exception as exc:                           # noqa: BLE001 - a report, never a failure of the front
                out.append({'tree': tree, 'pid': proc.pid, 'error': str(exc)})
        return out


def front_request(store, backends, inspect, method, path, headers, body=b''):
    """One request to a listener: review pages go to their tree's backend; the hub, static sites, unknown pages and the
    inspection listener's refusals are answered here; GET /_status reports the registry and every backend."""
    store.reload()
    backends.sync({tree_of(e) for e in store.reg.values() if e['kind'] == 'review'})
    if path.split('?', 1)[0] == '/_status':
        rep = {'registry': store.registry_path, 'build_state': BUILD_STATE, 'publish_hold': backends.publish_hold,
               'registry_mtime': store.reg_mtime, 'registry_state': list(store.reg_state) if store.reg_state else None,
               'backends': backends.status()}
        return 200, [('Content-Type', 'application/json')], json.dumps(rep, indent=1).encode()
    m = re.match(r'^/p/([A-Za-z0-9_.-]+)(/.*)?$', path.split('?', 1)[0])
    if m:
        e = store.entry(m.group(1))
        if e is not None and e['kind'] == 'review' and not (inspect and (not e.get('greenlit') or
                                                                         (method != 'GET' and not inspection_write_ok(method, path)))):
            try:
                return backends.forward(tree_of(e), inspect, method, path, headers, body)
            except Exception as exc:                           # noqa: BLE001 - one tree's failure never takes down the front
                return 502, [('Content-Type', 'text/plain; charset=utf-8')], f'page {e["id"]}: its backend failed: {exc}'.encode()
    return serve_request(store, inspect, method, path, headers, body)


def serve(registry_path, review_port, inspection_port, host='127.0.0.1', sock_dir=None, publish_hold=False):
    """Run both listeners until interrupted; the backends stop with the front."""
    from http.server import ThreadingHTTPServer
    backends = Backends(registry_path, sock_dir, publish_hold=publish_hold)     # refuses a socket directory too long first
    store = Store(registry_path, builds=False)
    srv = [ThreadingHTTPServer((host, review_port),
                               make_handler(lambda m, p, h, b: front_request(store, backends, False, m, p, h, b))),
           ThreadingHTTPServer((host, inspection_port),
                               make_handler(lambda m, p, h, b: front_request(store, backends, True, m, p, h, b)))]
    ts = [threading.Thread(target=s.serve_forever, daemon=True) for s in srv]
    for t in ts:
        t.start()
    print(f'two-port server: review http://{host}:{review_port}/  inspection http://{host}:{inspection_port}/  '
          f'({len(store.reg)} registry entries; one backend per code tree, sockets in {backends.sock_dir}'
          + ('; the publish hold ON' if publish_hold else '') + ')', flush=True)
    import signal

    def _term(*_a):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    try:
        for t in ts:
            t.join()
    except KeyboardInterrupt:
        for s in srv:
            s.shutdown()
    finally:
        backends.stop_all()


# ---------------------------------------------------------------------------------------------------------------------------
# seeding the registry from the servers running today
# ---------------------------------------------------------------------------------------------------------------------------


def _proc_args(pid):
    with open(f'/proc/{pid}/cmdline', 'rb') as f:
        a = [x.decode() for x in f.read().split(b'\0') if x]
    with open(f'/proc/{pid}/environ', 'rb') as f:
        env = dict(x.decode(errors='replace').split('=', 1) for x in f.read().split(b'\0') if b'=' in x)
    cwd = os.readlink(f'/proc/{pid}/cwd')
    return a, env, cwd


def _opt(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def seed_from_proc(script='classify_transients.py'):
    """Registry entries for every review server running now: its code (absolute), argv (without --port / --inspect / --host),
    P3_* environment and port; an --inspect twin of the same argv marks its page greenlit and is not an entry of its own."""
    import glob
    found = []
    for d in glob.glob('/proc/[0-9]*'):
        try:
            a, env, cwd = _proc_args(os.path.basename(d))
        except OSError:
            continue
        i = next((k for k, x in enumerate(a) if x.endswith(script)), None)
        if i is None or not a[0].endswith(('python', 'python3')) and 'python' not in os.path.basename(a[0]):
            continue
        code = a[i] if os.path.isabs(a[i]) else os.path.realpath(os.path.join(cwd, a[i]))
        argv, skip = [], False
        rest = a[i + 1:]
        for k, x in enumerate(rest):
            if skip:
                skip = False
                continue
            if x in ('--port', '--host'):
                skip = True
                continue
            if x == '--inspect':
                continue
            argv.append(x)
        found.append(dict(code=code, argv=argv, env={k: v for k, v in env.items() if k.startswith('P3_')},
                          port=int(_opt(rest, '--port') or 0), inspect='--inspect' in rest))
    def _key(f):                                   # a twin differs only in --inspect, its port and its title
        a = list(f['argv'])
        if '--title' in a:
            k = a.index('--title')
            del a[k:k + 2]
        return json.dumps([f['code'], a, sorted(f['env'].items())])
    pages, twins = [], {}
    for f in found:
        if f['inspect']:
            twins[_key(f)] = f
    for f in sorted(found, key=lambda x: x['port']):
        if f['inspect']:
            continue
        tw = twins.pop(_key(f), None)
        pages.append(dict(id=f'p{f["port"]}', kind='review', code=f['code'], argv=f['argv'], env=f['env'],
                          legacy_port=f['port'], greenlit=tw is not None,
                          inspection_legacy_port=tw['port'] if tw else None,
                          inspection_title=_opt(tw['argv'], '--title') if tw else None,
                          title=_opt(f['argv'], '--title') or os.path.basename(_opt(f['argv'], '--run') or '')))
    for f in twins.values():                        # an inspection server with no review twin is a page of its own
        pages.append(dict(id=f'p{f["port"]}', kind='review', code=f['code'], argv=f['argv'], env=f['env'],
                          legacy_port=f['port'], greenlit=True, inspection_only=True,
                          title=_opt(f['argv'], '--title') or os.path.basename(_opt(f['argv'], '--run') or '')))
    return pages


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='The two-port review server: seed a registry from the running servers, or serve one.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s1 = sub.add_parser('seed', help='print a registry of every review server running now (JSON)')
    s1.add_argument('--hub', help='the review hub site directory (static entry "hub")')
    s1.add_argument('--inspection-hub', help='the inspection hub site directory (static entry "hub_inspection")')
    s2 = sub.add_parser('serve', help='serve a registry on two ports')
    s2.add_argument('registry')
    s2.add_argument('--review-port', type=int, required=True)
    s2.add_argument('--inspection-port', type=int, required=True)
    s2.add_argument('--sock-dir', help=f'the backends\' sockets and logs (default {SOCK_DIR})')
    s2.add_argument('--publish-hold', action='store_true',
                    help='keep a run\'s built pages on their builds while <run>/' + HOLD_FILE + ' exists (an amendment option; '
                         'off = production: a page rebuilt meanwhile shows the marker\'s text)')
    s3 = sub.add_parser('backend', help='serve the pages of one code tree on a Unix socket (started by serve)')
    s3.add_argument('registry')
    s3.add_argument('--tree', required=True)
    s3.add_argument('--socket', required=True)
    s3.add_argument('--publish-hold', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'backend':
        serve_backend(a.registry, a.tree, a.socket, publish_hold=a.publish_hold)
        return 0
    if a.cmd == 'seed':
        pages = seed_from_proc()
        if a.hub:
            pages.append(dict(id='hub', kind='static', root=a.hub, greenlit=False))
        if a.inspection_hub:
            pages.append(dict(id='hub_inspection', kind='static', root=a.inspection_hub, greenlit=True))
        print(json.dumps({'pages': pages}, indent=1))
        return 0
    try:
        serve(a.registry, a.review_port, a.inspection_port, sock_dir=a.sock_dir, publish_hold=a.publish_hold)
    except ValueError as e:                  # a socket directory too long: refused before anything listens
        sys.exit(f'two-port server REFUSED: {e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
