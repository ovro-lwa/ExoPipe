"""The two-port review server (Gregg 2026-10-17, INTERFACE_REQUIREMENTS section 1 "TWO PORTS FOR EVERYTHING"; design:
refactor_20261012/plan/serve/TWO_PORT_SERVER_DESIGN.md, "Implementation approach").

One process, two listeners: REVIEW (every page, with the classification interface; Gregg only) and INSPECTION (the same pages
without it, greenlit entries only, every write refused). A page is published by adding it to the registry (registry.json);
the server re-reads the registry when it changes - no new port, no restart.

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
"""
import collections
import contextlib
import email.message
import html
import importlib.util
import io
import json
import os
import re
import sys
import threading
import time
import urllib.parse
from collections.abc import MutableMapping

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


class Page:
    """A built page: its module instance, its App, its environment and the lock of its labels directory."""

    def __init__(self, entry, inspect, mod, env, write_lock):
        self.entry, self.inspect, self.mod, self.env, self.write_lock = entry, inspect, mod, env, write_lock
        self.id = entry['id']
        self.base = f'/p/{self.id}'
        self.legacy_port = _port_for(entry, inspect)
        self.built = time.time()
        self.last_used = self.built


def build_page(entry, inspect, base_env, write_lock):
    """Build one page exactly as its process starts today: module instance, the page's environment, main() with its argv."""
    env = page_env(base_env, entry.get('env'))
    code = entry['code']
    name = f'_twoport_{entry["id"].replace("-", "_").replace(".", "_")}_{"i" if inspect else "r"}'
    with _BUILD_LOCK, _ENV.use(env):
        spec = importlib.util.spec_from_file_location(name, code)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        cdir = os.path.dirname(os.path.abspath(code))
        added = cdir not in sys.path
        if added:
            sys.path.insert(0, cdir)
        saved = sys.argv
        try:
            spec.loader.exec_module(mod)
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
        finally:
            sys.argv = saved
            if added:
                try:
                    sys.path.remove(cdir)
                except ValueError:
                    pass
    if getattr(mod.Handler, 'app', None) is None:
        raise RuntimeError(f'{entry["id"]}: built, but its Handler has no app')
    return Page(entry, inspect, mod, env, write_lock)


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
# the page store
# ---------------------------------------------------------------------------------------------------------------------------


class Store:
    """Pages built on first request and kept while used. A page requested within keep_s is never evicted; past max_pages the
    least recently used page goes. A page whose registry entry changed is rebuilt on its next request."""

    def __init__(self, registry_path, base_env=None, max_pages=80, keep_s=1800):
        install_env()
        self.registry_path = registry_path
        self.base_env = dict(base_env if base_env is not None else _ENV._base)
        self.max_pages, self.keep_s = max_pages, keep_s
        self.lock = threading.RLock()
        self.pages = {}                       # (page_id, inspect) -> Page
        self.building = {}                    # (page_id, inspect) -> threading.Event
        self.write_locks = collections.defaultdict(threading.Lock)
        self.errors = {}                      # (page_id, inspect) -> text of the last build failure
        self.reg, self.reg_mtime = collections.OrderedDict(), None
        self.reload()

    def reload(self):
        """Re-read the registry when it changed. Every built page is dropped and rebuilt on its next request: a page builds its
        cross-tier and sibling links from the other runs' catalogues when it starts, so a run published or replaced must reach
        the pages already built (the 2026-10-17 parity found live pages days old without the 10-minute LSTs published since)."""
        m = os.path.getmtime(self.registry_path)
        if m == self.reg_mtime:
            return
        reg = load_registry(self.registry_path)
        with self.lock:
            for key, pg in list(self.pages.items()):
                del self.pages[key]
                sys.modules.pop(pg.mod.__name__, None)
            self.reg, self.reg_mtime = reg, m

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

    def get(self, pid, inspect):
        key = (pid, bool(inspect))
        e = self.reg.get(pid)
        if e is None or e['kind'] != 'review':
            return None
        while True:
            with self.lock:
                pg = self.pages.get(key)
                if pg is not None:
                    pg.last_used = time.time()
                    return pg
                ev = self.building.get(key)
                if ev is None:
                    ev = self.building[key] = threading.Event()
                    mine = True
                else:
                    mine = False
            if not mine:
                ev.wait()
                if key in self.errors and key not in self.pages:
                    raise RuntimeError(self.errors[key])
                continue
            try:
                pg = build_page(e, inspect, self.base_env, self.write_locks[self.labels_dir(e)])
                with self.lock:
                    self.pages[key] = pg
                    self.errors.pop(key, None)
                    self._evict()
                return pg
            except Exception as exc:                           # one page's failure never takes down the process
                with self.lock:
                    self.errors[key] = f'{pid}: {type(exc).__name__}: {exc}'
                raise
            finally:
                with self.lock:
                    self.building.pop(key, None)
                ev.set()

    def _evict(self):
        now = time.time()
        if len(self.pages) <= self.max_pages:
            return
        for key, pg in sorted(self.pages.items(), key=lambda kv: kv[1].last_used):
            if len(self.pages) <= self.max_pages:
                break
            if now - pg.last_used > self.keep_s:
                del self.pages[key]
                sys.modules.pop(pg.mod.__name__, None)


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
        return 404, [('Content-Type', 'text/plain')], b'not found'
    kind, pid, rest = m.group(1), m.group(2), m.group(3) or '/'
    e = store.entry(pid)
    if e is None or (inspect and not e.get('greenlit')):
        return 404, [('Content-Type', 'text/plain')], b'unknown page'
    if inspect and method != 'GET':
        return 403, [('Content-Type', 'text/plain; charset=utf-8')], b'inspection mode: this view is read-only and records nothing'
    query = path[len(f'/{kind}/{pid}') + len(rest):] if '?' in path else ''
    if kind == 's':
        return static_file(e, rest, store.port_map())
    try:
        pg = store.get(pid, inspect)
    except Exception as exc:                                    # the page's error page, the process carries on
        return 500, [('Content-Type', 'text/plain; charset=utf-8')], f'page {pid} could not be built: {exc}'.encode()
    sub = rest + (('?' + path.split('?', 1)[1]) if '?' in path else '')
    lock = pg.write_lock if method == 'POST' else contextlib.nullcontext()
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
    ctype = dict((k.lower(), v) for k, v in hdrs).get('content-type', '')
    if ctype.startswith('text/html'):
        payload = rewrite_body(payload.decode('utf-8'), pg.base, pm, legacy_port=pg.legacy_port).encode('utf-8')
    return status, out, payload


def static_file(entry, rest, port_map=None):
    root = os.path.realpath(entry['root'])
    rel = urllib.parse.unquote(rest.split('?', 1)[0]).lstrip('/') or 'index.html'
    full = os.path.realpath(os.path.join(root, rel))
    if os.path.isdir(full):
        full = os.path.join(full, 'index.html')
    if not (full == root or full.startswith(root + os.sep)) or not os.path.isfile(full):
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


def make_handler(store, inspect):
    from http.server import BaseHTTPRequestHandler

    class Router(BaseHTTPRequestHandler):
        def _go(self, method):
            n = int(self.headers.get('Content-Length') or 0)
            body = self.rfile.read(n) if n else b''
            hdrs = {k: v for k, v in self.headers.items()}
            status, out, payload = serve_request(store, inspect, method, self.path, hdrs, body)
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

        def log_message(self, fmt, *args):
            pass
    return Router


def serve(registry_path, review_port, inspection_port, host='127.0.0.1'):
    """Run both listeners until interrupted."""
    from http.server import ThreadingHTTPServer
    store = Store(registry_path)
    srv = [ThreadingHTTPServer((host, review_port), make_handler(store, False)),
           ThreadingHTTPServer((host, inspection_port), make_handler(store, True))]
    ts = [threading.Thread(target=s.serve_forever, daemon=True) for s in srv]
    for t in ts:
        t.start()
    print(f'two-port server: review http://{host}:{review_port}/  inspection http://{host}:{inspection_port}/  '
          f'({len(store.reg)} registry entries)', flush=True)
    try:
        for t in ts:
            t.join()
    except KeyboardInterrupt:
        for s in srv:
            s.shutdown()


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
    a = ap.parse_args(argv)
    if a.cmd == 'seed':
        pages = seed_from_proc()
        if a.hub:
            pages.append(dict(id='hub', kind='static', root=a.hub, greenlit=False))
        if a.inspection_hub:
            pages.append(dict(id='hub_inspection', kind='static', root=a.inspection_hub, greenlit=True))
        print(json.dumps({'pages': pages}, indent=1))
        return 0
    serve(a.registry, a.review_port, a.inspection_port)
    return 0


if __name__ == '__main__':
    sys.exit(main())
