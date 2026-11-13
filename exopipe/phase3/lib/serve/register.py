"""Registration of review pages on the two-port server - serve REGISTERS a page instead of starting a server of its own.

Production since 2026-10-22 (Gregg 2026-10-17 R-GREGG-20261017-TWO-PORT-SERVING; "Retire once all pages are transferred and
live"): every review page is an entry of the two-port server's registry (/lustre/gh/main/phase3/serve: registry.json, which the
server reads - run_two_port.sh, 8765 review / 8795 inspection - beside registered_pages.json, greenlit_pages.json and
pinned_pages.json), and the server rebuilds its pages at a registry change. The serves register through
/lustre/gh/main/phase3/serve/register_page.py 'register --entry': snapshot_serve.sh (SNAP_REGISTER=1, snapshot_register_page.py
deferred + --flush), the forced gen5 serve (snapshot_forced_gen5_serve_20261026.sh), the 10-minute publish (sublst_cumulative.sh
-> sublst_register_pages.py) and the deep re-registration (deep_reregister_after_publish_20261031.py; the deep pass's data steps
run with P3S3_NO_SERVER=1).

register(entries, d)   register_page.py 'register --entry' (the TWO_PORT_DIR sandbox path), byte for byte - tests/test_serve.py
  runs both on copies of one sandbox and compares the files:
  - registered_pages.json := its entries whose id is not given, then the entries given (a registration REPLACES the entry of the
    same id);
  - registry.json := the registry's entries whose id is not registered, then every registered entry, with the greenlit layer
    (greenlit_overlay.apply: Gregg's list <d>/greenlit_pages.json sets 'greenlit' and the inspection title; it is never written
    here);
  - the merged registry is checked with the server's own loader (two_port.load_registry, on a temporary file in <d>) BEFORE
    anything is written; then registered_pages.json and registry.json are written - BOTH, as production: the 15-minute re-seed
    (reseed_loop_v5.sh -> registry_merge.py) rebuilds registry.json and keeps the registered entries - each ATOMICALLY (a
    temporary file in <d>, mode 0664, os.replace) and each with its modification time strictly later than before (Lustre keeps
    whole seconds: a second write within the same second looked unchanged to the server, B6A p9024 2026-10-22).
  THE REGISTRY LOCK (registry_lock, register_page.py since 2026-11-13): an exclusive flock on <d>/serve.TWO_PORT_LOCK held from
  the reads of prepare() to the writes of write() - production's register_page.py and the re-seed (reseed_loop_v5.sh) hold the
  same lock, so two writers never interleave (two concurrent registrations each read the old files and the last write dropped
  the other's page: 1 of 24 survived without the lock, 24 of 24 with it). It waits serve.TWO_PORT_LOCK_WAIT_S, then refuses
  with nothing written. /lustre is mounted with cluster-wide flock.
  ONE REGISTRATION PER SERVE: every page a serve passes goes into one register() call - one registry change (every change makes
  each backend rebuild every page it built; two registrations seconds apart lost the second, B6A; snapshot_serve.sh and the
  forced serve register in one call, the deep re-registration its four pages in one).
  THE WARM-UP (register_page.py:76-87 -> warm_pages.py: the pages built before the change, requested once each after it, this
  call's pages first): serve requests ITS pages itself, one at a time in registration order, and waits on their builds
  (wait_built); then every INSPECTION page built before the registration is requested once (built_inspection / warm) - the
  review pages of other entries the server rebuilds by itself since WEB SPEED step 2 (d07fa79). Not carried:
  snapshot_register_page.py's 'unchanged is no write' (--flush): a serve that rewrote a run's files in place must reach the
  pages even when their entries are unchanged.

page_entry(plan, page, utc)   the entry: the server a per-page serve would have started, in snapshot_register_page.py's form - id
  p<port>, kind 'review', code (the page code's absolute path), argv (the server's arguments without --port), env (the P3_*
  variables of the page's server environment, sorted), legacy_port (its port: the --sibling and cross-tier link key, the filter
  cookie), greenlit False (the greenlit layer decides), inspection_legacy_port (its former inspection twin's port, else None),
  inspection_title None, title (its --title).

THE PUBLISH MARKER (set_hold / lift_hold): <run>/PUBLISH_IN_PROGRESS.txt (two_port.HOLD_FILE) - the marker production's deep
  publish writes: a page built while it exists shows its text as a banner (classify_transients.py:1242, exopipe_ops 4a39c6b).
  serve sets it on the run it rewrites in place (the deep views stage, once the pages merge is current) and lifts it - renamed
  to <file>.removed_<UTC> - right before the registration of the run's pages is written, whatever was refused, as
  deep_reregister_after_publish_20261031.py:79-81. A two-port server started with --publish-hold (serve.TWO_PORT_PUBLISH_HOLD,
  an amendment option, off = production) also keeps a marked run's pages on their builds meanwhile. Written through a temporary
  file and os.replace, renamed by its own directory entry: a parity mirror's link to the production run's file is never
  followed.

check_server(...)  before anything is written: the two-port server answers /_status, reports the build-state protocol
  (two_port.BUILD_STATE: a server older than this tree's two_port.py reports no build state, so no page could ever be seen built
  - REFUSED at once, G16: the server must be redeployed with this two_port.py first), reads this registry (G13) and runs with
  the publish-hold option serve.TWO_PORT_PUBLISH_HOLD says (G13). Its refusal prints the server's command line in
  run_two_port.sh's form (its environment, nice, a short socket directory).
wait_built(...)   after the registration, each page's BUILD STATE from the server's GET /_status (two_port.backend_status) - not
  a fixed number of tries (CUTOVER_READINESS: the snapshot identity check went from 60 to 240 tries of 5 s because a registered
  page builds behind earlier ones): the page is requested once (X-Two-Port-Fresh: 1), then it is DONE when the server reports a
  build of the registered entry (two_port.entry_md5) that started after the registration began (taken before the registry is
  written); REFUSED at once when such a build failed (its own error, G24), when serve.TWO_PORT_MAX_FAILED_REQUESTS requests in a
  row got no 200 (a page whose build kills its backend, G24), when the page is held by a publish hold (G16), when another
  registration replaced the entry (G13), when the server does not know the page (G16) or reports no build state (G16); it
  WAITS while builds progress on the server, and is refused (G16) only after serve.TWO_PORT_STALL_S in which no build started
  or finished there.

paths.PRODUCTION_WRITES False (the refactor window): nothing is written under a production root (the live registry, a production
run's marker) - G23. A parity serve registers in <out>/two_port/ and needs a two-port server of its own on the parity listeners
(serve.HUB_PORT / INSPECTION_HUB_PORT + --port-offset; server_command prints its command line); serve never starts or stops a
two-port server.
"""
import contextlib
import fcntl
import hashlib
import json
import os
import shlex
import tempfile
import threading
import time
import urllib.error
import urllib.request

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.serve import two_port as TP

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))      # never through an http_proxy


def _CFG():
    from exopipe.phase3.config import CFG
    return CFG


def _S():
    return _CFG().serve


# ---- where -----------------------------------------------------------------------------------------------------------
def registry_dir(plan):
    """The registry directory a plan registers in: production serve.TWO_PORT_DIR, a parity plan <out>/two_port."""
    S = _S()
    return os.path.join(plan.out, S.TWO_PORT_PARITY_DIR) if plan.parity else S.TWO_PORT_DIR


def files(d):
    """(registry.json, registered_pages.json, greenlit_pages.json) of the registry directory ``d``."""
    S = _S()
    return (os.path.join(d, S.TWO_PORT_REGISTRY), os.path.join(d, S.TWO_PORT_REGISTERED), os.path.join(d, S.TWO_PORT_GREENLIT))


def listeners(plan):
    """(review, inspection) listener ports of the two-port server the plan's pages are registered with."""
    S = _S()
    return S.HUB_PORT + plan.offset, S.INSPECTION_HUB_PORT + plan.offset


def sock_dir(review_port):
    """A short socket directory for a two-port server on ``review_port`` (run_two_port.sh: /run/user/<uid>/two_port_main): a
    Unix socket path holds at most two_port.SOCK_PATH_MAX bytes, so never under a long --out."""
    return os.path.join(os.environ.get('XDG_RUNTIME_DIR') or '/tmp', f'two_port_{int(review_port)}')


def server_argv(registry, review_port, inspection_port, sockets=None, python=None, publish_hold=None):
    """The argv of a two-port server on ``registry`` (run_two_port.sh:7-8), run from the repository root with server_env()."""
    from exopipe.phase3.lib.serve import scripts
    hold = _S().TWO_PORT_PUBLISH_HOLD if publish_hold is None else publish_hold
    return [python or scripts.interpreter('default'), '-m', 'exopipe.phase3.lib.serve.two_port', 'serve', registry,
            '--review-port', str(review_port), '--inspection-port', str(inspection_port), '--sock-dir',
            sockets or sock_dir(review_port)] + (['--publish-hold'] if hold else [])


def server_env(base=None):
    """run_two_port.sh:5's environment over ``base`` (default os.environ)."""
    return dict(base if base is not None else os.environ, **_S().TWO_PORT_SERVER_ENV)


def server_command(registry, review_port, inspection_port, sockets=None):
    """The shell command that starts a two-port server as run_two_port.sh does: from the repository root, its environment,
    nice, the site interpreter, a short socket directory."""
    S = _S()
    env = ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in S.TWO_PORT_SERVER_ENV.items())
    argv = ' '.join(shlex.quote(a) for a in server_argv(registry, review_port, inspection_port, sockets))
    return f'cd {shlex.quote(TP.REPO_ROOT)} && env {env} nice -n{int(S.TWO_PORT_NICE)} {argv}'


def guard(path, what):
    """G23: refuse a write under a production root while paths.PRODUCTION_WRITES is False (the refactor window)."""
    C = _CFG()
    root = C.production_root_of(path)
    if root is not None and not C.paths.PRODUCTION_WRITES:
        raise Refusal(f'G23 refactor window: {what} {path} lies under the production root {root} and paths.PRODUCTION_WRITES '
                      f'is False - production pages are registered by the production serves until serve is cut over')


# ---- register_page.py ------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def registry_lock(d, wait_s=None):
    """register_page.py registry_lock: hold the exclusive flock on <d>/serve.TWO_PORT_LOCK for one read-modify-write of the
    registry; Refusal after ``wait_s`` (default serve.TWO_PORT_LOCK_WAIT_S) seconds held by another writer. The descriptor is
    close-on-exec (Python's default), so no program started inside keeps the lock (a bare fork would share it)."""
    S = _S()
    path = os.path.join(d, S.TWO_PORT_LOCK)
    guard(path, 'the registry lock')
    wait_s = float(S.TWO_PORT_LOCK_WAIT_S if wait_s is None else wait_s)
    f = open(path, 'a')
    try:
        t0 = time.time()
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - t0 > wait_s:
                    raise Refusal(f'registration: the registry lock {path} is held by another writer for more than '
                                  f'{wait_s:g} s - nothing written')
                time.sleep(0.5)
        yield path
    finally:
        f.close()                                   # closing releases the lock


def _load(path):
    """register_page.py:34-38."""
    if not os.path.exists(path):
        return {'pages': []}
    with open(path) as f:
        return json.load(f)


def atomic_write(path, obj):
    """register_page.py:41-53 (_atomic_write): a temporary file beside it, mode 0664, os.replace, and the modification time
    left strictly later than before (whole seconds)."""
    old = os.stat(path).st_mtime if os.path.exists(path) else None
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix='.' + os.path.basename(path) + '.')
    with os.fdopen(fd, 'w') as f:
        json.dump(obj, f, indent=1)
    os.chmod(tmp, 0o664)
    os.replace(tmp, path)
    if old is not None and int(os.stat(path).st_mtime) <= int(old):
        t = int(old) + 1
        os.utime(path, (t, t))


def greenlit(path):
    """greenlit_overlay.load: {page id: {...}} of Gregg's list, {} without the file."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f).get('pages', {})


def greenlit_apply(reg, green):
    """greenlit_overlay.apply: a copy of ``reg`` with greenlit True on the listed ids and their inspection title (the
    list's own, else none - the inspection build keeps the review page's title, Gregg 2026-10-27)."""
    out = []
    for e in reg.get('pages', []):
        if e.get('id') in green:
            e = dict(e)
            e['greenlit'] = True
            it = (green[e['id']] or {}).get('inspection_title')
            if it:
                e['inspection_title'] = it
            else:
                e.pop('inspection_title', None)
        out.append(e)
    return dict(reg, pages=out)


def merge(base, registered, green):
    """register_page.py:56-59: the base registry's entries whose id is not registered, then the registered entries, with the
    greenlit layer."""
    ids = {e['id'] for e in registered['pages']}
    return greenlit_apply({'pages': [e for e in base['pages'] if e['id'] not in ids] + list(registered['pages'])}, green)


def _validate(reg, d):
    """register_page.py:62-69: the server's own loader on a temporary file in ``d``."""
    fd, tmp = tempfile.mkstemp(dir=d, prefix='.validate.', suffix='.json')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(reg, f)
        return TP.load_registry(tmp)
    finally:
        os.remove(tmp)


def state_of(path):
    """two_port.registry_state of a file, or None when it is absent."""
    try:
        return list(TP.registry_state(os.stat(path)))
    except FileNotFoundError:
        return None


def prepare(entries, d):
    """The registration of ``entries`` in ``d``, checked and not written: register_page.py:104 and 149-154 (the registered
    entries, the given ones replacing their ids) and _apply's merge and validation (:73-74). Refuses (nothing written) on a
    duplicate id, a malformed result or a production registry in the refactor window."""
    REG, REGD, GREEN = files(d)
    guard(REG, 'the two-port registry')
    new = [dict(e) for e in entries]
    ids = [e['id'] for e in new]
    if len(set(ids)) != len(ids):
        raise Refusal(f'registration: an id given twice in one registration ({sorted(i for i in ids if ids.count(i) > 1)})')
    if not os.path.isdir(d):
        raise Refusal(f'registration: no registry directory {d}')
    registered = _load(REGD)
    registered = {'pages': [e for e in registered['pages'] if e['id'] not in set(ids)] + new}
    merged = merge(_load(REG), registered, greenlit(GREEN))
    try:
        _validate(merged, d)                            # refuses before anything is written
    except (TP.RegistryError, ValueError) as e:
        raise Refusal(f'registration refused by the server\'s loader (nothing written): {e}')
    return {'dir': d, 'registry': REG, 'registered': REGD, 'given': new, 'ids': ids, 'registered_pages': registered,
            'merged': merged}


def write(prep):
    """Write a prepared registration (_apply, :75-89 without the warm-up): registered_pages.json, then registry.json - the
    server's one registry change. Returns {'registry', 'registered', 'given', 'entries' (as merged into the registry),
    'state' (of registry.json after the write), 'previous_state', 't' (epoch s, taken just before registry.json is written:
    the earliest a build of the registered entries can start)}."""
    REG, REGD = prep['registry'], prep['registered']
    guard(REG, 'the two-port registry')
    before = state_of(REG)
    atomic_write(REGD, prep['registered_pages'])
    t = time.time()                     # BEFORE the registry write: a build that starts as the server sees it counts
    atomic_write(REG, prep['merged'])
    m = {e['id']: e for e in prep['merged']['pages']}
    return {'registry': REG, 'registered': REGD, 'given': prep['given'], 'entries': [m[i] for i in prep['ids']],
            'state': state_of(REG), 'previous_state': before, 't': t}


def register(entries, d):
    """register_page.py 'register --entry' with ``entries``: one call, one registry change (prepare, then write), under the
    registry lock."""
    with registry_lock(d):
        return write(prepare(entries, d))


def page_entry(plan, page, utc, root=None):
    """The registry entry of ``page`` (snapshot_register_page.py:47-54 form; see the module docstring)."""
    S = _S()
    argv = plan.argv(page, utc, root)
    code = argv[1] if os.path.isabs(argv[1]) else os.path.join(plan.ops, argv[1])
    a = list(argv[2:])
    if '--port' in a:
        i = a.index('--port')
        del a[i:i + 2]
    pre = tuple(S.TWO_PORT_ENV_PREFIXES)
    env = {k: v for k, v in sorted(plan.server_env_extra().items()) if k.startswith(pre)}
    title = a[a.index('--title') + 1] if '--title' in a else ''
    return dict(id=f'p{page.port}', kind='review', code=code, argv=a, env=env, legacy_port=page.port, greenlit=False,
                inspection_legacy_port=page.twin, inspection_title=None, title=title)


def registered_entry(d, page_id):
    """The registry's current entry ``page_id`` (registry.json of ``d``), or None."""
    REG = files(d)[0]
    try:
        with open(REG) as f:
            return next((e for e in json.load(f).get('pages', []) if e.get('id') == page_id), None)
    except (OSError, ValueError):
        return None


# ---- the publish hold ------------------------------------------------------------------------------------------------
def hold_file(run_dir):
    return os.path.join(run_dir, TP.HOLD_FILE)


def set_hold(run_dir, text):
    """Set the publish hold of ``run_dir`` with ``text`` (the page banner). A hold that stands is kept with its own text.
    Returns (path, True when set now)."""
    p = hold_file(run_dir)
    if os.path.exists(p):
        return p, False
    guard(run_dir, 'the publish hold of')
    fd, tmp = tempfile.mkstemp(dir=run_dir, prefix='.' + TP.HOLD_FILE + '.')
    with os.fdopen(fd, 'w') as f:
        f.write(text.rstrip('\n') + '\n')
    os.chmod(tmp, 0o664)
    os.replace(tmp, p)                                  # replaces a (dangling) link itself, never its target
    return p, True


def lift_hold(run_dir, stamp=None):
    """Lift the publish hold of ``run_dir``: the file renamed to <file>.removed_<UTC> (deep_reregister_after_publish_20261031.py
    :79-81). Returns the new path, or None when the run holds none."""
    p = hold_file(run_dir)
    if not os.path.lexists(p):
        return None
    guard(run_dir, 'the publish hold of')
    new = p + '.removed_' + (stamp or time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    os.rename(p, new)
    return new


# ---- the server and its build state ----------------------------------------------------------------------------------
def http(port, path, headers=None, timeout=30.0):
    """(status, body bytes) of GET http://127.0.0.1:<port><path>; (None, error text) when nothing answers."""
    req = urllib.request.Request(f'http://127.0.0.1:{int(port)}{path}', headers=headers or {})
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:                     # noqa: BLE001 - not answering, refused, reset, timeout
        return None, f'{type(e).__name__}: {e}'.encode()


def server_status(port, timeout=None):
    """The two-port server's GET /_status on its review listener ``port`` (JSON), or None."""
    st, body = http(port, '/_status', timeout=timeout or _S().TWO_PORT_STATUS_TIMEOUT_S)
    if st != 200:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


def old_server(status):
    """None when ``status`` (a front /_status) and every backend that answered in it report this tree's build-state protocol
    (two_port.BUILD_STATE), else what lacks it."""
    if not status or status.get('build_state') != TP.BUILD_STATE:
        return f'the server\'s /_status reports no build state {TP.BUILD_STATE} (it has {(status or {}).get("build_state")})'
    for b in status.get('backends') or []:
        if 'error' not in b and b.get('build_state') != TP.BUILD_STATE:
            return f'its backend of {b.get("tree")} reports no build state {TP.BUILD_STATE}'
    return None


def check_server(ports, d):
    """Before anything is written (a registration, a judgement carry): the two-port server on the review listener of ``ports``
    (review, inspection) answers /_status (G16), reports this tree's build state (G16: an older server reports no build state -
    no page could ever be seen built - so it must be redeployed with this two_port.py first), reads the registry of ``d`` (G13)
    and runs with the publish-hold option serve.TWO_PORT_PUBLISH_HOLD says (G13). Returns its report."""
    S = _S()
    REG = files(d)[0]
    st = server_status(ports[0])
    if st is None:
        raise Refusal(f'G16 two-port server: nothing answers http://127.0.0.1:{ports[0]}/_status - pages are registered in '
                      f'{REG} and served by a two-port server reading it (serve never starts one; as run_two_port.sh: '
                      f'{server_command(REG, ports[0], ports[1])}) - nothing registered')
    why = old_server(st)
    if why:
        raise Refusal(f'G16 two-port server on {ports[0]}: {why} - a server older than this tree\'s lib/serve/two_port.py, '
                      f'whose pages serve would wait on forever: the two-port server must be redeployed with this two_port.py '
                      f'first - nothing registered')
    if os.path.realpath(st.get('registry') or '') != os.path.realpath(REG):
        raise Refusal(f'G13 two-port server on {ports[0]} reads {st.get("registry")}, not {REG} - nothing registered')
    if bool(st.get('publish_hold')) != bool(S.TWO_PORT_PUBLISH_HOLD):
        raise Refusal(f'G13 two-port server on {ports[0]} runs with the publish hold {"on" if st.get("publish_hold") else "off"}'
                      f', serve.TWO_PORT_PUBLISH_HOLD is {S.TWO_PORT_PUBLISH_HOLD} - nothing registered')
    return st


def built_inspection(status):
    """The page ids whose INSPECTION build a front /_status reports (warm_pages.py snapshot's ':inspection' keys)."""
    out = set()
    for b in (status or {}).get('backends') or []:
        out.update(k[:-len(':inspection')] for k in (b.get('built') or {}) if k.endswith(':inspection'))
    return sorted(out)


def warm(port, pid, timeout=None):
    """GET http://127.0.0.1:<port>/p/<pid>/ once (warm_pages.py warm: the inspection listener's port for an inspection page).
    Returns (status, seconds)."""
    t = time.time()
    st, _ = http(port, f'/p/{pid}/', timeout=timeout or _S().TWO_PORT_WARM_TIMEOUT_S)
    return st, round(time.time() - t, 1)


def page_state(status, key, want, t0):
    """(state, info) of page ``key`` (its review-listener build) in a front /_status report: 'built' (a build of the entry
    ``want`` - two_port.entry_md5 - that started at or after ``t0``), 'building' (under way or waiting for the build lock),
    'failed' (such a build failed: info['error']), 'held' (kept on its build by its run's publish hold), 'waiting' (none of
    these yet), 'no-server'."""
    if not status:
        return 'no-server', {}
    backends = status.get('backends') or []
    for b in backends:
        x = (b.get('built') or {}).get(key)
        if x and x.get('entry') == want and (x.get('build_start') or 0) >= t0:
            return 'built', dict(x, tree=b.get('tree'))
    for b in backends:
        if key in (b.get('building') or []):
            return 'building', {'tree': b.get('tree')}
    for b in backends:
        x = (b.get('errors') or {}).get(key)
        if x and x.get('entry') == want and (x.get('t') or 0) >= t0:
            return 'failed', dict(x, tree=b.get('tree'))
    for b in backends:
        x = (b.get('built') or {}).get(key)
        if x and x.get('held'):
            return 'held', dict(x, tree=b.get('tree'))
    return 'waiting', {}


def progress(status):
    """What changes when a build starts or ends anywhere on the server (the stall rule's signature): the builds under way,
    the builds finished (two_port's builds_done: a page's own completion can wait behind the next build), the pages built
    and their last failures."""
    if not status:
        return None
    sig = []
    for b in status.get('backends') or []:
        t = b.get('tree')
        sig.append(('builds_done', t, b.get('builds_done')))
        sig += [('built', t, k, x.get('build_start')) for k, x in sorted((b.get('built') or {}).items())]
        sig += [('building', t, k) for k in sorted(b.get('building') or [])]
        sig += [('error', t, k, x.get('t')) for k, x in sorted((b.get('errors') or {}).items())]
    return tuple(sig)


class _Request(threading.Thread):
    """The page's request (as the warm-up's): GET /p/<id>/ with the fresh header, in the background."""

    def __init__(self, port, pid, timeout):
        super().__init__(daemon=True)
        self.port, self.pid, self.timeout = port, pid, timeout
        self.status = self.body = None
        self.counted = False                  # its answer counted in wait_built's requests in a row

    def run(self):
        self.status, self.body = http(self.port, f'/p/{self.pid}/', {TP.FRESH_HEADER: '1'}, timeout=self.timeout)


def wait_built(port, entry, t0, d=None, poll=None, stall_s=None, log=None):
    """Wait on the build of the registered ``entry`` (merged form) on the server's review listener ``port`` (see the module
    docstring). Returns {'state': 'built', 'tree', 'built', 'build_s', 'prime_s', 'build_start', 'entry', 'waited_s',
    'states' (the states seen, in order), 'requests'}; raises Refusal."""
    S = _S()
    poll = S.TWO_PORT_POLL_S if poll is None else poll
    stall_s = S.TWO_PORT_STALL_S if stall_s is None else stall_s
    pid, want = entry['id'], TP.entry_md5(entry)
    max_failed = int(S.TWO_PORT_MAX_FAILED_REQUESTS)
    t_start = time.time()
    req = _Request(port, pid, stall_s)
    req.start()
    requests, failed, seen = 1, 0, []
    last_sig, last_change = object(), time.time()
    while True:
        st = server_status(port)
        now = time.time()
        why = old_server(st) if st is not None else None
        if why:
            raise Refusal(f'G16 page {pid}: the two-port server on {port}: {why} - it must be redeployed with this tree\'s '
                          f'two_port.py first')
        sig = progress(st)
        if sig is not None and sig != last_sig:
            last_sig, last_change = sig, now
        state, info = page_state(st, pid, want, t0)
        if not seen or seen[-1] != state:
            seen.append(state)
            if log:
                log(f'page {pid}: {state} after {now - t_start:.0f} s')
        if state == 'built':
            return dict(info, state='built', waited_s=round(now - t_start, 1), states=seen, requests=requests,
                        entry=want)
        if state == 'failed':
            raise Refusal(f'G24 page {pid}: its build on the two-port server failed: {info.get("error")}')
        if state == 'held':
            raise Refusal(f'G16 page {pid}: held on its build of {info.get("built")} by its run\'s publish hold '
                          f'({TP.hold_path(entry)}) - not rebuilt from the registered entry')
        if d is not None:
            cur = registered_entry(d, pid)
            if cur is None or TP.entry_md5(cur) != want:
                raise Refusal(f'G13 page {pid}: its registry entry was ' + ('removed' if cur is None else
                              'replaced by another registration') + ' while its build was awaited')
        if not req.is_alive() and not req.counted:
            req.counted = True
            failed = 0 if req.status == 200 else failed + 1
        if not req.is_alive() and state != 'building':
            if req.status == 404:
                raise Refusal(f'G16 page {pid}: the two-port server on {port} does not serve it (404: '
                              f'{(req.body or b"")[:120].decode("utf-8", "replace")})')
            if req.status == 500:
                raise Refusal(f'G24 page {pid}: {(req.body or b"")[:400].decode("utf-8", "replace")}')
            if req.status != 200:                   # no answer, a reset, a forward that timed out (502): ask again ...
                if failed >= max_failed:            # ... but not forever (a build that kills its backend restarts it)
                    raise Refusal(f'G24 page {pid}: {failed} requests in a row got no 200 (last {req.status}: '
                                  f'{(req.body or b"")[:200].decode("utf-8", "replace")}) - its build does not complete')
                req = _Request(port, pid, stall_s)
                req.start()
                requests += 1
        if now - last_change > stall_s:
            raise Refusal(f'G16 page {pid}: no build started or finished on the two-port server for {stall_s:.0f} s '
                          f'(states seen: {", ".join(seen)}; last request {req.status})')
        time.sleep(poll)


def registration_record(out, name, res, holds):
    """Write the record of one registration (class page_registration) under <out>/_records/serve/registrations/. Returns its
    path."""
    from exopipe.phase3.core.record import atomic_write_json, utcnow
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(res['t']))
    h = hashlib.md5(json.dumps([e['id'] for e in res['entries']]).encode()).hexdigest()[:8]
    path = os.path.join(out, '_records', 'serve', 'registrations', f'{name}_{stamp}_{h}.json')
    atomic_write_json(path, dict(res, unit=name, holds=holds, written=utcnow()))
    return path
