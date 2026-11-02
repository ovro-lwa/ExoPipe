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
    (reseed_loop_v4.sh -> registry_merge.py) rebuilds registry.json and keeps the registered entries - each ATOMICALLY (a
    temporary file in <d>, mode 0664, os.replace) and each with its modification time strictly later than before (Lustre keeps
    whole seconds: a second write within the same second looked unchanged to the server, B6A p9024 2026-10-22).
  ONE REGISTRATION PER SERVE: every page a serve passes goes into one register() call - one registry change (every change makes
  each backend rebuild every page it built; two registrations seconds apart lost the second, B6A; snapshot_serve.sh and the
  forced serve register in one call, the deep re-registration its four pages in one).
  Not carried: register_page.py's warm-up outside its sandbox (warm_pages.py): serve requests ITS pages itself, one at a time in
  registration order, and waits on their builds (wait_built) - the warm-up's own first keys; every other page the server rebuilds
  by itself since WEB SPEED step 2 (d07fa79), where the warm-up's request returns at once. Nor snapshot_register_page.py's
  'unchanged is no write' (--flush): a serve that rewrote a run's files in place must reach the pages even when their entries
  are unchanged.

page_entry(plan, page, utc)   the entry: the server a per-page serve would have started, in snapshot_register_page.py's form - id
  p<port>, kind 'review', code (the page code's absolute path), argv (the server's arguments without --port), env (the P3_*
  variables of the page's server environment, sorted), legacy_port (its port: the --sibling and cross-tier link key, the filter
  cookie), greenlit False (the greenlit layer decides), inspection_legacy_port (its former inspection twin's port, else None),
  inspection_title None, title (its --title).

THE PUBLISH HOLD (set_hold / lift_hold): <run>/PUBLISH_IN_PROGRESS.txt (two_port.HOLD_FILE) - the marker production's deep publish
  writes (a page built while it exists shows its text, exopipe_ops 4a39c6b). The two-port server keeps a held run's pages on
  their build at a registry change. serve sets it on the run it rewrites in place (deep views stage) and lifts it - renamed to
  <file>.removed_<UTC>, as deep_reregister_after_publish_20261031.py:79-81 - right before that run's one registration, unless a
  page of the run is refused (its build stays held: the refused page keeps its previous page). Written through a temporary file
  and os.replace, renamed by its own directory entry: a parity mirror's link to the production run's file is never followed.

wait_built(...)   after the registration, each page's BUILD STATE from the server's GET /_status (two_port.backend_status) - not
  a fixed number of tries (CUTOVER_READINESS: the snapshot identity check went from 60 to 240 tries of 5 s because a registered
  page builds behind earlier ones): the page is requested once (X-Two-Port-Fresh: 1), then it is DONE when the server reports a
  build of the registered entry (two_port.entry_md5) that started after the registration; REFUSED at once when such a build
  failed (its own error, G24), when the page is held by a publish hold (G16), when another registration replaced the entry
  (G13), when the server does not know the page (G16); it WAITS while builds progress on the server, and is refused (G16) only
  after serve.TWO_PORT_STALL_S in which no build started or finished there.

paths.PRODUCTION_WRITES False (the refactor window): nothing is written under a production root (the live registry, a production
run's hold) - G23. A parity serve registers in <out>/two_port/ and needs a two-port server of its own on the parity listeners
(serve.HUB_PORT / INSPECTION_HUB_PORT + --port-offset; server_argv prints its command line); serve never starts or stops a
two-port server.
"""
import hashlib
import json
import os
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


def server_argv(registry, review_port, inspection_port, sock_dir, python=None):
    """The command line of a two-port server on ``registry`` (run_two_port.sh:7-8), run from the repository root."""
    return [python or 'python3', '-m', 'exopipe.phase3.lib.serve.two_port', 'serve', registry, '--review-port',
            str(review_port), '--inspection-port', str(inspection_port), '--sock-dir', sock_dir]


def guard(path, what):
    """G23: refuse a write under a production root while paths.PRODUCTION_WRITES is False (the refactor window)."""
    C = _CFG()
    root = C.production_root_of(path)
    if root is not None and not C.paths.PRODUCTION_WRITES:
        raise Refusal(f'G23 refactor window: {what} {path} lies under the production root {root} and paths.PRODUCTION_WRITES '
                      f'is False - production pages are registered by the production serves until serve is cut over')


# ---- register_page.py ------------------------------------------------------------------------------------------------
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
    'state' (of registry.json after the write), 'previous_state', 't' (epoch s, after the write)}."""
    REG, REGD = prep['registry'], prep['registered']
    guard(REG, 'the two-port registry')
    before = state_of(REG)
    atomic_write(REGD, prep['registered_pages'])
    atomic_write(REG, prep['merged'])
    t = time.time()
    m = {e['id']: e for e in prep['merged']['pages']}
    return {'registry': REG, 'registered': REGD, 'given': prep['given'], 'entries': [m[i] for i in prep['ids']],
            'state': state_of(REG), 'previous_state': before, 't': t}


def register(entries, d):
    """register_page.py 'register --entry' with ``entries``: one call, one registry change (prepare, then write)."""
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


def check_server(ports, d):
    """Before a registration: the two-port server on the review listener of ``ports`` (review, inspection) answers /_status
    and reads the registry of ``d`` (G16 / G13). Returns its report."""
    REG = files(d)[0]
    st = server_status(ports[0])
    if st is None:
        raise Refusal(f'G16 two-port server: nothing answers http://127.0.0.1:{ports[0]}/_status - pages are registered in '
                      f'{REG} and served by a two-port server reading it (serve never starts one; from the repository root: '
                      f'{" ".join(server_argv(REG, ports[0], ports[1], os.path.join(d, "socks")))}) - nothing registered')
    if os.path.realpath(st.get('registry') or '') != os.path.realpath(REG):
        raise Refusal(f'G13 two-port server on {ports[0]} reads {st.get("registry")}, not {REG} - nothing registered')
    return st


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
    t_start = time.time()
    req = _Request(port, pid, stall_s)
    req.start()
    requests, seen = 1, []
    last_sig, last_change = object(), time.time()
    while True:
        st = server_status(port)
        now = time.time()
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
        if not req.is_alive() and state != 'building':
            if req.status == 404:
                raise Refusal(f'G16 page {pid}: the two-port server on {port} does not serve it (404: '
                              f'{(req.body or b"")[:120].decode("utf-8", "replace")})')
            if req.status == 500:
                raise Refusal(f'G24 page {pid}: {(req.body or b"")[:400].decode("utf-8", "replace")}')
            if req.status != 200:                   # no answer, a reset, a forward that timed out (502): ask again
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
