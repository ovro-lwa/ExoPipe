"""Replace, twin, restart and stop the servers of serve's ports; the served_page records.

replace(plan, page, ...)   G13 port ownership -> G14 exact-pid stop of every server of the page's program on the port and
                           of the port's listener -> start (argv/env/cwd/log from the plan) -> G16/G24 readiness -> record.
                           Ports: p3s3_serve_stokes.sh:63-79, p3s3_serve_multiples.sh:139-152, sublst_serve.sh:118-140,
                           sublst_cumulative.sh:129-141, snapshot_serve.sh:52-76, snapshot_forced_serve_20261007.sh:42-58.
start_twin(plan, page)     restart_inspection_twin.sh (whole script): exactly one judging server on the port (G15); its
                           argv cloned with --inspect after the script, the twin port and the read-only title; the old twin
                           stopped by exact pid; started with the JUDGING server's cwd and environment (M11); readiness.
restart_port / stop_port   restart_server_same_argv.sh, for every port serve owns (serve.PAGES, their twins; refactor-chat
                           decision: one owner of ports and the hub): identity, argv / cwd / stdout log AND environment from
                           /proc, exact-pid stop, relaunch appending to the same log (the environment was the caller's in
                           production; copying it from /proc is the M11 rule the twins already follow).
A served page's record <out>/_records/serve/pages/<tier>_<port>.json (class served_page) holds port, pid, argv, cwd, the
SERVER_ENV values, run, queue sha1, view digest, expected_sources and log; its content tests probe the live port.
"""
import csv
import fnmatch
import json
import os
import time

from exopipe.phase3.core.record import Refusal, atomic_write_json, utcnow
from exopipe.phase3.lib.serve import common, procs, scripts


def _S():
    from exopipe.phase3.config import CFG
    return CFG.serve


def _under(path, root):
    rp, rr = os.path.realpath(path or ''), os.path.realpath(root)
    return rp == rr or rp.startswith(rr.rstrip('/') + '/')


def _describe(info):
    return f'{(info or {}).get("comm")} {" ".join((info or {}).get("argv", [])[:4])}'


def to_stop(plan, port, program='classify', inspect=False):
    """The servers to stop before ``port`` is (re)bound: every server of ``program`` configured on the port (classify:
    judging or, with inspect=True, twins) and the port's listener. Refuses (G13 / G23) a listener that is not a server of
    ``program``, a server the tier may not replace (PORT_OWNERSHIP), and - in a parity plan - any server whose run is not
    under --out."""
    S = _S()
    infos = {}
    if program == 'classify':
        infos = {s['pid']: s for s in procs.review_servers() if s['port'] == str(port) and s['inspect'] == inspect}
    for pid in procs.listeners(port):
        if pid in infos:
            continue
        info = procs.proc_info(pid)
        if not procs.is_server(info, program):
            raise Refusal(f'G13 port ownership: port {port} is held by pid {pid} ({_describe(info)}), not a {program} server '
                          f'- not replaced')
        info['run'] = procs.argv_value(info['argv'], '--run')
        infos[pid] = info
    own = S.PORT_OWNERSHIP[plan.tier]
    for info in infos.values():
        run = info.get('run') or ''
        if plan.parity:
            if program == 'classify' and not _under(run, plan.out):
                raise Refusal(f'G23 parity guard: port {port} is held by a server of {run} (pid {info["pid"]}), not of this '
                              f'parity root {plan.out} - a parity run never replaces a production server')
            if program != 'classify' and port in owned_ports(0):
                raise Refusal(f'G23 parity guard: port {port} is a production port')
        elif own != 'port':
            ok = (os.path.dirname(os.path.normpath(run)) == os.path.normpath(_run_root(plan.tier))
                  and any(fnmatch.fnmatch(os.path.basename(os.path.normpath(run)), pat) for pat in own))
            if not ok:
                raise Refusal(f'G13 port ownership: port {port} is held by a server that is not a {plan.tier} run '
                              f'({run}, pid {info["pid"]}; allowed {own}) - REFUSING (sublst_serve.sh:121-123)')
    return list(infos.values())


def _run_root(tier):
    from exopipe.phase3.lib.serve.plan import run_root
    return run_root(tier)


def _stop_all(infos, notes, timeout=None):
    for info in infos:
        notes.append(f'{procs.stop_exact(info, timeout=timeout)} ({" ".join(info["argv"][1:4])} ...)')


def write_record(path, rec):
    atomic_write_json(path, rec)
    return path


def _record(plan, page, popen_pid, argv, cwd, env_extra, root, expected, log, twin_of=None):
    U = root or page.root
    q = page.at('queue', U)
    cat = page.at('catalog', U)
    return {'tier': plan.tier, 'family': page.family, 'program': page.program, 'page': page.key,
            'port': page.twin if twin_of else page.port, 'pid': popen_pid, 'argv': argv, 'cwd': cwd, 'env': env_extra,
            'run': U if page.program == 'classify' else None, 'run_name': page.run_name,
            'queue': q, 'queue_sha1': common.sha1_file(q) if q and os.path.exists(q) else None,
            'catalog': cat, 'view_digest': common.dir_digest(cat) if cat and os.path.isdir(cat) else None,
            'expected_sources': (expected or {}).get('n'), 'expected_kind': (expected or {}).get('kind'),
            'log': log, 'started': utcnow(), 'inspect': bool(twin_of), 'judging_port': twin_of, 'parity': plan.parity}


def start_page(plan, page, utc, root=None):
    """Start the server of ``page`` (nothing listens on its port). Returns (Popen, argv, cwd, log)."""
    U = root or page.root
    argv = plan.argv(page, utc, U)
    log = page.log_path(U)
    if page.program == 'classify' and plan.tier in ('deep', 'sublst'):
        os.makedirs(page.at('labels', U), exist_ok=True)          # mkdir -p $U/judgements_$ST (p3s3_serve_stokes.sh:58)
    p = procs.start(argv, page.cwd(), plan.server_env(), log)
    return p, argv, page.cwd(), log


def replace(plan, page, utc, root=None, expected=None, notes=None, stop_timeout=None):
    """Replace the server of ``page``; returns (record path, record). Raises Refusal on a gate."""
    S = _S()
    notes = notes if notes is not None else []
    stops = to_stop(plan, page.port, page.program)
    _stop_all(stops, notes, stop_timeout)
    if stops:
        time.sleep(S.START_SETTLE_S)
    if procs.port_in_use(page.port):
        raise Refusal(f'G13 port {page.port} is still held after the exact-pid stops - not started')
    p, argv, cwd, log = start_page(plan, page, utc, root)
    ok, detail = procs.wait_ready(page.port, p, S.READY_TIMEOUT_S[plan.tier], S.READY_POLL_S, log)
    if not ok:
        gate = 'G24 server start refused' if p.poll() is not None else 'G16 readiness'
        raise Refusal(f'{gate}: page {page.key} on {page.port}: {detail} (the previous server was stopped: '
                      f'{len(stops)} pid(s))')
    notes.append(f'page {page.key} served on {page.port} by pid {p.pid} ({detail})')
    rec = _record(plan, page, p.pid, argv, cwd, plan.server_env_extra(), root, expected, log)
    return write_record(page.record_path(), rec), rec


def greenlit_label(port):
    try:
        with open(_S().INSPECTION_GREENLIT) as f:
            return next((r['label'] for r in csv.DictReader(f) if r['port'] == str(port)), '')
    except OSError:
        return ''


def start_twin(plan, page, root=None, notes=None, expected=None):
    """Restart the read-only inspection twin of ``page`` from the judging server's own argv/cwd/environ."""
    S = _S()
    notes = notes if notes is not None else []
    jp, ip = page.port, page.twin
    judges = [s for s in procs.review_servers() if s['port'] == str(jp) and not s['inspect']]
    if len(judges) != 1:
        raise Refusal(f'G15 twin source: {len(judges)} judging servers on {jp} - inspection twin {ip} NOT restarted '
                      f'(restart_inspection_twin.sh:30-33)')
    j = judges[0]
    cwd = j['cwd']
    run = procs.argv_value(j['argv'], '--run')
    new = list(j['argv'])
    new[new.index('--port') + 1] = str(ip)
    si = next(i for i, x in enumerate(new) if x.endswith(S.SERVER_IDENTITY['classify'][1]))
    new.insert(si + 1, '--inspect')
    if '--title' in new:
        t = new.index('--title') + 1
        new[t] = new[t] + S.TWIN_TITLE_SUFFIX
    else:
        lab = greenlit_label(page.twin_cfg)
        new += ['--title', S.TWIN_TITLE_NO_TITLE.format(label_dash=f'{lab} - ' if lab else '',
                                                        run_name=os.path.basename(os.path.normpath(run)))]
    stops = to_stop(plan, ip, 'classify', inspect=True)
    _stop_all(stops, notes)
    if procs.port_in_use(ip):
        raise Refusal(f'G13 twin port {ip} is still held after the exact-pid stops - not started')
    env = procs.environ(j['pid'])
    log = os.path.join(run, S.TWIN_LOG.format(port=page.twin_cfg))
    p = procs.start(new, cwd, env, log, append=True)
    ok, detail = procs.wait_ready(ip, p, S.READY_TIMEOUT_S[plan.tier], S.READY_POLL_S, log)
    if not ok:
        raise Refusal(f'G16 readiness: inspection twin {ip} (twin of {jp}): {detail}')
    notes.append(f'inspection twin {ip} serving (twin of {jp}, {run}) pid {p.pid}')
    env_extra = {k: env.get(k) for k in plan.server_env_extra()}
    rec = _record(plan, page, p.pid, new, cwd, env_extra, root, expected, log, twin_of=jp)
    return write_record(page.record_path(port=ip), rec), rec


# ---- --restart / --stop ---------------------------------------------------------------------------------------------
def owned_ports(offset=0):
    """{port: (tier, page key, program)} of every port serve owns (pages and their twins), shifted by ``offset``."""
    S = _S()
    out = {}
    for (t, k), c in S.PAGES.items():
        out[c['port'] + offset] = (t, k, c['program'])
        if c.get('twin'):
            out[c['twin'] + offset] = (t, f'{k}-twin', 'classify')
    return out


def _servers_on(port, program):
    infos = {}
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        info = procs.proc_info(pid)
        if procs.is_server(info, program) and info['state'] != 'Z' and procs.server_port(info) == str(port):
            infos[info['pid']] = info
    for pid in procs.listeners(port):
        if pid not in infos:
            info = procs.proc_info(pid)
            if not procs.is_server(info, program):
                raise Refusal(f'G13: port {port} is held by pid {pid} ({_describe(info)}), not a {program} server - refused')
            infos[pid] = info
    return list(infos.values())


def _window_guard(info, out, program):
    """While paths.PRODUCTION_WRITES is False, --restart / --stop act only on servers of this run's parity root."""
    from exopipe.phase3.config import CFG
    if CFG.paths.PRODUCTION_WRITES:
        return
    if program == 'classify':
        run = procs.argv_value(info['argv'], '--run') or ''
        if _under(run, out):
            return
        what = f'serves {run}'
    else:
        port = int(procs.server_port(info) or 0)
        if port not in owned_ports(0):
            return
        what = f'holds production port {port}'
    raise Refusal(f'refactor window: pid {info["pid"]} {what} (not under --out {out}); production servers are restarted by '
                  f'the production launchers until serve is cut over (paths.PRODUCTION_WRITES)')


def _owned(port, offset):
    S = _S()
    own = owned_ports(offset)
    if port in S.OTHER_SERVED_PORTS:
        raise Refusal(f'port {port} is not serve\'s: {S.OTHER_SERVED_PORTS[port]}')
    if port not in own:
        raise Refusal(f'port {port} is not a port serve owns (serve.PAGES and their twins'
                      + (f', offset {offset}' if offset else '') + ')')
    return own[port]


def restart_port(port, out, notes, offset=0):
    """Restart every server on ``port`` (a port serve owns) with its exact argv, cwd, stdout log and environment.
    Returns [(record path, record)]."""
    S = _S()
    tier, key, program = _owned(port, offset)
    infos = _servers_on(port, program)
    if not infos:
        raise Refusal(f'--restart {port}: no {program} server on the port')
    recs = []
    for info in infos:
        _window_guard(info, out, program)
    for info in infos:
        env = procs.environ(info['pid'])
        log = info.get('stdout') if info.get('stdout', '').startswith('/') else os.devnull
        argv, cwd = list(info['argv']), info['cwd']
        notes.append(procs.stop_exact(info, timeout=10.0, poll=0.5))     # restart_server_same_argv.sh:11 (20 x 0.5 s)
        p = procs.start(argv, cwd, env, log, append=True)
        ok, detail = procs.wait_ready(port, p, max(S.READY_TIMEOUT_S.values()), S.READY_POLL_S, log)
        if not ok:
            raise Refusal(f'G16 readiness after restart of {port}: {detail}')
        notes.append(f'{port} restarted: pid {info["pid"]} -> {p.pid} ({detail})')
        q = procs.argv_value(argv, '--queue')
        cat = procs.argv_value(argv, '--catalog-dir')
        rec = {'tier': tier, 'family': None, 'program': program, 'page': key, 'port': int(port), 'pid': p.pid,
               'argv': argv, 'cwd': cwd, 'env': {k: env[k] for k in sorted(env) if k.startswith('P3_')},
               'run': procs.argv_value(argv, '--run'), 'queue': q,
               'queue_sha1': common.sha1_file(q) if q and os.path.exists(q) else None, 'catalog': cat,
               'view_digest': common.dir_digest(cat) if cat and os.path.isdir(cat) else None,
               'expected_sources': None, 'expected_kind': 'restart (same argv, files unchanged)', 'log': log,
               'started': utcnow(), 'inspect': '--inspect' in argv, 'judging_port': None, 'restarted_from_pid': info['pid']}
        path = os.path.join(out, '_records', 'serve', 'pages', f'restart_{port}.json')
        recs.append((write_record(path, rec), rec))
    return recs


def stop_port(port, out, notes, offset=0):
    """Stop every server on ``port`` (a port serve owns) by exact pid. Returns (record path, record), class stopped_page."""
    tier, key, program = _owned(port, offset)
    infos = _servers_on(port, program)
    for info in infos:
        _window_guard(info, out, program)
    stopped = []
    for info in infos:
        notes.append(procs.stop_exact(info))
        stopped.append({'pid': info['pid'], 'argv': info['argv']})
    rec = {'port': int(port), 'program': program, 'page': key, 'stopped': stopped, 'at': utcnow()}
    path = os.path.join(out, '_records', 'serve', 'pages', f'stopped_{port}.json')
    return write_record(path, rec), rec


# ---- the App probe (expected sources of a classify page) -------------------------------------------------------------
def app_probe(plan, page, root=None, unit=None):
    """Run lib/serve/app_probe.py through core.legacy.run with the page's environment: the server's own load_sources +
    read_queue + App on the files the server will read (its child code fingerprint - classify_transients.py,
    review_filters.py, ... - goes into the unit record). Returns {'n', 'kind', 'dropped', 'unexplained', ...}."""
    from exopipe.phase3.core import legacy
    S = _S()
    U = root or page.root
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app_probe.py')
    res = legacy.run(script, [plan.ops, scripts.clv15_root(), U, page.at('catalog', U), page.at('queue', U)],
                     python='default', env=plan.server_env_extra(), cwd=plan.ops, timeout=S.APP_PROBE_TIMEOUT_S,
                     unit=unit, refusal_codes=())
    if res.rc != 0:
        raise Refusal(f'App probe of page {page.key} failed (rc {res.rc}): {scripts.last_line(res.tail)}')
    line = next((l for l in reversed(res.tail.splitlines()) if l.startswith('{')), None)
    if line is None:
        raise Refusal(f'App probe of page {page.key} printed no result: {scripts.last_line(res.tail)}')
    out = json.loads(line)
    out['kind'] = 'App first-presentation identities'
    return out


def queue_length(path):
    """grep -v '^#' Q | grep -c . (snapshot_rechain_batch.sh:421)."""
    with open(path) as f:
        return sum(1 for l in f if not l.startswith('#') and l.rstrip('\n'))
