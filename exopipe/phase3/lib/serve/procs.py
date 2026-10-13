"""Processes and ports of the review servers: /proc scans, listeners, exact-pid stop, start, readiness, index links.

Ports (one implementation instead of eight, spec serve.md 11 / DUP-10):
- review server identity = comm 'python3' and an argv element ending 'classify_transients.py' (restart_inspection_twin.sh:26;
  restart_server_same_argv.sh:7-9); its port = the argv value after '--port' (restart_inspection_twin.sh:27).
- port owner = the pid holding a LISTEN socket on the port, read from /proc/net/tcp{,6} and /proc/<pid>/fd (the launchers'
  `ss -ltnp | grep :PORT` with the pid, sublst_serve.sh:118; snapshot_serve.sh:53).
- exact-pid stop: identity checked, SIGTERM, wait STOP_TIMEOUT_S polling STOP_POLL_S, never SIGKILL; still alive = refusal
  (restart_inspection_twin.sh:50-58; restart_server_same_argv.sh:11-12).
- start: setsid + nohup semantics (new session, SIGHUP ignored), stdin /dev/null, stdout+stderr to the page log, every
  other descriptor closed (the sub-LST lock fd 9 leak, INCIDENT-CUMULATIVE-BUILD-LOCK-DEADLOCK, cannot recur).
- readiness: GET http://127.0.0.1:<port>/ answers 200 (restart_inspection_twin.sh:63-71; snapshot_rechain_batch.sh:416).
- index source links: grep -oE '/source/[^<> ?#]+' per line, quotes stripped, URL-decoded, distinct
  (snapshot_rechain_batch.sh:419-420).
"""
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

from exopipe.phase3.core.record import Refusal

# the index link protocol of classify_transients.py's index page (snapshot_rechain_batch.sh:420); '\n' excluded because grep
# matches within one line
SOURCE_LINK_RE = re.compile(r'/source/[^<> ?#\n]+')


def _S():
    from exopipe.phase3.config import CFG
    return CFG.serve


# ---- /proc ---------------------------------------------------------------------------------------------------------
def proc_info(pid):
    """{'pid', 'comm', 'argv', 'cwd', 'stdout', 'state'} of a live process, or None."""
    d = f'/proc/{pid}'
    try:
        comm = open(f'{d}/comm').read().strip()
        argv = [x.decode('utf-8', 'replace') for x in open(f'{d}/cmdline', 'rb').read().split(b'\0') if x]
        state = open(f'{d}/stat').read().rsplit(')', 1)[1].split()[0]
    except (OSError, IndexError):
        return None
    info = {'pid': int(pid), 'comm': comm, 'argv': argv, 'state': state}
    for k, link in (('cwd', 'cwd'), ('stdout', 'fd/1')):
        try:
            info[k] = os.readlink(f'{d}/{link}')
        except OSError:
            info[k] = None
    return info


def environ(pid):
    """The environment of a process (/proc/<pid>/environ), as restart_inspection_twin.sh:61 reads it."""
    raw = open(f'/proc/{pid}/environ', 'rb').read().decode(errors='replace')
    return dict(x.split('=', 1) for x in raw.split('\0') if '=' in x)


def alive(pid):
    info = proc_info(pid)
    return info is not None and info['state'] != 'Z'


def argv_value(argv, flag):
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def is_server(info, program='classify'):
    """True when ``info`` is a server of ``program`` (serve.SERVER_IDENTITY: comm + an argv element ending so)."""
    comm, script = _S().SERVER_IDENTITY[program]
    return bool(info) and info['comm'] == comm and any(a.endswith(script) for a in info['argv'])


def is_review_server(info):
    return is_server(info, 'classify')


def program_of(info):
    """The serve program a process runs ('classify', 'jupiter', 'http'), or None."""
    for prog in _S().SERVER_IDENTITY:
        if is_server(info, prog):
            return prog
    return None


def http_port_of(argv):
    """The port of `python -m http.server PORT ...` (first bare integer after 'http.server')."""
    try:
        i = argv.index('http.server')
    except ValueError:
        return None
    for a in argv[i + 1:]:
        if a.isdigit():
            return a
    return None


def server_port(info):
    """The port a serve-owned server binds per its argv: '--port P' (classify), 'jupiter_server.py ROOT PORT', or the
    http.server port."""
    prog = program_of(info)
    if prog == 'classify':
        return argv_value(info['argv'], '--port')
    if prog == 'jupiter':
        i = next((k for k, a in enumerate(info['argv']) if a.endswith('jupiter_server.py')), None)
        return info['argv'][i + 2] if i is not None and len(info['argv']) > i + 2 else None
    if prog == 'http':
        return http_port_of(info['argv'])
    return None


def review_servers():
    """Every live review server: [{..proc_info.., 'port': str|None, 'inspect': bool, 'run': str|None}]."""
    out = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        info = proc_info(pid)
        if not is_review_server(info) or info['state'] == 'Z':
            continue
        info['port'] = argv_value(info['argv'], '--port')
        info['inspect'] = '--inspect' in info['argv']
        info['run'] = argv_value(info['argv'], '--run')
        out.append(info)
    return out


def _listen_inodes(port):
    inodes = set()
    for tbl in ('/proc/net/tcp', '/proc/net/tcp6'):
        try:
            lines = open(tbl).read().splitlines()[1:]
        except OSError:
            continue
        for l in lines:
            f = l.split()
            if len(f) < 10 or f[3] != '0A':
                continue
            if int(f[1].rsplit(':', 1)[1], 16) == int(port):
                inodes.add(f[9])
    return inodes


def listeners(port):
    """Sorted pids holding a LISTEN socket on TCP ``port`` (processes of other users are not visible)."""
    want = {f'socket:[{i}]' for i in _listen_inodes(port)}
    if not want:
        return []
    pids = set()
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            for fd in os.listdir(f'/proc/{pid}/fd'):
                try:
                    if os.readlink(f'/proc/{pid}/fd/{fd}') in want:
                        pids.add(int(pid))
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return sorted(pids)


def port_in_use(port):
    return bool(_listen_inodes(port))


# ---- stop / start --------------------------------------------------------------------------------------------------
def stop_exact(info, timeout=None, poll=None):
    """Stop ONE process by exact pid after re-checking its identity (same comm and argv as ``info``); wait until it is
    gone. Raises Refusal (G14) when the identity changed or the pid is still alive after the timeout. Never SIGKILL."""
    S = _S()
    timeout = S.STOP_TIMEOUT_S if timeout is None else timeout
    poll = S.STOP_POLL_S if poll is None else poll
    pid = info['pid']
    now = proc_info(pid)
    if now is None or now['state'] == 'Z':
        return 'already gone'
    if now['comm'] != info['comm'] or now['argv'] != info['argv']:
        raise Refusal(f'G14 exact-pid stop: pid {pid} is no longer the process that was identified ({now["comm"]} '
                      f'{" ".join(now["argv"][:3])}) - not stopped')
    os.kill(pid, signal.SIGTERM)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not alive(pid):
            return f'stopped pid {pid} in {time.time() - t0:.1f} s'
        time.sleep(poll)
    if alive(pid):
        raise Refusal(f'G14 exact-pid stop: pid {pid} still alive {timeout:.0f} s after SIGTERM - not restarted '
                      f'(never SIGKILL)')
    return f'stopped pid {pid}'


def _nohup():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)


def start(argv, cwd, env, log_path, append=False):
    """Start a long-lived server: new session (setsid), SIGHUP ignored (nohup), stdin /dev/null, stdout+stderr to
    ``log_path`` ('>' or '>>'), every other descriptor closed. Returns the Popen."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, 'a' if append else 'w')
    try:
        return subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, close_fds=True, preexec_fn=_nohup)
    finally:
        log.close()


# ---- HTTP ----------------------------------------------------------------------------------------------------------
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never through an http_proxy


def http_get(port, path='/', timeout=5.0):
    """(status, body text) of GET http://127.0.0.1:<port><path>; (None, error text) when nothing answers."""
    try:
        with _OPENER.open(f'http://127.0.0.1:{int(port)}{path}', timeout=timeout) as r:
            return r.status, r.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as e:
        return e.code, ''
    except Exception as e:                     # noqa: BLE001 - not answering, refused, timeout
        return None, f'{type(e).__name__}: {e}'


def last_log_line(path):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8192))
            lines = [l for l in f.read().decode('utf-8', 'replace').splitlines() if l.strip()]
        return lines[-1] if lines else '(empty log)'
    except OSError:
        return '(no log)'


def wait_ready(port, popen=None, timeout=300.0, poll=5.0, log_path=None):
    """(True, detail) once GET / answers 200; (False, detail) when the process exits first (G24) or after ``timeout``
    (G16)."""
    t0 = time.time()
    while True:
        if popen is not None and popen.poll() is not None:
            return False, f'server exited (rc {popen.returncode}) before answering: {last_log_line(log_path)}'
        st, _ = http_get(port, timeout=5.0)
        if st == 200:
            return True, f'HTTP 200 after {time.time() - t0:.0f} s'
        if time.time() - t0 >= timeout:
            return False, f'no HTTP 200 on port {port} within {timeout:.0f} s: {last_log_line(log_path)}'
        time.sleep(poll)


def index_source_ids(html):
    """Distinct URL-decoded '/source/<id>' links of an index page (snapshot_rechain_batch.sh:420)."""
    return {urllib.parse.unquote(m.strip().strip('"\''))[len('/source/'):] for m in SOURCE_LINK_RE.findall(html or '')
            if m.strip()}
