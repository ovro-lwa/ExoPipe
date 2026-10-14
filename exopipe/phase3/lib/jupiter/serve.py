"""`exopipe3 jupiter serve` - the Jupiter servers: 8800 (Phase-1 page, exopipe_ops/jupiter_server.py ROOT PORT, in place) and
8801 (the V2 static site: `python -m http.server PORT --bind 127.0.0.1 --directory <out>/v2/site`, the argv of the live server
read on lwacalim07 2026-10-13). Neither had a launcher (contracts D21); the step owns both ports in the window (jupiter_serve.PORTS;
crosscheck D15: serve is told, the hub links them).

Default: start the server when its port is free; when this step's own server already listens, leave it (no restart) and
re-run the served content test. --restart: exact-pid stop then start; --stop: exact-pid stop. Gate J40: a process on the port
that is not this step's server for this root (argv and root checked from /proc) refuses; the old pid still alive
STOP_WAIT_S after SIGTERM refuses (never SIGKILL). A non-production --out serves only on parity ports (--port-offset >=
PARITY_PORT_OFFSET_MIN). Units serve.<port>; output the serve record <out>/_served/jupiter_<port>.json [jupiter_served_page].
"""
import json
import os
import signal
import subprocess
import sys
import time

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.jupiter import common as K
from exopipe.phase3.lib.jupiter import mirrors


def _C():
    from exopipe.phase3.config import CFG
    return CFG


# ---- /proc ----------------------------------------------------------------------------------------------------------
def _listen_inodes(port):
    inodes = set()
    for fn in ('/proc/net/tcp', '/proc/net/tcp6'):
        try:
            lines = open(fn).read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            p = line.split()
            if len(p) > 9 and p[3] == '0A' and int(p[1].rsplit(':', 1)[1], 16) == port:
                inodes.add(p[9])
    return inodes


def listeners(port):
    """[{'pid', 'argv', 'cwd'}] of the processes holding a LISTEN socket on the port."""
    ino = _listen_inodes(port)
    out = []
    if not ino:
        return out
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        try:
            fds = os.listdir(f'/proc/{pid}/fd')
        except OSError:
            continue
        for fd in fds:
            try:
                link = os.readlink(f'/proc/{pid}/fd/{fd}')
            except OSError:
                continue
            if link.startswith('socket:[') and link[8:-1] in ino:
                try:
                    argv = [x.decode('utf-8', 'replace') for x in open(f'/proc/{pid}/cmdline', 'rb').read().split(b'\0') if x]
                    cwd = os.readlink(f'/proc/{pid}/cwd')
                except OSError:
                    argv, cwd = [], None
                out.append({'pid': int(pid), 'argv': argv, 'cwd': cwd})
                break
    return out


def _abs(arg, cwd):
    return os.path.realpath(arg if os.path.isabs(arg) or not cwd else os.path.join(cwd, arg))


def is_ours(info, which, root):
    """This step's server for this root: 8800 = an argv element naming jupiter_server.py followed by the products root;
    8801 = `-m http.server` with --directory the site root."""
    a, cwd = info['argv'], info['cwd']
    if which == 'phase1':
        for i, x in enumerate(a[:-1]):
            if os.path.basename(x) == _C().jupiter.SERVER_MODULE:
                return _abs(a[i + 1], cwd) == os.path.realpath(root)
        return False
    if '-m' in a and 'http.server' in a and '--directory' in a:
        return _abs(a[a.index('--directory') + 1], cwd) == os.path.realpath(root)
    return False


def _alive(pid):
    """False once the process has exited: gone from /proc, or a zombie (state Z) - reaped here when it is our own child
    (a server this process started stays a zombie until its parent waits; one started by an earlier process is reaped
    by init)."""
    try:
        with open(f'/proc/{pid}/stat') as f:
            state = f.read().rsplit(')', 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    if state == 'Z':
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        return False
    return True


def stop_exact(info):
    S = _C().jupiter_serve
    try:
        os.kill(info['pid'], signal.SIGTERM)
    except ProcessLookupError:
        return
    t0 = time.time()
    while time.time() - t0 < S.STOP_WAIT_S[0]:
        if not _alive(info['pid']):
            return
        time.sleep(S.STOP_WAIT_S[1])
    raise Refusal(f'J40 pid {info["pid"]} still alive {S.STOP_WAIT_S[0]:g} s after SIGTERM (never SIGKILL): {" ".join(info["argv"])}')


def http_ok(port, timeout=5.0):
    import urllib.request
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=timeout) as r:
            return r.status == 200
    except Exception:      # noqa: BLE001
        return False


# ---- the action -----------------------------------------------------------------------------------------------------
def plan(args, roots):
    S, J = _C().jupiter_serve, _C().jupiter
    off = args.port_offset or 0
    if off and off < S.PARITY_PORT_OFFSET_MIN:
        raise Refusal(f'--port-offset {off}: 0 or >= {S.PARITY_PORT_OFFSET_MIN} (parity servers never take a live port)')
    if not roots.production and off == 0:
        raise Refusal(f'a non-production --out serves only on parity ports: pass --port-offset >= {S.PARITY_PORT_OFFSET_MIN}')
    from exopipe.phase3.core import legacy
    py = legacy.interpreter(K.interpreter_key('serve'))
    srv = {}
    site = os.path.join(roots.out, S.V2_SITE_DIR)
    srv['phase1'] = dict(port=S.PORTS['phase1'] + off, root=roots.products,
                         argv=[py, mirrors.module_path('server'), roots.products, str(S.PORTS['phase1'] + off)],
                         cwd=S.PHASE1_CWD if roots.production else roots.out, log=os.path.join(roots.out, S.PHASE1_LOG))
    srv['v2'] = dict(port=S.PORTS['v2'] + off, root=site,
                     argv=[py, '-m', 'http.server', str(S.PORTS['v2'] + off), '--bind', S.BIND, '--directory', site],
                     cwd=site, log=os.path.join(roots.out, S.V2_SERVER_LOG))
    return srv


def start(spec, roots):
    """setsid + nohup semantics: new session, stdin /dev/null, stdout+stderr to the log, other descriptors closed."""
    clear = tuple(getattr(_C().runner, 'CHILD_ENV_CLEAR_PREFIXES', ('P3_',)))    # as core.legacy.run: no inherited switch
    env = {k: v for k, v in os.environ.items() if not k.startswith(clear)}
    env.update(_C().jupiter.ENV)
    env.pop('PYTHONPATH', None)
    if roots.guard:
        from exopipe.phase3.core import guard
        env['PYTHONPATH'] = guard.SITECUSTOMIZE_DIR
        env[guard.ENV_ROOTS] = os.pathsep.join(roots.guard)
    os.makedirs(os.path.dirname(spec['log']), exist_ok=True)
    lf = open(spec['log'], 'a')
    lf.write(f'\n===== {K.utc()} {" ".join(spec["argv"])}\n')
    lf.flush()
    p = subprocess.Popen(spec['argv'], cwd=spec['cwd'], env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                         close_fds=True, start_new_session=True, preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))
    lf.close()
    return p


def run(args, run_, roots, stages):
    S = _C().jupiter_serve
    mirrors.check_literals(mirrors.ACTION_MODULES['serve'])
    srv = plan(args, roots)
    if args.restart and args.stop:
        raise Refusal('give --restart or --stop, not both')
    if run_.dry_run:
        print(f'jupiter serve: {",".join(stages)}; {roots.describe()}')
        for which in stages:
            s = srv[which]
            ls = listeners(s['port'])
            own = [i for i in ls if is_ours(i, which, s['root'])]
            other = [i for i in ls if i not in own]
            act = ('stop' if args.stop else 'restart' if args.restart else 'start' if not ls else
                   'keep (this step\'s server already listens)' if own and not other else 'REFUSE (J40)')
            if other:
                act = 'REFUSE (J40: the port is held by another process)'
            if which == 'phase1' and S.PHASE1_RETIRED:
                act = 'REFUSE (8800 retired)'
            print(f'  {which}: port {s["port"]}; root {s["root"]}; listening now: '
                  + ('; '.join(f'pid {i["pid"]} {" ".join(i["argv"])} (cwd {i["cwd"]})' for i in ls) or 'nobody'))
            print(f'    action: {act}; argv {" ".join(s["argv"])}; cwd {s["cwd"]}; log {s["log"]}')
            print(f'    output {os.path.join(roots.out, "_served", "jupiter_%d.json" % s["port"])} [jupiter_served_page]')
        return 0
    for which in stages:
        s = srv[which]
        with run_.unit(f'serve.{s["port"]}', action='serve', which=which, port=s['port']) as u:
            if which == 'phase1' and S.PHASE1_RETIRED and not args.stop:
                u.refuse('8800 is retired (jupiter_serve.PHASE1_RETIRED, decision 8)')
            if not os.path.isdir(s['root']):
                u.refuse(f'nothing to serve: {s["root"]} does not exist')
            ls = listeners(s['port'])
            own = [i for i in ls if is_ours(i, which, s['root'])]
            other = [i for i in ls if i not in own]
            if other:
                u.refuse(f'J40 port {s["port"]} is held by another process: ' +
                         '; '.join(f'pid {i["pid"]} {" ".join(i["argv"])}' for i in other))
            if (args.restart or args.stop) and own:
                for i in own:
                    stop_exact(i)
                    u.note(f'stopped pid {i["pid"]} ({" ".join(i["argv"])})')
                own = []
            rec = dict(which=which, port=s['port'], root=s['root'], argv=s['argv'], cwd=s['cwd'], log=s['log'], utc=K.utc())
            if args.stop:
                rec.update(which='stopped', pid=None)
            elif own:
                rec.update(pid=own[0]['pid'], started='earlier (kept)')
            else:
                p = start(s, roots)
                ok = False
                for _ in range(S.READY_POLLS[0]):
                    if p.poll() is not None:
                        break
                    if http_ok(s['port']):
                        ok = True
                        break
                    time.sleep(S.READY_POLLS[1])
                if not ok:
                    raise RuntimeError(f'{which} server on {s["port"]} not ready (exit {p.poll()}; log {s["log"]})')
                rec.update(pid=p.pid, started=K.utc())
            out = os.path.join(roots.out, '_served', f'jupiter_{s["port"]}.json')
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out + '.part', 'w') as f:
                json.dump(rec, f, indent=1)
            os.replace(out + '.part', out)
            u.add_output(out, 'jupiter_served_page')
        print(f'jupiter serve {which} ({s["port"]}): {u.status}' + (f' ({u.reason})' if u.reason else ''), file=sys.stderr)
    return None
