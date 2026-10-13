"""The multi-node runner behind `exopipe3 run` - one implementation for every fleet, queue and driver.

A JOB runs one step over a list of ITEMS (cells, LSTs, frame batches) on a set of nodes. Its record lives in
``CFG.runner.JOBS_ROOT/<job>/`` (a handful of files per item, on Lustre):

    job.json              id, step argv template, nodes and lanes, --after, --retry, state, created, launch host
    items.txt             the items in run order (rewritten by `run reorder`)
    items/<item>.json     per item: status (queued|running|done|refused|no-input|error|failed), host, pid, rc,
                          attempts, started, finished, log path
    claims/<item>/        atomic claim (mkdir) by the worker that runs the item
    logs/<item>.log       the step's stdout/stderr for that item
    HOLD                  present = workers take no new items (running items finish)
    DRAIN.<host>          present = workers on <host> exit after their current item

Workers are started with ssh on each node (``nohup setsid``), so a job survives the session that launched it. Each
item runs as ``python -m exopipe.phase3 <step> ... `` with the item substituted for ``{item}`` (or appended as
``--cells <item>``), pinned to the node's recorder-free cores (``--cores auto`` read at every launch). An item's status
is the step's exit code: 0 done, 2 refused, 3 no-input, other error; an error is retried up to --retry times.
``--after JOB`` waits until that job's record is finished (every item in a terminal state) - never on log text,
process names or pids.
"""
import json
import os
import shlex
import socket
import subprocess
import sys
import time

TERMINAL = ('done', 'refused', 'no-input', 'failed')
RC_STATUS = {0: 'done', 2: 'refused', 3: 'no-input'}


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def jobs_root():
    return _cfg().runner.JOBS_ROOT


def job_dir(job):
    return os.path.join(jobs_root(), job)


def _wjson(path, obj):
    from exopipe.phase3.core.record import atomic_write_json
    atomic_write_json(path, obj)


def _rjson(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _safe(item):
    return item.replace('/', '_')


def load_job(job):
    j = _rjson(os.path.join(job_dir(job), 'job.json'))
    if j is None:
        raise SystemExit(f'no such job {job!r} under {jobs_root()}')
    return j


def items(job):
    with open(os.path.join(job_dir(job), 'items.txt')) as f:
        return [l.strip() for l in f if l.strip()]


def item_state(job, item):
    return _rjson(os.path.join(job_dir(job), 'items', f'{_safe(item)}.json'), {'status': 'queued', 'attempts': 0})


def job_items(job, status):
    """Items of JOB with the given status (used by the item grammar 'JOB:STATUS')."""
    return [it for it in items(job) if item_state(job, it).get('status') == status]


def job_state(job):
    sts = [item_state(job, it).get('status', 'queued') for it in items(job)]
    by = {}
    for s in sts:
        by[s] = by.get(s, 0) + 1
    if os.path.exists(os.path.join(job_dir(job), 'HOLD')):
        state = 'held'
    elif sts and all(s in TERMINAL for s in sts):
        state = 'finished'
    elif 'running' in sts:
        state = 'running'
    else:
        state = 'queued'
    return state, by


def parse_nodes(spec):
    """'lwacalim09:8,lwacalim07:2' or a site pool name ('deep') -> [(host, lanes)]."""
    cfg = _cfg()
    out = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if tok in cfg.site.NODE_POOLS:
            for h in cfg.site.NODE_POOLS[tok]:
                out.append(tuple(h) if isinstance(h, (list, tuple)) else (h, 1))
            continue
        host, _, lanes = tok.partition(':')
        out.append((host, int(lanes or 1)))
    if not out:
        raise SystemExit('give --nodes host:lanes[,host:lanes] or a site pool name')
    return out


def submit(name, step_argv, item_list, nodes, after=None, retry=0, gpu=False, python=None, start=True):
    """Create a job record and (unless start=False) launch its supervisor in the background."""
    cfg = _cfg()
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    job = f'{stamp}_{name}' if name else f'{stamp}_{step_argv[0]}'
    d = job_dir(job)
    os.makedirs(os.path.join(d, 'items'))
    os.makedirs(os.path.join(d, 'claims'))
    os.makedirs(os.path.join(d, 'logs'))
    with open(os.path.join(d, 'items.txt'), 'w') as f:
        f.write(''.join(f'{it}\n' for it in item_list))
    if gpu:
        bad = [h for h, _ in nodes if h not in cfg.site.GPU_HOSTS]
        if bad:
            raise SystemExit(f'refused: --gpu but {bad} are not GPU hosts in the site file (GPU_HOSTS)')
    j = {'id': job, 'step_argv': list(step_argv), 'nodes': [list(n) for n in nodes], 'after': after,
         'retry': int(retry), 'gpu': bool(gpu), 'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'launch_host': socket.gethostname(), 'python': python or sys.executable,
         'pkg_root': os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
         'site_file': os.environ.get('EXOPIPE3_SITE', ''), 'n_items': len(item_list), 'jobs_root': jobs_root()}
    _wjson(os.path.join(d, 'job.json'), j)
    if start:
        log = os.path.join(d, 'supervisor.log')
        cmd = [j['python'], '-m', 'exopipe.phase3', 'run', 'supervise', job, '--cores', 'none'] + _root_set(j)
        subprocess.Popen(cmd, stdout=open(log, 'a'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         cwd=j['pkg_root'], start_new_session=True, env=_child_env(j))
    return job


def _child_env(j):
    env = dict(os.environ)
    env['PYTHONPATH'] = j['pkg_root'] + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    if j.get('site_file'):
        env['EXOPIPE3_SITE'] = j['site_file']
    env.update(_cfg().runner.ENV_DEFAULTS)
    return env


def supervise(job):
    """Wait for --after, then start one worker per lane on each node; return when the job is finished."""
    cfg = _cfg()
    j = load_job(job)
    if j.get('after'):
        print(f'{_now()} waiting for job {j["after"]} to finish', flush=True)
        while job_state(j['after'])[0] != 'finished':
            time.sleep(cfg.runner.POLL_S)
        print(f'{_now()} job {j["after"]} finished: {job_state(j["after"])[1]}', flush=True)
    me = socket.gethostname()
    for host, lanes in j['nodes']:
        for lane in range(int(lanes)):
            wl = os.path.join(job_dir(job), 'logs', f'_worker_{host}_{lane}.log')
            inner = (f'cd {shlex.quote(j["pkg_root"])} && '
                     + ' '.join(f'{k}={shlex.quote(v)}' for k, v in sorted(_worker_env(j).items()))
                     + f' nohup setsid {shlex.quote(j["python"])} -m exopipe.phase3 run worker {job} --lane {lane}'
                     + f' --cores none ' + ' '.join(map(shlex.quote, _root_set(j)))
                     + f' >> {shlex.quote(wl)} 2>&1 < /dev/null &')
            if host in (me, 'localhost'):
                subprocess.run(['bash', '-c', inner], check=False)
            else:
                env = dict(os.environ)
                env.update(cfg.runner.SSH_PREFIX_ENV)
                subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', host, inner], env=env,
                               check=False, timeout=60)
            print(f'{_now()} started worker {host} lane {lane}', flush=True)
    while job_state(job)[0] != 'finished':
        time.sleep(cfg.runner.POLL_S)
    print(f'{_now()} job finished: {job_state(job)[1]}', flush=True)


def _root_set(j):
    return ['--set', f"runner.JOBS_ROOT={j['jobs_root']}"]


def _worker_env(j):
    env = {'PYTHONPATH': j['pkg_root']}
    if j.get('site_file'):
        env['EXOPIPE3_SITE'] = j['site_file']
    env.update(_cfg().runner.ENV_DEFAULTS)
    return env


def _claim(job, item):
    try:
        os.mkdir(os.path.join(job_dir(job), 'claims', _safe(item)))
        return True
    except FileExistsError:
        return False


def _release(job, item):
    try:
        os.rmdir(os.path.join(job_dir(job), 'claims', _safe(item)))
    except OSError:
        pass


def _argv_for(j, item):
    argv = list(j['step_argv'])
    if any('{item}' in a for a in argv):
        argv = [a.replace('{item}', item) for a in argv]
    else:
        argv += ['--cells', item]
    return argv


def worker(job, lane=0):
    """Claim items in order and run them until none is left, HOLD stops new claims, or DRAIN.<host> exists."""
    cfg = _cfg()
    host = socket.gethostname()
    d = job_dir(job)
    while True:
        j = load_job(job)
        if os.path.exists(os.path.join(d, f'DRAIN.{host}')):
            print(f'{_now()} drain requested for {host}; worker exits', flush=True)
            return
        if os.path.exists(os.path.join(d, 'HOLD')):
            time.sleep(cfg.runner.POLL_S)
            continue
        picked = None
        for it in items(job):
            st = item_state(job, it)
            if st.get('status') in TERMINAL or st.get('status') == 'running':
                continue
            if _claim(job, it):
                picked = (it, st)
                break
        if picked is None:
            print(f'{_now()} no unclaimed items; worker exits', flush=True)
            return
        it, st = picked
        attempts = int(st.get('attempts', 0)) + 1
        log = os.path.join(d, 'logs', f'{_safe(it)}.log')
        argv = [j['python'], '-m', 'exopipe.phase3'] + _argv_for(j, it)
        rec = {'status': 'running', 'host': host, 'lane': lane, 'attempts': attempts, 'started': _now(), 'log': log,
               'argv': argv}
        _wjson(os.path.join(d, 'items', f'{_safe(it)}.json'), rec)
        with open(log, 'a') as lf:
            lf.write(f'\n===== {_now()} {host} lane {lane} attempt {attempts}: {" ".join(map(shlex.quote, argv))}\n')
            lf.flush()
            p = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 cwd=j['pkg_root'], env=_child_env(j))
            rec['pid'] = p.pid
            _wjson(os.path.join(d, 'items', f'{_safe(it)}.json'), rec)
            rc = p.wait()
        status = RC_STATUS.get(rc, 'error')
        if status == 'error' and attempts <= int(j.get('retry', 0)):
            status = 'queued'                       # retried by the next free worker
        elif status == 'error':
            status = 'failed'
        rec.update(status=status, rc=rc, finished=_now())
        _wjson(os.path.join(d, 'items', f'{_safe(it)}.json'), rec)
        _release(job, it)


def hold(job, on=True):
    p = os.path.join(job_dir(job), 'HOLD')
    if on:
        open(p, 'w').write(_now() + '\n')
    elif os.path.exists(p):
        os.remove(p)


def drain(job, host):
    open(os.path.join(job_dir(job), f'DRAIN.{host}'), 'w').write(_now() + '\n')


def retry(job, which=('failed',)):
    """Re-queue items whose status is in ``which`` (e.g. failed, refused) and start a supervisor again."""
    n = 0
    for it in items(job):
        st = item_state(job, it)
        if st.get('status') in which:
            st.update(status='queued', attempts=0)
            _wjson(os.path.join(job_dir(job), 'items', f'{_safe(it)}.json'), st)
            n += 1
    j = load_job(job)
    if n:
        log = os.path.join(job_dir(job), 'supervisor.log')
        subprocess.Popen([j['python'], '-m', 'exopipe.phase3', 'run', 'supervise', job, '--cores', 'none'] + _root_set(j),
                         stdout=open(log, 'a'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         cwd=j['pkg_root'], start_new_session=True, env=_child_env(j))
    return n


def reorder(job, first):
    """Move the given items to the front of the run order (takes effect at the next claim; nothing restarts)."""
    cur = items(job)
    first = [x for x in first if x in cur]
    new = first + [x for x in cur if x not in first]
    tmp = os.path.join(job_dir(job), f'items.txt.tmp.{os.getpid()}')
    with open(tmp, 'w') as f:
        f.write(''.join(f'{it}\n' for it in new))
    os.replace(tmp, os.path.join(job_dir(job), 'items.txt'))
    return len(first)


def list_jobs(n=20):
    root = jobs_root()
    if not os.path.isdir(root):
        return []
    return sorted(os.listdir(root))[-n:]


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
