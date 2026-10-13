"""The multi-node runner behind `exopipe3 run` - one implementation for every fleet, queue and driver.

A JOB runs one step (or a CHAIN of steps) over a list of ITEMS (cells, LSTs, frame batches) on a set of nodes. Its
record lives in ``CFG.runner.JOBS_ROOT/<job>/`` (a handful of files per item, on Lustre):

    job.json              id, step argv template, nodes and lanes, --after, --retry, state, created, launch host;
                          forward_argv / overrides (the --set, --config, --ruling-for, --parity given to `run submit`,
                          forwarded into every step command); chain + chain_steps (a chain job); code_commit +
                          code_root (a --code job)
    items.txt             the items in run order (rewritten by `run reorder`)
    items/<item>.json     per item: status (queued|running|done|refused|no-input|error|failed), host, pid, rc,
                          attempts, started, finished, log path; code_commit (--code); steps (chain: one entry per
                          step run, with its own status, rc, started, finished)
    claims/<item>/        atomic claim (mkdir) by the worker that runs the item
    logs/<item>.log       the step's stdout/stderr for that item (every step of a chain, in order)
    HOLD                  present = workers take no new items (running items finish)
    DRAIN.<host>          present = workers on <host> exit after their current item

Workers are started with ssh on each node (``nohup setsid``), so a job survives the session that launched it. Each
item runs as ``python -m exopipe.phase3 <step> ... `` with the item substituted for ``{item}`` (or appended as
``--cells <item>``), pinned to the node's recorder-free cores (``--cores auto`` read at every launch). An item's status
is the step's exit code: 0 done, 2 refused, 3 no-input, other error; an error is retried up to --retry times.
``--after JOB`` waits until that job's record is finished (every item in a terminal state) - never on log text,
process names or pids.

Chains (``run submit --chain NAME``; ``CFG.runner.CHAINS``): one worker runs every step of an item in order on its
node; a step that ends no-input (exit 3: nothing to do for this item) is recorded and the chain goes on; the item stops
at its first refused (2) or error exit, and its status is that step's. An item where every step had no input ends
no-input. A retried
chain item resumes at its first step not done.

Pinned code (``run submit --code COMMIT``): ``ensure_code_worktree`` makes ONE read-only ``git worktree add --detach``
of the commit under ``CFG.runner.CODE_WORKTREES_ROOT/<sha>`` (reused by every later job of that commit) and every step
process of the job runs from it (cwd and PYTHONPATH). The supervisor and worker loops run from the tree that submitted
the job, so the job record format is the submitting runner's; the step code is the commit's. Legacy code roots of the
site file (clv15, exopipe_ops) are still imported in place. To delete a code worktree: ``chmod -R u+w <path> && git -C
<repo> worktree remove <path>``.
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


def _pkg_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


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


# ---------------------------------------------------------------------------------------------------------------------
# pinned code: read-only git worktrees of a commit (run submit --code COMMIT)
# ---------------------------------------------------------------------------------------------------------------------

def code_repo():
    """The repository `--code` commits come from (runner.CODE_REPO, default the repository this package runs from)."""
    return _cfg().runner.CODE_REPO or _pkg_root()


def _git(repo, *args, timeout=600):
    return subprocess.run(['git', '-C', repo] + list(args), capture_output=True, text=True, timeout=timeout)


def _is_code_worktree(path, sha):
    if not os.path.isdir(path):
        return False
    r = _git(path, 'rev-parse', 'HEAD', timeout=60)
    return r.returncode == 0 and r.stdout.strip() == sha


def _make_read_only(path):
    for dp, dn, fn in os.walk(path, topdown=False):
        for x in fn + dn:
            p = os.path.join(dp, x)
            if not os.path.islink(p):
                os.chmod(p, os.stat(p).st_mode & ~0o222)
    os.chmod(path, os.stat(path).st_mode & ~0o222)


def ensure_code_worktree(commit, repo=None, root=None, python=None, wait_s=900):
    """(sha, path, created): a read-only ``git worktree add --detach`` of ``commit`` under ``root``/<full sha>, made
    once per commit (an existing worktree of that sha is reused; concurrent submits serialise on <path>.lock). The
    sources are byte-compiled with ``python`` before the tree is made read-only."""
    cfg = _cfg()
    repo = repo or code_repo()
    root = root or cfg.runner.CODE_WORKTREES_ROOT
    r = _git(repo, 'rev-parse', '--verify', '--quiet', f'{commit}^{{commit}}', timeout=60)
    sha = r.stdout.strip()
    if r.returncode != 0 or len(sha) != 40:
        raise SystemExit(f'refused: --code {commit}: not a commit of {repo}')
    path = os.path.join(root, sha)
    if _is_code_worktree(path, sha):
        return sha, path, False
    os.makedirs(root, exist_ok=True)
    lock = path + '.lock'
    t0 = time.time()
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            if time.time() - t0 > wait_s:
                raise SystemExit(f'refused: {lock} held for more than {wait_s} s (another submit is creating the '
                                 f'worktree, or a stale lock: remove it by hand)')
            time.sleep(2)
    try:
        if _is_code_worktree(path, sha):
            return sha, path, False
        if os.path.exists(path):
            raise SystemExit(f'refused: {path} exists but is not a worktree of {sha}; remove it by hand')
        r = _git(repo, 'worktree', 'add', '--detach', path, sha)
        if r.returncode != 0:
            raise SystemExit(f'refused: git worktree add {path} {sha} failed: {r.stderr.strip()[-400:]}')
        subprocess.run([python or sys.executable, '-m', 'compileall', '-q', path], capture_output=True, timeout=1800)
        _make_read_only(path)
    finally:
        os.rmdir(lock)
    return sha, path, True


# ---------------------------------------------------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------------------------------------------------

def chain_steps(name):
    """The step argv templates of the chain ``name`` (runner.CHAINS; each a command-line string or an argv list)."""
    chains = _cfg().runner.CHAINS
    if name not in chains:
        raise SystemExit(f'refused: no chain {name!r} in runner.CHAINS (chains: {", ".join(sorted(chains)) or "none"})')
    steps = [shlex.split(s) if isinstance(s, str) else list(s) for s in chains[name]]
    if not steps or not all(steps):
        raise SystemExit(f'refused: chain {name!r} has an empty step')
    return steps


def _overrides_of(forward):
    out = {'set': [], 'config': None, 'ruling_for': [], 'parity': False}
    i = 0
    while i < len(forward):
        a = forward[i]
        if a in ('--set', '--ruling-for') and i + 1 < len(forward):
            out['set' if a == '--set' else 'ruling_for'].append(forward[i + 1])
            i += 2
            continue
        if a == '--config' and i + 1 < len(forward):
            out['config'] = forward[i + 1]
            try:
                import hashlib
                out['config_sha1'] = hashlib.sha1(open(forward[i + 1], 'rb').read()).hexdigest()
            except OSError:
                out['config_sha1'] = None
            i += 2
            continue
        if a == '--parity':
            out['parity'] = True
        i += 1
    return out


def submit(name, step_argv, item_list, nodes, after=None, retry=0, gpu=False, python=None, start=True, chain=None,
           forward=(), code=None):
    """Create a job record and (unless start=False) launch its supervisor in the background.

    chain:   a runner.CHAINS name: every item runs through its steps in order (step_argv must then be empty).
    forward: argv tokens added to every step command (`run submit --set/--config/--ruling-for/--parity`).
    code:    a commit: step processes run from its read-only worktree (``ensure_code_worktree``)."""
    cfg = _cfg()
    if chain and step_argv:
        raise SystemExit('refused: give either --chain NAME or a step command after --, not both')
    steps = chain_steps(chain) if chain else None
    if gpu:
        bad = [h for h, _ in nodes if h not in cfg.site.GPU_HOSTS]
        if bad:
            raise SystemExit(f'refused: --gpu but {bad} are not GPU hosts in the site file (GPU_HOSTS)')
    python = python or sys.executable
    code_commit = code_root = None
    if code:
        code_commit, code_root, _ = ensure_code_worktree(code, python=python)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    label = name or chain or step_argv[0]
    job = f'{stamp}_{label}'
    d = job_dir(job)
    os.makedirs(os.path.join(d, 'items'))
    os.makedirs(os.path.join(d, 'claims'))
    os.makedirs(os.path.join(d, 'logs'))
    with open(os.path.join(d, 'items.txt'), 'w') as f:
        f.write(''.join(f'{it}\n' for it in item_list))
    forward = list(forward or ())
    j = {'id': job, 'step_argv': list(step_argv) if not chain else [f'chain:{chain}'],
         'nodes': [list(n) for n in nodes], 'after': after,
         'retry': int(retry), 'gpu': bool(gpu), 'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'launch_host': socket.gethostname(), 'python': python, 'pkg_root': _pkg_root(),
         'site_file': os.environ.get('EXOPIPE3_SITE', ''), 'n_items': len(item_list), 'jobs_root': jobs_root(),
         'forward_argv': forward, 'overrides': _overrides_of(forward),
         'chain': chain, 'chain_steps': steps,
         'code_commit': code_commit, 'code_root': code_root, 'code_repo': code_repo() if code else None}
    _wjson(os.path.join(d, 'job.json'), j)
    if start:
        start_supervisor(job)
    return job


def start_supervisor(job):
    """Launch the job's supervisor in the background (it starts the workers on every node)."""
    j = load_job(job)
    log = os.path.join(job_dir(job), 'supervisor.log')
    cmd = [j['python'], '-m', 'exopipe.phase3', 'run', 'supervise', job, '--cores', 'none'] + _root_set(j)
    subprocess.Popen(cmd, stdout=open(log, 'a'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     cwd=j['pkg_root'], start_new_session=True, env=_child_env(j))


def _child_env(j, root=None):
    """Environment of a process the runner starts. ``root`` (a --code worktree) replaces the package root on
    PYTHONPATH, so the step imports the pinned commit's exopipe and nothing from the submitting tree."""
    env = dict(os.environ)
    root = root or j['pkg_root']
    rest = [p for p in (env.get('PYTHONPATH') or '').split(os.pathsep) if p and p not in (j['pkg_root'], root)]
    env['PYTHONPATH'] = os.pathsep.join([root] + rest)
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


def _subst(argv, item):
    out = list(argv)
    for k, v in (getattr(_cfg().runner, 'CHAIN_VARS', None) or {}).items():   # named variables, e.g. {deep_run}
        out = [a.replace('{' + k + '}', str(v)) for a in out]
    if any('{item}' in a for a in out):
        return [a.replace('{item}', item) for a in out]
    return out + ['--cells', item]


def _argv_for(j, item):
    """The step command of one item (single-step job): the template with the item, plus the forwarded arguments."""
    return _subst(j['step_argv'], item) + list(j.get('forward_argv') or [])


def step_argvs(j, item):
    """Every step command of one item, in order (one for a step job, one per chain step for a chain job)."""
    if j.get('chain_steps'):
        return [_subst(a, item) + list(j.get('forward_argv') or []) for a in j['chain_steps']]
    return [_argv_for(j, item)]


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
        _run_item(j, job, picked[0], picked[1], host, lane)
        _release(job, picked[0])


def _run_item(j, job, it, st, host, lane):
    d = job_dir(job)
    ipath = os.path.join(d, 'items', f'{_safe(it)}.json')
    attempts = int(st.get('attempts', 0)) + 1
    log = os.path.join(d, 'logs', f'{_safe(it)}.log')
    argvs = step_argvs(j, it)
    chain = bool(j.get('chain_steps'))
    steps = list(st.get('steps') or []) if chain else None
    root = j.get('code_root') or j['pkg_root']
    head = [j['python'], '-m', 'exopipe.phase3']
    rec = {'status': 'running', 'host': host, 'lane': lane, 'attempts': attempts, 'started': _now(), 'log': log,
           'argv': head + argvs[0]}
    if j.get('code_commit'):
        rec['code_commit'] = j['code_commit']
    if chain:
        rec.update(chain=j['chain'], steps=steps)
    _wjson(ipath, rec)
    status, rc = 'done', 0
    n_noinput = 0
    with open(log, 'a') as lf:
        for k, argv in enumerate(argvs):
            if chain and k < len(steps) and steps[k].get('status') == 'done':
                continue                      # done in an earlier attempt: resume at the first step not done
            full = head + argv
            tag = f' step {k + 1}/{len(argvs)}' if chain else ''
            lf.write(f'\n===== {_now()} {host} lane {lane} attempt {attempts}{tag}: {" ".join(map(shlex.quote, full))}'
                     + (f'  [code {j["code_commit"]}]' if j.get('code_commit') else '') + '\n')
            lf.flush()
            p = subprocess.Popen(full, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=root,
                                 env=_child_env(j, root))
            rec.update(argv=full, pid=p.pid)
            if chain:
                entry = {'step': k, 'name': argv[0], 'argv': full, 'status': 'running', 'started': _now(),
                         'pid': p.pid, 'attempt': attempts}
                if k < len(steps):
                    steps[k] = entry
                else:
                    steps.append(entry)
                rec['step'] = k
            _wjson(ipath, rec)
            rc = p.wait()
            s = RC_STATUS.get(rc, 'error')
            if chain:
                entry.update(status=s, rc=rc, finished=_now())
                _wjson(ipath, rec)
            if rc == 3 and chain:
                n_noinput += 1                # nothing to do for this item in this step: the chain goes on (DESIGN 12)
                continue
            if rc != 0:
                status = s
                break
    if chain and status == 'done' and n_noinput == len(argvs):
        status = 'no-input'                   # every step had nothing to do
    if status == 'error' and attempts <= int(j.get('retry', 0)):
        status = 'queued'                       # retried by the next free worker
    elif status == 'error':
        status = 'failed'
    rec.update(status=status, rc=rc, finished=_now())
    _wjson(ipath, rec)


def hold(job, on=True):
    p = os.path.join(job_dir(job), 'HOLD')
    if on:
        open(p, 'w').write(_now() + '\n')
    elif os.path.exists(p):
        os.remove(p)


def drain(job, host):
    open(os.path.join(job_dir(job), f'DRAIN.{host}'), 'w').write(_now() + '\n')


def retry(job, which=('failed',), start=True):
    """Re-queue items whose status is in ``which`` (e.g. failed, refused) and start a supervisor again. A chain item
    keeps its per-step record and resumes at its first step not done."""
    n = 0
    for it in items(job):
        st = item_state(job, it)
        if st.get('status') in which:
            st.update(status='queued', attempts=0)
            _wjson(os.path.join(job_dir(job), 'items', f'{_safe(it)}.json'), st)
            n += 1
    if n and start:
        start_supervisor(job)
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
    """The latest job ids (directories holding a job.json; the code-worktree root and other entries are skipped)."""
    root = jobs_root()
    if not os.path.isdir(root):
        return []
    return sorted(x for x in os.listdir(root)
                  if not x.startswith(('_', '.')) and os.path.isfile(os.path.join(root, x, 'job.json')))[-n:]


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
