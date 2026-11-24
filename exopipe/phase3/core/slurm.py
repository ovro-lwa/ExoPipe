"""Slurm commands for the per-hour driver (core/driver.py): submit, list the live jobs, read accounting, cancel.

The commands are config keys (driver.SBATCH, SQUEUE, SACCT, SCANCEL; each a command line), so the data-free tests run the
driver against a fake Slurm (tests/fake_slurm.py). Every call is one subprocess; nothing here keeps state.

    submit(cmd, name=..., partition=..., ...)  -> job id (sbatch --parsable --wrap CMD)
    live(user)                                 -> {job id: {'state', 'reason', 'name'}} of every job squeue lists for the user
                                                  (pending, running, completing, requeued, suspended); None when squeue failed
    accounting(ids)                            -> {job id: {'state', 'exit', 'elapsed', 'maxrss', 'nodes'}} (sacct; jobs
                                                  slurmdbd does not know yet are absent)
    cancel(ids)                                -> scancel of exactly these job ids (the driver passes only ids it submitted)
"""
import shlex
import subprocess

SQUEUE_FORMAT = '%i|%T|%r|%j'
SACCT_FIELDS = 'JobIDRaw,State,ExitCode,Elapsed,MaxRSS,NodeList'
TIMEOUT_S = 60


class SlurmError(RuntimeError):
    """A Slurm command failed (its exit status and output are in the message)."""


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _cmd(key):
    return shlex.split(getattr(_cfg().driver, key))


def _run(argv, timeout=TIMEOUT_S):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SlurmError(f'{" ".join(map(shlex.quote, argv[:3]))} ...: {type(e).__name__} {e}')
    if r.returncode != 0:
        raise SlurmError(f'{" ".join(map(shlex.quote, argv[:3]))} ... exit {r.returncode}: '
                         f'{(r.stderr or r.stdout).strip()[-600:]}')
    return r.stdout


def sbatch_argv(cmd, name, partition, time_limit, mem, cpus, output, chdir, nodelist=None, dependency=None,
                licenses=None, requeue=True):
    """The sbatch command line of one job (without running it): every job gives --time and --mem (design 5.2: DefaultTime
    is unset and DefMemPerNode is UNLIMITED on the cluster), runs ``cmd`` (a shell command line) via --wrap, appends its
    output to ``output`` and is requeued when preempted."""
    if not (partition and time_limit and mem and cpus):
        raise SlurmError(f'job {name}: partition, --time, --mem and --cpus-per-task are all required '
                         f'(got {partition!r}, {time_limit!r}, {mem!r}, {cpus!r})')
    a = _cmd('SBATCH') + ['--parsable', f'--job-name={name}', f'--partition={partition}', f'--time={time_limit}',
                          f'--mem={mem}', f'--cpus-per-task={int(cpus)}', '--ntasks=1', '--nodes=1',
                          f'--output={output}', '--open-mode=append', f'--chdir={chdir}', '--export=ALL']
    a.append('--requeue' if requeue else '--no-requeue')
    if nodelist:
        a.append(f'--nodelist={nodelist}')
    if dependency:
        a.append(f'--dependency={dependency}')
    if licenses:
        a.append(f'--licenses={licenses}')
    return a + ['--wrap', cmd]


def submit(cmd, **kw):
    """Submit one job; returns its id (str). Raises SlurmError when sbatch refuses it."""
    out = _run(sbatch_argv(cmd, **kw)).strip()
    jid = out.split(';', 1)[0].strip()
    if not jid.isdigit():
        raise SlurmError(f'sbatch printed {out!r}, not a job id')
    return jid


def live(user):
    """{job id: {'state', 'reason', 'name'}} of every job squeue lists for ``user``; None when squeue failed (the caller
    then acts on nothing: a missing job never means a finished one unless squeue answered)."""
    try:
        out = _run(_cmd('SQUEUE') + ['-h', '-u', user, '-o', SQUEUE_FORMAT])
    except SlurmError as e:
        print(f'squeue failed: {e}', flush=True)
        return None
    jobs = {}
    for line in out.splitlines():
        parts = line.strip().split('|', 3)
        if len(parts) == 4 and parts[0]:
            jobs[parts[0]] = {'state': parts[1], 'reason': parts[2], 'name': parts[3]}
    return jobs


def _rss_bytes(text):
    t = (text or '').strip()
    if not t:
        return None
    mult = {'K': 2 ** 10, 'M': 2 ** 20, 'G': 2 ** 30, 'T': 2 ** 40}
    try:
        return int(float(t[:-1]) * mult[t[-1]]) if t[-1] in mult else int(float(t))
    except (ValueError, IndexError):
        return None


def accounting(ids):
    """{job id: {'state', 'exit', 'elapsed', 'maxrss' (bytes, the largest of the job's steps), 'nodes'}} from sacct for
    the given job ids. Jobs slurmdbd does not list (yet) are absent; a failed sacct gives {}."""
    ids = [str(i) for i in ids if i]
    if not ids:
        return {}
    try:
        out = _run(_cmd('SACCT') + ['-n', '-P', '-j', ','.join(ids), '-o', SACCT_FIELDS])
    except SlurmError as e:
        print(f'sacct failed: {e}', flush=True)
        return {}
    acct = {}
    for line in out.splitlines():
        p = line.strip().split('|')
        if len(p) < 6 or not p[0]:
            continue
        base, _, sub = p[0].partition('.')
        if base not in ids:
            continue
        a = acct.setdefault(base, {'state': None, 'exit': None, 'elapsed': None, 'maxrss': None, 'nodes': None})
        rss = _rss_bytes(p[4])
        if rss is not None and (a['maxrss'] is None or rss > a['maxrss']):
            a['maxrss'] = rss
        if not sub:                                   # the job's own line (not .batch / .extern)
            a.update(state=p[1].split()[0] if p[1] else p[1], exit=p[2], elapsed=p[3], nodes=p[5])
    return acct


def cancel(ids):
    """scancel exactly these job ids (never a name, user or pattern). Returns the ids passed."""
    ids = [str(i) for i in ids if i]
    if ids:
        _run(_cmd('SCANCEL') + ids)
    return ids
