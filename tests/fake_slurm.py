#!/usr/bin/env python3
"""A fake Slurm for tests/test_drive.py: sbatch, squeue, sacct and scancel over one JSON state file (env FAKE_SLURM_STATE),
in the output formats core/slurm.py reads; and, for the test itself, run_ready() / start() / wait(), which run the jobs.

    python tests/fake_slurm.py sbatch --parsable --job-name=N ... --dependency=afterok:1001,singleton --wrap CMD
    python tests/fake_slurm.py squeue -h -u USER -o '%i|%T|%r|%j'
    python tests/fake_slurm.py sacct -n -P -j 1001,1002 -o JobIDRaw,State,ExitCode,Elapsed,MaxRSS,NodeList
    python tests/fake_slurm.py scancel 1001 1002

Jobs run only when the test calls run_ready() (every eligible job, one at a time, in id order) or start() (one job, in the
background, so the test can scancel it mid-way). Eligible: PENDING, every afterok parent COMPLETED, and (singleton) no RUNNING
job of the same name. A pending job whose afterok parent ended other than COMPLETED shows the reason DependencyNeverSatisfied
and stays pending (Slurm without kill_invalid_depend, design 5.3). scancel ends a running job with SIGTERM to its process group
(as Slurm signals every process of the job) and marks it CANCELLED.
"""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

ENDED = ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL')


class State:
    """The state file under an exclusive flock for the block; written back when the block ends without an exception."""

    def __init__(self, path=None):
        self.path = path or os.environ['FAKE_SLURM_STATE']

    def __enter__(self):
        self.lf = open(self.path + '.lock', 'a')
        fcntl.flock(self.lf, fcntl.LOCK_EX)
        try:
            with open(self.path) as f:
                self.s = json.load(f)
        except FileNotFoundError:
            self.s = {'next': 1001, 'jobs': {}, 'refuse_sbatch': False, 'cancelled': []}
        return self.s

    def __exit__(self, exc, *rest):
        if exc is None:
            tmp = f'{self.path}.tmp'
            with open(tmp, 'w') as f:
                json.dump(self.s, f, indent=1)
            os.replace(tmp, self.path)
        fcntl.flock(self.lf, fcntl.LOCK_UN)
        self.lf.close()


def reason(s, j):
    if j['state'] == 'RUNNING':
        return 'None'
    for p in j['afterok']:
        pj = s['jobs'].get(p)
        if pj and pj['state'] in ENDED and not (pj['state'] == 'COMPLETED' and pj['rc'] == 0):
            return 'DependencyNeverSatisfied'
    for p in j['afterok']:
        if s['jobs'].get(p, {}).get('state') != 'COMPLETED':
            return 'Dependency'
    if j['singleton'] and any(o['state'] == 'RUNNING' and o['name'] == j['name'] and o['id'] != j['id']
                              for o in s['jobs'].values()):
        return 'Dependency'
    return 'Priority'


def sbatch(argv):
    opts, flags, cmd = {}, [], None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--wrap':
            cmd = argv[i + 1]
            i += 2
            continue
        if a.startswith('--') and '=' in a:
            k, v = a[2:].split('=', 1)
            opts[k] = v
        else:
            flags.append(a)
        i += 1
    with State() as s:
        if s.get('refuse_sbatch'):
            print('sbatch: error: Batch job submission failed: Invalid partition name specified', file=sys.stderr)
            return 1
        dep = opts.get('dependency', '')
        afterok = [t.split(':', 1)[1] for t in dep.split(',') if t.startswith('afterok:')]
        for p in afterok:
            if p not in s['jobs']:
                print(f'sbatch: error: Batch job submission failed: Job dependency problem ({p})', file=sys.stderr)
                return 1
        jid = str(s['next'])
        s['next'] += 1
        s['jobs'][jid] = {'id': jid, 'name': opts.get('job-name'), 'opts': opts, 'flags': flags, 'cmd': cmd,
                          'afterok': afterok, 'singleton': 'singleton' in dep.split(','), 'state': 'PENDING',
                          'rc': None, 'pid': None, 'submitted': time.time(), 'elapsed': 0}
    print(jid if '--parsable' in flags else f'Submitted batch job {jid}')
    return 0


def squeue(argv):
    with State() as s:
        for jid in sorted(s['jobs'], key=int):
            j = s['jobs'][jid]
            if j['state'] in ('PENDING', 'RUNNING'):
                print(f'{jid}|{j["state"]}|{reason(s, j)}|{j["name"]}')
    return 0


def sacct(argv):
    ids = argv[argv.index('-j') + 1].split(',')
    with State() as s:
        for jid in ids:
            j = s['jobs'].get(jid)
            if not j:
                continue
            rc = j['rc'] if j['rc'] is not None else 0
            sig = 15 if j['state'] == 'CANCELLED' else 0
            el = time.strftime('%H:%M:%S', time.gmtime(j['elapsed']))
            node = j['opts'].get('nodelist', 'fakenode')
            print(f'{jid}|{j["state"] if j["state"] != "CANCELLED" else "CANCELLED by 1008"}|{rc}:{sig}|{el}||{node}')
            if j['state'] in ENDED:
                print(f'{jid}.batch|{j["state"]}|{rc}:{sig}|{el}|2048K|{node}')
    return 0


def scancel(argv):
    with State() as s:
        for jid in argv:
            j = s['jobs'].get(jid)
            if not j or j['state'] in ENDED:
                continue
            if j['state'] == 'RUNNING' and j['pid']:
                try:
                    os.killpg(j['pid'], signal.SIGTERM)
                except ProcessLookupError:
                    pass
            j['state'] = 'CANCELLED'
            s['cancelled'].append(jid)
    return 0


# ---- used by the test: run the jobs ---------------------------------------------------------------------------------

def _shell(j, env):
    o = j['opts'].get('output')
    if o:
        os.makedirs(os.path.dirname(o), exist_ok=True)
    out = open(o, 'a') if o else subprocess.DEVNULL
    e = dict(env or os.environ)
    e['SLURM_JOB_ID'] = j['id']
    e['SLURM_RESTART_COUNT'] = '0'
    return subprocess.Popen(['bash', '-c', j['cmd']], stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            cwd=j['opts'].get('chdir'), env=e, start_new_session=True)


def start(path, jid, env=None):
    """Start one job in the background (it must be eligible); returns its Popen."""
    with State(path) as s:
        j = s['jobs'][jid]
        assert j['state'] == 'PENDING' and reason(s, j) == 'Priority', (jid, j['state'], reason(s, j))
        p = _shell(j, env)
        j.update(state='RUNNING', pid=p.pid, started=time.time())
    return p


def wait(path, jid, p, timeout=300):
    rc = p.wait(timeout=timeout)
    with State(path) as s:
        j = s['jobs'][jid]
        j['elapsed'] = int(time.time() - j.get('started', time.time()))
        if j['state'] == 'RUNNING':
            j.update(state='COMPLETED' if rc == 0 else 'FAILED', rc=rc)
        elif j['rc'] is None:
            j['rc'] = rc
    return rc


def run_ready(path, env=None, limit=200):
    """Run every eligible job, one at a time, in id order, until none is eligible; [(id, name, state, rc)]."""
    done = []
    for _ in range(limit):
        with State(path) as s:
            ready = [jid for jid in sorted(s['jobs'], key=int)
                     if s['jobs'][jid]['state'] == 'PENDING' and reason(s, s['jobs'][jid]) == 'Priority']
        if not ready:
            return done
        p = start(path, ready[0], env)
        rc = wait(path, ready[0], p)
        with State(path) as s:
            j = s['jobs'][ready[0]]
            done.append((j['id'], j['name'], j['state'], rc))
    raise AssertionError(f'run_ready: more than {limit} jobs')


def jobs(path):
    with State(path) as s:
        return s['jobs']


def main(argv):
    cmd, rest = argv[0], argv[1:]
    return {'sbatch': sbatch, 'squeue': squeue, 'sacct': sacct, 'scancel': scancel}[cmd](rest)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
