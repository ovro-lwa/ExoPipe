#!/usr/bin/env python3
"""STANDING TEST: the CLI's SIGTERM handler interrupts only the step's own process (cli._sigterm_interrupts).

While a step runs, SIGTERM raises Terminated in its main thread (fix verification N2, 2026-11-03). Every process forked from the
step inherits that handler: the pool workers of the step and of the in-place modules it calls, and Pool.terminate() (every
`with Pool(...)` exit) ends its workers with SIGTERM. Before 2026-11-17 a forked worker raised Terminated too:
 - the photometry parity run P3 (lwacalim09, 2026-11-17) wrote a second run record from a worker that had unwound into the step's
   own code (two records, identical but for the pid);
 - a reproduction (fork pools with maxtasksperchild=1) raised Terminated in a child during its after-fork lock reinit, i.e. before
   multiprocessing's own try block; the step process itself then received Terminated with no SIGTERM sent to it and hung at
   exit (killed by its 600 s timeout).

Checks (each in a fresh interpreter, so the signal dispositions are this test's own):
 1. pool churn under the handler: 12 rounds of a fork Pool(8, maxtasksperchild=1) over 64 tasks; nothing prints the SIGTERM
    notice or a Terminated traceback, the step process is never interrupted, only it runs on after each pool, and it exits.
 2. a forked child sent SIGTERM directly ends by SIGTERM's default action, silently, while the handler is installed.
 3. the step process itself is still interrupted: SIGTERM to it raises Terminated (its notice printed once), and SIGTERM is back
    to the default action after the step.

Run standalone: python tests/test_cli_sigterm_children.py    Under pytest: pytest tests/test_cli_sigterm_children.py
"""
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHURN = r'''
import multiprocessing as mp, os, sys, time
sys.path.insert(0, sys.argv[1])
from exopipe.phase3 import cli
out = sys.argv[2]


def task(i):
    time.sleep(0.001 * (i % 5))
    return i


owner = os.getpid()
with cli._sigterm_interrupts('photometry'):
    for k in range(12):
        try:
            with mp.get_context('fork').Pool(8, maxtasksperchild=1) as p:
                p.map(task, range(64), chunksize=1)
        except BaseException as e:
            with open(out, 'a') as f:
                f.write('exc %d %s\n' % (os.getpid(), type(e).__name__))
        with open(out, 'a') as f:
            f.write('after %d %d\n' % (k, os.getpid()))
with open(out, 'a') as f:
    f.write('end %d %d\n' % (os.getpid(), owner))
'''

CHILD = r'''
import os, signal, sys, time
sys.path.insert(0, sys.argv[1])
from exopipe.phase3 import cli
with cli._sigterm_interrupts('photometry'):
    pid = os.fork()
    if pid == 0:
        try:
            time.sleep(30)
        except BaseException as e:
            print('child raised %s' % type(e).__name__, flush=True)
        os._exit(7)
    time.sleep(0.5)
    os.kill(pid, signal.SIGTERM)
    _, st = os.waitpid(pid, 0)
    print('child signaled %s signal %s exit %s' % (os.WIFSIGNALED(st), os.WTERMSIG(st) if os.WIFSIGNALED(st) else None,
                                                   os.WEXITSTATUS(st) if os.WIFEXITED(st) else None), flush=True)
'''

OWNER = r'''
import os, signal, sys, time
sys.path.insert(0, sys.argv[1])
from exopipe.phase3 import cli
try:
    with cli._sigterm_interrupts('photometry'):
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)
        print('not interrupted', flush=True)
except cli.Terminated as e:
    print('interrupted %s' % e, flush=True)
print('after %s' % (signal.getsignal(signal.SIGTERM) == signal.SIG_DFL), flush=True)
'''

FAILED = []


def check(label, ok, detail=''):
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', label, '' if ok else '  [%s]' % str(detail)[:600]))
    if not ok:
        FAILED.append(label)


def run(code, *args, timeout=240):
    """(returncode or 'timeout', stdout, stderr) of ``code`` in a fresh interpreter, in a session of its own: on a timeout
    (the hang this test exists for) the whole process group is killed, so no orphaned pool worker outlives the test."""
    import signal
    env = dict(os.environ, NUMPY_MADVISE_HUGEPAGE='0')
    p = subprocess.Popen([sys.executable, '-c', code, REPO] + list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env=env, start_new_session=True)
    try:
        so, se = p.communicate(timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        so, se = p.communicate()
        rc = 'timeout'
    return rc, so.decode('utf-8', 'replace'), se.decode('utf-8', 'replace')


def test_pool_churn():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, 'out.txt')
        rc, so, se = run(CHURN, out, timeout=120)
        lines = open(out).read().split('\n') if os.path.exists(out) else []
        end = [ln.split() for ln in lines if ln.startswith('end ')]
        owner = end[0][2] if len(end) == 1 else None
        after = [ln.split() for ln in lines if ln.startswith('after ')]
        exc = [ln for ln in lines if ln.startswith('exc ')]
        check('pool churn under the handler: the step process exits (no hang), is never interrupted, and only it runs on after '
              'each of the 12 pools', rc == 0 and owner is not None and not exc and len(after) == 12
              and {a[2] for a in after} == {owner} and end[0][1] == owner, (rc, exc, after[:3], end, se[-400:]))
        check('pool churn under the handler: no SIGTERM notice and no Terminated traceback from any worker',
              'SIGTERM:' not in se and 'Terminated' not in se, se[-600:])
    assert not FAILED, FAILED


def test_forked_child_default_action():
    rc, so, se = run(CHILD)
    check('a forked child sent SIGTERM ends by the default action (signal 15), silently',
          rc == 0 and 'child signaled True signal 15 exit None' in so and 'child raised' not in so and 'SIGTERM:' not in se,
          (rc, so, se[-400:]))
    assert not FAILED, FAILED


def test_owner_still_interrupted():
    rc, so, se = run(OWNER)
    check('the step process itself: SIGTERM raises Terminated (notice printed once) and the default action is back after the '
          'step', rc == 0 and 'interrupted SIGTERM' in so and 'not interrupted' not in so and 'after True' in so
          and se.count('SIGTERM:') == 1, (rc, so, se[-400:]))
    assert not FAILED, FAILED


if __name__ == '__main__':
    for t in (test_pool_churn, test_forked_child_default_action, test_owner_still_interrupted):
        try:
            t()
        except AssertionError:
            pass
    print('%s: %d failed' % (os.path.basename(__file__), len(FAILED)))
    sys.exit(1 if FAILED else 0)
