#!/usr/bin/env python3
"""STANDING TEST: a SIGTERM sent to a process forked from a step ends it, whenever after the fork it arrives
(cli._sigterm_interrupts, cli._reset_inherited_sigterm).

From the fork until the CLI's at-fork reset gives the child SIGTERM's default disposition, the child still holds the step's
C-level signal catcher, and CPython drops a SIGTERM caught there: silently before the child clears its pending-signal flags
(PyOS_AfterFork_Child), and with "Signal 15 ignored due to race condition" inside the reset's own signal.signal() call.
Pool.terminate() signals workers that may have been forked a moment before; on lwacalim07 (2026-11-26) such workers lived on
with SIGTERM at its default and nothing pending, waiting for a task under the pool's queue lock, while the step waited in the
pool's join of them: tests/test_cli_sigterm_children.py hung. Since then SIGTERM is blocked in the forking thread through
every os.fork() and unblocked in the child only after the reset.

Checks (each in a fresh interpreter, so the signal dispositions and at-fork hooks are this test's own):
 1. the window is closed: an at-fork hook registered before the CLI's, which therefore runs in the child between the fork and
    the CLI's reset, finds SIGTERM blocked there - for a fork from the step's main thread and from another thread (a pool's
    worker handler forks from one); the child's own code then runs with SIGTERM unblocked at its default action; the forking
    thread has SIGTERM unblocked again after the fork.
 2. a SIGTERM delivered in that window: the same hook sends SIGTERM to the child itself; the child ends by signal 15 and its
    own code never runs.
 3. the race as Pool.terminate() meets it: on one CPU, the step sends SIGTERM to each child the moment os.fork() returns, so
    the signal is there before the child has run; 40 children forked from the main thread and 40 from another thread all end
    by signal 15, none runs its own code, and none prints the race-condition message. Before the fix all 80 lived on
    (lwacalim07, 2026-11-26).

Run standalone: python tests/test_cli_sigterm_fork_window.py    Under pytest: pytest tests/test_cli_sigterm_fork_window.py
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WINDOW = r'''
import os, signal, sys, threading
sys.path.insert(0, sys.argv[1])
kill = sys.argv[2] == 'kill'
r, w = os.pipe()


def in_window():                       # registered before the CLI's hooks: in the child it runs before the CLI's reset
    blocked = signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK, [])
    os.write(w, ('window %d blocked=%s\n' % (os.getpid(), blocked)).encode())
    if kill:
        os.kill(os.getpid(), signal.SIGTERM)


os.register_at_fork(after_in_child=in_window)
from exopipe.phase3 import cli
lines = []


def fork_one(where):
    pid = os.fork()
    if pid == 0:
        blocked = signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK, [])
        dfl = signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
        os.write(w, ('body %d blocked=%s default=%s\n' % (os.getpid(), blocked, dfl)).encode())
        os._exit(7)
    _, st = os.waitpid(pid, 0)
    lines.append('%s %d signal=%s exit=%s forker_blocked=%s' % (
        where, pid, os.WTERMSIG(st) if os.WIFSIGNALED(st) else None, os.WEXITSTATUS(st) if os.WIFEXITED(st) else None,
        signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK, [])))


with cli._sigterm_interrupts('photometry'):
    fork_one('main')
    t = threading.Thread(target=fork_one, args=('thread',))
    t.start()
    t.join()
os.close(w)
data = b''
while True:
    chunk = os.read(r, 65536)
    if not chunk:
        break
    data += chunk
print(data.decode() + '\n'.join(lines), flush=True)
'''

RACE = r'''
import os, signal, sys, threading
sys.path.insert(0, sys.argv[1])
from exopipe.phase3 import cli
n = int(sys.argv[2])
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})   # one CPU: the step runs on and signals before the child has run
counts = {}


def forks(where):
    k = {'signal15': 0, 'ran': 0, 'other': 0}
    for _ in range(n):
        pid = os.fork()
        if pid == 0:
            os._exit(7)                # the child's own code: reached only by a child whose SIGTERM was lost
        os.kill(pid, signal.SIGTERM)
        _, st = os.waitpid(pid, 0)
        if os.WIFSIGNALED(st) and os.WTERMSIG(st) == signal.SIGTERM:
            k['signal15'] += 1
        elif os.WIFEXITED(st) and os.WEXITSTATUS(st) == 7:
            k['ran'] += 1
        else:
            k['other'] += 1
    counts[where] = k


with cli._sigterm_interrupts('photometry'):
    forks('main')
    t = threading.Thread(target=forks, args=('thread',))   # the thread inherits the one-CPU affinity
    t.start()
    t.join()
for where in ('main', 'thread'):
    k = counts.get(where, {})
    print('%s signal15 %s ran %s other %s' % (where, k.get('signal15'), k.get('ran'), k.get('other')), flush=True)
'''

FAILED = []


def check(label, ok, detail=''):
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', label, '' if ok else '  [%s]' % str(detail)[:800]))
    if not ok:
        FAILED.append(label)


def run(code, *args, timeout=120):
    """(returncode or 'timeout', stdout, stderr) of ``code`` in a fresh interpreter, in a session of its own: on a timeout the
    whole process group is killed, so no child outlives the test."""
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


def test_window_closed():
    rc, so, se = run(WINDOW, 'probe')
    ln = so.split('\n')
    window = [x for x in ln if x.startswith('window ')]
    body = [x for x in ln if x.startswith('body ')]
    ends = [x for x in ln if x.startswith(('main ', 'thread '))]
    check('the window between the fork and the CLI\'s reset: SIGTERM is blocked in the child there (fork from the main thread '
          'and from another thread)', rc == 0 and len(window) == 2 and all(x.endswith('blocked=True') for x in window),
          (rc, so, se[-400:]))
    check('the child\'s own code: SIGTERM unblocked at its default action; each child exits 7 by itself; the forking thread '
          'has SIGTERM unblocked after the fork', rc == 0 and len(body) == 2
          and all(x.endswith('blocked=False default=True') for x in body) and len(ends) == 2
          and all(x.endswith('signal=None exit=7 forker_blocked=False') for x in ends), (rc, so, se[-400:]))
    assert not FAILED, FAILED


def test_sigterm_in_window():
    rc, so, se = run(WINDOW, 'kill')
    ln = so.split('\n')
    window = [x for x in ln if x.startswith('window ')]
    body = [x for x in ln if x.startswith('body ')]
    ends = [x for x in ln if x.startswith(('main ', 'thread '))]
    check('a SIGTERM sent to the child inside the window ends it by signal 15 before its own code runs (fork from the main '
          'thread and from another thread)', rc == 0 and len(window) == 2 and not body and len(ends) == 2
          and all(' signal=15 exit=None forker_blocked=False' in x for x in ends)
          and 'race condition' not in se and 'Terminated' not in se, (rc, so, se[-400:]))
    assert not FAILED, FAILED


def test_sigterm_at_fork_return():
    n = 40
    rc, so, se = run(RACE, str(n), timeout=300)
    want = ['main signal15 %d ran 0 other 0' % n, 'thread signal15 %d ran 0 other 0' % n]
    check('SIGTERM sent the moment os.fork() returns, on one CPU (before the child has run): all %d children from the main '
          'thread and all %d from another thread end by signal 15, none runs its own code' % (n, n),
          rc == 0 and all(x in so for x in want) and 'race condition' not in se and 'Terminated' not in se,
          (rc, so, se[-400:]))
    assert not FAILED, FAILED


if __name__ == '__main__':
    for t in (test_window_closed, test_sigterm_in_window, test_sigterm_at_fork_return):
        try:
            t()
        except AssertionError:
            pass
    print('%s: %d failed' % (os.path.basename(__file__), len(FAILED)))
    sys.exit(1 if FAILED else 0)
