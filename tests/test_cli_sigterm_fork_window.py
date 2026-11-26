#!/usr/bin/env python3
"""STANDING TEST: SIGTERM and the step's forks (cli._sigterm_interrupts, its fork wrappers and at-fork hooks,
cli._reset_inherited_sigterm, cli._end_descendants). A SIGTERM sent to a process forked from a step ends it, whenever after
the fork it arrives; a SIGTERM sent to the step while it forks interrupts it after the fork returns; the hooks keep each
thread's signal mask; an interrupted step ends its descendants, and only them, before its rollbacks run.

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
    on lwacalim07 the signal is there before the child has run; 40 children forked from the main thread and 40 from another
    thread all end by signal 15, none completes its own code, and none prints the race-condition message. The child's own
    code first waits until the step has sent the signal (EOF on a pipe the step closes after the kill), so a child that
    completes it lost a SIGTERM sent while it was alive, whichever process the scheduler runs first (GitHub's ubuntu-24.04
    runner ran a child first, 2026-11-26: without that wait 1 of 40 had exited before the kill and counted as lost).
    Before the fix all 80 lived on (lwacalim07, 2026-11-26).
 4. the step itself, sent SIGTERM inside its fork window (an at-fork hook sends it, in the before hooks or in the
    after_in_parent hooks, registered before the CLI's hooks or after them; fork from the main thread with and without
    another thread, and from another thread; with another thread, the hook waits until that thread has taken the signal):
    Python drops an exception raised in an at-fork hook, so before the first 2026-11-26 fix round the step's Terminated was
    lost ("Exception ignored in ...") and the step ran on with the handler already gone; with that round's design (a C
    after_in_parent hook re-sent the signal) a hook registered after the CLI's still dropped it. Now Terminated is raised in
    the step's own code (for a fork from the main thread: at the os.fork() line, then only the CLI's fork wrapper and the
    handler), once; the unit is recorded as interrupted; the child exits 7 by itself or is ended by the step (signal 15,
    its SIGTERM logged); a second SIGTERM ends the step at once (signal 15).
 5. masks: in a thread that had SIGTERM blocked before the fork, it is still blocked after it, in the parent and in the child
    (main thread and another thread; the hooks used to unblock it unconditionally); inside the step SIGTERM is blocked in
    the window (the forking thread's before hooks, the child before the CLI's reset); outside the step (the CLI's handler
    entered once and gone again) the hooks change nothing. The saved mask is the forking thread's own: thread A, in its
    before hooks, waits while thread B forks from start to end, with A's and B's masks different both ways round.
 6. a foreign Python handler (installed by the step's code over the CLI's: sys.exit(3)) and a SIGTERM pending in the child
    at the end of the fork window: the child ends by signal 15, the handler does not run inside the hook (where its
    SystemExit was dropped and the child ran on: exit 7); a SIGTERM the child's own code receives later runs it (exit 3).
 7. a subprocess child of an interrupted step: Terminated raised inside Popen's start-up (the child, between its fork and its
    exec, sends the step SIGTERM and waits until the step has taken it) leaves the child with no Popen object to kill it;
    the step ends it before its rollbacks, and nothing of the step's session outlives the step (a session scan once it has
    exited; before: bash, then sleep 60, ran on).
 8. an interrupted step ends its descendants before its rollbacks (cli._end_descendants), leading its session and as the
    child of a wrapper that leads it: a child that drops the first SIGTERM and then execs (a subprocess child forked in C
    drops one that reaches it before its exec; here bash ignores it, then execs sleep 60) is sent SIGTERM again (logged as
    an exec since the look) and ends by signal 15; a child that ignores SIGTERM ends by SIGKILL after the grace period
    (signal 9; the test sets cli._CHILD_TERM_S to 1 s); a child in a process group of its own ends by signal 15; a child
    with a session of its own (start_new_session) is left alone; the child of a script killed with SIGKILL (as
    subprocess.run does when interrupted) is re-parented to the step (a child subreaper) and ends by signal 15 (with
    the 59b5998 design it outlived a step that did not lead its session); the wrapper's monitor, in the step's session but
    not its descendant, is never signaled; nothing else of the step's session outlives it.
 9. the order at a SIGTERM: a child that keeps writing a file (every 10 ms) which the rollback removes has ended before the
    rollback (signal 15), the file is still absent 0.5 s after the rollback, and the unit is recorded interrupted (with the
    59b5998 design, which ended the children after the rollbacks, the file was there again and the writer still running).
 10. a step interrupted inside `with Pool(3)` (one worker busy with a child of its own, two idle): the worker's child is
    ended; the workers are not signaled by the step but left to their pool, whose exit ends them, and the step exits. An
    idle worker ended from outside can hold the pool's task-queue lock, and Pool.terminate() then never returns: with the
    workers not left to their pool this test ran into its 120 s timeout (lwacalim07, 2026-11-26).
 11. each signal is checked first (cli._signal_checked): not sent to a pid whose start time differs from the one recorded
    at the look (a pid taken again by another process) or that has left the step's session, and each withheld signal is
    logged; sent to the same process after an exec (name and executable changed, start time the same), the exec recorded
    and logged.

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
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:                   # the child's own code: completed only by a child whose SIGTERM was lost
            os.close(w)
            os.read(r, 1)              # EOF once the step has sent SIGTERM and closed its end
            os._exit(7)
        os.close(r)
        os.kill(pid, signal.SIGTERM)
        os.close(w)
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

PARENT = r'''
import os, shutil, signal, sys, tempfile, threading, time, traceback
sys.path.insert(0, sys.argv[1])
phase, where, helper, reg = sys.argv[2], sys.argv[3], sys.argv[4] == 'helper', sys.argv[5]
armed = []
other = helper or where == 'thread'    # another thread has SIGTERM unblocked: it takes the signal while the hook runs


def pending():                         # SIGTERM pending in the process: not yet taken by any of its threads
    with open('/proc/self/status') as f:
        for ln in f:
            if ln.startswith('ShdPnd:'):
                return int(ln.split()[1], 16) >> (signal.SIGTERM - 1) & 1


def send():                            # an at-fork hook: runs inside the step's fork call
    if armed:
        armed.clear()
        os.kill(os.getpid(), signal.SIGTERM)
        t = time.monotonic() + 10
        while other and pending() and time.monotonic() < t:   # taken by another thread before this hook returns
            time.sleep(0.001)
        if other:
            time.sleep(0.002)


if reg == 'first':                     # registered before the CLI's hooks
    os.register_at_fork(**{phase: send})
from exopipe.phase3 import cli
from exopipe.phase3.config import CFG
from exopipe.phase3.core import record
if helper:                             # an idle thread: a SIGTERM to the process goes to it while the main thread forks
    threading.Thread(target=threading.Event().wait, daemon=True).start()
d = tempfile.mkdtemp(prefix='test_cli_sigterm_fork_window_')
out = ['threads %d' % len(os.listdir('/proc/self/task'))]


def fork_child():
    armed.append(True)
    pid = os.fork()
    if pid == 0:
        os._exit(7)


with cli._sigterm_interrupts('photometry'):
    if reg == 'last':                  # registered after the CLI's hooks (entering the step registered them)
        os.register_at_fork(**{phase: send})
    run = record.Run('t7s', ['x'], d, CFG, config_sections=())
    t = threading.Thread(target=fork_child)
    try:
        with run.unit('u') as u:
            if where == 'main':
                fork_child()
            else:
                t.start()
                t.join()
            out.append('not interrupted')
    except cli.Terminated as e:
        out.append('terminated in ' + ' > '.join('%s:%d' % (f.name, f.lineno) for f in traceback.extract_tb(e.__traceback__)))
    if where == 'thread':
        t.join()
    out.append('unit %s %s' % (u.status, u.reason))
    wpid, st = os.waitpid(-1, 0)
    out.append('child %d %s' % (wpid, 'exit %d' % os.WEXITSTATUS(st) if os.WIFEXITED(st) else 'signal %d' % os.WTERMSIG(st)))
    out.append('default after Terminated %s' % (signal.getsignal(signal.SIGTERM) == signal.SIG_DFL))
    shutil.rmtree(d)
    print('\n'.join(out), flush=True)
    os.kill(os.getpid(), signal.SIGTERM)          # a second SIGTERM: the default action, at once
    time_left = 50
    while time_left:
        time_left -= 1
        threading.Event().wait(0.1)
    print('survived a second SIGTERM', flush=True)
'''

MASKS = r'''
import logging, os, signal, sys, threading
sys.path.insert(0, sys.argv[1])
inside = sys.argv[2] == 'inside'
r, w = os.pipe()
tl = threading.local()
seen = {}


def blocked():
    return signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK, [])


def obs_before():                      # registered after logging's, before the CLI's: runs after the CLI's before hook
    seen[tl.tag] = blocked()


def obs_child():                       # in the child, before the CLI's reset
    os.write(w, ('window %s %s\n' % (tl.tag, blocked())).encode())


os.register_at_fork(before=obs_before, after_in_child=obs_child)
from exopipe.phase3 import cli
lines = []


def probes(kind):
    for pre in (False, True):
        tl.tag = '%s-%s' % (kind, 'blocked' if pre else 'open')
        if pre:
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        pid = os.fork()
        if pid == 0:
            os.write(w, ('body %s %s\n' % (tl.tag, blocked())).encode())
            os._exit(0)
        os.waitpid(pid, 0)
        lines.append('parent %s hook %s after %s' % (tl.tag, seen.pop(tl.tag), blocked()))
        if pre:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})


ctx = cli._sigterm_interrupts('photometry')
ctx.__enter__()
if not inside:
    ctx.__exit__(None, None, None)     # the hooks registered, the CLI's handler gone again
probes('main')
t = threading.Thread(target=probes, args=('thread',))
t.start()
t.join()
if inside:
    ctx.__exit__(None, None, None)
os.close(w)
data = b''
while True:
    chunk = os.read(r, 65536)
    if not chunk:
        break
    data += chunk
print(data.decode() + '\n'.join(lines), flush=True)
'''

TLS = r'''
import logging, os, signal, sys, threading
sys.path.insert(0, sys.argv[1])
a_blocked = sys.argv[2] == 'A-blocked'
r, w = os.pipe()
tl = threading.local()
go_b, b_done = threading.Event(), threading.Event()


def blocked():
    return signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK, [])


def obs_before():                      # after the CLI's before hook: thread A waits there while thread B forks from start to end
    if getattr(tl, 'tag', None) == 'A':
        go_b.set()
        b_done.wait(60)


os.register_at_fork(before=obs_before)
from exopipe.phase3 import cli
lines = []


def forker(tag, pre, wait):
    tl.tag = tag
    if pre:
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    if wait is not None:
        wait.wait(60)
    pid = os.fork()
    if pid == 0:
        os.write(w, ('body %s %s\n' % (tag, blocked())).encode())
        os._exit(0)
    os.waitpid(pid, 0)
    lines.append('parent %s before %s after %s' % (tag, pre, blocked()))
    if tag == 'B':
        b_done.set()


with cli._sigterm_interrupts('photometry'):
    ta = threading.Thread(target=forker, args=('A', a_blocked, None))
    tb = threading.Thread(target=forker, args=('B', not a_blocked, go_b))
    tb.start()
    ta.start()
    ta.join()
    tb.join()
os.close(w)
data = b''
while True:
    chunk = os.read(r, 65536)
    if not chunk:
        break
    data += chunk
print(data.decode() + '\n'.join(lines), flush=True)
'''

FOREIGN = r'''
import os, signal, sys, time
sys.path.insert(0, sys.argv[1])
mode = sys.argv[2]


def in_window():                       # in the child, before the CLI's reset
    if mode == 'window':
        os.kill(os.getpid(), signal.SIGTERM)


os.register_at_fork(after_in_child=in_window)
from exopipe.phase3 import cli
with cli._sigterm_interrupts('photometry'):
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(3))      # a foreign handler over the CLI's
    pid = os.fork()
    if pid == 0:
        try:
            if mode == 'body':
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(5)
            os._exit(7)
        except SystemExit as e:
            os._exit(e.code)
    _, st = os.waitpid(pid, 0)
print('child signal=%s exit=%s' % (os.WTERMSIG(st) if os.WIFSIGNALED(st) else None,
                                   os.WEXITSTATUS(st) if os.WIFEXITED(st) else None), flush=True)
'''

ORPHAN = r'''
import os, signal, subprocess, sys, time, traceback
sys.path.insert(0, sys.argv[1])
from exopipe.phase3 import cli
tmp = sys.argv[2]
step = os.getpid()


def pending(pid):                      # SIGTERM pending in the process or in its main thread: not yet caught there
    bits = 0
    for path in ('/proc/%d/status' % pid, '/proc/%d/task/%d/status' % (pid, pid)):
        with open(path) as f:
            for ln in f:
                if ln.startswith(('ShdPnd:', 'SigPnd:')):
                    bits |= int(ln.split()[1], 16)
    return bits >> (signal.SIGTERM - 1) & 1


def pre():                             # in the child, between its fork and its exec
    os.kill(step, signal.SIGTERM)
    t = time.monotonic() + 30
    while pending(step) and time.monotonic() < t:
        time.sleep(0.001)


with cli._sigterm_interrupts('photometry'):
    try:
        subprocess.run(['bash', '-c', 'echo $$ > %s/pid; exec sleep 60' % tmp], preexec_fn=pre, stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print('not interrupted', flush=True)
    except cli.Terminated as e:
        print('terminated in ' + ' > '.join(f.name for f in traceback.extract_tb(e.__traceback__)), flush=True)
print('step done', flush=True)
'''

REAP = r'''
import os, signal, subprocess, sys, time
sys.path.insert(0, sys.argv[1])
from exopipe.phase3 import cli
tmp = sys.argv[2]
cli._CHILD_TERM_S = 1.0
go = os.path.join(tmp, 'go')
lost = []
real_kill = os.kill


def kill(pid, sig):                    # the first SIGTERM sent to the 'lost' child: it dropped it; now it may exec
    real_kill(pid, sig)
    if lost and pid == lost[0] and sig == signal.SIGTERM and not os.path.exists(go):
        open(go, 'w').close()


os.kill = kill
dn = subprocess.DEVNULL
with cli._sigterm_interrupts('photometry'):
    a = subprocess.Popen(['bash', '-c', 'trap "" TERM; echo $$ > %s/a; while [ ! -e %s ]; do sleep 0.01; done; trap - TERM; '
                          'exec sleep 60' % (tmp, go)], stdin=dn, stdout=dn, stderr=dn)
    lost.append(a.pid)
    b = subprocess.Popen(['sleep', '60'], preexec_fn=lambda: signal.signal(signal.SIGTERM, signal.SIG_IGN), stdin=dn,
                         stdout=dn, stderr=dn)
    c = subprocess.Popen(['sleep', '30'], start_new_session=True, stdin=dn, stdout=dn, stderr=dn)
    e = subprocess.Popen(['sleep', '60'], preexec_fn=os.setpgrp, stdin=dn, stdout=dn, stderr=dn)   # its own process group
    g = subprocess.Popen(['bash', '-c', 'sleep 60 & echo $! > %s/g.tmp; mv %s/g.tmp %s/g; wait' % (tmp, tmp, tmp)], stdin=dn,
                         stdout=dn, stderr=dn)
    while not (os.path.exists(os.path.join(tmp, 'a')) and os.path.exists(os.path.join(tmp, 'g'))):
        time.sleep(0.01)
    with open(os.path.join(tmp, 'g')) as f:
        orphan = int(f.read())
    g.kill()                           # as subprocess.run does to its script when interrupted: the script's child lives on
    g.wait()
    print('pids lost %d ignores %d detached %d group %d orphan %d' % (a.pid, b.pid, c.pid, e.pid, orphan), flush=True)
    t0 = time.monotonic()
    try:
        real_kill(os.getpid(), signal.SIGTERM)
        time.sleep(30)
        print('not interrupted', flush=True)
    except cli.Terminated:
        pass
print('ended in %.2f s' % (time.monotonic() - t0), flush=True)
'''

ROLLBACK = r'''
import os, signal, subprocess, sys, time
sys.path.insert(0, sys.argv[1])
from exopipe.phase3 import cli
from exopipe.phase3.config import CFG
from exopipe.phase3.core import record
tmp = sys.argv[2]
part = os.path.join(tmp, 'partial')
dn = subprocess.DEVNULL
with cli._sigterm_interrupts('photometry'):
    run = record.Run('t7s', ['x'], os.path.join(tmp, 'out'), CFG, config_sections=())
    w = None
    try:
        with run.unit('u') as u:
            w = subprocess.Popen(['bash', '-c', 'while :; do echo x >> %s; sleep 0.01; done' % part], stdin=dn, stdout=dn,
                                 stderr=dn)   # a child that keeps writing the unit's partial product
            while not os.path.exists(part):
                time.sleep(0.01)
            try:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(30)
                print('not interrupted', flush=True)
            finally:                   # the rollback: the partial product is removed
                if os.path.exists(part):
                    os.remove(part)
                print('rollback done', flush=True)
    except cli.Terminated:
        print('terminated', flush=True)
    time.sleep(0.5)                    # a writer still running writes the file again within 10 ms
    print('partial product after the rollback %s' % os.path.exists(part), flush=True)
    print('unit %s %s' % (u.status, u.reason), flush=True)
    print('writer %s' % w.poll(), flush=True)
'''

POOL = r'''
import multiprocessing, os, signal, subprocess, sys, time
sys.path.insert(0, sys.argv[1])
from exopipe.phase3 import cli
tmp = sys.argv[2]
mark = os.path.join(tmp, 'grandchild')


def task():                            # a pool worker's own child, then wait for it
    p = subprocess.Popen(['sleep', '60'], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(mark + '.tmp', 'w') as f:
        f.write(str(p.pid))
    os.replace(mark + '.tmp', mark)
    return p.wait()


t0 = None
with cli._sigterm_interrupts('photometry'):
    try:
        with multiprocessing.get_context('fork').Pool(3) as pool:   # one worker busy, two idle: one waits for a task
            r = pool.apply_async(task)                                # holding the task queue's lock
            while not os.path.exists(mark):
                time.sleep(0.01)
            with open(mark) as f:
                gc = int(f.read())
            print('workers %s grandchild %d' % (' '.join(str(p.pid) for p in pool._pool), gc), flush=True)
            t0 = time.monotonic()
            os.kill(os.getpid(), signal.SIGTERM)
            r.get(60)
            print('not interrupted', flush=True)
    except cli.Terminated:
        print('terminated; the pool exited %.1f s after the SIGTERM' % (time.monotonic() - t0), flush=True)
print('step done', flush=True)
'''

IDENTITY = r'''
import os, signal, subprocess, sys
sys.path.insert(0, sys.argv[1])
from exopipe.phase3 import cli
p = subprocess.Popen(['sleep', '60'], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
st = cli._proc_stat(p.pid)
while st[4] != 'sleep':                # until it has exec'd
    st = cli._proc_stat(p.pid)
sid, exe = os.getsid(0), cli._exe(p.pid)
print('child %d start %d exe %s' % (p.pid, st[3], exe), flush=True)
termed, killed = {}, set()
cli._signal_checked('photometry', p.pid, signal.SIGTERM, [st[3] + 1, 'sleep', exe], sid, termed, killed)
print('another start time: alive %s' % (p.poll() is None), flush=True)
cli._signal_checked('photometry', p.pid, signal.SIGTERM, [st[3], 'sleep', exe], sid + 1, termed, killed)
print('another session: alive %s' % (p.poll() is None), flush=True)
rec = [st[3], 'bash', '/usr/bin/bash']
cli._signal_checked('photometry', p.pid, signal.SIGTERM, rec, sid, termed, killed)
print('exec since the look: rc %s recorded %s %s' % (p.wait(), rec[1], rec[2] == exe), flush=True)
'''

WRAPPER = r'''
import os, subprocess, sys
dn = subprocess.DEVNULL
mon = subprocess.Popen(['sleep', '60'], stdin=dn, stdout=dn, stderr=dn)   # in the step's session, not its descendant
print('monitor %d' % mon.pid, flush=True)
rc = subprocess.call([sys.executable, '-c', os.environ['STEP_CODE']] + sys.argv[1:])
print('step rc %d; monitor alive after the step %s' % (rc, mon.poll() is None), flush=True)
mon.kill()
mon.wait()
'''

FORK_LINE = PARENT.split('\n').index('    pid = os.fork()') + 1      # line numbers of the snippet run with -c
CLI_FRAMES = {'_fork_in_step', '_forked', 'pthread_sigmask', '_raise_deferred', '_on_sigterm'}   # the wrapper's raise
HOOKS = {'send', 'fork_child', '_block_sigterm_for_fork', '_restore_mask_after_fork', '_reset_inherited_sigterm',
         '_acquireLock', '_releaseLock', '_after_at_fork_child_reinit_locks'}
FAILED = []


def check(label, ok, detail=''):
    print('%s %s%s' % ('ok  ' if ok else 'FAIL', label, '' if ok else '  [%s]' % str(detail)[:800]))
    if not ok:
        FAILED.append(label)


def session_processes(sid):
    """[(pid, name)] of the live processes in session ``sid`` (/proc/<pid>/stat)."""
    out = []
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            with open('/proc/%s/stat' % d, 'rb') as f:
                st = f.read()
        except OSError:
            continue
        x = st[st.rindex(b')') + 2:].split()
        if int(x[3]) == sid and x[0] not in (b'Z', b'X'):
            out.append((int(d), st[st.index(b'(') + 1:st.rindex(b')')].decode('utf-8', 'replace')))
    return out


def run(code, *args, timeout=120, env=None, left=None, wrapped=False):
    """(returncode or 'timeout', stdout, stderr) of ``code`` in a fresh interpreter, in a session of its own. Once it has
    exited, the processes still in that session (its own and nothing else) are appended to ``left`` ([(pid, name)]) and
    killed (SIGKILL, by pid; also on a timeout), so none outlives the test. ``wrapped``: the session's leader is WRAPPER,
    which starts a monitor process and then ``code`` as its child (so the step leads no session)."""
    import signal
    env = dict(os.environ, NUMPY_MADVISE_HUGEPAGE='0', **(env or {}))
    if wrapped:
        code, env['STEP_CODE'] = WRAPPER, code
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
    rest = session_processes(p.pid)
    for pid, _ in rest:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if left is not None:
        left.extend(rest)
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
          'thread and all %d from another thread end by signal 15, none completes its own code' % (n, n),
          rc == 0 and all(x in so for x in want) and 'race condition' not in se and 'Terminated' not in se,
          (rc, so, se[-400:]))
    assert not FAILED, FAILED


def test_sigterm_to_step_in_fork_window():
    import re
    one = dict(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for reg, phase in (('first', 'before'), ('first', 'after_in_parent'), ('last', 'before'), ('last', 'after_in_parent')):
        for where, helper, threads in (('main', 'none', 1), ('main', 'helper', 2), ('thread', 'none', 1)):
            rc, so, se = run(PARENT, phase, where, helper, reg, env=one)
            ln = so.split('\n')
            term = [x[len('terminated in '):].split(' > ') for x in ln if x.startswith('terminated in ')]
            names = [f.split(':')[0] for f in term[0]] if len(term) == 1 else []
            if where == 'main':     # <module> > fork_child at its os.fork() line > the wrapper > the handler's raise
                at = names[:2] == ['<module>', 'fork_child'] and term[0][1] == 'fork_child:%d' % FORK_LINE \
                    and names[-1] == '_on_sigterm' and set(names[2:]) <= CLI_FRAMES
            else:                   # the main thread's own code, wherever it was: no at-fork hook, not the forking thread
                at = names[:1] == ['<module>'] and names[-1:] == ['_on_sigterm'] and not set(names) & HOOKS
            child = re.search(r'^child (\d+) (exit 7|signal 15)$', so, re.M)
            child_ok = child is not None and (child.group(2) == 'exit 7' or 'SIGTERM to %s (' % child.group(1) in se)
            check('a SIGTERM to the step in its fork window (a %s hook registered %s the CLI\'s; fork from %s, %d '
                  'thread(s) when the step starts) raises Terminated in the step\'s own code (%s), once, after the fork; '
                  'the unit is recorded interrupted; the child exits 7 by itself or is ended by the step (signal 15, its '
                  'SIGTERM logged); a second SIGTERM ends the step at once'
                  % (phase, 'before' if reg == 'first' else 'after', 'the main thread' if where == 'main' else
                     'another thread', threads, 'at the os.fork() line' if where == 'main' else 'in the main thread'),
                  rc == -15 and 'threads %d' % threads in ln and at and 'not interrupted' not in ln
                  and 'unit error interrupted' in ln and child_ok and 'default after Terminated True' in ln
                  and 'survived a second SIGTERM' not in so and se.count('interrupted as for Ctrl-C') == 1
                  and 'Exception ignored' not in se, (rc, so, se[-600:]))
    assert not FAILED, FAILED


def test_masks_kept():
    for where in ('inside', 'outside'):
        rc, so, se = run(MASKS, where)
        ln = set(so.split('\n'))
        want = set()
        for kind in ('main', 'thread'):
            for pre in (False, True):
                tag = '%s-%s' % (kind, 'blocked' if pre else 'open')
                hook = True if where == 'inside' else pre
                want |= {'parent %s hook %s after %s' % (tag, hook, pre), 'window %s %s' % (tag, hook),
                         'body %s %s' % (tag, pre)}
        check('masks, %s the step: a thread that had SIGTERM blocked before the fork has it blocked after it, in the parent '
              'and in the child; one that had it unblocked has it unblocked (main thread and another thread); %s'
              % (where, 'SIGTERM blocked in the fork window' if where == 'inside' else 'the hooks change nothing'),
              rc == 0 and want <= ln and 'Exception ignored' not in se, (rc, sorted(want - ln), so, se[-400:]))
    for order in ('A-open', 'A-blocked'):
        rc, so, se = run(TLS, order)
        ln = set(so.split('\n'))
        a = order == 'A-blocked'
        want = {'parent A before %s after %s' % (a, a), 'parent B before %s after %s' % (not a, not a),
                'body A %s' % a, 'body B %s' % (not a)}
        check('the saved mask is the forking thread\'s own: thread A (SIGTERM %s) waits in its before hooks while thread B '
              '(SIGTERM %s) forks from start to end; each keeps its own mask, in the parent and in the child'
              % (('blocked', 'unblocked') if a else ('unblocked', 'blocked')),
              rc == 0 and want <= ln and 'Exception ignored' not in se, (rc, sorted(want - ln), so, se[-400:]))
    assert not FAILED, FAILED


def test_foreign_handler_in_child():
    rc, so, se = run(FOREIGN, 'window')
    check('a foreign Python handler (sys.exit(3)) and a SIGTERM pending in the child at the end of the fork window: the child '
          'ends by signal 15; the handler does not run inside the hook', rc == 0 and 'child signal=15 exit=None' in so
          and 'Exception ignored' not in se, (rc, so, se[-400:]))
    rc, so, se = run(FOREIGN, 'body')
    check('the foreign handler is set again in the child: a SIGTERM its own code receives later runs it (exit 3)',
          rc == 0 and 'child signal=None exit=3' in so and 'Exception ignored' not in se, (rc, so, se[-400:]))
    assert not FAILED, FAILED


def test_subprocess_child_of_interrupted_step():
    import re
    import shutil
    import signal
    import tempfile
    tmp = tempfile.mkdtemp(prefix='test_cli_sigterm_fork_window_')
    try:
        left = []
        rc, so, se = run(ORPHAN, tmp, timeout=120, left=left)
        term = [x for x in so.split('\n') if x.startswith('terminated in ')]
        check('Terminated raised inside Popen\'s start-up (the child, before its exec, has the step take a SIGTERM): the '
              'child is ended before the step\'s rollbacks; nothing of the step\'s session outlives the step',
              rc == 0 and len(term) == 1 and '_execute_child' in term[0] and 'step done' in so and not left
              and "the step's descendants, ended before the rollbacks" in se, (rc, left, so, se[-600:]))
        for wrapped in (False, True):
            for f in ('a', 'g', 'go'):
                if os.path.exists(os.path.join(tmp, f)):
                    os.remove(os.path.join(tmp, f))
            left = []
            rc, so, se = run(REAP, tmp, timeout=120, left=left, wrapped=wrapped)
            ln = so.split('\n')
            pids = [x.split() for x in ln if x.startswith('pids ')]
            ended = [x for x in se.split('\n') if "the step's descendants, ended before the rollbacks" in x]
            ok = rc == 0 and len(pids) == 1 and len(ended) == 1 and not left and 'not interrupted' not in so
            mon = [x.split()[1] for x in ln if x.startswith('monitor ')]
            if wrapped:                 # the monitor, in the step's session but not its descendant, is never signaled
                ok = ok and 'step rc 0; monitor alive after the step True' in ln and len(mon) == 1 \
                    and ' %s (' % mon[0] not in se
            detached_alive = False
            if len(pids) == 1:
                a, b, c, e, g = (int(pids[0][i]) for i in (2, 4, 6, 8, 10))
                named = dict(re.findall(r'\b(\d+) \(([^()]*)\)', ended[0])) if len(ended) == 1 else {}
                ok = ok and named.get(str(a)) == 'sleep: signal 15' and named.get(str(b)) == 'sleep: signal 9' \
                    and str(c) not in named and named.get(str(e)) == 'sleep: signal 15' \
                    and named.get(str(g)) == 'sleep: signal 15' \
                    and re.search(r'SIGTERM to %d \(sleep, [^)]*\): .*exec since the look \(was bash, ' % a, se) is not None
                try:
                    with open('/proc/%d/stat' % c, 'rb') as f:
                        st = f.read()
                    x = st[st.rindex(b')') + 2:].split()
                    detached_alive = x[0] not in (b'Z', b'X') and int(x[3]) == c and b'(sleep)' in st
                except OSError:
                    pass
                if detached_alive:      # the test's own detached child (its session, its name checked above)
                    os.kill(c, signal.SIGKILL)
            check('the interrupted step (%s) ends its descendants before its rollbacks: a child that dropped the first '
                  'SIGTERM and then exec\'d is sent SIGTERM again (signal 15); one that ignores SIGTERM ends by SIGKILL '
                  'after the grace period (signal 9); one in a process group of its own ends (signal 15); a script\'s '
                  'child orphaned when the script was killed is re-parented to the step and ended (signal 15); one with '
                  'a session of its own is left running; %snothing else of the step\'s session outlives the step'
                  % ('a child of a wrapper that leads the session' if wrapped else 'leading its session',
                     'the wrapper\'s monitor, not a descendant, is never signaled; ' if wrapped else ''),
                  ok and detached_alive, (rc, left, so, se[-1200:]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert not FAILED, FAILED


def test_rollback_after_descendants_end():
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix='test_cli_sigterm_fork_window_')
    try:
        left = []
        rc, so, se = run(ROLLBACK, tmp, timeout=120, left=left)
        ln = so.split('\n')
        check('the order at a SIGTERM: the step\'s children end, then the rollbacks run, then the unit is recorded '
              'interrupted - a child that keeps writing the partial product the rollback removes has ended before it '
              '(signal 15), and the file is still absent 0.5 s after the rollback',
              rc == 0 and 'terminated' in ln and 'rollback done' in ln and 'not interrupted' not in ln
              and 'partial product after the rollback False' in ln and 'unit error interrupted' in ln
              and 'writer -15' in ln and not left
              and se.find("the step's descendants, ended before the rollbacks") > se.find('interrupted as for Ctrl-C') >= 0,
              (rc, left, so, se[-800:]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert not FAILED, FAILED


def test_pool_workers_left_to_their_pool():
    import re
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix='test_cli_sigterm_fork_window_')
    try:
        left = []
        rc, so, se = run(POOL, tmp, timeout=120, left=left)
        ln = so.split('\n')
        w = [x.split() for x in ln if x.startswith('workers ')]
        workers, gc = (w[0][1:4], w[0][5]) if len(w) == 1 and len(w[0]) == 6 else ([], None)
        took = [float(m) for m in re.findall(r'^terminated; the pool exited ([\d.]+) s after the SIGTERM$', so, re.M)]
        pool_line = re.search(r'Pool workers left to their pool: ([\d, ]+)', se)
        check('a step interrupted inside `with Pool(3)` (one worker busy with a child of its own, two idle): the worker\'s '
              'child is ended (SIGTERM, logged); the workers are never signaled by the step but left to their pool, whose '
              'exit ends them (an idle worker ended from outside can hold the task queue\'s lock and hang that exit); the '
              'step exits; nothing of its session outlives it',
              rc == 0 and len(workers) == 3 and gc is not None and 'SIGTERM to %s (sleep, ' % gc in se
              and not any(' to %s (' % p in se for p in workers) and pool_line is not None
              and sorted(pool_line.group(1).replace(' ', '').split(',')) == sorted(workers)
              and len(took) == 1 and took[0] < 10 and 'step done' in ln and 'not interrupted' not in ln and not left,
              (rc, left, so, se[-800:]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert not FAILED, FAILED


def test_signal_identity_check():
    rc, so, se = run(IDENTITY)
    ln = so.split('\n')
    c = [x.split() for x in ln if x.startswith('child ')]
    pid = c[0][1] if len(c) == 1 else None
    check('each signal is checked first: not sent to a pid whose start time differs from the look\'s (a pid taken again) '
          'or that is in another session (each withheld signal logged); sent to the same process after an exec (changed '
          'name and executable, same start time), the exec recorded and logged',
          rc == 0 and pid is not None and 'another start time: alive True' in ln and 'another session: alive True' in ln
          and 'exec since the look: rc -15 recorded sleep True' in ln
          and se.count('SIGTERM not sent to %s (sleep)' % pid) == 2
          and 'exec since the look (was bash, /usr/bin/bash)' in se and se.count('SIGTERM to %s (' % pid) == 1,
          (rc, so, se[-800:]))
    assert not FAILED, FAILED


if __name__ == '__main__':
    for t in (test_window_closed, test_sigterm_in_window, test_sigterm_at_fork_return, test_sigterm_to_step_in_fork_window,
              test_masks_kept, test_foreign_handler_in_child, test_subprocess_child_of_interrupted_step,
              test_rollback_after_descendants_end, test_pool_workers_left_to_their_pool, test_signal_identity_check):
        try:
            t()
        except AssertionError:
            pass
    print('%s: %d failed' % (os.path.basename(__file__), len(FAILED)))
    sys.exit(1 if FAILED else 0)
