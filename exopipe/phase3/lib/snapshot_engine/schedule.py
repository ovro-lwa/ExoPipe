"""A small scheduler for the work of ONE snapshot cell: tasks with prerequisites and resource needs, run side by side.

Why: the fleet runs a cell's stages one after another (stage, flow, search, wideband, movies, light curves, cubes) and a node
sits at a load of 5-8 on 26 free cores for 4.5 hours (exopipe_ops/refactor_20261012/plan/STAGE_MAP_20261111.md section 1). The
stages only depend on each other through a few products: a subband's search needs that subband's flowed frames; its light
curves need the search's admission table; the wideband search needs every member's admission table; the frame movies need
every flowed subband. Everything else can overlap. This module holds the ordering and the budgets; what a task DOES is the
caller's (a step command, an engine script).

    s = Scheduler({'cores': 26, 'mem_gb': 250, 'gpu': 4})
    s.add('flow:64MHz', run=..., needs={'gpu': 1, 'cores': 1}, hold=True)
    s.add('search:64MHz', run=..., after=['flow:64MHz'], needs={'cores': 8, 'mem_gb': 21}, hold=True)
    s.add('curves:64MHz:I', run=..., after=['search:64MHz'], needs={'cores': 4})
    result = s.run()          # {name: Outcome(status, seconds, started, ended, detail)}

Rules (each is tested in tests/test_snapshot_schedule.py):
- a task starts only when every task in `after` is DONE, and only when its needs fit in what is free of every budget; a need
  larger than the whole budget is refused at add() (it could never start);
- among the tasks that can start, the one added first starts first (the caller's order is the priority order), and a task
  that does not fit does not block a later, smaller one from starting (no head-of-line blocking) - with two exceptions:
  a task added with hold=True (the cell's critical path: flow, search, wideband search) HOLDS, while it waits, what is
  free of every budget it is short of, so no later task can take it and the waiting task starts as soon as enough is
  released; and any task that has been passed over more than `max_passes` times holds in the same way (no starvation of
  a large task). A hold touches only the budgets the waiting task lacks: a flow waiting for a GPU slot holds no cores;
- a task FAILS when its run() raises or returns a non-zero / False value; a failure is never retried here. Every task that
  depends on a failed or skipped task is SKIPPED (not run), and with stop_on_failure (the default: one failure fails the
  cell) no new task starts after the first failure; tasks already running finish;
- the result has one entry per task, so nothing is silently missing: done, failed, or skipped with the name of the task that
  caused it.
Standard library only; Python 3.6+.
"""
import threading
import time
from collections import OrderedDict


class Outcome(object):
    __slots__ = ('status', 'seconds', 'started', 'ended', 'detail')

    def __init__(self, status, seconds=0.0, started=None, ended=None, detail=''):
        self.status, self.seconds, self.started, self.ended, self.detail = status, seconds, started, ended, detail

    def as_dict(self):
        return OrderedDict((k, getattr(self, k)) for k in self.__slots__)

    def __repr__(self):
        return 'Outcome(%s, %.1f s%s)' % (self.status, self.seconds, ', ' + self.detail if self.detail else '')


class _Task(object):
    __slots__ = ('name', 'run', 'after', 'needs', 'hold', 'passes')

    def __init__(self, name, run, after, needs, hold):
        self.name, self.run, self.after, self.needs, self.hold, self.passes = name, run, tuple(after), dict(needs), bool(hold), 0


class Scheduler(object):
    def __init__(self, budgets, stop_on_failure=True, max_passes=50, clock=time.time):
        self.budgets = dict(budgets)
        for k, v in self.budgets.items():
            if not v > 0:
                raise ValueError('budget %r must be positive, got %r' % (k, v))
        self.stop_on_failure, self.max_passes, self.clock = stop_on_failure, max_passes, clock
        self.tasks = OrderedDict()

    def add(self, name, run, after=(), needs=None, hold=False):
        needs = dict(needs or {})
        if name in self.tasks:
            raise ValueError('task %r added twice' % name)
        for k, v in needs.items():
            if k not in self.budgets:
                raise ValueError('task %r needs %r, which has no budget (budgets: %s)' % (name, k, sorted(self.budgets)))
            if v > self.budgets[k]:
                raise ValueError('task %r needs %s of %r but the whole budget is %s - it could never start'
                                 % (name, v, k, self.budgets[k]))
            if v < 0:
                raise ValueError('task %r: negative need %r' % (name, k))
        for d in after:
            if d not in self.tasks:
                raise ValueError('task %r comes after %r, which is not added yet (add prerequisites first)' % (name, d))
        self.tasks[name] = _Task(name, run, after, needs, hold)
        return name

    def run(self):
        """Run every task; returns OrderedDict {name: Outcome} in the order the tasks were added."""
        cv = threading.Condition()
        free = dict(self.budgets)
        out, running, threads = {}, set(), []
        failed_any = [False]

        def worker(t):
            t0 = self.clock()
            status, detail = 'done', ''
            try:
                r = t.run()
                if r is False or (isinstance(r, int) and not isinstance(r, bool) and r != 0):
                    status, detail = 'failed', 'returned %r' % (r,)
            except BaseException as e:                       # a task's SystemExit is its failure, not the scheduler's
                status, detail = 'failed', '%s: %s' % (type(e).__name__, str(e)[:300])
            t1 = self.clock()
            with cv:
                out[t.name] = Outcome(status, t1 - t0, t0, t1, detail)
                running.discard(t.name)
                for k, v in t.needs.items():
                    free[k] += v
                if status == 'failed':
                    failed_any[0] = True
                cv.notify_all()

        with cv:
            pending = list(self.tasks.values())
            while pending or running:
                progressed = False
                # tasks whose prerequisite failed or was skipped are skipped, with the cause named
                for t in list(pending):
                    bad = [d for d in t.after if d in out and out[d].status != 'done']
                    if bad:
                        out[t.name] = Outcome('skipped', detail='prerequisite %s %s' % (bad[0], out[bad[0]].status))
                        pending.remove(t)
                        progressed = True
                if failed_any[0] and self.stop_on_failure:
                    if pending:
                        for t in pending:
                            out[t.name] = Outcome('skipped', detail='not started: an earlier task of the cell failed')
                        pending = []
                        progressed = True
                    # (nothing pending: this thread must WAIT below for the running tasks - they need the lock to report)
                else:
                    held = {}                                  # per budget: what an earlier, waiting task holds of the free amount
                    for t in list(pending):
                        if not all(d in out and out[d].status == 'done' for d in t.after):
                            continue
                        short = [k for k, v in t.needs.items() if free[k] - held.get(k, 0) < v]
                        if not short:
                            for k, v in t.needs.items():
                                free[k] -= v
                            pending.remove(t)
                            running.add(t.name)
                            th = threading.Thread(target=worker, args=(t,), name='task-' + t.name)
                            th.daemon = True
                            threads.append(th)
                            th.start()
                            progressed = True
                        else:
                            t.passes += 1
                            if t.hold or t.passes > self.max_passes:
                                for k in short:                # nothing later takes what this task is waiting for
                                    held[k] = free[k]
                if pending and not running and not progressed:
                    # nothing runs, nothing can start: only possible if a prerequisite never finishes - a defect, never silent
                    for t in pending:
                        out[t.name] = Outcome('skipped', detail='never became ready (scheduler defect)')
                    pending = []
                    break
                if pending or running:
                    if not progressed:
                        cv.wait(timeout=5.0)
        for th in threads:
            th.join()
        return OrderedDict((name, out[name]) for name in self.tasks)
