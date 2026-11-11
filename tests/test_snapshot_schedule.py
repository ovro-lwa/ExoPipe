#!/usr/bin/env python3
"""The cell scheduler (exopipe/phase3/lib/snapshot_engine/schedule.py): the rules its docstring promises, on fake tasks.

  1. prerequisites and budgets: on random task graphs every task runs once, never before its prerequisites have finished, and
     the resources in use never exceed any budget;
  2. the cell's own shape: a subband's search starts only after its flow, its light curves only after its search, the wideband
     search only after every member's search, the frame movies only after every flow; at most N flows per GPU;
  3. a failure: the dependents are skipped with the cause named; with stop_on_failure nothing new starts; without it the
     independent tasks still run; every task has an outcome (nothing silently missing);
  4. a task that does not fit does not block a later, smaller one - until it has been passed over max_passes times;
  5. add() refuses a need without a budget, a need larger than the budget, a duplicate name and an unknown prerequisite;
  6. a non-zero or False return value and an exception (SystemExit included) are failures; None and 0 are not.
Data-free; standard library only. Run: python tests/test_snapshot_schedule.py   (or under pytest)
"""
import os
import random
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'snapshot_engine'))

_IMPORT_ERROR = None
try:
    import schedule as S
except Exception as e:                                    # pragma: no cover
    _IMPORT_ERROR = f'{e.__class__.__name__}: {e}'


class _World:
    """Fake tasks that record when they ran and check the budgets while they run."""

    def __init__(self, budgets):
        self.budgets, self.used, self.lock = dict(budgets), {k: 0 for k in budgets}, threading.Lock()
        self.start, self.end, self.over = {}, {}, []

    def task(self, name, needs, seconds=0.01, result=None, raises=None):
        def run():
            with self.lock:
                self.start[name] = time.time()
                for k, v in needs.items():
                    self.used[k] += v
                    if self.used[k] > self.budgets[k]:
                        self.over.append((name, k, self.used[k]))
            time.sleep(seconds)
            with self.lock:
                for k, v in needs.items():
                    self.used[k] -= v
                self.end[name] = time.time()
            if raises is not None:
                raise raises
            return result
        return run


def _run(s, deadline=60):
    """_run(s) with a deadline, so a stalled scheduler FAILS the test instead of hanging the job (2026-11-11: the first version
    looped without releasing its lock after a failure, and the running tasks could never report back)."""
    box = {}
    th = threading.Thread(target=lambda: box.setdefault('out', s.run()))
    th.daemon = True
    th.start()
    th.join(deadline)
    assert not th.is_alive(), 'the scheduler did not finish within %d s (a stall)' % deadline
    return box['out']


def test_prerequisites_and_budgets_on_random_graphs():
    for seed in range(6):
        rng = random.Random(seed)
        budgets = {'cores': 8, 'mem': 20, 'gpu': 2}
        w, s, deps = _World(budgets), S.Scheduler(budgets), {}
        names = []
        for i in range(40):
            name = 't%d' % i
            after = rng.sample(names, k=min(len(names), rng.choice((0, 0, 1, 1, 2, 3))))
            needs = {'cores': rng.choice((1, 1, 2, 4, 8))}
            if rng.random() < 0.5:
                needs['mem'] = rng.choice((1, 5, 10, 20))
            if rng.random() < 0.3:
                needs['gpu'] = 1
            s.add(name, w.task(name, needs, seconds=rng.choice((0.0, 0.005, 0.02))), after=after, needs=needs)
            names.append(name); deps[name] = after
        out = _run(s)
        assert list(out) == names, 'the result is not one entry per task in the order added'
        assert all(o.status == 'done' for o in out.values()), [(n, o) for n, o in out.items() if o.status != 'done'][:3]
        assert not w.over, f'a budget was exceeded (seed {seed}): {w.over[:3]}'
        for n in names:
            for d in deps[n]:
                assert w.start[n] >= w.end[d], f'{n} started before its prerequisite {d} had finished (seed {seed})'


def test_the_shape_of_a_cell():
    budgets = {'cores': 26, 'mem': 250, 'gpu0': 2, 'gpu1': 2}
    w, s = _World(budgets), S.Scheduler(budgets)
    sbs = ['%dMHz' % f for f in (82, 78, 73, 69, 64, 59, 55, 50, 46, 41, 36, 32, 27, 23, 18)]
    for k, sb in enumerate(sbs):
        g = 'gpu%d' % (k % 2)
        s.add('flow:' + sb, w.task('flow:' + sb, {g: 1, 'cores': 1}, 0.03), needs={g: 1, 'cores': 1})
        s.add('search:' + sb, w.task('search:' + sb, {'cores': 8, 'mem': 21}, 0.02), after=['flow:' + sb], needs={'cores': 8, 'mem': 21})
        for st in 'IV':
            s.add('curves:%s:%s' % (sb, st), w.task('curves:%s:%s' % (sb, st), {'cores': 4}, 0.01), after=['search:' + sb], needs={'cores': 4})
            s.add('cube:%s:%s' % (sb, st), w.task('cube:%s:%s' % (sb, st), {'cores': 1, 'mem': 70}, 0.02), after=['flow:' + sb], needs={'cores': 1, 'mem': 70})
    for st in 'VI':
        s.add('wideband:' + st, w.task('wideband:' + st, {'cores': 12, 'mem': 27}, 0.02), after=['search:' + sb for sb in sbs], needs={'cores': 12, 'mem': 27})
    s.add('movies', w.task('movies', {'cores': 8}, 0.02), after=['flow:' + sb for sb in sbs], needs={'cores': 8})
    out = _run(s)
    assert all(o.status == 'done' for o in out.values()) and not w.over, (w.over[:3], [n for n, o in out.items() if o.status != 'done'][:3])
    for sb in sbs:
        assert w.start['search:' + sb] >= w.end['flow:' + sb]
        assert w.start['curves:%s:I' % sb] >= w.end['search:' + sb] and w.start['cube:%s:V' % sb] >= w.end['flow:' + sb]
    assert w.start['wideband:V'] >= max(w.end['search:' + sb] for sb in sbs)
    assert w.start['movies'] >= max(w.end['flow:' + sb] for sb in sbs)
    # the point of the scheduler: the searches overlap the flows (the first search starts before the last flow ends)
    assert min(w.start['search:' + sb] for sb in sbs) < max(w.end['flow:' + sb] for sb in sbs), 'no search overlapped the flow'


def _failing(stop):
    budgets = {'cores': 4}
    w, s = _World(budgets), S.Scheduler(budgets, stop_on_failure=stop)
    s.add('a', w.task('a', {'cores': 1}, 0.05), needs={'cores': 1})
    s.add('bad', w.task('bad', {'cores': 1}, 0.01, raises=RuntimeError('REFUSED: no Master')), needs={'cores': 1})
    s.add('child', w.task('child', {'cores': 1}), after=['bad'], needs={'cores': 1})
    s.add('grandchild', w.task('grandchild', {'cores': 1}), after=['child'], needs={'cores': 1})
    s.add('late', w.task('late', {'cores': 4}, 0.01), after=['a'], needs={'cores': 4})
    return w, _run(s)


def test_a_failure_skips_its_dependents_and_names_the_cause():
    w, out = _failing(stop=True)
    assert set(out) == {'a', 'bad', 'child', 'grandchild', 'late'}, 'a task has no outcome'
    assert out['bad'].status == 'failed' and 'REFUSED: no Master' in out['bad'].detail
    assert out['child'].status == 'skipped' and 'bad' in out['child'].detail
    assert out['grandchild'].status == 'skipped' and 'child' in out['grandchild'].detail
    assert out['a'].status == 'done', 'a task that was already running did not finish'
    assert out['late'].status == 'skipped' and 'late' not in w.start, 'a new task started after the cell had failed'
    w, out = _failing(stop=False)
    assert out['late'].status == 'done' and out['child'].status == 'skipped' and out['grandchild'].status == 'skipped'


def test_a_task_that_does_not_fit_does_not_block_a_smaller_one_until_it_starves():
    budgets = {'cores': 4}
    w, s = _World(budgets), S.Scheduler(budgets)
    s.add('long', w.task('long', {'cores': 2}, 0.25), needs={'cores': 2})
    s.add('big', w.task('big', {'cores': 4}, 0.01), needs={'cores': 4})
    s.add('small', w.task('small', {'cores': 1}, 0.01), needs={'cores': 1})
    out = _run(s)
    assert all(o.status == 'done' for o in out.values())
    assert w.start['small'] < w.end['long'] <= w.start['big'], 'the small task waited behind the one that did not fit'
    # starvation guard: a chain of small tasks must not keep the big one out to the very end
    assert S.Scheduler(budgets).max_passes > 3                        # the default guard is looser than this test's
    w, s = _World(budgets), S.Scheduler(budgets, max_passes=3)
    s.add('first', w.task('first', {'cores': 1}, 0.05), needs={'cores': 1})
    s.add('big', w.task('big', {'cores': 4}, 0.01), needs={'cores': 4})
    prev = 'first'
    for i in range(30):
        s.add('small%d' % i, w.task('small%d' % i, {'cores': 1}, 0.02), after=[prev], needs={'cores': 1})
        prev = 'small%d' % i
    out = _run(s)
    assert all(o.status == 'done' for o in out.values())
    assert w.start['big'] < w.start['small29'], 'the big task was starved to the very end by the chain of small ones'


def test_add_refuses_what_could_never_run():
    s = S.Scheduler({'cores': 4})
    s.add('a', lambda: None, needs={'cores': 1})
    for kwargs, what in ((dict(name='b', needs={'gpu': 1}), 'no budget'), (dict(name='b', needs={'cores': 5}), 'never start'),
                         (dict(name='a'), 'twice'), (dict(name='b', after=['zzz']), 'not added yet')):
        try:
            s.add(kwargs['name'], lambda: None, after=kwargs.get('after', ()), needs=kwargs.get('needs'))
        except ValueError as e:
            assert what in str(e), (what, str(e))
        else:
            raise AssertionError(f'add() accepted a task that should be refused ({what})')
    try:
        S.Scheduler({'cores': 0})
    except ValueError:
        pass
    else:
        raise AssertionError('a zero budget was accepted')


def test_what_counts_as_a_failure():
    def exits():
        raise SystemExit('REFUSED: flow incomplete')
    s = S.Scheduler({'cores': 8}, stop_on_failure=False)
    for name, fn in (('none', lambda: None), ('zero', lambda: 0), ('true', lambda: True), ('one', lambda: 1), ('false', lambda: False),
                     ('exit', exits), ('raise', lambda: 1 / 0)):
        s.add(name, fn, needs={'cores': 1})
    out = _run(s)
    assert [out[n].status for n in ('none', 'zero', 'true')] == ['done'] * 3
    assert [out[n].status for n in ('one', 'false', 'exit', 'raise')] == ['failed'] * 4
    assert 'REFUSED: flow incomplete' in out['exit'].detail and 'ZeroDivisionError' in out['raise'].detail


TESTS = [test_prerequisites_and_budgets_on_random_graphs, test_the_shape_of_a_cell,
         test_a_failure_skips_its_dependents_and_names_the_cause,
         test_a_task_that_does_not_fit_does_not_block_a_smaller_one_until_it_starves,
         test_add_refuses_what_could_never_run, test_what_counts_as_a_failure]


def main():
    if _IMPORT_ERROR is not None:
        print(f'UNTESTED-IMPORT: {_IMPORT_ERROR}')
        return 1
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f'PASS  {t.__name__}')
        except Exception as e:
            failed += 1
            print(f'FAIL  {t.__name__}: {e.__class__.__name__}: {str(e)[:400]}')
    print(f'{len(TESTS) - failed}/{len(TESTS)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
