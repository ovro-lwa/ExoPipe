#!/usr/bin/env python3
"""The per-cell driver's plan (exopipe/phase3/lib/snapshot_engine/cell.py): the ORDERING RULES a cell's products depend on.

What a real run proves on data is in exopipe_ops/refactor_20261012/plan/STAGE_MAP_20261111.md; these data-free tests guard the
rules themselves:
  1. the task graph: a subband's search comes after its flow; its light curves after its search (they read the search's
     admission table) and after its flow fields are in the flow tree; Stokes V light curves after Stokes I (V carries the I
     flux scale); its cubes and flow movie after its flow; the completeness guard after EVERY search; the wideband search after
     the guard (it reads every member's admission table); the frame movies after EVERY flow;
  2. priority: the flows are planned first, high band first, each needing one slot of the pooled GPU budget; then the searches,
     the guard and the wideband search; then the rest. The tasks of that chain hold what they wait for. With stand-in tasks
     and the cores saturated, every flow starts when a GPU slot is released (a GPU slot never waits for CPU work);
  3. subsets and switches: no wideband, no frame movies, no cubes when switched off;
  4. the graph is accepted by the scheduler under the default budgets and runs to the end with stand-in tasks, in an order
     that respects every rule of 1;
  5. what fails the cell and what is only recorded (as the fleet): flows, searches and the two guards are fatal; movies, light
     curves, cubes and the field copies are not;
  6. the driver refuses an --out or --stage-root that is, is inside, or contains a production tree (through a symbolic link
     too), an unknown subband, and a run without a GPU or with a count below 1;
  7. run() always leaves a record and never a half-shipped cell: a failed fatal task, an exception while shipping, and a
     SIGTERM each end with the status recorded, the ship trees moved aside, the staging removed, and no engine process left;
  8. only the subbands with staged frames are planned.
Data-free; standard library only. Run: python tests/test_snapshot_cell.py   (or under pytest)
"""
import json
import os
import shutil
import signal
import sys
import tempfile
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'snapshot_engine'))

_IMPORT_ERROR = None
try:
    import cell as C
    import schedule as S
except Exception as e:                                    # pragma: no cover
    _IMPORT_ERROR = f'{e.__class__.__name__}: {e}'

THREADS = dict(search=8, curves=4, wideband=12, movies=8)


def _graph(subbands=None, wideband=True, frame_movies=True, cubes=True):
    t = C.build_tasks(subbands or C.ALL_SUBBANDS, wideband, frame_movies, cubes, THREADS)
    return t, {name: set(after) for name, after, _ in t}, {name: needs for name, _, needs in t}


def test_the_ordering_rules():
    tasks, after, needs = _graph()
    names = [n for n, _, _ in tasks]
    assert len(names) == len(set(names)), 'a task is planned twice'
    for sb in C.ALL_SUBBANDS:
        assert after['search:' + sb] == {'flow:' + sb}
        assert after['curves:%s:I' % sb] == {'search:' + sb, 'fields:' + sb}, 'Stokes I light curves must wait for the search AND the flow fields'
        assert after['curves:%s:V' % sb] == {'curves:%s:I' % sb}, 'Stokes V light curves must wait for the Stokes I table'
        assert after['fields:' + sb] == {'flow:' + sb} and after['flowmovie:' + sb] == {'flow:' + sb}
        for st in 'IV':
            assert after['cube:%s:%s' % (sb, st)] == {'flow:' + sb}
        assert after['flow:' + sb] == set()
    every_search = {'search:' + sb for sb in C.ALL_SUBBANDS}
    assert after['subbands'] == every_search, 'the completeness guard must wait for EVERY subband search'
    assert after['wideband:V'] == {'subbands'} and after['wideband:I'] == {'subbands'}, 'the wideband search reads every admission table'
    assert after['widebands'] == {'wideband:V', 'wideband:I'}
    assert after['framemovies'] == {'flow:' + sb for sb in C.ALL_SUBBANDS}, 'the frame movies need every flowed subband'
    # prerequisites are always planned before their dependents (the scheduler requires it)
    seen = set()
    for n, a, _ in tasks:
        assert set(a) <= seen, f'{n} is planned before its prerequisite(s) {set(a) - seen}'
        seen.add(n)


def _stand_in_run(tasks, budgets, seconds):
    """Run a plan with stand-in tasks; returns (outcomes, start, end)."""
    s = S.Scheduler(budgets)
    lock, start, end = threading.Lock(), {}, {}

    def stand_in(name):
        def run():
            with lock:
                start[name] = time.time()
            time.sleep(seconds(name))
            with lock:
                end[name] = time.time()
        return run
    for name, a, needs in tasks:
        s.add(name, stand_in(name), after=a, needs=needs, hold=name.startswith(C.HOLD_PREFIXES))
    box = {}
    th = threading.Thread(target=lambda: box.setdefault('out', s.run()))
    th.daemon = True
    th.start()
    th.join(120)
    assert not th.is_alive(), 'the plan did not finish within 120 s (a stall)'
    return box['out'], start, end


def test_priority_the_flows_first_and_never_waiting_for_cpu_work():
    tasks, _, needs = _graph()
    names = [n for n, _, _ in tasks]
    flows = [n for n in names if n.startswith('flow:')]
    freqs = [int(n.split(':')[1][:-3]) for n in flows]
    assert freqs == sorted(freqs, reverse=True), 'flows are not planned high band first'
    assert names[:len(flows)] == flows, 'something is planned before a flow: it could keep a GPU slot idle'
    assert all(needs[n] == {'gpu': 1, 'cores': 1} for n in flows)
    last_critical = max(i for i, n in enumerate(names) if n.startswith(('search:', 'subbands', 'wideband')))
    first_rest = min(i for i, n in enumerate(names) if n.startswith(('framemovies', 'curves:', 'cube:', 'flowmovie:')))
    assert last_critical < first_rest, 'a light curve, cube or movie is planned before a search or the wideband search'
    assert {n.split(':')[0] for n in names if n.startswith(C.HOLD_PREFIXES)} == {'flow', 'search', 'subbands', 'wideband', 'widebands'}
    # with the cores saturated by the work of the earlier subbands, the k-th flow still starts when the (k - slots)-th ends
    slots, longest = 4, 0.12

    def seconds(name):
        kind = name.split(':')[0]
        return {'flow': 0.15, 'search': 0.08, 'cube': longest, 'curves': 0.05, 'wideband': 0.08, 'framemovies': 0.08}.get(kind, 0.01)
    out, start, end = _stand_in_run(tasks, {'cores': 12, 'mem_gb': 250, 'gpu': slots}, seconds)
    assert all(o.status == 'done' for o in out.values())
    ended = sorted(end[n] for n in flows)
    late = [(n, round(start[n] - ended[k - slots], 3)) for k, n in enumerate(flows) if k >= slots and start[n] - ended[k - slots] > longest + 0.25]
    assert not late, f'flows waited for CPU work after a GPU slot was released: {late[:4]}'
    running_at_once = max(sum(1 for m in flows if start[m] <= start[n] < end[m]) for n in flows)
    assert running_at_once == slots, f'{running_at_once} flows ran at once, {slots} GPU slots'


def test_a_flow_takes_the_device_with_a_free_slot():
    c = C.Cell(C.parse_args(['--cell', '04h_2024-12-19', '--out', '/tmp/x', '--gpus', '0 1']))
    seen, lock, all_in = [], threading.Lock(), threading.Barrier(4)

    def fake_call(argv, logfile, env=None, **kw):                # stands in for the engine: all four are in flight together
        with lock:
            seen.append((env['CUDA_VISIBLE_DEVICES'], dict(c._gpu_load)))
        all_in.wait(30)
        return 0
    c.call, c.staged = fake_call, lambda sb: []
    ths = [threading.Thread(target=c.t_flow, args=(sb,)) for sb in ('82MHz', '78MHz', '73MHz', '69MHz')]
    for t in ths:
        t.start()
    for t in ths:
        t.join(60)
    assert sorted(g for g, _ in seen) == ['0', '0', '1', '1'], f'four flows on two devices were not two per device: {seen}'
    assert all(max(load.values()) <= 2 for _, load in seen) and c._gpu_load == {'0': 0, '1': 0}


def test_switches():
    names = {n for n, _, _ in _graph(wideband=False, frame_movies=False, cubes=False)[0]}
    assert not any(n.startswith(('wideband', 'framemovies', 'cube:')) for n in names) and 'subbands' in names
    red = ['18MHz', '23MHz', '27MHz', '32MHz', '36MHz']
    _, after, _ = _graph(subbands=red)
    assert after['subbands'] == {'search:' + sb for sb in red} and after['framemovies'] == {'flow:' + sb for sb in red}


def test_the_scheduler_runs_the_plan_in_a_valid_order():
    tasks, after, _ = _graph()
    out, start, end = _stand_in_run(tasks, {'cores': 26, 'mem_gb': 250, 'gpu': 4}, lambda name: 0.03 if name.startswith('flow:') else 0.005)
    assert all(o.status == 'done' for o in out.values()), [(n, o) for n, o in out.items() if o.status != 'done'][:3]
    for n, a in after.items():
        for d in a:
            assert start[n] >= end[d], f'{n} started before {d} had finished'
    assert min(start[n] for n in start if n.startswith('search:')) < max(end[n] for n in end if n.startswith('flow:')), \
        'no search overlapped the flows: the plan is not pipelined'


def test_what_fails_the_cell():
    names = [n for n, _, _ in _graph()[0]]
    fatal = {n for n in names if n.startswith(C.FATAL_PREFIXES)}
    assert {n.split(':')[0] for n in fatal} == {'flow', 'search', 'subbands', 'wideband', 'widebands'}
    assert not any(n.startswith(('flowmovie:', 'framemovies', 'curves:', 'cube:', 'fields:')) for n in fatal), \
        'a product the fleet only logs (movies, light curves, cubes) would fail the cell'


def test_refusals_before_any_work():
    tmp = tempfile.mkdtemp(prefix='cell_refusals_')
    try:
        link = os.path.join(tmp, 'looks_private')
        os.symlink('/lustre/gh/main/exopipe_ops/snapshot_fleet_out', link)
        cell = ['--cell', '04h_2024-12-19']
        for argv, what in ((cell + ['--out', '/lustre/pipeline/exopipe/phase3/x'], 'production tree'),
                           (cell + ['--out', '/lustre/gh/main/exopipe_ops/snapshot_fleet_out'], 'production tree'),
                           (cell + ['--out', '/lustre/gh/main/exopipe_ops'], 'production tree'),         # its ship trees would be production's
                           (cell + ['--out', '/lustre/gh/main'], 'production tree'),                     # a parent of them
                           (cell + ['--out', '/fast/gh/main/phase3/x'], 'production tree'),              # the fleet's node-local trees
                           (cell + ['--out', link], 'production tree'),                                  # through a symbolic link
                           (cell + ['--out', '/tmp/x', '--stage-root', '/fast/gh/main/phase3/snapshot_stage'], '--stage-root'),
                           (cell + ['--out', '/tmp/x', '--subbands', '18MHz 99MHz'], 'unknown subband'),
                           (cell + ['--out', '/tmp/x', '--gpus', ''], 'no GPU'),
                           (cell + ['--out', '/tmp/x', '--flow-per-gpu', '0'], 'below 1')):
            try:
                C.Cell(C.parse_args(argv))
            except SystemExit as e:
                assert 'REFUSED' in str(e) and what in str(e), (what, str(e))
            else:
                raise AssertionError(f'the driver accepted {argv} ({what})')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    c = C.Cell(C.parse_args(['--cell', '04h_2024-12-19', '--out', '/tmp/x', '--subbands', '18MHz 23MHz']))
    assert c.wideband is False and c.frame_movies is False, 'a subset must skip the wideband search and the frame movies unless asked'
    c = C.Cell(C.parse_args(['--cell', '04h_2024-12-19', '--out', '/tmp/x']))
    assert c.wideband is True and c.frame_movies is True and c.out == '/tmp/x/work/04h_2024-12-19'
    assert not os.path.exists('/lustre/gh/main/exopipe_refactor_x') and C._refuse_protected('/lustre/gh/main/exopipe_refactor_x', '--out') is None, \
        'a sibling whose name only begins like a production tree was refused'


def _stub_cell(tmp, tasks, ship=None):
    """A Cell under tmp whose staging, plan and ship are stand-ins: run() itself is what is tested."""
    c = C.Cell(C.parse_args(['--cell', '04h_2024-12-19', '--out', os.path.join(tmp, 'out'), '--subbands', '18MHz']))

    def prepare():
        c._own_stage = True
        for d in (c.stage, c.out, os.path.join(c.flowtree, c.cell)):      # the flow fields are in the flow tree during a run
            os.makedirs(d)

    def scheduler():
        s = S.Scheduler({'cores': 4})
        for name, fn in tasks:
            s.add(name, fn, needs={'cores': 1})
        return s, {}
    c.prepare, c.scheduler, c.ship_all = prepare, scheduler, (ship or (lambda outcomes: 'shipped'))
    return c


def _record(c):
    return json.load(open(os.path.join(c.D, 'cell_%s.json' % c.cell)))


def test_run_always_leaves_a_record_and_never_a_half_shipped_cell():
    tmp = tempfile.mkdtemp(prefix='cell_run_')
    try:
        # (a) everything done
        c = _stub_cell(os.path.join(tmp, 'a'), [('flow:18MHz', lambda: None), ('search:18MHz', lambda: None)])
        assert c.run() == 0 and _record(c)['status'] == 'done' and not os.path.exists(c.stage)
        assert os.path.isdir(os.path.join(c.flowtree, c.cell))
        # (b) a fatal task fails: nothing stays in a ship tree under its own name
        def bad():
            raise RuntimeError('FLOW-INCOMPLETE stand-in')
        c = _stub_cell(os.path.join(tmp, 'b'), [('flow:18MHz', bad)])
        assert c.run() == 1
        r = _record(c)
        assert r['status'] == 'failed' and r['steps']['flow:18MHz']['status'] == 'failed' and not os.path.exists(c.stage)
        assert not os.path.exists(os.path.join(c.flowtree, c.cell)), 'the flow fields of a failed cell stayed in the flow tree'
        assert [d for d in os.listdir(c.flowtree) if d.startswith(c.cell + '.failed_')], 'the failed cell\'s flow fields were not kept aside'
        assert 'CELL-FAILED' in open(c.logpath).read()
        # (c) an exception while shipping (a full disk): recorded, and the half-filled ship directory is moved aside
        def ship(outcomes):
            os.makedirs(c.ship)
            open(os.path.join(c.ship, 'candidates_18MHz_I_median.csv'), 'w').close()
            raise OSError(28, 'No space left on device')
        c = _stub_cell(os.path.join(tmp, 'c'), [('flow:18MHz', lambda: None)], ship=ship)
        assert c.run() == 1
        r = _record(c)
        assert r['status'] == 'failed' and 'No space left' in r['error'] and not os.path.exists(c.ship) and not os.path.exists(c.stage)
        assert [d for d in os.listdir(os.path.dirname(c.ship)) if d.startswith(c.cell + '.failed_')]
        # (d) SIGTERM while an engine process runs: the process is stopped, the record says interrupted
        if threading.current_thread() is threading.main_thread():
            pids = []

            def engine():
                return c.call(['sleep', '120'], os.path.join(c.out, 'engine.log'), pin=False)

            def killer():
                t_end = time.time() + 30
                while not c._children and time.time() < t_end:
                    time.sleep(0.02)
                pids.extend(c._children)
                os.kill(os.getpid(), signal.SIGTERM)
            c = _stub_cell(os.path.join(tmp, 'd'), [('flow:18MHz', engine), ('search:18MHz', killer)])
            before = signal.getsignal(signal.SIGTERM)
            assert c.run() == 130
            assert signal.getsignal(signal.SIGTERM) is before, 'run() did not restore the signal handler'
            assert _record(c)['status'] == 'interrupted' and not os.path.exists(c.stage)
            assert pids, 'the stand-in engine never started'
            t_end = time.time() + 10
            while time.time() < t_end and any(os.path.exists('/proc/%d' % p) for p in pids):
                time.sleep(0.1)
            assert not any(os.path.exists('/proc/%d' % p) for p in pids), 'an engine process survived the interruption'
        # (e) a refusal before any work (an earlier run's products under the same --out): recorded, and that run's staging is NOT removed
        c = _stub_cell(os.path.join(tmp, 'e'), [])
        del c.prepare                                               # the real prepare()
        os.makedirs(c.ship)
        os.makedirs(c.stage)
        assert c.run() == 2
        r = _record(c)
        assert r['status'] == 'refused' and 'earlier run' in r['refusal'] and os.path.isdir(c.stage) and os.path.isdir(c.ship)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_only_the_staged_subbands_are_planned():
    c = C.Cell(C.parse_args(['--cell', '04h_2024-12-19', '--out', '/tmp/x', '--subbands', '18MHz 23MHz 27MHz', '--wideband', 'yes']))
    c.cores_budget, c.mem_budget_gb, c.active = 26, 250, ['18MHz', '27MHz']       # 23 MHz has no frame this hour
    s, _ = c.scheduler()
    names = list(s.tasks)
    assert not any('23MHz' in n for n in names) and 'search:18MHz' in names and 'cube:27MHz:V' in names
    assert s.tasks['subbands'].after == ('search:27MHz', 'search:18MHz')
    assert all(s.tasks[n].hold == n.startswith(C.HOLD_PREFIXES) for n in names)


TESTS = [test_the_ordering_rules, test_priority_the_flows_first_and_never_waiting_for_cpu_work,
         test_a_flow_takes_the_device_with_a_free_slot, test_switches,
         test_the_scheduler_runs_the_plan_in_a_valid_order, test_what_fails_the_cell, test_refusals_before_any_work,
         test_run_always_leaves_a_record_and_never_a_half_shipped_cell, test_only_the_staged_subbands_are_planned]


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
