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
  6. the driver refuses an --out inside a production tree and an unknown subband.
Data-free; standard library only. Run: python tests/test_snapshot_cell.py   (or under pytest)
"""
import os
import sys
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
    seen, lock = [], threading.Lock()

    def fake_call(argv, logfile, env=None, **kw):                # stands in for the engine: records the device, takes a while
        with lock:
            seen.append((env['CUDA_VISIBLE_DEVICES'], dict(c._gpu_load)))
        time.sleep(0.05)
        return 0
    c.call, c.staged = fake_call, lambda sb: []
    ths = [threading.Thread(target=c.t_flow, args=(sb,)) for sb in ('82MHz', '78MHz', '73MHz', '69MHz')]
    for t in ths:
        t.start()
    for t in ths:
        t.join(30)
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
    for argv, what in ((['--cell', '04h_2024-12-19', '--out', '/lustre/pipeline/exopipe/phase3/x'], 'production tree'),
                       (['--cell', '04h_2024-12-19', '--out', '/lustre/gh/main/exopipe_ops/snapshot_fleet_out'], 'production tree'),
                       (['--cell', '04h_2024-12-19', '--out', '/tmp/x', '--subbands', '18MHz 99MHz'], 'unknown subband')):
        try:
            C.Cell(C.parse_args(argv))
        except SystemExit as e:
            assert 'REFUSED' in str(e) and what in str(e), (what, str(e))
        else:
            raise AssertionError(f'the driver accepted {argv} ({what})')
    c = C.Cell(C.parse_args(['--cell', '04h_2024-12-19', '--out', '/tmp/x', '--subbands', '18MHz 23MHz']))
    assert c.wideband is False and c.frame_movies is False, 'a subset must skip the wideband search and the frame movies unless asked'
    c = C.Cell(C.parse_args(['--cell', '04h_2024-12-19', '--out', '/tmp/x']))
    assert c.wideband is True and c.frame_movies is True and c.out == '/tmp/x/work/04h_2024-12-19'


TESTS = [test_the_ordering_rules, test_priority_the_flows_first_and_never_waiting_for_cpu_work,
         test_a_flow_takes_the_device_with_a_free_slot, test_switches,
         test_the_scheduler_runs_the_plan_in_a_valid_order, test_what_fails_the_cell, test_refusals_before_any_work]


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
