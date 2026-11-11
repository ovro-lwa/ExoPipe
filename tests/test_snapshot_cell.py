#!/usr/bin/env python3
"""The per-cell driver's plan (exopipe/phase3/lib/snapshot_engine/cell.py): the ORDERING RULES a cell's products depend on.

What a real run proves on data is in exopipe_ops/refactor_20261012/plan/STAGE_MAP_20261111.md; these data-free tests guard the
rules themselves:
  1. the task graph: a subband's search comes after its flow; its light curves after its search (they read the search's
     admission table) and after its flow fields are in the flow tree; Stokes V light curves after Stokes I (V carries the I
     flux scale); its cubes and flow movie after its flow; the completeness guard after EVERY search; the wideband search after
     the guard (it reads every member's admission table); the frame movies after EVERY flow;
  2. flows are dealt to the GPUs in turn, high band first, and need one slot of their GPU;
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


def _graph(subbands=None, gpus=('0', '1'), wideband=True, frame_movies=True, cubes=True):
    t = C.build_tasks(subbands or C.ALL_SUBBANDS, list(gpus), wideband, frame_movies, cubes, THREADS)
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


def test_flows_go_to_the_gpus_in_turn_high_band_first():
    tasks, _, needs = _graph()
    flows = [n for n, _, _ in tasks if n.startswith('flow:')]
    freqs = [int(n.split(':')[1][:-3]) for n in flows]
    assert freqs == sorted(freqs, reverse=True), 'flows are not planned high band first'
    gpus = [[k for k in needs[n] if k.startswith('gpu')] for n in flows]
    assert all(len(g) == 1 for g in gpus) and [g[0] for g in gpus] == ['gpu0', 'gpu1'] * 7 + ['gpu0']
    assert all(needs[n][g[0]] == 1 for n, g in zip(flows, gpus))
    one = _graph(gpus=('3',))[2]
    assert all('gpu3' in one[n] for n in one if n.startswith('flow:'))


def test_switches():
    names = {n for n, _, _ in _graph(wideband=False, frame_movies=False, cubes=False)[0]}
    assert not any(n.startswith(('wideband', 'framemovies', 'cube:')) for n in names) and 'subbands' in names
    red = ['18MHz', '23MHz', '27MHz', '32MHz', '36MHz']
    _, after, _ = _graph(subbands=red)
    assert after['subbands'] == {'search:' + sb for sb in red} and after['framemovies'] == {'flow:' + sb for sb in red}


def test_the_scheduler_runs_the_plan_in_a_valid_order():
    tasks, after, _ = _graph()
    budgets = {'cores': 26, 'mem_gb': 250, 'gpu0': 2, 'gpu1': 2}
    s = S.Scheduler(budgets)
    lock, start, end = threading.Lock(), {}, {}

    def stand_in(name):
        def run():
            with lock:
                start[name] = time.time()
            time.sleep(0.03 if name.startswith('flow:') else 0.005)
            with lock:
                end[name] = time.time()
        return run
    for name, a, needs in tasks:
        s.add(name, stand_in(name), after=a, needs=needs)
    box = {}
    th = threading.Thread(target=lambda: box.setdefault('out', s.run()))
    th.daemon = True
    th.start()
    th.join(60)
    assert not th.is_alive(), 'the plan did not finish within 60 s (a stall)'
    out = box['out']
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


TESTS = [test_the_ordering_rules, test_flows_go_to_the_gpus_in_turn_high_band_first, test_switches,
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
