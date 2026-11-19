#!/usr/bin/env python3
"""`exopipe3 run update` (PR-8; plan/specs/hybrid.md section 3.7): a partial update of the hybrid route as one plan of runner jobs.

Data-free (CI):
  1. the closure of a trigger: a subband cell brings its wideband and Full cells, a wideband or Full cell only itself; ordered
     subbands, widebands, Full; a bad or empty trigger refuses;
  2. the items of every item kind for the 11-13 lost-sky trigger (13h_Green, 13h_Full, 15h_Green): 3 hybrid cells, v3 and v5
     scopes Green and Full, 48 tile cells (96 tilefill units with the two weightings);
  3. runner.UPDATE_ORDER keeps the order of hybrid.md 2.16 item 9 (masters, dewarp, install, v3 mosaics + tables + push, the
     hybrids by scope class, v5 mosaics to conform, v5 tables, tilefill, push hybrid, push mosaic_v5);
  4. every step command of the plan, item substituted (runner._subst), parses with that step's own CLI parser; a chain's steps
     parse with the forwarded --resume; every non-push step carries --resume; a push carries --apply only from --apply;
  5. runner._subst: {lst} / {scope} from an <LST>_<scope> item, no --cells appended; a non-cell item refuses;
  6. tables_touched: run-wide tables and the updated scopes' tables, never another scope's or a carried table;
  7. the CLI: `run update` without --trigger or a work run refuses; --plan writes plan.json / plan.csv and submits nothing.
Cluster only (the installed archive present; else UNTESTED-CLUSTER, not a failure):
  8. the 11-13 replay: every file of the 11-13 v5 push manifest and every replaced file of the 11-13 hybrid manifest is in the
     plan's predicted push files; the hybrid extras are only the rebuilt units' NoTaper and diffuse-PB files.
Run: python tests/test_update_plan.py   (or under pytest)
"""
import argparse
import csv
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from exopipe.phase3.config import CFG                     # noqa: E402
from exopipe.phase3.core import runner                    # noqa: E402
from exopipe.phase3.lib.update import plan as UP          # noqa: E402

TRIG_1113 = ['13h_Green', '13h_Full', '15h_Green']
OUTS = dict(hybrid_out='/x/H', v3_out='/x/M3', v5_out='/x/M5')
MANIFESTS = '/lustre/gh/main/exopipe_ops/hybrid_highband_20261031/'


def _raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except (ValueError, SystemExit) as e:
        return str(e)
    raise AssertionError(f'{fn.__name__}{a} did not refuse')


def test_closure():
    C = CFG.hybrid
    sub = next(s for s in C.SCOPES if UP.scope_class(s) == 'sub')
    wbs = [wb for wb, mem in C.WIDEBAND_MEMBERS.items() if sub in mem]
    got = [f'{l}_{s}' for l, s in UP.closure([f'02h_{sub}'])]
    assert got == [f'02h_{sub}'] + [f'02h_{w}' for w in wbs] + ['02h_Full'], got
    assert [f'{l}_{s}' for l, s in UP.closure(TRIG_1113)] == ['13h_Green', '15h_Green', '13h_Full']
    assert UP.closure(['05h_Full']) == [('05h', 'Full')]
    for bad in ([], ['13h_Blu'], ['25h_Green'], ['13h-Green']):
        _raises(UP.closure, bad)


def test_items_1113():
    cells = UP.closure(TRIG_1113)
    it = {k: UP.items_for(k, cells) for k in UP.KINDS}
    assert it['cells'] == ['13h_Green', '15h_Green', '13h_Full'], it['cells']
    assert it['cells:sub'] == [] and it['cells:wb'] == ['13h_Green', '15h_Green'] and it['cells:full'] == ['13h_Full']
    assert it['lsts'] == ['13h', '15h']
    assert set(it['v3']) == {'Green', 'Full'} and set(it['v5']) == {'Green', 'Full'}, (it['v3'], it['v5'])
    assert len(it['tiles']) == 48 and len(set(it['tiles'])) == 48 and len(CFG.hybrid.WEIGHTINGS) * 48 == 96
    assert it['deep'] == ['deep']
    _raises(UP.items_for, 'cellz', cells)


def test_order():
    names = [e[0] for e in CFG.runner.UPDATE_ORDER]
    want = ['masters', 'dewarp', 'install', 'mosaic_v3', 'mosaic_v3_tables', 'push_mosaic_v3', 'hybrid_sub', 'hybrid_wb',
            'hybrid_full', 'mosaic_v5', 'mosaic_v5_tables', 'tilefill', 'push_hybrid', 'push_mosaic_v5']
    assert names == want, names
    assert all(k in UP.KINDS for _n, k, _c in CFG.runner.UPDATE_ORDER)


def _parser(step):
    from exopipe.phase3 import cli
    mod = importlib.import_module(f'exopipe.phase3.steps.{step}')
    sp = argparse.ArgumentParser(prog=f'exopipe3 {step}')
    mod.add_arguments(sp)
    cli._common(sp, mod)
    return sp


def _parse(argv):
    step, rest = argv[0], argv[1:]
    try:
        return _parser(step).parse_args(rest)
    except SystemExit:
        raise AssertionError(f'exopipe3 {" ".join(argv)}: the step parser refuses it')


def test_commands_parse():
    for apply in (False, True):
        pl = UP.build(TRIG_1113, OUTS, apply=apply)
        assert [e['name'] for e in pl['entries'] if e.get('skipped')] == ['hybrid_sub']
        for e in pl['entries']:
            for item in e['items'] or ['13h_Green']:
                if e['chain']:
                    assert e['forward'] == ['--resume']
                    for st in runner.chain_steps(e['chain']):
                        a = runner._subst(st.split() if isinstance(st, str) else list(st), item) + e['forward']
                        _parse(a)
                    continue
                a = runner._subst(e['argv'], item)
                ns = _parse(a)
                push = a[:2] == ['archive', 'push']
                assert ('--resume' in a) != push, a
                assert ('--apply' in a) == (push and apply), a
                assert not any('{' in x for x in a), a
                if e['kind'] in ('cells:wb', 'cells:full', 'tiles'):
                    lst, sc = item.split('_', 1)
                    assert a[a.index('--lsts') + 1] == lst and a[a.index('--scopes') + 1] == sc and '--cells' not in a, a
                if e['name'] in ('mosaic_v5',):
                    assert getattr(ns, 'until_stage', None) == 'conform', a


def test_subst():
    out = runner._subst(['hybrid', '--lsts', '{lst}', '--scopes', '{scope}'], '07h_Red')
    assert out == ['hybrid', '--lsts', '07h', '--scopes', 'Red'], out
    assert runner._subst(['install', '--tier', 'deep'], '07h_Red')[-2:] == ['--cells', '07h_Red']
    assert runner._subst(['x', '--scopes', '{item}'], 'Green') == ['x', '--scopes', 'Green']
    _raises(runner._subst, ['hybrid', '--lsts', '{lst}'], 'Green')


def test_tables_touched():
    d = tempfile.mkdtemp(prefix='upd_tables_', dir=os.environ.get('TMPDIR'))
    try:
        names = ['summary.csv', 'pkcorr_curves.csv', 'caveats_Taper_R0_Full_lsfill.csv', 'caveats_Taper_R0_Blue_lsfill.csv',
                 'summary_tpfill_Taper_R0.csv', 'taper_only.csv'] + list(CFG.mosaic_v3.TABLES_CARRIED)
        for n in names:
            open(os.path.join(d, n), 'w').close()
        got = {os.path.basename(p) for p in UP.tables_touched(d, 'mosaic_v5', ['Green', 'Full'])}
        assert 'caveats_Taper_R0_Full_lsfill.csv' in got and 'caveats_Taper_R0_Blue_lsfill.csv' not in got, got
        assert {'summary.csv', 'pkcorr_curves.csv', 'summary_tpfill_Taper_R0.csv', 'taper_only.csv'} <= got, got
        got3 = {os.path.basename(p) for p in UP.tables_touched(d, 'mosaic_v3', ['Green'])}
        assert not got3 & set(CFG.mosaic_v3.TABLES_CARRIED), got3
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _cli(args):
    env = dict(os.environ, PYTHONPATH=ROOT)
    return subprocess.run([sys.executable, '-m', 'exopipe.phase3', 'run', 'update'] + args, cwd=ROOT, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)


def test_cli():
    r = _cli(['--hybrid-out', '/x/H', '--v3-out', '/x/M3', '--v5-out', '/x/M5'])
    assert r.returncode != 0 and '--trigger' in r.stdout, r.stdout[-400:]
    r = _cli(['--trigger', '13h_Green', '--hybrid-out', '/x/H', '--v3-out', '/x/M3'])
    assert r.returncode != 0 and '--v5-out' in r.stdout, r.stdout[-400:]
    r = _cli(['--trigger', '13h_Grn', '--hybrid-out', '/x/H', '--v3-out', '/x/M3', '--v5-out', '/x/M5'])
    assert r.returncode != 0 and 'refused' in r.stdout, r.stdout[-400:]
    r = _cli(['--trigger', '13h_Green', '--hybrid-out', '/x/H', '--v3-out', '/x/M3', '--v5-out', '/x/M5'])
    assert r.returncode != 0 and '--nodes' in r.stdout, r.stdout[-400:]
    d = tempfile.mkdtemp(prefix='upd_plan_', dir=os.environ.get('TMPDIR'))
    try:
        jobs_before = sorted(os.listdir(runner.jobs_root())) if os.path.isdir(runner.jobs_root()) else []
        r = _cli(['--trigger', ','.join(TRIG_1113), '--hybrid-out', '/x/H', '--v3-out', '/x/M3', '--v5-out', '/x/M5', '--plan', d])
        assert r.returncode == 0, r.stdout[-600:]
        pl = json.load(open(os.path.join(d, 'plan.json')))
        assert pl['cells'] == ['13h_Green', '15h_Green', '13h_Full'] and not pl['apply']
        rows = list(csv.reader(open(os.path.join(d, 'plan.csv'))))
        assert rows[0] == ['order', 'entry', 'item', 'command', 'kind', 'action', 'dst'] and any(x[1] == 'tilefill' for x in rows[1:])
        jobs_after = sorted(os.listdir(runner.jobs_root())) if os.path.isdir(runner.jobs_root()) else []
        assert jobs_after == jobs_before, 'run update --plan submitted a job'
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_replay_1113_cluster():
    H = MANIFESTS + 'MANIFEST_LOSTSKY_HYBRID_TILES_20261113T191100Z.csv'
    V = MANIFESTS + 'MANIFEST_MOSAIC_V5_PUSH_LOSTSKY_20261113.csv'
    if not (os.path.isfile(H) and os.path.isfile(V) and os.path.isdir(CFG.hybrid.ARCHIVE_HYBRID)):
        print('UNTESTED-CLUSTER  test_replay_1113_cluster (manifests or the installed archive not here)')
        return
    pl = UP.build(TRIG_1113, OUTS)
    pred = {}
    for r in UP.predicted(pl):
        pred.setdefault(r['entry'], {})[r['dst']] = (r['unit'], r['kind'])
    hm = {r['src'] for r in csv.DictReader(open(H)) if r['kind'] == 'superseded'}
    ph = pred['push_hybrid']
    assert hm and hm <= set(ph), sorted(hm - set(ph))[:5]
    extra = {(ph[p][1], ph[p][0].rsplit('_', 1)[0]) for p in set(ph) - hm}
    assert {k for k, _c in extra} <= {'recon', 'recon_pbcorr', 'pbdiffuse'}, extra
    assert {c for _k, c in extra} <= {'13h_Green', '15h_Green', '13h_Full'}, extra
    vm = {r['dst'] for r in csv.DictReader(open(V))}
    pv = pred['push_mosaic_v5']
    assert vm and vm <= set(pv), sorted(vm - set(pv))[:5]
    assert {os.path.basename(p) for p in set(pv) - vm} <= {'taper_only.csv'}, sorted(set(pv) - vm)


TESTS = [test_closure, test_items_1113, test_order, test_commands_parse, test_subst, test_tables_touched, test_cli,
         test_replay_1113_cluster]


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f'PASS  {t.__name__}')
        except Exception as e:
            failed += 1
            print(f'FAIL  {t.__name__}: {e.__class__.__name__}: {str(e)[:600]}')
    print(f'{len(TESTS) - failed}/{len(TESTS)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
