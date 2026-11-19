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
     without --apply the plan stops before install (the first entry writing the installed tree or the archive);
  5. runner._subst: {item_lst} / {item_scope} from an <LST>_<scope> item, no --cells appended; a non-cell item refuses; other
     brace text passes through;
  6. tables_touched: run-wide tables and the updated scopes' tables, never another scope's or a carried table;
  6b. runner after_ok: a job whose --after job ended with a failed item refuses every item without starting a worker;
  7. the CLI: `run update` without exactly one of --trigger / --from-exclusions, or without a work run, refuses; --plan writes
     plan.json / plan.csv (with the quarantine and note columns) and submits nothing;
  7b. the round: every push carries --round <the plan's round> (default today), a malformed round refuses; quarantine_dst puts
     a hybrid file under <prefix><round>/<LST>/<scope>/ and a mosaic file under <prefix><round>/<path in the run>; over an
     empty archive every planned hybrid tile unit gets one 'publish' row per filled master (plain, pbcorr), without a
     quarantine destination, its note the dry run's reason when the work run cannot name it;
  7c. from_exclusions on a synthetic masters tree: a band master holding a night a member subband's row excludes and a V master
     holding an excluded night are triggers; a subband master without the night, a band master whose members are not
     excluded, a band without the subband are not; an unreadable master refuses.
Cluster only (the installed archive present; else UNTESTED-CLUSTER, not a failure):
  8. the 11-13 replay: every file of the 11-13 v5 push manifest and every replaced file of the 11-13 hybrid manifest is in the
     plan's predicted push files; the hybrid extras are only the rebuilt units' NoTaper and diffuse-PB files; with round
     20261113 every superseded file's planned quarantine destination is the manifest's;
  9. the 11-13 trigger: data_decisions.csv as committed with the lost-sky rows (clv15 8e26b39) and the masters before the
     update (every cell whose build record lists 2025-05-07, holding its in_master nights; the three rebuilt cells, whose new
     records drop the night, with it restored) give --from-exclusions exactly 13h Green, 13h Full and 15h Green.
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
        names = [e['name'] for e in pl['entries']]
        gated = names[names.index('install'):]
        want = ['hybrid_sub'] if apply else sorted(set(['hybrid_sub'] + gated), key=names.index)
        assert [e['name'] for e in pl['entries'] if e.get('skipped')] == want, [e['name'] for e in pl['entries'] if e.get('skipped')]
        assert [e['name'] for e in pl['entries'] if e['needs_apply']] == gated
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
                assert (a[a.index('--round') + 1] == pl['round']) if push else '--round' not in a, a
                assert ('--apply' in a) == (push and apply), a
                assert not any('{' in x for x in a), a
                if e['kind'] in ('cells:wb', 'cells:full', 'tiles'):
                    lst, sc = item.split('_', 1)
                    assert a[a.index('--lsts') + 1] == lst and a[a.index('--scopes') + 1] == sc and '--cells' not in a, a
                if e['name'] in ('mosaic_v5',):
                    assert getattr(ns, 'until_stage', None) == 'conform', a


def test_subst():
    out = runner._subst(['hybrid', '--lsts', '{item_lst}', '--scopes', '{item_scope}'], '07h_Red')
    assert out == ['hybrid', '--lsts', '07h', '--scopes', 'Red'], out
    assert runner._subst(['install', '--tier', 'deep'], '07h_Red')[-2:] == ['--cells', '07h_Red']
    assert runner._subst(['x', '--scopes', '{item}'], 'Green') == ['x', '--scopes', 'Green']
    _raises(runner._subst, ['hybrid', '--lsts', '{item_lst}'], 'Green')
    out = runner._subst(['x', '--set', 'a.GLOB={scope}/I/*', '--lsts', '{item}'], '09h')       # a config glob's brace text
    assert out == ['x', '--set', 'a.GLOB={scope}/I/*', '--lsts', '09h'], out


def test_after_ok():
    d = tempfile.mkdtemp(prefix='upd_jobs_', dir=os.environ.get('TMPDIR'))
    saved = runner.jobs_root
    try:
        runner.jobs_root = lambda: d
        up = '20261119T000000Z_up'
        os.makedirs(os.path.join(d, up, 'items'))
        with open(os.path.join(d, up, 'items.txt'), 'w') as f:
            f.write('13h_Green\n13h_Full\n')
        for it, st in (('13h_Green', 'done'), ('13h_Full', 'failed')):
            runner._wjson(os.path.join(d, up, 'items', f'{runner._safe(it)}.json'), {'status': st, 'attempts': 1})
        assert runner.job_state(up)[0] == 'finished'
        job = runner.submit('down', ['install', '--tier', 'deep'], ['13h_Green'], [('nohost', 1)], after=up, start=False,
                            after_ok=True)
        assert runner.load_job(job)['after_ok'] is True
        runner.supervise(job)                                   # refuses without starting a worker (nohost would fail)
        st = runner.item_state(job, '13h_Green')
        assert st['status'] == 'refused' and up in st['reason'] and 'failed' in st['reason'], st
        assert runner.job_state(job)[0] == 'finished'
    finally:
        runner.jobs_root = saved
        shutil.rmtree(d, ignore_errors=True)


def test_tables_touched():
    d = tempfile.mkdtemp(prefix='upd_tables_', dir=os.environ.get('TMPDIR'))
    try:
        names = ['summary.csv', 'pkcorr_curves.csv', 'caveats_Taper_R0_Full_lsfill.csv', 'caveats_Taper_R0_Blue_lsfill.csv',
                 'summary_tpfill_Taper_R0.csv', 'taper_only.csv', 'Full_summary.csv'] + list(CFG.mosaic_v3.TABLES_CARRIED)
        for n in names:
            open(os.path.join(d, n), 'w').close()
        assert 'Full_summary.csv' in {os.path.basename(p) for p in UP.tables_touched(d, 'mosaic_v5', ['Green'])}
        got = {os.path.basename(p) for p in UP.tables_touched(d, 'mosaic_v5', ['Green', 'Full'])}
        assert 'caveats_Taper_R0_Full_lsfill.csv' in got and 'caveats_Taper_R0_Blue_lsfill.csv' not in got, got
        assert {'summary.csv', 'pkcorr_curves.csv', 'summary_tpfill_Taper_R0.csv', 'taper_only.csv'} <= got, got
        got3 = {os.path.basename(p) for p in UP.tables_touched(d, 'mosaic_v3', ['Green'])}
        assert not got3 & set(CFG.mosaic_v3.TABLES_CARRIED), got3
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_round():
    for bad in ('2026111', '20261399', '20261113\n'):           # short, not a date, a trailing newline
        assert _raises(UP.build, TRIG_1113, OUTS, rnd=bad).startswith('the quarantine round'), bad
    assert 'never its own' in _raises(UP.command, 'archive push --class hybrid --round 20261201', OUTS, rnd='20261119')
    pl = UP.build(TRIG_1113, OUTS, apply=True, rnd='20261113')
    assert pl['round'] == '20261113' and all(e['argv'][e['argv'].index('--round') + 1] == '20261113' for e in pl['entries']
                                             if e['argv'] and e['argv'][:2] == ['archive', 'push'])
    assert UP.build(TRIG_1113, OUTS)['round'] == __import__('time').strftime('%Y%m%d', __import__('time').gmtime())
    R = '/R'
    h = UP.quarantine_dst('hybrid', R, '20261113', f'{R}/Coadd/Run_NOTAPER_HYBRID_20261028/13h/Green/x.fits',
                          f'{R}/Coadd/Run_NOTAPER_HYBRID_20261028')
    assert h == f'{R}/{CFG.archive_push.HYBRID_QUARANTINE_PREFIX}20261113/13h/Green/x.fits', h
    run = f'{R}/mosaic/Run_HYBRID_20241218-20260419'
    v = UP.quarantine_dst('mosaic_v5', R, '20261113', f'{run}/Green/I/healpix/y.fits', run)
    assert v == f'{R}/{CFG.archive_push.MOSAIC_V5_QUARANTINE_PREFIX}20261113/Green/I/healpix/y.fits', v
    v3 = UP.quarantine_dst('mosaic_v3', R, '20261113', f'{R}/mosaic/F3/tables/t.csv', f'{R}/mosaic/F3')
    assert v3 == f'{R}/{CFG.archive_push.MOSAIC_V3_QUARANTINE_PREFIX}20261113/tables/t.csv', v3


def test_publish_rows():
    d = tempfile.mkdtemp(prefix='upd_pub_', dir=os.environ.get('TMPDIR'))
    try:
        pl = UP.build(['13h_Green'], OUTS, apply=True)
        rows = UP.predicted(pl, archive_root=d)
        assert rows and all(r['action'] == UP.PUBLISH and r['entry'] == 'push_hybrid' and not r['quarantine'] for r in rows), rows[:2]
        assert len(rows) == 24 * len(CFG.hybrid.WEIGHTINGS) * 2, len(rows)       # 24 LSTs of Green (a wideband: no Full) x w x 2
        assert {r['kind'] for r in rows} == {'plain', 'pbcorr'} and all(r['note'].startswith('named by the step') for r in rows)
        reb = [r for r in rows if r['unit'].startswith('13h_Green_')]     # rebuilt by the plan: named only after the rebuild
        assert len(reb) == len(CFG.hybrid.WEIGHTINGS) * 2 and all('after the rebuild' in r['note'] for r in reb), reb
        assert all('dry run: NoInput' in r['note'] or 'dry run: OSError' in r['note'] for r in rows if r not in reb), rows[0]
        pl['trigger_evidence'] = [dict(cell='13h_Green', master='/m/I.fits', date='2025-05-07', verdict='STALE', note='n')]
        UP.write(pl, rows, os.path.join(d, 'plan'))
        assert not any(x[1] == 'trigger' for x in csv.reader(open(os.path.join(d, 'plan', 'plan.csv'))))
        assert list(csv.DictReader(open(os.path.join(d, 'plan', 'trigger.csv')))) == pl['trigger_evidence']
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _master(path, dates):
    import numpy as np
    from astropy.io import fits
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h = fits.Header()
    for d in dates:
        h.add_history(f'COADD constituent {d}')
    fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32), header=h).writeto(path, overwrite=True)


_PATCH = (('paths', 'DATA_DECISIONS'), ('paths', 'COADD_SB_ROOT'), ('status_conformance', 'DD_HISTORY_GLOB'))
_SAVED = {}


def _restore():
    """Undo _with_tree (CFG.reset() does not restore assigned keys)."""
    from exopipe.phase3.core import exclusions as EX
    from exopipe.phase3.lib.status import steps_exclconf as XC
    for (sec, key), v in _SAVED.items():
        setattr(getattr(CFG, sec), key, v)
    EX._CACHE.clear()
    XC._SNAPS = None


def _with_tree(T, dd_rows, masters):
    """CFG patched to a synthetic masters root and data_decisions.csv; masters {(cell, 'I'|'V'): dates | None (unreadable)}."""
    from exopipe.phase3.core import exclusions as EX
    from exopipe.phase3.lib.status import steps_exclconf as XC
    for sec, key in _PATCH:
        _SAVED.setdefault((sec, key), getattr(getattr(CFG, sec), key))
    os.makedirs(T, exist_ok=True)
    dd = os.path.join(T, 'data_decisions.csv')
    with open(dd, 'w') as f:
        f.write('lst,subband,date,verdict,reason,score_thermal_noise,score_flux_scale,score_ionosphere\n')
        f.writelines(f'{l},{sb},{d},exclude,synthetic,,,\n' for l, sb, d in dd_rows)
    CFG.paths.DATA_DECISIONS = dd
    CFG.paths.COADD_SB_ROOT = os.path.join(T, 'coadd_sb')
    CFG.status_conformance.DD_HISTORY_GLOB = os.path.join(T, 'no_backup_*')
    EX._CACHE.clear()
    XC._SNAPS = None
    os.makedirs(CFG.paths.COADD_SB_ROOT, exist_ok=True)
    for (cell, st), dates in masters.items():
        pat = CFG.status_conformance.WORKING_MASTERS[0 if st == 'I' else 1]
        p = os.path.join(CFG.paths.COADD_SB_ROOT, cell, 'coadd_final', pat.format(scope=cell.split('_', 1)[1]))
        if dates is None:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, 'wb').write(b'not a FITS file')
        else:
            _master(p, dates)


def test_from_exclusions():
    T = tempfile.mkdtemp(prefix='upd_fromex_', dir=os.environ.get('TMPDIR'))
    try:
        rows = [('13h', '50MHz', '2025-05-07'), ('15h', '50MHz', '2025-05-07')]
        m = {('13h_Green', 'I'): ['2025-04-05', '2025-05-07'],       # Green holds 50 MHz: excluded -> trigger
             ('13h_Full', 'I'): ['2025-04-05'], ('13h_Full', 'V'): ['2025-05-07'],   # the V master alone holds it -> trigger
             ('13h_50MHz', 'I'): ['2025-04-05'],                     # the subband master never held it
             ('14h_Green', 'I'): ['2025-05-07'],                     # no 14h row
             ('15h_Red', 'I'): ['2025-05-07']}                       # Red does not hold 50 MHz
        _with_tree(T, rows, m)
        trig, ev, n, fp = UP.from_exclusions()
        assert trig == ['13h_Full', '13h_Green'], (trig, ev)
        assert n == 6 and fp['n_exclude_rows'] == 2, (n, fp)
        assert {(e['cell'], os.path.basename(e['master'])[0], e['date'], e['verdict']) for e in ev} == {
            ('13h_Green', 'I', '2025-05-07', 'UNKNOWN-EPOCH'), ('13h_Full', 'V', '2025-05-07', 'UNKNOWN-EPOCH')}, ev
        pl = UP.build(trig, OUTS)
        assert pl['cells'] == ['13h_Green', '13h_Full'], pl['cells']
        _with_tree(T, rows, {**m, ('16h_Blue', 'I'): None})
        msg = _raises(UP.from_exclusions)
        assert 'unreadable' in msg and '16h_Blue' in msg, msg
        card = lambda t: t.encode().ljust(80)                      # noqa: E731
        trunc = card('SIMPLE  =                    T') + card('HISTORY COADD constituent 2025-04-05') * 40   # cut in block 2
        noend = (card('SIMPLE  =                    T') + card('HISTORY no constituent') + card('END')).ljust(2880)
        for bad in (trunc, noend):              # a partial constituent list (no END); a header without a constituent line
            p = os.path.join(CFG.paths.COADD_SB_ROOT, '16h_Blue', 'coadd_final', 'I_deep_Taper_R075_Blue.fits')
            open(p, 'wb').write(bad)
            msg = _raises(UP.from_exclusions)
            assert 'unreadable' in msg and '16h_Blue' in msg, msg
        _with_tree(os.path.join(T, 'empty'), [], {})
        assert UP.from_exclusions()[:2] == ([], [])
    finally:
        _restore()
        shutil.rmtree(T, ignore_errors=True)


def test_from_exclusions_1113_cluster():
    """Item 9: the 11-13 trigger from data_decisions.csv at clv15 8e26b39 and the masters as they were before the update."""
    from exopipe.phase3.core import exclusions as EX
    real = CFG.paths.COADD_SB_ROOT
    clv = '/lustre/gh/main/clv15'
    if not (os.path.isdir(real) and os.path.isdir(os.path.join(clv, '.git'))):
        print('UNTESTED-CLUSTER  test_from_exclusions_1113_cluster (the masters root or the clv15 repository not here)')
        return
    dd = subprocess.run(['git', '-C', clv, 'show', '8e26b39:data_decisions.csv'], stdout=subprocess.PIPE, universal_newlines=True,
                        check=True, timeout=300).stdout
    T = tempfile.mkdtemp(prefix='upd_fromex1113_', dir=os.environ.get('TMPDIR'))
    try:
        held = {}
        for c in sorted(os.listdir(real)):
            w = os.path.join(real, c, 'coadd_final', 'weights.csv')
            if not os.path.isfile(w):
                continue
            rows = list(csv.DictReader(open(w)))
            if c in TRIG_1113 or any(r['date'] == '2025-05-07' for r in rows):   # a rebuild's record drops an excluded night
                held[c] = sorted({r['date'] for r in rows if r['in_master'] == '1'} | ({'2025-05-07'} if c in TRIG_1113 else set()))
        assert set(TRIG_1113) <= set(held) and len(held) > len(TRIG_1113), sorted(held)
        _with_tree(T, [], {(c, 'I'): d for c, d in held.items()})
        open(CFG.paths.DATA_DECISIONS, 'w').write(dd)
        EX._CACHE.clear()
        trig, ev, n, _fp = UP.from_exclusions()
        assert trig == sorted(TRIG_1113) and {e['date'] for e in ev} == {'2025-05-07'}, (trig, ev)
        print(f'      11-13 trigger replay: {len(held)} cells listing 2025-05-07 -> {trig}')
    finally:
        _restore()
        shutil.rmtree(T, ignore_errors=True)


def _cli(args):
    env = dict(os.environ, PYTHONPATH=ROOT)
    return subprocess.run([sys.executable, '-m', 'exopipe.phase3', 'run', 'update'] + args + ['--cores', 'none'], cwd=ROOT, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)


def test_cli():
    r = _cli(['--hybrid-out', '/x/H', '--v3-out', '/x/M3', '--v5-out', '/x/M5'])
    assert r.returncode != 0 and 'exactly one of --trigger' in r.stdout, r.stdout[-400:]
    r = _cli(['--trigger', '13h_Green', '--from-exclusions', '--hybrid-out', '/x/H', '--v3-out', '/x/M3', '--v5-out', '/x/M5'])
    assert r.returncode != 0 and 'exactly one of --trigger' in r.stdout, r.stdout[-400:]
    r = _cli(['--from-exclusions', '--hybrid-out', '/x/H', '--v3-out', '/x/M3'])
    assert r.returncode != 0 and '--v5-out' in r.stdout, r.stdout[-400:]
    r = _cli(['--trigger', '13h_Green', '--hybrid-out', '/x/H', '--v3-out', '/x/M3'])
    assert r.returncode != 0 and '--v5-out' in r.stdout, r.stdout[-400:]
    r = _cli(['--trigger', '13h_Grn', '--hybrid-out', '/x/H', '--v3-out', '/x/M3', '--v5-out', '/x/M5'])
    assert r.returncode != 0 and 'refused' in r.stdout and '13h_Grn' in r.stdout, r.stdout[-400:]
    r = _cli(['--trigger', '@/nonexistent/trigger.txt', '--hybrid-out', '/x/H', '--v3-out', '/x/M3', '--v5-out', '/x/M5', '--plan', '/x/P'])
    assert r.returncode != 0 and 'refused' in r.stdout and 'trigger.txt' in r.stdout and 'Traceback' not in r.stdout, r.stdout[-400:]
    r = _cli(['--trigger', '13h_Green', '--hybrid-out', '/x/H', '--v3-out', '/x/M3', '--v5-out', '/x/M5'])
    assert r.returncode != 0 and 'give --nodes' in r.stdout, r.stdout[-400:]
    d = tempfile.mkdtemp(prefix='upd_plan_', dir=os.environ.get('TMPDIR'))
    try:
        jobs_before = sorted(os.listdir(runner.jobs_root())) if os.path.isdir(runner.jobs_root()) else []
        r = _cli(['--trigger', ','.join(TRIG_1113), '--hybrid-out', '/x/H', '--v3-out', '/x/M3', '--v5-out', '/x/M5', '--plan', d])
        assert r.returncode == 0, r.stdout[-600:]
        pl = json.load(open(os.path.join(d, 'plan.json')))
        assert pl['cells'] == ['13h_Green', '15h_Green', '13h_Full'] and not pl['apply']
        rows = list(csv.reader(open(os.path.join(d, 'plan.csv'))))
        assert rows[0] == ['order', 'entry', 'item', 'command', 'kind', 'action', 'dst', 'quarantine', 'note'] and any(
            x[1] == 'tilefill' for x in rows[1:])
        assert pl['round'] and 'quarantine round' in r.stdout, r.stdout[-400:]
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
    pl = UP.build(TRIG_1113, OUTS, apply=True, rnd='20261113')   # the preview of an --apply run (`--plan D --apply`)
    pred, qd = {}, {}
    for r in UP.predicted(pl):
        pred.setdefault(r['entry'], {})[r['dst']] = (r['unit'], r['kind'])
        qd[r['dst']] = r['quarantine']
    hrows = [r for r in csv.DictReader(open(H)) if r['kind'] == 'superseded']
    hm = {r['src'] for r in hrows}
    ph = pred['push_hybrid']
    assert hm and hm <= set(ph), sorted(hm - set(ph))[:5]
    wrong = [(r['src'], qd[r['src']], r['dst']) for r in hrows if qd[r['src']] != r['dst']]
    assert not wrong, wrong[:3]
    extra = {(ph[p][1], ph[p][0].rsplit('_', 1)[0]) for p in set(ph) - hm}
    assert {k for k, _c in extra} <= {'recon', 'recon_pbcorr', 'pbdiffuse'}, extra
    assert {c for _k, c in extra} <= {'13h_Green', '15h_Green', '13h_Full'}, extra
    vm = {r['dst'] for r in csv.DictReader(open(V))}
    pv = pred['push_mosaic_v5']
    assert vm and vm <= set(pv), sorted(vm - set(pv))[:5]
    assert {os.path.basename(p) for p in set(pv) - vm} <= {'taper_only.csv'}, sorted(set(pv) - vm)


TESTS = [test_closure, test_items_1113, test_order, test_commands_parse, test_subst, test_after_ok, test_tables_touched, test_round,
         test_publish_rows, test_from_exclusions, test_cli, test_replay_1113_cluster, test_from_exclusions_1113_cluster]


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
