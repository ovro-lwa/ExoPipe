#!/usr/bin/env python3
"""STANDING TEST (data-free) - the per-hour catalog step of the 10-second and 1-minute (snapshot) tiers:
`exopipe3 catalog --tier snapshot --incremental --epochs <LST>_<date>` (lib/catalog/incremental_snapshot.py).

Ruling: 2026-09-18 RULING-INCREMENTAL-PUBLISHING-TIE-AS-A-PER-CELL-STAGE-JUDGEMENT-CARRY-OVER (the per-cell pre-chain stage,
production ops/snapshot_cell_prestage.sh); R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS (the per-hour principles; the
cross-cell association has no ruled design, so its phase refuses).

Everything is synthetic, in a temporary directory of this test's own (tempfile.mkdtemp, removed at the end): a fleet-out
tree, a batch directory, light-curve tables (empty files), an exclusion stub, and STUB in-place modules (phase3_snapshot_chain.py
and phase3_snapshot_fit.py) run by core/legacy.run under this interpreter. The stub chain reads <fleet>/<cell>/dets.csv
(the detections that reach the per-frame fit step), lists those without a store row (--prefit-list-only) and ties the
association-pool ones not in the tie cache (--stop-after-tie), as production's chain does; the stub fitter APPENDS rows to
the store, as fit-grouped does. Every stub call writes its argv and environment to calls.jsonl.

Asserts:
  happy path   prefit round 1 lists the unfitted detections, the fitter appends them, round 2 lists none; the tie pass ties
               the pool; outputs (store, tie cache, chain_summary.json, prefit lists, prestage.log) recorded and content
               tested; the claim released; the chain arguments (--fit-store = the cell's store, --veto-record-root '',
               --stop-after-tie, P3_TIE_CACHE_DIR, P3_REFASTRO_SHARDS, the refastro scratch off the production roots)
  associate    always refused SI-G9 naming the four open choices; the run exits 2 (the hour is never complete); alone
               (--inc-phase associate) and in a dry run (exit 2, nothing written)
  rerun        a second run ties nothing new (every pool detection from the cache)
  gates        SI-G1 no fleet claim; SI-G2 dirty cell; SI-G3 a light-curve table missing (excluded and flow-unregistered
               subbands not required); SI-G4 a live claim refuses, a dead claim on this host is released, the production
               root's batch-chain hold refuses; SI-G5 not converged in PREFIT_MAX_ITER rounds; SI-G6 a chain exiting non-zero
               (its log copied beside the cell's products)
  no input     no detection at the fit step, or none in the pool: no-input, never done
  seeding      a non-production --out copies the production store and tie-cache files (tie_ and tie2_) it lacks; 'none'
               copies nothing; an existing copy is kept
  helpers      phases, seeds, the dirty list parse, the prestage.log filter; tier_snapshot.chain_args; the tie_cache
               content test on a tie2_ file; steps/catalog.py routes --tier snapshot --incremental here

Run standalone:  python tests/test_catalog_snapshot_incremental.py [test_name ...]
Under pytest:    pytest tests/test_catalog_snapshot_incremental.py
"""
import argparse
import csv
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

_IMPORT_ERROR = None
try:
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import exclusions as EX
    from exopipe.phase3.core import record, validate
    from exopipe.phase3.lib.catalog import incremental_snapshot as IS
    from exopipe.phase3.lib.catalog import tier_snapshot as TS
    from exopipe.phase3.lib.catalog import tier_sublst as TSL
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e
    if __name__ != '__main__':
        raise

CELL = '12h_2024-12-21'
SBS = ('18MHz', '41MHz', '73MHz')

STUB_CHAIN = r'''
import csv, json, os, sys
a = sys.argv[1:]
def opt(k):
    return a[a.index(k) + 1]
out, store, cell, proto = opt("--out"), opt("--fit-store"), opt("--cells"), opt("--proto-root")
with open(os.environ["STUB_CALLS"], "a") as f:
    f.write(json.dumps({"prog": "chain", "argv": a, "env": {k: v for k, v in os.environ.items() if k.startswith("P3_")}}) + "\n")
os.makedirs(out, exist_ok=True)
dets = list(csv.DictReader(open(os.path.join(proto, cell, "dets.csv"))))
if any(d["key"] == "CRASH" for d in dets):
    print("Traceback: stub chain crash"); sys.exit(1)
fitted = set()
if os.path.exists(store):
    fitted = {r["key"] for r in csv.DictReader(open(store)) if r["fit_status"] == "ok"}
summary = {"cells": [cell], "n_loaded_%s" % cell: 10 + len(dets)}
todo = [d for d in dets if d["key"] not in fitted]
summary["prefit"] = {"n_at_per_frame_fit_step": len(dets), "n_without_fit": len(todo)}
if "--prefit-list-only" in a:
    with open(os.path.join(out, "prefit_requests.csv"), "w") as f:
        f.write("cell,subband,stokes,tier,ts,ra_deg,dec_deg,snr,key\n")
        for d in todo:
            f.write("%s,18MHz,I,10s,%s,1.0,2.0,9.0,%s\n" % (cell, d["key"], d["key"]))
    json.dump(summary, open(os.path.join(out, "chain_summary.json"), "w"))
    print("PREFIT-LIST-ONLY: stopped after the per-frame fit step's list (%d to fit)" % len(todo)); sys.exit(0)
if todo:
    print("PREFIT-INCOMPLETE: %d - run REFUSED" % len(todo)); sys.exit(6)
assert "--stop-after-tie" in a
d_ = os.environ["P3_TIE_CACHE_DIR"]; os.makedirs(d_, exist_ok=True)
pool = [d for d in dets if d["pool"] == "1"]
p = os.path.join(d_, "tie_%s.csv" % cell)
cached = {r["ts"] for r in csv.DictReader(open(p))} if os.path.exists(p) else set()
p2 = os.path.join(d_, "tie2_%s.csv" % cell)
if os.path.exists(p2):
    cached |= {r["ts"] for r in csv.DictReader(open(p2))}
new = [d for d in pool if d["key"] not in cached]
if new:
    hdr = not os.path.exists(p)
    with open(p, "a") as f:
        if hdr:
            f.write("cell,sb,st,tier,ts,ra_pre,dec_pre,tie_status,ra,dec,errmodel\n")
        for d in new:
            f.write("%s,18MHz,I,10s,%s,1.00000,2.00000,tied,1.01,2.01,stub_model\n" % (cell, d["key"]))
summary["tie_cache"] = {"dir": d_, "from_cache": len(pool) - len(new), "tied_now": len(new)}
print("tie cache: %d detections from %s, %d tied now" % (len(pool) - len(new), d_, len(new)))
json.dump(summary, open(os.path.join(out, "chain_summary.json"), "w"))
print("STOP-AFTER-TIE: %s" % json.dumps(summary["tie_cache"]))
'''

STUB_FIT = r'''
import csv, json, os, sys
a = sys.argv[1:]
pend, store = a[a.index("--pending") + 1], a[a.index("--store") + 1]
with open(os.environ["STUB_CALLS"], "a") as f:
    f.write(json.dumps({"prog": "fit", "argv": a, "env": {k: v for k, v in os.environ.items()
                                                         if k.startswith("P3_") or k.endswith("_NUM_THREADS")}}) + "\n")
rows = [r for r in csv.DictReader(open(pend)) if not r["key"].startswith("UNFITTABLE")]
new = not os.path.exists(store)
with open(store, "a") as f:
    if new:
        f.write("cell,subband,stokes,tier,ts,key,fit_status\n")
    for r in rows:
        f.write("%s,%s,%s,%s,%s,%s,ok\n" % (r["cell"], r["subband"], r["stokes"], r["tier"], r["ts"], r["key"]))
print("fit-grouped DONE: %d fitted" % len(rows))
'''


class Patch:
    """Set attributes for the duration of a with-block (module functions, config section values)."""

    def __init__(self, *triples):
        self.triples, self.old = triples, []

    def __enter__(self):
        for obj, name, val in self.triples:
            self.old.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)
        return self

    def __exit__(self, *exc):
        for obj, name, val in reversed(self.old):
            setattr(obj, name, val)


class _Excl:
    def __init__(self, rows):
        self.rows = set(rows)

    def is_excluded(self, lst, scope, date):
        return (lst, scope, date) in self.rows


class Tree:
    """A synthetic fleet-out / batch / light-curve tree and stub in-place modules; config pointed at it inside the block."""

    def __init__(self, dets=(('d1', 1), ('d2', 1), ('d3', 0)), fitted=('d1',), claim_done=True, lc_missing=(),
                 excluded=(), flow_unregistered=(), dirty=False):
        self.dir = tempfile.mkdtemp(prefix='catalog_snapshot_incremental_')
        d = self.dir
        self.fleet, self.claims = f'{d}/fleet_out', f'{d}/fleet_claims'
        self.lc, self.batch, self.stub = f'{d}/lightcurves', f'{d}/batch', f'{d}/stub'
        self.out, self.work = f'{d}/out', f'{d}/scratch'
        self.calls = f'{d}/calls.jsonl'
        for p in (self.fleet, self.claims, self.lc, self.batch, f'{self.stub}/clv15', f'{self.stub}/exopipe_ops', self.work):
            os.makedirs(p, exist_ok=True)
        open(f'{self.stub}/clv15/phase3_snapshot_chain.py', 'w').write(STUB_CHAIN)
        open(f'{self.stub}/clv15/phase3_snapshot_fit.py', 'w').write(STUB_FIT)
        cf = f'{self.fleet}/{CELL}'
        os.makedirs(cf)
        lst, date = CELL.split('_', 1)
        for sb in SBS:
            open(f'{cf}/admission_{sb}.csv', 'w').write('ts,admitted\n')
            for st in ('I', 'V'):
                json.dump({'flow_registered': sb not in flow_unregistered, 'n_frames': 360},
                          open(f'{cf}/runinfo_{sb}_{st}_median.json', 'w'))
                if sb not in flow_unregistered and (lst, sb, date) not in excluded and f'{sb}{st}' not in lc_missing:
                    os.makedirs(f'{self.lc}/{CELL}', exist_ok=True)
                    open(f'{self.lc}/{CELL}/{CELL}_{sb}_{st}_lightcurves.fits', 'w').write('stub')
        with open(f'{cf}/dets.csv', 'w') as f:
            f.write('key,pool\n' + ''.join(f'{k},{p}\n' for k, p in dets))
        if claim_done:
            os.makedirs(f'{self.claims}/{CELL}')
            open(f'{self.claims}/{CELL}/done', 'w').write('stub done\n')
        self.dirty = f'{d}/dirty.txt'
        open(self.dirty, 'w').write('# dirty Stokes I cells\n' + ('%s %s\n' % (lst, date) if dirty else '04h 2025-01-01\n'))
        if fitted:
            os.makedirs(f'{self.batch}/fits')
            with open(f'{self.batch}/fits/fits_{CELL}.csv', 'w') as f:
                f.write('cell,subband,stokes,tier,ts,key,fit_status\n')
                for k in fitted:
                    f.write(f'{CELL},18MHz,I,10s,{k},{k},ok\n')
        self.excluded = excluded
        self._patch = Patch(
            (CFG.paths, 'SNAPSHOT_FLEET_OUT', self.fleet), (CFG.paths, 'SNAPSHOT_FLOW_ROOT', f'{d}/flow'),
            (CFG.curves, 'FLEET_CLAIMS_ROOT', self.claims),
            (CFG.curves, 'SEARCH_DONE_EVIDENCE', 'fleet-claim-done'), (CFG.curves, 'SKIP_FLOW_UNREGISTERED', True),
            (CFG.curves, 'PLANNED_SUBBANDS', 'admission'), (CFG.curves, 'MASTER_CATALOG_DIR', f'{d}/mastercats'),
            (CFG.instrument, 'ALL_SUBBANDS', SBS),
            (CFG.catalog_snapshot, 'LIGHTCURVES_ROOT', self.lc), (CFG.catalog_snapshot, 'BATCH_DIR', self.batch),
            (CFG.catalog_snapshot, 'DIRTY_CELLS', self.dirty), (CFG.catalog_snapshot, 'DIRTY_I_FLOOR', 0.0),
            (CFG.catalog_snapshot, 'WORK_ROOT', f'{d}/prodwork'),
            (CFG.site, 'PYTHON', {'default': sys.executable, 'py36': sys.executable}),
            (CFG.site, 'LEGACY_CODE_ROOTS', [f'{self.stub}/clv15', f'{self.stub}/exopipe_ops']),
            (EX, 'for_tier', lambda tier, **k: _Excl(self.excluded)))

    def __enter__(self):
        self._patch.__enter__()
        os.environ['STUB_CALLS'] = self.calls
        return self

    def __exit__(self, *exc):
        self._patch.__exit__(*exc)
        os.environ.pop('STUB_CALLS', None)
        shutil.rmtree(self.dir, ignore_errors=True)

    def args(self, **kw):
        a = argparse.Namespace(tier='snapshot', incremental=True, epochs=CELL, inc_phase=None, snap_inc_seed=None,
                               work=self.work)
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    def run(self, out=None, dry=False, **kw):
        out = out or self.out
        r = record.Run('catalog', ['test'], out, CFG, dry_run=dry)
        rc = IS.run(self.args(**kw), r)
        return r, rc

    def calls_of(self, prog):
        if not os.path.exists(self.calls):
            return []
        return [c for c in (json.loads(ln) for ln in open(self.calls)) if c['prog'] == prog]


def unit(r, phase):
    us = [u for u in r.units if u.name == f'{IS.UNIT}.{phase}.{CELL}']
    assert len(us) == 1, [u.name for u in r.units]
    return us[0]


def rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


# =====================================================================================================================
# tests
# =====================================================================================================================

def test_happy_path_and_associate_refusal():
    with Tree() as t:
        r, rc = t.run()
        u = unit(r, 'cell')
        assert rc is None and u.status == 'done', (u.status, u.reason)
        cd = f'{t.out}/cells/{CELL}'
        assert len(rows(f'{cd}/prefit1_requests.csv')) == 2 and len(rows(f'{cd}/prefit2_requests.csv')) == 0
        assert [x['round'] for x in u.meta['prefit_rounds']] == [1, 2] and u.meta['prefit_fitted'] == 2
        assert len(rows(f'{t.out}/fits/fits_{CELL}.csv')) == 3            # seeded 1 + fitted 2
        assert len(rows(f'{t.out}/tiecache/tie_{CELL}.csv')) == 2           # the two pool detections
        assert u.meta['tie_tied_now'] == 2 and u.meta['tie_from_cache'] == 0
        assert json.load(open(f'{cd}/chain_summary.json'))['tie_cache']['tied_now'] == 2
        assert 'tie cache' in open(f'{cd}/prestage.log').read()
        classes = sorted(o['class'] for o in u.outputs)
        assert classes == ['csv_table', 'csv_table', 'csv_table', 'snapshot_tie_summary', 'tie_cache'], classes
        assert not u.failed_tests(), u.failed_tests()
        assert not os.path.exists(f'{t.out}/claims/{CELL}')                # claim released
        assert os.path.exists(f'{t.out}/_work/{CELL}/tie/chain_summary.json')   # non-production keeps the work
        ch = t.calls_of('chain')
        assert len(ch) == 3 and '--prefit-list-only' in ch[0]['argv'] and '--stop-after-tie' in ch[2]['argv']
        for c in ch:
            a = c['argv']
            assert a[a.index('--fit-store') + 1] == f'{t.out}/fits/fits_{CELL}.csv'
            assert a[a.index('--veto-record-root') + 1] == '' and a[a.index('--cells') + 1] == CELL
            assert a[a.index('--lightcurves-root') + 1] == t.lc
            assert CFG.production_root_of(c['env']['P3_SNAP_REFASTRO_SCRATCH']) is None
        tie_env = ch[2]['env']
        assert tie_env['P3_TIE_CACHE_DIR'] == f'{t.out}/tiecache'
        assert tie_env['P3_REFASTRO_SHARDS'] == str(CFG.catalog_snapshot.TIE_SHARDS[2])
        assert 'P3_TIE_CACHE_DIR' not in ch[0]['env']
        fit = t.calls_of('fit')
        assert len(fit) == 1 and fit[0]['env']['OMP_NUM_THREADS'] == '1'
        assert fit[0]['argv'][fit[0]['argv'].index('--store') + 1] == f'{t.out}/fits/fits_{CELL}.csv'
        a = unit(r, 'associate')
        assert a.status == 'refused' and a.reason.startswith('SI-G9: ') and all(f'({i})' in a.reason for i in (1, 2, 3, 4))
        assert 'The cell phase is done' in a.reason
        assert r.exit_code() == record.EXIT_REFUSED
        rec = record.load_record(t.out, 'catalog', f'{IS.UNIT}.cell.{CELL}')
        assert rec['status'] == 'done' and rec['meta']['plan'][0].startswith('SI-G1 search shipped: fleet claim done')
        # rerun: every pool detection from the cache, nothing fitted
        r2, _ = t.run()
        u2 = unit(r2, 'cell')
        assert u2.status == 'done' and u2.meta['tie_tied_now'] == 0 and u2.meta['tie_from_cache'] == 2, u2.meta
        assert u2.meta['prefit_fitted'] == 0 and len(rows(f'{t.out}/tiecache/tie_{CELL}.csv')) == 2


def test_associate_alone_and_dry_run():
    with Tree() as t:
        r, rc = t.run(inc_phase='associate')
        assert [u.name for u in r.units] == [f'{IS.UNIT}.associate.{CELL}'] and r.exit_code() == record.EXIT_REFUSED
        assert 'not run (--inc-phase)' in r.units[0].reason and not os.path.exists(f'{t.out}/fits')
    with Tree() as t:
        r, rc = t.run(dry=True)
        assert rc == 2 and not r.units and not os.path.exists(t.out) and not t.calls_of('chain')
        r, rc = t.run(dry=True, inc_phase='cell')
        assert rc == 0 and not os.path.exists(t.out)


def test_gate_search_shipped():
    with Tree(claim_done=False) as t:
        r, _ = t.run()
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G1: ') and 'has not finished' in u.reason, u.reason
        assert not t.calls_of('chain') and not os.path.exists(f'{t.out}/fits')
        assert unit(r, 'associate').status == 'refused'


def test_gate_dirty():
    with Tree(dirty=True) as t:
        r, _ = t.run()
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G2: '), u.reason
        assert not t.calls_of('chain')


def test_gate_lightcurves():
    with Tree(lc_missing=('41MHzV',)) as t:
        r, _ = t.run()
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G3: ') and '41MHzV' in u.reason, u.reason
        assert not t.calls_of('chain')
    # an excluded subband and a flow-unregistered one need no table (curves plan_cell)
    lst, date = CELL.split('_', 1)
    with Tree(excluded=[(lst, '41MHz', date)], flow_unregistered=('73MHz',)) as t:
        r, _ = t.run()
        u = unit(r, 'cell')
        assert u.status == 'done', u.reason
        assert any('not required: 41MHz (excluded), 73MHz (flow-unregistered)' in x for x in u.meta['plan']), u.meta['plan']


def test_gate_claim():
    with Tree() as t:
        cd = f'{t.out}/claims/{CELL}'
        os.makedirs(cd)
        open(f'{cd}/owner', 'w').write(f'{socket.gethostname()}:{os.getpid()}\n')      # a live writer
        r, _ = t.run()
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G4: ') and 'claimed by' in u.reason, u.reason
        assert os.path.isdir(cd) and not t.calls_of('chain')                     # another writer's claim is kept
        p = subprocess.Popen([sys.executable, '-c', 'pass'])
        p.wait()
        open(f'{cd}/owner', 'w').write(f'{socket.gethostname()}:{p.pid}\n')        # a dead writer on this host
        r, _ = t.run()
        assert unit(r, 'cell').status == 'done' and not os.path.exists(cd)


def test_production_root_hold_and_layout():
    with Tree() as t:
        with Patch((CFG.paths, 'PRODUCTION_ROOTS', tuple(CFG.paths.PRODUCTION_ROOTS) + (t.batch,))):
            os.makedirs(f'{t.batch}/rechain_active')
            open(f'{t.batch}/rechain_active/Run_SNAPX.cells', 'w').write('%s %s\n' % tuple(CELL.split('_', 1)))
            r, _ = t.run(out=t.batch)
            u = unit(r, 'cell')
            assert u.status == 'refused' and u.reason.startswith('SI-G4: ') and 'batch chain' in u.reason, u.reason
            os.remove(f'{t.batch}/rechain_active/Run_SNAPX.cells')
            try:                                                                # seeding is for a non-production --out
                t.run(out=t.batch, snap_inc_seed='store')
                raise AssertionError('--snap-inc-seed was accepted on the production root')
            except record.Refusal as e:
                assert str(e).startswith('SI-G7: '), e
            r, _ = t.run(out=t.batch)
            u = unit(r, 'cell')
            assert u.status == 'done', u.reason
            assert len(rows(f'{t.batch}/tiecache/tie_{CELL}.csv')) == 2 and len(rows(f'{t.batch}/fits/fits_{CELL}.csv')) == 3
            assert not os.path.exists(f'{t.dir}/prodwork/prestage/{CELL}')     # prestage.sh:86 removes the work
            assert 'P3_SNAP_REFASTRO_SCRATCH' not in t.calls_of('chain')[-1]['env']
            other = f'{t.dir}/otherprod'
            with Patch((CFG.paths, 'PRODUCTION_ROOTS', tuple(CFG.paths.PRODUCTION_ROOTS) + (other,))):
                try:
                    t.run(out=other)
                    raise AssertionError('another production root was accepted')
                except record.Refusal as e:
                    assert str(e).startswith('SI-G4: ')


def test_gate_not_converged():
    with Tree(dets=(('d1', 1), ('UNFITTABLE1', 1))) as t:
        r, _ = t.run()
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G5: ') and 'after 3 prefit rounds' in u.reason, u.reason
        assert len(t.calls_of('fit')) == 3 and not os.path.exists(f'{t.out}/tiecache')


def test_gate_chain_failure():
    with Tree(dets=(('CRASH', 1),)) as t:
        r, _ = t.run()
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G6: ') and 'prefit list 1' in u.reason, u.reason
        assert os.path.exists(f'{t.out}/cells/{CELL}/prefit1.log')
        assert not os.path.exists(f'{t.out}/claims/{CELL}')


def test_no_input():
    with Tree(dets=(), fitted=()) as t:
        r, _ = t.run()
        u = unit(r, 'cell')
        assert u.status == 'no-input' and 'per-frame fit step' in u.reason, u.reason
        assert unit(r, 'associate').status == 'refused' and r.exit_code() == record.EXIT_REFUSED
    with Tree(dets=(('d1', 0),)) as t:
        r, _ = t.run()
        u = unit(r, 'cell')
        assert u.status == 'no-input' and 'association pool' in u.reason, u.reason


def test_seeding():
    with Tree(fitted=('d1', 'd2', 'd3')) as t:
        os.makedirs(f'{t.batch}/tiecache')
        open(f'{t.batch}/tiecache/tie_{CELL}.csv', 'w').write(
            'cell,sb,st,tier,ts,ra_pre,dec_pre,tie_status,ra,dec,errmodel\n%s,18MHz,I,10s,d1,1,2,tied,1,2,m\n' % CELL)
        open(f'{t.batch}/tiecache/tie2_{CELL}.csv', 'w').write(
            'cell,sb,st,tier,ts,ra_pre,dec_pre,tie_status,ra,dec,errmodel\n%s,18MHz,I,10s,d2,1,2,tied,1,2,m\n' % CELL)
        r, _ = t.run()
        u = unit(r, 'cell')
        assert u.status == 'done' and u.meta['tie_from_cache'] == 2 and u.meta['tie_tied_now'] == 0, u.meta
        assert u.meta['prefit_fitted'] == 0 and len(u.meta['seeded']) == 3, u.meta.get('seeded')
        assert sorted(o['class'] for o in u.outputs).count('tie_cache') == 2 and not u.failed_tests(), u.failed_tests()
        # an existing copy is kept: the production file changes, the run's copy does not
        open(f'{t.batch}/fits/fits_{CELL}.csv', 'a').write(f'{CELL},18MHz,I,10s,x9,x9,ok\n')
        r, _ = t.run()
        assert not unit(r, 'cell').meta.get('seeded') and len(rows(f'{t.out}/fits/fits_{CELL}.csv')) == 3
    with Tree() as t:
        r, _ = t.run(snap_inc_seed='none')
        u = unit(r, 'cell')
        assert u.status == 'done' and u.meta['prefit_fitted'] == 3 and not u.meta.get('seeded')


def test_helpers():
    a = argparse.Namespace(inc_phase=None, snap_inc_seed=None)
    assert IS.parse_phases(a) == ['cell', 'associate'] and IS.parse_seeds(a) == ['store', 'tiecache']
    assert IS.parse_phases(argparse.Namespace(inc_phase='associate,cell')) == ['cell', 'associate']
    for bad in ('epoch', ''):
        try:
            IS.parse_phases(argparse.Namespace(inc_phase=bad or ','))
            raise AssertionError(bad)
        except SystemExit:
            pass
    assert IS.parse_seeds(argparse.Namespace(snap_inc_seed='none')) == []
    assert IS.parse_seeds(argparse.Namespace(snap_inc_seed='tiecache')) == ['tiecache']
    d = tempfile.mkdtemp(prefix='catalog_snapshot_incremental_h_')
    try:
        p = f'{d}/dirty.txt'
        open(p, 'w').write('# 01h 2025-01-01 commented\n12h 2024-12-21 note\n13h\n')
        assert IS.dirty_cells(p) == {'12h_2024-12-21'} and IS.dirty_cells(f'{d}/none.txt') == set()
        assert IS.keep_lines('a WARNING tie\ntie cache: 1\nnothing\nSTEP load 1 s\nREFUSING x\n') == [
            'tie cache: 1', 'STEP load 1 s', 'REFUSING x']
        s = f'{d}/chain_summary.json'
        json.dump({'prefit': {'n_at_per_frame_fit_step': 7}, 'n_loaded_a': 3, 'n_loaded_b': 4}, open(s, 'w'))
        assert IS.summary_counts(s) == (7, 7)
        # the tie_cache content test reads the cell of a tie2_ file too
        t2 = f'{d}/tie2_{CELL}.csv'
        open(t2, 'w').write('cell,sb,st,tier,ts,ra_pre,dec_pre,tie_status,ra,dec,errmodel\n'
                            f'{CELL},18MHz,I,10s,t,1,2,tied,1.0,2.0,m\n')
        res = validate.run('tie_cache', t2)
        assert all(x['passed'] for x in res), res
        open(t2, 'a').write('01h_2025-01-01,18MHz,I,10s,t,1,2,tied,1.0,2.0,m\n')
        assert not all(x['passed'] for x in validate.run('tie_cache', t2))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # tier_snapshot.chain_args (moved out of Ctx.chainargs): every [arg] flag when the script has no defaults
    c = CFG.catalog_snapshot
    a = TS.chain_args(CFG, c, '/x/fits_c.csv', '/nonexistent/phase3_snapshot_chain.py')
    assert a[:10] == ['--proto-root', CFG.paths.SNAPSHOT_FLEET_OUT, '--master-catalogs', c.MASTER_CATALOGS,
                      '--curve-constants', c.CURVE_CONSTANTS, '--fit-store', '/x/fits_c.csv', '--lightcurves-root',
                      c.LIGHTCURVES_ROOT]
    assert a[a.index('--dirty-cells') + 1] == c.DIRTY_CELLS and '--curve-constants-restore' not in a
    b = TS.chain_args(CFG, c, 's', '/nonexistent', restore='/r.json', analysis=[('no-tie', '')])
    assert b[b.index('--curve-constants-restore') + 1] == '/r.json' and b[-1] == '--no-tie'
    assert CFG.catalog_snapshot.INC_PHASES_DEFAULT == ('cell', 'associate')
    assert TSL.count_lines  # the shared helper the module uses


def test_step_routes_here():
    from exopipe.phase3.steps import catalog as step
    a = argparse.Namespace(tier='snapshot', incremental=True, run_label=None)
    assert step.default_out(a) == CFG.catalog_snapshot.BATCH_DIR
    seen = []
    with Patch((IS, 'run', lambda args, run, epochs: seen.append(epochs) or 'routed')):
        a = argparse.Namespace(tier='snapshot', incremental=True, epochs=CELL)
        assert step.run(a, None) == 'routed' and seen == [[CELL]]


TESTS = [test_happy_path_and_associate_refusal, test_associate_alone_and_dry_run, test_gate_search_shipped, test_gate_dirty,
         test_gate_lightcurves, test_gate_claim, test_production_root_hold_and_layout, test_gate_not_converged,
         test_gate_chain_failure, test_no_input, test_seeding, test_helpers, test_step_routes_here]


def main(argv):
    if _IMPORT_ERROR is not None:
        print(f'UNTESTED-IMPORT: {type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}')
        return 2
    want = set(argv)
    failed = 0
    for t in TESTS:
        if want and t.__name__ not in want:
            continue
        try:
            t()
            print(f'PASS {t.__name__}')
        except Exception as e:
            failed += 1
            import traceback
            print(f'FAIL {t.__name__}: {type(e).__name__}: {e}')
            traceback.print_exc()
    print(f'{len(TESTS) - failed if not want else "selected"} passed, {failed} failed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
