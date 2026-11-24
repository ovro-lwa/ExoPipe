#!/usr/bin/env python3
"""STANDING TEST (data-free) - the per-hour catalog step of the 10-second and 1-minute (snapshot) tiers:
`exopipe3 catalog --tier snapshot --incremental --epochs <LST>_<date>` (lib/catalog/incremental_snapshot.py).

Ruling: 2026-09-18 RULING-INCREMENTAL-PUBLISHING-TIE-AS-A-PER-CELL-STAGE-JUDGEMENT-CARRY-OVER (the per-cell pre-chain stage,
production ops/snapshot_cell_prestage.sh); R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS (the per-hour principles; the
cross-cell association has no ruled design, so its phase refuses).

Everything is synthetic, in a temporary directory of this test's own (tempfile.mkdtemp, removed at the end): a fleet-out
tree, a batch directory, light-curve tables (empty files), an exclusion stub, and STUB in-place modules (phase3_snapshot_chain.py,
phase3_snapshot_fit.py, phase3_snapshot_search.py, phase3_snapshot_tiecache.py) run by core/legacy.run under this interpreter;
the node's available memory is fixed at room for 2 fit workers. The stub chain reads <fleet>/<cell>/dets.csv
(the detections that reach the per-frame fit step), lists those without a store row (--prefit-list-only) and ties the
association-pool ones not in the tie cache (--stop-after-tie), as production's chain does; the stub fitter APPENDS rows to
the store, as fit-grouped does. Every stub call writes its argv and environment to calls.jsonl.

Asserts:
  happy path   (on a production root) the store repair lists nothing, prefit round 1 lists the unfitted detections, the
               fitter appends them, round 2 lists none; the store test prints READY; the tie pass ties the pool; outputs
               (store, tie cache, chain_summary.json, prefit lists, prestage.log) recorded and content tested; the claim
               <out>/prefit_claims/<cell> released; the chain arguments (--fit-store = the cell's store, --veto-record-root
               '', --curve-constants-restore = INC_CURVE_CONSTANTS_RESTORE, --stop-after-tie, P3_TIE_CACHE_DIR,
               P3_REFASTRO_SHARDS); a tie pass is judged by its exit code alone
  associate    always refused SI-G9 naming the four open choices; the run exits 2 (the hour is never complete); alone
               (--inc-phase associate) and in a dry run (exit 2, nothing written)
  rerun        a second run ties nothing new (every pool detection from the cache)
  production   a cell production prestaged (PRESTAGE_DONE) is recorded skipped-valid (done by production) and not run,
               even with production's claim left beside it; an old claim without PRESTAGE_DONE holds the cell
  gates        SI-G1 no fleet claim; SI-G2 dirty cell; SI-G3 a light-curve table missing (excluded and flow-unregistered
               subbands not required); SI-G4 the hold lists, rechain_active, a running frame-time claim, a live claim
               (kept), a dead claim on this node and a stale owner-less claim (released); SI-G5 not converged in
               PREFIT_MAX_ITER rounds; SI-G6 a chain exiting non-zero (its log copied beside the cell's products); SI-G8
               forced stage 1 marked failed, not done, and gone by the end; SI-G10 NOT-READY, and a non-production store
               that differs from production's; SI-G11 a missing restore table
  repair       rows the store test refuses are listed, refitted into a new store and appended, then READY
  upstream     a cell an earlier step of the chain item refused: both units recorded UPSTREAM-REFUSED, nothing run
  no input     no detection at the fit step, or none in the pool: no-input, never done
  seeding      a non-production --out copies the production store and tie-cache files (tie_ and tie2_) it lacks and is
               store-tested through production's test while it holds production's content; 'none' copies nothing; an
               existing copy is kept
  attempts     (prefit.sh:62-72, :152-168; Gregg 2026-10-10 'Go on all 3') SI-G12: >= INC_PREFIT_MAX_FAILS (3) lines in
               <out>/cells/<cell>/PREFIT_FAILS park the cell on any root, 2 do not; on the production root one line per
               failed store repair (its logs kept), prefit loop (not converged, a chain exit or refusal), store test (its
               NOT-READY line) and forced stage 1 at the end; none for a tie-pass failure, none while a stop file exists,
               none on a non-production root
  stop files   SI-G13: PREFIT_STOP and PREFIT_STOP_<node> under --out before the cell and between fit shards (no failed
               attempt); STOP_<node> and another node's file are not honored
  chain check  SI-G14 (prefit.sh:82-95): flow-unregistered and paused 1-minute searches, candidate tables of a subband not
               excluded without their 1-minute table; refused before the claim, nothing run, no failed attempt
  done-by      PRESTAGE_DONE is done only with a tie_ or tie2_ file whose header is phase3_snapshot_tiecache.FIELDS (no file,
               or an older header: the unit runs); a dry run prints it and records nothing
  fits         shards (ceil(requests / INC_FIT_SHARD_REQUESTS), at most groups // (INC_FIT_TASKS_PER_WORKER x workers)),
               workers = min(free cores - reserve, memory's), >= 2, waiting for memory; niceness 19; the prefit-list passes
               under a 6-h limit and the tie pass under INC_CHAIN_TIMEOUT_H; an SI-G6 reason quotes the chain's REFUSING
               line, not the 60 warning lines after it
  helpers      phases, seeds, the cell-list parse, the prestage.log filter, the stale-claim rule; tier_snapshot.chain_args
               and chain_env; the tie_cache content test on a tie2_ file; steps/catalog.py routes --tier snapshot
               --incremental here

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
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

_IMPORT_ERROR = None
try:
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import exclusions as EX
    from exopipe.phase3.core import nodemem as NM
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
if os.environ.get("STUB_TOUCH"):
    open(os.environ["STUB_TOUCH"], "w").close()
os.makedirs(out, exist_ok=True)
dets = list(csv.DictReader(open(os.path.join(proto, cell, "dets.csv"))))
if any(d["key"] == "CRASH" for d in dets):
    print("Traceback: stub chain crash"); sys.exit(1)
if any(d["key"] == "REFUSE" for d in dets):
    print("REFUSING: stub chain refusal reason, the 1-minute tier of 41MHz is paused")
    for i in range(60):
        print("WARNING noise line %d after the refusal" % i)
    sys.exit(2)
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
if os.environ.get("STUB_RM"):
    os.remove(os.environ["STUB_RM"])
if os.environ.get("STUB_TIE_FAIL"):
    print("Traceback: stub tie failure"); sys.exit(1)
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
'''

STUB_FIT = r'''
import csv, json, os, sys
STORE_COLS = "cell,subband,stokes,tier,ts,key,fit_status"


def load_store(path):
    out = {}
    if path and os.path.exists(path):
        for r in csv.DictReader(open(path)):
            out[r["key"]] = r
    return out


def template_store_path(p):
    return (p[:-4] if p.endswith(".csv") else p) + "_template.csv"


def _group_order(r):
    return (r["subband"], r["stokes"])


if __name__ == "__main__":
    a = sys.argv[1:]
    pend, store = a[a.index("--pending") + 1], a[a.index("--store") + 1]
    shard, nshards = int(a[a.index("--shard") + 1]), int(a[a.index("--nshards") + 1])
    with open(os.environ["STUB_CALLS"], "a") as f:
        f.write(json.dumps({"prog": "fit", "argv": a, "nice": os.nice(0), "env": {
            k: v for k, v in os.environ.items() if k.startswith("P3_") or k.endswith("_NUM_THREADS")}}) + "\n")
    rows = [r for r in csv.DictReader(open(pend)) if not r["key"].startswith("UNFITTABLE")]
    rows = [r for i, r in enumerate(sorted(rows, key=lambda r: r["key"])) if i % nshards == shard]
    if os.environ.get("STUB_STOP_AFTER_SHARD") and shard == 0:
        open(os.environ["STUB_STOP_AFTER_SHARD"], "w").close()
    new = not os.path.exists(store)
    with open(store, "a") as f:
        if new:
            f.write(STORE_COLS + "\n")
        for r in rows:
            f.write("%s,%s,%s,%s,%s,%s,ok\n" % (r["cell"], r["subband"], r["stokes"], r["tier"], r["ts"], r["key"]))
    print("fit-grouped DONE: %d fitted" % len(rows))
'''

# the store scripts of exopipe_ops: rows whose newest fit_status is 'raw' are what the content test refuses (and the
# repair lists); 'bad' is refused but never listed (a NOT-READY the repair cannot mend)
STUB_REPAIR = r'''
import csv, json, os, sys
sys.path.insert(0, os.environ["STUB_CLV"])
from phase3_snapshot_fit import load_store
a = [x for x in sys.argv[1:] if not x.startswith("--")]
cell, out = a[0], a[1]
S = sys.argv[sys.argv.index("--store") + 1] if "--store" in sys.argv else os.path.join(os.environ["STUB_BATCH"], "fits", "fits_%s.csv" % cell)
with open(os.environ["STUB_CALLS"], "a") as f:
    f.write(json.dumps({"prog": "repair", "argv": sys.argv[1:]}) + "\n")
if os.environ.get("STUB_REPAIR_FAIL"):
    print("Traceback: stub repair failure"); sys.exit(1)
rows = [r for r in load_store(S).values() if r["fit_status"] == "raw"]
with open(out, "w") as f:
    f.write("cell,subband,stokes,tier,ts,ra_deg,dec_deg,snr,key\n")
    for r in rows:
        f.write("%s,%s,%s,%s,%s,1.0,2.0,9.0,%s\n" % (r["cell"], r["subband"], r["stokes"], r["tier"], r["ts"], r["key"]))
print("REPAIR-LIST %s %d raw-frame %d far-without-template 0 no-source<8-without-template 0 -> %s" % (cell, len(rows), len(rows), out))
'''

STUB_APPEND = r'''
import csv, json, os, sys
a = sys.argv[1:]
F = a[a.index("--fits-dir") + 1] if "--fits-dir" in a else os.path.join(os.environ["STUB_BATCH"], "fits")
srcs = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] != "--fits-dir")]
with open(os.environ["STUB_CALLS"], "a") as f:
    f.write(json.dumps({"prog": "append", "argv": a}) + "\n")
n = 0
for src in srcs:
    for r in csv.DictReader(open(src)):
        p = os.path.join(F, "fits_%s.csv" % r["cell"])
        new = not os.path.exists(p)
        with open(p, "a") as f:
            if new:
                f.write("cell,subband,stokes,tier,ts,key,fit_status\n")
            f.write(",".join(r[k] for k in ("cell", "subband", "stokes", "tier", "ts", "key", "fit_status")) + "\n")
        n += 1
print("APPENDED %d rows, 0 template" % n)
'''

STUB_READY = r'''
import json, os, sys
sys.path.insert(0, os.environ["STUB_CLV"])
from phase3_snapshot_fit import load_store
bad_any = False
for cell in sys.argv[1:]:
    S = os.path.join(os.environ["STUB_BATCH"], "fits", "fits_%s.csv" % cell)
    if not os.path.exists(S):
        print("READY %s no store (nothing fitted)" % cell); continue
    st = load_store(S)
    raw = sum(1 for r in st.values() if r["fit_status"] == "raw")
    bad = sum(1 for r in st.values() if r["fit_status"] == "bad")
    ok = raw == 0 and bad == 0
    bad_any |= not ok
    print("%s %s rows %d: raw-frame ok fits %d, bad %d" % ("READY" if ok else "NOT-READY", cell, len(st), raw, bad))
sys.exit(1 if bad_any else 0)
'''


STUB_SEARCH = r'''
import os


def excluded_subbands(lst, date, path=None):
    return {x for x in os.environ.get("STUB_EXCLUDED", "").split(",") if x}
'''

# the module's own form (FIELDS_V1 = [...]; FIELDS = FIELDS_V1 + [...]): the current header is read from its source
STUB_TIECACHE = r'''
import os
FIELDS_V1 = ["cell", "sb", "st", "tier", "ts", "ra_pre", "dec_pre", "tie_status", "tie_n_refs", "tie_dx_arcmin",
             "tie_dy_arcmin", "sys_arcmin", "ra", "dec", "ra_untied", "dec_untied", "errmodel"]
FIELDS = FIELDS_V1 + ["tie_search_deg", "tie_ref_max_sep_deg", "sys_scatter_arcmin"]
'''
TIE_FIELDS = ['cell', 'sb', 'st', 'tier', 'ts', 'ra_pre', 'dec_pre', 'tie_status', 'tie_n_refs', 'tie_dx_arcmin',
              'tie_dy_arcmin', 'sys_arcmin', 'ra', 'dec', 'ra_untied', 'dec_untied', 'errmodel', 'tie_search_deg',
              'tie_ref_max_sep_deg', 'sys_scatter_arcmin']


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
                 excluded=(), flow_unregistered=(), dirty=False, forced='done', store_rows=()):
        self.dir = tempfile.mkdtemp(prefix='catalog_snapshot_incremental_')
        d = self.dir
        self.fleet, self.claims = f'{d}/fleet_out', f'{d}/fleet_claims'
        self.lc, self.batch, self.stub = f'{d}/lightcurves', f'{d}/batch', f'{d}/stub'
        self.out, self.work = f'{d}/out', f'{d}/scratch'
        self.calls = f'{d}/calls.jsonl'
        self.fst, self.ftc = f'{d}/forced_state', f'{d}/frametime_claims'
        self.hold = (f'{d}/hold_b2.txt', f'{d}/batch/prefit_hold_cells.txt')
        self.restore = f'{d}/restore.json'
        for p in (self.fleet, self.claims, self.lc, self.batch, f'{self.stub}/clv15', f'{self.stub}/exopipe_ops', self.work,
                  f'{self.fst}/done', f'{self.fst}/failed', f'{self.fst}/claims', self.ftc):
            os.makedirs(p, exist_ok=True)
        for h in self.hold:
            open(h, 'w').write('# hold\n04h 2025-01-01\n')
        open(self.restore, 'w').write('{"stub": 1}\n')
        open(f'{self.stub}/clv15/phase3_snapshot_chain.py', 'w').write(STUB_CHAIN)
        open(f'{self.stub}/clv15/phase3_snapshot_fit.py', 'w').write(STUB_FIT)
        open(f'{self.stub}/clv15/phase3_snapshot_search.py', 'w').write(STUB_SEARCH)
        open(f'{self.stub}/clv15/phase3_snapshot_tiecache.py', 'w').write(STUB_TIECACHE)
        for n, src in (('snapshot_store_repair.py', STUB_REPAIR), ('snapshot_store_append.py', STUB_APPEND),
                       ('snapshot_store_ready.py', STUB_READY)):
            open(f'{self.stub}/exopipe_ops/{n}', 'w').write(src)
        if forced:
            open(f'{self.fst}/{forced}/{CELL}', 'w').close()
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
        if fitted or store_rows:
            os.makedirs(f'{self.batch}/fits')
            with open(f'{self.batch}/fits/fits_{CELL}.csv', 'w') as f:
                f.write('cell,subband,stokes,tier,ts,key,fit_status\n')
                for k in fitted:
                    f.write(f'{CELL},18MHz,I,10s,{k},{k},ok\n')
                for k, st in store_rows:
                    f.write(f'{CELL},18MHz,I,10s,{k},{k},{st}\n')
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
            (CFG.catalog_snapshot, 'INC_FORCED_STATE_DIR', self.fst), (CFG.catalog_snapshot, 'INC_HOLD_LISTS', self.hold),
            (CFG.catalog_snapshot, 'INC_FRAMETIME_CLAIMS', self.ftc),
            (CFG.catalog_snapshot, 'INC_CURVE_CONSTANTS_RESTORE', self.restore),
            (CFG.site, 'PYTHON', {'default': sys.executable, 'py36': sys.executable}),
            (CFG.site, 'LEGACY_CODE_ROOTS', [f'{self.stub}/clv15', f'{self.stub}/exopipe_ops']),
            (EX, 'for_tier', lambda tier, **k: _Excl(self.excluded)),
            # room for exactly 2 fit workers (INC_FIT_MEM_MARGIN_GB + 2 x INC_FIT_GB_PER_WORKER): one worker count everywhere
            (NM, 'mem_available_gb', lambda *a, **k: CFG.catalog_snapshot.INC_FIT_MEM_MARGIN_GB
             + 2 * CFG.catalog_snapshot.INC_FIT_GB_PER_WORKER))
        self._prod = Patch((CFG.paths, 'PRODUCTION_ROOTS', tuple(CFG.paths.PRODUCTION_ROOTS) + (self.batch,)))

    def __enter__(self):
        self._patch.__enter__()
        self._prod.__enter__()
        os.environ.update(STUB_CALLS=self.calls, STUB_BATCH=self.batch, STUB_CLV=f'{self.stub}/clv15',
                          STUB_EXCLUDED=','.join(sb for lst, sb, date in self.excluded if f'{lst}_{date}' == CELL))
        return self

    def __exit__(self, *exc):
        self._prod.__exit__(*exc)
        self._patch.__exit__(*exc)
        for k in ('STUB_CALLS', 'STUB_BATCH', 'STUB_CLV', 'STUB_RM', 'STUB_EXCLUDED', 'STUB_TOUCH', 'STUB_TIE_FAIL',
                  'STUB_REPAIR_FAIL', 'STUB_STOP_AFTER_SHARD', record.UPSTREAM_ENV):
            os.environ.pop(k, None)
        shutil.rmtree(self.dir, ignore_errors=True)

    def args(self, **kw):
        a = argparse.Namespace(tier='snapshot', incremental=True, epochs=CELL, inc_phase=None, snap_inc_seed=None,
                               work=self.work)
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    def run(self, out=None, dry=False, prod=False, **kw):
        """A run into --out (default: a non-production directory; prod=True: the batch directory, a production root)."""
        out = out or (self.batch if prod else self.out)
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


def fails(root):
    """The kinds of the PREFIT_FAILS lines of the cell under a root (prefit.sh:63: '<time> <node> <kind>'), the v2 suffix cut."""
    p = f'{root}/cells/{CELL}/PREFIT_FAILS'
    if not os.path.exists(p):
        return []
    return [ln.split(' ', 2)[2].rsplit(' (exopipe3 catalog', 1)[0] for ln in open(p).read().splitlines() if ln.strip()]


# =====================================================================================================================
# tests
# =====================================================================================================================

def test_happy_path_and_associate_refusal():
    with Tree() as t:
        r, rc = t.run(prod=True)
        u = unit(r, 'cell')
        assert rc is None and u.status == 'done', (u.status, u.reason)
        cd = f'{t.batch}/cells/{CELL}'
        assert u.meta['repaired'] == 0 and len(t.calls_of('repair')) == 1 and '--check' in t.calls_of('repair')[0]['argv']
        assert '--store' not in t.calls_of('repair')[0]['argv']                    # production: the script's own store
        assert len(rows(f'{cd}/prefit1_requests.csv')) == 2 and len(rows(f'{cd}/prefit2_requests.csv')) == 0
        assert [x['round'] for x in u.meta['prefit_rounds']] == [1, 2] and u.meta['prefit_fitted'] == 2
        assert u.meta['store_ready'].startswith(f'READY {CELL}') and 'store_same_as_production' not in u.meta
        assert len(rows(f'{t.batch}/fits/fits_{CELL}.csv')) == 3            # production 1 + fitted 2
        assert len(rows(f'{t.batch}/tiecache/tie_{CELL}.csv')) == 2           # the two pool detections
        assert u.meta['tie_tied_now'] == 2 and u.meta['tie_from_cache'] == 0
        assert json.load(open(f'{cd}/chain_summary.json'))['tie_cache']['tied_now'] == 2
        assert 'tie cache' in open(f'{cd}/prestage.log').read()
        classes = sorted(o['class'] for o in u.outputs)
        assert classes == ['csv_table', 'csv_table', 'csv_table', 'snapshot_tie_summary', 'tie_cache'], classes
        assert not u.failed_tests(), u.failed_tests()
        assert not os.path.exists(f'{t.batch}/prefit_claims/{CELL}')        # claim released
        assert not [n for n in os.listdir(f'{t.batch}/prefit_claims') if n.startswith('.released')]
        assert not os.path.exists(f'{t.batch}/cells/{CELL}/PRESTAGE_DONE')   # v2 writes no production marker
        assert not os.path.exists(f'{t.dir}/prodwork/prestage/{CELL}')      # prestage.sh:86 removes the work
        ch = t.calls_of('chain')
        assert len(ch) == 3 and '--prefit-list-only' in ch[0]['argv'] and '--stop-after-tie' in ch[2]['argv']
        for c in ch:
            a = c['argv']
            assert a[a.index('--fit-store') + 1] == f'{t.batch}/fits/fits_{CELL}.csv'
            assert a[a.index('--veto-record-root') + 1] == '' and a[a.index('--cells') + 1] == CELL
            assert a[a.index('--lightcurves-root') + 1] == t.lc
            assert a[a.index('--curve-constants-restore') + 1] == t.restore     # the restore-only 3-D gate table
            assert 'P3_SNAP_REFASTRO_SCRATCH' not in c['env']
        tie_env = ch[2]['env']
        assert tie_env['P3_TIE_CACHE_DIR'] == f'{t.batch}/tiecache'
        assert tie_env['P3_REFASTRO_SHARDS'] == str(CFG.catalog_snapshot.TIE_SHARDS[2])
        assert 'P3_TIE_CACHE_DIR' not in ch[0]['env']
        fit = t.calls_of('fit')
        assert len(fit) == 1 and fit[0]['env']['OMP_NUM_THREADS'] == '1'
        fa = fit[0]['argv']
        assert fa[fa.index('--store') + 1] == f'{t.batch}/fits/fits_{CELL}.csv'
        assert fa[fa.index('--procs') + 1] == '2' and fa[fa.index('--shard') + 1] == '0' and fa[fa.index('--nshards') + 1] == '1'
        assert fit[0]['nice'] == min(19, os.nice(0) + CFG.catalog_snapshot.INC_FIT_NICE)    # prefit.sh:32 nice -n 19
        assert not os.path.exists(f'{cd}/PREFIT_FAILS') and 'prefit_fails_appended' not in u.meta
        a = unit(r, 'associate')
        assert a.status == 'refused' and a.reason.startswith('SI-G9: ') and all(f'({i})' in a.reason for i in (1, 2, 3, 4))
        assert 'The cell phase is done' in a.reason and a.meta['cell'] == CELL and u.meta['cell'] == CELL
        assert r.exit_code() == record.EXIT_REFUSED
        rec = record.load_record(t.batch, 'catalog', f'{IS.UNIT}.cell.{CELL}')
        pl = rec['meta']['plan']
        assert rec['status'] == 'done' and pl[0].startswith('SI-G13 stop files') and pl[1].startswith('SI-G12 attempt cap: 0')
        assert pl[2].startswith('SI-G2 dirty Stokes I cell: False'), pl
        # rerun: every pool detection from the cache, nothing fitted
        r2, _ = t.run(prod=True)
        u2 = unit(r2, 'cell')
        assert u2.status == 'done' and u2.meta['tie_tied_now'] == 0 and u2.meta['tie_from_cache'] == 2, u2.meta
        assert u2.meta['prefit_fitted'] == 0 and len(rows(f'{t.batch}/tiecache/tie_{CELL}.csv')) == 2


def test_done_by_production():
    with Tree() as t:
        os.makedirs(f'{t.batch}/claims/{CELL}')                                # production's claim stays (owner = node)
        open(f'{t.batch}/claims/{CELL}/owner', 'w').write('lwacalim99\n')
        r, _ = t.run(prod=True)                                                 # without PRESTAGE_DONE: an old stage holds it
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G4: ') and 'old per-cell stage' in u.reason, u.reason
        os.makedirs(f'{t.batch}/cells/{CELL}')
        open(f'{t.batch}/cells/{CELL}/PRESTAGE_DONE', 'w').write('lwacalim99 Fri Oct 3 prefit_fitted=4 forced=done\n')
        # PRESTAGE_DONE without a tie-cache file of the current header (11 of production's 111 such cells): the unit runs
        r, _ = t.run(prod=True)
        u = unit(r, 'cell')
        assert u.status == 'done' and 'no tie_ or tie2_ file' in u.meta['prestage_done_not_valid'], (u.status, u.meta)
        assert len(t.calls_of('chain')) == 3 and 'PRESTAGE_DONE' in u.meta['plan'][0]
        r, _ = t.run(prod=True)                                                 # the stub's tie_ file: an older header
        u = unit(r, 'cell')
        assert u.status == 'done' and 'older header, 11 columns' in u.meta['prestage_done_not_valid'], u.meta
        os.remove(t.calls)
        with open(f'{t.batch}/tiecache/tie2_{CELL}.csv', 'w', newline='') as f:   # the current header (tiecache.FIELDS)
            csv.writer(f, lineterminator='\r\n').writerow(TIE_FIELDS)
        r, rc = t.run(prod=True, dry=True)                                      # a dry run prints it, records nothing
        assert not r.units and not t.calls_of('chain')
        before = sorted(os.listdir(t.batch))
        r, rc = t.run(prod=True)
        u = unit(r, 'cell')
        assert u.status == 'skipped-valid' and u.reason.startswith('done by production: ') and 'prefit_fitted=4' in u.reason
        assert f'tiecache/tie2_{CELL}.csv (current header)' in u.reason, u.reason
        assert u.meta['done_by'] == 'production' and not t.calls_of('chain') and not t.calls_of('repair')
        rec = record.load_record(t.batch, 'catalog', f'{IS.UNIT}.cell.{CELL}')
        assert rec['status'] == 'skipped-valid'
        assert open(f'{t.batch}/claims/{CELL}/owner').read() == 'lwacalim99\n'   # production's files untouched
        assert set(os.listdir(t.batch)) - {'_records'} == set(before) - {'_records'}
        assert unit(r, 'associate').status == 'refused' and 'done by production' in unit(r, 'associate').reason
        # a non-production --out does not read production's markers: it runs the cell
        r, _ = t.run()
        assert unit(r, 'cell').status != 'skipped-valid' and len(t.calls_of('chain')) > 0


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


def test_upstream_refused():
    with Tree() as t:
        os.environ[record.UPSTREAM_ENV] = json.dumps([{'cell': CELL, 'subband': None, 'step': 'curves --tier snapshot',
                                                       'code': 'C-G3'}])
        r, _ = t.run(prod=True)
        for ph in ('cell', 'associate'):
            u = unit(r, ph)
            assert u.status == 'refused' and u.reason.startswith(record.UPSTREAM_CODE) and u.meta['cell'] == CELL, u.reason
        assert not t.calls_of('chain') and not t.calls_of('repair') and not os.path.exists(f'{t.batch}/prefit_claims/{CELL}')


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
    # an excluded subband needs no table (curves plan_cell); a flow-unregistered one stops at SI-G14 first (prefit.sh:82-95)
    lst, date = CELL.split('_', 1)
    with Tree(excluded=[(lst, '41MHz', date)]) as t:
        r, _ = t.run(prod=True)
        u = unit(r, 'cell')
        assert u.status == 'done', u.reason
        assert any('not required: 41MHz (excluded)' in x for x in u.meta['plan']), u.meta['plan']


def test_gate_forced():
    with Tree(forced='failed') as t:                                            # prefit.sh:74 / prestage.sh:61
        r, _ = t.run()
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G8: ') and 'marked failed' in u.reason, u.reason
        assert not t.calls_of('chain') and not os.path.exists(f'{t.out}/prefit_claims/{CELL}')
    with Tree(forced=None) as t:
        os.makedirs(f'{t.fst}/claims/{CELL}')
        r, _ = t.run()
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G8: ') and 'a forced fleet holds it' in u.reason, u.reason
        assert 'photometry --tier snapshot --epochs' in u.reason and not t.calls_of('chain')
    with Tree() as t:                                                           # prestage.sh:83: done again at the end
        os.environ['STUB_RM'] = f'{t.fst}/done/{CELL}'
        r, _ = t.run(prod=True)
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G8: ') and 'at the end' in u.reason, u.reason
        assert len(t.calls_of('chain')) == 3 and not os.path.exists(f'{t.batch}/prefit_claims/{CELL}')
        assert fails(t.batch) == ['forced photometry stage 1'] and u.meta['prefit_fails_appended'], fails(t.batch)


def test_gate_holds_and_claim():
    lst, date = CELL.split('_', 1)
    with Tree() as t:
        for i, how in enumerate(('list0', 'list1', 'active', 'frametime')):
            if how.startswith('list'):
                p = t.hold[int(how[-1])]
                open(p, 'a').write(f'{CELL}\n')
            elif how == 'active':
                os.makedirs(f'{t.batch}/rechain_active', exist_ok=True)
                p = f'{t.batch}/rechain_active/Run_SNAPX.cells'
                open(p, 'w').write(f'{lst} {date}\n')
            else:
                p = f'{t.ftc}/{CELL}'
                os.makedirs(p)
            r, _ = t.run()
            u = unit(r, 'cell')
            assert u.status == 'refused' and u.reason.startswith('SI-G4: ') and p in u.reason, (how, u.reason)
            if how == 'frametime':
                open(f'{p}/done', 'w').close()                                  # a finished catch-up keeps its claim
            elif how.startswith('list'):
                open(p, 'w').write('# hold\n')
            else:
                os.remove(p)
        assert not t.calls_of('chain')
        r, _ = t.run(prod=True)
        assert unit(r, 'cell').status == 'done', unit(r, 'cell').reason
    with Tree() as t:
        cd = f'{t.batch}/prefit_claims/{CELL}'
        node = socket.gethostname().split('.')[0]
        os.makedirs(cd)
        open(f'{cd}/owner', 'w').write(f'{node}:{os.getpid()}\n')               # a live writer
        r, _ = t.run(prod=True)
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G4: ') and 'claimed by' in u.reason, u.reason
        assert os.path.isdir(cd) and not t.calls_of('chain')                     # another writer's claim is kept
        os.remove(f'{cd}/owner')                                                 # owner-less, fresh: still a claim
        r, _ = t.run(prod=True)
        assert unit(r, 'cell').status == 'refused' and os.path.isdir(cd)
        old = time.time() - CFG.catalog_snapshot.INC_OWNERLESS_CLAIM_STALE_S - 60  # owner-less and old: stale
        os.utime(cd, (old, old))
        r, _ = t.run(prod=True)
        assert unit(r, 'cell').status == 'done' and not os.path.exists(cd), unit(r, 'cell').reason
        p = subprocess.Popen([sys.executable, '-c', 'pass'])
        p.wait()
        os.makedirs(cd)
        open(f'{cd}/owner', 'w').write(f'{node}:{p.pid}\n')                     # a dead writer on this node
        r, _ = t.run(prod=True)
        assert unit(r, 'cell').status == 'done' and not os.path.exists(cd)
        os.makedirs(cd)
        open(f'{cd}/owner', 'w').write(f'othernode:{p.pid}\n')                  # another node's: never judged here
        r, _ = t.run(prod=True)
        assert unit(r, 'cell').status == 'refused' and os.path.isdir(cd)


def test_production_root_layout():
    with Tree() as t:
        try:                                                                    # seeding is for a non-production --out
            t.run(prod=True, snap_inc_seed='store')
            raise AssertionError('--snap-inc-seed was accepted on the production root')
        except record.Refusal as e:
            assert str(e).startswith('SI-G7: '), e
        other = f'{t.dir}/otherprod'
        with Patch((CFG.paths, 'PRODUCTION_ROOTS', tuple(CFG.paths.PRODUCTION_ROOTS) + (other,))):
            try:
                t.run(out=other)
                raise AssertionError('another production root was accepted')
            except record.Refusal as e:
                assert str(e).startswith('SI-G4: ')
        os.remove(t.restore)
        try:
            t.run(prod=True)
            raise AssertionError('a missing restore table was accepted')
        except record.Refusal as e:
            assert str(e).startswith('SI-G11: '), e


def test_store_repair_and_ready():
    with Tree(fitted=('d1', 'd2', 'd3'), store_rows=(('d1', 'raw'),)) as t:   # d1's newest row: a raw-frame fit
        r, _ = t.run(prod=True)
        u = unit(r, 'cell')
        assert u.status == 'done' and u.meta['repaired'] == 1, (u.status, u.reason)
        ap = t.calls_of('append')
        assert len(ap) == 1 and ap[0]['argv'][ap[0]['argv'].index('--fits-dir') + 1] == f'{t.batch}/fits'
        assert len(t.calls_of('repair')) == 2 and u.meta['store_ready'].startswith('READY')    # round 2 lists none
    with Tree(fitted=('d1', 'd2', 'd3'), store_rows=(('d2', 'bad'),)) as t:   # refused by the test, not repairable
        r, _ = t.run(prod=True)
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G10: ') and 'NOT-READY' in u.reason, u.reason
        assert os.path.exists(f'{t.batch}/cells/{CELL}/store_ready.log') and len(t.calls_of('chain')) == 1   # no tie pass
        f_ = fails(t.batch)
        assert len(f_) == 1 and f_[0].startswith(f'store test: NOT-READY {CELL} rows 3'), f_          # prefit.sh:168


def test_gate_not_converged():
    with Tree(dets=(('d1', 1), ('UNFITTABLE1', 1))) as t:
        r, _ = t.run(prod=True)
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G5: ') and 'after 3 prefit rounds' in u.reason, u.reason
        assert len(t.calls_of('fit')) == 3 and not os.path.exists(f'{t.batch}/tiecache')
        f_ = fails(t.batch)
        assert len(f_) == 1 and f_[0].startswith('prefit list not empty after 3 iterations or a step failed ('), f_


def test_gate_chain_failure():
    with Tree(dets=(('CRASH', 1),)) as t:
        r, _ = t.run()
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G6: ') and 'prefit list 1' in u.reason, u.reason
        assert os.path.exists(f'{t.out}/cells/{CELL}/prefit1.log')
        assert not os.path.exists(f'{t.out}/prefit_claims/{CELL}')
        assert not os.path.exists(f'{t.out}/cells/{CELL}/PREFIT_FAILS')        # a non-production root records no attempt
    with Tree(dets=(('REFUSE', 1),)) as t:          # the reason quotes the chain's refusal line, not the noise after it
        r, _ = t.run(prod=True)
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G6: ') and 'exit 2' in u.reason, u.reason
        assert 'REFUSING: stub chain refusal reason, the 1-minute tier of 41MHz is paused' in u.reason, u.reason
        assert 'noise line' not in u.reason.split('REFUSING')[0]
        f_ = fails(t.batch)
        assert len(f_) == 1 and f_[0].startswith('prefit list not empty after 3 iterations or a step failed'), f_


def test_no_input():
    with Tree(dets=(), fitted=()) as t:
        r, _ = t.run()
        u = unit(r, 'cell')
        assert u.status == 'no-input' and 'per-frame fit step' in u.reason, u.reason
        assert unit(r, 'associate').status == 'refused' and r.exit_code() == record.EXIT_REFUSED
    with Tree(dets=(('d1', 0),)) as t:
        r, _ = t.run(prod=True)
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
        assert u.status == 'done' and u.meta['tie_from_cache'] == 2 and u.meta['tie_tied_now'] == 0, (u.status, u.reason)
        assert u.meta['prefit_fitted'] == 0 and len(u.meta['seeded']) == 3, u.meta.get('seeded')
        assert u.meta['store_same_as_production'].startswith('STORE-SAME') and u.meta['store_ready'].startswith('READY')
        assert sorted(o['class'] for o in u.outputs).count('tie_cache') == 2 and not u.failed_tests(), u.failed_tests()
        assert t.calls_of('repair')[0]['argv'][t.calls_of('repair')[0]['argv'].index('--store') + 1] == \
            f'{t.out}/fits/fits_{CELL}.csv'
        for c in t.calls_of('chain'):                                           # refastro scratch off the production roots
            assert CFG.production_root_of(c['env']['P3_SNAP_REFASTRO_SCRATCH']) is None
        assert os.path.exists(f'{t.out}/_work/{CELL}/tie/chain_summary.json')   # non-production keeps the work
        # an existing copy is kept: the production file changes, the run's copy does not - and no longer equals it
        open(f'{t.batch}/fits/fits_{CELL}.csv', 'a').write(f'{CELL},18MHz,I,10s,x9,x9,ok\n')
        r, _ = t.run()
        u = unit(r, 'cell')
        assert not u.meta.get('seeded') and len(rows(f'{t.out}/fits/fits_{CELL}.csv')) == 3
        assert u.status == 'refused' and u.reason.startswith('SI-G10: ') and 'STORE-DIFFERS' in u.reason, u.reason
    with Tree() as t:                                                           # fitted rows production does not hold
        r, _ = t.run(snap_inc_seed='none')
        u = unit(r, 'cell')
        assert u.meta['prefit_fitted'] == 3 and not u.meta.get('seeded')
        assert u.status == 'refused' and u.reason.startswith('SI-G10: '), u.reason


def test_attempt_cap():
    """SI-G12 (prefit.sh:71-72; Gregg 2026-10-10 'Go on all 3'): >= INC_PREFIT_MAX_FAILS lines park the cell on any root."""
    lim = CFG.catalog_snapshot.INC_PREFIT_MAX_FAILS
    assert lim == 3
    with Tree() as t:
        for root, prod in ((t.out, False), (t.batch, True)):
            os.makedirs(f'{root}/cells/{CELL}', exist_ok=True)
            open(f'{root}/cells/{CELL}/PREFIT_FAILS', 'w').write(''.join(f'2026-10-10T0{i}:00:00Z n{i} store test\n'
                                                                          for i in range(lim)) + '\n')
            r, _ = t.run(prod=prod)
            u = unit(r, 'cell')
            assert u.status == 'refused' and u.reason.startswith('SI-G12: ') and f'{lim} failed attempts' in u.reason, u.reason
            assert not t.calls_of('chain') and not os.path.exists(f'{root}/prefit_claims/{CELL}')
            assert len(fails(root)) == lim                                     # a parked cell gains no line
        open(f'{t.batch}/cells/{CELL}/PREFIT_FAILS', 'w').write('2026-10-10T00:00:00Z n0 store test\n' * (lim - 1))
        r, _ = t.run(prod=True)
        assert unit(r, 'cell').status == 'done' and len(fails(t.batch)) == lim - 1, unit(r, 'cell').reason


def test_attempt_record_repair_tie_and_stop():
    """prefit.sh:63 failrec: the store repair counts (its logs kept); the tie pass does not (prefit.sh has none); nothing is
    recorded while a stop file exists or on a non-production root."""
    with Tree(fitted=('d1', 'd2', 'd3'), store_rows=(('d1', 'raw'),)) as t:
        os.environ['STUB_REPAIR_FAIL'] = '1'
        r, _ = t.run(prod=True)
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G6: ') and 'store repair list 1' in u.reason, u.reason
        assert 'stub repair failure' in u.reason
        f_ = fails(t.batch)
        assert len(f_) == 1 and f_[0] == f'store repair ({t.batch}/cells/{CELL}/repair*.log)', f_
        assert os.path.exists(f'{t.batch}/cells/{CELL}/repair1.log') and not t.calls_of('chain')
        r, _ = t.run()                                                          # non-production: no line
        assert unit(r, 'cell').status == 'refused' and not fails(t.out)
    with Tree() as t:
        os.environ['STUB_TIE_FAIL'] = '1'
        r, _ = t.run(prod=True)
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G6: ') and 'tie pass' in u.reason, u.reason
        assert not fails(t.batch)
    with Tree(fitted=('d1', 'd2', 'd3'), store_rows=(('d2', 'bad'),)) as t:   # a stop file appears during the attempt
        node = socket.gethostname().split('.')[0]
        os.environ['STUB_TOUCH'] = f'{t.batch}/PREFIT_STOP_{node}'
        r, _ = t.run(prod=True)
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G10: '), u.reason
        assert not fails(t.batch) and 'prefit_fails_appended' not in u.meta


def test_stop_files_and_fit_shards():
    """SI-G13 (prefit.sh:24, :68, :136): PREFIT_STOP / PREFIT_STOP_<node> under --out before each cell and between fit shards;
    prestage.sh's STOP_<node> is not honored. Shards: ceil(requests / INC_FIT_SHARD_REQUESTS), at most groups // (tasks per
    worker x workers) (prefit.sh:124-134)."""
    node = socket.gethostname().split('.')[0]
    with Tree() as t:
        for root, prod, name in ((t.out, False, 'PREFIT_STOP'), (t.batch, True, f'PREFIT_STOP_{node}')):
            os.makedirs(root, exist_ok=True)
            open(f'{root}/{name}', 'w').close()
            r, _ = t.run(prod=prod)
            u = unit(r, 'cell')
            assert u.status == 'refused' and u.reason.startswith('SI-G13: ') and name in u.reason, u.reason
            assert 'not a failed attempt' in u.reason and not t.calls_of('chain') and not fails(root)
            os.remove(f'{root}/{name}')
        open(f'{t.batch}/STOP_{node}', 'w').close()                             # stale prestage.sh stop files exist
        open(f'{t.batch}/PREFIT_STOP_othernode', 'w').close()
        r, _ = t.run(prod=True)
        assert unit(r, 'cell').status == 'done', unit(r, 'cell').reason
    dets = tuple((f'd{i}', 1) for i in range(1, 6))
    c = CFG.catalog_snapshot
    with Tree(dets=dets) as t, Patch((c, 'INC_FIT_SHARD_REQUESTS', 1), (c, 'INC_FIT_TASKS_PER_WORKER', 1)):
        r, _ = t.run(prod=True)                                                 # 4 requests, 4 groups, 2 workers: 2 shards
        u = unit(r, 'cell')
        assert u.status == 'done' and u.meta['prefit_fitted'] == 4, (u.status, u.reason)
        fa = [x['argv'] for x in t.calls_of('fit')]
        assert [(a[a.index('--shard') + 1], a[a.index('--nshards') + 1]) for a in fa] == [('0', '2'), ('1', '2')], fa
        assert len(rows(f'{t.batch}/fits/fits_{CELL}.csv')) == 5
    with Tree(dets=dets) as t, Patch((c, 'INC_FIT_SHARD_REQUESTS', 1), (c, 'INC_FIT_TASKS_PER_WORKER', 1)):
        os.environ['STUB_STOP_AFTER_SHARD'] = f'{t.batch}/PREFIT_STOP'          # a stop between the shards
        r, _ = t.run(prod=True)
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G13: ') and 'during the fits of prefit round 1' in u.reason
        assert len(t.calls_of('fit')) == 1 and not fails(t.batch) and not os.path.exists(f'{t.batch}/prefit_claims/{CELL}')
    # the worker count: free cores less FIT_PROCS_RESERVE, at most the memory's, >= 2; waits while memory holds < 2
    seq = iter([c.INC_FIT_MEM_MARGIN_GB, c.INC_FIT_MEM_MARGIN_GB + 30 * c.INC_FIT_GB_PER_WORKER])
    notes = []
    ctx = argparse.Namespace(c=c)
    s_ = argparse.Namespace(u=argparse.Namespace(note=notes.append))
    with Patch((NM, 'mem_available_gb', lambda *a, **k: next(seq)), (c, 'INC_FIT_MEM_POLL_S', 0.01)):
        assert IS.fit_procs(ctx, s_) == max(2, min(len(os.sched_getaffinity(0)) - c.FIT_PROCS_RESERVE, 30))
    assert notes and notes[0].startswith('fit workers: waited'), notes
    with Patch((NM, 'mem_available_gb', lambda *a, **k: None)):
        assert IS.memprocs(ctx) == 0


def test_gate_chain_refuses():
    """SI-G14 (prefit.sh:82-95): flow-unregistered or paused 1-minute searches, and candidate tables of a subband not excluded
    without their 1-minute table, wait before the claim - no chain call, no failed attempt."""
    lst, date = CELL.split('_', 1)
    with Tree(flow_unregistered=('73MHz',)) as t:
        r, _ = t.run(prod=True)
        u = unit(r, 'cell')
        assert u.status == 'refused' and u.reason.startswith('SI-G14: ') and 'runinfo_73MHz_I_median.json flow-unregistered' \
            in u.reason, u.reason
        assert not t.calls_of('chain') and not fails(t.batch) and not os.path.exists(f'{t.batch}/prefit_claims/{CELL}')
    with Tree(excluded=[(lst, '18MHz', date)]) as t:
        cf = f'{t.fleet}/{CELL}'
        ri = json.load(open(f'{cf}/runinfo_41MHz_V_median.json'))
        json.dump(dict(ri, tier_1min='Paused '), open(f'{cf}/runinfo_41MHz_V_median.json', 'w'))
        r, _ = t.run()
        u = unit(r, 'cell')
        assert u.status == 'refused' and 'runinfo_41MHz_V_median.json 1-minute tier paused' in u.reason, u.reason
        json.dump(ri, open(f'{cf}/runinfo_41MHz_V_median.json', 'w'))
        open(f'{cf}/candidates_73MHz_I_median.csv', 'w').write('x\n')
        open(f'{cf}/candidates_18MHz_I_median.csv', 'w').write('x\n')           # excluded: no 1-minute table needed
        r, _ = t.run()
        u = unit(r, 'cell')
        assert u.status == 'refused' and 'candidates_73MHz_I_median.csv without its 1-minute table' in u.reason, u.reason
        assert 'candidates_18MHz' not in u.reason
        open(f'{cf}/candidates_73MHz_I_median_1min.csv', 'w').write('x\n')
        r, _ = t.run(prod=True)
        assert unit(r, 'cell').status == 'done', unit(r, 'cell').reason
        assert not [n for n in os.listdir(f'{t.dir}/prodwork/prestage') if n.startswith('.precheck')]


def test_call_limits():
    """The prefit-list passes run under INC_PREFIT_TIMEOUT_H (prefit.sh:153 PREFIT_TIMEOUT 6h), the tie pass under
    INC_CHAIN_TIMEOUT_H (prestage.sh:62); fits at niceness INC_FIT_NICE (prefit.sh:32)."""
    seen = []
    real = TSL.call

    def rec(ctx, s, target, args, python='default', env=None, log=None, cwd=None, timeout=None, nice=None):
        seen.append((os.path.basename(target), args[:1] + [a for a in args if a in ('--prefit-list-only', '--stop-after-tie')],
                     timeout, nice))
        return real(ctx, s, target, args, python, env, log, cwd, timeout, nice)
    with Tree() as t, Patch((TSL, 'call', rec)):
        r, _ = t.run(prod=True)
        assert unit(r, 'cell').status == 'done', unit(r, 'cell').reason
    c = CFG.catalog_snapshot
    assert c.INC_PREFIT_TIMEOUT_H == 6 and c.INC_FIT_NICE == 19
    chain = [x for x in seen if x[0] == 'phase3_snapshot_chain.py']
    assert [x[2] for x in chain if '--prefit-list-only' in x[1]] == [6 * 3600] * 2, chain
    assert [x[2] for x in chain if '--stop-after-tie' in x[1]] == [c.INC_CHAIN_TIMEOUT_H * 3600], chain
    fit = [x for x in seen if x[0] == 'phase3_snapshot_fit.py']
    assert len(fit) == 1 and fit[0][3] == 19 and all(x[3] is None for x in seen if x not in fit), seen
    assert [x[1][0] for x in seen if x[0] == 'incremental_snapshot.py'] == ['chain-refuses', 'fit-groups'], seen


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
        open(p, 'w').write('# 01h 2025-01-01 commented\n12h 2024-12-21 note\n13h_2025-01-02\n')
        assert IS.dirty_cells(p) == {'12h_2024-12-21', '13h_2025-01-02'} and IS.dirty_cells(f'{d}/none.txt') == set()
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
        # the stale-claim rule
        ctx = argparse.Namespace(node='nodeA', c=argparse.Namespace(INC_OWNERLESS_CLAIM_STALE_S=600))
        cd = f'{d}/claim'
        os.makedirs(cd)
        assert IS.stale_claim(ctx, cd) is None                                   # owner-less, fresh
        assert 'no owner file' in IS.stale_claim(ctx, cd, now=time.time() + 700)
        open(f'{cd}/owner', 'w').write(f'nodeA:{os.getpid()}\n')
        assert IS.stale_claim(ctx, cd) is None
        open(f'{cd}/owner', 'w').write('nodeB:1\n')
        assert IS.stale_claim(ctx, cd) is None                                   # another node's claim
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # tier_snapshot.chain_args / chain_env (module level, shared with the batch Ctx)
    c = CFG.catalog_snapshot
    a = TS.chain_args(CFG, c, '/x/fits_c.csv', '/nonexistent/phase3_snapshot_chain.py')
    assert a[:10] == ['--proto-root', CFG.paths.SNAPSHOT_FLEET_OUT, '--master-catalogs', c.MASTER_CATALOGS,
                      '--curve-constants', c.CURVE_CONSTANTS, '--fit-store', '/x/fits_c.csv', '--lightcurves-root',
                      c.LIGHTCURVES_ROOT]
    assert a[a.index('--dirty-cells') + 1] == c.DIRTY_CELLS and '--curve-constants-restore' not in a
    b = TS.chain_args(CFG, c, 's', '/nonexistent', restore='/r.json', analysis=[('no-tie', '')])
    assert b[b.index('--curve-constants-restore') + 1] == '/r.json' and b[-1] == '--no-tie'
    d_ = TS.chain_args(CFG, c, 's', '/nonexistent', defaults={'--dirty-cells': c.DIRTY_CELLS})
    assert '--dirty-cells' not in d_                                              # a cached default equal to config
    e = TS.chain_env(c, '/scratch/x', P3_TIE_CACHE_DIR='/t', P3_REFASTRO_SHARDS=8)
    assert e['P3_SNAP_REFASTRO_SCRATCH'] == '/scratch/x' and e['P3_REFASTRO_SHARDS'] == '8'
    assert ('P3_SNAP_FIT_RAW' in e) == (not c.FLOW_MODE) and 'P3_SNAP_REFASTRO_SCRATCH' not in TS.chain_env(c)
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


TESTS = [test_happy_path_and_associate_refusal, test_done_by_production, test_associate_alone_and_dry_run,
         test_upstream_refused, test_gate_search_shipped, test_gate_dirty, test_gate_lightcurves, test_gate_forced,
         test_gate_holds_and_claim, test_production_root_layout, test_store_repair_and_ready, test_gate_not_converged,
         test_gate_chain_failure, test_no_input, test_seeding, test_attempt_cap, test_attempt_record_repair_tie_and_stop,
         test_stop_files_and_fit_shards, test_gate_chain_refuses, test_call_limits, test_helpers, test_step_routes_here]


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
