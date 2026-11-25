"""catalog --tier snapshot --incremental --epochs <LST>_<date>[,...] - the per-hour catalog step of the 10-second and 1-minute
(snapshot) tiers. Only the RULED part runs; the rest refuses.

RULED AND IMPLEMENTED - phase 'cell': the PER-CELL PRE-CHAIN STAGE of ruling 2026-09-18
(RULING-INCREMENTAL-PUBLISHING-TIE-AS-A-PER-CELL-STAGE-JUDGEMENT-CARRY-OVER; Gregg 2026-09-25 "please do", 2026-09-26
"Proceed!"), production ops/snapshot_cell_prestage.sh as its live successor ops/snapshot_cell_prefit.sh (2026-10-08/10) runs
it - prefit.sh where the two differ - plus the tie pass, for ONE night cell:
  done      on the production root, a cell production already prestaged (BATCH_DIR/cells/<cell>/PRESTAGE_DONE,
            prestage.sh:31) AND whose tie-cache file (tie_ or tie2_) has the current header (phase3_snapshot_tiecache.FIELDS)
            is recorded 'skipped-valid' (done by production; read only) and not run again; otherwise the unit runs
  gates     SI-G13 (stop files), SI-G12 (the attempt cap), SI-G1..SI-G4, SI-G8, SI-G14 (below;
            snapshot_cell_prefit.sh:68-121), in prefit.sh's order
  repair    snapshot_store_repair.py: the rows the store content test refuses, refitted by fit-grouped into a new store
            and appended (at most catalog_snapshot.INC_STORE_REPAIR_ROUNDS rounds; snapshot_cell_prefit.sh:141-152)
  prefit    FIT EVERY DETECTION BEFORE THE VETOES (Gregg 2026-10-07 Q6): at most catalog_snapshot.PREFIT_MAX_ITER rounds of
            clv15/phase3_snapshot_chain.py --cells <cell> --prefit-list-only (every detection that reaches the per-frame fit
            step without its own fit -> prefit_requests.csv), then clv15/phase3_snapshot_fit.py fit-grouped of that list into
            the cell's own fit store, until the list is empty (prestage.sh:62-78)
  store     snapshot_store_ready.py prints READY for the cell (SI-G10; snapshot_cell_prefit.sh:167-168)
  tie       phase3_snapshot_chain.py --cells <cell> --stop-after-tie with the tie cache on (P3_TIE_CACHE_DIR): every
            association-pool detection is tied at its fitted position and appended to the cell's tie-cache file; cached
            detections are not tied again (prestage.sh:79-80; phase3_snapshot_tiecache.py: the tie of a detection depends only
            on the detection and its cell's frames and Master references, so it does not depend on which other cells or
            nights a run holds)
  forced    forced photometry stage 1 of the cell still done (SI-G8 again; prestage.sh:83) before the unit is done
  attempts  on the production root every failed attempt after the claim (forced stage 1 at the end, store repair, prefit
            loop, store test) appends one line to BATCH_DIR/cells/<cell>/PREFIT_FAILS, none under a stop file; a cell with
            catalog_snapshot.INC_PREFIT_MAX_FAILS lines is parked (SI-G12) on every root (prefit.sh:25-28, :62-63, :71-72)
  fits      fit-grouped in shards (prefit.sh:124-140): the stop files are checked between shards; workers capped by the
            node's memory (INC_FIT_GB_PER_WORKER, INC_FIT_MEM_MARGIN_GB) at niceness INC_FIT_NICE (prefit.sh:20-21, :40-55);
            memory for fewer than 2 workers refuses SI-G15 at once (prefit.sh's np waits up to 6 h, then runs 2)
  Every in-place module runs IN PLACE through core/legacy.run under the site interpreter 'py36', with the chain arguments
  and environment of the batch port (tier_snapshot.chain_args with the restore-only 3-D gate table
  catalog_snapshot.INC_CURVE_CONSTANTS_RESTORE, tier_snapshot.chain_env: what the batch driver passes).

NOT RULED - phase 'associate' REFUSES (SI-G9) on every hour: the cross-cell association, anchor, routes, vetoes, nulls and
the combined view of a new cell against the cumulative catalog. Four open choices (OPEN_CHOICES) have no ruling; until
they do, a cell enters the served catalog through the batch chain (`catalog --tier snapshot --run Run_SNAP...`, which
reads this phase's tie-cache rows) and the combined-view rebuild. The refusal is a CELL refusal (exit 2 with unit meta cell,
below): the per-hour chain goes on with its other cells and steps, and the item ends 'cells-refused', never complete.

Unit of work: the epoch item <LST>_<date> = one snapshot night cell. Units 'incremental.snapshot.cell.<cell>' and
'incremental.snapshot.associate.<cell>', each with meta cell=<cell>: a refusal is a CELL refusal of the per-hour chain
(core/runner.py), and a cell an earlier step of the chain item refused is recorded UPSTREAM-REFUSED without opening the
unit (run.upstream_refused). Phases: --inc-phase (default catalog_snapshot.INC_PHASES_DEFAULT).

Inputs:  paths.SNAPSHOT_FLEET_OUT/<cell>/ (candidate tables, admission and runinfo files); catalog_snapshot.MASTER_CATALOGS,
         CURVE_CONSTANTS, INC_CURVE_CONSTANTS_RESTORE, LIGHTCURVES_ROOT, DIRTY_CELLS; curves.FLEET_CLAIMS_ROOT (G1 evidence in
         the window); the hold lists (INC_HOLD_LISTS, BATCH_DIR/rechain_active/*.cells, INC_FRAMETIME_CLAIMS), the forced
         stage-1 markers (INC_FORCED_STATE_DIR) and the v2 stage-1 measure records (photometry.OUT_ROOT['snapshot'] and,
         off production, --out: <root>/_records/photometry/measure.<cell>*.json), read only; data_decisions.csv (core.exclusions for the light-curve plan;
         the chain applies it in place).
Outputs (under --out; the production root is catalog_snapshot.BATCH_DIR, the layout of prestage.sh:11-12):
  fits/fits_<cell>.csv (+ _template.csv)     the cell's fit store, appended by fit-grouped                      csv_table
  tiecache/tie_<cell>.csv | tie2_<cell>.csv  the cell's tie-cache rows (tie2_ when tie_ has an older header)      tie_cache
  cells/<cell>/chain_summary.json            the tie pass summary (prestage.sh:81)                               snapshot_tie_summary
  cells/<cell>/prefit<i>_requests.csv        each prefit list                                                    csv_table
  cells/<cell>/prestage.log                  the chain and fitter lines prestage.sh:81 keeps
  prefit_claims/<cell>/owner                 the one-writer claim (node:pid), held while the unit runs: the directory the
                                             batch chain's P-G8 waits on (tier_snapshot.py st_preflight; prefit.sh:110)
  cells/<cell>/PREFIT_FAILS                  production root only: one line appended per failed attempt (prefit.sh:63)
  Work: production catalog_snapshot.WORK_ROOT/prestage/<cell> (prestage.sh:50; removed on success, prestage.sh:86); a
  non-production --out keeps it under <out>/_work/<cell>. A NON-PRODUCTION --out first COPIES the cell's production fit store
  and tie-cache files when it has none of its own (--snap-inc-seed), so it writes only under --out.
Failure modes (exit 2, the gate code first in the record's reason): GATES below. Exit 3 (no input): no detection of the
         cell reaches the per-frame fit step, or none reaches the tie. Exit 1 = error.
"""
import fcntl
import glob
import json
import os
import re
import shutil
import socket
import sys
import time

TIER = 'snapshot'
PHASES = ('cell', 'associate')
SEEDS = ('store', 'tiecache')
UNIT = 'incremental.snapshot'

GATES = {
    'SI-G1': 'the search shipped the cell: every searched (subband, Stokes) has a runinfo or night-refusal record and the '
             'evidence of curves.SEARCH_DONE_EVIDENCE holds (the fleet claim done in the window; prestage.sh:32, '
             'snapshot_cell_prefit.sh:76)',
    'SI-G2': 'not a dirty Stokes I cell while catalog_snapshot.DIRTY_I_FLOOR is 0 (snapshot_cell_prefit.sh:73; the chain '
             'refuses such a cell, phase3_snapshot_chain.py:3460-3468)',
    'SI-G3': 'a light-curve table under catalog_snapshot.LIGHTCURVES_ROOT for every planned subband and Stokes I and V '
             '(prestage.sh:33-48; planned = curves plan_cell)',
    'SI-G4': 'one writer per cell: no hold (catalog_snapshot.INC_HOLD_LISTS and BATCH_DIR/rechain_active/*.cells, '
             'snapshot_cell_prefit.sh:77; a frame-time catch-up claim without done, :81; on the production root an old '
             'per-cell stage claim BATCH_DIR/claims/<cell> without PRESTAGE_DONE, :78), then the claim '
             '<out>/prefit_claims/<cell> (:108-110: a dead owner on this node, or an owner-less claim older than '
             'INC_OWNERLESS_CLAIM_STALE_S, is released atomically) and the holds again after the claim (:112)',
    'SI-G5': 'the prefit list is empty within catalog_snapshot.PREFIT_MAX_ITER rounds (prestage.sh:66-78)',
    'SI-G6': 'every in-place call exits 0: the store repair lists, fits and appends, the prefit lists, the fits, the tie '
             'pass (snapshot_cell_prefit.sh:145-164; prestage.sh:80); the reason quotes the call\'s own refusal line, else '
             'its last line without "warn" (prefit.sh:159)',
    'SI-G7': 'non-production seeding: the production files did not change while they were copied',
    'SI-G8': 'forced photometry stage 1 of the cell done (INC_FORCED_STATE_DIR/done/<cell>, or done v2 measure records '
             'of every band under the photometry stage-1 root: forced_v2_evidence) and not marked failed (failed/<cell> '
             'newer than those records), before the chain passes and again before the unit ends (prestage.sh:51-61, :83; '
             'snapshot_cell_prefit.sh:74, :115-121; Gregg 2026-10-04: the stage "won\'t mark a cell done without it")',
    'SI-G9': 'the association / combined-view phase has no ruled per-hour design (always refuses)',
    'SI-G10': 'the store content test snapshot_store_ready.py prints READY for the cell after the prefit loop '
              '(snapshot_cell_prefit.sh:167-168). It reads only the production store directory, so a non-production store '
              'is tested through it only when its newest row per key (store and template) equals the production store\'s',
    'SI-G11': 'the restore-only 3-D gate table catalog_snapshot.INC_CURVE_CONSTANTS_RESTORE exists and is not empty (as '
              'the batch chain\'s P-G21)',
    'SI-G12': 'the attempt cap: fewer than catalog_snapshot.INC_PREFIT_MAX_FAILS lines in <out>/cells/<cell>/PREFIT_FAILS '
              '(snapshot_cell_prefit.sh:71-72; Gregg 2026-10-10 \'Go on all 3\' item 2: a parked cell is not retried until '
              'that file is removed after a diagnosis or a fix)',
    'SI-G13': 'no stop file (catalog_snapshot.INC_STOP_FILES under --out: PREFIT_STOP, PREFIT_STOP_<node>) before the cell '
              'and between fit shards (snapshot_cell_prefit.sh:68, :136); a stopped attempt is not a failure',
    'SI-G14': 'the chain would not refuse the cell: no search flow-unregistered or with a paused 1-minute tier, and a 1-minute '
              'table beside every candidate table of a subband not excluded (snapshot_cell_prefit.sh:82-95, run in place)',
    'SI-G15': 'memory for at least 2 fit workers before each fit shard: (fit_memory_gb: core.nodemem.cgroup_available_gb, '
              'prefit.sh\'s reading, else MemAvailable - catalog_snapshot.INC_FIT_MEM_MARGIN_GB) // INC_FIT_GB_PER_WORKER >= '
              '2; refused at once, never waited on (prefit.sh:122-123 waits up to 6 h, then runs 2 anyway); a node short of '
              'memory is not a failed attempt',
}

OPEN_CHOICES = (
    '(1) the stage-by-stage append design for the 10-s/1-min tiers (plan/PER_HOUR_GAP_DESIGNS_20261111.md 2.2 is generic; '
    'there is no 2.2a-style table for this tier)',
    '(2) the frozen populations of its pooled statistics - catalog-route E pool and footprint area, repeat routing, panel '
    'percentiles and the flood flag, deep-omission noise and rate distributions (DECISION-20261113 covers the deep and '
    '10-min tiers only)',
    '(3) re-routing merged identities in the combined view so a source seen again becomes a repeat '
    '(R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS ruling 1; the view does not re-run routing, '
    'R-REFACTOR-20261020-BUILD-COMBINED-SNAPSHOT-VIEW limit 1)',
    '(4) docs/specs/snapshot_combined_view.md:439 calls an incremental append of the view a design change, not a port',
)


def _T():
    from exopipe.phase3.lib.catalog import tier_sublst
    return tier_sublst


def _S():
    from exopipe.phase3.lib.catalog import tier_snapshot
    return tier_snapshot


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def refuse(msg):
    from exopipe.phase3.core.record import Refusal
    raise Refusal(msg)


def no_input(msg):
    from exopipe.phase3.core.record import NoInput
    raise NoInput(msg)


def default_out(args):
    """The production root of the per-cell products: catalog_snapshot.BATCH_DIR (prestage.sh:22, B=$OPS/snapshot_batch)."""
    return _cfg().catalog_snapshot.BATCH_DIR


def add_arguments(p):
    p.add_argument('--snap-inc-seed', metavar='LIST', default=None,
                   help="--incremental --tier snapshot, NON-PRODUCTION --out only: what to copy from catalog_snapshot.BATCH_DIR "
                        "into --out for a cell that has none of its own: comma list of store (fits_<cell>.csv + "
                        "_template.csv), tiecache (tie_<cell>.csv, tie2_<cell>.csv), or none (default: store,tiecache)")


def parse_phases(args):
    raw = getattr(args, 'inc_phase', None)
    want = [x.strip() for x in str(raw).split(',') if x.strip()] if raw else list(_cfg().catalog_snapshot.INC_PHASES_DEFAULT)
    bad = [x for x in want if x not in PHASES]
    if bad or not want:
        raise SystemExit('error: --inc-phase %s: the snapshot tier has phases %s' % (','.join(bad) or '(none)', ', '.join(PHASES)))
    return [x for x in PHASES if x in want]


def parse_seeds(args):
    raw = getattr(args, 'snap_inc_seed', None)
    if raw is None:
        return list(SEEDS)
    want = [x.strip() for x in str(raw).split(',') if x.strip()]
    if want == ['none']:
        return []
    bad = [x for x in want if x not in SEEDS]
    if bad or not want:
        raise SystemExit('error: --snap-inc-seed %s: give store, tiecache (comma list) or none' % raw)
    return [x for x in SEEDS if x in want]


def cells_of(*paths):
    """The cells of cell lists ('LST DATE' or 'LST_DATE' lines; tier_snapshot.parse_cells_file = prefit.sh:39 cellsof); a
    missing list holds none (cellsof's 2>/dev/null)."""
    out = set()
    for p in paths:
        if p and os.path.exists(p):
            out.update(_S().parse_cells_file(p))
    return out


def dirty_cells(path):
    """The dirty Stokes I cells of the list (snapshot_cell_prefit.sh:73: cellsof the list)."""
    return cells_of(path)


def lightcurve_missing(cell, plan, lc_root):
    """'<sb><S>' of the planned subbands without their table under lc_root (prestage.sh:37-47)."""
    from exopipe.phase3.lib.curves import plan as P
    return ['%s%s' % (sb, st) for sb in plan['planned'] for st in ('I', 'V')
            if not os.path.exists(os.path.join(lc_root, cell, P.table_name(cell, sb, st)))]


def rows_of(path):
    """Data rows of a csv list (`wc -l` - 1, prestage.sh:71)."""
    return max(0, _T().count_lines(path) - 1)


def summary_counts(path):
    """(n at the per-frame fit step, n loaded) of a chain pass's chain_summary.json (snapshot_cell_prefit.sh:102)."""
    d = json.load(open(path))
    return int(d['prefit']['n_at_per_frame_fit_step']), sum(v for k, v in d.items() if k.startswith('n_loaded_'))


def keep_lines(text):
    """The lines prestage.sh:81 keeps in prestage.log: no 'warn' (any case), and one of tie|fit|prefit|pool|REFUS|STEP."""
    return [ln for ln in text.splitlines() if 'warn' not in ln.lower() and re.search(r'tie|fit|prefit|pool|REFUS|STEP', ln)]


def associate_refusal(cell, cell_status):
    return ('SI-G9: %s: the 10-s/1-min per-hour association and combined view have no ruled design - %s. The cell phase is %s; '
            'the cell enters the served catalog through the batch chain (catalog --tier snapshot --run Run_SNAP..., which '
            'reads its tie-cache rows) and the combined-view rebuild. This hour is not complete.'
            % (cell, '; '.join(OPEN_CHOICES), cell_status))


# =====================================================================================================================
# context
# =====================================================================================================================

class Ctx:
    def __init__(self, args, run):
        cfg = _cfg()
        self.T, self.S, self.cfg, self.c = _T(), _S(), cfg, cfg.catalog_snapshot
        self.args, self.run = args, run
        self.dry = bool(run.dry_run)
        self.OUT = os.path.abspath(run.out_root)
        self.prod = cfg.production_root_of(self.OUT) is not None
        if self.prod and os.path.realpath(self.OUT) != os.path.realpath(self.c.BATCH_DIR):
            refuse('SI-G4: a production --out of the snapshot per-hour catalog is catalog_snapshot.BATCH_DIR (%s), the layout '
                   'the batch chain reads; %s is another production root' % (self.c.BATCH_DIR, self.OUT))
        self.fits = os.path.join(self.OUT, 'fits')
        self.tiecache = os.path.join(self.OUT, 'tiecache')
        self.cells = os.path.join(self.OUT, 'cells')
        self.claims = os.path.join(self.OUT, 'prefit_claims')          # the directory P-G8 waits on (prefit.sh:110)
        self.seeds = parse_seeds(args)
        if self.prod and getattr(args, 'snap_inc_seed', None) not in (None, 'none'):
            refuse('SI-G7: --snap-inc-seed copies production files into a NON-production --out; %s is the production root'
                   % self.OUT)
        self.node = socket.gethostname().split('.')[0]                 # prefit.sh:35 NODE=$(hostname -s)
        self.clv = self.T.legacy_root('clv15')
        self.ops = self.T.legacy_root('exopipe_ops')
        self.py36 = self.T.interpreter('py36', self.dry)
        self.chain_py = os.path.join(self.clv, 'phase3_snapshot_chain.py')
        self.fit_py = os.path.join(self.clv, 'phase3_snapshot_fit.py')
        self.work_arg = getattr(args, 'work', None)
        self.restore = self.c.INC_CURVE_CONSTANTS_RESTORE or None
        if self.restore and not (os.path.isfile(self.restore) and os.path.getsize(self.restore) > 0) and not self.dry:
            refuse('SI-G11: the restore-only 3-D gate table catalog_snapshot.INC_CURVE_CONSTANTS_RESTORE = %s is missing or '
                   'empty (the batch chain refuses the same, P-G21)' % self.restore)
        self.chain_defaults = self.S.argparse_defaults(self.chain_py) if os.path.exists(self.chain_py) else {}

    def opsc(self, n):
        return os.path.join(self.ops, n)

    def store(self, cell):
        return os.path.join(self.fits, 'fits_%s.csv' % cell)

    def prod_store(self, cell):
        return os.path.join(self.c.BATCH_DIR, 'fits', 'fits_%s.csv' % cell)

    def work(self, cell):
        if self.prod:
            return os.path.join(self.c.WORK_ROOT, 'prestage', cell)          # prestage.sh:50
        return os.path.join(self.OUT, '_work', cell)

    def chainargs(self, cell):
        return self.S.chain_args(self.cfg, self.c, self.store(cell), self.chain_py, self.restore, (), self.chain_defaults)

    def chain_env(self, **extra):
        """tier_snapshot.chain_env (the batch Ctx's): P3_SNAP_FIT_RAW unless catalog_snapshot.FLOW_MODE; a non-production
        run's refastro scratch on node-local disk (its default lies under a production root)."""
        refastro = None
        if not self.prod:
            from exopipe.phase3.core import fitsio
            from exopipe.phase3.core.record import Refusal
            try:
                root = fitsio.scratch_root(self.work_arg)
            except Refusal:
                if not self.dry:
                    raise
                root = '<node-local scratch: --work or site SCRATCH_ROOT>'
            refastro = os.path.join(root, 'catalog_snapshot_incremental_refastro')
        return self.S.chain_env(self.c, refastro, **extra)

    def fit_env(self):
        """fit-grouped's environment: the chain's switches without the refastro scratch, OMP/BLAS single-threaded per worker
        (prestage.sh:67; snapshot_cell_prefit.sh:31)."""
        return self.S.chain_env(self.c, None, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1)


# =====================================================================================================================
# the cell phase
# =====================================================================================================================

def gate_search(ctx, cell, plan_lines):
    """SI-G1 and the cell plan (curves plan_cell: admission tables, exclusions, flow-unregistered subbands)."""
    from exopipe.phase3.core import exclusions
    from exopipe.phase3.lib.curves import plan as P
    C = ctx.cfg.curves
    pl = P.plan_cell(cell, None, ('I', 'V'), exclusions.for_tier('snapshot'), C, ctx.cfg.instrument.ALL_SUBBANDS)
    if not pl['searched']:
        ok, why = False, 'no admission table under %s (the cell was not searched)' % pl['search_dir']
    else:
        ok, why = P.search_done(pl, C)
    plan_lines.append('SI-G1 search shipped: %s' % why)
    if not ok and not ctx.dry:
        refuse('SI-G1: %s: %s' % (cell, why))
    return pl


def gate_dirty(ctx, cell, plan_lines):
    dirty = cell in dirty_cells(ctx.c.DIRTY_CELLS)
    plan_lines.append('SI-G2 dirty Stokes I cell: %s (list %s, floor %s sigma)' % (dirty, ctx.c.DIRTY_CELLS, ctx.c.DIRTY_I_FLOOR))
    if dirty and not ctx.c.DIRTY_I_FLOOR and not ctx.dry:
        refuse('SI-G2: %s is a dirty Stokes I cell (%s) and catalog_snapshot.DIRTY_I_FLOOR is 0: the chain refuses it without '
               'a calibrated floor (phase3_snapshot_chain.py:3466-3468; snapshot_cell_prefit.sh:73 skips it)'
               % (cell, ctx.c.DIRTY_CELLS))


def gate_lightcurves(ctx, cell, pl, plan_lines):
    miss = lightcurve_missing(cell, pl, ctx.c.LIGHTCURVES_ROOT)
    skipped = ['%s (%s)' % (sb, i['status']) for sb, i in sorted(pl['subbands'].items())
               if i['status'] in ('excluded', 'flow-unregistered')]
    plan_lines.append('SI-G3 light-curve tables: %d planned subbands x I,V under %s; missing %s%s' % (
        len(pl['planned']), ctx.c.LIGHTCURVES_ROOT, ' '.join(miss) or 'none',
        ('; not required: ' + ', '.join(skipped)) if skipped else ''))
    if miss and not ctx.dry:
        refuse('SI-G3: %s: light-curve tables missing (%s) under %s - without them the 3-D bright-source gate re-measures '
               'from the raw frames (prestage.sh:33-36); `exopipe3 curves --tier snapshot --epochs %s` writes them'
               % (cell, ' '.join(miss), ctx.c.LIGHTCURVES_ROOT, cell))


def forced_v2_evidence(ctx, cell):
    """(ok, how, finished) of the v2 evidence of the cell's forced photometry stage 1: done `photometry --tier snapshot`
    measure records of the cell covering every snapshot band of both kinds (steps_snapshot.done_evidence, records only)
    under the photometry production stage-1 root (photometry.OUT_ROOT['snapshot']/STAGE1_DIR, the root production's fleet
    writes) or, for a non-production --out, <--out>/photometry.STAGE1_DIR (the per-hour chain gives both steps one --out);
    ``finished`` = the newest done record's end time (UTC ISO)."""
    from exopipe.phase3.lib.photometry import steps_snapshot as SS
    p = ctx.cfg.photometry
    roots = [os.path.join(p.OUT_ROOT['snapshot'], p.STAGE1_DIR)] + ([] if ctx.prod else [os.path.join(ctx.OUT, p.STAGE1_DIR)])
    whys = []
    for st1 in dict.fromkeys(os.path.abspath(r) for r in roots):
        res = [SS.done_evidence(st1, cell, k, records_only=True) for k in ('sb', 'wb')]
        if all(ok for ok, _ in res):
            t = max((r.get('finished') or '') for r in SS.measure_records(st1, cell) if r.get('status') == 'done')
            return True, 'v2 measure records done for every band, finished %s (%s)' % (t, st1), t
        whys.append('%s: %s' % (st1, '; '.join(why for ok, why in res if not ok)))
    return False, '; '.join(whys), None


def gate_forced(ctx, cell, plan_lines, when='before the chain passes'):
    """SI-G8: forced photometry stage 1 of the cell done and not marked failed (prestage.sh:55-61, :83; prefit.sh:74, :121).
    Production runs snapshot_forced_fleet.sh here when neither marker exists (prestage.sh:56-60) and waits while another
    fleet holds the cell; this stage refuses instead (stage 1 is its own step: `photometry --tier snapshot --epochs`, the
    step before this one in the per-hour chain) and the item is retried. Evidence of stage 1: production's markers, or the
    v2 step's done measure records (forced_v2_evidence), which also clear a failed marker older than them."""
    fst = ctx.c.INC_FORCED_STATE_DIR
    done, failed = os.path.join(fst, 'done', cell), os.path.join(fst, 'failed', cell)
    held = os.path.isdir(os.path.join(fst, 'claims', cell))
    st = 'marked failed' if os.path.exists(failed) else 'done' if os.path.exists(done) else \
        'not done (a forced fleet holds it: %s)' % os.path.join(fst, 'claims', cell) if held else 'not done'
    v2ok, v2how, v2t = forced_v2_evidence(ctx, cell)
    if v2ok and st != 'done':
        mt = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(os.path.getmtime(failed))) if st == 'marked failed' else None
        if mt is None or v2t > mt:
            st = 'done (%s%s)' % (v2how, '; newer than the failed marker of %s' % mt if mt else '')
        else:
            st = 'marked failed %s, after the v2 measure records (finished %s)' % (mt, v2t)
    elif st != 'done':
        st = '%s (v2 records: %s)' % (st, v2how)
    plan_lines.append('SI-G8 forced photometry stage 1 %s: %s (%s)' % (when, st, fst))
    detail, st = st, 'done' if st.startswith('done') else 'marked failed' if st.startswith('marked failed') else st
    if st != 'done' and not ctx.dry:
        refuse('SI-G8: %s: forced photometry stage 1 %s %s (%s; prestage.sh:61, :83): a cell is not prestaged without it '
               '(Gregg 2026-10-04)%s' % (cell, detail, when, failed if st == 'marked failed' else done,
                                         '' if st == 'marked failed' else '; production runs snapshot_forced_fleet.sh '
                                         'for it (prestage.sh:56-60), v2 `photometry --tier snapshot --epochs %s`' % cell))


def holds(ctx, cell):
    """Why the cell is held from every per-cell stage now ([] when it is not): the hold lists (catalog_snapshot.INC_HOLD_LISTS
    and BATCH_DIR/rechain_active/*.cells; prefit.sh:77, :112), a frame-time catch-up claim without done (prefit.sh:81),
    and on the production root an old per-cell stage claim without PRESTAGE_DONE (prefit.sh:78)."""
    B = ctx.c.BATCH_DIR
    out = []
    ra = os.path.join(B, 'rechain_active')
    lists = list(ctx.c.INC_HOLD_LISTS or ()) + (sorted(os.path.join(ra, n) for n in os.listdir(ra) if n.endswith('.cells'))
                                                 if os.path.isdir(ra) else [])
    for p in lists:
        if cell in cells_of(p):
            out.append('hold list %s' % p)
    ft = os.path.join(ctx.c.INC_FRAMETIME_CLAIMS, cell)
    if os.path.isdir(ft) and not os.path.exists(os.path.join(ft, 'done')):
        out.append('frame-time catch-up claim %s (running)' % ft)
    if ctx.prod:
        old = os.path.join(B, 'claims', cell)
        if os.path.isdir(old) and not os.path.exists(os.path.join(B, 'cells', cell, 'PRESTAGE_DONE')):
            out.append('an old per-cell stage claim %s without PRESTAGE_DONE' % old)
    return out


def stop_file(ctx):
    """The stop file present under --out (catalog_snapshot.INC_STOP_FILES: PREFIT_STOP, PREFIT_STOP_<node>;
    snapshot_cell_prefit.sh:24, :68, :136), or None. prestage.sh's STOP_<node> is not one (retired; stale files exist)."""
    for n in ctx.c.INC_STOP_FILES or ():
        p = os.path.join(ctx.OUT, n.format(node=ctx.node))
        if os.path.exists(p):
            return p
    return None


def gate_stop(ctx, cell, plan_lines, when='before the cell'):
    """SI-G13: a stop file ends the attempt (prefit.sh:68 before a cell, :136 between fit shards); not a failure."""
    p = stop_file(ctx)
    if when == 'before the cell':
        plan_lines.append('SI-G13 stop files (%s under %s): %s' % (', '.join(ctx.c.INC_STOP_FILES or ()), ctx.OUT,
                                                                  p or 'none'))
    if p and not ctx.dry:
        refuse('SI-G13: %s: stop file %s %s (snapshot_cell_prefit.sh:68, :136): not a failed attempt' % (cell, p, when))


def fails_path(ctx, cell):
    return os.path.join(ctx.cells, cell, 'PREFIT_FAILS')


def gate_parked(ctx, cell, plan_lines):
    """SI-G12, THE ATTEMPT CAP (snapshot_cell_prefit.sh:71-72; Gregg 2026-10-10 'Go on all 3' item 2): a cell whose
    <out>/cells/<cell>/PREFIT_FAILS holds catalog_snapshot.INC_PREFIT_MAX_FAILS or more lines is parked, on every root."""
    p = fails_path(ctx, cell)
    n = 0
    if os.path.exists(p):
        with open(p, errors='replace') as f:
            n = sum(1 for ln in f if ln.strip())                         # grep -c .
    lim = int(ctx.c.INC_PREFIT_MAX_FAILS)
    plan_lines.append('SI-G12 attempt cap: %d failed attempts in %s (parked at %d)' % (n, p, lim))
    if n >= lim and not ctx.dry:
        refuse('SI-G12: %s is parked: %d failed attempts (>= %d; %s) - not retried until that file is removed after a '
               'diagnosis or a fix (snapshot_cell_prefit.sh:71-72; Gregg 2026-10-10 \'Go on all 3\')' % (cell, n, lim, p))


def failrec(ctx, cell, what):
    """snapshot_cell_prefit.sh:63: on the production root, one line per failed attempt in BATCH_DIR/cells/<cell>/PREFIT_FAILS;
    none while a stop file exists (an attempt ended by a stop file is not a failure). Returns the line written, or None."""
    if not ctx.prod or stop_file(ctx):
        return None
    p = fails_path(ctx, cell)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    line = '%s %s %s (exopipe3 catalog --tier snapshot --incremental, pid %d)' % (
        time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), ctx.node, what, os.getpid())
    with open(p, 'a') as f:
        f.write(line + '\n')
    return line


class _Attempt:
    """One part of the attempt that prefit.sh counts (forced stage 1, store repair, the prefit loop, the store test): a
    refusal inside it, other than a stop (SI-G13), appends its PREFIT_FAILS line (failrec) before it propagates."""

    def __init__(self, ctx, s, cell, what, on_fail=None):
        self.ctx, self.s, self.cell, self.what, self.on_fail = ctx, s, cell, what, on_fail

    def __enter__(self):
        return self

    def __exit__(self, et, e, tb):
        from exopipe.phase3.core.record import Refusal, SpaceRefused
        if et is not None and issubclass(et, Refusal) and self.on_fail:
            self.on_fail()
        if et is not None and issubclass(et, Refusal) and not issubclass(et, SpaceRefused) \
                and not str(e).startswith(('SI-G13', 'SI-G15')):
            what = self.what(e) if callable(self.what) else self.what
            line = failrec(self.ctx, self.cell, what)
            if line:
                self.s.u.meta['prefit_fails_appended'] = line
        return False


def _eval_fields(node, names):
    import ast
    if isinstance(node, ast.List):
        return ast.literal_eval(node)
    if isinstance(node, ast.Name):
        return list(names[node.id])
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _eval_fields(node.left, names) + _eval_fields(node.right, names)
    raise ValueError('not a list expression: %s' % ast.dump(node)[:120])


def tie_cache_fields(ctx):
    """The current tie-cache header: FIELDS of the installed clv15/phase3_snapshot_tiecache.py, read from its source
    (FIELDS_V1 = [...]; FIELDS = FIELDS_V1 + [...]); a header other than it is never appended to (tiecache.save). Cached."""
    if not hasattr(ctx, '_tie_fields'):
        import ast
        p = os.path.join(ctx.clv, 'phase3_snapshot_tiecache.py')
        names = {}
        try:
            for node in ast.parse(open(p).read()).body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                        and node.targets[0].id in ('FIELDS_V1', 'FIELDS'):
                    names[node.targets[0].id] = _eval_fields(node.value, names)
            ctx._tie_fields = (names['FIELDS'], None)
        except (OSError, SyntaxError, ValueError, KeyError) as e:
            ctx._tie_fields = (None, '%s: FIELDS not read (%s)' % (p, e))
    return ctx._tie_fields


def current_tie_cache(ctx, cell):
    """(path, None) of the cell's production tie-cache file (tie_ or tie2_) whose header line is the current one, else
    (None, why)."""
    import csv
    fields, err = tie_cache_fields(ctx)
    if err:
        return None, err
    seen = []
    for n in ('tie_%s.csv', 'tie2_%s.csv'):
        p = os.path.join(ctx.c.BATCH_DIR, 'tiecache', n % cell)
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            continue
        with open(p, newline='') as f:
            hdr = next(csv.reader([f.readline()]), [])
        if hdr == fields:
            return p, None
        seen.append('%s (an older header, %d columns)' % (os.path.basename(p), len(hdr)))
    return None, ('no tie-cache file with the current header (%d columns, phase3_snapshot_tiecache.FIELDS): %s'
                  % (len(fields), '; '.join(seen) or 'no tie_ or tie2_ file'))


def prestaged_by_production(ctx, cell):
    """On the production root: (the PRESTAGE_DONE line production wrote, None) when production prestaged the cell
    (prestage.sh:84) and its tie-cache file has the current header; (None, why) when PRESTAGE_DONE exists but the tie cache
    does not hold the current header (the unit runs); (None, None) otherwise."""
    if not ctx.prod:
        return None, None
    p = os.path.join(ctx.c.BATCH_DIR, 'cells', cell, 'PRESTAGE_DONE')
    if not os.path.exists(p):
        return None, None
    tc, why = current_tie_cache(ctx, cell)
    if not tc:
        return None, '%s exists, but %s: the unit runs' % (p, why)
    return '%s: %s; tie cache %s (current header)' % (p, open(p, errors='replace').read().strip()[:300], tc), None


def gate_chain_refuses(ctx, s, cell, W, plan_lines):
    """SI-G14 (snapshot_cell_prefit.sh:82-95): the chain's own refusals (phase3_snapshot_chain.build) checked before the
    claim, run in place (child 'chain-refuses': phase3_snapshot_search.excluded_subbands): a search flow-unregistered or with
    a paused 1-minute tier, or a candidate table of a subband not excluded without its 1-minute table. prefit.sh waits on such
    a cell (no failed attempt); a failed check refuses too (prefit.sh's ${CREF:-1})."""
    plan_lines.append('SI-G14 the chain would refuse the cell (prefit.sh:82-95): %s %s chain-refuses %s %s %s' % (
        ctx.py36, __file__, ctx.clv, ctx.cfg.paths.SNAPSHOT_FLEET_OUT, cell))
    if ctx.dry:
        return
    d = os.path.join(os.path.dirname(W), '.precheck_%s_%d' % (cell, os.getpid()))   # before the claim: never the cell's W
    try:
        res = ctx.T.call(ctx, s, os.path.abspath(__file__), ['chain-refuses', ctx.clv, ctx.cfg.paths.SNAPSHOT_FLEET_OUT,
                                                             cell], 'py36', log=os.path.join(d, 'chain_refuses.log'), cwd=d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    line = ctx.T.last_line(res.tail, r'^CHAIN-REFUSES ')
    n = line.split()[1] if len(line.split()) > 1 else ''
    if not res.ok or n != '0':
        refuse('SI-G14: %s: the chain would refuse the cell (snapshot_cell_prefit.sh:82-95; prefit.sh waits on it): %s'
               % (cell, line[len('CHAIN-REFUSES '):][:300] if line else 'the check failed (%s)' % (res.reason or 'no output')))


class _ClaimLock:
    """An exclusive flock on <out>/prefit_claims/.v2_claim.lock while a claim is judged, released or made, so two v2
    writers never release a claim another one has just made. Production's prefit.sh takes no lock (its own release
    at :108-110 is check-then-remove); a release renames the claim aside first, so a remover never deletes a claim made
    after it judged the old one dead."""

    def __init__(self, ctx):
        self.path = os.path.join(ctx.claims, '.v2_claim.lock')

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.f = open(self.path, 'a')
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def _owner(cd):
    try:
        return open(os.path.join(cd, 'owner')).read().strip()
    except OSError:
        return None


def stale_claim(ctx, cd, now=None):
    """Why the claim cd is stale, or None: its owner is a process of this node that no longer runs (prefit.sh:108-109,
    kill -0), or it has no owner file and is older than catalog_snapshot.INC_OWNERLESS_CLAIM_STALE_S (its writer died
    between mkdir and the owner write)."""
    o = _owner(cd)
    if o is None:
        try:
            age = (now or time.time()) - os.stat(cd).st_mtime
        except FileNotFoundError:
            return None
        lim = float(ctx.c.INC_OWNERLESS_CLAIM_STALE_S)
        return 'no owner file, %.0f s old (> %.0f s)' % (age, lim) if age > lim else None
    host, _, pid = o.partition(':')
    if host == ctx.node and pid.isdigit():
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return 'owner %s is not running' % o
        except PermissionError:
            return None
    return None


def release_claim(ctx, cd, why, say=True):
    """Remove a claim atomically: rename it aside (a hidden name no gate reads), then remove the copy."""
    aside = os.path.join(os.path.dirname(cd), '.released.%s.%s.%d.%d' % (os.path.basename(cd), ctx.node, os.getpid(),
                                                                         int(time.time() * 1000)))
    try:
        os.rename(cd, aside)
    except FileNotFoundError:
        return False
    shutil.rmtree(aside, ignore_errors=True)
    if say:
        print('SI-G4: released claim %s (%s)' % (cd, why), file=sys.stderr)
    return True


def claim(ctx, cell, plan_lines):
    """SI-G4: the claim <out>/prefit_claims/<cell> with owner node:pid (prefit.sh:108-112), under the claim lock: a stale claim
    is released first; the holds are checked again after the claim (a batch driver may have started in between). Returns
    the claim directory (None in a dry run)."""
    cd = os.path.join(ctx.claims, cell)
    plan_lines.append('SI-G4 one writer: claim %s (owner %s:<pid>), the holds checked again after it' % (cd, ctx.node))
    if ctx.dry:
        return None
    me = '%s:%d' % (ctx.node, os.getpid())
    with _ClaimLock(ctx):
        why = stale_claim(ctx, cd) if os.path.isdir(cd) else None
        if why and release_claim(ctx, cd, why):
            plan_lines.append('SI-G4 released a stale claim %s: %s' % (cd, why))
        try:
            os.mkdir(cd)
        except FileExistsError:
            refuse('SI-G4: %s is claimed by %s (%s): one writer per cell' % (cell, _owner(cd) or 'a writer without owner '
                                                                            'file yet', cd))
        with open(os.path.join(cd, 'owner.tmp'), 'w') as f:
            f.write(me + '\n')
        os.replace(os.path.join(cd, 'owner.tmp'), os.path.join(cd, 'owner'))
    h = holds(ctx, cell)
    if h:
        unclaim(ctx, cd)
        refuse('SI-G4: %s is held (after the claim; prefit.sh:112): %s' % (cell, '; '.join(h)))
    return cd


def unclaim(ctx, cd):
    """Release this process's own claim (prestage.sh / prefit.sh rm -rf the claim), atomically, only while it is ours."""
    if not cd:
        return
    with _ClaimLock(ctx):
        if _owner(cd) == '%s:%d' % (ctx.node, os.getpid()):
            release_claim(ctx, cd, 'this unit ended', say=False)


def seed(ctx, s, cell, plan_lines):
    """Non-production: copy the cell's production fit store and tie-cache files that --out does not hold yet (SI-G7)."""
    if ctx.prod:
        return
    S = ctx.S
    src = {'store': [ctx.prod_store(cell)],
           'tiecache': [os.path.join(ctx.c.BATCH_DIR, 'tiecache', n % cell) for n in ('tie_%s.csv', 'tie2_%s.csv')]}
    src['store'].append(S.template_of(src['store'][0]))
    dst_dir = {'store': ctx.fits, 'tiecache': ctx.tiecache}
    todo = []
    for kind in ctx.seeds:
        for p in src[kind]:
            q = os.path.join(dst_dir[kind], os.path.basename(p))
            if os.path.exists(p) and not os.path.exists(q):
                todo.append((p, q))
    plan_lines.append('seed (non-production, --snap-inc-seed %s): %s' % (
        ','.join(ctx.seeds) or 'none', '; '.join('%s -> %s' % pq for pq in todo) or 'nothing to copy'))
    if ctx.dry:
        return
    got = {}
    for p, q in todo:
        h0 = ctx.T.sha1_file(p)
        os.makedirs(os.path.dirname(q), exist_ok=True)
        shutil.copy2(p, q + '.part')
        h1, h2 = ctx.T.sha1_file(q + '.part'), ctx.T.sha1_file(p)
        if not (h0 == h1 == h2):
            os.remove(q + '.part')
            refuse('SI-G7: %s changed while it was copied (sha1 %s, copy %s, now %s)' % (p, h0, h1, h2))
        os.replace(q + '.part', q)
        got[q] = {'from': p, 'sha1': h0}
    s.u.meta['seeded'] = got


def _keep_log(ctx, cell, W, logname):
    """A failed call's log copied beside the cell's products (prestage.sh:70, :76, :80; prefit.sh:152, :159, :164)."""
    cdir = os.path.join(ctx.cells, cell)
    os.makedirs(cdir, exist_ok=True)
    lp = os.path.join(W, logname)
    if os.path.exists(lp):
        shutil.copy2(lp, os.path.join(cdir, logname))


def reason_line(path):
    """The line a failed call's reason quotes, from its own section of the log (after the last '===== ' header): its last
    refusal line (REFUS; the chain's own reason, whole), else its last line without 'warn' (snapshot_cell_prefit.sh:159)."""
    try:
        text = open(path, errors='replace').read()
    except OSError:
        return ''
    part = text[text.rfind('\n===== ') + 1:] if '\n===== ' in text else text
    lines = [ln for ln in part.splitlines()[1 if part.startswith('===== ') else 0:] if ln.strip()]
    ref = [ln for ln in lines if 'REFUS' in ln]
    if ref:
        return ref[-1]
    nw = [ln for ln in lines if 'warn' not in ln.lower()]
    return nw[-1] if nw else ''


def _need(ctx, cell, W, res, logname, what):
    """Refuse SI-G6 unless the call exited 0 (tier_sublst.need_ok's rule), its log kept first; the reason quotes the call's
    own refusal line (reason_line) and legacy.run's sticky refusal when there is one."""
    if res.ok:
        return
    _keep_log(ctx, cell, W, logname)
    line = reason_line(os.path.join(W, logname))
    extra = res.reason if res.reason and 'REFUSED' in res.reason and (not line or line not in res.reason) else ''
    refuse('SI-G6: %s: %s: exit %s: %s' % (cell, what, res.rc, '; '.join(x for x in (line[:400], extra[:300]) if x)
                                          or '(no output)'))


def _chain(ctx, s, cell, W, name, extra, env, hours):
    """One chain pass (prestage.sh:62-65, prefit.sh:153-156): its own --out W/<name>, log W/<name>.log, a limit of ``hours``."""
    a = ctx.chainargs(cell) + ['--cells', cell, '--out', os.path.join(W, name), '--veto-record-root', ''] + extra
    return ctx.T.call(ctx, s, ctx.chain_py, a, 'py36', env=env, log=os.path.join(W, '%s.log' % name), cwd=W,
                      timeout=int(float(hours) * 3600))


def fit_memory_gb():
    """(GB, source) the fit workers may use: prefit.sh's reading (snapshot_cell_prefit.sh:40-55, memprocs: the /user.slice
    cgroup's limit less its usage net of reclaimable file cache) = core.nodemem.cgroup_available_gb(); only when no cgroup
    limit can be read, core.nodemem.mem_available_gb() (MemAvailable). (None, why) when neither can be read.
    Why not nodemem.mem_available_gb()'s min(MemAvailable, cgroup), v2's reading elsewhere: MemAvailable leaves out the ZFS
    ARC of the recorder pools, which shrinks under pressure (267 GB on lwacalim07 on 2026-11-24, where the min() gave 2
    workers and prefit.sh's reading about 72). The 2026-11-15 lwacalim02 MemoryError behind the min()
    (INCIDENT-20261115-TENMIN-07H-TIE-MEMORYERROR-GROUP-202-LWACALIM02-REWRITE-WORKERS-CONTRIBUTED) was ONE 24.7 GiB
    allocation refused by the kernel's overcommit estimate, which leaves out the ARC; prefit's 3 GB per-worker allocations
    have run fleet-wide with memprocs since 2026-10-08 (Snapshot chat, PR 158 review). nodemem and its other callers keep
    the min(). Units: nodemem's GB are 2**30 bytes (prefit.sh's 1e9), and it also counts active_file as reclaimable."""
    from exopipe.phase3.core import nodemem
    c = nodemem.cgroup_available_gb()
    if c is not None:
        return float(c), 'core.nodemem.cgroup_available_gb (the cgroup limit less usage net of file cache; prefit.sh:40-55)'
    m = nodemem.mem_available_gb()
    if m is not None:
        return float(m), 'core.nodemem.mem_available_gb (MemAvailable: no cgroup limit could be read)'
    return None, 'neither a cgroup limit nor MemAvailable could be read'


def memprocs(ctx):
    """Fit workers the memory holds now (snapshot_cell_prefit.sh:40-55): (fit_memory_gb - INC_FIT_MEM_MARGIN_GB) //
    INC_FIT_GB_PER_WORKER; 0 when nothing can be read."""
    m, _ = fit_memory_gb()
    return max(0, int((float(m or 0) - float(ctx.c.INC_FIT_MEM_MARGIN_GB)) // float(ctx.c.INC_FIT_GB_PER_WORKER)))


def fit_procs(ctx):
    """snapshot_cell_prefit.sh:122-123 np(): (this process's cores less FIT_PROCS_RESERVE, at most memprocs, at least 2,
    None) - or (None, why) when memprocs < 2: SI-G15 refuses at once (np waits up to 6 h for room for 2 workers, then runs 2
    anyway; v2 neither waits silently nor runs short of memory)."""
    c = ctx.c
    want = max(1, len(os.sched_getaffinity(0)) - int(c.FIT_PROCS_RESERVE))
    m = memprocs(ctx)
    if m < 2:
        gb, src = fit_memory_gb()
        return None, ('memory for %d fit workers, fewer than 2: %s GB from %s, less INC_FIT_MEM_MARGIN_GB %s, // '
                      'INC_FIT_GB_PER_WORKER %s' % (m, 'none' if gb is None else '%.1f' % gb, src,
                                                    c.INC_FIT_MEM_MARGIN_GB, c.INC_FIT_GB_PER_WORKER))
    return max(2, min(want, m)), None


def fit_groups(ctx, s, W, pending):
    """The (cell, minute, band group) tasks of a request list (snapshot_cell_prefit.sh:129-133, child 'fit-groups' in place:
    phase3_snapshot_fit._group_order); 1 when it cannot be counted (prefit.sh's `|| echo 1`)."""
    res = ctx.T.call(ctx, s, os.path.abspath(__file__), ['fit-groups', ctx.clv, pending], 'py36',
                     log=os.path.join(W, 'fit_groups.log'), cwd=W)
    line = ctx.T.last_line(res.tail, r'^FIT-GROUPS \d+')
    return int(line.split()[1]) if res.ok and line else 1


def _fit(ctx, s, cell, W, pending, store, logname):
    """fit-grouped of a request list into a store in shards (snapshot_cell_prefit.sh:128-140 fitg): k = ceil(requests /
    INC_FIT_SHARD_REQUESTS), at most groups // (INC_FIT_TASKS_PER_WORKER x workers), at least 1; the stop files checked
    before every shard; workers from fit_procs before every shard; OMP/BLAS 1 thread, niceness INC_FIT_NICE. Returns
    ('ok' | 'stopped' | 'memory' | 'failed', the last LegacyResult, the stop file or the memory reason)."""
    c = ctx.c
    n = rows_of(pending)
    p, why = fit_procs(ctx)
    if p is None:
        return 'memory', why + ' (before shard 0)'
    g = fit_groups(ctx, s, W, pending)
    k = max(1, min((n + int(c.INC_FIT_SHARD_REQUESTS) - 1) // int(c.INC_FIT_SHARD_REQUESTS),
                   g // (int(c.INC_FIT_TASKS_PER_WORKER) * p)))
    res = None
    for i in range(k):
        st = stop_file(ctx)
        if st:
            s.u.note('stop file %s during the %s fits (shard %d of %d)' % (st, cell, i, k))
            return 'stopped', st
        res = ctx.T.call(ctx, s, ctx.fit_py, ['fit-grouped', '--pending', pending, '--store', store, '--proto-root',
                                              ctx.cfg.paths.SNAPSHOT_FLEET_OUT, '--procs', str(p), '--shard', str(i),
                                              '--nshards', str(k)],
                         'py36', env=ctx.fit_env(), log=os.path.join(W, logname), cwd=W, nice=int(c.INC_FIT_NICE))
        if not res.ok:
            return 'failed', res
        if i + 1 < k:
            p, why = fit_procs(ctx)
            if p is None:
                return 'memory', why + ' (before shard %d of %d)' % (i + 1, k)
    return 'ok', res


def _fitted(ctx, cell, W, state, res, logname, what):
    """Refuse a fit that did not end 'ok': SI-G13 for a stop file, SI-G15 for memory, SI-G6 for a failed shard."""
    if state == 'stopped':
        refuse('SI-G13: %s: stop file %s during the %s (snapshot_cell_prefit.sh:136): not a failed attempt' % (cell, res, what))
    if state == 'memory':
        refuse('SI-G15: %s: %s for the %s: refused, not waited on; not a failed attempt' % (cell, res, what))
    if state == 'failed':
        _need(ctx, cell, W, res, logname, what)


def cell_phase(ctx, s, cell, plan_lines):
    """The gates in snapshot_cell_prefit.sh's order (:68-121), the claim, then the stage (_cell_run)."""
    c = ctx.c
    W = ctx.work(cell)
    gate_stop(ctx, cell, plan_lines)                                      # prefit.sh:68
    gate_parked(ctx, cell, plan_lines)                                    # prefit.sh:71-72
    gate_dirty(ctx, cell, plan_lines)                                     # prefit.sh:73
    fst_failed = os.path.join(c.INC_FORCED_STATE_DIR, 'failed', cell)
    if os.path.exists(fst_failed) and not ctx.dry:                        # prefit.sh:74
        gate_forced(ctx, cell, plan_lines)
    pl = gate_search(ctx, cell, plan_lines)                               # prefit.sh:76
    h = holds(ctx, cell)                                                  # prefit.sh:77, :78, :81
    plan_lines.append('SI-G4 holds: %s' % ('; '.join(h) or 'none'))
    if h and not ctx.dry:
        refuse('SI-G4: %s is held: %s' % (cell, '; '.join(h)))
    gate_chain_refuses(ctx, s, cell, W, plan_lines)                       # prefit.sh:82-95
    gate_lightcurves(ctx, cell, pl, plan_lines)                           # prefit.sh:96-107
    gate_forced(ctx, cell, plan_lines)                                    # prefit.sh:114-121 (refused, not run)
    maxit = int(c.PREFIT_MAX_ITER)
    store = ctx.store(cell)
    plan_lines.append('store repair (<= %d rounds; prefit.sh:141-152): %s %s %s %s/repair<r>.csv%s --check; rows > 0 -> '
                      'fit-grouped into %s/repair<r>.store.csv, then %s <that store> --fits-dir %s' % (
                          int(c.INC_STORE_REPAIR_ROUNDS), ctx.py36, ctx.opsc('snapshot_store_repair.py'), cell, W,
                          '' if ctx.prod else ' --store %s' % store, W, ctx.opsc('snapshot_store_append.py'), ctx.fits))
    plan_lines.append('prefit loop (<= %d rounds; prefit.sh:153-166): %s %s %s --cells %s --out %s/prefit<i> --veto-record-root '
                      '"" --prefit-list-only (%s h limit); rows > 0 -> %s fit-grouped --pending %s/prefit<i>/prefit_requests.csv '
                      '--store %s --proto-root %s --procs <min(free cores - %d, (cgroup headroom GB - %s) // %s), >= 2> --shard i '
                      '--nshards <ceil(rows / %d), <= groups // (%d x procs)>, niceness %d, stop files checked between shards, SI-G15 when '
                      'memory holds fewer than 2 workers' % (
                          maxit, ctx.py36, ctx.chain_py, ' '.join(ctx.chainargs(cell)), cell, W, c.INC_PREFIT_TIMEOUT_H,
                          ctx.fit_py, W, store, ctx.cfg.paths.SNAPSHOT_FLEET_OUT, c.FIT_PROCS_RESERVE,
                          c.INC_FIT_MEM_MARGIN_GB, c.INC_FIT_GB_PER_WORKER, c.INC_FIT_SHARD_REQUESTS,
                          c.INC_FIT_TASKS_PER_WORKER, c.INC_FIT_NICE))
    plan_lines.append('SI-G10 store content test (prefit.sh:167-168): %s%s %s %s -> READY' % (
        '' if ctx.prod else 'non-production: %s store-same %s %s (newest row per key, store and template), then ' % (
            __file__, store, ctx.prod_store(cell)), ctx.py36, ctx.opsc('snapshot_store_ready.py'), cell))
    plan_lines.append('tie pass (prestage.sh:79-80): P3_TIE_CACHE_DIR=%s P3_REFASTRO_SHARDS=<%d if >= %d detections at the fit '
                      'step else %d> %s --cells %s --out %s/tie --veto-record-root "" --stop-after-tie (%s h limit); judged by '
                      'its exit code' % (ctx.tiecache, c.TIE_SHARDS[1], c.TIE_SHARDS[0], c.TIE_SHARDS[2], ctx.chain_py, cell, W,
                                         c.INC_CHAIN_TIMEOUT_H))
    plan_lines.append('attempt record (prefit.sh:63): %s' % (
        'a refusal in forced stage 1 (at the end), the store repair, the prefit loop or the store test appends one line to '
        '%s, none under a stop file' % fails_path(ctx, cell) if ctx.prod else 'non-production: no PREFIT_FAILS line'))
    cd = claim(ctx, cell, plan_lines)
    try:
        seed(ctx, s, cell, plan_lines)
        if ctx.dry:
            return
        _cell_run(ctx, s, cell, W, store, plan_lines)
    finally:
        unclaim(ctx, cd)


def store_repair(ctx, s, cell, W, store):
    """snapshot_cell_prefit.sh:141-152: the rows the store content test refuses, listed by snapshot_store_repair.py (--check:
    its counts equal snapshot_store_ready.py's on the production store), fitted into a NEW store (fit-grouped never refits
    a key of its own output store) and appended to the cell's store - the newest row per key wins. Returns the rows."""
    T, nrep = ctx.T, 0
    for r in range(1, int(ctx.c.INC_STORE_REPAIR_ROUNDS) + 1):
        lst = os.path.join(W, 'repair%d.csv' % r)
        a = [cell, lst] + ([] if ctx.prod else ['--store', store]) + ['--check']
        res = T.call(ctx, s, ctx.opsc('snapshot_store_repair.py'), a, 'py36', log=os.path.join(W, 'repair%d.log' % r),
                     cwd=W)
        _need(ctx, cell, W, res, 'repair%d.log' % r, 'store repair list %d' % r)
        n = rows_of(lst)
        if n == 0:
            break
        nrep += n
        new = os.path.join(W, 'repair%d.store.csv' % r)
        for q in (new, ctx.S.template_of(new)):                           # prefit.sh:147
            if os.path.exists(q):
                os.remove(q)
        state, res = _fit(ctx, s, cell, W, lst, new, 'repair_fit%d.log' % r)
        if os.path.exists(new) and os.path.getsize(new) > 0:              # what a stopped or failed round fitted is appended
            ap = T.call(ctx, s, ctx.opsc('snapshot_store_append.py'), [new, '--fits-dir', ctx.fits], 'py36',
                        log=os.path.join(W, 'repair_append%d.log' % r), cwd=W)
            _need(ctx, cell, W, ap, 'repair_append%d.log' % r, 'store repair append %d' % r)
        _fitted(ctx, cell, W, state, res, 'repair_fit%d.log' % r, 'store repair fits %d' % r)
    return nrep


def store_ready(ctx, s, cell, W, store):
    """SI-G10 (snapshot_cell_prefit.sh:167-168): snapshot_store_ready.py's line for the cell must start READY. The test reads
    only the production store directory: a non-production store is tested through it only when its content (the newest row
    per key of the store and of its template side store, phase3_snapshot_fit.load_store) equals the production store's."""
    T = ctx.T
    if not ctx.prod:
        res = T.call(ctx, s, os.path.abspath(__file__), ['store-same', ctx.clv, store, ctx.prod_store(cell)], 'py36',
                     log=os.path.join(W, 'store_same.log'), cwd=W)
        line = T.last_line(res.tail, r'^STORE-(SAME|DIFFERS)')
        if not res.ok or not line.startswith('STORE-SAME'):
            refuse('SI-G10: %s: the store content test reads only the production store %s, and this run\'s store %s does '
                   'not hold its content (%s): a non-production store is tested only when it equals production\'s'
                   % (cell, ctx.prod_store(cell), store, line or res.reason))
        s.u.meta['store_same_as_production'] = line
    res = T.call(ctx, s, ctx.opsc('snapshot_store_ready.py'), [cell], 'py36', log=os.path.join(W, 'store_ready.log'), cwd=W)
    lines = [ln for ln in open(os.path.join(W, 'store_ready.log'), errors='replace').read().splitlines() if ln.strip()]
    line = lines[-1] if lines else ''                                     # prefit.sh:167 `| tail -1`
    s.u.check('SI-G10.store-ready', line.startswith('READY'), detail=line[:300])
    if not line.startswith('READY'):
        _keep_log(ctx, cell, W, 'store_ready.log')
        refuse('SI-G10: %s: store content test after the fits: %s' % (cell, line[:300] or res.reason))
    s.u.meta['store_ready'] = line[:400]


def _cell_run(ctx, s, cell, W, store, plan_lines):
    T, c = ctx.T, ctx.c
    if os.path.lexists(W):
        shutil.rmtree(W)                                                  # prestage.sh:50 rm -rf $W
    os.makedirs(W)
    os.makedirs(ctx.fits, exist_ok=True)
    cdir = os.path.join(ctx.cells, cell)
    os.makedirs(cdir, exist_ok=True)
    s.code(ctx.chain_py, ctx.fit_py, ctx.opsc('snapshot_store_repair.py'), ctx.opsc('snapshot_store_append.py'),
           ctx.opsc('snapshot_store_ready.py'))
    s.inp(store, ctx.S.template_of(store), ctx.restore)
    keep = lambda: [_keep_log(ctx, cell, W, os.path.basename(x)) for x in glob.glob(os.path.join(W, 'repair*.log'))]
    with _Attempt(ctx, s, cell, 'store repair (%s/repair*.log)' % cdir, on_fail=keep):     # prefit.sh:152
        s.u.meta['repaired'] = store_repair(ctx, s, cell, W, store)
    maxit = int(c.PREFIT_MAX_ITER)
    ok, nfit, nat, nload, it = False, 0, 0, 0, 0
    rounds = []
    with _Attempt(ctx, s, cell, 'prefit list not empty after %d iterations or a step failed (%s)' % (maxit, W)):  # :166
        for it in range(1, maxit + 1):
            res = _chain(ctx, s, cell, W, 'prefit%d' % it, ['--prefit-list-only'], ctx.chain_env(), c.INC_PREFIT_TIMEOUT_H)
            _need(ctx, cell, W, res, 'prefit%d.log' % it, 'prefit list %d' % it)
            req = os.path.join(W, 'prefit%d' % it, 'prefit_requests.csv')
            if not os.path.exists(req):
                _keep_log(ctx, cell, W, 'prefit%d.log' % it)
                refuse('SI-G6: %s: prefit list %d wrote no %s' % (cell, it, req))
            keep = os.path.join(cdir, 'prefit%d_requests.csv' % it)
            shutil.copy2(req, keep)
            s.out(keep, 'csv_table')
            n = rows_of(req)
            nat, nload = summary_counts(os.path.join(W, 'prefit%d' % it, 'chain_summary.json'))
            rounds.append({'round': it, 'without_fit': n, 'at_fit_step': nat, 'loaded': nload})
            if n == 0:
                ok = True
                break
            nfit += n
            state, res = _fit(ctx, s, cell, W, req, store, 'fits%d.log' % it)
            _fitted(ctx, cell, W, state, res, 'fits%d.log' % it, 'fits of prefit round %d' % it)
        s.u.meta.update(prefit_rounds=rounds, prefit_fitted=nfit, n_at_fit_step=nat, n_loaded=nload)
        if not ok:
            refuse('SI-G5: %s: %d detections still without their own fit after %d prefit rounds '
                   '(%s/prefit%d/prefit_requests.csv)' % (cell, rounds[-1]['without_fit'], it, W, it))
    with _Attempt(ctx, s, cell, lambda e: 'store test: %s' % str(e).split('after the fits: ', 1)[-1]):   # :168
        store_ready(ctx, s, cell, W, store)
    if nat == 0:
        no_input('%s: no detection reaches the per-frame fit step (%d loaded): nothing to tie' % (cell, nload))
    shards = ctx.S.tie_shards(nat, c.TIE_SHARDS)
    res = _chain(ctx, s, cell, W, 'tie', ['--stop-after-tie'],
                 ctx.chain_env(P3_TIE_CACHE_DIR=ctx.tiecache, P3_REFASTRO_SHARDS=shards), c.INC_CHAIN_TIMEOUT_H)
    _need(ctx, cell, W, res, 'tie.log', 'tie pass')                     # prestage.sh:80: the exit code (not counted: prefit.sh
                                                                        # has no tie pass)
    summ = json.load(open(os.path.join(W, 'tie', 'chain_summary.json')))
    tc = summ.get('tie_cache') or {}
    s.u.meta.update(tie_shards=shards, tie_from_cache=tc.get('from_cache'), tie_tied_now=tc.get('tied_now'))
    shutil.copy2(os.path.join(W, 'tie', 'chain_summary.json'), os.path.join(cdir, 'chain_summary.json'))
    text = ''
    for n in ['prefit%d.log' % i for i in range(1, it + 1)] + ['fits%d.log' % i for i in range(1, it + 1)] + ['tie.log']:
        p = os.path.join(W, n)
        if os.path.exists(p):
            text += open(p, errors='replace').read()
    with open(os.path.join(cdir, 'prestage.log'), 'w') as f:
        f.write('\n'.join(keep_lines(text)) + '\n')
    s.u.note(T.last_line(res.tail, 'tie cache:'))
    if not (tc.get('from_cache') or 0) + (tc.get('tied_now') or 0):
        no_input('%s: no detection of the %d at the per-frame fit step reaches the association pool: nothing tied' % (cell, nat))
    with _Attempt(ctx, s, cell, 'forced photometry stage 1'):           # prestage.sh:83; prefit.sh:121
        gate_forced(ctx, cell, plan_lines, when='at the end')
    s.out(os.path.join(cdir, 'chain_summary.json'), 'snapshot_tie_summary')
    for p in (store, ctx.S.template_of(store)):                          # the fit store and its template side store
        if os.path.exists(p):
            s.out(p, 'csv_table')
    for n in ('tie_%s.csv', 'tie2_%s.csv'):
        p = os.path.join(ctx.tiecache, n % cell)
        if os.path.exists(p):
            s.out(p, 'tie_cache')
    if ctx.prod:
        shutil.rmtree(W, ignore_errors=True)                              # prestage.sh:86


def child_store_same(clv, a, b):
    """Child (py36, clv15 in place): STORE-SAME when the newest row per key of store a and of its template side store equal
    store b's (phase3_snapshot_fit.load_store / template_store_path), else STORE-DIFFERS with the counts."""
    sys.path.insert(0, clv)
    from phase3_snapshot_fit import load_store, template_store_path
    out = []
    for kind, x, y in (('store', a, b), ('template', template_store_path(a), template_store_path(b))):
        p, q = load_store(x), load_store(y)
        only_p = len(set(p) - set(q))
        only_q = len(set(q) - set(p))
        diff = sum(1 for k in set(p) & set(q) if p[k] != q[k])
        out.append((kind, len(p), only_p, only_q, diff))
    same = all(o[2] == o[3] == o[4] == 0 for o in out)
    print('%s %s' % ('STORE-SAME' if same else 'STORE-DIFFERS', '; '.join(
        '%s: %d keys, %d only here, %d only in production, %d differ' % o for o in out)))
    return 0 if same else 1


# =====================================================================================================================
# entry point (steps/catalog.py: --incremental --tier snapshot)
# =====================================================================================================================

def run(args, run, epochs=None):
    """Run the phases for every epoch item (a night cell). Returns None (the CLI derives the exit code from the units), or
    in --dry-run 2 when the associate phase is selected (it refuses unconditionally), else 0."""
    from exopipe.phase3.core.cells import expand_epochs
    if epochs is not None:                      # steps/catalog.py expanded them already (a list)
        items = list(epochs)
    else:
        items = expand_epochs(args.epochs) if getattr(args, 'epochs', None) else []
    phases = parse_phases(args)
    if not items:
        raise SystemExit('error: catalog --tier snapshot --incremental needs --epochs <LST>_<date>[,...] (a night cell each)')
    ctx = Ctx(args, run)
    where = 'production' if ctx.prod else 'non-production (seed %s)' % (','.join(ctx.seeds) or 'none')
    print('catalog --tier snapshot --incremental: out %s (%s); phases %s; cells %s; code %s' % (
        ctx.OUT, where, ','.join(phases), ','.join(items), ctx.clv), file=sys.stderr)
    T = ctx.T
    if not ctx.dry:
        T.env_leaks(ctx.S.LEAKY_ENV)
    rc = 0
    for cell in items:
        status = 'not run (--inc-phase)'
        if 'cell' in phases:
            name = '%s.cell.%s' % (UNIT, cell)
            meta = dict(tier=TIER, phase='cell', cell=cell)
            done_by, not_done = prestaged_by_production(ctx, cell)
            if run.upstream_refused(name, **meta):                     # an earlier step of the chain item refused the cell
                status = 'refused upstream'
            elif done_by and ctx.dry:
                print('[%s]\n  skipped-valid: done by production: %s' % (name, done_by))
                status = 'done by production'
            elif done_by:
                # production prestaged the cell (prestage.sh:31 skips it) and its tie cache has the current header:
                # recorded read only, never run again or refused
                u = run.unit(name, done_by='production', prestage_done=done_by, **meta)
                u.status, u.reason = 'skipped-valid', 'done by production: %s' % done_by
                u.write()
                print('%s: skipped (%s)' % (name, u.reason), file=sys.stderr)
                status = 'done by production'
            elif ctx.dry:
                plan_lines = ['PRESTAGE_DONE: %s' % not_done] if not_done else []
                cell_phase(ctx, T.Stage(ctx, 'cell', T.DryUnit(), tier=TIER, in_place=()), cell, plan_lines)
                print('[%s]' % name)
                for ln in plan_lines:
                    print('  ' + ln)
                status = 'planned'
            else:
                plan_lines = ['PRESTAGE_DONE: %s' % not_done] if not_done else []
                if not_done:
                    meta['prestage_done_not_valid'] = not_done
                with run.unit(name, **meta) as u:
                    st = T.Stage(ctx, 'cell', u, tier=TIER, in_place=())
                    try:
                        cell_phase(ctx, st, cell, plan_lines)
                    finally:
                        u.meta['plan'] = plan_lines
                        st.close()
                status = u.status if not u.reason else '%s (%s)' % (u.status, u.reason[:200])
            if ctx.dry and status not in ('planned', 'done by production'):     # those two printed above
                print('[%s]\n  %s' % (name, status))
        if 'associate' in phases:
            name = '%s.associate.%s' % (UNIT, cell)
            meta = dict(tier=TIER, phase='associate', cell=cell)
            if run.upstream_refused(name, **meta):
                continue
            msg = associate_refusal(cell, status)
            if ctx.dry:
                print('[%s]' % name)
                print('  REFUSES %s' % msg)
                rc = 2
            else:
                with run.unit(name, open_choices=list(OPEN_CHOICES), **meta):
                    refuse(msg)
    if ctx.dry:
        return rc
    return None


def child_chain_refuses(clv, fleet_out, cell):
    """Child (py36, clv15 in place): snapshot_cell_prefit.sh:83-94 verbatim - the searches of the cell the chain refuses
    (phase3_snapshot_chain.build): runinfo flow-unregistered or with a paused 1-minute tier, and candidate tables of a subband
    not excluded (phase3_snapshot_search.excluded_subbands) without their 1-minute table. Prints CHAIN-REFUSES <n> <which>."""
    sys.path.insert(0, clv)
    from phase3_snapshot_search import excluded_subbands
    d = os.path.join(fleet_out, cell)
    ex = excluded_subbands(*cell.split('_', 1))
    why = []
    for rp in sorted(glob.glob(d + '/runinfo_*_median.json')):
        try:
            ri = json.load(open(rp))
        except Exception:
            continue
        if ri.get('flow_registered') is False or str(ri.get('tier_1min', '')).strip().lower() == 'paused':
            why.append(os.path.basename(rp) + (' flow-unregistered' if ri.get('flow_registered') is False
                                               else ' 1-minute tier paused'))
    for p in sorted(glob.glob(d + '/candidates_*_median.csv')):
        if os.path.basename(p).split('_')[1] not in ex and not os.path.exists(p[:-4] + '_1min.csv'):
            why.append(os.path.basename(p) + ' without its 1-minute table')
    print('CHAIN-REFUSES %d %s' % (len(why), '; '.join(why)))
    return 0


def child_fit_groups(clv, pending):
    """Child (py36, clv15 in place): snapshot_cell_prefit.sh:129-133 - the (cell, minute, band group) tasks of a request list
    (phase3_snapshot_fit._group_order). Prints FIT-GROUPS <n>."""
    import csv
    sys.path.insert(0, clv)
    from phase3_snapshot_fit import _group_order
    with open(pending) as f:
        print('FIT-GROUPS %d' % len({(r['cell'], r['ts'][:13], _group_order(r)[0]) for r in csv.DictReader(f)}))
    return 0


CHILDREN = {'store-same': (child_store_same, 3), 'chain-refuses': (child_chain_refuses, 3),
            'fit-groups': (child_fit_groups, 2)}

if __name__ == '__main__':          # child entry points through core/legacy.run (Python 3.6 + clv15 only)
    if len(sys.argv) >= 2 and sys.argv[1] in CHILDREN and len(sys.argv) == 2 + CHILDREN[sys.argv[1]][1]:
        sys.exit(CHILDREN[sys.argv[1]][0](*sys.argv[2:]))
    sys.exit('usage: incremental_snapshot.py store-same CLV STORE PRODUCTION_STORE | chain-refuses CLV FLEET_OUT CELL | '
             'fit-groups CLV PENDING')
