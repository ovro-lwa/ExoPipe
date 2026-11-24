"""catalog --tier snapshot --incremental --epochs <LST>_<date>[,...] - the per-hour catalog step of the 10-second and 1-minute
(snapshot) tiers. Only the RULED part runs; the rest refuses.

RULED AND IMPLEMENTED - phase 'cell': the PER-CELL PRE-CHAIN STAGE of ruling 2026-09-18
(RULING-INCREMENTAL-PUBLISHING-TIE-AS-A-PER-CELL-STAGE-JUDGEMENT-CARRY-OVER; Gregg 2026-09-25 "please do", 2026-09-26
"Proceed!"), production ops/snapshot_cell_prestage.sh, for ONE night cell:
  done      on the production root, a cell production already prestaged (BATCH_DIR/cells/<cell>/PRESTAGE_DONE,
            prestage.sh:31) is recorded 'skipped-valid' (done by production; read only) and not run again
  gates     SI-G1..SI-G4, SI-G8 (below; prestage.sh:31-61, snapshot_cell_prefit.sh:73-121), in prefit.sh's order
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
  Every in-place module runs IN PLACE through core/legacy.run under the site interpreter 'py36', with the chain arguments
  and environment of the batch port (tier_snapshot.chain_args with the restore-only 3-D gate table
  catalog_snapshot.INC_CURVE_CONSTANTS_RESTORE, tier_snapshot.chain_env: what the batch driver passes).

NOT RULED - phase 'associate' REFUSES (SI-G9) on every hour: the cross-cell association, anchor, routes, vetoes, nulls and
the combined view of a new cell against the cumulative catalog. Four open choices (OPEN_CHOICES) have no ruling; until
they do, a cell enters the served catalog through the batch chain (`catalog --tier snapshot --run Run_SNAP...`, which
reads this phase's tie-cache rows) and the combined-view rebuild. The refusal is exit 2, so the per-hour chain item stops
here and is never recorded complete.

Unit of work: the epoch item <LST>_<date> = one snapshot night cell. Units 'incremental.snapshot.cell.<cell>' and
'incremental.snapshot.associate.<cell>', each with meta cell=<cell>: a refusal is a CELL refusal of the per-hour chain
(core/runner.py), and a cell an earlier step of the chain item refused is recorded UPSTREAM-REFUSED without opening the
unit (run.upstream_refused). Phases: --inc-phase (default catalog_snapshot.INC_PHASES_DEFAULT).

Inputs:  paths.SNAPSHOT_FLEET_OUT/<cell>/ (candidate tables, admission and runinfo files); catalog_snapshot.MASTER_CATALOGS,
         CURVE_CONSTANTS, INC_CURVE_CONSTANTS_RESTORE, LIGHTCURVES_ROOT, DIRTY_CELLS; curves.FLEET_CLAIMS_ROOT (G1 evidence in
         the window); the hold lists (INC_HOLD_LISTS, BATCH_DIR/rechain_active/*.cells, INC_FRAMETIME_CLAIMS) and the forced
         stage-1 markers (INC_FORCED_STATE_DIR), read only; data_decisions.csv (core.exclusions for the light-curve plan;
         the chain applies it in place).
Outputs (under --out; the production root is catalog_snapshot.BATCH_DIR, the layout of prestage.sh:11-12):
  fits/fits_<cell>.csv (+ _template.csv)     the cell's fit store, appended by fit-grouped                      csv_table
  tiecache/tie_<cell>.csv | tie2_<cell>.csv  the cell's tie-cache rows (tie2_ when tie_ has an older header)      tie_cache
  cells/<cell>/chain_summary.json            the tie pass summary (prestage.sh:81)                               snapshot_tie_summary
  cells/<cell>/prefit<i>_requests.csv        each prefit list                                                    csv_table
  cells/<cell>/prestage.log                  the chain and fitter lines prestage.sh:81 keeps
  prefit_claims/<cell>/owner                 the one-writer claim (node:pid), held while the unit runs: the directory the
                                             batch chain's P-G8 waits on (tier_snapshot.py st_preflight; prefit.sh:110)
  Work: production catalog_snapshot.WORK_ROOT/prestage/<cell> (prestage.sh:50; removed on success, prestage.sh:86); a
  non-production --out keeps it under <out>/_work/<cell>. A NON-PRODUCTION --out first COPIES the cell's production fit store
  and tie-cache files when it has none of its own (--snap-inc-seed), so it writes only under --out.
Failure modes (exit 2, the gate code first in the record's reason): GATES below. Exit 3 (no input): no detection of the
         cell reaches the per-frame fit step, or none reaches the tie. Exit 1 = error.
"""
import fcntl
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
             'pass (snapshot_cell_prefit.sh:145-152; prestage.sh:70, :76, :80)',
    'SI-G7': 'non-production seeding: the production files did not change while they were copied',
    'SI-G8': 'forced photometry stage 1 of the cell done (INC_FORCED_STATE_DIR/done/<cell>) and not marked failed '
             '(failed/<cell>), before the chain passes and again before the unit ends (prestage.sh:51-61, :83; '
             'snapshot_cell_prefit.sh:74, :115-121; Gregg 2026-10-04: the stage "won\'t mark a cell done without it")',
    'SI-G9': 'the association / combined-view phase has no ruled per-hour design (always refuses)',
    'SI-G10': 'the store content test snapshot_store_ready.py prints READY for the cell after the prefit loop '
              '(snapshot_cell_prefit.sh:167-168). It reads only the production store directory, so a non-production store '
              'is tested through it only when its newest row per key (store and template) equals the production store\'s',
    'SI-G11': 'the restore-only 3-D gate table catalog_snapshot.INC_CURVE_CONSTANTS_RESTORE exists and is not empty (as '
              'the batch chain\'s P-G21)',
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


def gate_forced(ctx, cell, plan_lines, when='before the chain passes'):
    """SI-G8: forced photometry stage 1 of the cell done and not marked failed (prestage.sh:55-61, :83; prefit.sh:74, :121).
    Production runs snapshot_forced_fleet.sh here when neither marker exists (prestage.sh:56-60) and waits while another
    fleet holds the cell; this stage refuses instead (stage 1 is its own step) and the item is retried."""
    fst = ctx.c.INC_FORCED_STATE_DIR
    done, failed = os.path.join(fst, 'done', cell), os.path.join(fst, 'failed', cell)
    held = os.path.isdir(os.path.join(fst, 'claims', cell))
    st = 'marked failed' if os.path.exists(failed) else 'done' if os.path.exists(done) else \
        'not done (a forced fleet holds it: %s)' % os.path.join(fst, 'claims', cell) if held else 'not done'
    plan_lines.append('SI-G8 forced photometry stage 1 %s: %s (%s)' % (when, st, fst))
    if st != 'done' and not ctx.dry:
        refuse('SI-G8: %s: forced photometry stage 1 %s %s (%s; prestage.sh:61, :83): a cell is not prestaged without it '
               '(Gregg 2026-10-04)%s' % (cell, st, when, failed if st == 'marked failed' else done,
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


def prestaged_by_production(ctx, cell):
    """The PRESTAGE_DONE line production wrote for the cell on the production root (prestage.sh:84), else None."""
    if not ctx.prod:
        return None
    p = os.path.join(ctx.c.BATCH_DIR, 'cells', cell, 'PRESTAGE_DONE')
    if not os.path.exists(p):
        return None
    return '%s: %s' % (p, open(p, errors='replace').read().strip()[:300])


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


def _need(ctx, cell, W, res, logname, what, pattern):
    """tier_sublst.need_ok: refuse unless the call exited 0, its log kept first."""
    ctx.T.need_ok(res, 'SI-G6: %s: %s' % (cell, what), pattern, say=lambda: _keep_log(ctx, cell, W, logname))


def _chain(ctx, s, cell, W, name, extra, env):
    """One chain pass (prestage.sh:62-65): its own --out W/<name>, log W/<name>.log, the INC_CHAIN_TIMEOUT_H limit."""
    a = ctx.chainargs(cell) + ['--cells', cell, '--out', os.path.join(W, name), '--veto-record-root', ''] + extra
    return ctx.T.call(ctx, s, ctx.chain_py, a, 'py36', env=env, log=os.path.join(W, '%s.log' % name), cwd=W,
                      timeout=int(ctx.c.INC_CHAIN_TIMEOUT_H * 3600))


def _fit(ctx, s, cell, W, pending, store, logname):
    """fit-grouped of a request list into a store (prestage.sh:74-76): procs = this process's cores minus
    catalog_snapshot.FIT_PROCS_RESERVE."""
    procs = max(1, len(os.sched_getaffinity(0)) - int(ctx.c.FIT_PROCS_RESERVE))
    return ctx.T.call(ctx, s, ctx.fit_py, ['fit-grouped', '--pending', pending, '--store', store, '--proto-root',
                                           ctx.cfg.paths.SNAPSHOT_FLEET_OUT, '--procs', str(procs)],
                      'py36', env=ctx.fit_env(), log=os.path.join(W, logname), cwd=W)


def cell_phase(ctx, s, cell, plan_lines):
    """The gates in snapshot_cell_prefit.sh's order (:73-121), the claim, then the stage (_cell_run)."""
    c = ctx.c
    gate_dirty(ctx, cell, plan_lines)                                     # prefit.sh:73
    fst_failed = os.path.join(c.INC_FORCED_STATE_DIR, 'failed', cell)
    if os.path.exists(fst_failed) and not ctx.dry:                        # prefit.sh:74
        gate_forced(ctx, cell, plan_lines)
    pl = gate_search(ctx, cell, plan_lines)                               # prefit.sh:76
    h = holds(ctx, cell)                                                  # prefit.sh:77, :78, :81
    plan_lines.append('SI-G4 holds: %s' % ('; '.join(h) or 'none'))
    if h and not ctx.dry:
        refuse('SI-G4: %s is held: %s' % (cell, '; '.join(h)))
    gate_lightcurves(ctx, cell, pl, plan_lines)                           # prefit.sh:96-107
    gate_forced(ctx, cell, plan_lines)                                    # prestage.sh:55-61
    maxit = int(c.PREFIT_MAX_ITER)
    W = ctx.work(cell)
    store = ctx.store(cell)
    plan_lines.append('store repair (<= %d rounds; prefit.sh:141-152): %s %s %s %s/repair<r>.csv%s --check; rows > 0 -> '
                      'fit-grouped into %s/repair<r>.store.csv, then %s <that store> --fits-dir %s' % (
                          int(c.INC_STORE_REPAIR_ROUNDS), ctx.py36, ctx.opsc('snapshot_store_repair.py'), cell, W,
                          '' if ctx.prod else ' --store %s' % store, W, ctx.opsc('snapshot_store_append.py'), ctx.fits))
    plan_lines.append('prefit loop (<= %d rounds; prestage.sh:66-78): %s %s %s --cells %s --out %s/prefit<i> --veto-record-root '
                      '"" --prefit-list-only; rows > 0 -> %s fit-grouped --pending %s/prefit<i>/prefit_requests.csv --store %s '
                      '--proto-root %s --procs <free cores - %d>' % (
                          maxit, ctx.py36, ctx.chain_py, ' '.join(ctx.chainargs(cell)), cell, W, ctx.fit_py, W, store,
                          ctx.cfg.paths.SNAPSHOT_FLEET_OUT, c.FIT_PROCS_RESERVE))
    plan_lines.append('SI-G10 store content test (prefit.sh:167-168): %s%s %s %s -> READY' % (
        '' if ctx.prod else 'non-production: %s store-same %s %s (newest row per key, store and template), then ' % (
            __file__, store, ctx.prod_store(cell)), ctx.py36, ctx.opsc('snapshot_store_ready.py'), cell))
    plan_lines.append('tie pass (prestage.sh:79-80): P3_TIE_CACHE_DIR=%s P3_REFASTRO_SHARDS=<%d if >= %d detections at the fit '
                      'step else %d> %s --cells %s --out %s/tie --veto-record-root "" --stop-after-tie; judged by its exit '
                      'code' % (ctx.tiecache, c.TIE_SHARDS[1], c.TIE_SHARDS[0], c.TIE_SHARDS[2], ctx.chain_py, cell, W))
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
        _need(ctx, cell, W, res, 'repair%d.log' % r, 'store repair list %d' % r, r'REPAIR-LIST|REFUSED')
        n = rows_of(lst)
        if n == 0:
            break
        nrep += n
        new = os.path.join(W, 'repair%d.store.csv' % r)
        res = _fit(ctx, s, cell, W, lst, new, 'repair_fit%d.log' % r)
        if os.path.exists(new) and os.path.getsize(new) > 0:              # what a failed round fitted is appended too
            ap = T.call(ctx, s, ctx.opsc('snapshot_store_append.py'), [new, '--fits-dir', ctx.fits], 'py36',
                        log=os.path.join(W, 'repair_append%d.log' % r), cwd=W)
            _need(ctx, cell, W, ap, 'repair_append%d.log' % r, 'store repair append %d' % r, r'APPENDED|REFUSED')
        _need(ctx, cell, W, res, 'repair_fit%d.log' % r, 'store repair fits %d' % r, r'DONE|REFUS|Error')
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
    s.u.meta['repaired'] = store_repair(ctx, s, cell, W, store)
    maxit = int(c.PREFIT_MAX_ITER)
    ok, nfit, nat, nload, it = False, 0, 0, 0, 0
    rounds = []
    for it in range(1, maxit + 1):
        res = _chain(ctx, s, cell, W, 'prefit%d' % it, ['--prefit-list-only'], ctx.chain_env())
        _need(ctx, cell, W, res, 'prefit%d.log' % it, 'prefit list %d' % it, r'REFUS|Error|PREFIT')
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
        res = _fit(ctx, s, cell, W, req, store, 'fits%d.log' % it)
        _need(ctx, cell, W, res, 'fits%d.log' % it, 'fits of prefit round %d' % it, r'DONE|REFUS|Error')
    s.u.meta.update(prefit_rounds=rounds, prefit_fitted=nfit, n_at_fit_step=nat, n_loaded=nload)
    if not ok:
        refuse('SI-G5: %s: %d detections still without their own fit after %d prefit rounds (%s/prefit%d/prefit_requests.csv)'
               % (cell, rounds[-1]['without_fit'], it, W, it))
    store_ready(ctx, s, cell, W, store)
    if nat == 0:
        no_input('%s: no detection reaches the per-frame fit step (%d loaded): nothing to tie' % (cell, nload))
    shards = ctx.S.tie_shards(nat, c.TIE_SHARDS)
    res = _chain(ctx, s, cell, W, 'tie', ['--stop-after-tie'],
                 ctx.chain_env(P3_TIE_CACHE_DIR=ctx.tiecache, P3_REFASTRO_SHARDS=shards))
    _need(ctx, cell, W, res, 'tie.log', 'tie pass', r'STOP-AFTER-TIE|REFUS|Error')    # prestage.sh:80: the exit code
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
    gate_forced(ctx, cell, plan_lines, when='at the end')                # prestage.sh:83
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
            done_by = prestaged_by_production(ctx, cell)
            if run.upstream_refused(name, **meta):                     # an earlier step of the chain item refused the cell
                status = 'refused upstream'
            elif done_by:
                # production prestaged the cell (prestage.sh:31 skips it): recorded read only, never run again or refused
                u = run.unit(name, done_by='production', prestage_done=done_by, **meta)
                u.status, u.reason = 'skipped-valid', 'done by production: %s' % done_by
                u.write()
                print('%s: skipped (%s)' % (name, u.reason), file=sys.stderr)
                status = 'done by production'
            elif ctx.dry:
                plan_lines = []
                cell_phase(ctx, T.Stage(ctx, 'cell', T.DryUnit(), tier=TIER, in_place=()), cell, plan_lines)
                print('[%s]' % name)
                for ln in plan_lines:
                    print('  ' + ln)
                status = 'planned'
            else:
                plan_lines = []
                with run.unit(name, **meta) as u:
                    st = T.Stage(ctx, 'cell', u, tier=TIER, in_place=())
                    try:
                        cell_phase(ctx, st, cell, plan_lines)
                    finally:
                        u.meta['plan'] = plan_lines
                        st.close()
                status = u.status if not u.reason else '%s (%s)' % (u.status, u.reason[:200])
            if ctx.dry and status != 'planned':
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


if __name__ == '__main__':          # child entry point through core/legacy.run (Python 3.6 + clv15 only)
    if len(sys.argv) == 5 and sys.argv[1] == 'store-same':
        sys.exit(child_store_same(*sys.argv[2:5]))
    sys.exit('usage: incremental_snapshot.py store-same CLV STORE PRODUCTION_STORE')
