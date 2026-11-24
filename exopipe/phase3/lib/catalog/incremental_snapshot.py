"""catalog --tier snapshot --incremental --epochs <LST>_<date>[,...] - the per-hour catalog step of the 10-second and 1-minute
(snapshot) tiers. Only the RULED part runs; the rest refuses.

RULED AND IMPLEMENTED - phase 'cell': the PER-CELL PRE-CHAIN STAGE of ruling 2026-09-18
(RULING-INCREMENTAL-PUBLISHING-TIE-AS-A-PER-CELL-STAGE-JUDGEMENT-CARRY-OVER; Gregg 2026-09-25 "please do", 2026-09-26
"Proceed!"), production ops/snapshot_cell_prestage.sh, for ONE night cell:
  gates     SI-G1..SI-G4 (below; prestage.sh:31-49, snapshot_cell_prefit.sh:73-112)
  prefit    FIT EVERY DETECTION BEFORE THE VETOES (Gregg 2026-10-07 Q6): at most catalog_snapshot.PREFIT_MAX_ITER rounds of
            clv15/phase3_snapshot_chain.py --cells <cell> --prefit-list-only (every detection that reaches the per-frame fit
            step without its own fit -> prefit_requests.csv), then clv15/phase3_snapshot_fit.py fit-grouped of that list into
            the cell's own fit store, until the list is empty (prestage.sh:62-78)
  tie       phase3_snapshot_chain.py --cells <cell> --stop-after-tie with the tie cache on (P3_TIE_CACHE_DIR): every
            association-pool detection is tied at its fitted position and appended to the cell's tie-cache file; cached
            detections are not tied again (prestage.sh:79-80; phase3_snapshot_tiecache.py: the tie of a detection depends only
            on the detection and its cell's frames and Master references, so it does not depend on which other cells or
            nights a run holds)
  Every in-place module runs IN PLACE through core/legacy.run under the site interpreter 'py36', with the chain arguments
  of the batch port (tier_snapshot.chain_args: the same CHAINARGS the batch driver passes).

NOT RULED - phase 'associate' REFUSES (SI-G9) on every hour: the cross-cell association, anchor, routes, vetoes, nulls and
the combined view of a new cell against the cumulative catalog. Four open choices (OPEN_CHOICES) have no ruling; until
they do, a cell enters the served catalog through the batch chain (`catalog --tier snapshot --run Run_SNAP...`, which
reads this phase's tie-cache rows) and the combined-view rebuild. The refusal is exit 2, so the per-hour chain item stops
here and is never recorded complete.

Unit of work: the epoch item <LST>_<date> = one snapshot night cell. Units 'incremental.snapshot.cell.<cell>' and
'incremental.snapshot.associate.<cell>'. Phases: --inc-phase (default catalog_snapshot.INC_PHASES_DEFAULT).

Inputs:  paths.SNAPSHOT_FLEET_OUT/<cell>/ (candidate tables, admission and runinfo files); catalog_snapshot.MASTER_CATALOGS,
         CURVE_CONSTANTS, LIGHTCURVES_ROOT, DIRTY_CELLS; curves.FLEET_CLAIMS_ROOT (G1 evidence in the window);
         data_decisions.csv (core.exclusions for the light-curve plan; the chain applies it in place).
Outputs (under --out; the production root is catalog_snapshot.BATCH_DIR, the layout of prestage.sh:11-12):
  fits/fits_<cell>.csv (+ _template.csv)     the cell's fit store, appended by fit-grouped                      csv_table
  tiecache/tie_<cell>.csv | tie2_<cell>.csv  the cell's tie-cache rows (tie2_ when tie_ has an older header)      tie_cache
  cells/<cell>/chain_summary.json            the tie pass summary (prestage.sh:81)                               snapshot_tie_summary
  cells/<cell>/prefit<i>_requests.csv        each prefit list                                                    csv_table
  cells/<cell>/prestage.log                  the chain and fitter lines prestage.sh:81 keeps
  claims/<cell>/owner                        the one-writer claim, held while the unit runs (host:pid)
  Work: production catalog_snapshot.WORK_ROOT/prestage/<cell> (prestage.sh:50; removed on success, prestage.sh:86); a
  non-production --out keeps it under <out>/_work/<cell>. A NON-PRODUCTION --out first COPIES the cell's production fit store
  and tie-cache files when it has none of its own (--snap-inc-seed), so it writes only under --out.
Failure modes (exit 2, the gate code first in the record's reason): GATES below. Exit 3 (no input): no detection of the
         cell reaches the per-frame fit step, or none reaches the tie. Exit 1 = error.
"""
import csv
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
             'evidence of curves.SEARCH_DONE_EVIDENCE holds (the fleet claim done in the window; prestage.sh:32)',
    'SI-G2': 'not a dirty Stokes I cell while catalog_snapshot.DIRTY_I_FLOOR is 0 (snapshot_cell_prefit.sh:73; the chain '
             'refuses such a cell, phase3_snapshot_chain.py:3460-3468)',
    'SI-G3': 'a light-curve table under catalog_snapshot.LIGHTCURVES_ROOT for every planned subband and Stokes I and V '
             '(prestage.sh:33-48; planned = curves plan_cell)',
    'SI-G4': 'one writer per cell: the claim <out>/claims/<cell> (prestage.sh:49; a dead owner on this host is released, '
             'snapshot_cell_prefit.sh:108-110); on the production root also no batch chain holding the cell '
             '(rechain_active/*.cells) and no production per-cell prefit claim (snapshot_cell_prefit.sh:77, :108-112)',
    'SI-G5': 'the prefit list is empty within catalog_snapshot.PREFIT_MAX_ITER rounds (prestage.sh:66-78)',
    'SI-G6': 'every in-place call exits 0: the prefit lists, the fits, the tie pass (prestage.sh:70, :76, :80)',
    'SI-G7': 'non-production seeding: the production files did not change while they were copied',
    'SI-G9': 'the association / combined-view phase has no ruled per-hour design (always refuses)',
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


def dirty_cells(path):
    """The dirty Stokes I cells of the list, parsed as phase3_snapshot_chain.py:3460-3465 does ('LST DATE' lines)."""
    out = set()
    if path and os.path.exists(path):
        for ln in open(path):
            p_ = ln.split()
            if len(p_) >= 2 and not ln.startswith('#'):
                out.add('%s_%s' % (p_[0], p_[1]))
    return out


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
        self.claims = os.path.join(self.OUT, 'claims')
        self.seeds = parse_seeds(args)
        if self.prod and getattr(args, 'snap_inc_seed', None) not in (None, 'none'):
            refuse('SI-G7: --snap-inc-seed copies production files into a NON-production --out; %s is the production root'
                   % self.OUT)
        self.clv = self.T.legacy_root('clv15')
        self.py36 = self.T.interpreter('py36', self.dry)
        self.chain_py = os.path.join(self.clv, 'phase3_snapshot_chain.py')
        self.fit_py = os.path.join(self.clv, 'phase3_snapshot_fit.py')
        self.work_arg = getattr(args, 'work', None)

    def store(self, cell):
        return os.path.join(self.fits, 'fits_%s.csv' % cell)

    def work(self, cell):
        if self.prod:
            return os.path.join(self.c.WORK_ROOT, 'prestage', cell)          # prestage.sh:50
        return os.path.join(self.OUT, '_work', cell)

    def chainargs(self, cell):
        return self.S.chain_args(self.cfg, self.c, self.store(cell), self.chain_py)

    def chain_env(self, **extra):
        """P3_SNAP_FIT_RAW unless catalog_snapshot.FLOW_MODE; a non-production run's refastro scratch on node-local disk
        (its default lies under a production root) - as tier_snapshot.Ctx.chain_env."""
        e = {}
        if not self.c.FLOW_MODE:
            e['P3_SNAP_FIT_RAW'] = '1'
        if not self.prod:
            from exopipe.phase3.core import fitsio
            from exopipe.phase3.core.record import Refusal
            try:
                root = fitsio.scratch_root(self.work_arg)
            except Refusal:
                if not self.dry:
                    raise
                root = '<node-local scratch: --work or site SCRATCH_ROOT>'
            e['P3_SNAP_REFASTRO_SCRATCH'] = os.path.join(root, 'catalog_snapshot_incremental_refastro')
        e.update(dict((k, str(v)) for k, v in extra.items()))
        return e


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


def _held_by_batch(ctx, cell):
    """Production root only: the batch chains' hold lists (rechain_active/*.cells; snapshot_cell_prefit.sh:77, :112)."""
    d = os.path.join(ctx.c.BATCH_DIR, 'rechain_active')
    hits = []
    if os.path.isdir(d):
        for n in sorted(os.listdir(d)):
            if n.endswith('.cells'):
                for ln in open(os.path.join(d, n)):
                    p_ = ln.split()
                    if p_ and not ln.startswith('#') and ('_'.join(p_[:2]) if len(p_) >= 2 else p_[0]) == cell:
                        hits.append(n)
    return hits


def claim(ctx, cell, plan_lines):
    """SI-G4: mkdir <out>/claims/<cell> with owner host:pid (prestage.sh:49); a claim of a dead process on this host is
    released first (snapshot_cell_prefit.sh:108-110). Returns the claim directory."""
    cd = os.path.join(ctx.claims, cell)
    plan_lines.append('SI-G4 one writer: claim %s%s' % (cd, '; batch-chain holds and production prefit claims under %s'
                                                         % ctx.c.BATCH_DIR if ctx.prod else ''))
    if ctx.dry:
        return None
    if ctx.prod:
        held = _held_by_batch(ctx, cell)
        if held:
            refuse('SI-G4: %s is held by a batch chain (%s under %s/rechain_active)' % (cell, ', '.join(held), ctx.c.BATCH_DIR))
        pc = os.path.join(ctx.c.BATCH_DIR, 'prefit_claims', cell)
        if os.path.isdir(pc):
            refuse('SI-G4: %s: a production per-cell prefit claim holds the cell (%s)' % (cell, pc))
    os.makedirs(ctx.claims, exist_ok=True)
    own = os.path.join(cd, 'owner')
    if os.path.isdir(cd):
        try:
            host, _, pid = open(own).read().strip().partition(':')
        except OSError:
            host, pid = '', ''
        if host == socket.gethostname() and pid.isdigit() and not os.path.exists('/proc/%s' % pid):
            shutil.rmtree(cd, ignore_errors=True)
    try:
        os.mkdir(cd)
    except FileExistsError:
        who = open(own).read().strip() if os.path.exists(own) else '?'
        refuse('SI-G4: %s is claimed by %s (%s): one writer per cell' % (cell, who, cd))
    with open(own, 'w') as f:
        f.write('%s:%d\n' % (socket.gethostname(), os.getpid()))
    return cd


def seed(ctx, s, cell, plan_lines):
    """Non-production: copy the cell's production fit store and tie-cache files that --out does not hold yet (SI-G7)."""
    if ctx.prod:
        return
    S = ctx.S
    src = {'store': [os.path.join(ctx.c.BATCH_DIR, 'fits', 'fits_%s.csv' % cell)],
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


def _chain(ctx, s, cell, W, name, extra, env):
    """One chain pass (prestage.sh:62-65): its own --out W/<name>, log W/<name>.log, the 36-hour timeout."""
    from exopipe.phase3.core import legacy
    a = ctx.chainargs(cell) + ['--cells', cell, '--out', os.path.join(W, name), '--veto-record-root', ''] + extra
    return legacy.run(ctx.chain_py, a, python='py36', env=env, cwd=W, log=os.path.join(W, '%s.log' % name), unit=s.u,
                      timeout=int(ctx.c.INC_CHAIN_TIMEOUT_H * 3600))


def _fail(ctx, cell, W, logname, msg):
    """A failed call: its log copied beside the cell's products (prestage.sh:70, :76, :80), then the refusal."""
    cdir = os.path.join(ctx.cells, cell)
    os.makedirs(cdir, exist_ok=True)
    lp = os.path.join(W, logname)
    if os.path.exists(lp):
        shutil.copy2(lp, os.path.join(cdir, logname))
    refuse(msg)


def cell_phase(ctx, s, cell, plan_lines):
    T, c = ctx.T, ctx.c
    pl = gate_search(ctx, cell, plan_lines)
    gate_dirty(ctx, cell, plan_lines)
    gate_lightcurves(ctx, cell, pl, plan_lines)
    maxit = int(c.PREFIT_MAX_ITER)
    W = ctx.work(cell)
    store = ctx.store(cell)
    plan_lines.append('prefit loop (<= %d rounds; prestage.sh:66-78): %s %s %s --cells %s --out %s/prefit<i> --veto-record-root '
                      '"" --prefit-list-only; rows > 0 -> %s fit-grouped --pending %s/prefit<i>/prefit_requests.csv --store %s '
                      '--proto-root %s --procs <free cores - %d>' % (
                          maxit, ctx.py36, ctx.chain_py, ' '.join(ctx.chainargs(cell)), cell, W, ctx.fit_py, W, store,
                          ctx.cfg.paths.SNAPSHOT_FLEET_OUT, c.FIT_PROCS_RESERVE))
    plan_lines.append('tie pass (prestage.sh:79-80): P3_TIE_CACHE_DIR=%s P3_REFASTRO_SHARDS=<%d if >= %d detections at the fit '
                      'step else %d> %s --cells %s --out %s/tie --veto-record-root "" --stop-after-tie' % (
                          ctx.tiecache, c.TIE_SHARDS[1], c.TIE_SHARDS[0], c.TIE_SHARDS[2], ctx.chain_py, cell, W))
    cd = claim(ctx, cell, plan_lines)
    try:
        seed(ctx, s, cell, plan_lines)
        if ctx.dry:
            return
        _cell_run(ctx, s, cell, W, store)
    finally:
        if cd:
            shutil.rmtree(cd, ignore_errors=True)


def _cell_run(ctx, s, cell, W, store):
    T, c = ctx.T, ctx.c
    if os.path.lexists(W):
        shutil.rmtree(W)                                                  # prestage.sh:50 rm -rf $W
    os.makedirs(W)
    os.makedirs(ctx.fits, exist_ok=True)
    cdir = os.path.join(ctx.cells, cell)
    os.makedirs(cdir, exist_ok=True)
    s.code(ctx.chain_py, ctx.fit_py)
    s.inp(store, ctx.S.template_of(store))
    maxit = int(c.PREFIT_MAX_ITER)
    ok, nfit, nat, nload, it = False, 0, 0, 0, 0
    rounds = []
    for it in range(1, maxit + 1):
        res = _chain(ctx, s, cell, W, 'prefit%d' % it, ['--prefit-list-only'], ctx.chain_env())
        if not res.ok:
            _fail(ctx, cell, W, 'prefit%d.log' % it, 'SI-G6: %s: prefit list %d: exit %s: %s' % (
                cell, it, res.rc, T.last_line(res.tail, r'REFUS|Error|PREFIT') or res.reason))
        req = os.path.join(W, 'prefit%d' % it, 'prefit_requests.csv')
        if not os.path.exists(req):
            _fail(ctx, cell, W, 'prefit%d.log' % it, 'SI-G6: %s: prefit list %d wrote no %s' % (cell, it, req))
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
        procs = max(1, len(os.sched_getaffinity(0)) - int(c.FIT_PROCS_RESERVE))
        from exopipe.phase3.core import legacy
        env = dict(ctx.chain_env(), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        env.pop('P3_SNAP_REFASTRO_SCRATCH', None)
        res = legacy.run(ctx.fit_py, ['fit-grouped', '--pending', req, '--store', store, '--proto-root',
                                      ctx.cfg.paths.SNAPSHOT_FLEET_OUT, '--procs', str(procs)],
                         python='py36', env=env, cwd=W, log=os.path.join(W, 'fits%d.log' % it), unit=s.u)
        if not res.ok:
            _fail(ctx, cell, W, 'fits%d.log' % it, 'SI-G6: %s: fits of prefit round %d: exit %s: %s' % (
                cell, it, res.rc, T.last_line(res.tail, r'DONE|REFUS|Error') or res.reason))
    s.u.meta.update(prefit_rounds=rounds, prefit_fitted=nfit, n_at_fit_step=nat, n_loaded=nload)
    if not ok:
        refuse('SI-G5: %s: %d detections still without their own fit after %d prefit rounds (%s/prefit%d/prefit_requests.csv)'
               % (cell, rounds[-1]['without_fit'], it, W, it))
    if nat == 0:
        no_input('%s: no detection reaches the per-frame fit step (%d loaded): nothing to tie' % (cell, nload))
    shards = ctx.S.tie_shards(nat, c.TIE_SHARDS)
    res = _chain(ctx, s, cell, W, 'tie', ['--stop-after-tie'],
                 ctx.chain_env(P3_TIE_CACHE_DIR=ctx.tiecache, P3_REFASTRO_SHARDS=shards))
    if not res.ok or 'STOP-AFTER-TIE' not in (res.tail or ''):
        _fail(ctx, cell, W, 'tie.log', 'SI-G6: %s: tie pass: exit %s: %s' % (
            cell, res.rc, T.last_line(res.tail, r'STOP-AFTER-TIE|REFUS|Error') or res.reason))
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
            plan_lines = []
            if ctx.dry:
                cell_phase(ctx, T.Stage(ctx, 'cell', T.DryUnit(), tier=TIER, in_place=()), cell, plan_lines)
                print('[%s]' % name)
                for ln in plan_lines:
                    print('  ' + ln)
                status = 'planned'
            else:
                with run.unit(name, tier=TIER, phase='cell', cell=cell) as u:
                    st = T.Stage(ctx, 'cell', u, tier=TIER, in_place=())
                    try:
                        cell_phase(ctx, st, cell, plan_lines)
                    finally:
                        u.meta['plan'] = plan_lines
                        st.close()
                status = u.status if not u.reason else '%s (%s)' % (u.status, u.reason[:200])
        if 'associate' in phases:
            name = '%s.associate.%s' % (UNIT, cell)
            msg = associate_refusal(cell, status)
            if ctx.dry:
                print('[%s]' % name)
                print('  REFUSES %s' % msg)
                rc = 2
            else:
                with run.unit(name, tier=TIER, phase='associate', cell=cell, open_choices=list(OPEN_CHOICES)):
                    refuse(msg)
    if ctx.dry:
        return rc
    return None
