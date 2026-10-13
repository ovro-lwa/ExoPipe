"""`exopipe3 jupiter measure` - the Phase-1 Jupiter measurement of a cell (clv15/phase3_jupiter_measure.py, in place).

Ported from exopipe_ops/jupiter_phase1_run.sh (logic items 1-11 of spec 7.1): cell list (--cells @FILE), NPAR cells at once
(--workers, default jupiter_measure.PARALLEL_CELLS), NUMPY_MADVISE_HUGEPAGE=0, interpreter per stage (jupiter.INTERPRETER),
free cores read at every stage launch (--cores auto), per-stage logs <out>/products/logs/<stage>_<cell>.log, the first
failing stage stops the cell ('FAILED <cell> at <stage>' in run.log), 'DONE <cell> <verdict>' and 'ALL-DONE' lines. Added
(refusals only, no product change): J1 window, J2 exclusions, J4 Jupiter mask and J7 baked inputs before the ~35-min
extract, J8 CASA before fit, J9 frozen deep records before assign, J10 stale fits, fit pinned to the crops' node, J12, J13.
Units: measure.<stage>.<cell> (records <out>/_records/jupiter/measure.<stage>.<cell>.json).
"""
import csv
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from exopipe.phase3.core import record
from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.jupiter import common as K
from exopipe.phase3.lib.jupiter import mirrors

TIERS = ('10s', '1min', '10min')


def _C():
    from exopipe.phase3.config import CFG
    return CFG


def module_path():
    return mirrors.module_path('measure')


# ---- plan -----------------------------------------------------------------------------------------------------------
class CellPlan:
    """Inputs of one cell as the engine will look for them (stage_extract :155-178, :271-273), with the header gates."""

    def __init__(self, cell, subbands, widebands):
        import phase3_snapshot_cube as SC          # clv15, in place (cube_name)
        J = _C().jupiter
        self.cell = cell
        self.lst, self.date = cell.split('_', 1)
        self.sl, self.cubes = J.SUBLST_ROOT, J.CUBES_ROOT
        self.subbands, self.widebands = list(subbands), list(widebands)
        self.ephem = os.path.join(J.EPHEM_DIR, J.EPHEM_NAME.format(date=self.date))
        self.pairs = []
        for band in self.subbands:
            for S in ('I', 'V'):
                sd = os.path.join(self.sl, self.lst, cell, band, S)
                p = {'band': band, 'S': S, 'adm': os.path.join(sd, f'admission_{cell}_{band}_{S}.csv'),
                     'cube': os.path.join(self.cubes, cell, SC.cube_name(cell, band, S)), 'stacks': {}, 'masters': {}}
                if not (os.path.exists(p['adm']) and os.path.exists(p['cube'])):
                    p['status'] = 'no-input'
                    self.pairs.append(p)
                    continue
                for k in range(J.WINDOWS_PER_HOUR):
                    st = os.path.join(sd, f'sublst_{cell}_{band}_{S}_w{k}.fits')
                    ra = st[:-5] + '_resid_loo.fits'
                    if os.path.exists(st) and os.path.exists(ra):
                        p['stacks'][k] = (st, ra)
                for k in range(J.WINDOWS_PER_HOUR):
                    if k in p['stacks']:
                        continue
                    mp = os.path.join(self.sl, f'{self.lst}_masters', band, S, f'sublst_master_{self.lst}_{band}_{S}_w{k}.fits')
                    if os.path.exists(mp):
                        p['masters'][k] = mp
                p['status'] = 'planned' if p['stacks'] else 'no-windows'
                self.pairs.append(p)
        self.x2 = {}
        for band in self.subbands + self.widebands:
            for S in ('I', 'V'):
                fs = sorted(glob.glob(os.path.join(self.sl, f'{self.lst}_x2', cell, band, S,
                                                   f'sublst_{cell}_{band}_{S}_w[0-9]_resid_loo.fits')))
                if fs:
                    self.x2[(band, S)] = fs
        self._headers_done = False

    @property
    def planned(self):
        return [p for p in self.pairs if p['status'] == 'planned']

    def scopes_with_input(self):
        return sorted({p['band'] for p in self.planned} | {b for (b, _) in self.x2})

    def headers(self):
        """J4 (JUPMASK on arm-A and x2 subband residuals; window Masters used only with it) and J7 (FLUXBAKE) from the
        primary headers (header-only reads)."""
        if self._headers_done:
            return
        from concurrent.futures import ThreadPoolExecutor as _TP
        files = [x for p in self.planned for pair in p['stacks'].values() for x in pair]
        files += [mp for p in self.planned for mp in p['masters'].values()]
        files += [x for fs in self.x2.values() for x in fs]
        with _TP(max_workers=16) as ex:            # header-only reads, Lustre latency bound
            cards = dict(zip(files, ex.map(record.fits_primary_cards, files)))
        self.no_jupmask, self.baked, self.masters_unmasked, self.masters_used = [], [], [], {}
        for p in self.planned:
            for k, (st, ra) in sorted(p['stacks'].items()):
                hs, hr = cards[st], cards[ra]
                if 'JUPMASK' not in hr:
                    self.no_jupmask.append(ra)
                for path, h in ((st, hs), (ra, hr)):
                    if h.get('FLUXBAKE') is True:
                        self.baked.append(path)
            for k, mp in sorted(p['masters'].items()):
                hm = cards[mp]
                if 'JUPMASK' not in hm:
                    self.masters_unmasked.append(mp)
                    continue
                self.masters_used.setdefault((p['band'], p['S']), {})[k] = mp
                if hm.get('FLUXBAKE') is True:
                    self.baked.append(mp)
        for (band, S), fs in sorted(self.x2.items()):
            for path in fs:
                h = cards[path]
                if band in self.subbands and 'JUPMASK' not in h:
                    self.no_jupmask.append(path)
                if h.get('FLUXBAKE') is True:
                    self.baked.append(path)
        self._headers_done = True

    def input_files(self):
        out = [self.ephem]
        for p in self.planned:
            out += [p['adm'], p['cube']]
            for st, ra in p['stacks'].values():
                out += [st, ra]
            out += list(self.masters_used.get((p['band'], p['S']), {}).values()) if self._headers_done else []
        for fs in self.x2.values():
            out += fs
        return out


def frozen_records(lst, bands):
    """The deep co-add records tier rule 4 reads (phase3_tier_rules.flux_scale_failed, in place) for these bands."""
    import phase3_tier_rules as TR
    subs = []
    for b in bands:
        subs += TR.WIDEBAND_MEMBERS.get(b, [b])
    return {sb: os.path.join(TR.COADD_RUN, lst, sb, 'weights.csv') for sb in sorted(set(subs), key=lambda s: (len(s), s))}


def meas_bands(od):
    bands = set()
    for t in TIERS:
        p = os.path.join(od, f'meas_{t}.csv')
        if os.path.exists(p):
            with open(p, newline='') as f:
                bands |= {r['band'] for r in csv.DictReader(f)}
    return sorted(bands)


# ---- the action -----------------------------------------------------------------------------------------------------
def run(args, run_, roots, stages):
    J = _C().jupiter
    cells = K.select_cells(args, run_)
    if not cells:
        raise NoInput('no Jupiter cell selected')
    scopes = [s.strip() for s in (args.scopes or '').split(',') if s.strip()]
    bad = [s for s in scopes if s not in J.SUBBANDS + J.WIDEBANDS]
    if bad:
        raise Refusal(f'--scopes {bad}: the measure takes subbands {J.SUBBANDS[0]}..{J.SUBBANDS[-1]} and {", ".join(J.WIDEBANDS)}')
    if args.stokes and sorted(s.strip() for s in args.stokes.split(',')) != ['I', 'V']:
        raise Refusal('--stokes: the engine measures I and V together (literal loops, phase3_jupiter_measure.py:156, :272); '
                      'a one-Stokes run would need a code change')
    if scopes and roots.production:
        raise Refusal('J13 whole cell into production: --scopes is accepted only with a non-production --out (a production '
                      'cell product holds every band and both Stokes)')
    subbands = [s for s in J.SUBBANDS if not scopes or s in scopes]
    widebands = [w for w in J.WIDEBANDS if not scopes or w in scopes]
    scope_globals = {'BANDS': subbands, 'WIDE': widebands} if scopes else {}
    mirrors.check_literals(mirrors.ACTION_MODULES['measure'])
    if run_.dry_run:
        return dry_run(args, run_, roots, stages, cells, subbands, widebands)
    work = roots.need_work()
    workers = max(1, args.workers or _C().jupiter_measure.PARALLEL_CELLS)
    t0 = time.time()

    def one(cell):
        return run_cell(args, run_, roots, stages, cell, subbands, widebands, scope_globals, work)

    if workers == 1 or len(cells) == 1:
        for c in cells:
            one(c)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(cells))) as ex:
            list(ex.map(one, cells))
    if args.cells.startswith('@'):
        K.runlog(roots.logs, f'ALL-DONE {args.cells[1:]} ({len(cells)} cells, {time.time() - t0:.0f} s)')
    return None


def _unit_name(stage, cell):
    return f'measure.{stage}.{cell}'


def run_cell(args, run_, roots, stages, cell, subbands, widebands, scope_globals, work):
    """The cell's stages in order on this node; the first stage not done stops the cell (launcher :15)."""
    plan = CellPlan(cell, subbands, widebands)
    for stage in stages:
        name = _unit_name(stage, cell)
        if args.resume and not args.fresh:
            ok, why = run_.resume_check(name, inputs=None, admitted=[plan.date])
            if ok:
                u = run_.unit(name, action='measure', stage=stage, cell=cell)
                u.status, u.reason = 'skipped-valid', why
                print(f'jupiter measure {name}: skipped (resume: {why})', file=sys.stderr)
                continue
        try:
            with run_.unit(name, action='measure', stage=stage, cell=cell, host=K.host()) as u:
                u.set_admitted([plan.date])
                K.pin_free_cores(args)
                STAGE_FN[stage](u, args, run_, roots, plan, scope_globals, work)
        except Exception as e:          # the unit record says error; the other cells go on (launcher: per-cell failure)
            print(f'jupiter measure {name}: error ({type(e).__name__}: {e})', file=sys.stderr)
            K.runlog(roots.logs, f'FAILED {cell} at {stage}')
            return False
        print(f'jupiter measure {name}: {u.status}' + (f' ({u.reason})' if u.reason else ''), file=sys.stderr)
        if u.status != 'done':
            K.runlog(roots.logs, f'FAILED {cell} at {stage}')
            return False
        if stage == 'test':
            ct = os.path.join(roots.products, cell, 'CONTENT_TEST.txt')
            last = open(ct).read().strip().splitlines()[-1] if os.path.exists(ct) else '?'
            K.runlog(roots.logs, f'DONE {cell} {last}')
    return True


def _call(u, plan, roots, stage, fn_args, globals_, work, test_stage=False):
    res = K.in_place(u, module_path(), f'stage_{stage}', args=fn_args, globals_=globals_,
                     python=K.interpreter_key(f'measure.{stage}'), log=os.path.join(roots.logs, f'{stage}_{plan.cell}.log'),
                     cwd=os.path.join(work, plan.cell), guard=roots.guard, test_stage=test_stage)
    if res.kind == 'refused':
        u.refuse(f'{stage}: {res.reason}')
    if res.kind == 'error':
        raise RuntimeError(f'{stage} engine error: {res.reason} (log {res.log})')
    return res


def stage_extract(u, args, run_, roots, plan, scope_globals, work):
    if not os.path.exists(plan.ephem):
        u.refuse(f'no ephemeris cache {plan.ephem} for {plan.date} (exopipe3 jupiter ephem --dates {plan.date}, on a host '
                 f'that reaches JPL)')
    if not plan.planned and not plan.x2:
        raise NoInput(f'J3 no input: no (band, Stokes) of {plan.cell} has its admission table, cube and a window stack, and '
                      f'no x2 residual exists ({len(plan.pairs)} pairs looked for)')
    excl, fp = K.exclusion_gate(plan.lst, plan.date, plan.scopes_with_input())
    u.meta['exclusions'] = fp
    if excl:
        u.refuse(f'J2 exclusions: {plan.lst} {plan.date} {", ".join(excl)} excluded in data_decisions.csv but inputs exist '
                 f'(rebuild the stacks upstream first; exclusion is absolute)')
    plan.headers()
    if plan.no_jupmask:
        u.refuse(f'J4 Jupiter mask: {len(plan.no_jupmask)} residual(s) without the JUPMASK card ({K.short(plan.no_jupmask)}) '
                 f'(RULING-MOVING-JUPITER-MASK-IN-MASTERS; DEFECT-SUBLST-ARM-B-LOO-WITHOUT-JUPITER-MASK)')
    if plan.baked and not run_.waived('J7'):
        u.refuse(f'J7 baked inputs: {len(plan.baked)} input(s) carry FLUXBAKE True ({K.short(plan.baked)}); the measure divides '
                 f'by FLUXDC, valid only for unbaked stacks (spec 12.1, defect RI-clv-snapshot-rest-sublst-004) - every 10-s and '
                 f'1-min value would be wrong. Waivable only by --waive J7 --ruling (no ruling exists)')
    if plan.masters_unmasked:
        u.note(f'{len(plan.masters_unmasked)} window Master(s) without JUPMASK not used by the engine: {K.short(plan.masters_unmasked)}')
    for p in plan.pairs:
        if p['status'] != 'planned':
            u.note(f"J3/J5 {p['band']} {p['S']}: {p['status']}")
    for path in plan.input_files():
        u.add_input(path)
    u.add_input(_C().paths.DATA_DECISIONS)
    t0 = time.time() - 1
    gl = mirrors.redirect_globals('measure', scope_globals)
    _call(u, plan, roots, 'extract', [plan.cell, roots.products, work], gl, work)
    od = os.path.join(roots.products, plan.cell)
    fresh = lambda p: os.path.getmtime(p) >= t0          # noqa: E731
    written = 0
    for t in TIERS:
        p = os.path.join(od, f'meas_{t}.csv')
        if os.path.exists(p) and fresh(p):
            u.add_output(p, 'jupiter_meas')
            written += 1
    for p in sorted(glob.glob(os.path.join(od, 'cutouts_*.npz'))):
        if fresh(p):
            u.add_output(p, 'jupiter_cutouts')
    # the crops are node-local; their class test runs through the table whose rows name them (absolute paths)
    for t in ('10s', '1min'):
        p = os.path.join(od, f'meas_{t}.csv')
        if os.path.exists(p) and fresh(p):
            u.add_output(p, 'jupiter_crop')
    stale = [p for p in glob.glob(os.path.join(od, 'meas_*.csv')) + glob.glob(os.path.join(od, 'cutouts_*.npz')) if not fresh(p)]
    u.check('jupiter.measure.no_stale_extract_outputs', not stale, len(stale),
            f'{len(stale)} measure product(s) in {od} predate this extract and were not rewritten: {K.short(stale)} '
            f'(move them aside with --fresh --supersede TAG)' if stale else 'every meas/cutouts file rewritten')
    down = [p for p in glob.glob(os.path.join(od, 'fits_*.jsonl')) + glob.glob(os.path.join(od, 'assign_*.csv'))
            + [os.path.join(od, 'CONTENT_TEST.txt')] if os.path.exists(p) and not fresh(p)]
    if down:
        u.note(f'downstream products predate this extract (run --from fit): {K.short(down)}')
    if not written:
        raise NoInput(f'J3/J5: the engine measured nothing for {plan.cell} (every band NO-INPUT, NO-WINDOWS or EDGE; see '
                      f'{os.path.join(roots.logs, "extract_" + plan.cell + ".log")})')


def _crop_host_check(u, run_, roots, plan, od):
    for t in ('10s', '1min'):
        p = os.path.join(od, f'meas_{t}.csv')
        if not os.path.exists(p):
            continue
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                if r['crop']:
                    if not os.path.exists(r['crop']):
                        rec = record.load_record(run_.out_root, 'jupiter', _unit_name('extract', plan.cell)) or {}
                        u.refuse(f"the crops of {plan.cell} are node-local and not on {K.host()} (first: {r['crop']}); the "
                                 f"extract ran on {rec.get('host', 'an unknown host')} - run fit there")
                    return


def stage_fit(u, args, run_, roots, plan, scope_globals, work):
    od = os.path.join(roots.products, plan.cell)
    meas = [os.path.join(od, f'meas_{t}.csv') for t in TIERS if os.path.exists(os.path.join(od, f'meas_{t}.csv'))]
    if not meas:
        raise NoInput(f'no meas_<tier>.csv in {od}: run the extract stage first')
    K.check_casa(K.interpreter_key('measure.fit'), work)
    _crop_host_check(u, run_, roots, plan, od)
    old = [p for p in glob.glob(os.path.join(od, 'fits_*.jsonl'))]
    if old:
        rf = record.load_record(run_.out_root, 'jupiter', _unit_name('fit', plan.cell))
        re_ = record.load_record(run_.out_root, 'jupiter', _unit_name('extract', plan.cell))
        # only a DONE fit record vouches for the fits on disk (a refused or errored one is newer than the extract it
        # refused on and would hide the staleness); --fresh always moves existing fits aside (the engine appends and skips)
        stale = (rf is None or rf.get('status') != 'done' or re_ is None
                 or (re_.get('finished') or '') > (rf.get('finished') or ''))
        if args.fresh:
            d = K.supersede(old, od, args.supersede)
            u.note(f'J10: {len(old)} fits file(s) moved aside to {d}')
        elif stale:
            u.refuse(f'J10 stale fits: {K.short(old)} exist but no fit record of this root is newer than the extract record '
                     f'(the engine appends and skips crops already fitted, :314-323) - re-run with --fresh --supersede TAG')
    for p in meas:
        u.add_input(p)
    _call(u, plan, roots, 'fit', [plan.cell, roots.products], {}, work)
    for t in TIERS:
        if os.path.exists(os.path.join(od, f'meas_{t}.csv')):
            p = os.path.join(od, f'fits_{t}.jsonl')
            if os.path.exists(p):
                u.add_output(p, 'jupiter_fits')
            else:
                u.check(f'jupiter.measure.fits_written[{t}]', False, None, f'{p} not written although meas_{t}.csv exists')


def stage_assign(u, args, run_, roots, plan, scope_globals, work):
    od = os.path.join(roots.products, plan.cell)
    pairs = [(t, os.path.join(od, f'meas_{t}.csv'), os.path.join(od, f'fits_{t}.jsonl')) for t in TIERS]
    pairs = [x for x in pairs if os.path.exists(x[1]) and os.path.exists(x[2])]
    if not pairs:
        raise NoInput(f'no meas_<tier>.csv with its fits_<tier>.jsonl in {od}: run extract and fit first')
    if not os.path.exists(plan.ephem):
        u.refuse(f'no ephemeris cache {plan.ephem}')
    recs = frozen_records(plan.lst, meas_bands(od))
    missing = [p for p in recs.values() if not os.path.exists(p)]
    if missing:
        u.refuse(f'J9 tier rule 4: deep co-add record(s) missing: {K.short(missing)} (phase3_tier_rules.flux_scale_failed '
                 f'would raise REFUSING)')
    for _, m, f in pairs:
        u.add_input(m)
        u.add_input(f)
    for p in list(recs.values()) + [plan.ephem]:
        u.add_input(p)
    _call(u, plan, roots, 'assign', [plan.cell, roots.products], mirrors.redirect_globals('measure'), work)
    for t, _, _ in pairs:
        for name in (f'assign_{t}.csv', f'moons_{t}.json'):
            p = os.path.join(od, name)
            if os.path.exists(p):
                u.add_output(p, 'jupiter_assign')
            else:
                u.check(f'jupiter.measure.assign_written[{name}]', False, None, f'{p} not written')


def stage_test(u, args, run_, roots, plan, scope_globals, work):
    od = os.path.join(roots.products, plan.cell)
    if not glob.glob(os.path.join(od, 'meas_*.csv')):
        raise NoInput(f'no meas_<tier>.csv in {od}: nothing to test')
    for p in sorted(glob.glob(os.path.join(od, 'meas_*.csv')) + glob.glob(os.path.join(od, 'fits_*.jsonl'))
                    + glob.glob(os.path.join(od, 'assign_*.csv'))):
        u.add_input(p)
    plan.headers()
    for p in plan.input_files():
        u.add_input(p)
    gl = mirrors.redirect_globals('measure', scope_globals)
    lit = mirrors.literal_of('jupiter_measure.CROP_ROOT')[1]
    if os.path.abspath(work) != os.path.abspath(lit):
        gl['SCRATCH_DEFAULT'] = work          # stage_test reads SCRATCH_DEFAULT, not --scratch (:449)
    t0 = time.time() - 1
    res = _call(u, plan, roots, 'test', [plan.cell, roots.products], gl, work, test_stage=True)
    ct = os.path.join(od, 'CONTENT_TEST.txt')
    if not os.path.exists(ct) or os.path.getmtime(ct) < t0:
        raise RuntimeError(f'the test stage wrote no fresh {ct}')
    u.add_output(ct, 'jupiter_cell', fresh_verdict=True, engine_kind=res.kind)


STAGE_FN = {'extract': stage_extract, 'fit': stage_fit, 'assign': stage_assign, 'test': stage_test}


# ---- dry run --------------------------------------------------------------------------------------------------------
def dry_run(args, run_, roots, stages, cells, subbands, widebands):
    J = _C().jupiter
    print(f'jupiter measure: {len(cells)} cell(s), stages {",".join(stages)}; {roots.describe()}')
    print(f'  engine {module_path()} (in place); interpreters ' +
          ', '.join(f'{s}={K.interpreter_key("measure." + s)}' for s in stages))
    uni = K.window_universe()
    gl = mirrors.redirect_globals('measure', {'BANDS': subbands, 'WIDE': widebands} if args.scopes else {})
    print(f'  module globals set by the driver: {gl or "none (production values)"}')
    any_refusal = False
    for cell in cells:
        plan = CellPlan(cell, subbands, widebands)
        plan.headers()
        planned = plan.planned
        print(f'\n  cell {cell}: Jupiter {uni.get(cell, float("nan")):.1f} deg at mid-hour (window {"yes" if cell in uni else "NO - J1"})')
        print(f'    ephemeris {plan.ephem}: {"present" if os.path.exists(plan.ephem) else "MISSING"}')
        st = {}
        for p in plan.pairs:
            st.setdefault(p['status'], []).append(f"{p['band']}/{p['S']}")
        for k, v in sorted(st.items()):
            print(f'    pairs {k}: {len(v)}' + ('' if k == 'planned' else f' ({", ".join(v[:10])}{" ..." if len(v) > 10 else ""})'))
        nst = sum(len(p['stacks']) for p in planned)
        print(f'    window stacks + arm-A residuals: {nst}; window Masters used (JUPMASK): '
              f'{sum(len(v) for v in plan.masters_used.values())}; without JUPMASK (not used): {len(plan.masters_unmasked)}')
        print(f'    x2 residuals: {sum(len(v) for v in plan.x2.values())} files over {len(plan.x2)} (band, Stokes)')
        excl, _ = K.exclusion_gate(plan.lst, plan.date, plan.scopes_with_input())
        gates = [('J2 exclusions', not excl, ', '.join(excl) or 'none excluded'),
                 ('J4 JUPMASK', not plan.no_jupmask, f'{len(plan.no_jupmask)} residual(s) without it' + (
                     f': {K.short(plan.no_jupmask)}' if plan.no_jupmask else '')),
                 ('J7 unbaked inputs', not plan.baked or bool(run_.waived('J7')),
                  f'{len(plan.baked)} input(s) with FLUXBAKE True' + (f': {K.short(plan.baked)}' if plan.baked else ''))]
        od = os.path.join(roots.products, cell)
        if 'assign' in stages:
            recs = frozen_records(plan.lst, plan.scopes_with_input())
            miss = [p for p in recs.values() if not os.path.exists(p)]
            gates.append(('J9 frozen deep records', not miss, f'{len(recs) - len(miss)} of {len(recs)} present'))
        if 'fit' in stages:
            old = glob.glob(os.path.join(od, 'fits_*.jsonl'))
            rf = record.load_record(run_.out_root, 'jupiter', _unit_name('fit', cell))
            rx = record.load_record(run_.out_root, 'jupiter', _unit_name('extract', cell))
            current = (rf is not None and rx is not None and 'extract' not in stages
                       and (rx.get('finished') or '') <= (rf.get('finished') or ''))
            moved = bool(args.fresh and args.supersede)
            gates.append(('J10 stale fits', not old or current or moved,
                          f'{len(old)} existing fits file(s); fit record {"present" if rf else "absent"}, extract record '
                          f'{"present" if rx else "absent"}' + (' - moved aside first (--fresh --supersede)' if old and moved else '')
                          + (' - extract re-runs first, so fit refuses unless --fresh --supersede TAG'
                             if old and 'extract' in stages and not moved else '')))
        for name, ok, detail in gates:
            print(f'    gate {name}: {"pass" if ok else "REFUSE"} - {detail}')
            any_refusal |= not ok
        if not planned and not plan.x2:
            print(f'    J3 no input: no (band, Stokes) with admission table, cube and a window stack, no x2 residual - the '
                  f'extract unit would end no-input (exit 3)')
        for stage in stages:
            print(f'    unit {_unit_name(stage, cell)}: stage_{stage}(' +
                  {'extract': f'{cell!r}, {roots.products!r}, {roots.work!r}', 'fit': f'{cell!r}, {roots.products!r}',
                   'assign': f'{cell!r}, {roots.products!r}', 'test': f'{cell!r}, {roots.products!r}'}[stage] +
                  f') log {os.path.join(roots.logs, stage + "_" + cell + ".log")}')
        outs = {'extract': 'meas_<tier>.csv [jupiter_meas; its crops in '
                           f'{os.path.join(roots.work or "<work>", cell, "crops")}: jupiter_crop], '
                           'cutouts_<tier>_<band>_<S>.npz [jupiter_cutouts]',
                'fit': 'fits_<tier>.jsonl [jupiter_fits]', 'assign': 'assign_<tier>.csv, moons_<tier>.json [jupiter_assign]',
                'test': 'CONTENT_TEST.txt [jupiter_cell]'}
        for stage in stages:
            print(f'    outputs {stage}: {od}/' + outs[stage])
        if args.resume:
            for stage in stages:
                ok, why = run_.resume_check(_unit_name(stage, cell), inputs=None, admitted=[plan.date])
                print(f'    resume {stage}: {"skip" if ok else "run"} ({why})')
    if roots.production and not _C().paths.PRODUCTION_WRITES:
        print('\n  NOTE: --out is a production root and paths.PRODUCTION_WRITES is False: a real run refuses (J14)')
    if not roots.work:
        print(f'\n  NOTE: no node-local scratch root on {K.host()}: a real run refuses ({roots.work_problem}); pass --work')
    print(f'\n  plan: {"a real run would REFUSE at least one cell (gates above)" if any_refusal else "every gate passes"}')
    return record.EXIT_REFUSED if any_refusal else record.EXIT_DONE
