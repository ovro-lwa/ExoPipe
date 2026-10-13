"""`exopipe3 flow --tier deep`: the per-night co-add alignment flows of a deep cell (SURVEY / intermittent - part of a masters
build, DESIGN 12.2, chain 'deep-build-cell'; never per hour). One unit per cell (<LST>_<scope>, e.g. 23h_23MHz), named
<cell>, its record cumulative over the stages it ran (meta.stages_done):

  (masters)  `masters --tier deep --stages stage-in` writes the native, un-warped reference under the canonical names in
             <out>/<cell>/coadd_final/ (+ coadd_final_ref/) and carries the epoch-gate record. masters OWNS that
             (_CROSSCHECK decision 6, group 1); every flow stage refuses without its done record (G-D0).
  stage-in   copy the working cell's stored flows (masters' working-cell rule: paths.COADD_SB_ROOT/<cell> or its
             ref_override.csv root; --deep-flow-from DIR) into <out>/<cell>/flow/ keeping mtimes (rsync -a semantics,
             rebuild_cell.sh:44), so only stale or missing nights are recomputed. Files the masters stage-in already
             copied identically are not copied again. Output: a manifest of the copy [flow_stagein_manifest].
  prep       lib/masters/coadd.process_lst_freq IN PLACE (= production coadd_full_pipeline.py modulo imports): reference
             selection and fingerprint (.reference / .reference_resolved), astrometry gate (A-02), grid refusal, pre-cutover
             regrid, flow-input flux scale, prep/<date>_scaled.fits + _scale.json, the pair list (kept in the record).
  solve      lib/flow/batch_flow_align.py (byte-identical to production; coadd.BATCH_FLOW_SCRIPT) in a child under site
             PYTHON['flow'] through core.legacy.run, over the stale/missing pairs the prep record lists (DESIGN 11.5),
             under <out>/.gpu_lock and the site GPU locks of this host; stale outputs moved aside first (G-D13).
  check      freshness gate over every target night (G-D14: the provisional stage's own gate, applied here) and the
             content tests of every flow (also alone: the verdict on flows already on disk).

The in-place module is imported into this (py38) process; JAX never is. Module globals the step sets: WORKDIR_BASE (= the
out root) and, from masters_build (drift guard: assigned only for an explicit override), REF_MODE, REFERENCE_DATES,
CUTOVER_DATE, MIN_DATES_FOR_COADD, REF_ASTROM_MAX_AMIN; a night reference chosen at masters stage-in (--reference night:D)
sets REFERENCE_DATES[lst] for the unit, as masters' build does. Lines that differ from production: the batch call (absolute
flow interpreter instead of `conda run -n py310_orca`, ledger T2B-MERGE-COMPLETE; the manifest in the record area instead
of <work root>/_manifest_<fl>_<pid>.json; site GPU locks held in addition to <out>/.gpu_lock) and the pure fixes (G-D13
move-aside; exit codes; the freshness gate over every target night at the flow stage).
"""
import argparse
import importlib
import os
import shutil

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.flow import steps_common as C

STAGES = ('stage-in', 'prep', 'solve', 'check')
REF_I = 'I_deep_Taper_R075_{scope}.fits'          # coadd_full_pipeline.py:1620 (R-ALIGNMENT-REF-R075-EVERYWHERE)
REF_V = 'V_deep_Taper_R0_{scope}.fits'            # coadd_full_pipeline.py:1645


def add_arguments(p):
    g = p.add_argument_group('tier deep (per-night co-add flows; survey, part of a masters build)')
    g.add_argument('--deep-flow-from', dest='deep_flow_from', metavar='DIR',
                   help="stage-in: copy the stored flows from DIR (a flow/ directory with .reference_resolved) instead of "
                        "the working cell's flow/ (parity shadows)")
    g.add_argument('--deep-flow-all', dest='deep_flow_all', action='store_true',
                   help="solve: recompute every night's flow, not only stale/missing ones (coadd --stage flow --flow-all)")


def default_out(args):
    """The masters deep generation root (lib/masters/deep_stages.default_out: site MASTERS_GEN_ROOT[host], else
    masters.DEEP_GEN_ROOT), so `flow --tier deep` finds the stage-in record of `masters --tier deep` in a chain."""
    return importlib.import_module('exopipe.phase3.lib.masters.deep_stages').default_out(args)


# ---------------------------------------------------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------------------------------------------------

def items(args):
    from exopipe.phase3.core import cells
    if args.nights:
        raise Refusal('--nights is refused for tier deep: a deep flow set is every admitted night of the cell')
    if getattr(args, 'epochs', None):
        raise Refusal('--epochs is refused for tier deep: the per-night co-add flows belong to a masters build '
                      '(intermittent, DESIGN 12.2), not to the per-hour chain')
    out = list(cells.expand_cells(args.cells)) if args.cells else []
    for lst in cells.expand(args.lsts, universe=cells.LSTS, kind='LST') if args.lsts else []:
        out += [f'{lst}_{s}' for s in cells.scopes()]
    if args.scopes:
        keep = set(cells.expand(args.scopes, universe=cells.scopes(), kind='scope'))
        out = [c for c in out if cells.parse_cell(c)[1] in keep]
    if args.exclude_cells:
        drop = set(cells.expand_cells(args.exclude_cells))
        out = [c for c in out if c not in drop]
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------------------------------------------------
# the in-place co-add module
# ---------------------------------------------------------------------------------------------------------------------

def coadd_module(out_root, note):
    """lib/masters/coadd.py with the drift guard applied (masters_build and paths against the live module) and the out root
    set as its WORKDIR_BASE (R-COADD-ROOT-NO-DEFAULTS)."""
    CO = importlib.import_module('exopipe.phase3.lib.masters.coadd')
    C.drift_guard([
        ('masters_build.REF_MODE', CO.REF_MODE, lambda v: setattr(CO, 'REF_MODE', v)),
        ('masters_build.REFERENCE_DATES', CO.REFERENCE_DATES, lambda v: setattr(CO, 'REFERENCE_DATES', dict(v))),
        ('masters_build.REF_ASTROM_MAX_ARCMIN', CO.REF_ASTROM_MAX_AMIN, lambda v: setattr(CO, 'REF_ASTROM_MAX_AMIN', v)),
        ('masters_build.VLSSR_PEAKS', CO.VLSSR_REF, lambda v: setattr(CO, 'VLSSR_REF', v)),
        ('masters_build.CUTOVER_DATE', CO.CUTOVER_DATE, lambda v: setattr(CO, 'CUTOVER_DATE', v)),
        ('masters_build.MIN_DATES_FOR_COADD', CO.MIN_DATES_FOR_COADD, lambda v: setattr(CO, 'MIN_DATES_FOR_COADD', v)),
        ('paths.PHASE2_ROOT', CO.PHASE2_ROOT, lambda v: setattr(CO, 'PHASE2_ROOT', v)),
        ('paths.DATA_DECISIONS', CO.DECISIONS_CSV, None),
    ], note)
    if out_root:
        CO._update_workdir(out_root)
    return CO


def freq_str_of(scope):
    return scope[:-3] if scope.endswith('MHz') else scope


def masters_stagein_record(out_root, cell):
    from exopipe.phase3.core import record
    return record.load_record(out_root, 'masters', f'stage-in_{cell}')


def stored_flow_source(cell, args):
    """(flow dir to copy from, how): --deep-flow-from, else masters' working cell (deep_stagein.working_cell, which reads
    the sha1-pinned ref_override.csv)."""
    if args.deep_flow_from:
        return args.deep_flow_from, '--deep-flow-from'
    DS = importlib.import_module('exopipe.phase3.lib.masters.deep_stagein')
    ov, _ = DS.override_tables()
    Cdir = DS.working_cell(cell, ov)
    return os.path.join(Cdir, 'flow'), ('ref_override.csv row' if cell in ov else 'paths.COADD_SB_ROOT') + f' ({Cdir})'


def copy_flows(src, dst):
    """rsync -a SRC/ DST/ for the files that differ: a file already present with the same size and mtime (to the second, as
    masters' flows_identical) is left alone. Returns (copied, already) lists of destination paths. G-D7 on failure."""
    if not os.path.isdir(src):
        raise Refusal(f'G-D7: flow stage-in: {src} is not a directory (rebuild_cell.sh:44 "flow stage-in")')
    copied, already = [], []
    for root, _dirs, files in os.walk(src):
        tdir = os.path.normpath(os.path.join(dst, os.path.relpath(root, src)))
        os.makedirs(tdir, exist_ok=True)
        for f in files:
            s, t = os.path.join(root, f), os.path.join(tdir, f)
            try:
                if os.path.islink(s):
                    if os.path.lexists(t):
                        os.remove(t)
                    os.symlink(os.readlink(s), t)
                    copied.append(t)
                    continue
                if os.path.exists(t):
                    a, b = os.stat(s), os.stat(t)
                    if a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime):
                        already.append(t)
                        continue
                shutil.copy2(s, t)
                copied.append(t)
            except OSError as e:
                raise Refusal(f'G-D7: flow stage-in: copy {s} -> {t} failed ({e})')
    return copied, already


def stagein_manifest(src, dst):
    rows = []
    for root, _dirs, files in os.walk(dst):
        for f in sorted(files):
            p = os.path.join(root, f)
            st = os.stat(p)
            rows.append({'rel': os.path.relpath(p, dst), 'size': st.st_size, 'mtime_s': int(st.st_mtime),
                         'sha1': C.sha1_file(p) if st.st_size <= (1 << 16) else None})
    return {'source': os.path.abspath(src), 'dest': os.path.abspath(dst), 'files': rows}


def admitted_entries(CO, E, lst, scope, excl):
    """(discovered entries, admitted entries) - one rule, two readers in the window: core.exclusions and the module's
    is_excluded must agree (G-D1, B-38)."""
    fs = freq_str_of(scope)
    found = CO.discover_dates(lst, fs)
    a_core = set(E.admitted(lst, scope, [d for d, _, _ in found]))
    a_mod = {d for d, _, _ in found if not CO.is_excluded(lst, fs, d, excl)}
    if a_core != a_mod:
        raise Refusal(f'G-D1: core.exclusions and the co-add module disagree on the admitted nights of {lst}_{scope}: '
                      f'only core {sorted(a_core - a_mod)}, only module {sorted(a_mod - a_core)}')
    return found, [x for x in found if x[0] in a_core]


def targets_of(CO, cell, entries, ref_date):
    """The target nights process_lst_freq builds flows for: admitted minus grid-refused (twice, as the module does,
    coadd_full_pipeline.py:1683-1700) minus the reference night. Returns (targets, grid refused set)."""
    g0 = CO._grid_refused_dates(entries, cell)
    dates = [x for x in entries if x[0] not in g0]
    tgt = [x for x in dates if x[0] != ref_date]
    g1 = CO._grid_refused_dates(tgt, cell)
    return [x for x in tgt if x[0] not in g1], set(g0) | set(g1)


def readonly_reference(CO, lst, scope, W):
    """What _select_reference would choose, without writing: ('MASTER', signature) when the cell holds a previous-generation
    I Taper R-0.75 master with COADDOP (REF_MODE master), else (table night, None)."""
    from astropy.io import fits
    prev = os.path.join(W, 'coadd_final', REF_I.format(scope=scope))
    if CO.REF_MODE == 'master' and os.path.exists(prev):
        h = fits.getheader(prev)
        if h.get('COADDOP'):
            return 'MASTER', f"MASTER:{h.get('COADDOP')}:{h.get('CONSTDIG', 'nodigest')}"
    return CO.REFERENCE_DATES.get(lst), None


def brox_args(names):
    """The Brox parameters as CLI flags - only when the config differs from production (production passes none)."""
    f = C.cfg().flow
    defaults = {'ALPHA': 1.3, 'GAMMA': 150.0, 'SCALE_FACTOR': 0.7, 'PREPROCESS_WEIGHT': 1.5}
    out = []
    for key, flag in names:
        if getattr(f, key) != defaults[key]:
            out += [flag, repr(getattr(f, key))]
    return out


def batch_flow(u, args, cell, todo, out_root, script):
    """Port of coadd._run_batch_flow (coadd_full_pipeline.py:4186-4209): same script and pairs; through core.legacy.run
    under site PYTHON['flow'] (absolute interpreter; P3_* cleared); <out>/.gpu_lock plus the site GPU locks of this host
    held around the call; timeout BATCH_TIMEOUT_S; rc recorded (G-D15 - the refusal itself comes from the outputs)."""
    from exopipe.phase3.core import legacy
    c = C.cfg()
    mdir = os.path.join(out_root, '_records', 'flow', 'manifests')
    os.makedirs(mdir, exist_ok=True)
    man = os.path.join(mdir, f'{cell}_pairs_{C.utcstamp()}_{os.getpid()}.json')
    C.write_json_atomic(man, todo)
    gpus = C.parse_gpus(args.gpus) or C.site_gpu_ids()
    env = {'XLA_PYTHON_CLIENT_PREALLOCATE': c.flow.XLA_PREALLOCATE}
    if c.flow.XLA_MEM_FRACTION.get('deep') is not None:
        env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = c.flow.XLA_MEM_FRACTION['deep']
    if C.parse_gpus(args.gpus):
        env['CUDA_VISIBLE_DEVICES'] = gpus[0]        # production sets none (one GPU on lwacalim07)
    locks = ([os.path.join(out_root, '.gpu_lock')] if c.flow_prep.OUT_ROOT_GPU_LOCK else []) + \
        C.site_gpu_locks(gpus[0] if gpus else '*')
    log = os.path.join(out_root, '_records', 'flow', 'logs', f'{cell}_solve.log')
    os.makedirs(os.path.dirname(log), exist_ok=True)
    argv = ['--manifest', man] + brox_args((('ALPHA', '--alpha'), ('GAMMA', '--gamma'),
                                            ('SCALE_FACTOR', '--scale_factor'), ('PREPROCESS_WEIGHT', '--preprocess_weight')))
    with C.hold_locks(locks, c.flow.GPU_LOCK_WAIT_S, u.note) as held:
        res = legacy.run(script, argv, python='flow', env=env, log=log, timeout=c.flow_prep.BATCH_TIMEOUT_S,
                         refusal_codes=(), refusal_text=(), unit=u)
    u.meta['solve'] = dict(argv=res.argv, rc=res.rc, wall_s=res.wall_s, log=log, locks=held, n_pairs=len(todo), env=env,
                           manifest=man)
    for line in (res.tail or '').splitlines()[-12:]:
        u.note(f'batch_flow_align: {line}')
    if res.rc != 0:
        u.note(f'G-D15: batch_flow_align exit {res.rc}{" (timeout)" if res.rc == -9 else ""} - the outputs decide (G-D14, G-D16)')
    return res.rc


def move_aside(W, dates):
    """G-D13: move the existing outputs of the pairs to recompute to <W>/stale_flow_<stamp>/ (never deleted), so
    batch_flow_align (which skips a pair whose two outputs exist, :76-80) recomputes them."""
    stale = os.path.join(W, f'stale_flow_{C.utcstamp()}')
    moved = []
    for d in dates:
        for suf in ('_scaled_flow.npy', '_scaled_dewarp.fits'):
            p = os.path.join(W, 'flow', f'{d}{suf}')
            if os.path.exists(p):
                os.makedirs(stale, exist_ok=True)
                os.rename(p, os.path.join(stale, os.path.basename(p)))
                moved.append(p)
    return (stale if moved else None), moved


def _date_of_target(m):
    return os.path.basename(m['target']).split('_scaled')[0]


# ---------------------------------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------------------------------

def run(args, run):
    from exopipe.phase3.core import record
    stages = args.flow_stages
    try:
        cells_ = items(args)
    except ValueError as e:
        raise Refusal(str(e))
    if not cells_:
        raise NoInput('no deep cell selected (give --cells or --lsts)')
    if run.dry_run:
        return plan(args, run, cells_, stages)
    for cell in cells_:
        prev = record.load_record(run.out_root, 'flow', cell)
        if run.resume and prev and prev.get('status') == 'done' and \
                set(stages) <= set((prev.get('meta') or {}).get('stages_done') or ()):
            ok, why = run.resume_check(cell, inputs=None, admitted=prev.get('admitted'))
            if ok:
                u = run.unit(cell, tier='deep')
                u.status, u.reason = 'skipped-valid', why
                print(f'flow deep {cell}: skipped ({why})')
                continue
            print(f'flow deep {cell}: not resumed ({why})')
        with run.unit(cell, tier='deep', stages=list(stages)) as u:
            run_cell(u, args, run, cell, stages, prev)
    return None


def run_cell(u, args, run, cell, stages, prev):
    from exopipe.phase3.core import cells, exclusions, record
    out_root = run.out_root
    lst, sc = cells.parse_cell(cell)
    W = os.path.join(out_root, cell)
    u.meta.update(lst=lst, scope=sc, cell_dir=W)
    # ---- G-D0: masters owns the reference: its stage-in record must be done ---------------------------------------------
    ms = masters_stagein_record(out_root, cell)
    if not ms or ms.get('status') != 'done':
        raise Refusal(f'G-D0: no done `masters --tier deep --stages stage-in` record of {cell} under {out_root} '
                      f'({"status " + str(ms.get("status")) if ms else "none"}) - masters writes the reference the flows '
                      f'align to (_CROSSCHECK decision 6); run it first')
    u.add_input(record.record_path(out_root, 'masters', f'stage-in_{cell}'), extra={'role': 'masters stage-in record'})
    reference = (ms.get('meta') or {}).get('reference', 'previous-master')
    u.meta['masters_reference'] = reference
    if args.fresh and prev is not None:
        moves = C.supersede([os.path.join(W, 'flow'), os.path.join(W, 'prep')], out_root, out_root, args.supersede,
                            f'flow deep {cell} --fresh')
        u.note(f'--fresh: earlier flow/ and prep/ moved aside: {[m[1] for m in moves]}')
        prev = None
    done_before = list(((prev or {}).get('meta') or {}).get('stages_done') or [])
    CO = coadd_module(out_root, u.note)
    E = exclusions.load()
    u.add_input(E.path, extra={'role': 'data_decisions'})
    try:
        excl = CO.load_exclusions()
    except RuntimeError as e:
        raise Refusal(f'G-D1: {e}')
    saved = dict(CO.REFERENCE_DATES)
    if str(reference).startswith('night:'):
        CO.REFERENCE_DATES = dict(CO.REFERENCE_DATES, **{lst: reference.split(':', 1)[1]})
        u.note(f'reference night from masters stage-in: {reference} (REFERENCE_DATES[{lst}] for this unit)')
    try:
        _run_stages(u, args, run, cell, lst, sc, W, stages, prev, done_before, CO, E, excl)
    finally:
        CO.REFERENCE_DATES = saved


def _run_stages(u, args, run, cell, lst, sc, W, stages, prev, done_before, CO, E, excl):
    c = C.cfg()
    out_root = run.out_root
    fs = freq_str_of(sc)
    found, entries = admitted_entries(CO, E, lst, sc, excl)
    if not found:
        raise NoInput(f'G-D2: no Phase-2 alignment image for {cell} under {CO.PHASE2_ROOT}/{lst}')
    u.set_admitted([d for d, _, _ in entries])
    u.meta['excluded_nights'] = sorted({d for d, _, _ in found} - {d for d, _, _ in entries})
    for d, _, img in entries:
        u.add_input(img, extra={'role': f'alignment-image:{d}'})
    ref_i = os.path.join(W, 'coadd_final', REF_I.format(scope=sc))
    if os.path.exists(ref_i):
        u.add_input(ref_i, extra={'role': 'reference (masters stage-in)'})
    # ---- stage-in ----------------------------------------------------------------------------------------------------
    if 'stage-in' in stages:
        if c.flow_prep.STAGE_IN_FLOWS:
            src, how = stored_flow_source(cell, args)
            files = [os.path.join(r, f) for r, _, fl in os.walk(src) for f in fl] if os.path.isdir(src) else []
            C.add_listing_input(u, os.path.join(out_root, '_records', 'flow', 'inputs', f'{cell}_stagein_source.json'),
                                files, 'stored flows')
            copied, already = copy_flows(src, os.path.join(W, 'flow'))
            man = os.path.join(out_root, '_records', 'flow', 'manifests', f'{cell}_stagein.json')
            C.write_json_atomic(man, stagein_manifest(src, os.path.join(W, 'flow')))
            mk = os.path.join(W, 'flow', '.reference_resolved')
            u.meta['stage_in'] = {'source': src, 'how': how, 'copied': len(copied), 'already_identical': len(already),
                                  'marker': open(mk).read().strip() if os.path.exists(mk) else None}
            u.note(f'stage-in {cell}: {len(copied)} flow files copied, {len(already)} already identical (masters stage-in '
                   f'copies them too) from {src} (marker {u.meta["stage_in"]["marker"]})')
            u.add_output(man, 'flow_stagein_manifest')
        else:
            u.note('stage-in skipped: flow_prep.STAGE_IN_FLOWS is False (every night recomputed)')
    # ---- prep ----------------------------------------------------------------------------------------------------------
    prep_meta = ((prev or {}).get('meta') or {}).get('prep') if 'prep' not in stages else None
    if 'prep' in stages:
        if not os.path.exists(ref_i) and not str(u.meta.get('masters_reference')).startswith('night:'):
            raise Refusal(f'G-D8: masters stage-in wrote no reference {ref_i}')
        with C.Tee() as tee:
            try:
                manifest, prep = CO.process_lst_freq(lst, fs, excl, argparse.Namespace(flow_all=args.deep_flow_all))
            except RuntimeError as e:
                raise Refusal(f'G-D8: {e}')
        for line in tee.lines[-120:]:
            u.note(f'coadd: {line}')
        if prep is None:
            why = tee.last(r'REFUS|REJECT|Only \d+ dates|No images|not found') or 'process_lst_freq returned no prep'
            raise Refusal(f'prep refused {cell}: {why}')                         # G-D9 / G-D11 (module prints, returns None)
        targets, grid = targets_of(CO, cell, entries, prep['ref_date'])
        prep_meta = dict(ref_date=prep['ref_date'], ref_img_path=prep['ref_img_path'], manifest=manifest,
                         dates=[{k: d[k] for k in ('date', 'sci_dir', 'align_img', 'needs_regrid', 'weight')}
                                for d in prep['dates']],
                         targets=[list(x) for x in targets], grid_refused=sorted(grid),
                         shape_skipped=sorted({x[0] for x in targets} - {d['date'] for d in prep['dates']}))
        for d in prep['dates']:
            sp = os.path.join(W, 'prep', f'{d["date"]}_scaled.fits')
            inp = os.path.join(W, 'prep', f'{d["date"]}_regrid.fits') if d['needs_regrid'] else d['align_img']
            u.add_output(sp, 'flow_prep_scaled', input_image=inp)
        for g in prep_meta['grid_refused']:
            u.note(f'G-D10: {g} refused - wrong grid (Phase-2 anomaly), logged in refusals_grid.csv')
        for g in prep_meta['shape_skipped']:
            u.note(f'G-D12: {g} skipped by the prep (shape mismatch after regrid): no flow - G-D14 refuses the cell')
        if prep['ref_date'] == 'MASTER':
            for snap, name in (('ref_master_prev.fits', REF_I), ('ref_master_prev_V.fits', REF_V)):
                sp = os.path.join(W, 'prep', snap)
                if os.path.exists(sp):
                    kept = os.path.join(W, 'coadd_final_ref', name.format(scope=sc))
                    u.add_output(sp, 'flow_reference', copy_of=kept if os.path.exists(kept) else None)
        sig = os.path.join(W, 'flow', '.reference_resolved')
        prep_meta['reference_signature'] = open(sig).read().strip() if os.path.exists(sig) else None
    if prep_meta:
        u.meta['prep'] = prep_meta
    # ---- solve ---------------------------------------------------------------------------------------------------------
    recomputed = set()
    if 'solve' in stages:
        if not prep_meta:
            raise Refusal('solve needs the prep record of this unit (the pairs it lists, DESIGN 11.5): run --stages prep first')
        missing_in = [m['target'] for m in prep_meta['manifest'] if not os.path.exists(m['target'])]
        td = [(d['date'], d['sci_dir'], d['align_img']) for d in prep_meta['dates']]
        stale = {d for d, _, _ in td} if args.deep_flow_all else {d for d, _ in CO._flow_freshness(W, td)}
        stale |= _digest_stale(prev, prep_meta, W)
        todo = [m for m in prep_meta['manifest'] if _date_of_target(m) in stale]
        if [m for m in todo if m['target'] in missing_in]:
            raise Refusal(f'solve: flow inputs of the prep record are gone ({len(missing_in)}, e.g. {missing_in[:2]}; the '
                          f'masters provisional stage deletes prep/) - run --stages prep,solve,check again')
        u.note(f'{lst} {sc}: {len(todo)} of {len(prep_meta["manifest"])} epochs need a flow'
               + (f' ({", ".join(sorted(_date_of_target(m) for m in todo))})' if todo else ''))
        if todo and c.flow_prep.MOVE_ASIDE_STALE:
            sdir, moved = move_aside(W, [_date_of_target(m) for m in todo])
            if moved:
                u.note(f'G-D13: {len(moved)} outputs of stale pairs moved to {sdir}')
                u.meta['moved_aside'] = moved
        script = CO.BATCH_FLOW_SCRIPT
        u.add_input(script, extra={'role': 'code'})
        if todo:
            batch_flow(u, args, cell, todo, out_root, script)
            recomputed = {_date_of_target(m) for m in todo}
        u.meta['recomputed'] = sorted(recomputed)
    # ---- check (after solve; alone = the verdict on the flows on disk) -------------------------------------------------
    if 'solve' in stages or 'check' in stages:
        if prep_meta:
            targets = [tuple(x) for x in prep_meta['targets']]
            sigv = prep_meta.get('reference_signature')
        else:
            ref_date, sigv = readonly_reference(CO, lst, sc, W)
            targets, _ = targets_of(CO, cell, entries, ref_date)
            u.note(f'check without a prep record: targets from discovery (reference {ref_date})')
        left = CO._flow_freshness(W, targets)
        u.meta['freshness_after'] = left
        u.note(f'{lst} {sc}: flow stage done; stale/missing after: {left if left else "none"}')
        if left:
            raise Refusal(f'G-D14: flows still stale or missing: {left}')
        mk = os.path.join(W, 'flow', '.reference_resolved')
        sig_now = open(mk).read().strip() if os.path.exists(mk) else None
        prev_inputs = (prev or {}).get('inputs') or {}
        for d, _sd, img in targets:
            fl = u.plan(os.path.join(W, 'flow', f'{d}_scaled_flow.npy'), 'coadd_flow')
            dw = u.plan(os.path.join(W, 'flow', f'{d}_scaled_dewarp.fits'), 'flow_dewarp_image')
            fp = None if d in recomputed else prev_inputs.get(os.path.abspath(img))
            u.add_output(fl, 'coadd_flow', input_image=img, input_fp=fp, reference_signature=sigv or sig_now)
            u.add_output(dw, 'flow_dewarp_image')
        u.meta['reference_signature'] = sig_now
    u.meta['stages_done'] = [s for s in STAGES if s in set(done_before) | set(stages)]


def _digest_stale(prev, prep_meta, W):
    """Nights whose stored flow is fresh by mtime but whose input image changed since an earlier flow record of this unit
    computed it (record identity, DESIGN 11.5; spec Q7: both tests must pass)."""
    from exopipe.phase3.core import record
    out = set()
    pin = (prev or {}).get('inputs') or {}
    for d in prep_meta['dates']:
        fp = pin.get(os.path.abspath(d['align_img']))
        if fp and os.path.exists(os.path.join(W, 'flow', f'{d["date"]}_scaled_flow.npy')) and \
                not record.same_fingerprint(fp, record.fingerprint(d['align_img'])):
            out.add(d['date'])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------------------------------------------------

def plan(args, run, cells_, stages):
    """Print what each cell's unit would read, write and check; write nothing. Returns 0, or 2 when a gate would refuse."""
    import glob
    from exopipe.phase3.core import cells, exclusions, record
    c = C.cfg()
    rc_all = 0
    for cell in cells_:
        lst, sc = cells.parse_cell(cell)
        W = os.path.join(run.out_root, cell)
        P = []
        say = P.append
        refused = []
        try:
            CO = coadd_module(None, say)
            E = exclusions.load()
            excl = CO.load_exclusions()
            prev = record.load_record(run.out_root, 'flow', cell)
            ms = masters_stagein_record(run.out_root, cell)
            say(f'unit {cell} (tier deep, stages {",".join(stages)}); cell dir {W} '
                f'({"exists" if os.path.isdir(W) else "absent"}); flow record {prev.get("status") if prev else "none"}')
            say(f'masters stage-in record: {ms.get("status") if ms else "none"} '
                f'({record.record_path(run.out_root, "masters", "stage-in_" + cell)})')
            if not ms or ms.get('status') != 'done':
                refused.append('G-D0: no done masters stage-in record (masters writes the reference)')
            ref_i = os.path.join(W, 'coadd_final', REF_I.format(scope=sc))
            say(f'reference (masters): {ref_i} {"present" if os.path.exists(ref_i) else "ABSENT"}')
            found, entries = admitted_entries(CO, E, lst, sc, excl)
            say(f'Phase-2 alignment images: {len(found)} nights discovered, {len(entries)} admitted (core.exclusions and '
                f'the co-add module agree); excluded: {sorted({d for d, _, _ in found} - {d for d, _, _ in entries})}')
            nraw = sum(1 for _, _, img in entries if not img.endswith('_dewarped.fits'))
            if nraw:
                say(f'  {nraw} admitted night(s) without a dewarped image: the RAW image is the flow input (failsafe)')
            if not found:
                refused.append('G-D2: no Phase-2 input')
            if 'stage-in' in stages and c.flow_prep.STAGE_IN_FLOWS:
                try:
                    src, how = stored_flow_source(cell, args)
                except Refusal as e:
                    src, how = None, str(e)
                n = len(glob.glob(os.path.join(src, '*_scaled_flow.npy'))) if src else 0
                mk = os.path.join(src, '.reference_resolved') if src else ''
                old = open(mk).read().strip() if src and os.path.exists(mk) else None
                say(f'stage-in: {n} stored flows from {src} ({how}; marker {old}) -> {W}/flow/ [flow_stagein_manifest]')
            ref_date, sig = readonly_reference(CO, lst, sc, W)
            targets, grid = targets_of(CO, cell, entries, ref_date)
            say(f'prep: reference {ref_date}{" (" + sig + ")" if sig else ""}; {len(targets)} target nights; grid-refused '
                f'{sorted(grid)}; astrometry gate on the reference (A-02, fail closed) in place')
            if len(targets) < CO.MIN_DATES_FOR_COADD - 1:
                refused.append(f'G-D11: {len(targets)} target nights < {CO.MIN_DATES_FOR_COADD - 1}')
            fresh = {d for d, _ in CO._flow_freshness(W, targets)} if os.path.isdir(os.path.join(W, 'flow')) else \
                {d for d, _, _ in targets}
            for d, _, img in targets:
                say(f'  {d}: {os.path.basename(img)} -> prep/{d}_scaled.fits [flow_prep_scaled]; flow/{d}_scaled_flow.npy '
                    f'[coadd_flow] + _scaled_dewarp.fits [flow_dewarp_image] - '
                    + ('recompute (stale/missing in the out root now)' if d in fresh or args.deep_flow_all else 'fresh now'))
            if 'solve' in stages:
                gpus = C.parse_gpus(args.gpus) or C.site_gpu_ids()
                locks = ([os.path.join(run.out_root, '.gpu_lock')] if c.flow_prep.OUT_ROOT_GPU_LOCK else []) + \
                    C.site_gpu_locks(gpus[0] if gpus else '*')
                say(f'solve: {(c.site.PYTHON or {}).get("flow")} {CO.BATCH_FLOW_SCRIPT} --manifest <records>/manifests/'
                    f'{cell}_pairs_<UTC>.json{" " + " ".join(brox_args((("ALPHA", "--alpha"),))) if brox_args((("ALPHA", "--alpha"),)) else ""}'
                    f' over the stale/missing pairs{" (--deep-flow-all: all)" if args.deep_flow_all else ""}; G-D13 '
                    f'move-aside {c.flow_prep.MOVE_ASIDE_STALE}; locks {locks}; timeout {c.flow_prep.BATCH_TIMEOUT_S} s')
            say('gates: G-D0 masters stage-in, G-D1 exclusion readers agree, G-D8/9 reference + astrometry (in place), G-D10 '
                'grid, G-D11 >= 2 targets, G-D13 move-aside, G-D14 freshness of every target, G-D16 finite >= 0.5')
        except (Refusal, NoInput) as e:
            refused.append(str(e))
        for line in P:
            print(f'[plan] {line}')
        if refused:
            print(f'[plan] {cell}: WOULD REFUSE - {"; ".join(refused)}')
            rc_all = 2
    return rc_all
