"""D-ADMIT (a): the masters stages that admit a night newer than its cell's installed master (plan/wave2/DADMIT_DESIGN.md;
ruling R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32). On demand, after the cell's population record exists
(`masters --stages build-stats`):
  admit-prep   (slice 2, CPU) the nights the cell admits (admit.nights_to_admit), the reference (the installed masters in
               native units, admit/ref/), and each night's flow input by the co-add's own per-night prep
               (coadd.prep_night): <cell>/admit/prep/<date>_scaled.fits + _scale.json [flow_prep_scaled]; the flow pairs in
               the unit's record for the flow step.
  (flow)       `exopipe3 flow --tier deep --stages admit` (slice 3, GPU): the pairs' flows into <cell>/admit/flow/.
  admit        (slice 4) the admitted night's products in the installed generation, its gates, admissions.csv.
Each writes into the cell tree it is given (--deep-installed-root, default paths.COADD_SB_ROOT): a production root refuses
while paths.PRODUCTION_WRITES is False, so until cutover a stage runs on a copy of the cell. --out holds the run records.
"""
import os

from exopipe.phase3.core.record import NoInput, Refusal


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def cell_tree(args, cell):
    """The cell directory the admission stages read and write: <--deep-installed-root or paths.COADD_SB_ROOT>/<cell>."""
    return os.path.join(getattr(args, 'deep_installed_root', None) or _cfg().paths.COADD_SB_ROOT, cell)


def epochs_of(args, lst):
    """The nights --epochs names for this LST (None: no --epochs, every qualifying night; [] when --epochs names none of
    this LST's - then no night, never every night: 2026-10-15 review)."""
    from exopipe.phase3.core import cells as C
    if not getattr(args, 'epochs', None):
        return None
    return sorted({d for l, d in (C.parse_epoch(e) for e in C.expand_epochs(args.epochs)) if l == lst})


def nights(cell, args, co):
    """(take, why) for the cell tree the stage is given: admit.nights_to_admit over the co-add's Phase-2 nights of the cell,
    --epochs restricting - the stage's work list, its plan and its resume identity (the nights it admits now)."""
    import contextlib
    import io
    from exopipe.phase3.core import exclusions
    from exopipe.phase3.lib.masters import admit as A, deep_stages as DS
    lst, sc = cell.split('_', 1)
    with contextlib.redirect_stdout(io.StringIO()):          # the finders' failsafe notes: the build's, not ours
        entries = co.discover_dates(lst, DS._freq(sc))
    return A.nights_to_admit(lst, sc, cell_tree(args, cell), entries, exclusions.load(), co, nights=epochs_of(args, lst))


def prep_inputs(cell, args, co):
    """(take, why, inputs): the nights admit-prep takes now and every input its record holds for them, as the stage records
    them - the build's night record, the population record, the installed I and V masters, each night's survey image. Also
    the stage's --resume identity: another cell tree, a re-imaged night or a replaced record re-runs it (2026-10-15 review:
    comparing the nights alone skipped all three)."""
    from exopipe.phase3.lib.masters import admit as A, build_stats as BS, deep_stagein as DSI
    cd = cell_tree(args, cell)
    sc = cell.split('_', 1)[1]
    take, why = nights(cell, args, co)
    masters = [os.path.join(cd, 'coadd_final', f'{var}_{sc}.fits') for _st, var, _rob in DSI.REF_VARS]
    inputs = ([os.path.join(cd, A.BUILD_NIGHTS), os.path.join(cd, 'coadd_final', BS.FILE)] + masters
              + [img for _d, _sd, img in take])
    return take, why, list(dict.fromkeys(inputs))


def population_problem(cd, cell, co):
    """Why the cell's population record is not one to judge an admitted night by, else None: missing, failing its content
    test (build_stats.recheck - also a record a failed --fresh left in place), or holding a night excluded now (the
    build-stats stage's rule: exclusions are absolute)."""
    from exopipe.phase3.lib.masters import build_stats as BS, deep_stages as DS
    p = os.path.join(cd, 'coadd_final', BS.FILE)
    if not os.path.exists(p):
        return (f'{p} missing: the admitted night is judged against the build\'s population - run `masters --tier deep '
                f'--stages build-stats` on the cell first')
    try:
        bad = BS.recheck(p, co)
    except (OSError, ValueError, KeyError) as e:
        bad = [f'unreadable ({type(e).__name__}: {e})']
    if bad:
        return f'{p} fails its content test ({bad[0][:300]}) - re-run `masters --tier deep --stages build-stats --fresh`'
    _nights, stale = DS.build_stats_population(cd, cell)
    if stale:
        return (f'{cell}: night(s) {stale} of the build population are excluded in data_decisions.csv now - a population '
                f'holding them is not one to judge an admitted night by (rebuild the cell)')
    return None


def off_grid(d, img, target_shape, cutover):
    """B-03 (a) at admission: a post-cutover night whose survey image is not on the installed master's grid (shape, the
    co-add's test) - a Phase-2 wrong-grid night, kept out of a build until data_decisions.csv excludes it
    (input_resolver.wrong_grid_refusal), so never regridded into an admission. (shape) or None; 'unreadable' when the
    header cannot be read (the night is refused - nothing is assumed about its grid)."""
    from astropy.io import fits
    if d < cutover:
        return None
    try:
        h = fits.getheader(img)
        shape = (int(h['NAXIS2']), int(h['NAXIS1']))
    except Exception:                               # noqa: BLE001 - any unreadable header refuses the night
        return 'unreadable'
    return shape if shape != tuple(target_shape) else None


def move_aside(W, d, stamp):
    """A night's earlier prep files (<cell>/admit/prep/<d>_regrid.fits, _scaled.fits, _scale.json) moved to
    admit/prep/_superseded/<stamp>/ before it is prepared again - coadd.regrid_image reuses an existing regrid, so a re-run
    after a re-image would scale the old pixels (2026-10-15 review). Never deleted; ``stamp`` is unique per run (UTC and
    process id). Returns the names moved."""
    prep = os.path.join(W, 'prep')
    moved = []
    for f in (f'{d}_regrid.fits', f'{d}_scaled.fits', f'{d}_scale.json'):
        if os.path.exists(os.path.join(prep, f)):
            dest = os.path.join(prep, '_superseded', stamp)
            os.makedirs(dest, exist_ok=True)
            os.replace(os.path.join(prep, f), os.path.join(dest, f))
            moved.append(f)
    return moved


def reference_dict(co, path, ent):
    """The reference as coadd.prep_night takes it, from the written copy and its entry (installed_reference): data, WCS,
    beam in pixels and horizon mask as process_lst_freq derives them from its reference, the template grid of the copy."""
    import numpy as np
    from astropy.wcs import WCS
    tmpl_hdr, target_shape = co.build_template_header(path)
    data, hdr = co.load_image(path)
    cd = abs(hdr.get('CDELT1', hdr.get('CD1_1', 0.037)))
    return dict(img_path=path, data=data, wcs=WCS(hdr).celestial, beam_pix=hdr.get('BMAJ', 5 * cd) / cd,
                horizon_mask=~np.isfinite(data) | (data == 0), tmpl_hdr=tmpl_hdr, target_shape=target_shape)


def admit_prep(cell, args, run, out, ran_now):
    """Slice 2: the flow inputs of the nights the installed cell admits (module docstring). Refuses in a production tree
    before cutover, when the cell's population record (build_stats.json: the numbers the admitted night is judged by) is
    missing, fails its content test or holds a night excluded now (population_problem), and when the installed reference
    does not resolve; no input when no night qualifies (each night's reason recorded). With --epochs only cells of the
    named LSTs are selected (deep_stages.select_cells). Per night: a post-cutover survey image off the installed master's
    grid (or unreadable) is refused as an item (B-03 a: off_grid); its earlier prep files are moved aside (move_aside) and
    it is prepared again - a pre-cutover night regridded from its own WCS (prep_night), one whose shape still differs
    refused as an item. A re-run therefore makes every taken night's flow input again, and the flow step re-solves the
    flows older than their inputs - the price of a re-run, kept for correctness (fifth review: a reuse rule could not be
    made to see every way a prepared file goes stale). Outputs: each night's scaled image and the reference copies
    (admit/ref/, content-tested). Each run's record holds the nights of that run: --epochs prepares, and hands to the flow
    step, those nights only. --resume identity: the nights taken (the unit's admitted set), every input (prep_inputs),
    and the population gate passing now (deep_stages._resume_ok)."""
    from exopipe.phase3.core import record as R
    from exopipe.phase3.lib.masters import admit as A, deep_inplace as IP, deep_stages as DS
    lst, sc = cell.split('_', 1)
    cd = cell_tree(args, cell)
    W = os.path.join(cd, A.ADMIT_DIR)
    with run.unit(f'admit-prep_{cell}', tier='deep', stage='admit-prep', cell=cell) as u:
        DS.not_here(cd, 'the cell tree a night is admitted into', 'admit-prep')
        IP.refuse_production_write(W, 'admit-prep')
        mods, _ = DS.inplace(out)
        co = mods['coadd']
        why_pop = population_problem(cd, cell, co)
        u.check('admit_prep.population_record', why_pop is None, detail=why_pop or 'build_stats.json passes its content test')
        if why_pop:
            u.refuse(why_pop)
        take, why, inputs = prep_inputs(cell, args, co)
        u.set_admitted([d for d, _sd, _img in take])
        u.meta['not_admitted'] = why
        for p in inputs:
            u.add_input(p)
        if not take:
            raise NoInput(f'{cell}: no night to admit ({len(why)} night(s) do not qualify'
                          + (f', e.g. {sorted(why.items())[0]}' if why else '') + ')')
        ents = A.installed_reference(cell, cd)
        refs = A.write_reference(cd, sc, ents)
        for e in ents:
            if refs.get(e['stokes']):
                u.add_output(refs[e['stokes']], 'admit_reference', master=e['path'], factor=e['factor'])
        ref = reference_dict(co, refs['I'], ents[0])
        os.makedirs(os.path.join(W, 'prep'), exist_ok=True)
        os.makedirs(os.path.join(W, 'flow'), exist_ok=True)
        stamp = f'{R.utcnow().replace(":", "")}_{os.getpid()}'
        pairs, rows = [], []
        with IP.capture_stdout(DS._log(out, cell, 'admit-prep'), echo=False):
            for d, sd, img in take:
                sp = u.plan(os.path.join(W, 'prep', f'{d}_scaled.fits'), 'flow_prep_scaled')
                bad = off_grid(d, img, ref['target_shape'], co.CUTOVER_DATE)
                if bad:
                    u.refuse_item(sp, (f'{d}: survey image header unreadable - its grid cannot be checked (B-03 a)'
                                       if bad == 'unreadable' else
                                       f'{d}: survey image {bad[0]}x{bad[1]} is off the installed master\'s grid '
                                       f'{ref["target_shape"][0]}x{ref["target_shape"][1]} - a Phase-2 wrong-grid night, kept '
                                       f'out until data_decisions.csv excludes it (B-03 a; DECISIONS_PENDING item 3)'),
                                  gate='admit-prep.wrong_grid')
                    continue
                moved = move_aside(W, d, stamp)
                if moved:
                    u.note(f'{d}: earlier prep files moved aside to {A.ADMIT_DIR}/prep/_superseded/{stamp}/: {moved}')
                one = co.prep_night(d, sd, img, ref, W, [True])
                if one is None:
                    u.refuse_item(sp, f'{d}: shape differs from the master\'s grid after the regrid', gate='admit-prep.grid')
                    continue
                entry, row = one
                pairs.append(entry)
                rows.append(row)
                inp = os.path.join(W, 'prep', f'{d}_regrid.fits') if row['needs_regrid'] else img
                u.add_output(sp, 'flow_prep_scaled', input_image=inp)
        u.meta['admit'] = dict(reference={e['stokes']: dict(path=refs.get(e['stokes']), source=e['path'], factor=e['factor'],
                                                            constdig=e['constdig'], mdwarp=e['mdwarp']) for e in ents},
                               pairs=pairs, dates=[dict(r, needs_regrid=bool(r['needs_regrid'])) for r in rows])
        u.note(f'{cell}: {len(rows)} night(s) prepared for admission ({", ".join(r["date"] for r in rows)}) against the '
               f'installed master / {ents[0]["factor"]:.6f} (MDWARP {ents[0]["mdwarp"]}); {len(why)} night(s) not admitted')


def plan_admit_prep(cell, args, out, planned):
    """Dry-run plan of admit-prep: the nights, the reference, the outputs; gates as the stage applies them."""
    from astropy.io import fits
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.masters import admit as A, deep_inplace as IP, deep_stages as DS
    cd = cell_tree(args, cell)
    lines, gates = [], []
    r = IP.is_production_path(os.path.join(cd, A.ADMIT_DIR))
    if r and not _cfg().paths.PRODUCTION_WRITES:
        gates.append(f'{cd} is under the production root {r} (run on a copy of the cell: --deep-installed-root)')
    co = DS.inplace(out)[0]['coadd']
    why_pop = population_problem(cd, cell, co)
    if why_pop:
        gates.append(why_pop)
    try:
        take, why = nights(cell, args, co)
        ents = A.installed_reference(cell, cd, read_data=False)
    except Refusal as e:
        return lines, gates + [str(e)]
    h = fits.getheader(ents[0]['path'])
    shape = (int(h['NAXIS2']), int(h['NAXIS1']))
    lines.append(f'nights to admit: {[x[0] for x in take] or "none (no input)"}; not admitted: {len(why)}')
    lines.append(f'reference: {ents[0]["path"]} / {ents[0]["factor"]:.6f} -> {cd}/{A.ADMIT_DIR}/ref/ (MDWARP {ents[0]["mdwarp"]})')
    for d, _sd, img in take:
        bad = off_grid(d, img, shape, co.CUTOVER_DATE)
        lines.append(f'  {d}: {os.path.basename(img)} -> {A.ADMIT_DIR}/prep/{d}_scaled.fits [flow_prep_scaled]'
                     + ((' - REFUSED as an item: survey image header unreadable (B-03 a)' if bad == 'unreadable' else
                         f' - REFUSED as an item: survey image {bad[0]}x{bad[1]} off the master\'s grid (B-03 a)') if bad else ''))
    return lines, gates


# =====================================================================================================================
# slice 4 part 3: the admit stage
# =====================================================================================================================

ADMISSION_COLS = ('date', 'verdict', 'reason', 'image', 'image_sha1', 'flow', 'flow_sha1', 'reference_constdig',
                  'pixel_factor', 'dc1', 'dc2', 'dc2_nsrc', 'guard_median', 'guard_median_source', 'night_image',
                  'night_image_reason', 'epoch_gate', 'epoch_ratio', 'epoch_reason', 'qa3_scale', 'qa3_window',
                  'stage2_i_rms_mjy', 'stage2_i', 'stage2_v_rms_mjy', 'stage2_v', 'qa3_rms_mjy', 'beam_area', 'beam_gate',
                  'build_stats_sha1', 'commit', 'utc')
PRODUCTS = tuple((st, kind) for st in ('I', 'V') for kind in ('epoch', 'sub', 'clean'))
PROV = (('I', 'I_deep_Taper_R075'), ('V', 'V_deep_Taper_R0'))      # coadd_provisional/: the build's provisional masters


def guard_median(bst):
    """(median, source) of the cell's second-pass DC for the degeneracy guard (coadd.subtract_night): the build's own
    record (build_stats.json 'scale2', kept by builds from now on), else masters_admit.DC2_MEDIAN_WITHOUT_RECORD
    (DECISIONS_PENDING item 10) - (None, None) when neither: the stage refuses, it never substitutes on its own."""
    rec = ((bst or {}).get('scale2') or {}).get('median')
    if rec is not None:
        return float(rec), 'recorded'
    v = _cfg().masters_admit.DC2_MEDIAN_WITHOUT_RECORD
    return (None, None) if v is None else (float(v), 'ruled: masters_admit.DC2_MEDIAN_WITHOUT_RECORD')


def stage2_verdict(value, pop, s_hi, s_lo):
    """An admitted night's stage-2 verdict (mJy) against the recorded population (n, gate, median, MAD) with the sigmas in
    force - recorded, never refusing (an admitted night never joins the master)."""
    import math
    if value is None or not math.isfinite(value):
        return 'not measurable'
    if not pop.get('gate'):
        return f'no gate (population of {pop.get("n", 0)} below the minimum)'
    sig = 1.4826 * pop['mad']
    if value > pop['median'] + s_hi * sig:
        return 'outlier-high'
    if value < pop['median'] - s_lo * sig:
        return 'outlier-low'
    return 'pass'


def _sha1(p):
    import hashlib
    h = hashlib.sha1()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def admit_inputs(cell, args, co, out):
    """(todo, inputs, admit-prep record, flow record) of the admit stage, as the stage records them - also its --resume
    identity: the nights admit-prep prepared that still qualify now, with every input (prep_inputs' records, masters and
    images; the admit-prep and flow records; each night's scaled input and flow; the reference copies; the provisional
    masters). Refuses without a done admit-prep record or a done flow admit record made from it."""
    from exopipe.phase3.core import record as R
    from exopipe.phase3.lib.masters import admit as A, deep_stages as DS
    cd = cell_tree(args, cell)
    sc = cell.split('_', 1)[1]
    W = os.path.join(cd, A.ADMIT_DIR)
    prep = DS.done_record(out, 'admit-prep', cell)
    if not prep:
        raise Refusal(f'no done masters admit-prep record of {cell} under {out} - run `masters --tier deep --stages '
                      f'admit-prep` first')
    frec = R.load_record(out, 'flow', f'admit_{cell}')
    fmeta = ((frec or {}).get('meta') or {}).get('admit') or {}
    if not frec or frec.get('status') != 'done' or fmeta.get('admit_prep_record') != prep.get('run_id'):
        raise Refusal(f'no done flow admit record of {cell} made from admit-prep run {prep.get("run_id")} (found '
                      f'{(frec or {}).get("status", "none")}, from {fmeta.get("admit_prep_record")}) - run `flow --tier deep '
                      f'--stages admit` after admit-prep')
    take, _why, inputs = prep_inputs(cell, args, co)
    prepared = {r['date'] for r in ((prep.get('meta') or {}).get('admit') or {}).get('dates', [])}
    todo = [x for x in take if x[0] in prepared]
    inputs = list(inputs) + [R.record_path(out, 'masters', f'admit-prep_{cell}'), R.record_path(out, 'flow', f'admit_{cell}')]
    inputs += [os.path.join(W, 'ref', f'{v}_{sc}.fits') for _st, v in PROV]
    inputs += [p for p in (os.path.join(cd, 'coadd_provisional', f'{v}_{sc}.fits') for _st, v in PROV) if os.path.exists(p)]
    for d, _sd, _img in todo:
        inputs += [os.path.join(W, 'prep', f'{d}_scaled.fits'), os.path.join(W, 'flow', f'{d}_scaled_flow.npy')]
    return todo, list(dict.fromkeys(inputs)), prep, frec


def move_aside_files(folder, names, stamp):
    """Earlier files of a night in ``folder`` moved to <folder>/_superseded/<stamp>/ before they are made again (the co-add
    reuses an existing regrid; every other file is made again) - never deleted. Returns the names moved."""
    moved = []
    for f in names:
        if os.path.exists(os.path.join(folder, f)):
            dest = os.path.join(folder, '_superseded', stamp)
            os.makedirs(dest, exist_ok=True)
            os.replace(os.path.join(folder, f), os.path.join(dest, f))
            moved.append(f)
    return moved


def write_admissions(cd, rows):
    """Append ``rows`` to <cell>/coadd_final/admissions.csv (ADMISSION_COLS; written to a temporary name and renamed)."""
    import csv
    from exopipe.phase3.lib.masters import admit as A
    p = os.path.join(cd, 'coadd_final', A.ADMISSIONS)
    old = list(csv.DictReader(open(p, newline=''))) if os.path.exists(p) else []
    tmp = f'{p}.part{os.getpid()}'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ADMISSION_COLS), extrasaction='raise')
        w.writeheader()
        w.writerows(old + rows)
    os.replace(tmp, p)
    return p


def admit_stage(cell, args, run, out, ran_now):
    """Slice 4 (plan/wave2/DADMIT_DESIGN.md "Slice 4 part 3"): the nights admit-prep prepared and the flow step aligned,
    admitted into the INSTALLED master's frame and scale with the build's own per-night functions - coadd.final_night
    against the installed master in native units (the admission reference), the V night as the build's V loop aligns it,
    coadd.measure_scale2 against the same master, coadd.subtract_night with the cell's guard median, each product baked
    with shflux.bake (today's row = the installed factor, checked) and written to <cell>/subtracted/ with ADMITTED cards.
    Refuses before any night: a production tree before cutover; the population gate (population_problem); no done
    admit-prep / flow admit record (admit_inputs); a master-dewarped cell (DECISIONS_PENDING item 7); no guard median
    (guard_median, item 10); a scale row x correction that is not the installed factor. Per night: products already in
    subtracted/ or a stale flow refuse the item (retried later); a night refused at alignment, by the night-image check or
    by the epoch check gets a 'refused' row and no product (a rebuild decides); stage 2 (vs the provisional masters), the
    stage-3 window and the beam gate are recorded, never refusing. One row per night in coadd_final/admissions.csv.
    --resume identity: the nights (todo), every input (admit_inputs) and the population gate, applied now."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from exopipe.phase3.core import record as R
    from exopipe.phase3.lib.flow import steps_deep as FSD
    from exopipe.phase3.lib.masters import (admit as A, build_stats as BS, deep_inplace as IP, deep_stages as DS,
                                            flux_correction as FC, shflux as SH)
    lst, sc = cell.split('_', 1)
    fl, freq_str = sc, DS._freq(sc)
    cd = cell_tree(args, cell)
    W = os.path.join(cd, A.ADMIT_DIR)
    with run.unit(f'admit_{cell}', tier='deep', stage='admit', cell=cell) as u:
        DS.not_here(cd, 'the cell tree a night is admitted into', 'admit')
        for d_ in (W, os.path.join(cd, 'subtracted'), os.path.join(cd, 'coadd_final')):
            IP.refuse_production_write(d_, 'admit')
        mods, _ = DS.inplace(out)
        co = mods['coadd']
        why_pop = population_problem(cd, cell, co)
        u.check('admit.population_record', why_pop is None, detail=why_pop or 'build_stats.json passes its content test')
        if why_pop:
            u.refuse(why_pop)
        todo, inputs, prep, frec = admit_inputs(cell, args, co, out)
        u.set_admitted([x[0] for x in todo])
        for p in inputs:
            u.add_input(p)
        ents = A.installed_reference(cell, cd)
        if any(e['mdwarp'] for e in ents):
            u.refuse(f'{cell} carries the master dewarp (MDWARP): its admitted night\'s flow must be made against the '
                     f'dewarped master and its stage-2 number needs the provisional masters resampled with the archived '
                     f'screen - not written until the GPU test (DECISIONS_PENDING item 7)')
        refs = A.write_reference(cd, sc, ents)
        sig = FSD.admit_signature(refs['I'])
        if ((frec.get('meta') or {}).get('admit') or {}).get('reference_signature') != sig:
            u.refuse(f'the admit flows were made against another reference than {sig} - run admit-prep and the flow again')
        bst = BS.read(cd)
        med, med_src = guard_median(bst)
        if med is None:
            u.refuse(f'{cell}: its build recorded no median second-pass DC and masters_admit.DC2_MEDIAN_WITHOUT_RECORD is '
                     f'unset - the degeneracy guard needs one (DECISIONS_PENDING item 10; nothing is substituted)')
        row = DS.scale_row(lst, sc)
        corr_t = FC.load_table(_cfg().masters_fluxscale.FLUXCORR_TABLE)
        factor = float(row['scale']) * float(corr_t[sc]['corr'])
        for e in ents:
            if abs(factor - e['factor']) > 1e-9 * abs(e['factor']):
                u.refuse(f'today\'s scale row x correction {factor:.9f} is not the installed {e["var"]} master\'s pixel '
                         f'factor {e["factor"]:.9f} - an admitted night would not be on its scale')
        if not todo:
            raise NoInput(f'{cell}: no night admit-prep prepared still qualifies')
        ref = reference_dict(co, refs['I'], ents[0])
        ref_hdr = fits.getheader(refs['I'])
        cdelt = abs(ref_hdr.get('CDELT1', ref_hdr.get('CD1_1', 0.037)))
        fref = dict(data=ref['data'], wcs=ref['wcs'], beam_pix=ref['beam_pix'], horizon_mask=ref['horizon_mask'], cd=cdelt,
                    tmpl_hdr=ref['tmpl_hdr'], target_shape=ref['target_shape'])
        fin_v = co.load_image(refs['V'])[0] if refs.get('V') else None
        final_wcs = WCS(ref['tmpl_hdr']).celestial
        coadd_mask = ~np.isfinite(ref['data']) | (ref['data'] == 0)
        m_inst = np.squeeze(fits.getdata(ents[0]['path'])).astype(np.float64)        # the installed (baked) I master
        prov_d = {}
        for st, v in PROV:
            pp = os.path.join(cd, 'coadd_provisional', f'{v}_{sc}.fits')
            prov_d[st] = co.load_image(pp)[0] if os.path.exists(pp) else None
        S = BS.settings_in_force(co, _cfg())
        prep_final, native = os.path.join(W, 'prep_final'), os.path.join(W, 'sub_native')
        for d_ in (prep_final, native, os.path.join(cd, 'subtracted')):      # a copied tree may lack subtracted/
            os.makedirs(d_, exist_ok=True)
        stamp = f'{R.utcnow().replace(":", "")}_{os.getpid()}'
        commit = R.git_commit(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        bs_sha1 = _sha1(os.path.join(cd, 'coadd_final', BS.FILE))
        rows = []
        with IP.capture_stdout(DS._log(out, cell, 'admit'), echo=False):
            for d, sd, img in todo:
                rowd = dict.fromkeys(ADMISSION_COLS, '')
                rowd.update(date=d, image=img, image_sha1=_sha1(img), reference_constdig=ents[0]['constdig'] or '',
                            pixel_factor=f'{factor:.9f}', guard_median=f'{med:.6f}', guard_median_source=med_src,
                            build_stats_sha1=bs_sha1, commit=commit, utc=R.utcnow())
                dests = {k: os.path.join(cd, 'subtracted', f'{d}_{k[0]}_{k[1]}_{fl}.fits') for k in PRODUCTS}
                planned = {k: u.plan(p, 'admitted_epoch') for k, p in dests.items()}
                fp = os.path.join(W, 'flow', f'{d}_scaled_flow.npy')
                sp = os.path.join(W, 'prep', f'{d}_scaled.fits')
                if any(os.path.exists(p) for p in dests.values()):
                    u.refuse_item(planned[('I', 'sub')], f'{d}: products already in subtracted/ - a night is never admitted '
                                                         f'twice', gate='admit.exists')
                    continue
                if not os.path.exists(fp) or os.path.getmtime(fp) < os.path.getmtime(sp):
                    u.refuse_item(planned[('I', 'sub')], f'{d}: its admit flow is missing or older than its scaled input - '
                                                         f'run `flow --tier deep --stages admit`', gate='admit.flow')
                    continue
                rowd.update(flow=fp, flow_sha1=_sha1(fp))
                move_aside_files(prep_final, [f'{d}_regrid.fits', f'{d}_scale_remeasure.fits', f'{d}_I.fits', f'{d}_V.fits'],
                                 stamp)
                move_aside_files(native, [f'{d}_{st}_{kind}_{fl}.fits' for st, kind in PRODUCTS], stamp)
                nr = co.final_night(d, sd, img, fref, W, prep_final, [True], lst, fl, 'MASTER')
                if nr is None or nr['params'] is None:
                    reason = ('its survey image is off the master\'s grid' if nr is None else
                              (nr['gate'] or {}).get('reason') or 'blank field')
                    rowd.update(verdict='refused', reason=f'alignment: {reason}',
                                qa3_window='' if nr is None else ('inside' if nr['qa3_in'] else 'OUTSIDE'))
                    rows.append(rowd)
                    continue
                pa = nr['params']
                v_img = co.find_stokes_v_deep_image(sd, freq_str)
                v_out = os.path.join(prep_final, f'{d}_V.fits')
                if v_img:
                    co.process_variant_image(v_img, pa['flow'], pa['scale_map'], fref['tmpl_hdr'], fref['target_shape'],
                                             v_out, co._product_mapping(v_img, img, pa['regrid']))
                coeffs2, n2 = co.measure_scale2(nr['i_out'], ref['data'], final_wcs, ref['beam_pix'], coadd_mask, ref_hdr)
                co.subtract_night(d, sd, img, nr['i_out'], v_out, coeffs2, n2, med, ref['data'], fin_v, final_wcs,
                                  coadd_mask, ref['beam_pix'], fref['tmpl_hdr'], fref['target_shape'], native, fl, freq_str)
                baked = {}
                for k, dest in dests.items():
                    src = os.path.join(native, f'{d}_{k[0]}_{k[1]}_{fl}.fits')
                    if not os.path.exists(src):
                        continue
                    with fits.open(src) as h:
                        dd, hd = SH.bake(h[0].data, h[0].header.copy(), sc, row, corr_t)
                    hd['ADMITTED'] = (True, 'admitted into the installed master after its build (D-ADMIT)')
                    hd['ADMITREF'] = (str(ents[0]['constdig'] or '')[:32], 'CONSTDIG of the installed master')
                    baked[k] = (dest, dd, hd)
                ng = nr['gate']
                e_ok, e_why, e_med = False, 'no I epoch / residual product', None
                if ('I', 'epoch') in baked and ('I', 'sub') in baked:
                    e_ok, e_why, e_med, _n = co._epoch_content_arrays(
                        np.squeeze(baked[('I', 'epoch')][1]).astype(np.float64), m_inst,
                        np.squeeze(baked[('I', 'sub')][1]).astype(np.float64), 'I')
                rowd.update(dc1=pa.get('flux_dc', ''), dc2=f'{float(coeffs2[0]):.6f}', dc2_nsrc=n2,
                            night_image=ng['verdict'], night_image_reason=ng.get('reason', ''),
                            epoch_gate='pass' if e_ok else 'FAIL', epoch_ratio='' if e_med is None else f'{e_med:.4f}',
                            epoch_reason=e_why, qa3_scale=f'{pa["qa3_scale"]:.5f}', qa3_window=pa['qa3_window'],
                            qa3_rms_mjy=f'{1e3 * pa["qa3_rms"]:.2f}')
                for st, margin, col in (('I', 0.15, 'stage2_i'), ('V', 0.25, 'stage2_v')):
                    ap = nr['i_out'] if st == 'I' else v_out
                    val = None
                    if prov_d[st] is not None and os.path.exists(ap):
                        x = co.compute_residual_rms(co.load_image(ap)[0], prov_d[st], ref['horizon_mask'], margin_frac=margin)
                        val = float(x) * 1000 if np.isfinite(x) else None
                    rowd[f'{col}_rms_mjy'] = '' if val is None else f'{val:.4f}'
                    rowd[col] = ('not measurable (no provisional master)' if prov_d[st] is None else
                                 stage2_verdict(val, bst['stage2'][st], S['OUTLIER_SIGMA'], S['LOW_OUTLIER_SIGMA']))
                bimg = co.beam_gate_image(lst, fl, d, sci=sd)
                area = co.beam_area(bimg) if bimg else None
                bg = bst['beam_gate']
                rowd['beam_area'] = '' if area is None else f'{area:.6g}'
                rowd['beam_gate'] = ('not measurable' if area is None else
                                     f'no gate (population of {bg.get("n", 0)})' if not bg.get('gate') else
                                     'outlier' if area > S['BEAM_GATE_RATIO'] * bg['median_area'] else 'pass')
                if ng['verdict'] == 'FAIL' or not e_ok:
                    rowd.update(verdict='refused', reason=('night-image check FAIL: ' + ng.get('reason', '')
                                                           if ng['verdict'] == 'FAIL' else 'epoch-image check FAIL: ' + e_why))
                    rows.append(rowd)
                    continue                                   # no product published (the epoch gate's rule)
                for k, (dest, dd, hd) in baked.items():
                    tmp = f'{dest}.part{os.getpid()}'
                    fits.writeto(tmp, dd, hd)
                    os.replace(tmp, dest)
                    u.add_output(dest, 'admitted_epoch')
                rowd.update(verdict='admitted')
                rows.append(rowd)
        if rows:
            p = write_admissions(cd, rows)
            u.add_output(p, 'admissions')
        u.meta['admit'] = dict(admitted=[r['date'] for r in rows if r['verdict'] == 'admitted'],
                               refused={r['date']: r['reason'] for r in rows if r['verdict'] == 'refused'},
                               guard_median=med, guard_median_source=med_src, factor=factor,
                               admit_prep_record=prep.get('run_id'), flow_record=frec.get('run_id'))
        u.note(f'{cell}: {sum(r["verdict"] == "admitted" for r in rows)} night(s) admitted, '
               f'{sum(r["verdict"] == "refused" for r in rows)} refused (admissions.csv); guard median {med} ({med_src})')


def plan_admit(cell, args, out, planned):
    """Dry-run plan of the admit stage: the gates as the stage applies them, the nights and their products."""
    from exopipe.phase3.core.record import Refusal as _Ref
    from exopipe.phase3.lib.masters import admit as A, build_stats as BS, deep_inplace as IP, deep_stages as DS
    cd = cell_tree(args, cell)
    lines, gates = [], []
    r = IP.is_production_path(os.path.join(cd, A.ADMIT_DIR))
    if r and not _cfg().paths.PRODUCTION_WRITES:
        gates.append(f'{cd} is under the production root {r} (run on a copy of the cell: --deep-installed-root)')
    co = DS.inplace(out)[0]['coadd']
    why_pop = population_problem(cd, cell, co)
    if why_pop:
        return lines, gates + [why_pop]
    try:
        todo, _inputs, _prep, _frec = admit_inputs(cell, args, co, out)
        ents = A.installed_reference(cell, cd, read_data=False)
    except _Ref as e:
        return lines, gates + [str(e)]
    if any(e['mdwarp'] for e in ents):
        gates.append(f'{cell} carries the master dewarp (MDWARP) - not admitted until the GPU test (item 7)')
    med, med_src = guard_median(BS.read(cd))
    if med is None:
        gates.append(f'{cell}: no recorded median second-pass DC and masters_admit.DC2_MEDIAN_WITHOUT_RECORD unset (item 10)')
    lines.append(f'nights to admit: {[x[0] for x in todo] or "none (no input)"}; guard median {med} ({med_src})')
    sc = cell.split('_', 1)[1]
    for d, _sd, _img in todo:
        lines.append(f'  {d}: subtracted/{d}_{{I,V}}_{{epoch,sub,clean}}_{sc}.fits [admitted_epoch]; a row in '
                     f'coadd_final/{A.ADMISSIONS}')
    return lines, gates
