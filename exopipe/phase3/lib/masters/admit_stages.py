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

from exopipe.phase3.core.record import NoInput


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
