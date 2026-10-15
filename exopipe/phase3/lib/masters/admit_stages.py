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
    """The nights --epochs names for this LST (None: no --epochs, every qualifying night)."""
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
    return A.nights_to_admit(lst, sc, cell_tree(args, cell), entries, exclusions.load(), nights=epochs_of(args, lst))


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
    """Slice 2: the flow inputs of the nights the installed cell admits (module docstring). Refuses without the cell's
    population record (build_stats.json: the numbers the admitted night is judged by), when the installed reference does not
    resolve, and in a production tree before cutover; no input when no night qualifies (each night's reason recorded).
    A night whose image is off the master's grid is regridded from its own WCS (prep_night); one whose shape still
    differs is refused as an item. The nights taken are the unit's admitted set (resume identity: --resume re-runs when
    the nights that qualify now differ - a new night, an exclusion, another --epochs)."""
    from exopipe.phase3.lib.masters import admit as A, build_stats as BS, deep_inplace as IP, deep_stages as DS
    sc = cell.split('_', 1)[1]
    cd = cell_tree(args, cell)
    W = os.path.join(cd, A.ADMIT_DIR)
    with run.unit(f'admit-prep_{cell}', tier='deep', stage='admit-prep', cell=cell) as u:
        DS.not_here(cd, 'the cell tree a night is admitted into', 'admit-prep')
        IP.refuse_production_write(W, 'admit-prep')
        if BS.read(cd) is None:
            u.refuse(f'{cd}/coadd_final/{BS.FILE} missing: the admitted night is judged against the build\'s population - '
                     f'run `masters --tier deep --stages build-stats` on the cell first')
        mods, _ = DS.inplace(out)
        co = mods['coadd']
        take, why = nights(cell, args, co)
        u.set_admitted([d for d, _sd, _img in take])
        u.meta['not_admitted'] = why
        rec_p, _t, _built = A.build_record(cd)
        u.add_input(rec_p)
        if not take:
            raise NoInput(f'{cell}: no night to admit ({len(why)} night(s) do not qualify'
                          + (f', e.g. {sorted(why.items())[0]}' if why else '') + ')')
        ents = A.installed_reference(cell, cd)
        refs = A.write_reference(cd, sc, ents)
        for e in ents:
            u.add_input(e['path'])
        ref = reference_dict(co, refs['I'], ents[0])
        os.makedirs(os.path.join(W, 'prep'), exist_ok=True)
        os.makedirs(os.path.join(W, 'flow'), exist_ok=True)
        pairs, rows = [], []
        with IP.capture_stdout(DS._log(out, cell, 'admit-prep'), echo=False):
            for d, sd, img in take:
                sp = u.plan(os.path.join(W, 'prep', f'{d}_scaled.fits'), 'flow_prep_scaled')
                one = co.prep_night(d, sd, img, ref, W, [True])
                u.add_input(img)
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
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.masters import admit as A, build_stats as BS, deep_inplace as IP, deep_stages as DS
    cd = cell_tree(args, cell)
    lines, gates = [], []
    r = IP.is_production_path(os.path.join(cd, A.ADMIT_DIR))
    if r and not _cfg().paths.PRODUCTION_WRITES:
        gates.append(f'{cd} is under the production root {r} (run on a copy of the cell: --deep-installed-root)')
    if BS.read(cd) is None:
        gates.append(f'no {BS.FILE} (run --stages build-stats on the cell first)')
    try:
        take, why = nights(cell, args, DS.inplace(out)[0]['coadd'])
        ents = A.installed_reference(cell, cd, read_data=False)
    except Refusal as e:
        return lines, gates + [str(e)]
    lines.append(f'nights to admit: {[x[0] for x in take] or "none (no input)"}; not admitted: {len(why)}')
    lines.append(f'reference: {ents[0]["path"]} / {ents[0]["factor"]:.6f} -> {cd}/{A.ADMIT_DIR}/ref/ (MDWARP {ents[0]["mdwarp"]})')
    for d, _sd, img in take:
        lines.append(f'  {d}: {os.path.basename(img)} -> {A.ADMIT_DIR}/prep/{d}_scaled.fits [flow_prep_scaled]')
    return lines, gates
