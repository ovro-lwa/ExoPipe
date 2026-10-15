"""D-ADMIT (a): admitting a night newer than its cell's installed master (ruling R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-
B03-B05-B30-B32: "admit stage into the INSTALLED master's frame and scale, gates against the build's recorded statistics,
admission record beside weights.csv, E15 accepts admitted nights, membership.verify reports 'admitted, rebuild pending'; the
master rebuild is NOT triggered automatically"). Design: plan/wave2/DADMIT_DESIGN.md (slices 2-4).

This module: which nights an installed cell admits (nights_to_admit) and the reference they are admitted against
(installed_reference: the INSTALLED master divided by its pixel factor - native flux units, in the installed frame, so in a
dewarped cell the night's flow aligns it to the dewarped master directly and no screen is applied afterwards). Everything
it reads is the cell tree it is given; nothing here writes outside <cell>/admit/.
"""
import csv
import os

from exopipe.phase3.core.record import Refusal

ADMIT_DIR = 'admit'                      # <cell>/admit/: reference copies, scaled flow inputs, flows, native products
ADMISSIONS = 'admissions.csv'            # <cell>/coadd_final/admissions.csv: one row per admitted (or refused) night
BUILD_NIGHTS = 'residuals_provisional.csv'   # every night the installed build measured (member or refused)


def _rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def build_record(cell_dir):
    """(path, mtime, {nights}) of the installed build's night record - the nights it measured; its mtime is when the build
    fixed its night set. Refuses without it (a cell with no build record has no population to admit into)."""
    p = os.path.join(cell_dir, BUILD_NIGHTS)
    if not os.path.exists(p):
        raise Refusal(f'{p} missing: the installed build\'s night record (no population to admit a night into)')
    return p, os.path.getmtime(p), {r['date'] for r in _rows(p)}


def admitted_nights(cell_dir):
    """Nights coadd_final/admissions.csv holds (admitted or refused there): never admitted twice."""
    p = os.path.join(cell_dir, 'coadd_final', ADMISSIONS)
    return {r['date'] for r in _rows(p)} if os.path.exists(p) else set()


def nights_to_admit(lst, scope, cell_dir, entries, ex, epochs=None):
    """The nights of an installed cell to admit, and why each other night is not admitted.

    entries: the cell's Phase-2 nights as the co-add discovers them, [(date, science dir, survey image)]; ex: the exclusions
    (core.exclusions). A night is admitted when data_decisions.csv does not exclude it, admissions.csv does not hold it,
    the build's record does not hold it, and its survey image was written after the build fixed its night set - the build
    refuses some nights before it measures them (wrong grid, unmeasurable flux scale, blank field, a carried epoch-gate
    FAIL), and those are in no record. epochs: when given, only those nights; a named night that does not qualify is
    reported with its reason. Returns ([(date, sci_dir, img)], {date: reason})."""
    _p, t_build, built = build_record(cell_dir)
    done = admitted_nights(cell_dir)
    want = set(epochs) if epochs else None
    take, why = [], {}
    for d, sd, img in entries:
        if want is not None and d not in want:
            continue
        if ex.is_excluded(lst, scope, d):
            why[d] = 'excluded in data_decisions.csv'
        elif d in done:
            why[d] = f'already in coadd_final/{ADMISSIONS}'
        elif d in built:
            why[d] = f'in the installed build\'s record ({BUILD_NIGHTS}): a member or a night it refused - a rebuild decides'
        elif not img or not os.path.exists(img):
            why[d] = 'no survey image'
        elif os.path.getmtime(img) <= t_build:
            why[d] = ('survey image older than the installed build: the build saw this night and refused it before '
                      'measuring it - a rebuild decides')
        else:
            take.append((d, sd, img))
    if want is not None:
        for d in sorted(want - {x[0] for x in entries}):
            why[d] = 'no Phase-2 night of this cell'
    return take, why


def installed_reference(cell, cell_dir, read_data=True):
    """The reference a night is admitted against, per Stokes (I survey master, V master): the INSTALLED master divided by
    its pixel factor (SHSCALE x FLUXCORR when baked) - the installed master's frame and, in native units, its scale. Not
    the pre-dewarp master: of the 166 dewarped cells' masters only the Stokes I input of the dewarp survives (node-local
    copies, no V), so admission aligns the night to the installed frame itself and bakes at the end. Refuses when a master
    is missing or its pixel factor is outside masters_stagein's range. Each Stokes keeps its own constituents (the V
    master may hold fewer nights: per-Stokes sets, R-LOO-PER-STOKES-RATIFIED - 6 installed cells, 2026-10-15). Returns one
    dict per Stokes: stokes, var, path, factor, constdig, mdwarp, data (float32), header (SH*/FLUX* cards stripped as
    stage-in strips them), history."""
    from astropy.io import fits
    import numpy as np
    from exopipe.phase3.lib.masters import deep_stagein as DS, flux_correction as FC
    sc = cell.split('_', 1)[1]
    out = []
    for st, var, _rob in DS.REF_VARS:
        p = os.path.join(cell_dir, 'coadd_final', f'{var}_{sc}.fits')
        if not os.path.exists(p):
            raise Refusal(f'{cell}: installed master {p} missing - nothing to admit a night into')
        h = fits.getheader(p)
        f = FC.pixel_factor(h)
        f = 1.0 if f is None else f
        if not DS._factor_ok(f):
            raise Refusal(f'{cell}: pixel factor {f} of {p} outside masters_stagein.REF_PIXEL_FACTOR_RANGE')
        e = dict(stokes=st, var=var, path=p, factor=f, constdig=h.get('CONSTDIG'), mdwarp=bool(h.get('MDWARP')),
                 data=None, header=None,
                 history=f'ADMIT reference: installed {var} {p} / pixel factor {f:.6f} (CONSTDIG {h.get("CONSTDIG")}; '
                         f'MDWARP {bool(h.get("MDWARP"))})')
        if read_data:
            with fits.open(p) as hd:
                e['data'] = (hd[0].data.astype(np.float64) / f).astype(np.float32)
                hh = hd[0].header.copy()
            DS._strip(hh)
            e['header'] = hh
        out.append(e)
    return out


def write_reference(cell_dir, scope, ents):
    """The reference copies under <cell>/admit/ref/<var>_<scope>.fits (the canonical names), each with its resolution in
    HISTORY. Returns {stokes: path}; refuses to overwrite an existing copy with other content."""
    from astropy.io import fits
    import numpy as np
    d = os.path.join(cell_dir, ADMIT_DIR, 'ref')
    os.makedirs(d, exist_ok=True)
    out = {}
    for e in ents:
        if e['data'] is None:
            continue
        p = os.path.join(d, f'{e["var"]}_{scope}.fits')
        hh = e['header'].copy()
        hh['HISTORY'] = e['history']
        if os.path.exists(p):
            old = fits.getdata(p)
            if old.shape != e['data'].shape or not np.array_equal(old, e['data'], equal_nan=True):
                raise Refusal(f'{p} exists with other content: the installed master changed since the admission reference '
                              f'was written - admit into the new master from a fresh <cell>/{ADMIT_DIR}/')
        else:
            fits.writeto(p, e['data'], hh)
        out[e['stokes']] = p
    return out
