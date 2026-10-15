"""D-ADMIT (a), slices 2-4 (lib/masters/admit.py; plan/wave2/DADMIT_DESIGN.md): which nights an installed cell admits and
the reference they are admitted against.

Nights: a night is admitted only when no exclusion names it, admissions.csv does not hold it, the build's record does not
hold it, and its survey image was written after the build fixed its night set - each condition alone fails a check on a
synthetic cell; on real installed cells (read only) the nights in no build record were all seen by their builds, so none
is admitted. Reference: the installed master divided by its pixel factor, header stripped as stage-in strips it, in a
dewarped cell too; a cell whose V master holds a night fewer is not refused; the copies under admit/ref/ are never
silently replaced.
Run: python tests/test_masters_admit.py
"""
import csv
import os
import shutil
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.core import env as ENV                          # noqa: E402
ENV.legacy()
from exopipe.phase3.config import CFG                               # noqa: E402
from exopipe.phase3.core.record import Refusal                      # noqa: E402
from exopipe.phase3.lib.masters import admit as A                   # noqa: E402

T = '/tmp/exopipe3_tests_admit'
FAILS, N = [], [0]


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


class _Ex:
    def __init__(self, nights):
        self.nights = set(nights)

    def is_excluded(self, lst, scope, d):
        return d in self.nights


def test_nights_rule():
    cell = os.path.join(T, '23h_23MHz')
    os.makedirs(os.path.join(cell, 'coadd_final'))
    rec = os.path.join(cell, A.BUILD_NIGHTS)
    with open(rec, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lst', 'freq', 'date', 'i_rms_mjy'])
        w.writerows([['23h', '23MHz', '2025-01-01', '1'], ['23h', '23MHz', '2025-01-02', '1']])
    with open(os.path.join(cell, 'coadd_final', A.ADMISSIONS), 'w', newline='') as f:
        csv.writer(f).writerows([['date', 'verdict'], ['2025-01-04', 'admitted']])
    t = time.time() - 3600
    os.utime(rec, (t, t))
    entries = []
    for d, dt in (('2025-01-01', -100), ('2025-01-02', 100), ('2025-01-03', -100), ('2025-01-04', 100), ('2025-01-05', 100),
                  ('2025-01-06', 100)):
        img = os.path.join(T, f'{d}.fits')
        open(img, 'w').write('x')
        os.utime(img, (t + dt, t + dt))
        entries.append((d, os.path.join(T, 'Science_x'), img))
    entries.append(('2025-01-07', os.path.join(T, 'Science_x'), None))
    take, why = A.nights_to_admit('23h', '23MHz', cell, entries, _Ex({'2025-01-06'}))
    check('nights: only the night in no record, not excluded, not admitted, with an image newer than the build is taken',
          [x[0] for x in take] == ['2025-01-05'], (take, why))
    want = {'2025-01-01': 'installed build', '2025-01-02': 'installed build', '2025-01-03': 'older than the installed build',
            '2025-01-04': A.ADMISSIONS, '2025-01-06': 'excluded', '2025-01-07': 'no survey image'}
    check('nights: each other night says why (a recorded night stays out even with a newer image)',
          set(why) == set(want) and all(want[d] in why[d] for d in want), why)
    take, why = A.nights_to_admit('23h', '23MHz', cell, entries, _Ex(()), epochs=['2025-01-05', '2025-01-03', '2025-02-01'])
    check('nights: --epochs restricts to the named nights; a named night that does not qualify or does not exist says why',
          [x[0] for x in take] == ['2025-01-05'] and set(why) == {'2025-01-03', '2025-02-01'}
          and 'no Phase-2 night' in why['2025-02-01'], (take, why))
    os.remove(rec)
    try:
        A.nights_to_admit('23h', '23MHz', cell, entries, _Ex(()))
        check('nights: refused without the build\'s night record', False)
    except Refusal as e:
        check('nights: refused without the build\'s night record', A.BUILD_NIGHTS in str(e), str(e))


def test_nights_real():
    """Read only: cells whose builds left nights out of their records (wrong grid, refused before measurement)."""
    import contextlib
    import io
    from exopipe.phase3.core import exclusions as EX
    from exopipe.phase3.lib.masters import coadd as co
    ex = EX.load()
    root = CFG.paths.COADD_SB_ROOT
    for cell, night in (('02h_Green', '2025-12-26'), ('11h_Full', '2025-03-09'), ('23h_23MHz', '2025-10-29')):
        lst, sc = cell.split('_')
        with contextlib.redirect_stdout(io.StringIO()):
            entries = co.discover_dates(lst, sc.replace('MHz', '') if sc.endswith('MHz') else sc)
        take, why = A.nights_to_admit(lst, sc, os.path.join(root, cell), entries, ex)
        check(f'real {cell}: no night to admit; {night} (in no build record) was seen by the build',
              not take and 'older than the installed build' in why.get(night, ''), (take, why.get(night)))


def test_reference():
    from astropy.io import fits
    from exopipe.phase3.lib.masters import flux_correction as FC
    root = CFG.paths.COADD_SB_ROOT
    ents = A.installed_reference('13h_78MHz', os.path.join(root, '13h_78MHz'))
    ok = True
    for e in ents:
        with fits.open(e['path']) as h:
            raw, hh = h[0].data, h[0].header
            f = FC.pixel_factor(hh)
            sl = (Ellipsis, slice(1500, 1510), slice(1500, 1510))
            ok &= (abs(e['factor'] - f) < 1e-12 and np.array_equal(e['data'][sl], (raw[sl].astype(np.float64) / f).astype(np.float32),
                                                                   equal_nan=True))
        ok &= not any(k.startswith(tuple(CFG.masters_stagein.STRIP_CARD_PREFIXES)) for k in e['header'].keys())
    check('reference: the installed I and V masters / their pixel factor, SH*/FLUX* cards stripped (13h_78MHz)',
          ok and [e['stokes'] for e in ents] == ['I', 'V'] and not ents[0]['mdwarp'], [(e['stokes'], e['factor']) for e in ents])
    d = A.installed_reference('23h_23MHz', os.path.join(root, '23h_23MHz'), read_data=False)
    check('reference: a dewarped cell is admitted against its installed (dewarped) master', d[0]['mdwarp'] and d[1]['mdwarp'],
          [(e['stokes'], e['mdwarp']) for e in d])
    v = A.installed_reference('02h_55MHz', os.path.join(root, '02h_55MHz'), read_data=False)
    check('reference: a cell whose V master holds a night fewer (per-Stokes sets) is not refused',
          v[0]['constdig'] != v[1]['constdig'], [(e['stokes'], e['constdig']) for e in v])
    cell = os.path.join(T, 'ref_13h_78MHz')
    paths = A.write_reference(cell, '78MHz', ents)
    again = A.write_reference(cell, '78MHz', ents)
    h = fits.getheader(paths['I'])
    check('reference copies: admit/ref/<var>_<scope>.fits with the resolution in HISTORY; writing the same again is a no-op',
          paths == again and sorted(paths) == ['I', 'V'] and paths['I'].endswith('/admit/ref/I_deep_Taper_R075_78MHz.fits')
          and any('ADMIT reference: installed' in str(x) for x in h.get('HISTORY', [])), paths)
    ents[0]['data'] = ents[0]['data'] * np.float32(1.001)
    try:
        A.write_reference(cell, '78MHz', ents)
        check('reference copies: a copy with other content is never replaced', False)
    except Refusal as e:
        check('reference copies: a copy with other content is never replaced', 'other content' in str(e), str(e))


def _old_prep_body(co, date, sci_dir, align_img_path, workdir, tmpl_hdr, target_shape, ref_data, ref_wcs, beam_pix,
                   horizon_mask, ref_img_path, post_dates):
    """process_lst_freq's loop body as it was before prep_night (2026-10-15), pinned here: (manifest entry, row) or None."""
    import json
    needs_regrid = date < co.CUTOVER_DATE and post_dates
    if not needs_regrid and co._off_grid(date, align_img_path, target_shape):
        needs_regrid = True
    if needs_regrid:
        work_path = co.regrid_image(align_img_path, tmpl_hdr, f'{workdir}/prep/{date}_regrid.fits')
    else:
        work_path = align_img_path
    tgt_data, tgt_hdr = co.load_image(work_path)
    if tgt_data.shape != target_shape:
        return None
    tgt_mask = ~np.isfinite(tgt_data) | (tgt_data == 0)
    tgt_obs_time = co.get_midpoint_time(tgt_hdr, work_path)
    tgt_rms = co.get_uniform_rms(tgt_data, tgt_mask)
    scale_map, coeffs, n_src = co.measure_flux_scale(tgt_data, ref_data, ref_wcs, beam_pix, horizon_mask | tgt_mask, tgt_obs_time)
    mean_scale = float(coeffs[0])
    tgt_rms_scaled = tgt_rms / abs(mean_scale) if abs(mean_scale) > 0.1 else tgt_rms
    weight = float(mean_scale**2 / tgt_rms**2) if tgt_rms > 0 else 0.0
    safe_scale = np.where(np.abs(scale_map) > 0.1, scale_map, 1.0)
    scaled_path = f'{workdir}/prep/{date}_scaled.fits'
    co.save_image(scaled_path, tgt_data / safe_scale, tgt_hdr)
    with open(f'{workdir}/prep/{date}_scale.json', 'w') as f:
        json.dump({'coeffs': coeffs.tolist(), 'n_sources': n_src, 'mean_scale': mean_scale, 'tgt_rms': tgt_rms,
                   'tgt_rms_scaled': tgt_rms_scaled, 'weight': weight}, f)
    return ({'target': scaled_path, 'reference': ref_img_path, 'out_dir': f'{workdir}/flow'},
            {'date': date, 'sci_dir': sci_dir, 'align_img': align_img_path, 'needs_regrid': needs_regrid,
             'scale_map_coeffs': coeffs.tolist(), 'weight': weight, 'tgt_rms': tgt_rms, 'tgt_rms_scaled': tgt_rms_scaled})


def test_prep_night_equivalence():
    """coadd.prep_night (slice 2: the one per-night prep of a build and of an admission) writes exactly what the loop body
    of process_lst_freq wrote before it was extracted - same bytes, same manifest entry and row - on real nights of a real
    cell against a real reference (13h_78MHz: its installed master in native units)."""
    import contextlib
    import filecmp
    import io
    from astropy.wcs import WCS
    from exopipe.phase3.lib.masters import coadd as co
    root = CFG.paths.COADD_SB_ROOT
    ents = A.installed_reference('13h_78MHz', os.path.join(root, '13h_78MHz'))
    rp = A.write_reference(os.path.join(T, 'prepref'), '78MHz', ents)['I']
    tmpl_hdr, target_shape = co.build_template_header(rp)
    ref_data, ref_hdr = co.load_image(rp)
    cd = abs(ref_hdr.get('CDELT1', ref_hdr.get('CD1_1', 0.037)))
    ref = dict(img_path=rp, data=ref_data, wcs=WCS(ref_hdr).celestial, beam_pix=ref_hdr.get('BMAJ', 5 * cd) / cd,
               horizon_mask=~np.isfinite(ref_data) | (ref_data == 0), tmpl_hdr=tmpl_hdr, target_shape=target_shape)
    with contextlib.redirect_stdout(io.StringIO()):
        nights = co.discover_dates('13h', '78')
    picked = [x for x in nights if x[0] >= co.CUTOVER_DATE][:2]
    same, rows = True, []
    for date, sd, img in picked:
        wo, wn = os.path.join(T, 'prep_old'), os.path.join(T, 'prep_new')
        for w in (wo, wn):
            os.makedirs(os.path.join(w, 'prep'), exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            old = _old_prep_body(co, date, sd, img, wo, tmpl_hdr, target_shape, ref['data'], ref['wcs'], ref['beam_pix'],
                                 ref['horizon_mask'], rp, [True])
            new = co.prep_night(date, sd, img, ref, wn, [True])
        for f in (f'{date}_scaled.fits', f'{date}_scale.json'):
            same &= filecmp.cmp(os.path.join(wo, 'prep', f), os.path.join(wn, 'prep', f), shallow=False)
        same &= old[1] == new[1] and old[0]['target'].replace(wo, wn) == new[0]['target'] and old[0]['reference'] == new[0]['reference']
        rows.append(date)
    check(f'prep_night = the extracted loop body, byte for byte ({", ".join(rows)})', same and len(rows) == 2, rows)


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_nights_rule()
        test_nights_real()
        test_reference()
        test_prep_night_equivalence()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
