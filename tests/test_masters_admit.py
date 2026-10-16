"""D-ADMIT (a), slices 2-4 (lib/masters/admit.py; plan/wave2/DADMIT_DESIGN.md): which nights an installed cell admits and
the reference they are admitted against.

Nights: a night is admitted only when no exclusion names it, admissions.csv does not hold it, no record of the installed
build names it (admit.BUILD_RECORDS: measured, refused before measuring, refused as an input), and its survey image was
written after the build fixed its night set - each condition alone fails a check on a synthetic cell; on real installed
cells (read only) the nights missing from residuals_provisional.csv were seen by their builds, so none is admitted.
Reference: the installed master divided by its pixel factor, header stripped as stage-in strips it, in a dewarped cell
too; a cell whose V master holds a night fewer is not refused; a master without SHSCALE refuses; the copies under
admit/ref/ are never silently replaced. prep_night = the co-add's loop body (both branches). The admit-prep stage on a
copy of 13h_78MHz with one member night made new.
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


WRONG_GRID_ROWS_20261016 = 'written 2026-10-16 by the refactor chat'   # the eight wrong-grid exclusions (clv15 df3242b)


class _ExBefore:
    """data_decisions as it stood before the eight wrong-grid rows of 2026-10-16: every exclusion but the named night's."""
    def __init__(self, ex, lst, scope, night):
        self.ex, self.key = ex, (lst, scope, night)

    def is_excluded(self, lst, scope, d):
        return False if (lst, scope, d) == self.key else self.ex.is_excluded(lst, scope, d)


def data_decisions_before(path):
    """A copy of data_decisions.csv without the eight wrong-grid rows of 2026-10-16 (the state the B-03 checks were written
    for: those nights were refused by the build's own grid record, not excluded)."""
    rows = [l for l in open(CFG.paths.DATA_DECISIONS) if not ('wrong grid' in l and WRONG_GRID_ROWS_20261016 in l)]
    open(path, 'w').writelines(rows)
    return path


def _night(root, d, name, bands=('23MHz',), mtime=None):
    """A synthetic Phase-2 night dir <root>/<d>/<name>/<band>/I/deep/ holding a survey image the co-add's finder takes."""
    sd = os.path.join(root, d, name)
    img = None
    for b in bands:
        dd = os.path.join(sd, b, 'I', 'deep')
        os.makedirs(dd, exist_ok=True)
        img = os.path.join(dd, f'{b}-I-Taper-3561s-Robust-0.75-{d.replace("-", "")}_000000-image_dewarped.fits')
        open(img, 'w').write('x')
        if mtime is not None:
            os.utime(img, (mtime, mtime))
    return sd, img


def test_nights_rule():
    """The night rule on a synthetic cell and Phase-2 tree: a night is taken only when no record of the build names it, it
    is not excluded or already admitted, and Phase 2 first had it after the build fixed its night set - every Science dir
    holding the cell's survey image named after that time. File times play no part (a re-dewarped image in an old Science
    dir stays seen); a night re-imaged into a new Science dir after the build stays seen through its old one; an old
    Science dir without this band's image does not count."""
    import calendar
    from exopipe.phase3.lib.masters import coadd as co
    cell = os.path.join(T, '23h_23MHz')
    os.makedirs(os.path.join(cell, 'coadd_final'))
    rec = os.path.join(cell, A.BUILD_NIGHTS)
    with open(rec, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lst', 'freq', 'date', 'i_rms_mjy'])
        w.writerows([['23h', '23MHz', '2025-01-01', '1'], ['23h', '23MHz', '2025-01-02', '1']])
    from astropy.io import fits as _fits
    import numpy as _np
    _h = _fits.Header()
    _h['CONSTDIG'] = 'synthetic'
    _fits.writeto(os.path.join(cell, 'coadd_final', 'I_deep_Taper_R075_23MHz.fits'), _np.ones((4, 4), _np.float32), _h)
    ident = A.master_identity(cell)                              # an admission row names the master it was made into
    with open(os.path.join(cell, 'coadd_final', A.ADMISSIONS), 'w', newline='') as f:
        csv.writer(f).writerows([['date', 'verdict', 'reference_constdig', 'reference_md5'],
                                 ['2025-01-04', 'admitted', ident[0], ident[1]]])
    with open(os.path.join(cell, 'coadd_final', 'weights.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([['date', 'weight'], ['2025-01-01', '1'], ['2025-01-02', '1']])
    t = calendar.timegm((2026, 1, 1, 0, 0, 0))                    # the build fixed its night set here (its weights.csv)
    os.utime(os.path.join(cell, 'coadd_final', 'weights.csv'), (t, t))
    t2 = calendar.timegm((2026, 2, 15, 0, 0, 0))                  # residuals_provisional.csv rewritten later (315 cells)
    os.utime(rec, (t2, t2))
    p2 = os.path.join(T, 'phase2', '23h')
    old, new = 'Science_20251201_000000', 'Science_20260201_000000'
    entries = []
    for d in ('2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05', '2025-01-06'):
        sd, img = _night(p2, d, old if d == '2025-01-03' else new)
        entries.append((d, sd, img))
    entries.append(('2025-01-07', os.path.join(p2, '2025-01-07', new), None))
    _night(p2, '2025-01-08', old)                                   # seen by the build, then re-imaged into a new dir
    entries.append(('2025-01-08',) + _night(p2, '2025-01-08', new))
    _night(p2, '2025-01-09', old, bands=('27MHz',))                 # the old dir held another band only
    entries.append(('2025-01-09',) + _night(p2, '2025-01-09', new))
    entries.append(('2025-01-10',) + _night(p2, '2025-01-10', old, mtime=t + 86400 * 30))   # re-dewarped in place, later
    entries.append(('2025-01-11',) + _night(p2, '2025-01-11', 'Science_20260115_000000'))   # after weights.csv, before
    entries.append(('2025-01-12',) + _night(p2, '2025-01-12', 'Science_oddname'))          # the rewrite; undatable name
    take, why = A.nights_to_admit('23h', '23MHz', cell, entries, _Ex({'2025-01-06'}), co)
    check('nights: only nights Phase 2 first had after the build (its weights.csv, not a later rewrite of '
          'residuals_provisional.csv), in no record, not excluded, not admitted, are taken',
          [x[0] for x in take] == ['2025-01-05', '2025-01-09', '2025-01-11'], (take, why))
    want = {'2025-01-01': 'installed build', '2025-01-02': 'installed build', '2025-01-03': 'Phase 2 had it before',
            '2025-01-04': A.ADMISSIONS, '2025-01-06': 'excluded', '2025-01-07': 'no survey image',
            '2025-01-08': 'Phase 2 had it before', '2025-01-10': 'Phase 2 had it before',
            '2025-01-12': 'Phase 2 had it before'}
    check('nights: each other night says why (re-imaged after the build: seen in its old dir; re-dewarped later: seen; a '
          'Science dir whose name carries no time counts as before)',
          set(why) == set(want) and all(want[d] in why[d] for d in want) and old in why['2025-01-08']
          and 'Science_oddname' in why['2025-01-12'], why)
    with open(os.path.join(cell, 'coadd_final', A.ADMISSIONS), 'w', newline='') as f:
        csv.writer(f).writerows([['date', 'verdict', 'reference_constdig', 'reference_md5'],
                                 ['2025-01-04', 'admitted', ident[0], '0' * 32]])
    take2, _why2 = A.nights_to_admit('23h', '23MHz', cell, entries, _Ex({'2025-01-06'}), co)
    check('nights: a row made into another master (its data md5 another) blocks nothing - never twice is per master '
          '(eighth review)', '2025-01-04' in [x[0] for x in take2], take2)
    with open(os.path.join(cell, 'coadd_final', A.ADMISSIONS), 'w', newline='') as f:
        csv.writer(f).writerows([['date', 'verdict', 'reference_constdig', 'reference_md5'],
                                 ['2025-01-04', 'admitted', ident[0], ident[1]]])
    take, why = A.nights_to_admit('23h', '23MHz', cell, entries, _Ex(()), co, nights=['2025-01-05', '2025-01-03', '2025-02-01'])
    check('nights: --epochs restricts to the named nights; a named night that does not qualify or does not exist says why',
          [x[0] for x in take] == ['2025-01-05'] and set(why) == {'2025-01-03', '2025-02-01'}
          and 'no Phase-2 night' in why['2025-02-01'], (take, why))
    take, why = A.nights_to_admit('23h', '23MHz', cell, entries, _Ex(()), co, nights=[])
    check('nights: --epochs naming no night of this LST takes none (never every night)', take == [] and why == {}, (take, why))
    with open(os.path.join(cell, 'coadd_final', 'night_image_gate.csv'), 'w', newline='') as f:
        f.write('\ndate,verdict\n2025-01-05,FAIL\n')                   # a leading blank line: the header is not read
    try:
        A.nights_to_admit('23h', '23MHz', cell, entries, _Ex(()), co)
        check('nights: a build record whose header lacks its date column refuses (never silently skipped)', False)
    except Refusal as e:
        check('nights: a build record whose header lacks its date column refuses (never silently skipped)',
              'night_image_gate.csv' in str(e) and "'date'" in str(e), str(e))
    os.remove(os.path.join(cell, 'coadd_final', 'night_image_gate.csv'))
    os.rename(os.path.join(cell, 'coadd_final', 'weights.csv'), os.path.join(T, 'weights.aside.csv'))
    try:
        A.nights_to_admit('23h', '23MHz', cell, entries, _Ex(()), co)
        check('nights: refused without the build\'s weights.csv (no time the build fixed its nights)', False)
    except Refusal as e:
        check('nights: refused without the build\'s weights.csv (no time the build fixed its nights)',
              'weights.csv' in str(e), str(e))
    os.rename(os.path.join(T, 'weights.aside.csv'), os.path.join(cell, 'coadd_final', 'weights.csv'))
    os.remove(rec)
    try:
        A.nights_to_admit('23h', '23MHz', cell, entries, _Ex(()), co)
        check('nights: refused without the build\'s night record', False)
    except Refusal as e:
        check('nights: refused without the build\'s night record', A.BUILD_NIGHTS in str(e), str(e))


def test_nights_real():
    """Read only: cells whose builds left nights out of residuals_provisional.csv - seen in another record of the build
    (wrong grid: refusals_grid.csv; Stokes I sanity: sanity_excluded.csv) or, recorded nowhere in the cell, through
    Phase 2: a Science dir holding its survey image was named before the build's night record."""
    import contextlib
    import io
    from exopipe.phase3.core import exclusions as EX
    from exopipe.phase3.lib.masters import coadd as co
    ex = EX.load()
    root = CFG.paths.COADD_SB_ROOT
    for cell, night, because in (('02h_Green', '2025-12-26', 'refusals_grid.csv'), ('23h_23MHz', '2025-10-29', 'sanity_excluded.csv'),
                                  ('11h_Full', '2025-03-09', 'Phase 2 had it before the build')):
        lst, sc = cell.split('_')
        with contextlib.redirect_stdout(io.StringIO()):
            entries = co.discover_dates(lst, sc.replace('MHz', '') if sc.endswith('MHz') else sc)
        before = _ExBefore(ex, lst, sc, night)    # 02h Green and 11h Full: excluded since 2026-10-16 (wrong grid) - the
        take, why = A.nights_to_admit(lst, sc, os.path.join(root, cell), entries, before, co)   # build's own record, as before
        check(f'real {cell}: no night to admit; {night} (not in residuals_provisional.csv) was seen by the build ({because})',
              not take and because in why.get(night, ''), (take, why.get(night)))
        if cell in ('02h_Green', '11h_Full'):
            take, why = A.nights_to_admit(lst, sc, os.path.join(root, cell), entries, ex, co)
            check(f'real {cell}: {night} is excluded in data_decisions since 2026-10-16 (a wrong-grid image, Gregg\'s '
                  f'permission) - refused as excluded', not take and 'excluded' in why.get(night, ''), (take, why.get(night)))


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


def test_review2_nights_and_reference():
    """Second review (2026-10-15): a night ANY record of the installed build names is seen (here night_image_gate.csv and
    refusals_grid.csv, as in production), only the others fall to the Phase-2 rule; a master without SHSCALE refuses;
    a reference copy with another CONSTDIG is never replaced."""
    from astropy.io import fits
    cell = os.path.join(T, 'r2', '23h_23MHz')
    os.makedirs(os.path.join(cell, 'coadd_final'))
    rec = os.path.join(cell, A.BUILD_NIGHTS)
    with open(rec, 'w', newline='') as f:
        csv.writer(f).writerows([['lst', 'freq', 'date'], ['23h', '23MHz', '2025-01-01']])
    with open(os.path.join(cell, 'coadd_final', 'night_image_gate.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([['date', 'verdict', 'reason'], ['2025-01-02', 'FAIL', 'flux scale not measurable']])
    with open(os.path.join(cell, 'refusals_grid.csv'), 'w') as f:
        f.write('23h,23MHz,2025-01-03,wrong grid (Phase-2 anomaly),prep\n')
    import calendar
    from exopipe.phase3.lib.masters import coadd as co
    t = calendar.timegm((2026, 1, 1, 0, 0, 0))
    os.utime(rec, (t, t))
    with open(os.path.join(cell, 'coadd_final', 'weights.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([['date', 'weight'], ['2025-01-01', '1']])
    os.utime(os.path.join(cell, 'coadd_final', 'weights.csv'), (t, t))
    entries = []
    for d in ('2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'):   # every night first in Phase 2 after the build
        entries.append((d,) + _night(os.path.join(T, 'r2', 'phase2'), d, 'Science_20260201_000000'))
    take, why = A.nights_to_admit('23h', '23MHz', cell, entries, _Ex(()), co)
    check('review 2: a night any build record names is seen (night-image record, grid refusals) - re-imaged or not',
          [x[0] for x in take] == ['2025-01-04'] and 'night_image_gate.csv' in why['2025-01-02']
          and 'refusals_grid.csv' in why['2025-01-03'], (take, why))
    src = os.path.join(CFG.paths.COADD_SB_ROOT, '13h_78MHz', 'coadd_final')
    cd = os.path.join(T, 'r2', '13h_78MHz')
    os.makedirs(os.path.join(cd, 'coadd_final'))
    for v in ('I_deep_Taper_R075_78MHz.fits', 'V_deep_Taper_R0_78MHz.fits'):
        with fits.open(os.path.join(src, v)) as h:
            hh = h[0].header.copy()
            for k in ('SHSCALE', 'FLUXBAKE', 'FLUXCORR'):
                hh.remove(k, ignore_missing=True, remove_all=True)
            fits.writeto(os.path.join(cd, 'coadd_final', v), h[0].data, hh)
    try:
        A.installed_reference('13h_78MHz', cd, read_data=False)
        check('review 2: a master without a flux scale (SHSCALE) refuses, as at stage-in', False)
    except Refusal as e:
        check('review 2: a master without a flux scale (SHSCALE) refuses, as at stage-in', 'SHSCALE' in str(e), str(e))
    ents = A.installed_reference('13h_78MHz', os.path.join(CFG.paths.COADD_SB_ROOT, '13h_78MHz'))
    rc = os.path.join(T, 'r2', 'refcopy')
    A.write_reference(rc, '78MHz', ents)
    ents[0]['header'] = ents[0]['header'].copy()
    ents[0]['header']['CONSTDIG'] = '0' * 32
    try:
        A.write_reference(rc, '78MHz', ents)
        check('review 2: a reference copy whose CONSTDIG differs is never replaced', False)
    except Refusal as e:
        check('review 2: a reference copy whose CONSTDIG differs is never replaced', 'other content' in str(e), str(e))


def test_prep_night_regrid_and_shape():
    """prep_night's regrid branch (a pre-cutover night of a cell with post-cutover nights: 11h_50MHz 2024-12-26) writes the
    old loop body's bytes - regrid, scaled image, scale JSON - and keeps needs_regrid as the old body's value; a night whose
    shape differs from the template is skipped (None), as before."""
    import contextlib
    import filecmp
    import io
    from exopipe.phase3.lib.masters import coadd as co, admit_stages as AS
    root = CFG.paths.COADD_SB_ROOT
    ents = A.installed_reference('11h_50MHz', os.path.join(root, '11h_50MHz'))
    rp = A.write_reference(os.path.join(T, 'prep11'), '50MHz', ents)['I']
    ref = AS.reference_dict(co, rp, ents[0])
    with contextlib.redirect_stdout(io.StringIO()):
        nights = co.discover_dates('11h', '50')
    post = [x for x in nights if x[0] >= co.CUTOVER_DATE]
    pre = [x for x in nights if x[0] < co.CUTOVER_DATE]
    d, sd, img = [x for x in pre if x[0] == '2024-12-26'][0]
    wo, wn = os.path.join(T, 'p11_old'), os.path.join(T, 'p11_new')
    for w in (wo, wn):
        os.makedirs(os.path.join(w, 'prep'), exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        old = _old_prep_body(co, d, sd, img, wo, ref['tmpl_hdr'], ref['target_shape'], ref['data'], ref['wcs'],
                             ref['beam_pix'], ref['horizon_mask'], rp, post)
        new = co.prep_night(d, sd, img, ref, wn, post)
    same = all(filecmp.cmp(os.path.join(wo, 'prep', f), os.path.join(wn, 'prep', f), shallow=False)
               for f in (f'{d}_regrid.fits', f'{d}_scaled.fits', f'{d}_scale.json'))
    check('prep_night regrid branch (11h_50MHz 2024-12-26) = the old loop body, byte for byte, needs_regrid kept',
          same and old[1]['needs_regrid'] == new[1]['needs_regrid'] == post, (same, type(new[1]['needs_regrid'])))
    small = dict(ref, target_shape=(ref['target_shape'][0] - 1, ref['target_shape'][1]))
    with contextlib.redirect_stdout(io.StringIO()):
        skip = co.prep_night(post[0][0], post[0][1], post[0][2], small, wn, post)
    check('prep_night: a night whose shape differs from the template is skipped (None)', skip is None, skip)
    ws = os.path.join(T, 'p11_stale')                       # a stale regrid of this night left by an earlier run
    os.makedirs(os.path.join(ws, 'prep'), exist_ok=True)
    stale = os.path.join(ws, 'prep', f'{d}_regrid.fits')
    shutil.copy2(os.path.join(wn, 'prep', f'{d}_scaled.fits'), stale)          # any other image on the grid
    with contextlib.redirect_stdout(io.StringIO()):
        co.prep_night(d, sd, img, ref, ws, post)
    reused = filecmp.cmp(stale, os.path.join(wn, 'prep', f'{d}_scaled.fits'), shallow=False)
    moved = AS.move_aside(ws, d, 'stamp1')
    with contextlib.redirect_stdout(io.StringIO()):
        co.prep_night(d, sd, img, ref, ws, post)
    fresh = filecmp.cmp(stale, os.path.join(wn, 'prep', f'{d}_regrid.fits'), shallow=False)
    kept = os.path.exists(os.path.join(ws, 'prep', '_superseded', 'stamp1', f'{d}_regrid.fits'))
    check('stale regrid (a pre-cutover night): prep_night alone reuses it (the hazard); after move_aside it is made again '
          'from the image and the old one kept aside, not deleted', reused and fresh and kept
          and f'{d}_regrid.fits' in moved, (reused, fresh, kept, moved))


def make_admission_copy(cell, night, root):
    """A copy of an installed cell's records and masters under root/<cell>/ in which ``night`` - a member - is taken out
    of every record of the build (admit.BUILD_RECORDS, and stage2_verdict.csv) and the build's records are dated one
    day before the night's earliest Science dir (by its name): the night then qualifies, as a night Phase 2 first had
    after the build would (no real night does today). The copy is a consistent build without the night: the stage-2
    record carries the thresholds that population gives, and the population record is recomputed (and passes its test).
    Returns the copy's path."""
    import calendar
    import glob
    import re
    from exopipe.phase3.lib.masters import build_stats as BS, coadd as co
    src, dst = os.path.join(CFG.paths.COADD_SB_ROOT, cell), os.path.join(root, cell)
    os.makedirs(os.path.join(dst, 'coadd_final'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'coadd_provisional'), exist_ok=True)
    lst, sc = cell.split('_', 1)
    for rel in [r for r, _c in A.BUILD_RECORDS] + ['coadd_final/stage2_verdict.csv', f'coadd_final/I_deep_Taper_R075_{sc}.fits',
                                                   f'coadd_final/V_deep_Taper_R0_{sc}.fits',
                                                   f'coadd_provisional/I_deep_Taper_R075_{sc}.fits',   # stage 2's references
                                                   f'coadd_provisional/V_deep_Taper_R0_{sc}.fits']:
        if os.path.exists(os.path.join(src, rel)):
            shutil.copy2(os.path.join(src, rel), os.path.join(dst, rel))
    for rel, col in list(A.BUILD_RECORDS) + [('coadd_final/stage2_verdict.csv', 'date')]:
        p = os.path.join(dst, rel)
        if not os.path.exists(p):
            continue
        rows = list(csv.reader(open(p, newline='')))
        i = rows[0].index(col) if col and rows and col in rows[0] else 2
        keep = [r for r in rows if not (len(r) > i and r[i].strip() == night)]
        with open(p, 'w', newline='') as f:
            csv.writer(f).writerows(keep)
    s2p, _rows = BS.stage2_source(dst)                          # the thresholds a build without the night applied
    st2 = BS.stage2_stats(_rows, float(co.OUTLIER_SIGMA), float(co.LOW_OUTLIER_SIGMA))
    rows = list(csv.DictReader(open(s2p, newline='')))
    for r in rows:
        r['i_thresh_mjy'], r['v_thresh_mjy'] = repr(st2['I']['thr_hi']), repr(st2['V']['thr_hi'])
    with open(s2p, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    BS.write(dst, BS.compute(dst, lst, sc, co, CFG, 'stage'))   # the population without the night
    bad = BS.recheck(os.path.join(dst, 'coadd_final', BS.FILE), co)
    assert not bad, bad                                          # the copy is one consistent build
    names = [re.match(A.SCIENCE_DIR, os.path.basename(x)) for x in glob.glob(f'{co.PHASE2_ROOT}/{lst}/{night}/Science_*')]
    first = min(calendar.timegm(time.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')) for m in names if m)
    t = first - 86400
    for rel in (A.BUILD_NIGHTS, 'coadd_final/weights.csv'):     # weights.csv is the boundary (admit.build_time)
        os.utime(os.path.join(dst, rel), (t, t))
    return dst


def test_admit_prep():
    """`masters --stages admit-prep` on a copy of 13h_78MHz with one member night made 'new': only that night is prepared,
    against the installed masters in native units (admit/ref/), by prep_night; the flow pair in the record; the content
    test of the scaled image passes; --epochs restricts, and naming no night of the LST takes none; a re-run moves the
    earlier prep files aside; --resume (the real resume_check, the scratch record marked as from a clean tree) skips the
    same tree and re-runs for another cell tree or a re-imaged night; refusals (a population record that fails its content
    test or is missing, a production tree); B-03's wrong-grid test on a real wrong-grid night."""
    import argparse
    import contextlib
    import io
    import json
    import subprocess
    from astropy.io import fits
    from exopipe.phase3.core import record as R
    from exopipe.phase3.lib.masters import admit_stages as AS, coadd as co, deep_stages as DS
    cell = '13h_78MHz'
    w = list(csv.DictReader(open(os.path.join(CFG.paths.COADD_SB_ROOT, cell, 'coadd_final', 'weights.csv'))))
    members = sorted(r['date'] for r in w if r.get('in_master') == '1' and r.get('qa3_guard') != 'reference-identity')
    night = members[-1]
    wt = os.path.join(T, 'admit_wt')
    cd = make_admission_copy(cell, night, wt)
    out = os.path.join(T, 'admit_out')
    name = f'admit-prep_{cell}'
    rp = os.path.join(out, '_records', 'masters', f'{name}.json')

    def cli(*a):
        p = subprocess.run([sys.executable, '-m', 'exopipe.phase3', 'masters', '--tier', 'deep', '--cells', cell,
                            '--stages', 'admit-prep', '--cores', 'none', '--out', out] + list(a), cwd=REPO,
                           env=dict(os.environ), capture_output=True, text=True, timeout=1800)
        return p.returncode, (json.load(open(rp)) if os.path.exists(rp) else {}), p.stdout + p.stderr
    rc, rec, txt = cli('--deep-installed-root', wt, '--dry-run')
    check('admit-prep dry run: plans the one new night, writes nothing', rc == 0 and night in txt and
          not os.path.exists(os.path.join(cd, A.ADMIT_DIR)), txt[-400:])
    rc, rec, txt = cli('--deep-installed-root', wt)
    meta = (rec.get('meta') or {}).get('admit') or {}
    sp = os.path.join(cd, A.ADMIT_DIR, 'prep', f'{night}_scaled.fits')
    tests = {t['name']: t['passed'] for t in rec.get('tests', [])}
    check('admit-prep: only the night made new is prepared (exit 0), its flow pair against admit/ref/ recorded',
          rc == 0 and rec.get('status') == 'done' and [d['date'] for d in meta.get('dates', [])] == [night]
          and os.path.exists(sp) and meta['pairs'][0]['reference'].endswith('/admit/ref/I_deep_Taper_R075_78MHz.fits')
          and meta['pairs'][0]['out_dir'].endswith('/admit/flow') and rec.get('admitted') == [night],
          (rc, rec.get('reason'), txt[-400:]))
    check('admit-prep: the scaled image and the population record pass their content tests', tests and all(tests.values())
          and tests.get('admit_prep.population_record') is True, tests)

    class _Run:                                   # the real resume_check (core.record), as Run.resume_check calls it
        def resume_check(self, unit, inputs, admitted):
            return R.resume_check(out, 'masters', unit, inputs=inputs, admitted=admitted)
    rec_c = json.load(open(rp))
    rec_c['commit'] = str(rec_c.get('commit')).replace('-dirty', '')    # this scratch record: as if from a clean tree
    json.dump(rec_c, open(rp, 'w'))
    same = DS._resume_ok(_Run(), name, 'admit-prep', cell, out, argparse.Namespace(deep_installed_root=wt, epochs=None))
    wt2 = os.path.join(T, 'admit_wt2')
    shutil.copytree(wt, wt2)
    other = DS._resume_ok(_Run(), name, 'admit-prep', cell, out, argparse.Namespace(deep_installed_root=wt2, epochs=None))
    sd_new, _img = _night(os.path.join(T, 'reimaged', '13h'), night, 'Science_20991231_000000', bands=('78MHz',))
    img_new = os.path.join(sd_new, '78MHz', 'I', 'deep', os.path.basename(meta['dates'][0]['align_img']))
    shutil.copy2(meta['dates'][0]['align_img'], img_new)
    real_dd = co.discover_dates
    co.discover_dates = lambda l, f: [(d, sd_new, img_new) if d == night else (d, sd, im) for d, sd, im in real_dd(l, f)]
    try:
        reim = DS._resume_ok(_Run(), name, 'admit-prep', cell, out, argparse.Namespace(deep_installed_root=wt, epochs=None))
    finally:
        co.discover_dates = real_dd
    check('admit-prep --resume (real resume_check): the same tree skips; another cell tree and a re-imaged night re-run',
          same[0] is True and other == (False, 'input set changed') and reim == (False, 'input set changed'),
          (same, other, reim))
    from exopipe.phase3.core import exclusions as EXm
    real_load = EXm.load
    pop_night = members[0]

    class _ExPlus:                                # data_decisions.csv as it is, plus one population night excluded since
        def __init__(self, base):
            self.base = base

        def __getattr__(self, k):
            return getattr(self.base, k)

        def is_excluded(self, l, sc_, d):
            return (l, sc_, d) == ('13h', '78MHz', pop_night) or self.base.is_excluded(l, sc_, d)
    EXm.load = lambda *a, **k: _ExPlus(real_load(*a, **k))
    try:
        why_pop = AS.population_problem(cd, cell, co)
        excl = DS._resume_ok(_Run(), name, 'admit-prep', cell, out, argparse.Namespace(deep_installed_root=wt, epochs=None))
    finally:
        EXm.load = real_load
    check('admit-prep: a population night excluded after the build refuses the stage and makes --resume re-run (no input '
          'file changes - fourth review)', why_pop and pop_night in why_pop and excl[0] is False and pop_night in excl[1],
          (why_pop, excl))
    refI = os.path.join(cd, A.ADMIT_DIR, 'ref', 'I_deep_Taper_R075_78MHz.fits')
    os.rename(refI, refI + '.aside')
    try:
        gone = DS._resume_ok(_Run(), name, 'admit-prep', cell, out, argparse.Namespace(deep_installed_root=wt, epochs=None))
    finally:
        os.rename(refI + '.aside', refI)
    check('admit-prep: the reference copies are recorded outputs - one moved away makes --resume re-run',
          gone[0] is False and 'output missing' in gone[1] and 'admit/ref' in gone[1], gone)
    shutil.copy2(refI, refI + '.orig')
    with fits.open(refI, mode='update') as h:              # the copy's beam changed (its data untouched)
        h[0].header['BMAJ'] = float(h[0].header['BMAJ']) * 1.01
    try:
        tamper = DS._resume_ok(_Run(), name, 'admit-prep', cell, out, argparse.Namespace(deep_installed_root=wt, epochs=None))
    finally:
        os.replace(refI + '.orig', refI)
    check('admit-prep: a reference copy whose header no longer matches its master fails its own content test, so '
          '--resume re-runs (the inputs are unchanged - the content test is what sees it)',
          tamper[0] is False and 'admit_reference' in tamper[1], tamper)
    mI = os.path.join(cd, 'coadd_final', 'I_deep_Taper_R075_78MHz.fits')
    shutil.copy2(mI, mI + '.orig')
    with fits.open(mI, mode='update') as h:                # a header-only fix of the installed master (the beam)
        h[0].header['BMAJ'] = float(h[0].header['BMAJ']) * 1.01
    try:
        try:
            A.write_reference(cd, '78MHz', A.installed_reference(cell, cd))
            w_ref = 'written'
        except Refusal as e:
            w_ref = str(e)
    finally:
        os.replace(mI + '.orig', mI)
    check('admit-prep: a header-only change of the installed master is seen - write_reference refuses to replace the copy '
          '(admit into the new master from a fresh admit/)', 'other content' in w_ref, w_ref)
    rc, rec, _ = cli('--deep-installed-root', wt, '--epochs', f'13h_{night},13h_2025-01-10')
    meta = (rec.get('meta') or {}).get('admit') or {}
    sup = os.path.join(cd, A.ADMIT_DIR, 'prep', '_superseded')
    aside = sorted(os.listdir(sup)) if os.path.isdir(sup) else []
    moved = sorted(os.listdir(os.path.join(sup, aside[0]))) if aside else []
    check('admit-prep --epochs: the named night that qualifies is prepared again (a re-run re-prepares; its earlier prep '
          'files moved aside, none deleted), the excluded one says why',
          rc == 0 and [d['date'] for d in meta.get('dates', [])] == [night] and f'{night}_scaled.fits' in moved
          and 'excluded' in (rec.get('meta') or {}).get('not_admitted', {}).get('2025-01-10', ''), (rc, rec.get('reason'), moved))
    before = json.load(open(rp))
    rc, rec, txt = cli('--deep-installed-root', wt, '--epochs', '13h_2025-01-10')
    check('admit-prep --epochs naming only a night that does not qualify (excluded) selects no cell: refused, the done '
          'record unchanged', rc == 2 and 'names no night that a selected cell has to admit' in txt
          and json.load(open(rp)) == before, (rc, txt[-300:]))
    before = json.load(open(rp))
    rc, rec, txt = cli('--deep-installed-root', wt, '--epochs', '14h_2025-01-01')
    rc2, rec2, txt2 = cli('--deep-installed-root', wt, '--epochs', f'13h_{members[0]}')
    check('admit-prep --epochs naming no night the cell has to admit (another LST; a member night of its own) selects no '
          'cell: refused (exit 2) - never every night, and the cell\'s record is not replaced by "no input"',
          rc == 2 and rc2 == 2 and 'names no night that a selected cell has to admit' in txt + txt2
          and json.load(open(rp)) == before, (rc, rc2, txt[-200:], txt2[-200:]))
    bsp = os.path.join(cd, 'coadd_final', 'build_stats.json')
    good = open(bsp).read()
    st = json.loads(good)
    st['stage2']['I']['median'] = st['stage2']['I']['median'] * 2
    json.dump(st, open(bsp, 'w'))
    rc, rec, _ = cli('--deep-installed-root', wt)
    check('admit-prep refuses a population record that fails its content test', rc == 2 and 'fails its content test'
          in rec.get('reason', ''), rec.get('reason'))
    os.remove(bsp)
    rc, rec, _ = cli('--deep-installed-root', wt)
    check('admit-prep refuses without the population record (build_stats.json)', rc == 2 and 'build-stats' in rec.get('reason', ''),
          rec.get('reason'))
    rc, rec, _ = cli()
    check('admit-prep refuses the production tree until cutover', rc == 2 and 'production' in rec.get('reason', ''), rec.get('reason'))
    with contextlib.redirect_stdout(io.StringIO()):
        g = [x for x in co.discover_dates('02h', 'Green') if x[0] == '2025-12-26']
    mh = fits.getheader(os.path.join(CFG.paths.COADD_SB_ROOT, '02h_Green', 'coadd_final', 'I_deep_Taper_R075_Green.fits'))
    mshape = (int(mh['NAXIS2']), int(mh['NAXIS1']))
    bad = AS.off_grid('2025-12-26', g[0][2], mshape, co.CUTOVER_DATE) if g else 'no image'
    okn = AS.off_grid(night, meta['dates'][0]['align_img'], fits.getdata(sp).shape[-2:], co.CUTOVER_DATE) if meta.get('dates') else 'x'
    junk = os.path.join(T, 'not_a_fits.fits')
    open(junk, 'w').write('not FITS')
    unread = AS.off_grid('2025-06-01', junk, mshape, co.CUTOVER_DATE)
    check('B-03 at admission: the real wrong-grid night 02h Green 2025-12-26 is off the master\'s grid; a member night is '
          'not; an unreadable header is refused, never assumed on grid', isinstance(bad, tuple) and bad != mshape
          and okn is None and unread == 'unreadable', (bad, mshape, okn, unread))
    wtg = os.path.join(T, 'admit_wtg')
    make_admission_copy('02h_Green', '2025-12-26', wtg)          # the wrong-grid night made new (out of every record)
    for v_ in ('I_deep_Taper_R075', 'V_deep_Taper_R0'):          # 02h_Green keeps no coadd_provisional/ (41 cells): its final
        q_ = os.path.join(wtg, '02h_Green', 'coadd_provisional', f'{v_}_Green.fits')   # masters stand in, test only, so
        if not os.path.exists(q_):                                # the cell passes the provisional gate and reaches the night
            shutil.copy2(os.path.join(wtg, '02h_Green', 'coadd_final', f'{v_}_Green.fits'), q_)
    ddb = data_decisions_before(os.path.join(T, 'data_decisions_before_20261016.csv'))   # the night is excluded since
    pg = subprocess.run([sys.executable, '-m', 'exopipe.phase3', 'masters', '--tier', 'deep', '--cells', '02h_Green',   # then:
                         '--stages', 'admit-prep', '--cores', 'none', '--out', out, '--deep-installed-root', wtg,   # replay the
                         '--parity', '--set', f'paths.DATA_DECISIONS={ddb}'], cwd=REPO,   # list before (a --parity run may)
                        env=dict(os.environ), capture_output=True, text=True, timeout=1800)
    rpg = os.path.join(out, '_records', 'masters', 'admit-prep_02h_Green.json')
    recg = json.load(open(rpg)) if os.path.exists(rpg) else {}
    items = recg.get('refused_items') or []
    check('admit-prep (CLI): the wrong-grid night, made new, is refused as an item (B-03 a) - never regridded in',
          pg.returncode == 2 and items and items[0].get('gate') == 'admit-prep.wrong_grid'
          and 'off the installed master' in items[0].get('reason', '')
          and not os.path.exists(os.path.join(wtg, '02h_Green', A.ADMIT_DIR, 'prep', '2025-12-26_scaled.fits')),
          (pg.returncode, recg.get('reason'), items[:1], (pg.stdout + pg.stderr)[-300:]))

def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_nights_rule()
        test_nights_real()
        test_reference()
        test_prep_night_equivalence()
        test_review2_nights_and_reference()
        test_prep_night_regrid_and_shape()
        test_admit_prep()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
