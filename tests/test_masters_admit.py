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
    want = {'2025-01-01': 'installed build', '2025-01-02': 'installed build', '2025-01-03': 'older than the build\'s night record',
            '2025-01-04': A.ADMISSIONS, '2025-01-06': 'excluded', '2025-01-07': 'no survey image'}
    check('nights: each other night says why (a recorded night stays out even with a newer image)',
          set(why) == set(want) and all(want[d] in why[d] for d in want), why)
    take, why = A.nights_to_admit('23h', '23MHz', cell, entries, _Ex(()), nights=['2025-01-05', '2025-01-03', '2025-02-01'])
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
    """Read only: cells whose builds left nights out of residuals_provisional.csv - seen in another record of the build
    (wrong grid: refusals_grid.csv; Stokes I sanity: sanity_excluded.csv) or, recorded nowhere in the cell, by the time
    of their image (written before the build's night record)."""
    import contextlib
    import io
    from exopipe.phase3.core import exclusions as EX
    from exopipe.phase3.lib.masters import coadd as co
    ex = EX.load()
    root = CFG.paths.COADD_SB_ROOT
    for cell, night, because in (('02h_Green', '2025-12-26', 'refusals_grid.csv'), ('23h_23MHz', '2025-10-29', 'sanity_excluded.csv'),
                                  ('11h_Full', '2025-03-09', 'older than the build\'s night record')):
        lst, sc = cell.split('_')
        with contextlib.redirect_stdout(io.StringIO()):
            entries = co.discover_dates(lst, sc.replace('MHz', '') if sc.endswith('MHz') else sc)
        take, why = A.nights_to_admit(lst, sc, os.path.join(root, cell), entries, ex)
        check(f'real {cell}: no night to admit; {night} (not in residuals_provisional.csv) was seen by the build ({because})',
              not take and because in why.get(night, ''), (take, why.get(night)))


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
    refusals_grid.csv, as in production), only the others fall to the image-time rule; a master without SHSCALE refuses;
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
    t = time.time() - 3600
    os.utime(rec, (t, t))
    entries = []
    for d in ('2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'):
        img = os.path.join(T, 'r2', f'{d}.fits')
        open(img, 'w').write('x')
        os.utime(img, (t + 100, t + 100))               # every image newer than the build's night record
        entries.append((d, os.path.join(T, 'r2', 'Science_x'), img))
    take, why = A.nights_to_admit('23h', '23MHz', cell, entries, _Ex(()))
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

def make_admission_copy(cell, night, root):
    """A copy of an installed cell's records and masters under root/<cell>/ in which ``night`` - a member - is taken out
    of every record of the build (admit.BUILD_RECORDS, and stage2_verdict.csv) and the build's night record is dated
    2000-01-01: the night then qualifies, as a night the build never saw would (no real night does today). The copy's
    population record is recomputed without the night. Returns the copy's path."""
    import calendar
    from exopipe.phase3.lib.masters import build_stats as BS, coadd as co
    src, dst = os.path.join(CFG.paths.COADD_SB_ROOT, cell), os.path.join(root, cell)
    os.makedirs(os.path.join(dst, 'coadd_final'), exist_ok=True)
    sc = cell.split('_', 1)[1]
    for rel in [r for r, _c in A.BUILD_RECORDS] + ['coadd_final/stage2_verdict.csv', f'coadd_final/I_deep_Taper_R075_{sc}.fits',
                                                   f'coadd_final/V_deep_Taper_R0_{sc}.fits']:
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
    BS.write(dst, BS.compute(dst, *cell.split('_', 1), co, CFG, 'stage'))   # the population without the night
    t = calendar.timegm((2000, 1, 1, 0, 0, 0))
    os.utime(os.path.join(dst, A.BUILD_NIGHTS), (t, t))
    return dst


def test_admit_prep():
    """`masters --stages admit-prep` on a copy of 13h_78MHz with one member night made 'new': only that night is prepared,
    against the installed masters in native units (admit/ref/), by prep_night; the flow pair in the record; the content
    test of the scaled image passes; --epochs restricts; the resume identity is the nights that qualify now; refusals (no
    population record, a production tree)."""
    import argparse
    import json
    import subprocess
    from exopipe.phase3.lib.masters import deep_stages as DS
    cell = '13h_78MHz'
    w = list(csv.DictReader(open(os.path.join(CFG.paths.COADD_SB_ROOT, cell, 'coadd_final', 'weights.csv'))))
    members = sorted(r['date'] for r in w if r.get('in_master') == '1' and r.get('qa3_guard') != 'reference-identity')
    night = members[-1]
    wt = os.path.join(T, 'admit_wt')
    cd = make_admission_copy(cell, night, wt)
    out = os.path.join(T, 'admit_out')

    def cli(*a):
        p = subprocess.run([sys.executable, '-m', 'exopipe.phase3', 'masters', '--tier', 'deep', '--cells', cell,
                            '--stages', 'admit-prep', '--cores', 'none', '--out', out] + list(a), cwd=REPO,
                           env=dict(os.environ), capture_output=True, text=True, timeout=1800)
        rp = os.path.join(out, '_records', 'masters', f'admit-prep_{cell}.json')
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
    check('admit-prep: the scaled image passes its content test (input / scale re-derived)', tests and all(tests.values()), tests)

    class _Run:
        def resume_check(self, name, inputs, admitted):
            self.got = admitted
            return True, ''
    seen = {}
    for label, ep in (('none', None), ('the new night', f'13h_{night}'), ('a member', f'13h_{members[0]}')):
        r_ = _Run()
        ok, why = DS._resume_ok(r_, f'admit-prep_{cell}', 'admit-prep', cell, out,
                                argparse.Namespace(deep_installed_root=wt, epochs=ep))
        seen[label] = r_.got
    check('admit-prep --resume identity: the nights that qualify now ([new night] without --epochs or naming it; [] for a '
          'member) - compared with the record\'s admitted set', seen == {'none': [night], 'the new night': [night],
                                                                      'a member': []}, seen)
    rc, rec, _ = cli('--deep-installed-root', wt, '--epochs', f'13h_{night},13h_2025-01-10')
    meta = (rec.get('meta') or {}).get('admit') or {}
    check('admit-prep --epochs: the named night that qualifies is prepared, the excluded one says why',
          rc == 0 and [d['date'] for d in meta.get('dates', [])] == [night]
          and 'excluded' in (rec.get('meta') or {}).get('not_admitted', {}).get('2025-01-10', ''), (rc, rec.get('reason')))
    rc, rec, _ = cli('--deep-installed-root', wt, '--epochs', '13h_2025-01-10')
    check('admit-prep --epochs: a named night that does not qualify -> no input, with its reason',
          rc == 3 and 'no night to admit' in rec.get('reason', '') and 'excluded' in rec.get('reason', ''), rec.get('reason'))
    os.remove(os.path.join(cd, 'coadd_final', 'build_stats.json'))
    rc, rec, _ = cli('--deep-installed-root', wt)
    check('admit-prep refuses without the population record (build_stats.json)', rc == 2 and 'build-stats' in rec.get('reason', ''),
          rec.get('reason'))
    rc, rec, _ = cli()
    check('admit-prep refuses the production tree until cutover', rc == 2 and 'production' in rec.get('reason', ''), rec.get('reason'))


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
