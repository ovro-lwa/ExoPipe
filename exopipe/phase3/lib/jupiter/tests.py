"""Content tests of the Jupiter program's Phase-1 classes (spec plan/specs/jupiter.md section 6), registered with
@content_test (core/validate discovers lib/<step>/tests*.py). Each checks the file's CONTENT; none writes into a product
directory (the in-place cell test runs on a temporary directory of symlinks).

  jupiter_meas        rows_and_columns, admission_coverage      meas_<tier>.csv
  jupiter_cutouts     cutout_value_at_jupiter                    cutouts_<tier>_<band>_<S>.npz
  jupiter_crop        crop_equals_cutout                         meas_<tier>.csv (the node-local crops its rows name)
  jupiter_fits        one_record_per_crop                        fits_<tier>.jsonl
  jupiter_assign      assign_recomputed                          assign_<tier>.csv and moons_<tier>.json
  jupiter_cell        tests_1_to_4 (in place: stage_test)        CONTENT_TEST.txt
  jupiter_ephem       ephem_cache                                jupiter_ephem_<date>.csv
  jupiter_window      window_rows                                nights.csv / window.csv
  jupiter_flagged_copy verify_tree (in place)                    phase1_flagged/<LST>/<date>/<Run>/<sb>/ (a directory)
  jupiter_flag_verify verify_ok_line                             VERIFY_<stamp>.txt
  jupiter_served_page served                                     <out>/_served/jupiter_<port>.json (the serve record)
V2 and occurrence classes: tests_v2.py.
"""
import calendar
import csv
import glob
import json
import math
import os
import re
import shutil
import tempfile
import time

from exopipe.phase3.core.validate import content_test

MEAS_COLUMNS = ['cell', 'tier', 'band', 'stokes', 'key', 't_unix', 'jup_ra', 'jup_dec', 'jup_el', 'cml', 'io_phase', 'value',
                'local_rms', 'pix_snr', 'crop', 'window', 'used', 'refused', 'nframes']
FIT_COLS = ('peak', 'snr', 'converged', 'dx_arcmin', 'dy_arcmin', 'ra_err_arcmin', 'dec_err_arcmin', 'fwhm_maj_arcmin',
            'fwhm_min_arcmin')
F16_REL = 2.0 ** -11          # float16 unit roundoff (round to nearest)
_CACHE = {}


def _C():
    from exopipe.phase3.config import CFG
    return CFG


def _legacy():
    from exopipe.phase3.core import env
    env.legacy()


def _read_csv(path):
    st = os.stat(path)
    k = ('csv', path, st.st_mtime_ns, st.st_size)
    if k not in _CACHE:
        with open(path, newline='') as f:
            rd = csv.DictReader(f)
            rows = list(rd)
            _CACHE[k] = (rd.fieldnames or [], rows)
    return _CACHE[k]


def _tier_of(path, prefix):
    m = re.match(prefix + r'_(10s|1min|10min)[._]', os.path.basename(path))
    return m.group(1) if m else None


def _cell_of(path):
    return os.path.basename(os.path.dirname(os.path.abspath(path)))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def _ts_unix(key):
    return float(calendar.timegm(time.strptime(key, '%Y%m%d_%H%M%S')))


def _meas_rows(od, tier):
    p = os.path.join(od, f'meas_{tier}.csv')
    return _read_csv(p)[1] if os.path.exists(p) else None


def _pjm():
    """clv15/phase3_jupiter_measure, imported in place (its ephem() is the engine's own interpolation)."""
    _legacy()
    import phase3_jupiter_measure as PJM
    from exopipe.phase3.lib.jupiter import mirrors
    mirrors.check_literals(('measure',))
    return PJM


def _ephem_at(date):
    """The engine's ephem(date) interpolator on the cache under jupiter.EPHEM_DIR (its module global EPH is pointed there
    for the duration of the read when an override redirects it, as the in-place driver does for the engine)."""
    want = _C().jupiter.EPHEM_DIR
    k = ('ephem', want, date)
    if k not in _CACHE:
        PJM = _pjm()
        old = PJM.EPH
        try:
            PJM.EPH = want
            _CACHE[k] = PJM.ephem(date)
        finally:
            PJM.EPH = old
    return _CACHE[k]


def _summary(counts):
    return '; '.join(f'{k} {v}' for k, v in counts.items() if v) or 'none'


# ---------------------------------------------------------------------------------------------------------------------
# jupiter_meas
# ---------------------------------------------------------------------------------------------------------------------
@content_test('jupiter_meas')
def rows_and_columns(path, **ctx):
    """Columns = the production columns; types; pix_snr = value / local_rms (within print rounding); keys unique per (band,
    Stokes); 10-s key = file-name time = t_unix (TIME_BASIS 'file-name', ledger 2074); 1-min window = minute // 10; 10-min key =
    its x2 residual's window; crop names and the crop selection rule (|pix_snr| >= FIT_SNR; 1-min anchor I always)."""
    J, M = _C().jupiter, _C().jupiter_measure
    tier, cell = _tier_of(path, 'meas'), _cell_of(path)
    cols, rows = _read_csv(path)
    if cols != MEAS_COLUMNS:
        return False, len(rows), f'columns {cols} != the production columns {MEAS_COLUMNS}'
    if not rows:
        return False, 0, 'no rows (the engine writes meas_<tier>.csv only with rows)'
    bad = {k: 0 for k in ('identity', 'band', 'numbers', 'pix_snr', 'duplicate', 'key/time', 'window', 'crop name',
                          'crop rule', 'nframes')}
    first = {}
    seen = set()
    bands = J.SUBBANDS + (J.WIDEBANDS if tier == '10min' else ())
    for r in rows:
        if r['cell'] != cell or r['tier'] != tier:
            bad['identity'] += 1
        if r['band'] not in bands or r['stokes'] not in ('I', 'V'):
            bad['band'] += 1
        t, v, rms, snr = _f(r['t_unix']), _f(r['value']), _f(r['local_rms']), _f(r['pix_snr'])
        if not all(math.isfinite(_f(r[c])) for c in ('t_unix', 'jup_ra', 'jup_dec', 'jup_el', 'value')):
            bad['numbers'] += 1
        if math.isfinite(rms) and rms > 0 and math.isfinite(snr):
            tol = 0.0051 + 5e-5 / rms + abs(v) * 5e-5 / rms ** 2
            if abs(snr - v / rms) > tol:
                bad['pix_snr'] += 1
        k = (r['band'], r['stokes'], r['key'])
        if k in seen:
            bad['duplicate'] += 1
        seen.add(k)
        crop = r['crop']
        if tier == '10s':
            if not re.match(r'^\d{8}_\d{6}$', r['key']) or abs(t - _ts_unix(r['key'])) > 0.051:
                bad['key/time'] += 1
            if r['nframes'] != '1' or r['window'] not in [str(i) for i in range(J.WINDOWS_PER_HOUR)]:
                bad['window'] += 1
        elif tier == '1min':
            m = re.match(r'^m(\d\d)$', r['key'])
            if not m or str(int(m.group(1)) // 10) != r['window']:
                bad['window'] += 1
            if not r['nframes'].isdigit() or int(r['nframes']) < 1:
                bad['nframes'] += 1
        else:
            m = re.match(r'^w(\d)$', r['key'])
            if not m or m.group(1) != r['window'] or not crop.endswith(f"_{r['band']}_{r['stokes']}_w{m.group(1)}_resid_loo.fits"):
                bad['window'] += 1
        if tier in ('10s', '1min'):
            if crop and os.path.basename(crop) != f"{tier}_{cell}_{r['band']}_{r['stokes']}_{r['key']}.fits":
                bad['crop name'] += 1
            always = tier == '1min' and r['band'] in J.ANCHOR_SUBBANDS and r['stokes'] == 'I'
            if math.isfinite(snr) and abs(abs(snr) - M.FIT_SNR) > 0.006:
                want = always or abs(snr) >= M.FIT_SNR
                if bool(crop) != want:
                    bad['crop rule'] += 1
            elif always and not crop:
                bad['crop rule'] += 1
        first.setdefault((r['band'], r['stokes']), 0)
        first[(r['band'], r['stokes'])] += 1
    n_bad = sum(bad.values())
    return n_bad == 0, len(rows), (f'{len(rows)} rows over {len(first)} (band, Stokes); problems: {_summary(bad)}')


@content_test('jupiter_meas')
def admission_coverage(path, **ctx):
    """TEST4 at file level against the admission tables under jupiter.SUBLST_ROOT (the 2026-10-05 products need the pre-bake
    reserve: --set jupiter.SUBLST_ROOT=<reserve>): 10-s keys are admission rows with data (refused != blank, window set) and at
    most TEST4_MAX_MISSING of those are absent; 1-min nframes <= the admitted frames of that LST minute. Bands: every subband,
    or ctx subbands= (a --scopes run, recorded with the output so --resume re-tests the same set)."""
    J, M = _C().jupiter, _C().jupiter_measure
    tier, cell = _tier_of(path, 'meas'), _cell_of(path)
    if tier == '10min':
        return True, None, 'n/a for the 10-min tier (one row per x2 residual: rows_and_columns checks the window keys)'
    sl = ctx.get('sublst_root') or J.SUBLST_ROOT
    lst = cell.split('_', 1)[0]
    _, rows = _read_csv(path)
    got = {}
    for r in rows:
        got.setdefault((r['band'], r['stokes']), []).append(r)
    n_exp = n_miss = extra = over = no_adm = 0
    for band in (ctx.get('subbands') or J.SUBBANDS):        # a --scopes run measured only these (the engine's BANDS)
        for S in ('I', 'V'):
            adm = os.path.join(sl, lst, cell, band, S, f'admission_{cell}_{band}_{S}.csv')
            rr = got.get((band, S), [])
            if not os.path.exists(adm):
                no_adm += bool(rr)
                continue
            A = _read_csv(adm)[1]
            if tier == '10s':
                exp = {a['ts'] for a in A if a['refused'] != 'blank' and a['window'] != ''}
                keys = {r['key'] for r in rr}
                n_exp += len(exp)
                n_miss += len(exp - keys)
                extra += len(keys - exp)
            else:
                cnt = {}
                for a in A:
                    if a['used'] == '1' and a['window'] != '':
                        m = int(math.floor(float(a['lst_min'])))
                        cnt[m] = cnt.get(m, 0) + 1
                for r in rr:
                    if int(r['nframes']) > cnt.get(int(r['key'][1:]), 0):
                        over += 1
    if tier == '10s':
        ok = n_exp > 0 and extra == 0 and no_adm == 0 and n_miss <= M.TEST4_MAX_MISSING * n_exp
        return ok, n_miss, (f'{n_exp - n_miss} of {n_exp} admission frames with data listed (missing {n_miss}, allowed '
                            f'{M.TEST4_MAX_MISSING:.0%}); {extra} rows not in the admission tables; {no_adm} (band, Stokes) '
                            f'with rows but no admission table; root {sl}')
    ok = over == 0 and no_adm == 0
    return ok, over, f'{over} minute(s) with nframes above the admitted frames; {no_adm} without admission table; root {sl}'


# ---------------------------------------------------------------------------------------------------------------------
# jupiter_cutouts
# ---------------------------------------------------------------------------------------------------------------------
def _parse_cutout_name(path):
    m = re.match(r'^cutouts_(10s|1min|10min)_([0-9]+MHz|Red|Green|Blue)_([IV])\.npz$', os.path.basename(path))
    if not m:
        raise ValueError(f'not a cutouts_<tier>_<band>_<S>.npz name: {os.path.basename(path)}')
    return m.groups()


def _bilinear_tol(cut, x, y):
    import numpy as np
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    nb = cut[max(0, y0):y0 + 2, max(0, x0):x0 + 2].astype(np.float64)
    nb = np.nan_to_num(nb)
    return F16_REL * float(np.max(np.abs(nb))) + 5.1e-5 if nb.size else 1e-3


@content_test('jupiter_cutouts')
def cutout_value_at_jupiter(path, **ctx):
    """npz keys and dtypes; key order and times = the meas rows of (tier, band, Stokes); 10 s / 1 min: the bilinear value at
    Jupiter's recorded RA/Dec through the stored WCS = the meas value within float16 rounding (2^-11 of the neighbours) and the
    %.4f print; aux/refused = used/refused (10 s) or nframes (1 min); 10 min (no WCS stored): the value lies within the
    central 3 x 3 pixels' range (the cutout is centred on Jupiter's rounded pixel)."""
    import numpy as np
    tier, band, S = _parse_cutout_name(path)
    od = os.path.dirname(os.path.abspath(path))
    rows = [r for r in (_meas_rows(od, tier) or []) if r['band'] == band and r['stokes'] == S]
    z = np.load(path, allow_pickle=False)
    want = {'cut', 'key', 't_unix'} | ({'aux', 'refused', 'wcs', 'beam', 'fluxdc'} if tier != '10min' else set())
    if set(z.files) != want:
        return False, None, f'npz keys {sorted(z.files)} != {sorted(want)}'
    cut = z['cut']
    if cut.dtype != np.float16 or cut.ndim != 3 or cut.shape[1] != cut.shape[2] or cut.shape[1] % 2 != 1:
        return False, None, f'cut {cut.dtype} {cut.shape}: expected float16 (N, 2H+1, 2H+1)'
    keys = [str(k) for k in z['key']]
    if keys != [r['key'] for r in rows]:
        return False, len(keys), f'{len(keys)} keys vs {len(rows)} meas rows of ({tier}, {band}, {S}) - order or set differs'
    if not np.allclose(z['t_unix'], [float(r['t_unix']) for r in rows], atol=0.051, rtol=0):
        return False, len(keys), 't_unix differs from the meas rows'
    bad, worst = 0, 0.0
    H = cut.shape[1] // 2
    if tier != '10min':
        from astropy.io import fits
        from astropy.wcs import WCS
        from scipy import ndimage
        w = WCS(fits.Header.fromstring(str(z['wcs'])))
        xs, ys = w.world_to_pixel_values(np.array([float(r['jup_ra']) for r in rows]), np.array([float(r['jup_dec']) for r in rows]))
        aux = [str(a) for a in z['aux']]
        ref = [str(a) for a in z['refused']]
        for i, r in enumerate(rows):
            v = float(ndimage.map_coordinates(np.nan_to_num(cut[i].astype(np.float64)), [[ys[i]], [xs[i]]], order=1)[0])
            d = abs(v - float(r['value']))
            tol = _bilinear_tol(cut[i], xs[i], ys[i])
            worst = max(worst, d / tol if tol else 0.0)
            if d > tol:
                bad += 1
            want_aux = str(int(r['used'])) if tier == '10s' else r['nframes']
            if aux[i] != want_aux or ref[i] != (r['refused'] if tier == '10s' else ''):
                bad += 1
        if not float(z['fluxdc']) > 0 or z['beam'].shape != (3,) or not np.all(z['beam'][:2] > 0):
            return False, len(rows), f"fluxdc {z['fluxdc']} / beam {z['beam']} invalid"
    else:
        for i, r in enumerate(rows):
            c = np.nan_to_num(cut[i][H - 1:H + 2, H - 1:H + 2].astype(np.float64))
            v = float(r['value'])
            tol = F16_REL * float(np.max(np.abs(c))) + 5.1e-5
            if not (c.min() - tol <= v <= c.max() + tol):
                bad += 1
    return bad == 0, len(rows), (f'{len(rows)} cutouts ({cut.shape[1]} px side); {bad} disagree with the meas rows'
                                 + (f'; worst |diff| / tolerance {worst:.2f}' if tier != '10min' else ' (3 x 3 range check)'))


# ---------------------------------------------------------------------------------------------------------------------
# jupiter_crop
# ---------------------------------------------------------------------------------------------------------------------
@content_test('jupiter_crop')
def crop_equals_cutout(path, crop_dir=None, **ctx):
    """For every crop a meas_<tier>.csv row names (10 s / 1 min; the 10-min 'crop' is the x2 residual itself, an input):
    the file exists on this node (crops are node-local; crop_dir= reads them from an extracted archive by name), carries
    BMAJ/BMIN/BPA, MJD-MID (= t_unix) and FLUXDC (= the cutout file's), and its pixels equal the stored float16 cutout at the
    WCS offset within float16 rounding with the same NaN pattern."""
    import numpy as np
    from astropy.io import fits
    tier = _tier_of(path, 'meas')
    if tier == '10min':
        return True, None, 'n/a: the 10-min crop column names the x2 residual (an input)'
    od = os.path.dirname(os.path.abspath(path))
    rows = [r for r in _read_csv(path)[1] if r['crop']]
    if not rows:
        return True, 0, 'no crop referenced'
    bad = {'missing': 0, 'cards': 0, 'time': 0, 'fluxdc': 0, 'offset': 0, 'pixels': 0, 'nan': 0}
    npz = {}
    n = 0
    for r in rows:
        f = r['crop']
        if not os.path.exists(f) and crop_dir:
            f = os.path.join(crop_dir, os.path.basename(f))
        if not os.path.exists(f):
            bad['missing'] += 1
            continue
        k = (r['band'], r['stokes'])
        if k not in npz:
            z = np.load(os.path.join(od, f"cutouts_{tier}_{r['band']}_{r['stokes']}.npz"), allow_pickle=False)
            hz = fits.Header.fromstring(str(z['wcs']))
            npz[k] = (z['cut'], {str(x): i for i, x in enumerate(z['key'])}, float(z['fluxdc']), hz['CRPIX1'], hz['CRPIX2'])
        cut, idx, dc, c1, c2 = npz[k]
        with fits.open(f, memmap=False) as h:
            hd, d = h[0].header, np.asarray(h[0].data, dtype=np.float64)
        n += 1
        if any(c not in hd for c in ('BMAJ', 'BMIN', 'BPA', 'MJD-MID', 'FLUXDC')) or not hd['BMAJ'] > 0:
            bad['cards'] += 1
            continue
        if abs((hd['MJD-MID'] - 40587.0) * 86400.0 - float(r['t_unix'])) > 0.06:
            bad['time'] += 1
        if abs(hd['FLUXDC'] - dc) > 1e-9 * max(1.0, abs(dc)):
            bad['fluxdc'] += 1
        dx, dy = c1 - hd['CRPIX1'], c2 - hd['CRPIX2']
        if abs(dx - round(dx)) > 1e-6 or abs(dy - round(dy)) > 1e-6 or r['key'] not in idx:
            bad['offset'] += 1
            continue
        x0, y0 = int(round(dx)), int(round(dy))
        ref = cut[idx[r['key']]][y0:y0 + d.shape[0], x0:x0 + d.shape[1]].astype(np.float64)
        if ref.shape != d.shape:
            bad['offset'] += 1
            continue
        if not np.array_equal(np.isnan(ref), np.isnan(d)):
            bad['nan'] += 1
            continue
        fin = np.isfinite(d)
        if np.any(np.abs(d[fin] - ref[fin]) > F16_REL * np.abs(d[fin]) + 1e-7):
            bad['pixels'] += 1
    nb = sum(bad.values())
    return nb == 0, n, f'{n} of {len(rows)} crops read; problems: {_summary(bad)}'


# ---------------------------------------------------------------------------------------------------------------------
# jupiter_fits
# ---------------------------------------------------------------------------------------------------------------------
def _fit_selected(r, tier):
    J, M = _C().jupiter, _C().jupiter_measure
    if not r['crop']:
        return False, False
    if tier != '10min':
        return True, False
    snr = abs(_f(r['pix_snr']) if r['pix_snr'] else 0.0)
    always = r['band'] in J.ANCHOR_SUBBANDS + ('Green', 'Blue') and r['stokes'] == 'I'
    if always:
        return True, False
    return snr >= M.FIT_SNR, abs(snr - M.FIT_SNR) <= 0.005


@content_test('jupiter_fits')
def one_record_per_crop(path, **ctx):
    """Exactly one record per selected crop (10 s / 1 min: every crop; 10 min: |pix_snr| >= FIT_SNR or anchor/Green/Blue
    Stokes I, :325), none extra; band/Stokes/key = the meas row; dx/dy recomputed from fit_ra/fit_dec and the row's
    jup_ra/jup_dec to 1e-9 arcmin (:340-341); 'err' records counted."""
    tier = _tier_of(path, 'fits')
    od = os.path.dirname(os.path.abspath(path))
    rows = _meas_rows(od, tier)
    if rows is None:
        return False, None, f'no meas_{tier}.csv beside the fits file'
    by_crop = {r['crop']: r for r in rows if r['crop']}
    sel, ambiguous = set(), set()
    for r in rows:
        s, amb = _fit_selected(r, tier)
        if amb:
            ambiguous.add(r['crop'])
        elif s:
            sel.add(r['crop'])
    recs, dup, bad_id, bad_dxy, errs, parse = {}, 0, 0, 0, 0, 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                x = json.loads(line)
            except ValueError:
                parse += 1
                continue
            c = x.get('crop')
            if c in recs:
                dup += 1
            recs[c] = x
            r = by_crop.get(c)
            if r is None or (x.get('band'), x.get('stokes'), x.get('key')) != (r['band'], r['stokes'], r['key']):
                bad_id += 1
                continue
            if 'err' in x:
                errs += 1
            if 'fit_ra' in x:
                ra, dec = float(r['jup_ra']), float(r['jup_dec'])
                dx = ((x['fit_ra'] - ra + 180) % 360 - 180) * math.cos(math.radians(dec)) * 60
                dy = (x['fit_dec'] - dec) * 60
                if abs(dx - x.get('dx_arcmin', float('nan'))) > 1e-9 or abs(dy - x.get('dy_arcmin', float('nan'))) > 1e-9:
                    bad_dxy += 1
    missing = sel - set(recs)
    extra = set(recs) - sel - ambiguous
    ok = not (missing or extra or dup or bad_id or bad_dxy or parse)
    return ok, len(recs), (f'{len(recs)} records for {len(sel)} selected crops ({len(ambiguous)} at the S/N boundary); '
                           f'missing {len(missing)}, extra {len(extra)}, duplicates {dup}, identity {bad_id}, dx/dy {bad_dxy}, '
                           f'unparsable {parse}; imfit errors recorded {errs}')


# ---------------------------------------------------------------------------------------------------------------------
# jupiter_assign
# ---------------------------------------------------------------------------------------------------------------------
def _moons_test(path):
    tier = _tier_of(path, 'moons')
    od = os.path.dirname(os.path.abspath(path))
    ap = os.path.join(od, f'assign_{tier}.csv')
    if not os.path.exists(ap):
        return False, None, f'no assign_{tier}.csv beside {os.path.basename(path)}'
    rows = _read_csv(ap)[1]
    date = _cell_of(path).split('_', 1)[1]
    at = _ephem_at(date)
    moons = json.load(open(path))
    first = {}
    for r in rows:
        first.setdefault(r['key'], float(r['t_unix']))
    bad = 0
    if list(moons) != list(first):
        return False, len(moons), f'{len(moons)} keys vs {len(first)} integrations of assign_{tier}.csv (set or order differs)'
    for k, t in first.items():
        e = at(t)
        for b in _C().jupiter.BODIES[1:]:
            want = [round(e[f'{b}_dx_arcmin'], 3), round(e[f'{b}_dy_arcmin'], 3)]
            got = moons[k].get(b)
            if got is None or abs(got[0] - want[0]) > 5e-4 or abs(got[1] - want[1]) > 5e-4:
                bad += 1
    return bad == 0, len(moons), f'{len(moons)} integrations; {bad} moon offsets differ from the ephemeris at the first row time'


@content_test('jupiter_assign')
def assign_recomputed(path, **ctx):
    """assign_<tier>.csv: rows and order = the meas rows; meas columns copied (used/refused changed only by tier rule 4,
    phase3_tier_rules.flux_scale_failed in place); fit_* = the fit record; detected = converged and |snr| >= DET_SNR and (V or
    peak > 0); for detected rows sigma_arcmin = phase3_snapshot_astro.sigma_axis_arcmin(band or its wideband representative,
    |snr|, jup_el, fitted=True) floored at 1e-3 (TODAY's model: a product made before a ruled model change fails here, by
    band), separations and Rayleigh percentiles from the fit offsets and the ephemeris moon offsets, assignment / unclear_rank
    / reflection_candidate by the rule of :387-399. moons_<tier>.json: one entry per integration = the ephemeris offsets at its
    first row time (0.001 arcmin)."""
    if path.endswith('.json'):
        return _moons_test(path)
    J, M = _C().jupiter, _C().jupiter_measure
    _legacy()
    import phase3_snapshot_astro as astro
    from phase3_tier_rules import flux_scale_failed
    tier = _tier_of(path, 'assign')
    od = os.path.dirname(os.path.abspath(path))
    cell = _cell_of(path)
    lst, date = cell.split('_', 1)
    cols, rows = _read_csv(path)
    meas = _meas_rows(od, tier)
    fp = os.path.join(od, f'fits_{tier}.jsonl')
    if meas is None or not os.path.exists(fp):
        return False, None, f'meas_{tier}.csv or fits_{tier}.jsonl missing beside the assign file'
    fits_ = {}
    for line in open(fp):
        if line.strip():
            x = json.loads(line)
            fits_[x['crop']] = x
    if cols[:len(MEAS_COLUMNS)] != MEAS_COLUMNS or any('fit_' + k not in cols for k in FIT_COLS) or 'detected' not in cols:
        return False, len(rows), f'columns {cols[:25]}... lack the meas / fit_* / detected columns'
    if len(rows) != len(meas):
        return False, len(rows), f'{len(rows)} rows vs {len(meas)} meas rows'
    at = _ephem_at(date)
    bad = {k: 0 for k in ('identity', 'copied', 'rule4', 'fit fields', 'detected', 'sigma model', 'separation', 'assignment',
                          'non-detected filled')}
    sigma_bands = set()
    n_det = 0
    fsf_cache = {}
    for r, m in zip(rows, meas):
        if any(r[c] != m[c] for c in ('cell', 'tier', 'band', 'stokes', 'key', 'crop')):
            bad['identity'] += 1
            continue
        if any(r[c] != m[c] for c in MEAS_COLUMNS if c not in ('used', 'refused')):
            bad['copied'] += 1
        if r['band'] not in fsf_cache:
            fsf_cache[r['band']] = flux_scale_failed(lst, date, r['band'])
        if fsf_cache[r['band']]:
            want_ref = (m['refused'] + ';' if m['refused'] else '') + 'flux-scale'
            if r['used'] != '0' or r['refused'] != want_ref:
                bad['rule4'] += 1
        elif r['used'] != m['used'] or r['refused'] != m['refused']:
            bad['rule4'] += 1
        x = fits_.get(r['crop']) if r['crop'] else None
        for k in FIT_COLS:
            want = '' if x is None else ('' if k not in x else str(x[k]))
            if r['fit_' + k] != want:
                bad['fit fields'] += 1
                break
        ok = bool(x and x.get('converged') and abs(x.get('snr', 0)) >= M.DET_SNR and (r['stokes'] == 'V' or x.get('peak', 0) > 0))
        if r['detected'] != str(int(ok)):
            bad['detected'] += 1
            continue
        if not ok:
            if r.get('assignment') or r.get('sigma_arcmin'):
                bad['non-detected filled'] += 1
            continue
        n_det += 1
        e = at(float(r['t_unix']))
        sb = r['band'] if r['band'] in J.SUBBANDS else M.WIDE_REP_SUBBAND[r['band']]
        snr = abs(float(x['snr']))
        sig = max(astro.sigma_axis_arcmin(sb, snr, float(r['jup_el']), fitted=True), M.SIGMA_FLOOR_ARCMIN)
        exact = '%.2f' % sig == r.get('sigma_arcmin')
        if not exact:
            bad['sigma model'] += 1
            sigma_bands.add(r['band'])
            sig = _f(r.get('sigma_arcmin'))
        cons, amb = [], False
        for b in J.BODIES:
            bx = 0.0 if b == 'jupiter' else e[f'{b}_dx_arcmin']
            by = 0.0 if b == 'jupiter' else e[f'{b}_dy_arcmin']
            sep = math.hypot(x['dx_arcmin'] - bx, x['dy_arcmin'] - by)
            pct = 100.0 * (1.0 - math.exp(-0.5 * (sep / sig) ** 2))
            if '%.2f' % sep != r.get(f'sep_{b}_arcmin'):
                bad['separation'] += 1
            if exact and '%.1f' % pct != r.get(f'pct_{b}'):
                bad['separation'] += 1
            if not exact and abs(pct - M.ASSIGN_PCT) < 0.5:
                amb = True
            if pct <= M.ASSIGN_PCT:
                cons.append(b)
        if amb:
            continue
        moons = [b for b in cons if b != 'jupiter']
        cls = 'none' if not cons else (cons[0] if len(cons) == 1 else 'unclear:' + '+'.join(cons))
        rank = (str(M.UNCLEAR_RANK['moons_only'] if ('jupiter' not in cons and len(moons) > 1) else M.UNCLEAR_RANK['jupiter_and_moon'])
                if cls.startswith('unclear') else '')
        refl = str(int(bool(moons) and 'jupiter' not in cons))
        graz = all(r.get(f'{b}_grazing_deg') == '%.1f' % e.get(f'{b}_grazing_deg', float('nan')) for b in J.BODIES[1:])
        if r.get('assignment') != cls or r.get('unclear_rank') != rank or r.get('reflection_candidate') != refl or not graz:
            bad['assignment'] += 1
    nb = sum(bad.values())
    return nb == 0, len(rows), (f'{len(rows)} rows, {n_det} detected; problems: {_summary(bad)}'
                                + (f' (sigma model differs from today\'s in bands {sorted(sigma_bands)})' if sigma_bands else ''))


# ---------------------------------------------------------------------------------------------------------------------
# jupiter_cell (in place)
# ---------------------------------------------------------------------------------------------------------------------
def _verdict_lines(path):
    lines = [l.rstrip('\n') for l in open(path) if l.strip()]
    tests = [l for l in lines if re.match(r'^TEST[1-4] ', l)]
    return lines, tests


def _verdict_consistent(lines, tests):
    if len(tests) != 4 or [t[:5] for t in tests] != ['TEST1', 'TEST2', 'TEST3', 'TEST4']:
        return False, f'{len(tests)} TEST lines (want TEST1..TEST4 in order)'
    passes = [t.endswith('-> PASS') for t in tests]
    if not all(t.endswith('-> PASS') or t.endswith('-> FAIL') for t in tests):
        return False, 'a TEST line without its -> PASS / -> FAIL verdict'
    last = lines[-1]
    if last != ('ALL-PASS' if all(passes) else 'FAILED'):
        return False, f'last line {last!r} disagrees with the TEST verdicts {passes}'
    return True, ''


@content_test('jupiter_cell')
def tests_1_to_4(path, fresh_verdict=False, engine_kind=None, work=None, **ctx):
    """CONTENT_TEST.txt of a measured cell. Written by this unit's own test stage (fresh_verdict): its four TEST lines and the
    last line agree, and the verdict is ALL-PASS. Alone (verify, --resume): the engine's stage_test is run IN PLACE on a
    temporary directory of symlinks to the cell's files (TEST1 rebuild of 1 % of the 10-s rows from cube and Master, TEST2
    1 min vs 10 min, TEST3 positions = imfit records, TEST4 frames listed) with the inputs under jupiter.SUBLST_ROOT /
    CUBES_ROOT; PASS when that fresh verdict is ALL-PASS and the stored last line is ALL-PASS."""
    lines, tests = _verdict_lines(path)
    ok, why = _verdict_consistent(lines, tests)
    if not ok:
        return False, None, why
    if fresh_verdict:
        allp = lines[-1] == 'ALL-PASS'
        return allp and engine_kind in (None, 'ok'), lines[-1], ' | '.join(t[:90] for t in tests)
    return _cell_in_place(path, lines, tests, work)


def _cell_in_place(path, lines, tests, work):
    from exopipe.phase3.lib.jupiter import common as K
    from exopipe.phase3.lib.jupiter import mirrors
    _legacy()
    od = os.path.dirname(os.path.abspath(path))
    cell = os.path.basename(od)
    base = work
    if not base:
        from exopipe.phase3.core import fitsio
        try:
            base = fitsio.scratch_root(None)
        except Exception:      # noqa: BLE001
            base = tempfile.gettempdir()
    os.makedirs(base, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='jupiter_cell_test_', dir=base)
    try:
        td = os.path.join(tmp, cell)
        os.makedirs(td)
        for name in os.listdir(od):
            if name != 'CONTENT_TEST.txt':
                os.symlink(os.path.join(od, name), os.path.join(td, name))
        gl = mirrors.redirect_globals('measure', {'SCRATCH_DEFAULT': os.path.join(tmp, 'scratch')})
        res = K.in_place(None, mirrors.module_path('measure'), 'stage_test', args=[cell, tmp], globals_=gl,
                         python=K.interpreter_key('measure.test'), log=os.path.join(tmp, 'test.log'), cwd=os.path.join(tmp, 'cwd'),
                         guard=[tmp], test_stage=True)
        fresh = os.path.join(td, 'CONTENT_TEST.txt')
        if res.kind not in ('ok', 'test-failed') or not os.path.exists(fresh):
            tail = open(os.path.join(tmp, 'test.log')).read()[-400:] if os.path.exists(os.path.join(tmp, 'test.log')) else ''
            return False, None, f'in-place stage_test did not complete ({res.kind}: {res.reason}) {tail!r}'
        fl, ft = _verdict_lines(fresh)
        same = ft == tests
        ok = fl[-1] == 'ALL-PASS' and lines[-1] == 'ALL-PASS'
        return ok, fl[-1], (f'fresh: ' + ' | '.join(t[:100] for t in ft) + f'; stored lines {"identical" if same else "differ"}'
                            f'; SUBLST_ROOT {_C().jupiter.SUBLST_ROOT}')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------------
# jupiter_ephem, jupiter_window
# ---------------------------------------------------------------------------------------------------------------------
@content_test('jupiter_ephem')
def ephem_cache(path, **ctx):
    """1-min rows (monotonic, 60 s steps); every numeric column parses; angle columns in [0, 360); the rows cover the window
    hours of that date (jupiter.WINDOW_CSV utc_start..utc_end) and Jupiter's interpolated elevation at each window mid-hour
    is >= MIN_EL_DEG and equals the window's jup_el_mid within 0.15 deg."""
    import numpy as np
    from datetime import datetime, timezone
    m = re.match(r'^jupiter_ephem_(\d{4}-\d{2}-\d{2})\.csv$', os.path.basename(path))
    if not m:
        return False, None, 'not a jupiter_ephem_<date>.csv name'
    date = m.group(1)
    cols, rows = _read_csv(path)
    if not rows or 'utc' not in cols or 'jup_el' not in cols:
        return False, len(rows), 'no rows or no utc / jup_el column'

    def ts(s):
        return datetime.strptime(s[:19], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
    t = np.array([ts(r['utc']) for r in rows])
    steps = np.diff(t)
    bad_step = int(np.sum(np.abs(steps - 60.0) > 0.5))
    ang = [c for c in cols if c in ('jup_ra', 'cml_earth', 'io_phase_earth') or re.match(r'^(io|europa|ganymede|callisto)_(ra|cml|io_phase)$', c)]
    bad_num = bad_ang = 0
    for r in rows:
        for c in cols:
            if c in ('utc', 't_emit_utc'):
                continue
            v = _f(r[c])
            if r[c] not in ('nan', '') and not math.isfinite(v):
                bad_num += 1
            if c in ang and math.isfinite(v) and not 0.0 <= v < 360.0:
                bad_ang += 1
    el = np.array([_f(r['jup_el']) for r in rows])
    J = _C().jupiter
    bad_win, n_win = [], 0
    from exopipe.phase3.lib.jupiter import common as K
    for cell, w in K.window_rows():
        if w['date'] != date:
            continue
        n_win += 1
        a, b = ts(w['utc_start']), ts(w['utc_end'])
        if a < t[0] or b > t[-1]:
            bad_win.append(f'{cell} not covered')
            continue
        e = float(np.interp((a + b) / 2, t, el))
        if e < J.MIN_EL_DEG or abs(e - _f(w['jup_el_mid'])) > 0.15:
            bad_win.append(f'{cell} el {e:.2f} vs {w["jup_el_mid"]}')
    ok = not (bad_step or bad_num or bad_ang or bad_win)
    return ok, len(rows), (f'{len(rows)} rows; {bad_step} non-60-s steps, {bad_num} unparsable values, {bad_ang} angles outside '
                           f'[0, 360); {n_win} window cells on {date}: {"; ".join(bad_win[:4]) or "all covered, elevations agree"}')


@content_test('jupiter_window')
def window_rows(path, **ctx):
    """The Jupiter window (nights.csv: date,lst,utc_start,utc_end,jup_el_mid,...) or the backfill window (window.csv:
    lst,date,jup_el_mid): required columns, cells unique, LST/date grammar, every jup_el_mid >= MIN_EL_DEG (an explicit cell
    list - the old JFQ_CELLS - has every jup_el_mid blank)."""
    cols, rows = _read_csv(path)
    need = {'lst', 'date', 'jup_el_mid'}
    if not need <= set(cols):
        return False, len(rows), f'columns {cols} lack {sorted(need - set(cols))}'
    J = _C().jupiter
    seen, dup, bad_g, low, blank = set(), 0, 0, 0, 0
    for r in rows:
        c = (r['lst'], r['date'])
        dup += c in seen
        seen.add(c)
        if not re.match(r'^\d\dh$', r['lst']) or not re.match(r'^\d{4}-\d{2}-\d{2}$', r['date']):
            bad_g += 1
        if r['jup_el_mid'] == '':
            blank += 1
        elif _f(r['jup_el_mid']) < J.MIN_EL_DEG:
            low += 1
    explicit = blank == len(rows)
    ok = rows and not dup and not bad_g and (explicit or (low == 0 and blank == 0))
    return bool(ok), len(rows), (f'{len(rows)} cells; {dup} duplicates, {bad_g} bad LST/date, {low} below {J.MIN_EL_DEG:g} deg, '
                                 f'{blank} blank elevations' + (' (explicit cell list)' if explicit else ''))


# ---------------------------------------------------------------------------------------------------------------------
# flag backfill
# ---------------------------------------------------------------------------------------------------------------------
VERIFY_RE = re.compile(r"^VERIFY OK: (\d+) subband directories, 0 failures; (\d+) flagged-integration rows, (\d+) 'jupiter' rows, "
                       r"(\d+) copies, ([0-9.]+) GB$")


@content_test('jupiter_flag_verify')
def verify_ok_line(path, expected=None, **ctx):
    """The VERIFY file of the backfill's content test ends 'VERIFY OK: N subband directories, 0 failures; ...' with no FAIL
    line, and (when the verify unit passes them) its counts equal the counts that unit read from the tool's output."""
    lines = [l.rstrip('\n') for l in open(path) if l.strip()]
    fails = [l for l in lines if l.startswith('FAIL ')]
    last = next((l for l in reversed(lines) if l.startswith('VERIFY ')), '')
    m = VERIFY_RE.match(last)
    if not m or fails:
        return False, len(fails), f'last VERIFY line {last[:160]!r}; {len(fails)} FAIL lines'
    counts = dict(zip(('dirs', 'flagged_rows', 'jupiter_rows', 'copies', 'gb'), (int(m.group(1)), int(m.group(2)),
                                                                              int(m.group(3)), int(m.group(4)), float(m.group(5)))))
    if expected and any(expected.get(k) != v for k, v in counts.items() if k in expected):
        return False, None, f'counts {counts} differ from the verify unit {expected}'
    return True, counts['dirs'], last


@content_test('jupiter_flagged_copy', name='verify_tree')
def flagged_verify_tree(path, nproc=1, **ctx):
    """One copy directory <dest>/<LST>/<date>/<Run>/<sb>/ checked by the backfill tool's own content test, IN PLACE
    (exopipe/phase1/jupiter_flag_backfill.verify_tree on a temporary root holding only this directory): every copy's sha256
    = its manifest = the archive source; record rows and copies match; each 'jupiter' row re-measured from the copy (Jupiter
    detected within 1 beam at >= 10 sigma, Jupiter-removed statistic <= the night's threshold). The tool's sha256 must be the
    tested version (gate J30)."""
    from exopipe.phase3.lib.jupiter import common as K
    F = _C().jupiter_flag
    import exopipe.phase1.jupiter_flag_backfill as jfb
    import exopipe.phase1.jupiter_flag_qa as jfq
    for mod, want in ((jfb, F.TOOL_SHA256), (jfq, F.QA_SHA256)):
        if K.sha256(mod.__file__) != want:
            return False, None, f'{mod.__file__} is not the tested version (sha256 != the pinned value; gate J30)'
    d = os.path.abspath(path.rstrip('/'))
    if not glob.glob(os.path.join(d, jfq.RECORD_NAME.format(subband='*'))):
        return False, None, f'{d} holds no {jfq.RECORD_NAME}'
    parts = d.split(os.sep)
    tmp = tempfile.mkdtemp(prefix='jupiter_flag_verify_')
    try:
        link_parent = os.path.join(tmp, *parts[-4:-1])
        os.makedirs(link_parent)
        os.symlink(d, os.path.join(link_parent, parts[-1]))
        n, fails = jfb.verify_tree(tmp, F.PHASE1_ROOT, nproc=nproc)
        t = jfb.verify_tree.totals
        return (n == 1 and not fails), len(fails), (f'{n} directory; {len(fails)} failures'
                                                    + (f' ({fails[0][1][:120]})' if fails else '')
                                                    + f"; {t['flagged_rows']} flagged rows, {t['jupiter_rows']} jupiter rows, "
                                                      f"{t['copies']} copies")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------------
# served pages
# ---------------------------------------------------------------------------------------------------------------------
def _get(port, route, timeout):
    import urllib.request
    import urllib.error
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}{route}', timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b''


def integration_count(rows, tier, tol):
    """Integrations of a night page (exopipe_ops/jupiter_server.integrations after M33, 6b8a23a), recomputed: 10 s = groups of
    ALL bands' I and V rows, a group taking every row within tol seconds of its first row (rows in time order); 1 min / 10
    min = distinct (key, t_unix)."""
    if tier != '10s':
        return len({(r['key'], float(r['t_unix'])) for r in rows})
    n, g0 = 0, None
    for t in sorted(float(r['t_unix']) for r in rows):
        if g0 is None or t - g0 > tol:
            n, g0 = n + 1, t
    return n


def _png_ok(data):
    """A PNG that decodes (matplotlib's reader) to a non-empty image."""
    import io
    if not data.startswith(b'\x89PNG'):
        return False
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.image as mpimg
        a = mpimg.imread(io.BytesIO(data), format='png')
        return a.ndim >= 2 and a.shape[0] > 0 and a.shape[1] > 0
    except Exception:      # noqa: BLE001
        return False


@content_test('jupiter_served_page')
def served(path, **ctx):
    """The serve record (<out>/_served/jupiter_<port>.json: which, port, root). 8800 (phase1): GET / answers 200 and lists
    exactly the cells of the root whose CONTENT_TEST.txt holds ALL-PASS; each listed cell's night page answers 200 for every
    tier and its slider maximum = (number of integrations - 1) recomputed here from assign_<tier>.csv by the M33 rule (10 s:
    every band's I and V rows grouped within jupiter_serve.TOL_S of the group's first row, exopipe_ops 6b8a23a; 1 min /
    10 min: distinct (key, time)), with as many slider keys; /reflections answers 200 with one row link per reflection
    candidate (reflection_candidate 1) of the listed cells; for the first listed cell one panel PNG (first integration) and
    one Stokes I light-curve PNG per tier decode. 8801 (v2): every <cell>/index.html under the site root is served
    byte-identical; site.json's first root is the ruled root (jupiter_v2.RULED_ROOT)."""
    rec = json.load(open(path))
    S = _C().jupiter_serve
    port, root, which = rec['port'], rec['root'], rec['which']
    to = S.HTTP_TIMEOUT_S
    if which == 'stopped':
        from exopipe.phase3.lib.jupiter.serve import listeners
        ls = listeners(port)
        return not ls, len(ls), ('port %d free (server stopped)' % port if not ls else
                                 'port %d still held by pid(s) %s' % (port, [i['pid'] for i in ls]))
    if which == 'phase1':
        code, body = _get(port, '/', to)
        if code != 200:
            return False, code, f'GET / -> {code}'
        html = body.decode('utf-8', 'replace')
        listed = sorted(set(re.findall(r"<a href='/night/([^?']+)\?tier=10s'>", html)))
        want = sorted(d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, 'CONTENT_TEST.txt'))
                      and 'ALL-PASS' in open(os.path.join(root, d, 'CONTENT_TEST.txt')).read())
        if listed != want:
            return False, len(listed), f'index lists {len(listed)} cells, the root has {len(want)} ALL-PASS cells'
        bad = []
        tol = _C().jupiter_serve.TOL_S
        first_keys = {}
        for c in listed:
            for t in _C().jupiter.CADENCES:
                code, page = _get(port, f'/night/{c}?tier={t}', to)
                if code != 200:
                    bad.append(f'{c} {t} -> {code}')
                    continue
                page = page.decode('utf-8', 'replace')
                ap = os.path.join(root, c, f'assign_{t}.csv')
                rows = _read_csv(ap)[1] if os.path.exists(ap) else []
                n = integration_count(rows, t, tol[t])
                m = re.search(r"type='range' min='0' max='(\d+)'", page)
                k = re.search(r'const keys = (\[.*?\]);', page)
                keys = json.loads(k.group(1)) if k else None
                if not m or int(m.group(1)) != max(n - 1, 0) or keys is None or len(keys) != n:
                    bad.append(f'{c} {t}: slider max {m.group(1) if m else "?"}, {len(keys) if keys is not None else "?"} keys; '
                               f'{n} integrations by the M33 rule')
                elif keys:
                    first_keys[(c, t)] = keys[0]
        code, body = _get(port, '/reflections', to)
        n_ref = sum(int((r.get('reflection_candidate') or '0') == '1') for c in listed for t in _C().jupiter.CADENCES
                    for r in (_read_csv(os.path.join(root, c, f'assign_{t}.csv'))[1]
                              if os.path.exists(os.path.join(root, c, f'assign_{t}.csv')) else []))
        n_rows = body.decode('utf-8', 'replace').count("<tr><td><a href='/night/")
        if code != 200 or n_rows != n_ref:
            bad.append(f'/reflections -> {code}, {n_rows} rows vs {n_ref} reflection candidates')
        if listed:
            c = listed[0]
            for t in _C().jupiter.CADENCES:
                if (c, t) not in first_keys:
                    continue
                for route in (f'/panel/{c}/{t}/{first_keys[(c, t)]}.png', f'/lc/{c}/{t}/I.png'):
                    code, png = _get(port, route, to)
                    if code != 200 or not _png_ok(png):
                        bad.append(f'{route} -> {code}, not a decodable PNG')
        return not bad, len(listed), f'{len(listed)} cells served; problems: {"; ".join(bad[:4]) or "none"}'
    if which == 'v2':
        bad, n = [], 0
        for cdir in sorted(glob.glob(os.path.join(root, '*', 'index.html'))):
            c = os.path.basename(os.path.dirname(cdir))
            code, body = _get(port, f'/{c}/index.html', to)
            n += 1
            if code != 200 or body != open(cdir, 'rb').read():
                bad.append(f'{c}/index.html -> {code}')
            sj = os.path.join(os.path.dirname(cdir), 'site.json')
            if os.path.exists(sj):
                v = json.load(open(sj)).get('variants') or [{}]
                if v[0].get('label') != _C().jupiter_v2.RULED_ROOT:
                    bad.append(f'{c}/site.json first root {v[0].get("label")} != {_C().jupiter_v2.RULED_ROOT}')
        return n > 0 and not bad, n, f'{n} cell site(s); problems: {"; ".join(bad[:4]) or "none"}'
    return False, None, f'unknown served page kind {which!r}'
