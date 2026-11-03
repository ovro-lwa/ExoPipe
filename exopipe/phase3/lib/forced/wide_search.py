"""THE WIDE-AREA BURST SEARCH of family W (stage wide_search): every peak of |S/N| >= SNR_MIN, either sign, anywhere in a
forced target's gen5 cutouts, on EXACTLY the frames the forced photometry used. Port of phase3/refactor_analysis/
review_items_20261031/forced_wide/forced_wide_search.py (2026-11-02; Gregg 2026-11-01 "a wide search area that allows the
burst tapering to be identified", "Go"); its docstring is the method, the values are config (forced_wide_search):

For every (Stokes, target, band) of the jobs, every cutout of that target (one per LST-hour cell: a 300'-wide RESIDUAL cube -
frame minus hour median - on the cell's fixed grid, every 10-s frame with its admission flag):
  FRAMES   10 s: admitted (TIMES) and the forced evaluation's domain gate at the target (snapshot_forced_eval.gate_mask, in
           place: elevation >= 30 deg, A-team, the night rule, Sun / Jupiter); 1 min: gen5's blocks (admitted frames of one
           minute - MINUTE_RULE - >= BLOCK_MIN_FRAMES, averaged), gated the same way at the block time. A band excluded by
           data_decisions (core.exclusions) or refused by tier rule 3 for (LST, night) (phase3_tier_rules.deep_search_refused,
           in place) is skipped, as the evaluation drops it. The target's a-priori screen (zone / clean, snapshot_forced_eval.
           zone_flags) is recorded per file and tier: the forced queue lists CLEAN measurements only.
  PEAKS    local extrema of |value| over 3 x 3 pixels, EITHER SIGN, more than EDGE_PX inside the cutout; candidates at
           >= PRE_SIGMA x the plane's robust rms, then S/N = value / the LOCAL rms of the forced definition (1.4826 x MAD of the
           STAMP_PX stamp about the peak, its core excluded, >= STAMP_MIN_FRAC of it inside the cutout); kept at |S/N| >= SNR_MIN.
  POSITION a 2-D quadratic in log|value| over the 3 x 3 about the peak (xy term included); parabolic in value where a sample
           has the other sign; then the cutout's header WCS. A file is all or nothing: an error in either tier drops the whole
           file's peaks (FILE_ERRORS 'drop', production) or refuses the band (FILE_ERRORS 'refuse', the default).
Output per (Stokes, target, band): <S>_<identity>_<band>.npz with one row per peak (cell, tier, ts, x, y, ra, dec, snr, value,
rms) and a per-file table (f_cell, f_tier, f_planes, f_zone, f_status) and n_files - the analysis' members; its wall-clock
'seconds' is kept in the unit record instead, so the product is a function of its inputs only (byte-identical on a re-run).
The time caches (prefill_times) fill phase3_snapshot_search's Sun cache and an LST cache with one vectorized astropy call per
cutout - the scalar functions' own computations on a Time array (the analysis' test_gate_equiv.py: gate identical).
Everything here runs in the worker processes of the stage's pool: the in-place modules' caches are installed there only."""
import glob
import math
import os
import time

import numpy as np

NPZ_KEYS = ('cell', 'tier', 'ts', 'x', 'y', 'ra', 'dec', 'snr', 'value', 'rms', 'f_cell', 'f_tier', 'f_planes', 'f_zone',
            'f_status', 'n_files')
STATUSES = ('ok', 'no gated plane', 'excluded band', 'deep-search refused')     # + 'ERROR <type>: <text>' (FILE_ERRORS drop)
TIERS = ('10s', '1min')

_E = {}


def scfg():
    from exopipe.phase3.config import CFG
    return CFG.forced_wide_search


def eval_mod():
    """snapshot_forced_eval (gate_mask, zone_flags, noise_med), imported once per worker, with its per-frame time calls served
    from caches filled vectorized per cutout (prefill_times) and its per-call Master-catalogue and noise reads cached - the
    analysis' _eval_mod, installed in the worker process only. Records every noise / catalogue file it reads (inputs)."""
    if 'E' not in _E:
        from exopipe.phase3.lib.forced.wide_jobs import inplace
        E = inplace('snapshot_forced_eval')
        SS = inplace('phase3_snapshot_search')
        _E['E'] = E
        _E['SS'] = SS
        _E['LST'] = {}
        _E['GAUL'] = {}
        _E['NOISE'] = {}
        _E['READ'] = set()
        orig_lst = SS.lst_deg_at

        def lst_cached(epoch_seconds):
            v = _E['LST'].get(epoch_seconds)
            return v if v is not None else orig_lst(epoch_seconds)
        SS.lst_deg_at = lst_cached                      # gate_mask calls SS.lst_deg_at per frame
        orig_gaul = E.C.read_gaul

        def gaul_cached(path):
            if path not in _E['GAUL']:
                _E['GAUL'][path] = orig_gaul(path)
                _E['READ'].add(path)
            return _E['GAUL'][path].copy()
        E.C.read_gaul = gaul_cached                     # zone_flags reads the Master catalog per call
        orig_noise = E.noise_med

        def noise_cached(cell, sb, st, tier):
            k = (cell, sb, st, tier)
            if k not in _E['NOISE']:
                _E['NOISE'][k] = orig_noise(cell, sb, st, tier)
                p = os.path.join(E.PROTO, cell, 'noise_%s_%s_median%s.csv' % (sb, st, '_1min' if tier == '1min' else ''))
                if os.path.exists(p):
                    _E['READ'].add(p)
            return _E['NOISE'][k]
        E.noise_med = noise_cached
    return _E['E']


def prefill_times(ts_list):
    """fill phase3_snapshot_search's Sun cache (sun_state, keyed by the 15-character ts) and the LST cache (keyed by the epoch
    seconds gate_mask computes) for every timestamp, with ONE vectorized astropy call each - the scalar functions' own
    computations on a Time array (forced_wide_search.prefill_times, verbatim)"""
    SS = _E['SS']
    from astropy.time import Time
    from astropy.coordinates import get_sun, EarthLocation, AltAz, SkyCoord
    import astropy.units as u
    need = sorted({t[:15] for t in ts_list if t[:15] not in SS._SUN_CACHE})
    if need:
        lat, lon, height = scfg().SUN_SITE         # phase3_snapshot_search.sun_state's own location (MIRROR)
        loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.m)
        t = Time(['%s-%s-%sT%s:%s:%s' % (s[:4], s[4:6], s[6:8], s[9:11], s[11:13], s[13:15]) for s in need], scale='utc')
        sn = get_sun(t)
        sc = SkyCoord(sn.ra, sn.dec)
        alt = sc.transform_to(AltAz(obstime=t, location=loc)).alt.deg
        ha = ((t.sidereal_time('apparent', longitude=loc.lon).deg - sc.ra.deg + 180.0) % 360.0) - 180.0
        for s, a, h in zip(need, np.atleast_1d(alt), np.atleast_1d(ha)):
            br = 'evening' if h > 0 else 'morning'
            SS._SUN_CACHE[s] = (float(a), br, float(a) >= (SS.TWILIGHT_EVENING_DEG if br == 'evening' else SS.TWILIGHT_MORNING_DEG))
    secs = sorted({SS.ts_to_epoch_seconds(t) for t in ts_list} - set(_E['LST']))
    if secs:
        t = Time(np.array(secs, float), format='unix', scale='utc')
        v = t.sidereal_time('apparent', longitude=SS.OVRO_LON_DEG * u.deg).deg
        for s, x in zip(secs, np.atleast_1d(v)):
            _E['LST'][s] = float(x)


def stamp_rms(plane, x, y, c=None):
    """the forced definition's local rms at pixel (x, y): 1.4826 x MAD of the stamp, its central core excluded"""
    c = c or scfg()
    STAMP, CEN = c.STAMP_PX, c.STAMP_CORE_PX
    ny, nx = plane.shape
    h = STAMP // 2
    ya, yb, xa, xb = max(y - h, 0), min(y + h + 1, ny), max(x - h, 0), min(x + h + 1, nx)
    st = plane[ya:yb, xa:xb].copy()
    cy0, cx0 = y - ya, x - xa
    st[max(cy0 - CEN, 0):cy0 + CEN + 1, max(cx0 - CEN, 0):cx0 + CEN + 1] = np.nan
    v = st[np.isfinite(st)]
    if v.size < c.STAMP_MIN_FRAC * (STAMP * STAMP - (2 * CEN + 1) ** 2):
        return float('nan')
    m = np.median(v)
    return 1.4826 * float(np.median(np.abs(v - m)))


def interp3(fm, f0, fp, clip=0.5):
    """sub-pixel offset of a peak from three samples: Gaussian (log) where all are positive, else parabolic; clipped"""
    if fm > 0 and f0 > 0 and fp > 0:
        a, b, c = math.log(fm), math.log(f0), math.log(fp)
    else:
        a, b, c = fm, f0, fp
    den = a - 2 * b + c
    if den >= 0:
        return 0.0
    return max(-clip, min(clip, 0.5 * (a - c) / den))


_Q = []


def interp2d(b, c=None):
    """sub-pixel (dx, dy) of the peak of a 3 x 3 block b (centre = the peak pixel, sign made positive): least-squares quadratic
    a + bx + cy + dx^2 + ey^2 + fxy in log b (exact for a rotated elliptical Gaussian); the separate 1-D forms where a sample is
    not positive or the quadratic has no maximum within INTERP2D_CLIP_PX"""
    c = c or scfg()
    if not _Q:
        yy, xx = np.mgrid[-1:2, -1:2]
        X = np.stack([np.ones(9), xx.ravel(), yy.ravel(), xx.ravel() ** 2, yy.ravel() ** 2, (xx * yy).ravel()], 1)
        _Q.append(np.linalg.pinv(X))
    if np.all(b > 0):
        a0, bx, cy, dxx, eyy, fxy = _Q[0] @ np.log(b).ravel()
        H = np.array([[2 * dxx, fxy], [fxy, 2 * eyy]])
        if np.linalg.det(H) > 0 and H[0, 0] < 0:
            sx, sy = np.linalg.solve(H, [-bx, -cy])
            if abs(sx) <= c.INTERP2D_CLIP_PX and abs(sy) <= c.INTERP2D_CLIP_PX:
                return float(sx), float(sy)
    return (interp3(b[1, 0], b[1, 1], b[1, 2], c.INTERP1D_CLIP_PX),
            interp3(b[0, 1], b[1, 1], b[2, 1], c.INTERP1D_CLIP_PX))


def find_peaks(planes, c=None):
    """[(plane index, x, y, value, rms_local, snr, xs, ys)] of every local extremum of |value| at |S/N| >= SNR_MIN, either sign"""
    from scipy.ndimage import maximum_filter
    c = c or scfg()
    EDGE = c.EDGE_PX
    out = []
    A = np.abs(planes)                                  # both signs, both Stokes (module notes)
    A = np.where(np.isfinite(A), A, -np.inf)
    mx = maximum_filter(A, size=(1, 3, 3), mode='constant', cval=-np.inf)
    N, ny, nx = planes.shape
    for i in range(N):
        p = planes[i]
        fin = p[np.isfinite(p)]
        if fin.size < c.MIN_FINITE_PX:
            continue
        g = 1.4826 * float(np.median(np.abs(fin - np.median(fin))))
        if not g > 0:
            continue
        cand = np.argwhere((A[i] == mx[i]) & (A[i] >= c.PRE_SIGMA * g))
        for y, x in cand:
            if x < EDGE or y < EDGE or x >= nx - EDGE or y >= ny - EDGE:
                continue
            if not np.isfinite(p[y - 1:y + 2, x - 1:x + 2]).all():
                continue                    # beside blank pixels: not a confirmed maximum, no interpolation
            r = stamp_rms(p, int(x), int(y), c)
            if not r > 0:
                continue
            v = float(p[y, x])
            snr = v / r
            if abs(snr) < c.SNR_MIN:
                continue
            dx, dy = interp2d(p[y - 1:y + 2, x - 1:x + 2] * (1.0 if v > 0 else -1.0), c)
            out.append((i, int(x), int(y), v, r, snr, x + dx, y + dy))
    return out


def gates_10s(E, cell, band, st, tra, tde, ts, adm):
    idx = np.where(adm)[0]
    if not len(idx):
        return idx
    g = E.gate_mask(cell, band, st, np.array([tra]), np.array([tde]), [ts[i] for i in idx])[0]
    return idx[g]


def blocks_1min(cube, dobs, adm, ts_list, c=None):
    """gen5's 1-minute blocks (stage 1 as it ran for gen5): admitted frames by minute (MINUTE_RULE), >= BLOCK_MIN_FRAMES,
    averaged"""
    from collections import OrderedDict
    c = c or scfg()
    SS = _E['SS']
    mins = OrderedDict()
    for i in range(len(dobs)):
        if adm[i]:
            key = ts_list[i][:13] if c.MINUTE_RULE == 'filename' else SS.integration_minute_key(dobs[i])
            mins.setdefault(key, []).append(i)
    bk = [(k, ix) for k, ix in sorted(mins.items()) if len(ix) >= c.BLOCK_MIN_FRAMES]
    if not bk:
        return [], np.zeros((0,) + cube.shape[1:], np.float32)
    with np.errstate(all='ignore'):
        blk = np.stack([np.nanmean(cube[ix], axis=0) for _, ix in bk]).astype(np.float32)
    return [k + '00' for k, _ in bk], blk


def cutout_dir(root, job, band):
    for nm in (job['identity'], job['target'].replace(' ', '_')):
        d = os.path.join(root, job['sample'], band, nm)
        if os.path.isdir(d):
            return d
    return None


def cutout_files(root, job, band):
    """(directory, [cutout paths]) of a (job, band): the per-cell 10-s cutouts (forced_wide.CUTOUT_GLOB), sorted."""
    from exopipe.phase3.config import CFG
    d = cutout_dir(root, job, band)
    if d is None:
        return None, []
    return d, sorted(glob.glob(os.path.join(d, CFG.forced_wide.CUTOUT_GLOB.format(S=job['stokes']))))


# ---------------------------------------------------------------------------------------------------------------------
# the recorders' I/O guard (cooperative yield)
# ---------------------------------------------------------------------------------------------------------------------

def _guard_rounds(path, n=2):
    """The max node latency (s) of the last n guard rounds, latest first ([] when the log has none)."""
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 16384))
            lines = f.read().decode('utf-8', 'replace').splitlines()
    except OSError:
        return []
    out = []
    for ln in reversed(lines):
        if 'round:' not in ln:
            continue
        vals = []
        for tok in ln.split('round:', 1)[1].split():
            if '=' in tok and 's/' in tok:
                try:
                    vals.append(float(tok.split('=', 1)[1].split('s/', 1)[0]))
                except ValueError:
                    pass
        out.append(max(vals) if vals else 0.0)
        if len(out) >= n:
            break
    return out


def io_yield(c=None):
    """Wait while the recorders' I/O guard sees a slow node (forced_wide_search.IO_GUARD_*): returns the seconds waited."""
    c = c or scfg()
    if not c.IO_GUARD_LOG or not os.path.exists(c.IO_GUARD_LOG):
        return 0.0
    t0 = time.time()
    if t0 - _E.get('GUARD_T', 0.0) < c.IO_GUARD_POLL_S:       # the log is read at most once per poll interval
        return 0.0
    _E['GUARD_T'] = t0
    r = _guard_rounds(c.IO_GUARD_LOG)
    if not r or r[0] <= c.IO_GUARD_MAX_S:
        return 0.0
    while True:
        time.sleep(c.IO_GUARD_POLL_S)
        r = _guard_rounds(c.IO_GUARD_LOG)
        if len(r) >= 2 and r[0] <= c.IO_GUARD_MAX_S and r[1] <= c.IO_GUARD_MAX_S:
            return time.time() - t0


# ---------------------------------------------------------------------------------------------------------------------
# one (Stokes, target, band)
# ---------------------------------------------------------------------------------------------------------------------

def search_band(task):
    """One (Stokes, target, band): every cutout of the target in that band -> a staged npz. ``task`` = (job, band, cutout root,
    staged npz path). Returns a dict: path, n_files, n_peaks, statuses, errors, nights (searched), inputs (noise / catalogue
    files read), cutout_dir, listing (name, size, mtime_ns per cutout), seconds, waited."""
    job, band, root, part = task
    from astropy.io import fits
    from astropy.wcs import WCS
    from exopipe.phase3.core import exclusions as X
    from exopipe.phase3.lib.forced.wide_jobs import inplace
    c = scfg()
    E = eval_mod()
    deep_search_refused = inplace('phase3_tier_rules').deep_search_refused
    st = job['stokes']
    d, files = cutout_files(root, job, band)
    out = dict(path=part, cutout_dir=d, n_files=len(files), n_peaks=0, errors=[], nights=[], statuses={}, waited=0.0,
               listing=[])
    if d is None:
        out['missing'] = 'no cutout directory'
        return out
    for f in files:
        try:
            s_ = os.stat(f)
            out['listing'].append((os.path.basename(f), s_.st_size, s_.st_mtime_ns))
        except OSError:
            out['listing'].append((os.path.basename(f), None, None))
    rows, ftab = [], []
    t0 = time.time()
    nights = set()
    for f in files:
        out['waited'] += io_yield(c)
        try:
            with fits.open(f, memmap=False) as h:
                H = h[0].header
                cube = np.asarray(h[0].data, np.float32)
                T = h['TIMES'].data
            cell = H.get('CELL')
            lst, date = cell.split('_', 1)
            tra, tde = float(H.get('TARGRA', job['T'][0])), float(H.get('TARGDEC', job['T'][1]))   # stage 2's gate position
            if X.is_excluded(lst, band, date):
                ftab.append((cell, '-', 0, -1, 'excluded band'))
                continue
            if deep_search_refused(lst, date, band, st):
                ftab.append((cell, '-', 0, -1, 'deep-search refused'))
                continue
            ts = [t if isinstance(t, str) else t.decode() for t in T['ts']]
            dobs = [t if isinstance(t, str) else t.decode() for t in T['date_obs']]
            adm = np.asarray(T['admitted'], bool)
            w = WCS(H).celestial
            prefill_times([ts[i] for i in np.where(adm)[0]])
            frows, fft = [], []
            for tier in TIERS:
                if tier == '10s':
                    use = gates_10s(E, cell, band, st, tra, tde, ts, adm)
                    planes = cube[use]
                    keys = [ts[i] for i in use]
                else:
                    keys, planes = blocks_1min(cube, dobs, adm, ts, c)
                    if keys:
                        prefill_times(keys)
                        g = E.gate_mask(cell, band, st, np.array([tra]), np.array([tde]), keys)[0]
                        planes = planes[g]
                        keys = [k for k, ok in zip(keys, g) if ok]
                zf, zst = E.zone_flags(lst, band, st, tier, np.array([tra]), np.array([tde]), E.noise_med(cell, band, st, tier))
                zone = int(bool(zf[0])) if zst == 'ok' else -1
                if not len(keys):
                    fft.append((cell, tier, 0, zone, 'no gated plane'))
                    continue
                pk = find_peaks(planes, c)
                if pk:
                    xs = np.array([p[6] for p in pk])
                    ys = np.array([p[7] for p in pk])
                    ra, de = w.pixel_to_world_values(xs, ys)
                    for p, a, b in zip(pk, np.atleast_1d(ra), np.atleast_1d(de)):
                        frows.append((cell, tier, keys[p[0]], p[6], p[7], float(a) % 360.0, float(b), p[5], p[3], p[4]))
                fft.append((cell, tier, len(keys), zone, 'ok'))
            rows.extend(frows)
            ftab.extend(fft)            # all or nothing per file (module notes)
            nights.add(date[:10])
        except Exception as e:  # noqa: BLE001 - recorded per file; FILE_ERRORS decides (the stage)
            msg = f'ERROR {type(e).__name__}: {str(e)[:150]}'
            ftab.append((os.path.basename(f), '-', 0, -1, msg))
            out['errors'].append((os.path.basename(f), msg))
    write_npz(part, rows, ftab, len(files))
    out['n_peaks'] = len(rows)
    out['nights'] = sorted(nights)
    for t in ftab:
        k = t[4] if not t[4].startswith('ERROR') else 'ERROR'
        out['statuses'][k] = out['statuses'].get(k, 0) + 1
    out['inputs'] = sorted(_E['READ'])
    out['seconds'] = round(time.time() - t0, 1)
    return out


def write_npz(path, rows, ftab, n_files):
    """The npz of one (Stokes, target, band): the analysis' members and dtypes, in its order (without its wall-clock
    'seconds'); np.savez_compressed is deterministic (zip members dated 1980-01-01), so a re-run is byte-identical."""
    np.savez_compressed(path, cell=np.array([r[0] for r in rows], dtype='U16'), tier=np.array([r[1] for r in rows], dtype='U4'),
                        ts=np.array([r[2] for r in rows], dtype='U16'), x=np.array([r[3] for r in rows], float),
                        y=np.array([r[4] for r in rows], float), ra=np.array([r[5] for r in rows], float),
                        dec=np.array([r[6] for r in rows], float), snr=np.array([r[7] for r in rows], float),
                        value=np.array([r[8] for r in rows], float), rms=np.array([r[9] for r in rows], float),
                        f_cell=np.array([t[0] for t in ftab], dtype='U100'), f_tier=np.array([t[1] for t in ftab], dtype='U4'),
                        f_planes=np.array([t[2] for t in ftab], int), f_zone=np.array([t[3] for t in ftab], int),
                        f_status=np.array([t[4] for t in ftab], dtype='U200'), n_files=n_files)


def npz_name(job, band):
    return f"{job['stokes']}_{job['identity']}_{band}.npz"
