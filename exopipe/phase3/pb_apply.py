#!/usr/bin/env python
"""
pb_apply.py -- derive deep/wideband/Full PBs from the 90-map base library and apply
them to Phase-3 deep co-add masters (Stokes I).

Base library (build_pb_library.py): PB_{subband}_bin{k}.fits, 15 subbands x 6 bins,
all on a COMMON canonical sky frame (same T_REF CRVAL) so cross-band reprojection is
well defined.

Derivations:
  deep subband PB   = nanmean over the 6 bins for that subband
  band PB (R/G/B)   = inverse-variance-weighted sum of the band's 5 subband PBs
                      (same grid -> direct)
  Full PB           = inverse-variance-weighted sum over all 15 subband PBs,
                      Red/Green reprojected onto the Full (=Blue) grid
Weights = Sigma over the master's constituent nights of the per-subband `weight`
(=1/rms^2) column in that night's QA thermal_noise CSV (category=deep). Exact.

Floor: correct only where PB > 0.1 (below -> NaN).  See MOSAIC_DESIGN.md.
"""
import os
import re
import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

from exopipe.common import phase2_config as cfg

PB_DIR = os.environ.get("P3_PB_LIBRARY", "/lustre/pipeline/exopipe/phase3/pb_library")   # canonical library on Lustre (2026-09-08; node-local copies verified identical, never a production default)
# RESIDUAL PRIMARY BEAM (Gregg 2026-09-07): one additional multiplicative step in the PB correction of
# every master variant and residual. The library entry is a zenith-centred SIN map whose header carries the
# analytic fit (PBRF* cards); the gain is evaluated here on the TARGET grid from those coefficients, so any
# grid (1508/2358/3122 px) gets the exact same function. Achromatic (bandwidth smearing: dnu/nu x
# theta/theta_beam is frequency independent), so one entry serves every subband. Set to '' to disable
# (never in production; comparison runs only).
RESIDUAL_PB = os.environ.get('P3_RESIDUAL_PB', '/lustre/pipeline/exopipe/phase3/pb_library/residual_pb/residual_pb_v1.fits')
_RESID_CACHE = {}

# ─────────────────────────────────────────────────────────────────────────────
# THREE-FACTOR PRIMARY-BEAM CORRECTION (Gregg 2026-09-08, verbatim: "we will have the model
# PB, the correction for chromatic aberration, and a true residual. I think it's important to
# have the three separated."; landed 2026-09-09, contract step 22). PB_effective =
#   (1) MODEL PB from the library                                   -> PBMODEL
#   (2) CHROMATIC (bandwidth-smearing) term, a priori: the data are averaged over
#       CHAN_AVG_HZ = 4 x 24 kHz = 96 kHz; for a square bandpass and a Gaussian beam the
#       peak of a source at angular distance theta from the phase centre is reduced by
#       I/I0 = sqrt(pi)/(2 k) erf(k),  k = sqrt(ln 2) (dnu/nu) (theta/theta_HPBW)
#       (Bridle & Schwab, Synthesis Imaging II, eq. 18-24)              -> PBCHROM
#   (3) TRUE RESIDUAL per band, measured by Gregg's pair test with (1)x(2) applied, as a
#       gain exp(slope_b x ZA) from residual_pb_v2_perband.csv               -> PBRESID
# residual_pb_v1 (one lumped map fitted at 73 MHz) is SUPERSEDED; P3_PB_THREE_FACTOR=0
# restores it for comparison runs only, never production. A band absent from the per-band
# residual table REFUSES (no silent fallback).
# ─────────────────────────────────────────────────────────────────────────────
CHAN_AVG_HZ = float(os.environ.get('P3_CHAN_AVG_HZ', '96000'))
CHROM_VERSION = 'bs-sqbp-v1'
THREE_FACTOR = os.environ.get('P3_PB_THREE_FACTOR', '1') != '0'
RESIDUAL_V2_TABLE = os.environ.get('P3_RESIDUAL_PB_V2', '/lustre/pipeline/exopipe/phase3/pb_library/residual_pb/residual_pb_v2_perband.csv')
_CHROM_CACHE = {}


def _grid_theta_deg(hdr, shape):
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    l = np.radians((xx + 1 - hdr['CRPIX1']) * hdr['CDELT1'])
    m = np.radians((yy + 1 - hdr['CRPIX2']) * hdr['CDELT2'])
    rr = np.hypot(l, m)
    return np.degrees(np.arcsin(np.clip(rr, 0, 1))), rr <= 1.0


def chromatic_gain(hdr, shape, freq_hz, bmaj_deg, dnu_hz=None):
    """Factor (2): bandwidth-smearing peak attenuation on the zenith-centred grid of hdr for a
    subband at freq_hz imaged with beam FWHM bmaj_deg. Returns (gain float32, version)."""
    from scipy.special import erf
    dnu = CHAN_AVG_HZ if dnu_hz is None else float(dnu_hz)
    key = (shape, float(hdr['CRPIX1']), float(hdr['CRPIX2']), float(hdr['CDELT1']), float(hdr['CDELT2']),
           round(float(freq_hz)), round(float(bmaj_deg), 6), dnu)
    if key in _CHROM_CACHE:
        return _CHROM_CACHE[key]
    theta, above = _grid_theta_deg(hdr, shape)
    k = np.sqrt(np.log(2.0)) * (dnu / float(freq_hz)) * (theta / float(bmaj_deg))
    g = np.where(k > 1e-6, np.sqrt(np.pi) / (2.0 * np.maximum(k, 1e-6)) * erf(k), 1.0).astype(np.float32)
    g[~above] = np.nan
    _CHROM_CACHE[key] = (g, CHROM_VERSION)
    return g, CHROM_VERSION


_RESID2 = None


def residual_band_slope(band):
    """Factor (3) table: per-band slope d ln g / d ZA (per degree) measured with (1)x(2) applied."""
    global _RESID2
    if _RESID2 is None:
        if not os.path.exists(RESIDUAL_V2_TABLE):
            raise RuntimeError(f'three-factor PB: per-band residual table missing ({RESIDUAL_V2_TABLE}); REFUSING')
        import csv as _csv
        _RESID2 = {r['band']: r for r in _csv.DictReader(open(RESIDUAL_V2_TABLE))}
    if band not in _RESID2:
        raise RuntimeError(f'three-factor PB: no residual row for {band} in {RESIDUAL_V2_TABLE}; REFUSING')
    return float(_RESID2[band]['slope_per_deg']), str(_RESID2[band].get('version', 'residual_pb_v2'))


def residual_band_gain(hdr, shape, band):
    theta, above = _grid_theta_deg(hdr, shape)
    s, ver = residual_band_slope(band)
    g = np.exp(s * theta).astype(np.float32)
    g[~above] = np.nan
    return g, f'{ver}:{band}'


def residual_gain(hdr, shape, path=RESIDUAL_PB):
    """g(ZA, az) of the residual primary beam on the grid described by hdr (a zenith-centred SIN header:
    CRPIX1/2, CDELT1/2). Convention verified against astropy WCS on a master: intermediate x (deg) = +RA =
    east, y = north; l = x rad, m = y rad; ZA = asin(hypot(l, m)); az = atan2(l, m) from north through east.
    Returns (gain float32 array, version string); gain is NaN below the horizon."""
    if not path:
        return None, ''
    key = (path, shape, float(hdr['CRPIX1']), float(hdr['CRPIX2']), float(hdr['CDELT1']), float(hdr['CDELT2']))
    if key in _RESID_CACHE:
        return _RESID_CACHE[key]
    h = fits.getheader(path)
    c = np.array([h[f'PBRF{k}'] for k in ('C0', 'C2', 'C4', 'A1', 'B1', 'A2', 'B2', 'A3', 'B3')], float)
    c0 = float(h['PBRNORM']); zmax = float(h['PBRZMAX']); ver = str(h['PBRESID'])
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    l = np.radians((xx + 1 - hdr['CRPIX1']) * hdr['CDELT1'])
    m = np.radians((yy + 1 - hdr['CRPIX2']) * hdr['CDELT2'])
    rr = np.hypot(l, m); above = rr <= 1.0
    zd = np.minimum(np.degrees(np.arcsin(np.clip(rr, 0, 1))), zmax); a = np.arctan2(l, m)
    g = (c[0] + c[1] * zd ** 2 + c[2] * zd ** 4 + zd * (c[3] * np.cos(a) + c[4] * np.sin(a))
         + zd ** 2 * (c[5] * np.cos(a) + c[6] * np.sin(a)) + zd ** 2 * (c[7] * np.cos(2 * a) + c[8] * np.sin(2 * a))) / c0
    g = np.clip(g, 0.5, 1.2).astype(np.float32); g[~above] = np.nan
    _RESID_CACHE[key] = (g, ver)
    return g, ver
QA_ROOT = "/lustre/pipeline/exopipe/phase2"
PB_FLOOR = 0.1

_FREQ_TO_LABEL = {float(int(l[:-3])): l for l in cfg.ALL_SUBBANDS}  # 18.0 -> '18MHz'


# ----------------------------------------------------------------------
# base-library access
# ----------------------------------------------------------------------
def _bin_path(subband, k, pb_dir):
    return os.path.join(pb_dir, f"PB_{subband}_bin{k}.fits")


_DEEP_CACHE = {}          # subband -> (deep_pb, header)
_FULL_REPROJ_CACHE = {}   # subband -> deep_pb reprojected onto the Full grid


def _deep_path(subband, pb_dir):
    return os.path.join(pb_dir, f"PB_{subband}_deep.fits")


def subband_deep_pb(subband, pb_dir=PB_DIR):
    """Deep (60-min) subband PB (data, header). Cached.

    Reads the stored 60-min map `PB_{subband}_deep.fits` (built directly over the full
    hour by build_pb_library.build_deep). Falls back to the equal-weight mean of the six
    10-min bins if the deep map is absent (numerically identical, for backward compat)."""
    if subband in _DEEP_CACHE:
        return _DEEP_CACHE[subband]
    deep_path = _deep_path(subband, pb_dir)
    if os.path.exists(deep_path):
        with fits.open(deep_path) as h:
            deep = h[0].data.astype(np.float32)
            hdr = h[0].header.copy()
    else:                                    # fallback: mean of the 6 bins
        stack = []
        hdr = None
        for k in range(6):
            with fits.open(_bin_path(subband, k, pb_dir)) as h:
                stack.append(h[0].data.astype(np.float32))
                if hdr is None:
                    hdr = h[0].header.copy()
        with np.errstate(invalid='ignore'):
            deep = np.nanmean(np.stack(stack, axis=0), axis=0).astype(np.float32)
    _DEEP_CACHE[subband] = (deep, hdr)
    return deep, hdr


def subband_bin_pb(subband, k, pb_dir=PB_DIR):
    with fits.open(_bin_path(subband, k, pb_dir)) as h:
        return h[0].data.astype(np.float32), h[0].header.copy()


# ----------------------------------------------------------------------
# ARBITRARY-WINDOW MODEL PB (Gregg 2026-09-23: "create a custom PB for every problematic image and divide it by
# the expected PB for that sub-LST"). The library holds the six nominal 10-minute bins; a Phase-1 10-minute image
# covers whatever window WSClean -intervals-out 6 gave it (on 29% of nights not a sidereal bin). This evaluates the
# library's OWN construction (tools/build_pb_library.py: the snapshot beam of OVRO-LWA_MROsoil_updatedheight.h5,
# sampled at sub-times through the window, below 10 deg elevation -> 0, averaged) for any window [t0, t1] in
# minutes from the start of the LST hour, on the library's pixel grid. Sub-time spacing = the library's (10/3 min),
# so the nominal window of bin k reproduces PB_{sb}_bin{k}.fits. MODEL factor only: the chromatic and residual
# factors are static functions of the pixel for a fixed phase centre and cancel in a window/bin ratio.
# ----------------------------------------------------------------------
_WIN_CACHE = {}
_BEAM_INTERP = []
_GRID_CACHE = {}
WIN_STEP_MIN = 10.0 / 3.0


def _beam_interp():
    if not _BEAM_INTERP:
        from exopipe.phase2.extractor_pb_75 import BeamModel, BEAM_PATH
        b = BeamModel(BEAM_PATH); b.load_data(); _BEAM_INTERP.append(b.interpolator)
    return _BEAM_INTERP[0]


def window_model_pb(subband, t0_min, t1_min, pb_dir=PB_DIR, quantum_min=0.5):
    """Model PB of `subband` time-averaged over minutes [t0_min, t1_min] of the LST hour (0 = hour start), on the
    library grid. Window ends are rounded to `quantum_min` for caching. Returns (map float32, header)."""
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, AltAz
    import astropy.units as u
    from exopipe.common.pipeline_config import OVRO_LOC
    q0, q1 = round(t0_min / quantum_min) * quantum_min, round(t1_min / quantum_min) * quantum_min
    if q1 <= q0:
        q1 = q0 + quantum_min
    key = (subband, q0, q1)
    if key in _WIN_CACHE:
        return _WIN_CACHE[key]
    _, hdr = subband_bin_pb(subband, 0, pb_dir)
    # NODE-LOCAL DISK CACHE (2026-09-23): the map depends only on (subband, window), never on the cell or night, so a
    # window is computed once per node (about 50-100 s per sub-time) and reused by every cell and worker. Re-derivable
    # scratch -> node-local /fast, never Lustre (CLAUDE.md s7). P3_PB_WINDOW_CACHE='' disables it.
    cdir = os.environ.get('P3_PB_WINDOW_CACHE', '/fast/gh/main/phase3/pb_window_cache')
    nsub = max(1, int(round((q1 - q0) / WIN_STEP_MIN)))
    cfile = f'{cdir}/{subband}_{q0:05.1f}_{q1:05.1f}_n{nsub}_{os.path.basename(os.path.dirname(os.path.abspath(pb_dir)))}.npy' if cdir else None
    if cfile and os.path.exists(cfile):
        pb = np.load(cfile)
        if pb.shape == (hdr['NAXIS2'], hdr['NAXIS1']):
            h = hdr.copy(); h['PBLSTBIN'] = (-2, 'arbitrary window (see PBWIN0/PBWIN1)')
            h['PBWIN0'] = (q0, '[min] window start from the LST hour start'); h['PBWIN1'] = (q1, '[min] window end')
            h['PBNSUB'] = (nsub, 'Sub-times averaged within this window')
            _WIN_CACHE[key] = (pb, h)
            return pb, h
    gk = (hdr['NAXIS1'], hdr['CDELT1'], hdr['CRPIX1'])
    if gk not in _GRID_CACHE:
        yy, xx = np.indices((hdr['NAXIS2'], hdr['NAXIS1']))
        ra, dec = WCS(hdr).all_pix2world(xx, yy, 0)
        _GRID_CACHE[gk] = SkyCoord(ra.ravel(), dec.ravel(), unit='deg')
    sc = _GRID_CACHE[gk]
    nsub = max(1, int(round((q1 - q0) / WIN_STEP_MIN)))
    tref = Time(hdr.get('PBTREF', '2025-01-01T00:00:00'), location=OVRO_LOC)
    freq = float(cfg.SUBBAND_FREQ_HZ[subband]['center'])
    interp = _beam_interp()
    acc = np.zeros(sc.size, np.float64)
    for j in range(nsub):
        tau_h = (q0 + (j + 0.5) * (q1 - q0) / nsub - 30.0) / 60.0      # hours from the hour midpoint
        aa = sc.transform_to(AltAz(obstime=tref + tau_h * u.hour, location=OVRO_LOC))
        el = aa.alt.deg
        resp = np.asarray(interp(np.column_stack((np.full(el.size, freq), 90.0 - el, aa.az.deg % 360.0))), np.float64)
        resp[~(el >= 10.0)] = 0.0
        acc += resp
    pb = (acc / nsub).reshape((hdr['NAXIS2'], hdr['NAXIS1'])).astype(np.float32)
    if cfile:
        os.makedirs(cdir, exist_ok=True)
        tmpf = f'{cfile}.{os.getpid()}.npy'
        np.save(tmpf, pb); os.replace(tmpf, cfile)
    h = hdr.copy(); h['PBLSTBIN'] = (-2, 'arbitrary window (see PBWIN0/PBWIN1)')
    h['PBWIN0'] = (q0, '[min] window start from the LST hour start'); h['PBWIN1'] = (q1, '[min] window end')
    h['PBNSUB'] = (nsub, 'Sub-times averaged within this window')
    _WIN_CACHE[key] = (pb, h)
    return pb, h


def window_scope_pb(scope, lst, master_path, t0_min, t1_min, pb_dir=PB_DIR, category='deep'):
    """Model PB of a subband or a wideband scope (Red/Green/Blue) over an arbitrary window, combined over the
    member subbands with the SAME inverse-variance weights wideband_pb uses. Model factor only."""
    if scope in cfg.ALL_SUBBANDS:
        return window_model_pb(scope, t0_min, t1_min, pb_dir)[0]
    if scope == 'Full':
        raise RuntimeError('window_scope_pb: Full is not supported (cross-grid); REFUSING')
    weights, _u, _m = subband_weights(lst, constituent_dates(master_path), category=category)
    acc, ws = None, 0.0
    for sb in cfg.SUBBAND_BANDS[scope]:
        w = weights.get(sb, 0.0)
        if w <= 0:
            continue
        m = window_model_pb(sb, t0_min, t1_min, pb_dir)[0]
        acc = np.zeros(m.shape, np.float64) if acc is None else acc
        acc += w * np.nan_to_num(m, nan=0.0); ws += w
    if acc is None:
        raise RuntimeError(f'window_scope_pb: no weighted member of {scope} for {lst}; REFUSING')
    return (acc / ws).astype(np.float32)


# ----------------------------------------------------------------------
# weights from constituent-night QA CSVs
# ----------------------------------------------------------------------
def constituent_dates(master_path):
    """Parse 'COADD constituent YYYY-MM-DD' HISTORY lines from a master."""
    hdr = fits.getheader(master_path)
    hist = "\n".join(str(c) for c in hdr.get('HISTORY', []))
    return sorted(set(re.findall(r'constituent\s+(20\d\d-\d\d-\d\d)', hist)))


def _find_qa_csv(lst, date):
    pat = os.path.join(QA_ROOT, lst, date, 'Science_*', 'QA',
                       f'{date}_{lst}_thermal_noise.csv')
    hits = sorted(glob.glob(pat))
    return hits[-1] if hits else None


def subband_weights(lst, dates, category='deep'):
    """Sum per-subband inverse-variance weight over constituent nights.

    Returns (weights: dict label->float, used_dates, missing_dates).
    """
    import csv
    weights = {l: 0.0 for l in cfg.ALL_SUBBANDS}
    used, missing = [], []
    for date in dates:
        p = _find_qa_csv(lst, date)
        if not p:
            missing.append(date)
            continue
        used.append(date)
        with open(p) as f:
            for row in csv.DictReader(f):
                if row.get('category') != category:
                    continue
                label = _FREQ_TO_LABEL.get(float(row['freq']))
                w = float(row['weight'])
                if label and np.isfinite(w):
                    weights[label] += w
    return weights, used, missing


# ----------------------------------------------------------------------
# wideband / Full PB assembly
# ----------------------------------------------------------------------
def _canonical_full_wcs(pb_dir=PB_DIR):
    """Target WCS for the Full/Blue grid (3122^2, canonical CRVAL)."""
    _, hdr = subband_bin_pb('73MHz', 0, pb_dir)   # a Blue subband = Full geometry
    return WCS(hdr), hdr


def wideband_pb(scope, lst, master_path, pb_dir=PB_DIR, category='deep', binidx=None):
    """Effective PB for scope in {'Red','Green','Blue','Full', or a subband label}.

    binidx=None -> deep (mean of bins); binidx=k -> that 10-min sub-LST bin.
    Returns (pb_map, meta) where meta records weights/dates used.
    """
    dates = constituent_dates(master_path)
    weights, used, missing = subband_weights(lst, dates, category=category)

    if scope in cfg.ALL_SUBBANDS:
        members = [scope]
    elif scope == 'Full':
        members = list(cfg.ALL_SUBBANDS)
    else:
        members = list(cfg.SUBBAND_BANDS[scope])

    def _pb_of(sb):
        return (subband_deep_pb(sb, pb_dir) if binidx is None
                else subband_bin_pb(sb, binidx, pb_dir))

    if scope == 'Full':
        # combine on the Full (=Blue) grid; reproject Red/Green up.
        tgt_wcs, tgt_hdr = _canonical_full_wcs(pb_dir)
        shape = (tgt_hdr['NAXIS2'], tgt_hdr['NAXIS1'])
        acc = np.zeros(shape, np.float64)
        wsum = 0.0
        for sb in members:
            w = weights.get(sb, 0.0)
            if w <= 0:
                continue
            if binidx is None and sb in _FULL_REPROJ_CACHE:
                grid = _FULL_REPROJ_CACHE[sb]
            else:
                data, hdr = _pb_of(sb)
                if (hdr['NAXIS1'], hdr['NAXIS2']) == (tgt_hdr['NAXIS1'], tgt_hdr['NAXIS2']):
                    grid = data
                else:
                    grid, _ = reproject_interp((data, WCS(hdr)), tgt_wcs,
                                               shape_out=shape, order='bilinear')
                if binidx is None:
                    _FULL_REPROJ_CACHE[sb] = grid
            acc += w * np.nan_to_num(grid, nan=0.0)
            wsum += w
        pb = (acc / wsum).astype(np.float32) if wsum > 0 else acc.astype(np.float32)
    else:
        # single grid (subband or one band) -> direct weighted sum
        acc, wsum, ref = None, 0.0, None
        for sb in members:
            w = weights.get(sb, 0.0)
            if w <= 0:
                continue
            data, hdr = _pb_of(sb)
            if acc is None:
                acc = np.zeros_like(data, np.float64)
                ref = hdr
            acc += w * np.nan_to_num(data, nan=0.0)
            wsum += w
        pb = (acc / wsum).astype(np.float32) if (acc is not None and wsum > 0) else None

    meta = dict(scope=scope, lst=lst, cadence=('deep' if binidx is None else f'bin{binidx}'),
                n_dates=len(used), n_missing=len(missing), members=members,
                weights={k: weights[k] for k in members},
                used_dates=used, missing_dates=missing, residual_pb='', chromatic='', pb_model='v1-2026-07-04')
    if pb is not None and THREE_FACTOR:
        # factors (2) and (3), weighted over the member subbands exactly as the model PB is
        _ghdr = tgt_hdr if scope == 'Full' else ref
        try:
            _mh = fits.getheader(master_path)
            _bmaj = float(_mh['BMAJ']); _nu0 = float(_mh.get('RESTFRQ') or 0.0)
        except Exception as _e:
            raise RuntimeError(f'three-factor PB: master beam/RESTFRQ unreadable for {master_path}: {_e}; REFUSING')
        _acc = np.zeros(pb.shape, np.float64); _racc = np.zeros(pb.shape, np.float64); _ws = 0.0
        for sb in members:
            w = weights.get(sb, 0.0)
            if w <= 0:
                continue
            _nu = float(cfg.SUBBAND_FREQ_HZ[sb]['center'])
            _bm = _bmaj * (_nu0 / _nu) if _nu0 > 0 else _bmaj      # beam scales as 1/nu across a band's members
            _c, _cv = chromatic_gain(_ghdr, pb.shape, _nu, _bm)
            _r, _rv = residual_band_gain(_ghdr, pb.shape, sb)
            _acc += w * np.nan_to_num(_c, nan=1.0); _racc += w * np.nan_to_num(_r, nan=1.0); _ws += w
        if _ws > 0:
            pb = (pb * (_acc / _ws) * (_racc / _ws)).astype(np.float32)
            meta['chromatic'] = CHROM_VERSION
            meta['residual_pb'] = 'residual_pb_v2-perband'
    elif pb is not None and RESIDUAL_PB:
        # SUPERSEDED single lumped step (residual_pb_v1); comparison runs only
        _ghdr = tgt_hdr if scope == 'Full' else ref
        g, ver = residual_gain(_ghdr, pb.shape)
        if g is not None:
            pb = (pb * np.where(np.isfinite(g), g, 1.0)).astype(np.float32)
            meta['residual_pb'] = ver
    return pb, meta


# ----------------------------------------------------------------------
# apply
# ----------------------------------------------------------------------
def apply_pbcorr(master_path, pb_map, out_path, floor=PB_FLOOR, meta=None):
    """Divide master by pb_map where pb>floor (else NaN); write _pbcorr.fits."""
    with fits.open(master_path) as h:
        data = h[0].data.astype(np.float32)
        hdr = h[0].header.copy()
    img = data.squeeze()
    if img.shape != pb_map.shape:
        raise ValueError(f"shape mismatch {img.shape} vs PB {pb_map.shape}")
    valid = np.isfinite(pb_map) & (pb_map > floor)
    out = np.full_like(img, np.nan)
    out[valid] = img[valid] / pb_map[valid]
    hdr['PBCORR'] = (True, 'OVRO-LWA PB library correction applied')
    # the sibling must NOT inherit the master's PRODNAME (archive name collision): drop it; the
    # header-standard pass rebuilds it from the file name with the pbcorr token (2026-09-09)
    if 'PRODNAME' in hdr:
        _pn = str(hdr['PRODNAME'])
        hdr['PRODNAME'] = (_pn.replace('_dewarped_', '_pbcorr_dewarped_', 1) if '_dewarped_' in _pn and '_pbcorr_' not in _pn else _pn,
                           'Phase 3 product name (grammar; pbcorr sibling)')
    hdr['PBFLOOR'] = (floor, 'PB floor below which pixels set NaN')
    hdr['PBLIBV'] = ('v1-2026-07-04', 'PB library version')
    hdr['PBMODEL'] = ((meta or {}).get('pb_model', 'v1-2026-07-04'), 'factor 1: model PB library version')
    if meta and meta.get('chromatic'):
        hdr['PBCHROM'] = (meta['chromatic'], f'factor 2: bandwidth smearing, {CHAN_AVG_HZ:.0f} Hz channels')
    if meta and meta.get('residual_pb'):
        hdr['PBRESID'] = (meta['residual_pb'], 'factor 3: measured residual gain (x model x chromatic)')
    if meta:
        hdr['PBNDATE'] = (meta.get('n_dates', -1), 'Constituent nights with QA weights')
        if meta.get('n_missing'):
            hdr['PBMISS'] = (meta['n_missing'], 'Constituent nights missing QA CSV')
    fits.writeto(out_path, out.astype(np.float32), hdr, overwrite=True)
    return out_path
