#!/usr/bin/env python3
"""candinv/imaging.py — image loading, WCS, source fitting, local SNR.

Fitting follows the project rule "use imfit/BDSF, never a hand-rolled fitter":
the elliptical-Gaussian stamp fit here is the SAME routine the Phase-3 pipeline
uses (phase3_catalog._gauss2d / phase3_pipeline._fit_gaussian_stamp), reused so
flux/size measurements are consistent across the pipeline.  Local SNR is the
validated V recipe: peak / min-sector MAD over a control annulus (NOT /sqrt(sum
G^2), which is beam-correlated).
"""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import curve_fit
from scipy.signal import resample

from . import config as C

FWHM2SIG = 2.0 * np.sqrt(2.0 * np.log(2.0))


# --------------------------------------------------------------------------
def load_image(path):
    """Return (data2d, header, wcs_celestial). Squeezes degenerate axes."""
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        data = np.squeeze(hdul[0].data).astype(float)
    wcs = WCS(hdr).celestial
    return data, hdr, wcs


def cdelt_deg(hdr):
    for k in ('CDELT2', 'CD2_2'):
        if k in hdr:
            return abs(float(hdr[k]))
    raise KeyError('no CDELT2/CD2_2 in header')


def world_to_pix(wcs, ra_deg, dec_deg):
    x, y = wcs.all_world2pix(ra_deg, dec_deg, 0)
    return float(x), float(y)


def pix_to_world(wcs, x, y):
    ra, dec = wcs.all_pix2world(x, y, 0)
    return float(ra), float(dec)


def beam_pix(hdr, fallback_deg=None):
    """Beam FWHM (major) in pixels from BMAJ, else from a fallback FWHM(deg)."""
    cd = cdelt_deg(hdr)
    if 'BMAJ' in hdr and np.isfinite(hdr['BMAJ']) and hdr['BMAJ'] > 0:
        return float(hdr['BMAJ']) / cd
    if fallback_deg is not None:
        return float(fallback_deg) / cd
    return None


# --------------------------------------------------------------------------
def _gauss2d(c, a, x0, y0, sx, sy, th, o):
    X, Y = c
    xr = (X - x0) * np.cos(th) + (Y - y0) * np.sin(th)
    yr = -(X - x0) * np.sin(th) + (Y - y0) * np.cos(th)
    return (a * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2)) + o).ravel()


def matched_filter_snr(data, x, y, bpix, ctrl_in=6.0, ctrl_out=10.0,
                       n_ang=60):
    """Forced photometry at the FIXED position (x, y) — the validated recipe.

    flux = least-squares amplitude of a beam-shaped (FWHM=bpix) matched filter
    at the fixed pixel (NOT a free fit — a free fit runs away on noise and
    invents detections). Significance = |flux| / scatter of the SAME matched
    filter sampled around a control annulus (NOT /sqrt(sum G^2), which is
    beam-correlated). Returns (flux, snr, noise) — all NaN if unmeasurable.
    """
    bpix = float(bpix)
    R = int(round(3 * bpix))
    sig = bpix / FWHM2SIG
    iy, ix = int(round(y)), int(round(x))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    K = np.exp(-0.5 * (xx * xx + yy * yy) / sig ** 2)
    K = K / (K * K).sum()                       # least-squares matched filter

    def mf(cx, cy):
        st = data[cy - R:cy + R + 1, cx - R:cx + R + 1]
        if st.shape != K.shape or not np.isfinite(st).all():
            return np.nan
        return float((st * K).sum())

    on = mf(ix, iy)
    if not np.isfinite(on):
        return float('nan'), float('nan'), float('nan')
    samp = []
    for ang in np.linspace(0, 2 * np.pi, n_ang, endpoint=False):
        for rr in (ctrl_in * bpix, 0.5 * (ctrl_in + ctrl_out) * bpix,
                   ctrl_out * bpix):
            v = mf(int(round(ix + rr * np.cos(ang))),
                   int(round(iy + rr * np.sin(ang))))
            if np.isfinite(v):
                samp.append(v)
    if len(samp) < 30:
        return on, float('nan'), float('nan')
    nz = _mad(np.array(samp))
    return on, (abs(on) / nz if nz > 0 else float('nan')), float(nz)


def fit_source(data, x, y, bpix, cd, r=None, upsample=2, force_sign=None):
    """BOUNDED elliptical-Gaussian fit on a stamp — for SHAPE/POSITION only.

    Use matched_filter_snr() for flux/SNR; use this only when you need the
    fitted size/axis-ratio/position (T1 morphology, position refinement). The
    bounds (amplitude clamped to the stamp, width 0.3-3x beam) keep the fit
    from running away on noise. Returns a dict or None.
    """
    bpix = float(bpix)
    if r is None:
        r = max(8, int(round(2 * bpix)))
    iy, ix = int(round(y)), int(round(x))
    st = data[iy - r:iy + r + 1, ix - r:ix + r + 1]
    if st.shape != (2 * r + 1, 2 * r + 1) or not np.isfinite(st).all():
        return None
    fine = resample(resample(st.astype(float), st.shape[0] * upsample, axis=0),
                    st.shape[1] * upsample, axis=1)
    n = fine.shape[0]
    coords = np.mgrid[0:n, 0:n].astype(float) / upsample
    yy, xx = coords[0] - r, coords[1] - r
    if force_sign is None:
        sgn = np.sign(np.nanmean(st[r - 1:r + 2, r - 1:r + 2])) or 1.0
    else:
        sgn = float(force_sign)
    amax = float(np.nanmax(np.abs(st)))
    amp0 = sgn * amax
    sig0 = bpix / FWHM2SIG
    smin, smax = max(0.3 * sig0, 0.5), 3.0 * sig0
    sig0 = min(max(sig0, smin), smax)
    lo = [-5 * amax - 1e-9, -r, -r, smin, smin, -np.pi, -2 * amax - 1e-9]
    hi = [5 * amax + 1e-9, r, r, smax, smax, np.pi, 2 * amax + 1e-9]
    try:
        p, _ = curve_fit(_gauss2d, (xx, yy), fine.ravel(),
                         p0=[amp0, 0.0, 0.0, sig0, sig0, 0.0, 0.0],
                         bounds=(lo, hi), maxfev=10000)
    except Exception:
        return None
    a, x0, y0, sx, sy, th, off = p
    sx, sy = abs(sx), abs(sy)
    if sx >= sy:
        s_maj, s_min, th_maj = sx, sy, th
    else:
        s_maj, s_min, th_maj = sy, sx, th + np.pi / 2.0
    pa_sky = (np.rad2deg(th_maj) - 90.0) % 180.0
    beam_area = (np.pi * bpix * bpix) / (4 * np.log(2))
    integ = a * 2 * np.pi * sx * sy / max(beam_area, 1e-9)
    return dict(
        peak=float(a), int_flux=float(integ), offset=float(off),
        x_fit=float(ix + x0), y_fit=float(iy + y0),
        dx_pix=float(x0), dy_pix=float(y0),
        fwhm_maj_deg=float(s_maj * FWHM2SIG * cd),
        fwhm_min_deg=float(s_min * FWHM2SIG * cd),
        fwhm_maj_pix=float(s_maj * FWHM2SIG),
        fwhm_min_pix=float(s_min * FWHM2SIG),
        pa_deg=float(pa_sky),
        axis_ratio=float((s_maj) / max(s_min, 1e-6)))


# --------------------------------------------------------------------------
def local_mad_snr(data, x, y, bpix, signed_peak):
    """SNR = |signed_peak| / (min-sector MAD over a 2-10 beam annulus).

    Replicates search_v_transients_v2.assign_local_snr: 4 sectors, take the
    most conservative (lowest) sector noise.  Returns (snr, local_rms).
    """
    bpix = float(bpix)
    box = int(np.ceil(C.SNR_OUTER_BEAMS * bpix))
    iy, ix = int(round(y)), int(round(x))
    y0, y1 = iy - box, iy + box + 1
    x0, x1 = ix - box, ix + box + 1
    if y0 < 0 or x0 < 0 or y1 > data.shape[0] or x1 > data.shape[1]:
        # clip the box to the image; annulus may be partial near the edge
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(data.shape[0], y1), min(data.shape[1], x1)
    sub = data[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
    dy, dx = yy - iy, xx - ix
    d2 = dx * dx + dy * dy
    ann = (d2 >= (C.SNR_INNER_BEAMS * bpix) ** 2) & \
          (d2 <= (C.SNR_OUTER_BEAMS * bpix) ** 2) & np.isfinite(sub) & (sub != 0)
    if ann.sum() < C.SNR_MIN_SECTOR_PIX:
        vals = sub[ann]
        if vals.size < 50:
            return float('nan'), float('nan')
        rms = _mad(vals)
        return abs(signed_peak) / max(rms, 1e-12), rms
    ang = np.arctan2(dx, dy) % (2 * np.pi)
    sec = (ang / (2 * np.pi / C.SNR_N_SECTORS)).astype(int) % C.SNR_N_SECTORS
    mads = []
    for s in range(C.SNR_N_SECTORS):
        m = ann & (sec == s)
        if m.sum() >= C.SNR_MIN_SECTOR_PIX:
            mads.append(_mad(sub[m]))
    if not mads:
        rms = _mad(sub[ann])
    else:
        rms = min(mads)
    return abs(signed_peak) / max(rms, 1e-12), float(rms)


def _mad(v):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float('nan')
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


def central_rms(data):
    """Robust rms over the inner quarter (matches the detection rms region)."""
    n = data.shape[0]
    c = data[n // 4:3 * n // 4, n // 4:3 * n // 4]
    c = c[np.isfinite(c) & (c != 0)]
    if c.size < 100:
        return float('nan')
    return float(np.std(c[np.abs(c) < np.percentile(np.abs(c), 99)]))


def count_peaks_above(data, bpix, nsigma, sign=0):
    """Count distinct >nsigma peaks (both signs unless sign=+/-1).

    Cheap context metric (# of >7sigma 'transients' in the discovery image):
    high count => the image is noisy/striated => the candidate is less trusted.
    """
    rms = central_rms(data)
    if not np.isfinite(rms) or rms <= 0:
        return -1, float('nan')
    work = np.where(np.isfinite(data), data, 0.0).copy()
    if sign > 0:
        sig = work / rms
    elif sign < 0:
        sig = -work / rms
    else:
        sig = np.abs(work) / rms
    thr = float(nsigma)
    npk = 0
    excl = max(6, int(round(bpix)))
    # greedy peak picking with beam-sized exclusion
    masked = sig.copy()
    for _ in range(5000):
        iy, ix = np.unravel_index(np.argmax(masked), masked.shape)
        if masked[iy, ix] < thr:
            break
        npk += 1
        masked[max(0, iy - excl):iy + excl + 1,
               max(0, ix - excl):ix + excl + 1] = 0.0
    return npk, rms
