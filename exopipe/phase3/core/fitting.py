"""Source fits shared by every tier: the free Gaussian, the template fit, and the FITTED POSITION of a detection or a
forced measurement, with its error (CLAUDE.md 5; PHASE3_ARCHITECTURE.md 5).

Rulings carried
- CLAUDE.md 5 (Gregg 2026-10-04): Gaussian fit (free size and shape) at or above 7 sigma; below 7 sigma a template fit
  (the empirical beam fixed, amplitude and position free). Positions carry their errors; the forced value stays the flux
  of record at a target position.
- RULING-RADIO-POSITION-ONLY (Gregg 2026-10-04): only the radio source position is quoted; none when there is none.
- R-GREGG-20261012-FITTED-DETECTION-POSITIONS-REQUIRED: every detection position is a fitted position, fitted on the
  SEARCHED frame ("(a) fitted positions are required of course!"); when no fit is admissible the detection is KEPT at its
  peak pixel with a pixel-quantisation term (nothing is removed here - a removal would be a source-removing filter).
- The fit kind is decided by the FITTED source's own S/N (Gregg 2026-10-11).
- Template K per tier and Stokes from ``CFG.fitting.K_TEMPLATE`` (deep I ratified, deep V adopted 2026-10-11; sub-LST
  measured K ruled 2026-10-13).

Moved from (behaviour-identical at ``CFG.fitting.TEMPLATE_BEAM_ORIENTATION = 'legacy-mirrored'``):
  exopipe_ops/deep_forced_radio.py (radio_position, detection_position, template_fit cutout, gaussian_at),
  clv15/phase3_template_fit.py (beam_icov_pix, beam_model, peak_start, fit, k_for),
  exopipe/phase3/phase3_catalog.py (gaussian_photometry, _gauss2d).
  exopipe/phase3/identify.py (fit_empirical_beam, appended verbatim at the end; refactor stage 3, 2026-10-13).
Differences: one observatory latitude (core/astrometry) instead of 37.2398 deg here and 37.2397 deg in the catalogue
builder (elevations change by < 1e-4 deg); SYS_SCALE from config instead of parsed from build_source_catalogs_v3.py.
"""
import math

import numpy as np
from scipy.optimize import curve_fit      # module level, as in identify.py (fit_empirical_beam)

from exopipe.phase3.core import astrometry as A

RADIO_COLS = ['pos_kind', 'pos_ra', 'pos_dec', 'pos_err_maj_arcsec', 'pos_err_min_arcsec', 'pos_err_pa_deg',
              'pos_stat_maj_arcsec', 'pos_stat_min_arcsec', 'pos_sys_arcsec', 'pos_snr', 'pos_offset_arcmin', 'pos_reason']
DET_COLS = RADIO_COLS + ['peak_ra', 'peak_dec']
S2F = 1.0 / math.sqrt(8.0 * math.log(2.0))


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


# =====================================================================================================================
# Free 2-D Gaussian (the catalogue's fitter)
# =====================================================================================================================

def _gauss2d(c, a, x0, y0, sx, sy, th, o):
    X, Y = c
    xr = (X - x0) * np.cos(th) + (Y - y0) * np.sin(th)
    yr = -(X - x0) * np.sin(th) + (Y - y0) * np.cos(th)
    return (a * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2)) + o).ravel()


def gaussian_photometry(resid, x, y, bpix, cd):
    """2-D Gaussian fit on a stamp around integer pixel (x, y). Returns peak (Jy), integrated flux (Jy, = peak * 2 pi sx
    sy / beam area), fitted offset (dx, dy px), FWHM major (arcmin) and its 1-sigma error, axis ratio; None on failure."""
    from scipy.optimize import curve_fit
    r = max(8, int(round(2 * bpix)))
    st = resid[y - r:y + r + 1, x - r:x + r + 1]
    if st.shape != (2 * r + 1, 2 * r + 1) or not np.isfinite(st).all():
        return None
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    sgn = np.sign(np.nanmean(st[r - 1:r + 2, r - 1:r + 2])) or 1.0
    try:
        p, _pcov = curve_fit(_gauss2d, (xx, yy), st.ravel(),
                             p0=[sgn * np.nanmax(np.abs(st)), 0, 0, bpix / 2.355, bpix / 2.355, 0, 0], maxfev=10000)
    except Exception:
        return None
    peak = float(p[0])
    sx, sy = abs(p[3]), abs(p[4])
    beam_area = (np.pi * bpix * bpix) / (4 * np.log(2))      # px^2
    integ = peak * 2 * np.pi * sx * sy / max(beam_area, 1e-9)
    mj, mn = max(sx, sy) * 2.355, min(sx, sy) * 2.355
    try:                                                      # Card 9 (2026-09-04): keep the fit covariance
        _imaj = 3 if sx >= sy else 4
        _emaj = float(np.sqrt(abs(_pcov[_imaj, _imaj]))) * 2.355 if np.all(np.isfinite(_pcov)) else float('nan')
    except Exception:
        _emaj = float('nan')
    return dict(peak_jy=round(peak, 4), int_jy=round(float(integ), 4), dx=float(p[1]), dy=float(p[2]),
                fwhm_maj_arcmin=round(mj * cd * 60, 2),
                fwhm_maj_err_arcmin=(round(_emaj * cd * 60, 3) if np.isfinite(_emaj) else None),
                axis_ratio=round(mj / max(mn, 0.1), 2))


# =====================================================================================================================
# Template fit (empirical beam fixed; amplitude and position free)
# =====================================================================================================================

def k_snapshot(band):
    """(K major, K minor) of the snapshot template fit for a band (69-82 MHz carry the larger major-axis K)."""
    c = _cfg().fitting
    try:
        f = float(str(band).replace('MHz', ''))
    except ValueError:
        f = 0.0
    return (c.K_SNAPSHOT_MAJOR_HIGH if f >= c.K_SNAPSHOT_HIGH_MHZ else c.K_SNAPSHOT[0]), c.K_SNAPSHOT[1]


def beam_icov_pix(bmaj_deg, bmin_deg, bpa_deg, wcs, orientation=None):
    """Inverse covariance of the beam Gaussian in PIXEL coordinates of a celestial wcs; PA north through east.
    orientation 'legacy-mirrored' = the production behaviour (pixel +x taken as east: the mirror image of the beam);
    'wcs' = the beam as the WCS maps it."""
    orientation = orientation or _cfg().fitting.TEMPLATE_BEAM_ORIENTATION
    pa = math.radians(float(bpa_deg))
    R = np.array([[math.sin(pa), math.cos(pa)], [math.cos(pa), -math.sin(pa)]])     # columns: major, minor (east, north)
    C = R @ np.diag([(float(bmaj_deg) * S2F) ** 2, (float(bmin_deg) * S2F) ** 2]) @ R.T
    cd = wcs.pixel_scale_matrix
    if orientation == 'legacy-mirrored':
        Jm = np.array([[-cd[0, 0], -cd[0, 1]], [cd[1, 0], cd[1, 1]]])
    elif orientation == 'wcs':
        Jm = np.array([[cd[0, 0], cd[0, 1]], [cd[1, 0], cd[1, 1]]])
    else:
        raise ValueError(f'fitting.TEMPLATE_BEAM_ORIENTATION must be legacy-mirrored or wcs, got {orientation!r}')
    Ji = np.linalg.inv(Jm)
    return np.linalg.inv(Ji @ C @ Ji.T)


def beam_model(icov, amp, x, y, X, Y):
    dx, dy = X - x, Y - y
    return amp * np.exp(-0.5 * (icov[0, 0] * dx * dx + 2.0 * icov[0, 1] * dx * dy + icov[1, 1] * dy * dy))


def peak_start(img, x0, y0, icov, rsearch, sign):
    """Extremum of `sign` of the beam-smoothed image within rsearch pixels of (x0, y0) (sub-pixel by a parabola)."""
    from scipy import ndimage
    ny, nx = img.shape
    h = int(math.ceil(3.0 / math.sqrt(min(np.linalg.eigvalsh(icov)))))
    yk, xk = np.mgrid[-h:h + 1, -h:h + 1]
    ker = beam_model(icov, 1.0, 0.0, 0.0, xk.astype(float), yk.astype(float))
    sm = ndimage.convolve(np.nan_to_num(img), ker / ker.sum(), mode='constant')
    yy, xx = np.mgrid[0:ny, 0:nx]
    m = np.hypot(xx - x0, yy - y0) <= rsearch
    if not m.any():
        return x0, y0
    v = np.where(m, sign * sm, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(v)), v.shape)

    def para(a, b, c):
        d = a - 2 * b + c
        return 0.0 if d == 0 else 0.5 * (a - c) / d
    dx = para(sm[iy, ix - 1], sm[iy, ix], sm[iy, ix + 1]) if 0 < ix < nx - 1 else 0.0
    dy = para(sm[iy - 1, ix], sm[iy, ix], sm[iy + 1, ix]) if 0 < iy < ny - 1 else 0.0
    return ix + max(-0.5, min(0.5, dx)), iy + max(-0.5, min(0.5, dy))


def template_fit(img, wcs, x0, y0, bmaj_deg, bmin_deg, bpa_deg, rms, k=None, start_amp=None, band=None,
                 win_beams=None, max_shift_beams=None, search_beams=None):
    """Template fit of one image (2-D, Jy/beam) from the start pixel (x0, y0). Starts at the extremum of the
    beam-smoothed image within search_beams of the start (not at the target: started at the target the fit was pulled
    0.11-0.33 beam toward it, injection validation 2026-10-04). Returns dict: status (ok | at-bound | sign-flip |
    few-pixels | no-convergence; only ok gives a position), amp, x, y, ra, dec, snr, err_maj_arcmin, err_min_arcmin,
    pa_deg, shift_beams. k = (K major, K minor); None = the snapshot K for the band."""
    from scipy import optimize
    c = _cfg().fitting
    win_beams = c.TEMPLATE_WIN_BEAMS if win_beams is None else win_beams
    max_shift_beams = c.TEMPLATE_MAX_SHIFT_BEAMS if max_shift_beams is None else max_shift_beams
    search_beams = c.TEMPLATE_SEARCH_BEAMS if search_beams is None else search_beams
    img = np.asarray(img, dtype=float)
    pix_deg = abs(float(wcs.pixel_scale_matrix[1, 1]))
    fwhm_pix = float(bmaj_deg) / pix_deg
    rwin = max(2.0, win_beams * fwhm_pix)
    rmax = max_shift_beams * fwhm_pix
    icov = beam_icov_pix(bmaj_deg, bmin_deg, bpa_deg, wcs)
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    out = dict(status='no-convergence', amp=np.nan, x=np.nan, y=np.nan, ra=np.nan, dec=np.nan, snr=np.nan,
               err_maj_arcmin=np.nan, err_min_arcmin=np.nan, pa_deg=float(bpa_deg), shift_beams=np.nan)
    if start_amp is None:
        ix, iy = int(round(x0)), int(round(y0))
        start_amp = float(img[iy, ix]) if (0 <= ix < nx and 0 <= iy < ny and np.isfinite(img[iy, ix])) else 1.0
    sgn = 1.0 if start_amp >= 0 else -1.0
    px0, py0 = peak_start(img, float(x0), float(y0), icov, search_beams * fwhm_pix, sgn)
    ipx, ipy = int(round(px0)), int(round(py0))
    amp = float(img[ipy, ipx]) if np.isfinite(img[ipy, ipx]) else float(start_amp)
    cx, cy = px0, py0
    for _ in range(20):
        m = (np.hypot(xx - cx, yy - cy) <= rwin) & np.isfinite(img)
        if m.sum() < 9:
            out['status'] = 'few-pixels'
            return out
        X, Y, Z = xx[m].astype(float), yy[m].astype(float), img[m]
        lo = [-np.inf, px0 - rmax, py0 - rmax]
        hi = [np.inf, px0 + rmax, py0 + rmax]
        p0 = [amp, min(max(cx, lo[1] + 1e-6), hi[1] - 1e-6), min(max(cy, lo[2] + 1e-6), hi[2] - 1e-6)]
        try:
            r = optimize.least_squares(lambda p: beam_model(icov, p[0], p[1], p[2], X, Y) - Z, p0, bounds=(lo, hi))
        except Exception:
            return out
        if not r.success:
            return out
        amp, nxp, nyp = [float(v) for v in r.x]
        moved = math.hypot(nxp - cx, nyp - cy)
        cx, cy = nxp, nyp
        if moved < 0.02:
            break
    shift = math.hypot(cx - px0, cy - py0)
    out.update(amp=amp, x=cx, y=cy, shift_beams=math.hypot(cx - x0, cy - y0) / fwhm_pix)
    ra, dec = wcs.pixel_to_world_values(cx, cy)
    out.update(ra=float(ra) % 360.0, dec=float(dec))
    snr = abs(amp) / float(rms) if rms and rms > 0 else np.nan
    out['snr'] = snr
    if shift >= rmax - 1e-3:
        out['status'] = 'at-bound'
    elif amp * sgn <= 0:
        out['status'] = 'sign-flip'
    else:
        out['status'] = 'ok'
        km, kn = k if k is not None else k_snapshot(band)
        out['err_maj_arcmin'] = km * float(bmaj_deg) * 60.0 / snr
        out['err_min_arcmin'] = kn * float(bmin_deg) * 60.0 / snr
    return out


# =====================================================================================================================
# Fitted position of a detection / forced measurement (deep and sub-LST: the ONE shared function)
# =====================================================================================================================

def k_template(tier, stokes):
    k = _cfg().fitting.K_TEMPLATE.get((tier, stokes))
    if k is None:
        from exopipe.phase3.core.record import Refusal
        raise Refusal(f'no template error scale K for tier {tier!r} Stokes {stokes!r} in config fitting.K_TEMPLATE')
    return k


def blank():
    return dict.fromkeys(RADIO_COLS, '')


def _template_on_cutout(d, w, hdr, ra, dec, value, rms, snr, band):
    """template_fit on a cutout centred on (ra, dec): half-size 3.5 beam FWHM + 2 px (the 1-beam peak search, its
    smoothing kernel and the 0.5-beam shift with the 0.8-beam window all lie inside)."""
    cd = abs(float(hdr['CDELT2']))
    bmaj, bmin, bpa = float(hdr['BMAJ']), float(hdr['BMIN']), float(hdr.get('BPA', 0.0))
    x, y = [float(v) for v in w.world_to_pixel_values(ra, dec)]
    H = int(math.ceil(3.5 * bmaj / cd)) + 2
    xi, yi = int(round(x)), int(round(y))
    y0, x0 = max(0, yi - H), max(0, xi - H)
    sub = np.asarray(d[y0:yi + H + 1, x0:xi + H + 1], dtype=float)
    ws = w[y0:yi + H + 1, x0:xi + H + 1]
    return template_fit(sub, ws, x - x0, y - y0, bmaj, bmin, bpa, rms, start_amp=(value if value != 0 else snr),
                        band=band)


def gaussian_at(d, w, hdr, ra, dec, sign_ref, rms):
    """The free Gaussian on the stamp at the pixel of (ra, dec). Returns dict(ra, dec, snr = |peak| / rms, why): why = ''
    when the fit converged, its centre lies inside its own stamp and its peak has the sign of sign_ref."""
    cd = abs(float(hdr['CDELT2']))
    bmaj = float(hdr['BMAJ'])
    x, y = [float(v) for v in w.world_to_pixel_values(ra, dec)]
    xi, yi = int(round(x)), int(round(y))
    fit = gaussian_photometry(d, xi, yi, bmaj / cd, cd)
    r_st = max(8, int(round(2 * bmaj / cd)))
    o = dict(ra=float('nan'), dec=float('nan'), snr=float('nan'), why='')
    if not fit:
        o['why'] = 'Gaussian fit failed'
    elif abs(fit['dx']) > r_st or abs(fit['dy']) > r_st:
        o['why'] = (f'Gaussian fit centre outside its own fit stamp ({fit["dx"]:+.1f}, {fit["dy"]:+.1f} px; stamp '
                    f'half-size {r_st} px)')
    elif fit['peak_jy'] * sign_ref <= 0:
        o['why'] = 'Gaussian fit peak of the opposite sign (another feature)'
    else:
        pra, pdec = [float(v) for v in w.pixel_to_world_values(xi + fit['dx'], yi + fit['dy'])]
        o.update(ra=pra % 360.0, dec=pdec, snr=abs(float(fit['peak_jy'])) / float(rms))
        if not (math.isfinite(pra) and math.isfinite(pdec) and math.isfinite(o['snr'])):
            o['why'] = 'Gaussian fit position or S/N not finite'
    return o


def radio_position(d, w, hdr, ra, dec, value, rms, snr, band, stokes, lst_deg, k=None, tier='deep'):
    """The RADIO POSITION record (RADIO_COLS) of one measurement at (ra, dec) on frame d (2-D) with celestial WCS w and
    header hdr; value, rms, snr = the measurement there; lst_deg = the frame's LST at MJD-MID; k = (K major, K minor) of
    the template error (None = config fitting.K_TEMPLATE[(tier, stokes)])."""
    c = _cfg().fitting
    out = blank()
    bmaj, bmin, bpa = float(hdr['BMAJ']), float(hdr['BMIN']), float(hdr.get('BPA', 0.0))
    if not (abs(snr) >= c.RADIO_SNR_FLOOR):
        out.update(pos_kind='none', pos_reason=f'|SNR| below {c.RADIO_SNR_FLOOR:g}: not fitted')
        return out
    why_g = ''
    G = c.GAUSS_SNR
    if abs(snr) >= G:
        g = gaussian_at(d, w, hdr, ra, dec, value, rms)
        if g['why']:
            out.update(pos_kind='none', pos_reason=g['why'])
            return out
        if g['snr'] >= G:
            pra, pdec, psnr = g['ra'], g['dec'], g['snr']
            smaj = smin = A.stat_arcmin(bmaj * 60.0, psnr) * 60.0
            kind, pa = 'gaussian', ''
            reason = f"gaussian: fitted S/N {psnr:.2f} >= {G:g} (the Gaussian at the target; forced S/N {snr:.2f})"
        else:
            why_g = f"the Gaussian at the target has S/N {g['snr']:.2f} < {G:g} (forced S/N {snr:.2f}); "
    if not (abs(snr) >= G) or why_g:
        km, kn = k_template(tier, stokes) if k is None else k
        t = _template_on_cutout(d, w, hdr, ra, dec, value, rms, snr, band)
        if t['status'] != 'ok':
            out.update(pos_kind='none', pos_reason=f"{why_g}template fit {t['status']}")
            return out
        tra, tdec, ts = float(t['ra']) % 360.0, float(t['dec']), float(t['snr'])
        if ts >= G:
            g = gaussian_at(d, w, hdr, tra, tdec, t['amp'], rms)
            if not g['why']:
                pra, pdec, psnr = g['ra'], g['dec'], ts
                smaj = smin = A.stat_arcmin(bmaj * 60.0, psnr) * 60.0
                kind, pa = 'gaussian', ''
                reason = (f"gaussian: {why_g}fitted S/N {ts:.2f} >= {G:g} (template S/N; the Gaussian at the template "
                          f"position, its own S/N {g['snr']:.2f}; forced S/N {snr:.2f})")
            else:
                pra, pdec, psnr = tra, tdec, ts
                smaj, smin = km * bmaj * 3600.0 / psnr, kn * bmin * 3600.0 / psnr
                kind, pa = 'template', f'{bpa:.1f}'
                reason = (f"template kept: {why_g}fitted S/N {ts:.2f} >= {G:g} but the Gaussian at the template position "
                          f"failed ({g['why']}); forced S/N {snr:.2f}")
        else:
            pra, pdec, psnr = tra, tdec, ts
            smaj, smin = km * bmaj * 3600.0 / psnr, kn * bmin * 3600.0 / psnr
            kind, pa = 'template', f'{bpa:.1f}'
            reason = f"template: {why_g}fitted S/N {ts:.2f} < {G:g} (forced S/N {snr:.2f})"
    if not (math.isfinite(pra) and math.isfinite(pdec) and math.isfinite(smaj) and math.isfinite(smin)):
        out.update(pos_kind='none', pos_reason=f'{kind} fit position or error not finite')
        return out
    pra %= 360.0
    sy = A.sys_arcmin(band, A.elevation_deg(pra, pdec, lst_deg)) * 60.0 * A.sys_scale(stokes, band)
    out.update(pos_kind=kind, pos_ra=f'{pra:.5f}', pos_dec=f'{pdec:.5f}', pos_err_maj_arcsec=f'{math.hypot(smaj, sy):.1f}',
               pos_err_min_arcsec=f'{math.hypot(smin, sy):.1f}', pos_err_pa_deg=pa, pos_stat_maj_arcsec=f'{smaj:.1f}',
               pos_stat_min_arcsec=f'{smin:.1f}', pos_sys_arcsec=f'{sy:.1f}', pos_snr=f'{psnr:.2f}',
               pos_offset_arcmin=f'{A.sep_arcmin(pra, pdec, ra, dec):.2f}', pos_reason=reason)
    return out


def detection_position(d, w, hdr, ra, dec, value, rms, snr, band, stokes, lst_deg, k=None, tier='deep'):
    """The FITTED POSITION of one blind detection (R-GREGG-20261012-FITTED-DETECTION-POSITIONS-REQUIRED): radio_position
    started at the detection's peak-pixel position (ra, dec) on its SEARCHED frame. When no fit is admissible the
    detection is KEPT at its peak pixel: pos_kind 'peak' with a pixel-quantisation term (pixel / sqrt(12) per axis) in
    quadrature. Adds peak_ra / peak_dec (DET_COLS)."""
    out = radio_position(d, w, hdr, ra, dec, value, rms, snr, band, stokes, lst_deg, k=k, tier=tier)
    out['peak_ra'], out['peak_dec'] = f'{ra % 360.0:.5f}', f'{dec:.5f}'
    if out['pos_kind'] in ('gaussian', 'template'):
        return out
    bmaj, cd = float(hdr['BMAJ']), abs(float(hdr['CDELT2']))
    s = abs(float(snr))
    st = A.stat_arcmin(bmaj * 60.0, s) * 60.0
    q = cd * 3600.0 / math.sqrt(12.0)
    sy = A.sys_arcmin(band, A.elevation_deg(ra % 360.0, dec, lst_deg)) * 60.0 * A.sys_scale(stokes, band)
    stq = math.hypot(st, q)
    e = math.hypot(stq, sy)
    out.update(pos_kind='peak', pos_ra=f'{ra % 360.0:.5f}', pos_dec=f'{dec:.5f}', pos_err_maj_arcsec=f'{e:.1f}',
               pos_err_min_arcsec=f'{e:.1f}', pos_err_pa_deg='', pos_stat_maj_arcsec=f'{stq:.1f}',
               pos_stat_min_arcsec=f'{stq:.1f}', pos_sys_arcsec=f'{sy:.1f}', pos_snr=f'{s:.2f}', pos_offset_arcmin='0.00',
               pos_reason=f"peak: no admissible fit ({out['pos_reason']}); pixel-quantisation term {q:.1f} arcsec per axis")
    return out


# =====================================================================================================================
# Empirical beam measurement - absorbed VERBATIM from exopipe/phase3/identify.py (refactor stage 3, 2026-10-13), itself a
# verbatim extraction (2026-08-11) of fit_empirical_beam from clv15/phase3_final_metric.py. A different measurement from
# the fits above (the median Gaussian FWHM of bright unresolved sources in an image), kept as its own function.
# =====================================================================================================================

def fit_empirical_beam(img, nsrc=60, fmin=4.0, fmax=60.0):
    """Median Gaussian FWHM (pixels) of bright unresolved sources."""
    n = img.shape[0]
    q = n // 4
    w = np.where(np.isfinite(img), img, -np.inf).copy()
    w[:q, :] = w[3 * q:, :] = -np.inf
    w[:, :q] = w[:, 3 * q:] = -np.inf
    majs, mins_, pas = [], [], []
    tried = 0
    while len(majs) < nsrc and tried < 300:
        iy, ix = np.unravel_index(np.argmax(w), w.shape)
        v = w[iy, ix]
        if v < fmin:
            break
        w[iy - 12:iy + 12, ix - 12:ix + 12] = -np.inf
        tried += 1
        if v > fmax:
            continue
        r = 7
        st = img[iy - r:iy + r + 1, ix - r:ix + r + 1]
        if st.shape != (2 * r + 1, 2 * r + 1) or not np.isfinite(st).all():
            continue
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]

        def g(c, a, x0, y0, sx, sy, th, o):
            X, Y = c
            xr = (X - x0) * np.cos(th) + (Y - y0) * np.sin(th)
            yr = -(X - x0) * np.sin(th) + (Y - y0) * np.cos(th)
            return (a * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
                    + o).ravel()
        try:
            p, _ = curve_fit(g, (xx, yy), st.ravel(),
                             p0=[v, 0, 0, 2.2, 2.2, 0.8, 0], maxfev=8000)
        except Exception:
            continue
        sx, sy = abs(p[3]), abs(p[4])
        if not (0.8 < sx < 8 and 0.8 < sy < 8):
            continue
        majs.append(max(sx, sy) * 2.3548)
        mins_.append(min(sx, sy) * 2.3548)
        pas.append(np.rad2deg(p[5]) % 180)

    # Keep UNRESOLVED sources only (per Gregg): clip fits whose area
    # exceeds 1.25x the median fitted area — resolved sources bias the
    # beam estimate broad. One clipping pass is sufficient with a
    # median-based reference.
    majs = np.array(majs); mins_ = np.array(mins_); pas = np.array(pas)
    if len(majs) > 10:
        area = majs * mins_
        keep = area <= 1.25 * np.median(area)
        majs, mins_, pas = majs[keep], mins_[keep], pas[keep]
    return (float(np.median(majs)), float(np.median(mins_)),
            float(np.median(pas)), len(majs))

