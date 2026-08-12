#!/usr/bin/env python3
"""candinv/photometry.py — forced photometry via CASA imfit on a SMALL cutout.

THE method for this project (Gregg, standing rule): flux/shape from CASA imfit,
never a hand-rolled / unbounded Gaussian fit (a free curve_fit runs away on
noise and INVENTS detections — it once reported a -1094 Jy / 1680-sigma "source"
on a pure-thermal-noise snapshot frame). imfit is run on a small cutout (not the
full NaN-blanked all-sky frame, which triggers CASA's uncatchable SIGABRT on
blank regions). Significance = imfit peak / central-MAD (NOT /sqrt(sum G^2),
which is beam-correlated). Pattern mirrors flux_check_cutout.py.
"""
import os
import tempfile

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std
import astropy.units as u

try:
    from casatasks import imfit
    _CASA = True
except Exception:
    _CASA = False


def _shape_deg(comp):
    """Pull observed (convolved) FWHM maj/min [deg] + PA [deg] from an imfit
    component, handling the arcsec/arcmin/deg unit on the axes."""
    try:
        sh = comp['shape']
        def _to_deg(d):
            v, un = float(d['value']), d.get('unit', 'arcsec')
            return v / 3600.0 if un == 'arcsec' else (v / 60.0 if un == 'arcmin'
                                                      else v)
        maj = _to_deg(sh['majoraxis'])
        mn = _to_deg(sh['minoraxis'])
        pa = float(sh['positionangle']['value'])
        return maj, mn, pa
    except Exception:
        return np.nan, np.nan, np.nan


def forced_imfit(path, ra, dec, box_half_arcmin=None,
                 fallback_bpix=6.0, beam=None):
    """Forced photometry at (ra, dec): CASA imfit within a search box sized by
    the ASTROMETRIC ERROR (Gregg — all photometry is imfit, in an area defined
    by the astrometric error). Returns a dict {peak, err, snr, rms, ra, dec,
    sign, converged, fwhm_maj_deg, fwhm_min_deg, pa_deg, axis_ratio,
    box_half_pix} or None if unmeasurable.

    box_half_arcmin: half-size of the imfit search box (from T2 astrometry);
        floored at 1.5 beams so imfit has support, capped to stay local.
    peak: imfit Gaussian peak [Jy/beam] (matched-filter fallback only when there
        is no beam / no CASA / imfit fails to converge). snr = |peak|/central-MAD.
    """
    try:
        d = np.squeeze(fits.getdata(path)).astype(float)
        hdr = fits.getheader(path)
    except Exception:
        return None
    w = WCS(hdr).celestial
    try:
        cd = abs(float(hdr['CDELT2']))
    except KeyError:
        cd = abs(float(hdr.get('CD2_2', 0))) or None
    if not cd:
        return None
    # Beam: header BMAJ/BMIN/BPA, else a caller-supplied override (e.g. the
    # empirical I beam, since bundle _clean residuals carry no beam) so imfit
    # can still run.
    bm = None
    if float(hdr.get('BMAJ', 0) or 0) > 0:
        bm = (float(hdr['BMAJ']), float(hdr.get('BMIN', hdr['BMAJ'])),
              float(hdr.get('BPA', 0.0)))
    elif beam is not None:
        bm = (float(beam[0]), float(beam[1]), float(beam[2]))
    has_beam = bm is not None
    bpix = (bm[0] / cd) if has_beam else float(fallback_bpix)
    # search box from the astrometric error (>=1.5 beams for imfit support,
    # capped so it stays local). Cutout leaves room for the noise annulus.
    if box_half_arcmin and box_half_arcmin > 0:
        R = int(round(max(1.5 * bpix, (box_half_arcmin / 60.0) / cd)))
    else:
        R = max(3, int(round(2.5 * bpix)))
    R = min(R, 40)
    cutout_pix = 2 * R + 41
    sc = SkyCoord(ra * u.deg, dec * u.deg)
    try:
        cut = Cutout2D(d, sc, (cutout_pix, cutout_pix), wcs=w,
                       mode='partial', fill_value=np.nan)
    except Exception:
        return None
    cx, cy = [float(v) for v in cut.wcs.world_to_pixel(sc)]
    ix, iy = int(round(cx)), int(round(cy))
    reg = cut.data[max(0, iy - R):iy + R + 1, max(0, ix - R):ix + R + 1]
    # CASA aborts (SIGABRT, uncatchable) on fully-blank regions — pre-check.
    if reg.size == 0 or not (np.isfinite(reg).any() and
                             np.nanmax(np.abs(reg)) > 0):
        return None
    # central-MAD noise from an off-source annulus OUTSIDE the search box
    yy, xx = np.mgrid[0:cut.data.shape[0], 0:cut.data.shape[1]]
    rr = np.hypot(xx - cx, yy - cy)
    ann = (rr > (R + bpix)) & np.isfinite(cut.data)
    rms = float(mad_std(cut.data[ann])) if ann.sum() > 50 else np.nan
    if not np.isfinite(rms) or rms <= 0:
        return None

    # ALL photometry uses imfit (in the astrometric-error box). The
    # matched-filter is only a fallback when there is no beam / no CASA / imfit
    # fails to converge — never the quoted value for a measurable source.
    imfit_ran = bool(_CASA and has_beam)
    flux = err = fra = fdec = np.nan
    fmaj = fmin = fpa = ra_err = dec_err = np.nan
    conv = False
    if imfit_ran:
        tmp = tempfile.mktemp(suffix='.fits', dir='/dev/shm'
                              if os.path.isdir('/dev/shm') else None)
        nh = cut.wcs.to_header()
        nh['BMAJ'], nh['BMIN'], nh['BPA'] = bm[0], bm[1], bm[2]
        if 'BUNIT' in hdr:
            nh['BUNIT'] = hdr['BUNIT']
        try:
            fits.writeto(tmp, cut.data.astype(np.float32), nh, overwrite=True)
            box = f'{ix - R},{iy - R},{ix + R},{iy + R}'
            fit = imfit(imagename=tmp, box=box)
            if fit and fit.get('converged') and \
                    fit.get('results', {}).get('component0'):
                c = fit['results']['component0']
                flux = float(c['peak']['value'])
                err = float(c['peak']['error'])
                fra = np.degrees(c['shape']['direction']['m0']['value'])
                fdec = np.degrees(c['shape']['direction']['m1']['value'])
                fmaj, fmin, fpa = _shape_deg(c)
                # statistical positional error RETURNED BY imfit (do not compute):
                # shape.direction.error = {longitude: RA err, latitude: Dec err}
                de = c['shape']['direction'].get('error', {})
                _re = de.get('longitude', {}).get('value')   # arcsec, RA
                _dce = de.get('latitude', {}).get('value')   # arcsec, Dec
                if _re is not None:
                    ra_err = float(_re) / 60.0               # arcmin
                if _dce is not None:
                    dec_err = float(_dce) / 60.0             # arcmin
                conv = True
        except Exception:
            pass
        finally:
            for ext in ('', '.log'):
                try:
                    os.remove(tmp + ext)
                except OSError:
                    pass
    if not conv and not imfit_ran:
        # imfit could not run (no beam / no CASA) -> matched-filter fallback
        flux = _matched_filter(cut.data, cx, cy, bpix)
        fra, fdec = ra, dec
    ul = 3.0 * rms
    if conv:
        snr = float(abs(flux) / rms)
    elif imfit_ran:
        # imfit did NOT converge -> NON-DETECTION (no spurious flux; report UL)
        flux, fra, fdec, snr = np.nan, ra, dec, 0.0
    else:
        snr = float(abs(flux) / rms) if np.isfinite(flux) else 0.0
    ratio = (fmaj / fmin) if (np.isfinite(fmaj) and np.isfinite(fmin)
                              and fmin > 0) else np.nan
    return dict(peak=float(flux) if np.isfinite(flux) else np.nan,
                err=float(err) if np.isfinite(err) else np.nan,
                snr=snr, rms=rms, upper_limit=float(ul),
                ra=float(fra), dec=float(fdec),
                sign=int(np.sign(flux)) if np.isfinite(flux) else 0,
                converged=bool(conv), imfit_ran=imfit_ran,
                fwhm_maj_deg=float(fmaj) if np.isfinite(fmaj) else np.nan,
                fwhm_min_deg=float(fmin) if np.isfinite(fmin) else np.nan,
                pa_deg=float(fpa) if np.isfinite(fpa) else np.nan,
                axis_ratio=float(ratio) if np.isfinite(ratio) else np.nan,
                ra_err_arcmin=float(ra_err) if np.isfinite(ra_err) else np.nan,
                dec_err_arcmin=float(dec_err) if np.isfinite(dec_err) else np.nan)


def empirical_beam_imfit(path, exclude_radec=None, n_src=6, peak_sigma=8.0,
                         fallback_bpix=6.0, beam=None):
    """Empirical beam = median imfit FWHM of unresolved Stokes-I sources in
    THIS image (Gregg's method — no dirty PSF needed). Greedy-pick bright peaks
    (excluding the candidate + edges), imfit each, take the median. Returns
    (maj_deg, min_deg, pa_deg, n) or None.

    Run this PER-SUBBAND, never on the Full/wideband image — the wideband
    synthesized beam would false-diagnose a real narrow-band burst as resolved.
    """
    try:
        d = np.squeeze(fits.getdata(path)).astype(float)
        hdr = fits.getheader(path)
    except Exception:
        return None
    w = WCS(hdr).celestial
    n = d.shape[0]
    finite = d[np.isfinite(d) & (d != 0)]
    if finite.size < 1000:
        return None
    rms = 1.4826 * np.median(np.abs(finite - np.median(finite)))
    thr = peak_sigma * rms
    work = np.where(np.isfinite(d), d, -np.inf).copy()
    m = int(0.15 * n)
    work[:m] = work[-m:] = -np.inf
    work[:, :m] = work[:, -m:] = -np.inf
    ex_xy = None
    if exclude_radec is not None:
        ex_xy = [float(v) for v in
                 w.world_to_pixel(SkyCoord(exclude_radec[0] * u.deg,
                                           exclude_radec[1] * u.deg))]
    majs, mins, pas = [], [], []
    for _ in range(25):
        iy, ix = np.unravel_index(np.argmax(work), work.shape)
        if work[iy, ix] < thr:
            break
        work[max(0, iy - 15):iy + 15, max(0, ix - 15):ix + 15] = -np.inf
        if ex_xy is not None and np.hypot(ix - ex_xy[0], iy - ex_xy[1]) < 30:
            continue
        sky = w.pixel_to_world(ix, iy)
        ra, dec = float(sky.ra.deg), float(sky.dec.deg)
        r = forced_imfit(path, ra, dec, fallback_bpix=fallback_bpix, beam=beam)
        if r and r['converged'] and np.isfinite(r['fwhm_maj_deg']) and \
                r['fwhm_maj_deg'] > 0:
            majs.append(r['fwhm_maj_deg'])
            mins.append(r['fwhm_min_deg'])
            pas.append(r['pa_deg'])
        if len(majs) >= n_src:
            break
    if len(majs) < 3:
        return None
    return (float(np.median(majs)), float(np.median(mins)),
            float(np.median(pas)), len(majs))


def _matched_filter(data, cx, cy, bpix):
    """Fixed-position beam matched-filter amplitude (bounded fallback)."""
    R = int(round(3 * bpix))
    sig = bpix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    ix, iy = int(round(cx)), int(round(cy))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    K = np.exp(-0.5 * (xx * xx + yy * yy) / sig ** 2)
    K = K / (K * K).sum()
    st = data[iy - R:iy + R + 1, ix - R:ix + R + 1]
    if st.shape != K.shape or not np.isfinite(st).all():
        return np.nan
    return float((st * K).sum())
