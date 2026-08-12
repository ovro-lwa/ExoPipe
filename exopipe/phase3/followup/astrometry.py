#!/usr/bin/env python3
"""candinv/astrometry.py — candidate position + positional error.

The positional error is TWO independent terms (project_astrometry_error_design):

  STATISTICAL  per-source imfit fit error, ~ 0.42*FWHM/SNR per axis, for the
               transient's own detection.
  SYSTEMATIC   calibration + ionosphere, DIRECTION-DEPENDENT. The Phase-2
               dewarping (73 MHz RBF-multiquadric anchor vs VLSSr, cascaded
               sub-band by sub-band) + optical-flow alignment removes the bulk;
               the residual is bounded as a LOWER LIMIT from BRIGHT, unresolved
               Stokes-I sources NEAR the transient that are robustly matched to
               VLSSr via the 73 MHz anchor band. Lower limit because those very
               sources anchored the dewarp, so the error grows with distance from
               an anchor.

Rules: use the dewarped, ALIGNED, pre-sky-subtraction Stokes-I image at the
DETECTION sub-band/wideband (never Full unless Full-only, then warn). Source
positions are Gaussian-fit CENTROIDS (BDSF posn_sky_centroid), never peak pixel.
Match to VLSSr only at 73 MHz (spectral-index-safe). Search radius 2' for
dewarped deep/10-min, 6'x74/nu for snapshot. Anchor SNR floor is set RELATIVE to
the transient (a few times higher) so anchor statistical error is negligible.
"""
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from . import refs


def stat_pos_error_arcmin(fwhm_deg, snr):
    """Statistical positional error ~ 0.42*FWHM/SNR per axis [arcmin]
    (standard Gaussian-centroid precision)."""
    if not fwhm_deg or not snr or snr <= 0 or not np.isfinite(fwhm_deg):
        return float('nan')
    return float(0.42 * (fwhm_deg * 60.0) / snr)


def _mad(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if v.size == 0:
        return float('nan')
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


def _bdsf_centroids(ipath, snr_floor, near_coord=None, near_deg=None, nmax=400):
    """Bright unresolved Stokes-I sources via BDSF Gaussian-fit CENTROID
    (posn_sky_centroid — NEVER peak pixel). Returns [(SkyCoord, peak, snr)]."""
    try:
        import bdsf
        img = bdsf.process_image(ipath, thresh_pix=8.0, thresh_isl=5.0,
                                 adaptive_rms_box=True, quiet=True)
    except Exception:
        return []
    out = []
    for s in img.sources:
        if getattr(s, 'code', 'S') != 'S':        # single-Gaussian / unresolved
            continue
        try:
            ra, dec = s.posn_sky_centroid
        except Exception:
            continue
        if not np.isfinite(ra):
            continue
        rms = float(getattr(s, 'rms_isl', 0) or 0)
        snr = (float(s.peak_flux_max) / rms) if rms > 0 else 0.0
        if snr < snr_floor:
            continue
        out.append((SkyCoord(ra * u.deg, dec * u.deg), float(s.peak_flux_max), snr))
    if near_coord is not None and near_deg is not None:
        out = [o for o in out if o[0].separation(near_coord).deg <= near_deg]
    out.sort(key=lambda o: -o[1])
    return out[:nmax]


def local_systematic(coord, check_ipath, ref73_ipath, dewarped, freq_mhz,
                     transient_snr, near_deg=15.0, snr_factor=3.0,
                     min_snr_floor=15.0, vlss_radius=2.0):
    """LOWER-LIMIT systematic positional error near `coord` [arcmin].

    Bright nearby Stokes-I anchors (centroids in the DETECTION-band aligned I
    image) -> position-link to the 73 MHz aligned image -> match to VLSSr at
    73 MHz. Residual of the detection-band positions vs VLSSr = lower limit on the
    direction-dependent systematic in the transient's region.
    """
    out = {'near_deg': near_deg, 'n_near_bright': 0, 'n_anchor_match': 0}
    snr_floor = max(min_snr_floor, snr_factor * (transient_snr or 0))
    out['anchor_snr_floor'] = round(snr_floor, 1)
    radius = 2.0 if dewarped else float(np.clip(6.0 * 74.0 / max(freq_mhz, 10.0),
                                                2.0, 30.0))
    out['match_radius_arcmin'] = radius
    out['dewarped'] = dewarped
    if not check_ipath or not ref73_ipath:
        out['note'] = 'missing check/73MHz I-aligned image'
        return out
    bs = _bdsf_centroids(check_ipath, snr_floor, near_coord=coord,
                         near_deg=near_deg)
    out['n_near_bright'] = len(bs)
    if len(bs) < 3:
        out['note'] = 'fewer than 3 bright nearby anchors above the SNR floor'
        return out
    r73 = _bdsf_centroids(ref73_ipath, snr_floor)
    if len(r73) < 3:
        out['note'] = 'too few 73 MHz anchor sources'
        return out
    c73 = SkyCoord([s[0].ra.deg for s in r73] * u.deg,
                   [s[0].dec.deg for s in r73] * u.deg)
    vco, _ = refs.vlss_unresolved()
    dra, ddec, seps = [], [], []
    for sc, _flux, _snr in bs:
        i, sep73, _ = sc.match_to_catalog_sky(c73)      # link to 73 MHz frame
        if sep73.arcmin > radius:
            continue
        p73 = c73[int(i)]
        j, sepv, _ = p73.match_to_catalog_sky(vco)       # 73 vs 74 MHz: robust
        if sepv.arcmin > vlss_radius:
            continue
        vp = vco[int(j)]
        cosd = np.cos(np.radians(sc.dec.deg))
        dx = (sc.ra.deg - vp.ra.deg) * cosd * 60.0
        dy = (sc.dec.deg - vp.dec.deg) * 60.0
        dra.append(dx); ddec.append(dy); seps.append(np.hypot(dx, dy))
    out['n_anchor_match'] = len(seps)
    if len(seps) < 3:
        out['note'] = 'fewer than 3 robust 73 MHz-anchored matches'
        return out
    dra, ddec, seps = np.array(dra), np.array(ddec), np.array(seps)
    # Keep RA and Dec SEPARATE — the residual is direction-dependent (the
    # ionospheric structure depends on the bearing transient->anchor).
    out['sys_offset_ra_arcmin'] = float(np.median(dra))     # local net shift, RA
    out['sys_offset_dec_arcmin'] = float(np.median(ddec))   # local net shift, Dec
    out['sys_scatter_ra_arcmin'] = float(_mad(dra))
    out['sys_scatter_dec_arcmin'] = float(_mad(ddec))
    # single end-number (lower limit): RMS residual magnitude of nearby anchors.
    out['sys_lowerlimit_arcmin'] = float(np.sqrt(np.mean(seps ** 2)))
    return out
