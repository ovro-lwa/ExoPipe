"""Astrometry - the position-uncertainty model and the geometry every tier shares (PHASE3_ARCHITECTURE.md section 5).

Every position and association in Phase 3 flows through this module (CLAUDE.md 6); numpy is imported inside the
functions that need it (the module stays light for the processes that import it for one helper). Moved here from
exopipe_ops/p3s3_astrom.py (the production deep model) and the elevation helpers that were re-implemented in a dozen
places with four different observatory latitudes; the ONE observatory location is common/pipeline_config.OVRO_LOC.

Model (COADD_ASTROMETRIC_ERROR.md 2026-07-07/08, ratified):
- systematic, 1 sigma per axis: by band class and elevation bin (``CFG.astrometry.SYS_ARCMIN``), times the class
  scale ``CFG.astrometry.SYS_SCALE`` (e.g. Stokes V Red class x1.5);
- statistical: ``CFG.astrometry.STAT_COEFF`` (0.42) x beam FWHM / S/N (S3b);
- combined in quadrature per axis; the offset percentile is the Rayleigh CDF of a separation at that sigma.
Subbands map to the band class of their frequency (Red 18-36, Green 41-59, Blue 64-82 MHz; Gregg 2026-10-07 Q14);
Full uses the Green figures.
"""
import math


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def observatory_lat_deg():
    from exopipe.common.pipeline_config import OVRO_LOC
    return float(OVRO_LOC.lat.deg)


def band_class(band):
    if band in ('Red', 'Green', 'Blue'):
        return band
    if band == 'Full':
        return 'Green'
    try:
        f = int(str(band).replace('MHz', ''))
    except ValueError:
        return 'Green'
    return 'Red' if f <= 36 else ('Green' if f <= 59 else 'Blue')


def sys_arcmin(band, elevation_deg):
    """Per-axis systematic scatter (arcmin) of the deep model at an elevation (deg); None = zenith."""
    cls = band_class(band)
    el = 90.0 if elevation_deg is None else float(elevation_deg)
    table = _cfg().astrometry.SYS_ARCMIN
    for (lo, hi), v in table.items():
        if lo <= el <= hi:
            return v[cls]
    return table[min(table)][cls]


def sys_scale(stokes, band):
    """Class scale of the systematic (build_source_catalogs_v3._sys_det: Stokes V Red-class bands x1.5)."""
    cls = 'Full' if band == 'Full' else band_class(band)
    return float(_cfg().astrometry.SYS_SCALE.get((stokes, cls), 1.0))


def stat_arcmin(beam_fwhm_arcmin, snr):
    return _cfg().astrometry.STAT_COEFF * float(beam_fwhm_arcmin) / max(abs(float(snr)), 1.0)


def sigma_arcmin(band, elevation_deg, beam_fwhm_arcmin, snr):
    return math.hypot(sys_arcmin(band, elevation_deg), stat_arcmin(beam_fwhm_arcmin, snr))


def offset_pct(sep_arcmin, band, elevation_deg, beam_fwhm_arcmin, snr):
    """Rayleigh percentile (0-100) of a separation given the per-axis sigma."""
    s = max(sigma_arcmin(band, elevation_deg, beam_fwhm_arcmin, snr), 1e-3)
    return 100.0 * (1.0 - math.exp(-0.5 * (float(sep_arcmin) / s) ** 2))


def elevation_deg(ra_deg, dec_deg, lst_deg):
    """Elevation (deg) of (ra, dec) at local sidereal angle lst_deg, at the observatory latitude."""
    ha = math.radians(lst_deg - ra_deg)
    d = math.radians(dec_deg)
    lat = math.radians(observatory_lat_deg())
    return math.degrees(math.asin(math.sin(d) * math.sin(lat) + math.cos(d) * math.cos(lat) * math.cos(ha)))


def sep_arcmin(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c)))) * 60.0


# ---------------------------------------------------------------------------------------------------------------------
# association quantities of a Mahalanobis offset (the cross-Stokes link, lib/catalog/stokes_link.py;
# R-GREGG-20261124-CROSS-STOKES-LINK-TEST-APPROVED): the offset percentile and the chance-coincidence probability that
# CLAUDE.md asks of every association ("no hard association radius; record the offset percentile and the
# chance-coincidence probability, and mark ambiguous associations as ambiguous")
# ---------------------------------------------------------------------------------------------------------------------

def offset_pct_profile(u, prof):
    """Offset percentile (0-100) of a Mahalanobis offset u (in sigma of the pair's combined covariance) under a radial
    offset profile prof = (t, s1, s2, nu): weight 1 - t of a 2-D Gaussian of scale s1 and weight t of a 2-D Student t of
    scale s2 and nu degrees of freedom (design B's PROFILE_WS, config catalog_grouping). Closed form of the profile's
    cumulative mass within u (grouping._profile_mass integrates the same density numerically)."""
    t, s1, s2, nu = (float(x) for x in prof)
    q = float(u) ** 2
    return 100.0 * ((1.0 - t) * (1.0 - math.exp(-q / (2.0 * s1 * s1))) + t * (1.0 - (1.0 + q / (nu * s2 * s2)) ** (-nu / 2.0)))


def p_chance_ellipse(rho_per_arcsec2, u, det_cov_arcsec4):
    """Chance-coincidence probability (0-1) of an association at Mahalanobis offset u: the probability that at least one
    unrelated source of areal density rho (per arcsec^2) falls inside the error ellipse out to u, whose area is
    pi u^2 sqrt(det S) arcsec^2 for the pair's combined covariance S (det in arcsec^4); Poisson: 1 - exp(-rho x area)."""
    area = math.pi * float(u) ** 2 * math.sqrt(max(float(det_cov_arcsec4), 0.0))
    return 1.0 - math.exp(-float(rho_per_arcsec2) * area)


# ---------------------------------------------------------------------------------------------------------------------
# identity-level association (the cross-Stokes link, lib/catalog/stokes_link.py; R-GREGG-20261124-CROSS-STOKES-LINK-TEST-
# APPROVED): the mean position and covariance of a group of detections, and design B's association test between two such
# means. The arithmetic is design B's (lib/catalog/grouping.py offset_en, _inv2, _quad, _areal, mean_position), line for line,
# so the link's numbers are those of the grouping's identity level.
# ---------------------------------------------------------------------------------------------------------------------

def tangent_offset_arcsec(ra1, de1, ra2, de2):
    """(east, north) offset (arcsec) of position 2 from position 1, in 1's tangent plane (arrays; degrees in)"""
    import numpy as np
    a1, d1, a2, d2 = map(np.radians, (ra1, de1, ra2, de2))
    cosc = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(a2 - a1)
    x = np.cos(d2) * np.sin(a2 - a1) / cosc
    y = (np.cos(d1) * np.sin(d2) - np.sin(d1) * np.cos(d2) * np.cos(a2 - a1)) / cosc
    return np.degrees(x) * 3600, np.degrees(y) * 3600


def _inv2(A):
    import numpy as np
    a, b, c, d = A[..., 0, 0], A[..., 0, 1], A[..., 1, 0], A[..., 1, 1]
    det = a * d - b * c
    out = np.empty_like(A)
    out[..., 0, 0] = d / det; out[..., 0, 1] = -b / det; out[..., 1, 0] = -c / det; out[..., 1, 1] = a / det
    return out


def identity_means(ra, dec, C, snr, labels):
    """The inverse-covariance mean position of each group of detections and the covariance of that mean: per group (the
    distinct values of labels, sorted), in the tangent plane of its brightest member (largest |snr|), mean = (sum C^-1)^-1
    sum C^-1 x and Cov = (sum C^-1)^-1 (arcsec^2; C: each detection's 2x2 covariance in arcsec^2, east/north). Returns
    dict(ids, ra, dec (degrees), cov (n,2,2), anchor (index of each group's brightest detection), n_det)."""
    import numpy as np
    ids, inv = np.unique(np.asarray(labels), return_inverse=True)
    snr = np.abs(np.asarray(snr, float))
    order = np.lexsort((-snr, inv))                       # per group, its brightest detection first
    first = np.ones(len(order), bool); first[1:] = inv[order][1:] != inv[order][:-1]
    anchor = np.empty(len(ids), int); anchor[inv[order][first]] = order[first]
    ra, de, C = np.asarray(ra, float), np.asarray(dec, float), np.asarray(C, float)
    a = anchor[inv]
    dx, dy = tangent_offset_arcsec(ra[a], de[a], ra, de)
    Ci = _inv2(C)
    A = np.zeros((len(ids), 2, 2)); b = np.zeros((len(ids), 2))
    np.add.at(A, inv, Ci)
    np.add.at(b, inv, np.einsum('nij,nj->ni', Ci, np.c_[dx, dy]))
    cov = _inv2(A)
    m = np.einsum('nij,nj->ni', cov, b)
    mde = de[anchor] + m[:, 1] / 3600.0
    mra = (ra[anchor] + m[:, 0] / 3600.0 / np.cos(np.radians(de[anchor]))) % 360
    return dict(ids=ids, ra=mra, dec=mde, cov=cov, anchor=anchor, n_det=np.bincount(inv, minlength=len(ids)))


def identity_association(ra1, de1, cov1, ra2, de2, cov2, rho_per_arcsec2, prof, floor_arcsec, p_min, u_cap):
    """Design B's association test between mean positions (arrays of pairs): d = the offset of 2 from 1 (arcsec, 1's tangent
    plane); S = cov1 + cov2 + floor^2 (arcsec^2: the stage-B error floor once); u = sqrt(d^T S^-1 d); f = the offset profile
    prof (t, s1, s2, nu: weight 1 - t of a 2-D Gaussian of scale s1, t of a 2-D Student t of scale s2 and nu degrees of
    freedom) at u as an areal density (per arcsec^2); odds = f / rho (rho: the background density of unrelated sources per
    arcsec^2; prior weight 1, one counterpart); P = odds / (1 + odds). Associated when u <= u_cap and P >= p_min.
    Returns dict(offset_arcsec, u, p, detS (arcsec^4), associated)."""
    import numpy as np
    dx, dy = tangent_offset_arcsec(ra1, de1, ra2, de2)
    S = cov1 + cov2 + (floor_arcsec ** 2) * np.eye(2)[None]
    d = np.c_[dx, dy]
    a, b, c = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
    detS = a * c - b * b
    q = (c * d[..., 0] ** 2 - 2 * b * d[..., 0] * d[..., 1] + a * d[..., 1] ** 2) / detS
    t, s1, s2, nu = prof
    r = np.sqrt(detS)
    f = ((1 - t) * np.exp(-q / (2 * s1 * s1)) / (2 * math.pi * s1 * s1 * r)
         + t * (1 + q / (nu * s2 * s2)) ** (-(nu + 2) / 2) / (2 * math.pi * s2 * s2 * r))
    odds = f / rho_per_arcsec2
    p = odds / (1.0 + odds)
    u = np.sqrt(q)
    return dict(offset_arcsec=np.hypot(dx, dy), u=u, p=p, detS=detS, associated=(u <= u_cap) & (p >= p_min))
