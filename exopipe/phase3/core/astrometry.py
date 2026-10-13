"""Astrometry - the position-uncertainty model and the geometry every tier shares (PHASE3_ARCHITECTURE.md section 5).

Every position and association in Phase 3 flows through this module (CLAUDE.md 6). Moved here from
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
