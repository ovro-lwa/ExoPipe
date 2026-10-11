#!/usr/bin/env python
"""
add_ateam.py -- add the peeled A-team back into a Phase-3 mosaic for inspection.

Cyg A, Cas A and Vir A are peeled during imaging, leaving holes at their positions. Tau A is NOT
peeled: the Phase-1 peel model (sources.json) holds only Cyg A, Cas A and Vir A (FINDING-TAU-A-NOT-IN-PHASE1-PEEL-
MODEL-05H-MASTERS; Gregg 2026-10-11: a Phase-1 bug he fixes, nothing redone), so Tau A is already in the images and is
NOT added back (adding it would double it). This adds each back as a beam-shaped Gaussian at its position, with the flux
from the PIPELINE calibrator flux models (workingv18/generate_flux_models.py -- Baars-fit
Cyg A, epoch-2023.5 definitive Cas A, log-poly Vir A, 970*(nu/1GHz)^-0.30 Tau A),
evaluated at each band centre. The mosaics are PB-corrected, so the intrinsic (above-beam)
model flux is what is added (peak Jy/beam for an unresolved source). Positions for
Cyg/Cas/Vir are the peel-model (sources.json) centroids; Tau A is the Crab position.

Non-destructive: reads the canonical aligned HEALPix, writes an A-team copy to a separate
directory. Reproject/render with make_band_products.py / render_ds9log.py pointed there.
"""
import os, json, sys
import numpy as np
import healpy as hp
from astropy.io import fits
import astropy.coordinates as ac, astropy.units as u
from exopipe.common import pipeline_config as pc

sys.path.insert(0, "/lustre/gh/calibration/pipeline/workingv18")
import matplotlib; matplotlib.use('Agg')          # gfm imports pyplot
import generate_flux_models as gfm

SRC_HPX = "/fast/gh/main/phase3/mosaic/healpix_aligned_v2"
OUT_HPX = "/fast/gh/main/phase3/mosaic/post_fix/ateam/healpix"
VARIANT = "NoTaper_R075"
NSIDE = 2048
BAND_FREQ_MHZ = {'Red': 29.0, 'Green': 50.0, 'Blue': 74.0, 'Full': 60.0}
BAND_BEAM_ARCMIN = {'Red': 13.1, 'Green': 8.4, 'Blue': 6.0, 'Full': 7.4}

# Peel-model (sources.json) flux-weighted centroids for Cyg/Cas/Vir; Crab for Tau A.
ATEAM_POS = {
    'Cyg A': (299.8719, 40.7345),
    'Cas A': (350.8349, 58.8375),
    'Vir A': (187.7059, 12.3911),
}
# Tau A (Crab, 83.6331 +22.0145) is not peeled in Phase 1 (2026-10-11) - it stays in the images and is never added back here.


def _band_flux_models():
    """Per-source flux(nu_hz) callables from workingv18/generate_flux_models.py."""
    cyga = gfm.fit_polynomial_model(gfm.CYGA_FREQS_BAARS, gfm.CYGA_FLUX_BAARS,
                                    gfm.CYGA_ERR_PERCENT, 8)
    casa = gfm.fit_polynomial_model(gfm.CASA_FREQS_2023_MHZ, gfm.CASA_FLUX_2023_JY,
                                    gfm.CASA_ERR_2023_PERCENT, 10)
    return {
        'Cyg A': lambda fhz: gfm.get_flux_from_poly_fit(cyga, fhz),
        'Cas A': lambda fhz: gfm.get_flux_from_poly_fit(casa, fhz),
        'Vir A': lambda fhz: gfm.get_flux_from_log_poly(fhz, gfm.VIRA_COEFFS),
        'Tau A': lambda fhz: 970.0 * (np.asarray(fhz) / 1e6 / 1000.0) ** (-0.30),
    }


def ateam_sources():
    """Return [(name, ra_deg, dec_deg, {band: flux_Jy})] from the pipeline flux models."""
    models = _band_flux_models()
    out = []
    for name, (ra, dec) in ATEAM_POS.items():
        flux = {b: float(models[name](nu * 1e6)) for b, nu in BAND_FREQ_MHZ.items()}
        out.append((name, ra, dec, flux))
    return out


def add_gaussian(m, ra, dec, peak, fwhm_arcmin, nside=NSIDE):
    """Add a Gaussian (peak Jy/beam, FWHM arcmin) at (ra,dec) into RING map m, in place."""
    sigma = np.radians(fwhm_arcmin / 60.0) / 2.35482
    vec = hp.ang2vec(ra, dec, lonlat=True)
    disc = hp.query_disc(nside, vec, 4 * sigma)
    dirs = np.array(hp.pix2vec(nside, disc))            # 3 x N
    ang = np.arccos(np.clip(vec @ dirs, -1.0, 1.0))     # angular sep (rad)
    m[disc] += (peak * np.exp(-0.5 * (ang / sigma) ** 2)).astype(m.dtype)


def add_gaussian_wcs(img, w, shape, ra, dec, peak, fwhm_arcmin, frame):
    """Add an analytic circular Gaussian (true angular distance per pixel) into a WCS
    image, in place. Bounding-box limited. `frame` = 'galactic' or 'equatorial'."""
    sigma = fwhm_arcmin / 2.35482
    cdelt_am = abs(w.wcs.cdelt[1]) * 60.0
    r = int(np.ceil(5 * sigma / cdelt_am))
    src = ac.SkyCoord(ra * u.deg, dec * u.deg)
    if frame == 'galactic':
        g = src.galactic
        sx, sy = w.world_to_pixel_values(g.l.deg, g.b.deg)
    else:
        sx, sy = w.world_to_pixel_values(ra, dec)
    sxi, syi = int(round(float(sx))), int(round(float(sy)))
    ylo, yhi = max(0, syi - r), min(shape[0], syi + r + 1)
    xlo, xhi = max(0, sxi - r), min(shape[1], sxi + r + 1)
    if ylo >= yhi or xlo >= xhi:
        return
    ys, xs = np.mgrid[ylo:yhi, xlo:xhi]
    a, b = w.pixel_to_world_values(xs, ys)
    pts = (ac.SkyCoord(a * u.deg, b * u.deg, frame='galactic') if frame == 'galactic'
           else ac.SkyCoord(a * u.deg, b * u.deg))
    sep = src.separation(pts).arcmin
    img[ylo:yhi, xlo:xhi] += (peak * np.exp(-0.5 * (sep / sigma) ** 2)).astype(img.dtype)


def build_healpix(variant=VARIANT, src_hpx=SRC_HPX, out_hpx=OUT_HPX, srcs=None):
    """A-team added to the HEALPix maps (analytic Gaussian on the sphere -> round)."""
    os.makedirs(out_hpx, exist_ok=True)
    srcs = srcs or ateam_sources()
    for band in BAND_FREQ_MHZ:
        f = f"{src_hpx}/healpix_I_deep_{variant}_{band}_aligned_nside{NSIDE}.fits"
        m = hp.read_map(f, verbose=False).astype(np.float32)
        for name, ra, dec, flux in srcs:
            add_gaussian(m, ra, dec, flux[band], BAND_BEAM_ARCMIN[band])
        out = f"{out_hpx}/healpix_I_deep_{variant}_{band}_aligned_nside{NSIDE}.fits"
        col = fits.Column(name='SIGNAL', format='E', array=m)
        hdu = fits.BinTableHDU.from_columns([col])
        for k, v in [('PIXTYPE', 'HEALPIX'), ('ORDERING', 'RING'), ('NSIDE', NSIDE),
                     ('COORDSYS', 'C'), ('INDXSCHM', 'IMPLICIT'), ('BUNIT', 'JY/BEAM'),
                     ('ALIGNED', True), ('ATEAM', 'Cyg+Cas+Vir+Tau added (workingv18 flux models)')]:
            hdu.header[k] = v
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(out, overwrite=True)
        print(f"wrote {out}")


def build_projected(variant=VARIANT, src_hpx=SRC_HPX,
                    out_root="/fast/gh/main/phase3/mosaic/post_fix/ateam",
                    srcs=None, moll_W=12000, car_cdelt=1.5 / 60):
    """Mollweide + RA/Dec FITS: reproject the CLEAN mosaic, then add the A-team
    ANALYTICALLY in-frame (avoids the reproject-of-undersampled-peak squareness)."""
    from exopipe.phase3 import make_band_products as mbp
    srcs = srcs or ateam_sources()
    os.makedirs(f"{out_root}/mollweide", exist_ok=True)
    os.makedirs(f"{out_root}/radec", exist_ok=True)
    for band in BAND_FREQ_MHZ:
        m = hp.read_map(f"{src_hpx}/healpix_I_deep_{variant}_{band}_aligned_nside{NSIDE}.fits",
                        verbose=False)
        beam = BAND_BEAM_ARCMIN[band]
        # Mollweide (galactic)
        w, sh = mbp.moll_wcs(moll_W)
        img = mbp._sample(m, w, sh, 'galactic')
        for name, ra, dec, flux in srcs:
            add_gaussian_wcs(img, w, sh, ra, dec, flux[band], beam, 'galactic')
        h = w.to_header(); h['BUNIT'] = 'JY/BEAM'; h['BAND'] = band; h['ATEAM'] = 'in-frame Gaussian'
        fits.writeto(f"{out_root}/mollweide/{band}_{variant}_aligned_Mollweide.fits",
                     img.astype(np.float32), h, overwrite=True)
        # RA/Dec (equatorial CAR)
        w2, sh2 = mbp.car_wcs(car_cdelt)
        img2 = mbp._sample(m, w2, sh2, 'equatorial')
        for name, ra, dec, flux in srcs:
            add_gaussian_wcs(img2, w2, sh2, ra, dec, flux[band], beam, 'equatorial')
        h2 = w2.to_header(); h2['BUNIT'] = 'JY/BEAM'; h2['BAND'] = band; h2['ATEAM'] = 'in-frame Gaussian'
        fits.writeto(f"{out_root}/radec/{band}_{variant}_aligned_RADec.fits",
                     img2.astype(np.float32), h2, overwrite=True)
        print(f"wrote projected {band} (Mollweide + RADec, in-frame Gaussian)")


if __name__ == "__main__":
    srcs = ateam_sources()
    print("A-team from pipeline flux models (workingv18/generate_flux_models.py):")
    for name, ra, dec, flux in srcs:
        print(f"  {name:7} RA={ra:8.4f} Dec={dec:+8.4f}  "
              + "  ".join(f"{b}={flux[b]:.0f}Jy" for b in BAND_FREQ_MHZ))
    build_healpix(srcs=srcs)
    build_projected(srcs=srcs)
