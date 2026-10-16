#!/usr/bin/env python
"""
make_band_products.py -- per-band projected FITS from the aligned HEALPix mosaics.
For each band: Mollweide (Galactic, l=0) and RA/Dec (equatorial CAR/plate-carree),
real Jy/beam, NaN off-footprint, sampled >=3 px/beam.
The HEALPix FITS themselves are the native product (healpix_aligned*/...).
"""
import os, argparse
import numpy as np, healpy as hp
from astropy.wcs import WCS
from astropy.io import fits
import astropy.coordinates as ac, astropy.units as u

BEAMS = {'Red': 13.1, 'Green': 8.4, 'Blue': 6.0, 'Full': 7.4}


def _sample(m, w, shape, frame, dec_cut=None):
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    a, b = w.pixel_to_world_values(xs, ys)
    ok = np.isfinite(a)
    if frame == 'galactic':
        sc = ac.SkyCoord(np.nan_to_num(a)*u.deg, np.nan_to_num(b)*u.deg,
                         frame='galactic').icrs
        ra, dec = sc.ra.deg, sc.dec.deg
    else:
        ra, dec = np.nan_to_num(a), np.nan_to_num(b)
    img = hp.get_interp_val(m, ra, dec, lonlat=True)
    img[~ok] = np.nan
    img[np.nan_to_num(img) == 0] = np.nan
    if dec_cut is not None:          # the render's southern edge (mosaic_render.RENDER_DEC_CUT_DEG; R-GREGG-20261016-MOSAIC-DEC-MIN-30)
        img[np.asarray(dec) < dec_cut - 1e-9] = np.nan
    return img.astype(np.float32)


def _beam_arcmin(hpx_dir, variant, b):
    """Scalar beam for the render header. v2 (2026-09-29): the equivalent circular FWHM of the median per-pixel Omega_eff
    of the mosaic's own PSF map, which exists for every band including the subbands; else the July fixed table."""
    p = f"{hpx_dir}/healpix_PSF_I_deep_{variant}_{b}_nside2048.fits"
    if os.path.exists(p):
        om = hp.read_map(p, verbose=False); om = om[np.isfinite(om) & (om > 0) & (om != hp.UNSEEN)]
        if om.size:
            return float(np.degrees(np.sqrt(np.median(om) / 1.1331)) * 60.0)
    return BEAMS[b]


def moll_wcs(W):
    w = WCS(naxis=2); w.wcs.ctype = ['GLON-MOL', 'GLAT-MOL']
    w.wcs.crval = [0., 0.]; w.wcs.crpix = [W/2, W/4]; w.wcs.cdelt = [-360./W, 360./W]
    return w, (W//2, W)


def car_wcs(cdelt_deg, dec_lo=-45.0, dec_hi=90.0):
    nx = int(round(360.0/cdelt_deg)); ny = int(round((dec_hi-dec_lo)/cdelt_deg))
    w = WCS(naxis=2); w.wcs.ctype = ['RA---CAR', 'DEC--CAR']
    w.wcs.crval = [180.0, 0.0]
    w.wcs.crpix = [nx/2 + 0.5, (0.0 - dec_lo)/cdelt_deg + 0.5]
    w.wcs.cdelt = [-cdelt_deg, cdelt_deg]
    return w, (ny, nx)


def build(variant, hpx_dir, suffix, out, bands=('Red', 'Green', 'Blue', 'Full'),
          moll_W=12000, car_cdelt=1.5/60, dec_cut=None):
    os.makedirs(out, exist_ok=True)
    for b in bands:
        src = f"{hpx_dir}/healpix_I_deep_{variant}_{b}{suffix}_nside2048.fits"
        m = hp.read_map(src, verbose=False)
        bunit = fits.getheader(src, 1).get('BUNIT', 'JY/BEAM')   # v2 jysr maps are Jy/sr - was hardcoded JY/BEAM
        bm = _beam_arcmin(hpx_dir, variant, b)
        pxbm = bm / (360./moll_W*60)
        # Mollweide (galactic)
        w, sh = moll_wcs(moll_W)
        img = _sample(m, w, sh, 'galactic', dec_cut)
        h = w.to_header(); h['BUNIT'] = bunit; h['BAND'] = b; h['VARIANT'] = variant
        h['BMAJ'] = bm/60; h['PXPERBM'] = round(pxbm, 2)
        if dec_cut is not None:
            h['DECMIN'] = (float(dec_cut), '[deg] rendered declination >= DECMIN')
        fits.writeto(f"{out}/{b}_{variant}{suffix}_Mollweide.fits", img, h, overwrite=True)
        # RA/Dec (equatorial CAR)
        w2, sh2 = car_wcs(car_cdelt)
        img2 = _sample(m, w2, sh2, 'equatorial', dec_cut)
        h2 = w2.to_header(); h2['BUNIT'] = bunit; h2['BAND'] = b; h2['VARIANT'] = variant
        h2['BMAJ'] = bm/60; h2['PXPERBM'] = round(bm/(car_cdelt*60), 2)
        if dec_cut is not None:
            h2['DECMIN'] = (float(dec_cut), '[deg] rendered declination >= DECMIN')
        fits.writeto(f"{out}/{b}_{variant}{suffix}_RADec.fits", img2, h2, overwrite=True)
        print(f"wrote {b}: Mollweide {sh} ({pxbm:.1f}px/bm) + RADec {sh2} "
              f"({bm/(car_cdelt*60):.1f}px/bm)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='NoTaper_R075')
    ap.add_argument('--hpx-dir', default='/fast/gh/main/phase3/mosaic/healpix_aligned_v2')
    ap.add_argument('--suffix', default='_aligned')
    ap.add_argument('--out', default='/fast/gh/main/phase3/mosaic/fits_bands')
    ap.add_argument('--moll-width', type=int, default=12000)
    ap.add_argument('--car-arcmin', type=float, default=1.5)
    ap.add_argument('--bands', default='Red,Green,Blue,Full', help='comma list; subband labels allowed (v2)')
    ap.add_argument('--dec-cut', type=float, default=None, help='render declination >= this (default: no cut)')
    a = ap.parse_args()
    build(a.variant, a.hpx_dir, a.suffix, a.out, bands=a.bands.split(','), moll_W=a.moll_width,
          car_cdelt=a.car_arcmin/60, dec_cut=a.dec_cut)
