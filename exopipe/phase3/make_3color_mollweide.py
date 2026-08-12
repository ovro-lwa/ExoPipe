#!/usr/bin/env python
"""
make_3color_mollweide.py -- NoTaper 3-color Mollweide PNGs.
Goal (Gregg 2026-07-04): no rainbow, compact circular features, max resolution,
optimal sampling (err larger). Approach:
  - chroma  : R/G/B beam-matched to the coarsest band (Red 13.1') -> no chromatic bullseye
  - luminance: sharp Full band (7.4') -> keeps structure high-res
  - mask    : common coverage of all bands, eroded to kill the differential-coverage rim
  - stretch : linear (Phase-2 faithful) AND log
Sampling via hp.get_interp_val (validated correct). Galactic, l=0 centered.
"""
import numpy as np, healpy as hp
from astropy.wcs import WCS
import astropy.coordinates as ac, astropy.units as u
from scipy.ndimage import gaussian_filter, binary_erosion
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from exopipe.phase3 import pb_render as pr

HP = "/fast/gh/main/phase3/mosaic/healpix"
OUT = "/fast/gh/main/phase3/mosaic/png_v1"
BEAMS = {'Red': 13.1, 'Green': 8.4, 'Blue': 6.0, 'Full': 7.4}
TARGET = 13.1  # common chroma resolution (coarsest band)


PROJ = 'MOL'  # display projection code (MOL or AIT); set by build()


def reproject_mollweide(m, W, crval1=0.0):
    w = WCS(naxis=2); w.wcs.ctype = [f'GLON-{PROJ}', f'GLAT-{PROJ}']
    w.wcs.crval = [crval1, 0.0]; w.wcs.crpix = [W/2, W/4]; w.wcs.cdelt = [-360./W, 360./W]
    ys, xs = np.mgrid[0:W//2, 0:W]
    glon, glat = w.pixel_to_world_values(xs, ys); ok = np.isfinite(glon)
    sc = ac.SkyCoord(np.nan_to_num(glon)*u.deg, np.nan_to_num(glat)*u.deg,
                     frame='galactic').icrs
    img = hp.get_interp_val(m, sc.ra.deg, sc.dec.deg, lonlat=True)
    img[~ok] = np.nan
    return img, ok


def build(variant='NoTaper_R075', W=8000, erode_deg=1.5, out=OUT, proj='MOL',
          hpx_dir=HP, suffix='', beam_match=True):
    global PROJ; PROJ = proj
    import os; os.makedirs(out, exist_ok=True)
    arcmin_pix = 360./W*60
    bands = {}
    cov = None
    for b in ['Red', 'Green', 'Blue', 'Full']:
        m = hp.read_map(f"{hpx_dir}/healpix_I_deep_{variant}_{b}{suffix}_nside2048.fits",
                        verbose=False)
        img, ok = reproject_mollweide(m, W)
        bands[b] = img
        c = np.isfinite(img) & (img != 0)
        cov = c if cov is None else (cov & c)
    cov = binary_erosion(cov, iterations=int(erode_deg/(360./W)))

    def smooth_to(a, fr):
        if TARGET <= fr:
            return a
        s = np.sqrt(TARGET**2 - fr**2)/2.355/arcmin_pix
        mm = np.isfinite(a); x = np.where(mm, a, 0.)
        num = gaussian_filter(x, s); den = gaussian_filter(mm.astype(float), s)
        o = np.where(den > 0.05, num/den, np.nan); o[~mm] = np.nan; return o

    if beam_match:
        Rc = smooth_to(bands['Red'], BEAMS['Red'])
        Gc = smooth_to(bands['Green'], BEAMS['Green'])
        Bc = smooth_to(bands['Blue'], BEAMS['Blue'])
    else:  # native resolution per band (max resolution; chroma not homogenized)
        Rc, Gc, Bc = bands['Red'], bands['Green'], bands['Blue']
    normed, valid = pr._phase2_channel_norm([Rc, Gc, Bc])
    rgb = np.stack(normed, axis=-1)

    def lum(stretch):
        f = bands['Full']; fv = np.isfinite(f) & (f != 0)
        if stretch == 'linear':
            vmin, vmax = np.nanpercentile(f[fv], (0.5, 99.8))
            L = np.clip((f - vmin)/max(vmax-vmin, 1e-9), 0, 1)
        else:  # log with median pedestal
            vmin, vmax = np.nanpercentile(f[fv], (50, 99.9))
            L = np.clip((f - vmin)/max(vmax-vmin, 1e-9), 0, 1)
            L = np.log10(1000*L + 1)/np.log10(1001)
        return L

    def save(arr, path, title):
        fig = plt.figure(figsize=(16, 8), facecolor='black')
        ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(arr, origin='lower', interpolation='nearest'); ax.axis('off')
        ax.text(0.5, 0.97, title, color='white', fontsize=13, ha='center', va='top',
                transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.5))
        fig.savefig(path, dpi=110, facecolor='black'); plt.close(fig)

    projname = {'MOL': 'Mollweide', 'AIT': 'HammerAitoff'}.get(proj, proj)
    if suffix:
        projname += suffix   # e.g. _aligned
    projname += ('_beammatched' if beam_match else '_native')
    bmlabel = 'beam-matched' if beam_match else 'native-res'
    outs = []
    for stretch in ['linear', 'log']:
        L = lum(stretch)
        lrgb = 0.2126*rgb[..., 0] + 0.7152*rgb[..., 1] + 0.0722*rgb[..., 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            fac = np.where(lrgb > 1e-6, L/lrgb, 0.)
        comp = np.clip(rgb*fac[..., None], 0, 1)
        comp[~cov] = 0.0
        p = f"{out}/3color_{variant}_{projname}_fulllum_{stretch}.png"
        save(comp, p, f"OVRO-LWA 3-color {variant} {bmlabel}+Full-lum [{stretch}]")
        outs.append(p)
        # also plain (no luminance), same mask+beam-match
        plain = rgb.copy(); plain[~cov] = 0.0
        if stretch == 'log':  # apply log to plain channels too
            plain = np.stack([np.log10(1000*np.clip(c,0,1)+1)/np.log10(1001) for c in normed], axis=-1)
            plain[~cov] = 0.0
        p2 = f"{out}/3color_{variant}_{projname}_plain_{stretch}.png"
        save(plain, p2, f"OVRO-LWA 3-color {variant}   plain RGB  [{stretch}]")
        outs.append(p2)
    for p in outs:
        print("wrote", p)
    return outs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='NoTaper_R075')
    ap.add_argument('--width', type=int, default=8000)
    ap.add_argument('--erode-deg', type=float, default=1.5)
    ap.add_argument('--proj', default='MOL', choices=['MOL', 'AIT'])
    ap.add_argument('--hpx-dir', default=HP)
    ap.add_argument('--suffix', default='')
    ap.add_argument('--no-beam-match', action='store_true',
                    help='keep each band at native resolution (max resolution, no chroma '
                         'homogenization)')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    build(a.variant, a.width, a.erode_deg, out=a.out, proj=a.proj, hpx_dir=a.hpx_dir,
          suffix=a.suffix, beam_match=not a.no_beam_match)
