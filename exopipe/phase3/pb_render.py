#!/usr/bin/env python
"""
pb_render.py -- render HEALPix all-sky mosaics to Mollweide PNGs + a 3-field RGB
HEALPix FITS.

Reprojects each band's HEALPix mosaic onto a Mollweide WCS grid (equatorial ICRS by
default) and produces:
  * per-band grayscale Mollweide PNG (asinh)
  * 3-color Mollweide PNG using the Phase-2 scheme (spectral index -0.7, common valid
    mask, linear 0.5-99.8 percentile) -- PLAIN RGB
  * 3-color Mollweide PNG with the Full band as LUMINANCE (hue preserved)
  * a 3-field (R,G,B) HEALPix FITS -- re-stretchable 3-color data product

No healpy dependency: uses reproject.reproject_from_healpix. See MOSAIC_DESIGN.md.
"""
import os
import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_from_healpix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from exopipe.common import phase2_config as cfg

# Phase-2 3-color scheme constants
BAND_FREQS = {'Red': 29.5, 'Green': 52.5, 'Blue': 74.5}   # matches phase2_science_agg
ALPHA = getattr(cfg, 'THREE_COLOR_SPECTRAL_INDEX', -0.7)
PCT = (0.5, 99.8)


def _mollweide_wcs(width, frame='icrs', crval1=None):
    # Center the map on l,b=0 for Galactic (Gregg 2026-07-04); RA=180 for equatorial.
    if crval1 is None:
        crval1 = 0.0 if frame == 'galactic' else 180.0
    h = width // 2
    w = WCS(naxis=2)
    w.wcs.ctype = (['RA---MOL', 'DEC--MOL'] if frame == 'icrs'
                   else ['GLON-MOL', 'GLAT-MOL'])
    w.wcs.crval = [crval1, 0.0]
    w.wcs.crpix = [width / 2 + 0.5, h / 2 + 0.5]
    # full sky: 360 deg across 2*sqrt(2) native units; standard Mollweide scale
    scale = 360.0 / width
    w.wcs.cdelt = [-scale, scale]
    return w, (h, width)


def _read_hpx(path):
    with fits.open(path) as hdul:
        d = hdul[1].data
        col = 'SIGNAL' if 'SIGNAL' in d.columns.names else d.columns.names[0]
        arr = np.asarray(d[col], dtype=np.float64)
        nside = hdul[1].header['NSIDE']
        coordsys = hdul[1].header.get('COORDSYS', 'C')
    return arr, nside, ('icrs' if coordsys in ('C', 'Q') else 'galactic')


def _to_mollweide(hpx_path, width, frame='icrs'):
    arr, nside, hp_frame = _read_hpx(hpx_path)
    wcs, shape = _mollweide_wcs(width, frame=frame)
    grid, _ = reproject_from_healpix((arr, hp_frame), wcs, shape_out=shape,
                                     nested=False)
    return grid, wcs, shape


def _norm01(img, pct=PCT):
    finite = np.isfinite(img)
    if finite.sum() == 0:
        return np.zeros_like(img), np.zeros(img.shape, bool)
    vmin, vmax = np.nanpercentile(img[finite], pct)
    if vmax <= vmin:
        vmax = vmin + 1e-9
    return np.clip((img - vmin) / (vmax - vmin), 0, 1), finite


def _asinh_norm(img, pct=PCT):
    x, _ = _norm01(img, pct)
    return np.arcsinh(10 * x) / np.arcsinh(10)


def _log_norm(img, pct=(50.0, 99.9), a=1000.0):
    """Logarithmic stretch. Pedestal at the median (pct[0]) so the noise floor maps to
    0 and the bright end is log-compressed (a = dynamic range)."""
    x, _ = _norm01(img, pct)
    return np.log10(a * x + 1.0) / np.log10(a + 1.0)


def _log_norm_nped(img, a=1000.0):
    """Log stretch WITHOUT a pedestal (low percentile 0.5) -- faint/noise floor kept."""
    return _log_norm(img, pct=(0.5, 99.9), a=a)


_STRETCH = {'asinh': _asinh_norm, 'log': _log_norm, 'log_nped': _log_norm_nped}


def render_band_png(hpx_path, out_png, width, frame='icrs', title='', stretch='asinh'):
    grid, wcs, shape = _to_mollweide(hpx_path, width, frame)
    norm = _STRETCH[stretch](grid)
    norm[~np.isfinite(grid)] = 0.0
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.94])
    ax.imshow(norm, origin='lower', cmap='inferno', interpolation='nearest')
    ax.axis('off')
    if title:
        ax.set_title(title, color='white', fontsize=14,
                     bbox=dict(facecolor='black', alpha=0.7))
    fig.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    return grid


def _phase2_channel_norm(channels):
    """Phase-2 scheme: spectral-index prescale, common valid mask, linear percentile."""
    freqs = np.array([BAND_FREQS['Red'], BAND_FREQS['Green'], BAND_FREQS['Blue']])
    ref = np.exp(np.mean(np.log(freqs)))
    sf = (freqs / ref) ** ALPHA
    chans = [c * s for c, s in zip(channels, sf)]
    valid = np.ones(chans[0].shape, bool)
    for c in chans:
        valid &= np.isfinite(c) & (c != 0)
    normed = []
    for c in chans:
        px = c[valid]
        vmin, vmax = (np.nanpercentile(px, PCT) if px.size else (0, 1))
        if vmax <= vmin:
            vmax = vmin + 1
        s = np.clip((c - vmin) / (vmax - vmin), 0, 1)
        s[~valid] = 0.0
        normed.append(s)
    return normed, valid


def render_3color(rgb_hpx_paths, out_png, width, frame='icrs', title='',
                  full_hpx_path=None, out_png_lum=None):
    """rgb_hpx_paths = (Red, Green, Blue) HEALPix mosaic paths."""
    channels = [_to_mollweide(p, width, frame)[0] for p in rgb_hpx_paths]
    normed, valid = _phase2_channel_norm(channels)
    rgb = np.stack(normed, axis=-1)

    def _save(arr, path, ttl):
        fig = plt.figure(figsize=(14, 7))
        ax = fig.add_axes([0.01, 0.01, 0.98, 0.94])
        ax.imshow(arr, origin='lower', interpolation='nearest')
        ax.axis('off')
        if ttl:
            ax.set_title(ttl, color='white', fontsize=14,
                         bbox=dict(facecolor='black', alpha=0.7))
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig)

    _save(rgb, out_png, title + ' (RGB)')

    if full_hpx_path and out_png_lum:
        full = _to_mollweide(full_hpx_path, width, frame)[0]
        lum_tgt = np.zeros_like(full)
        fvalid = np.isfinite(full) & (full != 0)
        if fvalid.sum():
            vmin, vmax = np.nanpercentile(full[fvalid], PCT)
            if vmax <= vmin:
                vmax = vmin + 1
            lum_tgt = np.clip((full - vmin) / (vmax - vmin), 0, 1)
        lum_rgb = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = np.where(lum_rgb > 1e-6, lum_tgt / lum_rgb, 0.0)
        rgb_lum = np.clip(rgb * factor[..., None], 0, 1)
        rgb_lum[~valid] = 0.0
        _save(rgb_lum, out_png_lum, title + ' (Full luminance)')

    return out_png


def write_rgb_healpix(rgb_hpx_paths, out_path):
    """Combine R/G/B HEALPix mosaics into one 3-field FITS (re-stretchable 3-color)."""
    cols = []
    nside = None
    for name, p in zip(['RED', 'GREEN', 'BLUE'], rgb_hpx_paths):
        arr, ns, _ = _read_hpx(p)
        nside = ns
        cols.append(fits.Column(name=name, format='E', array=arr))
    hdu = fits.BinTableHDU.from_columns(cols)
    h = hdu.header
    h['PIXTYPE'] = 'HEALPIX'; h['ORDERING'] = 'RING'; h['NSIDE'] = nside
    h['COORDSYS'] = 'C'; h['INDXSCHM'] = 'IMPLICIT'; h['BUNIT'] = 'JY/BEAM'
    h['CONTENT'] = ('3-color RGB (Red/Green/Blue bands)', '')
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(out_path, overwrite=True)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--hpx-dir', default='/fast/gh/main/phase3/mosaic/healpix')
    ap.add_argument('--variant', default='Taper_R075')
    ap.add_argument('--nside', type=int, default=2048)
    ap.add_argument('--width', type=int, default=6000)
    ap.add_argument('--frame', default='galactic', choices=['icrs', 'galactic'])
    ap.add_argument('--out', default='/fast/gh/main/phase3/mosaic/render')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def hp(band):
        return os.path.join(args.hpx_dir,
                            f"healpix_I_deep_{args.variant}_{band}_nside{args.nside}.fits")

    for band in ['Red', 'Green', 'Blue', 'Full']:
        p = hp(band)
        if os.path.exists(p):
            render_band_png(p, os.path.join(args.out, f"mollweide_{args.variant}_{band}.png"),
                            args.width, args.frame,
                            title=f"OVRO-LWA {band} I deep {args.variant}")
            print("wrote band PNG", band, flush=True)

    rgb = [hp('Red'), hp('Green'), hp('Blue')]
    if all(os.path.exists(p) for p in rgb):
        render_3color(rgb, os.path.join(args.out, f"3color_{args.variant}.png"),
                      args.width, args.frame,
                      title=f"OVRO-LWA 3-color {args.variant}",
                      full_hpx_path=hp('Full'),
                      out_png_lum=os.path.join(args.out, f"3color_{args.variant}_fulllum.png"))
        write_rgb_healpix(rgb, os.path.join(args.out,
                          f"healpix_3color_{args.variant}_nside{args.nside}.fits"))
        print("wrote 3-color PNGs + RGB HEALPix FITS", flush=True)
