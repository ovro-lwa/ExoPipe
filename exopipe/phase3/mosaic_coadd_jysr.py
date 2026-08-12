#!/usr/bin/env python
"""
mosaic_coadd_jysr.py -- unit-correct (surface brightness) HEALPix co-add.

Canonical mosaic architecture (Gregg rulings 2026-07-11):
  1. Convert each tile from Jy/beam to Jy/sr BEFORE co-adding, using the analytic
     direction-dependent beam solid angle. In our zenith-centered SIN masters the PSF is
     shift-invariant in PIXEL space (established empirically, rank-48 method), so on the
     sky Omega_b(l,m) = Omega_zenith / n with n = sqrt(1 - l^2 - m^2) and Omega_zenith
     from the header restoring beam. Value-preserving (bilinear) interpolation then
     correctly preserves surface brightness through every regridding.
  2. Weights stay PER-BEAM inverse variance (PB^2/sigma_beam^2) -- point-source-optimal;
     an unbiased (if not min-variance-for-diffuse) surface-brightness mean. Deliberately
     NOT inverse variance in the Jy/sr domain, which would up-weight the elongated
     low-elevation beams.
  3. Companion product: effective-PSF map, Omega_eff = sum(w*Omega_b)/sum(w) per HEALPix
     pixel. Point-source flux = map(Jy/sr) x Omega_eff for unresolved sources.

Two modes:
  - native: co-add straight from the staged pbcorr masters (no alignment; iter0-style).
  - rebuild: consume an existing build_aligned_mosaic workroot -- read the LAST
    iteration's tgt_dewarp.fits (aligned tiles, Jy/beam, on the tile-zenith-centered
    common grid), convert to Jy/sr there, co-add. Reuses GPU flow warps at zero cost.
    (Dewarp moves emission to its true position, so evaluating Omega_b at the post-warp
    position is correct.)
"""
import os, glob, sys
import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp, reproject_to_healpix

NSIDE = 2048
DEC_FLOOR = -40.0
PBCORR_FLOOR = 0.10
PB_FEATHER = 0.15       # weight cosine-feathered to 0 over PB 0.15->0.10 (Gregg 2026-07-17)
N_FLOOR = 0.05          # cos(z) floor; weights are ~0 out there anyway (PB<=0.1 masked)
FWHM2OMEGA = np.pi / (4 * np.log(2))
K_B = 1.380649e-23; C_LIGHT = 2.99792458e8

MASTER_ROOT_DEFAULT = "/fast/gh/main/phase3/mosaic/masters"
GRID_N = 4200; CDELT = 0.0283   # must match build_aligned_mosaic common grid

# Optional per-cell beam override (subband tier: headers unreliable until the deferred
# repair fires on FLEET_ALL_DONE; the pipeline's dry-run census carries the repair
# values -- MOSAIC_INPUTS_STATUS 2026-07-11). Keyed '{lst}_{band}|{variant_name}'.
BEAM_OVERRIDE = {}
_PRODUCT_VARIANT = {'Vclean': 'V_deep_Taper_R0', 'Mstruct': 'V_deep_Taper_R0'}
SKIP_CELLS = set()   # '{lst}_{band}|{variant_name}' from the corrupt-variant census


def load_skip_csv(path):
    import csv as _csv
    for r in _csv.DictReader(open(path)):
        SKIP_CELLS.add(f"{r['cell']}|{r['variant']}")
    print(f"corrupt-variant skip list: {len(SKIP_CELLS)} cell-variants", flush=True)


def load_beam_csv(path):
    import csv as _csv
    n = 0
    for r in _csv.DictReader(open(path)):
        if r.get('bmaj_new_am') and r.get('status') == 'ok':
            BEAM_OVERRIDE[f"{r['cell']}|{r['variant']}"] = (
                float(r['bmaj_new_am']) / 60.0, float(r['bmin_new_am']) / 60.0)
            n += 1
    print(f"beam override loaded: {n} cell-variants from {path}", flush=True)


def omega_for(lst, band, variant, product, hdr):
    vname = _PRODUCT_VARIANT.get(product, f"I_deep_{variant}")
    bo = BEAM_OVERRIDE.get(f"{lst}_{band}|{vname}")
    if bo:
        return FWHM2OMEGA * bo[0] * bo[1] * (np.pi / 180.0) ** 2
    return omega_zen(hdr)


def _paths(root, lst, variant, band, product='I'):
    """product='I' -> I_deep_{variant}_pbcorr; 'Vclean' -> the cleaned Stokes-V master
    (sidelobe/bilobe/striation-subtracted, V convention = Taper_R0; variant ignored).
    V epoch images are DIRTY (wsclean -niter 0): the repaired header beam is the fitted
    dirty-beam lobe — a self-consistent Omega for peak photometry (the only meaningful
    photometry on dirty maps)."""
    if product == 'Vclean':
        name = f"V_deep_Taper_R0_clean_pbcorr_{band}.fits"
    elif product == 'Mstruct':
        # handled specially in tile_jysr_hpx: (raw V - cleaned V)/PB = the subtracted
        # sidelobe+bilobe+striation model (Gregg veto design 2026-07-12); this path
        # returns the raw-V master (its header carries the beam).
        name = f"V_deep_Taper_R0_{band}.fits"
    else:
        name = f"I_deep_{variant}_pbcorr_{band}.fits"
    return (os.path.join(root, lst, name),
            os.path.join(root, lst, f"PB_deep_{band}.fits"))


def _common_wcs_tile(lst):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---SIN', 'DEC--SIN']
    w.wcs.crval = [(15.0 * (int(lst[:2]) + 0.5)) % 360.0, 37.2158]
    w.wcs.crpix = [GRID_N / 2, GRID_N / 2]
    w.wcs.cdelt = [-CDELT, CDELT]
    return w


def n_map(w, shape):
    """cos(zenith angle) per pixel for a zenith-centered SIN grid: n=sqrt(1-l^2-m^2)."""
    ny, nx = shape
    x = np.arange(nx) + 1.0 - w.wcs.crpix[0]
    y = np.arange(ny) + 1.0 - w.wcs.crpix[1]
    l = np.radians(x * w.wcs.cdelt[0])[None, :] * np.ones((ny, 1), np.float32)
    m = np.radians(y * w.wcs.cdelt[1])[:, None] * np.ones((1, nx), np.float32)
    r2 = l * l + m * m
    return np.sqrt(np.clip(1.0 - r2, N_FLOOR**2, 1.0)).astype(np.float32)


def omega_zen(hdr):
    """Zenith restoring-beam solid angle [sr] from header BMAJ/BMIN (deg)."""
    return FWHM2OMEGA * hdr['BMAJ'] * hdr['BMIN'] * (np.pi / 180.0) ** 2


def _tile_freq(hdr):
    for k in ('RESTFRQ', 'RESTFREQ', 'CRVAL3'):
        if k in hdr and hdr[k] and hdr[k] > 1e6:
            return hdr[k]
    return np.nan


def tile_jysr_hpx(root, lst, variant, band, workroot=None, last_it=None, product='I',
                  domain='jysr'):
    """One tile -> (hpx signal, omega_zen, freq). rebuild mode if workroot.
    domain='jysr' (canonical surface brightness) or 'jybeam' (apparent-flux detection
    domain, 2026-07-27: pbcorr x PB, no Omega division — the peak observable reads
    apparent flux; intrinsic = apparent / PB_eff from the companion map)."""
    mp, pbp = _paths(root, lst, variant, band, product)
    mhdr = fits.getheader(mp)
    om0 = omega_for(lst, band, variant, product, mhdr); freq = _tile_freq(mhdr)
    if domain == 'jybeam':
        if workroot or product == 'Mstruct':
            raise ValueError("jybeam domain: native mode, I/Vclean products only")
        d = np.nan_to_num(fits.getdata(mp).astype(np.float32).squeeze())
        pb = np.nan_to_num(fits.getdata(pbp).astype(np.float32).squeeze())
        d = d * pb                                      # intrinsic -> apparent Jy/beam
        w = WCS(mhdr).celestial
        cw = _common_wcs_tile(lst)
        arr, _ = reproject_interp((d, w), cw, shape_out=(GRID_N, GRID_N), order='bilinear')
        hpx, _ = reproject_to_healpix((np.nan_to_num(arr).astype(np.float32), cw), 'icrs',
                                      nside=NSIDE, order='bilinear')
        return np.nan_to_num(hpx), om0, freq
    if workroot:
        dw = os.path.join(workroot, f"it{last_it}_{lst}", "tgt_dewarp.fits")
        arr = np.nan_to_num(fits.getdata(dw).astype(np.float32).squeeze())
        cw = _common_wcs_tile(lst)
        arr = arr * n_map(cw, arr.shape) / om0          # Jy/beam -> Jy/sr (post-warp grid)
        hpx, _ = reproject_to_healpix((arr, cw), 'icrs', nside=NSIDE, order='bilinear')
        return np.nan_to_num(hpx), om0, freq
    d = np.nan_to_num(fits.getdata(mp).astype(np.float32).squeeze())
    if product == 'Mstruct':
        cell = os.path.dirname(mp)
        clean = np.nan_to_num(fits.getdata(
            os.path.join(cell, f"V_deep_Taper_R0_clean_{band}.fits")).astype(np.float32).squeeze())
        pb = np.nan_to_num(fits.getdata(
            os.path.join(cell, f"PB_deep_{band}.fits")).astype(np.float32).squeeze())
        d = np.where(pb > PBCORR_FLOOR, (d - clean) / np.maximum(pb, PBCORR_FLOOR), 0.0)
    w = WCS(mhdr).celestial
    d = d * n_map(w, d.shape) / om0                     # Jy/beam -> Jy/sr (native grid)
    cw = _common_wcs_tile(lst)
    arr, _ = reproject_interp((d, w), cw, shape_out=(GRID_N, GRID_N), order='bilinear')
    hpx, _ = reproject_to_healpix((np.nan_to_num(arr).astype(np.float32), cw), 'icrs',
                                  nside=NSIDE, order='bilinear')
    return np.nan_to_num(hpx), om0, freq


def coadd_jysr(lsts, variant, band, root=MASTER_ROOT_DEFAULT, workroot=None, last_it=None,
               out_dir=None, tag='', product='I', domain='jysr'):
    """PB^2/sigma_beam^2-weighted co-add + effective-PSF (Omega_eff) map.
    domain='jybeam' -> apparent Jy/beam detection map + PB_eff map instead
    (Omega_eff identical to the jysr pair's psfmap — same weights — so not rewritten)."""
    npix = hp.nside2npix(NSIDE)
    num = np.zeros(npix); den = np.zeros(npix); psf_num = np.zeros(npix)
    pb_num = np.zeros(npix)
    freq = np.nan
    order = sorted(lsts, key=lambda l: int(l[:2]))
    for lst in order:
        mp, pbp = _paths(root, lst, variant, band, product)
        vname = _PRODUCT_VARIANT.get(product, f"I_deep_{variant}")
        if f"{lst}_{band}|{vname}" in SKIP_CELLS:
            print(f"[{lst}] CORRUPT (census) -- skipped", flush=True)
            continue
        if not (os.path.exists(mp) and os.path.exists(pbp)):
            print(f"[{lst}] MISSING -- skipped", flush=True)
            continue
        sig_hpx, om0, freq = tile_jysr_hpx(root, lst, variant, band, workroot, last_it,
                                           product, domain)
        pb = np.nan_to_num(fits.getdata(pbp).squeeze().astype(np.float32))
        wpb = WCS(fits.getheader(pbp)).celestial
        if product == 'Mstruct':
            # weights must match the V mosaic exactly: sigma from the CLEANED apparent V
            capp = np.nan_to_num(fits.getdata(os.path.join(
                os.path.dirname(mp), f"V_deep_Taper_R0_clean_{band}.fits")).squeeze().astype(np.float32))
            app = capp[pb > PBCORR_FLOOR]
        else:
            intr = np.nan_to_num(fits.getdata(mp).squeeze().astype(np.float32))
            app = (intr * pb)[pb > PBCORR_FLOOR]        # apparent-domain per-beam noise
        app = app[np.isfinite(app)]
        if app.size < 100:
            print(f"[{lst}] no valid pixels -- skipped", flush=True)
            continue
        med = np.median(app); sigma = 1.4826 * np.median(np.abs(app - med)) or 1e-6
        # all-zero/flat master guard (2026-07-18): a corrupt flat tile's sigma hits the
        # 1e-6 floor -> ~1e12 weight -> its zeros wipe the mosaic across its footprint
        # (staged 15h-19h NoTaper_R0 Blue/Full did exactly this). Real apparent-domain
        # MAD is never below ~1 mJy/beam.
        if not np.isfinite(sigma) or sigma < 1e-4:
            print(f"[{lst}] CORRUPT-FLAT (sigma={sigma:.2e}) -- skipped", flush=True)
            continue
        wmap = np.where(pb > PBCORR_FLOOR, pb ** 2 / sigma ** 2, 0.0).astype(np.float32)
        # PB-rim feather (Gregg-approved 2026-07-17): cosine-taper each tile's weight
        # to zero over PB 0.15->0.10 so tiles fade out BEFORE the pbcorr mask edge.
        # Removes the hard mask-exit weight steps + 1/PB-amplified rim contributions
        # that produced the dark sawtooth arcs along the southern boundary.
        ramp = np.clip((pb - PBCORR_FLOOR) / (PB_FEATHER - PBCORR_FLOOR), 0.0, 1.0)
        wmap *= (0.5 - 0.5 * np.cos(np.pi * ramp)).astype(np.float32)
        omap = (om0 / n_map(wpb, pb.shape)).astype(np.float32)   # local beam solid angle
        w_hpx, _ = reproject_to_healpix((wmap, wpb), 'icrs', nside=NSIDE, order='bilinear')
        wo_hpx, _ = reproject_to_healpix(((wmap * omap), wpb), 'icrs', nside=NSIDE,
                                         order='bilinear')
        w_hpx = np.nan_to_num(w_hpx); wo_hpx = np.nan_to_num(wo_hpx)
        num += w_hpx * sig_hpx; den += w_hpx; psf_num += wo_hpx
        if domain == 'jybeam':
            wp_hpx, _ = reproject_to_healpix(((wmap * pb), wpb), 'icrs', nside=NSIDE,
                                             order='bilinear')
            pb_num += np.nan_to_num(wp_hpx)
        print(f"[{lst}] added (omega_zen={om0:.3e} sr)", flush=True)
    if not np.any(den > 0):
        raise SystemExit(f"NO TILES CONTRIBUTED ({variant} {band} {product} {domain}) "
                         "-- check root/variant/band arguments")
    th, _ = hp.pix2ang(NSIDE, np.arange(npix))
    good = (den > 0) & ((90.0 - np.degrees(th)) >= DEC_FLOOR)
    mos = np.full(npix, np.nan, np.float32); psf = np.full(npix, np.nan, np.float32)
    mos[good] = (num[good] / den[good]).astype(np.float32)
    psf[good] = (psf_num[good] / den[good]).astype(np.float32)
    if domain == 'jybeam':
        pbeff = np.full(npix, np.nan, np.float32)
        pbeff[good] = (pb_num[good] / den[good]).astype(np.float32)
        psf = pbeff   # returned second product in jybeam mode
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        jy2k = 1e-26 * C_LIGHT ** 2 / (2 * K_B * freq ** 2) if np.isfinite(freq) else 0.0
        base = [('PIXTYPE', 'HEALPIX'), ('ORDERING', 'RING'), ('NSIDE', NSIDE),
                ('COORDSYS', 'C'), ('INDXSCHM', 'IMPLICIT'),
                ('PBCORR', domain != 'jybeam'),
                ('WEIGHTS', 'PB^2/sigma_beam^2 (per-beam, point-source-optimal)'),
                ('FEATHER', f'PB {PB_FEATHER}->{PBCORR_FLOOR} cosine weight taper'),
                ('ALIGNED', bool(workroot)), ('FREQ', freq)]
        stem = {'Vclean': f"V_deep_Taper_R0_clean_{band}",
                'Mstruct': f"Mstruct_Taper_R0_{band}"}.get(product,
                                                           f"I_deep_{variant}_{band}")
        if domain == 'jybeam':
            outputs = (
                (mos, f"healpix_{stem}{tag}_jybeam_nside{NSIDE}.fits",
                 'Jy/beam', [('COMMENT', 'APPARENT-flux detection map (PB as weight '
                                         'only, never divided out): SNR-optimal for '
                                         'unresolved sources.'),
                             ('COMMENT', 'intrinsic apparent flux = SIGNAL / PB_eff '
                                         '(companion PBeff map).'),
                             ('PSFMAP', f"healpix_PSF_{stem}_nside{NSIDE}.fits",
                              'Omega_eff identical to jysr pair (same weights)')]),
                (psf, f"healpix_PBeff_{stem}{tag}_nside{NSIDE}.fits",
                 '', [('COMMENT', 'PB_eff = sum(w*PB)/sum(w); intrinsic = '
                                  'apparent / PB_eff at any position.')]))
        else:
            outputs = (
                (mos, f"healpix_{stem}{tag}_jysr_nside{NSIDE}.fits",
                 'Jy/sr', [('JY2K', jy2k, 'K per (Jy/sr) at FREQ')]),
                (psf, f"healpix_PSF_{stem}{tag}_nside{NSIDE}.fits",
                 'sr', [('COMMENT', 'Omega_eff = sum(w*Omega_b)/sum(w); point-source '
                                    'flux [Jy] = I[Jy/sr] * Omega_eff')]))
        for arr, name, bunit, extra in outputs:
            hdu = fits.BinTableHDU.from_columns(
                [fits.Column(name='SIGNAL', format='E', array=arr)])
            for k, v in base + [('BUNIT', bunit)]:
                hdu.header[k] = v
            for e in extra:
                if e[0] == 'COMMENT':
                    hdu.header.add_comment(e[1])
                else:
                    hdu.header[e[0]] = (e[1], e[2])
            outp = os.path.join(out_dir, name)
            fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(outp, overwrite=True)
            print(f"WROTE {outp}", flush=True)
    return mos, psf


def last_iteration(workroot, lsts):
    its = [int(os.path.basename(d).split('_')[0][2:])
           for d in glob.glob(os.path.join(workroot, 'it*'))
           if os.path.exists(os.path.join(d, 'tgt_dewarp.fits'))]
    return max(its) if its else None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='Taper_R075')
    ap.add_argument('--band', default='Blue')
    ap.add_argument('--lsts', nargs='+', default=[f"{h:02d}h" for h in range(23)])
    ap.add_argument('--root', default=MASTER_ROOT_DEFAULT)
    ap.add_argument('--workroot', default=None, help='rebuild from this alignment workroot')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--product', default='I', choices=['I', 'Vclean', 'Mstruct'])
    ap.add_argument('--domain', default='jysr', choices=['jysr', 'jybeam'],
                    help='jybeam = apparent-flux detection map + PB_eff (2026-07-27)')
    ap.add_argument('--beam_csv', default=None,
                    help='dry-run census CSV with bmaj_new_am/bmin_new_am overrides')
    ap.add_argument('--skip_csv', default=None, help='corrupt-variant census to exclude')
    a = ap.parse_args()
    if a.beam_csv:
        load_beam_csv(a.beam_csv)
    if a.skip_csv:
        load_skip_csv(a.skip_csv)
    li = last_iteration(a.workroot, a.lsts) if a.workroot else None
    if a.workroot and li is None:
        sys.exit("workroot given but no completed iteration found")
    coadd_jysr(a.lsts, a.variant, a.band, a.root, a.workroot, li, a.out, a.tag, a.product,
               a.domain)
