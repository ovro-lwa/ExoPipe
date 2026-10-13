#!/usr/bin/env python
"""
verify_mosaic.py -- flux-scale + astrometry verification of the aligned HEALPix mosaics.

Flux: measure each Scaife-Heald 3C calibrator's peak (Jy/beam ~ point-source flux) in the
band mosaic and compare to the model flux at the band frequency (ratio measured/model).
Excludes the A-team (peeled).

Astrometry: cross-match bright unresolved detections to the VLSSr catalog and report the
RA/Dec residuals (median + robust scatter). Blue is the anchor band (73 MHz vs VLSSr).
"""
import numpy as np, healpy as hp
import astropy.coordinates as ac, astropy.units as u
from exopipe.common import pipeline_config as pc

BAND_FREQ = {'Red': 29.0, 'Green': 50.0, 'Blue': 74.0, 'Full': 60.0}  # approx band centres MHz
VLSSR = "/lustre/gh/calibration/pipeline/reference/surveys/vlssr_radecpeak_unresolved.txt"


def model_flux(coeffs, freq_mhz, scale='SH12'):
    if scale == 'SH12':   # Scaife & Heald 2012: S = A0 * 10^(A1 x + A2 x^2 + ...), A0=flux@150
        x = np.log10(freq_mhz / 150.0)
        log_s = np.log10(coeffs[0]) + sum(c * x**i for i, c in enumerate(coeffs) if i >= 1)
    else:                 # Perley-Butler 2017: log10(S) = sum a_i (log10 nu_GHz)^i
        x = np.log10(freq_mhz / 1000.0)
        log_s = sum(c * x**i for i, c in enumerate(coeffs))
    return 10 ** log_s


def peak_near(m, ra, dec, rad_deg=0.25):
    v = hp.ang2vec(ra, dec, lonlat=True)
    pix = hp.query_disc(2048, v, np.radians(rad_deg))
    vals = m[pix]
    return np.nanmax(vals) if np.isfinite(vals).any() else np.nan


def flux_check(band, path):
    m = hp.read_map(path, verbose=False)
    freq = BAND_FREQ[band]
    print(f"\n== FLUX CHECK {band} @ {freq:.0f} MHz (measured peak Jy/beam vs SH model) ==")
    ratios = []
    for name, d in pc.CALIB_DATA.items():
        c = d['coords']
        pk = peak_near(m, c.ra.deg, c.dec.deg)
        mod = model_flux(d['coeffs'], freq, d.get('scale', 'SH12'))
        if np.isfinite(pk):
            ratios.append(pk / mod)
            print(f"  {name:6s} meas={pk:7.2f}  model={mod:7.2f}  ratio={pk/mod:.3f}")
    if ratios:
        ratios = np.array(ratios)
        print(f"  --> median ratio={np.median(ratios):.3f}  scatter={np.std(ratios):.3f} "
              f"(N={len(ratios)})")


def astrometry_check(band, path, nsrc=400, minflux=8.0):
    m = hp.read_map(path, verbose=False)
    cat = np.loadtxt(VLSSR, skiprows=1)
    cat = cat[cat[:, 2] >= minflux]          # bright unresolved
    cat = cat[(cat[:, 1] > -25) & (cat[:, 1] < 85)]  # in coverage
    if len(cat) > nsrc:
        cat = cat[np.argsort(-cat[:, 2])[:nsrc]]
    dra, ddec = [], []
    for ra0, dec0, _ in cat:
        # centroid of the mosaic peak within 6' of the VLSSr position
        v = hp.ang2vec(ra0, dec0, lonlat=True); pix = hp.query_disc(2048, v, np.radians(0.1))
        vals = np.nan_to_num(m[pix]); vals = np.clip(vals - np.median(vals), 0, None)
        if vals.sum() <= 0:
            continue
        th, ph = hp.pix2ang(2048, pix); pra = np.degrees(ph); pdec = 90 - np.degrees(th)
        w = vals / vals.sum()
        cra = ((w * ((pra - ra0 + 180) % 360 - 180)).sum())
        cdec = (w * (pdec - dec0)).sum()
        dra.append(cra * np.cos(np.radians(dec0)) * 60); ddec.append(cdec * 60)  # arcmin
    dra, ddec = np.array(dra), np.array(ddec)
    print(f"\n== ASTROMETRY {band} vs VLSSr (N={len(dra)} matches, arcmin) ==")
    print(f"  dRA:  median={np.median(dra):+.2f}  MAD={1.4826*np.median(np.abs(dra-np.median(dra))):.2f}")
    print(f"  dDec: median={np.median(ddec):+.2f}  MAD={1.4826*np.median(np.abs(ddec-np.median(ddec))):.2f}")
    sep = np.hypot(dra, ddec)
    print(f"  |offset|: median={np.median(sep):.2f}  90th={np.percentile(sep,90):.2f}  max={sep.max():.2f}")
    return dra, ddec, cat


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--hpx-dir', default='/fast/gh/main/phase3/mosaic/healpix_aligned_v2')
    ap.add_argument('--variant', default='NoTaper_R075')
    ap.add_argument('--suffix', default='_aligned')
    ap.add_argument('--bands', nargs='+', default=['Red', 'Green', 'Blue', 'Full'])
    a = ap.parse_args()
    for b in a.bands:
        p = f"{a.hpx_dir}/healpix_I_deep_{a.variant}_{b}{a.suffix}_nside2048.fits"
        flux_check(b, p)
    for b in a.bands:
        p = f"{a.hpx_dir}/healpix_I_deep_{a.variant}_{b}{a.suffix}_nside2048.fits"
        astrometry_check(b, p)
