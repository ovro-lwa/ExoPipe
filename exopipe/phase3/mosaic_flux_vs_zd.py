#!/usr/bin/env python
"""
mosaic_flux_vs_zd.py -- flux-scale flatness vs zenith distance for a HEALPix mosaic.

The proof that the pre-refactor mosaics were APPARENT flux was the measured/VLSSr
ratio falling 0.61 (zenith) -> 0.15 (Dec -33). After the consume-_pbcorr refactor
(build_aligned_mosaic._coadd consumes the pipeline's intrinsic masters + PB_deep
weights) the ratio must be FLAT vs zenith distance. This measures it.

Method: bright unresolved VLSSr sources (74 MHz, matches Blue ~73 MHz), mosaic peak
within 0.25 deg, ratio binned by zenith distance |Dec - 37.2158| (meridian-drift
mosaic -> elevation is a function of Dec only). Blue is the meaningful band (no
spectral correction needed); other bands show a spectral-index offset but flatness
still holds if PB handling is right.
"""
import sys
import numpy as np, healpy as hp

VLSSR = "/lustre/gh/calibration/pipeline/reference/surveys/vlssr_radecpeak_unresolved.txt"
ZENITH_DEC = 37.2158
NSIDE = 2048


def run(path, minflux=8.0, out_csv=None):
    m = hp.read_map(path, verbose=False)
    cat = np.loadtxt(VLSSR, skiprows=1)
    cat = cat[(cat[:, 2] >= minflux) & (cat[:, 1] > -39.0) & (cat[:, 1] < 85.0)]
    rows = []
    for ra0, dec0, s_vlssr in cat:
        v = hp.ang2vec(ra0, dec0, lonlat=True)
        pix = hp.query_disc(NSIDE, v, np.radians(0.25))
        vals = m[pix]
        if not np.isfinite(vals).any():
            continue
        pk = np.nanmax(vals)
        if pk <= 0:
            continue
        rows.append((ra0, dec0, abs(dec0 - ZENITH_DEC), s_vlssr, pk, pk / s_vlssr))
    rows = np.array(rows)
    print(f"# {path}")
    print(f"# N={len(rows)} VLSSr matches (>= {minflux} Jy)")
    print(f"# {'zd_bin':>12s} {'N':>5s} {'med_ratio':>10s} {'MAD':>7s}")
    edges = [0, 10, 20, 30, 40, 50, 60, 70, 78]
    meds = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = rows[(rows[:, 2] >= lo) & (rows[:, 2] < hi)]
        if len(sel) < 3:
            continue
        r = sel[:, 5]
        med = np.median(r); mad = 1.4826 * np.median(np.abs(r - med))
        meds.append((0.5 * (lo + hi), med))
        print(f"  {lo:4d}-{hi:3d} deg {len(sel):5d} {med:10.3f} {mad:7.3f}")
    if len(meds) >= 2:
        meds = np.array(meds)
        rel = meds[:, 1] / meds[0, 1]
        print(f"# ratio vs innermost bin: min={rel.min():.2f} max={rel.max():.2f}"
              f"  (pre-refactor Blue fell to ~0.25 of zenith by Dec -33)")
        verdict = "FLAT (PASS)" if rel.min() > 0.7 else "FALLING (FAIL - still apparent-flux-like)"
        print(f"# VERDICT: {verdict}")
    if out_csv:
        hdr = (f"provenance: mosaic_flux_vs_zd.py on {path}\n"
               f"catalog: {VLSSR} (>= {minflux} Jy)\n"
               "ra_deg,dec_deg,zenith_dist_deg,vlssr_jy,mosaic_peak_jybm,ratio")
        np.savetxt(out_csv, rows, delimiter=",", header=hdr, fmt="%.4f")
        print(f"# wrote {out_csv}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--minflux", type=float, default=8.0)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    run(a.path, a.minflux, a.csv)
