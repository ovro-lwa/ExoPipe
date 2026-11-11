#!/usr/bin/env python
"""Regrid PILOT snapshot frames (4096 px at 1.875 arcmin, December 2024 before the 27th) to the Master grid
(1508 px at 4.6 arcmin, same SIN projection at zenith) so the flow and the search run on them like any other cell
(preamble E, Gregg 2026-09-18 approved).  Every *-image.fits in the staged subband directory whose NAXIS1 is not
the Master's is replaced IN THE STAGING (never the archive) by its regridded copy: bilinear resampling of the
Jy/beam frame onto the Master pixel grid through the two WCSs (astropy), NaN outside the source frame; the
header keeps the frame's own keys with the Master's NAXIS/CRPIX/CDELT/CRVAL and REGRID keys recording the
source grid.  Usage: phase3_snapshot_regrid.py <staged subband dir> <master fits> [--nproc N]"""
import argparse, glob, os, sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

ap = argparse.ArgumentParser(); ap.add_argument("stage_dir"); ap.add_argument("master"); ap.add_argument("--nproc", type=int, default=8)
a = ap.parse_args()
mh = fits.getheader(a.master)
MN = int(mh["NAXIS1"]); mw = WCS(mh).celestial
yy, xx = np.mgrid[0:MN, 0:MN]
ra, dec = mw.all_pix2world(xx.ravel(), yy.ravel(), 0)


def regrid(path):
    h = fits.getheader(path)
    if int(h["NAXIS1"]) == MN:
        return path, "already"
    d = np.squeeze(fits.getdata(path)).astype(np.float32)
    w = WCS(h).celestial
    px, py = w.all_world2pix(ra, dec, 0)
    out = map_coordinates(np.nan_to_num(d), [py.reshape(MN, MN), px.reshape(MN, MN)], order=1, mode="constant", cval=np.nan).astype(np.float32)
    bad = ~np.isfinite(px.reshape(MN, MN)) | ~np.isfinite(py.reshape(MN, MN))
    out[bad] = np.nan
    hh = h.copy()
    for k in ("NAXIS1", "NAXIS2", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2", "CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2"):
        if k in mh:
            hh[k] = mh[k]
    hh["REGRID"] = (True, "pilot frame resampled to the Master grid (phase3_snapshot_regrid)")
    hh["REGRIDN"] = (int(h["NAXIS1"]), "source NAXIS1"); hh["REGRIDCD"] = (float(h["CDELT2"]), "source CDELT2 deg")
    tmp = path + ".regrid.tmp"
    fits.PrimaryHDU(data=out, header=hh).writeto(tmp, overwrite=True)
    os.replace(tmp, path)
    return path, "regridded"


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(a.stage_dir, "*-image.fits")))
    from multiprocessing import Pool
    n = {"already": 0, "regridded": 0}
    with Pool(a.nproc) as p:
        for _, st in p.imap_unordered(regrid, files, chunksize=4):
            n[st] += 1
    print("REGRID-DONE %s: %d regridded, %d already on the Master grid" % (a.stage_dir, n["regridded"], n["already"]), flush=True)
