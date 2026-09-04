#!/usr/bin/env python3
"""
phase3_astrom_check.py — VLSSr astrometry acceptance check for a rebuilt
co-add cell (replicates the COADD_ASTROMETRIC_ERROR.md measurement):
brightest compact local maxima centroided and matched to VLSSr; reports the
median separation per band. Acceptance (healthy tiles): <= ACCEPT_AMIN.
Usage: python phase3_astrom_check.py <LST> [accept_arcmin=2.5]
Exit 0 = all bands pass; 1 = any band fails/missing.
"""
from exopipe.common.roots import coadd_root
import sys
import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import maximum_filter, center_of_mass

VLSSR = ('/lustre/gh/calibration/pipeline/reference/surveys/'
         'vlssr_radecpeak_unresolved.txt')


def main():
    lst = sys.argv[1]
    ACCEPT = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5

    v = np.loadtxt(VLSSR, skiprows=1)
    vcat = SkyCoord(v[:, 0] * u.deg, v[:, 1] * u.deg)

    ok = True
    for band in ('Blue', 'Green', 'Red', 'Full'):
        mp = (f'{coadd_root()}/{lst}_{band}/coadd_final/'
              f'I_deep_Taper_R075_{band}.fits')
        if not os.path.exists(mp):
            print(f'{lst} {band}: NO MASTER'); ok = False; continue
        a = fits.getdata(mp).squeeze().astype(float)
        a[~np.isfinite(a)] = 0.0
        h = fits.getheader(mp); w = WCS(h).celestial
        n = a.shape[0]; q = n // 8
        cen = a[q:-q, q:-q]                      # avoid the low-elevation rim
        mf = maximum_filter(cen, size=25)
        py, px = np.where((cen == mf) & (cen > 0))
        fx = cen[py, px]
        top = np.argsort(fx)[::-1][:60]
        seps = []
        for i in top:
            y, x = int(py[i]) + q, int(px[i]) + q
            box = a[y - 4:y + 5, x - 4:x + 5]
            if not np.all(np.isfinite(box)) or box.max() <= 0:
                continue
            cy, cx = center_of_mass(np.maximum(box, 0))
            sky = w.pixel_to_world(x - 4 + cx, y - 4 + cy)
            d = float(sky.separation(vcat).arcmin.min())
            if d < 30:
                seps.append(d)
        med = float(np.median(seps)) if seps else np.inf
        tag = 'PASS' if med <= ACCEPT else 'FAIL'
        print(f'{lst} {band}: median VLSSr sep = {med:.2f} arcmin '
              f'(n={len(seps)}) [{tag}]')
        ok &= (med <= ACCEPT)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
