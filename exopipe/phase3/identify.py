"""
exopipe.phase3.identify — seed module (refactor step 2, extracted 2026-08-11).

Origin: /lustre/gh/main/clv15/phase3_final_metric.py (VERBATIM extraction of
fit_empirical_beam). fit_empirical_beam had 8 production importers in clv15.
The remaining identification-capability content arrives at move time.
"""
import numpy as np
from scipy.optimize import curve_fit


def fit_empirical_beam(img, nsrc=60, fmin=4.0, fmax=60.0):
    """Median Gaussian FWHM (pixels) of bright unresolved sources."""
    n = img.shape[0]
    q = n // 4
    w = np.where(np.isfinite(img), img, -np.inf).copy()
    w[:q, :] = w[3 * q:, :] = -np.inf
    w[:, :q] = w[:, 3 * q:] = -np.inf
    majs, mins_, pas = [], [], []
    tried = 0
    while len(majs) < nsrc and tried < 300:
        iy, ix = np.unravel_index(np.argmax(w), w.shape)
        v = w[iy, ix]
        if v < fmin:
            break
        w[iy - 12:iy + 12, ix - 12:ix + 12] = -np.inf
        tried += 1
        if v > fmax:
            continue
        r = 7
        st = img[iy - r:iy + r + 1, ix - r:ix + r + 1]
        if st.shape != (2 * r + 1, 2 * r + 1) or not np.isfinite(st).all():
            continue
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]

        def g(c, a, x0, y0, sx, sy, th, o):
            X, Y = c
            xr = (X - x0) * np.cos(th) + (Y - y0) * np.sin(th)
            yr = -(X - x0) * np.sin(th) + (Y - y0) * np.cos(th)
            return (a * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
                    + o).ravel()
        try:
            p, _ = curve_fit(g, (xx, yy), st.ravel(),
                             p0=[v, 0, 0, 2.2, 2.2, 0.8, 0], maxfev=8000)
        except Exception:
            continue
        sx, sy = abs(p[3]), abs(p[4])
        if not (0.8 < sx < 8 and 0.8 < sy < 8):
            continue
        majs.append(max(sx, sy) * 2.3548)
        mins_.append(min(sx, sy) * 2.3548)
        pas.append(np.rad2deg(p[5]) % 180)

    # Keep UNRESOLVED sources only (per Gregg): clip fits whose area
    # exceeds 1.25x the median fitted area — resolved sources bias the
    # beam estimate broad. One clipping pass is sufficient with a
    # median-based reference.
    majs = np.array(majs); mins_ = np.array(mins_); pas = np.array(pas)
    if len(majs) > 10:
        area = majs * mins_
        keep = area <= 1.25 * np.median(area)
        majs, mins_, pas = majs[keep], mins_[keep], pas[keep]
    return (float(np.median(majs)), float(np.median(mins_)),
            float(np.median(pas)), len(majs))

