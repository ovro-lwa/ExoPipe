#!/usr/bin/env python3
"""
dipole_lib.py — TRUE-dipole finder + local-PA-coherence, for bilobe veto Parts 2-3.
A true dipole = a COMPARABLE-amplitude opposite-sign peak pair at the characteristic
separation (0.5-2.5 beam). Uses adaptive LOCAL rms (handles the flux-scale ramp) so
peak significance is correct across the field. This is the population map the
per-candidate veto queries for (Part 2) count and (Part 3) local PA coherence.
"""
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

SEP_LO, SEP_HI = 0.5, 1.5   # SEP_HI 2.5->1.5 (Gregg 2026-06-24): Red bilobes all <=1.3 beam
AMP_LO = 0.5                 # comparable-amplitude clamp (true dipole, not sidelobe)


def lobe_centroid(d, x, y, sign, beam):
    """Flux-weighted sub-pixel centroid of the same-sign lobe core in a 0.6-beam box.
    Returns (cx, cy) in pixel coords (falls back to the integer peak)."""
    r = max(2, int(round(0.6 * beam))); H, W = d.shape
    y0, y1, x0, x1 = max(0, y-r), min(H, y+r+1), max(0, x-r), min(W, x+r+1)
    sub = d[y0:y1, x0:x1].astype(float)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    w = np.clip(sub * sign, 0, None)
    if w.size == 0 or w.max() <= 0:
        return float(x), float(y)
    w = np.where(w > 0.3 * w.max(), w, 0.0); s = w.sum()
    if s <= 0:
        return float(x), float(y)
    return float((w*xx).sum()/s), float((w*yy).sum()/s)


def sky_pa(wcs, x1, y1, x2, y2):
    """Sky-frame position angle (E of N, mod 180) of the (x1,y1)->(x2,y2) axis."""
    c1 = wcs.pixel_to_world(x1, y1); c2 = wcs.pixel_to_world(x2, y2)
    return float(c1.position_angle(c2).deg) % 180

def local_rms_map(d, n=12):
    H, W = d.shape; r = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            b = d[i*H//n:(i+1)*H//n, j*W//n:(j+1)*W//n]; b = b[np.isfinite(b) & (b != 0)]
            if b.size > 50: r[i, j] = 1.4826*np.median(np.abs(b - np.median(b)))
    return r

def find_true_dipoles(d, beam, sig=4.0, nrms=12, wcs=None):
    """Return array of (y, x, PA_deg, ratio) for comparable opposite-sign pairs.
    PA = position angle of the +→- axis (mod 180), in the SKY frame from sub-pixel lobe
    centroids when `wcs` is given (else pixel-frame from integer peaks — legacy fallback).
    Local-rms-adaptive peaks; (y,x) is the centroid midpoint."""
    d = np.nan_to_num(d); H, W = d.shape
    rm = local_rms_map(d, nrms)
    def lr(y, x): return rm[min(nrms-1, y*nrms//H), min(nrms-1, x*nrms//W)]
    b = max(int(round(beam)), 2)
    pmax = maximum_filter(d, b); pmin = minimum_filter(d, b)
    pos = np.argwhere((d == pmax)); neg = np.argwhere((d == pmin))
    pos = np.array([p for p in pos if d[p[0], p[1]] > sig*lr(p[0], p[1])]) if len(pos) else pos
    neg = np.array([p for p in neg if d[p[0], p[1]] < -sig*lr(p[0], p[1])]) if len(neg) else neg
    if len(pos) == 0 or len(neg) == 0:
        return np.empty((0, 4))
    out = []
    for (py, px) in pos:
        dd = np.hypot(neg[:, 0]-py, neg[:, 1]-px)
        within = np.where((dd > SEP_LO*beam) & (dd < SEP_HI*beam))[0]
        best = None
        for k in within:
            ny, nx = neg[k]
            ap = d[py, px]; an = -d[ny, nx]
            ratio = min(ap, an)/max(ap, an)
            if ratio >= AMP_LO and (best is None or dd[k] < best[2]):
                best = (int(ny), int(nx), dd[k], ratio)
        if best is not None:
            ny, nx, _, ratio = best
            cxp, cyp = lobe_centroid(d, px, py, +1, beam)      # + lobe centroid
            cxn, cyn = lobe_centroid(d, nx, ny, -1, beam)      # - lobe centroid
            if wcs is not None:
                pa = sky_pa(wcs, cxp, cyp, cxn, cyn)           # sky-frame +->-
            else:
                pa = np.degrees(np.arctan2(cxn-cxp, cyn-cyp)) % 180
            out.append(((cyp+cyn)/2.0, (cxp+cxn)/2.0, pa, ratio))
    return np.array(out) if out else np.empty((0, 4))

def circ_R(pa_deg):
    """Circular concentration (0..1) of PAs (mod 180 -> double-angle)."""
    if len(pa_deg) == 0: return np.nan, np.nan
    a = np.radians(2*np.asarray(pa_deg))
    C, S = np.mean(np.cos(a)), np.mean(np.sin(a))
    R = np.hypot(C, S); mean = (np.degrees(0.5*np.arctan2(S, C))) % 180
    return R, mean

def pa_diff(a, b):
    """Smallest difference between two PAs (mod 180), in degrees."""
    d = abs((a - b) % 180); return min(d, 180-d)
