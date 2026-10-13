#!/usr/bin/env python3
"""
phase3_i_c50.py — blind injection-recovery for the Stokes-I search, per robust
(Gregg 2026-07-04: the R0-vs-R075 FP comparison is meaningless without matched
C50s). Mirrors phase3_iv_injloss for V, with I conventions.

Per (LST, band, date, robust):
  - build the search-chain epoch: R075 = stored I_sub + R075 master;
    R0 = Phase-2 R0 epoch -> guarded scale -> stage-B flow (constructed once)
  - N_FRAMES injection frames: N_PER point sources (empirical-beam shape,
    injected SNR uniform SNR_LO..SNR_HI x central residual rms, uniform random
    searchable positions, min separation)
  - full chain per frame: x2 refine (cached flow) -> subtract robust-matched
    master -> destripe -> detection (extract/local-SNR/merge) -> mstruct-I gate
  - record per injection: local snr, detected, passes mstruct
Outputs per-injection CSV to Run_I_GATES/c50/ + prints detect-C50 and mstruct-C50.
Usage: python phase3_i_c50.py <LST> <BAND> <DATE> <R0|R075>
"""
from exopipe.common.roots import coadd_root
import sys
_ARGV = list(sys.argv)
import os, csv, math, tempfile, shutil
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from exopipe.phase3.phase3_deep import _load, _residual
from exopipe.phase3.phase3_tenmin import _det_header
from exopipe.phase3.phase3_i_gate_ledger import _mstruct_i
from exopipe.phase3.identify import fit_empirical_beam
from exopipe.phase3.core.residuals import gaussian_stamp
from exopipe.phase3.coadd_full_pipeline import (discover_dates, find_wideband_deep_images,
                                 load_image, measure_flux_scale, apply_flow_field,
                                 regrid_image, get_midpoint_time)
from exopipe.phase3 import search_i_transients as sit

OUT = '/data07/gh/phase3/transients/Run_I_GATES/c50'
N_PER, N_FRAMES = 120, 3
SNR_LO, SNR_HI = 4.0, 16.0
MIN_SEP = 60


def central_rms(a):
    n = a.shape[0]
    c = a[n // 4:3 * n // 4, n // 4:3 * n // 4]
    c = c[np.isfinite(c) & (c != 0)]
    return 1.4826 * float(np.median(np.abs(c - np.median(c))))


def build_epoch(lst, band, date, robust, master, mh):
    base = f'{coadd_root()}/{lst}_{band}'
    if robust == 'R075':
        return _load(f'{base}/subtracted/{date}_I_sub_{band}.fits') + master
    sd = dict((d, s) for d, s, _ in discover_dates(lst, band)).get(date)
    wb = find_wideband_deep_images(sd, band, 'I', 'Taper_R0')
    ep, ehdr = load_image(wb)
    if ep.shape != master.shape:
        td = tempfile.mkdtemp()
        ep, ehdr = load_image(regrid_image(wb, mh, f'{td}/rg.fits'))
        shutil.rmtree(td, ignore_errors=True)
    hmask = ~np.isfinite(master) | (master == 0)
    emask = ~np.isfinite(ep) | (ep == 0)
    wcsm = WCS(mh).celestial
    cd = abs(mh['CDELT1']); bp = mh.get('BMAJ', 5 * cd) / cd
    _, coeffs, ns = measure_flux_scale(ep, master, wcsm, bp, hmask | emask,
                                       get_midpoint_time(ehdr, wb))
    h, w = master.shape
    yg, xg = np.indices((h, w), dtype=float)
    sm = coeffs[0] + coeffs[1] * (2 * xg / w - 1) + coeffs[2] * (2 * yg / h - 1)
    dc = float(coeffs[0]); tilt = float(np.hypot(coeffs[1], coeffs[2]))
    if ns < 8 or not (0.5 <= dc <= 3.0) or tilt > 0.3 * abs(dc):
        sm = np.full((h, w), dc if 0.3 < dc < 3 else 1.0)
    return apply_flow_field(np.where(np.abs(sm) > 0.1, ep / sm, ep),
                            np.load(f'{base}/flow/{date}_scaled_flow.npy'))


def main():
    lst, band, date, robust = _ARGV[1], _ARGV[2], _ARGV[3], _ARGV[4]
    os.makedirs(OUT, exist_ok=True)
    base = f'{coadd_root()}/{lst}_{band}'
    cw = f'/data07/gh/phase3/deep/{lst}_{band}'
    mp = f'{base}/coadd_final/I_deep_Taper_{robust}_{band}.fits'
    master = _load(mp); mh = fits.getheader(mp); cd = abs(mh['CDELT1'])
    if np.nanmax(np.abs(master)) > 1e6:
        print(f'{lst} {band} {robust}: master corrupt'); return
    em, en, epa, _ = fit_empirical_beam(master)
    beam = em if np.isfinite(em) and em > 0 else 7.0
    ms = _mstruct_i(master, mp, beam)
    ebeam = (beam * cd, (en if np.isfinite(en) and en > 0 else beam) * cd,
             epa if np.isfinite(epa) else 0.0)
    epoch0 = build_epoch(lst, band, date, robust, master, mh)
    up2 = np.load(f'{cw}/ep_{date}_up2_flow.npy')
    eh = fits.getheader(f'{base}/subtracted/{date}_I_sub_{band}.fits')
    hdr = _det_header(mh, band, beam, cd, eh.get('DATE-OBS'))
    span = float(eh.get('INTTIME') or 3591.0)
    hdr['INTTIME'] = span; hdr['DURATION'] = span
    wcs = WCS(hdr).celestial
    hdr_ref = _det_header(mh, band, beam, cd, None)

    # base residual (no injections) for sig0 + searchable mask
    resid0 = np.nan_to_num(_residual(epoch0, master, up2, hdr_ref, cd), nan=0.0)
    sig0 = central_rms(resid0)
    horizon, ateam, badwcs, obs_time, _ = sit.make_masks(hdr, resid0.shape, wcs)
    blank = horizon | ateam | badwcs
    searchable = ~blank
    n = epoch0.shape[0]
    print(f'{lst} {band} {date} {robust}: sig0 {sig0*1000:.1f} mJy, beam {beam:.1f}px',
          flush=True)

    rows = []
    for fr in range(N_FRAMES):
        rng = np.random.default_rng(700 + fr)
        inj = []
        while len(inj) < N_PER:
            x = int(rng.integers(120, n - 120)); y = int(rng.integers(120, n - 120))
            if not searchable[y, x]:
                continue
            if any((x - px) ** 2 + (y - py) ** 2 < MIN_SEP ** 2 for px, py, _ in inj):
                continue
            inj.append((x, y, float(rng.uniform(SNR_LO, SNR_HI))))
        ei = epoch0.copy()
        for x, y, s in inj:
            ei += gaussian_stamp(ei.shape, x, y, s * sig0, ebeam, cd)
        rI = np.nan_to_num(_residual(ei, master, up2, hdr_ref, cd), nan=0.0)
        rms0 = sit.get_uniform_rms(rI, horizon)
        cands = sit.extract_positive_peaks(rI.astype(np.float32), hdr, horizon,
                                           ateam, badwcs, wcs, obs_time, rms0, beam)
        sit.assign_local_snr(cands, rI, blank, beam, sit.CONFIG['SIGMA_THRESHOLD'])
        sit.merge_nearby(cands, beam, wcs)
        det = np.array([[c['pix_x'], c['pix_y']] for c in cands
                        if c['status'] == 'PASS'], float)
        det = det if len(det) else np.empty((0, 2))
        # injected local SNR from actual local rms at each position
        ic = [{'pix_x': x, 'pix_y': y, 'flux': s * sig0, 'status': 'PASS',
               'snr': 0.0} for x, y, s in inj]
        sit.assign_local_snr(ic, rI, blank, beam, 0.0)
        for (x, y, s), c in zip(inj, ic):
            lr = float(c.get('local_rms', np.nan))
            snr_loc = s * sig0 / lr if np.isfinite(lr) and lr > 0 else np.nan
            rec = bool(len(det) and
                       np.hypot(det[:, 0] - x, det[:, 1] - y).min() <= 1.5 * beam)
            msv = float(ms[y, x])
            passed = rec and (snr_loc >= 5 * max(msv, 1.0))
            # inj_flux_jy anchors C50 in ABSOLUTE flux so R0-vs-R075 (different
            # rms) compare on the same scale (Gregg 2026-07-04)
            rows.append([round(snr_loc, 3) if np.isfinite(snr_loc) else '',
                         int(rec), int(passed), round(msv, 2),
                         round(s * sig0, 4),
                         round(lr, 4) if np.isfinite(lr) else ''])
        print(f'  frame {fr}: {sum(r[1] for r in rows)} detected so far', flush=True)

    sp = f'{OUT}/i_c50_{band}_{lst}_{date}_{robust}.csv'
    with open(sp, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['snr_local', 'detected', 'pass_mstruct', 'mstruct_i',
                    'inj_flux_jy', 'local_rms_jy'])
        w.writerows(rows)
    rows = [r for r in rows if r[0] != '']
    snr = np.array([r[0] for r in rows], float)
    det = np.array([r[1] for r in rows], bool)
    pas = np.array([r[2] for r in rows], bool)
    edges = np.arange(4, 16.01, 0.75); ctr = 0.5 * (edges[1:] + edges[:-1])

    def c50(mask):
        fr_ = []
        for i in range(len(edges) - 1):
            m = (snr >= edges[i]) & (snr < edges[i + 1])
            fr_.append(mask[m].mean() if m.sum() >= 3 else np.nan)
        fr_ = np.array(fr_); ok = np.isfinite(fr_)
        x, y = ctr[ok], fr_[ok]
        for i in range(len(x) - 1):
            if y[i] < 0.5 <= y[i + 1]:
                return x[i] + (0.5 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
        return x[np.argmin(np.abs(y - 0.5))] if len(x) else np.nan
    print(f'=== {lst} {band} {date} {robust}: N={len(rows)} | '
          f'detect C50 = {c50(det):.2f} | post-mstruct C50 = {c50(pas):.2f} '
          f'sigma-local | sig0 = {sig0*1000:.1f} mJy ===', flush=True)
    # flux-anchored C50 (common scale across robusts)
    fx = np.array([r[4] for r in rows], float) * 1000.0   # mJy
    fe = np.percentile(fx, np.linspace(2, 98, 17))
    def c50f(mask):
        fr_ = []
        for i in range(len(fe) - 1):
            m = (fx >= fe[i]) & (fx < fe[i + 1])
            fr_.append(mask[m].mean() if m.sum() >= 3 else np.nan)
        fr_ = np.array(fr_); ctrf = 0.5 * (fe[1:] + fe[:-1]); ok = np.isfinite(fr_)
        x, y = ctrf[ok], fr_[ok]
        for i in range(len(x) - 1):
            if y[i] < 0.5 <= y[i + 1]:
                return x[i] + (0.5 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
        return x[np.argmin(np.abs(y - 0.5))] if len(x) else np.nan
    print(f'=== {lst} {band} {date} {robust}: FLUX-anchored detect C50 = '
          f'{c50f(det):.0f} mJy | post-mstruct C50 = {c50f(pas):.0f} mJy ===',
          flush=True)


if __name__ == '__main__':
    main()
