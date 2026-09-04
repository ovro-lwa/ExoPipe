#!/usr/bin/env python3
"""
phase3_i_psfmatch_ab.py — A/B test of symmetric cross-convolution ("ZOGY-lite")
subtraction for Stokes I (Gregg 2026-07-04): D = G_m ⊗ epoch − G_e ⊗ master,
with G_m/G_e = elliptical-Gaussian approximations of the master/epoch empirical
beams — both terms get the identical effective PSF, killing PSF-mismatch source
residuals. Acceptance criterion: little/no C50 impact (measured by injections).

Per (LST, band, date):
  A) artifact side — rebuild the residual with cross-convolution inserted after
     the x2 refine, run identical +-detection at 5 sigma, report +/- candidate
     counts (compare against the archived unmatched residual's ledger counts)
  B) sensitivity side — N_FRAMES x N_PER blind injections (epoch-beam-shaped,
     injected pre-convolution as real sources would be) through the matched
     chain -> detect + post-mstruct C50, flux-anchored
Writes matched residual FITS to i_resid archive with _psfmatch tag + a CSV of
injections to Run_I_GATES/c50/.
Usage: python phase3_i_psfmatch_ab.py <LST> <SCOPE> <DATE> [--no-c50]
"""
from exopipe.common.roots import coadd_root
import sys
_ARGV = list(sys.argv)
import os, csv
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel, convolve_fft
from exopipe.phase3.phase3_deep import _load, NPROC_RESID
from exopipe.phase3.phase3_deep import _residual as _refine_subtract   # not used for matched path
from exopipe.phase3.coadd_full_pipeline import apply_flow_field, build_striation_model, get_midpoint_time
from exopipe.phase3.residuals import fft_upsample2, fft_decimate2
from exopipe.phase3.phase3_tenmin import _det_header
from exopipe.phase3.phase3_i_gate_ledger import _mstruct_i
from exopipe.phase3.identify import fit_empirical_beam
from exopipe.phase3.residuals import gaussian_stamp
from exopipe.phase3.phase3_pipeline import STRIATION_WINDOW_DEG
from exopipe.phase3 import search_i_transients as sit
from exopipe.phase3 import phase3_veto as pv

OUT = '/data07/gh/phase3/transients/Run_I_GATES'
N_PER, N_FRAMES = 120, 3
SNR_LO, SNR_HI = 4.0, 16.0
S2F = 1.0 / 2.355


def gk(bmaj_px, bmin_px, pa_deg):
    return Gaussian2DKernel(x_stddev=max(bmaj_px, 0.8) * S2F,
                            y_stddev=max(bmin_px, 0.8) * S2F,
                            theta=np.radians(pa_deg + 90.0))


def dilution(be, bm):
    """Point-transient peak dilution through the cross-convolution — computed
    NUMERICALLY per pair from the exact kernels (Gregg 2026-07-04: never a
    constant; flux scale must be correct)."""
    yy, xx = np.mgrid[-50:51, -50:51].astype(float)
    th = np.radians(be[2])
    xr = xx * np.cos(th) + yy * np.sin(th); yr = -xx * np.sin(th) + yy * np.cos(th)
    src = np.exp(-0.5 * ((xr / (be[0] * S2F)) ** 2 + (yr / (be[1] * S2F)) ** 2))
    return float(convolve_fft(src, gk(*bm), normalize_kernel=True).max())


def matched_resid(ep_aligned, master, up2, hdr_ref, cd, be, bm, mfs_args=None):
    """x2 refine -> symmetric cross-convolution -> SECOND flux-scale fit on the
    matched pair (Gregg's fix: kills the PSF-peak-biased scale error) ->
    difference -> RENORMALIZE by the per-pair dilution factor (honest point-
    transient flux units) -> destripe."""
    from exopipe.phase3.coadd_full_pipeline import measure_flux_scale
    ref = fft_decimate2(apply_flow_field(fft_upsample2(ep_aligned), up2))
    invalid = (ref == 0) | (master == 0) | ~np.isfinite(ref)
    Ec = convolve_fft(np.nan_to_num(ref), gk(*bm), allow_huge=True,
                      normalize_kernel=True)
    Mc = convolve_fft(np.nan_to_num(master), gk(*be), allow_huge=True,
                      normalize_kernel=True)
    w = WCS(hdr_ref).celestial
    bp_eff = float(np.hypot(be[0], bm[0]))
    dc2 = 1.0
    if mfs_args is not None:
        mask = invalid | ~np.isfinite(Ec) | ~np.isfinite(Mc)
        _, co, ns = measure_flux_scale(Ec, Mc, w, bp_eff, mask, mfs_args)
        h_, w_ = master.shape
        yg, xg = np.indices((h_, w_), dtype=float)
        # P3_SCALE2_DC_ONLY=1: tilt A/B test switch (Gregg 2026-08-28) —
        # drop the tilt terms from the 2nd-pass matched-pair plane (this pass
        # has no tilt guard, unlike the co-add's first pass). Default unchanged.
        import os as _os
        if _os.environ.get('P3_SCALE2_DC_ONLY') == '1':
            sm = np.full((h_, w_), co[0])
        else:
            sm = co[0] + co[1] * (2 * xg / w_ - 1) + co[2] * (2 * yg / h_ - 1)
        dc2 = float(co[0])
        if ns >= 8 and 0.5 <= dc2 <= 2.0:
            Ec = np.where(np.abs(sm) > 0.1, Ec / sm, Ec)
    f = dilution(be, bm)
    d = (Ec - Mc) / max(f, 1e-3)          # honest Jy/beam_eff for point transients
    # window = 8 x effective beam (R-STRIATION-WINDOW-APPROVED-8BEAMS 2026-08-26)
    smod, _ = build_striation_model(np.nan_to_num(np.where(invalid, 0, d)), w, invalid,
                                    ra_window_beams=8.0 * bp_eff,
                                    beam_pix=1.0)
    return np.where(invalid, np.nan, d - smod), bp_eff, f, dc2


def _r0_aligned_epoch(base, lst, scope, date, master, mh, wcsm, beam):
    """Rebuild the R0 stage-B aligned epoch (recipe = phase3_i_inspect_r0.py):
    raw Taper_R0 wideband -> regrid -> guarded flux scale -> stage-B flow."""
    import tempfile, shutil
    from exopipe.phase3.coadd_full_pipeline import (discover_dates, find_wideband_deep_images,
                                     load_image, measure_flux_scale, regrid_image)
    sd = dict((d, s) for d, s, _ in discover_dates(lst, scope)).get(date)
    wb = find_wideband_deep_images(sd, scope, 'I', 'Taper_R0')
    fp = f'{base}/flow/{date}_scaled_flow.npy'
    if not wb or not os.path.exists(fp):
        raise RuntimeError(f'missing R0 epoch or stage-B flow for {date}')
    ep, ehdr = load_image(wb)
    if ep.shape != master.shape:
        td = tempfile.mkdtemp()
        ep, ehdr = load_image(regrid_image(wb, mh, f'{td}/rg.fits'))
        shutil.rmtree(td, ignore_errors=True)
    hmask = ~np.isfinite(master) | (master == 0)
    emask = ~np.isfinite(ep) | (ep == 0)
    ot = get_midpoint_time(ehdr, wb)
    _, co, ns = measure_flux_scale(ep, master, wcsm, beam, hmask | emask, ot)
    h, w = master.shape
    yg, xg = np.indices((h, w), dtype=float)
    sm = co[0] + co[1] * (2 * xg / w - 1) + co[2] * (2 * yg / h - 1)
    dc = float(co[0]); tilt = float(np.hypot(co[1], co[2]))
    if ns < 8 or not (0.5 <= dc <= 3.0) or tilt > 0.3 * abs(dc):
        sm = np.full((h, w), dc if 0.3 < dc < 3 else 1.0)
    return apply_flow_field(np.where(np.abs(sm) > 0.1, ep / sm, ep), np.load(fp))


def main():
    lst, scope, date = _ARGV[1], _ARGV[2], _ARGV[3]
    robust = 'R0' if 'R0' in _ARGV[4:] else 'R075'
    do_c50 = '--no-c50' not in _ARGV
    base = f'{coadd_root()}/{lst}_{scope}'
    cw = f'/data07/gh/phase3/deep/{lst}_{scope}'
    mp = f'{base}/coadd_final/I_deep_Taper_{robust}_{scope}.fits'
    master = _load(mp); mh = fits.getheader(mp); cd = abs(mh['CDELT1'])
    if np.nanmax(np.abs(master)) > 1e6 or (master != 0).sum() < 0.1 * master.size:
        print(f'{lst} {scope}: {robust} master corrupt — abort'); return
    em, en, epa, _ = fit_empirical_beam(master)
    bm = (em if np.isfinite(em) and em > 0 else 7.0,
          en if np.isfinite(en) and en > 0 else 7.0,
          epa if np.isfinite(epa) else 0.0)
    if robust == 'R0':
        ep0 = _r0_aligned_epoch(base, lst, scope, date, master, mh,
                                WCS(mh).celestial, bm[0])
    else:
        ep0 = _load(f'{base}/subtracted/{date}_I_sub_{scope}.fits') + master
    ee, ee2, eepa, _ = fit_empirical_beam(ep0)
    be = (ee if np.isfinite(ee) and ee > 0 else bm[0],
          ee2 if np.isfinite(ee2) and ee2 > 0 else bm[1],
          eepa if np.isfinite(eepa) else 0.0)
    up2 = np.load(f'{cw}/ep_{date}_up2_flow.npy')
    eh = fits.getheader(f'{base}/subtracted/{date}_I_sub_{scope}.fits')
    hdr_ref = _det_header(mh, scope, bm[0], cd, None)
    print(f'{lst} {scope} {date} [{robust}]: epoch beam {be[0]:.2f}x{be[1]:.2f}px '
          f'master {bm[0]:.2f}x{bm[1]:.2f}px', flush=True)

    ot0 = get_midpoint_time(eh, f'{base}/subtracted/{date}_I_sub_{scope}.fits')
    resid, bp_eff, fdil, dc2 = matched_resid(ep0, master, up2, hdr_ref, cd, be, bm, mfs_args=ot0)
    print(f'  dilution factor {fdil:.3f} | 2nd-pass DC {dc2:.4f}', flush=True)
    rI = np.nan_to_num(resid, nan=0.0)
    ms = _mstruct_i(convolve_fft(master, gk(*be), allow_huge=True,
                                 normalize_kernel=True), mp, bp_eff)
    hdr = _det_header(mh, scope, bp_eff, cd, eh.get('DATE-OBS'))
    span = float(eh.get('INTTIME') or 3591.0)
    hdr['INTTIME'] = span; hdr['DURATION'] = span
    wcs = WCS(hdr).celestial
    horizon, ateam, badwcs, obs_time, elmap = sit.make_masks(hdr, rI.shape, wcs)
    excl = horizon | ateam | badwcs
    rms0 = sit.get_uniform_rms(rI, horizon)
    counts = {}
    for sgn, img in ((1, rI), (-1, -rI)):
        cs = sit.extract_positive_peaks(img.astype(np.float32), hdr, horizon,
                                        ateam, badwcs, wcs, obs_time, rms0, bp_eff)
        sit.assign_local_snr(cs, img, excl, bp_eff, 5.0)
        sit.merge_nearby(cs, bp_eff, wcs)
        ok = [c for c in cs if c['status'] == 'PASS'
              and abs(float(c.get('local_snr', c.get('snr', 0)))) >= 6]
        msok = [c for c in ok if abs(float(c.get('local_snr', c.get('snr', 0))))
                >= 5 * max(float(ms[int(c['pix_y']), int(c['pix_x'])]), 1.0)]
        counts[sgn] = (len(ok), len(msok))
    print(f'MATCHED counts >=6s (raw / post-mstruct): +{counts[1][0]}/{counts[1][1]}  '
          f'-{counts[-1][0]}/{counts[-1][1]}', flush=True)
    ad = f'/data07/gh/phase3/i_resid/{lst}_{scope}'
    os.makedirs(ad, exist_ok=True)
    rp = f'{ad}/psfmatchv2_{date}_{robust}_I_resid.fits'
    fits.writeto(rp, resid.astype(np.float32), wcs.to_header(), overwrite=True)
    from exopipe.common import phase3_products as prod
    prod.write_standard_header(
        rp, scope=scope, stokes='I',
        start=eh.get('DATE-OBS'), end=eh.get('DATE-END'),
        inttime_s=span, time_basis='phase2',
        beam=(bp_eff * cd, bp_eff * cd, 0.0),
        aligned=True, alignref=os.path.basename(mp),
        msub=True, msubfile=mp, ssub=True, dewarped=True, validate=True,
        extra={'RUNLABEL': ('Run_I_GATES', 'PSF-match A/B v2'),
               'EPOCH_ID': (f'{lst}_{date}', 'LST_date epoch'),
               'PSFMATCH': (True, 'symmetric cross-convolution subtraction'),
               'RENORM': (round(1.0 / fdil, 4), 'per-pair point-src dilution corr'),
               'SCALE2B': (round(dc2, 4), '2nd-pass matched-pair flux DC')})

    if not do_c50:
        return
    # ---- injections through the matched chain ----
    n = ep0.shape[0]
    cen = rI[n // 4:3 * n // 4, n // 4:3 * n // 4]
    cen = cen[np.isfinite(cen) & (cen != 0)]
    sig0 = 1.4826 * float(np.median(np.abs(cen - np.median(cen))))
    searchable = ~excl
    ebeam = (be[0] * cd, be[1] * cd, be[2])
    rows = []
    for fr in range(N_FRAMES):
        rng = np.random.default_rng(900 + fr)
        inj = []
        while len(inj) < N_PER:
            x = int(rng.integers(120, n - 120)); y = int(rng.integers(120, n - 120))
            if not searchable[y, x]:
                continue
            if any((x - px) ** 2 + (y - py) ** 2 < 60 ** 2 for px, py, _ in inj):
                continue
            inj.append((x, y, float(rng.uniform(SNR_LO, SNR_HI))))
        ei = ep0.copy()
        for x, y, s in inj:
            ei += gaussian_stamp(ei.shape, x, y, s * sig0, ebeam, cd)
        rmi, _, _, _ = matched_resid(ei, master, up2, hdr_ref, cd, be, bm, mfs_args=ot0)
        rmi = np.nan_to_num(rmi, nan=0.0)
        rr = sit.get_uniform_rms(rmi, horizon)
        cs = sit.extract_positive_peaks(rmi.astype(np.float32), hdr, horizon,
                                        ateam, badwcs, wcs, obs_time, rr, bp_eff)
        sit.assign_local_snr(cs, rmi, excl, bp_eff, sit.CONFIG['SIGMA_THRESHOLD'])
        sit.merge_nearby(cs, bp_eff, wcs)
        det = np.array([[c['pix_x'], c['pix_y']] for c in cs
                        if c['status'] == 'PASS'], float)
        det = det if len(det) else np.empty((0, 2))
        ic = [{'pix_x': x, 'pix_y': y, 'flux': s * sig0, 'status': 'PASS',
               'snr': 0.0} for x, y, s in inj]
        sit.assign_local_snr(ic, rmi, excl, bp_eff, 0.0)
        hh_, ww_ = rmi.shape
        cbb = rmi[hh_ // 3:2 * hh_ // 3, ww_ // 3:2 * ww_ // 3]
        cbb = cbb[np.isfinite(cbb)]
        grms_i = (1.4826 * float(np.median(np.abs(cbb - np.median(cbb))))
                  if cbb.size > 100 else 0.0)
        for (x, y, s), c in zip(inj, ic):
            lr = float(c.get('local_rms', np.nan))
            snr_loc = s * sig0 / lr if np.isfinite(lr) and lr > 0 else np.nan
            rec = bool(len(det) and
                       np.hypot(det[:, 0] - x, det[:, 1] - y).min() <= 1.5 * bp_eff)
            msv = float(ms[y, x])
            blb = (int(pv.bilobe_ghost_reason(rmi, x, y, bp_eff, grms_i, cd)
                       != 'keep') if grms_i > 0 else '')
            g = pv._gfit(rmi, x, y, bp_eff)
            rows.append([round(snr_loc, 3) if np.isfinite(snr_loc) else '',
                         int(rec), int(rec and snr_loc >= 5 * max(msv, 1.0)),
                         round(msv, 2), round(s * sig0, 4),
                         round(lr, 4) if np.isfinite(lr) else '',
                         blb,
                         round(g['ar'], 2) if g else '',
                         round(g['resid'], 3) if g else '',
                         round(float(elmap[y, x]), 1)
                         if np.isfinite(elmap[y, x]) else ''])
        # empirical flux-scale self-check: with the per-pair renormalization,
        # a point transient's recovered peak must equal its injected flux
        ratios = [float(np.nanmax(rmi[y - 4:y + 5, x - 4:x + 5])) / (s * sig0)
                  for x, y, s in inj if s >= 10.0]
        mr = float(np.median(ratios)) if ratios else np.nan
        flag = '' if 0.85 <= mr <= 1.15 else '  *** FLUX-SCALE VIOLATION ***'
        print(f'  inj frame {fr} done | recovered/injected peak (bright) '
              f'median = {mr:.3f}{flag}', flush=True)
    os.makedirs(f'{OUT}/c50', exist_ok=True)
    sp = f'{OUT}/c50/i_c50_{scope}_{lst}_{date}_{robust}_PSFMATCHV2.csv'
    with open(sp, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['snr_local', 'detected', 'pass_mstruct', 'mstruct_i',
                    'inj_flux_jy', 'local_rms_jy', 'bilobe', 'gf_ar',
                    'gf_resid', 'el_deg'])
        w.writerows(rows)
    rows = [r for r in rows if r[0] != '']
    fx = np.array([r[4] for r in rows]) * 1000.0
    det_ = np.array([r[1] for r in rows], bool)
    pas = np.array([r[2] for r in rows], bool)
    fe = np.percentile(fx, np.linspace(2, 98, 17))

    def c50f(mask):
        fr_ = []
        for i in range(len(fe) - 1):
            m = (fx >= fe[i]) & (fx < fe[i + 1])
            fr_.append(mask[m].mean() if m.sum() >= 3 else np.nan)
        fr_ = np.array(fr_); ctr = 0.5 * (fe[1:] + fe[:-1]); ok = np.isfinite(fr_)
        x, y = ctr[ok], fr_[ok]
        for i in range(len(x) - 1):
            if y[i] < 0.5 <= y[i + 1]:
                return x[i] + (0.5 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
        return x[np.argmin(np.abs(y - 0.5))] if len(x) else np.nan
    print(f'=== PSFMATCH {lst} {scope} {date}: FLUX detect C50 = {c50f(det_):.0f} mJy '
          f'| post-mstruct C50 = {c50f(pas):.0f} mJy ===', flush=True)


if __name__ == '__main__':
    main()
