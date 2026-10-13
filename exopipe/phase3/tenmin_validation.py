#!/usr/bin/env python3
"""
10-MIN VALIDATION — pre-production gate before the ~1-2 week 10-min run.

Reuses the EXACT co-add pipeline alignment (flux-scale + saved flow +
process_variant_image) so the residuals are faithful to production. For
each cell (10h, good+bad, R/G/B, I+V), aligns the test epoch's 10-min
intervals and measures the residual two ways:

  A = aligned_interval - HOUR-DEEP        (same-night 1-hour deep)
  B = aligned_interval - SUB-LST MASTER   (coadd_provisional/{S}_10min_int0X,
                                           same t-index co-added across nights)

Reports per cell: residual rms A vs B (criterion #1 — lower=deeper), the
striation rms benefit on the 10-min residual (trivial vs difficult), and
(stage 2) a blind-injection completeness S-curve for whichever reference wins.

Stage 1 (this run): rms A-vs-B + striation. Stage 2 (--scurve): C50 + FP.
"""
from exopipe.phase3.config import coadd_root
import os, sys, glob
import numpy as np
_ARGV = list(sys.argv); sys.argv = ['tenmin_val']
from astropy.io import fits
from astropy.wcs import WCS
from exopipe.phase3.lib.masters import coadd as C
from exopipe.phase3.lib.search.identification import STRIATION_WINDOW_DEG
sys.argv = _ARGV              # restore after import-protection clobber

LST = '10h'
DATES = {'good': '2024-12-29', 'bad': '2024-12-19'}
BANDS = ['Red', 'Green', 'Blue']
WB = os.environ.get('P3_COADD_ROOT')   # NO default (Gregg 2026-09-04)
# ^ env override (2026-07-16): the subband 10-min tier runs against
#   /data07/gh/phase3/coadd_sb; wideband default unchanged.


def srcclip_rms(d):
    d = np.where((d == 0) | ~np.isfinite(d), np.nan, d)
    n = d.shape[0]; q = d[n//3:-n//3, n//3:-n//3].astype(float)
    q = q[np.isfinite(q)]
    if q.size < 100:
        return np.nan
    for _ in range(3):
        m = np.nanmedian(q); s = 1.4826*np.nanmedian(np.abs(q-m))
        if s == 0:
            break
        q = q[np.abs(q-m) < 4*s]
    return s*1e3


def striation_reduction(resid, wcs, cd):
    """rms-reduction % from striation subtraction on a residual."""
    inv = ~np.isfinite(resid) | (resid == 0)
    sm, _ = C.build_striation_model(np.nan_to_num(resid), wcs, inv,
                                    ra_window_beams=STRIATION_WINDOW_DEG/cd, beam_pix=1.0)
    b = srcclip_rms(resid); a = srcclip_rms(resid - sm)
    return b, a, (100*(1-a/b) if (b and np.isfinite(b) and np.isfinite(a)) else np.nan)


def setup_date_params(lst, freq_str):
    """Replicate run_provisional_10min_lst_freq per-date setup; return the
    common context + a function to align a given date's interval images."""
    ref_date = C.REFERENCE_DATES.get(lst)
    fl = C.freq_label(freq_str)
    workdir = f'{coadd_root()}/{lst}_{fl}'
    all_dates = C.discover_dates(lst, freq_str)
    dates = [(d, sd, img) for d, sd, img in all_dates]
    ref_entries = [x for x in dates if x[0] == ref_date]
    if not ref_entries:
        # reference night has no products for this cell (e.g. 05h 50MHz:
        # 2024-12-21 never recorded 50MHz) — fall back to the nearest
        # available date, mirroring the deep builder's reference fallback
        ref_entries = [min(dates, key=lambda x: abs(
            (np.datetime64(x[0]) - np.datetime64(ref_date))
            / np.timedelta64(1, 'D')))]
        ref_date = ref_entries[0][0]
        print(f'  [{lst} {fl}] reference fallback -> {ref_date}', flush=True)
    ref_date_entry = ref_entries[0]
    target_dates = [x for x in dates if x[0] != ref_date]
    post_dates = [x for x in dates if x[0] >= C.CUTOVER_DATE]
    # modal-grid template (NOT post_dates[0] — a single wrong-grid date must
    # not poison target_shape; the 19h Full lesson)
    tmpl_hdr, target_shape = (C._modal_grid_template(post_dates, ref_date_entry)
                              if post_dates
                              else C.build_template_header(ref_date_entry[2]))
    ref_img_path = ref_date_entry[2]
    ref_needs_regrid = ref_date < C.CUTOVER_DATE and bool(post_dates)
    if ref_needs_regrid:
        ref_img_path = C.regrid_image(ref_img_path, tmpl_hdr, f'/fast/gh/main/tmp/ref_{ref_date}_{fl}_{os.getpid()}.fits')
    ref_data, ref_hdr = C.load_image(ref_img_path)
    ref_wcs = WCS(ref_hdr).celestial
    cd = abs(ref_hdr.get('CDELT1', ref_hdr.get('CD1_1', 0.037)))
    beam_pix = ref_hdr.get('BMAJ', 5*cd) / cd
    horizon_mask = ~np.isfinite(ref_data) | (ref_data == 0)
    return dict(ref_date=ref_date, fl=fl, workdir=workdir, dates=dates, tmpl_hdr=tmpl_hdr,
                target_shape=target_shape, ref_data=ref_data, ref_wcs=ref_wcs, cd=cd,
                beam_pix=beam_pix, horizon_mask=horizon_mask, post_dates=bool(post_dates))


def align_date_intervals(ctx, date, freq_str, stokes, int_indices):
    """Flux-scale + flow-align the given date's 10-min intervals (faithful to
    production). Returns {int_idx: aligned_array}."""
    entry = [x for x in ctx['dates'] if x[0] == date]
    if not entry:
        return {}
    _, sci_dir, align_img_path = entry[0]
    needs_regrid = date < C.CUTOVER_DATE and ctx['post_dates']
    work_path = (C.regrid_image(align_img_path, ctx['tmpl_hdr'], f'/fast/gh/main/tmp/{date}_{ctx["fl"]}_rg_{os.getpid()}.fits')
                 if needs_regrid else align_img_path)
    tgt_data, tgt_hdr = C.load_image(work_path)
    if tgt_data.shape != ctx['target_shape']:
        return {}
    tgt_mask = ~np.isfinite(tgt_data) | (tgt_data == 0)
    tgt_obs_time = C.get_midpoint_time(tgt_hdr, work_path)
    coeffs = C.measure_flux_scale(tgt_data, ctx['ref_data'], ctx['ref_wcs'], ctx['beam_pix'],
                                  ctx['horizon_mask'] | tgt_mask, tgt_obs_time)[1]
    h, w = ctx['target_shape']
    yg, xg = np.indices((h, w), dtype=float)
    scale_map_full = coeffs[0] + coeffs[1]*(2.0*xg/w - 1.0) + coeffs[2]*(2.0*yg/h - 1.0)
    flow_path = f'{ctx["workdir"]}/flow/{date}_scaled_flow.npy'
    flow = np.load(flow_path) if os.path.exists(flow_path) else None
    regrid_coords = (C.precompute_regrid_mapping(align_img_path, ctx['tmpl_hdr'], ctx['target_shape'])
                     if needs_regrid else None)
    tmin = {idx: p for p, idx in C.find_10min_images(sci_dir, freq_str, stokes)}
    out = {}
    for idx in int_indices:
        if idx not in tmin:
            continue
        op = f'/fast/gh/main/tmp/{date}_{ctx["fl"]}_{stokes}_int{idx:02d}_{os.getpid()}.fits'
        if C.process_variant_image(tmin[idx], flow, scale_map_full, ctx['tmpl_hdr'],
                                   ctx['target_shape'], op, regrid_coords) and os.path.exists(op):
            out[idx] = C.load_image(op)[0]
            os.remove(op)
    return out


def run_stage1():
    print("10-MIN VALIDATION stage 1 — residual rms (mJy), FAIR (both routes striation-cleaned)", flush=True)
    print("  A=-hourdeep  B=-subLSTmaster.  A_raw->A_str = route-A before/after striation removal.", flush=True)
    print("%-14s %-3s %9s %9s %9s %8s  %s" % ('cell', 'stk', 'A_raw', 'A_str', 'B_str', 'B/A_str', 'A striation b->a'), flush=True)
    for band in BANDS:
        freq_str = band
        ctx = setup_date_params(LST, freq_str)
        fl = ctx['fl']; cd = ctx['cd']
        wd = ctx['workdir']
        # sub-LST masters (B) + hour-deep (clean) availability
        for stk in ['I', 'V']:
            subm = {}
            for idx in range(6):
                p = f'{wd}/coadd_provisional/{stk}_10min_int{idx:02d}_{fl}.fits'
                if os.path.exists(p):
                    subm[idx] = C.load_image(p)[0]
            if not subm:
                print(f"{band:14s} {stk}: no sub-LST masters", flush=True)
                continue
            for tag, date in DATES.items():
                clean_p = f'{wd}/subtracted/{date}_{stk}_clean_{fl}.fits'
                if not os.path.exists(clean_p):
                    print(f"{band+' '+tag:14s} {stk}: no clean hour-deep ({date})", flush=True)
                    continue
                hourdeep, clean_hdr = C.load_image(clean_p)
                wcs = WCS(clean_hdr).celestial
                aligned = align_date_intervals(ctx, date, freq_str, stk, list(subm.keys()))
                if not aligned:
                    print(f"{band+' '+tag:14s} {stk}: alignment failed", flush=True)
                    continue
                A_raw, A_str, B_str, Ab, Aa = [], [], [], [], []
                for j, (idx, al) in enumerate(sorted(aligned.items())):
                    if j >= 3:                      # 3 sampled intervals (striation is the slow step)
                        break
                    inv = (al == 0) | ~np.isfinite(al)
                    resA = np.where(inv | (hourdeep == 0), np.nan, al - hourdeep)
                    resB = np.where(inv | (subm[idx] == 0), np.nan, al - subm[idx])
                    ab, aa, _ = striation_reduction(resA, wcs, cd)   # route A: striation-cleaned (FAIR)
                    bb, ba, _ = striation_reduction(resB, wcs, cd)   # route B: striation-cleaned
                    A_raw.append(ab); A_str.append(aa); B_str.append(ba)
                    Ab.append(ab); Aa.append(aa)
                mAr, mAs, mB = np.nanmean(A_raw), np.nanmean(A_str), np.nanmean(B_str)
                print("%-14s %-3s %9.0f %9.0f %9.0f %8.2f  %5.0f->%5.0f (%2.0f%%)"
                      % (band+' '+tag, stk, mAr, mAs, mB, mB/mAs if mAs else np.nan,
                         np.nanmean(Ab), np.nanmean(Aa), 100*(1-np.nanmean(Aa)/np.nanmean(Ab))), flush=True)


if __name__ == '__main__':
    run_stage1()
