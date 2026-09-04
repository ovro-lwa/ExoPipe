#!/usr/bin/env python3
"""
phase3_pipeline.py — Redesigned Phase 3 co-add + subtraction + detection
pipeline (replaces coadd_full_pipeline.py).

WHY A REDESIGN
--------------
coadd_full_pipeline.py proved the algorithms (flux scale, flow alignment,
two-pass outlier rejection, striation removal) but: it deleted the
aligned epochs after co-adding, wrote no standard headers or provenance,
subtracted images with mismatched PSFs (per-epoch beams vary ~22% in
BMAJ under ionospheric smearing — the dominant source of dipole false
positives), and had no detection-validation loop. This pipeline keeps
the proven algorithms (imported from coadd_full_pipeline) and fixes the
products: every output goes through phase3_products (naming grammar,
full header standard, PROVENANCE tables), epochs are convolved to a
canonical per-band beam BEFORE subtraction, and an injection/recovery
validation stage closes the loop with end-to-end completeness + final
false-positive counts (per Gregg: completeness matters at least as much
as the FP rate).

STAGES (each skippable / resumable)
  1 discovery    epochs from /lustre/pipeline/exopipe/phase2/{lst}/{date}/
                 Science_*/ (repaired v2 headers required)
  2 prep         flux-scale + flow-align every epoch/variant to the
                 REFERENCE DATE (2024-12-29 for the 01h-10h block) —
                 aligned epochs are PERMANENT science products
  3 beams        per-epoch EMPIRICAL beams: pixel-integrated Gaussian
                 fits of bright unresolved sources on x2 FFT-upsampled
                 stamps, zonal (3x3) because smear is elevation-dependent
  4 canon        canonical beam = 1.05 x P90(empirical BMAJ); epochs
                 convolved to it only when they differ by >0.5%
  5 coadd        two-pass master co-add (provisional -> residual-rms
                 outlier rejection [aggressive] -> final) + PROVENANCE
  6 msub         epoch(canonical) - master; V gets a low-order leakage
                 screen removed (LEAKSUB); I optionally striation (ssub)
  7 sigma        per-pixel robust noise-structure map from the msub
                 residual ensemble, floored at 1.0 (multiplies the
                 detection threshold; clean sky keeps native depth)
  8 detect       injection/recovery validation through the REAL
                 identification chains + FP classification + DS9 review
                 packets
  9 summary      per-epoch rms, IMGRMS/THERMNOI, FP count, completeness

USAGE
  conda activate py38_orca_nkosogor
  python phase3_pipeline.py --lst 10h --band Green --dry_run
  python phase3_pipeline.py --lst 10h --band Green
  python phase3_pipeline.py --lst 10h --band Green \\
      --run_label 20260610_120000 --skip_prep --skip_beams
  python phase3_pipeline.py --selftest_beam
  python phase3_pipeline.py --selftest_leakage
"""
from exopipe.common.roots import coadd_root
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.stats import mad_std
from astropy.convolution import Gaussian2DKernel, convolve_fft
from scipy.ndimage import binary_dilation
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')

# Proven algorithms reused from the pipeline being replaced
from exopipe.phase3 import coadd_full_pipeline as cfp

# phase3_subtract_lab parses argv at import time; shield it
_argv = sys.argv
sys.argv = ['phase3_pipeline']
from exopipe.phase3.residuals import (match_to_beam, inner_rms, gaussian_stamp,
                                 beam_area_px)
sys.argv = _argv

from exopipe.common import phase2_config as cfg
from exopipe.common import phase3_products as p3p

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

PHASE2_ROOT = '/lustre/pipeline/exopipe/phase2'
STAGING_BASE = '/fast/gh/main/phase3_v2'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Reference date per LST for flux scale + astrometry. 2024-12-29
# SUPERSEDES coadd_full_pipeline.REFERENCE_DATES (2024-12-21) for the
# 01h-10h block: it is the canonical good ionosphere night (see
# good-night-priority) and post-cutover, so no regridding of the
# reference is ever needed.
REFERENCE_DATES = {}
for _h in range(1, 11):
    REFERENCE_DATES[f'{_h:02d}h'] = '2024-12-29'
REFERENCE_DATES['13h'] = '2025-01-08'
for _h in range(14, 21):
    REFERENCE_DATES[f'{_h:02d}h'] = '2025-05-03'

OUTLIER_SIGMA = 3.0          # aggressive (feedback_coadd_outlier_philosophy)
MIN_DATES_FOR_COADD = 3
MIN_EPOCHS_SIGMA = 5         # minimum msub residuals for sigma_stack
CANON_FACTOR = 1.05          # canonical beam = 1.05 x P90(empirical BMAJ)
CANON_PERCENTILE = 90.0
BEAM_MATCH_TOL = 0.005       # skip convolution when within beam fit error
N_INJECT = 25                # synthetic sources per epoch (detect stage)
INJECT_FLUX_SIGMAS = [5, 6, 8, 12, 20]   # 6 = the V gate edge
# Striations are fixed in ANGULAR size across frequency (Gregg,
# 2026-06-12) — the running-median window must NOT scale with the beam.
# 4.375 deg reproduces the original search_v_transients_v2 defaults
# (20 beams x 7 pix at 0.03125 deg/pix).
STRIATION_WINDOW_DEG = 4.375
# The V leakage screen is only constrained where bright I sources
# imprint measurable leakage (Gregg 2026-06-12: "the leakage fit should
# not happen below a certain source density"). Gate: require at least
# LEAKSCREEN_MIN_SRC master-I sources whose ~2% leakage would exceed
# the V thermal floor (I > LEAKSCREEN_FLUX_RMS x thermal).
LEAKSCREEN_MIN_SRC = 25
LEAKSCREEN_FLUX_RMS = 50.0


def leakage_screen_constrained(master_i, v_therm):
    """True if the field has enough bright I sources to constrain the
    V leakage screen; also returns the source count."""
    thr = LEAKSCREEN_FLUX_RMS * max(v_therm, 1e-4)
    w = np.where(np.isfinite(master_i), master_i, -np.inf).copy()
    n = 0
    while n < LEAKSCREEN_MIN_SRC:
        iy, ix = np.unravel_index(np.argmax(w), w.shape)
        if w[iy, ix] < thr:
            break
        n += 1
        w[max(0, iy - 6):iy + 6, max(0, ix - 6):ix + 6] = -np.inf
    return n >= LEAKSCREEN_MIN_SRC, n
FLOW_CONDA_ENV = 'py310_orca'
# GC-PRE-2 (Gregg 2026-08-12): flow batch script lives in exopipe/common/
# (cross-phase core infrastructure); shelled into py310_orca, never imported.
BATCH_FLOW_SCRIPT = os.path.join(os.path.dirname(SCRIPT_DIR), 'common',
                                 'batch_flow_align.py')

# Deep variants to align + archive per epoch. The (taper, robust,
# pbcorr) tuples below cover the wideband deep matrix Phase 2 produces.
# Masters and subtractions are built from the SCIENCE variants.
SCIENCE_I = ('Taper', 0.0, False)      # I Taper Robust+0.0 raw
SCIENCE_V = ('Taper', 0.0, False)      # V Taper Robust+0.0 raw
ALIGN_VARIANT = ('Taper', -0.75, False)  # flow/scale measured on this I

FWHM2SIG = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def vkey(stokes, taper, robust, pbcorr):
    """Canonical variant key, e.g. 'I_Taper_R+0.0' / 'V_Taper_R+0.0_pbcorr'."""
    r = f"{robust:+.2f}".rstrip('0')
    r = r + '0' if r.endswith('.') else r
    key = f"{stokes}_{taper}_R{r}"
    return key + '_pbcorr' if pbcorr else key


def log(msg=''):
    print(msg, flush=True)


# ═══════════════════════════════════════════════════════════════════
# Stage 1 — DISCOVERY
# ═══════════════════════════════════════════════════════════════════

def _read_n_anchors(sci_dir, band, date, lst):
    """Median dewarp anchor count over the band's subbands (QA sidecar)."""
    path = os.path.join(sci_dir, 'QA', f'{date}_{lst}_dewarp_summary.csv')
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        subs = [float(s.replace('MHz', ''))
                for s in cfg.SUBBAND_BANDS.get(band, [])]
        if subs:
            df = df[df['freq_mhz'].isin(subs)]
        if len(df):
            return int(np.median(df['n_anchors']))
    except Exception:
        pass
    return None


def _p1run_from_history(hdr):
    """Phase 1 run label from the WSClean HISTORY path."""
    m = re.search(r'Run_\d{8}_\d{6}', str(hdr.get('HISTORY', '')))
    return m.group(0) if m else None


def discover_epochs(lst, band):
    """Find all (date, Science dir) epochs with repaired deep products.

    Returns a list of epoch dicts sorted by date. Epochs whose headers
    lack TIMEFIX v2 are skipped: pre-repair DATE-OBS values are
    midpoint-shifted (the v1 bug) and would corrupt phase folding.
    """
    epochs = []
    lst_dir = os.path.join(PHASE2_ROOT, lst)
    if not os.path.isdir(lst_dir):
        return epochs

    for date in sorted(os.listdir(lst_dir)):
        if not re.match(r'\d{4}-\d{2}-\d{2}$', date):
            continue
        sci_dir = cfp.find_latest_science_dir(lst, date)
        if not sci_dir:
            continue

        variants = {}   # vkey -> {'path', 'parsed'}
        for stokes in ['I', 'V']:
            d = os.path.join(sci_dir, band, stokes, 'deep')
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.endswith('.fits'):
                    continue
                try:
                    p = p3p.parse_name(fn)
                except ValueError:
                    continue
                if p['grammar'] != 'phase2_wideband' or not p['dewarped']:
                    continue
                key = vkey(p['stokes'], p['taper'], p['robust'], p['pbcorr'])
                variants[key] = {'path': os.path.join(d, fn), 'parsed': p}

        align_key = vkey('I', *ALIGN_VARIANT)
        sci_i_key = vkey('I', *SCIENCE_I)
        if align_key not in variants or sci_i_key not in variants:
            continue

        hdr = fits.getheader(variants[sci_i_key]['path'])
        if not str(hdr.get('TIMEFIX', '')).startswith('v2'):
            log(f"    {date}: headers not v2-repaired — SKIPPED "
                f"(run repair_phase2_archive.py first)")
            continue

        n_10min = len(glob.glob(os.path.join(
            sci_dir, band, 'I', '10min', '*dewarped*.fits')))
        v_key = vkey('V', *SCIENCE_V)
        thermnoi = hdr.get('THERMNOI')

        epochs.append({
            'date': date,
            'sci_dir': sci_dir,
            'variants': variants,
            'align_path': variants[align_key]['path'],
            'date_obs': hdr.get('DATE-OBS'),
            'date_end': hdr.get('DATE-END'),
            'mjd_obs': hdr.get('MJD-OBS'),
            'mjd_mid': hdr.get('MJD-MID'),
            'inttime': hdr.get('INTTIME'),
            'thermnoi': thermnoi,
            'has_v': v_key in variants,
            'n_10min_i': n_10min,
            'n_anchors': _read_n_anchors(sci_dir, band, date, lst),
            'p1run': _p1run_from_history(hdr),
            'p2run': os.path.basename(sci_dir),
        })
    return epochs


# ═══════════════════════════════════════════════════════════════════
# Stage 2 — EPOCH PREP (flux scale + flow align, permanent products)
# ═══════════════════════════════════════════════════════════════════

def _epoch_product_name(ctx, ep, parsed, extra_proc=()):
    """Phase 3 name for an aligned epoch product."""
    proc = []
    if parsed['pbcorr']:
        proc.append('pbcorr')
    if parsed['dewarped']:
        proc.append('dewarped')
    proc += ['aligned'] + list(extra_proc)
    return p3p.build_name(
        scope=ctx['band'], stokes=parsed['stokes'], cadence='deep',
        taper=parsed['taper'], robust=parsed['robust'], proc=proc,
        lst=ctx['lst'], epoch=ep['date_obs'])


def _measure_imgrms(data, stokes):
    """Central rms per the agreed rules: I central region (iono smear
    statistic), V inner 25% of area (thermal noise)."""
    n = min(data.shape)
    if stokes == 'V':
        m = int(n * 0.25)   # central 50% per dim = inner 25% area
    else:
        m = int(n * 0.15)
    inner = data[m:data.shape[0] - m, m:data.shape[1] - m]
    return float(mad_std(inner[inner != 0], ignore_nan=True))


def stage_prep(ctx, dry_run=False):
    """Flux-scale + flow-align every epoch variant to the reference date.

    Aligned + scaled epochs are PERMANENT science products: written with
    full standard headers and Phase 3 names under
    {workdir}/{date}/P3_{runlabel}/ — never deleted by later stages.
    """
    lst, band = ctx['lst'], ctx['band']
    ref_date = ctx['ref_date']
    epochs = ctx['epochs']
    workdir = ctx['workdir']
    prep_dir = os.path.join(workdir, 'work', 'prep')
    flow_dir = os.path.join(workdir, 'work', 'flow')

    ref = next((e for e in epochs if e['date'] == ref_date), None)
    if ref is None:
        raise RuntimeError(f"Reference epoch {ref_date} not found for "
                           f"{lst} {band}")

    log(f"\n── Stage 2 PREP: {len(epochs)} epochs → reference {ref_date} ──")
    if dry_run:
        n_var = sum(len(e['variants']) for e in epochs)
        n_flow = sum(1 for e in epochs if e['date'] != ref_date)
        log(f"  [plan] template from {ref['align_path']}")
        log(f"  [plan] {n_flow} flow pairs (batch via {FLOW_CONDA_ENV}, "
            f"GPU), {n_var} variant images to scale+warp+archive")
        ex = _epoch_product_name(
            ctx, ref, ref['variants'][vkey('I', *SCIENCE_I)]['parsed'])
        log(f"  [plan] example product: {ref_date}/P3_{ctx['runlabel']}/{ex}")
        return

    os.makedirs(prep_dir, exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)

    tmpl_hdr, target_shape = cfp.build_template_header(ref['align_path'])
    ref_data, ref_hdr = cfp.load_image(ref['align_path'])
    ref_wcs = WCS(ref_hdr).celestial
    cd = abs(ref_hdr.get('CDELT1', 0.049))
    beam_pix = ref_hdr.get('BMAJ', 5 * cd) / cd
    horizon_mask = ~np.isfinite(ref_data) | (ref_data == 0)
    ctx['tmpl_hdr'], ctx['target_shape'] = tmpl_hdr, target_shape
    ctx['cdelt'] = cd

    # --- pass 1: flux scale on the alignment variant; flow manifest ---
    manifest = []
    for ep in epochs:
        date = ep['date']
        needs_regrid = date < cfp.CUTOVER_DATE
        work_path = ep['align_path']
        if needs_regrid:
            work_path = cfp.regrid_image(
                ep['align_path'], tmpl_hdr,
                os.path.join(prep_dir, f'{date}_regrid.fits'))
        tgt_data, tgt_hdr = cfp.load_image(work_path)
        if tgt_data.shape != target_shape:
            log(f"    {date}: shape {tgt_data.shape} != {target_shape}, "
                f"skipping epoch")
            ep['failed'] = True
            continue
        if date == ref_date:
            ep['scale_coeffs'] = [1.0, 0.0, 0.0]
            ep['weight'] = (1.0 / cfp.get_uniform_rms(ref_data,
                                                      horizon_mask) ** 2)
            ep['needs_regrid'] = needs_regrid
            continue
        tgt_mask = ~np.isfinite(tgt_data) | (tgt_data == 0)
        obs_time = cfp.get_midpoint_time(tgt_hdr, work_path)
        tgt_rms = cfp.get_uniform_rms(tgt_data, tgt_mask)
        scale_map, coeffs, n_src = cfp.measure_flux_scale(
            tgt_data, ref_data, ref_wcs, beam_pix,
            horizon_mask | tgt_mask, obs_time)
        mean_scale = float(coeffs[0])
        ep['scale_coeffs'] = [float(c) for c in coeffs]
        ep['weight'] = (float(mean_scale ** 2 / tgt_rms ** 2)
                        if tgt_rms > 0 else 0.0)
        ep['needs_regrid'] = needs_regrid
        # scaled alignment image is the flow input
        safe = np.where(np.abs(scale_map) > 0.1, scale_map, 1.0)
        scaled_path = os.path.join(prep_dir, f'{date}_scaled.fits')
        fits.writeto(scaled_path, (tgt_data / safe).astype(np.float32),
                     tgt_hdr, overwrite=True)
        manifest.append({'target': scaled_path,
                         'reference': ref['align_path'],
                         'out_dir': flow_dir})
        log(f"    {date}: scale={mean_scale:.3f} ({n_src} src) "
            f"w={ep['weight']:.3f}")

    # --- pass 2: batch optical flow (GPU, one JIT compile) ---
    todo = [m for m in manifest if not os.path.exists(os.path.join(
        flow_dir, os.path.basename(m['target']).replace(
            '.fits', '_flow.npy')))]
    if todo:
        man_path = os.path.join(workdir, 'work', 'flow_manifest.json')
        with open(man_path, 'w') as fh:
            json.dump(todo, fh)
        log(f"  Batch flow: {len(todo)} pairs")
        res = subprocess.run(
            ['conda', 'run', '-n', FLOW_CONDA_ENV, '--no-capture-output',
             'python', BATCH_FLOW_SCRIPT, '--manifest', man_path],
            timeout=14400)
        if res.returncode != 0:
            log(f"  WARNING: batch flow returned {res.returncode}")

    # --- pass 3: apply scale+flow to all variants, archive products ---
    h, w = target_shape
    yg, xg = np.indices((h, w), dtype=float)
    xgn, ygn = 2.0 * xg / w - 1.0, 2.0 * yg / h - 1.0

    for ep in epochs:
        if ep.get('failed'):
            continue
        date = ep['date']
        p3_dir = os.path.join(workdir, date, f"P3_{ctx['runlabel']}")
        os.makedirs(p3_dir, exist_ok=True)
        coeffs = ep['scale_coeffs']
        scale_map = coeffs[0] + coeffs[1] * xgn + coeffs[2] * ygn
        flow = None
        if date != ref_date:
            fpath = os.path.join(flow_dir, f'{date}_scaled_flow.npy')
            flow = np.load(fpath) if os.path.exists(fpath) else None
            if flow is None:
                log(f"    {date}: no flow field — flux-scale only")
        regrid_coords = None
        if ep['needs_regrid']:
            regrid_coords = cfp.precompute_regrid_mapping(
                ep['align_path'], tmpl_hdr, target_shape)

        ep['products'] = {}
        for key, var in ep['variants'].items():
            parsed = var['parsed']
            out_name = _epoch_product_name(ctx, ep, parsed)
            out_path = os.path.join(p3_dir, out_name)
            ep['products'][key] = out_path
            if os.path.exists(out_path):
                continue
            # One bad variant (blank image, corrupt file) must never
            # kill the run — log, drop the product, continue.
            try:
                data, src_hdr = cfp.load_image(var['path'])
                if data.shape != target_shape:
                    if regrid_coords is None:
                        log(f"    {date} {key}: shape mismatch, no regrid map")
                        continue
                    from scipy.ndimage import map_coordinates
                    data = map_coordinates(data, list(regrid_coords),
                                           order=1, mode='constant', cval=0.0)
                safe = np.where(np.abs(scale_map) > 0.1, scale_map, 1.0)
                data = data / safe
                if flow is not None:
                    data = cfp.apply_flow_field(data, flow)

                out_hdr = ctx['tmpl_hdr'].copy()
                for k in ['BMAJ', 'BMIN', 'BPA', 'BUNIT', 'TELESCOP']:
                    if k in src_hdr:
                        out_hdr[k] = src_hdr[k]
                fits.writeto(out_path, data.astype(np.float32), out_hdr,
                             overwrite=True)
                p3p.write_standard_header(
                    out_path, scope=band, stokes=parsed['stokes'],
                    start=ep['date_obs'], end=ep['date_end'],
                    inttime_s=ep['inttime'], time_basis='phase2',
                    beam=(src_hdr.get('BMAJ', 0), src_hdr.get('BMIN', 0),
                          src_hdr.get('BPA', 0)),
                    thermnoi=ep['thermnoi'],
                    imgrms=_measure_imgrms(data, parsed['stokes']),
                    pbcorr=parsed['pbcorr'], dewarped=parsed['dewarped'],
                    aligned=True, alignref=ref_date,
                    fluxref=ref_date, astref=ref_date,
                    scale_a=coeffs[0], p1run=ep['p1run'], p2run=ep['p2run'])
            except Exception as e:
                log(f"    {date} {key}: variant FAILED ({e}) — skipped")
                ep['products'].pop(key, None)
                if os.path.exists(out_path):
                    os.remove(out_path)
                continue
        log(f"    {date}: {len(ep['products'])} variants archived "
            f"→ {p3_dir}")

    # persist the prep manifest for resume
    save = [{k: v for k, v in ep.items()
             if k not in ('variants',)} for ep in epochs]
    for s, ep in zip(save, epochs):
        s['variants'] = {k: {'path': v['path'], 'parsed': v['parsed']}
                         for k, v in ep['variants'].items()}
    with open(ctx['manifest_path'], 'w') as fh:
        json.dump(save, fh, indent=1, default=str)


def load_prep_manifest(ctx):
    """Reload epoch state written by stage_prep (resume support)."""
    if not os.path.exists(ctx['manifest_path']):
        return False
    with open(ctx['manifest_path']) as fh:
        ctx['epochs'] = json.load(fh)
    ref = next(e for e in ctx['epochs'] if e['date'] == ctx['ref_date'])
    ctx['tmpl_hdr'], ctx['target_shape'] = cfp.build_template_header(
        ref['products'][vkey('I', *SCIENCE_I)])
    ctx['cdelt'] = abs(ctx['tmpl_hdr'].get('CDELT1', 0.049))
    return True


# ═══════════════════════════════════════════════════════════════════
# Stage 3 — EMPIRICAL BEAMS (pixel-integrated, zonal)
# ═══════════════════════════════════════════════════════════════════

def _fft_upsample(img, factor=2):
    """Band-limited (FFT zero-pad) upsample preserving sample values.

    The images are only ~3.2-3.5 px/FWHM, so Gaussian fits on the native
    grid alias pixelization into the fitted width; fitting on the x2
    band-limited grid is equivalent to a pixel-integrated fit at the
    original sampling. scipy.signal.resample is used because it gets
    the odd/even spectrum split right (a hand-rolled fftshift zero-pad
    misplaces DC by one bin for odd-sized stamps, which imprints a
    phase ramp that squeezes fitted widths).
    Output sample k lies at original coordinate k / factor.
    """
    from scipy.signal import resample
    h, w = img.shape
    return resample(resample(img, h * factor, axis=0), w * factor, axis=1)


def _fit_gaussian_stamp(img, iy, ix, r=8, upsample=2):
    """Fit one elliptical Gaussian on an upsampled stamp.

    Returns (fwhm_maj_px, fwhm_min_px, pa_deg_sky, peak) in ORIGINAL
    pixel units, or None. PA follows the phase3_subtract_lab convention
    (theta_pix = deg2rad(90 + bpa_sky) for x=-RA, y=Dec images).
    """
    st = img[iy - r:iy + r + 1, ix - r:ix + r + 1]
    if st.shape != (2 * r + 1, 2 * r + 1) or not np.isfinite(st).all():
        return None
    fine = _fft_upsample(st.astype(float), upsample)
    n = fine.shape[0]
    coords = np.mgrid[0:n, 0:n].astype(float) / upsample  # original px units
    yy, xx = coords[0] - r, coords[1] - r

    def g(c, a, x0, y0, sx, sy, th, o):
        X, Y = c
        xr = (X - x0) * np.cos(th) + (Y - y0) * np.sin(th)
        yr = -(X - x0) * np.sin(th) + (Y - y0) * np.cos(th)
        return (a * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
                + o).ravel()

    peak = float(np.nanmax(st))
    try:
        p, _ = curve_fit(
            g, (xx, yy), fine.ravel(),
            p0=[peak, 0.0, 0.0, 2.0, 2.0, 0.8, 0.0],
            bounds=([0, -3, -3, 0.5, 0.5, -np.pi, -np.inf],
                    [np.inf, 3, 3, 10, 10, np.pi, np.inf]),
            maxfev=8000)
    except Exception:
        return None
    a, x0, y0, sx, sy, th, off = p
    if sx >= sy:
        s_maj, s_min, th_maj = sx, sy, th
    else:
        s_maj, s_min, th_maj = sy, sx, th + np.pi / 2.0
    pa_sky = (np.rad2deg(th_maj) - 90.0) % 180.0
    return (s_maj / FWHM2SIG, s_min / FWHM2SIG, pa_sky, float(a))


def fit_empirical_beam_zonal(img, cdelt_deg, zones=3, nsrc_per_zone=30,
                             fmin=4.0, fmax=60.0, margin_frac=0.10,
                             upsample=2):
    """Zonal empirical beam from bright UNRESOLVED sources.

    Smear is elevation-dependent, so the field is split into zones x
    zones tiles over the central (1 - 2*margin_frac) of the image; each
    zone gets its own median beam, plus a pooled global beam. Fits with
    area > 1.25x the GLOBAL median fitted area are clipped as resolved
    (per Gregg — resolved sources bias the beam estimate broad).

    Returns dict:
      'global': (bmaj_deg, bmin_deg, bpa_deg, n_sources)
      'zones':  zones x zones nested list of the same tuple or None
    """
    n0, n1 = img.shape
    m0, m1 = int(n0 * margin_frac), int(n1 * margin_frac)
    work = np.where(np.isfinite(img), img, -np.inf).copy()
    work[:m0, :] = work[n0 - m0:, :] = -np.inf
    work[:, :m1] = work[:, n1 - m1:] = -np.inf

    zh = (n0 - 2 * m0) / zones
    zw = (n1 - 2 * m1) / zones
    fits_by_zone = [[[] for _ in range(zones)] for _ in range(zones)]
    counts = np.zeros((zones, zones), dtype=int)
    tried = np.zeros((zones, zones), dtype=int)

    while True:
        iy, ix = np.unravel_index(np.argmax(work), work.shape)
        v = work[iy, ix]
        if v < fmin:
            break
        work[max(0, iy - 12):iy + 12, max(0, ix - 12):ix + 12] = -np.inf
        zi = min(int((iy - m0) / zh), zones - 1)
        zj = min(int((ix - m1) / zw), zones - 1)
        if counts[zi, zj] >= nsrc_per_zone or tried[zi, zj] >= 120:
            if (counts >= nsrc_per_zone).all():
                break
            continue
        tried[zi, zj] += 1
        if v > fmax:
            continue
        res = _fit_gaussian_stamp(img, iy, ix, upsample=upsample)
        if res is None:
            continue
        fmaj, fmnr, pa, _peak = res
        if not (1.0 < fmaj < 20.0 and 1.0 < fmnr < 20.0):
            continue
        fits_by_zone[zi][zj].append((fmaj, fmnr, pa))
        counts[zi, zj] += 1

    all_fits = [f for row in fits_by_zone for z in row for f in z]
    if not all_fits:
        return {'global': None, 'zones': [[None] * zones
                                          for _ in range(zones)]}
    arr = np.array(all_fits)
    area = arr[:, 0] * arr[:, 1]
    med_area = np.median(area)
    keep_global = area <= 1.25 * med_area

    def _median_beam(sub):
        if len(sub) < 3:
            return None
        a = np.array(sub)
        return (float(np.median(a[:, 0])) * cdelt_deg,
                float(np.median(a[:, 1])) * cdelt_deg,
                float(np.median(a[:, 2])), len(sub))

    out_zones = []
    for zi in range(zones):
        row = []
        for zj in range(zones):
            sub = [f for f in fits_by_zone[zi][zj]
                   if f[0] * f[1] <= 1.25 * med_area]
            row.append(_median_beam(sub))
        out_zones.append(row)
    return {'global': _median_beam(arr[keep_global].tolist()),
            'zones': out_zones}


def stage_beams(ctx, dry_run=False):
    """Empirical beams per epoch per non-pbcorr I variant; V inherits I.

    V images have too few sources for an independent fit, and the V
    PSF equals the matching I weighting's PSF, so I beams are copied.
    """
    epochs = [e for e in ctx['epochs'] if not e.get('failed')]
    log(f"\n── Stage 3 BEAMS: {len(epochs)} epochs, zonal 3x3 ──")
    if dry_run:
        n_fits = sum(1 for e in epochs for k in e.get('variants', {})
                     if k.startswith('I_') and 'pbcorr' not in k)
        log(f"  [plan] {n_fits} zonal beam fits "
            f"(x2 FFT-upsampled, unresolved-filtered)")
        return

    cd = ctx['cdelt']
    for ep in epochs:
        beams = {}
        for key, path in ep['products'].items():
            if not key.startswith('I_') or 'pbcorr' in key:
                continue
            img, hdr = cfp.load_image(path)
            rms = inner_rms(img)
            fit = fit_empirical_beam_zonal(
                img, cd, fmin=max(4.0, 8.0 * rms))
            beams[key] = fit
            if fit['global']:
                bmaj, bmin, bpa, n = fit['global']
                log(f"    {ep['date']} {key}: "
                    f"{bmaj * 3600:.0f}\"x{bmin * 3600:.0f}\" "
                    f"@{bpa:.0f}deg ({n} srcs)")
        ep['ebeams'] = beams
        # stamp EBMAJ into all products (V + pbcorr inherit their I fit)
        for key, path in ep['products'].items():
            ikey = 'I_' + '_'.join(key.split('_')[1:]).replace('_pbcorr', '')
            src = beams.get(ikey, beams.get(vkey('I', *SCIENCE_I)))
            if src and src['global']:
                bmaj, bmin, bpa, _n = src['global']
                with fits.open(path, mode='update') as hdul:
                    hdul[0].header['EBMAJ'] = (
                        bmaj, '[deg] Empirical beam major (unresolved fits)')
                    hdul[0].header['EBMIN'] = (bmin,
                                               '[deg] Empirical beam minor')
                    hdul[0].header['EBPA'] = (bpa, '[deg] Empirical beam PA')
        with open(os.path.join(ctx['workdir'], ep['date'],
                               f"P3_{ctx['runlabel']}",
                               'empirical_beams.json'), 'w') as fh:
            json.dump(beams, fh, indent=1)
    with open(ctx['beams_path'], 'w') as fh:
        json.dump({e['date']: e['ebeams'] for e in epochs}, fh, indent=1)


# ═══════════════════════════════════════════════════════════════════
# Stage 4 — CANONICAL BEAM + conditional matching
# ═══════════════════════════════════════════════════════════════════

def match_to_canonical(data, beam, target_fwhm_deg, cdelt_deg):
    """match_to_beam with sub-pixel kernel support.

    When the deconvolution kernel sigma is < 0.7 px the native-grid
    kernel is undersampled and biases the result; the convolution is
    then done on the x2 band-limited grid (kernel sigmas doubled) and
    block-averaged back. Normalization is Jy/beam via beam areas exactly
    as phase3_subtract_lab.match_to_beam.
    """
    bmaj, bmin, bpa = beam
    sig_t = target_fwhm_deg * FWHM2SIG / cdelt_deg
    k_maj = np.sqrt(max(sig_t ** 2 - (bmaj * FWHM2SIG / cdelt_deg) ** 2, 0))
    k_min = np.sqrt(max(sig_t ** 2 - (bmin * FWHM2SIG / cdelt_deg) ** 2, 0))
    if min(k_maj, k_min) >= 0.7:
        return match_to_beam(data, beam, target_fwhm_deg, cdelt_deg)

    nanmask = ~np.isfinite(data)
    filled = np.where(nanmask, 0.0, data)
    up = _fft_upsample(filled, 2)
    theta = np.deg2rad(90.0 + bpa)
    kern = Gaussian2DKernel(x_stddev=max(2 * k_maj, 0.45),
                            y_stddev=max(2 * k_min, 0.45), theta=theta)
    kern.normalize()
    area_from = beam_area_px(bmaj, bmin, cdelt_deg)
    area_to = beam_area_px(target_fwhm_deg, target_fwhm_deg, cdelt_deg)
    sm = convolve_fft(up / area_from, kern, normalize_kernel=True,
                      nan_treatment='fill', fill_value=0.0, allow_huge=True)
    h2, w2 = sm.shape
    out = sm[:h2 - h2 % 2, :w2 - w2 % 2].reshape(
        h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3)) * area_to
    out = out[:data.shape[0], :data.shape[1]]
    out[nanmask] = np.nan
    return out


def compute_canonical_beams(ctx):
    """Canonical circular beam per I variant group across the LST set.

    1.05 x the 90th-percentile empirical BMAJ: large enough that nearly
    every epoch convolves UP to it (deconvolution is never needed), but
    not dominated by the single worst epoch.
    """
    epochs = [e for e in ctx['epochs'] if not e.get('failed')]
    canon = {}
    for key in {k for e in epochs for k in e.get('ebeams', {})}:
        majs = [e['ebeams'][key]['global'][0] for e in epochs
                if e.get('ebeams', {}).get(key, {}).get('global')]
        if len(majs) >= 3:
            canon[key] = CANON_FACTOR * float(
                np.percentile(majs, CANON_PERCENTILE))
    return canon


def stage_canon(ctx, dry_run=False):
    """Convolve every science-variant epoch image to the canonical beam,
    skipping epochs already within BEAM_MATCH_TOL (~the beam fit error).
    Outputs are working intermediates in work/canon/ — the permanent
    record of the canonical beam lives in the master/msub headers."""
    epochs = [e for e in ctx['epochs'] if not e.get('failed')]
    log(f"\n── Stage 4 CANON: conditional beam matching ──")
    if dry_run:
        log(f"  [plan] canonical FWHM = {CANON_FACTOR} x "
            f"P{CANON_PERCENTILE:.0f}(empirical BMAJ) per variant group; "
            f"convolve epochs differing by >{BEAM_MATCH_TOL:.1%}; "
            f"sub-pixel kernels built on the x2 grid")
        return

    canon = compute_canonical_beams(ctx)
    ctx['canonical'] = canon
    with open(os.path.join(ctx['coadd_dir'], 'canonical_beams.json'),
              'w') as fh:
        json.dump(canon, fh, indent=1)
    cd = ctx['cdelt']
    canon_dir = os.path.join(ctx['workdir'], 'work', 'canon')
    os.makedirs(canon_dir, exist_ok=True)

    sci_keys = [vkey('I', *SCIENCE_I), vkey('V', *SCIENCE_V)]
    for ep in epochs:
        ep['canon_paths'] = {}
        for key in sci_keys:
            if key not in ep.get('products', {}):
                continue
            ikey = 'I_' + '_'.join(key.split('_')[1:])
            target = canon.get(ikey)
            ebeam_info = ep.get('ebeams', {}).get(ikey, {}).get('global')
            if target is None or ebeam_info is None:
                continue
            ebeam = tuple(ebeam_info[:3])
            out = os.path.join(canon_dir, f"{ep['date']}_{key}.fits")
            ep['canon_paths'][key] = out
            if os.path.exists(out):
                continue
            data, hdr = cfp.load_image(ep['products'][key])
            if abs(ebeam[0] - target) / target > BEAM_MATCH_TOL:
                data = match_to_canonical(data, ebeam, target, cd)
                action = 'convolved'
            else:
                action = 'unchanged (within tol)'
            ohdr = hdr.copy()
            ohdr['BMAJ'] = ohdr['BMIN'] = target
            ohdr['BPA'] = 0.0
            fits.writeto(out, np.nan_to_num(data, nan=0.0).astype(
                np.float32), ohdr, overwrite=True)
            log(f"    {ep['date']} {key}: emp {ebeam[0] * 3600:.0f}\" → "
                f"canon {target * 3600:.0f}\"  [{action}]")


# ═══════════════════════════════════════════════════════════════════
# Stage 5 — MASTER CO-ADD (two-pass) + PROVENANCE
# ═══════════════════════════════════════════════════════════════════

def _coadd(entries, mask_zero=True):
    """Inverse-variance weighted mean of float arrays (zero = invalid)."""
    stack = np.array([cfp.load_image(e['path'])[0] for e in entries])
    w = np.array([e['weight'] for e in entries])[:, None, None]
    valid = stack != 0 if mask_zero else np.isfinite(stack)
    w_eff = w * valid
    sw = w_eff.sum(axis=0)
    sw[sw == 0] = 1.0
    out = (stack * w_eff).sum(axis=0) / sw
    out[valid.sum(axis=0) == 0] = 0.0
    return out


def _master_name(ctx, stokes, taper, robust, dates, extra_proc=()):
    return p3p.build_name(
        scope=ctx['band'], stokes=stokes, cadence='deep', taper=taper,
        robust=robust, proc=['dewarped', 'aligned', 'coadd'] +
        list(extra_proc),
        lst=ctx['lst'], date_start=min(dates), date_end=max(dates),
        n_epochs=len(dates))


def _write_master(ctx, key, entries, out_path):
    """Final co-add + full header + PROVENANCE table."""
    data = _coadd(entries)
    hdr = ctx['tmpl_hdr'].copy()
    stokes = key.split('_')[0]
    ikey = 'I_' + '_'.join(key.split('_')[1:])
    canon_fwhm = ctx['canonical'][ikey]
    starts = [Time(e['date_obs']) for e in entries]
    ends = [Time(e['date_end']) for e in entries]
    inttime = float(np.sum([e['inttime'] or 0.0 for e in entries]))
    fits.writeto(out_path, data.astype(np.float32), hdr, overwrite=True)
    p3p.write_standard_header(
        out_path, scope=ctx['band'], stokes=stokes,
        start=min(starts), end=max(ends), inttime_s=inttime,
        time_basis='coadd-aggregate',
        beam=(canon_fwhm, canon_fwhm, 0.0),
        imgrms=_measure_imgrms(data, stokes),
        thermnoi=None,
        pbcorr=False, dewarped=True, aligned=True,
        alignref=ctx['ref_date'], fluxref=ctx['ref_date'],
        astref=ctx['ref_date'],
        extra={'NCOADD': (len(entries), 'Number of co-added epochs'),
               'COADDW': ('inv_var', 'Co-addition weighting')})
    p3p.append_provenance_table(out_path, [
        dict(FILE=e['path'], MJD_OBS=e['mjd_obs'], MJD_MID=e['mjd_mid'],
             INTTIME=e['inttime'], WEIGHT=e['weight'],
             THERMNOI=e['thermnoi'], N_ANCHORS=e['n_anchors'],
             FLUXSCALE=e['fluxscale'], P1RUN=e['p1run'])
        for e in entries])
    return data


def stage_coadd(ctx, dry_run=False):
    """Two-pass master co-add per feedback_coadd_outlier_philosophy:
    provisional inverse-variance co-add → per-epoch residual rms
    (I: central region, V: inner 25% area) → MAD outlier rejection at
    OUTLIER_SIGMA (aggressive; outlier in EITHER Stokes excludes BOTH)
    → final co-add of survivors. All inputs are already at the
    canonical beam, so the master PSF is exact."""
    epochs = [e for e in ctx['epochs']
              if not e.get('failed') and (dry_run or e.get('canon_paths'))]
    log(f"\n── Stage 5 COADD: two-pass over {len(epochs)} epochs ──")
    dates = [e['date'] for e in epochs]
    if dry_run:
        if dates:
            log(f"  [plan] output: Coadd_{ctx['coadd_label']}/"
                f"{_master_name(ctx, 'I', 'Taper', 0.0, dates)}")
            log(f"  [plan] + V master, residuals.csv, PROVENANCE tables, "
                f"outlier sigma={OUTLIER_SIGMA} (joint I/V exclusion)")
        return

    def entry(ep, key):
        return dict(path=ep['canon_paths'][key], date=ep['date'],
                    weight=ep['weight'], mjd_obs=ep['mjd_obs'],
                    mjd_mid=ep['mjd_mid'], inttime=ep['inttime'],
                    thermnoi=ep['thermnoi'],
                    n_anchors=ep['n_anchors'],
                    fluxscale=ep['scale_coeffs'][0], p1run=ep['p1run'],
                    date_obs=ep['date_obs'], date_end=ep['date_end'])

    i_key, v_key = vkey('I', *SCIENCE_I), vkey('V', *SCIENCE_V)
    i_entries = [entry(e, i_key) for e in epochs if i_key in
                 e['canon_paths']]
    v_entries = [entry(e, v_key) for e in epochs if v_key in
                 e['canon_paths']]
    if len(i_entries) < MIN_DATES_FOR_COADD:
        log(f"  Only {len(i_entries)} epochs — skipping co-add")
        return

    # pass 1: provisional
    prov_i = _coadd(i_entries)
    prov_v = _coadd(v_entries) if len(v_entries) >= MIN_DATES_FOR_COADD \
        else None
    horizon = (prov_i == 0)

    rows = []
    for ep in epochs:
        i_rms = v_rms = np.nan
        if i_key in ep['canon_paths']:
            d, _ = cfp.load_image(ep['canon_paths'][i_key])
            i_rms = cfp.compute_residual_rms(d, prov_i, horizon,
                                             margin_frac=0.15)
        if prov_v is not None and v_key in ep['canon_paths']:
            d, _ = cfp.load_image(ep['canon_paths'][v_key])
            v_rms = cfp.compute_residual_rms(d, prov_v, horizon,
                                             margin_frac=0.25)
        rows.append(dict(date=ep['date'], i_rms=i_rms, v_rms=v_rms,
                         weight=ep['weight']))

    i_out, i_thr = cfp.detect_outliers_mad(
        [r['date'] for r in rows], [r['i_rms'] for r in rows],
        OUTLIER_SIGMA)
    v_out, v_thr = cfp.detect_outliers_mad(
        [r['date'] for r in rows], [r['v_rms'] for r in rows],
        OUTLIER_SIGMA)
    excluded = set(i_out) | set(v_out)   # joint MAD exclusion rule
    # --- cal bandpass amplitude max-σ gate + 3× thermal gate (2026-06-13) ---
    try:
        from exopipe.common import phase3_exclude as pex
        gpath = '/fast/gh/main/phase3/qa_assembly/cal_gain_summary.csv'
        ppath = '/fast/gh/main/phase3/qa_assembly/calibration_provenance.csv'
        cal_excl = set()
        if os.path.exists(gpath) and os.path.exists(ppath):
            gdf = pd.read_csv(gpath)
            pdf = pd.read_csv(ppath)
            pdf = pdf[pdf['lst'] == ctx['lst']]
            subs, _bad = pex.excluded_subband_epochs(gdf, pdf)
            cal_excl = {d for (l, d, s) in subs}
            if cal_excl:
                log(f"  CAL gate (bandpass max-σ>2): excluding {sorted(cal_excl)}")
        therm_excl = pex.thermal_outliers(
            [{'date': r['date'], 'thermnoi': e.get('thermnoi')}
             for r, e in zip(rows, epochs)])
        if therm_excl:
            log(f"  THERMAL gate (>3× median): excluding {sorted(therm_excl)}")
        excluded |= cal_excl | therm_excl
    except Exception as e:
        log(f"  [exclusion gate WARN] {e}")
    for r in rows:
        r['outlier'] = r['date'] in excluded
        r['i_outlier'] = r['date'] in set(i_out)
        r['v_outlier'] = r['date'] in set(v_out)
    pd.DataFrame(rows).to_csv(
        os.path.join(ctx['coadd_dir'], 'residuals.csv'), index=False)
    log(f"  Outliers (MAD {OUTLIER_SIGMA}σ): I>{i_thr * 1e3:.1f}mJy "
        f"({len(i_out)}), V>{v_thr * 1e3:.1f}mJy ({len(v_out)}) → "
        f"{len(excluded)} excluded: {sorted(excluded)}")

    # pass 2: final
    i_final = [e for e in i_entries if e['date'] not in excluded]
    v_final = [e for e in v_entries if e['date'] not in excluded]
    fdates = [e['date'] for e in i_final]
    i_path = os.path.join(ctx['coadd_dir'],
                          _master_name(ctx, 'I', 'Taper', 0.0, fdates))
    master_i = _write_master(ctx, i_key, i_final, i_path)
    ctx['master_i_path'] = i_path
    log(f"  I master: {len(i_final)} epochs → {os.path.basename(i_path)}")
    if len(v_final) >= MIN_DATES_FOR_COADD:
        v_path = os.path.join(
            ctx['coadd_dir'], _master_name(ctx, 'V', 'Taper', 0.0,
                                           [e['date'] for e in v_final]))
        master_v = _write_master(ctx, v_key, v_final, v_path)
        ctx['master_v_path'] = v_path
        # master THERMNOI from the V co-add itself (inner 25% area)
        with fits.open(v_path, mode='update') as hdul:
            tn = _measure_imgrms(master_v, 'V')
            hdul[0].header['THERMNOI'] = (
                tn, '[Jy/beam] Thermal noise (V co-add inner rms)')
        with fits.open(i_path, mode='update') as hdul:
            hdul[0].header['THERMNOI'] = (
                tn, '[Jy/beam] Thermal noise (from V master)')
        log(f"  V master: {len(v_final)} epochs → "
            f"{os.path.basename(v_path)}")
    ctx['included_dates'] = fdates


# ═══════════════════════════════════════════════════════════════════
# Stage 6 — SUBTRACTION (msub, V leakage screen, optional I ssub)
# ═══════════════════════════════════════════════════════════════════

def _poly2d_design(xn, yn, order):
    cols, terms = [], []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            cols.append((xn ** i) * (yn ** j))
            terms.append(f'x^{i}y^{j}')
    return np.column_stack(cols), terms


def fit_leakage_screen(data, order=4, mask=None, block=16):
    """Fit a smooth low-order 2D polynomial leakage screen.

    The residual x-y phase U→V leakage shows up after master
    subtraction as a smooth quadrupole across the field; a total-order
    <=4 polynomial captures it without absorbing compact sources, which
    are masked at 5 sigma (mad_std, dilated) before the fit. The fit is
    done on block medians (robust + fast) and evaluated at full
    resolution.

    Returns (screen, coeffs, terms).
    """
    h, w = data.shape
    finite = np.isfinite(data)
    rms = mad_std(data[finite]) if finite.any() else 0.0
    src = np.zeros_like(data, dtype=bool)
    if rms > 0:
        src = binary_dilation(np.abs(np.nan_to_num(data)) > 5.0 * rms,
                              iterations=3)
    valid = finite & ~src & (data != 0)

    nb0, nb1 = h // block, w // block
    med = np.full((nb0, nb1), np.nan)
    frac = np.zeros((nb0, nb1))
    for bi in range(nb0):
        for bj in range(nb1):
            cell = data[bi * block:(bi + 1) * block,
                        bj * block:(bj + 1) * block]
            ok = valid[bi * block:(bi + 1) * block,
                       bj * block:(bj + 1) * block]
            if ok.sum() > block * block * 0.3:
                med[bi, bj] = np.median(cell[ok])
                frac[bi, bj] = ok.mean()
    gy, gx = np.indices(med.shape, dtype=float)
    # block centers in normalized [-1, 1] full-image coordinates
    xn = (2.0 * (gx + 0.5) * block / w) - 1.0
    yn = (2.0 * (gy + 0.5) * block / h) - 1.0
    good = np.isfinite(med)
    A, terms = _poly2d_design(xn[good], yn[good], order)
    coeffs, _, _, _ = np.linalg.lstsq(A, med[good], rcond=None)

    yy, xx = np.indices((h, w), dtype=float)
    Af, _ = _poly2d_design((2.0 * xx / w - 1.0).ravel(),
                           (2.0 * yy / h - 1.0).ravel(), order)
    screen = (Af @ coeffs).reshape(h, w)
    return screen, coeffs, terms


def corner_asymmetry(data):
    """Quadrupole sign-asymmetry QA metric for V leakage.

    Quadrant medians over the inner half of the field (the xy-leakage
    quadrupole flips sign between adjacent quadrants):
        asym = (med(Q++) + med(Q--)) - (med(Q+-) + med(Q-+))
    Returned both raw [Jy/beam] and normalized by the inner rms.
    """
    h, w = data.shape
    qy, qx = h // 4, w // 4
    cy, cx = h // 2, w // 2
    inner = data[qy:3 * qy, qx:3 * qx]
    rms = mad_std(inner, ignore_nan=True)

    def qmed(ys, xs):
        v = data[ys, xs]
        v = v[np.isfinite(v) & (v != 0)]
        return float(np.median(v)) if v.size else 0.0
    pp = qmed(slice(cy, 3 * qy), slice(cx, 3 * qx))
    mm = qmed(slice(qy, cy), slice(qx, cx))
    pm = qmed(slice(cy, 3 * qy), slice(qx, cx))
    mp = qmed(slice(qy, cy), slice(cx, 3 * qx))
    raw = (pp + mm) - (pm + mp)
    return raw, (raw / rms if rms > 0 else np.nan)


def _msub_name(ctx, ep, stokes, taper, robust, extra_proc=(),
               native=False):
    tok = ('nmsub' if native else 'msub') + f"-c{ctx['coadd_gen']:02d}"
    return p3p.build_name(
        scope=ctx['band'], stokes=stokes, cadence='deep', taper=taper,
        robust=robust,
        proc=['dewarped', 'aligned', tok] + list(extra_proc),
        lst=ctx['lst'], epoch=ep['date_obs'])


def stage_msub(ctx, dry_run=False, striation=False, native=False):
    """epoch(canonical) - master, written beside the parent products.

    V residuals then get the low-order leakage screen removed (LEAKSUB,
    coefficients + corner-asymmetry stamped). I residuals optionally get
    striation removal (running median along Dec strips) → ssub.
    """
    epochs = [e for e in ctx['epochs']
              if not e.get('failed') and (dry_run or e.get('canon_paths'))]
    log(f"\n── Stage 6 MSUB: {len(epochs)} epochs - master "
        f"(gen c{ctx['coadd_gen']:02d}) ──")
    if dry_run:
        log(f"  [plan] I+V msub per epoch; V += leakage screen "
            f"(poly order<=4, 5σ source mask) + corner-asymmetry QA"
            + ("; I += striation ssub" if striation else ""))
        if epochs:
            ep = epochs[0]
            log(f"  [plan] example: {ep['date']}/P3_{ctx['runlabel']}/"
                f"{_msub_name(ctx, ep, 'V', 'Taper', 0.0)}")
        return
    if 'master_i_path' not in ctx:
        log("  No master co-add — skipping")
        return

    master_i, mhdr = cfp.load_image(ctx['master_i_path'])
    master_v = None
    if ctx.get('master_v_path'):
        master_v, _ = cfp.load_image(ctx['master_v_path'])
    wcs = WCS(ctx['tmpl_hdr']).celestial
    cd = ctx['cdelt']
    i_key, v_key = vkey('I', *SCIENCE_I), vkey('V', *SCIENCE_V)

    # source-density gate for the V leakage screen (per band/run)
    therms = [e['thermnoi'] for e in epochs if e.get('thermnoi')]
    med_therm = float(np.median(therms)) if therms else 0.03
    gate_ok, gate_n = leakage_screen_constrained(master_i, med_therm)
    ctx['leak_gate'] = gate_ok
    log(f"  Leakage-screen gate: {gate_n} I sources > "
        f"{LEAKSCREEN_FLUX_RMS:.0f}x thermal ({med_therm*1e3:.0f} mJy) → "
        f"screen {'ENABLED' if gate_ok else 'DISABLED (underconstrained)'}")

    for ep in epochs:
        p3_dir = os.path.join(ctx['workdir'], ep['date'],
                              f"P3_{ctx['runlabel']}")
        ep['msub_paths'] = {}
        included = ep['date'] in ctx.get('included_dates', [])
        canon_fwhm = ctx['canonical'][i_key]

        def _write_sub(sub, stokes, name, leak=None, ssub_flag=None,
                       extra=None):
            path = os.path.join(p3_dir, name)
            fits.writeto(path, sub.astype(np.float32), ctx['tmpl_hdr'],
                         overwrite=True)
            p3p.write_standard_header(
                path, scope=ctx['band'], stokes=stokes,
                start=ep['date_obs'], end=ep['date_end'],
                inttime_s=ep['inttime'], time_basis='phase2',
                beam=(canon_fwhm, canon_fwhm, 0.0),
                thermnoi=ep['thermnoi'],
                imgrms=_measure_imgrms(sub, stokes),
                pbcorr=False, dewarped=True, aligned=True,
                alignref=ctx['ref_date'], msub=True,
                msubfile=os.path.basename(ctx['master_i_path'] if
                                          stokes == 'I' else
                                          ctx['master_v_path']),
                ssub=ssub_flag, leaksub=leak,
                fluxref=ctx['ref_date'], astref=ctx['ref_date'],
                scale_a=ep['scale_coeffs'][0], p1run=ep['p1run'],
                p2run=ep['p2run'],
                extra=dict({'INCOADD': (included,
                                        'Epoch included in the master')},
                           **(extra or {})))
            return path

        if i_key in ep['canon_paths']:
            d, _ = cfp.load_image(ep['canon_paths'][i_key])
            sub = d - master_i
            sub[(d == 0) | (master_i == 0)] = 0.0
            name = _msub_name(ctx, ep, 'I', 'Taper', 0.0)
            ep['msub_paths']['I'] = _write_sub(sub, 'I', name)
            if striation:
                mask = (sub == 0) | ~np.isfinite(sub)
                model, _nb = cfp.build_striation_model(
                    sub, wcs, mask,
                    ra_window_beams=STRIATION_WINDOW_DEG / cd,
                    beam_pix=1.0)
                clean = sub - model
                clean[mask] = 0.0
                name = _msub_name(ctx, ep, 'I', 'Taper', 0.0, ['ssub'])
                ep['msub_paths']['I_ssub'] = _write_sub(
                    clean, 'I', name, ssub_flag=True)

        if master_v is not None and v_key in ep['canon_paths']:
            d, _ = cfp.load_image(ep['canon_paths'][v_key])
            sub = d - master_v
            invalid = (d == 0) | (master_v == 0)
            sub[invalid] = np.nan
            rms0 = _measure_imgrms(np.nan_to_num(sub), 'V')
            asym0_raw, asym0 = corner_asymmetry(sub)
            if ctx.get('leak_gate', True):
                screen, coeffs, terms = fit_leakage_screen(sub)
            else:
                screen, coeffs, terms = 0.0, [], []
            sub_clean = sub - screen
            # V is ALWAYS striation-subtracted (the original V flow's
            # V_clean): a ~7 mJy striation pattern dominates at the
            # ~25 mJy thermal floor (2026-06-12, 12-27 inspection:
            # SNR +3.8 -> +4.4 on the candidate).
            _smask = invalid | ~np.isfinite(sub_clean)
            _smodel, _ = cfp.build_striation_model(
                np.nan_to_num(sub_clean), wcs, _smask,
                ra_window_beams=STRIATION_WINDOW_DEG / cd, beam_pix=1.0)
            sub_clean = sub_clean - _smodel
            sub_clean[invalid] = 0.0
            rms1 = _measure_imgrms(sub_clean, 'V')
            asym1_raw, asym1 = corner_asymmetry(sub_clean)
            extra = {'VLKASYM0': (round(asym0, 3),
                                  'Corner asymmetry/rms before LEAKSUB'),
                     'VLKASYM1': (round(asym1, 3),
                                  'Corner asymmetry/rms after LEAKSUB')}
            for k, c in enumerate(coeffs):
                extra[f'LKSC{k:02d}'] = (float(c),
                                         f'Leakage screen coeff {terms[k]}')
            name = _msub_name(ctx, ep, 'V', 'Taper', 0.0, ['ssub'])
            ep['msub_paths']['V'] = _write_sub(
                sub_clean, 'V', name, leak=bool(ctx.get('leak_gate', True)),
                ssub_flag=True, extra=extra)
            log(f"    {ep['date']}: V leakage screen rms "
                f"{rms0 * 1e3:.1f}→{rms1 * 1e3:.1f} mJy, "
                f"asym {asym0:+.2f}→{asym1:+.2f}")

    if native:
        _native_msub(ctx, epochs)


def _native_msub(ctx, epochs):
    """Native-beam msub (nmsub): per-epoch masters, maximum sensitivity.

    Canonical msub convolves every epoch to the P90 beam before
    subtraction — uniform PSF, but it taxes point-source SNR most on
    the SHARPEST epochs (up to ~30%), i.e. exactly the good nights that
    dominate science yield (2026-06-11 discussion). Here each epoch is
    subtracted at its NATIVE beam against a master built from the other
    epochs matched DOWN to it: sharper epochs convolve (allowed),
    broader ones are admitted unmatched only when needed to reach a
    minimum membership (their inverse-variance weights are small, so
    the residual mismatch is second-order). The master noise is
    ~sqrt(N) below the epoch's, so subset masters stay negligible.

    Writes nmsub-cNN products beside the canonical msub ones and points
    ep['msub_paths'] at them so sigma_stack and detection (stages 7-8)
    operate on the native-sensitivity residuals; canonical paths are
    preserved under ep['msub_paths_canon'].
    """
    log(f"\n  ── native-beam msub (per-epoch masters) ──")
    i_key, v_key = vkey('I', *SCIENCE_I), vkey('V', *SCIENCE_V)
    cd = ctx['cdelt']
    included = set(ctx.get('included_dates', []))

    cache = {}
    for ep in epochs:
        eb = ep.get('ebeams', {}).get(i_key, {}).get('global')
        if not eb:
            continue
        ent = {'ebeam': tuple(eb[:3]), 'ep': ep}
        for sk, key in (('I', i_key), ('V', v_key)):
            p = ep.get('products', {}).get(key)
            if p and os.path.exists(p):
                ent[sk], _ = cfp.load_image(p)
        if 'I' not in ent:
            continue
        rms = inner_rms(np.where(ent['I'] == 0, np.nan, ent['I']))
        ent['w'] = 1.0 / rms ** 2 if np.isfinite(rms) and rms > 0 else 0.0
        cache[ep['date']] = ent

    for date, ent in cache.items():
        ep = ent['ep']
        tgt = ent['ebeam']
        sharp, broad = [], []
        for d2, e2 in cache.items():
            if d2 == date or (included and d2 not in included):
                continue
            (sharp if e2['ebeam'][0] <= tgt[0] * (1 + BEAM_MATCH_TOL)
             else broad).append(d2)
        members = sharp if len(sharp) >= 8 else sharp + broad
        if len(members) < MIN_EPOCHS_SIGMA:
            log(f"    {date}: only {len(members)} master members — skipped")
            continue

        ep['msub_paths_canon'] = dict(ep.get('msub_paths', {}))
        p3_dir = os.path.join(ctx['workdir'], date,
                              f"P3_{ctx['runlabel']}")
        for sk in ('I', 'V'):
            if sk not in ent:
                continue
            acc, wsum = None, 0.0
            for d2 in members:
                e2 = cache[d2]
                if sk not in e2:
                    continue
                arr = e2[sk]
                if e2['ebeam'][0] < tgt[0] and \
                        abs(e2['ebeam'][0] - tgt[0]) / tgt[0] > BEAM_MATCH_TOL:
                    arr = match_to_canonical(arr, e2['ebeam'], tgt[0], cd)
                acc = arr * e2['w'] if acc is None else acc + arr * e2['w']
                wsum += e2['w']
            if acc is None or wsum <= 0:
                continue
            master = acc / wsum
            data = ent[sk]
            sub = data - master
            invalid = (data == 0) | (master == 0)
            extra = {'NMASTERN': (len(members),
                                  'Epochs in per-epoch native master'),
                     'NATIVMSB': (True,
                                  'Native-beam msub (per-epoch master)')}
            leak = None
            if sk == 'V':
                sub[invalid] = np.nan
                asym0_raw, asym0 = corner_asymmetry(sub)
                if ctx.get('leak_gate', True):
                    screen, coeffs, terms = fit_leakage_screen(sub)
                else:
                    screen, coeffs, terms = 0.0, [], []
                sub = sub - screen
                _wcs_n = WCS(ctx['tmpl_hdr']).celestial
                _smask = invalid | ~np.isfinite(sub)
                _smodel, _ = cfp.build_striation_model(
                    np.nan_to_num(sub), _wcs_n, _smask,
                    ra_window_beams=STRIATION_WINDOW_DEG / cd,
                    beam_pix=1.0)
                sub = sub - _smodel
                asym1_raw, asym1 = corner_asymmetry(sub)
                extra['VLKASYM0'] = (round(asym0, 3),
                                     'Corner asym/rms before LEAKSUB')
                extra['VLKASYM1'] = (round(asym1, 3),
                                     'Corner asym/rms after LEAKSUB')
                leak = bool(ctx.get('leak_gate', True))
            sub[invalid] = 0.0
            name = _msub_name(ctx, ep, sk, 'Taper', 0.0,
                              ['ssub'] if sk == 'V' else (),
                              native=True)
            path = os.path.join(p3_dir, name)
            fits.writeto(path, sub.astype(np.float32), ctx['tmpl_hdr'],
                         overwrite=True)
            p3p.write_standard_header(
                path, scope=ctx['band'], stokes=sk,
                start=ep['date_obs'], end=ep['date_end'],
                inttime_s=ep['inttime'], time_basis='phase2',
                beam=tgt, ebeam=tgt,
                thermnoi=ep['thermnoi'],
                imgrms=_measure_imgrms(sub, sk),
                pbcorr=False, dewarped=True, aligned=True,
                alignref=ctx['ref_date'], msub=True,
                msubfile=f'per-epoch-native-master-N{len(members)}',
                leaksub=leak, ssub=(True if sk == 'V' else None),
                fluxref=ctx['ref_date'], astref=ctx['ref_date'],
                scale_a=ep['scale_coeffs'][0], p1run=ep['p1run'],
                p2run=ep['p2run'],
                extra=dict({'INCOADD': (date in included,
                                        'Epoch included in the master')},
                           **extra))
            ep['msub_paths'][sk] = path
        log(f"    {date}: nmsub at native {tgt[0]*3600:.0f}\" "
            f"({len(members)} master members)")


# ═══════════════════════════════════════════════════════════════════
# Stage 7 — SIGMA_STACK
# ═══════════════════════════════════════════════════════════════════

def build_sigma_stack(resid_paths, min_epochs=MIN_EPOCHS_SIGMA):
    """Per-pixel robust noise-structure map from msub residuals.

    Each residual is normalized by its own global inner rms (so nights
    of different depth contribute equally to the STRUCTURE estimate),
    then the per-pixel MAD is taken across epochs, renormalized so the
    clean-sky median is 1, and FLOORED AT 1.0: the map can only RAISE
    the detection threshold (near bright sources, sidelobe ridges, the
    Vir A halo) — clean sky keeps native sensitivity. Per Gregg this
    replaces fixed exclusion holes with proper normalization.
    """
    if len(resid_paths) < min_epochs:
        return None
    stack = []
    for p in resid_paths:
        d, _ = cfp.load_image(p)
        rms = inner_rms(d)
        if not np.isfinite(rms) or rms <= 0:
            continue
        d = np.where(d == 0, np.nan, d)
        stack.append(d / rms)
    if len(stack) < min_epochs:
        return None
    stack = np.array(stack)
    med = np.nanmedian(stack, axis=0)
    madmap = 1.4826 * np.nanmedian(np.abs(stack - med[None]), axis=0)
    n = madmap.shape[0]
    q = n // 4
    sky = np.nanmedian(madmap[q:3 * q, q:3 * q])
    madmap = madmap / sky
    madmap[~np.isfinite(madmap)] = 1.0
    return np.maximum(madmap, 1.0)


def stage_sigma(ctx, dry_run=False):
    epochs = [e for e in ctx['epochs'] if e.get('msub_paths')]
    log(f"\n── Stage 7 SIGMA_STACK ──")
    if dry_run:
        log(f"  [plan] per-pixel MAD over >= {MIN_EPOCHS_SIGMA} msub "
            f"residuals per Stokes, clean-sky normalized, floored at 1.0")
        return
    for stokes in ['I', 'V']:
        paths = [e['msub_paths'][stokes] for e in epochs
                 if stokes in e.get('msub_paths', {})]
        smap = build_sigma_stack(paths)
        if smap is None:
            log(f"  {stokes}: <{MIN_EPOCHS_SIGMA} residuals, skipped")
            continue
        dates = sorted(e['date'] for e in epochs
                       if stokes in e.get('msub_paths', {}))
        name = p3p.build_name(
            scope=ctx['band'], stokes=stokes, cadence='deep',
            taper='Taper', robust=0.0,
            proc=['dewarped', 'aligned', 'coadd', 'sigstack'],
            lst=ctx['lst'], date_start=min(dates), date_end=max(dates),
            n_epochs=len(paths))
        path = os.path.join(ctx['coadd_dir'], name)
        hdr = ctx['tmpl_hdr'].copy()
        hdr['BUNIT'] = ('SIGMA_RATIO', 'Local/global noise ratio, >=1')
        fits.writeto(path, smap.astype(np.float32), hdr, overwrite=True)
        ctx[f'sigma_{stokes}'] = path
        log(f"  {stokes}: {len(paths)} epochs → {name} "
            f"(max {np.nanmax(smap):.1f})")


# ═══════════════════════════════════════════════════════════════════
# Stage 8 — DETECTION (injection/recovery validation)
# ═══════════════════════════════════════════════════════════════════

def _detection_header(ctx, ep, stokes):
    """Header for the identification chains: template WCS + epoch time
    + canonical beam + exact frequency axis (BDSF requires CTYPE3)."""
    hdr = ctx['tmpl_hdr'].copy()
    canon = ctx['canonical'][vkey('I', *SCIENCE_I)]
    hdr['BMAJ'] = hdr['BMIN'] = canon
    hdr['BPA'] = 0.0
    hdr['BUNIT'] = 'JY/BEAM'
    p3p.write_standard_header(
        hdr, scope=ctx['band'], stokes=stokes,
        start=ep['date_obs'], end=ep['date_end'],
        inttime_s=ep['inttime'], time_basis='phase2')
    return hdr


def _apply_sigma_stack(cands, sigma_stack, base_thresh):
    """Multiplicative local threshold from the noise-structure map."""
    if sigma_stack is None:
        return 0
    n_rej = 0
    h, w = sigma_stack.shape
    for c in cands:
        if c['status'] != 'PASS':
            continue
        y = int(np.clip(c['pix_y'], 0, h - 1))
        x = int(np.clip(c['pix_x'], 0, w - 1))
        thr = base_thresh * max(float(sigma_stack[y, x]), 1.0)
        if abs(c.get('snr', 0.0)) < thr:
            c['status'] = 'REJECT_SIGMA_STACK'
            c['reject_reason'] = (f"SNR {c.get('snr', 0):+.1f} < "
                                  f"{thr:.1f} (sigma_stack)")
            n_rej += 1
    return n_rej


def run_identification_i(diff, master, hdr, bpix, sigma_stack=None):
    """The real Stokes I chain (search_i_transients) on a residual, with
    the sigma_stack as a multiplicative local threshold."""
    from exopipe.phase3 import search_i_transients as sit
    wcs = WCS(hdr).celestial
    horizon, ateam, badwcs, obs_time, _ = sit.make_masks(hdr, diff.shape,
                                                         wcs)
    excl = horizon | ateam | badwcs
    diff_rms = sit.get_uniform_rms(diff, horizon)
    cands = sit.extract_positive_peaks(
        diff.astype(np.float32), hdr, horizon, ateam, badwcs,
        wcs, obs_time, diff_rms, bpix)
    sit.assign_local_snr(cands, diff, excl, bpix,
                         sit.CONFIG['SIGMA_THRESHOLD'])
    _apply_sigma_stack(cands, sigma_stack, sit.CONFIG['SIGMA_THRESHOLD'])
    bright = sit.find_bright_sources(
        master, horizon, wcs, bpix, sit.CONFIG['BRIGHT_MASK_FLUX_JY'])
    sit.reject_bright_proximity(cands, bright, bpix, wcs)
    sit.merge_nearby(cands, bpix, wcs)
    return [c for c in cands if c['status'] == 'PASS']


def run_identification_v(vresid, master_i, hdr, bpix, sigma_stack=None,
                         sigma_thresh=6.0, i_leakage_mult=None):
    """Signed V detection (search_v_transients_v2.extract_v_peaks runs
    BDSF on +V and -V — both circular polarization handednesses), local SNR,
    and optional sigma_stack threshold. The I-counterpart leakage rule is OFF
    by default (i_leakage_mult=None): the veto-ROC test (Gregg 2026-06-30) showed
    it costs ~25% (Red) / ~14% (Green) injection retention for negligible FP
    rejection beyond mstruct — mstruct + morphology veto carry the FP rejection.
    Pass i_leakage_mult=20.0 to restore the old rule (reject if peak|I|>mult*|V|)."""
    from exopipe.phase3 import search_v_transients_v2 as sv2
    wcs = WCS(hdr).celestial
    horizon, ateam, badwcs, obs_time, _coords, _el = sv2.make_masks(
        hdr, vresid.shape)
    v_rms = float(sv2.get_uniform_rms(vresid, horizon))
    cands = sv2.extract_v_peaks(
        vresid.astype(np.float32), hdr, horizon, ateam, badwcs,
        wcs, obs_time, sv2.CONFIG['CANDIDATE_SIGMA'], v_rms)
    sv2.assign_local_snr(cands, vresid, horizon | ateam | badwcs, bpix,
                         sigma_thresh)
    _apply_sigma_stack(cands, sigma_stack, sigma_thresh)
    if i_leakage_mult is not None:        # OFF by default (Gregg 2026-06-30)
        rb = int(round(bpix))
        h, w = master_i.shape
        for c in cands:
            if c['status'] != 'PASS':
                continue
            y, x = int(c['pix_y']), int(c['pix_x'])
            box = master_i[max(0, y - rb):min(h, y + rb + 1),
                           max(0, x - rb):min(w, x + rb + 1)]
            ipk = float(np.nanmax(np.abs(box))) if box.size else 0.0
            if ipk > i_leakage_mult * abs(c['flux']):
                c['status'] = 'REJECT_I_LEAKAGE'
                c['reject_reason'] = f'I counterpart {ipk:.1f} Jy > {i_leakage_mult:.0f}x V'
    return [c for c in cands if c['status'] == 'PASS']


def classify_fps(fps, master, diff, bpix, wcs):
    """FP diagnosis per phase3_final_metric: master cross-match to 2 Jy
    + dipole trough test → on-source-residual / dipole / near-source /
    unexplained."""
    mwork = np.where(np.isfinite(master), master, -np.inf).copy()
    msrcs = []
    while True:
        iy, ix = np.unravel_index(np.argmax(mwork), mwork.shape)
        v = mwork[iy, ix]
        if v < 2.0 or len(msrcs) >= 2000:
            break
        msrcs.append((ix, iy, v))
        mwork[max(0, iy - 6):iy + 6, max(0, ix - 6):ix + 6] = -np.inf
    msx = np.array([s[0] for s in msrcs], float)
    msy = np.array([s[1] for s in msrcs], float)
    msf = np.array([s[2] for s in msrcs], float)

    rb = int(round(bpix))
    diag = []
    for c in fps:
        x, y = int(c['pix_x']), int(c['pix_y'])
        flux = abs(float(c['flux']))
        mbox = master[max(0, y - rb):y + rb + 1, max(0, x - rb):x + rb + 1]
        mpk = float(np.nanmax(np.abs(mbox))) if mbox.size else 0.0
        d = (np.sqrt((msx - x) ** 2 + (msy - y) ** 2) / bpix
             if len(msx) else np.array([]))
        j = int(np.argmin(d)) if len(d) else -1
        dsrc = float(d[j]) if j >= 0 else 99.0
        fsrc = float(msf[j]) if j >= 0 else 0.0
        r2 = int(round(2 * bpix))
        dbox = diff[max(0, y - r2):y + r2 + 1, max(0, x - r2):x + r2 + 1]
        # dipole test: an opposite-sign trough comparable to the peak
        # within 2 beams marks an astrometric-residual dipole
        sgn = np.sign(float(c['flux'])) or 1.0
        trough = float(np.nanmin(sgn * dbox)) if dbox.size else 0.0
        dipole = trough / flux if flux > 0 else 0.0
        if dsrc < 1.5 and fsrc > 5 * flux:
            klass = 'on-source-residual'
        elif dipole < -0.6 and dsrc < 4:
            klass = 'dipole'
        elif dsrc < 4:
            klass = 'near-source'
        else:
            klass = 'unexplained'
        sky = wcs.pixel_to_world(x, y)
        diag.append(dict(
            x=x, y=y, ra=round(sky.ra.deg, 5), dec=round(sky.dec.deg, 5),
            flux_jy=round(float(c['flux']), 3),
            local_snr=round(float(c.get('snr', c.get('local_snr', 0))), 1),
            master_peak_jy=round(mpk, 2), dist_src_beams=round(dsrc, 2),
            src_flux_jy=round(fsrc, 2), dipole_ratio=round(dipole, 2),
            klass=klass))
    return diag


_CLASS_COLORS = {'on-source-residual': 'red', 'dipole': 'magenta',
                 'near-source': 'yellow', 'unexplained': 'cyan'}


def write_review_packet(out_dir, label, resid_inj, hdr, epoch_path,
                        master_path, diag, injections, recovered):
    """Human-review packet: residual(+injections) FITS with WCS, links
    to the epoch + master products, class-colored DS9 regions, injection
    boxes, and a CSV diagnosis."""
    os.makedirs(out_dir, exist_ok=True)
    rh = WCS(hdr).celestial.to_header()
    for k in ['BMAJ', 'BMIN', 'BPA', 'BUNIT']:
        if k in hdr:
            rh[k] = hdr[k]
    fits.writeto(os.path.join(out_dir, f'{label}_resid_inj.fits'),
                 resid_inj.astype(np.float32), rh, overwrite=True)
    for src, ln in [(epoch_path, f'{label}_epoch.fits'),
                    (master_path, f'{label}_master.fits')]:
        link = os.path.join(out_dir, ln)
        if src and not os.path.exists(link):
            os.symlink(os.path.abspath(src), link)

    with open(os.path.join(out_dir, f'{label}_fp.reg'), 'w') as fh:
        fh.write('# Region file format: DS9\nimage\n')
        for d in diag:
            color = _CLASS_COLORS.get(d['klass'], 'white')
            fh.write(f"circle({d['x'] + 1},{d['y'] + 1},10) # "
                     f"color={color} text={{{d['klass']} "
                     f"{d['flux_jy']:.2f}Jy}}\n")
    with open(os.path.join(out_dir, f'{label}_injections.reg'), 'w') as fh:
        fh.write('# Region file format: DS9\nimage\n')
        for k, (x, y, f) in enumerate(injections):
            tag = 'REC' if k in recovered else 'MISSED'
            fh.write(f"box({x + 1},{y + 1},20,20,0) # color=green "
                     f"text={{{tag} {f:.2f}Jy}}\n")
    if diag:
        pd.DataFrame(diag).to_csv(
            os.path.join(out_dir, f'{label}_diagnosis.csv'), index=False)


def stage_detect(ctx, dry_run=False):
    """Validation-mode detection with UPSTREAM injections: canonical-beam
    point sources are planted in the canon (aligned + beam-matched) epoch
    image, propagated through master self-contamination (the injected
    epoch contributes weight/sum_w of its flux to the master), the
    subtraction, the V leakage screen and striation removal, and only
    then the real identification chains. Flux transfer is measured per
    stage by differencing paired with/without-injection images at the
    injected pixels -> flux_transfer.csv. NOT covered: alignment
    resampling (injections enter post-align; the flow warp is not
    re-run) and the native-msub branch (canonical chain only)."""
    epochs = [e for e in ctx['epochs']
              if e.get('msub_paths') or (dry_run and not e.get('failed'))]
    # INCOADD propagation: excluded epochs are NOT searched (2026-06-13).
    incl_dates = set(ctx.get('included_dates', []))
    if incl_dates and not dry_run:
        n0 = len(epochs)
        epochs = [e for e in epochs if e['date'] in incl_dates]
        if len(epochs) < n0:
            log(f"  INCOADD: searching {len(epochs)}/{n0} included epochs "
                f"(excluded epochs not searched)")
    log(f"\n── Stage 8 DETECT: upstream injection validation on "
        f"{len(epochs)} epochs ──")
    if dry_run:
        log(f"  [plan] {N_INJECT} injections/epoch at "
            f"{INJECT_FLUX_SIGMAS}x sigma planted in canon images, "
            f"propagated through msub/screen/ssub with per-stage flux "
            f"transfer; I chain = search_i_transients + sigma_stack; "
            f"V chain = signed extract_v_peaks + 20x I rule")
        return
    if 'master_i_path' not in ctx:
        log("  No master — skipping")
        return

    master_i, _ = cfp.load_image(ctx['master_i_path'])
    master_v = None
    if ctx.get('master_v_path'):
        master_v, _ = cfp.load_image(ctx['master_v_path'])
    canon = ctx['canonical'][vkey('I', *SCIENCE_I)]
    cd = ctx['cdelt']
    wcs = WCS(ctx['tmpl_hdr']).celestial
    bpix = canon / cd
    cbeam = (canon, canon, 0.0)
    rng = np.random.default_rng(20260610)
    sigma = {}
    for stokes in ['I', 'V']:
        spath = ctx.get(f'sigma_{stokes}')
        sigma[stokes] = cfp.load_image(spath)[0] if spath else None
    incl = ctx.get('included_dates', [])
    sumw = sum(e['weight'] for e in ctx['epochs']
               if e['date'] in incl and not e.get('failed')) or 1.0

    results, transfer = [], []
    for ep in epochs:
        for stokes in ['I', 'V']:
            key = 'I_ssub' if (stokes == 'I' and
                               'I_ssub' in ep['msub_paths']) else stokes
            ck = vkey(stokes, *(SCIENCE_I if stokes == 'I'
                                else SCIENCE_V))
            if (key not in ep['msub_paths']
                    or ck not in ep.get('canon_paths', {})):
                continue
            master = master_i if stokes == 'I' else master_v
            if master is None:
                continue
            d, _ = cfp.load_image(ep['canon_paths'][ck])
            resid, _ = cfp.load_image(ep['msub_paths'][key])
            resid = np.where(resid == 0, np.nan, resid)
            sig0 = inner_rms(resid)
            n = resid.shape[0]
            hdr = _detection_header(ctx, ep, stokes)

            # injections on clean sky (|master I| < 2 Jy), planted in
            # the canon image — upstream of subtraction + cleaning
            inj = []
            for k in range(N_INJECT):
                while True:
                    x = int(rng.integers(n // 4 + 50, 3 * n // 4 - 50))
                    y = int(rng.integers(n // 4 + 50, 3 * n // 4 - 50))
                    if np.isfinite(resid[y, x]) and abs(
                            master_i[y, x]) < 2.0:
                        break
                f = INJECT_FLUX_SIGMAS[k % len(INJECT_FLUX_SIGMAS)] * sig0
                if stokes == 'V' and k % 2 == 1:
                    f = -f      # opposite circular polarization handedness
                inj.append((x, y, f))
            stamps = np.zeros(d.shape)
            for x, y, f in inj:
                stamps += gaussian_stamp(d.shape, x, y, f, cbeam, cd)
            # master self-contamination: an included epoch contributes
            # weight/sum_w of any real source's flux to the master
            leak = (ep['weight'] / sumw) if ep['date'] in incl else 0.0

            def _chain(img, mst):
                """canon → msub → [screen] → [ssub], mirroring stage 6;
                returns each stage image for flux-transfer differencing."""
                out = {}
                sub = img - mst
                invalid = (img == 0) | (mst == 0)
                sub[invalid] = np.nan
                out['msub'] = sub.copy()
                cur = sub
                if stokes == 'V' and ctx.get('leak_gate', True):
                    screen, _c, _t = fit_leakage_screen(cur)
                    cur = cur - screen
                    out['screen'] = cur.copy()
                if stokes == 'V' or key == 'I_ssub':
                    smask = invalid | ~np.isfinite(cur)
                    smodel, _nb = cfp.build_striation_model(
                        np.nan_to_num(cur), wcs, smask,
                        ra_window_beams=STRIATION_WINDOW_DEG / cd,
                        beam_pix=1.0)
                    cur = cur - smodel
                    out['ssub'] = cur.copy()
                fin = np.nan_to_num(cur, nan=0.0)
                fin[invalid] = 0.0
                out['final'] = fin
                return out

            base = _chain(d, master)
            withinj = _chain(d + stamps, master + leak * stamps)
            resid_inj = withinj['final']
            for x, y, f in inj:
                row = dict(date=ep['date'], stokes=stokes,
                           level=round(abs(f) / sig0, 1),
                           f_inj_mjy=round(f * 1e3, 1),
                           recovered=0, f_meas_mjy=np.nan)
                for st in ['msub', 'screen', 'ssub']:
                    if st in base:
                        dv = withinj[st][y, x] - base[st][y, x]
                        row[f'after_{st}'] = (round(float(dv / f), 3)
                                              if np.isfinite(dv) else
                                              np.nan)
                transfer.append(row)

            if stokes == 'I':
                passed = run_identification_i(resid_inj, master_i, hdr,
                                              bpix, sigma['I'])
            else:
                passed = run_identification_v(resid_inj, master_i, hdr,
                                              bpix, sigma['V'])

            rmatch = 1.5 * bpix
            rec, fps = set(), []
            for c in passed:
                d2 = [(c['pix_x'] - x) ** 2 + (c['pix_y'] - y) ** 2
                      for x, y, _f in inj]
                j = int(np.argmin(d2))
                # signed match for V: detection must match polarity
                pol_ok = (stokes == 'I' or
                          np.sign(c['flux']) == np.sign(inj[j][2]))
                if d2[j] <= rmatch ** 2 and pol_ok:
                    rec.add(j)
                    trow = transfer[len(transfer) - len(inj) + j]
                    trow['recovered'] = 1
                    trow['f_meas_mjy'] = round(float(c['flux']) * 1e3, 1)
                else:
                    fps.append(c)
            diag = classify_fps(fps, master_i, resid_inj, bpix,
                                WCS(hdr).celestial)

            review_dir = os.path.join(ctx['workdir'], ep['date'],
                                      f"P3_{ctx['runlabel']}", 'review')
            write_review_packet(
                review_dir, f"{ep['date']}_{stokes}", resid_inj, hdr,
                ep['msub_paths'].get(key), ctx['master_i_path'],
                diag, inj, rec)

            imgrms = _measure_imgrms(np.nan_to_num(resid, nan=0.0),
                                     stokes)
            tn = ep['thermnoi'] or np.nan
            results.append(dict(
                date=ep['date'], stokes=stokes,
                rms_mjy=round(imgrms * 1e3, 2),
                rms_over_therm=round(imgrms / tn, 2)
                if tn and tn > 0 else np.nan,
                n_pass=len(passed), final_fp=len(fps),
                completeness=round(len(rec) / len(inj), 3),
                classes=json.dumps(
                    pd.Series([d['klass'] for d in diag])
                    .value_counts().to_dict())))
            r = results[-1]
            log(f"    {ep['date']} {stokes}: FP={r['final_fp']:3d} "
                f"compl={r['completeness']:4.0%} "
                f"rms/therm={r['rms_over_therm']}")
    ctx['detect_results'] = results
    pd.DataFrame(results).to_csv(
        os.path.join(ctx['coadd_dir'], 'detection_validation.csv'),
        index=False)
    if transfer:
        tdf = pd.DataFrame(transfer)
        tdf.to_csv(os.path.join(ctx['coadd_dir'], 'flux_transfer.csv'),
                   index=False)
        for stokes in ['I', 'V']:
            s = tdf[tdf.stokes == stokes]
            if not len(s):
                continue
            med = ', '.join(
                f"{c.replace('after_', '')} {s[c].median():.3f}"
                for c in ['after_msub', 'after_screen', 'after_ssub']
                if c in s.columns and s[c].notna().any())
            rec_by = ', '.join(
                f"{lvl:.0f}σ {g.recovered.mean():.0%}"
                for lvl, g in s.groupby('level'))
            log(f"  flux transfer {stokes} (median): {med}")
            log(f"  recovery {stokes}: {rec_by}")


# ═══════════════════════════════════════════════════════════════════
# Stage 8b — SEARCH (production candidate search on REAL residuals)
# ═══════════════════════════════════════════════════════════════════

def stage_search(ctx, dry_run=False):
    """Production search: run the I/V identification chains + the validated
    morphology veto on the REAL deep residuals (no injections) and write
    surviving candidates — astrometry + Gaussian photometry + FITS/PNG
    cutouts, channel-routed, repeat-matched — into the survey-wide
    transients/ catalog. Only INCOADD-included epochs are searched."""
    from exopipe.phase3 import phase3_veto as pv
    from exopipe.phase3 import phase3_catalog as pc
    from exopipe.phase3.identify import fit_empirical_beam
    epochs = [e for e in ctx['epochs'] if e.get('msub_paths')]
    incl_dates = set(ctx.get('included_dates', []))
    if incl_dates:
        epochs = [e for e in epochs if e['date'] in incl_dates]
    log(f"\n── Stage 8b SEARCH: production candidate search on "
        f"{len(epochs)} epochs ──")
    if dry_run:
        log("  [plan] I/V identify + validated veto -> "
            "transients/master_catalog + per-candidate cutouts")
        return
    if 'master_i_path' not in ctx:
        log("  No master — skipping")
        return
    master_i, _ = cfp.load_image(ctx['master_i_path'])
    master_v = None
    if ctx.get('master_v_path'):
        master_v, _ = cfp.load_image(ctx['master_v_path'])
    canon = ctx['canonical'][vkey('I', *SCIENCE_I)]
    cd = ctx['cdelt']
    bpix = canon / cd
    em, en, _ep, _nm = fit_empirical_beam(master_i)
    ar_max = max(2.0, 1.5 * em / max(en, 1e-3))
    sigma = {s: (cfp.load_image(ctx[f'sigma_{s}'])[0]
                 if ctx.get(f'sigma_{s}') else None) for s in ['I', 'V']}
    label = ctx['transients_label']
    root = pc.P.STAGING_ROOT
    nwritten = 0
    for ep in epochs:
        for stokes in ['I', 'V']:
            key = ('I_ssub' if (stokes == 'I' and 'I_ssub' in ep['msub_paths'])
                   else stokes)
            if key not in ep['msub_paths']:
                continue
            master = master_i if stokes == 'I' else master_v
            if master is None:
                continue
            resid, _ = cfp.load_image(ep['msub_paths'][key])
            resid = np.where(resid == 0, np.nan, resid)
            hdr = _detection_header(ctx, ep, stokes)
            wcs = WCS(hdr).celestial
            obs_time = hdr.get('DATE-OBS')
            if stokes == 'I':
                passed = run_identification_i(resid, master_i, hdr, bpix,
                                              sigma['I'])
            else:
                passed = run_identification_v(resid, master_i, hdr, bpix,
                                              sigma['V'])
            rclean = np.nan_to_num(resid, nan=0.0)
            kept, vetoed = pv.apply_veto(passed, rclean, master_i, hdr, em,
                                         cd, wcs, obs_time, ar_max=ar_max)
            if kept:
                pc.write_candidates(
                    root, label, kept, rclean, master_i, hdr,
                    ctx['lst'], ep['date'], ctx['band'], stokes, 'deep',
                    em, cd, obs_time)
                nwritten += len(kept)
            log(f"    {ep['date']} {stokes}: {len(passed)} pass -> "
                f"{len(kept)} kept ({len(vetoed)} vetoed)")
    log(f"  wrote {nwritten} candidates -> transients/Run_{label}")
    ctx['n_candidates'] = nwritten


# ═══════════════════════════════════════════════════════════════════
# Stage 9 — SUMMARY
# ═══════════════════════════════════════════════════════════════════

def stage_summary(ctx, dry_run=False):
    log(f"\n── Stage 9 SUMMARY ──")
    if dry_run:
        log("  [plan] per-epoch table: rms, IMGRMS/THERMNOI, FP count, "
            "end-to-end completeness → summary.csv")
        return
    res = ctx.get('detect_results', [])
    if not res:
        path = os.path.join(ctx['coadd_dir'], 'detection_validation.csv')
        if os.path.exists(path):
            res = pd.read_csv(path).to_dict('records')
    if not res:
        log("  No detection results to summarize")
        return
    log(f"  {'date':12s} {'St':2s} {'rms[mJy]':>9s} {'rms/therm':>10s} "
        f"{'FP':>4s} {'compl':>6s}")
    for r in res:
        log(f"  {r['date']:12s} {r['stokes']:2s} {r['rms_mjy']:9.2f} "
            f"{str(r['rms_over_therm']):>10s} {r['final_fp']:4d} "
            f"{r['completeness']:6.0%}")
    pd.DataFrame(res).to_csv(os.path.join(ctx['coadd_dir'],
                                          'summary.csv'), index=False)
    log(f"  summary.csv → {ctx['coadd_dir']}")


# ═══════════════════════════════════════════════════════════════════
# LST utilities
# ═══════════════════════════════════════════════════════════════════

def lst_bin_index(start_time, lst_hour_str, n_bins=6):
    """LST sub-hour bin from LST ARITHMETIC, never from file order.

    bin = floor((LST_apparent(start) - LST_hour) / (1h / n_bins)),
    clipped to [0, n_bins-1] with hour wrap-around.
    """
    t = start_time if isinstance(start_time, Time) else Time(start_time)
    lst = t.sidereal_time('apparent',
                          longitude=cfp.OVRO_LOC.lon).hour
    h0 = float(lst_hour_str.rstrip('h'))
    frac = (lst - h0) % 24.0
    return int(np.clip(np.floor(frac * n_bins), 0, n_bins - 1))


# ═══════════════════════════════════════════════════════════════════
# Self-tests (build validation; no pipeline run)
# ═══════════════════════════════════════════════════════════════════

def selftest_beam(path):
    """Zonal empirical beam on a staged co-add; sanity ~830x623 arcsec
    for the 10h Green master."""
    log(f"=== selftest: zonal empirical beam ===\n  {path}")
    img, hdr = cfp.load_image(path)
    cd = abs(hdr.get('CDELT1', 0.049))
    t0 = time.time()
    fit = fit_empirical_beam_zonal(img, cd)
    log(f"  fit time {time.time() - t0:.1f}s")
    g = fit['global']
    if not g:
        log("  FAIL: no global fit")
        return False
    log(f"  GLOBAL: {g[0] * 3600:.0f}\" x {g[1] * 3600:.0f}\" @ "
        f"{g[2]:.0f} deg  ({g[3]} unresolved srcs)")
    for zi, row in enumerate(fit['zones']):
        cells = []
        for zb in row:
            cells.append('     --     ' if zb is None else
                         f"{zb[0] * 3600:4.0f}x{zb[1] * 3600:4.0f}\" "
                         f"n={zb[3]:2d}")
        log(f"  zone row {zi}: " + ' | '.join(cells))
    ok = 700 < g[0] * 3600 < 1000 and 500 < g[1] * 3600 < 760
    log(f"  sanity (~830x623\"): {'PASS' if ok else 'FAIL'}")
    return ok


def selftest_leakage(path):
    """Leakage screen + corner asymmetry on a staged V residual."""
    log(f"=== selftest: V leakage screen ===\n  {path}")
    data, _ = cfp.load_image(path)
    data = np.where(data == 0, np.nan, data)
    rms0 = float(mad_std(data, ignore_nan=True))
    asym0_raw, asym0 = corner_asymmetry(data)
    screen, coeffs, terms = fit_leakage_screen(data)
    clean = data - screen
    rms1 = float(mad_std(clean, ignore_nan=True))
    asym1_raw, asym1 = corner_asymmetry(clean)
    log(f"  screen coefficients (mJy/beam, normalized coords):")
    for t, c in zip(terms, coeffs):
        log(f"    {t:8s} {c * 1e3:+9.3f}")
    log(f"  rms before/after: {rms0 * 1e3:.2f} / {rms1 * 1e3:.2f} "
        f"mJy/beam")
    log(f"  corner asymmetry: raw {asym0_raw * 1e3:+.2f} → "
        f"{asym1_raw * 1e3:+.2f} mJy/beam;  /rms {asym0:+.2f} → "
        f"{asym1:+.2f}")
    ok = rms1 <= rms0 * 1.001 and abs(asym1) <= abs(asym0) + 0.05
    log(f"  sanity (screen does not worsen rms/asymmetry): "
        f"{'PASS' if ok else 'FAIL'}")

    # Injection check: this staged residual may already be clean, so
    # prove the machinery by injecting a known xy quadrupole (the U->V
    # leakage signature) at ~1.5x rms amplitude and recovering it.
    h, w = data.shape
    yy, xx = np.indices((h, w), dtype=float)
    xn, yn = 2.0 * xx / w - 1.0, 2.0 * yy / h - 1.0
    amp = 1.5 * rms0
    quad = amp * xn * yn
    injected = data + quad
    asym_i_raw, asym_i = corner_asymmetry(injected)
    screen_i, coeffs_i, terms_i = fit_leakage_screen(injected)
    clean_i = injected - screen_i
    rms_i0 = float(mad_std(injected, ignore_nan=True))
    rms_i1 = float(mad_std(clean_i, ignore_nan=True))
    asym_c_raw, asym_c = corner_asymmetry(clean_i)
    k_xy = terms_i.index('x^1y^1')
    rec = coeffs_i[k_xy] - coeffs[k_xy]
    log(f"\n  injection check: xy quadrupole amp {amp * 1e3:+.2f} "
        f"mJy/beam injected")
    log(f"    recovered xy coeff: {rec * 1e3:+.2f} mJy/beam "
        f"({rec / amp:.1%} of injected)")
    log(f"    corner asym/rms: {asym0:+.2f} → {asym_i:+.2f} (injected) "
        f"→ {asym_c:+.2f} (after screen)")
    log(f"    rms: {rms_i0 * 1e3:.2f} → {rms_i1 * 1e3:.2f} mJy/beam")
    ok_inj = (abs(rec / amp - 1.0) < 0.15
              and abs(asym_i) > 5 * abs(asym0 if asym0 else 0.01)
              and abs(asym_c) < 0.3 * abs(asym_i)
              and rms_i1 < rms_i0)
    log(f"    sanity (>=85% recovery, metric detects + clears): "
        f"{'PASS' if ok_inj else 'FAIL'}")
    return ok and ok_inj


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Phase 3 co-add + subtraction + detection pipeline')
    ap.add_argument('--lst', help='LST hour, e.g. 10h')
    ap.add_argument('--band', help='Band: Red/Green/Blue/Full',
                    choices=['Red', 'Green', 'Blue', 'Full'])
    ap.add_argument('--run_label',
                    help='P3 run label to create/resume (default: now)')
    ap.add_argument('--coadd_label',
                    help='Co-add generation label (default: now)')
    ap.add_argument('--workdir_base', default=STAGING_BASE)
    ap.add_argument('--dry_run', action='store_true',
                    help='Discovery + work plan only, no heavy compute')
    ap.add_argument('--native_msub', action='store_true',
                    help='Also build native-beam msub products from '
                         'per-epoch masters (max point-source '
                         'sensitivity); stages 7-8 then use them')
    ap.add_argument('--striation', action='store_true',
                    help='Also produce striation-subtracted I (ssub)')
    ap.add_argument('--transients_label',
                    help='Survey-wide transients Run label (de-siloed catalog '
                         'shared across all LSTs; default: PROD)')
    for st in ['prep', 'beams', 'canon', 'coadd', 'msub', 'sigma',
               'detect', 'search']:
        ap.add_argument(f'--skip_{st}', action='store_true')
    ap.add_argument('--selftest_beam', nargs='?',
                    const='@P3_COADD_ROOT/10h_Green/coadd_final/I_deep_Taper_R0_Green.fits',
                    help='Run the zonal beam-fit self-test and exit')
    ap.add_argument('--selftest_leakage', nargs='?',
                    const='/fast/gh/main/phase3/subtract_lab/'
                          '2024-12-29_V_beammatch_resid.fits',
                    help='Run the leakage-screen self-test and exit')
    args = ap.parse_args()

    if args.selftest_beam:
        _p = args.selftest_beam.replace('@P3_COADD_ROOT', coadd_root())
        sys.exit(0 if selftest_beam(_p) else 1)
    if args.selftest_leakage:
        sys.exit(0 if selftest_leakage(args.selftest_leakage) else 1)
    if not args.lst or not args.band:
        ap.error('--lst and --band are required (or a --selftest_*)')

    t_start = time.time()
    now = time.strftime('%Y%m%d_%H%M%S')
    ctx = {
        'lst': args.lst, 'band': args.band,
        'ref_date': REFERENCE_DATES.get(args.lst),
        'runlabel': args.run_label or now,
        'coadd_label': args.coadd_label or now,
        'transients_label': args.transients_label or 'PROD',
        'workdir': os.path.join(args.workdir_base,
                                f'{args.lst}_{args.band}'),
    }
    if not ctx['ref_date']:
        ap.error(f"No reference date configured for LST {args.lst}")
    ctx['coadd_dir'] = os.path.join(ctx['workdir'],
                                    f"Coadd_{ctx['coadd_label']}")
    ctx['manifest_path'] = os.path.join(
        ctx['workdir'], 'work', f"prep_manifest_{ctx['runlabel']}.json")
    ctx['beams_path'] = os.path.join(
        ctx['workdir'], 'work', f"beams_{ctx['runlabel']}.json")
    # Existing Coadd_* dirs define the co-add generation number used in
    # msub-cNN tokens; multiple generations coexist.
    prior = sorted(glob.glob(os.path.join(ctx['workdir'], 'Coadd_*')))
    ctx['coadd_gen'] = len([p for p in prior
                            if os.path.basename(p) !=
                            f"Coadd_{ctx['coadd_label']}"]) + 1

    log(f"{'═' * 70}")
    log(f"  PHASE 3 PIPELINE  {args.lst} {args.band}"
        f"{'  [DRY RUN]' if args.dry_run else ''}")
    log(f"  reference={ctx['ref_date']}  run=P3_{ctx['runlabel']}  "
        f"coadd=Coadd_{ctx['coadd_label']} (gen c{ctx['coadd_gen']:02d})")
    log(f"  staging: {ctx['workdir']}")
    log(f"{'═' * 70}")

    if not args.dry_run:
        os.makedirs(ctx['coadd_dir'], exist_ok=True)
        os.makedirs(os.path.join(ctx['workdir'], 'work'), exist_ok=True)

    # Stage 1 — discovery (always; cheap)
    log(f"\n── Stage 1 DISCOVERY ──")
    ctx['epochs'] = discover_epochs(args.lst, args.band)
    log(f"  {len(ctx['epochs'])} epochs with v2-repaired deep products:")
    for ep in ctx['epochs']:
        log(f"    {ep['date']}  vars={len(ep['variants']):2d} "
            f"V={'y' if ep['has_v'] else 'n'} "
            f"10min={ep['n_10min_i']:2d} "
            f"anchors={ep['n_anchors'] if ep['n_anchors'] else '--'} "
            f"therm={ep['thermnoi'] * 1e3:.1f}mJy "
            f"{ep['p1run'] or ''}" if ep['thermnoi'] else
            f"    {ep['date']}  vars={len(ep['variants']):2d} (no THERMNOI)")
    if not ctx['epochs']:
        log("  Nothing to do.")
        return

    if not args.skip_prep:
        stage_prep(ctx, args.dry_run)
    elif not args.dry_run:
        if not load_prep_manifest(ctx):
            raise RuntimeError(f"--skip_prep but no manifest at "
                               f"{ctx['manifest_path']}")
        log(f"\n── Stage 2 PREP: skipped (manifest reloaded) ──")

    if not args.skip_beams:
        stage_beams(ctx, args.dry_run)
    elif not args.dry_run and os.path.exists(ctx['beams_path']):
        with open(ctx['beams_path']) as fh:
            byd = json.load(fh)
        for ep in ctx['epochs']:
            ep['ebeams'] = byd.get(ep['date'], {})
        log(f"\n── Stage 3 BEAMS: skipped (reloaded) ──")

    if not args.skip_canon:
        stage_canon(ctx, args.dry_run)
    if not args.skip_coadd:
        stage_coadd(ctx, args.dry_run)
    if not args.skip_msub:
        stage_msub(ctx, args.dry_run, striation=args.striation,
                   native=args.native_msub)
    if not args.skip_sigma:
        stage_sigma(ctx, args.dry_run)
    if not args.skip_detect:
        stage_detect(ctx, args.dry_run)
    if not args.skip_search:
        stage_search(ctx, args.dry_run)
    stage_summary(ctx, args.dry_run)

    log(f"\n{'═' * 70}")
    log(f"  Done in {(time.time() - t_start) / 60:.1f} min")
    log(f"{'═' * 70}")


if __name__ == '__main__':
    main()
