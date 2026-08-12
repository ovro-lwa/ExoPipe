#!/usr/bin/env python3
"""
phase2_science_agg.py  Phase 2 Science Aggregation

Runs after per-subband science extraction. Operates on the Phase 2
working directory (NVMe).

Steps:
  1. Flux check (calibrator validation)
  2. Wideband stacking (with corrected BMAJ/BMIN/DURATION propagation)
  3. 3-color PNGs
  4. Full-band co-add
  5. Wideband target photometry
  6. Wideband solar system photometry

Transient search and detection gathering are handled by the controller
to ensure proper ordering (transients run last, after all products exist).

Key fixes vs old post_process_science.py (from Feb 2026 change spec):
  - §1.1: BMAJ/BMIN propagated as inverse-variance weighted mean
  - §1.2: DURATION propagated from constituent headers
  - §1.3: Strict enforcement of dewarped images (no silent fallback)
  - §1.5: Full-band (18-85 MHz) co-add added
  - §1.6: Structured filename parsing (no substring matching)
  - §1.7/1.8: Separate subband/wideband/combined detection CSVs
"""
import os
import glob
import logging
import json
import shutil
import csv
import smtplib
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import mad_std
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

from exopipe.common import phase2_config as cfg


def _qa_prefix(run_dir):
    """Extract date_LST prefix from run directory for unique QA filenames."""
    import re
    path = run_dir.rstrip(os.sep)
    m = re.search(r'/(\d+h)/(\d{4}-\d{2}-\d{2})/', path + '/')
    if m:
        return f"{m.group(2)}_{m.group(1)}_"
    return ""

try:
    from exopipe.phase3 import flux_check_cutout
except ImportError:
    flux_check_cutout = None

try:
    from exopipe.phase3 import solar_system_cutout
except ImportError:
    solar_system_cutout = None

try:
    from exopipe.phase3 import cutout
except ImportError:
    cutout = None

logger = logging.getLogger('phase2')

EMAIL_REPORT_LINES = []
ATTACHMENT_FILES = []


# =====================================================================
# HELPERS
# =====================================================================

def get_inner_rms(fits_path):
    """Calculate RMS from the inner 1024×1024 pixels using MAD.

    Consistent with transient_search.calculate_uniform_rms() and
    phase2_controller._generate_thermal_noise_csv().
    """
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data.squeeze()
            h, w = data.shape
            half = min(512, h // 4, w // 4)
            ch, cw = h // 2, w // 2
            region = data[ch - half:ch + half, cw - half:cw + half]
            return float(mad_std(region, ignore_nan=True))
    except Exception as e:
        logger.warning(f"RMS failed for {os.path.basename(fits_path)}: {e}")
        return np.nan


# =====================================================================
# WIDEBAND STACKING (with P0 fixes)
# =====================================================================

def _weighted_center_freq_hz(freqs_mhz, weights):
    """Inverse-variance weighted center frequency in Hz.

    Uses exact subband center frequencies from config when available,
    falling back to approximate MHz labels.
    """
    total_w = sum(weights)
    if total_w <= 0:
        return None
    center_sum = 0.0
    for f_mhz, w in zip(freqs_mhz, weights):
        sb_label = f"{int(f_mhz)}MHz"
        sb_freq = cfg.SUBBAND_FREQ_HZ.get(sb_label)
        if sb_freq:
            center_sum += sb_freq['center'] * w
        else:
            center_sum += f_mhz * 1e6 * w
    return center_sum / total_w


STOKES_CODE = {'I': 1, 'Q': 2, 'U': 3, 'V': 4}


def _stack_images(img_list, weights, out_path, history_str,
                  center_freq_hz=None, freq_min_hz=None, freq_max_hz=None,
                  regrid_cache=None, stokes=None):
    """Inverse-variance weighted co-add of FITS images.

    FIX §1.1: Propagates weighted BMAJ/BMIN/BPA.
    FIX §1.2: Propagates DURATION.

    When input images have different pixel grids (variable resolution mode),
    all images are reprojected onto the highest-resolution grid using
    reproject_interp before co-adding.

    Parameters
    ----------
    regrid_cache : dict, optional
        Shared cache for regridded data: {filepath: (data, ref_shape)}.
        Avoids redundant regridding when the same subbands are stacked
        into multiple wideband products with the same reference grid.

    Returns True on success.
    """
    if not img_list:
        return False
    try:
        # -- First pass: load all images and headers --
        # Divergent-input guard (2026-07-02): a peel-diverged subband image
        # (|pixels| up to 1e30; seen in 13h/14h I 69-82MHz) enters the stack at
        # FULL weight because stacking weights come from the (clean) V sibling's
        # rms — one such input corrupts the whole wideband product. Exclude it.
        all_data = []
        all_hdrs = []
        kept_weights = []
        for img, w in zip(img_list, weights):
            with fits.open(img) as hdul:
                d = hdul[0].data.squeeze()
                h = hdul[0].header.copy()
            mx = float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else np.nan
            if not np.isfinite(mx) or mx > 1.0e6:
                logger.warning(f"  DIVERGENT input excluded from stack "
                               f"(max={mx:.2e}): {os.path.basename(img)}")
                continue
            all_data.append(d)
            all_hdrs.append(h)
            kept_weights.append(w)
        if not all_data:
            logger.warning("  all stack inputs divergent — skipping product")
            return False
        weights = kept_weights

        # Identify reference grid: largest image (highest resolution)
        ref_idx = max(range(len(all_data)), key=lambda i: all_data[i].size)
        ref_header = all_hdrs[ref_idx].copy()
        ref_shape = all_data[ref_idx].shape

        # Check if regridding needed
        needs_regrid = any(d.shape != ref_shape for d in all_data)
        if needs_regrid:
            try:
                from reproject import reproject_interp
            except ImportError:
                logger.error("  reproject package required for variable-resolution "
                             "stacking. Install with: pip install reproject")
                return False

            ref_wcs = WCS(ref_header).celestial
            n_regridded = 0
            for i in range(len(all_data)):
                if all_data[i].shape != ref_shape:
                    fpath = img_list[i]
                    cache_key = (fpath, ref_shape)

                    # Check cache first
                    if regrid_cache is not None and cache_key in regrid_cache:
                        all_data[i] = regrid_cache[cache_key]
                    else:
                        src_wcs = WCS(all_hdrs[i]).celestial
                        reprojected, footprint = reproject_interp(
                            (all_data[i], src_wcs), ref_wcs,
                            shape_out=ref_shape)
                        reprojected[footprint < 0.5] = 0.0
                        all_data[i] = reprojected
                        if regrid_cache is not None:
                            regrid_cache[cache_key] = reprojected
                    n_regridded += 1
            logger.info(f"  Regridded {n_regridded}/{len(all_data)} images "
                        f"to {ref_shape[0]}×{ref_shape[1]} reference grid")

        # -- Weighted co-add --
        data_sum = np.zeros(ref_shape, dtype=np.float64)
        weight_sum = 0.0

        # Beam tracking (§1.1)
        bmaj_list = []
        bmin_list = []
        bpa_list = []
        weight_list_beam = []

        # Time tracking (§1.2): constituents are simultaneous subbands
        # of the same time window, so the stack spans [min start,
        # max end] and INTTIME is the time-domain integration of the
        # most complete constituent — NOT a sum over frequency.
        from astropy.time import Time
        starts, ends, inttimes = [], [], []

        for data, hdr, w in zip(all_data, all_hdrs, weights):
            data_sum += (data * w)
            weight_sum += w

            # Collect beam info (§1.1)
            bmaj = hdr.get('BMAJ', 0)
            bmin = hdr.get('BMIN', 0)
            bpa = hdr.get('BPA', 0)
            if bmaj > 0:
                bmaj_list.append(bmaj)
                bmin_list.append(bmin)
                bpa_list.append(bpa)
                weight_list_beam.append(w)

            # Collect times (§1.2)
            try:
                if hdr.get('DATE-OBS'):
                    starts.append(Time(hdr['DATE-OBS']))
                if hdr.get('DATE-END'):
                    ends.append(Time(hdr['DATE-END']))
            except Exception:
                pass
            it = hdr.get('INTTIME') or hdr.get('DURATION')
            if it and it > 0:
                inttimes.append(float(it))

        if weight_sum <= 0:
            return False

        final_data = data_sum / weight_sum

        # Propagate weighted beam (§1.1)
        if bmaj_list and weight_list_beam:
            w_arr = np.array(weight_list_beam)
            w_norm = w_arr / w_arr.sum()
            ref_header['BMAJ'] = float(np.sum(w_norm * np.array(bmaj_list)))
            ref_header['BMIN'] = float(np.sum(w_norm * np.array(bmin_list)))
            ref_header['BPA'] = float(np.sum(w_norm * np.array(bpa_list)))

        # Propagate aggregated times (§1.2)
        if starts and ends:
            t0 = min(starts, key=lambda t: t.mjd)
            t1 = max(ends, key=lambda t: t.mjd)
            duration = (t1 - t0).sec
            ref_header['DATE-OBS'] = (t0.isot,
                                      'Start of first integration (UTC)')
            ref_header['MJD-OBS'] = (t0.mjd, 'Start MJD (UTC)')
            ref_header['DATE-END'] = (t1.isot,
                                      'End of last integration (UTC)')
            ref_header['DURATION'] = (float(duration),
                                      '[s] Wall-clock span '
                                      '(DATE-END - DATE-OBS)')
            ref_header['INTTIME'] = (float(max(inttimes)) if inttimes
                                     else float(duration),
                                     '[s] Summed integration time included')
            ref_header['MJD-MID'] = ((t0.mjd + t1.mjd) / 2.0,
                                     'Midpoint MJD (UTC)')
            ref_header['TIMESYS'] = ('UTC', 'Time system')

        # Per-subband WSClean bookkeeping keywords describe only the
        # reference constituent — drop them from the stack (the wsclean
        # command line stays in HISTORY for imaging-parameter
        # provenance).
        for key in [k for k in list(ref_header.keys())
                    if str(k).startswith('WSC')]:
            del ref_header[key]

        ref_header['BTYPE'] = 'Intensity'
        ref_header['GITTAG'] = (cfg.PIPELINE_GITTAG,
                                'Pipeline code version')
        ref_header['HISTORY'] = history_str

        # Strip stale axes 3+4, then rebuild correct FREQ and STOKES WCS
        for ax in [3, 4]:
            for prefix in ['NAXIS', 'CTYPE', 'CRPIX', 'CRVAL', 'CDELT',
                            'CUNIT', 'CROTA']:
                key = f'{prefix}{ax}'
                if key in ref_header:
                    del ref_header[key]
        ref_header['NAXIS'] = 2

        # Frequency scalar metadata
        bw_hz = None
        if center_freq_hz is not None:
            ref_header['RESTFRQ'] = (center_freq_hz,
                                      '[Hz] Weighted center frequency')
        if freq_min_hz is not None:
            ref_header['FREQ_MIN'] = (freq_min_hz,
                                       '[Hz] Lower edge of stacked band')
        if freq_max_hz is not None:
            ref_header['FREQ_MAX'] = (freq_max_hz,
                                       '[Hz] Upper edge of stacked band')
        if freq_min_hz is not None and freq_max_hz is not None:
            bw_hz = freq_max_hz - freq_min_hz
            ref_header['BANDWDTH'] = (bw_hz, '[Hz] Total bandwidth')

        # FREQ WCS axis (axis 3) — BDSF reads this to determine frequency
        if center_freq_hz is not None:
            ref_header['CTYPE3'] = ('FREQ', 'Frequency axis')
            ref_header['CRVAL3'] = (float(center_freq_hz),
                                     '[Hz] Band center frequency')
            ref_header['CDELT3'] = (float(bw_hz or 1.0),
                                     '[Hz] Total bandwidth')
            ref_header['CRPIX3'] = (1.0, 'Reference pixel')
            ref_header['CUNIT3'] = ('Hz', 'Frequency unit')

        # Stokes WCS axis (axis 4)
        stokes_code = STOKES_CODE.get(stokes)
        if stokes_code is not None:
            ref_header['CTYPE4'] = ('STOKES', 'Stokes axis')
            ref_header['CRVAL4'] = (float(stokes_code),
                                     f'Stokes {stokes}')
            ref_header['CDELT4'] = (1.0, 'Stokes increment')
            ref_header['CRPIX4'] = (1.0, 'Reference pixel')

        # Provenance: constituent paths relative to the run directory
        # (subband/pol/category/filename)
        ref_header['NSTACK'] = (len(img_list), 'Number of images in stack')
        for fpath in img_list:
            rel = '/'.join(os.path.normpath(fpath).split(os.sep)[-4:])
            ref_header['HISTORY'] = f'  constituent: {rel}'

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fits.writeto(out_path, final_data.astype(np.float32),
                     ref_header, overwrite=True)
        return True

    except Exception as e:
        logger.error(f"Stack failed for {out_path}: {e}")
        traceback.print_exc()
        return False


def _find_subband_image(sb_dir, pol, category, taper_kw, suffix_kw,
                        require_dewarped=True, include_pbcorr=False):
    """Find images matching criteria in a subband directory.

    FIX §1.3: require_dewarped=True enforces dewarped images.
    Only matches *-image* files (excludes dirty, model, psf, residual, mask).

    Parameters
    ----------
    include_pbcorr : bool
        If True, match PB-corrected images (*-image.pbcorr*).
        If False (default), exclude them.

    Returns list of matching paths.
    """
    base = os.path.join(sb_dir, pol, category)
    if not os.path.isdir(base):
        return []

    if require_dewarped:
        if include_pbcorr:
            candidates = glob.glob(os.path.join(base, "*-image*pbcorr*_dewarped.fits"))
        else:
            candidates = glob.glob(os.path.join(base, "*-image*_dewarped.fits"))
            candidates = [f for f in candidates
                          if "pbcorr" not in os.path.basename(f)]
        candidates = [f for f in candidates
                      if "_dewarped_dewarped" not in os.path.basename(f)]
    else:
        if include_pbcorr:
            candidates = glob.glob(os.path.join(base, "*-image*pbcorr*.fits"))
        else:
            candidates = glob.glob(os.path.join(base, "*-image*.fits"))
            candidates = [f for f in candidates
                          if "pbcorr" not in os.path.basename(f)]
        candidates = [f for f in candidates
                      if "dewarped" not in os.path.basename(f)]

    # Filter out pybdsf logs
    candidates = [f for f in candidates
                  if ".pybdsf" not in os.path.basename(f)]

    # Taper filter
    if taper_kw == 'NoTaper':
        candidates = [f for f in candidates
                      if 'NoTaper' in os.path.basename(f)]
    else:
        candidates = [f for f in candidates
                      if 'NoTaper' not in os.path.basename(f)]

    # Suffix filter (e.g. "Robust-0-" must not match "Robust-0.75")
    if suffix_kw:
        filtered = [f for f in candidates
                    if suffix_kw in os.path.basename(f)]
        if filtered:
            candidates = filtered

    return sorted(candidates)


# =====================================================================
# MAIN AGGREGATION
# =====================================================================

def run_science_aggregation(run_dir, target_files, catalog_path, logger_in):
    """Run the full science aggregation pipeline.

    Parameters
    ----------
    run_dir : str
        Phase 2 working directory (e.g. /fast/gh/main/phase2/10h/2024-12-22/).
    target_files : list of str or None
        Paths to target CSV files.
    catalog_path : str or None
        Path to source catalog for transient cross-matching.
    logger_in : logging.Logger
    """
    global logger
    logger = logger_in

    logger.info("\n" + "=" * 60)
    logger.info("SCIENCE AGGREGATION")
    logger.info("=" * 60)

    # -- Step 1: Flux Check --
    run_flux_check(run_dir)

    # -- Step 2: Wideband Stacking --
    run_wideband_stacking(run_dir)

    # -- Step 3: 3-color PNGs --
    # (generated inside run_wideband_stacking)

    # -- Step 4: Wideband Solar System Photometry --
    if solar_system_cutout is not None:
        try:
            solar_system_cutout.process_wideband_solar_system(
                run_dir, logger=logger)
        except Exception as e:
            logger.error(f"Wideband solar system photometry failed: {e}")
    else:
        logger.warning("solar_system_cutout not available")

    # -- Step 5: Wideband Target Photometry --
    run_wideband_target_photometry(run_dir, target_files)

    # -- Step 6: Archive wideband products --
    archive_wideband_products(run_dir)

    logger.info("\nScience aggregation complete")


# =====================================================================
# STEP 1: FLUX CHECK
# =====================================================================

def run_flux_check(run_dir):
    logger.info("\n--- Flux Scale Check ---")
    if flux_check_cutout is None:
        logger.warning("flux_check_cutout module not available — skipping")
        return

    # Diagnose: show function signature (does it expect logger or other args?)
    import inspect
    try:
        sig = inspect.signature(flux_check_cutout.run_flux_check)
        logger.info(f"  flux_check_cutout.run_flux_check signature: {sig}")
    except Exception:
        pass

    logger.info(f"  run_dir: {run_dir}")
    try:
        result = flux_check_cutout.run_flux_check(run_dir)
        logger.info(f"  Flux check returned: {result}")
    except TypeError as e:
        # If it needs more arguments, try passing logger
        logger.warning(f"  TypeError with 1 arg: {e}")
        logger.info("  Retrying with logger argument...")
        try:
            result = flux_check_cutout.run_flux_check(run_dir, logger=logger)
            logger.info(f"  Flux check returned (with logger): {result}")
        except Exception as e2:
            logger.error(f"  Retry also failed: {e2}")
    except Exception as e:
        logger.error(f"  Flux check failed: {e}")
        import traceback
        traceback.print_exc()

    # Verify expected outputs
    qa_dir = os.path.join(run_dir, "QA")
    fc_pngs = glob.glob(os.path.join(qa_dir, "*Flux_Check_*.png"))
    fc_csvs = glob.glob(os.path.join(qa_dir, "*flux_check_hybrid.csv"))
    if fc_pngs:
        logger.info(f"  Outputs: {len(fc_pngs)} PNGs, "
                    f"CSV={'yes' if fc_csvs else 'no'}")
    else:
        logger.warning(f"  No Flux_Check_*.png found in {qa_dir}")


# =====================================================================
# STEP 2: WIDEBAND STACKING
# =====================================================================

def run_wideband_stacking(run_dir, skip_bands=()):
    # skip_bands (Gregg ruling 2026-07-09, per-band refinement): a wideband
    # product is only formed when ALL of its constituent subbands exist —
    # incomplete bands (and Full, if any of R/G/B is incomplete) are skipped;
    # complete bands still form. Subbands always process.
    logger.info("\n--- Step 2: Wideband Co-addition ---")
    if skip_bands:
        logger.warning(f"Wideband bands SKIPPED (incomplete subband set): "
                       f"{sorted(skip_bands)}")

    # Flux check runs first (before noise analysis / stacking).
    # Called here rather than from the controller to match the original
    # working structure where it lived inside run_wideband_formation.
    run_flux_check(run_dir)

    # Extract date and LST from directory path for filenames
    # Path: .../phase2/{lst}/{date}/{run_label}/
    path_parts = os.path.normpath(run_dir).split(os.sep)
    date_str = path_parts[-2] if len(path_parts) >= 3 else 'unknown'
    lst_label = path_parts[-3] if len(path_parts) >= 4 else 'unknown'
    date_compact = date_str.replace('-', '')  # 20241222
    file_tag = f"{date_compact}_LST{lst_label}"  # 20241222_LST10h

    # -- 2a. Noise Analysis --
    sb_dirs = sorted(glob.glob(os.path.join(run_dir, "*MHz")))
    if not sb_dirs:
        logger.warning("No subband directories found")
        return

    import re

    noise_data = []
    deep_rms = {}      # {subband: rms} for THERMNOI stamping
    tenmin_rms = {}    # {(subband, timestamp): rms}
    for sb_dir in sb_dirs:
        try:
            freq = float(os.path.basename(sb_dir).replace('MHz', ''))
        except ValueError:
            continue

        # Prefer dewarped V deep for noise measurement
        v_imgs = _find_subband_image(sb_dir, 'V', 'deep', 'Taper', '',
                                      require_dewarped=True)
        if not v_imgs:
            v_imgs = _find_subband_image(sb_dir, 'V', 'deep', 'Taper', '',
                                          require_dewarped=False)
        if not v_imgs:
            # Debug: list what's actually in V/deep/
            v_deep_dir = os.path.join(sb_dir, 'V', 'deep')
            if os.path.isdir(v_deep_dir):
                all_files = os.listdir(v_deep_dir)
                logger.warning(f"  {freq:.0f}MHz: V/deep/ has {len(all_files)} files "
                               f"but none match filter. Files: "
                               f"{all_files[:5]}{'...' if len(all_files) > 5 else ''}")
            else:
                logger.warning(f"  {freq:.0f}MHz: V/deep/ directory does not exist")
            continue

        rms = get_inner_rms(v_imgs[0])
        if np.isfinite(rms) and rms > 0:
            noise_data.append({
                'freq': freq, 'rms': rms, 'weight': 1.0 / (rms ** 2),
                'file': os.path.basename(v_imgs[0])
            })
            deep_rms[os.path.basename(sb_dir)] = rms
            logger.info(f"  {freq:.0f}MHz: RMS = {rms*1000:.2f} mJy/beam "
                        f"({os.path.basename(v_imgs[0])})")
        else:
            logger.warning(f"  {freq:.0f}MHz: invalid RMS = {rms} "
                           f"from {os.path.basename(v_imgs[0])}")

    if not noise_data:
        logger.warning("No valid noise measurements  aborting stacking")
        return

    df_noise = pd.DataFrame(noise_data)
    qa_dir = os.path.join(run_dir, "QA")
    os.makedirs(qa_dir, exist_ok=True)

    # Also measure 10-min interval noise (for diagnostics, not stacking weights)
    noise_10min = []
    for sb_dir in sb_dirs:
        try:
            freq = float(os.path.basename(sb_dir).replace('MHz', ''))
        except ValueError:
            continue
        v_10min_imgs = _find_subband_image(sb_dir, 'V', '10min', 'Taper', '',
                                            require_dewarped=True)
        if not v_10min_imgs:
            v_10min_imgs = _find_subband_image(sb_dir, 'V', '10min', 'Taper', '',
                                                require_dewarped=False)
        for vf in (v_10min_imgs or []):
            rms = get_inner_rms(vf)
            if np.isfinite(rms) and rms > 0:
                noise_10min.append({
                    'freq': freq, 'rms': rms, 'weight': 1.0 / (rms ** 2),
                    'category': '10min', 'file': os.path.basename(vf)
                })
                m = re.search(r'\d{8}_\d{6}', os.path.basename(vf))
                if m:
                    tenmin_rms[(os.path.basename(sb_dir), m.group(0))] = rms

    # Save combined CSV (deep + 10min)
    df_deep = df_noise.copy()
    df_deep['category'] = 'deep'
    if noise_10min:
        df_all = pd.concat([df_deep, pd.DataFrame(noise_10min)],
                           ignore_index=True)
        logger.info(f"  10-min noise: {len(noise_10min)} measurements "
                    f"across {len(set(d['freq'] for d in noise_10min))} "
                    f"subbands")
    else:
        df_all = df_deep

    noise_csv = os.path.join(qa_dir, f"{_qa_prefix(run_dir)}thermal_noise.csv")
    df_all.to_csv(noise_csv, index=False)

    # Thermal noise plot
    _plot_thermal_noise(df_noise, qa_dir, run_dir)

    # Dewarp cross-frequency diagnostic
    _generate_dewarp_summary(run_dir, qa_dir)

    # -- 2b. Stack per band --
    # Transient search images: Tapered, Dewarped, non-PB-corrected
    transient_targets = [
        ('I', 'deep',  'Taper', 'Robust-0-',  'Robust-0'),
        ('I', '10min', 'Taper', 'Robust-0-',  'Robust-0'),
        ('V', 'deep',  'Taper', 'Robust-0-',  'Robust-0'),
        ('V', '10min', 'Taper', 'Robust-0-',  'Robust-0'),
    ]

    # Science images: dewarped, non-PB-corrected
    science_targets = [
        ('I', 'deep', 'NoTaper', 'Robust-0-',    'Robust-0'),
        ('I', 'deep', 'NoTaper', 'Robust-0.75-',  'Robust-0.75'),
        ('I', 'deep', 'Taper',   'Robust-0.75-',  'Robust-0.75'),
    ]

    # PB-corrected science images: dewarped, for photometry/cutouts
    pbcorr_targets = [
        ('I', 'deep',  'Taper', 'Robust-0.75-', 'Robust-0.75'),
        ('I', 'deep',  'Taper', 'Robust-0-',    'Robust-0'),
        ('V', 'deep',  'Taper', 'Robust-0-',    'Robust-0'),
        ('I', '10min', 'Taper', 'Robust-0-',    'Robust-0'),
        ('V', '10min', 'Taper', 'Robust-0-',    'Robust-0'),
    ]

    # Shared regridding cache: avoids redundant reprojection when the
    # same subbands are stacked into multiple products at the same grid
    regrid_cache = {}

    # Compute per-band total weights (for Full-band combination later)
    band_total_weights = {}
    band_nstack = {}
    for b_name, (f_min, f_max) in cfg.WIDEBAND_BANDS.items():
        if b_name in skip_bands:
            continue
        if b_name == 'Full':
            continue
        subset = df_noise[(df_noise['freq'] >= f_min) & (df_noise['freq'] < f_max)]
        if not subset.empty:
            band_total_weights[b_name] = float(subset['weight'].sum())
            band_nstack[b_name] = len(subset)

    for b_name, (f_min, f_max) in cfg.WIDEBAND_BANDS.items():
        if b_name in skip_bands:
            continue
        if b_name == 'Full':
            continue  # Full-band created from band images below

        subset = df_noise[(df_noise['freq'] >= f_min) & (df_noise['freq'] < f_max)]
        if subset.empty:
            continue

        logger.info(f"\n--- Stacking {b_name} ({f_min}-{f_max} MHz, "
                    f"{len(subset)} subbands) ---")

        band_center_hz = _weighted_center_freq_hz(
            subset['freq'].tolist(), subset['weight'].tolist())

        # Stack transient images (non-PBcorr)
        for pol, cat, taper_kw, suffix_kw, robust_label in transient_targets:
            _stack_for_band(run_dir, b_name, subset,
                            pol, cat, taper_kw, suffix_kw, robust_label,
                            band_center_hz, file_tag,
                            require_dewarped=True,
                            regrid_cache=regrid_cache)

        # Stack science images (non-PBcorr)
        for pol, cat, taper_kw, suffix_kw, robust_label in science_targets:
            _stack_for_band(run_dir, b_name, subset,
                            pol, cat, taper_kw, suffix_kw, robust_label,
                            band_center_hz, file_tag,
                            require_dewarped=True,
                            regrid_cache=regrid_cache)

        # Stack PB-corrected science images
        for pol, cat, taper_kw, suffix_kw, robust_label in pbcorr_targets:
            _stack_for_band(run_dir, b_name, subset,
                            pol, cat, taper_kw, suffix_kw, robust_label,
                            band_center_hz, file_tag,
                            require_dewarped=True, include_pbcorr=True,
                            regrid_cache=regrid_cache)

    # -- 2c. Flush accumulated 10min interval stacks --
    _flush_10min_stacks(run_dir, file_tag, regrid_cache=regrid_cache)

    # Log cache stats
    if regrid_cache:
        n_cached = len(regrid_cache)
        mem_mb = sum(v.nbytes for v in regrid_cache.values()) / 1e6
        logger.info(f"  Regrid cache: {n_cached} entries, {mem_mb:.0f} MB")
    regrid_cache.clear()  # free memory

    # -- 2d. Full-band products from Red+Green+Blue --
    _create_full_band_products(run_dir, df_noise, band_total_weights,
                               band_nstack, file_tag)

    # -- 2e. THERMNOI header stamping --
    _stamp_subband_thermnoi(run_dir, deep_rms, tenmin_rms)
    _stamp_wideband_thermnoi(run_dir, skip_bands=skip_bands)

    # -- 2f. 3-color PNGs --
    _generate_3color_pngs(run_dir, file_tag)


def _write_thermnoi(path, rms, src):
    """Write the THERMNOI keyword into one FITS header."""
    try:
        with fits.open(path, mode='update') as hdul:
            # Keep value+comment within the 80-char card limit
            hdul[0].header['THERMNOI'] = (float(round(rms, 9)),
                                          f'[Jy/beam] Noise ({src})')
            hdul.flush()
        return 1
    except Exception as e:
        logger.warning(f"  THERMNOI failed for {os.path.basename(path)}: {e}")
        return 0


def _stamp_subband_thermnoi(run_dir, deep_rms, tenmin_rms):
    """Stamp THERMNOI on every subband image product.

    The value is the inner-region rms of the matching dewarped Stokes V
    Taper image of the same subband and time window — V is thermal-
    noise-limited, so it serves as the thermal reference for the I
    products of the same data.
    """
    import re

    n = 0
    for sb_dir in sorted(glob.glob(os.path.join(run_dir, "*MHz"))):
        sb = os.path.basename(sb_dir)
        for pol in ['I', 'V']:
            for cat in ['deep', '10min']:
                d = os.path.join(sb_dir, pol, cat)
                if not os.path.isdir(d):
                    continue
                for f in glob.glob(os.path.join(d, "*-image*.fits")):
                    if cat == 'deep':
                        rms = deep_rms.get(sb)
                    else:
                        m = re.search(r'\d{8}_\d{6}', os.path.basename(f))
                        rms = tenmin_rms.get((sb, m.group(0))) if m else None
                    if rms:
                        n += _write_thermnoi(f, rms,
                                             'matching V Taper inner rms')
    if n:
        logger.info(f"  THERMNOI stamped on {n} subband files")
    return n


def _stamp_wideband_thermnoi(run_dir, skip_bands=()):
    """Measure and stamp THERMNOI on wideband products.

    V products are measured directly (inner rms of the stacked image);
    I products take the value from the matching V product of the same
    band/category/interval (V exists only as Taper Robust-0).
    """
    import re

    n = 0
    for band in ['Red', 'Green', 'Blue', 'Full']:
        if band in skip_bands:
            continue
        for cat in ['deep', '10min']:
            v_rms = {}  # interval key -> (preference, rms)
            v_dir = os.path.join(run_dir, band, 'V', cat)
            if os.path.isdir(v_dir):
                for f in sorted(glob.glob(os.path.join(v_dir, "*.fits"))):
                    rms = get_inner_rms(f)
                    if not (np.isfinite(rms) and rms > 0):
                        continue
                    n += _write_thermnoi(f, rms, 'inner rms of this V image')
                    bn = os.path.basename(f)
                    m = re.search(r'_t(\d{4})', bn)
                    tkey = m.group(1) if m else 'deep'
                    # Prefer the non-pbcorr dewarped V as the reference
                    pref = ('pbcorr' not in bn, '_dewarped' in bn)
                    if tkey not in v_rms or pref > v_rms[tkey][0]:
                        v_rms[tkey] = (pref, rms)
            i_dir = os.path.join(run_dir, band, 'I', cat)
            if os.path.isdir(i_dir) and v_rms:
                for f in sorted(glob.glob(os.path.join(i_dir, "*.fits"))):
                    m = re.search(r'_t(\d{4})', os.path.basename(f))
                    tkey = m.group(1) if m else 'deep'
                    if tkey in v_rms:
                        n += _write_thermnoi(f, v_rms[tkey][1],
                                             'matching V product inner rms')
    if n:
        logger.info(f"  THERMNOI stamped on {n} wideband files")
    return n


def _stack_for_band(run_dir, band_name, noise_subset,
                     pol, category, taper_kw, suffix_kw, robust_label,
                     center_freq_hz, file_tag, require_dewarped=True,
                     include_pbcorr=False, regrid_cache=None):
    """Stack all subbands in a band for a specific image type.

    Output goes to {run_dir}/{band_name}/{pol}/{category}/ mirroring
    the subband directory structure.
    """
    # Compute exact freq range from the actual subbands in this stack,
    # using the correlator-derived boundaries. This handles missing subbands
    # correctly (the range reflects what's actually included).
    freq_min_hz = None
    freq_max_hz = None
    for _, row in noise_subset.iterrows():
        sb_label = f"{int(row['freq'])}MHz"
        sb_freq = cfg.SUBBAND_FREQ_HZ.get(sb_label)
        if sb_freq:
            if freq_min_hz is None or sb_freq['lo'] < freq_min_hz:
                freq_min_hz = sb_freq['lo']
            if freq_max_hz is None or sb_freq['hi'] > freq_max_hz:
                freq_max_hz = sb_freq['hi']
    # Fallback if config lookup fails
    if freq_min_hz is None:
        freq_min_hz = noise_subset['freq'].min() * 1e6
    if freq_max_hz is None:
        freq_max_hz = noise_subset['freq'].max() * 1e6
    img_list = []
    weights = []
    all_dewarped = True  # Track whether all inputs are dewarped

    for _, row in noise_subset.iterrows():
        sb_dir = os.path.join(run_dir, f"{int(row['freq'])}MHz")
        candidates = _find_subband_image(
            sb_dir, pol, category, taper_kw, suffix_kw,
            require_dewarped=require_dewarped,
            include_pbcorr=include_pbcorr
        )

        if not candidates:
            if require_dewarped:
                fallback = _find_subband_image(
                    sb_dir, pol, category, taper_kw, suffix_kw,
                    require_dewarped=False,
                    include_pbcorr=include_pbcorr
                )
                if fallback:
                    logger.warning(f"  {int(row['freq'])}MHz/{pol}/{category}: "
                                   f"no dewarped images  using raw "
                                   f"(astrometry may be inconsistent!)")
                    candidates = fallback
                    all_dewarped = False
            if not candidates:
                continue
        else:
            # Check if the selected candidate is actually dewarped
            if 'dewarped' not in os.path.basename(candidates[0]):
                all_dewarped = False

        # For 10min intervals, group by interval tag (t0001..t0006)
        if category == '10min':
            is_dewarped = ('dewarped' in os.path.basename(candidates[0])
                           if candidates else False)
            _stack_10min_intervals(candidates, row['weight'],
                                   band_name, pol, taper_kw, robust_label,
                                   center_freq_hz, file_tag, include_pbcorr,
                                   is_dewarped,
                                   img_list, weights)
        else:
            # Deep: expect 1 file per subband
            img_list.append(candidates[0])
            weights.append(row['weight'])

    if category == '10min':
        return

    if not img_list:
        pbcorr_str = " (pbcorr)" if include_pbcorr else ""
        tl = "Taper" if taper_kw != 'NoTaper' else "NoTaper"
        logger.warning(f"  No images found for {band_name} {pol} {category} "
                       f"{tl} {suffix_kw}{pbcorr_str}  skipping")
        return

    taper_label = "Taper" if taper_kw != 'NoTaper' else "NoTaper"
    pbcorr_tag = "_pbcorr" if include_pbcorr else ""
    dewarped_tag = "_dewarped" if all_dewarped else ""
    out_name = (f"{band_name}_{pol}_{category}_{taper_label}_{robust_label}"
                f"{pbcorr_tag}{dewarped_tag}_{file_tag}.fits")
    out_dir = os.path.join(run_dir, band_name, pol, category)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    history = (f"{band_name} {pol} {category} {taper_label} {robust_label}: "
               f"inverse-variance stack of {len(img_list)} subbands"
               f"{' (all dewarped)' if all_dewarped else ' (WARNING: some not dewarped)'}")

    if _stack_images(img_list, weights, out_path, history, center_freq_hz,
                     freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz,
                     regrid_cache=regrid_cache, stokes=pol):
        logger.info(f"  Stacked: {out_name} ({len(img_list)} images)")


def _stack_10min_intervals(candidates, weight, band_name, pol,
                            taper_kw, robust_label, center_freq_hz,
                            file_tag, include_pbcorr, is_dewarped,
                            _unused_img_list, _unused_weights):
    """Handle 10-minute interval stacking  group by interval tag.

    For 10min images, each subband produces 6 files with different
    timestamps. We group by temporal interval so that the same 10-minute
    window from each subband is co-added together.

    Since timestamps may differ slightly across subbands, we sort by
    time and assign interval indices (t0001..t0006).
    """
    _10min_accumulator.setdefault(
        (band_name, pol, taper_kw, robust_label, center_freq_hz,
         file_tag, include_pbcorr), []
    ).append((candidates, weight, is_dewarped))


# Module-level accumulator for 10min intervals
_10min_accumulator = {}


def _flush_10min_stacks(run_dir, file_tag, regrid_cache=None):
    """Stack accumulated 10min intervals across subbands.

    Called after all subbands have been processed for all bands.
    Groups files by interval index (sorted by timestamp within each subband).
    Output goes to {run_dir}/{band}/{pol}/10min/.
    """
    global _10min_accumulator

    for key, subband_entries in _10min_accumulator.items():
        band_name, pol, taper_kw, robust_label, center_freq_hz, _ft, incl_pb = key
        taper_label = "Taper" if taper_kw != 'NoTaper' else "NoTaper"
        pbcorr_tag = "_pbcorr" if incl_pb else ""

        # Compute freq range from actual subbands in the stack
        freq_min_hz = None
        freq_max_hz = None
        for candidates, _, _ in subband_entries:
            if candidates:
                # Extract subband label from file path (e.g., .../59MHz/I/10min/...)
                import re
                m = re.search(r'/(\d+MHz)/', candidates[0])
                if m:
                    sb_freq = cfg.SUBBAND_FREQ_HZ.get(m.group(1))
                    if sb_freq:
                        if freq_min_hz is None or sb_freq['lo'] < freq_min_hz:
                            freq_min_hz = sb_freq['lo']
                        if freq_max_hz is None or sb_freq['hi'] > freq_max_hz:
                            freq_max_hz = sb_freq['hi']
        # Fallback to approximate band config
        if freq_min_hz is None:
            freq_range = cfg.WIDEBAND_BANDS.get(band_name)
            freq_min_hz = freq_range[0] * 1e6 if freq_range else None
            freq_max_hz = freq_range[1] * 1e6 if freq_range else None

        # Track whether all inputs are dewarped
        all_dewarped = all(dw for _, _, dw in subband_entries)
        dewarped_tag = "_dewarped" if all_dewarped else ""

        # Each entry is (candidates_list, weight, is_dewarped) from one
        # subband. Group files into temporal intervals by their PARSED
        # filename timestamps — sorting the filename strings is wrong
        # because the duration token precedes the timestamp (e.g.
        # '...592s...083455' sorts before '...602s...082458'), which
        # scrambles chronological order and can pair mismatched time
        # windows across subbands.
        import re
        from datetime import datetime

        def _ts_seconds(fname):
            m = re.search(r'\d{8}_\d{6}', os.path.basename(fname))
            if not m:
                return None
            return datetime.strptime(m.group(0), '%Y%m%d_%H%M%S').timestamp()

        tagged = []  # (t_seconds, file, weight)
        for candidates, weight, _ in subband_entries:
            for f in candidates:
                t = _ts_seconds(f)
                if t is not None:
                    tagged.append((t, f, weight))
        if not tagged:
            continue

        # Cluster timestamps across subbands (identical to within a few
        # seconds; anything closer than 60 s is the same interval).
        times = sorted(set(t for t, _, _ in tagged))
        clusters = []
        for t in times:
            if clusters and t - clusters[-1][-1] <= 60.0:
                clusters[-1].append(t)
            else:
                clusters.append([t])
        cluster_of = {t: i for i, c in enumerate(clusters) for t in c}

        interval_files = {}  # cluster index -> [(file, weight)]
        for t, f, w in tagged:
            interval_files.setdefault(cluster_of[t], []).append((f, w))

        counts = sorted(len(v) for v in interval_files.values())
        if counts and counts[0] != counts[-1]:
            logger.warning(
                f"  {band_name}/{pol} 10min: uneven subband coverage "
                f"across intervals ({counts[0]}-{counts[-1]} images); "
                f"stacking what exists per interval")

        out_dir = os.path.join(run_dir, band_name, pol, '10min')
        os.makedirs(out_dir, exist_ok=True)

        # Stack interval-by-interval in chronological order
        for i in sorted(interval_files.keys()):
            img_list = [f for f, _ in interval_files[i]]
            weights = [w for _, w in interval_files[i]]

            if not img_list:
                continue

            out_name = (f"{band_name}_{pol}_10min_{taper_label}_"
                        f"{robust_label}{pbcorr_tag}{dewarped_tag}_{file_tag}_t{i+1:04d}.fits")
            out_path = os.path.join(out_dir, out_name)

            history = (f"{band_name} {pol} 10min interval {i+1} "
                       f"{taper_label} {robust_label}: "
                       f"inverse-variance stack of {len(img_list)} subbands")

            if _stack_images(img_list, weights, out_path, history,
                             center_freq_hz,
                             freq_min_hz=freq_min_hz,
                             freq_max_hz=freq_max_hz,
                             regrid_cache=regrid_cache, stokes=pol):
                logger.info(f"  Stacked: {out_name} ({len(img_list)} images)")

    _10min_accumulator.clear()


def _create_one_full_product(task):
    """Worker: create a single Full-band product from R/G/B band images.

    Designed for ProcessPoolExecutor — reads FITS from disk, regrids,
    writes output. No shared state.
    """
    from reproject import reproject_interp
    from astropy.io import fits
    from astropy.wcs import WCS

    (variant_key, band_paths, band_total_weights, band_nstack,
     full_center_hz, full_freq_min_hz, full_freq_max_hz,
     total_nstack, run_dir) = task

    # Read band images
    band_data = {}
    band_hdrs = {}
    for b_name, fpath in band_paths.items():
        try:
            with fits.open(fpath) as hdul:
                band_data[b_name] = hdul[0].data.squeeze()
                band_hdrs[b_name] = hdul[0].header.copy()
        except Exception:
            continue

    if len(band_data) < 2:
        return None

    # Reference grid: Blue > Green > Red
    for ref_name in ['Blue', 'Green', 'Red']:
        if ref_name in band_data:
            break
    ref_shape = band_data[ref_name].shape
    ref_wcs = WCS(band_hdrs[ref_name]).celestial
    ref_header = band_hdrs[ref_name].copy()

    n_regrid = 0
    for b_name in list(band_data.keys()):
        if band_data[b_name].shape != ref_shape:
            src_wcs = WCS(band_hdrs[b_name]).celestial
            reprojected, footprint = reproject_interp(
                (band_data[b_name], src_wcs), ref_wcs,
                shape_out=ref_shape)
            reprojected[footprint < 0.5] = 0.0
            band_data[b_name] = reprojected
            n_regrid += 1

    # Weighted average
    from astropy.time import Time
    data_sum = np.zeros(ref_shape, dtype=np.float64)
    weight_sum = 0.0
    bands_used = []
    bmaj_list, bmin_list, bpa_list, beam_wt = [], [], [], []
    starts, ends, inttimes = [], [], []

    for b_name in ['Red', 'Green', 'Blue']:
        if b_name not in band_data:
            continue
        w = band_total_weights.get(b_name, 1.0)
        data_sum += band_data[b_name] * w
        weight_sum += w
        bands_used.append(b_name)
        hdr_b = band_hdrs[b_name]
        bmaj = hdr_b.get('BMAJ', 0)
        if bmaj > 0:
            bmaj_list.append(bmaj)
            bmin_list.append(hdr_b.get('BMIN', 0))
            bpa_list.append(hdr_b.get('BPA', 0))
            beam_wt.append(w)
        try:
            if hdr_b.get('DATE-OBS'):
                starts.append(Time(hdr_b['DATE-OBS']))
            if hdr_b.get('DATE-END'):
                ends.append(Time(hdr_b['DATE-END']))
        except Exception:
            pass
        it = hdr_b.get('INTTIME') or hdr_b.get('DURATION')
        if it and it > 0:
            inttimes.append(float(it))

    if weight_sum <= 0:
        return None

    final_data = (data_sum / weight_sum).astype(np.float32)

    # Build header — strip stale axes, rebuild proper FREQ + Stokes WCS
    for ax in [3, 4]:
        for prefix in ['NAXIS', 'CTYPE', 'CRPIX', 'CRVAL', 'CDELT',
                        'CUNIT', 'CROTA']:
            key = f'{prefix}{ax}'
            if key in ref_header:
                del ref_header[key]
    ref_header['NAXIS'] = 2
    ref_header['BTYPE'] = 'Intensity'
    ref_header['RESTFRQ'] = (full_center_hz, '[Hz] Weighted center frequency')
    ref_header['FREQ_MIN'] = (full_freq_min_hz, '[Hz] Lower edge of stacked band')
    ref_header['FREQ_MAX'] = (full_freq_max_hz, '[Hz] Upper edge of stacked band')
    full_bw_hz = full_freq_max_hz - full_freq_min_hz
    ref_header['BANDWDTH'] = (full_bw_hz, '[Hz] Total bandwidth')

    ref_header['CTYPE3'] = ('FREQ', 'Frequency axis')
    ref_header['CRVAL3'] = (float(full_center_hz), '[Hz] Band center frequency')
    ref_header['CDELT3'] = (float(full_bw_hz), '[Hz] Total bandwidth')
    ref_header['CRPIX3'] = (1.0, 'Reference pixel')
    ref_header['CUNIT3'] = ('Hz', 'Frequency unit')

    stokes_char = os.path.basename(variant_key).split('_')[0]
    stokes_code = STOKES_CODE.get(stokes_char)
    if stokes_code is not None:
        ref_header['CTYPE4'] = ('STOKES', 'Stokes axis')
        ref_header['CRVAL4'] = (float(stokes_code), f'Stokes {stokes_char}')
        ref_header['CDELT4'] = (1.0, 'Stokes increment')
        ref_header['CRPIX4'] = (1.0, 'Reference pixel')

    ref_header['NSTACK'] = (total_nstack, 'Total subbands across all bands')
    if bmaj_list:
        w_arr = np.array(beam_wt)
        w_norm = w_arr / w_arr.sum()
        ref_header['BMAJ'] = float(np.sum(w_norm * np.array(bmaj_list)))
        ref_header['BMIN'] = float(np.sum(w_norm * np.array(bmin_list)))
        ref_header['BPA'] = float(np.sum(w_norm * np.array(bpa_list)))

    # Aggregated times: R/G/B cover the same time window, so INTTIME is
    # the time-domain integration, not a sum over bands.
    if starts and ends:
        t0 = min(starts, key=lambda t: t.mjd)
        t1 = max(ends, key=lambda t: t.mjd)
        ref_header['DATE-OBS'] = (t0.isot, 'Start of first integration (UTC)')
        ref_header['MJD-OBS'] = (t0.mjd, 'Start MJD (UTC)')
        ref_header['DATE-END'] = (t1.isot, 'End of last integration (UTC)')
        ref_header['DURATION'] = (float((t1 - t0).sec),
                                  '[s] Wall-clock span (DATE-END - DATE-OBS)')
        ref_header['INTTIME'] = (float(max(inttimes)) if inttimes
                                 else float((t1 - t0).sec),
                                 '[s] Summed integration time included')
        ref_header['MJD-MID'] = ((t0.mjd + t1.mjd) / 2.0,
                                 'Midpoint MJD (UTC)')
        ref_header['TIMESYS'] = ('UTC', 'Time system')

    # Drop per-subband WSClean bookkeeping from the band header
    for key in [k for k in list(ref_header.keys())
                if str(k).startswith('WSC')]:
        del ref_header[key]

    ref_header['GITTAG'] = (cfg.PIPELINE_GITTAG, 'Pipeline code version')
    ref_header['HISTORY'] = (
        f"Full band: inverse-variance weighted stack of "
        f"{'+'.join(bands_used)} ({total_nstack} subbands)")
    for b_name, fpath in band_paths.items():
        rel = '/'.join(os.path.normpath(fpath).split(os.sep)[-4:])
        ref_header['HISTORY'] = f'  constituent: {rel}'

    out_name = f"Full_{os.path.basename(variant_key)}"
    rel_dir = os.path.dirname(variant_key)
    out_dir = os.path.join(run_dir, "Full", rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    fits.writeto(out_path, final_data, ref_header, overwrite=True)

    return n_regrid


def _create_full_band_products(run_dir, df_noise, band_total_weights,
                               band_nstack, file_tag):
    """Create Full-band products by combining Red, Green, Blue band images.

    Mathematically equivalent to stacking all 15 subbands directly
    (inverse-variance weighted mean is associative), but requires only
    2 reproject_interp calls per product (Red + Green → Blue grid)
    instead of 10. Products are processed in parallel.
    """
    logger.info("\n--- Full-band from Red+Green+Blue ---")

    full_freq_min_hz = cfg.WIDEBAND_BANDS_HZ['Full'][0]
    full_freq_max_hz = cfg.WIDEBAND_BANDS_HZ['Full'][1]
    all_subset = df_noise[(df_noise['freq'] >= 18) & (df_noise['freq'] < 85)]
    full_center_hz = _weighted_center_freq_hz(
        all_subset['freq'].tolist(), all_subset['weight'].tolist())
    total_nstack = sum(band_nstack.values())

    # Scan for band-level products and group by variant
    band_products = {}
    for b_name in ['Red', 'Green', 'Blue']:
        b_dir = os.path.join(run_dir, b_name)
        if not os.path.isdir(b_dir):
            continue
        for root, dirs, files in os.walk(b_dir):
            for fname in files:
                if not fname.endswith('.fits') or not fname.startswith(f"{b_name}_"):
                    continue
                variant_key = fname[len(b_name) + 1:]
                rel_dir = os.path.relpath(root, b_dir)
                full_key = os.path.join(rel_dir, variant_key)
                band_products.setdefault(full_key, {})[b_name] = \
                    os.path.join(root, fname)

    # Build task list
    tasks = []
    for variant_key, bands in sorted(band_products.items()):
        if len(bands) < 2:
            continue
        tasks.append((
            variant_key, bands, band_total_weights, band_nstack,
            full_center_hz, full_freq_min_hz, full_freq_max_hz,
            total_nstack, run_dir
        ))

    if not tasks:
        logger.warning("  No matching band products found for Full-band")
        return

    # Process in parallel
    import concurrent.futures
    max_workers = min(len(tasks), int(os.environ.get('FULLBAND_WORKERS', 8)))
    n_created = 0
    n_regrid = 0

    logger.info(f"  {len(tasks)} products, {max_workers} parallel workers")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        for result in pool.map(_create_one_full_product, tasks):
            if result is not None:
                n_created += 1
                n_regrid += result

    logger.info(f"  Full-band: {n_created} products, {n_regrid} regrids "
                f"(was {n_created * 10} with per-subband regridding)")


def _find_band_image(run_dir, band, pol, category, robust, file_tag):
    """Find a wideband stacked image, preferring dewarped."""
    base = os.path.join(run_dir, band, pol, category)
    # Prefer dewarped
    pat = os.path.join(base, f"{band}_{pol}_{category}_Taper_{robust}*_dewarped_*{file_tag}.fits")
    hits = [f for f in glob.glob(pat) if 'pbcorr' not in os.path.basename(f)]
    if hits:
        return sorted(hits)[0]
    # Fall back to non-dewarped
    pat = os.path.join(base, f"{band}_{pol}_{category}_Taper_{robust}*{file_tag}.fits")
    hits = [f for f in glob.glob(pat) if 'pbcorr' not in os.path.basename(f)]
    if hits:
        return sorted(hits)[0]
    return None


    """Find a wideband stacked image, preferring dewarped."""
    base = os.path.join(run_dir, band, pol, category)
    # Prefer dewarped
    pat = os.path.join(base, f"{band}_{pol}_{category}_Taper_{robust}*_dewarped_*{file_tag}.fits")
    hits = [f for f in glob.glob(pat) if 'pbcorr' not in os.path.basename(f)]
    if hits:
        return sorted(hits)[0]
    # Fall back to non-dewarped
    pat = os.path.join(base, f"{band}_{pol}_{category}_Taper_{robust}*{file_tag}.fits")
    hits = [f for f in glob.glob(pat) if 'pbcorr' not in os.path.basename(f)]
    if hits:
        return sorted(hits)[0]
    return None


def _generate_3color_pngs(run_dir, file_tag):
    """Generate 3-color RGB PNGs from wideband science stacks.

    Reads from per-band directories: {run_dir}/{band}/I/deep/
    Prefers dewarped images. Writes PNGs to QA directory.
    """
    logger.info("\n--- 3-Color PNGs ---")

    qa_dir = os.path.join(run_dir, "QA")
    os.makedirs(qa_dir, exist_ok=True)

    combos = [
        {
            'name': 'Taper_Robust-0.75',
            'red': _find_band_image(run_dir, 'Red', 'I', 'deep', 'Robust-0.75', file_tag),
            'green': _find_band_image(run_dir, 'Green', 'I', 'deep', 'Robust-0.75', file_tag),
            'blue': _find_band_image(run_dir, 'Blue', 'I', 'deep', 'Robust-0.75', file_tag),
        },
    ]

    for combo in combos:
        r = combo['red']
        g = combo['green']
        b = combo['blue']

        # Check at least 2 exist
        exists = sum(1 for f in [r, g, b] if f and os.path.exists(f))
        if exists < 2:
            logger.info(f"  {combo['name']}: only {exists} bands  skipping")
            continue

        out_png = os.path.join(qa_dir, f"I_deep_{combo['name']}_{file_tag}_3color.png")

        try:
            _make_3color_png(
                r if r and os.path.exists(r) else None,
                g if g and os.path.exists(g) else None,
                b if b and os.path.exists(b) else None,
                out_png,
                title=f"OVRO-LWA 3-Color ({combo['name']})"
            )
            ATTACHMENT_FILES.append(out_png)
        except Exception as e:
            logger.error(f"  3-color {combo['name']} failed: {e}")


def _make_3color_png(red_path, green_path, blue_path, out_png, title=""):
    """Generate a 3-color PNG from Red/Green/Blue FITS images.

    Applies spectral index correction and horizon masking.
    When input images have different pixel grids (variable resolution mode),
    all channels are reprojected onto the largest (highest-resolution) grid.
    Normalization uses the common valid region across all channels to ensure
    consistent color balance regardless of FOV differences.
    """
    channels = []
    headers = []
    footprints = []  # Track regridding footprint per channel
    for fpath in [red_path, green_path, blue_path]:
        if fpath and os.path.exists(fpath):
            with fits.open(fpath) as hdul:
                channels.append(hdul[0].data.squeeze().astype(np.float64))
                headers.append(hdul[0].header.copy())
                footprints.append(None)  # filled during regridding if needed
        else:
            channels.append(None)
            headers.append(None)
            footprints.append(None)

    valid = [c for c in channels if c is not None]
    if len(valid) < 2:
        return

    # Find the largest image as the reference grid
    ref_idx = None
    ref_size = 0
    for i in range(3):
        if channels[i] is not None and channels[i].size > ref_size:
            ref_size = channels[i].size
            ref_idx = i
    ref_shape = channels[ref_idx].shape
    ref_header = headers[ref_idx]

    # Regrid mismatched channels, tracking footprints
    needs_regrid = any(c is not None and c.shape != ref_shape for c in channels)
    if needs_regrid:
        try:
            from reproject import reproject_interp
            ref_wcs = WCS(ref_header).celestial
            for i in range(3):
                if channels[i] is not None and channels[i].shape != ref_shape:
                    src_wcs = WCS(headers[i]).celestial
                    reprojected, footprint = reproject_interp(
                        (channels[i], src_wcs), ref_wcs,
                        shape_out=ref_shape)
                    reprojected[footprint < 0.5] = 0.0
                    channels[i] = reprojected
                    footprints[i] = footprint
                    n_outside = np.sum(footprint < 0.5)
                    logger.info(f"  3-color: regridded channel {i} to {ref_shape} "
                                f"({100*n_outside/footprint.size:.1f}% outside FOV)")
        except ImportError:
            logger.warning("  reproject not available — zeroing mismatched channels")

    for i in range(3):
        if channels[i] is None:
            channels[i] = np.zeros(ref_shape, dtype=np.float64)
        elif channels[i].shape != ref_shape:
            channels[i] = np.zeros(ref_shape, dtype=np.float64)

    # Spectral index correction
    freqs = np.array([29.5, 52.5, 74.5])
    ref_freq = np.exp(np.mean(np.log(freqs)))
    alpha = cfg.THREE_COLOR_SPECTRAL_INDEX
    scale_factors = (freqs / ref_freq) ** alpha
    for i in range(3):
        channels[i] = channels[i] * scale_factors[i]

    # Build common valid mask: finite, nonzero, and within FOV for ALL channels
    valid_mask = np.ones(ref_shape, dtype=bool)
    for i in range(3):
        valid_mask &= np.isfinite(channels[i]) & (channels[i] != 0)
        if footprints[i] is not None:
            valid_mask &= (footprints[i] >= 0.5)

    n_valid = np.sum(valid_mask)
    logger.info(f"  3-color: common valid region {n_valid} pixels "
                f"({100*n_valid/valid_mask.size:.1f}% of frame)")

    # Normalize each channel using the common valid region
    normed = []
    for ch in channels:
        pixels = ch[valid_mask]
        if len(pixels) > 0:
            vmin, vmax = np.nanpercentile(pixels, [0.5, 99.8])
        else:
            vmin, vmax = 0, 1
        if vmax <= vmin:
            vmax = vmin + 1
        clipped = np.clip(ch, vmin, vmax)
        scaled = (clipped - vmin) / (vmax - vmin)
        # Zero out pixels outside the common valid region
        scaled[~valid_mask] = 0.0
        normed.append(scaled)

    rgb = np.stack(normed, axis=-1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.imshow(rgb, origin='lower', interpolation='nearest')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=14, color='white',
                     bbox=dict(facecolor='black', alpha=0.7))
    plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    logger.info(f"  Saved: {os.path.basename(out_png)}")


# =====================================================================
# HELPERS
# =====================================================================

def _plot_thermal_noise(df_noise, qa_dir, run_dir):
    """Generate thermal noise vs frequency plot."""
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(df_noise['freq'], df_noise['rms'] * 1000, 'o-',
                     color='#2196F3', markersize=6, linewidth=1.5)
        ax.set_xlabel('Frequency (MHz)', fontsize=12)
        ax.set_ylabel('RMS Noise (mJy/beam)', fontsize=12)
        ax.set_title('Thermal Noise vs Frequency', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(15, 85)
        png_path = os.path.join(qa_dir, f"{_qa_prefix(run_dir)}thermal_noise_vs_freq.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=120)
        plt.close()
        ATTACHMENT_FILES.append(png_path)
    except Exception as e:
        logger.warning(f"Thermal noise plot failed: {e}")


def _generate_dewarp_summary(run_dir, qa_dir):
    """Cross-frequency dewarping diagnostic from per-subband stats JSON."""
    stats_files = sorted(glob.glob(
        os.path.join(run_dir, "Dewarp_Diagnostics", "warp_stats_*.json")))
    if not stats_files:
        return

    records = []
    for sf in stats_files:
        try:
            with open(sf) as f:
                records.append(json.load(f))
        except Exception:
            pass

    if len(records) < 3:
        return

    records.sort(key=lambda r: r['freq_mhz'])
    freqs = np.array([r['freq_mhz'] for r in records])
    median_shift = np.array([r.get('median_shift_arcmin', 0) for r in records])

    # Save summary CSV
    csv_path = os.path.join(qa_dir, f"{_qa_prefix(run_dir)}dewarp_summary.csv")
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['freq_mhz', 'median_shift_arcmin', 'n_anchors',
                             'ref_label'])
            for r in records:
                writer.writerow([
                    r['freq_mhz'],
                    f"{r.get('median_shift_arcmin', 0):.4f}",
                    r.get('n_anchors', 0),
                    r.get('ref_label', ''),
                ])
    except Exception:
        pass

    # Plot
    try:
        valid = median_shift > 0
        if np.sum(valid) < 3:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(freqs[valid], median_shift[valid], 'o-', color='#2196F3',
                markersize=6)

        # ??² model
        log_freq = np.log10(freqs[valid])
        log_shift = np.log10(median_shift[valid])
        coeffs = np.polyfit(log_freq, log_shift, 1)
        freq_model = np.linspace(freqs.min() * 0.9, freqs.max() * 1.1, 100)
        shift_model = 10**coeffs[1] * freq_model**coeffs[0]
        ax.plot(freq_model, shift_model, '--', color='#FF5722',
                label=f'Fit: ?^{{{coeffs[0]:.2f}}}')

        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Median Shift (arcmin)')
        ax.set_title('Ionospheric Distortion vs Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

        png_path = os.path.join(qa_dir, f"{_qa_prefix(run_dir)}dewarp_vs_frequency.png")
        plt.tight_layout()
        plt.savefig(png_path, dpi=120)
        plt.close()
        ATTACHMENT_FILES.append(png_path)
    except Exception as e:
        logger.warning(f"Dewarp summary plot failed: {e}")


# =====================================================================
# WIDEBAND TARGET PHOTOMETRY
# =====================================================================

def run_wideband_target_photometry(run_dir, target_files, skip_bands=()):
    """Run target photometry on wideband stacked images."""
    logger.info("\n--- Wideband Target Photometry ---")
    if cutout is None or not target_files:
        return

    samples_dir = os.path.join(run_dir, "samples")
    detections_dir = os.path.join(run_dir, "detections")

    for band in ['Red', 'Green', 'Blue', 'Full']:
        if band in skip_bands:
            continue
        band_dir = os.path.join(run_dir, band)
        if not os.path.isdir(band_dir):
            continue

        for t_file in target_files:
            if not os.path.exists(t_file):
                continue
            try:
                s_name = os.path.splitext(os.path.basename(t_file))[0]
                targets = cutout.load_targets(t_file)
                logger.info(f"  {band}: {len(targets)} targets from {os.path.basename(t_file)}")

                for nm, crd, det_stokes, confusing_sources in targets:
                    try:
                        cutout.process_target(
                            band_dir, nm, crd, s_name,
                            samples_dir, detections_dir,
                            fallback_dir=band_dir,
                            detection_stokes=det_stokes,
                            confusing_sources=confusing_sources
                        )
                    except Exception as e:
                        logger.error(f"    WB target '{nm}' @ {band} failed: {e}")
            except Exception as e:
                logger.error(f"  WB target file {t_file} @ {band} failed: {e}")


# =====================================================================
# ARCHIVE (placeholder  products stay in place for now)
# =====================================================================

def archive_wideband_products(run_dir):
    """Archive wideband products to the Phase 2 archive location.

    Currently a no-op since PHASE2_ARCHIVE_DIR == PHASE2_WORKING_DIR.
    """
    if cfg.PHASE2_ARCHIVE_DIR == cfg.PHASE2_WORKING_DIR:
        logger.info("\nArchive: products are already in final location")
        return

    # Future: copy from working dir to archive
    logger.info(f"\nArchiving to {cfg.PHASE2_ARCHIVE_DIR}...")
    # TODO: Implement structured copy when archive moves to Lustre


# =====================================================================
# GATHER DETECTIONS (§1.7, §1.8)
# =====================================================================

def gather_detections(run_dir):
    """Compile all detections into separate CSV files.

    FIX §1.7: No double-counting between subband and wideband.
    FIX §1.8: Three separate files.
    """
    logger.info("\n--- Gathering Detections ---")

    subband_rows = []
    wideband_rows = []

    # -- Target photometry detections --
    for csv_path in glob.glob(os.path.join(run_dir, "samples", "*", "*",
                                            "*", "*_photometry.csv")):
        try:
            df = pd.read_csv(csv_path)
            det = df[df.get('Detection', False) == True]
            for _, row in det.iterrows():
                d = row.to_dict()
                d['source_csv'] = os.path.basename(csv_path)
                # Determine if wideband from path
                parts = csv_path.split(os.sep)
                is_wb = any('WB-' in p for p in parts)
                if is_wb:
                    wideband_rows.append(d)
                else:
                    subband_rows.append(d)
        except Exception:
            pass

    # -- Transient detections --
    _wb_band_prefixes = ('Red_', 'Green_', 'Blue_', 'Full_')
    for csv_path in glob.glob(os.path.join(run_dir, "detections", "transients",
                                            "candidates_*.csv")):
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                d = row.to_dict()
                img_name = str(d.get('image', ''))
                is_wb = img_name.startswith(_wb_band_prefixes)
                if is_wb:
                    wideband_rows.append(d)
                else:
                    subband_rows.append(d)
        except Exception:
            pass

    # -- Solar system detections --
    for csv_path in glob.glob(os.path.join(run_dir, "samples", "SolarSystem",
                                            "*", "*_photometry.csv")):
        try:
            df = pd.read_csv(csv_path)
            det = df[df.get('Detection', False) == True]
            for _, row in det.iterrows():
                d = row.to_dict()
                parts = csv_path.split(os.sep)
                is_wb = any('WB-' in p for p in parts)
                if is_wb:
                    wideband_rows.append(d)
                else:
                    subband_rows.append(d)
        except Exception:
            pass

    # Write CSVs
    if subband_rows:
        csv_path = os.path.join(run_dir, "subband_detections.csv")
        pd.DataFrame(subband_rows).to_csv(csv_path, index=False)
        logger.info(f"  Subband detections: {len(subband_rows)} ? {csv_path}")
        ATTACHMENT_FILES.append(csv_path)

    if wideband_rows:
        csv_path = os.path.join(run_dir, "wideband_detections.csv")
        pd.DataFrame(wideband_rows).to_csv(csv_path, index=False)
        logger.info(f"  Wideband detections: {len(wideband_rows)} ? {csv_path}")
        ATTACHMENT_FILES.append(csv_path)

    combined = subband_rows + wideband_rows
    if combined:
        csv_path = os.path.join(run_dir, "combined_detections.csv")
        pd.DataFrame(combined).to_csv(csv_path, index=False)
        logger.info(f"  Combined detections: {len(combined)} ? {csv_path}")

    if not combined:
        logger.info("  No detections found")
        EMAIL_REPORT_LINES.append("No significant detections found.")


# =====================================================================
# EMAIL REPORT
# =====================================================================

def _summarize_transient_detections(det_dir):
    """Scan detections directory for transient search CSVs, build HTML summary.

    Finds CSVs from both V search (search_*.csv) and I search
    (candidates_I_*.csv), counts PASS candidates, returns HTML string.
    """
    if not os.path.isdir(det_dir):
        return ""

    v_total = 0
    v_pass = 0
    i_total = 0
    i_pass = 0
    top_v = []
    top_i = []

    try:
        # Stokes V CSVs: search_*.csv
        for csv_path in sorted(glob.glob(os.path.join(det_dir, "search_*.csv"))):
            try:
                df = pd.read_csv(csv_path)
                if 'status' not in df.columns:
                    continue
                n = len(df)
                p = len(df[df['status'] == 'PASS'])
                v_total += n
                v_pass += p
                if p > 0:
                    passed = df[df['status'] == 'PASS'].head(3)
                    for _, row in passed.iterrows():
                        top_v.append({
                            'snr': row.get('snr', 0),
                            'ra': row.get('ra_hms', ''),
                            'dec': row.get('dec_dms', ''),
                            'file': os.path.basename(csv_path),
                        })
                ATTACHMENT_FILES.append(csv_path)
            except Exception:
                pass

        # Stokes I CSVs: candidates_I_*.csv
        for csv_path in sorted(glob.glob(os.path.join(det_dir, "candidates_I_*.csv"))):
            try:
                df = pd.read_csv(csv_path)
                if 'status' not in df.columns:
                    continue
                n = len(df)
                p = len(df[df['status'] == 'PASS'])
                i_total += n
                i_pass += p
                if p > 0:
                    passed = df[df['status'] == 'PASS'].nlargest(
                        3, 'local_snr' if 'local_snr' in df.columns else 'global_snr')
                    for _, row in passed.iterrows():
                        top_i.append({
                            'snr': row.get('local_snr', row.get('global_snr', 0)),
                            'ra': row.get('ra_hms', ''),
                            'dec': row.get('dec_dms', ''),
                            'file': os.path.basename(csv_path),
                        })
                ATTACHMENT_FILES.append(csv_path)
            except Exception:
                pass

    except Exception:
        return ""

    if v_total == 0 and i_total == 0:
        return ""

    html = "<ul>"
    html += f"<li>Stokes V: {v_pass} PASS / {v_total} candidates"
    if v_pass > 0:
        html += f" (top: SNR {top_v[0]['snr']:.1f} at {top_v[0]['ra']})"
    html += "</li>"
    html += f"<li>Stokes I: {i_pass} PASS / {i_total} candidates"
    if i_pass > 0:
        html += f" (top: SNR {top_i[0]['snr']:.1f} at {top_i[0]['ra']})"
    html += "</li>"
    html += "</ul>"

    if v_pass + i_pass > 0:
        html += f"<p>See attached CSVs for full candidate lists.</p>"

    return html

def post_to_slack(run_dir, run_duration=None):
    """Post QA summary to Slack via webhook."""
    if not cfg.SLACK_WEBHOOK_URL:
        return

    import urllib.request

    prefix = _qa_prefix(run_dir)
    run_name = os.path.basename(run_dir)

    # Build summary
    lines = [f"*Phase 2 Complete:* `{run_name}`"]
    if run_duration:
        lines.append(f"Duration: {int(run_duration//60)}m {int(run_duration%60)}s")

    # Ionosphere
    for ws in sorted(glob.glob(os.path.join(
            run_dir, "Dewarp_Diagnostics", "*warp_stats_*73MHz*.json"))):
        try:
            with open(ws) as f:
                d = json.load(f)
            dist = d.get('disturbance_rms_arcmin')
            qual = d.get('disturbance_quality', '')
            if dist:
                lines.append(f"Ionosphere: {qual} (RMS {dist:.2f}')")
        except Exception:
            pass

    for vf in sorted(glob.glob(os.path.join(
            run_dir, "Dewarp_Diagnostics", "*ionosphere_variability_*.json"))):
        try:
            with open(vf) as f:
                d = json.load(f)
            lines.append(f"Variability: {d.get('quality', '?')} "
                         f"(mean {d.get('mean_variability', 0):.3f})")
        except Exception:
            pass

    # Noise
    for nc in sorted(glob.glob(os.path.join(run_dir, "QA", "*thermal_noise.csv"))):
        try:
            df = pd.read_csv(nc)
            deep = df[df['category'] == 'deep'] if 'category' in df.columns else df
            lines.append(f"Noise: median {deep['rms'].median()*1000:.1f} mJy, "
                         f"best {deep['rms'].min()*1000:.1f} mJy")
        except Exception:
            pass

    # Flux check
    fc_pngs = glob.glob(os.path.join(run_dir, "QA", "*Flux_Check_*.png"))
    lines.append(f"Flux check: {len(fc_pngs)} sources")

    payload = json.dumps({"text": "\n".join(lines)}).encode()
    try:
        req = urllib.request.Request(
            cfg.SLACK_WEBHOOK_URL, data=payload,
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        logger.info("  Slack notification sent")
    except Exception as e:
        logger.warning(f"  Slack post failed: {e}")


def send_email_report(run_dir, run_command=None, run_duration=None, lst=None, date=None):
    """Send summary email with diagnostics and detection list."""
    logger.info("\n--- Sending Email Report ---")

    secrets_path = os.path.expanduser(cfg.SECRETS_FILE)
    creds = None
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path) as f:
                creds = json.load(f)
        except Exception:
            pass

    msg = MIMEMultipart()
    msg['From'] = creds['email'] if creds else "pipeline@astro.caltech.edu"
    recips = cfg.EMAIL_RECIPIENT
    msg['To'] = ", ".join(recips) if isinstance(recips, (list, tuple)) else recips
    obs_tag = " ".join(p for p in (lst, date) if p)   # e.g. "11h 2024-12-24"
    subj_obs = f"{obs_tag} — " if obs_tag else ""
    msg['Subject'] = f"OVRO-LWA Phase 2: {subj_obs}{os.path.basename(run_dir)}"

    body = "<h3>Phase 2 Pipeline Complete</h3>"
    body += f"<p>Run Directory: <code>{run_dir}</code></p>"
    if run_command:
        body += f"<p>Command: <code>{run_command}</code></p>"
    if run_duration is not None:
        hours = int(run_duration // 3600)
        mins = int((run_duration % 3600) // 60)
        body += f"<p>Duration: {hours}h {mins}m ({run_duration:.0f}s)</p>"

    # Attach the pipeline log file
    log_files = sorted(glob.glob(os.path.join(run_dir, "phase2_*.log")))
    for lf in log_files:
        ATTACHMENT_FILES.append(lf)

    # Thermal noise summary
    noise_csvs = sorted(glob.glob(os.path.join(run_dir, "QA", "*thermal_noise.csv")))
    if noise_csvs:
        noise_csv = noise_csvs[-1]
        try:
            df = pd.read_csv(noise_csv)
            body += "<h4>Thermal Noise</h4><ul>"
            body += f"<li>Median: {df['rms'].median()*1000:.1f} mJy/beam</li>"
            body += f"<li>Best: {df['rms'].min()*1000:.1f} mJy/beam "
            body += f"@ {df.loc[df['rms'].idxmin()]['freq']:.0f} MHz</li>"
            body += "</ul>"
            ATTACHMENT_FILES.append(noise_csv)
        except Exception:
            pass

    # Flux check QA files
    qa_dir = os.path.join(run_dir, "QA")
    for fc_png in sorted(glob.glob(os.path.join(qa_dir, "*Flux_Check_*.png"))):
        ATTACHMENT_FILES.append(fc_png)
    fc_csvs = sorted(glob.glob(os.path.join(qa_dir, "*flux_check_hybrid.csv")))
    for fc in fc_csvs:
        ATTACHMENT_FILES.append(fc)

    # Ionosphere quality summary and dewarp diagnostic attachments
    diag_dir = os.path.join(run_dir, "Dewarp_Diagnostics")
    if os.path.isdir(diag_dir):
        body += "<h4>Ionosphere Quality</h4>"

        # Disturbance metric from 73 MHz deep anchor (warp_stats JSON)
        stats_jsons = sorted(glob.glob(
            os.path.join(diag_dir, "*warp_stats_*73MHz*.json")))
        if stats_jsons:
            try:
                with open(stats_jsons[-1]) as _f:
                    anchor_stats = json.load(_f)
                dist_rms = anchor_stats.get('disturbance_rms_arcmin')
                dist_qual = anchor_stats.get('disturbance_quality')
                med_shift = anchor_stats.get('median_shift_arcmin', 0)
                if dist_rms is not None:
                    body += (f"<p><strong>Disturbance: {dist_qual} "
                             f"(RMS {dist_rms:.2f}')</strong> — "
                             f"median shift {med_shift:.2f}', "
                             f"{anchor_stats.get('n_anchors', 0)} anchors "
                             f"at 73 MHz</p>")
            except Exception:
                pass

        # Variability summary from ionosphere_variability JSON
        var_jsons = sorted(glob.glob(
            os.path.join(diag_dir, "*ionosphere_variability_*.json")))
        if var_jsons:
            try:
                with open(var_jsons[-1]) as _f:
                    iono_data = json.load(_f)
                body += (f"<p><strong>Variability: "
                         f"{iono_data.get('quality', 'Unknown')}"
                         f"</strong></p><ul>")
                body += (f"<li>Mean variability: "
                         f"{iono_data.get('mean_variability', 0):.3f}</li>")
                body += (f"<li>Max variability: "
                         f"{iono_data.get('max_variability', 0):.3f}</li>")
                body += (f"<li>{iono_data.get('n_frames', 0)} intervals at "
                         f"{iono_data.get('subband', '73MHz')}</li>")
                body += "</ul>"
            except Exception:
                pass

        if not stats_jsons and not var_jsons:
            body += "<p>No ionosphere diagnostics available.</p>"

        body += ("<p><small>Disturbance: spatial structure in the deep "
                 "anchor (0 = smooth). Variability: frame-to-frame "
                 "evolution across 10-min intervals (0 = frozen, "
                 "~1 = rapidly reorganizing).</small></p>")

        # Attach anchor vector map (73 MHz deep)
        anchor_pngs = sorted(glob.glob(
            os.path.join(diag_dir, "*anchor_vectors_*73MHz*.png")))
        for ap in anchor_pngs:
            ATTACHMENT_FILES.append(ap)

        # Attach ionosphere evolution GIF
        iono_gifs = sorted(glob.glob(
            os.path.join(diag_dir, "*ionosphere_evolution_*.gif")))
        for ig in iono_gifs:
            ATTACHMENT_FILES.append(ig)

        # Attach dewarp-vs-frequency summary
        dewarp_pngs = sorted(glob.glob(
            os.path.join(qa_dir, "*dewarp_vs_frequency.png")))
        for dp in dewarp_pngs:
            ATTACHMENT_FILES.append(dp)

    # Transient detection summary  scan detections directory for CSVs
    body += "<h4>Transient Detections</h4>"
    det_dir = os.path.join(run_dir, "detections")
    transient_summary = _summarize_transient_detections(det_dir)
    if transient_summary:
        body += transient_summary
    elif EMAIL_REPORT_LINES:
        body += "<pre>" + "\n".join(EMAIL_REPORT_LINES) + "</pre>"
    else:
        body += "<p>No transient search results found.</p>"

    msg.attach(MIMEText(body, 'html'))

    # Attachments (deduplicate, size limit)
    total_size = 0
    seen = set()
    for f in ATTACHMENT_FILES:
        if f in seen or not os.path.exists(f):
            continue
        seen.add(f)
        s = os.path.getsize(f)
        if total_size + s > 24 * 1024 * 1024:
            break
        total_size += s
        with open(f, "rb") as att:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(att.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",
                        f"attachment; filename={os.path.basename(f)}")
        msg.attach(part)

    try:
        if creds:
            s = smtplib.SMTP(creds['server'], creds['port'])
            s.starttls()
            s.login(creds['email'], creds['password'])
        else:
            s = smtplib.SMTP('localhost')
        s.send_message(msg)
        s.quit()
        logger.info(f"Email sent to {msg['To']}")
    except Exception as e:
        logger.error(f"Email failed: {e}")

    # Post to Slack
    post_to_slack(run_dir, run_duration=run_duration)
