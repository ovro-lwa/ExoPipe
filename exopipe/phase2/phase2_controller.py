#!/usr/bin/env python3
"""
phase2_controller.py  OVRO-LWA Phase 2 Science Pipeline Controller

Takes images produced by the Celery imaging pipeline (on Lustre) and runs
all science extraction steps:

  1. Discovery  find all subband images for a given LST/date
  2. Copy to NVMe  stage images for fast I/O (no snapshots)
  3. Cascade dewarping  73 MHz anchor, cascade to all subbands
  4. Wideband product formation  noise analysis, stacking, 3-color PNGs
  5. Photometry  source-grouped: each target across all subbands + wideband
  6. Transient search  per-subband + wideband
  7. Gather detections + email report

Usage:
  python phase2_controller.py \\
      --lst 10h --date 2024-12-22 \\
      --targets 10pc_sample.csv HotJupiters.csv \\
      --catalog OVRO_LWA_Local_Volume_Targets.csv

  # Wideband products only (first test after dewarping):
  python phase2_controller.py \\
      --lst 10h --date 2024-12-22 \\
      --run_label Science_20260404_191323 \\
      --skip_copy --skip_dewarp --skip_photometry --skip_transients
"""
import argparse
import os
import sys
import subprocess
import glob
import shutil
import logging
import time
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from exopipe.common import phase2_config as cfg

# --- Logging ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PHASE2] - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('phase2')
sys.stdout.reconfigure(line_buffering=True)


# =====================================================================
# HELPERS
# =====================================================================

def _glob_prefer_dewarped(pattern, exclude_pbcorr=True):
    """Glob for FITS files, preferring dewarped variants.

    If both dewarped and non-dewarped versions exist, returns only dewarped.
    The pattern should match both variants (e.g. {band}_I_deep_Taper_*.fits).
    """
    hits = sorted(glob.glob(pattern))
    if exclude_pbcorr:
        hits = [f for f in hits if 'pbcorr' not in os.path.basename(f)]

    # Separate dewarped and non-dewarped
    dewarped = [f for f in hits if '_dewarped' in os.path.basename(f)]
    if dewarped:
        return dewarped
    return hits


# =====================================================================
# STEP 1: DISCOVERY
# =====================================================================

def discover_subbands(lst_label, date_str, quiet=False):
    """Scan Celery output for available subbands.

    Handles multi-run case: if a subband appears in multiple Run_
    directories, takes the latest by timestamp in the run label.
    quiet=True suppresses the per-subband logging (used when polling
    repeatedly while waiting for Phase 1 to finish).

    Returns dict: {subband: full_path_to_subband_dir}
    """
    base = os.path.join(cfg.CELERY_IMAGE_ROOT, lst_label, date_str)
    if not os.path.isdir(base):
        logger.error(f"Input directory not found: {base}")
        sys.exit(1)

    # Find all Run_ directories
    run_dirs = sorted(glob.glob(os.path.join(base, "Run_*")))
    if not run_dirs:
        logger.error(f"No Run_ directories found in {base}")
        sys.exit(1)

    if not quiet:
        logger.info(f"Found {len(run_dirs)} run(s) in {base}:")
        for rd in run_dirs:
            logger.info(f"  {os.path.basename(rd)}")

    # Build subband map  latest run wins if duplicated
    subband_map = {}
    for run_dir in run_dirs:
        run_label = os.path.basename(run_dir)
        for sb_dir in glob.glob(os.path.join(run_dir, "*MHz")):
            sb = os.path.basename(sb_dir)
            if sb in cfg.SKIP_SUBBANDS:
                continue
            if sb in subband_map and not quiet:
                logger.info(f"  {sb}: found in {run_label} "
                            f"(supersedes {os.path.basename(os.path.dirname(subband_map[sb]))})")
            subband_map[sb] = sb_dir

    if not quiet:
        logger.info(f"\nDiscovered {len(subband_map)} subbands:")
        for sb in sorted(subband_map.keys(),
                         key=lambda x: int(x.replace('MHz', ''))):
            run = os.path.basename(os.path.dirname(subband_map[sb]))
            n_fits = len(glob.glob(os.path.join(subband_map[sb], "*", "*", "*.fits")))
            logger.info(f"  {sb:>6s}  {run} ({n_fits} FITS files)")

    return subband_map


def validate_subband(sb_dir, subband, logger):
    """Check that a subband directory has the expected image products.

    Returns list of warnings (empty if all good).
    """
    warnings_list = []

    expected_dirs = [
        ("I/deep", "Stokes I deep images"),
        ("V/deep", "Stokes V deep images"),
        ("I/10min", "Stokes I 10-minute intervals"),
        ("V/10min", "Stokes V 10-minute intervals"),
    ]

    for subdir, desc in expected_dirs:
        d = os.path.join(sb_dir, subdir)
        if not os.path.isdir(d):
            warnings_list.append(f"{subband}: missing {subdir} ({desc})")
            continue
        fits_files = glob.glob(os.path.join(d, "*.fits"))
        if not fits_files:
            warnings_list.append(f"{subband}: {subdir} is empty")

    # Check for PB-corrected images (needed for photometry)
    pbcorr = glob.glob(os.path.join(sb_dir, "I", "deep", "*pbcorr*.fits"))
    if not pbcorr:
        warnings_list.append(f"{subband}: no PB-corrected images in I/deep")

    return warnings_list


# =====================================================================
# STEP 2: COPY TO NVMe
# =====================================================================

def copy_to_nvme(subband_map, work_dir):
    """Copy image data from Lustre to NVMe working directory.

    Copies only *image* and *pbcorr* FITS files for each subband.
    Skips wsclean auxiliary products (dirty, model, psf, residual, mask).
    """
    logger.info(f"\nCopying to NVMe: {work_dir}")

    for d in ['Dewarp_Diagnostics', 'QA']:
        os.makedirs(os.path.join(work_dir, d), exist_ok=True)

    for subband, src_dir in sorted(subband_map.items(),
                                    key=lambda x: int(x[0].replace('MHz', ''))):
        dest_sb = os.path.join(work_dir, subband)

        if os.path.isdir(dest_sb):
            n_existing = len(glob.glob(os.path.join(dest_sb, "*", "*", "*.fits")))
            if n_existing > 0:
                logger.info(f"  {subband}: already on NVMe ({n_existing} files)  skipping copy")
                continue

        logger.info(f"  Copying {subband}...")

        # Only copy *image* FITS files (the only science products).
        # All other wsclean outputs (dirty, model, psf, residual, mask)
        # are excluded by requiring '-image' in the filename.
        n_copied = 0
        for pol in ['I', 'V']:
            src_pol = os.path.join(src_dir, pol)
            if not os.path.isdir(src_pol):
                continue
            for cat in ['deep', '10min']:
                src_cat = os.path.join(src_pol, cat)
                if not os.path.isdir(src_cat):
                    continue
                dest_cat = os.path.join(dest_sb, pol, cat)
                os.makedirs(dest_cat, exist_ok=True)
                for f in glob.glob(os.path.join(src_cat, "*.fits")):
                    bn = os.path.basename(f)
                    if '-image' not in bn:
                        continue
                    if ".pybdsf" in bn:
                        continue
                    shutil.copy2(f, os.path.join(dest_cat, bn))
                    n_copied += 1

        logger.info(f"  {subband}: {n_copied} image files copied")

    return work_dir


# =====================================================================
# STEP 4: WIDEBAND PRODUCT FORMATION
# =====================================================================

def run_wideband_formation(work_dir, logger, skip_wideband=False):
    """Create all wideband data products.

    Runs immediately after dewarping  no dependency on photometry.

    Steps:
      4a. Flux scale check (calibrator validation)
      4b. Noise analysis + inverse-variance weighted stacking
      4c. 3-color PNGs
      4d. Archive (placeholder)
    """
    logger.info("\n" + "=" * 60)
    logger.info("WIDEBAND PRODUCT FORMATION")
    logger.info("=" * 60)

    from exopipe.phase2.phase2_science_agg import (
        run_wideband_stacking,
        archive_wideband_products,
    )

    # Set the module-level logger in phase2_science_agg
    from exopipe.phase2 import phase2_science_agg
    phase2_science_agg.logger = logger

    # Ensure directories exist (may not if --skip_copy was used)
    for d in ['QA']:
        os.makedirs(os.path.join(work_dir, d), exist_ok=True)

    # Wideband stacking (includes noise analysis + 3-color PNGs).
    # SKIPPED when the subband set was incomplete (Gregg ruling 2026-07-09:
    # no wideband formation from partial sets — per-subband products only).
    skip_bands = set(skip_wideband) if skip_wideband else set()
    if skip_bands >= {'Red', 'Green', 'Blue', 'Full'}:
        logger.warning("ALL wideband bands incomplete — wideband formation "
                       "skipped entirely (per-band ruling 2026-07-09).")
    else:
        run_wideband_stacking(work_dir, skip_bands=skip_bands)

        # 4d. Archive
        archive_wideband_products(work_dir)


# =====================================================================
# STEP 5: PHOTOMETRY (subband-parallel)
# =====================================================================

def _photometry_worker(task):
    """Worker: process all targets for one frequency band.

    Runs in a subprocess. Uses the FITS cache so each all-sky image
    is loaded only once per subband, then reused for all targets.

    Returns list of (sample_name, target_name, summary_row_dict) tuples.
    """
    from exopipe.phase3.lib.photometry import cutout

    band_label = task['band_label']
    targets_by_sample = task['targets_by_sample']
    samples_dir = task['samples_dir']
    detections_dir = task['detections_dir']

    # Enable debug flux logging if requested
    cutout.set_debug_log(task.get('debug_log'))

    results = []

    for s_name, target_list in targets_by_sample:
        for nm, crd, det_stokes, confusing_sources in target_list:
            try:
                if task.get('explicit_i') and task.get('explicit_v'):
                    summary = cutout.process_target(
                        task['run_dir'], nm, crd, s_name,
                        samples_dir, detections_dir,
                        detection_stokes=det_stokes,
                        confusing_sources=confusing_sources,
                        deep_i_path=task['explicit_i'],
                        deep_v_path=task['explicit_v'],
                        freq_label=band_label,
                        snapshots_10min_explicit=task.get('snapshots_10min'),
                        deep_i_sub_ref=task.get('deep_i_sub_ref')
                    )
                else:
                    summary = cutout.process_target(
                        task['sb_dir'], nm, crd, s_name,
                        samples_dir, detections_dir,
                        fallback_dir=task['sb_dir'],
                        detection_stokes=det_stokes,
                        confusing_sources=confusing_sources,
                        freq_label=band_label
                    )
                if summary:
                    for row in summary:
                        results.append((s_name, nm, row))
            except Exception as e:
                print(f"    {nm} @ {band_label}: {e}")

    cutout.clear_fits_cache()
    return band_label, results


def run_all_photometry(work_dir, subband_list, args, logger):
    """Run all target and solar system photometry, parallelized by frequency.

    Each subband (and each wideband band) is processed as an independent
    worker. Within each worker, the FITS cache ensures all-sky images are
    read only once, then reused across all targets.
    """
    logger.info("\n" + "=" * 60)
    logger.info("PHOTOMETRY (subband-parallel)")
    logger.info("=" * 60)

    try:
        from exopipe.phase3.lib.photometry import cutout
    except ImportError:
        cutout = None
        logger.warning("cutout module not available  skipping target photometry")

    try:
        from exopipe.phase3.lib.photometry import solar_system_cutout
    except ImportError:
        solar_system_cutout = None
        logger.warning("solar_system_cutout not available  skipping solar system")

    target_files = args.targets or []
    samples_dir = os.path.join(work_dir, "samples")
    detections_dir = os.path.join(work_dir, "detections")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(detections_dir, exist_ok=True)

    debug_log_path = None
    if getattr(args, 'debug', False):
        debug_log_path = os.path.join(work_dir, "flux_debug.log")
        logger.info(f"  Debug flux log: {debug_log_path}")

    # Build wideband PBcorr image pairs (per band)
    # Discovers both deep (for photometry) and 10min (for differenced photometry)
    # Also finds Robust-0 non-PBcorr deep I for subtraction reference
    # Wideband products live in {work_dir}/{band}/{pol}/{category}/
    wb_bands = []  # (band, deep_i, deep_v, snapshots_10min, deep_i_sub_ref)
    for band in ['Red', 'Green', 'Blue', 'Full']:
        band_dir = os.path.join(work_dir, band)
        if not os.path.isdir(band_dir):
            continue

        # Deep I PBcorr (for photometry)  prefer dewarped
        i_pbcorr = None
        for robust in ['Robust-0.75', 'Robust-0']:
            hits = _glob_prefer_dewarped(
                os.path.join(band_dir, 'I', 'deep',
                             f"{band}_I_deep_Taper_{robust}*pbcorr*.fits"),
                exclude_pbcorr=False)
            if hits:
                i_pbcorr = hits[0]
                break
        # Deep V PBcorr  prefer dewarped
        v_hits = _glob_prefer_dewarped(
            os.path.join(band_dir, 'V', 'deep',
                         f"{band}_V_deep_Taper_*pbcorr*.fits"),
            exclude_pbcorr=False)
        v_pbcorr = v_hits[0] if v_hits else None

        # Deep I Robust-0 non-PBcorr (for subtraction reference)  prefer dewarped
        sub_ref = None
        sub_ref_hits = _glob_prefer_dewarped(
            os.path.join(band_dir, 'I', 'deep',
                         f"{band}_I_deep_Taper_Robust-0_*.fits"))
        # Exclude Robust-0.75
        sub_ref_hits = [f for f in sub_ref_hits
                        if 'Robust-0.75' not in os.path.basename(f)]
        if sub_ref_hits:
            sub_ref = sub_ref_hits[0]

        # 10min I non-PBcorr (for subtraction) + V PBcorr (for direct measurement)
        wb_10min_pairs = []
        # I 10min: prefer dewarped non-PBcorr Robust-0
        i_10min_hits = _glob_prefer_dewarped(
            os.path.join(band_dir, 'I', '10min',
                         f"{band}_I_10min_Taper_Robust-0_*.fits"))
        if not i_10min_hits:
            # Fall back to PBcorr
            i_10min_hits = _glob_prefer_dewarped(
                os.path.join(band_dir, 'I', '10min',
                             f"{band}_I_10min_Taper_*pbcorr*.fits"),
                exclude_pbcorr=False)
        # V 10min: PBcorr preferred, dewarped preferred
        v_10min_hits = _glob_prefer_dewarped(
            os.path.join(band_dir, 'V', '10min',
                         f"{band}_V_10min_Taper_*pbcorr*.fits"),
            exclude_pbcorr=False)
        if not v_10min_hits:
            v_10min_hits = _glob_prefer_dewarped(
                os.path.join(band_dir, 'V', '10min',
                             f"{band}_V_10min_Taper_*.fits"))
        if i_10min_hits and v_10min_hits:
            import re as _re_wb
            # Match I/V by interval tag (tNNNN) or timestamp (YYYYMMDD_HHMMSS)
            v_by_tag = {}
            for vf in v_10min_hits:
                bn = os.path.basename(vf)
                m = _re_wb.search(r'(t\d{4})', bn)
                if not m:
                    m = _re_wb.search(r'(\d{8}_\d{6})', bn)
                if m:
                    v_by_tag[m.group(1)] = vf
            for if_path in i_10min_hits:
                bn = os.path.basename(if_path)
                m = _re_wb.search(r'(t\d{4})', bn)
                if not m:
                    m = _re_wb.search(r'(\d{8}_\d{6})', bn)
                if m and m.group(1) in v_by_tag:
                    wb_10min_pairs.append(
                        (if_path, v_by_tag[m.group(1)], m.group(1)))

        if i_pbcorr and v_pbcorr:
            wb_bands.append((band, i_pbcorr, v_pbcorr, wb_10min_pairs, sub_ref))
            parts = ["deep"]
            if wb_10min_pairs:
                parts.append(f"{len(wb_10min_pairs)} 10min pairs")
            if sub_ref:
                parts.append(f"sub_ref={os.path.basename(sub_ref)}")
            logger.info(f"  Wideband {band}: {', '.join(parts)}")

    # -- Target Photometry (parallel by frequency) --
    if cutout and target_files:
        # Load all target lists once
        all_targets_by_sample = []
        for t_file in target_files:
            if not os.path.exists(t_file):
                logger.warning(f"  Target file not found: {t_file}")
                continue
            try:
                s_name = os.path.splitext(os.path.basename(t_file))[0]
                targets = cutout.load_targets(t_file)
                all_targets_by_sample.append((s_name, targets))
                logger.info(f"  Loaded {len(targets)} targets from {os.path.basename(t_file)}")
            except Exception as e:
                logger.error(f"  Failed to load {t_file}: {e}")

        if all_targets_by_sample:
            # Build task list: one task per frequency band
            tasks = []
            for sb in subband_list:
                sb_dir = os.path.join(work_dir, sb)
                if os.path.isdir(sb_dir):
                    tasks.append({
                        'band_label': sb,
                        'sb_dir': sb_dir,
                        'run_dir': work_dir,
                        'targets_by_sample': all_targets_by_sample,
                        'samples_dir': samples_dir,
                        'detections_dir': detections_dir,
                        'debug_log': debug_log_path,
                    })

            for band, i_path, v_path, wb_10min, sub_ref in wb_bands:
                band_dir = os.path.join(work_dir, band)
                tasks.append({
                    'band_label': band,
                    'sb_dir': band_dir,
                    'run_dir': band_dir,
                    'explicit_i': i_path,
                    'explicit_v': v_path,
                    'snapshots_10min': wb_10min if wb_10min else None,
                    'deep_i_sub_ref': sub_ref,
                    'targets_by_sample': all_targets_by_sample,
                    'samples_dir': samples_dir,
                    'detections_dir': detections_dir,
                    'debug_log': debug_log_path,
                })

            if not tasks:
                logger.warning("  No image bands found  nothing to process")
            else:
                n_workers = min(args.workers, len(tasks))
                logger.info(f"\n  Processing {len(tasks)} bands × "
                            f"{sum(len(t) for _, t in all_targets_by_sample)} targets "
                            f"with {n_workers} workers")

                # Run parallel
                all_results = []  # (sample_name, target_name, summary_row)
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {executor.submit(_photometry_worker, t): t['band_label']
                               for t in tasks}
                    for future in as_completed(futures):
                        band_label = futures[future]
                        try:
                            bl, results = future.result()
                            all_results.extend(results)
                            logger.info(f"  {bl}: done ({len(results)} measurements)")
                        except Exception as e:
                            logger.error(f"  {band_label} worker failed: {e}")
                            traceback.print_exc()

                # Merge per-target CSVs
                logger.info(f"\n  Merging {len(all_results)} measurements into per-target CSVs")
                from collections import defaultdict
                by_target = defaultdict(list)
                for s_name, t_name, row in all_results:
                    by_target[(s_name, t_name)].append(row)

                for (s_name, t_name), rows in by_target.items():
                    cutout.write_photometry_csv(rows, t_name, s_name,
                                                 samples_dir, detections_dir)

                logger.info(f"  Written CSVs for {len(by_target)} targets")

    # -- Solar System Photometry --
    # Compute ephemeris positions and feed through same pipeline as targets
    if cutout and subband_list:
        logger.info("\n--- Solar System Photometry ---")
        try:
            from exopipe.phase3.lib.photometry.solar_system_cutout import (
                SOLAR_SYSTEM_BODIES, get_body_position, get_obs_midpoint,
                MIN_ELEVATION
            )
            from astropy.io import fits as _ss_fits

            # Get observation time from first available deep image
            obs_time = None
            for sb in subband_list:
                deep_dir = os.path.join(work_dir, sb, "I", "deep")
                if not os.path.isdir(deep_dir):
                    continue
                for f in sorted(glob.glob(os.path.join(deep_dir, "*.fits"))):
                    try:
                        with _ss_fits.open(f) as hdul:
                            obs_time = get_obs_midpoint(hdul[0].header, fits_path=f)
                        break
                    except Exception:
                        continue
                if obs_time:
                    break

            if obs_time is None:
                logger.warning("  Cannot determine observation time for solar system")
            else:
                from exopipe.common.pipeline_config import OVRO_LOC as _ss_loc
                ss_targets = []
                for body_name in SOLAR_SYSTEM_BODIES:
                    coord, el, dist_au = get_body_position(body_name, obs_time, _ss_loc)
                    if coord is None or el < MIN_ELEVATION:
                        logger.info(f"  {body_name.capitalize()}: below horizon or failed")
                        continue
                    display_name = body_name.capitalize()
                    ss_targets.append((display_name, coord, 'IV', []))
                    logger.info(f"  {display_name}: el={el:.1f}°")

                if ss_targets:
                    ss_sample = [("SolarSystem", ss_targets)]

                    # Build tasks for subbands + wideband
                    ss_tasks = []
                    for sb in subband_list:
                        sb_dir = os.path.join(work_dir, sb)
                        if os.path.isdir(sb_dir):
                            ss_tasks.append({
                                'band_label': sb,
                                'sb_dir': sb_dir,
                                'run_dir': work_dir,
                                'targets_by_sample': ss_sample,
                                'samples_dir': samples_dir,
                                'detections_dir': detections_dir,
                                'debug_log': debug_log_path,
                            })

                    for band, i_path, v_path, wb_10min, sub_ref in wb_bands:
                        band_dir = os.path.join(work_dir, band)
                        ss_tasks.append({
                            'band_label': band,
                            'sb_dir': band_dir,
                            'run_dir': band_dir,
                            'explicit_i': i_path,
                            'explicit_v': v_path,
                            'snapshots_10min': wb_10min if wb_10min else None,
                            'deep_i_sub_ref': sub_ref,
                            'targets_by_sample': ss_sample,
                            'samples_dir': samples_dir,
                            'detections_dir': detections_dir,
                            'debug_log': debug_log_path,
                        })

                    if ss_tasks:
                        n_ss_workers = min(args.workers, len(ss_tasks))
                        logger.info(f"  Processing {len(ss_tasks)} bands × "
                                    f"{len(ss_targets)} bodies with {n_ss_workers} workers")

                        ss_results = []
                        with ProcessPoolExecutor(max_workers=n_ss_workers) as executor:
                            futures = {executor.submit(_photometry_worker, t): t['band_label']
                                       for t in ss_tasks}
                            for future in as_completed(futures):
                                band_label = futures[future]
                                try:
                                    bl, results = future.result()
                                    ss_results.extend(results)
                                    logger.info(f"  SS {bl}: done ({len(results)} measurements)")
                                except Exception as e:
                                    logger.error(f"  SS {band_label} worker failed: {e}")
                                    traceback.print_exc()

                        # Merge per-body CSVs
                        from collections import defaultdict
                        ss_by_target = defaultdict(list)
                        for s_name, t_name, row in ss_results:
                            ss_by_target[(s_name, t_name)].append(row)
                        for (s_name, t_name), rows in ss_by_target.items():
                            cutout.write_photometry_csv(rows, t_name, s_name,
                                                         samples_dir, detections_dir)
                        logger.info(f"  Written CSVs for {len(ss_by_target)} solar system bodies")
                else:
                    logger.info("  No solar system bodies above horizon")

        except Exception as e:
            logger.error(f"  Solar system photometry failed: {e}")
            traceback.print_exc()

    # -- Thermal Noise Lookup Table --
    _generate_thermal_noise_csv(work_dir, subband_list, logger)


def _generate_thermal_noise_csv(work_dir, subband_list, logger):
    """Generate thermal_noise_all.csv from all Stokes V images (deep + 10min + wideband).

    Measures MAD-based RMS from inner 1024×1024 of each V image. One row per image.
    Columns: Freq_Label, Category, Interval, Date, V_RMS_Jy, Image
    """
    from astropy.io import fits as _tn_fits
    from astropy.stats import mad_std as _tn_mad
    import numpy as np
    import pandas as pd

    logger.info("\n--- Thermal Noise Table ---")
    rows = []

    def _measure(fits_path, freq_label, category, interval=''):
        try:
            with _tn_fits.open(fits_path) as hdul:
                data = hdul[0].data.squeeze()
                h, w = data.shape
                half = min(512, h // 4, w // 4)
                ch, cw = h // 2, w // 2
                region = data[ch - half:ch + half, cw - half:cw + half]
                rms = float(_tn_mad(region, ignore_nan=True))
                # Extract timestamp
                ts = ''
                try:
                    from astropy.time import Time
                    ts = Time(hdul[0].header['DATE-OBS'], format='isot',
                              scale='utc').datetime.strftime("%Y%m%d_%H%M%S")
                except Exception:
                    pass
                rows.append({
                    'Freq_Label': freq_label, 'Category': category,
                    'Interval': interval, 'Date': ts,
                    'V_RMS_Jy': rms, 'Image': os.path.basename(fits_path)
                })
        except Exception as e:
            logger.warning(f"  Thermal noise failed for {os.path.basename(fits_path)}: {e}")

    # Per-subband (non-PBcorr V for unbiased thermal noise)
    for sb in subband_list:
        sb_dir = os.path.join(work_dir, sb)
        # Deep V: prefer dewarped non-PBcorr
        for pat in ['*V*Deep*_dewarped.fits', '*V*Deep*-image*.fits']:
            hits = [f for f in glob.glob(os.path.join(sb_dir, 'V', 'deep', pat))
                    if 'pbcorr' not in os.path.basename(f)]
            if hits:
                _measure(sorted(hits)[0], sb, 'deep')
                break
        # 10min V: prefer dewarped non-PBcorr
        for pat in ['*V*10min*_dewarped.fits', '*V*10min*-image*.fits']:
            hits = [f for f in glob.glob(os.path.join(sb_dir, 'V', '10min', pat))
                    if 'pbcorr' not in os.path.basename(f)]
            for f in sorted(hits):
                import re as _tn_re
                m = _tn_re.search(r'(t\d{4}|\d{8}_\d{6})', os.path.basename(f))
                itag = m.group(1) if m else os.path.basename(f)
                _measure(f, sb, '10min', itag)
            if hits:
                break

    # Wideband (non-PBcorr V in per-band directories)  prefer dewarped
    for band in ['Red', 'Green', 'Blue', 'Full']:
        band_dir = os.path.join(work_dir, band)
        if not os.path.isdir(band_dir):
            continue
        # Deep
        hits = _glob_prefer_dewarped(
            os.path.join(band_dir, 'V', 'deep', f"{band}_V_deep_Taper_*.fits"))
        if hits:
            _measure(hits[0], band, 'deep')
        # 10min
        hits = _glob_prefer_dewarped(
            os.path.join(band_dir, 'V', '10min', f"{band}_V_10min_Taper_*.fits"))
        for f in hits:
            import re as _tn_re
            m = _tn_re.search(r'(t\d{4}|\d{8}_\d{6})', os.path.basename(f))
            itag = m.group(1) if m else os.path.basename(f)
            _measure(f, band, '10min', itag)

    if rows:
        qa_dir = os.path.join(work_dir, "QA")
        os.makedirs(qa_dir, exist_ok=True)
        csv_path = os.path.join(qa_dir, "thermal_noise_all.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info(f"  Saved {len(rows)} measurements to {csv_path}")
    else:
        logger.warning("  No thermal noise measurements produced")


# =====================================================================
# STEP 6: TRANSIENT SEARCH (per-subband + wideband)
# =====================================================================

def run_all_transient_searches(work_dir, subband_list, catalog_path, logger,
                               max_workers=19):
    """Run transient search on all images: per-subband + wideband, parallelized.

    Two phases:
      Phase A: Batch flow alignment for all I 10-min snapshots (sequential,
               GPU, minimizes JAX cold starts)
      Phase B: Parallel V and I searches across all bands (CPU only)

    Called AFTER wideband stacking is complete, so all image products exist.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TRANSIENT SEARCH (all subbands + wideband)")
    logger.info("=" * 60)

    # Verify search scripts are available. sv2 lives in phase3/ after the
    # absorb (R-GC9, Gregg 2026-08-13): missing callee = hard refusal, never
    # a silent no-op of the Phase-2 per-epoch V search.
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v_script = os.path.join(pkg_dir, 'phase3', 'lib', 'search', 'search_v.py')
    if not os.path.exists(v_script):
        raise RuntimeError(
            f"search_v_transients_v2.py not found at {v_script} — "
            "Phase-2 V transient search REFUSED (R-GC9)")
    try:
        from exopipe.phase3.lib.search.search_i import run_search as _test_import
        del _test_import
    except ImportError:
        logger.warning("search_i_transients not importable — I search disabled")

    detections_dir = os.path.join(work_dir, "detections")
    os.makedirs(detections_dir, exist_ok=True)

    # -- Discover all bands and images --
    band_configs = []

    for subband in subband_list:
        sb_dir = os.path.join(work_dir, subband)
        if not os.path.isdir(sb_dir):
            continue
        band_configs.append({
            'band_label': subband, 'band_dir': sb_dir,
            'is_wideband': False,
        })

    for band in ['Red', 'Green', 'Blue', 'Full']:
        band_dir = os.path.join(work_dir, band)
        if not os.path.isdir(band_dir):
            continue
        band_configs.append({
            'band_label': band, 'band_dir': band_dir,
            'is_wideband': True,
        })

    if not band_configs:
        logger.warning("  No bands found for transient search")
        return

    # -- Phase A: Batch flow alignment (sequential, GPU) --
    logger.info(f"\n  Phase A: Flow alignment for I snapshots")
    n_aligned = _batch_flow_alignment(band_configs, detections_dir, logger)
    logger.info(f"  Flow alignment complete: {n_aligned} images aligned")

    # -- Phase B: Parallel searches (CPU only, no_align=True) --
    tasks = []
    for bc in band_configs:
        tasks.append({
            'band_label': bc['band_label'],
            'band_dir': bc['band_dir'],
            'catalog_path': catalog_path,
            'detections_dir': detections_dir,
            'is_wideband': bc['is_wideband'],
        })

    n_workers = min(max_workers, len(tasks))
    logger.info(f"\n  Phase B: Searching {len(tasks)} bands "
                f"with {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_transient_worker, t): t['band_label']
                   for t in tasks}
        for future in as_completed(futures):
            band_label = futures[future]
            try:
                bl, n_det = future.result()
                logger.info(f"  {bl}: done ({n_det} detections)")
            except Exception as e:
                logger.error(f"  {band_label} transient search failed: {e}")
                traceback.print_exc()


def _batch_flow_alignment(band_configs, detections_dir, logger):
    """Run all I 10-min flow alignments sequentially to share GPU/JAX cache.

    For each band, finds all 10-min I snapshots and their deep reference,
    runs align_images.py once for each pair. The aligned images are saved
    in the detections directory so the I search can find them.

    Returns number of images aligned.
    """
    import subprocess

    align_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'align_images.py')
    if not os.path.exists(align_script):
        logger.warning("  align_images.py not found  skipping alignment")
        return 0

    # Collect all (tgt, ref) pairs
    pairs = []
    for bc in band_configs:
        band_label = bc['band_label']
        band_dir = bc['band_dir']

        # Find deep I reference
        if bc['is_wideband']:
            ref_matches = _glob_prefer_dewarped(
                os.path.join(band_dir, 'I', 'deep',
                             f"{band_label}_I_deep_Taper_Robust-0_*.fits"))
            ref_matches = [f for f in ref_matches
                           if 'Robust-0.75' not in os.path.basename(f)]
        else:
            ref_matches = _find_transient_images(
                band_dir, "I", "deep", suffix_filter="Robust-0-")

        if not ref_matches:
            continue
        ref_i = ref_matches[0]

        # Find 10-min I snapshots
        if bc['is_wideband']:
            tgt_imgs = _glob_prefer_dewarped(
                os.path.join(band_dir, 'I', '10min',
                             f"{band_label}_I_10min_*.fits"))
        else:
            tgt_imgs = _find_transient_images(band_dir, "I", "10min")

        for tgt in tgt_imgs:
            pairs.append({
                'tgt': tgt, 'ref': ref_i, 'band': band_label,
            })

    if not pairs:
        return 0

    logger.info(f"  {len(pairs)} I snapshots to align across "
                f"{len(band_configs)} bands")

    n_ok = 0
    conda_env = 'py310_orca'

    for i, pair in enumerate(pairs):
        tgt_base = os.path.splitext(os.path.basename(pair['tgt']))[0]
        align_dir = os.path.join(detections_dir, f"align_{tgt_base}")
        os.makedirs(align_dir, exist_ok=True)

        # Write flux-scaled reference (simple: scale=1 for intra-session)
        # The I search will re-do flux scaling; this alignment uses raw images
        ref_scaled_path = os.path.join(
            align_dir,
            os.path.basename(pair['ref']).replace('.fits', '_scaled.fits'))

        # Only copy ref if not already there
        if not os.path.exists(ref_scaled_path):
            import shutil
            shutil.copy2(pair['ref'], ref_scaled_path)

        # Check if already aligned
        dewarp_path = os.path.join(
            align_dir, f"{tgt_base}_dewarp.fits")
        if os.path.exists(dewarp_path):
            n_ok += 1
            continue

        cmd = [
            "conda", "run", "-n", conda_env, "--no-capture-output",
            "python", align_script,
            "--target", pair['tgt'],
            "--reference", ref_scaled_path,
            "--out_dir", align_dir,
            "--no_qa",
        ]

        logger.info(f"  [{i+1}/{len(pairs)}] {pair['band']}: "
                     f"{os.path.basename(pair['tgt'])}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=600)
            if result.returncode == 0:
                n_ok += 1
            else:
                logger.warning(f"    Alignment failed (rc={result.returncode})")
        except subprocess.TimeoutExpired:
            logger.warning(f"    Alignment timed out")
        except FileNotFoundError:
            logger.warning(f"    conda not found  skipping all alignments")
            return n_ok

    return n_ok


def _transient_worker(task):
    """Worker: run transient search for one frequency band.

    Runs in a subprocess. Handles both subband and wideband bands.
    Calls search_v_transients_v2.py (subprocess) for V
    and search_i_transients.run_search() (direct) for I.
    Returns (band_label, n_detections).
    """
    band_label = task['band_label']
    band_dir = task['band_dir']
    catalog_path = task['catalog_path']
    detections_dir = task['detections_dir']
    is_wideband = task.get('is_wideband', False)

    n_det = 0

    if is_wideband:
        # Wideband: find ref I in band_dir/I/deep/  prefer dewarped
        ref_matches = _glob_prefer_dewarped(
            os.path.join(band_dir, 'I', 'deep',
                         f"{band_label}_I_deep_Taper_Robust-0_*.fits"))
        # Exclude Robust-0.75
        ref_matches = [f for f in ref_matches
                       if 'Robust-0.75' not in os.path.basename(f)]
        if not ref_matches:
            print(f"  {band_label}: no deep I reference  skipping")
            return band_label, 0
        ref_i_path = ref_matches[0]
        print(f"  {band_label}: ref = {os.path.basename(ref_i_path)}")

        # Stokes V: direct search  prefer dewarped
        for cat in ['deep', '10min']:
            v_dir = os.path.join(band_dir, 'V', cat)
            if not os.path.isdir(v_dir):
                continue
            v_imgs = _glob_prefer_dewarped(
                os.path.join(v_dir, f"{band_label}_V_{cat}_*.fits"))
            # Find matching I image for each V image
            i_dir = os.path.join(band_dir, 'I', cat)
            for v_img in v_imgs:
                i_img = _find_matching_i(v_img, i_dir, band_label, cat)
                if i_img is None:
                    i_img = ref_i_path  # fallback to deep ref
                try:
                    nd = _run_v_search(v_img, i_img,
                                       catalog_path=catalog_path,
                                       out_dir=detections_dir)
                    n_det += nd
                except Exception as e:
                    print(f"  {band_label} V failed on "
                          f"{os.path.basename(v_img)}: {e}")

        # Stokes I: subtract deep from 10min
        i_imgs = _glob_prefer_dewarped(
            os.path.join(band_dir, 'I', '10min',
                          f"{band_label}_I_10min_*.fits"))
        for i_img in i_imgs:
            try:
                nd = _run_i_search(i_img, ref_i_path,
                                   catalog_path=catalog_path,
                                   out_dir=detections_dir)
                n_det += nd
            except Exception as e:
                print(f"  {band_label} I failed on "
                      f"{os.path.basename(i_img)}: {e}")

    else:
        # Subband
        ref_i_imgs = _find_transient_images(band_dir, "I", "deep",
                                             suffix_filter="Robust-0-")
        if not ref_i_imgs:
            print(f"  {band_label}: no deep I reference  skipping")
            return band_label, 0
        ref_i_path = ref_i_imgs[0]
        print(f"  {band_label}: ref = {os.path.basename(ref_i_path)}")

        # Stokes V: direct search
        v_deep = _find_transient_images(band_dir, "V", "deep")
        v_10min = _find_transient_images(band_dir, "V", "10min")
        for v_img in v_deep + v_10min:
            # Find matching I
            cat = "deep" if v_img in v_deep else "10min"
            i_dir = os.path.join(band_dir, 'I', cat)
            i_img = _find_matching_i(v_img, i_dir, band_label, cat)
            if i_img is None:
                i_img = ref_i_path
            try:
                nd = _run_v_search(v_img, i_img,
                                   catalog_path=catalog_path,
                                   out_dir=detections_dir)
                n_det += nd
            except Exception as e:
                print(f"  {band_label} V failed on "
                      f"{os.path.basename(v_img)}: {e}")

        # Stokes I: subtract deep from 10min
        i_snaps = _find_transient_images(band_dir, "I", "10min")
        for i_img in i_snaps:
            try:
                nd = _run_i_search(i_img, ref_i_path,
                                   catalog_path=catalog_path,
                                   out_dir=detections_dir)
                n_det += nd
            except Exception as e:
                print(f"  {band_label} I failed on "
                      f"{os.path.basename(i_img)}: {e}")

    return band_label, n_det


def _run_v_search(v_image, i_image, catalog_path=None, out_dir=None,
                  sigma=5.0):
    """Run Stokes V transient search via subprocess."""
    # sv2 lives in phase3/ after the absorb (R-GC9, Gregg 2026-08-13):
    # missing callee = hard refusal, never a silent 0-detection return.
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(pkg_dir, 'phase3', 'lib', 'search', 'search_v.py')
    if not os.path.exists(script):
        raise RuntimeError(
            f"search_v_transients_v2.py not found at {script} — "
            "Phase-2 V transient search REFUSED (R-GC9)")

    if out_dir is None:
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(v_image))),
            "detections")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, script,
        '--v_image', v_image,
        '--i_image', i_image,
        '--out_dir', out_dir,
        '--sigma', str(sigma),
    ]
    if catalog_path:
        cmd.extend(['--catalog', catalog_path])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=900)
        if result.returncode != 0:
            print(f"  [V] FAILED (rc={result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split('\n')[-5:]:
                    print(f"  [V-err] {line}")
            return 0

        n_pass = 0
        for line in result.stdout.split('\n'):
            if 'PASSED:' in line or 'Total PASS:' in line:
                try:
                    n_pass = int(line.strip().split()[-1])
                except (ValueError, IndexError):
                    pass
        return n_pass

    except subprocess.TimeoutExpired:
        print(f"  [V] TIMEOUT (900s)")
        return 0
    except Exception as e:
        print(f"  [V] ERROR: {e}")
        return 0


def _run_i_search(tgt_image, ref_image, catalog_path=None, out_dir=None,
                  sigma=7.0):
    """Run Stokes I transient search via direct import."""
    try:
        from exopipe.phase3.lib.search.search_i import run_search
    except ImportError:
        print(f"  WARNING: search_i_transients not importable")
        return 0

    if out_dir is None:
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(tgt_image))),
            "detections")

    try:
        passed = run_search(
            tgt_image, ref_image,
            catalog_path=catalog_path,
            out_dir=out_dir,
            sigma=sigma,
        )
        return len(passed)
    except Exception as e:
        print(f"  [I] FAILED: {e}")
        return 0


def _find_matching_i(v_image_path, i_dir, band_label, category):
    """Find the Stokes I image matching a V image (same timestamp)."""
    import re
    v_bn = os.path.basename(v_image_path)
    # Extract timestamp from V filename
    m = re.search(r'(\d{8}_\d{6})', v_bn)
    if not m:
        # Try interval label (t0001, etc.)
        m = re.search(r'(t\d{4})', v_bn)
    if m:
        ts = m.group(1)
        candidates = glob.glob(os.path.join(i_dir, f"*{ts}*Taper*-image*.fits"))
        candidates = [f for f in candidates
                      if 'pbcorr' not in f and 'NoTaper' not in f]
        # Prefer dewarped
        dewarped = [f for f in candidates if '_dewarped' in f]
        if dewarped:
            return sorted(dewarped)[0]
        if candidates:
            return sorted(candidates)[0]
    return None


def _find_transient_images(base_dir, pol, category, suffix_filter=None):
    """Find tapered, dewarped, non-PB-corrected images for transient search.

    Only matches *-image* files to exclude wsclean auxiliary products
    (dirty, model, psf, residual) that may have been dewarped.
    """
    # Prefer dewarped image products
    pat = os.path.join(base_dir, pol, category, "*Taper*-image*_dewarped.fits")
    imgs = [f for f in glob.glob(pat)
            if "pbcorr" not in f and "_dewarped_dewarped" not in f]
    if not imgs:
        # Fallback: raw image products
        pat = os.path.join(base_dir, pol, category, "*Taper*-image*.fits")
        imgs = [f for f in glob.glob(pat)
                if "pbcorr" not in f and "dewarped" not in f]

    if suffix_filter:
        filtered = [f for f in imgs
                    if suffix_filter in os.path.basename(f)
                    and "NoTaper" not in os.path.basename(f)]
        if filtered:
            imgs = filtered

    return imgs




# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OVRO-LWA Phase 2 Science Pipeline")

    parser.add_argument("--lst", required=True,
                        help="LST label (e.g. '10h')")
    parser.add_argument("--date", required=True,
                        help="Observation date (YYYY-MM-DD)")
    parser.add_argument("--run_label",
                        help="Run label (default: Science_YYYYMMDD_HHMMSS). "
                             "Required with --skip_copy to identify existing run.")
    parser.add_argument("--targets", nargs='+',
                        help="Target CSV files for photometry")
    parser.add_argument("--catalog",
                        help="Source catalog for transient cross-matching")
    parser.add_argument("--skip_copy", action="store_true",
                        help="Skip copy step (data already on NVMe)")
    parser.add_argument("--skip_dewarp", action="store_true",
                        help="Skip dewarping (already done)")
    parser.add_argument("--dewarp_anchor_only", action="store_true",
                        help="Run only the 73 MHz anchor step, skip cascade "
                             "and 10-min dewarping (useful for testing)")
    parser.add_argument("--allow_incomplete", action="store_true",
                        help="Proceed even if some expected subbands are missing "
                             "from Phase 1 (skips the wait; default: wait then "
                             "abort). The 73 MHz anchor is always required.")
    parser.add_argument("--phase1_wait_max", type=int, default=7200,
                        help="Max seconds to WAIT for Phase 1 to finish imaging "
                             "all expected subbands before giving up (default "
                             "7200=2h; 0 = don't wait, abort immediately). Polls "
                             "every --phase1_poll seconds.")
    parser.add_argument("--phase1_poll", type=int, default=120,
                        help="Seconds between Phase-1 completeness re-checks "
                             "while waiting (default 120).")
    parser.add_argument("--skip_wideband", action="store_true",
                        help="Skip wideband product formation (stacking, 3-color)")
    parser.add_argument("--skip_photometry", action="store_true",
                        help="Skip all photometry (target + solar system)")
    parser.add_argument("--skip_transients", action="store_true",
                        help="Skip transient search")
    parser.add_argument("--debug", action="store_true",
                        help="Write flux measurement debug log (flux_debug.log)")
    parser.add_argument("--workers", type=int, default=19,
                        help="Max parallel workers for photometry (default: 19)")

    # Dewarping method selection
    parser.add_argument("--anchor_method",
                        choices=['rbf', 'zernike', 'flow'],
                        default=cfg.ANCHOR_METHOD,
                        help="Dewarping method for 73 MHz anchor vs VLSS catalog "
                             f"(default: {cfg.ANCHOR_METHOD})")
    parser.add_argument("--cascade_method",
                        choices=['rbf', 'zernike', 'flow'],
                        default=cfg.CASCADE_METHOD,
                        help="Dewarping method for subband-to-subband cascade "
                             "and 10-min intervals "
                             f"(default: {cfg.CASCADE_METHOD})")
    parser.add_argument("--rbf_kernel",
                        choices=['thin_plate_spline', 'multiquadric',
                                 'inverse_multiquadric', 'gaussian', 'cubic'],
                        default=cfg.WARP_RBF_KERNEL,
                        help="RBF interpolation kernel "
                             f"(default: {cfg.WARP_RBF_KERNEL})")
    parser.add_argument("--rbf_smoothing", type=float,
                        default=cfg.WARP_RBF_SMOOTHING,
                        help="RBF smoothing parameter "
                             f"(default: {cfg.WARP_RBF_SMOOTHING})")
    parser.add_argument("--rbf_epsilon", type=float,
                        default=cfg.WARP_RBF_EPSILON,
                        help="RBF shape parameter (required for multiquadric, "
                             "inverse_multiquadric, gaussian kernels; "
                             "default: auto from anchor spacing)")
    parser.add_argument("--rbf_snr_weighting", action='store_true',
                        default=cfg.WARP_RBF_SNR_WEIGHTING,
                        help="Per-anchor smoothing weighted by 1/flux²  "
                             "bright sources constrained tightly, faint "
                             "sources relaxed (default: off)")

    parser.add_argument("--skip_header_correction", action='store_true',
                        help="Skip header validation and correction step")

    args = parser.parse_args()

    # Generate or validate run label
    if args.run_label:
        run_label = args.run_label
    else:
        run_label = f"Science_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    if args.skip_copy and not args.run_label:
        base = os.path.join(cfg.PHASE2_WORKING_DIR, args.lst, args.date)
        if os.path.isdir(base):
            candidates = sorted([
                d for d in os.listdir(base)
                if d.startswith('Science_') and os.path.isdir(os.path.join(base, d))
            ])
            if candidates:
                run_label = candidates[-1]
                logger.info(f"Auto-detected latest run: {run_label}")
            else:
                logger.error("--skip_copy requires --run_label or an existing "
                             "Science_* directory")
                sys.exit(1)

    if args.targets:
        args.targets = [os.path.abspath(t) for t in args.targets]
    if args.catalog:
        args.catalog = os.path.abspath(args.catalog)

    work_dir = os.path.join(cfg.PHASE2_WORKING_DIR, args.lst, args.date, run_label)

    os.makedirs(work_dir, exist_ok=True)
    log_file = os.path.join(
        work_dir,
        f"phase2_{args.lst}_{args.date}_{run_label}.log"
    )
    try:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - [PHASE2] - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    except Exception:
        pass

    logger.info("=" * 60)
    logger.info("OVRO-LWA PHASE 2 SCIENCE PIPELINE")
    logger.info("=" * 60)
    run_command = ' '.join(sys.argv)
    t0_pipeline = time.time()
    logger.info(f"Command: {run_command}")
    logger.info(f"Run Label: {run_label}")
    logger.info(f"LST: {args.lst}")
    logger.info(f"Date: {args.date}")
    logger.info(f"Work Dir: {work_dir}")
    logger.info(f"Targets: {args.targets}")
    logger.info(f"Catalog: {args.catalog}")
    logger.info(f"Log: {log_file}")

    # -- STEP 1: Discovery --
    subband_map = discover_subbands(args.lst, args.date)
    # -- STEP 1a: EXCLUSIONS (Gregg 2026-09-05: 'the policy is to not form a wideband
    # product with an excluded subband. Maintain that policy.'). Subbands excluded in
    # data_decisions for this night, or whose Stokes I Taper Robust-0.75 DIRTY image is
    # corrupt (central rms > 5x the healthy baseline; R-REEXCLUDE-81-DIRTY-CORRUPT-
    # APPROVED), are removed from the subband set BEFORE dewarp and stacking: they are
    # never a cascade member or reference, and every wideband band containing one is
    # NOT formed (Full is not formed when any band is not formed). Recorded in
    # {work_dir}/excluded_subbands.csv. Phase-1-incomplete gaps keep the wait/refuse
    # logic below; exclusion gaps do not need --allow_incomplete.
    from exopipe.common.exclusions import excluded_subbands_for_night, dirty_gated_subbands
    _excl_dd = excluded_subbands_for_night(args.lst, args.date, list(subband_map))
    _excl_dirty = dirty_gated_subbands(args.lst, subband_map)
    _excluded = {sb: 'data_decisions exclude' for sb in _excl_dd}
    for sb, (rms, b) in _excl_dirty.items():
        _excluded.setdefault(sb, f'dirty-rms gate: {rms:.0f} mJy > 5 x baseline {b:.0f} mJy')
    if _excluded:
        _BSUB0 = {'Red': {'18MHz', '23MHz', '27MHz', '32MHz', '36MHz'},
                  'Green': {'41MHz', '46MHz', '50MHz', '55MHz', '59MHz'},
                  'Blue': {'64MHz', '69MHz', '73MHz', '78MHz', '82MHz'}}
        _skip_x = {b for b, subs in _BSUB0.items() if subs & set(_excluded)}
        if _skip_x:
            _skip_x.add('Full')
        args._skip_wideband_excluded = _skip_x
        args._excluded_subbands = set(_excluded)
        for sb in sorted(_excluded, key=lambda x: int(x.replace('MHz', ''))):
            logger.warning(f"EXCLUDED subband {sb}: {_excluded[sb]} -- removed from the "
                           f"subband set (never dewarped, stacked or used as a reference)")
            subband_map.pop(sb, None)
        logger.warning(f"Wideband bands NOT FORMED (excluded constituent): {sorted(_skip_x)}")
        os.makedirs(work_dir, exist_ok=True)
        with open(os.path.join(work_dir, 'excluded_subbands.csv'), 'w') as _fh:
            _fh.write('subband,reason\n')
            for sb in sorted(_excluded):
                _fh.write(f'{sb},{_excluded[sb]}\n')
    else:
        args._skip_wideband_excluded = set()
        args._excluded_subbands = set()

    # -- STEP 1b: Phase-1 completeness gate (auto-wait) --
    # Discovery snapshots whatever Phase 1 has imaged so far. Launching Phase 2
    # before Phase 1 finishes silently drops subbands; if the 73 MHz anchor is
    # among the missing ones the whole cascade aborts (observed 2026-06: 22h
    # Aug-2025 dates ran mid-Phase-1-imaging -> 10/15 subbands, anchor absent).
    # Instead of failing, WAIT (poll) for Phase 1 to finish, up to
    # --phase1_wait_max; only abort if still incomplete after the timeout.
    if not args.dewarp_anchor_only:
        expected = [sb for sb in cfg.ALL_SUBBANDS if sb not in cfg.SKIP_SUBBANDS
                    and sb not in getattr(args, '_excluded_subbands', set())]
        missing = [sb for sb in expected if sb not in subband_map]
        if missing and not args.allow_incomplete and args.phase1_wait_max > 0:
            t0 = time.time()
            logger.info(f"Subband set incomplete ({len(subband_map)}/{len(expected)}; "
                        f"missing {missing}). Waiting up to {args.phase1_wait_max/60:.0f} "
                        f"min for Phase 1 (poll {args.phase1_poll}s)...")
            while missing and (time.time() - t0) < args.phase1_wait_max:
                time.sleep(args.phase1_poll)
                subband_map = discover_subbands(args.lst, args.date, quiet=True)
                missing = [sb for sb in expected if sb not in subband_map]
                logger.info(f"  ...Phase-1 check: {len(subband_map)}/{len(expected)} "
                            f"present ({(time.time()-t0)/60:.0f} min elapsed)"
                            + ("" if missing else "  -> complete!"))
            if not missing:
                logger.info(f"Phase 1 complete after "
                            f"{(time.time()-t0)/60:.0f} min wait; proceeding.")
        # Final decision after any wait
        missing = [sb for sb in expected if sb not in subband_map]
        if missing:
            from exopipe.phase2.cascade_dewarp import ANCHOR_FALLBACK
            anchor_avail = any(sb in subband_map for sb in ANCHOR_FALLBACK)
            msg = (f"Incomplete subband set after wait: {len(subband_map)}/"
                   f"{len(expected)} present; missing {missing}.")
            if not anchor_avail:
                logger.error(msg + f"  No anchor candidate {ANCHOR_FALLBACK} "
                             f"present -- cascade cannot run.  Re-run once Phase 1 "
                             f"is complete.")
                sys.exit(1)
            if not args.allow_incomplete:
                logger.error(msg + "  Pass --allow_incomplete to proceed with a "
                             f"partial set (a fallback anchor is available).  "
                             f"Phase 1 did not finish imaging {args.lst} {args.date} "
                             f"within the wait window; re-run once complete.")
                sys.exit(1)
            logger.warning(msg + "  Proceeding (--allow_incomplete); a fallback "
                           "anchor will be used if 73 MHz is missing.")
            # RULING (Gregg 2026-07-09, per-band): a wideband product is
            # only formed when ALL of its constituent subbands exist. Compute
            # the incomplete bands (Full requires all three) — subbands
            # always process.
            _BSUB = {'Red': {'18MHz', '23MHz', '27MHz', '32MHz', '36MHz'},
                     'Green': {'41MHz', '46MHz', '50MHz', '55MHz', '59MHz'},
                     'Blue': {'64MHz', '69MHz', '73MHz', '78MHz', '82MHz'}}
            _skip = {b for b, subs in _BSUB.items()
                     if subs & set(missing)}
            if _skip:
                _skip.add('Full')
            args._skip_wideband_incomplete = _skip
            logger.warning(f"Wideband bands to SKIP (incomplete): "
                           f"{sorted(_skip)}; complete bands still form.")

    # If anchor-only, restrict to the (first available) fallback anchor subband
    if args.dewarp_anchor_only:
        from exopipe.phase2.cascade_dewarp import ANCHOR_FALLBACK
        anchor_sb = next((sb for sb in ANCHOR_FALLBACK if sb in subband_map), None)
        if anchor_sb:
            subband_map = {anchor_sb: subband_map[anchor_sb]}
            logger.info(f"\n--dewarp_anchor_only: restricting to {anchor_sb}")
        else:
            logger.error(f"No anchor candidate {ANCHOR_FALLBACK} found on Lustre")
            sys.exit(1)

    all_warnings = []
    for sb, sb_dir in subband_map.items():
        all_warnings.extend(validate_subband(sb_dir, sb, logger))
    if all_warnings:
        logger.warning(f"\nValidation warnings ({len(all_warnings)}):")
        for w in all_warnings:
            logger.warning(f"  {w}")

    # -- STEP 2: Copy to NVMe --
    if args.skip_copy:
        if not os.path.isdir(work_dir):
            logger.error(f"Work directory not found: {work_dir}")
            sys.exit(1)
        logger.info(f"\nSkipping copy  using existing: {work_dir}")
    else:
        copy_to_nvme(subband_map, work_dir)

    # EXCLUDED subbands (STEP 1a) must not exist as directories in the work dir: the
    # cascade and the stacker discover subbands from the work dir, not from subband_map
    # (2026-09-05: attempt 4 still tried the excluded 73 MHz as the anchor). Renamed
    # <sb> -> <sb>.excluded so nothing globs them; recorded in excluded_subbands.csv.
    for _sb in sorted(getattr(args, '_excluded_subbands', set()) or set()):
        _d = os.path.join(work_dir, _sb)
        if os.path.isdir(_d):
            os.rename(_d, _d + '.excluded')
            logger.warning(f"EXCLUDED subband {_sb}: work-dir directory renamed to {_sb}.excluded")
    subband_list = sorted(
        [sb for sb in subband_map.keys()
         if os.path.isdir(os.path.join(work_dir, sb))],
        key=lambda x: int(x.replace('MHz', ''))
    )
    logger.info(f"\nSubbands on NVMe: {subband_list}")

    # -- STEP 3: Cascade Dewarping --
    # -- STEP 2b: Header Validation and Correction --
    # Runs after all files are on NVMe
    if not args.skip_header_correction:
        from exopipe.common.validate_fits_headers import correct_interval_headers
        # subband_map gives the Phase 1 dirs: exact pristine start times
        # and provenance integration counts.
        correct_interval_headers(work_dir, logger,
                                 subband_src_map=subband_map)
    else:
        logger.info("\nSkipping header correction")

    if args.skip_dewarp:
        logger.info("\nSkipping dewarping")
    else:
        from exopipe.phase2.cascade_dewarp import run_cascade_dewarping
        dewarp_results = run_cascade_dewarping(
            work_dir, logger,
            anchor_method=args.anchor_method,
            cascade_method=args.cascade_method,
            rbf_kernel=args.rbf_kernel,
            rbf_smoothing=args.rbf_smoothing,
            rbf_epsilon=args.rbf_epsilon,
            snr_weighting=args.rbf_snr_weighting,
            anchor_only=args.dewarp_anchor_only,
        )

    # --dewarp_anchor_only implies skip everything after dewarping
    if args.dewarp_anchor_only:
        logger.info("\n--dewarp_anchor_only: skipping wideband, photometry, "
                    "transients")
        return

    # -- STEP 4: Flux Scale Check (always runs) --
    from exopipe.phase2 import phase2_science_agg
    from exopipe.phase2.phase2_science_agg import run_flux_check
    phase2_science_agg.logger = logger
    run_flux_check(work_dir)

    # -- STEP 5: Wideband Product Formation --
    if args.skip_wideband:
        logger.info("\nSkipping wideband product formation")
    else:
        run_wideband_formation(work_dir, logger, skip_wideband=(set(getattr(args, '_skip_wideband_incomplete', None) or set())
                                                                 | set(getattr(args, '_skip_wideband_excluded', None) or set())) or None)

    # -- STEP 6: Photometry (source-grouped) --
    if args.skip_photometry:
        logger.info("\nSkipping photometry")
    else:
        run_all_photometry(work_dir, subband_list, args, logger)

    # -- STEP 7: Transient Search --
    if args.skip_transients:
        logger.info("\nSkipping transient search")
    else:
        run_all_transient_searches(work_dir, subband_list, args.catalog, logger,
                                   max_workers=args.workers)

    # -- STEP 8: Gather Detections + Email --
    any_science = (not args.skip_wideband
                   or not args.skip_photometry
                   or not args.skip_transients)
    if any_science:
        from exopipe.phase2.phase2_science_agg import gather_detections, send_email_report
        from exopipe.phase2 import phase2_science_agg
        phase2_science_agg.logger = logger
        gather_detections(work_dir)
        send_email_report(work_dir,
                          run_command=run_command,
                          run_duration=time.time() - t0_pipeline,
                          lst=args.lst, date=args.date)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 PIPELINE COMPLETE")
    logger.info(f"Products: {work_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
