#!/usr/bin/env python3
"""
process_subband.py  OVRO-LWA Pipeline Worker
Handles all three stages of per-subband processing:
  SETUP ? PEEL ? IMAGE (+ Science + Archive)

All compute-heavy work happens on NVMe (/fast/gh/main/).
Only final archiving copies results to Lustre.

MERGED: Old stable base + new improvements (run_casa_task, etc.)
"""
import argparse
import os
import sys
import socket
import shutil
import logging
import glob
import subprocess
import pickle
import traceback
import numpy as np
from datetime import datetime
from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS

import pipeline_config as config
import pipeline_utils as utils
from casacore.tables import table

# --- OPTIONAL HEAVY IMPORTS (guarded) ---
try:
    import hot_baseline_worker
except ImportError:
    hot_baseline_worker = None

try:
    from extractor_pb_75 import (
        generate_warp_screens,
        apply_warp,
        load_catalog as load_ref_catalog
    )
except ImportError:
    generate_warp_screens = None
    apply_warp = None
    load_ref_catalog = None

try:
    import cutout
except ImportError:
    cutout = None

try:
    import solar_system_cutout
except ImportError:
    solar_system_cutout = None

try:
    import transient_search as trans_tools
except ImportError:
    trans_tools = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
except ImportError:
    animation = None

os.environ['OPENBLAS_NUM_THREADS'] = '1'

try:
    from pb_correct import apply_pb_correction
except ImportError:
    apply_pb_correction = None

# --- LOGGING ---
# Use logger name = SLURM node for easy identification in logs
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(os.environ.get('SLURMD_NODENAME', socket.gethostname()))


# =============================================================================
# HELPERS
# =============================================================================

def save_state(work_dir, args):
    """Persist argparse Namespace to disk so later stages can recover it."""
    try:
        with open(os.path.join(work_dir, "pipeline_args.pkl"), "wb") as f:
            pickle.dump(args, f)
    except Exception as e:
        logger.warning(f"Failed to save state: {e}")


def load_state(work_dir):
    """Load persisted argparse Namespace from a previous stage."""
    try:
        with open(os.path.join(work_dir, "pipeline_args.pkl"), "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def run_subprocess(cmd, description):
    """Wrapper that logs and raises on failure."""
    logger.info(f"START: {description}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"DONE: {description}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FAILED: {description} (Exit Code: {e.returncode})")
        raise


# =============================================================================
# HELPER: MOVIE GENERATION (runs on NVMe)
# =============================================================================

def generate_local_movies(work_dir, freq_str, snap_source_dir=None):
    """Generate raw + filtered MP4 movies from pilot snapshots.
    
    Args:
        work_dir: Output directory for movies
        freq_str: Frequency label (e.g., '55MHz')
        snap_source_dir: Directory containing snapshots/ (defaults to work_dir)
    """
    if animation is None:
        logger.warning("matplotlib.animation not available  skipping movie generation.")
        return

    logger.info("Generating Movies (Local NVMe)...")
    movie_dir = os.path.join(work_dir, "Movies")
    os.makedirs(movie_dir, exist_ok=True)
    source = snap_source_dir or work_dir
    snap_dir = os.path.join(source, "snapshots")

    for pol in ['I', 'V']:
        files = sorted(glob.glob(os.path.join(snap_dir, f"*{pol}-image*.fits")))
        if len(files) < 10:
            continue

        frames = []
        for f in files[:150]:
            try:
                frames.append(fits.getdata(f).squeeze())
            except Exception:
                pass

        if not frames:
            continue

        try:
            cube = np.array(frames)
            mid = len(cube) // 2

            # 1. Raw Movie (both I and V)  grayscale
            if pol == 'V':
                rms = np.nanstd(cube[mid])
                vmin, vmax = -5 * rms, 5 * rms
            else:
                vmin, vmax = np.nanpercentile(cube[mid], [1, 99.5])

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            ims = [[plt.imshow(fr, animated=True, origin='lower', cmap='gray',
                               vmin=vmin, vmax=vmax)] for fr in cube]
            ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
            ani.save(os.path.join(movie_dir, f"{freq_str}_{pol}_Raw.mp4"),
                     writer='ffmpeg', dpi=150)
            plt.close(fig)

            # 2. Filtered Movie (Stokes I only  median subtraction)
            if pol == 'I':
                med = np.median(cube, axis=0)
                diff = cube - med
                rms = np.std(diff)
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis('off')
                ims = [[plt.imshow(fr, animated=True, origin='lower', cmap='gray',
                                   vmin=-3 * rms, vmax=5 * rms)] for fr in diff]
                ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
                ani.save(os.path.join(movie_dir, f"{freq_str}_{pol}_Filtered.mp4"),
                         writer='ffmpeg', dpi=150)
                plt.close(fig)

        except Exception as e:
            logger.error(f"Movie gen failed for {pol}: {e}")


# =============================================================================
# STAGE 1: SETUP (NVMe)
# =============================================================================

def run_stage_setup(args):
    """
    Stage 1: Copy raw data to NVMe, flag, calibrate, then submit Peel + Image.
    All work happens on /fast/ (NVMe).
    """
    logger.info(f"--- STAGE 1: SETUP ({args.subband} | {args.run_label}) ---")

    # Skip 13 MHz subband (known to be unusable)
    if "13MHz" in args.subband or "13mhz" in args.subband.lower():
        logger.warning("13 MHz subband detected. Skipping.")
        sys.exit(0)

    # Parse observation date
    try:
        date_str = args.start_time.replace('T', ':').split('.')[0]
        obs_date_str = datetime.strptime(date_str, '%Y-%m-%d:%H:%M:%S').strftime('%Y-%m-%d')
    except Exception:
        obs_date_str = "unknown"

    # Create work directory on NVMe
    work_dir = f"/fast/gh/main/{args.lst_label}/{obs_date_str}/{args.run_label}/{args.subband}"
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, "QA"), exist_ok=True)
    os.chdir(work_dir)
    utils.redirect_casa_log(work_dir)

    # 1. Find & copy archive files to NVMe
    archive_files = utils.find_archive_files_for_subband(
        utils.valid_datetime(args.start_time),
        utils.valid_datetime(args.end_time),
        args.subband, logger,
        input_dir=getattr(args, 'input_dir', None)
    )
    if not archive_files:
        logger.error("No archive files found for this subband/time range.")
        sys.exit(1)

    ms_files = utils.copy_and_prepare_files(archive_files, work_dir, args.subband, logger)

    # 2. Flag bad antennas (via MNC health data in 'development' conda env)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for ms in ms_files:
        utils.run_antenna_flagging(ms, {}, script_dir, logger)

    # 3. Apply bandpass + XY-phase calibration
    valid_ms_files = []
    for ms in ms_files:
        if utils.apply_calibration(ms, args.bp_table, args.xy_table, logger):
            valid_ms_files.append(ms)
        else:
            logger.warning(f"Calibration failed for {os.path.basename(ms)}  removing.")
            shutil.rmtree(ms)

    if not valid_ms_files:
        logger.error("All files failed calibration. Aborting.")
        sys.exit(1)

    # 4. Save state & peel list
    save_state(work_dir, args)
    with open("peel_list.txt", "w") as f:
        for ms in valid_ms_files:
            f.write(f"{ms}\n")

    # 5. Submit Peel stage as Slurm array job (same node)
    current_node = os.environ.get('SLURMD_NODENAME', 'localhost')
    log_dir = os.path.join(
        config.LUSTRE_ARCHIVE_DIR, args.lst_label, obs_date_str,
        args.run_label, args.subband, "logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    peel_log = os.path.join(log_dir, "Peel_%a.log")

    cmd_peel = [
        'sbatch',
        f'--job-name=Peel_{args.subband}',
        '--partition=general',
        f'--nodelist={current_node}',
        f'--array=0-{len(valid_ms_files) - 1}%4',
        '--cpus-per-task=8', '--mem=90G',
        f'--output={peel_log}', f'--error={peel_log}',
        f'--chdir={script_dir}',
        '--wrap',
        (
            f"source ~/.bashrc && conda activate {config.REQUIRED_CONDA_ENV} && "
            f"export PYTHONPATH={script_dir}:$PYTHONPATH && "
            f"python -u {os.path.abspath(__file__)} --mode peel --work_dir {work_dir}"
        )
    ]
    res = subprocess.run(cmd_peel, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"Peel array job submission failed: {res.stderr}")
        sys.exit(1)

    peel_job_id = res.stdout.strip().split()[-1]
    logger.info(f"Peel array job submitted: {peel_job_id}")

    # 6. Submit Image stage with dependency on peel completion
    img_log = os.path.join(log_dir, "Image.log")

    img_cpus, img_mem, _ = config.get_image_resources(args.subband)
    logger.info(f"Image resources for {args.subband} on {current_node}: {img_cpus} CPUs, {img_mem}G mem")

    cmd_img = [
        'sbatch',
        f'--job-name=Image_{args.subband}',
        '--partition=general',
        f'--nodelist={current_node}',
        f'--cpus-per-task={img_cpus}', f'--mem={img_mem}G',
        f'--dependency=afterok:{peel_job_id}',
        f'--output={img_log}', f'--error={img_log}',
        '--wrap',
        (
            f"source ~/.bashrc && conda activate {config.REQUIRED_CONDA_ENV} && "
            f"export PYTHONPATH={script_dir}:$PYTHONPATH && "
            f"python -u {os.path.abspath(__file__)} --mode image --work_dir {work_dir}"
        )
    ]
    res_img = subprocess.run(cmd_img, capture_output=True, text=True)
    if res_img.returncode != 0:
        logger.error(f"Image job submission failed: {res_img.stderr}")
        sys.exit(1)

    image_job_id = res_img.stdout.strip().split()[-1]
    logger.info(f"Image job queued ({image_job_id}) with dependency on Peel {peel_job_id}.")

    # Write Image job ID to shared dependency file so Science_Agg can wait on it
    dep_file = os.path.join(
        config.LUSTRE_ARCHIVE_DIR, args.lst_label,
        obs_date_str, args.run_label, ".image_job_ids"
    )
    try:
        os.makedirs(os.path.dirname(dep_file), exist_ok=True)
        with open(dep_file, "a") as f:
            f.write(f"{image_job_id}\n")
        logger.info(f"Registered Image job {image_job_id} in {dep_file}")
    except Exception as e:
        logger.warning(f"Could not write dependency file: {e}")


# =============================================================================
# STAGE 2: PEEL (NVMe)
# =============================================================================

def run_stage_peel(args):
    """
    Stage 2: Subtract bright sources (sky + RFI) from individual MS files.
    Runs as a Slurm array job  each task handles one MS file on NVMe.
    TTCal requires its own conda environment (Julia-based).
    """
    try:
        org_args = load_state(args.work_dir)
        os.chdir(args.work_dir)
        utils.redirect_casa_log(args.work_dir)

        if 'args' not in config.PEELING_PARAMS:
            logger.error("FATAL: 'args' key missing in PEELING_PARAMS! Check pipeline_config.py.")
            sys.exit(1)

        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
        with open("peel_list.txt", "r") as f:
            files = f.read().splitlines()

        if task_id >= len(files):
            logger.warning(f"Task ID {task_id} exceeds file list ({len(files)}). Exiting.")
            sys.exit(0)

        ms = files[task_id]
        logger.info(f"--- STAGE 2: PEEL (Task {task_id})  {os.path.basename(ms)} ---")

        peel_env = os.environ.copy()
        peel_env["OMP_NUM_THREADS"] = "8"

        if org_args and org_args.peel_sky:
            cmd = (
                f"source ~/.bashrc && conda activate {config.PEELING_PARAMS['sky_env']} && "
                f"ttcal.jl zest {ms} {config.PEELING_PARAMS['sky_model']} "
                f"{config.PEELING_PARAMS['args']}"
            )
            logger.info(f"START: Peel Sky Sources")
            subprocess.run(cmd, shell=True, check=True, executable='/bin/bash', env=peel_env)
            logger.info(f"DONE: Peel Sky Sources")

        if org_args and org_args.peel_rfi:
            cmd = (
                f"source ~/.bashrc && conda activate {config.PEELING_PARAMS['rfi_env']} && "
                f"ttcal.jl zest {ms} {config.PEELING_PARAMS['rfi_model']} "
                f"{config.PEELING_PARAMS['args']}"
            )
            logger.info(f"START: Peel RFI Sources")
            subprocess.run(cmd, shell=True, check=True, executable='/bin/bash', env=peel_env)
            logger.info(f"DONE: Peel RFI Sources")

    except Exception as e:
        logger.error(f"Peel crashed on task {os.environ.get('SLURM_ARRAY_TASK_ID', '?')}: {e}")
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# STAGE 3: IMAGE + SCIENCE + ARCHIVE
# =============================================================================

def run_stage_image(args):
    """
    Stage 3: Concatenate, QA, image, dewarp, science analysis, and archive.
    All compute on NVMe (/fast/). Final archive to Lustre at the very end.
    """
    logger.info(f"--- STAGE 3 START --- Node: {socket.gethostname()}")

    try:
        # --- RESOLVE WORK DIRECTORY ---
        if not args.work_dir and getattr(args, 'resume_ms', None):
            args.work_dir = os.path.dirname(args.resume_ms)
        if not args.work_dir:
            raise ValueError("No work_dir provided and no --resume_ms to infer from.")

        org_args = load_state(args.work_dir)
        os.chdir(args.work_dir)
        utils.redirect_casa_log(args.work_dir)

        # --- 1. CREATE DIRECTORY STRUCTURE (NVMe) ---
        subdirs = ["I/deep", "V/deep", "I/10min", "V/10min", "snapshots", "QA"]
        for d in subdirs:
            os.makedirs(os.path.join(args.work_dir, d), exist_ok=True)
        logger.info(f"Created directory structure: {subdirs}")

        concat_ms = None
        skip_to_science = getattr(args, 'resume_science', False)
        subband_val = org_args.subband if org_args else args.subband

        if skip_to_science:
            # Skip concat, AOFlagger, wsclean, PB correction  go straight to science
            # Requires that images already exist in work_dir from a prior run
            # Science outputs go to a separate directory (read from work_dir, write to science_out_dir)
            science_out_dir = getattr(args, 'science_out_dir', None)
            if science_out_dir:
                os.makedirs(science_out_dir, exist_ok=True)
                for d in ['detections', 'samples', 'Movies', 'Dewarp_Diagnostics',
                           'QA', 'logs']:
                    os.makedirs(os.path.join(science_out_dir, d), exist_ok=True)
                logger.info(f"RESUME_SCIENCE: Reading images from {args.work_dir}")
                logger.info(f"RESUME_SCIENCE: Writing science outputs to {science_out_dir}")
            else:
                science_out_dir = args.work_dir
                logger.warning("RESUME_SCIENCE: No --science_out_dir specified, "
                               "writing to source work_dir (not recommended)")
            # Find the concat MS for timestamp lookups (used by movies etc.)
            concat_candidates = glob.glob(os.path.join(args.work_dir, "*_concat.ms"))
            concat_ms = concat_candidates[0] if concat_candidates else None

        # --- 2. CONCATENATE / RESUME ---
        elif getattr(args, 'resume_ms', None):
            science_out_dir = args.work_dir
            if not os.path.exists(args.resume_ms):
                raise FileNotFoundError(f"Resume MS missing: {args.resume_ms}")
            concat_ms = args.resume_ms
            logger.info(f"Resuming from existing MS: {concat_ms}")
        else:
            science_out_dir = args.work_dir
            peel_list_file = os.path.join(args.work_dir, "peel_list.txt")
            if not os.path.exists(peel_list_file):
                raise FileNotFoundError("peel_list.txt not found. Setup may have failed.")
            with open(peel_list_file, "r") as f:
                ms_files = f.read().splitlines()

            subband_for_concat = org_args.subband if org_args else args.subband
            concat_ms = utils.concatenate_with_auto_heal(
                ms_files, args.work_dir, subband_for_concat, logger
            )

            if concat_ms is None:
                raise RuntimeError(f"Concatenation failed irrecoverably for {subband_for_concat}.")

            # Clean up intermediate MS files (NVMe space is precious)
            if not (org_args and org_args.skip_cleanup):
                for ms in ms_files:
                    if os.path.exists(ms):
                        shutil.rmtree(ms)

        # --- 3. PRE-PROCESSING through 6. IMAGING + PB CORRECTION ---
        if not skip_to_science:
            logger.info("Applying Field ID Fix (Force to 0)...")
            utils.fix_field_id(concat_ms, logger)

            lst_label = org_args.lst_label if org_args else args.lst_label
            hour_int = int(lst_label.replace('h', ''))

            run_subprocess(
                ['chgcentre', concat_ms, f"{hour_int:02d}h30m00s", "37d12m57.057s"],
                "Phase Center Change"
            )
            run_subprocess(
                ['aoflagger', '-strategy', config.AOFLAGGER_STRATEGY, concat_ms],
                "AOFlagger (Post-Concat)"
            )

            # --- 4. PILOT SNAPSHOT QA ---
            try:
                t = table(concat_ms, ack=False)
                times = t.getcol("TIME")
                n_ints = len(np.unique(times))
                t.close()
            except Exception:
                n_ints = 100

            subband_val = org_args.subband if org_args else args.subband
            pilot_name = f"{subband_val}-{config.SNAPSHOT_PARAMS['suffix']}"
            pilot_path = os.path.join(args.work_dir, "snapshots", pilot_name)

            # Use node-aware thread limit for wsclean
            _, _, wsclean_j_pilot = config.get_image_resources(subband_val)

            cmd_pilot = (
                [os.environ.get('WSCLEAN_BIN', 'wsclean')]
                + ['-j', str(wsclean_j_pilot)]
                + config.SNAPSHOT_PARAMS['args']
                + ['-name', pilot_path, '-intervals-out', str(n_ints), concat_ms]
            )
            run_subprocess(cmd_pilot, "Pilot Snapshot Imaging")

            utils.add_timestamps_to_images(
                os.path.join(args.work_dir, "snapshots"),
                pilot_name, concat_ms, n_ints, logger
            )

            # QA: Flag bad integrations based on Stokes V RMS
            pilot_v = sorted(glob.glob(
                os.path.join(args.work_dir, "snapshots", f"{pilot_name}*-V-image.fits")
            ))
            bad_idx, stats = utils.analyze_snapshot_quality(pilot_v, logger)
            utils.plot_snapshot_diagnostics(stats, bad_idx, args.work_dir, subband_val)

            if bad_idx:
                utils.flag_bad_integrations(concat_ms, bad_idx, n_ints, logger)

            # --- 5. HOT BASELINE REMOVAL ---
            run_hot = org_args.hot_baselines if org_args else getattr(args, 'hot_baselines', False)
            if run_hot and hot_baseline_worker is not None:
                class HotArgs:
                    ms = concat_ms
                    col = "CORRECTED_DATA"
                    uv_cut = 0.0
                    uv_cut_lambda = config.HOT_BASELINE_PARAMS['uv_cut_lambda']
                    sigma = config.HOT_BASELINE_PARAMS['heatmap_sigma']
                    uv_sigma = config.HOT_BASELINE_PARAMS['uv_sigma']
                    threshold = config.HOT_BASELINE_PARAMS['bad_antenna_threshold']
                    uv_window_size = config.HOT_BASELINE_PARAMS.get('uv_window_size', 100)
                    apply_antenna_flags = config.HOT_BASELINE_PARAMS['apply_flags']
                    apply_baseline_flags = config.HOT_BASELINE_PARAMS['apply_flags']
                    run_uv = config.HOT_BASELINE_PARAMS['run_uv_analysis']
                    run_heatmap = config.HOT_BASELINE_PARAMS['run_heatmap_analysis']

                qa_dir = os.path.join(args.work_dir, "QA")
                cwd = os.getcwd()
                os.chdir(qa_dir)
                try:
                    hot_baseline_worker.run_diagnostics(HotArgs, logger)
                except Exception as e:
                    logger.error(f"Hot Baseline Analysis crashed: {e}")
                    traceback.print_exc()
                finally:
                    os.chdir(cwd)

            # --- 6. SCIENCE IMAGING (wsclean) ---
            logger.info(f"Starting Science Imaging for {subband_val}...")

            # Limit wsclean threads on shared nodes
            _, _, wsclean_j = config.get_image_resources(subband_val)
            logger.info(f"wsclean thread limit: -j {wsclean_j}")

            for step in config.IMAGING_STEPS:
                target_dir = os.path.join(args.work_dir, step['pol'], step['category'])
                os.makedirs(target_dir, exist_ok=True)
                base = f"{subband_val}-{step['suffix']}"
                full_path = os.path.join(target_dir, base)

                cmd = (
                    [os.environ.get('WSCLEAN_BIN', 'wsclean')]
                    + ['-j', str(wsclean_j)]
                    + step['args']
                    + ['-name', full_path]
                )

                # Handle interval output
                n_out = 1
                if '-intervals-out' in step['args']:
                    n_out = int(step['args'][step['args'].index('-intervals-out') + 1])
                elif step.get('per_integration'):
                    cmd += ['-intervals-out', str(n_ints)]
                    n_out = n_ints

                cmd.append(concat_ms)
                run_subprocess(cmd, f"Imaging {step['suffix']}")

                # Add timestamps to interval outputs
                utils.add_timestamps_to_images(target_dir, base, concat_ms, n_out, logger)

                # Apply Primary Beam Correction
                if apply_pb_correction:
                    logger.info(f"PB Correcting {base}...")
                    for img in glob.glob(os.path.join(target_dir, f"{base}*image*.fits")):
                        if "pbcorr" in img:
                            continue
                        try:
                            apply_pb_correction(img)
                        except Exception:
                            traceback.print_exc()

        # =================================================================
        # SCIENCE PHASE (all on NVMe)
        # =================================================================
        try:
            freq_mhz = float(subband_val.replace('MHz', ''))
        except Exception:
            freq_mhz = 50.0

        # --- A. Ionospheric Dewarping (VLSSr cross-match) ---
        logger.info("--- Science A: Ionospheric Dewarping (VLSSr) ---")
        if generate_warp_screens is not None and load_ref_catalog is not None:
            vlssr = load_ref_catalog(config.VLSSR_CATALOG, "VLSSr")
            # Dewarp both PB-corrected (for cutouts/photometry) and
            # non-PB-corrected (for transient search) images
            files_to_warp = glob.glob(os.path.join(args.work_dir, "*", "*", "*pbcorr.fits"))
            # Exclude already-dewarped files
            files_to_warp = [f for f in files_to_warp if "_dewarped" not in f]
            # Also find raw image files (not pbcorr) for transient search
            raw_images = glob.glob(os.path.join(args.work_dir, "*", "*", "*image*.fits"))
            raw_images = [f for f in raw_images if "pbcorr" not in f and "_dewarped" not in f]
            files_to_warp.extend(raw_images)
            calc_img = utils.find_deep_image(args.work_dir, freq_mhz, 'I')

            if calc_img and vlssr:
                df = utils.extract_sources_to_df(calc_img, logger)
                with fits.open(calc_img) as h:
                    wcs = WCS(h[0].header).celestial
                    shape = h[0].data.squeeze().shape

                # Write warp diagnostics to science_out_dir so they survive archival
                diag_dir = os.path.join(science_out_dir, "Dewarp_Diagnostics")
                os.makedirs(diag_dir, exist_ok=True)
                warp_base = os.path.join(diag_dir, f"{subband_val}_warp")

                # generate_warp_screens writes some plots relative to cwd,
                # so chdir into diag_dir to capture them
                prev_cwd = os.getcwd()
                os.chdir(diag_dir)
                try:
                    sx, sy, _, _ = generate_warp_screens(
                        df, vlssr, wcs, shape, freq_mhz, 74.0,
                        5.0 / 60.0, 5.0, base_name=warp_base
                    )
                finally:
                    os.chdir(prev_cwd)
                if sx is not None:
                    for f in files_to_warp:
                        # Write dewarped files to science_out_dir, mirroring subdir structure
                        rel_path = os.path.relpath(f, args.work_dir)
                        out = os.path.join(science_out_dir,
                                           rel_path.replace('.fits', '_dewarped.fits'))
                        os.makedirs(os.path.dirname(out), exist_ok=True)
                        if os.path.exists(out):
                            continue
                        try:
                            with fits.open(f) as hf:
                                if hf[0].data.squeeze().shape == sx.shape:
                                    fits.writeto(
                                        out,
                                        apply_warp(hf[0].data.squeeze(), sx, sy),
                                        hf[0].header, overwrite=True
                                    )
                        except Exception:
                            pass
            else:
                logger.warning("No deep I image or VLSSr catalog  skipping dewarping.")
        else:
            logger.warning("Warp modules not available  skipping dewarping.")

        # --- B. Target Photometry ---
        logger.info("--- Science B: Target Photometry ---")
        target_files = org_args.targets if org_args else getattr(args, 'targets', None)

        if cutout is None:
            logger.warning("cutout module not available  skipping target photometry. "
                           "Check that cutout.py is on PYTHONPATH.")
        elif not target_files:
            logger.info("No target files specified  skipping target photometry.")
        else:
            local_samples = os.path.join(science_out_dir, "samples")
            local_detects = os.path.join(science_out_dir, "detections")
            os.makedirs(local_samples, exist_ok=True)
            os.makedirs(local_detects, exist_ok=True)
            for t_file in target_files:
                if not os.path.exists(t_file):
                    logger.warning(f"Target file not found: {t_file}")
                    continue
                logger.info(f"Processing target file: {t_file}")
                try:
                    s_name = os.path.splitext(os.path.basename(t_file))[0]
                    targets = cutout.load_targets(t_file)
                    logger.info(f"  Loaded {len(targets)} targets from {os.path.basename(t_file)}")
                    for nm, crd, det_stokes, confusing_sources in targets:
                        try:
                            conf_str = f" [{len(confusing_sources)} confusing src(s)]" if confusing_sources else ""
                            logger.info(f"  Cutout: {nm} @ ({crd.ra.deg:.3f}, {crd.dec.deg:.3f}) "
                                        f"det={det_stokes}{conf_str}")
                            cutout.process_target(
                                science_out_dir, nm, crd, s_name,
                                local_samples, local_detects,
                                fallback_dir=args.work_dir,
                                detection_stokes=det_stokes,
                                confusing_sources=confusing_sources
                            )
                        except Exception as e:
                            logger.error(f"  Target '{nm}' failed: {e}")
                except Exception as e:
                    logger.error(f"Failed to process target file {t_file}: {e}")
                    traceback.print_exc()

        # --- B2. Solar System Body Photometry ---
        logger.info("--- Science B2: Solar System Photometry ---")
        if solar_system_cutout is not None:
            local_samples = os.path.join(science_out_dir, "samples")
            local_detects = os.path.join(science_out_dir, "detections")
            os.makedirs(local_samples, exist_ok=True)
            os.makedirs(local_detects, exist_ok=True)
            try:
                solar_system_cutout.process_solar_system(
                    science_out_dir, local_samples, local_detects,
                    fallback_dir=args.work_dir, logger=logger
                )
            except Exception as e:
                logger.error(f"Solar system photometry failed: {e}")
                traceback.print_exc()
        else:
            logger.warning("solar_system_cutout not available  skipping solar system photometry.")

        # --- C. Transient Search ---
        logger.info("--- Science C: Transient Search ---")
        catalog = org_args.catalog if org_args else getattr(args, 'catalog', None)
        if trans_tools and catalog:
            # Use TAPERED, DEWARPED, NOT PB-CORRECTED images for transient search
            # Fallback: if dewarped non-pbcorr don't exist, use raw (non-dewarped, non-pbcorr)
            def find_transient_images(pol, category, suffix_filter=None):
                """Find tapered, dewarped, non-pbcorr images. Optionally filter by suffix."""
                # Prefer dewarped in science_out_dir (that's where dewarping writes)
                pat = os.path.join(science_out_dir, pol, category, "*Taper*_dewarped.fits")
                imgs = [f for f in glob.glob(pat)
                        if "pbcorr" not in f and "_dewarped_dewarped" not in f]
                if not imgs:
                    # Also check work_dir for dewarped files from prior runs
                    pat = os.path.join(args.work_dir, pol, category, "*Taper*_dewarped.fits")
                    imgs = [f for f in glob.glob(pat)
                            if "pbcorr" not in f and "_dewarped_dewarped" not in f]
                if not imgs:
                    # Fallback: raw (non-dewarped, non-pbcorr) from work_dir
                    pat = os.path.join(args.work_dir, pol, category, "*Taper*image*.fits")
                    imgs = [f for f in glob.glob(pat) if "pbcorr" not in f and "dewarped" not in f]
                    if imgs:
                        logger.warning(f"No dewarped non-pbcorr images for {pol}/{category}  "
                                       f"falling back to {len(imgs)} raw images")
                        img_type = "raw"
                    else:
                        return [], "none"
                else:
                    img_type = "dewarped"

                # Apply suffix filter if specified (e.g., match robust weighting)
                if suffix_filter:
                    filtered = [f for f in imgs
                                if suffix_filter in os.path.basename(f)
                                and "NoTaper" not in os.path.basename(f)]
                    if filtered:
                        imgs = filtered
                    else:
                        logger.warning(f"No images matching suffix '{suffix_filter}' in {pol}/{category}, "
                                       f"using all {len(imgs)} available")
                return imgs, img_type

            # Output directory for transient search cutouts and debug FITS
            local_transient_detections = os.path.join(science_out_dir, "detections")
            os.makedirs(local_transient_detections, exist_ok=True)

            # --- Find the deep Stokes I image (for V masking and I reference) ---
            # The 10min I snapshots use Robust-0 (suffix 'I-Taper-10min'),
            # so the deep reference must also use Robust-0 to avoid subtraction artifacts.
            ref_i_imgs, ref_i_type = find_transient_images("I", "deep", suffix_filter="Robust-0-")
            # The filter "Robust-0-" matches "Robust-0-image" but not "Robust-0.75-image"
            ref_i_path = ref_i_imgs[0] if ref_i_imgs else None

            if ref_i_path:
                logger.info(f"Deep I reference ({ref_i_type}): {os.path.basename(ref_i_path)}")
            else:
                logger.warning("No deep I image found for masking or subtraction.")

            # --- Stokes V: NO subtraction, search each image directly ---
            # Pass the deep I image for bright source masking (NOT the V image!)
            logger.info("Running Stokes V Blind Search (no subtraction)...")
            v_deep, v_deep_type = find_transient_images("V", "deep")
            v_10min, v_10min_type = find_transient_images("V", "10min")
            v_all = v_deep + v_10min
            v_detections = []
            for v_img in v_all:
                try:
                    # ref=None (no subtraction), ref_i=deep I image (for masking + cutout I panel)
                    result = trans_tools.run_test(None, v_img, ref_i_path, catalog,
                                                  output_dir=local_transient_detections)
                    if result:
                        v_detections.extend(result if isinstance(result, list) else [result])
                except Exception as e:
                    logger.error(f"V transient search failed on {os.path.basename(v_img)}: {e}")

            # --- Stokes I: subtract deep from each 10min snapshot ---
            # Both must use the same robust weighting (Robust-0)
            logger.info("Running Stokes I Subtraction Search...")
            i_snaps, i_snap_type = find_transient_images("I", "10min")
            i_detections = []

            if ref_i_path:
                logger.info(f"Using {ref_i_type} reference: {os.path.basename(ref_i_path)}")
                logger.info(f"Searching {len(i_snaps)} {i_snap_type} 10min snapshots")
                for i_img in i_snaps:
                    try:
                        result = trans_tools.run_test(ref_i_path, i_img, ref_i_path, catalog,
                                                      output_dir=local_transient_detections)
                        if result:
                            i_detections.extend(result if isinstance(result, list) else [result])
                    except Exception as e:
                        logger.error(f"I transient search failed on {os.path.basename(i_img)}: {e}")
            else:
                logger.warning("No deep I Robust-0 image found  skipping I subtraction search.")

            # Quality gate: flag if too many candidates (likely bad data)
            total_det = len(v_detections) + len(i_detections)
            if total_det > 10:
                logger.warning(
                    f"QUALITY FLAG: {total_det} transient candidates found  "
                    f"expected very few. Data quality may be poor. "
                    f"Only top 10 by SNR will have cutouts."
                )
            logger.info(f"Transient candidates: {len(v_detections)} Stokes V, {len(i_detections)} Stokes I")

        else:
            if not trans_tools:
                logger.warning("transient_search not available  skipping transient search.")
            elif not catalog:
                logger.info("No catalog specified  skipping transient search.")

        # --- D. Movies ---
        # Snapshots live in original work_dir; movies go to science_out_dir
        generate_local_movies(science_out_dir, subband_val,
                              snap_source_dir=args.work_dir)

        # =================================================================
        # 7. ARCHIVE TO LUSTRE (only step that touches Lustre)
        # =================================================================
        try:
            start_time_val = org_args.start_time if org_args else args.start_time
            date_str = start_time_val.replace('T', ':').split('.')[0]
            obs_date = datetime.strptime(date_str, '%Y-%m-%d:%H:%M:%S').strftime('%Y-%m-%d')
        except Exception:
            obs_date = "unknown"

        # Determine archive label: use science_out_dir label if resume_science
        run_label_val = (
            args.run_label if args.run_label
            else (org_args.run_label if org_args else "Unknown_Run")
        )
        if skip_to_science and getattr(args, 'science_out_dir', None):
            # Extract the science label from science_out_dir path
            # science_out_dir = /fast/gh/main/{lst}/{date}/{science_label}/{subband}
            archive_label = os.path.basename(os.path.dirname(science_out_dir))
        else:
            archive_label = run_label_val
        lst_label_val = org_args.lst_label if org_args else args.lst_label

        archive_base = os.path.join(
            config.LUSTRE_ARCHIVE_DIR, lst_label_val, obs_date,
            archive_label, subband_val
        )
        os.makedirs(archive_base, exist_ok=True)

        logger.info(f"Archiving results to Lustre: {archive_base}")

        # Archive from science_out_dir (which is the same as work_dir for normal runs)
        archive_source = science_out_dir

        # Structured copy  preserves directory layout for run-local products
        for top_level in ['I', 'V', 'snapshots', 'QA', 'Movies', 'Dewarp_Diagnostics']:
            src = os.path.join(archive_source, top_level)
            dest = os.path.join(archive_base, top_level)
            if os.path.exists(src):
                if os.path.exists(dest):
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)

        # Copy loose files (logs, debug FITS, etc.)
        for f in glob.glob(os.path.join(archive_source, "*")):
            if os.path.isfile(f):
                shutil.copy(f, archive_base)

        # --- Centralized samples archive ---
        # /lustre/gh/main/samples/{sample_name}/{target_name}/{subband}/
        samples_src = os.path.join(archive_source, "samples")
        if os.path.exists(samples_src):
            lustre_samples_root = os.path.join(config.LUSTRE_ARCHIVE_DIR, "samples")
            for sample_dir in glob.glob(os.path.join(samples_src, "*")):
                if not os.path.isdir(sample_dir):
                    continue
                sample_name = os.path.basename(sample_dir)
                for target_dir in glob.glob(os.path.join(sample_dir, "*")):
                    if not os.path.isdir(target_dir):
                        # Loose file (e.g. photometry CSV) at sample level
                        dest_dir = os.path.join(lustre_samples_root, sample_name, subband_val)
                        os.makedirs(dest_dir, exist_ok=True)
                        shutil.copy(target_dir, dest_dir)
                        continue
                    target_name = os.path.basename(target_dir)
                    dest_dir = os.path.join(
                        lustre_samples_root, sample_name, target_name, subband_val)
                    os.makedirs(dest_dir, exist_ok=True)
                    for f in glob.glob(os.path.join(target_dir, "*")):
                        if os.path.isfile(f):
                            shutil.copy(f, dest_dir)
                        elif os.path.isdir(f):
                            dest_sub = os.path.join(dest_dir, os.path.basename(f))
                            if os.path.exists(dest_sub):
                                shutil.rmtree(dest_sub)
                            shutil.copytree(f, dest_sub)
            logger.info(f"Samples archived to {lustre_samples_root}")

        # --- Centralized detections archive ---
        # Transients: /lustre/gh/main/detections/transients/{I,V}/{J-name}/{subband}/
        # SolarSystem: /lustre/gh/main/detections/SolarSystem/{Body}/{subband}/
        # Target detections: /lustre/gh/main/detections/{sample_name}/{target_name}/{subband}/
        detections_src = os.path.join(archive_source, "detections")
        if os.path.exists(detections_src):
            lustre_det_root = os.path.join(config.LUSTRE_ARCHIVE_DIR, "detections")

            # Transients
            for stokes in ['I', 'V']:
                trans_src = os.path.join(detections_src, "transients", stokes)
                if not os.path.exists(trans_src):
                    continue
                for jdir in glob.glob(os.path.join(trans_src, "*")):
                    if not os.path.isdir(jdir):
                        continue
                    jname = os.path.basename(jdir)
                    dest_dir = os.path.join(
                        lustre_det_root, "transients", stokes, jname, subband_val)
                    os.makedirs(dest_dir, exist_ok=True)
                    for f in glob.glob(os.path.join(jdir, "*")):
                        if os.path.isfile(f):
                            shutil.copy(f, dest_dir)

            # Transient debug files
            debug_src = os.path.join(detections_src, "transients", "debug")
            if os.path.exists(debug_src):
                dest_dir = os.path.join(lustre_det_root, "transients", "debug", subband_val)
                os.makedirs(dest_dir, exist_ok=True)
                for f in glob.glob(os.path.join(debug_src, "*")):
                    if os.path.isfile(f):
                        shutil.copy(f, dest_dir)

            # Solar System
            ss_src = os.path.join(detections_src, "SolarSystem")
            if os.path.exists(ss_src):
                for body_dir in glob.glob(os.path.join(ss_src, "*")):
                    if not os.path.isdir(body_dir):
                        continue
                    body = os.path.basename(body_dir)
                    dest_dir = os.path.join(
                        lustre_det_root, "SolarSystem", body, subband_val)
                    os.makedirs(dest_dir, exist_ok=True)
                    for f in glob.glob(os.path.join(body_dir, "*")):
                        if os.path.isfile(f):
                            shutil.copy(f, dest_dir)

            # Target-based detections (sample_name/target_name subdirs)
            for item in glob.glob(os.path.join(detections_src, "*")):
                bn = os.path.basename(item)
                if bn in ('transients', 'SolarSystem', 'debug'):
                    continue
                if os.path.isdir(item):
                    # This is a sample_name or target_name directory
                    for target_dir in glob.glob(os.path.join(item, "*")):
                        if os.path.isdir(target_dir):
                            target_name = os.path.basename(target_dir)
                            dest_dir = os.path.join(
                                lustre_det_root, bn, target_name, subband_val)
                            os.makedirs(dest_dir, exist_ok=True)
                            for f in glob.glob(os.path.join(target_dir, "*")):
                                if os.path.isfile(f):
                                    shutil.copy(f, dest_dir)

            logger.info(f"Detections archived to {lustre_det_root}")

        # Clean up NVMe
        if not (org_args and org_args.skip_cleanup):
            if concat_ms and os.path.exists(concat_ms):
                shutil.rmtree(concat_ms)

        logger.info("--- PIPELINE COMPLETE ---")

    except Exception as e:
        logger.critical(f"Pipeline Failed: {e}")
        traceback.print_exc()
        sys.exit(1)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OVRO-LWA Per-Subband Pipeline Worker")

    parser.add_argument("--mode", default="setup", choices=["setup", "peel", "image"])
    parser.add_argument("--work_dir")
    parser.add_argument("--subband")
    parser.add_argument("--start_time")
    parser.add_argument("--end_time")
    parser.add_argument("--bp_table")
    parser.add_argument("--xy_table")
    parser.add_argument("--leakage")
    parser.add_argument("--input_dir")
    parser.add_argument("--run_label")
    parser.add_argument("--lst_label", default="14h")
    parser.add_argument("--peel_sky", action="store_true")
    parser.add_argument("--peel_rfi", action="store_true")
    parser.add_argument("--hot_baselines", action="store_true")
    parser.add_argument("--override_range", action="store_true")
    parser.add_argument("--skip-cleanup", action="store_true", dest="skip_cleanup")
    parser.add_argument("--resume_ms")
    parser.add_argument("--resume_science", action="store_true",
                        help="Skip to science phases (dewarping, cutouts, transients). "
                             "Requires images to already exist in work_dir.")
    parser.add_argument("--science_out_dir",
                        help="Separate output directory for science products (used with --resume_science). "
                             "If not specified, outputs go to work_dir.")
    parser.add_argument("--targets", nargs='+')
    parser.add_argument("--catalog")

    args = parser.parse_args()

    # Normalize time strings
    if args.start_time:
        args.start_time = args.start_time.replace('T', ':')
    if args.end_time:
        args.end_time = args.end_time.replace('T', ':')

    # Dispatch
    if args.mode == "setup":
        run_stage_setup(args)
    elif args.mode == "peel":
        run_stage_peel(args)
    elif args.mode == "image":
        run_stage_image(args)
