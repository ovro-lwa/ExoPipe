"""
pipeline_utils.py — Shared utilities for the OVRO-LWA pipeline.

MERGED: Old stable base + new run_casa_task subprocess wrapper.
FIXES: utils.calculate_spwmap → calculate_spwmap (NameError fix),
       restored _with_scrcols cleanup in auto-heal.
"""
import os
import sys
import shutil
import logging
import subprocess
import glob
import re
import json
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import casacore.tables as pt
from casatasks import casalog, concat, flagdata
from astropy.time import Time
from astropy.io import fits
from astropy import units as u
from astropy.stats import mad_std
import bdsf
import traceback


def get_logger(name):
    return logging.getLogger(name)


def redirect_casa_log(work_dir):
    """Redirect all CASA logs to a logs/ subdirectory to prevent pollution."""
    log_dir = os.path.join(work_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'casa_pipeline.log')
    try:
        casalog.setlogfile(log_path)
        # Also set env var to catch any subprocess CASA instances
        os.environ['CASALOGFILE'] = log_path
    except Exception as e:
        print(f"Failed to redirect CASA log: {e}")


# --- CASA SUBPROCESS WRAPPER (NEW — kept from Gemini rewrite) ---
def run_casa_task(work_dir, task_code, logger):
    """Execute a CASA task in a subprocess for crash isolation."""
    timestamp = int(time.time())
    log_dir = os.path.join(work_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    task_script = os.path.join(work_dir, f"casa_task_{timestamp}.py")
    log_path = os.path.join(log_dir, 'casa_pipeline.log')

    full_code = f"""
__casatask__ = True
import sys
import os
from casatasks import casalog
casalog.setlogfile('{log_path}')
try:
{task_code}
except Exception as e:
    print(f"INTERNAL_CASA_ERROR: {{e}}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
    try:
        with open(task_script, "w") as f:
            f.write(full_code)
        result = subprocess.run([sys.executable, task_script], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"CASA Subprocess Failure (Exit {result.returncode})")
            logger.error(f"Captured Traceback:\n{result.stderr.strip()}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to launch CASA subprocess: {e}")
        return False
    finally:
        if os.path.exists(task_script):
            os.remove(task_script)


# --- SPW MAPPING ---
def calculate_spwmap(ms_path, caltable_path, logger=None):
    if logger is None: logger = get_logger('Utils.SPWMap')
    try:
        with pt.table(os.path.join(ms_path, 'SPECTRAL_WINDOW'), ack=False) as t:
            ms_freqs = [t.getcell('CHAN_FREQ', i) for i in range(t.nrows())]
        with pt.table(os.path.join(caltable_path, 'SPECTRAL_WINDOW'), ack=False) as t:
            cal_freqs = [t.getcell('CHAN_FREQ', i) for i in range(t.nrows())]
        spwmap = []
        for ms_f in ms_freqs:
            ms_min, ms_max = np.min(ms_f), np.max(ms_f)
            best_match, best_overlap = -1, 0.0
            for cal_spw_idx, cal_f in enumerate(cal_freqs):
                cal_min, cal_max = np.min(cal_f), np.max(cal_f)
                overlap_min, overlap_max = max(ms_min, cal_min), min(ms_max, cal_max)
                if overlap_max > overlap_min:
                    overlap_bw = overlap_max - overlap_min
                    if overlap_bw > best_overlap:
                        best_overlap, best_match = overlap_bw, cal_spw_idx
            if best_match == -1:
                ms_center = np.mean(ms_f)
                diffs = [np.abs(np.mean(cf) - ms_center) for cf in cal_freqs]
                best_match = np.argmin(diffs)
            spwmap.append(int(best_match))
        return spwmap
    except Exception as e:
        logger.error(f"Error calculating SPW map: {e}")
        return None


# --- CALIBRATION ---
def apply_calibration(ms_path, bp_table, xy_table, logger):
    try:
        with pt.table(os.path.join(ms_path, "SPECTRAL_WINDOW"), ack=False) as t:
            freqs = t.getcol("CHAN_FREQ").ravel()
            max_freq_hz = np.max(freqs)
        if max_freq_hz > 85e6:
            logger.info(f"Flagging frequencies > 85 MHz in {os.path.basename(ms_path)}...")
            flagdata(vis=ms_path, mode='manual', spw='*:85.0~100.0MHz', flagbackup=False)
    except Exception as e:
        logger.warning(f"Could not check frequencies for >85MHz flagging: {e}")

    # FIX: direct call (was broken as utils.calculate_spwmap in Gemini version)
    bp_map = calculate_spwmap(ms_path, bp_table, logger)
    xy_map = calculate_spwmap(ms_path, xy_table, logger)

    if bp_map is None or xy_map is None:
        logger.error("Failed to map SPWs. Skipping calibration application.")
        return False

    python_code = f"""
import sys
from casatasks import clearcal, applycal
try:
    clearcal(vis='{ms_path}', addmodel=False)
    applycal(vis='{ms_path}', gaintable=['{bp_table}', '{xy_table}'], spwmap=[{bp_map}, {xy_map}], flagbackup=False, calwt=False)
except Exception as e:
    print(f"CASA Error: {{e}}")
    sys.exit(1)
"""
    try:
        result = subprocess.run([sys.executable, "-c", python_code], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Calibration subprocess failed: {result.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"Subprocess launch failed: {e}")
        return False


# --- AUTO-HEAL CONCATENATION ---
def find_culprit_generalized(log_path, logger):
    if not os.path.exists(log_path): return None
    ms_pattern = re.compile(r"concatenating (.*\.ms) into")
    error_patterns = ["FilebufIO::readBlock", "BucketCache", "RuntimeError", "SEVERE"]
    try:
        with open(log_path, 'r') as f: lines = f.readlines()
        for i in range(len(lines) - 1, -1, -1):
            if any(p in lines[i] for p in error_patterns):
                for j in range(i, -1, -1):
                    match = ms_pattern.search(lines[j])
                    if match: return match.group(1).strip()
    except Exception as e:
        logger.error(f"Log parsing failed: {e}")
    return None


def concatenate_with_auto_heal(ms_files, work_dir, subband, logger):
    output_ms = os.path.join(work_dir, f"{subband}_concat.ms")
    os.environ['CASA_USE_NO_LOCKING'] = '1'
    current_list = sorted(list(ms_files))
    attempt, max_retries, success = 0, 10, False

    while attempt < max_retries and not success:
        attempt_log = os.path.join(work_dir, f"concat_attempt_{attempt}.log")
        casalog.setlogfile(attempt_log)
        logger.info(f"Concat Attempt {attempt}: Merging {len(current_list)} files...")
        try:
            if os.path.exists(output_ms): shutil.rmtree(output_ms)
            concat(vis=current_list, concatvis=output_ms, timesort=True)
            logger.info(f"SUCCESS: Created {os.path.basename(output_ms)}")
            success = True
        except Exception:
            logger.warning(f"Merge failed (Attempt {attempt}). Analyzing log for culprit...")
            bad_ms = find_culprit_generalized(attempt_log, logger)
            if bad_ms and bad_ms in current_list:
                logger.error(f"--> AUTO-HEAL: Pruning corrupt file: {os.path.basename(bad_ms)}")
                current_list.remove(bad_ms)
                # Clean up scratch column dirs (restored from old version)
                for suffix in ["", "_with_scrcols"]:
                    path = bad_ms + suffix
                    if os.path.exists(path) and suffix != "":
                        shutil.rmtree(path)
            else:
                logger.error("Culprit identification failed. Retrying...")
        attempt += 1
    return output_ms if success else None


# --- FIELD ID FIX ---
def fix_field_id(ms_path, logger):
    logger.info(f"Applying Field ID Fix (Set to 0) on {os.path.basename(ms_path)}...")
    try:
        with pt.table(ms_path, readonly=False, ack=False) as t:
            if 'FIELD_ID' in t.colnames():
                t.putcol('FIELD_ID', np.zeros(t.nrows(), dtype=np.int32))
        return True
    except Exception as e:
        logger.error(f"Failed to fix FIELD_ID: {e}")
        return False


# --- FLAGGING & QA ---
def run_antenna_flagging(ms_path, context, script_dir, logger):
    import astropy.time
    try:
        with pt.table(os.path.join(ms_path, 'OBSERVATION'), ack=False) as t:
            time_range = t.getcol('TIME_RANGE')[0]
            obs_mjd = astropy.time.Time(time_range[0], format='mjd', scale='utc').mjd
    except Exception: return False

    helper_script = os.path.join(script_dir, "get_bad_antennas_mnc.py")
    if not os.path.exists(helper_script): return False

    cmd = ['conda', 'run', '-n', 'development', 'python', helper_script, str(obs_mjd)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        lines = res.stdout.strip().split('\n')
        data = None
        for line in reversed(lines):
            try: data = json.loads(line); break
            except Exception: continue
        if data and data.get('bad_correlator_numbers'):
            bad_ant_str = ",".join(map(str, data['bad_correlator_numbers']))
            logger.warning(f"!!! FLAGGING BAD ANTENNAS: {bad_ant_str} !!!")
            flagdata(vis=ms_path, mode='manual', antenna=bad_ant_str, flagbackup=False)
        return True
    except Exception as e:
        logger.error(f"Flagging helper failed: {e}")
        return False


def analyze_snapshot_quality(image_list, logger):
    """Analyze Stokes V snapshot images for QA. Returns bad scan indices and stats.

    Each snapshot image corresponds to one scan in the MS (1:1 mapping).
    The index in the sorted image list == the scan index.
    """
    stats = []
    for idx, img_path in enumerate(image_list):
        try:
            with fits.open(img_path) as hdul:
                data = hdul[0].data.squeeze()
                h, w = data.shape
                cw, ch = w // 2, h // 2
                box_r = 512
                center_region = data[ch-box_r:ch+box_r, cw-box_r:cw+box_r]
                rms = np.nanstd(center_region)
                peak = np.nanmax(np.abs(data))
                stats.append({'idx': idx, 'rms': rms, 'peak': peak, 'file': os.path.basename(img_path)})
        except Exception: continue

    if not stats: return [], []
    rmses = np.array([s['rms'] for s in stats])
    med_rms = np.nanmedian(rmses)
    std_rms = mad_std(rmses, ignore_nan=True)
    high_rms_thresh = med_rms + (3.0 * std_rms)
    low_rms_thresh = 0.5 * med_rms

    bad_indices = []
    for s in stats:
        if s['rms'] > high_rms_thresh: bad_indices.append(s['idx'])
        elif s['rms'] < low_rms_thresh: bad_indices.append(s['idx'])
    logger.info(f"QA: Flagged {len(bad_indices)}/{len(stats)} integrations. Median RMS: {med_rms:.4f}")
    return bad_indices, stats


def plot_snapshot_diagnostics(stats, bad_indices, work_dir, subband):
    if not stats: return
    rmses = [s['rms'] for s in stats]; indices = [s['idx'] for s in stats]
    plt.figure(figsize=(10, 6))
    plt.plot(indices, rmses, 'b.-', label='RMS (Stokes V)')
    if bad_indices:
        bad_rmses = [stats[i]['rms'] for i in range(len(stats)) if stats[i]['idx'] in bad_indices]
        bad_idxs = [stats[i]['idx'] for i in range(len(stats)) if stats[i]['idx'] in bad_indices]
        plt.plot(bad_idxs, bad_rmses, 'rx', markersize=10, label='Flagged')
    plt.xlabel("Scan Index"); plt.ylabel("Image RMS (Jy/beam)")
    plt.title(f"Snapshot Quality: {subband}"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(work_dir, "QA", f"snapshot_rms_vs_time_{subband}.png"))
    plt.close()


def flag_bad_integrations(ms_path, bad_indices, n_total, logger):
    """Flag bad scans using SCAN_NUMBER.

    Each snapshot index maps 1:1 to a scan in the MS. We read the unique
    scan numbers, map bad_indices to scan numbers, and flag by scan.
    This is cleaner and more reliable than time-range flagging.
    """
    if not bad_indices:
        return
    try:
        with pt.table(ms_path, ack=False) as t:
            scans = t.getcol("SCAN_NUMBER")
            unique_scans = sorted(set(scans))

        # Map snapshot indices to scan numbers
        bad_scans = []
        for idx in sorted(bad_indices):
            if idx < len(unique_scans):
                bad_scans.append(unique_scans[idx])
            else:
                logger.warning(f"Snapshot index {idx} exceeds scan count ({len(unique_scans)})")

        if not bad_scans:
            logger.warning("No valid scan numbers found for flagging.")
            return

        scan_str = ",".join(str(s) for s in bad_scans)
        logger.info(f"Applying QA flags on {len(bad_scans)} scans: {scan_str}")
        flagdata(vis=ms_path, mode='manual', scan=scan_str, flagbackup=False)
    except Exception as e:
        logger.error(f"Failed to apply QA flags: {e}")


# --- IMAGE UTILITIES ---
def find_deep_image(run_dir, freq_mhz, pol='I'):
    pat_root = os.path.join(run_dir, f"{int(freq_mhz)}MHz", pol, "deep", "*Taper*pbcorr.fits")
    pat_local = os.path.join(run_dir, pol, "deep", "*Taper*pbcorr.fits")
    candidates = glob.glob(pat_root) + glob.glob(pat_local)
    if not candidates:
        pat_root_raw = os.path.join(run_dir, f"{int(freq_mhz)}MHz", pol, "deep", "*Taper*image.fits")
        pat_local_raw = os.path.join(run_dir, pol, "deep", "*Taper*image.fits")
        candidates = glob.glob(pat_root_raw) + glob.glob(pat_local_raw)
    candidates = [c for c in candidates if "NoTaper" not in c]
    if candidates:
        return sorted(candidates)[0]
    return None


def extract_sources_to_df(filename, logger=None, thresh_pix=10.0):
    if logger is None: logger = get_logger("Utils.Extractor")
    try:
        logger.info(f"Extracting sources from {os.path.basename(filename)} (thresh_pix={thresh_pix:.0f})...")
        img = bdsf.process_image(filename, thresh_pix=thresh_pix, thresh_isl=5.0,
                                 adaptive_rms_box=True, quiet=True)
        sources_raw = []
        for s in img.sources:
            if not np.isnan(s.posn_sky_max[0]):
                sources_raw.append({
                    'ra': s.posn_sky_max[0], 'dec': s.posn_sky_max[1],
                    'flux_peak_I_app': s.peak_flux_max,
                    'maj': getattr(s, 'maj_axis', 0.0),
                    'min': getattr(s, 'min_axis', 0.0)
                })
        if not sources_raw:
            logger.warning(f"No sources found in {filename}")
            return pd.DataFrame()
        return pd.DataFrame(sources_raw)
    except Exception as e:
        logger.error(f"BDSF extraction failed for {filename}: {e}")
        return pd.DataFrame()


def add_timestamps_to_images(target_dir, prefix, ms_path, n_intervals, logger):
    try:
        with pt.table(ms_path, ack=False) as t:
            times = t.getcol("TIME")
            t_min, t_max = np.min(times), np.max(times)
            duration = t_max - t_min

        search_pat = f"{prefix}*-*.fits"
        files = glob.glob(os.path.join(target_dir, search_pat))
        if not files: return False

        for i in range(n_intervals):
            chunk_len = duration / n_intervals if n_intervals > 1 else duration
            mid_mjd_sec = t_min + (i * chunk_len) + (chunk_len / 2.0)
            ts_str = Time(mid_mjd_sec / 86400.0, format='mjd', scale='utc').datetime.strftime("%Y%m%d_%H%M%S")
            old_suffix = f"-t{i:04d}" if n_intervals > 1 else ""
            new_suffix = f"-{ts_str}"

            for f_path in files:
                f_name = os.path.basename(f_path)
                if n_intervals > 1:
                    if old_suffix in f_name:
                        new_name = f_name.replace(old_suffix, new_suffix)
                        shutil.move(f_path, os.path.join(target_dir, new_name))
                else:
                    if new_suffix not in f_name:
                        base, ext = os.path.splitext(f_name)
                        new_name = f"{base}{new_suffix}{ext}"
                        shutil.move(f_path, os.path.join(target_dir, new_name))
        return True
    except Exception as e:
        logger.error(f"Error renaming images: {e}")
        traceback.print_exc()
        return False


# --- FILE UTILITIES ---
def valid_datetime(s):
    try: return datetime.strptime(s.replace('T', ':').split('.')[0], '%Y-%m-%d:%H:%M:%S')
    except ValueError: raise argparse.ArgumentTypeError(f"Not a valid datetime: '{s}'")


def find_archive_files_for_subband(start_dt, end_dt, subband, logger, input_dir=None):
    file_list = []
    filename_pattern = re.compile(r'(\d{8})_(\d{6})_' + re.escape(subband) + r'(?:|_averaged)\.ms')
    if input_dir:
        search_pattern = os.path.join(input_dir, f'*{subband}*.ms')
        for f_path in glob.glob(search_pattern):
            match = filename_pattern.search(os.path.basename(f_path))
            if match:
                date_str_file, time_str_file = match.groups()
                try:
                    file_start_dt = datetime.strptime(date_str_file + time_str_file, '%Y%m%d%H%M%S')
                    if start_dt <= (file_start_dt + timedelta(seconds=5)) < end_dt:
                        file_list.append(f_path)
                except ValueError: pass
    else:
        base_dir = '/lustre/pipeline/night-time/averaged/'
        current_hour = start_dt.replace(minute=0, second=0, microsecond=0)
        end_buffer = end_dt + timedelta(hours=1)
        while current_hour <= end_buffer:
            date_str = current_hour.strftime('%Y-%m-%d')
            hour_str = current_hour.strftime('%H')
            target_dir = os.path.join(base_dir, subband, date_str, hour_str)
            if os.path.isdir(target_dir):
                for f in os.listdir(target_dir):
                    match = filename_pattern.search(f)
                    if match:
                        date_str_file, time_str_file = match.groups()
                        try:
                            file_start_dt = datetime.strptime(date_str_file + time_str_file, '%Y%m%d%H%M%S')
                            if start_dt <= (file_start_dt + timedelta(seconds=5)) < end_dt:
                                file_list.append(os.path.join(target_dir, f))
                        except ValueError: pass
            current_hour += timedelta(hours=1)
    return sorted(file_list)


def copy_and_prepare_files(file_list, dest_dir, subband, logger):
    copied_files = []
    for f in file_list:
        dest = os.path.join(dest_dir, os.path.basename(f))
        if os.path.exists(dest): shutil.rmtree(dest)
        try:
            shutil.copytree(f, dest)
            copied_files.append(dest)
        except Exception as e:
            logger.error(f"Failed to copy {f}: {e}")
    return copied_files
