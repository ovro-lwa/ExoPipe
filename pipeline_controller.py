"""
pipeline_controller.py — OVRO-LWA Pipeline Orchestrator
Converts LST ranges to UTC, segments by LST hour, and submits
per-subband Slurm workers. Optionally waits for completion and
triggers Phase 2 Science Aggregation.

MERGED: Old stable base + fixed argparse ordering.
"""
import argparse
import subprocess
import logging
import sys
import os
import time
import numpy as np
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u
import pipeline_config as config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CONTROLLER] - %(message)s')
logger = logging.getLogger()

OVRO_LOC = config.OVRO_LOC


class LSTScheduler:
    def __init__(self):
        self.location = OVRO_LOC

    def parse_range(self, range_str, date_str=None):
        """Parse LST range (e.g. '14h-18h') or direct UTC range."""
        # Try direct UTC format first: '2025-01-15:03:00:00,2025-01-15:07:00:00'
        if ':' in range_str and ',' in range_str:
            try:
                s, e = range_str.split(',')
                t_start = Time(datetime.strptime(s, '%Y-%m-%d:%H:%M:%S'),
                               scale='utc', location=self.location)
                t_end = Time(datetime.strptime(e, '%Y-%m-%d:%H:%M:%S'),
                             scale='utc', location=self.location)
                return t_start, t_end
            except ValueError:
                pass

        if date_str is None:
            logger.error("LST Range requires --date argument.")
            sys.exit(1)

        try:
            clean_range = range_str.lower().replace('h', '')
            lst_start_val, lst_end_val = map(float, clean_range.split('-'))
            ref_date = datetime.strptime(date_str, '%Y-%m-%d')

            t_ref = Time(ref_date, scale='utc', location=self.location)
            lst_ref = t_ref.sidereal_time('mean').hour

            diff_start = (lst_start_val - lst_ref) % 24
            diff_end = (lst_end_val - lst_ref) % 24
            if diff_end < diff_start:
                diff_end += 24

            sidereal_factor = 0.99726958
            t_start = t_ref + (diff_start * u.hour * sidereal_factor)
            t_end = t_ref + (diff_end * u.hour * sidereal_factor)

            logger.info(f"Calculated UTC Range: {t_start.isot} to {t_end.isot}")
            return t_start, t_end
        except Exception as e:
            logger.error(f"Failed to parse range: {e}")
            sys.exit(1)

    def generate_jobs(self, t_start, t_end, override=False):
        """Split time range into ~1-hour LST segments."""
        if override:
            return [{'start': t_start, 'end': t_end, 'lst_label': 'custom_override'}]

        jobs = []
        current_t = t_start
        while (t_end - current_t).sec > 10.0:
            current_lst = current_t.sidereal_time('mean').hour
            next_lst_hour = np.floor(current_lst) + 1.0
            dt_solar = (next_lst_hour - current_lst) * 0.99726958 * u.hour
            segment_end = current_t + dt_solar if (current_t + dt_solar) < t_end else t_end

            if (segment_end - current_t).sec < 10.0:
                current_t = segment_end
                continue

            midpoint = current_t + (segment_end - current_t) / 2
            mid_lst = int(np.floor(midpoint.sidereal_time('mean').hour)) % 24
            jobs.append({
                'start': current_t,
                'end': segment_end,
                'lst_label': f"{mid_lst:02d}h"
            })
            current_t = segment_end
        return jobs


def pre_flight_audit(args):
    """
    Check all input paths and tables BEFORE submitting any Slurm jobs.
    Saves hours of debugging when a path is wrong.
    """
    logger.info("--- Starting Pre-Flight Audit ---")

    essential_files = []

    if args.resume_science:
        # Science-only mode: cal tables not needed
        logger.info("Resume-science mode: skipping calibration table checks.")
    else:
        # Full run: cal tables are mandatory
        if not args.bp_table or not args.xy_table:
            logger.error("AUDIT FAILED: --bp_table and --xy_table are required for full pipeline runs.")
            sys.exit(1)
        essential_files = [args.bp_table, args.xy_table]

    if args.targets:
        essential_files.extend(args.targets)
    if args.catalog:
        essential_files.append(args.catalog)

    missing = []
    for f in essential_files:
        abs_f = os.path.abspath(f)
        if not os.path.exists(abs_f):
            missing.append(abs_f)
        else:
            logger.info(f"Verified: {abs_f}")

    if missing:
        logger.error("AUDIT FAILED: The following required files were not found:")
        for m in missing:
            logger.error(f"  MISSING: {m}")
        sys.exit(1)

    logger.info("--- Audit Passed. Proceeding to Submission. ---")


def submit_worker_job(subband, job_config, run_label, args, science_label=None):
    """Submit a single per-subband Slurm job."""
    lst_label = job_config['lst_label']
    if args.override_range:
        lst_label = "custom"

    s_str = job_config['start'].isot
    e_str = job_config['end'].isot

    try:
        obs_date = job_config['start'].datetime.strftime('%Y-%m-%d')
    except Exception:
        obs_date = "unknown"

    # Log directory: use science_label if resume_science, otherwise run_label
    log_label = science_label if science_label else run_label
    log_dir = os.path.join(
        config.LUSTRE_ARCHIVE_DIR, lst_label, obs_date,
        log_label, subband, "logs"
    )
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"Master_{subband}.log")
    casa_log_target = os.path.join(log_dir, "casa_pipeline_init.log")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script = os.path.join(script_dir, "process_subband.py")

    # Force absolute paths for calibration tables (if provided)
    abs_bp = os.path.abspath(args.bp_table) if args.bp_table else None
    abs_xy = os.path.abspath(args.xy_table) if args.xy_table else None

    worker_args = (
        f"--subband {subband} --start_time '{s_str}' --end_time '{e_str}' "
        f"--run_label '{run_label}' --lst_label '{lst_label}' "
        f"{('--bp_table ' + abs_bp + ' ') if abs_bp else ''}"
        f"{('--xy_table ' + abs_xy + ' ') if abs_xy else ''}"
        f"{'--leakage ' + args.leakage if args.leakage else ''} "
        f"{'--peel_sky' if args.peel_sky else ''} "
        f"{'--peel_rfi' if args.peel_rfi else ''} "
        f"{'--hot_baselines' if args.hot_baselines else ''} "
        f"{'--override_range' if args.override_range else ''} "
        f"{'--skip-cleanup' if args.skip_cleanup else ''} "
        f"{'--input_dir ' + args.input_dir if args.input_dir else ''}"
    )

    # Pass absolute paths for targets and catalogs
    if args.targets:
        abs_targets = [os.path.abspath(t) for t in args.targets]
        worker_args += f" --targets {' '.join(abs_targets)}"

    if args.catalog:
        worker_args += f" --catalog {os.path.abspath(args.catalog)}"

    # Resume mode: skip setup, go directly to science phases
    if args.resume_science:
        # Source images: original run directory (read-only)
        source_work_dir = f"/fast/gh/main/{lst_label}/{obs_date}/{run_label}/{subband}"
        # Science output: new directory alongside original run
        sci_out_label = science_label or f"Science_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        science_out_dir = f"/fast/gh/main/{lst_label}/{obs_date}/{sci_out_label}/{subband}"
        worker_args += (f" --work_dir {source_work_dir}"
                        f" --science_out_dir {science_out_dir}"
                        f" --mode image --resume_science")
    elif args.resume_from_existing:
        expected_work_dir = f"/fast/gh/main/{lst_label}/{obs_date}/{run_label}/{subband}"
        expected_ms = os.path.join(expected_work_dir, f"{subband}_concat.ms")
        worker_args += f" --resume_ms {expected_ms} --work_dir {expected_work_dir} --mode image"
    else:
        worker_args += " --mode setup"

    wrap_cmd = (
        f"#!/bin/bash\n"
        f"source ~/.bashrc\n"
        f"conda activate {config.REQUIRED_CONDA_ENV}\n"
        f"export PYTHONPATH={script_dir}:$PYTHONPATH\n"
        f"export CASA_USE_NO_LOCKING=1\n"
        f"export CASALOGFILE={casa_log_target}\n"
        f"python -u {worker_script} {worker_args}\n"
    )

    # Scale resources for shared nodes
    pipe_cpus, pipe_mem, _ = config.get_image_resources(subband)

    slurm_cmd = [
        'sbatch',
        f'--job-name=Pipe_{subband}_{lst_label}',
        '--partition=general',
        '--nodes=1', '--ntasks=1', f'--cpus-per-task={pipe_cpus}', f'--mem={pipe_mem}G',
        f'--output={log_file}', f'--error={log_file}',
        f'--chdir={script_dir}'
    ]

    assigned_node = config.NODE_SUBBAND_MAP.get(subband)
    if assigned_node:
        slurm_cmd.append(f'--nodelist={assigned_node}')

    logger.info(f"Submitting {subband} ({lst_label}) to {assigned_node or 'ANY'}")
    proc = subprocess.run(
        slurm_cmd,
        input=wrap_cmd, capture_output=True, text=True
    )
    # Extract job ID from sbatch output (e.g. "Submitted batch job 12345")
    job_id = None
    if proc.stdout:
        parts = proc.stdout.strip().split()
        if parts:
            job_id = parts[-1]
    return job_id



def main():
    parser = argparse.ArgumentParser(description="OVRO-LWA Pipeline Controller")

    # ---- Define ALL arguments BEFORE calling parse_args ----
    parser.add_argument("--range", required=True,
                        help="LST range (e.g. '14h-18h') or UTC range (comma-separated)")
    parser.add_argument("--date", help="Observation date (YYYY-MM-DD), required for LST ranges")
    parser.add_argument("--run_label", help="Label for this pipeline run")
    parser.add_argument("--override_range", action="store_true",
                        help="Bypass LST segmentation — process entire range as one job")
    parser.add_argument("--bp_table", help="Path to bandpass calibration table")
    parser.add_argument("--xy_table", help="Path to XY-phase calibration table")
    parser.add_argument("--leakage", help="Path to D-term / leakage table")
    parser.add_argument("--input_dir", help="Custom input directory (overrides Lustre default)")
    parser.add_argument("--peel_sky", action="store_true", help="Enable sky source peeling")
    parser.add_argument("--peel_rfi", action="store_true", help="Enable RFI source peeling")
    parser.add_argument("--hot_baselines", action="store_true", help="Enable hot baseline analysis")
    parser.add_argument("--skip-cleanup", action="store_true", dest="skip_cleanup",
                        help="Keep intermediate files on NVMe")
    parser.add_argument("--resume_from_existing", action="store_true",
                        help="Skip setup — resume from existing concat MS on NVMe")
    parser.add_argument("--resume_science", action="store_true",
                        help="Skip to science phases only (dewarping, cutouts, transients). "
                             "Requires prior run with images already on NVMe.")
    parser.add_argument("--targets", nargs='+', help="List of target CSV files")
    parser.add_argument("--catalog", help="Bulk transient catalog path")
    parser.add_argument("--skip-phase2", action="store_true",
                        help="Skip Science Aggregation after subband processing")

    args = parser.parse_args()

    # ---- 1. PRE-FLIGHT AUDIT ----
    pre_flight_audit(args)

    run_label = args.run_label or f"Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # For resume_science, create a separate output label
    if args.resume_science:
        science_label = f"Science_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not args.run_label:
            logger.error("--resume_science requires --run_label to identify the source run.")
            sys.exit(1)
        logger.info(f"--- Science Reprocessing --- Source: {run_label} → Output: {science_label}")
    else:
        science_label = None
    logger.info(f"--- Pipeline Start --- Run Label: {run_label}")

    # ---- 2. GENERATE JOBS ----
    scheduler = LSTScheduler()
    t_start, t_end = scheduler.parse_range(args.range, args.date)
    jobs = scheduler.generate_jobs(t_start, t_end, args.override_range)

    # Submit high-priority science bands first
    priority = ['73MHz', '78MHz', '69MHz', '82MHz']
    all_subs = list(config.NODE_SUBBAND_MAP.keys())
    ordered_subs = priority + [s for s in all_subs if s not in priority]

    # ---- 3. SUBMIT SUBBAND JOBS ----
    submitted_lsts = []
    submitted_job_ids = []
    first_job_date_str = jobs[0]['start'].isot

    for job in jobs:
        submitted_lsts.append(job['lst_label'])
        for subband in ordered_subs:
            if subband in config.NODE_SUBBAND_MAP:
                job_id = submit_worker_job(subband, job, run_label, args, science_label)
                if job_id:
                    submitted_job_ids.append(job_id)
                time.sleep(0.2)

    logger.info(f"Submitted {len(submitted_job_ids)} Phase 1 jobs. Monitor with: squeue -u $USER")

    # ---- 4. PHASE 2: SCIENCE AGGREGATION ----
    # In resume_science mode, the submitted jobs ARE the final workers (no
    # Setup→Image chain), so Phase 2 depends directly on them with afterok.
    # In normal mode, each Pipe_ setup job spawns an Image_ job whose ID is
    # written to a file — a collector job reads those IDs and submits Science_Agg.
    if not args.skip_phase2 and submitted_job_ids:
        logger.info("Submitting Phase 2 Science Aggregation...")

        try:
            obs_date_dir = Time(first_job_date_str).datetime.strftime('%Y-%m-%d')
        except Exception:
            obs_date_dir = "unknown"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        science_script = os.path.join(script_dir, "post_process_science.py")

        for lst_label in set(submitted_lsts):
            p2_label = science_label if science_label else run_label
            run_dir = os.path.join(
                config.LUSTRE_ARCHIVE_DIR, lst_label, obs_date_dir, p2_label
            )

            sci_log_dir = os.path.join(run_dir, "logs")
            os.makedirs(sci_log_dir, exist_ok=True)
            sci_log_file = os.path.join(sci_log_dir, "Science_Aggregation.log")

            p2_cmd = f"python {science_script} --run_dir {run_dir}"
            if args.targets:
                p2_cmd += " --targets " + " ".join([os.path.abspath(t) for t in args.targets])
            if args.catalog:
                p2_cmd += f" --catalog {os.path.abspath(args.catalog)}"

            p2_wrap = (
                f"source ~/.bashrc && conda activate {config.REQUIRED_CONDA_ENV} && "
                f"export PYTHONPATH={script_dir}:$PYTHONPATH && "
                f"{p2_cmd}"
            )

            if args.resume_science:
                # Direct dependency: science-only workers have no child Image_ jobs
                dep_str = "afterok:" + ":".join(submitted_job_ids)
                logger.info(f"  resume_science: Phase 2 depends directly on {len(submitted_job_ids)} workers")

                slurm_cmd = [
                    'sbatch', '--job-name=Science_Agg', '--partition=general',
                    '--nodes=1', '--ntasks=1', '--cpus-per-task=16', '--mem=64G',
                    f'--dependency={dep_str}',
                    f'--output={sci_log_file}', f'--error={sci_log_file}',
                    '--wrap', p2_wrap
                ]
                logger.info(f"Submitting Science_Agg for {lst_label} (afterok on workers)")
                subprocess.run(slurm_cmd)

            else:
                # Normal mode: collector pattern (waits for Pipe_ jobs, reads Image_ IDs)
                collector_log = os.path.join(sci_log_dir, "SciAgg_Collector.log")
                dep_file = os.path.join(run_dir, ".image_job_ids")
                pipe_dep_str = "afterany:" + ":".join(submitted_job_ids)

                collector_script = (
                    f"source ~/.bashrc && conda activate {config.REQUIRED_CONDA_ENV} && "
                    f"export PYTHONPATH={script_dir}:$PYTHONPATH && "
                    f"if [ -f {dep_file} ]; then "
                    f"  IMAGE_IDS=$(cat {dep_file} | tr '\\n' ':' | sed 's/:$//'); "
                    f"  echo \"Image job IDs: $IMAGE_IDS\"; "
                    # Filter to only IDs still known to Slurm (completed jobs may be purged)
                    f"  VALID_IDS=''; "
                    f"  for JID in $(cat {dep_file}); do "
                    f"    if squeue -j $JID -h 2>/dev/null | grep -q .; then "
                    f"      VALID_IDS=\"${{VALID_IDS:+$VALID_IDS:}}$JID\"; "
                    f"    fi; "
                    f"  done; "
                    f"  if [ -n \"$VALID_IDS\" ]; then "
                    f"    echo \"Pending/running Image jobs: $VALID_IDS\"; "
                    f"    sbatch --job-name=Science_Agg --partition=general "
                    f"      --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G "
                    f"      --dependency=afterany:$VALID_IDS "
                    f"      --output={sci_log_file} --error={sci_log_file} "
                    f"      --wrap '{p2_cmd}'; "
                    f"  else "
                    f"    echo \"All Image jobs already completed — submitting immediately\"; "
                    f"    sbatch --job-name=Science_Agg --partition=general "
                    f"      --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G "
                    f"      --output={sci_log_file} --error={sci_log_file} "
                    f"      --wrap '{p2_cmd}'; "
                    f"  fi; "
                    f"else "
                    f"  echo 'WARNING: No .image_job_ids file found — submitting without dependency'; "
                    f"  sbatch --job-name=Science_Agg --partition=general "
                    f"    --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G "
                    f"    --output={sci_log_file} --error={sci_log_file} "
                    f"    --wrap '{p2_cmd}'; "
                    f"fi"
                )

                slurm_cmd = [
                    'sbatch', '--job-name=SciAgg_Collector', '--partition=general',
                    '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=1G',
                    '--time=00:05:00',
                    f'--dependency={pipe_dep_str}',
                    f'--output={collector_log}', f'--error={collector_log}',
                    '--wrap', collector_script
                ]
                logger.info(f"Submitting dependency collector for {lst_label} (waits for Pipe_ jobs)")
                subprocess.run(slurm_cmd)


if __name__ == "__main__":
    main()
