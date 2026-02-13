#!/usr/bin/env python3
"""
post_process_science.py  Phase 2 Science Aggregation

Runs AFTER all per-subband workers complete. Operates on Lustre.
Steps:
  1. Flux check (calibrator validation)
  2. Wideband stacking (inverse-variance weighted co-adds)
  3. Wideband transient search
  4. Gather detections from all subbands
  5. Email report

MERGED: New version as base (wideband stacking, thermal noise, SMTP auth).
"""
import argparse
import os
import glob
import logging
import pandas as pd
import smtplib
import shutil
import json
import numpy as np
from astropy.io import fits
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import flux_check_cutout

try:
    import transient_search as trans_tools
except ImportError:
    trans_tools = None

try:
    import solar_system_cutout
except ImportError:
    solar_system_cutout = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SCIENCE] - %(message)s')
logger = logging.getLogger()

# --- CONFIGURATION ---
EMAIL_RECIPIENT = "gh@astro.caltech.edu"
OUTPUT_CAT_DIR = "/lustre/gh/main/catalogs/"
SECRETS_FILE = os.path.expanduser("~/pipeline_cred.json")

EMAIL_REPORT_LINES = []
ATTACHMENT_FILES = []


# --- HELPER: NOISE ANALYSIS ---
def get_inner_rms(fits_path):
    """Calculates RMS from the inner 50% (area=25%) of the image."""
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data.squeeze()
            h, w = data.shape
            ch, cw = h // 2, w // 2
            rh, rw = h // 4, w // 4
            region = data[ch - rh:ch + rh, cw - rw:cw + rw]
            return np.nanstd(region)
    except Exception as e:
        logger.warning(f"Failed to calc RMS for {os.path.basename(fits_path)}: {e}")
        return np.nan


# --- STEP 1: WIDEBAND STACKING ---

def _stack_images(img_list, weights, out_path, history_str):
    """Inverse-variance weighted co-add of FITS images. Returns True on success."""
    if not img_list:
        return False
    try:
        data_sum = None
        weight_sum = 0.0
        ref_header = None

        for img, w in zip(img_list, weights):
            with fits.open(img) as hdul:
                data = hdul[0].data.squeeze()
                if data_sum is None:
                    data_sum = np.zeros_like(data, dtype=np.float64)
                    ref_header = hdul[0].header.copy()
                if data.shape != data_sum.shape:
                    continue
                data_sum += (data * w)
                weight_sum += w

        if weight_sum > 0:
            final_data = data_sum / weight_sum
            ref_header['BTYPE'] = 'Intensity'
            ref_header['HISTORY'] = history_str
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fits.writeto(out_path, final_data.astype(np.float32),
                         ref_header, overwrite=True)
            return True
    except Exception as e:
        logger.error(f"Stack failed for {out_path}: {e}")
    return False


def _make_3color_png(red_path, green_path, blue_path, out_png, title=""):
    """Generate a 3-color PNG from Red/Green/Blue FITS images."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        channels = []
        for fpath in [red_path, green_path, blue_path]:
            if fpath and os.path.exists(fpath):
                with fits.open(fpath) as hdul:
                    channels.append(hdul[0].data.squeeze().astype(np.float64))
            else:
                channels.append(None)

        # Need at least 2 channels
        valid = [c for c in channels if c is not None]
        if len(valid) < 2:
            logger.warning(f"3-color PNG needs =2 bands, got {len(valid)}  skipping {out_png}")
            return

        ref_shape = valid[0].shape
        for i in range(3):
            if channels[i] is None:
                channels[i] = np.zeros(ref_shape, dtype=np.float64)
            elif channels[i].shape != ref_shape:
                logger.warning(f"Shape mismatch in 3-color: {channels[i].shape} vs {ref_shape}")
                channels[i] = np.zeros(ref_shape, dtype=np.float64)

        # Normalize each channel: clip at [0.5th, 99.8th] percentile, scale 0-1
        normed = []
        for ch in channels:
            vmin, vmax = np.nanpercentile(ch, [0.5, 99.8])
            if vmax <= vmin:
                vmax = vmin + 1
            clipped = np.clip(ch, vmin, vmax)
            normed.append((clipped - vmin) / (vmax - vmin))

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
        logger.info(f"  Saved 3-color PNG: {os.path.basename(out_png)}")
    except Exception as e:
        logger.error(f"3-color PNG failed: {e}")


def run_wideband_stacking(run_dir, catalog_path):
    logger.info("Starting Wideband Co-addition...")

    bands = {
        'Red':   (18, 41),
        'Green': (41, 64),
        'Blue':  (64, 85)
    }

    wb_dir = os.path.join(run_dir, "Wideband")
    os.makedirs(wb_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Noise Analysis & Weighting (from Stokes V deep images)
    # ----------------------------------------------------------------
    noise_data = []
    v_images = glob.glob(os.path.join(run_dir, "*MHz", "V", "deep", "*_dewarped.fits"))

    if not v_images:
        logger.warning("No dewarped V images found for noise weighting  trying raw V deep.")
        v_images = glob.glob(os.path.join(run_dir, "*MHz", "V", "deep", "*image*.fits"))
        v_images = [f for f in v_images if "pbcorr" not in f and "dewarped" not in f]

    if not v_images:
        logger.warning("No V images found at all  cannot stack.")
        return

    logger.info(f"Analyzing noise in {len(v_images)} sub-bands...")
    for v_img in v_images:
        try:
            freq_str = v_img.split('/')[-4].replace('MHz', '')
            freq = float(freq_str)
            rms = get_inner_rms(v_img)
            if np.isfinite(rms) and rms > 0:
                noise_data.append({'freq': freq, 'rms': rms, 'weight': 1.0 / (rms ** 2)})
        except Exception:
            continue

    if not noise_data:
        logger.warning("No valid noise measurements  aborting stacking.")
        return

    df_noise = pd.DataFrame(noise_data)
    noise_csv = os.path.join(wb_dir, "thermal_noise.csv")
    df_noise.to_csv(noise_csv, index=False)
    logger.info(f"Saved thermal noise profile to {noise_csv}")

    # ----------------------------------------------------------------
    # 2. Define stacking targets
    # ----------------------------------------------------------------
    # A) Transient search images: Tapered, Dewarped, NO PB correction, Briggs 0
    #    - Stokes I deep + 10min, Stokes V deep + 10min
    #    suffix_kw is used for filename matching; robust_label is used for output naming/lookup
    transient_targets = [
        # (pol, category, taper_kw, suffix_kw, file_ext, is_raw, robust_label)
        ('I', 'deep',  'Taper', 'Robust-0-',  '_dewarped.fits', False, 'Robust-0'),
        ('I', '10min', 'Taper', '10min',       '_dewarped.fits', False, 'Robust-0'),
        ('V', 'deep',  'Taper', 'Deep',        '_dewarped.fits', False, 'Robust-0'),
        ('V', '10min', 'Taper', '10min',       '_dewarped.fits', False, 'Robust-0'),
    ]

    # B) Science images: NoTaper, dewarped, NOT PB-corrected
    #    - Stokes I deep: Briggs 0 and Briggs -0.75
    science_targets = [
        # (pol, category, taper_kw, suffix_kw, file_ext, is_raw, robust_label)
        ('I', 'deep', 'NoTaper', 'Robust-0-',    '_dewarped.fits', False, 'Robust-0'),
        ('I', 'deep', 'NoTaper', 'Robust-0.75-',  '_dewarped.fits', False, 'Robust-0.75'),
    ]

    # ----------------------------------------------------------------
    # 3. Stack per band
    # ----------------------------------------------------------------
    # Track produced files for transient search and 3-color PNGs
    wideband_files = {}  # key: (band, pol, cat, taper_label) -> path

    for b_name, (f_min, f_max) in bands.items():
        subset = df_noise[(df_noise['freq'] >= f_min) & (df_noise['freq'] < f_max)]
        if subset.empty:
            continue

        logger.info(f"--- Stacking Wideband {b_name} ({f_min}-{f_max} MHz, {len(subset)} sub-bands) ---")

        # A) Transient search products
        for pol, cat, taper_kw, suffix_kw, file_ext, is_raw, robust_label in transient_targets:
            img_list = []
            weights = []

            for _, row in subset.iterrows():
                freq_dir = os.path.join(run_dir, f"{int(row['freq'])}MHz", pol, cat)
                if not os.path.isdir(freq_dir):
                    continue

                # Find matching files
                candidates = glob.glob(os.path.join(freq_dir, f"*{suffix_kw}*{file_ext}"))
                # For dewarped transient images: exclude pbcorr
                if not is_raw:
                    candidates = [f for f in candidates
                                  if "pbcorr" not in os.path.basename(f)
                                  and "_dewarped_dewarped" not in os.path.basename(f)]
                else:
                    # For raw: exclude pbcorr and dewarped
                    candidates = [f for f in candidates
                                  if "pbcorr" not in os.path.basename(f)
                                  and "dewarped" not in os.path.basename(f)]
                # Taper filter
                if taper_kw == 'NoTaper':
                    candidates = [f for f in candidates if 'NoTaper' in os.path.basename(f)]
                else:
                    candidates = [f for f in candidates if 'NoTaper' not in os.path.basename(f)]

                # Robust filter: "Robust-0-" must not match "Robust-0.75"
                if suffix_kw == 'Robust-0-':
                    candidates = [f for f in candidates
                                  if 'Robust-0.75' not in os.path.basename(f)]

                if candidates:
                    # For 10min: add ALL interval images (not just the first)
                    if cat == '10min':
                        for c in sorted(candidates):
                            img_list.append(c)
                            weights.append(row['weight'])
                    else:
                        img_list.append(candidates[0])
                        weights.append(row['weight'])

            if not img_list:
                continue

            taper_str = "Taper" if taper_kw != 'NoTaper' else "NoTaper"
            robust_str = robust_label

            if cat == '10min':
                # Group 10min images by interval number and stack each separately
                import re
                interval_groups = {}
                for img, w in zip(img_list, weights):
                    bn = os.path.basename(img)
                    # Match interval tag like t0001, t0002, etc.
                    m = re.search(r'(t\d{4})', bn)
                    key = m.group(1) if m else 'all'
                    if key not in interval_groups:
                        interval_groups[key] = ([], [])
                    interval_groups[key][0].append(img)
                    interval_groups[key][1].append(w)

                for int_key, (int_imgs, int_weights) in sorted(interval_groups.items()):
                    out_name = f"Wideband_{b_name}_{pol}_{cat}_{taper_str}_{robust_str}_{int_key}.fits"
                    out_path = os.path.join(wb_dir, out_name)
                    history = (f"Wideband Stack: {b_name} ({f_min}-{f_max} MHz), "
                               f"{pol} {cat} {taper_str} {robust_str} {int_key}")
                    if _stack_images(int_imgs, int_weights, out_path, history):
                        logger.info(f"  Stacked: {out_name} ({len(int_imgs)} images)")
                        wideband_files[(b_name, pol, cat, taper_str, robust_str, int_key)] = out_path
            else:
                out_name = f"Wideband_{b_name}_{pol}_{cat}_{taper_str}_{robust_str}.fits"
                out_path = os.path.join(wb_dir, out_name)
                history = f"Wideband Stack: {b_name} ({f_min}-{f_max} MHz), {pol} {cat} {taper_str} {robust_str}"
                if _stack_images(img_list, weights, out_path, history):
                    logger.info(f"  Stacked: {out_name} ({len(img_list)} images)")
                    wideband_files[(b_name, pol, cat, taper_str, robust_str)] = out_path

        # B) Science products (NoTaper, dewarped, not PB-corrected)
        for pol, cat, taper_kw, suffix_kw, file_ext, is_raw, robust_label in science_targets:
            img_list = []
            weights = []

            for _, row in subset.iterrows():
                freq_dir = os.path.join(run_dir, f"{int(row['freq'])}MHz", pol, cat)
                if not os.path.isdir(freq_dir):
                    continue

                candidates = glob.glob(os.path.join(freq_dir, f"*{suffix_kw}*{file_ext}"))
                # Exclude pbcorr and double-dewarped
                candidates = [f for f in candidates
                              if "pbcorr" not in os.path.basename(f)
                              and "_dewarped_dewarped" not in os.path.basename(f)]
                if taper_kw == 'NoTaper':
                    candidates = [f for f in candidates if 'NoTaper' in os.path.basename(f)]

                if suffix_kw == 'Robust-0-':
                    candidates = [f for f in candidates
                                  if 'Robust-0.75' not in os.path.basename(f)]
                elif suffix_kw == 'Robust-0.75-':
                    candidates = [f for f in candidates
                                  if 'Robust-0.75' in os.path.basename(f)]

                if candidates:
                    img_list.append(candidates[0])
                    weights.append(row['weight'])

            if not img_list:
                continue

            taper_str = "NoTaper"
            robust_str = robust_label
            out_name = f"Wideband_{b_name}_{pol}_{cat}_{taper_str}_{robust_str}.fits"
            out_path = os.path.join(wb_dir, out_name)

            history = f"Wideband Stack: {b_name} ({f_min}-{f_max} MHz), {pol} {cat} {taper_str} {robust_str}"
            if _stack_images(img_list, weights, out_path, history):
                logger.info(f"  Stacked: {out_name} ({len(img_list)} images)")
                wideband_files[(b_name, pol, cat, taper_str, robust_str)] = out_path

    # ----------------------------------------------------------------
    # 4. 3-Color PNGs for NoTaper dewarped science images
    # ----------------------------------------------------------------
    logger.info("Generating 3-color PNGs...")

    # A) Mixed robust weighting for best visual result:
    #    Red (18-41 MHz): Robust -0.75 (higher resolution where beam is fattest)
    #    Green (41-64 MHz): Robust 0
    #    Blue (64-85 MHz): Robust 0
    r_mixed = wideband_files.get(('Red',   'I', 'deep', 'NoTaper', 'Robust-0.75'))
    g_mixed = wideband_files.get(('Green', 'I', 'deep', 'NoTaper', 'Robust-0'))
    b_mixed = wideband_files.get(('Blue',  'I', 'deep', 'NoTaper', 'Robust-0'))

    if any(p is not None for p in [r_mixed, g_mixed, b_mixed]):
        png_path = os.path.join(wb_dir, "Wideband_I_deep_NoTaper_mixed_3color.png")
        _make_3color_png(r_mixed, g_mixed, b_mixed, png_path,
                         title="Wideband Stokes I  Red:Robust-0.75 Green/Blue:Robust-0")
        ATTACHMENT_FILES.append(png_path)

    # B) Uniform robust weighting versions
    for robust_str in ['Robust-0', 'Robust-0.75']:
        r_path = wideband_files.get(('Red',   'I', 'deep', 'NoTaper', robust_str))
        g_path = wideband_files.get(('Green', 'I', 'deep', 'NoTaper', robust_str))
        b_path = wideband_files.get(('Blue',  'I', 'deep', 'NoTaper', robust_str))

        if any(p is not None for p in [r_path, g_path, b_path]):
            png_name = f"Wideband_I_deep_NoTaper_{robust_str}_3color.png"
            png_path = os.path.join(wb_dir, png_name)
            _make_3color_png(r_path, g_path, b_path, png_path,
                             title=f"Wideband Stokes I  NoTaper {robust_str}")
            ATTACHMENT_FILES.append(png_path)

    # ----------------------------------------------------------------
    # 5. Wideband Transient Search
    # ----------------------------------------------------------------
    if trans_tools and catalog_path:
        det_dir = os.path.join(run_dir, "detections")
        os.makedirs(det_dir, exist_ok=True)

        for b_name in bands:
            # --- Stokes V: no subtraction, search deep + 10min ---
            # Use wideband I deep (Taper, Briggs 0) for bright-source masking
            wb_i_deep = wideband_files.get((b_name, 'I', 'deep', 'Taper', 'Robust-0'))

            # V deep
            wb_v_deep = wideband_files.get((b_name, 'V', 'deep', 'Taper', 'Robust-0'))
            if wb_v_deep:
                logger.info(f"  Transient search: Wideband {b_name} V deep...")
                try:
                    trans_tools.run_test(
                        None, wb_v_deep, wb_i_deep, catalog_path,
                        output_dir=det_dir, mode='V'
                    )
                except Exception as e:
                    logger.error(f"  Wideband V transient search failed ({b_name} deep): {e}")

            # V 10min  search each interval individually
            v_10min_keys = [k for k in wideband_files
                           if len(k) >= 6 and k[0] == b_name and k[1] == 'V'
                           and k[2] == '10min']
            for k in sorted(v_10min_keys):
                wb_v_10 = wideband_files[k]
                int_tag = k[5] if len(k) > 5 else ''
                logger.info(f"  Transient search: Wideband {b_name} V 10min {int_tag}...")
                try:
                    trans_tools.run_test(
                        None, wb_v_10, wb_i_deep, catalog_path,
                        output_dir=det_dir, mode='V'
                    )
                except Exception as e:
                    logger.error(f"  Wideband V transient search failed ({b_name} 10min {int_tag}): {e}")

            # --- Stokes I: subtract deep from each 10min interval ---
            wb_i_deep_ref = wideband_files.get((b_name, 'I', 'deep', 'Taper', 'Robust-0'))
            i_10min_keys = [k for k in wideband_files
                           if len(k) >= 6 and k[0] == b_name and k[1] == 'I'
                           and k[2] == '10min']
            for k in sorted(i_10min_keys):
                wb_i_10 = wideband_files[k]
                int_tag = k[5] if len(k) > 5 else ''
                if wb_i_deep_ref:
                    logger.info(f"  Transient search: Wideband {b_name} I (10min - deep) {int_tag}...")
                    try:
                        trans_tools.run_test(
                            wb_i_deep_ref, wb_i_10, wb_i_deep_ref, catalog_path,
                            output_dir=det_dir, mode='I'
                        )
                    except Exception as e:
                        logger.error(f"  Wideband I transient search failed ({b_name} {int_tag}): {e}")
    else:
        if not trans_tools:
            logger.warning("transient_search module not available  skipping wideband transient search.")
        if not catalog_path:
            logger.info("No catalog specified  skipping wideband transient search.")


# --- STEP 2: FLUX CHECK ---
def run_flux_check(run_dir):
    logger.info("Running Flux Check (Lustre)...")
    try:
        flux_check_cutout.run_flux_check(run_dir)
        qa_dir = os.path.join(run_dir, "QA")
        for f in glob.glob(os.path.join(qa_dir, "flux_check*")):
            shutil.copy(f, OUTPUT_CAT_DIR)
            ATTACHMENT_FILES.append(f)
    except Exception as e:
        logger.error(f"Flux check failed: {e}")


# --- STEP 3: GATHER DETECTIONS ---
def gather_detections(run_dir):
    logger.info("Gathering Detections from nodes...")

    # --- A) Target photometry detections (per-subband + wideband) ---
    det_glob = os.path.join(run_dir, "*MHz", "detections", "*_photometry.csv")
    csvs = glob.glob(det_glob)
    # Also check wideband samples
    csvs.extend(glob.glob(os.path.join(run_dir, "Wideband", "*_photometry.csv")))

    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
            if 'Detection' in df.columns and df['Detection'].any():
                det_rows = df[df['Detection'] == True]
                for _, row in det_rows.iterrows():
                    tgt = row.get('Target', 'Unknown')
                    freq = row.get('Freq_MHz', 0)
                    flux = row.get('I_Src_Jy', row.get('I_Flux_Jy', 0))
                    EMAIL_REPORT_LINES.append(f"DETECTED: {tgt} @ {freq}MHz ({flux:.3f} Jy)")

                    # Try to find corresponding diagnostic PNG
                    base_csv = os.path.basename(csv_path)
                    png_name = base_csv.replace("_photometry.csv",
                                                f"_{row.get('Date', 'unk')}_Diagnostic.png")
                    png_path = os.path.join(os.path.dirname(csv_path), png_name)

                    if not os.path.exists(png_path):
                        png_path = csv_path.replace(".csv", ".png")

                    if os.path.exists(png_path):
                        ATTACHMENT_FILES.append(png_path)
        except Exception:
            pass

    # --- B) Transient detections ---
    # Scan per-subband transient directories
    for stokes in ['I', 'V']:
        trans_dirs = glob.glob(os.path.join(
            run_dir, "*MHz", "detections", "transients", stokes, "*"))
        # Also wideband
        trans_dirs.extend(glob.glob(os.path.join(
            run_dir, "detections", "transients", stokes, "*")))

        for jdir in trans_dirs:
            if not os.path.isdir(jdir):
                continue
            jname = os.path.basename(jdir)
            n_files = len(os.listdir(jdir))
            if n_files > 0:
                EMAIL_REPORT_LINES.append(
                    f"TRANSIENT: Stokes {stokes} candidate {jname} ({n_files} files)")
                # Attach first PNG as preview
                pngs = sorted(glob.glob(os.path.join(jdir, "*.png")))
                if pngs and len(ATTACHMENT_FILES) < 20:
                    ATTACHMENT_FILES.append(pngs[0])

    # --- C) Solar system detections ---
    ss_dirs = glob.glob(os.path.join(run_dir, "*MHz", "detections", "SolarSystem", "*"))
    ss_dirs.extend(glob.glob(os.path.join(run_dir, "detections", "SolarSystem", "*")))
    for ss_dir in ss_dirs:
        if os.path.isdir(ss_dir):
            body = os.path.basename(ss_dir)
            n_files = len(os.listdir(ss_dir))
            if n_files > 0:
                EMAIL_REPORT_LINES.append(f"SOLAR SYSTEM: {body} detected ({n_files} files)")


# --- STEP 4: EMAIL REPORT ---
def send_email_report(run_dir):
    # Try authenticated SMTP first, fall back to localhost
    creds = None
    if os.path.exists(SECRETS_FILE):
        try:
            with open(SECRETS_FILE) as f:
                creds = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read secrets file: {e}")

    msg = MIMEMultipart()
    msg['From'] = creds['email'] if creds else "pipeline@astro.caltech.edu"
    msg['To'] = EMAIL_RECIPIENT
    msg['Subject'] = f"OVRO-LWA Results: {os.path.basename(run_dir)}"

    body = "<h3>Pipeline Run Completed</h3>"
    body += f"<p>Run Directory: {run_dir}</p>"

    # Add Thermal Noise Summary if available
    noise_csv = os.path.join(run_dir, "Wideband", "thermal_noise.csv")
    if os.path.exists(noise_csv):
        try:
            df = pd.read_csv(noise_csv)
            body += "<h4>Thermal Noise Profile</h4><ul>"
            body += f"<li>Median RMS: {df['rms'].median() * 1000:.1f} mJy</li>"
            body += f"<li>Min RMS: {df['rms'].min() * 1000:.1f} mJy "
            body += f"(@ {df.loc[df['rms'].idxmin()]['freq']:.0f} MHz)</li>"
            body += "</ul>"
            ATTACHMENT_FILES.append(noise_csv)
        except Exception:
            pass

    body += "<h4>Detections</h4>"
    if EMAIL_REPORT_LINES:
        body += "<ul>" + "".join([f"<li>{l}</li>" for l in EMAIL_REPORT_LINES]) + "</ul>"
    else:
        body += "<p>No significant detections found.</p>"

    # Collect attachments with size limit
    total_size = 0
    final_atts = []
    unique_atts = list(set(ATTACHMENT_FILES))

    for f in unique_atts:
        if os.path.exists(f):
            s = os.path.getsize(f)
            if total_size + s < 24 * 1024 * 1024:
                total_size += s
                final_atts.append(f)

    msg.attach(MIMEText(body, 'html'))

    for f in final_atts:
        with open(f, "rb") as att:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(att.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(f)}")
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
        logger.info(f"Email sent successfully to {EMAIL_RECIPIENT}.")
    except Exception as e:
        logger.error(f"Email failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OVRO-LWA Phase 2 Science Aggregation")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--targets", nargs='+')
    parser.add_argument("--catalog")
    args = parser.parse_args()

    os.makedirs(OUTPUT_CAT_DIR, exist_ok=True)

    # Set up persistent file logging (in addition to stdout ? Slurm log)
    log_file = os.path.join(args.run_dir, "post_process_science.log")
    try:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - [SCIENCE] - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
        logger.info(f"Logging to {log_file}")
    except Exception as e:
        logger.warning(f"Could not set up file logging: {e}")

    logger.info(f"Run directory: {args.run_dir}")
    logger.info(f"Targets: {args.targets}")
    logger.info(f"Catalog: {args.catalog}")

    # 1. Standard Flux Check
    run_flux_check(args.run_dir)

    # 2. Wideband Stacking & Transient Search
    run_wideband_stacking(args.run_dir, args.catalog)

    # 3. Wideband Solar System Photometry
    if solar_system_cutout is not None:
        try:
            solar_system_cutout.process_wideband_solar_system(args.run_dir, logger=logger)
        except Exception as e:
            logger.error(f"Wideband solar system photometry failed: {e}")
    else:
        logger.warning("solar_system_cutout not available  skipping wideband solar system.")

    # 4. Gather & Report
    gather_detections(args.run_dir)
    send_email_report(args.run_dir)
