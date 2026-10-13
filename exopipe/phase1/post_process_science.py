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
from exopipe.phase3.lib.photometry import flux_check_cutout

try:
    import transient_search as trans_tools
except ImportError:
    trans_tools = None

try:
    from exopipe.phase3.lib.photometry import solar_system_cutout
except ImportError:
    solar_system_cutout = None

try:
    from exopipe.phase3.lib.photometry import cutout
except ImportError:
    cutout = None

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

def _weighted_center_freq_hz(freqs_mhz, weights):
    """Return inverse-variance weighted center frequency in Hz."""
    total_w = sum(weights)
    if total_w <= 0:
        return None
    return sum(f * w for f, w in zip(freqs_mhz, weights)) / total_w * 1e6


def _stack_images(img_list, weights, out_path, history_str, center_freq_hz=None):
    """Inverse-variance weighted co-add of FITS images. Returns True on success.

    Parameters
    ----------
    img_list : list of str
        Paths to FITS images to co-add.
    weights : list of float
        Inverse-variance weights (1/rms^2) for each image.
    out_path : str
        Output FITS path.
    history_str : str
        HISTORY card to add.
    center_freq_hz : float, optional
        Weighted center frequency in Hz.  Written to CRVAL3 (frequency axis)
        so the header reflects the true co-add center rather than inheriting
        the value from whichever image is read first.
    """
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

            # Ensure the output data is 2D but retains frequency metadata.
            # astropy.io.fits will set NAXIS=2 for 2D data, potentially
            # stripping higher-axis WCS keywords.  We force a minimal
            # frequency axis (NAXIS3=1) so the frequency information
            # survives, and reshape the data to (1, ny, nx).
            if center_freq_hz is not None:
                # Write to a 3D cube with NAXIS3=1 so CRVAL3 is preserved
                final_data = final_data[np.newaxis, :, :]  # (1, ny, nx)
                ref_header['NAXIS'] = 3
                ref_header['NAXIS3'] = 1
                ref_header['CTYPE3'] = 'FREQ'
                ref_header['CRVAL3'] = center_freq_hz
                ref_header['CDELT3'] = 1.0
                ref_header['CRPIX3'] = 1.0
                ref_header['CUNIT3'] = 'Hz'
                # Remove axis 4 if present (Stokes)
                for key in list(ref_header.keys()):
                    if key.endswith('4') and key[:5] in (
                            'NAXIS', 'CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT', 'CROTA'):
                        del ref_header[key]
            else:
                # Pure 2D output  strip axes 3 and 4
                for ax in [3, 4]:
                    for key in list(ref_header.keys()):
                        if key.endswith(str(ax)) and key[:5] in (
                                'NAXIS', 'CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT', 'CROTA'):
                            del ref_header[key]

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fits.writeto(out_path, final_data.astype(np.float32),
                         ref_header, overwrite=True)
            return True
    except Exception as e:
        logger.error(f"Stack failed for {out_path}: {e}")
    return False


def _make_3color_png(red_path, green_path, blue_path, out_png, title="",
                     red_freq=29.5, green_freq=52.5, blue_freq=74.5,
                     spectral_index=-0.7, horizon_mask_deg=5.0):
    """Generate a 3-color PNG from Red/Green/Blue FITS images.
    
    Applies a spectral index correction so that a source with the given
    spectral index appears white. Also masks pixels below the horizon.
    
    Args:
        red/green/blue_path: Paths to FITS images for each color channel.
        out_png: Output PNG path.
        title: Title string for the image.
        red/green/blue_freq: Central frequency of each band in MHz.
        spectral_index: Spectral index for "white" normalization (default -0.7).
        horizon_mask_deg: Mask pixels below this elevation (degrees). Set to 0 to disable.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        channels = []
        wcs_ref = None
        for fpath in [red_path, green_path, blue_path]:
            if fpath and os.path.exists(fpath):
                with fits.open(fpath) as hdul:
                    channels.append(hdul[0].data.squeeze().astype(np.float64))
                    if wcs_ref is None:
                        try:
                            wcs_ref = WCS(hdul[0].header).celestial
                        except Exception:
                            pass
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

        # Spectral index correction: scale each channel so a source with
        # the given spectral index (default -0.7) appears white.
        # Normalize to the geometric mean frequency.
        freqs = np.array([red_freq, green_freq, blue_freq])
        ref_freq = np.exp(np.mean(np.log(freqs)))  # geometric mean
        scale_factors = (freqs / ref_freq) ** spectral_index
        for i in range(3):
            channels[i] = channels[i] * scale_factors[i]

        # Horizon mask: compute elevation for each pixel, mask below threshold
        if horizon_mask_deg > 0 and wcs_ref is not None:
            try:
                ny, nx = ref_shape
                # Sample a grid (every 16th pixel for speed)
                step = 16
                yy, xx = np.mgrid[0:ny:step, 0:nx:step]
                sky = wcs_ref.pixel_to_world(xx.ravel(), yy.ravel())
                # Elevation = 90 - zenith angle; for an all-sky image centered at zenith,
                # use the angular distance from the image center as a proxy
                from astropy.coordinates import AltAz
                from astropy.time import Time
                # Use a simple geometric mask: distance from image center
                cx, cy = nx // 2, ny // 2
                center_sky = wcs_ref.pixel_to_world(cx, cy)
                seps = center_sky.separation(sky).deg.reshape(yy.shape)
                # At zenith, elevation = 90 - zenith_angle
                # Horizon is at ~90 deg from zenith
                el_grid = 90.0 - seps
                # Interpolate to full resolution
                from scipy.ndimage import zoom
                el_full = zoom(el_grid, (ny / el_grid.shape[0], nx / el_grid.shape[1]),
                               order=1)
                # Ensure correct shape after zoom
                el_full = el_full[:ny, :nx]
                horizon_mask = el_full < horizon_mask_deg
                for i in range(3):
                    channels[i][horizon_mask] = 0.0
            except Exception as e:
                logger.warning(f"Horizon mask failed: {e}")

        # Normalize each channel: clip at [0.5th, 99.8th] percentile, scale 0-1
        normed = []
        for ch in channels:
            nonzero = ch[ch != 0]
            if len(nonzero) > 0:
                vmin, vmax = np.nanpercentile(nonzero, [0.5, 99.8])
            else:
                vmin, vmax = 0, 1
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


def _generate_dewarp_summary(run_dir, qa_dir):
    """Read per-subband warp_stats JSON files and produce a cross-frequency diagnostic plot.

    Generates:
      QA/dewarp_vs_frequency.png    3-panel plot:
        Top:    Median ionospheric shift (arcmin) vs frequency with ??² model fit
        Middle: Median dx/dy components (arcmin) vs frequency
        Bottom: Residual from ??² fit (arcmin) vs frequency, flagging outliers
      QA/dewarp_summary.csv         tabulated stats per subband
    """
    import json as json_mod

    # Collect warp stats from all subbands
    stats_files = sorted(glob.glob(os.path.join(run_dir, "*MHz", "Dewarp_Diagnostics", "warp_stats_*.json")))
    if not stats_files:
        logger.info("No warp stats files found  skipping dewarp summary.")
        return

    records = []
    for sf in stats_files:
        try:
            with open(sf) as f:
                records.append(json_mod.load(f))
        except Exception:
            pass

    if len(records) < 3:
        logger.info(f"Only {len(records)} warp stats files  need at least 3 for diagnostic.")
        return

    records.sort(key=lambda r: r['freq_mhz'])
    freqs = np.array([r['freq_mhz'] for r in records])
    median_shift = np.array([r['median_shift_arcmin'] for r in records])
    std_shift = np.array([r['std_shift_arcmin'] for r in records])
    max_shift = np.array([r['max_shift_arcmin'] for r in records])
    median_dx = np.array([r['median_dx_pix'] * r['pix_scale_arcmin'] for r in records])
    median_dy = np.array([r['median_dy_pix'] * r['pix_scale_arcmin'] for r in records])

    # Save summary CSV
    os.makedirs(qa_dir, exist_ok=True)
    csv_path = os.path.join(qa_dir, "dewarp_summary.csv")
    try:
        import csv as csv_mod
        with open(csv_path, 'w', newline='') as f:
            writer = csv_mod.writer(f)
            writer.writerow(['freq_mhz', 'median_shift_arcmin', 'std_shift_arcmin',
                             'max_shift_arcmin', 'median_dx_arcmin', 'median_dy_arcmin'])
            for r in records:
                writer.writerow([
                    r['freq_mhz'],
                    f"{r['median_shift_arcmin']:.4f}",
                    f"{r['std_shift_arcmin']:.4f}",
                    f"{r['max_shift_arcmin']:.4f}",
                    f"{r['median_dx_pix'] * r['pix_scale_arcmin']:.4f}",
                    f"{r['median_dy_pix'] * r['pix_scale_arcmin']:.4f}",
                ])
        logger.info(f"Saved dewarp summary CSV: {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to write dewarp summary CSV: {e}")

    # Fit ??² model: shift = A * ??²
    # Using log-space linear fit: log(shift) = log(A) - 2*log(?)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        valid = median_shift > 0
        if np.sum(valid) < 3:
            logger.info("Not enough valid warp measurements for diagnostic plot.")
            return

        log_freq = np.log10(freqs[valid])
        log_shift = np.log10(median_shift[valid])

        # Fit: log(shift) = a + b * log(freq), expect b  -2
        coeffs = np.polyfit(log_freq, log_shift, 1)
        fitted_slope = coeffs[0]
        fitted_A = 10**coeffs[1]

        # Model curve
        freq_model = np.linspace(freqs.min() * 0.9, freqs.max() * 1.1, 100)
        shift_model = fitted_A * freq_model**fitted_slope

        # Residuals from fit
        shift_predicted = fitted_A * freqs**fitted_slope
        residuals = median_shift - shift_predicted

        # Flag outliers: > 3× median absolute residual
        med_resid = np.median(np.abs(residuals[valid]))
        outlier_thresh = 3.0 * med_resid if med_resid > 0 else 0.5
        outliers = np.abs(residuals) > outlier_thresh

        # --- Plot ---
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 2, 2]})

        # Panel 1: Median shift vs frequency with model
        ax = axes[0]
        ax.errorbar(freqs, median_shift, yerr=std_shift, fmt='o', color='#2196F3',
                     markersize=6, capsize=3, label='Median shift')
        ax.plot(freq_model, shift_model, '--', color='#FF5722', linewidth=1.5,
                label=f'Fit: A·?^{{{fitted_slope:.2f}}}')
        # Also show pure ??² for comparison
        A_ref = median_shift[valid][len(median_shift[valid])//2] * freqs[valid][len(freqs[valid])//2]**2
        ax.plot(freq_model, A_ref * freq_model**-2, ':', color='gray', alpha=0.5,
                label='Pure ??² reference')
        if np.any(outliers):
            ax.scatter(freqs[outliers], median_shift[outliers], s=120, facecolors='none',
                       edgecolors='red', linewidths=2, zorder=5, label='Outlier')
        ax.set_ylabel('Median Shift (arcmin)', fontsize=11)
        ax.set_title('Ionospheric Dewarping: Shift vs Frequency', fontsize=13)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Panel 2: dx/dy components
        ax = axes[1]
        ax.plot(freqs, median_dx, 's-', color='#4CAF50', markersize=5, label='Median ?x (RA)')
        ax.plot(freqs, median_dy, 'D-', color='#9C27B0', markersize=5, label='Median ?y (Dec)')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
        ax.set_ylabel('Shift Component (arcmin)', fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Panel 3: Residuals from fit
        ax = axes[2]
        colors = ['red' if o else '#2196F3' for o in outliers]
        ax.bar(freqs, residuals, width=2.5, color=colors, alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axhline(outlier_thresh, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axhline(-outlier_thresh, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_xlabel('Frequency (MHz)', fontsize=11)
        ax.set_ylabel('Residual (arcmin)', fontsize=11)
        ax.set_title(f'Residual from fit (slope={fitted_slope:.2f}, expect  -2.0)', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        png_path = os.path.join(qa_dir, "dewarp_vs_frequency.png")
        plt.savefig(png_path, dpi=120)
        plt.close()
        ATTACHMENT_FILES.append(png_path)
        logger.info(f"Saved dewarp diagnostic plot: {png_path}")
        if np.any(outliers):
            outlier_freqs = freqs[outliers]
            logger.warning(f"  Dewarp outliers at: {', '.join(f'{f:.0f} MHz' for f in outlier_freqs)}")

    except Exception as e:
        logger.warning(f"Dewarp summary plot failed: {e}")


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

    # Discover all subband directories
    sb_dirs = sorted(glob.glob(os.path.join(run_dir, "*MHz")))
    if not sb_dirs:
        logger.warning("No subband directories found  cannot stack.")
        return

    logger.info(f"Measuring noise across {len(sb_dirs)} sub-bands...")
    for sb_dir in sb_dirs:
        try:
            freq_str = os.path.basename(sb_dir).replace('MHz', '')
            freq = float(freq_str)
        except (ValueError, IndexError):
            continue

        # Try to find a V deep image in priority order
        v_img = None
        for pattern in [
            os.path.join(sb_dir, "V", "deep", "*Taper*_dewarped.fits"),
            os.path.join(sb_dir, "V", "deep", "*Taper*image*.fits"),
        ]:
            hits = [f for f in glob.glob(pattern)
                    if "pbcorr" not in f and "_dewarped_dewarped" not in f]
            if hits:
                v_img = hits[0]
                break

        if v_img is None:
            logger.warning(f"  {freq_str}MHz: No V deep image found  excluding from stacking")
            continue

        rms = get_inner_rms(v_img)
        if np.isfinite(rms) and rms > 0:
            noise_data.append({'freq': freq, 'rms': rms, 'weight': 1.0 / (rms ** 2)})
            logger.info(f"  {freq_str}MHz: RMS = {rms*1000:.2f} mJy/beam")
        else:
            logger.warning(f"  {freq_str}MHz: Invalid RMS ({rms})  excluding")

    if not noise_data:
        logger.warning("No valid noise measurements  aborting stacking.")
        return

    df_noise = pd.DataFrame(noise_data)
    noise_csv = os.path.join(wb_dir, "thermal_noise.csv")
    df_noise.to_csv(noise_csv, index=False)
    logger.info(f"Saved thermal noise profile to {noise_csv}")

    # Copy to QA and generate diagnostic plot
    qa_dir = os.path.join(run_dir, "QA")
    os.makedirs(qa_dir, exist_ok=True)
    qa_noise_csv = os.path.join(qa_dir, "thermal_noise.csv")
    shutil.copy2(noise_csv, qa_noise_csv)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(df_noise['freq'], df_noise['rms'] * 1000, 'o-', color='#2196F3',
                     markersize=6, linewidth=1.5)
        ax.set_xlabel('Frequency (MHz)', fontsize=12)
        ax.set_ylabel('RMS Noise (mJy/beam)', fontsize=12)
        ax.set_title('Thermal Noise vs Frequency', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(15, 85)
        noise_png = os.path.join(qa_dir, "thermal_noise_vs_freq.png")
        plt.tight_layout()
        plt.savefig(noise_png, dpi=120)
        plt.close()
        ATTACHMENT_FILES.append(noise_png)
        logger.info(f"Saved thermal noise plot to {noise_png}")
    except Exception as e:
        logger.warning(f"Thermal noise plot failed: {e}")

    # Generate dewarping cross-frequency diagnostic
    _generate_dewarp_summary(run_dir, qa_dir)

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

    # B) Science images: dewarped, NOT PB-corrected
    #    - NoTaper Stokes I deep: Briggs 0 and Briggs -0.75
    #    - Taper Stokes I deep: Briggs -0.75 (for 3-color)
    science_targets = [
        # (pol, category, taper_kw, suffix_kw, file_ext, is_raw, robust_label)
        ('I', 'deep', 'NoTaper', 'Robust-0-',    '_dewarped.fits', False, 'Robust-0'),
        ('I', 'deep', 'NoTaper', 'Robust-0.75-',  '_dewarped.fits', False, 'Robust-0.75'),
        ('I', 'deep', 'Taper',   'Robust-0.75-',  '_dewarped.fits', False, 'Robust-0.75'),
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

        # Compute inverse-variance weighted center frequency for this band
        band_center_hz = _weighted_center_freq_hz(
            subset['freq'].tolist(), subset['weight'].tolist())

        # A) Transient search products
        for pol, cat, taper_kw, suffix_kw, file_ext, is_raw, robust_label in transient_targets:
            img_list = []
            weights = []

            for _, row in subset.iterrows():
                freq_dir = os.path.join(run_dir, f"{int(row['freq'])}MHz", pol, cat)
                if not os.path.isdir(freq_dir):
                    continue

                # Find matching files  prefer dewarped, fall back to raw
                candidates = glob.glob(os.path.join(freq_dir, f"*{suffix_kw}*{file_ext}"))
                # For dewarped transient images: exclude pbcorr
                if not is_raw:
                    candidates = [f for f in candidates
                                  if "pbcorr" not in os.path.basename(f)
                                  and "_dewarped_dewarped" not in os.path.basename(f)]
                    # Fallback: if no dewarped images, use raw (non-dewarped, non-pbcorr)
                    if not candidates:
                        raw_candidates = glob.glob(os.path.join(freq_dir, f"*{suffix_kw}*image*.fits"))
                        candidates = [f for f in raw_candidates
                                      if "pbcorr" not in os.path.basename(f)
                                      and "dewarped" not in os.path.basename(f)]
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
                # Group 10min images by interval ordinal and stack each separately.
                # Within each subband, images are sorted by name (which sorts by
                # timestamp or interval number), then assigned ordinal tags
                # t0001, t0002, ... This ensures images from the same interval
                # across subbands are stacked together even if their timestamps
                # or naming conventions differ.
                import re
                interval_groups = {}

                # First, group candidates by subband frequency
                from collections import defaultdict
                per_freq = defaultdict(list)
                freq_weight = {}
                for img, w in zip(img_list, weights):
                    # Extract freq from path: .../{freq}MHz/...
                    parts = img.split('/')
                    freq_key = None
                    for p in parts:
                        if p.endswith('MHz'):
                            freq_key = p
                            break
                    if freq_key is None:
                        freq_key = 'unknown'
                    per_freq[freq_key].append(img)
                    freq_weight[freq_key] = w

                # Assign ordinal tags within each subband
                for freq_key, imgs in per_freq.items():
                    w = freq_weight[freq_key]
                    for idx, img in enumerate(sorted(imgs)):
                        ordinal_tag = f"t{idx+1:04d}"
                        if ordinal_tag not in interval_groups:
                            interval_groups[ordinal_tag] = ([], [])
                        interval_groups[ordinal_tag][0].append(img)
                        interval_groups[ordinal_tag][1].append(w)

                for int_key, (int_imgs, int_weights) in sorted(interval_groups.items()):
                    out_name = f"Wideband_{b_name}_{pol}_{cat}_{taper_str}_{robust_str}_{int_key}.fits"
                    out_path = os.path.join(wb_dir, out_name)
                    history = (f"Wideband Stack: {b_name} ({f_min}-{f_max} MHz), "
                               f"{pol} {cat} {taper_str} {robust_str} {int_key}")
                    if _stack_images(int_imgs, int_weights, out_path, history,
                                     center_freq_hz=band_center_hz):
                        logger.info(f"  Stacked: {out_name} ({len(int_imgs)} images)")
                        wideband_files[(b_name, pol, cat, taper_str, robust_str, int_key)] = out_path
            else:
                out_name = f"Wideband_{b_name}_{pol}_{cat}_{taper_str}_{robust_str}.fits"
                out_path = os.path.join(wb_dir, out_name)
                history = f"Wideband Stack: {b_name} ({f_min}-{f_max} MHz), {pol} {cat} {taper_str} {robust_str}"
                if _stack_images(img_list, weights, out_path, history,
                                 center_freq_hz=band_center_hz):
                    logger.info(f"  Stacked: {out_name} ({len(img_list)} images)")
                    wideband_files[(b_name, pol, cat, taper_str, robust_str)] = out_path

        # B) Science products (dewarped, not PB-corrected)
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
                # Fallback: if no dewarped images, use raw (non-dewarped, non-pbcorr)
                if not candidates:
                    raw_candidates = glob.glob(os.path.join(freq_dir, f"*{suffix_kw}*image*.fits"))
                    candidates = [f for f in raw_candidates
                                  if "pbcorr" not in os.path.basename(f)
                                  and "dewarped" not in os.path.basename(f)]
                # Taper filter
                if taper_kw == 'NoTaper':
                    candidates = [f for f in candidates if 'NoTaper' in os.path.basename(f)]
                else:
                    candidates = [f for f in candidates if 'NoTaper' not in os.path.basename(f)]

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

            taper_str = "Taper" if taper_kw != 'NoTaper' else "NoTaper"
            robust_str = robust_label
            out_name = f"Wideband_{b_name}_{pol}_{cat}_{taper_str}_{robust_str}.fits"
            out_path = os.path.join(wb_dir, out_name)

            history = f"Wideband Stack: {b_name} ({f_min}-{f_max} MHz), {pol} {cat} {taper_str} {robust_str}"
            if _stack_images(img_list, weights, out_path, history,
                             center_freq_hz=band_center_hz):
                logger.info(f"  Stacked: {out_name} ({len(img_list)} images)")
                wideband_files[(b_name, pol, cat, taper_str, robust_str)] = out_path

    # ----------------------------------------------------------------
    # 3b. Full-band co-adds (all subbands, inverse-variance weighted)
    # ----------------------------------------------------------------
    # Products requested:
    #   V  Robust 0,      Taper,   dewarped  (deep)
    #   I  Robust 0,      Taper,   dewarped  (deep)
    #   I  Robust -0.75,  Taper,   dewarped  (deep)
    #   I  Robust 0,      NoTaper, dewarped  (deep)
    #   I  Robust -0.75,  NoTaper, dewarped  (deep)
    fullband_targets = [
        # (pol, taper_kw, suffix_kw,       file_ext,          robust_label)
        ('V', 'Taper',   'Robust-0-',      '_dewarped.fits',  'Robust-0'),
        ('I', 'Taper',   'Robust-0-',      '_dewarped.fits',  'Robust-0'),
        ('I', 'Taper',   'Robust-0.75-',   '_dewarped.fits',  'Robust-0.75'),
        ('I', 'NoTaper', 'Robust-0-',      '_dewarped.fits',  'Robust-0'),
        ('I', 'NoTaper', 'Robust-0.75-',   '_dewarped.fits',  'Robust-0.75'),
    ]

    full_center_hz = _weighted_center_freq_hz(
        df_noise['freq'].tolist(), df_noise['weight'].tolist())
    full_f_min = int(df_noise['freq'].min())
    full_f_max = int(df_noise['freq'].max())
    logger.info(f"--- Stacking Full-band ({full_f_min}-{full_f_max} MHz, "
                f"{len(df_noise)} sub-bands, center {full_center_hz/1e6:.1f} MHz) ---")

    for pol, taper_kw, suffix_kw, file_ext, robust_label in fullband_targets:
        img_list = []
        weights = []

        for _, row in df_noise.iterrows():
            freq_dir = os.path.join(run_dir, f"{int(row['freq'])}MHz", pol, "deep")
            if not os.path.isdir(freq_dir):
                continue

            candidates = glob.glob(os.path.join(freq_dir, f"*{suffix_kw}*{file_ext}"))
            candidates = [f for f in candidates
                          if "pbcorr" not in os.path.basename(f)
                          and "_dewarped_dewarped" not in os.path.basename(f)]
            # Fallback to non-dewarped
            if not candidates:
                raw_candidates = glob.glob(os.path.join(freq_dir, f"*{suffix_kw}*image*.fits"))
                candidates = [f for f in raw_candidates
                              if "pbcorr" not in os.path.basename(f)
                              and "dewarped" not in os.path.basename(f)]
            # Taper filter
            if taper_kw == 'NoTaper':
                candidates = [f for f in candidates if 'NoTaper' in os.path.basename(f)]
            else:
                candidates = [f for f in candidates if 'NoTaper' not in os.path.basename(f)]
            # Robust filter
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

        taper_str = "Taper" if taper_kw != 'NoTaper' else "NoTaper"
        out_name = f"Wideband_Full_{pol}_deep_{taper_str}_{robust_label}.fits"
        out_path = os.path.join(wb_dir, out_name)
        history = (f"Wideband Stack: Full ({full_f_min}-{full_f_max} MHz), "
                   f"{pol} deep {taper_str} {robust_label}")
        if _stack_images(img_list, weights, out_path, history,
                         center_freq_hz=full_center_hz):
            logger.info(f"  Stacked: {out_name} ({len(img_list)} images)")
            wideband_files[('Full', pol, 'deep', taper_str, robust_label)] = out_path

    # ----------------------------------------------------------------
    # 4. 3-Color PNGs for dewarped science images
    # ----------------------------------------------------------------
    logger.info("Generating 3-color PNGs...")

    # A) Taper Robust -0.75 (dewarped, not pbcorr)
    r_taper = wideband_files.get(('Red',   'I', 'deep', 'Taper', 'Robust-0.75'))
    g_taper = wideband_files.get(('Green', 'I', 'deep', 'Taper', 'Robust-0.75'))
    b_taper = wideband_files.get(('Blue',  'I', 'deep', 'Taper', 'Robust-0.75'))

    if any(p is not None for p in [r_taper, g_taper, b_taper]):
        png_path = os.path.join(wb_dir, "Wideband_I_deep_Taper_Robust-0.75_3color.png")
        _make_3color_png(r_taper, g_taper, b_taper, png_path,
                         title="Wideband Stokes I \u2014 Taper Robust-0.75")

    # B) NoTaper mixed robust weighting:
    #    Red (18-41 MHz): Robust -0.75 (higher resolution where beam is largest)
    #    Green (41-64 MHz): Robust 0
    #    Blue (64-85 MHz): Robust 0
    r_mixed = wideband_files.get(('Red',   'I', 'deep', 'NoTaper', 'Robust-0.75'))
    g_mixed = wideband_files.get(('Green', 'I', 'deep', 'NoTaper', 'Robust-0'))
    b_mixed = wideband_files.get(('Blue',  'I', 'deep', 'NoTaper', 'Robust-0'))

    if any(p is not None for p in [r_mixed, g_mixed, b_mixed]):
        png_path = os.path.join(wb_dir, "Wideband_I_deep_NoTaper_mixed_3color.png")
        _make_3color_png(r_mixed, g_mixed, b_mixed, png_path,
                         title="Wideband Stokes I \u2014 Red:Robust-0.75 Green/Blue:Robust-0")

    # C) Uniform robust weighting versions
    for robust_str in ['Robust-0', 'Robust-0.75']:
        r_path = wideband_files.get(('Red',   'I', 'deep', 'NoTaper', robust_str))
        g_path = wideband_files.get(('Green', 'I', 'deep', 'NoTaper', robust_str))
        b_path = wideband_files.get(('Blue',  'I', 'deep', 'NoTaper', robust_str))

        if any(p is not None for p in [r_path, g_path, b_path]):
            png_name = f"Wideband_I_deep_NoTaper_{robust_str}_3color.png"
            png_path = os.path.join(wb_dir, png_name)
            _make_3color_png(r_path, g_path, b_path, png_path,
                             title=f"Wideband Stokes I \u2014 NoTaper {robust_str}")

    # ----------------------------------------------------------------
    # 5. Wideband Transient Search
    # ----------------------------------------------------------------
    if trans_tools and catalog_path:
        det_dir = os.path.join(run_dir, "detections", "wideband")
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
                    results = trans_tools.run_test(
                        None, wb_v_deep, wb_i_deep, catalog_path,
                        output_dir=det_dir, mode='V'
                    )
                    logger.info(f"    -> {len(results)} candidate(s)")
                    for r in results:
                        logger.info(f"      {r['match_info']}: SNR={r['snr']:.1f} "
                                    f"Flux={r['flux']:.3f} Jy El={r['el']:.1f}")
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
                    results = trans_tools.run_test(
                        None, wb_v_10, wb_i_deep, catalog_path,
                        output_dir=det_dir, mode='V'
                    )
                    logger.info(f"    -> {len(results)} candidate(s)")
                    for r in results:
                        logger.info(f"      {r['match_info']}: SNR={r['snr']:.1f} "
                                    f"Flux={r['flux']:.3f} Jy El={r['el']:.1f}")
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
                        results = trans_tools.run_test(
                            wb_i_deep_ref, wb_i_10, wb_i_deep_ref, catalog_path,
                            output_dir=det_dir, mode='I'
                        )
                        logger.info(f"    -> {len(results)} candidate(s)")
                        for r in results:
                            logger.info(f"      {r['match_info']}: SNR={r['snr']:.1f} "
                                        f"Flux={r['flux']:.3f} Jy El={r['el']:.1f}")
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
        # Only attach the diagnostic PNG, not the CSV
        diag_png = os.path.join(qa_dir, "flux_check_diagnostic.png")
        if os.path.exists(diag_png):
            ATTACHMENT_FILES.append(diag_png)
    except Exception as e:
        logger.error(f"Flux check failed: {e}")


# --- STEP 2b: WIDEBAND TARGET PHOTOMETRY ---
def run_wideband_target_photometry(run_dir, target_files):
    """Run target photometry on wideband stacked images.

    For each wideband band (Red/Green/Blue/Full), create a temporary directory
    structure that cutout.find_all_images() expects (I/deep/, V/deep/, I/10min/,
    V/10min/) using symlinks to the actual wideband FITS files, then call
    cutout.process_target() for each target.
    """
    if cutout is None:
        logger.warning("cutout module not available  skipping wideband target photometry.")
        return
    if not target_files:
        logger.info("No target files specified  skipping wideband target photometry.")
        return

    logger.info("--- Wideband Target Photometry ---")
    wb_dir = os.path.join(run_dir, "Wideband")
    if not os.path.isdir(wb_dir):
        logger.warning("No Wideband directory found  skipping wideband target photometry.")
        return

    # Band definitions: name -> freq label for archiving
    band_defs = {
        'Red': '18_41MHz',
        'Green': '41_64MHz',
        'Blue': '64_85MHz',
        'Full': '18_85MHz',
    }

    import tempfile

    for band_name, freq_label in band_defs.items():
        # Find wideband images for this band
        # Deep I: prefer Taper Robust-0.75, fall back to Robust-0
        i_deep_075 = os.path.join(wb_dir, f"Wideband_{band_name}_I_deep_Taper_Robust-0.75.fits")
        i_deep_0 = os.path.join(wb_dir, f"Wideband_{band_name}_I_deep_Taper_Robust-0.fits")
        i_deep = i_deep_075 if os.path.exists(i_deep_075) else (
            i_deep_0 if os.path.exists(i_deep_0) else None)

        v_deep = os.path.join(wb_dir, f"Wideband_{band_name}_V_deep_Taper_Robust-0.fits")
        if not os.path.exists(v_deep):
            v_deep = None

        if i_deep is None or v_deep is None:
            logger.info(f"  Skipping {band_name}: missing deep I or V image")
            continue

        logger.info(f"  Processing {band_name} ({freq_label}):")
        logger.info(f"    I deep: {os.path.basename(i_deep)}")
        logger.info(f"    V deep: {os.path.basename(v_deep)}")

        # Create temporary directory structure for cutout.find_all_images()
        tmp_dir = os.path.join(wb_dir, f"_tmp_{band_name}")
        for subdir in ['I/deep', 'V/deep', 'I/10min', 'V/10min']:
            os.makedirs(os.path.join(tmp_dir, subdir), exist_ok=True)

        # Symlink deep images with filenames that exactly match cutout's patterns.
        # cutout._find_best_image searches for:
        #   I deep: *I-Deep-Taper-Robust-0.75* or *I-Deep-Taper-Robust-0*
        #   V deep: *V-Taper-Deep*
        # The new "raw" fallback also catches *pattern*.fits without pbcorr/dewarped.
        robust_tag = "Robust-0.75" if "Robust-0.75" in os.path.basename(i_deep) else "Robust-0"
        i_link_name = f"Wideband_{band_name}_I-Deep-Taper-{robust_tag}.fits"
        v_link_name = f"Wideband_{band_name}_V-Taper-Deep-{robust_tag}.fits"

        i_link = os.path.join(tmp_dir, "I", "deep", i_link_name)
        v_link = os.path.join(tmp_dir, "V", "deep", v_link_name)
        for link, target in [(i_link, i_deep), (v_link, v_deep)]:
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(os.path.abspath(target), link)

        # Symlink 10min interval images if they exist
        # cutout looks for *10min* in I/10min/ and V/10min/
        for stokes in ['I', 'V']:
            pat = os.path.join(wb_dir,
                               f"Wideband_{band_name}_{stokes}_10min_Taper_Robust-0_t*.fits")
            for snap in sorted(glob.glob(pat)):
                # Keep original name  it contains "10min" which is enough
                link = os.path.join(tmp_dir, stokes, "10min", os.path.basename(snap))
                if os.path.islink(link) or os.path.exists(link):
                    os.remove(link)
                os.symlink(os.path.abspath(snap), link)

        # Output directories
        wb_samples = os.path.join(run_dir, "Wideband", "samples")
        wb_detects = os.path.join(run_dir, "Wideband", "detections")
        os.makedirs(wb_samples, exist_ok=True)
        os.makedirs(wb_detects, exist_ok=True)

        # Process each target file
        for t_file in target_files:
            if not os.path.exists(t_file):
                logger.warning(f"    Target file not found: {t_file}")
                continue
            try:
                s_name = os.path.splitext(os.path.basename(t_file))[0]
                targets = cutout.load_targets(t_file)
                logger.info(f"    {s_name}: {len(targets)} targets")
                for nm, crd, det_stokes, confusing_sources in targets:
                    try:
                        cutout.process_target(
                            tmp_dir, nm, crd, s_name,
                            wb_samples, wb_detects,
                            detection_stokes=det_stokes,
                            confusing_sources=confusing_sources
                        )
                    except Exception as e:
                        logger.error(f"    Target '{nm}' failed for {band_name}: {e}")
            except Exception as e:
                logger.error(f"    Failed to process {t_file} for {band_name}: {e}")

        # Clean up temp dir
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

    logger.info("Wideband target photometry complete.")


# --- STEP 2c: ARCHIVE WIDEBAND PRODUCTS ---
def archive_wideband_products(run_dir):
    """Archive wideband samples, detections, and transients to both:

    1. Run-local: run_dir/{band_freq}/samples/ and run_dir/{band_freq}/detections/
       Parallel to per-subband dirs like run_dir/55MHz/
    2. Centralized: /lustre/gh/main/samples/ and /lustre/gh/main/detections/
       With {band_freq} replacing the subband tag.
    """
    try:
        from exopipe.common.pipeline_config import LUSTRE_ARCHIVE_DIR
    except ImportError:
        LUSTRE_ARCHIVE_DIR = "/lustre/gh/main"

    logger.info("--- Archiving Wideband Products ---")
    wb_dir = os.path.join(run_dir, "Wideband")
    if not os.path.isdir(wb_dir):
        logger.warning("No Wideband directory  skipping archive.")
        return

    band_defs = {
        'Red': '18_41MHz',
        'Green': '41_64MHz',
        'Blue': '64_85MHz',
        'Full': '18_85MHz',
    }

    # --- Run-local archive ---
    # Create run_dir/{band_freq}/ directories parallel to subband dirs
    for band_name, freq_label in band_defs.items():
        band_dir = os.path.join(run_dir, freq_label)
        os.makedirs(band_dir, exist_ok=True)

        # Copy wideband FITS images for this band
        for f in glob.glob(os.path.join(wb_dir, f"Wideband_{band_name}_*.fits")):
            dest = os.path.join(band_dir, os.path.basename(f))
            if not os.path.exists(dest):
                shutil.copy2(f, dest)

    # Copy samples and detections to run-local band dirs
    wb_samples = os.path.join(wb_dir, "samples")
    wb_detects = os.path.join(wb_dir, "detections")

    if os.path.isdir(wb_samples):
        for sample_dir in glob.glob(os.path.join(wb_samples, "*")):
            if not os.path.isdir(sample_dir):
                continue
            sample_name = os.path.basename(sample_dir)
            for target_dir in glob.glob(os.path.join(sample_dir, "*")):
                if not os.path.isdir(target_dir):
                    continue
                target_name = os.path.basename(target_dir)
                for band_name, freq_label in band_defs.items():
                    band_target_dir = os.path.join(
                        run_dir, freq_label, "samples", sample_name, target_name)
                    # Check if any files match this band
                    band_files = []
                    for root, dirs, files in os.walk(target_dir):
                        for fn in files:
                            if band_name in fn:
                                band_files.append(os.path.join(root, fn))
                    if band_files:
                        os.makedirs(band_target_dir, exist_ok=True)
                        for f in band_files:
                            shutil.copy2(f, band_target_dir)
                    # Also copy photometry CSV (it spans all bands)
                    for csv_f in glob.glob(os.path.join(target_dir, "*.csv")):
                        os.makedirs(band_target_dir, exist_ok=True)
                        shutil.copy2(csv_f, band_target_dir)

    # Copy transient detections to run-local band dirs
    # Walk the wideband detections dir (from transient search)
    det_wb = os.path.join(run_dir, "detections", "wideband")
    for det_source in [wb_detects, det_wb]:
        if not os.path.isdir(det_source):
            continue
        for stokes in ['I', 'V']:
            trans_src = os.path.join(det_source, "transients", stokes)
            if not os.path.isdir(trans_src):
                continue
            for jdir in glob.glob(os.path.join(trans_src, "*")):
                if not os.path.isdir(jdir):
                    continue
                jname = os.path.basename(jdir)
                for band_name, freq_label in band_defs.items():
                    band_files = [f for f in os.listdir(jdir) if band_name in f]
                    if band_files:
                        dest = os.path.join(
                            run_dir, freq_label, "detections",
                            "transients", stokes, jname)
                        os.makedirs(dest, exist_ok=True)
                        for f in band_files:
                            shutil.copy2(os.path.join(jdir, f), dest)

        # Transient candidate CSVs
        for csv_f in glob.glob(os.path.join(det_source, "transients", "candidates_*.csv")):
            bn = os.path.basename(csv_f)
            for band_name, freq_label in band_defs.items():
                if band_name in bn:
                    dest_dir = os.path.join(
                        run_dir, freq_label, "detections", "transients")
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy2(csv_f, dest_dir)

        # Debug maps
        debug_src = os.path.join(det_source, "transients", "debug")
        if os.path.isdir(debug_src):
            for f in glob.glob(os.path.join(debug_src, "*")):
                bn = os.path.basename(f)
                for band_name, freq_label in band_defs.items():
                    if band_name in bn:
                        dest_dir = os.path.join(
                            run_dir, freq_label, "detections",
                            "transients", "debug")
                        os.makedirs(dest_dir, exist_ok=True)
                        shutil.copy2(f, dest_dir)

    logger.info("  Run-local wideband archive complete.")

    # --- Centralized archive ---
    lustre_samples_root = os.path.join(LUSTRE_ARCHIVE_DIR, "samples")
    lustre_det_root = os.path.join(LUSTRE_ARCHIVE_DIR, "detections")

    # Samples: /lustre/gh/main/samples/{sample}/{target}/{band_freq}/
    if os.path.isdir(wb_samples):
        for sample_dir in glob.glob(os.path.join(wb_samples, "*")):
            if not os.path.isdir(sample_dir):
                continue
            sample_name = os.path.basename(sample_dir)
            for target_dir in glob.glob(os.path.join(sample_dir, "*")):
                if not os.path.isdir(target_dir):
                    # Loose file at sample level
                    for band_name, freq_label in band_defs.items():
                        dest_dir = os.path.join(lustre_samples_root, sample_name, freq_label)
                        os.makedirs(dest_dir, exist_ok=True)
                        shutil.copy2(target_dir, dest_dir)
                    continue
                target_name = os.path.basename(target_dir)
                for band_name, freq_label in band_defs.items():
                    dest_dir = os.path.join(
                        lustre_samples_root, sample_name, target_name, freq_label)
                    for root, dirs, files in os.walk(target_dir):
                        for fn in files:
                            if band_name in fn or fn.endswith('.csv'):
                                os.makedirs(dest_dir, exist_ok=True)
                                shutil.copy2(os.path.join(root, fn), dest_dir)
        logger.info(f"  Centralized samples archived to {lustre_samples_root}")

    # Detections: /lustre/gh/main/detections/transients/{I,V}/{RA_hour}/{J-name}/{band_freq}/
    for det_source in [wb_detects, det_wb]:
        if not os.path.isdir(det_source):
            continue
        for stokes in ['I', 'V']:
            trans_src = os.path.join(det_source, "transients", stokes)
            if not os.path.isdir(trans_src):
                continue
            for jdir in glob.glob(os.path.join(trans_src, "*")):
                if not os.path.isdir(jdir):
                    continue
                jname = os.path.basename(jdir)
                ra_hour = jname[1:3] if len(jname) >= 3 else "00"
                for band_name, freq_label in band_defs.items():
                    band_files = [f for f in os.listdir(jdir) if band_name in f]
                    if band_files:
                        dest = os.path.join(
                            lustre_det_root, "transients", stokes,
                            ra_hour, jname, freq_label)
                        os.makedirs(dest, exist_ok=True)
                        for f in band_files:
                            shutil.copy2(os.path.join(jdir, f), dest)

    # Target-based detections (sample_name/target_name from cutout detection copies)
    if os.path.isdir(wb_detects):
        for item in glob.glob(os.path.join(wb_detects, "*")):
            bn = os.path.basename(item)
            if bn in ('transients', 'SolarSystem', 'debug', 'wideband'):
                continue
            if os.path.isdir(item):
                for target_dir in glob.glob(os.path.join(item, "*")):
                    if not os.path.isdir(target_dir):
                        continue
                    target_name = os.path.basename(target_dir)
                    for band_name, freq_label in band_defs.items():
                        dest_dir = os.path.join(
                            lustre_det_root, bn, target_name, freq_label)
                        for fn in os.listdir(target_dir):
                            if band_name in fn or fn.endswith('.csv'):
                                os.makedirs(dest_dir, exist_ok=True)
                                shutil.copy2(
                                    os.path.join(target_dir, fn), dest_dir)

    logger.info(f"  Centralized detections archived to {lustre_det_root}")


# --- STEP 3b: ARCHIVE PER-SUBBAND PRODUCTS TO RUN-LOCAL ---
# (Per-subband workers already archive to centralized. This adds samples/detections
#  to the run-local directory which was previously missing.)
def archive_subband_to_runlocal(run_dir):
    """Copy per-subband samples/ and detections/ from NVMe output into the
    run-local Lustre directory.  This is called from post_process_science
    because by this point all subband workers have completed and their
    products exist on Lustre in the centralized paths.

    The per-subband workers already write to:
      run_dir/{subband}/samples/  and run_dir/{subband}/detections/
    during their archive step (Step D in process_subband.py).
    If those dirs are missing, we can reconstruct from the centralized archive.
    """
    # Per-subband workers now copy samples/detections to run_dir/{subband}/
    # in their own archive step.  Nothing more to do here  this function
    # exists as a placeholder in case we discover gaps.
    pass


# --- STEP 4: GATHER DETECTIONS ---
def gather_detections(run_dir):
    logger.info("Gathering Detections from nodes...")

    all_detection_rows = []  # For unified CSV output

    # --- A) Target photometry detections (per-subband + wideband) ---
    det_glob = os.path.join(run_dir, "*MHz", "detections", "*_photometry.csv")
    csvs = glob.glob(det_glob)
    # Also check wideband samples and detections (multiple levels deep)
    csvs.extend(glob.glob(os.path.join(run_dir, "Wideband", "*_photometry.csv")))
    csvs.extend(glob.glob(os.path.join(
        run_dir, "Wideband", "samples", "*", "*", "*_photometry.csv")))
    csvs.extend(glob.glob(os.path.join(
        run_dir, "Wideband", "detections", "*", "*", "*_photometry.csv")))

    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
            if 'Detection' in df.columns and df['Detection'].any():
                det_rows = df[df['Detection'] == True]
                for _, row in det_rows.iterrows():
                    tgt = row.get('Target', 'Unknown')
                    freq = row.get('Freq_MHz', 0)
                    flux = row.get('I_Src_Jy', row.get('I_Flux_Jy', 0))
                    ra = row.get('RA_deg', 0)
                    dec = row.get('Dec_deg', 0)
                    snr = row.get('SNR', row.get('I_SNR', 0))
                    sample = os.path.splitext(os.path.basename(csv_path))[0].replace('_photometry', '')
                    EMAIL_REPORT_LINES.append(
                        f"DETECTED: {tgt} @ {freq}MHz ({flux:.3f} Jy, SNR={snr:.1f})")
                    all_detection_rows.append({
                        'type': sample, 'name': tgt,
                        'ra_deg': ra, 'dec_deg': dec,
                        'ra_hms': '', 'dec_dms': '',
                        'el_deg': '', 'flux_jy': flux,
                        'snr': snr, 'stokes': 'I',
                        'image': f"{freq}MHz"
                    })

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

    # --- B) Transient detections (from candidate CSVs written by transient_search) ---
    # Collect all candidate CSVs from subbands and wideband
    trans_csvs = sorted(glob.glob(os.path.join(
        run_dir, "*MHz", "detections", "transients", "candidates_*.csv")))
    trans_csvs.extend(sorted(glob.glob(os.path.join(
        run_dir, "detections", "wideband", "transients", "candidates_*.csv"))))
    trans_csvs.extend(sorted(glob.glob(os.path.join(
        run_dir, "detections", "transients", "candidates_*.csv"))))

    # Group by image for structured reporting
    image_detections = {}  # image_name -> list of detection rows
    for csv_path in trans_csvs:
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                img = row.get('image', os.path.basename(csv_path))
                if img not in image_detections:
                    image_detections[img] = []
                image_detections[img].append(row.to_dict())
                all_detection_rows.append(row.to_dict())
        except Exception as e:
            logger.warning(f"Failed to read transient CSV {csv_path}: {e}")

    # Build structured report lines
    if image_detections:
        EMAIL_REPORT_LINES.append("")  # spacer
        EMAIL_REPORT_LINES.append("=== TRANSIENT SEARCH RESULTS ===")
        for img_name, dets in sorted(image_detections.items()):
            EMAIL_REPORT_LINES.append("")
            EMAIL_REPORT_LINES.append("-" * 124)
            EMAIL_REPORT_LINES.append(f"Image: {img_name} ({len(dets)} candidate(s))")
            EMAIL_REPORT_LINES.append(
                f"  {'Type':<35} {'Name':<15} {'RA (deg)':>9} {'Dec (deg)':>9} "
                f"{'RA (hms)':>14} {'Dec (dms)':>14} {'El':>5} {'Flux':>9} {'SNR':>6}")
            for d in dets:
                EMAIL_REPORT_LINES.append(
                    f"  {str(d.get('type','')):<35} {str(d.get('name','')):<15} "
                    f"{str(d.get('ra_deg','')):>9} {str(d.get('dec_deg','')):>9} "
                    f"{str(d.get('ra_hms','')):>14} {str(d.get('dec_dms','')):>14} "
                    f"{str(d.get('el_deg','')):>5} "
                    f"{str(d.get('flux_jy','')) + ' Jy':>9} {str(d.get('snr','')):>6}")

            # Attach first PNG from each transient candidate directory
            for d in dets[:3]:
                stk = d.get('stokes', 'V')
                # Look for cutout PNGs in the transient subdirectories
                for search_base in [
                    os.path.join(run_dir, "*MHz", "detections", "transients", stk, "*", "*.png"),
                    os.path.join(run_dir, "detections", "wideband", "transients", stk, "*", "*.png"),
                ]:
                    pngs = sorted(glob.glob(search_base))
                    for png in pngs:
                        if png not in ATTACHMENT_FILES and len(ATTACHMENT_FILES) < 20:
                            ATTACHMENT_FILES.append(png)
                            break
                    if len(ATTACHMENT_FILES) >= 20:
                        break

    # --- C) Solar system detections ---
    ss_dirs = glob.glob(os.path.join(run_dir, "*MHz", "detections", "SolarSystem", "*"))
    ss_dirs.extend(glob.glob(os.path.join(run_dir, "detections", "SolarSystem", "*")))
    for ss_dir in ss_dirs:
        if os.path.isdir(ss_dir):
            body_name = os.path.basename(ss_dir)
            n_files = len(os.listdir(ss_dir))
            if n_files > 0:
                EMAIL_REPORT_LINES.append(f"SOLAR SYSTEM: {body_name} detected ({n_files} files)")

    # --- D) Write unified detection CSV ---
    if all_detection_rows:
        unified_csv = os.path.join(run_dir, "all_detections.csv")
        fieldnames = ['type', 'name', 'ra_deg', 'dec_deg', 'ra_hms', 'dec_dms',
                      'el_deg', 'flux_jy', 'snr', 'stokes', 'image']
        try:
            import csv as csv_mod
            with open(unified_csv, 'w', newline='') as f:
                writer = csv_mod.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(all_detection_rows)
            logger.info(f"Wrote {len(all_detection_rows)} detections to {unified_csv}")
            ATTACHMENT_FILES.append(unified_csv)
        except Exception as e:
            logger.error(f"Failed to write unified detection CSV: {e}")


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

            # Generate thermal noise vs frequency plot
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.semilogy(df['freq'], df['rms'] * 1000, 'o-', color='#2196F3',
                            markersize=6, linewidth=1.5)
                ax.set_xlabel('Frequency (MHz)', fontsize=12)
                ax.set_ylabel('RMS Noise (mJy/beam)', fontsize=12)
                ax.set_title('Thermal Noise vs Frequency', fontsize=13)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(15, 85)
                noise_png = os.path.join(run_dir, "Wideband", "thermal_noise.png")
                plt.tight_layout()
                plt.savefig(noise_png, dpi=120)
                plt.close()
                ATTACHMENT_FILES.append(noise_png)
            except Exception as e:
                logger.warning(f"Thermal noise plot failed: {e}")
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

    # 3b. Wideband Target Photometry
    run_wideband_target_photometry(args.run_dir, args.targets)

    # 4. Archive wideband products to run-local and centralized paths
    archive_wideband_products(args.run_dir)

    # 5. Gather & Report
    gather_detections(args.run_dir)
    send_email_report(args.run_dir)
