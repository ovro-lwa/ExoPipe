#!/usr/bin/env python3
"""
flow_cascade_dewarp.py  Optical Flow Cascade Dewarping for OVRO-LWA Phase 2

Replaces cascade_dewarp.py.  Uses optical flow (image_plane_correction
package) instead of BDSF source extraction + TPS warp screen.

  - Anchor: 73 MHz vs VLSS synthetic sky (with best PB model)
  - Cascade: frequency-by-frequency, dewarped neighbor as reference
  - 10-min: each snapshot against dewarped deep reference

Runs in the py310_orca conda environment (requires JAX).
Called from phase2_controller.py via subprocess:

  conda run -n py310_orca --no-capture-output python flow_cascade_dewarp.py \\
      --work_dir /fast/gh/main/phase2/10h/2024-12-22/Science_...

Writes _dewarped.fits files and a dewarp_results.json summary.
"""
import os
import sys

# JAX GPU memory: don't pre-allocate, cap at 40% per process.
# Must be set BEFORE importing JAX/jax/image_plane_correction.
# Use direct assignment (not setdefault) to override any inherited values.
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'

# Log GPU config for debugging
_gpu_dev = os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')
print(f"[GPU-DEBUG] PID={os.getpid()} CUDA_VISIBLE_DEVICES={_gpu_dev} "
      f"PREALLOCATE={os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']} "
      f"MEM_FRACTION={os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']}")

# Persist JAX JIT cache  avoids ~43s recompilation per subprocess call.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                       os.path.expanduser("~/.cache/jax"))

import argparse
import glob
import json
import time
import logging
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

warnings.filterwarnings('ignore', category=FutureWarning)


# =====================================================================
# CONSTANTS (embedded to avoid py38 pipeline_config dependency)
# =====================================================================

ALL_SUBBANDS = [
    '18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz',
    '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz'
]

ANCHOR_SUBBAND = '73MHz'

CASCADE_DOWN = [
    '69MHz', '64MHz', '59MHz', '55MHz', '50MHz', '46MHz',
    '41MHz', '36MHz', '32MHz', '27MHz', '23MHz', '18MHz'
]
CASCADE_UP = ['78MHz', '82MHz']

MAX_CONSECUTIVE_FAILURES = 3


# =====================================================================
# LOGGING
# =====================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [FLOW-DEWARP] - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('flow_dewarp')
sys.stdout.reconfigure(line_buffering=True)


# =====================================================================
# FILE DISCOVERY
# =====================================================================

def find_deep_image(work_dir, subband):
    """Find the deep tapered non-PBcorr Stokes I image for a subband.

    This is the primary image for flow computation  apparent flux
    (beam-attenuated), which matches the synthetic sky from calcflow.

    Returns path or None.
    """
    pat = os.path.join(work_dir, subband, "I", "deep", "*Taper*image*.fits")
    candidates = [f for f in glob.glob(pat)
                  if "NoTaper" not in os.path.basename(f)
                  and "pbcorr" not in os.path.basename(f)
                  and "_dewarped" not in os.path.basename(f)
                  and ".pybdsf" not in os.path.basename(f)]
    if not candidates:
        logger.warning(f"  No deep tapered raw image found for {subband}")
        return None
    # Prefer Robust-0.75 (primary science image)
    robust075 = [f for f in candidates if "Robust-0.75" in os.path.basename(f)]
    if robust075:
        return sorted(robust075)[0]
    return sorted(candidates)[0]


def find_dewarped_deep_image(work_dir, subband):
    """Find the dewarped deep tapered non-PBcorr Stokes I image.

    Used as reference for 10-minute interval dewarping.
    Returns path or None.
    """
    pat = os.path.join(work_dir, subband, "I", "deep",
                       "*Taper*_dewarped.fits")
    candidates = [f for f in glob.glob(pat)
                  if "NoTaper" not in os.path.basename(f)
                  and "pbcorr" not in os.path.basename(f)
                  and ".pybdsf" not in os.path.basename(f)
                  and "_dewarped_dewarped" not in os.path.basename(f)]
    if not candidates:
        return None
    robust075 = [f for f in candidates if "Robust-0.75" in os.path.basename(f)]
    if robust075:
        return sorted(robust075)[0]
    return sorted(candidates)[0]


def find_all_warpable(work_dir, subband, categories=None):
    """Find all FITS images that need dewarping for a subband.

    Includes I and V, image and pbcorr variants.
    Excludes already-dewarped, BDSF outputs, and non-image products.

    Parameters
    ----------
    categories : list of str, optional
        e.g. ['deep'] or ['deep', '10min']. Default: ['deep'].
    """
    base = os.path.join(work_dir, subband)
    cats = categories or ['deep']
    all_fits = []
    for pol in ['I', 'V']:
        for cat in cats:
            d = os.path.join(base, pol, cat)
            if not os.path.isdir(d):
                continue
            for f in glob.glob(os.path.join(d, "*.fits")):
                bn = os.path.basename(f)
                if "_dewarped" in bn:
                    continue
                if ".pybdsf" in bn:
                    continue
                if '-image' not in bn:
                    continue
                all_fits.append(f)
    return sorted(all_fits)


def find_10min_images(work_dir, subband):
    """Find non-PBcorr 10-min Stokes I snapshots for flow computation.

    Returns list of (image_path, timestamp) tuples, sorted by timestamp.
    """
    import re
    d = os.path.join(work_dir, subband, "I", "10min")
    if not os.path.isdir(d):
        return []

    images = []
    for f in sorted(glob.glob(os.path.join(d, "*Taper*image*.fits"))):
        bn = os.path.basename(f)
        if "_dewarped" in bn or ".pybdsf" in bn or "pbcorr" in bn:
            continue
        if "NoTaper" in bn:
            continue
        m = re.search(r'(\d{8}_\d{6})', bn)
        ts = m.group(1) if m else bn
        images.append((f, ts))
    return images


def find_interval_files(work_dir, subband, timestamp):
    """Find all files for a specific 10-min interval (I + V, all variants).

    Returns list of paths to warp.
    """
    files = []
    for pol in ['I', 'V']:
        d = os.path.join(work_dir, subband, pol, "10min")
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, f"*{timestamp}*.fits")):
            bn = os.path.basename(f)
            if "_dewarped" in bn or ".pybdsf" in bn:
                continue
            if '-image' not in bn:
                continue
            files.append(f)
    return sorted(files)


# =====================================================================
# FLOW HELPERS
# =====================================================================

def apply_flow_to_files(flow, file_list):
    """Apply a Flow object to multiple FITS files, writing _dewarped.fits.

    Preserves BUNIT, beam keywords (BMAJ/BMIN/BPA), and fixes the
    NAXIS structure to match the squeezed 2D output.

    Returns number of files successfully dewarped.
    """
    import jax.numpy as jnp

    n_warped = 0
    for fpath in file_list:
        out_path = fpath.replace('.fits', '_dewarped.fits')
        if os.path.exists(out_path):
            n_warped += 1
            continue
        try:
            with fits.open(fpath) as hdul:
                orig_header = hdul[0].header.copy()
                data = hdul[0].data.squeeze()

                # Sanitize non-finite pixels (flow.apply expects clean data)
                finite_mask = np.isfinite(data)
                if not np.all(finite_mask):
                    fill = float(np.median(data[finite_mask])) if np.any(finite_mask) else 0.0
                    data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill)

                # Apply flow (JAX)
                dewarped = np.asarray(flow.apply(jnp.array(data, dtype=jnp.float32)))

                # Build clean 2D header from original
                header = _fix_header_2d(orig_header)

                # Write
                fits.writeto(out_path, dewarped.astype(np.float32),
                             header, overwrite=True)
                n_warped += 1
        except Exception as e:
            logger.warning(f"    Flow apply failed for "
                           f"{os.path.basename(fpath)}: {e}")
    return n_warped


def _fix_header_2d(orig):
    """Build a clean 2D FITS header from a potentially 4D original.

    Preserves WCS for the celestial axes, beam keywords, BUNIT,
    observation metadata, and all non-structural keywords.
    Removes NAXIS3/4, CTYPE3/4, etc. so the header matches
    the squeezed 2D data array.

    Preserves frequency information regardless of which axis it's on
    (CTYPE3='FREQ' or CTYPE4='FREQ').
    """
    header = orig.copy()

    # Fix NAXIS to 2
    header['NAXIS'] = 2

    # Find and preserve frequency information from ANY axis (3 or 4)
    freq_hz = None
    freq_unit = None
    bandwidth = None
    for ax in [3, 4]:
        ctype = orig.get(f'CTYPE{ax}', '')
        if 'FREQ' in str(ctype).upper():
            freq_hz = orig.get(f'CRVAL{ax}')
            freq_unit = orig.get(f'CUNIT{ax}', 'Hz')
            bandwidth = orig.get(f'CDELT{ax}')
            break

    if freq_hz is not None and 'RESTFRQ' not in header:
        header['RESTFRQ'] = (freq_hz, 'Reference frequency [Hz]')
    # Also preserve as explicit keywords for tools that need them
    if freq_hz is not None:
        if 'CRVAL3' not in header or 'FREQ' not in str(orig.get('CTYPE3', '')).upper():
            header['OBSFREQ'] = (freq_hz, '[Hz] Observation frequency')
        if bandwidth is not None and 'BANDWDTH' not in header:
            header['BANDWDTH'] = (abs(bandwidth), '[Hz] Channel bandwidth')

    # Preserve Stokes info
    for ax in [3, 4]:
        ctype = orig.get(f'CTYPE{ax}', '')
        if 'STOKES' in str(ctype).upper():
            stokes_val = orig.get(f'CRVAL{ax}')
            if stokes_val is not None and 'STOKES' not in header:
                stokes_map = {1.0: 'I', 2.0: 'Q', 3.0: 'U', 4.0: 'V'}
                header['STOKES'] = (stokes_map.get(stokes_val, str(stokes_val)),
                                    'Stokes parameter')
            break

    # Remove higher-dimension axis keywords
    for ax in [3, 4]:
        for prefix in ['NAXIS', 'CTYPE', 'CRVAL', 'CDELT', 'CRPIX',
                        'CUNIT', 'CROTA']:
            key = f'{prefix}{ax}'
            if key in header:
                del header[key]

    # Also remove PC/CD matrix entries for axes 3,4
    for i in [1, 2, 3, 4]:
        for j in [3, 4]:
            for prefix in ['PC', 'CD']:
                key = f'{prefix}{i}_{j}'
                if key in header:
                    del header[key]
                key = f'{prefix}{j}_{i}'
                if key in header:
                    del header[key]

    # Ensure BUNIT is present  default to Jy/beam if missing
    if 'BUNIT' not in header:
        if 'BUNIT' in orig:
            header['BUNIT'] = orig['BUNIT']
        else:
            header['BUNIT'] = 'Jy/beam'
            header.comments['BUNIT'] = 'Assumed; not in original header'

    # Ensure beam keywords are present
    for key in ['BMAJ', 'BMIN', 'BPA']:
        if key not in header and key in orig:
            header[key] = orig[key]

    # Preserve time/observation metadata
    for key in ['DATE-OBS', 'DATE-END', 'DURATION', 'MJD-OBS', 'MJD-MID',
                'TIMESYS', 'LST-OBS', 'TELESCOP', 'INSTRUME', 'OBSERVER',
                'OBJECT', 'ORIGIN']:
        if key not in header and key in orig:
            header[key] = orig[key]
            if orig.comments[key]:
                header.comments[key] = orig.comments[key]

    # Mark as dewarped
    header['DEWARPED'] = (True, 'Optical flow dewarping applied')

    return header


def flow_stats_str(flow):
    """Format flow statistics for logging."""
    offsets = np.asarray(flow.offsets)
    magnitude = np.sqrt(offsets[:, :, 0]**2 + offsets[:, :, 1]**2)

    # Mask to center circle (FoV)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    circle = (xx - cx)**2 + (yy - cy)**2 < (min(h, w) * 0.4)**2
    mag_fov = magnitude[circle]

    return (f"median={np.median(mag_fov):.3f} pix, "
            f"MAD={np.median(np.abs(mag_fov - np.median(mag_fov)))*1.4826:.3f} pix, "
            f"max={np.max(mag_fov):.3f} pix")


# =====================================================================
# DIAGNOSTICS
# =====================================================================

def save_flow_diagnostic(flow, subband, ref_label, diag_dir, shape):
    """Save flow magnitude map as PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    offsets = np.asarray(flow.offsets)
    magnitude = np.sqrt(offsets[:, :, 0]**2 + offsets[:, :, 1]**2)

    # Mask outside FoV
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    horizon = dist > min(h, w) * 0.45

    display = magnitude.copy()
    display[horizon] = np.nan

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(display, origin='lower', cmap='inferno',
                   vmin=0, vmax=np.nanpercentile(display, 99))
    plt.colorbar(im, ax=ax, label='Shift (pixels)')
    ax.set_title(f'Optical Flow: {subband} (ref: {ref_label})\n'
                 f'Median: {np.nanmedian(display):.3f} pix')

    png_path = os.path.join(diag_dir, f"flow_map_{subband}.png")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"    Saved diagnostic: {os.path.basename(png_path)}")


# =====================================================================
# ANCHOR STEP: 73 MHz vs VLSS CATALOG
# =====================================================================

def run_anchor(work_dir, diag_dir, catalog_path):
    """Dewarp 73 MHz anchor using VLSS synthetic sky.

    Returns
    -------
    (dewarped_data, flow, success) : (np.ndarray or None, Flow or None, bool)
        dewarped_data is the dewarped 73 MHz image (numpy array)
        used as reference for the first cascade step.
    """
    import jax.numpy as jnp
    from image_plane_correction.flow import calcflow

    logger.info(f"\n{'='*50}")
    logger.info(f"ANCHOR: {ANCHOR_SUBBAND} vs VLSS catalog")
    logger.info(f"{'='*50}")

    img_path = find_deep_image(work_dir, ANCHOR_SUBBAND)
    if img_path is None:
        logger.error(f"  No image found for {ANCHOR_SUBBAND}")
        return None, None, False

    logger.info(f"  Image: {os.path.basename(img_path)}")

    # Read obs_date and freq from header (for logging)
    with fits.open(img_path) as hdul:
        hdr = hdul[0].header
        obs_date = hdr.get('DATE-OBS')
        freq_hz = hdr.get('CRVAL3', hdr.get('RESTFRQ', 73e6))
    logger.info(f"  obs_date={obs_date}, freq={freq_hz/1e6:.1f} MHz")

    t0 = time.time()
    try:
        # As of May 2026 the upstream image_plane_correction fixes resolve:
        #   - flow.py:358    calcflow now passes obs_date/freq_hz from header
        #   - catalogs.py:123  beam response returns 0 (not NaN) below horizon
        #   - catalogs.py      resolved-source filtering in catalog loading
        #   - catalogs.py      synthetic sky returned in Jy/beam (not Jy/pixel)
        #   - catalogs.py:279  min_flux handled in correct units
        #   - catalogs.py      Dec  0° gap in VLSS catalog fixed
        # Previous monkey patches (_patched_tsky, _patched_beam) removed.

        image, ref_sky, flow, dewarped = calcflow(
            img_path,
            cleaned=True,
            use_best_pb_model=True,
            qa=False,
            write=False,
            catalog="VLSSR",
            catalog_path=catalog_path,
        )[:4]
    except Exception as e:
        import traceback
        logger.error(f"  calcflow failed for anchor: {e}")
        logger.error(traceback.format_exc())
        return None, None, False

    dt = time.time() - t0
    logger.info(f"  Flow computed in {dt:.1f}s  {flow_stats_str(flow)}")

    # Apply flow to all deep images for this subband
    files = find_all_warpable(work_dir, ANCHOR_SUBBAND, ['deep'])
    n = apply_flow_to_files(flow, files)
    logger.info(f"  Dewarped {n}/{len(files)} files for {ANCHOR_SUBBAND}")

    # Save diagnostic
    shape = np.asarray(image).shape
    save_flow_diagnostic(flow, ANCHOR_SUBBAND, 'VLSS', diag_dir, shape)

    return np.asarray(dewarped), flow, True


# =====================================================================
# CASCADE STEP: SUBBAND vs DEWARPED NEIGHBOR
# =====================================================================

def run_cascade_step(work_dir, subband, reference_data, ref_label, diag_dir):
    """Dewarp one subband using the dewarped neighbor as reference.

    Parameters
    ----------
    reference_data : np.ndarray
        Dewarped image from the previous cascade step (2D array).
    ref_label : str
        Label for logging, e.g. '73MHz'.

    Returns
    -------
    (dewarped_data, success) : (np.ndarray or None, bool)
    """
    import jax.numpy as jnp
    from image_plane_correction.flow import calcflow

    img_path = find_deep_image(work_dir, subband)
    if img_path is None:
        return None, False

    logger.info(f"  {subband} (ref: {ref_label}): "
                f"{os.path.basename(img_path)}")

    t0 = time.time()
    try:
        image, _, flow, dewarped = calcflow(
            img_path,
            reference_sky=jnp.array(reference_data, dtype=jnp.float32),
            qa=False,
            write=False,
        )[:4]
    except Exception as e:
        logger.error(f"    calcflow failed: {e}")
        return None, False

    dt = time.time() - t0
    dewarped_np = np.asarray(dewarped)

    # Sanity check: flow field not all NaN
    offsets = np.asarray(flow.offsets)
    if not np.any(np.isfinite(offsets)):
        logger.error(f"    Flow field is all NaN  skipping {subband}")
        return None, False

    logger.info(f"    {flow_stats_str(flow)} [{dt:.1f}s]")

    # Apply flow to all deep images
    files = find_all_warpable(work_dir, subband, ['deep'])
    n = apply_flow_to_files(flow, files)
    logger.info(f"    Dewarped {n}/{len(files)} files")

    save_flow_diagnostic(flow, subband, ref_label, diag_dir,
                         dewarped_np.shape)
    return dewarped_np, True


# =====================================================================
# IONOSPHERE EVOLUTION MOVIE
# =====================================================================

def _make_ionosphere_movie(movie_frames, diag_dir):
    """Generate an animated GIF of ionospheric distortion at 73 MHz.

    Parameters
    ----------
    movie_frames : list of dict
        Each dict has 'timestamp', 'magnitude' (2D array), 'median_shift'.
        Sorted by timestamp.
    diag_dir : str
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import imageio
    except ImportError as e:
        logger.warning(f"  Cannot make movie  missing dependency: {e}")
        return

    if len(movie_frames) < 3:
        logger.warning(f"  Only {len(movie_frames)} frames  "
                       f"need =3 for movie")
        return

    logger.info(f"\n--- Ionosphere Evolution Movie "
                f"({len(movie_frames)} frames) ---")

    # Mask outside FoV
    h, w = movie_frames[0]['magnitude'].shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    horizon = np.sqrt((xx - cx)**2 + (yy - cy)**2) > min(h, w) * 0.45

    # Global color scale
    all_mag = []
    for fr in movie_frames:
        mag = fr['magnitude'].copy()
        mag[horizon] = np.nan
        all_mag.append(mag)
    vmax = np.nanpercentile(np.concatenate([m.ravel() for m in all_mag]), 99)

    # Render frames
    frames_rgb = []
    for fr, mag_display in zip(movie_frames, all_mag):
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(mag_display, origin='lower', cmap='inferno',
                       vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Shift (pixels)', shrink=0.8)

        ts = fr['timestamp']
        time_label = (f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} "
                      f"{ts[9:11]}:{ts[11:13]}:{ts[13:15]} UTC")
        ax.set_title(f'Ionosphere: 73MHz  {time_label}\n'
                     f'Median shift = {fr["median_shift"]:.3f} pix')
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.asarray(buf)[:, :, :3].copy()  # RGBA ? RGB
        frames_rgb.append(frame)
        plt.close(fig)

    # Save GIF
    gif_path = os.path.join(diag_dir, "ionosphere_evolution_73MHz.gif")
    imageio.mimsave(gif_path, frames_rgb, duration=1.0, loop=0)
    logger.info(f"  Saved: {gif_path} ({len(frames_rgb)} frames)")

    # Individual PNGs
    for i, (fr, mag_display) in enumerate(zip(movie_frames, all_mag)):
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(mag_display, origin='lower', cmap='inferno',
                       vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Shift (pixels)', shrink=0.8)
        ts = fr['timestamp']
        time_label = (f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} "
                      f"{ts[9:11]}:{ts[11:13]}:{ts[13:15]}")
        ax.set_title(f'Ionosphere: 73MHz  {time_label}')
        png_path = os.path.join(diag_dir,
                                f"iono_10min_73MHz_{i:02d}_{ts}.png")
        plt.savefig(png_path, dpi=100, bbox_inches='tight')
        plt.close(fig)


# =====================================================================
# 10-MINUTE INTERVAL DEWARPING
# =====================================================================

def run_10min_flow(work_dir, successful_subbands, diag_dir):
    """Dewarp 10-minute snapshots using dewarped deep as reference.

    Each snapshot gets its own flow because the ionosphere changes
    on ~10 minute timescales.  Collects 73 MHz flow fields in memory
    for the ionosphere evolution movie.
    """
    import jax.numpy as jnp
    from image_plane_correction.flow import calcflow

    logger.info(f"\n{'='*50}")
    logger.info("10-MINUTE INTERVAL DEWARPING")
    logger.info(f"{'='*50}")

    n_total = 0
    n_ok = 0
    n_fail = 0

    # Collect 73 MHz flow data for ionosphere movie
    movie_frames = []

    for subband in sorted(successful_subbands,
                          key=lambda x: int(x.replace('MHz', ''))):
        # Find dewarped deep reference
        ref_path = find_dewarped_deep_image(work_dir, subband)
        if ref_path is None:
            logger.warning(f"  {subband}: no dewarped deep reference  skipping")
            continue

        with fits.open(ref_path) as hdul:
            ref_data = hdul[0].data.squeeze().astype(np.float32)

        # Sanitize reference
        if not np.all(np.isfinite(ref_data)):
            fill = float(np.median(ref_data[np.isfinite(ref_data)]))
            ref_data = np.nan_to_num(ref_data, nan=fill, posinf=fill,
                                      neginf=fill)

        ref_jax = jnp.array(ref_data)

        # Find 10-min snapshots
        snapshots = find_10min_images(work_dir, subband)
        if not snapshots:
            continue

        logger.info(f"\n  {subband}: {len(snapshots)} intervals "
                    f"(ref: {os.path.basename(ref_path)})")

        for img_path, timestamp in snapshots:
            n_total += 1
            t0 = time.time()

            try:
                image, _, flow, dewarped = calcflow(
                    img_path,
                    reference_sky=ref_jax,
                    qa=False,
                    write=False,
                )[:4]
            except Exception as e:
                logger.warning(f"    {timestamp}: calcflow failed  {e}")
                n_fail += 1
                continue

            dt = time.time() - t0

            # Sanity check: flow field not all NaN
            offsets = np.asarray(flow.offsets)
            if not np.any(np.isfinite(offsets)):
                logger.warning(f"    {timestamp}: flow field all NaN  skipping")
                n_fail += 1
                continue

            # Apply to all files for this interval
            interval_files = find_interval_files(work_dir, subband,
                                                 timestamp)
            n_warped = apply_flow_to_files(flow, interval_files)
            logger.info(f"    {timestamp}: {n_warped} files, "
                        f"{flow_stats_str(flow)} [{dt:.1f}s]")
            n_ok += 1

            # Collect 73 MHz flow data for ionosphere movie
            if subband == '73MHz':
                magnitude = np.sqrt(offsets[:, :, 0]**2 +
                                    offsets[:, :, 1]**2)
                # Mask to FoV for stats
                h, w = magnitude.shape
                cy, cx = h // 2, w // 2
                yy, xx = np.ogrid[:h, :w]
                circle = ((xx - cx)**2 + (yy - cy)**2 <
                          (min(h, w) * 0.4)**2)
                movie_frames.append({
                    'timestamp': timestamp,
                    'magnitude': magnitude,
                    'median_shift': float(np.median(magnitude[circle])),
                })

    logger.info(f"\n  10-min summary: {n_ok} OK, {n_fail} failed "
                f"out of {n_total}")

    # Generate ionosphere evolution movie
    if movie_frames:
        movie_frames.sort(key=lambda x: x['timestamp'])
        _make_ionosphere_movie(movie_frames, diag_dir)


# =====================================================================
# MAIN CASCADE
# =====================================================================

def run_flow_cascade(work_dir, catalog_path):
    """Run the full optical flow cascade dewarping pipeline.

    Returns dict of {subband: bool} indicating success/failure.
    """
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("OPTICAL FLOW CASCADE DEWARPING")
    logger.info(f"  work_dir: {work_dir}")
    logger.info("=" * 60)

    diag_dir = os.path.join(work_dir, "Dewarp_Diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    results = {}

    # Check anchor subband exists
    anchor_dir = os.path.join(work_dir, ANCHOR_SUBBAND)
    if not os.path.isdir(anchor_dir):
        logger.error(f"Anchor subband {ANCHOR_SUBBAND} not found "
                     f"in {work_dir}")
        return results

    # -- Phase 1: Anchor (73 MHz vs VLSS) --
    dewarped_73, anchor_flow, anchor_ok = run_anchor(
        work_dir, diag_dir, catalog_path)
    results[ANCHOR_SUBBAND] = anchor_ok

    if not anchor_ok:
        logger.error("Anchor dewarping failed  aborting cascade")
        _write_results(work_dir, results)
        return results

    # -- Phase 2a: Cascade Down --
    logger.info(f"\n{'='*50}")
    logger.info("CASCADE DOWN")
    logger.info(f"{'='*50}")

    ref_data = dewarped_73
    ref_label = ANCHOR_SUBBAND
    consecutive_failures = 0

    for subband in CASCADE_DOWN:
        if not os.path.isdir(os.path.join(work_dir, subband)):
            logger.warning(f"  {subband}: directory not found  skipping")
            results[subband] = False
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(f"  {MAX_CONSECUTIVE_FAILURES} consecutive "
                             f"failures  halting cascade arm")
                break
            continue

        dewarped, ok = run_cascade_step(
            work_dir, subband, ref_data, ref_label, diag_dir)
        results[subband] = ok

        if ok:
            ref_data = dewarped
            ref_label = subband
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            logger.warning(f"  Next subband will bridge to {ref_label} "
                           f"(gap = {consecutive_failures})")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(f"  {MAX_CONSECUTIVE_FAILURES} consecutive "
                             f"failures  halting cascade arm")
                break

    # -- Phase 2b: Cascade Up --
    logger.info(f"\n{'='*50}")
    logger.info("CASCADE UP")
    logger.info(f"{'='*50}")

    ref_data = dewarped_73
    ref_label = ANCHOR_SUBBAND
    consecutive_failures = 0

    for subband in CASCADE_UP:
        if not os.path.isdir(os.path.join(work_dir, subband)):
            logger.warning(f"  {subband}: directory not found  skipping")
            results[subband] = False
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(f"  {MAX_CONSECUTIVE_FAILURES} consecutive "
                             f"failures  halting cascade arm")
                break
            continue

        dewarped, ok = run_cascade_step(
            work_dir, subband, ref_data, ref_label, diag_dir)
        results[subband] = ok

        if ok:
            ref_data = dewarped
            ref_label = subband
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(f"  {MAX_CONSECUTIVE_FAILURES} consecutive "
                             f"failures  halting cascade arm")
                break

    # -- Summary --
    n_ok = sum(1 for v in results.values() if v)
    n_fail = sum(1 for v in results.values() if not v)
    n_missing = len(ALL_SUBBANDS) - len(results)
    logger.info(f"\n{'='*50}")
    logger.info(f"CASCADE SUMMARY: {n_ok} OK, {n_fail} failed, "
                f"{n_missing} not attempted")
    logger.info(f"{'='*50}")

    # -- Phase 3: 10-minute intervals --
    successful = [sb for sb, ok in results.items() if ok]
    if successful:
        run_10min_flow(work_dir, successful, diag_dir)

    # Write results
    _write_results(work_dir, results)

    dt = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"FLOW CASCADE COMPLETE  {dt:.0f}s total")
    logger.info(f"{'='*60}")

    return results


def _write_results(work_dir, results):
    """Write results JSON for phase2_controller to parse."""
    out_path = os.path.join(work_dir, "dewarp_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Results: {out_path}")


# =====================================================================
# SINGLE-STEP ENTRY POINTS (called by cascade_dewarp.py via subprocess)
# =====================================================================

def run_single_cascade_step(work_dir, subband, ref_subband, diag_dir,
                            ref_image_path=None):
    """Dewarp one subband using the dewarped neighbor as reference.

    Called by cascade_dewarp.py when cascade_method='flow'.
    Reads the dewarped reference from disk, runs calcflow, applies
    flow to all deep images for the subband.

    Parameters
    ----------
    ref_image_path : str, optional
        Override auto-discovery with a specific reference image path
        (e.g., a regridded image for band-boundary crossings).

    Returns True on success.
    """
    logger.info(f"\n--- Single cascade step: {subband} (ref: {ref_subband}) ---")

    if ref_image_path and os.path.exists(ref_image_path):
        ref_path = ref_image_path
        logger.info(f"  Reference (override): {os.path.basename(ref_path)}")
    else:
        ref_path = find_dewarped_deep_image(work_dir, ref_subband)
        if ref_path is None:
            logger.error(f"  No dewarped reference found for {ref_subband}")
            return False
        logger.info(f"  Reference: {os.path.basename(ref_path)}")

    with fits.open(ref_path) as hdul:
        ref_data = hdul[0].data.squeeze().astype(np.float32)

    # Sanitize non-finite pixels
    if not np.all(np.isfinite(ref_data)):
        fill = float(np.median(ref_data[np.isfinite(ref_data)]))
        ref_data = np.nan_to_num(ref_data, nan=fill, posinf=fill, neginf=fill)

    _, ok = run_cascade_step(work_dir, subband, ref_data, ref_subband, diag_dir)
    return ok


def run_single_10min_step(work_dir, subband, target_path, reference_path,
                           diag_dir):
    """Dewarp a single 10-min interval using a reference image.

    Called by cascade_dewarp.py when cascade_method='flow'.
    Runs calcflow on the target vs reference, applies the flow to all
    files for this interval (I + V variants).

    Returns True on success.
    """
    import re
    import jax.numpy as jnp
    from image_plane_correction.flow import calcflow

    # Extract timestamp from target filename
    m = re.search(r'(\d{8}_\d{6})', os.path.basename(target_path))
    timestamp = m.group(1) if m else 'unknown'

    logger.info(f"  {subband} {timestamp}: {os.path.basename(target_path)}")

    # Read reference
    with fits.open(reference_path) as hdul:
        ref_data = hdul[0].data.squeeze().astype(np.float32)
    if not np.all(np.isfinite(ref_data)):
        fill = float(np.median(ref_data[np.isfinite(ref_data)]))
        ref_data = np.nan_to_num(ref_data, nan=fill, posinf=fill, neginf=fill)
    ref_jax = jnp.array(ref_data)

    t0 = time.time()
    try:
        image, _, flow, dewarped = calcflow(
            target_path,
            reference_sky=ref_jax,
            qa=False,
            write=False,
        )[:4]
    except Exception as e:
        logger.error(f"    calcflow failed: {e}")
        return False

    dt = time.time() - t0

    # Sanity check
    offsets = np.asarray(flow.offsets)
    if not np.any(np.isfinite(offsets)):
        logger.error(f"    Flow field all NaN")
        return False

    logger.info(f"    {flow_stats_str(flow)} [{dt:.1f}s]")

    # Save distortion map + flow offsets for 73 MHz (for ionosphere GIF)
    if subband == '73MHz':
        save_flow_diagnostic(flow, f"73MHz_10min_{timestamp}",
                             'deep_dewarped', diag_dir, ref_data.shape)
        # Save offsets for variability metric
        offsets = np.asarray(flow.offsets)
        npz_path = os.path.join(diag_dir,
                                f"flow_offsets_73MHz_10min_{timestamp}.npz")
        np.savez_compressed(npz_path,
                            offset_y=offsets[:, :, 0],
                            offset_x=offsets[:, :, 1])

    # Apply to all files for this interval
    interval_files = find_interval_files(work_dir, subband, timestamp)
    n = apply_flow_to_files(flow, interval_files)
    logger.info(f"    Dewarped {n}/{len(interval_files)} files")

    return True


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optical flow cascade dewarping for OVRO-LWA Phase 2")
    parser.add_argument("--work_dir", required=True,
                        help="Phase 2 working directory")
    parser.add_argument("--catalog_path",
                        default="/lustre/gh/calibration/pipeline/reference/surveys/vlssr_radecpeak_unresolved.txt",
                        help="Path to VLSSr catalog (unresolved sources only)")
    parser.add_argument("--pipeline_dir", default=None,
                        help="Path to pipeline scripts directory "
                             "(for pb_correct.py). Default: same dir "
                             "as this script.")
    parser.add_argument("--skip_10min", action='store_true',
                        help="Skip 10-minute interval dewarping")
    parser.add_argument("--anchor_only", action='store_true',
                        help="Run only the 73 MHz anchor step, then exit. "
                             "Used by cascade_dewarp.py when anchor_method='flow'.")
    parser.add_argument("--cascade_step", action='store_true',
                        help="Run a single cascade step: dewarp --subband "
                             "using --ref_subband as reference.")
    parser.add_argument("--tenmin_step", action='store_true',
                        help="Run a single 10-min interval: dewarp --target "
                             "using --reference as reference.")
    parser.add_argument("--subband", default=None,
                        help="Subband for --cascade_step or --10min_step")
    parser.add_argument("--ref_subband", default=None,
                        help="Reference subband for --cascade_step")
    parser.add_argument("--target", default=None,
                        help="Target FITS path for --10min_step")
    parser.add_argument("--reference", default=None,
                        help="Reference FITS path for --10min_step")
    parser.add_argument("--ref_image", default=None,
                        help="Override reference image path for --cascade_step "
                             "(e.g., regridded image for band-boundary crossing)")
    args = parser.parse_args()

    # Ensure pb_correct.py is importable
    pipeline_dir = args.pipeline_dir or os.path.dirname(
        os.path.abspath(__file__))
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)

    diag_dir = os.path.join(args.work_dir, "Dewarp_Diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    if args.anchor_only:
        _, _, ok = run_anchor(args.work_dir, diag_dir, args.catalog_path)
        sys.exit(0 if ok else 1)

    if args.cascade_step:
        if not args.subband or not args.ref_subband:
            parser.error("--cascade_step requires --subband and --ref_subband")
        ok = run_single_cascade_step(args.work_dir, args.subband,
                                      args.ref_subband, diag_dir,
                                      ref_image_path=args.ref_image)
        sys.exit(0 if ok else 1)

    if args.tenmin_step:
        if not args.subband or not args.target or not args.reference:
            parser.error("--10min_step requires --subband, --target, "
                         "and --reference")
        ok = run_single_10min_step(args.work_dir, args.subband,
                                    args.target, args.reference, diag_dir)
        sys.exit(0 if ok else 1)

    results = run_flow_cascade(args.work_dir, args.catalog_path)

    # Exit code: 0 if anchor succeeded, 1 otherwise
    sys.exit(0 if results.get(ANCHOR_SUBBAND, False) else 1)


if __name__ == "__main__":
    main()
