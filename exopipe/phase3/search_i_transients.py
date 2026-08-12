#!/usr/bin/env python3
"""
search_i_transients.py  Stokes I Transient Search via Difference Imaging

Searches for transients by subtracting a deep (1-hr) reference from
10-minute snapshot targets. The pipeline:

  Phase 1: Setup (load images, masks, beam parameters)
  Phase 2: Flux scale correction (spatial gradient fit)
  Phase 3: Relative alignment (optical flow, optional)
  Phase 4: Subtract and extract positive peaks
  Phase 5: Local SNR assignment and filtering

Designed for intra-session subtraction: deep and 10-min images share
the same calibration and uv-coverage. The dominant residual is the
sidelobe sea from CLEAN depth differences, handled by local RMS.

Usage:
  python search_i_transients.py \\
      --tgt Green_I_10min_..._t0003.fits \\
      --ref Green_I_deep_....fits \\
      --catalog 50pc.csv \\
      --out_dir detections/

  # Skip alignment (faster, ~1% less sidelobe suppression):
  python search_i_transients.py \\
      --tgt ... --ref ... --no_align
"""
import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy.stats import mad_std
from scipy.ndimage import maximum_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D

import warnings
warnings.filterwarnings('ignore')

try:
    from exopipe.common.pipeline_config import OVRO_LOC
except ImportError:
    OVRO_LOC = EarthLocation(lat=37.23977727*u.deg, lon=-118.2816667*u.deg,
                             height=1222*u.m)

# ================================================================
# CONFIGURATION
# ================================================================
CONFIG = {
    'MIN_ELEVATION': 25.0,   # Gregg 2026-07-04: 25 deg for Stokes I (20-25 deg
                              # rim = faint-source leakage; 10h Blue 12-27 funnel)
    'SIGMA_THRESHOLD': 7.0,      # Detection threshold in local SNR
    'BDSF_THRESH_PIX': 5.0,      # BDSF pixel threshold (lower than final)
    'BDSF_THRESH_ISL': 3.0,      # BDSF island threshold

    # Flux scale measurement
    'SCALE_MIN_FLUX': 10.0,      # Jy  minimum reference flux for scale sources
    'SCALE_MAX_FLUX': 200.0,     # Jy  maximum (avoid resolved/saturated)
    'SCALE_MIN_EL': 45.0,        # deg  only use high-elevation sources

    # Local SNR (sector-minimum annulus MAD)
    'LOCAL_SNR_INNER_BEAMS': 2.0,
    'LOCAL_SNR_OUTER_BEAMS': 10.0,
    'LOCAL_SNR_N_SECTORS': 4,
    'LOCAL_SNR_MIN_PIX': 400,

    # Bright source proximity masking (tiered)
    'BRIGHT_MASK_FLUX_JY': 7.0,    # Mask around ref sources brighter than this
    'BRIGHT_MASK_RADIUS_BEAMS': 2.0, # Mask radius in beam widths
    'BRIGHT_EXCL_FLUX_JY': 70.0,   # Wider exclusion for very bright sources
    'BRIGHT_EXCL_RADIUS_DEG': 1.5, # Exclusion radius in degrees

    # A-team exclusion (enable with --ateam_mask)
    'ATEAM_MASK': True,
    'A_TEAM': {
        'Cas A': {'coord': SkyCoord('23h23m24s', '+58d48m54s'), 'radius': 15.0},
        'Cyg A': {'coord': SkyCoord('19h59m28s', '+40d44m02s'), 'radius': 15.0},
        'Tau A': {'coord': SkyCoord('05h34m31s', '+22d00m52s'), 'radius': 10.0},
        'Vir A': {'coord': SkyCoord('12h30m49s', '+12d23m28s'), 'radius': 12.0},
        'Hya A': {'coord': SkyCoord('09h18m06s', '-12d05m44s'), 'radius': 3.0},
        # Added 2026-06-10: dominated the yellow/red FP classes in the
        # 10h Green review (scatter residuals after subtraction)
        'Her A': {'coord': SkyCoord('16h51m08s', '+04d59m33s'), 'radius': 5.0},
    },

    # Alignment
    'CONDA_ENV': 'py310_orca',

    # Cutouts
    'CUTOUT_SIZE_DEG': 2.0,       # Cutout size (degrees on a side)
    'CUTOUT_SIGMA': 7.0,          # SNR threshold for cutout generation
    'MAX_CUTOUTS': 10,            # Maximum cutouts per search
}


# ================================================================
# UTILITIES
# ================================================================
class _Tee:
    """Write to both stdout and a log file."""
    def __init__(self, path):
        self.file = open(path, 'w')
        self.stdout = sys.__stdout__
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()


def load_image(path):
    with fits.open(path) as hdul:
        return hdul[0].data.squeeze().astype(np.float32), hdul[0].header


def get_midpoint_time(header, fits_path=None):
    duration = float(header.get('DURATION', 0.0))
    if duration <= 0 and fits_path:
        bn = os.path.basename(fits_path).lower()
        if '10min' in bn:
            duration = 600.0
        elif 'deep' in bn or '/deep/' in bn:
            duration = 3600.0
    if duration <= 0:
        duration = 3600.0
    return Time(header.get('DATE-OBS'), format='isot', scale='utc') + \
           TimeDelta(duration * 0.5, format='sec')


def get_freq_mhz(header, fits_path=None):
    """Extract frequency in MHz from header or filename."""
    freq = header.get('CRVAL3', header.get('RESTFRQ', 0))
    if freq > 1e6:
        return freq / 1e6
    if fits_path:
        import re
        m = re.search(r'(\d+)MHz', os.path.basename(fits_path))
        if m:
            return float(m.group(1))
    return None


def get_uniform_rms(data, mask):
    """Compute uniform RMS from image center, excluding masked pixels."""
    h, w = data.shape
    half = min(512, h // 4, w // 4)
    cy, cx = h // 2, w // 2
    region = data[cy-half:cy+half, cx-half:cx+half].copy()
    region[mask[cy-half:cy+half, cx-half:cx+half]] = np.nan
    return float(mad_std(region, ignore_nan=True))


# ================================================================
# PHASE 1: SETUP  MASKS
# ================================================================
def make_masks(header, shape, wcs, fits_path=None):
    """Build horizon mask, A-team mask, invalid WCS mask."""
    h, w = shape
    y_idx, x_idx = np.indices((h, w))
    ra, dec = wcs.all_pix2world(x_idx, y_idx, 0)

    obs_time = get_midpoint_time(header, fits_path)

    # Elevation grid
    coords = SkyCoord(ra, dec, unit='deg')
    altaz = coords.transform_to(AltAz(obstime=obs_time, location=OVRO_LOC))
    el_deg = altaz.alt.deg

    # Horizon mask
    horizon_mask = (el_deg < CONFIG['MIN_ELEVATION']) | np.isnan(ra) | np.isnan(dec)

    # Invalid WCS (NaN coordinates)
    invalid_wcs = np.isnan(ra) | np.isnan(dec)

    # A-team exclusion (only when enabled). P3_ATEAM_MASK_OFF: A-team ruled a
    # VETO stage, not a pre-detection blank (Gregg 2026-07-14, handoff §5.15g);
    # pilot runs set the env var — production default unchanged (mask ON).
    ateam_mask = np.zeros(shape, dtype=bool)
    if CONFIG['ATEAM_MASK'] and not os.environ.get('P3_ATEAM_MASK_OFF'):
        for name, info in CONFIG['A_TEAM'].items():
            sep = coords.separation(info['coord'])
            ateam_mask |= (sep.deg < info['radius'])

    return horizon_mask, ateam_mask, invalid_wcs, obs_time, el_deg


# ================================================================
# PHASE 2: FLUX SCALE CORRECTION
# ================================================================
def measure_flux_scale(tgt_data, ref_data, wcs, beam_pix, horizon_mask,
                       obs_time):
    """Measure flux scale between target and reference using bright sources.

    Finds local maxima in the reference, measures peak flux in both images,
    fits a spatial gradient model: scale(x,y) = a + b*x_norm + c*y_norm.

    Returns (scale_map, coeffs, n_sources).
    """
    from scipy.ndimage import maximum_filter as maxfilt

    h, w = ref_data.shape
    ap_r = max(int(np.ceil(beam_pix * 0.75)), 3)
    ms = max(int(beam_pix * 1.5), 5)
    ref_rms = get_uniform_rms(ref_data, horizon_mask)

    # Find bright peaks in reference
    lm = maxfilt(ref_data, size=ms)
    snr_map = ref_data / ref_rms
    peaks_y, peaks_x = np.where(
        (ref_data == lm) & (snr_map > 10) &
        (ref_data >= CONFIG['SCALE_MIN_FLUX']) &
        (ref_data < CONFIG['SCALE_MAX_FLUX']) &
        ~horizon_mask)

    ratios = []
    for sy, sx in zip(peaks_y, peaks_x):
        # Elevation cut
        try:
            coord = wcs.pixel_to_world(sx, sy)
            altaz = coord.transform_to(
                AltAz(obstime=obs_time, location=OVRO_LOC))
            if altaz.alt.deg < CONFIG['SCALE_MIN_EL']:
                continue
        except Exception:
            continue

        y1, y2 = max(0, sy - ap_r), min(h, sy + ap_r + 1)
        x1, x2 = max(0, sx - ap_r), min(w, sx + ap_r + 1)
        if y2 - y1 < 3 or x2 - x1 < 3:
            continue

        tgt_peak = float(np.nanmax(tgt_data[y1:y2, x1:x2]))
        ref_peak = float(np.nanmax(ref_data[y1:y2, x1:x2]))

        if ref_peak > 0 and tgt_peak > 0:
            # Reject resolved sources
            ref_patch = ref_data[y1:y2, x1:x2]
            total = float(np.nansum(ref_patch[ref_patch > 0.3 * ref_peak]))
            if total > 2.5 * ref_peak * (2 * ap_r + 1):
                continue
            ratios.append({
                'ratio': tgt_peak / ref_peak,
                'pix_x': int(sx), 'pix_y': int(sy),
            })

    if len(ratios) < 5:
        print(f"  WARNING: Only {len(ratios)} scale sources  using scale=1.0")
        return np.ones((h, w)), np.array([1.0, 0.0, 0.0]), len(ratios)

    # Fit spatial gradient: ratio = a + b*x_norm + c*y_norm
    xs = np.array([r['pix_x'] for r in ratios], dtype=float)
    ys = np.array([r['pix_y'] for r in ratios], dtype=float)
    vals = np.array([r['ratio'] for r in ratios])
    xn = 2.0 * xs / w - 1.0
    yn = 2.0 * ys / h - 1.0
    A = np.column_stack([np.ones_like(xn), xn, yn])

    # Iteratively reweighted least squares
    weights = np.ones_like(vals)
    for _ in range(3):
        Aw = A * weights[:, None]
        vw = vals * weights
        try:
            coeffs, _, _, _ = np.linalg.lstsq(Aw, vw, rcond=None)
        except Exception:
            coeffs = np.array([np.median(vals), 0, 0])
            break
        residuals = vals - A @ coeffs
        mad = np.median(np.abs(residuals))
        if mad > 0:
            weights = np.where(np.abs(residuals) < 3 * mad * 1.4826, 1.0, 0.1)

    # Build spatial scale map
    yg, xg = np.indices((h, w), dtype=float)
    xgn = 2.0 * xg / w - 1.0
    ygn = 2.0 * yg / h - 1.0
    scale_map = coeffs[0] + coeffs[1] * xgn + coeffs[2] * ygn

    return scale_map, coeffs, len(ratios)


# ================================================================
# PHASE 3: RELATIVE ALIGNMENT
# ================================================================
def run_alignment(tgt_path, ref_scaled_path, out_dir, conda_env):
    """Run optical flow alignment via align_images.py subprocess.

    Checks for pre-existing aligned images (from batch alignment phase)
    before running flow. Returns path to dewarped target, or None.
    """
    # Check for pre-existing aligned image (from batch flow phase)
    tgt_base = os.path.basename(tgt_path).replace('.fits', '')
    pre_aligned = [
        os.path.join(out_dir, f"{tgt_base}_dewarp.fits"),
        os.path.join(out_dir, os.path.basename(tgt_path).replace(
            '.fits', '_dewarp.fits')),
    ]
    for pa in pre_aligned:
        if os.path.exists(pa):
            print(f"  Using pre-aligned image: {os.path.basename(pa)}")
            return pa

    # GC-PRE-3 sibling (Gregg 2026-08-12): align_images.py lives in
    # exopipe/phase2/ in the refactored tree (was beside this script in the
    # frozen flat clv15 layout).
    align_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'phase2', 'align_images.py')
    if not os.path.exists(align_script):
        print("  align_images.py not found  skipping alignment")
        return None

    cmd = [
        "conda", "run", "-n", conda_env, "--no-capture-output",
        "python", align_script,
        "--target", tgt_path,
        "--reference", ref_scaled_path,
        "--out_dir", out_dir,
        "--no_qa",
    ]

    print(f"  Running optical flow alignment...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=600)
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                print(f"  [flow] {line}")
        if result.returncode != 0:
            print(f"  Alignment failed (rc={result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split('\n')[-5:]:
                    print(f"  [err] {line}")
            return None
    except subprocess.TimeoutExpired:
        print("  Alignment timed out (600s)")
        return None
    except FileNotFoundError:
        print("  conda not found  skipping alignment")
        return None

    # Find dewarped output
    tgt_base = os.path.basename(tgt_path).replace('.fits', '')
    candidates = [
        os.path.join(out_dir, f"{tgt_base}_dewarp.fits"),
        os.path.join(out_dir, os.path.basename(tgt_path).replace(
            '.fits', '_dewarp.fits')),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    dewarp_files = glob.glob(os.path.join(out_dir, '*_dewarp.fits'))
    if dewarp_files:
        return sorted(dewarp_files)[-1]

    print("  WARNING: No dewarped output found")
    return None


# ================================================================
# PHASE 4: EXTRACT POSITIVE PEAKS
# ================================================================
def _run_bdsf(path, thresh_pix, thresh_isl, rms_value=None, mean_map='default'):
    """Run PyBDSF on a FITS image, return list of source dicts."""
    import bdsf
    # Transients are POINT SOURCES — everything extended is a false positive
    # (Gregg). So: no à-trous wavelet decomposition of extended emission, flag
    # Gaussians larger than ~3 beams, and drop the costly adaptive rms box when
    # an rms is already supplied (it's the main BDSF choke on source-dense low
    # bands). These keep point-source detection while cutting BDSF time ~3-10x.
    kwargs = dict(thresh_pix=thresh_pix, thresh_isl=thresh_isl,
                  quiet=True, atrous_do=False,
                  flag_maxsize_bm=3.0, flagging_opts=True)
    if rms_value is not None:
        kwargs['rms_value'] = rms_value
        kwargs['rms_map'] = False
        kwargs['adaptive_rms_box'] = False   # rms supplied → skip the adaptive box
    else:
        kwargs['adaptive_rms_box'] = True
    if mean_map == 'zero':
        kwargs['mean_map'] = 'zero'

    img = bdsf.process_image(path, **kwargs)
    wcs_img = WCS(fits.getheader(path)).celestial
    sources = []
    for s in img.sources:
        try:
            ra_d = float(s.posn_sky_centroid[0])
            dec_d = float(s.posn_sky_centroid[1])
        except Exception:
            ra_d = float(s.posn_sky_max[0])
            dec_d = float(s.posn_sky_max[1])
        if np.isnan(ra_d) or np.isnan(dec_d):
            continue
        try:
            px, py = wcs_img.all_world2pix([[ra_d, dec_d]], 0)[0]
            px, py = int(round(px)), int(round(py))
        except Exception:
            continue
        sources.append({
            'coord': SkyCoord(ra_d, dec_d, unit='deg'),
            'flux': float(s.peak_flux_max),
            'total_flux': float(getattr(s, 'total_flux', s.peak_flux_max)),
            'snr': float(s.peak_flux_max / getattr(s, 'rms', 1e-3)),
            'pix_x': px, 'pix_y': py,
        })
    return sources


def extract_positive_peaks(diff_data, diff_header, horizon_mask, ateam_mask,
                           invalid_wcs_mask, wcs, obs_time, diff_rms, beam_pix):
    """Extract positive peaks from difference image using BDSF.

    Only searches for positive residuals (transient brighter in target
    than reference). Falls back to simple peak-finding if BDSF fails.
    """
    post_mask = horizon_mask | ateam_mask | invalid_wcs_mask
    candidates = []

    tmp_dir = tempfile.mkdtemp(prefix='bdsf_idiff_')
    try:
        # Write positive difference to temp FITS
        path = os.path.join(tmp_dir, 'diff_positive.fits')
        # Blank below-horizon/A-team region BEFORE BDSF (not just filter after):
        # otherwise BDSF deblends the low-elevation horizon ring and hangs on
        # noisy subband diffs. el cut = CONFIG['MIN_ELEVATION'].
        bdsf_data = np.where(post_mask, 0.0, diff_data).astype(np.float32)
        fits.writeto(path, bdsf_data, diff_header, overwrite=True)

        print(f"  Running BDSF on difference image (rms={diff_rms*1000:.2f} mJy)...")
        try:
            srcs = _run_bdsf(path,
                             thresh_pix=CONFIG['BDSF_THRESH_PIX'],
                             thresh_isl=CONFIG['BDSF_THRESH_ISL'],
                             rms_value=diff_rms,
                             mean_map='zero')

            # Filter by mask
            h, w = diff_data.shape
            srcs = [s for s in srcs
                    if 0 <= s['pix_y'] < h and 0 <= s['pix_x'] < w
                    and not post_mask[s['pix_y'], s['pix_x']]]

            # Only keep positive peaks
            srcs = [s for s in srcs if s['flux'] > 0]

            print(f"  BDSF: {len(srcs)} positive peaks above horizon/A-team")

        except Exception as e:
            print(f"  BDSF failed: {e}")
            print(f"  Falling back to simple peak-finding...")
            srcs = _fallback_peak_find(diff_data, diff_rms, beam_pix,
                                       post_mask, wcs)

        # Build candidate list
        for s in srcs:
            try:
                altaz = s['coord'].transform_to(
                    AltAz(obstime=obs_time, location=OVRO_LOC))
                el = float(altaz.alt.deg)
            except Exception:
                continue

            candidates.append({
                'coord': s['coord'],
                'flux': s['flux'],
                'snr': s['flux'] / diff_rms,  # initial global SNR
                'el': el,
                'pix_x': s['pix_x'],
                'pix_y': s['pix_y'],
                'status': 'PASS',
                'reject_reason': '',
            })

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    candidates.sort(key=lambda c: c['snr'], reverse=True)
    print(f"  Total candidates: {len(candidates)}")
    return candidates


def _fallback_peak_find(data, rms, beam_pix, mask, wcs):
    """Simple peak-finding fallback when BDSF fails."""
    snr_map = data / rms
    snr_map[mask] = 0
    ms = max(int(beam_pix * 1.5), 5)
    lm = maximum_filter(snr_map, size=ms)
    py, px = np.where((snr_map == lm) & (snr_map > CONFIG['BDSF_THRESH_PIX'])
                       & (data > 0))
    srcs = []
    for y, x in zip(py, px):
        try:
            co = wcs.pixel_to_world(int(x), int(y))
            if not np.isnan(co.ra.deg):
                srcs.append({
                    'coord': co,
                    'flux': float(data[y, x]),
                    'snr': float(snr_map[y, x]),
                    'pix_x': int(x), 'pix_y': int(y),
                })
        except Exception:
            pass
    print(f"  Fallback: {len(srcs)} peaks found")
    return srcs


# ================================================================
# PHASE 5: LOCAL SNR AND FILTERING
# ================================================================
def assign_local_snr(candidates, diff_data, exclusion_mask, beam_pix,
                     sigma_thresh):
    """Assign local SNR using sector-minimum annulus MAD.

    Same algorithm as the Stokes V pipeline. For each candidate,
    computes MAD in N angular sectors of an annulus, takes the minimum.
    This measures the local noise floor (sidelobe sea level) at each
    position, robust to bright source contamination in any one sector.
    """
    cfg = CONFIG
    h, w = diff_data.shape
    inner = cfg['LOCAL_SNR_INNER_BEAMS']
    outer = cfg['LOCAL_SNR_OUTER_BEAMS']
    n_sec = cfg['LOCAL_SNR_N_SECTORS']
    min_pix = cfg['LOCAL_SNR_MIN_PIX']

    box = int(np.ceil(outer * beam_pix))
    dy, dx = np.indices((2*box+1, 2*box+1), dtype=float)
    dy -= box; dx -= box
    d_sq = dx**2 + dy**2
    ann = (d_sq >= (inner*beam_pix)**2) & (d_sq <= (outer*beam_pix)**2)
    angles = np.arctan2(dx, dy) % (2*np.pi)
    sec_w = 2*np.pi / n_sec
    sec_ids = (angles / sec_w).astype(int) % n_sec
    ann_dy = dy[ann].astype(int)
    ann_dx = dx[ann].astype(int)
    ann_sec = sec_ids[ann]

    n_rej = 0
    for c in candidates:
        if c['status'] != 'PASS':
            continue
        c['global_snr'] = c['snr']
        cy, cx = c['pix_y'], c['pix_x']

        # Annulus pixel coordinates
        ay = cy + ann_dy
        ax = cx + ann_dx
        asec = ann_sec.copy()

        # Clip to image bounds
        valid = (ay >= 0) & (ay < h) & (ax >= 0) & (ax < w)
        ay, ax, asec = ay[valid], ax[valid], asec[valid]

        # Exclude masked pixels
        not_masked = ~exclusion_mask[ay, ax]
        ay, ax, asec = ay[not_masked], ax[not_masked], asec[not_masked]

        if len(ay) == 0:
            c['local_rms'] = np.nan
            continue

        vals = diff_data[ay, ax]

        # Sector-minimum MAD
        mads = []
        for s in range(n_sec):
            sec_vals = vals[asec == s]
            if len(sec_vals) >= min_pix:
                mads.append(float(mad_std(sec_vals)))

        if not mads:
            if len(vals) >= min_pix:
                lr = float(mad_std(vals))
            else:
                c['local_rms'] = np.nan
                continue
        else:
            lr = min(mads)

        if lr <= 0:
            c['local_rms'] = np.nan
            continue

        c['local_rms'] = lr
        lsnr = c['flux'] / lr
        c['snr'] = float(lsnr)

        if lsnr < sigma_thresh:
            c['status'] = 'REJECT_LOCAL_SNR'
            c['reject_reason'] = f"Local SNR={lsnr:.1f}"
            n_rej += 1

    return n_rej


def merge_nearby(candidates, beam_pix, wcs):
    """Merge candidates within 1 beam of each other, keeping highest SNR."""
    if len(candidates) < 2:
        return 0

    passed = [c for c in candidates if c['status'] == 'PASS']
    passed.sort(key=lambda c: c['snr'], reverse=True)

    cd = abs(wcs.wcs.cdelt[0]) if hasattr(wcs.wcs, 'cdelt') else 0.03125
    merge_rad = beam_pix * cd  # in degrees

    n_merged = 0
    for i, c in enumerate(passed):
        if c['status'] != 'PASS':
            continue
        for j in range(i+1, len(passed)):
            if passed[j]['status'] != 'PASS':
                continue
            sep = c['coord'].separation(passed[j]['coord']).deg
            if sep < merge_rad:
                passed[j]['status'] = 'REJECT_MERGED'
                passed[j]['reject_reason'] = f"Merged with SNR={c['snr']:.1f}"
                n_merged += 1

    return n_merged


def find_bright_sources(ref_data, horizon_mask, wcs, beam_pix, min_flux_jy):
    """Find bright sources in the reference image for proximity masking.

    Uses simple local-maximum peak finding on the reference image.

    Returns list of dicts with 'coord', 'flux', 'pix_x', 'pix_y'.
    """
    ms = max(int(beam_pix * 1.5), 5)
    lm = maximum_filter(ref_data, size=ms)
    peaks_y, peaks_x = np.where(
        (ref_data == lm) & (ref_data >= min_flux_jy) & ~horizon_mask)

    sources = []
    for sy, sx in zip(peaks_y, peaks_x):
        try:
            coord = wcs.pixel_to_world(int(sx), int(sy))
            if np.isnan(coord.ra.deg):
                continue
            sources.append({
                'coord': coord,
                'flux': float(ref_data[sy, sx]),
                'pix_x': int(sx), 'pix_y': int(sy),
            })
        except Exception:
            continue

    sources.sort(key=lambda s: s['flux'], reverse=True)
    return sources


def reject_bright_proximity(candidates, bright_sources, beam_pix, wcs,
                            mask_radius_beams=None):
    """Reject candidates near bright reference sources (two tiers).

    Tier 1: Sources > BRIGHT_MASK_FLUX_JY ? exclude within N beams.
    Tier 2: Sources > BRIGHT_EXCL_FLUX_JY ? exclude within N degrees.
    """
    if mask_radius_beams is None:
        mask_radius_beams = CONFIG['BRIGHT_MASK_RADIUS_BEAMS']
    cd = abs(wcs.wcs.cdelt[0]) if hasattr(wcs.wcs, 'cdelt') else 0.03125
    mask_radius_deg = mask_radius_beams * beam_pix * cd

    excl_flux = CONFIG['BRIGHT_EXCL_FLUX_JY']
    excl_radius_deg = CONFIG['BRIGHT_EXCL_RADIUS_DEG']

    n_rejected = 0
    for c in candidates:
        if c['status'] != 'PASS':
            continue
        for bs in bright_sources:
            sep = c['coord'].separation(bs['coord']).deg

            # Tier 2: wide exclusion for very bright sources
            if bs['flux'] >= excl_flux and sep < excl_radius_deg:
                c['status'] = 'REJECT_BRIGHT_PROX'
                c['reject_reason'] = (f"Within {sep*60:.1f}' of "
                                       f"{bs['flux']:.1f} Jy source "
                                       f"(excl {excl_radius_deg:.1f}°)")
                n_rejected += 1
                break

            # Tier 1: beam-scale exclusion for moderately bright sources
            if sep < mask_radius_deg:
                c['status'] = 'REJECT_BRIGHT_PROX'
                c['reject_reason'] = (f"Within {sep*60:.1f}' of "
                                       f"{bs['flux']:.1f} Jy source")
                n_rejected += 1
                break

    return n_rejected


def catalog_crossmatch(candidates, catalog_path, match_radius_arcmin=2.0):
    """Cross-match candidates against a source catalog."""
    if not catalog_path or not os.path.exists(catalog_path):
        return 0

    try:
        import pandas as pd
        try:
            df = pd.read_csv(catalog_path)
        except Exception:
            df = pd.read_csv(catalog_path, sep=r'\s+')
        df.columns = [c.lower() for c in df.columns]

        ra_key = 'ra_current' if 'ra_current' in df.columns else 'ra'
        dec_key = 'dec_current' if 'dec_current' in df.columns else 'dec'
        name_key = next((k for k in ['common_name', 'name', 'source']
                         if k in df.columns), None)

        if ra_key not in df.columns or dec_key not in df.columns:
            return 0

        cat_coords = SkyCoord(df[ra_key].values, df[dec_key].values, unit='deg')
        cat_names = df[name_key].values if name_key else ['']*len(df)
        print(f"  Catalog: {len(df)} sources loaded")

    except Exception as e:
        print(f"  Catalog load failed: {e}")
        return 0

    n_match = 0
    for c in candidates:
        if c['status'] != 'PASS':
            continue
        idx, d2d, _ = c['coord'].match_to_catalog_sky(cat_coords)
        if d2d.arcmin < match_radius_arcmin:
            name = cat_names[idx] if cat_names[idx] else f"Cat#{idx}"
            c['catalog_match'] = name
            n_match += 1
        else:
            c['catalog_match'] = ''

    return n_match


# ================================================================
# OUTPUT
# ================================================================
def write_outputs(candidates, out_dir, label, diff_data, diff_header):
    """Write CSV, DS9 regions, and debug FITS."""
    os.makedirs(out_dir, exist_ok=True)

    passed = [c for c in candidates if c['status'] == 'PASS']
    rejected = [c for c in candidates if c['status'] != 'PASS']

    # CSV
    csv_path = os.path.join(out_dir, f"candidates_I_{label}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ra_deg', 'dec_deg', 'ra_hms', 'dec_dms',
                          'el_deg', 'flux_jy', 'global_snr', 'local_snr',
                          'local_rms_jy', 'status', 'reject_reason',
                          'catalog_match'])
        for c in candidates:
            co = c['coord']
            ra_hms = co.ra.to_string(unit=u.hour, sep=':', precision=1,
                                      pad=True)
            dec_dms = co.dec.to_string(unit=u.deg, sep=':', precision=1,
                                       alwayssign=True, pad=True)
            writer.writerow([
                f"{co.ra.deg:.5f}", f"{co.dec.deg:.5f}",
                ra_hms, dec_dms,
                f"{c['el']:.1f}", f"{c['flux']:.4f}",
                f"{c.get('global_snr', c['snr']):.1f}",
                f"{c['snr']:.1f}",
                f"{c.get('local_rms', 0):.5f}",
                c['status'], c['reject_reason'],
                c.get('catalog_match', ''),
            ])
    print(f"  CSV: {csv_path}")

    # DS9 regions  all candidates
    reg_path = os.path.join(out_dir, f"search_I_{label}.reg")
    color_map = {
        'PASS': 'green', 'REJECT_LOCAL_SNR': 'cyan',
        'REJECT_MERGED': 'gray',
    }
    with open(reg_path, 'w') as f:
        f.write("# Region file format: DS9 version 4.1\nfk5\n")
        for c in candidates:
            color = color_map.get(c['status'], 'white')
            f.write(f"circle({c['coord'].ra.deg:.5f},"
                    f"{c['coord'].dec.deg:.5f},24') "
                    f"# color={color} text={{{c['snr']:.1f}}}\n")
    print(f"  Regions: {reg_path}")

    # DS9 regions  passed only
    if passed:
        pass_reg = os.path.join(out_dir, f"passed_I_{label}.reg")
        with open(pass_reg, 'w') as f:
            f.write("# Region file format: DS9 version 4.1\nfk5\n")
            for c in passed:
                match = c.get('catalog_match', '')
                lbl = f"SNR={c['snr']:.1f}"
                if match:
                    lbl = f"{match} {lbl}"
                f.write(f"circle({c['coord'].ra.deg:.5f},"
                        f"{c['coord'].dec.deg:.5f},24') "
                        f"# color=green text={{{lbl}}}\n")
        print(f"  PASS regions: {pass_reg}")

    # Debug FITS  difference map
    diff_path = os.path.join(out_dir, f"diff_I_{label}.fits")
    fits.writeto(diff_path, diff_data.astype(np.float32), diff_header,
                 overwrite=True)

    return passed


# ================================================================
# CUTOUT GENERATION
# ================================================================
CUTOUT_SIZE = None  # set from CONFIG at runtime

def _truncated_jname(coord):
    """Generate truncated J-name: J0553+31 format for directory grouping."""
    ra_str = coord.ra.to_string(unit=u.hour, sep='', precision=0, pad=True)
    dec_str = coord.dec.to_string(unit=u.deg, sep='', precision=0,
                                   alwayssign=True, pad=True)
    return f"J{ra_str[:4]}{dec_str[:3]}"


def _full_jname(coord):
    """Full J-name for unique filenames."""
    ra_str = coord.ra.to_string(unit=u.hour, sep='', precision=0, pad=True)
    dec_str = coord.dec.to_string(unit=u.deg, sep='', precision=0,
                                   alwayssign=True, pad=True)
    return f"J{ra_str}{dec_str}".replace(' ', '')


def _make_cutout_header(original_header, cutout_wcs, candidate, stokes_label,
                        basename_tag, cutout_size_deg):
    """Build a cutout FITS header with detection metadata."""
    h = original_header.copy()
    # Remove higher-dimensional axes
    for key in list(h.keys()):
        if any(key.endswith(str(n)) for n in [3, 4]) and key[:5] in (
                'NAXIS', 'CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT', 'CROTA'):
            del h[key]
    h['NAXIS'] = 2
    h.update(cutout_wcs.to_header())
    coord = candidate['coord']
    h['DET_RA'] = (coord.ra.deg, '[deg] Detection RA (ICRS)')
    h['DET_DEC'] = (coord.dec.deg, '[deg] Detection Dec (ICRS)')
    h['DET_FLUX'] = (candidate['flux'], '[Jy] Detection peak flux')
    h['DET_SNR'] = (candidate['snr'], 'Detection SNR')
    h['DET_EL'] = (candidate['el'], '[deg] Elevation at detection')
    h['DET_POL'] = (stokes_label, 'Stokes parameter / panel')
    h['DET_SRC'] = (basename_tag, 'Source image identifier')
    h['CUTSIZE'] = (cutout_size_deg, '[deg] Cutout angular size')
    return h


def generate_cutouts_i(candidates, tgt_data, ref_data, diff_data,
                       wcs, tgt_hdr, ref_hdr, out_dir, basename_tag,
                       cutout_sigma=None, max_cutouts=None):
    """Generate 3-panel cutouts for Stokes I candidates above threshold.

    For each candidate above cutout_sigma:
      - FITS cutouts: reference, target, difference
      - Summary PNG: 3-panel (reference | target | difference)

    Cutouts are organized by truncated J-name subdirectory.

    Parameters
    ----------
    candidates : list of dict
        Candidate list from the search pipeline.
    tgt_data, ref_data, diff_data : ndarray
        Target (aligned), reference (scaled), difference images.
    wcs : WCS
        Celestial WCS.
    tgt_hdr, ref_hdr : Header
        FITS headers for target and reference.
    out_dir : str
        Base output directory.
    basename_tag : str
        Identifier for filenames.
    cutout_sigma : float, optional
        SNR threshold for cutout generation.
    max_cutouts : int, optional
        Maximum number of cutouts to generate.

    Returns
    -------
    int
        Number of cutouts generated.
    """
    cutout_sigma = cutout_sigma or CONFIG['CUTOUT_SIGMA']
    max_cutouts = max_cutouts or CONFIG['MAX_CUTOUTS']
    cutout_size = CONFIG['CUTOUT_SIZE_DEG'] * u.deg

    # Select candidates above cutout threshold
    eligible = [c for c in candidates
                if c['status'] == 'PASS' and abs(c['snr']) >= cutout_sigma]
    eligible.sort(key=lambda c: abs(c['snr']), reverse=True)

    if not eligible:
        return 0

    if len(eligible) > max_cutouts:
        print(f"  WARNING: {len(eligible)} candidates above cutout threshold "
              f"({cutout_sigma}σ), generating top {max_cutouts} only")
        eligible = eligible[:max_cutouts]

    count = 0
    for cand in eligible:
        try:
            path = _make_one_cutout_i(
                cand, tgt_data, ref_data, diff_data,
                wcs, tgt_hdr, ref_hdr,
                out_dir, basename_tag, cutout_size)
            if path:
                cand['cutout_path'] = path
                count += 1
        except Exception as e:
            print(f"  Cutout failed for "
                  f"({cand['coord'].ra.deg:.3f}, "
                  f"{cand['coord'].dec.deg:.3f}): {e}")

    if count > 0:
        print(f"  Created {count} cutout(s)")
    return count


def _make_one_cutout_i(candidate, tgt_data, ref_data, diff_data,
                       wcs, tgt_hdr, ref_hdr,
                       out_dir, basename_tag, cutout_size):
    """Generate FITS + PNG cutouts for one Stokes I candidate.

    3-panel PNG: deep reference | 10-min target | difference
    """
    coord = candidate['coord']
    try:
        cut_ref = Cutout2D(ref_data, coord, (cutout_size, cutout_size),
                           wcs=wcs, mode='trim')
        cut_tgt = Cutout2D(tgt_data, coord, (cutout_size, cutout_size),
                           wcs=wcs, mode='trim')
        cut_diff = Cutout2D(diff_data, coord, (cutout_size, cutout_size),
                            wcs=wcs, mode='trim')
    except Exception:
        return None

    full_jname = _full_jname(coord)
    short_jname = _truncated_jname(coord)
    cutout_size_deg = cutout_size.to(u.deg).value

    jname_dir = os.path.join(out_dir, "cutouts", "I", short_jname)
    os.makedirs(jname_dir, exist_ok=True)

    # --- Save FITS cutouts ---
    head_ref = _make_cutout_header(ref_hdr, cut_ref.wcs, candidate,
                                    'I_ref', basename_tag, cutout_size_deg)
    head_tgt = _make_cutout_header(tgt_hdr, cut_tgt.wcs, candidate,
                                    'I_tgt', basename_tag, cutout_size_deg)
    head_diff = _make_cutout_header(tgt_hdr, cut_diff.wcs, candidate,
                                     'I_diff', basename_tag, cutout_size_deg)
    head_diff['HISTORY'] = 'Difference image: 10min snapshot minus deep reference'

    fits.writeto(os.path.join(jname_dir,
                 f"det_I_{basename_tag}_{full_jname}_ref.fits"),
                 cut_ref.data, head_ref, overwrite=True)
    fits.writeto(os.path.join(jname_dir,
                 f"det_I_{basename_tag}_{full_jname}_tgt.fits"),
                 cut_tgt.data, head_tgt, overwrite=True)
    fits.writeto(os.path.join(jname_dir,
                 f"det_I_{basename_tag}_{full_jname}_diff.fits"),
                 cut_diff.data, head_diff, overwrite=True)

    # --- 3-panel PNG ---
    fig = plt.figure(figsize=(18, 6))

    # Panel 1: Deep reference
    ax1 = fig.add_subplot(131, projection=cut_ref.wcs)
    vmin_r, vmax_r = np.nanpercentile(cut_ref.data, [1, 99.5])
    ax1.imshow(cut_ref.data, origin='lower', cmap='inferno',
               vmin=vmin_r, vmax=vmax_r)
    ax1.plot_coord(coord, '+', color='cyan', ms=20, markeredgewidth=2)
    ax1.set_title("Stokes I (deep reference)")
    _ref_flux = _get_center_flux(cut_ref)
    ref_str = f"{_ref_flux*1000:.1f} mJy" if np.isfinite(_ref_flux) else "N/A"
    ax1.text(0.05, 0.95, f"Ref flux: {ref_str}",
             transform=ax1.transAxes, color='white', fontsize=10,
             verticalalignment='top',
             bbox=dict(facecolor='black', alpha=0.5))

    # Panel 2: 10-min target
    ax2 = fig.add_subplot(132, projection=cut_tgt.wcs)
    vmin_t, vmax_t = np.nanpercentile(cut_tgt.data, [1, 99.5])
    ax2.imshow(cut_tgt.data, origin='lower', cmap='inferno',
               vmin=vmin_t, vmax=vmax_t)
    ax2.plot_coord(coord, '+', color='cyan', ms=20, markeredgewidth=2)
    ax2.set_title("Stokes I (10min snapshot)")
    _tgt_flux = _get_center_flux(cut_tgt)
    tgt_str = f"{_tgt_flux*1000:.1f} mJy" if np.isfinite(_tgt_flux) else "N/A"
    ax2.text(0.05, 0.95, f"Tgt flux: {tgt_str}",
             transform=ax2.transAxes, color='white', fontsize=10,
             verticalalignment='top',
             bbox=dict(facecolor='black', alpha=0.5))

    # Panel 3: Difference
    ax3 = fig.add_subplot(133, projection=cut_diff.wcs)
    vmin_d, vmax_d = np.nanpercentile(cut_diff.data, [1, 99.5])
    ax3.imshow(cut_diff.data, origin='lower', cmap='inferno',
               vmin=vmin_d, vmax=vmax_d)
    ax3.plot_coord(coord, '+', color='cyan', ms=20, markeredgewidth=2)
    ax3.set_title("Stokes I (10min − deep)")
    diff_str = (f"{candidate['flux']*1000:.1f} mJy"
                if np.isfinite(candidate['flux']) else "N/A")
    snr_str = f"{candidate['snr']:.1f}"
    match = candidate.get('catalog_match', '')
    label = f"Diff flux: {diff_str}\nSNR: {snr_str}"
    if match:
        label += f"\n{match}"
    ax3.text(0.05, 0.95, label,
             transform=ax3.transAxes, color='white', fontsize=10,
             verticalalignment='top',
             bbox=dict(facecolor='black', alpha=0.5))

    png_path = os.path.join(jname_dir,
                            f"det_I_{basename_tag}_{full_jname}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=100)
    plt.close()
    return png_path


def _get_center_flux(cutout):
    """Get flux at the center pixel of a Cutout2D."""
    try:
        if hasattr(cutout, 'position_cutout'):
            ix = int(round(float(cutout.position_cutout[0])))
            iy = int(round(float(cutout.position_cutout[1])))
        else:
            ny, nx = cutout.data.shape
            ix, iy = nx // 2, ny // 2
        if 0 <= iy < cutout.data.shape[0] and 0 <= ix < cutout.data.shape[1]:
            return float(cutout.data[iy, ix])
    except Exception:
        pass
    return np.nan


# ================================================================
# MAIN PIPELINE
# ================================================================
def run_search(tgt_path, ref_path, catalog_path=None, out_dir=None,
               sigma=None, no_align=False, bright_mask_flux=None,
               bright_mask_beams=None, cutout_sigma=None,
               max_cutouts=None, no_cutouts=False,
               save_intermediates=False):
    """Run the full Stokes I transient search pipeline.

    Pipeline order:
      1. Load images + beam/WCS setup
      2. Flux scale correction (applied to reference)
      3. Relative alignment (optional, applied to target)
      4. Subtract: aligned_target - scaled_reference
      5. Build masks (horizon, A-team) on the corrected data
      6. Extract positive peaks from difference
      7. Local SNR, bright source proximity, filtering
      8. Cutout generation for high-SNR candidates
      9. Output

    Parameters
    ----------
    tgt_path : str
        Path to target (10-min) FITS image.
    ref_path : str
        Path to reference (1-hr deep) FITS image.
    catalog_path : str, optional
        Source catalog for cross-matching.
    out_dir : str, optional
        Output directory.
    sigma : float, optional
        Detection threshold (default: CONFIG['SIGMA_THRESHOLD']).
    no_align : bool
        Skip optical flow alignment.
    cutout_sigma : float, optional
        SNR threshold for cutout generation (default: CONFIG['CUTOUT_SIGMA']).
    max_cutouts : int, optional
        Maximum cutouts to generate (default: CONFIG['MAX_CUTOUTS']).
    no_cutouts : bool
        Skip cutout generation entirely.
    save_intermediates : bool
        Keep intermediate FITS files (alignment dir, scaled ref).

    Returns
    -------
    list of dict
        Candidates that passed all filters.
    """
    t0 = time.time()
    sigma_thresh = sigma or CONFIG['SIGMA_THRESHOLD']
    mask_flux = bright_mask_flux if bright_mask_flux is not None else CONFIG['BRIGHT_MASK_FLUX_JY']
    mask_beams = bright_mask_beams if bright_mask_beams is not None else CONFIG['BRIGHT_MASK_RADIUS_BEAMS']

    if out_dir is None:
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(tgt_path))),
            "detections")
    os.makedirs(out_dir, exist_ok=True)

    tgt_base = os.path.splitext(os.path.basename(tgt_path))[0]
    log_path = os.path.join(out_dir, f"search_I_{tgt_base}.log")
    tee = _Tee(log_path)
    sys.stdout = tee

    print(f"\n{'='*65}")
    print(f"  STOKES I TRANSIENT SEARCH")
    print(f"{'='*65}")
    print(f"  Target: {os.path.basename(tgt_path)}")
    print(f"  Ref:    {os.path.basename(ref_path)}")
    print(f"  Sigma:  {sigma_thresh}")
    print(f"  Output: {out_dir}")

    # ============================================================
    # STEP 1: LOAD IMAGES
    # ============================================================
    print(f"\n{'-'*65}")
    print(f"  STEP 1: LOAD IMAGES")
    print(f"{'-'*65}")

    tgt_data, tgt_hdr = load_image(tgt_path)
    ref_data, ref_hdr = load_image(ref_path)

    if tgt_data.shape != ref_data.shape:
        print(f"  ERROR: shape mismatch {tgt_data.shape} vs {ref_data.shape}")
        sys.stdout = tee.stdout
        return []

    wcs = WCS(tgt_hdr).celestial
    cd = abs(wcs.wcs.cdelt[0]) if hasattr(wcs.wcs, 'cdelt') else 0.03125
    bmaj = tgt_hdr.get('BMAJ', ref_hdr.get('BMAJ', 0))
    if bmaj <= 0:
        bmaj = 5 * cd
    bpix = bmaj / cd
    h, w = tgt_data.shape
    obs_time = get_midpoint_time(tgt_hdr, tgt_path)
    print(f"  Shape: {h}x{w}, Beam: {bmaj*60:.1f}' ({bpix:.1f} pix)")
    print(f"  Obs time: {obs_time.isot}")

    # Quick NaN mask for RMS (before full mask creation)
    nan_mask = ~np.isfinite(tgt_data) | ~np.isfinite(ref_data)

    ref_rms = get_uniform_rms(ref_data, nan_mask)
    tgt_rms = get_uniform_rms(tgt_data, nan_mask)
    print(f"  Ref RMS: {ref_rms*1000:.2f} mJy")
    print(f"  Tgt RMS: {tgt_rms*1000:.2f} mJy")

    # Find bright sources in reference for proximity masking
    bright_sources = find_bright_sources(
        ref_data, nan_mask, wcs, bpix, mask_flux)
    print(f"  Bright ref sources (>{mask_flux:.0f} Jy): "
          f"{len(bright_sources)}")

    # ============================================================
    # STEP 2: FLUX SCALE CORRECTION
    # ============================================================
    print(f"\n{'-'*65}")
    print(f"  STEP 2: FLUX SCALE CORRECTION")
    print(f"{'-'*65}")

    # measure_flux_scale does its own per-source elevation check
    scale_map, coeffs, n_scale = measure_flux_scale(
        tgt_data, ref_data, wcs, bpix, nan_mask, obs_time)
    print(f"  Scale sources: {n_scale}")
    print(f"  Fit: a={coeffs[0]:.5f} + b={coeffs[1]:.5f}*x + "
          f"c={coeffs[2]:.5f}*y")

    ref_scaled = ref_data * scale_map

    # ============================================================
    # STEP 3: RELATIVE ALIGNMENT (optional)
    # ============================================================
    tgt_aligned = tgt_data

    if not no_align:
        print(f"\n{'-'*65}")
        print(f"  STEP 3: RELATIVE ALIGNMENT")
        print(f"{'-'*65}")

        align_dir = os.path.join(out_dir, f"align_{tgt_base}")
        os.makedirs(align_dir, exist_ok=True)

        # Write flux-scaled reference for flow
        ref_scaled_path = os.path.join(
            align_dir,
            os.path.basename(ref_path).replace('.fits', '_scaled.fits'))
        fits.writeto(ref_scaled_path, ref_scaled.astype(np.float32),
                     ref_hdr, overwrite=True)

        dewarp_path = run_alignment(
            tgt_path, ref_scaled_path, align_dir, CONFIG['CONDA_ENV'])

        if dewarp_path and os.path.exists(dewarp_path):
            tgt_aligned, _ = load_image(dewarp_path)

            # QA: did alignment improve center RMS?
            diff_pre = tgt_data - ref_scaled
            diff_post = tgt_aligned - ref_scaled
            rms_pre = get_uniform_rms(diff_pre, nan_mask)
            rms_post = get_uniform_rms(diff_post, nan_mask)

            if rms_post < rms_pre:
                print(f"  Alignment QA PASSED  RMS: "
                      f"{rms_pre*1000:.2f} -> {rms_post*1000:.2f} mJy "
                      f"({(1-rms_post/rms_pre)*100:.1f}% improvement)")
            else:
                print(f"  Alignment QA FAILED  reverting "
                      f"({rms_pre*1000:.2f} -> {rms_post*1000:.2f} mJy)")
                tgt_aligned = tgt_data
        else:
            print(f"  Alignment failed  proceeding without")
    else:
        print(f"\n  Alignment skipped (--no_align)")

    # ============================================================
    # STEP 4: SUBTRACT
    # ============================================================
    print(f"\n{'-'*65}")
    print(f"  STEP 4: SUBTRACT")
    print(f"{'-'*65}")

    diff = tgt_aligned - ref_scaled

    # ============================================================
    # STEP 5: BUILD MASKS (on corrected data)
    # ============================================================
    print(f"\n{'-'*65}")
    print(f"  STEP 5: BUILD MASKS")
    print(f"{'-'*65}")

    horizon_mask, ateam_mask, invalid_wcs, _, el_deg = \
        make_masks(tgt_hdr, tgt_data.shape, wcs, fits_path=tgt_path)
    exclusion_mask = horizon_mask | ateam_mask | invalid_wcs

    n_valid = np.sum(~exclusion_mask)
    n_total = h * w
    print(f"  Horizon: {np.sum(horizon_mask)} pixels masked")
    print(f"  A-team:  {np.sum(ateam_mask & ~horizon_mask)} pixels masked")
    print(f"  Valid:   {n_valid} / {n_total} "
          f"({100*n_valid/n_total:.1f}%)")

    diff_rms = get_uniform_rms(diff, horizon_mask)
    print(f"  Difference RMS: {diff_rms*1000:.2f} mJy")

    # ============================================================
    # STEP 6: EXTRACT POSITIVE PEAKS
    # ============================================================
    print(f"\n{'-'*65}")
    print(f"  STEP 6: EXTRACT POSITIVE PEAKS")
    print(f"{'-'*65}")

    candidates = extract_positive_peaks(
        diff, tgt_hdr, horizon_mask, ateam_mask, invalid_wcs,
        wcs, obs_time, diff_rms, bpix)

    if not candidates:
        print(f"\n  No candidates found.")
        write_outputs([], out_dir, tgt_base, diff, tgt_hdr)
        sys.stdout = tee.stdout
        return []

    # ============================================================
    # STEP 7: LOCAL SNR AND FILTERING
    # ============================================================
    print(f"\n{'-'*65}")
    print(f"  STEP 7: LOCAL SNR AND FILTERING")
    print(f"{'-'*65}")

    n_snr_rej = assign_local_snr(
        candidates, diff, exclusion_mask, bpix, sigma_thresh)
    print(f"  Rejected by local SNR: {n_snr_rej}")

    n_bright_rej = reject_bright_proximity(
        candidates, bright_sources, bpix, wcs,
        mask_radius_beams=mask_beams)
    print(f"  Rejected by bright source proximity "
          f"(>{mask_flux:.0f} Jy, "
          f"<{mask_beams:.0f} beams): {n_bright_rej}")

    n_merged = merge_nearby(candidates, bpix, wcs)
    if n_merged:
        print(f"  Merged nearby: {n_merged}")

    # Catalog cross-match
    if catalog_path:
        n_match = catalog_crossmatch(candidates, catalog_path)
        print(f"  Catalog matches: {n_match}")

    # ============================================================
    # RESULTS
    # ============================================================
    passed = [c for c in candidates if c['status'] == 'PASS']
    rejected = [c for c in candidates if c['status'] != 'PASS']

    print(f"\n{'='*65}")
    print(f"  RESULTS: {len(passed)} candidates PASS, "
          f"{len(rejected)} rejected")
    print(f"{'='*65}")

    if passed:
        print(f"\n  {'RA':>10s} {'Dec':>10s} {'El':>5s} "
              f"{'Flux(mJy)':>10s} {'Loc SNR':>8s} {'Match':>20s}")
        print(f"  {'-'*70}")
        for c in sorted(passed, key=lambda x: x['snr'], reverse=True):
            ra_s = c['coord'].ra.to_string(unit=u.hour, sep=':', precision=0,
                                            pad=True)
            dec_s = c['coord'].dec.to_string(unit=u.deg, sep=':', precision=0,
                                              alwayssign=True, pad=True)
            match = c.get('catalog_match', '')
            print(f"  {ra_s:>10s} {dec_s:>10s} {c['el']:>5.1f} "
                  f"{c['flux']*1000:>10.1f} {c['snr']:>8.1f} "
                  f"{match:>20s}")

    # Write outputs
    write_outputs(candidates, out_dir, tgt_base, diff, tgt_hdr)

    # ============================================================
    # STEP 8: CUTOUT GENERATION
    # ============================================================
    if passed and not no_cutouts:
        print(f"\n{'-'*65}")
        print(f"  STEP 8: CUTOUT GENERATION")
        print(f"{'-'*65}")
        generate_cutouts_i(
            candidates, tgt_aligned, ref_scaled, diff,
            wcs, tgt_hdr, ref_hdr, out_dir, tgt_base,
            cutout_sigma=cutout_sigma, max_cutouts=max_cutouts)

    # ============================================================
    # CLEANUP
    # ============================================================
    if not save_intermediates:
        # Remove alignment intermediates
        align_dir = os.path.join(out_dir, f"align_{tgt_base}")
        if os.path.isdir(align_dir):
            shutil.rmtree(align_dir, ignore_errors=True)
        # Remove scaled reference FITS
        for f in glob.glob(os.path.join(out_dir,
                           f"*{tgt_base}*_scaled.fits")):
            os.remove(f)

    dt = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  DONE  {dt:.0f}s")
    print(f"{'='*65}")

    sys.stdout = tee.stdout
    return passed


# ================================================================
# CLI
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Stokes I transient search via difference imaging")
    parser.add_argument("--tgt", required=True,
                        help="Target (10-min) image")
    parser.add_argument("--ref", required=True,
                        help="Reference (1-hr deep) image")
    parser.add_argument("--catalog", default=None,
                        help="Source catalog for cross-matching")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory")
    parser.add_argument("--sigma", type=float, default=None,
                        help=f"Detection threshold "
                             f"(default: {CONFIG['SIGMA_THRESHOLD']})")
    parser.add_argument("--no_align", action='store_true',
                        help="Skip optical flow alignment")
    parser.add_argument("--bright_mask_flux", type=float,
                        default=CONFIG['BRIGHT_MASK_FLUX_JY'],
                        help="Mask around ref sources brighter than this (Jy)")
    parser.add_argument("--bright_mask_beams", type=float,
                        default=CONFIG['BRIGHT_MASK_RADIUS_BEAMS'],
                        help="Mask radius in beam widths")
    parser.add_argument("--no_ateam_mask", action='store_true',
                        help="Disable A-team exclusion zones")
    parser.add_argument("--cutout_sigma", type=float,
                        default=CONFIG['CUTOUT_SIGMA'],
                        help="SNR threshold for cutout generation "
                             f"(default: {CONFIG['CUTOUT_SIGMA']})")
    parser.add_argument("--max_cutouts", type=int,
                        default=CONFIG['MAX_CUTOUTS'],
                        help=f"Max cutouts (default: {CONFIG['MAX_CUTOUTS']})")
    parser.add_argument("--no_cutouts", action='store_true',
                        help="Skip cutout generation")
    parser.add_argument("--save_intermediates", action='store_true',
                        help="Keep intermediate FITS files")
    args = parser.parse_args()

    # Apply CLI overrides to config
    if args.no_ateam_mask:
        CONFIG['ATEAM_MASK'] = False

    results = run_search(
        args.tgt, args.ref,
        catalog_path=args.catalog,
        out_dir=args.out_dir,
        sigma=args.sigma,
        no_align=args.no_align,
        bright_mask_flux=args.bright_mask_flux,
        bright_mask_beams=args.bright_mask_beams,
        cutout_sigma=args.cutout_sigma,
        max_cutouts=args.max_cutouts,
        no_cutouts=args.no_cutouts,
        save_intermediates=args.save_intermediates,
    )
    print(f"\nTotal PASS: {len(results)}")


if __name__ == "__main__":
    main()
