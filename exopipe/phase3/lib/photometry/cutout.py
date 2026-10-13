import argparse
import os
import glob
import sys
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy import units as u
from astropy.time import Time
from astropy.stats import mad_std

# --- CASA TASKS ---
try:
    from casatasks import imfit, imstat
    CASA_AVAILABLE = True
except ImportError:
    CASA_AVAILABLE = False

# --- OBSERVATORY ---
try:
    from exopipe.common.pipeline_config import OVRO_LOC
except ImportError:
    OVRO_LOC = EarthLocation(lat=37.23977727*u.deg, lon=-118.2816667*u.deg, height=1222*u.m)

warnings.simplefilter('ignore', category=FITSFixedWarning)

# --- FITS CACHE ---
# Module-level cache to avoid re-reading the same all-sky FITS files
# across multiple targets. Key = file path, value = (data, header).
_fits_cache = {}

def _cached_fits_open(path):
    """Return (data_2d, header) from cache, loading if needed."""
    if path not in _fits_cache:
        with fits.open(path) as hdul:
            _fits_cache[path] = (hdul[0].data.squeeze().copy(),
                                  hdul[0].header.copy())
    return _fits_cache[path]

def clear_fits_cache():
    """Free all cached FITS data. Call after finishing a batch."""
    _fits_cache.clear()
    _vrms_zenith_cache.clear()


# --- V RMS ZENITH CACHE ---
# Caches the non-PBcorr Stokes V MAD RMS per subband (keyed by PBcorr V path).
# Only the scalar is cached, not the image data.
_vrms_zenith_cache = {}

def _get_vrms_zenith(v_pbcorr_path):
    """Compute non-PBcorr V RMS (zenith thermal noise) without caching image."""
    if v_pbcorr_path in _vrms_zenith_cache:
        return _vrms_zenith_cache[v_pbcorr_path]
    v_rms_zn = np.nan
    try:
        v_rms_path = v_pbcorr_path
        if ".pbcorr_dewarped.fits" in v_pbcorr_path:
            candidate = v_pbcorr_path.replace(".pbcorr_dewarped.fits", "_dewarped.fits")
            if os.path.exists(candidate):
                v_rms_path = candidate
        elif ".pbcorr.fits" in v_pbcorr_path:
            candidate = v_pbcorr_path.replace(".pbcorr.fits", ".fits")
            if os.path.exists(candidate):
                v_rms_path = candidate
        with fits.open(v_rms_path) as hdul:
            data = hdul[0].data.squeeze()
            v_rms_zn = float(mad_std(data, ignore_nan=True))
    except Exception:
        pass
    _vrms_zenith_cache[v_pbcorr_path] = v_rms_zn
    return v_rms_zn


# --- DEBUG LOG ---
_debug_log_path = None
_debug_log_lock = None

def set_debug_log(path):
    """Enable flux debug logging to a file. Called per-worker."""
    global _debug_log_path, _debug_log_lock
    _debug_log_path = path
    if path:
        import threading
        _debug_log_lock = threading.Lock()

def _debug_print(msg):
    """Print message and optionally append to debug log file."""
    print(msg, flush=True)
    if _debug_log_path:
        try:
            with _debug_log_lock:
                with open(_debug_log_path, 'a') as f:
                    f.write(msg + '\n')
        except Exception:
            pass


def write_photometry_csv(summary_data, target_name, sample_name,
                          base_out_dir, detections_dir):
    """Write merged photometry CSV and copy to detection I/V dirs.

    Called by the controller after all subbands+wideband are processed
    for a given target.
    """
    if not summary_data:
        return
    safe_target = target_name.replace(" ", "_")
    out_dir_base = os.path.join(base_out_dir, sample_name, safe_target)
    os.makedirs(out_dir_base, exist_ok=True)
    csv_name = f"{safe_target}_photometry.csv"
    csv_path = os.path.join(out_dir_base, csv_name)
    pd.DataFrame(summary_data).to_csv(csv_path, index=False)

    has_i_det = any(d.get('I_Detection', False) for d in summary_data)
    has_v_det = any(d.get('V_Detection', False) for d in summary_data)
    if has_i_det:
        det_i_dir = os.path.join(detections_dir, "I", sample_name, safe_target)
        os.makedirs(det_i_dir, exist_ok=True)
        shutil.copy(csv_path, os.path.join(det_i_dir, csv_name))
    if has_v_det:
        det_v_dir = os.path.join(detections_dir, "V", sample_name, safe_target)
        os.makedirs(det_v_dir, exist_ok=True)
        shutil.copy(csv_path, os.path.join(det_v_dir, csv_name))

# --- CONFIG ---
DETECTION_SIGMA = 5.0
MIN_ELEVATION = 30.0  # degrees  skip targets below this elevation
BEAM_SEARCH_MULTIPLIER = 1.0  
IONOSPHERE_PAD_ARCSEC_85MHZ = 120.0  # 2 arcmin at 85 MHz (top of band)
IONOSPHERE_REF_FREQ_MHZ = 85.0       # Reference frequency for scaling
DEFAULT_BEAM_ARCSEC = 300.0   
CUTOUT_SIZE = 2.0 * u.deg
FIT_SIGMA_THRESHOLD = 5.0    # Only run imfit if peak exceeds this many sigma
                             # (B-01 ruling 2026-08-11: 3.0 -> 5.0; a free 2-D
                             # Gaussian fit at 3 sigma is noise-dominated)

# B-03 (Gregg 2026-08-11): the PRIMARY reported value is FORCED photometry at
# the catalog position; the local peak within the search region is SECONDARY
# (clearly labeled). Tier-1 association percentile of the peak offset is
# recorded where the calibration exists (deep-co-add table; wideband stack
# labels like '18_41MHz' return blank).
try:
    from exopipe.phase3.phase3_hwj_lightcurves import assoc_percentile as _assoc_percentile
except Exception:
    _assoc_percentile = None


def _assoc_pct(freq_label, res, el):
    """Tier-1 association percentile of the secondary peak's offset from the
    catalog position ('' where the calibration table has no entry)."""
    if _assoc_percentile is None:
        return '', ''
    try:
        off = res.get('offset_arcmin')
        rms = res.get('rms')
        pk = res.get('peak_flux')
        if off is None or not np.isfinite(off) or not rms or not np.isfinite(pk):
            return '', ''
        return _assoc_percentile(freq_label, float(off), pk / rms, el)
    except Exception:
        return '', ''

def get_timestamp_from_header(header):
    try:
        t = Time(header['DATE-OBS'], format='isot', scale='utc')
        return t.datetime.strftime("%Y%m%d_%H%M%S")
    except:
        return "UnknownDate"

def get_beam_size_arcsec(header):
    if 'BMAJ' in header:
        return header['BMAJ'] * 3600.0
    return DEFAULT_BEAM_ARCSEC

def robust_measure_flux(data, cutout_wcs, sky_coord, beam_arcsec, freq_mhz,
                        fits_path=None, confusing_sources=None):
    """
    Measure flux at source position and find peak within beam + ionospheric padding.

    Args:
        data:        2D numpy array (cutout pixel data, already in memory)
        cutout_wcs:  astropy WCS (celestial, 2D) from Cutout2D
        sky_coord:   SkyCoord of the target
        beam_arcsec: Beam FWHM in arcseconds
        freq_mhz:    Observing frequency in MHz (for ionospheric scaling)
        fits_path:   Optional path to cutout FITS for CASA imfit
        confusing_sources: Optional list of SkyCoord for known confusing sources
                         to mask within one beam FWHM before measurement.

    Returns dict with keys:
        src_flux, peak_flux, peak_coord, offset_arcmin, rms,
        fit_flux, fit_err, method, n_confusers_masked
    """
    result = {
        'src_flux': np.nan, 'peak_flux': np.nan, 'peak_coord': None,
        'offset_arcmin': np.nan, 'rms': np.nan,
        'fit_flux': None, 'fit_err': None, 'method': 'Failed',
        'n_confusers_masked': 0, 'search_radius_arcmin': np.nan,
        'primary_flux': np.nan, 'primary_method': 'Failed'
    }
    try:
        cx, cy = cutout_wcs.world_to_pixel(sky_coord)
        cx, cy = float(cx), float(cy)
        ny, nx = data.shape

        # Compute true angular pixel scale LOCAL to the target position
        # by measuring the actual angular distance of ±1 pixel offsets
        sky_center = cutout_wcs.pixel_to_world(cx, cy)
        sky_dx = cutout_wcs.pixel_to_world(cx + 1, cy)
        sky_dy = cutout_wcs.pixel_to_world(cx, cy + 1)
        scale_x = sky_center.separation(sky_dx).arcsec  # true arcsec per pixel in x
        scale_y = sky_center.separation(sky_dy).arcsec  # true arcsec per pixel in y
        cdelt = np.sqrt(scale_x * scale_y)  # geometric mean, arcsec/pixel

        # Flux at exact source position (nearest pixel)
        ix_src, iy_src = int(round(cx)), int(round(cy))
        if 0 <= ix_src < nx and 0 <= iy_src < ny:
            result['src_flux'] = float(data[iy_src, ix_src])

        # Search region: 1 BMAJ around source position
        search_arcsec = beam_arcsec
        radius_pix = int(np.ceil(search_arcsec / cdelt))
        result['search_radius_arcmin'] = search_arcsec / 60.0

        x1 = max(0, ix_src - radius_pix)
        x2 = min(nx, ix_src + radius_pix + 1)
        y1 = max(0, iy_src - radius_pix)
        y2 = min(ny, iy_src + radius_pix + 1)

        _debug_print(f"    [flux] cdelt={cdelt:.1f}\"/pix (scale_x={scale_x:.1f}, scale_y={scale_y:.1f}) "
                     f"beam={beam_arcsec:.0f}\" "
                     f"search={search_arcsec:.0f}\" radius={radius_pix}pix "
                     f"src=({ix_src},{iy_src}) box=[{x1}:{x2},{y1}:{y2}] img={nx}x{ny}")

        if x1 >= x2 or y1 >= y2:
            result['method'] = 'OutOfBounds'
            return result

        stamp = data[y1:y2, x1:x2]

        # Apply circular mask: only consider pixels within radius_pix of source
        sy, sx = np.ogrid[0:stamp.shape[0], 0:stamp.shape[1]]
        # Source position relative to stamp origin
        cx_stamp = ix_src - x1
        cy_stamp = iy_src - y1
        dist_sq = (sx - cx_stamp)**2 + (sy - cy_stamp)**2
        circular_mask = dist_sq <= radius_pix**2

        masked_stamp = np.where(circular_mask, stamp, np.nan)

        # Mask known confusing sources within one beam FWHM
        n_masked = 0
        if confusing_sources:
            beam_radius_pix = beam_arcsec / cdelt  # 1 FWHM in pixels
            for conf_coord in confusing_sources:
                try:
                    conf_cx, conf_cy = cutout_wcs.world_to_pixel(conf_coord)
                    # Confuser position relative to stamp origin
                    conf_cx_stamp = float(conf_cx) - x1
                    conf_cy_stamp = float(conf_cy) - y1
                    # Mask pixels within 1 beam FWHM of confuser
                    conf_dist_sq = (sx - conf_cx_stamp)**2 + (sy - conf_cy_stamp)**2
                    conf_mask = conf_dist_sq <= beam_radius_pix**2
                    n_before = np.count_nonzero(np.isfinite(masked_stamp))
                    masked_stamp = np.where(conf_mask, np.nan, masked_stamp)
                    n_after = np.count_nonzero(np.isfinite(masked_stamp))
                    if n_before > n_after:
                        n_masked += 1
                        _debug_print(f"    [flux] Masked confuser at ({conf_coord.ra.deg:.3f}, "
                                     f"{conf_coord.dec.deg:.3f}): {n_before - n_after} pixels")
                except Exception:
                    pass  # Skip confuser if coordinate conversion fails
        result['n_confusers_masked'] = n_masked

        rms = float(mad_std(masked_stamp, ignore_nan=True))

        # Clamp RMS: it cannot be lower than the MAD-based RMS of the
        # central 50% of the full cutout (area = 25%).  This prevents
        # artificially low RMS from small stamps dominated by source flux.
        ny_full, nx_full = data.shape
        cy_c, cx_c = ny_full // 2, nx_full // 2
        ry, rx = ny_full // 4, nx_full // 4
        inner_region = data[cy_c - ry:cy_c + ry, cx_c - rx:cx_c + rx]
        rms_floor = float(mad_std(inner_region, ignore_nan=True))
        if np.isfinite(rms_floor) and rms_floor > 0:
            rms = max(rms, rms_floor)

        result['rms'] = rms

        # Find peak pixel (max absolute value) in circular search region
        peak_idx = np.nanargmax(np.abs(masked_stamp))
        py_stamp, px_stamp = np.unravel_index(peak_idx, stamp.shape)
        px_full = x1 + px_stamp
        py_full = y1 + py_stamp
        result['peak_flux'] = float(data[py_full, px_full])

        # Convert peak pixel to sky coordinate using the cutout WCS
        peak_sky = cutout_wcs.pixel_to_world(float(px_full), float(py_full))
        result['peak_coord'] = peak_sky
        result['offset_arcmin'] = float(sky_coord.separation(peak_sky).arcmin)

        _debug_print(f"    [flux] peak at pix ({px_full},{py_full}) = {result['peak_flux']*1000:.1f} mJy, "
                     f"offset={result['offset_arcmin']:.1f}'")

        # B-03 (2026-08-11): primary = FORCED value at the catalog position.
        # 'method' now labels the PRIMARY measurement ('Forced'/'Gaussian');
        # the peak stays in peak_flux/offset_arcmin as the labeled secondary.
        result['method'] = 'Forced'
        result['primary_flux'] = result['src_flux']
        result['primary_method'] = 'Forced'

        # If significant detection and CASA available, try Gaussian fit on saved FITS
        if fits_path and rms > 0 and np.isfinite(rms):
            snr = result['peak_flux'] / rms
            if abs(snr) >= FIT_SIGMA_THRESHOLD and CASA_AVAILABLE:
                box_str = f"{x1},{y1},{x2-1},{y2-1}"
                try:
                    # For negative peaks, negate the image, fit, then negate result
                    if result['peak_flux'] < 0:
                        neg_path = fits_path.replace('.fits', '_neg.fits')
                        neg_data = -data
                        fits.writeto(neg_path, neg_data,
                                         cutout_wcs.to_header(), overwrite=True)
                        fit = imfit(imagename=neg_path, box=box_str)
                        if (fit and fit.get('converged', False)
                                and 'results' in fit
                                and 'component0' in fit['results']):
                            comp = fit['results']['component0']
                            result['fit_flux'] = -comp['peak']['value']
                            result['fit_err'] = comp['peak']['error']
                            result['method'] = 'Gaussian'
                        try:
                            os.remove(neg_path)
                        except OSError:
                            pass
                    else:
                        fit = imfit(imagename=fits_path, box=box_str)
                        if (fit and fit.get('converged', False)
                                and 'results' in fit
                                and 'component0' in fit['results']):
                            comp = fit['results']['component0']
                            result['fit_flux'] = comp['peak']['value']
                            result['fit_err'] = comp['peak']['error']
                            result['method'] = 'Gaussian'
                except Exception:
                    pass

        # >= FIT threshold the fitted value is the measurement (8.2 doctrine)
        if result['method'] == 'Gaussian' and result['fit_flux'] is not None:
            result['primary_flux'] = result['fit_flux']
            result['primary_method'] = 'Gaussian'

        return result

    except Exception as e:
        _debug_print(f"    [flux] Exception: {e}")
        return result

def _find_best_image(pol_dir, stokes_pattern, fallback_pol_dir=None):
    """Find best available image: dewarped pbcorr first, then raw pbcorr, then any match."""
    # 1. Dewarped pbcorr (in primary dir)
    hits = glob.glob(os.path.join(pol_dir, f"*{stokes_pattern}*pbcorr_dewarped.fits"))
    if hits:
        return hits[0], "dewarped_pbcorr"
    # 2. Raw pbcorr (in primary dir)
    hits = [f for f in glob.glob(os.path.join(pol_dir, f"*{stokes_pattern}*pbcorr.fits"))
            if "dewarped" not in f]
    if hits:
        return hits[0], "pbcorr"
    # 3. Any FITS matching the pattern (wideband stacks don't have pbcorr/dewarped in name)
    hits = [f for f in glob.glob(os.path.join(pol_dir, f"*{stokes_pattern}*.fits"))
            if "pbcorr" not in f and "dewarped" not in f]
    if hits:
        return hits[0], "raw"
    # 4. If fallback dir provided, try there
    if fallback_pol_dir and fallback_pol_dir != pol_dir:
        hits = glob.glob(os.path.join(fallback_pol_dir, f"*{stokes_pattern}*pbcorr_dewarped.fits"))
        if hits:
            return hits[0], "dewarped_pbcorr(fallback)"
        hits = [f for f in glob.glob(os.path.join(fallback_pol_dir, f"*{stokes_pattern}*pbcorr.fits"))
                if "dewarped" not in f]
        if hits:
            return hits[0], "pbcorr(fallback)"
        hits = [f for f in glob.glob(os.path.join(fallback_pol_dir, f"*{stokes_pattern}*.fits"))
                if "pbcorr" not in f and "dewarped" not in f]
        if hits:
            return hits[0], "raw(fallback)"
    return None, "none"


def find_all_images(run_dir, fallback_dir=None):
    """
    Find all image sets for target cutouts:
      - Deep I + V pair (for 1hr photometry)
      - 10min I + V snapshot pairs (for differenced photometry)

    Returns dict with:
        'freq': float,
        'deep_i': path, 'deep_v': path,
        'snapshots_10min': list of (i_path, v_path, interval_tag) tuples
    Or empty dict if deep pair not found.
    """
    fb_i_deep = os.path.join(fallback_dir, "I", "deep") if fallback_dir else None
    fb_v_deep = os.path.join(fallback_dir, "V", "deep") if fallback_dir else None

    i_deep, i_type = _find_best_image(
        os.path.join(run_dir, "I", "deep"), "I-Deep-Taper-Robust-0.75", fb_i_deep)
    if i_deep is None:
        i_deep, i_type = _find_best_image(
            os.path.join(run_dir, "I", "deep"), "I-Deep-Taper-Robust-0", fb_i_deep)

    v_deep, v_type = _find_best_image(
        os.path.join(run_dir, "V", "deep"), "V-Taper-Deep", fb_v_deep)

    if i_deep is None or v_deep is None:
        return {}

    # Determine freq from filename or header
    freq = 0.0
    fname = os.path.basename(i_deep)
    if "MHz" in fname:
        try:
            freq = float(fname.split('MHz')[0].split('-')[0].split('_')[-1])
        except (ValueError, IndexError):
            pass

    print(f"  [Cutout] I deep ({i_type}): {os.path.basename(i_deep)}")
    print(f"  [Cutout] V deep ({v_type}): {os.path.basename(v_deep)}")

    result = {
        'freq': freq,
        'deep_i': i_deep,
        'deep_v': v_deep,
        'deep_i_sub_ref': None,  # Robust-0 dewarped non-PBcorr for subtraction
        'snapshots_10min': []
    }

    # Find Robust-0 dewarped non-PBcorr deep I for subtraction reference
    # Separate from the photometry image (deep_i) which is Robust-0.75 PBcorr
    i_deep_dir = os.path.join(run_dir, "I", "deep")
    sub_ref_hits = [f for f in glob.glob(
        os.path.join(i_deep_dir, "*I-Deep-Taper-Robust-0*_dewarped.fits"))
        if "pbcorr" not in f and "Robust-0." not in os.path.basename(f)]
    if not sub_ref_hits and fb_i_deep:
        sub_ref_hits = [f for f in glob.glob(
            os.path.join(fb_i_deep, "*I-Deep-Taper-Robust-0*_dewarped.fits"))
            if "pbcorr" not in f and "Robust-0." not in os.path.basename(f)]
    if sub_ref_hits:
        result['deep_i_sub_ref'] = sorted(sub_ref_hits)[0]
        print(f"  [Cutout] I deep sub ref (Robust-0 dewarped): "
              f"{os.path.basename(result['deep_i_sub_ref'])}")

    # Find 10min interval image pairs
    # Search both run_dir and fallback_dir for 10min images
    import re
    for search_dir in [run_dir, fallback_dir]:
        if search_dir is None:
            continue

        i_10min_dir = os.path.join(search_dir, "I", "10min")
        v_10min_dir = os.path.join(search_dir, "V", "10min")

        if not os.path.isdir(i_10min_dir):
            continue

        # Find I 10min images  try multiple patterns
        # wsclean interval outputs use -NNNN- numbering (e.g., -0000-, -0001-)
        # which may be preserved or renamed with tNNNN tags by add_timestamps_to_images
        i_snaps = []
        for pat in [
            os.path.join(i_10min_dir, "*10min*pbcorr_dewarped.fits"),
            os.path.join(i_10min_dir, "*10min*pbcorr.fits"),
            os.path.join(i_10min_dir, "*10min*image*.fits"),
        ]:
            hits = glob.glob(pat)
            if pat.endswith("pbcorr.fits"):
                hits = [f for f in hits if "dewarped" not in f]
            elif pat.endswith("image*.fits"):
                hits = [f for f in hits if "pbcorr" not in f and "dewarped" not in f]
            if hits:
                i_snaps = sorted(hits)
                break

        if not i_snaps:
            continue

        for i_snap in i_snaps:
            bn = os.path.basename(i_snap)
            # Extract interval tag  try multiple formats:
            # 1. tNNNN tag (e.g., t0001) from add_timestamps_to_images
            # 2. wsclean -NNNN- numbering (e.g., -0001-)
            # 3. Timestamp after 10min- (e.g., 10min-20241220_115953)
            m = re.search(r'(t\d{4})', bn)
            if m:
                int_tag = m.group(1)
            else:
                m = re.search(r'-(\d{4})-', bn)
                if m:
                    int_tag = f"t{m.group(1)}"
                else:
                    # Timestamp-based naming: extract YYYYMMDD_HHMMSS after 10min-
                    m = re.search(r'10min-(\d{8}_\d{6})', bn)
                    if m:
                        int_tag = m.group(1)
                    else:
                        continue

            # Find matching V 10min image with same interval identifier
            v_snap = None
            # Build search tags: for tNNNN use both t-prefixed and numeric;
            # for timestamps use the timestamp string directly
            search_tags = [int_tag]
            if int_tag.startswith('t') and len(int_tag) == 5:
                search_tags.append(int_tag[1:])  # strip 't' prefix
            for v_pat_tag in search_tags:
                for v_pat in [
                    os.path.join(v_10min_dir, f"*10min*{v_pat_tag}*pbcorr_dewarped.fits"),
                    os.path.join(v_10min_dir, f"*10min*{v_pat_tag}*pbcorr.fits"),
                    os.path.join(v_10min_dir, f"*10min*{v_pat_tag}*image*.fits"),
                ]:
                    v_hits = glob.glob(v_pat)
                    if v_pat.endswith("pbcorr.fits"):
                        v_hits = [f for f in v_hits if "dewarped" not in f]
                    elif v_pat.endswith("image*.fits"):
                        v_hits = [f for f in v_hits if "pbcorr" not in f and "dewarped" not in f]
                    if v_hits:
                        v_snap = v_hits[0]
                        break
                if v_snap:
                    break

            if v_snap:
                result['snapshots_10min'].append((i_snap, v_snap, int_tag))

        if result['snapshots_10min']:
            break  # Found snapshots, don't check fallback

    if result['snapshots_10min']:
        print(f"  [Cutout] Found {len(result['snapshots_10min'])} 10min interval pairs")
    else:
        # Diagnostic: log what directories were searched
        for search_dir in [run_dir, fallback_dir]:
            if search_dir is None:
                continue
            i_10min_dir = os.path.join(search_dir, "I", "10min")
            if os.path.isdir(i_10min_dir):
                contents = os.listdir(i_10min_dir)
                print(f"  [Cutout] No 10min pairs found. I/10min/ has {len(contents)} files:")
                for fn in sorted(contents)[:5]:
                    print(f"    {fn}")
                if len(contents) > 5:
                    print(f"    ... and {len(contents)-5} more")
            else:
                print(f"  [Cutout] No 10min pairs found. {i_10min_dir} does not exist")

    return result


def find_image_pairs(run_dir, fallback_dir=None):
    """Legacy wrapper: returns list of (freq, i_path, v_path) for deep images only."""
    info = find_all_images(run_dir, fallback_dir)
    if not info:
        return []
    return [(info['freq'], info['deep_i'], info['deep_v'])]

def process_target(run_dir, target_name, coord, sample_name, base_out_dir, detections_dir,
                   fallback_dir=None, detection_stokes='IV', confusing_sources=None,
                   deep_i_path=None, deep_v_path=None, freq_mhz=None, freq_label=None,
                   snapshots_10min_explicit=None, deep_i_sub_ref=None):
    """
    Extract cutouts for a target from deep and 10min images.
    
    Deep Stokes I: uses confusing source masking if sources specified.
    10min Stokes I: uses (10min - deep) difference image  no masking needed.
    Stokes V: measured directly for both deep and 10min.
    
    Args:
        run_dir: Path to processed subband (NVMe)  primary search location
        target_name: Name of the target
        coord: SkyCoord of the target
        sample_name: Name of the sample (from CSV filename)
        base_out_dir: Where to save samples (NVMe/samples)
        detections_dir: Where to copy detections (NVMe/detections)
        fallback_dir: Secondary search location for images
        detection_stokes: 'I', 'V', or 'IV'  what triggers a detection
        confusing_sources: list of SkyCoord for known confusing sources
        deep_i_path: Optional explicit path to deep I image (bypasses find_all_images)
        deep_v_path: Optional explicit path to deep V image (bypasses find_all_images)
        freq_mhz: Optional explicit frequency in MHz (required with explicit paths)
        freq_label: Frequency label for filenames/dirs (e.g. '18MHz', 'Red', 'Full').
                    If None, derived from frequency as '{freq:.0f}MHz'.
        snapshots_10min_explicit: Optional list of (i_path, v_path, interval_tag)
                    tuples for 10min snapshots when using explicit paths (wideband).
        deep_i_sub_ref: Optional explicit path to Robust-0 non-PBcorr deep I
                    for subtraction reference (wideband). If None, found by find_all_images.
    """
    if deep_i_path and deep_v_path:
        img_info = {
            'freq': freq_mhz or 0.0,
            'deep_i': deep_i_path,
            'deep_v': deep_v_path,
            'deep_i_sub_ref': deep_i_sub_ref,
            'snapshots_10min': snapshots_10min_explicit or []
        }
        print(f"  [Cutout] I deep (explicit): {os.path.basename(deep_i_path)}")
        print(f"  [Cutout] V deep (explicit): {os.path.basename(deep_v_path)}")
        if deep_i_sub_ref:
            print(f"  [Cutout] I deep sub ref (explicit): {os.path.basename(deep_i_sub_ref)}")
        if img_info['snapshots_10min']:
            print(f"  [Cutout] 10min snapshots (explicit): {len(img_info['snapshots_10min'])} pairs")
    else:
        img_info = find_all_images(run_dir, fallback_dir=fallback_dir)
    if not img_info:
        return

    freq = img_info['freq']
    safe_target = target_name.replace(" ", "_")
    out_dir_base = os.path.join(base_out_dir, sample_name, safe_target)

    summary_data = []

    # Non-PBcorr V RMS (zenith thermal noise)  computed once per subband,
    # scalar cached to avoid re-reading the full image for every target.
    v_rms_zenith = _get_vrms_zenith(img_info['deep_v'])

    # --- Helper: process one image set (deep or 10min) ---
    def _process_one(i_path, v_path, category, interval_tag=None,
                     i_diff_data=None, i_diff_wcs=None, i_diff_header=None):
        nonlocal summary_data
        tag = f"_{interval_tag}" if interval_tag else ""
        try:
            data_i_raw, head_i = _cached_fits_open(i_path)
            wcs_i = WCS(head_i).celestial
            ts_str = get_timestamp_from_header(head_i)
            beam_arcsec = get_beam_size_arcsec(head_i)

            if freq == 0.0:
                nonlocal_freq = head_i.get('CRVAL3', 0) / 1e6
            else:
                nonlocal_freq = freq

            # Determine freq label for filenames and subdirectory
            if freq_label is not None:
                fl = freq_label
            else:
                fl = f"{nonlocal_freq:.0f}MHz"

            # Output directory: samples/{sample}/{target}/{category}/{freq_label}/
            out_dir = os.path.join(out_dir_base, category, fl)

            # Horizon check
            el = np.nan
            try:
                obs_time = Time(head_i['DATE-OBS'], format='isot', scale='utc')
                duration = float(head_i.get('DURATION', 0.0))
                mid_time = obs_time + duration * 0.5 * u.s
                altaz = coord.transform_to(AltAz(obstime=mid_time, location=OVRO_LOC))
                el = altaz.alt.deg
                if el < MIN_ELEVATION:
                    print(f"  [Cutout SKIP] {target_name} {category}{tag} {fl}: "
                          f"el={el:.1f}° < {MIN_ELEVATION}°", flush=True)
                    return
            except Exception as e:
                print(f"  [Cutout SKIP] {target_name} {category}{tag}: "
                      f"elevation check failed  {e}", flush=True)
                return

            data_v, head_v = _cached_fits_open(v_path)

            # Determine what I data to use for cutout and measurement
            if i_diff_data is not None:
                # 10min mode: use difference image for I
                data_i_measure = i_diff_data
                wcs_i_measure = i_diff_wcs
                i_category_label = f"{category}_diff"
            else:
                # Deep mode: use raw I with confusing source masking
                data_i_measure = data_i_raw
                wcs_i_measure = wcs_i
                i_category_label = category

            # Create cutouts
            cut_i = Cutout2D(data_i_measure, coord, (CUTOUT_SIZE, CUTOUT_SIZE),
                             wcs=wcs_i_measure, mode='trim')
            cut_v = Cutout2D(data_v, coord, (CUTOUT_SIZE, CUTOUT_SIZE),
                             wcs=WCS(head_v).celestial, mode='trim')

            # Build filenames using freq label
            fname_i = f"{safe_target}_{fl}_{category}{tag}_{ts_str}_I.fits"
            fname_v = f"{safe_target}_{fl}_{category}{tag}_{ts_str}_V.fits"

            def make_cutout_header(original_header, cutout_wcs):
                h = original_header.copy()
                for key in list(h.keys()):
                    if any(key.endswith(str(n)) for n in [3, 4]) and key[:5] in (
                        'NAXIS', 'CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT', 'CROTA'):
                        del h[key]
                h['NAXIS'] = 2
                h.update(cutout_wcs.to_header())
                return h

            head_cutout_i = make_cutout_header(
                i_diff_header if i_diff_header is not None else head_i, cut_i.wcs)
            head_cutout_v = make_cutout_header(head_v, cut_v.wcs)

            if i_diff_data is not None:
                head_cutout_i['HISTORY'] = 'Difference image: 10min snapshot minus deep reference'

            # --- Science metadata for downstream processing ---
            try:
                # Frequency info (preserved from original before axis 3 was stripped)
                freq_hz = head_i.get('CRVAL3', nonlocal_freq * 1e6)
                freq_bw = abs(head_i.get('CDELT3', 0.0))
                freq_lo_mhz = (freq_hz - freq_bw / 2.0) / 1e6
                freq_hi_mhz = (freq_hz + freq_bw / 2.0) / 1e6

                # Stokes V thermal noise from full PBcorr image (robust estimator)
                v_rms_pb = float(mad_std(data_v, ignore_nan=True))

                # Observation end time
                obs_end_str = ''
                try:
                    obs_end = obs_time + duration * u.s
                    obs_end_str = obs_end.isot
                except Exception:
                    pass

                for h in [head_cutout_i, head_cutout_v]:
                    h['OBJECT']   = (target_name, 'Target name')
                    h['SAMPLE']   = (sample_name, 'Target sample')
                    h['FREQ_MHZ'] = (nonlocal_freq, '[MHz] Center frequency')
                    h['FREQ_LO']  = (freq_lo_mhz, '[MHz] Low frequency edge')
                    h['FREQ_HI']  = (freq_hi_mhz, '[MHz] High frequency edge')
                    h['ELEV']     = (el, '[deg] Source elevation at obs midpoint')
                    h['VRMS_PB']  = (v_rms_pb, '[Jy/beam] Stokes V RMS full image (PBcorr)')
                    h['VRMS_ZN']  = (v_rms_zenith, '[Jy/beam] Stokes V RMS zenith (non-PBcorr)')
                    h['CATEGORY'] = (category, 'Image category (deep/10min)')
                    h['CUTSIZE']  = (CUTOUT_SIZE.value, '[deg] Cutout angular size')
                    if obs_end_str:
                        h['DATE-END'] = (obs_end_str, 'Observation end time (UTC)')
                    if interval_tag:
                        h['INTERVAL'] = (interval_tag, '10min interval identifier')
            except Exception as e_meta:
                print(f"  [Cutout WARN] Metadata failed for {target_name} {fl}: {e_meta}")

            # Create output directory only when we have data to write
            os.makedirs(out_dir, exist_ok=True)
            fits.writeto(os.path.join(out_dir, fname_i), cut_i.data, head_cutout_i, overwrite=True)
            fits.writeto(os.path.join(out_dir, fname_v), cut_v.data, head_cutout_v, overwrite=True)

            # Measure flux
            i_fits_path = os.path.join(out_dir, fname_i)
            v_fits_path = os.path.join(out_dir, fname_v)

            # Confusing source masking only for deep I (not differenced)
            i_confusers = confusing_sources if (i_diff_data is None) else None
            i_res = robust_measure_flux(cut_i.data, cut_i.wcs, coord, beam_arcsec,
                                        nonlocal_freq, fits_path=i_fits_path,
                                        confusing_sources=i_confusers)
            v_res = robust_measure_flux(cut_v.data, cut_v.wcs, coord, beam_arcsec,
                                        nonlocal_freq, fits_path=v_fits_path)

            # Diagnostic Plot
            # 10min: 3 panels (original I, subtracted I, Stokes V)
            # Deep:  2 panels (I, V)
            if i_diff_data is not None:
                cut_i_orig = Cutout2D(data_i_raw, coord, (CUTOUT_SIZE, CUTOUT_SIZE),
                                       wcs=wcs_i, mode='trim')
                i_orig_res = robust_measure_flux(cut_i_orig.data, cut_i_orig.wcs,
                                                  coord, beam_arcsec, nonlocal_freq)
                n_panels = 3
                panels = [
                    (cut_i_orig.wcs, cut_i_orig, i_orig_res, 'I (orig)'),
                    (cut_i.wcs, cut_i, i_res, 'I (diff)'),
                    (cut_v.wcs, cut_v, v_res, 'V'),
                ]
                fig = plt.figure(figsize=(18, 6))
            else:
                n_panels = 2
                panels = [
                    (cut_i.wcs, cut_i, i_res, 'I'),
                    (cut_v.wcs, cut_v, v_res, 'V'),
                ]
                fig = plt.figure(figsize=(12, 6))

            for panel_idx, (ax_proj, cut, res, panel_label) in enumerate(panels):
                ax = fig.add_subplot(1, n_panels, panel_idx + 1, projection=ax_proj)
                vmin, vmax = np.nanpercentile(cut.data, [1, 99.5])
                ax.imshow(cut.data, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)

                ax.plot_coord(coord, '+', color='cyan', ms=20, markeredgewidth=2)

                if res['peak_coord'] is not None:
                    ax.plot_coord(res['peak_coord'], 'D', color='lime',
                                  ms=10, markeredgewidth=1.5, fillstyle='none')

                # Red circles: confusing source positions (deep I panel only)
                if panel_label == 'I' and i_diff_data is None and confusing_sources:
                    for conf_c in confusing_sources:
                        ax.plot_coord(conf_c, 'o', color='red',
                                      ms=12, markeredgewidth=2, fillstyle='none')

                ax.set_title(f"Stokes {panel_label} ({fl}) {category}{tag}")

                lines = []
                lines.append(f"Forced: {res['src_flux']*1000:.1f} mJy (primary)")
                lines.append(f"Peak(2nd): {res['peak_flux']*1000:.1f} mJy")
                lines.append(f"RMS:  {res['rms']*1000:.1f} mJy")
                if res['fit_flux'] is not None:
                    lines.append(f"Fit:  {res['fit_flux']*1000:.1f} ± {res['fit_err']*1000:.1f} mJy")
                lines.append(f"Offset: {res['offset_arcmin']:.1f}'")
                lines.append(f"Search: {res.get('search_radius_arcmin', 0):.1f}'")
                if res.get('n_confusers_masked', 0) > 0:
                    lines.append(f"Masked: {res['n_confusers_masked']} source(s)")
                lines.append(f"El: {el:.1f}°")
                lines.append(f"({res['method']})")

                txt = "\n".join(lines)
                ax.text(0.05, 0.95, txt, transform=ax.transAxes, color='white',
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(facecolor='black', alpha=0.6))

            png_name = f"{safe_target}_{fl}_{category}{tag}_{ts_str}_Diagnostic.png"
            png_path = os.path.join(out_dir, png_name)
            plt.tight_layout()
            plt.savefig(png_path, dpi=100)
            plt.close()

            # Detection logic  evaluate I and V independently
            i_peak_snr = i_res['peak_flux'] / i_res['rms'] if i_res['rms'] > 0 else 0
            v_peak_snr = v_res['peak_flux'] / v_res['rms'] if v_res['rms'] > 0 else 0

            i_detected = i_peak_snr > DETECTION_SIGMA
            v_detected = abs(v_peak_snr) > DETECTION_SIGMA

            # Apply detection_stokes filter
            if detection_stokes == 'I':
                v_detected = False
            elif detection_stokes == 'V':
                i_detected = False
            # 'IV' keeps both

            is_detection = i_detected or v_detected

            # Route detections to I/ and V/ subdirectories
            if i_detected:
                det_i_dir = os.path.join(detections_dir, "I", sample_name,
                                         safe_target, category, fl)
                os.makedirs(det_i_dir, exist_ok=True)
                shutil.copy(png_path, os.path.join(det_i_dir, png_name))
                for det_fits in [fname_i, fname_v]:
                    src_fits = os.path.join(out_dir, det_fits)
                    if os.path.exists(src_fits):
                        shutil.copy(src_fits, os.path.join(det_i_dir, det_fits))

            if v_detected:
                det_v_dir = os.path.join(detections_dir, "V", sample_name,
                                         safe_target, category, fl)
                os.makedirs(det_v_dir, exist_ok=True)
                shutil.copy(png_path, os.path.join(det_v_dir, png_name))
                for det_fits in [fname_i, fname_v]:
                    src_fits = os.path.join(out_dir, det_fits)
                    if os.path.exists(src_fits):
                        shutil.copy(src_fits, os.path.join(det_v_dir, det_fits))

            summary_data.append({
                'Target': target_name, 'Freq_MHz': nonlocal_freq,
                'Freq_Label': fl, 'Date': ts_str,
                'Category': category,
                'Interval': interval_tag if interval_tag else '',
                'Elevation_deg': el,
                'I_Type': i_category_label,
                'I_Primary_Jy': i_res.get('primary_flux'),
                'I_Primary_Method': i_res.get('primary_method'),
                'I_Src_Jy': i_res['src_flux'], 'I_Peak_Jy': i_res['peak_flux'],
                'I_RMS_Jy': i_res['rms'], 'I_Offset_arcmin': i_res['offset_arcmin'],
                'I_Fit_Jy': i_res['fit_flux'], 'I_FitErr_Jy': i_res['fit_err'],
                'I_Method': i_res['method'],
                'I_Assoc_Pctile': _assoc_pct(fl, i_res, el)[0],
                'I_Assoc_Class': _assoc_pct(fl, i_res, el)[1],
                'I_Confusers_Masked': i_res.get('n_confusers_masked', 0),
                'V_Primary_Jy': v_res.get('primary_flux'),
                'V_Primary_Method': v_res.get('primary_method'),
                'V_Src_Jy': v_res['src_flux'], 'V_Peak_Jy': v_res['peak_flux'],
                'V_RMS_Jy': v_res['rms'], 'V_Offset_arcmin': v_res['offset_arcmin'],
                'V_Fit_Jy': v_res['fit_flux'], 'V_FitErr_Jy': v_res['fit_err'],
                'V_Method': v_res['method'],
                'V_Assoc_Pctile': _assoc_pct(fl, v_res, el)[0],
                'V_Assoc_Class': _assoc_pct(fl, v_res, el)[1],
                'Detection_Stokes': detection_stokes,
                'I_Detection': i_detected,
                'V_Detection': v_detected,
                'Detection': is_detection
            })

        except Exception as e:
            plt.close('all')
            import traceback
            print(f"  [Cutout ERROR] {target_name} {category}{tag if interval_tag else ''}: {e}")
            traceback.print_exc()

    # ===== DEEP images (1hr) =====
    _process_one(img_info['deep_i'], img_info['deep_v'], 'deep')

    # ===== 10min snapshots (differenced I) =====
    if img_info['snapshots_10min']:
        # Deep I subtraction reference: Robust-0 dewarped non-PBcorr
        # Falls back to deep_i (photometry image) if sub_ref not available
        # Cached because reused across all 10min intervals
        sub_ref_path = img_info.get('deep_i_sub_ref') or img_info['deep_i']
        if img_info.get('deep_i_sub_ref') is None:
            print(f"  [Cutout WARN] No Robust-0 non-PBcorr deep I for subtraction "
                  f" falling back to {os.path.basename(img_info['deep_i'])}")
        deep_i_data, deep_i_header = _cached_fits_open(sub_ref_path)
        deep_i_wcs = WCS(deep_i_header).celestial

        for i_snap_path, v_snap_path, int_tag in img_info['snapshots_10min']:
            try:
                # Try non-PBcorr dewarped version of the 10min I image
                # by swapping .pbcorr_dewarped.fits -> _dewarped.fits
                i_sub_path = i_snap_path
                if ".pbcorr_dewarped.fits" in i_snap_path:
                    candidate = i_snap_path.replace(".pbcorr_dewarped.fits",
                                                     "_dewarped.fits")
                    if os.path.exists(candidate):
                        i_sub_path = candidate
                    else:
                        print(f"  [Cutout WARN] No non-PBcorr dewarped 10min: "
                              f"{os.path.basename(candidate)}  using PBcorr")

                # Read subtraction image WITHOUT caching to avoid OOM
                # (each 10min non-PBcorr image is ~64MB; caching all 6 would
                # double memory usage per worker)
                with fits.open(i_sub_path) as _hdul:
                    snap_i_data = _hdul[0].data.squeeze().copy()
                    snap_i_header = _hdul[0].header.copy()
                snap_i_wcs = WCS(snap_i_header).celestial

                # Compute difference: 10min - deep
                # Both should have same shape and WCS (same subband)
                if snap_i_data.shape == deep_i_data.shape:
                    diff_data = snap_i_data - deep_i_data
                    del snap_i_data  # free before _process_one loads more
                    _process_one(i_snap_path, v_snap_path, '10min',
                                 interval_tag=int_tag,
                                 i_diff_data=diff_data,
                                 i_diff_wcs=snap_i_wcs,
                                 i_diff_header=snap_i_header)
                else:
                    print(f"  [Cutout WARN] Shape mismatch for {int_tag}: "
                          f"snap={snap_i_data.shape} vs deep={deep_i_data.shape}")
                    del snap_i_data
                    # Fall back to unsubtracted
                    _process_one(i_snap_path, v_snap_path, '10min',
                                 interval_tag=int_tag)
            except Exception as e:
                print(f"  [Cutout ERROR] 10min {int_tag}: {e}")

    # Return summary data for controller to merge across subbands
    return summary_data

def load_targets(filepath):
    """Load targets from CSV. Returns list of (name, coord, detection_stokes, confusers).
    
    detection_stokes is read from a 'detection_stokes' column if present.
    Valid values: 'I', 'V', 'IV' (or 'both'). Defaults to 'IV' if missing.
    
    confusers is read from a 'confusers' column if present.
    Format: semicolon-separated ra,dec pairs in degrees, e.g. "164.088,7.001;164.092,6.998"
    Returns list of SkyCoord objects, or empty list if no confusers.
    """
    try:
        df = pd.read_csv(filepath)
        if len(df.columns) == 1: df = pd.read_csv(filepath, sep=r'\s+')
    except: 
        df = pd.read_csv(filepath, sep=r'\s+')
    
    cols = {c.lower(): c for c in df.columns}
    name_col = next((k for k in ['common_name', 'name', 'source', 'id'] if k in cols), None)
    ra_col = next((k for k in ['ra_current', 'ra_deg', 'ra'] if k in cols), None)
    dec_col = next((k for k in ['dec_current', 'dec_deg', 'dec'] if k in cols), None)
    det_col = next((k for k in ['detection_stokes', 'det_stokes', 'stokes'] if k in cols), None)
    conf_col = next((k for k in ['confusing_sources', 'confusers', 'confuser_coords'] if k in cols), None)
    
    if not (ra_col and dec_col): return []
    targets = []
    for _, row in df.iterrows():
        n = str(row[cols[name_col]]) if name_col else f"Target_{_}"
        try:
            r, d = row[cols[ra_col]], row[cols[dec_col]]
            if isinstance(r, (float, int)): c = SkyCoord(r, d, unit='deg')
            else: c = SkyCoord(r, d, unit=(u.hourangle, u.deg))
            # Read detection stokes mode
            det_stokes = 'IV'
            if det_col:
                val = str(row[cols[det_col]]).strip().upper()
                if val in ('I', 'V', 'IV', 'VI', 'BOTH'):
                    det_stokes = 'IV' if val in ('IV', 'VI', 'BOTH') else val
            # Parse confusing source coordinates
            confusing_sources = []
            if conf_col:
                conf_str = str(row[cols[conf_col]]).strip()
                if conf_str and conf_str.lower() not in ('nan', '', 'none'):
                    for pair in conf_str.split(';'):
                        pair = pair.strip()
                        if ',' in pair:
                            try:
                                cra, cdec = pair.split(',')
                                confusing_sources.append(SkyCoord(float(cra), float(cdec), unit='deg'))
                            except (ValueError, TypeError):
                                pass
            targets.append((n, c, det_stokes, confusing_sources))
        except: pass
    return targets
