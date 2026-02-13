#!/usr/bin/env python3
"""
solar_system_cutout.py — OVRO-LWA Solar System Body Photometry

Computes ephemerides for major solar system bodies and extracts
Stokes I + V cutouts from per-subband and wideband images.

Bodies: Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune

Unlike fixed-target cutouts (cutout.py), positions are computed at each
image's observation midpoint using astropy.coordinates.get_body().

Called from:
  - process_subband.py  (per-subband, Science Phase B2)
  - post_process_science.py  (wideband images)
"""
import os
import glob
import shutil
import warnings
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, get_body
from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.stats import mad_std

warnings.simplefilter('ignore', category=FITSFixedWarning)

# --- OBSERVATORY ---
try:
    from pipeline_config import OVRO_LOC
except ImportError:
    OVRO_LOC = EarthLocation(lat=37.23977727 * u.deg, lon=-118.2816667 * u.deg, height=1222 * u.m)

# --- Reuse flux measurement from cutout.py ---
try:
    from cutout import robust_measure_flux, CUTOUT_SIZE, MIN_ELEVATION, get_beam_size_arcsec
except ImportError:
    # Fallback defaults if cutout.py not importable
    CUTOUT_SIZE = 2.0 * u.deg
    MIN_ELEVATION = 30.0
    robust_measure_flux = None

    def get_beam_size_arcsec(header):
        if 'BMAJ' in header:
            return header['BMAJ'] * 3600.0
        return 300.0

# --- CASA TASKS ---
try:
    from casatasks import imfit
    CASA_AVAILABLE = True
except ImportError:
    CASA_AVAILABLE = False


# --- CONFIGURATION ---
SOLAR_SYSTEM_BODIES = [
    'moon', 'mercury', 'venus', 'mars',
    'jupiter', 'saturn', 'uranus', 'neptune'
]

SAMPLE_NAME = "SolarSystem"
DETECTION_SIGMA = 5.0

# Mean angular diameters at 1 AU (arcsec), used to compute apparent size
# For the Moon, this is the mean angular diameter at mean Earth-Moon distance
ANGULAR_DIAMETER_1AU = {
    'moon':    None,  # Special case: ~1865" mean, not distance-scaled like planets
    'mercury': 6.74,
    'venus':   16.92,
    'mars':    9.36,
    'jupiter': 197.14,
    'saturn':  165.60,  # equatorial, excluding rings
    'uranus':  71.30,
    'neptune': 68.76,
}
MOON_MEAN_DIAMETER_ARCSEC = 1865.0  # Mean angular diameter
MOON_MEAN_DISTANCE_KM = 384400.0


def _apparent_diameter_arcsec(body_name, dist_au):
    """Compute apparent angular diameter in arcsec given distance in AU."""
    if body_name == 'moon':
        # Moon distance is in AU; convert to km for scaling
        dist_km = dist_au * 1.496e8
        if dist_km > 0:
            return MOON_MEAN_DIAMETER_ARCSEC * (MOON_MEAN_DISTANCE_KM / dist_km)
        return np.nan
    diam_1au = ANGULAR_DIAMETER_1AU.get(body_name)
    if diam_1au is not None and dist_au > 0:
        return diam_1au / dist_au
    return np.nan


def get_obs_midpoint(header):
    """Extract observation midpoint from FITS header."""
    t_start = Time(header['DATE-OBS'], format='isot', scale='utc')
    duration = float(header.get('DURATION', 0.0))
    return t_start + TimeDelta(duration * 0.5, format='sec')


def get_body_position(body_name, obs_time, location):
    """
    Compute ICRS position, elevation, and distance of a solar system body.

    Returns:
        (SkyCoord in ICRS, elevation_deg, distance_au) or (None, None, None) on failure.
    """
    try:
        body_coord = get_body(body_name, obs_time, location)
        # Convert to ICRS for WCS operations
        icrs = body_coord.transform_to('icrs')
        # Compute elevation
        altaz = body_coord.transform_to(AltAz(obstime=obs_time, location=location))
        # Distance
        dist_au = body_coord.distance.to(u.au).value if hasattr(body_coord, 'distance') else np.nan
        return icrs, altaz.alt.deg, dist_au
    except Exception:
        return None, None, None


def get_timestamp_from_header(header):
    """Extract timestamp string for filenames."""
    try:
        t = Time(header['DATE-OBS'], format='isot', scale='utc')
        return t.datetime.strftime("%Y%m%d_%H%M%S")
    except Exception:
        return "UnknownDate"


def find_solar_system_images(run_dir, fallback_dir=None):
    """
    Find all Stokes I + V image pairs for solar system cutouts.
    Returns list of (freq_mhz, category, i_path, v_path) tuples.

    Categories: 'deep' and '10min'.
    Preference: dewarped pbcorr > raw pbcorr.
    """

    def _find_best(directory, pattern, fallback=None):
        """Find best image matching pattern: dewarped pbcorr → raw pbcorr."""
        # 1. Dewarped PB-corrected
        hits = glob.glob(os.path.join(directory, f"*{pattern}*pbcorr_dewarped.fits"))
        if hits:
            return sorted(hits), "dewarped_pbcorr"
        # 2. Raw PB-corrected
        hits = [f for f in glob.glob(os.path.join(directory, f"*{pattern}*pbcorr.fits"))
                if "dewarped" not in f]
        if hits:
            return sorted(hits), "pbcorr"
        # 3. Fallback directory
        if fallback and fallback != directory:
            hits = glob.glob(os.path.join(fallback, f"*{pattern}*pbcorr_dewarped.fits"))
            if hits:
                return sorted(hits), "dewarped_pbcorr(fallback)"
            hits = [f for f in glob.glob(os.path.join(fallback, f"*{pattern}*pbcorr.fits"))
                    if "dewarped" not in f]
            if hits:
                return sorted(hits), "pbcorr(fallback)"
        return [], "none"

    pairs = []

    # --- Deep images ---
    fb_i = os.path.join(fallback_dir, "I", "deep") if fallback_dir else None
    fb_v = os.path.join(fallback_dir, "V", "deep") if fallback_dir else None

    # Stokes I deep: prefer Robust-0.75 tapered
    i_deep, i_type = _find_best(
        os.path.join(run_dir, "I", "deep"), "I-Deep-Taper-Robust-0.75", fb_i)
    if not i_deep:
        i_deep, i_type = _find_best(
            os.path.join(run_dir, "I", "deep"), "I-Deep-Taper-Robust-0", fb_i)

    # Stokes V deep
    v_deep, v_type = _find_best(
        os.path.join(run_dir, "V", "deep"), "V-Taper-Deep", fb_v)

    if i_deep and v_deep:
        # Deep: single pair
        freq = _extract_freq(i_deep[0])
        pairs.append((freq, 'deep', i_deep[0], v_deep[0]))

    # --- 10-minute snapshot images ---
    fb_i_10 = os.path.join(fallback_dir, "I", "10min") if fallback_dir else None
    fb_v_10 = os.path.join(fallback_dir, "V", "10min") if fallback_dir else None

    i_10min, _ = _find_best(
        os.path.join(run_dir, "I", "10min"), "I-Taper-10min", fb_i_10)
    v_10min, _ = _find_best(
        os.path.join(run_dir, "V", "10min"), "V-Taper-10min", fb_v_10)

    if i_10min and v_10min:
        # Match I and V 10min snapshots by timestamp in filename
        # Each snapshot has a unique timestamp; pair by closest match
        i_times = {}
        for f in i_10min:
            ts = _extract_timestamp_tag(f)
            if ts:
                i_times[ts] = f

        for v_path in v_10min:
            ts = _extract_timestamp_tag(v_path)
            if ts and ts in i_times:
                freq = _extract_freq(i_times[ts])
                pairs.append((freq, '10min', i_times[ts], v_path))

    return pairs


def _extract_freq(filepath):
    """Extract frequency in MHz from filepath or filename."""
    # Try directory name first (e.g., .../73MHz/I/deep/...)
    parts = filepath.replace('\\', '/').split('/')
    for p in parts:
        if p.endswith('MHz'):
            try:
                return float(p.replace('MHz', ''))
            except ValueError:
                pass
    # Try filename
    basename = os.path.basename(filepath)
    if 'MHz' in basename:
        try:
            return float(basename.split('MHz')[0].split('-')[0].split('_')[-1])
        except (ValueError, IndexError):
            pass
    return 0.0


def _extract_timestamp_tag(filepath):
    """Extract timestamp portion from a 10min snapshot filename for matching."""
    # Filenames like: 73MHz-I-Taper-10min-t0006-image_20241220_034500_pbcorr_dewarped.fits
    # Extract the date_time portion: 20241220_034500
    basename = os.path.basename(filepath)
    # Look for YYYYMMDD_HHMMSS pattern
    import re
    match = re.search(r'(\d{8}_\d{6})', basename)
    if match:
        return match.group(1)
    # Fallback: match by interval number (t0001, etc.)
    match = re.search(r'(t\d{4})', basename)
    if match:
        return match.group(1)
    return None


def process_solar_system(run_dir, out_dir, detections_dir,
                         fallback_dir=None, logger=None):
    """
    Main entry point: extract cutouts for all solar system bodies.

    Args:
        run_dir:         Path to processed subband or wideband directory
        out_dir:         Base output directory (e.g., science_out_dir/samples)
        detections_dir:  Where to copy detections
        fallback_dir:    Secondary search location for images
        logger:          Optional logger (uses print if None)
    """
    log = logger.info if logger else print
    log_warn = logger.warning if logger else print
    log_err = logger.error if logger else print

    if robust_measure_flux is None:
        log_err("cutout.robust_measure_flux not available — cannot measure fluxes.")
        return

    log(f"--- Solar System Photometry ---")
    log(f"  Source dir: {run_dir}")

    pairs = find_solar_system_images(run_dir, fallback_dir=fallback_dir)
    if not pairs:
        log_warn("No suitable image pairs found for solar system cutouts.")
        return

    log(f"  Found {len(pairs)} image pairs (deep + 10min)")

    all_summary = []

    for body_name in SOLAR_SYSTEM_BODIES:
        display_name = body_name.capitalize()
        safe_name = display_name
        body_out = os.path.join(out_dir, SAMPLE_NAME, safe_name)
        os.makedirs(body_out, exist_ok=True)

        for freq, category, i_path, v_path in pairs:
            try:
                # Load I header for timing
                with fits.open(i_path) as h_i:
                    data_i = h_i[0].data.squeeze()
                    head_i = h_i[0].header
                    wcs_i = WCS(head_i).celestial
                    beam_arcsec = get_beam_size_arcsec(head_i)

                # Get observation midpoint
                obs_mid = get_obs_midpoint(head_i)
                ts_str = get_timestamp_from_header(head_i)

                # If freq unknown from path, try header
                if freq == 0.0:
                    freq = head_i.get('CRVAL3', 0) / 1e6

                # Compute ephemeris
                coord, elevation, dist_au = get_body_position(body_name, obs_mid, OVRO_LOC)
                if coord is None:
                    continue

                # Elevation check
                if elevation < MIN_ELEVATION:
                    log(f"  {display_name} @ {freq:.0f}MHz {category} {ts_str}: "
                        f"el={elevation:.1f}° < {MIN_ELEVATION}° — skipped")
                    continue

                ang_diam = _apparent_diameter_arcsec(body_name, dist_au)

                log(f"  {display_name} @ {freq:.0f}MHz {category} {ts_str}: "
                    f"RA={coord.ra.to_string(u.hour, sep=':', precision=1)} "
                    f"Dec={coord.dec.to_string(u.deg, sep=':', precision=1)} "
                    f"el={elevation:.1f}° dist={dist_au:.4f}AU diam={ang_diam:.1f}\"")

                # Load V image
                with fits.open(v_path) as h_v:
                    data_v = h_v[0].data.squeeze()
                    head_v = h_v[0].header

                # Create cutouts (the body moves, so coord is different per image)
                try:
                    cut_i = Cutout2D(data_i, coord, (CUTOUT_SIZE, CUTOUT_SIZE),
                                     wcs=wcs_i, mode='trim')
                    cut_v = Cutout2D(data_v, coord, (CUTOUT_SIZE, CUTOUT_SIZE),
                                     wcs=wcs_i, mode='trim')
                except Exception as e:
                    log_warn(f"    Cutout failed for {display_name}: {e}")
                    continue

                # Build cutout headers with full metadata
                def make_cutout_header(orig, cut_wcs):
                    h = orig.copy()
                    for key in list(h.keys()):
                        if any(key.endswith(str(n)) for n in [3, 4]) and key[:5] in (
                                'NAXIS', 'CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT', 'CROTA'):
                            del h[key]
                    h['NAXIS'] = 2
                    h.update(cut_wcs.to_header())
                    return h

                head_cut_i = make_cutout_header(head_i, cut_i.wcs)
                head_cut_v = make_cutout_header(head_v, cut_v.wcs)

                # Add ephemeris and observation metadata to both headers
                for h in [head_cut_i, head_cut_v]:
                    h['OBJECT'] = display_name
                    h['EPH_RA'] = (coord.ra.deg, '[deg] Ephemeris RA (ICRS) at obs midpoint')
                    h['EPH_DEC'] = (coord.dec.deg, '[deg] Ephemeris Dec (ICRS) at obs midpoint')
                    h['EPH_EL'] = (elevation, '[deg] Elevation at obs midpoint')
                    h['EPH_DIST'] = (dist_au, '[AU] Geocentric distance at obs midpoint')
                    h['EPH_DIAM'] = (ang_diam, '[arcsec] Apparent angular diameter')
                    h['EPH_TIME'] = (obs_mid.isot, 'UTC time of ephemeris evaluation')
                    h['FREQ_MHZ'] = (freq, '[MHz] Observing frequency')
                    h['CATEGORY'] = (category, 'Image category (deep/10min)')
                    h['CUTSIZE'] = (CUTOUT_SIZE.value, '[deg] Cutout angular size')

                # Save FITS cutouts
                cat_tag = category  # 'deep' or '10min'
                fname_i = f"{safe_name}_{freq:.0f}MHz_{cat_tag}_{ts_str}_I.fits"
                fname_v = f"{safe_name}_{freq:.0f}MHz_{cat_tag}_{ts_str}_V.fits"
                fits.writeto(os.path.join(body_out, fname_i), cut_i.data,
                             head_cut_i, overwrite=True)
                fits.writeto(os.path.join(body_out, fname_v), cut_v.data,
                             head_cut_v, overwrite=True)

                # Measure flux
                i_fits_path = os.path.join(body_out, fname_i)
                v_fits_path = os.path.join(body_out, fname_v)

                i_res = robust_measure_flux(cut_i.data, cut_i.wcs, coord,
                                            beam_arcsec, freq, fits_path=i_fits_path)
                v_res = robust_measure_flux(cut_v.data, cut_v.wcs, coord,
                                            beam_arcsec, freq, fits_path=v_fits_path)

                # Diagnostic PNG
                fig = plt.figure(figsize=(12, 6))

                for panel_idx, (cut, res, stokes) in enumerate([
                    (cut_i, i_res, 'I'),
                    (cut_v, v_res, 'V'),
                ]):
                    ax = fig.add_subplot(1, 2, panel_idx + 1, projection=cut.wcs)
                    vmin, vmax = np.nanpercentile(cut.data, [1, 99.5])
                    ax.imshow(cut.data, origin='lower', cmap='inferno',
                              vmin=vmin, vmax=vmax)

                    # Cyan cross: ephemeris position
                    ax.plot_coord(coord, '+', color='cyan', ms=20, markeredgewidth=2)

                    # Yellow diamond: peak pixel
                    if res['peak_coord'] is not None:
                        ax.plot_coord(res['peak_coord'], 'D', color='yellow',
                                      ms=10, markeredgewidth=1.5, fillstyle='none')

                    ax.set_title(f"{display_name} — Stokes {stokes} "
                                 f"({freq:.0f} MHz, {cat_tag})")

                    lines = [
                        f"Src:  {res['src_flux'] * 1000:.1f} mJy",
                        f"Peak: {res['peak_flux'] * 1000:.1f} mJy",
                        f"RMS:  {res['rms'] * 1000:.1f} mJy",
                    ]
                    if res['fit_flux'] is not None:
                        lines.append(
                            f"Fit:  {res['fit_flux'] * 1000:.1f} "
                            f"± {res['fit_err'] * 1000:.1f} mJy")
                    lines.append(f"Offset: {res['offset_arcmin']:.1f}'")
                    lines.append(f"El: {elevation:.1f}°")
                    lines.append(f"({res['method']})")

                    ax.text(0.05, 0.95, "\n".join(lines),
                            transform=ax.transAxes, color='white', fontsize=9,
                            verticalalignment='top', fontfamily='monospace',
                            bbox=dict(facecolor='black', alpha=0.6))

                png_name = f"{safe_name}_{freq:.0f}MHz_{cat_tag}_{ts_str}_Diagnostic.png"
                png_path = os.path.join(body_out, png_name)
                plt.tight_layout()
                plt.savefig(png_path, dpi=100)
                plt.close()

                # Detection logic
                is_detection = False
                i_peak_snr = (i_res['peak_flux'] / i_res['rms']
                              if i_res['rms'] > 0 and np.isfinite(i_res['rms']) else 0)
                if i_peak_snr > DETECTION_SIGMA:
                    is_detection = True
                    det_body_dir = os.path.join(detections_dir, SAMPLE_NAME, safe_name)
                    os.makedirs(det_body_dir, exist_ok=True)
                    shutil.copy(png_path, os.path.join(det_body_dir, png_name))

                all_summary.append({
                    'Body': display_name,
                    'Freq_MHz': freq,
                    'Category': category,
                    'Date': ts_str,
                    'RA_deg': coord.ra.deg,
                    'Dec_deg': coord.dec.deg,
                    'Elevation_deg': elevation,
                    'Distance_AU': dist_au,
                    'AngDiam_arcsec': ang_diam,
                    'I_Src_Jy': i_res['src_flux'],
                    'I_Peak_Jy': i_res['peak_flux'],
                    'I_RMS_Jy': i_res['rms'],
                    'I_Offset_arcmin': i_res['offset_arcmin'],
                    'I_Fit_Jy': i_res['fit_flux'],
                    'I_FitErr_Jy': i_res['fit_err'],
                    'I_Method': i_res['method'],
                    'V_Src_Jy': v_res['src_flux'],
                    'V_Peak_Jy': v_res['peak_flux'],
                    'V_RMS_Jy': v_res['rms'],
                    'V_Offset_arcmin': v_res['offset_arcmin'],
                    'V_Fit_Jy': v_res['fit_flux'],
                    'V_FitErr_Jy': v_res['fit_err'],
                    'V_Method': v_res['method'],
                    'Detection': is_detection,
                })

            except Exception as e:
                plt.close('all')
                log_err(f"  {display_name} @ {freq:.0f}MHz {category}: {e}")
                traceback.print_exc()

    # Save combined summary CSV
    if all_summary:
        csv_path = os.path.join(out_dir, SAMPLE_NAME, "SolarSystem_photometry.csv")
        pd.DataFrame(all_summary).to_csv(csv_path, index=False)
        log(f"  Saved solar system photometry: {csv_path} ({len(all_summary)} measurements)")

        # Copy CSV to detections if any detections
        if any(d['Detection'] for d in all_summary):
            det_ss_dir = os.path.join(detections_dir, SAMPLE_NAME)
            os.makedirs(det_ss_dir, exist_ok=True)
            shutil.copy(csv_path, os.path.join(det_ss_dir, "SolarSystem_photometry.csv"))
    else:
        log("  No solar system measurements produced.")


def process_wideband_solar_system(run_dir, logger=None):
    """
    Extract solar system cutouts from wideband stacked images.
    Called from post_process_science.py after wideband stacking.

    Looks for Wideband_*_I_deep_*.fits and matching V files in run_dir.
    """
    log = logger.info if logger else print
    log_warn = logger.warning if logger else print

    if robust_measure_flux is None:
        if logger:
            logger.error("cutout.robust_measure_flux not available.")
        return

    log("--- Wideband Solar System Photometry ---")

    wb_dir = os.path.join(run_dir, "Wideband")
    out_dir = os.path.join(run_dir, "samples")
    det_dir = os.path.join(run_dir, "detections")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(det_dir, exist_ok=True)

    # Find wideband image pairs: I deep + V deep for each band
    # Also 10min if available
    pairs = []

    for band in ['Red', 'Green', 'Blue']:
        for cat in ['deep', '10min']:
            # Look for any I wideband image for this band/category
            i_pattern = os.path.join(wb_dir, f"Wideband_{band}_I_{cat}_*.fits")
            i_matches = glob.glob(i_pattern)

            v_pattern = os.path.join(wb_dir, f"Wideband_{band}_V_{cat}_*.fits")
            v_matches = glob.glob(v_pattern)

            if i_matches and v_matches:
                # Prefer Taper versions for cutouts
                i_taper = [f for f in i_matches if 'Taper' in os.path.basename(f)
                           and 'NoTaper' not in os.path.basename(f)]
                i_pick = i_taper[0] if i_taper else i_matches[0]
                v_pick = v_matches[0]

                # Use band name as freq label
                freq_label = {'Red': 30.0, 'Green': 52.0, 'Blue': 74.0}
                pairs.append((freq_label.get(band, 0), cat, i_pick, v_pick))

    if not pairs:
        log_warn("No wideband image pairs found for solar system cutouts.")
        return

    log(f"  Found {len(pairs)} wideband image pairs")

    all_summary = []

    for body_name in SOLAR_SYSTEM_BODIES:
        display_name = body_name.capitalize()
        safe_name = display_name
        body_out = os.path.join(out_dir, SAMPLE_NAME, safe_name)
        os.makedirs(body_out, exist_ok=True)

        for freq, category, i_path, v_path in pairs:
            try:
                with fits.open(i_path) as h_i:
                    data_i = h_i[0].data.squeeze()
                    head_i = h_i[0].header
                    wcs_i = WCS(head_i).celestial
                    beam_arcsec = get_beam_size_arcsec(head_i)

                obs_mid = get_obs_midpoint(head_i)
                ts_str = get_timestamp_from_header(head_i)

                coord, elevation, dist_au = get_body_position(body_name, obs_mid, OVRO_LOC)
                if coord is None or elevation < MIN_ELEVATION:
                    continue

                ang_diam = _apparent_diameter_arcsec(body_name, dist_au)

                # Determine band name from filename for labeling
                basename = os.path.basename(i_path)
                band_tag = "WB"
                for bn in ['Red', 'Green', 'Blue']:
                    if bn in basename:
                        band_tag = f"WB-{bn}"
                        break

                log(f"  {display_name} @ {band_tag} {category} {ts_str}: "
                    f"el={elevation:.1f}° dist={dist_au:.4f}AU")

                with fits.open(v_path) as h_v:
                    data_v = h_v[0].data.squeeze()
                    head_v = h_v[0].header

                try:
                    cut_i = Cutout2D(data_i, coord, (CUTOUT_SIZE, CUTOUT_SIZE),
                                     wcs=wcs_i, mode='trim')
                    cut_v = Cutout2D(data_v, coord, (CUTOUT_SIZE, CUTOUT_SIZE),
                                     wcs=wcs_i, mode='trim')
                except Exception:
                    continue

                def make_cutout_header(orig, cut_wcs):
                    h = orig.copy()
                    for key in list(h.keys()):
                        if any(key.endswith(str(n)) for n in [3, 4]) and key[:5] in (
                                'NAXIS', 'CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CUNIT', 'CROTA'):
                            del h[key]
                    h['NAXIS'] = 2
                    h.update(cut_wcs.to_header())
                    return h

                head_cut_i = make_cutout_header(head_i, cut_i.wcs)
                head_cut_v = make_cutout_header(head_v, cut_v.wcs)
                for h in [head_cut_i, head_cut_v]:
                    h['OBJECT'] = display_name
                    h['EPH_RA'] = (coord.ra.deg, '[deg] Ephemeris RA (ICRS) at obs midpoint')
                    h['EPH_DEC'] = (coord.dec.deg, '[deg] Ephemeris Dec (ICRS) at obs midpoint')
                    h['EPH_EL'] = (elevation, '[deg] Elevation at obs midpoint')
                    h['EPH_DIST'] = (dist_au, '[AU] Geocentric distance at obs midpoint')
                    h['EPH_DIAM'] = (ang_diam, '[arcsec] Apparent angular diameter')
                    h['EPH_TIME'] = (obs_mid.isot, 'UTC time of ephemeris evaluation')
                    h['FREQ_MHZ'] = (freq, '[MHz] Nominal band center frequency')
                    h['CATEGORY'] = (category, 'Image category (deep/10min)')
                    h['WB_BAND'] = (band_tag, 'Wideband color band')
                    h['CUTSIZE'] = (CUTOUT_SIZE.value, '[deg] Cutout angular size')

                fname_i = f"{safe_name}_{band_tag}_{category}_{ts_str}_I.fits"
                fname_v = f"{safe_name}_{band_tag}_{category}_{ts_str}_V.fits"
                fits.writeto(os.path.join(body_out, fname_i), cut_i.data,
                             head_cut_i, overwrite=True)
                fits.writeto(os.path.join(body_out, fname_v), cut_v.data,
                             head_cut_v, overwrite=True)

                i_res = robust_measure_flux(cut_i.data, cut_i.wcs, coord,
                                            beam_arcsec, freq,
                                            fits_path=os.path.join(body_out, fname_i))
                v_res = robust_measure_flux(cut_v.data, cut_v.wcs, coord,
                                            beam_arcsec, freq,
                                            fits_path=os.path.join(body_out, fname_v))

                # Diagnostic PNG
                fig = plt.figure(figsize=(12, 6))
                for panel_idx, (cut, res, stokes) in enumerate([
                    (cut_i, i_res, 'I'), (cut_v, v_res, 'V')
                ]):
                    ax = fig.add_subplot(1, 2, panel_idx + 1, projection=cut.wcs)
                    vmin, vmax = np.nanpercentile(cut.data, [1, 99.5])
                    ax.imshow(cut.data, origin='lower', cmap='inferno',
                              vmin=vmin, vmax=vmax)
                    ax.plot_coord(coord, '+', color='cyan', ms=20, markeredgewidth=2)
                    if res['peak_coord'] is not None:
                        ax.plot_coord(res['peak_coord'], 'D', color='yellow',
                                      ms=10, markeredgewidth=1.5, fillstyle='none')
                    ax.set_title(f"{display_name} — Stokes {stokes} ({band_tag}, {category})")

                    lines = [
                        f"Src:  {res['src_flux'] * 1000:.1f} mJy",
                        f"Peak: {res['peak_flux'] * 1000:.1f} mJy",
                        f"RMS:  {res['rms'] * 1000:.1f} mJy",
                    ]
                    if res['fit_flux'] is not None:
                        lines.append(f"Fit: {res['fit_flux'] * 1000:.1f} "
                                     f"± {res['fit_err'] * 1000:.1f} mJy")
                    lines.append(f"Offset: {res['offset_arcmin']:.1f}'")
                    lines.append(f"El: {elevation:.1f}°")
                    ax.text(0.05, 0.95, "\n".join(lines),
                            transform=ax.transAxes, color='white', fontsize=9,
                            verticalalignment='top', fontfamily='monospace',
                            bbox=dict(facecolor='black', alpha=0.6))

                png_name = f"{safe_name}_{band_tag}_{category}_{ts_str}_Diagnostic.png"
                plt.tight_layout()
                plt.savefig(os.path.join(body_out, png_name), dpi=100)
                plt.close()

                is_det = False
                snr = (i_res['peak_flux'] / i_res['rms']
                       if i_res['rms'] > 0 and np.isfinite(i_res['rms']) else 0)
                if snr > DETECTION_SIGMA:
                    is_det = True
                    det_body = os.path.join(det_dir, SAMPLE_NAME, safe_name)
                    os.makedirs(det_body, exist_ok=True)
                    shutil.copy(os.path.join(body_out, png_name),
                                os.path.join(det_body, png_name))

                all_summary.append({
                    'Body': display_name, 'Band': band_tag,
                    'Freq_MHz': freq, 'Category': category, 'Date': ts_str,
                    'RA_deg': coord.ra.deg, 'Dec_deg': coord.dec.deg,
                    'Elevation_deg': elevation,
                    'Distance_AU': dist_au,
                    'AngDiam_arcsec': ang_diam,
                    'I_Src_Jy': i_res['src_flux'], 'I_Peak_Jy': i_res['peak_flux'],
                    'I_RMS_Jy': i_res['rms'], 'I_Offset_arcmin': i_res['offset_arcmin'],
                    'I_Fit_Jy': i_res['fit_flux'], 'I_FitErr_Jy': i_res['fit_err'],
                    'I_Method': i_res['method'],
                    'V_Src_Jy': v_res['src_flux'], 'V_Peak_Jy': v_res['peak_flux'],
                    'V_RMS_Jy': v_res['rms'], 'V_Offset_arcmin': v_res['offset_arcmin'],
                    'V_Fit_Jy': v_res['fit_flux'], 'V_FitErr_Jy': v_res['fit_err'],
                    'V_Method': v_res['method'],
                    'Detection': is_det,
                })

            except Exception as e:
                plt.close('all')
                if logger:
                    logger.error(f"  WB {display_name}: {e}")
                traceback.print_exc()

    if all_summary:
        csv_path = os.path.join(out_dir, SAMPLE_NAME, "SolarSystem_wideband_photometry.csv")
        pd.DataFrame(all_summary).to_csv(csv_path, index=False)
        log(f"  Saved wideband solar system photometry: {csv_path}")
