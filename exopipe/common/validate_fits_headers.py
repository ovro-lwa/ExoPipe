#!/usr/bin/env python3
"""
validate_fits_headers.py  FITS Header Validation for the OVRO-LWA Pipeline

Checks that image headers contain the required keywords for correct
co-addition, phase folding, and pipeline processing. Designed to run
at key pipeline checkpoints to catch header problems early.

Usage as a module:
    from exopipe.common.validate_fits_headers import validate_header, validate_directory

Usage from CLI:
    python validate_fits_headers.py /path/to/image.fits
    python validate_fits_headers.py /path/to/directory/ --recursive
"""
import os
import sys
import glob
import logging
import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger('phase2')


# =====================================================================
# REQUIRED KEYWORDS BY CATEGORY
# =====================================================================

# Critical: pipeline will produce wrong results without these
CRITICAL_KEYWORDS = {
    'BMAJ':     'Beam major axis [deg]',
    'BMIN':     'Beam minor axis [deg]',
    'BPA':      'Beam position angle [deg]',
}

# Required: needed for correct science products
#
# Time semantics (see _stamp_times):
#   DATE-OBS/MJD-OBS  start of the first integration included
#   DATE-END          end of the last integration included
#   DURATION          wall-clock span = DATE-END - DATE-OBS [s]
#                     (days for multi-night LST stacks)
#   INTTIME           summed integration time actually included [s]
#                     (equals DURATION only for contiguous single-epoch
#                     images; independent of DURATION in general)
#   MJD-MID           midpoint of [DATE-OBS, DATE-END]
REQUIRED_KEYWORDS = {
    'DATE-OBS': 'Observation start time (ISO 8601)',
    'DURATION': 'Wall-clock span [seconds]',
    'INTTIME':  'Summed integration time [seconds]',
}

# Frequency: at least one of these must be present
FREQUENCY_KEYWORDS = ['CRVAL3', 'CRVAL4', 'RESTFRQ', 'RESTFREQ', 'OBSFREQ']

# Recommended: useful but non-blocking
RECOMMENDED_KEYWORDS = {
    'DATE-END':  'Observation end time',
    'TIMESYS':   'Time system (should be UTC)',
    'TELESCOP':  'Telescope name',
    'BUNIT':     'Brightness unit',
    'STOKES':    'Stokes parameter',
}

# Phase 2 keywords: added during pipeline processing
PHASE2_KEYWORDS = {
    'MJD-MID':   'Midpoint MJD for phase folding',
    'FREQ_MIN':  'Lower edge of band [Hz]',
    'FREQ_MAX':  'Upper edge of band [Hz]',
    'BANDWDTH': 'Total bandwidth [Hz]',
}


# =====================================================================
# VALIDATION FUNCTIONS
# =====================================================================

class HeaderReport:
    """Validation report for a single FITS file."""

    def __init__(self, path):
        self.path = path
        self.filename = os.path.basename(path)
        self.errors = []      # Critical failures
        self.warnings = []    # Required but missing
        self.info = []        # Recommendations
        self.keywords_found = {}

    @property
    def ok(self):
        return len(self.errors) == 0

    @property
    def clean(self):
        return len(self.errors) == 0 and len(self.warnings) == 0

    def summary(self):
        status = "OK" if self.ok else "FAIL"
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} errors")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warnings")
        if self.info:
            parts.append(f"{len(self.info)} recommendations")
        detail = ", ".join(parts) if parts else "all checks passed"
        return f"[{status}] {self.filename}: {detail}"

    def full_report(self):
        lines = [self.summary()]
        for e in self.errors:
            lines.append(f"  ERROR:   {e}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        for i in self.info:
            lines.append(f"  INFO:    {i}")
        return "\n".join(lines)


def validate_header(fits_path, level='standard', logger_in=None):
    """Validate a FITS image header.

    Parameters
    ----------
    fits_path : str
        Path to FITS file.
    level : str
        Validation level:
        - 'critical': only beam + frequency (fast, for bulk checks)
        - 'standard': critical + required + frequency (default)
        - 'full': all checks including recommended and Phase 2 keywords
    logger_in : logging.Logger, optional

    Returns
    -------
    HeaderReport
    """
    log = logger_in or logger
    report = HeaderReport(fits_path)

    try:
        with fits.open(fits_path) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data
    except Exception as e:
        report.errors.append(f"Cannot open FITS file: {e}")
        return report

    # --- Data shape ---
    if data is None:
        report.errors.append("No data in primary HDU")
        return report

    shape = data.shape
    ndim = len(shape)
    report.keywords_found['SHAPE'] = str(shape)

    # --- Critical: Beam ---
    for key, desc in CRITICAL_KEYWORDS.items():
        val = hdr.get(key)
        if val is None:
            report.errors.append(f"Missing {key} ({desc})")
        else:
            report.keywords_found[key] = val
            # Sanity checks
            if key == 'BMAJ' and (val <= 0 or val > 10):
                report.errors.append(
                    f"BMAJ={val} deg looks wrong (expected 0.01-5 deg)")
            if key == 'BMIN' and (val <= 0 or val > 10):
                report.errors.append(
                    f"BMIN={val} deg looks wrong (expected 0.01-5 deg)")

    # --- Critical: Frequency ---
    freq_found = False
    freq_value = None
    for key in FREQUENCY_KEYWORDS:
        val = hdr.get(key)
        if val is not None and val > 0:
            freq_found = True
            freq_value = val
            report.keywords_found[key] = val
            break

    # Also check CTYPE3/4 for frequency axis
    if not freq_found:
        for ax in [3, 4]:
            ctype = hdr.get(f'CTYPE{ax}', '')
            if 'FREQ' in str(ctype).upper():
                val = hdr.get(f'CRVAL{ax}')
                if val is not None and val > 0:
                    freq_found = True
                    freq_value = val
                    report.keywords_found[f'CRVAL{ax}'] = val
                    break

    if not freq_found:
        report.errors.append(
            "No frequency information found (need CRVAL3/4 with "
            "CTYPE='FREQ', RESTFRQ, RESTFREQ, or OBSFREQ)")
    elif freq_value is not None:
        # Sanity: OVRO-LWA operates 10-88 MHz
        freq_mhz = freq_value / 1e6
        if freq_mhz < 5 or freq_mhz > 200:
            report.warnings.append(
                f"Frequency {freq_mhz:.1f} MHz outside expected "
                f"OVRO-LWA range (10-88 MHz)")

    if level == 'critical':
        return report

    # --- Required: Time ---
    for key, desc in REQUIRED_KEYWORDS.items():
        val = hdr.get(key)
        if val is None:
            report.warnings.append(f"Missing {key} ({desc})")
        else:
            report.keywords_found[key] = val

    # DATE-OBS format check
    date_obs = hdr.get('DATE-OBS')
    if date_obs is not None:
        if not isinstance(date_obs, str) or len(date_obs) < 10:
            report.warnings.append(
                f"DATE-OBS='{date_obs}' doesn't look like ISO 8601")

    # DURATION/INTTIME sanity
    duration = hdr.get('DURATION')
    inttime = hdr.get('INTTIME')
    if duration is not None:
        if duration <= 0:
            report.warnings.append(f"DURATION={duration} <= 0")
        elif duration > 86400 and inttime is None:
            # Multi-night stacks legitimately span days, but must then
            # carry INTTIME so the actual integration time is known.
            report.warnings.append(
                f"DURATION={duration}s > 24 hours with no INTTIME")
    if inttime is not None and duration is not None and \
            inttime > duration + 1.0:
        report.warnings.append(
            f"INTTIME={inttime}s exceeds DURATION={duration}s")

    # --- WCS ---
    try:
        wcs = WCS(hdr)
        if not wcs.has_celestial:
            report.warnings.append("No valid celestial WCS found")
        else:
            # Check pixel scale is reasonable
            cdelt1 = abs(hdr.get('CDELT1', 0))
            cdelt2 = abs(hdr.get('CDELT2', 0))
            if cdelt1 > 0 and (cdelt1 < 0.001 or cdelt1 > 1.0):
                report.warnings.append(
                    f"CDELT1={cdelt1} deg looks unusual for OVRO-LWA")
    except Exception:
        report.warnings.append("WCS parsing failed")

    if level == 'standard':
        return report

    # --- Full: Recommended ---
    for key, desc in RECOMMENDED_KEYWORDS.items():
        val = hdr.get(key)
        if val is None:
            report.info.append(f"Missing {key} ({desc})")
        else:
            report.keywords_found[key] = val

    # --- Full: Phase 2 ---
    for key, desc in PHASE2_KEYWORDS.items():
        val = hdr.get(key)
        if val is None:
            report.info.append(f"Missing Phase 2 keyword {key} ({desc})")
        else:
            report.keywords_found[key] = val

    # TIMESYS check
    timesys = hdr.get('TIMESYS')
    if timesys and timesys.upper() != 'UTC':
        report.warnings.append(
            f"TIMESYS='{timesys}' — expected 'UTC'")

    # Check HISTORY for WSClean
    history = str(hdr.get('HISTORY', ''))
    if 'wsclean' not in history.lower() and 'WSCLEAN' not in str(hdr):
        report.info.append("No WSClean HISTORY found in header")

    return report


def validate_directory(directory, pattern="*.fits", level='standard',
                       recursive=False, logger_in=None):
    """Validate all FITS files in a directory.

    Parameters
    ----------
    directory : str
        Directory to scan.
    pattern : str
        Glob pattern for FITS files.
    level : str
        Validation level ('critical', 'standard', 'full').
    recursive : bool
        Search subdirectories.
    logger_in : logging.Logger, optional

    Returns
    -------
    list of HeaderReport
    """
    log = logger_in or logger

    if recursive:
        fits_files = sorted(glob.glob(
            os.path.join(directory, "**", pattern), recursive=True))
    else:
        fits_files = sorted(glob.glob(os.path.join(directory, pattern)))

    if not fits_files:
        log.warning(f"No FITS files matching '{pattern}' in {directory}")
        return []

    reports = []
    for f in fits_files:
        report = validate_header(f, level=level, logger_in=log)
        reports.append(report)

    # Summary
    n_ok = sum(1 for r in reports if r.ok)
    n_clean = sum(1 for r in reports if r.clean)
    n_fail = sum(1 for r in reports if not r.ok)
    n_warn = sum(1 for r in reports if r.ok and not r.clean)

    log.info(f"Header validation: {len(reports)} files checked — "
             f"{n_clean} clean, {n_warn} warnings, {n_fail} errors")

    if n_fail > 0:
        log.error("Files with critical errors:")
        for r in reports:
            if not r.ok:
                log.error(f"  {r.full_report()}")

    return reports


def validate_checkpoint(fits_paths, checkpoint_name, level='critical',
                        abort_on_error=True, logger_in=None):
    """Validate headers at a pipeline checkpoint.

    Designed to be called inline in the controller, e.g.:
        validate_checkpoint(image_list, "post-dewarp", abort_on_error=True)

    Parameters
    ----------
    fits_paths : list of str
        FITS files to check.
    checkpoint_name : str
        Label for log messages (e.g., 'post-copy', 'post-dewarp').
    level : str
        Validation level.
    abort_on_error : bool
        If True, raise RuntimeError on critical failures.
    logger_in : logging.Logger, optional

    Returns
    -------
    bool : True if all files pass.
    """
    log = logger_in or logger

    if not fits_paths:
        return True

    reports = []
    for f in fits_paths:
        if os.path.exists(f):
            reports.append(validate_header(f, level=level, logger_in=log))

    n_fail = sum(1 for r in reports if not r.ok)

    if n_fail > 0:
        log.error(f"[{checkpoint_name}] {n_fail}/{len(reports)} files "
                  f"failed header validation")
        for r in reports:
            if not r.ok:
                for e in r.errors:
                    log.error(f"  {r.filename}: {e}")

        if abort_on_error:
            raise RuntimeError(
                f"Header validation failed at {checkpoint_name}: "
                f"{n_fail} files with critical errors. "
                f"Fix headers before proceeding.")
        return False

    n_warn = sum(1 for r in reports if not r.clean)
    if n_warn > 0:
        log.warning(f"[{checkpoint_name}] {n_warn}/{len(reports)} files "
                    f"have non-critical header warnings")

    return True


# =====================================================================
# HEADER CORRECTION (Phase 2 ingest step)
# =====================================================================

# LWA integration time (seconds) — verified from MS TIME column diagnostic
LWA_DT = 10.0310

def _build_phase1_time_lookup(subband_src_map, log):
    """Read pristine start times and provenance from the Phase 1 archive.

    Phase 1 image headers on Lustre are never modified by Phase 2 (the
    corrections run on the NVMe copies), so their WSClean DATE-OBS is
    the exact start of the first integration. Used in repair mode to
    recover starts that the old correction overwrote.

    Returns (starts, n_valid):
      starts[(subband, pol, cat)] = {filename_timestamp: Time}
      n_valid[subband] = valid MS count from provenance.json, or absent
    """
    import json
    from astropy.time import Time

    starts, n_valid = {}, {}
    for sb, src in (subband_src_map or {}).items():
        prov_path = os.path.join(src, 'provenance.json')
        if os.path.isfile(prov_path):
            try:
                with open(prov_path) as fh:
                    prov = json.load(fh)
                nv = prov.get('n_valid_ms') or prov.get('n_ms_files')
                if nv:
                    n_valid[sb] = int(nv)
            except Exception:
                pass

        for pol in ['I', 'V']:
            for cat in ['10min', 'deep']:
                d = os.path.join(src, pol, cat)
                if not os.path.isdir(d):
                    continue
                seen = {}
                for f in sorted(glob.glob(os.path.join(d, "*image*.fits"))):
                    ts = _filename_timestamp(os.path.basename(f))
                    if not ts or ts in seen:
                        continue
                    try:
                        hdr = fits.getheader(f)
                    except Exception:
                        continue
                    # Only trust pristine WSClean headers
                    if 'DURATION' not in hdr and hdr.get('DATE-OBS'):
                        try:
                            seen[ts] = Time(hdr['DATE-OBS'])
                        except Exception:
                            pass
                if seen:
                    starts[(sb, pol, cat)] = seen

    if starts:
        log.info(f"  Phase 1 time lookup: pristine starts for "
                 f"{len(starts)} subband/pol/category groups, provenance "
                 f"counts for {len(n_valid)} subbands")
    return starts, n_valid


def correct_interval_headers(work_dir, logger_in=None, subband_src_map=None,
                             rename=True, force_v_beam=False):
    """Correct timing headers for all images in a Phase 2 work directory.

    Filename timestamps are interval MIDPOINTS (Phase 1 convention), so
    start times are recovered from WSClean's original DATE-OBS (exact),
    the Phase 1 archive (repair mode), or midpoint arithmetic as a last
    resort. Writes the standard time block (see _stamp_times) in-place,
    including dewarped copies.

    Also matches V beams to the correct I variant, adds exact frequency
    metadata, optionally renames files to the standardized format, and
    checks cross-subband interval alignment.

    Parameters
    ----------
    work_dir : str
        Phase 2 run directory.
    logger_in : logging.Logger, optional
    subband_src_map : dict, optional
        {subband: phase1_subband_dir} (from discover_subbands). Enables
        exact start recovery and provenance integration counts.
    rename : bool
        Run filename standardization (disable for archive repair so
        existing references stay valid).
    force_v_beam : bool
        Re-copy V beams even when already set (repair mode).
    """
    from exopipe.common import phase2_config as cfg

    log = logger_in or logger

    log.info("\n" + "=" * 60)
    log.info("HEADER VALIDATION AND CORRECTION")
    log.info("=" * 60)

    p1_starts, p1_nvalid = _build_phase1_time_lookup(subband_src_map, log)

    n_corrected = 0

    # Collect all 10-min start times per subband for cross-check
    cross_subband = {}  # {interval_index: {subband: start_time_mjd}}

    for subband in cfg.ALL_SUBBANDS:
        nv = p1_nvalid.get(subband)
        for pol in ['I', 'V']:
            # --- 10-min images (also yields the interval solution) ---
            tenmin_dir = os.path.join(work_dir, subband, pol, '10min')
            sol = []
            if os.path.isdir(tenmin_dir):
                n, sol = _correct_10min_headers(
                    tenmin_dir, subband, pol, cross_subband, log,
                    p1_starts=p1_starts.get((subband, pol, '10min')),
                    n_valid_total=nv)
                n_corrected += n

            # --- Deep images ---
            deep_dir = os.path.join(work_dir, subband, pol, 'deep')
            if os.path.isdir(deep_dir):
                n = _correct_deep_headers(
                    deep_dir, subband, pol, log,
                    p1_starts=p1_starts.get((subband, pol, 'deep')),
                    n_valid_total=nv, tenmin_solution=sol)
                n_corrected += n

            # --- Snapshots ---
            snap_dir = os.path.join(work_dir, subband, pol, 'snapshots')
            if os.path.isdir(snap_dir):
                n = _correct_snapshot_headers(snap_dir, subband, pol, log)
                n_corrected += n

    # Also correct the V beam from the variant-matched I image
    _correct_v_beam(work_dir, log, force=force_v_beam)

    # Add exact frequency metadata from config
    _correct_frequency_headers(work_dir, log)

    # Rename files to standardized format
    n_renamed = _standardize_filenames(work_dir, log) if rename else 0

    # Cross-subband consistency check
    _check_cross_subband_consistency(cross_subband, log)

    log.info(f"\nHeader correction complete: {n_corrected} headers updated, "
             f"{n_renamed} files renamed")
    return n_corrected



def _correct_v_beam(work_dir, log, force=False):
    """Copy beam keywords from the matching I image to each V image.

    Phase 1 writes BMAJ=BMIN=0 for Stokes V. The beam depends on the
    weighting/taper AND on the data interval, so the I donor is matched
    on (taper, robust, timestamp); any other variant introduces ~1%
    beam errors.

    force=True re-copies even when BMAJ is already set (repair mode —
    the old version copied from an arbitrary I variant).
    """
    from exopipe.common import phase2_config as cfg

    n_fixed = 0
    for subband in cfg.ALL_SUBBANDS:
        for cat in ['deep', '10min']:
            i_dir = os.path.join(work_dir, subband, 'I', cat)
            v_dir = os.path.join(work_dir, subband, 'V', cat)
            if not os.path.isdir(i_dir) or not os.path.isdir(v_dir):
                continue

            # Beam lookup from I images: (taper, robust, timestamp) -> beam
            beams = {}
            for f in sorted(glob.glob(os.path.join(i_dir, "*-image*.fits"))):
                bn = os.path.basename(f)
                if '_dewarped' in bn:
                    continue
                parsed = _parse_old_filename(bn)
                if not parsed:
                    continue
                key = (parsed['taper'], parsed['robust'],
                       parsed['timestamp'])
                if key in beams:
                    continue
                try:
                    hdr = fits.getheader(f)
                except Exception:
                    continue
                bmaj = hdr.get('BMAJ', 0) or 0
                if bmaj > 0:
                    beams[key] = (bmaj, hdr.get('BMIN', bmaj),
                                  hdr.get('BPA', 0))

            if not beams:
                continue

            # V files, including dewarped copies (repair mode)
            for f in sorted(glob.glob(os.path.join(v_dir, "*-image*.fits"))):
                bn = os.path.basename(f)
                parsed = _parse_old_filename(bn)
                if not parsed:
                    continue
                key = (parsed['taper'], parsed['robust'],
                       parsed['timestamp'])
                beam = beams.get(key)
                note = 'matched I image'
                if beam is None:
                    # Same interval, different weighting — better than
                    # nothing, but warn.
                    cands = [v for k, v in sorted(beams.items())
                             if k[2] == parsed['timestamp']]
                    if cands:
                        beam = cands[0]
                        note = 'I image (variant fallback)'
                        log.warning(f"  V beam: no exact I match for {bn}, "
                                    f"using variant fallback")
                if beam is None:
                    continue
                try:
                    with fits.open(f, mode='update') as hdul:
                        hdr = hdul[0].header
                        if force or hdr.get('BMAJ', 0) == 0:
                            hdr['BMAJ'] = (beam[0],
                                           f'[deg] Beam major axis ({note})')
                            hdr['BMIN'] = (beam[1],
                                           f'[deg] Beam minor axis ({note})')
                            hdr['BPA'] = (beam[2],
                                          f'[deg] Beam PA ({note})')
                            hdul.flush()
                            n_fixed += 1
                except Exception:
                    pass

    if n_fixed > 0:
        log.info(f"  V beam correction: fixed {n_fixed} files "
                 f"(variant-matched from I)")


def _correct_frequency_headers(work_dir, log):
    """Add exact frequency metadata to all subband images from config.

    Sets RESTFRQ (center), FREQ_MIN (lower edge), FREQ_MAX (upper edge),
    BANDWDTH, and ensures a proper CTYPE3='FREQ' WCS axis exists so that
    BDSF and other tools can determine the observing frequency.

    Applies to both raw and dewarped images.
    """
    from exopipe.common import phase2_config as cfg

    n_fixed = 0
    for subband in cfg.ALL_SUBBANDS:
        freq_info = cfg.SUBBAND_FREQ_HZ.get(subband)
        if not freq_info:
            continue

        for pol in ['I', 'V']:
            for cat in ['deep', '10min', 'snapshots']:
                img_dir = os.path.join(work_dir, subband, pol, cat)
                if not os.path.isdir(img_dir):
                    continue

                for f in glob.glob(os.path.join(img_dir, "*.fits")):
                    try:
                        with fits.open(f, mode='update') as hdul:
                            hdr = hdul[0].header
                            changed = False

                            if hdr.get('RESTFRQ') is None or hdr.get('RESTFRQ', 0) == 0:
                                hdr['RESTFRQ'] = (freq_info['center'],
                                                   '[Hz] Subband center frequency')
                                changed = True
                            if hdr.get('FREQ_MIN') is None:
                                hdr['FREQ_MIN'] = (freq_info['lo'],
                                                    '[Hz] Subband lower edge')
                                changed = True
                            if hdr.get('FREQ_MAX') is None:
                                hdr['FREQ_MAX'] = (freq_info['hi'],
                                                    '[Hz] Subband upper edge')
                                changed = True
                            if hdr.get('BANDWDTH') is None:
                                hdr['BANDWDTH'] = (freq_info['bw'],
                                                    '[Hz] Subband bandwidth')
                                changed = True

                            # Ensure FREQ WCS axis exists
                            if hdr.get('CTYPE3') != 'FREQ' or \
                               not hdr.get('CRVAL3'):
                                hdr['CTYPE3'] = ('FREQ', 'Frequency axis')
                                hdr['CRVAL3'] = (freq_info['center'],
                                                  '[Hz] Subband center')
                                hdr['CDELT3'] = (freq_info['bw'],
                                                  '[Hz] Subband bandwidth')
                                hdr['CRPIX3'] = (1.0, 'Reference pixel')
                                hdr['CUNIT3'] = ('Hz', 'Frequency unit')
                                changed = True

                            if changed:
                                hdul.flush()
                                n_fixed += 1
                    except Exception:
                        pass

    if n_fixed > 0:
        log.info(f"  Frequency metadata: set RESTFRQ/FREQ_MIN/FREQ_MAX/BANDWDTH"
                 f"/CTYPE3 for {n_fixed} files")


STOKES_CODE = {'I': 1, 'Q': 2, 'U': 3, 'V': 4}


def repair_wideband_headers(run_dir, log=None, dry_run=False):
    """Repair FREQ/Stokes WCS axes on wideband products (Red/Green/Blue/Full).

    These products were written by _stack_images before the FREQ WCS fix.
    They have RESTFRQ+BANDWDTH scalars but no CTYPE3='FREQ' axis, so BDSF
    cannot determine the frequency. This repairs them in-place.

    Parameters
    ----------
    run_dir : str
        Phase 2 run directory (e.g., /fast/gh/main/phase2/10h/2025-01-01/Science_...)
    dry_run : bool
        If True, report what would be fixed without modifying files.

    Returns
    -------
    int
        Number of files repaired.
    """
    from exopipe.common import phase2_config as cfg
    import re

    if log is None:
        log = logging.getLogger('phase2')

    n_fixed = 0
    for band in ['Red', 'Green', 'Blue', 'Full']:
        band_dir = os.path.join(run_dir, band)
        if not os.path.isdir(band_dir):
            continue

        band_hz = cfg.WIDEBAND_BANDS_HZ.get(band)

        for root, dirs, files in os.walk(band_dir):
            for fname in files:
                if not fname.endswith('.fits'):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with fits.open(fpath, mode='update') as hdul:
                        hdr = hdul[0].header
                        changed = False

                        restfrq = hdr.get('RESTFRQ', 0)
                        bandwdth = hdr.get('BANDWDTH', 0)

                        # Fill missing frequency scalars from config
                        if (not restfrq or restfrq == 0) and band_hz:
                            restfrq = (band_hz[0] + band_hz[1]) / 2.0
                            hdr['RESTFRQ'] = (restfrq,
                                               '[Hz] Band center frequency')
                            changed = True
                        if (not bandwdth or bandwdth == 0) and band_hz:
                            bandwdth = band_hz[1] - band_hz[0]
                            hdr['BANDWDTH'] = (bandwdth,
                                                '[Hz] Total bandwidth')
                            changed = True
                        if hdr.get('FREQ_MIN') is None and band_hz:
                            hdr['FREQ_MIN'] = (band_hz[0],
                                                '[Hz] Lower edge')
                            changed = True
                        if hdr.get('FREQ_MAX') is None and band_hz:
                            hdr['FREQ_MAX'] = (band_hz[1],
                                                '[Hz] Upper edge')
                            changed = True

                        # Add FREQ WCS axis if missing
                        if hdr.get('CTYPE3') != 'FREQ' and restfrq > 0:
                            if dry_run:
                                log.info(f"  [dry-run] would fix: {fname}")
                                n_fixed += 1
                                continue
                            hdr['CTYPE3'] = ('FREQ', 'Frequency axis')
                            hdr['CRVAL3'] = (float(restfrq),
                                              '[Hz] Band center frequency')
                            hdr['CDELT3'] = (float(bandwdth) if bandwdth
                                              else 1.0, '[Hz] Total bandwidth')
                            hdr['CRPIX3'] = (1.0, 'Reference pixel')
                            hdr['CUNIT3'] = ('Hz', 'Frequency unit')
                            changed = True

                        # Add Stokes WCS axis if missing
                        if hdr.get('CTYPE4') != 'STOKES':
                            pol_match = re.search(
                                r'[_/]([IVQU])(?:[_/]|$)', fname)
                            if pol_match:
                                sc = STOKES_CODE.get(pol_match.group(1))
                                if sc is not None:
                                    hdr['CTYPE4'] = ('STOKES', 'Stokes axis')
                                    hdr['CRVAL4'] = (
                                        float(sc),
                                        f'Stokes {pol_match.group(1)}')
                                    hdr['CDELT4'] = (1.0, 'Stokes increment')
                                    hdr['CRPIX4'] = (1.0, 'Reference pixel')
                                    changed = True

                        if changed:
                            hdul.flush()
                            n_fixed += 1

                except Exception as e:
                    log.warning(f"  Repair failed for {fname}: {e}")

    if n_fixed > 0:
        log.info(f"  Wideband header repair: fixed {n_fixed} files in "
                 f"{os.path.basename(run_dir)}")
    return n_fixed


def _build_standard_name(freq, stokes, taper, duration_s, robust,
                          timestamp, product='image', pbcorr=False,
                          dewarped=False):
    """Build a standardized filename.

    Format: FREQ-STOKES-TAPER-DURATIONs-ROBUST-TIMESTAMP-PRODUCT[.pbcorr][_dewarped].fits
    e.g. 69MHz-I-Taper-592s-Robust-0-20241218_120740-image.fits
    """
    dur_str = f"{int(round(duration_s))}s"
    parts = [freq, stokes, taper, dur_str, robust, timestamp, product]
    name = '-'.join(parts)
    if pbcorr:
        name += '.pbcorr'
    if dewarped:
        name += '_dewarped'
    name += '.fits'
    return name


def _parse_old_filename(filename):
    """Parse an existing filename into its components.

    Returns dict or None if unparseable.
    """
    import re

    name = filename
    dewarped = '_dewarped' in name
    name = name.replace('_dewarped', '')

    pbcorr = '.pbcorr' in name
    name = name.replace('.pbcorr', '')

    if name.endswith('.fits'):
        name = name[:-5]

    timestamps = re.findall(r'\d{8}_\d{6}', name)
    if not timestamps:
        return None

    timestamp = timestamps[0]

    freq_match = re.match(r'^(\d+MHz)', name)
    if not freq_match:
        return None
    freq = freq_match.group(1)

    stokes = None
    for s in ['I', 'V', 'Q', 'U']:
        if f'-{s}-' in name:
            stokes = s
            break
    if not stokes:
        return None

    if '-NoTaper-' in name:
        taper = 'NoTaper'
    elif '-Taper-' in name:
        taper = 'Taper'
    else:
        taper = 'Taper'

    robust_match = re.search(r'Robust-([\d.]+)', name)
    robust = f"Robust-{robust_match.group(1)}" if robust_match else 'Robust-0'

    if '-Deep-' in name or name.endswith('-Deep') or '-Deep' in name.split('-'):
        duration_label = 'Deep'
    elif '-10min-' in name:
        duration_label = '10min'
    else:
        duration_label = 'unknown'

    product = 'image'
    for p in ['dirty', 'residual', 'model', 'psf', 'image']:
        if f'-{p}' in name:
            product = p
            break

    return {
        'freq': freq, 'stokes': stokes, 'taper': taper,
        'duration_label': duration_label, 'robust': robust,
        'timestamp': timestamp, 'product': product,
        'pbcorr': pbcorr, 'dewarped': dewarped,
    }


def _standardize_filenames(work_dir, log):
    """Rename all image files to the standardized naming convention.

    Standard: FREQ-STOKES-TAPER-DURATIONs-ROBUST-TIMESTAMP-PRODUCT[.pbcorr][_dewarped].fits
    Reads DURATION from the corrected headers. Writes a rename mapping file.
    """
    from exopipe.common import phase2_config as cfg

    log.info("\n--- Filename Standardization ---")

    renames = []
    n_renamed = 0

    for subband in cfg.ALL_SUBBANDS:
        for pol in ['I', 'V']:
            for cat in ['deep', '10min', 'snapshots']:
                img_dir = os.path.join(work_dir, subband, pol, cat)
                if not os.path.isdir(img_dir):
                    continue

                fits_files = sorted(glob.glob(os.path.join(img_dir, "*.fits")))

                for fpath in fits_files:
                    bn = os.path.basename(fpath)
                    parsed = _parse_old_filename(bn)
                    if parsed is None:
                        continue

                    try:
                        hdr = fits.getheader(fpath)
                        duration = hdr.get('DURATION')
                    except Exception:
                        duration = None

                    if duration is None or duration <= 0:
                        continue

                    new_name = _build_standard_name(
                        freq=parsed['freq'], stokes=parsed['stokes'],
                        taper=parsed['taper'], duration_s=duration,
                        robust=parsed['robust'], timestamp=parsed['timestamp'],
                        product=parsed['product'], pbcorr=parsed['pbcorr'],
                        dewarped=parsed['dewarped'],
                    )

                    if new_name == bn:
                        continue

                    new_path = os.path.join(img_dir, new_name)
                    if os.path.exists(new_path):
                        continue

                    os.rename(fpath, new_path)
                    renames.append((bn, new_name))
                    n_renamed += 1

    if renames:
        mapping_path = os.path.join(work_dir, "filename_renames.txt")
        with open(mapping_path, 'w') as fh:
            fh.write(f"# Filename standardization: {len(renames)} files renamed\n")
            fh.write(f"# Format: FREQ-STOKES-TAPER-DURATIONs-ROBUST-TIMESTAMP-PRODUCT"
                     f"[.pbcorr][_dewarped].fits\n#\n")
            for old, new in renames:
                fh.write(f"{old}  ->  {new}\n")
        log.info(f"  Renamed {n_renamed} files (mapping in filename_renames.txt)")
    else:
        log.info("  No files needed renaming")

    return n_renamed


TIMEFIX_VERSION = 'v2'


def _filename_timestamp(filename):
    """Extract the first YYYYMMDD_HHMMSS timestamp string, or None.

    IMPORTANT: Phase 1 filename timestamps are interval MIDPOINTS, not
    start times. pipeline_utils.add_timestamps_to_images() names each
    image by the center of its uniform time chunk over the MS TIME span
    (deep images: the midpoint of the whole hour). The true start time
    lives in WSClean's original DATE-OBS, which equals the MS TIME
    column value of the first integration (verified to be integration
    START times at OVRO-LWA, matching the raw MS filenames).
    """
    import re
    m = re.findall(r'\d{8}_\d{6}', filename)
    return m[0] if m else None


def _ts_to_time(ts):
    """Convert a YYYYMMDD_HHMMSS string to an astropy Time (UTC)."""
    from astropy.time import Time
    return Time(f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T"
                f"{ts[9:11]}:{ts[11:13]}:{ts[13:15]}")


def _stamp_times(hdr, start, end, duration_s, inttime_s, basis, pol):
    """Write the standard time block into a header.

    DATE-OBS/MJD-OBS  start of the first integration included
    DATE-END          end of the last integration included
    DURATION          wall-clock span = DATE-END - DATE-OBS [s]
    INTTIME           summed integration time actually included [s];
                      equals DURATION for a contiguous single-epoch
                      image, but is independent in general (a multi-
                      night LST stack has DURATION of days and INTTIME
                      of hours)
    MJD-MID           midpoint of [DATE-OBS, DATE-END]
    TIMEFIX           correction version + basis of the start time
    """
    if hdr.get('DATE-OBS') and 'WSCDATEO' not in hdr \
            and 'DURATION' not in hdr:
        # Preserve WSClean's original stamp before the first rewrite
        hdr['WSCDATEO'] = (hdr['DATE-OBS'], 'Original WSClean DATE-OBS')
    hdr['DATE-OBS'] = (start.isot, 'Start of first integration (UTC)')
    hdr['MJD-OBS'] = (start.mjd, 'Start MJD (UTC)')
    hdr['DATE-END'] = (end.isot, 'End of last integration (UTC)')
    hdr['MJD-END'] = (end.mjd, 'End MJD (UTC)')   # 2026-09-27: never leave an inherited MJD-END beside a new DATE-END
    hdr['DURATION'] = (float(duration_s),
                       '[s] Wall-clock span (DATE-END - DATE-OBS)')
    hdr['INTTIME'] = (float(inttime_s),
                      '[s] Summed integration time included')
    hdr['MJD-MID'] = ((start.mjd + end.mjd) / 2.0, 'Midpoint MJD (UTC)')
    hdr['TIMESYS'] = ('UTC', 'Time system')
    hdr['TIMEFIX'] = (f'{TIMEFIX_VERSION}-{basis}',
                      'Time correction version-basis')
    if 'STOKES' not in hdr:
        hdr['STOKES'] = (pol, 'Stokes parameter')


def _group_start_time(files, p1_starts, ts):
    """Determine the true start time for one interval group of files.

    Preference order:
      1. 'phase1'   — pristine DATE-OBS from the Phase 1 archive
                      (repair mode; exact)
      2. 'wsclean'  — the file's own DATE-OBS if Phase 2 never touched
                      it (no DURATION keyword; exact)
      3. 'priorfix' — DATE-OBS of a file already corrected by this
                      version (TIMEFIX present; idempotent re-run)
      4. None       — caller falls back to midpoint arithmetic
    Returns (Time or None, basis or None).
    """
    if p1_starts and ts in p1_starts:
        return p1_starts[ts], 'phase1'
    from astropy.time import Time

    priorfix = None
    for f in sorted(files, key=lambda x: '_dewarped' in os.path.basename(x)):
        try:
            hdr = fits.getheader(f)
        except Exception:
            continue
        if not hdr.get('DATE-OBS'):
            continue
        if 'DURATION' not in hdr:
            return Time(hdr['DATE-OBS']), 'wsclean'
        if priorfix is None and str(hdr.get('TIMEFIX', '')).startswith(
                TIMEFIX_VERSION):
            priorfix = Time(hdr['DATE-OBS'])
    if priorfix is not None:
        return priorfix, 'priorfix'
    return None, None


def _solve_interval_times(img_dir, p1_starts=None, n_valid_total=None):
    """Build the time solution for the 10-min intervals in one directory.

    Filename timestamps are interval midpoints (see _filename_timestamp),
    so the true start of each interval is recovered per _group_start_time;
    only files corrupted by the old correction (which wrote DATE-OBS =
    midpoint) with no Phase 1 archive available fall back to
    start = midpoint - duration/2 (~3 s accuracy).

    Per-interval integration counts come from gaps between consecutive
    starts (exact when starts are known), with the last interval taken
    from n_valid_total (provenance.json) when available, else the mode
    of the other intervals.

    Returns a list of dicts sorted by time:
      {'ts', 'files', 'mid', 'start', 'basis', 'n_integ', 'duration'}
    """
    from astropy.time import TimeDelta
    from collections import Counter

    fits_files = sorted(glob.glob(os.path.join(img_dir, "*-image*.fits")))
    if not fits_files:
        return []

    groups = {}
    for f in fits_files:
        ts = _filename_timestamp(os.path.basename(f))
        if ts:
            groups.setdefault(ts, []).append(f)
    if not groups:
        return []

    intervals = []
    for ts, files in groups.items():
        start, basis = _group_start_time(files, p1_starts, ts)
        intervals.append({'ts': ts, 'files': files, 'mid': _ts_to_time(ts),
                          'start': start, 'basis': basis})
    intervals.sort(key=lambda d: d['mid'].mjd)

    # Integration counts from gaps between consecutive reference times.
    # Start gaps are exact interval lengths; midpoint gaps are the
    # uniform-chunk approximation (only used when starts are unknown).
    all_starts = all(d['start'] is not None for d in intervals)
    refs = [(d['start'] if all_starts else d['mid']).mjd for d in intervals]
    n_list = []
    for i in range(len(intervals) - 1):
        gap_s = (refs[i + 1] - refs[i]) * 86400.0
        n_list.append(int(round(gap_s / LWA_DT)))

    n_last = None
    if n_valid_total and n_list:
        n_last = int(n_valid_total) - sum(n_list)
        if n_last <= 0 or n_last > 2 * max(n_list):
            n_last = None  # provenance inconsistent with gaps
    if n_last is None:
        n_last = (Counter(n_list).most_common(1)[0][0] if n_list
                  else int(round(597.0 / LWA_DT)))
    n_list.append(n_last)

    for d, n in zip(intervals, n_list):
        d['n_integ'] = n
        d['duration'] = n * LWA_DT
        if d['start'] is None:
            d['start'] = d['mid'] - TimeDelta(d['duration'] / 2.0,
                                              format='sec')
            d['basis'] = 'midpoint'
    return intervals


def _correct_10min_headers(img_dir, subband, pol, cross_subband, log,
                           p1_starts=None, n_valid_total=None):
    """Correct timing headers for 10-min interval images.

    Returns (n_corrected, interval_solution) — the solution is reused
    by _correct_deep_headers for the deep image span.
    """
    from astropy.time import TimeDelta

    intervals = _solve_interval_times(img_dir, p1_starts, n_valid_total)
    if not intervals:
        return 0, []

    n_corrected = 0
    for idx, iv in enumerate(intervals):
        cross_subband.setdefault(idx, {})[subband] = iv['start'].mjd
        end_time = iv['start'] + TimeDelta(iv['duration'], format='sec')
        for f in iv['files']:
            try:
                with fits.open(f, mode='update') as hdul:
                    _stamp_times(hdul[0].header, iv['start'], end_time,
                                 iv['duration'], iv['n_integ'] * LWA_DT,
                                 iv['basis'], pol)
                    hdul.flush()
                    n_corrected += 1
            except Exception as e:
                log.warning(f"  Failed to correct {os.path.basename(f)}: {e}")

    if n_corrected > 0:
        bases = sorted(set(iv['basis'] for iv in intervals))
        log.info(f"  {subband}/{pol}/10min: corrected {n_corrected} files "
                 f"({len(intervals)} intervals, start basis: "
                 f"{'/'.join(bases)})")

    return n_corrected, intervals


def _correct_deep_headers(img_dir, subband, pol, log, p1_starts=None,
                          n_valid_total=None, tenmin_solution=None):
    """Correct timing headers for deep (1-hr) images.

    The deep filename timestamp is the midpoint of the whole hour
    (exactly start + (N-1)*dt/2, the mean of the MS TIME values), so
    the true start comes from _group_start_time, falling back to the
    first 10-min interval start (same MS, same start) and finally to
    midpoint arithmetic.
    """
    from astropy.time import TimeDelta

    fits_files = sorted(glob.glob(os.path.join(img_dir, "*-image*.fits")))
    if not fits_files:
        return 0

    ts = _filename_timestamp(os.path.basename(fits_files[0]))
    sol = tenmin_solution or []

    # Total integrations: provenance count is authoritative, else the
    # sum over the 10-min intervals, else the nominal hour.
    if n_valid_total:
        n_integ = int(n_valid_total)
    elif sol:
        n_integ = sum(iv['n_integ'] for iv in sol)
    else:
        n_integ = 358
    inttime = n_integ * LWA_DT

    start, basis = _group_start_time(fits_files, p1_starts, ts)
    if start is None and sol:
        # The deep image and the first 10-min interval share the same
        # first integration.
        start, basis = sol[0]['start'], '10min'
    if start is None and ts:
        start = _ts_to_time(ts) - TimeDelta((n_integ - 1) * LWA_DT / 2.0,
                                            format='sec')
        basis = 'midpoint'
    if start is None:
        return 0

    # Wall-clock span from the interval solution (captures recorder
    # gaps); contiguous assumption otherwise.
    if sol:
        end_time = sol[-1]['start'] + TimeDelta(sol[-1]['duration'],
                                                format='sec')
        if (sol[0]['start'] - start).sec > 2.0 or \
                (start - sol[0]['start']).sec > 2.0:
            log.warning(f"  {subband}/{pol}/deep: start differs from first "
                        f"10-min interval by "
                        f"{abs((start - sol[0]['start']).sec):.1f}s")
        duration = (end_time - start).sec
    else:
        duration = inttime
        end_time = start + TimeDelta(duration, format='sec')

    n_corrected = 0
    for f in fits_files:
        try:
            with fits.open(f, mode='update') as hdul:
                _stamp_times(hdul[0].header, start, end_time,
                             duration, inttime, basis, pol)
                hdul.flush()
                n_corrected += 1
        except Exception as e:
            log.warning(f"  Failed to correct {os.path.basename(f)}: {e}")

    if n_corrected > 0:
        log.info(f"  {subband}/{pol}/deep: corrected {n_corrected} files "
                 f"(start {start.isot}, duration {duration:.1f}s, "
                 f"inttime {inttime:.1f}s, basis {basis})")

    return n_corrected


def _correct_snapshot_headers(img_dir, subband, pol, log):
    """Correct headers for snapshot (single integration) images.

    Snapshot filenames carry the MS timestamp, which is the integration
    START (unlike deep/10min names, which are midpoints) — but a
    pristine WSClean DATE-OBS is still preferred when present.
    """
    from astropy.time import Time, TimeDelta

    fits_files = sorted(glob.glob(os.path.join(img_dir, "*-image*.fits")))
    if not fits_files:
        return 0

    n_corrected = 0
    for f in fits_files:
        ts = _filename_timestamp(os.path.basename(f))
        if not ts:
            continue
        try:
            with fits.open(f, mode='update') as hdul:
                hdr = hdul[0].header

                if 'DURATION' not in hdr and hdr.get('DATE-OBS'):
                    start, basis = Time(hdr['DATE-OBS']), 'wsclean'
                else:
                    start, basis = _ts_to_time(ts), 'filename'
                end_time = start + TimeDelta(LWA_DT, format='sec')

                _stamp_times(hdr, start, end_time, LWA_DT, LWA_DT,
                             basis, pol)
                hdul.flush()
                n_corrected += 1
        except Exception as e:
            log.warning(f"  Failed to correct {os.path.basename(f)}: {e}")

    if n_corrected > 0:
        log.info(f"  {subband}/{pol}/snapshots: corrected {n_corrected} files "
                 f"(duration {LWA_DT}s each)")

    return n_corrected


def _check_cross_subband_consistency(cross_subband, log):
    """Check that 10-min intervals are aligned across subbands.

    Warns prominently if any interval has >30s spread in start times
    across subbands.
    """
    log.info("\n--- Cross-subband interval alignment ---")

    if not cross_subband:
        log.info("  No 10-min intervals to check")
        return

    n_warn = 0
    for interval_idx in sorted(cross_subband.keys()):
        subband_starts = cross_subband[interval_idx]
        if len(subband_starts) < 2:
            continue

        mjds = list(subband_starts.values())
        spread_s = (max(mjds) - min(mjds)) * 86400.0

        if spread_s > 30.0:
            n_warn += 1
            earliest = min(subband_starts.items(), key=lambda x: x[1])
            latest = max(subband_starts.items(), key=lambda x: x[1])
            log.warning(
                f"  *** WARNING: Interval {interval_idx} misaligned by "
                f"{spread_s:.1f}s across subbands ***")
            log.warning(
                f"      Earliest: {earliest[0]} at MJD {earliest[1]:.8f}")
            log.warning(
                f"      Latest:   {latest[0]} at MJD {latest[1]:.8f}")
            log.warning(
                f"      Wideband stacking for this interval may mix "
                f"different time windows")

    if n_warn == 0:
        log.info(f"  All {len(cross_subband)} intervals aligned across "
                 f"subbands (spread < 30s)")
    else:
        log.warning(f"\n  *** {n_warn} intervals misaligned by >30s — "
                    f"check before stacking ***")


# =====================================================================
# PHASE 1 AUDIT
# =====================================================================

def audit_phase1(lst, date, run_dir=None, logger_in=None):
    """Audit all Phase 1 image headers that the Phase 2 pipeline would copy.

    Uses the same file discovery as phase2_controller.py Step 1.
    Checks individual headers plus cross-file consistency.

    Parameters
    ----------
    lst : str
        LST label (e.g., '10h').
    date : str
        Date string (e.g., '2024-12-18').
    run_dir : str, optional
        Specific Run directory. If None, uses the most recent.
    logger_in : logging.Logger, optional

    Returns
    -------
    dict with keys: 'reports', 'cross_checks', 'summary'
    """
    from exopipe.common import phase2_config as cfg
    from astropy.time import Time

    log = logger_in or logger

    # Find the Run directory (same logic as controller)
    lustre_base = os.path.join(cfg.CELERY_IMAGE_ROOT, lst, date)
    if run_dir is None:
        run_dirs = sorted(glob.glob(os.path.join(lustre_base, "Run_*")))
        if not run_dirs:
            log.error(f"No Run directories in {lustre_base}")
            return None
        run_dir = run_dirs[-1]

    log.info(f"{'='*70}")
    log.info(f"PHASE 1 HEADER AUDIT")
    log.info(f"  LST: {lst}  Date: {date}")
    log.info(f"  Run: {run_dir}")
    log.info(f"{'='*70}")

    # Discover files the same way the controller does: *-image*.fits
    all_reports = []
    file_metadata = []  # for cross-file checks

    for subband in cfg.ALL_SUBBANDS:
        for pol in ['I', 'V']:
            for cat in ['deep', '10min']:
                img_dir = os.path.join(run_dir, subband, pol, cat)
                if not os.path.isdir(img_dir):
                    continue

                fits_files = sorted(glob.glob(
                    os.path.join(img_dir, "*-image*.fits")))
                # Exclude dewarped, pbcorr_dewarped (Phase 2 products)
                fits_files = [f for f in fits_files
                              if '_dewarped' not in os.path.basename(f)]

                for fpath in fits_files:
                    report = validate_header(fpath, level='full',
                                             logger_in=log)
                    all_reports.append(report)

                    # Extract metadata for cross-file checks
                    try:
                        hdr = fits.getheader(fpath)
                        bn = os.path.basename(fpath)
                        meta = {
                            'path': fpath,
                            'filename': bn,
                            'subband': subband,
                            'pol': pol,
                            'category': cat,
                            'CRVAL1': hdr.get('CRVAL1'),
                            'CRVAL2': hdr.get('CRVAL2'),
                            'DATE-OBS': hdr.get('DATE-OBS'),
                            'DURATION': hdr.get('DURATION'),
                            'NAXIS1': hdr.get('NAXIS1'),
                            'NAXIS2': hdr.get('NAXIS2'),
                        }
                        # Extract timestamp from filename
                        import re
                        m = re.search(r'(\d{8}_\d{6})', bn)
                        meta['file_timestamp'] = m.group(1) if m else None

                        # Find frequency on any axis
                        for ax in [3, 4]:
                            if 'FREQ' in str(hdr.get(f'CTYPE{ax}', '')).upper():
                                meta['freq_hz'] = hdr.get(f'CRVAL{ax}')
                                break
                        else:
                            meta['freq_hz'] = hdr.get('RESTFRQ',
                                               hdr.get('OBSFREQ'))

                        # Check HISTORY
                        hist = []
                        if 'HISTORY' in hdr:
                            hist = [str(c) for c in hdr['HISTORY']]
                        meta['n_history'] = len(hist)
                        meta['has_wsclean'] = any(
                            'wsclean' in h.lower() for h in hist)

                        file_metadata.append(meta)
                    except Exception as e:
                        log.warning(f"  Metadata extraction failed for "
                                    f"{os.path.basename(fpath)}: {e}")

    # --- Individual header summary ---
    n_files = len(all_reports)
    n_errors = sum(1 for r in all_reports if not r.ok)
    n_warnings = sum(1 for r in all_reports if r.ok and not r.clean)
    n_clean = sum(1 for r in all_reports if r.clean)

    log.info(f"\n{'='*70}")
    log.info(f"INDIVIDUAL HEADER RESULTS: {n_files} files")
    log.info(f"  Clean: {n_clean}  Warnings: {n_warnings}  Errors: {n_errors}")
    log.info(f"{'='*70}")

    if n_errors > 0:
        log.error("\nFiles with ERRORS:")
        for r in all_reports:
            if not r.ok:
                log.error(f"  {r.full_report()}")

    # Tally missing keywords across all files
    missing_tally = {}
    for r in all_reports:
        for w in r.warnings + r.info:
            if 'Missing' in w:
                key = w.split('(')[0].strip().replace('Missing ', '').replace(
                    'Phase 2 keyword ', '')
                missing_tally[key] = missing_tally.get(key, 0) + 1

    if missing_tally:
        log.info("\nMissing keyword frequency:")
        for key, count in sorted(missing_tally.items(),
                                  key=lambda x: -x[1]):
            log.info(f"  {key:20s}: {count}/{n_files} files")

    # --- Cross-file consistency checks ---
    cross_checks = []
    log.info(f"\n{'='*70}")
    log.info(f"CROSS-FILE CONSISTENCY CHECKS")
    log.info(f"{'='*70}")

    # Check 1: CRVAL1 vs DATE-OBS consistency
    # Group by subband+pol, sort by DATE-OBS, check if CRVAL1 spacing
    # matches time spacing
    _check_crval1_vs_dateobs(file_metadata, cross_checks, log)

    # Check 2: Filename timestamp vs DATE-OBS
    _check_filename_vs_dateobs(file_metadata, cross_checks, log)

    # Check 3: Image sizes within subbands (should be uniform)
    _check_image_sizes(file_metadata, cross_checks, log)

    # Check 4: Frequency values match subband labels
    _check_freq_vs_subband(file_metadata, cross_checks, log)

    # Check 5: WSClean HISTORY / provenance
    _check_provenance(file_metadata, cross_checks, log)

    # Check 6: 10-min interval duration consistency
    _check_10min_durations(file_metadata, cross_checks, log)

    # Check 7: Filename conventions
    _check_filename_conventions(file_metadata, cross_checks, log)

    return {
        'reports': all_reports,
        'cross_checks': cross_checks,
        'file_metadata': file_metadata,
        'summary': {
            'n_files': n_files,
            'n_errors': n_errors,
            'n_warnings': n_warnings,
            'n_clean': n_clean,
        }
    }


def _check_crval1_vs_dateobs(metadata, checks, log):
    """Check that CRVAL1 spacing is consistent with DATE-OBS spacing."""
    from astropy.time import Time

    log.info("\n--- CRVAL1 vs DATE-OBS consistency ---")
    issues = []

    # Group deep images by subband+pol
    groups = {}
    for m in metadata:
        if m['category'] != 'deep':
            continue
        key = (m['subband'], m['pol'])
        groups.setdefault(key, []).append(m)

    for (subband, pol), files in sorted(groups.items()):
        if len(files) < 2:
            continue

        # Sort by DATE-OBS
        valid = [f for f in files if f['DATE-OBS'] and f['CRVAL1'] is not None]
        if len(valid) < 2:
            continue

        try:
            valid.sort(key=lambda f: f['DATE-OBS'])
        except TypeError:
            continue

        for i in range(len(valid) - 1):
            f1, f2 = valid[i], valid[i + 1]
            try:
                t1 = Time(f1['DATE-OBS'])
                t2 = Time(f2['DATE-OBS'])
                dt_hours = (t2.mjd - t1.mjd) * 24.0

                d_crval1 = f2['CRVAL1'] - f1['CRVAL1']
                d_crval1_hours = d_crval1 / 15.0  # deg to hours RA

                # Should be approximately equal
                if abs(dt_hours) > 0.01:  # avoid div by zero
                    ratio = d_crval1_hours / dt_hours
                    if abs(ratio - 1.0) > 0.1:  # >10% discrepancy
                        msg = (f"{subband} {pol}: CRVAL1 changes by "
                               f"{d_crval1:.2f}° ({d_crval1_hours:.2f}h RA) "
                               f"but DATE-OBS changes by {dt_hours:.2f}h "
                               f"between {f1['filename']} and {f2['filename']}")
                        issues.append(msg)
                        log.warning(f"  INCONSISTENT: {msg}")
            except Exception:
                pass

    if not issues:
        log.info("  All CRVAL1/DATE-OBS pairs are consistent")
    else:
        log.warning(f"  {len(issues)} inconsistencies found")

    checks.extend([('crval1_dateobs', msg) for msg in issues])


def _check_filename_vs_dateobs(metadata, checks, log):
    """Check that filename timestamp matches DATE-OBS."""
    from astropy.time import Time

    log.info("\n--- Filename timestamp vs DATE-OBS ---")
    issues = []

    for m in metadata:
        if not m['file_timestamp'] or not m['DATE-OBS']:
            continue

        # Parse filename timestamp: YYYYMMDD_HHMMSS
        ft = m['file_timestamp']
        try:
            file_time = Time(f"{ft[:4]}-{ft[4:6]}-{ft[6:8]}T"
                             f"{ft[9:11]}:{ft[11:13]}:{ft[13:15]}")
            header_time = Time(m['DATE-OBS'])

            diff_sec = abs(file_time.mjd - header_time.mjd) * 86400.0

            # Allow up to DURATION/2 difference (center vs start)
            # or 1 hour for LST bin offsets
            if diff_sec > 7200:  # > 2 hours is definitely wrong
                msg = (f"{m['filename']}: file timestamp {ft} differs from "
                       f"DATE-OBS {m['DATE-OBS']} by {diff_sec/3600:.1f} hours")
                issues.append(msg)
                log.warning(f"  MISMATCH: {msg}")
            elif diff_sec > 3600:
                msg = (f"{m['filename']}: file timestamp {ft} differs from "
                       f"DATE-OBS {m['DATE-OBS']} by {diff_sec/3600:.1f} hours")
                issues.append(msg)
                log.info(f"  NOTE: {msg}")
        except Exception:
            pass

    if not issues:
        log.info("  All filename timestamps consistent with DATE-OBS")
    else:
        log.warning(f"  {len(issues)} discrepancies found")

    checks.extend([('filename_dateobs', msg) for msg in issues])


def _check_image_sizes(metadata, checks, log):
    """Check image size consistency within subbands."""
    log.info("\n--- Image size consistency ---")
    issues = []

    # Group by subband
    groups = {}
    for m in metadata:
        groups.setdefault(m['subband'], []).append(m)

    for subband, files in sorted(groups.items()):
        sizes = set()
        for f in files:
            if f['NAXIS1'] and f['NAXIS2']:
                sizes.add((f['NAXIS1'], f['NAXIS2']))
        if len(sizes) > 1:
            msg = f"{subband}: mixed image sizes {sizes}"
            issues.append(msg)
            log.warning(f"  {msg}")

    if not issues:
        log.info("  All subbands have consistent image sizes")

    checks.extend([('image_size', msg) for msg in issues])


def _check_freq_vs_subband(metadata, checks, log):
    """Check that header frequency matches subband label."""
    log.info("\n--- Frequency vs subband label ---")
    issues = []

    for m in metadata:
        if m['freq_hz'] is None:
            continue
        freq_mhz = m['freq_hz'] / 1e6
        # Extract expected freq from subband name
        import re
        match = re.match(r'(\d+)MHz', m['subband'])
        if not match:
            continue
        expected_mhz = float(match.group(1))

        # Allow ±5 MHz tolerance (subbands are wide)
        if abs(freq_mhz - expected_mhz) > 5:
            msg = (f"{m['filename']}: header freq {freq_mhz:.1f} MHz "
                   f"doesn't match subband label {m['subband']}")
            issues.append(msg)
            log.warning(f"  {msg}")

    if not issues:
        log.info("  All frequencies match subband labels")

    checks.extend([('freq_subband', msg) for msg in issues])


def _check_provenance(metadata, checks, log):
    """Check for WSClean HISTORY and provenance info."""
    log.info("\n--- Provenance (HISTORY / WSClean) ---")

    n_total = len(metadata)
    n_no_history = sum(1 for m in metadata if m['n_history'] == 0)
    n_no_wsclean = sum(1 for m in metadata
                        if m['n_history'] > 0 and not m['has_wsclean'])
    n_has_wsclean = sum(1 for m in metadata if m['has_wsclean'])

    log.info(f"  {n_has_wsclean}/{n_total} have WSClean HISTORY")
    if n_no_history > 0:
        log.warning(f"  {n_no_history}/{n_total} have NO HISTORY at all")
        checks.append(('provenance',
                        f"{n_no_history} files have no HISTORY entries"))
    if n_no_wsclean > 0:
        log.info(f"  {n_no_wsclean}/{n_total} have HISTORY but no WSClean info")


def _check_10min_durations(metadata, checks, log):
    """Check cross-subband time consistency for 10-min intervals.

    When stacking across subbands to make wideband products, the
    contributing intervals must cover the same time range. Different
    flagging per subband (from WSClean -intervals-out) can cause
    the effective time window to differ across frequency.
    """
    from astropy.time import Time

    log.info("\n--- 10-min cross-subband time consistency ---")

    tenmin = [m for m in metadata
              if m['category'] == '10min' and m['pol'] == 'I']
    if not tenmin:
        log.info("  No 10-min I images found")
        return

    no_duration = [m for m in tenmin if m['DURATION'] is None]
    if no_duration:
        log.warning(f"  {len(no_duration)}/{len(tenmin)} 10-min images "
                    f"have no DURATION keyword — cannot verify time consistency")
        checks.append(('10min_time',
                        f"{len(no_duration)} 10-min images missing DURATION"))

    # Group by timestamp (interval identifier)
    # Extract first timestamp from filename
    import re
    intervals = {}
    for m in tenmin:
        ts_matches = re.findall(r'\d{8}_\d{6}', m['filename'])
        if not ts_matches:
            continue
        # Use the first timestamp as the interval key
        interval_key = ts_matches[0]
        intervals.setdefault(interval_key, []).append(m)

    if not intervals:
        return

    log.info(f"  {len(intervals)} intervals across "
             f"{len(set(m['subband'] for m in tenmin))} subbands")

    n_inconsistent = 0
    max_spread = 0

    for ts, files in sorted(intervals.items()):
        if len(files) < 2:
            continue

        # Check DATE-OBS and DURATION consistency across subbands
        valid = [f for f in files
                 if f['DATE-OBS'] and f['DURATION'] is not None]
        if len(valid) < 2:
            continue

        try:
            starts = []
            ends = []
            for f in valid:
                t_start = Time(f['DATE-OBS']).mjd
                t_end = t_start + f['DURATION'] / 86400.0
                starts.append(t_start)
                ends.append(t_end)

            start_spread_s = (max(starts) - min(starts)) * 86400.0
            end_spread_s = (max(ends) - min(ends)) * 86400.0
            dur_spread_s = max(f['DURATION'] for f in valid) - \
                           min(f['DURATION'] for f in valid)

            spread = max(start_spread_s, end_spread_s)
            max_spread = max(max_spread, spread)

            # Flag if start or end times differ by >30s across subbands
            if spread > 30:
                n_inconsistent += 1
                subbands = [f['subband'] for f in valid]
                durations = [f'{f["DURATION"]:.0f}s' for f in valid]
                if n_inconsistent <= 5:  # don't flood the log
                    log.warning(
                        f"  Interval {ts}: time spread {spread:.0f}s "
                        f"across {len(valid)} subbands "
                        f"(durations: {', '.join(durations[:5])})")
        except Exception:
            pass

    if n_inconsistent > 0:
        log.warning(f"  {n_inconsistent}/{len(intervals)} intervals have "
                    f">30s time spread across subbands "
                    f"(max spread: {max_spread:.0f}s)")
        log.info(f"  Impact: wideband stacks may mix data from different "
                 f"time windows across frequency")
        checks.append(('10min_time',
                        f"{n_inconsistent} intervals have >30s cross-subband "
                        f"time spread (max {max_spread:.0f}s)"))
    else:
        log.info(f"  All intervals consistent across subbands "
                 f"(max spread: {max_spread:.0f}s)")

    # Report filename convention
    n_dual_ts = sum(1 for m in tenmin
                    if len(re.findall(r'\d{8}_\d{6}', m['filename'])) >= 2)
    if n_dual_ts > 0:
        log.info(f"  Filename convention: {n_dual_ts}/{len(tenmin)} files "
                 f"have dual timestamps (midpoint_start — known issue)")


def _check_filename_conventions(metadata, checks, log):
    """Analyze filename conventions and report inconsistencies.

    Parses each filename into ordered fields and groups by pattern.
    Reports the distinct patterns found and flags inconsistencies.
    """
    import re

    log.info("\n--- Filename convention analysis ---")

    # Define recognizable field types
    def classify_field(field):
        """Classify a filename field into a semantic category."""
        f = field.lower()
        if re.match(r'^\d+mhz$', f):
            return 'FREQ'
        if f in ('i', 'v', 'q', 'u'):
            return 'STOKES'
        if f in ('deep', '10min', '1hr'):
            return 'DURATION'
        if f in ('taper', 'notaper'):
            return 'TAPER'
        if re.match(r'^robust-?[\d.]+$', f):
            return 'ROBUST'
        if f == 'image':
            return 'IMAGE'
        if f in ('dirty', 'residual', 'model', 'psf'):
            return 'PRODUCT'
        if re.match(r'^\d{8}$', f):
            return 'DATE'
        if re.match(r'^\d{6}$', f):
            return 'TIME'
        if f == 'horizon-mask':
            return 'MASK'
        return f'OTHER({field})'

    # Parse each filename
    patterns = {}  # pattern_key -> list of filenames
    field_orders = {}  # (stokes, category) -> set of patterns

    for m in metadata:
        bn = m['filename']
        # Strip extensions
        name = bn
        for ext in ['.pbcorr.fits', '.fits']:
            if name.endswith(ext):
                name = name[:-len(ext)]
                break

        # Check for pbcorr suffix
        has_pbcorr = '.pbcorr' in bn

        # Split on hyphens, but keep compound fields like "Robust-0.75"
        # and timestamps like "20241218_123232"
        # First, rejoin timestamps
        name = re.sub(r'(\d{8})_(\d{6})', r'\1-\2', name)
        fields = name.split('-')

        # Classify each field
        classified = []
        i = 0
        while i < len(fields):
            # Handle "Robust-0.75" as one field
            if fields[i].lower() == 'robust' and i + 1 < len(fields):
                classified.append('ROBUST')
                i += 2
            # Handle "horizon-mask" as one field
            elif fields[i].lower() == 'horizon' and i + 1 < len(fields) and fields[i+1].lower() == 'mask':
                classified.append('MASK')
                i += 2
            else:
                classified.append(classify_field(fields[i]))
                i += 1

        if has_pbcorr:
            classified.append('PBCORR')

        pattern_key = '-'.join(classified)

        patterns.setdefault(pattern_key, []).append(bn)

        # Track by stokes+category for comparison
        group_key = (m['pol'], m['category'])
        field_orders.setdefault(group_key, {}).setdefault(
            pattern_key, []).append(bn)

    # Report patterns
    log.info(f"  {len(patterns)} distinct filename patterns across "
             f"{sum(len(v) for v in patterns.values())} files:")
    for pattern, files in sorted(patterns.items(),
                                  key=lambda x: -len(x[1])):
        log.info(f"    [{len(files):3d} files] {pattern}")
        log.info(f"              e.g. {files[0]}")

    # Check consistency within groups
    log.info(f"\n  Patterns by Stokes/category:")
    inconsistencies = []
    for (pol, cat), pats in sorted(field_orders.items()):
        if len(pats) == 1:
            pat = list(pats.keys())[0]
            n = len(list(pats.values())[0])
            log.info(f"    {pol}/{cat}: {n} files, 1 pattern — {pat}")
        else:
            log.warning(f"    {pol}/{cat}: MIXED patterns!")
            for pat, files in sorted(pats.items(), key=lambda x: -len(x[1])):
                log.warning(f"      [{len(files):3d}] {pat}")
                log.warning(f"            e.g. {files[0]}")
            inconsistencies.append(f"{pol}/{cat} has {len(pats)} patterns")

    # Check field ordering consistency across groups
    log.info(f"\n  Cross-group field order comparison:")

    # Compare I vs V for the same category
    for cat in ['deep', '10min']:
        i_pats = set(field_orders.get(('I', cat), {}).keys())
        v_pats = set(field_orders.get(('V', cat), {}).keys())
        if i_pats and v_pats:
            if i_pats == v_pats:
                log.info(f"    {cat}: I and V use same pattern(s)")
            else:
                log.warning(f"    {cat}: I and V use DIFFERENT patterns!")
                log.warning(f"      I: {i_pats}")
                log.warning(f"      V: {v_pats}")
                inconsistencies.append(
                    f"{cat} I vs V naming mismatch")

    # Proposed standard
    log.info(f"\n  Proposed standard: "
             f"FREQ-STOKES-TAPER-DURATION-ROBUST-DATE-TIME-IMAGE[.pbcorr].fits")
    log.info(f"  e.g. 69MHz-I-Taper-592s-Robust-0-20241218-121406-image.fits")

    if inconsistencies:
        for msg in inconsistencies:
            checks.append(('filename_convention', msg))


# =====================================================================
# CLI
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Validate FITS image headers for OVRO-LWA pipeline. '
                    'Use --lst/--date for Phase 1 audit, or pass a path '
                    'for direct validation.')
    parser.add_argument('path', nargs='?', default=None,
                        help='FITS file or directory to validate')
    parser.add_argument('--level', choices=['critical', 'standard', 'full'],
                        default='standard',
                        help='Validation level (default: standard)')
    parser.add_argument('--recursive', action='store_true',
                        help='Search subdirectories')
    parser.add_argument('--quiet', action='store_true',
                        help='Only show errors and warnings')
    parser.add_argument('--lst', default=None,
                        help='LST label for Phase 1 audit (e.g., 10h)')
    parser.add_argument('--date', default=None,
                        help='Date for Phase 1 audit (e.g., 2024-12-18)')
    parser.add_argument('--repair-wideband', action='store_true',
                        help='Repair FREQ/Stokes WCS on wideband products '
                             'in a Phase 2 run directory (in-place)')
    parser.add_argument('--repair-all-runs', default=None, metavar='PHASE2_ROOT',
                        help='Repair all Phase 2 runs under a root directory '
                             '(e.g., /fast/gh/main/phase2/)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be repaired without modifying files')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s')
    log = logging.getLogger('phase2')

    # Repair all runs mode
    if args.repair_all_runs:
        root = args.repair_all_runs
        total = 0
        for lst_dir in sorted(glob.glob(os.path.join(root, '*h'))):
            for date_dir in sorted(glob.glob(os.path.join(lst_dir, '2*'))):
                for run_dir in sorted(glob.glob(
                        os.path.join(date_dir, 'Science_*'))):
                    n = repair_wideband_headers(run_dir, log=log,
                                                 dry_run=args.dry_run)
                    total += n
        tag = "[dry-run] " if args.dry_run else ""
        print(f"\n{tag}Total: {total} wideband files repaired")
        sys.exit(0)

    # Repair single run mode
    if args.repair_wideband:
        if not args.path:
            print("Error: --repair-wideband requires a path to a Phase 2 "
                  "run directory")
            sys.exit(1)
        n = repair_wideband_headers(args.path, log=log, dry_run=args.dry_run)
        tag = "[dry-run] " if args.dry_run else ""
        print(f"{tag}{n} wideband files repaired")
        sys.exit(0)

    # Audit mode
    if args.lst and args.date:
        result = audit_phase1(args.lst, args.date)
        if result:
            s = result['summary']
            print(f"\nFinal: {s['n_files']} files — "
                  f"{s['n_clean']} clean, {s['n_warnings']} warnings, "
                  f"{s['n_errors']} errors")
            print(f"Cross-checks: {len(result['cross_checks'])} issues")
            sys.exit(0 if s['n_errors'] == 0 else 1)
        sys.exit(1)

    # Validate mode
    if args.path:
        if os.path.isfile(args.path):
            report = validate_header(args.path, level=args.level)
            print(report.full_report())
            sys.exit(0 if report.ok else 1)
        elif os.path.isdir(args.path):
            reports = validate_directory(args.path, level=args.level,
                                         recursive=args.recursive)
            for r in reports:
                if not args.quiet or not r.clean:
                    print(r.full_report())
            n_fail = sum(1 for r in reports if not r.ok)
            sys.exit(0 if n_fail == 0 else 1)
        else:
            print(f"Error: {args.path} not found")
            sys.exit(1)

    parser.print_help()
    sys.exit(1)
