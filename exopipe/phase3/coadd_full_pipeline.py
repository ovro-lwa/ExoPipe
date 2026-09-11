#!/usr/bin/env python3
"""Full co-addition pipeline: align, flux-scale, flow-align, outlier-detect, co-add, movies.

Computes flow + flux correction on I Taper Robust-0.75 dewarped non-pbcorr deep,
then applies the same correction to ALL Stokes I variants, Stokes V, deep and 10-min.

Usage:
  # Single frequency, all LSTs:
  conda run -n py38_orca_nkosogor python coadd_full_pipeline.py --freq 73

  # Single LST + frequency:
  conda run -n py38_orca_nkosogor python coadd_full_pipeline.py --freq 73 --lst 10h

  # All frequencies and LSTs:
  conda run -n py38_orca_nkosogor python coadd_full_pipeline.py --all

  # Wideband:
  conda run -n py38_orca_nkosogor python coadd_full_pipeline.py --freq Blue
"""
import argparse
import csv
import fcntl
import os
import shutil
import sys
import re
import glob
import json
import time
import subprocess
import warnings
from multiprocessing import Pool
import numpy as np
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*FITSFixedWarning.*')
warnings.filterwarnings('ignore', message='.*datfix.*')
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
warnings.filterwarnings('ignore', category=AstropyWarning)
warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)
from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u
from reproject import reproject_interp
from scipy.ndimage import maximum_filter as maxfilt
from scipy.ndimage import map_coordinates
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

PHASE2_ROOT = '/lustre/pipeline/exopipe/phase2'
WORKDIR_BASE = os.environ.get('P3_COADD_ROOT')   # NO default (Gregg 2026-09-04): --workdir or P3_COADD_ROOT required


def _workdir_base():
    if not WORKDIR_BASE:
        from exopipe.common.roots import required_root
        return required_root('P3_COADD_ROOT', 'co-add work/output root; or pass --workdir')
    return WORKDIR_BASE
DECISIONS_CSV = '/lustre/gh/main/clv15/data_decisions.csv'  # canonical (Gregg move 2026-08-10; -> exopipe_ops/ at refactor)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# GC-PRE-2 (Gregg 2026-08-12): flow batch script lives in exopipe/common/
# (cross-phase core infrastructure); shelled into py310_orca, never imported.
BATCH_FLOW_SCRIPT = os.path.join(os.path.dirname(SCRIPT_DIR), 'common',
                                 'batch_flow_align.py')

OVRO_LOC = EarthLocation(lat=37.2397*u.deg, lon=-118.2816*u.deg, height=1222*u.m)
CUTOVER_DATE = '2024-12-27'

SUBBANDS = [18, 23, 27, 32, 36, 41, 46, 50, 55, 59, 64, 69, 73, 78, 82]
WIDEBAND_BANDS = ['Red', 'Green', 'Blue', 'Full']
EXCLUDED_LSTS = set()           # 11h/12h no longer hard-excluded (Gregg 2026-06-26)
MIN_DATES_FOR_COADD = 3

REFERENCE_DATES = {
    # 00h: mosaic-coverage cell (Gregg 2026-07-05) — deliberately SHALLOW,
    # built from ONLY 2025-08-09 + 2025-08-16 via run_00h_coadd.py (the
    # third archive date 2024-12-18 has no dewarped R075 and is not used).
    '00h': '2025-08-09',
    '01h': '2024-12-21', '02h': '2024-12-21', '03h': '2024-12-21',
    # 05h: 2024-12-21 -> 2024-12-29 (Gregg 2026-08-28): 12-21 has NO 50MHz
    # deep products (empty archive dir, found by the A-02 reference gate);
    # 12-29 is P1/P2-complete 15/15 incl. 50MHz deep, not excluded. Absolute
    # flux (Scaife & Heald) and astrometry (VLSSr) are corrected on the
    # co-adds downstream, so the reference only anchors internal alignment.
    '04h': '2024-12-21', '05h': '2024-12-29', '06h': '2024-12-21',
    '07h': '2024-12-21', '08h': '2024-12-21', '09h': '2024-12-21',
    '10h': '2024-12-21',
    # 12h: peel-reprocessed; 2024-12-29 (peel-fixed + P2-complete;
    # 2024-12-21 not usable there).
    # 11h: 2024-12-29 -> 2024-12-28 (Gregg 2026-07-08): 12-29's Green dewarp
    # FAILED and its raw Green is ~20' warped — the raw-fallback failsafe made
    # it the Green alignment reference and shifted the whole 11h Green cell
    # -14.5' in RA (COADD_ASTROMETRIC_ERROR.md §2b/§4). 12-28 has all 4 bands
    # dewarped + clean QA. 12h keeps 12-29 (its dewarps are fine).
    '11h': '2024-12-28', '12h': '2024-12-29',
    # 13h: was 2025-01-08, whose Blue/Full I-R075 Phase-2 stacks are .CORRUPT
    # (divergent peel, pending re-peel) -> reference change ruled by Gregg
    # 2026-07-05. 2025-05-03 = the 14h-20h block reference; for 13h it is
    # non-outlier both scopes (Blue weight 91.0, Full 25.3), dewarped R075
    # alignment images verified sane.
    '13h': '2025-05-03',
    '14h': '2025-05-03', '15h': '2025-05-03', '16h': '2025-05-03',
    '17h': '2025-05-03', '18h': '2025-05-03', '19h': '2025-05-03',
    '20h': '2025-05-03',
    # 21h/22h added 2026-07-05 (Gregg go-ahead; August-2025 block).
    # 2025-08-05 ruled by Gregg.
    '21h': '2025-08-05', '22h': '2025-08-05',
    # 23h REINSTATED 2026-07-07 (Gregg): Phase-1 re-peel (Run_20260706_*) +
    # Phase-2 reprocess of 2025-08-18/22/23/26/27; old-peel 08-09/16/17 stay
    # excluded in data_decisions. Was 2025-08-18 (re-peeled), but its
    # exclusion rows (2026-07-05) were never lifted, so the A-02 reference
    # gate refused every 23h build -> Gregg designated 2025-09-06 as the new
    # reference (2026-08-29; not excluded, P1/P2-complete). The 08-18
    # peeling-vs-overclean question is under investigation separately.
    '23h': '2025-09-06',
}

BAND_FOR_SUBBAND = {
    18: 'Red', 23: 'Red', 27: 'Red', 32: 'Red', 36: 'Red',
    41: 'Green', 46: 'Green', 50: 'Green', 55: 'Green', 59: 'Green',
    64: 'Blue', 69: 'Blue', 73: 'Blue', 78: 'Blue', 82: 'Blue',
}

OUTLIER_SIGMA = 3.0
LOW_OUTLIER_SIGMA = float(os.environ.get('P3_LOW_OUTLIER_SIGMA', '4.0'))   # R-GC5 low side: 4 sigma (Gregg 2026-09-05)



def _update_workdir(val):
    global WORKDIR_BASE
    WORKDIR_BASE = val


def _update_sigma(val):
    global OUTLIER_SIGMA
    OUTLIER_SIGMA = val


# ═══════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════

def load_image(path):
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header.copy()
    while data.ndim > 2:
        data = data[0]
    return np.nan_to_num(data, nan=0.0), header


def save_image(path, data, header):
    fits.writeto(path, data.astype(np.float32), header, overwrite=True)


def get_uniform_rms(data, mask):
    h, w = data.shape
    margin = int(min(h, w) * 0.15)
    inner = data[margin:h-margin, margin:w-margin]
    inner_mask = mask[margin:h-margin, margin:w-margin]
    valid = inner[~inner_mask]
    if len(valid) < 100:
        return float(np.nanstd(data[~mask])) if np.any(~mask) else 1.0
    med = np.median(valid)
    mad = np.median(np.abs(valid - med))
    sigma = mad * 1.4826
    clip = valid[np.abs(valid - med) < 4 * sigma]
    return float(np.std(clip)) if len(clip) > 50 else sigma


def get_midpoint_time(header, fits_path=None):
    for key in ['DATE-OBS', 'DATE-BEG', 'MJD-OBS']:
        val = header.get(key)
        if val:
            try:
                if key == 'MJD-OBS':
                    return Time(float(val), format='mjd')
                return Time(val, format='isot', scale='utc')
            except Exception:
                continue
    if fits_path:
        m = re.search(r'(\d{8})_(\d{6})', os.path.basename(fits_path))
        if m:
            return Time(f'{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}T'
                       f'{m.group(2)[:2]}:{m.group(2)[2:4]}:{m.group(2)[4:6]}',
                       format='isot', scale='utc')
    return Time('2025-01-01T00:00:00', scale='utc')


# Time-block keywords copied from the per-epoch source onto each per-date residual.
_EPOCH_TIME_KEYS = ['DATE-OBS', 'DATE-BEG', 'DATE-END', 'MJD-OBS', 'MJD-BEG',
                    'MJD-END', 'MJD-MID', 'INTTIME', 'DURATION', 'TELAPSE', 'TIMEFIX']


def epoch_time_header(tmpl_hdr, src_path):
    """Copy of the co-add TEMPLATE header (WCS / grid) with the TIME block overwritten
    from the per-epoch source image (`src_path`), so each per-date subtracted/_clean_
    residual carries its OWN observation time — not the template's. Fixes the systematic
    bug where every residual inherited tmpl_hdr's DATE-OBS (one epoch reused for all)."""
    out = tmpl_hdr.copy()
    try:
        src = fits.getheader(src_path)
    except Exception:
        src = None
    for k in _EPOCH_TIME_KEYS:
        if src is not None and k in src:
            out[k] = src[k]
            try:
                out.comments[k] = src.comments[k]
            except Exception:
                pass
        elif k in out:
            del out[k]                      # drop stale template time value
    # span midpoint if the source didn't carry one (DATE-OBS=start, DURATION=span [s])
    if 'MJD-MID' not in out and 'MJD-OBS' in out and 'DURATION' in out:
        try:
            out['MJD-MID'] = (float(out['MJD-OBS']) + float(out['DURATION']) / 2.0 / 86400.0,
                              'span midpoint (MJD)')
        except Exception:
            pass
    out.add_history(f'Time block from per-epoch source {os.path.basename(str(src_path))}')
    return out


# ═══════════════════════════════════════════════════════════════════
# R1 AMENDMENT #1 — per-epoch beam+RESTFRQ stamping + validate-on-write
# refusal for stage-B per-epoch residuals.
# Ruling: fix_ledger R1-ratified scope (a) + R1-amendment1-GO (Gregg
# 2026-08-13); R-A123 (A3's fix = this write-time refusal); CLAUDE.md rule 7
# (epoch products carry their OWN headers — a residual matching its master's
# beam is a write failure) + rule 3 (gates refuse, they do not warn).
# Defect being closed: build_template_header propagates the co-add TEMPLATE
# epoch's BMAJ/BMIN/BPA/RESTFRQ onto every stage-B per-epoch residual;
# epoch_time_header fixes the time block only (demonstrated in
# FAIL_11h_Green_investigation.md §2–§3, FAIL_13h_Green_investigation.md §F).
# Correct values: the epoch's OWN Phase-2 source image header, per Stokes —
# I residuals from the epoch's I deep Taper_R075 alignment image (the same
# per-image source the time block comes from); V residuals from the epoch's
# V deep Taper_R0 image via find_stokes_v_deep_image (restamp finding F2 /
# R-RAT1: V products carry V-epoch beams, never I-derived ones).
# ═══════════════════════════════════════════════════════════════════

_EPOCH_BEAM_KEYS = ('BMAJ', 'BMIN', 'BPA', 'RESTFRQ')
_R1_RULING = ('fix_ledger R1-ratified / R1-amendment1-GO, Gregg 2026-08-13; '
              'CLAUDE.md rule 7')


class EpochHeaderRefusal(RuntimeError):
    """Validate-on-write refusal for a stage-B per-epoch residual header.

    Raised BEFORE any bytes go to disk; no product is written and no
    pre-existing file is deleted by a refusal (gates refuse, they do not
    warn — CLAUDE.md rule 3)."""


def _r1_same(a, b, rtol=1e-9):
    """Float-key equality for the R1 gate (tolerates only header write/read
    round-trip jitter, e.g. RESTFRQ through WCS.to_header; real template-vs-
    epoch beam differences are orders of magnitude larger)."""
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return False
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-300)


def _epoch_beam_source_values(src_path):
    """Read BMAJ/BMIN/BPA/RESTFRQ from the epoch's OWN Phase-2 source header.

    Returns (values, problems). Any missing/corrupt source or key is a
    PROBLEM for the caller to refuse on — never defaulted, never fallen back
    to template values."""
    if not src_path:
        return {}, ['no Phase-2 source path resolved']
    if not os.path.exists(src_path):
        return {}, [f'source does not exist: {src_path}']
    try:
        src = fits.getheader(src_path)
    except Exception as e:
        return {}, [f'unreadable source header ({e.__class__.__name__}: {e})']
    vals, problems = {}, []
    for k in _EPOCH_BEAM_KEYS:
        if k not in src:
            problems.append(f'{k} missing from source header')
            continue
        try:
            v = float(src[k])
        except (TypeError, ValueError):
            problems.append(f'{k} non-numeric in source header ({src[k]!r})')
            continue
        if not np.isfinite(v):
            problems.append(f'{k} non-finite in source header ({v!r})')
            continue
        vals[k] = v
    return vals, problems


def _r1_fmt_keys(getter):
    """One-line BMAJ/BMIN/BPA/RESTFRQ rendering (full float repr)."""
    parts = []
    for k in _EPOCH_BEAM_KEYS:
        v = getter.get(k)
        try:
            parts.append(f'{k}={float(v)!r}')
        except (TypeError, ValueError):
            parts.append(f'{k}={v!r}')
    return ' '.join(parts)


def epoch_beam_header(hdr, src_path, stokes):
    """Copy of `hdr` with BMAJ/BMIN/BPA/RESTFRQ stamped from the epoch's OWN
    Phase-2 source image header (`src_path`), with provenance (BEAMSRC +
    HISTORY). REFUSES (EpochHeaderRefusal) when the source or any of the four
    keys cannot be resolved — fallback to template/master values is forbidden
    (R1 amendment #1)."""
    vals, problems = _epoch_beam_source_values(src_path)
    if problems:
        raise EpochHeaderRefusal(
            f"REFUSED [R1-amendment1] stage-B epoch residual header ({stokes}):\n"
            f"  cannot resolve the epoch's OWN BMAJ/BMIN/BPA/RESTFRQ from its "
            f"Phase-2 source\n"
            f"  source: {src_path or 'NONE (no source resolved)'}\n"
            f"  problems: {'; '.join(problems)}\n"
            f"  Fallback to template/master values is forbidden ({_R1_RULING}; "
            f"gates refuse, they do not warn). No product written.")
    out = hdr.copy()
    for k in _EPOCH_BEAM_KEYS:
        out[k] = vals[k]
    out['BEAMSRC'] = ('epoch',
                      'beam+RESTFRQ from epoch OWN P2 source (R1 amendment #1)')
    out.add_history(f'Beam+RESTFRQ ({stokes}) from per-epoch source '
                    f'{os.path.basename(str(src_path))} [R1-amendment1]')
    return out


def validate_epoch_residual_header(out_path, out_hdr, src_path, tmpl_hdr):
    """Write gate for stage-B per-epoch residual headers. REFUSES (raises
    EpochHeaderRefusal, nothing written) when:
      (a) the outgoing beam matches the co-add TEMPLATE/master beam while the
          epoch's own Phase-2 header says otherwise (the R1 signature), or the
          outgoing keys otherwise deviate from the epoch's own header;
      (b) the four keys cannot be resolved from the epoch's own source
          (missing/corrupt source = refusal, not fallback).
    Passing when own == template (single-night template == epoch) is correct
    by construction."""
    vals, problems = _epoch_beam_source_values(src_path)
    if problems:
        raise EpochHeaderRefusal(
            f"REFUSED [R1-amendment1] stage-B epoch residual write: {out_path}\n"
            f"  cannot resolve the epoch's OWN BMAJ/BMIN/BPA/RESTFRQ from its "
            f"Phase-2 source\n"
            f"  source: {src_path or 'NONE (no source resolved)'}\n"
            f"  problems: {'; '.join(problems)}\n"
            f"  Fallback to template/master values is forbidden ({_R1_RULING}; "
            f"gates refuse, they do not warn). No product written.")
    mismatched = [k for k in _EPOCH_BEAM_KEYS
                  if not _r1_same(out_hdr.get(k), vals[k])]
    if not mismatched:
        return
    src_base = os.path.basename(str(src_path))
    beam3 = ('BMAJ', 'BMIN', 'BPA')
    out_is_template = all(_r1_same(out_hdr.get(k), tmpl_hdr.get(k))
                          for k in beam3)
    own_differs_template = any(not _r1_same(vals[k], tmpl_hdr.get(k))
                               for k in beam3)
    if out_is_template and own_differs_template:
        raise EpochHeaderRefusal(
            f"REFUSED [R1-amendment1] stage-B epoch residual write: {out_path}\n"
            f"  outgoing header carries the co-add TEMPLATE/master beam while "
            f"the epoch's own Phase-2 header says otherwise — the R1 "
            f"write-failure signature ({_R1_RULING}).\n"
            f"  template : {_r1_fmt_keys(tmpl_hdr)}\n"
            f"  epoch own ({src_base}): {_r1_fmt_keys(vals)}\n"
            f"  outgoing : {_r1_fmt_keys(out_hdr)}\n"
            f"  No product written.")
    raise EpochHeaderRefusal(
        f"REFUSED [R1-amendment1] stage-B epoch residual write: {out_path}\n"
        f"  outgoing {'/'.join(mismatched)} do not match the epoch's OWN "
        f"Phase-2 header ({_R1_RULING}: epoch products carry their OWN "
        f"headers).\n"
        f"  epoch own ({src_base}): {_r1_fmt_keys(vals)}\n"
        f"  outgoing : {_r1_fmt_keys(out_hdr)}\n"
        f"  No product written.")


def save_epoch_residual(path, data, epoch_hdr, src_path, tmpl_hdr, stokes):
    """THE gated write path for stage-B per-epoch residuals (R1 amendment #1):
    stamp the epoch's OWN beam+RESTFRQ (per Stokes), validate-on-write, then
    write. On refusal nothing is written and any pre-existing file at `path`
    is untouched; if the physical write itself dies, the partial product is
    removed so it cannot masquerade as complete."""
    out_hdr = epoch_beam_header(epoch_hdr, src_path, stokes)
    validate_epoch_residual_header(path, out_hdr, src_path, tmpl_hdr)
    try:
        save_image(path, data, out_hdr)
    except BaseException:
        try:
            if os.path.exists(path):
                os.remove(path)          # partial product must not survive
        except OSError:
            pass
        raise
    return path


def _phase2_time_source(lst, date, scope):
    """Deep Phase-2 image whose time block = this epoch (shared across scopes)."""
    base = f'/lustre/pipeline/exopipe/phase2/{lst}/{date}'
    pats = [f'{base}/*/{scope}/I/deep/*Taper_Robust-0.75*dewarped*.fits',
            f'{base}/**/*-I-Taper-3*s-Robust-0.75-*-image_dewarped.fits']
    best, bestdur = None, -1.0
    for p in pats:
        for h in glob.glob(p, recursive=True):
            if 'pbcorr' in h:
                continue
            try:
                dur = float(fits.getheader(h).get('DURATION', 0) or 0)
            except Exception:
                dur = 0.0
            if dur > bestdur:
                best, bestdur = h, dur
        if best is not None:
            break
    return best


def stamp_master_span(path, lst, scope, dates):
    """Stamp a deep MASTER co-add with the constituent SPAN time block (header
    standard): DATE-OBS=earliest start, DATE-END=latest end, DURATION=wall span,
    INTTIME=summed integration, MJD-MID=span midpoint, NSTACK=#epochs. Without this
    the master inherits one epoch's date from tmpl_hdr."""
    starts, ends, intt = [], [], 0.0
    for d in sorted(set(dates)):
        src = _phase2_time_source(lst, d, scope)
        if not src:
            continue
        try:
            sh = fits.getheader(src)
        except Exception:
            continue
        if sh.get('DATE-OBS'):
            starts.append(Time(sh['DATE-OBS'], scale='utc'))
        if sh.get('DATE-END'):
            ends.append(Time(sh['DATE-END'], scale='utc'))
        try:
            intt += float(sh.get('INTTIME', 0) or 0)
        except Exception:
            pass
    if not starts or not ends:
        return
    t0, t1 = min(starts), max(ends)
    try:
        with fits.open(path, mode='update') as hd:
            H = hd[0].header
            H['DATE-OBS'] = (t0.isot, 'start of earliest constituent')
            H['MJD-OBS'] = (float(t0.mjd), 'MJD start')
            H['DATE-END'] = (t1.isot, 'end of latest constituent')
            H['MJD-END'] = (float(t1.mjd), 'MJD end')
            H['MJD-MID'] = (float((t0.mjd + t1.mjd) / 2.0), 'span midpoint (MJD)')
            H['DURATION'] = (float((t1.mjd - t0.mjd) * 86400.0), 'wall-clock span [s]')
            H['INTTIME'] = (float(intt), 'summed integration time [s]')
            H['NSTACK'] = (len(set(dates)), 'epochs in stack')
            H['TIMEFIX'] = ('v2-coadd-span', 'master time = constituent span')
            hd.flush()
    except Exception:
        pass


def is_subband(freq_str):
    try:
        return int(freq_str) in SUBBANDS
    except (ValueError, TypeError):
        return False


def freq_label(freq_str):
    """Human-readable label: '73MHz' or 'Blue'."""
    if is_subband(freq_str):
        return f'{freq_str}MHz'
    return str(freq_str)


# ═══════════════════════════════════════════════════════════════════
# Exclusion loading
# ═══════════════════════════════════════════════════════════════════

_PERBAND_EXCL = set()   # (lst, sub, date) rows marked [PERBAND] — see load_exclusions


def load_exclusions():
    """Load prior exclusions from data_decisions.csv. Returns set of (lst, freq_str, date)."""
    excl = set()
    if not os.path.exists(DECISIONS_CSV):
        # C-09 (Gregg 2026-08-11): missing exclusion list = HARD REFUSAL.
        # Proceeding with an empty set silently re-admits every excluded
        # cell into co-adds and searches.
        raise RuntimeError(
            f"REFUSING to run: exclusion list not found at {DECISIONS_CSV} "
            f"(C-09 ruling 2026-08-11: never proceed with an empty exclusion set)")
    df = pd.read_csv(DECISIONS_CSV)
    _PERBAND_EXCL.clear()
    for _, row in df.iterrows():
        if row.get('verdict') == 'exclude':
            lst = row['lst']
            sub = str(row['subband']).replace('MHz', '')
            date = row['date']
            excl.add((lst, sub, date))
            # [PERBAND] rows (QA-3 era, Gregg 2026-08-26: per-image
            # exclusion, spatially-localized vortex) do NOT count toward
            # the whole-night inference — they exclude only their own band.
            if '[PERBAND]' in str(row.get('reason', '')):
                _PERBAND_EXCL.add((lst, sub, date))
    return excl


# Subbands whose individual exclusion does NOT propagate to their wideband.
# Today this is only 18 MHz (Gregg 2026-06-29): an 18-only exclusion drops the
# 18 MHz subband image but its Red wideband is KEPT (accepted contamination).
SUBBAND_ONLY_EXCL = {'18'}

# Wideband constituent map (mirrors phase2_config.SUBBAND_BANDS; Full = all).
_WB_SUBS = {'Red':   {'18', '23', '27', '32', '36'},
            'Green': {'41', '46', '50', '55', '59'},
            'Blue':  {'64', '69', '73', '78', '82'}}
_WB_SUBS['Full'] = _WB_SUBS['Red'] | _WB_SUBS['Green'] | _WB_SUBS['Blue']


def is_whole_night_excluded(lst, date, exclusions):
    """True iff every wideband scope of (lst, date) is excluded under the
    per-band rule below — retained for callers that ask night-level questions.
    (No longer the wideband exclusion primitive; see is_excluded.)"""
    return all(is_excluded(lst, b, date, exclusions)
               for b in ('Red', 'Green', 'Blue'))


def is_excluded(lst, freq_str, date, exclusions):
    """Exclusion test shared by the co-add and the validation gate.

    WIDEBAND (Red/Green/Blue/Full): excluded iff any of ITS OWN constituent
    subbands is excluded (18 MHz carve-out and [PERBAND] rows excepted), or an
    explicit per-scope row exists. PER-BAND RULE (Gregg 2026-08-31): a night
    bad only above 50 MHz keeps its Red wideband — excluded subbands drop out,
    widebands form only from fully clean constituent sets, QA still gates.
    Supersedes the 2026-06-29 whole-night inference (any non-18MHz exclusion
    killed every wideband of the night); whole-night verdicts still exclude
    everything because their rows cover all subbands.

    SUBBAND (e.g. '18'): excluded iff that exact (lst, subband, date) is listed.
    """
    fs = str(freq_str).replace('MHz', '')
    if fs in _WB_SUBS:
        bad = {s for (l, s, d) in exclusions
               if l == lst and d == date and (l, s, d) not in _PERBAND_EXCL}
        # per-scope wideband rows honored since 2026-08-26 (Gregg ruling:
        # per-band/per-image exclusion; QA-3 flags individual scopes)
        return (bool(_WB_SUBS[fs] & (bad - SUBBAND_ONLY_EXCL))
                or (lst, fs, date) in exclusions)
    return (lst, fs, date) in exclusions


def excluded_constituents(lst, freq_str, dates, exclusions):
    """VALIDATION GATE (Gregg 2026-06-29). Return the subset of `dates` that are
    currently excluded for (lst, freq_str). A non-empty result means a co-add
    ingested data that data_decisions says to drop — callers (publish, search)
    MUST refuse. Uses is_excluded, so it can never disagree with the co-add."""
    return sorted(d for d in dates if is_excluded(lst, freq_str, d, exclusions))


def master_constituent_dates(fits_path):
    """Constituent observation dates parsed from a master's HISTORY block —
    the actual nights baked into the published image."""
    import re
    hdr = fits.getheader(fits_path)
    txt = '\n'.join(str(x) for x in hdr.get('HISTORY', []))
    return sorted(set(re.findall(r'20\d\d-\d\d-\d\d', txt)))


# ═══════════════════════════════════════════════════════════════════
# Image discovery
# ═══════════════════════════════════════════════════════════════════


def _modal_grid_template(post_dates, fallback_entry):
    """Pick the template from a date whose grid matches the MODAL grid of
    post_dates (a single wrong-grid date must not poison target_shape —
    the 19h Full 2025-04-05 lesson, cf. the §5.15d post_dates[0] beam bug).
    Returns (tmpl_hdr, target_shape)."""
    from astropy.io import fits as _fits
    from collections import Counter as _Counter
    shapes = {}
    for x in post_dates:
        try:
            h = _fits.getheader(x[2])
            shapes[x[0]] = (h['NAXIS2'], h['NAXIS1'])
        except Exception:
            continue
    if shapes:
        mode = _Counter(shapes.values()).most_common(1)[0][0]
        for x in post_dates:
            if shapes.get(x[0]) == mode:
                return build_template_header(x[2])
    return build_template_header(
        post_dates[0][2] if post_dates else fallback_entry[2])

def find_latest_science_dir(lst, date):
    """Find the latest Science_ directory for an (lst, date)."""
    pattern = f'{PHASE2_ROOT}/{lst}/{date}/Science_*'
    dirs = sorted(glob.glob(pattern))
    return dirs[-1] if dirs else None


def find_subband_alignment_image(sci_dir, freq_mhz):
    """I Taper Robust-0.75 non-pbcorr deep alignment image. Prefer the dewarped
    product; FALL BACK to the RAW (non-dewarped) image when the dewarp failed, so the
    date still enters the co-add and the flow-alignment-to-reference failsafe dewarps it
    against the good reference night — instead of silently dropping the date."""
    d = f'{sci_dir}/{freq_mhz}MHz/I/deep'
    if not os.path.isdir(d):
        return None
    files = sorted(os.listdir(d))
    base = lambda f: ('Taper' in f and 'NoTaper' not in f and 'Robust-0.75' in f
                      and 'pbcorr' not in f and f.endswith('.fits'))
    for f in files:                                  # preferred: dewarped
        if base(f) and f.endswith('_dewarped.fits'):
            return os.path.join(d, f)
    for f in files:                                  # failsafe: raw -> flow-align to ref
        if base(f) and 'dewarped' not in f and f.endswith('-image.fits'):
            print(f"    [failsafe] no dewarped {freq_mhz}MHz deep in {os.path.basename(sci_dir)}; "
                  f"using RAW {f} (flow-align to reference will dewarp)")
            return os.path.join(d, f)
    return None


def find_subband_variant_image(sci_dir, freq_mhz, variant):
    """Subband I deep image for a coadd VARIANT ('Taper_R0', 'NoTaper_R0',
    'NoTaper_R075'); legacy Phase-2 names {s}MHz-I-{taper}-<dur>s-Robust-{r}-
    ...-image[_dewarped].fits. Prefer dewarped, fall back to raw (the flow-
    alignment-to-reference failsafe dewarps constituents). Added 2026-07-08
    for the subband-deep tier (was a TODO stub)."""
    taper, rob = variant.split('_')
    rtok = 'Robust-0.75' if rob == 'R075' else 'Robust-0'
    d = f'{sci_dir}/{freq_mhz}MHz/I/deep'
    if not os.path.isdir(d):
        return None
    files = sorted(os.listdir(d))

    def base(f):
        if 'pbcorr' in f or not f.endswith('.fits'):
            return False
        if taper == 'Taper' and 'NoTaper' in f:
            return False
        if taper == 'NoTaper' and 'NoTaper' not in f:
            return False
        if rtok == 'Robust-0':
            return 'Robust-0-' in f
        return 'Robust-0.75' in f

    for f in files:
        if base(f) and f.endswith('_dewarped.fits'):
            return os.path.join(d, f)
    for f in files:
        if base(f) and 'dewarped' not in f and f.endswith('-image.fits'):
            return os.path.join(d, f)
    return None


def find_wideband_alignment_image(sci_dir, band, lst, date):
    """Wideband I Taper Robust-0.75 non-pbcorr deep alignment image. Prefer dewarped;
    FALL BACK to the RAW image when dewarp failed (flow-align-to-reference failsafe)."""
    d = f'{sci_dir}/{band}/I/deep'
    if not os.path.isdir(d):
        return None
    files = sorted(os.listdir(d))
    base = lambda f: ('Taper' in f and 'NoTaper' not in f and 'Robust-0.75' in f
                      and 'pbcorr' not in f and f.endswith('.fits'))
    for f in files:                                  # preferred: dewarped
        if base(f) and 'dewarped' in f:
            return os.path.join(d, f)
    for f in files:                                  # failsafe: raw -> flow-align to ref
        if base(f) and 'dewarped' not in f:
            print(f"    [failsafe] no dewarped {band} wideband in {os.path.basename(sci_dir)}; "
                  f"using RAW {f} (flow-align to reference will dewarp)")
            return os.path.join(d, f)
    return None


def find_stokes_v_deep_image(sci_dir, freq_str):
    """V deep Taper Robust-0 non-pbcorr. Prefer the dewarped product; FALL BACK
    to RAW (flow-align-to-reference dewarps it via process_variant_image) so
    dewarp-failed nights still enter the V master instead of being dropped."""
    if is_subband(freq_str):
        d = f'{sci_dir}/{freq_str}MHz/V/deep'
    else:
        d = f'{sci_dir}/{freq_str}/V/deep'
    if not os.path.isdir(d):
        return None
    base = lambda f: ('Taper' in f and 'NoTaper' not in f and
                      'Robust-0' in f and 'Robust-0.75' not in f and
                      'pbcorr' not in f and f.endswith('.fits'))
    files = sorted(os.listdir(d))
    for f in files:                                  # preferred: dewarped
        if base(f) and 'dewarped' in f:
            return os.path.join(d, f)
    for f in files:                                  # failsafe: RAW -> flow dewarps
        if base(f) and 'dewarped' not in f:
            return os.path.join(d, f)
    return None


def find_10min_images(sci_dir, freq_str, stokes):
    """Find non-pbcorr dewarped Taper Robust-0 10-min images.
    Returns list of (path, interval_idx) sorted by interval."""
    if is_subband(freq_str):
        d = f'{sci_dir}/{freq_str}MHz/{stokes}/10min'
    else:
        d = f'{sci_dir}/{freq_str}/{stokes}/10min'

    if not os.path.isdir(d):
        return []

    matches = []
    for f in sorted(os.listdir(d)):
        if not f.endswith('.fits') or 'dewarped' not in f or 'pbcorr' in f:
            continue
        if 'Taper' not in f or 'NoTaper' in f:
            continue
        if 'Robust-0' not in f or 'Robust-0.75' in f:
            continue
        matches.append(os.path.join(d, f))

    results = []
    for i, path in enumerate(sorted(matches)):
        m = re.search(r'_t(\d{4})', os.path.basename(path))
        idx = int(m.group(1)) - 1 if m else i
        results.append((path, idx))

    return sorted(results, key=lambda x: x[1])


def find_all_dewarped_variants(sci_dir, freq_str):
    """Find ALL dewarped FITS variants for this freq (I+V, deep+10min).
    Returns list of (path, stokes, depth, variant_key)."""
    if is_subband(freq_str):
        freq_dir = f'{freq_str}MHz'
    else:
        freq_dir = str(freq_str)

    base = f'{sci_dir}/{freq_dir}'
    variants = []

    for stokes in ['I', 'V']:
        for depth in ['deep', '10min']:
            d = f'{base}/{stokes}/{depth}'
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if not f.endswith('.fits'):
                    continue
                if 'dewarped' not in f:
                    continue
                if f.endswith('.pybdsf.log'):
                    continue
                vkey = classify_variant(f, stokes, depth, is_subband(freq_str))
                variants.append((os.path.join(d, f), stokes, depth, vkey))

    return variants


def classify_variant(filename, stokes, depth, is_sub):
    """Create a variant key for grouping: e.g. 'I_deep_Taper_R075' or 'V_10min_Taper_R0_pbcorr_t0003'."""
    parts = [stokes, depth]

    if 'NoTaper' in filename:
        parts.append('NoTaper')
    elif 'Taper' in filename:
        parts.append('Taper')

    if 'Robust-0.75' in filename:
        parts.append('R075')
    elif 'Robust-0' in filename:
        parts.append('R0')

    if 'pbcorr' in filename:
        parts.append('pbcorr')

    # 10-min interval number
    m = re.search(r'_t(\d{4})', filename)
    if m:
        parts.append(f't{m.group(1)}')
    elif is_sub and depth == '10min':
        # Subband 10-min: extract timestamp for ordering
        m2 = re.search(r'-(\d{8}_\d{6})-', filename)
        if m2:
            parts.append(f'ts{m2.group(1)}')

    return '_'.join(parts)


def discover_dates(lst, freq_str):
    """Find all processed dates for this (lst, freq)."""
    dates = []
    lst_dir = f'{PHASE2_ROOT}/{lst}'
    if not os.path.isdir(lst_dir):
        return dates

    for date in sorted(os.listdir(lst_dir)):
        if not re.match(r'\d{4}-\d{2}-\d{2}$', date):
            continue
        sci_dir = find_latest_science_dir(lst, date)
        if not sci_dir:
            continue

        if is_subband(freq_str):
            img = find_subband_alignment_image(sci_dir, int(freq_str))
        else:
            img = find_wideband_alignment_image(sci_dir, freq_str, lst, date)

        if img:
            dates.append((date, sci_dir, img))

    return dates


# ═══════════════════════════════════════════════════════════════════
# Regridding
# ═══════════════════════════════════════════════════════════════════

def build_template_header(fits_path):
    """Build a 2D template header from a FITS file."""
    data, hdr = load_image(fits_path)
    wcs = WCS(hdr).celestial
    tmpl = wcs.to_header()
    tmpl['NAXIS'] = 2
    tmpl['NAXIS1'] = data.shape[1]
    tmpl['NAXIS2'] = data.shape[0]
    for key in ['BMAJ', 'BMIN', 'BPA', 'BUNIT']:
        if key in hdr:
            tmpl[key] = hdr[key]
    return tmpl, data.shape


def regrid_image(input_path, template_header, output_path):
    """Reproject input to match template WCS/shape."""
    if os.path.exists(output_path):
        return output_path

    data, hdr = load_image(input_path)
    wcs_in = WCS(hdr).celestial
    hdr_2d = wcs_in.to_header()
    hdr_2d['NAXIS'] = 2
    hdr_2d['NAXIS1'] = data.shape[1]
    hdr_2d['NAXIS2'] = data.shape[0]

    wcs_out = WCS(template_header).celestial
    shape_out = (template_header['NAXIS2'], template_header['NAXIS1'])

    reprojected, _ = reproject_interp((data, hdr_2d), wcs_out, shape_out=shape_out)
    reprojected = np.nan_to_num(reprojected, nan=0.0)

    out_hdr = template_header.copy()
    for key in ['BMAJ', 'BMIN', 'BPA', 'BUNIT', 'TELESCOP', 'INSTRUME',
                'DATE-OBS', 'DATE-BEG', 'MJD-OBS', 'RESTFRQ']:
        if key in hdr:
            out_hdr[key] = hdr[key]
    out_hdr['REGRID'] = (True, 'Regridded from pre-cutover resolution')

    save_image(output_path, reprojected, out_hdr)
    return output_path


# ═══════════════════════════════════════════════════════════════════
# Flux scale measurement
# ═══════════════════════════════════════════════════════════════════

def measure_flux_scale(tgt_data, ref_data, wcs, beam_pix, horizon_mask, obs_time):
    """Measure flux scale target/reference using bright unresolved sources.
    Returns (scale_map, coeffs, n_sources)."""
    h, w = ref_data.shape
    ap_r = max(int(np.ceil(beam_pix * 0.75)), 3)
    ms = max(int(beam_pix * 1.5), 5)
    ref_rms = get_uniform_rms(ref_data, horizon_mask)

    lm = maxfilt(ref_data, size=ms)
    snr_map = ref_data / ref_rms
    peaks_y, peaks_x = np.where(
        (ref_data == lm) & (snr_map > 10) &
        (ref_data >= 10.0) & (ref_data < 200.0) &
        ~horizon_mask)

    ratios = []
    for sy, sx in zip(peaks_y, peaks_x):
        try:
            coord = wcs.pixel_to_world(sx, sy)
            altaz = coord.transform_to(AltAz(obstime=obs_time, location=OVRO_LOC))
            if altaz.alt.deg < 45.0:
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
            ref_patch = ref_data[y1:y2, x1:x2]
            total = float(np.nansum(ref_patch[ref_patch > 0.3 * ref_peak]))
            if total > 2.5 * ref_peak * (2 * ap_r + 1):
                continue
            ratios.append({'ratio': tgt_peak / ref_peak,
                           'pix_x': int(sx), 'pix_y': int(sy)})

    if len(ratios) < 5:
        return np.ones((h, w)), np.array([1.0, 0.0, 0.0]), len(ratios)

    xs = np.array([r['pix_x'] for r in ratios], dtype=float)
    ys = np.array([r['pix_y'] for r in ratios], dtype=float)
    vals = np.array([r['ratio'] for r in ratios])
    xn = 2.0 * xs / w - 1.0
    yn = 2.0 * ys / h - 1.0
    A = np.column_stack([np.ones_like(xn), xn, yn])

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

    yg, xg = np.indices((h, w), dtype=float)
    xgn = 2.0 * xg / w - 1.0
    ygn = 2.0 * yg / h - 1.0
    scale_map = coeffs[0] + coeffs[1] * xgn + coeffs[2] * ygn

    return scale_map, coeffs, len(ratios)


# ═══════════════════════════════════════════════════════════════════
# Flow field application
# ═══════════════════════════════════════════════════════════════════

def apply_flow_field(image_data, flow_offsets):
    """Apply a saved flow field to a 2D image using scipy.
    Matches the convention from image_plane_correction.flow.Flow.apply:
      output[y,x] = input[y + dy, x + dx]
    where offsets[:,:,0] = dx, offsets[:,:,1] = dy.
    """
    H, W = image_data.shape
    y, x = np.mgrid[:H, :W].astype(np.float64)
    src_y = y + flow_offsets[:, :, 1]
    src_x = x + flow_offsets[:, :, 0]
    result = map_coordinates(image_data, [src_y, src_x],
                             order=1, mode='constant', cval=0.0)
    return result


def precompute_regrid_mapping(input_path, template_header, target_shape):
    """Precompute pixel coordinate mapping from output to input WCS.
    Returns (y_in, x_in) arrays that can be reused for all variants
    sharing the same WCS. Uses low-level WCS methods to avoid
    SkyCoord overhead on millions of pixels."""
    _, hdr = load_image(input_path)
    wcs_in = WCS(hdr).celestial
    wcs_out = WCS(template_header).celestial
    H_out, W_out = target_shape
    y_out, x_out = np.mgrid[:H_out, :W_out].astype(float)
    world = wcs_out.all_pix2world(x_out.ravel(), y_out.ravel(), 0)
    pix_in = wcs_in.all_world2pix(world[0], world[1], 0)
    return pix_in[1].reshape(H_out, W_out), pix_in[0].reshape(H_out, W_out)


def process_variant_image(src_path, flow_offsets, scale_map, template_header,
                          target_shape, out_path, regrid_coords=None):
    """Load a variant image, optionally regrid, apply flux scale + flow, save.
    regrid_coords: precomputed (y_in, x_in) mapping for fast regridding."""
    if os.path.exists(out_path):
        return True

    data, hdr = load_image(src_path)

    if data.shape != target_shape:
        if regrid_coords is not None:
            y_in, x_in = regrid_coords
            data = map_coordinates(data, [y_in, x_in],
                                   order=1, mode='constant', cval=0.0)
        else:
            return False

    safe_scale = np.where(np.abs(scale_map) > 0.1, scale_map, 1.0)
    data = data / safe_scale

    if flow_offsets is not None:
        data = apply_flow_field(data, flow_offsets)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_hdr = template_header.copy()
    for key in ['BMAJ', 'BMIN', 'BPA', 'BUNIT', 'TELESCOP', 'INSTRUME',
                'DATE-OBS', 'DATE-BEG', 'MJD-OBS', 'RESTFRQ', 'CRVAL3']:
        if key in hdr:
            out_hdr[key] = hdr[key]
    save_image(out_path, data, out_hdr)
    return True


# ═══════════════════════════════════════════════════════════════════
# Residual metrics
# ═══════════════════════════════════════════════════════════════════

def compute_residual_rms(data, ref_data, horizon_mask, margin_frac=0.15):
    """Compute sigma-clipped RMS of (data - ref) in an inner region."""
    diff = data - ref_data
    h, w = diff.shape
    margin = int(min(h, w) * margin_frac)
    inner_mask = np.ones_like(horizon_mask)
    inner_mask[margin:h-margin, margin:w-margin] = horizon_mask[margin:h-margin, margin:w-margin]
    valid = diff[~inner_mask]

    if len(valid) < 100:
        return np.nan

    med = np.median(valid)
    mad = np.median(np.abs(valid - med))
    sigma = mad * 1.4826
    clip = valid[np.abs(valid - med) < 4 * sigma]
    return float(np.std(clip)) if len(clip) > 50 else float(sigma)



def coadd_beam_gate(lst, fl, dates_list):
    """CO-ADD BEAM GATE (Gregg approved 2026-09-02; shipped in the wideband path
    2026-09-03; extended to the subband and 10-min paths 2026-09-05 after a census
    found 20 of 360 subband masters holding a constituent above 1.5x): refuse epochs
    whose Phase-2 survey Stokes I (Taper R-0.75) fitted-beam AREA exceeds 1.5x the
    median of the candidate set (long-baseline dropout). Keyed on I. Fails OPEN per
    epoch when no survey product/beam header is found. Returns (refused_dates, areas)."""
    _bareas = {}
    for _d in dates_list:
        _g = [x for x in glob.glob(
            f'/lustre/pipeline/exopipe/phase2/{lst}/{_d}/*/{fl}/I/deep/'
            f'*Taper_Robust-0.75_dewarped*.fits')
            + glob.glob(
            f'/lustre/pipeline/exopipe/phase2/{lst}/{_d}/*/{fl}/I/deep/'
            f'*Taper-*-Robust-0.75-*dewarped*.fits')
            if 'pbcorr' not in os.path.basename(x)]
        if _g:
            try:
                _h = fits.getheader(_g[0])
                _bareas[_d] = float(_h['BMAJ']) * float(_h['BMIN'])
            except Exception:
                pass
    _bfin = sorted(_bareas.values())
    beam_outliers = []
    if len(_bfin) >= 5:
        _bmed = _bfin[len(_bfin) // 2]
        beam_outliers = [d for d in dates_list
                         if _bareas.get(d, 0) > 1.5 * _bmed]
        if beam_outliers:
            print(f"  BEAM GATE: {len(beam_outliers)} epoch(s) refused "
                  f"from the master (fitted-beam area >1.5x set median): "
                  f"{[(d, round(_bareas[d]/_bmed, 2)) for d in beam_outliers]}")
    return beam_outliers, _bareas

# ═══════════════════════════════════════════════════════════════════
# Outlier detection (MAD-based)
# ═══════════════════════════════════════════════════════════════════

def detect_outliers_mad(dates, rms_values, sigma_thresh=None):
    """Flag dates where RMS > median + sigma_thresh * MAD * 1.4826.
    Returns list of outlier dates and the threshold used.
    sigma_thresh=None reads OUTLIER_SIGMA at CALL time (C-03 fix 2026-08-11:
    the old default arg froze 3.0 at import, making --outlier_sigma a silent
    no-op while the log lines printed the overridden value)."""
    if sigma_thresh is None:
        sigma_thresh = OUTLIER_SIGMA
    valid = [(d, r) for d, r in zip(dates, rms_values) if np.isfinite(r)]
    if len(valid) < 5:
        return [], np.nan

    vals = np.array([r for _, r in valid])
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    sigma = mad * 1.4826
    thresh = med + sigma_thresh * sigma
    # ASYMMETRIC gate (R-GC5 ruled 2026-08-13, Gregg 2026-09-05 "I wanted that
    # asymmetric gate"): the LOW side rejects too at LOW_OUTLIER_SIGMA (4 or 5;
    # survey statistics 2026-09-05: 0 epochs below -4 sigma, 3 below -3 sigma of
    # 6,997) - a night far below the cell population is a fingerprint of a
    # broken image (empty field, wrong scale), not of good weather.
    thresh_lo = med - LOW_OUTLIER_SIGMA * sigma
    outliers = [d for d, r in valid if r > thresh or r < thresh_lo]
    low = [d for d, r in valid if r < thresh_lo]
    if low:
        print(f"  LOW-SIDE outlier(s) rejected (< median - {LOW_OUTLIER_SIGMA} sigma = {thresh_lo:.1f}): {low}")
    return outliers, float(thresh)


# ═══════════════════════════════════════════════════════════════════
# Co-addition
# ═══════════════════════════════════════════════════════════════════

def _purge_stale_loo(aligned, out_dir, tag):
    """Remove this Stokes' leave-one-out masters whose epoch is not in `aligned` (the current
    constituents; empty when the stack is outside [2,20] and keeps none). Purge ONLY this
    Stokes' files: a date-only match let the V pass delete the I side's LOO masters
    (found 2026-09-03, 06h_55MHz)."""
    if not os.path.isdir(out_dir):
        return
    _keep = {d for d, _p, _w in aligned}
    for _f in os.listdir(out_dir):
        _m = re.match(r'(\d{4}-\d{2}-\d{2})_', _f)
        if _m and tag in _f and _m.group(1) not in _keep:
            try:
                os.remove(os.path.join(out_dir, _f))
                print(f'    purged stale LOO master (epoch no longer a '
                      f'constituent, or stack outside [2,20]): {_f}', flush=True)
            except OSError:
                pass


def write_loo_masters(aligned, master_path, out_dir, tag):
    """Leave-one-out per-epoch masters (R-LOO-MASTERS, Gregg 2026-08-25;
    threshold corrected same day: REQUIRED for cells with N_epochs <= 20).
    Exact identity with the plain-mean master: LOO_i = (S - w_i e_i) /
    (W - w_i), computed with per-pixel validity exactly as the no-clip
    co-add (zeros = unimaged). Each product carries LOOEXCL + NCOADD."""
    # PURGE FIRST (2026-09-10, INCIDENT-STALE-V2-LOO-MASTERS-KEPT-WHEN-STACK-GREW-PAST-20): the purge
    # below used to run only after this N-range return, so a cell whose stack grew above 20 kept the
    # previous generation's leave-one-out masters (03h Red: 40 v2 files beside an N=22 v3 master); the
    # search refuses them by COADDOP/PARENTDG and the cell fails. A stack outside [2,20] keeps NO
    # leave-one-out master of this Stokes.
    _purge_stale_loo(aligned if 2 <= len(aligned) <= 20 else [], out_dir, tag)
    if len(aligned) > 20 or len(aligned) < 2:
        print(f"    LOO skipped for {tag}: N_epochs={len(aligned)} "
              f"outside [2,20] (R-LOO-MASTERS)", flush=True)
        return 0
    imgs = {}
    S = None
    Wmap = None
    for d, p, w in aligned:
        a, _ = load_image(p)
        v = (a != 0).astype(np.float32)
        imgs[d] = (a, w, v)
        S = a * w if S is None else S + a * w
        Wmap = v * w if Wmap is None else Wmap + v * w
    os.makedirs(out_dir, exist_ok=True)
    # PURGE STALE LOO MASTERS (2026-09-02). write_loo_masters only ever
    # ADDED files, so a cell rebuilt with a different epoch set kept the
    # previous set's leave-one-out masters alongside the new ones. Found on
    # 23h_32MHz: loo/ held 23 files for an 11-epoch master because
    # 2025-08-18_I_loo_* survived from an earlier build although 2025-08-18
    # is no longer a constituent - and it had no V counterpart, so the
    # directory was internally inconsistent too. A consumer indexing LOO
    # masters by date would have been handed a master built against an
    # epoch set that no longer exists. Remove anything whose date is not in
    # the current aligned set before writing.
    mhdr = fits.getheader(master_path)
    n = 0
    for d, p, w in aligned:
        a, _w, v = imgs[d]
        den = Wmap - v * _w
        loo = np.where(den > 0, (S - a * _w) / np.where(den > 0, den, 1.0), 0.0)
        hdr = mhdr.copy()
        hdr['LOOEXCL'] = (d, 'epoch excluded from this LOO master')
        hdr['PARENTDG'] = (str(mhdr.get('CONSTDIG', ''))[:32], 'CONSTDIG of the parent central master')
        hdr['NCOADD'] = (len(aligned) - 1, 'constituents (LOO)')
        hdr['HISTORY'] = f'LOO master excluding {d} (R-LOO-MASTERS 2026-08-25)'
        save_image(f'{out_dir}/{d}_{tag}', loo.astype(np.float32), hdr)
        n += 1
    return n


def coadd_inverse_variance(image_paths, weights, output_path, ref_header,
                           dates=None, sigclip=True, nsig=4.0):
    """Per-pixel SIGMA-CLIPPED inverse-variance co-add (Gregg 2026-06-21).
    Clipping rejects moving/variable per-pixel outliers (Jupiter streak, Crab
    variability, transients) that a plain mean smears in, while keeping full
    thermal depth on clean pixels. Falls back to plain mean for <4 images.
    `dates` (optional, parallel to image_paths) is stamped as HISTORY provenance."""
    if not image_paths:
        return None

    stack = []
    beam_entries = []
    for p in image_paths:
        d, h = load_image(p)
        stack.append(d)
        if h is not None and 'BMAJ' in h:
            beam_entries.append((h['BMAJ'], h.get('BMIN', h['BMAJ']),
                                 h.get('BPA', 0.0)))
        else:
            beam_entries.append(None)
    stack = np.array(stack)
    w_arr = np.array(weights, dtype=float)[:, np.newaxis, np.newaxis]
    valid_mask = stack != 0

    if sigclip and len(image_paths) >= 4:
        med = np.median(stack, axis=0)
        mad = 1.4826 * np.median(np.abs(stack - med[None]), axis=0)
        mad = np.where(mad > 0, mad, np.std(stack, axis=0))
        keep = (np.abs(stack - med[None]) <= nsig * mad[None]) & valid_mask
    else:
        keep = valid_mask
    w_eff = w_arr * keep.astype(float)
    sum_w = np.sum(w_eff, axis=0)
    safe = sum_w.copy(); safe[safe == 0] = 1.0
    result = np.sum(stack * w_eff, axis=0) / safe
    # where everything got clipped/zeroed, fall back to plain valid mean
    none_kept = np.sum(keep, axis=0) == 0
    if np.any(none_kept):
        vm = valid_mask.astype(float); sw = np.sum(w_arr * vm, axis=0); sw[sw == 0] = 1.0
        fb = np.sum(stack * w_arr * vm, axis=0) / sw
        result[none_kept] = fb[none_kept]
    result[np.sum(valid_mask, axis=0) == 0] = 0.0

    out_hdr = ref_header.copy()
    # BEAM FIX (2026-07-11): the template header carries ONE epoch's fitted
    # beam (post_dates[0], Taper_R075) — wrong epoch for bad-night templates
    # and wrong variant for R0/V stacks. Stamp the stack-weighted mean of the
    # constituents' own per-variant beams instead (BPA: 180deg-period circular
    # mean), preserving the template values as *ORIG.
    _be = [(b, w) for b, w in zip(beam_entries, weights) if b is not None]
    if _be:
        _w = np.array([w for _, w in _be], dtype=float)
        if _w.sum() <= 0:
            _w = np.ones(len(_be))
        _bm = np.array([b for b, _ in _be], dtype=float)
        for k, v in (('BMAJORIG', out_hdr.get('BMAJ')),
                     ('BMINORIG', out_hdr.get('BMIN')),
                     ('BPAORIG', out_hdr.get('BPA'))):
            if v is not None:
                out_hdr[k] = (v, 'template-epoch value (pre beam fix)')
        out_hdr['BMAJ'] = float(np.sum(_bm[:, 0] * _w) / _w.sum())
        out_hdr['BMIN'] = float(np.sum(_bm[:, 1] * _w) / _w.sum())
        _ang = np.deg2rad(_bm[:, 2] * 2.0)
        out_hdr['BPA'] = float(np.rad2deg(0.5 * np.arctan2(
            np.sum(_w * np.sin(_ang)), np.sum(_w * np.cos(_ang)))))
        out_hdr['BEAMSRC'] = (f'ivw-mean {len(_be)} constituents',
                              'per-variant stack-weighted beam')
    out_hdr['NCOADD'] = (len(image_paths), 'Number of co-added images')
    out_hdr['COADDW'] = ('inv_var', 'Co-addition weighting method')
    out_hdr['COMBINE'] = ('sigclip_ivw' if (sigclip and len(image_paths) >= 4)
                          else 'ivw_mean', '%g-sigma per-pixel clip'%nsig)
    # Operator version (R-LOO-MASTERS 2026-08-25): clipped-era and plain-mean
    # masters must never be confusable on disk.
    # v3 = flat-DC flux scale (R-TILT-REMOVED-NEXTGEN landed 2026-09-08); v2
    # masters were built with the affine tilt and are superseded.
    _tilt_on = os.environ.get('P3_FLUXSCALE_TILT') == '1'
    out_hdr['COADDOP'] = ('ivw-clip-v1' if (sigclip and len(image_paths) >= 4)
                          else ('ivw-mean-v2' if _tilt_on else 'ivw-mean-v3'),
                          'co-add operator version')
    out_hdr['FLUXFIT'] = ('affine-tilt' if _tilt_on else 'flat-dc',
                          'per-epoch flux-scale fit applied to constituents')
    if dates:
        for d in dates:
            out_hdr['HISTORY'] = f"COADD constituent {d}"
        # Constituent-generation digest (ruled 2026-08-12, approved 2026-09-05):
        # md5 of the sorted constituent date list; every master-derived product
        # inherits the card so a product built from a superseded master is
        # detectable from its header alone.
        import hashlib as _hl
        out_hdr['CONSTDIG'] = (_hl.md5(','.join(sorted(set(dates))).encode()).hexdigest(),
                               'md5 of sorted constituent dates')
    save_image(output_path, result, out_hdr)
    return output_path


# ═══════════════════════════════════════════════════════════════════
# Diff-median movie
# ═══════════════════════════════════════════════════════════════════

def make_diff_median_movie(aligned_paths, dates, ref_path, output_path,
                           lst, freq_str, stokes, movie_size=800, fps=2):
    """Create diff-from-median MP4 movie."""
    images = []
    for p in aligned_paths:
        d, _ = load_image(p)
        images.append(d)

    if len(images) < 2:
        return

    stack = np.array(images)
    median_img = np.median(stack, axis=0)
    horizon_mask = (median_img == 0)

    # Determine color scale from typical residual
    all_p99 = []
    for img in images:
        diff = img - median_img
        valid = diff[~horizon_mask & (img != 0)]
        if len(valid) > 0:
            all_p99.append(np.percentile(np.abs(valid), 99))
    diff_scale = float(np.median(all_p99)) if all_p99 else 1.0

    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 28)
        font_sm = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    frames = []
    for i, (date, img) in enumerate(zip(dates, images)):
        diff = img - median_img
        norm = np.clip(diff / diff_scale, -1, 1)
        r = np.where(norm > 0, 255, (128 + 127 * (1 + norm)).astype(np.uint8))
        g = (255 * (1 - np.abs(norm))).astype(np.uint8)
        b = np.where(norm < 0, 255, (128 + 127 * (1 - norm)).astype(np.uint8))
        rgb = np.stack([r.astype(np.uint8), g, b.astype(np.uint8)], axis=-1)
        rgb = np.flipud(rgb)
        pil = Image.fromarray(rgb, mode='RGB').resize(
            (movie_size, movie_size), Image.LANCZOS)

        # Compute per-frame RMS
        valid_diff = diff[~horizon_mask & (img != 0)]
        if len(valid_diff) > 100:
            med_d = np.median(valid_diff)
            mad_d = np.median(np.abs(valid_diff - med_d))
            sig = mad_d * 1.4826
            clip = valid_diff[np.abs(valid_diff - med_d) < 4 * sig]
            rms = float(np.std(clip)) if len(clip) > 50 else float(sig)
        else:
            rms = 0.0

        draw = ImageDraw.Draw(pil)
        top_text = f"LST {lst}  {freq_label(freq_str)}  {stokes} — {date} (diff from median)"
        bot_text = (f"Frame {i+1}/{len(images)}  |  "
                    f"scale=±{diff_scale:.2f} Jy  RMS={rms*1000:.0f} mJy")
        for txt, fnt, ypos in [(top_text, font, 12), (bot_text, font_sm, movie_size - 32)]:
            bbox = draw.textbbox((0, 0), txt, font=fnt)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([8, ypos - 4, 16 + tw, ypos + th + 4], fill=(0, 0, 0, 200))
            draw.text((12, ypos), txt, font=fnt, fill=(255, 255, 255))

        frames.append(pil)

    w = movie_size if movie_size % 2 == 0 else movie_size + 1
    h = w
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}', '-pix_fmt', 'rgb24', '-r', str(fps),
        '-i', '-',
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
        '-crf', '18', '-preset', 'medium',
        output_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for frame in frames:
        frame = frame.resize((w, h), Image.LANCZOS)
        proc.stdin.write(frame.convert('RGB').tobytes())
    proc.stdin.close()
    proc.wait()


# ═══════════════════════════════════════════════════════════════════
# Main pipeline logic
# ═══════════════════════════════════════════════════════════════════

REF_ASTROM_MAX_AMIN = 3.0
VLSSR_REF = ('/lustre/gh/calibration/pipeline/reference/surveys/'
             'vlssr_radecpeak_unresolved.txt')
_VLSSR_CACHE = None


def _ref_astrom_thresh_amin(freq_label):
    """Frequency-scaled gate threshold (Gregg 2026-07-10): the achievable
    reference astrometry degrades toward the bottom of the band (measured
    medians: 0.4-0.6' at 64-82 MHz, 1.7' at 23 MHz, 2.7' at 18 MHz), while
    the failure class this gate exists for is 16-25'. Below a 36 MHz pivot
    scale the 3' wideband threshold by nu^-1.5 (the pipeline's dewarp
    match-radius law): 27 MHz 3.9', 23 MHz 5.9', 18 MHz 8.5'. Wideband
    labels (Red/Green/Blue/Full) keep the validated 3'."""
    try:
        nu = float(str(freq_label).replace('MHz', ''))
        return REF_ASTROM_MAX_AMIN * max(1.0, (36.0 / nu) ** 1.5)
    except (TypeError, ValueError):
        return REF_ASTROM_MAX_AMIN


def _reference_astrom_ok(ref_img_path, label='', freq_label=None):
    """REFERENCE-QUALITY GATE (Gregg 2026-07-08): the alignment reference
    image must itself be astrometrically sane — a bad reference frame warps
    the WHOLE cell (13h ~25' via stale flows; 11h Green ~15' via the
    raw-fallback failsafe applied to a dewarp-failed reference). Quick VLSSr
    peak match; block the cell build if the median separation exceeds the
    frequency-scaled threshold (_ref_astrom_thresh_amin). A-02 REVISED
    (2026-08-11): fails CLOSED on measurement error AND on <10 anchors —
    any validation failure on the designated reference stops the build
    loudly (ESCALATE-TO-GREGG); no silent passes, no substitution."""
    global _VLSSR_CACHE
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from scipy.ndimage import maximum_filter, center_of_mass
        if _VLSSR_CACHE is None:
            vv = np.loadtxt(VLSSR_REF, skiprows=1)
            _VLSSR_CACHE = SkyCoord(vv[:, 0] * u.deg, vv[:, 1] * u.deg)
        a, hdr = load_image(ref_img_path)
        # load_image() already zero-fills NaN, so a finite-fraction here
        # would always read 100%. Record what the test actually looks at.
        a = np.nan_to_num(np.asarray(a, dtype=float))
        _npos = int((a > 0).sum())
        w = WCS(hdr).celestial
        n = a.shape[0]
        q = n // 8
        cen = a[q:-q, q:-q]
        mfl = maximum_filter(cen, size=25)
        py, px = np.where((cen == mfl) & (cen > 0))
        if not len(py):
            # A-02 SILENT PASS FIXED (2026-09-01). This returned True — a
            # silent pass — contradicting this function's own docstring
            # ("fails CLOSED on measurement error ... no silent passes").
            # It fires when the reference image has no positive local
            # maximum at all, i.e. it is BLANK. Measured on the archive:
            # 01h/2024-12-21/69MHz and 16h/2025-05-03/69MHz are 0.0%
            # finite (raw AND dewarped — dead upstream), and
            # 15h/2025-05-03/78MHz has a 100%-finite raw whose DEWARPED
            # product is 0.0% finite — the dewarp destroyed it. All three
            # were being passed as acceptable alignment references.
            print(f"  [ref-astrom] ERROR: {label or os.path.basename(ref_img_path)} "
                  f"has NO positive local maximum — the reference image is "
                  f"BLANK ({_npos} positive pixels, max {float(a.max()):.4g}). "
                  f"REJECTING (A-02: fail closed, ESCALATE-TO-GREGG)")
            return False
        fx = cen[py, px]
        seps = []
        for i in np.argsort(fx)[::-1][:50]:
            y, x = int(py[i]) + q, int(px[i]) + q
            box = a[y - 4:y + 5, x - 4:x + 5]
            if box.size != 81 or not np.all(np.isfinite(box)) \
                    or box.max() <= 0:
                continue
            cy, cx = center_of_mass(np.maximum(box, 0))
            sky = w.pixel_to_world(x - 4 + cx, y - 4 + cy)
            d = float(sky.separation(_VLSSR_CACHE).arcmin.min())
            if d < 60:
                seps.append(d)
        if len(seps) < 10:
            # A-02 REVISED (Gregg 2026-08-11 round 2): <10 VLSSr anchors on
            # the DESIGNATED reference is a validation failure — loud stop,
            # never a silent pass. ESCALATE-TO-GREGG.
            print(f"  [ref-astrom] ERROR: only {len(seps)} VLSSr anchors on "
                  f"{label or os.path.basename(ref_img_path)} (<10) — REJECTING "
                  f"designated reference. ESCALATE-TO-GREGG (A-02 revised "
                  f"2026-08-11: investigate or reassign REFERENCE_DATES)")
            return False
        med = float(np.median(seps))
        thr = _ref_astrom_thresh_amin(freq_label)
        ok = med <= thr
        print(f"  [ref-astrom] {label or os.path.basename(ref_img_path)}: "
              f"median VLSSr sep {med:.2f} arcmin (thresh {thr:.1f}) -> "
              f"{'OK' if ok else 'REJECTED'}")
        return ok
    except Exception as e:
        # A-02 (Gregg 2026-08-11): fail CLOSED, loudly. A reference whose
        # astrometry cannot be measured must not anchor an alignment.
        print(f"  [ref-astrom] ERROR: check FAILED ({e}) — REJECTING reference "
              f"(A-02 ruling 2026-08-11: fail closed on measurement error)")
        return False


def _ensure_reference_fingerprint(workdir, lst):
    """A REFERENCE_DATES change MUST invalidate the cell's alignment caches
    (flow/, prep/, aligned/, prep_final/) — silently reusing flow fields
    computed against the old reference warps every epoch into the old frame
    (the 2026-07-05 13h ~25' warp; COADD_ASTROMETRIC_ERROR.md §4). Marker:
    flow/.reference. Stale caches are moved aside, never deleted."""
    ref = REFERENCE_DATES.get(lst) or ''
    mk = f'{workdir}/flow/.reference'
    old = open(mk).read().strip() if os.path.exists(mk) else None
    if old is not None and old != ref:
        stale = f'{workdir}/stale_ref_{old}_{time.strftime("%Y%m%d_%H%M")}'
        os.makedirs(stale, exist_ok=True)
        for sub in ('flow', 'prep', 'aligned', 'prep_final'):
            p = f'{workdir}/{sub}'
            if os.path.isdir(p):
                shutil.move(p, f'{stale}/{sub}')
        print(f"  [reference-change] {os.path.basename(workdir)}: caches for "
              f"old reference {old} moved to {stale}; recomputing vs {ref}")
    os.makedirs(f'{workdir}/flow', exist_ok=True)
    with open(mk, 'w') as fh:
        fh.write(ref)


def _resolve_reference_entry(lst, fl, ref_date, dates, workdir):
    """PER-SUBBAND REFERENCE FALLBACK (Gregg 2026-07-10, for 05h 50MHz: the
    LST's reference date can lack ONE subband's Phase-2 products while the
    rest of the cell is healthy). If ref_date has no entry in `dates`, walk
    the available dates by proximity to ref_date (tie -> earlier) and take
    the first that PASSES the astrometry gate. Deterministic given the same
    inputs, so all four stages resolve identically; the resolved date is
    fingerprinted (flow/.reference_resolved) and a change moves the
    alignment caches aside exactly like a REFERENCE_DATES change."""
    entries = [x for x in dates if x[0] == ref_date]
    chosen = None
    if entries:
        chosen = entries[0]
    else:
        # A-02 REVISED (Gregg 2026-08-11 round 2): the reference night per
        # LST is GREGG'S designation; the pipeline NEVER substitutes it.
        # The 2026-07-10 per-subband proximity-fallback walk is retired —
        # a designated reference with no product STOPS the build loudly.
        raise RuntimeError(
            f"A-02 REVISED: designated reference {ref_date} has no product "
            f"for {lst} {fl} — STOPPING this build. ESCALATE-TO-GREGG "
            f"(investigate or reassign REFERENCE_DATES; no automatic "
            f"fallback to other nights, ruling 2026-08-11)")
    mk = f'{workdir}/flow/.reference_resolved'
    old = open(mk).read().strip() if os.path.exists(mk) else None
    if old is not None and old != chosen[0]:
        stale = f'{workdir}/stale_ref_{old}_{time.strftime("%Y%m%d_%H%M")}'
        os.makedirs(stale, exist_ok=True)
        for sub in ('flow', 'prep', 'aligned', 'prep_final'):
            p = f'{workdir}/{sub}'
            if os.path.isdir(p):
                shutil.move(p, f'{stale}/{sub}')
        print(f"  [reference-change] {os.path.basename(workdir)}: resolved "
              f"reference moved {old} -> {chosen[0]}; caches -> {stale}")
    os.makedirs(f'{workdir}/flow', exist_ok=True)
    with open(mk, 'w') as fh:
        fh.write(chosen[0])
    return chosen


REF_MODE = os.environ.get('P3_REF_MODE', 'master')   # 'master' (default): previous-generation master; 'table': fixed night only


def _select_reference(lst, fl, workdir, dates):
    """ALIGNMENT + FLUX-SCALE REFERENCE (Gregg 2026-09-09, verbatim: "previous generation
    Master serves as Reference for alignment. If one doesn't exist, draw from a fixed
    table."; ledger R-ALIGNMENT-REFERENCE-IS-PREVIOUS-GENERATION-MASTER).
    When the cell's previous-generation I Taper R-0.75 master exists it is SNAPSHOTTED
    (before the rebuild overwrites coadd_final/) and returned as the reference entry
    ('MASTER', None, snapshot); every night, the fixed-table night included, is then a
    target with its own flow and scale. The snapshot is fingerprinted by COADDOP+CONSTDIG
    in flow/.reference_resolved so a changed reference moves the alignment caches aside
    exactly as a REFERENCE_DATES change does. Without a previous master the fixed-table
    night is resolved as before (A-02: never substituted)."""
    prev = f'{workdir}/coadd_final/I_deep_Taper_R075_{fl}.fits'
    if REF_MODE == 'master' and os.path.exists(prev):
        try:
            h = fits.getheader(prev)
        except Exception:
            h = None
        if h is not None and h.get('COADDOP'):
            sig = f"MASTER:{h.get('COADDOP')}:{h.get('CONSTDIG', 'nodigest')}"
            mk = f'{workdir}/flow/.reference_resolved'
            old = open(mk).read().strip() if os.path.exists(mk) else None
            if old is not None and old != sig:
                stale = f'{workdir}/stale_ref_{old.replace(":", "_")[:40]}_{time.strftime("%Y%m%d_%H%M")}'
                os.makedirs(stale, exist_ok=True)
                for sub in ('flow', 'prep', 'aligned', 'prep_final'):
                    pth = f'{workdir}/{sub}'
                    if os.path.isdir(pth):
                        shutil.move(pth, f'{stale}/{sub}')
                print(f"  [reference-change] {os.path.basename(workdir)}: reference "
                      f"{old} -> {sig}; caches -> {stale}")
            os.makedirs(f'{workdir}/flow', exist_ok=True)
            os.makedirs(f'{workdir}/prep', exist_ok=True)
            snap = f'{workdir}/prep/ref_master_prev.fits'
            if old != sig or not os.path.exists(snap):
                shutil.copy2(prev, snap + '.part'); os.replace(snap + '.part', snap)
                prev_v = f'{workdir}/coadd_final/V_deep_Taper_R0_{fl}.fits'
                if os.path.exists(prev_v):
                    _sv = snap.replace('.fits', '_V.fits')
                    shutil.copy2(prev_v, _sv + '.part'); os.replace(_sv + '.part', _sv)
            with open(mk, 'w') as fh:
                fh.write(sig)
            print(f"  [reference] {lst} {fl}: previous-generation master is the alignment/flux "
                  f"reference ({sig}; snapshot {os.path.basename(snap)})")
            return ('MASTER', None, snap)
    ref_date = REFERENCE_DATES.get(lst)
    if not ref_date:
        raise RuntimeError(f"{lst} {fl}: no previous-generation master and no REFERENCE_DATES "
                           f"entry - no alignment reference; REFUSING")
    print(f"  [reference] {lst} {fl}: no previous-generation master; fixed-table night {ref_date}")
    return _resolve_reference_entry(lst, fl, ref_date, dates, workdir)


def process_lst_freq(lst, freq_str, exclusions, args):
    """Full pipeline for one (LST, freq) combination.
    Returns (manifest_entries, prep_data) for batch flow phase."""
    ref_date = REFERENCE_DATES.get(lst) or 'MASTER'

    fl = freq_label(freq_str)
    workdir = f'{_workdir_base()}/{lst}_{fl}'
    _ensure_reference_fingerprint(workdir, lst)
    for sub in ['prep', 'flow', 'aligned', 'coadd', 'movies']:
        os.makedirs(f'{workdir}/{sub}', exist_ok=True)

    # Discover all dates
    all_dates = discover_dates(lst, freq_str)
    if not all_dates:
        print(f"  No images found for {lst} {fl}")
        return [], None

    # Filter exclusions
    dates = [(d, sd, img) for d, sd, img in all_dates
             if not is_excluded(lst, freq_str, d, exclusions)]
    # wrong-grid epochs are refused BEFORE the template is chosen (19h Full: the first post-cutover
    # night, 2025-04-05 on the Green grid, was the template and every 3122-grid night then
    # 'skipped' with no flow) - the template comes from the MODAL grid, as in the other stages
    _gr0 = _grid_refused_dates(dates, f'{lst}_{fl}')
    if _gr0:
        with open(f'{workdir}/refusals_grid.csv', 'a') as _rf:
            for _d in sorted(_gr0):
                _rf.write(f'{lst},{fl},{_d},wrong grid (Phase-2 anomaly),prep\n')
        dates = [x for x in dates if x[0] not in _gr0]

    # Check reference exists (per-subband fallback if its product is missing)
    ref_date_entry = _select_reference(lst, fl, workdir, dates)
    if ref_date_entry is None:
        print(f"  Reference {ref_date} not found for {lst} {fl} and no usable "
              f"fallback (available: {[d for d,_,_ in dates[:5]]}...)")
        return [], None

    ref_date = ref_date_entry[0]
    target_dates = [x for x in dates if x[0] != ref_date]
    target_dates = [x for x in target_dates if x[0] not in _grid_refused_dates(target_dates, f'{lst}_{fl}')]

    if len(target_dates) < MIN_DATES_FOR_COADD - 1:
        print(f"  Only {len(target_dates)} dates after filtering, need {MIN_DATES_FOR_COADD}, skipping")
        return [], None

    print(f"\n{'═'*70}")
    print(f"  {lst} {fl}: {len(target_dates)} dates + reference ({ref_date})")
    print(f"  Workdir: {workdir}")
    print(f"{'═'*70}")

    # Build template from the MODAL post-cutover grid (never post_dates[0]: template poisoning)
    post_dates = [x for x in dates if x[0] >= CUTOVER_DATE]
    if post_dates:
        tmpl_hdr, target_shape = _modal_grid_template(post_dates, ref_date_entry)
    else:
        # All pre-cutover: use ref as template (it'll be self-consistent)
        tmpl_hdr, target_shape = build_template_header(ref_date_entry[2])

    # Prepare reference
    ref_img_path = ref_date_entry[2]
    if ref_date < CUTOVER_DATE and post_dates:
        regrid_path = f'{workdir}/prep/ref_{ref_date}_regrid.fits'
        ref_img_path = regrid_image(ref_img_path, tmpl_hdr, regrid_path)
        print(f"  Regridded reference")

    if not _reference_astrom_ok(ref_img_path, f'{lst} {fl} ref {ref_date}',
                                freq_label=fl):
        print(f"  {lst} {fl}: REFERENCE {ref_date} REJECTED by the astrometry "
              f"gate — fix its dewarp or change REFERENCE_DATES; NOT building")
        return [], None

    ref_data, ref_hdr = load_image(ref_img_path)
    if ref_date == 'MASTER' and ref_data.shape != target_shape:
        os.makedirs(f'{workdir}/prep', exist_ok=True)
        ref_img_path = regrid_image(ref_img_path, tmpl_hdr, f'{workdir}/prep/ref_master_prev_regrid.fits')
        ref_data, ref_hdr = load_image(ref_img_path)
        print(f"  [reference] {lst} {fl}: previous master regridded to the template grid")
    ref_wcs = WCS(ref_hdr).celestial
    cd = abs(ref_hdr.get('CDELT1', ref_hdr.get('CD1_1', 0.037)))
    bmaj = ref_hdr.get('BMAJ', 5 * cd)
    beam_pix = bmaj / cd
    horizon_mask = ~np.isfinite(ref_data) | (ref_data == 0)
    ref_rms = get_uniform_rms(ref_data, horizon_mask)

    # Save reference info
    prep_data = {
        'lst': lst, 'freq_str': freq_str, 'ref_date': ref_date,
        'workdir': workdir, 'ref_img_path': ref_img_path,
        'tmpl_hdr': tmpl_hdr, 'target_shape': target_shape,
        'ref_sci_dir': ref_date_entry[1],
        'dates': [],
    }

    manifest_entries = []

    for date, sci_dir, align_img_path in target_dates:
        t0 = time.time()

        # Regrid if needed
        needs_regrid = date < CUTOVER_DATE and post_dates
        if needs_regrid:
            rg_path = f'{workdir}/prep/{date}_regrid.fits'
            work_path = regrid_image(align_img_path, tmpl_hdr, rg_path)
        else:
            work_path = align_img_path

        tgt_data, tgt_hdr = load_image(work_path)
        if tgt_data.shape != target_shape:
            print(f"  {date}: shape mismatch {tgt_data.shape} vs {target_shape}, skipping")
            continue

        tgt_mask = ~np.isfinite(tgt_data) | (tgt_data == 0)
        tgt_obs_time = get_midpoint_time(tgt_hdr, work_path)
        tgt_rms = get_uniform_rms(tgt_data, tgt_mask)

        # Flux scale
        scale_map, coeffs, n_src = measure_flux_scale(
            tgt_data, ref_data, ref_wcs, beam_pix,
            horizon_mask | tgt_mask, tgt_obs_time)

        mean_scale = float(coeffs[0])
        tgt_rms_scaled = tgt_rms / abs(mean_scale) if abs(mean_scale) > 0.1 else tgt_rms
        weight = float(mean_scale**2 / tgt_rms**2) if tgt_rms > 0 else 0.0

        # Save flux-scaled alignment image for flow input
        safe_scale = np.where(np.abs(scale_map) > 0.1, scale_map, 1.0)
        tgt_scaled = tgt_data / safe_scale
        scaled_path = f'{workdir}/prep/{date}_scaled.fits'
        save_image(scaled_path, tgt_scaled, tgt_hdr)

        # Save scale parameters
        scale_path = f'{workdir}/prep/{date}_scale.json'
        with open(scale_path, 'w') as f:
            json.dump({
                'coeffs': coeffs.tolist(), 'n_sources': n_src,
                'mean_scale': mean_scale, 'tgt_rms': tgt_rms,
                'tgt_rms_scaled': tgt_rms_scaled, 'weight': weight,
            }, f)

        # Add to flow manifest
        manifest_entries.append({
            'target': scaled_path,
            'reference': ref_img_path,
            'out_dir': f'{workdir}/flow',
        })

        dt = time.time() - t0
        print(f"  {date}: scale={mean_scale:.3f} ({n_src} src)  "
              f"rms={tgt_rms*1000:.1f}→{tgt_rms_scaled*1000:.1f} mJy  "
              f"w={weight:.3f}  {dt:.1f}s")

        prep_data['dates'].append({
            'date': date, 'sci_dir': sci_dir,
            'align_img': align_img_path,
            'needs_regrid': needs_regrid,
            'scale_map_coeffs': coeffs.tolist(),
            'weight': weight,
            'tgt_rms': tgt_rms, 'tgt_rms_scaled': tgt_rms_scaled,
        })

    return manifest_entries, prep_data


def _require_flow(flow_path, date, ref_date, tag):
    """Load the epoch's co-add alignment flow, or REFUSE (Gregg 2026-09-09, contract A5:
    a raw night with no flow is refused; 'using flux-scale only' is no longer a path).
    The reference epoch is the alignment target and carries no flow by construction."""
    if os.path.exists(flow_path):
        return np.load(flow_path)
    if date == ref_date:
        return None
    raise RuntimeError(f"{tag} {date}: no co-add alignment flow at {flow_path} - REFUSING "
                       f"(run --stage flow first; the flux-scale-only fallback was retired "
                       f"2026-09-09, contract A5)")


def post_flow_processing(prep_data, exclusions, args):
    """After batch flow: apply corrections to all variants, detect outliers,
    co-add, make movies."""
    lst = prep_data['lst']
    freq_str = prep_data['freq_str']
    ref_date = prep_data['ref_date']
    workdir = prep_data['workdir']
    fl = freq_label(freq_str)
    target_shape = prep_data['target_shape']
    tmpl_hdr = prep_data['tmpl_hdr']
    ref_img_path = prep_data['ref_img_path']
    ref_sci_dir = prep_data['ref_sci_dir']

    ref_data, ref_hdr = load_image(ref_img_path)
    horizon_mask = ~np.isfinite(ref_data) | (ref_data == 0)

    # Also find reference Stokes V for V residuals
    ref_v_data = None
    if ref_sci_dir is None:      # master reference: the previous V master snapshot
        _pv = f'{workdir}/prep/ref_master_prev_V.fits'
        if os.path.exists(_pv):
            _d, _ = load_image(_pv)
            if _d.shape != target_shape:
                regrid_image(_pv, tmpl_hdr, f'{workdir}/prep/ref_V_regrid.fits')
                _d, _ = load_image(f'{workdir}/prep/ref_V_regrid.fits')
            ref_v_data = _d
    ref_v_variants = find_all_dewarped_variants(ref_sci_dir, freq_str) if ref_sci_dir else []
    for vpath, vstokes, vdepth, vkey in ref_v_variants:
        if vstokes == 'V' and vdepth == 'deep' and 'pbcorr' not in vkey:
            d, _ = load_image(vpath)
            if d.shape == target_shape:
                ref_v_data = d
                break
            elif prep_data.get('needs_regrid_ref', ref_date < CUTOVER_DATE):
                rg_path = f'{workdir}/prep/ref_V_regrid.fits'
                regrid_image(vpath, tmpl_hdr, rg_path)
                ref_v_data, _ = load_image(rg_path)
                break

    # Track per-date metrics and aligned paths by variant
    metrics_rows = []
    aligned_by_variant = {}  # variant_key -> [(date, path, weight)]

    for dinfo in prep_data['dates']:
        date = dinfo['date']
        sci_dir = dinfo['sci_dir']
        needs_regrid = dinfo['needs_regrid']
        coeffs = np.array(dinfo['scale_map_coeffs'])
        weight = dinfo['weight']

        # Load flow field
        scaled_base = f'{date}_scaled'
        flow_path = f'{workdir}/flow/{scaled_base}_flow.npy'
        flow_offsets = _require_flow(flow_path, date, ref_date, f'{lst} {fl}')

        # Rebuild scale map
        h, w = target_shape
        yg, xg = np.indices((h, w), dtype=float)
        xgn = 2.0 * xg / w - 1.0
        ygn = 2.0 * yg / h - 1.0
        scale_map = coeffs[0] + coeffs[1] * xgn + coeffs[2] * ygn

        # Precompute regrid mapping for pre-cutover dates (once per date)
        regrid_coords = None
        if needs_regrid:
            all_variants = find_all_dewarped_variants(sci_dir, freq_str)
            if all_variants:
                t_rg = time.time()
                regrid_coords = precompute_regrid_mapping(
                    all_variants[0][0], tmpl_hdr, target_shape)
                print(f"  {date}: regrid mapping computed in {time.time()-t_rg:.1f}s")
        else:
            all_variants = find_all_dewarped_variants(sci_dir, freq_str)

        n_ok = 0
        n_fail = 0

        for src_path, stokes, depth, vkey in all_variants:
            out_path = f'{workdir}/aligned/{date}/{os.path.basename(src_path)}'
            ok = process_variant_image(
                src_path, flow_offsets, scale_map, tmpl_hdr,
                target_shape, out_path, regrid_coords)
            if ok:
                n_ok += 1
                if vkey not in aligned_by_variant:
                    aligned_by_variant[vkey] = []
                aligned_by_variant[vkey].append((date, out_path, weight))
            else:
                n_fail += 1

        # Also process reference variants (first time only)
        ref_aligned_dir = f'{workdir}/aligned/{ref_date}'
        if not os.path.exists(ref_aligned_dir):
            os.makedirs(ref_aligned_dir, exist_ok=True)
            ref_variants = find_all_dewarped_variants(ref_sci_dir, freq_str)
            ref_regrid_coords = None
            if ref_date < CUTOVER_DATE:
                if ref_variants:
                    ref_regrid_coords = precompute_regrid_mapping(
                        ref_variants[0][0], tmpl_hdr, target_shape)
            for src_path, stokes, depth, vkey in ref_variants:
                out_path = f'{ref_aligned_dir}/{os.path.basename(src_path)}'
                identity_scale = np.ones(target_shape)
                ok = process_variant_image(
                    src_path, None, identity_scale, tmpl_hdr,
                    target_shape, out_path, ref_regrid_coords)
                if ok:
                    if vkey not in aligned_by_variant:
                        aligned_by_variant[vkey] = []
                    ref_rms_val = get_uniform_rms(ref_data, horizon_mask)
                    ref_weight = 1.0 / ref_rms_val**2 if ref_rms_val > 0 else 1.0
                    aligned_by_variant[vkey].insert(0, (ref_date, out_path, ref_weight))

        # Compute I residual (using the flow-aligned alignment image)
        align_dewarp_path = f'{workdir}/flow/{scaled_base}_dewarp.fits'
        if os.path.exists(align_dewarp_path):
            aligned_data, _ = load_image(align_dewarp_path)
        else:
            # Fall back to scaled (no flow)
            aligned_data, _ = load_image(f'{workdir}/prep/{date}_scaled.fits')

        i_rms = compute_residual_rms(aligned_data, ref_data, horizon_mask)

        # Compute V residual
        v_rms = np.nan
        if ref_v_data is not None:
            v_aligned_path = None
            for src_path, stokes, depth, vkey in all_variants:
                if stokes == 'V' and depth == 'deep' and 'pbcorr' not in vkey:
                    v_aligned_path = f'{workdir}/aligned/{date}/{os.path.basename(src_path)}'
                    break
            if v_aligned_path and os.path.exists(v_aligned_path):
                v_data, _ = load_image(v_aligned_path)
                v_rms = compute_residual_rms(v_data, ref_v_data, horizon_mask)

        metrics_rows.append({
            'date': date, 'lst': lst, 'freq': fl,
            'i_rms_mjy': i_rms * 1000 if np.isfinite(i_rms) else np.nan,
            'v_rms_mjy': v_rms * 1000 if np.isfinite(v_rms) else np.nan,
            'scale_a': coeffs[0], 'weight': weight,
            'n_variants': n_ok, 'n_failed': n_fail,
            'has_flow': flow_offsets is not None,
        })

        print(f"  {date}: I_rms={i_rms*1000:.1f} mJy  "
              f"V_rms={v_rms*1000:.1f} mJy  "
              f"{n_ok} variants OK")

    if not metrics_rows:
        return

    # Save metrics
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(f'{workdir}/residuals.csv', index=False)

    # ── Outlier detection ──
    dates_list = [r['date'] for r in metrics_rows]
    i_rms_list = [r['i_rms_mjy'] for r in metrics_rows]
    v_rms_list = [r['v_rms_mjy'] for r in metrics_rows]

    i_outliers, i_thresh = detect_outliers_mad(dates_list, i_rms_list)
    v_outliers, v_thresh = detect_outliers_mad(dates_list, v_rms_list)
    # ── CO-ADD BEAM GATE (Gregg approved 2026-09-02; master-side shipped
    # 2026-09-03 under R-ROUND2-RATIFIED): an epoch whose Phase-2 survey
    # Stokes I (Taper R-0.75) fitted-beam AREA exceeds 1.5x the median of
    # the candidate epoch set was imaged from degraded uv coverage (long-
    # baseline dropout) and would smear the master's PSF. One-directional:
    # sharp epochs are fine in a stack. Keyed on I ("I is king"); refusal
    # is logged with the other outlier rows. Fails OPEN per epoch when no
    # survey product/beam header is found (a missing header is a Phase-2
    # gap, not evidence of a bad beam).
    beam_outliers, _bareas = coadd_beam_gate(lst, fl, dates_list)
    all_outliers = sorted(set(i_outliers) | set(v_outliers)
                          | set(beam_outliers))

    outlier_rows = []
    for d in all_outliers:
        reasons = []
        if d in i_outliers:
            reasons.append('I_rms')
        if d in v_outliers:
            reasons.append('V_rms')
        if d in beam_outliers:
            reasons.append('beam')
        outlier_rows.append({'date': d, 'reason': '+'.join(reasons)})

    outlier_df = pd.DataFrame(outlier_rows)
    outlier_df.to_csv(f'{workdir}/outliers.csv', index=False)

    print(f"\n  Outlier detection (MAD, {OUTLIER_SIGMA}σ):")
    print(f"    I threshold: {i_thresh:.1f} mJy  → {len(i_outliers)} outliers: {i_outliers}")
    print(f"    V threshold: {v_thresh:.1f} mJy  → {len(v_outliers)} outliers: {v_outliers}")

    # ── Co-addition ──
    good_dates = set(dates_list) - set(all_outliers)
    good_dates.add(ref_date)  # always include reference

    print(f"\n  Co-adding {len(good_dates)} dates "
          f"(excluded {len(all_outliers)} outliers + prior exclusions)")

    coadd_dir = f'{workdir}/coadd'
    os.makedirs(coadd_dir, exist_ok=True)

    n_coadds = 0
    for vkey, date_paths in sorted(aligned_by_variant.items()):
        # Only co-add deep variants (skip individual 10-min intervals for now)
        if 'deep' not in vkey:
            continue

        good = [(d, p, w) for d, p, w in date_paths if d in good_dates]
        if len(good) < MIN_DATES_FOR_COADD:
            continue

        paths = [p for _, p, _ in good]
        weights = [w for _, _, w in good]
        out_path = f'{coadd_dir}/{vkey}_{fl}.fits'

        coadd_inverse_variance(paths, weights, out_path, tmpl_hdr)
        n_coadds += 1

    print(f"  Created {n_coadds} co-adds in {coadd_dir}/")

    # ── Diff-median movies ──
    movie_dir = f'{workdir}/movies'
    os.makedirs(movie_dir, exist_ok=True)

    # I deep Taper R075 movie
    i_vkey = 'I_deep_Taper_R075'
    if i_vkey in aligned_by_variant:
        i_good = [(d, p, w) for d, p, w in aligned_by_variant[i_vkey]
                  if d in good_dates]
        if len(i_good) >= 3:
            i_dates = [d for d, _, _ in i_good]
            i_paths = [p for _, p, _ in i_good]
            movie_path = f'{movie_dir}/diff_median_I_{fl}.mp4'
            make_diff_median_movie(i_paths, i_dates, ref_img_path, movie_path,
                                   lst, freq_str, 'I')
            print(f"  Movie: {movie_path}")

    # V deep Taper R0 movie
    v_vkey = 'V_deep_Taper_R0'
    if v_vkey in aligned_by_variant and ref_v_data is not None:
        v_good = [(d, p, w) for d, p, w in aligned_by_variant[v_vkey]
                  if d in good_dates]
        if len(v_good) >= 3:
            v_dates = [d for d, _, _ in v_good]
            v_paths = [p for _, p, _ in v_good]
            movie_path = f'{movie_dir}/diff_median_V_{fl}.mp4'
            make_diff_median_movie(v_paths, v_dates, ref_img_path, movie_path,
                                   lst, freq_str, 'V')
            print(f"  Movie: {movie_path}")

    # ── Cleanup intermediate files to save disk ──
    aligned_dir = f'{workdir}/aligned'
    if os.path.isdir(aligned_dir):
        sz = sum(os.path.getsize(os.path.join(dp, f))
                 for dp, _, fns in os.walk(aligned_dir) for f in fns)
        shutil.rmtree(aligned_dir)
        print(f"  Cleaned up aligned/ ({sz/1e9:.1f} GB)")
    prep_dir = f'{workdir}/prep'
    if os.path.isdir(prep_dir):
        shutil.rmtree(prep_dir)
        print(f"  Cleaned up prep/")
    flow_dir = f'{workdir}/flow'
    if os.path.isdir(flow_dir):
        for fp in glob.glob(f'{flow_dir}/*_dewarp.fits'):
            os.remove(fp)
        print(f"  Cleaned up flow dewarp images (kept .npy flow fields)")


# ═══════════════════════════════════════════════════════════════════
# Provisional stage (two-pass co-add, pass 1)
# ═══════════════════════════════════════════════════════════════════

def run_provisional_lst_freq(lst, freq_str, exclusions):
    """Provisional co-add + residual-based outlier analysis for one (LST, freq).

    Re-derives flux scale, loads existing flow fields (.npy), aligns
    I + V deep images, creates provisional co-add, measures residual
    RMS for outlier detection.  Cleans up temporary files, keeps
    provisional co-adds and metrics CSV.
    """
    ref_date = REFERENCE_DATES.get(lst) or 'MASTER'

    fl = freq_label(freq_str)
    workdir = f'{_workdir_base()}/{lst}_{fl}'
    _ensure_reference_fingerprint(workdir, lst)
    prep_dir = f'{workdir}/prep'
    prov_dir = f'{workdir}/coadd_provisional'

    if not os.path.isdir(f'{workdir}/flow'):
        print(f"  No flow/ directory for {lst} {fl}, skipping")
        return None

    os.makedirs(prep_dir, exist_ok=True)
    os.makedirs(prov_dir, exist_ok=True)

    # ── Discover dates ──
    all_dates = discover_dates(lst, freq_str)
    if not all_dates:
        print(f"  No images for {lst} {fl}")
        return None

    dates = [(d, sd, img) for d, sd, img in all_dates
             if not is_excluded(lst, freq_str, d, exclusions)]

    ref_date_entry = _select_reference(lst, fl, workdir, dates)
    if ref_date_entry is None:
        print(f"  Reference {ref_date} not found for {lst} {fl} and no usable fallback")
        return None

    ref_date = ref_date_entry[0]
    target_dates = [x for x in dates if x[0] != ref_date]
    _gr = _grid_refused_dates(target_dates, f'{lst}_{fl}')
    if _gr:
        with open(f'{workdir}/refusals_grid.csv', 'a') as _rf:
            for _d in sorted(_gr):
                _rf.write(f'{lst},{fl},{_d},wrong grid (Phase-2 anomaly),provisional\n')
        target_dates = [x for x in target_dates if x[0] not in _gr]
        dates = [x for x in dates if x[0] not in _gr]

    if len(target_dates) < MIN_DATES_FOR_COADD - 1:
        print(f"  Only {len(target_dates)+1} dates, need {MIN_DATES_FOR_COADD}, skipping")
        return None

    print(f"\n  {lst} {fl}: {len(target_dates)+1} dates (ref={ref_date})")

    # FLOW FRESHNESS GATE (2026-09-06, recovery chain): refuse the cell when any epoch's
    # alignment flow is missing or older than its input image (a recovered or restacked
    # epoch would be aligned with the flow of its superseded image). Fix = --stage flow.
    _stale = _flow_freshness(workdir, target_dates)
    if _stale and os.environ.get('P3_ALLOW_STALE_FLOW') != '1':
        print(f"  REFUSING {lst} {fl}: stale/missing alignment flow for "
              f"{', '.join(f'{d} ({r})' for d, r in _stale)} - run --stage flow first")
        return None

    # ── Template ──
    post_dates = [x for x in dates if x[0] >= CUTOVER_DATE]
    if post_dates:
        tmpl_hdr, target_shape = _modal_grid_template(post_dates, ref_date_entry)
    else:
        tmpl_hdr, target_shape = build_template_header(ref_date_entry[2])

    # ── Reference I ──
    ref_img_path = ref_date_entry[2]
    ref_needs_regrid = ref_date < CUTOVER_DATE and bool(post_dates)
    if ref_needs_regrid:
        ref_img_path = regrid_image(ref_img_path, tmpl_hdr,
                                     f'{prep_dir}/ref_{ref_date}_regrid.fits')

    if not _reference_astrom_ok(ref_img_path, f'{lst} {fl} ref {ref_date}',
                                freq_label=fl):
        print(f"  {lst} {fl}: REFERENCE {ref_date} REJECTED by the astrometry "
              f"gate — fix its dewarp or change REFERENCE_DATES; NOT building")
        return None

    ref_data, ref_hdr = load_image(ref_img_path)
    if ref_date == 'MASTER' and ref_data.shape != target_shape:
        os.makedirs(f'{workdir}/prep', exist_ok=True)
        ref_img_path = regrid_image(ref_img_path, tmpl_hdr, f'{workdir}/prep/ref_master_prev_regrid.fits')
        ref_data, ref_hdr = load_image(ref_img_path)
        print(f"  [reference] {lst} {fl}: previous master regridded to the template grid")
    ref_wcs = WCS(ref_hdr).celestial
    cd = abs(ref_hdr.get('CDELT1', ref_hdr.get('CD1_1', 0.037)))
    bmaj = ref_hdr.get('BMAJ', 5 * cd)
    beam_pix = bmaj / cd
    horizon_mask = ~np.isfinite(ref_data) | (ref_data == 0)
    ref_rms = get_uniform_rms(ref_data, horizon_mask)
    ref_weight = 1.0 / ref_rms**2 if ref_rms > 0 else 1.0

    ref_i_path = f'{prep_dir}/{ref_date}_aligned_I.fits'
    if not os.path.exists(ref_i_path):
        save_image(ref_i_path, ref_data, tmpl_hdr)

    # ── Reference V ──
    ref_v_src = find_stokes_v_deep_image(ref_date_entry[1], freq_str) if ref_date_entry[1] else None
    ref_v_path = None
    if ref_v_src:
        ref_v_path = f'{prep_dir}/{ref_date}_aligned_V.fits'
        if not os.path.exists(ref_v_path):
            identity_scale = np.ones(target_shape)
            rg_coords = None
            if ref_needs_regrid:
                rg_coords = precompute_regrid_mapping(
                    ref_v_src, tmpl_hdr, target_shape)
            process_variant_image(ref_v_src, None, identity_scale, tmpl_hdr,
                                  target_shape, ref_v_path, rg_coords)

    i_aligned = [] if ref_date == 'MASTER' else [(ref_date, ref_i_path, ref_weight)]
    v_aligned = [(ref_date, ref_v_path, ref_weight)] if (ref_v_path and ref_date != 'MASTER') else []

    # ── Target dates ──
    for date, sci_dir, align_img_path in target_dates:
        t0 = time.time()
        needs_regrid = date < CUTOVER_DATE and bool(post_dates)

        if needs_regrid:
            work_path = regrid_image(align_img_path, tmpl_hdr,
                                      f'{prep_dir}/{date}_regrid.fits')
        else:
            work_path = align_img_path

        tgt_data, tgt_hdr = load_image(work_path)
        if tgt_data.shape != target_shape:
            print(f"    {date}: shape mismatch, skipping")
            continue

        tgt_mask = ~np.isfinite(tgt_data) | (tgt_data == 0)
        if tgt_mask.shape != horizon_mask.shape:
            # wrong-grid epoch (e.g. 19h Full 2025-04-05 raw-failsafe built
            # on the Green grid — Phase-2 anomaly, problem ledger) — skip
            print(f"    {date}: WRONG GRID {tgt_mask.shape} vs "
                  f"{horizon_mask.shape}, skipping", flush=True)
            continue
        tgt_obs_time = get_midpoint_time(tgt_hdr, work_path)
        tgt_rms = get_uniform_rms(tgt_data, tgt_mask)

        scale_map, coeffs, n_src = measure_flux_scale(
            tgt_data, ref_data, ref_wcs, beam_pix,
            horizon_mask | tgt_mask, tgt_obs_time)

        mean_scale = float(coeffs[0])
        weight = float(mean_scale**2 / tgt_rms**2) if tgt_rms > 0 else 0.0

        flow_path = f'{workdir}/flow/{date}_scaled_flow.npy'
        flow_offsets = _require_flow(flow_path, date, ref_date, f'{lst} {fl}')

        h, w = target_shape
        yg, xg = np.indices((h, w), dtype=float)
        scale_map_full = (coeffs[0] + coeffs[1] * (2.0 * xg / w - 1.0)
                          + coeffs[2] * (2.0 * yg / h - 1.0))
        # BLANK-FIELD REFUSAL (Gregg 2026-09-02: "QA should refuse blank
        # fields and report them"). Previously a blank epoch satisfied the
        # n_src<8 arm below and was handed a flat DC of 1.0, i.e. ACCEPTED.
        # It then left the co-add only by accident: get_uniform_rms returns
        # 0.0 for an all-zero frame, so the weight line zeroes it - and
        # nothing recorded that the epoch had dropped out. That accident
        # does NOT cover a frame that is finite with no positive pixel: it
        # has a nonzero rms, hence a nonzero weight, and would enter the
        # co-add. Refuse explicitly and say so.
        _npos = int(np.count_nonzero(np.nan_to_num(tgt_data) > 0))
        _finite = float(np.isfinite(tgt_data).mean())
        if _npos == 0 or _finite == 0.0:
            print(f"    {date}: BLANK FIELD REFUSED — {_npos} positive "
                  f"pixels, {100*_finite:.1f}% finite. Not scaled, not "
                  f"weighted, NOT added to the co-add (Gregg 2026-09-02).",
                  flush=True)
            continue
        # Flux-degeneracy guard (Gregg 2026-06-21): a degenerate affine fit
        # (few/clustered sources) ramps 2-8x across the field -> sharp-edge
        # artifact. If out of physical range, drop the tilt, use robust median DC.
        _mean = float(coeffs[0]); _tilt = float(np.hypot(coeffs[1], coeffs[2]))
        # R-TILT-REMOVED-NEXTGEN (Gregg 2026-08-28, landed 2026-09-08): the x/y
        # tilt is REMOVED from production - flat DC always. P3_FLUXSCALE_TILT=1
        # restores the affine plane for comparison runs only, never production.
        _degen = n_src < 8 or not (0.7 <= _mean <= 1.4) or _tilt > 0.3
        if os.environ.get('P3_FLUXSCALE_TILT') != '1' or _degen:
            _dc = _mean if (0.3 < _mean < 3.0) else 1.0
            scale_map_full = np.full((h, w), _dc, dtype=float)
            print(f"    {date}: flat DC={_dc:.3f} (measured tilt {_tilt:.3f} "
                  f"dropped per R-TILT-REMOVED-NEXTGEN; n={n_src}"
                  f"{'; degenerate' if _degen else ''})")

        regrid_coords = None
        if needs_regrid:
            regrid_coords = precompute_regrid_mapping(
                align_img_path, tmpl_hdr, target_shape)

        i_out = f'{prep_dir}/{date}_aligned_I.fits'
        process_variant_image(work_path, flow_offsets, scale_map_full,
                              tmpl_hdr, target_shape, i_out)
        i_aligned.append((date, i_out, weight))

        v_src = find_stokes_v_deep_image(sci_dir, freq_str)
        if v_src:
            v_out = f'{prep_dir}/{date}_aligned_V.fits'
            # 2026-09-06 (11h 2025-04-12 Full): the archived V stack of an epoch can sit on a different
            # pixel grid from its I stack (the I stack was restacked onto the current grid, the V stack
            # kept the pre-restack grid). process_variant_image cannot align it with the I flow and
            # regrid mapping, and the co-add then crashed on the missing aligned_V file. The epoch's V
            # frame is DROPPED loudly (the I frame stays: per-Stokes constituent sets are the ruled
            # structure, R-LOO-PER-STOKES-RATIFIED); never a silent skip, never a crash.
            try:
                _vh = fits.getheader(v_src); _ih = fits.getheader(align_img_path)
                _v_shape = (int(_vh['NAXIS2']), int(_vh['NAXIS1'])); _i_shape = (int(_ih['NAXIS2']), int(_ih['NAXIS1']))
            except Exception as _e:
                _v_shape, _i_shape = None, None
            if _v_shape is not None and _v_shape != _i_shape:
                print(f"    {date}: V frame grid {_v_shape} differs from the I frame grid {_i_shape} - "
                      f"V DROPPED for this epoch (I kept) [{os.path.basename(v_src)}]")
            else:
                process_variant_image(v_src, flow_offsets, scale_map_full,
                                      tmpl_hdr, target_shape, v_out, regrid_coords)
                if os.path.exists(v_out):
                    v_aligned.append((date, v_out, weight))
                else:
                    print(f"    {date}: V frame could not be aligned ({os.path.basename(v_src)}) - V DROPPED for this epoch (I kept)")

        flow_str = 'flow' if flow_offsets is not None else 'no-flow'
        print(f"    {date}: scale={mean_scale:.3f} ({n_src} src)  "
              f"w={weight:.3f}  {flow_str}  {time.time()-t0:.1f}s")

    if len(i_aligned) < MIN_DATES_FOR_COADD:
        print(f"  Too few aligned dates ({len(i_aligned)})")
        shutil.rmtree(prep_dir, ignore_errors=True)
        return None

    # ── Provisional co-add ──
    print(f"  Provisional co-add: {len(i_aligned)} I, {len(v_aligned)} V")

    prov_i_path = f'{prov_dir}/I_deep_Taper_R075_{fl}.fits'
    coadd_inverse_variance(
        [p for _, p, _ in i_aligned],
        [w for _, _, w in i_aligned],
        prov_i_path, tmpl_hdr)

    prov_v_path = None
    if len(v_aligned) >= MIN_DATES_FOR_COADD:
        prov_v_path = f'{prov_dir}/V_deep_Taper_R0_{fl}.fits'
        coadd_inverse_variance(
            [p for _, p, _ in v_aligned],
            [w for _, _, w in v_aligned],
            prov_v_path, tmpl_hdr)

    # ── Residual analysis ──
    prov_i_data, _ = load_image(prov_i_path)
    prov_v_data = None
    if prov_v_path:
        prov_v_data, _ = load_image(prov_v_path)

    metrics = []
    for date, i_path, weight in i_aligned:
        i_data, _ = load_image(i_path)
        i_rms = compute_residual_rms(i_data, prov_i_data, horizon_mask,
                                      margin_frac=0.15)

        v_rms = np.nan
        v_match = [x for x in v_aligned if x[0] == date]
        if v_match and prov_v_data is not None:
            v_data, _ = load_image(v_match[0][1])
            v_rms = compute_residual_rms(v_data, prov_v_data, horizon_mask,
                                          margin_frac=0.25)

        metrics.append({
            'lst': lst, 'freq': fl, 'date': date,
            'i_rms_mjy': float(i_rms * 1000) if np.isfinite(i_rms) else np.nan,
            'v_rms_mjy': float(v_rms * 1000) if np.isfinite(v_rms) else np.nan,
            'weight': weight,
            'has_flow': os.path.exists(f'{workdir}/flow/{date}_scaled_flow.npy'),
            'is_reference': date == ref_date,
        })

    # ── Outlier detection ──
    dates_list = [m['date'] for m in metrics]
    i_rms_list = [m['i_rms_mjy'] for m in metrics]
    v_rms_list = [m['v_rms_mjy'] for m in metrics]

    i_outliers, i_thresh = detect_outliers_mad(dates_list, i_rms_list)
    v_outliers, v_thresh = detect_outliers_mad(dates_list, v_rms_list)
    beam_outliers, _ = coadd_beam_gate(lst, fl, dates_list)   # 2026-09-05: subband path
    all_outliers = set(i_outliers) | set(v_outliers) | set(beam_outliers)

    for m in metrics:
        m['i_outlier'] = m['date'] in set(i_outliers)
        m['v_outlier'] = m['date'] in set(v_outliers)
        m['beam_outlier'] = m['date'] in set(beam_outliers)
        m['outlier'] = m['date'] in all_outliers
        m['i_thresh_mjy'] = i_thresh
        m['v_thresh_mjy'] = v_thresh

    n_out = len(all_outliers)
    print(f"  Outliers (MAD {OUTLIER_SIGMA}σ): "
          f"I>{i_thresh:.1f}mJy({len(i_outliers)})  "
          f"V>{v_thresh:.1f}mJy({len(v_outliers)})  "
          f"→ {n_out} excluded, {len(i_aligned)-n_out} good")

    pd.DataFrame(metrics).to_csv(
        f'{workdir}/residuals_provisional.csv', index=False)

    # ── Cleanup temporary files (keep provisional co-adds) ──
    shutil.rmtree(prep_dir, ignore_errors=True)

    return metrics


def run_provisional_10min_lst_freq(lst, freq_str, exclusions):
    """Provisional co-add + residual analysis for 10-min intervals.

    Uses deep alignment images to compute flux scale and loads existing
    flow fields. Each interval index gets its own provisional co-add
    and residual outlier analysis across dates.
    """
    ref_date = REFERENCE_DATES.get(lst) or 'MASTER'

    fl = freq_label(freq_str)
    workdir = f'{_workdir_base()}/{lst}_{fl}'
    _ensure_reference_fingerprint(workdir, lst)
    prep_dir = f'{workdir}/prep_10min'
    prov_dir = f'{workdir}/coadd_provisional'

    if not os.path.isdir(f'{workdir}/flow'):
        print(f"  No flow/ directory for {lst} {fl}, skipping")
        return None

    os.makedirs(prep_dir, exist_ok=True)
    os.makedirs(prov_dir, exist_ok=True)

    all_dates = discover_dates(lst, freq_str)
    if not all_dates:
        print(f"  No images for {lst} {fl}")
        return None

    dates = [(d, sd, img) for d, sd, img in all_dates
             if not is_excluded(lst, freq_str, d, exclusions)]

    ref_date_entry = _select_reference(lst, fl, workdir, dates)
    if ref_date_entry is None:
        print(f"  Reference {ref_date} not found for {lst} {fl} and no usable fallback")
        return None

    ref_date = ref_date_entry[0]
    target_dates = [x for x in dates if x[0] != ref_date]

    if len(target_dates) < MIN_DATES_FOR_COADD - 1:
        print(f"  Only {len(target_dates)+1} dates, need {MIN_DATES_FOR_COADD}, skipping")
        return None

    post_dates = [x for x in dates if x[0] >= CUTOVER_DATE]
    if post_dates:
        tmpl_hdr, target_shape = _modal_grid_template(post_dates, ref_date_entry)
    else:
        tmpl_hdr, target_shape = build_template_header(ref_date_entry[2])

    ref_img_path = ref_date_entry[2]
    ref_needs_regrid = ref_date < CUTOVER_DATE and bool(post_dates)
    if ref_needs_regrid:
        ref_img_path = regrid_image(ref_img_path, tmpl_hdr,
                                     f'{prep_dir}/ref_{ref_date}_regrid.fits')

    if not _reference_astrom_ok(ref_img_path, f'{lst} {fl} ref {ref_date}',
                                freq_label=fl):
        print(f"  {lst} {fl}: REFERENCE {ref_date} REJECTED by the astrometry "
              f"gate — fix its dewarp or change REFERENCE_DATES; NOT building")
        return None

    ref_data, ref_hdr = load_image(ref_img_path)
    if ref_date == 'MASTER' and ref_data.shape != target_shape:
        os.makedirs(f'{workdir}/prep', exist_ok=True)
        ref_img_path = regrid_image(ref_img_path, tmpl_hdr, f'{workdir}/prep/ref_master_prev_regrid.fits')
        ref_data, ref_hdr = load_image(ref_img_path)
        print(f"  [reference] {lst} {fl}: previous master regridded to the template grid")
    ref_wcs = WCS(ref_hdr).celestial
    cd = abs(ref_hdr.get('CDELT1', ref_hdr.get('CD1_1', 0.037)))
    bmaj = ref_hdr.get('BMAJ', 5 * cd)
    beam_pix = bmaj / cd
    horizon_mask = ~np.isfinite(ref_data) | (ref_data == 0)
    ref_rms = get_uniform_rms(ref_data, horizon_mask)
    ref_weight = 1.0 / ref_rms**2 if ref_rms > 0 else 1.0

    if ref_date == 'MASTER':
        raise RuntimeError(f'{lst} {fl}: the 10-min provisional stage is retired under the previous-master reference; per-bin masters replace it (R-TENMIN-PER-BIN-MASTERS-APPROVED 2026-09-09)')
    ref_i_10min = find_10min_images(ref_date_entry[1], freq_str, 'I')
    if not ref_i_10min:
        print(f"  No 10-min I images for reference {ref_date}, skipping")
        shutil.rmtree(prep_dir, ignore_errors=True)
        return None

    n_intervals = max(idx for _, idx in ref_i_10min) + 1
    print(f"\n  {lst} {fl} 10min: {len(target_dates)+1} dates, "
          f"{n_intervals} intervals (ref={ref_date})")

    # ── Pre-compute flux scale + flow for all dates ──
    date_params = {}
    identity_scale = np.ones(target_shape)
    rg_ref = (precompute_regrid_mapping(ref_date_entry[2], tmpl_hdr, target_shape)
              if ref_needs_regrid else None)
    date_params[ref_date] = (identity_scale, None, ref_weight, rg_ref)

    for date, sci_dir, align_img_path in target_dates:
        needs_regrid = date < CUTOVER_DATE and bool(post_dates)
        if needs_regrid:
            work_path = regrid_image(align_img_path, tmpl_hdr,
                                      f'{prep_dir}/{date}_regrid.fits')
        else:
            work_path = align_img_path

        tgt_data, tgt_hdr = load_image(work_path)
        if tgt_data.shape != target_shape:
            print(f"    {date}: shape mismatch, skipping")
            continue

        tgt_mask = ~np.isfinite(tgt_data) | (tgt_data == 0)
        if tgt_mask.shape != horizon_mask.shape:
            # wrong-grid epoch (e.g. 19h Full 2025-04-05 raw-failsafe built
            # on the Green grid — Phase-2 anomaly, problem ledger) — skip
            print(f"    {date}: WRONG GRID {tgt_mask.shape} vs "
                  f"{horizon_mask.shape}, skipping", flush=True)
            continue
        tgt_obs_time = get_midpoint_time(tgt_hdr, work_path)
        tgt_rms = get_uniform_rms(tgt_data, tgt_mask)

        scale_map, coeffs, n_src = measure_flux_scale(
            tgt_data, ref_data, ref_wcs, beam_pix,
            horizon_mask | tgt_mask, tgt_obs_time)

        mean_scale = float(coeffs[0])
        weight = float(mean_scale**2 / tgt_rms**2) if tgt_rms > 0 else 0.0

        flow_path = f'{workdir}/flow/{date}_scaled_flow.npy'
        flow_offsets = _require_flow(flow_path, date, ref_date, f'{lst} {fl}')

        h, w = target_shape
        yg, xg = np.indices((h, w), dtype=float)
        scale_map_full = (coeffs[0] + coeffs[1] * (2.0 * xg / w - 1.0)
                          + coeffs[2] * (2.0 * yg / h - 1.0))
        # BLANK-FIELD REFUSAL (Gregg 2026-09-02: "QA should refuse blank
        # fields and report them"). Previously a blank epoch satisfied the
        # n_src<8 arm below and was handed a flat DC of 1.0, i.e. ACCEPTED.
        # It then left the co-add only by accident: get_uniform_rms returns
        # 0.0 for an all-zero frame, so the weight line zeroes it - and
        # nothing recorded that the epoch had dropped out. That accident
        # does NOT cover a frame that is finite with no positive pixel: it
        # has a nonzero rms, hence a nonzero weight, and would enter the
        # co-add. Refuse explicitly and say so.
        _npos = int(np.count_nonzero(np.nan_to_num(tgt_data) > 0))
        _finite = float(np.isfinite(tgt_data).mean())
        if _npos == 0 or _finite == 0.0:
            print(f"    {date}: BLANK FIELD REFUSED — {_npos} positive "
                  f"pixels, {100*_finite:.1f}% finite. Not scaled, not "
                  f"weighted, NOT added to the co-add (Gregg 2026-09-02).",
                  flush=True)
            continue
        # Flux-degeneracy guard (Gregg 2026-06-21): a degenerate affine fit
        # (few/clustered sources) ramps 2-8x across the field -> sharp-edge
        # artifact. If out of physical range, drop the tilt, use robust median DC.
        _mean = float(coeffs[0]); _tilt = float(np.hypot(coeffs[1], coeffs[2]))
        # R-TILT-REMOVED-NEXTGEN (Gregg 2026-08-28, landed 2026-09-08): the x/y
        # tilt is REMOVED from production - flat DC always. P3_FLUXSCALE_TILT=1
        # restores the affine plane for comparison runs only, never production.
        _degen = n_src < 8 or not (0.7 <= _mean <= 1.4) or _tilt > 0.3
        if os.environ.get('P3_FLUXSCALE_TILT') != '1' or _degen:
            _dc = _mean if (0.3 < _mean < 3.0) else 1.0
            scale_map_full = np.full((h, w), _dc, dtype=float)
            print(f"    {date}: flat DC={_dc:.3f} (measured tilt {_tilt:.3f} "
                  f"dropped per R-TILT-REMOVED-NEXTGEN; n={n_src}"
                  f"{'; degenerate' if _degen else ''})")

        regrid_coords = (precompute_regrid_mapping(align_img_path, tmpl_hdr, target_shape)
                         if needs_regrid else None)

        date_params[date] = (scale_map_full, flow_offsets, weight, regrid_coords)
        flow_str = 'flow' if flow_offsets is not None else 'no-flow'
        print(f"    {date}: scale={mean_scale:.3f} ({n_src} src) "
              f"w={weight:.3f} {flow_str}")

    # ── Pre-discover all 10-min images ──
    all_dates_with_sci = ([(ref_date, ref_date_entry[1])]
                          + [(d, sd) for d, sd, _ in target_dates])
    date_10min = {}
    for date, sci_dir in all_dates_with_sci:
        if date not in date_params:
            continue
        date_10min[date] = {
            'I': find_10min_images(sci_dir, freq_str, 'I'),
            'V': find_10min_images(sci_dir, freq_str, 'V'),
        }

    # bin5 fix (Gregg 2026-08-11 round 2): enumerate intervals over ALL
    # dates, not the reference alone — a 5-interval reference silently
    # dropped int05 for every cell (10 of 11 bin5 gaps traced here).
    # Dates lacking an index are skipped per-interval below, so widening
    # the range is safe.
    all_idx = [idx for dm in date_10min.values()
               for st in ('I', 'V') for _, idx in dm[st]]
    if all_idx and max(all_idx) + 1 > n_intervals:
        print(f"  [bin5-fix] widening n_intervals {n_intervals} -> "
              f"{max(all_idx) + 1} (reference had fewer intervals than "
              f"the archive)")
        n_intervals = max(all_idx) + 1

    # ── Process each interval ──
    all_metrics = []

    for int_idx in range(n_intervals):
        int_label = f'int{int_idx:02d}'
        i_aligned = []
        v_aligned = []

        for date, sci_dir in all_dates_with_sci:
            if date not in date_10min:
                continue
            scale_map_full, flow_offsets, weight, regrid_coords = date_params[date]

            i_matches = [p for p, idx in date_10min[date]['I'] if idx == int_idx]
            if i_matches:
                i_out = f'{prep_dir}/{date}_{int_label}_I.fits'
                ok = process_variant_image(i_matches[0], flow_offsets, scale_map_full,
                                           tmpl_hdr, target_shape, i_out, regrid_coords)
                if ok and os.path.exists(i_out):
                    i_aligned.append((date, i_out, weight))

            v_matches = [p for p, idx in date_10min[date]['V'] if idx == int_idx]
            if v_matches:
                v_out = f'{prep_dir}/{date}_{int_label}_V.fits'
                ok = process_variant_image(v_matches[0], flow_offsets, scale_map_full,
                                           tmpl_hdr, target_shape, v_out, regrid_coords)
                if ok and os.path.exists(v_out):
                    v_aligned.append((date, v_out, weight))

        if len(i_aligned) < MIN_DATES_FOR_COADD:
            print(f"    Interval {int_idx}: too few dates ({len(i_aligned)})")
            continue

        prov_i_path = f'{prov_dir}/I_10min_{int_label}_{fl}.fits'
        coadd_inverse_variance(
            [p for _, p, _ in i_aligned],
            [w for _, _, w in i_aligned],
            prov_i_path, tmpl_hdr)

        prov_v_path = None
        if len(v_aligned) >= MIN_DATES_FOR_COADD:
            prov_v_path = f'{prov_dir}/V_10min_{int_label}_{fl}.fits'
            coadd_inverse_variance(
                [p for _, p, _ in v_aligned],
                [w for _, _, w in v_aligned],
                prov_v_path, tmpl_hdr)

        prov_i_data, _ = load_image(prov_i_path)
        prov_v_data = None
        if prov_v_path:
            prov_v_data, _ = load_image(prov_v_path)

        for date, i_path, weight in i_aligned:
            i_data, _ = load_image(i_path)
            i_rms = compute_residual_rms(i_data, prov_i_data, horizon_mask,
                                          margin_frac=0.15)

            v_rms = np.nan
            v_match = [x for x in v_aligned if x[0] == date]
            if v_match and prov_v_data is not None:
                v_data, _ = load_image(v_match[0][1])
                v_rms = compute_residual_rms(v_data, prov_v_data, horizon_mask,
                                              margin_frac=0.25)

            all_metrics.append({
                'lst': lst, 'freq': fl, 'date': date,
                'interval': int_idx,
                'i_rms_mjy': float(i_rms * 1000) if np.isfinite(i_rms) else np.nan,
                'v_rms_mjy': float(v_rms * 1000) if np.isfinite(v_rms) else np.nan,
                'weight': weight,
                'has_flow': os.path.exists(f'{workdir}/flow/{date}_scaled_flow.npy'),
                'is_reference': date == ref_date,
            })

        for _, p, _ in i_aligned:
            if os.path.exists(p):
                os.remove(p)
        for _, p, _ in v_aligned:
            if os.path.exists(p):
                os.remove(p)

        print(f"    Interval {int_idx}: {len(i_aligned)} I, {len(v_aligned)} V")

    if not all_metrics:
        shutil.rmtree(prep_dir, ignore_errors=True)
        return None

    # ── Outlier detection per interval ──
    for int_idx in range(n_intervals):
        int_metrics = [m for m in all_metrics if m['interval'] == int_idx]
        if not int_metrics:
            continue

        dates_list = [m['date'] for m in int_metrics]
        i_rms_list = [m['i_rms_mjy'] for m in int_metrics]
        v_rms_list = [m['v_rms_mjy'] for m in int_metrics]

        i_outliers, i_thresh = detect_outliers_mad(dates_list, i_rms_list)
        v_outliers, v_thresh = detect_outliers_mad(dates_list, v_rms_list)
        beam_outliers, _ = coadd_beam_gate(lst, fl, dates_list)   # 2026-09-05: 10-min path
        combined = set(i_outliers) | set(v_outliers) | set(beam_outliers)

        for m in int_metrics:
            m['i_outlier'] = m['date'] in set(i_outliers)
            m['v_outlier'] = m['date'] in set(v_outliers)
            m['outlier'] = m['date'] in combined
            m['i_thresh_mjy'] = i_thresh
            m['v_thresh_mjy'] = v_thresh

        n_out = len(combined)
        print(f"  Outliers int{int_idx}: I>{i_thresh:.1f}({len(i_outliers)})  "
              f"V>{v_thresh:.1f}({len(v_outliers)})  → {n_out} excluded")

    pd.DataFrame(all_metrics).to_csv(
        f'{workdir}/residuals_provisional_10min.csv', index=False)

    shutil.rmtree(prep_dir, ignore_errors=True)
    return all_metrics


def build_striation_model(data, wcs, mask, ra_window_beams=20, beam_pix=7.0):
    """Running median along RA within Dec strips to model striations.
    Ported from search_v_transients_v2.py."""
    h, w = data.shape
    y_idx, x_idx = np.indices((h, w))
    ra_map = np.full((h, w), np.nan, dtype=np.float64)
    dec_map = np.full((h, w), np.nan, dtype=np.float64)
    for y0 in range(0, h, 512):
        ye = min(y0 + 512, h)
        try:
            r, d = wcs.all_pix2world(x_idx[y0:ye], y_idx[y0:ye], 0)
            ra_map[y0:ye] = r
            dec_map[y0:ye] = d
        except Exception:
            pass
    valid = ~mask & np.isfinite(dec_map) & np.isfinite(ra_map)
    if np.sum(valid) == 0:
        return np.zeros_like(data), 0
    dmin, dmax = np.nanmin(dec_map[valid]), np.nanmax(dec_map[valid])
    cd = abs(wcs.wcs.cdelt[1]) if hasattr(wcs.wcs, 'cdelt') else 0.03125
    de = np.arange(dmin, dmax + cd, cd)
    nb = len(de) - 1
    dbi = np.clip(np.digitize(dec_map, de) - 1, 0, nb - 1)
    wp = int(ra_window_beams * beam_pix)
    if wp % 2 == 0:
        wp += 1
    hw = wp // 2
    model = np.zeros_like(data)
    na = 0
    for b in range(nb):
        ib = (dbi == b) & valid
        n = np.sum(ib)
        if n < 20:
            continue
        na += 1
        py, px = np.where(ib)
        vals = data[py, px]
        ras = ra_map[py, px]
        o = np.argsort(ras)
        py, px, vals = py[o], px[o], vals[o]
        ns = len(vals)
        if ns < wp:
            model[py, px] = float(np.median(vals))
        else:
            # centered rolling median, min_periods=1 reproduces the original
            # shrinking-window edges (vals[max(0,i-hw):min(ns,i+hw+1)]) exactly
            # but vectorised in C — ~100x faster than the per-pixel loop.
            import pandas as pd
            mv = pd.Series(vals).rolling(wp, center=True,
                                         min_periods=1).median().to_numpy()
            model[py, px] = mv
    return model, nb


def get_outlier_dates(workdir, depth='deep'):
    """Read outlier dates from provisional CSV. Returns set of outlier dates."""
    if depth == 'deep':
        csv_path = f'{workdir}/residuals_provisional.csv'
    else:
        csv_path = f'{workdir}/residuals_provisional_10min.csv'
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    return set(df[df['outlier']]['date'].unique())


def get_good_dates_for_interval(workdir, interval=None):
    """Get dates that are good for a specific 10-min interval.
    If interval is None, returns dates good in the deep analysis."""
    if interval is None:
        csv_path = f'{workdir}/residuals_provisional.csv'
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        return set(df[~df['outlier']]['date'].unique())

    csv_path = f'{workdir}/residuals_provisional_10min.csv'
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    sub = df[df['interval'] == interval]
    return set(sub[~sub['outlier']]['date'].unique())


def find_wideband_deep_images(sci_dir, band, stokes, variant):
    """Find a specific wideband deep image variant. Prefer the dewarped product;
    FALL BACK to RAW (flow-align-to-reference dewarps it via process_variant_image)
    so dewarp-failed nights still enter every variant master, not just the
    Taper/R0.75 alignment master.
    variant: 'Taper_R0', 'Taper_R075', 'NoTaper_R0', 'NoTaper_R075'"""
    d = f'{sci_dir}/{band}/{stokes}/deep'
    if not os.path.isdir(d):
        return None
    taper_key = 'NoTaper' if 'NoTaper' in variant else 'Taper'
    robust_key = 'Robust-0.75' if 'R075' in variant else 'Robust-0'

    def matches(f):
        if not f.endswith('.fits') or 'pbcorr' in f:
            return False
        if taper_key == 'NoTaper' and 'NoTaper' not in f:
            return False
        if taper_key == 'Taper' and 'NoTaper' in f:
            return False
        if robust_key == 'Robust-0.75' and 'Robust-0.75' not in f:
            return False
        if robust_key == 'Robust-0' and 'Robust-0.75' in f:
            return False
        return robust_key in f and taper_key in f

    files = sorted(os.listdir(d))
    for f in files:                                  # preferred: dewarped
        if matches(f) and 'dewarped' in f:
            return os.path.join(d, f)
    for f in files:                                  # failsafe: RAW -> flow dewarps
        if matches(f) and 'dewarped' not in f:
            return os.path.join(d, f)
    return None


def _variant_input_divergent(path, max_jy=1e5, min_finite=0.99):
    """PER-VARIANT DIVERGENCE GATE (Gregg 2026-09-06/07, R-NONSURVEY-VARIANT-RECOVERY-CAMPAIGN-GO:
    'should the co-add refuse a variant whose input epoch is divergent' - yes). The provisional
    outlier stage measures the SURVEY variant only, so a CLEAN-diverged NoTaper/Taper-R0 epoch
    image (|max| ~1e14) entered the variant masters and corrupted 155 of them. Returns a reason
    string when the image is divergent (|max| above max_jy Jy/beam or finite fraction below
    min_finite), else ''. Cheap: one read of the file."""
    try:
        d = np.asarray(fits.getdata(path)).squeeze()
        fin = np.isfinite(d)
        if not fin.any():
            return 'no finite pixel'
        mx = float(np.max(np.abs(d[fin])))
        ff = float(fin.mean())
        if mx > max_jy:
            return f'|max| {mx:.3g} Jy/beam > {max_jy:.0e}'
        if ff < min_finite:
            return f'finite fraction {ff:.3f} < {min_finite}'
        return ''
    except Exception as e:
        return f'unreadable ({e})'


def _stamp_variant_gate(path, refused):
    """HISTORY provenance of the per-variant divergence gate on a variant master."""
    if not refused:
        return
    with fits.open(path, mode='update') as hd:
        for d, why in refused:
            hd[0].header.add_history(f'VARIANT-GATE refused {d}: divergent input ({why})')
        hd[0].header['NVARREF'] = (len(refused), 'epochs refused by the per-variant divergence gate')
        hd.flush()


# ─────────────────────────────────────────────────────────────────────
# CONFORMANT-FROM-BIRTH MASTERS + ARCHIVE WRITE (Gregg 2026-09-04 "that includes
# headers"; Gregg 2026-09-09 "writing these files straight to Lustre"; contract
# steps 24 and 51). Every master, variant, pbcorr sibling and LOO master gets the
# full Phase-3 header standard and its grammar name (PRODNAME) at write time, and
# is written into the ratified archive tree under the generation's Run directory:
#   Coadd/{RUN}/{LST}/{band}/<PRODNAME>.fits              central masters, PB map, weights.csv
#   {LST}/{date}/{RUN}/{band}/{I|V}/deep/<PRODNAME>.fits   LOO masters + aligned epoch images
# P3_ARCHIVE_RUN names the generation (e.g. Run_20260909_000000); P3_ARCHIVE_ROOT
# defaults to the ratified root. The working root keeps its copy as the consumers'
# cache. P3_ARCHIVE_WRITE=0 is the ONLY way to build without archiving (a
# deliberate scratch build, never production).
# ─────────────────────────────────────────────────────────────────────
def _archive_root():
    return os.environ.get('P3_ARCHIVE_ROOT', '/lustre/pipeline/exopipe/phase3')   # read at CALL time
_MASTER_RE = re.compile(r'^(?P<st>[IV])_deep_(?P<taper>Taper|NoTaper)_(?P<rob>R075|R0)(?P<clean>_clean)?(?P<pb>_pbcorr)?(?P<sh>_shflux)?(?P<md>_mdewarp)?_(?P<fl>[A-Za-z0-9]+)\.fits$')
_LOO_RE = re.compile(r'^(?P<date>\d{4}-\d{2}-\d{2})_(?P<st>[IV])_loo_(?P<taper>Taper|NoTaper)_(?P<rob>R075|R0)(?P<sh>_shflux)?(?P<md>_mdewarp)?_(?P<fl>[A-Za-z0-9]+)\.fits$')
_EPOCH_RE = re.compile(r'^(?P<date>\d{4}-\d{2}-\d{2})_(?P<st>[IV])_(?P<kind>clean|sub)(?P<sh>_shflux)?(?P<md>_mdewarp)?_(?P<fl>[A-Za-z0-9]+)\.fits$')


def _constituent_dates(hdr):
    return sorted(str(x).split()[-1] for x in hdr.get('HISTORY', []) if str(x).startswith('COADD constituent'))


def _stamp_master_standard(path, lst, fl, thermnoi=None):
    """Full header standard + PRODNAME on one master/variant/pbcorr/LOO file (validate=True:
    a file that cannot carry the standard is a write failure, not a warning)."""
    from exopipe.common.phase3_products import build_name, write_standard_header
    bn = os.path.basename(path)
    m = _MASTER_RE.match(bn); l = _LOO_RE.match(bn)
    if not (m or l):
        return None
    g = (m or l).groupdict()
    st, taper = g['st'], g['taper']
    robust = -0.75 if g['rob'] == 'R075' else 0.0
    hdr = fits.getheader(path)
    proc = ['dewarped', 'aligned', 'coadd']
    if m and g.get('clean'):
        proc.append('vclean')
    if m and g.get('pb'):
        proc.append('pbcorr')
    if l:
        proc.append(f"loo-{g['date'].replace('-', '')}")
    if g.get('sh') or hdr.get('SHSCALE') is not None:
        proc.append('shflux')          # absolute flux scale baked in (token from the name or the SHSCALE card)
    if g.get('md') or hdr.get('MDWARP') is not None:
        proc.append('mdewarp')         # final master dewarp applied (MDWARP card)
    cons = _constituent_dates(hdr)
    d0 = cons[0] if cons else str(hdr.get('DATE-OBS', ''))[:10]
    d1 = cons[-1] if cons else str(hdr.get('DATE-END', ''))[:10]
    n = int(hdr.get('NCOADD') or len(cons) or 0)
    name = build_name(fl, st, 'deep', taper, robust, proc=proc, lst=lst,
                      date_start=d0.replace('-', ''), date_end=d1.replace('-', ''), n_epochs=n)
    data, _ = load_image(path)
    mask = ~np.isfinite(data) | (data == 0)
    rms = get_uniform_rms(data, mask)
    beam = (float(hdr['BMAJ']), float(hdr['BMIN']), float(hdr.get('BPA', 0.0))) if hdr.get('BMAJ') else None
    ref_sig = ''
    try:
        _mk = f'{os.path.dirname(os.path.dirname(path.rstrip("/")))}/flow/.reference_resolved'
        _mk = _mk if os.path.exists(_mk) else f'{os.path.dirname(path)}/../flow/.reference_resolved'
        ref_sig = open(_mk).read().strip() if os.path.exists(_mk) else ''
    except Exception:
        ref_sig = ''
    extra = {'PRODNAME': (name, 'Phase 3 product name (grammar; CONTINUE cards)'),
             'MSTRGEN': (str(hdr.get('COADDOP', ''))[:20], 'master generation of this product')}
    write_standard_header(path, scope=fl, stokes=st,
                          start=hdr.get('DATE-OBS'), end=hdr.get('DATE-END'),
                          inttime_s=hdr.get('INTTIME'), time_basis='coadd-span',
                          beam=beam, thermnoi=thermnoi, imgrms=rms,
                          pbcorr=bool(m and g.get('pb')), dewarped=True, aligned=True,
                          alignref=ref_sig[:68] or None, msub=False, ssub=False,
                          extra=extra, validate=True)
    return name


def _stamp_all_masters(final_dir, lst, fl):
    """Stamp every master/variant/pbcorr file in coadd_final/ and every LOO master."""
    vpath = f'{final_dir}/V_deep_Taper_R0_{fl}.fits'
    thermnoi = None
    if os.path.exists(vpath):
        _d, _ = load_image(vpath)
        thermnoi = get_uniform_rms(_d, ~np.isfinite(_d) | (_d == 0))
    n = 0
    for f in sorted(os.listdir(final_dir)):
        if f.endswith('.fits') and _MASTER_RE.match(f):
            if _stamp_master_standard(f'{final_dir}/{f}', lst, fl, thermnoi):
                n += 1
    loo = f'{final_dir}/loo'
    if os.path.isdir(loo):
        for f in sorted(os.listdir(loo)):
            if _LOO_RE.match(f) and _stamp_master_standard(f'{loo}/{f}', lst, fl, thermnoi):
                n += 1
    print(f"  {lst} {fl}: header standard + PRODNAME stamped on {n} master products")
    return n


def _atomic_copy(src, dst):
    """Archive write = copy to .part, VERIFY, rename. 2026-09-10: a cross-filesystem copy from Lustre came
    back short (28.2 of 39.0 MB) with no error, so a size check is a write gate here: a short copy is
    removed and REFUSED, never renamed into the archive."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst + '.part')
    _ss, _ds = os.path.getsize(src), os.path.getsize(dst + '.part')
    if _ss != _ds:
        os.remove(dst + '.part')
        raise RuntimeError(f'archive copy short: {dst} got {_ds} of {_ss} bytes; REFUSING (source {src})')
    os.replace(dst + '.part', dst)


def _archive_publish_cell(final_dir, sub_dir, lst, fl):
    """Write the cell's products into the ratified archive tree (contract 0.1/0.4)."""
    if os.environ.get('P3_ARCHIVE_WRITE') == '0':
        print(f"  {lst} {fl}: P3_ARCHIVE_WRITE=0 - scratch build, NOT archived (never production)")
        return 0
    run = os.environ.get('P3_ARCHIVE_RUN')
    if not run:
        raise RuntimeError(f"{lst} {fl}: P3_ARCHIVE_RUN is unset - the generation's Run directory "
                           f"must be named (e.g. Run_20260909_000000); REFUSING (contract step 51)")
    from exopipe.common.phase3_products import build_name, write_standard_header
    cdir = f'{_archive_root()}/Coadd/{run}/{lst}/{fl}'
    n = 0
    for f in sorted(os.listdir(final_dir)):
        src = f'{final_dir}/{f}'
        if f.endswith('.fits') and _MASTER_RE.match(f):
            name = fits.getheader(src).get('PRODNAME')
            if not name:
                raise RuntimeError(f"{lst} {fl}: {f} has no PRODNAME - not stamped; REFUSING to archive")
            _atomic_copy(src, f'{cdir}/{name if name.endswith(".fits") else name + ".fits"}'); n += 1
        elif f == f'PB_deep_{fl}.fits' or f == 'weights.csv':
            _atomic_copy(src, f'{cdir}/{f}'); n += 1
    loo = f'{final_dir}/loo'
    if os.path.isdir(loo):
        for f in sorted(os.listdir(loo)):
            l = _LOO_RE.match(f)
            if not l:
                continue
            name = fits.getheader(f'{loo}/{f}').get('PRODNAME')
            if not name:
                raise RuntimeError(f"{lst} {fl}: LOO {f} has no PRODNAME; REFUSING to archive")
            _atomic_copy(f'{loo}/{f}', f"{_archive_root()}/{lst}/{l['date']}/{run}/{fl}/{l['st']}/deep/{name if name.endswith('.fits') else name + '.fits'}"); n += 1
    # aligned epoch images (the '_clean_' product of subtracted/ = flux-scaled, master-aligned epoch).
    # Only epochs in the generation's own record (weights.csv) are archived: a product left in
    # subtracted/ by a previous generation for a night with no current input (02h 2025-01-02,
    # INPUT-PROVENANCE 2026-09-05) is a stale derivative - logged, never archived.
    _wdates = set()
    try:
        _wdates = {r['date'] for r in csv.DictReader(open(f'{final_dir}/weights.csv'))}
    except Exception:
        _wdates = set()
    if os.path.isdir(sub_dir):
        for f in sorted(os.listdir(sub_dir)):
            e = _EPOCH_RE.match(f)
            if not e or e['kind'] != 'clean':
                continue
            if _wdates and e['date'] not in _wdates:
                print(f"  {lst} {fl}: {f} is not in the generation record (weights.csv) - stale epoch product, NOT archived")
                continue
            src = f'{sub_dir}/{f}'
            h = fits.getheader(src)
            st = e['st']; taper = 'Taper'; robust = -0.75 if st == 'I' else 0.0
            name = h.get('PRODNAME')
            if not name:
                if not h.get('DATE-OBS'):
                    raise RuntimeError(f"{lst} {fl}: epoch product {f} has no time block; REFUSING to archive")
                _eproc = (['dewarped', 'aligned'] + (['shflux'] if (e.group('sh') or h.get('SHSCALE') is not None) else [])
                          + (['mdewarp'] if (e.group('md') or h.get('MDWARP') is not None) else []))
                name = build_name(fl, st, 'deep', taper, robust, proc=_eproc, lst=lst,
                                  epoch=str(h['DATE-OBS']))
                beam = (float(h['BMAJ']), float(h['BMIN']), float(h.get('BPA', 0.0))) if h.get('BMAJ') else None
                write_standard_header(src, scope=fl, stokes=st, start=h.get('DATE-OBS'), end=h.get('DATE-END'),
                                      inttime_s=h.get('INTTIME'), time_basis='epoch', beam=beam,
                                      dewarped=True, aligned=True, msub=False,
                                      extra={'PRODNAME': (name, 'Phase 3 product name (grammar; CONTINUE cards)')}, validate=True)
            _atomic_copy(src, f"{_archive_root()}/{lst}/{e['date']}/{run}/{fl}/{st}/deep/{name if name.endswith('.fits') else name + '.fits'}"); n += 1
    print(f"  {lst} {fl}: {n} products written to the archive tree ({_archive_root()}, {run})")
    return n


def rebuild_variants_from_record(lst, freq_str, variants=('Taper_R0', 'NoTaper_R0')):
    """VARIANT-ONLY REBUILD (Gregg 2026-09-11 "go" on the variant-recovery steps 2-4; the 2026-09-06
    campaign approval required a variant-only mode so the survey pair keeps its generation).
    Rebuild the requested non-survey Stokes I variants of one cell FROM THE GENERATION'S OWN RECORDS:
      - constituents, per-night flat-DC factor (qa3_scale) and weight from coadd_final/weights.csv,
      - the night's co-add alignment flow from flow/<date>_scaled_flow.npy (none for a reference-identity
        night),
      - the generation's grid = the survey master's grid (template rebuilt as the final stage does),
      - the per-variant divergence gate of dc6f33e on every input,
      - the cell's absolute flux factor (SHSCALE/SH* cards of the survey master) baked in, as in the
        generation, and the full header standard stamped.
    NO realignment, NO reference selection, NOTHING else written: the survey pair, LOO masters, weights,
    residuals and PB stay untouched. Output: <workdir>/variants_rebuilt/I_deep_<variant>_<fl>.fits.
    Returns {variant: (path, n_constituents, refused)}; a variant with fewer than MIN_DATES_FOR_COADD
    admitted inputs is reported, not written."""
    import csv as _csv
    fl = freq_label(freq_str)
    workdir = f'{_workdir_base()}/{lst}_{fl}'
    final_dir = f'{workdir}/coadd_final'
    out_dir = f'{workdir}/variants_rebuilt'
    prep_dir = f'{workdir}/prep_variants'
    master = f'{final_dir}/I_deep_Taper_R075_{fl}.fits'
    wcsv = f'{final_dir}/weights.csv'
    for _p in (master, wcsv):
        if not os.path.exists(_p):
            raise RuntimeError(f'{lst} {fl}: {_p} missing - the generation record is incomplete; REFUSING')
    mhdr = fits.getheader(master)
    if mhdr.get('COADDOP') != 'ivw-mean-v3' or not mhdr.get('CONSTDIG'):
        raise RuntimeError(f'{lst} {fl}: survey master is not a current-generation master '
                           f'(COADDOP={mhdr.get("COADDOP")!r}); REFUSING')
    rec = list(_csv.DictReader(open(wcsv)))
    members = [r for r in rec if str(r.get('in_master', '')).strip() in ('1', 'True', 'true')]
    if not members:
        raise RuntimeError(f'{lst} {fl}: weights.csv lists no constituent; REFUSING')
    # the generation's grid, rebuilt exactly as the final stage builds it
    all_dates = discover_dates(lst, freq_str)
    if not all_dates:
        raise RuntimeError(f'{lst} {fl}: no Phase-2 images found; REFUSING')
    post_dates = [x for x in all_dates if x[0] >= CUTOVER_DATE]
    if post_dates:
        tmpl_hdr, target_shape = _modal_grid_template(post_dates, all_dates[0])
    else:
        tmpl_hdr, target_shape = build_template_header(all_dates[0][2])
    mshape = tuple(int(mhdr[f'NAXIS{i}']) for i in (2, 1))
    if tuple(target_shape) != mshape:
        raise RuntimeError(f'{lst} {fl}: template grid {target_shape} != survey master grid {mshape}; REFUSING')
    sci_of = {d: sd for d, sd, _ in all_dates}
    os.makedirs(out_dir, exist_ok=True); os.makedirs(prep_dir, exist_ok=True)
    results = {}
    for vk in variants:
        aligned = []; refused = []
        for r in members:
            date = r['date']
            sci_dir = r.get('sci_dir') or sci_of.get(date)
            if not sci_dir or not os.path.isdir(sci_dir):
                refused.append((date, 'no Science dir in the record')); continue
            img = (find_subband_variant_image(sci_dir, freq_str, vk) if is_subband(freq_str)
                   else find_wideband_deep_images(sci_dir, freq_str, 'I', vk))
            if not img:
                refused.append((date, f'no {vk} input')); continue
            why = _variant_input_divergent(img)
            if why:
                print(f"  VARIANT GATE {vk} {date}: input REFUSED - {why} ({os.path.basename(img)})")
                refused.append((date, why)); continue
            if (r.get('qa3_guard') or '') == 'reference-identity':
                flow = None; dc = 1.0
            else:
                fp = f'{workdir}/flow/{date}_scaled_flow.npy'
                if not os.path.exists(fp):
                    raise RuntimeError(f'{lst} {fl} {date}: recorded constituent without its flow at {fp}; REFUSING')
                flow = np.load(fp)
                _mean = float(r['qa3_scale'])
                dc = _mean if (0.3 < _mean < 3.0) else 1.0      # the flat-DC rule of the build
            weight = float(r['weight'])
            needs_regrid = date < CUTOVER_DATE and bool(post_dates)
            rg = precompute_regrid_mapping(img, tmpl_hdr, target_shape) if needs_regrid else None
            out = f'{prep_dir}/{date}_{vk}.fits'
            if os.path.exists(out):
                os.remove(out)
            ok = process_variant_image(img, flow, np.full(target_shape, dc, dtype=float),
                                       tmpl_hdr, target_shape, out, rg)
            if ok and os.path.exists(out):
                aligned.append((date, out, weight))
            else:
                refused.append((date, 'alignment failed'))
        if len(aligned) < MIN_DATES_FOR_COADD:
            print(f"  {lst} {fl} {vk}: only {len(aligned)} admitted inputs (need {MIN_DATES_FOR_COADD}); NOT written")
            results[vk] = (None, len(aligned), refused); continue
        path = f'{out_dir}/I_deep_{vk}_{fl}.fits'
        coadd_inverse_variance([p for _, p, _ in aligned], [w for _, _, w in aligned], path, tmpl_hdr,
                               dates=[d for d, _, _ in aligned], sigclip=False)
        stamp_master_span(path, lst, fl, [d for d, _, _ in aligned])
        _stamp_variant_gate(path, refused)
        # bake in the cell's absolute flux factor exactly as the generation carries it
        if mhdr.get('SHSCALE') is not None:
            with fits.open(path, mode='update') as hh:
                hh[0].data = (np.asarray(hh[0].data, dtype=np.float32) * np.float32(mhdr['SHSCALE']))
                for k in ('SHSCALE', 'SHERR', 'SHNSRC', 'SHSRCS', 'SHMETHOD', 'SHFREQ', 'SHDATE'):
                    if k in mhdr:
                        hh[0].header[k] = (mhdr[k], mhdr.comments[k])
                hh[0].header['HISTORY'] = (f"variant-only rebuild from the generation record (weights.csv, flow/) "
                                           f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}; shflux x {mhdr['SHSCALE']:.5f} "
                                           f"inherited from the survey master (Gregg 2026-09-11 go)")
                hh.flush()
        _stamp_master_standard(path, lst, fl)
        print(f"  {lst} {fl} {vk}: variant rebuilt from the record: {len(aligned)} constituents, "
              f"{len(refused)} refused -> {path}")
        results[vk] = (path, len(aligned), refused)
    shutil.rmtree(prep_dir, ignore_errors=True)
    return results


def run_final_lst_freq(lst, freq_str, exclusions):
    """Final co-add + subtraction + striation removal for one (LST, freq).

    Steps:
      1. Load good dates from provisional outlier analysis
      2. Re-align good dates (flux scale + flow) to reference
      3. Create final co-add from good dates
      4. For each date: new flow alignment (date → co-add), scale, subtract
      5. Striation removal on V subtracted images
    """
    ref_date = REFERENCE_DATES.get(lst) or 'MASTER'

    fl = freq_label(freq_str)
    workdir = f'{_workdir_base()}/{lst}_{fl}'
    _ensure_reference_fingerprint(workdir, lst)
    final_dir = f'{workdir}/coadd_final'
    sub_dir = f'{workdir}/subtracted'
    if os.environ.get('P3_ARCHIVE_WRITE') != '0' and not os.environ.get('P3_ARCHIVE_RUN'):
        raise RuntimeError(f"{lst} {fl}: P3_ARCHIVE_RUN unset (generation Run directory) and P3_ARCHIVE_WRITE!=0 - REFUSING (contract step 51)")

    done_marker = f'{final_dir}/.done'
    if os.path.exists(done_marker):
        # GENERATION-AWARE SKIP (2026-09-01). The bare .done guard is the
        # banned stale-skip-guard class (CLAUDE.md section 3: "a rebuild
        # that silently no-ops is a bug"). It made the master-policy
        # rebuild impossible to run: a 3-cell pilot reported success while
        # leaving all three cells on the superseded sigma-clipped masters,
        # because .done predated the 2026-08-25 policy. The marker now only
        # authorises a skip when the product it guards is ALREADY the
        # current generation; otherwise the cell is rebuilt.
        _cur = None
        for _v in (f'{final_dir}/I_deep_Taper_R075_{fl}.fits',
                   f'{final_dir}/V_deep_Taper_R0_{fl}.fits'):
            if os.path.exists(_v):
                try:
                    _cur = fits.getheader(_v).get('COADDOP')
                except Exception:
                    _cur = None
                break
        if _cur == 'ivw-mean-v3' and os.environ.get('P3_FORCE_REBUILD') != '1':
            print(f"  {lst} {fl}: already complete (COADDOP={_cur}), skipping")
            return {'lst': lst, 'freq': fl, 'status': 'skipped'}
        if os.environ.get('P3_FORCE_REBUILD') == '1':
            print(f"  {lst} {fl}: P3_FORCE_REBUILD=1 — rebuilding despite "
                  f"current COADDOP (beam-gate rebuild wave 2026-09-03)")
        print(f"  {lst} {fl}: .done present but master COADDOP={_cur!r} is "
              f"NOT the current generation — REBUILDING")
        try:
            os.remove(done_marker)
        except OSError:
            pass
    prep_dir = f'{workdir}/prep_final'

    deep_outliers = get_outlier_dates(workdir, 'deep')
    if not deep_outliers and not os.path.exists(f'{workdir}/residuals_provisional.csv'):
        print(f"  No provisional results for {lst} {fl}, skipping")
        return None

    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)
    os.makedirs(prep_dir, exist_ok=True)

    all_dates = discover_dates(lst, freq_str)
    if not all_dates:
        print(f"  No images for {lst} {fl}")
        return None

    dates = [(d, sd, img) for d, sd, img in all_dates
             if not is_excluded(lst, freq_str, d, exclusions)]

    good_dates_deep = get_good_dates_for_interval(workdir, interval=None)
    if good_dates_deep is None:
        good_dates_deep = set(d for d, _, _ in dates) - deep_outliers

    _gr = _grid_refused_dates([x for x in dates], f'{lst}_{fl}')
    if _gr:
        with open(f'{workdir}/refusals_grid.csv', 'a') as _rf:
            for _d in sorted(_gr):
                _rf.write(f'{lst},{fl},{_d},wrong grid (Phase-2 anomaly),final\n')
        dates = [x for x in dates if x[0] not in _gr]
        good_dates_deep = {d for d in good_dates_deep if d not in _gr}
    dates_good = [(d, sd, img) for d, sd, img in dates if d in good_dates_deep]
    dates_bad = [(d, sd, img) for d, sd, img in dates if d not in good_dates_deep]

    ref_date_entry = _select_reference(lst, fl, workdir, dates)
    if ref_date_entry is None:
        print(f"  Reference {ref_date} not found for {lst} {fl} and no usable fallback")
        return None

    ref_date = ref_date_entry[0]

    if len(dates_good) < MIN_DATES_FOR_COADD:
        print(f"  Only {len(dates_good)} good dates, need {MIN_DATES_FOR_COADD}, skipping")
        return None

    print(f"\n  {lst} {fl} FINAL: {len(dates_good)}/{len(dates)} good dates "
          f"({len(dates_bad)} excluded)")

    post_dates = [x for x in dates if x[0] >= CUTOVER_DATE]
    if post_dates:
        tmpl_hdr, target_shape = _modal_grid_template(post_dates, ref_date_entry)
    else:
        tmpl_hdr, target_shape = build_template_header(ref_date_entry[2])

    ref_img_path = ref_date_entry[2]
    ref_needs_regrid = ref_date < CUTOVER_DATE and bool(post_dates)
    if ref_needs_regrid:
        ref_img_path = regrid_image(ref_img_path, tmpl_hdr,
                                     f'{prep_dir}/ref_{ref_date}_regrid.fits')

    if not _reference_astrom_ok(ref_img_path, f'{lst} {fl} ref {ref_date}',
                                freq_label=fl):
        print(f"  {lst} {fl}: REFERENCE {ref_date} REJECTED by the astrometry "
              f"gate — fix its dewarp or change REFERENCE_DATES; NOT building")
        return None

    ref_data, ref_hdr = load_image(ref_img_path)
    if ref_date == 'MASTER' and ref_data.shape != target_shape:
        os.makedirs(f'{workdir}/prep', exist_ok=True)
        ref_img_path = regrid_image(ref_img_path, tmpl_hdr, f'{workdir}/prep/ref_master_prev_regrid.fits')
        ref_data, ref_hdr = load_image(ref_img_path)
        print(f"  [reference] {lst} {fl}: previous master regridded to the template grid")
    ref_wcs = WCS(ref_hdr).celestial
    cd = abs(ref_hdr.get('CDELT1', ref_hdr.get('CD1_1', 0.037)))
    bmaj = ref_hdr.get('BMAJ', 5 * cd)
    beam_pix = bmaj / cd
    horizon_mask = ~np.isfinite(ref_data) | (ref_data == 0)
    ref_rms = get_uniform_rms(ref_data, horizon_mask)
    ref_weight = 1.0 / ref_rms**2 if ref_rms > 0 else 1.0

    # ── Step 1: Align good dates and create final co-add ──
    i_aligned = []
    date_params = {}

    identity_scale = np.ones(target_shape)
    rg_ref = (precompute_regrid_mapping(ref_date_entry[2], tmpl_hdr, target_shape)
              if ref_needs_regrid else None)

    # Reference date (a NIGHT reference is an identity member; a MASTER reference is not
    # a constituent - every night is aligned and scaled to it, Gregg 2026-09-09)
    if ref_date != 'MASTER':
        ref_i_out = f'{prep_dir}/{ref_date}_I.fits'
        process_variant_image(ref_img_path, None, identity_scale,
                              tmpl_hdr, target_shape, ref_i_out, rg_ref)
        if ref_date in good_dates_deep:
            i_aligned.append((ref_date, ref_i_out, ref_weight))
        date_params[ref_date] = {
            'scale_map': identity_scale, 'flow': None, 'weight': ref_weight,
            'regrid': rg_ref, 'sci_dir': ref_date_entry[1],
            'qa3_scale': 1.0, 'qa3_tilt': 0.0, 'qa3_nsrc': -1,
            'qa3_guard': 'reference-identity', 'qa3_rms': ref_rms,
        }

    target_dates = [(d, sd, img) for d, sd, img in dates if d != ref_date]

    for date, sci_dir, align_img_path in target_dates:
        needs_regrid = date < CUTOVER_DATE and bool(post_dates)
        if needs_regrid:
            work_path = regrid_image(align_img_path, tmpl_hdr,
                                      f'{prep_dir}/{date}_regrid.fits')
        else:
            work_path = align_img_path

        tgt_data, tgt_hdr = load_image(work_path)
        if tgt_data.shape != target_shape:
            print(f"    {date}: shape mismatch, skipping")
            continue

        tgt_mask = ~np.isfinite(tgt_data) | (tgt_data == 0)
        if tgt_mask.shape != horizon_mask.shape:
            # wrong-grid epoch (e.g. 19h Full 2025-04-05 raw-failsafe built
            # on the Green grid — Phase-2 anomaly, problem ledger) — skip
            print(f"    {date}: WRONG GRID {tgt_mask.shape} vs "
                  f"{horizon_mask.shape}, skipping", flush=True)
            continue
        tgt_obs_time = get_midpoint_time(tgt_hdr, work_path)
        tgt_rms = get_uniform_rms(tgt_data, tgt_mask)

        scale_map, coeffs, n_src = measure_flux_scale(
            tgt_data, ref_data, ref_wcs, beam_pix,
            horizon_mask | tgt_mask, tgt_obs_time)

        mean_scale = float(coeffs[0])
        weight = float(mean_scale**2 / tgt_rms**2) if tgt_rms > 0 else 0.0

        flow_path = f'{workdir}/flow/{date}_scaled_flow.npy'
        flow_offsets = _require_flow(flow_path, date, ref_date, f'{lst} {fl}')

        h, w = target_shape
        yg, xg = np.indices((h, w), dtype=float)
        scale_map_full = (coeffs[0] + coeffs[1] * (2.0 * xg / w - 1.0)
                          + coeffs[2] * (2.0 * yg / h - 1.0))
        # BLANK-FIELD REFUSAL (Gregg 2026-09-02: "QA should refuse blank
        # fields and report them"). Previously a blank epoch satisfied the
        # n_src<8 arm below and was handed a flat DC of 1.0, i.e. ACCEPTED.
        # It then left the co-add only by accident: get_uniform_rms returns
        # 0.0 for an all-zero frame, so the weight line zeroes it - and
        # nothing recorded that the epoch had dropped out. That accident
        # does NOT cover a frame that is finite with no positive pixel: it
        # has a nonzero rms, hence a nonzero weight, and would enter the
        # co-add. Refuse explicitly and say so.
        _npos = int(np.count_nonzero(np.nan_to_num(tgt_data) > 0))
        _finite = float(np.isfinite(tgt_data).mean())
        if _npos == 0 or _finite == 0.0:
            print(f"    {date}: BLANK FIELD REFUSED — {_npos} positive "
                  f"pixels, {100*_finite:.1f}% finite. Not scaled, not "
                  f"weighted, NOT added to the co-add (Gregg 2026-09-02).",
                  flush=True)
            continue
        # Flux-degeneracy guard (Gregg 2026-06-21): a degenerate affine fit
        # (few/clustered sources) ramps 2-8x across the field -> sharp-edge
        # artifact. If out of physical range, drop the tilt, use robust median DC.
        _mean = float(coeffs[0]); _tilt = float(np.hypot(coeffs[1], coeffs[2]))
        # R-TILT-REMOVED-NEXTGEN (Gregg 2026-08-28, landed 2026-09-08): the x/y
        # tilt is REMOVED from production - flat DC always. P3_FLUXSCALE_TILT=1
        # restores the affine plane for comparison runs only, never production.
        _degen = n_src < 8 or not (0.7 <= _mean <= 1.4) or _tilt > 0.3
        if os.environ.get('P3_FLUXSCALE_TILT') != '1' or _degen:
            _dc = _mean if (0.3 < _mean < 3.0) else 1.0
            scale_map_full = np.full((h, w), _dc, dtype=float)
            print(f"    {date}: flat DC={_dc:.3f} (measured tilt {_tilt:.3f} "
                  f"dropped per R-TILT-REMOVED-NEXTGEN; n={n_src}"
                  f"{'; degenerate' if _degen else ''})")

        regrid_coords = (precompute_regrid_mapping(align_img_path, tmpl_hdr, target_shape)
                         if needs_regrid else None)

        date_params[date] = {
            'scale_map': scale_map_full, 'flow': flow_offsets, 'weight': weight,
            'regrid': regrid_coords, 'sci_dir': sci_dir,
            # third-stage QA record (R-THIRD-STAGE-QA-FLUXALIGN 2026-08-25):
            # recorded per epoch; drops nothing until thresholds are approved
            'qa3_scale': _mean, 'qa3_tilt': _tilt, 'qa3_nsrc': n_src,
            'qa3_guard': ('degenerate-flatDC' if (n_src < 8 or not
                          (0.7 <= _mean <= 1.4) or _tilt > 0.3) else 'pass'),
            'qa3_rms': tgt_rms,
        }

        # Two-tier policy (Gregg 2026-08-26): per-epoch products are built
        # for EVERY non-data_decisions epoch; only co-add membership is
        # gated by the QA verdict.
        i_out = f'{prep_dir}/{date}_I.fits'
        process_variant_image(align_img_path, flow_offsets, scale_map_full,
                              tmpl_hdr, target_shape, i_out, regrid_coords)
        if date in good_dates_deep:
            i_aligned.append((date, i_out, weight))
            flow_str = 'flow' if flow_offsets is not None else 'no-flow'
            print(f"    {date}: scale={mean_scale:.3f} ({n_src} src) "
                  f"w={weight:.3f} {flow_str} ✓")
        else:
            print(f"    {date}: coadd-omitted (QA); products still built")

    if len(i_aligned) < MIN_DATES_FOR_COADD:
        print(f"  Only {len(i_aligned)} aligned, need {MIN_DATES_FOR_COADD}")
        shutil.rmtree(prep_dir, ignore_errors=True)
        return None

    # Final I co-add — PLAIN weighted mean, NO per-pixel clip
    # (R-LOO-MASTERS-AND-REDO-SEQUENCE, Gregg 2026-08-25: the 4-sigma-MAD
    # per-pixel clip introduces beam-scale structure into the masters).
    final_i_path = f'{final_dir}/I_deep_Taper_R075_{fl}.fits'
    coadd_inverse_variance(
        [p for _, p, _ in i_aligned],
        [w for _, _, w in i_aligned],
        final_i_path, tmpl_hdr, dates=[d for d, _, _ in i_aligned],
        sigclip=False)
    stamp_master_span(final_i_path, lst, fl, [d for d, _, _ in i_aligned])
    print(f"  Final I co-add: {len(i_aligned)} dates → {final_i_path}")

    # Also create V co-add
    v_aligned = []
    for date, sci_dir, _ in dates:
        if date not in date_params:
            continue
        params = date_params[date]
        v_img = find_stokes_v_deep_image(sci_dir, freq_str)
        if not v_img:
            continue
        v_out = f'{prep_dir}/{date}_V.fits'
        ok = process_variant_image(v_img, params['flow'], params['scale_map'],
                                    tmpl_hdr, target_shape, v_out, params['regrid'])
        # two-tier policy: V prep for every epoch; co-add membership QA-gated
        if ok and os.path.exists(v_out) and date in good_dates_deep:
            v_aligned.append((date, v_out, params['weight']))

    final_v_path = None
    if len(v_aligned) >= MIN_DATES_FOR_COADD:
        final_v_path = f'{final_dir}/V_deep_Taper_R0_{fl}.fits'
        coadd_inverse_variance(
            [p for _, p, _ in v_aligned],
            [w for _, _, w in v_aligned],
            final_v_path, tmpl_hdr, dates=[d for d, _, _ in v_aligned],
            sigclip=False)
        stamp_master_span(final_v_path, lst, fl, [d for d, _, _ in v_aligned])
        print(f"  Final V co-add: {len(v_aligned)} dates → {final_v_path}")

    # ── weights + third-stage QA record (R-THIRD-STAGE-QA-FLUXALIGN) and
    # LOO masters for thin cells (R-LOO-MASTERS, N_epochs <= 20) ──
    import csv as _csv
    with open(f'{final_dir}/weights.csv', 'w', newline='') as _f:
        _w = _csv.writer(_f)
        _w.writerow(['date', 'weight', 'qa3_scale', 'qa3_tilt', 'qa3_nsrc',
                     'qa3_guard', 'qa3_rms_mjy', 'sci_dir', 'in_master'])
        for _d in sorted(date_params):
            _p = date_params[_d]
            _w.writerow([_d, f"{_p['weight']:.6f}", f"{_p.get('qa3_scale', 1):.5f}",
                         f"{_p.get('qa3_tilt', 0):.5f}", _p.get('qa3_nsrc', ''),
                         _p.get('qa3_guard', ''),
                         f"{1e3 * _p.get('qa3_rms', float('nan')):.2f}",
                         _p.get('sci_dir', ''), int(_d in good_dates_deep)])
    _n_loo = write_loo_masters(i_aligned, final_i_path,
                               f'{final_dir}/loo', f'I_loo_Taper_R075_{fl}.fits')
    if final_v_path:
        _n_loo += write_loo_masters(v_aligned, final_v_path,
                                    f'{final_dir}/loo', f'V_loo_Taper_R0_{fl}.fits')
    if _n_loo:
        print(f"  LOO masters written: {_n_loo} (N_epochs<=20 rule)")

    # Also create I Taper R0 co-add for subtraction
    _variant_refused = {}      # variant -> [(date, reason)] refused by the per-variant divergence gate
    i_r0_aligned = []
    for date, sci_dir, _ in dates:
        if date not in good_dates_deep or date not in date_params:
            continue
        params = date_params[date]
        if is_subband(freq_str):
            i_r0_img = find_subband_variant_image(sci_dir, freq_str, 'Taper_R0')
        else:
            i_r0_img = find_wideband_deep_images(sci_dir, freq_str, 'I', 'Taper_R0')
        if not i_r0_img:
            continue
        _why = _variant_input_divergent(i_r0_img)
        if _why:
            print(f"  VARIANT GATE Taper_R0 {date}: input REFUSED - {_why} ({os.path.basename(i_r0_img)})")
            _variant_refused.setdefault('Taper_R0', []).append((date, _why))
            continue
        i_r0_out = f'{prep_dir}/{date}_I_R0.fits'
        ok = process_variant_image(i_r0_img, params['flow'], params['scale_map'],
                                    tmpl_hdr, target_shape, i_r0_out, params['regrid'])
        if ok and os.path.exists(i_r0_out):
            i_r0_aligned.append((date, i_r0_out, params['weight']))

    if len(i_r0_aligned) >= MIN_DATES_FOR_COADD:
        final_i_r0_path = f'{final_dir}/I_deep_Taper_R0_{fl}.fits'
        coadd_inverse_variance(
            [p for _, p, _ in i_r0_aligned],
            [w for _, _, w in i_r0_aligned],
            final_i_r0_path, tmpl_hdr, dates=[d for d, _, _ in i_r0_aligned],
            sigclip=False)
        stamp_master_span(final_i_r0_path, lst, fl, [d for d, _, _ in i_r0_aligned])
        _stamp_variant_gate(final_i_r0_path, _variant_refused.get('Taper_R0', []))
        print(f"  Final I R0 co-add: {len(i_r0_aligned)} dates")

    # Also create the NoTaper R0 / R0.75 variants (2-stage QA'd) so coadd_final
    # carries EVERY I-deep variant per PHASE3_STRUCTURE, not just Taper.
    for _vk, _out_name in (('NoTaper_R0', f'I_deep_NoTaper_R0_{fl}.fits'),
                           ('NoTaper_R075', f'I_deep_NoTaper_R075_{fl}.fits')):
        _aligned = []
        for date, sci_dir, _ in dates:
            if date not in good_dates_deep or date not in date_params:
                continue
            params = date_params[date]
            if is_subband(freq_str):
                _img = find_subband_variant_image(sci_dir, freq_str, _vk)
            else:
                _img = find_wideband_deep_images(sci_dir, freq_str, 'I', _vk)
            if not _img:
                continue
            _why = _variant_input_divergent(_img)
            if _why:
                print(f"  VARIANT GATE {_vk} {date}: input REFUSED - {_why} ({os.path.basename(_img)})")
                _variant_refused.setdefault(_vk, []).append((date, _why))
                continue
            _out = f'{prep_dir}/{date}_{_vk}.fits'
            ok = process_variant_image(_img, params['flow'], params['scale_map'],
                                       tmpl_hdr, target_shape, _out, params['regrid'])
            if ok and os.path.exists(_out):
                _aligned.append((date, _out, params['weight']))
        if len(_aligned) >= MIN_DATES_FOR_COADD:
            _path = f'{final_dir}/{_out_name}'
            coadd_inverse_variance(
                [p for _, p, _ in _aligned],
                [w for _, _, w in _aligned],
                _path, tmpl_hdr, dates=[d for d, _, _ in _aligned],
                sigclip=False)
            stamp_master_span(_path, lst, fl, [d for d, _, _ in _aligned])
            _stamp_variant_gate(_path, _variant_refused.get(_vk, []))
            print(f"  Final {_vk} co-add: {len(_aligned)} dates")

    # ── Step 2: Per-date subtraction ──
    final_i_data, _ = load_image(final_i_path)
    final_v_data = None
    if final_v_path:
        final_v_data, _ = load_image(final_v_path)

    final_wcs = WCS(tmpl_hdr).celestial
    coadd_mask = ~np.isfinite(final_i_data) | (final_i_data == 0)

    print(f"\n  Per-date subtraction ({len(dates_good)} dates):")
    # ── PASS 1: measure every date's flux scale vs the co-add, so degenerate
    # fits can be guarded against the cell's ROBUST MEDIAN DC. This is the
    # THIRD call site of the flux-degeneracy guard (Gregg 2026-06-21) — it was
    # missing here, and a storm-displaced epoch (e.g. 11h 2025-01-06 Green)
    # mismatches sources, fits a wild affine plane (DC 0.29 vs sibling ~2.15),
    # and the division amplifies + sign-flips the epoch in bands → the
    # diagonal ghost-lane bug (identified by Gregg, debugged 2026-07-02).
    # The absolute 0.7–1.4 window used at the other call sites is wrong here
    # (the date→co-add DC is not ~1), hence the median-relative test.
    scale2_meas = {}
    for date, sci_dir, align_img_path in dates:
        if date not in date_params:
            continue
        i_path = f'{prep_dir}/{date}_I.fits'
        if not os.path.exists(i_path):
            continue
        i_data, _ = load_image(i_path)
        combined_mask = coadd_mask | (~np.isfinite(i_data)) | (i_data == 0)
        obs_time = get_midpoint_time(ref_hdr, i_path)
        _, coeffs2, n2 = measure_flux_scale(
            i_data, final_i_data, final_wcs, beam_pix, combined_mask, obs_time)
        scale2_meas[date] = (coeffs2, n2)
    _dcs = [float(c[0][0]) for c in scale2_meas.values() if c[1] >= 8]
    med_dc = float(np.median(_dcs)) if _dcs else 1.0

    for date, sci_dir, align_img_path in dates:
        if date not in scale2_meas:
            continue
        params = date_params[date]
        i_path = f'{prep_dir}/{date}_I.fits'
        i_data, _ = load_image(i_path)

        # Per-epoch time block (from this date's Phase-2 source) stamped onto every
        # residual below — keeps tmpl_hdr's WCS/grid, fixes the inherited-DATE-OBS bug.
        # Beam+RESTFRQ are stamped per Stokes at each write via
        # save_epoch_residual (R1 amendment #1) — never left as tmpl_hdr's.
        epoch_hdr = epoch_time_header(tmpl_hdr, align_img_path)

        coeffs2, n2 = scale2_meas[date]
        h, w = target_shape
        yg, xg = np.indices((h, w), dtype=float)
        if os.environ.get('P3_FLUXSCALE_TILT') != '1':
            coeffs2 = np.array([coeffs2[0], 0.0, 0.0])   # DC-only second pass (R-TILT-REMOVED-NEXTGEN)
        scale2_full = (coeffs2[0] + coeffs2[1] * (2.0 * xg / w - 1.0)
                       + coeffs2[2] * (2.0 * yg / h - 1.0))
        # Flux-degeneracy guard, median-relative (see PASS 1 comment)
        _dc = float(coeffs2[0]); _tilt = float(np.hypot(coeffs2[1], coeffs2[2]))
        if (n2 < 8 or not (0.6 * med_dc <= _dc <= 1.6 * med_dc)
                or _tilt > 0.3 * abs(med_dc)):
            print(f"    {date}: scale2-degenerate (DC={_dc:.3f} tilt={_tilt:.3f} "
                  f"n={n2}; cell median DC={med_dc:.3f}) -> flat DC={med_dc:.3f}")
            scale2_full = np.full((h, w), med_dc if med_dc > 0 else 1.0)
            coeffs2 = np.array([med_dc, 0.0, 0.0])

        # Apply flux correction and subtract
        safe_scale = np.where(np.abs(scale2_full) > 0.1, scale2_full, 1.0)
        i_corrected = i_data / safe_scale
        i_sub = i_corrected - final_i_data

        sub_i_path = f'{sub_dir}/{date}_I_sub_{fl}.fits'
        save_epoch_residual(sub_i_path, i_sub, epoch_hdr,
                            align_img_path, tmpl_hdr, 'I')

        # Striation removal on I
        i_mask = coadd_mask | (~np.isfinite(i_sub)) | (i_sub == 0)
        # window = 8 x beam (R-STRIATION-WINDOW-APPROVED-8BEAMS 2026-08-25;
        # supersedes both the 20-beam default and the fixed 4.375 deg)
        i_stri_model, _ = build_striation_model(
            i_sub, final_wcs, i_mask, ra_window_beams=8.0, beam_pix=beam_pix)
        i_clean = i_sub - i_stri_model
        clean_i_path = f'{sub_dir}/{date}_I_clean_{fl}.fits'
        save_epoch_residual(clean_i_path, i_clean, epoch_hdr,
                            align_img_path, tmpl_hdr, 'I')

        i_rms_before = get_uniform_rms(i_sub, i_mask)
        i_rms_after = get_uniform_rms(i_clean, i_mask)

        # V subtraction + striation removal
        v_path = f'{prep_dir}/{date}_V.fits'
        if os.path.exists(v_path) and final_v_data is not None:
            # R1 amendment #1 / restamp finding F2: V residuals carry the V
            # epoch image's OWN beam (Taper_R0 convention), never I-derived.
            v_src = find_stokes_v_deep_image(sci_dir, freq_str)
            v_data, _ = load_image(v_path)
            v_corrected = v_data / safe_scale
            v_sub = v_corrected - final_v_data

            sub_v_path = f'{sub_dir}/{date}_V_sub_{fl}.fits'
            save_epoch_residual(sub_v_path, v_sub, epoch_hdr,
                                v_src, tmpl_hdr, 'V')

            v_mask = coadd_mask | (~np.isfinite(v_sub)) | (v_sub == 0)
            v_stri_model, _ = build_striation_model(
                v_sub, final_wcs, v_mask, ra_window_beams=8.0, beam_pix=beam_pix)
            v_clean = v_sub - v_stri_model

            clean_v_path = f'{sub_dir}/{date}_V_clean_{fl}.fits'
            save_epoch_residual(clean_v_path, v_clean, epoch_hdr,
                                v_src, tmpl_hdr, 'V')

            v_rms_before = get_uniform_rms(v_sub, v_mask)
            v_rms_after = get_uniform_rms(v_clean, v_mask)
            print(f"    {date}: scale2={coeffs2[0]:.4f} ({n2} src)  "
                  f"I {i_rms_before*1000:.1f}→{i_rms_after*1000:.1f}  "
                  f"V {v_rms_before*1000:.1f}→{v_rms_after*1000:.1f} mJy")
        else:
            print(f"    {date}: scale2={coeffs2[0]:.4f} ({n2} src)  "
                  f"I {i_rms_before*1000:.1f}→{i_rms_after*1000:.1f} mJy")

    print(f"  Subtracted images → {sub_dir}/")

    # Cleanup prep aligned images (keep co-adds and subtracted) — UNLESS
    # P3_KEEP_PREP=1: the ratified structure (§15, 2026-08-25) archives the
    # per-epoch ALIGNED image as a first-class product, so the rebuild
    # fleet retains prep_final for the placement pass.
    if os.environ.get('P3_KEEP_PREP') != '1':
        shutil.rmtree(prep_dir, ignore_errors=True)
    _stamp_all_masters(final_dir, lst, fl)
    _archive_publish_cell(final_dir, sub_dir, lst, fl)
    with open(done_marker, 'w') as f:
        f.write(f'{len(dates_good)}/{len(dates)} dates\n')
    return {'lst': lst, 'freq': fl, 'n_good': len(dates_good), 'n_total': len(dates)}


def _run_final_worker(args_tuple):
    lst, freq_str, exclusions = args_tuple
    try:
        return run_final_lst_freq(lst, freq_str, exclusions)
    except Exception as e:
        fl = freq_label(freq_str)
        print(f"  ERROR {lst} {fl}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_final(lsts, freqs, exclusions, args):
    """Run final stage across all (LST, freq) combinations."""
    all_results = []
    t_start = time.time()
    n_parallel = getattr(args, 'parallel', 1) or 1

    for freq_str in freqs:
        fl = freq_label(freq_str)
        print(f"\n{'╔'+'═'*68+'╗'}")
        print(f"  FINAL: {fl}  ({n_parallel} parallel)")
        print(f"{'╚'+'═'*68+'╝'}", flush=True)

        work_items = [(lst, freq_str, exclusions) for lst in lsts]

        if n_parallel > 1:
            with Pool(n_parallel) as pool:
                results = pool.map(_run_final_worker, work_items)
            for r in results:
                if r:
                    all_results.append(r)
        else:
            for lst in lsts:
                result = run_final_lst_freq(lst, freq_str, exclusions)
                if result:
                    all_results.append(result)

    n_combos = len(all_results)
    print(f"\n{'═'*70}")
    print(f"  FINAL STAGE COMPLETE")
    print(f"  {n_combos} (LST, freq) combinations processed")
    print(f"  Time: {(time.time()-t_start)/60:.1f} min")
    print(f"{'═'*70}")


def run_provisional(lsts, freqs, exclusions, args, depth='deep'):
    """Run provisional stage across all (LST, freq) combinations."""
    all_metrics = []
    t_start = time.time()
    suffix = '' if depth == 'deep' else f'_{depth}'
    label = 'PROVISIONAL' if depth == 'deep' else f'PROVISIONAL {depth.upper()}'

    for freq_str in freqs:
        fl = freq_label(freq_str)
        print(f"\n{'╔'+'═'*68+'╗'}")
        print(f"  {label}: {fl}")
        print(f"{'╚'+'═'*68+'╝'}")

        for lst in lsts:
            if depth == '10min':
                metrics = run_provisional_10min_lst_freq(lst, freq_str, exclusions)
            else:
                metrics = run_provisional_lst_freq(lst, freq_str, exclusions)
            if metrics:
                all_metrics.extend(metrics)

    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_path = f'{_workdir_base()}/provisional{suffix}_summary.csv'
        summary_df.to_csv(summary_path, index=False)

        n_total = len(summary_df)
        n_outliers = int(summary_df['outlier'].sum())
        n_combos = summary_df.groupby(['lst', 'freq']).ngroups

        print(f"\n{'═'*70}")
        print(f"  {label} STAGE COMPLETE")
        print(f"  {n_combos} (LST, freq) combinations, {n_total} total entries")
        print(f"  {n_outliers} outliers ({100*n_outliers/n_total:.1f}%)")
        print(f"  Time: {(time.time()-t_start)/60:.1f} min")
        print(f"  Review: {summary_path}")
        print(f"{'═'*70}")
    else:
        print("\n  No data processed.")


_GRID_REFUSED_CACHE = {}


def _grid_refused_dates(target_dates, tag=''):
    """POST-CUTOVER epochs whose image grid differs from the cell's modal post-cutover grid (a
    Phase-2 anomaly such as 02h Green 2025-12-26 on 4096^2 instead of 2358^2): they cannot be
    flowed or stacked, so they are REFUSED as epochs (printed once per cell, logged in
    <workdir>/refusals_grid.csv by the caller) and excluded from every stage. Pre-cutover epochs
    are on their own grid by design and are REGRIDDED by the prep, never refused here."""
    post = [x for x in target_dates if x[0] >= CUTOVER_DATE]
    shapes = {}
    for date, _sd, img in post:
        try:
            _h = fits.getheader(img); shapes[date] = (int(_h['NAXIS2']), int(_h['NAXIS1']))
        except Exception:
            shapes[date] = None
    vals = [s for s in shapes.values() if s]
    modal = max(set(vals), key=vals.count) if vals else None
    wrong = {d for d, s in shapes.items() if s and modal and s != modal}
    for d in sorted(wrong):
        key = (tag, d)
        if key not in _GRID_REFUSED_CACHE:
            _GRID_REFUSED_CACHE[key] = True
            print(f"  {tag} {d}: epoch REFUSED - wrong grid {shapes[d]} vs modal post-cutover grid {modal} "
                  f"(Phase-2 anomaly; cannot be flowed; excluded from the co-add, logged)", flush=True)
    return wrong


def _flow_freshness(workdir, target_dates):
    """2026-09-06 (recovery chain): which non-reference epochs have NO alignment flow
    or a flow OLDER than their input image. The co-add only ever LOADS flow/<date>_scaled_flow.npy
    (never recomputes), so a recovered / restacked epoch would otherwise be aligned with the
    flow measured on its superseded image, or not at all ('flux-scale only'). Cells whose flow/
    holds no flow at all (wideband: inputs dewarped in Phase 2) are not gated.
    Returns [(date, reason)]."""
    fdir = f'{workdir}/flow'
    # wideband cells (Red/Green/Blue/Full) carry no co-add flows by design (inputs dewarped in
    # Phase 2): not gated. A SUBBAND cell with an empty flow/ is a cell whose every epoch would
    # be co-added flux-scale only - every epoch counts as missing (2026-09-06 05:0x UT, found on
    # 23h 36MHz whose flow/ was empty after the 2026-08-30 recovery rebuild).
    # 2026-09-09 (Gregg, contract A5 / FINDING-V3-WIDEBAND-MASTERS-STILL-FLOW-LESS): the
    # wideband exemption is REMOVED. Every non-reference epoch of every cell, wideband
    # included, needs a fresh co-add alignment flow; a raw (non-dewarped) wideband night
    # is dewarped by that flow and by nothing else.
    out = []
    # 2026-09-09 (02h Green: Phase-2 night 2025-12-26 delivered on a 4096^2 grid instead of the
    # Green 2358^2 grid): an epoch whose image grid differs from the cell's modal grid can never
    # be flowed - the prep skips it ('shape mismatch') - so it is REFUSED as an epoch (logged) and
    # must not read as a missing flow that refuses the whole cell.
    target_dates = [x for x in target_dates if x[0] not in _grid_refused_dates(target_dates, os.path.basename(workdir))]
    for date, _sd, img in target_dates:
        fp = f'{fdir}/{date}_scaled_flow.npy'
        if not os.path.exists(fp):
            out.append((date, 'no flow'))
        elif os.path.exists(img) and os.path.getmtime(img) > os.path.getmtime(fp):
            out.append((date, 'flow older than input image'))
    return out


def _run_batch_flow(all_manifest, fl):
    """Batch optical-flow alignment for a manifest (GPU-locked; py310_orca)."""
    manifest_path = f'{_workdir_base()}/_manifest_{fl}.json'
    with open(manifest_path, 'w') as f:
        json.dump(all_manifest, f)
    print(f"\n{'─'*70}")
    print(f"  Batch flow: {len(all_manifest)} pairs for {fl}")
    print(f"{'─'*70}")
    cmd = ["conda", "run", "-n", "py310_orca", "--no-capture-output",
           "python", BATCH_FLOW_SCRIPT, "--manifest", manifest_path]
    t0 = time.time()
    gpu_lock_path = os.path.join(_workdir_base(), '.gpu_lock')
    with open(gpu_lock_path, 'w') as lockf:
        print(f"  Waiting for GPU lock...")
        fcntl.flock(lockf, fcntl.LOCK_EX)
        print(f"  GPU lock acquired, running batch flow")
        result = subprocess.run(cmd, capture_output=False, timeout=7200)
    if result.returncode != 0:
        print(f"  WARNING: batch_flow returned {result.returncode}")
    print(f"  Batch flow done in {time.time()-t0:.0f}s")
    return result.returncode


def run_flow_stage(lsts, freqs, exclusions, args):
    """--stage flow (2026-09-06): (re)compute the alignment flows of the epochs whose flow is
    missing or older than the input image (all epochs with --flow-all), through the same prep
    + batch-flow code as the full path, and stop there. This is the step the provisional
    stage's freshness gate asks for."""
    for freq_str in freqs:
        fl = freq_label(freq_str)
        for lst in lsts:
            manifest, prep = process_lst_freq(lst, freq_str, exclusions, args)
            if not manifest:
                print(f"  {lst} {fl}: nothing to prep"); continue
            workdir = f'{_workdir_base()}/{lst}_{fl}'
            if getattr(args, 'flow_all', False):
                todo = manifest
            else:
                stale = {d for d, _ in _flow_freshness(workdir, [(x['date'], x['sci_dir'], x['align_img']) for x in prep['dates']])}
                todo = [m for m in manifest if os.path.basename(m['target']).split('_scaled')[0] in stale]
            print(f"  {lst} {fl}: {len(todo)} of {len(manifest)} epochs need a flow"
                  + (f" ({', '.join(sorted(os.path.basename(m['target']).split('_scaled')[0] for m in todo))})" if todo else ''))
            if todo:
                _run_batch_flow(todo, fl)
            left = _flow_freshness(workdir, [(x['date'], x['sci_dir'], x['align_img']) for x in prep['dates']])
            print(f"  {lst} {fl}: flow stage done; stale/missing after: {left if left else 'none'}")


def process_frequency(freq_str, lsts, exclusions, args):
    """Process one frequency across all LSTs.
    Groups flow alignment into one batch call per frequency."""
    fl = freq_label(freq_str)
    print(f"\n{'╔'+'═'*68+'╗'}")
    print(f"  Processing {fl} across {len(lsts)} LSTs")
    print(f"{'╚'+'═'*68+'╝'}")

    all_manifest = []
    all_prep = []

    # Phase 1: Prep (regrid, flux scale)
    for lst in lsts:
        manifest, prep = process_lst_freq(lst, freq_str, exclusions, args)
        all_manifest.extend(manifest)
        if prep:
            all_prep.append(prep)

    if not all_manifest:
        print(f"\n  No flow pairs to process for {fl}")
    else:
        # Phase 2: Batch flow alignment
        manifest_path = f'{_workdir_base()}/_manifest_{fl}.json'
        with open(manifest_path, 'w') as f:
            json.dump(all_manifest, f)

        print(f"\n{'─'*70}")
        print(f"  Batch flow: {len(all_manifest)} pairs for {fl}")
        print(f"{'─'*70}")

        cmd = ["conda", "run", "-n", "py310_orca", "--no-capture-output",
               "python", BATCH_FLOW_SCRIPT,
               "--manifest", manifest_path]
        t0 = time.time()
        gpu_lock_path = os.path.join(_workdir_base(), '.gpu_lock')
        with open(gpu_lock_path, 'w') as lockf:
            print(f"  Waiting for GPU lock...")
            fcntl.flock(lockf, fcntl.LOCK_EX)
            print(f"  GPU lock acquired, running batch flow")
            result = subprocess.run(cmd, capture_output=False, timeout=7200)
        if result.returncode != 0:
            print(f"  WARNING: batch_flow returned {result.returncode}")
        print(f"  Batch flow done in {time.time()-t0:.0f}s")

    # Phase 3: Post-flow processing
    for prep in all_prep:
        post_flow_processing(prep, exclusions, args)


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Full co-addition pipeline')
    parser.add_argument('--freq', type=str,
                        help='Frequency: subband MHz (e.g. 73) or wideband (Red/Green/Blue/Full)')
    parser.add_argument('--lst', type=str,
                        help='Single LST to process (e.g. 10h)')
    parser.add_argument('--all', action='store_true',
                        help='Process all frequencies and LSTs')
    parser.add_argument('--workdir', default=None,
                        help='Output base directory')
    parser.add_argument('--outlier_sigma', type=float, default=OUTLIER_SIGMA,
                        help='MAD outlier threshold in sigma')
    parser.add_argument('--flow-all', action='store_true', dest='flow_all',
                        help='--stage flow: recompute every epoch, not only stale/missing')
    parser.add_argument('--stage', choices=['flow', 'provisional', 'provisional-10min', 'final', 'variants'],
                        help='Two-pass co-add: provisional (deep outlier analysis), '
                             'provisional-10min (10-min interval outlier analysis), or final')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of LSTs to process in parallel (final stage)')
    args = parser.parse_args()

    if args.workdir:
        _update_workdir(args.workdir)
    _workdir_base()   # refuse to run without an explicit root
    if args.outlier_sigma != OUTLIER_SIGMA:
        _update_sigma(args.outlier_sigma)

    if not args.freq and not args.all:
        parser.error("Specify --freq or --all")

    # Determine LSTs
    if args.lst:
        lsts = [args.lst]
    else:
        lsts = sorted(REFERENCE_DATES.keys())

    # Load exclusions
    exclusions = load_exclusions()
    print(f"Loaded {len(exclusions)} exclusion entries from data_decisions.csv")

    # Determine frequencies
    if args.all:
        if args.stage == 'final':
            freq_order = (
                ['Red', 'Green', 'Blue', 'Full'] +
                [82, 78, 73, 69, 64] +  # Blue subbands
                [59, 55, 50, 46, 41] +  # Green subbands
                [36, 32, 27, 23, 18]    # Red subbands
            )
        else:
            freq_order = (
                [82, 78, 73, 69, 64] +  # Blue
                [59, 55, 50, 46, 41] +  # Green
                [36, 32, 27, 23, 18] +  # Red
                ['Blue', 'Green', 'Red', 'Full']
            )
        freqs = [str(f) for f in freq_order]
    else:
        freqs = [args.freq]

    t_start = time.time()

    if args.stage == 'flow':
        run_flow_stage(lsts, freqs, exclusions, args)
    elif args.stage == 'provisional':
        run_provisional(lsts, freqs, exclusions, args, depth='deep')
    elif args.stage == 'provisional-10min':
        run_provisional(lsts, freqs, exclusions, args, depth='10min')
    elif args.stage == 'final':
        run_final(lsts, freqs, exclusions, args)
    elif args.stage == 'variants':
        # variant-only rebuild from the generation record (Gregg 2026-09-11); P3_VARIANTS_ONLY=Taper_R0,NoTaper_R0
        _vks = tuple(v for v in os.environ.get('P3_VARIANTS_ONLY', 'Taper_R0,NoTaper_R0').split(',') if v)
        for _l in lsts:
            for _f in freqs:
                _r = rebuild_variants_from_record(_l, _f, variants=_vks)
                for _vk, (_p, _n, _ref) in _r.items():
                    print(f'VARIANTS-STAGE {_l} {freq_label(_f)} {_vk}: {"WRITTEN " + _p if _p else "NOT WRITTEN"} ({_n} constituents, {len(_ref)} refused)', flush=True)
    else:
        for freq_str in freqs:
            process_frequency(freq_str, lsts, exclusions, args)

    elapsed = time.time() - t_start
    print(f"\n{'═'*70}")
    print(f"  Complete. Total time: {elapsed/60:.1f} min")
    print(f"  Output: {_workdir_base()}/")
    print(f"{'═'*70}")


if __name__ == '__main__':
    main()
