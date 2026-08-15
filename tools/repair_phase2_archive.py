#!/usr/bin/env python3
"""
repair_phase2_archive.py — In-place header repair for archived Phase 2 runs.

Fixes the systematic header errors found in the 2026-06 audit
(reviewed against 07h/2024-12-29):

  1. TIMES: the old correction treated filename timestamps as start
     times, but they are interval MIDPOINTS — every DATE-OBS/MJD-OBS/
     DATE-END/MJD-MID was shifted late by half the duration (deep:
     +30 min, 10-min: +5 min). Exact starts are recovered from the
     pristine WSClean headers in the Phase 1 archive on Lustre.
     DURATION (wall-clock span) and INTTIME (summed integration time)
     are written as independent quantities.
  2. V BEAMS: were copied from an arbitrary I variant; re-copied from
     the I image matched on (taper, robust, timestamp).
  3. WIDEBAND TIMES: stacks inherited the corrupted subband times;
     rebuilt from the repaired subband solution. 10-min stack t-indices
     were assigned by filename string sort (scrambled by the duration
     token); files are renamed to chronological order.
  4. WIDEBAND WCS: missing CTYPE3='FREQ' axis (BDSF break) and Stokes
     axis added; stale per-subband WSC* keywords stripped.
  5. PROVENANCE: GITTAG stamped everywhere; constituent file lists
     reconstructed on stacks when they can be verified against NSTACK;
     THERMNOI stamped from matching Stokes V measurements.

Usage:
    # Single run, verbose:
    python repair_phase2_archive.py --lst 07h --date 2024-12-29

    # Everything under the default roots, 8 parallel runs:
    python repair_phase2_archive.py --all --workers 8

    # List what would be processed:
    python repair_phase2_archive.py --all --dry_run

A marker file (.repair_v2.done) is written into each run directory;
already-marked runs are skipped unless --force is given. All header
edits are idempotent.
"""
import os
import re
import sys
import glob
import json
import logging
import argparse
import traceback
import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta

import phase2_config as cfg
from validate_fits_headers import (correct_interval_headers, _stamp_times,
                                   repair_wideband_headers, LWA_DT)
from phase2_science_agg import (get_inner_rms, _stamp_subband_thermnoi,
                                _stamp_wideband_thermnoi)

DEFAULT_ROOTS = [
    '/lustre/pipeline/exopipe/phase2',
    '/lustre/pipeline/exopipe/phase2/flow',
]
MARKER = '.repair_v2.done'

WIDEBAND_NAME_RE = re.compile(
    r'^(Red|Green|Blue|Full)_([IV])_(deep|10min)_(NoTaper|Taper)_'
    r'(Robust-[\d.]+)(_pbcorr)?(_dewarped)?_(\d{8}_LST[^_]+)'
    r'(_t(\d{4}))?\.fits$')

logger = logging.getLogger('repair')


# ---------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------

def find_runs(roots, lst=None, date=None):
    """Yield (lst, date, run_dir) for archived Science_* run directories.

    The flow archive lives under <root>/flow/<lst>/..., so a plain
    <root>/<lst> glob on the parent root must not descend into 'flow'
    twice — handled by requiring the LST component to look like '00h'.
    """
    seen = set()
    runs = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for lst_dir in sorted(glob.glob(os.path.join(root, '*h'))):
            lst_label = os.path.basename(lst_dir)
            if not re.match(r'^\d{2}h$', lst_label):
                continue
            if lst and lst_label != lst:
                continue
            for date_dir in sorted(glob.glob(os.path.join(lst_dir, '*-*-*'))):
                date_label = os.path.basename(date_dir)
                if date and date_label != date:
                    continue
                for run_dir in sorted(glob.glob(
                        os.path.join(date_dir, 'Science_*'))):
                    real = os.path.realpath(run_dir)
                    if real in seen:
                        continue
                    seen.add(real)
                    runs.append((lst_label, date_label, run_dir))
    return runs


def find_phase1_subbands(lst, date):
    """Locate the Phase 1 subband dirs for a run — latest Run_* wins.
    Mirrors phase2_controller.discover_subbands without sys.exit."""
    base = os.path.join(cfg.CELERY_IMAGE_ROOT, lst, date)
    subband_map = {}
    for run_dir in sorted(glob.glob(os.path.join(base, 'Run_*'))):
        for sb_dir in glob.glob(os.path.join(run_dir, '*MHz')):
            sb = os.path.basename(sb_dir)
            if sb in cfg.SKIP_SUBBANDS:
                continue
            subband_map[sb] = sb_dir
    return subband_map


# ---------------------------------------------------------------------
# Wideband repair
# ---------------------------------------------------------------------

def _collect_subband_solution(run_dir):
    """Read the repaired subband headers and build the time solution.

    Returns (deep, intervals):
      deep[subband] = {'start','end','inttime'}
      intervals = list of clusters sorted by time, each
        {'start','end','inttime','mid','members':{subband: ts_string}}
    """
    deep = {}
    raw = []  # (subband, ts, start, end, inttime)
    for sb_dir in sorted(glob.glob(os.path.join(run_dir, '*MHz'))):
        sb = os.path.basename(sb_dir)
        for cat in ['deep', '10min']:
            d = os.path.join(sb_dir, 'I', cat)
            if not os.path.isdir(d):
                d = os.path.join(sb_dir, 'V', cat)
            if not os.path.isdir(d):
                continue
            seen = set()
            for f in sorted(glob.glob(os.path.join(d, '*-image*.fits'))):
                m = re.search(r'\d{8}_\d{6}', os.path.basename(f))
                if not m or m.group(0) in seen:
                    continue
                seen.add(m.group(0))
                try:
                    hdr = fits.getheader(f)
                except Exception:
                    continue
                if 'TIMEFIX' not in hdr or not hdr.get('DATE-OBS'):
                    continue
                start = Time(hdr['DATE-OBS'])
                end = Time(hdr['DATE-END']) if hdr.get('DATE-END') else None
                inttime = float(hdr.get('INTTIME') or hdr.get('DURATION', 0))
                if cat == 'deep':
                    if end is not None:
                        deep[sb] = {'start': start, 'end': end,
                                    'inttime': inttime}
                    break
                raw.append((sb, m.group(0), start, end, inttime))

    # Cluster 10-min entries by start time (within 60 s = same interval)
    raw.sort(key=lambda r: r[2].mjd)
    intervals = []
    for sb, ts, start, end, inttime in raw:
        if intervals and (start.mjd - intervals[-1]['_last']) * 86400.0 <= 60.0:
            iv = intervals[-1]
        else:
            iv = {'starts': [], 'ends': [], 'inttimes': [],
                  'members': {}, '_last': start.mjd}
            intervals.append(iv)
        iv['starts'].append(start)
        if end is not None:
            iv['ends'].append(end)
        iv['inttimes'].append(inttime)
        iv['members'][sb] = ts
        iv['_last'] = start.mjd

    out = []
    for iv in intervals:
        start = min(iv['starts'], key=lambda t: t.mjd)
        end = (max(iv['ends'], key=lambda t: t.mjd) if iv['ends']
               else start + TimeDelta(max(iv['inttimes']), format='sec'))
        out.append({'start': start, 'end': end,
                    'inttime': max(iv['inttimes']),
                    'mid': Time((start.mjd + end.mjd) / 2.0, format='mjd'),
                    'members': iv['members']})
    return deep, out


def _band_window(deep, band):
    """Aggregate the deep window over the subbands of one band."""
    sbs = (cfg.ALL_SUBBANDS if band == 'Full'
           else cfg.SUBBAND_BANDS.get(band, []))
    entries = [deep[sb] for sb in sbs if sb in deep]
    if not entries:
        entries = list(deep.values())
    if not entries:
        return None
    start = min((e['start'] for e in entries), key=lambda t: t.mjd)
    end = max((e['end'] for e in entries), key=lambda t: t.mjd)
    return {'start': start, 'end': end,
            'inttime': max(e['inttime'] for e in entries)}


def _match_interval(hdr, intervals):
    """Find the interval a 10-min stack belongs to.

    Old-corrupted stacks carry DATE-OBS = the interval midpoint of
    their first constituent; repaired stacks carry MJD-MID. Match by
    nearest interval midpoint within 5 minutes.
    """
    try:
        if 'TIMEFIX' in hdr and hdr.get('MJD-MID'):
            t = float(hdr['MJD-MID'])
        elif hdr.get('DATE-OBS'):
            t = Time(hdr['DATE-OBS']).mjd
        else:
            return None
    except Exception:
        return None
    best, best_dt = None, 300.0
    for iv in intervals:
        dt = abs(t - iv['mid'].mjd) * 86400.0
        if dt < best_dt:
            best, best_dt = iv, dt
    return best


def _constituent_paths(run_dir, band, pol, cat, taper, robust, pbcorr,
                       dewarped, interval):
    """Reconstruct the subband files that went into one stack."""
    sbs = (cfg.ALL_SUBBANDS if band == 'Full'
           else cfg.SUBBAND_BANDS.get(band, []))
    suffix = ('image' + ('.pbcorr' if pbcorr else '')
              + ('_dewarped' if dewarped else '') + '.fits')
    paths = []
    for sb in sbs:
        d = os.path.join(run_dir, sb, pol, cat)
        if not os.path.isdir(d):
            continue
        if cat == '10min':
            if interval is None or sb not in interval['members']:
                continue
            ts = interval['members'][sb]
            pat = f"{sb}-{pol}-{taper}-*-{robust}-{ts}-{suffix}"
        else:
            pat = f"{sb}-{pol}-{taper}-*-{robust}-*-{suffix}"
        hits = sorted(glob.glob(os.path.join(d, pat)))
        if len(hits) == 1:
            paths.append(hits[0])
    return paths


def repair_wideband_products(run_dir, log):
    """Repair times, t-indices, and provenance on wideband stacks."""
    deep, intervals = _collect_subband_solution(run_dir)
    if not deep and not intervals:
        log.warning("  No repaired subband headers found — skipping "
                    "wideband time repair")
        return 0

    n_fixed = 0
    renames = []

    for band in ['Red', 'Green', 'Blue', 'Full']:
        window = _band_window(deep, band)
        for pol in ['I', 'V']:
            for cat in ['deep', '10min']:
                d = os.path.join(run_dir, band, pol, cat)
                if not os.path.isdir(d):
                    continue
                # variant group -> [(start_mjd, path)] for t-relabeling
                groups = {}
                for f in sorted(glob.glob(os.path.join(d, '*.fits'))):
                    bn = os.path.basename(f)
                    m = WIDEBAND_NAME_RE.match(bn)
                    if not m:
                        log.warning(f"  Unparsed wideband name: {bn}")
                        continue
                    taper, robust = m.group(4), m.group(5)
                    pbcorr, dewarped = bool(m.group(6)), bool(m.group(7))

                    interval = None
                    if cat == 'deep':
                        win = window
                    else:
                        try:
                            hdr0 = fits.getheader(f)
                        except Exception:
                            continue
                        interval = _match_interval(hdr0, intervals)
                        win = interval
                        if interval is None:
                            log.warning(f"  No interval match for {bn}")
                    try:
                        with fits.open(f, mode='update') as hdul:
                            hdr = hdul[0].header
                            if win is not None:
                                _stamp_times(hdr, win['start'], win['end'],
                                             (win['end'] - win['start']).sec,
                                             win['inttime'],
                                             'subband-solution', pol)
                            # Strip stale per-subband WSClean keywords
                            for key in [k for k in list(hdr.keys())
                                        if str(k).startswith('WSC')]:
                                del hdr[key]
                            hdr['GITTAG'] = (cfg.PIPELINE_GITTAG,
                                             'Pipeline code version')
                            # Constituents, only when verifiable
                            nstack = hdr.get('NSTACK')
                            already = any('constituent:' in str(c) for c in
                                          hdr.get('HISTORY', []))
                            if nstack and not already:
                                paths = _constituent_paths(
                                    run_dir, band, pol, cat, taper, robust,
                                    pbcorr, dewarped, interval)
                                if len(paths) == nstack:
                                    for p in paths:
                                        rel = '/'.join(os.path.normpath(p)
                                                       .split(os.sep)[-4:])
                                        hdr['HISTORY'] = \
                                            f'  constituent: {rel}'
                                else:
                                    log.info(
                                        f"  {bn}: constituents not "
                                        f"reconstructable ({len(paths)} "
                                        f"candidates vs NSTACK={nstack})")
                            hdul.flush()
                            n_fixed += 1
                            if cat == '10min' and win is not None:
                                gkey = re.sub(r'_t\d{4}', '', bn)
                                groups.setdefault(gkey, []).append(
                                    (win['start'].mjd, f))
                    except Exception as e:
                        log.warning(f"  Wideband repair failed for {bn}: {e}")

                # Chronological t-index relabeling
                for gkey, members in groups.items():
                    members.sort(key=lambda x: x[0])
                    moves = []
                    for new_idx, (_, f) in enumerate(members, start=1):
                        bn = os.path.basename(f)
                        m = WIDEBAND_NAME_RE.match(bn)
                        cur_idx = int(m.group(10)) if m and m.group(10) \
                            else None
                        if cur_idx is not None and cur_idx != new_idx:
                            new_bn = bn.replace(f'_t{cur_idx:04d}',
                                                f'_t{new_idx:04d}')
                            moves.append((f, os.path.join(d, new_bn)))
                    if moves:
                        # Two-phase rename to avoid collisions
                        for src, dst in moves:
                            os.rename(src, dst + '.tmp_relabel')
                        for src, dst in moves:
                            os.rename(dst + '.tmp_relabel', dst)
                            renames.append((os.path.basename(src),
                                            os.path.basename(dst)))

    if renames:
        map_path = os.path.join(run_dir, 'wideband_tindex_renames.txt')
        with open(map_path, 'a') as fh:
            fh.write(f"# t-indices relabeled to chronological order "
                     f"({len(renames)} files)\n")
            for old, new in renames:
                fh.write(f"{old}  ->  {new}\n")
        log.info(f"  Relabeled {len(renames)} 10-min stack t-indices "
                 f"to chronological order")

    return n_fixed


# ---------------------------------------------------------------------
# THERMNOI from the QA CSV (cheap) with direct measurement fallback
# ---------------------------------------------------------------------

def stamp_thermal_noise(run_dir, log):
    deep_rms, tenmin_rms = {}, {}
    csvs = glob.glob(os.path.join(run_dir, 'QA', '*thermal_noise.csv'))
    if csvs:
        import csv as csvmod
        with open(csvs[0]) as fh:
            for row in csvmod.DictReader(fh):
                try:
                    freq = float(row['freq'])
                    rms = float(row['rms'])
                except (KeyError, ValueError):
                    continue
                sb = f"{freq:.0f}MHz"
                if row.get('category', 'deep') == 'deep':
                    deep_rms[sb] = rms
                else:
                    m = re.search(r'\d{8}_\d{6}', row.get('file', '') or '')
                    if m:
                        tenmin_rms[(sb, m.group(0))] = rms
    if not deep_rms:
        # Fallback: measure the archived V images directly
        for sb_dir in sorted(glob.glob(os.path.join(run_dir, '*MHz'))):
            sb = os.path.basename(sb_dir)
            cands = sorted(glob.glob(os.path.join(
                sb_dir, 'V', 'deep', '*Taper*-image_dewarped.fits')))
            if cands:
                rms = get_inner_rms(cands[0])
                if np.isfinite(rms) and rms > 0:
                    deep_rms[sb] = rms

    n = _stamp_subband_thermnoi(run_dir, deep_rms, tenmin_rms)
    n += _stamp_wideband_thermnoi(run_dir)
    log.info(f"  THERMNOI stamped on {n} files")
    return n


# ---------------------------------------------------------------------
# Per-run driver
# ---------------------------------------------------------------------

def repair_run(lst, date, run_dir, force=False, log_dir=None):
    marker = os.path.join(run_dir, MARKER)
    if os.path.exists(marker) and not force:
        return 'skipped'

    log = logging.getLogger(f'repair.{lst}.{date}')
    log.setLevel(logging.INFO)
    log.handlers = []
    fmt = logging.Formatter('%(asctime)s %(message)s')
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(log_dir, f'{lst}_{date}.log'), mode='a')
        fh.setFormatter(fmt)
        log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)

    log.info(f"=== Repairing {lst} {date}: {run_dir} ===")
    try:
        subband_map = find_phase1_subbands(lst, date)
        if not subband_map:
            log.warning("  Phase 1 archive not found — falling back to "
                        "midpoint arithmetic for start times")

        # 1+2. Subband times, V beams, frequency metadata (no renames)
        correct_interval_headers(run_dir, log,
                                 subband_src_map=subband_map or None,
                                 rename=False, force_v_beam=True)

        # 3. Wideband times, t-indices, GITTAG, constituents
        repair_wideband_products(run_dir, log)

        # 4. Wideband FREQ/Stokes WCS axes
        repair_wideband_headers(run_dir, log)

        # 5. THERMNOI
        stamp_thermal_noise(run_dir, log)

        with open(marker, 'w') as fh:
            fh.write(f"repaired {Time.now().isot} gittag="
                     f"{cfg.PIPELINE_GITTAG}\n")
        log.info(f"=== Done: {lst} {date} ===")
        return 'ok'
    except Exception as e:
        log.error(f"  REPAIR FAILED for {run_dir}: {e}")
        log.error(traceback.format_exc())
        return 'failed'


def main():
    ap = argparse.ArgumentParser(
        description='Repair archived Phase 2 headers in place.')
    ap.add_argument('--roots', nargs='+', default=DEFAULT_ROOTS)
    ap.add_argument('--lst', help='Restrict to one LST label (e.g. 07h)')
    ap.add_argument('--date', help='Restrict to one date (e.g. 2024-12-29)')
    ap.add_argument('--all', action='store_true',
                    help='Process every run found under the roots')
    ap.add_argument('--dry_run', action='store_true',
                    help='List runs without modifying anything')
    ap.add_argument('--force', action='store_true',
                    help='Re-repair runs that already have a done marker')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--log_dir', default='repair_logs')
    args = ap.parse_args()

    if not args.all and not (args.lst and args.date):
        ap.error('Give --lst and --date for a single run, or --all')

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    runs = find_runs(args.roots, lst=args.lst, date=args.date)
    print(f"Found {len(runs)} run(s)")
    if args.dry_run:
        for lst, date, rd in runs:
            done = os.path.exists(os.path.join(rd, MARKER))
            print(f"  {lst} {date}  {rd}  "
                  f"{'[done]' if done and not args.force else '[pending]'}")
        return

    results = {'ok': 0, 'failed': 0, 'skipped': 0}
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(repair_run, lst, date, rd, args.force,
                              args.log_dir): (lst, date)
                    for lst, date, rd in runs}
            for fut in as_completed(futs):
                lst, date = futs[fut]
                try:
                    status = fut.result()
                except Exception:
                    status = 'failed'
                results[status] += 1
                print(f"[{sum(results.values())}/{len(runs)}] "
                      f"{lst} {date}: {status}")
    else:
        for i, (lst, date, rd) in enumerate(runs, 1):
            status = repair_run(lst, date, rd, args.force, args.log_dir)
            results[status] += 1
            print(f"[{i}/{len(runs)}] {lst} {date}: {status}")

    print(f"\nSummary: {results['ok']} repaired, {results['skipped']} "
          f"skipped (already done), {results['failed']} FAILED")
    if results['failed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
