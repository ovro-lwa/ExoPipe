#!/usr/bin/env python
"""Phase 3 snapshot-tier transient search — gated engine (redesign 2026-09-05).

Searches the 10-second Phase-1 snapshot archive of one (LST, date) cell, one
subband, one Stokes, for transient sources against a sky model that is either
(a) the hour's own median image ("median" mode) or (b) the deep Taper-Robust-0
master warped INTO the hour's ionospheric frame ("master" mode).

DETECTION STATISTIC (Gregg 2026-09-05, approved): for every searched image
(10-s frame, 1-min block, group frame) the residual (image - model) is divided
by that image's OWN global noise, the MAD-based rms of the residual over the
central 25% of the image area on the on-sky/elevation domain.  No per-pixel
noise map is used anywhere (the per-pixel temporal MAD of the prototype
selected its own estimation errors; audit F1) and every image carries its own
noise so group frames of varying composition are normalised correctly
(audit F2).  Per-image noise is PERSISTED in noise_{tag}.csv (10-s) and
noise_{tag}_1min.csv (blocks, with member frames), which makes every detection
reproducible without cube reconstruction (audit F11).

SEARCH DOMAIN: on-sky pixels with elevation >= 30 deg (Gregg 2026-09-09; was 20) evaluated PER FRAME
(zenith moves 15 deg/hour; six time bins per hour), ratified domain rule
implemented at frame level (audit F10-ELEV).

FRAME ADMISSION: per-Stokes noise refusal on the MEDIAN-SUBTRACTED frame
(Stokes I on the I residual, Stokes V on the V residual; central-25% MAD more
than NOISE_OUTLIER_K robust sigma above the cell median is refused; Gregg
2026-09-09 - the raw-V rms flag is now recorded only); beam gate (fitted-beam
area > 1.5x cell median refused); blank-frame refusal (finite fraction < 0.5);
and FLOOD REFUSAL: a frame whose count of above-threshold components exceeds
max(FLOOD_MIN_ABS, FLOOD_REL x the cell's median per-frame count), capped at
FLOOD_ABS_MAX, is refused outright (never truncated; audit F2).  Refusals are
recorded per frame with their reason.

1-MINUTE TIER: blocks are the admitted frames whose INTEGRATION MIDPOINT (header
DATE-OBS + half an integration) lies in one UTC minute (identical block
boundaries across subbands; block ts = YYYYMMDD_HHMM00; audit F6; frame-time
fix 2026-10-07, minimal form - see INTEGRATION_S), >= 3 frames required; the
block residual is high-passed (65-px NaN-aware boxcar) before detection.
--tier1min-only re-runs that tier alone from a cell's kept 10-s products.

Outputs under --out:
  candidates_{tag}.csv / events_{tag}.csv         10-s detections / clusters
  candidates_{tag}_1min.csv / events_{tag}_1min.csv
  noise_{tag}.csv / noise_{tag}_1min.csv          per-image noise + refusals
  admission_{sb}.csv                              per-frame QA + gate verdicts
  model_{tag}.fits                                the sky model subtracted
  runinfo_{tag}.json
Thresholds here are HARVEST thresholds (store floor); designation thresholds
live in phase3_snapshot_chain.py.
"""
import argparse
import csv
csv.field_size_limit(100000000)
import datetime
import glob
import json
import logging
import os
import re
import sys
import time
from collections import OrderedDict

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage
from scipy.spatial import cKDTree

log = logging.getLogger("snapsearch")

# P3_SNAP_PHASE1_ROOT (2026-11-09, the Dec 18-27 2024 search, Gregg 11-08 "a full search of I and V is warranted"): the frame
# root of a run - the observatory GPU pipeline's /lustre/pipeline/snapshots_gpu_pipeline holds the same <LST>/<date>/Run_*/<sb>/
# layout and file names (snapshots_clean/ Stokes I CLEANed, snapshots/ Stokes V) that discover_frames reads; unset = exopipe Phase 1
PHASE1_ROOT = os.environ.get("P3_SNAP_PHASE1_ROOT", "/lustre/pipeline/exopipe/phase1")
PHASE2_ROOT = "/lustre/pipeline/exopipe/phase2"
PB_LIBRARY = "/lustre/pipeline/exopipe/phase3/pb_library"
MASTER_ROOT = "/data07/gh/phase3/coadd_sb"

OVRO_LON_DEG, OVRO_LAT_DEG = -118.2817, 37.2398
MIN_ELEVATION_DEG = 30.0   # Gregg 2026-09-09: snapshot-tier floor 30 deg (was 20)
N_TIME_BINS = 6
# flood refusal (Gregg 2026-09-05): relative to the cell's own per-frame
# component count, with an absolute floor and ceiling on the limit
FLOOD_REL = 5.0
FLOOD_MIN_ABS = 50
FLOOD_ABS_MAX = 500
HARD_CAP_COMPONENTS = 20000      # memory guard: never extract beyond this

SCIENCE_SUBBANDS = ["18MHz", "23MHz", "27MHz", "32MHz", "36MHz", "41MHz",
                    "46MHz", "50MHz", "55MHz", "59MHz", "64MHz", "69MHz",
                    "73MHz", "78MHz", "82MHz"]

# filename patterns across run generations
_PAT_OLD = re.compile(
    r"^(?P<sb>\d+MHz)-Pilot-Snapshot-(?P<ts>\d{8}_\d{6})-(?P<st>[IV])-image\.fits(?P<fz>\.f[sz])?$")
_PAT_NEW_V = re.compile(
    r"^(?P<sb>\d+MHz)-Pilot-Snapshot-(?P<ts>\d{8}_\d{6})-image\.fits(?P<fz>\.f[sz])?$")
_PAT_NEW_I = re.compile(
    r"^(?P<sb>\d+MHz)-Clean-Snapshot-(?P<ts>\d{8}_\d{6})-image\.fits(?P<fz>\.f[sz])?$")


# ---------------------------------------------------------------- time

def ts_to_datetime(ts):
    """'YYYYMMDD_HHMMSS' (UTC) -> datetime."""
    return datetime.datetime.strptime(ts[:15], "%Y%m%d_%H%M%S")


_EPOCH = datetime.datetime(1970, 1, 1)


def ts_to_epoch_seconds(ts):
    return (ts_to_datetime(ts) - _EPOCH).total_seconds()


def ts_minute_key(ts):
    """Wall-clock minute key shared by every subband: 'YYYYMMDD_HHMM'."""
    return ts[:13]


# ---------------------------------------------------------------- integration time (frame-time fix, MINIMAL FORM)
# Gregg 2026-10-03 (R-FRAME-TIME-FROM-INTEGRATION-ALL-STAGES), verbatim: "Recommendation approved. How long to implement? Fix
# it at all stages of the pipeline."; 2026-10-07 on the strict staged version (every pre-fix product refused, 1,380 node-hours
# of re-search): "Were not redoing 1,380 hours of anything. Fix it." and "A few seconds timing error doesnt really matter. Go
# with your recommendation" (RULINGS-GREGG-20261007-STATUS-ANSWERS, Q1).  So a frame's IDENTITY stays its Phase-1 file-name
# time ts (10-s products, fit stores, cutouts and flow-field lookups keep their keys), and only the stages that put frames of
# one integration TOGETHER use each frame's own integration time from its header: (1) 1-minute block membership
# (build_minute_blocks), (2) wideband member pairing (phase3_snapshot_group_search), (3) the forced-photometry 1-minute blocks
# (exopipe_ops snapshot_forced_phot.minute_key).  Old file names drift from the integration by -21..+26 s, differently per
# subband (FINDING-PHASE1-SNAPSHOT-FILENAME-TIMES-WRONG); DATE-OBS is exact and one value per integration in every subband and
# both Stokes (358 integrations, 10,706 and 10,596 frames on 19h 2025-05-03 and 14h 2025-04-21, measured 2026-10-07).
# Integration time = DATE-OBS (the integration START: WSClean writes the interval's first timestep, exopipe 192c81c
# interval_midpoints) + half an integration = its MIDPOINT, which is how Phase 1 names frames since exopipe 192c81c - for
# new-era frames the name minute and the midpoint minute therefore agree.
INTEGRATION_S = 10.03102041      # s, OVRO-LWA integration: MS INTERVAL of 20250503_1206*_27MHz_averaged.ms (read 2026-10-07)
TIER1MIN_TIME_BASIS = "integration-midpoint-dateobs-v1"   # runinfo frame_time_1min of a 1-minute tier built this way
_DATEOBS_KEY = b"DATE-OBS="
_DATEOBS_MEMO = {}               # path -> DATE-OBS, this process


def dateobs_unix(date_obs):
    """FITS DATE-OBS 'YYYY-MM-DDTHH:MM:SS[.fff]' (UTC) -> unix seconds; ValueError when it is not one."""
    s = str(date_obs).strip()
    base = datetime.datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
    return (base - _EPOCH).total_seconds() + (float("0" + s[19:]) if len(s) > 19 else 0.0)


def integration_mid_unix(date_obs):
    """Unix seconds of the MIDPOINT of the 10-s integration whose header DATE-OBS (its start) is date_obs."""
    return dateobs_unix(date_obs) + INTEGRATION_S / 2.0


def integration_minute_key(date_obs):
    """'YYYYMMDD_HHMM' of the UTC minute holding the integration MIDPOINT of the frame whose DATE-OBS is date_obs."""
    return (_EPOCH + datetime.timedelta(seconds=int(np.floor(integration_mid_unix(date_obs))))).strftime("%Y%m%d_%H%M")


def integration_label(date_obs):
    """'YYYYMMDD_HHMMSS' of the integration MIDPOINT truncated to the second - the name Phase 1 gives the frame since exopipe
    192c81c.  A DISPLAY time and a cross-subband grouping key (frame movies), never a frame's identity (its file-name ts)."""
    return (_EPOCH + datetime.timedelta(seconds=int(np.floor(integration_mid_unix(date_obs))))).strftime("%Y%m%d_%H%M%S")


def read_dateobs(path, nblk=8):
    """The DATE-OBS card of a FITS file (plain, or tile-compressed .fz/.fs: the image extension's header), read from the
    first nblk 2880-byte blocks only (a buffered open pulls the whole 1.4-39 MB frame); '' if there is none."""
    fd = os.open(path, os.O_RDONLY)
    try:
        buf = os.pread(fd, 2880 * nblk, 0)
    finally:
        os.close(fd)
    for i in range(0, len(buf) - 79, 80):
        card = buf[i:i + 80]
        if card.startswith(_DATEOBS_KEY):
            return card[10:].split(b"/")[0].strip().strip(b"'").strip().decode()
    if len(buf) == 2880 * nblk:              # a header longer than nblk blocks: let astropy find the card
        with fits.open(path, memmap=False) as hl:
            for hdu in hl:
                if "DATE-OBS" in hdu.header:
                    return str(hdu.header["DATE-OBS"]).strip()
    return ""


def frame_dateobs(paths, nthreads=16):
    """{path: DATE-OBS} for many frame files: this process's memo, then the headers themselves, nthreads at a time
    (staged node-local copies cost well under 1 ms each; the Lustre archive 4 ms per file at 16 threads, measured 2026-10-07)."""
    out = {p: _DATEOBS_MEMO[p] for p in paths if p in _DATEOBS_MEMO}
    todo = [p for p in paths if p not in out]
    if todo:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, min(nthreads, len(todo)))) as ex:
            for p, d in zip(todo, ex.map(read_dateobs, todo)):
                out[p] = d
                if d:
                    _DATEOBS_MEMO[p] = d
    return out


def lst_deg_at(epoch_seconds):
    from astropy.time import Time
    import astropy.units as u
    t = Time(epoch_seconds, format="unix", scale="utc")
    return float(t.sidereal_time("apparent", longitude=OVRO_LON_DEG * u.deg).deg)


# ---------------------------------------------------------------- discovery

PHASE1_RUN_REPORT = {}           # (lst, date, subband) -> what discover_frames found where several Phase-1 runs hold frames
_PHASE1_RUN_PRINTED = set()
MAX_FRAMES_PER_HOUR = int(3600.0 / INTEGRATION_S) + 2     # integrations in the hour, +2 (core/inputs.phase1_frames content test)


class Phase1DiscoveryRefused(Exception):
    """discover_frames cannot give one frame per integration (refactor chat ruling 2026-10-18: both guards of
    core/inputs.phase1_frames refuse loudly).  Deliberately NOT a RuntimeError: the group search reads a RuntimeError from
    discovery as 'no frames, subband dropped', which would turn the refusal into a silent drop."""


def _one_frame_per_integration(lst, date, subband, per_run):
    """{'I': {ts: archive path}, 'V': {...}} from per_run = [(run name, {'I': {ts: path}, 'V': {...}}), ...] in run order
    (sorted Run_<UTC> names: the last is the latest).  One run holding frames of a Stokes: its frames, unchanged (no header
    is read).  Several: each frame's integration from its archive header DATE-OBS, later runs first, so the latest run's frame
    of an integration is kept and an older run only fills integrations the later runs lack (core/inputs.phase1_frames)."""
    out = {"I": {}, "V": {}}
    rep = {"runs": OrderedDict((r, {"I": len(f["I"]), "V": len(f["V"])}) for r, f in per_run), "mixed": False,
           "dropped": {"I": 0, "V": 0}, "integrations": {"I": 0, "V": 0}, "disagree": {"I": False, "V": False},
           "ts_collisions": {"I": 0, "V": 0},
           "shadowed": {"I": 0, "V": 0}, "lost": {"I": [], "V": []}, "rule": "one frame per integration (header DATE-OBS), latest Phase-1 run wins"}
    for st in ("I", "V"):
        contributing = [(r, f[st]) for r, f in per_run if f[st]]
        if len(contributing) <= 1:
            out[st] = dict(contributing[0][1]) if contributing else {}
            continue
        rep["mixed"] = True
        dob = frame_dateobs([p for _, f in contributing for p in f.values()])
        chosen = {}          # integration label -> (ts, path)
        sets = []
        for r, fr in reversed(contributing):
            own = set()
            for ts, p in sorted(fr.items()):
                try:
                    key = integration_label(dob.get(p, ""))
                except (ValueError, TypeError):
                    # REFUSED (refactor chat 2026-10-18; core/inputs.phase1_frames): a frame without a readable DATE-OBS
                    # cannot be assigned to an integration, so the runs cannot be de-duplicated
                    raise Phase1DiscoveryRefused("%s: no readable DATE-OBS (%r) - cannot assign the frame to an integration, "
                                                 "%s %s %s holds several Phase-1 runs" % (p, dob.get(p, ""), lst, date, subband))
                own.add(key)
                if key in chosen:
                    rep["dropped"][st] += 1
                else:
                    chosen[key] = (ts, p)
            sets.append(own)
        rep["integrations"][st] = len(chosen)
        rep["disagree"][st] = any(s != sets[0] for s in sets[1:])
        for key, (ts, p) in chosen.items():      # insertion order: the latest run first
            if ts in out[st]:        # two integrations under one file-name second: the later run's frame is kept, counted
                # (frame identity = the file-name second, refactor chat 2026-10-18: the lost integration is accepted and listed)
                rep["ts_collisions"][st] += 1
                rep["lost"][st].append({"label": key, "date_obs": dob.get(p, ""), "file_ts": ts,
                                        "run": next((x for x in p.split(os.sep) if x.startswith("Run_")), "")})
                continue
            out[st][ts] = p
        # a kept older-run frame whose FILE NAME a later run also uses (for another integration): a staging keyed by file name
        # (fleet stage, exploded cube) may hold the later run's file under that name - counted for review, not resolved here
        later = set()
        for r, fr in reversed(contributing):
            for ts, p in fr.items():
                if out[st].get(ts) == p and os.path.basename(p) in later:
                    rep["shadowed"][st] += 1
            later.update(os.path.basename(p) for p in fr.values())
    # REFUSED (refactor chat 2026-10-18; core/inputs.phase1_frames content test): never more frames than integrations in the
    # hour (+2) - frames of several runs would still be mixed
    for st in ("I", "V"):
        if len(out[st]) > MAX_FRAMES_PER_HOUR:
            raise Phase1DiscoveryRefused("%s %s %s Stokes %s: %d frames for one hour > %d integrations - frames from several "
                                         "runs are still mixed" % (lst, date, subband, st, len(out[st]), MAX_FRAMES_PER_HOUR))
    if rep["mixed"]:
        PHASE1_RUN_REPORT[(lst, date, subband)] = rep
        if (lst, date, subband) not in _PHASE1_RUN_PRINTED:
            _PHASE1_RUN_PRINTED.add((lst, date, subband))
            print("PHASE1-RUNS-MIXED %s %s %s: runs %s; kept I %d V %d (one per integration, latest run wins), dropped "
                  "older-run copies I %d V %d; runs disagree on the integration set: I %s V %s; file-name collisions "
                  "(integration lost) I %d V %d%s; kept older-run names a later run also uses I %d V %d"
                  % (lst, date, subband, ",".join("%s(I%d,V%d)" % (r, n["I"], n["V"]) for r, n in rep["runs"].items()),
                                 len(out["I"]), len(out["V"]), rep["dropped"]["I"], rep["dropped"]["V"],
                                 rep["disagree"]["I"], rep["disagree"]["V"],
                                 rep["ts_collisions"]["I"], rep["ts_collisions"]["V"],
                                 "".join(" [%s lost %s DATE-OBS %s, %s]" % (st, x["label"], x["date_obs"], x["run"])
                                         for st in ("I", "V") for x in rep["lost"][st]),
                                 rep["shadowed"]["I"], rep["shadowed"]["V"]),
                  file=sys.stderr, flush=True)     # stderr: callers capture discovery's stdout as a file list
    else:
        PHASE1_RUN_REPORT.pop((lst, date, subband), None)
    return out


RUN_PINS = os.environ.get("P3_SNAP_RUN_PINS", "/lustre/gh/main/exopipe_ops/snapshot_phase1_run_pins.csv")
_RUN_PIN_CACHE = None


def phase1_run_pin(lst, date):
    """The pinned Phase-1 run (directory name) of the cell, or None - read once per process from RUN_PINS (columns cell, run,
    ruling, reason); no file = no pins."""
    global _RUN_PIN_CACHE
    if _RUN_PIN_CACHE is None:
        pins = {}
        if os.path.exists(RUN_PINS):
            with open(RUN_PINS, newline="") as fh:
                for r in csv.DictReader(fh):
                    pins[r["cell"].strip()] = r["run"].strip()
        _RUN_PIN_CACHE = pins
    return _RUN_PIN_CACHE.get("%s_%s" % (lst, date))


def discover_frames(lst, date, subband, phase1_root=None,
                    stage_root=None, flow_select=True):
    # flow_select=False (2026-10-04, DEFECT-FIT-PENDING-THREAD-RACE-FITTED-UNFLOWED-FRAMES): the plain frame list, no flowed
    # substitution and no refusal - what the fitter and the panels used to get by switching the global FLOW_MODE off, which
    # raced with their worker threads.  Nothing may toggle FLOW_MODE any more.
    """Return {'I': [(ts, path), ...], 'V': [...]} sorted by timestamp.

    Scans every Run_* under the cell (patch runs included).  If stage_root
    is given and contains a staged copy of this subband's snapshots, the
    staged paths are used instead of the /lustre ones (same filenames).

    ONE FRAME PER INTEGRATION, THE LATEST PHASE-1 RUN WINS (port of the refactor's core/inputs.phase1_frames rule; run-mixing
    defect handed to the refactor 2026-10-13, Gregg "Hand it to refactor"; ledger FINDING-DUPLICATE-PHASE1-RUNS-MERGED-IN-
    SNAPSHOT-DISCOVERY-20261018-UPDATE): two runs of one hour name the same integration 1-9 s apart, so keying by the file-name
    second kept both copies (12h 2025-01-05 73MHz: 677 frames for 356 integrations).  Where more than one run holds frames of
    this subband and Stokes, each archive frame is assigned to its integration by its own header DATE-OBS (integration_label:
    the midpoint, truncated to the second) and the latest run's frame is kept; an older run contributes only integrations the
    later runs lack.  A kept frame's identity stays its own file-name ts (the frame-identity rule, 2026-10-07), so a cell with
    one run returns exactly what it returned before.  A cell is never refused for holding several runs: a mixed cell-band is
    printed once per process and recorded in PHASE1_RUN_REPORT[(lst, date, subband)] (the full search copies it into runinfo
    phase1_runs), with every integration lost to a file-name collision listed by label and DATE-OBS.  REFUSED
    (Phase1DiscoveryRefused, refactor chat 2026-10-18): a frame of a mixed cell-band without DATE-OBS, and more frames than
    integrations in the hour (MAX_FRAMES_PER_HOUR) in any cell.
    """
    if phase1_root is None:          # the cell's own frame root (frame_root: GPU-frame cells list, 2026-11-10)
        phase1_root = frame_root(lst, date)
    rundirs = sorted(glob.glob(os.path.join(phase1_root, lst, date, "Run_*")))
    if not rundirs:
        raise RuntimeError("no phase-1 runs for %s %s" % (lst, date))
    # PHASE-1 RUN PIN (Gregg 2026-11-01, 18h 2025-05-05: "Pin to June run (Recommended)"): a cell listed in the pin file
    # (RUN_PINS) resolves to its pinned run only - the run its served snapshot search used - while a later Phase-1 run of the hour,
    # not adopted by the snapshot tier, exists; REFUSES when the pinned run is not in the archive.  A cell without a pin: unchanged.
    pin = phase1_run_pin(lst, date)
    if pin:
        rundirs = [r for r in rundirs if os.path.basename(r) == pin]
        if not rundirs:
            raise RuntimeError("REFUSED: %s %s is pinned to Phase-1 run %s, which is not under %s" % (lst, date, pin, phase1_root))
    per_run = []
    for rd in rundirs:
        fr = {"I": {}, "V": {}}
        for sub, stokes_hint in (("snapshots", None), ("snapshots_clean", "I")):
            d = os.path.join(rd, subband, sub)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                m = _PAT_OLD.match(fn)
                if m:
                    fr[m.group("st")][m.group("ts")] = os.path.join(d, fn)
                    continue
                m = _PAT_NEW_I.match(fn)
                if m:
                    fr["I"][m.group("ts")] = os.path.join(d, fn)
                    continue
                m = _PAT_NEW_V.match(fn)
                if m and stokes_hint != "I":
                    fr["V"][m.group("ts")] = os.path.join(d, fn)
        if fr["I"] or fr["V"]:
            per_run.append((os.path.basename(rd), fr))
    frames = _one_frame_per_integration(lst, date, subband, per_run)
    if stage_root:
        for st in ("I", "V"):
            for ts, p in list(frames[st].items()):
                staged = os.path.join(stage_root, os.path.basename(p))
                if os.path.exists(staged):
                    frames[st][ts] = staged
                    continue
                # staged copies may have been decompressed: the archive uses
                # .fs on the 2024 runs and .fz on the 2025 ones (Gregg's fleet,
                # 2026-09-08 - a .fz cell staged fine but every search raised
                # "no V frames" because only .fs was stripped here)
                for _sfx in (".fs", ".fz"):
                    if staged.endswith(_sfx) and os.path.exists(staged[:-len(_sfx)]):
                        frames[st][ts] = staged[:-len(_sfx)]
                        break
    # PER-FRAME FLOW TO THE MASTER (Gregg 2026-09-12; test implementation):
    # with P3_SNAP_FLOW=1 the engine searches the <base>_flowed.fits frames
    # written by phase3_snapshot_flow.py (registered to the multi-night
    # master), both Stokes; a frame without a flowed copy is REFUSED so a
    # half-flowed cell can never be searched half-registered.
    if flow_select and flow_applies(subband) and stage_root and os.path.exists(os.path.join(stage_root, ".no_master")):
        # HOURLY-SCREEN FALLBACK RETIRED (Gregg 2026-10-04: "the cube is flow corrected ... What screen is needed?!",
        # then "go on all three"): every frame is flow-aligned to the Master or the subband is not searched.  The
        # fallback searched 24 of 6,142 fleet subbands (Master missing or its link broken) in the hourly screen's frame,
        # where the fitter and the reference tie were never consistent (J1919-0132 served ~14.5 arcmin off).
        raise SystemExit("REFUSED: no Master for %s %s %s - every frame must be flow-aligned to the Master; the hourly "
                         "screen fallback is retired (2026-10-04)" % (lst, date, subband))
    elif flow_select and flow_applies(subband):
        # ZERO-VISIBILITY FRAMES ARE REFUSED, not missing (Gregg 2026-10-11 'You cant have a source if there are zero
        # visibilities'; 2026-10-12 '4. yes' for the snapshot fleet): a frame whose OWN header records zero gridded
        # visibilities (WSCNVIS = 0, an all-NaN / all-zero image of a Phase-1-flagged integration) cannot be flowed, so it is
        # dropped here with a count instead of refusing the whole subband for its missing _flowed.fits
        from astropy.io import fits as _fits
        nzero = 0
        for st in ("I", "V"):
            for ts, p in list(frames[st].items()):
                try:
                    _nv = _fits.getheader(p).get("WSCNVIS", 1)
                except Exception:
                    continue
                if _nv is not None and float(_nv) == 0:
                    del frames[st][ts]; nzero += 1
        if nzero:
            print("ZERO-VIS %s %s %s: %d frames with WSCNVIS = 0 refused (Gregg 2026-10-11/12)" % (lst, date, subband, nzero),
                  flush=True)
        missing = []
        for st in ("I", "V"):
            for ts, p in list(frames[st].items()):
                fp = p[:-5] + "_flowed.fits" if p.endswith(".fits") else None
                if fp and os.path.exists(fp):
                    frames[st][ts] = fp
                else:
                    missing.append((st, ts))
        if missing:
            raise RuntimeError("P3_SNAP_FLOW=1 but %d frames of %s %s %s have no _flowed.fits "
                               "(first: %s) - REFUSING" % (len(missing), lst, date, subband, missing[0]))
    return {st: sorted(frames[st].items()) for st in ("I", "V")}


_FUNPACK = "/opt/devel/pipeline/envs/development/bin/funpack"

# IN-PROCESS FUNPACK (2026-10-18, speed only; refactor-chat ruling 2026-10-18, veto fix 1).  Every funpack fallback below
# forks this process; forked from a ~30 GB chain worker that fork cost ~1.2 s per frame (78% of the veto pre-computation).
# With FUNPACK_INPROC set, the same bytes are made in this process by the library funpack itself links
# (ldd funpack -> <env>/lib/libcfitsio.so.8, CFITSIO 3.470, built reentrant) with funpack -S's own calls: per HDU
# fits_img_decompress + fits_write_chksum (a compressed image) or fits_copy_hdu, then fits_write_chksum on the primary,
# into a memory file.  Verified byte-identical to funpack -S stdout (apart from the time stamp in the CHECKSUM comments,
# which differs between two funpack runs as well) on frames of every band, Stokes I and V, .fs and .fz.  Off by default:
# the veto pre-computation's workers turn it on; any cfitsio error raises (no silent fallback to the subprocess).
FUNPACK_INPROC = False
_CFITSIO = {}


def _funpack_bytes_inproc(path):
    """funpack -S's output for path, as bytes, made in this process (see IN-PROCESS FUNPACK)."""
    import ctypes as _ct
    if not _CFITSIO:
        lib = _ct.CDLL(os.path.join(os.path.dirname(os.path.dirname(_FUNPACK)), "lib", "libcfitsio.so.8"))
        libc = _ct.CDLL(None)
        libc.free.argtypes = [_ct.c_void_p]
        _CFITSIO.update(lib=lib, libc=libc,
                        realloc=_ct.CFUNCTYPE(_ct.c_void_p, _ct.c_void_p, _ct.c_size_t)(("realloc", libc)))
    L, st = _CFITSIO["lib"], _ct.c_int(0)
    inf, outf = _ct.c_void_p(), _ct.c_void_p()
    buf, bsz = _ct.c_void_p(None), _ct.c_size_t(0)
    try:
        L.ffopen(_ct.byref(inf), path.encode(), 0, _ct.byref(st))
        L.ffimem(_ct.byref(outf), _ct.byref(buf), _ct.byref(bsz), _ct.c_size_t(0), _CFITSIO["realloc"], _ct.byref(st))
        while st.value == 0:
            if L.fits_is_compressed_image(inf, _ct.byref(st)):
                L.fits_img_decompress(inf, outf, _ct.byref(st))
                L.ffpcks(outf, _ct.byref(st))
            else:
                L.ffcopy(inf, outf, 0, _ct.byref(st))
            L.ffmrhd(inf, 1, None, _ct.byref(st))
        if st.value == 107:                       # END_OF_FILE: past the last HDU
            st.value = 0
        L.ffmahd(outf, 1, None, _ct.byref(st))
        L.ffpcks(outf, _ct.byref(st))
        L.ffflus(outf, _ct.byref(st))
        nh, hs, ds, de = _ct.c_int(0), _ct.c_longlong(), _ct.c_longlong(), _ct.c_longlong()
        L.ffthdu(outf, _ct.byref(nh), _ct.byref(st))
        L.ffmahd(outf, nh.value, None, _ct.byref(st))
        L.ffghadll(outf, _ct.byref(hs), _ct.byref(ds), _ct.byref(de), _ct.byref(st))
        if st.value:
            raise RuntimeError("in-process funpack of %s: CFITSIO status %d" % (path, st.value))
        size = (de.value + 2879) // 2880 * 2880
        if outf.value:
            s2 = _ct.c_int(0); L.ffclos(outf, _ct.byref(s2)); outf.value = None
        return _ct.string_at(buf.value, size)
    finally:
        s2 = _ct.c_int(0)
        if outf.value:
            L.ffclos(outf, _ct.byref(s2))
        if inf.value:
            L.ffclos(inf, _ct.byref(s2))
        if buf.value:
            _CFITSIO["libc"].free(buf.value)


# FRAME READ MEMO (2026-10-18, speed only; refactor-chat ruling 2026-10-18, veto fix 2): None = off (every caller but the
# veto pre-computation's workers, which set an OrderedDict).  When on, read_frame returns the (data, header) it read before
# for the same path, the READ_MEMO_MAX most recently used kept: a frame is opened and decompressed once however many
# sources, instants, 1-minute blocks and wideband frames use it.  Callers return the cached objects; none writes into them
# (registration, residuals, block means and dewarps all make new arrays; headers are copied before a card is set).
READ_MEMO = None
READ_MEMO_MAX = 16


def read_frame(path):
    """read_frame_uncached through READ_MEMO when it is on."""
    m = READ_MEMO
    if m is None:
        return read_frame_uncached(path)
    if path in m:
        m.move_to_end(path)
        return m[path]
    v = read_frame_uncached(path)
    m[path] = v
    while len(m) > READ_MEMO_MAX:
        m.popitem(last=False)
    return v


def read_frame_uncached(path):
    """Read one snapshot; return (2d float32 array, header).

    The py3.6 astropy in the development env cannot decompress
    tile-compressed HDUs with 4 axes ("only 1D, 2D, or 3D images are
    currently supported"); fall back to funpack streaming to stdout."""
    try:
        with fits.open(path, memmap=False) as hdul:
            hdu = hdul[1] if len(hdul) > 1 and hdul[0].data is None else hdul[0]
            data = np.squeeze(hdu.data).astype(np.float32)
            return data, hdu.header
    except RuntimeError:
        import io as _io
        import subprocess
        exe = _FUNPACK if os.path.exists(_FUNPACK) else "funpack"
        out = _funpack_bytes_inproc(path) if FUNPACK_INPROC else subprocess.check_output([exe, "-S", path])
        with fits.open(_io.BytesIO(out), memmap=False) as hdul:
            hdu = hdul[1] if len(hdul) > 1 and hdul[0].data is None else hdul[0]
            data = np.squeeze(hdu.data).astype(np.float32)
            return data, hdu.header


def load_cube(frame_list, nthreads=8):
    """Load [T, Y, X] float32 cube + per-frame metadata."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    first, hdr0 = read_frame(frame_list[0][1])
    ny, nx = first.shape
    nt = len(frame_list)
    cube = np.empty((nt, ny, nx), dtype=np.float32)
    cube[0] = first
    meta = [None] * nt
    meta[0] = _frame_meta(frame_list[0][0], hdr0, first)

    def _load(i):
        d, h = read_frame(frame_list[i][1])
        cube[i] = d
        meta[i] = _frame_meta(frame_list[i][0], h, d)

    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        list(ex.map(_load, range(1, nt)))
    # CUBE HEADER FROM THE FIRST VALID FRAME (fix 2026-09-09, FIX-SNAP-CUBE-HEADER-
    # FROM-FIRST-VALID-FRAME): a blank CLEAN failure archived as frame 0 carries
    # BMAJ = BMIN = 0, and that header was returned for the whole cube - every
    # injection stamp at 69 and 82 MHz Stokes I on 03h 2024-12-28 was therefore
    # zero (SNAP-BAT-2 first cell), and the beam-derived clustering tolerance was
    # wrong.  Take the header of the first frame that is mostly finite and has a
    # positive beam; WCS is common to all frames of a subband.
    if not (np.isfinite(meta[0]["bmaj"]) and meta[0]["bmaj"] > 0 and meta[0].get("finite_frac", 1.0) >= 0.5):
        for i in range(nt):
            mi = meta[i]
            if np.isfinite(mi["bmaj"]) and mi["bmaj"] > 0 and mi.get("finite_frac", 1.0) >= 0.5:
                hdr0 = read_frame(frame_list[i][1])[1]
                log.warning("cube header taken from frame %d (%s): frame 0 is blank or has no beam",
                            i, frame_list[i][0])
                break
    log.info("loaded cube %s in %.1f s (%.1f GB)", cube.shape,
             time.time() - t0, cube.nbytes / 2**30)
    return cube, meta, hdr0


def _frame_meta(ts, hdr, data=None):
    m = {"ts": ts, "date_obs": hdr.get("DATE-OBS", ""),
         "bmaj": float(hdr.get("BMAJ", np.nan)),
         "bmin": float(hdr.get("BMIN", np.nan))}
    if data is not None:
        # blank-frame guard: the CLEANed generation contains fully-NaN
        # frames (failed deconvolutions archived empty); one such frame
        # poisons a plain median at every pixel
        sub = data[::8, ::8]
        fin = np.isfinite(sub)
        m["finite_frac"] = float(fin.mean())
        # no-data frame guard (Gregg 2026-09-12, "measure and remove them before
        # entering the median"): an integration with zero gridded visibilities
        # (Phase-1 per-scan QA flag, WSCNVIS=0) is archived as an all-NaN CLEANed
        # Stokes I frame but as an all-ZERO finite Stokes V dirty frame, which the
        # finite-fraction test cannot see; 92 of 358 V frames on 04h 2024-12-28
        # 18 MHz entered the hour median that way
        m["all_zero"] = bool(fin.any() and not np.any(sub[fin] != 0))
    return m


# ---------------------------------------------------------------- admission

def central_rms(img, frac=0.25, mask=None):
    """Robust rms (MAD*1.4826) over the central 25% of the image AREA
    (optionally restricted to mask=True pixels)."""
    ny, nx = img.shape
    h = int(ny * np.sqrt(frac) / 2)
    w = int(nx * np.sqrt(frac) / 2)
    sl = (slice(ny // 2 - h, ny // 2 + h), slice(nx // 2 - w, nx // 2 + w))
    c = img[sl]
    ok = np.isfinite(c)
    if mask is not None:
        ok &= mask[sl]
    c = c[ok]
    if c.size < 1000:
        return float("nan")
    med = np.median(c)
    return float(1.4826 * np.median(np.abs(c - med)))


# NIGHT RULE (Gregg 2026-09-26 "Avoid day time, sunrise and sunset!"; refined 2026-09-27 after the deep chat's solar
# audit, CARD_SOLAR_CONTAMINATION_FOR_SNAPSHOT_20260926.md - "1. agree. 2. agree. 3. refuse."):
#   1. a frame with the Sun ABOVE THE HORIZON is refused in every band ('sun-up'): no noise test catches daytime
#      ionospheric lensing (J2026-1557, 19h 2025-04-10, Sun +1.5..+6 deg, passes the audit);
#   2. a TWILIGHT frame (Sun between 0 and -12 deg rising, 0 and -18 deg setting) is admitted unless the solar audit
#      (/lustre/pipeline/exopipe/phase3/QA/solar_contamination_audit.csv, per LST, night and subband below 46 MHz) marks
#      that night's band QUARANTINE ('twilight-flagged');
#   3. ... or 'suspect' (also refused).  A band below 46 MHz with no audit row is refused ('twilight-unaudited': the
#      gate cannot be checked, so it refuses).  A wideband group is refused if any constituent below 46 MHz is.
#   SUPERSEDED IN PART 2026-10-13 (Gregg, on the twilight cells: "The same elevation cutoff applied to deep, applies here
#   too."; criteria "everything else approved"): the rule is now the DEEP tier's daytime gate (exopipe_ops/p3s2_conform.py,
#   DAYTIME GATE): Sun above the horizon refused in every band; twilight refused ONLY in a band/night the solar audit marks
#   QUARANTINE (a wideband when any constituent is).  'suspect' verdicts and bands with no audit row are ADMITTED, as deep
#   admits them; 'twilight-unaudited' is no longer returned.  The audit covers 18-41 MHz only, so the 46 MHz bound is unchanged.
# Shared by the snapshot search (admission + hour median), the snapshot chain (domain_gate) and the sub-LST stacks.
TWILIGHT_MORNING_DEG = -12.0
TWILIGHT_EVENING_DEG = -18.0
SOLAR_AUDIT = os.environ.get("P3_SOLAR_AUDIT", "/lustre/pipeline/exopipe/phase3/QA/solar_contamination_audit.csv")
SOLAR_REFUSE = ("QUARANTINE",)          # 2026-10-13: the deep gate refuses QUARANTINE only (was QUARANTINE + suspect)
SOLAR_AUDIT_BELOW_MHZ = 46
_NIGHT_GROUPS = {"Red": ["18MHz", "23MHz", "27MHz", "32MHz", "36MHz"], "Green": ["41MHz", "46MHz", "50MHz", "55MHz", "59MHz"],
                 "Blue": ["64MHz", "69MHz", "73MHz", "78MHz", "82MHz"]}
_NIGHT_GROUPS["Full"] = _NIGHT_GROUPS["Red"] + _NIGHT_GROUPS["Green"] + _NIGHT_GROUPS["Blue"]
_SUN_CACHE = {}
_AUDIT = None


def sun_state(ts):
    """(Sun elevation deg, 'morning'|'evening', in_twilight_or_day bool) at OVRO for a YYYYMMDD_HHMMSS timestamp."""
    if ts in _SUN_CACHE:
        return _SUN_CACHE[ts]
    from astropy.time import Time
    from astropy.coordinates import get_sun, EarthLocation, AltAz, SkyCoord
    import astropy.units as u
    loc = EarthLocation(lat=37.2398 * u.deg, lon=-118.2817 * u.deg, height=1222 * u.m)
    t = Time("%s-%s-%sT%s:%s:%s" % (ts[:4], ts[4:6], ts[6:8], ts[9:11], ts[11:13], ts[13:15]), scale="utc")
    sn = get_sun(t); sc = SkyCoord(sn.ra, sn.dec)
    alt = float(sc.transform_to(AltAz(obstime=t, location=loc)).alt.deg)
    ha = ((float(t.sidereal_time("apparent", longitude=loc.lon).deg) - sc.ra.deg + 180.0) % 360.0) - 180.0
    br = "evening" if ha > 0 else "morning"
    out = (alt, br, alt >= (TWILIGHT_EVENING_DEG if br == "evening" else TWILIGHT_MORNING_DEG))
    _SUN_CACHE[ts] = out
    return out


def _solar_audit():
    global _AUDIT
    if _AUDIT is None:
        if not os.path.exists(SOLAR_AUDIT):
            raise RuntimeError("REFUSING: solar audit %s not found (the twilight rule cannot be applied)" % SOLAR_AUDIT)
        with open(SOLAR_AUDIT) as fh:
            _AUDIT = {(r["lst"], r["date"], r["subband"]): r["verdict"] for r in csv.DictReader(fh)}
    return _AUDIT


def night_verdict(ts, lst, date, subband):
    """'' (admitted) | 'sun-up' | 'twilight-flagged' for one frame of (lst, date, subband) ('twilight-unaudited' retired 2026-10-13)."""
    alt, br, zone = sun_state(ts)
    if alt >= 0.0:
        return "sun-up"
    if not zone:
        return ""
    for sb in _NIGHT_GROUPS.get(subband, [subband]):
        try:
            if int(sb[:2]) >= SOLAR_AUDIT_BELOW_MHZ:
                continue
        except ValueError:
            continue
        v = _solar_audit().get((lst, date, sb))
        if v is None:
            continue                    # 2026-10-13: no audit row = admitted, as in the deep gate (was 'twilight-unaudited')
        if v in SOLAR_REFUSE:
            return "twilight-flagged"
    return ""


def night_refused_mask(meta, lst, date, subband):
    return np.array([night_verdict(m["ts"], lst, date, subband) != "" for m in meta], dtype=bool)


# A BAND REFUSED BY THE NIGHT RULE IS ACCOUNTED FOR, NOT MISSING (Gregg 2026-10-13, ruling 4: "The same elevation cutoff applied
# to deep, applies here too" - the deep tier refuses a band's twilight frames and ships the cell's other bands).  When the night
# rule ALONE refuses every frame of a (subband or wideband group, Stokes) search, the search writes nightrefused_<tag>.json
# beside where runinfo_<tag>.json would go, and the fleet's completeness guard counts it in place of the runinfo.  Any other
# failure (blank / all-zero frames, a crash) writes no record and stays a failure.  The criteria (night_verdict) are unchanged.
NIGHT_REFUSED_RECORD = "nightrefused_%s.json"


class NightRuleRefused(RuntimeError):
    """Every frame of one search is refused by the night rule (night_verdict); .record is the refusal record."""
    def __init__(self, msg, record):
        RuntimeError.__init__(self, msg)
        self.record = record


def night_refusal_record(tss, lst, date, subband, stokes, mode):
    """The refusal record of one search whose every frame the night rule refuses, or None when any frame passes it."""
    verdicts = [night_verdict(ts, lst, date, subband) for ts in tss]
    if not verdicts or any(v == "" for v in verdicts):
        return None
    alts = [sun_state(ts)[0] for ts in tss]
    counts = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    return OrderedDict([
        ("lst", lst), ("date", date), ("subband", subband), ("stokes", stokes), ("mode", mode),
        ("status", "refused-night-rule"),
        ("rule", "night rule (phase3_snapshot_search.night_verdict): Sun above the horizon in every band; twilight (0 to %g deg "
                 "rising, 0 to %g deg setting) refused in a band below %d MHz the solar audit marks %s (no row = admitted, as the deep gate); a "
                 "wideband group refused when a constituent is" % (TWILIGHT_MORNING_DEG, TWILIGHT_EVENING_DEG,
                                                                 SOLAR_AUDIT_BELOW_MHZ, "/".join(SOLAR_REFUSE))),
        ("solar_audit", SOLAR_AUDIT),
        ("n_frames", len(tss)), ("verdict_counts", counts),
        ("sun_elevation_deg_min", round(min(alts), 2)), ("sun_elevation_deg_max", round(max(alts), 2)),
        ("branch", sorted({sun_state(ts)[1] for ts in tss})),
        ("first_ts", min(tss)), ("last_ts", max(tss)),
        ("accounting", "a recorded refusal: the fleet completeness guard counts this file in place of runinfo_%s_%s_%s.json "
                       "(Gregg 2026-10-13, ruling 4)" % (subband, stokes, mode)),
    ])


def write_night_refusal(out_dir, record):
    p = os.path.join(out_dir, NIGHT_REFUSED_RECORD % ("%s_%s_%s" % (record["subband"], record["stokes"], record["mode"])))
    with open(p + ".tmp", "w") as f:
        json.dump(record, f, indent=2)
    os.replace(p + ".tmp", p)
    log.warning("NIGHT-RULE REFUSAL RECORDED: %s (%s)", p, record["verdict_counts"])
    return p


def admission_table(v_cube, meta_i, meta_v, rms_outlier_k=5.0,
                    beam_area_ratio_max=1.5, ctx=None):
    """Per-frame QA.  Returns (rows, admitted_bool_array, vrms).

    V central-25% rms MAD-outlier flag + beam-area gate on BOTH Stokes
    headers.  A frame is admitted only if it passes everything.
    """
    nt = v_cube.shape[0]
    vrms = np.array([central_rms(v_cube[i]) for i in range(nt)])
    fin = np.isfinite(vrms)
    med = np.median(vrms[fin]) if fin.any() else np.nan
    mad = np.median(np.abs(vrms[fin] - med)) if fin.any() else np.nan
    rms_bad = ~fin | (np.abs(vrms - med) > rms_outlier_k * 1.4826 * max(mad, 1e-12))

    area_i = np.array([m["bmaj"] * m["bmin"] for m in meta_i]) if meta_i else None
    area_v = np.array([m["bmaj"] * m["bmin"] for m in meta_v])
    beam_bad = np.zeros(nt, dtype=bool)
    for area in (a for a in (area_i, area_v) if a is not None and len(a) == nt):
        good = np.isfinite(area)
        if good.any():
            ratio = area / np.median(area[good])
            beam_bad |= ratio > beam_area_ratio_max

    # RULING (Gregg 2026-09-09): frame noise is judged PER STOKES on the
    # median-subtracted frame (MAD), in detect() below.  The raw-frame Stokes V
    # rms outlier is recorded here for the table but no longer gates admission
    # (it gated BOTH Stokes from the raw V frame until 2026-09-09).
    if ctx is None:
        raise RuntimeError("admission_table: ctx=(lst, date, subband) is required for the night rule")
    nverd = [night_verdict(m["ts"], *ctx) for m in meta_v]
    sun_up = np.array([v != "" for v in nverd], dtype=bool)
    admitted = ~beam_bad & ~sun_up
    rows = []
    for i in range(nt):
        rows.append(OrderedDict([
            ("idx", i), ("ts", meta_v[i]["ts"]),
            ("date_obs", meta_v[i]["date_obs"]),
            ("v_central_rms_jy", "%.6g" % vrms[i]),
            ("rms_outlier", int(rms_bad[i])),
            ("beam_refused", int(beam_bad[i])),
            ("sun_elev_deg", "%.2f" % sun_state(meta_v[i]["ts"])[0]),
            ("night_refused", nverd[i]),
            ("admitted", int(admitted[i]))]))
    log.info("admission: %d/%d admitted (%d rms outliers, %d beam refusals, %d sun-up refusals)",
             admitted.sum(), nt, rms_bad.sum(), beam_bad.sum(), sun_up.sum())
    return rows, admitted, vrms


# ---------------------------------------------------------------- domain

def elevation_masks(hdr, times_epoch, nbins=N_TIME_BINS,
                    min_el=MIN_ELEVATION_DEG, shape=None):
    """Per-time-bin on-sky + elevation>=min_el masks on the image grid.

    Returns (masks[list of bool arrays], bin_of[int array per frame],
    bin_lst_deg[list]).  Elevation is evaluated from the pixel sky
    position and the local sidereal time at each bin's mid-time."""
    w = WCS(hdr).celestial
    if shape is None:
        shape = (int(hdr["NAXIS2"]), int(hdr["NAXIS1"]))
    ny, nx = shape
    ra = np.empty(shape, dtype=np.float32)
    dec = np.empty(shape, dtype=np.float32)
    chunk = 256
    for r in range(0, ny, chunk):
        gy, gx = np.mgrid[r:min(r + chunk, ny), 0:nx]
        a, d = w.all_pix2world(gx, gy, 0)
        ra[r:r + gy.shape[0]] = a
        dec[r:r + gy.shape[0]] = d
    onsky = np.isfinite(ra) & np.isfinite(dec)
    times = np.asarray(times_epoch, dtype=float)
    t0, t1 = float(times.min()), float(times.max())
    span = max(t1 - t0, 1.0)
    bin_of = np.minimum((np.floor((times - t0) / span * nbins)).astype(int),
                        nbins - 1)
    masks, lsts = [], []
    sin_lat, cos_lat = np.sin(np.radians(OVRO_LAT_DEG)), np.cos(np.radians(OVRO_LAT_DEG))
    sin_dec = np.sin(np.radians(dec))
    cos_dec = np.cos(np.radians(dec))
    for b in range(nbins):
        tmid = t0 + (b + 0.5) * span / nbins
        lst = lst_deg_at(tmid)
        lsts.append(lst)
        ha = np.radians(ra - lst)
        sin_el = sin_dec * sin_lat + cos_dec * cos_lat * np.cos(ha)
        el = np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))
        masks.append(onsky & (el >= min_el))
    log.info("domain: on-sky %.1f%% of grid; elevation>=%.0f deg %.1f%%-%.1f%% "
             "across %d time bins", 100 * onsky.mean(), min_el,
             100 * min(m.mean() for m in masks),
             100 * max(m.mean() for m in masks), nbins)
    return masks, bin_of, lsts


# ---------------------------------------------------------------- sky model

def _threaded_rows(fn, ny, chunk, nthreads):
    from concurrent.futures import ThreadPoolExecutor
    ranges = [(r, min(r + chunk, ny)) for r in range(0, ny, chunk)]
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        list(ex.map(lambda rr: fn(*rr), ranges))


def chunked_median(cube, chunk=128, nthreads=6):
    ny = cube.shape[1]
    out = np.empty(cube.shape[1:], dtype=np.float32)

    def _do(r0, r1):
        out[r0:r1] = np.median(cube[:, r0:r1, :], axis=0)

    _threaded_rows(_do, ny, chunk, nthreads)
    return out


def highpass_blocks(blocks, size=65):
    """Remove smooth spatial structure (primary-beam drift, ionospheric
    focusing) from residual images while leaving point sources (~7 px
    beam) intact.  NaN-aware boxcar of `size` pixels (~2 deg)."""
    from scipy.ndimage import uniform_filter
    out = np.empty_like(blocks)
    for i in range(blocks.shape[0]):
        r = blocks[i]
        finite = np.isfinite(r)
        rz = np.where(finite, r, 0).astype(np.float32)
        norm = uniform_filter(finite.astype(np.float32), size)
        sm = uniform_filter(rz, size) / np.maximum(norm, 1e-3)
        out[i] = np.where(finite, r - sm, np.nan)
    return out


FLOW_MODE = os.environ.get("P3_SNAP_FLOW", "0") == "1"   # per-frame flow to the master (Gregg 2026-09-12: pipeline product)
# Bands that use the flowed frames when FLOW_MODE is on (measured 2026-09-12 on 02h 2024-12-29: per-frame flow
# beats the hourly screen at 18-69 MHz and degrades registration at 73-82 MHz).  Default = every science band
# (as ruled); set P3_SNAP_FLOW_MAX_MHZ to cap it (e.g. 69) once Gregg rules on the split.  Bands above the cap
# are searched with the hourly screen exactly as before.
FLOW_MAX_MHZ = float(os.environ.get("P3_SNAP_FLOW_MAX_MHZ", "1000"))


_NO_MASTER = set()            # subbands searched with the hourly screen because no master exists (marker .no_master)


def flow_applies(subband):
    if subband in _NO_MASTER:
        return False
    try:
        return FLOW_MODE and float(str(subband).replace("MHz", "")) <= FLOW_MAX_MHZ
    except ValueError:
        return FLOW_MODE          # wideband groups follow their members


def load_warp_screen(lst, date, subband):
    """Return (screen_x, screen_y) pixel-offset maps, or (None, None)."""
    pats = glob.glob(os.path.join(
        PHASE2_ROOT, lst, date, "Science_*", "Dewarp_Diagnostics", "maps",
        "%s_%s_%s_warp_screen.fits" % (date, lst, subband)))
    if not pats:
        return None, None
    with fits.open(sorted(pats)[-1]) as h:
        return (h["SCREEN_X"].data.astype(np.float32),
                h["SCREEN_Y"].data.astype(np.float32))


def master_in_raw_frame(master_path, snap_hdr, screen_x, screen_y,
                        chunk=256):
    """Warp the dewarped deep master INTO the hour's ionospheric frame,
    on the snapshot grid.  Raw-frame pixel q sees the sky that dewarping
    would place at p ~ q - screen(q); we sample the master there.
    """
    mdata = np.squeeze(fits.getdata(master_path)).astype(np.float32)
    mhdr = fits.getheader(master_path)
    wsnap = WCS(snap_hdr).celestial
    wmast = WCS(mhdr).celestial
    ny, nx = screen_x.shape
    out = np.empty((ny, nx), dtype=np.float32)
    for r in range(0, ny, chunk):
        gy, gx = np.mgrid[r:min(r + chunk, ny), 0:nx]
        px = gx - screen_x[r:r + chunk][:gy.shape[0]]
        py = gy - screen_y[r:r + chunk][:gy.shape[0]]
        ra, dec = wsnap.all_pix2world(px, py, 0)
        mx, my = wmast.all_world2pix(ra, dec, 0)
        bad = ~(np.isfinite(mx) & np.isfinite(my))
        mx = np.where(bad, -1e6, mx)
        my = np.where(bad, -1e6, my)
        out[r:r + gy.shape[0]] = ndimage.map_coordinates(
            mdata, [my, mx], order=1, mode="constant", cval=np.nan)
    return out


def robust_scale(model_a, model_b, lo=5.0):
    """Median ratio a/b over mutually finite, bright pixels of b."""
    m = np.isfinite(model_a) & np.isfinite(model_b)
    s = 1.4826 * np.median(np.abs(model_b[m] - np.median(model_b[m])))
    bright = m & (np.abs(model_b) > lo * max(s, 1e-12))
    if bright.sum() < 100:
        return 1.0, int(bright.sum())
    return float(np.median(model_a[bright] / model_b[bright])), int(bright.sum())


# ---------------------------------------------------------------- detection

NOISE_OUTLIER_K = 5.0   # per-Stokes residual-MAD frame refusal (Gregg 2026-09-09)


def detect(cube, model, admitted, meta, thresh, stokes, masks, bin_of,
           flood_rel=FLOOD_REL, flood_min_abs=FLOOD_MIN_ABS,
           flood_abs_max=FLOOD_ABS_MAX, hard_cap=HARD_CAP_COMPONENTS,
           noise_outlier_k=NOISE_OUTLIER_K):
    """Threshold (image - model) / global_rms(image) per admitted image on
    the per-bin elevation domain.

    Stokes V searches |snr| (circular polarization has sign); Stokes I
    positive only.  Returns (candidate rows, per-image noise rows,
    admitted_after_flood, flood_info).  The flood refusal is evaluated
    after all images are measured: an image whose above-threshold
    component count exceeds max(flood_min_abs, flood_rel x median count)
    (limit capped at flood_abs_max) is REFUSED and its detections dropped.
    """
    nt = cube.shape[0]
    cands_by_t = {}
    rows = []
    for t in range(nt):
        row = OrderedDict([
            ("idx", t), ("ts", meta[t]["ts"]),
            ("date_obs", meta[t].get("date_obs", "")),
            ("elev_bin", int(bin_of[t])),
            ("n_subbands", meta[t].get("n_subbands", "")),
            ("wsum", ("%.6g" % meta[t]["wsum"]) if "wsum" in meta[t] else ""),
            ("admitted", int(bool(admitted[t]))),
            ("global_rms_jyb", ""), ("n_components", ""),
            ("flood_refused", 0), ("refuse_reason", "")])
        if not admitted[t]:
            row["refuse_reason"] = "admission"
            rows.append(row)
            continue
        mask = masks[bin_of[t]]
        resid = cube[t] - model
        sig = central_rms(resid, mask=mask)
        if not (np.isfinite(sig) and sig > 0):
            row["admitted"] = 0
            row["refuse_reason"] = "no-noise-estimate"
            admitted[t] = False
            rows.append(row)
            continue
        row["global_rms_jyb"] = "%.6g" % sig
        snr_img = resid / np.float32(sig)
        work = np.abs(snr_img) if stokes == "V" else snr_img
        work = np.where(mask & np.isfinite(work), work, 0).astype(np.float32)
        m = work > thresh
        lab, nlab = ndimage.label(m)
        row["n_components"] = nlab
        if nlab > hard_cap:
            row["flood_refused"] = 1
            row["refuse_reason"] = "hard-cap"
            row["admitted"] = 0
            admitted[t] = False
            rows.append(row)
            continue
        if nlab:
            index = np.arange(1, nlab + 1)
            pos = ndimage.maximum_position(work, lab, index=index)
            npix = ndimage.sum(m, lab, index=index)
            cl = []
            for k, (py, px) in enumerate(pos):
                cl.append({
                    "t": t, "ts": meta[t]["ts"],
                    "date_obs": meta[t].get("date_obs", ""),
                    "x": int(px), "y": int(py),
                    "snr": float(snr_img[py, px]),
                    "peak_jyb": float(resid[py, px]),
                    "npix": int(npix[k]),
                    "sigma_frame_jyb": float(sig),
                    "elev_bin": int(bin_of[t]),
                })
            cands_by_t[t] = cl
        rows.append(row)
    # ---- noise-outlier refusal (RULING, Gregg 2026-09-09: Stokes I noise is
    # measured on the Stokes I median-subtracted frame, Stokes V on the Stokes V
    # median-subtracted frame, both as MAD).  sig above is exactly that
    # quantity (central 25 per cent of the elevation domain, MAD x 1.4826); a
    # frame whose sig is more than NOISE_OUTLIER_K robust sigmas above the
    # median sig of the admitted frames is REFUSED and its detections dropped.
    sigs = np.array([float(r["global_rms_jyb"]) for r in rows
                     if r["admitted"] and r["global_rms_jyb"] != ""], dtype=float)
    n_noise = 0
    if len(sigs) >= 10:
        smed = float(np.median(sigs))
        smad = float(1.4826 * np.median(np.abs(sigs - smed)))
        lim_sig = smed + noise_outlier_k * max(smad, 1e-12)
        for r in rows:
            if r["admitted"] and r["global_rms_jyb"] != "" and float(r["global_rms_jyb"]) > lim_sig:
                r["refuse_reason"] = "noise-outlier"
                r["admitted"] = 0
                admitted[r["idx"]] = False
                cands_by_t.pop(r["idx"], None)
                n_noise += 1
    else:
        smed = smad = lim_sig = float("nan")
    # ---- flood refusal
    counts = np.array([r["n_components"] for r in rows
                       if r["admitted"] and r["n_components"] != ""], dtype=float)
    med = float(np.median(counts)) if len(counts) else 0.0
    limit = min(max(flood_min_abs, flood_rel * med), flood_abs_max)
    n_flood = 0
    for r in rows:
        if r["admitted"] and r["n_components"] != "" and r["n_components"] > limit:
            r["flood_refused"] = 1
            r["refuse_reason"] = "flood"
            r["admitted"] = 0
            admitted[r["idx"]] = False
            cands_by_t.pop(r["idx"], None)
            n_flood += 1
    cands = [c for t in sorted(cands_by_t) for c in cands_by_t[t]]
    info = {"flood_median_components": med, "flood_limit": limit,
            "n_flood_refused": n_flood,
            "noise_median_jyb": smed, "noise_mad_jyb": smad,
            "noise_outlier_limit_jyb": lim_sig, "n_noise_refused": n_noise,
            "n_images_admitted_after_flood": int(np.sum(admitted))}
    log.info("raw detections >%.1f sigma: %d on %d images (median %.0f "
             "components/image, flood limit %.0f, %d flood-refused, "
             "%d noise-outlier-refused at > %.4g Jy/beam)",
             thresh, len(cands), info["n_images_admitted_after_flood"],
             med, limit, n_flood, n_noise, lim_sig)
    return cands, rows, admitted, info


def motion_flags(cands, tol_pix):
    """For each candidate, was there a >thresh peak within tol_pix in the
    ADJACENT images (t±1)?  Persistence vs single-image discriminator."""
    by_t = {}
    for c in cands:
        by_t.setdefault(c["t"], []).append((c["x"], c["y"]))
    for c in cands:
        n_adj = 0
        for dt in (-1, 1):
            pts = by_t.get(c["t"] + dt, [])
            for (x, y) in pts:
                if (x - c["x"]) ** 2 + (y - c["y"]) ** 2 <= tol_pix ** 2:
                    n_adj += 1
                    break
        c["n_adjacent_frames"] = n_adj
    return cands


def cluster_events(cands, tol_pix):
    """Union spatially coincident detections (any time) into harvest
    events.  The CHAIN (phase3_snapshot_chain) re-derives repeats from the
    candidates with the ruled definition; this clustering is a store-level
    convenience only."""
    if not cands:
        return []
    xy = np.array([[c["x"], c["y"]] for c in cands], dtype=float)
    tree = cKDTree(xy)
    pairs = tree.query_pairs(r=tol_pix)
    parent = list(range(len(cands)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra_, rb_ = find(a), find(b)
        if ra_ != rb_:
            parent[ra_] = rb_
    groups = {}
    for i in range(len(cands)):
        groups.setdefault(find(i), []).append(i)
    events = []
    for gid, idxs in groups.items():
        rows = [cands[i] for i in idxs]
        best = max(rows, key=lambda r: abs(r["snr"]))
        times = sorted(set(r["t"] for r in rows))
        events.append({
            "n_det": len(rows), "n_frames": len(times),
            "t_first": times[0], "t_last": times[-1],
            "x": best["x"], "y": best["y"],
            "best_snr": best["snr"], "best_peak_jyb": best["peak_jyb"],
            "best_sigma_frame_jyb": best["sigma_frame_jyb"],
            "best_ts": best["ts"], "elev_bin": best["elev_bin"],
            "member_ids": ";".join(str(i) for i in idxs),
        })
    events.sort(key=lambda e: -abs(e["best_snr"]))
    return events


# ------------------------------------------------------- position / PB

def attach_positions(rows, snap_hdr, screen_x, screen_y, bin_lsts=None):
    """Warp-corrected sky position per row; elevation at the row's time bin
    (for the astrometric error model) when bin_lsts is given."""
    w = WCS(snap_hdr).celestial
    sin_lat, cos_lat = np.sin(np.radians(OVRO_LAT_DEG)), np.cos(np.radians(OVRO_LAT_DEG))
    for r in rows:
        x, y = r["x"], r["y"]
        ra_raw, dec_raw = w.all_pix2world(float(x), float(y), 0)
        r["ra_raw_deg"] = float(ra_raw)
        r["dec_raw_deg"] = float(dec_raw)
        if screen_x is not None:
            dx = float(screen_x[y, x])
            dy = float(screen_y[y, x])
            ra_c, dec_c = w.all_pix2world(x - dx, y - dy, 0)
            r["warp_dx_pix"], r["warp_dy_pix"] = dx, dy
            r["ra_deg"], r["dec_deg"] = float(ra_c), float(dec_c)
            r["astrom_degraded"] = 0
        else:
            r["warp_dx_pix"] = r["warp_dy_pix"] = 0.0
            r["ra_deg"], r["dec_deg"] = float(ra_raw), float(dec_raw)
            r["astrom_degraded"] = 1
        if bin_lsts is not None and np.isfinite(r["ra_deg"]):
            ha = np.radians(r["ra_deg"] - bin_lsts[int(r.get("elev_bin", 0))])
            d = np.radians(r["dec_deg"])
            s = np.sin(d) * sin_lat + np.cos(d) * cos_lat * np.cos(ha)
            r["elevation_deg"] = float(np.degrees(np.arcsin(np.clip(s, -1, 1))))
        else:
            r["elevation_deg"] = float("nan")
    return rows


_PB_CACHE = {}


def pb_value(subband, tbin, x, y, snap_shape):
    """Primary-beam value at snapshot pixel (pb_library is LST-independent
    in pixel space); nearest-pixel with grid-ratio scaling."""
    key = (subband, tbin)
    if key not in _PB_CACHE:
        p = os.path.join(PB_LIBRARY, "PB_%s_bin%d.fits" % (subband, tbin))
        _PB_CACHE[key] = (np.squeeze(fits.getdata(p)).astype(np.float32)
                          if os.path.exists(p) else None)
    pb = _PB_CACHE[key]
    if pb is None:
        return np.nan
    sy = pb.shape[0] / float(snap_shape[0])
    sx = pb.shape[1] / float(snap_shape[1])
    yy = min(int(round(y * sy)), pb.shape[0] - 1)
    xx = min(int(round(x * sx)), pb.shape[1] - 1)
    return float(pb[yy, xx])


def attach_pb(rows, subband, snap_shape):
    for r in rows:
        tbin = min(5, int(r.get("elev_bin", 0)))
        v = pb_value(subband, tbin, r["x"], r["y"], snap_shape)
        r["pb"] = v
        r["peak_jyb_pbcorr"] = (r["peak_jyb"] / v
                                if np.isfinite(v) and v > 0.05 else np.nan)
    return rows


# ---------------------------------------------------------------- driver

def write_csv(path, rows):
    if not rows:
        open(path, "w").write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        wcsv = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow(r)


def save_map(path, arr, ref_hdr, comment):
    h = fits.PrimaryHDU(arr)
    for k in ("CTYPE1", "CRPIX1", "CRVAL1", "CDELT1", "CUNIT1",
              "CTYPE2", "CRPIX2", "CRVAL2", "CDELT2", "CUNIT2", "BUNIT"):
        if k in ref_hdr:
            h.header[k] = ref_hdr[k]
    h.header["COMMENT"] = comment
    h.writeto(path, overwrite=True)


def _align_by_ts(meta_ref, admitted_ref, meta_new):
    """Map the reference admission verdicts onto a new frame list by
    timestamp; frames absent from the reference are refused."""
    verdict = {m["ts"]: bool(admitted_ref[k]) for k, m in enumerate(meta_ref)}
    return np.array([verdict.get(m["ts"], False) for m in meta_new])


def run_search(lst, date, subband, stokes, mode, out_dir, thresh=5.0,
               stage_root=None, nthreads=8, inject=None):
    """stokes: 'V', 'I', or 'VI' (pair; V cube loaded once for admission
    and the V search, then the I cube reuses the admission).

    inject: optional SNAP-BAT manifest CSV; beam-shaped Gaussians are
    added to the in-memory cubes AFTER admission is computed (sparse
    injections must not perturb the frame-QA metric), archive untouched.
    Injection pixel bookkeeping is written per (subband, Stokes)."""
    # AN INJECTED RUN WRITES NO PRODUCTION PRODUCT (2026-10-24, INCIDENT-INJECTED-SEARCH-WROTE-PRODUCTION-PHOTOMETRY-CUTOUTS-
    # HS-PSC-EVENT-UNREPRODUCIBLE-20261024: the 2026-09-14 completeness run of 04h 2024-12-29 wrote its cutouts - synthetic sources
    # included - into the production photometry product, and the forced pages served them): the cutout writer refuses below
    if inject:
        os.environ["P3_SNAP_INJECTED"] = "1"
    else:
        os.environ.pop("P3_SNAP_INJECTED", None)
    os.makedirs(out_dir, exist_ok=True)
    frames = discover_frames(lst, date, subband, stage_root=stage_root)
    stokes_list = ["V", "I"] if stokes == "VI" else [stokes]
    if not frames["V"]:
        raise RuntimeError("no V frames (admission metric) for %s %s %s"
                           % (lst, date, subband))

    v_cube, meta_v, hdr_v = load_cube(frames["V"], nthreads=nthreads)
    adm_rows, admitted_v, _ = admission_table(v_cube, None, meta_v, ctx=(lst, date, subband))
    write_csv(os.path.join(out_dir, "admission_%s.csv" % subband), adm_rows)

    screen_x, screen_y = load_warp_screen(lst, date, subband)
    if flow_applies(subband):
        # frames were registered to the master per frame: the hourly screen
        # is replaced by the identity (positions, injections and cutouts
        # then read the master WCS directly)
        screen_x = np.zeros(v_cube.shape[1:], np.float32)
        screen_y = np.zeros(v_cube.shape[1:], np.float32)
        log.info("P3_SNAP_FLOW=1: frames are flow-registered to the master; identity screen")
    if screen_x is None:
        log.warning("NO WARP SCREEN for %s %s %s — positions degraded+flagged",
                    lst, date, subband)
    elif screen_x.shape != v_cube.shape[1:]:
        log.warning("warp screen grid %s != snapshot grid %s — position "
                    "correction skipped, degraded+flagged",
                    screen_x.shape, v_cube.shape[1:])
        screen_x = screen_y = None

    def _do_inject(cube, hdr, st):
        if not inject:
            return
        from phase3_snapshot_inject import inject_cube, load_manifest
        book = inject_cube(cube, hdr, screen_x, screen_y,
                           load_manifest(inject, st, subband), log=log)
        write_csv(os.path.join(out_dir, "injected_%s_%s.csv" % (subband, st)),
                  [OrderedDict([("inj_id", b[0]), ("px", "%.2f" % b[1]),
                                ("py", "%.2f" % b[2]), ("t0", b[3]),
                                ("t1", b[4])]) for b in book])

    results = {}
    refused = []        # night-rule refusals recorded (NightRuleRefused); the process still ends in a refusal after both Stokes
    if "V" in stokes_list:
        _do_inject(v_cube, hdr_v, "V")
        try:
            results["V"] = _search_one_stokes(
                v_cube, meta_v, hdr_v, admitted_v.copy(), "V", mode, lst, date,
                subband, out_dir, thresh, screen_x, screen_y)
        except NightRuleRefused as e:
            write_night_refusal(out_dir, e.record); refused.append(str(e))
    # the I frames' night verdicts read only their timestamps: when the night rule refuses every one, the I search would
    # refuse the same way after loading the cube - recorded without loading it
    rec_i = (night_refusal_record([ts for ts, _ in frames["I"]], lst, date, subband, "I", mode)
             if "I" in stokes_list and frames["I"] else None)
    if rec_i is not None:
        write_night_refusal(out_dir, rec_i)
        refused.append("REFUSED (night rule): no frame of %s %s %s I passes" % (lst, date, subband))
    elif "I" in stokes_list:
        if not frames["I"]:
            log.warning("no I frames for %s %s %s", lst, date, subband)
        else:
            del v_cube
            i_cube, meta_i, hdr_i = load_cube(frames["I"], nthreads=nthreads)
            _do_inject(i_cube, hdr_i, "I")
            admitted_i = _align_by_ts(meta_v, admitted_v, meta_i)
            # I-side beam gate from the I headers
            area = np.array([m["bmaj"] * m["bmin"] for m in meta_i])
            good = np.isfinite(area)
            if good.any():
                ratio = area / np.median(area[good])
                admitted_i &= ~(ratio > 1.5)
            log.info("I admission (aligned + I beam gate): %d/%d",
                     int(admitted_i.sum()), len(meta_i))
            try:
                results["I"] = _search_one_stokes(
                    i_cube, meta_i, hdr_i, admitted_i, "I", mode, lst, date,
                    subband, out_dir, thresh, screen_x, screen_y)
            except NightRuleRefused as e:
                write_night_refusal(out_dir, e.record); refused.append(str(e))
    if refused:     # the process still ends in a refusal, as before the record existed (exit status unchanged)
        raise RuntimeError("; ".join(refused) + " - refusal recorded (%s)" % (NIGHT_REFUSED_RECORD % "<subband>_<Stokes>_<mode>"))
    return results


CUBE_ROOT = os.environ.get("P3_SNAP_CUBE_ROOT", "/lustre/pipeline/exopipe/phase3/snapshot_cubes")


SEARCH_OUT_ROOT = os.environ.get("P3_SNAP_SEARCH_OUT_ROOT", "/lustre/gh/main/exopipe_ops/snapshot_fleet_out")


def flowcube_frame_index(lst, date, subband, stokes, tss, hdrs):
    """Indices of the cube frames the search used.  ONE FRAME PER INTEGRATION (2026-10-25: 66 of 10,214 consolidated cubes - 33
    (cell, subband) pairs, both Stokes, of 16 duplicate-run cells - hold the frames of TWO Phase-1 runs, ~half the frames repeating an
    integration's DATE-OBS; the search keeps one per integration, latest run wins, discover_frames): a cube without a repeated
    DATE-OBS is used whole, as before; one with repeats is restricted to the frames of the search's own frame list (its noise table
    in SEARCH_OUT_ROOT, cube order kept).  REFUSES when that list is missing or itself repeats an integration (the cell needs the
    duplicate-run re-run first)."""
    dob = [str(h.get("DATE-OBS", "")) for h in hdrs]
    if len(set(dob)) == len(dob):
        return list(range(len(tss)))
    p = os.path.join(SEARCH_OUT_ROOT, "%s_%s" % (lst, date), "noise_%s_%s_median.csv" % (subband, stokes))
    if not os.path.exists(p):
        raise RuntimeError("REFUSED: the %s %s %s %s cube repeats %d integrations and the search's frame list %s is missing"
                           % (lst, date, subband, stokes, len(dob) - len(set(dob)), p))
    with open(p, newline="") as fh:
        keep = {r["ts"] for r in csv.DictReader(fh)}
    idx = [i for i, t in enumerate(tss) if t in keep]
    kd = [dob[i] for i in idx]
    if len(set(kd)) != len(kd) or len(idx) != len(keep):
        raise RuntimeError("REFUSED: the search's own frame list of %s %s %s %s repeats integrations or names frames the cube lacks "
                           "(%d listed, %d in the cube, %d distinct DATE-OBS) - the cell needs the duplicate-run re-run first"
                           % (lst, date, subband, stokes, len(keep), len(idx), len(set(kd))))
    log.info("one frame per integration: %s %s %s %s cube %d frames -> the search's %d (%d repeated integrations dropped)",
             lst, date, subband, stokes, len(tss), len(idx), len(tss) - len(idx))
    return idx


def load_cube_from_flowcube(lst, date, subband, stokes):
    """The consolidated flow cube of (cell, subband, Stokes) as load_cube returns its frames: [T, Y, X] float32, per-frame meta from
    each frame's OWN header (FRAME_TABLE 'header', newline-separated cards) and the first valid frame's header (load_cube's rule).
    The cube's frames are the flowed frames the search reads (registered to the master per frame, on the Phase 1 hour grid); a cube
    holding two Phase-1 runs is restricted to the search's own frames (flowcube_frame_index)."""
    p = os.path.join(CUBE_ROOT, "%s_%s" % (lst, date), "%s_%s_%s_%s_flowcube.fits" % (lst, date, subband, stokes))
    t0 = time.time()
    with fits.open(p, memmap=True) as h:
        tab = h["FRAME_TABLE"].data
        tss = [str(x) for x in tab["ts"]]; hdrs = [fits.Header.fromstring(str(x), sep="\n") for x in tab["header"]]
        idx = flowcube_frame_index(lst, date, subband, stokes, tss, hdrs)
        cube = np.array(h["FRAMES"].data, dtype=np.float32)
        if len(idx) != len(tss):
            cube = np.ascontiguousarray(cube[idx]); tss = [tss[i] for i in idx]; hdrs = [hdrs[i] for i in idx]
    meta = [_frame_meta(tss[i], hdrs[i], cube[i]) for i in range(len(tss))]
    k0 = 0
    for i, m in enumerate(meta):
        if np.isfinite(m["bmaj"]) and m["bmaj"] > 0 and m.get("finite_frac", 1.0) >= 0.5:
            k0 = i; break
    log.info("loaded flow cube %s %s in %.1f s (%.1f GB)", os.path.basename(p), cube.shape, time.time() - t0, cube.nbytes / 2**30)
    return cube, meta, hdrs[k0]


def regen_cutouts(lst, date, subband, out_dir):
    """PHOTOMETRY CUTOUTS FROM THE CURRENT CUBES (2026-10-25; Gregg: "Forced photometry is also approved - HWJ and 10pc, then solar
    system planets"; INCIDENT-INJECTED-SEARCH-WROTE-PRODUCTION-PHOTOMETRY-CUTOUTS-HS-PSC-EVENT-UNREPRODUCIBLE-20261024): run_search's own
    path up to the cutouts on the consolidated flow cubes - V admission (admission_table), identity screens (the cubes are
    flow-registered), per Stokes the blank / zero / night rules, hour median, 10-s detection (its per-Stokes noise admission and noise
    rows) and write_cutouts to P3_SNAP_CUTOUT_ROOT - then stops (P3_SNAP_CUTOUTS_ONLY).  Never an injected run."""
    os.environ["P3_SNAP_CUTOUTS_ONLY"] = "1"; os.environ.pop("P3_SNAP_INJECTED", None); os.environ["P3_SNAP_FLOW"] = "1"
    if "P3_SNAP_CUTOUT_ROOT" not in os.environ:
        raise RuntimeError("REFUSED: regen_cutouts needs P3_SNAP_CUTOUT_ROOT (a new root - never over the existing product)")
    os.makedirs(out_dir, exist_ok=True)
    v_cube, meta_v, hdr_v = load_cube_from_flowcube(lst, date, subband, "V")
    adm_rows, admitted_v, _ = admission_table(v_cube, None, meta_v, ctx=(lst, date, subband))
    write_csv(os.path.join(out_dir, "admission_%s.csv" % subband), adm_rows)
    screen_x = np.zeros(v_cube.shape[1:], np.float32); screen_y = np.zeros(v_cube.shape[1:], np.float32)
    results, refused = {}, []
    try:
        results["V"] = _search_one_stokes(v_cube, meta_v, hdr_v, admitted_v.copy(), "V", "median", lst, date, subband, out_dir, 5.0,
                                          screen_x, screen_y)
    except NightRuleRefused as e:
        refused.append(str(e))
    del v_cube
    i_cube, meta_i, hdr_i = load_cube_from_flowcube(lst, date, subband, "I")
    admitted_i = _align_by_ts(meta_v, admitted_v, meta_i)
    area = np.array([m["bmaj"] * m["bmin"] for m in meta_i]); good = np.isfinite(area)
    if good.any():
        admitted_i &= ~(area / np.median(area[good]) > 1.5)
    try:
        results["I"] = _search_one_stokes(i_cube, meta_i, hdr_i, admitted_i, "I", "median", lst, date, subband, out_dir, 5.0,
                                          screen_x, screen_y)
    except NightRuleRefused as e:
        refused.append(str(e))
    log.info("CUTOUTS-REGEN %s %s %s: V %s I %s%s", lst, date, subband, (results.get("V") or {}).get("photometry_cutout_cubes"),
             (results.get("I") or {}).get("photometry_cutout_cubes"), ("; refused: " + "; ".join(refused)) if refused else "")
    return results


def build_minute_blocks(cube, meta, admitted, model):
    """1-minute blocks = admitted frames whose INTEGRATION MIDPOINT (header DATE-OBS + INTEGRATION_S / 2) lies in one
    wall-clock UTC minute (identical boundaries across subbands and widebands).  block_ts = YYYYMMDD_HHMM00; member_ts lists
    the members' frame identities (their file-name ts).  FRAME-TIME FIX, minimal form (Gregg 2026-10-07, "A few seconds timing
    error doesnt really matter. Go with your recommendation"): membership followed the FILE-NAME minute until 2026-10-07, and
    old names drift from the integration by up to 26 s, so 774 and 1,154 of the 1,830 subband-Stokes blocks of 19h 2025-05-03
    and 14h 2025-04-21 held frames of a neighbouring minute.  An ADMITTED frame without a readable DATE-OBS is REFUSED (the
    search raises: it cannot be placed in time); a refused frame without one counts in no minute.  Returns (blocks[array],
    blk_meta, blk_rows) where blk_rows records every minute incl. refused
    ones (< 3 admitted frames)."""
    minutes = OrderedDict()
    for i, m in enumerate(meta):
        try:
            key = integration_minute_key(m.get("date_obs", ""))
        except (ValueError, TypeError):
            if admitted[i]:
                raise RuntimeError("REFUSED: admitted frame %s has no readable DATE-OBS (%r) - its 1-minute block is chosen by "
                                   "its own integration time (frame-time fix, Gregg 2026-10-03/07)" % (m["ts"], m.get("date_obs")))
            continue
        minutes.setdefault(key, []).append(i)
    minutes = OrderedDict(sorted(minutes.items()))
    frames, bmeta, brows = [], [], []
    for key, idxs in minutes.items():
        sel = [i for i in idxs if admitted[i]]
        row = OrderedDict([("block_ts", key + "00"), ("n_frames_in_minute", len(idxs)),
                           ("n_frames_used", len(sel)),
                           ("member_ts", ";".join(meta[i]["ts"] for i in sel)),
                           ("admitted", int(len(sel) >= 3)),
                           ("refuse_reason", "" if len(sel) >= 3 else "lt3-frames"),
                           # filled after detection (every row carries the
                           # keys so the CSV header is complete)
                           ("elev_bin", ""), ("n_subbands", ""), ("wsum", ""),
                           ("global_rms_jyb", ""), ("n_components", ""),
                           ("flood_refused", 0)])
        brows.append(row)
        if len(sel) < 3:
            continue
        frames.append(cube[sel].mean(axis=0))
        bmeta.append({"ts": key + "00", "date_obs": meta[sel[0]].get("date_obs", ""),
                      "members": sel,
                      "n_subbands": meta[sel[0]].get("n_subbands", ""),
                      "wsum": float(np.sum([meta[i].get("wsum", 0.0) for i in sel]))
                      if "wsum" in meta[sel[0]] else None})
    for bm in bmeta:
        if bm.get("wsum") is None:
            bm.pop("wsum", None)
    blocks = np.stack(frames) if frames else np.zeros((0,) + model.shape, np.float32)
    return blocks, bmeta, brows


def _minute_tier(cube, meta, admitted, model, times, masks, thresh, stokes, tol_pix, hdr0, screen_x, screen_y, bin_lsts,
                 subband, info):
    """The 1-minute tier of one (band, Stokes): blocks by integration time (build_minute_blocks), high-passed block
    residual, detection, positions, PB, clustering.  ONE function for the full search and the --tier1min-only re-run
    (frame-time fix 2026-10-07), so the two cannot differ.  Returns (blk_rows, cands1, events1); records flood_1min and
    frame_time_1min in info."""
    blocks, blk_meta, blk_rows = build_minute_blocks(cube, meta, admitted, model)
    info["frame_time_1min"] = TIER1MIN_TIME_BASIS
    n_blk = len(blk_meta)
    cands1, events1 = [], []
    if n_blk:
        cube1 = highpass_blocks(blocks - model[None, :, :])
        del blocks
        zero1 = np.zeros_like(model)
        btimes = [ts_to_epoch_seconds(b["ts"]) for b in blk_meta]
        t0, span = min(times), max(max(times) - min(times), 1.0)
        bbin = np.minimum((np.floor((np.array(btimes) - t0) / span * N_TIME_BINS)).astype(int),
                          N_TIME_BINS - 1)
        blk_adm = np.ones(n_blk, dtype=bool)
        cands1, noise1, blk_adm, flood1 = detect(cube1, zero1, blk_adm, blk_meta,
                                                 thresh, stokes, masks, bbin)
        info["flood_1min"] = flood1
        del cube1
        # merge the per-minute refusal rows with the per-block noise rows
        by_ts = {r["ts"]: r for r in noise1}
        for r in blk_rows:
            n = by_ts.get(r["block_ts"])
            if n is not None:
                r["global_rms_jyb"] = n["global_rms_jyb"]
                r["n_components"] = n["n_components"]
                r["flood_refused"] = n["flood_refused"]
                r["admitted"] = n["admitted"]
                r["elev_bin"] = n["elev_bin"]
                r["n_subbands"] = n["n_subbands"]
                r["wsum"] = n["wsum"]
                if n["refuse_reason"]:
                    r["refuse_reason"] = n["refuse_reason"]
        cands1 = motion_flags(cands1, tol_pix)
        for i, c in enumerate(cands1):
            c["id"] = i
        cands1 = attach_positions(cands1, hdr0, screen_x, screen_y, bin_lsts)
        cands1 = attach_pb(cands1, subband, cube.shape[1:])
        events1 = cluster_events(cands1, tol_pix)
        events1 = attach_positions(events1, hdr0, screen_x, screen_y, bin_lsts)
    return blk_rows, cands1, events1


def _search_one_stokes(cube, meta, hdr0, admitted, stokes, mode, lst, date,
                       subband, out_dir, thresh, screen_x, screen_y):
    t_start = time.time()
    nt = cube.shape[0]
    admitted = np.array(admitted, dtype=bool)
    # blank-frame refusal (CLEANed generation): mostly-NaN frames are
    # refused AND excluded from the statistics stacks — a single all-NaN
    # frame otherwise poisons the plain median at every pixel
    finite_ok = np.array([m.get("finite_frac", 1.0) >= 0.5 for m in meta])
    if not finite_ok.all():
        log.warning("%d/%d frames mostly blank (finite fraction <0.5) — "
                    "refused and excluded from statistics",
                    int((~finite_ok).sum()), nt)
        admitted = admitted & finite_ok
    # no-data (all-zero) frame refusal, Gregg 2026-09-12: refused AND excluded
    # from the statistics stack, like the all-NaN frames above
    zero_frame = np.array([bool(m.get("all_zero", False)) for m in meta])
    if zero_frame.any():
        log.warning("%d/%d frames are all-zero (no gridded visibilities) — "
                    "refused and excluded from statistics",
                    int(zero_frame.sum()), nt)
        admitted = admitted & ~zero_frame
    # NIGHT RULE: refused frames are refused AND kept out of the hour median (same rule as admission_table)
    sun_up = night_refused_mask(meta, lst, date, subband)
    admitted = admitted & ~sun_up
    stats_ok = finite_ok & ~zero_frame & ~sun_up
    if not stats_ok.any():
        rec = night_refusal_record([m["ts"] for m in meta], lst, date, subband, stokes, mode) if sun_up.all() else None
        if rec is not None:
            raise NightRuleRefused("REFUSED (night rule): no frame of %s %s %s %s passes (Sun above the horizon, or twilight "
                                   "in a band the solar audit flags)" % (lst, date, subband, stokes), rec)
        raise RuntimeError("REFUSED: no frame of %s %s %s %s passes (night rule %d, mostly blank %d, all-zero %d of %d frames) "
                           "- not a night-rule refusal, no refusal record" % (lst, date, subband, stokes, int(sun_up.sum()),
                                                                              int((~finite_ok).sum()), int(zero_frame.sum()), nt))
    log.info("%s %s %s %s: %d frames (%d admitted), mode=%s",
             lst, date, subband, stokes, nt, int(admitted.sum()), mode)
    info = {"lst": lst, "date": date, "subband": subband, "stokes": stokes,
            "mode": mode, "thresh": thresh, "n_frames": nt,
            "n_admitted": int(admitted.sum()),
            "n_blank_frames": int((~finite_ok).sum()),
            "n_zero_frames": int(zero_frame.sum()),
            "statistic": "residual / per-image global MAD rms (central 25% area, "
                         "elevation domain)",
            "flow_registered": bool(flow_applies(subband)),
            "frame_admission": "beam area <= 1.5x median (both Stokes headers); "
                               "blank fraction < 0.5; all-zero (no-data) frame "
                               "refused (Gregg 2026-09-12); per-Stokes residual MAD "
                               "> median + %.1f robust sigma refused (Gregg "
                               "2026-09-09)" % NOISE_OUTLIER_K,
            "min_elevation_deg": MIN_ELEVATION_DEG,
            "flood_rel": FLOOD_REL, "flood_min_abs": FLOOD_MIN_ABS,
            "flood_abs_max": FLOOD_ABS_MAX}
    if (lst, date, subband) in PHASE1_RUN_REPORT:       # several Phase-1 runs: what discovery kept and dropped (2026-10-18)
        info["phase1_runs"] = PHASE1_RUN_REPORT[(lst, date, subband)]

    stats_cube = cube if stats_ok.all() else cube[np.where(stats_ok)[0]]
    hour_median = chunked_median(stats_cube)
    if stats_cube is not cube:
        del stats_cube
    if mode == "median":
        model = hour_median
    elif mode == "master":
        mpath = os.path.join(MASTER_ROOT, "%s_%s" % (lst, subband),
                             "coadd_final",
                             "%s_deep_Taper_R0_%s.fits" % (stokes, subband))
        if not os.path.exists(mpath):
            raise RuntimeError("master not found: %s" % mpath)
        if screen_x is None:
            raise RuntimeError("master mode requires a warp screen")
        mhdr = fits.getheader(mpath)
        gate = str(mhdr.get("COADDOP", ""))
        if gate != "ivw-mean-v2" and os.environ.get("P3_ALLOW_OLD_MASTERS") != "1":
            raise RuntimeError("master generation gate: COADDOP=%r != "
                               "ivw-mean-v2 (%s)" % (gate, mpath))
        model = master_in_raw_frame(mpath, hdr0, screen_x, screen_y)
        scale, n_bright = robust_scale(hour_median, model)
        info["master_path"] = mpath
        info["master_scale_vs_hour_median"] = scale
        info["master_scale_n_bright_pix"] = n_bright
        log.info("master mode: flux scale (hour-median / warped master) "
                 "= %.4f over %d bright pixels — applied", scale, n_bright)
        model = model * np.float32(scale)
        bad = ~np.isfinite(model)
        model[bad] = hour_median[bad]
        info["master_nan_filled_pix"] = int(bad.sum())
    else:
        raise ValueError(mode)

    # search domain per time bin (on-sky + elevation >= MIN_ELEVATION_DEG)
    times = [ts_to_epoch_seconds(m["ts"]) for m in meta]
    masks, bin_of, bin_lsts = elevation_masks(hdr0, times, shape=cube.shape[1:])
    info["bin_lst_deg"] = [round(x, 3) for x in bin_lsts]

    beam_fwhm_pix = abs(float(hdr0.get("BMAJ", 0.2)) /
                        float(hdr0.get("CDELT2", 0.03125)))
    tol_pix = max(2.0, beam_fwhm_pix)
    tag = "%s_%s_%s" % (subband, stokes, mode)

    # ---- 10-s tier
    cands, noise_rows, admitted, flood = detect(cube, model, admitted, meta,
                                                thresh, stokes, masks, bin_of)
    info.update({"flood_10s": flood})
    cands = motion_flags(cands, tol_pix)
    for i, c in enumerate(cands):
        c["id"] = i
    cands = attach_positions(cands, hdr0, screen_x, screen_y, bin_lsts)
    cands = attach_pb(cands, subband, cube.shape[1:])
    events = cluster_events(cands, tol_pix)
    events = attach_positions(events, hdr0, screen_x, screen_y, bin_lsts)
    write_csv(os.path.join(out_dir, "noise_%s.csv" % tag), noise_rows)
    write_csv(os.path.join(out_dir, "candidates_%s.csv" % tag), cands)
    write_csv(os.path.join(out_dir, "events_%s.csv" % tag), events)
    # PHOTOMETRY CUTOUT PRODUCT (Gregg 2026-09-11): every HWJ / 10 pc / planet target, every frame,
    # 5x5 deg residual, dewarped, one cube per (object, band, Stokes, cell-hour).  Disabled with
    # P3_SNAP_CUTOUTS=0; root from P3_SNAP_CUTOUT_ROOT.
    if os.environ.get("P3_SNAP_INJECTED") == "1":
        info["photometry_cutout_cubes"] = "REFUSED: injected run (synthetic sources in the cube) - no production cutouts"
        log.warning("photometry cutouts REFUSED for %s %s %s: injected run", lst, subband, stokes)
    elif os.environ.get("P3_SNAP_CUTOUTS", "1") != "0":
        try:
            from phase3_snapshot_cutouts import write_cutouts, ROOT_DEFAULT
            n_cut = write_cutouts(cube, model, meta, admitted, hdr0, screen_x, screen_y, lst, date, subband, stokes,
                                  noise_rows=noise_rows, root=os.environ.get("P3_SNAP_CUTOUT_ROOT", ROOT_DEFAULT))
            info["photometry_cutout_cubes"] = int(n_cut)
        except Exception as e:      # the product must never take the search down; the gap is recorded
            log.error("photometry cutouts FAILED for %s %s %s: %s", lst, subband, stokes, e)
            info["photometry_cutout_cubes"] = "FAILED: %s" % str(e)[:200]
            if os.environ.get("P3_SNAP_CUTOUTS_ONLY") == "1":
                raise
    # CUTOUTS ONLY (2026-10-25, Gregg "Forced photometry is also approved"; regen_cutouts): the photometry cutouts regenerated from
    # the current cubes - the search stops here, after the admission, the 10-s detection's noise rows and the cutouts
    if os.environ.get("P3_SNAP_CUTOUTS_ONLY") == "1":
        return info

    # ---- 1-minute tier: wall-clock-minute blocks of admitted frames
    # PAUSE (Gregg 2026-10-07 on frame times, "Your rec accepted": pause the wideband 10-s search and the 1-minute stages
    # until the frame-time fix lands - block membership still follows file-name times).  P3_SNAP_PAUSE_1MIN=1 builds no
    # blocks and writes NO 1-minute file, only runinfo tier_1min = "paused", so no consumer can read a paused cell as zero
    # 1-minute candidates; the cell's 1-minute stage is run after the fix.
    pause1 = os.environ.get("P3_SNAP_PAUSE_1MIN") == "1"
    blk_rows, cands1, events1 = ([], [], []) if pause1 else _minute_tier(
        cube, meta, admitted, model, times, masks, thresh, stokes, tol_pix, hdr0, screen_x, screen_y, bin_lsts, subband, info)
    if pause1:
        info["tier_1min"] = "paused"
    else:
        write_csv(os.path.join(out_dir, "noise_%s_1min.csv" % tag), blk_rows)
        write_csv(os.path.join(out_dir, "candidates_%s_1min.csv" % tag), cands1)
        write_csv(os.path.join(out_dir, "events_%s_1min.csv" % tag), events1)
        info.update({"n_candidates_1min": len(cands1),
                     "n_events_1min": len(events1),
                     "n_blocks_1min": len(blk_rows),
                     "n_blocks_admitted_1min": int(sum(r["admitted"] for r in blk_rows))})

    save_map(os.path.join(out_dir, "model_%s.fits" % tag), model, hdr0,
             "snapshot search sky model, mode=%s" % mode)
    info.update({"n_candidates": len(cands), "n_events": len(events),
                 "n_admitted_after_flood": int(admitted.sum()),
                 "beam_fwhm_pix": beam_fwhm_pix,
                 "runtime_s": round(time.time() - t_start, 1)})
    with open(os.path.join(out_dir, "runinfo_%s.json" % tag), "w") as f:
        json.dump(info, f, indent=2)
    log.info("done: %d candidates, %d events, %d 1-min candidates, %.1f s",
             len(cands), len(events), len(cands1), info["runtime_s"])
    return info


NOISE_REPRO_MAX = 0.005  # --tier1min-only content gate: |recomputed / recorded frame noise - 1| allowed.  MEASURED 2026-10-07 on
                         # 19h 2025-05-03, 9,988 admitted frames, 30 subband-Stokes: archived-cube frames reproduce the recorded
                         # noise to median 0.01-0.02%, max 0.10%; the same frames UNFLOWED (wrong registration) differ by
                         # median 2.66%, min 2.0% (336 of 336 refused) - the limit sits 5x above one and 4x below the other


def run_minute_tier_only(lst, date, subband, stokes, mode, out_dir, stage_root=None, nthreads=8):
    """RE-RUN OF THE 1-MINUTE TIER ALONE (frame-time fix, minimal form; Gregg 2026-10-07, "Were not redoing 1,380 hours of
    anything. Fix it." / "Go with your recommendation": keep the 10-s subband work, re-run the 1-minute stage).

    Per Stokes, the frames come from the staging exactly as the full search reads them (flowed copies: a re-flow, or the
    frames exploded from the archived cube), and the 10-s search's OWN products in out_dir supply the rest:
    noise_{tag}.csv (the frame list and every frame's FINAL admission, after the noise-outlier and flood refusals),
    model_{tag}.fits (the hour median it subtracted) and runinfo_{tag}.json (its harvest threshold and flow state).
    REFUSES (RuntimeError) unless (a) the staged frame list is the noise table's, frame for frame, the search was
    flow-registered as this run is, and every frame's elevation bin is the recorded one; and (b) CONTENT: every admitted
    frame's residual noise, recomputed from the staged frame and the kept model exactly as detect() measured it, equals the
    recorded global_rms_jyb within NOISE_REPRO_MAX (relative) - the 1-minute blocks are built from the frames the 10-s tier
    searched, not from another registration.  Then the full search's own 1-minute function (_minute_tier) and its three
    files; runinfo_{tag}.json keeps every 10-s key and gains the 1-minute keys and tier_1min_rerun.  No 10-s product and no
    cutout is written."""
    stokes_list = ["V", "I"] if stokes == "VI" else [stokes]
    frames = discover_frames(lst, date, subband, stage_root=stage_root)
    done = {}
    for st in stokes_list:
        t_start = time.time()
        tag = "%s_%s_%s" % (subband, st, mode)
        npath, mpath, rpath = (os.path.join(out_dir, f % tag) for f in ("noise_%s.csv", "model_%s.fits", "runinfo_%s.json"))
        for p in (npath, mpath, rpath):
            if not os.path.exists(p):
                raise RuntimeError("REFUSED (--tier1min-only): %s is missing - the 1-minute re-run reads the 10-s search's "
                                   "products" % p)
        with open(npath) as f:
            nrows = list(csv.DictReader(f))
        info = json.load(open(rpath))
        fl = frames[st]
        if [ts for ts, _ in fl] != [r["ts"] for r in nrows]:
            raise RuntimeError("REFUSED (--tier1min-only): %s %s %s %s: the staged frame list (%d) is not the 10-s search's "
                               "(%d frames in %s)" % (lst, date, subband, st, len(fl), len(nrows), os.path.basename(npath)))
        if (info.get("subband"), info.get("stokes"), info.get("mode")) != (subband, st, mode) \
                or bool(info.get("flow_registered")) != bool(flow_applies(subband)):
            raise RuntimeError("REFUSED (--tier1min-only): %s describes %s/%s/%s flow_registered=%s, this run is %s/%s/%s "
                               "flow=%s" % (rpath, info.get("subband"), info.get("stokes"), info.get("mode"),
                                            info.get("flow_registered"), subband, st, mode, flow_applies(subband)))
        thresh = float(info["thresh"])
        cube, meta, hdr0 = load_cube(fl, nthreads=nthreads)
        model = np.squeeze(fits.getdata(mpath)).astype(np.float32)
        if model.shape != cube.shape[1:]:
            raise RuntimeError("REFUSED (--tier1min-only): model %s is %s, frames are %s" % (mpath, model.shape, cube.shape[1:]))
        admitted = np.array([r["admitted"] == "1" for r in nrows], dtype=bool)
        times = [ts_to_epoch_seconds(m["ts"]) for m in meta]
        masks, bin_of, bin_lsts = elevation_masks(hdr0, times, shape=cube.shape[1:])
        rel = []
        for t in np.where(admitted)[0]:
            if int(nrows[t]["elev_bin"]) != int(bin_of[t]):
                raise RuntimeError("REFUSED (--tier1min-only): frame %s elevation bin %s, recorded %s"
                                   % (meta[t]["ts"], bin_of[t], nrows[t]["elev_bin"]))
            rel.append(central_rms(cube[t] - model, mask=masks[bin_of[t]]) / float(nrows[t]["global_rms_jyb"]) - 1.0)
        rel = np.abs(np.array(rel, dtype=float))
        nbad = int(np.sum(~(rel <= NOISE_REPRO_MAX)))
        repro = {"n_admitted_frames": int(admitted.sum()), "n_beyond_limit": nbad, "limit_rel": NOISE_REPRO_MAX,
                 "median_abs_rel": float(np.median(rel)) if len(rel) else None, "max_abs_rel": float(np.max(rel)) if len(rel) else None}
        log.info("%s %s: frame-noise reproduction against %s: %s", tag, lst + "_" + date, os.path.basename(npath), repro)
        if nbad:
            raise RuntimeError("REFUSED (--tier1min-only): %d of %d admitted frames of %s %s do not reproduce the 10-s search's "
                               "recorded noise within %.1f%% (median |diff| %.4f, max %.4f) - these are not the frames it "
                               "searched" % (nbad, len(rel), lst + "_" + date, tag, 100 * NOISE_REPRO_MAX,
                                             repro["median_abs_rel"], repro["max_abs_rel"]))
        beam_fwhm_pix = abs(float(hdr0.get("BMAJ", 0.2)) / float(hdr0.get("CDELT2", 0.03125)))
        tol_pix = max(2.0, beam_fwhm_pix)
        if flow_applies(subband):
            screen_x = screen_y = np.zeros(cube.shape[1:], np.float32)
        else:
            screen_x, screen_y = load_warp_screen(lst, date, subband)
            if screen_x is not None and screen_x.shape != cube.shape[1:]:
                screen_x = screen_y = None
        info.pop("tier_1min", None)            # a paused cell's marker (P3_SNAP_PAUSE_1MIN) is replaced by the tier itself
        blk_rows, cands1, events1 = _minute_tier(cube, meta, admitted, model, times, masks, thresh, st, tol_pix, hdr0,
                                                 screen_x, screen_y, bin_lsts, subband, info)
        del cube
        write_csv(os.path.join(out_dir, "noise_%s_1min.csv" % tag), blk_rows)
        write_csv(os.path.join(out_dir, "candidates_%s_1min.csv" % tag), cands1)
        write_csv(os.path.join(out_dir, "events_%s_1min.csv" % tag), events1)
        info.update({"n_candidates_1min": len(cands1), "n_events_1min": len(events1), "n_blocks_1min": len(blk_rows),
                     "n_blocks_admitted_1min": int(sum(r["admitted"] for r in blk_rows))})
        info["tier_1min_rerun"] = {
            "utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "frames": stage_root or "archive",
            "frame_noise_reproduction": repro, "runtime_s": round(time.time() - t_start, 1),
            "why": "frame-time fix, minimal form (Gregg 2026-10-07): 1-minute blocks by each frame's integration midpoint; "
                   "10-s products kept"}
        tmp = rpath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(info, f, indent=2)
        os.replace(tmp, rpath)
        log.info("1-minute re-run done %s: %d blocks (%d admitted), %d candidates, %d events, %.1f s", tag, len(blk_rows),
                 info["n_blocks_admitted_1min"], len(cands1), len(events1), time.time() - t_start)
        done[st] = info
    return done


DATA_DECISIONS = "/lustre/gh/main/clv15/data_decisions.csv"


def excluded_subbands(lst, date, path=DATA_DECISIONS):
    """Subbands of one (LST, date) cell carrying verdict 'exclude' in
    data_decisions.csv (CLAUDE.md S2: exclusion is absolute and total, co-adds
    AND searches; no code path re-admits an excluded cell).  Gate added
    2026-09-12 with Gregg's exclusion of 02h 2025-01-02 (all) and 03h 2025-01-02
    (18, 73 MHz): the snapshot tier had no exclusion check before."""
    out = set()
    gpu = frames_from_gpu(lst, date)
    with open(path) as f:
        for r in csv.DictReader(f):
            if (r.get("lst", "").strip() == lst and r.get("date", "").strip() == date
                    and r.get("verdict", "").strip() == "exclude"):
                # GPU-frame re-admission (Gregg 2026-11-10): a tagged row is skipped only when this run reads the GPU frames
                if gpu and SNAPSHOT_GPU_READMIT_TAG in r.get("reason", ""):
                    continue
                out.add(r.get("subband", "").strip())
    return out


GPU_FRAME_ROOT = "/lustre/pipeline/snapshots_gpu_pipeline"
SNAPSHOT_GPU_READMIT_TAG = "[SNAPSHOT-GPU-READMITTED"   # = exopipe coadd_full_pipeline.SNAPSHOT_GPU_READMIT_TAG


GPU_CELLS_FILE = os.environ.get("P3_SNAP_GPU_CELLS", "/lustre/gh/main/exopipe_ops/snapshot_gpu_frame_cells.txt")
_GPU_CELLS = None


def gpu_cells():
    """{(LST, date)} whose 10-s frames come from the GPU pipeline (GPU_CELLS_FILE: the Dec 18-27 2024 cells without the survey's own
    CLEANed Stokes I, Gregg 2026-11-08/10); a missing file is an empty set."""
    global _GPU_CELLS
    if _GPU_CELLS is None:
        _GPU_CELLS = set()
        if os.path.exists(GPU_CELLS_FILE):
            for ln in open(GPU_CELLS_FILE):
                p = ln.split()
                if len(p) >= 2 and not ln.startswith("#"):
                    _GPU_CELLS.add((p[0], p[1]))
    return _GPU_CELLS


def frame_root(lst, date):
    """The frame root of ONE cell (2026-11-10): a run-wide P3_SNAP_PHASE1_ROOT override wins (the Dec 18-27 fleet); otherwise a
    GPU-frame cell reads GPU_FRAME_ROOT and every other cell the exopipe Phase 1 root - so one chain or 10-minute run can mix them."""
    if os.path.realpath(PHASE1_ROOT) != os.path.realpath("/lustre/pipeline/exopipe/phase1"):
        return PHASE1_ROOT
    return GPU_FRAME_ROOT if (lst, date) in gpu_cells() else PHASE1_ROOT


def frames_from_gpu(lst=None, date=None):
    """True when the frames read - of cell (lst, date), or run-wide without one - are the observatory GPU pipeline's, the only frames
    for which the snapshot tiers re-admit the cell-bands tagged SNAPSHOT-GPU-READMITTED in data_decisions.csv (Gregg 2026-11-10)."""
    root = frame_root(lst, date) if lst and date else PHASE1_ROOT
    return os.path.realpath(root) == os.path.realpath(GPU_FRAME_ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lst", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--subband", required=True)
    ap.add_argument("--stokes", required=True, choices=["I", "V", "VI"])
    ap.add_argument("--mode", default="median", choices=["median", "master"])
    ap.add_argument("--thresh", type=float, default=5.0,
                    help="harvest threshold in per-image global sigma")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage-root", default=None,
                    help="node-local dir holding staged copies of the frames")
    ap.add_argument("--nthreads", type=int, default=8)
    ap.add_argument("--cutouts-from-cubes", action="store_true",
                    help="regenerate the photometry cutouts from the consolidated flow cubes (regen_cutouts; needs P3_SNAP_CUTOUT_ROOT)")
    ap.add_argument("--inject", default=None,
                    help="injection manifest CSV (SNAP-BAT battery); "
                         "sources are added to the in-memory cubes only")
    ap.add_argument("--tier1min-only", action="store_true",
                    help="re-run ONLY the 1-minute tier from the 10-s search's products already in --out "
                         "(run_minute_tier_only; frame-time fix 2026-10-07)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.subband in excluded_subbands(args.lst, args.date):
        print("EXCLUDED: %s %s %s carries verdict 'exclude' in %s — REFUSING to search "
              "(no output written)" % (args.lst, args.date, args.subband, DATA_DECISIONS))
        raise SystemExit(3)
    if args.cutouts_from_cubes:
        if args.inject:
            raise SystemExit("REFUSED: --cutouts-from-cubes with --inject")
        regen_cutouts(args.lst, args.date, args.subband, args.out)
        return
    if args.tier1min_only:
        if os.environ.get("P3_SNAP_PAUSE_1MIN") == "1" or args.inject:
            raise SystemExit("REFUSED: --tier1min-only with P3_SNAP_PAUSE_1MIN=1 or --inject")
        run_minute_tier_only(args.lst, args.date, args.subband, args.stokes, args.mode, args.out,
                             stage_root=args.stage_root, nthreads=args.nthreads)
        return
    run_search(args.lst, args.date, args.subband, args.stokes, args.mode,
               args.out, thresh=args.thresh, stage_root=args.stage_root,
               nthreads=args.nthreads, inject=args.inject)


if __name__ == "__main__":
    main()
