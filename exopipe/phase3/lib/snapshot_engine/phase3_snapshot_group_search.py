#!/usr/bin/env python
"""Phase 3 snapshot-tier GROUP search: Red/Green/Blue/Full coadds (2026-09-03).

Builds dewarped, inverse-variance-weighted snapshot coadds across subband
groups and searches them with the same detection machinery as the
per-subband pass (hour-median sky model, per-pixel temporal noise, 10-s and
1-minute tiers).

Groups (nominal labels; exact frequencies live in cfg.SUBBAND_FREQ_HZ):
  Red  = 18,23,27,32,36 MHz   Green = 41,46,50,55,59 MHz
  Blue = 64,69,73,78,82 MHz   Full  = all fifteen

Method per time chunk: for every subband, the frame of the SAME INTEGRATION as the reference frame
(header DATE-OBS within PAIR_TOL_S; frame-time fix 2026-10-07, minimal form - until then the frame
whose FILE-NAME time was nearest within 12 s, and old file names drift from the integration by up to
26 s differently per subband, so 60-97 per cent of wideband frames mixed integrations on 19h 2025-05-03
and 14h 2025-04-21) is read from the staged copy, DEWARPED
with that subband's own warp screen (differential ionospheric refraction
across a group is what smears an undewarped coadd), weighted by
1/central-rms^2, and accumulated into its group cube AND the Full cube —
each frame is read and dewarped exactly once.

Because group frames are dewarped, candidate positions need no warp
correction (warp_dx=0; the engine flags astrom_degraded=1 because no
screen is passed - group positions ARE dewarped).  Frame matching is
one-to-one (2026-09-05 redesign) and per-instant composition (number of
subbands, weight sum) is written to the persisted noise table.  Admission verdicts are
inherited from the per-subband pass (admission_{sb}.csv in --out); frames
refused there get zero weight here.  A wideband frame keeps the REFERENCE subband frame's file-name
time as its identity (ts) and carries the integration's DATE-OBS (noise table date_obs, cutouts, 1-minute
blocks by integration midpoint); every member is listed in wideband_members_{Stokes}.csv.

EXCLUSION (Gregg 2026-10-07, Q29 "Fine"; canon PHASE3_ARCHITECTURE.md section 14, 2026-08-31 item 2, and Gregg
2026-09-05 "the policy is to not form a wideband product with an excluded subband. Maintain that policy."): a
wideband whose scope is excluded under the deep rule (exopipe coadd_full_pipeline.is_excluded: any of its own
constituent subbands excluded, the ratified 18 MHz carve-out and [PERBAND] rows excepted, or a row of its own) is
REFUSED - not built, not searched, a wideband_<G>_<Stokes>_REFUSED.txt marker says why.  An excluded subband's
frames never enter any wideband (exclusion is absolute and total, CLAUDE.md 2).

RAM: 4 accumulator cubes + weights (~100 GB for the 4096^2 generation) —
run this ALONE, after the per-subband sweep, never concurrently with it.
"""
import argparse
import csv
csv.field_size_limit(100000000)
import json
import logging
import os
import threading
import time

import numpy as np
from scipy import ndimage

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3_snapshot_search import (SCIENCE_SUBBANDS, _search_one_stokes, excluded_subbands, FLOW_MODE, flow_applies,
                                    NightRuleRefused, write_night_refusal,
                                    central_rms, discover_frames,
                                    load_warp_screen, read_frame, write_csv,
                                    ts_to_epoch_seconds, frame_dateobs, dateobs_unix)

log = logging.getLogger("snapgroup")

GROUPS = {"Red": SCIENCE_SUBBANDS[0:5],
          "Green": SCIENCE_SUBBANDS[5:10],
          "Blue": SCIENCE_SUBBANDS[10:15]}
# Instants accumulated at a time by accumulate() (speed only; identical products for every value).  0 = production's own loop,
# unchanged.  Read from the environment at import; the step wiring that sets it from config comes with the next pull request.
GROUP_THREADS = max(0, int(os.environ.get("P3_SNAP_GROUP_THREADS", "0")))
_STAGED_READ_FRAME = read_frame      # the staged-file reader; regen_group_cutouts swaps read_frame for its member-cube reader
LUSTRE_READERS = 6                   # frames read from /lustre at a time by the per-instant accumulation (REVIEW.md)
MATCH_TOL_S = 12.0      # RETIRED 2026-10-07 (pairing by file-name time); kept for readers of old logs only
PAIR_TOL_S = 1.0        # s: members = frames of the SAME integration, DATE-OBS equal within 1 s (distinct integrations of a
                        # cell are 10.0-10.1 s apart: 358 per cell on 19h 2025-05-03 and 14h 2025-04-21, measured 2026-10-07)



def _excl_scope(lst=None, date=None):
    """'snapshot-gpu' when the cell's frames are the GPU pipeline's (Gregg 2026-11-10 re-admission), else 'snapshot' (the deep rule
    plus the snapshot-tier-only rows, [SNAPSHOT-TIERS-ONLY, 2026-11-13)."""
    import phase3_snapshot_search as _S
    return 'snapshot-gpu' if _S.frames_from_gpu(lst, date) else 'snapshot'

def pair_by_integration(ref_secs, secs, tol=PAIR_TOL_S):
    """Index into one subband's frame list per reference instant: the frame whose DATE-OBS (unix s) lies within tol of the
    reference frame's - the same integration - else -1.  One-to-one, as before (audit F6): an instant takes at most one
    frame per subband (the nearest) and a frame feeds at most one instant."""
    ref_secs = np.asarray(ref_secs, dtype=float)
    secs = np.asarray(secs, dtype=float)
    order = np.argsort(ref_secs, kind="mergesort")
    rs = ref_secs[order]
    match = np.full(len(ref_secs), -1, dtype=int)
    bestd = np.full(len(ref_secs), np.inf)
    idx = np.searchsorted(rs, secs)
    for j in range(len(secs)):
        tb, db = -1, np.inf
        for k in (idx[j] - 1, idx[j]):
            if 0 <= k < len(rs):
                d = abs(float(secs[j] - rs[k]))
                if d < db:
                    tb, db = int(order[k]), d
        if tb >= 0 and db <= tol and db < bestd[tb]:
            match[tb], bestd[tb] = j, db
    return match


def dateobs_seconds(fl, what):
    """Unix DATE-OBS of every (ts, path) of a frame list, from each file's own header; REFUSES a frame without one."""
    dob = frame_dateobs([p for _, p in fl])
    out = []
    for ts, p in fl:
        try:
            out.append(dateobs_unix(dob.get(p, "")))
        except ValueError:
            raise RuntimeError("REFUSED: %s frame %s (%s) has no readable DATE-OBS (%r) - a wideband pairs frames by their own "
                               "integration (frame-time fix, Gregg 2026-10-03/07)" % (what, ts, p, dob.get(p)))
    return np.array(out, dtype=float), dob


def _ts_seconds(ts):
    """Seconds since epoch (date-aware; UTC-midnight cells match correctly)."""
    return ts_to_epoch_seconds(ts)


def read_admission(out_dir, subband):
    """ts -> admitted (bool) from the per-subband pass; empty dict if the
    per-subband pass has not run (then all frames are eligible)."""
    p = os.path.join(out_dir, "admission_%s.csv" % subband)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return {r["ts"]: r["admitted"] == "1" for r in csv.DictReader(f)}


def dewarp_coords(shape, screen_x, screen_y):
    """The sample coordinates dewarp_frame builds for a frame of this shape and these screens."""
    gy, gx = np.mgrid[0:shape[0], 0:shape[1]]
    return [gy + screen_y, gx + screen_x]


def dewarp_frame(data, screen_x, screen_y, coords=None):
    """coords (refactor, 2026-11-11): dewarp_coords(data.shape, screen_x, screen_y) built once by the caller for many frames of
    one subband - the same arrays, so the same result; None builds them here, as production does for every frame."""
    if coords is None:
        coords = dewarp_coords(data.shape, screen_x, screen_y)
    return ndimage.map_coordinates(
        data, coords, order=1,
        mode="constant", cval=np.nan).astype(np.float32)


def accumulate(nt, sb_frames, sb_match, sb_adm, sb_screens, grid_of, group_of, full_ok, ref_list, sb_dsec, ref_dob,
               cubes, wsums, nsubs, nthreads, chunk_frames, t0, per_instant):
    """Add every admitted member frame, weighted 1/rms^2, into its wideband cube(s) (cubes[g][t], wsums[g][t], nsubs[g][t], in
    place) and return the member rows in production's append order.

    per_instant False = PRODUCTION'S LOOP, unchanged (clv15 be63765 :299-338): workers read and dewarp the frames of a chunk
    subband after subband, and this thread adds them.  It is the default (P3_SNAP_GROUP_THREADS = 0) and the only order used
    when the frames come from member cubes (regen_group_cutouts swaps read_frame for a reader that keeps two cubes in memory and
    must be read member after member).

    per_instant True = SPEED-UP 3 (refactor, 2026-11-11; identical products; plan/STAGE_MAP_20261111.md hot spot 3):
     (a) a subband's dewarp coordinates depend only on the subband, so they are built once per subband and passed to the same
         map_coordinates call (dewarp_frame; production rebuilt them for every frame);
     (b) one worker handles one INSTANT t: it reads that instant's member frames in production's subband order and adds them
         into cubes[g][t] / wsums[g][t] / nsubs[g][t] in that order.  The adds of one (group, instant) are therefore the same
         float operations in the same order as production's; different instants touch different array elements;
     (c) the member rows are put back in production's append order (chunk, subband, instant, group)."""
    from concurrent.futures import ThreadPoolExecutor
    coords, coord_locks = {}, {sb: threading.Lock() for sb in sb_frames}

    def _one(sb, t):
        j = sb_match[sb][t]
        if j < 0:
            return None
        ts, path = sb_frames[sb][j]
        adm = sb_adm[sb]
        if adm and not adm.get(ts, False):
            return None
        data, _ = read_frame(path)
        if data.shape != grid_of[group_of[sb]]:
            log.warning("%s frame %s has grid %s, wideband %s expects %s — dropped",
                        sb, ts, data.shape, group_of[sb], grid_of[group_of[sb]])
            return None
        sx, sy = sb_screens[sb]
        if per_instant:
            key = (sb, data.shape)
            with coord_locks[sb]:                 # one lock per subband: building one grid does not hold the others up
                if key not in coords:
                    coords[key] = dewarp_coords(data.shape, sx, sy)
            dw = dewarp_frame(data, sx, sy, coords=coords[key])
        else:
            dw = dewarp_frame(data, sx, sy)
        rms = central_rms(dw)
        if not (np.isfinite(rms) and rms > 0):
            return None
        return sb, t, dw, 1.0 / rms ** 2

    if not per_instant:
        member_rows = []
        for c0 in range(0, nt, chunk_frames):
            c1 = min(c0 + chunk_frames, nt)
            jobs = [(sb, t) for sb in sb_frames for t in range(c0, c1)]
            with ThreadPoolExecutor(max_workers=nthreads) as ex:
                for res in ex.map(lambda a: _one(*a), jobs):
                    if res is None:
                        continue
                    sb, t, dw, w = res
                    contrib = np.nan_to_num(dw, nan=0.0) * np.float32(w)
                    gmask = np.isfinite(dw)
                    for g in ((group_of[sb], "Full") if full_ok else (group_of[sb],)):
                        cubes[g][t] += contrib
                        wsums[g][t] += w
                        nsubs[g][t] += 1
                        j = sb_match[sb][t]
                        member_rows.append((g, ref_list[t][0], sb, sb_frames[sb][j][0], sb_dsec[sb][1][sb_frames[sb][j][1]],
                                            ref_dob[ref_list[t][1]], w))
            log.info("chunk %d-%d accumulated (%.0f s elapsed)",
                     c0, c1, time.time() - t0)
        return member_rows

    sb_order = {sb: k for k, sb in enumerate(sb_frames)}

    def _instant(t):
        rows_t = []
        for sb in sb_frames:
            res = _one(sb, t)
            if res is None:
                continue
            _, _, dw, w = res
            contrib = np.nan_to_num(dw, nan=0.0) * np.float32(w)
            for gi, g in enumerate((group_of[sb], "Full") if full_ok else (group_of[sb],)):
                cubes[g][t] += contrib
                wsums[g][t] += w
                nsubs[g][t] += 1
                j = sb_match[sb][t]
                rows_t.append(((t // chunk_frames, sb_order[sb], t, gi),
                               (g, ref_list[t][0], sb, sb_frames[sb][j][0], sb_dsec[sb][1][sb_frames[sb][j][1]],
                                ref_dob[ref_list[t][1]], w)))
        return rows_t

    # LUSTRE READS (REVIEW.md: at most 6 Lustre-heavy workers; review 2026-11-11): frames staged node-local are read
    # GROUP_THREADS at a time; frames read from /lustre (no staging given) at most LUSTRE_READERS at a time
    workers = GROUP_THREADS
    if any(str(fl[0][1]).startswith("/lustre/") for fl in sb_frames.values() if fl):
        workers = min(workers, LUSTRE_READERS)
    keyed_rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, rows_t in enumerate(ex.map(_instant, range(nt))):
            keyed_rows.extend(rows_t)
            if (k + 1) % 24 == 0 or k + 1 == nt:
                log.info("instants 0-%d accumulated (%.0f s elapsed)", k + 1, time.time() - t0)
    return [r for _, r in sorted(keyed_rows, key=lambda kr: kr[0])]


def run_group_search(lst, date, stokes, out_dir, thresh=5.0,
                     stage_root_fmt=None, nthreads=6, chunk_frames=24):
    """OUTCOME RECORD (2026-10-13, refactor defect M20/M34; gates refuse): every exit writes wideband_outcome_<Stokes>.json
    with each band group's outcome - searched | night-refused | excluded | no-members | too-few-frames | not-run - and the
    error if one ended the search.  The fleet's completeness guard reads it: a group left 'not-run' (a crash) fails the cell;
    before, the fleet never checked the group search (00h 2025-11-01 shipped with no Stokes V wideband search)."""
    os.makedirs(out_dir, exist_ok=True)
    outcome = {g: "not-run" for g in GROUPS}
    err = None
    try:
        return _run_group_search(lst, date, stokes, out_dir, thresh, stage_root_fmt, nthreads, chunk_frames, outcome)
    except BaseException as e:
        err = "%s: %s" % (type(e).__name__, e)
        raise
    finally:
        rec = dict(lst=lst, date=date, stokes=stokes, groups=outcome, error=err,
                   written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        fp = os.path.join(out_dir, "wideband_outcome_%s.json" % stokes)
        with open(fp + ".tmp", "w") as f:
            json.dump(rec, f, indent=2)
        os.replace(fp + ".tmp", fp)


def _run_group_search(lst, date, stokes, out_dir, thresh, stage_root_fmt, nthreads, chunk_frames, outcome):
    t0 = time.time()
    # PORT (refactor, 2026-11-11): as run_search - a missing default cutout target list refuses before any work
    if os.environ.get("P3_SNAP_CUTOUTS", "1") != "0" and os.environ.get("P3_SNAP_INJECTED") != "1":
        from phase3_snapshot_cutouts import check_default_lists
        check_default_lists()

    # WIDEBAND EXCLUSION = the deep rule (Q29, Gregg 2026-10-07 "Fine"): a refused wideband is neither built nor searched
    sys.path.insert(0, "/lustre/gh/main/exopipe")
    from exopipe.phase3.coadd_full_pipeline import load_exclusions, is_excluded
    _ex = load_exclusions(scope=_excl_scope(lst, date))
    refused_groups = {}
    for g in ("Red", "Green", "Blue", "Full"):
        if is_excluded(lst, g, date, _ex):
            subs = SCIENCE_SUBBANDS if g == "Full" else GROUPS[g]
            why = ("REFUSED: wideband %s of %s %s is excluded under the deep rule (coadd_full_pipeline.is_excluded; Q29, Gregg "
                   "2026-10-07; canon PHASE3_ARCHITECTURE.md 14, 2026-08-31 item 2): excluded constituents %s%s - not built, "
                   "not searched" % (g, lst, date, ",".join(sorted(s for s in subs if is_excluded(lst, s, date, _ex))) or "none",
                                     "; a row for the wideband itself" if (lst, g, date) in _ex else ""))
            refused_groups[g] = why
            if g in outcome:
                outcome[g] = "excluded"
            log.warning(why)
            with open(os.path.join(out_dir, "wideband_%s_%s_REFUSED.txt" % (g, stokes)), "w") as f:
                f.write(why + "\n")
    if all(g in refused_groups for g in GROUPS):
        raise SystemExit("REFUSED: every wideband of %s %s is excluded (data_decisions.csv) - no group search" % (lst, date))

    # per-subband frame lists, screens, admissions
    sb_frames, sb_screens, sb_adm = {}, {}, {}
    excl = excluded_subbands(lst, date)
    for sb in SCIENCE_SUBBANDS:
        if sb in excl:
            log.warning("group build: %s is EXCLUDED in data_decisions.csv for %s %s — "
                        "dropped from all groups (exclusion is absolute and total)", sb, lst, date)
            continue
        if all(g in refused_groups for g in ("Red", "Green", "Blue") if sb in GROUPS[g]):
            continue                    # its only wideband is refused: never read (Full alone cannot be formed without it)
        sroot = stage_root_fmt % sb if stage_root_fmt else None
        try:
            fl = discover_frames(lst, date, sb, stage_root=sroot)[stokes]
        except RuntimeError:
            fl = []
        if not fl:
            log.warning("group build: no %s frames for %s — dropped from "
                        "all groups", stokes, sb)
            continue
        sx, sy = load_warp_screen(lst, date, sb)
        if flow_applies(sb):
            # flowed frames are in the master frame already: identity screen
            _, h0 = read_frame(fl[0][1])
            sx = np.zeros((int(h0["NAXIS2"]), int(h0["NAXIS1"])), np.float32)
            sy = np.zeros_like(sx)
        if sx is None:
            log.warning("group build: no warp screen for %s — subband "
                        "DROPPED (undewarped frames would smear the coadd)",
                        sb)
            continue
        sb_frames[sb] = fl
        sb_screens[sb] = (sx, sy)
        sb_adm[sb] = read_admission(out_dir, sb)

    if not sb_frames:
        raise RuntimeError("no usable subbands for group build")

    # reference timeline: the subband with the most frames
    ref_sb = max(sb_frames, key=lambda s: len(sb_frames[s]))
    ref_list = sb_frames[ref_sb]
    nt = len(ref_list)
    # each frame's OWN integration time (header DATE-OBS) - pairing, wideband date_obs, 1-minute blocks (frame-time fix)
    sb_dsec = {}
    for sb, fl in sb_frames.items():
        sb_dsec[sb] = dateobs_seconds(fl, sb)
    ref_secs, ref_dob = sb_dsec[ref_sb]
    _, ref_hdr = read_frame(ref_list[0][1])
    ny, nx = int(ref_hdr["NAXIS2"]), int(ref_hdr["NAXIS1"])
    log.info("reference timeline: %s, %d frames; grid %dx%d; subbands: %s",
             ref_sb, nt, ny, nx, ",".join(sorted(sb_frames)))
    # PER-WIDEBAND GRID (fix 2026-09-09, INCIDENT-SNAP-GROUP-BANDS-GREEN-BLUE-
    # EMPTY-FULL-EQUALS-RED-ON-FLEET-CELLS): the CLEANed generation images the
    # three band groups on different grids (1508 / 2358 / 3122 px).  Each
    # wideband (Red, Green, Blue) is a co-add of five equally gridded subbands
    # and takes its OWN grid and header from its first available member; the
    # shape check below is against that grid, not the reference subband's.
    # Full mixes the three grids and is built only when every subband shares
    # one grid (the dirty generation); otherwise it is DISCARDED with a marker
    # file (Gregg 2026-09-09: "You can discard Full").
    grid_of, hdr_of = {}, {}
    for g, sbs in GROUPS.items():
        if g in refused_groups:
            continue
        for sb in sbs:
            if sb in sb_frames:
                _, h = read_frame(sb_frames[sb][0][1])
                grid_of[g] = (int(h["NAXIS2"]), int(h["NAXIS1"]))
                hdr_of[g] = h
                break
    # Full = ALL FIFTEEN subbands: built only when Red, Green AND Blue are all present on one grid.  Before
    # 2026-10-02 the test was "one grid among the widebands present", so a run that saw only the Red subbands
    # (the 2026-09-26 Red-group re-search) built Full = the Red stack and wrote it as Full on 62 cells
    # (INCIDENT-SNAP-FULL-CUTOUTS-EQUAL-RED-62-CELLS; Gregg 2026-10-02 "Go on both preambles").
    for g in GROUPS:                    # a group none of whose members has usable frames is not expected (outcome record)
        if g not in grid_of and g not in refused_groups:
            outcome[g] = "no-members"
    full_ok = len(set(grid_of.values())) == 1 and set(grid_of) == set(GROUPS) and "Full" not in refused_groups
    if full_ok:
        grid_of["Full"], hdr_of["Full"] = list(grid_of.values())[0], ref_hdr
    elif "Full" not in refused_groups:
        reason = ("Full wideband DISCARDED: %s (Gregg 2026-09-09: discard Full; 2026-10-02: Full only with Red, Green "
                  "and Blue all present)" % (
                      ("member widebands are on different grids (%s); a Full co-add would need regridding"
                       % ", ".join("%s %dx%d" % (g, *grid_of[g]) for g in sorted(grid_of)))
                      if len(set(grid_of.values())) > 1 else
                      "widebands present: %s - not all of Red, Green, Blue" % (",".join(sorted(grid_of)) or "none")))
        log.warning(reason)
        with open(os.path.join(out_dir, "wideband_Full_%s_DISCARDED.txt" % stokes), "w") as f:
            f.write(reason + "\n")
    for g in sorted(grid_of):
        log.info("wideband %s grid %dx%d", g, *grid_of[g])
    active_groups = [g for g in ("Red", "Green", "Blue", "Full") if g in grid_of]

    # frame matching per subband: index into that subband's list per ref t - the frame of the SAME INTEGRATION (DATE-OBS
    # within PAIR_TOL_S; frame-time fix 2026-10-07, minimal form: was the nearest FILE-NAME time within 12 s), one-to-one
    # (audit F6); a subband without that integration contributes nothing to the instant, never a neighbouring integration
    sb_match = {}
    for sb, fl in sb_frames.items():
        match = pair_by_integration(ref_secs, sb_dsec[sb][0])
        n_ok = int((match >= 0).sum())
        if n_ok < nt:
            log.info("%s: %d/%d ref frames matched", sb, n_ok, nt)
        sb_match[sb] = match

    cubes = {g: np.zeros((nt,) + grid_of[g], dtype=np.float32) for g in active_groups}
    wsums = {g: np.zeros(nt, dtype=np.float64) for g in active_groups}
    nsubs = {g: np.zeros(nt, dtype=int) for g in active_groups}
    # per-(group, t) weight sums are scalar because each subband contributes
    # one whole frame with one weight; per-pixel NaNs handled at the end
    group_of = {}
    for g, sbs in GROUPS.items():
        for sb in sbs:
            group_of[sb] = g

    # the member frames are added into the wideband cubes by accumulate() (module level, so its two orders can be tested
    # against each other on synthetic frames: tests/test_snapshot_engine.py)
    member_rows = accumulate(nt, sb_frames, sb_match, sb_adm, sb_screens, grid_of, group_of, full_ok, ref_list, sb_dsec, ref_dob,
                             cubes, wsums, nsubs, nthreads, chunk_frames, t0,
                             per_instant=GROUP_THREADS > 0 and read_frame is _STAGED_READ_FRAME)
    # MEMBER TABLE (frame-time fix 2026-10-07): every wideband frame's members, each with its own DATE-OBS, so "the members of
    # a wideband frame share one integration" is checked on the product, and readers rebuild a wideband frame from exactly
    # its members (by member file-name ts) instead of re-pairing by name
    write_csv(os.path.join(out_dir, "wideband_members_%s.csv" % stokes),
              [dict(group=g, ts=ts, date_obs=rdob, subband=sb, member_ts=mts, member_date_obs=mdob, weight="%.6g" % w)
               for g, ts, sb, mts, mdob, rdob, w in sorted(member_rows, key=lambda r: (r[0], r[1], r[2]))])

    results = {}
    night_refused = []
    for g in active_groups:
        w = wsums[g]
        admitted = w > 0
        nzw = np.where(w > 0, w, 1.0).astype(np.float32)
        cubes[g] /= nzw[:, None, None]
        # per-instant composition travels with the frame so the persisted
        # noise table records it (audit F2: group noise is non-stationary;
        # the per-image global rms in _search_one_stokes normalises it)
        meta = [{"ts": ts, "date_obs": ref_dob[p], "bmaj": np.nan, "bmin": np.nan,
                 "n_subbands": int(nsubs[g][t]), "wsum": float(w[t])}
                for t, (ts, p) in enumerate(ref_list)]
        log.info("group %s: %d/%d frames with data (subbands per frame: "
                 "min %d, median %.0f, max %d)", g, int(admitted.sum()), nt,
                 int(nsubs[g][admitted].min()) if admitted.any() else 0,
                 float(np.median(nsubs[g][admitted])) if admitted.any() else 0,
                 int(nsubs[g][admitted].max()) if admitted.any() else 0)
        if admitted.sum() < 10:
            log.warning("group %s: too few frames, skipped", g)
            if g in outcome:
                outcome[g] = "too-few-frames"
            continue
        try:
            results[g] = _search_one_stokes(
                cubes[g], meta, hdr_of[g], admitted, stokes, "median",
                lst, date, g, out_dir, thresh, None, None)
        except NightRuleRefused as e:
            # a group refused by the night rule (a constituent below 46 MHz flagged) is RECORDED and the other groups are
            # still searched (2026-10-13, ruling 4; before, the refusal ended the whole group search)
            write_night_refusal(out_dir, e.record); night_refused.append(str(e)); cubes[g] = None
            if g in outcome:
                outcome[g] = "night-refused"
            continue
        cubes[g] = None  # free
        rp = os.path.join(out_dir, "runinfo_%s_%s_median.json" % (g, stokes))
        ri = json.load(open(rp))
        ri["wideband_pairing"] = ("members = frames of the reference frame's integration (header DATE-OBS within %.1f s; "
                                  "frame-time fix 2026-10-07); reference subband %s; members in wideband_members_%s.csv"
                                  % (PAIR_TOL_S, ref_sb, stokes))
        ri["wideband_excluded_members"] = sorted(s for s in (SCIENCE_SUBBANDS if g == "Full" else GROUPS[g]) if s in excl)
        with open(rp + ".tmp", "w") as f:
            json.dump(ri, f, indent=2)
        os.replace(rp + ".tmp", rp)
        if g in outcome:
            outcome[g] = "searched"
    log.info("group search done in %.0f s", time.time() - t0)
    if night_refused:   # still ends in a refusal, as before (the other groups' products are written)
        raise RuntimeError("; ".join(night_refused) + " - refusal recorded (nightrefused_<group>_<Stokes>_median.json)")
    return results


class _FlowCubeFrames:
    """One cell's member frames from its consolidated subband flow cubes, in the staged-frame interface the group search reads:
    discover(lst, date, sb, stage_root) -> {'I': [(ts, key)], 'V': [...]}, read(key) -> (frame, its own header), dateobs(keys) ->
    {key: DATE-OBS}.  Only the members of the one group being rebuilt are offered (the others read as no frames: their outcome is
    no-members and Full is never formed).  astropy 4.1 decompresses a CompImageHDU whole, so a member cube is loaded once and the
    two most recent are kept (the regen runs a single time chunk, member after member)."""

    def __init__(self, lst, date, members, stokes):
        from phase3_snapshot_search import CUBE_ROOT
        import threading
        from collections import OrderedDict
        self.root, self.lst, self.date, self.members, self.stokes = CUBE_ROOT, lst, date, set(members), stokes
        self.tab, self.cache, self.lock = {}, OrderedDict(), threading.Lock()

    def _path(self, sb):
        cell = "%s_%s" % (self.lst, self.date)
        return os.path.join(self.root, cell, "%s_%s_%s_flowcube.fits" % (cell, sb, self.stokes))

    def _table(self, sb):
        if sb not in self.tab:
            from astropy.io import fits
            with fits.open(self._path(sb), memmap=True) as h:
                t = h["FRAME_TABLE"].data
                self.tab[sb] = ([str(x) for x in t["ts"]], [fits.Header.fromstring(str(x), sep="\n") for x in t["header"]])
        return self.tab[sb]

    def discover(self, lst, date, sb, stage_root=None, **kw):
        out = {"I": [], "V": []}
        if sb in self.members and os.path.exists(self._path(sb)):
            # one frame per integration (2026-10-25): a member cube holding two Phase-1 runs offers only the search's own frames
            # (phase3_snapshot_search.flowcube_frame_index; keys keep the cube index, so read() takes exactly those frames)
            from phase3_snapshot_search import flowcube_frame_index
            tss, hdrs = self._table(sb)
            out[self.stokes] = [(tss[k], "flowcube:%s:%d" % (sb, k)) for k in flowcube_frame_index(lst, date, sb, self.stokes, tss, hdrs)]
        return out

    def read(self, key):
        _, sb, k = key.split(":"); k = int(k)
        with self.lock:
            if sb not in self.cache:
                from astropy.io import fits
                t1 = time.time()
                with fits.open(self._path(sb), memmap=True) as h:
                    self.cache[sb] = np.array(h["FRAMES"].data, dtype=np.float32)
                log.info("member cube %s %s loaded in %.0f s", sb, self.stokes, time.time() - t1)
                while len(self.cache) > 2:
                    self.cache.popitem(last=False)
            cube = self.cache[sb]
        return cube[k], self._table(sb)[1][k]

    def dateobs(self, keys, nthreads=16):
        out = {}
        for key in keys:
            _, sb, k = key.split(":")
            out[key] = str(self._table(sb)[1][int(k)].get("DATE-OBS", ""))
        return out


def regen_group_cutouts(lst, date, group, stokes, out_dir):
    """WIDEBAND PHOTOMETRY CUTOUTS FROM THE CURRENT CUBES (2026-10-25; Gregg: "Forced photometry is also approved - HWJ and 10pc, then
    solar system planets"; ledger PREAMBLE-LAUNCHED-PHOTOMETRY-CUTOUT-REGEN-GEN5-20261025, second step): run_group_search's own path -
    the deep exclusion rule, excluded members dropped, pairing by integration, dewarp (identity: flowed frames), 1/rms^2 weights,
    _search_one_stokes - for ONE band group, with the member frames read from the consolidated subband flow cubes instead of the staged
    Phase-1 frames, each member's admission recomputed from its own V cube by the search's admission_table (what run_search writes as
    admission_<sb>.csv and the group search reads), then stops after the cutouts (P3_SNAP_CUTOUTS_ONLY).  Never an injected run.
    Returns 'excluded' | 'refused: <why>' | the number of cutout cubes written."""
    import phase3_snapshot_search as S
    if not FLOW_MODE:
        raise RuntimeError("REFUSED: regen_group_cutouts needs P3_SNAP_FLOW=1 at start-up (the cubes are flow-registered frames)")
    if "P3_SNAP_CUTOUT_ROOT" not in os.environ:
        raise RuntimeError("REFUSED: regen_group_cutouts needs P3_SNAP_CUTOUT_ROOT (a new root - never over the existing product)")
    if group not in GROUPS:
        raise RuntimeError("REFUSED: group %r is not one of %s" % (group, sorted(GROUPS)))
    os.environ["P3_SNAP_CUTOUTS_ONLY"] = "1"; os.environ.pop("P3_SNAP_INJECTED", None)
    sys.path.insert(0, "/lustre/gh/main/exopipe")
    from exopipe.phase3.coadd_full_pipeline import load_exclusions, is_excluded
    if is_excluded(lst, group, date, load_exclusions(scope=_excl_scope(lst, date))):
        log.warning("CUTOUTS-REGEN-WIDEBAND %s %s %s %s: excluded under the deep rule - not built", lst, date, group, stokes)
        return "excluded"
    os.makedirs(out_dir, exist_ok=True)
    excl = excluded_subbands(lst, date)
    members = [sb for sb in GROUPS[group] if sb not in excl]
    if not any(os.path.exists(_FlowCubeFrames(lst, date, [sb], stokes)._path(sb)) for sb in members):
        # no member subband of this group has a consolidated cube (e.g. 12h 2025-05-01: 73-82 MHz only) - the search built no such
        # wideband and there is nothing to regenerate (2026-10-26; before, run_group_search's 'no usable subbands' refused and retried)
        log.warning("CUTOUTS-REGEN-WIDEBAND %s %s %s %s: no member cube - nothing to build (the search built none)", lst, date, group, stokes)
        return "no-members"
    for sb in members:
        if not os.path.exists(_FlowCubeFrames(lst, date, [sb], "V")._path(sb)):
            continue
        if os.path.getsize(os.path.join(out_dir, "admission_%s.csv" % sb)) if os.path.exists(os.path.join(out_dir, "admission_%s.csv" % sb)) else 0:
            log.info("admission_%s.csv present in %s (this job's other Stokes) - reused", sb, out_dir)
            continue
        v_cube, meta_v, _ = S.load_cube_from_flowcube(lst, date, sb, "V")
        adm_rows, _, _ = S.admission_table(v_cube, None, meta_v, ctx=(lst, date, sb))
        write_csv(os.path.join(out_dir, "admission_%s.csv" % sb), adm_rows)
        del v_cube
    src = _FlowCubeFrames(lst, date, members, stokes)

    def _search_then_runinfo(cube, meta, hdr0, admitted, st, mode, lst_, date_, tag_band, out, *a, **k):
        # the cutouts-only stop returns before runinfo_<tag>.json, which the group search reads next: written here (scratch)
        info = S._search_one_stokes(cube, meta, hdr0, admitted, st, mode, lst_, date_, tag_band, out, *a, **k)
        with open(os.path.join(out, "runinfo_%s_%s_%s.json" % (tag_band, st, mode)), "w") as f:
            json.dump(info, f, indent=2, default=str)
        return info

    g = globals()
    saved = {k: g[k] for k in ("discover_frames", "read_frame", "frame_dateobs", "_search_one_stokes")}
    g.update(discover_frames=src.discover, read_frame=src.read, frame_dateobs=src.dateobs, _search_one_stokes=_search_then_runinfo)
    try:
        res = run_group_search(lst, date, stokes, out_dir, chunk_frames=1000000)
    except RuntimeError as e:
        if "night" not in str(e).lower():
            raise
        log.warning("CUTOUTS-REGEN-WIDEBAND %s %s %s %s: refused: %s", lst, date, group, stokes, e)
        return "refused: %s" % e
    finally:
        g.update(saved)
    n = (res.get(group) or {}).get("photometry_cutout_cubes")
    log.info("CUTOUTS-REGEN-WIDEBAND %s %s %s %s: %s cutout cubes", lst, date, group, stokes, n)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lst", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--stokes", required=True, choices=["I", "V"])
    ap.add_argument("--thresh", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage-root-fmt", default=None)
    ap.add_argument("--nthreads", type=int, default=6)
    ap.add_argument("--cutouts-from-cubes", default="", metavar="GROUP",
                    help="regenerate one band group's photometry cutouts from the subband flow cubes, then stop (regen_group_cutouts)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.cutouts_from_cubes:
        r = regen_group_cutouts(args.lst, args.date, args.cutouts_from_cubes, args.stokes, args.out)
        raise SystemExit(3 if r == "excluded" else 0)
    run_group_search(args.lst, args.date, args.stokes, args.out,
                     thresh=args.thresh, stage_root_fmt=args.stage_root_fmt,
                     nthreads=args.nthreads)


if __name__ == "__main__":
    main()
