#!/usr/bin/env python
"""Snapshot-tier injection battery support (SNAP-BAT-1, 2026-09-04).

Two entry points:

1. make_manifest: generate the injection manifest for one cell — random
   sky positions (elevation >= 20 deg, > 3 deg from A-team sources,
   inside the projection), a log flux grid, four temporal classes, both
   Stokes independently (V injections carry random handedness sign).
   Sparse by construction: no frame ever holds more than MAX_PER_FRAME
   active injections in one (subband, Stokes).

2. inject_cube: called by phase3_snapshot_search when --inject is given —
   adds header-beam-shaped Gaussians to the in-memory cube at the
   raw-frame pixel (dewarped position + warp screen), archive untouched.

Recovery matching lives in phase3_snapshot_recover.py.
"""
import argparse
import csv
import json
csv.field_size_limit(100000000)
import math
import os
import random

import numpy as np

ATEAM = [(350.850, 58.815), (299.868, 40.734), (83.633, 22.015),
         (187.706, 12.391)]
FLUX_GRID_JYB = [2.0, 3.5, 6.0, 10.0, 18.0, 32.0, 56.0, 100.0]
TEMPORAL = [("1f", 1), ("3f", 3), ("18f", 18), ("60f", 60)]
MAX_PER_FRAME = 5


def _sepd(r1, d1, r2, d2):
    a, b, c, d = map(math.radians, (r1, d1, r2, d2))
    return math.degrees(math.acos(max(-1, min(1,
        math.sin(b) * math.sin(d) +
        math.cos(b) * math.cos(d) * math.cos(a - c)))))


def make_manifest(out_csv, pc_ra, pc_dec, n_frames, n_per_bin=8, seed=20260904):
    """n_per_bin injections per (flux bin x temporal class x Stokes).
    With 15 subbands each injection appears in EVERY subband (same sky
    position, per-subband pixel), so per-bin statistics aggregate to
    n_per_bin x 15 >= 100 easily."""
    rng = random.Random(seed)
    rows = []
    frame_load = {}
    inj_id = 0
    for st in ("I", "V"):
        for fl in FLUX_GRID_JYB:
            for tname, nf in TEMPORAL:
                placed = 0
                attempts = 0
                while placed < n_per_bin and attempts < 10000:
                    attempts += 1
                    # uniform on cap within 70 deg of phase centre
                    zmax = math.radians(70)
                    z = math.acos(1 - rng.random() * (1 - math.cos(zmax)))
                    az = rng.random() * 2 * math.pi
                    dec = math.degrees(math.asin(
                        math.sin(math.radians(pc_dec)) * math.cos(z) +
                        math.cos(math.radians(pc_dec)) * math.sin(z) * math.cos(az)))
                    ra = pc_ra + math.degrees(math.atan2(
                        math.sin(az) * math.sin(z),
                        math.cos(z) * math.cos(math.radians(pc_dec)) -
                        math.sin(z) * math.sin(math.radians(pc_dec)) * math.cos(az)))
                    ra %= 360.0
                    if min(_sepd(ra, dec, a, b) for a, b in ATEAM) < 3.0:
                        continue
                    t0 = rng.randint(0, max(0, n_frames - nf - 1))
                    ok = True
                    for t in range(t0, t0 + nf):
                        if frame_load.get((st, t), 0) >= MAX_PER_FRAME:
                            ok = False
                            break
                    if not ok:
                        continue
                    for t in range(t0, t0 + nf):
                        frame_load[(st, t)] = frame_load.get((st, t), 0) + 1
                    rows.append({
                        "inj_id": inj_id, "stokes": st,
                        "ra_deg": "%.5f" % ra, "dec_deg": "%.5f" % dec,
                        "flux_jyb": fl, "profile": tname,
                        "t0": t0, "n_frames": nf,
                        "sign": rng.choice([1, -1]) if st == "V" else 1,
                    })
                    inj_id += 1
                    placed += 1
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def load_manifest(path, stokes, subband=None):
    """Manifest rows of one Stokes.  SNAP-BAT-2 manifests carry a per-subband
    flux column flux_<subband> (SNR x that subband's median frame rms); when
    present and a subband is given it replaces flux_jyb for the injection."""
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["stokes"] == stokes]
    if subband:
        col = "flux_%s" % subband
        for r in rows:
            if r.get(col):
                r["flux_jyb"] = r[col]
    return rows


def make_manifest_snr(out_csv, cell_out_dir, lst, date, n_per_stokes=1000,
                      snr_lo=5.0, snr_hi=40.0, min_el_deg=30.0, max_per_frame=12,
                      seed=20260909, subbands=None):
    """SNAP-BAT-2 manifest (Gregg 2026-09-09; injections drawn in SNR, not Jy).

    For each Stokes: n_per_stokes injections, SNR log-uniform in [snr_lo, snr_hi],
    flux per subband = SNR x the cell's median admitted-frame global rms for that
    (subband, Stokes), read from the BASELINE search's noise tables in
    cell_out_dir.  Positions uniform on the sky above min_el_deg at the injection
    time (zenith at the frame's LST), no avoidance of ordinary bright sources
    (the 2-D curve is what the battery calibrates); A-team avoidance 3 deg kept.
    Durations: half one frame ('1f'), half six frames ('6f', the 1-minute tier).
    Stokes V handedness random.  Sparse: at most max_per_frame active injections
    per (Stokes, frame)."""
    import glob
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from phase3_snapshot_search import (SCIENCE_SUBBANDS, lst_deg_at,
                                        ts_to_epoch_seconds)
    from phase3_snapshot_astro import OVRO_LAT_DEG
    subbands = subbands or SCIENCE_SUBBANDS
    rng = random.Random(seed)
    # frame timeline + per-(subband, Stokes) median rms from the baseline noise tables
    rms, timeline = {}, None
    for sb in subbands:
        for st in ("I", "V"):
            npth = os.path.join(cell_out_dir, "noise_%s_%s_median.csv" % (sb, st))
            if not os.path.exists(npth):
                continue
            vals, tss = [], []
            for r in csv.DictReader(open(npth)):
                tss.append(r["ts"])
                if r.get("admitted") == "1" and r.get("global_rms_jyb"):
                    vals.append(float(r["global_rms_jyb"]))
            if vals:
                rms[(sb, st)] = float(np.median(vals))
            if timeline is None or len(tss) > len(timeline):
                timeline = tss
    if not rms or not timeline:
        raise RuntimeError("make_manifest_snr: no baseline noise tables in %s" % cell_out_dir)
    n_frames = len(timeline)
    epochs = [ts_to_epoch_seconds(t) for t in timeline]
    rows, frame_load, inj_id = [], {}, 0
    zmax = math.radians(90.0 - min_el_deg)
    for st in ("I", "V"):
        sbs = [sb for sb in subbands if (sb, st) in rms]
        placed, attempts = 0, 0
        while placed < n_per_stokes and attempts < 200000:
            attempts += 1
            nf = rng.choice([1, 6])
            t0 = rng.randint(0, max(0, n_frames - nf - 1))
            if any(frame_load.get((st, t), 0) >= max_per_frame for t in range(t0, t0 + nf)):
                continue
            # zenith at the injection time: RA = LST, Dec = latitude
            lst_deg = lst_deg_at(epochs[t0])
            z = math.acos(1 - rng.random() * (1 - math.cos(zmax)))
            az = rng.random() * 2 * math.pi
            lat = math.radians(OVRO_LAT_DEG)
            dec = math.degrees(math.asin(math.sin(lat) * math.cos(z) +
                                         math.cos(lat) * math.sin(z) * math.cos(az)))
            ra = (lst_deg + math.degrees(math.atan2(
                math.sin(az) * math.sin(z),
                math.cos(z) * math.cos(lat) - math.sin(z) * math.sin(lat) * math.cos(az)))) % 360.0
            if min(_sepd(ra, dec, a, b) for a, b in ATEAM) < 3.0:
                continue
            snr = math.exp(rng.uniform(math.log(snr_lo), math.log(snr_hi)))
            for t in range(t0, t0 + nf):
                frame_load[(st, t)] = frame_load.get((st, t), 0) + 1
            row = {"inj_id": inj_id, "stokes": st, "ra_deg": "%.5f" % ra, "dec_deg": "%.5f" % dec,
                   "snr": "%.3f" % snr, "profile": "1f" if nf == 1 else "6f",
                   "t0": t0, "n_frames": nf, "sign": rng.choice([1, -1]) if st == "V" else 1,
                   "el_deg": "%.2f" % (90.0 - math.degrees(z)),
                   "flux_jyb": "%.5g" % (snr * float(np.median([rms[(sb, st)] for sb in sbs])))}
            for sb in subbands:
                row["flux_%s" % sb] = "%.5g" % (snr * rms[(sb, st)]) if (sb, st) in rms else ""
            rows.append(row)
            inj_id += 1
            placed += 1
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    meta = {"cell": "%s_%s" % (lst, date), "n_per_stokes": n_per_stokes, "snr_lo": snr_lo,
            "snr_hi": snr_hi, "min_el_deg": min_el_deg, "max_per_frame": max_per_frame,
            "seed": seed, "n_frames": n_frames, "rms_jyb": {"%s_%s" % k: v for k, v in rms.items()}}
    with open(out_csv.replace(".csv", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return len(rows)


def inject_cube(cube, hdr, screen_x, screen_y, manifest_rows, log=None):
    """Add beam-shaped Gaussians in place.  Returns per-injection pixel
    bookkeeping [(inj_id, px, py, t0, t1)]."""
    from astropy.wcs import WCS
    w = WCS(hdr).celestial
    bmaj_px = abs(float(hdr["BMAJ"]) / float(hdr["CDELT2"]))
    bmin_px = abs(float(hdr["BMIN"]) / float(hdr["CDELT2"]))
    bpa = math.radians(float(hdr.get("BPA", 0.0)))
    sx = bmaj_px / 2.3548
    sy = bmin_px / 2.3548
    half = int(math.ceil(3.5 * sx))
    ny, nx = cube.shape[1:]
    book = []
    for r in manifest_rows:
        ra, dec = float(r["ra_deg"]), float(r["dec_deg"])
        px, py = w.all_world2pix(ra, dec, 0)
        px, py = float(px), float(py)
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        iy, ix = int(round(py)), int(round(px))
        if not (0 <= iy < ny and 0 <= ix < nx):
            continue
        if screen_x is not None:
            px += float(screen_x[iy, ix])
            py += float(screen_y[iy, ix])
        x0, x1 = int(px) - half, int(px) + half + 1
        y0, y1 = int(py) - half, int(py) + half + 1
        if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
            continue
        gy, gx = np.mgrid[y0:y1, x0:x1]
        dx, dy = gx - px, gy - py
        u = dx * math.sin(bpa) + dy * math.cos(bpa)
        v = -dx * math.cos(bpa) + dy * math.sin(bpa)
        stamp = (float(r["flux_jyb"]) * int(r["sign"]) *
                 np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))).astype(np.float32)
        t0 = int(r["t0"])
        t1 = min(t0 + int(r["n_frames"]), cube.shape[0])
        cube[t0:t1, y0:y1, x0:x1] += stamp[None, :, :]
        book.append((int(r["inj_id"]), px, py, t0, t1))
    if log:
        log.info("injected %d/%d manifest rows", len(book), len(manifest_rows))
    return book


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--snr-from-cell", default=None,
                    help="SNAP-BAT-2: baseline cell out dir; manifest drawn in SNR from its noise tables")
    ap.add_argument("--lst", default=None)
    ap.add_argument("--date", default=None)
    ap.add_argument("--n-per-stokes", type=int, default=1000)
    ap.add_argument("--pc-ra", type=float, default=None)
    ap.add_argument("--pc-dec", type=float, default=None)
    ap.add_argument("--n-frames", type=int, default=358)
    ap.add_argument("--n-per-bin", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()
    if args.snr_from_cell:
        n = make_manifest_snr(args.out, args.snr_from_cell, args.lst, args.date,
                              n_per_stokes=args.n_per_stokes, seed=args.seed)
    else:
        n = make_manifest(args.out, args.pc_ra, args.pc_dec, args.n_frames,
                          args.n_per_bin, args.seed)
    print("wrote %d injections -> %s" % (n, args.out))


if __name__ == "__main__":
    main()
