#!/usr/bin/env python3
"""jupiter_flag_backfill.py - the Phase-1 Jupiter check of V-noise-flagged integrations (jupiter_flag_qa, Gregg 2026-10-07 ruling
on question 12) for nights ALREADY in the Phase-1 archive, plus a census and a content test.

Where the flag decision of an archived night is recorded (checked 2026-10-07 on the Orca archive /lustre/pipeline/exopipe/phase1):
  1. QA/snapshot_qa_<subband>.csv, column 'flagged' (Orca runs since 2026-08-11): the decision itself, with every
     integration's statistic (so the night's threshold is exact);
  2. snapshots_clean/: a flagged integration's CLEAN Stokes I snapshot is empty (the integration was flagged in the MS before
     it was imaged), so its fpack-compressed file is ~0.45 MB against ~15-18 MB (runs with clean_snapshots);
  3. otherwise the decision is recomputed from the archived pilot V images exactly as Phase 1 computed it (std of the central
     1024 x 1024 pixels; median + 3 x MAD-sigma / half the median).  The archived pilots are fpack-quantised (RICE_1,
     SUBTRACTIVE_DITHER_1), so the recomputed statistic differs from Phase 1's in the last digits; whenever all pilots are read
     the recomputed flagged set is compared with the record and any difference is reported.
The night's threshold is needed only when a flagged integration shows Jupiter (step 2 of the test); then it comes from record 1,
or from reading every pilot V of that subband-night.

Modes:
  census  (metadata only: directory listings and file sizes)  -> per (cell, subband): run, pilot era, record source, flagged
          count, bytes of the flagged integrations' pilot V / I files.
  run     classify the flagged integrations of the given cells and copy the Jupiter-caused ones (copy, never move) to
          --dest-root/<LST>/<date>/<Run>/<subband>/ with the record, MANIFEST.sha256, manifest.csv and README.txt.
  verify  content test of a destination tree (see verify_tree).

The Jupiter window: cells of the archive whose Jupiter elevation at mid-hour (first pilot integration + 30 min) is >= --min-el
(default 30 deg), the Jupiter program's definition (Gregg 2026-10-04: nights with Jupiter > 30 deg).
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_HERE) == "phase1" and os.path.basename(os.path.dirname(_HERE)) == "exopipe":
    sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))      # the exopipe repo root (installed layout)
try:
    from exopipe.phase1 import jupiter_flag_qa as jfq
except ImportError:                                                  # scratch layout: module next to this file
    sys.path.insert(0, _HERE)
    import jupiter_flag_qa as jfq

PHASE1_ROOT = "/lustre/pipeline/exopipe/phase1"
CLEAN_EMPTY_FRACTION = 0.1   # a CLEAN-I snapshot below 10% of the run's median file size is empty (flagged)
RMS_TOLERANCE = 0.02         # recomputed statistic from the fpack-quantised archive vs the recorded one (relative)


def excluded(lst, date, subbands):
    """data_decisions.csv exclusions through the single reader (CLAUDE.md 2: exclusion is absolute; an excluded subband-night
    is neither classified nor copied)."""
    from exopipe.common.exclusions import excluded_subbands_for_night
    return excluded_subbands_for_night(lst, date, list(subbands))


def _location():
    from exopipe.common.pipeline_config import OVRO_LOC          # the observatory location's single source of truth
    return OVRO_LOC


# ----------------------------------------------------------------------------------------------------------------------------
#  archive layout
# ----------------------------------------------------------------------------------------------------------------------------
TS_RE = re.compile(r"-(\d{8}_\d{6})-")


def latest_runs(cell_dir):
    """{subband: subband_dir}: the latest Run_ wins (phase2_controller rule)."""
    m = {}
    for rd in sorted(glob.glob(os.path.join(cell_dir, "Run_*"))):
        for sd in glob.glob(os.path.join(rd, "*MHz")):
            if os.path.isdir(sd):
                m[os.path.basename(sd)] = sd
    return m


def list_pilots(sb_dir):
    """(pilot V list sorted = the flag's index order, era, psf list).  era 'IV': -I-image and -V-image; 'V': V-only '-image'."""
    snap = os.path.join(sb_dir, "snapshots")
    names = sorted(os.listdir(snap)) if os.path.isdir(snap) else []
    vIV = [n for n in names if "Pilot-Snapshot-" in n and "-V-image.fits" in n]
    vV = [n for n in names if "Pilot-Snapshot-" in n and re.search(r"-\d{8}_\d{6}-image\.fits", n)]
    psf = [n for n in names if "Pilot-Snapshot-" in n and "-psf.fits" in n]
    nI = sum(1 for n in names if "Pilot-Snapshot-" in n and "-I-image.fits" in n)
    if vIV:
        return [os.path.join(snap, n) for n in vIV], ("IV" if nI else "V-named"), [os.path.join(snap, n) for n in psf]
    return [os.path.join(snap, n) for n in vV], "V", [os.path.join(snap, n) for n in psf]


def flag_record(sb_dir, subband, pilot_v):
    """(source, flagged index list or None, {idx: rms} or None, {idx: scan} or None) from what the archive records."""
    qa_csv = os.path.join(sb_dir, "QA", "snapshot_qa_%s.csv" % subband)
    if os.path.exists(qa_csv):
        rows = list(csv.DictReader(open(qa_csv, newline="")))
        names = {os.path.basename(p).split(".fits")[0]: i for i, p in enumerate(pilot_v)}
        fl, rms, scans = [], {}, {}
        for r in rows:
            i = int(r["snapshot_index"])
            stem = r["filename"].split(".fits")[0]
            if names.get(stem, i) != i:
                raise RuntimeError("snapshot_qa index %d names %s, the archive's index %d" % (i, stem, names.get(stem)))
            rms[i] = float(r["rms_jy_per_beam"])
            scans[i] = r["scan_number"]
            if r["flagged"].strip().lower() == "true":
                fl.append(i)
        return "snapshot_qa_csv", sorted(fl), rms, scans
    cdir = os.path.join(sb_dir, "snapshots_clean")
    if os.path.isdir(cdir):
        sizes = {}
        for e in os.scandir(cdir):
            if "Clean-Snapshot-" in e.name and re.search(r"-\d{8}_\d{6}-image\.fits", e.name):
                sizes[TS_RE.search(e.name).group(1)] = e.stat().st_size
        if sizes:
            med = float(np.median(list(sizes.values())))
            tsi = {TS_RE.search(os.path.basename(p)).group(1): i for i, p in enumerate(pilot_v)}
            if set(sizes) == set(tsi):
                return "clean_i_empty", sorted(tsi[t] for t, s in sizes.items() if s < CLEAN_EMPTY_FRACTION * med), None, None
            return "clean_i_unmatched", None, None, None
    return "none", None, None, None


# ----------------------------------------------------------------------------------------------------------------------------
#  workers
# ----------------------------------------------------------------------------------------------------------------------------
_PSF = {}


def _psf(path):
    if path not in _PSF:
        _PSF.clear()
        _PSF[path] = jfq.load_psf(path)
    return _PSF[path]


def _stat_job(path):
    """The flag statistic of one archived pilot V, the way Orca computes it (Python float of np.nanstd of the float32 box)."""
    try:
        V, _ = jfq.load_image(path)
        return float(np.nanstd(V[jfq._box(V.shape)]))
    except Exception:
        return None


def _measure_job(args):
    idx, vp, ip, psf_path = args
    try:
        return idx, jfq.measure_integration(vp, _location(), psf=_psf(psf_path) if psf_path else None, i_path=ip), None
    except Exception as e:
        return idx, {"status": "error", "v_file": os.path.basename(vp)}, "%s: %s" % (type(e).__name__, e)


# ----------------------------------------------------------------------------------------------------------------------------
#  one subband-night
# ----------------------------------------------------------------------------------------------------------------------------
def process_subband(pool, lst, date, sb_dir, dest_root, copy=True, copy_reasons=jfq.COPY_REASONS):
    t0 = time.time()
    subband = os.path.basename(sb_dir)
    run = os.path.basename(os.path.dirname(sb_dir))
    pilot_v, era, psfs = list_pilots(sb_dir)
    out = dict(lst=lst, date=date, run=run, subband=subband, era=era, n_frames=len(pilot_v), record_source="", n_flagged=0,
               n_jupiter=0, n_other=0, n_detected=0, threshold_source="", record_vs_recomputed="", n_read=0, copied=0,
               copy_bytes=0, wall_s=0.0, error="")
    if not pilot_v:
        out["error"] = "no pilot V"
        return out, []
    src, flagged, rms, scans = flag_record(sb_dir, subband, pilot_v)
    out["record_source"] = src
    psf_path = psfs[0] if psfs else None
    all_rms = None
    if flagged is None:                       # no record: recompute the decision from every pilot V
        all_rms = pool.map(_stat_job, pilot_v, chunksize=4)
        out["n_read"] += len(pilot_v)
        stats = [dict(idx=i, rms=r) for i, r in enumerate(all_rms) if r is not None]
        med, hi, lo = jfq.flag_thresholds([s["rms"] for s in stats])
        flagged = jfq.flagged_from_thresholds(stats, hi, lo)
        out["record_source"] = "recomputed"
    out["n_flagged"] = len(flagged)
    if not flagged:
        out["wall_s"] = round(time.time() - t0, 1)
        return out, []
    jobs = [(i, pilot_v[i], jfq.i_path_for(pilot_v[i]), psf_path) for i in flagged]
    meas = dict((i, (m, e)) for i, m, e in pool.map(_measure_job, jobs, chunksize=1))
    out["n_read"] += len(jobs) * (2 if era == "IV" else 1)
    detected = [i for i in flagged if meas[i][1] is None and meas[i][0].get("status") == "measured"
                and np.isfinite(meas[i][0].get("snr_v", np.nan)) and meas[i][0]["snr_v"] >= jfq.DETECT_SNR]
    out["n_detected"] = len(detected)
    # the night's threshold
    if rms is not None:
        stats = [dict(idx=i, rms=r) for i, r in sorted(rms.items())]
        out["threshold_source"] = "snapshot_qa_csv"
    else:
        if all_rms is None and detected:
            all_rms = pool.map(_stat_job, pilot_v, chunksize=4)
            out["n_read"] += len(pilot_v)
        if all_rms is not None:
            stats = [dict(idx=i, rms=r) for i, r in enumerate(all_rms) if r is not None]
            out["threshold_source"] = "recomputed_all_frames"
        else:
            stats = None
            out["threshold_source"] = "not_needed (no flagged frame shows Jupiter)"
    if stats is not None:
        med, hi, lo = jfq.flag_thresholds([s["rms"] for s in stats])
        rec = jfq.flagged_from_thresholds(stats, hi, lo)
        diff_a, diff_b = sorted(set(rec) - set(flagged)), sorted(set(flagged) - set(rec))
        out["record_vs_recomputed"] = "identical" if not (diff_a or diff_b) else (
            "recomputed-only %s; record-only %s" % (diff_a[:10], diff_b[:10]))
        rms_of = {s["idx"]: s["rms"] for s in stats}
    else:
        med = hi = lo = np.nan
        rms_of = {}
    rows = []
    for i in flagged:
        m, err = meas[i]
        qa = rms_of.get(i, m.get("qa_rms_recomputed", np.nan))
        if err:
            reason, detail = "other", "error: " + err
        elif not np.isfinite(hi):
            reason, detail = ("other", "jupiter_not_detected") if m.get("status") == "measured" else ("other", m.get("status"))
        else:
            reason, detail = jfq.classify(m, float(qa), float(hi), float(lo))
            rc = m.get("qa_rms_recomputed")
            if rms is not None and rc is not None and abs(rc - qa) > RMS_TOLERANCE * abs(qa):
                reason, detail = "other", "file_mismatch: recomputed statistic %.6g vs record %.6g" % (rc, qa)
        if out["record_source"] != "recomputed" and stats is not None and i not in rec and reason == "jupiter":
            detail += "; not flagged by the recomputed threshold"
        row = {k: "" for k in jfq.FIELDS}
        row.update({k: v for k, v in m.items() if k in jfq.FIELDS})
        row.update(test_version=jfq.TEST_VERSION, lst=lst, date=date, run=run, subband=subband, snapshot_index=i,
                   scan_number=(scans.get(i, "") if scans else ""), timestamp=TS_RE.search(os.path.basename(pilot_v[i])).group(1),
                   flag_side=("low" if np.isfinite(lo) and qa < lo else "high"), qa_rms=float(qa), qa_median=float(med),
                   qa_high=float(hi), qa_low=float(lo), reason=reason, detail=detail,
                   psf_file=os.path.basename(psf_path) if psf_path else "")
        rows.append(row)
    out["n_jupiter"] = sum(r["reason"] == "jupiter" for r in rows)
    out["n_other"] = len(rows) - out["n_jupiter"]
    dest = jfq.dest_dir_for(lst, date, run, subband, dest_root)
    if rows:                                  # the record of every flagged integration, copies or not
        os.makedirs(dest, exist_ok=True)
        jfq.write_record(rows, os.path.join(dest, jfq.RECORD_NAME.format(subband=subband)))
        if copy:
            s = jfq.copy_jupiter_rows(rows, os.path.join(sb_dir, "snapshots"), dest, record_path=None,
                                      source_run_dir=sb_dir, copy_reasons=copy_reasons,
                                      written_by="jupiter_flag_backfill (archive backfill)")
            out["copied"] = s["copied"] + s["present"]
            out["copy_bytes"] = s["bytes"]
            if s["missing"]:
                out["error"] = "missing in archive: %s" % s["missing"][:5]
    out["wall_s"] = round(time.time() - t0, 1)
    return out, rows


# ----------------------------------------------------------------------------------------------------------------------------
#  cells
# ----------------------------------------------------------------------------------------------------------------------------
def jupiter_window(phase1_root, min_el):
    """[(lst, date, el_mid)] of archive cells with Jupiter's elevation at mid-hour >= min_el."""
    from astropy.time import Time
    import astropy.units as u
    loc = _location()
    out = []
    for cd in sorted(glob.glob(os.path.join(phase1_root, "[0-2][0-9]h", "20*"))):
        lst, date = cd.split("/")[-2], cd.split("/")[-1]
        t_first = None
        for sd in sorted(latest_runs(cd).values()):
            snap = os.path.join(sd, "snapshots")
            if os.path.isdir(snap):
                for n in sorted(os.listdir(snap)):
                    m = TS_RE.search(n)
                    if m and "Pilot-Snapshot-" in n:
                        t_first = m.group(1)
                        break
            if t_first:
                break
        if not t_first:
            continue
        t = Time("%s-%s-%sT%s:%s:%s" % (t_first[:4], t_first[4:6], t_first[6:8], t_first[9:11], t_first[11:13], t_first[13:15]),
                 scale="utc") + 30 * u.min
        _, el = jfq.jupiter_direction(t, loc)
        if el >= min_el:
            out.append((lst, date, round(el, 1)))
    return out


def census(cells, phase1_root, out_csv, nproc=1):
    if nproc > 1:
        with Pool(nproc) as p:
            parts = p.starmap(_census_cell, [(c, phase1_root) for c in cells], chunksize=1)
    else:
        parts = [_census_cell(c, phase1_root) for c in cells]
    rows = [r for part in parts for r in part]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def _census_cell(cell, phase1_root):
    """Census rows of one cell (metadata only)."""
    lst, date, el = cell
    rows = []
    runs = latest_runs(os.path.join(phase1_root, lst, date))
    ex = excluded(lst, date, runs)
    for sb, sd in sorted(runs.items(), key=lambda kv: int(kv[0][:-3])):
        pv, era, psfs = list_pilots(sd)
        try:
            src, fl, rms, _ = flag_record(sd, sb, pv)
        except Exception as e:
            src, fl = "error: %s" % e, None
        vb = ib = 0
        if fl:
            for i in fl:
                vb += os.path.getsize(pv[i])
                ip = jfq.i_path_for(pv[i])
                if ip:
                    ib += os.path.getsize(ip)
        rows.append(dict(lst=lst, date=date, jup_el_mid=el, subband=sb, excluded=int(sb in ex),
                         run=os.path.basename(os.path.dirname(sd)), era=era,
                         n_frames=len(pv), n_psf=len(psfs), record_source=src,
                         n_flagged=("" if fl is None else len(fl)), flagged_v_bytes=vb, flagged_i_bytes=ib,
                         frame_v_bytes_median=(int(np.median([os.path.getsize(p) for p in pv[::40]])) if pv else 0)))
    return rows


# ----------------------------------------------------------------------------------------------------------------------------
#  content test
# ----------------------------------------------------------------------------------------------------------------------------
def verify_tree(dest_root, phase1_root=PHASE1_ROOT, remeasure=True, nproc=8):
    """CONTENT TEST of a destination tree, per subband directory holding a record:
      (a) every file in MANIFEST.sha256 hashes to its listed sha256, AND its source_path in manifest.csv (the archive) hashes to
          the same value (the copy is the archive's file);
      (b) every record row with reason 'jupiter' has its pilot V copy (and its pilot I copy when the record names one), and no
          copied pilot image lacks a 'jupiter' row (no extra or missing copies);
      (c) re-measured from the COPY (not the record): Jupiter's |V| within 1 beam of its ephemeris position is >= DETECT_SNR x
          the robust noise for every 'jupiter' row, and the copy's QA-box statistic reproduces the row's recomputed statistic
          (it is that integration's own image);
      (d) the record's Jupiter-removed statistic is <= its threshold for every 'jupiter' row.
    Returns (n_dirs, failures list)."""
    fails = []
    dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(dest_root, "*", "*", "*", "*", jfq.RECORD_NAME.format(subband="*")))})
    jobs = []
    for d in dirs:
        rec = glob.glob(os.path.join(d, jfq.RECORD_NAME.format(subband="*")))[0]
        rows = jfq.read_record(rec)
        man = {m["file"]: m for m in jfq.read_record(os.path.join(d, "manifest.csv"))} if os.path.exists(os.path.join(d, "manifest.csv")) else {}
        sums = {}
        if os.path.exists(os.path.join(d, "MANIFEST.sha256")):
            for line in open(os.path.join(d, "MANIFEST.sha256")):
                h, n = line.strip().split("  ", 1)
                sums[n] = h
        if set(sums) != set(man):
            fails.append((d, "MANIFEST.sha256 and manifest.csv list different files"))
        for n, h in sums.items():
            p = os.path.join(d, n)
            if not os.path.exists(p):
                fails.append((d, "listed copy missing: %s" % n))
                continue
            if jfq.sha256(p) != h:
                fails.append((d, "copy sha256 differs from manifest: %s" % n))
            src = man.get(n, {}).get("source_path", "")
            if not src or not os.path.exists(src) or jfq.sha256(src) != h:
                fails.append((d, "archive source missing or sha256 differs: %s" % n))
        jup = [r for r in rows if r["reason"] == "jupiter"]
        want = set()
        for r in jup:
            for key in ("v_file", "i_file"):
                if r.get(key):
                    nm = [n for n in sums if n.split(".fits")[0] == r[key].split(".fits")[0]]
                    if not nm:
                        fails.append((d, "jupiter row %s: no copy of %s" % (r["snapshot_index"], r[key])))
                    want.update(nm)
            try:
                if float(r["s_jupiter_removed"]) > float(r["qa_high"]):
                    fails.append((d, "jupiter row %s: Jupiter-removed statistic above threshold" % r["snapshot_index"]))
            except ValueError:
                fails.append((d, "jupiter row %s: missing statistic" % r["snapshot_index"]))
            vcopy = [n for n in sums if n.split(".fits")[0] == r["v_file"].split(".fits")[0]]
            if remeasure and vcopy:
                ic = [n for n in sums if r.get("i_file") and n.split(".fits")[0] == r["i_file"].split(".fits")[0]]
                psf = [n for n in sums if man[n]["role"] == "psf_first_interval"]
                jobs.append((d, r, os.path.join(d, vcopy[0]), os.path.join(d, ic[0]) if ic else None,
                             os.path.join(d, psf[0]) if psf else None))
        extra = [n for n in sums if man.get(n, {}).get("role", "").startswith("pilot_") and n not in want]
        if extra:
            fails.append((d, "copies without a 'jupiter' row: %s" % extra[:5]))
    if jobs:
        with Pool(nproc) as p:
            res = p.map(_verify_job, jobs, chunksize=1)
        for d, idx, msg in res:
            if msg:
                fails.append((d, "row %s: %s" % (idx, msg)))
    return len(dirs), fails


def _verify_job(job):
    d, r, vp, ip, pp = job
    try:
        m = jfq.measure_integration(vp, _location(), psf=_psf(pp) if pp else None, i_path=ip)
        msgs = []
        if not (m.get("status") == "measured" and m["snr_v"] >= jfq.DETECT_SNR):
            msgs.append("copy shows Jupiter at S/N %.1f < %.0f" % (m.get("snr_v", np.nan), jfq.DETECT_SNR))
        want = float(r["qa_rms_recomputed"])
        if abs(m["qa_rms_recomputed"] - want) > 1e-6 * max(abs(want), 1e-30) + 1e-9:
            msgs.append("copy's QA statistic %.9g != record %.9g (not this integration's image)" % (m["qa_rms_recomputed"], want))
        if m.get("date_obs") != r.get("date_obs"):
            msgs.append("copy DATE-OBS %s != record %s" % (m.get("date_obs"), r.get("date_obs")))
        return d, r["snapshot_index"], "; ".join(msgs)
    except Exception as e:
        return d, r["snapshot_index"], "re-measure failed: %s" % e


# ----------------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["window", "census", "run", "verify"])
    ap.add_argument("--phase1-root", default=PHASE1_ROOT)
    ap.add_argument("--dest-root", default=jfq.JOVIAN_ROOT)
    ap.add_argument("--cells", help="comma list LST_DATE (e.g. 04h_2024-12-24); default: the Jupiter window")
    ap.add_argument("--subbands", help="comma list (default: all)")
    ap.add_argument("--min-el", type=float, default=30.0)
    ap.add_argument("--window-csv", help="read/write the window cell list here")
    ap.add_argument("--out", help="census / run summary CSV")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--no-copy", action="store_true")
    a = ap.parse_args()
    if a.mode == "verify":
        n, fails = verify_tree(a.dest_root, a.phase1_root, nproc=a.nproc)
        for d, msg in fails:
            print("FAIL %s: %s" % (d, msg))
        print("VERIFY %s: %d subband directories, %d failures" % ("OK" if not fails else "FAILED", n, len(fails)))
        sys.exit(1 if fails else 0)
    if a.cells:
        cells = [(c.split("_")[0], c.split("_")[1], "") for c in a.cells.split(",")]
    elif a.window_csv and os.path.exists(a.window_csv):
        cells = [(r["lst"], r["date"], r["jup_el_mid"]) for r in csv.DictReader(open(a.window_csv))]
    else:
        cells = jupiter_window(a.phase1_root, a.min_el)
        if a.window_csv:
            with open(a.window_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["lst", "date", "jup_el_mid"])
                w.writerows(cells)
    if a.mode == "window":
        print("%d cells with Jupiter >= %.0f deg at mid-hour" % (len(cells), a.min_el))
        return
    if a.mode == "census":
        rows = census(cells, a.phase1_root, a.out, nproc=a.nproc)
        print("census: %d subband-nights -> %s" % (len(rows), a.out))
        return
    subs = set(a.subbands.split(",")) if a.subbands else None
    summ = []
    with Pool(a.nproc) as pool:
        for lst, date, el in cells:
            runs = latest_runs(os.path.join(a.phase1_root, lst, date))
            ex = excluded(lst, date, runs)
            for sb, sd in sorted(runs.items(), key=lambda kv: int(kv[0][:-3])):
                if subs and sb not in subs:
                    continue
                if sb in ex:
                    o = dict(lst=lst, date=date, subband=sb, run=os.path.basename(os.path.dirname(sd)),
                             error="excluded (data_decisions.csv): not classified, not copied", jup_el_mid=el)
                    summ.append(o)
                    print(json.dumps(o), flush=True)
                    continue
                try:
                    o, _ = process_subband(pool, lst, date, sd, a.dest_root, copy=not a.no_copy)
                except Exception as e:
                    o = dict(lst=lst, date=date, subband=sb, error="%s: %s" % (type(e).__name__, e))
                o["jup_el_mid"] = el
                summ.append(o)
                print(json.dumps(o), flush=True)
    if a.out and summ:
        keys = []
        for o in summ:
            keys += [k for k in o if k not in keys]
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(summ)


if __name__ == "__main__":
    main()
