#!/usr/bin/env python3
"""jupiter_flag_qa.py - Phase-1 Stokes V noise flag: is a flagged integration's excess caused by Jupiter?  Record + copy.

RULING (Gregg 2026-10-07, on question 12, verbatim): "12 and 13 approved; However, the Stokes V statistic is used to remove
data that will compromise the deep Stokes V imaging and a MJy Jovian burst would do that. Better to have a pipeline that
identifies if an excess is caused by Jupiter and copying the quarantined I/V files to a different dorectory for further
Jovian processing."

THE FLAG IS NOT CHANGED.  pipeline_utils.analyze_snapshot_quality decides and flag_bad_integrations applies the flag exactly as
before (Phase 1 calls this module only after the flag is applied).  This module reads the pilot images and the QA statistics
the flag used, writes one record per flagged integration (QA/jupiter_flag_qa_<subband>.csv, next to the other QA products),
and, after the run is archived, COPIES (never moves) the archived pilot images of the Jupiter-caused integrations to
JOVIAN_ROOT/<LST>/<date>/<Run>/<subband>/ with a sha256 manifest and a provenance README.  Opt-out: EXOPIPE_JUPITER_FLAG_QA=0.

THE TEST (TEST_VERSION below), per flagged integration; the pilot is the dirty WSClean image of that one integration:
  0. Flag side: an integration flagged LOW (statistic < 0.5 x the night's median) -> other / low_statistic (Jupiter can only
     raise the statistic).
  1. Jupiter's apparent direction at the integration midpoint (DATE-OBS + HALF_INT_S): get_body('jupiter', t, location) ->
     SkyCoord(jb.ra, jb.dec).  Never separation(get_body(...)): that reprojects the body to barycentric ICRS and moves it by
     up to ~2.7 deg near opposition.  Elevation from AltAz; below the horizon or off the image -> other.
  2. Detection: peak |V| within SEARCH_BEAMS x BMAJ of that pixel (signed value kept) over the robust noise sigma_V =
     1.4826 x MAD of the QA box (the central 1024 x 1024 pixels the flag statistic uses) outside MASK_BEAMS x BMAJ of Jupiter;
     detected iff |peak| / sigma_V >= DETECT_SNR.
  3. Jupiter-removed flag statistic: the flag statistic (std of the QA box) recomputed with Jupiter's pixels within
     MASK_BEAMS x BMAJ removed (s_mask), then Jupiter's PSF response in the rest of the box removed in quadrature:
     s_rem = sqrt(max(s_mask^2 - (kappa x peak)^2, 0)), kappa = std over the same masked box of the run's FIRST-interval pilot
     PSF (peak 1) centred on Jupiter's peak pixel.  (Jupiter lies outside or at the edge of the box in most cells, so its
     sidelobes, not its main lobe, are what it adds to the box; the first-interval PSF is the one PSF the archive keeps, so the
     archive backfill reproduces this exactly.  Jovian bursts are narrow-band, so their sidelobes can exceed the wide-band PSF's:
     kappa then underestimates and the test errs towards 'other'.)
  4. reason 'jupiter' iff detected AND s_rem <= the night's high threshold (median + 3 x MAD-sigma of the statistic over the
     night's integrations: the flag's own threshold); else 'other' (detail 'jupiter_detected_excess_remains' when Jupiter is
     detected but something else keeps the statistic above the threshold).
The pilot I image (when the run made one) is measured at the same position and recorded; it does not enter the decision.
"""
import csv
import glob
import hashlib
import io
import logging
import os
import re
import shutil
import subprocess
import time
import traceback

import numpy as np

TEST_VERSION = "JFQ-1 2026-10-07"
QA_BOX_HALF = 512            # analyze_snapshot_quality: central 1024 x 1024 pixels
SEARCH_BEAMS = 1.0           # peak search radius around the ephemeris pixel, in BMAJ (beam FWHM, major axis)
MASK_BEAMS = 3.0             # Jupiter mask radius for the noise and the Jupiter-removed statistic, in BMAJ
DETECT_SNR = 10.0            # |peak V| / sigma_V for 'Jupiter detected'
HALF_INT_S = 5.0             # integration midpoint = DATE-OBS (WSClean: interval start) + 5 s (10-s integrations)
JOVIAN_ROOT = "/lustre/gh/main/phase3/jupiter/phase1_flagged"
COPY_REASONS = ("jupiter",)
RECORD_NAME = "jupiter_flag_qa_{subband}.csv"
FUNPACK_CANDIDATES = ("/opt/devel/pipeline/envs/development/bin/funpack",)
RULING = ("Gregg 2026-10-07 (question 12): \"12 and 13 approved; However, the Stokes V statistic is used to remove data that "
          "will compromise the deep Stokes V imaging and a MJy Jovian burst would do that. Better to have a pipeline that "
          "identifies if an excess is caused by Jupiter and copying the quarantined I/V files to a different dorectory for "
          "further Jovian processing.\"")

FIELDS = ["test_version", "lst", "date", "run", "subband", "snapshot_index", "scan_number", "timestamp", "date_obs", "t_mid_utc",
          "flag_side", "qa_rms", "qa_rms_recomputed", "qa_median", "qa_high", "qa_low",
          "jup_ra_deg", "jup_dec_deg", "jup_el_deg", "jup_x", "jup_y", "jup_in_box", "bmaj_arcmin",
          "peak_v_jy", "peak_offset_arcmin", "sigma_v_jy", "snr_v", "s_masked", "kappa", "s_jupiter_removed",
          "peak_i_jy", "sigma_i_jy", "snr_i", "reason", "detail", "v_file", "i_file", "psf_file"]


def _log(logger):
    return logger if logger is not None else logging.getLogger("jupiter_flag_qa")


# ----------------------------------------------------------------------------------------------------------------------------
#  reading
# ----------------------------------------------------------------------------------------------------------------------------
def _funpack():
    for p in (os.environ.get("JFQ_FUNPACK"),) + FUNPACK_CANDIDATES:
        if p and os.path.isfile(p):
            return p
    return shutil.which("funpack")


def load_image(path):
    """(2-D float32 array, header) of a pilot image; tile-compressed archive files (.fz/.fs) via funpack -S (astropy < 6
    cannot decompress the 4-axis WSClean cubes)."""
    from astropy.io import fits
    if path.endswith((".fz", ".fs")):
        fun = _funpack()
        if not fun:
            raise RuntimeError("funpack not found for %s" % path)
        raw = subprocess.run([fun, "-S", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
        hdul = fits.open(io.BytesIO(raw))
    else:
        hdul = fits.open(path, memmap=False)
    with hdul:
        hdu = hdul[0]
        return np.squeeze(hdu.data), hdu.header.copy()


# ----------------------------------------------------------------------------------------------------------------------------
#  thresholds (exactly analyze_snapshot_quality's)
# ----------------------------------------------------------------------------------------------------------------------------
def flag_thresholds(rms_values):
    """(median, high, low) of the flag statistic exactly as pipeline_utils.analyze_snapshot_quality (and the Orca copy)
    computes them; pass the values in the type the pipeline used (np.float32 for exopipe, Python float for Orca)."""
    from astropy.stats import mad_std
    rmses = np.array(list(rms_values))
    med_rms = np.nanmedian(rmses)
    std_rms = mad_std(rmses, ignore_nan=True)
    return med_rms, med_rms + (3.0 * std_rms), 0.5 * med_rms


def flagged_from_thresholds(stats, high, low):
    return sorted(s["idx"] for s in stats if s["rms"] > high or s["rms"] < low)


# ----------------------------------------------------------------------------------------------------------------------------
#  measurement of one integration
# ----------------------------------------------------------------------------------------------------------------------------
def _box(shape):
    ny, nx = shape
    ch, cw = ny // 2, nx // 2
    return slice(ch - QA_BOX_HALF, ch + QA_BOX_HALF), slice(cw - QA_BOX_HALF, cw + QA_BOX_HALF)


def _mad_sigma(a):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    return 1.4826 * float(np.median(np.abs(a - np.median(a))))


def jupiter_direction(t, location):
    """Apparent (topocentric) direction of Jupiter as a plain angular SkyCoord, and its elevation (deg)."""
    from astropy.coordinates import AltAz, SkyCoord, get_body
    jb = get_body("jupiter", t, location=location)
    el = float(jb.transform_to(AltAz(obstime=t, location=location)).alt.deg)
    return SkyCoord(jb.ra, jb.dec), el


def measure_integration(v_path, location, psf=None, i_path=None, half_int_s=HALF_INT_S):
    """Measurements of one pilot integration (no decision).  psf = (2-D array normalised to peak 1, (peak_y, peak_x)) or None."""
    from astropy.time import Time
    from astropy.wcs import WCS
    import astropy.units as u
    V, hd = load_image(v_path)
    ny, nx = V.shape
    bs = _box(V.shape)
    out = dict(date_obs=str(hd.get("DATE-OBS", "")), v_file=os.path.basename(v_path),
               i_file=os.path.basename(i_path) if i_path else "")
    out["qa_rms_recomputed"] = float(np.nanstd(V[bs]))
    t = Time(hd["DATE-OBS"], scale="utc") + half_int_s * u.s
    out["t_mid_utc"] = t.isot
    jc, el = jupiter_direction(t, location)
    out.update(jup_ra_deg=float(jc.ra.deg), jup_dec_deg=float(jc.dec.deg), jup_el_deg=el)
    w = WCS(hd).celestial
    x, y = [float(v) for v in w.world_to_pixel(jc)]
    pix = abs(float(hd["CDELT2"]))
    bmaj = float(hd["BMAJ"]) / pix
    out.update(jup_x=x, jup_y=y, bmaj_arcmin=float(hd["BMAJ"]) * 60.0,
               jup_in_box=int(bs[1].start <= x < bs[1].stop and bs[0].start <= y < bs[0].stop))
    if el <= 0.0:
        out["status"] = "jupiter_below_horizon"
        return out
    if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < nx and 0 <= y < ny) or not np.isfinite(V[int(round(y)), int(round(x))]):
        out["status"] = "jupiter_not_in_image"
        return out
    # peak |V| within SEARCH_BEAMS x BMAJ of the ephemeris pixel
    r = SEARCH_BEAMS * bmaj
    y0, y1 = max(0, int(np.floor(y - r))), min(ny, int(np.ceil(y + r)) + 1)
    x0, x1 = max(0, int(np.floor(x - r))), min(nx, int(np.ceil(x + r)) + 1)
    sub = V[y0:y1, x0:x1].astype(np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    m = (np.hypot(xx - x, yy - y) <= r) & np.isfinite(sub)
    k = np.argmax(np.where(m, np.abs(sub), -1.0))
    py, px = int(yy.ravel()[k]), int(xx.ravel()[k])
    peak = float(V[py, px])
    out.update(peak_v_jy=peak, peak_offset_arcmin=float(np.hypot(px - x, py - y) * pix * 60.0))
    # Jupiter mask over the QA box
    by, bx = np.mgrid[bs[0], bs[1]]
    keep = np.hypot(bx - x, by - y) > MASK_BEAMS * bmaj
    vb = V[bs].astype(np.float32)
    sigma = _mad_sigma(vb[keep])
    out.update(sigma_v_jy=sigma, snr_v=abs(peak) / sigma if sigma > 0 else np.nan)
    vm = np.where(keep, vb, np.nan)
    s_mask = float(np.nanstd(vm))
    out["s_masked"] = s_mask
    kappa = np.nan
    if psf is not None:
        P, (pcy, pcx) = psf
        # PSF value at box pixel (Y, X) for a source at (py, px): P[Y - py + pcy, X - px + pcx]
        Y = by - py + pcy
        X = bx - px + pcx
        inside = (Y >= 0) & (Y < P.shape[0]) & (X >= 0) & (X < P.shape[1]) & keep
        if inside.any():
            vals = P[Y[inside], X[inside]]
            kappa = float(np.nanstd(vals))
    out["kappa"] = kappa
    pred = (kappa * abs(peak)) if np.isfinite(kappa) else 0.0
    out["s_jupiter_removed"] = float(np.sqrt(max(s_mask ** 2 - pred ** 2, 0.0)))
    out["status"] = "measured"
    if i_path and os.path.exists(i_path):
        I, hi = load_image(i_path)
        si = I[y0:y1, x0:x1].astype(np.float64)
        mi = (np.hypot(xx - x, yy - y) <= r) & np.isfinite(si)
        ki = np.argmax(np.where(mi, si, -np.inf))
        pi_ = float(si.ravel()[ki])
        sig_i = _mad_sigma(I[bs].astype(np.float32)[keep])
        out.update(peak_i_jy=pi_, sigma_i_jy=sig_i, snr_i=pi_ / sig_i if sig_i > 0 else np.nan)
    return out


def classify(meas, qa_rms, high, low):
    """(reason, detail) for one flagged integration."""
    if qa_rms < low:
        return "other", "low_statistic"
    st = meas.get("status")
    if st != "measured":
        return "other", st or "not_measured"
    snr = meas.get("snr_v")
    if not (snr is not None and np.isfinite(snr) and snr >= DETECT_SNR):
        return "other", "jupiter_not_detected"
    if meas["s_jupiter_removed"] <= high:
        return "jupiter", "jupiter_explains_excess"
    return "other", "jupiter_detected_excess_remains"


def load_psf(psf_path):
    if not psf_path or not os.path.exists(psf_path):
        return None
    P, _ = load_image(psf_path)
    P = P.astype(np.float64)
    pk = np.unravel_index(np.nanargmax(P), P.shape)
    P = P / P[pk]
    return P, (int(pk[0]), int(pk[1]))


def _ts_of(name):
    m = re.search(r"(\d{8}_\d{6})", os.path.basename(name))
    return m.group(1) if m else ""


def i_path_for(v_path):
    """Pilot I image of the same integration when the run made one (-pol IV: ...-V-image -> ...-I-image)."""
    if "-V-image" in os.path.basename(v_path):
        p = os.path.join(os.path.dirname(v_path), os.path.basename(v_path).replace("-V-image", "-I-image"))
        return p if os.path.exists(p) else None
    return None


def first_psf(snap_dir):
    c = sorted(glob.glob(os.path.join(snap_dir, "*Pilot-Snapshot-*-psf.fits*")))
    return c[0] if c else None


def classify_flagged(pilot_v, stats, bad_idx, location, psf_path=None, scans=None, cell=None, nproc=1,
                     thresholds=None, rms_tolerance=0.0, logger=None):
    """Rows (FIELDS) for every flagged integration.  pilot_v: the flag's own sorted pilot V list (index = snapshot index);
    stats: the flag's per-integration statistics [{'idx', 'rms'}]; bad_idx: the flag's decision.  Refuses (raises) if the
    thresholds recomputed from stats do not reproduce bad_idx exactly - the record must describe the flag that was applied."""
    log = _log(logger)
    med, high, low = thresholds if thresholds is not None else flag_thresholds([s["rms"] for s in stats])
    repro = flagged_from_thresholds(stats, high, low)
    if sorted(bad_idx) != repro:
        raise RuntimeError("flag thresholds do not reproduce the flagged set (%d recomputed vs %d applied)"
                           % (len(repro), len(bad_idx)))
    rms_of = {s["idx"]: s["rms"] for s in stats}
    cell = cell or {}
    psf = load_psf(psf_path)
    jobs = [(i, pilot_v[i], i_path_for(pilot_v[i])) for i in sorted(bad_idx)]

    def _one(job):
        i, vp, ip = job
        try:
            return i, measure_integration(vp, location, psf=psf, i_path=ip), None
        except Exception as e:  # one unreadable frame is recorded, never silently dropped
            return i, {"status": "error", "v_file": os.path.basename(vp)}, "%s: %s" % (type(e).__name__, e)

    if nproc > 1:
        from multiprocessing.pool import ThreadPool
        with ThreadPool(nproc) as tp:
            res = tp.map(_one, jobs)
    else:
        res = [_one(j) for j in jobs]
    rows = []
    for i, meas, err in res:
        qa = rms_of[i]
        side = "low" if qa < low else "high"
        if err:
            reason, detail = "other", "error: " + err
        else:
            rec = meas.get("qa_rms_recomputed")
            if rec is not None and np.isfinite(rec) and abs(rec - float(qa)) > rms_tolerance * max(abs(float(qa)), 1e-30):
                reason, detail = "other", "file_mismatch: recomputed statistic %.6g vs flag %.6g" % (rec, float(qa))
            else:
                reason, detail = classify(meas, float(qa), float(high), float(low))
        row = {k: "" for k in FIELDS}
        row.update({k: v for k, v in meas.items() if k in FIELDS})
        row.update(test_version=TEST_VERSION, lst=cell.get("lst", ""), date=cell.get("date", ""), run=cell.get("run", ""),
                   subband=cell.get("subband", ""), snapshot_index=i,
                   scan_number=(scans[i] if scans is not None and 0 <= i < len(scans) else ""),
                   timestamp=_ts_of(pilot_v[i]), flag_side=side, qa_rms=float(qa), qa_median=float(med),
                   qa_high=float(high), qa_low=float(low), reason=reason, detail=detail,
                   psf_file=os.path.basename(psf_path) if psf_path else "")
        rows.append(row)
    n_j = sum(r["reason"] == "jupiter" for r in rows)
    log.info("Jupiter check of flagged integrations: %d flagged, %d jupiter, %d other (%s)" % (len(rows), n_j, len(rows) - n_j,
                                                                                            TEST_VERSION))
    return rows


def write_record(rows, path):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.10g" % r[k]) if isinstance(r.get(k), float) else r.get(k, "") for k in FIELDS})
    os.replace(tmp, path)
    return path


def read_record(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------------------------------------------------------
#  Phase-1 hook 1: at QA time (after the flag is applied)
# ----------------------------------------------------------------------------------------------------------------------------
def record_flagged(pilot_v, stats, bad_idx, qa_dir, subband, location, snap_dir=None, scans=None, cell=None, logger=None):
    """Write QA/jupiter_flag_qa_<subband>.csv for the flagged integrations (nothing if none).  Never raises: on failure writes
    QA/jupiter_flag_qa_<subband>.FAILED.txt with the traceback (the archive backfill redoes such runs) and returns None."""
    log = _log(logger)
    path = os.path.join(qa_dir, RECORD_NAME.format(subband=subband))
    try:
        if os.environ.get("EXOPIPE_JUPITER_FLAG_QA", "1") == "0":
            log.info("Jupiter check of flagged integrations: disabled (EXOPIPE_JUPITER_FLAG_QA=0)")
            return None
        stats = [dict(s) for s in stats]          # never touch the flag's own objects
        bad = list(bad_idx)
        snap_dir = snap_dir or (os.path.dirname(pilot_v[0]) if pilot_v else None)
        psf_path = first_psf(snap_dir) if snap_dir else None
        rows = classify_flagged(list(pilot_v), stats, bad, location, psf_path=psf_path, scans=scans,
                                cell=dict(cell or {}, subband=subband), logger=log)
        os.makedirs(qa_dir, exist_ok=True)
        write_record(rows, path)
        log.info("Jupiter check record: %s" % path)
        return path
    except Exception:
        try:
            os.makedirs(qa_dir, exist_ok=True)
            with open(os.path.join(qa_dir, RECORD_NAME.format(subband=subband).replace(".csv", ".FAILED.txt")), "w") as f:
                f.write("%s\n%s\n" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), traceback.format_exc()))
        except Exception:
            pass
        log.error("Jupiter check of flagged integrations FAILED (flag unaffected): %s" % traceback.format_exc())
        return None


# ----------------------------------------------------------------------------------------------------------------------------
#  copying (Phase-1 hook 2 after archiving, and the archive backfill)
# ----------------------------------------------------------------------------------------------------------------------------
def sha256(path, bufsize=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _archived(snap_dir, name):
    """The archived file of a pilot image named on NVMe (the archive may have fpack-compressed it: .fz / .fs)."""
    if not name:
        return None
    for suf in ("", ".fz", ".fs"):
        p = os.path.join(snap_dir, name + suf)
        if os.path.exists(p):
            return p
    p = os.path.join(snap_dir, name)
    return p if os.path.exists(p) else None


def copy_verified(src, dest_dir):
    """Copy src into dest_dir: sha256 of the source, copy to a temporary name, sha256 of the copy must equal, rename.  An
    existing file of the same name is kept if identical ('present') and never overwritten if different (raises)."""
    name = os.path.basename(src)
    dst = os.path.join(dest_dir, name)
    h_src = sha256(src)
    if os.path.exists(dst):
        h_dst = sha256(dst)
        if h_dst != h_src:
            raise RuntimeError("REFUSED: %s exists with a different sha256 (%s vs source %s)" % (dst, h_dst, h_src))
        return dst, h_src, "present"
    tmp = os.path.join(dest_dir, ".tmp_" + name)
    shutil.copyfile(src, tmp)
    h_tmp = sha256(tmp)
    if h_tmp != h_src:
        os.replace(tmp, os.path.join(dest_dir, ".failed_" + name))      # kept for inspection, never used
        raise RuntimeError("copy of %s failed verification (sha256 %s vs %s)" % (src, h_tmp, h_src))
    os.replace(tmp, dst)
    return dst, h_src, "copied"


MANIFEST_FIELDS = ["file", "sha256", "bytes", "role", "snapshot_index", "timestamp", "reason", "snr_v", "peak_v_jy",
                   "source_path"]


def _git_version():
    try:
        d = os.path.dirname(os.path.abspath(__file__))
        return subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True, timeout=20).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def copy_jupiter_rows(rows, snap_dir, dest_dir, record_path=None, source_run_dir=None, copy_reasons=COPY_REASONS,
                      written_by="Phase 1", logger=None):
    """Put the record of every flagged integration in dest_dir (the Jovian tree holds the record of each subband-night with a
    flagged integration, copies or not); copy the archived pilot V and I images of the rows whose reason is in copy_reasons and
    the run's first-interval pilot PSF; write MANIFEST.sha256 (sha256sum -c format), manifest.csv and README.txt.  Returns a
    summary."""
    log = _log(logger)
    sel = [r for r in rows if r.get("reason") in copy_reasons]
    summ = dict(dest=dest_dir, n_rows=len(rows), n_selected=len(sel), copied=0, present=0, missing=[], bytes=0)
    if not rows:
        return summ
    os.makedirs(dest_dir, exist_ok=True)
    man_rows = {}
    mp = os.path.join(dest_dir, "manifest.csv")
    if os.path.exists(mp):
        for r in read_record(mp):
            man_rows[r["file"]] = r
    todo = []
    for r in sel:
        for role, key in (("pilot_V", "v_file"), ("pilot_I", "i_file")):
            if r.get(key):
                todo.append((role, r, _archived(snap_dir, r[key]), r[key]))
    psf_name = next((r.get("psf_file") for r in sel if r.get("psf_file")), "")
    if psf_name:
        todo.append(("psf_first_interval", {}, _archived(snap_dir, psf_name), psf_name))
    for role, r, src, name in todo:
        if src is None:
            summ["missing"].append(name)
            continue
        dst, h, how = copy_verified(src, dest_dir)
        summ["copied" if how == "copied" else "present"] += 1
        summ["bytes"] += os.path.getsize(dst)
        man_rows[os.path.basename(dst)] = dict(file=os.path.basename(dst), sha256=h, bytes=os.path.getsize(dst), role=role,
                                               snapshot_index=r.get("snapshot_index", ""), timestamp=r.get("timestamp", ""),
                                               reason=r.get("reason", ""), snr_v=r.get("snr_v", ""),
                                               peak_v_jy=r.get("peak_v_jy", ""), source_path=os.path.abspath(src))
    if record_path and os.path.exists(record_path):
        rec_dst = os.path.join(dest_dir, os.path.basename(record_path))
        tmp = os.path.join(dest_dir, ".tmp_" + os.path.basename(record_path))
        shutil.copyfile(record_path, tmp)
        os.replace(tmp, rec_dst)
    if not man_rows:
        _write_readme(dest_dir, rows, sel, snap_dir, source_run_dir, record_path, copy_reasons, written_by)
        log.info("Jupiter-flagged record -> %s: %d flagged, none attributed to Jupiter (nothing copied)" % (dest_dir, len(rows)))
        return summ
    # manifests (rewritten whole, atomically)
    ordered = sorted(man_rows.values(), key=lambda m: m["file"])
    tmp = mp + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(ordered)
    os.replace(tmp, mp)
    ms = os.path.join(dest_dir, "MANIFEST.sha256")
    with open(ms + ".tmp", "w") as f:
        for m in ordered:
            f.write("%s  %s\n" % (m["sha256"], m["file"]))
    os.replace(ms + ".tmp", ms)
    _write_readme(dest_dir, rows, sel, snap_dir, source_run_dir, record_path, copy_reasons, written_by)
    log.info("Jupiter-flagged copies -> %s: %d copied, %d already present, %d missing, %.1f MB"
             % (dest_dir, summ["copied"], summ["present"], len(summ["missing"]), summ["bytes"] / 1e6))
    return summ


def _write_readme(dest_dir, rows, sel, snap_dir, source_run_dir, record_path, copy_reasons, written_by):
    readme = os.path.join(dest_dir, "README.txt")
    with open(readme + ".tmp", "w") as f:
        f.write(
            "Phase-1 Stokes V noise-FLAGGED integrations of one subband-night: the record of every flagged integration and copies of\n"
            "those whose excess was attributed to Jupiter, for Jovian processing (%d flagged, %d attributed to Jupiter)\n\n"
            % (len(rows), len(sel)))
        f.write(
            "Ruling: %s\n\n"
            "What these files are: verbatim copies (sha256 in MANIFEST.sha256, `sha256sum -c MANIFEST.sha256`) of archived Phase-1 PILOT\n"
            "snapshot images - DIRTY (no CLEAN), one 10-s integration each, made before the per-integration Stokes V noise flag; the\n"
            "flag removed these integrations from the measurement set, so every later image of them (CLEAN snapshots, deep, 10-min)\n"
            "is empty or lacks them.  They are NOT flow-registered to the Master and NOT primary-beam corrected; names and headers are\n"
            "the archive's.  Originals stay in the archive (copy, never move).  pilot_I exists only for runs that imaged I in the pilot\n"
            "(-pol IV); psf_first_interval is the run's one archived pilot PSF (first integration).\n\n"
            "Source run: %s\nTest: %s (see %s and jupiter_flag_qa.py docstring): detection |V| >= %.0f sigma within %.1f beam of\n"
            "Jupiter's apparent position (astropy get_body, SkyCoord(jb.ra, jb.dec)); excess attributed to Jupiter when the flag\n"
            "statistic with Jupiter masked (%.0f beams) and its PSF sidelobes removed in quadrature passes the night's threshold.\n"
            "Copied reasons: %s.  Written by: %s, exopipe %s, %s UTC.\n"
            % (RULING, source_run_dir or snap_dir, TEST_VERSION,
               RECORD_NAME.format(subband="<subband>"), DETECT_SNR, SEARCH_BEAMS, MASK_BEAMS,
               ",".join(copy_reasons), written_by, _git_version(), time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())))
    os.replace(readme + ".tmp", readme)


def dest_dir_for(lst, date, run, subband, root=JOVIAN_ROOT):
    return os.path.join(root, lst, date, run, subband)


def copy_from_archive(archive_base, lst, date, run, subband, root=None, logger=None):
    """Phase-1 hook 2 (after the run is archived): copy the Jupiter-caused flagged integrations recorded in
    <archive_base>/QA/jupiter_flag_qa_<subband>.csv.  Never raises (logs; a failure leaves the copy to the archive backfill)."""
    log = _log(logger)
    try:
        if os.environ.get("EXOPIPE_JUPITER_FLAG_QA", "1") == "0":
            return None
        rec = os.path.join(archive_base, "QA", RECORD_NAME.format(subband=subband))
        if not os.path.exists(rec):
            log.info("Jupiter-flagged copies: no record %s (no flagged integration, or the check did not run)" % rec)
            return None
        rows = read_record(rec)
        return copy_jupiter_rows(rows, os.path.join(archive_base, "snapshots"),
                                 dest_dir_for(lst, date, run, subband, root or JOVIAN_ROOT), record_path=rec,
                                 source_run_dir=archive_base, written_by="Phase 1 (in-pipeline)", logger=log)
    except Exception:
        log.error("Jupiter-flagged copies FAILED (flag and archive unaffected): %s" % traceback.format_exc())
        return None
