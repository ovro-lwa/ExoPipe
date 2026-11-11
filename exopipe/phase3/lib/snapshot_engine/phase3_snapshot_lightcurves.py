#!/usr/bin/env python
"""Phase 3 snapshot-tier LIGHT-CURVE PRODUCT (preamble C, Gregg 2026-09-17 approved; six-cell test).

For one cell and subband, one pass over the staged RAW frames of each Stokes: every row of the source list is
sampled in every frame at its FLOW-CORRECTED raw pixel (raw = flowed + flow[flowed // 8], the fleet-shipped
per-frame flow field, nearest valid instant when the frame's own instant has none - the flow stage's rule).

Rows (no threshold - a threshold column is free to apply later, a missing row is not):
  master   every PyBDSF single-Gaussian (S_Code 'S') source of this hour's Master catalogue for this subband
  chain    every served source of a closed chain run (repeat, cross-cell, singleton), at its served position
  target   every position of the target samples (Hot/Warm Jupiters, 10 pc, 25 pc, LPTs, flare/UCD, Sydney
           stars, ATNF pulsars) that falls inside the frame
Per row and frame: raw flux (Jy/beam, bilinear) and local rms (1.4826 x MAD of a 41x41 stamp at the row's
flowed pixel, central 5x5 excluded - the photometry stage's stamp_noise); per row and 10-minute bin: the
MODEL-beam value (Gregg 2026-09-16: "model beam is all that is needed for snapshot"); per frame: admission
flag from the search's noise table, bin index, flow instant used.  Flux scale of the cell/subband = median of
(row median flux / Master peak) over bright compact Master sources (peak/rms >= 20), i.e. hourly level over
Master level, stored in the header.  Nothing is PB-corrected or scaled in the stored arrays: corrected flux
= FLUX / PB, corrected noise = RMS / PB, Master-relative = / FLUXSCAL.

Output: <out>/<cell>_<subband>_<Stokes>_lightcurves.fits with HDUs FLUX [rows x frames], RMS, PB [rows x 6],
SOURCES (binary table), FRAMES (binary table).  Refuses if the frames of a subband do not share one WCS.
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import time

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase3_snapshot_search as SS
from phase3_snapshot_search import discover_frames, pb_value, ts_to_epoch_seconds
import phase3_snapshot_refastro as R
import phase3_snapshot_chain as C

STAMP = 41
NBINS = 6
REF_SNR = 20.0
CODE = os.path.dirname(os.path.abspath(__file__))


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", CODE, "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_rows(lst, sb, master_dir, run_dir, samples_dir):
    rows = []
    gp = os.path.join(master_dir, "master_R075_%s_%s.gaul.csv" % (lst, sb))
    if not os.path.exists(gp):
        raise RuntimeError("no Master catalogue %s" % gp)
    for i, g in enumerate(R.read_gaul_full(gp)):
        if not (np.isfinite(g["ra"]) and np.isfinite(g["dec"])):
            continue
        # every Gaussian of the Master catalogue, not only the single-Gaussian ('S') ones: PSR J0218+4232 is a
        # 'C' component at 41-73 MHz and was missing from the six-cell test rows (found 2026-09-17); the code is
        # kept in the kind so a compact-only selection stays a column filter
        # row id = PyBDSF's own Gaus_id, NOT the position in the file: identical to the old enumeration index on all
        # 3,431,076 rows of the 456 catalogues (verified 2026-09-23), and unchanged when the builder drops
        # non-finite-position rows (Gregg 2026-09-23), which would renumber every later row under the index
        gid = g["gid"] if g.get("gid") is not None else i
        rows.append(dict(id="M_%s_%s_%05d" % (lst, sb, gid), kind="master:%s" % (g["code"] or "?"), name=C.jname(g["ra"], g["dec"]),
                         ra=g["ra"], dec=g["dec"], master_peak=g["peak"], master_rms=g["rms"]))
    if run_dir:
        seen = set()
        for p in sorted(glob.glob(os.path.join(run_dir, "source_catalogs", "*_sources_survivors.csv"))):
            for r in csv.DictReader(open(p)):
                if r["source_id"] in seen:
                    continue
                seen.add(r["source_id"])
                rows.append(dict(id="C_%s" % r["source_id"], kind="chain:%s" % r.get("rung", ""), name=r["source_id"],
                                 ra=float(r["ra"]), dec=float(r["dec"]), master_peak=np.nan, master_rms=np.nan))
    if samples_dir:
        for tag, lst_ in C.load_samples(samples_dir).items():
            for ra, dec, nm in lst_:
                rows.append(dict(id="T_%s_%s" % (tag, nm.replace(" ", "_")), kind="target:%s" % tag, name=nm,
                                 ra=ra, dec=dec, master_peak=np.nan, master_rms=np.nan))
    return rows


def extract(cell, sb, stage, proto_root, master_dir, run_dir, samples_dir, out_dir, stokes_list=("I", "V"), log=print,
            search_frames=False):
    lst, date = cell.split("_", 1)
    t_start = time.time()
    rows = load_rows(lst, sb, master_dir, run_dir, samples_dir)
    frames = discover_frames(lst, date, sb, stage_root=stage, flow_select=False)   # never toggle FLOW_MODE (2026-10-04)
    # admission table (per subband; the search's joint noise table)
    adm = {}
    ap = os.path.join(proto_root, cell, "admission_%s.csv" % sb)
    if os.path.exists(ap):
        for r in csv.DictReader(open(ap)):
            adm[r["ts"]] = r.get("admitted", "1") == "1"
    os.makedirs(out_dir, exist_ok=True)
    written = []
    scale_I, nref_I = np.nan, 0
    for st in stokes_list:
        fl = sorted(frames[st])          # discover_frames returns [(ts, path), ...] per Stokes
        # SEARCH FRAMES (2026-11-01, FINDING-SNAPSHOT-LIGHTCURVE-TABLES-HOLD-DUPLICATE-RUN-FRAMES-20261028: "regenerate ... on the
        # search's frame lists"): with search_frames the table holds exactly the frames of the search's own list (the noise table
        # <proto>/<cell>/noise_<sb>_<St>_median.csv) - a frame the search dropped (e.g. 12h 2024-12-28, re-searched on re-flowed
        # frames, dropped the frames that did not flow) is not sampled; REFUSES when a listed frame is not discovered
        if search_frames:
            np_ = os.path.join(proto_root, cell, "noise_%s_%s_median.csv" % (sb, st))
            keep = [r["ts"] for r in csv.DictReader(open(np_))]
            have = {t for t, _ in fl}
            miss = [t for t in keep if t not in have]
            if miss:
                raise RuntimeError("%s %s %s: %d frames of the search's list are not discovered (e.g. %s) - refusing" % (cell, sb, st, len(miss), miss[0]))
            ks = set(keep); n0 = len(fl); fl = [x for x in fl if x[0] in ks]
            log("%s %s %s: the search's frame list - %d of %d discovered frames" % (cell, sb, st, len(fl), n0))
        if not fl:
            log("%s %s %s: no frames" % (cell, sb, st))
            continue
        # SS.read_frame (2026-10-13): a staged funpacked frame or the Phase-1 archive's tile-compressed one (image in HDU 1;
        # funpack fallback under the development env) - the same pixels, so a table can be re-extracted without staging
        _, hdr0 = SS.read_frame(fl[0][1])
        w = WCS(hdr0).celestial
        ny, nx = int(hdr0["NAXIS2"]), int(hdr0["NAXIS1"])
        ras = np.array([r["ra"] for r in rows]); decs = np.array([r["dec"] for r in rows])
        px, py = w.all_world2pix(ras, decs, 0)
        px, py = np.asarray(px, float), np.asarray(py, float)
        m = STAMP // 2 + 2
        infield = np.isfinite(px) & np.isfinite(py) & (px >= m) & (px < nx - m) & (py >= m) & (py < ny - m)
        keep = [r for r, k in zip(rows, infield) if k]
        px, py = px[infield], py[infield]
        n = len(keep)
        if n == 0:
            raise RuntimeError("%s %s %s: no rows inside the frame" % (cell, sb, st))
        # fixed stamp index arrays at the flowed integer pixel (central 5x5 excluded)
        ipx, ipy = np.round(px).astype(int), np.round(py).astype(int)
        dy, dx = np.mgrid[-(STAMP // 2):STAMP // 2 + 1, -(STAMP // 2):STAMP // 2 + 1]
        sel = ~((np.abs(dy) <= 2) & (np.abs(dx) <= 2))
        dy, dx = dy[sel].ravel(), dx[sel].ravel()
        yy = np.clip(ipy[:, None] + dy[None, :], 0, ny - 1)
        xx = np.clip(ipx[:, None] + dx[None, :], 0, nx - 1)
        secs = np.array([ts_to_epoch_seconds(ts) for ts, _ in fl])
        t0, span = secs.min(), max(secs.max() - secs.min(), 1.0)
        bin_of = np.minimum(np.floor((secs - t0) / span * NBINS).astype(int), NBINS - 1)
        nf = len(fl)
        flux = np.full((n, nf), np.nan, np.float32); rms = np.full((n, nf), np.nan, np.float32)
        fr_ts, fr_adm, fr_flow, fr_exact, fr_sec = [], [], [], [], []
        t1 = time.time()
        for j, (ts, path) in enumerate(fl):
            d, h = SS.read_frame(path)
            if abs(float(h["CRVAL1"]) - float(hdr0["CRVAL1"])) > 1e-3 or abs(float(h["CRVAL2"]) - float(hdr0["CRVAL2"])) > 1e-3:
                raise RuntimeError("%s %s %s: frame %s WCS differs from the first frame - refusing" % (cell, sb, st, ts))
            d = np.asarray(d, np.float32)
            fp, fts, exact = R.flow_file_for(cell, sb, ts)
            if fp is None:
                raise RuntimeError("%s %s: no flow field at all for %s" % (cell, sb, ts))
            off = np.load(fp).astype(np.float32)
            iy = np.clip(np.round(py / 8.0).astype(int), 0, off.shape[0] - 1)
            ix = np.clip(np.round(px / 8.0).astype(int), 0, off.shape[1] - 1)
            rx = px + off[iy, ix, 0]; ry = py + off[iy, ix, 1]        # raw pixel that maps onto the flowed position
            flux[:, j] = map_coordinates(d, [ry, rx], order=1, mode="constant", cval=np.nan)
            st_ = d[yy, xx]
            med = np.nanmedian(st_, axis=1)
            rms[:, j] = 1.4826 * np.nanmedian(np.abs(st_ - med[:, None]), axis=1)
            fr_ts.append(ts); fr_adm.append(adm.get(ts, True)); fr_flow.append(fts); fr_exact.append(exact); fr_sec.append(secs[j])
            if j % 50 == 0:
                log("%s %s %s: frame %d/%d (%.0fs)" % (cell, sb, st, j, nf, time.time() - t1))
        # model-beam value per row and bin
        pb = np.full((n, NBINS), np.nan, np.float32)
        for b in range(NBINS):
            for i in range(n):
                pb[i, b] = pb_value(sb, b, float(px[i]), float(py[i]), (ny, nx))
        # flux scale: hourly level over Master level on bright compact Master sources
        admv = np.array(fr_adm, bool)
        medflux = np.nanmedian(flux[:, admv], axis=1) if admv.any() else np.nanmedian(flux, axis=1)
        mp = np.array([r["master_peak"] for r in keep], float); mr = np.array([r["master_rms"] for r in keep], float)
        compact = np.array([r["kind"] == "master:S" for r in keep])
        ref = compact & np.isfinite(mp) & np.isfinite(mr) & (mr > 0) & (mp / np.where(mr > 0, mr, np.nan) >= REF_SNR) & np.isfinite(medflux)
        if st == "I":
            scale = float(np.median(medflux[ref] / mp[ref])) if ref.sum() >= 3 else np.nan
            scale_I, nref_I = scale, int(ref.sum())
        else:
            # the scale is a gain term (hourly level over Master level, Stokes I references); Stokes V carries the
            # Stokes I value of the same pass - a V median over an I peak is meaningless (smoke 2026-09-17: -0.016)
            # 2026-10-13 (refactor defect M30): the fleet runs ONE process per (subband, Stokes), so a V-only pass had no
            # Stokes I scale and wrote FLUXSCAL -1.0 (30 of 30 sampled V tables).  A V-only pass now reads FLUXSCAL and
            # NREFSCAL from this cell's Stokes I table of the same subband (written atomically by the I pass, which the fleet
            # runs beside it), waiting up to P3_LC_WAIT_I_S seconds; no I table = REFUSED (no V table is written).
            if not np.isfinite(scale_I) and nref_I == 0:
                ip = os.path.join(out_dir, "%s_%s_I_lightcurves.fits" % (cell, sb))
                t_w = time.time()
                while not os.path.exists(ip) and time.time() - t_w < float(os.environ.get("P3_LC_WAIT_I_S", "3600")):
                    time.sleep(15)
                if not os.path.exists(ip):
                    raise RuntimeError("%s %s V: no Stokes I light-curve table %s to take the flux scale from - refusing "
                                       "(Stokes V carries the Stokes I scale)" % (cell, sb, ip))
                hI = fits.getheader(ip)
                scale_I = float(hI["FLUXSCAL"]) if float(hI["FLUXSCAL"]) != -1.0 else np.nan
                nref_I = int(hI["NREFSCAL"])
                log("%s %s V: flux scale %s on %d refs taken from the Stokes I table %s"
                    % (cell, sb, "%.4f" % scale_I if np.isfinite(scale_I) else "none", nref_I, os.path.basename(ip)))
            scale, ref = scale_I, np.zeros(n, bool)
        # write
        ph = fits.PrimaryHDU()
        H = ph.header
        H["PRODUCT"] = ("snapshot_lightcurves", "Phase 3 snapshot light-curve product (test)")
        H["CELL"] = cell; H["LST"] = lst; H["DATE"] = date; H["SUBBAND"] = sb; H["STOKES"] = st
        H["NROWS"] = n; H["NFRAMES"] = nf; H["NADMIT"] = int(admv.sum())
        H["BUNIT"] = ("JY/BEAM", "raw frame units; no PB correction, no flux scale applied")
        H["PBCORR"] = (False, "FLUX/PB and RMS/PB give beam-corrected values")
        H["PBMODEL"] = ("model beam only", "Gregg 2026-09-16 ruling for the snapshot tier")
        H["PBLIB"] = SS.PB_LIBRARY[:68]
        H["FLUXSCAL"] = (scale if np.isfinite(scale) else -1.0, "median(row median / Master peak), bright compact refs")
        H["NREFSCAL"] = (int(ref.sum()) if st == "I" else nref_I, "Stokes I references; V carries the I scale")
        H["FLOWTREE"] = R.FLOWTREE[:68]
        H["STAMP"] = (STAMP, "local rms stamp size, central 5x5 excluded")
        H["NBINS"] = (NBINS, "10-minute bins over the frame span, PB per bin")
        H["CODEVER"] = git_hash(); H["EXTRACT"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # PARENT-CATALOGUE GENERATION (2026-10-13, after the contract rebuild REBUILT-SNAPSHOT-PARENT-CATALOGUES-ONE-RECIPE-20261013):
        # the sha256 of the exact Master catalogue the rows came from, so a consumer can refuse a table built on other parents
        H["MCATSHA"] = (_sha256(os.path.join(master_dir, "master_R075_%s_%s.gaul.csv" % (lst, sb)))[:64], "sha256 of the Master catalogue of the rows")
        H["MASTCAT"] = os.path.basename(master_dir)[:68]; H["RUNDIR"] = (os.path.basename(run_dir.rstrip("/")) if run_dir else "")[:68]
        hdus = [ph, fits.ImageHDU(flux, name="FLUX"), fits.ImageHDU(rms, name="RMS"), fits.ImageHDU(pb, name="PB")]
        src = fits.BinTableHDU.from_columns([
            fits.Column(name="id", format="40A", array=np.array([r["id"] for r in keep])),
            fits.Column(name="kind", format="24A", array=np.array([r["kind"] for r in keep])),
            fits.Column(name="name", format="40A", array=np.array([r["name"][:40] for r in keep])),
            fits.Column(name="ra", format="D", array=np.array([r["ra"] for r in keep])),
            fits.Column(name="dec", format="D", array=np.array([r["dec"] for r in keep])),
            fits.Column(name="px", format="E", array=px.astype(np.float32)),
            fits.Column(name="py", format="E", array=py.astype(np.float32)),
            fits.Column(name="master_peak", format="E", array=mp.astype(np.float32)),
            fits.Column(name="master_rms", format="E", array=mr.astype(np.float32)),
            fits.Column(name="median_flux", format="E", array=medflux.astype(np.float32)),
            fits.Column(name="scale_ref", format="L", array=ref)], name="SOURCES")
        frm = fits.BinTableHDU.from_columns([
            fits.Column(name="ts", format="15A", array=np.array(fr_ts)),
            fits.Column(name="epoch_s", format="D", array=np.array(fr_sec)),
            fits.Column(name="admitted", format="L", array=admv),
            fits.Column(name="bin", format="I", array=bin_of.astype(np.int16)),
            fits.Column(name="flow_ts", format="15A", array=np.array(fr_flow)),
            fits.Column(name="flow_exact", format="L", array=np.array(fr_exact, bool))], name="FRAMES")
        hdus += [src, frm]
        outp = os.path.join(out_dir, "%s_%s_%s_lightcurves.fits" % (cell, sb, st))
        # atomic (2026-10-13): a V-only pass of the same subband reads this file's FLUXSCAL while the fleet runs both
        fits.HDUList(hdus).writeto(outp + ".part", overwrite=True, output_verify="exception")
        os.replace(outp + ".part", outp)
        if st == "V":   # content test: V carries the I scale
            hV = fits.getheader(outp)
            want = scale_I if np.isfinite(scale_I) else -1.0
            if abs(float(hV["FLUXSCAL"]) - want) > 1e-9 or int(hV["NREFSCAL"]) != nref_I:
                raise RuntimeError("%s %s V: FLUXSCAL %s / NREFSCAL %s is not the Stokes I scale %s / %d - refusing"
                                   % (cell, sb, hV["FLUXSCAL"], hV["NREFSCAL"], want, nref_I))
        written.append(outp)
        log("%s %s %s: %d rows x %d frames (%d admitted), scale %.3f on %d refs, %.0fs -> %s"
            % (cell, sb, st, n, nf, int(admv.sum()), scale if np.isfinite(scale) else -1, int(ref.sum()), time.time() - t_start, outp))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract"])
    ap.add_argument("--cell", required=True)
    ap.add_argument("--subband", required=True)
    ap.add_argument("--stage", default=None, help="staged raw frames of this cell/subband (node-local); omitted = the Phase-1 "
                    "archive frames themselves (re-extraction without staging, 2026-10-13)")
    ap.add_argument("--proto-root", required=True)
    ap.add_argument("--master-catalogs", required=True)
    ap.add_argument("--run-dir", default=None, help="closed chain run whose served sources become rows")
    ap.add_argument("--samples-dir", default=CODE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stokes", default="I,V")
    ap.add_argument("--search-frames", action="store_true", help="sample exactly the search's own frame list (its noise table); "
                    "refuses when a listed frame is not discovered")
    a = ap.parse_args()
    extract(a.cell, a.subband, a.stage, a.proto_root, a.master_catalogs, a.run_dir, a.samples_dir, a.out,
            tuple(a.stokes.split(",")), search_frames=a.search_frames)


if __name__ == "__main__":
    main()
