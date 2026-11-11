#!/opt/devel/pipeline/envs/py310_orca/bin/python
"""Per-frame flow registration of snapshot frames to the multi-night master
(Gregg 2026-09-12: "Per-frame flow to the multi-night Master followed by
median subtraction").  Runs in the JAX flow environment (py310_orca, GPU);
the search engine (python 3.6, development env) never imports this.

For every Stokes I frame in the staged subband directory:
  preprocess(frame, master) -> Flow.brox -> flow.apply(frame) and
  flow.apply(V frame of the same timestamp)  -> <base>_flowed.fits (2-D)
The flow field is kept at 1/LOWRES resolution (float16) in
<out>/flow_<subband>.npz together with per-frame statistics in
<out>/flow_<subband>.csv (median / p90 / max |flow| inside the field of
view, mean dx, dy, seconds).  The 1-min blocks, the hour median and the
wideband groups are then formed from the flowed frames by the engine
(P3_SNAP_FLOW=1), so nothing downstream is aligned separately.

usage: phase3_snapshot_flow.py --stage DIR --reference MASTER.fits --subband 18MHz
                               --out DIR [--lowres 8] [--frames LISTFILE] [--overwrite]
Test implementation 2026-09-12 (one cell, 02h 2024-12-29); flow parameters =
align_images.py defaults (preprocess_weight 1.5, alpha 1.3, gamma 150,
scale_factor 0.7).
"""
import argparse, csv, glob, os, re, sys, time
import numpy as np

PAT = re.compile(r"(?P<sb>\d+MHz)-(?P<kind>Clean|Pilot)-Snapshot-(?P<ts>\d{8}_\d{6})(?:-(?P<st>[IV]))?-image\.fits$")
# (?P<st>) = the PILOT-ERA naming (pre-2024-12-27 cells: <sb>-Pilot-Snapshot-<ts>-I-image.fits and -V-image.fits, the
# search engine's _PAT_OLD).  Without it the helper selected no Stokes I frame on a pilot cell and flowed 0 of 10,024
# (01h 2024-12-20 and 12-21, 2026-09-20: FLOW-INCOMPLETE on both, the dirty Stokes I pilot never reached the search).


def fix_header_2d(orig):
    h = orig.copy()
    h["NAXIS"] = 2
    for ax in (3, 4):
        for k in ("NAXIS", "CTYPE", "CRVAL", "CRPIX", "CDELT", "CUNIT", "CROTA"):
            h.pop("%s%d" % (k, ax), None)
    for k in list(h.keys()):
        if re.match(r"PC[34]_\d|PC\d_[34]|CD[34]_\d|CD\d_[34]", k):
            h.pop(k, None)
    h.pop("WCSAXES", None)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--subband", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lowres", type=int, default=8)
    ap.add_argument("--frames", default=None, help="file listing the I frame paths to process (default: all in --stage)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--preprocess-weight", type=float, default=1.5)
    ap.add_argument("--alpha", type=float, default=1.3)
    ap.add_argument("--gamma", type=float, default=150.0)
    ap.add_argument("--scale-factor", type=float, default=0.7)
    ap.add_argument("--horizon-elevation-deg", type=float, default=10.0)
    ap.add_argument("--block-s", type=int, default=0,
                    help="solve the flow on the mean of the frames in each wall-clock block of this many seconds "
                         "(60 = 1-min blocks) and apply it to every frame of the block; 0 = per frame")
    a = ap.parse_args()

    import jax.numpy as jnp
    from astropy.io import fits
    from image_plane_correction import flow as F
    from image_plane_correction.preprocessing import preprocess
    from image_plane_correction import data as ipdata

    os.makedirs(a.out, exist_ok=True)
    files = [l.strip() for l in open(a.frames) if l.strip()] if a.frames else \
        sorted(glob.glob(os.path.join(a.stage, "*Clean-Snapshot-*-image.fits")) +
               glob.glob(os.path.join(a.stage, "*Pilot-Snapshot-*-I-image.fits")))
    vmap = {}
    for f in glob.glob(os.path.join(a.stage, "*Pilot-Snapshot-*-image.fits")):
        m = PAT.search(os.path.basename(f))
        if m and m.group("st") != "I":
            vmap[m.group("ts")] = f

    ref, refwcs = ipdata.fits_image(a.reference)
    ref = jnp.array(np.nan_to_num(np.asarray(ref, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0))
    n = int(np.asarray(ref).shape[0])
    horizon_r = F.horizon_r_normalized(refwcs, n=n, horizon_elevation_deg=a.horizon_elevation_deg)
    h, w = np.asarray(ref).shape
    yy, xx = np.ogrid[:h, :w]
    fov = (xx - w // 2) ** 2 + (yy - h // 2) ** 2 < (min(h, w) * 0.4) ** 2

    stats = []
    lowres = {}
    t_all = time.time()

    def ts_sec(ts):
        return int(ts[9:11]) * 3600 + int(ts[11:13]) * 60 + int(ts[13:15])

    def load_i(f):
        img, imwcs = ipdata.fits_image(f)
        img_np = np.asarray(img, dtype=np.float32)
        fin = np.isfinite(img_np)
        blank = fin.mean() < 0.5 or not np.any(img_np[fin] != 0)
        return img_np, fin, blank

    flow_cache = {}          # ts -> Flow object of a VALID I frame (kept small: nearest lookups only)

    def compute_flow(f, img_np, fin):
        img = jnp.array(np.nan_to_num(img_np, nan=float(np.median(img_np[fin])), posinf=0.0, neginf=0.0))
        ip, sp = preprocess(img, ref, weight=a.preprocess_weight, horizon_r=horizon_r)
        return img, F.Flow.brox(ip, sp, alpha=a.alpha, gamma=a.gamma, scale_factor=a.scale_factor)

    valid_ts = {}            # ts -> path of I frames with data (filled on the first pass over files)
    for f in files:
        m = PAT.search(os.path.basename(f))
        if m:
            valid_ts[m.group("ts")] = f

    def nearest_valid_flow(ts):
        """Flow of the nearest-in-time I frame that has data (for an instant whose
        I frame is blank: the V frame of that instant still needs registering)."""
        order = sorted(valid_ts, key=lambda t: abs(ts_sec(t) - ts_sec(ts)))
        for t in order:
            if t == ts:
                continue
            if t in flow_cache:
                return t, flow_cache[t]
            img_np, fin, blank = load_i(valid_ts[t])
            if blank:
                continue
            _, fl = compute_flow(valid_ts[t], img_np, fin)
            if len(flow_cache) > 4:
                flow_cache.clear()
            flow_cache[t] = fl
            return t, fl
        return None, None

    block_flow = {}          # block key -> Flow solved on the block mean (--block-s > 0)

    def block_key(ts):
        return ts[:13] if a.block_s == 60 else ts[:9] + "%05d" % ((ts_sec(ts) // a.block_s) * a.block_s)

    def flow_for_block(key):
        if key in block_flow:
            return block_flow[key]
        members = [valid_ts[t] for t in valid_ts if block_key(t) == key]
        acc, n = None, 0
        for fm in members:
            im, fi, bl = load_i(fm)
            if bl:
                continue
            im = np.where(fi, im, np.nan)
            acc = im if acc is None else np.nansum(np.stack([acc, im]), axis=0) if False else (acc + np.nan_to_num(im))
            n += 1
        if n == 0:
            block_flow[key] = None
            return None
        mean = acc / n
        fin = np.isfinite(mean)
        _, fl = compute_flow(None, mean.astype(np.float32), fin)
        if len(block_flow) > 3:
            block_flow.clear()
        block_flow[key] = fl
        return fl

    for k, f in enumerate(files):
        m = PAT.search(os.path.basename(f))
        ts = m.group("ts") if m else os.path.basename(f)
        out_i = f[:-5] + "_flowed.fits"
        fv = vmap.get(ts)
        out_v = fv[:-5] + "_flowed.fits" if fv else None
        if not a.overwrite and os.path.exists(out_i) and (out_v is None or os.path.exists(out_v)):
            continue
        t0 = time.time()
        try:
            img_np, fin, blank = load_i(f)
            if blank:
                # BLANK I FRAME (no-data integration, Phase-1 QA flag): written through
                # unchanged (it is refused by the engine's admission as before); the V
                # frame of this instant takes the flow of the nearest valid I instant
                hdr = fix_header_2d(fits.getheader(f))
                hdr["FLOWED"] = (False, "blank frame written through; no flow")
                fits.writeto(out_i, img_np, hdr, overwrite=True)
                if fv:
                    tnear, fl = nearest_valid_flow(ts)
                    vimg, _ = ipdata.fits_image(fv)
                    v_np = np.asarray(vimg, dtype=np.float32); vfin = np.isfinite(v_np)
                    hv = fix_header_2d(fits.getheader(fv))
                    if fl is None:
                        dv = v_np; hv["FLOWED"] = (False, "no valid I flow in this subband-hour")
                    else:
                        v = jnp.array(np.nan_to_num(v_np, nan=float(np.median(v_np[vfin])) if vfin.any() else 0.0, posinf=0.0, neginf=0.0))
                        dv = np.asarray(fl.apply(v), dtype=np.float32)
                        hv["FLOWED"] = (True, "flow of the nearest valid I instant %s" % tnear)
                        hv["FLOWREF"] = (os.path.basename(a.reference)[:60], "per-frame flow reference (multi-night master)")
                    fits.writeto(out_v, dv, hv, overwrite=True)
                stats.append([ts, "blank-I; V flowed from nearest valid instant", "", "", "", "", "", "%.2f" % (time.time() - t0)])
                continue
            if a.block_s > 0:
                flow = flow_for_block(block_key(ts))
                img = jnp.array(np.nan_to_num(img_np, nan=float(np.median(img_np[fin])), posinf=0.0, neginf=0.0))
                if flow is None:
                    img, flow = compute_flow(f, img_np, fin)
            else:
                img, flow = compute_flow(f, img_np, fin)
            if len(flow_cache) > 4:
                flow_cache.clear()
            flow_cache[ts] = flow
            dew = np.asarray(flow.apply(img), dtype=np.float32)
            hdr = fix_header_2d(fits.getheader(f))
            hdr["FLOWREF"] = (os.path.basename(a.reference)[:60], "per-frame flow reference (multi-night master)")
            hdr["FLOWED"] = (True, "frame registered to the master by per-frame flow" if a.block_s == 0 else "flow solved on the %d-s block mean" % a.block_s)
            hdr["FLOWBLK"] = (int(a.block_s), "flow block length in seconds (0 = per frame)")
            fits.writeto(out_i, dew, hdr, overwrite=True)
            if fv:
                vimg, _ = ipdata.fits_image(fv)
                v_np = np.asarray(vimg, dtype=np.float32)
                vfin = np.isfinite(v_np)
                v = jnp.array(np.nan_to_num(v_np, nan=float(np.median(v_np[vfin])) if vfin.any() else 0.0, posinf=0.0, neginf=0.0))
                dv = np.asarray(flow.apply(v), dtype=np.float32)
                hv = fix_header_2d(fits.getheader(fv))
                hv["FLOWREF"] = (os.path.basename(a.reference)[:60], "flow taken from the Stokes I frame of this instant")
                hv["FLOWED"] = (True, "frame registered to the master by per-frame flow")
                fits.writeto(out_v, dv, hv, overwrite=True)
            off = np.asarray(flow.offsets, dtype=np.float32)          # (H, W, 2): dx, dy in pixels
            mag = np.hypot(off[..., 0], off[..., 1])[fov]
            lr = off[::a.lowres, ::a.lowres].astype(np.float16)
            np.save(os.path.join(a.out, "flow_%s_%s.npy" % (a.subband, ts)), lr)
            lowres[ts] = lr
            dt = time.time() - t0
            stats.append([ts, "ok", "%.3f" % np.median(mag), "%.3f" % np.percentile(mag, 90), "%.3f" % mag.max(),
                          "%.4f" % off[fov, 0].mean(), "%.4f" % off[fov, 1].mean(), "%.2f" % dt])
            print("%s %s median|flow| %.2f px p90 %.2f max %.2f  %.1fs  (%d/%d)" % (
                a.subband, ts, np.median(mag), np.percentile(mag, 90), mag.max(), dt, k + 1, len(files)), flush=True)
        except Exception as e:
            stats.append([ts, "FAILED: %s" % str(e)[:80], "", "", "", "", "", "%.2f" % (time.time() - t0)])
            print("%s %s FAILED: %s" % (a.subband, ts, e), flush=True)
    with open(os.path.join(a.out, "flow_%s.csv" % a.subband), "a") as fh:
        wcsv = csv.writer(fh)
        if fh.tell() == 0:
            wcsv.writerow(["ts", "state", "median_flow_px", "p90_flow_px", "max_flow_px", "mean_dx_px", "mean_dy_px", "seconds"])
        wcsv.writerows(stats)
    print("done %s: %d frames in %.0f s" % (a.subband, len(files), time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
