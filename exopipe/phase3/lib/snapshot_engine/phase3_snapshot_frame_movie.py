#!/opt/devel/pipeline/envs/py310_orca/bin/python
"""MOVIES OF THE 10-s FRAMES of one cell (Gregg 2026-09-26: "A movie of the 10-sec frames is also welcome for every cell
if not expensive or large in size"; 2026-09-27: "19 (not 15) is preferred"): one movie per subband (15) and per wideband
group (Red, Green, Blue, Full).  Each movie frame = the flowed Stokes I frame from the cell's cube minus the night's
per-pixel median of the frames the night rule admits (the all-sky SUBTRACTED view in which ionospheric features show),
2x2-binned; a wideband frame is the inverse-variance weighted mean of its subbands' subtracted frames of that INTEGRATION.
FRAME TIME (frame-time fix 2026-10-07, minimal form; Gregg 2026-10-07 "Go with your recommendation"): every frame is placed,
labelled and judged by the night rule at its own integration midpoint (header DATE-OBS + half an integration, truncated to the
second - phase3_snapshot_search.integration_label), so a wideband movie frame combines frames of ONE integration; until then
frames were keyed by Phase-1 file-name times, which drift by up to 26 s differently per subband, and a wideband movie frame held
whichever subbands happened to share a name (680-1,489 movie frames per hour instead of 358).
Greyscale sky (CLAUDE.md 8b), one fixed linear stretch per movie (+/- 12 x the median frame rms, Gregg's choice); UTC, LST, Sun elevation
and the night-rule verdict annotated in colour (red = the frame is refused by the night rule).
Sources: the cell's flowed cubes (default), or --frames-root DIR = the fleet's node-local staging (<DIR>/<subband>/
*-Snapshot-<ts>[-I]-image_flowed.fits), so the fleet makes the movies while the frames are staged (Gregg 2026-09-27:
movies are made whenever a cell is staged; no dedicated backfill).
usage: phase3_snapshot_frame_movie.py CELL OUT_DIR [--cube-root R] [--alt-cube-root R2] [--frames-root S] [--bands all|list] [--k 6]
"""
import argparse, os, sys, subprocess
import numpy as np
from astropy.io import fits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3_snapshot_search import sun_state, night_verdict, ts_to_epoch_seconds, lst_deg_at, _NIGHT_GROUPS, frame_dateobs, \
    integration_label
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FFMPEG = os.environ.get("FFMPEG", "/opt/devel/pipeline/envs/py38_orca_nkosogor/bin/ffmpeg")
SUBS = ["18MHz", "23MHz", "27MHz", "32MHz", "36MHz", "41MHz", "46MHz", "50MHz", "55MHz", "59MHz", "64MHz", "69MHz", "73MHz", "78MHz", "82MHz"]
# PORT + SPEED-UP 5 (refactor, 2026-11-11; identical movies).  Production runs this file top to bottom at module level; here the
# same statements live in main() and build_band(), so the module imports without side effects (tests/test_import_side_effects.py),
# and two loops run in parallel (speed only):
#   P3_SNAP_MOVIE_THREADS  frames of a subband read and binned at a time (each frame fills its own slot of the cube);
#   P3_SNAP_MOVIE_PROCS    bands rendered at a time, each in a forked child that already holds the residual cubes; the parent
#                          prints the per-band lines and adds the sizes in production's band order.
# 1 and 1 = production's serial order.  The functions read the parsed arguments, the cell and RES as module globals, as before.
MOVIE_THREADS = max(1, int(os.environ.get("P3_SNAP_MOVIE_THREADS", "1")))
MOVIE_PROCS = max(1, int(os.environ.get("P3_SNAP_MOVIE_PROCS", "1")))
a = None
lst = date = None
RES = {}          # subband -> (ts list, residual cube float16, rms)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("cell"); ap.add_argument("out")
    ap.add_argument("--cube-root", default="/lustre/pipeline/exopipe/phase3/snapshot_cubes")
    ap.add_argument("--alt-cube-root", default="/lustre/gh/main/phase3/sublst/cubes_deepflow_18_23",
                    help="preferred source for a subband whose flowed cube lives here (19h 18/23 MHz re-flow)")
    ap.add_argument("--frames-root", default=None, help="read staged flowed frames instead of cubes")
    ap.add_argument("--bands", default="all")
    ap.add_argument("--k", type=float, default=12.0, help="linear stretch +/- K x the median frame rms (Gregg 2026-09-27 chose +/-12 "
                    "from +/-3, 6, 12 on 19h 2025-04-10 32 MHz)")
    ap.add_argument("--size", type=int, default=754, help="target movie width in pixels: each band is binned by round(width/size), "
                    "so the 1508-px low bands and the 2358-px high bands come out alike (2026-09-27: the high bands were 100+ MB)")
    ap.add_argument("--fps", type=float, default=24.0); ap.add_argument("--crf", type=int, default=26)
    return ap.parse_args(argv)


def load_staged(sb):
    import glob as _g, re as _re
    fl = []
    # Stokes I only, as the fleet selects it: new-era CLEAN frames (Clean-Snapshot; Pilot-Snapshot without a Stokes tag is
    # the V frame of that era) and pilot-era frames tagged -I-
    for f in sorted(_g.glob(os.path.join(a.frames_root, sb, "*Clean-Snapshot-*-image_flowed.fits")) +
                    _g.glob(os.path.join(a.frames_root, sb, "*Pilot-Snapshot-*-I-image_flowed.fits"))):
        m = _re.search(r"-Snapshot-(\d{8}_\d{6})", os.path.basename(f))
        if m: fl.append((m.group(1), f))
    if len(fl) < 10:
        print("too few staged flowed frames for", sb); return None
    dob = frame_dateobs([f for _, f in fl])               # each frame's own integration (frame-time fix 2026-10-07)
    nod = [f for _, f in fl if not dob.get(f)]
    if nod:
        sys.exit("REFUSED: %d staged %s frames have no DATE-OBS (first %s) - a movie frame is placed by its own integration"
                 % (len(nod), sb, nod[0]))
    fl = sorted((integration_label(dob[f]), f) for _, f in fl)
    ts = [t for t, _ in fl]
    d = np.squeeze(fits.getdata(fl[0][1])).astype(np.float32)
    ny, nx = d.shape; b = max(1, int(round(nx / float(a.size)))); ny2, nx2 = ny // b, nx // b
    cube = np.empty((len(fl), ny2, nx2), np.float32)
    cube[0] = np.nanmean(d[:ny2 * b, :nx2 * b].reshape(ny2, b, nx2, b), axis=(1, 3))

    def _one(i):
        d = np.squeeze(fits.getdata(fl[i][1])).astype(np.float32)
        cube[i] = np.nanmean(d[:ny2 * b, :nx2 * b].reshape(ny2, b, nx2, b), axis=(1, 3))

    if MOVIE_THREADS > 1 and len(fl) > 2:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MOVIE_THREADS) as ex:
            list(ex.map(_one, range(1, len(fl))))
    else:
        for i in range(1, len(fl)):
            _one(i)
    return ts, cube


def load(sb):
    if a.frames_root:
        r = load_staged(sb)
        if r is None: return None
        ts, cube = r; ny2, nx2 = cube.shape[1:]
        return _residual(sb, ts, cube, ny2, nx2)
    p = None
    for root in (a.alt_cube_root, a.cube_root):
        q = os.path.join(root, a.cell, "%s_%s_I_flowcube.fits" % (a.cell, sb))
        if os.path.exists(q):
            with fits.open(q, memmap=True) as h:
                if h[0].header.get("FLOWED") and not h[0].header.get("NOMASTER"):
                    p = q; break
    if p is None:
        print("no flowed cube for", sb); return None
    with fits.open(p, memmap=True) as h:
        # each frame's own integration label from its stored header (the 'ts' column is the file-name time; frame-time fix)
        ts = [integration_label(fits.Header.fromstring(x if isinstance(x, str) else x.decode(), sep="\n")["DATE-OBS"])
              for x in h["FRAME_TABLE"].data["header"]]
        # the cube stores NaN pixels as 0 and marks them in MASK (phase3_snapshot_cube.Cube.frame restores them); read
        # directly, a blank instant came out as all zeros and its movie frame as the night median inverted (Gregg
        # 2026-09-27, frames_19h_2025-04-10_32MHz_I.mp4) - the mask is applied here as the reader does
        F = h["FRAMES"].data; MK = h["MASK"].data; n, ny, nx = F.shape; b = max(1, int(round(nx / float(a.size)))); ny2, nx2 = ny // b, nx // b
        cube = np.empty((n, ny2, nx2), np.float32)
        for i in range(n):
            f = np.array(F[i], np.float32); f[np.asarray(MK[i]) != 0] = np.nan
            f = f[:ny2 * b, :nx2 * b]
            cube[i] = np.nanmean(f.reshape(ny2, b, nx2, b), axis=(1, 3))
    return _residual(sb, ts, cube, ny2, nx2)


def _residual(sb, ts, cube, ny2, nx2):
    for i in range(len(cube)):                     # an all-zero frame (no gridded visibilities) is a blank frame too
        if np.isfinite(cube[i]).any() and not np.any(np.nan_to_num(cube[i]) != 0):
            cube[i] = np.nan
    ok = np.array([night_verdict(t, lst, date, sb) == "" for t in ts]) & np.array([np.isfinite(c).mean() > 0.5 for c in cube])
    use = ok if ok.sum() >= 10 else np.array([np.isfinite(c).mean() > 0.5 for c in cube])
    if ok.sum() < 10:   # recorded, not silent (refactor chat 2026-10-14: every minimum-count skip is counted)
        print("BACKGROUND %s: %d admitted frames < 10 - background from all %d frames with data" % (sb, int(ok.sum()), int(use.sum())), flush=True)
    res = cube - np.nanmedian(cube[use], axis=0)
    c = res[:, ny2 // 4:3 * ny2 // 4, nx2 // 4:3 * nx2 // 4]
    rms = float(np.nanmedian([1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))) for x in c[use]]))
    return ts, res.astype(np.float16), rms


def movie(band, ts, frames, rms, out):
    lim = a.k * rms; enc = None
    for t, fr in zip(ts, frames):
        img = np.clip((np.nan_to_num(fr.astype(np.float32)) + lim) / (2 * lim), 0, 1)
        ny2, nx2 = img.shape
        fig = plt.figure(figsize=(nx2 / 100, ny2 / 100 + 0.4), dpi=100)
        ax = fig.add_axes([0, 0, 1, ny2 / (ny2 + 40)]); ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest"); ax.axis("off")
        alt, br, _ = sun_state(t); v = night_verdict(t, lst, date, band); lh = lst_deg_at(ts_to_epoch_seconds(t)) / 15.0
        if not np.isfinite(fr.astype(np.float32)).mean() > 0.5:
            v = (v + "; " if v else "") + "BLANK FRAME (no data)"
        fig.text(0.01, 0.985, "%s %s  %s-%s-%s %s:%s:%s UTC  LST %02dh%02dm  Sun %+.1f deg %s  %s  frame minus night median, +/-%.2f Jy/beam" % (
            a.cell, band, t[:4], t[4:6], t[6:8], t[9:11], t[11:13], t[13:15], int(lh), int(round((lh % 1) * 60)) % 60, alt, br,
            ("NIGHT RULE: " + v) if v else "admitted", lim), color="red" if v else "tab:blue", fontsize=8, va="top")
        fig.canvas.draw(); rgb = np.ascontiguousarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]); plt.close(fig)
        if enc is None:
            H, W = rgb.shape[:2]
            enc = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
                                    "-r", str(a.fps), "-i", "-", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
                                    "-c:v", "libx264", "-crf", str(a.crf), out], stdin=subprocess.PIPE)
        enc.stdin.write(rgb.tobytes())
    enc.stdin.close(); enc.wait()
    return os.path.getsize(out) / 1e6


def build_band(band):
    """(ts, frames, rms) of one movie - a subband's own residual cube, or the inverse-variance weighted mean of a wideband's
    subbands per integration - or None when no constituent has a flowed cube.  Production's loop body, unchanged."""
    subs = [s for s in _NIGHT_GROUPS.get(band, [band]) if s in RES]
    if not subs:
        return None
    if len(subs) == 1 and band in RES:
        ts, fr, rms = RES[band]
    else:
        tsets = sorted(set().union(*[set(RES[s][0]) for s in subs]))
        # constituents on different pixel grids (the Full group mixes 1508-px and 2358-px bands; same sky extent): each is
        # resampled to the smallest shape of the group
        shp = min((RES[s][1].shape[1:] for s in subs), key=lambda t: t[0])
        from scipy.ndimage import zoom as _zoom
        def _fit(x):
            if x.shape == shp: return x
            m = np.isfinite(x); z = _zoom(np.where(m, x, 0.0), (shp[0] / x.shape[0], shp[1] / x.shape[1]), order=1)
            zm = _zoom(m.astype(np.float32), (shp[0] / x.shape[0], shp[1] / x.shape[1]), order=1)
            return np.where(zm > 0.5, z / np.maximum(zm, 1e-6), np.nan)[:shp[0], :shp[1]]
        idx = {s: {t: i for i, t in enumerate(RES[s][0])} for s in subs}
        fr = []
        for t in tsets:
            acc = None; wsum = None
            for s in subs:
                i = idx[s].get(t)
                if i is None: continue
                x = _fit(RES[s][1][i].astype(np.float32)); w = 1.0 / RES[s][2] ** 2
                m = np.isfinite(x)
                acc = np.where(m, x * w, 0.0) if acc is None else acc + np.where(m, x * w, 0.0)
                wsum = np.where(m, w, 0.0) if wsum is None else wsum + np.where(m, w, 0.0)
            fr.append((acc / np.where(wsum > 0, wsum, np.nan)).astype(np.float16))
        ts = tsets; rms = 1.0 / np.sqrt(sum(1.0 / RES[s][2] ** 2 for s in subs))
    return ts, fr, rms


def _render(band):
    """One band's movie file; returns (number of movie frames, size in MB), or None when the band is skipped."""
    r = build_band(band)
    if r is None:
        return None
    ts, fr, rms = r
    mb = movie(band, ts, fr, rms, os.path.join(a.out, "frames_%s_%s_I.mp4" % (a.cell, band)))
    return len(ts), mb


def main(argv=None):
    global a, lst, date
    a = parse_args(argv)
    lst, date = a.cell.split("_", 1)
    os.makedirs(a.out, exist_ok=True)
    want = SUBS + ["Red", "Green", "Blue", "Full"] if a.bands == "all" else a.bands.split(",")
    need = sorted({s for w in want for s in _NIGHT_GROUPS.get(w, [w])}, key=lambda s: int(s[:2]))
    for sb in need:
        r = load(sb)
        if r: RES[sb] = r
    if MOVIE_PROCS > 1 and len(want) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        sys.stdout.flush()
        # forked children inherit RES (nothing is pickled in).  ProcessPoolExecutor, not multiprocessing.Pool: when a child
        # dies (killed for memory, a hard exit) the executor raises BrokenProcessPool here, where Pool.map would wait forever
        # (review 2026-11-11) - the movie step fails loudly, as the serial loop does
        with ProcessPoolExecutor(max_workers=MOVIE_PROCS, mp_context=multiprocessing.get_context("fork")) as pool:
            done = list(pool.map(_render, want))
    else:
        done = None
    tot = 0.0
    for k, band in enumerate(want):
        r = done[k] if done is not None else _render(band)
        if r is None:
            print(band, "skipped: no flowed cube"); continue
        nfr, mb = r
        tot += mb
        print("%s %s: %d frames, %.1f MB" % (a.cell, band, nfr, mb), flush=True)
    print("TOTAL %.0f MB" % tot)


if __name__ == "__main__":
    main()
