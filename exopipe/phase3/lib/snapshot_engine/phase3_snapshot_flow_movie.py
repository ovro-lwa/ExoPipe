#!/opt/devel/pipeline/envs/py310_orca/bin/python
"""Movie of the per-frame flow field of one subband-hour (from the low-res
flow_<sb>_<ts>.npy files written by phase3_snapshot_flow.py): magnitude map
(arcmin) + quiver, one frame per 10-s image, plus a PNG of the per-frame
median / p90 / max |flow| through the hour.  usage:
  phase3_snapshot_flow_movie.py --flows DIR --subband 18MHz --pixel-arcmin 4.6 --out movie.mp4
"""
import argparse, glob, os, re, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase3_snapshot_search import ts_to_epoch_seconds, lst_deg_at   # UTC -> LST at OVRO-LWA


def ts_labels(ts):
    """'YYYYMMDD_HHMMSS' -> ('YYYY-MM-DD HH:MM:SS UTC', 'LST HHhMMm')."""
    utc = "%s-%s-%s %s:%s:%s UTC" % (ts[0:4], ts[4:6], ts[6:8], ts[9:11], ts[11:13], ts[13:15])
    lst_h = lst_deg_at(ts_to_epoch_seconds(ts)) / 15.0
    return utc, "LST %02dh%02dm" % (int(lst_h), int(round((lst_h % 1) * 60)) % 60)

ap = argparse.ArgumentParser()
ap.add_argument("--flows", required=True); ap.add_argument("--subband", required=True)
ap.add_argument("--pixel-arcmin", type=float, default=None, help="default: |CDELT2| x 60 of --header-fits (else 4.602)"); ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=float, default=None, help="frames per second (default: set by --duration-s)")
ap.add_argument("--duration-s", type=float, default=10.0, help="movie length in seconds (Gregg 2026-09-12: 10 s)")
ap.add_argument("--vmax-arcmin", type=float, default=None)
ap.add_argument("--header-fits", default=None, help="any frame of the subband: its WCS gives the horizon and elevation rings")
ap.add_argument("--lowres", type=int, default=8)
a = ap.parse_args()
# PIXEL SCALE (2026-10-13, refactor defect M31): the subband's own |CDELT2| from --header-fits (the fleet passes it, never
# --pixel-arcmin); the fixed 4.602 arcmin (the 18 MHz pixel) overstated the offsets 1.57x at 41 MHz and 2.07x at 82 MHz
if a.pixel_arcmin is None:
    if a.header_fits:
        from astropy.io import fits as _fits0
        a.pixel_arcmin = abs(float(_fits0.getheader(a.header_fits)["CDELT2"])) * 60.0
    else:
        a.pixel_arcmin = 4.602
        print("WARNING: no --header-fits and no --pixel-arcmin - assuming 4.602 arcmin per pixel (the 18 MHz grid)")
print("pixel scale %.4f arcmin per native pixel" % a.pixel_arcmin)
files = sorted(glob.glob(os.path.join(a.flows, "flow_%s_*.npy" % a.subband)))
if not files:
    raise SystemExit("no flow files")
if not a.fps:
    a.fps = max(1.0, len(files) / a.duration_s)
mags = []
for f in files:
    o = np.load(f).astype(np.float32); mags.append(np.hypot(o[..., 0], o[..., 1]) * a.pixel_arcmin)
h, w = mags[0].shape; yy, xx = np.mgrid[0:h, 0:w]
# HORIZON AND ELEVATION RINGS (Gregg 2026-09-12): zenith SIN projection, so a
# zenith angle z sits at radius sin(z) * (180/pi) / CDELT pixels from CRPIX
rings = []
if a.header_fits:
    from astropy.io import fits as _fits
    hd = _fits.getheader(a.header_fits)
    cd = abs(float(hd["CDELT1"])); cx0 = (float(hd["CRPIX1"]) - 1) / a.lowres; cy0 = (float(hd["CRPIX2"]) - 1) / a.lowres
    for el, style, lab in ((0.0, "w-", "horizon"), (10.0, "w:", "el 10 (flow mask)"), (30.0, "w--", "el 30 (search floor)")):
        rr = np.sin(np.radians(90.0 - el)) * (180.0 / np.pi) / cd / a.lowres
        rings.append((rr, style, lab))
    r_hor = rings[0][0]
    fov = (xx - cx0) ** 2 + (yy - cy0) ** 2 < r_hor ** 2
    mags = [np.where(fov, m, np.nan) for m in mags]
else:
    cx0, cy0 = w / 2, h / 2
    fov = (xx - cx0) ** 2 + (yy - cy0) ** 2 < (min(h, w) * 0.4) ** 2
vmax = a.vmax_arcmin or float(np.nanpercentile(np.stack(mags), 99))
frames_png = []
tmpdir = a.out + "_frames"; os.makedirs(tmpdir, exist_ok=True)
series = []
for f, mag in zip(files, mags):
    ts = re.search(r"_(\d{8}_\d{6})\.npy$", f).group(1); o = np.load(f).astype(np.float32)
    med, p90, mx = np.nanmedian(mag[fov]), np.nanpercentile(mag[fov], 90), np.nanmax(mag[fov])
    series.append((ts, med, p90, mx))
    fig, ax = plt.subplots(figsize=(7.2, 6.9)); fig.subplots_adjust(bottom=0.09)
    im = ax.imshow(mag, origin="lower", cmap="magma", vmin=0, vmax=vmax)
    st = max(1, h // 24)
    q = ax.quiver(xx[::st, ::st], yy[::st, ::st], np.where(fov, o[..., 0], np.nan)[::st, ::st], np.where(fov, o[..., 1], np.nan)[::st, ::st], color="cyan", scale=None, width=0.003)
    if rings:
        from matplotlib.patches import Circle
        q.set_clip_path(Circle((cx0, cy0), rings[0][0], transform=ax.transData))   # vectors never cross the horizon
    th = np.linspace(0, 2 * np.pi, 361)
    for rr, style, lab in rings:
        ax.plot(cx0 + rr * np.cos(th), cy0 + rr * np.sin(th), style, lw=1.0, alpha=0.9, label=lab)
    if rings:
        lg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=8,
                       facecolor="black", edgecolor="gray", framealpha=0.9)
        for t in lg.get_texts():
            t.set_color("white")
    ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(-0.5, h - 0.5)
    utc_lab, lst_lab = ts_labels(ts)
    ax.set_title("%s per-frame flow to the multi-night master\n%s   %s\n|flow| median %.1f  p90 %.1f  max %.1f arcmin" % (a.subband, utc_lab, lst_lab, med, p90, mx), fontsize=9)
    plt.colorbar(im, ax=ax, label="|flow| (arcmin)"); ax.set_xticks([]); ax.set_yticks([])
    png = os.path.join(tmpdir, "%s.png" % ts); fig.savefig(png, dpi=80); plt.close(fig); frames_png.append(png)
# encode with the ffmpeg binary (imageio here has no video backend); GIF fallback
import subprocess
FFMPEG = os.environ.get("FFMPEG", "/opt/devel/pipeline/envs/py38_orca_nkosogor/bin/ffmpeg")
lst = os.path.join(tmpdir, "frames.txt")
with open(lst, "w") as fh:
    for p in frames_png:
        fh.write("file '%s'\nduration %.4f\n" % (p, 1.0 / a.fps))
r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p", "-c:v", "libx264", "-crf", "23", a.out])
if r.returncode != 0:
    gif = a.out.rsplit(".", 1)[0] + ".gif"
    imageio.mimsave(gif, [imageio.imread(p) for p in frames_png], duration=1.0 / a.fps)
    print("ffmpeg failed (rc %d); wrote GIF %s instead" % (r.returncode, gif)); a.out = gif
t = np.arange(len(series)); fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(t, [s[1] for s in series], label="median |flow|"); ax.plot(t, [s[2] for s in series], label="p90"); ax.plot(t, [s[3] for s in series], label="max", alpha=0.6)
lab = [s[0][9:13] for s in series]; ax.set_xticks(t[::30]); ax.set_xticklabels([l[:2] + ":" + l[2:] for l in lab[::30]])
ax.set_ylabel("arcmin"); ax.set_xlabel("UT"); ax.set_title("%s per-frame flow to the master, inside the field of view" % a.subband); ax.legend(); fig.tight_layout()
fig.savefig(a.out.rsplit(".", 1)[0] + "_series.png", dpi=110)
with open(a.out.rsplit(".", 1)[0] + "_series.csv", "w") as fh:
    cw = csv.writer(fh); cw.writerow(["ts", "median_arcmin", "p90_arcmin", "max_arcmin"]); cw.writerows(series)
print("movie %s (%d frames), series png/csv written" % (a.out, len(frames_png)))
