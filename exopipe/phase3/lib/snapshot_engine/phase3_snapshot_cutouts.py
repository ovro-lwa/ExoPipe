#!/usr/bin/env python
"""PHOTOMETRY CUTOUT PRODUCT (Gregg 2026-09-11, approved):

  "Irrespective of a detection or not, I want to make a 5x5 degree cutout for each Hot/Warm
   Jupiter and for each 10pc star for each band from every image after dewarping and median
   subtraction. I also want the same for Jupiter, Saturn, Uranus and Neptune ... /lustre/pipeline/
   exopipe/phase3/photometry/sample/band/object/ where sample is HWJ or 10pc or SolarSystem ...
   headers, WCS etc included for every image."

Implementation (inside the search engine, while the cube is in memory): for every target above
MIN_EL_DEG at the cell's mid-time and inside the image, cut a CUT_DEG box of the RESIDUAL
(frame minus hour median) from every admitted 10-s frame, dewarped locally (raw box + warp
screen -> WCS grid), and write ONE FITS cube per (object, band, Stokes, cell-hour):
  <root>/<sample>/<band>/<object>/<object>_<LST>_<UTCstart>_<Stokes>.fits
  HDU 0: cube [n_frames, ny, nx] float32, Jy/beam, full celestial WCS of the cutout, beam,
         OBJECT, SAMPLE, LST, DATE, BAND, STOKES, CELL, UTCSTART, TARGRA/TARGDEC, MODEL, TIER
  HDU 1 (TIMES): per-frame ts, DATE-OBS, MJD-MID, admitted flag, frame global rms
One cube per cell-hour rather than one file per frame: 65 targets x 18 bands x 358 frames x 2
Stokes would be 840,000 files per cell-hour, which Lustre creates at 181 per second.
Moving targets (planets) are cut at their per-frame ephemeris position (10-min quantised) and the
per-frame RA/Dec go into the TIMES table.
"""
import os, csv, math, logging, datetime
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage

log = logging.getLogger("snapcut")
CUT_DEG = 5.0
MIN_EL_DEG = 20.0          # targets considered when above this at the cell's mid-time (the search domain is 30)
MERGE_ARCMIN = 1.0         # list entries sharing one name within this are one object (one cutout file)
ROOT_DEFAULT = "/lustre/pipeline/exopipe/phase3/photometry"
SAMPLES = (("HWJ", "Hot_Warm_Jupiters.csv"), ("10pc", "10pc.csv"))
PLANETS = ("jupiter", "saturn", "uranus", "neptune")
SAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
_TARGETS = None
_EPH = {}


def _samples():
    """[(tag, csv path)] - AD-HOC TARGET SETS (2026-10-28, Gregg: "Can you produce forced photometry of UX Ari and FI Cnc please?" ...
    "all nights"): P3_SNAP_CUTOUT_SAMPLES="TAG=path.csv[,TAG2=path2.csv]" replaces the HWJ and 10 pc lists for this process (and the
    planets are then not cut unless P3_SNAP_CUTOUT_PLANETS=1); unset = the lists and planets as before."""
    env = os.environ.get("P3_SNAP_CUTOUT_SAMPLES", "")
    if env:
        return [tuple(x.split("=", 1)) for x in env.split(",") if "=" in x]
    return [(tag, os.path.join(SAMPLES_DIR, fn)) for tag, fn in SAMPLES]


def _planets():
    return PLANETS if (not os.environ.get("P3_SNAP_CUTOUT_SAMPLES") or os.environ.get("P3_SNAP_CUTOUT_PLANETS") == "1") else ()


def _targets():
    global _TARGETS
    if _TARGETS is None:
        out = []
        for tag, p in _samples():
            if not os.path.exists(p):
                continue
            for r in csv.DictReader(open(p, encoding="utf-8-sig")):
                ra = r.get("ra_current") or r.get("ra") or r.get("RA"); dec = r.get("dec_current") or r.get("dec") or r.get("DEC")
                nm = (r.get("common_name") or r.get("name") or r.get("Name") or "").strip()
                try:
                    out.append((tag, nm, float(ra), float(dec)))
                except (TypeError, ValueError):
                    continue
        # A NAME GIVEN TO DISTINCT OBJECTS (2026-10-25: 10pc.csv names 11 different white dwarfs 'Gaia DR3 WD_Unknown_ID', and every
        # cutout of a cell and band went to the one file of that name, so only the last-written survived): entries of one name more
        # than MERGE_ARCMIN apart each take the name plus their J-tag; entries within it stay one object (one file), as before
        by = {}
        for i, (tag, nm, ra, dec) in enumerate(out):
            by.setdefault((tag, safe_name(nm)), []).append(i)
        for idx in by.values():
            if len(idx) > 1 and max(_sep_arcmin(out[i][2:], out[j][2:]) for i in idx for j in idx) > MERGE_ARCMIN:
                for i in idx:
                    tag, nm, ra, dec = out[i]
                    out[i] = (tag, "%s %s" % (nm, _jtag(ra, dec)), ra, dec)
        by = {}
        for tag, nm, ra, dec in out:
            by.setdefault((tag, safe_name(nm)), []).append((ra, dec))
        for k, pos in by.items():
            if len(pos) > 1 and max(_sep_arcmin(a, b) for a in pos for b in pos) > MERGE_ARCMIN:
                raise RuntimeError("REFUSED: target name %s/%s still names objects more than %.1f arcmin apart: %s" % (k + (MERGE_ARCMIN, pos)))
        _TARGETS = out
    return _TARGETS


def _jtag(ra, dec):
    """'J<hhmm><+/-><ddmm>' of a list position (truncated, as catalog designations are; the forced panels' tag)."""
    m = int(ra / 15.0 * 60.0); dm = int(abs(dec) * 60.0)
    return "J%02d%02d%s%02d%02d" % (m // 60, m % 60, "-" if dec < 0 else "+", dm // 60, dm % 60)


def _sep_arcmin(a, b):
    r1, d1, r2, d2 = (math.radians(float(x)) for x in (a[0], a[1], b[0], b[1]))
    s = math.sin((d2 - d1) / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(s)))) * 60.0


def planet_position(body, ts):
    """Topocentric RA/Dec (deg) of a planet at the frame time, quantised to 10 minutes."""
    key = (body, ts[:12])
    if key not in _EPH:
        from astropy.time import Time
        from astropy.coordinates import EarthLocation, get_body, SkyCoord
        import astropy.units as u
        loc = EarthLocation(lat=37.2398 * u.deg, lon=-118.2817 * u.deg, height=1183.0 * u.m)
        t = Time("%s-%s-%sT%s:%s:00" % (ts[0:4], ts[4:6], ts[6:8], ts[9:11], ts[11:13]), format="isot", scale="utc")
        b = get_body(body, t, loc); c = SkyCoord(b.ra, b.dec)       # parallax gotcha: flatten first
        _EPH[key] = (float(c.ra.deg), float(c.dec.deg))
    return _EPH[key]


def safe_name(nm):
    return "".join(ch if (ch.isalnum() or ch in "+-._") else "_" for ch in nm.strip()).strip("_")


def _elevation(ra, dec, lst_deg, lat=37.2398):
    ha = math.radians(ra - lst_deg)
    se = math.sin(math.radians(dec)) * math.sin(math.radians(lat)) + math.cos(math.radians(dec)) * math.cos(math.radians(lat)) * math.cos(ha)
    return math.degrees(math.asin(max(-1, min(1, se))))


def write_cutouts(cube, model, meta, admitted, hdr, screen_x, screen_y, lst, date, band, stokes,
                  noise_rows=None, root=ROOT_DEFAULT, cut_deg=CUT_DEG):
    """Cut every target from the residual cube and write the per-object cubes.  Returns the
    number of cubes written.  Frames not admitted are still cut (flagged in TIMES)."""
    # REFUSED in an injected run (2026-10-24, INCIDENT-INJECTED-SEARCH-WROTE-PRODUCTION-PHOTOMETRY-CUTOUTS-HS-PSC-EVENT-
    # UNREPRODUCIBLE-20261024): a cube carrying synthetic sources never reaches the photometry product
    if os.environ.get("P3_SNAP_INJECTED") == "1":
        raise RuntimeError("REFUSED: write_cutouts in an injected run (P3_SNAP_INJECTED=1)")
    from phase3_snapshot_search import lst_deg_at, ts_to_epoch_seconds
    if not len(meta):
        return 0
    w = WCS(hdr).celestial
    cd = abs(float(hdr["CDELT2"])); n = int(round(cut_deg / cd)); half = n // 2
    nt, ny, nx = cube.shape
    times = [ts_to_epoch_seconds(m["ts"]) for m in meta]
    lst_mid = lst_deg_at(0.5 * (min(times) + max(times)))
    pad = 2 if screen_x is None else int(math.ceil(float(np.nanmax(np.abs(screen_x))))) + int(math.ceil(float(np.nanmax(np.abs(screen_y))))) + 2
    rms = {}
    for r in (noise_rows or []):
        if r.get("global_rms_jyb"):
            rms[r["ts"]] = float(r["global_rms_jyb"])
    # fixed targets
    jobs = []
    for tag, nm, ra, dec in _targets():
        if _elevation(ra, dec, lst_mid) < MIN_EL_DEG:
            continue
        px, py = w.all_world2pix(ra, dec, 0)
        if not (np.isfinite(px) and np.isfinite(py)) or not (0 <= px < nx and 0 <= py < ny):
            continue
        jobs.append((tag, nm, [(ra, dec)] * nt))
    # planets: per-frame positions (none under an ad-hoc target set unless P3_SNAP_CUTOUT_PLANETS=1 - _planets)
    for body in _planets():
        pos = [planet_position(body, m["ts"]) for m in meta]
        ra0, dec0 = pos[nt // 2]
        if _elevation(ra0, dec0, lst_mid) < MIN_EL_DEG:
            continue
        px, py = w.all_world2pix(ra0, dec0, 0)
        if not (np.isfinite(px) and np.isfinite(py)) or not (0 <= px < nx and 0 <= py < ny):
            continue
        jobs.append(("SolarSystem", body.capitalize(), pos))
    if not jobs:
        return 0
    ut0 = meta[0]["ts"]; utc = "%sT%s" % (ut0[:8], ut0[9:15])
    written = 0
    for tag, nm, pos in jobs:
        out = np.full((nt, n, n), np.nan, dtype=np.float32)
        cx_list, cy_list = [], []
        for t in range(nt):
            ra, dec = pos[t]
            px, py = w.all_world2pix(ra, dec, 0); px, py = float(px), float(py)
            if not (np.isfinite(px) and np.isfinite(py)):
                cx_list.append(np.nan); cy_list.append(np.nan); continue
            x0, y0 = int(round(px)) - half, int(round(py)) - half
            cx_list.append(px - x0); cy_list.append(py - y0)
            xa, xb, ya, yb = x0 - pad, x0 + n + pad, y0 - pad, y0 + n + pad
            xa_, xb_, ya_, yb_ = max(0, xa), min(nx, xb), max(0, ya), min(ny, yb)
            if xb_ <= xa_ or yb_ <= ya_:
                continue
            raw = cube[t, ya_:yb_, xa_:xb_] - model[ya_:yb_, xa_:xb_]
            if screen_x is not None:
                gy, gx = np.mgrid[ya_:yb_, xa_:xb_]
                dew = ndimage.map_coordinates(raw, [gy - ya_ + screen_y[ya_:yb_, xa_:xb_], gx - xa_ + screen_x[ya_:yb_, xa_:xb_]], order=1, mode="constant", cval=np.nan)
            else:
                dew = raw
            # place into the n x n box (WCS grid coordinates)
            oy0, ox0 = y0 - ya_, x0 - xa_
            sy0, sx0 = max(0, -oy0), max(0, -ox0)
            ty0, tx0 = max(0, oy0), max(0, ox0)
            h_ = min(n - sy0, dew.shape[0] - ty0); w_ = min(n - sx0, dew.shape[1] - tx0)
            if h_ > 0 and w_ > 0:
                out[t, sy0:sy0 + h_, sx0:sx0 + w_] = dew[ty0:ty0 + h_, tx0:tx0 + w_]
        # header: cutout WCS = frame WCS with CRPIX shifted to the FIRST frame's box (moving targets carry per-frame RA/Dec)
        ra0, dec0 = pos[0]
        px0, py0 = w.all_world2pix(ra0, dec0, 0); x00, y00 = int(round(float(px0))) - half, int(round(float(py0))) - half
        h = w.to_header()
        h["CRPIX1"] = float(h["CRPIX1"]) - x00; h["CRPIX2"] = float(h["CRPIX2"]) - y00
        for k in ("BMAJ", "BMIN", "BPA"):
            if k in hdr and np.isfinite(hdr[k]):
                h[k] = float(hdr[k])
        h["BUNIT"] = "Jy/beam"; h["OBJECT"] = nm; h["SAMPLE"] = tag; h["LST"] = lst; h["DATE"] = date
        h["BAND"] = band; h["STOKES"] = stokes; h["CELL"] = "%s_%s" % (lst, date); h["UTCSTART"] = utc
        h["TARGRA"] = (ra0, "deg, target position (first frame for moving targets)"); h["TARGDEC"] = (dec0, "deg")
        h["CUTDEG"] = (cut_deg, "cutout size, deg"); h["TIER"] = "10s"; h["NFRAMES"] = nt
        h["MODEL"] = ("hour-median subtracted, dewarped locally", "residual = frame - hour median")
        h["PRODUCT"] = "snapshot photometry cutout (Gregg 2026-09-11)"; h["DATE-CRE"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        cols = [fits.Column(name="ts", format="15A", array=np.array([m["ts"] for m in meta])),
                fits.Column(name="date_obs", format="26A", array=np.array([m.get("date_obs", "") for m in meta])),
                fits.Column(name="admitted", format="L", array=np.array([bool(a) for a in admitted])),
                fits.Column(name="global_rms_jyb", format="E", array=np.array([rms.get(m["ts"], np.nan) for m in meta], dtype=np.float32)),
                fits.Column(name="ra_deg", format="D", array=np.array([p[0] for p in pos])),
                fits.Column(name="dec_deg", format="D", array=np.array([p[1] for p in pos])),
                fits.Column(name="target_x", format="E", array=np.array(cx_list, dtype=np.float32)),
                fits.Column(name="target_y", format="E", array=np.array(cy_list, dtype=np.float32))]
        tab = fits.BinTableHDU.from_columns(cols, name="TIMES")
        d = os.path.join(root, tag, band, safe_name(nm)); os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, "%s_%s_%s_%s.fits" % (safe_name(nm), lst, utc.replace(":", ""), stokes))
        fits.HDUList([fits.PrimaryHDU(out, header=h), tab]).writeto(fn, overwrite=True)
        written += 1
    log.info("photometry cutouts: %d objects x %d frames, %s %s %s -> %s (names: distinct objects unique, 2026-10-25)", written, nt, lst,
             band, stokes, root)
    return written
