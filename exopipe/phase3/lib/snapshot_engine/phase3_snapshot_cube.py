#!/usr/bin/env python
"""Snapshot FLOWED-FRAME CUBES: one compressed FITS file per (cell, subband, Stokes) instead of one file per
10-second frame (Gregg 2026-09-19: "join the FITS images into a cube ... and compress that instead of the individual
images. You could apply the flow as part of the process"; design CARD_SNAPSHOT_CUBE_DESIGN_20260919.md; Gregg
2026-09-23 "Create the cubes").

WHAT IS STORED - the frames the search actually searched, i.e. the fleet's full-resolution-flow registered frames
(<base>_flowed.fits in the node-local staging), not a re-registration.  The full-resolution flow field is larger
than a compressed frame (1508 x 1508 x 2 float16 = 9 MB), so a raw-frame cube could only ever be re-registered
approximately from the shipped 1/8-grid field; storing the flowed frames keeps the searched data exact and the
1/8-grid field rides along so the raw frame stays recoverable to interpolation accuracy.

LAYOUT of <cell>_<subband>_<Stokes>_flowcube.fits:
  [0] PRIMARY   header only: cell, subband, Stokes, n frames, quantisation, code commit, provenance
  [1] FRAMES    RICE_1 compressed float32 cube (n, ny, nx), one tile per frame, quantize level Q (default 8,
                measured 2026-09-19 on 60 real frames: noise ratio 0.9998, worst pixel 2 per cent of the noise, peak
                flux ratio 1.00000); NaN pixels preserved.  WCS = the Master grid (every flowed frame shares it).
  [2] FLOW      int16 (n, gy, gx, 2), step FLOWSCAL px (1e-3 unless the cube's largest |flow| needs more): the per-frame 1/8-grid flow field (dx, dy) the fleet shipped
                (quantisation 0.0005 px, finer than the float16 it came from); frames without their own field (a blank
                Stokes I instant; a subband with no Master) carry the BLANK value and say why in FRAME_TABLE.
  [2b] MASK     int16 (n, ny, nx) RICE (lossless): 1 where the frame pixel was NaN; FRAMES holds 0 there.
  [3] FRAME_TABLE  one row per frame: ts, flowed (bool), flow_note, has_flow, blank, source_file, source_bytes,
                source_md5 (of the ORIGINAL raw frame file, so a later replacement of the archive frames can be verified
                against it), header (the frame's own complete FITS header as text - time block, beam, FLOWREF).

A frame's own header is restored exactly by the reader, so every consumer sees the per-frame time and beam.

Subcommands
  write   --stage <staged subband dir> --cell --subband --stokes {I,V} --flows <dir of flow_<sb>_<ts>.npy> --out <dir>
          [--stamp [--noise-dir D]]              --stamp (opt-in, default OFF): the standard header stamp (stamp_cards)
  verify  <cube> --stage <staged subband dir>    re-read every frame and compare with the flowed FITS it came from
  explode <cube> --out <dir>                     write every frame back out as <base>_flowed.fits (test harness)
  rescale <cube> --fluxscal F --out <dir> [--noise-dir D]   Master units + the standard header stamp, gated by the
                                                 Release 1 acceptance cube check
Python API: open_cube(path) -> Cube; Cube.frame(ts) -> (data float32, astropy Header); Cube.timestamps
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import warnings
from collections import OrderedDict

import numpy as np

QUANT = 8.0
FLOW_SCALE = 1e-3            # px per int16 unit
FLOW_BLANK = -32768
PAT = re.compile(r"(?P<sb>\d+MHz)-(?P<kind>Clean|Pilot)-Snapshot-(?P<ts>\d{8}_\d{6})(?:-(?P<st>[IV]))?-image\.fits$")


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse",
                                        "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def cube_name(cell, sb, st):
    return "%s_%s_%s_flowcube.fits" % (cell, sb, st)


def raw_frames(stage, st):
    """{ts: raw staged path} for one Stokes, both naming eras (CLEANed: Clean=I, Pilot=V; pilot era: -I-/-V-)."""
    out = {}
    for f in os.listdir(stage):
        m = PAT.match(f)
        if not m:
            continue
        if m.group("st"):
            s = m.group("st")
        else:
            s = "I" if m.group("kind") == "Clean" else "V"
        if s == st:
            out[m.group("ts")] = os.path.join(stage, f)
    return dict(sorted(out.items()))


def md5_file(path, block=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def comp_hdu(fits, cube, header, quant):
    """RICE_1, one tile per frame, SUBTRACTIVE_DITHER_1, NaN kept as ZBLANK nulls.  WRITE WITH ASTROPY >= 5.3 (the
    py310_orca env): astropy 4.1 (development env) wrote no ZBLANK and no dither - NaN pixels came back as -2.6e8
    (caught by verify on the synthetic test, 2026-09-23).  Refuses to write under an older astropy."""
    import astropy
    major, minor = (int(x) for x in astropy.__version__.split(".")[:2])
    if (major, minor) < (5, 3):
        sys.exit("REFUSED: astropy %s cannot write NaN-safe compressed float cubes; run the writer under "
                 "/opt/devel/pipeline/envs/py310_orca/bin/python" % astropy.__version__)
    # DITHER SEED FROM THE DATA (2026-11-12, R-GREGG-20261112-CUBE-DITHER-SEED-FROM-THE-DATA): astropy's default takes the
    # seed of the subtractive dither from the clock, so two writes of the same frames gave two cubes whose pixels differed
    # at 1-2% of the noise, and no cube could be proven by writing it a second time. DITHER_SEED_CHECKSUM takes the seed
    # from the bytes of the first tile (recorded in ZDITHER0, as before): same pixels in, same stored pixels out. The noise
    # the quantization adds is unchanged; only which realization a cube holds. The stored BYTES are the same too, except
    # in a tile that cannot be quantized (a blank instant, all zeros): astropy stores it losslessly through Python's gzip,
    # which writes the time of day into its 10-byte header (astropy 6.1.7, measured 2026-11-12) - so a proof by a second
    # write compares the decoded pixels, or the bytes apart from those gzip headers. astropy.io.fits itself does not
    # export the constant (6.1.7): it lives in astropy.io.fits.hdu.compressed (a module in 5.3, a package since 6.0)
    from astropy.io.fits.hdu.compressed import DITHER_SEED_CHECKSUM
    return fits.CompImageHDU(data=cube, header=header, name="FRAMES", compression_type="RICE_1",
                             tile_shape=(1,) + cube.shape[1:], quantize_level=quant,
                             quantize_method=1,       # 1 = SUBTRACTIVE_DITHER_1
                             dither_seed=DITHER_SEED_CHECKSUM)


# ---------------------------------------------------------------- THE STANDARD HEADER STAMP of a cube's PRIMARY header
# Spec exopipe_ops/snapshot_batch/CUBE_STAMP_SPEC_20261110.md (Gregg 2026-11-10 "all approved", item 6 of
# R-GREGG-20261110-HEADER-REPAIR-SIX-DECISIONS), done INSIDE the rewrite of the cube (rescale; write --stamp) instead of as a
# separate in-place header rewrite of archived files.  The acceptance test is exopipe_ops/release1_acceptance.py header_check
# (class snapshot_cubes: the 7-item standard on the primary, P2RUN not applicable, time block consistent to 1 s).  Cards are
# ADDED; an existing card is never changed (CHECKSUM/DATASUM are recomputed by writeto(checksum=True), the spec's one
# exception).  A card without a source is LEFT OUT, printed (STAMP-LEFT-OUT) and listed in HISTORY as '<CARD> not written:
# <reason>' - never a placeholder (Gregg's partial policy, decision 3).  Every stamped card prints its value and source (STAMP).
# rescale refuses unless the acceptance check reads complete7 = 1 (a sourced card missing, a card both listed and present, or
# an inconsistent time block refuses).
STAMP_SPEC = "CUBE_STAMP_SPEC_20261110"
ACCEPTANCE_PY = "/lustre/gh/main/exopipe_ops/release1_acceptance.py"
EXOPIPE_MAIN = "/lustre/gh/main/exopipe"      # production package: phase2_config.SUBBAND_FREQ_HZ (exact frequencies)
LIGHTCURVE_ROOT = os.environ.get("P3_SNAP_LC_ROOT", "/lustre/gh/main/exopipe_ops/snapshot_lightcurves")  # = snapshot_fluxscal.LC_ROOT
NOISE_HALF_MAX = 512                          # central box half-size min(512, N/4) (header_repair_release1 EPOCH NOISE)
ARCHIVE_ROOT = "/lustre/pipeline/exopipe"     # HISTORY names the Masters by their path relative to it
GEN4_ROOT = "/lustre/pipeline/exopipe/phase3/snapshot_masters_gen4"     # the frozen generation-4 Masters (2026-10-04)
FLEET_MASTERS = "/lustre/pipeline/exopipe/phase3/snapshot_masters"      # the fleet's / cube pass's --reference root (links)
MASTER_CATALOGS = "/lustre/gh/main/exopipe_ops/snapshot_master_catalogs"  # light-curve MASTCAT: master_R075_<LST>_<sb>.gaul.csv
FRAME_INT_CARD = "FRAMEINT"                   # per-frame integration time (not a card of products.py; INTTIME is the sum)


def _snap_search():
    """clv15 phase3_snapshot_search (INTEGRATION_S, dateobs_unix, frame_root, phase1_run_pin, SEARCH_OUT_ROOT), on first use."""
    for d in (os.path.dirname(os.path.abspath(__file__)), "/lustre/gh/main/clv15"):
        if os.path.exists(os.path.join(d, "phase3_snapshot_search.py")):
            if d not in sys.path:
                sys.path.append(d)
            break
    import phase3_snapshot_search
    return phase3_snapshot_search


def _cfg():
    """production exopipe.common.phase2_config (cfg.SUBBAND_FREQ_HZ), as phase3_snapshot_detrecord imports it."""
    if EXOPIPE_MAIN not in sys.path:
        sys.path.insert(0, EXOPIPE_MAIN)
    from exopipe.common import phase2_config
    return phase2_config


def central_rms(img):
    """The header standard's central rms: mad_std of the central box (half = min(512, N/4)), NaN ignored, float64, rounded to 9
    decimals (header_repair_release1.central_box_rms; phase2 get_inner_rms).  NaN for a blank box."""
    from astropy.stats import mad_std
    ny, nx = img.shape[-2:]
    half = min(NOISE_HALF_MAX, ny // 4, nx // 4)
    box = np.asarray(img[ny // 2 - half:ny // 2 + half, nx // 2 - half:nx // 2 + half], dtype=np.float64)
    v = float(mad_std(box, ignore_nan=True))
    return round(v, 9) if np.isfinite(v) else float("nan")


def admitted_frames(cell, sb, st, noise_dir=None):
    """(set of admitted ts, source) from the search's own noise table noise_<sb>_<S>_median.csv ('admitted' = 1, after its
    noise-outlier and flood refusals), or (None, reason)."""
    d = noise_dir or os.path.join(_snap_search().SEARCH_OUT_ROOT, cell)
    p = os.path.join(d, "noise_%s_%s_median.csv" % (sb, st))
    if not os.path.exists(p):
        return None, "no search noise table %s (admitted frames unknown)" % p
    with open(p, newline="") as fh:
        return {r["ts"].strip() for r in csv.DictReader(fh) if str(r.get("admitted", "")).strip() == "1"}, p


def noise_median(tss, admitted, frame_of):
    """(median central rms over the cube's admitted frames, n) - frame_of(i) = frame i in the cube's own units, NaN restored."""
    v = [central_rms(frame_of(i)) for i, ts in enumerate(tss) if ts in admitted]
    v = [x for x in v if np.isfinite(x) and x > 0]
    return (float(np.median(v)), len(v)) if v else (None, 0)


def cube_file_noise(path, admitted, fs):
    """(median central rms, n, source) of an archived cube's admitted frames in Master units, or (None, 0, reason).  FRAMES and
    MASK one tile per frame, memmap=False (whole-HDU fallback past 2 GiB, as Cube.frame); a frame-unit cube is divided by fs, a
    scaled one must carry FSAPPLD = fs."""
    from astropy.io import fits
    if not os.path.exists(path):
        return None, 0, "no cube %s" % path
    with fits.open(path, memmap=False) as h:
        div = fs
        if "FSAPPLD" in h[0].header:
            if fs is None or abs(float(h[0].header["FSAPPLD"]) - fs) > 2e-6:
                return None, 0, "%s carries FSAPPLD=%s, this cube %s" % (path, h[0].header["FSAPPLD"], fs)
            div = None
        t = h["FRAME_TABLE"].data
        tss = [s.strip() if isinstance(s, str) else s.decode().strip() for s in t["ts"]]
        fr, mk, keep = h["FRAMES"], h["MASK"], {}

        def frame_of(i):
            try:
                d = np.array(fr.section[i], dtype=np.float32)
            except (AttributeError, OverflowError):
                if "d" not in keep:
                    keep["d"] = np.asarray(fr.data, dtype=np.float32)
                d = np.array(keep["d"][i])
            try:
                m = np.asarray(mk.section[i]) != 0
            except (AttributeError, OverflowError):
                if "m" not in keep:
                    keep["m"] = np.asarray(mk.data) != 0
                m = keep["m"][i]
            d = d / np.float32(div) if div is not None else d
            d[m] = np.nan
            return d
        v, n = noise_median(tss, admitted, frame_of)
    return (v, n, "the Stokes V cube %s%s" % (path, "" if div is None else " (frame units / %.6f)" % div)) if v else \
        (None, 0, "no admitted frame of %s with a finite central rms" % path)


def staged_noise(stage, st, admitted, fs):
    """(median central rms, n, source) of the staged frames of one Stokes as write() reads them (flowed copy, else the raw frame
    of a .no_master subband) / fs, over the admitted frames, or (None, 0, reason)."""
    from astropy.io import fits
    no_master = os.path.exists(os.path.join(stage, ".no_master"))
    raws = raw_frames(stage, st)
    tss, srcs = [], []
    for ts, raw in raws.items():
        fl = raw[:-5] + "_flowed.fits"
        if os.path.exists(fl) or no_master:
            tss.append(ts); srcs.append(fl if os.path.exists(fl) else raw)

    def frame_of(i):
        with fits.open(srcs[i], memmap=False) as h:
            d = np.asarray(h[0].data, dtype=np.float32)
        d = d.reshape(d.shape[-2:])
        return d / np.float32(fs) if fs is not None else d
    v, n = noise_median(tss, admitted, frame_of)
    return (v, n, "the staged Stokes %s frames %s" % (st, stage)) if v else (None, 0, "no admitted staged %s frame" % st)


def noise_cards(cell, sb, st, tss, frame_of, v_noise=None, noise_dir=None):
    """{'IMGRMS': (value, comment, source) | (None, None, reason), 'THERMNOI': ...}: cube-level medians over the frames the
    search admitted (spec: per-frame values stay in the search's noise table).  IMGRMS = this cube's own central rms; THERMNOI =
    the Stokes V central rms - a V cube's own, an I cube's from v_noise(admitted V ts) -> (value, n, source) of the V frames of
    the same cell and subband."""
    out = {}
    adm, src = admitted_frames(cell, sb, st, noise_dir)
    if adm is None:
        out["IMGRMS"] = out["THERMNOI"] = (None, None, src)
    else:
        v, n = noise_median(tss, adm, frame_of)
        s = "central rms of this cube's frames, median over the %d frames admitted in %s" % (n, src)
        out["IMGRMS"] = (v, "[Jy/beam] median of %d admitted frames" % n, s) if v \
            else (None, None, "no admitted frame with a finite central rms (%s)" % src)
        if st == "V":
            out["THERMNOI"] = (v, "[Jy/beam] V rms, median of %d admitted frames" % n, s) if v else out["IMGRMS"]
    if st != "V" and v_noise is None:
        out["THERMNOI"] = (None, None, "no Stokes V source given for the I cube")
    elif st != "V":
        adv, srcv = admitted_frames(cell, sb, "V", noise_dir)
        if adv is None:
            out["THERMNOI"] = (None, None, srcv)
        else:
            v, n, s = v_noise(adv)
            out["THERMNOI"] = (v, "[Jy/beam] V rms, median of %d admitted frames" % n,
                               "central rms of %s, median over the %d V frames admitted in %s" % (s, n, srcv)) if v \
                else (None, None, s)
    return out


def _group_values(pairs):
    """[(key, value)] -> 'value' when one distinct value, else 'v1:k,k;v2:k' (header_repair_release1.group_values)."""
    order = OrderedDict()
    for k, v in pairs:
        order.setdefault(v, []).append(k)
    return next(iter(order)) if len(order) == 1 else ";".join("%s:%s" % (v, ",".join(ks)) for v, ks in order.items())


def _same_frame(path, rec):
    """True when the archive frame's own header carries the recorded frame header's identifying cards (DATE-OBS, the beam, and
    WSClean's WSC* cards - visibility counts and weight sums differ between two Phase 1 runs of one integration); False when
    unreadable or when the recorded header has none of them."""
    from astropy.io import fits
    keys = [k for k in rec if k == "DATE-OBS" or k.startswith("WSC") or k in ("BMAJ", "BMIN", "BPA")]
    try:
        with fits.open(path, memmap=False) as h:
            hh = h[1].header if len(h) > 1 else h[0].header     # .fz/.fs: the image header of the compressed extension
            return bool(keys) and all(k in hh and str(hh[k]).strip() == str(rec[k]).strip() for k in keys)
    except Exception:
        return False


def phase1_cards(cell, sb, source_files, headers=None):
    """{'P1RUN': (value, comment, source) | (None, None, reason), 'CALBP': ...} of the frames the cube holds.  Each FRAME_TABLE
    source_file (the staged, decompressed name) is located in the cell's frame root (phase3_snapshot_search.frame_root: the
    exopipe Phase 1 tree, or the GPU pipeline's for GPU-frame cells - Run_GSIbacklog_v1) under Run_*/<sb>/snapshots{,_clean}/
    (archive names .fz/.fs).  A name held by several runs (duplicate-run cells) resolves to the cell's pinned run
    (phase1_run_pin), else to the one run whose archive frame's own header matches the recorded frame header (headers: the
    FRAME_TABLE header column; _same_frame), else both cards are left out (the cube does not record which).  P1RUN = the run; frames of several runs 'Run_a:n;Run_b:m' (frame counts).  CALBP =
    bp_table of <run>/<sb>/provenance.json, whose subband, obs_date and lst_label must match (header_repair_release1.phase1_bp);
    a run without provenance.json (Phase 1 runs before 2026-03-11) leaves CALBP out (decision 2 of the six)."""
    S = _snap_search()
    lst, date = cell.split("_", 1)
    night = os.path.join(S.frame_root(lst, date), lst, date)
    where = {}
    for rd in sorted(glob.glob(os.path.join(night, "Run_*"))):
        for sub in ("snapshots", "snapshots_clean"):
            d = os.path.join(rd, sb, sub)
            for fn in (os.listdir(d) if os.path.isdir(d) else []):
                base = fn[:-3] if fn.endswith((".fz", ".fs")) else fn
                where.setdefault(base, {})[os.path.basename(rd)] = os.path.join(d, fn)
    pin = S.phase1_run_pin(lst, date)
    count, miss, amb, by_header = OrderedDict(), [], [], 0
    for i, f in enumerate(source_files):
        rr = set(where.get(f, {}))
        if len(rr) > 1 and pin in rr:
            rr = {pin}
        elif len(rr) > 1 and headers is not None:
            hit = {r for r in rr if _same_frame(where[f][r], headers[i])}
            by_header += len(hit) == 1
            rr = hit if len(hit) == 1 else rr
        if not rr:
            miss.append(f)
        elif len(rr) > 1:
            amb.append("%s in %s" % (f, ",".join(sorted(rr))))
        else:
            r = next(iter(rr))
            count[r] = count.get(r, 0) + 1
    if miss or amb:
        why = "%d of %d frames not found under %s/Run_*/%s (e.g. %s); %d held by several runs (e.g. %s)" % (
            len(miss), len(source_files), night, sb, (miss or ["-"])[0], len(amb), (amb or ["-"])[0])
        return {"P1RUN": (None, None, why), "CALBP": (None, None, why)}
    runs = sorted(count)
    out = {"P1RUN": (runs[0] if len(runs) == 1 else ";".join("%s:%d" % (r, count[r]) for r in runs),
                     "Phase 1 run of the frames" if len(runs) == 1 else "Phase 1 runs of the frames (run:frames)",
                     "all %d FRAME_TABLE source_file names located under %s/<run>/%s/snapshots{,_clean}%s%s"
                     % (len(source_files), night, sb, " (pin %s)" % pin if pin else "",
                        "; %d names held by several runs resolved by the frame's own header" % by_header if by_header else ""))}
    bps, bad, notes = [], [], []
    for r in runs:
        pj = os.path.join(night, r, sb, "provenance.json")
        try:
            d = json.load(open(pj))
        except FileNotFoundError:
            bad.append("the Phase 1 run directory %s has no provenance.json (no bandpass record)" % os.path.dirname(pj))
            continue
        except Exception as ex:
            bad.append("%s unreadable (%s)" % (pj, str(ex)[:60]))
            continue
        mism = [k for k, v in (("subband", sb), ("obs_date", date), ("lst_label", lst)) if str(d.get(k, "")) != v]
        bp = d.get("bp_table")
        if mism:
            bad.append("%s: %s do not match this subband / night" % (pj, ",".join(mism)))
        elif not isinstance(bp, str) or not bp.strip():
            bad.append("%s: no bp_table" % pj)
        else:
            bps.append((r, bp.strip()))
            if str(d.get("run_label", "")) != r:
                notes.append("%s run_label %s" % (r, d.get("run_label")))
    out["CALBP"] = (None, None, "; ".join(bad[:3])) if bad else \
        (_group_values(bps), "Bandpass calibration table", "bp_table of %s/<run>/%s/provenance.json (%s)%s"
         % (night, sb, ",".join(r for r, _ in bps), "; " + "; ".join(notes) if notes else ""))
    return out


def _sha256(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def _catalog_master_md5(path):
    """the md5 a Master catalog's .gen.json records as master_md5_data: phase3_deep._load (squeeze, float64, non-finite and
    |x| > 1e6 zeroed) as float32 bytes (snapshot_parentcat_contract.build_one)."""
    from astropy.io import fits
    a = np.asarray(fits.getdata(path, memmap=False)).squeeze().astype(float)
    a[~np.isfinite(a)] = 0.0
    a[np.abs(a) > 1.0e6] = 0.0
    return hashlib.md5(np.ascontiguousarray(a.astype(np.float32)).tobytes()).hexdigest()


def _checksum_time(h):
    """unix time of a header's 'HDU checksum updated <time>' comment (astropy writes the node's local time; the lwacalim nodes
    run UTC), or None"""
    import calendar
    import time as _time
    m = re.search(r"updated (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", str(h.comments["CHECKSUM"]) if "CHECKSUM" in h else "")
    return calendar.timegm(_time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")) if m else None


def flux_master(cell, sb, fs):
    """(gen-4 Master path, its sha256, HISTORY note) of the Master the cube's flux scale was measured against, or (None, None,
    reason).  The factor (FSAPPLD) is the light-curve table's FLUXSCAL = median(row median / Master peak), the Master peaks from
    the catalog whose sha256 the table records (MCATSHA); that catalog's .gen.json names the Master it was built from and the md5
    of its data.  Verified: FLUXSCAL = fs, the catalog's sha256 = MCATSHA, the Master lies in the gen-4 tree and its data md5 =
    the recorded one."""
    from astropy.io import fits
    lst = cell.split("_")[0]
    lc = os.path.join(LIGHTCURVE_ROOT, cell, "%s_%s_I_lightcurves.fits" % (cell, sb))
    try:
        lh = fits.getheader(lc, 0, memmap=False)
    except OSError as ex:
        return None, None, "light-curve table %s unreadable (%s)" % (lc, str(ex)[:60])
    mcat = str(lh.get("MCATSHA", "")).strip()
    if abs(float(lh.get("FLUXSCAL", -1.0)) - fs) > 2e-6:
        return None, None, "%s FLUXSCAL %s is not the factor applied %s" % (lc, lh.get("FLUXSCAL"), fs)
    cat = os.path.join(MASTER_CATALOGS, "master_R075_%s_%s.gaul.csv" % (lst, sb))
    if not os.path.exists(cat) or _sha256(cat) != mcat:
        return None, None, "%s is not the catalog the scale was measured against (MCATSHA %s)" % (cat, mcat or "absent")
    try:
        gen = json.load(open(cat + ".gen.json"))
    except Exception as ex:
        return None, None, "%s.gen.json unreadable (%s)" % (cat, str(ex)[:60])
    m = str(gen.get("master", ""))
    if not m.startswith(GEN4_ROOT + "/") or not os.path.isfile(m):
        return None, None, "the scale catalog's Master %r is not a gen-4 Master on disk" % m
    if _catalog_master_md5(m) != gen.get("master_md5_data"):
        return None, None, "%s data md5 differs from the one its catalog recorded (%s)" % (m, gen.get("master_md5_data"))
    sha = _sha256(m)
    return m, sha, ("FLUXREF source (its ALIGNREF): the gen-4 Master %s sha256 %s, the Master of the light-curve scale catalog %s "
                    "(MCATSHA %s)"
                    % (os.path.relpath(m, ARCHIVE_ROOT), sha, os.path.basename(cat), mcat))


def flow_master(cell, sb, flowref, grid, hdrs):
    """(gen-4 Master path, its sha256, HISTORY note) of the Master the frames were flowed to, or (None, None, reason).  The fleet
    and the cube pass flow to --reference <FLEET_MASTERS>/<LST>_<sb>/coadd_final/I_deep_Taper_R0_<sb>.fits; those links point
    into GEN4_ROOT since the gen-4 freeze (their own mtime, 2026-10-04T04:58Z); the frames record only the basename (FLOWREF).
    Verified: the gen-4 file of that basename exists, the fleet link resolves to it, every frame was staged (its header's
    CHECKSUM time) after the link was set - a frame staged before it was flowed to the Master then linked, which is not
    recorded - and the cube's grid is the Master's."""
    from astropy.io import fits
    lst = cell.split("_")[0]
    g4 = os.path.join(GEN4_ROOT, "%s_%s" % (lst, sb), "coadd_final", flowref)
    link = os.path.join(FLEET_MASTERS, "%s_%s" % (lst, sb), "coadd_final", flowref)
    if not os.path.isfile(g4):
        return None, None, "no gen-4 Master %s" % g4
    if not os.path.islink(link) or os.path.realpath(link) != os.path.realpath(g4):
        return None, None, "the fleet's Master link %s does not resolve to %s" % (link, g4)
    t_link = os.lstat(link).st_mtime
    staged = [_checksum_time(h) for h in hdrs]
    if any(t is None for t in staged) or min(staged) < t_link:
        import time as _time
        return None, None, ("frames staged %s, before the fleet's Master link moved to gen-4 (%s) or undated: the frames were "
                            "flowed to the Master then linked, which is not recorded" % (
                                "undated" if any(t is None for t in staged) else
                                _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(min(staged))),
                                _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(t_link))))
    mh = fits.getheader(g4, 0, memmap=False)
    keys = ("NAXIS1", "NAXIS2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2")
    mism = [k for k in keys if k not in grid or k not in mh or not np.isclose(float(grid[k]), float(mh[k]), rtol=1e-9, atol=1e-9)]
    if mism:
        return None, None, "the cube's grid differs from %s in %s" % (g4, " ".join(mism))
    sha = _sha256(g4)
    return g4, sha, ("ASTREF source (its ALIGNREF): the gen-4 Master %s sha256 %s, the flow reference of the frames (FLOWREF)"
                     % (os.path.relpath(g4, ARCHIVE_ROOT), sha))


def stamp_cards(ph, frame_table, cell, sb, st, fs=None, noise=None, prodname=None, label="", admitted=None, grid=None):
    """Add the spec's cards to a cube's primary header ph (in place).  frame_table: the FRAME_TABLE rows (ts, source_file,
    header = each frame's own header).  fs: the flux factor divided out of the pixels (FSAPPLD) or None; noise: noise_cards();
    prodname: the cube's file name; admitted: admitted_frames() of this cube (INTTIME); grid: the FRAMES image header (the flow
    Master's grid check).  Every card left out is listed in HISTORY as '<CARD> not written: <reason>', one line each (the
    cutouts_job.py form, which release1_acceptance.header_check accepts since exopipe_ops 3f3c6ee).  Returns [(card, reason)]
    of the cards LEFT OUT (each also printed)."""
    from astropy.io import fits
    from astropy.time import Time, TimeDelta
    S = _snap_search()
    left, label = [], label or prodname or cell

    def put(key, value, comment, source):
        if key in ph:
            print("STAMP-KEPT %s %s = %r (existing card, never changed)" % (label, key, ph[key]))
        elif value is None:
            left.append((key, source))
            print("STAMP-LEFT-OUT %s %s: %s" % (label, key, source))
        else:
            with warnings.catch_warnings():       # a long value shortens the comment (astropy warns), never the value
                warnings.simplefilter("ignore", fits.verify.VerifyWarning)
                ph[key] = (value, comment)
            print("STAMP %s %s = %r | %s" % (label, key, value, source))

    hdrs = [fits.Header.fromstring(x if isinstance(x, str) else x.decode(), sep="\n") for x in frame_table["header"]]
    srcf = [(x if isinstance(x, str) else x.decode()).strip() for x in frame_table["source_file"]]
    # 1 BEAM, cube level: median BMAJ and BMIN, and the median BPA taken about the axial mean (0 and 180 deg are one direction),
    # over the frames carrying a beam (BMAJ > 0; a blank instant carries 0); per-frame beams stay in the frame headers
    bm = np.array([[float(h.get(k, np.nan)) for k in ("BMAJ", "BMIN", "BPA")] for h in hdrs], dtype=float)
    ok = np.all(np.isfinite(bm), axis=1) & (bm[:, 0] > 0) & (bm[:, 1] > 0)
    if ok.any():
        th = np.radians(bm[ok, 2])
        m = 0.5 * np.arctan2(np.mean(np.sin(2 * th)), np.mean(np.cos(2 * th)))
        pa = np.degrees(m) + float(np.median((np.degrees(th - m) + 90.0) % 180.0 - 90.0))
        pa = pa % 180.0 if (bm[ok, 2] >= 0).all() else (pa + 90.0) % 180.0 - 90.0
        src = "FRAME_TABLE header BMAJ/BMIN/BPA of the %d of %d frames carrying a beam" % (int(ok.sum()), len(hdrs))
        put("BMAJ", float(np.median(bm[ok, 0])), "[deg] median over %d frames" % ok.sum(), src)
        put("BMIN", float(np.median(bm[ok, 1])), "[deg] median over %d frames" % ok.sum(), src)
        put("BPA", float(pa), "[deg] axial median over %d frames" % ok.sum(), src)
    else:
        for k in ("BMAJ", "BMIN", "BPA"):
            put(k, None, None, "no frame carries a beam (BMAJ > 0)")
    # 3 FREQUENCY: exact correlator edges, cfg.SUBBAND_FREQ_HZ (band labels are nominal); STOKES exists (cube writer)
    try:
        cfg = _cfg()
        f = cfg.SUBBAND_FREQ_HZ[sb]
        src = "cfg.SUBBAND_FREQ_HZ[%r] (%s)" % (sb, cfg.__file__)
        put("RESTFRQ", float(f["center"]), "[Hz] Band center frequency", src)
        put("FREQ_MIN", float(f["lo"]), "[Hz] Lower band edge (exact correlator edge)", src)
        put("FREQ_MAX", float(f["hi"]), "[Hz] Upper band edge (exact correlator edge)", src)
        put("BANDWDTH", float(f["hi"] - f["lo"]), "[Hz] Total bandwidth", src)
    except Exception as ex:
        for k in ("RESTFRQ", "FREQ_MIN", "FREQ_MAX", "BANDWDTH"):
            put(k, None, None, "cfg.SUBBAND_FREQ_HZ[%r] unavailable (%s: %s)" % (sb, type(ex).__name__, str(ex)[:80]))
    put("STOKES", st, "Stokes parameter", "the cube's Stokes")
    # 4 TIME, the cube's OWN span: first frame start (the earliest frame DATE-OBS = its integration START) to last frame end (the
    # latest DATE-OBS + INTEGRATION_S, the MS INTERVAL); INTTIME = the per-frame integration time (spec), per-frame times stay in
    # the frame headers.  Any frame without a readable DATE-OBS leaves the whole block out (the span would not be exact).
    tcards = ("DATE-OBS", "MJD-OBS", "DATE-END", "MJD-END", "DURATION", "INTTIME", FRAME_INT_CARD, "MJD-MID", "TIMESYS",
              "TIMEFIX")
    try:
        dob = [str(h.get("DATE-OBS", "")).strip() for h in hdrs]
        u = [S.dateobs_unix(x) for x in dob]
        t0 = Time(dob[int(np.argmin(u))], format="isot", scale="utc")
        t1 = Time(dob[int(np.argmax(u))], format="isot", scale="utc") + TimeDelta(S.INTEGRATION_S, format="sec")
        src = ("frame DATE-OBS (FRAME_TABLE header) of %d frames: start = earliest, end = latest + INTEGRATION_S %.8f s "
               "(phase3_snapshot_search)" % (len(dob), S.INTEGRATION_S))
        put("DATE-OBS", t0.isot, "Start of first integration (UTC)", src)
        put("MJD-OBS", float(t0.mjd), "Start MJD (UTC)", src)
        put("DATE-END", t1.isot, "End of last integration (UTC)", src)
        put("MJD-END", float(t1.mjd), "End MJD (UTC)", src)
        put("DURATION", float((t1 - t0).sec), "[s] Wall-clock span (DATE-END - DATE-OBS)", src)
        # INTTIME = the standard's '[s] Summed integration time included': the cube's ADMITTED frames x INTEGRATION_S; the
        # per-frame integration time in its own card (FRAME_INT_CARD)
        adm, asrc = admitted if admitted is not None else (None, "admitted frames not given")
        if adm is None:
            put("INTTIME", None, None, asrc)
        else:
            tss = [(x if isinstance(x, str) else x.decode()).strip() for x in frame_table["ts"]]
            na = sum(1 for t in tss if t in adm)
            put("INTTIME", float(na * S.INTEGRATION_S), "[s] Summed integration of %d admitted frames" % na,
                "%d admitted frames of the cube (%s) x INTEGRATION_S" % (na, asrc))
        put(FRAME_INT_CARD, float(S.INTEGRATION_S), "[s] Integration time of one frame",
            "phase3_snapshot_search.INTEGRATION_S (MS INTERVAL)")
        put("MJD-MID", float((t0.mjd + t1.mjd) / 2.0), "Midpoint MJD (UTC)", src)
        put("TIMESYS", "UTC", "Time system", "frame DATE-OBS are UTC")
        put("TIMEFIX", "v2-framedateobs", "Time basis: frame DATE-OBS = integration start", src)
    except (ValueError, TypeError) as ex:
        for k in tcards:
            put(k, None, None, "a frame without a readable DATE-OBS (%s)" % str(ex)[:80])
    # 5 NOISE: cube-level medians over the admitted frames (noise_cards)
    for k in ("THERMNOI", "IMGRMS"):
        v, c, s = (noise or {}).get(k, (None, None, "not measured"))
        put(k, v, c, s)
    # 6 PROVENANCE (P2RUN does not apply to cubes)
    put("PRODNAME", prodname, "Product file name",
        "the cube's file name" if prodname else "no product name given")
    # FLUXREF / ASTREF: the build's recorded reference - the ALIGNREF card AS WRITTEN of the gen-4 Master the cube's scale
    # (FSAPPLD; FLUXREF) and flow (FLOWREF; ASTREF) refer to (Gregg's header-repair decision 1, R-GREGG-20261110-HEADER-REPAIR-
    # SIX-DECISIONS, applied to the cubes by the spec owner 2026-11-12); a Master without ALIGNREF leaves that card out.  HISTORY
    # names each Master (path, sha256) and the scale catalog (MCATSHA).
    notes = []

    def from_master(key, m, sha, note):
        if m is None:
            put(key, None, None, note)
            return
        notes.append(note)
        mh = fits.getheader(m, 0, memmap=False)
        rel = os.path.relpath(m, ARCHIVE_ROOT)
        ok = "ALIGNREF" in mh and str(mh["ALIGNREF"]).strip() != ""
        put(key, mh["ALIGNREF"] if ok else None,
            ("Flux scale" if key == "FLUXREF" else "Astrometric") + " reference: gen-4 Master ALIGNREF",
            "the ALIGNREF card as written of the gen-4 Master %s sha256 %s" % (rel, sha) if ok else
            "the gen-4 Master %s (sha256 %s) carries no ALIGNREF card" % (rel, sha))
    # the flow reference: the frames' own FLOWREF as recorded, unique; the primary FLOWREF agrees or is empty (cubes written
    # before the 2026-10-08 writer fix took it from frame 0, often a blank I instant naming none)
    refs = sorted({str(h.get("FLOWREF", "")).strip() for h in hdrs} - {""})
    fr = str(ph.get("FLOWREF", "")).strip()
    if len(refs) == 1 and fr in ("", refs[0]):
        flow = flow_master(cell, sb, refs[0], grid, hdrs) if grid is not None else (None, None, "the cube's grid header not given")
    else:
        flow = (None, None, "flow reference not unique as recorded: primary %r, frames %s" % (fr, refs[:3]))
    if fs is None:
        put("SCALE_A", None, None, "cube in frame units: no flux factor applied")
        put("FLUXREF", None, None, "cube in frame units: not flux-scaled to a Master")
    else:
        put("SCALE_A", float(fs), "Flux factor divided out of all frames",
            "the hour-band factor applied to all frames (FSAPPLD; snapshot_fluxscal.band_factors)")
        # FLUXREF independent of the flow side (spec owner 2026-11-12): the verified flux-side Master's ALIGNREF for every cube,
        # the cells staged before the 2026-10-04 link move included; ASTREF only where the flow side verifies
        from_master("FLUXREF", *flux_master(cell, sb, fs))
    from_master("ASTREF", *flow)
    for k, v in phase1_cards(cell, sb, srcf, hdrs).items():
        put(k, *v)
    # 7 GITTAG: the commit of this code
    g = git_hash()
    put("GITTAG", None if g == "unknown" else g, "commit of phase3_snapshot_cube.py (stamp)",
        "git rev-parse of %s" % os.path.dirname(os.path.abspath(__file__)) if g != "unknown" else "git commit unreadable")
    # 2 HISTORY: the stamp, the Masters FLUXREF / ASTREF take ALIGNREF from, and every card left out with its reason, one line each
    # in the cutouts_job.py form '<CARD> not written: <reason>' (';' never inside a reason: the tenmin form separates by it)
    import time as _time
    ph.add_history("Standard header stamped %s by phase3_snapshot_cube.py %s (%s): beam = frame medians, time = own span of "
                   "the frames, INTTIME = admitted frames, noise = medians over the search's admitted frames"
                   % (_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()), g, STAMP_SPEC))
    for n in notes:
        ph.add_history(n)
    for k, why in left:
        ph.add_history("%s not written: %s" % (k, " ".join(str(why).replace(";", ",").split())))
    print("STAMP %s HISTORY += stamp line | %s" % (label, STAMP_SPEC))
    return left


def acceptance_check(path, left=None):
    """(ok, detail): release1_acceptance.header_check of a cube's primary header (class snapshot_cubes) - THE acceptance test,
    run in a child interpreter: it puts the refactor's package first on sys.path, this module reads production's phase2_config.
    ok = complete7 = 1 (since exopipe_ops 3f3c6ee: every card present or listed in HISTORY as '<CARD> not written: <reason>',
    no card both listed and present, a consistent time block) and, given left (the stamp's left-out list), the checker read
    exactly those cards as left out."""
    code = ("import json, sys; sys.path.insert(0, %r); import release1_acceptance as RA; print('ACCEPTANCE-ROW ' + "
            "json.dumps(RA.header_check(('snapshot_cubes', 'snapshot cube (primary header)', sys.argv[1]))))"
            % os.path.dirname(ACCEPTANCE_PY))
    rc = subprocess.run([sys.executable, "-c", code, path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = rc.stdout.decode(errors="replace").splitlines()
    row = next((json.loads(l[len("ACCEPTANCE-ROW "):]) for l in lines if l.startswith("ACCEPTANCE-ROW ")), None)
    if row is None:
        return False, "the acceptance check did not run (exit %d): %s" % (rc.returncode, " | ".join(lines[-3:]))
    ok = row.get("complete7") == 1 and not row.get("unreadable")
    lo = set(str(row.get("left_out", "")).split())
    same = left is None or lo == {k for k, _ in left}
    return ok and same, "complete7=%s items=%s missing=[%s] left_out=[%s] time_bad=[%s]%s%s" % (
        row.get("complete7"), "".join(str(row.get(k, "?")) for k in sorted(row) if k[:1].isdigit()), row.get("missing", ""),
        row.get("left_out", ""), row.get("time_bad", ""), (" unreadable=%s" % row["unreadable"]) if row.get("unreadable") else "",
        "" if same else " - the checker's left-out list differs from the stamp's [%s]" % " ".join(k for k, _ in left))


def write(a):
    from astropy.io import fits
    raws = raw_frames(a.stage, a.stokes)
    if not raws:
        sys.exit("REFUSED: no %s frames under %s" % (a.stokes, a.stage))
    # FLUX SCALE INTO THE PIXELS (2026-11-12, R-GREGG-20261112-FLUXSCAL-NO-BSCALE-PIPELINE-WRITES-PIXELS-...: "No user will know to
    # check some obscure BSCALE factor"): --fluxscal F divides every stored pixel by the hour-band's frame-to-Master factor
    # (snapshot_fluxscal.band_factors) and records it as FSAPPLD; without it the cube is in frame units, as before
    fs = getattr(a, "fluxscal", None)
    if fs is not None and not (np.isfinite(fs) and fs > 0):
        sys.exit("REFUSED: --fluxscal %r is not a finite factor > 0" % fs)
    no_master = os.path.exists(os.path.join(a.stage, ".no_master"))
    # MEMORY (2026-11-12, memory only; the cube file is unchanged byte for byte): the frames go straight into ONE
    # preallocated cube and the NaN mask into one int16 array, frame by frame. Before, every frame was held twice (a list
    # and its stacked copy) beside two whole-cube masks (bool, then int16): 51.6 GB at the peak for a 14 GB cube of 358
    # frames of 3122 x 3122 px, against 35.3 GB now
    cube = nanmask16 = None
    rows, flows = [], []
    hdr0 = None
    flowref = ""        # the first frame that names its flow reference (frame 0 is often a blank I instant without one, 2026-10-08)
    for i, (ts, raw) in enumerate(raws.items()):
        fl = raw[:-5] + "_flowed.fits"
        if os.path.exists(fl):
            src, flowed = fl, True
        elif no_master:
            src, flowed = raw, False          # the search used this subband unflowed (hourly screen): stored as searched
        else:
            sys.exit("REFUSED: %s has no flowed copy and the subband is not marked .no_master - a cube never mixes "
                     "registered and unregistered frames silently" % raw)
        with fits.open(src, memmap=False) as h:
            d = np.asarray(h[0].data, dtype=np.float32)
            hh = h[0].header
        d = d.reshape(d.shape[-2:])
        if fs is not None:
            d = d / np.float32(fs)
        if hdr0 is None:
            hdr0 = hh.copy(); shape = d.shape
        if not flowref and str(hh.get("FLOWREF", "")).strip():
            flowref = str(hh.get("FLOWREF")).strip()
        if d.shape != shape:
            sys.exit("REFUSED: %s has shape %s, the cube's frames are %s" % (src, d.shape, shape))
        if cube is None:
            cube = np.empty((len(raws),) + shape, dtype=np.float32)
            nanmask16 = np.empty(cube.shape, dtype=np.int16)
        fin = np.isfinite(d)
        fp = os.path.join(a.flows, "flow_%s_%s.npy" % (a.subband, ts)) if a.flows else None
        if fp and os.path.exists(fp):
            f = np.load(fp).astype(np.float32)
            flows.append(f); has_flow = True
        else:
            flows.append(None); has_flow = False
        note = str(hh.get("FLOWED", "")) + " " + str(hh.comments["FLOWED"] if "FLOWED" in hh else "")
        blank = (not fin.any()) or (fin.mean() < 0.5)
        # NaN pixels (outside the projected sky; blank instants) are stored as 0 and restored from the MASK extension:
        # astropy 4.1, which the chain runs under, ignores ZBLANK when decompressing and returned -2.6e8 for every
        # NaN pixel of a cube written correctly by astropy 6.1 (2026-09-23 synthetic test), so the cube never relies
        # on the compression's own null handling
        np.logical_not(fin, out=fin)                      # now: True where the frame pixel is NaN or infinite
        cube[i] = d
        cube[i][fin] = 0.0
        nanmask16[i] = fin
        del d, fin
        rows.append((ts, flowed, note.strip()[:80], has_flow, blank, os.path.basename(raw), os.path.getsize(raw),
                     md5_file(raw), hh.tostring(sep="\n")))
    nframes = len(rows)
    gshape = next((f.shape for f in flows if f is not None), None)
    if gshape is not None:
        # storage step chosen PER CUBE: 1e-3 px unless the largest |flow| needs more room (the 2026-09-23 cube test
        # refused 73/78/82 MHz at 32.5-32.7 px under a fixed 1e-3 step); a NaN in a field is stored as FLOWNULL
        fmax = max(float(np.nanmax(np.abs(f))) for f in flows if f is not None and np.isfinite(f).any())
        scale = max(FLOW_SCALE, fmax / 32000.0)
        fl = np.full((len(flows),) + gshape, FLOW_BLANK, dtype=np.int16)
        for i, f in enumerate(flows):
            if f is not None:
                q = np.round(np.where(np.isfinite(f), f, 0.0) / scale).astype(np.int16)
                q[~np.isfinite(f)] = FLOW_BLANK
                fl[i] = q
    # primary
    ph = fits.Header()
    ph["CELL"] = (a.cell, "LST_date"); ph["SUBBAND"] = a.subband; ph["STOKES"] = a.stokes
    ph["NFRAMES"] = (nframes, "10-s frames, time order"); ph["QUANTLEV"] = (a.quant, "RICE quantize level")
    ph["FLOWED"] = (not no_master, "frames are the fleet's flowed (registered) frames")
    ph["NOMASTER"] = (no_master, "subband had no Master: frames stored as searched (unflowed)")
    ph["FLOWREF"] = (flowref[:60], "flow reference (multi-night master), first frame naming one")
    ph["CODE"] = (git_hash(), "clv15 commit of phase3_snapshot_cube.py")
    ph["PRODUCT"] = "snapshot-flowcube-v1"
    if fs is not None:
        ph["FSAPPLD"] = (float(fs), "frame-to-Master flux scale divided out of the pixels")
    ph.add_comment("Frames = the search's own registered frames; FLOW = 1/8-grid per-frame flow (dx, dy) px;")
    ph.add_comment("raw frame recoverable to interpolation accuracy; FRAME_TABLE header column = each frame's header.")
    # frames: WCS/celestial keys of the first frame, 3rd axis = frame index
    ch = fits.ImageHDU(data=cube).header.copy()      # astropy 4.x CompImageHDU needs BITPIX/NAXISn present
    for k in ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2", "CUNIT1", "CUNIT2",
              "PC1_1", "PC1_2", "PC2_1", "PC2_2", "LONPOLE", "LATPOLE", "RADESYS", "EQUINOX", "BUNIT"):
        if k in hdr0:
            ch[k] = hdr0[k]
    ch["CTYPE3"] = ("FRAMEIDX", "row of FRAME_TABLE"); ch["CRPIX3"] = 1; ch["CRVAL3"] = 0; ch["CDELT3"] = 1
    hdus = [fits.PrimaryHDU(header=ph),
            comp_hdu(fits, cube, ch, a.quant),
            fits.CompImageHDU(data=nanmask16, name="MASK", compression_type="RICE_1",
                              tile_shape=(1,) + cube.shape[1:])]      # 1 = pixel was NaN in the frame; lossless
    if gshape is not None:
        # NOT BSCALE/BLANK: astropy refuses to memory-map a scaled image, so the scale and the no-field value are
        # carried in keywords of their own and applied by Cube.flow()
        fh = fits.Header(); fh["FLOWSCAL"] = (scale, "px per stored unit (per cube)"); fh["FLOWMAX"] = (fmax, "largest |flow| px in this cube"); fh["FLOWNULL"] = (FLOW_BLANK, "no field for this frame")
        fh["FLOWUNIT"] = ("pixel", "dx, dy on the frame grid, sampled every LOWRES px")
        fh["LOWRES"] = (int(round(cube.shape[-1] / float(gshape[1]))), "grid step of the stored field (px)")
        hdus.append(fits.ImageHDU(data=fl, header=fh, name="FLOW"))
    cols = [fits.Column(name="ts", format="15A", array=[r[0] for r in rows]),
            fits.Column(name="flowed", format="L", array=[r[1] for r in rows]),
            fits.Column(name="flow_note", format="80A", array=[r[2] for r in rows]),
            fits.Column(name="has_flow", format="L", array=[r[3] for r in rows]),
            fits.Column(name="blank", format="L", array=[r[4] for r in rows]),
            fits.Column(name="source_file", format="80A", array=[r[5] for r in rows]),
            fits.Column(name="source_bytes", format="K", array=[r[6] for r in rows]),
            fits.Column(name="source_md5", format="32A", array=[r[7] for r in rows]),
            fits.Column(name="header", format="%dA" % max(len(r[8]) for r in rows), array=[r[8] for r in rows])]
    hdus.append(fits.BinTableHDU.from_columns(cols, name="FRAME_TABLE"))
    if getattr(a, "stamp", False):
        # THE STANDARD HEADER STAMP, opt-in (--stamp; default OFF so a running fleet's output is unchanged): the same cards as
        # rescale (stamp_cards); noise over the search's admitted frames, an I cube's THERMNOI from the staged V frames
        nz = noise_cards(a.cell, a.subband, a.stokes, [r[0] for r in rows],
                         lambda i: np.where(nanmask16[i] != 0, np.float32(np.nan), cube[i]),
                         (lambda adm: staged_noise(a.stage, "V", adm, fs)) if a.stokes == "I" else None,
                         getattr(a, "noise_dir", None))
        stamp_cards(hdus[0].header, hdus[-1].data, a.cell, a.subband, a.stokes, fs=fs, noise=nz,
                    prodname=cube_name(a.cell, a.subband, a.stokes),
                    admitted=admitted_frames(a.cell, a.subband, a.stokes, getattr(a, "noise_dir", None)), grid=ch)
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, cube_name(a.cell, a.subband, a.stokes))
    tmp = out + ".tmp"
    fits.HDUList(hdus).writeto(tmp, overwrite=True, checksum=True)
    os.replace(tmp, out)
    print("CUBE %s frames=%d flowed=%s flow_fields=%d size=%.1f MB (%.2f MB/frame)" % (
        out, nframes, not no_master, sum(r[3] for r in rows), os.path.getsize(out) / 1e6,
        os.path.getsize(out) / 1e6 / nframes))
    return out


class Cube(object):
    def __init__(self, path):
        from astropy.io import fits
        self.path = path
        self.h = fits.open(path, memmap=True)
        t = self.h["FRAME_TABLE"].data
        self.timestamps = [s.strip() if isinstance(s, str) else s.decode().strip() for s in t["ts"]]
        self._idx = {ts: i for i, ts in enumerate(self.timestamps)}
        self._t = t
        self.primary = self.h[0].header

    def frame(self, ts):
        from astropy.io import fits
        i = self._idx[ts]
        hdu = self.h["FRAMES"]
        try:
            d = np.asarray(hdu.section[i], dtype=np.float32)      # astropy >= 5.3: decompresses one tile (one frame)
        except (AttributeError, OverflowError):
            # OverflowError (2026-10-13): astropy 6.1 + numpy 2 add the tile's 32-bit heap offset to the HDU's file offset in
            # int32, so section[] fails for an HDU that starts past 2 GiB (05h 2024-12-27 41 MHz I: MASK at 2,277,648,000
            # bytes; the 10-minute few-frame stage refused); the whole-HDU decompression below does not overflow
            # astropy 4.x (the development env) has no section on compressed HDUs: the whole cube is decompressed ONCE
            # and kept for the life of this Cube - fine for sequential readers, costly for scattered single frames
            if not hasattr(self, "_all"):
                self._all = np.asarray(hdu.data, dtype=np.float32)
            d = self._all[i]
        d = np.array(d, dtype=np.float32)
        m = self.h["MASK"]
        try:
            mk = np.asarray(m.section[i]) != 0
        except (AttributeError, OverflowError):                  # OverflowError: see the FRAMES read above
            if not hasattr(self, "_allmask"):
                self._allmask = np.asarray(m.data) != 0
            mk = self._allmask[i]
        d[mk] = np.nan
        raw = self._t["header"][i]
        raw = raw if isinstance(raw, str) else raw.decode()
        return d, fits.Header.fromstring(raw, sep="\n")

    def flow(self, ts):
        if "FLOW" not in self.h:
            return None
        i = self._idx[ts]
        fh = self.h["FLOW"].header
        q = np.asarray(self.h["FLOW"].data[i])
        f = q.astype(np.float32) * float(fh.get("FLOWSCAL", FLOW_SCALE))
        f[q == int(fh.get("FLOWNULL", FLOW_BLANK))] = np.nan
        return None if np.isnan(f).all() else f

    def close(self):
        self.h.close()


def open_cube(path):
    return Cube(path)


def verify(a):
    """Every frame of the cube against the flowed FITS it was written from: finite-pixel masks identical, and the
    difference rms and worst pixel as fractions of the frame's own robust noise."""
    from astropy.io import fits
    c = open_cube(a.cube)
    raws = raw_frames(a.stage, c.primary["STOKES"])
    fsv = float(c.primary["FSAPPLD"]) if "FSAPPLD" in c.primary else None
    worst_rms, worst_pix, nbad, n = 0.0, 0.0, 0, 0
    for ts in c.timestamps:
        src = raws[ts][:-5] + "_flowed.fits" if c.primary["FLOWED"] else raws[ts]
        with fits.open(src, memmap=False) as h:
            d0 = np.asarray(h[0].data, dtype=np.float32).reshape(-1, h[0].data.shape[-1])
            h0 = h[0].header
        if fsv is not None:
            d0 = d0 / np.float32(fsv)     # a scaled cube (FSAPPLD) against its frame-unit source frames / the factor
        d1, h1 = c.frame(ts)
        m0, m1 = np.isfinite(d0), np.isfinite(d1)
        n += 1
        if (m0 != m1).any():
            nbad += 1; print("MASK DIFFERS", ts); continue
        if not m0.any():
            continue
        med = np.median(d0[m0]); mad = 1.4826 * np.median(np.abs(d0[m0] - med))
        if mad <= 0:
            continue
        diff = (d1 - d0)[m0]
        worst_rms = max(worst_rms, float(np.sqrt(np.mean(diff ** 2)) / mad))
        worst_pix = max(worst_pix, float(np.max(np.abs(diff)) / mad))
        for k in ("DATE-OBS", "BMAJ", "BMIN", "BPA", "FLOWREF"):
            if str(h0.get(k)) != str(h1.get(k)):
                nbad += 1; print("HEADER %s DIFFERS at %s: %s vs %s" % (k, ts, h0.get(k), h1.get(k)))
    c.close()
    ok = nbad == 0
    print("VERIFY %s frames=%d mismatches=%d worst diff rms/noise=%.4f worst pixel/noise=%.4f -> %s" % (
        os.path.basename(a.cube), n, nbad, worst_rms, worst_pix, "OK" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


def explode(a):
    """Test harness: every frame back out as the fleet's staging file (<raw base>_flowed.fits) plus a zero-length
    placeholder at the raw name so discover_frames(stage_root=...) resolves to the staging as it does in the fleet."""
    from astropy.io import fits
    c = open_cube(a.cube)
    os.makedirs(a.out, exist_ok=True)
    for i, ts in enumerate(c.timestamps):
        d, h = c.frame(ts)
        raw = c._t["source_file"][i]
        raw = (raw if isinstance(raw, str) else raw.decode()).strip()
        if c.primary.get("FLOWED"):
            fits.writeto(os.path.join(a.out, raw[:-5] + "_flowed.fits"), d, h, overwrite=True)
            open(os.path.join(a.out, raw), "a").close()
        else:          # a no-Master subband was searched on the raw frames themselves: rebuild them under the raw name
            fits.writeto(os.path.join(a.out, raw), d, h, overwrite=True)
    if c.primary.get("NOMASTER"):
        open(os.path.join(a.out, ".no_master"), "a").close()
    print("EXPLODED %s -> %s (%d frames)" % (os.path.basename(a.cube), a.out, len(c.timestamps)))
    c.close()


def rescale(a):
    """Rewrite an archived frame-unit cube in Master units (2026-11-12, R-GREGG-20261112-FLUXSCAL-NO-BSCALE-PIPELINE-WRITES-PIXELS-...:
    "existing cubes rewritten", new files, no BSCALE): FRAMES / F re-compressed as write() does (same tiling, the cube's own
    QUANTLEV, dither); MASK, FLOW and FRAME_TABLE carried over unchanged; the primary gains FSAPPLD and THE STANDARD HEADER
    STAMP (stamp_cards, CUBE_STAMP_SPEC_20261110 - done in this rewrite, not as a separate in-place header rewrite).  Writes
    <out>/<same name>, never over the source; REFUSES a cube already carrying FSAPPLD.  Content test before the rename: masks and
    FRAME_TABLE identical, every frame's |new - old / F| at most RESCALE_TOL of that frame's robust noise / F, and the Release 1
    acceptance cube check (release1_acceptance.header_check) passes on the new primary - else the temporary file is removed."""
    from astropy.io import fits
    fs = a.fluxscal
    if not (np.isfinite(fs) and fs > 0):
        sys.exit("REFUSED: --fluxscal %r is not a finite factor > 0" % fs)
    with fits.open(a.cube, memmap=False) as h:
        ph = h[0].header.copy()
        if "FSAPPLD" in ph:
            sys.exit("REFUSED: %s already carries FSAPPLD=%s" % (a.cube, ph["FSAPPLD"]))
        new = np.asarray(h["FRAMES"].data, dtype=np.float32)
        ch = h["FRAMES"].header.copy()
        mask = np.asarray(np.asarray(h["MASK"].data), dtype=np.int16)      # int16 as stored: no copy
        rest = [fits.ImageHDU(data=np.array(h["FLOW"].data), header=h["FLOW"].header.copy(), name="FLOW")] \
            if "FLOW" in h else []
        rest.append(fits.BinTableHDU(data=h["FRAME_TABLE"].data.copy(), header=h["FRAME_TABLE"].header.copy(), name="FRAME_TABLE"))
    # MEMORY (2026-11-12, memory only; the rewritten cube is unchanged byte for byte): the decompressed cube is divided by
    # the factor IN PLACE, frame by frame, and is from here on the cube in Master units BEFORE re-quantization (what
    # "old / F" was). Before, a second whole cube (old / F) and two int16 copies of the mask were held beside it
    if not new.flags.writeable:
        new = new.copy()
    f32 = np.float32(fs)
    for i in range(new.shape[0]):
        np.divide(new[i], f32, out=new[i])
    quant = float(ph.get("QUANTLEV", QUANT))
    for k in [k for k in ch if k.startswith("Z") or k in ("XTENSION", "PCOUNT", "GCOUNT", "TFIELDS", "TTYPE1", "TFORM1")]:
        del ch[k]
    ph["FSAPPLD"] = (float(fs), "frame-to-Master flux scale divided out of the pixels")
    ph["RESCALED"] = (git_hash(), "clv15 commit of the rescale")
    # THE STANDARD HEADER STAMP in this rewrite (CUBE_STAMP_SPEC_20261110): noise measured on old / F (the pixels before
    # re-quantization), an I cube's THERMNOI from the Stokes V cube beside the source
    cell, sb, st = (str(ph.get(k, "")).strip() for k in ("CELL", "SUBBAND", "STOKES"))
    if cube_name(cell, sb, st) != os.path.basename(a.cube):
        sys.exit("REFUSED: %s carries CELL/SUBBAND/STOKES %s %s %s - not its own name" % (a.cube, cell, sb, st))
    tss = [s.strip() if isinstance(s, str) else s.decode().strip() for s in rest[-1].data["ts"]]
    vpath = os.path.join(os.path.dirname(os.path.abspath(a.cube)), cube_name(cell, sb, "V"))
    nz = noise_cards(cell, sb, st, tss, lambda i: np.where(mask[i] != 0, np.float32(np.nan), new[i]),
                     (lambda adm: cube_file_noise(vpath, adm, fs)) if st == "I" else None, a.noise_dir)
    left = stamp_cards(ph, rest[-1].data, cell, sb, st, fs=fs, noise=nz, prodname=os.path.basename(a.cube),
                       admitted=admitted_frames(cell, sb, st, a.noise_dir), grid=ch)
    hdus = [fits.PrimaryHDU(header=ph), comp_hdu(fits, new, ch, quant),
            fits.CompImageHDU(data=mask, name="MASK", compression_type="RICE_1",
                              tile_shape=(1,) + mask.shape[1:])] + rest
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, os.path.basename(a.cube))
    if os.path.abspath(out) == os.path.abspath(a.cube):
        sys.exit("REFUSED: --out is the source directory - a rescale never overwrites its source")
    tmp = out + ".tmp"
    fits.HDUList(hdus).writeto(tmp, overwrite=True, checksum=True)
    worst = 0.0
    with fits.open(tmp, memmap=False) as n:
        # the whole MASK at once, not section[]: in a high-band cube the MASK starts past 2 GiB, where section[] overflows
        # (astropy 6.1 + numpy 2, see Cube.frame); compared frame by frame so that no whole-cube temporary is made
        nm = np.asarray(n["MASK"].data)
        if nm.shape != mask.shape or any(not np.array_equal(nm[i], mask[i]) for i in range(mask.shape[0])):
            os.remove(tmp); sys.exit("RESCALE FAILED %s: MASK differs" % out)
        del nm
        if n["FRAME_TABLE"].data.tobytes() != rest[-1].data.tobytes():
            os.remove(tmp); sys.exit("RESCALE FAILED %s: FRAME_TABLE differs" % out)
        # FRAMES is read by section[]: it is the first extension (data at byte 11,520), and astropy gives it 32-bit
        # pointers only when the decoded cube is under 4 GiB, whose compressed heap is under 1 GiB - no sum passes 2 GiB
        # (82 MHz, 358 frames, 2.19 GB compressed: 64-bit pointers, every frame read back, 2026-11-12)
        for i in range(new.shape[0]):
            d0, d1 = new[i], np.asarray(n["FRAMES"].section[i], dtype=np.float32)
            m = np.isfinite(d0)
            if (m != np.isfinite(d1)).any():
                os.remove(tmp); sys.exit("RESCALE FAILED %s: NaN mask differs in frame %d" % (out, i))
            if m.any():
                mad = 1.4826 * np.median(np.abs(d0[m] - np.median(d0[m])))
                if mad > 0:
                    worst = max(worst, float(np.max(np.abs(d1[m] - d0[m])) / mad))
    if worst > RESCALE_TOL:
        os.remove(tmp); sys.exit("RESCALE FAILED %s: worst pixel %.4f of frame noise > %.2f" % (out, worst, RESCALE_TOL))
    ok, detail = acceptance_check(tmp, left)    # the Release 1 acceptance test of the new primary: a gate, it refuses
    if not ok:
        os.remove(tmp); sys.exit("RESCALE FAILED %s: Release 1 acceptance cube check: %s; left out: %s"
                                 % (out, detail, "; ".join("%s (%s)" % (k, r[:120]) for k, r in left) or "none"))
    print("ACCEPTANCE %s PASS: %s; cards left out: %d" % (out, detail, len(left)))
    os.replace(tmp, out)
    print("RESCALE %s FSAPPLD=%.6f frames=%d worst pixel/noise=%.4f -> OK" % (out, fs, new.shape[0], worst))


RESCALE_TOL = 0.05     # re-quantization of an already-quantized cube: measured 0.006 per quantization (2026-11-12 write test)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd")
    w = sp.add_parser("write")
    w.add_argument("--stage", required=True); w.add_argument("--cell", required=True)
    w.add_argument("--subband", required=True); w.add_argument("--stokes", required=True, choices=["I", "V"])
    w.add_argument("--flows", default=None); w.add_argument("--out", required=True)
    w.add_argument("--quant", type=float, default=QUANT)
    w.add_argument("--fluxscal", type=float, default=None, help="divide the pixels by this hour-band factor (FSAPPLD)")
    w.add_argument("--stamp", action="store_true", help="the standard header stamp (CUBE_STAMP_SPEC_20261110); default OFF")
    w.add_argument("--noise-dir", default=None, help="the search's noise tables (default SEARCH_OUT_ROOT/<cell>)")
    v = sp.add_parser("verify"); v.add_argument("cube"); v.add_argument("--stage", required=True)
    e = sp.add_parser("explode"); e.add_argument("cube"); e.add_argument("--out", required=True)
    r = sp.add_parser("rescale"); r.add_argument("cube"); r.add_argument("--fluxscal", type=float, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--noise-dir", default=None, help="the search's noise tables (default SEARCH_OUT_ROOT/<cell>)")
    a = ap.parse_args()
    {"write": write, "verify": verify, "explode": explode, "rescale": rescale}[a.cmd](a)


if __name__ == "__main__":
    main()
