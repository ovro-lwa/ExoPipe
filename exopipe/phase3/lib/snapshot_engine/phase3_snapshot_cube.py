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
  verify  <cube> --stage <staged subband dir>    re-read every frame and compare with the flowed FITS it came from
  explode <cube> --out <dir>                     write every frame back out as <base>_flowed.fits (test harness)
Python API: open_cube(path) -> Cube; Cube.frame(ts) -> (data float32, astropy Header); Cube.timestamps
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys

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
    return fits.CompImageHDU(data=cube, header=header, name="FRAMES", compression_type="RICE_1",
                             tile_shape=(1,) + cube.shape[1:], quantize_level=quant,
                             quantize_method=1)       # 1 = SUBTRACTIVE_DITHER_1


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
    # MEMORY (refactor, 2026-11-12, speed only; the cube file is unchanged byte for byte): the frames go straight into ONE
    # preallocated cube and the NaN mask into one int16 array, frame by frame. Before, every frame was held twice (a list
    # and its stacked copy) beside two whole-cube masks (bool, then int16): 51.6 GB at the peak for a 14 GB cube of 358
    # frames of 3122 x 3122 px, which let only four or five writers of the high bands run side by side in a cell
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
    QUANTLEV, dither); MASK, FLOW and FRAME_TABLE carried over unchanged; the primary gains FSAPPLD.  Writes <out>/<same name>,
    never over the source; REFUSES a cube already carrying FSAPPLD.  Content test before the rename: masks and FRAME_TABLE
    identical, and every frame's |new - old / F| at most RESCALE_TOL of that frame's robust noise / F."""
    from astropy.io import fits
    fs = a.fluxscal
    if not (np.isfinite(fs) and fs > 0):
        sys.exit("REFUSED: --fluxscal %r is not a finite factor > 0" % fs)
    with fits.open(a.cube, memmap=False) as h:
        ph = h[0].header.copy()
        if "FSAPPLD" in ph:
            sys.exit("REFUSED: %s already carries FSAPPLD=%s" % (a.cube, ph["FSAPPLD"]))
        old = np.asarray(h["FRAMES"].data, dtype=np.float32)
        ch = h["FRAMES"].header.copy()
        mask = np.asarray(h["MASK"].data)
        rest = [fits.ImageHDU(data=np.array(h["FLOW"].data), header=h["FLOW"].header.copy(), name="FLOW")] \
            if "FLOW" in h else []
        rest.append(fits.BinTableHDU(data=h["FRAME_TABLE"].data.copy(), header=h["FRAME_TABLE"].header.copy(), name="FRAME_TABLE"))
    quant = float(ph.get("QUANTLEV", QUANT))
    for k in [k for k in ch if k.startswith("Z") or k in ("XTENSION", "PCOUNT", "GCOUNT", "TFIELDS", "TTYPE1", "TFORM1")]:
        del ch[k]
    ph["FSAPPLD"] = (float(fs), "frame-to-Master flux scale divided out of the pixels")
    ph["RESCALED"] = (git_hash(), "clv15 commit of the rescale")
    hdus = [fits.PrimaryHDU(header=ph), comp_hdu(fits, old / np.float32(fs), ch, quant),
            fits.CompImageHDU(data=mask.astype(np.int16), name="MASK", compression_type="RICE_1",
                              tile_shape=(1,) + mask.shape[1:])] + rest
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, os.path.basename(a.cube))
    if os.path.abspath(out) == os.path.abspath(a.cube):
        sys.exit("REFUSED: --out is the source directory - a rescale never overwrites its source")
    tmp = out + ".tmp"
    fits.HDUList(hdus).writeto(tmp, overwrite=True, checksum=True)
    worst = 0.0
    with fits.open(tmp, memmap=False) as n:
        if not np.array_equal(np.asarray(n["MASK"].data), mask.astype(np.int16)):
            os.remove(tmp); sys.exit("RESCALE FAILED %s: MASK differs" % out)
        if n["FRAME_TABLE"].data.tobytes() != rest[-1].data.tobytes():
            os.remove(tmp); sys.exit("RESCALE FAILED %s: FRAME_TABLE differs" % out)
        for i in range(old.shape[0]):
            d0, d1 = old[i] / np.float32(fs), np.asarray(n["FRAMES"].section[i], dtype=np.float32)
            m = np.isfinite(d0)
            if (m != np.isfinite(d1)).any():
                os.remove(tmp); sys.exit("RESCALE FAILED %s: NaN mask differs in frame %d" % (out, i))
            if m.any():
                mad = 1.4826 * np.median(np.abs(d0[m] - np.median(d0[m])))
                if mad > 0:
                    worst = max(worst, float(np.max(np.abs(d1[m] - d0[m])) / mad))
    if worst > RESCALE_TOL:
        os.remove(tmp); sys.exit("RESCALE FAILED %s: worst pixel %.4f of frame noise > %.2f" % (out, worst, RESCALE_TOL))
    os.replace(tmp, out)
    print("RESCALE %s FSAPPLD=%.6f frames=%d worst pixel/noise=%.4f -> OK" % (out, fs, old.shape[0], worst))


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
    v = sp.add_parser("verify"); v.add_argument("cube"); v.add_argument("--stage", required=True)
    e = sp.add_parser("explode"); e.add_argument("cube"); e.add_argument("--out", required=True)
    r = sp.add_parser("rescale"); r.add_argument("cube"); r.add_argument("--fluxscal", type=float, required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"write": write, "verify": verify, "explode": explode, "rescale": rescale}[a.cmd](a)


if __name__ == "__main__":
    main()
