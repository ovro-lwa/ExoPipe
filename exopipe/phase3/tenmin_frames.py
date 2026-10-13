#!/usr/bin/env python3
"""
tenmin_frames.py — Stage-1 frame builder for the 10-MINUTE cadence of the deep
transient search (Gregg 2026-09-08 design: "the 10-min processing be identical
to the deep processing for V and I respectively").

Per cell (LST x scope), per night, per 10-minute bin k (0..5), per Stokes, this
module turns the Phase 2 10-minute image (Taper Robust 0, the only archived
variant) into the same kind of Stage-1 product the deep search consumes:

    <out_root>/<lst>_<scope>/subtracted/<date>T<kk>_<S>_sub_<scope>.fits
        = frame_k * PB_deep / PB_bin_k  / DC_night  , flow-aligned, MINUS the
          deep Taper Robust 0 master (central master, no leave-one-out)

with the FRAME'S OWN time block and beam in the header, so phase3_deep can run
its ratified chain unchanged (cadence='10min': same flow refine, striation
clean, gates, ledger, archive frames).

The three 10-minute-specific steps, in order (Gregg 2026-09-08):
  1. PRIMARY-BEAM RATIO. The phase centre is fixed while the zenith moves across
     the hour, so a 10-minute frame carries bin k's gain pattern and the master
     carries the hour-averaged one. Multiplying by PB_deep/PB_bin_k (pb_library,
     composed per scope by pb_apply.wideband_pb) puts the frame on the master's
     gain pattern; where PB_bin_k < PB_FLOOR the frame is blanked (not searched),
     never amplified.
  2. SCALAR FLUX FACTOR. The night's DC flux scale from the co-add's own record
     (coadd_final/weights.csv, qa3_scale). NO gradient (R-TILT-REMOVED-NEXTGEN).
  3. NIGHT FLOW. The night's saved alignment flow (flow/<date>_scaled_flow.npy),
     applied at native resolution by coadd_full_pipeline.process_variant_image
     (which also regrids pre-cutover nights to the master grid). The per-frame
     x2 refinement against the master happens in phase3_deep, as for deep epochs.

Bin index: assigned from each frame's own DATE-OBS relative to the earliest
10-minute DATE-OBS of that night (any scope), in 600 s steps — never from the
file's position in a directory listing (the old phase3_tenmin defect: subband
filenames carry no _tNNNN token, so a missing interval shifted every later one).

Injection (battery only): `inject` = list of dicts {key, ra, dec, flux_jy} adds a
beam-shaped Gaussian (the frame's own BMAJ/BMIN/BPA) to the aligned, PB-ratioed
frame BEFORE subtraction; the truth is written to injections.csv. Injected trees
are built under a separate out_root and never mixed with production frames.

Refusals (never silent): master not the declared generation; night without a
DC record or a Phase 2 10-minute product; PB map unavailable.  A night WITHOUT A
FLOW is not refused: it is processed with the flux scale only (has_flow=0,
FLOWUSED=False).  This module built the frames of the RETIRED old 10-minute chain
(Run_P3T4); the production 10-minute tier is the cube-based sub-LST chain.
(Docstring corrected 2026-10-07, Gregg "All approved", Q15(2).)
"""
import os, sys, csv, glob, re, json, time
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from exopipe.common.roots import coadd_root, required_root
from exopipe.phase3.core import products as pp
from exopipe.common import phase2_config as cfg
from exopipe.phase3 import coadd_full_pipeline as C
from exopipe.phase3 import pb_apply as PB
from exopipe.phase3.coadd_full_pipeline import load_exclusions, is_excluded

PB_FLOOR = float(os.environ.get('P3_TENMIN_PB_FLOOR', str(PB.PB_FLOOR)))   # 0.1 as pbcorr
BIN_SEC = 600.0
NBINS = 6
MASTER_GEN = os.environ.get('P3_MASTER_GEN', 'ivw-mean-v3')
# PER-BIN LEAVE-ONE-OUT MASTERS (Gregg 2026-09-09: "Per bin Masters are approved" / "Use leave one out"):
# each 10-minute frame is subtracted against the leave-one-out master of ITS OWN BIN, built from the
# same bin across nights (inverse-variance weighted by the co-add's night weights, no clipping,
# LOO for every bin whatever N). No PB ratio: the bin's frames share the beam pattern. Measured on the
# battery: Stage-1 I residual 12-18 per cent lower, V 4-6 per cent. P3_TENMIN_PER_BIN=0 restores the
# deep-master x PB-ratio form for comparison only.
PER_BIN = os.environ.get('P3_TENMIN_PER_BIN', '1') != '0'
BIN_COADDOP = 'ivw-mean-v3-bin'


def tenmin_root():
    """Bulk working root for the 10-minute Stage-1 trees (CLAUDE.md s7 amended
    2026-09-07: bulk products on Lustre). No default (R-COADD-ROOT-NO-DEFAULTS)."""
    return required_root('P3_TENMIN_ROOT', '10-minute Stage-1 frame trees (bulk, Lustre)')


def frame_key(date, k):
    return f'{date}T{k:02d}'


def key_date(key):
    """'2025-04-14T03' -> '2025-04-14' (also passes a plain date through)."""
    return key[:10]


def _p2_10min_files(sci_dir, scope, stokes):
    """Phase 2 10-minute Taper Robust 0, non-pbcorr images with their DATE-OBS.
    Dewarped products first; RAW FALLBACK (A-05, approved: the deep chain's raw-fallback
    pattern extends to the 10-minute tier) when the night has no dewarped 10-minute
    product - recorded per frame (frames.csv src_dewarped, header P2DEWARP)."""
    out = []
    files = [p for p, _i in C.find_10min_images(sci_dir, scope.replace('MHz', ''), stokes)]
    dewarped = True
    if not files:
        d = (f'{sci_dir}/{scope}/{stokes}/10min' if not C.is_subband(scope.replace('MHz', ''))
             else f'{sci_dir}/{scope}/{stokes}/10min')
        if os.path.isdir(d):
            files = sorted(os.path.join(d, f) for f in os.listdir(d)
                           if f.endswith('.fits') and 'pbcorr' not in f and 'dewarped' not in f
                           and 'Taper' in f and 'NoTaper' not in f
                           and 'Robust-0' in f and 'Robust-0.75' not in f)
        dewarped = False
    for p in files:
        try:
            h = fits.getheader(p)
        except Exception:
            continue
        t = h.get('DATE-OBS')
        if not t:
            continue
        h['P2DEWARP'] = (dewarped, 'Phase 2 10-minute source was dewarped')
        out.append((Time(t, scale='utc').mjd, p, h))
    return sorted(out)


def _night_anchor_mjd(sci_dir):
    """Earliest 10-minute DATE-OBS of the night over every scope and Stokes I:
    the schedule anchor for bin assignment."""
    mj = []
    for d in glob.glob(f'{sci_dir}/*/I/10min'):
        for f in os.listdir(d):
            if not f.endswith('.fits') or 'pbcorr' in f or 'dewarped' not in f:
                continue
            try:
                t = fits.getheader(os.path.join(d, f)).get('DATE-OBS')
                if t:
                    mj.append(Time(t, scale='utc').mjd)
            except Exception:
                pass
    return min(mj) if mj else None


def _bin_of(mjd, anchor):
    k = int(round((mjd - anchor) * 86400.0 / BIN_SEC))
    return k if 0 <= k < NBINS else None


# SIDEREAL BINS + WINDOW PB CORRECTION (Gregg 2026-09-23, "2. approved" = adopt arm b of the PB-window pilot, ledger
# MEASURED-PBWIN-PILOT-FOUR-ARMS-...). Phase 1 cuts the hour with WSClean -intervals-out 6, i.e. six equal chunks of the
# timesteps PRESENT, so on ~29% of nights a frame's window is not its bin's 10 minutes of LST (1.5-10 min, anywhere in
# the hour), and _bin_of's night-anchored ordinal made that worse. Each frame is now assigned to the sidereal bin that
# holds the MIDPOINT of its own DATE-OBS..DATE-END in LST (midpoint outside the hour: refused), and a frame whose window
# is not the bin's nominal window is multiplied by PB_bin_k / PB_window (model PB, pb_apply.window_scope_pb - the
# library's own construction), blanked where either is below PB_FLOOR, so every frame of a bin - and so its master -
# carries the bin's gain pattern. Pilot: frames that change bin, 09h Green east 10-25 deg residual/master +0.365 -> +0.063,
# west -0.173 -> -0.004. P3_TENMIN_SIDEREAL=0 restores the ordinal bins without correction (comparison runs only).
SIDEREAL = os.environ.get('P3_TENMIN_SIDEREAL', '1') != '0'
NOMINAL_TOL_MIN = 0.25
_WINCORR = {}


def _window_lst_min(sh, lst):
    """(t0, t1) of a Phase-2 10-minute frame in minutes of LST from the start of the cell's LST hour."""
    from exopipe.common.pipeline_config import OVRO_LOC
    t0 = Time(sh['DATE-OBS'], scale='utc')
    t1 = Time(sh['DATE-END'], scale='utc') if sh.get('DATE-END') else t0 + float(sh['INTTIME']) / 86400.0
    hour = int(str(lst)[:2])
    return tuple(((t.sidereal_time('apparent', longitude=OVRO_LOC.lon).hour - hour + 12) % 24 - 12) * 60.0 for t in (t0, t1))


def _window_correction(scope, lst, master_path, k, t0, t1, hdr0, shape):
    """(ratio map PB_bin_k / PB_window on the frame grid, blank fraction, median ratio); None when the window is nominal."""
    if abs(t0 - 10.0 * k) < NOMINAL_TOL_MIN and abs(t1 - 10.0 * k - 10.0) < NOMINAL_TOL_MIN:
        return None
    key = (scope, lst, k, round(t0 * 2) / 2, round(t1 * 2) / 2)
    if key not in _WINCORR:
        sb = scope if scope in cfg.ALL_SUBBANDS else cfg.SUBBAND_BANDS[scope][0]
        _, pbhdr = PB.subband_bin_pb(sb, 0)
        nom = _pb_on_grid(PB.window_scope_pb(scope, lst, master_path, 10.0 * k, 10.0 * k + 10.0), pbhdr, hdr0, shape)
        win = _pb_on_grid(PB.window_scope_pb(scope, lst, master_path, t0, t1), pbhdr, hdr0, shape)
        ok = (nom >= PB_FLOOR) & (win >= PB_FLOOR)
        r = np.where(ok, nom / np.where(ok, win, 1.0), 0.0).astype(np.float32)
        _WINCORR[key] = (r, float(1.0 - ok.mean()), float(np.median(r[ok])) if ok.any() else np.nan)
    return _WINCORR[key]


def _pb_on_grid(pb, pbhdr, hdr0, shape):
    if pb.shape == tuple(shape):
        return pb
    from reproject import reproject_interp
    g, _ = reproject_interp((pb, WCS(pbhdr).celestial), WCS(hdr0).celestial,
                            shape_out=tuple(shape), order='bilinear')
    return np.nan_to_num(g, nan=0.0).astype(np.float32)


def pb_ratio_maps(scope, lst, master_path, hdr0, shape):
    """{k: (ratio_k, blank_frac_k)}; ratio = PB_deep/PB_bin_k, 0 where PB_bin_k < PB_FLOOR."""
    deep, meta = PB.wideband_pb(scope, lst, master_path, binidx=None)
    if deep is None:
        raise RuntimeError(f'{lst} {scope}: no deep PB for the ratio - REFUSING')
    # wideband_pb returns the map on the members' grid; carry a header for reprojection
    sb = meta['members'][0]
    _, pbhdr = PB.subband_deep_pb(sb)
    deep = _pb_on_grid(deep, pbhdr, hdr0, shape)
    out = {}
    for k in range(NBINS):
        b, _m = PB.wideband_pb(scope, lst, master_path, binidx=k)
        if b is None:
            raise RuntimeError(f'{lst} {scope}: no bin-{k} PB for the ratio - REFUSING')
        b = _pb_on_grid(b, pbhdr, hdr0, shape)
        ok = (b >= PB_FLOOR) & (deep > 0)
        r = np.where(ok, deep / np.where(ok, b, 1.0), 0.0).astype(np.float32)
        out[k] = (r, float(1.0 - ok.mean()), float(np.nanmedian(r[ok])) if ok.any() else np.nan)
    return out, meta


def _gauss_inject(img, hdr0, cd, ra, dec, flux, bmaj_deg, bmin_deg, bpa_deg):
    w = WCS(hdr0).celestial
    x0, y0 = w.world_to_pixel_values(ra, dec)
    if not (np.isfinite(x0) and np.isfinite(y0)):
        return False
    sx = bmaj_deg / cd / 2.3548; sy = bmin_deg / cd / 2.3548
    th = np.radians(bpa_deg + 90.0)     # BPA east of north -> array angle
    n = int(6 * max(sx, sy)) + 1
    xa, xb = max(0, int(x0) - n), min(img.shape[1], int(x0) + n + 1)
    ya, yb = max(0, int(y0) - n), min(img.shape[0], int(y0) + n + 1)
    if xb <= xa or yb <= ya:
        return False
    yy, xx = np.mgrid[ya:yb, xa:xb]
    dx, dy = xx - x0, yy - y0
    u = dx * np.cos(th) + dy * np.sin(th); v = -dx * np.sin(th) + dy * np.cos(th)
    img[ya:yb, xa:xb] += flux * np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
    return True


_CTX = {}


def _build_night(args):
    """One night, all bins, all Stokes (worker; context in _CTX after fork). Returns
    (frame rows, injection rows, refusals, built keys)."""
    date, sci_dir, align_img = args
    c = _CTX
    lst, scope, stokes, base, od, shape, hdr0, cd = (c['lst'], c['scope'], c['stokes'], c['base'],
                                                    c['od'], c['shape'], c['hdr0'], c['cd'])
    masters, ratios, pbmeta, dc, inj_by_key, overwrite = (c['masters'], c['ratios'], c['pbmeta'],
                                                          c['dc'], c['inj_by_key'], c['overwrite'])
    mI, mV = c['mI'], c['mV']
    rows, irows, refused, built = [], [], [], []
    fp = f'{base}/flow/{date}_scaled_flow.npy'
    # PARITY WITH THE DEEP CO-ADD: a night without a saved flow is aligned by flux
    # scale only there (coadd_full_pipeline 'no flow field found, using flux-scale
    # only'; the standing state of most wideband nights, FINDING 2026-09-08), so the
    # 10-minute frames of that night get the same treatment - recorded per frame
    # (frames.csv has_flow, header FLOWUSED), never silent. The per-frame x2
    # refinement in phase3_deep is the alignment both cadences rely on.
    has_flow = os.path.exists(fp)
    if date not in dc:
        return rows, irows, [(date, 'no DC in weights.csv')], built
    flow = np.load(fp) if has_flow else None
    anchor = _night_anchor_mjd(sci_dir)
    if anchor is None:
        return rows, irows, [(date, 'no 10-minute products')], built
    scale_map = np.full(shape, dc[date], dtype=float)
    regrid = None
    for S in stokes:
        files = _p2_10min_files(sci_dir, scope, S)
        if not files:
            refused.append((f'{date} {S}', 'no 10-minute Phase 2 images')); continue
        if overwrite:                                     # a rebuild re-keys frames (sidereal bins): clear the night's old
            for stale in glob.glob(f'{od}/{date}T??_{S}_sub_{scope}.fits') + glob.glob(f'{od}/{date}T??_{S}_al_{scope}.fits'):
                os.remove(stale)                          # frames first, so none survives under a key no longer built
        placed = []                                       # (mjd, src, sh, k, t0, t1)
        for mjd, src, sh in files:
            if SIDEREAL:
                t0, t1 = _window_lst_min(sh, lst)
                mid = 0.5 * (t0 + t1)
                if not 0.0 <= mid < 60.0:
                    refused.append((f'{date} {S} {os.path.basename(src)}', f'window midpoint {mid:.1f} min outside the LST hour')); continue
                placed.append((mjd, src, sh, int(mid // 10.0), t0, t1))
            else:
                k = _bin_of(mjd, anchor)
                if k is None:
                    refused.append((f'{date} {S} {os.path.basename(src)}', 'DATE-OBS outside the six bins')); continue
                placed.append((mjd, src, sh, k, None, None))
        # SAME-BIN FRAMES ARE COMBINED (Gregg 2026-09-23, "combine them"): two short frames of one night whose window
        # midpoints fall in one sidereal bin (1.5% of frames at 69 MHz) are each aligned, DC-scaled and window-PB-corrected
        # to the bin's nominal PB, then averaged weighted by window duration with per-pixel validity. The combined frame's
        # time block runs from the first DATE-OBS to the last DATE-END, INTTIME = the summed data duration, NCOMB = 2+.
        groups = {}
        for p in placed:
            groups.setdefault(p[3], []).append(p)
        for k in sorted(groups):
            members = sorted(groups[k], key=lambda p: p[0])
            key = frame_key(date, k)
            sub_path = f'{od}/{key}_{S}_sub_{scope}.fits'
            # DIVERGENCE GATE AT 10 MINUTES (Gregg 2026-09-22, "rebuild the corrupt Masters"): the per-variant gate
            # of the deep co-add (_variant_input_divergent, |max| > 1e5 Jy/beam or finite fraction < 0.99) was never
            # applied here, so CLEAN-diverged frames (13h 69MHz 2025-05-07 bin5, 3e31 Jy/beam) entered 25 bin masters.
            # Checked BEFORE the skip-if-built test, per member, and a key whose members are all refused has its stale
            # products removed, so a frame built before the gate existed can never be searched or stacked again.
            ok_members = []
            for p in members:
                why = C._variant_input_divergent(p[1])
                if why:
                    refused.append((f'{key} {S}', f'divergent input ({why})'))
                else:
                    ok_members.append(p)
            if not ok_members:
                for stale in (sub_path, f'{od}/{key}_{S}_al_{scope}.fits'):
                    if os.path.exists(stale):
                        os.remove(stale)
                continue
            if os.path.exists(sub_path) and not overwrite:
                built.append((key, S)); continue
            parts = []                                    # (frame, duration_s, sh, src, t0, t1, wc)
            for mjd, src, sh, _k, t0, t1 in ok_members:
                try:
                    d0 = fits.getdata(src, memmap=True).squeeze().shape
                except Exception as e:
                    refused.append((f'{key} {S}', f'unreadable ({e})')); continue
                if tuple(d0) != tuple(shape) and regrid is None:
                    regrid = C.precompute_regrid_mapping(src, hdr0, shape)
                tmp = f'{od}/.tmp_{key}_{S}_{os.getpid()}.fits'
                ok = C.process_variant_image(src, flow, scale_map, hdr0, shape, tmp,
                                             regrid if tuple(d0) != tuple(shape) else None)
                if not ok or not os.path.exists(tmp):
                    refused.append((f'{key} {S}', 'alignment failed (grid)')); continue
                f1 = np.squeeze(fits.getdata(tmp)).astype(np.float32); os.remove(tmp)
                f1[~np.isfinite(f1)] = 0.0
                wc1 = _window_correction(scope, lst, c['mI'] if 'I' in stokes else c['mV'], k, t0, t1, hdr0, shape) if SIDEREAL else None
                if wc1 is not None:
                    f1 = (f1 * wc1[0]).astype(np.float32)      # window PB -> the bin's nominal PB (0 below the floor)
                dur = ((t1 - t0) * 60.0 / 1.0027379) if t0 is not None else float(sh.get('INTTIME') or 600.0)
                parts.append((f1, dur, sh, src, t0, t1, wc1))
            if not parts:
                continue
            if len(parts) == 1:
                fr, _d, sh, src, t0, t1, wc = parts[0]
                ncomb, src2 = 1, None
            else:
                num = np.zeros(shape, np.float64); den = np.zeros(shape, np.float64)
                for f1, dur, *_r in parts:
                    v = f1 != 0; num += dur * f1; den += dur * v
                fr = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0).astype(np.float32)
                lng = max(parts, key=lambda q: q[1])
                sh = lng[2].copy(); src = lng[3]; wc = next((q[6] for q in parts if q[6] is not None), None)
                sh['DATE-OBS'] = min(q[2]['DATE-OBS'] for q in parts)
                ends = [q[2].get('DATE-END') for q in parts]
                if all(ends):
                    sh['DATE-END'] = max(ends)
                sh['INTTIME'] = round(sum(q[1] for q in parts), 2)
                sig = [q[2].get('THERMNOI') for q in parts]
                if all(x is not None for x in sig):
                    wsum = sum(q[1] for q in parts)
                    sh['THERMNOI'] = float(np.sqrt(sum((q[1] * float(x)) ** 2 for q, x in zip(parts, sig)))) / wsum
                t0 = min(q[4] for q in parts); t1 = max(q[5] for q in parts)
                ncomb, src2 = len(parts), [os.path.basename(q[3]) for q in parts if q[3] != src][0]
            if PER_BIN:
                r, blank, rmed = 1.0, (wc[1] if wc else 0.0), (wc[2] if wc else 1.0)   # frames.csv: window correction
            else:
                r, blank, rmed = ratios[k]
                fr = fr * r                               # step 1: PB ratio (0 below the floor)
            bmaj, bmin, bpa = (float(sh.get('BMAJ') or 0), float(sh.get('BMIN') or 0),
                               float(sh.get('BPA') or 0.0))
            if not (bmaj > 0 and bmin > 0):
                refused.append((f'{key} {S}', 'no beam in the Phase 2 header')); continue
            p2run = ('Science_' + src.split('Science_')[1].split('/')[0]) if 'Science_' in src else None
            ninj = 0
            for it in inj_by_key.get(key, []):
                if it.get('stokes', S) != S:
                    continue
                w = WCS(hdr0).celestial; x, y = w.world_to_pixel_values(it['ra'], it['dec'])
                if _gauss_inject(fr, hdr0, cd, it['ra'], it['dec'], it['flux_jy'], bmaj, bmin, bpa):
                    ninj += 1
                    irows.append([key, S, it['ra'], it['dec'], it['flux_jy'], round(float(x), 2),
                                  round(float(y), 2), sub_path])
            if PER_BIN:
                # pass 1 ends here: the aligned, DC-scaled frame is kept for the per-bin
                # leave-one-out subtraction in build_cell_frames (cards it needs travel in the header)
                al_path = f'{od}/{key}_{S}_al_{scope}.fits'
                ha = WCS(hdr0).celestial.to_header()
                for kk, vv in (('DATE-OBS', sh.get('DATE-OBS')), ('DATE-END', sh.get('DATE-END')), ('INTTIME', sh.get('INTTIME')),
                               ('TIMEFIX', sh.get('TIMEFIX')), ('THERMNOI', sh.get('THERMNOI')), ('BMAJ', bmaj), ('BMIN', bmin), ('BPA', bpa),
                               ('P2DEWARP', bool(sh.get('P2DEWARP', True))), ('SRCFRAME', os.path.basename(src)[:68]),
                               ('P2RUN', p2run), ('FLOWUSED', bool(has_flow)),
                               ('NINJECT', int(ninj)), ('FLUXDC', round(dc[date], 5)), ('LSTBIN', int(k)), ('NIGHT', date),
                               ('LSTWIN0', round(t0, 2) if t0 is not None else None), ('LSTWIN1', round(t1, 2) if t1 is not None else None),
                               ('PBWCORR', bool(wc is not None) if SIDEREAL else None),
                               ('NCOMB', int(ncomb) if ncomb > 1 else None), ('SRCFRM2', src2[:68] if src2 else None)):
                    if vv is not None:
                        ha[kk] = vv
                fits.writeto(al_path + '.part', fr.astype(np.float32), ha, overwrite=True); os.replace(al_path + '.part', al_path)
                rows.append([key, date, k, S, src, round(dc[date], 5), int(has_flow), int(bool(sh.get('P2DEWARP', True))),
                             round(blank, 4), round(rmed, 4), bmaj, bmin, bpa, ninj, sub_path])
                built.append((key, S))
                continue
            valid = (fr != 0) & (masters[S] != 0)
            sub = np.where(valid, fr - masters[S], 0.0).astype(np.float32)
            h = WCS(hdr0).celestial.to_header()
            h['BUNIT'] = ('Jy/beam', 'Brightness unit'); h['TELESCOP'] = ('OVRO-LWA', 'Telescope')
            start = sh.get('DATE-OBS'); end = sh.get('DATE-END')
            inttime = sh.get('INTTIME')
            if end is None and start and inttime:
                end = (Time(start, scale='utc') + float(inttime) / 86400.0).isot
            p2run = ('Science_' + src.split('Science_')[1].split('/')[0]) if 'Science_' in src else None
            extra = {'CADENCE': ('10min', 'Frame cadence'),
                     'LSTBIN': (int(k), '10-minute bin within the LST hour (0-5)'),
                     'PBRATIO': (f'deep/bin{k}', 'PB_deep/PB_bin gain-pattern ratio applied'),
                     'PBFLOOR': (float(PB_FLOOR), 'bin PB below this: blanked, not searched'),
                     'PBBLANK': (round(blank, 4), 'fraction of the grid blanked by PBFLOOR'),
                     'PBRESID': (str(pbmeta.get('residual_pb', ''))[:20], 'residual PB version in the ratio'),
                     'FLUXDC': (round(dc[date], 5), 'night DC flux factor (no gradient)'),
                     'FLOWUSED': (bool(has_flow), 'night alignment flow applied (deep co-add parity)'),
                     'NINJECT': (int(ninj), 'injected sources (battery only; 0 in production)'),
                     'SRCFRAME': (os.path.basename(src)[:68], 'Phase 2 10-minute source image'),
                     'P2DEWARP': (bool(sh.get('P2DEWARP', True)), 'Phase 2 source dewarped (False = raw fallback, A-05)')}
            fits.writeto(sub_path + '.part', sub, h, overwrite=True)
            hh, ww = sub.shape; inn = sub[hh // 4:3 * hh // 4, ww // 4:3 * ww // 4]
            pp.write_standard_header(
                sub_path + '.part', scope=scope, stokes=S, start=start, end=end,
                inttime_s=inttime, time_basis=str(sh.get('TIMEFIX', 'v2-phase2')).replace('v2-', '', 1),
                beam=(bmaj, bmin, bpa), thermnoi=sh.get('THERMNOI'),
                imgrms=float(np.nanstd(inn[inn != 0])) if np.any(inn != 0) else None,
                dewarped=True, aligned=True, alignref=os.path.basename(mI if 'I' in stokes else mV),
                msub=True, msubfile=os.path.basename(mI if S == 'I' else mV),
                lstbin=int(k), p2run=p2run, scale_a=dc[date], extra=extra, validate=True)
            os.replace(sub_path + '.part', sub_path)
            rows.append([key, date, k, S, src, round(dc[date], 5), int(has_flow), int(bool(sh.get('P2DEWARP', True))),
                         round(blank, 4), round(rmed, 4), bmaj, bmin, bpa, ninj, sub_path])
            built.append((key, S))
    return rows, irows, refused, built


def _per_bin_loo_subtract(out_root, cell, lst, scope, stokes, hdr0, shape, cd, base, mI, mV, overwrite, gate_refused=None):
    """Pass 2 of the per-bin design: for every Stokes and bin, the inverse-variance weighted plain
    mean of the aligned frames of that bin across nights (weights = the co-add's own night weights,
    weights.csv), written as the central BIN MASTER (bin_masters/), and each frame minus its own
    LEAVE-ONE-OUT bin master, LOO_i = (S - w_i a_i) / (W - w_i) with per-pixel validity (exact
    identity, no clipping). Every bin is LOO regardless of N (Gregg 2026-09-09). Aligned
    intermediates are deleted after use (node-local scratch)."""
    import hashlib
    from exopipe.phase3.core import products as pp
    od = f'{out_root}/{cell}/subtracted'; bd = f'{out_root}/{cell}/bin_masters'; os.makedirs(bd, exist_ok=True)
    wts = {}
    for r in csv.DictReader(open(f'{base}/coadd_final/weights.csv')):
        try:
            wts[r['date']] = float(r['weight'])
        except (KeyError, ValueError, TypeError):
            pass
    def _retire_bin(S, k, n_left):
        """A bin the divergence gate leaves with fewer than two frames has no master and no search frame. Its
        previous master, weights and search frames (all built from the refused inputs) move to superseded/ in the
        work root and the archive - never deleted - its bins.csv rows are dropped, and <S>_bin<k>_<scope>.refused.txt
        records why, which the verifier accepts in place of the master (2026-09-22)."""
        why = [f'{a}: {b}' for a, b in gate_refused[(S, k)]]
        txt = f'{S} bin{k} {scope}: {n_left} frame(s) left after the divergence gate - no bin master, bin not searched\n' + '\n'.join(why) + '\n'
        sup = f'{bd}/superseded'
        for f in ([f'{bd}/{S}_bin{k}_{scope}.fits', f'{bd}/weights_{S}_bin{k}.csv']
                  + glob.glob(f'{od}/*T{k:02d}_{S}_sub_{scope}.fits') + glob.glob(f'{od}/*T{k:02d}_{S}_al_{scope}.fits')):
            if os.path.exists(f):
                os.makedirs(sup, exist_ok=True); os.replace(f, f'{sup}/{os.path.basename(f)}')
        open(f'{bd}/{S}_bin{k}_{scope}.refused.txt', 'w').write(txt)
        _arun = os.environ.get('P3_ARCHIVE_RUN')
        if os.environ.get('P3_ARCHIVE_WRITE') != '0' and _arun:
            _cd = f"{os.environ.get('P3_ARCHIVE_ROOT', '/lustre/pipeline/exopipe/phase3')}/Coadd/{_arun}/{lst}/{scope}/10min"
            os.makedirs(_cd, exist_ok=True)
            for _old in glob.glob(f'{_cd}/*_{S}_10min_*coadd*_bin{k}_*.fits'):
                os.makedirs(f'{_cd}/superseded', exist_ok=True); os.replace(_old, f'{_cd}/superseded/{os.path.basename(_old)}')
            _bcsv = f'{_cd}/bins.csv'
            if os.path.exists(_bcsv):
                rows_ = list(csv.reader(open(_bcsv)))
                with open(_bcsv + '.part', 'w', newline='') as fb_:
                    csv.writer(fb_, lineterminator='\n').writerows([rows_[0]] + [r for r in rows_[1:] if not (r[0] == S and r[1] == str(k))])
                os.replace(_bcsv + '.part', _bcsv)
            open(f'{_cd}/{S}_bin{k}_{scope}.refused.txt', 'w').write(txt)
        print(f'  [per-bin] {cell} {S} bin{k}: RETIRED - {n_left} frame(s) after the divergence gate ({len(why)} refused)', flush=True)

    gate_refused = gate_refused or {}
    for S in stokes:
        for k in range(6):
            als = sorted(glob.glob(f'{od}/*T{k:02d}_{S}_al_{scope}.fits'))
            if not als:
                if (S, k) in gate_refused:
                    _retire_bin(S, k, 0)
                continue
            imgs, w, keys, hdrs = [], [], [], []
            for f in als:
                key = os.path.basename(f).split(f'_{S}_al_')[0]; date = key_date(key)
                if date not in wts:
                    print(f'  [per-bin] {cell} {S} bin{k} {key}: night has no co-add weight - frame REFUSED', flush=True); continue
                with fits.open(f, memmap=False) as hh:
                    a = np.squeeze(hh[0].data).astype(np.float32); a[~np.isfinite(a)] = 0.0
                    hdrs.append(hh[0].header.copy())
                imgs.append(a); w.append(wts[date]); keys.append(key)
            if len(imgs) < 2:
                print(f'  [per-bin] {cell} {S} bin{k}: {len(imgs)} frame(s) - no leave-one-out possible; frames REFUSED', flush=True)
                if (S, k) in gate_refused:
                    _retire_bin(S, k, len(imgs))
                continue
            w = np.array(w, np.float64)
            Ssum = np.zeros(shape, np.float64); W = np.zeros(shape, np.float64)
            for a, wi in zip(imgs, w):
                v = (a != 0)
                Ssum += wi * a; W += wi * v
            M = np.where(W > 0, Ssum / np.where(W > 0, W, 1.0), 0.0).astype(np.float32)
            dates = sorted(key_date(x) for x in keys)
            dig = hashlib.md5(','.join(dates).encode()).hexdigest()
            mh = fits.getheader(mI if S == 'I' else mV).copy()
            mh['PARENTDG'] = (str(mh.get('CONSTDIG', '') or ''), 'CONSTDIG of the deep master this bin master derives from (2026-09-12)')
            mh['COADDOP'] = (BIN_COADDOP, 'per-bin inverse-variance plain mean (Gregg 2026-09-09)')
            mh['NCOADD'] = (len(imgs), 'constituent frames (bin)'); mh['LSTBIN'] = (int(k), '10-minute bin')
            mh['CONSTDIG'] = (dig, 'md5 of the sorted constituent night list'); mh['PARENTMS'] = (os.path.basename(mI if S == 'I' else mV)[:68], 'deep master of the cell')
            name = pp.build_name(scope, S, '10min', 'Taper', 0.0, proc=['dewarped', 'aligned', 'coadd'], lst=lst, lst_bin=k,
                                 date_start=dates[0].replace('-', ''), date_end=dates[-1].replace('-', ''), n_epochs=len(imgs))
            mh['PRODNAME'] = (name, 'Phase 3 product name (grammar; CONTINUE cards)')
            for d in dates:
                mh['HISTORY'] = f'COADD constituent {d}'
            _gr = gate_refused.get((S, k), [])
            for _key, _why in _gr:
                mh['HISTORY'] = f'VARIANT-GATE refused {_key}: {_why}'
            mh['NVARREF'] = (len(_gr), 'frames refused by the divergence gate')
            mpath = f'{bd}/{S}_bin{k}_{scope}.fits'
            fits.writeto(mpath + '.part', M, mh, overwrite=True); os.replace(mpath + '.part', mpath)
            if os.path.exists(f'{bd}/{S}_bin{k}_{scope}.refused.txt'):          # the bin has a master again
                os.replace(f'{bd}/{S}_bin{k}_{scope}.refused.txt', f'{bd}/{S}_bin{k}_{scope}.refused.txt.superseded')
            with open(f'{bd}/weights_{S}_bin{k}.csv', 'w', newline='') as fw_:
                cw = csv.writer(fw_, lineterminator='\n'); cw.writerow(['key', 'weight']); cw.writerows(zip(keys, w))
            # ARCHIVE (contract 4.0.1/steps 61 and 66, Gregg 2026-09-12): when the generation Run directory is named
            # (P3_ARCHIVE_RUN) and archiving is not switched off (P3_ARCHIVE_WRITE=0 = scratch/battery), the central
            # bin master goes to Coadd/{RUN}/{LST}/{band}/10min/<PRODNAME>.fits with bins.csv beside it, and each
            # aligned, flux-scaled frame to {LST}/{night}/{RUN}/{band}/{S}/10min/<PRODNAME>.fits before its local copy
            # is removed. Unset run + archiving on = REFUSAL (a 10-minute product is born in the archive or not at all).
            _arun = os.environ.get('P3_ARCHIVE_RUN'); _aw = os.environ.get('P3_ARCHIVE_WRITE') != '0'
            if _aw and not _arun:
                raise RuntimeError(f'{cell} {S} bin{k}: P3_ARCHIVE_RUN unset with archiving on - REFUSING to build 10-minute '
                                   f'products outside the archive (contract step 61); set P3_ARCHIVE_WRITE=0 for scratch')
            if _aw:
                import shutil as _sh
                _aroot = os.environ.get('P3_ARCHIVE_ROOT', '/lustre/pipeline/exopipe/phase3')
                _cd = f'{_aroot}/Coadd/{_arun}/{lst}/{scope}/10min'; os.makedirs(_cd, exist_ok=True)
                _dst = f'{_cd}/{name if name.endswith(".fits") else name + ".fits"}'
                _sh.copy2(mpath, _dst + '.part'); os.replace(_dst + '.part', _dst)
                # REBUILD REPLACES, NEVER APPENDS (2026-09-22): a bin master of this (Stokes, bin) under another name
                # (a different constituent count) is superseded - moved to 10min/superseded/, never deleted - and the
                # (Stokes, bin) rows of bins.csv are rewritten, so the verifier reads exactly one master per bin.
                for _old in glob.glob(f'{_cd}/*_{S}_10min_*coadd*_bin{k}_*.fits'):
                    if os.path.basename(_old) != os.path.basename(_dst):
                        os.makedirs(f'{_cd}/superseded', exist_ok=True)
                        os.replace(_old, f'{_cd}/superseded/{os.path.basename(_old)}')
                        print(f'  [per-bin] {cell} {S} bin{k}: superseded {os.path.basename(_old)} -> 10min/superseded/', flush=True)
                _bcsv = f'{_cd}/bins.csv'
                _hdr = ['stokes', 'bin', 'key', 'date', 'weight', 'bin_master', 'constdig', 'n_constituents']
                _keep = ([r for r in csv.reader(open(_bcsv))][1:] if os.path.exists(_bcsv) else [])
                _keep = [r for r in _keep if not (r[0] == S and r[1] == str(k))]
                with open(_bcsv + '.part', 'w', newline='') as fb_:
                    bw = csv.writer(fb_, lineterminator='\n')
                    bw.writerow(_hdr); bw.writerows(_keep)
                    for key_, wi_ in zip(keys, w):
                        bw.writerow([S, k, key_, key_date(key_), wi_, os.path.basename(_dst), dig, len(imgs)])
                os.replace(_bcsv + '.part', _bcsv)
                if os.path.exists(f'{_cd}/{S}_bin{k}_{scope}.refused.txt'):
                    os.replace(f'{_cd}/{S}_bin{k}_{scope}.refused.txt', f'{_cd}/{S}_bin{k}_{scope}.refused.txt.superseded')
                for f_, key_, ha_ in zip(als, keys, hdrs):
                    _pa = pp.build_name(scope, S, '10min', 'Taper', 0.0, proc=['dewarped', 'aligned'], lst=lst, lst_bin=k,
                                        epoch=str(ha_.get('DATE-OBS')))
                    _fd = f'{_aroot}/{lst}/{key_date(key_)}/{_arun}/{scope}/{S}/10min'; os.makedirs(_fd, exist_ok=True)
                    _fdst = f'{_fd}/{_pa if _pa.endswith(".fits") else _pa + ".fits"}'
                    with fits.open(f_, memmap=False) as _hh:
                        _hd = _hh[0].header.copy(); _hd['PRODNAME'] = (_pa, 'Phase 3 product name (grammar; CONTINUE cards)')
                        _hd['CADENCE'] = ('10min', 'Frame cadence'); _hd['LSTBIN'] = (int(k), '10-minute bin')
                        fits.writeto(_fdst + '.part', np.squeeze(_hh[0].data).astype(np.float32), _hd, overwrite=True)
                    os.replace(_fdst + '.part', _fdst)
            for a, wi, key, ha in zip(imgs, w, keys, hdrs):
                v = (a != 0)
                den = W - wi * v
                loo = np.where(den > 0, (Ssum - wi * a) / np.where(den > 0, den, 1.0), 0.0)
                valid = v & (den > 0)
                sub = np.where(valid, a - loo, 0.0).astype(np.float32)
                sub_path = f'{od}/{key}_{S}_sub_{scope}.fits'
                h = WCS(hdr0).celestial.to_header()
                h['BUNIT'] = ('Jy/beam', 'Brightness unit'); h['TELESCOP'] = ('OVRO-LWA', 'Telescope')
                _pn = pp.build_name(scope, S, '10min', 'Taper', 0.0, proc=['dewarped', 'aligned', 'msub'], lst=lst, lst_bin=k,
                                    epoch=str(ha.get('DATE-OBS')))
                extra = {'CADENCE': ('10min', 'Frame cadence'), 'LSTBIN': (int(k), '10-minute bin within the LST hour (0-5)'),
                         'PRODNAME': (_pn, 'Phase 3 product name (grammar; CONTINUE cards)'),
                         'PBRATIO': ('none', 'per-bin master: no PB ratio (Gregg 2026-09-09)'),
                         'BINMSTR': (os.path.basename(mpath), 'central bin master (bin_masters/)'),
                         'BINLOO': (True, 'subtracted its own leave-one-out bin master'),
                         'BINNCO': (len(imgs) - 1, 'constituents of the LOO bin master'),
                         'BINDIG': (dig, 'constituent digest of the bin master'),
                         'FLUXDC': (ha.get('FLUXDC'), 'night DC flux factor (no gradient)'),
                         'FLOWUSED': (bool(ha.get('FLOWUSED', False)), 'night alignment flow applied'),
                         'NINJECT': (int(ha.get('NINJECT', 0)), 'injected sources (battery only; 0 in production)'),
                         'SRCFRAME': (ha.get('SRCFRAME', ''), 'Phase 2 10-minute source image'),
                         'P2DEWARP': (bool(ha.get('P2DEWARP', True)), 'Phase 2 source dewarped (False = raw fallback, A-05)')}
                for _c, _cm in (('LSTWIN0', '[min] frame window start from the LST hour start'), ('LSTWIN1', '[min] frame window end'),
                                ('PBWCORR', 'window PB corrected to the bin nominal PB (2026-09-23)'),
                                ('NCOMB', 'same-night frames combined in this bin (duration-weighted)'), ('SRCFRM2', 'another combined source frame')):
                    if ha.get(_c) is not None:
                        extra[_c] = (ha.get(_c), _cm)
                fits.writeto(sub_path + '.part', sub, h, overwrite=True)
                hh, ww = sub.shape; inn = sub[hh // 4:3 * hh // 4, ww // 4:3 * ww // 4]
                pp.write_standard_header(
                    sub_path + '.part', scope=scope, stokes=S, start=ha.get('DATE-OBS'), end=ha.get('DATE-END'),
                    inttime_s=ha.get('INTTIME'), time_basis=str(ha.get('TIMEFIX', 'v2-phase2')).replace('v2-', '', 1),
                    beam=(float(ha['BMAJ']), float(ha['BMIN']), float(ha.get('BPA', 0.0))), thermnoi=ha.get('THERMNOI'),
                    imgrms=float(np.nanstd(inn[inn != 0])) if np.any(inn != 0) else None,
                    dewarped=True, aligned=True, alignref=os.path.basename(mI if 'I' in stokes else mV),
                    msub=True, msubfile=os.path.basename(mpath), lstbin=int(k), p2run=ha.get('P2RUN'),
                    scale_a=ha.get('FLUXDC'), extra=extra, validate=True)
                os.replace(sub_path + '.part', sub_path)
            for f in als:
                try:
                    os.remove(f)
                except OSError:
                    pass
            print(f'  [per-bin] {cell} {S} bin{k}: {len(imgs)} frames, LOO bin master N={len(imgs)}', flush=True)


def _write_bin_pb_maps(out_root, cell, lst, scope, hdr0, shape, master_path):
    """BIN PRIMARY-BEAM MAPS (contract step 61, Gregg 2026-09-12): for each 10-minute bin k the three-factor
    map (model PB x chromatic term x per-band residual, composed by pb_apply.wideband_pb with binidx=k, exactly
    as the deep PB_deep map but for the bin's own zenith-angle window) on the master grid, written to
    bin_masters/PB_bin{k}_{scope}.fits with the deep map's header cards, and archived beside the bin masters
    when archiving is on. These are the pbcorr siblings' input at 10 minutes. A missing library map REFUSES."""
    bd = f'{out_root}/{cell}/bin_masters'; os.makedirs(bd, exist_ok=True)
    _arun = os.environ.get('P3_ARCHIVE_RUN'); _aw = os.environ.get('P3_ARCHIVE_WRITE') != '0'
    _aroot = os.environ.get('P3_ARCHIVE_ROOT', '/lustre/pipeline/exopipe/phase3')
    pbhdr = None
    for k in range(NBINS):
        outp = f'{bd}/PB_bin{k}_{scope}.fits'
        if os.path.exists(outp):
            continue
        b, meta = PB.wideband_pb(scope, lst, master_path, binidx=k)
        if b is None:
            raise RuntimeError(f'{cell}: no bin-{k} PB map from the library - REFUSING (contract step 61)')
        if pbhdr is None:
            _, pbhdr = PB.subband_deep_pb(meta['members'][0])
        b = _pb_on_grid(b, pbhdr, hdr0, shape)
        ph = fits.getheader(master_path).copy()
        ph.pop('PBCORR', None)
        ph['PBLIBV'] = ('v1-2026-07-04', 'PB library version'); ph['PBMETHOD'] = ('time-integrated-library', 'canonical (2026-07-11)')
        ph['PBWINDOW'] = (f'10min-bin{k}', 'PB integration window'); ph['LSTBIN'] = (int(k), '10-minute bin')
        ph['PBMODEL'] = (meta.get('pb_model', 'v1-2026-07-04'), 'factor 1: model PB library version')
        if meta.get('chromatic'):
            ph['PBCHROM'] = (meta['chromatic'], 'factor 2: bandwidth-smearing term (a priori)')
        if meta.get('residual_pb'):
            ph['PBRESID'] = (meta['residual_pb'], 'factor 3: measured residual gain per band')
        fits.writeto(outp + '.part', b.astype(np.float32), ph, overwrite=True); os.replace(outp + '.part', outp)
        if _aw and _arun:
            import shutil as _sh
            _cd = f'{_aroot}/Coadd/{_arun}/{lst}/{scope}/10min'; os.makedirs(_cd, exist_ok=True)
            _sh.copy2(outp, f'{_cd}/PB_bin{k}_{scope}.fits.part'); os.replace(f'{_cd}/PB_bin{k}_{scope}.fits.part', f'{_cd}/PB_bin{k}_{scope}.fits')
    print(f'  [per-bin] {cell}: bin PB maps present for bins 0-{NBINS - 1} ({bd})', flush=True)


def build_cell_frames(lst, scope, out_root=None, stokes='IV', dates=None, inject=None,
                      overwrite=False, nproc=6):
    """Build the 10-minute Stage-1 tree for one cell (nights in parallel, fork pool).
    Returns the sorted list of frame keys built."""
    import multiprocessing as mp
    t0 = time.time()
    out_root = out_root or tenmin_root()
    base = f'{coadd_root()}/{lst}_{scope}'
    cell = f'{lst}_{scope}'
    od = f'{out_root}/{cell}/subtracted'; os.makedirs(od, exist_ok=True)
    mI = f'{base}/coadd_final/I_deep_Taper_R0_{scope}.fits'      # Robust 0: the frames' variant
    mV = f'{base}/coadd_final/V_deep_Taper_R0_{scope}.fits'
    masters = {}
    for S, mp_ in (('I', mI), ('V', mV)):
        if S not in stokes:
            continue
        if not os.path.exists(mp_):
            raise RuntimeError(f'{cell}: no Taper Robust 0 {S} master at {mp_} - REFUSING')
        h = fits.getheader(mp_)
        if h.get('COADDOP') != MASTER_GEN:
            raise RuntimeError(f'{cell}: {os.path.basename(mp_)} is COADDOP={h.get("COADDOP")!r}, '
                               f'required {MASTER_GEN!r} - REFUSING')
        a = np.squeeze(fits.getdata(mp_)).astype(np.float32)
        a[~np.isfinite(a)] = 0.0
        masters[S] = a
    hdr0 = fits.getheader(mI if 'I' in stokes else mV)
    shape = masters[next(iter(masters))].shape
    cd = abs(float(hdr0.get('CDELT1', hdr0.get('CD1_1'))))
    wcsv = f'{base}/coadd_final/weights.csv'
    if not os.path.exists(wcsv):
        raise RuntimeError(f'{cell}: {wcsv} missing - no per-night DC record; REFUSING')
    dc = {}
    for r in csv.DictReader(open(wcsv)):
        try:
            dc[r['date']] = float(r['qa3_scale'])
        except (KeyError, ValueError, TypeError):
            pass
    if PER_BIN:
        ratios, pbmeta = {k: (1.0, 0.0, 1.0) for k in range(6)}, {'residual_pb': ''}
    else:
        ratios, pbmeta = pb_ratio_maps(scope, lst, mI if 'I' in stokes else mV, hdr0, shape)
    freq_str = scope.replace('MHz', '')
    excl = load_exclusions()
    nights = [x for x in C.discover_dates(lst, freq_str) if not is_excluded(lst, scope, x[0], excl)]
    if dates:
        nights = [x for x in nights if x[0] in dates]
    inj_by_key = {}
    for it in (inject or []):
        inj_by_key.setdefault(it['key'], []).append(it)
    _CTX.update(dict(lst=lst, scope=scope, stokes=stokes, base=base, od=od, shape=shape, hdr0=hdr0,
                     cd=cd, masters=masters, ratios=ratios, pbmeta=pbmeta, dc=dc,
                     inj_by_key=inj_by_key, overwrite=overwrite, mI=mI, mV=mV))
    if nproc > 1 and len(nights) > 1:
        with mp.get_context('fork').Pool(min(nproc, len(nights))) as pool:
            results = pool.map(_build_night, nights)
    else:
        results = [_build_night(n) for n in nights]
    if PER_BIN:
        gate_refused = {}
        for _r0, _i, r, _b in results:
            for label, why in r:
                if why.startswith('divergent input'):
                    key, S = label.split()[:2]
                    gate_refused.setdefault((S, int(key.split('T')[1])), []).append((key, why))
        _per_bin_loo_subtract(out_root, cell, lst, scope, stokes, hdr0, shape, cd, base, mI, mV, overwrite, gate_refused)
        _write_bin_pb_maps(out_root, cell, lst, scope, hdr0, shape, mI if 'I' in stokes else mV)
    frames_csv = f'{out_root}/{cell}/frames.csv'
    inj_csv = f'{out_root}/{cell}/injections.csv'
    new_frames = not os.path.exists(frames_csv)
    with open(frames_csv, 'a', newline='') as f:
        fw = csv.writer(f, lineterminator='\n')
        if new_frames:
            fw.writerow(['key', 'date', 'bin', 'stokes', 'src', 'dc', 'has_flow', 'src_dewarped', 'pb_blank_frac', 'pb_ratio_median',
                         'bmaj_deg', 'bmin_deg', 'bpa_deg', 'n_inject', 'sub_path'])
        for rows, _i, _r, _b in results:
            fw.writerows(rows)
    with open(inj_csv, 'a', newline='') as f:
        iw = csv.writer(f, lineterminator='\n')
        if os.path.getsize(inj_csv) == 0:
            iw.writerow(['key', 'stokes', 'ra', 'dec', 'flux_jy', 'x', 'y', 'sub_path'])
        for _r0, irows, _r, _b in results:
            iw.writerows(irows)
    refused = [x for _r0, _i, r, _b in results for x in r]
    built = [x for _r0, _i, _r, b in results for x in b]
    with open(f'{out_root}/{cell}/refusals.csv', 'a', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows([[a, b] for a, b in refused])
    nf = sum(1 for r in csv.DictReader(open(frames_csv)) if r.get('has_flow') == '0')
    print(f'  [tenmin_frames] {cell}: {len(built)} frames ({len({k for k, _ in built})} keys; {nf} without a night flow), '
          f'{len(refused)} refusals, master {MASTER_GEN}, PB floor {PB_FLOOR}, '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)
    return sorted({k for k, _ in built})


if __name__ == '__main__':
    lst, scope = sys.argv[1], sys.argv[2]
    st = sys.argv[3] if len(sys.argv) > 3 else 'IV'
    build_cell_frames(lst, scope, stokes=st)
