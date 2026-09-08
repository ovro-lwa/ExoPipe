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
flow, a DC record, or a Phase 2 10-minute product; PB map unavailable.
"""
import os, sys, csv, glob, re, json, time
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from exopipe.common.roots import coadd_root, required_root
from exopipe.common import phase3_products as pp
from exopipe.common import phase2_config as cfg
from exopipe.phase3 import coadd_full_pipeline as C
from exopipe.phase3 import pb_apply as PB
from exopipe.phase3.coadd_full_pipeline import load_exclusions, is_excluded

PB_FLOOR = float(os.environ.get('P3_TENMIN_PB_FLOOR', str(PB.PB_FLOOR)))   # 0.1 as pbcorr
BIN_SEC = 600.0
NBINS = 6
MASTER_GEN = os.environ.get('P3_MASTER_GEN', 'ivw-mean-v3')


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
    """Phase 2 10-minute Taper Robust 0 dewarped, non-pbcorr images with their DATE-OBS."""
    out = []
    for p, _i in C.find_10min_images(sci_dir, scope.replace('MHz', ''), stokes):
        try:
            h = fits.getheader(p)
        except Exception:
            continue
        t = h.get('DATE-OBS')
        if not t:
            continue
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
        for mjd, src, sh in files:
            k = _bin_of(mjd, anchor)
            if k is None:
                refused.append((f'{date} {S} {os.path.basename(src)}', 'DATE-OBS outside the six bins')); continue
            key = frame_key(date, k)
            sub_path = f'{od}/{key}_{S}_sub_{scope}.fits'
            if os.path.exists(sub_path) and not overwrite:
                built.append((key, S)); continue
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
            fr = np.squeeze(fits.getdata(tmp)).astype(np.float32); os.remove(tmp)
            fr[~np.isfinite(fr)] = 0.0
            r, blank, rmed = ratios[k]
            fr = fr * r                                   # step 1: PB ratio (0 below the floor)
            bmaj, bmin, bpa = (float(sh.get('BMAJ') or 0), float(sh.get('BMIN') or 0),
                               float(sh.get('BPA') or 0.0))
            if not (bmaj > 0 and bmin > 0):
                refused.append((f'{key} {S}', 'no beam in the Phase 2 header')); continue
            ninj = 0
            for it in inj_by_key.get(key, []):
                if it.get('stokes', S) != S:
                    continue
                w = WCS(hdr0).celestial; x, y = w.world_to_pixel_values(it['ra'], it['dec'])
                if _gauss_inject(fr, hdr0, cd, it['ra'], it['dec'], it['flux_jy'], bmaj, bmin, bpa):
                    ninj += 1
                    irows.append([key, S, it['ra'], it['dec'], it['flux_jy'], round(float(x), 2),
                                  round(float(y), 2), sub_path])
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
                     'SRCFRAME': (os.path.basename(src)[:68], 'Phase 2 10-minute source image')}
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
            rows.append([key, date, k, S, src, round(dc[date], 5), int(has_flow), round(blank, 4), round(rmed, 4),
                         bmaj, bmin, bpa, ninj, sub_path])
            built.append((key, S))
    return rows, irows, refused, built


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
    frames_csv = f'{out_root}/{cell}/frames.csv'
    inj_csv = f'{out_root}/{cell}/injections.csv'
    new_frames = not os.path.exists(frames_csv)
    with open(frames_csv, 'a', newline='') as f:
        fw = csv.writer(f, lineterminator='\n')
        if new_frames:
            fw.writerow(['key', 'date', 'bin', 'stokes', 'src', 'dc', 'has_flow', 'pb_blank_frac', 'pb_ratio_median',
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
