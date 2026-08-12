#!/usr/bin/env python3
"""HWJ snapshot-tier forced photometry (tier 5; runs on lwacalim02).
Gregg spec 2026-07-27: search-only, frames written ONLY for >=5-sigma;
V (Pilot-Snapshot, dirty) priority then I (Clean-Snapshot); no sky
subtraction (I noise ~800 mJy/image); deep dewarp applied in POSITION
space from the archived warp screens.

Work unit = (lst, date, subband, stokes):
  stage {SB}/snapshots[_clean]/*.fits.fz from the phase1 run to local
  scratch -> funpack -> per snapshot, per planet at el>=20: shift the
  nominal pixel by the epoch/subband deep warp screen (SCREEN_X/Y at the
  planet pixel; screens map original->dewarped so the original-frame
  position = nominal + screen), forced photometry (fixed pos + peak in
  1 beam; SNR vs 1.4826*MAD of 2-6 beam-radius annulus). pb_at_pos from
  the time-integrated subband PB library; dual flux columns.
  |snr| >= 5 -> 5x5 deg cutout into the unit's detections dir.
Results land under RESULTS/{unit}/ locally (02); sync_results.sh merges
into the 07 run root as {obj}/{S}/snapshot/{band}/.
Resumable: RESULTS/{unit}.done markers.

Usage: phase3_hwj_snap.py V|I [--nproc 24] [--lst 05h] [--limit N]
"""
import os, re, sys, csv, glob, shutil, argparse, subprocess, warnings
from multiprocessing import Pool

import numpy as np
warnings.filterwarnings('ignore')
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
from astropy import units as u

P1 = '/lustre/pipeline/exopipe/phase1'
P2 = '/lustre/pipeline/exopipe/phase2'
SCRATCH = '/fast/gh/main/phase3/snap_stage'
RESULTS = '/fast/gh/main/phase3/snap_phot_results'
FUNPACK = '/opt/devel/pipeline/envs/development/bin/funpack'
TARGETS_CSV = '/lustre/gh/main/clv15/Hot_Warm_Jupiters.csv'
OVRO = EarthLocation(lat=37.23977727*u.deg, lon=-118.2816667*u.deg,
                     height=1222*u.m)
MIN_EL = 20.0
DET_SNR = 5.0
CUT_DEG = 5.0
SNAPDIR = {'V': 'snapshots', 'I': 'snapshots_clean'}

COLS = ['obj', 'name', 'stokes', 'band', 'lst', 'date', 'mjd', 'el_deg',
        'src_jy', 'peak_jy', 'src_snr', 'peak_snr', 'rms_jy', 'pb_at_pos',
        'src_jy_pbcorr', 'warp_dx_pix', 'warp_dy_pix', 'dewarped', 'image']


def load_targets():
    out = []
    for i, r in enumerate(csv.DictReader(open(TARGETS_CSV,
                                              encoding='utf-8-sig')), 1):
        try:
            ra, dec = float(r['ra_current']), float(r['dec_current'])
        except (KeyError, ValueError):
            continue
        name = r['common_name'].strip()
        out.append({'obj': f'hwj{i:03d}_' + re.sub(r'[\s/]+', '_', name),
                    'name': name, 'ra': ra, 'dec': dec,
                    'sc': SkyCoord(ra*u.deg, dec*u.deg)})
    return out


def find_units(stokes, lst_filter=None):
    units = []
    for ld in sorted(glob.glob(f'{P1}/[0-2][0-9]h')):
        lst = os.path.basename(ld)
        if lst_filter and lst != lst_filter:
            continue
        for dd in sorted(glob.glob(f'{ld}/2*-*-*')):
            date = os.path.basename(dd)
            runs = sorted(glob.glob(f'{dd}/Run_*'), key=os.path.getmtime)
            if not runs:
                continue
            run = runs[-1]
            for sd in sorted(glob.glob(f'{run}/*MHz/{SNAPDIR[stokes]}')):
                sb = sd.split('/')[-2]
                if glob.glob(f'{sd}/*-image.fits*'):
                    units.append({'lst': lst, 'date': date, 'sb': sb,
                                  'stokes': stokes, 'src': sd})
    return units


def warp_screen_path(lst, date, sb):
    hits = glob.glob(f'{P2}/{lst}/{date}/Science_*/Dewarp_Diagnostics/'
                     f'maps/{date}_{lst}_{sb}_warp_screen.fits')
    return sorted(hits)[-1] if hits else None


def measure(data, wcs, sc, hdr, shift):
    x, y = wcs.world_to_pixel(sc)
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    xi = int(round(float(x) + shift[0]))
    yi = int(round(float(y) + shift[1]))
    ny, nx = data.shape
    if not (0 <= xi < nx and 0 <= yi < ny) or not np.isfinite(data[yi, xi]):
        return None
    try:
        bp = max(2.0, abs(hdr['BMAJ']) / abs(hdr['CDELT1']))
    except (KeyError, ZeroDivisionError):
        bp = 6.0
    r1, r2 = int(2 * bp), int(6 * bp)
    sl = np.s_[max(0, yi - r2):min(ny, yi + r2 + 1),
               max(0, xi - r2):min(nx, xi + r2 + 1)]
    yy, xx = np.ogrid[sl]
    rr = np.hypot(yy - yi, xx - xi)
    ann = data[sl][(rr >= r1) & (rr <= r2)]
    ann = ann[np.isfinite(ann) & (ann != 0)]
    if ann.size < 50:
        return None
    med = float(np.median(ann))
    mad = float(1.4826 * np.median(np.abs(ann - med)))
    if mad <= 0:
        return None
    rp = int(np.ceil(bp))
    st = data[max(0, yi - rp):min(ny, yi + rp + 1),
              max(0, xi - rp):min(nx, xi + rp + 1)]
    stf = np.where(np.isfinite(st), st, med)
    peak = float(stf[np.unravel_index(np.argmax(np.abs(stf - med)),
                                      stf.shape)])
    src = float(data[yi, xi])
    return {'src': src, 'peak': peak, 'rms': mad,
            'src_snr': (src - med) / mad, 'peak_snr': (peak - med) / mad,
            'xi': xi, 'yi': yi}


def do_unit(task):
    unit = f"{task['lst']}_{task['date']}_{task['sb']}_{task['stokes']}"
    mark = f'{RESULTS}/{unit}.done'
    if os.path.exists(mark):
        return unit, 0, 0
    targets = task['targets']
    band = task['sb']
    stage = f'{SCRATCH}/{unit}'
    os.makedirs(stage, exist_ok=True)
    udir = f'{RESULTS}/{unit}'
    os.makedirs(udir, exist_ok=True)
    # warp screen (deep) -> per-planet pixel shift, computed once
    wsp = warp_screen_path(task['lst'], task['date'], task['sb'])
    shifts = {}
    dewarped = 0
    if wsp:
        try:
            with fits.open(wsp) as hd:
                sx = hd['SCREEN_X'].data
                sy = hd['SCREEN_Y'].data
                swcs = WCS(hd['SCREEN_X'].header).celestial
            for tg in targets:
                px, py = swcs.world_to_pixel(tg['sc'])
                if np.isfinite(px) and np.isfinite(py):
                    xi = int(np.clip(round(float(px)), 0, sx.shape[1]-1))
                    yi = int(np.clip(round(float(py)), 0, sx.shape[0]-1))
                    dx, dy = float(sx[yi, xi]), float(sy[yi, xi])
                    if np.isfinite(dx) and np.isfinite(dy):
                        shifts[tg['obj']] = (dx, dy)
            dewarped = 1
        except Exception:
            shifts = {}
    # PB at position (time-integrated subband deep PB)
    from exopipe.phase3 import pb_apply as PB
    pb_at = {}
    try:
        pbd, pbh = PB.subband_deep_pb(task['sb'])
        pw = WCS(pbh).celestial
        pbd = np.squeeze(pbd)
        for tg in targets:
            px, py = pw.world_to_pixel(tg['sc'])
            if np.isfinite(px) and np.isfinite(py):
                xi = int(np.clip(round(float(px)), 0, pbd.shape[1]-1))
                yi = int(np.clip(round(float(py)), 0, pbd.shape[0]-1))
                v = float(pbd[yi, xi])
                if np.isfinite(v) and v > 0.05:
                    pb_at[tg['obj']] = v
    except Exception:
        pass
    rows, ndet = [], 0
    aa_cache = {}
    # snapshots are .fits.fz / .fits.fs (both fpack; py38 astropy cannot
    # decompress the 4-D CompImageHDU, so funpack) or plain .fits
    for fz in sorted(glob.glob(f"{task['src']}/*-image.fits*")):
        loc = f'{stage}/{os.path.basename(fz)}'
        raw = re.sub(r'\.f[zs]$', '', loc)
        try:
            shutil.copy(fz, loc)
            if raw != loc:
                subprocess.run([FUNPACK, '-O', raw, loc], check=True,
                               capture_output=True)
            with fits.open(raw) as hd:
                h0 = next(h for h in hd if h.data is not None)
                data = np.squeeze(h0.data).astype(np.float32)
                hdr = h0.header
            w = WCS(hdr).celestial
            t = Time(hdr['DATE-OBS'])
            aa = AltAz(obstime=t, location=OVRO)
            for tg in targets:
                el = float(tg['sc'].transform_to(aa).alt.deg)
                if el < MIN_EL:
                    continue
                sh = shifts.get(tg['obj'], (0.0, 0.0))
                m = measure(data, w, tg['sc'], hdr, sh)
                if m is None:
                    continue
                pb = pb_at.get(tg['obj'], '')
                is_det = abs(m['peak_snr']) >= DET_SNR
                rows.append([tg['obj'], tg['name'], task['stokes'], band,
                             task['lst'], task['date'],
                             round(float(t.mjd), 7), round(el, 2),
                             m['src'], m['peak'], round(m['src_snr'], 2),
                             round(m['peak_snr'], 2), m['rms'], pb,
                             (m['src']/pb if pb else ''),
                             round(sh[0], 2), round(sh[1], 2),
                             dewarped, os.path.basename(raw)])
                if is_det:
                    ndet += 1
                    ddir = f'{udir}/detections/{tg["obj"]}'
                    os.makedirs(ddir, exist_ok=True)
                    try:
                        co = Cutout2D(data, tg['sc'],
                                      (CUT_DEG*u.deg, CUT_DEG*u.deg),
                                      wcs=w, mode='partial',
                                      fill_value=np.nan)
                        ch = hdr.copy()
                        ch.update(co.wcs.to_header())
                        from exopipe.phase3.phase3_hwj_phot import strip_degenerate_axes
                        strip_degenerate_axes(ch)
                        ch['TARGET'] = tg['name']
                        ch['ELEV'] = el
                        ch['CUTDEG'] = CUT_DEG
                        if pb:
                            ch['PBATPOS'] = pb
                        fits.PrimaryHDU(co.data, header=ch).writeto(
                            f'{ddir}/{tg["obj"]}_'
                            f'{os.path.basename(raw)}', overwrite=True)
                    except Exception:
                        pass
        except Exception:
            pass
        finally:
            for f in (loc, raw):
                if os.path.exists(f):
                    os.remove(f)
    with open(f'{udir}/photometry.csv', 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(COLS)
        wcsv.writerows(rows)
    shutil.rmtree(stage, ignore_errors=True)
    open(mark, 'w').close()
    return unit, len(rows), ndet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stokes', choices=['V', 'I'])
    ap.add_argument('--nproc', type=int, default=24)
    ap.add_argument('--lst', default=None)
    ap.add_argument('--date', default=None)
    ap.add_argument('--sb', default=None)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    targets = load_targets()
    units = find_units(args.stokes, args.lst)
    if args.date:
        units = [u for u in units if u['date'] == args.date]
    if args.sb:
        units = [u for u in units if u['sb'] == args.sb]
    todo = [u for u in units if not os.path.exists(
        f"{RESULTS}/{u['lst']}_{u['date']}_{u['sb']}_{u['stokes']}.done")]
    if args.limit:
        todo = todo[:args.limit]
    for u in todo:
        u['targets'] = targets
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(SCRATCH, exist_ok=True)
    print(f'{args.stokes}: {len(units)} units, {len(todo)} to do, '
          f'{len(targets)} targets', flush=True)
    n_rows = n_det = n_done = 0
    with Pool(args.nproc) as pool:
        for unit, nr, nd in pool.imap_unordered(do_unit, todo):
            n_rows += nr; n_det += nd; n_done += 1
            print(f'{unit}: {nr} rows, {nd} det '
                  f'({n_done}/{len(todo)})', flush=True)
    print(f'SNAP {args.stokes} DONE: {n_rows} rows, {n_det} detections',
          flush=True)


if __name__ == '__main__':
    main()
