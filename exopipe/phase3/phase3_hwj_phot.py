#!/usr/bin/env python3
"""HWJ Phase-3 forced photometry + cutouts over the pbcorr residual archive
(Gregg spec 2026-07-27).

Tiers (run in order): wb_deep -> sb_deep -> wb_10min -> sb_10min.
Within each tier V first, then I. Sources = time-integrated-PB-corrected
residuals on Lustre:
  wb_deep  V: v_resid/{cell}/*Robust+0.0_pbcorr*_msub_ssub_*   (clean)
           I: i_resid/{cell}/*Robust-0.75_pbcorr*_msub_ssub_*  (full chain)
  sb_deep    : {lst}/{date}/P3_PROD/{sb}/{S}/deep/*_pbcorr_*msub*
  wb_10min   : {lst}/{date}/P3_PROD10/{band}/{S}/10min/*_pbcorr_*msub*
  sb_10min   : {lst}/{date}/P3_SB10/{sb}/{S}/10min/*_pbcorr_*msub*

Targets: Hot_Warm_Jupiters.csv (CSV order -> hwjNNN ids). Gate: elevation
>= 20 deg at image midpoint AND finite pbcorr pixel at position.
Per (image, planet): 5x5 deg cutout ->
  {ROOT}/{obj}/{V|I}/{deep|10min}/{band}/cutouts/
forced photometry (fixed-pos + peak within 1 beam; SNR vs 1.4826*MAD of
2-6 beam-radius annulus); imfit Gaussian fit when |peak snr| >= 3; rows to
photometry.csv per band dir; |snr| >= 5 -> frame copy + row into
detections/. data_decisions exclusions carried as excluded=1 (measured,
flagged). Resumable via .progress/{tier}_{stokes}_{cell}_{date}.done.

Usage: phase3_hwj_phot.py TIER STOKES [--nproc N] [--lst 05h] [--limit N]
  TIER in wb_deep sb_deep wb_10min sb_10min; STOKES in V I
"""
import os, re, sys, csv, glob, argparse, warnings
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
warnings.filterwarnings('ignore')
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
from astropy import units as u

LU = '/lustre/pipeline/exopipe/phase3'
ROOT = '/data07/gh/phase3/photometry/Run_HWJ_PHOT1'
TARGETS_CSV = '/lustre/gh/main/clv15/Hot_Warm_Jupiters.csv'
DECISIONS = '/lustre/gh/main/clv15/data_decisions.csv'
OVRO = EarthLocation(lat=37.23977727*u.deg, lon=-118.2816667*u.deg,
                     height=1222*u.m)
MIN_EL = 20.0
CUT_DEG = 5.0
FIT_SNR = 3.0
DET_SNR = 5.0
WIDE = {'Red', 'Green', 'Blue', 'Full'}

TIER_GLOBS = {
    ('wb_deep', 'V'):  f'{LU}/v_resid/*_*/*Robust+0.0_pbcorr*_msub_ssub_*.fits',
    ('wb_deep', 'I'):  f'{LU}/i_resid/*_*/*Robust-0.75_pbcorr*_msub_ssub_*.fits',
    ('sb_deep', 'V'):  f'{LU}/[0-2][0-9]h/*/P3_PROD/*MHz/V/deep/*_pbcorr_*msub*.fits',
    ('sb_deep', 'I'):  f'{LU}/[0-2][0-9]h/*/P3_PROD/*MHz/I/deep/*_pbcorr_*msub*.fits',
    ('wb_10min', 'V'): f'{LU}/[0-2][0-9]h/*/P3_PROD10/*/V/10min/*_pbcorr_*msub*.fits',
    ('wb_10min', 'I'): f'{LU}/[0-2][0-9]h/*/P3_PROD10/*/I/10min/*_pbcorr_*msub*.fits',
    ('sb_10min', 'V'): f'{LU}/[0-2][0-9]h/*/P3_SB10/*MHz/V/10min/*_pbcorr_*msub*.fits',
    ('sb_10min', 'I'): f'{LU}/[0-2][0-9]h/*/P3_SB10/*MHz/I/10min/*_pbcorr_*msub*.fits',
}

PHOT_COLS = ['obj', 'name', 'stokes', 'cadence', 'band', 'lst', 'date',
             'bin', 'mjd_mid', 'el_deg', 'src_jy', 'peak_jy', 'peak_snr',
             'src_snr', 'rms_jy', 'fit_jy', 'fit_err_jy', 'fit_ra',
             'fit_dec', 'fit_offset_arcmin', 'excluded', 'image']


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


WB_SUBBANDS = {'Red': ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz'),
               'Green': ('41MHz', '46MHz', '50MHz', '55MHz', '59MHz'),
               'Blue': ('64MHz', '69MHz', '73MHz', '78MHz', '82MHz')}
WB_SUBBANDS['Full'] = (WB_SUBBANDS['Red'] + WB_SUBBANDS['Green']
                       + WB_SUBBANDS['Blue'])


# Mirrors coadd_full_pipeline.SUBBAND_ONLY_EXCL (Gregg 2026-06-29): the
# only per-subband carve-out; its exclusion does not make the night bad.
SUBBAND_ONLY_EXCL = {'18MHz'}


def load_exclusions():
    """data_decisions.csv exclusions with the RATIFIED production semantics
    (coadd_full_pipeline.is_excluded / is_whole_night_excluded, Gregg
    2026-06-29): returns {'triples': {(lst, date, subband)},
    'whole_night': {(lst, date)}} where whole_night = any excluded subband
    beyond the 18MHz carve-out."""
    triples, whole = set(), set()
    if os.path.exists(DECISIONS):
        for r in csv.DictReader(open(DECISIONS)):
            lst = (r.get('lst') or r.get('LST') or '').strip()
            date = (r.get('date') or r.get('DATE') or '').strip()
            sb = (r.get('subband') or '').strip()
            if lst and date and sb:
                triples.add((lst, date, sb))
                if sb not in SUBBAND_ONLY_EXCL:
                    whole.add((lst, date))
    return {'triples': triples, 'whole_night': whole}


def is_excluded(lst, date, band, exc):
    """Ratified semantics (Gregg 2026-06-29, coadd_full_pipeline):
    wideband image excluded iff the night is whole-night excluded (18MHz-
    only nights keep wideband — accepted contamination); subband image
    excluded iff its exact (lst, date, subband) is listed."""
    if band in WB_SUBBANDS:
        return int((lst, date) in exc['whole_night'])
    return int((lst, date, band) in exc['triples'])


def strip_degenerate_axes(hdr):
    """Remove 3rd/4th-axis WCS keywords inherited from the 4-D parent —
    on 2-D cutout data they make astropy build a 4-axis WCS and most
    viewers (DS9/CARTA) then show no WCS at all."""
    for ax in ('3', '4'):
        for base in ('CTYPE', 'CRVAL', 'CRPIX', 'CDELT', 'CUNIT',
                     'CROTA', 'NAXIS'):
            hdr.remove(f'{base}{ax}', ignore_missing=True)
        for other in ('1', '2', '3', '4'):
            hdr.remove(f'PC{ax}_{other}', ignore_missing=True)
            hdr.remove(f'PC{other}_{ax}', ignore_missing=True)
            hdr.remove(f'CD{ax}_{other}', ignore_missing=True)
            hdr.remove(f'CD{other}_{ax}', ignore_missing=True)
    if hdr.get('WCSAXES', 2) > 2:
        hdr['WCSAXES'] = 2
    return hdr


def parse_image_path(path, tier):
    b = os.path.basename(path)
    lst = re.search(r'LST(\d\dh)', b).group(1)
    band = b.split('_')[0]
    mbin = re.search(r'_bin(\d+)_', b)
    ts = re.search(r'(\d{8}T\d{6})', b)
    if tier in ('wb_deep',):
        cell = path.split('/')[-2]
        date_grp = ts.group(1)[:8]
        date = f'{date_grp[:4]}-{date_grp[4:6]}-{date_grp[6:]}'
    else:
        parts = path.replace(LU + '/', '').split('/')
        date = parts[1]
    return {'lst': lst, 'band': band, 'date': date,
            'bin': mbin.group(1) if mbin else '',
            'ts': ts.group(1) if ts else ''}


def measure_at(data, wcs, sc, hdr):
    """Forced photometry at sc: fixed-pos value + peak within 1 beam,
    SNR vs annulus MAD (2-6 beam radii). Returns dict or None (off image /
    NaN at position)."""
    x, y = wcs.world_to_pixel(sc)
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    xi, yi = int(round(float(x))), int(round(float(y)))
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
    src = float(data[yi, xi])
    rp = int(np.ceil(bp))
    st = data[max(0, yi - rp):min(ny, yi + rp + 1),
              max(0, xi - rp):min(nx, xi + rp + 1)]
    stf = np.where(np.isfinite(st), st, med)
    pk_i = np.unravel_index(np.argmax(np.abs(stf - med)), stf.shape)
    peak = float(stf[pk_i])
    return {'src': src, 'peak': peak, 'rms': mad,
            'src_snr': (src - med) / mad, 'peak_snr': (peak - med) / mad,
            'beam_pix': bp}


def imfit_cutout(fits_path, box_pix):
    """CASA imfit on the cutout around the peak; returns (flux, err, ra,
    dec) or Nones."""
    try:
        from casatasks import imfit
        x1, y1, x2, y2 = box_pix
        fit = imfit(imagename=fits_path, box=f'{x1},{y1},{x2},{y2}')
        comp = fit['results']['component0']
        flux = comp['flux']['value'][0]
        ferr = comp['flux']['error'][0]
        d = comp['shape']['direction']
        ra = np.degrees(d['m0']['value']) % 360.0
        dec = np.degrees(d['m1']['value'])
        return flux, ferr, ra, dec
    except Exception:
        return None, None, None, None


def do_group(task):
    """One (tier, stokes, cell-date group): all images sharing an LST/date
    for one band. Returns rows + ndet."""
    tier, stokes, key, paths, targets, excluded = (
        task['tier'], task['stokes'], task['key'], task['paths'],
        task['targets'], task['excluded'])
    cadence = 'deep' if 'deep' in tier else '10min'
    rows, ndet = [], 0
    for path in paths:
        try:
            with fits.open(path) as hd:
                data = np.squeeze(hd[0].data).astype(np.float32)
                hdr = hd[0].header
        except Exception:
            continue
        w = WCS(hdr).celestial
        meta = parse_image_path(path, tier)
        mjd = hdr.get('MJD-MID')
        if mjd is None:
            try:
                mjd = Time(hdr['DATE-OBS']).mjd + hdr.get('DURATION', 0)/172800.
            except Exception:
                continue
        t = Time(mjd, format='mjd')
        aa = AltAz(obstime=t, location=OVRO)
        for tg in targets:
            el = float(tg['sc'].transform_to(aa).alt.deg)
            if el < MIN_EL:
                continue
            m = measure_at(data, w, tg['sc'], hdr)
            if m is None:
                continue
            odir = os.path.join(ROOT, tg['obj'], stokes, cadence,
                                meta['band'])
            cdir = os.path.join(odir, 'cutouts')
            ddir = os.path.join(odir, 'detections')
            os.makedirs(cdir, exist_ok=True)
            cut_name = f"{tg['obj']}_{os.path.basename(path)}"
            cut_path = os.path.join(cdir, cut_name)
            fit = (None, None, None, None)
            try:
                co = Cutout2D(data, tg['sc'], (CUT_DEG*u.deg, CUT_DEG*u.deg),
                              wcs=w, mode='partial', fill_value=np.nan)
                ch = hdr.copy()
                ch.update(co.wcs.to_header())
                strip_degenerate_axes(ch)
                ch['TARGET'] = tg['name']
                ch['TGRA'], ch['TGDEC'] = tg['ra'], tg['dec']
                ch['CUTDEG'] = CUT_DEG
                ch['ELEV'] = (el, '[deg] target elevation at image midpoint')
                fits.PrimaryHDU(co.data, header=ch).writeto(cut_path,
                                                            overwrite=True)
                if abs(m['peak_snr']) >= FIT_SNR:
                    cx, cy = co.wcs.world_to_pixel(tg['sc'])
                    bp = int(np.ceil(m['beam_pix']))
                    box = (max(0, int(cx)-2*bp), max(0, int(cy)-2*bp),
                           min(co.data.shape[1]-1, int(cx)+2*bp),
                           min(co.data.shape[0]-1, int(cy)+2*bp))
                    src_fit = cut_path
                    if stokes == 'V' and m['peak_snr'] < 0:
                        neg = cut_path.replace('.fits', '.neg.fits')
                        fits.PrimaryHDU(-co.data, header=ch).writeto(
                            neg, overwrite=True)
                        src_fit = neg
                    fit = imfit_cutout(src_fit, box)
                    if src_fit != cut_path and os.path.exists(src_fit):
                        os.remove(src_fit)
                    if fit[0] is not None and stokes == 'V' \
                            and m['peak_snr'] < 0:
                        fit = (-fit[0], fit[1], fit[2], fit[3])
            except Exception:
                pass
            off = ''
            if fit[2] is not None:
                off = float(SkyCoord(fit[2]*u.deg, fit[3]*u.deg)
                            .separation(tg['sc']).arcmin)
            is_det = abs(m['peak_snr']) >= DET_SNR
            if is_det:
                os.makedirs(ddir, exist_ok=True)
                try:
                    os.link(cut_path, os.path.join(ddir, cut_name))
                except OSError:
                    import shutil
                    shutil.copy2(cut_path, os.path.join(ddir, cut_name))
                ndet += 1
            rows.append([tg['obj'], tg['name'], stokes, cadence,
                         meta['band'], meta['lst'], meta['date'],
                         meta['bin'], round(float(mjd), 6), round(el, 2),
                         m['src'], m['peak'], round(m['peak_snr'], 2),
                         round(m['src_snr'], 2), m['rms'],
                         fit[0], fit[1], fit[2], fit[3], off,
                         is_excluded(meta['lst'], meta['date'],
                                     meta['band'], excluded),
                         os.path.basename(path)])
    return task['key'], rows, ndet


def flush_rows(all_rows):
    by_dir = defaultdict(list)
    for r in all_rows:
        d = os.path.join(ROOT, r[0], r[2], r[3], r[4])
        by_dir[d].append(r)
    for d, rs in by_dir.items():
        os.makedirs(d, exist_ok=True)
        pcsv = os.path.join(d, 'photometry.csv')
        new = not os.path.exists(pcsv)
        with open(pcsv, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(PHOT_COLS)
            w.writerows(rs)
        dets = [r for r in rs if abs(r[12]) >= DET_SNR]
        if dets:
            dd = os.path.join(d, 'detections')
            os.makedirs(dd, exist_ok=True)
            dcsv = os.path.join(dd, 'detections.csv')
            new = not os.path.exists(dcsv)
            with open(dcsv, 'a', newline='') as f:
                w = csv.writer(f)
                if new:
                    w.writerow(PHOT_COLS)
                w.writerows(dets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tier', choices=['wb_deep', 'sb_deep', 'wb_10min',
                                     'sb_10min'])
    ap.add_argument('stokes', choices=['V', 'I'])
    ap.add_argument('--nproc', type=int, default=12)
    ap.add_argument('--lst', default=None)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    targets = load_targets()
    excluded = load_exclusions()
    prog = os.path.join(ROOT, '.progress')
    os.makedirs(prog, exist_ok=True)

    paths = sorted(glob.glob(TIER_GLOBS[(args.tier, args.stokes)]))
    groups = defaultdict(list)
    for p in paths:
        meta = parse_image_path(p, args.tier)
        if args.lst and meta['lst'] != args.lst:
            continue
        groups[f"{meta['lst']}_{meta['band']}_{meta['date']}"].append(p)
    todo = []
    for key, ps in sorted(groups.items()):
        mark = os.path.join(prog, f'{args.tier}_{args.stokes}_{key}.done')
        if os.path.exists(mark):
            continue
        todo.append({'tier': args.tier, 'stokes': args.stokes, 'key': key,
                     'paths': ps, 'targets': targets, 'excluded': excluded})
    if args.limit:
        todo = todo[:args.limit]
    print(f'{args.tier}/{args.stokes}: {len(paths)} images, '
          f'{len(groups)} groups, {len(todo)} to do '
          f'({len(targets)} targets)', flush=True)
    if not todo:
        return
    n_rows = n_det = 0
    with Pool(args.nproc) as pool:
        for key, rows, ndet in pool.imap_unordered(do_group, todo):
            flush_rows(rows)
            open(os.path.join(
                prog, f'{args.tier}_{args.stokes}_{key}.done'), 'w').close()
            n_rows += len(rows); n_det += ndet
            print(f'{key}: {len(rows)} rows, {ndet} det '
                  f'(tot {n_rows}/{n_det})', flush=True)
    print(f'TIER DONE {args.tier} {args.stokes}: {n_rows} measurements, '
          f'{n_det} detections', flush=True)


if __name__ == '__main__':
    main()
