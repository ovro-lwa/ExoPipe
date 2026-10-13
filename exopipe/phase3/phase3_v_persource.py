#!/usr/bin/env python3
"""Rebuild the Stokes-V final list in the Stokes-I convention (Gregg
2026-07-11): ONE row and ONE candidate directory per source.

- V_deep_FINAL_transients.csv becomes per-source (best-SNR detection is the
  rank row; n_det / n_epochs / repeat_lst_dates summarize repeats; adds
  review_dir). The full per-detection table is preserved as
  V_deep_all_detections.csv (plus a .bak of the original).
- Cross-match recomputed with the I-convention MEASURED radii
  (COADD_ASTROMETRIC_ERROR.md: base B3/G3.5/R5(/F5)' + sigma_stat in
  quadrature, x1.5 below el 40 deg) — replaces the old nu-scaled radii the
  V assembler still used (Red 8.1' was rejected for I as admitting chance
  matches). NEW: n_xmatch + xmatch_all record EVERY sample object inside
  the radius (nearest-only was silently hiding multiples).
- candidates/ dirs merged/renamed accordingly (477 detection dirs -> one
  per source), rank prefixes on the 6-frame files updated; all moves
  logged; nothing deleted.
"""
import sys
_ARGV = list(sys.argv); sys.argv = ['vpersource']
import os, csv, glob, re, shutil
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from collections import defaultdict

from exopipe.phase3.lib.catalog import xcat as fcsv  # load_xcat, CATS
sys.argv = _ARGV              # restore after import-protection clobber

RF = '/data07/gh/phase3/transients/Run_FINAL'
CD = f'{RF}/candidates'
CSV = f'{RF}/V_deep_FINAL_transients.csv'
ALLDET = f'{RF}/V_deep_all_detections.csv'
BAK = f'{RF}/V_deep_FINAL_transients.csv.bak_perdet_20260711'
LOG = f'{RF}/candidates_rename_log_20260711.csv'

# I-convention measured cross-match radii (phase3_i_deep_final)
XR_BASE = {'Blue': 3.0, 'Green': 3.5, 'Red': 5.0, 'Full': 5.0}
XR_BEAM_AMIN = {'Blue': 11.0, 'Green': 16.0, 'Red': 26.0, 'Full': 16.0}

COLS = ['cand_id', 'source_id', 'lst', 'date', 'band', 'ra', 'dec',
        'ra_hms', 'dec_dms', 'gal_lat_deg', 'ecl_lat_deg',
        'snr', 'peak_jy', 'int_jy', 'sign', 'elevation_deg',
        'gfit_fwhm_beams', 'gfit_ar', 'gfit_resid',
        'peak_jy_Red', 'peak_jy_Green', 'peak_jy_Blue', 'peak_jy_Full',
        'mstruct', 'iv0', 'bilobe', 'gaussfit', 'fp_class', 'n_v_epoch',
        'nearest_ateam', 'jupiter_deg', 'n_det', 'n_epochs',
        'repeat_lst_dates', 'c50_mjy', 'body',
        'xmatch', 'xmatch_sep_arcmin', 'xmatch_sample', 'xmatch_dist_pc',
        'n_xmatch', 'xmatch_all', 'review_dir']

def main():
    rows = list(csv.DictReader(open(CSV)))
    r7 = [r for r in rows if float(r['snr']) >= 7]
    by_src = defaultdict(list)
    for r in r7:
        by_src[r['source_id']].append(r)
    print(f'{len(rows)} detection rows, {len(r7)} >=7sig, {len(by_src)} sources')

    xcat, xnames, xsamples, xdists = fcsv.load_xcat()

    def xmatch_multi(ra, dec, band, snr, elev):
        c = SkyCoord(ra * u.deg, dec * u.deg)
        sig_stat = 0.42 * XR_BEAM_AMIN[band] / max(snr, 5.0)
        rad = float(np.hypot(XR_BASE[band], sig_stat))
        try:
            if elev != '' and float(elev) < 40.0:
                rad *= 1.5
        except ValueError:
            pass
        seps = c.separation(xcat).arcmin
        idx = np.where(seps < rad)[0]
        idx = idx[np.argsort(seps[idx])]
        hits = [(xnames[int(i)], xsamples[int(i)], float(seps[int(i)]),
                 xdists[int(i)]) for i in idx]
        return rad, hits

    # ── build per-source rows ──
    sources = []
    for sid, rr in by_src.items():
        rr.sort(key=lambda r: -float(r['snr']))
        best = rr[0]
        s = {k: best.get(k, '') for k in COLS if k in best}
        s['source_id'] = sid
        s['n_det'] = len(rr)
        s['n_epochs'] = len(set((r['lst'], r['date']) for r in rr))
        s['repeat_lst_dates'] = ';'.join(sorted(set(
            f"{r['lst']}:{r['date']}" for r in rr)))
        for b in ('Red', 'Green', 'Blue', 'Full'):
            vals = [float(r[f'peak_jy_{b}']) for r in rr if r.get(f'peak_jy_{b}')]
            s[f'peak_jy_{b}'] = round(max(vals, key=abs), 4) if vals else ''
        rad, hits = xmatch_multi(float(best['ra']), float(best['dec']),
                                 best['band'], float(best['snr']),
                                 best['elevation_deg'])
        if hits:
            nm, tag, sep, dist = hits[0]
            s['xmatch'], s['xmatch_sep_arcmin'] = nm, round(sep, 2)
            s['xmatch_sample'], s['xmatch_dist_pc'] = tag, dist
        else:
            s['xmatch'] = s['xmatch_sep_arcmin'] = ''
            s['xmatch_sample'] = s['xmatch_dist_pc'] = ''
        s['n_xmatch'] = len(hits)
        s['xmatch_all'] = ';'.join(f"{nm}|{tag}|{sep:.2f}'"
                                   for nm, tag, sep, _ in hits)
        s['_best_old'] = best['cand_id']
        s['_old_ids'] = [r['cand_id'] for r in rr]
        sources.append(s)

    sources.sort(key=lambda s: -float(s['snr']))
    for k, s in enumerate(sources):
        sfx = s['_best_old'].split('_', 1)[1]
        s['cand_id'] = f'rank{k+1:04d}_{sfx}'
        s['review_dir'] = f'{CD}/{s["cand_id"]}'

    multi = [s for s in sources if int(s['n_xmatch']) > 1]
    print(f'per-source rows: {len(sources)}; multi-xmatch sources: {len(multi)}')
    for s in multi[:10]:
        print(f"  {s['cand_id']}: {s['xmatch_all']}")

    # ── directory merge/rename (two-phase, logged) ──
    FR = re.compile(r'^rank\d+_((I|V)_(master|epoch|subtracted)_.*cutout\.fits)$')
    logrows = []
    dirs = set(os.listdir(CD))
    # phase 1: every old dir -> __mv__<new>/<sub for secondaries>
    for s in sources:
        new = s['cand_id']
        for old in s['_old_ids']:
            if old not in dirs:
                logrows.append((old, new, 'MISSING-OLD-DIR'))
                continue
            role = 'primary' if old == s['_best_old'] else 'merge'
            os.rename(f'{CD}/{old}', f'{CD}/__mv__{new}__{role}__{old}')
            logrows.append((old, new, role))
    leftover = [d for d in os.listdir(CD) if not d.startswith('__mv__')]
    if leftover:
        print('WARNING leftover dirs (not in any source):', leftover[:5])
    # phase 2: primaries become the new dir; merges pour in
    for s in sources:
        new = s['cand_id']
        prim = f'{CD}/__mv__{new}__primary__{s["_best_old"]}'
        if os.path.isdir(prim):
            os.rename(prim, f'{CD}/{new}')
        else:
            os.makedirs(f'{CD}/{new}', exist_ok=True)
        for g in glob.glob(f'{CD}/__mv__{new}__merge__*'):
            for f in os.listdir(g):
                if f in ('detections.csv',):
                    os.remove(f'{g}/{f}'); continue
                if f == 'allsky_images.txt':
                    old_l = open(f'{g}/{f}').read().splitlines()
                    dst = f'{CD}/{new}/allsky_images.txt'
                    cur = set(open(dst).read().splitlines()) if os.path.exists(dst) else set()
                    with open(dst, 'a') as fh:
                        for l in old_l:
                            if l not in cur:
                                fh.write(l + '\n')
                    os.remove(f'{g}/{f}'); continue
                if os.path.exists(f'{CD}/{new}/{f}'):
                    os.remove(f'{g}/{f}')      # identical J-name cutout duplicate
                else:
                    shutil.move(f'{g}/{f}', f'{CD}/{new}/{f}')
            os.rmdir(g)
        # unify rank prefixes on the 6-frame files
        pref = new.split('_')[0]
        for f in os.listdir(f'{CD}/{new}'):
            m = FR.match(f)
            if m and not f.startswith(pref + '_'):
                tgt = f'{pref}_{m.group(1)}'
                if not os.path.exists(f'{CD}/{new}/{tgt}'):
                    os.rename(f'{CD}/{new}/{f}', f'{CD}/{new}/{tgt}')

    # regenerate detections.csv per source (ALL >=5sig rows of the source)
    by_src_all = defaultdict(list)
    for r in rows:
        by_src_all[r['source_id']].append(r)
    for s in sources:
        det = sorted(by_src_all[s['source_id']], key=lambda r: -float(r['snr']))
        with open(f'{CD}/{s["cand_id"]}/detections.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in det:
                w.writerow(r)

    with open(LOG, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['old', 'new', 'action'])
        w.writerows(logrows)

    # ── write CSVs ──
    shutil.copy2(CSV, BAK)
    shutil.copy2(CSV, ALLDET)
    with open(CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for s in sources:
            w.writerow({k: s.get(k, '') for k in COLS})

    nd = len(os.listdir(CD))
    print(f'FINAL: {len(sources)} rows, {nd} dirs '
          f'({"OK" if nd == len(sources) else "MISMATCH"})')
    print('anchors:', [s['cand_id'] for s in sources
                       if s['source_id'] in ('J0832+2918', 'J2215+5137')][:4])


if __name__ == '__main__':
    main()
