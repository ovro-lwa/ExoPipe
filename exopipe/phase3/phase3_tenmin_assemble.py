#!/usr/bin/env python3
"""
phase3_tenmin_assemble.py — 10-MIN tier per-source assembly (§5.15k),
wideband (Run_PROD10*) and subband (Run_SB10*), Stokes V and I, riding the
phase3_sb_assemble machinery. Ledger schema: GLED10_COLS (lst, scope,
stokes, date, interval, ra, dec, pix, snr, peak_jy, morph_pass,
veto_reason, final).

Grouping radii: wideband = measured (Blue 3' / Green 3.5' / Red 5' /
Full 5'); subband = band base x (36/nu)^1.5 below 36 MHz (same as deep).
Catalogs -> the tier run roots:
  Run_PROD10/{V,I}_10min_wideband_transients_20260719_v1.csv
  Run_SB10/{V,I}_10min_subband_transients_20260719_v1.csv
(+ *_all_detections sidecars). pb_at_pos/peak_jy_pbcorr from the cell PB
maps (wideband: /fast coadd coadd_final; subband: coadd_sb).

Usage: phase3_tenmin_assemble.py {wb|sb} {V|I} [--lst 01h]
"""
import os, sys, csv, glob, argparse
import numpy as np
from collections import defaultdict

from exopipe.phase3 import phase3_sb_assemble as SBA

TR = '/data07/gh/phase3/transients'
DATE = '20260719'
PB_FLOOR = 0.1
WB_AMIN = {'Red': 5.0, 'Green': 3.5, 'Blue': 3.0, 'Full': 5.0}
_SB_RADIUS = SBA.radius_amin   # capture BEFORE the monkeypatch below


def radius_amin(scope):
    if scope in WB_AMIN:
        return WB_AMIN[scope]
    return _SB_RADIUS(scope)


SBA.radius_amin = radius_amin   # grouping + xmatch use tier-aware radii


def pb_at(tier, lst, scope, ra, dec):
    from astropy.io import fits
    from astropy.wcs import WCS
    root = ('/fast/gh/main/phase3/coadd' if tier == 'wb'
            else '/data07/gh/phase3/coadd_sb')
    key = (lst, scope)
    if key not in SBA._pb_cache:
        p = f'{root}/{lst}_{scope}/coadd_final/PB_deep_{scope}.fits'
        SBA._pb_cache[key] = None
        if os.path.exists(p):
            SBA._pb_cache[key] = (np.squeeze(fits.getdata(p)),
                                  WCS(fits.getheader(p)).celestial)
    ent = SBA._pb_cache[key]
    if ent is None:
        return np.nan
    try:
        x, y = ent[1].world_to_pixel_values(float(ra), float(dec))
        return float(ent[0][int(round(float(y))), int(round(float(x)))])
    except Exception:
        return np.nan


def load_rows(tier, stokes, lst_filter):
    pats = (['Run_PROD10', 'Run_PROD10_lwacalim01', 'Run_PROD10_lwacalim02']
            if tier == 'wb' else
            ['Run_SB10', 'Run_SB10_lwacalim01', 'Run_SB10_lwacalim02'])
    rows, seen = [], set()
    for rr in pats:
        p = f'{TR}/{rr}/v_10min_ledger.csv'
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            if r.get('stokes') != stokes:
                continue
            if lst_filter and r['lst'] != lst_filter:
                continue
            snr = SBA._f(r.get('snr'))
            if not np.isfinite(snr) or snr < 5.0:
                continue
            k = (r['lst'], r['scope'], r['date'], r.get('interval'),
                 round(SBA._f(r['ra']), 4), round(SBA._f(r['dec']), 4),
                 r.get('snr'))
            if k in seen:
                continue
            seen.add(k)
            rows.append(dict(lst=r['lst'], scope=r['scope'], date=r['date'],
                             interval=r.get('interval', ''),
                             ra=SBA._f(r['ra']), dec=SBA._f(r['dec']),
                             snr=snr, peak_jy=SBA._f(r.get('peak_jy')),
                             final=r.get('final', ''),
                             veto_reason=r.get('veto_reason', ''),
                             morph_pass=r.get('morph_pass', '')))
    return rows


def assemble(tier, stokes, lst_filter):
    from exopipe.phase3.phase3_source_ids import assign_ids
    rows = load_rows(tier, stokes, lst_filter)
    cells = {(r['lst'], r['scope']) for r in rows}
    tname = 'wideband' if tier == 'wb' else 'subband'
    print(f'{stokes} {tname} 10min: {len(rows)} deduped >=5sig detections '
          f'over {len(cells)} cells', flush=True)
    if not rows:
        return
    members = SBA.group(rows)
    print(f'{stokes} {tname} 10min: {len(members)} sources', flush=True)
    cats = SBA.load_samples()
    root = f'{TR}/Run_PROD10' if tier == 'wb' else f'{TR}/Run_SB10'
    os.makedirs(root, exist_ok=True)
    tag = '' if not lst_filter else f'_smoke_{lst_filter}'
    detp = f'{root}/{stokes}_10min_{tname}_all_detections_{DATE}{tag}.csv'
    catp = f'{root}/{stokes}_10min_{tname}_transients_{DATE}_v1{tag}.csv'

    bests = [max(mem, key=lambda i: rows[i]['snr']) for mem in members]
    stubs = [dict(ra=rows[b]['ra'], dec=rows[b]['dec']) for b in bests]
    assign_ids(stubs, origin=f'10min_{tname}_{stokes}_{DATE}')

    src_rows = []
    for mi, mem in enumerate(members):
        b = rows[bests[mi]]
        r_am = radius_amin(b['scope'])
        hits = SBA.xmatch(b['ra'], b['dec'], r_am, cats)
        pb = pb_at(tier, b['lst'], b['scope'], b['ra'], b['dec'])
        pbc = max(pb, PB_FLOOR) if np.isfinite(pb) else np.nan
        dates = {rows[i]['date'] for i in mem}
        nfin = sum(1 for i in mem if rows[i]['final'] == '1')
        src_rows.append(dict(
            source_id=stubs[mi]['source_id'], lst=b['lst'], date=b['date'],
            scope=b['scope'], interval=b['interval'], ra=b['ra'],
            dec=b['dec'], snr=b['snr'], peak_jy=b['peak_jy'],
            pb_at_pos=f'{pb:.4f}' if np.isfinite(pb) else '',
            peak_jy_pbcorr=f'{b["peak_jy"]/pbc:.4f}'
                if np.isfinite(pbc) and np.isfinite(b['peak_jy']) else '',
            n_det_5sig=len(mem), n_epochs=len(dates),
            n_det_7sig=sum(1 for i in mem if rows[i]['snr'] >= 7),
            n_final=nfin, match_radius_amin=f'{r_am:.1f}',
            n_xmatch=len(hits),
            xmatch=hits[0][1] if hits else '',
            xmatch_sample=hits[0][0] if hits else '',
            xmatch_sep_arcmin=f'{hits[0][2]:.2f}' if hits else '',
            xmatch_all=';'.join(f'{h[0]}:{h[1]}:{h[2]:.1f}' for h in hits),
            final=b['final'], veto_reason=b['veto_reason'], _members=mem))
    src_rows.sort(key=lambda r: -r['snr'])
    for k, r in enumerate(src_rows):
        r['rank'] = k + 1
        r['cand_id'] = (f"rank{k+1:04d}_{r['lst']}_{r['date']}_{r['scope']}"
                        f"_t{r['interval']}")
    with open(detp, 'w', newline='') as f:
        cols = list(rows[0].keys()) + ['source_id', 'cand_id']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in src_rows:
            for i in r['_members']:
                w.writerow(dict(rows[i], source_id=r['source_id'],
                                cand_id=r['cand_id']))
    with open(catp, 'w', newline='') as f:
        cols = ['rank', 'cand_id', 'source_id'] + \
            [c for c in src_rows[0] if c not in ('rank', 'cand_id',
                                                 'source_id', '_members')]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(src_rows)
    n7 = sum(1 for r in src_rows if r['snr'] >= 7)
    nx = sum(1 for r in src_rows if r['n_xmatch'] > 0)
    nf = sum(1 for r in src_rows if r['n_final'] > 0)
    print(f'{stokes} {tname} 10min: wrote {len(src_rows)} sources '
          f'({n7} >=7sig, {nx} xmatched, {nf} w/ final-passing det) -> {catp}',
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tier', choices=['wb', 'sb'])
    ap.add_argument('stokes', choices=['V', 'I', 'both'])
    ap.add_argument('--lst', default=None)
    a = ap.parse_args()
    for st in (['V', 'I'] if a.stokes == 'both' else [a.stokes]):
        assemble(a.tier, st, a.lst)


if __name__ == '__main__':
    main()
