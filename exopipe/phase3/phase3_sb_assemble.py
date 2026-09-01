#!/usr/bin/env python3
"""
phase3_sb_assemble.py — SUBBAND-DEEP per-source assembly (§5.15k standard;
catalogs stage — the dir/frame packager is phase3_sb_package.py).

Union of the subband deep search ledgers:
  V: Run_GLBACKFILL/v_deep_ledger.csv (210 pre-parity cells)
     + Run_V_SB_lwacalim01|02 aliases Run_SB1_lwacalim01|02 (post-parity)
  I: Run_I_SB/i_deep_ledger.csv + Run_I_SB/nodes/*/i_deep_ledger.csv
Dedup (lst,scope,stokes,date,ra4,dec4,snr). ALL >=5 sigma rows enter the
catalog (locked §5.9 convention); gate verdicts carried per detection.

Grouping: greedy positional clustering by descending SNR, radius =
band-base measured radius (Blue 3' / Green 3.5' / Red 5') scaled
x max(1,(36/nu)^1.5) below 36 MHz; a pair links when sep <= max(r1,r2).
Source IDs from the shared registry (phase3_source_ids.assign).

Cross-match per source (nearest + n_xmatch + xmatch_all, same nu-scaled
radius): sample CSVs 10pc/25pc/50pc/Hot_Warm_Jupiters/LPTs/flare_ucd/sydney_stars
+ pulsars_atnf.csv (v2.8.1).

Outputs (run roots Run_V_SB / Run_I_SB):
  {V,I}_deep_subband_transients_20260719_v1.csv   per-source rows, ranked
  {V,I}_deep_subband_all_detections_20260719.csv  per-detection rows
Columns include pb_at_pos / peak_jy_pbcorr (PB_deep_{sb} cell maps, floor
0.1, at the best detection). Coverage report: distinct (lst,scope) cells
per Stokes vs the 330 expected.

Usage: phase3_sb_assemble.py [--stokes V|I|both] [--lst 16h]  (lst = smoke)
"""
import os, sys, csv, glob, argparse
import numpy as np
from collections import defaultdict


TR = '/data07/gh/phase3/transients'
COADD_SB = '/data07/gh/phase3/coadd_sb'
CLV = '/lustre/gh/main/clv15'
DATE = '20260719'
PB_FLOOR = 0.1
SAMPLES = ['10pc.csv', '25pc.csv', '50pc.csv', 'Hot_Warm_Jupiters.csv',
           'LPTs.csv', 'flare_ucd.csv', 'sydney_stars.csv', 'pulsars_atnf.csv']

SB_BASE_AMIN = {  # band-base measured radii, nu-scaled below 36 MHz
    '18MHz': 5.0, '23MHz': 5.0, '27MHz': 5.0, '32MHz': 5.0, '36MHz': 5.0,
    '41MHz': 3.5, '46MHz': 3.5, '50MHz': 3.5, '55MHz': 3.5, '59MHz': 3.5,
    '64MHz': 3.0, '69MHz': 3.0, '73MHz': 3.0, '78MHz': 3.0, '82MHz': 3.0}


def radius_amin(scope):
    nu = float(scope.replace('MHz', ''))
    return SB_BASE_AMIN[scope] * max(1.0, (36.0 / nu) ** 1.5)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _sep_amin(ra1, dec1, ra2, dec2):
    r1, d1 = np.radians(ra1), np.radians(dec1)
    r2, d2 = np.radians(np.asarray(ra2, float)), np.radians(np.asarray(dec2, float))
    s = (np.sin((d2 - d1) / 2) ** 2
         + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2)
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(s, 0, 1)))) * 60.0


def load_v(lst_filter=None):
    rows, seen = [], set()
    paths = ([f'{TR}/Run_GLBACKFILL/v_deep_ledger.csv']
             + glob.glob(f'{TR}/Run_SB1_lwacalim0*/v_deep_ledger.csv'))
    for p in paths:
        for r in csv.DictReader(open(p)):
            if r.get('stokes', 'V') != 'V':
                continue
            if lst_filter and r['lst'] != lst_filter:
                continue
            if _f(r.get('snr')) < 5.0:
                continue
            k = (r['lst'], r['scope'], r['date'],
                 round(_f(r['ra']), 4), round(_f(r['dec']), 4), r.get('snr'))
            if k in seen:
                continue
            seen.add(k)
            rows.append(dict(lst=r['lst'], scope=r['scope'], date=r['date'],
                             ra=_f(r['ra']), dec=_f(r['dec']),
                             snr=_f(r['snr']), peak_jy=_f(r.get('peak_jy')),
                             final=r.get('final', ''),
                             veto_reason=r.get('veto_reason', ''),
                             morph_pass=r.get('morph_pass', ''),
                             mstruct=r.get('mstruct', ''),
                             jupiter_deg=r.get('jupiter_deg', '')))
    return rows


def load_i(lst_filter=None):
    rows, seen = [], set()
    paths = ([f'{TR}/Run_I_SB/i_deep_ledger.csv']
             + glob.glob(f'{TR}/Run_I_SB/nodes/*/i_deep_ledger.csv'))
    for p in paths:
        for r in csv.DictReader(open(p)):
            scope = r.get('band') or r.get('scope')
            if lst_filter and r['lst'] != lst_filter:
                continue
            if _f(r.get('snr')) < 5.0:
                continue
            k = (r['lst'], scope, r['date'],
                 round(_f(r['ra']), 4), round(_f(r['dec']), 4), r.get('snr'))
            if k in seen:
                continue
            seen.add(k)
            rows.append(dict(lst=r['lst'], scope=scope, date=r['date'],
                             ra=_f(r['ra']), dec=_f(r['dec']),
                             snr=_f(r['snr']), peak_jy=_f(r.get('peak_jy')),
                             final=r.get('final', r.get('pass_all', '')),
                             veto_reason=r.get('veto_reason', ''),
                             gf_fwhm_beams=r.get('gf_fwhm_beams', ''),
                             mstruct=r.get('mstruct_i', '')))
    return rows


def group(rows):
    """Greedy clustering by descending SNR; nu-scaled link radii.
    Grid-indexed (1-deg cells; radii <= ~14', so 3x3 neighborhood is
    exhaustive for linkable centers) — exact same link rule as the O(n*k)
    scan, at O(n * local density)."""
    order = sorted(range(len(rows)), key=lambda i: -rows[i]['snr'])
    ra = np.array([r['ra'] for r in rows])
    dec = np.array([r['dec'] for r in rows])
    rad = np.array([radius_amin(r['scope']) for r in rows])
    cra, cdec, crad, members = [], [], [], []
    grid = {}

    def cell(r, d):
        return (int(r % 360.0), int(d + 90.0))

    for i in order:
        cands = []
        c0 = cell(ra[i], dec[i])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cands.extend(grid.get(((c0[0] + dx) % 360, c0[1] + dy), ()))
        if cands:
            aj = np.array(cands)
            seps = _sep_amin(ra[i], dec[i],
                             np.array([cra[j] for j in cands]),
                             np.array([cdec[j] for j in cands]))
            lim = np.maximum(np.array([crad[j] for j in cands]), rad[i])
            k = int(np.argmin(seps / lim))
            if seps[k] <= lim[k]:
                members[aj[k]].append(i)
                continue
        j = len(cra)
        cra.append(ra[i]); cdec.append(dec[i]); crad.append(rad[i])
        members.append([i])
        grid.setdefault(cell(ra[i], dec[i]), []).append(j)
    return members


def xmatch(ra, dec, r_amin, cats):
    hits = []
    for name, (cra, cdec, labels) in cats.items():
        if not len(cra):
            continue
        seps = _sep_amin(ra, dec, cra, cdec)
        for j in np.where(seps <= r_amin)[0]:
            hits.append((name, labels[j], float(seps[j])))
    hits.sort(key=lambda h: h[2])
    return hits


def load_samples():
    cats = {}
    for s in SAMPLES:
        p = f'{CLV}/{s}'
        if not os.path.exists(p):
            continue
        cra, cdec, lab = [], [], []
        # sample sheets use common_name/ra_current/dec_current (BOM-prefixed
        # header on some); fall back to generic ra/dec variants
        with open(p, encoding='utf-8-sig') as fh:
            for r in csv.DictReader(fh):
                rr = _f(r.get('ra_current') or r.get('ra') or r.get('RA')
                        or r.get('RAJD'))
                dd = _f(r.get('dec_current') or r.get('dec') or r.get('DEC')
                        or r.get('Dec') or r.get('DECJD'))
                if np.isfinite(rr) and np.isfinite(dd):
                    cra.append(rr); cdec.append(dd)
                    lab.append(r.get('common_name') or r.get('name')
                               or r.get('Name') or r.get('NAME')
                               or r.get('pl_name') or r.get('PSRJ') or '?')
        cats[s.replace('.csv', '')] = (np.array(cra), np.array(cdec), lab)
    return cats


_pb_cache = {}


def pb_at(lst, scope, ra, dec):
    from astropy.io import fits
    from astropy.wcs import WCS
    key = (lst, scope)
    if key not in _pb_cache:
        p = f'{COADD_SB}/{lst}_{scope}/coadd_final/PB_deep_{scope}.fits'
        _pb_cache[key] = None
        if os.path.exists(p):
            _pb_cache[key] = (np.squeeze(fits.getdata(p)),
                              WCS(fits.getheader(p)).celestial)
    ent = _pb_cache[key]
    if ent is None:
        return np.nan
    try:
        x, y = ent[1].world_to_pixel_values(float(ra), float(dec))
        return float(ent[0][int(round(float(y))), int(round(float(x)))])
    except Exception:
        return np.nan


def assemble(stokes, lst_filter):
    rows = load_v(lst_filter) if stokes == 'V' else load_i(lst_filter)
    cells = {(r['lst'], r['scope']) for r in rows}
    print(f'{stokes}: {len(rows)} deduped >=5sig detections over '
          f'{len(cells)} cells', flush=True)
    members = group(rows)
    print(f'{stokes}: {len(members)} sources', flush=True)
    from exopipe.phase3.phase3_source_ids import assign_ids
    cats = load_samples()

    root = f'{TR}/Run_V_SB' if stokes == 'V' else f'{TR}/Run_I_SB'
    tag = '' if not lst_filter else f'_smoke_{lst_filter}'
    detp = f'{root}/{stokes}_deep_subband_all_detections_{DATE}{tag}.csv'
    catp = f'{root}/{stokes}_deep_subband_transients_{DATE}_v1{tag}.csv'

    bests = [max(mem, key=lambda i: rows[i]['snr']) for mem in members]
    id_stubs = [dict(ra=rows[b]['ra'], dec=rows[b]['dec']) for b in bests]
    assign_ids(id_stubs, origin=f'sb_deep_{stokes}_{DATE}')
    src_rows = []
    for mi, mem in enumerate(members):
        best = bests[mi]
        b = rows[best]
        r_am = radius_amin(b['scope'])
        sid = id_stubs[mi]['source_id']
        hits = xmatch(b['ra'], b['dec'], r_am, cats)
        pb = pb_at(b['lst'], b['scope'], b['ra'], b['dec'])
        pbc = max(pb, PB_FLOOR) if np.isfinite(pb) else np.nan
        dates = {rows[i]['date'] for i in mem}
        sbs = sorted({rows[i]['scope'] for i in mem},
                     key=lambda s: float(s.replace('MHz', '')))
        src_rows.append(dict(
            source_id=sid, lst=b['lst'], date=b['date'], scope=b['scope'],
            ra=b['ra'], dec=b['dec'], snr=b['snr'], peak_jy=b['peak_jy'],
            pb_at_pos=f'{pb:.4f}' if np.isfinite(pb) else '',
            peak_jy_pbcorr=f'{b["peak_jy"]/pbc:.4f}'
                if np.isfinite(pbc) and np.isfinite(b['peak_jy']) else '',
            n_det_5sig=len(mem), n_epochs=len(dates),
            n_det_7sig=sum(1 for i in mem if rows[i]['snr'] >= 7),
            subbands='+'.join(sbs), match_radius_amin=f'{r_am:.1f}',
            n_xmatch=len(hits),
            xmatch=hits[0][1] if hits else '',
            xmatch_sample=hits[0][0] if hits else '',
            xmatch_sep_arcmin=f'{hits[0][2]:.2f}' if hits else '',
            xmatch_all=';'.join(f'{h[0]}:{h[1]}:{h[2]:.1f}' for h in hits),
            final=b.get('final', ''), veto_reason=b.get('veto_reason', ''),
            _members=mem))
    src_rows.sort(key=lambda r: -r['snr'])
    for k, r in enumerate(src_rows):
        r['rank'] = k + 1
        r['cand_id'] = (f"rank{k+1:04d}_{r['lst']}_{r['date']}_{r['scope']}")

    with open(detp, 'w', newline='') as f:
        cols = [c for c in rows[0].keys()] + ['source_id', 'cand_id']
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
    print(f'{stokes}: wrote {len(src_rows)} sources ({n7} >=7sig, {nx} '
          f'xmatched) -> {catp}', flush=True)
    print(f'{stokes}: coverage {len(cells)} cells (expect 330 full-run)',
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stokes', default='both', choices=['V', 'I', 'both'])
    ap.add_argument('--lst', default=None)
    a = ap.parse_args()
    for st in (['V', 'I'] if a.stokes == 'both' else [a.stokes]):
        assemble(st, a.lst)


if __name__ == '__main__':
    main()
