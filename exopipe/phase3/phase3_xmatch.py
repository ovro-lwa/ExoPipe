#!/usr/bin/env python3
"""
phase3_xmatch.py — cross-match the deep candidate catalog against science
target lists (LPTs, Hot/Warm Jupiters, 50pc). Per-candidate positional radius
= 3σ centroid (beam/(2.355·SNR) ⊕ 1.5' astrometry floor), capped at one beam.

The 50pc list is dense, so single matches are chance-dominated — the script
reports the expected-by-chance count per list and ranks targets by the number
of INDEPENDENT epochs/bands they are hit in. A target detected in ≥2 epochs is
the real signal (chance of the same target re-aligning is negligible).

Outputs: xmatch_<list>.csv + console summary. Rerunnable as the survey grows.
"""
import csv, os
import numpy as np
from collections import defaultdict
from astropy.coordinates import SkyCoord
import astropy.units as u

CAT = '/fast/gh/main/phase3/transients/Run_PROD/master_catalog.csv'
OUT = '/fast/gh/main/phase3/transients/Run_PROD/xmatch'
os.makedirs(OUT, exist_ok=True)
LISTS = [
    ('LPT',        '/lustre/gh/main/clv15/LPTs.csv',              'name',        'ra',         'dec',         None),
    ('HotWarmJup', '/lustre/gh/main/clv15/Hot_Warm_Jupiters.csv', 'common_name', 'ra_current', 'dec_current', 'distance'),
    ('50pc',       '/lustre/gh/main/clv15/50pc.csv',              'common_name', 'ra_current', 'dec_current', 'distance'),
]
BEAM = {'Red': 12.8, 'Green': 17.3, 'Blue': 13.3, 'Full': 16.3}   # arcmin FWHM
ASTROM = 1.5      # arcmin astrometric floor
A_SKY = 20626.0   # deg^2 visible hemisphere (chance-rate denominator)


def fnum(r, k, d=0.0):
    try: return float(r[k])
    except: return d


rows = list(csv.DictReader(open(CAT)))
for r in rows:
    bm = BEAM.get(r.get('scope'), 15.0); snr = max(abs(fnum(r, 'snr')), 3.0)
    sig = np.hypot(bm / (2.355 * snr), ASTROM)
    r['_rad'] = float(min(bm, max(2.0, 3 * sig)))            # arcmin
ccoord = SkyCoord([fnum(r, 'ra') for r in rows] * u.deg,
                  [fnum(r, 'dec') for r in rows] * u.deg)

for tag, path, nk, rak, deck, dk in LISTS:
    if not os.path.exists(path):
        print(f"{tag}: MISSING {path}"); continue
    trows = [t for t in csv.DictReader(open(path))
             if t.get(rak, '').strip() and t.get(deck, '').strip()]
    tcoord = SkyCoord([float(t[rak]) for t in trows] * u.deg,
                      [float(t[deck]) for t in trows] * u.deg)
    idx, sep2d, _ = ccoord.match_to_catalog_sky(tcoord)
    sepam = sep2d.arcmin
    matches = []
    for i, r in enumerate(rows):
        if sepam[i] <= r['_rad']:
            t = trows[idx[i]]
            matches.append(dict(target=t[nk], sep=round(float(sepam[i]), 2),
                                dist_pc=(round(float(t[dk]), 1) if dk and t.get(dk) else ''),
                                lst=r['lst'], date=r['date'], scope=r['scope'],
                                stokes=r['stokes'], snr=round(fnum(r, 'snr'), 1),
                                ra=r['ra'], dec=r['dec'], rad=round(r['_rad'], 1)))
    exp = sum(np.pi * (r['_rad'] / 60.0) ** 2 / A_SKY for r in rows) * len(trows)
    bytarget = defaultdict(list)
    for m in matches:
        bytarget[m['target']].append(m)
    multi = {k: v for k, v in bytarget.items()
             if len({(m['lst'], m['date']) for m in v}) > 1}
    print(f"\n=== {tag}: {len(trows)} targets | {len(matches)} candidate-matches on "
          f"{len(bytarget)} targets | ~{exp:.1f} expected by chance | "
          f"{len(multi)} multi-epoch (real-signal) ===")
    for tname, ms in sorted(bytarget.items(), key=lambda kv: -len({(m['lst'], m['date']) for m in kv[1]})):
        eps = sorted({(m['lst'], m['date']) for m in ms})
        bands = ','.join(sorted({m['scope'] for m in ms}))
        stk = ','.join(sorted({m['stokes'] for m in ms}))
        best = max(abs(m['snr']) for m in ms); dpc = ms[0]['dist_pc']
        flag = '  <<< MULTI-EPOCH' if len(eps) > 1 else ''
        print(f"  {tname:26s} {len(ms):3d}det {len(eps):2d}ep [{bands}] {stk} "
              f"bestSNR={best:5.1f} sep~{min(m['sep'] for m in ms):4.1f}' d={dpc}pc{flag}")
    with open(f'{OUT}/xmatch_{tag}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['target', 'sep', 'dist_pc', 'lst', 'date',
                                          'scope', 'stokes', 'snr', 'ra', 'dec', 'rad'])
        w.writeheader()
        for m in sorted(matches, key=lambda m: (m['target'], m['sep'])):
            w.writerow(m)
print(f"\nwrote {OUT}/xmatch_<list>.csv  (catalog snapshot: {len(rows)} candidates)")
