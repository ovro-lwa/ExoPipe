#!/usr/bin/env python3
"""Stokes-I wideband catalog v5 (Gregg 2026-07-11): ONE CSV, per-source
rows down to 5 sigma — same convention as the V v2 catalog.

- The 948 v4 (>=7 sigma, quarantine-applied) rows keep their ranks,
  cand_ids and veto columns VERBATIM; they gain n_det_5sig, and their
  n_epochs / repeat_lst_dates / per-band best fluxes are extended over the
  attached 5-7 sigma detections (dc2-corrected fluxes, v4 scale).
- Sub-7-sigma sources (rank0949+): ledger detections passing the full
  ratified gate chain at >=5 sigma, not attached to any v4 source, grouped
  at 0.5 deg. Groups whose BEST detection is >=7 sigma but absent from v4
  are the officially vetoed/quarantined sources — EXCLUDED (logged), so the
  >=7 sigma content of v5 is exactly v4.
- Rank directories (Gregg 2026-07-11): repeats (n_det_5sig>=2) and
  cross-matched sources (n_xmatch>=1) get dirs too, at every SNR; this
  script computes eligibility + review_dir, dir creation is a separate
  packaging pass.
Writes I_deep_wideband_transients_20260711_v5.csv (+ sidecar
i_v5_new_source_members.csv mapping new sources -> ledger detections) and
repoints the I_deep_FINAL_transients.csv symlink.
"""
import sys
_ARGV = list(sys.argv); sys.argv = ['iv5']
import os, csv, glob, math
import numpy as np
from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u
from collections import defaultdict

from exopipe.phase3 import assemble as live
from exopipe.phase3 import phase3_i_deep_final as F
from exopipe.phase3 import phase3_final_csv as fc
from astropy.time import Time

ROOT = live.ROOT
V4 = f'{ROOT}/I_deep_wideband_transients_20260710_v4.csv'
OUT = f'{ROOT}/I_deep_wideband_transients_20260711_v5.csv'
LINK = f'{ROOT}/I_deep_FINAL_transients.csv'
MEMBERS = f'{ROOT}/i_v5_new_source_members.csv'
CANDDIR = f'{ROOT}/candidates'

SIZE = {'Blue': 1.26, 'Green': 1.32, 'Red': 1.66, 'Full': 2.11}
JUP = {'Red': 5.0, 'Green': 0.5, 'Blue': 0.5, 'Full': 5.0}
XR_BASE = {'Blue': 3.0, 'Green': 3.5, 'Red': 5.0, 'Full': 5.0}
XR_BEAM = {'Blue': 11.0, 'Green': 16.0, 'Red': 26.0, 'Full': 16.0}
ASSOC = {'0.25': 'assoc_m025_jy', '0.5': 'assoc_m05_jy', '1.0': 'assoc_m10_jy',
         '1.5': 'assoc_m15_jy', '2.0': 'assoc_m20_jy'}
MERGE = 0.5


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


# ── 1. v4 rows ──
v4 = list(csv.DictReader(open(V4)))
v4cols = list(v4[0].keys())
print(f'v4 rows: {len(v4)}')

# ── 2. chain-passing >=5sig ledger detections ──
vk = set()
kp = f'{ROOT}/dyn2d_vetoed_keys.csv'
if os.path.exists(kp):
    vk = {(k['lst'], k['date'], k['band'], k['pix_x'], k['pix_y'])
          for k in csv.DictReader(open(kp))}
dets, seen = [], set()
for p in ([f'{ROOT}/i_deep_ledger.csv']
          + sorted(glob.glob(f'{ROOT}/nodes/*/i_deep_ledger.csv'))):
    for r in csv.DictReader(open(p)):
        k = (r['lst'], r['date'], r['band'], r['pix_x'], r['pix_y'], r['sign'])
        if k in seen:
            continue
        seen.add(k)
        snr = fnum(r['snr'])
        if r['sign'] != '+' or not snr >= 5:
            continue
        if r['pass_mstruct'] != '1' or r['bilobe'] == '1':
            continue
        if snr > 8 and r['gaussfit'] == '1':
            continue
        gfw = fnum(r['gf_fwhm_beams'])
        if gfw == gfw and gfw > SIZE.get(r['band'], 99):
            continue
        jd = fnum(r['jupiter_deg'])
        if jd == jd and jd < JUP.get(r['band'], 0):
            continue
        d70 = fnum(r['dist_b70_deg'])
        if d70 == d70 and d70 <= 1.5:
            continue
        if r['img_discard'] == '1' or r.get('dyn_veto', '') == '1':
            continue
        fcs, rb = fnum(r.get('fc_star', '')), r.get('rb_star', '')
        if fcs == fcs and rb in ASSOC:
            a = fnum(r[ASSOC[rb]])
            if a == a and a >= fcs:
                continue
        if (r['lst'], r['date'], r['band'], r['pix_x'], r['pix_y']) in vk:
            continue
        dets.append(r)
print(f'chain-passing >=5sig detections: {len(dets)}')

# ── 3. attach to v4 sources (0.5 deg, grid-hashed) ──
grid = defaultdict(list)
for i, s in enumerate(v4):
    ra, dec = float(s['ra']), float(s['dec'])
    cd = max(math.cos(math.radians(dec)), 0.05)
    grid[(int(ra * cd / MERGE), int(dec / MERGE))].append((i, ra, dec))

def near_v4(ra, dec):
    cd = max(math.cos(math.radians(dec)), 0.05)
    gx, gy = int(ra * cd / MERGE), int(dec / MERGE)
    best, bd = -1, MERGE
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for i, ra0, dec0 in grid[(gx + dx, gy + dy)]:
                dra = ((ra - ra0 + 180) % 360 - 180) * cd
                d = math.hypot(dra, dec - dec0)
                if d < bd:
                    best, bd = i, d
    return best

attached = defaultdict(list)
pool = []
for r in dets:
    i = near_v4(float(r['ra']), float(r['dec']))
    if i >= 0:
        attached[i].append(r)
    else:
        pool.append(r)
print(f'attached to v4 sources: {sum(map(len, attached.values()))}, '
      f'pool: {len(pool)}')

# ── 4. group the pool; drop officially-removed >=7sig groups ──
pool.sort(key=lambda r: -fnum(r['snr']))
pgrid = defaultdict(list)
groups = []
for r in pool:
    ra, dec = float(r['ra']), float(r['dec'])
    cd = max(math.cos(math.radians(dec)), 0.05)
    gx, gy = int(ra * cd / MERGE), int(dec / MERGE)
    hit = None
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for gi, ra0, dec0 in pgrid[(gx + dx, gy + dy)]:
                dra = ((ra - ra0 + 180) % 360 - 180) * cd
                if math.hypot(dra, dec - dec0) < MERGE:
                    hit = gi
                    break
            if hit is not None:
                break
        if hit is not None:
            break
    if hit is None:
        hit = len(groups)
        groups.append([])
        pgrid[(gx, gy)].append((hit, ra, dec))
    groups[hit].append(r)

removed7 = [g for g in groups if fnum(g[0]['snr']) >= 7]
newgroups = [g for g in groups if fnum(g[0]['snr']) < 7]
print(f'pool groups: {len(groups)}; excluded (best>=7sig, officially '
      f'removed from v4): {len(removed7)}; new sub-7sig sources: '
      f'{len(newgroups)}')

# ── 5. build new source rows ──
xcat, xnames, xsamples, xdists = fc.load_xcat()
xra, xdec = xcat.ra.deg, xcat.dec.deg
img_rms = F._image_rms()

def xmulti(ra, dec, band, snr, elev):
    sig = 0.42 * XR_BEAM[band] / max(snr, 5.0)
    rad = float(np.hypot(XR_BASE[band], sig))
    if elev == elev and elev < 40.0:
        rad *= 1.5
    cd = max(math.cos(math.radians(dec)), 0.05)
    box = (np.abs(xdec - dec) < rad / 60.0 + 0.02) & \
          (np.abs((xra - ra + 180) % 360 - 180) < (rad / 60.0 + 0.02) / cd)
    idx0 = np.where(box)[0]
    if not len(idx0):
        return []
    seps = SkyCoord(ra * u.deg, dec * u.deg).separation(xcat[idx0]).arcmin
    keep = np.where(seps < rad)[0]
    keep = keep[np.argsort(seps[keep])]
    return [(xnames[int(idx0[i])], xsamples[int(idx0[i])], float(seps[int(i)]),
             xdists[int(idx0[i])]) for i in keep]

newrows, memrows = [], []
for g in newgroups:
    g.sort(key=lambda r: -fnum(r['snr']))
    b = g[0]
    dc2 = fnum(b['dc2']) if b['dc2'] else 1.0
    if dc2 != dc2:
        dc2 = 1.0
    pk = fnum(b['peak_jy'])
    eps = sorted({f"{r['lst']}:{r['date']}" for r in g})
    s = {c: '' for c in v4cols}
    s.update(source_id='', lst=b['lst'], date=b['date'], band=b['band'],
             ra=round(float(b['ra']), 5), dec=round(float(b['dec']), 5),
             snr=round(fnum(b['snr']), 2), sign=b['sign'],
             peak_jy=round(pk * dc2, 4), peak_jy_meas=round(pk, 4),
             mstruct=b['mstruct_i'], assoc_m05_jy=b['assoc_m05_jy'],
             bilobe=b['bilobe'], gaussfit=b['gaussfit'],
             gfit_fwhm_beams=b['gf_fwhm_beams'], gfit_ar=b['gf_ar'],
             gfit_resid=b['gf_resid'], jupiter_deg=b['jupiter_deg'],
             fc_star=b['fc_star'], img_floor=b['floor'], dc2=b['dc2'],
             n_det=0, n_det_5sig=len(g), n_epochs=len(eps),
             repeat_lst_dates='|'.join(eps),
             n_i_image=b['n_epoch_raw'],
             img_rms_mjy=img_rms.get((b['lst'], b['date'], b['band']), ''),
             pos_flag='')
    lo, hi = live.FP_CLASS[b['band']]
    try:
        nim = int(b['n_epoch_raw'])
        s['fp_class'] = ('Low' if nim <= lo else
                         ('High' if nim > hi else 'Int'))
    except ValueError:
        s['fp_class'] = ''
    bb = {}
    for r in g:
        d2 = fnum(r['dc2']) if r['dc2'] else 1.0
        if d2 != d2:
            d2 = 1.0
        v = abs(fnum(r['peak_jy'])) * d2
        if r['band'] not in bb or v > bb[r['band']]:
            bb[r['band']] = v
    for band in ('Red', 'Green', 'Blue', 'Full'):
        s[f'peak_jy_{band}'] = round(bb[band], 4) if band in bb else ''
    c = SkyCoord(s['ra'] * u.deg, s['dec'] * u.deg)
    s['ra_hms'] = c.ra.to_string(unit=u.hourangle, sep=':', precision=1,
                                 pad=True)
    s['dec_dms'] = c.dec.to_string(sep=':', precision=0, alwayssign=True,
                                   pad=True)
    s['gal_lat_deg'] = round(float(c.galactic.b.deg), 2)
    s['ecl_lat_deg'] = round(float(c.barycentricmeanecliptic.lat.deg), 2)
    seps = [(nm, float(c.separation(co).deg)) for nm, co in live.ATEAM]
    nm, dd = min(seps, key=lambda t: t[1])
    s['nearest_ateam'] = f'{nm.replace(" ", "")}:{dd:.2f}'
    t, utc = live.epoch_mid(s['lst'], s['band'], s['date'])
    elev = float('nan')
    if t is not None:
        elev = float(c.transform_to(
            AltAz(obstime=t, location=fc.OVRO)).alt.deg)
        s['elevation_deg'] = round(elev, 2)
    s['body'] = fc.body_label(s['ra'], s['dec'], utc) if utc else ''
    hits = xmulti(s['ra'], s['dec'], s['band'], float(s['snr']), elev)
    if hits:
        s['xmatch'], s['xmatch_sep_arcmin'] = hits[0][0], round(hits[0][2], 2)
        s['xmatch_sample'], s['xmatch_dist_pc'] = hits[0][1], hits[0][3]
    s['n_xmatch'] = len(hits)
    s['xmatch_all'] = ';'.join(f"{n}|{tg}|{sp:.2f}'" for n, tg, sp, _ in hits)
    s['_members'] = g
    newrows.append(s)

newrows.sort(key=lambda s: -float(s['snr']))

# ── 6. extend v4 rows over attached sub-7sig detections ──
for i, s in enumerate(v4):
    sub = [r for r in attached.get(i, []) if fnum(r['snr']) < 7]
    s['n_det_5sig'] = int(s['n_det']) + len(sub)
    if sub:
        eps = set(s['repeat_lst_dates'].split('|')) if s['repeat_lst_dates'] \
            else set()
        eps |= {f"{r['lst']}:{r['date']}" for r in sub}
        s['repeat_lst_dates'] = '|'.join(sorted(eps))
        s['n_epochs'] = len(eps)
        for r in sub:
            d2 = fnum(r['dc2']) if r['dc2'] else 1.0
            if d2 != d2:
                d2 = 1.0
            v = abs(fnum(r['peak_jy'])) * d2
            cur = fnum(s[f"peak_jy_{r['band']}"])
            if cur != cur or v > cur:
                s[f"peak_jy_{r['band']}"] = round(v, 4)

# ── 7. ranks, review_dir eligibility, write ──
allrows = v4 + newrows
for k, s in enumerate(allrows):
    if k >= len(v4):
        s['cand_id'] = (f"rank{k+1:04d}_{s['lst']}_{s['date']}_{s['band']}")
    n7 = int(s['n_det']) if s['n_det'] != '' else 0
    eligible = ((n7 > 0 and (float(s['snr']) > 8 or n7 >= 2))
                or int(s['n_det_5sig']) >= 2
                or (s['n_xmatch'] != '' and int(s['n_xmatch']) >= 1))
    have = os.path.isdir(f"{CANDDIR}/{s['cand_id']}")
    s['review_dir'] = f"{CANDDIR}/{s['cand_id']}" if (eligible or have) else ''

cols = v4cols if 'n_det_5sig' in v4cols else \
    v4cols[:v4cols.index('n_epochs')] + ['n_det_5sig'] + \
    v4cols[v4cols.index('n_epochs'):]
with open(OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    for s in allrows:
        w.writerow(s)
with open(MEMBERS, 'w', newline='') as fh:
    w = csv.writer(fh)
    # node-ledger rows predate the 42-col schema migration (no
    # assoc_m025_jy / rb_star) — take the central header, blank-fill
    with open(f'{ROOT}/i_deep_ledger.csv') as ch:
        ledcols = next(csv.reader(ch))
    w.writerow(['cand_id'] + ledcols)
    for s in newrows:
        for r in s['_members']:
            w.writerow([s['cand_id']] + [r.get(c, '') for c in ledcols])
    # attached sub-7sig detections of v4 sources, for their detections.csv
    for i, s4 in enumerate(v4):
        for r in attached.get(i, []):
            if fnum(r['snr']) < 7:
                w.writerow([s4['cand_id']] + [r.get(c, '') for c in ledcols])

if os.path.islink(LINK) or os.path.exists(LINK):
    os.remove(LINK)
os.symlink(os.path.basename(OUT), LINK)

nel = sum(1 for s in allrows if s['review_dir'])
nnew = sum(1 for s in allrows if s['review_dir'] and
           not os.path.isdir(s['review_dir']))
print(f'v5 written: {len(allrows)} rows ({len(v4)} v4 + {len(newrows)} new); '
      f'dir-eligible {nel} (new dirs needed {nnew})')
print('IV5-BUILD-DONE')
