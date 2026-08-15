#!/usr/bin/env python3
"""Repair BMAJ/BMIN/BPA on all Phase-3 deep co-add masters (2026-07-11).

BUG (mosaic-chat report + root cause): coadd_full_pipeline builds tmpl_hdr from
build_template_header(post_dates[0]) — the FIRST post-cutover epoch's Taper_R075
image — and coadd_inverse_variance stamps that header on every master. Two
consequences:
  1. Template-epoch bug: the master carries ONE arbitrary epoch's fitted beam.
     Where post_dates[0] = 2024-12-26/27 (bad ionosphere; at 05h Red a raw
     dewarp-failsafe image) the beam is ~2x inflated: 01h-04h all bands +
     05h Red wideband, and the same LST range in coadd_sb. Mild elsewhere.
  2. Wrong-variant bug: ALL variants (I Taper/NoTaper x R0/R075, V Taper_R0)
     share the Taper_R075 template beam. R0/V true beams are ~1.5-2x larger,
     so V-master integrated fluxes (imfit/BDSF) are biased HIGH ~2x archive-wide.

FIX: for every master, recompute the beam as the stack-weight-weighted mean of
its constituents' per-VARIANT fitted beams (constituents from COADD-constituent
HISTORY; weights from the cell's residuals.csv; per-date beams from the Phase-2
archive image of the matching variant, dewarped preferred, raw fallback — i.e.
exactly the images the co-add consumed). BPA = weighted circular mean (180 deg
period). Siblings (_pbcorr, _clean, _clean_pbcorr) get the same repaired beam.
Old values preserved in BMAJORIG/BMINORIG/BPAORIG + BEAMSRC/BEAMFIX provenance.

Usage: python repair_master_beams.py [--apply] [--wideband|--subband]
Default is DRY RUN: writes the ledger CSV only.
Ledger: /lustre/gh/main/clv15/master_beam_repair_20260711.csv
"""
import os, re, sys, glob, csv
import numpy as np
from astropy.io import fits

APPLY = '--apply' in sys.argv
DO_WB = '--subband' not in sys.argv
DO_SB = '--wideband' not in sys.argv
# --exclude_lsts 20h,21h,22h — skip cells whose masters are being READ by
# running searches (in-place header update can shift the data offset).
EXCL = set()
if '--exclude_lsts' in sys.argv:
    EXCL = set(sys.argv[sys.argv.index('--exclude_lsts') + 1].split(','))
# --only_lsts 20h,21h,22h — the complement, for the deferred pass.
ONLY = set()
if '--only_lsts' in sys.argv:
    ONLY = set(sys.argv[sys.argv.index('--only_lsts') + 1].split(','))
_tag = ('apply' if '--apply' in sys.argv else 'dryrun')
_scope_tag = ('wb' if '--subband' not in sys.argv else '') + \
             ('sb' if '--wideband' not in sys.argv else '')
LEDGER = f'/lustre/gh/main/clv15/master_beam_repair_20260711_{_scope_tag}_{_tag}.csv'
P2 = '/lustre/pipeline/exopipe/phase2'

# variant key -> (stokes, taper, robust-token)
VARIANTS = {
    'I_deep_Taper_R075':   ('I', 'Taper',   '0.75'),
    'I_deep_Taper_R0':     ('I', 'Taper',   '0'),
    'I_deep_NoTaper_R075': ('I', 'NoTaper', '0.75'),
    'I_deep_NoTaper_R0':   ('I', 'NoTaper', '0'),
    'V_deep_Taper_R0':     ('V', 'Taper',   '0'),
}

def find_epoch_image(lst, date, scope, stokes, taper, robust):
    """Locate the Phase-2 archive deep image the co-add consumed for this
    (date, variant): newest Science run, dewarped preferred, raw fallback."""
    sci = sorted(glob.glob(f'{P2}/{lst}/{date}/Science_*'))
    if not sci:
        return None
    d = f'{sci[-1]}/{scope}/{stokes}/deep'
    cands = glob.glob(f'{d}/*.fits')
    def ok(name):
        if 'pbcorr' in name:
            return False
        if taper == 'Taper' and 'NoTaper' in name:
            return False
        if taper == 'NoTaper' and 'NoTaper' not in name:
            return False
        # Robust-0.75 contains Robust-0 as a prefix — match on the delimiter
        if robust == '0.75':
            return 'Robust-0.75' in name
        return bool(re.search(r'Robust-0[_-]', name))
    cands = [c for c in cands if ok(os.path.basename(c))]
    dew = [c for c in cands if 'dewarped' in os.path.basename(c)]
    pick = sorted(dew or cands)
    return pick[0] if pick else None

def load_weights(workdir):
    w = {}
    p = f'{workdir}/residuals.csv'
    if os.path.exists(p):
        with open(p) as f:
            for row in csv.DictReader(f):
                try:
                    w[row['date']] = float(row['weight'])
                except (KeyError, ValueError):
                    pass
    return w

def weighted_beam(entries):
    """entries: list of (bmaj, bmin, bpa, weight) in degrees. BPA circular
    mean with 180-deg period."""
    a = np.array(entries, dtype=float)
    w = a[:, 3]
    if w.sum() <= 0:
        w = np.ones(len(a))
    bmaj = float(np.sum(a[:, 0] * w) / w.sum())
    bmin = float(np.sum(a[:, 1] * w) / w.sum())
    ang = np.deg2rad(a[:, 2] * 2.0)
    bpa = float(np.rad2deg(0.5 * np.arctan2(np.sum(w * np.sin(ang)),
                                            np.sum(w * np.cos(ang)))))
    return bmaj, bmin, bpa

def repair_cell(root, cell, rows):
    lst, scope = cell.split('_', 1)
    cf = f'{root}/{cell}/coadd_final'
    if not os.path.isdir(cf):
        return
    weights = load_weights(f'{root}/{cell}')
    for vkey, (stokes, taper, robust) in VARIANTS.items():
        master = f'{cf}/{vkey}_{scope}.fits'
        if not os.path.exists(master):
            continue
        hdr = fits.getheader(master)
        dates = [str(h).split()[-1] for h in hdr.get('HISTORY', [])
                 if str(h).startswith('COADD constituent')]
        entries, missing = [], []
        for date in sorted(set(dates)):
            img = find_epoch_image(lst, date, scope, stokes, taper, robust)
            if img is None:
                missing.append(date)
                continue
            eh = fits.getheader(img)
            if 'BMAJ' not in eh:
                missing.append(date)
                continue
            entries.append((eh['BMAJ'], eh['BMIN'], eh.get('BPA', 0.0),
                            weights.get(date, 1.0)))
        status = 'ok'
        if not dates:
            status = 'no-history'
        elif not entries:
            status = 'no-constituent-images'
        if entries:
            bmaj, bmin, bpa = weighted_beam(entries)
        else:
            bmaj = bmin = bpa = None
        old = (hdr.get('BMAJ'), hdr.get('BMIN'), hdr.get('BPA'))
        ratio = (bmaj / old[0]) if (bmaj and old[0]) else None
        # siblings that must carry the same beam as this variant's master
        sibs = [master]
        for suffix in ('_pbcorr', '_clean', '_clean_pbcorr'):
            s = f'{cf}/{vkey}{suffix}_{scope}.fits'
            if os.path.exists(s):
                sibs.append(s)
        rows.append({
            'cell': cell, 'variant': vkey, 'status': status,
            'n_constituents': len(dates), 'n_beams': len(entries),
            'missing_dates': ';'.join(missing),
            'bmaj_old_am': round(old[0] * 60, 2) if old[0] else '',
            'bmin_old_am': round(old[1] * 60, 2) if old[1] else '',
            'bmaj_new_am': round(bmaj * 60, 2) if bmaj else '',
            'bmin_new_am': round(bmin * 60, 2) if bmin else '',
            'ratio': round(ratio, 3) if ratio else '',
            'n_files': len(sibs),
        })
        if APPLY and bmaj:
            for s in sibs:
                with fits.open(s, mode='update') as hd:
                    H = hd[0].header
                    for k, v in (('BMAJORIG', H.get('BMAJ')),
                                 ('BMINORIG', H.get('BMIN')),
                                 ('BPAORIG', H.get('BPA'))):
                        if v is not None and k not in H:
                            H[k] = (v, 'pre-repair (template-epoch) value')
                    H['BMAJ'] = bmaj
                    H['BMIN'] = bmin
                    H['BPA'] = bpa
                    H['BEAMSRC'] = (f'ivw-mean {len(entries)} constituents',
                                    'per-variant stack-weighted beam')
                    H['BEAMFIX'] = ('2026-07-11', 'template-epoch beam bug repair')
                    H.add_history('BEAM repaired 2026-07-11: template-epoch/'
                                  'wrong-variant bug (repair_master_beams.py)')

rows = []
cells = []
if DO_WB:
    root = '/fast/gh/main/phase3/coadd'
    cells += [(root, c) for c in sorted(os.listdir(root))
              if re.match(r'^\d\dh_(Red|Green|Blue|Full)$', c)]
if DO_SB:
    root = '/data07/gh/phase3/coadd_sb'
    if os.path.isdir(root):
        cells += [(root, c) for c in sorted(os.listdir(root))
                  if re.match(r'^\d\dh_\d+MHz$', c)]
if EXCL:
    cells = [(r, c) for r, c in cells if c.split('_')[0] not in EXCL]
if ONLY:
    cells = [(r, c) for r, c in cells if c.split('_')[0] in ONLY]
print(f'{len(cells)} cells, APPLY={APPLY}')
for i, (root, cell) in enumerate(cells):
    repair_cell(root, cell, rows)
    if (i + 1) % 50 == 0:
        print(f'  {i+1}/{len(cells)}')

with open(LEDGER, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
bad = [r for r in rows if r['status'] != 'ok']
big = [r for r in rows if r['ratio'] and abs(float(r['ratio']) - 1) > 0.15]
print(f'{len(rows)} master variants; {len(bad)} problems; '
      f'{len(big)} with >15% beam change')
print(f'Ledger: {LEDGER}')
