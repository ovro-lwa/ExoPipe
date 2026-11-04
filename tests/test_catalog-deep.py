#!/usr/bin/env python3
"""Tests of the catalog-deep unit (steps/catalog.py deep path, lib/catalog/deep_*.py, lib/catalog/tests.py).

Run:  NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py \\
      /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_catalog-deep.py [test_name ...]

Synthetic data only, in a fresh /tmp/exopipe3_tests_catalog-deep/ (removed at the end). The deep chain runs IN PROCESS
(cli.main) over a synthetic run with label P3S9 (cells 05h_Full and 05h_23MHz, Stokes V and I, three nights in 2030).
In-place modules: a stub exopipe_ops root holds
  - thin runpy wrappers that run the REAL production modules unchanged (read-only, from /lustre/gh/main/exopipe_ops):
    p3s3_sources.py, p3s3_merge.py (+ the real p3s3_stale_exclusions and data_decisions.csv), select_v3.py,
    p3s2_ephem_filter.py, p3s3_identity.py, p3s2_transient_list.py;
  - stubs with the production file contract for the stages that need masters, frames or tables: p3s5_parent_union.py,
    p3s2_conform.py, deep_fit_positions.py, build_source_catalogs_v3.py, multiples_fork.py, build_localizations.py.
    The conform stub applies production's AXIS step (a row above 8 sigma with axis_norm > 2.0 removed in its words,
    ops/p3s2_conform.py:632-637) - with no switch, as production today, unless a test writes the pending patch's
    P3_CONFORM_AXIS switch into it (stub_conform) - and the fork stub runs it for its validity step, as production's
    multiples_fork.py does (fork mode, table off, its removals kept under the tag A as well).
The test patches CFG in process (stub legacy root, collect roots, masters root, OPS_ROOT; binding check off for the stubs)
- protected keys cannot be overridden on the command line, by design. Read-only checks on production: the binding check
against the real exopipe_ops modules. Nothing outside the scratch directory is written.
"""
import csv
import glob
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import contextlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
REAL_OPS = '/lustre/gh/main/exopipe_ops'
T = '/tmp/exopipe3_tests_catalog-deep'
LABEL = 'P3S9'
U = f'{T}/out/Run_{LABEL}'
DATES = ('2030-01-01', '2030-01-02', '2030-01-03')

# ---------------------------------------------------------------------------------------------------------------------
# stub exopipe_ops root
# ---------------------------------------------------------------------------------------------------------------------
WRAP = '''import runpy, sys
sys.path.append({real!r})
runpy.run_path({real!r} + '/{name}', run_name='__main__')
'''

STUBCAT = r'''
import csv, json, os, collections
def rcsv(p):
    with open(p, newline='') as f:
        rd = csv.DictReader(f); return list(rd.fieldnames or []), list(rd)
def wcsv(p, cols, rows):
    with open(p + '.part', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    os.replace(p + '.part', p)
def jname(ra, dec):
    h = ra / 15.0; hh = int(h); mm = int(round((h - hh) * 60))
    s = '+' if dec >= 0 else '-'; d = abs(dec); dd = int(d); dm = int(round((d - dd) * 60))
    return f'J{hh:02d}{mm:02d}{s}{dd:02d}{dm:02d}'
SCOLS = ['rank', 'source_id', 'identity_id', 'identity_bands', 'ra', 'dec', 'pos_err_arcsec', 'stat_err_arcsec',
         'sys_err_arcsec', 'band', 'lst_cells', 'n_det', 'n_nights_searched', 'best_cand_id', 'best_date', 'best_snr',
         'peak_jy_best', 'sign', 'source_class', 'promotion', 'review_dir']
MCOLS = ['source_id', 'band', 'cand_id', 'det_ra', 'det_dec', 'det_snr', 'det_date', 'det_epoch_qa', 'det_stat_err_arcsec']
def build(rows, outdir, min_n=1, thr=7.0):
    """Group rows per band by position (0.1 deg); write <band>_sources_survivors/_master/_source_members.csv."""
    os.makedirs(outdir, exist_ok=True)
    byb = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        byb[r['band']][(round(float(r['ra']), 1), round(float(r['dec']), 1))].append(r)
    n = 0
    for band, groups in byb.items():
        srcs, mem = [], []
        for (ra, dec), rs in groups.items():
            best = max(rs, key=lambda r: abs(float(r['snr'])))
            if len(rs) < min_n or abs(float(best['snr'])) < 5.0:
                continue
            sid = jname(ra, dec)
            srcs.append(dict(source_id=sid, identity_id='', identity_bands='', ra=f'{ra:.5f}', dec=f'{dec:.5f}',
                             pos_err_arcsec='60.0', stat_err_arcsec='30.0', sys_err_arcsec='52.0', band=band,
                             lst_cells='+'.join(sorted({r['lst'] + '_' + band for r in rs})), n_det=str(len(rs)),
                             n_nights_searched='3', best_cand_id=best['cand_id'], best_date=best['date'],
                             best_snr=best['snr'], peak_jy_best=best.get('peak_jy', ''), sign='+', source_class='',
                             promotion='', review_dir=''))
            for r in rs:
                mem.append(dict(source_id=sid, band=band, cand_id=r['cand_id'], det_ra=r['ra'], det_dec=r['dec'],
                                det_snr=r['snr'], det_date=r['date'], det_epoch_qa='', det_stat_err_arcsec='30.0'))
        srcs.sort(key=lambda s: -abs(float(s['best_snr'])))
        surv = [s for s in srcs if abs(float(s['best_snr'])) >= thr]
        for i, s in enumerate(surv, 1):
            s['rank'] = f'rank{i:04d}'
        ids = {s['source_id'] for s in surv}
        wcsv(f'{outdir}/{band}_sources_survivors.csv', SCOLS, surv)
        wcsv(f'{outdir}/{band}_sources_master.csv', SCOLS, srcs)
        wcsv(f'{outdir}/{band}_source_members.csv', MCOLS, [m for m in mem if m['source_id'] in ids])
        n += len(surv)
    json.dump({}, open(f'{outdir}/E_params.json', 'w'))
    return n
'''

STUBS = {
    'p3s5_parent_union.py': r'''
import glob, os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__)); T = os.path.dirname(HERE)
out = sys.argv[1]; os.makedirs(out, exist_ok=True); ok, miss = 0, []
for cdir in sorted(glob.glob(f'{T}/coadd_sb/[0-2][0-9]h_*')):
    cell = os.path.basename(cdir); lst, sc = cell.split('_', 1); name = f'master_R075_{lst}_{sc}.gaul.csv'
    if os.path.exists(f'{T}/cands/{name}') and os.path.exists(f'{T}/cands/{name}.gen.json'):
        for f in (name, name + '.gen.json'):
            shutil.copy2(f'{T}/cands/{f}', f'{out}/{f}')
        ok += 1
    else:
        miss.append((cell, 'no catalogue on the current master'))
print(f'{ok} cells with a current parent catalogue in {out}; {len(miss)} without: {miss}')
''',
    'p3s2_conform.py': r'''
import csv, os, shutil, sys
from _stubcat import rcsv, wcsv
U = sys.argv[1]; CD = os.environ.get('P3_CONFORM_CATALOG_DIR') or U
FORK = os.environ.get('P3_CONFORM_FORK') == '1'
AXIS_STEP = True   # AXIS-SWITCH (production today: the axis step has no switch; stub_conform writes the pending patch's here)
if not os.environ.get('P3_PARENT_DIR'):
    sys.exit('conform stub: no P3_PARENT_DIR - REFUSING')
if os.environ.get('P3_CONFORM_SIZE') != '0' or os.environ.get('P3_CADENCE') != 'deep' or os.environ.get('P3_TABLE_ELEV') != '1':
    sys.exit('conform stub: the pass environment is wrong - REFUSING')
if FORK and os.environ.get('P3_CONFORM_TABLE') != 'off':
    sys.exit("conform stub: the fork's validity run has its table off - REFUSING")
cols, raw = rcsv(f'{CD}/master_catalog.csv')
if not raw and not FORK:
    sys.exit('conform: no rows - REFUSING')
shutil.copy(f'{CD}/master_catalog.csv', f'{CD}/master_catalog_asrun.csv')
out, rem = [], []
for r in raw:
    if 'RMV' in r['cand_id'] and not FORK:
        # production's words for a row-line removal (ops/p3s2_conform.py; removal_rules.rowline_reason): the per-instant
        # content test matches every removal reason whole against the writers' templates
        rem.append(dict(r, removal_reason='row-line: >= 5 finals of this epoch on one pixel row (Gregg 2026-09-05)')); continue
    r = dict(r); r['axis_flag'] = ''
    try:
        an, snr = float(r.get('axis_norm') or 'nan'), abs(float(r['snr']))
    except ValueError:
        an, snr = float('nan'), 0.0
    if an == an and snr > 8.0 and an > 2.0:            # production's axis step (ops/p3s2_conform.py:632-637), in its words
        if AXIS_STEP:
            rem.append(dict(r, removal_reason=f'axis ratio {r["axis_norm"]} x beam > 2.0 above 8.0 sigma (Gregg 2026-09-05)'))
            continue
        r['axis_flag'] = 'axis > 2.0 x beam (kept: point-like at its S/N by the size / axis rule)'
    r['fp_prediction'] = 'not-FP' if r['stokes'] == 'V' else ''; out.append(r)
oc = cols + [c for c in ('fp_prediction', 'axis_flag') if c not in cols]
wcsv(f'{CD}/master_catalog.csv', oc, out); wcsv(f'{CD}/master_catalog_removed.csv', oc + ['removal_reason'], rem)
if FORK:
    print(f'conform: {len(raw)} raw -> {len(out)} conformant | removed: {len(rem)} (fork mode)')
    sys.exit(0)
keys = sorted({(r['stokes'], r['lst'], r['band'], r['date']) for r in raw})
wcsv(f'{U}/frame_times.csv', ['stokes', 'lst', 'band', 'date', 'date_obs', 'date_end'],
     [dict(stokes=s, lst=l, band=b, date=d, date_obs=d + 'T12:00:00.000', date_end=d + 'T13:00:00.000') for s, l, b, d in keys])
wcsv(f'{U}/frame_beam_ratio.csv', ['stokes', 'lst', 'band', 'date', 'ratio'],
     [dict(stokes=s, lst=l, band=b, date=d, ratio='1.2') for s, l, b, d in keys])
# the frames' beams (production's D8 cache, ops 283a513): the grouping stage reads them (item F.3)
wcsv(f'{U}/frame_beam_bmaj.csv', ['stokes', 'lst', 'band', 'date', 'bmaj_deg'],
     [dict(stokes=s, lst=l, band=b, date=d, bmaj_deg='0.15' if 'MHz' in b else '0.25') for s, l, b, d in keys])
print(f'conform: {len(raw)} raw -> {len(out)} conformant | removed: {len(rem)}')
''',
    'deep_fit_positions.py': r'''
import json, os, sys
from _stubcat import rcsv, wcsv
U = sys.argv[1]; cols, rows = rcsv(f'{U}/master_catalog.csv')
if 'pos_kind' in cols:
    sys.exit('REFUSED: already carries pos_kind')
DET = ['pos_kind', 'pos_ra', 'pos_dec', 'pos_err_maj_arcsec', 'pos_err_min_arcsec', 'pos_err_pa_deg', 'pos_stat_maj_arcsec',
       'pos_stat_min_arcsec', 'pos_sys_arcsec', 'pos_snr', 'pos_offset_arcmin', 'pos_reason', 'peak_ra', 'peak_dec']
for r in rows:
    o = dict(pos_kind='gaussian' if abs(float(r['snr'])) >= 7 else 'template', pos_ra=f"{float(r['ra']) + 0.001:.6f}",
             pos_dec=f"{float(r['dec']) + 0.001:.6f}", pos_err_maj_arcsec='60.0', pos_err_min_arcsec='55.0', pos_err_pa_deg='0.0',
             pos_stat_maj_arcsec='30.0', pos_stat_min_arcsec='25.0', pos_sys_arcsec='52.0', pos_snr=r['snr'],
             pos_offset_arcmin='0.08', pos_reason='', peak_ra=r['ra'], peak_dec=r['dec'])
    r.update(o); r['ra'], r['dec'] = o['pos_ra'], o['pos_dec']
wcsv(f'{U}/master_catalog.csv', cols + DET, rows)
wcsv(f'{U}/fit_positions_cache.csv', ['key'], [dict(key=r['cand_id']) for r in rows])
json.dump(dict(rows=len(rows), fitted_rows=len(rows), fitted_on_pixel_centre=0, refused=[], seconds=0.1),
          open(f'{U}/fit_positions_summary.json', 'w'))
print(f'FIT-POSITIONS-OK {len(rows)} rows')
''',
    'build_source_catalogs_v3.py': r'''
import sys
from _stubcat import rcsv, build
sel, outdir, img = sys.argv[1:4]
_, rows = rcsv(sel)
st = {r['stokes'] for r in rows}
if len(st) > 1:
    sys.exit(f'REFUSED: {sel} holds more than one Stokes {sorted(st)}')
print(f'built {build(rows, outdir)} survivors -> {outdir}')
''',
    'multiples_fork.py': r'''
import collections, json, os, shutil, subprocess, sys
from _stubcat import rcsv, wcsv, build
HERE = os.path.dirname(os.path.abspath(__file__))
mode, S = sys.argv[1], sys.argv[2]
U, OUT = os.environ['FORK_RUN_DIR'], os.environ['FORK_OUT_DIR']
if not os.environ.get('P3_PARENT_DIR'):
    sys.exit('FORK REFUSED: no P3_PARENT_DIR')
w = f'{U}/fork_{S}'
if mode == 'run':
    shutil.rmtree(w, ignore_errors=True); os.makedirs(w)
    cols, rows = rcsv(f'{U}/master_catalog_asrun.csv')
    rows = [dict(r) for r in rows if r['stokes'] == S and abs(float(r['snr'])) >= {'I': 7.0, 'V': 5.0}[S]]
    # 2 validity: the conformance in place, fork mode, table off (multiples_fork.conform: its environment from os.environ),
    # its removals kept under the run's tag too - so the chain reads the same rows in two files
    wcsv(f'{w}/master_catalog.csv', cols, rows)
    e = dict(os.environ, P3_CONFORM_CATALOG_DIR=w, P3_CONFORM_FORK='1', P3_CONFORM_TABLE='off', P3_CONFORM_SIZE='0',
             P3_CADENCE='deep')
    p = subprocess.run([sys.executable, os.path.join(HERE, 'p3s2_conform.py'), U], env=e, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f'FORK REFUSED: conform (A, table off) exit {p.returncode}: {(p.stdout + p.stderr)[-300:]}')
    shutil.copyfile(f'{w}/master_catalog_removed.csv', f'{w}/master_catalog_removed_A.csv')
    cols, rows = rcsv(f'{w}/master_catalog.csv')
    g = collections.defaultdict(set)
    for r in rows:
        g[(r['band'], round(float(r['ra']), 1), round(float(r['dec']), 1))].add((r['lst'], r['date']))
    for r in rows:
        r['fork_repeat'] = '1' if len(g[(r['band'], round(float(r['ra']), 1), round(float(r['dec']), 1))]) >= 2 else '0'
    wcsv(f'{w}/master_catalog.csv', cols + ['fork_repeat'], rows)
    kept = [r for r in rows if r['fork_repeat'] == '1']
    wcsv(f'{w}/fork_kept_{S}.csv', cols + ['fork_repeat'], kept)
    for t in ('wb', 'sb'):
        k = [r for r in kept if ('MHz' in r['band']) == (t == 'sb')]
        if not k:
            continue
        d = f'{OUT}/source_catalogs_{S}_f{t}'
        build(k, d, min_n=2)
        wcsv(f'{d}/repeat_shuffle.csv', ['source_id', 'p_shuffle'], [])
        # production's fork leaves its working catalogues in its work tree (fork_<S>/tmp_catalogs/f<t>/), which its record
        # lists and no later stage reads (the replay of 2026-10-15)
        os.makedirs(f'{w}/tmp_catalogs/f{t}', exist_ok=True)
        for f_ in os.listdir(d):
            if f_.endswith('_sources_survivors.csv'):
                shutil.copy(f'{d}/{f_}', f'{w}/tmp_catalogs/f{t}/{f_}')
        open(f'{w}/build_{S}_{t}_final.log', 'w').write('sigma-stack singleton gate: off\n')
    json.dump(dict(kept=len(kept)), open(f'{w}/fork_stats_{S}.json', 'w'))
    print(f'FORK-DONE {S}: kept {len(kept)}')
else:
    print(f'FORK CONTENT TEST {S}: stub sources -> PASS')
''',
    'build_localizations.py': r'''
import csv, glob, os, sys
from _stubcat import rcsv, wcsv
U = sys.argv[1]; OUT = os.environ.get('P3_LOC_OUT') or f'{U}/source_catalogs'
os.makedirs(OUT, exist_ok=True)
pat = os.environ.get('P3_LOC_CATS', 'source_catalogs_[IV]_*')
dets = {}
for p in glob.glob(f'{U}/{pat}/*_sources_survivors.csv'):
    _, ss = rcsv(p)
    mp = p.replace('_sources_survivors.csv', '_source_members.csv')
    _, mm = rcsv(mp) if os.path.exists(mp) else ([], [])
    for s in ss:
        ident = s.get('identity_id') or s['source_id']
        d = dets.setdefault(ident, {})
        for m in mm:
            if m['source_id'] == s['source_id']:
                d[m['cand_id']] = m
irows, drows = [], []
for ident in sorted(dets):
    ms = list(dets[ident].values())
    if not ms:
        continue
    ra = sum(float(m['det_ra']) for m in ms) / len(ms); dec = sum(float(m['det_dec']) for m in ms) / len(ms)
    irows.append(dict(identity_id=ident, ra=round(ra, 5), dec=round(dec, 5), err_arcsec=30.0, err_raw_arcsec=30.0, chi2=0.5,
                      n_det=len(ms)))
    for m in ms:
        drows.append(dict(cand_id=m['cand_id'], identity_id=ident, sep_arcsec=5.0, sep_beams=0.01, offset_pct=10.0,
                          det_sigma_arcsec=60.0))
wcsv(f'{OUT}/localizations.csv', ['identity_id', 'ra', 'dec', 'err_arcsec', 'err_raw_arcsec', 'chi2', 'n_det'], irows)
wcsv(f'{OUT}/localization_offsets.csv', ['cand_id', 'identity_id', 'sep_arcsec', 'sep_beams', 'offset_pct', 'det_sigma_arcsec'], drows)
print(f'{len(irows)} identities localized (0 without detections), {len(drows)} detection offsets')
''',
}
# a stub builder carrying production's patch points (tests/grouping_stub_builder.py): design B in the chain runs the
# grouping runner on it (test_grouping_design_b_in_the_chain)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grouping_stub_builder import PATCHABLE_BUILDER   # noqa: E402
REAL = ('p3s3_sources.py', 'p3s3_merge.py', 'select_v3.py', 'p3s2_ephem_filter.py', 'p3s3_identity.py',
        'p3s2_transient_list.py')


def make_ops():
    ops = f'{T}/exopipe_ops'
    os.makedirs(ops, exist_ok=True)
    os.makedirs(f'{T}/clv15', exist_ok=True)
    for name in REAL:
        open(f'{ops}/{name}', 'w').write(WRAP.format(real=REAL_OPS, name=name))
    for name, text in STUBS.items():
        open(f'{ops}/{name}', 'w').write(text.lstrip())
    open(f'{ops}/_stubcat.py', 'w').write(STUBCAT)
    return ops


# ---------------------------------------------------------------------------------------------------------------------
# synthetic run: masters, parents, search state, published attempts
# ---------------------------------------------------------------------------------------------------------------------
# the in-chain Jupiter separation of the attempts' I rows: the search writes it on every row since 2026-09-04 ('Ephemeris
# always-on for I'; the served Run_P3S5 has no I row without it) - test_b30_ephem_pre_check blanks it
WORLD_I_JUPITER_DEG = '50.0'
MC_COLS = ['cand_id', 'lst', 'date', 'band', 'stokes', 'cadence', 'ra', 'dec', 'peak_jy', 'snr', 'global_snr', 'jupiter_deg',
           'moon_deg', 'epoch_qa', 'size_beams', 'size_flag', 'axis_norm']
LED_COLS = ['lst', 'scope', 'stokes', 'date', 'final', 'ra', 'dec', 'veto_reason', 'snr', 'peak_jy', 'pix_y']
IMG_COLS = ['lst', 'scope', 'stokes', 'date', 'frame']
SRC = {'A': (82.5, 37.0), 'B': (83.0, 38.0), 'C': (82.0, 36.5)}


def write_csv(p, cols, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def make_master(cell):
    import numpy as np
    from astropy.io import fits
    lst, sc = cell.split('_', 1)
    d = f'{T}/coadd_sb/{cell}/coadd_final'
    os.makedirs(d, exist_ok=True)
    data = (np.arange(64, dtype=np.float32).reshape(8, 8) + len(cell))
    fits.writeto(f'{d}/I_deep_Taper_R075_{sc}.fits', data, overwrite=True)
    return f'{d}/I_deep_Taper_R075_{sc}.fits', hashlib.md5(np.ascontiguousarray(data).tobytes()).hexdigest()


def make_parent(cell, md5):
    lst, sc = cell.split('_', 1)
    os.makedirs(f'{T}/cands', exist_ok=True)
    name = f'{T}/cands/master_R075_{lst}_{sc}.gaul.csv'
    with open(name, 'w') as f:
        f.write('# Gaussian list\n# Gaus_id, Isl_id, Source_id, Wave_id, RA, E_RA, DEC, E_DEC, Total_flux, E_Total_flux, Peak_flux\n')
        f.write('0, 0, 0, 0, 82.4, 0.01, 37.1, 0.01, 12.0, 0.1, 10.0\n')
    json.dump({'master': f'I_deep_Taper_R075_{sc}.fits', 'master_md5_data': md5}, open(name + '.gen.json', 'w'))


def make_attempt(cell, st, node='nodeA', att='20300104T000000'):
    lst, band = cell.split('_', 1)
    d = f'{T}/collect/{LABEL.lower()}_collect_{node}/attempts/{cell}_{st}_{att}/transients/Run_{LABEL}'
    rows, led, img = [], [], []
    n = 0

    def det(key, date, snr, extra=''):
        nonlocal n
        ra, dec = SRC[key]
        n += 1
        cid = f'{st}{extra}{lst}_{date}_{band}_{n:04d}'
        rows.append(dict(cand_id=cid, lst=lst, date=date, band=band, stokes=st, cadence='deep', ra=f'{ra:.5f}',
                         dec=f'{dec:.5f}', peak_jy=f'{snr / 2:.3f}', snr=f'{snr:.2f}', global_snr=f'{snr:.2f}',
                         jupiter_deg=WORLD_I_JUPITER_DEG if st == 'I' else '50.0', moon_deg='', epoch_qa='', size_beams='1.0',
                         size_flag='', axis_norm='1.0'))
        return ra, dec

    for date in DATES:                                   # source A on every night (a repeat for the fork)
        ra, dec = det('A', date, 9.0 if band == 'Full' else 8.0)
        led.append(dict(lst=lst, scope=band, stokes=st, date=date, final='1', ra=ra, dec=dec, veto_reason='', snr='9', pix_y='3'))
        img.append(dict(lst=lst, scope=band, stokes=st, date=date, frame=f'{date}_{st}'))
    if band == 'Full':
        ra, dec = det('B', DATES[0], 7.5)                # B once: a survivor of the singles tiers only
        led.append(dict(lst=lst, scope=band, stokes=st, date=DATES[0], final='1', ra=ra, dec=dec, snr='7.5', pix_y='4'))
        ra, dec = det('C', DATES[1], 5.5)                # C: selected (>= 5) but not a survivor
        led.append(dict(lst=lst, scope=band, stokes=st, date=DATES[1], final='1', ra=ra, dec=dec, snr='5.5', pix_y='5'))
        det('C', DATES[1], 5.5)                          # duplicate of C's key: refused 'duplicate'
        ra, dec = det('B', DATES[2], 6.0, extra='RMV')   # removed by the conform stub
        led.append(dict(lst=lst, scope=band, stokes=st, date=DATES[2], final='1', ra=ra, dec=dec, snr='6', pix_y='6'))
        det('B', DATES[1], 6.5)                          # no final ledger row of its key: refused
    write_csv(f'{d}/master_catalog.csv', MC_COLS, rows)
    write_csv(f'{d}/v_deep_ledger.csv', LED_COLS, led)
    write_csv(f'{d}/v_deep_images.csv', IMG_COLS, img)
    os.makedirs(f'{d}/candidates/{rows[0]["cand_id"]}', exist_ok=True)
    open(f'{d}/candidates/{rows[0]["cand_id"]}/info.txt', 'w').write('x')
    return node, att, len(rows)


def make_world():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    make_ops()
    for cell in ('05h_Full', '05h_23MHz'):
        _, md5 = make_master(cell)
        make_parent(cell, md5)
    state = f'{T}/ops_state/{LABEL.lower()}_state'
    os.makedirs(state)
    for st in ('V', 'I'):
        os.makedirs(f'{state}/markers_{st}')
        lines = []
        for cell in ('05h_Full', '05h_23MHz'):
            node, att, _ = make_attempt(cell, st)
            lines.append(f'{cell},{node},{att}\n')
            open(f'{state}/markers_{st}/{cell}', 'w').write('done\n')
        open(f'{state}/published_{st}.csv', 'w').writelines(lines)


# ---------------------------------------------------------------------------------------------------------------------
# in-process CLI with the test's config patches
# ---------------------------------------------------------------------------------------------------------------------
_SAVED = {}
_PATCH = (('site', 'LEGACY_CODE_ROOTS'), ('site', 'PYTHON'), ('catalog', 'COLLECT_ROOTS'), ('catalog', 'BINDING_CHECK'),
          ('paths', 'COADD_SB_ROOT'), ('paths', 'OPS_ROOT'), ('catalog_conform', 'MODE'), ('catalog_conform', 'DEEP_ELEV_EPOCH'),
          ('catalog_grouping', 'GROUPING'), ('catalog_grouping', 'MULTIPLES_RULE'), ('catalog_grouping', 'BUILDER_SHA1'))


def patch_cfg():
    from exopipe.phase3.config import CFG
    CFG.reset()
    for sec, key in _PATCH:
        _SAVED.setdefault((sec, key), getattr(getattr(CFG, sec), key))
    CFG.site.LEGACY_CODE_ROOTS = [f'{T}/clv15', f'{T}/exopipe_ops']
    CFG.site.PYTHON = dict(_SAVED[('site', 'PYTHON')], default=sys.executable)
    CFG.catalog.COLLECT_ROOTS = (f'{T}/collect',)
    CFG.catalog.BINDING_CHECK = False                   # the stubs carry no production literals (tested separately)
    CFG.paths.COADD_SB_ROOT = f'{T}/coadd_sb'
    CFG.paths.OPS_ROOT = f'{T}/ops_state'
    # the chain's production path - production's conform (here its stub), as the parity file runs it; the check and the
    # pulsar stage (catalog_conform.MODE 'check', B-02 steps 2 and 4) are tested in tests/test_conform_check.py and below
    CFG.catalog_conform.MODE = getattr(sys.modules[__name__], '_MODE', 'remove')
    CFG.catalog_conform.DEEP_ELEV_EPOCH = getattr(sys.modules[__name__], '_EPOCH', 'mid-hour')
    # production's grouping and its two multiples chains, as the parity file runs them (the stub builder carries none of the
    # grouping runner's patch points; design B in the chain: test_grouping_design_b_in_the_chain)
    CFG.catalog_grouping.GROUPING = getattr(sys.modules[__name__], '_GROUPING', 'production')
    CFG.catalog_grouping.MULTIPLES_RULE = getattr(sys.modules[__name__], '_MRULE', 'production')
    # the grouping runner's builder pin (review 2026-11-04 F7) also accepts the stub builder carrying the patch points
    CFG.catalog_grouping.BUILDER_SHA1 = tuple(_SAVED[('catalog_grouping', 'BUILDER_SHA1')]) + (_stub_builder_sha1(),)
    return CFG


def _stub_builder_sha1():
    import importlib.util
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    spec = importlib.util.spec_from_file_location('bg', GS.builder_runner())
    bg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bg)
    return bg.normalized_sha1(f'{T}/exopipe_ops/build_source_catalogs_v3.py', PATCHABLE_BUILDER.lstrip())


def unpatch_cfg():
    from exopipe.phase3.config import CFG
    CFG.reset()
    for (sec, key), v in _SAVED.items():
        setattr(getattr(CFG, sec), key, v)
    return CFG


def cli(*argv, expect=None):
    from exopipe.phase3 import cli as C
    patch_cfg()
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = C.main(['catalog', '--tier', 'deep', '--run', LABEL, '--out', U, '--cores', 'none'] + list(argv))
    text = err.getvalue()
    if expect is not None and rc != expect:
        print(text[-4000:])
        raise AssertionError(f'catalog {" ".join(argv)}: exit {rc}, expected {expect}')
    return rc, text


def rec(unit):
    p = f'{U}/_records/catalog/{unit}.json'
    return json.load(open(p)) if os.path.exists(p) else None


def rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------------------------------------------
def test_help_and_dry_run():
    r = subprocess.run([sys.executable, '-m', 'exopipe.phase3', 'catalog', '--help'], cwd=REPO, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0 and 'preflight' in r.stdout and 'masters --tier deep --stages sources' in r.stdout, r.stderr[-800:]
    make_world()
    rc, _ = cli('--dry-run', expect=0)
    assert not os.path.exists(U), 'dry-run wrote the run dir'


def test_preflight_refuses_missing_parent():
    make_world()
    make_master('05h_Red')                                   # installed, no parent catalogue
    rc, err = cli('--stages', 'preflight', expect=2)
    r = rec(f'{LABEL}.preflight')
    assert r['status'] == 'refused' and 'exopipe3 masters --tier deep --stages sources --cells 05h_Red' in r['reason'], r['reason']
    assert not glob.glob(f'{T}/exopipe_ops/p3s5_parents*'), 'a parent build was attempted'
    # review 2026-10-13: a cell directory WITHOUT a master refuses, as production (p3s5_parent_union.py:19-20 'no master';
    # fork_serve_p3s5_20261008.sh:22-27 refuses the pass on it)
    make_world()
    os.makedirs(f'{T}/coadd_sb/06h_Blue/coadd_final')
    rc, err = cli('--stages', 'preflight', expect=2)
    r = rec(f'{LABEL}.preflight')
    assert r['status'] == 'refused' and 'without a master' in r['reason'] and '06h_Blue' in r['reason'], r['reason']
    # the parent_snapshot test reads the masters root from paths.COADD_SB_ROOT when no context is given (verify CLI; it
    # raised AttributeError on the removed catalog.DEEP_COADD_ROOT before the review)
    shutil.rmtree(f'{T}/coadd_sb/06h_Blue')
    patch_cfg()
    from exopipe.phase3.core import validate
    res = validate.run('parent_snapshot', f'{T}/cands', sample=2)
    unpatch_cfg()
    assert res and all(x['passed'] for x in res), res


def test_preflight_fails_on_an_empty_parent_copy():
    """Production 2026-10-31 (deep chat, exopipe_ops 9d6a331): 7 of 912 files of the deep pass's parent snapshot were 0-byte
    copies of intact Lustre sources (six catalogues, one .gen.json); the pass failed at conformance with a JSONDecodeError in
    gate_inputs.parents, and production's union now re-copies with a size check. The sidecar records no catalogue size or
    checksum, so a reader trusting it would take an empty catalogue for a cell without parent sources. The port refuses at
    the preflight, naming each file (before this test it crashed there on the same JSONDecodeError), and every later unit
    refuses when a snapshot file changed size since the preflight's manifest (a truncation in place leaves the directory's
    fingerprint unchanged)."""
    make_world()
    cats = sorted(glob.glob(f'{T}/cands/master_R075_*.gaul.csv'))
    assert len(cats) >= 2, cats
    keep = {p: open(p, 'rb').read() for p in (cats[0], cats[1] + '.gen.json')}
    open(cats[0], 'w').close()                               # the union's copies of these are empty, as on 10-31
    open(cats[1] + '.gen.json', 'w').close()
    rc, err = cli('--stages', 'preflight', expect=2)
    r = rec(f'{LABEL}.preflight')
    assert r['status'] == 'refused', (r['status'], r.get('reason', '')[:600])
    assert f'{os.path.basename(cats[0])}: empty catalogue (0 bytes)' in r['reason'], r['reason'][:600]
    assert f'{os.path.basename(cats[1])}.gen.json: sidecar unreadable' in r['reason'], r['reason'][:600]
    for p, b in keep.items():
        open(p, 'wb').write(b)
    cli('--stages', 'preflight', expect=0)                   # repaired: the preflight passes and records the sizes
    par = rec(f'{LABEL}.preflight')['meta']['parent_dir']
    snap = sorted(glob.glob(f'{par}/master_R075_*.gaul.csv'))[0]
    with open(snap, 'r+') as f:                              # truncated in place after the preflight
        f.truncate(10)
    rc, err = cli('--stages', 'harvest,merge', expect=2)     # harvest needs no snapshot; every unit after it checks it
    r = rec(f'{LABEL}.merge')
    assert r['status'] == 'refused' and 'changed since the preflight' in r['reason'] \
        and os.path.basename(snap) in r['reason'], (r['status'], r.get('reason', '')[:600])


def test_chain_to_fork_then_union_to_list():
    make_world()
    rc, err = cli(expect=0)                                  # preflight..fork
    r = rec(f'{LABEL}.pulsars')                              # B-02 step 4: production's conform judges the groups
    assert r and r['status'] == 'no-input' and "MODE='remove'" in r['reason'], (r and r['status'], r and r['reason'])
    for unit in ('preflight', 'harvest', 'merge', 'conform', 'fitpos', 'select', 'associate.V_wb', 'associate.I_wb',
                 'associate.V_sb', 'associate.I_sb', 'fork.V', 'fork.I'):
        r = rec(f'{LABEL}.{unit}')
        assert r and r['status'] == 'done', (unit, r and r['status'], r and r['reason'])
        bad = [t for t in r['tests'] if not t['passed']]
        assert not bad, (unit, bad[:3])
        assert all(c.get('rc') == 0 for c in r['meta'].get('commands', [])), unit
    # the merge gate ran for real: duplicate + no-ledger rows refused, the rest accepted
    ref = rows(f'{U}/master_catalog_refused.csv')
    reasons = sorted(r['refusal_reason'] for r in ref)
    assert reasons == ['duplicate'] * 2 + ['no final ledger row'] * 2, reasons
    n_acc = len(rows(f'{U}/master_catalog_asrun.csv'))
    assert n_acc == 2 * (8 + 3 - 2), n_acc               # per Stokes: 8 Full + 3 23MHz rows in, 2 refused
    # the staleness gate edited the COPY of the search state only (nothing stale here; the copy exists)
    assert os.path.isdir(f'{U}/_state/{LABEL.lower()}_state/markers_V')
    # select ran for real with the verdict file: V rows carry the conform stub's fp_prediction (M8)
    sa = rows(f'{U}/selected_all.csv')
    assert sa and all(r['fp_prediction'] == 'not-FP' for r in sa if r['stokes'] == 'V')
    # the in-place calls got the stage environment (P3_PARENT_DIR from the preflight record) through core.legacy.run
    cmd = rec(f'{LABEL}.conform')['meta']['commands'][0]
    assert cmd['env']['P3_PARENT_DIR'].startswith(f'{U}/_parents/') and cmd['argv'][0] == sys.executable
    assert rec(f'{LABEL}.conform')['meta'].get('legacy_calls'), 'core.legacy.run was not used'
    # union refuses without a forced record newer than the builds (D-G22)
    rc, err = cli('--from', 'union', expect=2)
    assert 'D-G22' in rec(f'{LABEL}.union')['reason']
    # emulate `exopipe3 forced`: forced_* columns added in place to the tier survivors, then its done record
    for p in glob.glob(f'{U}/source_catalogs_*/*_sources_survivors.csv'):
        rr = rows(p)
        cols = list(rr[0].keys()) + ['forced_jy'] if rr else ['source_id', 'forced_jy']
        write_csv(p, cols, [dict(r, forced_jy='1.0') for r in rr])
    time.sleep(1.1)
    os.makedirs(f'{U}/_records/forced', exist_ok=True)
    json.dump({'status': 'done', 'finished': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'unit': LABEL},
              open(f'{U}/_records/forced/{LABEL}.json', 'w'))
    rc, err = cli('--from', 'union', expect=0)
    for unit in ('union', 'identity', 'localize', 'list'):
        r = rec(f'{LABEL}.{unit}')
        assert r['status'] == 'done', (unit, r['reason'])
        assert not [t for t in r['tests'] if not t['passed']], (unit, [t for t in r['tests'] if not t['passed']][:3])
    u = rows(f'{U}/source_catalogs/Full_sources_survivors.csv')
    assert {r['stokes'] for r in u} == {'V', 'I'} and all(r['identity_id'] for r in u)
    # one identity for source A across the tiers and Stokes (the real p3s3_identity.py)
    a = [r for p in glob.glob(f'{U}/source_catalogs_*/*_sources_survivors.csv') for r in rows(p) if r['source_id'] == 'J0530+3700']
    assert len({r['identity_id'] for r in a}) == 1 and len(a) >= 6, [(r['source_id'], r['identity_id']) for r in a]
    lst = os.path.realpath(f'{U}/TRANSIENT_LIST_{LABEL}_latest.csv')
    assert os.path.exists(lst) and 'tier' in rows_after_header(lst)[0]


def test_grouping_design_b_in_the_chain():
    """Item F.3 under the RULED default (design B + the one multiples rule): the group stage groups each Stokes of the fitted
    catalogue (no state: refused until --group-new-state), the builder runs IN PLACE through the grouping runner with the
    grouping's identities (source_id = the identity's display name, the multiples link columns), the deep fork ends no-input,
    and union..list run on the design-B catalogues; a second pass carries every ID."""
    from exopipe.phase3.lib.catalog import grouping_stage as GS
    m = sys.modules[__name__]
    make_world()
    open(f'{T}/exopipe_ops/build_source_catalogs_v3.py', 'w').write(PATCHABLE_BUILDER.lstrip())
    # a previous production pass's fork page products (the pages step wrote them; review 2026-11-04 F6): the one rule's fork
    # stage moves them aside, so serve's multiples queue refuses instead of rebuilding a stale page
    stale = [f'{U}/judgements/queue_p3s3_V_fwb_raw.txt', f'{U}/source_catalogs/pages_input_V_fsb.csv',
             f'{U}/source_catalogs/Red_pages_index_I_fwb.csv']
    for p_ in stale:
        os.makedirs(os.path.dirname(p_), exist_ok=True)
        open(p_, 'w').write('stale\n')
    # the grouping maps positions to the common frame through production's astrom_frame, in place (read-only: the real module
    # and its frozen inputs, loaded by the stub root's name)
    open(f'{T}/exopipe_ops/astrom_frame.py', 'w').write(
        f"import sys as _s\n_s.path.append({REAL_OPS!r})\n_R = {REAL_OPS + '/astrom_frame.py'!r}\n"
        f"exec(compile(open(_R).read(), _R, 'exec'), globals())\n")
    m._GROUPING, m._MRULE = 'design-b', 'one-rule'
    try:
        rc, err = cli('--until', 'group', expect=2)
        r = rec(f'{LABEL}.group.V')
        assert r and r['status'] == 'refused' and 'D-G24' in r['reason'] and 'new=True' in r['reason'], (r and r['reason'])
        assert not os.path.exists(f'{U}/_grouping_state/grouping_deep_V_state.json')
        rc, err = cli('--from', 'group', '--until', 'fork', '--group-new-state', expect=0)
        for S in ('V', 'I'):
            r = rec(f'{LABEL}.group.{S}')
            assert r['status'] == 'done', (S, r['reason'])
            assert not [t for t in r['tests'] if not t['passed']], [t for t in r['tests'] if not t['passed']][:3]
            assert {t['name'].split('.')[-1] for t in r['tests']} >= {'grouping_table', 'grouping_state'}, r['tests']
            f = rec(f'{LABEL}.fork.{S}')
            assert f['status'] == 'no-input' and 'one-rule' in f['reason'], (S, f['status'], f['reason'])
            assert not glob.glob(f'{U}/source_catalogs_{S}_f*')
            moved = f['meta'].get('fork_page_products_moved') or []
            assert sorted(moved) == sorted(p_ for p_ in stale if f'_{S}_f' in p_), (S, moved)
        assert not any(os.path.exists(p_) for p_ in stale), [p_ for p_ in stale if os.path.exists(p_)]
        for S in ('V', 'I'):
            T_ = {x['cand_id']: x for x in rows(f'{U}/grouping_{S}/detections.csv')}
            assert len(T_) == sum(1 for x in rows(f'{U}/master_catalog.csv') if x['stokes'] == S)
            for t in ('wb', 'sb'):
                a = rec(f'{LABEL}.associate.{S}_{t}')
                assert a['status'] == 'done', (S, t, a['reason'])
                line = ' '.join(a['meta'].get('grouping_builder') or [])
                assert 'GROUPING-BUILDER done' in line and 'mode design-b' in line, line
                for p in glob.glob(f'{U}/source_catalogs_{S}_{t}/*_source_members.csv'):
                    for mr in rows(p):
                        assert mr['source_id'] == T_[mr['cand_id']]['identity_name'], (mr['source_id'], T_[mr['cand_id']])
                for p in glob.glob(f'{U}/source_catalogs_{S}_{t}/*_sources_survivors.csv'):
                    rr = rows(p)
                    assert rr and set(GS.SOURCE_COLS) <= set(rr[0]), (p, list(rr[0]) if rr else None)
                    for x in rr:
                        e = T_[x['best_cand_id']]
                        assert x['source_id'] == x['identity_id'] == e['identity_name'], x
                        assert (x['group_identity_id'], x['multiples_id'], x['multiple']) == \
                            (e['identity_id'], e['multiples_id'], e['multiple']), (x, e)
                        assert x['multi_integration'] == e['multiple'], x
        # downstream on the design-B catalogues: a forced record, then union..list as production runs them
        for p in glob.glob(f'{U}/source_catalogs_*/*_sources_survivors.csv'):
            rr = rows(p)
            write_csv(p, (list(rr[0].keys()) if rr else ['source_id']) + ['forced_jy'], [dict(r_, forced_jy='1.0') for r_ in rr])
        time.sleep(1.1)
        os.makedirs(f'{U}/_records/forced', exist_ok=True)
        json.dump({'status': 'done', 'finished': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'unit': LABEL},
                  open(f'{U}/_records/forced/{LABEL}.json', 'w'))
        rc, err = cli('--from', 'union', expect=0)
        for unit in ('union', 'identity', 'localize', 'list'):
            r = rec(f'{LABEL}.{unit}')
            assert r['status'] == 'done', (unit, r['reason'])
            assert not [t for t in r['tests'] if not t['passed']], (unit, [t for t in r['tests'] if not t['passed']][:3])
        # the identity stage under design B: the grouping's identity per Stokes, written into every tier file and the union
        # (p3s3_identity.py's single linkage is not run), checked by the identity_table content test against the grouping
        r = rec(f'{LABEL}.identity')
        assert r['meta'].get('identity_design_b') and not any('p3s3_identity' in ' '.join(c.get('argv') or [])
                                                              for c in r['meta'].get('legacy_calls', [])), r['meta']
        assert any(t['name'].endswith('identity_table.identity_columns') and 'design-B identity' in (t.get('detail') or '')
                   for t in r['tests']), [t['name'] for t in r['tests']][:5]
        for S in ('V', 'I'):
            T_ = {x['cand_id']: x for x in rows(f'{U}/grouping_{S}/detections.csv')}
            tiers = {}
            for t in ('wb', 'sb'):
                for p in glob.glob(f'{U}/source_catalogs_{S}_{t}/*_sources_survivors.csv'):
                    for x in rows(p):
                        assert x['identity_id'] == T_[x['best_cand_id']]['identity_name'], x
                        tiers.setdefault(x['identity_id'], set()).add(f'{S}_{t}')
            for p in glob.glob(f'{U}/source_catalogs_{S}_[ws]b/*_sources_survivors.csv'):
                for x in rows(p):
                    assert x['identity_tiers'] == '+'.join(sorted(tiers[x['identity_id']])), x
        u_rows = rows(f'{U}/source_catalogs/Full_sources_survivors.csv')
        assert u_rows and all(x['identity_tiers'] for x in u_rows)
        # a second pass over the same detections: every identity and multiples ID carried, no ID event
        ids0 = {S: {x['cand_id']: (x['identity_id'], x['multiples_id']) for x in rows(f'{U}/grouping_{S}/detections.csv')}
                for S in ('V', 'I')}
        rc, err = cli('--from', 'group', '--until', 'associate', expect=0)
        for S in ('V', 'I'):
            ids1 = {x['cand_id']: (x['identity_id'], x['multiples_id']) for x in rows(f'{U}/grouping_{S}/detections.csv')}
            assert ids1 == ids0[S], S
            ev = [json.loads(l) for l in open(f'{U}/grouping_{S}/events.jsonl')]
            assert [e['event'] for e in ev] == ['run'], ev[:3]
    finally:
        m._GROUPING, m._MRULE = 'production', 'production'


def rows_after_header(p):
    with open(p) as f:
        return list(csv.DictReader([ln for ln in f if not ln.startswith('#')]))


def test_stale_directory_refused_d9():
    assert rec(f'{LABEL}.identity'), 'run test_chain_to_fork_then_union_to_list first'
    stale = f'{U}/source_catalogs_V_old'
    os.makedirs(stale, exist_ok=True)
    shutil.copy(f'{U}/source_catalogs_V_wb/Full_sources_survivors.csv', f'{stale}/Full_sources_survivors.csv')
    rc, err = cli('--stages', 'identity', expect=2)
    assert 'source_catalogs_V_old' in rec(f'{LABEL}.identity')['reason']
    shutil.rmtree(stale)


def test_changed_input_makes_records_stale():
    assert rec(f'{LABEL}.fork.V'), 'run the chain first'
    p = f'{U}/selected_V_wb.csv'
    keep = open(p).read()
    open(p, 'a').write(keep.splitlines()[1] + '\n')
    rc, err = cli('--from', 'fork', '--stokes', 'V', expect=2)
    r = rec(f'{LABEL}.fork.V')
    assert r['status'] == 'refused' and 'selected_V_wb.csv' in r['reason'], r['reason']
    open(p, 'w').write(keep)


def test_overrides_of_bound_keys_refused():
    make_world()
    rc, err = cli('--stages', 'preflight', '--set', 'catalog_conform.SANITY_X=4.0')
    assert rc == 2 and 'PROTECTED' in err, (rc, err[-400:])
    rc, err = cli('--stages', 'preflight', '--set', 'astrometry.K_ASSOC=3.1', expect=2)
    assert 'carry as literals' in rec(f'{LABEL}.preflight')['reason']
    rc, err = cli('--stages', 'preflight', '--set', 'catalog_conform.TWILIGHT_RISING_DEG=-10.0', expect=2)
    assert 'does not use' in rec(f'{LABEL}.preflight')['reason']
    rc, err = cli('--dry-run', '--stages', 'preflight', '--set', 'catalog_conform.TABLE_PREFIX[sublst]=x', expect=0)


def test_harvest_refuses_unharvested_outside_production():
    make_world()
    with open(f'{T}/ops_state/{LABEL.lower()}_state/published_I.csv', 'a') as f:
        f.write('05h_Red,nodeB,20300105T000000\n')
    rc, err = cli('--stages', 'harvest', expect=2)
    assert 'D-G3' in rec(f'{LABEL}.harvest')['reason']


def test_b30_unharvested_merge():
    """B-30 (a): a published attempt the merge finds unharvested (its ledger gone after the harvest) refuses the merge under
    the ruled catalog_merge.UNHARVESTED 'refuse', the site named; under production's 'warn' p3s3_merge drops its rows with a
    warning and the merged_record content test refuses the unit anyway (rows read != accepted + refused)."""
    make_world()
    cli('--stages', 'preflight,harvest', expect=0)
    led = sorted(p for p in glob.glob(f'{T}/collect/**/transients/Run_{LABEL}/v_deep_ledger.csv', recursive=True))
    assert len(led) == 4, led
    os.remove(led[0])
    cli('--stages', 'merge', expect=2)
    r = rec(f'{LABEL}.merge')
    assert r['status'] == 'refused' and r['reason'].startswith('B-30 (catalog_merge.UNHARVESTED=refuse)'), r['reason'][:300]
    cli('--stages', 'merge', '--set', 'catalog_merge.UNHARVESTED=warn', expect=2)
    r = rec(f'{LABEL}.merge')
    assert 'merged_record' in r['reason'] and r['meta'].get('unharvested_warning'), r['reason'][:300]


def test_merge_refresh_replaces_a_changed_attempt():
    """The merge refresh (production exopipe_ops e668670, the deep chat 2026-10-30): the first merge records each (cell, Stokes)'s
    attempt in merge_provenance.csv; when one pair's published attempt changes, the next merge replaces that pair's merged candidate
    directories with the new attempt's (the superseded ones moved to .superseded_<UTC>/), leaves the other pairs alone, passes its
    own content test and records the new attempt; the stage records both provenance files and the script's refresh lines.
    Production attributes a merged candidate directory to its (cell, Stokes) by the deep search's naming (VD_/ID_<lst>_<date>_<band>_<n>),
    so the stamps here carry that name."""
    make_world()
    att_dir = lambda cell, st, att: (f'{T}/collect/{LABEL.lower()}_collect_nodeA/attempts/{cell}_{st}_{att}/transients/Run_{LABEL}'
                                     f'/candidates')
    stamps = {('05h_Full', 'V'): 'VD_05h_2030-01-01_Full_0001', ('05h_Full', 'I'): 'ID_05h_2030-01-01_Full_0001'}
    for (cell, st), name in stamps.items():
        os.makedirs(f'{att_dir(cell, st, "20300104T000000")}/{name}')
        open(f'{att_dir(cell, st, "20300104T000000")}/{name}/info.txt', 'w').write('first search')
    cli('--stages', 'preflight,harvest,merge', expect=0)
    prov = {(r['cell'], r['stokes']): r for r in rows(f'{U}/merge_provenance.csv')}
    assert set(prov) == {(c, s_) for c in ('05h_Full', '05h_23MHz') for s_ in ('V', 'I')}, sorted(prov)
    r = rec(f'{LABEL}.merge')
    assert any('content test PASS' in h for h in r['meta'].get('merge_refresh', [])), r['meta'].get('merge_refresh')
    assert os.path.abspath(f'{U}/merge_provenance.csv') in r['meta'].get('out_fp', {}), sorted(r['meta'].get('out_fp', {}))[:6]
    assert os.path.exists(f'{U}/refusals_provenance.csv')
    assert all(open(f'{U}/candidates/{n}/info.txt').read() == 'first search' for n in stamps.values())
    # a re-search of 05h_Full Stokes V: a newer attempt, published in place of the first, whose stamp differs
    node, att, _ = make_attempt('05h_Full', 'V', att='20300105T000000')
    name = stamps[('05h_Full', 'V')]
    os.makedirs(f'{att_dir("05h_Full", "V", att)}/{name}')
    open(f'{att_dir("05h_Full", "V", att)}/{name}/info.txt', 'w').write('re-searched')
    pub = f'{T}/ops_state/{LABEL.lower()}_state/published_V.csv'
    lines = [ln if not ln.startswith('05h_Full,') else f'05h_Full,{node},{att}\n' for ln in open(pub)]
    open(pub, 'w').writelines(lines)
    cli('--stages', 'preflight,harvest,merge', expect=0)       # preflight copies the search state the merge reads (non-production --out)
    assert open(f'{U}/candidates/{name}/info.txt').read() == 're-searched', 'the changed pair kept the earlier stamp'
    sup = glob.glob(f'{U}/.superseded_*/**/{name}/info.txt', recursive=True)
    assert sup and open(sup[0]).read() == 'first search', sup
    assert open(f'{U}/candidates/{stamps[("05h_Full", "I")]}/info.txt').read() == 'first search', 'an unchanged pair was replaced'
    prov2 = {(r['cell'], r['stokes']): r for r in rows(f'{U}/merge_provenance.csv')}
    assert prov2[('05h_Full', 'V')]['attempt_dir'].endswith(f'05h_Full_V_{att}'), prov2[('05h_Full', 'V')]
    assert all(prov2[k]['attempt_dir'] == prov[k]['attempt_dir'] for k in prov if k != ('05h_Full', 'V')), 'an unchanged pair moved'
    r = rec(f'{LABEL}.merge')
    assert r['status'] == 'done' and any('content test PASS' in h for h in r['meta'].get('merge_refresh', [])), r['meta']
    assert any('1 of 4 (cell, Stokes) replaced' in h for h in r['meta'].get('merge_refresh', [])), r['meta'].get('merge_refresh')

def test_merge_refusal_record_keeps_extraction_rows():
    """The merge's refusal record (production exopipe_ops 2309a0e, the deep chat 2026-10-31): a source-extraction refusal row
    (phase3_deep._refuse_extraction: lst, scope, date, Stokes, reason, ATTEMPT TAG, UTC time) carries its attempt tag in the sixth
    field, not the last. An attempt holding an extraction_<S> file beside another gate's file keeps its extraction rows in the
    merged record - read by the last field, the UTC time sorts below the search tag and the rows were dropped - and every merged
    row ends in the cell's newest tag, as the record's readers (the newest tag per band) expect."""
    make_world()
    att = '20300104T000000'
    d = f'{T}/collect/{LABEL.lower()}_collect_nodeA/attempts/05h_Full_V_{att}/transients/Run_{LABEL}/refusals'
    os.makedirs(d, exist_ok=True)
    tier = [['05h', 'Full', DATES[0], 'V', '0.210', '0.180', '2.33', att]]
    extr = [['05h', 'Full', DATES[1], 'V', 'source extraction failed', att, '2030-01-04T00:10:00Z']]
    for g, rr in (('searchtier', tier), ('extraction_V', extr)):
        with open(f'{d}/05h_Full_{g}_excluded.csv', 'w', newline='') as f:
            csv.writer(f, lineterminator='\n').writerows(rr)
    cli('--stages', 'preflight,harvest,merge', expect=0)
    got = {g: [r for r in csv.reader(open(f'{U}/refusals/05h_Full_{g}_excluded.csv')) if r]
           for g in ('searchtier', 'extraction_V') if os.path.exists(f'{U}/refusals/05h_Full_{g}_excluded.csv')}
    assert set(got) == {'searchtier', 'extraction_V'}, ('the extraction rows were dropped from the merged record', sorted(got))
    assert got['extraction_V'] == [extr[0][:-1] + [att]], got['extraction_V']
    assert got['searchtier'] == tier, got['searchtier']
    r = rec(f'{LABEL}.merge')
    assert r['status'] == 'done' and any('content test PASS' in h for h in r['meta'].get('merge_refresh', [])), r['meta']


def test_b02_check_mode_in_the_chain():
    """B-02 steps 2 and 4 in the chain: under the ruled catalog_conform.MODE 'check' the pulsar stage runs between the merge
    and the check and refuses a run whose attempts were not searched under the once-in-search rules (the synthetic run's
    attempts are production-fleet ones: no rules record), naming them, before any conform; under 'remove' with the ruled
    DEEP_ELEV_EPOCH 'frame-mid' the first unit refuses - production's conform judges at mid-hour, so the record would
    claim a convention it never applied."""
    mod = sys.modules[__name__]
    make_world()
    try:
        mod._MODE, mod._EPOCH = 'check', 'frame-mid'
        cli('--stages', 'preflight,harvest,merge,pulsars,conform', expect=2)
        r = rec(f'{LABEL}.pulsars')
        assert r['status'] == 'refused' and 'not searched under the once-in-search rules' in r['reason'] \
            and "searched under 'production'" in r['reason'], r['reason'][:300]
        assert rec(f'{LABEL}.conform') is None, 'conform ran after a refused pulsar stage'
        mod._MODE, mod._EPOCH = 'remove', 'frame-mid'
        cli('--stages', 'preflight', expect=2)
        r = rec(f'{LABEL}.preflight')
        assert r['status'] == 'refused' and "MODE='remove'" in r['reason'] and "'mid-hour'" in r['reason'], r['reason'][:300]
        # a 'remove' pass (pulsars no-input), then conform alone under 'check': refused, naming the stage to run (not a crash)
        mod._MODE, mod._EPOCH = 'remove', 'mid-hour'
        cli('--stages', 'preflight,harvest,merge,pulsars', expect=0)
        assert rec(f'{LABEL}.pulsars')['status'] == 'no-input'
        mod._MODE, mod._EPOCH = 'check', 'frame-mid'
        cli('--stages', 'conform', expect=2)
        r = rec(f'{LABEL}.conform')
        assert r['status'] == 'refused' and 'run the pulsars stage first' in r['reason'], (r['status'], r['reason'][:300])
    finally:
        mod._MODE, mod._EPOCH = 'remove', 'mid-hour'


def test_b02_check_chain_to_conform():
    """B-02 steps 2 and 4 through the chain under the ruled 'check': the world's attempts given a once-in-search rules record,
    the pulsar stage runs to done (its catalogue recorded as class pulsar_catalog, whose content test passes: the merged
    rows, nothing restored or removed) and conform runs the check, which refuses cleanly - a 'refused' unit with its report
    copied into the stage directory, not an error (the stub world's attempts hold no frame_noise.csv nor search frames)."""
    import json as _json
    from exopipe.phase3.lib.search import removal_rules as RR
    mod = sys.modules[__name__]
    make_world()
    from exopipe.phase3.lib.search import size_rule as SZ
    for d in sorted(glob.glob(f'{T}/collect/*/attempts/*')):
        with open(os.path.join(d, RR.RULES_RECORD), 'w') as f:
            _json.dump(RR.rules_record('once-in-search'), f)
        rd = os.path.join(d, 'transients', f'Run_{LABEL}')       # ... and searched under the configured size / axis rule
        if os.path.isdir(rd) and SZ.configured_record('deep')['rule'] == 'envelope':
            with open(os.path.join(rd, SZ.RUN_RECORD), 'w') as f:
                _json.dump(SZ.configured_record('deep'), f)
    try:
        mod._MODE, mod._EPOCH = 'check', 'frame-mid'
        cli('--stages', 'preflight,harvest,merge,pulsars,conform', expect=2)
        r = rec(f'{LABEL}.pulsars')
        assert r['status'] == 'done', (r['status'], r['reason'][:300])
        t = [x for x in r['tests'] if x['name'].startswith('pulsar_catalog.')]
        assert t and all(x['passed'] for x in t), t
        c = rec(f'{LABEL}.conform')
        assert c['status'] == 'refused' and 'conform check exit 2' in c['reason'] and c['meta'].get('check_report') \
            and os.path.exists(c['meta']['check_report']), (c['status'], c['reason'][:300])
    finally:
        mod._MODE, mod._EPOCH = 'remove', 'mid-hour'


def stub_conform(kind):
    """The conform stub's axis switch (its AXIS-SWITCH line): 'today' - none, as production's p3s2_conform.py now; 'patched' -
    the pending patch's (docs/pending_production_patch_p3s2_conform_axis.diff: os.environ.get('P3_CONFORM_AXIS', '1')); 'ignored'
    - the switch read and then the step forced on (a script that does not act on it)."""
    p = f'{T}/exopipe_ops/p3s2_conform.py'
    lines = open(p).read().splitlines(True)
    i = next(i for i, ln in enumerate(lines) if '# AXIS-SWITCH' in ln)
    sw = "AXIS_STEP = os.environ.get('P3_CONFORM_AXIS', '1') != '0'"
    lines[i] = {'today': 'AXIS_STEP = True', 'patched': sw, 'ignored': sw + '; AXIS_STEP = True'}[kind] + '   # AXIS-SWITCH\n'
    open(p, 'w').writelines(lines)


def size_world(records=True):
    """make_world, every attempt searched under the configured size / axis rule (its record), and source A's Stokes I Full
    detection of the first night (S/N 9) given an axis ratio of 2.4 x beam: the rule keeps it (deep I high band at S/N 9:
    limit 2.73), production's conformance axis step (2.0 above 8 sigma) removes it. Returns its cand_id."""
    import json as _json
    from exopipe.phase3.lib.search import size_rule as SZ
    make_world()
    if records:
        for rd in sorted(glob.glob(f'{T}/collect/*/attempts/*/transients/Run_{LABEL}')):
            with open(os.path.join(rd, SZ.RUN_RECORD), 'w') as f:
                _json.dump(SZ.configured_record('deep'), f)
    p = glob.glob(f'{T}/collect/*/attempts/05h_Full_I_*/transients/Run_{LABEL}/master_catalog.csv')[0]
    rr = rows(p)
    hit = [r for r in rr if r['date'] == DATES[0] and r['ra'] == f'{SRC["A"][0]:.5f}' and r['snr'] == '9.00']
    assert len(hit) == 1, hit
    hit[0]['axis_norm'] = '2.4'
    write_csv(p, list(rr[0].keys()), rr)
    return hit[0]['cand_id']


def test_size_rule_in_the_chain():
    """The size / axis rule (R-GREGG-20261022-ALL-APPROVED-SIZE-GATE-READMISSION-SUBAGENTS) in the chain: attempts of two rules
    are refused at the merge, before their ledgers are joined; attempts searched under the rule give the builder (and the
    fork) P3_SIZE_GATE_7_8=inf - its 7-8 sigma gate replaced - and production's do not. The conformance's fixed AXIS step on
    the rule's attempts (catalog_conform.CONFORM_AXIS_UNDER_RULE, PENDING GREGG), with the fork stub running production's
    step for its validity run as multiples_fork.py does: 'second_gate' (default) - a row at 9 sigma with axis 2.4 (kept by
    the rule) is removed by the main conformance and by the fork, each recorded once in its unit, the pass completing;
    'replace' - refused at the merge, before any conformance, while production's script has no switch; with the pending
    patch's switch P3_CONFORM_AXIS=0 reaches the conformance and the fork and the row is kept (flagged); a script that reads
    the switch but still removes is refused after its conformance."""
    import json as _json
    from exopipe.phase3.lib.search import size_rule as SZ
    rec_ = SZ.configured_record('deep')
    make_world()
    rds = sorted(glob.glob(f'{T}/collect/*/attempts/*/transients/Run_{LABEL}'))
    assert len(rds) >= 2, rds
    with open(os.path.join(rds[0], SZ.RUN_RECORD), 'w') as f:
        _json.dump(rec_, f)
    cli('--stages', 'preflight,harvest,merge', expect=2)
    r = rec(f'{LABEL}.merge')
    assert r['status'] == 'refused' and 'size rules' in r['reason'] and 'REFUSING' in r['reason'], (r['status'], r['reason'][:300])
    # 'second_gate' (the default): production's axis step stands on the rule's attempts - removed, recorded, not refused
    wide = size_world()
    cli('--stages', 'preflight,harvest,merge,conform,fitpos,select,associate,fork', expect=0)
    m = rec(f'{LABEL}.merge')['meta']
    assert m['size_rule'] == rec_ and m['conform_axis'] == 'second_gate', (m.get('size_rule'), m.get('conform_axis'))
    for K in ('V_wb', 'I_wb'):
        a = rec(f'{LABEL}.associate.{K}')
        assert a['meta']['size_rule'] == 'envelope' and a['meta']['commands'][0]['env'].get('P3_SIZE_GATE_7_8') == 'inf', \
            (K, a['meta'].get('size_rule'), a['meta']['commands'][0]['env'])
    AXW = 'axis ratio 2.4 x beam > 2.0 above 8.0 sigma (Gregg 2026-09-05)'
    c_ = rec(f'{LABEL}.conform')
    sg = c_['meta']['size_rule_second_gate']['conform']
    assert sg['mode'] == 'second_gate' and sg['rows'] == 1 and sg['basis'] == SZ.AXIS_STEP_BASIS['second_gate'], sg
    assert any('as the second gate' in n for n in c_['notes']), c_['notes']
    assert 'P3_CONFORM_AXIS' not in c_['meta']['commands'][0]['env']
    assert [(x['cand_id'], x['removal_reason']) for x in rows(f'{U}/master_catalog_removed.csv') if x['cand_id'] == wide] == \
        [(wide, AXW)]
    fk = rec(f'{LABEL}.fork.I')
    assert fk['status'] == 'done' and fk['meta']['commands'][0]['env'].get('P3_SIZE_GATE_7_8') == 'inf', fk['status']
    fsg = fk['meta']['size_rule_second_gate']['multiples fork I']
    both = [x['cand_id'] for n in ('master_catalog_removed.csv', 'master_catalog_removed_A.csv')
            for x in rows(f'{U}/fork_I/{n}')]
    assert fsg['rows'] == 1 and both == [wide, wide], (fsg, both)       # one row, though two records hold it
    assert rec(f'{LABEL}.fork.V')['meta']['size_rule_second_gate']['multiples fork V']['rows'] == 0
    # 'replace' while production's p3s2_conform.py has no switch: refused at the merge, before any stage runs it
    REPL = ('--set', 'catalog_conform.CONFORM_AXIS_UNDER_RULE=replace')
    size_world()
    cli('--stages', 'preflight,harvest,merge,conform', *REPL, expect=2)
    r = rec(f'{LABEL}.merge')
    assert r['status'] == 'refused' and SZ.CONFORM_AXIS_PATCH in r['reason'] and 'P3_CONFORM_AXIS' in r['reason'], r['reason'][:400]
    assert rec(f'{LABEL}.conform') is None, 'the conformance ran after the merge refused'
    # ... with the pending patch's switch: P3_CONFORM_AXIS=0 to the conformance and the fork; the row kept and flagged
    wide = size_world()
    stub_conform('patched')
    cli('--stages', 'preflight,harvest,merge,conform,fitpos,select,associate,fork', *REPL, expect=0)
    c_ = rec(f'{LABEL}.conform')
    assert c_['meta']['commands'][0]['env'].get('P3_CONFORM_AXIS') == '0', c_['meta']['commands'][0]['env']
    assert c_['meta']['size_rule_second_gate']['conform'] == dict(mode='replace', basis=SZ.AXIS_STEP_BASIS['replace'], rows=0,
                                                                  examples=[]), c_['meta']['size_rule_second_gate']
    kept = [x for x in rows(os.path.join(c_['meta']['copies']['master_catalog.csv'])) if x['cand_id'] == wide]
    assert len(kept) == 1 and kept[0]['axis_flag'].startswith('axis > 2.0 x beam (kept'), kept
    for S in ('V', 'I'):
        fk = rec(f'{LABEL}.fork.{S}')
        assert fk['status'] == 'done' and fk['meta']['commands'][0]['env'].get('P3_CONFORM_AXIS') == '0', (S, fk['status'])
    assert not [x for n in glob.glob(f'{U}/fork_I/master_catalog_removed*.csv') for x in rows(n) if x['cand_id'] == wide]
    # ... a script that reads the switch and removes all the same: refused after its conformance (the switch did not act)
    size_world()
    stub_conform('ignored')
    cli('--stages', 'preflight,harvest,merge,conform', *REPL, expect=2)
    r = rec(f'{LABEL}.conform')
    assert r['status'] == 'refused' and 'did not switch its axis step off' in r['reason'], r['reason'][:400]
    make_world()                                          # production's attempts: the builder keeps its own 1.7
    cli('--stages', 'preflight,harvest,merge,conform,fitpos,select,associate', expect=0)
    a = rec(f'{LABEL}.associate.I_wb')
    assert a['meta']['size_rule'] == 'production' and 'P3_SIZE_GATE_7_8' not in a['meta']['commands'][0]['env'], a['meta']
    assert rec(f'{LABEL}.merge')['meta']['conform_axis'] == "production's step"


def test_b30_identity_pre_check():
    """B-30 (a) pre-check of ops/p3s3_identity.py (deep_ported.identity_row_fallbacks, in its own parse order): a survivors
    row without a positive finite pos_err_arcsec refuses the identity stage under the ruled value (the module would take 60
    arcsec; a NaN would stop every association of the run) and is counted under 'fallback' (production's result); an
    unparsable row is a counted per-item drop."""
    from exopipe.phase3.lib.catalog import deep_ported as P
    d = f'{T}/b30_identity/source_catalogs_V_wb'
    os.makedirs(d, exist_ok=True)
    base = dict(source_id='J0000+0000', band='Full', ra='10.0', dec='20.0', best_snr='9.0', pos_err_arcsec='60.0')
    cases = ['60.0', '', '0', 'nan', 'inf', '-5']
    rws = [dict(base, pos_err_arcsec=v) for v in cases] + [dict(base, ra='x'), dict(base, best_snr='y', pos_err_arcsec='')]
    write_csv(f'{d}/Full_sources_survivors.csv', list(base), rws)
    noerr, bad = P.identity_row_fallbacks({f'{d}/Full_sources_survivors.csv'})
    tag = 'source_catalogs_V_wb/Full_sources_survivors.csv'
    assert noerr == [f'{tag}:{i}' for i in (1, 2, 3, 4, 5)] and bad == [f'{tag}:6', f'{tag}:7'], (noerr, bad)
    # the chain: one tier row loses its error between the fork and the union (where the forced step edits the files)
    make_world()
    cli(expect=0)
    blank = None
    for p in sorted(glob.glob(f'{U}/source_catalogs_*/*_sources_survivors.csv')):
        rr = rows(p)
        cols = list(rr[0].keys()) + ['forced_jy'] if rr else ['source_id', 'forced_jy']
        if rr and blank is None:
            rr[0]['pos_err_arcsec'], blank = '', p
        write_csv(p, cols, [dict(r, forced_jy='1.0') for r in rr])
    time.sleep(1.1)
    os.makedirs(f'{U}/_records/forced', exist_ok=True)
    json.dump({'status': 'done', 'finished': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'unit': LABEL},
              open(f'{U}/_records/forced/{LABEL}.json', 'w'))
    k = 'fallbacks.CATALOG_IDENTITY_POS_ERR_FALLBACK'
    cli('--from', 'union', expect=2)
    r = rec(f'{LABEL}.identity')
    assert r['status'] == 'refused' and k in r['reason'] and os.path.basename(blank) in r['reason'], r['reason'][:300]
    cli('--from', 'identity', '--set', f'{k}=fallback', expect=0)
    r = rec(f'{LABEL}.identity')
    assert r['status'] == 'done' and r.get('fallbacks') == {k: 1}, (r['status'], r.get('fallbacks'))


def test_b30_ephem_pre_check():
    """B-30 (a) pre-check of ops/p3s2_ephem_filter.py (deep_ported.ephem_retro_rows): I rows of selected_all.csv without
    in-chain Jupiter / Moon separations would be computed at the epoch DATE-OBS, not the frame mid-time - the select stage
    refuses under the ruled value, counts under 'fallback'; the filter's printed LST-approximation count is parsed."""
    global WORLD_I_JUPITER_DEG
    from exopipe.phase3.lib.catalog import deep_ported as P
    line = ('ephem retro-filter: dropped 0 Jupiter + 0 Moon I rows; 12 remain; epoch times from Phase-2/frame headers, '
            '3 (cell,date) fell back to the LST approximation')
    assert P.ephem_time_fallbacks(line) == 3 and P.ephem_time_fallbacks(line.replace(' 3 (', ' 0 (')) == 0
    k = 'fallbacks.CATALOG_EPHEM_RETRO_FALLBACK'
    WORLD_I_JUPITER_DEG = ''
    try:
        make_world()
    finally:
        WORLD_I_JUPITER_DEG = '50.0'
    cli('--stages', 'preflight,harvest,merge,conform,fitpos,select', expect=2)
    r = rec(f'{LABEL}.select')
    assert r['status'] == 'refused' and k in r['reason'] and 'I row(s)' in r['reason'], r['reason'][:300]
    n = P.ephem_retro_rows(U)           # the refused select wrote selected_all.csv: the chain re-runs to it
    k2 = 'fallbacks.CATALOG_EPHEM_TIME_FALLBACK'      # the world's 2030 epochs have no Phase-2 image: the filter's LST guess
    cli('--stages', 'preflight,harvest,merge,conform,fitpos,select', '--set', f'{k}=fallback', '--set', f'{k2}=fallback',
        expect=0)
    r = rec(f'{LABEL}.select')
    fb = r.get('fallbacks') or {}
    assert n > 0 and r['status'] == 'done' and fb.get(k) == n and fb.get(k2, 0) > 0 and set(fb) == {k, k2}, (n, r['status'], fb)
    assert P.ephem_retro_rows(U) == 0, 'the in-place filter fills the separations it computes'


def test_binding_against_production_and_a_mismatch():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.catalog import deep_binding as B
    CFG = unpatch_cfg()
    res = B.check(CFG, REAL_OPS)                                  # read-only AST of the production modules
    assert len(res) >= 70 and all(r['ok'] for r in res), [r['detail'] for r in res if not r['ok']][:4]
    fake = f'{T}/fake_ops'
    os.makedirs(fake, exist_ok=True)
    src = open(f'{REAL_OPS}/p3s3_identity.py').read().replace('K_ASSOC = 3.034', 'K_ASSOC = 3.5')
    open(f'{fake}/p3s3_identity.py', 'w').write(src)
    r = [x for x in B.check(CFG, fake) if x['key'] == 'catalog_identity.IDENTITY_K'][0]
    assert not r['ok'] and '3.5' in r['detail'], r


def test_ported_heredocs_match_production_bytes():
    """The PYS split and PYU0 union of ops/p3s3_pass.sh (read from the production file) and the ports write the same bytes."""
    from exopipe.phase3.lib.catalog import deep_ported as P
    text = open(f'{REAL_OPS}/p3s3_pass.sh').read()
    blocks = {}
    for tag in ('PYS', 'PYU0'):
        m = re.search(r"<<'%s'.*?\n(.*?)\n%s\n" % (tag, tag), text, re.S)
        assert m, tag
        blocks[tag] = m.group(1)
    a, b = f'{T}/hd_a/Run_X', f'{T}/hd_b/Run_X'
    for d in (a, b):
        os.makedirs(f'{d}/source_catalogs', exist_ok=True)
        rr = [dict(cand_id=f'c{i}', stokes=s, band=bd, snr='6', note='x,y') for i, (s, bd) in
              enumerate([('V', 'Full'), ('I', 'Full'), ('V', '23MHz'), ('I', '41MHz'), ('V', 'Red')])]
        write_csv(f'{d}/selected_all.csv', ['cand_id', 'stokes', 'band', 'snr', 'note'], rr)
        for K, extra in (('V_wb', 'a'), ('I_wb', 'b'), ('V_sb', 'c'), ('I_sb', 'd'), ('V', 'view'), ('V_fwb', 'fork')):
            for kind in ('sources_survivors', 'sources_master', 'source_members'):
                band = 'Full' if 'wb' in K or K == 'V' else '23MHz'
                write_csv(f'{d}/source_catalogs_{K}/{band}_{kind}.csv', ['source_id', 'band', f'col_{extra}'],
                          [dict(source_id='J0530+3700', band=band, **{f'col_{extra}': K}),
                           dict(source_id='J0530+3700', band=band, **{f'col_{extra}': 'dup'}),
                           dict(source_id='', band=band, **{f'col_{extra}': 'noid'})])
    env = dict(os.environ, P3S_LABEL='X')
    for tag in ('PYS', 'PYU0'):
        code = blocks[tag].replace("""U=f"/data07/gh/phase3/transients/Run_{os.environ.get('P3S_LABEL','P3S3')}\"""", f'U={a!r}')
        assert f'U={a!r}' in code, tag
        r = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr[-600:]
    P.split_selection(b)
    files = {K: glob.glob(f'{b}/source_catalogs_{K}/*.csv') for K in ('V_wb', 'V_sb', 'I_wb', 'I_sb')}
    P.union_tiers(b, files, ('V_wb', 'V_sb', 'I_wb', 'I_sb'), ('sources_survivors', 'sources_master', 'source_members'))
    fa = sorted(os.path.relpath(p, a) for p in glob.glob(f'{a}/selected_*.csv') + glob.glob(f'{a}/source_catalogs/*.csv'))
    fb = sorted(os.path.relpath(p, b) for p in glob.glob(f'{b}/selected_*.csv') + glob.glob(f'{b}/source_catalogs/*.csv'))
    assert fa == fb and len(fa) == 5 + 6, (fa, fb)
    for rel in fa:
        assert open(f'{a}/{rel}', 'rb').read() == open(f'{b}/{rel}', 'rb').read(), rel


def test_content_tests_catch_defects():
    from exopipe.phase3.core import validate
    import exopipe.phase3.lib.catalog.tests  # noqa: F401
    d = f'{T}/ct'
    os.makedirs(d, exist_ok=True)

    def ok(cls, p, **ctx):
        return all(r['passed'] for r in validate.run(cls, p, **ctx))
    # transient list: header counts must equal the rows
    tl = f'{d}/TRANSIENT_LIST_X_20300101T0000.csv'
    body = ('source_id,stokes,tier,best_snr,identity_id,page\n'
            'J1,V,V_wb,9.0,,http://localhost:8766/source/J1\nJ2,I,I_fsb,8.0,J9,http://localhost:8773/source/J9\n')
    open(tl, 'w').write('# Rows per tier (one row per tier, Stokes and source): V_wb 1, I_fsb 1; Stokes V 1, Stokes I 1.\n' + body)
    assert ok('transient_list', tl)
    open(tl, 'w').write('# Rows per tier (one row per tier, Stokes and source): V_wb 2, I_fsb 1; Stokes V 2, Stokes I 1.\n' + body)
    assert not ok('transient_list', tl)
    # localization offsets: one cand_id twice in one identity (M12)
    lo = f'{d}/localization_offsets.csv'
    write_csv(lo, ['cand_id', 'identity_id', 'det_sigma_arcsec'], [dict(cand_id='c1', identity_id='J1', det_sigma_arcsec='5'),
                                                                     dict(cand_id='c1', identity_id='J1', det_sigma_arcsec='5')])
    assert not ok('localization_table', lo)
    # fitted positions (2026-10-14 replay): peak_ra/peak_dec equal the input as NUMBERS at the writer's 5 decimals - an input
    # '42.9882' written '42.98820' passes; a real move fails; a fitted row whose statistical error is 0.0 fails (two such Gaussian
    # fits in production's Run_P3S5, fitted S/N 26,889 and 47,189)
    from exopipe.phase3.lib.catalog import tests as CT
    fi, fo = f'{d}/fit_in.csv', f'{d}/fit_out.csv'
    write_csv(fi, ['cand_id', 'ra', 'dec'], [dict(cand_id='c1', ra='336.51088', dec='42.9882'), dict(cand_id='c2', ra='31.3357', dec='6.5207')])
    json.dump({'fitted_rows': 2, 'fitted_on_pixel_centre': 0, 'refused': []}, open(f'{d}/fit_positions_summary.json', 'w'))

    def out_rows(stat='0.5', pra2='31.33570'):
        base = dict(pos_kind='gaussian', pos_err_maj_arcsec='9.0', pos_err_min_arcsec='9.0', pos_err_pa_deg='', pos_stat_maj_arcsec=stat,
                    pos_stat_min_arcsec=stat, pos_sys_arcsec='8.0', pos_snr='9.0', pos_offset_arcmin='0.1', pos_reason='x')
        return [dict(base, cand_id='c1', ra='336.51090', dec='42.98821', pos_ra='336.51090', pos_dec='42.98821', peak_ra='336.51088',
                     peak_dec='42.98820'),
                dict(base, cand_id='c2', ra='31.33572', dec='6.52071', pos_ra='31.33572', pos_dec='6.52071', peak_ra=pra2,
                     peak_dec='6.52070')]
    cols = ['cand_id', 'ra', 'dec'] + list(CT.DET_COLS)
    ctx = dict(input=fi, **{'fit_positions_summary.json': f'{d}/fit_positions_summary.json'})
    write_csv(fo, cols, out_rows())
    assert ok('fitted_catalog', fo, **ctx), validate.run('fitted_catalog', fo, **ctx)
    write_csv(fo, cols, out_rows(pra2='31.33580'))
    assert not ok('fitted_catalog', fo, **ctx), 'a moved peak position passed'
    write_csv(fo, cols, out_rows(stat='0.0'))
    assert not ok('fitted_catalog', fo, **ctx), 'a fitted row with a 0.0 statistical error passed'
    from exopipe.phase3.config import CFG
    CFG.catalog_fitpos.ZERO_STAT_ERROR = 'report'                     # production (the parity file): reported, not failed
    try:
        res = validate.run('fitted_catalog', fo, **ctx)
        assert all(r['passed'] for r in res) and any('reported' in r['detail'] for r in res), res
    finally:
        CFG.catalog_fitpos.ZERO_STAT_ERROR = 'refuse'
    # union without the stokes column (pre-D9)
    os.makedirs(f'{d}/run/source_catalogs', exist_ok=True)
    write_csv(f'{d}/run/source_catalogs/Full_sources_survivors.csv', ['source_id', 'band'], [dict(source_id='J1', band='Full')])
    assert not ok('union_catalog', f'{d}/run/source_catalogs/Full_sources_survivors.csv')
    # selection: |SNR| below the floor
    write_csv(f'{d}/run/selected_V_wb.csv', ['cand_id', 'stokes', 'band', 'snr', 'parent_excluded'],
              [dict(cand_id='c1', stokes='V', band='Full', snr='4.0', parent_excluded='0')])
    assert not ok('selection', f'{d}/run/selected_V_wb.csv')
    # fitted catalogue: a row without a kind
    write_csv(f'{d}/run/fitted.csv', ['cand_id', 'ra', 'dec', 'pos_kind', 'pos_ra', 'pos_dec', 'pos_err_maj_arcsec',
                                      'pos_err_min_arcsec', 'pos_err_pa_deg', 'pos_stat_maj_arcsec', 'pos_stat_min_arcsec',
                                      'pos_sys_arcsec', 'pos_snr', 'pos_offset_arcmin', 'pos_reason', 'peak_ra', 'peak_dec'],
              [dict(cand_id='c1', ra='1', dec='2', pos_kind='', pos_ra='1', pos_dec='2')])
    assert not ok('fitted_catalog', f'{d}/run/fitted.csv')
    # review 2026-10-13: selection M8 - deep: fp_prediction must be the verdict file's; sub-LST (cadence 10min, no verdict
    # file, spec Q7): not compared (catalog-tiers F2)
    os.makedirs(f'{d}/m8', exist_ok=True)
    write_csv(f'{d}/m8/master_catalog.csv', ['cand_id', 'fp_prediction'], [dict(cand_id='c1', fp_prediction='not-FP')])
    sel_cols = ['cand_id', 'stokes', 'band', 'snr', 'parent_excluded', 'cadence', 'fp_prediction']
    write_csv(f'{d}/m8/selected_all.csv', sel_cols, [dict(cand_id='c1', stokes='V', band='Full', snr='6.0',
                                                          parent_excluded='0', cadence='deep', fp_prediction='pending')])
    assert not ok('selection', f'{d}/m8/selected_all.csv'), 'deep selection with a pending fp_prediction passed (M8)'
    write_csv(f'{d}/m8/selected_all.csv', sel_cols, [dict(cand_id='c1', stokes='V', band='Full', snr='6.0',
                                                          parent_excluded='0', cadence='10min', fp_prediction='pending')])
    from exopipe.phase3.lib.catalog import tests as T_
    assert T_.selection_rows(f'{d}/m8/selected_all.csv')[0], 'sub-LST selection compared with a verdict file it never read'
    assert T_.selection_rows(f'{d}/m8/selected_all.csv', tier='sublst')[0]
    # review 2026-10-13: frame_beam_bmaj.csv (D8 cache, ops 283a513) is a frame_table: a non-positive BMAJ fails
    write_csv(f'{d}/run/frame_beam_bmaj.csv', ['stokes', 'lst', 'band', 'date', 'bmaj_deg'],
              [dict(stokes='V', lst='05h', band='Full', date='2030-01-01', bmaj_deg='0.21')])
    assert ok('frame_table', f'{d}/run/frame_beam_bmaj.csv')
    write_csv(f'{d}/run/frame_beam_bmaj.csv', ['stokes', 'lst', 'band', 'date', 'bmaj_deg'],
              [dict(stokes='V', lst='05h', band='Full', date='2030-01-01', bmaj_deg='0')])
    assert not ok('frame_table', f'{d}/run/frame_beam_bmaj.csv')


def test_compileall():
    files = [f'{REPO}/exopipe/phase3/steps/catalog.py', f'{REPO}/exopipe/phase3/config/steps/catalog.py',
             __file__] + glob.glob(f'{REPO}/exopipe/phase3/lib/catalog/deep_*.py') + [f'{REPO}/exopipe/phase3/lib/catalog/tests.py']
    r = subprocess.run([sys.executable, '-m', 'py_compile'] + files, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


ORDER = ['test_help_and_dry_run', 'test_preflight_refuses_missing_parent', 'test_preflight_fails_on_an_empty_parent_copy',
         'test_chain_to_fork_then_union_to_list',
         'test_stale_directory_refused_d9', 'test_changed_input_makes_records_stale', 'test_overrides_of_bound_keys_refused',
         'test_harvest_refuses_unharvested_outside_production', 'test_b30_unharvested_merge',
         'test_merge_refresh_replaces_a_changed_attempt', 'test_merge_refusal_record_keeps_extraction_rows',
         'test_size_rule_in_the_chain', 'test_b30_identity_pre_check',
         'test_b30_ephem_pre_check', 'test_b02_check_mode_in_the_chain', 'test_b02_check_chain_to_conform',
         'test_binding_against_production_and_a_mismatch', 'test_grouping_design_b_in_the_chain',
         'test_ported_heredocs_match_production_bytes', 'test_content_tests_catch_defects', 'test_compileall']


def main(names):
    t0 = time.time()
    names = names or ORDER
    fails = 0
    try:
        for n in names:
            t = time.time()
            try:
                globals()[n]()
                print(f'PASS {n} ({time.time() - t:.1f} s)', flush=True)
            except Exception as e:
                fails += 1
                import traceback
                print(f'FAIL {n}: {type(e).__name__}: {e}', flush=True)
                traceback.print_exc()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{len(names) - fails} of {len(names)} passed in {time.time() - t0:.0f} s')
    return 1 if fails else 0


if __name__ == '__main__':
    os.chdir(REPO)
    sys.exit(main(sys.argv[1:]))
