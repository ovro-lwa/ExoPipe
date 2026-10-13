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
if not os.environ.get('P3_PARENT_DIR'):
    sys.exit('conform stub: no P3_PARENT_DIR - REFUSING')
if os.environ.get('P3_CONFORM_SIZE') != '0' or os.environ.get('P3_CADENCE') != 'deep' or os.environ.get('P3_TABLE_ELEV') != '1':
    sys.exit('conform stub: the pass environment is wrong - REFUSING')
cols, raw = rcsv(f'{CD}/master_catalog.csv')
if not raw:
    sys.exit('conform: no rows - REFUSING')
shutil.copy(f'{CD}/master_catalog.csv', f'{CD}/master_catalog_asrun.csv')
out, rem = [], []
for r in raw:
    if 'RMV' in r['cand_id']:
        rem.append(dict(r, removal_reason='row-line: stub removal')); continue
    r = dict(r); r['fp_prediction'] = 'not-FP' if r['stokes'] == 'V' else ''; out.append(r)
oc = cols + ['fp_prediction']
wcsv(f'{CD}/master_catalog.csv', oc, out); wcsv(f'{CD}/master_catalog_removed.csv', oc + ['removal_reason'], rem)
keys = sorted({(r['stokes'], r['lst'], r['band'], r['date']) for r in raw})
wcsv(f'{U}/frame_times.csv', ['stokes', 'lst', 'band', 'date', 'date_obs', 'date_end'],
     [dict(stokes=s, lst=l, band=b, date=d, date_obs=d + 'T12:00:00.000', date_end=d + 'T13:00:00.000') for s, l, b, d in keys])
wcsv(f'{U}/frame_beam_ratio.csv', ['stokes', 'lst', 'band', 'date', 'ratio'],
     [dict(stokes=s, lst=l, band=b, date=d, ratio='1.2') for s, l, b, d in keys])
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
import collections, json, os, shutil, sys
from _stubcat import rcsv, wcsv, build
mode, S = sys.argv[1], sys.argv[2]
U, OUT = os.environ['FORK_RUN_DIR'], os.environ['FORK_OUT_DIR']
if not os.environ.get('P3_PARENT_DIR'):
    sys.exit('FORK REFUSED: no P3_PARENT_DIR')
w = f'{U}/fork_{S}'
if mode == 'run':
    shutil.rmtree(w, ignore_errors=True); os.makedirs(w)
    cols, rows = rcsv(f'{U}/master_catalog_asrun.csv')
    rows = [dict(r) for r in rows if r['stokes'] == S and abs(float(r['snr'])) >= {'I': 7.0, 'V': 5.0}[S]]
    g = collections.defaultdict(set)
    for r in rows:
        g[(r['band'], round(float(r['ra']), 1), round(float(r['dec']), 1))].add((r['lst'], r['date']))
    for r in rows:
        r['fork_repeat'] = '1' if len(g[(r['band'], round(float(r['ra']), 1), round(float(r['dec']), 1))]) >= 2 else '0'
    wcsv(f'{w}/master_catalog.csv', cols + ['fork_repeat'], rows)
    shutil.copy(f'{w}/master_catalog.csv', f'{w}/master_catalog_asrun.csv')
    kept = [r for r in rows if r['fork_repeat'] == '1']
    wcsv(f'{w}/fork_kept_{S}.csv', cols + ['fork_repeat'], kept)
    for t in ('wb', 'sb'):
        k = [r for r in kept if ('MHz' in r['band']) == (t == 'sb')]
        if not k:
            continue
        d = f'{OUT}/source_catalogs_{S}_f{t}'
        build(k, d, min_n=2)
        wcsv(f'{d}/repeat_shuffle.csv', ['source_id', 'p_shuffle'], [])
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
                         jupiter_deg='' if st == 'I' else '50.0', moon_deg='', epoch_qa='', size_beams='1.0',
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
          ('paths', 'COADD_SB_ROOT'), ('paths', 'OPS_ROOT'))


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
    return CFG


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


def test_chain_to_fork_then_union_to_list():
    make_world()
    rc, err = cli(expect=0)                                  # preflight..fork
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


def test_compileall():
    files = [f'{REPO}/exopipe/phase3/steps/catalog.py', f'{REPO}/exopipe/phase3/_cfgdraft_catalog-deep.py',
             __file__] + glob.glob(f'{REPO}/exopipe/phase3/lib/catalog/deep_*.py') + [f'{REPO}/exopipe/phase3/lib/catalog/tests.py']
    r = subprocess.run([sys.executable, '-m', 'py_compile'] + files, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


ORDER = ['test_help_and_dry_run', 'test_preflight_refuses_missing_parent', 'test_chain_to_fork_then_union_to_list',
         'test_stale_directory_refused_d9', 'test_changed_input_makes_records_stale', 'test_overrides_of_bound_keys_refused',
         'test_harvest_refuses_unharvested_outside_production', 'test_binding_against_production_and_a_mismatch',
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
