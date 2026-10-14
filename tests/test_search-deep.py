#!/usr/bin/env python3
"""Tests of unit search-deep (exopipe3 search --tier deep): plain asserts, runnable with python, under 2 minutes.

    EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py NUMPY_MADVISE_HUGEPAGE=0 \
        /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_search-deep.py

Synthetic data in a fresh /tmp/exopipe3_tests_search-deep/ (removed at the end). READ-ONLY checks on production: the content
tests of two published P3S5 attempts, and `exopipe3 search --dry-run` on a real cell (skipped where the inputs are not on this
node). No engine run on real data, no ssh, no GPU: the one engine child started here refuses at its first gate (no I master).
"""
import ast
import csv
import glob
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
TMP = '/tmp/exopipe3_tests_search-deep'
PY = sys.executable
COLLECT = '/lustre/gh/main/phase3_collect/p3s5_collect_lwacalim07/attempts'
REAL_SMALL = f'{COLLECT}/00h_18MHz_V_20261006T042022'
REAL_BIG = f'{COLLECT}/12h_Red_I_20261013T004458'

from exopipe.phase3.core import env as _env   # noqa: E402
_env.apply()
_env.legacy()
from exopipe.phase3.config import CFG, ConfigRefusal   # noqa: E402
from exopipe.phase3.core import record, validate     # noqa: E402
from exopipe.phase3.lib.search import deep_step as DS, tests as T   # noqa: E402

RESULTS = []


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond), detail))
    print(f"{'ok  ' if cond else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)


def cli(*args, timeout=180):
    env = dict(os.environ, PYTHONPATH=REPO)
    p = subprocess.run([PY, '-m', 'exopipe.phase3', 'search'] + list(args), capture_output=True, text=True, env=env,
                       timeout=timeout, cwd=REPO)
    return p.returncode, p.stdout + p.stderr


# ---------------------------------------------------------------------------------------------------------------------------
def test_config():
    secs = CFG.sections()
    for s in ('search', 'search_inputs', 'search_detect', 'search_tables', 'search_fpgate', 'search_fit', 'search_run'):
        check(f'config section {s} loaded', s in secs)
    try:
        CFG.set('search_detect.MSTRUCT_K', '4.0')
        check('engine literal search_detect.MSTRUCT_K refuses --set (PROTECTED_KEYS)', False)
    except ConfigRefusal:
        check('engine literal search_detect.MSTRUCT_K refuses --set (PROTECTED_KEYS)', True)
    CFG.authorize(out=TMP)
    CFG.set('search_detect.SEARCH_TIER_SIGMA', '4.0')          # RULED: allowed into a non-production --out
    check('ruled key overridden into a non-production out', CFG.search_detect.SEARCH_TIER_SIGMA == 4.0)
    CFG.reset()
    check('override reset', CFG.search_detect.SEARCH_TIER_SIGMA == 5.0)
    for sec in ('search', 'search_inputs', 'search_detect', 'search_tables', 'search_fpgate', 'search_fit', 'search_run'):
        bad = [k for k in getattr(getattr(CFG, sec), 'PROTECTED_KEYS', ()) if not hasattr(getattr(CFG, sec), k)]
        check(f'{sec}.PROTECTED_KEYS name existing keys', not bad, bad)
    check('interpreter keys from the fixed vocabulary', {CFG.search.ENGINE_PYTHON, CFG.search_fpgate.PYTHON,
                                                        CFG.search_detect.FLOW_PYTHON} <= {'default', 'flow', 'py36', 'casa', 'tess'})


def test_engine_truth():
    mism = DS.literal_mismatches()
    check('every engine literal in config equals the engine', not mism, mism[:3])
    from exopipe.phase3.lib.search import deep_search as E, candidates as CA
    check('GLED_COLS of the tests = the engine', T.GLED_COLS == E.GLED_COLS)
    check('GIMG_COLS of the tests = the engine', T.GIMG_COLS == E.GIMG_COLS)
    src = inspect.getsource(CA.write_candidates)
    m = re.search(r'fields = (\[.*?\])', re.sub(r'#[^\n]*', '', src), re.S)
    fields = ast.literal_eval(m.group(1))
    check('catalogue fields of the tests = candidates.write_candidates', T.CAT_FIELDS == fields)
    srcs = inspect.getsource(E)
    reasons = set(re.findall(r"reason='([^']+)'", srcs)) | {'v110:FP', 'v110:marginal'}
    for mod in ('search_i', 'search_v', 'identification'):
        mm = __import__(f'exopipe.phase3.lib.search.{mod}', fromlist=['x'])
        reasons |= {s.lower() for s in re.findall(r"'(REJECT_[A-Z_]+)'", inspect.getsource(mm))}
    missing = sorted(reasons - T.VETO_REASONS)
    check('every veto_reason the engine can write is known to gate_ledger.complete', not missing, missing)


def test_env_and_items():
    env = DS.engine_env('/x/coadd', '/x/att', '/x/flow', 'P3S5', '/x/arch', '/x/tmp')
    check('engine env: elevation floor', env['P3_DEEP_MIN_ELEVATION'] == '25.0')
    check('engine env: table elevation bands', env['P3_TABLE_ELEV_BANDS'] == '25-35,35-50,50-90')
    check('engine env: deep min epochs 1, sigma-stack min 3', (env['P3_DEEP_MIN_EPOCHS'], env['P3_MIN_EPOCHS']) == ('1', '3'))
    check('engine env: roots', (env['P3_STAGING_ROOT'], env['P3_IRESID_ROOT'], env['P3_ARCHIVE_RUN']) ==
          ('/x/att', '/x/att/Run_P3S5/iresid', CFG.paths.ARCHIVE_RUN))
    comp = [k for k in env if k in ('P3_ALLOW_OLD_MASTERS', 'P3_BEAM_GATE_OFF', 'P3_V110_OFF', 'P3_V110_READING',
                                    'P3_SSTACK_REMOVES', 'P3_STRIATION_OFF', 'P3_GATE_LEDGER_ONLY', 'P3_STREAM_RESID',
                                    'P3_REFINE_OFF', 'P3_ARCHIVE_WRITE', 'P3_TABLE_PREFIX', 'P3_TABLE_RECORD_ONLY')]
    check('engine env: no comparison / sub-LST variable at production values', not comp, comp)
    CFG.authorize(out=TMP)
    CFG.set('search_detect.BEAM_GATE_ON', 'false')
    env2 = DS.engine_env('/x', '/x', '/x', 'P3S5', '/x', '/x')
    check('comparison key BEAM_GATE_ON False -> P3_BEAM_GATE_OFF=1', env2.get('P3_BEAM_GATE_OFF') == '1')
    check('comparison deviation reported', DS.comparison_deviations() == [('search_detect.BEAM_GATE_ON', False)])
    CFG.reset()

    class A:
        stokes = None; cells = '12h_Red'; lsts = None; epochs = None; nights = None; scopes = None; exclude_cells = None
        replay_attempt = None
    it = DS.plan_items(A, list(CFG.search.STAGES))
    check('bulk items: V before I', [i.name for i in it] == ['12h_Red_V', '12h_Red_I'])
    A.cells, A.epochs, A.scopes = None, '12h_2025-05-10', 'Red,18MHz'
    it = DS.plan_items(A, list(CFG.search.STAGES))
    check('per-hour items: every selected band of the LST, incremental', [i.name for i in it] ==
          ['12h_18MHz_V_2025-05-10', '12h_18MHz_I_2025-05-10', '12h_Red_V_2025-05-10', '12h_Red_I_2025-05-10']
          and {i.mode for i in it} == {'incremental'} and it[0].base == 'published', [i.name for i in it])
    A.nights = '2025-05-10'
    try:
        DS.plan_items(A, list(CFG.search.STAGES))
        check('--epochs with --nights refused', False)
    except ValueError:
        check('--epochs with --nights refused', True)
    A.epochs, A.nights, A.scopes, A.cells, A.replay_attempt = None, None, None, None, REAL_BIG
    it = DS.plan_items(A, ['fit', 'verify'])
    check('replay item from an attempt directory', len(it) == 1 and it[0].name == '12h_Red_I_replay_20261013T004458'
          and it[0].mode == 'replay')
    from exopipe.phase3.steps.search import resolve_stages

    class S:
        stages = None; from_stage = 'detect'; until_stage = 'verify'; dry_run = False
    check('--from/--until cut the stage list', resolve_stages(S, CFG.search.STAGES) == ['detect', 'fit', 'verify'])
    S.from_stage = S.until_stage = None; S.stages = 'verify,fit'
    check('--stages in pipeline order', resolve_stages(S, CFG.search.STAGES) == ['fit', 'verify'])
    S.dry_run = True
    check('--dry-run = preflight only', resolve_stages(S, CFG.search.STAGES) == ['preflight'])
    check('production trees refused (G3)', DS.production_tree('/lustre/gh/main/phase3_collect/x') and
          DS.production_tree('/fast/gh/main/phase3/s2work_p3s5') and not DS.production_tree(TMP))


# ---------------------------------------------------------------------------------------------------------------------------
# synthetic attempt
CELL, LST, SCOPE, ATT_TAG, D1, D2, D3 = '12h_Red', '12h', 'Red', '20261013T120000', '2030-01-01', '2030-01-02', '2030-01-03'


def make_frame(path, sources, n=256, cd=0.03, bmaj=0.15, seed=7):
    import numpy as np
    from astropy.io import fits
    rng = np.random.default_rng(seed)
    d = rng.normal(0.0, 1.0, (n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    s = bmaj / cd / math.sqrt(8 * math.log(2))
    for x, y, a in sources:
        d += a * np.exp(-0.5 * (((xx - x) / s) ** 2 + ((yy - y) / s) ** 2))
    h = fits.Header()
    h['CTYPE1'], h['CTYPE2'] = 'RA---SIN', 'DEC--SIN'
    h['CRVAL1'], h['CRVAL2'] = 180.0, 37.0
    h['CRPIX1'], h['CRPIX2'] = n / 2 + 1, n / 2 + 1
    h['CDELT1'], h['CDELT2'] = -cd, cd
    h['BMAJ'], h['BMIN'], h['BPA'] = bmaj, bmaj, 0.0
    h['MJD-MID'] = 62502.3
    h['STOKES'] = 'I'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.writeto(path, d.astype(np.float32), h, overwrite=True)
    from astropy.wcs import WCS
    return WCS(h).celestial


def make_attempt():
    att = os.path.join(TMP, 'attempts', f'{CELL}_I_{ATT_TAG}')
    rd = os.path.join(att, 'transients', 'Run_P3S5')
    os.makedirs(os.path.join(rd, 'refusals'), exist_ok=True)
    srcs = [(100, 120, 25.0), (160, 90, 6.0)]
    w = make_frame(T.frame_path(att, 'P3S5', LST, SCOPE, D1, 'I'), srcs)
    make_frame(T.frame_path(att, 'P3S5', LST, SCOPE, D2, 'I'), [], seed=8)
    cat = []
    for i, (x, y, a) in enumerate(srcs):
        ra, dec = [float(v) for v in w.pixel_to_world_values(x, y)]
        row = dict.fromkeys(T.CAT_FIELDS, '')
        row.update(cand_id=f'ID_{LST}_{D1}_{SCOPE}_{i:04d}', lst=LST, date=D1, band=SCOPE, stokes='I', cadence='deep',
                   ra=round(ra % 360, 5), dec=round(dec, 5), snr=round(max(a, 7.0), 2), mstruct=1.0)
        cat.append(row)
    with open(os.path.join(rd, 'master_catalog.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=T.CAT_FIELDS)
        wr.writeheader()
        wr.writerows(cat)
    led = []
    for r in cat:
        g = dict.fromkeys(T.GLED_COLS, '')
        g.update(lst=LST, scope=SCOPE, stokes='I', date=D1, ra=r['ra'], dec=r['dec'], snr=r['snr'], final=1, mstruct=1.0,
                 veto_reason='', morph_pass=1)
        led.append(g)
    led.append(dict(dict.fromkeys(T.GLED_COLS, ''), lst=LST, scope=SCOPE, stokes='I', date=D2, ra=181.0, dec=36.0, snr=5.1,
                    final=0, veto_reason='reject_local_snr', morph_pass=1))
    led.append(dict(dict.fromkeys(T.GLED_COLS, ''), lst=LST, scope=SCOPE, stokes='I', date=D2, ra=181.2, dec=36.2, snr=7.5,
                    final=0, veto_reason='mstruct_I', mstruct=3.0, morph_pass=1))
    with open(os.path.join(rd, 'v_deep_ledger.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=T.GLED_COLS)
        wr.writeheader()
        wr.writerows(led)
    with open(os.path.join(rd, 'v_deep_images.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(T.GIMG_COLS)
        wr.writerow([LST, SCOPE, 'I', D1, 5, 4, '', 4, 2, 9, 25.0])
        wr.writerow([LST, SCOPE, 'I', D2, 3, 2, '', 2, 0, 5, 25.0])
    with open(os.path.join(rd, 'refusals', f'{CELL}_beam_excluded.csv'), 'w') as f:
        f.write(f'{LST},{SCOPE},{D3},1.62,1.5,{ATT_TAG}\n')
    return att, rd


def results_of(cls, path, **ctx):
    rs = validate.run(cls, path, **ctx)
    return all(r['passed'] for r in rs), '; '.join(f"{r['name']}={r['passed']}:{str(r['detail'])[:120]}" for r in rs)


def test_content_tests_synthetic():
    att, rd = make_attempt()
    for cls, name in (('detections', 'master_catalog.csv'), ('gate_ledger', 'v_deep_ledger.csv'),
                      ('image_ledger', 'v_deep_images.csv')):
        ok, det = results_of(cls, os.path.join(rd, name), stokes='I', cell=CELL, tier='deep')
        check(f'{cls} passes on a consistent synthetic attempt', ok, det)
    ok, det = results_of('refusal_record', os.path.join(rd, 'refusals', f'{CELL}_beam_excluded.csv'))
    check('refusal_record passes', ok, det)
    # breakages
    p = os.path.join(rd, 'master_catalog.csv')
    keep = open(p).read()
    with open(p, 'a') as f:
        f.write(f'ID_{LST}_{D1}_{SCOPE}_0002,{LST},{D1},{SCOPE},I,deep,,170.0,30.0,1,,9.0' + ',' * 19 + '\n')
    ok, det = results_of('detections', p, stokes='I', cell=CELL)
    check('detections FAILS on a catalogue row without a final ledger row', not ok and 'without a final ledger row' in det, det)
    open(p, 'w').write(keep)
    p = os.path.join(rd, 'v_deep_ledger.csv')
    keep_l = open(p).read()
    open(p, 'w').write(keep_l.replace(',reject_local_snr,', ',not_a_reason,'))
    ok, det = results_of('gate_ledger', p, stokes='I', cell=CELL)
    check('gate_ledger FAILS on an unknown veto reason', not ok and 'unknown veto_reason' in det, det)
    open(p, 'w').write(keep_l)
    ok, det = results_of('gate_ledger', p, stokes='I', cell=CELL, tier='sublst')
    check('gate_ledger (sub-LST): no table rows -> passes', ok, det)
    p = os.path.join(rd, 'v_deep_images.csv')
    keep_i = open(p).read()
    with open(p, 'a') as f:
        f.write(f'{LST},{SCOPE},I,{D3},1,1,,1,0,1,25.0\n')
    ok, det = results_of('image_ledger', p, stokes='I', cell=CELL)
    check('image_ledger FAILS on an image row of a refused date', not ok and 'refused dates' in det, det)
    open(p, 'w').write(keep_i.replace(',25.0', ',20.0'))
    ok, det = results_of('image_ledger', p, stokes='I', cell=CELL)
    check('image_ledger FAILS on the pre-fix literal 20.0 in a current attempt', not ok and 'elev_min_deg' in det, det)
    open(p, 'w').write(keep_i)
    p = os.path.join(rd, 'refusals', f'{CELL}_beam_excluded.csv')
    open(p, 'w').write(f'{LST},{SCOPE},{D3},1.62,{ATT_TAG}\n')
    ok, det = results_of('refusal_record', p)
    check('refusal_record FAILS on a wrong column count', not ok and 'columns' in det, det)
    open(p, 'w').write(f'{LST},{SCOPE},{D3},1.62,1.5,{ATT_TAG}\n')
    # review 2026-10-13: verify_p3s3_stage conditions the tests now carry
    p = os.path.join(rd, 'master_catalog.csv')
    keep = open(p).read()
    lines = keep.splitlines()
    dup = lines[1].split(',')
    dup[0] = f'ID_{LST}_{D1}_{SCOPE}_0002'
    open(p, 'w').write('\n'.join(lines + [','.join(dup)]) + '\n')
    ok, det = results_of('detections', p, stokes='I', cell=CELL)
    check('detections FAILS on a duplicate row (same date and position; verify_p3s3_stage.py:144-149)',
          not ok and 'duplicate catalogue rows' in det, det)
    open(p, 'w').write(keep)
    extra = T.frame_path(att, 'P3S5', LST, SCOPE, D3, 'I')
    make_frame(extra, [], seed=9)
    ok, det = results_of('image_ledger', os.path.join(rd, 'v_deep_images.csv'), stokes='I', cell=CELL)
    check('image_ledger FAILS on a frame without an image row (frames = image rows, verify_p3s3_stage.py:130)',
          not ok and 'without an image row' in det, det)
    os.remove(extra)
    dd = os.path.join(TMP, 'data_decisions.csv')
    with open(dd, 'w') as fh:
        fh.write('lst,subband,date,verdict,reason\n')
        fh.write(f'{LST},27MHz,{D2},exclude,synthetic test\n')      # a Red member subband -> Red excluded on D2
    real_dd = CFG.paths.DATA_DECISIONS
    CFG.paths.DATA_DECISIONS = dd
    try:
        ok, det = results_of('image_ledger', os.path.join(rd, 'v_deep_images.csv'), stokes='I', cell=CELL)
        check('image_ledger FAILS on an epoch searched on a night excluded now (STALE, p3s3_stale_exclusions.py)',
              not ok and 'STALE' in det and D2 in det, det)
    finally:
        CFG.paths.DATA_DECISIONS = real_dd
    return att, rd


def test_fit_stage(att, rd):
    from exopipe.phase3.core.record import Refusal
    out = os.path.join(att, 'fit_positions.csv')
    t0 = time.time()
    s = DS.fit_attempt(att, 'P3S5', out, nproc=2)
    rows = list(csv.DictReader(open(out)))
    check('fit: one sidecar row per catalogue row, in order', [r['cand_id'] for r in rows] ==
          [r['cand_id'] for r in csv.DictReader(open(os.path.join(rd, 'master_catalog.csv')))], f'{time.time() - t0:.1f} s')
    kinds = [r['pos_kind'] for r in rows]
    check('fit: bright source Gaussian, faint source fitted (template or Gaussian)', kinds[0] == 'gaussian' and
          kinds[1] in ('template', 'gaussian'), kinds)
    check('fit: peak_ra/peak_dec = the catalogue position', all(abs(float(r['peak_ra']) - float(c['ra'])) < 1e-5 for r, c in
                                                               zip(rows, csv.DictReader(open(os.path.join(rd, 'master_catalog.csv'))))))
    check('fit: summary reports the peak-fallback fraction per frame', 'peak_fallback_frac' in s and len(s['per_frame']) == 1
          and s['per_frame'][0]['rows'] == 2, s.get('per_frame'))
    ok, det = results_of('detection_positions', out, tier='deep')
    check('detection_positions passes on the sidecar', ok, det)
    # sub-LST reading of the same class: the tier's own section (search_sublst.FIT_BLANKED), never search_fit.BLANK_SYSTEMATIC
    if hasattr(CFG, 'search_sublst'):
        ok, det = results_of('detection_positions', out, tier='sublst')
        check('detection_positions (sub-LST) FAILS when the error-model-A columns are not blank', not ok and 'not blank' in det,
              det)
        keep_s = open(out).read()
        rows_s = list(csv.DictReader(open(out)))
        for r in rows_s:
            for c in CFG.search_sublst.FIT_BLANKED:
                r[c] = ''
        with open(out, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows_s[0].keys()))
            wr.writeheader()
            wr.writerows(rows_s)
        ok, det = results_of('detection_positions', out, tier='sublst')
        check('detection_positions (sub-LST) passes with FIT_BLANKED blank', ok, det)
        open(out, 'w').write(keep_s)
    # breakage: a frame changed after the fit
    fr = T.frame_path(att, 'P3S5', LST, SCOPE, D1, 'I')
    keep = open(fr, 'rb').read()
    with open(fr, 'ab') as f:
        f.write(b'\0' * 2880)
    ok, det = results_of('detection_positions', out, tier='deep')
    check('detection_positions FAILS when the searched frame changed', not ok and 'sha1' in det, det)
    open(fr, 'wb').write(keep)
    # gates of the fit stage
    cat = os.path.join(rd, 'master_catalog.csv')
    keep_c = open(cat).read()
    lines = keep_c.splitlines()
    open(cat, 'w').write('\n'.join([lines[0] + ',pos_kind'] + [l + ',peak' for l in lines[1:]]) + '\n')
    try:
        DS.fit_attempt(att, 'P3S5', out + '.x', nproc=1)
        check('fit refuses a catalogue that already carries pos_kind', False)
    except Refusal:
        check('fit refuses a catalogue that already carries pos_kind', True)
    open(cat, 'w').write(keep_c)
    os.rename(fr, fr + '.away')
    try:
        DS.fit_attempt(att, 'P3S5', out + '.x', nproc=1)
        check('fit refuses a detection without its searched frame', False)
    except Refusal:
        check('fit refuses a detection without its searched frame', True)
    os.rename(fr + '.away', fr)


def test_publish_path(att, rd):
    """fit -> verify -> publish -> records on the synthetic attempt (after_detect, the code that follows the engine): published
    layout, register row, unit record done with every output's tests; a failing content test refuses BEFORE publishing."""
    out = os.path.join(TMP, 'out_pub')
    fr = os.path.join(TMP, 'w_pub', 'flow')
    cw = os.path.join(fr, CELL)
    os.makedirs(os.path.join(cw, 'catdir'), exist_ok=True)
    open(os.path.join(cw, 'catdir', f'master_R075_{CELL}.gaul.csv'), 'w').write('# synthetic\n')
    real = DS._tests_registered
    # classes whose tests need the real master / conformant headers (masters', frames') are left out of this synthetic run
    DS._tests_registered = lambda cls: cls not in ('master_catalog', 'search_frame', 'noise_map', 'frame_refusals') and real(cls)
    try:
        for name in ('fit_positions.csv', 'fit_positions_summary.json'):
            if os.path.exists(os.path.join(att, name)):
                os.remove(os.path.join(att, name))
        run = record.Run('search', ['test'], out, CFG, config_sections=None)
        item = DS.Item(CELL, 'I', 'bulk')
        ctx = dict(out=out, flow_root=fr, attempts=os.path.dirname(att), archive_root=os.path.join(out, 'archive_mirror'),
                   label='P3S5', tmp=os.path.join(TMP, 'w_pub', 'tmp'))
        f = dict(attempt=att, label='P3S5', att_tag=ATT_TAG, engine={'status': 'done', 'n_written': 2}, log=None,
                 admitted=[D1, D2], dates_i=[D1, D2], dates_v=[], cell_work=cw, t_detect=time.time())
        CFG.authorize(out=out)
        CFG.set('search_fit.NPROC', '2')
        with run.unit(item.name, tier='deep') as u:
            DS.after_detect(u, item, ctx, f, ['fit', 'verify', 'publish'])
        CFG.reset()
        dest = os.path.join(out, 'attempts', os.path.basename(att))
        check('publish: attempt copied in the collect layout', os.path.exists(os.path.join(dest, 'transients', 'Run_P3S5',
                                                                                          'master_catalog.csv'))
              and os.path.exists(os.path.join(dest, 'fit_positions.csv')))
        check('publish: parent catalogue copied', os.path.exists(os.path.join(out, 'parent_catalogs', f'master_R075_{CELL}.gaul.csv')))
        reg = open(os.path.join(out, 'published_I.csv')).read().split(',')
        check('publish: register row cell,host,attempt', reg[0] == CELL and reg[2].strip() == ATT_TAG, reg)
        rec = record.load_record(out, 'search', item.name)
        names = {t['name'] for t in rec['tests']}
        check('record: done, outputs at the published paths, tests recorded', rec['status'] == 'done' and
              all(o['path'].startswith(out) for o in rec['outputs']) and 'detections.schema_and_backing' in names
              and 'search.unaccounted_epochs' in names and 'search.fit_peak_fallback' in names, rec['status'] + ' ' + rec['reason'])
        # a failing content test refuses before anything is published
        out2 = os.path.join(TMP, 'out_pub2')
        ctx['out'] = out2
        p = os.path.join(rd, 'v_deep_ledger.csv')
        keep = open(p).read()
        open(p, 'w').write(keep.replace(',reject_local_snr,', ',not_a_reason,'))
        run2 = record.Run('search', ['test'], out2, CFG)
        with run2.unit(item.name, tier='deep') as u2:
            DS.after_detect(u2, item, ctx, f, ['verify', 'publish'])
        open(p, 'w').write(keep)
        check('verify failure refuses the unit and publishes nothing', u2.status == 'refused' and 'verify:' in u2.reason and
              not os.path.exists(os.path.join(out2, 'attempts')) and not os.path.exists(os.path.join(out2, 'published_I.csv')),
              u2.reason[:160])
    finally:
        DS._tests_registered = real


def test_engine_child():
    """The engine child through core.legacy.run (module mode): it starts, installs the legacy finder, imports the engine,
    calls run_deep_cell and maps its first gate (no I master: 'REFUSING') to exit 2 / Refusal; the child's code fingerprint
    reaches the unit record."""
    from exopipe.phase3.core.record import Refusal
    run = record.Run('search', ['test'], os.path.join(TMP, 'out'), CFG, dry_run=True)
    u = run.unit('child_test')
    item = DS.Item(CELL, 'I', 'bulk')
    ctx = dict(attempts=os.path.join(TMP, 'w', 'attempts'), tmp=os.path.join(TMP, 'w', 'tmp'),
               flow_root=os.path.join(TMP, 'w', 'flow'), archive_root=os.path.join(TMP, 'out', 'archive_mirror'), label='P3S5',
               out=os.path.join(TMP, 'out'))
    for d in (ctx['attempts'], ctx['tmp'], ctx['flow_root']):
        os.makedirs(d, exist_ok=True)
    f = dict(base=os.path.join(TMP, 'coadd', CELL))
    os.makedirs(f['base'], exist_ok=True)
    t0 = time.time()
    try:
        DS.stage_detect(u, item, ctx, f)
        check('engine child: first gate refuses (no I master)', False, 'no refusal')
    except Refusal as e:
        check('engine child: first gate refuses (no I master)', 'REFUSING' in str(e) and f.get('engine_rc') == 2,
              f'{str(e)[:120]} ({time.time() - t0:.1f} s)')
    code = u.meta.get('legacy_code', {})
    check('engine child: code fingerprint of the engine recorded', any(p.endswith('lib/search/deep_search.py') for p in code),
          f'{len(code)} files')
    check('engine child: environment recorded without inherited P3_*', u.meta.get('engine_env', {}).get('P3_COADD_ROOT')
          == os.path.join(TMP, 'coadd'))
    log = open(f['log']).read()
    check('engine child: log in the attempt, sentinel withheld on refusal', 'REFUSING' in log and 'P3S3-CELL-DONE' not in log)


def test_cli():
    rc, out = cli('--help')
    check('exopipe3 search --help', rc == 0 and '--deep-flow-from' in out and '--replay-attempt' in out)
    g3 = '/lustre/gh/main/phase3_collect/_exopipe3_g3_test_do_not_create'
    rc, out = cli('--tier', 'deep', '--cells', '12h_Red', '--dry-run', '--out', g3, '--cores', 'none')
    check('G3: --out under a production tree refused, nothing written', rc == 2 and not os.path.exists(g3), out.strip()[-160:])
    rc, out = cli('--tier', 'deep', '--cells', '12h_Red', '--inject', 'x.csv', '--out', os.path.join(TMP, 'cli'),
                  '--work', os.path.join(TMP, 'w'), '--cores', 'none')
    check('G4: deep --inject refused', rc == 2 and 'Q9' in out, out.strip()[-160:])
    rc, out = cli('--tier', 'deep', '--cells', '12h_Red', '--set', 'search_detect.MSTRUCT_K=4', '--dry-run', '--out',
                  os.path.join(TMP, 'cli'), '--cores', 'none')
    check('engine literal override refused at the command line', rc == 2 and 'PROTECTED' in out, out.strip()[-160:])
    if os.path.isdir('/data07/gh/phase3/coadd_sb/12h_Red'):
        rc, out = cli('--tier', 'deep', '--cells', '12h_Red', '--dry-run', '--out', os.path.join(TMP, 'dry'), '--work',
                      os.path.join(TMP, 'w'), '--cores', 'none')
        check('dry-run on the real cell 12h_Red: plan printed, exit 0, nothing written',
              rc == 0 and 'planned 2' in out and 'gate E3 membership rulings: ok' in out and not os.path.exists(
                  os.path.join(TMP, 'dry')), out.strip()[-200:])
        # per hour on a night of the master's generation record (12h Red 2025-05-10: in weights.csv, beam-refused in the
        # published attempt): a re-admitted epoch is never appended (R-SIGMA-STACK-INCLUDES-THE-RE-ADMITTED-EPOCH-...)
        rc, out = cli('--tier', 'deep', '--epochs', '12h_2025-05-10', '--scopes', 'Red', '--stokes', 'I', '--dry-run', '--out',
                      os.path.join(TMP, 'dry'), '--work', os.path.join(TMP, 'w'), '--cores', 'none')
        check('per hour: a night of the generation record refuses (full re-search ruling), nothing written',
              rc == 2 and 'R-SIGMA-STACK-INCLUDES-THE-RE-ADMITTED-EPOCH-BACKFILL-BY-RE-SEARCH' in out
              and not os.path.exists(os.path.join(TMP, 'dry')), out.strip()[-240:])
        rc, out = cli('--tier', 'deep', '--epochs', '12h_2031-01-01', '--scopes', 'Red', '--stokes', 'I', '--dry-run', '--out',
                      os.path.join(TMP, 'dry'), '--work', os.path.join(TMP, 'w'), '--cores', 'none')
        check('per hour: a night without an epoch product is no-input (exit 3)', rc == 3 and 'no epoch product' in out,
              out.strip()[-200:])
    else:
        print('skip: /data07/gh/phase3/coadd_sb not on this node (dry-run on a real cell)')


def test_preflight_review():
    """Review 2026-10-13: on a GPU host a flow cache the currency script does not call current is dropped
    (p3s5_search_fleet.sh:96-98); one search of a cell at a time on a work root (p3s5_search_fleet.sh:94,169)."""
    from exopipe.phase3.core.record import Refusal
    base = '/data07/gh/phase3/coadd_sb/12h_Red'
    if not (os.path.isdir(base) and DS._gpu_host()):
        print('skip: preflight review checks need the real cell 12h_Red on a GPU host')
        return
    fr = os.path.join(TMP, 'w_pf', 'flowroot')
    cw = os.path.join(fr, CELL)
    os.makedirs(cw)
    import numpy as np
    np.save(os.path.join(cw, 'ep_2025-05-10_up2_flow.npy'), np.zeros(4, np.float32))     # a legacy cache: no sidecar
    ctx = dict(out=os.path.join(TMP, 'pf_out'), roots=DS.roots_table(), flow_root=fr, work_fs=REPO, label='P3S5',
               flow_from=None, dry_run=True)
    item = DS.Item(CELL, 'V', 'bulk')
    try:
        f = DS.preflight(item, ctx, fix=False)
        check('GPU host, unverified cache, plan: reported as would be dropped, nothing removed',
              f['flow_state'] in ('unverified', 'partial') and any('would be dropped' in n for n in f['notes'])
              and os.path.exists(os.path.join(cw, 'ep_2025-05-10_up2_flow.npy')), f['notes'][-1:])
        f = DS.preflight(item, ctx, fix=True)
        check('GPU host, unverified cache, run: dropped, the search computes the flows',
              f['flow_state'] == 'none' and not glob.glob(os.path.join(cw, 'ep_*_flow.npy')), f['notes'][-1:])
    except Refusal as e:
        check('GPU host preflight on 12h_Red', False, str(e)[:200])
    # --resume identity: the recorded input SET (a new epoch product / LOO / V master makes a unit not resumable: the fleet's
    # adoption test incl. D6), on a record written by add_inputs over the real cell (commit made clean for the probe)
    f = DS.preflight(item, ctx, fix=False, resume_probe=True)
    small, big = DS.input_paths(item, f)
    rout = os.path.join(TMP, 'resume_out')
    run = record.Run('search', ['test'], rout, CFG)
    u = run.unit(item.name)
    DS.add_inputs(u, item, f)
    u.status = 'done'
    u.write()
    rp = record.record_path(rout, 'search', item.name)
    rec = json.load(open(rp))
    rec['commit'] = 'abc1234'
    json.dump(rec, open(rp, 'w'))
    ok1, why1 = record.resume_check(rout, 'search', item.name, inputs=small + big, admitted=f['admitted'],
                                    cfg_hash=run.config_hash)
    ok2, why2 = record.resume_check(rout, 'search', item.name, inputs=small + big + [os.path.join(
        f['base'], 'subtracted', '2031-01-01_I_sub_Red.fits')], admitted=f['admitted'], cfg_hash=run.config_hash)
    check('resume: same input set resumes; one more epoch product does not', ok1 and not ok2 and 'input set' in why2,
          (why1, why2))
    ctx['dry_run'] = False
    lock = os.path.join(fr, f'.{CELL}.search.lock')
    probe = ('import fcntl,sys; f=open(sys.argv[1],"a")\ntry:\n fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB); print("FREE")\n'
             'except BlockingIOError:\n print("HELD")')
    with DS.cell_lock(ctx, item):
        held = subprocess.run([PY, '-c', probe, lock], capture_output=True, text=True).stdout.strip()
    free = subprocess.run([PY, '-c', probe, lock], capture_output=True, text=True).stdout.strip()
    check('cell lock: a second search of the cell waits while a unit runs, free after', (held, free) == ('HELD', 'FREE'),
          (held, free))


def test_real_products():
    """READ-ONLY: this unit's content tests on two published production attempts."""
    for att, classes in ((REAL_SMALL, ('detections', 'gate_ledger', 'image_ledger', 'refusal_record', 'candidate_stamps')),
                         (REAL_BIG, ('detections', 'gate_ledger', 'image_ledger', 'refusal_record'))):
        if not os.path.isdir(att):
            print(f'skip: {att} not reachable')
            continue
        rd = glob.glob(os.path.join(att, 'transients', 'Run_*'))[0]
        files = {'detections': [os.path.join(rd, 'master_catalog.csv')], 'gate_ledger': [os.path.join(rd, 'v_deep_ledger.csv')],
                 'image_ledger': [os.path.join(rd, 'v_deep_images.csv')],
                 'refusal_record': sorted(glob.glob(os.path.join(rd, 'refusals', '*_excluded.csv'))),
                 'candidate_stamps': [os.path.join(rd, 'candidates')]}
        for cls in classes:
            for p in files[cls]:
                ok, det = results_of(cls, p)
                check(f'production {os.path.basename(att)} {cls} {os.path.basename(p)}', ok, det[:200])
    v = '/lustre/gh/main/phase3_collect/p3s5_collect_lwacalim07/v110_products/00h_18MHz/V_top100_fp_gate_p3.csv'
    if os.path.exists(v):
        ok, det = results_of('fp_gate_verdicts', v, catalog=os.path.join(REAL_SMALL, 'transients', 'Run_P3S5', 'master_catalog.csv'))
        check('production fp_gate_verdicts 00h_18MHz (with its V catalogue)', ok, det[:200])


def main():
    t0 = time.time()
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    try:
        test_config()
        test_engine_truth()
        test_env_and_items()
        att, rd = test_content_tests_synthetic()
        test_fit_stage(att, rd)
        test_publish_path(att, rd)
        test_engine_child()
        test_cli()
        test_preflight_review()
        test_real_products()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    bad = [r for r in RESULTS if not r[1]]
    print(f'\n{len(RESULTS) - len(bad)} of {len(RESULTS)} checks passed in {time.time() - t0:.0f} s')
    for n, _ok, d in bad:
        print(f'FAILED: {n}  {d}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
