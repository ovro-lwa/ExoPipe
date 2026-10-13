#!/usr/bin/env python3
"""Tests of `exopipe3 pages` (unit pages of the refactor window). Plain asserts; run with
    NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py \
        /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_pages.py

Synthetic part (fresh directory /tmp/exopipe3_tests_pages/, removed at the end): a stub legacy code root holding a stub
build_i14_pages_std7.py (the builder's page plan, sidecar fingerprint skip, index and queue rules in miniature) and a stub
tess_coverage.py, plus LINKS to the real in-place scripts p3s3_tier_pages_input.py, p3s3_merge_review_dir.py and
classify_transients.py. Checked: the full deep chain through the CLI (exit codes, records, planned outputs, content tests),
incremental re-run (0 rendered), one changed row (1 rendered), --full (all rendered), --dry-run writes nothing, P1 / P2 / P4 /
protected-key refusals, the environment scrub, M14 per-set layout + keyed merge, the singles pages-input port byte-identical to
the production heredoc extracted from p3s3_pass.sh, the fork rows recomputation equal to the real p3s3_tier_pages_input.py,
mirror drift detection, negative content tests.
Read-only part: the content tests on existing production products (Run_P3S5, Run_SLST14B, Run_SNAPB1R_20261004) - results are
PRINTED (production products may be mid-pass or older than their catalogues), asserted only to run without raising.
"""
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
PY = sys.executable
T = '/tmp/exopipe3_tests_pages'
OPS = '/lustre/gh/main/exopipe_ops'
INDEX_COLS = ['source_id', 'band', 'rank', 'review_dir', 'kind', 'date', 'ra', 'dec', 'flux_jy', 'xm_best', 'xm_all',
              'xm_dist_pc', 'xm_sep_arcmin', 'xm_offset_pct', 'xm_p_chance']

STUB_BUILDER = r'''#!/usr/bin/env python3
"""STUB of build_i14_pages_std7.py for tests: same argv, env, page plan, panel names, sidecar fingerprint skip, pass-T0 cleanup,
index (shard 0) and queue rules; draws a blank 1705x968 RGBA panel instead of the 7-cutout figure."""
import csv, glob, hashlib, json, os, re, sys
from PIL import Image
SURV, RUN = sys.argv[1], sys.argv[2]
SHARD, N = int(sys.argv[3]), int(sys.argv[4])
with open(os.path.join(RUN, '_stub_env_%s_%d.json' % (os.path.basename(SURV), SHARD)), 'w') as f:
    json.dump({k: v for k, v in os.environ.items() if k.startswith(('P3', 'SUBLST', 'SNAP'))}, f)
D = os.path.dirname(os.path.abspath(__file__))
CODE = hashlib.sha1(b''.join(open(p, 'rb').read() for p in (os.path.abspath(__file__), os.path.join(D, 'std7_panel.py'),
                                                               os.path.join(D, 'p3s3_astrom.py')))).hexdigest()[:16]
def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return float('nan')
rows = list(csv.DictReader(open(SURV)))
rows.sort(key=lambda r: -abs(fnum(r['snr' if os.environ.get('P3_SORT_SNR') == '1' else 'peak_jy'])))
only = set(open(os.environ['P3_PAGES_ONLY_FILE']).read().split()) if os.environ.get('P3_PAGES_ONLY_FILE') else None
for i, r in enumerate(rows):
    r['_rank'] = i + 1
    j = r['_sid']
    r['_sid'] = '%s_%s_%s_%s' % (j, r['lst'], r['date'], r['band']) if j.startswith('J') else 'i14_%04d_%s_%s_%s' % (i + 1, r['lst'], r['date'], r['band'])
K = re.search(r'pages_input_([A-Za-z]+_[a-z]+)\.csv$', SURV).group(1)
mem = {}
for mp in glob.glob('%s/source_catalogs_%s/*_source_members.csv' % (RUN, K)):
    for m in csv.DictReader(open(mp)):
        mem.setdefault((m['source_id'], m['band']), []).append(m)
jobs = []
for r in rows:
    if only is not None and r['_sid'] not in only:
        continue
    keys = [(r['lst'], r['date'])]
    for m in mem.get((r['_sid'].rsplit('_', 3)[0], r['band']), []):
        lst = m['cand_id'].split('_')[1]
        if (lst, m['det_date']) != (r['lst'], r['date']):
            keys.append((lst, m['det_date']))
    for lst, date in keys:
        jobs.append((r, lst, date))
t0 = float(os.environ.get('P3S3_PASS_T0', '0') or 0)
rend = kept = 0
for n, (r, lst, date) in enumerate(jobs):
    if n % N != SHARD:
        continue
    band = r['band']
    ddir = '%s/candidates/I/%s' % (RUN, r['_sid'])
    os.makedirs(ddir, exist_ok=True)
    png = '%s/rank%04d_%s_%s_%s.png' % (ddir, r['_rank'], lst, date, band)
    fp = hashlib.sha1(json.dumps({k: v for k, v in r.items() if k != '_rank'}, sort_keys=True).encode() + lst.encode() + date.encode()).hexdigest()
    reuse = None
    for old in glob.glob('%s/rank*_%s_%s_%s.png' % (ddir, lst, date, band)):
        try:
            if json.load(open(old[:-4] + '.page.json')).get('fp') == fp:
                reuse = old; break
        except (OSError, ValueError):
            pass
    if reuse:
        if reuse != png:
            os.replace(reuse, png); os.remove(reuse[:-4] + '.page.json')
        os.utime(png, None); kept += 1
    else:
        Image.new('RGBA', (1705, 968), (255, 255, 255, 255)).save(png); rend += 1
    json.dump(dict(fp=fp, code=CODE, ctx_lines=['hours searched el>25d: 1 of 1'], rendered=not reuse), open(png[:-4] + '.page.json', 'w'))
    for old in glob.glob('%s/rank*_%s_%s_%s.png' % (ddir, lst, date, band)):
        if old != png and (not t0 or os.path.getmtime(old) < t0):
            os.remove(old)
            if os.path.exists(old[:-4] + '.page.json'): os.remove(old[:-4] + '.page.json')
if SHARD == 0 and only is None:
    stk = K if K.split('_')[-1] not in ('wb', 'sb') else K[0]
    for b in sorted({r['band'] for r in rows}):
        with open('%s/source_catalogs/%s_pages_index_%s.csv' % (RUN, b, stk), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(@@COLS@@)
            for r in rows:
                if r['band'] == b:
                    w.writerow([r['_sid'], b, r['_rank'], 'candidates/I/' + r['_sid'], 'I14', r['date'], r['ra'], r['dec'], r['peak_jy'], '', '', '', '', '', ''])
    os.makedirs(RUN + '/judgements', exist_ok=True)
    with open('%s/judgements/%s' % (RUN, os.environ.get('P3S3_QUEUE_NAME', 'queue_i14.txt')), 'w') as f:
        f.write('# queue_name: i14_posthoc_std7\n# selection: stub\n')
        for r in rows: f.write(r['_sid'] + '\n')
print('DONE shard %d/%d (%d pages: %d rendered, %d unchanged and kept) -> %s' % (SHARD, N, rend + kept, rend, kept, RUN))
'''.replace('@@COLS@@', repr(INDEX_COLS))

STUB_TESS = r'''#!/usr/bin/env python3
import csv, sys, time
run = sys.argv[sys.argv.index('--run') + 1]
rows = [r for r in csv.DictReader(open(run + '/master_catalog.csv')) if r['stokes'] in ('I', 'V')]
with open(run + '/tess_coverage.csv', 'w', newline='') as f:
    f.write('# tess_coverage.py %s; orbit table cached; stub\n' % time.strftime('%Y-%m-%dT%H:%M:%SZ'))
    w = csv.DictWriter(f, fieldnames=['cand_id', 'stokes', 'status'], extrasaction='ignore'); w.writeheader()
    for r in rows: w.writerow(dict(cand_id=r['cand_id'], stokes=r['stokes'], status='not observed'))
print('stub tess', len(rows))
'''


def sh(cmd, env=None):
    p = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return p.returncode, p.stdout


def write_csv(path, rows, cols=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = cols or list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def make_stub_root():
    root = os.path.join(T, 'code', 'exopipe_ops')
    os.makedirs(os.path.join(root, 'tess'), exist_ok=True)
    open(os.path.join(root, 'build_i14_pages_std7.py'), 'w').write(STUB_BUILDER)
    open(os.path.join(root, 'std7_panel.py'), 'w').write('# stub\n')
    open(os.path.join(root, 'p3s3_astrom.py'), 'w').write('# stub\n')
    open(os.path.join(root, 'tess', 'tess_coverage.py'), 'w').write(STUB_TESS)
    for f in ('p3s3_tier_pages_input.py', 'p3s3_merge_review_dir.py', 'classify_transients.py'):
        os.symlink(os.path.join(OPS, f), os.path.join(root, f))
    return root


def make_run(U):
    """A small deep run: V_wb (2 sources, one with a second epoch), V_fwb (1 source), I_wb (1 source)."""
    surv = {'V_wb': [dict(source_id='J0101+0101', best_cand_id='Vn_04h_2025-10-14_Full_0001', best_date='2025-10-14', band='Full',
                          ra='15.3', dec='1.02', best_snr='-9.5', peak_jy_best='0.5', n_det='2', identity_id='J0101+0101',
                          xm_best='', stokes='V'),
                     dict(source_id='J0202+0202', best_cand_id='Vn_05h_2025-10-15_Full_0002', best_date='2025-10-15', band='Full',
                          ra='30.5', dec='2.03', best_snr='7.0', peak_jy_best='0.3', n_det='1', identity_id='J0202+0202',
                          xm_best='', stokes='V')],
            'V_fwb': [dict(source_id='J0101+0101', best_cand_id='Vn_04h_2025-10-14_Full_0001', best_date='2025-10-14', band='Full',
                           ra='15.3', dec='1.02', best_snr='-9.5', peak_jy_best='0.5', n_det='2', identity_id='J0101+0101',
                           xm_best='', stokes='V')],
            'I_wb': [dict(source_id='J0303+0303', best_cand_id='In_06h_2025-10-16_Red_0003', best_date='2025-10-16', band='Red',
                          ra='45.7', dec='3.04', best_snr='12.0', peak_jy_best='1.1', n_det='1', identity_id='J0303+0303',
                          xm_best='', stokes='I')]}
    for K, rows in surv.items():
        write_csv(f'{U}/source_catalogs_{K}/{rows[0]["band"]}_sources_survivors.csv', [{k: v for k, v in r.items() if k != 'stokes'} for r in rows])
    write_csv(f'{U}/source_catalogs_V_wb/Full_source_members.csv',
              [dict(source_id='J0101+0101', band='Full', cand_id='Vn_04h_2025-10-14_Full_0001', det_date='2025-10-14',
                    det_ra='15.3', det_dec='1.02', det_snr='-9.5'),
               dict(source_id='J0101+0101', band='Full', cand_id='Vn_07h_2025-10-20_Full_0009', det_date='2025-10-20',
                    det_ra='15.31', det_dec='1.01', det_snr='-6.1')])
    write_csv(f'{U}/master_catalog.csv',
              [dict(cand_id='Vn_04h_2025-10-14_Full_0001', fp_prediction='not-FP', stokes='V', lst='04h', band='Full',
                    date='2025-10-14', ra='15.3', dec='1.02', size_beams='1.1', size_flag='', axis_ratio='1.2', conform_note=''),
               dict(cand_id='Vn_05h_2025-10-15_Full_0002', fp_prediction='marginal', stokes='V', lst='05h', band='Full',
                    date='2025-10-15', ra='30.5', dec='2.03', size_beams='', size_flag='', axis_ratio='',
                    conform_note='table-exempt: pulsar parent|other'),
               dict(cand_id='In_06h_2025-10-16_Red_0003', fp_prediction='', stokes='I', lst='06h', band='Red', date='2025-10-16',
                    ra='45.7', dec='3.04', size_beams='', size_flag='', axis_ratio='', conform_note='')])
    union = [dict(r, stokes=K[0]) for K in ('V_wb', 'I_wb') for r in surv[K]]
    write_csv(f'{U}/source_catalogs/Full_sources_survivors.csv', [r for r in union if r['band'] == 'Full'])
    write_csv(f'{U}/source_catalogs/Red_sources_survivors.csv', [r for r in union if r['band'] == 'Red'])
    write_csv(f'{U}/source_catalogs/localizations.csv', [dict(identity_id='J0101+0101', ra='15.3', dec='1.02', err_arcsec='20.0', n_det='2')])
    write_csv(f'{U}/source_catalogs/localization_offsets.csv',
              [dict(cand_id='Vn_04h_2025-10-14_Full_0001', identity_id='J0101+0101', sep_arcsec='3.0', sep_beams='0.1', offset_pct='20')])
    write_csv(f'{U}/frame_times.csv', [dict(stokes='V', lst='04h', band='Full', date='2025-10-14', date_obs='2025-10-14T04:00:00',
                                            date_end='2025-10-14T05:00:00')])
    os.makedirs(f'{U}/localization_figs', exist_ok=True)
    from PIL import Image
    Image.new('RGBA', (600, 400)).save(f'{U}/localization_figs/J0101+0101.png')


def cli(out, *extra, env=None):
    stub = os.path.join(T, 'code', 'exopipe_ops')
    cmd = [PY, '-m', 'exopipe.phase3', 'pages', '--tier', 'deep', '--run', 'T1', '--out', out, '--cores', 'none',
           '--legacy-root', f'ops={stub}', '--deep-parent-dir', os.path.join(T, 'parents'), '--shards', '2',
           '--set', 'pages.MIRROR_CHECK=False'] + list(extra)
    return sh(cmd, env=env)


def rec(out, unit):
    p = os.path.join(out, '_records', 'pages', f'{unit}.json')
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------------------------------------------------------------------------------------------------------------

def test_heredoc_port_byte_identical():
    """lib/pages/pages_input.singles == the production heredoc of p3s3_pass.sh:137-169 on the same run, byte for byte."""
    from exopipe.phase3.lib.pages import pages_input as PI
    src = open(os.path.join(OPS, 'p3s3_pass.sh')).read()
    body = src.split("<<'PYP'", 1)[1].split('\nPYP\n', 1)[0].split('\n', 1)[1]
    body = re.sub(r'import os; U=f"/data07/gh/phase3/transients/Run_\{os\.environ\.get\(\'P3S_LABEL\',\'P3S3\'\)\}"',
                  'import os; U=os.environ["TEST_U"]', body)
    assert 'TEST_U' in body, 'heredoc U line not found - the production heredoc changed: re-port'
    a, b = os.path.join(T, 'hd_a', 'Run_T1'), os.path.join(T, 'hd_b', 'Run_T1')
    make_run(a)
    shutil.copytree(a, b)
    hd = os.path.join(T, 'heredoc.py')
    open(hd, 'w').write(body)
    env = dict(os.environ, TEST_U=a, P3S_LABEL='T1')
    r = subprocess.run([PY, hd], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert r.returncode == 0, r.stdout
    PI.singles(b, 'T1')
    for K in PI.SINGLES_SETS:
        fa, fb = f'{a}/source_catalogs/pages_input_{K}.csv', f'{b}/source_catalogs/pages_input_{K}.csv'
        assert open(fa, 'rb').read() == open(fb, 'rb').read(), f'{K}: port differs from the heredoc'
    print('PASS heredoc port byte-identical (4 sets)')


def test_deep_chain_cli():
    from exopipe.phase3.core import validate
    out = os.path.join(T, 'out')
    U = os.path.join(out, 'Run_T1')
    make_run(U)
    os.makedirs(os.path.join(T, 'parents'), exist_ok=True)
    env = dict(os.environ, P3S3_TIERS='V_wb', P3S_MERGE_STOKES_KEY='1', P3S_LABEL='WRONG')   # strays the step must scrub
    rc, o = cli(out, '--stages', 'tess,input,build,merge', '--sets', 'V_wb,I_wb,V_fwb', env=env)
    assert rc == 0, o
    for unit in ('T1:tess', 'T1:input:V_wb', 'T1:input:I_wb', 'T1:input:V_fwb', 'T1:build:V_wb', 'T1:build:I_wb',
                 'T1:build:V_fwb', 'T1:merge'):
        r = rec(out, unit)
        assert r and r['status'] == 'done', (unit, r and (r['status'], r['reason']))
    b = rec(out, 'T1:build:V_wb')
    assert b['meta']['rendered'] == 3 and b['meta']['kept_unchanged'] == 0, b['meta']
    assert len(b['planned']) == 3 + 1 + 1, b['planned']          # 3 panels (one member epoch), 1 index, 1 queue
    envs = [json.load(open(p)) for p in glob.glob(f'{U}/_stub_env_pages_input_V_wb.csv_*.json')]
    assert envs and all(e.get('P3S_LABEL') == 'T1' and 'P3S3_TIERS' not in e and 'P3S_MERGE_STOKES_KEY' not in e
                        and e.get('P3_SORT_SNR') == '1' and e.get('P3_CADENCE') == 'deep' and e.get('P3S3_PASS_T0')
                        and e.get('P3_PARENT_DIR') for e in envs), envs
    assert os.path.exists(f'{U}/judgements/queue_p3s3_V_wb.txt') and not os.path.exists(f'{U}/judgements/queue_i14.txt')
    assert os.path.exists(f'{U}/judgements/queue_p3s3_V_fwb_raw.txt')
    assert os.path.exists(f'{U}/source_catalogs/Full_pages_index_V_fwb.csv')
    fork = list(csv.DictReader(open(f'{U}/source_catalogs/pages_input_V_fwb.csv')))
    assert fork and fork[0]['label'] == 'multiples fork' and fork[0]['run'] == 'T1' and fork[0]['identity_tiers'] == 'V_fwb'
    merged = {r['source_id']: r['review_dir'] for r in csv.DictReader(open(f'{U}/source_catalogs/Full_sources_survivors.csv'))}
    assert merged['J0101+0101'].startswith('candidates/I/J0101+0101_04h_2025-10-14_Full'), merged
    m = rec(out, 'T1:merge')
    assert all(t['passed'] for t in m['tests']), [t for t in m['tests'] if not t['passed']]
    # every recorded test of every unit passed (content tests ran on every output)
    for p in glob.glob(f'{out}/_records/pages/*.json'):
        r = json.load(open(p))
        assert not [t for t in r.get('tests', []) if not t['passed'] and not t.get('waived')], (p, r['tests'])
    # M14 in the shared layout (production behaviour): the V_fwb page of J0101+0101 has the same page id and rank as its
    # V_wb page, so the V_fwb build overwrote the V_wb panel file; the next V_wb build renders that one page again
    rc, o = cli(out, '--stages', 'input,build', '--sets', 'V_wb')
    assert rc == 0, o
    assert rec(out, 'T1:build:V_wb')['meta']['rendered'] == 1, rec(out, 'T1:build:V_wb')['meta']
    # incremental: nothing changed -> 0 rendered, 3 kept
    rc, o = cli(out, '--stages', 'input,build', '--sets', 'V_wb')
    assert rc == 0, o
    b2 = rec(out, 'T1:build:V_wb')
    assert b2['meta']['rendered'] == 0 and b2['meta']['kept_unchanged'] == 3, b2['meta']
    # one row changes -> exactly that page re-renders
    p = f'{U}/source_catalogs_V_wb/Full_sources_survivors.csv'
    rows = list(csv.DictReader(open(p)))
    rows[1]['peak_jy_best'] = '0.31'
    write_csv(p, rows)
    rc, o = cli(out, '--stages', 'input,build', '--sets', 'V_wb')
    assert rc == 0, o
    b3 = rec(out, 'T1:build:V_wb')
    assert b3['meta']['rendered'] == 1 and b3['meta']['kept_unchanged'] == 2, b3['meta']
    # --resume with nothing changed: the units are kept
    rc, o = cli(out, '--stages', 'input,build', '--sets', 'V_wb', '--resume')
    assert rc == 0 and 'skipped-valid 2' in o, o
    # --full: every page renders
    rc, o = cli(out, '--stages', 'build', '--sets', 'V_wb', '--full')
    assert rc == 0, o
    assert rec(out, 'T1:build:V_wb')['meta']['rendered'] == 3
    # --epochs (per hour): accepted, the epoch's panels counted
    rc, o = cli(out, '--stages', 'build', '--sets', 'V_wb', '--epochs', '07h_2025-10-20', '--fresh')
    assert rc == 0, o
    assert rec(out, 'T1:build:V_wb')['meta']['epoch_panels'] == 1
    print('PASS deep chain through the CLI (tess, input, build, merge; incremental 0 / 1 / full / resume / epochs; env scrub)')
    return out


def test_dry_run_and_refusals(out):
    U = os.path.join(out, 'Run_T1')
    before = sorted(os.listdir(os.path.join(out, '_records', 'pages')))
    rc, o = cli(out, '--dry-run', '--sets', 'V_wb')
    assert rc == 0 and 'DRY RUN' in o and 'planned panels' in o, o
    assert sorted(os.listdir(os.path.join(out, '_records', 'pages'))) == before
    # P1: no --run
    rc, o = sh([PY, '-m', 'exopipe.phase3', 'pages', '--tier', 'deep', '--out', out, '--cores', 'none'])
    assert rc == 2 and 'P1' in o, o
    # protected (mirror) key
    rc, o = cli(out, '--stages', 'build', '--set', 'pages.PNG_DPI=100')
    assert rc == 2 and 'PROTECTED' in o, o
    # P4: localization tables missing
    lo = f'{U}/source_catalogs/localizations.csv'
    os.rename(lo, lo + '.x')
    rc, o = cli(out, '--stages', 'build', '--sets', 'V_wb')
    os.rename(lo + '.x', lo)
    assert rc == 2 and 'P4' in json.dumps(rec(out, 'T1:build:V_wb')), o
    # P2: build of a set whose input record does not exist
    out2 = os.path.join(T, 'out2')
    make_run(os.path.join(out2, 'Run_T1'))
    rc, o = cli(out2, '--stages', 'build', '--sets', 'V_wb')
    assert rc == 2 and 'P2' in rec(out2, 'T1:build:V_wb')['reason'], o
    print('PASS dry run writes nothing; P1, protected key, P4, P2 refuse (exit 2)')


def test_m14_per_set():
    out = os.path.join(T, 'out_m14')
    U = os.path.join(out, 'Run_T1')
    make_run(U)
    rc, o = cli(out, '--stages', 'input,build,merge', '--sets', 'V_wb,I_wb,V_fwb', '--set', 'pages.PANEL_DIR_LAYOUT=per-set',
                '--set', 'pages.DEEP_MERGE=keyed')
    assert rc == 0, o
    assert glob.glob(f'{U}/candidates/V_wb/J0101+0101_*/*.png') and glob.glob(f'{U}/candidates/V_fwb/J0101+0101_*/*.png')
    assert not glob.glob(f'{U}/candidates/I/*'), 'per-set layout wrote into candidates/I'
    idx = list(csv.DictReader(open(f'{U}/source_catalogs/Full_pages_index_V.csv')))
    assert all(r['review_dir'].startswith('candidates/V_wb/') for r in idx), idx
    merged = {(r['source_id'], r['stokes']): r['review_dir'] for r in csv.DictReader(open(f'{U}/source_catalogs/Full_sources_survivors.csv'))}
    assert merged[('J0101+0101', 'V')].startswith('candidates/V_wb/'), merged
    print('PASS M14 per-set layout (candidates/<K>/), index rewritten, keyed merge, fork index kept apart')


def test_fork_rows_equal_real_script():
    from exopipe.phase3.lib.pages import pages_input as PI
    U = os.path.join(T, 'out', 'Run_T1')
    got = list(csv.DictReader(open(f'{U}/source_catalogs/pages_input_V_fwb.csv')))
    exp, _ = PI.fork_rows(U, 'V_fwb', 'multiples fork', 'T1')
    assert [dict(r) for r in got] == [{k: str(v) for k, v in r.items()} for r in exp]
    print('PASS fork rows recomputed = real p3s3_tier_pages_input.py output')


def test_mirror_drift():
    from exopipe.phase3.lib.pages import mirror, common
    d = os.path.join(T, 'drift', 'exopipe_ops')
    os.makedirs(d)
    src = open(os.path.join(OPS, 'build_i14_pages_std7.py')).read()
    assert 'dpi=110' in src
    open(os.path.join(d, 'build_i14_pages_std7.py'), 'w').write(src.replace('fig.savefig(_png, dpi=110)', 'fig.savefig(_png, dpi=100)'))
    res = mirror.check({'ops': d, 'clv15': '/lustre/gh/main/clv15'}, ['build_i14_pages_std7.py'])
    bad = [r for r in res if not r['ok']]
    assert any(r['key'] == 'pages.PNG_DPI' for r in bad), bad
    live = mirror.check(common.legacy_roots())
    assert all(r['ok'] for r in live), [r for r in live if not r['ok']]
    print(f'PASS mirror drift detected; {len(live)} live anchors agree with config')


def test_negative_content_tests():
    from exopipe.phase3.core import validate
    from PIL import Image
    d = os.path.join(T, 'neg', 'Run_X', 'candidates', 'I', 'J0000+0000_04h_2025-10-14_Full')
    os.makedirs(d)
    p = os.path.join(d, 'rank0001_04h_2025-10-14_Full.png')
    Image.new('RGBA', (100, 100)).save(p)
    res = {r['name']: r for r in validate.run('review_page', p, tier='deep')}
    assert not res['review_page.decodes']['passed'] and not res['review_page.sidecar']['passed']
    q = os.path.join(T, 'neg', 'q.txt')
    open(q, 'w').write('J1\nJ2\n')
    assert not all(r['passed'] for r in validate.run('review_queue', q))
    s = os.path.join(T, 'neg', 'Run_X', 'source_catalogs', 'Full_sources_survivors.csv')
    write_csv(s, [dict(source_id='J0000+0000', band='Full', best_date='2025-10-14', ra='1', review_dir='')])
    from exopipe.phase3.lib.pages import common
    pre = common.digest_without(s, 'review_dir')
    write_csv(s, [dict(source_id='J0000+0000', band='Full', best_date='2025-10-14', ra='2',
                       review_dir='candidates/I/J0000+0000_04h_2025-10-14_Full')])
    r = validate.run('survivors_review_dir', s, pre_digest=pre)
    assert not r[0]['passed'] and 'other than review_dir' in r[0]['detail'], r
    print('PASS negative content tests (wrong panel size, missing sidecar, queue without header, survivors column changed)')


def test_real_products_read_only():
    """The content tests on existing production products (read only). Results printed; assert only that no test raised."""
    from exopipe.phase3.core import validate
    P5 = '/data07/gh/phase3/transients/Run_P3S5'
    items = []
    if os.path.isdir(P5):
        d = f'{P5}/candidates/I/J2215+5136_00h_2025-09-06_Full'
        items += [('review_page', f'{d}/rank0005_00h_2025-09-06_Full.png'), ('review_page', f'{d}/rank0007_00h_2025-09-06_Full.png')]
        items += [('review_page', p) for p in sorted(glob.glob(f'{P5}/candidates/I/J0832+2918_08h_2025-01-10_Full/rank*.png'))[:1]]
        items += [('page_index', f'{P5}/source_catalogs/Full_pages_index_V.csv'),
                  ('page_index', f'{P5}/source_catalogs/Full_pages_index_V_fwb.csv'),
                  ('review_queue', f'{P5}/judgements/queue_p3s3_V_fwb_raw.txt'),
                  ('review_queue', f'{P5}/judgements/queue_p3s3_V_wb.txt'),
                  ('pages_input', f'{P5}/source_catalogs/pages_input_V_fwb.csv'),
                  ('pages_input', f'{P5}/source_catalogs/pages_input_V_wb.csv'),
                  ('survivors_review_dir', f'{P5}/source_catalogs/Full_sources_survivors.csv'),
                  ('tess_coverage', f'{P5}/tess_coverage.csv'),
                  ('strat_queue', f'{P5}/judgements/queue_strat_V.txt')]
        items += [('astrometry_figure', p) for p in sorted(glob.glob(f'{P5}/localization_figs/*.png'))[:1]]
    S14 = '/data07/gh/phase3/transients/Run_SLST14B'
    if os.path.isdir(S14):
        items += [('sublst_page_run', S14),
                  ('review_queue', f'{S14}/judgements/queue_p3s3_V_sb.txt')]
        items += [('review_page', p) for p in sorted(glob.glob(f'{S14}/candidates/I/*/rank*.png'))[:1]]
        items += [('page_index', p) for p in sorted(glob.glob(f'{S14}/source_catalogs/*_pages_index_V.csv'))[:1]]
    SN = '/data07/gh/phase3/snapshot_search/review/Run_SNAPB1R_20261004'
    if os.path.isdir(SN):
        items += [('snapshot_page_run', SN), ('filter_catalog', f'{SN}/master_catalog.csv'), ('review_queue', f'{SN}/queue_V_10s.txt')]
        items += [('review_page', p) for p in sorted(glob.glob(f'{SN}/candidates/V/*/rank*.png'))[:1]]
    raised = []
    for cls, p in items:
        if not os.path.exists(p):
            print(f'  SKIP {cls} {p} (absent)')
            continue
        t = time.time()
        for r in validate.run(cls, p):
            print(f"  {'PASS' if r['passed'] else 'FAIL'} {r['name']:36s} {str(r['value'])[:14]:>14} "
                  f"{os.path.relpath(p, '/data07/gh/phase3')}  {str(r['detail'])[:150]}  ({time.time() - t:.1f}s)")
            if 'test raised' in str(r['detail']):
                raised.append((cls, p, r['detail']))
    assert not raised, raised
    print(f'PASS content tests ran on {len(items)} production products (results above)')


def main():
    t0 = time.time()
    if os.path.exists(T):
        shutil.rmtree(T)
    os.makedirs(T)
    try:
        make_stub_root()
        test_heredoc_port_byte_identical()
        out = test_deep_chain_cli()
        test_fork_rows_equal_real_script()
        test_dry_run_and_refusals(out)
        test_m14_per_set()
        test_mirror_drift()
        test_negative_content_tests()
        test_real_products_read_only()
    finally:
        shutil.rmtree(T, ignore_errors=True) if not os.environ.get("KEEP_T") else None
    print(f'ALL PASS in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
