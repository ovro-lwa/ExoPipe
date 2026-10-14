#!/usr/bin/env python3
"""STANDING TEST - `exopipe3 followup` (steps/followup.py, config/steps/followup.py, followup/_cfg_adapter.py).

  cli          --help; design-pending ops refuse before any read (G5, nothing written); a flag the op does not use (exit 1)
  config       the seven sections load; PROTECTED keys refuse (G6); RULED ledger IDs exist; pinned input into a production
               --out (G6); mirrored literals present in the real in-place sources and G6b on a source without them;
               G7 production root (patched root in /tmp); G8 TESS interpreter missing
  ep_plan      fetch through the REAL ep_wxt_plan_fetch.py with a synthetic network (a stored raw served from disk):
               pass / refused (synthetic 7 h pointing: refused_<tag> kept) / download failure (error); audit; replay of real
               stored raws (read only) equal to the archive's plans (P-F1) and of a bad one (refused)
  tess         fake TESS script: done / no sector (exit 3) / SIMBAD failure (G9)
  investigate  the package in-process: G11 on an empty synthetic bundle; G13 when the package config was imported with
               other paths; dossier writers of the package -> the followup_* content tests (column constants checked)
  summary      synthetic top-50 + one evidence file: ranked summary, missing dossier row, re-score written back
  hone         the REAL phase3_honing.py on a 2-candidate CSV: every candidate no-products (exit 3), as production today
  periodicity  fake stage scripts: lightcurves -> search (G28 refuses 1 of 24 cells; then all 24) -> page; a second
               invocation reads the first one's records through --lc-root; an unrecorded light curve refuses (G28)
  fold         fake fold script: outputs and content tests; source-page dry-run
  negative     each content test fails on a corrupted synthetic copy
  real         READ-ONLY: content tests on production products; --dry-run on real items

Scratch: /tmp/exopipe3_tests_followup/ (removed at the end). Production trees are only read. Every step call runs
in-process through cli.main with --cores none and a --out under the scratch directory.
Run standalone:  python tests/test_followup.py [test_name ...]
"""
import contextlib
import csv
import glob
import gzip
import io
import json
import os
import shutil
import struct
import sys
import time
import zlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
SCRATCH = '/tmp/exopipe3_tests_followup'
ARCH = '/lustre/gh/main/catalogs/einstein_probe/wxt_plan'
RAW_OK = f'{ARCH}/2026-10/raw_20261012T124103Z.json.gz'
RAW_REFUSED = f'{ARCH}/2026-10/refused_20261013T034102Z.json.gz'
BW = '/lustre/gh/main/exopipe_ops/steep_catalog/battery_work'


def _fresh(name):
    d = os.path.join(SCRATCH, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _step():
    from exopipe.phase3.core import env
    env.legacy()
    import exopipe.phase3.steps.followup as F
    return F


@contextlib.contextmanager
def _patched(**kw):
    """Patch config section attributes for one block: _patched(followup__SCRIPTS={...}) (section__KEY=value)."""
    CFG, F = _cfg(), _step()
    old = []
    for k, v in kw.items():
        sec, key = k.split('__', 1)
        s = getattr(CFG, sec)
        old.append((s, key, getattr(s, key)))
        setattr(s, key, v)
    F._EP_MOD.clear()
    try:
        yield
    finally:
        for s, key, v in reversed(old):
            setattr(s, key, v)
        F._EP_MOD.clear()


def _cli(argv):
    """exopipe3 followup <argv> in-process: (rc, stdout, stderr). Overrides are undone afterwards."""
    from exopipe.phase3 import cli
    o, e = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
            try:
                rc = cli.main(['followup'] + list(argv) + ['--cores', 'none'])
            except SystemExit as x:
                rc = x.code if isinstance(x.code, int) else 1
                e.write(str(x.code))
    finally:
        _cfg().reset()
    return rc, o.getvalue(), e.getvalue()


def _records(out, unit=None):
    d = os.path.join(out, '_records', 'followup')
    recs = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(d, '*.json'))}
    return recs[unit] if unit else recs


def _png(path, n=64):
    """A valid PNG of noise (> 1 kB) without matplotlib."""
    import random
    rnd = random.Random(1)
    raw = b''.join(b'\x00' + bytes(rnd.randrange(256) for _ in range(n)) for _ in range(n))

    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', n, n, 8, 0, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


PNG_FUNC = '''
def _png(path, n=64):
    import random, struct, zlib
    rnd = random.Random(1)
    raw = b''.join(b'\\x00' + bytes(rnd.randrange(256) for _ in range(n)) for _ in range(n))
    def chunk(t, d):
        return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    open(path, 'wb').write(b'\\x89PNG\\r\\n\\x1a\\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', n, n, 8, 0, 0, 0, 0))
                           + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))
'''

# ---------------------------------------------------------------------------------------------------------------------
# fake / wrapper scripts (written into the scratch; the step runs them through core.legacy.run as it runs production's)
# ---------------------------------------------------------------------------------------------------------------------

EP_WRAPPER = '''import gzip, importlib.util, os, sys
_spec = importlib.util.spec_from_file_location('ep_real', os.environ['FU_EP_REAL'])
m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m)
globals().update({k: getattr(m, k) for k in dir(m) if not k.startswith('__')})
class _R:
    def __init__(self, b): self.b = b
    def read(self): return self.b
def _urlopen(url, timeout=None):
    if os.environ.get('FU_EP_FAIL'): raise OSError('synthetic network failure')
    return _R(gzip.open(os.environ['FU_EP_RAW']).read())
m.urllib.request.urlopen = _urlopen
if __name__ == '__main__':
    root = sys.argv[1].rstrip('/'); os.makedirs(root, exist_ok=True)
    sys.exit(m.audit(root) if '--audit' in sys.argv else m.fetch(root))
'''

TESS_FAKE = PNG_FUNC + '''
import argparse, os, sys
ap = argparse.ArgumentParser()
for x in ('--name', '--radio-start', '--radio-end', '--radio-label', '--out', '--key', '--sector'):
    ap.add_argument(x)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
if a.name == 'Nobody':
    sys.exit(a.name + ': not resolved by SIMBAD - give the SIMBAD name (or a Gaia DR3 id)')
lines = ['STAR ' + a.name + ': synthetic', 'RADIO ' + a.radio_label + ': ' + a.radio_start + ' - ' + a.radio_end]
if a.name == 'Dark':
    lines.append('NO TESS sector observed the star during the radio epoch')
else:
    lines.append('SECTOR 91')
    with open(os.path.join(a.out, a.key + '_s91_SPOC120.csv'), 'w') as f:
        f.write('time,flux,flux_err\\n' + ''.join('%.5f,%.6f,0.0001\\n' % (3775 + i * 0.01, 1 + 0.001 * ((i % 7) - 3))
                                                  for i in range(200)))
    _png(os.path.join(a.out, a.key + '_s91.png'))
open(os.path.join(a.out, a.key + '_summary.txt'), 'w').write('\\n'.join(lines) + '\\n')
print('\\n'.join(lines))
'''

LC_FAKE = '''import csv, glob, os, sys
import numpy as np
POS, OUT, LST, BAND, S = sys.argv[1:6]
GEN, ARC = os.environ['P3_GEN'], os.environ['P3_ARCHIVE_ROOT']
src = list(csv.DictReader(open(POS)))
files = sorted(glob.glob(f'{ARC}/{LST}/*/{GEN}/{BAND}/{S}/10min/{BAND}_{S}_10min_Taper_Robust+0.0_pbcorr_dewarped_aligned_msub_LST{LST}_bin*.fits'))
if not files:
    sys.exit(0)
nf, npos = len(files), len(src)
a = np.ones((nf, npos), np.float32)
os.makedirs(OUT, exist_ok=True)
np.savez_compressed(f'{OUT}/lc3_{LST}_{BAND}_{S}.npz', source_id=np.array([r['source_id'] for r in src]),
    ra=np.array([float(r['ra']) for r in src]), dec=np.array([float(r['dec']) for r in src]),
    el_mid=np.full(npos, 45.0, np.float32), mjd=60000.0 + np.arange(nf), night=np.array([f.split('/')[-6] for f in files]),
    bin=np.arange(nf), path=np.array(files), peak=a, ctr=a, max3=a, rms=a, master_jy=np.ones(npos, np.float32),
    fixedpos=a, off_ra_arcsec=a, off_dec_arcsec=a, master_path=np.array(['']))
print(LST, BAND, S, 'DONE')
'''

SEARCH_FAKE = '''import csv, os, sys
import numpy as np
LCD, TIER, STOKES, OUT = sys.argv[1:5]
cols = {cols!r}
ids = {{}}
for h in range(24):
    p = f'{{LCD}}/lc3_{{h:02d}}h_{{TIER}}_{{STOKES}}.npz'
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        for s, r, d in zip(z['source_id'], z['ra'], z['dec']):
            ids[str(s)] = (float(r), float(d))
if not ids:
    sys.exit(f'no lc3 files for {{TIER}} {{STOKES}} in {{LCD}}')
os.makedirs(OUT, exist_ok=True)
with open(f'{{OUT}}/periodicity_{{TIER}}_{{STOKES}}.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    for s, (r, d) in sorted(ids.items()):
        w.writerow({{'source_id': s, 'ra': r, 'dec': d, 'n_blocks': 20, 'n_blocks_all': 20}})
print(TIER, STOKES, 'DONE', os.environ.get('ELEV_FLOOR'), os.environ.get('MIN_BLOCKS'))
'''

PAGE_FAKE = '''import csv, os, shutil, sys
PD, LCD, PAGE, LUS = sys.argv[1:5]
STK = os.environ.get('P3_STOKES', 'I')
keys = {keys!r}
os.makedirs(f'{{PAGE}}/img', exist_ok=True); os.makedirs(LUS, exist_ok=True)
T = {{}}
for tier in ('Red', 'Green', 'Blue'):
    p = f'{{PD}}/periodicity_{{tier}}_{{STK}}.csv'
    for r in csv.DictReader(open(p)):
        T.setdefault(r['source_id'], {{'source_id': r['source_id'], 'ra': r['ra'], 'dec': r['dec']}})[tier] = r
cols = ['source_id', 'ra', 'dec'] + [f'{{t}}_{{k}}' for t in ('Red', 'Green', 'Blue') for k in keys]
rows = []
for s, d in T.items():
    r = {{'source_id': s, 'ra': d['ra'], 'dec': d['dec']}}
    for t in ('Red', 'Green', 'Blue'):
        for k in keys:
            r[f'{{t}}_{{k}}'] = d.get(t, {{}}).get(k, '')
    rows.append(r)
with open(f'{{LUS}}/allsky_periodicity_{{STK}}.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
shutil.copy(f'{{LUS}}/allsky_periodicity_{{STK}}.csv', f'{{PAGE}}/allsky_periodicity.csv')
open(f'{{PAGE}}/index.html', 'w').write('<html><body><a href="allsky_periodicity.csv">table</a></body></html>')
'''

FOLD_FAKE = PNG_FUNC + '''
import csv, os, sys
CSV, OUT, PRE, ra, dec, PCAT, BANDS = sys.argv[1:8]
BANDS = BANDS.split(',')
R = [r for r in csv.DictReader(open(CSV)) if r['band'] in BANDS and r.get('cadence', '10min') == '10min' and r['flux_jy'] != '']
with open(f'{OUT}/{PRE}_folded.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['band', 'lst', 'night', 'mjd_mid_utc', 'mjd_bary_tdb', 'phase', 'flux_jy', 'err_jy', 'normalised_flux',
                'kept_after_mad'])
    for i, r in enumerate(R):
        ph = ((float(r['mjd_mid']) / float(PCAT)) % 1) - 0.5
        w.writerow([r['band'], r.get('lst', ''), r['night'], r['mjd_mid'], r['mjd_mid'], '%+.4f' % ph, r['flux_jy'],
                    r['err_jy'], '1.000', 1])
for s in ('folded', 'smoothed', 'period_search'):
    _png(f'{OUT}/{PRE}_{s}.png')
'''


def _write(path, text):
    with open(path, 'w') as f:
        f.write(text)
    return path


# =====================================================================================================================
# tests
# =====================================================================================================================

def test_cli():
    import subprocess
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=REPO)
    h = subprocess.run([py, '-m', 'exopipe.phase3', 'followup', '--help'], capture_output=True, text=True, env=env,
                       cwd='/tmp', timeout=120)
    assert h.returncode == 0 and 'design pending' in h.stdout and 'Failure modes' in h.stdout, h.stderr[-500:]
    out = os.path.join(SCRATCH, 'pending')
    for op, sec in (('dynspec', '9.3'), ('lpt-period', '9.6'), ('dedisperse', '9.7'), ('reprocess', '9.1')):
        rc, o, e = _cli([op, '--out', out])
        assert rc == 2 and 'G5' in e and f'architecture {sec}' in e, (op, rc, e)
    assert not os.path.exists(out), 'a design-pending op wrote something'
    rc, o, e = _cli(['tess', '--cells', '04h_Red', '--out', out, '--dry-run'])
    assert rc == 1 and 'does not use --cells' in e, (rc, e)
    rc, o, e = _cli(['fold', '--dry-run'])
    assert rc == 1 and 'no production root' in e, (rc, e)
    rc, o, e = _cli(['periodicity', '--stages', 'search,lightcurves', '--out', out, '--dry-run'])
    assert rc == 1 and 'pipeline order' in e, (rc, e)
    assert not os.path.exists(out)


def test_config():
    from exopipe.phase3.core import ledger
    CFG, F = _cfg(), _step()
    for s in ('followup', 'followup_ep_plan', 'followup_tess', 'followup_investigate', 'followup_hone',
              'followup_periodicity', 'followup_fold'):
        sec = getattr(CFG, s)
        for k in tuple(getattr(sec, 'PROTECTED_KEYS', ())) + tuple(getattr(sec, 'RULED', {})) + tuple(
                getattr(sec, 'pinned_inputs', ())):
            assert hasattr(sec, k), (s, k)
        for k, lid in getattr(sec, 'RULED', {}).items():
            assert ledger.has_id(lid, CFG.paths.FIX_LEDGER), (s, k, lid)
    assert CFG.followup_ep_plan.POINTING_MAX_H == 6.0
    out = os.path.join(SCRATCH, 'config')
    rc, o, e = _cli(['ep-plan', '--set', 'followup_ep_plan.POINTING_MAX_H=3', '--out', out, '--dry-run'])
    assert rc == 2 and 'PROTECTED' in e, (rc, e)
    rc, o, e = _cli(['periodicity', '--stages', 'search', '--set', 'followup_periodicity.SHORT_PERIOD_D=(0.05,0.03)',
                     '--out', out, '--dry-run'])
    assert rc == 2 and 'PROTECTED' in e, (rc, e)
    # a RULED key into a scratch --out is allowed (recorded as non-production out)
    rc, o, e = _cli(['periodicity', '--stages', 'search', '--set', 'followup_periodicity.MIN_BLOCKS=50', '--out', out,
                     '--dry-run'])
    assert rc == 0 and 'MIN_BLOCKS=50' in o, (rc, o, e)
    # the real in-place sources carry every mirrored literal
    for key in CFG.followup.MIRRORS:
        assert F._mirror_problems(CFG, key) == [], F._mirror_problems(CFG, key)
    # G6b on a source without the literal; G7 on a production root; G6 pinned input into a production --out
    fake = _write(os.path.join(_fresh('config'), 'fold_no_literals.py'), 'print(1)\n')
    with _patched(followup__SCRIPTS=dict(CFG.followup.SCRIPTS, fold=fake)):
        rc, o, e = _cli(['fold', '--lightcurve', fake, '--prefix', 'x', '--period-d', '0.1', '--fold-bands', 'Red',
                         '--ra', '1', '--dec', '2', '--out', out])
        assert rc == 2 and 'G6b' in e, (rc, e)
    prod = os.path.join(out, 'prod')
    with _patched(followup__PRODUCTION_ROOTS=CFG.followup.PRODUCTION_ROOTS + (prod,)):
        rc, o, e = _cli(['ep-plan', '--action', 'audit', '--out', prod])
        assert rc == 2 and 'G7' in e, (rc, e)
        assert not os.path.exists(prod), 'G7 refusal wrote under the production root'
    # G6: a pinned input overridden for a production --out. A child process (the package config is imported once per
    # process, gate G13), a dry-run (nothing written) against the real production root of the dossiers
    import subprocess
    top = os.path.join(SCRATCH, 'config', 'top50_copy.csv')
    shutil.copyfile(CFG.followup_investigate.TOP50_CSV, top)
    p = subprocess.run([sys.executable, '-m', 'exopipe.phase3', 'followup', 'summary', '--set',
                        f'followup_investigate.TOP50_CSV={top}', '--out', CFG.followup_investigate.OUT_ROOT,
                        '--dry-run', '--cores', 'none'], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=REPO), cwd=SCRATCH, timeout=120)
    assert p.returncode == 2 and 'G6' in p.stdout and 'pinned input' in p.stdout, (p.returncode, p.stdout, p.stderr)
    with _patched(site__PYTHON=dict(CFG.site.PYTHON, tess='/nonexistent/python')):
        rc, o, e = _cli(['tess', '--name', 'X', '--radio-start', '2025-01-01T00:00:00', '--radio-end',
                         '2025-01-01T01:00:00', '--radio-label', 'L', '--out', out])
        assert rc == 2 and 'G8' in e, (rc, e)


def test_ep_plan():
    CFG, F = _cfg(), _step()
    d = _fresh('ep')
    wrapper = _write(os.path.join(d, 'ep_wrapper.py'), EP_WRAPPER)
    os.environ['FU_EP_REAL'] = F._script('ep_plan')
    os.environ['FU_EP_RAW'] = RAW_OK
    root = os.path.join(d, 'archive')
    try:
        with _patched(followup__SCRIPTS=dict(CFG.followup.SCRIPTS, ep_plan=wrapper), followup__MIRRORS={}):
            rc, o, e = _cli(['ep-plan', '--out', root])                                      # pass
            assert rc == 0, (rc, e)
            r = [v for k, v in _records(root).items() if k.startswith('fetch_')][0]
            assert r['status'] == 'done' and {x['class'] for x in r['outputs']} == {
                'ep_wxt_plan', 'ep_wxt_raw', 'ep_wxt_fetch_log'}, r['outputs']
            assert all(t['passed'] for t in r['tests']), r['tests']
            time.sleep(1.2)
            bad = json.loads(gzip.open(RAW_OK).read())                                       # refused: a 7 h pointing
            from datetime import datetime, timedelta
            t0 = datetime.fromisoformat(bad[3]['start_date_UTC'])
            bad[3]['end_date_UTC'] = (t0 + timedelta(hours=7)).isoformat()
            badraw = os.path.join(d, 'raw_20300101T000000Z.json.gz')
            with gzip.open(badraw, 'wb') as g:
                g.write(json.dumps(bad).encode())
            os.environ['FU_EP_RAW'] = badraw
            for f in glob.glob(os.path.join(root, '_records', 'followup', 'fetch_*.json')):
                os.remove(f)
            rc, o, e = _cli(['ep-plan', '--out', root])
            assert rc == 2 and 'G1/G2' in e, (rc, e)
            r = [v for k, v in _records(root).items() if k.startswith('fetch_')][0]
            assert r['status'] == 'refused' and 'duration 7:00:00' in r['reason'], r['reason']
            assert 'ep_wxt_refused' in {x['class'] for x in r['outputs']}
            assert all(t['passed'] for t in r['tests']), [t for t in r['tests'] if not t['passed']]
            time.sleep(1.2)
            os.environ['FU_EP_FAIL'] = '1'                                                   # download failure: error
            rc, o, e = _cli(['ep-plan', '--out', root])
            assert rc == 1, (rc, e)
            del os.environ['FU_EP_FAIL']
            rows = list(csv.DictReader(open(os.path.join(root, 'fetch_log.csv'))))
            assert [x['status'] for x in rows] == ['pass', 'FAIL', 'FAIL'] and rows[2]['detail'].startswith('download')
            rc, o, e = _cli(['ep-plan', '--action', 'audit', '--out', root])                # audit
            assert rc == 0, (rc, e)
            assert any(k.startswith('audit_') and v['status'] == 'done' for k, v in _records(root).items())
        # replay: real stored downloads (read only) + the synthetic bad one; never into the archive root
        rp = os.path.join(d, 'replay_out')
        rc, o, e = _cli(['ep-plan', '--action', 'replay', '--raw', f'{RAW_OK},{RAW_REFUSED},{badraw}', '--out', rp])
        assert rc == 2, (rc, e)                                                              # the bad one refuses
        recs = _records(rp)
        ok = recs['replay_raw_20261012T124103Z']
        assert ok['status'] == 'done' and any(t['name'] == 'ep_plan.replay_equals_archive' and t['passed']
                                              for t in ok['tests']), ok['tests']
        assert recs['replay_refused_20261013T034102Z']['status'] == 'done'   # passes under today's 6 h bound
        assert recs['replay_raw_20300101T000000Z']['status'] == 'refused'
        rc, o, e = _cli(['ep-plan', '--action', 'replay', '--raw', RAW_OK, '--out', os.path.join(ARCH, 'replay')])
        assert rc == 2 and ('G7' in e or 'archive root' in e), (rc, e)
        assert not os.path.exists(os.path.join(ARCH, 'replay'))
    finally:
        for k in ('FU_EP_REAL', 'FU_EP_RAW', 'FU_EP_FAIL'):
            os.environ.pop(k, None)


def test_tess():
    CFG = _cfg()
    d = _fresh('tess')
    fake = _write(os.path.join(d, 'tess_fake.py'), TESS_FAKE)
    base = ['--radio-start', '2025-04-12T11:29:23.8', '--radio-end', '2025-04-12T12:27:34.5', '--radio-label', 'J1704',
            '--out', d]
    with _patched(followup__SCRIPTS=dict(CFG.followup.SCRIPTS, tess=fake)):
        rc, o, e = _cli(['tess', '--name', 'HD 154088'] + base)
        assert rc == 0, (rc, e)
        r = _records(d, 'tess_HD154088')
        assert {x['class'] for x in r['outputs']} == {'tess_summary', 'tess_lightcurve', 'followup_figure'}, r['outputs']
        assert all(t['passed'] for t in r['tests'])
        rc, o, e = _cli(['tess', '--name', 'Dark'] + base)
        assert rc == 3 and _records(d, 'tess_Dark')['status'] == 'no-input', (rc, e)
        rc, o, e = _cli(['tess', '--name', 'Nobody'] + base)
        assert rc == 2 and 'G9' in _records(d, 'tess_Nobody')['reason'], (rc, e)


def test_investigate():
    CFG, F = _cfg(), _step()
    d = _fresh('investigate')
    bundle = os.path.join(d, 'bundle')
    os.makedirs(bundle)
    out, work = os.path.join(d, 'out'), os.path.join(d, 'work')
    args = ['investigate', '--ra', '94.89483019', '--dec', '41.09169194', '--lst', '07h', '--date', '2024-12-31',
            '--band', '23MHz', '--set', f'followup_investigate.BUNDLE_DIR={bundle}', '--work', work, '--out', out]
    rc, o, e = _cli(args)
    assert rc == 2, (rc, e)
    r = _records(out, 'rank00_J061934+410530_LST07h')
    assert r['status'] == 'refused' and r['reason'].startswith('G11'), r['reason']
    assert r['meta']['candinv_env']['CANDINV_BUNDLE_DIR'] == bundle
    assert any('cross-match empty' in n for n in r['notes']), r['notes']
    from exopipe.phase3.core import exclusions
    assert r['admitted'] == ([] if exclusions.is_excluded('07h', '23MHz', '2024-12-31') else ['2024-12-31']), r
    assert r['meta']['exclusions']['mode'] == 'report'
    assert not glob.glob(os.path.join(work, '*')), 'scratch not removed'
    # G13: the package config is now imported with this bundle; another bundle cannot apply in this process
    other = os.path.join(d, 'bundle2')
    os.makedirs(other)
    rc, o, e = _cli(args[:-6] + ['--set', f'followup_investigate.BUNDLE_DIR={other}', '--work', work, '--out', out])
    assert rc == 2 and 'G13' in e, (rc, e)
    # the package writers -> the followup_* content tests (checks the column constants of steps/followup.py)
    from exopipe.phase3.followup import candidate as CM, io_products as IO, score as SC
    from exopipe.phase3.core import validate
    c = CM.Candidate(rank=7, ra=94.89483019, dec=41.09169194, band='23MHz', date='2024-12-31', cadence='deep', lst='07h')
    cdir = os.path.join(d, 'dossier', c.cand_id)
    os.makedirs(cdir)
    spec = [{'scope': '23MHz', 'freq_mhz': 23.0, 'v_peak': 1.0, 'v_snr': 7.0}, {'scope': '27MHz', 'freq_mhz': 27.0}]
    ev = {'candidate': {'cand_id': c.cand_id, 'rank': 7}, 'spectrum': spec, 'context': {'ateam_sep_deg': 30.0}}
    ev['p_astro'] = SC.p_astrophysical(ev)
    paths = {'evidence': IO.write_json(cdir, c, ev, inputs=[], params={}),
             'spectrum': IO.write_spectrum_csv(cdir, c, spec),
             'snap': IO.write_snapshot_csv(cdir, c, {'frames': [{'time': 't0', 'mjd': 60000.1, 'peak': 1, 'snr': 5,
                                                                 'sign': 1},
                                                                {'time': 't1', 'mjd': 60000.2, 'peak': 2, 'snr': 6,
                                                                 'sign': 1}]}),
             'snapspec': IO.write_snapshot_spectrum_csv(cdir, c, [{'scope': '23MHz', 'V_snr': 5.0}]),
             'reg': IO.write_reg(cdir, c, [(c.ra, c.dec, 'yellow', 'disc'), (c.ra + 0.01, c.dec, 'green', 'refined')])}
    for key, cls in (('evidence', 'followup_dossier'), ('spectrum', 'followup_table'), ('snap', 'followup_table'),
                     ('snapspec', 'followup_table'), ('reg', 'followup_region')):
        res = validate.run(cls, paths[key])
        assert all(t['passed'] for t in res), (key, res)
    # a conformant cutout through the package writer (core.products standard header) -> followup_fits
    import numpy as np
    from astropy.io import fits
    h = fits.Header()
    for k, v in (('NAXIS', 2), ('NAXIS1', 400), ('NAXIS2', 400), ('CTYPE1', 'RA---SIN'), ('CTYPE2', 'DEC--SIN'),
                 ('CRVAL1', c.ra), ('CRVAL2', c.dec), ('CRPIX1', 201), ('CRPIX2', 201), ('CDELT1', -0.03),
                 ('CDELT2', 0.03), ('BMAJ', 0.5), ('BMIN', 0.4), ('BPA', 10.0), ('BUNIT', 'Jy/beam'),
                 ('CTYPE3', 'FREQ'), ('CRVAL3', 23e6), ('RESTFRQ', 23e6)):
        h[k] = v
    src = os.path.join(d, 'allsky.fits')
    fits.writeto(src, np.random.RandomState(1).normal(size=(400, 400)).astype(np.float32), h)
    cut = IO.write_cutout(src, cdir, c, '23MHz', 'V', ['dewarped', 'aligned', 'msub', 'ssub'], c.ra, c.dec,
                          ('2024-12-31T07:00:00', '2024-12-31T08:00:00', 3600.0), beam=(0.5, 0.4, 10.0),
                          ebeam=(0.5, 0.4, 10.0), dewarped=True, aligned=True, msub=True, ssub=True)
    assert cut, 'the package cutout writer failed on a synthetic all-sky image'
    res = validate.run('followup_fits', cut)
    assert res[0]['passed'] and res[0]['value'] == [161, 161], res


def test_summary():
    from exopipe.phase3.followup import candidate as CM, score as SC
    d = _fresh('summary')
    out = os.path.join(d, 'out')
    top50 = os.path.join(d, 'top50.csv')
    with open(top50, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['rank', 'ra', 'dec', 'band', 'date', 'cadence', 'lst', 'peak_snr'])
        w.writeheader()
        w.writerow(dict(rank=1, ra=94.9, dec=41.1, band='Red', date='2024-12-31', cadence='deep', lst='07h', peak_snr=7))
        w.writerow(dict(rank=2, ra=120.0, dec=20.0, band='Blue', date='2025-01-10', cadence='deep', lst='08h', peak_snr=8))
    c1 = CM.load_top50(top50)[0]
    # the package config was imported by test_investigate with its own OUT_ROOT: summary must use the run's --out, so
    # this test runs the step in a child process (one process = one set of package paths, gate G13)
    cdir = os.path.join(out, 'transients', 'Run_PROD', 'candinv', c1.cand_id)
    os.makedirs(cdir)
    ev = {'candidate': {'cand_id': c1.cand_id}, 'context': {'ateam_sep_deg': 40.0, 'elevation_deg': 60.0},
          'p_astro': {'p_astro': 0.123}}
    with open(os.path.join(cdir, f'{c1.cand_id}_evidence.json'), 'w') as f:
        json.dump(ev, f)
    with open(os.path.join(cdir, f'{c1.cand_id}_evidence.provenance.json'), 'w') as f:
        json.dump({'gittag': 'x', 'inputs': [], 'params': {}, 'generator': 'candinv'}, f)
    import subprocess
    p = subprocess.run([sys.executable, '-m', 'exopipe.phase3', 'followup', 'summary', '--set',
                        f'followup_investigate.TOP50_CSV={top50}', '--out', out, '--cores', 'none'],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=REPO), cwd=d, timeout=120)
    assert p.returncode == 0, p.stderr[-800:]
    r = _records(out, 'summary_investigation_summary')
    assert r['status'] == 'done' and r['meta']['n_without_evidence'] == 1, r
    rows = list(csv.DictReader(open(os.path.join(out, 'transients', 'Run_PROD', 'candinv',
                                                 'investigation_summary.csv'))))
    want = SC.p_astrophysical(ev)['p_astro']
    assert rows[0]['rank'] == '1' and float(rows[0]['p_astro']) == want and rows[1]['error'], rows
    assert json.load(open(os.path.join(cdir, f'{c1.cand_id}_evidence.json')))['p_astro']['p_astro'] == want
    # snap_unphysical: the attic rule (|V S/N| >= 7.0 and V/I > 1.25, attic/run_all_top50.py:51-52) through config
    from exopipe.phase3.steps import followup as F
    attic = lambda ss: any(abs(r.get('V_snr', 0) or 0) >= 7.0 and (r.get('vi_ratio') or 0) > 1.25 for r in ss)  # noqa
    for ss in ([{'V_snr': 7.0, 'vi_ratio': 1.26}], [{'V_snr': -6.99, 'vi_ratio': 3.0}], [{'V_snr': 9.0, 'vi_ratio': 1.25}],
               [{'V_snr': 5.0, 'vi_ratio': 2.0}, {'V_snr': -8.0, 'vi_ratio': None}, {'V_snr': 7.5, 'vi_ratio': 1.3}]):
        assert F._snap_peak_vi({'snapshot_spectrum': ss})[2] == attic(ss), ss
    # G28 provenance time: a record's 'finished' (UTC text) compared in UTC (calendar.timegm, not mktime)
    rdir = os.path.join(d, 'rec')
    os.makedirs(rdir)
    tgt = _write(os.path.join(d, 'lc3_x.npz'), 'x')
    os.utime(tgt, (1760000000, 1760000000))                    # 2025-10-09T08:53:20Z
    for fin, want in (('2025-10-09T08:53:21Z', True), ('2025-10-09T08:53:17Z', False)):
        with open(os.path.join(rdir, 'lc3_x.json'), 'w') as f:
            json.dump({'status': 'done', 'finished': fin, 'outputs': [{'path': os.path.abspath(tgt)}]}, f)
        run_ = type('R', (), {'units': []})()
        assert (F._recorded(run_, tgt, 'lc3_x', [rdir]) is not None) == want, fin


def test_hone():
    d = _fresh('hone')
    cands = _write(os.path.join(d, 'cands.csv'), 'cand_id,lst,date,ra,dec,snr,band\n'
                                                 'c1,07h,2024-12-31,94.89,41.09,6.5,Red\nc2,14h,2025-05-08,217.15,49.84,-7.1,\n')
    out = os.path.join(d, 'out')
    rc, o, e = _cli(['hone', '--cand-csv', cands, '--work', os.path.join(d, 'work'), '--out', out])
    assert rc == 3, (rc, e)
    recs = _records(out)
    assert {k: v['status'] for k, v in recs.items()} == {'c1': 'no-input', 'c2': 'no-input'}, recs.keys()
    assert all(v['reason'].startswith('G17: no-products') for v in recs.values())
    rows = list(csv.DictReader(open(os.path.join(out, 'honing', 'honed.csv'))))
    assert [r['status'] for r in rows] == ['no-products', 'no-products']
    assert any('phase3_honing.py' in p for p in recs['c1']['inputs'])
    # content test on a synthetic table of every status
    from exopipe.phase3.core import validate
    t = _write(os.path.join(d, 'honed_syn.csv'),
               'cand_id,lst,date,ra,dec,snr,status,stack_all_snr,honed_snr,box_t,box_f,emp_p\n'
               'a,07h,2024-12-31,1,2,6.0,honed,5.9,8.1,1-3,64MHz-73MHz,0.111\n'
               'b,07h,2024-12-31,1,2,6.0,repro-fail,2.0,,,,\n'
               'c,07h,2024-12-31,1,2,6.0,no-products,,,,,\n')
    assert validate.run('honing_table', t, cand_id='a')[0]['passed']


def _lc_world(d, lsts, bands=('Red', 'Green', 'Blue'), nights=('2025-01-01', '2025-01-02')):
    """A synthetic archive of (empty) 10-minute frame files for Stokes I and a 3-position list."""
    P = _cfg().followup_periodicity
    arch = os.path.join(d, 'archive')
    for lst in lsts:
        for b in bands:
            for n in nights:
                fd = os.path.join(arch, lst, n, P.LC_GENERATION, b, 'I', '10min')
                os.makedirs(fd, exist_ok=True)
                for k in (0, 1):
                    open(os.path.join(fd, f'{b}_I_10min_Taper_Robust+0.0_pbcorr_dewarped_aligned_msub_LST{lst}_bin{k}_'
                                          f'{n.replace("-", "")}_0100.fits'), 'w').close()
    pos = _write(os.path.join(d, 'positions.csv'), 'source_id,ra,dec\nS1,10,40\nS2,20,45\nS3,30,50\n')
    return arch, pos


def test_periodicity():
    CFG, F = _cfg(), _step()
    from exopipe.phase3.core import cells
    d = _fresh('periodicity')
    cols = F._script_literal('search', 'cols')
    keys = F._script_literal('page', 'keys')
    scripts = dict(CFG.followup.SCRIPTS,
                   lightcurves=_write(os.path.join(d, 'lc_fake.py'), LC_FAKE),
                   search=_write(os.path.join(d, 'search_fake.py'), SEARCH_FAKE.format(cols=cols)),
                   page=_write(os.path.join(d, 'page_fake.py'), PAGE_FAKE.format(keys=keys)))
    arch, pos = _lc_world(d, cells.LSTS)
    sets = ['--set', f'followup_periodicity.LC_ARCHIVE_ROOT={arch}', '--set', f'followup_periodicity.LC_POSITIONS={pos}']
    out1, out2 = os.path.join(d, 'out1'), os.path.join(d, 'out2')
    with _patched(followup__SCRIPTS=scripts, followup__MIRRORS={}):
        # one cell only: the search refuses (G28: 1 of 24 cells)
        rc, o, e = _cli(['periodicity', '--stages', 'lightcurves,search', '--cells', '06h_Red', '--bands', 'Red',
                         '--stokes', 'I'] + sets + ['--out', out1])
        assert rc == 2, (rc, e)
        r = _records(out1)
        assert r['lc3_06h_Red_I']['status'] == 'done' and r['Red_I']['status'] == 'refused', r.keys()
        assert 'G28' in r['Red_I']['reason'] and '1 of 24' in r['Red_I']['reason'], r['Red_I']['reason']
        assert r['lc3_06h_Red_I']['admitted'] == ['2025-01-01', '2025-01-02']
        # the whole chain (default jobs Red_I, Green_I, Blue_I: 72 light-curve units, 3 searches, the page)
        rc, o, e = _cli(['periodicity'] + sets + ['--out', out1])
        assert rc == 0, (rc, e)
        r = _records(out1)
        assert sum(k.startswith('lc3_') and v['status'] == 'done' for k, v in r.items()) == 72, sorted(r)[:5]
        assert all(r[f'{b}_I']['status'] == 'done' for b in ('Red', 'Green', 'Blue'))
        assert len(r['Red_I']['meta']['lc_provenance']) == 24 and r['page_I']['status'] == 'done', r['page_I']
        assert all(t['passed'] for v in r.values() for t in v.get('tests', [])), \
            [t for v in r.values() for t in v.get('tests', []) if not t['passed']]
        log = open(os.path.join(out1, '_records', 'followup', 'logs', 'Red_I.log')).read()
        assert 'DONE 40 100' in log, log[-300:]              # ELEV_FLOOR / MIN_BLOCKS as run_p4_fleet.sh passes them
        assert os.path.exists(os.path.join(out1, 'page', 'periodicity_page_I', 'index.html'))
        # a second invocation reads the first one's light curves through their records
        rc, o, e = _cli(['periodicity', '--stages', 'search', '--lc-root', os.path.join(out1, 'lightcurves'),
                         '--bands', 'Red', '--stokes', 'I'] + sets + ['--out', out2])
        assert rc == 0, (rc, e)
        assert 'record ' in list(_records(out2, 'Red_I')['meta']['lc_provenance'].values())[0]
        # a light curve rewritten after its record is not recorded any more: refused
        f = os.path.join(out1, 'lightcurves', 'lc3_07h_Red_I.npz')
        os.utime(f, (time.time() + 60, time.time() + 60))
        rc, o, e = _cli(['periodicity', '--stages', 'search', '--lc-root', os.path.join(out1, 'lightcurves'),
                         '--bands', 'Red', '--stokes', 'I', '--fresh'] + sets + ['--out', out2])
        assert rc == 2 and 'G28' in _records(out2, 'Red_I')['reason'], (rc, e)
        # no light curves at all: G27 (exit 3) for the search; the page then refuses (G28: no tables)
        rc, o, e = _cli(['periodicity', '--stages', 'search,page', '--lc-root', os.path.join(d, 'nowhere'),
                         '--bands', 'Red', '--stokes', 'I'] + sets + ['--out', os.path.join(d, 'out3')])
        r3 = _records(os.path.join(d, 'out3'))
        assert rc == 2 and r3['Red_I']['status'] == 'no-input' and r3['page_I']['status'] == 'refused', (rc, r3.keys())


def test_fold_and_source_page():
    CFG = _cfg()
    d = _fresh('fold')
    lc = os.path.join(d, 'lc.csv')
    with open(lc, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['band', 'mjd_mid', 'flux_jy', 'err_jy', 'night', 'lst', 'cadence'])
        for i in range(120):
            w.writerow(['Red' if i % 2 else 'Green', 60000 + i * 0.007, 1.0 + 0.01 * (i % 5), 0.05, '2025-01-01', '18h',
                        '10min'])
        w.writerow(['Blue', 60001.0, 1.0, 0.05, '2025-01-01', '18h', '10min'])
    out = os.path.join(d, 'out')
    with _patched(followup__SCRIPTS=dict(CFG.followup.SCRIPTS, fold=_write(os.path.join(d, 'fold_fake.py'), FOLD_FAKE)),
                  followup__MIRRORS={}):
        rc, o, e = _cli(['fold', '--lightcurve', lc, '--prefix', 'J1810', '--ra', '272.6', '--dec', '17.7',
                         '--period-d', '0.14817', '--fold-bands', 'Red,Green', '--out', out])
        assert rc == 0, (rc, e)
        r = _records(out, 'J1810')
        assert sorted(x['class'] for x in r['outputs']) == ['fold_table'] + ['followup_figure'] * 3
        assert all(t['passed'] for t in r['tests']) and r['tests'][0]['value'] == 120, r['tests']
    rc, o, e = _cli(['source-page', '--source-id', 'S0064780', '--ra', '279.6', '--dec', '-6.9', '--out', out,
                     '--dry-run'])
    assert rc == 0 and 'source_diag_page.py S0064780' in o, (rc, o, e)


def test_negative():
    import numpy as np
    from exopipe.phase3.core import validate
    _step()
    d = _fresh('negative')
    # plan with a 7 h pointing
    z = dict(np.load(f'{ARCH}/2026-10/wxtplan_20261012T154103Z.npz'))
    z['end_utc'] = z['end_utc'].copy()
    z['end_utc'][0] = '2026-10-12T23:59:59'
    p = os.path.join(d, 'wxtplan_20300101T000000Z.npz')
    np.savez_compressed(p, **z)
    assert not validate.run('ep_wxt_plan', p)[0]['passed']
    # fetch log: unknown status
    log = os.path.join(d, 'fetch_log.csv')
    _write(log, 'fetch_utc,status,n_pointings,plan_start,plan_end,span_h,file,detail\n2026-01-01T00:00:00,maybe,,,,,,x\n')
    assert not validate.run('ep_wxt_fetch_log', log)[0]['passed']
    # coverage gaps that do not match the plans
    shutil.copyfile(f'{ARCH}/2026-10/wxtplan_20261012T154103Z.npz', os.path.join(d, 'wxtplan_20261012T154103Z.npz'))
    os.remove(p)
    gp = _write(os.path.join(d, 'coverage_gaps.csv'), 'gap_start_utc,gap_end_utc,minutes\n2026-10-12T00:00:00,'
                                                      '2026-10-12T01:00:00,60.0\n')
    assert not validate.run('ep_wxt_coverage_gaps', gp)[0]['passed']
    # region with a wrong radius
    rg = _write(os.path.join(d, 'x.reg'), 'global\nfk5\ncircle(1.0,2.0,2.0d) # a\ncircle(1.0,2.0,2.0d) # b\n')
    _write(rg + '.provenance.json', json.dumps({'gittag': 1, 'inputs': [], 'params': {}, 'generator': 'candinv'}))
    assert not validate.run('followup_region', rg)[0]['passed']
    # dossier whose p_astro is not the logistic of its logit
    ev = os.path.join(d, 'rank01_J000000+000000_LST00h_evidence.json')
    _write(ev, json.dumps({'candidate': {'cand_id': 'rank01_J000000+000000_LST00h'},
                           'p_astro': {'p_astro': 0.9, 'logit': -3.0, 'ateam_veto': False}}))
    _write(ev.replace('.json', '.provenance.json'), json.dumps({'gittag': 1, 'inputs': [], 'params': {},
                                                                 'generator': 'candinv'}))
    assert not validate.run('followup_dossier', ev)[0]['passed']
    # TESS light curve outside the BTJD range; summary without RADIO line
    tl = _write(os.path.join(d, 'k_s1_SPOC120.csv'), 'time,flux\n100.0,1.0\n')
    assert not validate.run('tess_lightcurve', tl)[0]['passed']
    ts = _write(os.path.join(d, 'k_summary.txt'), 'STAR x\nSECTOR 1\n')
    assert not validate.run('tess_summary', ts)[0]['passed']
    # truncated PNG; fold table with a phase out of range; honing repro-fail inside tolerance; stamp of the wrong size
    png = os.path.join(d, 'f.png')
    _png(png)
    with open(png, 'r+b') as f:
        f.truncate(500)
    assert not validate.run('followup_figure', png)[0]['passed']
    ft = _write(os.path.join(d, 'p_folded.csv'), 'band,lst,night,mjd_mid_utc,mjd_bary_tdb,phase,flux_jy,err_jy,'
                                                 'normalised_flux,kept_after_mad\nRed,,n,1,1,+0.7000,1,1,1,1\n')
    assert not validate.run('fold_table', ft)[0]['passed']
    ht = _write(os.path.join(d, 'honed.csv'), 'cand_id,lst,date,ra,dec,snr,status,stack_all_snr\n'
                                              'a,07h,2024-12-31,1,2,6.0,repro-fail,5.9\n')
    assert not validate.run('honing_table', ht)[0]['passed']
    from astropy.io import fits
    st = os.path.join(d, 'c_honedbox.fits')
    fits.writeto(st, np.zeros((10, 10), np.float32))
    assert not validate.run('honing_stamp', st)[0]['passed']
    # periodicity table with a duplicate source and a short period out of range
    import exopipe.phase3.steps.followup as F
    cols = F._script_literal('search', 'cols')
    pt = os.path.join(d, 'periodicity_Red_I.csv')
    with open(pt, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerow({'source_id': 'S1', 'dip_P': '5.0', 'n_blocks': 200})
        w.writerow({'source_id': 'S1'})
    res = validate.run('periodicity_table', pt)[0]
    assert not res['passed'] and 'duplicate' in res['detail'] and 'short' in res['detail'], res
    # page with a broken link
    pg = _write(os.path.join(d, 'index.html'), '<html><body><img src="img/none.png"></body></html>')
    assert not validate.run('followup_page', pg)[0]['passed']


def test_real_products():
    """READ-ONLY: content tests on production products, and --dry-run on real items."""
    from exopipe.phase3.core import validate
    _step()
    items = [('ep_wxt_plan', f'{ARCH}/2026-10/wxtplan_20261012T124103Z.npz'),
             ('ep_wxt_raw', RAW_OK), ('ep_wxt_refused', RAW_REFUSED),
             ('ep_wxt_fetch_log', f'{ARCH}/fetch_log.csv'), ('ep_wxt_coverage_gaps', f'{ARCH}/coverage_gaps.csv'),
             ('tess_lightcurve', '/lustre/gh/main/exopipe_ops/tess_more/tess_HD154088_s91_SPOC120.csv'),
             ('followup_figure', '/lustre/gh/main/exopipe_ops/tess_more/tess_HD154088_s91.png'),
             ('periodicity_lightcurves', '/lustre/gh/main/phase3/periodicity5/lc3_06h_Red_I.npz'),
             ('periodicity_table', f'{BW}/periodicity_p5/periodicity_Red_I.csv'),
             ('periodicity_merged', f'{BW}/allsky_periodicity_I.csv')]
    for cls, p in items:
        res = validate.run(cls, p)
        print(f"  {'PASS' if all(r['passed'] for r in res) else 'FAIL'} {cls:26s} {os.path.basename(p)}: "
              f"{res[0]['detail'][:110]}")
        assert all(r['passed'] for r in res), (cls, p, res)
    out = os.path.join(SCRATCH, 'dry')
    import subprocess                               # a child: the package paths are set once per process (G13)
    p = subprocess.run([sys.executable, '-m', 'exopipe.phase3', 'followup', 'investigate', '--ra', '94.89483019',
                        '--dec', '41.09169194', '--lst', '07h', '--date', '2024-12-31', '--band', '23MHz', '--dry-run',
                        '--out', out, '--cores', 'none'], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=REPO), cwd=SCRATCH, timeout=120)
    assert p.returncode == 0 and 'rank00_J061934+410530_LST07h' in p.stdout and 'Vresidual: /data07' in p.stdout, \
        (p.returncode, p.stdout, p.stderr[-500:])
    rc, o, e = _cli(['periodicity', '--stages', 'search,page', '--dry-run', '--out', out])
    assert rc == 0 and o.count('24 of 24 cells present now') == 3, (rc, o, e)
    rc, o, e = _cli(['ep-plan', '--action', 'audit', '--dry-run'])
    assert rc == 2 and 'G7' in o and 'stored plans' in o, (rc, o, e)
    assert not os.path.exists(out), 'a dry-run wrote'


TESTS = ['test_cli', 'test_config', 'test_ep_plan', 'test_tess', 'test_real_products', 'test_investigate',
         'test_summary', 'test_hone', 'test_periodicity', 'test_fold_and_source_page', 'test_negative']


def main(argv):
    names = argv or TESTS
    shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(SCRATCH)
    here = os.getcwd()
    os.chdir(SCRATCH)
    failed = []
    t00 = time.time()
    try:
        for n in names:
            t0 = time.time()
            try:
                globals()[n]()
                print(f'PASS {n} ({time.time() - t0:.1f} s)')
            except Exception:
                import traceback
                traceback.print_exc()
                failed.append(n)
                print(f'FAIL {n} ({time.time() - t0:.1f} s)')
    finally:
        os.chdir(here)
        shutil.rmtree(SCRATCH, ignore_errors=True)
    print(f'{len(names) - len(failed)} of {len(names)} passed in {time.time() - t00:.0f} s'
          + (f'; failed: {", ".join(failed)}' if failed else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
