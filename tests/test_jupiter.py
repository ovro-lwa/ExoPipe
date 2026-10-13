"""Standing test of `exopipe3 jupiter` (plain asserts, no pytest):

    NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py \
        /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_jupiter.py

Synthetic data in a fresh directory under /tmp/exopipe3_tests_jupiter/ (removed at the end):
- end2end: a synthetic Jupiter cell (2 subbands x I,V; 12 frames; flowed cubes, window stacks, arm-A residuals, x2
  residuals, admission tables, ephemeris) measured by the REAL engine clv15/phase3_jupiter_measure.py in place (extract under
  py310_orca, fit with CASA imfit, assign, test) into a /tmp --out under the write guard; every content test passes; the
  engine's own CONTENT_TEST.txt is ALL-PASS; run.log lines; --resume skips all four units; a re-run of extract refuses the
  stale fits (J10) unless --fresh --supersede TAG; corrupted products fail their class tests.
- gates: J1, J7 (FLUXBAKE), J4 (no JUPMASK), J12, J13, pinned-input override into production, MIRROR override refused,
  stage selection.
- flag-backfill: J30 code gate, explicit window, shard assignment = the launcher's awk, budget integer division, J31, J32,
  verify of an empty copy tree through the real tool.
- serve: the v2 static server started on a parity port, its served content test, exact-pid stop.
- dry runs of every action on the real cell 05h_2024-12-29 (read only).
Read-only checks of real production products (content tests on 05h_2024-12-29, the window, the ephemeris, the 2026-10-07
VERIFY file, one flagged copy directory).
"""
import csv
import io
import json
import math
import os
import shutil
import socket
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

import numpy as np                                    # noqa: E402

from exopipe.phase3 import cli                        # noqa: E402
from exopipe.phase3.config import CFG                 # noqa: E402
from exopipe.phase3.core import env, validate         # noqa: E402

env.legacy()
BASE = '/tmp/exopipe3_tests_jupiter'
os.makedirs(BASE, exist_ok=True)
TMP = tempfile.mkdtemp(prefix='run_', dir=BASE)
CELL = '05h_2099-01-01'
REAL_CELL = '05h_2024-12-29'
REAL = '/lustre/gh/main/phase3/jupiter/products/' + REAL_CELL
T0 = time.time()
RESULTS = []


def step(argv, quiet=True):
    """Run `exopipe3 <argv>` in this process; returns (rc, stdout+stderr text)."""
    CFG.reset()
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        rc = cli.main(list(argv))
    txt = buf.getvalue()
    if not quiet:
        print(txt)
    return rc, txt


def records(out):
    d = os.path.join(out, '_records', 'jupiter')
    return {f[:-5]: json.load(open(os.path.join(d, f))) for f in os.listdir(d) if f.endswith('.json')} if os.path.isdir(d) else {}


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond)))
    print(f"{'ok  ' if cond else 'FAIL'} {name}" + (f'  [{detail}]' if detail and not cond else ''))
    assert cond, f'{name}: {detail}'


# ---------------------------------------------------------------------------------------------------------------------
# synthetic cell
# ---------------------------------------------------------------------------------------------------------------------
N, PIX, RA0, DEC0 = 96, 0.125, 71.82, 21.77
DC = 1.2
AMP = {('41MHz', 'I'): 60.0}


def _wcs_header():
    from astropy.io import fits
    h = fits.Header()
    h['NAXIS'] = 2
    h['NAXIS1'] = N
    h['NAXIS2'] = N
    h['CTYPE1'], h['CTYPE2'] = 'RA---SIN', 'DEC--SIN'
    h['CRVAL1'], h['CRVAL2'] = RA0, DEC0
    h['CRPIX1'], h['CRPIX2'] = 49.0, 49.0
    h['CDELT1'], h['CDELT2'] = -PIX, PIX
    h['CUNIT1'], h['CUNIT2'] = 'deg', 'deg'
    h['RADESYS'] = 'ICRS'
    h['EQUINOX'] = 2000.0
    h['BMAJ'], h['BMIN'], h['BPA'] = 3 * PIX, 3 * PIX, 0.0
    h['BUNIT'] = 'JY/BEAM'
    return h


def make_cell(root, bake=False, no_jupmask=False):
    """The inputs of one synthetic Jupiter cell; returns (sublst root, cubes root, ephem dir, window csv)."""
    from astropy.io import fits
    from astropy.time import Time
    lst, date = CELL.split('_', 1)
    sl, cubes, eph = os.path.join(root, 'sublst'), os.path.join(root, 'cubes'), os.path.join(root, 'ephem')
    for d in (sl, cubes, eph):
        os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:N, 0:N]
    s = 3.0 / 2.3548
    G = np.exp(-((xx - 48.0) ** 2 + (yy - 48.0) ** 2) / (2 * s * s))
    M = (0.01 * xx + 0.02 * yy).astype(np.float64)
    t0 = Time('2099-01-01T06:20:00').unix
    ts = [time.strftime('%Y%m%d_%H%M%S', time.gmtime(t0 + 10 * k)) for k in range(12)]
    hdr = _wcs_header()
    for band in ('18MHz', '41MHz'):
        for S in ('I', 'V'):
            amp = AMP.get((band, S), 0.0)
            sd = os.path.join(sl, lst, CELL, band, S)
            os.makedirs(sd, exist_ok=True)
            frames = np.stack([(DC * (M + amp * G + rng.normal(0, 1.0, (N, N)))).astype(np.float32) for _ in ts])
            hs = hdr.copy()
            hs['FLUXDC'] = DC
            if bake:
                hs['FLUXBAKE'] = True
            fits.writeto(os.path.join(sd, f'sublst_{CELL}_{band}_{S}_w0.fits'), (M + 0.5).astype(np.float32), hs)
            hr = hdr.copy()
            if not no_jupmask:
                hr['JUPMASK'] = 2.0
            fits.writeto(os.path.join(sd, f'sublst_{CELL}_{band}_{S}_w0_resid_loo.fits'), np.full((N, N), 0.5, np.float32), hr)
            with open(os.path.join(sd, f'admission_{CELL}_{band}_{S}.csv'), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['ts', 'ts_int', 'date_obs', 't_mid_unix', 'lst_min', 'window', 'finite_frac', 'beam_area_deg2',
                            'noise_jyb', 'sun_elev_deg', 'sun_branch', 'used', 'refused'])
                for k, t in enumerate(ts):
                    w.writerow([t, t, '', f'{t0 + 10 * k:.3f}', f'{k / 6.0:.3f}', 0, '1.000', '0.05', '1.0', '-60', 'evening', 1, ''])
            cd = os.path.join(cubes, CELL)
            os.makedirs(cd, exist_ok=True)
            hs_str = hdr.tostring(sep='\n')
            cols = [fits.Column(name='ts', format='15A', array=ts),
                    fits.Column(name='header', format='%dA' % len(hs_str), array=[hs_str] * len(ts))]
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(frames, hdr.copy(), name='FRAMES'),
                          fits.ImageHDU(np.zeros(frames.shape, np.int16), name='MASK'),
                          fits.BinTableHDU.from_columns(cols, name='FRAME_TABLE')]).writeto(
                os.path.join(cd, f'{CELL}_{band}_{S}_flowcube.fits'))
            xd = os.path.join(sl, f'{lst}_x2', CELL, band, S)
            os.makedirs(xd, exist_ok=True)
            hx = hdr.copy()
            hx['MJD-MID'] = Time(t0 + 55.0, format='unix').mjd
            hx['NFRAMES'] = len(ts)
            if not no_jupmask:
                hx['JUPMASK'] = 2.0
            if bake:
                hx['FLUXBAKE'] = True
            fits.writeto(os.path.join(xd, f'sublst_{CELL}_{band}_{S}_w0_resid_loo.fits'),
                         np.mean(frames.astype(np.float64) / DC - M, axis=0).astype(np.float32), hx)
    with open(os.path.join(eph, f'jupiter_ephem_{date}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        moons = {'io': (2.0, 0.3), 'europa': (-1.5, 0.2), 'ganymede': (5.0, 0.5), 'callisto': (0.6, 0.5)}
        head = ['utc', 'jup_ra', 'jup_dec', 'jup_az', 'jup_el', 'cml_earth', 'io_phase_earth']
        for m in moons:
            head += [f'{m}_dx_arcmin', f'{m}_dy_arcmin', f'{m}_grazing_deg']
        w.writerow(head)
        for i in range(91):
            t = Time('2099-01-01T06:00:00').unix + 60 * i
            row = [time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(t)), RA0, DEC0, 180.0, 70.0, (100 + 0.6 * i) % 360,
                   (200 + 0.2 * i) % 360]
            for m, (dx, dy) in moons.items():
                row += [dx, dy, 45.0]
            w.writerow(row)
    win = os.path.join(root, 'nights.csv')
    with open(win, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'lst', 'utc_start', 'utc_end', 'jup_el_mid', 'searched', 'excluded_any_band'])
        w.writerow([date, lst, '2099-01-01T06:20:00', '2099-01-01T07:20:00', 70.0, 1, 0])
    return sl, cubes, eph, win


class window_patch:
    """jupiter.WINDOW_CSV is protected (no --set): the test points the section at its synthetic window for a while."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.old = CFG.jupiter.WINDOW_CSV
        CFG.jupiter.WINDOW_CSV = self.path

    def __exit__(self, *a):
        CFG.jupiter.WINDOW_CSV = self.old


# ---------------------------------------------------------------------------------------------------------------------
def test_end2end():
    root = os.path.join(TMP, 'e2e')
    sl, cubes, eph, win = make_cell(root)
    out, work = os.path.join(root, 'out'), os.path.join(root, 'work')
    common = ['--cells', CELL, '--out', out, '--work', work, '--sublst-root', sl, '--set', f'jupiter.CUBES_ROOT={cubes}',
              '--set', f'jupiter.EPHEM_DIR={eph}', '--scopes', '18MHz,41MHz', '--cores', 'none']
    with window_patch(win):
        rc, txt = step(['jupiter', 'measure', '--dry-run'] + common)
        check('end2end dry-run plans the 4 pairs, every gate passes', rc == 0 and 'pairs planned: 4' in txt
              and 'every gate passes' in txt, txt[-1500:])
        t = time.time()
        rc, txt = step(['jupiter', 'measure'] + common)
        recs = records(out)
        check('end2end measure exit 0, 4 units done', rc == 0 and all(recs.get(f'measure.{s}.{CELL}', {}).get('status') == 'done'
                                                                       for s in ('extract', 'fit', 'assign', 'test')),
              txt[-3000:])
        print(f'     (measure of the synthetic cell: {time.time() - t:.0f} s)')
        od = os.path.join(out, 'products', CELL)
        ct = open(os.path.join(od, 'CONTENT_TEST.txt')).read().strip().splitlines()
        check('engine CONTENT_TEST ALL-PASS', ct[-1] == 'ALL-PASS', ' | '.join(ct))
        tests = [x for r in recs.values() for x in r['tests']]
        check('every content test of every output passed', tests and all(x['passed'] for x in tests),
              [x for x in tests if not x['passed']])
        names = {x['name'] for x in tests}
        for cls in ('jupiter_meas.rows_and_columns', 'jupiter_meas.admission_coverage', 'jupiter_cutouts.cutout_value_at_jupiter',
                    'jupiter_crop.crop_equals_cutout', 'jupiter_fits.one_record_per_crop', 'jupiter_assign.assign_recomputed',
                    'jupiter_cell.tests_1_to_4'):
            check(f'content test {cls} ran', cls in names)
        fits10 = [json.loads(l) for l in open(os.path.join(od, 'fits_10s.jsonl'))]
        check('imfit converged on the bright 41 MHz I frames', len(fits10) == 12 and all(x.get('converged') for x in fits10),
              [x.get('err') for x in fits10][:3])
        a = list(csv.DictReader(open(os.path.join(od, 'assign_10s.csv'))))
        check('detected rows = the 12 bright frames', sum(int(r['detected']) for r in a) == 12)
        ex = recs[f'measure.extract.{CELL}']
        check('record: admitted night, in-place code identity, globals redirected',
              ex['admitted'] == ['2099-01-01'] and any(p.endswith('phase3_jupiter_measure.py') for p in ex['inputs'])
              and set(ex['meta']['inplace'][0]['globals']) >= {'SL', 'CUBES', 'EPH', 'BANDS', 'WIDE'})
        runlog = open(os.path.join(out, 'products', 'logs', 'run.log')).read()
        check('run.log DONE line as the launcher', f'DONE {CELL} ALL-PASS' in runlog, runlog)
        check('nothing written outside --out / --work', not os.path.exists(os.path.join(sl, '..', 'stray')))
        # resume: every unit skipped (records, inputs, code, config unchanged; outputs re-tested, the cell test in place)
        rc, txt = step(['jupiter', 'measure', '--resume'] + common)
        check('--resume skips the 4 units', rc == 0 and txt.count('skipped (resume') == 4, txt[-1500:])
        # J10: extract again, then fit refuses the stale fits; --fresh --supersede moves them aside and fits again
        rc, txt = step(['jupiter', 'measure', '--stages', 'extract,fit'] + common)
        check('J10 stale fits refused after a new extract', rc == 2 and 'J10 stale fits' in txt, txt[-1500:])
        rc, txt = step(['jupiter', 'measure', '--stages', 'fit', '--fresh', '--supersede', 't1'] + common)
        check('--fresh --supersede moves the fits aside and fits again', rc == 0 and
              os.path.exists(os.path.join(od, '_superseded', 't1', 'MANIFEST.csv')), txt[-1500:])
        # corrupted products fail their class tests
        corrupt_checks(od, work)


def corrupt_checks(od, work):
    tmp = os.path.join(TMP, 'corrupt', CELL)
    shutil.copytree(od, tmp)
    p = os.path.join(tmp, 'meas_10s.csv')
    rows = list(csv.DictReader(open(p)))
    rows[3]['pix_snr'] = '%.2f' % (float(rows[3]['pix_snr']) + 1.0)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    r = validate.run('jupiter_meas', p)
    check('corrupt pix_snr fails jupiter_meas', not r[0]['passed'] and 'pix_snr 1' in r[0]['detail'], r[0]['detail'])
    p = os.path.join(tmp, 'fits_10s.jsonl')
    lines = open(p).read().splitlines()
    open(p, 'w').write('\n'.join(lines + lines[:1]) + '\n')
    r = validate.run('jupiter_fits', p)
    check('duplicate fit record fails jupiter_fits', not r[0]['passed'] and 'duplicates 1' in r[0]['detail'], r[0]['detail'])
    p = os.path.join(tmp, 'assign_10s.csv')
    rows = list(csv.DictReader(open(p)))
    i = next(i for i, x in enumerate(rows) if x['detected'] == '1')
    rows[i]['assignment'] = 'io'
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    r = validate.run('jupiter_assign', p)
    check('changed assignment fails jupiter_assign', not r[0]['passed'] and 'assignment 1' in r[0]['detail'], r[0]['detail'])
    from astropy.io import fits
    crop = next(x['crop'] for x in csv.DictReader(open(os.path.join(od, 'meas_10s.csv'))) if x['crop'])
    cdir = os.path.join(TMP, 'corrupt', 'crops')
    os.makedirs(cdir)
    with fits.open(crop) as h:
        d = h[0].data.copy()
        d[5, 5] += 3.0
        fits.writeto(os.path.join(cdir, os.path.basename(crop)), d, h[0].header)
    from exopipe.phase3.lib.jupiter import tests as T
    os.rename(crop, crop + '.away')
    try:
        ok, n, det = T.crop_equals_cutout(os.path.join(od, 'meas_10s.csv'), crop_dir=cdir)
    finally:
        os.rename(crop + '.away', crop)
    check('changed crop pixel fails jupiter_crop', not ok and 'pixels 1' in det, det)
    lines = open(os.path.join(od, 'CONTENT_TEST.txt')).read().splitlines()
    p = os.path.join(tmp, 'CONTENT_TEST.txt')
    open(p, 'w').write('\n'.join(lines[:-1] + ['FAILED']) + '\n')
    ok, _, det = T.tests_1_to_4(p, fresh_verdict=True)
    check('inconsistent verdict file fails jupiter_cell', not ok, det)


def test_gates():
    root = os.path.join(TMP, 'gates')
    sl, cubes, eph, win = make_cell(os.path.join(root, 'baked'), bake=True)
    out = os.path.join(root, 'out')
    base = ['--out', out, '--work', os.path.join(root, 'work'), '--sublst-root', sl, '--set', f'jupiter.CUBES_ROOT={cubes}',
            '--set', f'jupiter.EPHEM_DIR={eph}', '--scopes', '18MHz,41MHz', '--cores', 'none']
    with window_patch(win):
        rc, txt = step(['jupiter', 'measure', '--cells', CELL, '--stages', 'extract'] + base)
        check('J7 baked inputs refused before the extract', rc == 2 and 'J7 baked inputs' in txt, txt[-800:])
        rc, txt = step(['jupiter', 'measure', '--cells', '05h_2099-01-02', '--dry-run'] + base)
        check('J1 a cell outside the window refused', rc == 2 and 'J1 Jupiter window' in txt, txt[-500:])
        sl2, cubes2, eph2, _ = make_cell(os.path.join(root, 'nomask'), no_jupmask=True)
        b2 = [x.replace(sl, sl2).replace(cubes, cubes2) for x in base]
        rc, txt = step(['jupiter', 'measure', '--cells', CELL, '--stages', 'extract'] + b2)
        check('J4 residual without JUPMASK refused', rc == 2 and 'J4 Jupiter mask' in txt, txt[-800:])
        os.environ['JFQ_DEST'] = '/x'
        try:
            rc, txt = step(['jupiter', 'measure', '--cells', CELL, '--dry-run'] + base)
        finally:
            del os.environ['JFQ_DEST']
        check('J12 a JFQ_ variable in the environment refused', rc == 2 and 'J12' in txt, txt[-300:])
    rc, txt = step(['jupiter', 'measure', '--cells', REAL_CELL, '--dry-run', '--scopes', '41MHz'])
    check('J13 --scopes into production refused', rc == 2 and 'J13' in txt, txt[-300:])
    rc, txt = step(['jupiter', 'measure', '--cells', REAL_CELL, '--dry-run', '--sublst-root', '/tmp/x'])
    check('pinned input overridden into production refused', rc == 2 and 'pinned input' in txt, txt[-300:])
    rc, txt = step(['jupiter', 'measure', '--cells', REAL_CELL, '--dry-run', '--out', out, '--set',
                    'jupiter_measure.FIT_SNR=5.0'])
    check('MIRROR override the engine would ignore refused', rc == 2 and 'jupiter_measure.FIT_SNR' in txt, txt[-300:])
    rc, txt = step(['jupiter', 'measure', '--cells', REAL_CELL, '--dry-run', '--out', out, '--set', 'jupiter.EPHEM_NAME=x'])
    check('protected key refused', rc == 2 and 'PROTECTED' in txt, txt[-300:])
    from exopipe.phase3.lib.jupiter import common as K

    class A:
        stages, stage_from, stage_until = None, 'fit', 'assign'
    check('--from/--until select a contiguous range', K.select_stages('measure', A) == ['fit', 'assign'])
    A.stage_from, A.stage_until, A.stages = None, None, 'test,extract'
    check('--stages keeps the action order', K.select_stages('measure', A) == ['extract', 'test'])
    A.stages = None
    check('flag-backfill default stages exclude verify', K.select_stages('flag-backfill', A) == ['window', 'run'])


def test_flag():
    from exopipe.phase3.lib.jupiter import flag as F
    out = os.path.join(TMP, 'flag')
    rc, txt = step(['jupiter', 'flag-backfill', '--stages', 'window', '--cells', '03h_2025-01-05,04h_2024-12-24', '--out', out])
    state = os.path.join(out, CFG.jupiter_flag.STATE_SUBDIR)
    rows = list(csv.reader(open(os.path.join(state, 'window.csv'))))
    check('explicit window (old JFQ_CELLS) written and tested', rc == 0 and rows == [['lst', 'date', 'jup_el_mid'],
                                                                                     ['03h', '2025-01-05', ''], ['04h', '2024-12-24', '']], txt)
    w = os.path.join(TMP, 'w.csv')
    with open(w, 'w') as f:
        f.write('lst,date,jup_el_mid\n' + ''.join(f'0{i % 10}h,2025-01-{i + 1:02d},40\n' for i in range(9)))
    nl = ['n1', 'n2', 'n3', 'n4']
    awk = {n: [i for i in range(9) if i % 4 == k] for k, n in enumerate(nl)}      # awk 'NR==1 || (NR-2) % m == k'
    got = {n: [int(r[1][-2:]) - 1 for r in F.shard_rows(w, nl, n)[1]] for n in nl}
    check('shards = the launcher awk round robin', got == awk, got)

    class A:
        budget_gb = None
    check('per-shard budget = int(2000 // 4) = 500 GB', F.budget_gb(A, 4) == 500 and F.budget_gb(A, 3) == 666)
    host = socket.gethostname()
    rc, txt = step(['jupiter', 'flag-backfill', '--stages', 'run', '--nodes', host, '--out', out])
    check('J31 a copying shard without APPROVED refused', rc == 2 and 'J31' in txt, txt[-500:])
    open(os.path.join(state, f'run_{host}.log'), 'w').write('still running\n')
    rc, txt = step(['jupiter', 'flag-backfill', '--stages', 'run', '--nodes', host, '--no-copy', '--out', out])
    check('J32 a shard log without its exit line refused', rc == 2 and 'J32' in txt, txt[-500:])
    empty = os.path.join(TMP, 'empty_tree')
    os.makedirs(empty)
    rc, txt = step(['jupiter', 'flag-backfill', '--stages', 'verify', '--flagged-root', empty, '--out', out, '--cores', 'none'])
    vf = [f for f in os.listdir(state) if f.startswith('VERIFY_')]
    check('verify through the real tool on an empty tree: VERIFY OK, class test passes', rc == 0 and len(vf) == 1 and
          'VERIFY OK: 0 subband directories' in open(os.path.join(state, vf[0])).read(), txt[-800:])
    rc, txt = step(['jupiter', 'flag-backfill', '--dry-run', '--nodes', 'lwacalim01,lwacalim02,lwacalim04,lwacalim05'])
    check('dry-run on the production state: 222 cells, 4 shards, 500 GB each', 'run: 222 cells over 4 nodes' in txt and
          '--max-copy-gb 500' in txt, txt[-1500:])


def test_serve():
    out = os.path.join(TMP, 'serve')
    site = os.path.join(out, 'v2', 'site', REAL_CELL)
    os.makedirs(site)
    open(os.path.join(site, 'index.html'), 'w').write('<html>jupiter test</html>')
    json.dump({'variants': [{'label': 'V2'}, {'label': 'prod'}]}, open(os.path.join(site, 'site.json'), 'w'))
    off = 1000
    port = CFG.jupiter_serve.PORTS['v2'] + off
    from exopipe.phase3.lib.jupiter import serve as S
    if S.listeners(port):
        print(f'skip serve test: port {port} in use')
        return
    rc, txt = step(['jupiter', 'serve', '--stages', 'v2', '--port-offset', str(off), '--out', out])
    rec = os.path.join(out, '_served', f'jupiter_{port}.json')
    check('v2 static server started on the parity port, served test passes', rc == 0 and os.path.exists(rec)
          and json.load(open(rec))['pid'], txt[-800:])
    rc, txt = step(['jupiter', 'serve', '--stages', 'v2', '--port-offset', str(off), '--out', out])
    check('second call keeps the running server', rc == 0 and json.load(open(rec)).get('started') == 'earlier (kept)', txt[-500:])
    rc, txt = step(['jupiter', 'serve', '--stages', 'v2', '--port-offset', '0', '--out', out])
    check('a non-production --out on a live port refused', rc == 2 and 'parity ports' in txt, txt[-300:])
    rc, txt = step(['jupiter', 'serve', '--stages', 'v2', '--port-offset', str(off), '--out', out, '--stop'])
    check('exact-pid stop; port free', rc == 0 and not S.listeners(port), txt[-500:])


def test_dry_runs_real():
    rc, txt = step(['jupiter', 'v2', '--cells', REAL_CELL, '--dry-run'])
    check('v2 dry-run on the production root plans every default stage', rc == 0 and 'J24 disclosure pass' in txt
          and 'derived roots existing' in txt, txt[-1500:])
    rc, txt = step(['jupiter', 'v2', '--cells', REAL_CELL, '--dry-run', '--out', os.path.join(TMP, 'v2'), '--work', TMP])
    check('v2 into a non-production --out: masters, metrics, summary refused', rc == 2 and txt.count('REFUSE into a non-production') == 3,
          txt[-1500:])
    rc, txt = step(['jupiter', 'v2', '--cells', '05h_2024-12-27', '--dry-run'])
    check('J20 a second V2 cell refused', rc == 2 and 'J20' in txt, txt[-300:])
    rc, txt = step(['jupiter', 'occurrence', '--dry-run', '--out', os.path.join(TMP, 'occ')])
    check('occurrence into a non-production --out refused (verify only)', rc == 2 and 'REFUSE' in txt, txt[-500:])
    rc, txt = step(['jupiter', 'ephem', '--dates', '2024-12-29', '--dry-run'])
    check('ephem dry-run: the cached night is not fetched again', rc == 0 and 'exists - not fetched again' in txt, txt[-500:])
    rc, txt = step(['jupiter', 'ephem', '--dates', '2024-12-29', '--out', os.path.join(TMP, 'eph')])
    check('ephem refuses without a network host', rc == 2 and 'NETWORK_HOSTS' in txt, txt[-300:])
    rc, txt = step(['jupiter', 'quarantine', '--cells', REAL_CELL, '--dry-run'])
    check('quarantine dry-run lists no superseded render left', rc == 0 and '0 superseded render dir' in txt, txt[-500:])
    rc, txt = step(['jupiter', 'serve', '--dry-run'])
    check('serve dry-run names the live 8800 / 8801 processes', rc == 0 and 'port 8800' in txt and 'port 8801' in txt, txt[-800:])


def test_config_and_mirrors():
    from exopipe.phase3.core import ledger
    from exopipe.phase3.lib.jupiter import mirrors
    for s in ('jupiter', 'jupiter_measure', 'jupiter_v2', 'jupiter_occurrence', 'jupiter_flag', 'jupiter_serve'):
        for k, v in (getattr(CFG.sections()[s], 'RULED', None) or {}).items():
            for lid in (v if isinstance(v, tuple) else (v,)):
                check(f'ruling {s}.{k} = a ledger row', ledger.has_id(lid, CFG.paths.FIX_LEDGER), lid)
    n = sum(mirrors.check_literals(m) for m in mirrors.ACTION_MODULES.values())
    check(f'every MIRROR key equals its in-place module literal ({n} keys)', n >= 50)


def test_real_products_readonly():
    """Content tests on real production products (read only). 05h_2024-12-29 was measured 2026-10-05: its assign tables
    predate the Q14 band classes (2026-10-07), so assign_recomputed must fail exactly on the 41 / 64 MHz sigma model."""
    for t in ('10s', '1min', '10min'):
        r = validate.run('jupiter_meas', f'{REAL}/meas_{t}.csv')
        check(f'real meas_{t} rows_and_columns', r[0]['passed'], r[0]['detail'])
        r = validate.run('jupiter_fits', f'{REAL}/fits_{t}.jsonl')
        check(f'real fits_{t}', r[0]['passed'], r[0]['detail'])
        r = validate.run('jupiter_assign', f'{REAL}/moons_{t}.json')
        check(f'real moons_{t}', r[0]['passed'], r[0]['detail'])
    for f in ('cutouts_10s_18MHz_I.npz', 'cutouts_1min_73MHz_V.npz', 'cutouts_10min_Blue_I.npz'):
        r = validate.run('jupiter_cutouts', f'{REAL}/{f}')
        check(f'real {f}', r[0]['passed'], r[0]['detail'])
    r = validate.run('jupiter_assign', f'{REAL}/assign_1min.csv')
    check('real assign_1min fails only on the pre-Q14 sigma model (41, 64 MHz)', not r[0]['passed'] and
          "problems: sigma model" in r[0]['detail'] and "['41MHz', '64MHz']" in r[0]['detail'] and 'assignment' not in r[0]['detail'],
          r[0]['detail'])
    RES = '/lustre/gh/main/phase3/sublst/superseded_reserve_20261007/05h/root/lustre/gh/main/phase3/sublst'
    from exopipe.phase3.lib.jupiter import tests as T
    check('real meas_1min admission coverage against the pre-bake reserve', T.admission_coverage(f'{REAL}/meas_1min.csv',
                                                                                                  sublst_root=RES)[0])
    r = validate.run('jupiter_ephem', '/lustre/gh/main/phase3/jupiter/ephem/jupiter_ephem_2024-12-29.csv')
    check('real ephemeris cache', r[0]['passed'], r[0]['detail'])
    r = validate.run('jupiter_window', CFG.jupiter.WINDOW_CSV)
    check('real Jupiter window: 224 cells', r[0]['passed'] and r[0]['value'] == 224, r[0]['detail'])
    r = validate.run('jupiter_flag_verify', os.path.join(CFG.jupiter_flag.STATE_DIR, 'VERIFY_20261007T111100Z.txt'))
    check('real backfill VERIFY file', r[0]['passed'] and r[0]['value'] == 2581, r[0]['detail'])
    r = validate.run('jupiter_occurrence', '/lustre/gh/main/phase3/jupiter/v2/occurrence/validation/content_test_results.json')
    check('real occurrence results = the accepted waiver', r[0]['passed'], r[0]['detail'])


def main():
    try:
        for t in (test_config_and_mirrors, test_gates, test_flag, test_serve, test_dry_runs_real, test_real_products_readonly,
                  test_end2end):
            print(f'--- {t.__name__}')
            t()
    finally:
        CFG.reset()
        shutil.rmtree(TMP, ignore_errors=True)
    print(f'\n{sum(ok for _, ok in RESULTS)} of {len(RESULTS)} checks passed in {time.time() - T0:.0f} s')


if __name__ == '__main__':
    main()
