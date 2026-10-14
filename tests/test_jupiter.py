"""Standing test of `exopipe3 jupiter` (plain asserts, no pytest):

    NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py \
        /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_jupiter.py

Synthetic data in a fresh directory under /tmp/exopipe3_tests_jupiter/ (removed at the end):
- end2end: a synthetic Jupiter cell (2 subbands x I,V; 12 frames; flowed cubes, window stacks, arm-A residuals, x2
  residuals, admission tables, ephemeris) measured by the REAL engine clv15/phase3_jupiter_measure.py in place (extract under
  py310_orca, fit with CASA imfit, assign, test) into a /tmp --out under the write guard; every content test passes; the
  engine's own CONTENT_TEST.txt is ALL-PASS; run.log lines; --resume skips all four units (from a dirty worktree: the
  dry run shows each unit re-run, 'record written from a dirty tree'); fit refuses on another node than its extract; a re-run
  of extract refuses the stale fits (J10) unless --fresh --supersede TAG; corrupted products fail their class tests (meas,
  fits, cutouts, assign, crop, cell); the 8800 page server (jupiter_server.py in place) on a parity port passes its served
  test (slider = the M33 integrations, PNGs decode) and fails it when the table on disk changes; exact-pid stop.
- gates: J1, J7 (FLUXBAKE), J4 (no JUPMASK), J12, J13, pinned-input override into production, MIRROR override refused,
  stage selection.
- flag-backfill: J30 code gate, explicit window, shard assignment = the launcher's awk, budget integer division, J31, J32,
  verify of an empty copy tree through the real tool.
- serve: the v2 static server started on a parity port, its served content test, exact-pid stop.
- dry runs of every action on the real cell 05h_2024-12-29 (read only).
Read-only checks of real production products (content tests on 05h_2024-12-29 Phase-1 and V2 products, the window, the
ephemeris, the 2026-10-07 VERIFY file, the occurrence results); class_negatives: every class test fails on a corrupted copy
(real products copied to the scratch directory, minimal synthetic layouts for the large V2 products).
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
        # resume: every unit skipped (records, inputs, code, config unchanged; outputs re-tested, the cell test in place).
        # A record written from a dirty tree is never resumed (core/record): with other units being edited in the worktree
        # the check asserts that refusal instead, through the dry run (which plans the resume decision and writes nothing).
        if not str(ex.get('commit') or '').endswith('-dirty'):
            rc, txt = step(['jupiter', 'measure', '--resume'] + common)
            check('--resume skips the 4 units', rc == 0 and txt.count('skipped (resume') == 4, txt[-1500:])
        else:
            rc, txt = step(['jupiter', 'measure', '--resume', '--dry-run'] + common)
            check('--resume from a dirty tree: the 4 units re-run (record written from a dirty tree, never resumed)',
                  txt.count('run (record written from a dirty tree') == 4 and 'skip (' not in txt, txt[-1500:])
        # fit runs on the node that ran extract (crops are node-local): an extract record from another host refuses
        xp = os.path.join(out, '_records', 'jupiter', f'measure.extract.{CELL}.json')
        saved = open(xp).read()
        x = json.loads(saved)
        x['meta']['host'] = 'otherhost'
        open(xp, 'w').write(json.dumps(x))
        try:
            rc, txt = step(['jupiter', 'measure', '--stages', 'fit', '--fresh', '--supersede', 't0'] + common)
        finally:
            open(xp, 'w').write(saved)
        check('fit refused on another node than the extract (crops node-local); nothing moved aside', rc == 2 and
              'node-local to otherhost' in txt and not os.path.exists(os.path.join(od, '_superseded', 't0')), txt[-800:])
        # J10: extract again, then fit refuses the stale fits; --fresh --supersede moves them aside and fits again
        rc, txt = step(['jupiter', 'measure', '--stages', 'extract,fit'] + common)
        check('J10 stale fits refused after a new extract', rc == 2 and 'J10 stale fits' in txt, txt[-1500:])
        rc, txt = step(['jupiter', 'measure', '--stages', 'fit', '--fresh', '--supersede', 't1'] + common)
        check('--fresh --supersede moves the fits aside and fits again', rc == 0 and
              os.path.exists(os.path.join(od, '_superseded', 't1', 'MANIFEST.csv')), txt[-1500:])
        # corrupted products fail their class tests
        corrupt_checks(od, work)
        serve_phase1(od)


def serve_phase1(od):
    """The 8800 Phase-1 page server (exopipe_ops/jupiter_server.py, in place) on a parity port over a copy of the synthetic
    products: the served test passes (index = the ALL-PASS cells, night pages, slider maximum by the M33 rule, PNGs decode);
    an assign table changed on disk after serving (one more 10-s integration) fails the slider check; exact-pid stop."""
    from exopipe.phase3.lib.jupiter import serve as S
    off = 1000
    port = CFG.jupiter_serve.PORTS['phase1'] + off
    if S.listeners(port):
        print(f'skip phase1 serve test: port {port} in use')
        return
    out = os.path.join(TMP, 'serve1')
    shutil.copytree(od, os.path.join(out, 'products', CELL), ignore=shutil.ignore_patterns('_superseded'))
    rc, txt = step(['jupiter', 'serve', '--stages', 'phase1', '--port-offset', str(off), '--out', out])
    rec = os.path.join(out, '_served', f'jupiter_{port}.json')
    try:
        tests = [x for r in records(out).values() for x in r['tests']]
        check('phase1 server started on the parity port; served test passes (slider = M33 integrations, PNGs decode)',
              rc == 0 and tests and all(x['passed'] for x in tests), txt[-1500:])
        ap = os.path.join(out, 'products', CELL, 'assign_10s.csv')
        rows = list(csv.DictReader(open(ap)))
        rows[-1]['t_unix'] = '%.1f' % (float(rows[-1]['t_unix']) + 100.0)      # a 13th integration the server never showed
        with open(ap, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        r = validate.run('jupiter_served_page', rec)
        check('served page whose slider disagrees with the M33 integrations fails jupiter_served_page',
              not r[0]['passed'] and 'slider max 11' in r[0]['detail'] and '13 integrations' in r[0]['detail'], r[0]['detail'])
    finally:
        rc, txt = step(['jupiter', 'serve', '--stages', 'phase1', '--port-offset', str(off), '--out', out, '--stop'])
    check('phase1 server: exact-pid stop; port free', rc == 0 and not S.listeners(port), txt[-500:])


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
    # cutouts: the stored pixels around Jupiter of one frame changed -> the bilinear value no longer equals the meas value
    p = os.path.join(tmp, 'cutouts_10s_41MHz_I.npz')
    z = dict(np.load(p, allow_pickle=False))
    r = validate.run('jupiter_cutouts', p)
    check('copied cutouts pass jupiter_cutouts', r[0]['passed'], r[0]['detail'])
    cut = z['cut'].copy()
    H = cut.shape[1] // 2
    cut[4, H - 2:H + 3, H - 2:H + 3] += np.float16(5.0)
    z['cut'] = cut
    np.savez_compressed(p, **z)
    r = validate.run('jupiter_cutouts', p)
    check('changed cutout pixels at Jupiter fail jupiter_cutouts', not r[0]['passed'] and '1 disagree' in r[0]['detail'],
          r[0]['detail'])
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
    vtxt = open(os.path.join(state, vf[0])).read() if len(vf) == 1 else ''
    check('verify through the real tool on an empty tree: VERIFY OK, class test passes', rc == 0 and len(vf) == 1 and
          'VERIFY OK: 0 subband directories' in vtxt, txt[-800:])
    check('the VERIFY file holds the tool output only (no runner header), as the launcher tee', '=====' not in vtxt
          and vtxt.splitlines()[0].startswith('VERIFY'), vtxt[:300])
    # negative: a VERIFY file with a FAIL line, or with counts other than the verify unit read, fails its class test
    bad = os.path.join(TMP, 'VERIFY_bad.txt')
    real = open(os.path.join(CFG.jupiter_flag.STATE_DIR, 'VERIFY_20261007T111100Z.txt')).read()
    open(bad, 'w').write('FAIL 03h/2025-01-05/Run_x/18MHz: sha256 differs\n' + real)
    r = validate.run('jupiter_flag_verify', bad)
    check('VERIFY file with a FAIL line fails jupiter_flag_verify', not r[0]['passed'] and '1 FAIL lines' in r[0]['detail'],
          r[0]['detail'])
    open(bad, 'w').write(real)
    r = validate.run('jupiter_flag_verify', bad, expected={'dirs': 2581, 'jupiter_rows': 2117})
    check('VERIFY counts that differ from the verify unit fail jupiter_flag_verify', not r[0]['passed'] and 'differ' in
          r[0]['detail'], r[0]['detail'])
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
    import re
    txt = open(os.path.join(CFG.jupiter.V2_CODE_ROOT, 'integrate', 'run_integration.sh')).read()
    notes = dict(re.findall(r'^NOTE\[(\w+)\]="(.*)"$', txt, re.M))
    check('jupiter_v2.ROOT_NOTES = run_integration.sh NOTE[...] verbatim (ledger 1973)', notes == dict(CFG.jupiter_v2.ROOT_NOTES),
          sorted(set(notes) ^ set(CFG.jupiter_v2.ROOT_NOTES)))


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
    import csv as _csv
    pre_untied = 'sys_source' not in (next(_csv.reader(open(f'{REAL}/assign_1min.csv'))) or [])
    if pre_untied:
        # made before clv15 493d860 (10-s / 1-min fits take the untied systematic; no sys_source column): today's sigma model
        # differs in every band of a detected row - a stale product, correctly failed; nothing else may fail
        check('real assign_1min (pre-493d860) fails only on the sigma model', not r[0]['passed'] and
              "problems: sigma model" in r[0]['detail'] and 'assignment' not in r[0]['detail']
              and 'identity' not in r[0]['detail'] and 'fit fields' not in r[0]['detail'], r[0]['detail'])
    else:
        check('real assign_1min (made with the untied systematic) passes', r[0]['passed'], r[0]['detail'])
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
    # the V2 classes on the published 05h_2024-12-29 products (the fast ones; renders, quarantine and the masters tree are
    # re-hashed in 1-12 min on Lustre - checked by the review run, plan/impl/jupiter.md section 4)
    for cls, rel in (('jupiter_geom_raw', 'nightgeom/horizons_2024-12-29/manifest.json'),
                     ('jupiter_geom', f'nightgeom/{REAL_CELL}/CONTENT_TEST.txt'),
                     ('jupiter_v2_root', f'masters/derived/V2/{REAL_CELL}/CONTENT_TEST.txt'),
                     ('jupiter_blue_anchor', f'anchor/{REAL_CELL}/V2/MANIFEST.sha256'),
                     ('jupiter_refcheck', f'integrate/inputs/ref_offsets_{REAL_CELL}.csv'),
                     ('jupiter_metrics', f'integrate/{REAL_CELL}/CONTENT_TEST_metrics.txt'),
                     ('jupiter_site_summary', f'site/{REAL_CELL}/CONTENT_TEST.txt'),
                     ('jupiter_disclosure', 'integrate/occurrence_disclosed_failures.json')):
        r = validate.run(cls, os.path.join(V2, rel))
        check(f'real {cls} ({rel})', all(x['passed'] for x in r), [x['detail'] for x in r])


V2 = '/lustre/gh/main/phase3/jupiter/v2'


def _rewrite(path, fn):
    s = open(path).read()
    t = fn(s)
    assert t != s, f'corruption of {path} changed nothing'
    open(path, 'w').write(t)


def test_class_negatives():
    """Each class test fails on a corrupted COPY of a product (real products copied to /tmp, or a minimal synthetic layout
    where the real product is large): ephemeris, window, the V2 and occurrence classes."""
    import tarfile
    base = os.path.join(TMP, 'neg')
    os.makedirs(base)

    def neg(cls, path, why, **ctx):
        r = validate.run(cls, path, **ctx)
        check(f'{why} fails {cls}', not all(x['passed'] for x in r), [x['detail'] for x in r])

    def pos(cls, path, why, **ctx):
        r = validate.run(cls, path, **ctx)
        check(f'{why} passes {cls}', all(x['passed'] for x in r), [x['detail'] for x in r])

    # ephemeris: an angle outside [0, 360) / rows cut so the window is not covered
    e = os.path.join(base, 'ephem', 'jupiter_ephem_2024-12-29.csv')
    os.makedirs(os.path.dirname(e))
    shutil.copy('/lustre/gh/main/phase3/jupiter/ephem/jupiter_ephem_2024-12-29.csv', e)
    pos('jupiter_ephem', e, 'copied ephemeris')
    lines = open(e).read().splitlines()
    i = lines[0].split(',').index('cml_earth')
    f = lines[5].split(',')
    f[i] = '400.0'
    lines[5] = ','.join(f)
    open(e, 'w').write('\n'.join(lines) + '\n')
    neg('jupiter_ephem', e, 'an ephemeris angle of 400 deg')
    open(e, 'w').write('\n'.join(lines[:1] + [lines[1]] + lines[200:]) + '\n')
    neg('jupiter_ephem', e, 'an ephemeris with a gap')
    # window: a duplicated cell / a cell below 30 deg
    w = os.path.join(base, 'nights.csv')
    real = open(CFG.jupiter.WINDOW_CSV).read().splitlines()
    open(w, 'w').write('\n'.join(real + real[1:2]) + '\n')
    neg('jupiter_window', w, 'a window with a duplicated cell')
    f = real[1].split(',')
    f[4] = '29.9'
    open(w, 'w').write('\n'.join(real[:1] + [','.join(f)] + real[2:]) + '\n')
    neg('jupiter_window', w, 'a window cell at 29.9 deg')
    # a backfill copy directory (the tool's own verify_tree in place): one copied image byte changed
    src = '/lustre/gh/main/phase3/jupiter/phase1_flagged/07h/2025-01-08/Run_20260506_012543/23MHz'
    pos('jupiter_flagged_copy', src, 'a real backfill copy directory (read only)')
    cd = os.path.join(base, 'flagged', *src.split(os.sep)[-4:])
    shutil.copytree(src, cd)
    pos('jupiter_flagged_copy', cd, 'its copy')
    img = next(os.path.join(cd, x) for x in sorted(os.listdir(cd)) if x.endswith('-image.fits.fs'))
    b = bytearray(open(img, 'rb').read())
    b[-100] ^= 0xFF
    open(img, 'wb').write(bytes(b))
    neg('jupiter_flagged_copy', cd, 'a copied pilot image changed')
    # raw Horizons fetch: one table changed
    hz = os.path.join(base, 'horizons')
    shutil.copytree(os.path.join(V2, 'nightgeom', 'horizons_2024-12-29'), hz)
    pos('jupiter_geom_raw', os.path.join(hz, 'manifest.json'), 'copied Horizons fetch')
    t = sorted(x for x in os.listdir(hz) if x.endswith('.txt'))[0]
    _rewrite(os.path.join(hz, t), lambda s: s.replace('.', ',', 1))
    neg('jupiter_geom_raw', os.path.join(hz, 'manifest.json'), 'a changed raw Horizons table')
    # nightgeom: a geometry table rewritten after its verdict / a verdict whose last line disagrees
    ng = os.path.join(base, 'nightgeom', REAL_CELL)
    shutil.copytree(os.path.join(V2, 'nightgeom', REAL_CELL), ng)
    os.utime(os.path.join(ng, 'CONTENT_TEST.txt'), (time.time() - 100, time.time() - 100))
    for x in os.listdir(ng):
        if x != 'CONTENT_TEST.txt':
            os.utime(os.path.join(ng, x), (time.time() - 200, time.time() - 200))
    pos('jupiter_geom', os.path.join(ng, 'CONTENT_TEST.txt'), 'copied nightgeom cell')
    os.utime(os.path.join(ng, 'geom_10s.csv'))
    neg('jupiter_geom', os.path.join(ng, 'CONTENT_TEST.txt'), 'a geometry table newer than its verdict')
    # derived root: one assign row dropped
    dr = os.path.join(base, 'derived', REAL_CELL)
    shutil.copytree(os.path.join(V2, 'masters', 'derived', 'V2', REAL_CELL), dr, symlinks=True)
    pos('jupiter_v2_root', os.path.join(dr, 'CONTENT_TEST.txt'), 'copied V2 derived root')
    _rewrite(os.path.join(dr, 'assign_10min.csv'), lambda s: '\n'.join(s.splitlines()[:-1]) + '\n')
    neg('jupiter_v2_root', os.path.join(dr, 'CONTENT_TEST.txt'), 'a derived root with an assign row missing')
    # metrics: one metrics file unreadable
    md = os.path.join(base, 'integrate', REAL_CELL)
    shutil.copytree(os.path.join(V2, 'integrate', REAL_CELL), md)
    now = time.time()
    for x in os.listdir(md):
        os.utime(os.path.join(md, x), (now - 300, now - 300))
    os.utime(os.path.join(md, 'CONTENT_TEST_metrics.txt'), (now - 100, now - 100))
    pos('jupiter_metrics', os.path.join(md, 'CONTENT_TEST_metrics.txt'), 'copied metrics')
    _rewrite(os.path.join(md, 'metrics_V2.json'), lambda s: s[:len(s) // 2])
    neg('jupiter_metrics', os.path.join(md, 'CONTENT_TEST_metrics.txt'), 'a truncated metrics_V2.json')
    # site summary: freshness line failed / a failure outside the accepted set named
    ss = os.path.join(base, 'site_CONTENT_TEST.txt')
    shutil.copy(os.path.join(V2, 'site', REAL_CELL, 'CONTENT_TEST.txt'), ss)
    pos('jupiter_site_summary', ss, 'copied site summary')
    _rewrite(ss, lambda s: s.replace('FRESHNESS -> PASS', 'FRESHNESS -> FAIL'))
    neg('jupiter_site_summary', ss, 'a summary whose freshness failed')
    shutil.copy(os.path.join(V2, 'site', REAL_CELL, 'CONTENT_TEST.txt'), ss)
    _rewrite(ss, lambda s: s.replace('L1.IoD', 'L1.IoX'))
    neg('jupiter_site_summary', ss, 'a summary naming a failure outside the accepted set')
    # occurrence results + disclosure record: a failing value changed in the live results
    occ = os.path.join(base, 'v2', 'occurrence', 'validation', 'content_test_results.json')
    dis = os.path.join(base, 'v2', CFG.jupiter_occurrence.DISCLOSURE)
    os.makedirs(os.path.dirname(occ))
    os.makedirs(os.path.dirname(dis))
    shutil.copy(os.path.join(V2, 'occurrence', 'validation', 'content_test_results.json'), occ)
    d = json.load(open(os.path.join(V2, CFG.jupiter_occurrence.DISCLOSURE)))
    d['source'] = occ
    json.dump(d, open(dis, 'w'))
    pos('jupiter_occurrence', occ, 'copied occurrence results')
    pos('jupiter_disclosure', dis, 'copied disclosure record')
    res = json.load(open(occ))
    x = next(r for r in res['results'] if r.get('gating') and not r.get('passed'))
    x['value'] = (x['value'] + 1.0) if isinstance(x.get('value'), (int, float)) else 'changed'
    json.dump(res, open(occ, 'w'))
    neg('jupiter_occurrence', occ, 'an accepted failure whose live value changed')
    neg('jupiter_disclosure', dis, 'a disclosure record whose live value changed')
    res = json.load(open(os.path.join(V2, 'occurrence', 'validation', 'content_test_results.json')))
    next(r for r in res['results'] if r.get('gating') and r.get('passed'))['passed'] = False
    json.dump(res, open(occ, 'w'))
    neg('jupiter_occurrence', occ, 'a fourth failing gating test')
    # minimal synthetic layouts for the large V2 products (masters tree 710 files, Blue anchor 55 MB, crop archives 170 MB,
    # renders 898 files per root, quarantine 43,086 files)
    m = os.path.join(base, 'masters')
    os.makedirs(os.path.join(m, 'V2', REAL_CELL))
    open(os.path.join(m, 'V2', REAL_CELL, 'meas_10s.csv'), 'w').write('a\n1\n')
    os.utime(os.path.join(m, 'V2', REAL_CELL, 'meas_10s.csv'), (now - 300, now - 300))
    open(os.path.join(m, f'CONTENT_TEST_{REAL_CELL}.txt'), 'w').write('T1 ... -> PASS\nALL-PASS\n')
    pos('jupiter_v2_masters', os.path.join(m, f'CONTENT_TEST_{REAL_CELL}.txt'), 'synthetic masters tree')
    os.utime(os.path.join(m, 'V2', REAL_CELL, 'meas_10s.csv'), (time.time() + 100, time.time() + 100))
    neg('jupiter_v2_masters', os.path.join(m, f'CONTENT_TEST_{REAL_CELL}.txt'), 'a variant file newer than the masters verdict')
    open(os.path.join(m, f'CONTENT_TEST_{REAL_CELL}.txt'), 'w').write('T1 ... -> FAIL\nALL-PASS\n')
    neg('jupiter_v2_masters', os.path.join(m, f'CONTENT_TEST_{REAL_CELL}.txt'), 'a masters verdict with a FAIL line')
    import hashlib

    def manifest_dir(d, names, verdict):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'MANIFEST.sha256'), 'w') as fh:
            for nm in names:
                p = os.path.join(d, nm)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, 'w').write(f'content of {nm}\n')
                fh.write(f'{hashlib.sha256(open(p, "rb").read()).hexdigest()}  {nm}\n')
        if verdict:
            open(verdict, 'w').write('B1 ... -> PASS\nALL-PASS\n')
            os.utime(verdict, (time.time() + 5, time.time() + 5))
    an = os.path.join(base, 'anchor', REAL_CELL, 'V2')
    manifest_dir(an, ['blue_10s.csv', 'blue_1min.csv'], os.path.join(base, 'anchor', REAL_CELL, 'CONTENT_TEST_blue_V2.txt'))
    pos('jupiter_blue_anchor', os.path.join(an, 'MANIFEST.sha256'), 'synthetic Blue anchor')
    open(os.path.join(an, 'blue_10s.csv'), 'a').write('x\n')
    os.utime(os.path.join(an, 'blue_10s.csv'), (now - 300, now - 300))
    neg('jupiter_blue_anchor', os.path.join(an, 'MANIFEST.sha256'), 'a Blue anchor member changed after its manifest')
    rd = os.path.join(base, 'renders', 'V2.r20990101T000000Z')
    manifest_dir(rd, ['10s/panel_a.png', 'index.json'], None)
    open(os.path.join(rd, 'RENDER_TEST.txt'), 'w').write('T1 -> PASS\nALL-PASS\n')
    pos('jupiter_site_render', os.path.join(rd, 'MANIFEST.sha256'), 'synthetic render')
    os.remove(os.path.join(rd, '10s', 'panel_a.png'))
    neg('jupiter_site_render', os.path.join(rd, 'MANIFEST.sha256'), 'a render with a manifest member missing')
    q = os.path.join(base, 'quarantine')
    manifest_dir(q, ['renders/V0.r1/a.png', 'renders/V1.r2/b.png'], None)
    open(os.path.join(q, 'DELETED_x_list.txt'), 'w').write('V1.r2\n')       # one render directory name per line, as the real list
    shutil.rmtree(os.path.join(q, 'renders', 'V1.r2'))
    pos('jupiter_render_quarantine', os.path.join(q, 'MANIFEST.sha256'), 'synthetic quarantine (a recorded deletion)')
    open(os.path.join(q, 'renders', 'V0.r1', 'a.png'), 'a').write('x')
    neg('jupiter_render_quarantine', os.path.join(q, 'MANIFEST.sha256'), 'a quarantined file changed')
    pv = os.path.join(base, 'provenance')
    os.makedirs(pv)
    src = os.path.join(pv, 'c1.fits')
    open(src, 'wb').write(b'crop bytes')
    with tarfile.open(os.path.join(pv, 'crops_V2.tar'), 'w') as tf:
        tf.add(src, arcname='c1.fits')
    open(os.path.join(pv, 'crops_V2.csv'), 'w').write(
        f'member,bytes,sha256\nc1.fits,10,{hashlib.sha256(b"crop bytes").hexdigest()}\n')
    open(os.path.join(pv, 'CONTENT_TEST_crops_V2.txt'), 'w').write('TRANSFER -> PASS\nALL-PASS\n')
    pos('jupiter_crop_archive', os.path.join(pv, 'crops_V2.tar'), 'synthetic crop archive')
    open(os.path.join(pv, 'crops_V2.csv'), 'w').write('member,bytes,sha256\nc1.fits,10,' + '0' * 64 + '\n')
    neg('jupiter_crop_archive', os.path.join(pv, 'crops_V2.tar'), 'a crop archive member whose sha256 differs from its row')
    rc_ = os.path.join(base, 'inputs')
    os.makedirs(rc_)
    ro = os.path.join(rc_, f'ref_offsets_{REAL_CELL}.csv')
    for name in (f'ref_offsets_{REAL_CELL}.csv', f'window_summary_{REAL_CELL}.csv'):
        shutil.copy(os.path.join(V2, 'integrate', 'inputs', name), rc_)
    pos('jupiter_refcheck', ro, 'copied reference-tie tables')
    rows = list(csv.DictReader(open(ro)))
    rows[0]['dx'] = '%.3f' % (float(rows[0]['dx']) + 9.0)        # one reference offset changed: its window's medians move
    with open(ro, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    neg('jupiter_refcheck', ro, 'a reference offset that no longer gives the window summary')
    open(os.path.join(rc_, f'window_summary_{REAL_CELL}.csv'), 'w').write(
        open(os.path.join(V2, 'integrate', 'inputs', f'window_summary_{REAL_CELL}.csv')).readline())
    neg('jupiter_refcheck', ro, 'a reference-tie window summary without rows')


def main():
    try:
        for t in (test_config_and_mirrors, test_gates, test_flag, test_serve, test_dry_runs_real, test_real_products_readonly,
                  test_class_negatives, test_end2end):
            print(f'--- {t.__name__}')
            t()
    finally:
        CFG.reset()
        shutil.rmtree(TMP, ignore_errors=True)
    print(f'\n{sum(ok for _, ok in RESULTS)} of {len(RESULTS)} checks passed in {time.time() - T0:.0f} s')


if __name__ == '__main__':
    main()
