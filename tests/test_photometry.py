"""Standing test of `exopipe3 photometry` (plain asserts, no pytest):

    NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py \\
        nice -n 10 /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_photometry.py

Synthetic data in a fresh directory under /tmp/exopipe3_tests_photometry/ (removed at the end). Production code and
products are only READ.

- mirrors: every MIRROR value of the config equals the in-place engine's literal (ast / source text of
  deep_forced_phot.py, deep_forced_frames.py, deep_forced_eval.py, deep_forced_radio.py, snapshot_forced_phot.py,
  snapshot_forced_eval.py, gate_inputs.py, phase3_snapshot_cutouts.py, phase3_snapshot_astro.py, phase3_tier_rules.py);
  the product formats the content tests read (forced, radio, entry, blank, admission columns) = the engine's header rows;
  the in-place fitter's GAUSS_SNR / RADIO_SNR_FLOOR / deep K = config fitting.
- ports: the content tests' own measurement code equals the production functions (deep_forced_phot.measure and
  snapshot_forced_phot.measure, extracted from the engine source and run here); the zone recomputation (Z1) equals
  test_snapshot_forced_reserve.zones_cell on real cells (read only).
- gates: PHOT-ENV, PHOT-SUBLST-NO-ENGINE, PHOT-FULL, PHOT-SAMPLE-PIN, protected-key overrides, F-SRCRUN, F-PARENTDIR,
  F-NOFRAMES (exit 3) vs F-PARENTS (exit 2) when the frame-list engine writes no row, stage-name errors.
- deep end to end (FAKE frames and measure engines reached through the step's engine seam - the fake measure engine
  computes values with the production measure function on synthetic frames - and the REAL deep_forced_eval.py in place):
  frames -> stage (not needed) -> measure (2 of 4 shards own cells) -> evaluate in ONE invocation, every content test
  passes; the per-hour form (--epochs, Stokes V noise gate over the full night list as an item refusal); M-CHAIN (frame
  list changed, a staged list of another frame list ignored, a missing shard record, stray forced / blank files);
  --resume (skipped in a clean tree; a record of a dirty tree is never resumed).
- snapshot end to end (REAL snapshot_forced_phot.py and snapshot_forced_eval.py in place under the site py36 interpreter)
  on synthetic cutouts of real current targets (cell 18h_2025-04-13, which has no search outputs): measure -> evaluate
  sb + wb -> zone flags, every content test passes; SM-NOCUT (exit 3), SM-SUBSET, SM-STALE (--fresh moves the stale file
  to a quarantine directory with a sha256 manifest, nothing deleted), SE-LISTED, SE-DONE, SE-NONE; --resume;
  --refused-cells wb=<the 2026-10-07 production list> passed to the engine as written (refused_cells.csv byte-identical to
  production's) and a list omitting a pinned cell refused; --survey-cells as a recorded config override, refused (RULED)
  into the production root.
- dry runs: the multi-stage dry run plans evaluate from the measure stage of the same invocation (no 'WOULD REFUSE
  M-CHAIN'); evaluate alone refuses (exit 2); nothing written; one dry run on a real deep cell (where /data07 is readable)
  and on the real snapshot cell 18h_2025-04-11 (widebands).
- negative: each content test of each product class fails on a corrupted copy (E4_plot is a display product that never
  gates: no negative).
- real products (read only): the SERVED generation-5 stage-1 npz and manifest of 18h_2025-04-11, the 2026-10-07 wideband
  evaluation (grid, rows, stage 1 on 3 cells; V1 / Z1 reported), the served eval_gen5 / eval_wb_gen5 (every test, 3 cells)
  and zone flags, the 2026-10-07 zone-flag file, the 2026-10-07 per-kind refused-cell lists (= the step's), the deep run3 frame
  list and stage list where readable (pre-PB-step-2: FL3 fails as expected).
- served generation (read only): every served snapshot forced page was built on photometry.CUTOUT_ROOT and
  <OUT_ROOT[snapshot]>/STAGE1_DIR (its FT-START record), forced_snapshot names the same, the served stage 1 and tier forced
  manifests measured those cutouts, the tier / planets pages read photometry.TIER_FORCED_ROOTS; falsified with generation 4.
- cutouts: mirrors of the regeneration engines, the tier tools and the regeneration content test; ports (the writer's
  J-tag naming, one frame per integration, the tool's box, rrms); identity on synthetic cubes (subband: shifted plane and a
  5-sigma source fail; wideband: the weights' '%.6g' rounding passes the exact bound 5e-6 x (A + |S|), fails without it and
  with the stack-magnitude bound); one file per key (stale time stamp, orphan; CM2); the snapshot cutouts stage end to end
  through a FAKE engine running the REAL writer on synthetic consolidated cubes (done, manifest, --resume, in-place re-run,
  C-STALE and --fresh quarantine, C-IDENTITY before placement, C-EXCLUDED, C-CELLS, C-ROOT); the deep and 10-minute tier
  cutouts (the REAL tool's cell_job on synthetic residuals) and cutmeasure (the REAL tier forced tool) end to end; ad-hoc
  target lists (the REAL stage 1 on ADHOC cutouts; PHOT-ADHOC gates); stage selection (cutouts only when named; sub-LST).
"""
import ast
import contextlib
import csv
import glob
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

import numpy as np                                    # noqa: E402
import warnings                                       # noqa: E402
from astropy.wcs import FITSFixedWarning              # noqa: E402

warnings.simplefilter('ignore', FITSFixedWarning)     # synthetic frames carry DATE-OBS only (astropy fills MJD-OBS)

from exopipe.phase3 import cli                        # noqa: E402
from exopipe.phase3.config import CFG                 # noqa: E402
from exopipe.phase3.core import env, record, validate  # noqa: E402

env.legacy()
from exopipe.phase3.lib.photometry import steps_common as K      # noqa: E402
from exopipe.phase3.lib.photometry import steps_deep as KD       # noqa: E402
from exopipe.phase3.lib.photometry import steps_snapshot as KS   # noqa: E402
from exopipe.phase3.lib.photometry import tests as T             # noqa: E402

BASE = '/tmp/exopipe3_tests_photometry'
os.makedirs(BASE, exist_ok=True)
TMP = tempfile.mkdtemp(prefix='run_', dir=BASE)
OPS = '/lustre/gh/main/exopipe_ops'
CLV = '/lustre/gh/main/clv15'
REAL_SNAP_CELL = '18h_2025-04-11'
SYN_SNAP_CELL = '18h_2025-04-13'           # no search output, no exclusion, no tier-3 pin row (checked below)
REAL_STAGE1 = os.path.join(CFG.photometry.OUT_ROOT['snapshot'], CFG.photometry.STAGE1_DIR)   # the served generation's
S1 = CFG.photometry.STAGE1_DIR
T0 = time.time()
RESULTS = []
for _k in list(os.environ):                 # PHOT-ENV: the caller environment must not carry engine switches
    if _k.startswith(tuple(CFG.photometry.ENV_FORBIDDEN_PREFIXES)):
        del os.environ[_k]


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond)))
    print(f"{'ok  ' if cond else 'FAIL'} {name}" + (f'  [{str(detail)[-1500:]}]' if detail and not cond else ''), flush=True)
    assert cond, f'{name}: {detail}'


def step(argv, quiet=True):
    """`exopipe3 <argv>` in this process; returns (rc, stdout+stderr)."""
    CFG.reset()
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        rc = cli.main(list(argv))
    txt = buf.getvalue()
    if not quiet:
        print(txt)
    return rc, txt


def records(out):
    d = os.path.join(out, '_records', 'photometry')
    return {f[:-5]: json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith('.json')} \
        if os.path.isdir(d) else {}


@contextlib.contextmanager
def patched(obj, **kw):
    old = {k: getattr(obj, k) for k in kw}
    for k, v in kw.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def run_tests(cls, path, **ctx):
    T._CACHE.clear()
    return {r['name'].split('.', 1)[1]: r for r in validate.run(cls, path, **ctx)}


def all_pass(cls, path, **ctx):
    res = run_tests(cls, path, **ctx)
    bad = {k: r['detail'] for k, r in res.items() if not r['passed']}
    return not bad, bad


# ---------------------------------------------------------------------------------------------------------------------
# engine seam: the step resolves each in-place engine through steps_common.engine; the test maps some keys to fakes
# ---------------------------------------------------------------------------------------------------------------------

REAL_ENGINE = K.engine
FAKES = {}
K.engine = lambda key: FAKES.get(key) or REAL_ENGINE(key)
FAKE_DIR = os.path.join(TMP, 'fake_engines')
os.makedirs(FAKE_DIR)
PARAMS = os.path.join(FAKE_DIR, 'fake_params.json')

FAKE_FRAMES = r'''#!/usr/bin/env python3
"""FAKE deep_forced_frames.py for tests/test_photometry.py: the production logic (deep_forced_frames.py:17-61) over the
test's cell roots (fake_params.json) instead of the pinned roots.csv; parents = presence of the catalogue + .gen.json."""
import csv, glob, json, os, re, sys
from astropy.io import fits
P = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fake_params.json')))
OUT = sys.argv[1]; ONLY = set(sys.argv[2:])
U = os.environ.get('DEEP_FORCED_SRC_RUN') or sys.exit('REFUSED: set DEEP_FORCED_SRC_RUN=<the searched deep run dir>')
PAR = os.environ['P3_PARENT_DIR']
rx = re.compile(r'^(\d{4}-\d\d-\d\d)_([IV])_redo_resid_(\w+)\.fits$')
rows, refused = [], []
for cell in sorted(P['roots']):
    if ONLY and cell not in ONLY:
        continue
    lst, band = cell.split('_', 1)
    cf = '%s/%s/coadd_final' % (P['roots'][cell], cell)
    pb, wt = '%s/PB_deep_%s.fits' % (cf, band), '%s/weights.csv' % cf
    par = '%s/master_R075_%s_%s.gaul.csv' % (PAR, lst, band)
    if not (os.path.exists(par) and os.path.exists(par + '.gen.json')):
        refused.append((cell, 'parents: %s %s: no Robust -0.75 parent catalogue at %s - REFUSING' % (lst, band, par))); continue
    if not os.path.exists(pb):
        refused.append((cell, 'no PB_deep map')); continue
    n0 = len(rows)
    for S, sub in (('I', 'iresid'), ('V', 'vresid')):
        for p in sorted(glob.glob('%s/%s/%s/*_%s_redo_resid_%s.fits' % (U, sub, cell, S, band))):
            m = rx.match(os.path.basename(p))
            if not m:
                continue
            rp = os.path.realpath(p)
            tn = str(fits.getheader(rp).get('THERMNOI') or '') if S == 'V' else ''
            pbS = pb if S == 'V' else pb.replace('PB_deep_%s.fits' % band, 'PB_deep_R075_%s.fits' % band)
            if not os.path.exists(pbS):
                refused.append((cell, 'no %s map' % os.path.basename(pbS))); continue
            rows.append(dict(lst=lst, band=band, date=m.group(1), stokes=S, path=rp, pb_path=pbS, parent_path=par,
                             weights_path=wt if os.path.exists(wt) else '', thermnoi=tn))
    if len(rows) == n0:
        refused.append((cell, 'no search frames'))
with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('%d frames -> %s' % (len(rows), OUT))
for c, why in refused:
    print('  REFUSED %s: %s' % (c, why), file=sys.stderr)
'''

FAKE_MEASURE = r'''#!/usr/bin/env python3
"""FAKE deep_forced_phot.py for tests/test_photometry.py: the production file layout (deep_forced_phot.py:197-323) with the
PRODUCTION measure() function (extracted from the engine source) on synthetic frames; targets, refusals and exclusions
from fake_params.json; flags constant; radio positions 'template' at |S/N| >= 4."""
import ast, csv, gzip, json, math, os, sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
P = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fake_params.json')))
ns = {'np': np, 'map_coordinates': map_coordinates}
for node in ast.parse(open(P['deep_phot_engine']).read()).body:
    if isinstance(node, ast.FunctionDef) and node.name == 'measure':
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'deep_forced_phot.measure', 'exec'), ns)
measure = ns['measure']
FRAMES, SHARD, NSH, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.makedirs(OUT, exist_ok=True)
frames = list(csv.DictReader(open(FRAMES)))
cells = sorted({(r['lst'], r['band']) for r in frames})
mycells = [c for i, c in enumerate(cells) if i % NSH == SHARD]
FO = gzip.open('%s/forced_%03d.csv.gz' % (OUT, SHARD), 'wt', newline=''); W = csv.writer(FO); W.writerow(P['columns'])
BO = open('%s/blank_%03d.csv' % (OUT, SHARD), 'w', newline=''); BW = csv.writer(BO)
BW.writerow(['lst', 'band', 'date', 'stokes', 'n', 'ge3', 'ge4', 'ge5', 'ge6', 'ge7'])
FA = open('%s/frames_%03d.csv' % (OUT, SHARD), 'w', newline=''); FW = csv.writer(FA)
FW.writerow(['lst', 'band', 'date', 'stokes', 'path', 'admitted', 'reason', 'n_targets', 'n_controls'])
rng = np.random.default_rng(20260928 + SHARD)
NR = len(P['radio_columns'])
def load(p):
    with fits.open(p, memmap=False) as h:
        return np.squeeze(h[0].data).astype(np.float64), WCS(h[0].header).celestial, h[0].header
for (lst, band) in mycells:
    cf = [r for r in frames if (r['lst'], r['band']) == (lst, band)]
    pbd, pbw, _ = load(cf[0]['pb_path'])
    tn = {r['date']: float(r['thermnoi']) for r in cf if r['stokes'] == 'V' and r['thermnoi'] not in ('', '0', '0.0')}
    badV = {d_ for d_, v in tn.items() if len(tn) >= 5 and v > 3 * float(np.median(list(tn.values())))}
    for r in cf:
        S, date, path = r['stokes'], r['date'], r['path']
        why = ('excluded epoch (data_decisions)' if '%s_%s_%s' % (lst, band, date) in P['excluded'] else
               '3x per-Stokes noise gate' if (S == 'V' and date in badV) else '')
        if why:
            FW.writerow([lst, band, date, S, path, 0, why, 0, 0]); continue
        d, w, h = load(path)
        bmaj = float(h['BMAJ']); mjd = float(h['MJD-MID'])
        nt = nc = 0
        for tag, nm, ra, dec in P['targets']:
            m = measure(d, w, np.array([ra]), np.array([dec]))[0]
            if not (np.isfinite(m[0]) and m[1] > 0):
                continue
            pbv = measure(pbd, pbw, np.array([ra]), np.array([dec]))[0, 0]
            snr = m[0] / m[1]
            rp = (['template', '%.5f' % ra, '%.5f' % dec, '60.0', '40.0', '10.0', '50.0', '30.0', '20.0', '%.2f' % snr,
                   '0.10', 'template fit (fake engine)'] if abs(snr) >= 4.0 else [''] * NR)
            W.writerow([tag, nm, lst, band, date, S, 'target', '', '%.5f' % ra, '%.5f' % dec, '%.6f' % mjd, '45.00',
                        '%.5g' % m[0], '%.5g' % m[1], '%.3f' % snr, '%.4f' % pbv,
                        '%.5g' % (m[0] / pbv) if pbv > 0.1 else '', 0, '', '', '50.00', '90.00', '90.00', '',
                        '', '', '', '', '', '', '', '%.3f' % (snr + 5), '%.3f' % (snr + 10), '0.1', '%.5f' % bmaj, 0]
                       + rp + ['%.5f' % ra, '%.5f' % dec])
            nt += 1
            th = np.radians(np.arange(12) * 30.0); rdeg = 4.0 * bmaj
            cdec = dec + rdeg * np.sin(th); cra = (ra + rdeg * np.cos(th) / max(math.cos(math.radians(dec)), 0.05)) % 360
            for c, (vv, rr) in enumerate(measure(d, w, cra, cdec)):
                if np.isfinite(vv) and rr > 0:
                    W.writerow([tag, nm, lst, band, date, S, 'control', c, '%.5f' % cra[c], '%.5f' % cdec[c],
                                '%.6f' % mjd, '', '%.5g' % vv, '%.5g' % rr, '%.3f' % (vv / rr), '', '', 0, '', '']
                               + [''] * 4 + [''] * 7 + ['', '', '', '', 0] + [''] * NR + ['%.5f' % ra, '%.5f' % dec])
                    nc += 1
        ny, nx = d.shape; ys, xs = rng.integers(30, ny - 30, 800), rng.integers(30, nx - 30, 800)
        sky = w.pixel_to_world_values(xs, ys); bra, bdec = np.asarray(sky[0]) % 360, np.asarray(sky[1])
        sel = np.where(np.isfinite(bra))[0][:200]
        mb = measure(d, w, bra[sel], bdec[sel]); sn = np.abs(mb[:, 0] / mb[:, 1]); sn = sn[np.isfinite(sn)]
        BW.writerow([lst, band, date, S, len(sn)] + [int((sn >= t).sum()) for t in (3, 4, 5, 6, 7)])
        FW.writerow([lst, band, date, S, path, 1, '', nt, nc])
FO.close(); BO.close(); FA.close()
print('DEEP-FORCED shard %d/%d: %d cells done -> %s' % (SHARD, NSH, len(mycells), OUT), flush=True)
'''


# ---------------------------------------------------------------------------------------------------------------------
# synthetic deep data
# ---------------------------------------------------------------------------------------------------------------------

DEEP = os.path.join(TMP, 'deep')
SRC = os.path.join(DEEP, 'transients', 'Run_T')
COADD = os.path.join(DEEP, 'coadd')
PARENTS = os.path.join(DEEP, 'parents')
RA0, DEC0 = 154.25, 49.8                 # frame centre (LST 10h), between the two real list targets below
NPIX, CD, BMAJ = 200, 0.05, 0.3           # pixels, deg/pixel, deg
DATES = ['2099-01-0%d' % i for i in range(1, 6)]
# REAL list entries (the 10 pc list prints 9 decimals: T4 must re-measure at the exact list position, not the row's 5)
DEEP_TARGETS = [('HWJ', 'HAT-P-22', 155.6815, 50.1287), ('10pc', 'Gaia DR3 823773494718931968', 152.8271039, 49.45058383)]
NOISY_V_DATE = DATES[4]                   # its THERMNOI is 10x the others: the Stokes V 3x noise gate refuses it
ROOTS = {'10h_46MHz': COADD, '10h_Green': COADD, '10h_Blue': COADD, '10h_82MHz': COADD}


def wcs_header(n=NPIX, cd=CD, ra=RA0, dec=DEC0):
    from astropy.io import fits
    h = fits.Header()
    h['CTYPE1'], h['CTYPE2'] = 'RA---SIN', 'DEC--SIN'
    h['CRVAL1'], h['CRVAL2'] = ra, dec
    h['CRPIX1'], h['CRPIX2'] = n / 2.0 + 0.5, n / 2.0 + 0.5
    h['CDELT1'], h['CDELT2'] = -cd, cd
    h['CUNIT1'], h['CUNIT2'] = 'deg', 'deg'
    return h


def make_deep(excluded):
    """Two cells with 5 nights x I,V; 10h_Blue has frames but no parent catalogue; 10h_82MHz has a parent but no frame;
    the HWJ target carries a 10-sigma source in Stokes I; one excluded (lst, band, date) of data_decisions adds a frame."""
    from astropy.io import fits
    from astropy.wcs import WCS
    rng = np.random.default_rng(7)
    w = WCS(wcs_header()).celestial
    yy, xx = np.mgrid[0:NPIX, 0:NPIX]
    for cell in sorted(ROOTS):
        lst, band = cell.split('_', 1)
        cf = os.path.join(COADD, cell, 'coadd_final')
        os.makedirs(cf)
        pb = np.full((NPIX, NPIX), 0.8, np.float32)
        pb[:, 110:] = 0.05                        # the 10 pc target sits where PB <= 0.1: flux_pbcorr blank there
        for name in ('PB_deep_%s.fits' % band, 'PB_deep_R075_%s.fits' % band):
            fits.writeto(os.path.join(cf, name), pb, wcs_header(), overwrite=True)
        master = rng.normal(0, 0.01, (NPIX, NPIX)).astype(np.float32)
        mp = os.path.join(cf, 'I_deep_Taper_R075_%s.fits' % band)
        fits.writeto(mp, master, wcs_header(), overwrite=True)
        with open(os.path.join(cf, 'weights.csv'), 'w') as f:
            f.write('date,qa3_rms_mjy,in_master\n' + ''.join('%s,10.0,0\n' % d for d in DATES))
        if cell != '10h_Blue':
            os.makedirs(PARENTS, exist_ok=True)
            pp = os.path.join(PARENTS, 'master_R075_%s_%s.gaul.csv' % (lst, band))
            with open(pp, 'w') as f:
                f.write('# Gaul_id, Island_id, Source_id, Wave_id, RA, E_RA, DEC, E_DEC, Total_flux, E_Total_flux, Peak_flux\n')
                f.write('0, 0, 0, 0, %.5f, 0.0, %.5f, 0.0, 2.0, 0.1, 1.5\n' % (RA0 + 4.0, DEC0 - 3.0))
            d32 = np.squeeze(fits.getdata(mp)).astype(np.float32)
            json.dump({'master': os.path.basename(mp),
                       'master_md5_data': hashlib.md5(np.ascontiguousarray(d32).tobytes()).hexdigest()},
                      open(pp + '.gen.json', 'w'))
        if cell == '10h_82MHz':
            continue
        dates = list(DATES) + ([excluded[2]] if excluded and cell == '%s_%s' % excluded[:2] else [])
        for S, sub in (('I', 'iresid'), ('V', 'vresid')):
            d_ = os.path.join(SRC, sub, cell)
            os.makedirs(d_, exist_ok=True)
            for k, date in enumerate(dates):
                if S == 'V' and date not in DATES:
                    continue
                img = rng.normal(0, 0.1, (NPIX, NPIX)).astype(np.float32)
                if S == 'I':
                    x, y = w.world_to_pixel_values(DEEP_TARGETS[0][2], DEEP_TARGETS[0][3])
                    img += (1.0 * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / (BMAJ / CD / 2.3548) ** 2)).astype(np.float32)
                h = wcs_header()
                h['BMAJ'], h['BMIN'], h['BPA'] = BMAJ, BMAJ * 0.8, 0.0
                h['DATE-OBS'], h['DATE-END'] = '%sT10:00:00' % date, '%sT11:00:00' % date
                h['MJD-MID'] = 51544.0 + k
                if S == 'V':
                    h['THERMNOI'] = 0.010 * (10 if date == NOISY_V_DATE else 1) * (1 + 0.01 * k)
                fits.writeto(os.path.join(d_, '%s_%s_redo_resid_%s.fits' % (date, S, band)), img, h, overwrite=True)


def write_params(excluded):
    json.dump({'roots': ROOTS, 'deep_phot_engine': os.path.join(OPS, 'deep_forced_phot.py'),
               'columns': list(T.FORCED_COLUMNS) + list(T.RADIO_COLUMNS) + list(T.ENTRY_COLUMNS),
               'radio_columns': list(T.RADIO_COLUMNS), 'targets': [list(t) for t in DEEP_TARGETS],
               'excluded': ['%s_%s_%s' % excluded] if excluded else []}, open(PARAMS, 'w'))
    for key, src in (('frames', FAKE_FRAMES), ('measure_deep', FAKE_MEASURE)):
        p = os.path.join(FAKE_DIR, CFG.photometry.ENGINE[key])
        open(p, 'w').write(src)
        FAKES[key] = p


def excluded_frame():
    """One (lst, band, date) that core.exclusions excludes for a test cell (data_decisions.csv read live), or None."""
    from exopipe.phase3.core import exclusions
    ex = exclusions.load()
    for cell in ('10h_46MHz', '10h_Green'):
        lst, band = cell.split('_', 1)
        nights = sorted(ex.excluded_nights(lst, band))
        if nights:
            return lst, band, nights[0]
    return None


DEEP_COMMON = ['--tier', 'deep', '--src-run', SRC, '--parent-dir', PARENTS, '--cores', 'none',
               '--set', 'paths.COADD_SB_ROOT=%s' % COADD]


# ---------------------------------------------------------------------------------------------------------------------
# synthetic snapshot cutouts (templates: real cutouts of the real cell, read only)
# ---------------------------------------------------------------------------------------------------------------------

SNAP = os.path.join(TMP, 'snap')
CUTS = os.path.join(SNAP, 'cutouts')
SNAP_BANDS = ('64MHz', '73MHz', 'Blue')
NF = 24                                     # frames per synthetic cutout (4 one-minute blocks)
CUT_ROOT_REAL = CFG.photometry.CUTOUT_ROOT       # the served generation's cutouts (templates, read only)
ATEAM = [(350.866, 58.815), (299.868, 40.734), (83.633, 22.014), (187.706, 12.391), (139.524, -12.096), (252.784, 4.993)]


def _elev_lst(ra, dec, lst_deg=277.5, lat=37.2398):
    ha = math.radians(lst_deg - ra)
    d, la = math.radians(dec), math.radians(lat)
    return math.degrees(math.asin(math.sin(d) * math.sin(la) + math.cos(d) * math.cos(la) * math.cos(ha)))


def _sep(a, b):
    ra1, d1, ra2, d2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    return math.degrees(math.acos(min(1.0, math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(ra1 - ra2))))


def pick_templates():
    """{(sample, band): object dir} - one CURRENT target per sample with I and V cutouts in every band of SNAP_BANDS for
    the real cell, high in the sky (>= 45 deg at the cell's LST) and >= 20 deg from every A-team source."""
    from astropy.io import fits
    cur = T.current_safe_names()
    out = {}
    for smp in CFG.photometry.SAMPLES_MEASURED:
        objs = None
        for b in SNAP_BANDS:
            have = {os.path.basename(os.path.dirname(p)) for p in
                    glob.glob(os.path.join(CUT_ROOT_REAL, smp, b, '*', '*_18h_20250411T*_I.fits'))}
            objs = have if objs is None else objs & have
        for o in sorted(objs or ()):
            if (smp, o) not in cur:
                continue
            p = sorted(glob.glob(os.path.join(CUT_ROOT_REAL, smp, SNAP_BANDS[0], o, '*_18h_20250411T*_I.fits')))[0]
            H = fits.getheader(p)
            pos = (float(H['TARGRA']), float(H['TARGDEC']))
            if _elev_lst(*pos) >= 45 and all(_sep(pos, a) >= 20 for a in ATEAM):
                out[smp] = o
                break
    return out


def make_snapshot(objs):
    """Synthetic cutouts of SYN_SNAP_CELL: the real cutout's header and TIMES (first NF frames, 2 days later), Gaussian
    noise at the frame's rms, an 8-sigma source at the HWJ target in Stokes I of every band; one HWJ cutout of a target
    no longer in the current list (the engine's 'skipped' row)."""
    from astropy.io import fits
    rng = np.random.default_rng(11)
    lst, date = SYN_SNAP_CELL.split('_', 1)
    for smp, o in sorted(objs.items()):
        for b in SNAP_BANDS:
            for S in ('I', 'V'):
                src = sorted(glob.glob(os.path.join(CUT_ROOT_REAL, smp, b, o, '*_18h_20250411T*_%s.fits' % S)))[0]
                with fits.open(src, memmap=True) as h:
                    H = h[0].header.copy()
                    tab = h['TIMES'].data[:NF]
                    n = h[0].data.shape[1]
                ts = [str(t if isinstance(t, str) else t.decode()) for t in tab['ts']]
                ts2 = [date.replace('-', '') + t[8:] for t in ts]
                do2 = [date + str(t)[10:] for t in tab['date_obs']]
                rms = np.asarray(tab['global_rms_jyb'], np.float32)
                cube = (rng.normal(0, 1, (NF, n, n)) * np.nan_to_num(rms, nan=1.0)[:, None, None]).astype(np.float32)
                if smp == 'HWJ' and S == 'I':
                    tx, ty = float(np.nanmedian(tab['target_x'])), float(np.nanmedian(tab['target_y']))
                    yy, xx = np.mgrid[0:n, 0:n]
                    sig = 2.0
                    cube += (8.0 * np.nan_to_num(rms, nan=1.0)[:, None, None]
                             * np.exp(-0.5 * ((xx - tx) ** 2 + (yy - ty) ** 2) / sig ** 2)[None]).astype(np.float32)
                H['CELL'], H['DATE'] = SYN_SNAP_CELL, date
                H['UTCSTART'] = ts2[0][:8] + 'T' + ts2[0][9:15]
                H['NFRAMES'] = NF
                tcols = [fits.Column(name='ts', format='15A', array=np.array(ts2)),
                         fits.Column(name='date_obs', format='26A', array=np.array(do2)),
                         fits.Column(name='admitted', format='L', array=np.ones(NF, bool)),
                         fits.Column(name='global_rms_jyb', format='E', array=rms),
                         fits.Column(name='ra_deg', format='D', array=np.asarray(tab['ra_deg'], float)),
                         fits.Column(name='dec_deg', format='D', array=np.asarray(tab['dec_deg'], float)),
                         fits.Column(name='target_x', format='E', array=np.asarray(tab['target_x'], np.float32)),
                         fits.Column(name='target_y', format='E', array=np.asarray(tab['target_y'], np.float32))]
                d = os.path.join(CUTS, smp, b, o)
                os.makedirs(d, exist_ok=True)
                fn = os.path.join(d, '%s_%s_%s_%s.fits' % (o, lst, H['UTCSTART'], S))
                fits.HDUList([fits.PrimaryHDU(cube, header=H),
                              fits.BinTableHDU.from_columns(tcols, name='TIMES')]).writeto(fn)
    # a target that is not in the current HWJ list: measured by nobody (manifest 'skipped: not in the current HWJ list')
    o = objs['HWJ']
    for p in glob.glob(os.path.join(CUTS, 'HWJ', SNAP_BANDS[0], o, '*_I.fits')):
        d = os.path.join(CUTS, 'HWJ', SNAP_BANDS[0], 'Not_A_Current_Target')
        os.makedirs(d, exist_ok=True)
        shutil.copy(p, os.path.join(d, os.path.basename(p).replace(o, 'Not_A_Current_Target')))


SNAP_COMMON = ['--tier', 'snapshot', '--cores', 'none', '--cutout-root', CUTS]


# ---------------------------------------------------------------------------------------------------------------------
# 1. config MIRROR values = the in-place engines' literals (read only)
# ---------------------------------------------------------------------------------------------------------------------

def mod_literals(path):
    """{name: value} of the module-level literal assignments of a source file (tuple unpacking included)."""
    out = {}
    for node in ast.parse(open(path).read()).body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            try:
                val = ast.literal_eval(node.value)
            except ValueError:
                continue
            if isinstance(tgt, ast.Name):
                out[tgt.id] = val
            elif isinstance(tgt, ast.Tuple) and isinstance(val, tuple) and len(val) == len(tgt.elts):
                for e, v in zip(tgt.elts, val):
                    if isinstance(e, ast.Name):
                        out[e.id] = v
    return out


def test_mirrors():
    m, ev, f, ph, cu = CFG.photometry_measure, CFG.photometry_evaluate, CFG.photometry_frames, CFG.photometry, \
        CFG.photometry_cutouts
    dp = os.path.join(OPS, 'deep_forced_phot.py')
    src = open(dp).read()
    L = mod_literals(dp)
    lat, lon, height = m.SITE['deep']
    want = {'LATD': lat, 'LOND': lon, 'MIN_EL': m.MIN_ELEVATION_DEG['deep'], 'N_CTRL': m.N_CONTROLS,
            'R_CTRL_BEAMS': m.RING_BEAMS, 'AVOID_BEAMS': m.AVOID_BEAMS, 'AVOID_JY': m.AVOID_JY, 'N_BLANK': m.N_BLANK['deep'],
            'ATEAM_R': m.ATEAM_RADIUS_DEG['deep'], 'MOON_R': m.MOON_RADIUS_DEG['deep'],
            'JUP_R': m.JUPITER_RADIUS_DEG['deep'],
            'STRUCT_R': {k: [tuple(x) for x in v] for k, v in m.STRUCT_RADIUS_BEAMS['deep'].items()}}
    got = {k: L.get(k) for k in want}
    got['STRUCT_R'] = {k: [tuple(x) for x in v] for k, v in (L.get('STRUCT_R') or {}).items()}
    check('mirror deep_forced_phot.py module literals (site, floors, ring, A-team, Moon, Jupiter, structure radii)',
          got == want, {k: (got[k], want[k]) for k in want if got[k] != want[k]})
    edge, h = m.EDGE_PX['deep'], m.STAMP_PX // 2
    a = h - m.STAMP_CORE_PX // 2
    lo, hi = m.SIZE_GATE['deep'][0][2], m.SIZE_GATE['deep'][1][2]
    ra_s = [x.split()[0] for x in m.ATEAM_SOURCES['deep']]
    de_s = [x.split()[1] for x in m.ATEAM_SOURCES['deep']]
    texts = ['height=%d * u.m' % int(height), '(x > %d) & (y > %d) & (x < nx - %d) & (y < ny - %d)' % (edge, edge, edge + 1, edge + 1),
             'd[yi - %d:yi + %d, xi - %d:xi + %d]' % (h, h + 1, h, h + 1), 's[%d:%d, %d:%d] = np.nan' % (a, a + 5, a, a + 5),
             's.size > %d' % m.STAMP_MIN_PIXELS['deep'], '%s * np.median' % m.RMS_MAD_FACTOR,
             'if not abs(snr) >= %g:' % m.FIT_SNR['deep'], "'fit-failed' if abs(snr) >= %g" % m.FIT_FAILED_FLAG_SNR['deep'],
             'lim = %s if abs(snr) < %g else %s' % (lo, m.SIZE_GATE['deep'][0][1], hi), 'abs(snr) >= %g and (' % m.SIZE_GATE['deep'][0][0],
             'for amp in %r:' % (m.INJECT_SIGMA['deep'],), 'd[yi - %d:yi + %d' % (m.INJECT_HALF_PX['deep'], m.INJECT_HALF_PX['deep'] + 1),
             "if pbv[i] > %s else ''" % m.PB_FLOOR['deep'], repr(ra_s).replace('"', "'"), repr(de_s).replace('"', "'"),
             'np.array([0, .25, .5, .75, 1.0])', 'thr = %s if el[-1] > el[0] else %s' % (m.TWILIGHT_SUN_DEG['deep']['rising'],
                                                                                         m.TWILIGHT_SUN_DEG['deep']['setting']),
             'len(qa) >= %d and v > %d *' % (m.NOISE_GATE_MIN_NIGHTS['deep'], m.NOISE_GATE_X['deep']),
             'len(tn) >= %d and v > %d *' % (m.NOISE_GATE_MIN_NIGHTS['deep'], m.NOISE_GATE_X['deep']),
             'default_rng(%d + SHARD)' % m.RNG_SEED_BASE['deep'],
             'rng.integers(%d, ny - %d, 4 * N_BLANK)' % (m.BLANK_EDGE_PX['deep'], m.BLANK_EDGE_PX['deep']),
             'for t in %r]' % (m.BLANK_SNR_COUNTS['deep'],), "open('%s')" % m.PARENT_TABLES['deep'],
             "open('%s')" % m.SOLAR_AUDIT['deep'], 'np.arange(0, 24, %s)' % m.JUPITER_TRACK['deep']['step_h'],
             '+ %s) + 12) %% 24) - 12)' % m.JUPITER_TRACK['deep']['lst_centre_offset_h'],
             '.alt.deg >= %d' % m.JUPITER_TRACK['deep']['min_el_deg'],
             "(('HWJ', '%s'), ('10pc', '%s'))" % (ph.SAMPLE_FILES['HWJ'], ph.SAMPLE_FILES['10pc']),
             "encoding='%s'" % ph.SAMPLE_ENCODING, "Time(h['DATE-OBS']), Time(h['DATE-END'])"]
    miss = [t for t in texts if t not in src]
    check('mirror deep_forced_phot.py in-code literals (edge, stamp, fit / size gate, injections, PB floor, day / twilight, '
          'noise gate, seed, blank sky, tables, samples)', not miss, miss)
    # the product FORMAT the content tests read (T1, B1, A1): the engine's header rows, in order
    hdr = re.search(r"W\.writerow\(\[(.*?)\] \+ DFR\.RADIO_COLS", src, re.S)
    eng = re.findall(r"'([a-zA-Z0-9_]+)'", hdr.group(1)) if hdr else []
    R = mod_literals(os.path.join(OPS, 'deep_forced_radio.py'))
    ok = (eng == list(T.FORCED_COLUMNS) and R.get('RADIO_COLS') == list(T.RADIO_COLUMNS)
          and "+ ['target_ra', 'target_dec'])" in src and 'BW.writerow(%r)' % list(T.BLANK_COLUMNS) in src
          and 'FW.writerow(%r)' % list(T.ADMISSION_COLUMNS) in src)
    check('format deep_forced_phot.py / deep_forced_radio.py: forced, radio, entry, blank and admission columns = '
          'tests.FORCED_COLUMNS / RADIO_COLUMNS / ENTRY_COLUMNS / BLANK_COLUMNS / ADMISSION_COLUMNS', ok,
          ([c for c in eng if c not in T.FORCED_COLUMNS], R.get('RADIO_COLS')))
    fit = CFG.fitting
    ok = (R.get('GAUSS_SNR') == fit.GAUSS_SNR and R.get('RADIO_SNR_FLOOR') == fit.RADIO_SNR_FLOOR
          and {S: tuple(v) for S, v in (R.get('K_TEMPLATE') or {}).items()} == {S: tuple(fit.K_TEMPLATE[('deep', S)])
                                                                                 for S in 'IV'})
    check('mirror deep_forced_radio.py (the in-place fitter of every listed row): GAUSS_SNR, RADIO_SNR_FLOOR, deep '
          'K_TEMPLATE = config fitting (T3 reads RADIO_SNR_FLOOR)', ok,
          (R.get('GAUSS_SNR'), R.get('RADIO_SNR_FLOOR'), R.get('K_TEMPLATE'), fit.K_TEMPLATE))
    fr = open(os.path.join(OPS, 'deep_forced_frames.py')).read()
    gi = open(os.path.join(OPS, 'gate_inputs.py')).read()
    texts = ["open('%s')" % f.ROOTS_CSV, "re.compile(r'%s')" % f.FRAME_NAME_RE, "(('I', '%s'), ('V', '%s'))" % (
        f.FRAME_SUBDIR['I'], f.FRAME_SUBDIR['V']), "f'{cf}/%s'" % f.PB_NAME['V'].replace('{band}', '{band}'),
        "f'%s'" % f.PB_NAME['I'], "f'{cf}/%s'" % f.WEIGHTS_NAME,
        "/%s'" % f.PARENT_NAME, 'list(rows[0])']
    miss = [t for t in texts if t not in fr] + ([f.MASTER_NAME.replace('{band}', '{band}')] if
                                                f.MASTER_NAME.replace('{band}', '{band}') not in gi else [])
    check('mirror deep_forced_frames.py / gate_inputs.py (roots table, frame names, PB by weighting, weights, parents, '
          'master)', not miss, miss)
    de = open(os.path.join(OPS, 'deep_forced_eval.py')).read()
    lo_, hi_, st_ = ev.THRESHOLD_GRID['deep']
    texts = ['np.arange(%s, %s, %s)' % (lo_, round(hi_ + 0.01, 2), st_), "ATEAM = {'I': %s, 'V': %s}; MOON = %s" % (
        m.ATEAM_RADIUS_DEG['deep']['I'], m.ATEAM_RADIUS_DEG['deep']['V'], m.MOON_RADIUS_DEG['deep']),
        'for t in %r:' % (ev.COMPLETENESS_T['deep'],), 'for A in %r]' % (ev.COMPLETENESS_A['deep'],),
        'tt.snr.abs() >= %d' % ev.LIST_SNR, 'dpi=%d' % ev.PNG_DPI['deep'], "for smp in ('HWJ', '10pc'):"]
    JUP = mod_literals(os.path.join(OPS, 'deep_forced_eval.py')).get('JUP')
    miss = [t for t in texts if t not in de] + (['JUP'] if JUP != m.JUPITER_RADIUS_DEG['deep'] else [])
    check('mirror deep_forced_eval.py (threshold grid, A-team / Moon / Jupiter, completeness, list S/N, plot)', not miss, miss)
    sp = os.path.join(OPS, 'snapshot_forced_phot.py')
    L = mod_literals(sp)
    sps = open(sp).read()
    want = {'ROOT': ph.RETIRED_CUTOUT_ROOTS[0], 'MCAT': CFG.catalog_snapshot.MASTER_CATALOGS,
            'SAMPLES': tuple(ph.SAMPLES_MEASURED), 'STAMP': m.STAMP_PX, 'NCTL': m.N_CONTROLS, 'RING_BEAMS': m.RING_BEAMS,
            'AVOID_BEAMS': m.AVOID_BEAMS, 'AVOID_JY': m.AVOID_JY, 'MIN_STAMP_FRAC': m.MIN_STAMP_FRAC['snapshot'],
            'EDGE_PX': m.EDGE_PX['snapshot']}
    got = {k: L.get(k) for k in want}
    # ROOT and SAMPLES are environment switches since 2026-10-25 / 10-28 (43eaf17 P3_PHOT_ROOT, c4fa0e3 P3_SNAP_CUTOUT_SAMPLES;
    # the step forbids both in its environment): their DEFAULTS are the mirror - ROOT's default is the RETIRED generation 4,
    # which is why the step always passes --root (the served generation, photometry.CUTOUT_ROOT)
    if "ROOT = os.environ.get('P3_PHOT_ROOT', %r)" % want['ROOT'] in sps:
        got['ROOT'] = want['ROOT']
    if "if os.environ.get('P3_SNAP_CUTOUT_SAMPLES') else %r)" % (want['SAMPLES'],) in sps:
        got['SAMPLES'] = want['SAMPLES']
    c2 = m.STAMP_CORE_PX // 2
    from exopipe.phase3.lib.photometry.tests import MINUTE_BASIS_INTEGRATION
    miss = [t for t in ('(np.abs(dy) <= %d) & (np.abs(dx) <= %d)' % (c2, c2), 'len(ix) >= %d]' % m.BLOCK_MIN_FRAMES['snapshot'],
                        '%s * np.nanmedian' % m.RMS_MAD_FACTOR, 'AST.beam_fwhm_arcmin(band)',
                        'SS.integration_minute_key(date_obs)', 'minute_key(dobs[i])', "minute_basis=np.array('%s')" % MINUTE_BASIS_INTEGRATION)
            if t not in sps]
    ss_lit = mod_literals(os.path.join(CLV, 'phase3_snapshot_search.py')).get('INTEGRATION_S')
    check('mirror snapshot_forced_phot.py (roots, samples, stamp, ring, avoidance, partial stamps, blocks, beam source, '
          'minute key = the integration-midpoint minute since exopipe_ops 40611cc, npz minute_basis, the integration time the '
          'block test uses = phase3_snapshot_search.INTEGRATION_S)', got == want and not miss and m.BEAM_SOURCE['snapshot'] == 'chain-table'
          and m.MINUTE_KEY['snapshot'] == 'integration-midpoint' and ss_lit == CFG.search_snapshot.INTEGRATION_S,
          ({k: (got[k], want[k]) for k in want if got[k] != want[k]}, miss, ss_lit))
    se = open(os.path.join(OPS, 'snapshot_forced_eval.py')).read()
    L = mod_literals(os.path.join(OPS, 'snapshot_forced_eval.py'))
    lo_, hi_, st_ = ev.THRESHOLD_GRID['snapshot']
    hlo, hhi, hst = ev.HIST_EDGES['snapshot']
    texts = ['TGRID = np.round(np.arange(%s, %s, %s), 2)' % (lo_, round(hi_ + 0.0001, 4), st_),
             'EDGES = np.arange(%s, %s, %s)' % (hlo, round(hhi + 0.0001, 4), hst),
             "'--survey-cells', type=int, default=%d" % ev.SURVEY_CELLS['snapshot'],
             'for t in %r:' % (ev.COMPLETENESS_T['snapshot'],), 'for A in %r' % (ev.COMPLETENESS_A['snapshot'],),
             '(stat[i, 0] >= %s)' % float(ev.LIST_SNR), "choices=['sb', 'wb']"]
    miss = [t for t in texts if t not in se]
    miss += [k for k, v in (('PROTO', CFG.paths.SNAPSHOT_FLEET_OUT), ('MCAT', CFG.catalog_snapshot.MASTER_CATALOGS))
             if L.get(k) != v]
    check('mirror snapshot_forced_eval.py (grid, histogram, survey size, completeness, list S/N, kinds, roots)', not miss, miss)
    L = mod_literals(os.path.join(CLV, 'phase3_snapshot_cutouts.py'))
    cs = open(os.path.join(CLV, 'phase3_snapshot_cutouts.py')).read()
    pl = cu.PLANET_SITE
    ok = (L.get('CUT_DEG') == cu.CUT_DEG and L.get('MIN_EL_DEG') == cu.MIN_EL_DEG and L.get('PLANETS') == cu.PLANETS
          and L.get('ROOT_DEFAULT') == ph.RETIRED_CUTOUT_ROOTS[0] and L.get('MERGE_ARCMIN') == cu.MERGE_ARCMIN
          and 'lat=%s * u.deg, lon=%s * u.deg, height=%s * u.m' % pl in cs and 'key = (body, ts[:12])' in cs
          and cu.PLANET_EPHEM_QUANTUM_MIN == 10)
    check('mirror phase3_snapshot_cutouts.py (cutout side, floor, planets, the retired default root, planet site, 10-minute '
          'ephemeris, the one-object merge radius)', ok)
    L = mod_literals(os.path.join(CLV, 'phase3_snapshot_astro.py'))
    rep = {k: v for k, v in (L.get('BAND_REP_FREQ') or {}).items() if k != 'Full'}
    ok = (L.get('FWHM_ARCMIN') == ev.SNAPSHOT_FWHM_ARCMIN and rep == ev.SNAPSHOT_BAND_REP_MHZ
          and L.get('OVRO_LAT_DEG') == ev.SNAPSHOT_SITE_LAT_DEG)
    check('mirror phase3_snapshot_astro.py (beam table, wideband representative frequency, site latitude)', ok)
    L = mod_literals(os.path.join(CLV, 'phase3_tier_rules.py'))
    check('mirror phase3_tier_rules.py DEEP_SEARCH_PIN', L.get('DEEP_SEARCH_PIN') == ev.DEEP_SEARCH_PIN['snapshot'])


# ---------------------------------------------------------------------------------------------------------------------
# 2. ports: the content tests' measurement / zone code = the production functions (read only)
# ---------------------------------------------------------------------------------------------------------------------

def _prod_function(path, name, glb):
    ns = dict(glb)
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(compile(ast.Module(body=[node], type_ignores=[]), '%s:%s' % (os.path.basename(path), name), 'exec'), ns)
            return ns[name]
    raise KeyError(name)


def test_ports():
    from astropy.wcs import WCS
    from scipy.ndimage import map_coordinates
    m = CFG.photometry_measure
    rng = np.random.default_rng(3)
    d = rng.normal(0, 0.1, (NPIX, NPIX))
    d[60:70, 60:70] = 0.0                           # zero pixels are left out of the stamp (deep rule)
    d[120:125, 30:35] = np.nan
    w = WCS(wcs_header()).celestial
    prod = _prod_function(os.path.join(OPS, 'deep_forced_phot.py'), 'measure', {'np': np, 'map_coordinates': map_coordinates})
    ras = RA0 + np.array([0.0, 1.0, -2.0, 4.4, 4.9, -3.1]) / math.cos(math.radians(DEC0))
    decs = DEC0 + np.array([0.0, -1.5, 2.0, 0.3, -4.6, 1.2])
    pv = prod(d, w, ras, decs)
    mine = np.array([T._measure_one(d, w, a, b, m.EDGE_PX['deep'], m.STAMP_PX, m.STAMP_CORE_PX, m.STAMP_MIN_PIXELS['deep'],
                                    m.RMS_MAD_FACTOR) for a, b in zip(ras, decs)])
    same = np.array_equal(np.isnan(pv), np.isnan(mine)) and np.allclose(np.nan_to_num(pv), np.nan_to_num(mine), rtol=0, atol=0)
    check('port T4: tests._measure_one = deep_forced_phot.measure (value and stamp rms, edge and NaN cases, exact)', same,
          (pv, mine))
    sp = os.path.join(OPS, 'snapshot_forced_phot.py')
    L = mod_literals(sp)
    prod = _prod_function(sp, 'measure', {'np': np, 'STAMP': L['STAMP'], 'MIN_STAMP_FRAC': L['MIN_STAMP_FRAC']})
    cube = rng.normal(0, 1, (5, 65, 65)).astype(np.float32)
    cube[:, :, :3] = np.nan
    xs, ys = np.array([32.3, 5.2, 60.7, 2.0, 40.5]), np.array([30.6, 31.1, 50.2, 2.0, 63.5])
    pv, pr = prod(cube, xs, ys)
    ok = True
    for i in range(len(xs)):
        g = T._snap_measure(cube, xs[i], ys[i], m.STAMP_PX, m.STAMP_CORE_PX, m.MIN_STAMP_FRAC['snapshot'], m.RMS_MAD_FACTOR)
        if g is None:
            ok &= bool(np.isnan(pv[i]).all())
        else:
            ok &= bool(np.allclose(g[0], pv[i], rtol=1e-6, equal_nan=True) and np.allclose(g[1], pr[i], rtol=1e-6, equal_nan=True))
    check('port N4: tests._snap_measure = snapshot_forced_phot.measure (partial stamps, edge cases)', ok)
    ev = '/lustre/gh/main/phase3/snapshot_forced/eval_wb_20261007'
    if not os.path.exists(os.path.join(ev, 'targets_ge4.csv')):
        print('skip  Z1 port on real cells (evaluation not readable here)')
        return
    import importlib.util
    spec = importlib.util.spec_from_file_location('_tsfr', os.path.join(OPS, 'test_snapshot_forced_reserve.py'))
    Pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Pm)
    J = json.load(open(Pm.CURVES))
    Pm._TABLES = {k: v.get('table') for k, v in J['curves'].items()}
    by = {}
    for r in T.read_csv(os.path.join(ev, 'targets_ge4.csv'))[1]:
        by.setdefault(r['cell'], []).append(r)
    cells = sorted(by)[::120][:2]
    # the 2026-10-07 evaluation was built on generation 4's stage 1 (<snapshot_forced>/stage1, the reserve test's own STAGE1
    # default): the port is compared on those inputs
    s1_gen4 = os.path.join(CFG.photometry.OUT_ROOT['snapshot'], 'stage1')
    same = Pm.STAGE1 == s1_gen4 and all(Pm.zones_cell((c, by[c])) == T.zones_cell((c, by[c], s1_gen4)) for c in cells)
    check('port Z1: tests.zones_cell = test_snapshot_forced_reserve.zones_cell on real cells %s' % cells, same)


# ---------------------------------------------------------------------------------------------------------------------
# 3. gates (run level and deep frames)
# ---------------------------------------------------------------------------------------------------------------------

def test_gates():
    out = os.path.join(TMP, 'gates')
    os.environ['SF_BANDS'] = 'Red'
    try:
        rc, txt = step(['photometry'] + SNAP_COMMON + ['--stages', 'measure', '--cells', SYN_SNAP_CELL, '--out', out,
                                                        '--work', os.path.join(TMP, 'work')])
    finally:
        del os.environ['SF_BANDS']
    check('PHOT-ENV: a caller environment with SF_* refuses (exit 2)', rc == 2 and 'PHOT-ENV' in txt and 'SF_BANDS' in txt, txt)
    rc, txt = step(['photometry', '--tier', 'sublst', '--stages', 'measure', '--out', out, '--cores', 'none'])
    check('PHOT-SUBLST-NO-ENGINE: --tier sublst --stages measure refuses (exit 2; sub-LST has cutouts and cutmeasure only)',
          rc == 2 and 'PHOT-SUBLST-NO-ENGINE' in txt, txt)
    rc, txt = step(['photometry'] + SNAP_COMMON + ['--scopes', 'Full', '--cells', SYN_SNAP_CELL, '--out', out])
    check('PHOT-FULL: snapshot --scopes Full refuses (exit 2)', rc == 2 and 'PHOT-FULL' in txt, txt)
    rc, txt = step(['photometry'] + SNAP_COMMON + ['--cells', SYN_SNAP_CELL + '_Full', '--out', out])
    check('PHOT-FULL: a snapshot cell item with the Full band refuses (exit 2)', rc == 2 and 'PHOT-FULL' in txt, txt)
    rc, txt = step(['photometry'] + SNAP_COMMON + ['--cells', SYN_SNAP_CELL, '--out', out,
                                                    '--set', 'photometry.SAMPLE_SHA1={"HWJ": "0", "10pc": "0"}'])
    check('PHOT-SAMPLE-PIN: the sample pins cannot be overridden (protected key, exit 2)', rc == 2 and 'REFUSED' in txt, txt)
    rc, txt = step(['photometry'] + SNAP_COMMON + ['--cells', SYN_SNAP_CELL, '--out', out,
                                                    '--set', 'photometry_measure.RING_BEAMS=3.0'])
    check('a MIRROR key cannot be overridden (the engine keeps its own literal; exit 2)', rc == 2 and 'REFUSED' in txt, txt)
    CFG.reset()
    paths = K.gate_samples()
    check('PHOT-SAMPLE-PIN: the measured lists match their pinned sha1 today (%s)' % sorted(paths), len(paths) == 2)
    with patched(CFG.photometry, SAMPLE_SHA1={'HWJ': 'ede4fe33ec4e2686dfff9a0c69b54a8113b1a842', '10pc': 'f' * 40}):
        try:
            K.gate_samples()
            ok = False
        except record.Refusal as e:
            ok = 'PHOT-SAMPLE-PIN' in str(e) and '10pc' in str(e)
    check('PHOT-SAMPLE-PIN: a list whose sha1 differs from its pin refuses', ok)
    rc, txt = step(['photometry', '--tier', 'deep', '--stages', 'bogus', '--src-run', SRC, '--out', out, '--cores', 'none'])
    check('an unknown stage name is an error (exit 1)', rc == 1 and 'unknown stage' in txt, txt)
    base = ['photometry', '--tier', 'deep', '--stages', 'frames', '--cells', '10h_46MHz', '--cores', 'none',
            '--set', 'paths.COADD_SB_ROOT=%s' % COADD]
    rc, txt = step(base + ['--parent-dir', PARENTS, '--out', os.path.join(TMP, 'g1')])
    check('F-SRCRUN: no --src-run / --run refuses the frames unit (exit 2)', rc == 2 and 'F-SRCRUN' in txt, txt)
    rc, txt = step(base + ['--src-run', SRC, '--out', os.path.join(TMP, 'g2')])
    check('F-PARENTDIR: no --parent-dir refuses the frames unit (exit 2)', rc == 2 and 'F-PARENTDIR' in txt, txt)
    rc, txt = step(base[:6] + ['10h_82MHz'] + base[7:] + ['--src-run', SRC, '--parent-dir', PARENTS,
                                                           '--out', os.path.join(TMP, 'g3')])
    why = records(os.path.join(TMP, 'g3')).get('frames', {}).get('reason', '')
    check('F-NOFRAMES: a cell without search frames (the engine writes no row) is NO INPUT (exit 3)',
          rc == 3 and 'F-NOFRAMES' in why, (why, txt))
    rc, txt = step(base[:6] + ['10h_Blue'] + base[7:] + ['--src-run', SRC, '--parent-dir', PARENTS,
                                                          '--out', os.path.join(TMP, 'g4')])
    check('F-PARENTS: a cell WITH frames whose parent catalogue is missing (engine wrote no row) is REFUSED (exit 2), '
          'not no-input', rc == 2 and 'F-PARENTS' in txt, txt)
    rc, txt = step(base[:6] + ['10h_46MHz,10h_Blue'] + base[7:] + ['--src-run', SRC, '--parent-dir', PARENTS,
                                                                    '--out', os.path.join(TMP, 'g5')])
    rec = records(os.path.join(TMP, 'g5')).get('frames', {})
    ri = rec.get('refused_items') or []
    check('F-PARENTS per cell: the other cells are listed, the refused cell is an item refusal with its gate (exit 0)',
          rc == 0 and rec.get('status') == 'done' and [r['gate'] for r in ri] == ['F-PARENTS'], (txt, ri))


# ---------------------------------------------------------------------------------------------------------------------
# 4. deep end to end
# ---------------------------------------------------------------------------------------------------------------------

DEEP_OUT = os.path.join(TMP, 'deep_out')


def test_deep_e2e(excluded):
    # dry run first: nothing on disk; evaluate is planned from the measure stage of this invocation
    argv = ['photometry'] + DEEP_COMMON + ['--cells', '10h_46MHz,10h_Green', '--shards', '4', '--out', DEEP_OUT]
    rc, txt = step(argv + ['--dry-run'])
    check('deep multi-stage --dry-run: evaluate planned from the measure stage of this run, no M-CHAIN refusal, exit 0, '
          'nothing written', rc == 0 and 'shard records from the measure stage of this run' in txt
          and 'WOULD REFUSE' not in txt and 'frames stage of this run' in txt and not os.path.exists(DEEP_OUT), txt)
    rc, txt = step(['photometry'] + DEEP_COMMON + ['--stages', 'evaluate', '--out', DEEP_OUT, '--dry-run'])
    check('deep --stages evaluate --dry-run with no measure record: WOULD REFUSE M-CHAIN, exit 2, nothing written',
          rc == 2 and 'WOULD REFUSE M-CHAIN' in txt and not os.path.exists(DEEP_OUT), txt)
    t = time.time()
    rc, txt = step(argv)
    recs = records(DEEP_OUT)
    st = {k: v['status'] for k, v in recs.items()}
    check('deep frames -> stage -> measure -> evaluate in one invocation: exit 0 (%.0f s)' % (time.time() - t),
          rc == 0 and st == {'frames': 'done', 'measure.shard_00': 'done', 'measure.shard_01': 'done', 'evaluate': 'done'},
          (st, txt[-3000:]))
    check('stage not needed (every input readable here): no stage unit, measure read frames_07.csv from the frames record',
          'stage] not needed' in txt and recs['measure.shard_00']['meta']['frame_list'].endswith('frames_07.csv'), txt)
    failed = {k: [t_['name'] for t_ in v['tests'] if not t_['passed']] for k, v in recs.items()}
    ntests = {k: len(v['tests']) for k, v in recs.items()}
    check('every content test of every output passed (%s)' % ntests, not any(failed.values()) and all(ntests.values()), failed)
    ev = os.path.join(DEEP_OUT, 'out', 'eval')
    th = T.read_csv(os.path.join(ev, 'thresholds.csv'))[1]
    g4 = T.read_csv(os.path.join(ev, 'targets_ge4.csv'))[1]
    check('the REAL deep_forced_eval.py wrote 4 threshold rows and lists the injected HWJ source (|S/N| >= 4) with its radio '
          'position', len(th) == 4 and g4 and all(r['target'] == DEEP_TARGETS[0][1] and r['pos_kind'] == 'template' for r in g4),
          (th, g4[:2]))
    code = recs['measure.shard_00']['meta'].get('legacy_code') or {}
    inp = recs['measure.shard_00']['inputs']
    fr = [p for p in inp if p.endswith('_redo_resid_46MHz.fits') or p.endswith('_redo_resid_Green.fits')]
    check('measure records: the engine code fingerprint, the frame list, samples, pinned tables, every frame (size + mtime '
          '+ cards, no full sha1)', any(k.endswith('deep_forced_phot.py') for k in code) and fr
          and all('sha1' not in inp[p] for p in fr) and recs['measure.shard_00'].get('admitted'), (list(code)[:3], fr[:2]))
    adm = T.read_csv(os.path.join(DEEP_OUT, 'out', 'frames_000.csv'))[1] + T.read_csv(os.path.join(DEEP_OUT, 'out',
                                                                                                   'frames_001.csv'))[1]
    noisy = [r for r in adm if r['date'] == NOISY_V_DATE and r['stokes'] == 'V']
    exc = [r for r in adm if excluded and (r['lst'], r['band'], r['date']) == excluded]
    check('admission: the noisy V night refused by the 3x noise gate%s' % (', the data_decisions epoch refused' if excluded
                                                                          else ''),
          noisy and all(r['admitted'] == '0' and r['reason'] == '3x per-Stokes noise gate' for r in noisy)
          and (not excluded or (exc and all(r['reason'] == 'excluded epoch (data_decisions)' for r in exc))), (noisy, exc))
    # --resume: a clean-tree record is skipped; a dirty-tree record never resumes
    with patched(record, git_commit=lambda path: 'abc1234'):
        rc, txt = step(argv)                                    # records written under a clean commit
        rc2, txt2 = step(argv + ['--resume'])
    check('--resume (clean tree): frames, both shards and evaluate skipped (inputs, code, config unchanged; outputs pass)',
          rc == 0 and rc2 == 0 and txt2.count('resume ') == 4 and txt2.count(': skipped (') == 4, txt2[-2000:])
    with patched(record, git_commit=lambda path: 'abc1234-dirty'):
        step(argv)
        rc3, txt3 = step(argv + ['--resume'])
    check('--resume (dirty tree): never resumed - every unit re-runs', rc3 == 0 and 'dirty tree' in txt3
          and ': skipped (' not in txt3, txt3[-1500:])
    # per hour: one night; the Stokes V noise gate decided over the full night list (an item refusal)
    out_e = os.path.join(TMP, 'deep_epoch')
    rc, txt = step(['photometry'] + DEEP_COMMON + ['--epochs', '10h_%s' % NOISY_V_DATE, '--shards', '4', '--out', out_e])
    re_ = records(out_e)
    fl = T.read_csv(os.path.join(out_e, 'frames_07.csv'))[1]
    ri = re_.get('frames', {}).get('refused_items') or []
    noise = [r for r in ri if r['gate'] == 'M-NOISE']
    check('--epochs: frames + stage + measure of one night (no evaluate); the noisy V frames are M-NOISE item refusals '
          '(the gate decided over the full night list), the other cells of the LST refused by their own gates',
          rc == 0 and sorted(re_) == ['frames', 'measure.shard_00', 'measure.shard_01'] and {r['date'] for r in fl} ==
          {NOISY_V_DATE} and len(noise) == 2 and all('_V_' in r['path'] for r in noise)
          and {r['stokes'] for r in fl} == {'I'} and {r['gate'] for r in ri} == {'M-NOISE', 'F-PARENTS', 'F-NOFRAMES'},
          (sorted(re_), ri, txt[-1500:]))
    # M-CHAIN: a staged list made from ANOTHER frame list is never read
    fsha = recs['frames']['meta']['frame_list_sha1']
    srec = record.record_path(DEEP_OUT, 'photometry', 'stage')
    record.atomic_write_json(srec, {'unit': 'stage', 'status': 'done', 'meta': {'source_list_sha1': 'f' * 40,
                                                                                'frame_list_sha1': 'e' * 40},
                                    'outputs': [{'path': os.path.join(DEEP_OUT, 'frames_09.csv'),
                                                 'class': 'photometry_frame_list'}]})
    rc, txt = step(['photometry'] + DEEP_COMMON + ['--stages', 'measure', '--shards', '4', '--out', DEEP_OUT, '--dry-run'])
    ok1 = rc == 0 and '(frames record;' in txt and 'frames_09' not in txt
    record.atomic_write_json(srec, {'unit': 'stage', 'status': 'done', 'meta': {'source_list_sha1': fsha,
                                                                                'frame_list_sha1': 'e' * 40},
                                    'outputs': [{'path': os.path.join(DEEP_OUT, 'frames_09.csv'),
                                                 'class': 'photometry_frame_list'}]})
    rc, txt2 = step(['photometry'] + DEEP_COMMON + ['--stages', 'measure', '--shards', '4', '--out', DEEP_OUT, '--dry-run'])
    os.remove(srec)
    check('M-CHAIN: a stage record made from another frame list is ignored; one made from the current list is used',
          ok1 and 'frames_09.csv from the stage record' in txt2, (txt[-800:], txt2[-800:]))
    fl7 = os.path.join(DEEP_OUT, 'frames_07.csv')
    keep = open(fl7, 'rb').read()
    open(fl7, 'ab').write(b'\r\n')
    rc, txt = step(['photometry'] + DEEP_COMMON + ['--stages', 'measure', '--shards', '4', '--out', DEEP_OUT])
    open(fl7, 'wb').write(keep)
    check('M-CHAIN: a frame list changed since its record refuses measure (exit 2)', rc == 2 and 'changed since' in txt, txt)
    stray = os.path.join(DEEP_OUT, 'out', 'blank_031.csv')
    shutil.copy(os.path.join(DEEP_OUT, 'out', 'blank_000.csv'), stray)
    rc, txt = step(['photometry'] + DEEP_COMMON + ['--stages', 'evaluate', '--out', DEEP_OUT])
    os.remove(stray)
    check('M-CHAIN: a blank_*.csv no shard record lists refuses evaluate (the engine globs it; exit 2)',
          rc == 2 and 'blank_031.csv' in txt, txt)
    stray = os.path.join(DEEP_OUT, 'out', 'forced_031.csv.gz')
    shutil.copy(os.path.join(DEEP_OUT, 'out', 'forced_000.csv.gz'), stray)
    rc, txt = step(['photometry'] + DEEP_COMMON + ['--stages', 'evaluate', '--out', DEEP_OUT])
    os.remove(stray)
    check('M-CHAIN: a forced_*.csv.gz no shard record lists refuses evaluate (exit 2)', rc == 2 and 'forced_031' in txt, txt)
    r1 = record.record_path(DEEP_OUT, 'photometry', 'measure.shard_01')
    keep = open(r1).read()
    os.remove(r1)
    rc, txt = step(['photometry'] + DEEP_COMMON + ['--stages', 'evaluate', '--out', DEEP_OUT])
    open(r1, 'w').write(keep)
    check('M-CHAIN: a shard without its done record refuses evaluate (exit 2)', rc == 2 and 'missing [1]' in txt, txt)


# ---------------------------------------------------------------------------------------------------------------------
# 5. snapshot end to end (REAL engines in place, py36)
# ---------------------------------------------------------------------------------------------------------------------

SNAP_OUT = os.path.join(SNAP, 'out')
WORK = os.path.join(TMP, 'work')


def test_snapshot_e2e():
    argv = ['photometry'] + SNAP_COMMON + ['--cells', SYN_SNAP_CELL, '--out', SNAP_OUT, '--work', WORK, '--eval-tag', 'T1',
                                           '--band-procs', '4']
    rc, txt = step(argv + ['--dry-run'])
    check('snapshot multi-stage --dry-run: evaluate planned from the measure stage of this run, exit 0, nothing written',
          rc == 0 and 'from the measure stage of this run' in txt and 'WOULD REFUSE' not in txt
          and not os.path.exists(SNAP_OUT), txt)
    t = time.time()
    rc, txt = step(argv)
    recs = records(SNAP_OUT)
    st = {k: v['status'] for k, v in recs.items()}
    want = {'measure.%s' % SYN_SNAP_CELL: 'done', 'evaluate.sb.T1': 'done', 'evaluate.wb.T1': 'done',
            'evaluate.zones.T1': 'done'}
    check('snapshot measure -> evaluate sb, wb -> zone flags (REAL snapshot_forced_phot.py / snapshot_forced_eval.py): '
          'exit 0 (%.0f s)' % (time.time() - t), rc == 0 and st == want, (st, txt[-3000:]))
    failed = {k: [t_['name'] for t_ in v['tests'] if not t_['passed']] for k, v in recs.items()}
    ntests = {k: len(v['tests']) for k, v in recs.items()}
    check('every content test of every output passed (%s)' % ntests, not any(failed.values()) and all(ntests.values()), failed)
    s1 = os.path.join(SNAP_OUT, S1, SYN_SNAP_CELL)
    npz = sorted(os.path.basename(p) for p in glob.glob(os.path.join(s1, 'forced_*.npz')))
    man = T.read_csv(os.path.join(s1, 'forced_manifest.csv'))[1]
    check('stage 1: 12 npz (2 samples x 3 bands x I,V), the merged manifest (12 ok cutouts, 1 skipped non-current target)',
          len(npz) == 12 and sum(r['status'] == 'ok' for r in man) == 12
          and sum(r['status'].startswith('skipped') for r in man) == 1, (npz, [r['status'] for r in man]))
    sb = T.read_csv(os.path.join(SNAP_OUT, 'eval_T1', 'thresholds.csv'))[1]
    g4 = T.read_csv(os.path.join(SNAP_OUT, 'eval_T1', 'targets_ge4.csv'))[1]
    wb = T.read_csv(os.path.join(SNAP_OUT, 'eval_wb_T1', 'targets_ge4.csv'))[1]
    rf = T.read_csv(os.path.join(SNAP_OUT, 'refused_cells_wb_T1.csv'))[1]
    pin = [r['cell'] for r in T.read_csv(CFG.photometry_evaluate.REFUSED_CELLS['snapshot'])[1]]
    check('evaluations: 8 threshold rows, the injected HWJ source listed in the sb (64/73 MHz) and wb (Blue) lists; the '
          'refused-cell list keeps the pinned file\'s order (production reserve :140-141)',
          len(sb) == 8 and g4 and wb and {r['band'] for r in wb} == {'Blue'} and [r['cell'] for r in rf] == pin,
          (len(sb), len(g4), len(wb), [r['cell'] for r in rf]))
    code = recs['measure.%s' % SYN_SNAP_CELL]['meta'].get('legacy_code') or {}
    check('the measure record names the in-place engine code (child fingerprint)',
          any(k.endswith('snapshot_forced_phot.py') for k in code) and any(k.endswith('phase3_snapshot_cutouts.py')
                                                                           for k in code), list(code)[:5])
    with patched(record, git_commit=lambda path: 'abc1234'):
        step(argv)
        rc, txt = step(argv + ['--resume'])
    check('--resume (clean tree): measure and both evaluations skipped (zone flags recomputed)',
          rc == 0 and txt.count(': skipped (') == 3, txt[-1500:])
    rc, txt = step(['photometry'] + SNAP_COMMON + ['--stages', 'measure', '--cells', '18h_2025-04-15', '--out', SNAP_OUT,
                                                    '--work', WORK])
    why = records(SNAP_OUT).get('measure.18h_2025-04-15', {}).get('reason', '')
    check('SM-NOCUT: a cell without HWJ / 10 pc cutouts is NO INPUT (exit 3)', rc == 3 and 'SM-NOCUT' in why, (why, txt))
    rc, txt = step(['photometry'] + SNAP_COMMON + ['--stages', 'measure', '--cells', SYN_SNAP_CELL, '--scopes', '64MHz',
                                                    '--out', SNAP_OUT, '--work', WORK])
    check('SM-SUBSET: a band subset over a full-band manifest refuses (exit 2)', rc == 2 and 'SM-SUBSET' in txt, txt)
    stale = os.path.join(s1, 'forced_25pc_64MHz_I.npz')
    shutil.copy(os.path.join(s1, 'forced_HWJ_64MHz_I.npz'), stale)
    meas = ['photometry'] + SNAP_COMMON + ['--stages', 'measure', '--cells', SYN_SNAP_CELL, '--out', SNAP_OUT,
                                           '--work', WORK, '--band-procs', '4']
    rc, txt = step(meas)
    check('SM-STALE: an npz of a measured band this measurement did not write refuses (exit 2)',
          rc == 2 and 'SM-STALE' in txt and os.path.exists(stale), txt)
    h0 = hashlib.sha256(open(stale, 'rb').read()).hexdigest()
    rc, txt = step(meas + ['--fresh'])
    q = glob.glob(os.path.join(SNAP_OUT, 'quarantine_superseded_photometry_*', SYN_SNAP_CELL, 'forced_25pc_64MHz_I.npz'))
    mrow = T.read_csv(glob.glob(os.path.join(SNAP_OUT, 'quarantine_superseded_photometry_*', 'MANIFEST_sha256.csv'))[0])[1] \
        if q else []
    check('SM-STALE --fresh: the stale file moved to a quarantine directory with its sha256 in the manifest (not deleted)',
          rc == 0 and not os.path.exists(stale) and q and mrow and mrow[0]['sha256'] == h0, (txt[-800:], q, mrow))
    ev = ['photometry'] + SNAP_COMMON + ['--stages', 'evaluate', '--out', SNAP_OUT, '--eval-tag', 'T2']
    rc, txt = step(ev + ['--cells', '%s,%s' % (SYN_SNAP_CELL, pin[0])])
    check('SE-LISTED: a selected cell that is listed refused refuses the evaluation (exit 2)', rc == 2 and 'SE-LISTED' in txt,
          txt)
    bare = os.path.join(SNAP_OUT, S1, '18h_2025-04-16')
    os.makedirs(bare)
    rc, txt = step(ev)
    os.rmdir(bare)
    check('SE-DONE: a stage-1 cell without a done measure record or manifest refuses the evaluation (exit 2)',
          rc == 2 and 'SE-DONE' in txt and '18h_2025-04-16' in txt, txt)
    rc, txt = step(['photometry'] + SNAP_COMMON + ['--stages', 'evaluate', '--out', os.path.join(TMP, 'empty')])
    check('SE-NONE: no stage-1 cell refuses (exit 2)', rc == 2 and 'SE-NONE' in txt, txt)
    # --refused-cells KIND=FILE (replay of a production evaluation's per-kind list) and --survey-cells (a RULED key)
    prod = CFG.photometry.OUT_ROOT['snapshot']
    rwb = os.path.join(prod, 'refused_cells_wb_20261007.csv')
    if not os.path.exists(rwb):
        print('skip  --refused-cells replay (%s not readable here)' % rwb)
        return
    short = os.path.join(TMP, 'refused_wb_short.csv')
    open(short, 'w').write('cell,reason\n')
    rc, txt = step(ev + ['--kinds', 'wb', '--refused-cells', 'wb=%s' % short, '--dry-run'])
    check('SE-LISTED: a --refused-cells list that omits a pinned refused cell refuses (dry run, exit 2)',
          rc == 2 and 'omits the pinned refused cell' in txt, txt)
    t = time.time()
    rc, txt = step(['photometry'] + SNAP_COMMON + ['--stages', 'evaluate', '--out', SNAP_OUT, '--eval-tag', 'T3', '--kinds',
                                                    'wb', '--refused-cells', 'wb=%s' % rwb, '--survey-cells', '300'])
    r3 = records(SNAP_OUT).get('evaluate.wb.T3', {})
    ov = [o for o in r3.get('config_overrides') or [] if o[0] == 'photometry_evaluate.SURVEY_CELLS[snapshot]']
    th3 = T.read_csv(os.path.join(SNAP_OUT, 'eval_wb_T3', 'thresholds.csv'))[1] if r3 else []
    same = r3 and open(os.path.join(SNAP_OUT, 'eval_wb_T3', 'refused_cells.csv'), 'rb').read() == open(
        os.path.join(prod, 'eval_wb_20261007', 'refused_cells.csv'), 'rb').read()
    check('--refused-cells wb=<the 2026-10-07 list> is passed to the engine as written (refused_cells.csv byte-identical '
          'to production eval_wb_20261007); --survey-cells 300 is a recorded config override (non-production out) that '
          'reaches the engine (%.0f s)' % (time.time() - t),
          rc == 0 and r3.get('status') == 'done' and same and r3['meta'].get('refused_cells_file') == rwb and ov
          and 'non-production out' in str(ov[0][4]) and th3 and all('(300 cells / 1 here)' in r['survey_note'] for r in th3),
          (rc, r3.get('status'), r3.get('reason'), same, ov, [r['survey_note'] for r in th3][:1], txt[-1200:]))
    had = os.path.exists(os.path.join(prod, '_records'))
    rc, txt = step(['photometry'] + SNAP_COMMON + ['--stages', 'evaluate', '--out', prod, '--survey-cells', '300',
                                                    '--dry-run'])
    check('--survey-cells other than the config value refuses into the production root without --ruling-for '
          '(RULED key; dry run, exit 2, nothing written)', rc == 2 and 'SURVEY_CELLS' in txt and 'carries a ruling' in txt
          and os.path.exists(os.path.join(prod, '_records')) == had, txt)


# ---------------------------------------------------------------------------------------------------------------------
# 6. negative: every content test fails on a corrupted copy
# ---------------------------------------------------------------------------------------------------------------------

NEG = os.path.join(TMP, 'neg')


def _csv_rewrite(src, dst, fn, gz=False):
    op = (lambda p, m: gzip.open(p, m + 't', newline='')) if gz else (lambda p, m: open(p, m, newline=''))
    with op(src, 'r') as f:
        rows = list(csv.reader(f))
    rows = fn(rows)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with op(dst, 'w') as f:
        csv.writer(f).writerows(rows)
    return dst


def neg(cls, test, path, **ctx):
    res = run_tests(cls, path, **ctx)
    r = res.get(test)
    return r is not None and not r['passed'], (r or {}).get('detail')


def test_negative():
    # ---- deep frame list (frames_07.csv), stage list
    fl = os.path.join(DEEP_OUT, 'frames_07.csv')
    ok, bad = all_pass('photometry_frame_list', fl, coadd_root=COADD)
    check('frame list original passes FL1-FL6', ok, bad)
    d = os.path.join(NEG, 'fl')
    hdr = T.read_csv(fl)[0]
    iv = hdr.index('thermnoi')
    cases = [('FL1_header', lambda r: [x[:-1] for x in r], {}),
             ('FL2_frames', lambda r: r + [r[1]], {}),
             ('FL3_pb_by_weighting', lambda r: [r[0]] + [x[:5] + [x[5].replace('PB_deep_R075_', 'PB_deep_')] + x[6:]
                                                         for x in r[1:]], {}),
             ('FL5_thermnoi', lambda r: [r[0]] + [x[:iv] + ['9.99' if x[3] == 'V' else x[iv]] for x in r[1:]], {}),
             ('FL6_refused_cells', lambda r: r, {'refused_cells': ['10h_46MHz']}),
             ('FL6_refused_cells', lambda r: r, {'refused_stokes_i_cells': ['10h_46MHz']})]
    for i, (test, fn, ctx) in enumerate(cases):
        p = _csv_rewrite(fl, os.path.join(d, str(i), 'frames_07.csv'), fn)
        f_, det = neg('photometry_frame_list', test, p, coadd_root=COADD, **ctx)
        check('negative %s fails on a corrupted frame list (%s)' % (test, ctx or 'rewritten rows'), f_, det)
    # deep_forced_frames.py:50-51: without its Robust -0.75 PB map each Stokes I frame of a cell is refused and its V
    # rows stay listed - FL6 accepts exactly that
    p = _csv_rewrite(fl, os.path.join(d, 'vonly', 'frames_07.csv'),
                     lambda r: [r[0]] + [x for x in r[1:] if not (x[0] == '10h' and x[1] == '46MHz' and x[3] == 'I')])
    res = run_tests('photometry_frame_list', p, coadd_root=COADD, refused_stokes_i_cells=['10h_46MHz'])
    check('FL6: a cell whose Stokes I frames were refused (no I PB map) with its V rows listed passes',
          res['FL6_refused_cells']['passed'], res['FL6_refused_cells']['detail'])
    pd_ = os.path.join(d, 'parents')
    shutil.copytree(PARENTS, pd_)
    g = json.load(open(os.path.join(pd_, 'master_R075_10h_46MHz.gaul.csv.gen.json')))
    g['master_md5_data'] = '0' * 32
    json.dump(g, open(os.path.join(pd_, 'master_R075_10h_46MHz.gaul.csv.gen.json'), 'w'))
    ip = hdr.index('parent_path')
    p = _csv_rewrite(fl, os.path.join(d, 'fl4', 'frames_07.csv'),
                     lambda r: [r[0]] + [x[:ip] + [x[ip].replace(PARENTS, pd_)] + x[ip + 1:] for x in r[1:]])
    with patched(CFG.photometry_frames, FL4_DIGEST_CELLS=10):
        f_, det = neg('photometry_frame_list', 'FL4_parents', p, coadd_root=COADD)
    check('negative FL4_parents fails when a parent catalogue was built on another master generation', f_, det)
    sd = os.path.join(d, 'stage')
    os.makedirs(sd)
    rows = [['lst', 'band', 'date', 'stokes', 'path', 'pb_path', 'parent_path', 'weights_path', 'thermnoi'],
            ['10h', '46MHz', '2099-01-01', 'I', '/data07/x/a.fits', '/data07/x/pb.fits', '/data07/x/p.csv', '', '']]
    _csv_rewrite(fl, os.path.join(sd, 'frames_07.csv'), lambda r: rows)
    sl = os.path.join(sd, 'stage_list.txt')
    open(sl, 'w').write('\n'.join(sorted(['/data07/x/a.fits', '/data07/x/pb.fits', '/data07/x/p.csv'])) + '\n')
    ok, bad = all_pass('photometry_stage_list', sl)
    open(sl, 'w').write('\n'.join(['/data07/x/pb.fits', '/data07/x/a.fits', '/data07/x/p.csv', '/data07/y.fits']) + '\n')
    f_, det = neg('photometry_stage_list', 'SL1_paths', sl)
    check('stage list: original passes SL1, an unsorted list with a path the frame list does not name fails', ok and f_,
          (bad, det))
    # ---- deep table, blank null, admission, evaluation
    out = os.path.join(NEG, 'deep_out')
    shutil.copytree(os.path.join(DEEP_OUT, 'out'), out)
    shutil.rmtree(os.path.join(out, 'eval'))
    fo = os.path.join(DEEP_OUT, 'out', 'forced_000.csv.gz')
    ctx = dict(frames_csv=fl, shard=0, nshards=4)
    ok, bad = all_pass('photometry_table', fo, **ctx)
    check('forced table original passes T1-T6', ok, bad)
    H = list(T.FORCED_COLUMNS) + list(T.RADIO_COLUMNS) + list(T.ENTRY_COLUMNS)
    ix = {h: i for i, h in enumerate(H)}

    def first_target(r, fn):
        i = next(k for k, x in enumerate(r) if k and x[ix['kind']] == 'target')
        r[i] = fn(list(r[i]))
        return r

    def ge4(r, fn):
        for k, x in enumerate(r):
            if k and x[ix['kind']] == 'target' and abs(float(x[ix['snr']])) >= 4:
                r[k] = fn(list(x))
                return r
        raise AssertionError('no |S/N| >= 4 row')

    def t4(x):
        x[ix['value']] = '%.5g' % (float(x[ix['value']]) * 1.5)
        x[ix['snr']] = '%.3f' % (float(x[ix['value']]) / float(x[ix['rms']]))
        if x[ix['flux_pbcorr']]:
            x[ix['flux_pbcorr']] = '%.5g' % (float(x[ix['value']]) / float(x[ix['pb']]))
        return x
    tcases = [('T1_header', lambda r: [r[0][:-1] + ['target_decl']] + r[1:]),
              ('T2_entry_join', lambda r: [r[0], r[2], r[1]] + r[3:]),
              ('T3_radio_record', lambda r: ge4(r, lambda x: x[:ix['pos_kind']] + [''] + x[ix['pos_kind'] + 1:])),
              ('T4_remeasure', lambda r: [r[0]] + [t4(list(x)) if x[ix['kind']] == 'target' else x for x in r[1:]]),
              ('T5_floors_exclusions', lambda r: first_target(r, lambda x: x[:ix['elev']] + ['10.00'] + x[ix['elev'] + 1:])),
              ('T6_arithmetic', lambda r: first_target(r, lambda x: x[:ix['snr']] + ['99.000'] + x[ix['snr'] + 1:]))]
    for test, fn in tcases:
        p = _csv_rewrite(fo, os.path.join(out, 'forced_000.csv.gz'), fn, gz=True)
        f_, det = neg('photometry_table', test, p, **ctx)
        check('negative %s fails on a corrupted forced table' % test, f_, det)
    shutil.copy(fo, os.path.join(out, 'forced_000.csv.gz'))
    bo = os.path.join(DEEP_OUT, 'out', 'blank_000.csv')
    ok, bad = all_pass('photometry_blank_null', bo)
    p = _csv_rewrite(bo, os.path.join(out, 'blank_000.csv'), lambda r: [r[0]] + [x[:5] + [x[5], str(int(x[5]) + 1)] + x[7:]
                                                                                  for x in r[1:]])
    f_, det = neg('photometry_blank_null', 'B1_counts', p)
    check('blank null: original passes B1, counts increasing with |S/N| fail', ok and f_, (bad, det))
    shutil.copy(bo, os.path.join(out, 'blank_000.csv'))
    ao = os.path.join(DEEP_OUT, 'out', 'frames_000.csv')
    ok, bad = all_pass('photometry_admission', ao, **ctx)
    p = _csv_rewrite(ao, os.path.join(out, 'frames_000.csv'),
                     lambda r: [r[0]] + [x[:5] + ['0', 'cloudy', '0', '0'] if k == 0 else x for k, x in enumerate(r[1:])])
    f_, det = neg('photometry_admission', 'A1_admission', p, **ctx)
    check('admission: original passes A1, a refusal with an unknown reason fails', ok and f_, (bad, det))
    shutil.copy(ao, os.path.join(out, 'frames_000.csv'))
    evo = os.path.join(DEEP_OUT, 'out', 'eval')
    ok, bad = all_pass('photometry_eval', evo)
    check('deep evaluation original passes E1-E4', ok, bad)
    ecases = [('E1_thresholds_or_G_grid', 'thresholds.csv',
               lambda r: [r[0]] + [x[:5] + ['%.2f' % (float(x[5]) + 0.5)] + x[6:] for x in r[1:]]),
              ('E2_lists_or_V1_kind_exclusions', 'targets_ge4.csv', lambda r: r[:-1]),
              ('E3_exceedance_or_V2_rows', 'exceedance.csv', lambda r: [r[0], r[1][:4] + [str(int(r[1][4]) + 7)] + r[1][5:]]
               + r[2:])]
    for test, fname, fn in ecases:
        e2 = os.path.join(out, 'eval')
        shutil.rmtree(e2, ignore_errors=True)
        shutil.copytree(evo, e2)
        _csv_rewrite(os.path.join(evo, fname), os.path.join(e2, fname), fn)
        f_, det = neg('photometry_eval', test, e2)
        check('negative %s fails on a corrupted deep evaluation (%s)' % (test, fname), f_, det)
    # ---- snapshot npz, manifest, evaluation, cutout, zone flags
    s1 = os.path.join(SNAP_OUT, S1, SYN_SNAP_CELL)
    nz = os.path.join(s1, 'forced_HWJ_64MHz_I.npz')
    ok, bad = all_pass('photometry_snapshot_npz', nz, cutout_root=CUTS)
    check('snapshot npz original passes N1-N5', ok, bad)
    z = dict(np.load(nz, allow_pickle=False))
    nd = os.path.join(NEG, 'stage1', SYN_SNAP_CELL)
    os.makedirs(nd)

    def put(name, **chg):
        zz = dict(z)
        for k, v in chg.items():
            if v is None:
                zz.pop(k)
            else:
                zz[k] = v
        p = os.path.join(nd, name)
        np.savez_compressed(p, **zz)
        return p
    v10 = z['val10'].copy()
    v10[:, 0, :] += 1.0
    # each corruption is written when its case runs (ONE file name: N4 / N5 read sample, band and Stokes from it)
    ncases = [('N1_shapes', lambda: put('forced_HWJ_64MHz_I.npz', bn=None)),
              ('N1_shapes', lambda: put('forced_HWJ_64MHz_I.npz', val1=z['val1'][:, :, :-1])),
              ('N2_blocks', lambda: put('forced_HWJ_64MHz_I.npz', bn=z['bn'] + 1)),
              ('N3_ring', lambda: put('forced_HWJ_64MHz_I.npz', ring_beams=z['ring_beams'] + 4.5)),
              ('N4_remeasure', lambda: put('forced_HWJ_64MHz_I.npz', val10=v10)),
              ('N5_names', lambda: put('forced_HWJ_64MHz_I.npz',
                                       names=np.array(['Nobody_%d' % i for i in range(len(z['names']))])))]
    for test, mk in ncases:
        p = mk()
        f_, det = neg('photometry_snapshot_npz', test, p, cutout_root=CUTS)
        check('negative %s fails on a corrupted stage-1 npz' % test, f_, det)
        if test != 'N5_names':
            np.savez_compressed(p, **z)
    mp = os.path.join(s1, 'forced_manifest.csv')
    ok, bad = all_pass('photometry_manifest', mp)
    check('snapshot manifest original passes M1-M3', ok, bad)
    for name in os.listdir(s1):
        if name.endswith('.npz'):
            shutil.copy(os.path.join(s1, name), os.path.join(nd, name))
    mcases = [('M1_statuses', lambda r: [r[0], r[1][:1] + ['weird'] + r[1][2:]] + r[2:]),
              ('M2_completion', lambda r: r + [[r[1][0], 'REFUSED: header CELL x', '', '']]),
              ('M3_coverage', lambda r: [r[0]] + [x for x in r[1:] if x[1] != 'ok'] + [x for x in r[1:] if x[1] == 'ok'][1:])]
    for test, fn in mcases:
        p = _csv_rewrite(mp, os.path.join(nd, 'forced_manifest.csv'), fn)
        f_, det = neg('photometry_manifest', test, p)
        check('negative %s fails on a corrupted manifest' % test, f_, det)
    evs = os.path.join(SNAP_OUT, 'eval_T1')
    ctx = dict(stage1=os.path.join(SNAP_OUT, S1), procs=1)
    ok, bad = all_pass('photometry_eval', evs, **ctx)
    check('snapshot evaluation original passes G, V1, V2, Z1, S', ok, bad)
    th_h = T.read_csv(os.path.join(evs, 'thresholds.csv'))[0]
    g4_h = T.read_csv(os.path.join(evs, 'targets_ge4.csv'))[0]
    ih = th_h.index('thr_here')

    def one_step_off(v):        # the production grid test's mutant 'a threshold one step off' (nan -> the grid bottom)
        return '3.0' if v == 'nan' else '%.2f' % (float(v) + 0.05)
    scases = [('E1_thresholds_or_G_grid', 'thresholds.csv',
               lambda r: [r[0]] + [x[:ih] + [one_step_off(x[ih])] + x[ih + 1:] for x in r[1:]]),
              ('E2_lists_or_V1_kind_exclusions', 'targets_ge4.csv',
               lambda r: [r[0], r[1][:g4_h.index('band')] + ['Full'] + r[1][g4_h.index('band') + 1:]] + r[2:]),
              ('E3_exceedance_or_V2_rows', 'thresholds.csv', lambda r: r[:-1]),
              ('Z1_zones', 'targets_ge4.csv',
               lambda r: [r[0], r[1][:g4_h.index('screen')] + ['zone' if r[1][g4_h.index('screen')] == 'clean' else 'clean']
                          + r[1][g4_h.index('screen') + 1:]] + r[2:]),
              ('Z1_zones', 'targets_ge4.csv', lambda r: r[:1])]
    for test, fname, fn in scases:
        e2 = os.path.join(NEG, 'eval_T1')
        shutil.rmtree(e2, ignore_errors=True)
        shutil.copytree(evs, e2)
        _csv_rewrite(os.path.join(evs, fname), os.path.join(e2, fname), fn)
        f_, det = neg('photometry_eval', test, e2, **ctx)
        check('negative %s fails on a corrupted snapshot evaluation (%s)' % (test, fname), f_, det)
    proto = os.path.join(NEG, 'proto', SYN_SNAP_CELL)
    os.makedirs(proto)
    for b, st_ in (('64MHz', 'I'), ('82MHz', 'I')):
        json.dump({'photometry_cutout_cubes': 40}, open(os.path.join(proto, 'runinfo_%s_%s_median.json' % (b, st_)), 'w'))
    bad_s = T.stage1_cell((SYN_SNAP_CELL, 'sb', os.path.join(SNAP_OUT, S1), os.path.dirname(proto)))[0]
    check('negative S fails: a searched band (82 MHz, 40 cutout cubes) without stage 1, and stage 1 older than its search '
          'runinfo', any('without stage 1' in b for b in bad_s) and any('older than its search' in b for b in bad_s), bad_s)
    cut = sorted(p for p in glob.glob(os.path.join(CUTS, 'HWJ', '64MHz', '*', '*_I.fits'))
                 if '/Not_A_Current_Target/' not in p)[-1]      # (that copy's OBJECT card names the real target: C3 fails)
    ok, bad = all_pass('photometry_cutout', cut)
    from astropy.io import fits
    cc = os.path.join(NEG, 'cut', 'HWJ', '64MHz', os.path.basename(os.path.dirname(cut)), os.path.basename(cut))
    os.makedirs(os.path.dirname(cc))
    with fits.open(cut) as h:
        h[0].header['CDELT2'] = 0.05
        h.writeto(cc)
    f_, det = neg('photometry_cutout', 'C1_C4_cube', cc)
    check('cutout: a synthetic cutout passes C1-C4; a wrong pixel scale fails', ok and f_, (bad, det))
    zf = os.path.join(SNAP_OUT, CFG.photometry_evaluate.ZONE_FLAGS_FILE.format(tag='T1'))
    ok, bad = all_pass('photometry_zone_flags', zf)
    zh = T.read_csv(zf)[0]
    p = _csv_rewrite(zf, os.path.join(NEG, os.path.basename(zf)),
                     lambda r: [r[0], r[1][:-1] + ['zone' if r[1][-1] == 'clean' else 'clean']] + r[2:])
    f_, det = neg('photometry_zone_flags', 'ZF1_rows', p)
    check('zone flags: original passes ZF1, a row whose recomputed screen differs fails', ok and f_ and zh == list(
        T.ZONE_FLAG_COLUMNS), (bad, det))


# ---------------------------------------------------------------------------------------------------------------------
# 7. real production products (read only) and dry runs on real items
# ---------------------------------------------------------------------------------------------------------------------

def test_real():
    s1 = os.path.join(REAL_STAGE1, REAL_SNAP_CELL)
    if not os.path.isdir(s1):
        print('skip  real products: %s not readable here' % s1)
        return
    for name in ('forced_HWJ_18MHz_I.npz', 'forced_10pc_Red_V.npz'):
        # the SERVED stage 1 (stage1_gen5, built 2026-10-25/26 before exopipe_ops 40611cc): 1-minute blocks on the FILE-NAME
        # minute, so N2 must fail on them for that reason until the stage-1 re-run (Gregg's call); every other test passes - N4
        # re-measures from the served cutouts (photometry.CUTOUT_ROOT), N5 knows the writer's J-tagged names (10 pc Red V holds
        # Gaia_DR3_WD_Unknown_ID_J1748+7053 and _J2151+5917, which the reader without the 8f2126b naming called non-current)
        res = run_tests('photometry_snapshot_npz', os.path.join(s1, name))
        n2r = res.get('N2_blocks', {})
        check('real %s/%s passes N1, N3-N5 (N4 re-measured from the archive cutouts); N2 fails as a pre-2026-10-28 product on the '
              'file-name minute' % (REAL_SNAP_CELL, name),
              all(r['passed'] for k, r in res.items() if k != 'N2_blocks') and not n2r.get('passed', True)
              and 'minute_basis' in str(n2r.get('detail', '')),
              {k: (r['passed'], str(r['detail'])[:160]) for k, r in res.items()})
    ok, bad = all_pass('photometry_manifest', os.path.join(s1, 'forced_manifest.csv'))
    check('real %s/forced_manifest.csv passes M1-M3' % REAL_SNAP_CELL, ok, bad)
    ev = '/lustre/gh/main/phase3/snapshot_forced/eval_wb_20261007'
    res = run_tests('photometry_eval', ev, max_cells=3, procs=1)
    check('real eval_wb_20261007: grid G1-G6, 8 threshold rows, stage 1 of 3 evaluated cells pass (V1 / Z1 reported: the '
          'exclusions and the gen-4 Master catalogues changed since 2026-10-07)',
          all(res[k]['passed'] for k in ('E1_thresholds_or_G_grid', 'E3_exceedance_or_V2_rows', 'S_stage1')),
          {k: (r['passed'], str(r['detail'])[:200]) for k, r in res.items()})
    print('      (report) V1: %s | Z1: %s' % (str(res['E2_lists_or_V1_kind_exclusions']['detail'])[:160],
                                             str(res['Z1_zones']['detail'])[:160]))
    prod = os.path.dirname(REAL_STAGE1)
    zf = os.path.join(prod, 'zones_recomputed_20261007.csv')
    ok, bad = all_pass('photometry_zone_flags', zf)
    check('real zones_recomputed_20261007.csv (the 2026-10-07 zone-flag product) passes ZF1', ok, bad)
    # the SERVED generation's evaluations (eval_gen5 / eval_wb_gen5 on stage1_gen5 - the S test finds that stage 1 by the
    # evaluation's tag) and their zone flags: every test passes on 3 cells
    for ev in ('eval_wb_gen5', 'eval_gen5'):
        res = run_tests('photometry_eval', os.path.join(prod, ev), max_cells=3, procs=1)
        check('real %s (served): G1-G6, V1, V2, Z1 and S (stage 1 = %s) pass on 3 cells' % (ev, S1),
              all(r['passed'] for r in res.values()) and 'stage-1 files' in str(res['S_stage1']['detail']),
              {k: (r['passed'], str(r['detail'])[:160]) for k, r in res.items()})
    ok, bad = all_pass('photometry_zone_flags', os.path.join(prod, 'zones_recomputed_gen5.csv'))
    check('real zones_recomputed_gen5.csv (the served zone flags) passes ZF1', ok, bad)
    same = {}
    for kind in ('sb', 'wb'):
        ref, _ = KS.refused_for(kind, {}, REAL_STAGE1, [])
        same[kind] = KS.refused_text(ref).encode() == open(os.path.join(prod, 'refused_cells_%s_20261007.csv' % kind),
                                                           'rb').read()
    check('the per-kind refused-cell lists the step writes from the pinned file are byte-identical to the 2026-10-07 '
          'production lists (reserve :138-147)', all(same.values()), same)
    fl = '/fast/gh/main/phase3/deep_forced/run3/frames_07.csv'
    if os.path.exists(fl):
        a, b = T.fl1_header(fl), T.fl3_pb(fl)
        check('real deep run3 frames_07.csv: FL1 passes (16,697 frames); FL3 fails as expected (built 2026-09-29, before PB '
              'step 2 gave Stokes I its own R075 PB map)', a[0] and a[1] > 16000 and not b[0] and 'PB_deep_R075_' in b[2],
              (a, b[2][:200]))
        sl = os.path.join(os.path.dirname(fl), 'stage_list.txt')
        ok, bad = all_pass('photometry_stage_list', sl)
        check('real deep run3 stage_list.txt passes SL1 (sorted unique /data07 inputs, each named by frames_07.csv, none '
              'missing)', ok, bad)
    else:
        print('skip  real deep run3 frame list (node-local to lwacalim07)')


def test_dry_real():
    out = os.path.join(TMP, 'dry_real')
    t = time.time()
    # the widebands only: a cutout glob of all 18 bands costs ~30 s of Lustre metadata (the deep dry run below covers
    # every stage of the other tier)
    rc, txt = step(['photometry', '--tier', 'snapshot', '--cells', REAL_SNAP_CELL, '--scopes', 'Red,Green,Blue', '--out',
                    out, '--work', os.path.join(TMP, 'w_dry'), '--cores', 'none', '--dry-run'])
    check('dry run on the real snapshot cell %s (widebands): 3 bands with cutouts planned, evaluate from this run\'s '
          'measure, exit 0, nothing written (%.0f s)' % (REAL_SNAP_CELL, time.time() - t),
          rc == 0 and '3 bands, 3 with cutouts' in txt and 'from the measure stage of this run' in txt
          and 'WOULD REFUSE' not in txt and not os.path.exists(out), txt[-1500:])
    par = '/data07/gh/phase3/transients/Run_P3S5/parent_catalogs'
    if not os.path.isdir(par):
        print('skip  dry run on a real deep cell (/data07 not readable here)')
        return
    with patched(KD, roots_cells=REAL_ROOTS):
        rc, txt = step(['photometry', '--tier', 'deep', '--run', 'P3S5', '--parent-dir', par, '--cells', '10h_46MHz',
                        '--out', out, '--cores', 'none', '--dry-run'])
    check('dry run on the real deep cell 10h_46MHz (Run_P3S5): every stage planned, evaluate from the measure stage of this '
          'run (no M-CHAIN refusal), exit 0, nothing written', rc == 0 and 'shard records from the measure stage of this run'
          in txt and 'WOULD REFUSE' not in txt and not os.path.exists(out), txt[-1500:])


REAL_ROOTS = KD.roots_cells


# ---------------------------------------------------------------------------------------------------------------------
# 8. the SERVED generation (the config names the cutout and stage-1 roots production serves; read only)
# ---------------------------------------------------------------------------------------------------------------------

def _served_forced_runs():
    """{(S, cadence): run} of the snapshot forced pages the review hub serves (registry <S> 10-sec-forced / 1-min-forced,
    status.REGISTRY_FILE under serve.HUB_DIR), or (None, why) - as tests/test_forced.py reads them."""
    p = CFG.status.REGISTRY_FILE.format(hub=CFG.serve.HUB_DIR)
    try:
        with open(p) as fh:
            reg = json.load(fh)
        return {(S, cad): reg[S][key]['run'] for S in ('V', 'I') for key, cad in (('10-sec-forced', '10s'),
                                                                                 ('1-min-forced', '1min'))}, None
    except (OSError, ValueError, KeyError, TypeError) as e:
        return None, '%s: %s' % (p, e)


def served_generation_mismatches():
    """[what differs] between the photometry config's generation and what production serves (empty = the config follows it):
    (1) each served snapshot forced page's own record - the FT-START line of snapshot_forced_pages_run_<run>.log - names
    P3_PHOT_ROOT = photometry.CUTOUT_ROOT and P3_FORCED_STAGE1 = <OUT_ROOT[snapshot]>/<STAGE1_DIR>; (2) the forced step names
    the same two roots (forced_snapshot.PHOT_ROOT, STAGE1_ROOT); (3) the served stage 1 measured the configured cutouts (every
    manifest row of 3 cells names a file under CUTOUT_ROOT); (4) the tier forced roots are the ones the serving pages read
    (planets_forced_page_20261026.py FT default for HWJ / 10 pc - tier_forced_page_20261027.py reads it -, and
    planets_page_orbit_build_20261028.sh P3_FORCED_TIERS for the planets), and their manifests name cutouts under CUTOUT_ROOT."""
    ph, fs = CFG.photometry, CFG.forced_snapshot
    out = []
    stage1 = os.path.join(ph.OUT_ROOT['snapshot'], ph.STAGE1_DIR)
    runs, why = _served_forced_runs()
    if not runs:
        return ['the review hub registry: %s' % why]
    for (S, cad), run in sorted(runs.items()):
        log = os.path.join(CFG.paths.OPS_ROOT, 'snapshot_forced_pages_run_%s.log' % run)
        start = [ln for ln in open(log, errors='replace') if ('FT-START %s ' % run) in ln] if os.path.exists(log) else []
        if not start:
            out.append('%s: no FT-START record in %s' % (run, log))
            continue
        for var, want in (('P3_PHOT_ROOT', ph.CUTOUT_ROOT), ('P3_FORCED_STAGE1', stage1)):
            m = re.search(r' %s=(\S+)' % var, start[-1])
            if not m or m.group(1) != want:
                out.append('%s (Stokes %s %s) was built with %s=%s, the config says %s' % (run, S, cad, var, m and m.group(1),
                                                                                           want))
    if fs.PHOT_ROOT != ph.CUTOUT_ROOT or fs.STAGE1_ROOT != stage1:
        out.append('forced_snapshot.PHOT_ROOT / STAGE1_ROOT %s / %s != photometry %s / %s' % (fs.PHOT_ROOT, fs.STAGE1_ROOT,
                                                                                            ph.CUTOUT_ROOT, stage1))
    pre = ph.CUTOUT_ROOT.rstrip('/') + '/'
    cells = sorted(d for d in os.listdir(stage1) if os.path.isfile(os.path.join(stage1, d, 'forced_manifest.csv')))[::120][:3] \
        if os.path.isdir(stage1) else []
    if not cells:
        out.append('no served stage-1 cell under %s' % stage1)
    for cl in cells:
        rows = T.read_csv(os.path.join(stage1, cl, 'forced_manifest.csv'))[1]
        bad = [r['file'] for r in rows if not r['file'].startswith(pre)]
        if bad or not rows:
            out.append('served stage 1 %s: %d of %d rows measured cutouts outside %s (e.g. %s)' % (cl, len(bad), len(rows),
                                                                                                  ph.CUTOUT_ROOT, bad[:1]))
    src = open(os.path.join(OPS, 'planets_forced_page_20261026.py')).read()
    m = re.search(r"^FT = os\.environ\.get\('P3_FORCED_TIERS', '([^']+)'\)", src, re.M)
    tp = open(os.path.join(OPS, 'tier_forced_page_20261027.py')).read()
    for smp in ('HWJ', '10pc'):
        if not m or m.group(1) != ph.TIER_FORCED_ROOTS[smp] or 'import planets_forced_page_20261026 as PP' not in tp:
            out.append('the HWJ / 10 pc tier pages read %s, TIER_FORCED_ROOTS[%s] = %s' % (m and m.group(1), smp,
                                                                                         ph.TIER_FORCED_ROOTS[smp]))
    drv = open(os.path.join(OPS, 'planets_page_orbit_build_20261028.sh')).read()
    m = re.search(r'P3_FORCED_TIERS=(\S+)', drv)
    if not m or m.group(1) != ph.TIER_FORCED_ROOTS[ph.PLANET_SAMPLE]:
        out.append('the planets page reads %s, TIER_FORCED_ROOTS[%s] = %s' % (m and m.group(1), ph.PLANET_SAMPLE,
                                                                            ph.TIER_FORCED_ROOTS[ph.PLANET_SAMPLE]))
    for smp, root in sorted(ph.TIER_FORCED_ROOTS.items()):
        for tier in ('10min', 'deep'):
            rows, lst = [], None
            for mp in sorted(glob.glob(os.path.join(root, tier, '[0-2][0-9]h', 'forced_manifest.csv'))):
                rows = [r for r in T.read_csv(mp)[1] if r['file'].split('/')[-4] == smp]
                if rows:            # the first LST whose manifest holds rows of the sample (planets cross only some hours)
                    lst = os.path.basename(os.path.dirname(mp))
                    break
            bad = [r['file'] for r in rows if not r['file'].startswith(pre)]
            if bad or not rows:
                out.append('%s %s %s: %d of %d %s rows measured cutouts outside %s' % (root, tier, lst, len(bad), len(rows),
                                                                                      smp, ph.CUTOUT_ROOT))
    return out


def test_served_generation():
    bad = served_generation_mismatches()
    check('served generation: the photometry config names the roots production serves - every served snapshot forced page '
          'was built on photometry.CUTOUT_ROOT and <OUT_ROOT[snapshot]>/STAGE1_DIR (FT-START), forced_snapshot names the same, '
          'the served stage 1 measured those cutouts, the tier forced roots are the ones the tier / planets pages read',
          not bad, bad)
    with patched(CFG.photometry, CUTOUT_ROOT=CFG.photometry.RETIRED_CUTOUT_ROOTS[0], STAGE1_DIR='stage1'):
        bad4 = served_generation_mismatches()
    check('served generation falsified: with generation 4\'s roots (PHASE3_ARCHIVE_ROOT/photometry, <out>/stage1 - what the '
          'config named before) the check fails on every served page (both roots), the forced step and the tier manifests',
          sum('P3_PHOT_ROOT' in b for b in bad4) == 4 and sum('P3_FORCED_STAGE1' in b for b in bad4) == 4
          and any('forced_snapshot' in b for b in bad4) and any('rows measured cutouts outside' in b for b in bad4), bad4)


# ---------------------------------------------------------------------------------------------------------------------
# 9. cutouts: mirrors and ports of the in-place engines and the production content tests they port (read only)
# ---------------------------------------------------------------------------------------------------------------------

from exopipe.phase3.lib.photometry import steps_cutouts as SC     # noqa: E402
from exopipe.phase3.lib.photometry import tests_cutouts as TC     # noqa: E402

TIER_TOOL = os.path.join(OPS, 'photometry_tier_cutouts_20261025.py')
TIER_FORCED_TOOL = os.path.join(OPS, 'photometry_tier_forced_20261026.py')


def test_cutout_mirrors():
    cc, ph = CFG.photometry_cutouts, CFG.photometry
    L, src = mod_literals(TIER_TOOL), open(TIER_TOOL).read()
    check('mirror photometry_tier_cutouts_20261025.py: its inputs (DEEP, TROOT, SB), the retired root it refuses (OLD_ROOT), '
          'the planets through the 10-s step\'s selector (e1e24ff), the content test seed',
          L.get('DEEP') == cc.TIER_DEEP_RUN and L.get('TROOT') == cc.TIER_TENMIN_ROOT and L.get('SB') == cc.TIER_TENMIN_SEARCH
          and L.get('OLD_ROOT') == ph.RETIRED_CUTOUT_ROOTS[0] and 'for body in CUT._planets():' in src
          and 'random.Random(%d)' % cc.TIER_RECUT_SEED in src,
          {k: L.get(k) for k in ('DEEP', 'TROOT', 'SB', 'OLD_ROOT')})
    cols = re.findall(r"fits\.Column\(name=(?:'searched' if tier == '10min' else 'deep_refused'|'([a-z_0-9]+)')", src)
    five = "fits.Column(name='searched' if tier == '10min' else 'deep_refused'" in src
    want = {t: list(v) for t, v in cc.TIER_TIMES_COLUMNS.items()}
    got = {t: [c or ('searched' if t == 'sublst' else 'deep_refused') for c in cols] for t in want}
    cards = [k for k in cc.TIER_HEADER_CARDS if "hh['%s']" % k not in src]
    check('format photometry_tier_cutouts_20261025.py: TIMES columns per tier (searched / deep_refused 5th), header cards, '
          'tier tokens = photometry_cutouts.TIER_TIMES_COLUMNS, TIER_HEADER_CARDS, TIER_TOKEN',
          five and got == want and not cards and "choices=('deep', '10min', 'both')" in src
          and set(cc.TIER_TOKEN.values()) == {'deep', '10min'}, (got, want, cards))
    LF, fsrc = mod_literals(TIER_FORCED_TOOL), open(TIER_FORCED_TOOL).read()
    check('mirror photometry_tier_forced_20261026.py: SAMPLES, MOVING, its manifest header and completion line',
          tuple(LF.get('SAMPLES') or ()) == tuple(ph.TIER_FORCED_SAMPLES) and tuple(LF.get('MOVING') or ()) == (ph.PLANET_SAMPLE,)
          and "fieldnames=%r" % list(TC.TIER_MANIFEST_HEADER) in fsrc
          and 'PHOTOMETRY-TIER-FORCED-DONE %d cutouts measured, %d refused' in fsrc, (LF.get('SAMPLES'), LF.get('MOVING')))
    ss = open(os.path.join(CLV, 'phase3_snapshot_search.py')).read()
    gs = open(os.path.join(CLV, 'phase3_snapshot_group_search.py')).read()
    cs = open(os.path.join(CLV, 'phase3_snapshot_cutouts.py')).read()
    fp = open(os.path.join(OPS, 'snapshot_forced_phot.py')).read()
    texts = [(ss, 'log.info("%sV %%s I %%s%%s"' % cc.REGEN_DONE_SUBBAND.replace('REGEN ', 'REGEN %s %s %s: ')),
             (ss, 'raise SystemExit(%d)' % cc.REGEN_EXCLUDED_RC),
             (ss, 'CUBE_ROOT = os.environ.get("P3_SNAP_CUBE_ROOT", "%s")' % CFG.paths.SNAPSHOT_CUBES_ROOT),
             (ss, 'SEARCH_OUT_ROOT = os.environ.get("P3_SNAP_SEARCH_OUT_ROOT", "%s")' % CFG.paths.SNAPSHOT_FLEET_OUT),
             (ss, 'raise RuntimeError("REFUSED: regen_cutouts needs P3_SNAP_CUTOUT_ROOT'),
             (gs, 'log.info("%s%%s %%s %%s %%s: %%s cutout cubes"' % cc.REGEN_DONE_WIDEBAND),
             (gs, 'raise SystemExit(%d if r == "excluded" else 0)' % cc.REGEN_EXCLUDED_RC),
             (cs, 'env = os.environ.get("%s", "")' % ph.ADHOC_ENV),
             (cs, 'os.environ.get("%s") == "1"' % ph.ADHOC_PLANETS_ENV),
             (cs, 'jobs.append(("%s", body.capitalize(), pos))' % ph.PLANET_SAMPLE),
             (fp, "os.environ['%s'].split(',')" % ph.ADHOC_ENV)]
    miss = [t for s_, t in texts if t not in s_]
    check('mirror the regeneration engines and the ad-hoc switches: CUTOUTS-REGEN / -WIDEBAND lines, exit 3 on exclusion, the '
          'cube and search-output roots (= paths.SNAPSHOT_CUBES_ROOT / SNAPSHOT_FLEET_OUT, passed explicitly), the required '
          'root, P3_SNAP_CUTOUT_SAMPLES / _PLANETS in the writer and stage 1, the planet sample name', not miss, miss)
    rt = open(os.path.join(OPS, 'snapshot_cutout_regen_20261025_test.py')).read()
    Lr = mod_literals(os.path.join(OPS, 'snapshot_cutout_regen_20261025_test.py'))
    check('mirror snapshot_cutout_regen_20261025_test.py (the content test the identity check ports): TOL, 2 x eps32, the '
          'weights\' first-order bound 5e-6 x (A + |S|) (bd8bf26)',
          Lr.get('TOL') == cc.IDENTITY_TOL and '- 2 * EPS32 * (np.abs(box_) + np.abs(Rm)[None])' in rt
          and cc.EPS32_FACTOR == 2.0 and 'R - Rm[None]) - 5e-6 * (A_ + np.abs(S_) + np.nanmedian(A_ + np.abs(S_), axis=0)[None])'
          in rt and cc.WEIGHT_REL_PRECISION == 5e-6, Lr.get('TOL'))
    pys = {}
    for drv, key in (('snapshot_cutout_regen_20261025.sh', 'cutouts_subband'), ('snapshot_cutout_regen_wb_20261025.sh',
                                                                                 'cutouts_wideband')):
        m = re.search(r'PY=(\S+)', open(os.path.join(OPS, drv)).read())
        pys[key] = (m and m.group(1), CFG.site.PYTHON[ph.ENGINE_PYTHON_BY_KEY[key]])
    check('the regeneration engines run under the interpreter their drivers named (photometry.ENGINE_PYTHON_BY_KEY; the tier '
          'tools under py38_orca_nkosogor, measured byte for byte - parity run)', all(a == b for a, b in pys.values())
          and ph.ENGINE_PYTHON_BY_KEY['cutouts_tier'] == 'default', pys)
    for key in ('cutouts_subband', 'cutouts_wideband', 'cutouts_tier', 'cutmeasure_tier'):
        check('engine %s resolves to exactly one in-place file (%s)' % (key, ph.ENGINE[key]), os.path.isfile(K.engine(key)))


def test_cutout_ports():
    import importlib
    os.environ.pop(CFG.photometry.ADHOC_ENV, None)
    CUT = importlib.import_module('phase3_snapshot_cutouts')
    CUT._TARGETS = None
    prod = sorted((t, CUT.safe_name(n), ra, dec) for t, n, ra, dec in CUT._targets())
    mine = sorted((t, T.safe_name(n), ra, dec) for t, n, ra, dec in T.sample_targets())
    tagged = sorted(n for t, n, _, _ in mine if '_J' in n and 'Unknown' in n)
    check('port: tests.sample_targets = phase3_snapshot_cutouts._targets on the real lists, distinct objects of one name '
          'J-tagged (8f2126b; %d entries, %d J-tagged)' % (len(prod), len(tagged)), prod == mine and len(tagged) >= 2,
          (len(prod), len(mine), [x for x in prod if x not in mine][:3]))
    adhoc = os.path.join(TMP, 'adhoc_port.csv')
    with open(adhoc, 'w') as f:
        f.write('name,ra,dec\nStar A,10.0,20.0\nStar A,10.5,20.0\nStar B,30.0,-5.0\nNo Position,x,y\n')
    try:
        os.environ[CFG.photometry.ADHOC_ENV] = 'ADHOC=%s' % adhoc
        CUT._TARGETS = None
        prod = sorted(CUT._targets())
    finally:
        os.environ.pop(CFG.photometry.ADHOC_ENV, None)
        CUT._TARGETS = None
    mine = sorted(T.sample_targets(lists={'ADHOC': adhoc}))
    check('port: an ad-hoc list (name / ra / dec columns, a name given to two objects 30 arcmin apart, an unparsable row) is '
          'read as the writer reads it', prod == mine and len(mine) == 3 and mine[0][1].startswith('Star A J'), (prod, mine))
    # one frame per integration (phase3_snapshot_search.flowcube_frame_index), the tool's box, the regen test's rrms
    import logging
    nz = os.path.join(TMP, 'fleet_out', '18h_2025-04-13')
    os.makedirs(nz, exist_ok=True)
    tss = ['20250413_%06d' % (10 * i) for i in range(6)]
    from astropy.io import fits
    hd = [fits.Header({'DATE-OBS': '2025-04-13T00:00:%02d' % (10 * (i % 4))}) for i in range(6)]   # 2 repeated integrations
    with open(os.path.join(nz, 'noise_64MHz_V_median.csv'), 'w') as f:
        f.write('ts,x\n' + ''.join('%s,1\n' % t for t in tss[:4]))
    prod_fi = _prod_function(os.path.join(CLV, 'phase3_snapshot_search.py'), 'flowcube_frame_index',
                             {'os': os, 'csv': csv, 'SEARCH_OUT_ROOT': os.path.join(TMP, 'fleet_out'),
                              'log': logging.getLogger('x')})
    a = prod_fi('18h', '2025-04-13', '64MHz', 'V', tss, hd)
    b = TC.frame_index('18h', '2025-04-13', '64MHz', 'V', tss, hd, os.path.join(TMP, 'fleet_out'))
    whole = TC.frame_index('18h', '2025-04-13', '64MHz', 'V', tss[:4], hd[:4], os.path.join(TMP, 'fleet_out'))
    try:
        TC.frame_index('18h', '2025-04-13', '64MHz', 'I', tss, hd, os.path.join(TMP, 'fleet_out'))
        refused = False
    except ValueError:
        refused = True
    check('port: tests_cutouts.frame_index = phase3_snapshot_search.flowcube_frame_index (a cube repeating integrations '
          'restricted to the search\'s frame list; whole without repeats; refused without that list)',
          a == b == [0, 1, 2, 3] and whole == [0, 1, 2, 3] and refused, (a, b, whole))
    rng = np.random.default_rng(9)
    img = rng.normal(0, 1, (1, 60, 50)).astype(np.float32)
    prod_box = _prod_function(TIER_TOOL, 'box', {'np': np})
    ok = True
    for px, py in ((25.3, 30.7), (2.2, 3.9), (48.6, 58.1), (-5.0, 10.0), (70.0, 70.0)):
        p_, x0, y0 = prod_box(img, 60, 50, px, py, 21, (0,))
        m_ = TC.tier_box(img, 60, 50, px, py, 21, (0,))
        ok &= bool(np.array_equal(np.isnan(p_), np.isnan(m_)) and np.array_equal(p_[np.isfinite(p_)], m_[np.isfinite(m_)]))
    rr = _prod_function(os.path.join(OPS, 'snapshot_cutout_regen_20261025_test.py'), 'rrms', {'np': np})
    a_ = rng.normal(0, 2, 500).astype(np.float32)
    a_[:20] = np.nan
    check('port: tests_cutouts.tier_box = the tier tool\'s box (inside, at and beyond the edges) and rrms = the regeneration '
          'test\'s rrms', ok and rr(a_) == TC.rrms(a_) and np.isnan(TC.rrms(a_[:40])) and np.isnan(rr(a_[:40])))


# ---------------------------------------------------------------------------------------------------------------------
# 10. cutouts: synthetic consolidated flow cubes, the identity tests (subband, the weighted-stack rebuild bound), one file
#     per key, the snapshot / tier stages end to end (FAKE engines that call the REAL writers on the synthetic inputs)
# ---------------------------------------------------------------------------------------------------------------------

CUBES = os.path.join(TMP, 'cubes')
CUT_SB = '64MHz'
WB_GROUP, WB_SBS = 'Blue', ('64MHz', '69MHz', '73MHz')
NPX_C, CD_C, NT_C = 300, 0.05, 6                 # cube pixels, deg / pixel, frames
CRA, CDEC = 275.0, 60.0                           # 23 deg from the zenith at LST 18h: HD 163607, HIP 91258, Struve 2398 A


def cube_header(k, ts, cd=CD_C, n=NPX_C):
    from astropy.io import fits
    h = fits.Header()
    h['NAXIS'], h['NAXIS1'], h['NAXIS2'] = 2, n, n
    for kk, v in wcs_header(n=n, cd=cd, ra=CRA, dec=CDEC).items():
        h[kk] = v
    h['BMAJ'], h['BMIN'], h['BPA'] = 0.2, 0.15, 0.0
    h['DATE-OBS'] = '%s-%s-%sT%s:%s:%s.000' % (ts[0:4], ts[4:6], ts[6:8], ts[9:11], ts[11:13], ts[13:15])
    h['MJD-MID'] = 60778.0 + k * 10.031 / 86400.0
    return h


def cube_ts(n=NT_C):
    """Frame times of the synthetic cell 18h 2025-04-13 (LST 18h at OVRO is ~12:27 UTC that day), 10 s apart."""
    return ['20250413_12%02d%02d' % (27 + (k * 10) // 60, (k * 10) % 60) for k in range(n)]


def make_cube(cell, sb, stokes, seed, scale=1.0, offset=0.0, n=NPX_C, nt=NT_C):
    """A consolidated flow cube of the search's layout (FRAMES [T, Y, X] float32, FRAME_TABLE ts + newline-separated header
    cards): a fixed sky (the hour-median model) plus per-frame noise."""
    from astropy.io import fits
    rng = np.random.default_rng(seed)
    sky = (rng.normal(0, 1, (n, n)) * scale + offset).astype(np.float32)
    frames = (sky[None] + rng.normal(0, 0.3 * abs(scale) + 0.01, (nt, n, n))).astype(np.float32)
    ts = cube_ts(nt)
    hdrs = [cube_header(k, ts[k], n=n).tostring(sep='\n') for k in range(nt)]
    d = os.path.join(CUBES, cell)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, '%s_%s_%s_flowcube.fits' % (cell, sb, stokes))
    tab = fits.BinTableHDU.from_columns([fits.Column(name='ts', format='15A', array=np.array(ts)),
                                         fits.Column(name='header', format='%dA' % max(len(x) for x in hdrs),
                                                     array=np.array(hdrs))], name='FRAME_TABLE')
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(frames, name='FRAMES'), tab]).writeto(p, overwrite=True)
    return p, frames, ts


def write_test_cutout(path, cube, model, x0, y0, n, ts, cell, band, stokes, sample, obj, h0):
    """A 10-s cutout of the writer's layout (fixed target): plane k = cube_k - model over the box with corner (x0, y0), the
    WCS of the first frame shifted to the box."""
    from astropy.io import fits
    from astropy.wcs import WCS
    out = (cube[:, y0:y0 + n, x0:x0 + n] - model[None, y0:y0 + n, x0:x0 + n]).astype(np.float32)
    h = WCS(h0).celestial.to_header()
    h['CRPIX1'], h['CRPIX2'] = float(h['CRPIX1']) - x0, float(h['CRPIX2']) - y0
    lst, date = cell.split('_', 1)
    for k, v in (('OBJECT', obj), ('SAMPLE', sample), ('LST', lst), ('DATE', date), ('BAND', band), ('STOKES', stokes),
                 ('CELL', cell), ('UTCSTART', ts[0][:8] + 'T' + ts[0][9:15]), ('TARGRA', 0.0), ('TARGDEC', 0.0),
                 ('CUTDEG', 5.0)):
        h[k] = v
    nt = cube.shape[0]
    cols = [fits.Column(name='ts', format='15A', array=np.array(ts)),
            fits.Column(name='date_obs', format='26A', array=np.array([''] * nt)),
            fits.Column(name='admitted', format='L', array=np.ones(nt, bool)),
            fits.Column(name='global_rms_jyb', format='E', array=np.ones(nt, np.float32)),
            fits.Column(name='ra_deg', format='D', array=np.zeros(nt)), fits.Column(name='dec_deg', format='D', array=np.zeros(nt)),
            fits.Column(name='target_x', format='E', array=np.full(nt, n / 2.0, np.float32)),
            fits.Column(name='target_y', format='E', array=np.full(nt, n / 2.0, np.float32))]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(out, header=h), fits.BinTableHDU.from_columns(cols, name='TIMES')]).writeto(path, overwrite=True)
    return path


def test_identity():
    """The identity tests on synthetic cubes: a subband cutout equals its cube (minus the model) and fails when one plane is
    shifted or a 5-sigma source is added; a wideband cutout equals the stack rebuilt from its member cubes and the job's
    member table, whose '%.6g' weights the EXACT first-order bound covers while the stack-magnitude bound does not."""
    from astropy.io import fits
    cell = '18h_2025-04-13'
    with patched(CFG.paths, SNAPSHOT_CUBES_ROOT=CUBES):
        p, frames, ts = make_cube(cell, '82MHz', 'V', 1)
        model = np.median(frames, axis=0)
        h0 = fits.Header.fromstring(str(fits.getdata(p, 'FRAME_TABLE')['header'][0]), sep='\n')
        root = os.path.join(TMP, 'ident', 'root')
        good = write_test_cutout(os.path.join(root, 'HWJ', '82MHz', 'Obj', 'Obj_18h_20250413T122700_V.fits'), frames, model,
                                 100, 120, 60, ts, cell, '82MHz', 'V', 'HWJ', 'Obj', h0)
        shifted = frames.copy()
        shifted[3] = np.roll(frames[3], 1, axis=1)
        bad1 = write_test_cutout(os.path.join(root, 'HWJ', '82MHz', 'Obj2', 'Obj2_18h_20250413T122700_V.fits'), shifted,
                                 model, 100, 120, 60, ts, cell, '82MHz', 'V', 'HWJ', 'Obj2', h0)
        src = frames.copy()
        src[2, 150, 130] += 5.0 * 0.3
        bad2 = write_test_cutout(os.path.join(root, 'HWJ', '82MHz', 'Obj3', 'Obj3_18h_20250413T122700_V.fits'), src, model,
                                 100, 120, 60, ts, cell, '82MHz', 'V', 'HWJ', 'Obj3', h0)
        res = {os.path.basename(f): (ok, v) for f, ok, v, _ in TC.identity_subband([good, bad1, bad2], cell, '82MHz', 'V')}
    check('identity (subband): a cutout = cube - model passes (%.1e x rms); a plane shifted by one pixel and a 5-sigma '
          'source in one plane fail' % res[os.path.basename(good)][1],
          res[os.path.basename(good)][0] and not res[os.path.basename(bad1)][0] and not res[os.path.basename(bad2)][0], res)
    # ---- wideband: members of +-1000 Jy that cancel, weights stored to 6 significant digits (22h 2025-08-16 Green V,
    #      22h 2025-08-17 Blue V): the rebuilt stack differs from the group frame by up to 5e-6 x (A + |S|)
    rng = np.random.default_rng(4)
    n, nt = 80, NT_C
    with patched(CFG.paths, SNAPSHOT_CUBES_ROOT=CUBES):
        cubes = {}
        for i, (sb, off) in enumerate(zip(WB_SBS, (1000.0, -1000.0, 0.0))):
            cubes[sb] = make_cube(cell, sb, 'V', 10 + i, scale=1.0, offset=off, n=n)[1]
        w = {('64MHz', k): 1.0000049 if k % 2 == 0 else 1.0000001 for k in range(nt)}   # '%.6g' -> 1: errors 4.9e-6 / 1e-7
        w.update({('69MHz', k): 1.0 for k in range(nt)})
        w.update({('73MHz', k): 1e-3 * (1 + 0.1 * k) for k in range(nt)})
        stack = np.zeros((nt, n, n))
        for (sb, k), wt in w.items():
            stack[k] += wt * cubes[sb][k].astype(np.float64)
        stack = (stack / np.array([sum(w[(sb, k)] for sb in WB_SBS) for k in range(nt)])[:, None, None]).astype(np.float32)
        model = np.median(stack, axis=0)
        mem = os.path.join(TMP, 'ident', 'wideband_members_V.csv')
        with open(mem, 'w') as f:
            f.write('group,ts,subband,member_ts,weight\n' + ''.join(
                '%s,%s,%s,%s,%.6g\n' % (WB_GROUP, ts[k], sb, ts[k], w[(sb, k)]) for sb in WB_SBS for k in range(nt)))
        h0 = cube_header(0, ts[0], n=n)
        wgood = write_test_cutout(os.path.join(root, 'HWJ', WB_GROUP, 'W1', 'W1_18h_20250413T122700_V.fits'), stack, model,
                                  10, 12, 50, ts, cell, WB_GROUP, 'V', 'HWJ', 'W1', h0)
        inj = stack.copy()
        rms5 = 5.0 * TC.rrms(stack[:, 12:62, 10:60] - model[None, 12:62, 10:60])
        inj[4, 30, 30] += rms5
        wbad = write_test_cutout(os.path.join(root, 'HWJ', WB_GROUP, 'W2', 'W2_18h_20250413T122700_V.fits'), inj, model,
                                 10, 12, 50, ts, cell, WB_GROUP, 'V', 'HWJ', 'W2', h0)
        res = {os.path.basename(f): (ok, v) for f, ok, v, _ in TC.identity_wideband([wgood, wbad], cell, WB_GROUP, 'V', mem)}
        tss, stacks, boxes, mabs = TC.wideband_stack(cell, 'V', mem, WB_GROUP, [wgood])
        cutg = fits.getdata(wgood).astype(np.float32)
        exact = TC.wideband_identity(cutg, stacks[wgood], mabs[wgood], False)
        none_ = TC.wideband_identity(cutg, stacks[wgood], mabs[wgood], False, rel=0.0)
        R = stacks[wgood] - cutg
        old = float(np.nanmax(np.abs(R - np.nanmedian(R, axis=0)[None]) - 4e-6 * np.abs(stacks[wgood]))) / TC.rrms(cutg)
    tol = CFG.photometry_cutouts.IDENTITY_TOL
    check('identity (wideband, the corrected rebuild bound): members of +-1000 Jy with %%.6g weights - the exact first-order '
          'bound 5e-6 x (A + |S|) passes (%.1e x rms), no allowance fails (%.1e), the first bound 4e-6 x |S| (before '
          '9dea97a: the stack magnitude alone) fails (%.1e), a 5-sigma source in one plane fails' % (exact, none_, old),
          res[os.path.basename(wgood)][0] and exact <= tol < none_ and old > tol and not res[os.path.basename(wbad)][0],
          (res, exact, none_, old))


def make_cutout_file(root, smp, band, obj, lst, stamp, stokes, cell):
    """A minimal cutout of the 10-s naming (header CELL / DATE / LST ...) for the one-file-per-key tests."""
    from astropy.io import fits
    p = os.path.join(root, smp, band, obj, '%s_%s_%s_%s.fits' % (obj, lst, stamp, stokes))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    h = fits.Header()
    h['CELL'], h['LST'], h['DATE'] = cell, lst, cell.split('_', 1)[1]
    fits.PrimaryHDU(np.zeros((1, 2, 2), np.float32), header=h).writeto(p, overwrite=True)
    return p


def test_one_file_per_key():
    root = os.path.join(TMP, 'opk')
    cell = '12h_2024-12-31'
    keep = ['HWJ/46MHz/A/A_12h_20241231T131122_I.fits', 'HWJ/46MHz/A/A_12h_20241231T131122_V.fits',
            'HWJ/46MHz/B/B_12h_20241231T131122_I.fits']
    for rel in keep:
        q = rel.split('/')
        make_cutout_file(root, q[0], q[1], q[2], '12h', q[3].split('_')[2], q[3][-6], cell)
    stale = make_cutout_file(root, 'HWJ', '46MHz', 'A', '12h', '20241231T131121', 'I', cell)       # 12h 2024-12-31's defect
    other_night = make_cutout_file(root, 'HWJ', '46MHz', 'A', '12h', '20241231T235959', 'I', '12h_2025-01-01')
    orphan = make_cutout_file(root, 'HWJ', '46MHz', 'C', '12h', '20241231T131122', 'V', cell)
    other_smp = make_cutout_file(root, 'ADHOC', '46MHz', 'D', '12h', '20241231T131122', 'V', cell)
    st, orph = TC.superseded(root, '10s', cell, '46MHz', ['HWJ', '10pc', 'SolarSystem'], keep)
    rel = lambda p: os.path.relpath(p, root)
    check('one file per key: another time stamp of a written (object, LST, date, Stokes) is stale (T131121 beside T131122), '
          'an object the unit did not write is an orphan; a file of another night (header CELL) or of another sample is '
          'neither', st == [rel(stale)] and orph == [rel(orphan)], (st, orph, rel(other_night), rel(other_smp)))
    man = TC.write_manifest(os.path.join(TC.manifest_dir(root, '10s'), '%s_46MHz.csv' % cell),
                            [dict(file=r, sha256=TC.sha256_file(os.path.join(root, r)), bytes=os.path.getsize(os.path.join(root, r)),
                                  sample='HWJ', object=r.split('/')[2], stokes=r[-6]) for r in keep])
    r1 = run_tests('photometry_cutout_manifest', man, samples=['HWJ', '10pc', 'SolarSystem'])
    for p in (stale, orphan):
        os.remove(p)
    r2 = run_tests('photometry_cutout_manifest', man, samples=['HWJ', '10pc', 'SolarSystem'])
    check('CM2 fails while the stale and the orphan file are in the root and passes once they are gone',
          not r1['CM2_one_per_key']['passed'] and r2['CM2_one_per_key']['passed'],
          (r1['CM2_one_per_key']['detail'], r2['CM2_one_per_key']['detail']))
    check('CM1 fails on the minimal files (no WCS, no TIMES: the structure test C1-C4 is part of CM1, each file named) - the '
          'e2e manifests pass it', not r2['CM1_files']['passed'] and 'structure test raised' in str(r2['CM1_files']['detail']),
          r2['CM1_files']['detail'])


FAKE_REGEN = r'''#!/usr/bin/env python
"""FAKE phase3_snapshot_search.py --cutouts-from-cubes for tests/test_photometry.py: the synthetic consolidated cube of the cell /
subband (P3_SNAP_CUBE_ROOT), the hour-median model, every frame admitted, identity screens - then the REAL writer
(clv15 phase3_snapshot_cutouts.write_cutouts) into P3_SNAP_CUTOUT_ROOT, V then I, and the regeneration's CUTOUTS-REGEN line.
fake_params.json: 'excluded' cells exit 3 (EXCLUDED:), 'corrupt' shifts plane 1 of every written cutout by one pixel."""
import argparse, glob, json, os, sys
sys.path.insert(0, '/lustre/gh/main/clv15')
import numpy as np
from astropy.io import fits
P = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fake_params.json')))
ap = argparse.ArgumentParser()
for k in ('--lst', '--date', '--subband', '--stokes', '--mode', '--out'):
    ap.add_argument(k)
ap.add_argument('--cutouts-from-cubes', action='store_true')
a = ap.parse_args()
cell = '%s_%s' % (a.lst, a.date)
if cell in P.get('excluded', []):
    print("EXCLUDED: %s %s %s carries verdict 'exclude' - REFUSING to search (no output written)" % (a.lst, a.date, a.subband))
    raise SystemExit(3)
import phase3_snapshot_cutouts as CUT
root = os.environ['P3_SNAP_CUTOUT_ROOT']
os.makedirs(a.out, exist_ok=True)
n = {}
for S in ('V', 'I'):
    p = os.path.join(os.environ['P3_SNAP_CUBE_ROOT'], cell, '%s_%s_%s_flowcube.fits' % (cell, a.subband, S))
    with fits.open(p) as h:
        cube = np.array(h['FRAMES'].data, np.float32)
        t = h['FRAME_TABLE'].data
        tss = [str(x) for x in t['ts']]
        hdrs = [fits.Header.fromstring(str(x), sep='\n') for x in t['header']]
    model = np.median(cube, axis=0).astype(np.float32)
    meta = [dict(ts=ts, date_obs=str(hd['DATE-OBS'])) for ts, hd in zip(tss, hdrs)]
    z = np.zeros(cube.shape[1:], np.float32)
    n[S] = CUT.write_cutouts(cube, model, meta, [True] * len(tss), hdrs[0], z, z, a.lst, a.date, a.subband, S, root=root)
    if P.get('corrupt'):
        for f in glob.glob(os.path.join(root, '*', a.subband, '*', '*_%s.fits' % S)):
            with fits.open(f, mode='update') as h:
                h[0].data[1] = np.roll(h[0].data[1], 1, axis=1)
print('2026-11-02 00:00:00,000 INFO CUTOUTS-REGEN %s %s %s: V %s I %s' % (a.lst, a.date, a.subband, n['V'], n['I']), flush=True)
'''


def setup_fake_regen(excluded=(), corrupt=False):
    p = os.path.join(FAKE_DIR, 'fake_params.json')
    prm = json.load(open(p)) if os.path.exists(p) else {}
    prm.update(excluded=list(excluded), corrupt=corrupt)
    json.dump(prm, open(p, 'w'))
    fp = os.path.join(FAKE_DIR, CFG.photometry.ENGINE['cutouts_subband'])
    open(fp, 'w').write(FAKE_REGEN)
    FAKES['cutouts_subband'] = fp


CUT_OUT = os.path.join(TMP, 'cut_out')
CUT_ROOT = os.path.join(CUT_OUT, 'photometry_genT')
CUT_CELL = '18h_2025-04-13'
CUT_COMMON = ['--tier', 'snapshot', '--stages', 'cutouts', '--cores', 'none', '--set', 'paths.SNAPSHOT_CUBES_ROOT=%s' % CUBES]


def test_cutouts_snapshot():
    """The snapshot cutouts stage through its engine seam: a FAKE regeneration engine that runs the REAL writer on synthetic
    consolidated cubes (the default HWJ / 10 pc lists and the planets: what lands in the synthetic image)."""
    make_cube(CUT_CELL, CUT_SB, 'V', 21)
    make_cube(CUT_CELL, CUT_SB, 'I', 22)
    setup_fake_regen()
    argv = ['photometry'] + CUT_COMMON + ['--cells', '%s_%s' % (CUT_CELL, CUT_SB), '--cutout-root', CUT_ROOT, '--out', CUT_OUT]
    rc, txt = step(argv + ['--dry-run'])
    check('cutouts --dry-run: one unit planned, nothing written', rc == 0 and 'unit cutouts.%s.%s' % (CUT_CELL, CUT_SB) in txt
          and not os.path.exists(CUT_OUT), txt)
    rc, txt = step(argv)
    rec = records(CUT_OUT).get('cutouts.%s.%s' % (CUT_CELL, CUT_SB), {})
    man = [o['path'] for o in rec.get('outputs', [])]
    tests = {t['name']: t['passed'] for t in rec.get('tests', [])}
    files = sorted(glob.glob(os.path.join(CUT_ROOT, '*', CUT_SB, '*', '*.fits')))
    check('cutouts (FAKE engine + the REAL writer on synthetic cubes): unit done, %d cutouts placed, the manifest written, '
          'identity V / I and CM1 / CM2 pass, the staging directory removed' % len(files),
          rc == 0 and rec.get('status') == 'done' and len(man) == 1 and files and all(tests.values())
          and set(tests) >= {'photometry_cutout_manifest.I1_identity_V', 'photometry_cutout_manifest.I1_identity_I',
                             'photometry_cutout_manifest.CM1_files', 'photometry_cutout_manifest.CM2_one_per_key'}
          and not os.path.exists(os.path.join(CUT_ROOT, CFG.photometry_cutouts.STAGING_SUBDIR)), (rc, rec.get('reason'), tests,
                                                                                                  txt[-1500:]))
    rows = TC.read_manifest(man[0])[1] if man else []
    check('manifest: one row per placed cutout, sha256 of the placed bytes', len(rows) == len(files) and all(
        TC.sha256_file(os.path.join(CUT_ROOT, r['file'])) == r['sha256'] for r in rows), (len(rows), len(files)))
    smp = sorted({r['sample'] for r in rows})
    check('the default target set: HWJ and 10 pc (the pinned lists) in the synthetic image (%s)' % smp,
          set(smp) <= {'HWJ', '10pc', 'SolarSystem'} and 'HWJ' in smp, smp)
    h0 = {r['file']: r['sha256'] for r in rows}
    with patched(record, git_commit=lambda path: 'abc1234'):
        step(argv)
        rc, txt = step(argv + ['--resume'])
    check('--resume (clean tree): the unit is skipped (inputs, cutout root, outputs unchanged)',
          rc == 0 and ': skipped (' in txt, txt[-800:])
    rc, txt = step(argv)
    rows2 = TC.read_manifest(man[0])[1]
    check('a re-run replaces the unit\'s files in place (same names: no stale time stamp), done',
          rc == 0 and sorted(r['file'] for r in rows2) == sorted(h0), txt[-800:])
    # a stale time stamp of a written key (the 12h 2024-12-31 defect) and an orphan: refused, then quarantined with --fresh
    f0 = os.path.join(CUT_ROOT, rows2[0]['file'])
    stale = f0.replace(f0.split('_')[-2], f0.split('_')[-2][:-1] + ('0' if f0.split('_')[-2][-1] != '0' else '1'))
    shutil.copy(f0, stale)
    orphan_d = os.path.join(CUT_ROOT, rows2[0]['sample'], CUT_SB, 'Not_A_Written_Object')
    os.makedirs(orphan_d)
    orphan = os.path.join(orphan_d, 'Not_A_Written_Object_' + os.path.basename(f0).split('_', len(rows2[0]['object'].split('_')))[-1])
    shutil.copy(f0, orphan)
    rc, txt = step(argv)
    check('C-STALE: another time stamp of a written key and a file of the cell the unit did not write refuse the unit (exit 2), '
          'nothing placed over them', rc == 2 and 'C-STALE' in txt and os.path.exists(stale) and os.path.exists(orphan), txt[-1200:])
    hs = TC.sha256_file(stale)
    rc, txt = step(argv + ['--fresh'])
    q = glob.glob(os.path.join(CUT_ROOT, CFG.photometry_cutouts.QUARANTINE_PREFIX + '*', 'MANIFEST_sha256.csv'))
    qrows = T.read_csv(q[0])[1] if q else []
    check('C-STALE --fresh: both moved to <root>/quarantine_superseded_cutouts_<run id>/ with their sha256 (never deleted), the '
          'unit done and CM2 passes', rc == 0 and not os.path.exists(stale) and not os.path.exists(orphan) and len(qrows) == 2
          and hs in {r['sha256'] for r in qrows}, (txt[-800:], qrows))
    # an identity failure refuses BEFORE anything reaches the root; staging kept
    before = {p: TC.sha256_file(p) for p in glob.glob(os.path.join(CUT_ROOT, '*', CUT_SB, '*', '*.fits'))}
    setup_fake_regen(corrupt=True)
    rc, txt = step(argv)
    after = {p: TC.sha256_file(p) for p in glob.glob(os.path.join(CUT_ROOT, '*', CUT_SB, '*', '*.fits'))}
    rec = records(CUT_OUT).get('cutouts.%s.%s' % (CUT_CELL, CUT_SB), {})
    check('C-IDENTITY: a cutout whose plane is shifted refuses the unit (exit 2), the root unchanged, the staging directory '
          'kept and named', rc == 2 and 'C-IDENTITY' in txt and before == after and rec.get('meta', {}).get('staging')
          and os.path.isdir(rec['meta']['staging']), txt[-1200:])
    setup_fake_regen(excluded=[CUT_CELL])
    rc, txt = step(argv)
    why = records(CUT_OUT).get('cutouts.%s.%s' % (CUT_CELL, CUT_SB), {}).get('reason', '')
    check('C-EXCLUDED: the engine\'s data_decisions refusal (exit 3) is no input (exit 3), never a failure',
          rc == 3 and 'C-EXCLUDED' in why, (why, txt[-600:]))
    setup_fake_regen()
    rc, txt = step(['photometry'] + CUT_COMMON + ['--cutout-root', CUT_ROOT, '--out', CUT_OUT])
    check('C-CELLS: a regeneration without --cells refuses (exit 2)', rc == 2 and 'C-CELLS' in txt, txt)
    for root, what in ((CFG.photometry.RETIRED_CUTOUT_ROOTS[0], 'the retired generation 4'),
                       (os.path.join(CFG.photometry.CUTOUT_ROOT), 'the served generation (a production root)')):
        rc, txt = step(['photometry'] + CUT_COMMON + ['--cells', '%s_%s' % (CUT_CELL, CUT_SB), '--cutout-root', root,
                                                      '--out', CUT_OUT, '--dry-run'])
        check('C-ROOT: --cutout-root %s refuses (dry run, exit 2, nothing written)' % what,
              rc == 2 and 'C-ROOT' in txt and not os.path.exists(os.path.join(root, CFG.photometry_cutouts.STAGING_SUBDIR)), txt)


FAKE_TIER = r'''#!/usr/bin/env python3
"""FAKE photometry_tier_cutouts_20261025.py for tests/test_photometry.py: the REAL tool's code (imported from its file) with its
input roots DEEP / TROOT / SB pointed at the test's synthetic tree and p3s2_frames.master_path at the synthetic master
(fake_params.json), then the tool's own __main__ logic (--root --tier --cells --workers) - its cell_job unchanged."""
import argparse, importlib.util, json, os, sys
sys.path.insert(0, '/lustre/gh/main/clv15'); sys.path.insert(0, '/lustre/gh/main/exopipe_ops')
P = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fake_params.json')))
spec = importlib.util.spec_from_file_location('ptc', P['tier_tool'])
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
M.DEEP, M.TROOT, M.SB = P['deep_run'], P['tenmin_root'], P['tenmin_search']
import p3s2_frames
p3s2_frames.master_path = lambda lst, band, S: P['master'] if S == 'I' else None
ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True); ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--tier', default='both'); ap.add_argument('--cells', nargs='*')
a = ap.parse_args()
tot = 0
for t in (('deep', '10min') if a.tier == 'both' else (a.tier,)):
    for c in a.cells:
        for S in ('I', 'V'):
            (_, lst, band, _, _), nw, msg = M.cell_job((t, c.split('_', 1)[0], c.split('_', 1)[1], S, a.root))
            tot += nw
            print('%s %s_%s %s: %d cutouts (%s)' % (t, lst, band, S, nw, msg), flush=True)
print('PHOTOMETRY-TIER-DONE %d cutouts written under %s' % (tot, a.root))
'''

TIER = os.path.join(TMP, 'tier')
TIER_CELL = '18h_64MHz'
TIER_DATES = ('2025-04-11', '2025-04-12', '2025-04-13')


def make_tier_inputs():
    """Synthetic tier residuals of 18h_64MHz around (275, 60): deep = 3 nights (I, V) in <TIER>/Run_FAKE/{iresid,vresid}/;
    10-minute = 2 windows per night (frames.csv sub_path), the first searched (v_deep_images.csv); a Stokes I master."""
    from astropy.io import fits
    rng = np.random.default_rng(31)
    deep, troot, sbr = os.path.join(TIER, 'Run_FAKE'), os.path.join(TIER, 'tenmin'), os.path.join(TIER, 'search_B')
    rows = []
    for k, date in enumerate(TIER_DATES):
        for S, sub in (('I', 'iresid'), ('V', 'vresid')):
            h = cube_header(k, '%s_123000' % date.replace('-', ''))
            h['MJD-MID'] = 60776.52 + k
            d = os.path.join(deep, sub, TIER_CELL)
            os.makedirs(d, exist_ok=True)
            fits.writeto(os.path.join(d, '%s_%s_redo_resid_64MHz.fits' % (date, S)),
                         rng.normal(0, 0.1, (NPX_C, NPX_C)).astype(np.float32), h, overwrite=True)
            for w_ in range(2):
                key = '%s_%d' % (date.replace('-', ''), w_)
                p = os.path.join(troot, TIER_CELL, 'win', '%s_%s.fits' % (key, S))
                os.makedirs(os.path.dirname(p), exist_ok=True)
                h['MJD-MID'] = 60776.52 + k + w_ * 0.007
                fits.writeto(p, rng.normal(0, 0.2, (NPX_C, NPX_C)).astype(np.float32), h, overwrite=True)
                rows.append(dict(key=key, date=date, stokes=S, sub_path=p))
    os.makedirs(os.path.join(deep, 'refusals'), exist_ok=True)
    with open(os.path.join(troot, TIER_CELL, 'frames.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['key', 'date', 'stokes', 'sub_path'])
        w.writeheader()
        w.writerows(rows)
    for S in 'IV':
        d = os.path.join(sbr, '%s_%s' % (TIER_CELL, S), 'transients', 'Run_SLST18B')
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, 'v_deep_images.csv'), 'w') as f:
            f.write('date\n' + ''.join('%s_0\n' % d_.replace('-', '') for d_ in TIER_DATES))
    master = os.path.join(TIER, 'I_deep_Taper_R075_64MHz.fits')
    fits.writeto(master, rng.normal(0, 1, (NPX_C, NPX_C)).astype(np.float32), cube_header(0, '20250413_123000'), overwrite=True)
    p = os.path.join(FAKE_DIR, 'fake_params.json')
    prm = json.load(open(p)) if os.path.exists(p) else {}
    prm.update(tier_tool=TIER_TOOL, deep_run=deep, tenmin_root=troot, tenmin_search=sbr, master=master)
    json.dump(prm, open(p, 'w'))
    fp = os.path.join(FAKE_DIR, CFG.photometry.ENGINE['cutouts_tier'])
    open(fp, 'w').write(FAKE_TIER)
    FAKES['cutouts_tier'] = fp
    return deep, troot, sbr


def test_cutouts_tiers():
    """The deep and 10-minute tier cutouts through the step (the REAL tool's cell_job on synthetic residuals) and the REAL
    tier forced tool on them (cutmeasure, in place): an ad-hoc HWJ subset, never into the served roots."""
    deep, troot, sbr = make_tier_inputs()
    adhoc = os.path.join(TMP, 'tier_hwj.csv')
    src = [r for r in csv.DictReader(open(os.path.join(CFG.paths.CLV15_ROOT, CFG.photometry.SAMPLE_FILES['HWJ']),
                                          encoding='utf-8-sig')) if r['common_name'] in ('HD 163607', 'HIP 91258')]
    with open(adhoc, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(src[0]))
        w.writeheader()
        w.writerows(src)
    out, root, tout = os.path.join(TMP, 'tier_out'), os.path.join(TMP, 'tier_out', 'photometry_genT'), \
        os.path.join(TMP, 'tier_out', 'forced_tiers_genT')
    cc = CFG.photometry_cutouts
    with patched(cc, TIER_DEEP_RUN=deep, TIER_TENMIN_ROOT=troot, TIER_TENMIN_SEARCH=sbr):
        rcs, recs = {}, {}
        for tier, token in (('deep', 'deep'), ('sublst', '10min')):
            argv = ['photometry', '--tier', tier, '--stages', 'cutouts,cutmeasure', '--cells', TIER_CELL, '--targets',
                    'HWJ=%s' % adhoc, '--cutout-root', root, '--tier-out', tout, '--out', out, '--cores', 'none']
            rcs[tier], txt = step(argv)
            recs[tier] = records(out)
            u = recs[tier].get('cutouts.%s.%s.HWJ' % (token, TIER_CELL), {})
            m = recs[tier].get('cutmeasure.%s.18h.HWJ' % token, {})
            files = sorted(glob.glob(os.path.join(root, 'HWJ', '64MHz', '*', '*_18h_%s_[IV].fits' % token)))
            npz = sorted(glob.glob(os.path.join(tout, token, '18h', 'forced_HWJ_64MHz_[IV].npz')))
            ut = {t['name'].split('.', 1)[1]: t['passed'] for t in u.get('tests', [])}
            mt = {t['name'].split('.', 1)[1]: t['passed'] for t in m.get('tests', [])}
            check('tier %s (%s): cutouts (the REAL tool\'s cell_job on synthetic residuals) and cutmeasure (the REAL '
                  'photometry_tier_forced_20261026.py) done; %d cutouts of the 2 HWJ targets, one plane each re-cut from its '
                  'source, CM1 / CM2; npz I and V pass TN1 / TN2, the manifest TM1 / TM2' % (tier, token, len(files)),
                  rcs[tier] == 0 and u.get('status') == 'done' and m.get('status') == 'done' and len(files) == 4
                  and len(npz) == 2 and all(ut.values()) and {'I1_recut_I', 'I1_recut_V', 'CM1_files', 'CM2_one_per_key'} <= set(ut)
                  and all(mt.values()) and {'TN1_shapes', 'TN2_target_remeasure', 'TM1_statuses', 'TM2_coverage'} <= set(mt),
                  (rcs[tier], u.get('reason'), m.get('reason'), ut, mt, files, txt[-1500:]))
        from astropy.io import fits
        f0 = sorted(glob.glob(os.path.join(root, 'HWJ', '64MHz', '*', '*_18h_10min_I.fits')))
        T10 = fits.getdata(f0[0], 'TIMES') if f0 else None
        check('10-minute tier file: one plane per window, the searched flag from v_deep_images.csv (TIMES searched)',
              T10 is not None and len(T10) == 6 and [str(x).strip() for x in T10['searched']] == ['1', '0'] * 3,
              None if T10 is None else list(T10['searched']))
        served_t = dict(CFG.photometry.TIER_FORCED_ROOTS, HWJ=os.path.join(TMP, 'served_tier_tmp'))
        with patched(CFG.photometry, TIER_FORCED_ROOTS=served_t):
            rc, txt = step(['photometry', '--tier', 'deep', '--stages', 'cutmeasure', '--cells', TIER_CELL, '--targets',
                            'HWJ=%s' % adhoc, '--cutout-root', root, '--out', out, '--cores', 'none', '--dry-run'])
        check('PHOT-ADHOC: cutmeasure of an ad-hoc set into the served tier forced root (no --tier-out) refuses (dry run, '
              'exit 2)', rc == 2 and 'PHOT-ADHOC' in txt and 'served generation' in txt, txt[-600:])
        rc, txt = step(['photometry', '--tier', 'deep', '--stages', 'cutmeasure', '--cells', TIER_CELL, '--cutout-root', root,
                        '--out', out, '--cores', 'none', '--dry-run'])
        check('C-ROOT: cutmeasure into the production tier forced roots refuses in the refactor window (dry run, exit 2)',
              rc == 2 and 'C-ROOT' in txt, txt[-600:])
        rc, txt = step(['photometry', '--tier', 'deep', '--stages', 'cutmeasure', '--cells', TIER_CELL, '--targets',
                        'ADHOC=%s' % adhoc, '--cutout-root', root, '--tier-out', tout, '--out', out, '--cores', 'none',
                        '--dry-run'])
        check('PHOT-ADHOC: cutmeasure of a tag the tier tool does not measure (ADHOC) refuses', rc == 2 and 'PHOT-ADHOC' in txt
              and 'SAMPLES' in txt, txt[-600:])
        rc, txt = step(['photometry', '--tier', 'deep', '--stages', 'cutouts', '--cells', TIER_CELL, '--src-run',
                        os.path.join(TMP, 'deep', 'transients', 'Run_T'), '--cutout-root', root, '--out', out, '--cores', 'none',
                        '--dry-run'])
        check('C-RUN: --src-run naming another run than the tier tool reads refuses', rc == 2 and 'C-RUN' in txt, txt[-600:])


def test_adhoc_and_selection(objs):
    """Ad-hoc target lists (the snapshot measure through the REAL snapshot_forced_phot.py with P3_SNAP_CUTOUT_SAMPLES, as
    stage1_when_done_20261028.sh) and the gates; stage selection (cutouts only when named; sub-LST)."""
    from exopipe.phase3.lib.photometry import steps_common as K_
    o = objs['HWJ']
    ent = [r for r in csv.DictReader(open(os.path.join(CFG.paths.CLV15_ROOT, CFG.photometry.SAMPLE_FILES['HWJ']),
                                          encoding='utf-8-sig')) if T.safe_name(r['common_name']) == o][0]
    adhoc = os.path.join(TMP, 'adhoc_list.csv')
    with open(adhoc, 'w') as f:
        f.write('name,ra,dec\n%s,%s,%s\n' % (ent['common_name'], ent['ra_current'], ent['dec_current']))
    ah_root = os.path.join(TMP, 'cuts_adhoc')
    for p in glob.glob(os.path.join(CUTS, 'HWJ', '*', o, '*.fits')):
        rel = os.path.relpath(p, os.path.join(CUTS, 'HWJ'))
        os.makedirs(os.path.dirname(os.path.join(ah_root, 'ADHOC', rel)), exist_ok=True)
        shutil.copy(p, os.path.join(ah_root, 'ADHOC', rel))
    out = os.path.join(TMP, 'adhoc_out')
    rc, txt = step(['photometry', '--tier', 'snapshot', '--stages', 'measure', '--cells', SYN_SNAP_CELL, '--targets',
                    'ADHOC=%s' % adhoc, '--cutout-root', ah_root, '--out', out, '--work', WORK, '--cores', 'none',
                    '--band-procs', '3'])
    rec = records(out).get('measure.%s' % SYN_SNAP_CELL, {})
    npz = sorted(os.path.basename(p) for p in glob.glob(os.path.join(out, S1, SYN_SNAP_CELL, 'forced_*.npz')))
    failed = [t['name'] for t in rec.get('tests', []) if not t['passed']]
    check('ad hoc measure (--targets ADHOC=csv): the REAL snapshot_forced_phot.py measures the ADHOC cutouts (6 npz: 3 bands x '
          'I, V), every content test passes with the ad-hoc list (N5 names)', rc == 0 and rec.get('status') == 'done'
          and npz == sorted('forced_ADHOC_%s_%s.npz' % (b, s_) for b in SNAP_BANDS for s_ in 'IV') and not failed
          and any(t['name'].endswith('N5_names') for t in rec.get('tests', [])), (rc, rec.get('reason'), npz, failed, txt[-1200:]))
    base = ['photometry', '--tier', 'snapshot', '--cells', SYN_SNAP_CELL, '--out', os.path.join(TMP, 'sel'), '--cores', 'none']
    rc, txt = step(base + ['--stages', 'evaluate', '--targets', 'ADHOC=%s' % adhoc])
    check('PHOT-ADHOC: --targets with the evaluation refuses (exit 2)', rc == 2 and 'PHOT-ADHOC' in txt, txt[-500:])
    rc, txt = step(base + ['--stages', 'measure', '--planets', '--dry-run'])
    check('PHOT-ADHOC: --planets without --targets refuses', rc == 2 and 'PHOT-ADHOC' in txt, txt[-500:])
    rc, txt = step(['photometry'] + DEEP_COMMON + ['--stages', 'frames', '--targets', 'ADHOC=%s' % adhoc, '--out',
                                                   os.path.join(TMP, 'sel_d'), '--dry-run'])
    check('PHOT-ADHOC: --targets with the deep frame-table chain refuses', rc == 2 and 'PHOT-ADHOC' in txt, txt[-500:])
    bad = os.path.join(TMP, 'adhoc_bad.csv')
    open(bad, 'w').write('name,ra,dec\nX,not,a number\n')
    for spec, what in (('SolarSystem=%s' % adhoc, 'the planet sample as a tag'), ('A=%s' % bad, 'no parsable row'),
                       ('A=/nonexistent.csv', 'a missing file'), ('A=%s,A=%s' % (adhoc, adhoc), 'a tag twice'),
                       ('bad tag=%s' % adhoc, 'a tag that is no directory name')):
        rc, txt = step(base + ['--stages', 'measure', '--targets', spec, '--cutout-root', ah_root, '--dry-run'])
        check('PHOT-ADHOC: --targets %s refuses' % what, rc == 2 and 'PHOT-ADHOC' in txt, txt[-500:])
    served_tmp = os.path.join(TMP, 'served_gen_tmp')
    with patched(CFG.photometry, CUTOUT_ROOT=served_tmp):
        rc, txt = step(['photometry'] + CUT_COMMON + ['--cells', '%s_%s' % (CUT_CELL, CUT_SB), '--targets', 'ADHOC=%s' % adhoc,
                                                      '--cutout-root', served_tmp, '--out', CUT_OUT, '--dry-run'])
    check('PHOT-ADHOC: an ad-hoc cutout run into the served generation\'s root refuses (dry run, exit 2)',
          rc == 2 and 'PHOT-ADHOC' in txt and 'served generation' in txt, txt[-500:])
    rc, txt = step(base + ['--stages', 'measure', '--targets', 'ADHOC=%s' % adhoc, '--cutout-root', ah_root, '--out',
                           CFG.photometry.OUT_ROOT['snapshot'], '--dry-run'])
    check('PHOT-ADHOC: an ad-hoc measure into the served stage 1 refuses (dry run, exit 2)', rc == 2 and 'PHOT-ADHOC' in txt,
          txt[-500:])
    rc, txt = step(base + ['--dry-run'])
    check('stage selection: without --stages the snapshot tier plans measure and evaluate, never the cutout regeneration',
          rc == 0 and '[measure]' in txt and '[cutouts]' not in txt and 'stages measure,evaluate' in txt, txt[-800:])
    rc, txt = step(base + ['--from', 'cutouts', '--dry-run'])
    check('stage selection: --from cutouts without --stages is an error (a regeneration is named)', rc == 1
          and 'runs only when named' in txt, txt[-500:])
    rc, txt = step(['photometry', '--tier', 'sublst', '--epochs', '18h_2025-04-13', '--out', os.path.join(TMP, 'sel_s'),
                    '--cores', 'none', '--dry-run'])
    check('stage selection: the sub-LST tier has no per-hour stage (--epochs is an error)', rc == 1
          and 'no per-hour stage' in txt, txt[-500:])
    check('stage selection: the configured defaults (deep frames..evaluate, snapshot measure,evaluate, sublst cutouts,'
          'cutmeasure)', K_.resolve_stages(A(tier='deep'), 'deep') == ['frames', 'stage', 'measure', 'evaluate']
          and K_.resolve_stages(A(tier='snapshot'), 'snapshot') == ['measure', 'evaluate']
          and K_.resolve_stages(A(tier='sublst'), 'sublst') == ['cutouts', 'cutmeasure']
          and K_.resolve_stages(A(stages='evaluate,cutouts'), 'snapshot') == ['cutouts', 'evaluate'])


class A:
    def __init__(self, **kw):
        self.stages = self.stage_from = self.stage_until = self.epochs = None
        self.__dict__.update(kw)


def main():
    excluded = excluded_frame()
    check('synthetic snapshot cell %s has no search output, no exclusion, no tier-3 pin row' % SYN_SNAP_CELL,
          not os.path.exists(os.path.join(CFG.paths.SNAPSHOT_FLEET_OUT, SYN_SNAP_CELL))
          and not any(r['date'] == SYN_SNAP_CELL[4:] and r['lst'] == SYN_SNAP_CELL[:3]
                      for r in csv.DictReader(open(CFG.paths.DATA_DECISIONS)))
          and SYN_SNAP_CELL[4:] not in open(CFG.photometry_evaluate.DEEP_SEARCH_PIN['snapshot']).read())
    test_mirrors()
    test_ports()
    make_deep(excluded)
    write_params(excluded)
    KD.roots_cells = lambda: dict(ROOTS)
    objs = pick_templates()
    check('snapshot templates: one current, high, A-team-free target per sample %s' % objs, sorted(objs) == ['10pc', 'HWJ'])
    make_snapshot(objs)
    test_gates()
    test_deep_e2e(excluded)
    test_snapshot_e2e()
    test_negative()
    test_served_generation()
    test_cutout_mirrors()
    test_cutout_ports()
    test_identity()
    test_one_file_per_key()
    test_cutouts_snapshot()
    test_cutouts_tiers()
    test_adhoc_and_selection(objs)
    test_real()
    test_dry_real()


if __name__ == '__main__':
    try:
        main()
    finally:
        K.engine = REAL_ENGINE
        KD.roots_cells = REAL_ROOTS
        shutil.rmtree(TMP, ignore_errors=True)
        try:
            os.rmdir(BASE)
        except OSError:
            pass
    n = len(RESULTS)
    print('\n%d of %d checks passed in %.0f s' % (sum(ok for _, ok in RESULTS), n, time.time() - T0))
