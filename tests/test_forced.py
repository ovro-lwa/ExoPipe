#!/usr/bin/env python3
"""Tests of `exopipe3 forced` (unit forced; plain asserts, runnable as `python tests/test_forced.py`, under 2 minutes).

Synthetic data only in a fresh /tmp/exopipe3_tests_forced/ (removed at the end); READ-ONLY checks of the production
scripts the step runs in place (config truth: every in-place literal the config describes equals the literal in the file).
No processing on real data, no ssh, no GPU. The catflux stage runs the real p3s3_forced_phot.py (core.legacy.run) on a
synthetic run directory when this host holds paths.COADD_SB_ROOT (lwacalim07); elsewhere that part is reported skipped.

Run with: NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py \
          /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_forced.py
"""
import ast
import contextlib
import csv
import io
import json
import math
import os
import re
import shutil
import struct
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.core import env as _env       # noqa: E402

_env.legacy()
from exopipe.phase3.config import CFG, ConfigRefusal   # noqa: E402
from exopipe.phase3.core import validate, record       # noqa: E402

TMP = '/tmp/exopipe3_tests_forced'
OPS = '/lustre/gh/main/exopipe_ops'
CLV = '/lustre/gh/main/clv15'
RESULTS = []


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond), detail))
    print('%s  %s%s' % ('PASS' if cond else 'FAIL', name, ('  -- ' + detail) if detail and not cond else ''), flush=True)


_LAST_OUT = ['']


def _out_last():
    return _LAST_OUT[0]


def cli(argv):
    """exopipe3 in-process; returns (exit code, stderr text); stdout kept in _out_last(). Overrides applied by a call are
    undone after it."""
    from exopipe.phase3 import cli as C
    err, out = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
        try:
            rc = C.main(argv)
        finally:
            CFG.reset()
    _LAST_OUT[0] = out.getvalue()
    return rc, err.getvalue()


# ---------------------------------------------------------------------------------------------------------------------
# 1. config: sections, protected and ruled keys, the in-place override refusal
# ---------------------------------------------------------------------------------------------------------------------

def test_config():
    secs = CFG.sections()
    for s in ('forced', 'forced_panels', 'forced_catflux', 'forced_snapshot', 'forced_assoc'):
        check('config section %s loads' % s, s in secs)
    for s in ('forced_frames', 'forced_stage', 'forced_measure', 'forced_eval'):
        check('family-A measurement section %s not drafted here (photometry owns it, decision 6)' % s, s not in secs)
    for k in ('forced.SAMPLE_FILES', 'forced_catflux.ROOTS_CSV', 'forced_snapshot.FROZEN_MASTERS',
              'forced_snapshot.MASTER_CATS', 'forced_assoc.RADIO_CATDIR'):
        check('%s protected' % k, CFG.protected(k))
        try:
            CFG.set(k, "'/x'")
            ok = False
        except ConfigRefusal:
            ok = True
        CFG.reset()
        check('%s override refused' % k, ok)
    check('ruled key forced_snapshot.REFUSE_FULL names its ledger ID',
          CFG.ruled('forced_snapshot.REFUSE_FULL') == ('INCIDENT-SNAP-FULL-CUTOUTS-EQUAL-RED-62-CELLS',))
    from exopipe.phase3.lib.forced import common as K
    for k, v in (('forced_catflux.STAMP_RADIUS_BEAMS', '2.0'), ('fitting.GAUSS_SNR', '6.0'),
                 ('forced_snapshot.MERGE_ARCMIN', '2.0')):
        CFG.authorize([], out=TMP)
        CFG.set(k, v)
        try:
            K.check_overrides()
            ok = False
        except record.Refusal:
            ok = True
        CFG.reset()
        check('G37: override of the in-place literal %s refused' % k, ok)
    CFG.authorize([], out=TMP)
    CFG.set('forced_snapshot.NPROC[fit]', '8')
    try:
        K.check_overrides()
        ok = True
    except record.Refusal:
        ok = False
    CFG.reset()
    check('G37: override of a steering key (forced_snapshot.NPROC[fit]) allowed', ok)
    import importlib
    mod = importlib.import_module('exopipe.phase3.steps.forced')
    from exopipe.phase3 import config
    check('steps.forced CONFIG declares only known sections', config.declared_sections(mod) is not None)


# ---------------------------------------------------------------------------------------------------------------------
# 2. config truth: the in-place literals (read-only parse of the production files)
# ---------------------------------------------------------------------------------------------------------------------

def _assigns(path):
    out = {}
    for node in ast.parse(open(path).read(), path).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return out


def _src(path):
    return open(path).read()


def test_inplace_truth():
    s, a, pc, cf, f = CFG.forced_snapshot, CFG.forced_assoc, CFG.forced_panels, CFG.forced_catflux, CFG.forced
    P = _assigns(os.path.join(OPS, 'snapshot_forced_panels.py'))
    PS = _src(os.path.join(OPS, 'snapshot_forced_panels.py'))
    for key, name in (('MERGE_ARCMIN', 'MERGE_ARCMIN'), ('ASTROM_SNR', 'ASTROM_SNR'), ('VI_NOTE_SIGMA', 'VI_NOTE_SIGMA'),
                      ('TIE_LARGE_X_SYS', 'TIE_LARGE_X_SYS'), ('FROZEN_MASTERS', 'FROZEN_MASTERS'),
                      ('MASTER_CATS', 'MASTER_CATS'), ('PHOT_ROOT', 'PHOT'), ('STAGE1_ROOT', 'STAGE1')):
        check('truth forced_snapshot.%s = snapshot_forced_panels.%s' % (key, name), getattr(s, key) == P.get(name),
              '%r vs %r' % (getattr(s, key), P.get(name)))
    check('truth forced_snapshot.SKIP_FIELDS', tuple(s.SKIP_FIELDS) == tuple(P.get('SKIP_FIELDS') or ()))
    check('truth paths.SNAPSHOT_FLEET_OUT = snapshot_forced_panels.PROTO', CFG.paths.SNAPSHOT_FLEET_OUT == P.get('PROTO'))
    check('truth paths.COADD_SB_ROOT = snapshot_forced_panels.COADD_ROOT', CFG.paths.COADD_SB_ROOT == P.get('COADD_ROOT'))
    for txt, want in (('std7_panel.EL_MIN = %s' % s.PANEL_EL_MIN_DEG, True),
                      ('stp = %d if tier == "10s" else %d' % (s.LIGHTCURVE_TICK_STEP['10s'], s.LIGHTCURVE_TICK_STEP['1min']), True),
                      ('ThreadPoolExecutor(%d)' % s.FIG_THREADS, True), ('timeout=%d' % s.FIG_TIMEOUT_S, True),
                      ('shape[0] < %d' % s.VERIFY_PANEL_MIN_HEIGHT_PX, True),
                      ('random.Random(%d).sample(rows, min(%d' % (s.VERIFY_SNR_SAMPLE[1], s.VERIFY_SNR_SAMPLE[0]), True),
                      ('float(r["snr"])) > %g' % s.VERIFY_SNR_TOL, True),
                      ('ft.add_argument("--nproc", type=int, default=%d)' % s.NPROC['fit'], True),
                      ('te.add_argument("--nproc", type=int, default=%d)' % s.NPROC['tie'], True),
                      ('r.add_argument("--nproc", type=int, default=%d)' % s.NPROC['render'], True),
                      ('s_(d["ts"])) <= %d' % s.TIE_FRAME_WINDOW_S, True),
                      ('> %g or abs(dy' % s.TIE_RECON_TOL_ARCMIN, True),
                      ('os.environ.setdefault("P3_SNAP_FLOW", "%s")' % s.FLOW_ENV['P3_SNAP_FLOW'], True),
                      ('"/lustre/gh/main/exopipe_ops/snapshot_forced_state/done"', s.HOURS_SEARCHED_DIR ==
                       '/lustre/gh/main/exopipe_ops/snapshot_forced_state/done')):
        check('truth snapshot_forced_panels contains %r' % txt, (txt in PS) == want)
    for line, key in ((s.OK_LINES['build'], 'build'), (s.OK_LINES['fit'], 'fit'), (s.OK_LINES['tie'].strip(), 'tie'),
                      (s.OK_LINES['loc'].strip(), 'loc'), (s.OK_LINES['render'], 'render'), (s.OK_LINES['verify'], 'verify')):
        check('truth success line %s printed by snapshot_forced_panels' % key, line in PS)
    A = _assigns(os.path.join(OPS, 'snapshot_forced_assoc.py'))
    AS = _src(os.path.join(OPS, 'snapshot_forced_assoc.py'))
    for key, name in (('CEIL_PCT', 'CEIL_PCT'), ('LR_DECISIVE', 'LR_DECISIVE'), ('MIN_S74_JY', 'MIN_S74_JY'),
                      ('ALPHA', 'ALPHA'), ('GROUP_LINK_ARCSEC', 'GROUP_LINK_ARCSEC'), ('DENS_CONE_DEG', 'DENS_CONE_DEG'),
                      ('DENS_MIN_COUNT', 'DENS_MIN_COUNT'), ('NEAREST_OUT_DEG', 'NEAREST_OUT_DEG'),
                      ('SITE_LAT_DEG', 'OVRO_LAT_DEG')):
        check('truth forced_assoc.%s = snapshot_forced_assoc.%s' % (key, name), getattr(a, key) == A.get(name),
              '%r vs %r' % (getattr(a, key), A.get(name)))
    check('truth forced_assoc.RADIO_CATDIR = the P3_RADIO_CATDIR default', "'P3_RADIO_CATDIR', '%s'" % a.RADIO_CATDIR in AS)
    check('truth forced_assoc.NO_LABEL = {ADLEO}', tuple(a.NO_LABEL) == (A.get('ADLEO'),) and 'NO_LABEL = {ADLEO:' in AS)
    check('truth forced_assoc.FILES', tuple(a.FILES) == (A.get('F_HYP'), A.get('F_IDENT'), A.get('F_EV')))
    m = re.search(r"^LABELS = \((.*)\)$", AS, re.M)
    labs = tuple(x.strip().strip("'") for x in m.group(1).split(',')) if m else ()
    check('truth forced_assoc.LABELS', tuple(a.LABELS[:-1]) == labs[:-1] and labs[-1] == 'NOT_LABELLED'
          and A.get('NOT_LABELLED') == a.LABELS[-1])
    D = _assigns(os.path.join(OPS, 'deep_forced_panels.py'))
    DS = _src(os.path.join(OPS, 'deep_forced_panels.py'))
    for key in ('MERGE_ARCMIN', 'GAUSS_SNR', 'WRAPPER_VERSION'):
        check('truth forced_panels.%s = deep_forced_panels.%s' % (key, key), getattr(pc, key) == D.get(key))
    check('truth forced_panels.BUILDER', pc.BUILDER == '%s/%s' % (OPS, re.search(r"BUILDER = f'\{OPS\}/(\S+)'", DS).group(1)))
    check('truth forced_panels.INFO_LINE_CHARS', "[:%d]" % pc.INFO_LINE_CHARS in DS)
    check('truth forced_panels.SRC_LINKS', 'for name in %r' % (pc.SRC_LINKS,) in DS.replace('\n', '').replace(
        '             ', ' ').replace("',  '", "', '"), 'tuple not found verbatim')
    check('truth forced_panels.SRC_COPIES', 'for name in %r' % (pc.SRC_COPIES,) in DS)
    check('truth forced_panels.VERIFY_OK / TESS_OK / QUEUE', pc.VERIFY_OK in DS and pc.TESS_OK in DS
          and "queue_forced.txt" in DS)
    FS = _src(os.path.join(OPS, 'p3s3_forced_phot.py'))
    for txt in ("glob.glob(f'{U}/%s')" % cf.CATALOG_GLOB, "glob.glob(f'{d}/%s')" % cf.SURVIVORS_GLOB,
                'max(%d, round(%s * bmaj))' % (cf.STAMP_RADIUS_MIN_PX, cf.STAMP_RADIUS_BEAMS),
                'TF.beam_icov_pix(', 'TF.beam_model(icov, 1.0', 'len(_cache) > %d' % cf.FRAME_CACHE,
                'round(A, %d)' % cf.DECIMALS, "forced_method='%s'" % cf.NO_FRAME):
        check('truth p3s3_forced_phot contains %r' % txt, txt in FS)
    check('truth forced_catflux.DONE_RE matches the script\'s summary line',
          re.search(cf.DONE_RE, 'forced photometry: 12669 measured, 0 without a frame') is not None
          and "print(f'forced photometry: {n_done} measured, {n_miss} without a frame')" in FS)
    PF = _src(os.path.join(OPS, 'p3s2_frames.py'))
    check('truth forced_catflux.ROOTS_CSV = the P3S2_ROOTS_CSV default', "'P3S2_ROOTS_CSV', '%s'" % cf.ROOTS_CSV in PF)
    for fn in f.SAMPLE_FILES.values():
        check('truth sample file %s exists under paths.CLV15_ROOT and is read in place' % fn,
              os.path.exists(os.path.join(CFG.paths.CLV15_ROOT, fn)) and fn in DS)
    TF = _src(os.path.join(CLV, 'phase3_template_fit.py'))
    inplace = 'wcs' if re.search(r'^\s*Jm = np\.array\(cd, dtype=float\)', TF, re.M) else (
        'legacy-mirrored' if re.search(r'Jm = np\.array\(\[\[-cd\[0, 0\]', TF) else 'unknown')
    check('truth forced.INPLACE_TEMPLATE_ORIENTATION = the in-place template fit (clv15 phase3_template_fit)',
          f.INPLACE_TEMPLATE_ORIENTATION == inplace, 'config %s, clv15 %s' % (f.INPLACE_TEMPLATE_ORIENTATION, inplace))
    for k in ('catflux', 'panels', 'snapshot', 'assoc', 'reserve_test'):
        check('in-place script %s exists' % f.INPLACE_SCRIPTS[k], os.path.exists(os.path.join(OPS, f.INPLACE_SCRIPTS[k])))
    for k in set(f.PYTHON_KEY.values()):
        check('site PYTHON[%r] defined (decision 2 vocabulary)' % k, k in CFG.site.PYTHON and
              k in ('default', 'flow', 'py36', 'casa', 'tess'))


# ---------------------------------------------------------------------------------------------------------------------
# 3. stage selection
# ---------------------------------------------------------------------------------------------------------------------

class A:
    def __init__(self, **kw):
        self.stages = self.stage_from = self.stage_until = self.epochs = None
        self.__dict__.update(kw)


def test_stages():
    from exopipe.phase3.lib.forced import common as K
    check('deep default = catflux', K.selected_stages(A(), 'deep') == ['catflux'])
    check('deep per hour = catflux', K.selected_stages(A(epochs='04h_2025-10-14'), 'deep') == ['catflux'])
    check('snapshot default = 7 stages in order', K.selected_stages(A(), 'snapshot') ==
          ['build', 'fit', 'tie', 'loc', 'assoc', 'render', 'verify'])
    check('snapshot --stages any order -> pipeline order',
          K.selected_stages(A(stages='verify,build'), 'snapshot') == ['build', 'verify'])
    check('snapshot --from/--until', K.selected_stages(A(stage_from='tie', stage_until='assoc'), 'snapshot') ==
          ['tie', 'loc', 'assoc'])
    for bad in (A(stages='measure'), A(stages='frames')):
        try:
            K.selected_stages(bad, 'deep')
            ok = False
        except ValueError:
            ok = True
        check('deep stage %s unknown here (photometry owns it)' % bad.stages, ok)


# ---------------------------------------------------------------------------------------------------------------------
# 4. catflux: the real p3s3_forced_phot.py on a synthetic run (and the 90-degree beam finding), its content test
# ---------------------------------------------------------------------------------------------------------------------

def _frame(path, bmaj_deg, bmin_deg, bpa_deg, amp, n=201, cd=0.05):
    """Synthetic SIN frame (CDELT1 < 0: +x west; +y north) with a point source of the header beam at the centre, drawn with
    the beam's TRUE orientation (major axis at BPA from north through east) on noise-free sky."""
    import numpy as np
    from astropy.io import fits
    y, x = np.mgrid[0:n, 0:n].astype(float)
    c = (n - 1) / 2.0
    e, nn = -(x - c), (y - c)                    # east, north in pixels
    pa = math.radians(bpa_deg)
    u = e * math.sin(pa) + nn * math.cos(pa)     # along the major axis
    v = e * math.cos(pa) - nn * math.sin(pa)
    smaj, smin = bmaj_deg / cd / 2.3548, bmin_deg / cd / 2.3548
    d = amp * np.exp(-0.5 * ((u / smaj) ** 2 + (v / smin) ** 2))
    h = fits.Header()
    h.update(CTYPE1='RA---SIN', CTYPE2='DEC--SIN', CRVAL1=150.0, CRVAL2=30.0, CRPIX1=c + 1, CRPIX2=c + 1, CDELT1=-cd,
             CDELT2=cd, CUNIT1='deg', CUNIT2='deg', BMAJ=bmaj_deg, BMIN=bmin_deg, BPA=bpa_deg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.writeto(path, d.astype(np.float32), h, overwrite=True)


def _survivors(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ['rank', 'source_id', 'ra', 'dec', 'band', 'lst_cells', 'best_cand_id', 'best_date', 'best_snr']
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_catflux():
    from exopipe.phase3.lib.forced import common as K
    U = os.path.join(TMP, 'transients', 'Run_TESTF')
    # round beam (forced = amplitude) and elongated beam BMAJ = 2 BMIN, BPA 0 (major axis north)
    _frame(os.path.join(U, 'vresid', '04h_46MHz', '2025-01-01_V_redo_resid_46MHz.fits'), 0.5, 0.5, 0.0, 10.0)
    _frame(os.path.join(U, 'vresid', '04h_55MHz', '2025-01-02_V_redo_resid_55MHz.fits'), 0.6, 0.3, 0.0, 10.0)
    os.makedirs(os.path.join(U, 'iresid'), exist_ok=True)
    rows = [dict(rank='rank0001', source_id='J1000+3000', ra='150.00000', dec='30.00000', band='46MHz', lst_cells='04h',
                 best_cand_id='VD_04h_2025-01-01_46MHz_0001', best_date='2025-01-01', best_snr='20'),
            dict(rank='rank0002', source_id='JNOFRAME', ra='150.00000', dec='30.00000', band='46MHz', lst_cells='04h',
                 best_cand_id='VD_04h_2025-01-09_46MHz_0002', best_date='2025-01-09', best_snr='9')]
    _survivors(os.path.join(U, 'source_catalogs_V_sb', '46MHz_sources_survivors.csv'), rows)
    _survivors(os.path.join(U, 'source_catalogs_V_sb', '55MHz_sources_survivors.csv'),
               [dict(rank='rank0001', source_id='J1000+3000b', ra='150.00000', dec='30.00000', band='55MHz',
                     lst_cells='04h', best_cand_id='VD_04h_2025-01-02_55MHz_0001', best_date='2025-01-02', best_snr='20')])
    # a catalogue without rows: p3s3_forced_phot.py:73-74 leaves it as it is (no forced columns) and the pass goes on
    empty = os.path.join(U, 'source_catalogs_V_sb', '64MHz_sources_survivors.csv')
    _survivors(empty, [])
    rc, err = cli(['forced', '--tier', 'deep', '--run', 'TESTF', '--stages', 'catflux', '--out', U, '--cores', 'none',
                   '--dry-run'])
    check('catflux --dry-run on a synthetic run exits 0', rc == 0, err[-300:])
    if K.host_of(CFG.paths.COADD_SB_ROOT) not in (None, K.hostname()):
        print('SKIP  catflux in-place run: this host does not hold %s' % CFG.paths.COADD_SB_ROOT)
        return
    t0 = time.time()
    rc, err = cli(['forced', '--tier', 'deep', '--run', 'TESTF', '--out', U, '--cores', 'none'])
    rec = record.load_record(U, 'forced', 'catflux') or {}
    check('catflux in place on a synthetic run: exit 0, record done (%.0f s)' % (time.time() - t0),
          rc == 0 and rec.get('status') == 'done', '%s %s %s' % (rc, rec.get('status'), (rec.get('reason') or err)[-400:]))
    check('catflux record: legacy code fingerprint holds p3s3_forced_phot.py and p3s2_frames.py',
          any(p.endswith('/p3s3_forced_phot.py') for p in (rec.get('meta') or {}).get('legacy_code', {}))
          and any(p.endswith('/p3s2_frames.py') for p in (rec.get('meta') or {}).get('legacy_code', {})))
    check('catflux record: measured 2, without a frame 1', (rec.get('meta') or {}).get('measured') == 2
          and (rec.get('meta') or {}).get('without_frame') == 1, str(rec.get('meta', {}).get('measured')))
    r46 = list(csv.DictReader(open(os.path.join(U, 'source_catalogs_V_sb', '46MHz_sources_survivors.csv'))))
    r55 = list(csv.DictReader(open(os.path.join(U, 'source_catalogs_V_sb', '55MHz_sources_survivors.csv'))))
    check('catflux round beam: forced_jy = the injected 10 Jy', abs(float(r46[0]['forced_jy']) - 10.0) < 1e-3,
          r46[0]['forced_jy'])
    check('catflux no frame row: method "no frame / off-frame", blank values',
          r46[1]['forced_method'] == 'no frame / off-frame' and r46[1]['forced_jy'] == '')
    # exopipe_ops 8639b21 (Gregg 2026-10-14 'Yes to the forced-flux fix'): the WCS-oriented beam model recovers the
    # injected flux of an elongated beam (the hand-built Gaussian before it gave 10 x 2ab/(a^2+b^2) = 8.000)
    check('catflux elongated beam (BMAJ = 2 BMIN, BPA 0): forced_jy = the injected 10 Jy (8639b21; was 8.000)',
          abs(float(r55[0]['forced_jy']) - 10.0) < 0.02, r55[0]['forced_jy'])
    t = [x for x in rec.get('tests', []) if x['name'] == 'catalog_forced_columns.catflux_columns']
    check('catflux content test ran on the three outputs and passed', len(t) == 3 and all(x['passed'] for x in t),
          str([x.get('detail') for x in t]))
    check('catflux leaves a catalogue without rows as it is (no forced columns), as p3s3_forced_phot.py:73-74',
          'forced_jy' not in open(empty).readline())
    # --resume: skipped only from a clean tree with unchanged inputs/code/config and outputs passing now
    rc2, err2 = cli(['forced', '--tier', 'deep', '--run', 'TESTF', '--out', U, '--cores', 'none', '--resume'])
    if str(rec.get('commit', '')).endswith('-dirty'):
        check('--resume never skips a record written from a dirty tree',
              rc2 == 0 and 'not skipped (record written from a dirty tree' in _out_last(), err2[-300:])
    else:
        check('--resume skips the done catflux unit (skipped-valid)', rc2 == 0 and 'skipped' in _out_last(), err2[-300:])
    # mutation: a forced_jy that is not the measurement
    p = os.path.join(U, 'source_catalogs_V_sb', '46MHz_sources_survivors.csv')
    txt = open(p).read()
    open(p, 'w').write(txt.replace(r46[0]['forced_jy'], '9.99000', 1))
    res = validate.run('catalog_forced_columns', p)
    check('catflux content test refuses a forced_jy that is not the re-measurement', not res[0]['passed'],
          res[0]['detail'])
    open(p, 'w').write(txt)
    shutil.rmtree(os.path.join(U, 'vresid', '04h_46MHz'))
    res = validate.run('catalog_forced_columns', p)
    check('catflux content test refuses a measured row whose frame is gone', not res[0]['passed'], res[0]['detail'])
    # D-G22 interface: catalog finds this done record under <run>/_records/forced
    from glob import glob
    done = [json.load(open(x)).get('status') for x in glob(os.path.join(U, '_records', 'forced', '*.json'))]
    check('catalog D-G22 interface: a done forced record under <run>/_records/forced', 'done' in done)


# ---------------------------------------------------------------------------------------------------------------------
# 5. snapshot: synthetic run files, the six content tests and a mutation each; per-hour no-input; G35
# ---------------------------------------------------------------------------------------------------------------------

def _png(path, w=1600, h=1400):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + struct.pack('>II', w, h) + b'\x08\x02\x00\x00\x00')


def _w(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r.get(c, '') for c in cols})


def _snap_run(root):
    """A synthetic snapshot forced run (Stokes V, 10s) in the production layout, consistent across its files."""
    ev_sb, ev_wb = os.path.join(root, 'eval_sb'), os.path.join(root, 'eval_wb')
    th = [dict(sample='10pc', stokes='V', tier='10s', thr_survey_gauss='5.0', n_target_trials='1000'),
          dict(sample='10pc', stokes='V', tier='1min', thr_survey_gauss='5.0', n_target_trials='100')]
    tcols = ['cell', 'sample', 'target', 'band', 'stokes', 'tier', 'ts', 'snr', 'screen', 'sample_frac_ge', 'p_gauss']
    sel = [dict(cell='04h_2025-01-01', sample='10pc', target='StarA', band='46MHz', stokes='V', tier='10s',
                ts='20250101_040000', snr='8.0', screen='clean', sample_frac_ge='1e-6', p_gauss='1e-9'),
           dict(cell='04h_2025-01-01', sample='10pc', target='StarB', band='55MHz', stokes='V', tier='10s',
                ts='20250101_040010', snr='-6.0', screen='clean', sample_frac_ge='1e-6', p_gauss='1e-9'),
           dict(cell='04h_2025-01-01', sample='10pc', target='StarA', band='46MHz', stokes='V', tier='10s',
                ts='20250101_040020', snr='4.0', screen='clean', sample_frac_ge='1e-6', p_gauss='1e-9'),     # below thr
           dict(cell='05h_2025-01-02', sample='10pc', target='StarA', band='46MHz', stokes='V', tier='10s',
                ts='20250102_050000', snr='9.0', screen='zone', sample_frac_ge='1e-6', p_gauss='1e-9')]      # zone
    for ev, rows in ((ev_sb, sel), (ev_wb, [])):
        _w(os.path.join(ev, 'thresholds.csv'), list(th[0]), th)
        _w(os.path.join(ev, 'targets_ge4.csv'), tcols, rows)
    W = os.path.join(root, 'work')
    R = os.path.join(W, 'Run_SNAPFORCED_V10s_TEST')
    os.makedirs(R, exist_ok=True)
    with open(os.path.join(R, 'queue.txt'), 'w') as fh:
        fh.write('# forced photometry (HWJ + 10 pc), Stokes V, 10s tier only, clean screen, S/N (V: |S/N|) >= Gaussian '
                 'survey threshold; eval %s + %s\nStarA\nStarB\n' % (ev_sb, ev_wb))
    dcols = ['lst', 'date', 'band', 'subband', 'stokes', 'ts', 'snr', 'ra', 'dec', 'tier', 'cand_id', 'sample', 'target',
             'identity']
    eva = dict(lst='04h', date='2025-01-01', band='46MHz', subband='46MHz', stokes='V', ts='20250101_040000', snr='8.0',
               ra='10.00000', dec='20.00000', tier='10s', cand_id='F_04h_2025-01-01_46MHz_20250101_040000_StarA',
               sample='10pc', target='StarA', identity='StarA')
    evb = dict(lst='04h', date='2025-01-01', band='55MHz', subband='55MHz', stokes='V', ts='20250101_040010', snr='-6.0',
               ra='50.00000', dec='-10.00000', tier='10s', cand_id='F_04h_2025-01-01_55MHz_20250101_040010_StarB',
               sample='10pc', target='StarB', identity='StarB')
    _w(os.path.join(R, 'candidates', '46MHz', 'rank0000_46MHz_StarA', 'detections.csv'), dcols, [eva])
    _w(os.path.join(R, 'candidates', '55MHz', 'rank0001_55MHz_StarB', 'detections.csv'), dcols, [evb])
    _w(os.path.join(R, 'source_catalogs', '46MHz_sources_survivors.csv'), ['source_id', 'ra', 'dec'],
       [dict(source_id='StarA', ra='10.0', dec='20.0')])
    _w(os.path.join(R, 'source_catalogs', '55MHz_sources_survivors.csv'), ['source_id', 'ra', 'dec'],
       [dict(source_id='StarB', ra='50.0', dec='-10.0')])
    _w(os.path.join(R, 'skipped_events.csv'), list(CFG.forced_snapshot.SKIP_FIELDS), [])
    fcols = ['cand_id', 'identity', 'target', 'sample', 'cell', 'band', 'stokes', 'tier', 'ts', 'snr', 'fit_kind',
             'fit_status', 'fit_ra_deg', 'fit_dec_deg', 'astrometric']
    fa = dict(cand_id=eva['cand_id'], identity='StarA', target='StarA', sample='10pc', cell='04h_2025-01-01', band='46MHz',
              stokes='V', tier='10s', ts=eva['ts'], snr='8.0', fit_kind='gaussian', fit_status='ok',
              fit_ra_deg='10.01000', fit_dec_deg='20.01000', astrometric='1')
    fb = dict(cand_id=evb['cand_id'], identity='StarB', target='StarB', sample='10pc', cell='04h_2025-01-01', band='55MHz',
              stokes='V', tier='10s', ts=evb['ts'], snr='-6.0', fit_kind='template', fit_status='ok',
              fit_ra_deg='50.02000', fit_dec_deg='-10.02000', astrometric='')
    _w(os.path.join(R, 'forced_fits.csv'), fcols, [fa, fb])
    _w(os.path.join(R, 'master_catalog.csv'), ['cand_id', 'size_beams', 'axis_ratio'],
       [dict(cand_id=fa['cand_id'], size_beams='1.05', axis_ratio='1.10'), dict(cand_id=fb['cand_id'])])
    tc = ['cand_id', 'tie_status', 'ra_tied', 'dec_tied', 'tie_dx_arcmin', 'tie_dy_arcmin', 'used_dx_arcmin',
          'used_dy_arcmin']
    _w(os.path.join(R, 'forced_ties.csv'), tc,
       [dict(cand_id=fa['cand_id'], tie_status='tied', ra_tied='10.02000', dec_tied='20.02000', tie_dx_arcmin='+0.50',
             tie_dy_arcmin='-0.20', used_dx_arcmin='+0.50', used_dy_arcmin='-0.20'),
        dict(cand_id=fb['cand_id'], tie_status='untied-few')])
    _w(os.path.join(R, 'source_catalogs', 'localizations.csv'), ['identity_id', 'n_det', 'ra', 'dec'],
       [dict(identity_id='StarA', n_det='1', ra='10.02', dec='20.02')])
    _w(os.path.join(R, 'forced_assoc_identity.csv'), ['identity_id', 'label', 'ra', 'dec'],
       [dict(identity_id='StarA', label='target', ra='10.02000', dec='20.02000'),
        dict(identity_id='StarB', label='no radio position')])
    sa = dict(eva, ra='10.02000', dec='20.02000')
    sb = dict(evb, ra='50.02000', dec='-10.02000')
    _w(os.path.join(R, 'candidates', 'V', 'rank0000_StarA', 'detections.csv'), dcols, [sa])
    _w(os.path.join(R, 'candidates', 'V', 'rank0001_StarB', 'detections.csv'), dcols, [sb])
    _png(os.path.join(R, 'candidates', 'V', 'rank0000_StarA', 'StarA_04h_2025-01-01_46MHz.png'))
    _png(os.path.join(R, 'candidates', 'V', 'rank0001_StarB', 'StarB_04h_2025-01-01_55MHz.png'))
    open(os.path.join(R, 'missing_panels.csv'), 'w').close()
    return W, R, ev_sb, ev_wb


def _one(cls, path, **ctx):
    r = validate.run(cls, path, **ctx)
    return all(x['passed'] for x in r), '; '.join('%s %s' % (x['name'], x['detail']) for x in r)[:400]


def _mut(path, old, new):
    txt = open(path).read()
    assert old in txt, (path, old)
    open(path, 'w').write(txt.replace(old, new, 1))
    return txt


def test_snapshot_tests():
    W, R, ev_sb, ev_wb = _snap_run(os.path.join(TMP, 'snap'))
    cases = [('forced_event_build', os.path.join(R, 'queue.txt'), {}),
             ('forced_fits', os.path.join(R, 'forced_fits.csv'), {}),
             ('forced_ties', os.path.join(R, 'forced_ties.csv'), {}),
             ('forced_localization', os.path.join(R, 'source_catalogs', 'localizations.csv'), {}),
             ('forced_assoc', os.path.join(R, 'forced_assoc_identity.csv'), {'inplace': False}),
             ('forced_event_panels', os.path.join(R, 'missing_panels.csv'), {})]
    for cls, p, ctx in cases:
        ok, d = _one(cls, p, **ctx)
        check('%s passes on a consistent synthetic run' % cls, ok, d)
    muts = [('forced_event_build', os.path.join(R, 'skipped_events.csv'), 'cell,', 'cellx,', 'skipped_events columns'),
            ('forced_event_build', os.path.join(ev_sb, 'targets_ge4.csv'), '20250101_040020,4.0,', '20250101_040020,7.0,',
             'a selected evaluation row not built'),
            ('forced_event_build', os.path.join(R, 'queue.txt'), 'StarB\n', 'StarC\n', 'queued identity without events'),
            ('forced_fits', os.path.join(R, 'forced_fits.csv'), ',ok,10.01000,20.01000,1', ',ok,10.01000,20.01000,0',
             'astrometric rule'),
            ('forced_fits', os.path.join(R, 'master_catalog.csv'), '1.05,1.10', ',', 'ok fit without size/axis'),
            ('forced_ties', os.path.join(R, 'forced_ties.csv'), '+0.50,-0.20,+0.50', '+0.50,-0.20,+0.60',
             'reconstruction differs by 0.1 arcmin'),
            ('forced_localization', os.path.join(R, 'source_catalogs', 'localizations.csv'), 'StarA,1,', 'StarA,2,',
             'n_det differs'),
            ('forced_assoc', os.path.join(R, 'forced_assoc_identity.csv'), 'StarA,target', 'StarA,maybe', 'label undeclared'),
            ('forced_event_panels', os.path.join(R, 'candidates', 'V', 'rank0000_StarA', 'detections.csv'),
             '10.02000,20.02000', '10.00000,20.00000', 'served position = the target position, not the radio position')]
    for cls, p, old, new, what in muts:
        txt = _mut(p, old, new)
        path = dict((c, q) for c, q, _ in cases)[cls]
        ok, d = _one(cls, path, **({'inplace': False} if cls == 'forced_assoc' else {}))
        check('%s refuses a mutated run (%s)' % (cls, what), not ok, d)
        open(p, 'w').write(txt)
    _png(os.path.join(R, 'candidates', 'V', 'rank0000_StarA', 'StarA_04h_2025-01-01_46MHz.png'), h=900)
    ok, d = _one('forced_event_panels', os.path.join(R, 'missing_panels.csv'))
    check('forced_event_panels refuses a panel without its light-curve strip (900 px)', not ok, d)
    _png(os.path.join(R, 'candidates', 'V', 'rank0000_StarA', 'StarA_04h_2025-01-01_46MHz.png'))
    # the selection used for no-input matches the synthetic evaluation (2 of 4 rows: below threshold and zone excluded)
    from exopipe.phase3.lib.forced.tier_snapshot import selected_events
    check('selected_events = event_rows selection (clean, >= threshold, cadence)',
          len(selected_events((ev_sb, ev_wb), 'V', '10s')) == 2)
    return W, R, ev_sb, ev_wb


def test_snapshot_cli(W, ev_sb, ev_wb):
    base = ['forced', '--tier', 'snapshot', '--cores', 'none', '--out', W, '--snap-eval-sb', ev_sb, '--snap-eval-wb', ev_wb]
    rc, err = cli(base + ['--stokes', 'V', '--snap-cadence', '10s', '--snap-name', 'Run_SNAPFORCED_V10s_NEW', '--dry-run'])
    check('snapshot --dry-run on synthetic evaluations exits 0', rc == 0, err[-300:])
    rc, err = cli(base + ['--stokes', 'V', '--snap-cadence', '10s', '--snap-name', 'Run_SNAPFORCED_V10s_TEST',
                          '--stages', 'build'])
    rec = record.load_record(W, 'forced', 'C_build_Run_SNAPFORCED_V10s_TEST') or {}
    check('G35: a build on an existing run name is refused (exit 2) before any in-place call',
          rc == 2 and rec.get('status') == 'refused' and 'G35' in rec.get('reason', '') and not rec['meta'].get('legacy_calls'),
          '%s %s' % (rc, rec.get('reason')))
    rc, err = cli(base + ['--epochs', '06h_2025-01-03'])
    recs = [record.load_record(W, 'forced', 'C_build_Run_SNAPFORCED_%s_E06h_2025-01-03' % x) or {}
            for x in ('I10s', 'I1min', 'V10s', 'V1min')]
    check('per hour: an epoch without events -> 4 no-input units (I/V x 10s/1min), exit 3',
          rc == 3 and all(r.get('status') == 'no-input' for r in recs), '%s %s' % (rc, [r.get('status') for r in recs]))
    rc, err = cli(base + ['--stokes', 'V', '--snap-cadence', '10s', '--snap-name', 'X', '--stages', 'fit'])
    rec = record.load_record(W, 'forced', 'C_fit_X') or {}
    check('a stage without its previous stage\'s done record is refused', rc == 2 and 'no record of the build' in
          rec.get('reason', ''), '%s %s' % (rc, rec.get('reason')))
    rc, err = cli(['forced', '--tier', 'snapshot', '--cores', 'none', '--out', W, '--stokes', 'I,V',
                   '--snap-cadence', '10s', '--snap-name', 'Y', '--dry-run'])
    check('snapshot without --epochs refuses two Stokes in one run', rc == 2, err[-200:])


def test_deep_cli():
    rc, err = cli(['forced', '--tier', 'deep', '--cores', 'none', '--out', os.path.join(TMP, 'x'), '--dry-run'])
    check('deep without --run refused (G1, D7)', rc == 2 and 'G1' in err, err[-200:])
    rc, err = cli(['forced', '--tier', 'deep', '--run', 'P3S5', '--stages', 'catflux,panels', '--cores', 'none',
                   '--out', os.path.join(TMP, 'x'), '--dry-run'])
    check('catflux and panels in one invocation refused', rc == 2, err[-200:])
    rc, err = cli(['forced', '--tier', 'deep', '--run', 'P3S5', '--stages', 'panels', '--cores', 'none',
                   '--out', os.path.join(TMP, 'pages'), '--dry-run'])
    check('panels without rows (--deep-eval-run / --deep-targets-csv) refused', rc == 2 and '--deep-eval-run' in err,
          err[-200:])
    os.makedirs(os.path.join(TMP, 'phot', '_records', 'photometry'), exist_ok=True)
    tcsv = os.path.join(TMP, 'phot', 'out', 'eval', 'targets_above_threshold.csv')
    os.makedirs(os.path.dirname(tcsv), exist_ok=True)
    open(tcsv, 'w').write('threshold,sample,target\n')
    json.dump({'unit': 'evaluate', 'status': 'done', 'outputs': [{'path': tcsv, 'class': 'photometry_eval'}]},
              open(os.path.join(TMP, 'phot', '_records', 'photometry', 'evaluate.json'), 'w'))
    rc, err = cli(['forced', '--tier', 'deep', '--run', 'P3S5', '--stages', 'panels', '--deep-eval-run',
                   os.path.join(TMP, 'phot'), '--cores', 'none', '--out', os.path.join(TMP, 'pages'), '--dry-run'])
    check('panels reads the rows from photometry\'s evaluate record (dry run exit 0)', rc == 0, err[-300:])
    rc, err = cli(['forced', '--tier', 'deep', '--run', 'P3S5', '--cells', '04h_Full', '--cores', 'none',
                   '--out', os.path.join(TMP, 'x'), '--dry-run'])
    check('deep stages refuse --cells', rc == 2, err[-200:])
    rc, err = cli(['forced', '--tier', 'sublst', '--cores', 'none', '--out', os.path.join(TMP, 'x')])
    check('sublst: no production chain -> exit 3', rc == 3, err[-200:])
    rc, err = cli(['forced', '--tier', 'deep', '--run', 'P3S5', '--cores', 'none', '--set',
                   'forced_catflux.DECIMALS=3', '--out', os.path.join(TMP, 'x'), '--dry-run'])
    check('cli: override of an in-place literal refused (G37, exit 2)', rc == 2 and 'G37' in err, err[-200:])


def test_import_side_effects():
    sys.path.insert(0, os.path.join(REPO, 'tests'))
    import test_import_side_effects as T
    mine = [b for b in T.ast_violations() if '/forced' in b[0] or '_cfgdraft_forced' in b[0]]
    check('no module-level side effects in the forced files (AST layer of the standing test)', not mine, str(mine))


def main():
    t0 = time.time()
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    try:
        test_config()
        test_inplace_truth()
        test_stages()
        test_catflux()
        W, R, ev_sb, ev_wb = test_snapshot_tests()
        test_snapshot_cli(W, ev_sb, ev_wb)
        test_deep_cli()
        test_import_side_effects()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    bad = [r for r in RESULTS if not r[1]]
    print('\n%d checks, %d failed, %.0f s' % (len(RESULTS), len(bad), time.time() - t0))
    for n, _, d in bad:
        print('FAILED: %s  %s' % (n, d))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
