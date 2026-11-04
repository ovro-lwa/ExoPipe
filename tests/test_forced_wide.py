#!/usr/bin/env python3
"""Tests of `exopipe3 forced` families W (the forced runs' wide-area burst test) and N (the night-activity table) - plain
asserts, runnable as `python tests/test_forced_wide.py`, a few minutes.

Synthetic data in a fresh /tmp/exopipe3_tests_forced_wide/ (removed at the end); READ-ONLY checks of the production modules the
stages call in place and of the analysis scripts they were ported from (config truth: the defaults are the approved method's
literals; the mirrors are the in-place literals). The real radio catalogues (forced_assoc.RADIO_CATDIR, pinned) are read where a
test needs a real cataloged source. No processing on real data, no ssh, no GPU.

Run with: NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py \\
          /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_forced_wide.py
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
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.core import env as _env       # noqa: E402

_env.legacy()
from exopipe.phase3.config import CFG, ConfigRefusal   # noqa: E402
from exopipe.phase3.core import validate, record       # noqa: E402

import numpy as np                                     # noqa: E402

TMP = '/tmp/exopipe3_tests_forced_wide'
OPS = '/lustre/gh/main/exopipe_ops'
CLV = '/lustre/gh/main/clv15'
ANA = '/lustre/gh/main/phase3/refactor_analysis/review_items_20261031'
RESULTS = []
# KELT-24 (I, 10 s) as the analysis found it (jobs.json, results_v3.csv; the served forced_wide_v3.csv's named_radec)
KELT24_T = [161.909, 71.6557]
KELT24_NAMED = "VLSSr J105246.5+715924 10.5 Jy at 31.3'; VLSSr J105425.8+715204 3.1 Jy at 34.3'"
KELT24_RADEC = '163.19400,71.99017;163.60771,71.86800'


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond), detail))
    print('%s  %s%s' % ('PASS' if cond else 'FAIL', name, ('  -- ' + str(detail)) if detail and not cond else ''), flush=True)


_LAST_OUT = ['']


def cli(argv, env=None):
    """exopipe3 in-process; returns (exit code, stderr text); stdout kept in _LAST_OUT. Overrides applied by a call are undone."""
    from exopipe.phase3 import cli as C
    err, out = io.StringIO(), io.StringIO()
    saved = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            try:
                rc = C.main(argv)
            finally:
                CFG.reset()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    _LAST_OUT[0] = out.getvalue()
    return rc, err.getvalue()


def rec(out, unit):
    return record.load_record(out, 'forced', unit) or {}


def write_csv(path, rows, cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def fake_record(out, unit, outputs, meta=None, status='done', inputs=None):
    """a minimal run record of a previous stage (the record chain reads status, outputs, meta and inputs)"""
    p = os.path.join(out, '_records', 'forced', unit + '.json')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump({'unit': unit, 'status': status, 'outputs': [{'path': a, 'class': b} for a, b in outputs], 'meta': meta or {},
               'inputs': inputs or {}}, open(p, 'w'))


@contextlib.contextmanager
def catdir(path):
    """snapshot_forced_assoc.CATDIR (fixed at its import from $P3_RADIO_CATDIR) pointed elsewhere for a block"""
    from exopipe.phase3.lib.forced import wide_jobs as WJ
    fa = WJ.FA()
    old = fa.CATDIR
    fa.CATDIR = path
    try:
        yield
    finally:
        fa.CATDIR = old


def runs8(root):
    """the eight forced run directories (empty) under root: a host-independent --wide-runs for dry runs"""
    for s_, c_ in CFG.forced_wide.AMBIGUOUS_RUNS:
        os.makedirs(os.path.join(root, CFG.forced_wide.RUN_NAME.format(S=s_, cadence=c_)), exist_ok=True)
    return root


# ---------------------------------------------------------------------------------------------------------------------
# 1. config
# ---------------------------------------------------------------------------------------------------------------------

def test_config():
    secs = CFG.sections()
    for s in ('forced_wide', 'forced_wide_search', 'forced_wide_test', 'night_activity'):
        check('config section %s loads' % s, s in secs)
    for k in ('forced_wide.PUBLISHED', 'forced_wide_search.SUN_SITE', 'night_activity.FILES'):
        check('%s protected (a mirror of the reader / the in-place code)' % k, CFG.protected(k))
        try:
            CFG.set(k, "'x'")
            ok = False
        except (ConfigRefusal, ValueError):
            ok = True
        CFG.reset()
        check('%s override refused' % k, ok)
    import importlib
    from exopipe.phase3 import config
    mod = importlib.import_module('exopipe.phase3.steps.forced')
    decl = config.declared_sections(mod)
    check('steps.forced CONFIG declares the W / N sections and photometry (the cutout root)',
          {'forced_wide', 'forced_wide_search', 'forced_wide_test', 'night_activity', 'photometry'} <= set(decl), decl)
    f = CFG.forced
    check('snapshot stages: family C, then W, then N', f.STAGES['snapshot'][-5:] == ('wide_jobs', 'wide_search', 'wide_test',
                                                                                      'wide_publish', 'activity'))
    check('W / N never a default nor per hour', not {'wide_jobs', 'activity'} & set(f.DEFAULT_STAGES['snapshot'] +
                                                                                   f.EPOCH_STAGES['snapshot']))
    t = CFG.forced_wide_test
    check('the approved amplitude bound MU (forced_wide_v3.py:80)',
          t.MU == {('I', '10s'): 4.0, ('I', '1min'): 2.5, ('V', '10s'): 2.5, ('V', '1min'): 2.0})
    from exopipe.phase3.core import ledger
    V3ID, ORDID = 'R-GREGG-20261103-FORCED-WIDE-V3-APPROVED', 'R-REFACTOR-20261104-FORCED-WIDE-CONTROLS-ORDER-INDEPENDENT'
    AMDID = ORDID + '-AMENDED-PRECISION-KNIFE-EDGE'
    MC = ('TAPER_CTL_MAX', 'BOOT_Z_SE', 'BOOT_MAX', 'BOOT_DRAW', 'KNIFE_EDGE_SE', 'KNIFE_EDGE_DRAWS', 'KNIFE_EDGE_DRAWS_MAX',
          'KNIFE_EDGE_DECIDE')
    ruled = {(sec, k): v for sec in ('forced_wide', 'forced_wide_search', 'forced_wide_test', 'night_activity')
             for k, v in (getattr(CFG._section(sec), 'RULED', None) or {}).items()}
    check('RULED: CONTROL_ORDER and V2_CTL_DRAW cite the refactor decision, the Monte Carlo keys its amendment, the rest '
          'Gregg\'s v3 approval',
          ruled[('forced_wide_test', 'CONTROL_ORDER')] == ORDID and ruled[('forced_wide_test', 'V2_CTL_DRAW')] == ORDID
          and all(ruled[('forced_wide_test', k)] == AMDID for k in MC)
          and {v for k, v in ruled.items() if k[1] not in ('CONTROL_ORDER', 'V2_CTL_DRAW') + MC} == {V3ID}
          and not any(k[0] == 'night_activity' for k in ruled), sorted(set(ruled.values())))
    check('the 900-line cap is gone: TAPER_CTL_MAX None by default, no CTL_CAP key', t.TAPER_CTL_MAX is None
          and not hasattr(t, 'CTL_CAP'))
    check('the bootstrap: its own seed per (target, tier), until the centroid z\'s Monte Carlo error <= 0.02; the knife-edge '
          'window 4 errors, its rows decided by draws (20 at a time, up to 100)', (t.BOOT_DRAW, t.BOOT_Z_SE, t.BOOTSTRAP,
                                                                                  t.KNIFE_EDGE_SE, t.KNIFE_EDGE_DRAWS,
                                                                                  t.KNIFE_EDGE_DRAWS_MAX, t.KNIFE_EDGE_DECIDE)
          == ('target', 0.02, 1000, 4.0, 20, 100, 0.95)
          and t.BOOT_MAX >= 100 * t.BOOTSTRAP)
    check('RULED: the approved method\'s thresholds are ruled (MU, the taper, the centroid, v2), operations are not',
          all(('forced_wide_test', k) in ruled for k in ('MU', 'P_CONSISTENT', 'R_EXCESS', 'CEN_FLOOR', 'Z_OFF', 'V2_POWER_MIN',
                                                          'BOOTSTRAP')) and ('forced_wide_search', 'SNR_MIN') in ruled
          and ('forced_wide', 'EXT_MIN_S74_JY') in ruled and not any(k[1] in ('WORKERS', 'MAX_WORKERS', 'SEED', 'MINUTE_RULE',
                                                                            'TARGET_MEMBER', 'EVENT_ORDER', 'FILE_ERRORS')
                                                                     for k in ruled))
    check('the three ruling IDs are rows of the fix ledger', ledger.has_id(V3ID, CFG.paths.FIX_LEDGER)
          and ledger.has_id(ORDID, CFG.paths.FIX_LEDGER) and ledger.has_id(AMDID, CFG.paths.FIX_LEDGER))
    # production's values (parity config, # WAVE2) apply and are the three PROPOSED keys
    p = os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg')
    lines = [ln.split('#')[0].strip() for ln in open(p) if ln.split('#')[0].strip().startswith(('forced_wide', 'night_activity'))]
    check('parity config: production values of family W', sorted(lines) == sorted([
        "forced_wide.EVENT_ORDER = 'directory'", "forced_wide.TARGET_MEMBER = 'first'",
        "forced_wide_search.FILE_ERRORS = 'drop'", "forced_wide_test.V2_CTL_DRAW = (300, 20261102)",
        "forced_wide_test.CONTROL_ORDER = 'pool'", "forced_wide_test.TAPER_CTL_MAX = 900", "forced_wide_test.BOOT_DRAW = 'job'",
        "forced_wide_test.BOOT_Z_SE = None", "forced_wide_test.KNIFE_EDGE_SE = None",
        "forced_wide_test.KNIFE_EDGE_DRAWS = None"]), lines)
    CFG.authorize([], out=TMP)                 # two of them are RULED: overridden into a non-production --out, as parity runs do
    for ln in lines:
        k, v = ln.split('=', 1)
        CFG.set(k.strip(), v.strip())
    t = CFG.forced_wide_test
    from exopipe.phase3.lib.forced import wide_v3 as V3
    try:
        V3.mc_settings()
        V3.v2_draw()
        V3.control_order()
        ok_ = True
    except ValueError as e:                                # the parity run's own key checks (round 7: one refused it)
        ok_ = str(e)
    check('the parity config values pass the stage\'s key checks (mc_settings, v2_draw, control_order)', ok_ is True, ok_)
    check('the parity config values apply', (CFG.forced_wide.EVENT_ORDER, CFG.forced_wide.TARGET_MEMBER,
                                             CFG.forced_wide_search.FILE_ERRORS, t.V2_CTL_DRAW, t.CONTROL_ORDER,
                                             t.TAPER_CTL_MAX, t.BOOT_DRAW, t.BOOT_Z_SE, t.KNIFE_EDGE_SE, t.KNIFE_EDGE_DRAWS)
          == ('directory', 'first', 'drop', (300, 20261102), 'pool', 900, 'job', None, None, None))
    CFG.reset()
    check('the controls are order-independent by default', CFG.forced_wide_test.CONTROL_ORDER == 'canonical')
    CFG.authorize([], out='/lustre/gh/main/phase3/x')
    try:
        CFG.set('forced_wide_test.CONTROL_ORDER', 'pool')
        ok = False
    except ConfigRefusal:
        ok = True
    CFG.reset()
    check('CONTROL_ORDER is RULED: an override into a production root needs --ruling-for', ok)
    check('v2 controls: the default is every eligible pair (no random cap)', CFG.forced_wide_test.V2_CTL_DRAW is None)
    check('the test caps its processes like the search (W-WORKERS)', CFG.forced_wide_test.MAX_WORKERS == 6
          == CFG.forced_wide_search.MAX_WORKERS)
    w = CFG.forced_wide
    check('the ambiguous list reads the eight forced runs, the tested four first (forced_ambiguous.py:28)',
          w.AMBIGUOUS_RUNS[:4] == w.RUNS and [c for _s, c in w.AMBIGUOUS_RUNS[4:]] == ['10min', '10min', 'deep', 'deep'])


# ---------------------------------------------------------------------------------------------------------------------
# 2. config truth: the defaults are the analysis' literals (the approved method), the mirrors the in-place literals
# ---------------------------------------------------------------------------------------------------------------------

def _assigns(path):
    """{NAME: value} of a file's module-level literal assignments (tuple targets unpacked)."""
    out = {}
    for node in ast.parse(open(path).read(), path).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tg = node.targets[0]
            try:
                v = ast.literal_eval(node.value)
            except Exception:
                continue
            if isinstance(tg, ast.Name):
                out[tg.id] = v
            elif isinstance(tg, ast.Tuple) and isinstance(v, tuple) and len(v) == len(tg.elts):
                for e, x in zip(tg.elts, v):
                    if isinstance(e, ast.Name):
                        out[e.id] = x
    return out


def test_method_truth():
    fw = os.path.join(ANA, 'forced_wide')
    if not os.path.isdir(fw):
        check('analysis scripts readable (config truth skipped)', True)
        return
    t, s, w = CFG.forced_wide_test, CFG.forced_wide_search, CFG.forced_wide
    v3 = _assigns(os.path.join(fw, 'v3', 'forced_wide_v3.py'))
    pairs = [('MU', t.MU), ('R_CAND', t.R_CAND), ('NEAR', t.NEAR), ('DISC', t.DISC), ('STRIP', t.STRIP), ('BIN', t.BIN),
             ('SIDE', t.SIDE), ('S_MIN', t.S_MIN_JY), ('CTL_FLUX', t.CTL_FLUX), ('CTL_MIN_JY', t.CTL_MIN_JY),
             ('CTL_DIRS', t.CTL_DIRS), ('MIN_NOBS', t.MIN_NOBS), ('CUT', t.CUT_HALF_ARCMIN), ('P_CONSISTENT', t.P_CONSISTENT),
             ('P_EXCESS', t.P_EXCESS), ('P_NIGHT', t.P_NIGHT), ('R_EXCESS', t.R_EXCESS), ('CEN_DET', t.CEN_DET),
             ('CEN_FLOOR', t.CEN_FLOOR), ('CEN_NIGHT', t.CEN_NIGHT)]
    bad = [(k, v3.get(k), v) for k, v in pairs if v3.get(k) != v]
    check('forced_wide_test = forced_wide_v3.py\'s literals (%d keys)' % len(pairs), not bad, bad)
    src = open(os.path.join(fw, 'v3', 'forced_wide_v3.py')).read()
    par = {ln.split('#')[0].split('=', 1)[0].strip(): ast.literal_eval(ln.split('#')[0].split('=', 1)[1].strip())
           for ln in open(os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg'))
           if ln.startswith('forced_wide_test.')}
    for lit, ok in (('default_rng(20261102)', t.SEED == 20261102), ('for _ in range(1000)', t.BOOTSTRAP == 1000
                                                                    and par.get('forced_wide_test.BOOT_Z_SE', 0) is None),
                    ('cap=900', par.get('forced_wide_test.TAPER_CTL_MAX') == 900),
                    ('rng.integers(0, nn, nn)', par.get('forced_wide_test.BOOT_DRAW') == 'job'),
                    ('COVER = 2.0 / 3.0', t.COVER == 2.0 / 3.0),
                    ('round(s[0] * 20) / 20', t.CTL_KEY_PER_DEG == 20), ('np.linspace(-DISC, DISC, 41)', t.DISC_GRID == 41),
                    ('if frac >= 0.5:', t.CAPABLE_FRACTION == 0.5), ('s74 * (74.0 / float(band[:-3])) ** 0.7',
                                                                    (t.REF_MHZ, t.SPECTRAL_INDEX) == (74.0, 0.7))):
        check('v3 literal %r = its config key' % lit, lit in src and ok)
    se = _assigns(os.path.join(fw, 'forced_wide_search.py'))
    pairs = [('STAMP', s.STAMP_PX), ('CEN', s.STAMP_CORE_PX), ('MIN_STAMP_FRAC', s.STAMP_MIN_FRAC), ('EDGE', s.EDGE_PX),
             ('SNR_MIN', s.SNR_MIN), ('PRE_K', s.PRE_SIGMA)]
    bad = [(k, se.get(k), v) for k, v in pairs if se.get(k) != v]
    check('forced_wide_search = forced_wide_search.py\'s literals', not bad, bad)
    ssrc = open(os.path.join(fw, 'forced_wide_search.py')).read()
    check("MINUTE_RULE = the search's default 'filename'", "os.environ.get('FW_MINUTE_RULE', 'filename')" in ssrc
          and s.MINUTE_RULE == 'filename')
    v2 = _assigns(os.path.join(fw, 'forced_wide_analyze.py'))
    pairs = [('CTL_FLUX', t.V2_CTL_FLUX), ('CTL_BRIGHTER', t.V2_CTL_BRIGHTER), ('CTL_MIN_CELLS', t.V2_CTL_MIN_CELLS),
             ('CTL_MIN_N', t.V2_CTL_MIN_N), ('MIN_CELLS_DECIDE', t.V2_MIN_NIGHTS),
             ('CUT_HALF', t.CUT_HALF_ARCMIN), ('MARGIN', t.V2_CUT_MARGIN_ARCMIN)]
    bad = [(k, v2.get(k), v) for k, v in pairs if v2.get(k) != v]
    check('forced_wide_test V2_* = forced_wide_analyze.py\'s literals', not bad and max(v2.get('KS')) == t.V2_K_MAX
          and t.V2_K in v2.get('KS'), bad)
    # production's capped draw = the parity config's V2_CTL_DRAW (the default draws none: every eligible pair)
    v2src = open(os.path.join(fw, 'forced_wide_analyze.py')).read()
    m = re.search(r'if len\(out\) > CTL_MAX:\s*\n\s*rng = np\.random\.default_rng\((\d+)\)', v2src)
    pl = [ln.split('#')[0].split('=', 1)[1].strip() for ln in open(os.path.join(REPO, 'exopipe', 'phase3', 'parity',
                                                                                   'production_20261013.cfg'))
          if ln.startswith('forced_wide_test.V2_CTL_DRAW')]
    check('parity V2_CTL_DRAW = forced_wide_analyze.py\'s (CTL_MAX, its draw\'s seed)',
          m is not None and pl and ast.literal_eval(pl[0]) == (v2.get('CTL_MAX'), int(m.group(1))), (pl, m and m.group(1)))
    jo = _assigns(os.path.join(fw, 'forced_wide_jobs.py'))
    check('GROUP_ARCMIN = forced_wide_jobs.py\'s', jo.get('GROUP_ARCMIN') == w.GROUP_ARCMIN)
    na = open(os.path.join(ANA, 'figure_v2', 'night_activity_build.py')).read()
    check('night_activity.DETECTIONS = night_activity_build.py\'s SRC', "SRC = '%s'" % CFG.night_activity.DETECTIONS in na)


def test_mirrors():
    src = open(os.path.join(CLV, 'phase3_snapshot_search.py')).read()
    m = re.search(r'def sun_state.*?EarthLocation\(lat=([-\d.]+) \* u\.deg, lon=([-\d.]+) \* u\.deg, height=([-\d.]+) \* u\.m\)',
                  src, re.S)
    check('forced_wide_search.SUN_SITE = phase3_snapshot_search.sun_state\'s location',
          m and (float(m.group(1)), float(m.group(2)), float(m.group(3))) == CFG.forced_wide_search.SUN_SITE,
          m.groups() if m else None)
    # the figure: source_astrometry_panel.py (v2 since exopipe_ops 7e6904a, 2026-11-04; source_astrometry_panel_v2.py its alias)
    s2 = open(os.path.join(OPS, 'source_astrometry_panel.py')).read()
    check('forced_wide.PUBLISHED = the name the figure reads', "os.path.join(U, '%s')" % CFG.forced_wide.PUBLISHED in s2)
    aa = open(os.path.join(OPS, 'astrometry_assoc.py')).read()
    m = re.search(r"NIGHT_ACT = os\.environ\.get\('P3_NIGHT_ACTIVITY', '([^']+)'\)", aa)
    na = CFG.night_activity
    check('night_activity OUT_ROOT / FILES[0] = astrometry_assoc.NIGHT_ACT', m and m.group(1) == os.path.join(na.OUT_ROOT,
                                                                                                             na.FILES[0]))
    check('night_activity FILES[1] = astrometry_assoc.NIGHT_ACT_NIGHTS', "NIGHT_ACT.replace('%s', '%s')" % na.FILES in aa)
    v3m = re.search(r"V3_MAP = \{'BACKGROUND SOURCE'", aa)
    check('the figure maps BACKGROUND SOURCE and NOT CENTERED ON THE TARGET (source not identified)', v3m is not None and
          "'NOT CENTERED ON THE TARGET (source not identified)'" in aa)
    from exopipe.phase3.lib.forced.wide_v3 import VERDICTS
    check('both mapped verdicts are v3 verdicts', 'BACKGROUND SOURCE' in VERDICTS and
          'NOT CENTERED ON THE TARGET (source not identified)' in VERDICTS)


# ---------------------------------------------------------------------------------------------------------------------
# 3. stage selection and the families' gates (CLI)
# ---------------------------------------------------------------------------------------------------------------------

class A:
    def __init__(self, **kw):
        self.stages = kw.get('stages')
        self.stage_from = kw.get('stage_from')
        self.stage_until = kw.get('stage_until')
        self.epochs = kw.get('epochs')


def test_families():
    from exopipe.phase3.lib.forced import wide_stages as WS
    from exopipe.phase3.core.record import Refusal
    check('no --stages: family C (the event panels)', WS.family(A())[0] == 'C')
    check('wide stages: family W, pipeline order', WS.family(A(stages='wide_test,wide_jobs')) == ('W', ['wide_jobs', 'wide_test']))
    check('activity: family N', WS.family(A(stages='activity'))[0] == 'N')
    for st in ('build,wide_jobs', 'wide_publish,activity'):
        try:
            WS.family(A(stages=st))
            ok = False
        except Refusal as e:
            ok = 'W-FAMILY' in str(e)
        check('W-FAMILY: %s refused (one family per invocation)' % st, ok)
    check('--wide-only parses S:IDENTITY and S:*', WS.parse_only('I:KELT-24,V:*') == {('I', 'KELT-24'), ('V', '*')})
    check('selected(): S:* selects every job of that Stokes', WS.selected({'stokes': 'V', 'identity': 'x'}, {('V', '*')})
          and not WS.selected({'stokes': 'I', 'identity': 'x'}, {('V', '*')}))


def test_cli_gates():
    out = os.path.join(TMP, 'w')
    base = ['forced', '--tier', 'snapshot', '--cores', 'none', '--out', out]
    rc, err = cli(base + ['--stages', 'wide_jobs,build', '--dry-run'])
    check('cli: stages of two families refused (exit 2)', rc == 2 and 'W-FAMILY' in err, err[-300:])
    rc, err = cli(base + ['--stages', 'wide_test', '--epochs', '04h_2025-10-14'])
    check('cli: a W stage per hour refused', rc == 2 and 'W-FAMILY' in err and '--epochs' in err, err[-300:])
    rc, err = cli(base + ['--stages', 'wide_search', '--stokes', 'I'])
    check('cli: --stokes refused for W (use --wide-only)', rc == 2 and '--wide-only' in err, err[-300:])
    r8 = runs8(os.path.join(TMP, 'runs8'))
    for s_, c_ in CFG.forced_wide.RUNS:                # the publication's run directories under <out>/review (a clean dry run)
        os.makedirs(os.path.join(out, 'review', CFG.forced_wide.RUN_NAME.format(S=s_, cadence=c_)), exist_ok=True)
    rc, err = cli(base + ['--stages', 'wide_jobs,wide_search,wide_test,wide_publish', '--wide-runs', r8, '--dry-run'])
    check('cli: W dry run exit 0, plans every stage', rc == 0 and all(('[%s]' % s) in _LAST_OUT[0] for s in
                                                                      ('wide_jobs', 'wide_search', 'wide_test', 'wide_publish')),
          (rc, err[-300:], _LAST_OUT[0][-400:]))
    check('dry run: the jobs plan names all eight runs, the tier runs for their queue only',
          _LAST_OUT[0].count('[wide_jobs]   run ') == 8 and _LAST_OUT[0].count('its queue only') == 4)
    shutil.rmtree(os.path.join(r8, 'Run_SNAPFORCED_Vdeep_gen5'))
    rc, err = cli(base + ['--stages', 'wide_jobs', '--wide-runs', r8, '--dry-run'])
    check('dry run: a missing tier run is planned as W-HOST, exit 2', rc == 2 and 'W-HOST' in _LAST_OUT[0]
          and 'Run_SNAPFORCED_Vdeep_gen5' in _LAST_OUT[0], (rc, _LAST_OUT[0][-300:]))
    runs8(r8)
    rc, err = cli(base + ['--stages', 'wide_jobs', '--wide-runs', r8, '--dry-run', '--set',
                          "forced_wide.AMBIGUOUS_RUNS=(('I','10s'),('V','10s'),('I','1min'))"])
    check('dry run: AMBIGUOUS_RUNS without a tested run is planned as W-RUNS, exit 2', rc == 2 and 'W-RUNS' in _LAST_OUT[0],
          (rc, _LAST_OUT[0][-300:]))
    rc, err = cli(base + ['--stages', 'wide_search', '--wide-cutout-root', '/lustre/pipeline/exopipe/phase3/photometry',
                          '--dry-run'])
    check('dry run: a retired cutout root is planned as W-CUTROOT (exit 2), its files never listed (review F6)',
          rc == 2 and 'W-CUTROOT' in _LAST_OUT[0] and 'cutout files' not in _LAST_OUT[0], (rc, _LAST_OUT[0][-300:]))
    rc, err = cli(base + ['--stages', 'wide_test', '--workers', '7', '--dry-run'])
    check('dry run: wide_test with 7 processes is planned as W-WORKERS, exit 2 (review F3)', rc == 2
          and 'W-WORKERS' in _LAST_OUT[0], (rc, _LAST_OUT[0][-300:]))
    rc, err = cli(base + ['--stages', 'wide_search', '--dry-run', '--set',
                          "forced_wide_search.IO_GUARD_LOG='%s'" % os.path.join(TMP, 'no_such_guard.log')])
    check('dry run: a set but missing I/O-guard log is said in the plan (review F7)', rc == 0 and 'ABSENT' in _LAST_OUT[0],
          (rc, _LAST_OUT[0][-300:]))
    rc, err = cli(base + ['--stages', 'wide_search'])
    check('cli: wide_search without the jobs record refused (W-RECORD, exit 2)', rc == 2 and 'W-RECORD' in err, err[-300:])
    rc, err = cli(base + ['--stages', 'wide_jobs'], env={'P3_RADIO_CATDIR': '/tmp/x'})
    r = rec(out, 'W_jobs')
    check('cli: W-ENV refuses a caller P3_RADIO_CATDIR', rc == 2 and r.get('status') == 'refused' and 'W-ENV' in r.get('reason', ''),
          (rc, r.get('reason')))
    # W-WORKERS: a fabricated jobs record, 7 processes
    jobs = os.path.join(out, 'jobs.json')
    os.makedirs(out, exist_ok=True)
    json.dump({'base': [], 'ext': [{'stokes': 'I', 'identity': 'X', 'target': 'X', 'sample': 'HWJ', 'T': [10.0, 20.0],
                                    'bands': ['27MHz'], 'bright': []}], 'runs': []}, open(jobs, 'w'))
    fake_record(out, 'W_jobs', [(jobs, 'forced_wide_jobs'), (os.path.join(out, 'events.csv'), 'forced_wide_events')])
    rc, err = cli(base + ['--stages', 'wide_search', '--workers', '7'])
    check('cli: W-WORKERS more than 6 processes refused', rc == 2 and 'W-WORKERS' in err, err[-300:])
    rc, err = cli(base + ['--stages', 'wide_search', '--wide-cutout-root', '/lustre/pipeline/exopipe/phase3/photometry'])
    check('cli: W-CUTROOT a retired cutout generation refused', rc == 2 and 'W-CUTROOT' in err, err[-300:])
    rc, err = cli(base + ['--stages', 'wide_test'])
    r = rec(out, 'W_test')
    check('cli: W-POOL a job band without its search record refused', rc == 2 and 'W-POOL' in r.get('reason', ''),
          (rc, r.get('reason')))
    rc, err = cli(base + ['--stages', 'wide_test', '--workers', '7'])
    check('cli: W-WORKERS wide_test with more than 6 processes refused (review F3)', rc == 2 and 'W-WORKERS' in err, err[-300:])
    rc, err = cli(base + ['--stages', 'wide_test', '--set', 'forced_wide_test.V2_CTL_DRAW=(0, 1)'])
    check('cli: a V2_CTL_DRAW that is neither None nor (cap >= 1, seed) refused', rc == 2 and 'V2_CTL_DRAW' in err, err[-300:])
    for kv, name in (('forced_wide_test.TAPER_CTL_MAX=0', 'TAPER_CTL_MAX'), ("forced_wide_test.BOOT_DRAW='x'", 'BOOT_DRAW'),
                     ('forced_wide_test.BOOT_Z_SE=-0.1', 'BOOT_Z_SE'), ('forced_wide_test.BOOT_MAX=10', 'BOOT_MAX'),
                     ('forced_wide_test.KNIFE_EDGE_SE=0.0', 'KNIFE_EDGE_SE'), ('forced_wide_test.KNIFE_EDGE_DRAWS=1', 'KNIFE_EDGE_DRAWS'),
                     ('forced_wide_test.KNIFE_EDGE_DECIDE=0.4', 'KNIFE_EDGE_DECIDE'),
                     ('forced_wide_test.KNIFE_EDGE_DRAWS_MAX=30', 'KNIFE_EDGE_DRAWS_MAX')):
        rc, err = cli(base + ['--stages', 'wide_test', '--set', kv])
        check('cli: %s refused before any job runs' % kv, rc == 2 and name in err, err[-300:])
    with catdir('/tmp/not_the_radio_catalogues'):
        rc, err = cli(base + ['--stages', 'wide_test'])
    r = rec(out, 'W_test')
    check('cli: W-CATDIR wide_test refuses catalogues other than forced_assoc.RADIO_CATDIR (review F4)',
          rc == 2 and 'W-CATDIR' in r.get('reason', ''), (rc, r.get('reason')))
    # W-PARTIAL / W-RUNDIR / the publication
    res = os.path.join(out, 'results_v3.csv')
    fake_record(out, 'W_test', [(res, 'forced_wide_results')], meta={'partial': True, 'only': ['I:X']})
    rc, err = cli(base + ['--stages', 'wide_publish'])
    r = rec(out, 'W_publish_Run_SNAPFORCED_I10s_gen5')
    check('cli: W-PARTIAL a partial test is never published', rc == 2 and 'W-PARTIAL' in r.get('reason', ''),
          (rc, r.get('reason')))


def test_publish():
    """the publication on a synthetic result of KELT-24 (the real catalogues find its named sources)"""
    from exopipe.phase3.lib.forced.tests_wide import RESULT_COLUMNS
    out = os.path.join(TMP, 'pub')
    jobs = os.path.join(out, 'jobs.json')
    os.makedirs(out, exist_ok=True)
    json.dump({'base': [], 'ext': [{'stokes': 'I', 'identity': 'KELT-24', 'target': 'KELT-24', 'sample': 'HWJ',
                                    'T': KELT24_T, 'bands': ['27MHz'], 'bright': []}], 'runs': []}, open(jobs, 'w'))
    fake_record(out, 'W_jobs', [(jobs, 'forced_wide_jobs'), (os.path.join(out, 'events.csv'), 'forced_wide_events')])
    row = dict.fromkeys(RESULT_COLUMNS, '')
    row.update(stokes='I', identity='KELT-24', target='KELT-24', tier='10s', n_events='68', event_jy_median='12.11',
               bursts_at_target='46', bursts_by_band='27MHz:16 32MHz:13 36MHz:12 50MHz:3 55MHz:2', n_candidates='23',
               n_capable='3', centroid_fwhm='0.141', centroid_arcmin='3.5', centroid_pa='94.0', centroid_err_fwhm='0.042',
               centroid_z='3.3', centroid_nights='9', verdict='BACKGROUND SOURCE', named=KELT24_NAMED, why='x')
    res = os.path.join(out, 'results_v3.csv')
    write_csv(res, [row], list(RESULT_COLUMNS))
    t = validate.run('forced_wide_results', res)
    check('results content test passes a consistent row', all(x['passed'] for x in t), t)
    fake_record(out, 'W_test', [(res, 'forced_wide_results')], meta={'partial': False})
    base = ['forced', '--tier', 'snapshot', '--cores', 'none', '--out', out, '--stages', 'wide_publish']
    rc, err = cli(base, env={'P3_RADIO_CATDIR': '/tmp/x'})
    r = rec(out, 'W_publish_Run_SNAPFORCED_I10s_gen5')
    check('cli: W-ENV wide_publish refuses a caller P3_RADIO_CATDIR (review F4)', rc == 2 and 'W-ENV' in r.get('reason', ''),
          (rc, r.get('reason')))
    with catdir('/tmp/not_the_radio_catalogues'):
        rc, err = cli(base)
    r = rec(out, 'W_publish_Run_SNAPFORCED_I10s_gen5')
    check('cli: W-CATDIR wide_publish refuses catalogues other than forced_assoc.RADIO_CATDIR (review F4)',
          rc == 2 and 'W-CATDIR' in r.get('reason', ''), (rc, r.get('reason')))
    rc, err = cli(base)
    r = rec(out, 'W_publish_Run_SNAPFORCED_I10s_gen5')
    check('cli: W-RUNDIR the run directory absent at the review root', rc == 2 and 'W-RUNDIR' in r.get('reason', ''),
          (rc, r.get('reason')))
    for s_, c_ in CFG.forced_wide.RUNS:
        os.makedirs(os.path.join(out, 'review', CFG.forced_wide.RUN_NAME.format(S=s_, cadence=c_)), exist_ok=True)
    rc, err = cli(base)
    p = os.path.join(out, 'review', 'Run_SNAPFORCED_I10s_gen5', 'forced_wide_v3.csv')
    got = list(csv.DictReader(open(p))) if os.path.exists(p) else []
    check('cli: publish done (exit 0), the I 10 s file holds the row', rc == 0 and len(got) == 1, (rc, err[-300:]))
    check('named_radec = the served file\'s for KELT-24 (the named sources found by name within 200\')',
          got and got[0]['named_radec'] == KELT24_RADEC, got[0]['named_radec'] if got else None)
    empty = list(csv.DictReader(open(os.path.join(out, 'review', 'Run_SNAPFORCED_V1min_gen5', 'forced_wide_v3.csv'))))
    check('a run without rows gets the header only (as production)', empty == [])
    r = rec(out, 'W_publish_Run_SNAPFORCED_I10s_gen5')
    check('publish record done, its output passed its content test', r.get('status') == 'done' and r['tests']
          and all(x['passed'] for x in r['tests']), r.get('reason'))
    # the content test refuses a row whose named source lies elsewhere
    bad = os.path.join(out, 'bad.csv')
    with open(p) as f:
        txt = f.read()
    open(bad, 'w').write(txt.replace('163.19400,71.99017', '163.19400,72.99017'))
    t = validate.run('forced_wide_published', bad, results=res, jobs=jobs, stokes='I', tier='10s')
    check('published content test fails a named position at the wrong distance', not all(x['passed'] for x in t))
    # publishing again: the file it replaces is backed up beside it (<file>.bak_<UTC>), its sha1 and the new one's recorded
    before = open(p, 'rb').read()
    rc, err = cli(base + ['--fresh'])
    r = rec(out, 'W_publish_Run_SNAPFORCED_I10s_gen5')
    m = r.get('meta') or {}
    bak = m.get('backup') or ''
    import hashlib
    check('cli: a replaced file is backed up first (<file>.bak_<UTC>, the same bytes), its sha1 and the new file\'s in the record',
          rc == 0 and bak.startswith(p + '.bak_') and os.path.exists(bak) and open(bak, 'rb').read() == before
          and m.get('backup_sha1') == hashlib.sha1(before).hexdigest() == m.get('replaced_sha1')
          and m.get('published_sha1') == hashlib.sha1(open(p, 'rb').read()).hexdigest(), (rc, m))
    rc, err = cli(base[:-2] + ['--stages', 'wide_publish', '--dry-run', '--snap-review-root', '/data07/gh/phase3/snapshot_search/review'])
    check('dry run: publishing into the served runs (a production root) is planned as W-ROOT for each of the 4 runs, exit 2',
          rc == 2 and _LAST_OUT[0].count('W-ROOT') >= 4, (rc, _LAST_OUT[0][-500:]))
    rc, err = cli(base[:-2] + ['--stages', 'wide_publish', '--dry-run', '--snap-review-root', os.path.join(TMP, 'no_review')])
    check('dry run: a run directory absent at the review root is planned as W-RUNDIR, exit 2', rc == 2 and 'W-RUNDIR' in _LAST_OUT[0],
          (rc, _LAST_OUT[0][-400:]))
    # W-NAMED: a name not found around the target
    row2 = dict(row, named="VLSSr J000000.0+000000 9.9 Jy at 31.3'")
    write_csv(res, [row2], list(RESULT_COLUMNS))
    rc, err = cli(base + ['--fresh'])
    r = rec(out, 'W_publish_Run_SNAPFORCED_I10s_gen5')
    check('cli: W-NAMED a named source not found around its target refuses', rc == 2 and 'W-NAMED' in r.get('reason', ''),
          (rc, r.get('reason')))


def test_v2_draw():
    """v2's control finder: every eligible pair by default; production's capped draw (cap, seed) only when configured"""
    from exopipe.phase3.lib.forced import wide_v3 as V3

    class FJ:
        pass

    def fj(center, srcs, ident):
        j = FJ()
        j.center, j.srcs, j.very_bright, j.bands = center, srcs, [], {'27MHz'}
        j.job = {'stokes': 'I', 'identity': ident}
        return j
    Jx = fj((10.0, 30.0), [], 'X')
    Jx.job.update(S=[10.05, 30.0], T=[10.0, 30.0], s74=1.0, bands=['27MHz'])
    pool = []
    for k in range(3):
        c0 = (40.0 + 10 * k, 30.0)
        pool.append(fj(c0, [(c0[0] + dx / 60.0 / math.cos(math.radians(30.0)), c0[1] + dy / 60.0, 1.0, 'S%d' % k)
                            for dx in range(-50, 51, 10) for dy in range(-50, 51, 10)], 'P%d' % k))
    key = lambda e: (pool.index(e[0]), round(e[1][0], 6), round(e[1][1], 6))     # noqa: E731
    every = [key(e) for e in V3.v2_controls_for(Jx, pool)]
    check('v2 controls, default: every eligible pair (3 x 121 synthetic sources)', len(every) == 363 and len(set(every)) == 363,
          len(every))
    CFG.authorize([], out=TMP)
    CFG.set('forced_wide_test.V2_CTL_DRAW', '(50, 7)')
    drawn = [key(e) for e in V3.v2_controls_for(Jx, pool)]
    want = [every[i] for i in sorted(np.random.default_rng(7).choice(363, 50, replace=False))]
    check('v2 controls, (cap, seed): production\'s draw - default_rng(seed).choice of cap indices, sorted', drawn == want)
    CFG.set('forced_wide_test.V2_CTL_DRAW', '(400, 7)')
    check('v2 controls: a cap above the eligible count draws nothing', [key(e) for e in V3.v2_controls_for(Jx, pool)] == every)
    import random
    for draw in ('None', '(50, 7)'):
        CFG.set('forced_wide_test.V2_CTL_DRAW', draw)
        base = [key(e) for e in V3.v2_controls_for(Jx, pool)]
        for seed in (1, 2):
            sh = list(pool)
            random.Random(seed).shuffle(sh)
            got = [key(e) for e in V3.v2_controls_for(Jx, sh)]
            check('v2 controls, canonical, V2_CTL_DRAW %s: a shuffled job list gives the same controls in the same order '
                  '(seed %d)' % (draw, seed), got == base and len(base) == (363 if draw == 'None' else 50), len(got))
    CFG.set('forced_wide_test.CONTROL_ORDER', 'pool')
    CFG.set('forced_wide_test.V2_CTL_DRAW', '(50, 7)')
    sh = list(reversed(pool))
    check('v2 controls, production order with its draw: a reversed job list draws other controls (what canonical removes)',
          {key(e) for e in V3.v2_controls_for(Jx, sh)} != {key(e) for e in V3.v2_controls_for(Jx, pool)})
    CFG.set('forced_wide_test.CONTROL_ORDER', 'canonical')
    for bad in ('(0, 7)', '(5,)', '(5, -1)', "'x'"):
        try:
            CFG.set('forced_wide_test.V2_CTL_DRAW', bad)       # the config's own type check refuses a non-tuple
            V3.v2_draw()
            ok = False
        except ValueError:
            ok = True
        check('V2_CTL_DRAW %s refused (config type or v2_draw)' % bad, ok)
    CFG.reset()


class _FJob:
    """a synthetic job for the control lines: a 300' cutout plane (arcmin) with cataloged sources and one band of bursts"""

    def __init__(self, ident, srcs, bursts, band='27MHz', tier='10s'):
        from scipy.spatial import cKDTree
        self.job = {'stokes': 'I', 'identity': ident}
        self.st = 'I'
        self.srcs = [(ra, dec, s74, name) for ra, dec, s74, name, _x, _y in srcs]
        self.sxy = np.array([(x, y) for *_r, x, y in srcs]) if srcs else np.zeros((0, 2))
        e, n = bursts[:, 0], bursts[:, 1]
        self.b = {(band, tier): dict(e=e, n=n, tree=cKDTree(np.c_[e, n]), frames=100)}

    def has(self, band, tier):
        d = self.b.get((band, tier))
        return d is not None and d['frames'] > 0


def _fjob(ident, k, offsets, s74=5.0, seed=0):
    """job k: sources at the offsets (arcmin; their cells and names unique per job), a taper of bursts about each and a flat
    background"""
    rng = np.random.default_rng(seed + 101 * k)
    srcs = [(100.0 + 5.0 * k + x / 60.0, 30.0 + y / 60.0, s74, 'VLSSr J%d_%d' % (k, i), x, y) for i, (x, y) in enumerate(offsets)]
    pts = [rng.uniform(-148.0, 148.0, (4000, 2))]
    for x, y in offsets:
        pts.append(rng.normal((x, y), 12.0, (600, 2)))
    return _FJob(ident, srcs, np.clip(np.concatenate(pts), -147.9, 147.9))


def test_control_order():
    """the taper test's control lines are ORDER-INDEPENDENT (R-REFACTOR-20261104-FORCED-WIDE-CONTROLS-ORDER-INDEPENDENT): a
    shuffled job list gives the same lines; an unrelated job changes nothing; a job that adds sources to the pool adds lines
    and, at the cap, displaces only the end of the order; production's 'pool' order does depend on the job order"""
    import random
    from exopipe.phase3.lib.forced import wide_v3 as V3
    offs = [(-60.0, -60.0), (60.0, -40.0), (-30.0, 70.0), (50.0, 50.0)]
    pool = [_fjob('J_%s' % c_, k, offs, seed=7) for k, c_ in enumerate('ABCDE')]
    J0 = _fjob('T0', 9, [], seed=7)
    cand = {'bands_far': ['27MHz'], 'd': 30.0, 's74': 5.0}

    def lines(p):
        zs, prov = V3.control_lines_canonical(p, J0, cand, '10s')
        return zs, prov, dict(zip(prov, zs))
    zs1, prov1, m1 = lines(pool)
    check('canonical control lines: 5 jobs x 4 sources give lines, each with its provenance', len(zs1) > 50
          and len(prov1) == len(zs1) and len({p_[0] for p_ in prov1}) == 20, len(zs1))
    for seed in (1, 2, 3):
        sh = list(pool)
        random.Random(seed).shuffle(sh)
        zs2, prov2, _m = lines(sh)
        check('a shuffled job list (seed %d) gives the same control lines, values and order' % seed, zs2 == zs1
              and prov2 == prov1)
    unrel = _fjob('J_U', 7, [(10.0, 10.0), (-80.0, 20.0)], s74=0.5, seed=7)   # its sources are outside the 0.5-2x flux range
    zs3, prov3, _m = lines(pool + [unrel])
    check('an unrelated job (no source in the flux range) leaves every control line unchanged', zs3 == zs1 and prov3 == prov1)
    rel = _fjob('J_R', 8, [(-90.0, 0.0), (0.0, -90.0), (80.0, 80.0)], seed=7)
    zs4, prov4, m4 = lines(pool + [rel])
    added = [p_ for p_ in prov4 if p_[1] == 'I:J_R']
    check('a job adding 3 sources adds their lines and changes no other line', added and set(prov1) <= set(prov4)
          and all(m4[k_] == m1[k_] for k_ in prov1) and len(prov4) == len(prov1) + len(added), (len(prov1), len(prov4)))
    full = V3.control_lines(pool, J0, cand, '10s', None)
    check('control_lines (canonical): the lines of control_lines_canonical, each with its control source\'s identity',
          [q[:4] for q in full] == zs1 and [q[4] for q in full] == [p_[0] for p_ in prov1])
    CFG.authorize([], out=TMP)
    CFG.set('forced_wide_test.TAPER_CTL_MAX', '40')
    zs5, prov5, m5 = lines(pool)
    zs6, prov6, m6 = lines(pool + [rel])
    first = next((i for i, p_ in enumerate(prov6) if p_[1] == 'I:J_R'), len(prov6))
    kept = [p_ for p_ in prov5 if p_ in m6]
    check('at the cap (40 lines): the order before the first added line is unchanged, the added lines displace only the end',
          len(zs5) == 40 == len(zs6) and prov6[:first] == prov5[:first] and kept == prov5[:len(kept)]
          and all(m6[k_] == m5[k_] for k_ in kept), (first, len(kept)))
    sh = list(pool)
    random.Random(4).shuffle(sh)
    check('at the cap, a shuffled job list still gives the same 40 lines', V3.control_lines_canonical(sh, J0, cand, '10s')[0]
          == zs5)
    CFG.reset()
    # a tie between two host jobs of one source (identical cutout geometry): the smaller job identity, in any order
    twin = _fjob('J_0', 0, offs, seed=7)            # job k = 0: J_A's sources and offsets exactly, a smaller identity
    for order in ([pool[0], twin], [twin, pool[0]]):
        prov = V3.control_lines_canonical(order, J0, cand, '10s')[1]
        hosts = {p_[1] for p_ in prov}
        check('a host tie goes to the smaller job identity, in either order (%s first)' % job_name(order[0]),
              hosts == {'I:J_0'}, hosts)
    # the rotations are the line's own stable hashes
    u = V3.stable_uniform('rot', 20261102, 'I:T0', 'VLSSr J0_0|100.00000|29.00000', 3)
    check('stable_uniform: the same value in every call, another for another target', u == V3.stable_uniform(
        'rot', 20261102, 'I:T0', 'VLSSr J0_0|100.00000|29.00000', 3) != V3.stable_uniform(
        'rot', 20261102, 'I:T1', 'VLSSr J0_0|100.00000|29.00000', 3) and 0.0 <= u < 1.0)
    # production's order depends on the job list (what the canonical order removes)
    CFG.set('forced_wide_test.CONTROL_ORDER', 'pool')
    a = V3.control_lines(pool, J0, cand, '10s', np.random.default_rng(20261102))
    b = V3.control_lines(list(reversed(pool)), J0, cand, '10s', np.random.default_rng(20261102))
    CFG.reset()
    check('production\'s pool order (CONTROL_ORDER pool) gives other lines for a reversed job list', a != b
          and len(a) == len(b))


def job_name(J):
    return J.job['identity']


class _CJob:
    """a synthetic job for the centroid and the verdict: bursts at the target in one band, nights with a shared offset each"""

    def __init__(self, ident, nights, per, off, seed, night_sd=0.08, band='27MHz', tier='10s'):
        from scipy.spatial import cKDTree
        from exopipe.phase3.lib.forced import wide_v3 as V3
        g = np.random.default_rng(seed)
        fw = V3.fw_table()[band]
        e, n, cell = [], [], []
        for k in range(nights):
            sh = g.normal(0, night_sd * fw, 2)
            for _j in range(max(1, int(g.poisson(per)))):
                e.append(off[0] * fw + sh[0] + g.normal(0, 0.05 * fw))
                n.append(off[1] * fw + sh[1] + g.normal(0, 0.05 * fw))
                cell.append('c_%02d' % k)
        e, n = np.array(e), np.array(n)
        self.b = {(band, tier): dict(e=e, n=n, cell=np.array(cell), tree=cKDTree(np.c_[e, n]), frames=100)}
        self.job = {'stokes': 'I', 'identity': ident, 'target': ident}
        self.st = 'I'

    def has(self, band, tier):
        return (band, tier) in self.b


def _cand(name, d, p, ratio, z_c=9.0, s74=4.0, n_obs=12, band='27MHz'):
    """a capable far candidate's analysis as verdict_row reads it (calibrated taper, no centering, no night test)"""
    nan = float('nan')
    return dict(name='VLSSr ' + name, s74=s74, d=d, xy=(d, 0.0), bands_far=[band], bands_near=[],
                taper=dict(n_obs=n_obs, n_pred=4.0, var=0.5, bands=[dict(band=band)]), lam=4.0, kappa=1.0, ratio=ratio,
                p_emp=p, ctl=(200, 1.0, 0.0), ctl_ratio_q=(1.0, 2.0, 3.0), night=(nan, nan, 0), z_c=z_c, reach={band: (1, 1, 1)})


def test_monte_carlo():
    """the amendment R-REFACTOR-20261104-FORCED-WIDE-CONTROLS-ORDER-INDEPENDENT-AMENDED-PRECISION-KNIFE-EDGE: the Monte Carlo
    errors match salt reruns (within a factor of 1.5), the bootstrap reaches BOOT_Z_SE per (target, tier) from its own seed,
    production's draws stay exact, and the knife-edge flag names exactly the statistics that decide a row"""
    from exopipe.phase3.lib.forced import wide_v3 as V3
    from exopipe.phase3.lib.forced.tests_wide import knife_edge_problems
    t = CFG.forced_wide_test
    # calibrate's errors: the binomial p and the ratio estimator's spread, exactly
    ctl = [(0.0, n, 2.0, 0.1, 'S%d' % (i // 4)) for i, n in enumerate((0, 1, 2, 2, 3, 1, 2, 4, 2, 2, 1, 3, 2, 2, 0, 2))]
    tt = {'n_obs': 9, 'n_pred': 2.0, 'var': 0.1}
    r = V3.calibrate(tt, ctl)
    no = np.array([q[1] for q in ctl], float)
    kap = (no.sum() + 0.5) / (2.0 * 16 + 0.5)
    se_k = math.sqrt(np.sum((no - kap * 2.0) ** 2) * 16 / 15) / (2.0 * 16 + 0.5)
    lam = kap * 2.0
    check('calibrate: p_se_line = sqrt(p (1 - p) / n), ratio_se_line = ratio lambda / (lambda + 0.5) SE(kappa) / kappa',
          abs(r['p_se_line'] - math.sqrt(r['p_emp'] * (1 - r['p_emp']) / 16)) < 1e-12
          and abs(r['ratio_se_line'] - r['ratio'] * lam / (lam + 0.5) * se_k / kap) < 1e-12, r)
    r3 = V3.calibrate({'n_obs': 3, 'n_pred': 2.0, 'var': 0.1}, ctl)      # lines on both sides of the target within sources
    rc = (no + 0.5) / (kap * 2.0 + 0.5)
    ind = (rc >= r3['ratio']).astype(float)
    d = no - kap * 2.0
    vk = vp = 0.0
    for j in range(4):                 # 4 control sources of 4 lines each
        a, b = ind[4 * j:4 * j + 4], d[4 * j:4 * j + 4]
        vp += float(np.sum((a - a.mean()) ** 2)) * 4 / 3
        vk += float(np.sum((b - b.mean()) ** 2)) * 4 / 3
    check('calibrate: the within-source errors (SE_KIND source) - the binomial and the spread within each control source',
          vp > 0 and abs(r3['p_se_source'] - math.sqrt(vp) / 16) < 1e-12
          and abs(r3['ratio_se_source'] - r3['ratio'] * lam / (lam + 0.5) * math.sqrt(vk) / (2.0 * 16 + 0.5) / kap) < 1e-12
          and V3.SE_KIND == 'source' and 'p_se_source' not in V3.calibrate(tt, [q[:4] for q in ctl]), r3)
    # a degenerate within-source estimate (no line at or above the target: every source all one side) is no zero error
    flat = [(0.0, 1, 2.0, 0.1, 'S%d' % (i // 4)) for i in range(16)]
    r0 = V3.calibrate({'n_obs': 30, 'n_pred': 2.0, 'var': 0.1}, flat)
    check('calibrate: a zero within-source estimate falls back to the binomial over all lines', r0['p_se_source']
          == r0['p_se_line'] > 0 and r0['p_emp'] == 1 / 17, r0)
    sel = [q for q in flat if q[4] in ('S0', 'S1')]
    check('the counts\' error: a zero within-source estimate (every source wholly in or out) falls back to the binomial',
          V3._count_se([q[4] for q in flat], sel, 16) == math.sqrt(8 * (1 - 8 / 16)))
    CFG.authorize([], out=TMP)
    # the centroid's bootstrap: z's analytic Monte Carlo error against 200 salt reruns (BOOTSTRAP resamples each)
    CFG.set('forced_wide_test.BOOT_Z_SE', 'None')
    J = _CJob('CJ1', 12, 6, (0.06, 0.02), 1)
    zz = []
    for salt in range(200):
        CFG.set('forced_wide_test.SEED', str(7000 + salt))
        c_ = V3.centroid(J, '10s', ['27MHz'], None)
        zz.append((c_['z'], c_['z_se'], c_['boot_n']))
    a = np.array(zz)
    q = a[:, 0].std(ddof=1) / a[:, 1].mean()
    check('the centroid z\'s Monte Carlo error matches 200 salt reruns within a factor of 1.5 (sd / error %.2f)' % q,
          1 / 1.5 <= q <= 1.5 and set(a[:, 2]) == {t.BOOTSTRAP}, q)
    CFG.reset()
    # the taper p and ratio: their errors (SE_KIND source) against 60 salt reruns (each salt rotates every control line). A
    # salt moves a line within its 45-degree sector only; on the sky the burst field changes within a sector (30 salts on real
    # rows: salt sd / error median 0.89 for p, 1.00 for the ratio - se_validate.py), and so does this smooth synthetic field
    # at 2 FWHM, where a sector spans 1.6 FWHM; at 1 FWHM (a sector of 0.8 FWHM, the disc 0.7 FWHM across) it changes less
    # and the error is conservative, never an underestimate
    offs = [(-60.0, -60.0), (60.0, -40.0), (-30.0, 70.0), (50.0, 50.0)]
    pool = [_fjob('J_%s' % c_, k, offs, seed=7) for k, c_ in enumerate('ABCDE')]
    J0 = _fjob('T0', 9, [], seed=7)
    for dist, within in ((60.0, True), (30.0, False)):
        cd = {'bands_far': ['27MHz'], 'd': dist, 's74': 5.0}
        tt = V3.taper_test(J0, (dist, 0.0), (0.0, 0.0), ['27MHz'], '10s')
        CFG.authorize([], out=TMP)
        got = []
        for salt in range(60):
            CFG.set('forced_wide_test.SEED', str(9000 + salt))
            zs = V3.control_lines(pool, J0, cd, '10s', None)
            got.append(V3.calibrate(tt, V3.matched(zs, tt['n_pred'])))
        CFG.reset()
        for k in ('p_emp', 'ratio'):
            e = 'p_se_source' if k == 'p_emp' else 'ratio_se_source'
            sd = float(np.std([g[k] for g in got], ddof=1))
            se = float(np.mean([g[e] for g in got]))
            if within:
                check('the taper %s\'s Monte Carlo error matches 60 salt reruns within a factor of 1.5 at 2 FWHM (sd %.4f, '
                      'error %.4f)' % (k, sd, se), se > 0 and 1 / 1.5 <= sd / se <= 1.5, (sd, se))
            else:
                check('the taper %s\'s Monte Carlo error at 1 FWHM is conservative, never an underestimate (sd %.4f, error '
                      '%.4f)' % (k, sd, se), se > 0 and sd / se <= 1.5, (sd, se))
    # BOOT_Z_SE: more resamples until the error of z is at most 0.02; BOOT_MAX bounds them; the seed is the target's own
    Jh = _CJob('CJ2', 40, 4, (0.3, 0.1), 2, night_sd=0.1)
    c1 = V3.centroid(Jh, '10s', ['27MHz'], None)
    check('BOOT_Z_SE 0.02: a z of %.1f gets %d resamples, its error %.4f <= 0.02' % (c1['z'], c1['boot_n'], c1['z_se']),
          c1['boot_n'] > t.BOOTSTRAP and c1['z_se'] <= t.BOOT_Z_SE and c1['boot_n'] % t.BOOTSTRAP == 0, c1)
    c2 = V3.centroid(Jh, '10s', ['27MHz'], np.random.default_rng(1))
    check('BOOT_DRAW target: the same (target, tier) draws the same resamples whatever generator is passed', c2 == c1)
    Jh2 = _CJob('CJ2b', 40, 4, (0.3, 0.1), 2, night_sd=0.1)          # the same bursts, another identity
    check('BOOT_DRAW target: another target identity draws other resamples', V3.centroid(Jh2, '10s', ['27MHz'], None)['err']
          != c1['err'])
    CFG.authorize([], out=TMP)
    CFG.set('forced_wide_test.BOOT_MAX', '2000')
    c3 = V3.centroid(Jh, '10s', ['27MHz'], None)
    check('BOOT_MAX bounds the resamples (2000; the error reached is reported)', c1['boot_n'] > 2000 and c3['boot_n'] == 2000
          and c3['z_se'] > t.BOOT_Z_SE, (c1['boot_n'], c3))
    CFG.reset()
    # BOOT_DRAW job, BOOT_Z_SE None = production's bootstrap exactly (forced_wide_v3.py:477-481)
    CFG.authorize([], out=TMP)
    CFG.set('forced_wide_test.BOOT_DRAW', "'job'")
    CFG.set('forced_wide_test.BOOT_Z_SE', 'None')
    c4 = V3.centroid(J, '10s', ['27MHz'], np.random.default_rng(20261102))
    CFG.reset()
    D = J.b[('27MHz', '10s')]
    fw = V3.fw_table()['27MHz']
    m = np.hypot(D['e'], D['n']) / fw < t.DISC
    xs, ys, cells = D['e'][m] / fw, D['n'][m] / fw, D['cell'][m]
    rng = np.random.default_rng(20261102)
    uc = sorted(set(cells))
    nn = len(uc)
    idx = [np.where(cells == c_)[0] for c_ in uc]
    bs = np.array([(xs[ii].mean(), ys[ii].mean()) for ii in (np.concatenate([idx[k] for k in rng.integers(0, nn, nn)])
                                                             for _ in range(1000))])
    stat = math.hypot(bs[:, 0].std(), bs[:, 1].std()) / math.sqrt(2)
    err = math.hypot(stat, max(t.CEN_FLOOR, t.CEN_NIGHT / math.sqrt(nn)))
    check('BOOT_DRAW job, BOOT_Z_SE None: production\'s 1000 resamples from the job\'s generator, bit for bit',
          c4['err'] == err and c4['z'] == math.hypot(xs.mean(), ys.mean()) / err and c4['boot_n'] == 1000)
    # THE KNIFE-EDGE FLAG: decisive statistics only
    Jv = _CJob('KV', 12, 6, (0.0, 0.0), 3, night_sd=0.02)
    ev = [('27MHz', '20250101_000000', 5.0)] * 5
    cen = dict(n=60, nights=12, x=0.1, y=0.0, off=0.1, err=0.0334, z=2.99, pa=90.0, ifw=1.0 / fw, z_se=0.02, boot_n=4000,
               rel=0.0067)

    def ke(res, cen_):
        row = V3.verdict_row(Jv, '10s', ev, [], res, cen_)
        return row['verdict'], V3.knife_edges(Jv, '10s', ev, [], res, cen_, row)
    v, ks = ke([], cen)
    check('knife-edge: no capable candidate, centroid z 2.99 +- 0.02 decides UNDECIDED / NOT CENTERED -> flagged',
          v == 'UNDECIDED' and [k[0] for k in ks] == ['centroid z'], (v, ks))
    v, ks = ke([], dict(cen, z=2.9))
    check('knife-edge: centroid z 2.90 +- 0.02 is 5 errors from 3 -> not flagged', v == 'UNDECIDED' and not ks, ks)
    cand = _cand('J010000.0+100000', d=20.0, p=0.051, ratio=1.5)
    for r_ in [cand]:
        r_.update(p_se=0.005, ratio_se=0.05, z_c_se=0.06)
    v, ks = ke([cand], dict(cen, z=2.5))
    check('knife-edge: the taper p 0.051 +- 0.005 decides CONSISTENT / UNDECIDED -> flagged, its ratio 1.5 +- 0.05 not',
          v == 'CONSISTENT WITH A BACKGROUND SOURCE' and [k[0] for k in ks] == ['taper p of J010000.0+100000'], (v, ks))
    v, ks = ke([cand], dict(cen, z=3.5))
    check('knife-edge: the same p near 0.05 does not decide NOT CENTERED (z 3.5) -> not flagged',
          v == 'NOT CENTERED ON THE TARGET (source not identified)' and not ks, (v, ks))
    cand2 = dict(cand, ratio=1.97, p_emp=0.3, p_se=0.03, ratio_se=0.02)
    v, ks = ke([cand2], dict(cen, z=2.5))
    check('knife-edge: the ratio 1.97 +- 0.02 decides CONSISTENT -> flagged with its value', [k[0] for k in ks] ==
          ['taper ratio of J010000.0+100000'] and abs(ks[0][1] - 1.97) < 1e-12, ks)
    cand3 = dict(cand2, ratio=1.2, z_c=1.98, z_c_se=0.03, d=5.0)
    v, ks = ke([cand3], dict(cen, z=2.5))
    check('knife-edge: z_c 1.98 +- 0.03 (the bursts center on it) decides the named evidence -> flagged',
          v == 'BACKGROUND SOURCE' and [k[0] for k in ks] == ['centroid z to J010000.0+100000'], (v, ks))
    # the matched-line count: 15 in the 3x window; at 14 matched() takes the 10x window, whose calibration says no
    lines = [(0.0, 6, 2.0, 0.1, 'S%d' % i) for i in range(15)] + [(0.0, 1, 8.0, 0.1, 'T%d' % i) for i in range(25)]
    ttc = {'n_obs': 9, 'n_pred': 2.0, 'var': 0.1, 'z': 0.0, 'bands': [dict(band='27MHz')]}
    cal = V3.calibrate(ttc, V3.matched(lines, 2.0))
    cand4 = dict(_cand('J020000.0+200000', d=20.0, p=cal['p_emp'], ratio=cal['ratio']), taper=ttc, **cal)
    cand4['_windows'] = ([q[4] for q in lines], len(lines), V3.match_windows(lines, 2.0))
    cand4.update(z_c_se=0.06)
    V3.mc_errors(cand4)
    v, ks = ke([cand4], dict(cen, z=2.5))
    labs = [k[0] for k in ks]
    check('knife-edge: 15 matched lines (3x) decide the calibration -> flagged; the counts\' errors binomial',
          v == 'CONSISTENT WITH A BACKGROUND SOURCE' and 'matched control lines (within 3x) of J020000.0+200000' in labs
          and abs(cand4['_counts'][0][2] - math.sqrt(15 * (1 - 15 / 40))) < 1e-12, (v, ks, cand4['_counts']))
    # the column's text and its content test
    txt = V3.knife_text([('centroid z', 2.97, 3.0, 0.0213, '%.2f'), ('taper p of J1', 0.0496, 0.05, 0.0069, '%.3g')])
    check('knife_text: the statistic, its value, the threshold and the Monte Carlo error',
          txt == 'knife-edge: centroid z 2.97 vs 3, Monte Carlo error 0.021; taper p of J1 0.0496 vs 0.05, Monte Carlo error '
                 '0.0069', txt)
    row = dict(bootstrap_n='4000', centroid_z_se='0.0213', centroid_z='3.0', knife_edge=txt)
    check('knife-edge content test: a consistent entry passes', not knife_edge_problems(row, 2.0),
          knife_edge_problems(row, 2.0))
    bad = dict(row, knife_edge=txt.replace('0.0496 vs', '0.0796 vs'))
    check('knife-edge content test: an entry outside its window fails', knife_edge_problems(bad, 2.0))
    bad = dict(row, knife_edge=txt.replace('vs 3,', 'vs 4,'))
    check('knife-edge content test: an unknown threshold fails', knife_edge_problems(bad, 2.0))
    two = V3.knife_text([('taper ratio of J2', 2.6, 2.0, 0.5, '%.2f'), ('taper ratio of J2', 2.6, 3.0, 0.5, '%.2f')])
    check('knife_text: a statistic deciding at both thresholds is named once, its content test passes',
          two == 'knife-edge: taper ratio of J2 2.60 vs 2 and 3, Monte Carlo error 0.5'
          and not knife_edge_problems(dict(row, knife_edge=two), 2.0), two)


def test_draws():
    """THE KNIFE-EDGE ROWS' DRAWS (KNIFE_EDGE_DRAWS, KNIFE_EDGE_DECIDE): a flagged row is DECIDED only when its own K draws decide
    it - its verdict class and each named source by a Wilson 95% interval wholly above 1/2 - else UNDECIDED (knife-edge: the
    draws split); a source named in an undecided fraction is listed apart, never named; unflagged rows keep their single draw;
    the salts are fixed per target"""
    from exopipe.phase3.lib.forced import wide_v3 as V3
    from exopipe.phase3.lib.forced.tests_wide import draws_problems, verdict_problems, RESULT_COLUMNS

    class J_:
        def __init__(self, ident):
            self.job = {'stokes': 'I', 'identity': ident}
    a = J_('A')
    salts = [V3.draw_salt(a, k) for k in range(20)]
    check('draw salts: a stable hash of (SEED, the target, k) - fixed, 20 distinct, another set for another target',
          salts == [V3.draw_salt(J_('A'), k) for k in range(20)] and len(set(salts)) == 20
          and not set(salts) & {V3.draw_salt(J_('B'), k) for k in range(20)} and salts[0] == V3.stable_int('draw', 20261102, 'I:A', 0))
    w = {n: V3.wilson(n, 20, 0.95) for n in range(21)}
    check('Wilson 95%, K = 20: decided at 15 of 20 or more (lower bound above 1/2), decidedly not at 5 or fewer, undecided 6-14',
          [n for n in range(21) if w[n][0] > 0.5] == list(range(15, 21)) and [n for n in range(21) if w[n][1] < 0.5] == list(range(6))
          and abs(w[14][0] - 0.481) < 0.001 and abs(w[15][0] - 0.531) < 0.001, {n: tuple(round(x, 3) for x in w[n]) for n in (5, 6, 14, 15)})
    X, Y, Z = "VLSSr J1 3.0 Jy at 10.0'", "VLSSr J2 2.0 Jy at 20.0'", "VLSSr J3 5.0 Jy at 30.0'"
    BG, CO, NC = 'BACKGROUND SOURCE', 'CONSISTENT WITH A BACKGROUND SOURCE', 'NOT CENTERED ON THE TARGET (source not identified)'

    def row(ident, tier, v, named='', ke='', z='2.5'):
        return dict(stokes='I', identity=ident, tier=tier, verdict=v, named=named, knife_edge=ke, centroid_z=z, named_knife_edge='',
                    mc_draws='', mc_n='', mc_outcomes='', mc_row='')

    def crow(ident, tier, tag):
        return dict(stokes='I', identity=ident, tier=tier, candidate='VLSSr J1', tag=tag)
    KE = 'knife-edge: taper ratio of VLSSr J2 1.97 vs 2, Monte Carlo error 0.02'
    jobs = ['A', 'B', 'C', 'E', 'F', 'G']
    outs = [([row('A', '10s', CO, X, KE), row('A', '1min', BG, X)], [crow('A', '10s', 'default'), crow('A', '1min', 'default')]),
            ([row('B', '10s', BG, X, KE, '3.1')], [crow('B', '10s', 'default')]),
            ([row('C', '1min', NC, '', KE, '3.0')], [crow('C', '1min', 'default')]),
            ([row('E', '10s', CO, X + '; ' + Y, KE)], [crow('E', '10s', 'default')]),
            ([row('F', '10s', CO, X, KE)], [crow('F', '10s', 'default')]),
            ([row('G', '10s', CO, X + '; ' + Y, KE)], [crow('G', '10s', 'default')]),
            ([row('D', '10s', CO, Z)], [crow('D', '10s', 'default')])]
    plan = {0: [(CO, X + '; ' + Y)] * 14 + [(CO, X)] * 6,          # class decided; X named 20, Y 14: knife-edge
            1: [(BG, X)] * 10 + [(NC, '')] * 10,                     # the class splits 10 / 10
            2: [(NC, '')] * 19 + [(CO, Z)],                          # decided 19 of 20; Z (1 of 20) decidedly not named
            3: [(CO, X)] * 16 + [(CO, X + '; ' + Y)] * 4,            # decided, the default's named set {X, Y} is not: a draw's row
            4: [(CO, X)] * 10 + [(CO, Y)] * 10,                      # a naming class decided, no source decidedly named: split
            5: [(CO, X + '; ' + Y)] * 19 + [(CO, X)]}                # KELT-24-like: both named (19 of 20)
    tasks = V3.draw_tasks([None] * 7, outs, 20)
    check('draw_tasks: K draws of every job with a knife-edge row, its flagged tiers only; unflagged jobs none',
          len(tasks) == 120 and {t[0] for t in tasks} == set(range(6)) and {t[2] for t in tasks if t[0] == 0} == {('10s',)})

    def drawn(i, k, tier):
        v, nm = plan[i][(k + 3) % 20]                      # the draws' order is not the outcomes'
        return dict(row(jobs[i], tier, v, nm, '', '2.6'), centroid_z='2.%d' % (k % 10)), [crow(jobs[i], tier, 'draw %d' % k)]
    by = {(i, tiers[0]): {} for i, _k, tiers in tasks}
    for i, k, tiers in tasks:
        by[(i, tiers[0])][k] = drawn(i, k, tiers[0])
    CFG.authorize([], out=TMP)
    new = V3.apply_draws(outs, by)
    CFG.reset()
    r0, r1, r2, r3, r4, r5 = (new[i][0][0] for i in range(6))
    check('decided: CONSISTENT in 20 of 20, X named in 20 (named), Y in 14 (knife-edge, never named); the default draw\'s row',
          r0['verdict'] == CO and r0['named'] == X and r0['mc_row'] == 'default' and r0['knife_edge'] == KE
          and r0['named_knife_edge'] == 'knife-edge: VLSSr J2 named in 14 of 20 draws'
          and r0['mc_draws'] == '%s in 20 of 20 Monte Carlo draws; named: VLSSr J1 in 20 of 20' % CO, r0)
    check('split 10 / 10: UNDECIDED (knife-edge: the method\'s random draws split ...), no named source, X knife-edge named',
          r1['verdict'] == "UNDECIDED (knife-edge: the method's random draws split - %s in 10 of 20, %s in 10 of 20)" % (BG, NC)
          and r1['named'] == '' and r1['mc_row'] == 'split' and r1['mc_draws'] == '' and r1['centroid_z'] == '3.1'
          and r1['named_knife_edge'] == 'knife-edge: VLSSr J1 named in 10 of 20 draws' and new[1][1][0]['tag'] == 'default', r1)
    check('decided 19 of 20 (not centered), the one source of the 20th draw decidedly not named, nothing knife-edge',
          r2['verdict'] == NC and r2['mc_row'] == 'default' and r2['named_knife_edge'] == ''
          and r2['mc_draws'] == '%s in 19 of 20 Monte Carlo draws' % NC, r2)
    k3 = next(k for k in range(20) if plan[3][(k + 3) % 20] == (CO, X))
    check('decided, the default draw\'s named set {X, Y} is not the decided {X}: the first draw\'s row that has it, its candidates',
          r3['named'] == X and r3['mc_row'] == 'draw %d' % k3 and r3['centroid_z'] == '2.%d' % (k3 % 10) and r3['knife_edge'] == KE
          and r3['named_knife_edge'] == '' and new[3][1][0]['tag'] == 'draw %d' % k3, r3)
    check('THE SETTLED CASE (a naming class decided, no source decidedly named): the class with its sources\' split, no named '
          'source, both knife-edge named',
          r4['verdict'] == '%s (source split: VLSSr J1 in 10 of 20, VLSSr J2 in 10 of 20 draws)' % CO and r4['named'] == ''
          and r4['mc_row'] == 'default' and r4['mc_draws'] == '%s in 20 of 20 Monte Carlo draws' % CO
          and r4['named_knife_edge'] == ('knife-edge: VLSSr J1 named in 10 of 20 draws; knife-edge: VLSSr J2 '
                                         'named in 10 of 20 draws') and V3.verdict_class(r4['verdict']) == CO, r4)
    check('KELT-24-like: 19 of 20 name both sources - both decidedly named, the default draw\'s row',
          r5['named'] == X + '; ' + Y and r5['mc_row'] == 'default' and r5['named_knife_edge'] == ''
          and r5['mc_draws'] == '%s in 20 of 20 Monte Carlo draws; named: VLSSr J1 in 20 of 20, VLSSr J2 in 19 of 20' % CO, r5)
    check('an unflagged row keeps its single draw', new[0][0][1] == outs[0][0][1] and new[6][0][0] == outs[6][0][0])
    rows_ = [r0, r1, r2, r3, r4, r5, new[0][0][1], new[6][0][0]]
    check('mc_n: every flagged row holds its number of draws (20 here), an unflagged row none',
          all(r['mc_n'] == 20 for r in (r0, r1, r2, r3, r4, r5)) and new[6][0][0]['mc_n'] == '')
    CFG.authorize([], out=TMP)
    check('draws content test: every row passes (the decision re-derived from its mc_outcomes; batches of 20, at most 20)',
          not any(draws_problems(r, 20, 20) for r in rows_), [draws_problems(r, 20, 20) for r in rows_])
    check('draws content test: a knife-edge source in the named set, a split published as decided, a wrong fraction fail',
          draws_problems(dict(r0, named=X + '; ' + Y), 20, 20) and draws_problems(dict(r1, verdict=BG, named=X, mc_row='default'), 20, 20)
          and draws_problems(dict(r5, mc_draws=r5['mc_draws'].replace('19 of', '18 of')), 20, 20)
          and draws_problems(dict(r0, named_knife_edge=''), 20, 20) and draws_problems(dict(new[6][0][0], mc_row='default'), 20, 20)
          and draws_problems(dict(r4, verdict=CO), 20, 20))
    check('draws content test: drawing stopped at 20 of 100 with a source undecided fails (sequential draws)',
          draws_problems(r0, 20, 100) and not draws_problems(r2, 20, 100), draws_problems(r0, 20, 100))
    CFG.reset()
    # SEQUENTIAL DRAWS: 20, and while the class or a source stays undecided 20 more, up to 100 (draw k = the same salt in any batch)
    UN = 'UNDECIDED'
    seq = {0: lambda k: (CO, X + '; ' + Y) if k == 0 else (CO, X),                       # settled at 20 (Y 1 of 20: decidedly not)
           1: lambda k: (NC, '') if k % 20 < (14 if k < 20 else 17) else (UN, ''),        # 14 of 20 split, 31 of 40 decided
           2: lambda k: (BG, X) if k % 2 else (NC, ''),                                   # 50 / 50: split at 100
           3: lambda k: (CO, X + '; ' + Y) if k % 2 else (CO, X),                         # Y at 50%: knife-edge at 100
           4: lambda k: (CO, X) if k % 2 else (CO, Y)}                                    # the settled case at 100
    outs2 = [([row(jobs[i], '10s', seq[i](0)[0], seq[i](0)[1], KE, '2.5')], [crow(jobs[i], '10s', 'default')]) for i in range(5)]
    seen = []

    def fake_draw(task):
        i, k, tiers = task
        seen.append((i, k))
        v, nm = seq[i](k)
        return [dict(row(jobs[i], tiers[0], v, nm, '', '2.6'), centroid_z='2.%d' % (k % 10))], [crow(jobs[i], tiers[0], 'draw %d' % k)]
    real = V3._draw
    V3._draw = fake_draw
    logs = []
    try:
        by2 = V3.sequential_draws([None] * 5, outs2, lambda f, t: [f(x) for x in t], log=logs.append)
    finally:
        V3._draw = real
    new2 = V3.apply_draws(outs2, by2)
    q = [new2[i][0][0] for i in range(5)]
    check('sequential draws: settled at 20 / 40, still undecided at 100 -> 20, 40, 100, 100, 100 draws, k counting on (0 .. 99)',
          [r['mc_n'] for r in q] == [20, 40, 100, 100, 100] and sorted(k for i, k in seen if i == 2) == list(range(100))
          and sorted(k for i, k in seen if i == 1) == list(range(40)), ([r['mc_n'] for r in q], logs))
    check('sequential draws: 31 of 40 decides NOT CENTERED; 50 / 50 splits at 100; Y at 50 of 100 stays knife-edge; the settled case',
          q[1]['verdict'] == NC and q[1]['mc_draws'] == '%s in 31 of 40 Monte Carlo draws' % NC
          and q[2]['verdict'].startswith(V3.SPLIT_PREFIX) and 'in 50 of 100' in q[2]['verdict']
          and q[3]['named'] == X and q[3]['named_knife_edge'] == 'knife-edge: VLSSr J2 named in 50 of 100 draws'
          and q[4]['verdict'] == '%s (source split: VLSSr J1 in 50 of 100, VLSSr J2 in 50 of 100 draws)' % CO, [r['verdict'] for r in q])
    CFG.authorize([], out=TMP)
    check('sequential draws: every row passes the content test (batches of 20 up to 100)', not any(draws_problems(r, 20, 100)
                                                                                                   for r in q),
          [draws_problems(r, 20, 100) for r in q])
    CFG.reset()
    base = dict.fromkeys(RESULT_COLUMNS, '')
    base.update(stokes='I', identity='X', target='X', tier='10s', n_events='9', event_jy_median='5', bursts_at_target='5',
                bursts_by_band='27MHz:5', n_candidates='4', n_capable='2', centroid_fwhm='0.350', centroid_err_fwhm='0.100',
                centroid_z='3.5', centroid_nights='5')
    sv = r1['verdict']
    check('verdict rule: a split row is UNDECIDED (knife-edge ...) whatever its z; at z 3.5 a plain UNDECIDED is refused',
          not verdict_problems(dict(base, verdict=sv, mc_row='split')) and verdict_problems(dict(base, verdict='UNDECIDED'))
          and verdict_problems(dict(base, verdict=sv)) and verdict_problems(dict(base, verdict=BG, named=X, mc_row='split')),
          (verdict_problems(dict(base, verdict=sv, mc_row='split')), verdict_problems(dict(base, verdict=sv))))
    ss = '%s (source split: VLSSr J1 in 10 of 20, VLSSr J2 in 10 of 20 draws)' % BG
    check('verdict rule: a source split reads as its class (BACKGROUND at z 3.5), with no named source, only on a row with draws',
          not verdict_problems(dict(base, verdict=ss, mc_row='default')) and verdict_problems(dict(base, verdict=ss))
          and verdict_problems(dict(base, verdict=ss, named=X, mc_row='default'))
          and verdict_problems(dict(base, verdict=ss.replace(BG, CO), mc_row='default')),
          verdict_problems(dict(base, verdict=ss, mc_row='default')))


def _probe_threads(_x):
    """a pool worker after 1.5 s of matrix products: (each loaded OpenBLAS pool's num_threads, the main thread's CPU ticks, the
    other threads' CPU ticks in all, OMP_NUM_THREADS)"""
    import scipy.spatial                                  # noqa: F401 - scipy's own OpenBLAS, as the analysis loads it
    from threadpoolctl import threadpool_info
    a = np.random.default_rng(0).random((900, 900))
    t0 = time.time()
    while time.time() - t0 < 1.5:
        _b = a @ a
    time.sleep(0.3)
    ticks = {}
    for t in os.listdir('/proc/self/task'):
        f = open('/proc/self/task/%s/stat' % t).read().rsplit(')', 1)[1].split()
        ticks[int(t)] = int(f[11]) + int(f[12])
    main = ticks.pop(os.getpid())
    nt = sorted({d['num_threads'] for d in threadpool_info() if d.get('internal_api') == 'openblas'})
    return nt, main, sum(ticks.values()), os.environ.get('OMP_NUM_THREADS')


def test_worker_threads():
    """wide_search / wide_test pool workers run --threads BLAS / OpenMP threads (else 1). The CLI's environment variables come
    after numpy's OpenBLAS is loaded, so the initializer caps the loaded pools (threadpoolctl; the republish verification
    2026-11-04 saw 26 threads in each of 6 workers, all running). OpenBLAS re-creates its pool in a forked child at its
    load-time width, so the threads still exist - capped, they stay idle: the check is the CPU they use."""
    from multiprocessing import get_context
    from exopipe.phase3.lib.forced import wide_search as WS
    ncpu = len(os.sched_getaffinity(0))
    with get_context('fork').Pool(1) as p:
        plain = p.map(_probe_threads, [0])[0]
    with get_context('fork').Pool(1, initializer=WS.pool_worker_init, initargs=(1,)) as p:
        one = p.map(_probe_threads, [0])[0]
    with get_context('fork').Pool(1, initializer=WS.pool_worker_init, initargs=(3,)) as p:
        three = p.map(_probe_threads, [0])[0]
    check('pool workers without the cap run every allowed core (%d): other threads %d ticks against the main thread\'s %d'
          % (ncpu, plain[2], plain[1]), ncpu < 3 or plain[2] > 2 * plain[1], plain)
    check('pool_worker_init(1): every OpenBLAS pool at 1 thread, the other threads idle (%d ticks against the main thread\'s %d), '
          'OMP_NUM_THREADS 1' % (one[2], one[1]), one[0] == [1] and one[2] < 0.75 * one[1] and one[3] == '1', one)
    check('pool_worker_init(3): every OpenBLAS pool at 3 threads (--threads 3)', three[0] == [3] and three[3] == '3', three)


def test_pool_provenance():
    """wide_test pools burst files of ONE search: the current jobs.json and the selected cutout generation (review F8)"""
    import types
    from exopipe.phase3.core import record as R
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.forced import wide_stages as WS
    out = os.path.join(TMP, 'pool')
    os.makedirs(os.path.join(out, 'bursts'), exist_ok=True)
    jobs = os.path.join(out, 'jobs.json')
    job = {'stokes': 'I', 'identity': 'X', 'target': 'X', 'sample': 'HWJ', 'T': [10.0, 20.0], 'bands': ['27MHz'], 'bright': []}
    json.dump({'base': [], 'ext': [job], 'runs': []}, open(jobs, 'w'))
    npz = os.path.join(out, 'bursts', 'I_X_27MHz.npz')
    open(npz, 'wb').write(b'x')
    gen = '/lustre/pipeline/exopipe/phase3/photometry_gen5'
    c = types.SimpleNamespace(args=types.SimpleNamespace(wide_bursts=None), W=out, cutout_root=gen)

    def search_rec(jobs_fp, root):
        fake_record(out, 'W_search_I_X', [(npz, 'forced_wide_bursts')], meta={'cutout_root': root},
                    inputs={os.path.abspath(jobs): jobs_fp})

    def pool():
        try:
            return WS.pool_files(c, [job], jobs), ''
        except Refusal as e:
            return None, str(e)
    search_rec(R.fingerprint(jobs), gen)
    got, why = pool()
    check('pool: the current jobs.json and the selected generation -> pooled', got and got[0] == {('I', 'X', '27MHz'): npz}, why)
    search_rec(dict(R.fingerprint(jobs), sha1='0' * 40), gen)
    got, why = pool()
    check('pool: a search of another jobs.json refused (W-POOL)', got is None and 'W-POOL' in why and 'jobs.json' in why, why)
    search_rec(R.fingerprint(jobs), '/lustre/pipeline/exopipe/phase3/photometry_gen6')
    got, why = pool()
    check('pool: a search of another cutout generation refused (W-POOL)', got is None and 'W-POOL' in why
          and 'photometry_gen6' in why, why)


# ---------------------------------------------------------------------------------------------------------------------
# 4. the burst search's functions and its product
# ---------------------------------------------------------------------------------------------------------------------

def _gauss(ny, nx, x0, y0, amp, smaj=2.2, smin=1.6, pa=0.6):
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    xr = (xx - x0) * math.cos(pa) + (yy - y0) * math.sin(pa)
    yr = -(xx - x0) * math.sin(pa) + (yy - y0) * math.cos(pa)
    return amp * np.exp(-0.5 * (xr ** 2 / smaj ** 2 + yr ** 2 / smin ** 2))


def test_search_functions():
    from exopipe.phase3.lib.forced import wide_search as WS
    c = CFG.forced_wide_search
    rng = np.random.default_rng(7)
    ny = nx = 80
    plane = rng.normal(0, 1.0, (ny, nx)) + _gauss(ny, nx, 30.3, 40.8, 25.0) + _gauss(ny, nx, 55.6, 20.2, -18.0) \
        + _gauss(ny, nx, 1.0, 60.0, 30.0)
    pk = WS.find_peaks(plane[None].astype(np.float32), c)
    pos = [(round(p[6], 1), round(p[7], 1), p[5] > 0) for p in pk]
    near = lambda x, y, s: any(abs(a - x) < 0.25 and abs(b - y) < 0.25 and sg == s for a, b, sg in pos)  # noqa: E731
    check('find_peaks: the positive and the negative burst found at their sub-pixel positions', near(30.3, 40.8, True)
          and near(55.6, 20.2, False), pos)
    check('find_peaks: nothing within EDGE_PX of the border', all(c.EDGE_PX <= p[1] < nx - c.EDGE_PX for p in pk), pos)
    check('find_peaks: |S/N| >= SNR_MIN, S/N = value / local rms', all(abs(p[5]) >= c.SNR_MIN and p[5] == p[3] / p[4] for p in pk))
    x, y = 30, 41
    st = plane.astype(np.float32)[y - 20:y + 21, x - 20:x + 21].astype(float).copy()
    st[18:23, 18:23] = np.nan
    v = st[np.isfinite(st)]
    want = 1.4826 * float(np.median(np.abs(v - np.median(v))))
    check('stamp_rms = 1.4826 MAD of the 41 x 41 stamp without its 5 x 5 core',
          abs(WS.stamp_rms(plane.astype(np.float32), x, y, c) - want) < 1e-5)
    err = []
    for i in range(200):
        x0, y0 = rng.uniform(-0.5, 0.5, 2)
        b = _gauss(3, 3, 1 + x0, 1 + y0, 1.0, 5.3 / 2.355, 3.8 / 2.355, math.radians(58.8))
        dx, dy = WS.interp2d(b, c)
        err.append(max(abs(dx - x0), abs(dy - y0)))
    check('interp2d exact on a rotated elliptical Gaussian', max(err) < 1e-6, max(err))
    # gen5's 1-minute blocks: the file-name minute, >= 3 admitted frames, averaged
    cube = np.arange(6 * 4, dtype=np.float32).reshape(6, 2, 2)
    ts = ['20250101_010001', '20250101_010011', '20250101_010021', '20250101_010101', '20250101_010111', '20250101_010121']
    adm = np.array([1, 1, 1, 1, 1, 0], bool)
    WS._E['SS'] = None
    keys, blk = WS.blocks_1min(cube, ts, adm, ts, c)
    check('blocks_1min: one block of the minute holding >= 3 admitted frames, its mean, key + 00',
          keys == ['20250101_010000'] and np.array_equal(blk[0], cube[:3].mean(0)), (keys, blk))
    WS._E.pop('SS', None)
    # the product: deterministic bytes, its content test
    out = os.path.join(TMP, 'npz')
    os.makedirs(out, exist_ok=True)
    rows = [('19h_2025-05-03', '10s', '20250503_124151', 30.3, 40.8, 292.7, -21.0, 25.0 / 1.25, 25.0, 1.25)]
    ftab = [('19h_2025-05-03', '10s', 300, 0, 'ok'), ('19h_2025-05-03', '1min', 50, 0, 'ok'),
            ('19h_2025-05-04', '-', 0, -1, 'deep-search refused')]
    a, b = os.path.join(out, 'I_X_27MHz.npz'), os.path.join(out, 'b', 'I_X_27MHz.npz')
    os.makedirs(os.path.dirname(b), exist_ok=True)
    WS.write_npz(a, rows, ftab, 2)
    time.sleep(1.1)
    WS.write_npz(b, rows, ftab, 2)
    check('write_npz is byte-identical on a re-run (no wall-clock member)', open(a, 'rb').read() == open(b, 'rb').read())
    t = validate.run('forced_wide_bursts', a, stokes='I', identity='X', band='27MHz')
    check('bursts content test passes a consistent file', all(x['passed'] for x in t), t)
    WS.write_npz(b, [rows[0][:7] + (6.5, 6.5 * 1.25, 1.25)], ftab, 2)
    t = validate.run('forced_wide_bursts', b, stokes='I', identity='X', band='27MHz')
    check('bursts content test fails a peak below 7 sigma', not all(x['passed'] for x in t))
    WS.write_npz(b, [('19h_2025-05-05',) + rows[0][1:]], ftab, 2)
    t = validate.run('forced_wide_bursts', b, stokes='I', identity='X', band='27MHz')
    check('bursts content test fails a peak of a file that was not searched', not all(x['passed'] for x in t))


def test_io_yield():
    from exopipe.phase3.lib.forced import wide_search as WS
    log = os.path.join(TMP, 'guard.log')
    rnd = lambda t: '%s round: lwacalim01=%.3fs/0MB lwacalim04=0.300s/0MB\n' % (time.strftime('%FT%TZ'), t)  # noqa: E731
    open(log, 'w').write(rnd(0.5) + rnd(6.2))
    check('guard rounds: the latest first', WS._guard_rounds(log) == [6.2, 0.5], WS._guard_rounds(log))
    CFG.set('forced_wide_search.IO_GUARD_LOG', repr(log))
    CFG.set('forced_wide_search.IO_GUARD_POLL_S', '1')
    WS._E.pop('GUARD_T', None)

    def calm():
        time.sleep(1.5)
        with open(log, 'a') as f:
            f.write(rnd(1.0) + rnd(2.0))
    th = threading.Thread(target=calm)
    th.start()
    waited = WS.io_yield()
    th.join()
    check('io_yield waits while the latest round is slow, until two rounds are calm', waited >= 1.0, waited)
    WS._E.pop('GUARD_T', None)
    check('io_yield does not wait when the latest round is calm', WS.io_yield() == 0.0)
    CFG.reset()


# ---------------------------------------------------------------------------------------------------------------------
# 5. the jobs
# ---------------------------------------------------------------------------------------------------------------------

def test_jobs_functions():
    from exopipe.phase3.lib.forced import wide_jobs as WJ
    e, n = WJ.tan_off(10.2, 20.3, 10.0, 20.0)
    ra, de = WJ.tan_pos(e, n, 10.0, 20.0)
    check('tan_pos inverts tan_off', abs(ra - 10.2) < 1e-9 and abs(de - 20.3) < 1e-9)
    check('master_beam: the nearest tabulated frequency, ties to the first', WJ.master_beam(20.5) == 20.7
          and WJ.master_beam(46.0) == 9.0)
    check('event_mhz: subbands by name, widebands by the table, others the default',
          (WJ.event_mhz('27MHz'), WJ.event_mhz('Blue'), WJ.event_mhz('X')) == (27.0, 78, 46.0))
    tg = {('10pc', 'B_member'): (1.0, 2.0, 'B', 0), ('10pc', 'A_ident'): (1.1, 2.1, 'A', 0)}
    check('TARGET_MEMBER first = the first in build_targets order', WJ.target_member('A_ident', tg, 'first')[0][1] == 'B_member')
    check('TARGET_MEMBER identity = the identity\'s own member', WJ.target_member('A_ident', tg, 'identity')[0][1] == 'A_ident')
    check('TARGET_MEMBER identity: else the first in (sample, target) order',
          WJ.target_member('Z', tg, 'identity')[0][1] == 'A_ident')
    # the events of a synthetic run tree: sorted vs directory order, a conflict between two trees
    run = os.path.join(TMP, 'runs', 'Run_SNAPFORCED_I10s_gen5')
    cols = ['band', 'identity', 'cand_id', 'ts', 'snr', 'sigma_frame_jyb']
    write_csv(os.path.join(run, 'candidates', '41MHz', 'r2_41MHz_T', 'detections.csv'),
              [dict(band='41MHz', identity='T', cand_id='F_41MHz_20250101_010101_T', ts='20250101_010101', snr='8.1',
                    sigma_frame_jyb='2.0')], cols)
    write_csv(os.path.join(run, 'candidates', '27MHz', 'r1_27MHz_T', 'detections.csv'),
              [dict(band='27MHz', identity='T', cand_id='F_27MHz_20250101_010101_T', ts='20250101_010101', snr='9.5',
                    sigma_frame_jyb='3.0'),
               dict(band='Red', identity='T', cand_id='F_Red_x', ts='20250101_010101', snr='9', sigma_frame_jyb='1')], cols)
    write_csv(os.path.join(run, 'candidates', 'I', 'T_27MHz', 'detections.csv'),
              [dict(band='27MHz', identity='T', cand_id='F_27MHz_20250101_010101_T', ts='20250101_010101', snr='9.5',
                    sigma_frame_jyb='3.0')], cols)
    runs = [('I10s', 'I', '10s', run)]
    ev = WJ.event_table(runs, 'sorted')
    check('events: subbands only, one per (identity, frame id), sorted order = frequency order',
          [r['band'] for r in ev] == ['27MHz', '41MHz'], ev)
    root = os.path.join(run, 'candidates')
    lst, want, seen = os.listdir(root), [], set()
    for b in lst:                                   # the first sight of each frame id in os.listdir order (load_events)
        for d in os.listdir(os.path.join(root, b)):
            for r in csv.DictReader(open(os.path.join(root, b, d, 'detections.csv'))):
                if r['band'].endswith('MHz') and r['cand_id'] not in seen:
                    seen.add(r['cand_id'])
                    want.append(r['band'])
    evd = WJ.event_table(runs, 'directory')
    check('events: directory order = os.listdir order', [r['band'] for r in evd] == want, ([r['band'] for r in evd], lst))
    check('event_conflicts: none when both trees agree', WJ.event_conflicts(runs) == [])
    write_csv(os.path.join(run, 'candidates', 'I', 'T_27MHz', 'detections.csv'),
              [dict(band='27MHz', identity='T', cand_id='F_27MHz_20250101_010101_T', ts='20250101_010101', snr='9.6',
                    sigma_frame_jyb='3.0')], cols)
    check('event_conflicts: a frame read twice with different values', len(WJ.event_conflicts(runs)) == 1)
    p = os.path.join(TMP, 'events.csv')
    WJ.write_events(p, ev)
    t = validate.run('forced_wide_events', p)
    check('events content test passes', all(x['passed'] for x in t), t)
    WJ.write_events(p + '.dup', ev + ev[:1])
    t = validate.run('forced_wide_events', p + '.dup')
    check('events content test fails an event listed twice', not all(x['passed'] for x in t))
    le = WJ.load_events(p)
    check('load_events: value = S/N x frame sigma', le[('I', '10s')]['T'][0] == ('27MHz', '20250101_010101', 9.5 * 3.0))
    # adopted job lists: the carried fields, base then extension
    a1, a2 = os.path.join(TMP, 'jobs_a.json'), os.path.join(TMP, 'jobs_b.json')
    json.dump([dict(stokes='I', identity='T', sample='HWJ', target='T', T=[1, 2], S=[1, 2.1], s74=0.7, bg_name='VLSSr X',
                    sep_arcmin=6.0, mid=[1, 2], bands=['27MHz'], r_max_arcmin=9, controls=[], bg_not_isolated=[],
                    median_mhz=27.0)], open(a1, 'w'))
    json.dump([dict(stokes='V', identity='U', target='U', sample='10pc', T=[3, 4], bands=['41MHz'],
                    bright=[[3, 4.1, 5.0, 'VLSSr Y']])], open(a2, 'w'))
    b_, e_ = WJ.adopt_jobs([a1, a2])
    check('adopt_jobs: carried fields only, base and extension', list(b_[0]) == list(WJ.BASE_KEYS) and
          list(e_[0]) == list(WJ.EXT_KEYS) and 'controls' not in b_[0])


def test_jobs_content():
    """the jobs content test on jobs built from the real catalogues around KELT-24"""
    from exopipe.phase3.lib.forced import wide_jobs as WJ
    fa = WJ.FA()
    T = KELT24_T
    near = WJ.physical_sources(T[0], T[1], 12.0, 0.5)
    near.sort(key=lambda s: float(fa.sep_arcsec(s[0], s[1], T[0], T[1])))
    S = near[0]
    sep = float(fa.sep_arcsec(S[0], S[1], T[0], T[1])) / 60.0
    base = [dict(stokes='I', identity='KELT-24', sample='HWJ', target='KELT-24', T=T, S=[S[0], S[1]], s74=S[2], bg_name=S[3],
                 sep_arcmin=sep, bands=['27MHz', '32MHz'], median_mhz=36.0)]
    br = WJ.physical_sources(T[0], T[1], 2.0 * WJ.beam_fwhm_arcmin('27MHz'), 3.0)
    br.sort(key=lambda s: -s[2])
    ext = [dict(stokes='V', identity='KELT-24', target='KELT-24', sample='HWJ', T=T, bands=['27MHz'],
                bright=[list(s) for s in br])]
    check('KELT-24: its 1-beam background is VLSSr J104826.4+714344 (the analysis\' job)', S[3] == 'VLSSr VLSSr J104826.4+714344', S)
    check('KELT-24: the 10.5 Jy source is among the >= 3 Jy within 2 FWHM at 27 MHz',
          any('J105246.5+715924' in s[3] for s in br), [s[3] for s in br])
    p = os.path.join(TMP, 'jobs.json')
    WJ.write_jobs(p, base, ext, [])
    t = validate.run('forced_wide_jobs', p)
    check('jobs content test passes jobs built by the rule', all(x['passed'] for x in t), t)
    base[0]['sep_arcmin'] = sep + 0.5
    WJ.write_jobs(p, base, ext, [])
    t = validate.run('forced_wide_jobs', p)
    check('jobs content test fails a background separation that its positions do not give', not all(x['passed'] for x in t))


# ---------------------------------------------------------------------------------------------------------------------
# 6. the v3 functions and the results' content tests
# ---------------------------------------------------------------------------------------------------------------------

def test_v3_functions():
    from exopipe.phase3.lib.forced import wide_v3 as V3
    t = CFG.forced_wide_test
    # a pure exponential across the line: the near / far windows give the exact rate at their centers
    aT = 1.0
    bins = []
    for lo in np.r_[np.arange(aT - 0.65, aT - 0.35 - 1e-9, 0.1), np.arange(aT + 0.35, aT + 0.65 - 1e-9, 0.1)]:
        a = lo + 0.05
        bins.append((a, 0, 0.1 * 2 * 0.35))
    npred, var, how = V3.fit_predict(bins, aT)
    check('fit_predict: equal empty windows predict the flat (count + 0.5) / area rate over the disc',
          abs(npred - 0.5 / (3 * 0.07) * math.pi * 0.35 ** 2) < 0.02 * npred, (npred, how))
    xs, w = V3.disc_profile()
    check('disc_profile: equal-area samples of the 0.35 disc', abs(w * len(xs) - math.pi * 0.35 ** 2) < 1e-12)
    ctl = [(0.0, n, 2.0, 0.1) for n in (0, 1, 2, 2, 3, 1, 2, 4, 2, 2, 1, 3, 2, 2, 0, 2)]
    r = V3.calibrate({'n_obs': 9, 'n_pred': 2.0, 'var': 0.1}, ctl)
    kappa = (sum(q[1] for q in ctl) + 0.5) / (sum(q[2] for q in ctl) + 0.5)
    rt = (9 + 0.5) / (kappa * 2.0 + 0.5)
    rc = [(q[1] + 0.5) / (kappa * q[2] + 0.5) for q in ctl]
    check('calibrate: kappa, ratio and the permutation p', abs(r['kappa'] - kappa) < 1e-12 and abs(r['ratio'] - rt) < 1e-12
          and r['p_emp'] == (sum(1 for x in rc if x >= rt) + 1) / (len(rc) + 1), r)
    check('calibrate: fewer than MATCH_MIN control lines -> not calibrated',
          math.isnan(V3.calibrate({'n_obs': 1, 'n_pred': 1.0, 'var': 0.1}, ctl[:5])['p_emp']))
    m = V3.matched([(0, 0, x, 0) for x in (0.5, 1.0, 2.0, 5.0, 40.0)], 2.0)
    check('matched: 1/3-3x the prediction with the floors, widened below MATCH_MIN', [q[2] for q in m] == [1.0, 2.0, 5.0], m)
    check('v2_verdict: too few nights', V3.v2_verdict(0.5, 0.1, t.V2_MIN_NIGHTS - 1, [])[0] == 'too few nights')
    good = [dict(t=x, err=0.05, n_cells=5) for x in np.linspace(-0.1, 0.1, 12)]
    v = V3.v2_verdict(0.02, 0.05, 8, good)
    check('v2_verdict: the bursts center like the controls (background emits), power high -> BACKGROUND SOURCE',
          v[0] == 'BACKGROUND SOURCE', v)
    v = V3.v2_verdict(0.95, 0.05, 8, good)
    check('v2_verdict: centered at the target -> NOT THE BACKGROUND SOURCE', v[0] == 'NOT THE BACKGROUND SOURCE', v)


def test_results_content():
    from exopipe.phase3.lib.forced.tests_wide import RESULT_COLUMNS, verdict_problems
    base = dict.fromkeys(RESULT_COLUMNS, '')
    base.update(stokes='I', identity='X', target='X', tier='10s', n_events='9', event_jy_median='5', bursts_at_target='5',
                bursts_by_band='27MHz:5', n_candidates='4', n_capable='2', centroid_fwhm='0.300', centroid_err_fwhm='0.100',
                centroid_z='3.0', centroid_nights='5')
    cases = [(dict(verdict='BACKGROUND SOURCE', named="A 1.0 Jy at 3.0'"), True),
             (dict(verdict='NOT CENTERED ON THE TARGET (source not identified)'), True),
             (dict(verdict='NOT EXPLAINED BY A CATALOGED SOURCE (centered on the target)', centroid_z='3.0'), False),
             (dict(verdict='UNDECIDED', centroid_fwhm='0.050', centroid_z='0.5'), True),
             (dict(verdict='NOT CENTERED ON THE TARGET (no cataloged source can make the events)', n_capable='0'), True),
             (dict(verdict='BACKGROUND SOURCE', named=''), False),
             (dict(verdict='UNDECIDED (fewer than 3 bursts at the target)', bursts_at_target='2', bursts_by_band='27MHz:2',
                   centroid_z='', centroid_fwhm='', centroid_err_fwhm=''), True),
             (dict(verdict='UNDECIDED', bursts_by_band='27MHz:4'), False),
             (dict(verdict='UNDECIDED', centroid_z='2.5', centroid_fwhm='0.250'), True),
             (dict(verdict='UNDECIDED', centroid_z='2.5'), False)]
    cols = ['stokes', 'identity', 'tier', 'candidate', 's74', 'sep_arcmin', 'bands_far', 'bands_near', 'reach', 'n_obs', 'n_pred',
            'pred_err', 'n_ctl', 'kappa', 'phi', 'background', 'ratio', 'ctl_ratio_median', 'ctl_ratio_p95', 'p_emp', 'p_nb', 'shown',
            'centering', 'center_t', 'centroid_to_candidate_sigma', 'night_rho', 'night_p', 'night_n', 'explains', 'excess']
    crow = dict.fromkeys(cols, '')
    crow.update(stokes='I', identity='X', tier='10s', candidate='VLSSr A', bands_far='27MHz', n_obs='10', ratio='1.2',
                p_emp='0.4', shown='False', explains='True', excess='False', night_n='0')
    cp = os.path.join(TMP, 'cand.csv')
    write_csv(cp, [crow], cols)
    t = validate.run('forced_wide_candidates', cp)
    check('candidates content test passes a consistent row', all(x['passed'] for x in t), t)
    write_csv(cp, [dict(crow, explains='False')], cols)
    t = validate.run('forced_wide_candidates', cp)
    check('candidates content test fails explains False with p 0.4 and ratio 1.2', not all(x['passed'] for x in t))
    write_csv(cp, [dict(crow, excess='True')], cols)
    t = validate.run('forced_wide_candidates', cp)
    check('candidates content test fails excess True with p 0.4 and ratio 1.2', not all(x['passed'] for x in t))
    for upd, ok in cases:
        p = verdict_problems(dict(base, **upd))
        check('verdict rule: %s z %s capable %s -> %s' % (upd.get('verdict'), upd.get('centroid_z', base['centroid_z']),
                                                         upd.get('n_capable', '2'), 'consistent' if ok else 'refused'),
              (not p) == ok, p)


# ---------------------------------------------------------------------------------------------------------------------
# 7. the night-activity table
# ---------------------------------------------------------------------------------------------------------------------

def test_activity():
    from exopipe.phase3.lib.forced import night_activity as NA
    src = os.path.join(TMP, 'dets.csv')
    rows = []

    def det(lst, night, band, k, ra='10.123456789', dec='20.12345678', stokes='I', tier='10s'):
        rows.append(dict(page='p', source='s', default_listed='1', stokes=stokes, tier=tier, lst=lst, night=night, band=band,
                         cand_id='%s_%s_%s_%d' % (lst, night, band, k), ra=ra, dec=dec, snr='8'))
    for night, n in (('2025-01-01', 10), ('2025-01-02', 20), ('2025-01-03', 40)):
        for k in range(n):
            det('00h', night, '18MHz', k)
    for night, n in (('2025-01-01', 6), ('2025-01-03', 2)):
        for k in range(n):
            det('00h', night, '27MHz', k)
    det('00h', '2025-01-01', '18MHz', 0, ra='10.1234567999')       # the same frame and position to 9 characters: one detection
    det('01h', '2025-01-05', '18MHz', 0)
    cols = list(rows[0])
    write_csv(src, rows, cols)
    cnt, nights, n, nu, wrong = NA.read_detections(src)
    check('read_detections: duplicates by (frame id, position[:9]) counted once', (n, nu, wrong) == (len(rows), len(rows) - 1, []),
          (n, nu, wrong))
    b, nr = NA.build(cnt, nights)
    got = {(r['lst'], r['night'], r['band']): (r['median_other'], r['ratio']) for r in b}
    check('bands: the count over the median of the band\'s OTHER searched nights',
          got[('00h', '2025-01-01', '18MHz')] == (30.0, '0.333') and got[('00h', '2025-01-03', '18MHz')] == (15.0, '2.67')
          and got[('00h', '2025-01-01', '27MHz')] == (2, '3'), got)
    check('bands: a band with no detection that night is not searched there (no row)', ('00h', '2025-01-02', '27MHz') not in got)
    check('bands: one searched night only -> nan ratio', got[('01h', '2025-01-05', '18MHz')][1] == 'nan')
    check('bands: nights in date order within (LST, band)', [r['night'] for r in b if r['band'] == '18MHz' and r['lst'] == '00h']
          == ['2025-01-01', '2025-01-02', '2025-01-03'])
    nn = {(r['lst'], r['night']): r for r in nr}
    a1 = nn[('00h', '2025-01-01')]
    check('nights: activity = the median over the searched bands, rank, counts',
          a1['activity'] == '%.3g' % ((10 / 30 + 6 / 2) / 2) and a1['n_bands_searched'] == 2 and a1['n_det'] == 16
          and a1['rank'] == 1 and a1['n_nights'] == 3 and nn[('01h', '2025-01-05')]['activity'] == 'nan'
          and nn[('01h', '2025-01-05')]['rank'] == '', a1)
    out = os.path.join(TMP, 'na')
    rc, err = cli(['forced', '--tier', 'snapshot', '--stages', 'activity', '--cores', 'none', '--out', out,
                   '--na-detections', src])
    r = rec(out, 'N_activity')
    check('cli: activity done (exit 0), both tables pass their content tests', rc == 0 and r.get('status') == 'done'
          and len(r['outputs']) == 2 and all(x['passed'] for x in r['tests']), (rc, r.get('reason'), err[-300:]))
    bp = os.path.join(out, CFG.night_activity.FILES[0])
    txt = open(bp).read()
    open(bp + '.bad', 'w').write(txt.replace(',2025-01-02,18MHz,20,25.0,0.8', ',2025-01-02,18MHz,20,25.0,0.9'))
    t = validate.run('night_activity_bands', bp + '.bad')
    check('bands content test fails a ratio its counts do not give', '25.0,0.8' in txt and not all(x['passed'] for x in t))
    open(bp + '.bad2', 'w').write(txt.replace(',2025-01-02,18MHz,20,', ',2025-01-02,18MHz,21,'))
    np_ = os.path.join(out, CFG.night_activity.FILES[1])
    t = validate.run('night_activity_nights', np_, bands=bp)
    check('nights content test passes against its own bands table', all(x['passed'] for x in t), t)
    t = validate.run('night_activity_nights', np_, bands=bp + '.bad2')
    check('nights content test fails against another bands table', not all(x['passed'] for x in t))
    det('02h', '2025-01-05', '18MHz', 0, stokes='V')
    write_csv(src, rows, cols)
    rc, err = cli(['forced', '--tier', 'snapshot', '--stages', 'activity', '--cores', 'none', '--out', out, '--na-detections', src])
    r = rec(out, 'N_activity')
    check('cli: N-INPUT a Stokes V row refuses the table', rc == 2 and 'N-INPUT' in r.get('reason', ''), (rc, r.get('reason')))
    rc, err = cli(['forced', '--tier', 'snapshot', '--stages', 'activity', '--cores', 'none', '--out', out, '--wide-only', 'I:X'])
    check('cli: a family-W flag with activity refused (W-FAMILY)', rc == 2 and 'W-FAMILY' in err, err[-300:])


def test_import_side_effects():
    sys.path.insert(0, os.path.join(REPO, 'tests'))
    import test_import_side_effects as T
    mine = [b for b in T.ast_violations() if ('lib/forced/wide' in b[0] or 'night_activity' in b[0] or 'tests_wide' in b[0])]
    check('no module-level side effects in the W / N files (AST layer of the standing test)', not mine, mine)


def main():
    t0 = time.time()
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    try:
        test_config()
        test_method_truth()
        test_mirrors()
        test_families()
        test_cli_gates()
        test_publish()
        test_v2_draw()
        test_control_order()
        test_monte_carlo()
        test_draws()
        test_worker_threads()
        test_pool_provenance()
        test_search_functions()
        test_io_yield()
        test_jobs_functions()
        test_jobs_content()
        test_v3_functions()
        test_results_content()
        test_activity()
        test_import_side_effects()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    bad = [r for r in RESULTS if not r[1]]
    print('\n%d checks, %d failed, %.0f s' % (len(RESULTS), len(bad), time.time() - t0))
    for n, _, d in bad:
        print('FAILED: %s  %s' % (n, str(d)[:400]))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
