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
    ruled = {(sec, k): v for sec in ('forced_wide', 'forced_wide_search', 'forced_wide_test', 'night_activity')
             for k, v in (getattr(CFG._section(sec), 'RULED', None) or {}).items()}
    check('RULED: CONTROL_ORDER and V2_CTL_DRAW cite the refactor decision, the rest Gregg\'s v3 approval',
          ruled[('forced_wide_test', 'CONTROL_ORDER')] == ORDID and ruled[('forced_wide_test', 'V2_CTL_DRAW')] == ORDID
          and {v for k, v in ruled.items() if k[1] not in ('CONTROL_ORDER', 'V2_CTL_DRAW')} == {V3ID}
          and not any(k[0] == 'night_activity' for k in ruled), sorted(set(ruled.values())))
    check('RULED: the approved method\'s thresholds are ruled (MU, the taper, the centroid, v2), operations are not',
          all(('forced_wide_test', k) in ruled for k in ('MU', 'P_CONSISTENT', 'R_EXCESS', 'CEN_FLOOR', 'Z_OFF', 'V2_POWER_MIN',
                                                          'CTL_CAP')) and ('forced_wide_search', 'SNR_MIN') in ruled
          and ('forced_wide', 'EXT_MIN_S74_JY') in ruled and not any(k[1] in ('WORKERS', 'MAX_WORKERS', 'SEED', 'MINUTE_RULE',
                                                                            'TARGET_MEMBER', 'EVENT_ORDER', 'FILE_ERRORS')
                                                                     for k in ruled))
    check('both ruling IDs are rows of the fix ledger', ledger.has_id(V3ID, CFG.paths.FIX_LEDGER)
          and ledger.has_id(ORDID, CFG.paths.FIX_LEDGER))
    # production's values (parity config, # WAVE2) apply and are the three PROPOSED keys
    p = os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg')
    lines = [ln.split('#')[0].strip() for ln in open(p) if ln.split('#')[0].strip().startswith(('forced_wide', 'night_activity'))]
    check('parity config: production values of family W', sorted(lines) == sorted([
        "forced_wide.EVENT_ORDER = 'directory'", "forced_wide.TARGET_MEMBER = 'first'",
        "forced_wide_search.FILE_ERRORS = 'drop'", "forced_wide_test.V2_CTL_DRAW = (300, 20261102)",
        "forced_wide_test.CONTROL_ORDER = 'pool'"]), lines)
    CFG.authorize([], out=TMP)                 # two of them are RULED: overridden into a non-production --out, as parity runs do
    for ln in lines:
        k, v = ln.split('=', 1)
        CFG.set(k.strip(), v.strip())
    check('the parity config values apply', (CFG.forced_wide.EVENT_ORDER, CFG.forced_wide.TARGET_MEMBER,
                                             CFG.forced_wide_search.FILE_ERRORS, CFG.forced_wide_test.V2_CTL_DRAW,
                                             CFG.forced_wide_test.CONTROL_ORDER)
          == ('directory', 'first', 'drop', (300, 20261102), 'pool'))
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
    for lit, ok in (('default_rng(20261102)', t.SEED == 20261102), ('for _ in range(1000)', t.BOOTSTRAP == 1000),
                    ('cap=900', t.CTL_CAP == 900), ('COVER = 2.0 / 3.0', t.COVER == 2.0 / 3.0),
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
    CFG.authorize([], out=TMP)
    CFG.set('forced_wide_test.CTL_CAP', '40')
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
