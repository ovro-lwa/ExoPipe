"""Tests of `exopipe3 search --tier sublst|snapshot` (unit search-tiers). Plain asserts, no pytest:

    NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py \\
        /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_search-tiers.py

Synthetic data in a fresh /tmp/exopipe3_tests_search-tiers/ (removed at the end); READ-ONLY checks on production files
(config values against the engines and launchers, the dry-run plan of a real sub-LST cell). No processing of real data, no
ssh, no GPU. The sub-LST engine runs for real only up to its first gate (a superseded master: refused in ~2 s); the snapshot
engines are replaced by fake scripts that write the files production writes, so the orchestration runs end to end.
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time

os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
TMP = '/tmp/exopipe3_tests_search-tiers'
PY = sys.executable

import numpy as np                                   # noqa: E402
from astropy.io import fits                          # noqa: E402

from exopipe.phase3.core import env as _env          # noqa: E402
_env.legacy()
from exopipe.phase3.config import CFG, ConfigRefusal  # noqa: E402
from exopipe.phase3.core import record, validate, ledger   # noqa: E402
from exopipe.phase3.lib.search import tier_sublst as TS, tier_snapshot as TN   # noqa: E402

CLV = '/lustre/gh/main/clv15'
OPS = '/lustre/gh/main/exopipe_ops'
BR = os.path.join(REPO, 'exopipe', 'phase3')
_SAVED = []


_MISSING = object()


def setattr_tmp(obj, name, value):
    _SAVED.append((obj, name, getattr(obj, name, _MISSING)))
    setattr(obj, name, value)


def restore():
    while _SAVED:
        obj, name, v = _SAVED.pop()
        if v is _MISSING:
            delattr(obj, name)
        else:
            setattr(obj, name, v)


def ns(**kw):
    base = dict(cells=None, lsts=None, epochs=None, stokes=None, scopes=None, exclude_cells=None, stages=None,
                from_stage=None, until_stage=None, sublst_arm=None, sublst_keep_attempt=False, replay_attempt=None,
                inject=None, inject_tag=None, work=None, workers=None, snap_bands='all', snap_stage_root=None,
                snap_pause_1min=False, snap_inject_n=None, dry_run=False)
    base.update(kw)
    return argparse.Namespace(**base)


def grep_const(path, pattern):
    with open(path) as f:
        m = re.search(pattern, f.read(), re.M)
    assert m, f'{pattern!r} not found in {path}'
    return m.group(1)


# =====================================================================================================================
def test_config():
    s, n = CFG.search_sublst, CFG.search_snapshot
    # PROTECTED_KEYS refuse every override (config.py); RULED keys need a ruling or a non-production --out
    for key in ('search_sublst.MASTER_GEN', 'search_sublst.CADENCE', 'search_snapshot.MIN_ELEVATION_DEG',
                'search_snapshot.SOLAR_AUDIT_MD5'):
        try:
            CFG.set(key, 'x')
            raise AssertionError(f'{key} was overridable')
        except ConfigRefusal:
            pass
    ids = ledger.ids(CFG.paths.FIX_LEDGER)
    for sec in (s, n):
        for k, lid in sec.RULED.items():
            assert hasattr(sec, k), k
            for x in (lid if isinstance(lid, (tuple, list)) else (lid,)):
                assert x in ids, f'{sec.__name__}.RULED[{k}] = {x} is not a fix_id of the ledger'
        for k in sec.PROTECTED_KEYS:
            assert hasattr(sec, k), k
    # engine-hard-coded values = the engines' constants (read-only parse of production code)
    ss = os.path.join(CLV, 'phase3_snapshot_search.py')
    assert float(grep_const(ss, r'^MIN_ELEVATION_DEG = ([\d.]+)')) == n.MIN_ELEVATION_DEG
    assert float(grep_const(ss, r'^INTEGRATION_S = ([\d.]+)')) == n.INTEGRATION_S
    assert grep_const(ss, r'^TIER1MIN_TIME_BASIS = "([^"]+)"') == n.TIER1MIN_TIME_BASIS
    assert grep_const(ss, r'\("status", "([^"]+)"\)') == n.NIGHT_RULE_STATUS
    assert int(grep_const(ss, r'\("admitted", int\(len\(sel\) >= (\d+)\)\)')) == n.MIN_BLOCK_FRAMES
    assert grep_const(ss, r'^SOLAR_AUDIT = os.environ.get\("P3_SOLAR_AUDIT", "([^"]+)"\)') == n.SOLAR_AUDIT
    assert float(grep_const(os.path.join(CLV, 'phase3_snapshot_group_search.py'), r'^PAIR_TOL_S = ([\d.]+)')) == n.PAIR_TOL_S
    assert float(grep_const(os.path.join(BR, 'lib/search/search_i.py'), r"'MIN_ELEVATION': ([\d.]+)")) == s.I_MASK_FLOOR_DEG
    ds = os.path.join(BR, 'lib/search/deep_search.py')
    for env, key in (('P3_MIN_EPOCHS', 'MIN_EPOCHS'), ('P3_SIZE_FLOOR_BEAMS', 'SIZE_FLOOR_BEAMS'),
                     ('P3_AXIS_REMOVE', 'AXIS_REMOVE'), ('P3_SANITY_I_X', 'SANITY_I_X'),
                     ('P3_SEARCH_TIER_SIGMA', 'SEARCH_TIER_SIGMA'), ('P3_BEAM_GATE_RATIO', 'BEAM_GATE_RATIO'),
                     ('P3_V110_TIMEOUT', 'V110_TIMEOUT_S'), ('P3_TENMIN_MIN_ELEVATION', 'MIN_ELEVATION_DEG'),
                     ('P3_MASTER_GEN', 'MASTER_GEN')):
        v = grep_const(ds, r"os\.environ\.get\('" + env + r"', '([^']+)'\)")
        assert str(v) == str(getattr(s, key)) or float(v) == float(getattr(s, key)), (env, v, getattr(s, key))
    pct = os.path.join(BR, 'lib/search/parent_curve_table.py')
    assert grep_const(pct, r"P3_TABLE_ELEV_BANDS', '([^']+)'") == s.TABLE_ELEV_BANDS
    # launcher values (sublst_search_pass.sh, snapshot_fleet_node_v9.sh)
    sp = open(os.path.join(OPS, 'sublst_search_pass.sh')).read()
    assert f'P3_TABLE_PREFIX={s.TABLE_PREFIX}' in sp and 'P3_TABLE_RECORD_ONLY=1' in sp and 'P3_REFINE_OFF=1' in sp
    assert f'SEARCH_FAST_MIN_GB:-{s.FAST_MIN_FREE_GB}' in sp and f'P3_MASTER_GEN={s.MASTER_GEN}' in sp
    v9 = open(os.path.join(OPS, 'snapshot_fleet_node_v9.sh')).read()
    assert f'--thresh {n.THRESH_SIGMA:g}' in v9 and f'--nthreads {n.NTHREADS}' in v9 and f'NPAR=${{1:-{n.NPAR}}}' in v9
    assert f'P3_FLEET_MIN_FRAMES:-{n.MIN_STAGED_FRAMES}' in v9
    # interpreters: the fixed vocabulary
    assert s.ENGINE_PYTHON in CFG.site.PYTHON and n.ENGINE_PYTHON in CFG.site.PYTHON
    assert n.ENGINE_PYTHON == 'py36' and s.ENGINE_PYTHON == 'default'
    print('config: PROTECTED/RULED honoured, ledger IDs present, engine and launcher values match')


# =====================================================================================================================
def test_items():
    u = TS.expand_units(ns(cells='09h_18MHz,09h_Red_V'))
    assert [(c, S, d) for c, _l, _b, S, d in u] == [('09h_Red', 'V', None), ('09h_18MHz', 'I', None),
                                                     ('09h_18MHz', 'V', None)], u
    u = TS.expand_units(ns(cells='09h_2025-01-02_18MHz', stokes='I'))
    assert u == [('09h_18MHz', '09h', '18MHz', 'I', '2025-01-02')], u
    u = TS.expand_units(ns(epochs='09h_2025-01-02', cells='09h_Blue'))
    assert {(c, d) for c, _l, _b, _S, d in u} == {('09h_Blue', '2025-01-02')} and len(u) == 2, u
    assert len(TS.expand_units(ns(lsts='09h'))) == 36
    assert len(TS.expand_units(ns(cells='09h_*', scopes='Red,18MHz'))) == 4
    for bad in ('09h_Full', 'foo', '09h_2025-01-02_Full'):
        try:
            TS.expand_units(ns(cells=bad))
            raise AssertionError(bad)
        except record.Refusal:
            pass
    setattr_tmp(CFG.search_sublst, 'EPOCHS_MODE', 'refuse')
    try:
        TS.expand_units(ns(epochs='09h_2025-01-02'))
        raise AssertionError('per-hour mode not refused')
    except record.Refusal:
        pass
    restore()
    it = TN.expand_cells(ns(cells='21h_2025-08-26,21h_2025-08-26_36MHz', epochs='21h_2025-08-26'))
    assert it == [('21h_2025-08-26', '21h_2025-08-26', None), ('21h_2025-08-26_36MHz', '21h_2025-08-26', '36MHz')], it
    for bad in ('21h_Full', '21h_2025-08-26_Red'):
        try:
            TN.expand_cells(ns(cells=bad))
            raise AssertionError(bad)
        except record.Refusal:
            pass
    print('items: LST-band cells, unit names, night cells, --epochs, --lsts, filters, refusals')


# =====================================================================================================================
def _fits(path, data, **cards):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h = fits.Header()
    for k, v in cards.items():
        h[k.replace('_', '-') if k == 'MJD_MID' else k] = v
    fits.writeto(path, np.asarray(data, dtype=np.float32), h, overwrite=True)


def _wcs_cards(n, cd=0.0625, crval=(150.0, 30.0)):
    return dict(CTYPE1='RA---SIN', CTYPE2='DEC--SIN', CRPIX1=n / 2 + 1, CRPIX2=n / 2 + 1, CRVAL1=crval[0],
                CRVAL2=crval[1], CDELT1=-cd, CDELT2=cd, CUNIT1='deg', CUNIT2='deg')


def make_sublst_cell(root, nights=('2025-01-01', '2025-01-02'), op='ivw-mean-v3'):
    tree = os.path.join(root, 'tenmin_root_x2', '09h_18MHz', 'subtracted')
    for n_ in nights:
        for k in (0, 1):
            for S in 'IV':
                _fits(os.path.join(tree, f'{n_}T{k:02d}_{S}_sub_18MHz.fits'), np.zeros((8, 8)), BMAJ=0.5, BMIN=0.4)
    cf = os.path.join(root, 'coadd_09h', '09h_18MHz', 'coadd_final')
    for name in ('I_deep_Taper_R0_18MHz.fits', 'V_deep_Taper_R0_18MHz.fits'):
        _fits(os.path.join(cf, name), np.ones((8, 8)), COADDOP=op, **_wcs_cards(8))
    with open(os.path.join(root, 'coadd_09h', '09h_18MHz', 'residuals_provisional.csv'), 'w') as f:
        f.write('date,stokes,rms\n')


def test_sublst_preflight():
    root = os.path.join(TMP, 'sublst')
    setattr_tmp(CFG.paths, 'SUBLST_ROOT', root)
    make_sublst_cell(root)
    run = record.Run('search', ['test'], os.path.join(TMP, 'out'), CFG, dry_run=True)
    p = TS.preflight(ns(), run, '09h_18MHz', '09h', '18MHz', 'I', None)
    assert not p['refuse'] and not p['noinput'], p['gates']
    assert len(p['windows']) == 4 and sorted(p['admitted'] + p['excluded_nights']) == ['2025-01-01', '2025-01-02']
    assert any(x.endswith('I_deep_Taper_R0_18MHz.fits') for x in p['inputs'])
    env = TS.engine_env(p, '/fast/x/att')
    assert env['P3_TENMIN_ROOT'] == os.path.join(root, 'tenmin_root_x2') and env['P3_COADD_ROOT'] == os.path.join(root, 'coadd_09h')
    assert env['P3_TABLE_PREFIX'] == 'sublst_search' and env['P3_TABLE_RECORD_ONLY'] == '1' and env['P3_REFINE_OFF'] == '1'
    assert env['P3_ARCHIVE_WRITE'] == '0' and env['P3_STAGING_ROOT'] == '/fast/x/att' and env['P3_MIN_EPOCHS'] == '3'
    assert env['P3_VRESID_ROOT'] == '/fast/x/att/Run_SLST09B/vresid'
    # V without the v110 null table: refused (G2)
    setattr_tmp(CFG.search_sublst, 'V110_NULL_TRIALS', os.path.join(TMP, 'missing.csv'))
    p = TS.preflight(ns(), run, '09h_18MHz', '09h', '18MHz', 'V', None)
    assert p['refuse'] and 'G2' in p['refuse'], p['refuse']
    CFG.search_sublst.V110_NULL_TRIALS = _SAVED[-1][2]
    # too few windows -> no-input (the engine's P3-POLICY-SKIP)
    setattr_tmp(CFG.search_sublst, 'MIN_EPOCHS', 5)
    p = TS.preflight(ns(), run, '09h_18MHz', '09h', '18MHz', 'I', None)
    assert p['noinput'] and 'P3-POLICY-SKIP' in p['noinput']
    CFG.search_sublst.MIN_EPOCHS = _SAVED[-1][2]
    # per-hour: no published attempt -> refused; a night with no window -> no-input
    p = TS.preflight(ns(), run, '09h_18MHz', '09h', '18MHz', 'I', '2025-01-02')
    assert p['refuse'] and 'published attempt' in p['refuse']
    p = TS.preflight(ns(), run, '09h_18MHz', '09h', '18MHz', 'I', '2025-01-05')
    assert p['noinput'] and 'no window' in p['noinput']
    # a superseded master: refused (E2)
    _fits(os.path.join(root, 'coadd_09h', '09h_18MHz', 'coadd_final', 'I_deep_Taper_R0_18MHz.fits'), np.ones((8, 8)),
          COADDOP='ivw-mean-v2', **_wcs_cards(8))
    p = TS.preflight(ns(), run, '09h_18MHz', '09h', '18MHz', 'I', None)
    assert p['refuse'] and p['refuse'].startswith('E2'), p['refuse']
    # NO_WINDOWS.txt -> no-input
    open(os.path.join(root, 'tenmin_root_x2', '09h_18MHz', 'NO_WINDOWS.txt'), 'w').write('09h 18MHz: no window\n')
    p = TS.preflight(ns(), run, '09h_18MHz', '09h', '18MHz', 'I', None)
    assert p['noinput'] and 'NO_WINDOWS' in p['noinput']
    os.remove(os.path.join(root, 'tenmin_root_x2', '09h_18MHz', 'NO_WINDOWS.txt'))
    restore()
    print('sub-LST preflight: windows, admitted nights, engine environment, gates E2/E6/G2, per-hour refusals')


def test_sublst_engine_child():
    """The REAL engine in a core.legacy.run child, up to its master-generation gate: a superseded master is refused (exit 2
    -> Refusal); an unreadable master is an error (RuntimeError)."""
    root = os.path.join(TMP, 'sublst_engine')
    setattr_tmp(CFG.paths, 'SUBLST_ROOT', root)
    make_sublst_cell(root, op='ivw-mean-v2')
    run = record.Run('search', ['test'], os.path.join(TMP, 'out_engine'), CFG, dry_run=True)
    plan = TS.preflight(ns(), run, '09h_18MHz', '09h', '18MHz', 'I', None)
    plan['unit'] = 'sublst-B_09h_18MHz_I'
    u = run.unit(plan['unit'])
    att = os.path.join(TMP, 'att1')
    os.makedirs(att)
    t0 = time.time()
    try:
        TS.detect(run, u, plan, att)
        raise AssertionError('engine did not refuse a superseded master')
    except record.Refusal as e:
        assert 'REFUSING to search a superseded master' in str(e), e
    assert u.meta['engine']['exit'] == 2 and u.meta.get('legacy_code'), u.meta.get('engine')
    assert any(k.endswith('deep_search.py') for k in u.meta['legacy_code']), list(u.meta['legacy_code'])[:5]
    with open(os.path.join(att, '09h_18MHz.log')) as f:
        log = f.read()
    assert 'COADDOP=' in log and "required 'ivw-mean-v3'" in log
    open(os.path.join(root, 'coadd_09h', '09h_18MHz', 'coadd_final', 'I_deep_Taper_R0_18MHz.fits'), 'w').write('not fits')
    att2 = os.path.join(TMP, 'att2')
    os.makedirs(att2)
    try:
        TS.detect(run, u, plan, att2)
        raise AssertionError('an unreadable master was not an error')
    except record.Refusal:
        raise AssertionError('an unreadable master must be an error, not a refusal')
    except RuntimeError as e:
        assert 'engine' in str(e)
    restore()
    print(f'sub-LST engine child: refusal (exit 2) and error paths through core.legacy.run ({time.time() - t0:.1f} s)')


# =====================================================================================================================
CAT = ['cand_id', 'lst', 'date', 'band', 'stokes', 'cadence', 'interval', 'ra', 'dec', 'peak_jy', 'int_jy', 'snr',
       'axis_ratio', 'fwhm_maj_arcmin', 'c50_mjy', 'mstruct', 'mstruct_raw', 'sstack', 'jupiter_deg', 'n_v_epoch', 'body',
       'repeat_group', 'epoch_qa', 'fp_prediction', 'global_snr', 'size_beams', 'size_flag', 'nearest_ateam', 'moon_deg',
       'axis_norm', 'axis_flag']


def make_attempt(att, label='SLST09B', srcs=((100.37, 101.61, 60.0), (60.21, 140.48, 5.5))):
    """A sub-LST attempt: one 10-minute frame with Gaussian sources (frame noise 1), the catalogue rows at their peak
    pixels and the search's residual cutouts of them (copies of the frame pixels with the frame's WCS)."""
    from astropy.wcs import WCS
    n = 200
    rng = np.random.default_rng(3)
    d = rng.normal(0, 1.0, (n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    sig = (0.5 / 0.0625) / 2.3548
    for x0, y0, a in srcs:
        d += a * np.exp(-0.5 * (((xx - x0) / sig) ** 2 + ((yy - y0) / sig) ** 2))
    cards = dict(_wcs_cards(n), BMAJ=0.5, BMIN=0.5, BPA=0.0, BUNIT='Jy/beam', MJD_MID=60676.2)
    fp = os.path.join(att, f'Run_{label}', 'iresid', '09h_18MHz', '2025-01-02T01_I_redo_resid_18MHz.fits')
    _fits(fp, d, **cards)
    w = WCS(fits.getheader(fp)).celestial
    rr = os.path.join(att, 'transients', f'Run_{label}')
    os.makedirs(rr, exist_ok=True)
    rows = []
    d32 = fits.getdata(fp).astype(np.float64)
    for i, (x0, y0, a) in enumerate(srcs):
        xi, yi = int(round(x0)), int(round(y0))
        ra, dec = [float(v) for v in w.pixel_to_world_values(xi, yi)]
        cid = f'I1_09h_2025-01-02T01_18MHz_{i:04d}'
        r = 10
        cut = d32[yi - r:yi + r + 1, xi - r:xi + r + 1]
        cc = dict(cards)
        cc['CRPIX1'] -= (xi - r)
        cc['CRPIX2'] -= (yi - r)
        _fits(os.path.join(rr, 'candidates', cid, f'{cid}_resid.fits'), cut, **cc)
        row = dict.fromkeys(CAT, '')
        row.update(cand_id=cid, lst='09h', date='2025-01-02T01', band='18MHz', stokes='I', cadence='10min',
                   ra=f'{ra:.5f}', dec=f'{dec:.5f}', snr=f'{a:.2f}')
        rows.append(row)
    with open(os.path.join(rr, 'master_catalog.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=CAT)
        wr.writeheader()
        wr.writerows(rows)
    return fp


def test_fit_sidecar():
    att = os.path.join(TMP, 'fit_att')
    make_attempt(att)
    out = os.path.join(TMP, 'fit_out')
    p, js, summ = TS.fit_sidecar(att, 'SLST09B', out, nproc=1, say=lambda m: None)
    with open(p, newline='') as f:
        rows = list(csv.DictReader(f))
    from exopipe.phase3.core.fitting import DET_COLS
    assert list(rows[0].keys()) == ['cand_id'] + DET_COLS + ['frame', 'frame_sha1', 'fit_frac_px']
    kinds = [r['pos_kind'] for r in rows]
    assert kinds == ['gaussian', 'template'], kinds
    for r in rows:
        assert r['pos_sys_arcsec'] == '' and r['pos_err_maj_arcsec'] == '' and float(r['pos_stat_maj_arcsec']) > 0
    assert summ['cutouts_equal'] == 2 and summ['attempt'] == att and summ['peak_fallback_fraction'] == 0.0
    if not hasattr(CFG.search_fit, 'BLANK_SYSTEMATIC'):
        # EXTERNAL (step owner): lib/search/tests.py reads search_fit.BLANK_SYSTEMATIC['sublst'], which
        # _cfgdraft_search-deep.py no longer defines (the sub-LST value is search_sublst.FIT_BLANKED) - supplied here so the
        # sidecar is still checked against the owner's schema; reported in plan/impl/search-tiers.md
        print('  NOTE external: search_fit.BLANK_SYSTEMATIC missing (owner test reads it); supplied for this check')
        setattr_tmp(CFG.search_fit, 'BLANK_SYSTEMATIC', {'deep': False, 'sublst': True})
    res = validate.run('detection_positions', p, tier='sublst', attempt=att)
    assert all(r['passed'] for r in res), res
    restore()
    # a cutout that is not the frame's pixels: F1 refuses (sidecar not written)
    cut = os.path.join(att, 'transients', 'Run_SLST09B', 'candidates', 'I1_09h_2025-01-02T01_18MHz_0000',
                       'I1_09h_2025-01-02T01_18MHz_0000_resid.fits')
    with fits.open(cut, mode='update') as h:
        h[0].data[3, 3] += 1.0
    try:
        TS.fit_sidecar(att, 'SLST09B', os.path.join(TMP, 'fit_out2'), nproc=1, say=lambda m: None)
        raise AssertionError('F1 did not refuse')
    except record.Refusal as e:
        assert 'F1' in str(e)
    assert not os.path.exists(os.path.join(TMP, 'fit_out2', 'fit_positions.csv'))
    print('fit sidecar: gaussian/template on the searched frame, error model A, owner test passes, F1 refuses')


def test_sublst_replay_unit():
    """run_unit on a synthetic attempt with --replay-attempt and --stages fit: the sidecar goes to <out>/replay/<unit>/,
    is recorded as detection_positions, the attempt is read only, the unit is done; detect/publish with a replay refuse."""
    root = os.path.join(TMP, 'sublst_replay')
    setattr_tmp(CFG.paths, 'SUBLST_ROOT', root)
    make_sublst_cell(root)
    if not hasattr(CFG.search_fit, 'BLANK_SYSTEMATIC'):
        setattr_tmp(CFG.search_fit, 'BLANK_SYSTEMATIC', {'deep': False, 'sublst': True})    # EXTERNAL, see test_fit_sidecar
    att = os.path.join(TMP, 'replay_att')
    make_attempt(att)
    before = {p: os.stat(os.path.join(dp, p)).st_mtime_ns for dp, _d, fs in os.walk(att) for p in fs}
    out = os.path.join(TMP, 'replay_out')
    run = record.Run('search', ['test'], out, CFG)
    TS.run(ns(cells='09h_18MHz', stokes='I', stages='fit', replay_attempt=att, work=os.path.join(TMP, 'w')), run)
    u = run.units[0]
    assert u.status == 'done', (u.status, u.reason)
    side = os.path.join(out, 'replay', 'sublst-B_09h_18MHz_I', 'fit_positions.csv')
    assert [o['class'] for o in u.outputs] == ['detection_positions'] and u.outputs[0]['path'] == side
    after = {p: os.stat(os.path.join(dp, p)).st_mtime_ns for dp, _d, fs in os.walk(att) for p in fs}
    assert before == after, 'the replayed attempt was written'
    rec = json.load(open(os.path.join(out, '_records', 'search', 'sublst-B_09h_18MHz_I.json')))
    assert rec['status'] == 'done' and rec['admitted'] == ['2025-01-01', '2025-01-02']
    run = record.Run('search', ['test'], out, CFG)
    TS.run(ns(cells='09h_18MHz', stokes='I', stages='detect,fit', replay_attempt=att, work=os.path.join(TMP, 'w')), run)
    assert run.units[0].status == 'refused' and 'replay' in run.units[0].reason
    restore()
    print('sub-LST run_unit: replay of an attempt through the fit stage (read only, recorded), detect+replay refused')


# =====================================================================================================================
def test_lease():
    root = os.path.join(TMP, 'leases')
    a = TS.Lease(root, 'mem', 'node-memory', capacity=2, poll_s=0.1, heartbeat_s=0.2, info={'unit': 'a'})
    b = TS.Lease(root, 'mem', 'node-memory', capacity=2, poll_s=0.1, heartbeat_s=0.2, info={'unit': 'b'})
    c = TS.Lease(root, 'mem', 'node-memory', capacity=2, poll_s=0.1, wait_s=0.3, info={'unit': 'c'})
    a.acquire(say=lambda m: None)
    b.acquire(say=lambda m: None)
    assert not c.try_acquire()[0]
    try:
        c.acquire(say=lambda m: None)
        raise AssertionError('third holder admitted')
    except RuntimeError:
        pass
    a.release()
    assert c.try_acquire()[0]
    c.release()
    b.release()
    # a holder whose process is gone (this host) is reclaimed
    p = subprocess.Popen(['true'])
    p.wait()
    os.makedirs(os.path.join(root, 'slot', 'holders'), exist_ok=True)
    with open(os.path.join(root, 'slot', 'holders', 'dead.json'), 'w') as f:
        json.dump({'host': a.host, 'pid': p.pid, 'group': '10h', 'unit': 'dead'}, f)
    m1 = TS.Lease(root, 'slot', 'lst-mutex', group='09h', poll_s=0.1)
    assert m1.try_acquire()[0], 'stale holder not reclaimed'
    m2 = TS.Lease(root, 'slot', 'lst-mutex', group='09h', poll_s=0.1)
    m3 = TS.Lease(root, 'slot', 'lst-mutex', group='10h', poll_s=0.1)
    assert m2.try_acquire()[0] and not m3.try_acquire()[0]
    for x in (m1, m2):
        x.release()
    assert m3.try_acquire()[0]
    m3.release()
    print('lease: node-memory capacity, wait timeout, stale-holder reclaim, one-LST mutex')


def test_publish_rollback():
    root = os.path.join(TMP, 'pub')
    att = os.path.join(root, 'att')
    pubdir = os.path.join(root, 'out', 'search_B', '09h_18MHz_I')
    rr = os.path.join(att, 'transients', 'Run_SLST09B')
    os.makedirs(rr)
    open(os.path.join(rr, 'master_catalog.csv'), 'w').write('new\n')
    open(os.path.join(rr, 'extra.csv'), 'w').write('extra\n')
    open(os.path.join(att, '09h_18MHz.log'), 'w').write('night log\n')
    os.makedirs(os.path.join(pubdir, 'transients', 'Run_SLST09B'))
    open(os.path.join(pubdir, 'transients', 'Run_SLST09B', 'master_catalog.csv'), 'w').write('old\n')
    open(os.path.join(pubdir, '09h_18MHz.log'), 'w').write('pass log\n')
    run = record.Run('search', ['test'], os.path.join(root, 'out'), CFG, dry_run=True)
    u = run.unit('x')
    plan = dict(publish=pubdir, label='SLST09B', cell='09h_18MHz', arm='B', unit='x', date='2025-01-02')
    pub, rollback = TS.publish(u, run, plan, att)
    assert open(os.path.join(pub, 'transients', 'Run_SLST09B', 'master_catalog.csv')).read() == 'new\n'
    assert open(os.path.join(pub, '09h_18MHz.log')).read() == 'pass log\n'           # the pass's log kept
    assert open(os.path.join(pub, '09h_18MHz_2025-01-02.log')).read() == 'night log\n'
    rollback()
    assert open(os.path.join(pub, 'transients', 'Run_SLST09B', 'master_catalog.csv')).read() == 'old\n'
    assert not os.path.exists(os.path.join(pub, 'transients', 'Run_SLST09B', 'extra.csv'))
    assert not os.path.exists(os.path.join(pub, '09h_18MHz_2025-01-02.log'))
    print('publish: per-hour append keeps the pass log, transactional rollback restores the published cell')


# =====================================================================================================================
SB = ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz',
      '78MHz', '82MHz')

FAKE_COMMON = r'''
import argparse, csv, json, os, sys
import numpy as np
from astropy.io import fits

def tables(out, band, st, n=6, ncand=1, night=False):
    tag = '%s_%s_median' % (band, st)
    if night:
        json.dump({'lst': '21h', 'date': '2025-08-26', 'subband': band, 'stokes': st, 'mode': 'median',
                   'status': 'refused-night-rule', 'rule': 'night rule', 'solar_audit': '/x', 'n_frames': n,
                   'verdict_counts': {'sun-up': n}, 'sun_elevation_deg_min': 1.0, 'sun_elevation_deg_max': 5.0,
                   'branch': ['evening'], 'first_ts': '20250826_063402', 'last_ts': '20250826_063452'},
                  open(os.path.join(out, 'nightrefused_%s.json' % tag), 'w'))
        return
    ts = ['20250826_0634%02d' % (2 + 10 * i) for i in range(n)]
    dob = ['2025-08-26T06:33:%02d.1' % (57 + 0) if i == 0 else '2025-08-26T06:34:%02d.1' % (7 + 10 * (i - 1)) for i in range(n)]
    with open(os.path.join(out, 'noise_%s.csv' % tag), 'w') as f:
        w = csv.writer(f); w.writerow(['idx', 'ts', 'date_obs', 'admitted', 'global_rms_jyb'])
        for i in range(n):
            w.writerow([i, ts[i], dob[i], 1, 4.0])
    with open(os.path.join(out, 'candidates_%s.csv' % tag), 'w') as f:
        w = csv.writer(f); w.writerow(['t', 'ts', 'x', 'y', 'snr'])
        for i in range(ncand):
            w.writerow([0, ts[0], 4, 4, -6.5 if st == 'V' else 6.5])
    with open(os.path.join(out, 'events_%s.csv' % tag), 'w') as f:
        w = csv.writer(f); w.writerow(['n_det', 'best_snr']); w.writerow([1, 6.5])
    with open(os.path.join(out, 'noise_%s_1min.csv' % tag), 'w') as f:
        w = csv.writer(f); w.writerow(['block_ts', 'n_frames_in_minute', 'n_frames_used', 'member_ts', 'admitted',
                                       'refuse_reason', 'flood_refused'])
        w.writerow(['20250826_063400', n, n, ';'.join(ts), 1, '', 0])
    with open(os.path.join(out, 'candidates_%s_1min.csv' % tag), 'w') as f:
        csv.writer(f).writerow(['t', 'ts', 'snr'])
    with open(os.path.join(out, 'events_%s_1min.csv' % tag), 'w') as f:
        csv.writer(f).writerow(['n_det'])
    h = fits.Header()
    for k, v in (('CTYPE1', 'RA---SIN'), ('CTYPE2', 'DEC--SIN'), ('CRPIX1', 5.0), ('CRPIX2', 5.0), ('CRVAL1', 10.0),
                 ('CRVAL2', 20.0), ('CDELT1', -0.03), ('CDELT2', 0.03)):
        h[k] = v
    fits.writeto(os.path.join(out, 'model_%s.fits' % tag), np.ones((8, 8), np.float32), h, overwrite=True)
    json.dump({'lst': '21h', 'date': '2025-08-26', 'subband': band, 'stokes': st, 'mode': 'median', 'thresh': 5.0,
               'n_frames': n, 'n_admitted': n, 'n_candidates': ncand, 'n_events': 1, 'flow_registered': True,
               'min_elevation_deg': 30.0, 'frame_time_1min': 'integration-midpoint-dateobs-v1', 'n_candidates_1min': 0,
               'n_blocks_1min': 1, 'n_blocks_admitted_1min': 1}, open(os.path.join(out, 'runinfo_%s.json' % tag), 'w'))
'''

FAKE_SEARCH = FAKE_COMMON + r'''
ap = argparse.ArgumentParser()
for a in ('--lst', '--date', '--subband', '--stokes', '--mode', '--thresh', '--out', '--stage-root', '--nthreads', '--inject'):
    ap.add_argument(a)
a = ap.parse_args()
assert os.environ.get('P3_SNAP_FLOW') == '1' and 'P3_STALE' not in os.environ, dict(os.environ)
assert os.environ.get('P3_SNAP_CUTOUT_ROOT', '').endswith('/photometry')
print('fake search', a.subband)
if not os.path.isdir(a.stage_root):
    raise RuntimeError('no V frames (admission metric)')                 # an unstaged subband fails fast, as in v9
with open(os.path.join(a.out, 'admission_%s.csv' % a.subband), 'w') as f:
    w = csv.writer(f); w.writerow(['idx', 'ts', 'admitted'])
    for i in range(6):
        w.writerow([i, '20250826_0634%02d' % (2 + 10 * i), 1])
night = a.subband == '23MHz'
for st in ('V', 'I'):
    tables(a.out, a.subband, st, night=night)
cut = os.path.join(os.environ['P3_SNAP_CUTOUT_ROOT'], 'HWJ', a.subband, 'X')     # a cutout cube as write_cutouts writes it
os.makedirs(cut, exist_ok=True)
h = fits.Header()
for k, v in (('CTYPE1', 'RA---SIN'), ('CTYPE2', 'DEC--SIN'), ('CDELT1', -0.25), ('CDELT2', 0.25), ('OBJECT', 'X'),
             ('SAMPLE', 'HWJ'), ('LST', '21h'), ('DATE', '2025-08-26'), ('BAND', a.subband), ('STOKES', 'V'),
             ('CELL', '21h_2025-08-26'), ('UTCSTART', '20250826T063357'), ('TARGRA', 10.0), ('TARGDEC', 20.0),
             ('CUTDEG', 5.0)):
    h[k] = v
cols = [fits.Column(name='ts', format='15A', array=np.array(['20250826_063402'] * 3)),
        fits.Column(name='date_obs', format='26A', array=np.array(['2025-08-26T06:33:57.1'] * 3)),
        fits.Column(name='admitted', format='L', array=np.array([True] * 3)),
        fits.Column(name='global_rms_jyb', format='E', array=np.ones(3, np.float32))]
cols += [fits.Column(name=n, format=f, array=np.full(3, v)) for n, f, v in
         (('ra_deg', 'D', 10.0), ('dec_deg', 'D', 20.0), ('target_x', 'E', 10.0), ('target_y', 'E', 10.0))]
fits.HDUList([fits.PrimaryHDU(np.zeros((3, 20, 20), np.float32), header=h),
              fits.BinTableHDU.from_columns(cols, name='TIMES')]).writeto(
    os.path.join(cut, 'X_21h_20250826T063357_V.fits'), overwrite=True)
if night:
    raise RuntimeError('REFUSED (night rule): no frame passes - refusal recorded')
'''

FAKE_GROUP = FAKE_COMMON + r'''
ap = argparse.ArgumentParser()
for a in ('--lst', '--date', '--stokes', '--thresh', '--out', '--stage-root-fmt', '--nthreads'):
    ap.add_argument(a)
a = ap.parse_args()
st = a.stokes
tables(a.out, 'Red', st)
tables(a.out, 'Green', st, night=True)
with open(os.path.join(a.out, 'wideband_members_%s.csv' % st), 'w') as f:
    w = csv.writer(f); w.writerow(['group', 'ts', 'date_obs', 'subband', 'member_ts', 'member_date_obs', 'weight'])
    w.writerow(['Red', '20250826_063402', '2025-08-26T06:33:57.1', '18MHz', '20250826_063402', '2025-08-26T06:33:57.4', 1])
json.dump({'lst': a.lst, 'date': a.date, 'stokes': st, 'error': 'RuntimeError: REFUSED (night rule) Green',
           'groups': {'Red': 'searched', 'Green': 'night-refused', 'Blue': 'too-few-frames'}},
          open(os.path.join(a.out, 'wideband_outcome_%s.json' % st), 'w'))
raise SystemExit('REFUSED (night rule) Green - refusal recorded')
'''


def make_snapshot_cell(stage, masters, staged=('18MHz', '23MHz', '41MHz')):
    for sb in staged:
        d = os.path.join(stage, sb)
        os.makedirs(d)
        names = []
        for i in range(2):
            for kind in ('Pilot', 'Clean'):
                n = f'{sb}-{kind}-Snapshot-20250826_0634{2 + 10 * i:02d}-image.fits'
                _fits(os.path.join(d, n), np.ones((4, 4)), WSCNVIS=100)
                _fits(os.path.join(d, n[:-5] + '_flowed.fits'), np.ones((4, 4)))
                names.append(n)
        with open(os.path.join(d, '.filelist'), 'w') as f:
            f.write(''.join(f'/archive/{n}.fz\n' for n in names))
    for sb in SB:
        p = os.path.join(masters, f'21h_{sb}', 'coadd_final', f'I_deep_Taper_R0_{sb}.fits')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'w').write('master')


def test_snapshot_preflight():
    stage_root = os.path.join(TMP, 'snapstage')
    masters = os.path.join(TMP, 'snapmasters')
    make_snapshot_cell(os.path.join(stage_root, '21h_2025-08-26'), masters)
    setattr_tmp(CFG.search_snapshot, 'MASTERS', masters)
    setattr_tmp(CFG.search_snapshot, 'MIN_STAGED_FRAMES', 4)
    run = record.Run('search', ['test'], os.path.join(TMP, 'out_snap'), CFG, dry_run=True)
    a = ns(snap_stage_root=stage_root)
    p = TN.preflight(a, run, '21h_2025-08-26')
    assert not p['refuse'], p['gates']
    assert p['staged'] == {'18MHz': 4, '23MHz': 4, '41MHz': 4} and p['n_staged'] == 12
    d = os.path.join(stage_root, '21h_2025-08-26', '41MHz')
    zv = os.path.join(d, '41MHz-Pilot-Snapshot-20250826_063402-image.fits')
    os.remove(zv[:-5] + '_flowed.fits')
    p = TN.preflight(a, run, '21h_2025-08-26')
    assert p['refuse'] and p['refuse'].startswith('FLOW-INCOMPLETE'), p['refuse']
    _fits(zv, np.zeros((4, 4)), WSCNVIS=0)                                 # zero visibilities: out of both sets
    p = TN.preflight(a, run, '21h_2025-08-26')
    assert not p['refuse'] and p['zero_vis']['41MHz'] == 1, p['gates']
    open(os.path.join(d, 'x.fz'), 'w').write('')
    p = TN.preflight(a, run, '21h_2025-08-26')
    assert p['refuse'] and p['refuse'].startswith('STAGE-REFUSED')
    os.remove(os.path.join(d, 'x.fz'))
    m = os.path.join(masters, '21h_82MHz', 'coadd_final', 'I_deep_Taper_R0_82MHz.fits')
    os.remove(m)
    os.symlink('/nonexistent/master.fits', m)
    p = TN.preflight(a, run, '21h_2025-08-26')
    assert p['refuse'] and 'broken Master link' in p['refuse']
    os.remove(m)
    p = TN.preflight(a, run, '21h_2025-08-26')
    assert p['refuse'] and 'no Taper R0 Master: 82MHz' in p['refuse']
    open(m, 'w').write('master')
    setattr_tmp(CFG.search_snapshot, 'MIN_STAGED_FRAMES', 1000)
    p = TN.preflight(a, run, '21h_2025-08-26')
    assert p['refuse'] and 'need 1000' in p['refuse']
    restore()
    print('snapshot preflight: staging, Master links, per-subband flowed SET with zero-visibility frames, min frames')


def test_snapshot_end_to_end():
    stage_root = os.path.join(TMP, 'snapstage2')
    masters = os.path.join(TMP, 'snapmasters2')
    make_snapshot_cell(os.path.join(stage_root, '21h_2025-08-26'), masters)
    fake = os.path.join(TMP, 'fakeclv15')
    os.makedirs(fake)
    open(os.path.join(fake, 'phase3_snapshot_search.py'), 'w').write(FAKE_SEARCH)
    open(os.path.join(fake, 'phase3_snapshot_group_search.py'), 'w').write(FAKE_GROUP)
    setattr_tmp(CFG.search_snapshot, 'MASTERS', masters)
    setattr_tmp(CFG.search_snapshot, 'MIN_STAGED_FRAMES', 4)
    setattr_tmp(TN, 'engine_dir', lambda: fake)
    setattr_tmp(CFG.search_snapshot, 'OUT_MIN_FREE_GB', 1)
    out = os.path.join(TMP, 'snap_out')
    os.environ['P3_STALE'] = 'leaks'                                     # core.legacy.run must clear inherited P3_*
    try:
        def go(fresh=False):
            run = record.Run('search', ['test'], out, CFG, fresh=fresh, config_sections=None)
            a = ns(cells='21h_2025-08-26', snap_stage_root=stage_root, work=os.path.join(TMP, 'work'))
            TN.run(a, run)
            return run
        t0 = time.time()
        run = go()
        u = run.units[0]
        assert u.status == 'done', (u.status, u.reason, [t for t in u.tests if not t['passed']][:3])
        ship = os.path.join(out, '21h_2025-08-26')
        assert os.path.exists(os.path.join(ship, 'runinfo_18MHz_V_median.json'))
        assert os.path.exists(os.path.join(ship, 'nightrefused_23MHz_I_median.json'))
        assert os.path.exists(os.path.join(ship, 'wideband_outcome_V.json'))
        assert os.path.exists(os.path.join(ship, 'search_27MHz.log'))       # unstaged subband: searched, failed, log shipped
        ref = {os.path.basename(r['path']): r['gate'] for r in u.refused_items}
        assert ref.get('runinfo_23MHz_V_median.json') == 'night-rule' and ref.get('runinfo_Green_I_median.json') == \
            'wideband-night-refused' and ref.get('runinfo_Blue_V_median.json') == 'wideband-too-few-frames', ref
        assert len(u.planned) == 6 + 6 and len(u.refused_items) == 2 + 4
        classes = {o['class'] for o in u.outputs}
        assert {'snapshot_search', 'snapshot_1min', 'snapshot_model', 'night_refusal', 'snapshot_wideband',
                'snapshot_table'} <= classes, classes
        assert u.meta['photometry_cutouts']['n'] == 3                        # one cube per staged subband searched
        rec = json.load(open(os.path.join(out, '_records', 'search', '21h_2025-08-26.json')))
        assert rec['status'] == 'done' and rec['meta']['legacy_calls'] and all(c['argv'][0] == CFG.site.PYTHON['py36']
                                                                              for c in rec['meta']['legacy_calls'])
        assert not os.path.exists(os.path.join(TMP, 'work', 'snapshot_out', '21h_2025-08-26'))   # node-local output dropped
        # a re-run ships identical files (no clobber needed); a changed shipped file refuses the unit; --fresh replaces it
        run = go()
        assert run.units[0].status == 'done', run.units[0].reason
        with open(os.path.join(ship, 'events_18MHz_V_median.csv'), 'a') as f:
            f.write('9,9\n')
        run = go()
        assert run.units[0].status == 'refused' and 'ship.no_clobber' in run.units[0].reason, run.units[0].reason
        run = go(fresh=True)
        assert run.units[0].status == 'done' and glob_one(os.path.join(out, '_superseded', '*', 'events_18MHz_V_median.csv'))
        print(f'snapshot end to end (fake engines via core.legacy.run): outcomes, night-rule items, completeness, verify, '
              f'ship no-clobber, record ({time.time() - t0:.1f} s)')
    finally:
        os.environ.pop('P3_STALE', None)
        restore()


def glob_one(pat):
    import glob
    return glob.glob(pat)


def test_snapshot_completeness_gates():
    s = CFG.search_snapshot
    run = record.Run('search', ['test'], os.path.join(TMP, 'out_c'), CFG, dry_run=True)
    plan = dict(cell='21h_2025-08-26', subbands=['18MHz', '23MHz'], staged={'18MHz': 4, '23MHz': 4}, lst='21h',
                date='2025-08-26', excluded={})
    u = run.unit('c1')
    ok = {'18MHz': {'stokes': {'V': 'searched', 'I': 'searched'}}, '23MHz': {'stokes': {'V': 'night-refused',
                                                                                       'I': 'night-refused'}}}
    TN.subband_completeness(u, plan, '/x', ok)
    bad = {'18MHz': {'stokes': {'V': 'searched', 'I': 'failed: exit 1'}}, '23MHz': ok['23MHz']}
    try:
        TN.subband_completeness(u, plan, '/x', bad)
        raise AssertionError('incomplete accepted')
    except RuntimeError as e:
        assert 'SEARCH-INCOMPLETE' in str(e)
    excl = {'18MHz': ok['18MHz'], '23MHz': {'stokes': {'V': 'excluded', 'I': 'excluded'}}}
    assert s.EXCLUDED_STAGED_SUBBAND == 'accounted'          # v9 since exopipe_ops a40b088
    TN.subband_completeness(u, plan, '/x', excl)
    setattr_tmp(s, 'EXCLUDED_STAGED_SUBBAND', 'unaccounted')  # v9 before it
    try:
        TN.subband_completeness(u, plan, '/x', excl)
        raise AssertionError('excluded staged subband accepted under the pre-a40b088 accounting')
    except record.Refusal as e:
        assert 'gate exclusion' in str(e)
    restore()
    none = {'18MHz': {'stokes': {'V': 'failed: x', 'I': 'failed: x'}}}
    try:
        TN.subband_completeness(u, dict(plan, subbands=['18MHz'], staged={'18MHz': 4}), '/x', none)
        raise AssertionError('no runinfo accepted')
    except RuntimeError as e:
        assert 'SEARCH-INCOMPLETE' in str(e) or 'FLEET-ABORT' in str(e)
    d = os.path.join(TMP, 'wb')
    os.makedirs(d)
    json.dump({'stokes': 'V', 'groups': {'Red': 'not-run', 'Green': 'no-members', 'Blue': 'excluded'}},
              open(os.path.join(d, 'wideband_outcome_V.json'), 'w'))
    try:
        TN.wideband_completeness(u, plan, d)
        raise AssertionError('wideband not-run accepted')
    except RuntimeError as e:
        assert 'Red_V:not-run' in str(e) and 'I:no-outcome-record' in str(e)
    print('completeness gates: night-refused and excluded accounted, crash/unaccounted/abort and wideband not-run refused')


def test_content_tests_negative():
    """The snapshot content tests fail on corrupted copies of real shipped products (read from production, written to TMP)."""
    src = '/lustre/gh/main/exopipe_ops/snapshot_fleet_out/08h_2025-04-15'
    d = os.path.join(TMP, 'neg')
    os.makedirs(d)
    for n in ('runinfo_Blue_V_median.json', 'noise_Blue_V_median.csv', 'candidates_Blue_V_median.csv',
              'events_Blue_V_median.csv', 'candidates_Blue_V_median_1min.csv', 'noise_Blue_V_median_1min.csv',
              'nightrefused_18MHz_V_median.json', 'wideband_outcome_V.json', 'wideband_members_V.csv',
              'nightrefused_Green_V_median.json'):
        shutil.copy2(os.path.join(src, n), d)
    for cls, n in (('snapshot_search', 'runinfo_Blue_V_median.json'), ('snapshot_1min', 'noise_Blue_V_median_1min.csv'),
                   ('night_refusal', 'nightrefused_18MHz_V_median.json'), ('snapshot_wideband', 'wideband_outcome_V.json')):
        assert all(r['passed'] for r in validate.run(cls, os.path.join(d, n))), (cls, validate.run(cls, os.path.join(d, n)))
    with open(os.path.join(d, 'candidates_Blue_V_median.csv'), 'a') as f:      # one row more than runinfo counts
        f.write(open(os.path.join(src, 'candidates_Blue_V_median.csv')).read().splitlines()[1] + '\n')
    assert not all(r['passed'] for r in validate.run('snapshot_search', os.path.join(d, 'runinfo_Blue_V_median.json')))
    rows = open(os.path.join(d, 'noise_Blue_V_median_1min.csv')).read().splitlines()
    parts = rows[1].split(',')
    parts[0] = parts[0][:11] + '%02d00' % ((int(parts[0][11:13]) + 7) % 60)       # members outside the block minute
    open(os.path.join(d, 'noise_Blue_V_median_1min.csv'), 'w').write('\n'.join([rows[0], ','.join(parts)] + rows[2:]) + '\n')
    assert not all(r['passed'] for r in validate.run('snapshot_1min', os.path.join(d, 'noise_Blue_V_median_1min.csv')))
    shutil.copy2(os.path.join(src, 'runinfo_Blue_V_median.json'), os.path.join(d, 'runinfo_18MHz_V_median.json'))
    assert not all(r['passed'] for r in validate.run('night_refusal', os.path.join(d, 'nightrefused_18MHz_V_median.json')))
    w = json.load(open(os.path.join(d, 'wideband_outcome_V.json')))
    w['groups']['Blue'] = 'not-run'
    json.dump(w, open(os.path.join(d, 'wideband_outcome_V.json'), 'w'))
    assert not all(r['passed'] for r in validate.run('snapshot_wideband', os.path.join(d, 'wideband_outcome_V.json')))
    print('content tests: pass on real shipped products, fail on corrupted copies (counts, minute, runinfo, not-run)')


def test_cli_dry_run():
    env = dict(os.environ, PYTHONPATH=REPO)
    r = subprocess.run([PY, '-m', 'exopipe.phase3', 'search', '--tier', 'sublst', '--cells', '09h_18MHz', '--stokes', 'I',
                        '--dry-run', '--cores', 'none', '--out', os.path.join(TMP, 'dry')], capture_output=True, text=True,
                       env=env, cwd=REPO, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'gate PASS E2.coaddop_I' in r.stdout and "run_deep_cell('09h', '18MHz', label='SLST09B', stokes='I'" in r.stdout
    assert not os.path.exists(os.path.join(TMP, 'dry', '_records'))
    r = subprocess.run([PY, '-m', 'exopipe.phase3', 'search', '--tier', 'snapshot', '--cells', '21h_2025-08-26',
                        '--dry-run', '--cores', 'none', '--out', os.path.join(TMP, 'dry')], capture_output=True, text=True,
                       env=env, cwd=REPO, timeout=120)
    assert r.returncode == 2 and 'REFUSED: no staged frames' in r.stdout, r.stdout + r.stderr
    r = subprocess.run([PY, '-m', 'exopipe.phase3', 'search', '--tier', 'snapshot', '--cells', '21h_2025-08-26',
                        '--cores', 'none', '--out', '/lustre/gh/main/exopipe_ops/snapshot_fleet_out/x'],
                       capture_output=True, text=True, env=env, cwd=REPO, timeout=120)
    # refused by cli._guard_out (snapshot_fleet_out is in paths.PRODUCTION_ROOTS) or by the tier's own guard (exit 2)
    assert r.returncode != 0 and 'production root' in r.stderr and not os.path.exists(
        '/lustre/gh/main/exopipe_ops/snapshot_fleet_out/x'), r.stdout + r.stderr
    print('CLI: sub-LST dry-run plan of a real cell (exit 0), snapshot plan refused without staging (exit 2), '
          'production --out refused')


def main():
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP)
    t0 = time.time()
    try:
        for t in (test_config, test_items, test_sublst_preflight, test_sublst_engine_child, test_fit_sidecar, test_sublst_replay_unit, test_lease,
                  test_publish_rollback, test_snapshot_preflight, test_snapshot_completeness_gates,
                  test_snapshot_end_to_end, test_content_tests_negative, test_cli_dry_run):
            t()
    finally:
        restore()
        shutil.rmtree(TMP, ignore_errors=True)
    print(f'ALL search-tiers TESTS PASSED ({time.time() - t0:.1f} s)')


if __name__ == '__main__':
    main()
