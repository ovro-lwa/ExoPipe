#!/usr/bin/env python3
"""STANDING TEST - `exopipe3 curves` (the snapshot light-curve product), on synthetic data plus read-only real products.

  items       snapshot cell grammar, --scopes / --exclude-cells, unit names
  config      the curves section (draft), RULED ledger IDs exist, window lock (G13), production-tree guard, G16 / stages
  plan        planned / excluded / flow-unregistered / not-searched subbands; G1 search finished; G4 stage check; LC8
  publish     published / skipped-valid / conflict (G12) / superseded with MANIFEST
  shim        the worker shim under the PRODUCTION interpreter: window lock refuses (G13); gate mapping of module messages
  end2end     the whole step on a synthetic cell: the REAL clv15 extractor runs in place on synthetic staged frames (a test
              seam points discover_frames at a synthetic Phase-1 tree); tables published and tested; V carries the I scale;
              G6 item refusals; --resume skips; re-run conflicts (G12); --supersede moves the old tables
  negative    each content test fails on a corrupted copy of a synthetic table
  real        READ-ONLY: content tests on production tables (I passes all; V written before clv15 98595b3 fails only
              flux_scale), and --dry-run on the real cell 18h_2025-04-11

Scratch: /tmp/exopipe3_tests_curves/ (removed at the end). Nothing outside it is written; production trees are only read.
Needs: EXOPIPE3_SITE (the site file; its LEGACY_CODE_ROOTS name clv15), the production interpreter
/opt/devel/pipeline/envs/development/bin/python, the PB library (read).

Run standalone:  python tests/test_curves.py [test_name ...]
"""
import contextlib
import io
import json
import os
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
SCRATCH = '/tmp/exopipe3_tests_curves'
SEAM = os.path.join(SCRATCH, 'seam')
# Synthetic tests only: discover_frames reads the Phase-1 archive through a module constant bound as a default argument
# (clv15 phase3_snapshot_search.py:72, :206). This sitecustomize - chained by core/parity_sitecustomize in every child that
# core.legacy.run starts - points it at the synthetic archive. It lives in the test scratch, never in the step.
SEAM_CODE = """import os, sys
_root, _clv = os.environ.get('EXOPIPE3_TEST_PHASE1_ROOT'), os.environ.get('EXOPIPE3_TEST_CLV15')
if _root and _clv:
    sys.path.insert(0, _clv)
    import phase3_snapshot_search as _SS
    _SS.discover_frames.__defaults__ = (_root,) + tuple(_SS.discover_frames.__defaults__[1:])
    _SS.PHASE1_ROOT = _root
"""
REAL_LC = '/lustre/gh/main/exopipe_ops/snapshot_lightcurves'
CELL = '18h_2030-01-01'
LST, DATE = '18h', '2030-01-01'
SB = '36MHz'
NX = 128
TS = [f'20300101_12{m:02d}{s:02d}' for m in range(0, 2) for s in range(0, 60, 10)]      # 12 frames, 10 s apart


def _fresh(name):
    d = os.path.join(SCRATCH, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


# ---------------------------------------------------------------------------------------------------------------------
# synthetic cell
# ---------------------------------------------------------------------------------------------------------------------

def _wcs_header(ra0=270.0, dec0=40.0):
    from astropy.io import fits
    h = fits.Header()
    h['NAXIS'] = 2
    h['NAXIS1'] = NX
    h['NAXIS2'] = NX
    h['CTYPE1'], h['CTYPE2'] = 'RA---SIN', 'DEC--SIN'
    h['CRVAL1'], h['CRVAL2'] = ra0, dec0
    h['CRPIX1'], h['CRPIX2'] = NX / 2 + 1, NX / 2 + 1
    h['CDELT1'], h['CDELT2'] = -0.0625, 0.0625
    h['CUNIT1'], h['CUNIT2'] = 'deg', 'deg'
    h['BUNIT'] = 'JY/BEAM'
    return h


def make_cell(root, subbands=(SB,), staged=(SB,), catalog_for=(SB,), flow_registered_false=(), with_claim=True,
              nan_target=False):
    """A synthetic searched cell under ``root``: Phase-1 placeholders, staged frames, search outputs, Master catalogue,
    samples, flow fields. Returns the dict of roots."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    R = {k: os.path.join(root, k) for k in ('phase1', 'stage', 'search', 'claims', 'mcat', 'samples', 'flow', 'out',
                                            'work')}
    for d in R.values():
        os.makedirs(d, exist_ok=True)
    hdr = _wcs_header()
    w = WCS(hdr)
    rng = np.random.default_rng(1)
    # Master catalogue: 6 compact bright sources (refs), 1 'M' source, 1 outside the frame, 1 non-finite position
    srcs = [(1, 268.0, 41.0, 30.0, 'S'), (2, 271.5, 39.0, 25.0, 'S'), (3, 270.5, 42.0, 40.0, 'S'), (4, 269.0, 38.5, 22.0, 'S'),
            (5, 272.0, 41.5, 35.0, 'S'), (6, 268.5, 39.5, 28.0, 'S'), (7, 270.0, 40.0, 10.0, 'M'), (8, 200.0, 10.0, 50.0, 'S')]
    for sb in subbands:
        os.makedirs(os.path.join(R['search'], CELL), exist_ok=True)
        with open(os.path.join(R['search'], CELL, f'admission_{sb}.csv'), 'w') as f:
            f.write('idx,ts,date_obs,v_central_rms_jy,rms_outlier,beam_refused,sun_elev_deg,night_refused,admitted\n')
            for j, ts in enumerate(TS[:-1]):                      # the last frame is absent: counted fallback
                f.write(f'{j},{ts},,1.0,0,0,-20,,{0 if j == 3 else 1}\n')
        for st in ('I', 'V'):
            with open(os.path.join(R['search'], CELL, f'runinfo_{sb}_{st}_median.json'), 'w') as f:
                json.dump({'n_frames': len(TS), 'flow_registered': sb not in flow_registered_false}, f)
        if sb in catalog_for:
            with open(os.path.join(R['mcat'], f'master_R075_{LST}_{sb}.gaul.csv'), 'w') as f:
                f.write('# Gaussian list (synthetic)\n# Gaus_id, Isl_id, RA, DEC, Peak_flux, Isl_rms, S_Code\n')
                for gid, ra, dec, pk, code in srcs:
                    f.write(f'{gid}, {gid}, {ra - 360.0 if gid == 3 else ra}, {dec}, {pk}, 1.0, {code}\n')
                f.write('9, 9, nan, nan, 5.0, 1.0, S\n')
    with open(os.path.join(R['samples'], '10pc.csv'), 'w', encoding='utf-8') as f:
        f.write('﻿common_name,ra_current,dec_current\nTarget A,270.4,40.6\nTarget A,270.4,40.6\nFar,100.0,-30.0\n')
    if with_claim:
        os.makedirs(os.path.join(R['claims'], CELL), exist_ok=True)
        with open(os.path.join(R['claims'], CELL, 'done'), 'w') as f:
            f.write('synthetic\n')
    yy, xx = np.mgrid[0:NX, 0:NX]
    for sb in staged:
        run = os.path.join(R['phase1'], LST, DATE, 'Run_20300101_000000', sb, 'snapshots')
        os.makedirs(run, exist_ok=True)
        sd = os.path.join(R['stage'], sb)
        os.makedirs(sd, exist_ok=True)
        os.makedirs(os.path.join(R['flow'], CELL), exist_ok=True)
        for j, ts in enumerate(TS):
            for st in ('I', 'V'):
                name = f'{sb}-Pilot-Snapshot-{ts}-{st}-image.fits'
                open(os.path.join(run, name), 'w').close()          # archive placeholder: only the staged copy is read
                img = rng.normal(0.0, 0.2, (NX, NX)).astype(np.float32)
                if st == 'I':
                    for gid, ra, dec, pk, code in srcs[:7]:
                        x, y = w.all_world2pix([[ra, dec]], 0)[0]
                        img += (0.5 * pk * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5 ** 2))).astype(np.float32)
                if nan_target:                                    # 'Target A' outside the imaged disc: NaN in every frame
                    x, y = w.all_world2pix([[270.4, 40.6]], 0)[0]
                    img[int(y) - 3:int(y) + 4, int(x) - 3:int(x) + 4] = np.nan
                h = hdr.copy()
                h['WSCNVIS'] = 0 if (st == 'V' and j == 5) else 1000
                fits.writeto(os.path.join(sd, name), img, h)
            if j % 2 == 0:                                            # flow fields for every other instant only
                off = np.zeros((NX // 8 + 1, NX // 8 + 1, 2), np.float16)
                np.save(os.path.join(R['flow'], CELL, f'flow_{sb}_{ts}.npy'), off)
        with open(os.path.join(sd, '.filelist'), 'w') as f:
            for ts in TS:
                for st in ('I', 'V'):
                    f.write(os.path.join(run, f'{sb}-Pilot-Snapshot-{ts}-{st}-image.fits') + '\n')
    return R


def _sets(R):
    return ['--set', f"paths.SNAPSHOT_FLEET_OUT={R['search']}", '--set', f"curves.MASTER_CATALOG_DIR={R['mcat']}",
            '--set', f"paths.CLV15_ROOT={R['samples']}", '--set', f"paths.SNAPSHOT_FLOW_ROOT={R['flow']}",
            '--set', f"curves.FLEET_CLAIMS_ROOT={R['claims']}"]


@contextlib.contextmanager
def _step_env(R):
    """The extractor children see the synthetic Phase-1 archive (SEAM); config and environment restored afterwards."""
    CFG = _cfg()
    from exopipe.phase3.core import env, record
    os.makedirs(SEAM, exist_ok=True)
    with open(os.path.join(SEAM, 'sitecustomize.py'), 'w') as f:
        f.write(SEAM_CODE)
    saved = {k: os.environ.get(k) for k in ('PYTHONPATH', 'EXOPIPE3_TEST_PHASE1_ROOT', 'EXOPIPE3_TEST_CLV15')}
    os.environ['PYTHONPATH'] = SEAM + (os.pathsep + saved['PYTHONPATH'] if saved['PYTHONPATH'] else '')
    os.environ['EXOPIPE3_TEST_PHASE1_ROOT'] = R['phase1']
    os.environ['EXOPIPE3_TEST_CLV15'] = env.legacy_root('clv15')
    old_py, old_git = dict(CFG.site.PYTHON), record.git_commit
    record.git_commit = lambda path: 'testcommit'      # other agents edit this worktree: --resume needs a clean commit
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        record.git_commit = old_git
        CFG.site.PYTHON = old_py
        CFG.reset()


def _cli(argv):
    from exopipe.phase3 import cli
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.main(argv)
    _cfg().reset()
    return rc, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------------------------------------------

def test_items():
    from exopipe.phase3.lib.curves import items
    assert items.parse_item('18h_2025-04-11') == ('18h', '2025-04-11', None)
    assert items.parse_item('18h_2025-04-11_36MHz') == ('18h', '2025-04-11', '36MHz')
    for bad in ('18h_Full', '24h_2025-04-11', '18h_2025-04-11_37MHz', '18h_2025-4-11'):
        try:
            items.parse_item(bad)
            raise AssertionError(f'{bad} accepted')
        except ValueError:
            pass
    root = _fresh('items')
    os.makedirs(os.path.join(root, '18h_2025-04-11'))
    open(os.path.join(root, '18h_2025-04-11', 'admission_36MHz.csv'), 'w').close()
    os.makedirs(os.path.join(root, '18h_2025-04-12'))                 # no admission table: not in the universe
    assert items.universe(root) == ['18h_2025-04-11']
    sel = items.select('18h_*', None, None, None, root)
    assert sel == [('18h_2025-04-11', None)], sel
    sel = items.select('18h_2025-04-11_36MHz,18h_2025-04-11_41MHz', None, None, '41MHz,46MHz', root)
    assert sel == [('18h_2025-04-11', ['41MHz'])], sel
    sel = items.select('18h_2025-04-11', '19h_2025-04-11', '18h_2025-04-11_18MHz', None, root)
    assert sel[0][0] == '18h_2025-04-11' and '18MHz' not in sel[0][1] and len(sel[0][1]) == 14, sel
    assert sel[1] == ('19h_2025-04-11', None)
    try:
        items.select('18h_2025-04-11', None, None, 'Red', root)
        raise AssertionError('band scope accepted')
    except ValueError:
        pass
    assert items.unit_name('18h_2025-04-11', None, ('I', 'V')) == '18h_2025-04-11'
    assert items.unit_name('18h_2025-04-11', ['36MHz', '41MHz'], ('I',)) == '18h_2025-04-11_36MHz+41MHz_I'


def test_config():
    CFG = _cfg()
    from exopipe.phase3.core import ledger
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.steps import curves as S
    C = CFG.curves
    keys = dict(C.items())
    assert len(keys) >= 42, len(keys)
    for k in C.PROTECTED_KEYS:
        assert k in keys, k
    for k in ('SEARCH_OUT_ROOT', 'FLOWTREE_ROOT', 'SAMPLES_DIR'):          # shared roots live in config.paths
        assert k not in keys, k
    for k, ids in C.RULED.items():
        assert k in keys, k
        for lid in (ids if isinstance(ids, tuple) else (ids,)):
            assert ledger.has_id(lid, CFG.paths.FIX_LEDGER), f'{k}: ledger ID {lid} not in the fix ledger'
    assert 'curves' in S.CONFIG and S.TIERS == ('snapshot',) and S.PYTHON in CFG.site.PYTHON
    # G16 and stokes
    assert S._stokes(None) == ('I', 'V') and S._stokes('V,I') == ('I', 'V') and S._stokes('i') == ('I',)
    try:
        S._stokes('V')
        raise AssertionError('V alone accepted')
    except Refusal as e:
        assert 'G16' in str(e)
    # window lock: the mirrored keys are PROTECTED_KEYS - no override, into any --out (config.py refuses)
    from exopipe.phase3.config import ConfigRefusal
    CFG.authorize([], out='/tmp/x')
    for key, val in (('curves.STAMP_PX', '43'), ('curves.ADMISSION_MISSING', 'refuse'), ('curves.PB_LIBRARY', '/tmp/pb')):
        try:
            CFG.set(key, val)
            raise AssertionError(f'{key} override accepted')
        except ConfigRefusal as e:
            assert 'PROTECTED' in str(e)
    CFG.reset()
    # production trees of this step (spec B2): refused while PRODUCTION_WRITES is False; ruled override needs a ruling
    probs = S._config_gates(CFG, os.path.join(C.PRODUCT_ROOT, 'x'))
    assert any('PRODUCTION_WRITES' in p for p in probs), probs
    assert S._config_gates(CFG, '/tmp/exopipe3_tests_curves/out') == []
    CFG.authorize([], out='/tmp/x')
    CFG.set('curves.MASTER_CATALOG_DIR', '/tmp/mcat')
    CFG.set('paths.SNAPSHOT_FLOW_ROOT', '/tmp/flows')
    probs = S._config_gates(CFG, os.path.join(C.PRODUCT_ROOT, 'x'))
    assert any('carries a ruling' in p and 'MASTER_CATALOG_DIR' in p for p in probs), probs
    assert '/lustre/pipeline/exopipe/phase3/snapshot_flow' in S._guard_roots(CFG)     # the original value stays guarded
    assert S._production_out(CFG, '/lustre/pipeline/exopipe/phase3/snapshot_flow/x')
    CFG.reset()


def test_stages():
    import argparse
    from exopipe.phase3.steps import curves as S
    ns = argparse.Namespace
    assert S._stages(ns(stages=None, stage_root='/x')) == ('extract', 'publish')
    assert S._stages(ns(stages=None, stage_root=None)) == ('extract', 'publish')            # frames' staging (auto)
    assert S._stages(ns(stages=None, stage_root=None), auto=False) == ('stage', 'extract', 'publish')
    assert S._stages(ns(stages='stage,extract,publish', stage_root=None)) == ('stage', 'extract', 'publish')
    for bad, auto in ((ns(stages='extract', stage_root='/x'), True), (ns(stages='publish', stage_root='/x'), True),
                      (ns(stages='stage,extract,publish', stage_root='/x'), True),
                      (ns(stages='extract,publish', stage_root=None), False), (ns(stages='bogus', stage_root='/x'), True)):
        try:
            S._stages(bad, auto=auto)
            raise AssertionError(f'{bad} accepted')
        except ValueError:
            pass
    # stage root: --stage-root with {cell}; one literal DIR for several cells refused; auto = frames' staging
    CFG = _cfg()
    assert S._stage_root(ns(stage_root='/s/{cell}'), 'c1', None, 2) == '/s/c1'
    assert S._stage_root(ns(stage_root='/s/x'), 'c1', None, 1) == '/s/x'
    try:
        S._stage_root(ns(stage_root='/s/x'), 'c1', None, 2)
        raise AssertionError('one --stage-root for two cells accepted')
    except ValueError:
        pass
    assert S._stage_root(ns(stage_root=None), 'c1', '/w', 3) == os.path.join('/w', CFG.frames_snapshot.STAGE_SUBDIR, 'c1')


def test_plan():
    from exopipe.phase3.core.exclusions import Exclusions
    from exopipe.phase3.lib.curves import plan as P
    from exopipe.phase3.steps import curves as S
    CFG = _cfg()
    root = _fresh('plan')
    R = make_cell(root, subbands=('36MHz', '41MHz', '46MHz'), staged=('36MHz',), catalog_for=('36MHz',),
                  flow_registered_false=('46MHz',))
    dd = os.path.join(root, 'data_decisions.csv')
    with open(dd, 'w') as f:
        f.write('lst,subband,date,verdict,reason\n18h,41MHz,2030-01-01,exclude,synthetic\n')
    ex = Exclusions(dd)
    CFG.authorize([], out=R['out'])
    for a in _sets(R)[1::2]:
        CFG.apply_override_strings([a])
    try:
        C = CFG.curves
        pl = P.plan_cell(CELL, None, ('I', 'V'), ex, C, CFG.instrument.ALL_SUBBANDS)
        st = {sb: i['status'] for sb, i in pl['subbands'].items()}
        assert st['36MHz'] == 'planned' and st['41MHz'] == 'excluded' and st['46MHz'] == 'flow-unregistered', st
        assert st['18MHz'] == 'not-searched' and pl['planned'] == ['36MHz'], pl['planned']
        assert P.admitted(pl) == ['2030-01-01/36MHz']
        ok, why = P.search_done(pl, C)
        assert ok, why
        os.remove(os.path.join(R['search'], CELL, 'runinfo_46MHz_V_median.json'))
        pl2 = P.plan_cell(CELL, ['36MHz'], ('I', 'V'), ex, C, CFG.instrument.ALL_SUBBANDS)
        ok, why = P.search_done(pl2, C)          # G1 covers every searched subband, not only the selected one
        assert not ok and '46MHz_V' in why, why
        with open(os.path.join(R['search'], CELL, 'nightrefused_46MHz_V_median.json'), 'w') as f:
            f.write('{}')
        assert P.search_done(P.plan_cell(CELL, ['36MHz'], ('I', 'V'), ex, C, CFG.instrument.ALL_SUBBANDS), C)[0]
        os.remove(os.path.join(R['claims'], CELL, 'done'))
        ok, why = P.search_done(pl, C)
        assert not ok and 'not finished' in why, why
        # G4 stage problems
        assert P.stage_problems(R['stage'], pl) == {}
        open(os.path.join(R['stage'], '36MHz', 'x-image.fits.fz'), 'w').close()
        assert 'leftovers' in P.stage_problems(R['stage'], pl)['36MHz']
        os.remove(os.path.join(R['stage'], '36MHz', 'x-image.fits.fz'))
        os.remove(os.path.join(R['stage'], '36MHz', f'36MHz-Pilot-Snapshot-{TS[0]}-I-image.fits'))
        assert 'listed' in P.stage_problems(R['stage'], pl)['36MHz']
        assert 'no staged directory' in P.stage_problems(os.path.join(root, 'nostage'), pl)['36MHz']
        # inputs: catalogue, admission, samples, PB library, flow dir, code files
        ins = dict(P.inputs(pl, C, R['phase1'], ['/x/code.py']))
        assert os.path.join(R['mcat'], 'master_R075_18h_36MHz.gaul.csv') in ins and C.PB_LIBRARY in ins
        assert os.path.join(R['flow'], CELL) in ins and '/x/code.py' in ins
        # LC8
        os.makedirs(os.path.join(R['out'], CELL))
        open(os.path.join(R['out'], CELL, f'{CELL}_41MHz_I_lightcurves.fits'), 'w').close()
        open(os.path.join(R['out'], CELL, f'{CELL}_36MHz_I_lightcurves.fits'), 'w').close()
        bad = S._excluded_products(R['out'], CELL, ex)
        assert [os.path.basename(b) for b in bad] == [f'{CELL}_41MHz_I_lightcurves.fits'], bad
    finally:
        CFG.reset()


def test_publish():
    from exopipe.phase3.lib.curves import publish as PUB
    root = _fresh('publish')
    a, b = os.path.join(root, 'a.fits'), os.path.join(root, 'b.fits')
    with open(a, 'wb') as f:
        f.write(b'A' * 100)
    with open(b, 'wb') as f:
        f.write(b'B' * 100)
    out = os.path.join(root, 'out')
    dst = os.path.join(out, 'c', 't.fits')
    assert PUB.publish(a, dst, out, 'c')[0] == 'published' and open(dst, 'rb').read() == b'A' * 100
    assert PUB.publish(a, dst, out, 'c')[0] == 'skipped-valid'
    st, det = PUB.publish(b, dst, out, 'c')
    assert st == 'conflict' and 'G12' in det and open(dst, 'rb').read() == b'A' * 100
    st, det = PUB.publish(b, dst, out, 'c', supersede='t1')
    assert st == 'superseded' and open(dst, 'rb').read() == b'B' * 100
    moved = os.path.join(out, '_superseded', 't1', 'c', 't.fits')
    assert open(moved, 'rb').read() == b'A' * 100
    man = open(os.path.join(out, '_superseded', 't1', 'c', 'MANIFEST.csv')).read().splitlines()
    assert man[0] == 'utc,file,sha1,size,reason' and ',t.fits,' in man[1] and ',100,' in man[1]
    assert not [n for n in os.listdir(os.path.dirname(dst)) if '.tmp.' in n]


def test_lock_and_classify():
    """G13 on the real clv15 sources (match) and on a modified copy (refuse); outcome classification of the extractor's
    exit code, table and final exception line."""
    from exopipe.phase3.core import env, legacy
    from exopipe.phase3.lib.curves import lock, workers as W
    CFG = _cfg()
    clv = env.legacy_root('clv15')
    ok, probs, files = lock.check(clv, CFG.curves, CFG.paths)
    assert ok and len(files) == 4, probs
    root = _fresh('lock')
    for f in files:
        shutil.copy(f, root)
    lc = os.path.join(root, 'phase3_snapshot_lightcurves.py')
    src = open(lc).read()
    assert '\nSTAMP = 41\n' in src
    open(lc, 'w').write(src.replace('\nSTAMP = 41\n', '\nSTAMP = 43\n').replace('"P3_LC_WAIT_I_S"', '"P3_LC_WAIT"'))
    ra = os.path.join(root, 'phase3_snapshot_refastro.py')
    open(ra, 'w').write(open(ra).read().replace('os.environ.get("P3_SNAP_FLOWTREE", ', 'os.environ.get("P3_FLOW", '))
    ok, probs, _ = lock.check(root, CFG.curves, CFG.paths)
    assert not ok and any('STAMP = 43' in p for p in probs) and any('P3_SNAP_FLOWTREE' in p for p in probs) \
        and any('P3_LC_WAIT_I_S' in p for p in probs), probs
    # gate mapping and classification
    assert W.gate_of('18h x 36MHz I: no rows inside the frame') == 'G7'
    assert W.gate_of('frame 1 WCS differs from the first frame - refusing') == 'G8'
    assert W.gate_of('no flow field at all for 2030') == 'G9'
    assert W.gate_of('V: FLUXSCAL -1 / NREFSCAL 0 is not the Stokes I scale 0.4 / 60 - refusing') == 'G10'
    assert W.gate_of('no Master catalogue /x') == 'G6' and W.gate_of('something else') is None
    tab = os.path.join(root, 't.fits')

    def res(rc, tail):
        return legacy.LegacyResult(rc, None, tail, {}, False, '', 0.1, [])
    assert W.classify(res(1, 'Traceback\n  File x\nRuntimeError: 18h_x 36MHz I: no rows inside the frame'), tab, CELL,
                      SB, 'I')['gate'] == 'G7'
    o = W.classify(res(1, 'Traceback\nValueError: bad'), tab, CELL, SB, 'I')
    assert o['status'] == 'error' and o['reason'] == 'ValueError: bad', o
    assert W.classify(res(0, f'{CELL} {SB} V: no frames'), tab, CELL, SB, 'V')['gate'] == 'G11'
    assert W.classify(res(0, ''), tab, CELL, SB, 'V')['status'] == 'error'
    open(tab, 'w').close()
    assert W.classify(res(0, ''), tab, CELL, SB, 'I')['status'] == 'done'
    o = W.classify(res(1, 'RuntimeError: x V: FLUXSCAL -1 / NREFSCAL 0 is not the Stokes I scale 0.4 / 60 - refusing'),
                   tab, CELL, SB, 'V')
    assert o['gate'] == 'G10' and o['unpublished_table'] == tab, o
    # the production argument list (v9:366-370): no --run-dir, no --samples-dir by default
    a = W.argv(CFG.curves, CFG.paths, os.path.join(clv, W.SCRIPT), CELL, SB, 'I', '/st/36MHz', '/o')
    assert a == ['extract', '--cell', CELL, '--subband', SB, '--stage', '/st/36MHz', '--proto-root',
                 CFG.paths.SNAPSHOT_FLEET_OUT, '--master-catalogs', CFG.curves.MASTER_CATALOG_DIR, '--out', '/o',
                 '--stokes', 'I'], a


def test_end2end():
    """The step on a synthetic cell, the real extractor in place under the production interpreter."""
    from astropy.io import fits
    from exopipe.phase3.core import validate
    root = _fresh('end2end')
    R = make_cell(root, subbands=('36MHz', '41MHz'), staged=('36MHz',), catalog_for=('36MHz',))
    base = ['curves', '--cells', CELL, '--stage-root', R['stage'], '--out', R['out'], '--work', R['work'],
            '--cores', 'none'] + _sets(R)
    t0 = time.time()
    with _step_env(R):
        rc, out, err = _cli(base)
    assert rc == 0, (rc, out, err[-3000:])
    rec = json.load(open(os.path.join(R['out'], '_records', 'curves', f'{CELL}.json')))
    assert rec['status'] == 'done', rec['reason']
    pI = os.path.join(R['out'], CELL, f'{CELL}_36MHz_I_lightcurves.fits')
    pV = os.path.join(R['out'], CELL, f'{CELL}_36MHz_V_lightcurves.fits')
    assert sorted(o['path'] for o in rec['outputs']) == [pI, pV], rec['outputs']
    assert all(t['passed'] for t in rec['tests']), [t for t in rec['tests'] if not t['passed']]
    names = {t['name'] for t in rec['tests']}
    for n in ('structure', 'frames', 'values', 'flux_scale', 'rows', 'provenance'):
        assert f'light_curve.{n}' in names, names
    assert 'curves.complete' in names and 'curves.no_excluded_products' in names
    # G6: 41MHz has no Master catalogue -> both pairs refused by the named gate, the unit completes
    gates = sorted((os.path.basename(r['path']), r['gate']) for r in rec['refused_items'])
    assert gates == [(f'{CELL}_41MHz_I_lightcurves.fits', 'G6'), (f'{CELL}_41MHz_V_lightcurves.fits', 'G6')], gates
    hI, hV = fits.getheader(pI), fits.getheader(pV)
    assert hI['NREFSCAL'] == 6 and 0.3 < hI['FLUXSCAL'] < 0.7, (hI['NREFSCAL'], hI['FLUXSCAL'])
    assert hV['FLUXSCAL'] == hI['FLUXSCAL'] and hV['NREFSCAL'] == hI['NREFSCAL']        # ledger 2311 (M30)
    assert hI['NFRAMES'] == len(TS) and hI['NADMIT'] == len(TS) - 1                       # frame 3 not admitted
    with fits.open(pI) as h:
        kinds = [k.strip() for k in h['SOURCES'].data['kind']]
        assert kinds.count('master:S') == 6 and kinds.count('master:M') == 1 and kinds.count('target:10pc') == 2, kinds
        fr = h['FRAMES'].data
        assert fr['admitted'][-1] and not fr['admitted'][3]                   # absent from the admission table: admitted
        assert (~fr['flow_exact'].astype(bool)).sum() == len(TS) // 2         # nearest valid flow instant (ledger 1243)
    pair = rec['meta']['pairs']['36MHz_I']
    assert pair['status'] == 'published' and pair['fallbacks']['frames_absent_from_admission'] == 1, pair
    assert len(pair['fallbacks']['missing_sample_files']) == 6, pair['fallbacks']
    assert rec['meta']['pairs']['36MHz_V']['fallbacks']['zero_vis_frames_kept'] == 1
    assert any(p.endswith('/phase3_snapshot_lightcurves.py') for p in rec['inputs']), 'extractor code not recorded'
    lc = rec['meta']['legacy_code']                         # sha1 of every clv15 module the children imported (core.legacy)
    assert any(p.endswith('/phase3_snapshot_refastro.py') for p in lc) and all(p in rec['inputs'] for p in lc), lc
    calls = rec['meta']['legacy_calls']
    assert len(calls) == 2 and all(c['argv'][1].endswith('/phase3_snapshot_lightcurves.py') and c['rc'] == 0
                                   for c in calls), calls
    assert rec['admitted'] == ['2030-01-01/36MHz', '2030-01-01/41MHz'], rec['admitted']    # planned (G6 = item refusal)
    assert not os.path.exists(rec['meta']['scratch']), 'scratch kept after a clean unit'
    # verify alone (only the synthetic catalogue and flow root as context): the published tables pass
    ctx = {'master_catalog': os.path.join(R['mcat'], f'master_R075_{LST}_36MHz.gaul.csv'), 'flowtree': R['flow']}
    for p in (pI, pV):
        res = validate.run('light_curve', p, **ctx)
        assert all(r['passed'] for r in res), [r for r in res if not r['passed']]
    # --resume: record done, inputs unchanged -> skipped
    with _step_env(R):
        rc, out, err = _cli(base + ['--resume'])
    assert rc == 0 and 'skipped-valid 1' in err, (rc, err[-2000:])
    # re-run without --supersede: differing tables (EXTRACT) -> G12, unit refused, old tables kept
    time.sleep(1.1)
    old = open(pI, 'rb').read()
    with _step_env(R):
        rc, out, err = _cli(base)
    rec2 = json.load(open(os.path.join(R['out'], '_records', 'curves', f'{CELL}.json')))
    assert rc == 2 and rec2['status'] == 'refused' and 'G12' in rec2['reason'], (rc, rec2['reason'])
    assert open(pI, 'rb').read() == old
    # --supersede: the old tables move with a manifest, the new ones are published
    with _step_env(R):
        rc, out, err = _cli(base + ['--supersede', 'redo1'])
    assert rc == 0, (rc, err[-2000:])
    sup = os.path.join(R['out'], '_superseded', 'redo1', CELL)
    assert sorted(os.listdir(sup)) == sorted(['MANIFEST.csv', os.path.basename(pI), os.path.basename(pV)])
    assert open(os.path.join(sup, os.path.basename(pI)), 'rb').read() == old
    # G1: no fleet 'done' -> the unit is refused before any work
    os.remove(os.path.join(R['claims'], CELL, 'done'))
    with _step_env(R):
        rc, out, err = _cli(base + ['--supersede', 'redo2'])
    rec3 = json.load(open(os.path.join(R['out'], '_records', 'curves', f'{CELL}.json')))
    assert rc == 2 and rec3['reason'].startswith('G1'), (rc, rec3['reason'])
    print(f'    end2end: 4 step runs in {time.time() - t0:.1f} s')


def test_negative():
    """Each content test fails on a corrupted copy of a valid synthetic table."""
    import numpy as np
    from astropy.io import fits
    from exopipe.phase3.core import validate
    root = _fresh('negative')
    R = make_cell(root)
    # no --stage-root: frames' node-local staging <work>/<frames_snapshot.STAGE_SUBDIR>/<cell> (chains pass none)
    os.makedirs(os.path.join(R['work'], _cfg().frames_snapshot.STAGE_SUBDIR))
    os.symlink(R['stage'], os.path.join(R['work'], _cfg().frames_snapshot.STAGE_SUBDIR, CELL))
    with _step_env(R):
        rc, out, err = _cli(['curves', '--tier', 'snapshot', '--epochs', CELL, '--out', R['out'], '--work', R['work'],
                             '--cores', 'none'] + _sets(R))
    assert rc == 0, err[-2000:]
    rec = json.load(open(os.path.join(R['out'], '_records', 'curves', f'{CELL}.json')))
    assert rec['meta']['stage_root'] == os.path.join(R['work'], _cfg().frames_snapshot.STAGE_SUBDIR, CELL), rec['meta']
    pI = os.path.join(R['out'], CELL, f'{CELL}_{SB}_I_lightcurves.fits')
    pV = os.path.join(R['out'], CELL, f'{CELL}_{SB}_V_lightcurves.fits')
    ctx = {'master_catalog': os.path.join(R['mcat'], f'master_R075_{LST}_{SB}.gaul.csv'), 'flowtree': R['flow']}
    assert all(r['passed'] for r in validate.run('light_curve', pI, **ctx))
    bad_dir = os.path.join(root, 'bad', CELL)
    os.makedirs(bad_dir)

    def corrupt(src, fn, name=None):
        dst = os.path.join(bad_dir, name or os.path.basename(src))
        with fits.open(src) as h:
            h2 = fits.HDUList([x.copy() for x in h])
        fn(h2)
        h2.writeto(dst, overwrite=True)
        return {r['name'].split('.', 1)[1]: r for r in validate.run('light_curve', dst, **ctx)}

    def failed(res):
        return sorted(k for k, r in res.items() if not r['passed'])

    def set_card(k, v):
        return lambda h: h[0].header.__setitem__(k, v)

    assert failed(corrupt(pI, set_card('FLUXSCAL', 0.9))) == ['flux_scale']
    assert failed(corrupt(pI, set_card('PBCORR', True))) == ['provenance']
    assert failed(corrupt(pI, set_card('SUBBAND', '41MHz'))) == ['structure']
    assert failed(corrupt(pI, set_card('NADMIT', 1))) == ['structure']

    def bad_bin(h):
        h['FRAMES'].data['bin'][0] = 5
    assert failed(corrupt(pI, bad_bin)) == ['frames']

    def bad_epoch(h):
        h['FRAMES'].data['epoch_s'][2] += 1.0
    assert failed(corrupt(pI, bad_epoch)) == ['frames']

    def bad_row(h):
        h['SOURCES'].data['ra'][0] += 1e-6
    assert failed(corrupt(pI, bad_row)) == ['rows']

    def bad_kind(h):
        h['SOURCES'].data['kind'][-1] = 'target:nosuch'
    assert failed(corrupt(pI, bad_kind)) == ['rows']

    def bad_ref(h):
        h['SOURCES'].data['scale_ref'][:] = False
    assert 'flux_scale' in failed(corrupt(pI, bad_ref))

    def nan_row(h):
        h['FLUX'].data[0, :] = np.nan                                      # row 0 is a master row
    assert failed(corrupt(pI, nan_row)) == ['values']

    def nan_target(h):
        h['FLUX'].data[-1, :] = np.nan                                     # the last row is a target row: reported only
    res = corrupt(pI, nan_target)
    assert failed(res) == [] and 'no finite FLUX (outside the imaged disc): 1' in res['values']['detail'], res['values']

    def neg_rms(h):
        h['RMS'].data[0, 0] = -1.0
    assert failed(corrupt(pI, neg_rms)) == ['values']
    # V: the pre-98595b3 header (-1.0 / 0) fails flux_scale only; a V table without its I sibling fails
    shutil.copy(pI, os.path.join(bad_dir, os.path.basename(pI)))
    res = corrupt(pV, lambda h: (h[0].header.__setitem__('FLUXSCAL', -1.0), h[0].header.__setitem__('NREFSCAL', 0)))
    assert failed(res) == ['flux_scale'] and 'M30' in res['flux_scale']['detail']
    os.remove(os.path.join(bad_dir, os.path.basename(pI)))
    res = {r['name'].split('.', 1)[1]: r for r in validate.run('light_curve', os.path.join(bad_dir, os.path.basename(pV)),
                                                               **ctx)}
    assert failed(res) == ['flux_scale'] and 'no sibling' in res['flux_scale']['detail']
    # frame-size bound (ctx naxis): a frame smaller than the rows' pixels fails values
    res = {r['name'].split('.', 1)[1]: r for r in validate.run('light_curve', pI, naxis=(60, 60), **ctx)}
    assert failed(res) == ['values']


def test_real_products():
    """READ-ONLY: content tests on production tables and --dry-run on the real cell 18h_2025-04-11."""
    from exopipe.phase3.core import validate
    CFG = _cfg()
    cell = '18h_2025-04-11'
    report = []
    for p, want_fail in ((f'{REAL_LC}/{cell}/{cell}_18MHz_I_lightcurves.fits', []),
                         (f'{REAL_LC}/{cell}/{cell}_18MHz_V_lightcurves.fits', ['light_curve.flux_scale']),
                         (f'{REAL_LC}/09h_2024-12-24/09h_2024-12-24_36MHz_I_lightcurves.fits', [])):
        if not os.path.exists(p):
            print(f'    real: {p} absent - skipped')
            continue
        res = validate.run('light_curve', p)
        fails = sorted(r['name'] for r in res if not r['passed'])
        report.append((os.path.basename(p), fails))
        assert fails == want_fail, (p, [(r['name'], r['detail']) for r in res if not r['passed']])
    print('    real tables: ' + '; '.join(f'{n}: {"all pass" if not f else "fails " + ",".join(f)}' for n, f in report))
    # --dry-run: the real cell plans 15 subbands x 2 Stokes; the missing stage refuses (exit 2); nothing written
    out = _fresh('dry')
    rc, so, se = _cli(['curves', '--cells', cell, '--stage-root', os.path.join(out, 'nostage'), '--out', out,
                       '--work', os.path.join(out, 'w'), '--dry-run', '--cores', 'none'])
    assert rc == 2 and so.count(' planned ') == 15 and 'outputs: 30 light_curve tables' in so, (rc, so[-2000:], se)
    assert 'G1 search finished: ok' in so and 'REFUSE G4' in so
    assert os.listdir(out) == [], os.listdir(out)


def test_failures():
    """A table failing a content test is never published (item refused, unit refused, scratch kept); a worker that
    dies without a result is an error (exit 1), never a refusal or a done unit."""
    root = _fresh('failures')
    R = make_cell(root, nan_target=True)
    base = ['curves', '--cells', CELL, '--stage-root', R['stage'], '--out', R['out'], '--work', R['work'],
            '--cores', 'none'] + _sets(R)
    recp = os.path.join(R['out'], '_records', 'curves', f'{CELL}.json')
    with _step_env(R):
        rc, out, err = _cli(base + ['--set', 'curves.LC3_TARGET_ROWS_NEED_FLUX=True'])
    rec = json.load(open(recp))
    assert rc == 2 and rec['status'] == 'refused' and 'light_curve.values' in rec['reason'], (rc, rec['reason'])
    assert {r['gate'] for r in rec['refused_items']} == {'content-test', 'G10'}, rec['refused_items']   # V not alone
    assert not os.path.exists(os.path.join(R['out'], CELL)) or not os.listdir(os.path.join(R['out'], CELL))
    assert os.path.isdir(rec['meta']['scratch']), 'scratch removed after a failed content test'
    shutil.rmtree(rec['meta']['scratch'])
    with _step_env(R):                                   # default: target rows outside the disc are reported, not failed
        rc, out, err = _cli(base)
    rec = json.load(open(recp))
    assert rc == 0 and rec['status'] == 'done', (rc, rec['reason'])
    vals = [t for t in rec['tests'] if t['name'] == 'light_curve.values' and t['path'].endswith('_I_lightcurves.fits')]
    assert vals and 'outside the imaged disc): 2' in vals[0]['detail'], vals
    # an extractor interpreter that exits 1 with no output: an error (exit 1), never a refusal or a done unit
    CFG = _cfg()
    with _step_env(R):
        CFG.site.PYTHON = dict(CFG.site.PYTHON, py36='/bin/false')
        rc, out, err = _cli(base + ['--supersede', 'x'])
    rec = json.load(open(recp))
    assert rc == 1 and rec['status'] == 'error', (rc, rec['status'], rec['reason'])
    assert rec['meta']['pairs']['36MHz_I']['status'] == 'error' and rec['meta']['pairs']['36MHz_I']['reason'] == 'exit 1', \
        rec['meta']['pairs']
    assert rec['meta']['pairs']['36MHz_V']['gate'] == 'G10', rec['meta']['pairs']        # V not attempted without I


TESTS = [test_items, test_config, test_stages, test_plan, test_publish, test_lock_and_classify, test_end2end,
         test_failures, test_negative, test_real_products]


def main(argv):
    want = set(argv[1:])
    t0 = time.time()
    failed = []
    os.makedirs(SCRATCH, exist_ok=True)
    try:
        for t in TESTS:
            if want and t.__name__ not in want:
                continue
            t1 = time.time()
            try:
                t()
                print(f'PASS {t.__name__} ({time.time() - t1:.1f} s)')
            except Exception:
                import traceback
                failed.append(t.__name__)
                print(f'FAIL {t.__name__}\n{traceback.format_exc()}')
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    print(f'{len(failed)} failed: {failed}' if failed else f'all passed ({time.time() - t0:.1f} s)')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
