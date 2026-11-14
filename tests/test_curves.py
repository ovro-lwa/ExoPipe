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
  negative    each content test fails on a corrupted copy of a synthetic table, incl. the regeneration gates T1-T4 in the
              step's context (first-frame completeness, MCATSHA required, admission flags, same_as_previous - also across
              a changed frame list), search_frames only where claimed, integrations (a second run's frame, too many frames)
  archive     curves.FRAME_SOURCE 'archive' (default): the real extractor reads synthetic archive frames (no --stage);
              --supersede reproduces the published values (same_as_previous); G17 off-grid archive frames refused
  searchframes curves.SEARCH_FRAMES (the opt-in of clv15 55469ba; default = production's plain call): default every
              discovered frame; --supersede with the opt-in holds the search's list (counts recorded); night-refused pairs
              keep the plain call (B-70); G18 before extraction (no list) and from the extractor (a listed frame not
              discovered); the dry run shows the run's own plan (G6 before G18)
  real        READ-ONLY: production tables on the pre-contract parents fail rows exactly (+ flux_scale for V written before
              clv15 98595b3); tables on the contract parents (exopipe_ops 3035b47 regeneration) pass everything, also with
              the step's context against the table they replaced; the 55469ba tables of 12h 2024-12-28 27 MHz pass
              search_frames; G17 on the pilot-era cell 01h_2024-12-24; --dry-run on the real cell 18h_2025-04-11

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
# the 2026-10-13 regeneration for the contract parent catalogues (exopipe_ops 3035b47): versioned set before install, and
# the quarantine the old (pre-contract) tables move to at install - both read only
REAL_V2 = '/lustre/gh/main/exopipe_ops/snapshot_lightcurves_v2_20261013'
REAL_QUAR = '/lustre/gh/main/clv15/quarantine_superseded_20261013/snapshot_lightcurves_pre_contractcat'
CELL = '18h_2030-01-01'
LST, DATE = '18h', '2030-01-01'
SB = '36MHz'
NX = 128
TS = [f'20300101_12{m:02d}{s:02d}' for m in range(0, 2) for s in range(0, 60, 10)]      # 12 frames, 10 s apart
# the search's own frame list (noise_<sb>_<S>_median.csv) leaves out one discovered frame - as the re-flowed search of
# 12h 2024-12-28 dropped the frames that did not flow (clv15 55469ba): the plain call (default) samples it, --search-frames
# (curves.SEARCH_FRAMES, the opt-in) does not
SEARCH_DROPPED = TS[4]
SEARCH_TS = [t for t in TS if t != SEARCH_DROPPED]


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
              nan_target=False, archive_fits=False, night_refused=(), search_ts=None):
    """A synthetic searched cell under ``root``: Phase-1 placeholders (real frames with ``archive_fits``: the archive the
    extractor reads with curves.FRAME_SOURCE 'archive'), staged frames, search outputs - admission table, runinfo and the
    search's own frame list noise_<sb>_<S>_median.csv (``search_ts``, default SEARCH_TS),
    or for a subband in ``night_refused`` the engine's night-refusal records, no frame list and an admission table that
    admits no frame - Master catalogue, samples, flow fields. Returns the dict of roots."""
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
            if sb in night_refused:                               # the night rule refuses every frame (SS:1024-1026)
                for j, ts in enumerate(TS):
                    f.write(f'{j},{ts},,1.0,0,0,2.0,sun-up,0\n')
            else:
                for j, ts in enumerate(TS[:-1]):                  # the last frame is absent: counted fallback
                    f.write(f'{j},{ts},,1.0,0,0,-20,,{0 if j == 3 else 1}\n')
        for st in ('I', 'V'):
            if sb in night_refused:                               # the engine's record; no runinfo, no frame list
                with open(os.path.join(R['search'], CELL, f'nightrefused_{sb}_{st}_median.json'), 'w') as f:
                    json.dump({'rule': 'synthetic night rule', 'first_ts': TS[0], 'last_ts': TS[-1]}, f)
                continue
            with open(os.path.join(R['search'], CELL, f'runinfo_{sb}_{st}_median.json'), 'w') as f:
                json.dump({'n_frames': len(TS), 'flow_registered': sb not in flow_registered_false}, f)
            with open(os.path.join(R['search'], CELL, f'noise_{sb}_{st}_median.csv'), 'w') as f:   # production's columns
                f.write('idx,ts,date_obs,elev_bin,n_subbands,wsum,admitted,global_rms_jyb,n_components,flood_refused,'
                        'refuse_reason\n')
                for j, ts in enumerate(SEARCH_TS if search_ts is None else search_ts):
                    f.write(f'{j},{ts},,0,,,{0 if ts == TS[3] else 1},0.2,3,0,\n')
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
                if archive_fits:                                  # the archive frame itself (uncompressed: read_frame HDU 0)
                    fits.writeto(os.path.join(run, name), img, h, overwrite=True)
            if j % 2 == 0:                                            # flow fields for every other instant only
                off = np.zeros((NX // 8 + 1, NX // 8 + 1, 2), np.float16)
                np.save(os.path.join(R['flow'], CELL, f'flow_{sb}_{ts}.npy'), off)
        with open(os.path.join(sd, '.filelist'), 'w') as f:
            for ts in TS:
                for st in ('I', 'V'):
                    f.write(os.path.join(run, f'{sb}-Pilot-Snapshot-{ts}-{st}-image.fits') + '\n')
    return R


def _sets(R, staged=True):
    return ['--set', f"paths.SNAPSHOT_FLEET_OUT={R['search']}", '--set', f"curves.MASTER_CATALOG_DIR={R['mcat']}",
            '--set', f"paths.CLV15_ROOT={R['samples']}", '--set', f"paths.SNAPSHOT_FLOW_ROOT={R['flow']}",
            '--set', f"curves.FLEET_CLAIMS_ROOT={R['claims']}"] + (['--set', 'curves.FRAME_SOURCE=staged'] if staged else [])


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
    CFG = _cfg()
    assert CFG.curves.FRAME_SOURCE == 'archive'                       # the regeneration recipe (clv15 e0785b7, ops 3035b47)
    assert S._archive(ns(stage_root=None)) and not S._archive(ns(stage_root='/x'))
    assert S._stages(ns(stages=None, stage_root=None), auto=False) == ('extract', 'publish')   # archive: nothing to stage
    try:
        S._stages(ns(stages='stage,extract,publish', stage_root=None))
        raise AssertionError('--stages stage accepted with archive frames')
    except ValueError:
        pass
    CFG.set('curves.FRAME_SOURCE', 'staged')
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
    CFG.reset()
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
        # inputs: catalogue, admission, samples, PB library, flow dir, code files; the search's frame lists only with
        # curves.SEARCH_FRAMES (the opt-in: production's plain call reads none)
        ins = dict(P.inputs(pl, C, R['phase1'], ['/x/code.py']))
        assert os.path.join(R['mcat'], 'master_R075_18h_36MHz.gaul.csv') in ins and C.PB_LIBRARY in ins
        assert os.path.join(R['flow'], CELL) in ins and '/x/code.py' in ins
        assert not C.SEARCH_FRAMES and not any('noise_' in k for k in ins), sorted(ins)       # default: production's call
        # the frame root of a GPU-frame cell (production frame_root, clv15 5c83ddc): under paths.PHASE1_ROOT a listed cell
        # reads paths.SNAPSHOT_GPU_FRAME_ROOT and another cell PHASE1_ROOT; any other root (this synthetic archive, a
        # run-wide override) wins for every cell
        from exopipe.phase3.core import exclusions as EX
        saved_gpu = EX.gpu_frame_cells
        try:
            EX.gpu_frame_cells = lambda path=None: {(pl['lst'], pl['date'])}
            assert P.frame_root(pl, CFG.paths.PHASE1_ROOT) == CFG.paths.SNAPSHOT_GPU_FRAME_ROOT
            assert P.frame_root(pl, R['phase1']) == R['phase1']
            EX.gpu_frame_cells = lambda path=None: set()
            assert P.frame_root(pl, CFG.paths.PHASE1_ROOT) == CFG.paths.PHASE1_ROOT
        finally:
            EX.gpu_frame_cells = saved_gpu
        assert P.pair_plan(pl, C, ('I', 'V'), ['36MHz']) == {('36MHz', 'I'): ('extract', False),
                                                             ('36MHz', 'V'): ('extract', False)}
        CFG.set('curves.SEARCH_FRAMES', 'True')
        C = CFG.curves
        ins = dict(P.inputs(pl, C, R['phase1'], ['/x/code.py']))
        assert all(os.path.join(R['search'], CELL, f'noise_36MHz_{st}_median.csv') in ins for st in ('I', 'V')), sorted(ins)
        assert P.pair_plan(pl, C, ('I', 'V'), ['36MHz']) == {('36MHz', 'I'): ('extract', True),
                                                             ('36MHz', 'V'): ('extract', True)}
        # a runinfo without its frame list: G18 before extraction, its V G10; the absence fingerprinted by the directory
        lI = os.path.join(R['search'], CELL, 'noise_36MHz_I_median.csv')
        saved = open(lI).read()
        os.remove(lI)
        pl3 = P.plan_cell(CELL, None, ('I', 'V'), ex, C, CFG.instrument.ALL_SUBBANDS)
        pp = P.pair_plan(pl3, C, ('I', 'V'), ['36MHz'])
        assert pp[('36MHz', 'I')][:2] == ('refused', 'G18') and pp[('36MHz', 'V')][:2] == ('refused', 'G10'), pp
        assert dict(P.inputs(pl3, C, R['phase1'], []))[os.path.join(R['search'], CELL)] == \
            {'absent': 'noise_36MHz_I_median.csv'}
        assert P.pair_plan(pl3, C, ('I',), ['36MHz']).keys() == {('36MHz', 'I')}
        # a night-refused pair (record, no list): the plain call, as today (B-70) - never refused; no directory fingerprint
        with open(os.path.join(R['search'], CELL, 'nightrefused_36MHz_I_median.json'), 'w') as f:
            f.write('{}')
        pl3 = P.plan_cell(CELL, None, ('I', 'V'), ex, C, CFG.instrument.ALL_SUBBANDS)
        pp = P.pair_plan(pl3, C, ('I', 'V'), ['36MHz'])
        assert pp == {('36MHz', 'I'): ('extract', False), ('36MHz', 'V'): ('extract', True)}, pp
        assert os.path.join(R['search'], CELL) not in dict(P.inputs(pl3, C, R['phase1'], []))
        CFG.set('curves.SEARCH_FRAMES', 'False')
        C = CFG.curves
        os.remove(os.path.join(R['search'], CELL, 'nightrefused_36MHz_I_median.json'))
        open(lI, 'w').write(saved)
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
    # the paths production reads from the environment at import (RUN_PINS, PHASE1_ROOT, GPU_CELLS_FILE): another path named
    # in this environment refuses
    for var in ('P3_SNAP_RUN_PINS', 'P3_SNAP_PHASE1_ROOT', 'P3_SNAP_GPU_CELLS'):
        saved_v = os.environ.get(var)
        os.environ[var] = '/tmp/elsewhere'
        try:
            ok, probs, _ = lock.check(clv, CFG.curves, CFG.paths)
        finally:
            if saved_v is None:
                del os.environ[var]
            else:
                os.environ[var] = saved_v
        assert not ok and len(probs) == 1 and probs[0].startswith(var + '=/tmp/elsewhere'), (var, probs)
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
    assert not any('--search-frames' in p for p in probs), probs
    # SS.INTEGRATION_S (content test light_curve.integrations) is a locked literal too
    ss = os.path.join(root, 'phase3_snapshot_search.py')
    sst = open(ss).read()
    assert '\nINTEGRATION_S = 10.03102041' in sst
    open(ss, 'w').write(sst.replace('\nINTEGRATION_S = 10.03102041', '\nINTEGRATION_S = 10.0'))
    ok, probs, _ = lock.check(root, CFG.curves, CFG.paths)
    assert not ok and any('INTEGRATION_S = 10.0 ' in p for p in probs), probs
    open(ss, 'w').write(sst)
    # the frame roots of frame_root (clv15 5c83ddc): the GPU root is a locked literal, PHASE1_ROOT is read from
    # P3_SNAP_PHASE1_ROOT with paths.PHASE1_ROOT its default
    gpu = '\nGPU_FRAME_ROOT = "/lustre/pipeline/snapshots_gpu_pipeline"\n'
    p1 = '\nPHASE1_ROOT = os.environ.get("P3_SNAP_PHASE1_ROOT", "/lustre/pipeline/exopipe/phase1")\n'
    assert gpu in sst and p1 in sst
    open(ss, 'w').write(sst.replace(gpu, '\nGPU_FRAME_ROOT = "/x"\n')
                        .replace(p1, '\nPHASE1_ROOT = "/lustre/pipeline/exopipe/phase1"\n'))
    ok, probs, _ = lock.check(root, CFG.curves, CFG.paths)
    want_p1 = "PHASE1_ROOT = '/lustre/pipeline/exopipe/phase1' != os.environ.get('P3_SNAP_PHASE1_ROOT'"
    ssp = [p for p in probs if p.startswith('phase3_snapshot_search.py ')]       # the copy's other modules stay modified
    assert not ok and len(ssp) == 2 and any("GPU_FRAME_ROOT = '/x'" in p for p in ssp) \
        and any(want_p1 in p for p in ssp), probs
    open(ss, 'w').write(sst)
    # --search-frames (clv15 55469ba) joins the lock only with the opt-in curves.SEARCH_FRAMES: a module without the option,
    # the frame-list literal or the G18 refusal then refuses before extraction; production's plain call (the default)
    # needs none of them
    open(lc, 'w').write(src.replace('"--search-frames"', '"--search-framez"').replace('noise_%s_%s_median.csv', 'noise.csv'))
    ok, probs, _ = lock.check(root, CFG.curves, CFG.paths)
    assert not CFG.curves.SEARCH_FRAMES and not any('--search-frames' in p for p in probs), probs
    CFG.authorize([], out=os.path.join(SCRATCH, 'out'))
    CFG.set('curves.SEARCH_FRAMES', 'True')
    ok, probs, _ = lock.check(root, CFG.curves, CFG.paths)
    assert not ok and sum('--search-frames' in p for p in probs) == 2, probs
    CFG.reset()
    # gate mapping and classification
    assert W.gate_of('18h x 36MHz I: no rows inside the frame') == 'G7'
    assert W.gate_of('frame 1 WCS differs from the first frame - refusing') == 'G8'
    assert W.gate_of('no flow field at all for 2030') == 'G9'
    assert W.gate_of('V: FLUXSCAL -1 / NREFSCAL 0 is not the Stokes I scale 0.4 / 60 - refusing') == 'G10'
    assert W.gate_of('no Master catalogue /x') == 'G6' and W.gate_of('something else') is None
    assert W.gate_of("12h_2024-12-28 27MHz I: 3 frames of the search's list are not discovered (e.g. 20241228_120001) - "
                     "refusing") == 'G18'                                    # clv15 55469ba
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
    # the production argument list (v9:366-370): no --run-dir, no --samples-dir, no --search-frames by default
    a = W.argv(CFG.curves, CFG.paths, os.path.join(clv, W.SCRIPT), CELL, SB, 'I', '/st/36MHz', '/o')
    assert a == ['extract', '--cell', CELL, '--subband', SB, '--stage', '/st/36MHz', '--proto-root',
                 CFG.paths.SNAPSHOT_FLEET_OUT, '--master-catalogs', CFG.curves.MASTER_CATALOG_DIR, '--out', '/o',
                 '--stokes', 'I'], a
    # archive frames (the regeneration recipe, exopipe_ops 3035b47): no --stage
    a = W.argv(CFG.curves, CFG.paths, os.path.join(clv, W.SCRIPT), CELL, SB, 'I', None, '/o')
    assert a == ['extract', '--cell', CELL, '--subband', SB, '--proto-root', CFG.paths.SNAPSHOT_FLEET_OUT,
                 '--master-catalogs', CFG.curves.MASTER_CATALOG_DIR, '--out', '/o', '--stokes', 'I'], a
    # a pair plan.pair_plan extracts with the search's own frame list (curves.SEARCH_FRAMES, clv15 55469ba)
    a = W.argv(CFG.curves, CFG.paths, os.path.join(clv, W.SCRIPT), CELL, SB, 'V', None, '/o', search_frames=True)
    assert a[-1] == '--search-frames' and a[:-1] == ['extract', '--cell', CELL, '--subband', SB, '--proto-root',
                                                      CFG.paths.SNAPSHOT_FLEET_OUT, '--master-catalogs',
                                                      CFG.curves.MASTER_CATALOG_DIR, '--out', '/o', '--stokes', 'V'], a
    sp = W.make_spec(CFG.curves, CFG.paths, os.path.join(clv, W.SCRIPT), 'py36', CELL, SB, ['I', 'V'], None,
                     os.path.join(root, 'spec'), search_frames={'I': False, 'V': True})
    assert '--search-frames' not in sp['argv']['I'] and sp['argv']['V'][-1] == '--search-frames', sp['argv']


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
    for n in ('structure', 'frames', 'values', 'flux_scale', 'rows', 'provenance', 'search_frames', 'integrations'):
        assert f'light_curve.{n}' in names, names
    assert 'curves.complete' in names and 'curves.no_excluded_products' in names
    # G6: 41MHz has no Master catalogue -> both pairs refused by the named gate, the unit completes
    gates = sorted((os.path.basename(r['path']), r['gate']) for r in rec['refused_items'])
    assert gates == [(f'{CELL}_41MHz_I_lightcurves.fits', 'G6'), (f'{CELL}_41MHz_V_lightcurves.fits', 'G6')], gates
    hI, hV = fits.getheader(pI), fits.getheader(pV)
    assert hI['NREFSCAL'] == 6 and 0.3 < hI['FLUXSCAL'] < 0.7, (hI['NREFSCAL'], hI['FLUXSCAL'])
    assert hV['FLUXSCAL'] == hI['FLUXSCAL'] and hV['NREFSCAL'] == hI['NREFSCAL']        # ledger 2311 (M30)
    # production's call (curves.SEARCH_FRAMES False): every discovered frame, the one the search's list dropped too; frame 3
    # is not admitted
    assert hI['NFRAMES'] == len(TS) and hI['NADMIT'] == len(TS) - 1, (hI['NFRAMES'], hI['NADMIT'])
    for p in (pI, pV):
        with fits.open(p) as h:
            got = [str(x).strip() for x in h['FRAMES'].data['ts']]
        assert got == TS and SEARCH_DROPPED in got, (p, got)
    with fits.open(pI) as h:
        kinds = [k.strip() for k in h['SOURCES'].data['kind']]
        assert kinds.count('master:S') == 6 and kinds.count('master:M') == 1 and kinds.count('target:10pc') == 2, kinds
        fr = h['FRAMES'].data
        assert fr['admitted'][-1] and not fr['admitted'][3]                   # absent from the admission table: admitted
        assert (~fr['flow_exact'].astype(bool)).sum() == len(TS) // 2         # nearest valid flow instant (ledger 1243)
    pair = rec['meta']['pairs']['36MHz_I']
    assert pair['status'] == 'published' and pair['fallbacks']['frames_absent_from_admission'] == 1, pair
    assert pair['discovered_not_sampled'] == 0 and pair['search_frames'] is False, pair
    sf = [t for t in rec['tests'] if t['name'] == 'light_curve.search_frames']
    assert len(sf) == 2 and all(t['passed'] and t['value'] == 'n/a' for t in sf), sf          # the plain call: not compared
    ig = [t for t in rec['tests'] if t['name'] == 'light_curve.integrations']
    assert len(ig) == 2 and all(t['passed'] and t['value'] == len(TS) for t in ig), ig
    assert not any('--search-frames' in c['argv'] for c in rec['meta']['legacy_calls']), rec['meta']['legacy_calls']
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
    # verify alone (only the synthetic catalogue, flow root and search outputs as context): the published tables pass
    ctx = {'master_catalog': os.path.join(R['mcat'], f'master_R075_{LST}_36MHz.gaul.csv'), 'flowtree': R['flow'],
           'search_dir': os.path.join(R['search'], CELL)}
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
    ctx = {'master_catalog': os.path.join(R['mcat'], f'master_R075_{LST}_{SB}.gaul.csv'), 'flowtree': R['flow'],
           'search_dir': os.path.join(R['search'], CELL)}
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
    from astropy.io import fits as _f
    if _f.getheader(pI).get('MCATSHA'):                       # the extractor stamps it since clv15 e0785b7
        assert failed(corrupt(pI, set_card('MCATSHA', '0' * 64))) == ['parent_catalogue']

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

    # search_frames (clv15 55469ba; LC-REGEN-TEST (b)) applies only to a table extracted with --search-frames (ctx
    # search_frames; default curves.SEARCH_FRAMES False): this table is production's plain call - all 12 frames, also the one
    # the search's list dropped - so it is not compared by default, and fails where it is claimed to be the list's
    def search_dir_with(name, ts_list, night_refused=False):
        d = os.path.join(root, name, CELL)
        os.makedirs(d, exist_ok=True)
        for st in ('I', 'V'):
            if night_refused:
                open(os.path.join(d, f'nightrefused_{SB}_{st}_median.json'), 'w').write('{}')
                continue
            with open(os.path.join(d, f'noise_{SB}_{st}_median.csv'), 'w') as f:
                f.write('idx,ts,admitted\n' + ''.join(f'{j},{t},1\n' for j, t in enumerate(ts_list)))
        return d

    def run_sf(p, **kw):
        return {r['name'].split('.', 1)[1]: r for r in validate.run('light_curve', p, **dict(ctx, **kw))}
    res = run_sf(pI)
    assert failed(res) == [] and res['search_frames']['value'] == 'n/a', res['search_frames']
    res = run_sf(pI, search_frames=True)                       # the cell's own list dropped one of the table's frames
    assert failed(res) == ['search_frames'] and '1 not in the search' in res['search_frames']['detail'], res['search_frames']
    for name, lst_, why in (('same', TS, None), ('more', TS + ['20300101_120200'], '1 of the list missing'),
                            ('reordered', TS[::-1], None)):
        res = run_sf(pI, search_frames=True, search_dir=search_dir_with(name, lst_))
        assert failed(res) == ([] if why is None else ['search_frames']), (name, res['search_frames'])
        assert why is None or why in res['search_frames']['detail'], (name, res['search_frames'])
    res = run_sf(pI, search_frames=True, search_dir=os.path.join(root, 'nolist'))
    assert failed(res) == ['search_frames'] and 'no search frame list' in res['search_frames']['detail'], res
    res = run_sf(pI, search_frames=True, search_dir=search_dir_with('nightref', None, night_refused=True))
    assert failed(res) == [] and 'refused the night' in res['search_frames']['detail'], res['search_frames']   # B-70

    def dup_frame(h):                            # a frame twice in FRAMES fails search_frames (and the frames test)
        h['FRAMES'].data['ts'][1] = h['FRAMES'].data['ts'][0]
    dup = os.path.join(bad_dir, os.path.basename(pI))
    corrupt(pI, dup_frame)
    res = run_sf(dup, search_frames=True, search_dir=search_dir_with('same', TS))
    assert 'search_frames' in failed(res) and '1 repeated' in res['search_frames']['detail'], res['search_frames']
    # integrations (the reviewer's gap check), on the frames' recorded DATE-OBS (the admission table): distinct integrations
    # pass even with stamps 5 s apart (two runs' naming: start / midpoint, as 12h 2024-12-28 27 MHz); two frames of ONE
    # integration fail; without a recorded DATE-OBS a stamp 1 s from another fails; more frames than one LST hour fails
    import datetime as _dt

    def iso(t, dt=0.0):
        return (_dt.datetime.strptime(t, '%Y%m%d_%H%M%S') + _dt.timedelta(seconds=dt)).strftime('%Y-%m-%dT%H:%M:%S.%f')[:21]

    def admission_with(name, dates):
        p_ = os.path.join(root, f'admission_{name}.csv')
        with open(p_, 'w') as f:
            f.write('idx,ts,date_obs,admitted\n' + ''.join(f'{j},{t},{d},1\n' for j, (t, d) in enumerate(dates)))
        return p_
    # (these admission tables admit every frame, so light_curve.frames differs from them: only integrations is read here)
    res = run_sf(pI, admission=admission_with('ok', [(t, iso(t, -4.9)) for t in TS]))
    assert res['integrations']['passed'] and f'{len(TS)} by recorded DATE-OBS' in res['integrations']['detail'], \
        res['integrations']
    res = run_sf(pI, admission=admission_with('twice', [(t, iso(TS[4] if j == 5 else t, -4.9)) for j, t in enumerate(TS)]))
    assert not res['integrations']['passed'] and '1 integrations held by more than one frame' in \
        res['integrations']['detail'] and f'{TS[4]} and {TS[5]}' in res['integrations']['detail'], res['integrations']

    def stamp_5s(h):                            # frame 1 stamped 5 s after frame 0 (another run's naming), a distinct integration
        t1 = (_dt.datetime.strptime(TS[0], '%Y%m%d_%H%M%S') + _dt.timedelta(seconds=5)).strftime('%Y%m%d_%H%M%S')
        h['FRAMES'].data['ts'][1] = t1
        h['FRAMES'].data['epoch_s'][1] = h['FRAMES'].data['epoch_s'][0] + 5.0
    corrupt(pI, stamp_5s)
    s5 = os.path.join(bad_dir, os.path.basename(pI))
    t1 = (_dt.datetime.strptime(TS[0], '%Y%m%d_%H%M%S') + _dt.timedelta(seconds=5)).strftime('%Y%m%d_%H%M%S')
    dates = [(t, iso(t, -4.9)) for t in TS]
    dates[1] = (t1, iso(TS[1], -4.9))                                # its DATE-OBS: the second integration
    res = run_sf(s5, admission=admission_with('stamp5', dates))
    assert 'integrations' not in failed(res), res['integrations']
    dates[1] = (t1, iso(TS[0], -4.9))                                # the same stamps, but one integration recorded twice
    res = run_sf(s5, admission=admission_with('stamp5twice', dates))
    assert 'integrations' in failed(res) and 'held by more than one frame' in res['integrations']['detail'], res

    def second_run_frame(h):
        t1 = (_dt.datetime.strptime(TS[0], '%Y%m%d_%H%M%S') + _dt.timedelta(seconds=1)).strftime('%Y%m%d_%H%M%S')
        h['FRAMES'].data['ts'][1] = t1
        h['FRAMES'].data['epoch_s'][1] = h['FRAMES'].data['epoch_s'][0] + 1.0
    res = corrupt(pI, second_run_frame)                                # the synthetic admission table records no DATE-OBS
    assert failed(res) == ['integrations'] and 'stamped less than 4.0 s away' in res['integrations']['detail'], \
        res['integrations']
    CFG_ = _cfg()
    CFG_.authorize([], out=os.path.join(root, 'out'))
    CFG_.set('curves.MAX_FRAMES_PER_CELL', '10')
    try:
        res = run_sf(pI)
        assert failed(res) == ['integrations'] and '12 frames > 10' in res['integrations']['detail'], res['integrations']
    finally:
        CFG_.reset()
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
    # ---- the regeneration recipe's gates T1-T4 (exopipe_ops 3035b47 snapshot_lightcurves_regen_test.py) in the step's ctx
    from exopipe.phase3.lib.curves import workers as W
    adm = os.path.join(R['search'], CELL, f'admission_{SB}.csv')
    fa = W.table_facts(pI, 'I', os.path.join(R['stage'], SB), adm, R['samples'], _cfg().curves.SAMPLES)
    assert fa['naxis'] == (NX, NX) and fa['frame_header']['NAXIS1'] == NX and fa['mixed_grid_frames'] == 0, fa
    full = dict(ctx, frame_header=fa['frame_header'], require_mcatsha=True, admission=adm)

    def run_ctx(p, **kw):
        return {r['name'].split('.', 1)[1]: r for r in validate.run('light_curve', p, **dict(full, **kw))}
    res = run_ctx(pI)
    assert failed(res) == [] and 'the 7 catalogue Gaussians inside the first frame' in res['rows']['detail'], res['rows']
    # T2: a first frame on which fewer Gaussians fall inside the margin -> the table has extra master rows
    res = run_ctx(pI, frame_header=dict(fa['frame_header'], NAXIS1=60, NAXIS2=60))
    assert failed(res) == ['rows'] and 'extra' in res['rows']['detail'] and 'missing 0' in res['rows']['detail'], res['rows']
    # T2: a catalogue with one more Gaussian inside the frame -> a parent without its row (and MCATSHA no longer matches: T1)
    cat2 = os.path.join(root, 'cat_plus_one.gaul.csv')
    with open(ctx['master_catalog']) as f, open(cat2, 'w') as g:
        g.write(f.read() + '10, 10, 270.2, 40.3, 15.0, 1.0, S\n')
    res = run_ctx(pI, master_catalog=cat2)
    assert failed(res) == ['parent_catalogue', 'rows'] and 'missing 1' in res['rows']['detail'], res['rows']
    # T1: a table without MCATSHA passes alone (rows decide) and fails where the step requires the card
    res = corrupt(pI, lambda h: h[0].header.remove('MCATSHA'))
    assert failed(res) == [] and 'no MCATSHA' in res['parent_catalogue']['detail']
    nom = os.path.join(bad_dir, os.path.basename(pI))
    assert failed(run_ctx(nom)) == ['parent_catalogue']
    # T3: admission flags differing from the search's admission table (frame 3 admitted there now)
    adm2 = os.path.join(root, 'admission_changed.csv')
    with open(adm) as f, open(adm2, 'w') as g:
        g.write(f.read().replace(f',{TS[3]},,1.0,0,0,-20,,0', f',{TS[3]},,1.0,0,0,-20,,1'))
    res = run_ctx(pI, admission=adm2)
    assert failed(res) == ['frames'] and 'admission table' in res['frames']['detail'], res['frames']
    # T3/T4: against the table it replaces - itself passes; a value changed at an identical position fails; an admitted
    # flag changed at a frame both hold fails
    assert failed(run_ctx(pI, previous=pI)) == []
    prev = os.path.join(root, 'previous_I.fits')
    with fits.open(pI) as h:
        h2 = fits.HDUList([x.copy() for x in h])
    h2['FLUX'].data[0, 1] += 0.001
    h2.writeto(prev, overwrite=True)
    res = run_ctx(pI, previous=prev)
    assert failed(res) == ['same_as_previous'] and 'FLUX differs' in res['same_as_previous']['detail'], res
    h2['FLUX'].data[0, 1] -= 0.001
    h2['FRAMES'].data['admitted'][0] = not h2['FRAMES'].data['admitted'][0]
    h2.writeto(prev, overwrite=True)
    res = run_ctx(pI, previous=prev)
    assert failed(res) == ['same_as_previous'] and 'admitted flag differs at 1 of the 12 frames both hold' in \
        res['same_as_previous']['detail'], res['same_as_previous']
    # the frame list may change under --supersede (2026-11-02 review): a previous table without frame 4 (here: the new one
    # adds it) passes, the values compared at the 11 frames both hold - and a changed value there still fails; a previous
    # table sharing no frame fails
    import numpy as np
    keep = [j for j in range(len(TS)) if j != 4]
    with fits.open(pI) as h:
        h3 = fits.HDUList([x.copy() for x in h])
    h3['FLUX'].data = np.ascontiguousarray(h3['FLUX'].data[:, keep])
    h3['RMS'].data = np.ascontiguousarray(h3['RMS'].data[:, keep])
    h3['FRAMES'].data = h3['FRAMES'].data[keep]
    h3[0].header['NFRAMES'] = len(keep)
    h3.writeto(prev, overwrite=True)
    res = run_ctx(pI, previous=prev)
    assert failed(res) == [] and 'frames previous 11, new 12, both 11 (dropped 0, added 1)' in \
        res['same_as_previous']['detail'], res['same_as_previous']
    h3['FLUX'].data[0, 4] += 0.001                             # the previous table's frame 4 is the new table's frame 5
    h3.writeto(prev, overwrite=True)
    res = run_ctx(pI, previous=prev)
    assert failed(res) == ['same_as_previous'] and 'FLUX differs' in res['same_as_previous']['detail'], res
    h3['FRAMES'].data['ts'] = ['20300101_130000'] * len(keep)
    h3.writeto(prev, overwrite=True)
    res = run_ctx(pI, previous=prev)
    assert failed(res) == ['same_as_previous'] and 'no frame in common' in res['same_as_previous']['detail'], res


@contextlib.contextmanager
def _parent_archive(phase1, master):
    """Synthetic archive for the step's OWN process too (archive frames: workers.archive_frames lists them, G17 reads the
    first frame and the own Master): discover_frames' default root and workers.master_file patched, restored afterwards."""
    from exopipe.phase3.core import env
    from exopipe.phase3.lib.curves import workers as W
    env.legacy()
    import phase3_snapshot_search as SS
    old_def, old_mf = SS.discover_frames.__defaults__, W.master_file
    SS.discover_frames.__defaults__ = (phase1,) + tuple(old_def[1:])
    W.master_file = lambda lst, sb: master
    try:
        yield
    finally:
        SS.discover_frames.__defaults__ = old_def
        W.master_file = old_mf


def test_archive():
    """curves.FRAME_SOURCE 'archive' (default; the regeneration recipe, exopipe_ops 3035b47 / clv15 e0785b7): the REAL
    extractor reads the archive frames itself (no --stage); the step's context carries the regeneration gates T1-T4; a
    --supersede re-extraction reproduces the published values (same_as_previous); G17 refuses archive frames off the own
    Master's grid (the fleet would regrid them)."""
    import numpy as np
    from astropy.io import fits
    root = _fresh('archive')
    R = make_cell(root, archive_fits=True)
    master_on = os.path.join(root, 'master_on_grid.fits')       # the own Master on the frames' grid (128 px)
    fits.writeto(master_on, np.zeros((NX, NX), 'float32'), _wcs_header())

    def argv(out, *extra):
        return (['curves', '--cells', CELL, '--out', out, '--work', R['work'], '--cores', 'none'] + _sets(R, staged=False)
                + list(extra))
    recp = os.path.join(R['out'], '_records', 'curves', f'{CELL}.json')
    with _step_env(R), _parent_archive(R['phase1'], master_on):
        rc, out, err = _cli(argv(R['out']))
    rec = json.load(open(recp))
    assert rc == 0 and rec['status'] == 'done', (rc, rec['reason'], err[-2000:])
    assert rec['meta']['frame_source'] == 'archive' and rec['meta']['stage_root'] is None, rec['meta']
    assert all('--stage' not in c['argv'] for c in rec['meta']['legacy_calls']), rec['meta']['legacy_calls']
    names = {t['name'] for t in rec['tests']}
    assert {'light_curve.parent_catalogue', 'light_curve.same_as_previous'} <= names, names
    rows = [t for t in rec['tests'] if t['name'] == 'light_curve.rows']
    assert rows and all('catalogue Gaussians inside the first frame' in t['detail'] for t in rows), rows
    o = rec['outputs'][0]['ctx']
    assert o['require_mcatsha'] and o['frame_header']['NAXIS1'] == NX and o['admission'].endswith(f'admission_{SB}.csv'), o
    # --supersede: the re-extraction from the same frames and flow fields reproduces the published values (T3/T4)
    time.sleep(1.1)
    with _step_env(R), _parent_archive(R['phase1'], master_on):
        rc, out, err = _cli(argv(R['out'], '--supersede', 'regen1'))
    rec = json.load(open(recp))
    assert rc == 0 and rec['status'] == 'done', (rc, rec['reason'])
    sap = [t for t in rec['tests'] if t['name'] == 'light_curve.same_as_previous']
    assert len(sap) == 2 and all(t['passed'] and 'identical-position rows equal' in t['detail'] for t in sap), sap
    assert all(o['ctx']['previous'].startswith(os.path.join(R['out'], '_superseded', 'regen1')) for o in rec['outputs'])
    # G17: the own Master on another grid (the fleet regrids the archive frames onto it) -> both pairs refused before
    # extraction; every planned pair refused = the unit is refused
    master_off = os.path.join(root, 'master_off_grid.fits')
    h = _wcs_header()
    h['NAXIS1'] = h['NAXIS2'] = 2 * NX
    fits.writeto(master_off, np.zeros((2 * NX, 2 * NX), 'float32'), h)
    out2 = os.path.join(root, 'out2')
    with _step_env(R), _parent_archive(R['phase1'], master_off):
        rc, out, err = _cli(argv(out2))
    rec = json.load(open(os.path.join(out2, '_records', 'curves', f'{CELL}.json')))
    assert rc == 2 and rec['status'] == 'refused', (rc, rec['status'], rec['reason'])
    assert sorted(r['gate'] for r in rec['refused_items']) == ['G17', 'G17'] and not rec['meta'].get('legacy_calls'), rec
    with _step_env(R), _parent_archive(R['phase1'], master_off):          # the dry-run shows it
        rc, out, err = _cli(argv(out2, '--dry-run'))
    assert rc == 2 and f'{SB} pairs refused - G17' in out, (rc, out[-1500:])
    with _step_env(R), _parent_archive(R['phase1'], master_off):          # the fleet recipe (--stage-root): no G17
        rc, out, err = _cli(argv(out2, '--stage-root', R['stage'], '--dry-run'))
    assert rc == 0 and 'G17' not in out, (rc, out[-1500:])


def _real_table(cell, name, new):
    """A production table read only: built on the contract parent catalogues (new: MCATSHA present, clv15 e0785b7; the
    regeneration exopipe_ops 3035b47 installs them in place) or on the pre-contract ones (no MCATSHA: still installed, or
    moved to the regeneration's quarantine at install). None when absent."""
    from astropy.io import fits
    for root in ((REAL_LC, REAL_V2) if new else (REAL_LC, REAL_QUAR)):
        p = os.path.join(root, cell, name)
        if os.path.exists(p) and bool(fits.getheader(p).get('MCATSHA')) == new:
            return p
    return None


def test_real_products():
    """READ-ONLY: content tests on production tables (pre-contract parents fail rows - and V written before clv15 98595b3
    fails flux_scale - exactly; contract-parent tables pass everything, also with the step's context: first-frame
    completeness T2, MCATSHA T1, admission T3, and same_as_previous T3/T4 against the table they replaced - production's
    regeneration test passed them); G17 on a pilot-era cell; --dry-run on the real cell 18h_2025-04-11."""
    from exopipe.phase3.core import validate
    from exopipe.phase3.lib.curves import workers as W
    CFG = _cfg()
    cell = '18h_2025-04-11'
    report = []
    n_old = n_new = 0
    for c, sb in (('18h_2025-04-11', '18MHz'), ('09h_2024-12-24', '36MHz'), ('02h_2024-12-28', '18MHz'),
                  ('02h_2024-12-28', '82MHz')):
        for st in ('I', 'V'):
            name = f'{c}_{sb}_{st}_lightcurves.fits'
            p = _real_table(c, name, new=False)
            if p:                       # pre-contract parents: rows MUST fail (every catalogue was rebuilt, a670970);
                n_old += 1              # V written before 98595b3 (M30) also fails flux_scale; nothing else may fail
                res = validate.run('light_curve', p)
                fails = sorted(r['name'] for r in res if not r['passed'])
                want = ['light_curve.rows'] + (['light_curve.flux_scale'] if st == 'V' else [])
                assert fails == sorted(want), (p, [(r['name'], r['detail']) for r in res if not r['passed']])
                report.append((f'{name} (pre-contract parents)', fails))
            q = _real_table(c, name, new=True)
            if q:                       # contract parents: everything passes, alone and with the step's context
                n_new += 1
                res = validate.run('light_curve', q)
                assert all(r['passed'] for r in res), (q, [(r['name'], r['detail']) for r in res if not r['passed']])
                fa = W.table_facts(q, st, None, os.path.join(CFG.paths.SNAPSHOT_FLEET_OUT, c, f'admission_{sb}.csv'),
                                   CFG.paths.CLV15_ROOT, CFG.curves.SAMPLES, cell=c, sb=sb)
                assert fa['naxis'] and fa['naxis'][0] in (1508, 2358, 3122) and fa['frame_header'], fa   # .fs and .fz
                assert fa['unstaged_frames'] == 0 and fa['mixed_grid_frames'] == 0, fa
                ctx = {'naxis': fa['naxis'], 'frame_header': fa['frame_header'], 'require_mcatsha': True,
                       'admission': os.path.join(CFG.paths.SNAPSHOT_FLEET_OUT, c, f'admission_{sb}.csv')}
                if p:
                    ctx['previous'] = p
                res = {r['name']: r for r in validate.run('light_curve', q, **ctx)}
                assert all(r['passed'] for r in res.values()), [(r['name'], r['detail']) for r in res.values()
                                                                 if not r['passed']]
                assert 'catalogue Gaussians inside the first frame' in res['light_curve.rows']['detail']
                assert not p or res['light_curve.same_as_previous']['value'] > 0, res['light_curve.same_as_previous']
                report.append((f'{name} (contract parents)', []))
    assert n_old and n_new, (n_old, n_new)
    # the tables the 2026-11-01 regeneration extracted with --search-frames (clv15 55469ba): 12h 2024-12-28 27 MHz - the
    # re-flowed search dropped 14 Stokes I frames that did not flow; with ctx search_frames they equal the search's list
    for st, n in (('I', 207), ('V', 221)):
        p = os.path.join(REAL_LC, '12h_2024-12-28', f'12h_2024-12-28_27MHz_{st}_lightcurves.fits')
        if not os.path.exists(p):
            continue
        res = {r['name']: r for r in validate.run('light_curve', p, search_frames=True)}
        assert all(r['passed'] for r in res.values()), [(r['name'], r['detail']) for r in res.values() if not r['passed']]
        assert res['light_curve.search_frames']['value'] == n and res['light_curve.integrations']['value'] == n, res
        report.append((f'{os.path.basename(p)} (--search-frames, 55469ba)', []))
    print('    real tables: ' + '; '.join(f'{n}: {"all pass" if not f else "fails " + ",".join(f)}' for n, f in report))
    # G17: a pilot-era cell's archive frames are 4096 px, its Masters 1508/2358/3122 px - the fleet regrids them in the staging
    why = W.archive_grid_problem('01h_2024-12-24', '18MHz')
    assert why and why.startswith('G17') and '4096 px' in why, why
    assert W.archive_grid_problem(cell, '18MHz') is None
    # --dry-run: the real cell plans 15 subbands x 2 Stokes; the missing stage refuses (exit 2); nothing written
    out = _fresh('dry')
    rc, so, se = _cli(['curves', '--cells', cell, '--stage-root', os.path.join(out, 'nostage'), '--out', out,
                       '--work', os.path.join(out, 'w'), '--dry-run', '--cores', 'none'])
    assert rc == 2 and so.count(' planned ') == 15 and 'outputs: 30 light_curve tables' in so, (rc, so[-2000:], se)
    assert 'G1 search finished: ok' in so and 'REFUSE G4' in so
    assert os.listdir(out) == [], os.listdir(out)
    # --dry-run with archive frames (default FRAME_SOURCE): no stage to check, no G4 refusal
    rc, so, se = _cli(['curves', '--cells', cell, '--out', out, '--work', os.path.join(out, 'w'), '--dry-run',
                       '--cores', 'none'])
    assert rc == 0 and 'frames: the Phase-1 archive' in so and 'REFUSE G4' not in so and 'G17' not in so, \
        (rc, so[-1500:], se[-500:])
    assert os.listdir(out) == [], os.listdir(out)
    # --dry-run of a pilot-era cell with archive frames: every planned subband G17 -> the unit would be refused (exit 2)
    rc, so, se = _cli(['curves', '--cells', '01h_2024-12-24', '--out', out, '--work', os.path.join(out, 'w'), '--dry-run',
                       '--cores', 'none'])
    n_planned = so.count(' planned ')
    assert rc == 2 and n_planned and so.count('pairs refused - G17') == n_planned, (rc, n_planned, so[-1500:])
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


def test_search_frames():
    """curves.SEARCH_FRAMES (the opt-in of clv15 55469ba, the 2026-11-01 regeneration; default False = production's fleet
    call): the REAL extractor in place. Default: every discovered frame, light_curve.search_frames not applied. --supersede
    with the opt-in: the table now holds the search's own list (one frame fewer) - same_as_previous compares the frames both
    hold and the pair records the counts (2026-11-02 review item 4). A night-refused pair under the opt-in is extracted with
    the plain call (B-70: all frames, none admitted; not compared); a runinfo without a list is refused before extraction
    (G18, its V G10); a listed frame that discovery does not give is the extractor's own refusal (G18), never an error; the
    dry run shows the same plan as the run."""
    from astropy.io import fits
    root = _fresh('searchframes')
    R = make_cell(root, subbands=('36MHz', '41MHz'), staged=('36MHz', '41MHz'), catalog_for=('36MHz', '41MHz'),
                  night_refused=('41MHz',))
    base = ['curves', '--cells', CELL, '--stage-root', R['stage'], '--out', R['out'], '--work', R['work'],
            '--cores', 'none'] + _sets(R)
    recp = os.path.join(R['out'], '_records', 'curves', f'{CELL}.json')

    def ts_of(p):
        with fits.open(p) as h:
            return [str(x).strip() for x in h['FRAMES'].data['ts']], int(h[0].header['NADMIT'])
    # (1) default: production's plain call everywhere - all 12 frames; the night-refused subband admits none (B-70)
    with _step_env(R):
        rc, out, err = _cli(base)
    rec = json.load(open(recp))
    assert rc == 0 and rec['status'] == 'done' and not rec['refused_items'], (rc, rec['reason'], rec['refused_items'])
    assert not any('--search-frames' in c['argv'] for c in rec['meta']['legacy_calls'])
    for sb, nadm in (('36MHz', len(TS) - 1), ('41MHz', 0)):
        got, n = ts_of(os.path.join(R['out'], CELL, f'{CELL}_{sb}_I_lightcurves.fits'))
        assert got == TS and n == nadm, (sb, len(got), n)
    # (2) --supersede with the opt-in: 36 MHz now holds the search's 11 frames (the dropped one is not sampled) and passes
    # search_frames; the pair records the frame counts; 41 MHz (night-refused, no list) keeps the plain call: unchanged
    time.sleep(1.1)
    with _step_env(R):
        rc, out, err = _cli(base + ['--supersede', 'sf1', '--set', 'curves.SEARCH_FRAMES=True'])
    rec = json.load(open(recp))
    assert rc == 0 and rec['status'] == 'done', (rc, rec['reason'], err[-2000:])
    calls = {(c['argv'][c['argv'].index('--subband') + 1], c['argv'][c['argv'].index('--stokes') + 1]):
             '--search-frames' in c['argv'] for c in rec['meta']['legacy_calls']}
    assert calls == {('36MHz', 'I'): True, ('36MHz', 'V'): True, ('41MHz', 'I'): False, ('41MHz', 'V'): False}, calls
    for st in ('I', 'V'):
        got, _n = ts_of(os.path.join(R['out'], CELL, f'{CELL}_36MHz_{st}_lightcurves.fits'))
        assert got == SEARCH_TS and SEARCH_DROPPED not in got, (st, got)
        assert ts_of(os.path.join(R['out'], CELL, f'{CELL}_41MHz_{st}_lightcurves.fits'))[0] == TS
    p36 = rec['meta']['pairs']['36MHz_I']
    assert p36['search_frames'] is True and p36['discovered_not_sampled'] == 1 and p36['superseded_frames'] == {
        'previous': 12, 'new': 11, 'common': 11, 'dropped': 1, 'added': 0}, p36
    assert rec['meta']['pairs']['41MHz_I']['search_frames'] is False, rec['meta']['pairs']['41MHz_I']
    sap = {t['path'].split('_')[-3]: t for t in rec['tests'] if t['name'] == 'light_curve.same_as_previous'
           and t['path'].endswith('_I_lightcurves.fits')}
    assert sap['36MHz']['passed'] and 'frames previous 12, new 11, both 11 (dropped 1, added 0)' in sap['36MHz']['detail'], sap
    sf = {t['path']: t for t in rec['tests'] if t['name'] == 'light_curve.search_frames'}
    assert all(t['passed'] for t in sf.values()), sf
    assert {p_.split('_')[-3]: t['value'] for p_, t in sf.items() if p_.endswith('_I_lightcurves.fits')} == {
        '36MHz': len(SEARCH_TS), '41MHz': 'n/a'}, sf
    # (3) the opt-in, a runinfo without its frame list: refused before extraction (G18; V G10); the dry run shows the same
    root2 = _fresh('searchframes_nolist')
    R2 = make_cell(root2, subbands=('36MHz', '41MHz'), staged=('36MHz', '41MHz'), catalog_for=('36MHz',))
    os.remove(os.path.join(R2['search'], CELL, 'noise_36MHz_I_median.csv'))      # the I list only: I G18, its V G10
    base2 = ['curves', '--cells', CELL, '--stage-root', R2['stage'], '--out', R2['out'], '--work', R2['work'], '--cores',
             'none', '--set', 'curves.SEARCH_FRAMES=True'] + _sets(R2)
    with _step_env(R2):
        rc, out, err = _cli(base2 + ['--dry-run'])
    # 41MHz has no Master catalogue (G6, before the frame list): only G6 for it - as the run records - never G18
    assert rc == 2 and '36MHz I pair refused - G18' in out and '36MHz V pair refused - G10' in out and '41MHz I pair' not in \
        out and 'MISSING (G6' in out, (rc, out[-2500:])
    # the dry run still checks the stage with the opt-in (the plan notes never replace the stage check)
    with _step_env(R2):
        rc, out, err = _cli([a if a != R2['stage'] else os.path.join(root2, 'nostage') for a in base2] + ['--dry-run'])
    assert rc == 2 and 'REFUSE G4: staged cell' in out and 'does not exist' in out, (rc, out[-1500:])
    with _step_env(R2):
        rc, out, err = _cli(base2)
    rec = json.load(open(os.path.join(R2['out'], '_records', 'curves', f'{CELL}.json')))
    assert rc == 2 and rec['status'] == 'refused' and not rec['meta'].get('legacy_calls'), (rc, rec['status'])
    assert sorted((os.path.basename(r['path']).split('_')[2], os.path.basename(r['path']).split('_')[3], r['gate'])
                  for r in rec['refused_items']) == [('36MHz', 'I', 'G18'), ('36MHz', 'V', 'G10'), ('41MHz', 'I', 'G6'),
                                                     ('41MHz', 'V', 'G6')], rec['refused_items']
    # (4) the opt-in, a listed frame that discovery does not give: the extractor refuses (clv15 55469ba) - G18, its V G10
    root3 = _fresh('searchframes_g18')
    R3 = make_cell(root3, search_ts=SEARCH_TS + ['20300101_120200'])
    with _step_env(R3):
        rc, out, err = _cli(['curves', '--cells', CELL, '--stage-root', R3['stage'], '--out', R3['out'], '--work',
                             R3['work'], '--cores', 'none', '--set', 'curves.SEARCH_FRAMES=True'] + _sets(R3))
    rec = json.load(open(os.path.join(R3['out'], '_records', 'curves', f'{CELL}.json')))
    assert rc == 2 and rec['status'] == 'refused', (rc, rec['status'], rec['reason'])
    assert {r['gate'] for r in rec['refused_items']} == {'G18', 'G10'}
    assert "not discovered" in rec['meta']['pairs'][f'{SB}_I']['reason'], rec['meta']['pairs']


TESTS = [test_items, test_config, test_stages, test_plan, test_publish, test_lock_and_classify, test_end2end,
         test_failures, test_negative, test_archive, test_search_frames, test_real_products]


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
