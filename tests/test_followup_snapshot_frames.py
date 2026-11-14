#!/usr/bin/env python3
"""STANDING TEST - the follow-up 10-s light curve (T5, followup/snapshots.py) reads the search's frames and honors exclusion.

The snapshot chat's judgment of 2026-11-14 (JUDGED-20261114-FOLLOWUP-SNAPSHOT-CUTOUTS-SURVEY-ROOT-ONLY): production's T5 read the
survey's Phase-1 tree only and no exclusion, so on the GPU-frame cells (Dec 18-27 2024) it read frames other than the search's,
including subband-nights excluded for the snapshot tiers. Asserts, on synthetic frame trees and a synthetic exclusion list:
  (i)   a GPU-frame cell's admitted subband reads the GPU pipeline's tree, never the survey's;
  (ii)  a subband-night the snapshot tiers re-admit for GPU frames only ('[SNAPSHOT-GPU-READMITTED') is read there;
  (iii) an excluded subband-night is refused (snap_refusal names it; no directory, no frame, has_frames False), on a GPU-frame
        cell and on a survey cell alike;
  (iv)  a GPU-frame cell's subband without GPU frames has no directory: no fallback to the survey tree;
  (v)   a survey cell's admitted subband reads the survey tree;
  (vi)  the light curve and the morphology of a refused subband-night return the refusal, read nothing;
  (vii) the warp screen is applied only to a frame on its own grid (the snapshot chat's review of PR 70): a survey-size frame
        (another shape and pixel scale), a frame of the screen's shape on another pixel scale or reference pixel, or a screen
        without a grid leave the frame as read; a frame on the screen's grid is warped;
  (viii) the light curve and the spectrum record what was done: on the GPU-frame cell (frames on the screen's grid)
        dewarped True and every frame counted; on the survey cell (4096-like frames) dewarped False, none counted.

Scratch: a fresh directory under $EXOPIPE3_TEST_SCRATCH, else the system temp directory, removed at exit. Data-free (CI).
Run standalone:  python tests/test_followup_snapshot_frames.py
"""
import atexit
import os
import shutil
import sys
import tempfile
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

for _m in ('bdsf', 'h5py'):    # the data-free runner: phase2.cascade_dewarp and extractor_pb_75 import them at load;
    try:                       # apply_warp uses neither
        __import__(_m)
    except ImportError:
        sys.modules[_m] = types.ModuleType(_m)
from exopipe.phase3.config import CFG                               # noqa: E402
from exopipe.phase3.core import exclusions as EX                    # noqa: E402

FAILS, N = [], [0]
GPU_CELL, SURVEY_CELL = ('01h', '2024-12-18'), ('02h', '2025-01-05')
TS = {'2024-12-18': '20241218_030422', '2025-01-05': '20250105_041010'}
HEADER = 'lst,subband,date,verdict,reason,score_thermal_noise,score_flux_scale,score_ionosphere\n'
ROWS = ['01h,18MHz,2024-12-18,exclude,[SNAPSHOT-GPU-READMITTED] synthetic: survey frames bad; GPU frames good,,,',
        '01h,23MHz,2024-12-18,exclude,synthetic: excluded in every scope,,,',
        '02h,18MHz,2025-01-05,exclude,synthetic: excluded,,,']


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def _frames(root, lst, date, sbs):
    for sb in sbs:
        for sub, kind in (('snapshots', 'Pilot'), ('snapshots_clean', 'Clean')):
            d = os.path.join(root, lst, date, 'Run_x', sb, sub)
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, f'{sb}-{kind}-Snapshot-{TS[date]}-image.fits.fz'), 'wb').close()


def setup():
    base = os.environ.get('EXOPIPE3_TEST_SCRATCH') or tempfile.gettempdir()
    os.makedirs(base, exist_ok=True)
    T = tempfile.mkdtemp(prefix='followup_frames_', dir=base)
    atexit.register(shutil.rmtree, T, True)
    p1, gpu = os.path.join(T, 'phase1'), os.path.join(T, 'gpu')
    _frames(p1, *GPU_CELL, ('18MHz', '23MHz', '27MHz', '36MHz'))      # the survey tree holds 36 MHz, the GPU tree does not
    _frames(gpu, *GPU_CELL, ('18MHz', '23MHz', '27MHz'))
    _frames(p1, *SURVEY_CELL, ('18MHz', '27MHz'))
    dd = os.path.join(T, 'data_decisions.csv')
    open(dd, 'w').write(HEADER + '\n'.join(ROWS) + '\n')
    cells = os.path.join(T, 'gpu_cells.txt')
    open(cells, 'w').write('# synthetic\n' + ' '.join(GPU_CELL) + '\n')
    CFG.paths.PHASE1_ROOT, CFG.paths.SNAPSHOT_GPU_FRAME_ROOT = p1, gpu
    CFG.paths.DATA_DECISIONS, CFG.paths.SNAPSHOT_GPU_FRAME_CELLS = dd, cells
    for v in ('P3_SNAP_GPU_CELLS', 'P3_SNAP_PHASE1_ROOT'):
        os.environ.pop(v, None)
    EX._CACHE.clear()
    return p1, gpu


def _hdr(cdelt, crpix):
    """A SIN grid of ``cdelt`` deg per pixel and reference pixel ``crpix``."""
    from astropy.io import fits
    h = fits.Header()
    for k, v in (('CTYPE1', 'RA---SIN'), ('CTYPE2', 'DEC--SIN'), ('CDELT1', -cdelt), ('CDELT2', cdelt), ('CRPIX1', crpix),
                 ('CRPIX2', crpix), ('CRVAL1', 187.5), ('CRVAL2', 37.2)):
        h[k] = v
    return h


def _img(path, n, cdelt, crpix):
    """A 2-D frame of n x n pixels on that grid."""
    from astropy.io import fits
    fits.writeto(path, np.ones((n, n), np.float32), _hdr(cdelt, crpix), overwrite=True)


def _screen(path, n, cdelt, crpix, wcs=True):
    """A warp screen of n x n pixels (zero shifts), its SCREEN_X / SCREEN_Y on that grid (no WCS: ``wcs`` False)."""
    from astropy.io import fits
    h = _hdr(cdelt, crpix) if wcs else fits.Header()
    fits.HDUList([fits.PrimaryHDU()] + [fits.ImageHDU(np.zeros((n, n)), h, name=nm) for nm in ('SCREEN_X', 'SCREEN_Y')]
                 ).writeto(path, overwrite=True)
    return path


def warp_checks(T, S):
    """(vii) and (viii)."""
    from exopipe.phase3.followup import imaging as I
    from exopipe.phase3.followup import photometry as PHOT
    from exopipe.phase3.followup import resolve as R
    GN, GD, GP = 40, 0.049, 20.0                        # the screen's grid (a 2358 x 2358, 0.049 deg, CRPIX 1180 stand-in)
    scr_path = _screen(os.path.join(T, 'screen.fits'), GN, GD, GP)
    R_ws, cf, fi, tf = R.warp_screen, S._copy_funpack, PHOT.forced_imfit, S.tempfile
    R.warp_screen = lambda lst, date, scope: scr_path
    scr = S._load_warp('27MHz', '01h', '2024-12-18')
    cases = {'survey-size frame (64 px at 0.03125 deg)': (64, 0.03125, 32.0), 'the screen\'s grid': (GN, GD, GP),
             'the screen\'s shape, another pixel scale': (GN, 0.05, GP), 'the screen\'s shape and scale, another CRPIX': (GN, GD, 21.0)}
    got = {}
    for label, (n, cd, cp) in cases.items():
        fp = os.path.join(T, 'f.fits')
        _img(fp, n, cd, cp)
        data, hdr, wcs = I.load_image(fp)
        got[label] = S._warp(data, wcs, scr)[1]
    check('(vii) the screen is applied only on its own grid (shape, CDELT, CRPIX)',
          got == {k: k == 'the screen\'s grid' for k in cases}, got)
    R.warp_screen = lambda lst, date, scope: _screen(os.path.join(T, 'screen_nowcs.fits'), GN, GD, GP, wcs=False)
    sng = S._load_warp('27MHz', '01h', '2024-12-18')
    _img(os.path.join(T, 'f.fits'), GN, GD, GP)
    data, hdr, wcs = I.load_image(os.path.join(T, 'f.fits'))
    check('(vii) a screen without a grid is not applied', sng[2] is None and S._warp(data, wcs, sng)[1] is False, sng[2])
    R.warp_screen = lambda lst, date, scope: scr_path

    def fake_funpack(src, dst, retries=3):              # the frame as its tree holds it: GPU on the screen's grid
        fp = dst[:-3] if dst.endswith('.fz') else dst + '.fits'
        n, cd, cp = (GN, GD, GP) if '/gpu/' in src else (64, 0.03125, 32.0)
        _img(fp, n, cd, cp)
        return fp
    S._copy_funpack = fake_funpack
    PHOT.forced_imfit = lambda *a, **k: {'peak': 1.0, 'snr': 5.0, 'sign': 1, 'err': 0.2}
    S.tempfile = types.SimpleNamespace(mkdtemp=lambda prefix='', dir=None: tempfile.mkdtemp(prefix=prefix, dir=T))  # not /fast
    try:
        res = {}
        for (lst, date), iso in ((GPU_CELL, ('2024-12-18T03:04:00', '2024-12-18T03:06:00')),
                                 (SURVEY_CELL, ('2025-01-05T04:09:30', '2025-01-05T04:11:00'))):
            lc = S.snapshot_lightcurve(lst, date, '27MHz', 10.0, 40.0, *iso, verbose=False)
            sp = S.snapshot_spectrum(lst, date, 10.0, 40.0, TS[date], scopes=['27MHz'], verbose=False)
            res[lst] = (lc.get('dewarped'), lc.get('n_frames_dewarped'), lc.get('n_frames_read'), sp[0].get('dewarped'))
        check('(viii) GPU-frame cell: the light curve and the spectrum row are dewarped, every frame counted',
              res[GPU_CELL[0]] == (True, 1, 1, True), res)
        check('(viii) survey cell: not dewarped (the frames are off the screen\'s grid), none counted',
              res[SURVEY_CELL[0]] == (False, 0, 1, False), res)
    finally:
        R.warp_screen, S._copy_funpack, PHOT.forced_imfit, S.tempfile = R_ws, cf, fi, tf


def main():
    p1, gpu = setup()
    from exopipe.phase3.followup import snapshots as S
    lst, date = GPU_CELL
    d = S._phase1_snap_dir(lst, date, '27MHz')
    check('(i) a GPU-frame cell reads the GPU pipeline\'s tree', bool(d) and d.startswith(gpu + os.sep), d)
    f = S._frame_path(lst, date, '27MHz', 'I', TS[date])
    check('(i) its Stokes I frame is the GPU tree\'s snapshots_clean/', bool(f) and f.startswith(gpu + os.sep)
          and '/snapshots_clean/' in f, f)
    d = S._phase1_snap_dir(lst, date, '18MHz')
    check('(ii) a subband re-admitted for GPU frames only is read from the GPU tree',
          S.snap_refusal(lst, date, '18MHz') is None and bool(d) and d.startswith(gpu + os.sep), d)
    why = S.snap_refusal(lst, date, '23MHz')
    check('(iii) an excluded subband-night of a GPU-frame cell is refused (scope snapshot-gpu)',
          bool(why) and 'snapshot-gpu' in why and S._phase1_snap_dir(lst, date, '23MHz') is None
          and S._frame_path(lst, date, '23MHz', 'V', TS[date]) is None and not S.has_frames(lst, date, '23MHz'), why)
    check('(iv) no GPU frames: no directory, no fallback to the survey tree',
          S._phase1_snap_dir(lst, date, '36MHz') is None and not S.has_frames(lst, date, '36MHz'))
    slst, sdate = SURVEY_CELL
    why = S.snap_refusal(slst, sdate, '18MHz')
    check('(iii) an excluded subband-night of a survey cell is refused (scope snapshot)',
          bool(why) and 'scope snapshot)' in why and S._phase1_snap_dir(slst, sdate, '18MHz') is None, why)
    d = S._phase1_snap_dir(slst, sdate, '27MHz')
    check('(v) a survey cell\'s admitted subband reads the survey tree', bool(d) and d.startswith(p1 + os.sep), d)
    lc = S.snapshot_lightcurve(lst, date, '23MHz', 10.0, 40.0, '2024-12-18T03:04:00', '2024-12-18T03:06:00', verbose=False)
    check('(vi) the light curve of a refused subband-night returns the refusal and no frame',
          lc.get('refused') and lc.get('available') is False and not lc.get('frames'), lc)
    mo = S.snapshot_morphology(slst, sdate, '18MHz', 10.0, 40.0, TS[sdate], verbose=False)
    check('(vi) the morphology of a refused subband-night returns the refusal', bool(mo.get('refused')), mo)
    warp_checks(os.path.dirname(p1), S)
    print(f'\n{N[0] - len(FAILS)} of {N[0]} checks passed' + (f'; FAILED: {FAILS}' if FAILS else ''))
    return 1 if FAILS else 0


def test_followup_snapshot_frames():
    assert main() == 0


if __name__ == '__main__':
    sys.exit(main())
