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
  (vi)  the light curve and the morphology of a refused subband-night return the refusal, read nothing.

Scratch: a fresh directory under $EXOPIPE3_TEST_SCRATCH, else the system temp directory, removed at exit. Data-free (CI).
Run standalone:  python tests/test_followup_snapshot_frames.py
"""
import atexit
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

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
    print(f'\n{N[0] - len(FAILS)} of {N[0]} checks passed' + (f'; FAILED: {FAILS}' if FAILS else ''))
    return 1 if FAILS else 0


def test_followup_snapshot_frames():
    assert main() == 0


if __name__ == '__main__':
    sys.exit(main())
