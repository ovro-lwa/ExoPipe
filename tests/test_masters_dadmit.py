"""D-ADMIT (a), slice 1: the build's recorded statistics (lib/masters/build_stats.py, coadd_final/build_stats.json).

Each statistic must be the number its gate computes: stage 2 as coadd.detect_outliers_mad (and equal to the threshold the
build itself recorded), the beam gate's and the sanity gates' upper middle element, the search tier's numpy median with
1.4826 x MAD; the content test re-derives every number and fails on an edited one. Real cells (read only): the
re-derived stage-2 thresholds equal production's recorded ones.
Run: python tests/test_masters_dadmit.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.core import env as ENV                          # noqa: E402
ENV.legacy()
from exopipe.phase3.config import CFG                               # noqa: E402
from exopipe.phase3.lib.masters import build_stats as BS            # noqa: E402
from exopipe.phase3.lib.masters import coadd as co                  # noqa: E402

FAILS, N = [], [0]


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def test_formulas():
    rng = np.random.default_rng(15)
    vals = list(rng.normal(100, 10, 9)) + [float('nan'), 400.0]
    dates = [f'2025-01-{i + 1:02d}' for i in range(len(vals))]
    rows = [dict(i_rms_mjy=v, v_rms_mjy=v / 4) for v in vals]
    st = BS.stage2_stats(rows, float(co.OUTLIER_SIGMA), float(co.LOW_OUTLIER_SIGMA))
    _out, thr = co.detect_outliers_mad(dates, vals)
    check('stage 2: the high threshold is coadd.detect_outliers_mad\'s (finite values only)',
          st['I']['gate'] and abs(st['I']['thr_hi'] - thr) < 1e-9 and st['I']['n'] == 10, (st['I'], thr))
    few = BS.stage2_stats(rows[:4], 3.0, 4.0)
    check('stage 2: fewer than 5 finite values - no gate', few['I'] == {'n': 4, 'gate': False})
    areas = {'a': 1.0, 'b': 4.0, 'c': 2.0, 'd': 3.0, 'e': 5.0, 'f': 6.0}
    bs = BS.beam_stats(areas, 1.5, 5)
    _bfin = sorted(areas.values())
    check('beam gate: the upper middle element of the sorted areas, as coadd.coadd_beam_gate takes it',
          bs['median_area'] == _bfin[len(_bfin) // 2] == 4.0 and BS.beam_stats(dict(list(areas.items())[:4]), 1.5, 5)['gate'] is False, bs)
    w = [dict(qa3_rms_mjy=x) for x in ('10', '20', '30', '40', '', 'nan', '-5', '50', '60')]
    ss = BS.sanity_stats(w, 3.0, 5)
    check('sanity: the upper middle element of the finite positive qa3 values', ss['qa3_median'] == 40.0 and ss['n'] == 6, ss)
    prov = [dict(i_rms_mjy=x, v_rms_mjy=x) for x in (1.0, 2.0, 3.0, 4.0, 10.0, 6.0)]
    tt = BS.search_tier_stats(prov, 5.0, 5)['I']
    v = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 10.0])
    med, mad = float(np.median(v)), float(np.median(np.abs(v - np.median(v)))) * 1.4826
    check('search tier: numpy median and 1.4826 x MAD (not the upper middle element)',
          abs(tt['median'] - med) < 1e-12 and abs(tt['thr'] - (med + 5.0 * mad)) < 1e-12 and med == 3.5, (tt, med, mad))


def test_real():
    for cell in ('23h_23MHz', '11h_Red'):
        lst, sc = cell.split('_')
        st = BS.compute(f'/data07/gh/phase3/coadd_sb/{cell}', lst, sc, co, CFG)
        rec = st['stage2_recorded_thr']
        same = all(len(rec[s]) == 1 and abs(rec[s][0] - st['stage2'][s]['thr_hi']) < 1e-5 for s in ('I', 'V'))
        check(f'real {cell}: the re-derived stage-2 thresholds equal the ones the build recorded', same, (rec, st['stage2']))
        d = tempfile.mkdtemp(prefix='bs_')
        try:
            os.makedirs(os.path.join(d, 'coadd_final'))
            p = BS.write(d, st)
            from exopipe.phase3.core import validate
            res = validate.run('build_stats', p)
            ok = all(t['passed'] for t in res)
            s2 = json.load(open(p))
            s2['search_tier']['V']['thr'] *= 1.01
            json.dump(s2, open(p, 'w'))
            res2 = validate.run('build_stats', p)
            s3 = json.load(open(p))
            s3['search_tier']['V']['thr'] /= 1.01
            k = next(iter(s3['beam_gate']['nights']))
            s3['beam_gate']['nights'][k]['area'] *= 2
            json.dump(s3, open(p, 'w'))
            res3 = validate.run('build_stats', p)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        check(f'real {cell}: the content test passes on the record and fails on an edited threshold or beam area',
              ok and not all(t['passed'] for t in res2) and not all(t['passed'] for t in res3)
              and 'search_tier.V.thr' in res2[0]['detail'] and 'area' in res3[0]['detail'], (res, res2, res3))


def test_wiring():
    from exopipe.phase3.lib.masters import deep_stages as DS
    import inspect
    src = inspect.getsource(DS.build)
    check('the build writes the record only under masters_build.WRITE_BUILD_STATS, after the co-add, before the outputs '
          'are tested', 0 < src.index('if _cfg().masters_build.WRITE_BUILD_STATS:') < src.index('for p, cls in _build_outputs(')
          and 'coadd_final/build_stats.json' in DS.BUILD_PRODUCT_PATHS)
    prod = open(os.path.join(REPO, 'exopipe/phase3/parity/production_20261013.cfg')).read()
    check('production values write no record (parity file)', 'masters_build.WRITE_BUILD_STATS = False' in prod
          and CFG.masters_build.WRITE_BUILD_STATS is True)


def main():
    test_formulas()
    test_real()
    test_wiring()
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
