"""D-ADMIT: the final stage records its second-pass DCs (coadd_final/scale2.csv, masters_build.SCALE2_RECORD) and an admitted
night's degeneracy guard takes the build's own median from it (admit_stages.guard_median; plan/wave2/DADMIT_DESIGN.md "Next:
the build records its second-pass DCs").

The writer (coadd._write_scale2_record) on synthetic PASS-1 results: every night's DC and source count, which entered the
median (n >= 8, the co-add's rule) and the median the guard used - 1.0 over none, as the co-add; the record's own checks
(admit.scale2_problems, the class scale2_record content test): a changed median, a night of another build, a broken file;
guard_median: the record's median when there is one, the ruled value for a cell built without it, a refusal for a record
that fails its checks; the switch: the step sets the co-add's SCALE2_RECORD from config ('record' ruled, 'none' in the
parity values), the insertion sits between PASS 1's median and PASS 2 (the extracted loops' pins in test_masters_admit4
unchanged), install carries the record with the cell's other build records.
Run: python tests/test_masters_scale2.py
"""
import csv
import inspect
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.core import env as ENV                          # noqa: E402
ENV.legacy()
from exopipe.phase3.config import CFG                               # noqa: E402
from exopipe.phase3.core.record import Refusal                      # noqa: E402
from exopipe.phase3.lib.masters import admit as A                   # noqa: E402

T = '/tmp/exopipe3_tests_scale2'
FAILS, N = [], [0]


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def cell(root, nights):
    cd = os.path.join(root, '05h_78MHz')
    os.makedirs(os.path.join(cd, 'coadd_final'), exist_ok=True)
    with open(os.path.join(cd, 'coadd_final', 'weights.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'in_master'])
        for d in nights:
            w.writerow([d, 1])
    return cd


def test_writer_and_checks():
    from exopipe.phase3.lib.masters import coadd as co
    from exopipe.phase3.lib.masters import tests as _MT       # noqa: F401 (registers the content tests)
    from exopipe.phase3.core import validate
    nights = ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04']
    cd = cell(os.path.join(T, 'a'), nights)
    meas = {nights[0]: (np.array([0.99, 0, 0]), 30), nights[1]: (np.array([1.02, 0, 0]), 12),
            nights[2]: (np.array([1.50, 0, 0]), 5), nights[3]: (np.array([1.005, 0, 0]), 8)}
    _dcs = [float(c[0][0]) for c in meas.values() if c[1] >= 8]          # the co-add's own lines (run_final_lst_freq)
    med_dc = float(np.median(_dcs)) if _dcs else 1.0
    p = co._write_scale2_record(os.path.join(cd, 'coadd_final'), meas, med_dc)
    rows = list(csv.DictReader(open(p, newline='')))
    probs, med, n = A.scale2_problems(cd)
    check('the record: every night\'s DC and count, in_median by the co-add\'s n >= 8, the median row the guard\'s median',
          [r['date'] for r in rows] == nights + ['median'] and [r['in_median'] for r in rows[:4]] == ['1', '1', '0', '1']
          and not probs and med == med_dc and n == 3, (rows, probs, med, n))
    res = validate.run('scale2_record', p)
    check('its content test passes', res and all(t['passed'] for t in res), res)
    cd0 = cell(os.path.join(T, 'b'), nights[:1])
    co._write_scale2_record(os.path.join(cd0, 'coadd_final'), {nights[0]: (np.array([0.8, 0, 0]), 3)}, 1.0)
    probs0, med0, n0 = A.scale2_problems(cd0)
    check('no night with 8 sources: the median row is the co-add\'s 1.0 over 0 nights, and that is consistent',
          not probs0 and med0 == 1.0 and n0 == 0, (probs0, med0, n0))
    keep = open(p, newline='').read()
    lines = keep.splitlines(True)
    lines[-1] = lines[-1].replace(repr(med_dc), repr(med_dc + 0.01))     # the median row only
    open(p, 'w', newline='').write(''.join(lines))
    bad1 = A.scale2_problems(cd)[0]
    open(p, 'w', newline='').write(keep)
    with open(os.path.join(cd, 'coadd_final', 'weights.csv'), 'w', newline='') as f:
        csv.writer(f).writerows([['date', 'in_master']] + [[d, 1] for d in nights[1:]])
    bad2 = A.scale2_problems(cd)[0]
    res2 = validate.run('scale2_record', p)
    check('a changed median, a night not of this build\'s weights.csv (a record of another build): each a problem, the '
          'content test fails', bad1 and 'median' in bad1[0] and bad2 and 'another build' in bad2[0]
          and not all(t['passed'] for t in res2), (bad1, bad2, res2))
    return cd, cd0


def test_guard_median(cd0):
    from exopipe.phase3.lib.masters import admit_stages as AS
    got = AS.guard_median(cd0)
    cd_none = cell(os.path.join(T, 'c'), ['2025-01-01'])
    saved = CFG.masters_admit.DC2_MEDIAN_WITHOUT_RECORD
    try:
        CFG.masters_admit.DC2_MEDIAN_WITHOUT_RECORD = None
        none = AS.guard_median(cd_none)
        CFG.masters_admit.DC2_MEDIAN_WITHOUT_RECORD = 1.0
        ruled = AS.guard_median(cd_none)
        rec = AS.guard_median(cd0)
    finally:
        CFG.masters_admit.DC2_MEDIAN_WITHOUT_RECORD = saved
    check('guard_median: the build\'s recorded median when the cell has the record (the ruled value is not read)',
          got[0] == 1.0 and got[1].startswith('recorded') and rec[1].startswith('recorded'), (got, rec))
    check('guard_median: without a record the ruled value; unset, none (the stage refuses)', none == (None, None)
          and ruled == (1.0, 'ruled: masters_admit.DC2_MEDIAN_WITHOUT_RECORD'), (none, ruled))
    p = os.path.join(cd0, 'coadd_final', A.SCALE2)
    open(p, 'a').write('2099-01-01,1.0,30,1\n')
    try:
        AS.guard_median(cd0)
        r = 'no refusal'
    except Refusal as e:
        r = str(e)
    check('guard_median: a record that fails its checks refuses (a stale record is never read)', 'second-pass DC record' in r,
          r)


def test_switch():
    from exopipe.phase3.lib.masters import coadd as co, deep_inplace as IP, deep_stages as DS
    from exopipe.phase3.lib.install import steps_deep as ISD
    src = inspect.getsource(co.run_final_lst_freq)
    i_med = src.find("med_dc = float(np.median(_dcs)) if _dcs else 1.0")
    i_rec = src.find("if SCALE2_RECORD == 'record':")
    i_p2 = src.find("subtract_night(date, sci_dir, align_img_path,")
    check('the record is written between PASS 1\'s median and PASS 2, under the switch', 0 < i_med < i_rec < i_p2,
          (i_med, i_rec, i_p2))
    check('the co-add\'s own default is production\'s (no record); the step wires masters_build.SCALE2_RECORD onto it',
          "\nSCALE2_RECORD = 'none'\n" in open(co.__file__).read()
          and IP.WIRING['masters_build'].get('SCALE2_RECORD') == 'attr:coadd.SCALE2_RECORD')
    out = os.path.join(T, 'inplace')
    os.makedirs(out, exist_ok=True)
    mods, _ = DS.inplace(out)
    ruled = mods['coadd'].SCALE2_RECORD
    par = open(os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg')).read()
    check("config 'record' (ruled) reaches the in-place co-add; the parity values set production's 'none'",
          CFG.masters_build.SCALE2_RECORD == 'record' and ruled == 'record'
          and "masters_build.SCALE2_RECORD = 'none'" in par, (CFG.masters_build.SCALE2_RECORD, ruled))
    check('install carries the record with the cell\'s other build records; a build in a cell already holding one is '
          'an earlier build', ISD.SCALE2 in ISD.CARRIED_RECORDS and 'coadd_final/scale2.csv' in DS.BUILD_PRODUCT_PATHS)
    keep = CFG.masters_build.SCALE2_RECORD
    for bad in ('Record', True, ''):
        CFG.masters_build.SCALE2_RECORD = bad
        try:
            DS.inplace(out)
            why = None
        except Refusal as e:
            why = str(e)
        finally:
            CFG.masters_build.SCALE2_RECORD = keep
        check(f'MINOR (ninth review): masters_build.SCALE2_RECORD = {bad!r} refuses before any build (the co-add tests only '
              f'for \'record\': a typo would build cells without the record)', why and 'SCALE2_RECORD' in why, why)
    DS.inplace(out)                                  # the in-place co-add set back to the configured value


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        _cd, cd0 = test_writer_and_checks()
        test_guard_median(cd0)
        test_switch()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
