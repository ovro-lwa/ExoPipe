"""D-ADMIT (a), slice 4 part 3 (plan/wave2/DADMIT_DESIGN.md "Slice 4 part 3"): the masters admit stage end to end, and
admission_reproduces.

On a copy of 05h_78MHz (built 2026-10-08 with today's flux rules - a cell built before them, such as 13h_78MHz of
2026-09-10, carries the old rule in its members: +0.27% and +0.31% at bright peaks there; no master dewarp: the members'
stored flows apply - no GPU) two member nights are taken out of
every record of the build (a consistent build without them) and admitted again: `masters --stages admit-prep`, the flow
step's admit stage with the stored flows copied in (fresh: nothing solved), then `masters --stages admit` with the guard
median DECISIONS_PENDING item 10 recommends (1.0, set for this test only - the config default refuses). The admitted
epoch must equal the member's installed epoch (its installed residual + the installed master) at the master's bright
compact sources: median |ratio - 1| <= REPRODUCE_TOL (1e-3, Gregg's tolerance), the measured values printed - where they
differ by design (the first DC read on another image; the guard median) the test reports by how much. The admitted
residual's identity (epoch - residual - master) holds; admissions.csv passes its content test; a second run admits
nothing (never twice); without a guard median the stage refuses.
Run: python tests/test_masters_admit6.py
"""
import calendar
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.core import env as ENV                          # noqa: E402
ENV.legacy()
from exopipe.phase3.config import CFG                               # noqa: E402
from exopipe.phase3.lib.masters import admit as A                   # noqa: E402

T = '/tmp/exopipe3_tests_admit6'
FAILS, N = [], [0]
CELL, LST, SC = '05h_78MHz', '05h', '78MHz'
REPRODUCE_TOL = 1e-3


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def copy_without(nights, root):
    """A copy of the installed cell's records, masters and provisional masters under root/<cell>/ in which ``nights`` -
    members - are out of every record of the build, the stage-2 record carries the thresholds that population gives, the
    population record is recomputed (and passes its test), and the build's records are dated one day before the earliest
    Science dir of those nights (admit.build_time: they qualify as nights Phase 2 first had after the build)."""
    from exopipe.phase3.lib.masters import build_stats as BS, coadd as co
    src, dst = os.path.join(CFG.paths.COADD_SB_ROOT, CELL), os.path.join(root, CELL)
    os.makedirs(os.path.join(dst, 'coadd_final'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'coadd_provisional'), exist_ok=True)
    for rel in [r for r, _c in A.BUILD_RECORDS] + ['coadd_final/stage2_verdict.csv', f'coadd_final/I_deep_Taper_R075_{SC}.fits',
                                                   f'coadd_final/V_deep_Taper_R0_{SC}.fits',
                                                   f'coadd_provisional/I_deep_Taper_R075_{SC}.fits',
                                                   f'coadd_provisional/V_deep_Taper_R0_{SC}.fits']:
        if os.path.exists(os.path.join(src, rel)):
            shutil.copy2(os.path.join(src, rel), os.path.join(dst, rel))
    for rel, col in list(A.BUILD_RECORDS) + [('coadd_final/stage2_verdict.csv', 'date')]:
        p = os.path.join(dst, rel)
        if not os.path.exists(p):
            continue
        rows = list(csv.reader(open(p, newline='')))
        i = rows[0].index(col) if col and rows and col in rows[0] else 2
        keep = [r for r in rows if not (len(r) > i and r[i].strip() in nights)]
        with open(p, 'w', newline='') as f:
            csv.writer(f).writerows(keep)
    s2p, rows2 = BS.stage2_source(dst)
    st2 = BS.stage2_stats(rows2, float(co.OUTLIER_SIGMA), float(co.LOW_OUTLIER_SIGMA))
    rows = list(csv.DictReader(open(s2p, newline='')))
    for r in rows:
        r['i_thresh_mjy'], r['v_thresh_mjy'] = repr(st2['I']['thr_hi']), repr(st2['V']['thr_hi'])
    with open(s2p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    BS.write(dst, BS.compute(dst, LST, SC, co, CFG, 'stage'))
    assert not BS.recheck(os.path.join(dst, 'coadd_final', BS.FILE), co)
    first = min(calendar.timegm(time.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S'))
                for d in nights for x in glob.glob(f'{co.PHASE2_ROOT}/{LST}/{d}/Science_*')
                for m in [re.match(A.SCIENCE_DIR, os.path.basename(x))] if m)
    for rel in (A.BUILD_NIGHTS, 'coadd_final/weights.csv'):
        os.utime(os.path.join(dst, rel), (first - 86400, first - 86400))
    return dst


def cli(step, *a, timeout=3600):
    p = subprocess.run([sys.executable, '-m', 'exopipe.phase3', step, '--tier', 'deep', '--cells', CELL, '--cores', 'none']
                       + list(a), cwd=REPO, env=dict(os.environ), capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def rec(out, step, unit):
    p = os.path.join(out, '_records', step, f'{unit}.json')
    return json.load(open(p)) if os.path.exists(p) else {}


def bright_ratio(e, m):
    """Median and count of e / m at the master's bright compact sources - the selection of coadd._epoch_content_arrays."""
    from scipy.ndimage import maximum_filter
    ok = np.isfinite(e) & np.isfinite(m) & (m != 0)
    v = m[ok]
    rr = 1.4826 * np.median(np.abs(v - np.median(v)))
    mf = np.nan_to_num(m)
    pk = (mf == maximum_filter(mf, size=9)) & (mf > 20 * rr) & (mf < 50)
    yy, xx = np.nonzero(pk)
    h, w = m.shape
    keep = np.hypot(xx - w / 2, yy - h / 2) / (w / 2) < 0.9
    yy, xx = yy[keep], xx[keep]
    r = e[yy, xx] / m[yy, xx]
    r = r[np.isfinite(r)]
    return float(np.median(r)), float(np.median(np.abs(r - 1))), len(r)


def test_admit_end_to_end():
    from astropy.io import fits
    from exopipe.phase3.core import validate
    w = list(csv.DictReader(open(os.path.join(CFG.paths.COADD_SB_ROOT, CELL, 'coadd_final', 'weights.csv'))))
    flows = os.path.join(CFG.paths.COADD_SB_ROOT, CELL, 'flow')
    cands = sorted(r['date'] for r in w if r.get('in_master') == '1' and r.get('qa3_guard') == 'pass'
                   and os.path.exists(os.path.join(flows, f'{r["date"]}_scaled_flow.npy')))
    nights = cands[-2:]
    wt, out = os.path.join(T, 'wt'), os.path.join(T, 'out')
    cd = copy_without(set(nights), wt)
    common = ['--out', out, '--deep-installed-root', wt]
    rc, txt = cli('masters', '--stages', 'admit-prep', *common)
    pr = rec(out, 'masters', f'admit-prep_{CELL}')
    check('admit-prep prepares the two nights made new', rc == 0 and sorted(d['date'] for d in pr['meta']['admit']['dates'])
          == nights, (rc, pr.get('reason'), txt[-400:]))
    for d in nights:                                   # the members' stored flows: fresh, so the flow step solves nothing
        for f in (f'{d}_scaled_flow.npy', f'{d}_scaled_dewarp.fits'):
            shutil.copy2(os.path.join(flows, f), os.path.join(cd, A.ADMIT_DIR, 'flow', f))
            t = time.time()
            os.utime(os.path.join(cd, A.ADMIT_DIR, 'flow', f), (t, t))
    rc, txt = cli('flow', '--stages', 'admit', '--out', out)      # the pairs carry the cell tree's paths
    fr = rec(out, 'flow', f'admit_{CELL}')
    check('the flow step\'s admit stage takes the stored flows (nothing solved, no GPU) and is done',
          rc == 0 and fr.get('status') == 'done' and not (fr.get('meta') or {}).get('admit', {}).get('recomputed'),
          (rc, fr.get('reason'), txt[-400:]))
    rc, txt = cli('masters', '--stages', 'admit', *common)
    r0 = rec(out, 'masters', f'admit_{CELL}')
    check('without a guard median (recorded or ruled) the stage refuses - nothing substituted (item 10)',
          rc == 2 and 'item 10' in r0.get('reason', '') and not glob.glob(os.path.join(cd, 'subtracted', '*')),
          (rc, r0.get('reason')))
    rc, txt = cli('masters', '--stages', 'admit', *common, '--set', 'masters_admit.DC2_MEDIAN_WITHOUT_RECORD=1.0')
    r1 = rec(out, 'masters', f'admit_{CELL}')
    adm = os.path.join(cd, 'coadd_final', A.ADMISSIONS)
    rows = list(csv.DictReader(open(adm))) if os.path.exists(adm) else []
    check('with the recommended guard median (1.0, test only) both nights are admitted, one admissions.csv row each',
          rc == 0 and r1.get('status') == 'done' and sorted(r['date'] for r in rows) == nights
          and all(r['verdict'] == 'admitted' for r in rows), (rc, r1.get('reason'), [(r['date'], r['verdict'], r['reason'])
                                                                                       for r in rows], txt[-500:]))
    tests = {t['name']: t['passed'] for t in r1.get('tests', [])}
    check('every product and admissions.csv pass their content tests (identity, own beam, cards; the admission record)',
          any(k.startswith('admitted_epoch') for k in tests) and any(k.startswith('admissions') for k in tests)
          and all(tests.values()), {k: v for k, v in tests.items() if not v} or sorted(tests)[:6])
    inst = os.path.join(CFG.paths.COADD_SB_ROOT, CELL)
    m = np.squeeze(fits.getdata(os.path.join(inst, 'coadd_final', f'I_deep_Taper_R075_{SC}.fits'))).astype(np.float64)
    res = {}
    for d in nights:
        pa = os.path.join(cd, 'subtracted', f'{d}_I_epoch_{SC}.fits')
        if not os.path.exists(pa):
            res[d] = (float('nan'), float('nan'), 0)
            continue
        e_adm = np.squeeze(fits.getdata(pa)).astype(np.float64)
        pe = os.path.join(inst, 'subtracted', f'{d}_I_epoch_{SC}.fits')        # the member's installed epoch
        e_mem = (np.squeeze(fits.getdata(pe)).astype(np.float64) if os.path.exists(pe) else
                 np.squeeze(fits.getdata(os.path.join(inst, 'subtracted', f'{d}_I_sub_{SC}.fits'))).astype(np.float64) + m)
        res[d] = bright_ratio(e_adm, e_mem)
    print('  admission_reproduces (admitted / member epoch at the master\'s bright compact sources): '
          + '; '.join(f'{d}: median {v[0]:.6f}, median |r-1| {v[1]:.2e}, {v[2]} sources' for d, v in res.items()))
    check(f'admission_reproduces: each admitted epoch equals the member\'s at bright peaks within {REPRODUCE_TOL:g}',
          all(abs(v[0] - 1) <= REPRODUCE_TOL and v[2] >= 20 for v in res.values()), res)
    rc, txt = cli('masters', '--stages', 'admit', *common, '--set', 'masters_admit.DC2_MEDIAN_WITHOUT_RECORD=1.0')
    check('a second run admits nothing (a night is never admitted twice)', rc == 3, (rc, txt[-300:]))
    ok = validate.run('admissions', adm)
    check('admissions.csv content test, run on its own', ok and all(t['passed'] for t in ok), ok)


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_admit_end_to_end()
    finally:
        if not os.environ.get('EXOPIPE3_TEST_KEEP'):            # set to inspect the admission copy afterwards
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
