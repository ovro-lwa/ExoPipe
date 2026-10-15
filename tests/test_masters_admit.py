"""D-ADMIT (a), slices 2-4 (lib/masters/admit.py; plan/wave2/DADMIT_DESIGN.md): which nights an installed cell admits and
the reference they are admitted against.

Nights: a night is admitted only when no exclusion names it, admissions.csv does not hold it, the build's record does not
hold it, and its survey image was written after the build fixed its night set - each condition alone fails a check on a
synthetic cell; on real installed cells (read only) the nights in no build record were all seen by their builds, so none
is admitted. Reference: the installed master divided by its pixel factor, header stripped as stage-in strips it, in a
dewarped cell too; a cell whose V master holds a night fewer is not refused; the copies under admit/ref/ are never
silently replaced.
Run: python tests/test_masters_admit.py
"""
import csv
import os
import shutil
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
from exopipe.phase3.core.record import Refusal                      # noqa: E402
from exopipe.phase3.lib.masters import admit as A                   # noqa: E402

T = '/tmp/exopipe3_tests_admit'
FAILS, N = [], [0]


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


class _Ex:
    def __init__(self, nights):
        self.nights = set(nights)

    def is_excluded(self, lst, scope, d):
        return d in self.nights


def test_nights_rule():
    cell = os.path.join(T, '23h_23MHz')
    os.makedirs(os.path.join(cell, 'coadd_final'))
    rec = os.path.join(cell, A.BUILD_NIGHTS)
    with open(rec, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lst', 'freq', 'date', 'i_rms_mjy'])
        w.writerows([['23h', '23MHz', '2025-01-01', '1'], ['23h', '23MHz', '2025-01-02', '1']])
    with open(os.path.join(cell, 'coadd_final', A.ADMISSIONS), 'w', newline='') as f:
        csv.writer(f).writerows([['date', 'verdict'], ['2025-01-04', 'admitted']])
    t = time.time() - 3600
    os.utime(rec, (t, t))
    entries = []
    for d, dt in (('2025-01-01', -100), ('2025-01-02', 100), ('2025-01-03', -100), ('2025-01-04', 100), ('2025-01-05', 100),
                  ('2025-01-06', 100)):
        img = os.path.join(T, f'{d}.fits')
        open(img, 'w').write('x')
        os.utime(img, (t + dt, t + dt))
        entries.append((d, os.path.join(T, 'Science_x'), img))
    entries.append(('2025-01-07', os.path.join(T, 'Science_x'), None))
    take, why = A.nights_to_admit('23h', '23MHz', cell, entries, _Ex({'2025-01-06'}))
    check('nights: only the night in no record, not excluded, not admitted, with an image newer than the build is taken',
          [x[0] for x in take] == ['2025-01-05'], (take, why))
    want = {'2025-01-01': 'installed build', '2025-01-02': 'installed build', '2025-01-03': 'older than the installed build',
            '2025-01-04': A.ADMISSIONS, '2025-01-06': 'excluded', '2025-01-07': 'no survey image'}
    check('nights: each other night says why (a recorded night stays out even with a newer image)',
          set(why) == set(want) and all(want[d] in why[d] for d in want), why)
    take, why = A.nights_to_admit('23h', '23MHz', cell, entries, _Ex(()), epochs=['2025-01-05', '2025-01-03', '2025-02-01'])
    check('nights: --epochs restricts to the named nights; a named night that does not qualify or does not exist says why',
          [x[0] for x in take] == ['2025-01-05'] and set(why) == {'2025-01-03', '2025-02-01'}
          and 'no Phase-2 night' in why['2025-02-01'], (take, why))
    os.remove(rec)
    try:
        A.nights_to_admit('23h', '23MHz', cell, entries, _Ex(()))
        check('nights: refused without the build\'s night record', False)
    except Refusal as e:
        check('nights: refused without the build\'s night record', A.BUILD_NIGHTS in str(e), str(e))


def test_nights_real():
    """Read only: cells whose builds left nights out of their records (wrong grid, refused before measurement)."""
    import contextlib
    import io
    from exopipe.phase3.core import exclusions as EX
    from exopipe.phase3.lib.masters import coadd as co
    ex = EX.load()
    root = CFG.paths.COADD_SB_ROOT
    for cell, night in (('02h_Green', '2025-12-26'), ('11h_Full', '2025-03-09'), ('23h_23MHz', '2025-10-29')):
        lst, sc = cell.split('_')
        with contextlib.redirect_stdout(io.StringIO()):
            entries = co.discover_dates(lst, sc.replace('MHz', '') if sc.endswith('MHz') else sc)
        take, why = A.nights_to_admit(lst, sc, os.path.join(root, cell), entries, ex)
        check(f'real {cell}: no night to admit; {night} (in no build record) was seen by the build',
              not take and 'older than the installed build' in why.get(night, ''), (take, why.get(night)))


def test_reference():
    from astropy.io import fits
    from exopipe.phase3.lib.masters import flux_correction as FC
    root = CFG.paths.COADD_SB_ROOT
    ents = A.installed_reference('13h_78MHz', os.path.join(root, '13h_78MHz'))
    ok = True
    for e in ents:
        with fits.open(e['path']) as h:
            raw, hh = h[0].data, h[0].header
            f = FC.pixel_factor(hh)
            sl = (Ellipsis, slice(1500, 1510), slice(1500, 1510))
            ok &= (abs(e['factor'] - f) < 1e-12 and np.array_equal(e['data'][sl], (raw[sl].astype(np.float64) / f).astype(np.float32),
                                                                   equal_nan=True))
        ok &= not any(k.startswith(tuple(CFG.masters_stagein.STRIP_CARD_PREFIXES)) for k in e['header'].keys())
    check('reference: the installed I and V masters / their pixel factor, SH*/FLUX* cards stripped (13h_78MHz)',
          ok and [e['stokes'] for e in ents] == ['I', 'V'] and not ents[0]['mdwarp'], [(e['stokes'], e['factor']) for e in ents])
    d = A.installed_reference('23h_23MHz', os.path.join(root, '23h_23MHz'), read_data=False)
    check('reference: a dewarped cell is admitted against its installed (dewarped) master', d[0]['mdwarp'] and d[1]['mdwarp'],
          [(e['stokes'], e['mdwarp']) for e in d])
    v = A.installed_reference('02h_55MHz', os.path.join(root, '02h_55MHz'), read_data=False)
    check('reference: a cell whose V master holds a night fewer (per-Stokes sets) is not refused',
          v[0]['constdig'] != v[1]['constdig'], [(e['stokes'], e['constdig']) for e in v])
    cell = os.path.join(T, 'ref_13h_78MHz')
    paths = A.write_reference(cell, '78MHz', ents)
    again = A.write_reference(cell, '78MHz', ents)
    h = fits.getheader(paths['I'])
    check('reference copies: admit/ref/<var>_<scope>.fits with the resolution in HISTORY; writing the same again is a no-op',
          paths == again and sorted(paths) == ['I', 'V'] and paths['I'].endswith('/admit/ref/I_deep_Taper_R075_78MHz.fits')
          and any('ADMIT reference: installed' in str(x) for x in h.get('HISTORY', [])), paths)
    ents[0]['data'] = ents[0]['data'] * np.float32(1.001)
    try:
        A.write_reference(cell, '78MHz', ents)
        check('reference copies: a copy with other content is never replaced', False)
    except Refusal as e:
        check('reference copies: a copy with other content is never replaced', 'other content' in str(e), str(e))


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_nights_rule()
        test_nights_real()
        test_reference()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
