#!/usr/bin/env python
"""lib/photometry/stamp_engine.py, the engine of ad hoc 10-s subband cutouts by the stamp method (Gregg 2026-11-21,
R-GREGG-20261121-THREE-DECISIONS-APPROVED item 2): its comparison helpers, the crop box, the generation-6 record and reference
lookups, and its gates run as the step runs it (a child process) on a stub code root: the exclusion (exit 3, before anything is
read), no ad hoc set, no flux-scale record and a changed cube (exit 2, 'REFUSED:'). The pixel parity with generation 6 on real
data is the cluster run quoted in the pull request; tests/test_photometry.py runs the step through the engine's seam.
Run: python tests/test_stamp_engine.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from astropy.wcs import WCS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from exopipe.phase3.lib.photometry import stamp_engine as E   # noqa: E402

FAILS = []
ENGINE = os.path.abspath(E.__file__)


def check(label, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + label + ('' if cond else '  ' + str(detail)))
    if not cond:
        FAILS.append(label)


def refused(fn, *a):
    try:
        fn(*a)
    except E.Refused as e:
        return str(e)
    return None


STUB_SS = '''import os
CUBE_ROOT = os.environ.get('P3_SNAP_CUBE_ROOT', '/nonexistent')
DATA_DECISIONS = 'stub_data_decisions.csv'
def excluded_subbands(lst, date):
    return {'55MHz'} if (lst, date) == ('00h', '2025-01-02') else set()
'''
STUB_CUT = '''_TARGETS = []
def _planets():
    return ()
def _targets():
    return []
'''


def main():
    d = tempfile.mkdtemp(prefix='test_stamp_engine_')
    try:
        # ---- comparison helpers
        a = np.array([[1.0, np.nan], [3.0, 4.0]], np.float32)
        check('same_array: equal arrays with the same NaN pattern', E.same_array(a, a.copy())[0])
        b = a.copy(); b[0, 0] = np.nan
        check('same_array: a different NaN pattern differs', not E.same_array(a, b)[0] and 'NaN' in E.same_array(a, b)[1])
        b = a.copy(); b[1, 1] += 1e-6
        check('same_array: one value off by 1e-6 differs (exact equality)', not E.same_array(a, b)[0])
        check('same_array: shapes differ', not E.same_array(a, a[:1])[0])
        check('same_array: strings', E.same_array(np.array(['x', 'y']), np.array(['x', 'y']))[0]
              and not E.same_array(np.array(['x', 'y']), np.array(['x', 'z']))[0])
        times = {c: np.arange(3.0) for c in E.TIMES_COLS}
        mine, ref = dict(data=a, times=times), dict(data=a.copy(), times={c: v.copy() for c, v in times.items()})
        check('parity: equal pixels and TIMES', E.parity(mine, ref) == [])
        ref['times']['admitted'] = np.array([0.0, 1.0, 1.0])
        check('parity: a differing TIMES column is reported', any('admitted' in x for x in E.parity(mine, ref)))

        # ---- crop box: the union of every frame's box plus pad 2 and the margin, clipped to the image
        w = WCS(naxis=2)
        w.wcs.ctype = ['RA---SIN', 'DEC--SIN']
        w.wcs.crval = [10.0, 40.0]
        w.wcs.crpix = [51.0, 51.0]          # 1-based: pixel (50, 50) 0-based
        w.wcs.cdelt = [-0.1, 0.1]
        n, half = 11, 5
        box = E.crop_box(w, [(10.0, 40.0)], half, n, 100, 100)
        m = 2 + E.MARGIN_PX
        check('crop_box: one position -> its box + pad + margin', box == (45 - m, 45 + n + m, 45 - m, 45 + n + m), box)
        box2 = E.crop_box(w, [(10.0, 40.0), (10.0, 40.5)], half, n, 100, 100)
        check('crop_box: two positions -> the union (5 pixels apart in y)', box2 == (45 - m, 50 + n + m, 45 - m, 45 + n + m), box2)
        edge = E.crop_box(w, [(w.all_pix2world(1, 1, 0)[0].item(), w.all_pix2world(1, 1, 0)[1].item())], half, n, 100, 100)
        check('crop_box: a box at the image edge is clipped at 0', edge is not None and edge[0] == 0 and edge[2] == 0, edge)
        check('crop_box: no finite position -> None', E.crop_box(w, [(np.nan, np.nan)], half, n, 100, 100) is None)

        # ---- generation-6 record and reference lookups
        ref_root = os.path.join(d, 'gen6')
        why = refused(E.fluxscale_record, ref_root, '00h_2025-09-06', '55MHz')
        check('fluxscale_record: no record -> refused', why and 'no generation-6 flux-scale record' in why, why)
        cube = os.path.join(d, 'cube.fits')
        with open(cube, 'wb') as f:
            f.write(b'x' * 2880)
        st = os.stat(cube)
        rec = {'cubes': {'V': {'path': cube, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}}}
        p = os.path.join(ref_root, '_fluxscale', '55MHz', '00h_2025-09-06_55MHz_fluxscale.json')
        os.makedirs(os.path.dirname(p))
        json.dump(rec, open(p, 'w'))
        check('fluxscale_record: the record is read', E.fluxscale_record(ref_root, '00h_2025-09-06', '55MHz')[1] == rec)
        check('check_cube_unchanged: same path, size and mtime -> accepted', refused(E.check_cube_unchanged, rec, 'V', cube) is None)
        why = refused(E.check_cube_unchanged, rec, 'I', cube)
        check('check_cube_unchanged: no record of the Stokes -> refused', why is not None, why)
        os.utime(cube, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
        why = refused(E.check_cube_unchanged, rec, 'V', cube)
        check('check_cube_unchanged: a newer mtime -> refused (changed since generation 6)', why and 'changed' in why, why)
        os.utime(cube, ns=(st.st_atime_ns, st.st_mtime_ns))
        why = refused(E.find_reference, ref_root, '00h_2025-09-06', '55MHz', 'V')
        check('find_reference: no cutout -> refused', why and 'no generation-6 cutout' in why, why)
        for sample, obj in (('HWJ', 'A_star'), ('LPT', 'Z_lpt'), ('LPT', 'B_lpt')):
            od = os.path.join(ref_root, sample, '55MHz', obj)
            os.makedirs(od, exist_ok=True)
            for S in 'VI':
                open(os.path.join(od, '%s_00h_20250906T031500_%s.fits' % (obj, S)), 'w').close()
        os.makedirs(os.path.join(ref_root, 'LPT', '55MHz', 'A_other'), exist_ok=True)
        open(os.path.join(ref_root, 'LPT', '55MHz', 'A_other', 'A_other_00h_20250907T031500_V.fits'), 'w').close()
        got = E.find_reference(ref_root, '00h_2025-09-06', '55MHz', 'V')
        check('find_reference: LPT first, objects sorted, only the cell\'s date', got.endswith('/LPT/55MHz/B_lpt/'
              'B_lpt_00h_20250906T031500_V.fits'), got)

        # ---- the engine's gates, run as the step runs it (a child process) on a stub code root
        code = os.path.join(d, 'code')
        os.makedirs(code)
        open(os.path.join(code, 'phase3_snapshot_search.py'), 'w').write(STUB_SS)
        open(os.path.join(code, 'phase3_snapshot_cutouts.py'), 'w').write(STUB_CUT)
        out = os.path.join(d, 'out')

        def engine(cell, band, samples=True, cube_root=None):
            env = {k: v for k, v in os.environ.items() if not k.startswith('P3_')}
            if samples:
                env['P3_SNAP_CUTOUT_SAMPLES'] = 'ADHOC=%s' % os.path.join(d, 'adhoc.csv')
            if cube_root:
                env['P3_SNAP_CUBE_ROOT'] = cube_root
            p = subprocess.run([sys.executable, ENGINE, '--code-root', code, '--ref-root', ref_root, '--cell', cell, '--band',
                                band, '--out', out], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=os.path.dirname(ENGINE),
                               env=env)
            return p.returncode, p.stdout.decode()

        rc, txt = engine('00h_2025-01-02', '55MHz')
        check('engine: an excluded (cell, subband) -> EXCLUDED, exit 3, nothing written', rc == 3 and txt.startswith('EXCLUDED:')
              and not os.path.exists(out), (rc, txt))
        rc, txt = engine('00h_2025-09-06', '55MHz', samples=False)
        check('engine: no ad hoc set -> REFUSED, exit 2', rc == 2 and 'REFUSED:' in txt and 'P3_SNAP_CUTOUT_SAMPLES' in txt, (rc, txt))
        rc, txt = engine('00h_2025-09-06', '64MHz')
        check('engine: no flux-scale record -> REFUSED, exit 2', rc == 2 and 'REFUSED:' in txt and 'flux-scale record' in txt, (rc, txt))
        cr = os.path.join(d, 'cubes')
        os.makedirs(os.path.join(cr, '00h_2025-09-06'))
        shutil.copy2(cube, os.path.join(cr, '00h_2025-09-06', '00h_2025-09-06_55MHz_V_flowcube.fits'))
        rc, txt = engine('00h_2025-09-06', '55MHz', cube_root=cr)
        check('engine: a cube other than the one generation 6 cut -> REFUSED, exit 2', rc == 2 and 'REFUSED:' in txt
              and 'generation 6 cut V from' in txt, (rc, txt))
        check('engine: never writes outside --out (no file in the stub code root but the stubs)',
              sorted(f for f in os.listdir(code) if not f.startswith('__pycache__'))
              == ['phase3_snapshot_cutouts.py', 'phase3_snapshot_search.py'], os.listdir(code))
    finally:
        shutil.rmtree(d)
    print('ALL PASS' if not FAILS else 'FAILED: %d' % len(FAILS))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
