"""B-32 (a): one localization estimator and one error source in every tier - the import wrap of the in-place production
exopipe_ops/source_localization.py (core/wrap.py, core/parity_sitecustomize/sitecustomize.py,
lib/catalog/wraps/source_localization.py).

Each check fails when the wrap is wrong: pins (an edited production function refuses), the estimator (= production's own
10-minute branch, weights positive, a non-positive variance refuses), the error source (the fitted row wins over a fork-tier
row, a detection with no fitted row refuses), the child path end to end (wrapped under the ruled values, untouched under the
production values, a failing identity refuses the unit although the script swallows it, an edited production module refuses
at import).
Run: python tests/test_catalog_b32.py
"""
import csv
import importlib.util
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.config import CFG                                # noqa: E402
from exopipe.phase3.core import fallback, legacy, record, wrap      # noqa: E402

OPS = '/lustre/gh/main/exopipe_ops'
T = '/tmp/exopipe3_tests_catalog_b32'
PATCH = wrap.WRAPS['source_localization']['patch']
FAILS, N = [], [0]


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def load(path, name):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def wcsv(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


CHILD = """
import sys
sys.path.insert(0, %r)
import source_localization as SL
print('WRAPPED', getattr(SL, 'B32_WRAPPED', None) is not None, (SL.combine.__doc__ or '')[:40])
if len(sys.argv) > 1 and sys.argv[1] == 'fail':
    SL.detections = lambda ident, U: 1 / 0
    try:
        SL.locate('ID1', '/nonexistent')
    except Exception as e:                     # build_localizations.py swallows it the same way (outside the 10-minute tier)
        print('swallowed', type(e).__name__)
"""


def production_values():
    CFG.authorize([], out=T)
    CFG.set('astrometry.LOC_ESTIMATOR', 'production')
    CFG.set('astrometry.LOC_ERROR_SOURCE', 'production')
    CFG.set('fallbacks.LOC_MASTER_BEAM_FALLBACK', 'fallback')
    CFG.set('fallbacks.LOCALIZATION_FALLBACK', 'fallback')


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    W = load(PATCH, '_b32_patch')
    src = open(f'{OPS}/source_localization.py').read()
    try:
        # ---- pins -------------------------------------------------------------------------------------------------
        check('pins match the production source_localization.py', W.check_pins(src) == [], W.check_pins(src))
        edited = src.replace('w = (1.0 / s2) / float(np.sum(1.0 / s2))', 'w = (1.0 / s2) / float(np.sum(1.0 / s2)) ', 1)
        check('an edited production function fails its pin (combine)', W.check_pins(edited) == ['combine'],
              W.check_pins(edited))

        # ---- the production module in process (it puts the production roots on sys.path itself) ------------------------
        keep_path = list(sys.path)
        SL = load(f'{OPS}/source_localization.py', '_b32_sl')
        sys.path[:] = keep_path
        dets = [dict(ra=150.0 + i * 1e-4, dec=37.0 - i * 2e-4, sig=s, epoch=('14h', f'2025-08-1{i}'), band='41MHz')
                for i, s in enumerate((20.0, 35.0, 50.0))]
        # the production 10-minute branch on the same detections (sublst_pertie active, its correlation pinned here)
        sp = SL._SP
        saved = (sp.active, sp.tenmin_rho, sp.rho_between)
        sp.active, sp.tenmin_rho, sp.rho_between = (lambda: True), (lambda: {'x': 1}), (lambda a, b, R: 0.2)
        prod10 = SL.combine(dets)
        mine = W.ivw_combine(SL)(dets)
        sp.active, sp.tenmin_rho, sp.rho_between = saved
        check('the wrapped estimator = production\'s 10-minute branch (same V)', np.allclose(prod10, mine, rtol=0, atol=1e-12),
              (prod10, mine))
        w = (1 / np.array([20.0, 35.0, 50.0]) ** 2)
        mine_deep = W.ivw_combine(SL)(dets)
        V = SL._cov(dets, SL.RHO_EPOCH, SL.RHO_FAR, False)
        wn = w / w.sum()
        check('deep tier: positive inverse-variance weights, error sqrt(w^T V w)', abs(mine_deep[2] - float(np.sqrt(wn @ V @ wn)))
              < 1e-12 and all(wn > 0), mine_deep)
        saved_cov = SL._cov
        SL._cov = lambda dets, a, b, frame: -np.eye(len(dets))
        try:
            W.ivw_combine(SL)(dets)
            nonpos = 'no refusal'
        except Exception as e:                     # noqa: BLE001 - any other exception is a failure of this check
            nonpos = isinstance(e, RuntimeError) and 'not a positive variance' in str(e) or f'{type(e).__name__}: {e}'
        finally:
            SL._cov = saved_cov
        check('a non-positive variance refuses', nonpos is True, nonpos)

        # ---- the error source: the fitted row wins; a detection without one refuses --------------------------------------
        U = f'{T}/run'
        sv = dict(source_id='S1', identity_id='ID1', band='41MHz')
        wcsv(f'{U}/source_catalogs_V_sb/41MHz_sources_survivors.csv', [sv])
        wcsv(f'{U}/source_catalogs_V_fsb/41MHz_sources_survivors.csv', [dict(sv, source_id='F1')])
        mem = dict(source_id='S1', band='41MHz', cand_id='V1_14h_2025-08-10_41MHz_0001', det_ra='150.0', det_dec='37.0',
                   det_snr='9.0', det_date='2025-08-10', det_stat_err_arcsec='5.0')
        wcsv(f'{U}/source_catalogs_V_sb/41MHz_source_members.csv', [mem])
        wcsv(f'{U}/source_catalogs_V_fsb/41MHz_source_members.csv', [dict(mem, source_id='F1', det_stat_err_arcsec='')])
        saved_beam = SL.beam_as
        SL.beam_as = lambda band: 600.0
        _, d_prod = SL.detections('ID1', U)
        try:
            _, d_fit = W.fitted_detections(SL)('ID1', U)
        except Exception as e:                     # noqa: BLE001 - reported by the check below
            d_fit = f'{type(e).__name__}: {e}'
        check('production keeps the FIRST row (the fork tier sorts first): the model error', len(d_prod) == 1
              and d_prod[0]['stat_source'] == 'model', d_prod)
        check('fitted: the singles row with the fitted error wins', isinstance(d_fit, list) and len(d_fit) == 1
              and d_fit[0]['stat'] == 5.0 and d_fit[0]['stat_source'] == 'own', d_fit)
        wcsv(f'{U}/source_catalogs_V_fsb/41MHz_source_members.csv',
             [dict(mem, source_id='F1', det_stat_err_arcsec=''),
              dict(mem, source_id='F1', cand_id='V1_14h_2025-08-11_41MHz_0002', det_date='2025-08-11', det_stat_err_arcsec='')])
        try:
            W.fitted_detections(SL)('ID1', U)
            refused = 'no refusal'
        except Exception as e:                     # noqa: BLE001 - any other exception is a failure of this check
            refused = isinstance(e, RuntimeError) and '0002' in str(e) or f'{type(e).__name__}: {e}'
        check('fitted: a fork-only detection (no fitted error anywhere) refuses', refused is True, refused)
        SL.beam_as = saved_beam

        # ---- the child path end to end ---------------------------------------------------------------------------------
        script = f'{T}/child.py'
        open(script, 'w').write(CHILD % OPS)
        out = f'{T}/a.csv'
        open(out, 'w').write('a\n1\n')
        CFG.reset()
        r = legacy.run(script, [], log=f'{T}/c1.log')
        check('ruled values: the child\'s source_localization is wrapped (estimator B-32)', r.ok and 'WRAPPED True B-32' in
              r.tail, r.tail[-300:])
        production_values()
        r = legacy.run(script, [], log=f'{T}/c2.log')
        check('production values: nothing wrapped (production code byte for byte)', r.ok and 'WRAPPED False' in r.tail,
              r.tail[-300:])
        CFG.reset()
        run = record.Run('tb32', ['x'], T, CFG, config_sections=())
        with run.unit('loc_fails') as u:
            r = legacy.run(script, ['fail'], log=f'{T}/c3.log', unit=u)
            u.add_output(out, 'csv_table')
        check('a failing identity swallowed by the script refuses the unit (LOCALIZATION site, sticky)', r.rc == 0
              and r.refused and u.status == 'refused' and 'LOCALIZATION_FALLBACK' in u.reason, (r.rc, u.status, u.reason))
        production_values()
        CFG.set('astrometry.LOC_ESTIMATOR', 'ivw-sandwich')          # wrapped, the sites at their production value
        with run.unit('loc_counted') as u2:
            r = legacy.run(script, ['fail'], log=f'{T}/c4.log', unit=u2)
            u2.add_output(out, 'csv_table')
        rec = record.load_record(T, 'tb32', 'loc_counted')
        check('the same failure under the production value of the site: counted, the unit done', u2.status == 'done'
              and rec.get('fallbacks') == {'fallbacks.LOCALIZATION_FALLBACK': 1}, (u2.status, rec.get('fallbacks')))
        CFG.reset()
        fake = f'{T}/fakeops'
        os.makedirs(fake)
        open(f'{fake}/source_localization.py', 'w').write(edited)
        open(f'{T}/child_fake.py', 'w').write(CHILD.replace('%r', repr(fake)).replace(
            'import source_localization as SL', f'sys.path.insert(1, {OPS!r})\nimport source_localization as SL'))
        r = legacy.run(f'{T}/child_fake.py', [], log=f'{T}/c5.log')
        check('an edited production module refuses at import (pin): the child is REFUSED, not an error', r.refused
              and not r.ok and 're-port' in r.tail, (r.refused, r.tail[-300:]))
    finally:
        CFG.reset()
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
