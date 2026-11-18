#!/usr/bin/env python3
"""The hybrid route, pull request PR-4 (plan/specs/hybrid.md section 4): the hybrid and full stages (lib/hybrid/master.py, the
content-test classes hybrid_master_product, hybrid_pbdiffuse and hybrid_record in lib/hybrid/tests.py, and
steps/hybrid.hybrid_master_unit).

Data-free (CI):
  1. the constants bound from the config equal production's (NL 62, 71, 944-949, 1109-1111, 1681; the tolerances of NL 1134,
     1220, 1247, 1259 and 1300);
  2. the record content tests: a record of the rule passes every test; each rule broken alone fails its own test (identity, NaN
     mask, large scales, small scales outside the caveat scopes, the verdict, compact sources with the 18 MHz tolerance, the
     beam-corrected identity, the CLEAN residual's read-back or recorded reason, a product changed after its record);
  3. stack_refused finds a refused Taper residual stack in the stack unit's v2 record;
  4. the unit: no reconstruction record = no input; a wideband member without a passed hybrid refuses; a refusal removes what
     the unit wrote; the CLEAN residual a refused Taper residual stack costs is a refused item (the unit done); the unit's
     inputs are the files the stage read, never its own outputs; its admitted nights are the hybrid's span.
Cluster parity (not here): 18h 46 MHz R075, 04h Blue R075 (striation-restored base) and 18h Full R0 against production's
products and records, run by hand on lwacalim07; the record tests over production's 912 hybrid records.
Run: python tests/test_hybrid_master.py   (or under pytest)
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
from astropy.io import fits

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from exopipe.phase3.config import CFG  # noqa: E402
from exopipe.phase3.core import record, validate  # noqa: E402
from exopipe.phase3.lib.hybrid import master as M  # noqa: E402
from exopipe.phase3.lib.hybrid import recon as REC  # noqa: E402
import exopipe.phase3.lib.hybrid.tests  # noqa: E402,F401
from exopipe.phase3.lib.install.steps_common import md5  # noqa: E402
from exopipe.phase3.steps import hybrid as HS  # noqa: E402

TMP = os.path.join(os.environ.get('TMPDIR', tempfile.gettempdir()), 'test_hybrid_master')
RED = ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz')
GREEN = ('41MHz', '46MHz', '50MHz', '55MHz', '59MHz')
BLUE = ('64MHz', '69MHz', '73MHz', '78MHz', '82MHz')
NAME = '46MHz_I_deep_Taper_Robust-0.75_shflux_mdewarp{pb}_dewarped_aligned_coadd_lsrecon_lsfill{cr}_LST18h_20250405-20251014_N015.fits'


def test_bind_equals_production():
    g = M.bind()
    want = dict(HYB_U_IN=30.0, HYB_U_OUT=60.0, OWN_BEAM_MAX=1e-4, IDENTITY=1e-5, LARGE_REL=1e-3, STRIATION_IDENTITY=1e-6,
                STRIATION_SELECTION='/lustre/pipeline/exopipe/phase3/mosaic/Run_FEATHER3_20241218-20251029/tables/selection_20261022.csv',
                STRIATION_RESTORED='/lustre/gh/main/phase3/lsrecon/striation_restored_20261022',
                STRIATION_SCOPES=('59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz', 'Blue'),
                COADD_SB='/data07/gh/phase3/coadd_sb', HYB_SMALL_LIMIT=5e-3, RECON_GEN='lsrecon-v3-horizon-grad')
    bad = {k: (g[k], v) for k, v in want.items() if g[k] != v}
    assert not bad, bad
    assert {k: tuple(v) for k, v in g['WIDEBANDS'].items()} == dict(Red=RED, Green=GREEN, Blue=BLUE, Full=RED + GREEN + BLUE)
    assert set(g['RAW_NIGHT_CAVEAT']) == {('13h', '64MHz')}
    assert M.IDENTITY == 1e-5 and M.WIDEBANDS['Full'][-1] == '82MHz'           # bound as module globals for the ported functions


def _write(path, data, **cards):
    h = fits.Header({'CTYPE1': 'RA---SIN', 'CTYPE2': 'DEC--SIN', 'CRVAL1': 270.0, 'CRVAL2': 37.0, 'CRPIX1': 9.0, 'CRPIX2': 9.0,
                     'CDELT1': -0.5, 'CDELT2': 0.5})
    for k, v in cards.items():
        h[k] = v
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.writeto(path, np.asarray(data, np.float32), h, overwrite=True)
    return dict(path=path, prodname=cards.get('PRODNAME'), md5=md5(path))


def _good_record(scope='46MHz', root=None):
    root = root or os.path.join(TMP, 'rec')
    img = np.ones((16, 16))
    return dict(lst='18h', scope=scope, weighting='R075', span=dict(nights=[dict(date='2025-04-05'), dict(date='2025-10-14')]),
                test_hybrid=dict(max_rel_err=1e-8, nan_mask_ok=True, large_scale_rel=2e-4, small_scale_rel=1e-3, verdict='pass',
                                 small_scale_caveat=False, test_compact_sources=dict(n=200, median=0.99)),
                product=_write(os.path.join(root, 'h.fits'), img, PRODNAME=NAME.format(pb='', cr='')),
                product_pbcorr=dict(_write(os.path.join(root, 'p.fits'), img, PRODNAME=NAME.format(pb='_pbcorr', cr='')),
                                    test_hybrid_pbcorr=dict(verdict='pass', max_rel_err=1e-8, nan_masks_equal=True)),
                product_cleanres=dict(_write(os.path.join(root, 'c.fits'), img, PRODNAME=NAME.format(pb='', cr='_cleanres')),
                                      test_readback=dict(verdict='pass', max_rel_err=0.0)),
                cleanres_skipped=None)


def _tests(cls, rec, **ctx):
    os.makedirs(TMP, exist_ok=True)
    p = os.path.join(TMP, 'rec.json')
    with open(p, 'w') as f:
        json.dump(rec, f)
    return {r['name'].split('.', 1)[1]: r['passed'] for r in validate.run(cls, p, **ctx)}


def test_record_content_tests():
    got = _tests('hybrid_record', _good_record(), lst='18h', scope='46MHz', weighting='R075')
    assert set(got) == {'hybrid_record_unit', 'hybrid_test', 'hybrid_product_tests', 'hybrid_products'} and all(got.values()), got

    def only(mutate, scope='46MHz'):
        r = _good_record(scope)
        r['scope'] = scope
        mutate(r)
        return [k for k, v in _tests('hybrid_record', r, lst='18h', scope=scope, weighting='R075').items() if not v]
    th = lambda **kw: (lambda r: r['test_hybrid'].update(**kw))
    assert only(th(max_rel_err=2e-5)) == ['hybrid_test']
    assert only(th(nan_mask_ok=False)) == ['hybrid_test']
    assert only(th(large_scale_rel=2e-3)) == ['hybrid_test']
    assert only(th(small_scale_rel=6e-3, verdict='caveat', small_scale_caveat=True)) == ['hybrid_test']      # not a caveat scope
    assert only(th(small_scale_rel=6e-3, verdict='caveat', small_scale_caveat=True), scope='82MHz') == []
    assert only(th(small_scale_rel=6e-3), scope='82MHz') == ['hybrid_test']                                # verdict must say caveat
    assert only(lambda r: r['test_hybrid']['test_compact_sources'].update(median=1.15)) == ['hybrid_test']
    assert only(lambda r: r['test_hybrid']['test_compact_sources'].update(median=1.15), scope='18MHz') == []  # 20% at 18 MHz
    assert only(lambda r: r['test_hybrid']['test_compact_sources'].update(n=49)) == ['hybrid_test']
    assert only(lambda r: r['product_pbcorr']['test_hybrid_pbcorr'].update(max_rel_err=2e-5)) == ['hybrid_product_tests']
    assert only(lambda r: r['product_cleanres']['test_readback'].update(verdict='FAILED')) == ['hybrid_product_tests']
    assert only(lambda r: r['product'].update(md5='0' * 32)) == ['hybrid_products']                        # changed after the record
    assert only(lambda r: r.update(product_cleanres=None, cleanres_skipped='T075 Taper residual stack refused: HG5')) == []
    assert only(lambda r: r.update(product_cleanres=None)) == ['hybrid_product_tests']                    # no reason recorded
    assert only(lambda r: r.update(weighting='R0')) == ['hybrid_record_unit']


def test_product_content_tests():
    d = os.path.join(TMP, 'prod')
    taper = _write(os.path.join(d, 'taper.fits'), np.ones((16, 16)))['path']
    ctx = dict(scope='46MHz', weighting='R075', taper=taper)
    for kind, pb, cr in (('hybrid', '', ''), ('pbcorr', '_pbcorr', ''), ('cleanres', '', '_cleanres')):
        p = _write(os.path.join(d, f'{kind}.fits'), np.ones((16, 16)), PRODNAME=NAME.format(pb=pb, cr=cr))['path']
        got = {r['name'].split('.', 1)[1]: r['passed'] for r in validate.run('hybrid_master_product', p, kind=kind, **ctx)}
        assert got == {'hybrid_name': True, 'hybrid_grid': True, 'hybrid_finite': True, 'hybrid_caveat_cards': True}, (kind, got)
        wrong = 'hybrid' if kind != 'hybrid' else 'pbcorr'                                            # the kind's token checked
        got = {r['name'].split('.', 1)[1]: r['passed'] for r in validate.run('hybrid_master_product', p, kind=wrong, **ctx)}
        assert [k for k, v in got.items() if not v] == ['hybrid_name'], (kind, got)
    p = _write(os.path.join(d, 'r0.fits'), np.ones((16, 16)), PRODNAME=NAME.format(pb='', cr=''))['path']
    got = {r['name'].split('.', 1)[1]: r['passed'] for r in validate.run('hybrid_master_product', p, kind='hybrid', **dict(ctx, weighting='R0'))}
    assert [k for k, v in got.items() if not v] == ['hybrid_name'], got                                   # robust from the weighting
    p = _write(os.path.join(d, 'shift.fits'), np.ones((16, 16)), PRODNAME=NAME.format(pb='', cr=''), CRVAL1=271.0)['path']
    got = {r['name'].split('.', 1)[1]: r['passed'] for r in validate.run('hybrid_master_product', p, kind='hybrid', **ctx)}
    assert [k for k, v in got.items() if not v] == ['hybrid_grid'], got
    blue = '82MHz_I_deep_Taper_Robust-0.75_shflux_mdewarp_dewarped_aligned_coadd_lsrecon_lsfill_LST18h_20250405-20251014_N015.fits'
    p = _write(os.path.join(d, 'b.fits'), np.ones((16, 16)), PRODNAME=blue)['path']
    got = {r['name'].split('.', 1)[1]: r['passed'] for r in validate.run('hybrid_master_product', p, kind='hybrid', **dict(ctx, scope='82MHz'))}
    assert [k for k, v in got.items() if not v] == ['hybrid_caveat_cards'], got                          # caveat scope: cards owed
    pbd = _write(os.path.join(d, 'PBdiffuse_deep_Full.fits'), np.full((16, 16), 0.5), LSDIAG='diffuse primary beam')['path']
    got = {r['name'].split('.', 1)[1]: r['passed'] for r in validate.run('hybrid_pbdiffuse', pbd, taper=taper)}
    assert all(got.values()) and len(got) == 3, got


def test_stack_refused_from_v2_record():
    run_dir = os.path.join(TMP, 'sr')
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(os.path.join(run_dir, '18h_46MHz'))
    os.makedirs(os.path.join(run_dir, '_records', 'hybrid'))
    M.bind()
    assert M.stack_refused(run_dir, '18h', '46MHz', 'T075') is None                         # no record: not run
    item = dict(path=os.path.abspath(os.path.join(run_dir, '18h_46MHz', 'T075_residual_stack.fits')),
                reason='HG5: 18h 46MHz T075: test_stack_rebuild FAILED - REFUSING', gate='HG5')
    with open(os.path.join(run_dir, '_records', 'hybrid', 'stack.18h_46MHz_R075.json'), 'w') as f:
        json.dump(dict(status='done', refused_items=[item]), f)
    assert M.stack_refused(run_dir, '18h', '46MHz', 'T075') == item['reason']
    assert M.stack_refused(run_dir, '18h', '46MHz', 'T0') is None                           # the R0 unit has no record
    open(os.path.join(run_dir, '18h_46MHz', 'T075_stack.json'), 'w').close()
    assert M.stack_refused(run_dir, '18h', '46MHz', 'T075') is None                         # a stack record: passed


def _unit(out, stage, scope, w, fn):
    run = record.Run('hybrid', ['test'], out, CFG)
    with run.unit(f'{stage}.18h_{scope}_{w}', stage=stage, lst='18h', scope=scope, weighting=w) as u:
        fn(u)
    return u


def test_hybrid_unit_outcomes():
    out = os.path.join(TMP, 'out')
    shutil.rmtree(out, ignore_errors=True)
    wd = os.path.join(out, '18h_46MHz')
    os.makedirs(wd)
    run = lambda: _unit(out, 'hybrid', '46MHz', 'R075', lambda u: HS.hybrid_master_unit(u, '18h', '46MHz', 'R075', out))
    u = run()
    assert u.status == 'no-input' and 'no reconstruction record' in u.reason, (u.status, u.reason)
    with open(os.path.join(wd, 'R075_lsrecon.json'), 'w') as f:
        json.dump({}, f)
    plan = HS.hybrid_plan(out, '18h', '46MHz', 'R075')
    assert [os.path.basename(p) for p, _c in plan] == [
        'I_deep_Taper_R075_lsrecon_lsfill_46MHz.fits', 'I_deep_Taper_R075_lsrecon_lsfill_pbcorr_46MHz.fits',
        'I_deep_Taper_R075_lsrecon_lsfill_cleanres_46MHz.fits', 'R075_hybrid.json']
    taper = _write(os.path.join(TMP, 'archive', 'taper.fits'), np.ones((16, 16)))['path']
    an_input = os.path.join(wd, 'R075_lsrecon_model.fits')
    open(an_input, 'w').close()
    real = M.hybrid_unit

    def refusing(lst, scope, v, run_dir):
        open(plan[0][0], 'w').close()                                   # a partial write the refusal must remove
        raise REC.ReconRefusal(f'{lst} {scope} {v}: test_hybrid FAILED: {{...}} - REFUSING')

    def passing(lst, scope, v, run_dir):
        M.READS.update((an_input, plan[0][0]))                          # an input, and an output read back
        rec = _good_record(root=wd)
        for k, (p, _c) in zip(('product', 'product_pbcorr'), plan[:2]):
            os.replace(rec[k]['path'], p)
            rec[k]['path'] = p
        os.remove(rec['product_cleanres']['path'])
        rec.update(taper_master=taper, product_cleanres=None, cleanres_skipped='T075 Taper residual stack refused: HG5 ...')
        with open(plan[3][0], 'w') as f:
            json.dump(rec, f)
        return rec
    try:
        M.hybrid_unit = refusing
        u = run()
        assert u.status == 'refused' and 'test_hybrid FAILED' in u.reason, (u.status, u.reason)
        assert not any(os.path.exists(p) for p, _c in plan)
        M.hybrid_unit = passing
        u = run()
        assert u.status == 'done', (u.status, u.reason, [t for t in u.tests if not t['passed']])
        assert [(os.path.basename(r['path']), r['gate']) for r in u.refused_items] == [
            ('I_deep_Taper_R075_lsrecon_lsfill_cleanres_46MHz.fits', 'taper-residual-stack')], u.refused_items
        assert set(u.inputs) == {os.path.abspath(an_input)}, u.inputs                 # never its own outputs
        assert u.admitted == ['2025-04-05', '2025-10-14'] and all(t['passed'] for t in u.tests)
    finally:
        M.hybrid_unit = real
    # a wideband unit whose member holds no passed hybrid refuses before it runs
    os.makedirs(os.path.join(out, '18h_Blue'))
    with open(os.path.join(out, '18h_Blue', 'R075_lsrecon_wb.json'), 'w') as f:
        json.dump({}, f)
    u = _unit(out, 'hybrid', 'Blue', 'R075', lambda u: HS.hybrid_master_unit(u, '18h', 'Blue', 'R075', out))
    assert u.status == 'refused' and '64MHz holds no passed hybrid' in u.reason, (u.status, u.reason)
    u = _unit(out, 'full', 'Full', 'R0', lambda u: HS.hybrid_master_unit(u, '18h', 'Full', 'R0', out))
    assert u.status == 'no-input' and 'no subband hybrid' in u.reason, (u.status, u.reason)
    assert [os.path.basename(p) for p, _c in HS.hybrid_plan(out, '18h', 'Full', 'R0')] == [
        'I_deep_Taper_R0_lsrecon_lsfill_Full.fits', 'I_deep_Taper_R0_lsrecon_lsfill_pbcorr_Full.fits', 'PBdiffuse_deep_Full.fits',
        'R0_hybrid.json']


class TestHybridMaster(unittest.TestCase):
    def test_all(self):
        for name, fn in sorted(globals().items()):
            if name.startswith('test_') and callable(fn):
                with self.subTest(name):
                    fn()


if __name__ == '__main__':
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            n += 1
            print('PASS', name, flush=True)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f'{n}/{n} PASS')
