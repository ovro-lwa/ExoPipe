#!/usr/bin/env python3
"""The hybrid route, pull request PR-3 (plan/specs/hybrid.md section 4): the recon and wideband stages (lib/hybrid/recon.py,
wideband.py, the content tests in lib/hybrid/tests.py and steps/hybrid.recon_unit / wideband_unit).

Data-free (CI):
  1. the constants bound from the config equal production's (NL 62-80, 828-831 at 4c7098c);
  2. the record content tests: a record of the rule passes every test; each rule broken alone fails its own test (prior fit,
     CLEAN and step convergence, compact sources with the 18 MHz tolerance, the level rule with the caveat scopes, the PSF2x
     registry entry); the wideband record tests likewise (weights, identity, the level only in the caveat scopes);
  3. the recon unit: no stack record = no input; the convergence cascade (NITER and MAXCYC x1 horizon, x3 horizon, x3 elev10,
     each later try told the earlier ones) and a refusal after the last try; a non-convergence refusal stops at once; whatever
     a refused unit wrote is removed;
  4. the wideband unit: a member without a passed reconstruction refuses.
Cluster parity (not here): 18h 46 MHz R075 and 18h 82 MHz R0 against production's records and products, run by hand on lwacalim07.
Run: python tests/test_hybrid_recon.py   (or under pytest)
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from exopipe.phase3.config import CFG  # noqa: E402
from exopipe.phase3.core import record, validate  # noqa: E402
from exopipe.phase3.lib.hybrid import recon as REC  # noqa: E402
import exopipe.phase3.lib.hybrid.tests  # noqa: E402,F401
from exopipe.phase3.steps import hybrid as HS  # noqa: E402

TMP = os.path.join(os.environ.get('TMPDIR', tempfile.gettempdir()), 'test_hybrid_recon')


def test_bind_equals_production():
    g = REC.bind()
    want = dict(SCALES=(0, 12, 24, 47), NITER=500000, GAIN=0.1, MGAIN=0.6, BIAS=0.6, AUTOMASK=3.0, AUTOTHRESH=0.5, PATCH=256,
                MAXACT=100000, MAXLOOP=20000, MAXCYC=400, TAU=0.15, STEP_FWHM_DEG=1.0, CG_TOL=1e-6, CG_MAX=3000,
                LFSS_DIR='/lustre/gh/main/catalogs/lwa1_lfss', CATALOGS='/lustre/gh/main/phase3/deep_parent_catalogs_union_20261014T0203',
                LEVEL_LIMITS=(0.5, 1.6), DEC_BANDS=((40, 90), (10, 40), (-10, 10), (-20, -10), (-30, -20)), HYB_SMALL_LIMIT=5e-3,
                CAVEAT_SCOPES=('64MHz', '69MHz', '73MHz', '78MHz', '82MHz', 'Blue', 'Full'), RECON_GEN='lsrecon-v3-horizon-grad',
                ARCHIVE_MASTERS='/lustre/pipeline/exopipe/phase3/Coadd/Run_20260909_000000', PRIOR_SCALE_RANGE=(0.5, 2.0),
                PRIOR_RMS_FRAC=0.6, COMPACT_TOL=0.10, COMPACT_TOL_LOW=0.20, COMPACT_N_MIN=50, COMPACT_N_SOURCES=200)
    bad = {k: (g[k], v) for k, v in want.items() if g[k] != v}
    assert not bad, bad
    assert REC.NITER == 500000 and REC.SCALES == (0, 12, 24, 47)          # bound as module globals for the ported functions
    assert [c[1:] for c in CFG.hybrid_recon.CASCADE] == [(1, 'horizon'), (3, 'horizon'), (3, 'elev10')]


def _good_record(band='46MHz', w='R075'):
    path, md5, _ants = CFG.hybrid_psf2x.REGISTRY[(band, w)]
    return dict(lst='18h', band=band, weighting=w, generation='lsrecon-v3-horizon-grad', psf2x=path, psf2x_md5=md5,
                test_psf2x=dict(verdict='pass', core_rel_rms=0.05), prior=dict(test_prior_backconversion=dict(median_ratio=1.0, verdict='pass')),
                prior_fit=dict(scale=1.19, rms=0.1, data_rms=1.0), clean=dict(residual_rms_test_region=0.5, dirty_rms_test_region=2.0),
                step=dict(cg_relative_residual=5e-7), test_compact_sources=dict(n=200, median=0.99),
                test_large_scale_level=dict(ratio_by_dec_band={'+40..+90': 0.98, '+10..+40': 1.01}, verdict='pass'),
                product=dict(md5='a' * 32), product_pbcorr=dict(md5='b' * 32))


def _tests(cls, rec, **ctx):
    os.makedirs(TMP, exist_ok=True)
    p = os.path.join(TMP, 'rec.json')
    with open(p, 'w') as f:
        json.dump(rec, f)
    return {r['name'].split('.', 1)[1]: r['passed'] for r in validate.run(cls, p, **ctx)}


def test_record_content_tests():
    ctx = dict(lst='18h', band='46MHz', weighting='R075')
    got = _tests('hybrid_recon_record', _good_record(), **ctx)
    assert set(got) == {'recon_unit', 'prior_backconversion', 'psf2x_core', 'prior_fit', 'clean_converged', 'step_converged',
                        'compact_sources', 'large_scale_level'} and all(got.values()), got

    def only(name, mutate, band='46MHz', w='R075'):
        r = _good_record(band, w)
        mutate(r)
        res = _tests('hybrid_recon_record', r, lst='18h', band=band, weighting=w)
        return [k for k, v in res.items() if not v]
    assert only('prior_fit', lambda r: r['prior_fit'].update(scale=2.1)) == ['prior_fit']
    assert only('prior_fit', lambda r: r['prior_fit'].update(rms=0.61)) == ['prior_fit']
    assert only('clean', lambda r: r['clean'].update(residual_rms_test_region=2.5, dirty_rms_test_region=9.0)) == ['clean_converged']
    assert only('clean', lambda r: r['clean'].update(residual_rms_test_region=1.5, dirty_rms_test_region=1.0)) == ['clean_converged']
    assert only('step', lambda r: r['step'].update(cg_relative_residual=2e-6)) == ['step_converged']
    assert only('compact', lambda r: r['test_compact_sources'].update(median=1.15)) == ['compact_sources']
    assert only('compact', lambda r: r['test_compact_sources'].update(median=1.15), band='18MHz') == []          # 20% at 18 MHz
    assert only('compact', lambda r: r['test_compact_sources'].update(n=49)) == ['compact_sources']
    low = {'+40..+90': 0.98, '+10..+40': 0.4}
    assert only('level', lambda r: r['test_large_scale_level'].update(ratio_by_dec_band=low)) == ['large_scale_level']
    assert only('level', lambda r: r['test_large_scale_level'].update(ratio_by_dec_band=low, verdict='caveat'), band='82MHz', w='R0') == []
    assert only('level', lambda r: r['test_large_scale_level'].update(ratio_by_dec_band=low), band='82MHz', w='R0') == ['large_scale_level']
    assert only('psf', lambda r: r.update(psf2x_md5='0' * 32)) == ['psf2x_core']
    assert only('psf', lambda r: r['test_psf2x'].update(core_rel_rms=0.21)) == ['psf2x_core']
    assert only('backconv', lambda r: r['prior']['test_prior_backconversion'].update(median_ratio=1.00001)) == ['prior_backconversion']
    assert only('unit', lambda r: r.update(generation='lsrecon-v2')) == ['recon_unit']


def test_wideband_record_tests():
    mem = list(CFG.hybrid.WIDEBAND_MEMBERS['Green'])
    good = dict(lst='18h', scope='Green', weighting='R075', subbands=[dict(band=b, weight=0.2) for b in mem],
                test_wideband_identity=dict(verdict='pass', max_rel_err=1e-8, nan_masks_equal=True, prodname_ok=True),
                test_compact_sources=dict(n=200, median=1.01))
    ctx = dict(lst='18h', scope='Green', weighting='R075', members=mem)
    got = _tests('hybrid_wideband_record', good, **ctx)
    assert set(got) == {'wideband_unit', 'wideband_identity', 'wideband_compact_sources', 'wideband_large_scale_level'} and all(got.values()), got
    bad = json.loads(json.dumps(good))
    bad['subbands'][0]['weight'] = 0.3
    assert [k for k, v in _tests('hybrid_wideband_record', bad, **ctx).items() if not v] == ['wideband_unit']
    bad = json.loads(json.dumps(good))
    bad['test_wideband_identity']['max_rel_err'] = 2e-6
    assert [k for k, v in _tests('hybrid_wideband_record', bad, **ctx).items() if not v] == ['wideband_identity']
    bad = dict(good, test_large_scale_level=dict(ratio_by_dec_band={'+40..+90': 1.0}, verdict='pass'))
    assert [k for k, v in _tests('hybrid_wideband_record', bad, **ctx).items() if not v] == ['wideband_large_scale_level']
    blue = dict(good, scope='Blue', subbands=[dict(band=b, weight=0.2) for b in CFG.hybrid.WIDEBAND_MEMBERS['Blue']])
    bctx = dict(ctx, scope='Blue', members=list(CFG.hybrid.WIDEBAND_MEMBERS['Blue']))
    assert [k for k, v in _tests('hybrid_wideband_record', blue, **bctx).items() if not v] == ['wideband_large_scale_level']   # Blue needs it
    blue['test_large_scale_level'] = dict(ratio_by_dec_band={'+40..+90': 0.45}, verdict='caveat')
    assert all(_tests('hybrid_wideband_record', blue, **bctx).values())


def _unit(out, stage, scope, w, fn):
    run = record.Run('hybrid', ['test'], out, CFG)
    with run.unit(f'{stage}.18h_{scope}_{w}', stage=stage, lst='18h', scope=scope, weighting=w) as u:
        fn(u)
    return u


def test_recon_unit_cascade_and_cleanup():
    out = os.path.join(TMP, 'out')
    shutil.rmtree(out, ignore_errors=True)
    wd = os.path.join(out, '18h_46MHz')
    os.makedirs(wd)
    u = _unit(out, 'recon', '46MHz', 'R075', lambda u: HS.recon_unit(u, '18h', '46MHz', 'R075', out))
    assert u.status == 'no-input' and 'no stack record' in u.reason, (u.status, u.reason)
    with open(os.path.join(wd, 'R075_stack.json'), 'w') as f:
        json.dump(dict(nights=[dict(date='2025-01-01')]), f)
    calls = []
    product = HS.recon_plan(out, '18h', '46MHz', 'R075')[0][0]
    real = REC.recon_variant

    def fake(msg):
        def f(lst, band, v, workdir, psf2x, freq, ni, mc, cr, escalation=()):
            calls.append((ni, mc, cr, len(escalation)))
            open(product, 'w').close()                                     # a partial write the refusal must remove
            raise REC.ReconRefusal(msg)
        return f
    try:
        REC.recon_variant = fake('test_clean_converged FAILED: {...} - REFUSING')
        u = _unit(out, 'recon', '46MHz', 'R075', lambda u: HS.recon_unit(u, '18h', '46MHz', 'R075', out))
        assert u.status == 'refused' and 'test_clean_converged FAILED' in u.reason, (u.status, u.reason)
        assert calls == [(500000, 400, 'horizon', 0), (1500000, 1200, 'horizon', 1), (1500000, 1200, 'elev10', 2)], calls
        assert not os.path.exists(product)
        del calls[:]
        REC.recon_variant = fake('test_compact_sources FAILED: {...} - REFUSING')
        u = _unit(out, 'recon', '46MHz', 'R075', lambda u: HS.recon_unit(u, '18h', '46MHz', 'R075', out))
        assert u.status == 'refused' and 'test_compact_sources' in u.reason and len(calls) == 1, (u.status, calls)
        assert not os.path.exists(product)
    finally:
        REC.recon_variant = real
    assert {p for p, _c in HS.recon_plan(out, '18h', '46MHz', 'R075')} == {os.path.join(wd, f) for f in (
        'I_deep_NoTaper_R075_lsrecon_46MHz.fits', 'I_deep_NoTaper_R075_lsrecon_pbcorr_46MHz.fits', 'R075_lsrecon_model.fits',
        'R075_lsrecon_clean_residual.fits', 'R075_lsrecon_compact_restored.fits', 'R075_lsrecon_final_residual.fits', 'R075_lsrecon.json')}


def test_wideband_member_missing_refuses():
    out = os.path.join(TMP, 'out_wb')
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    u = _unit(out, 'wideband', 'Green', 'R075', lambda u: HS.wideband_unit(u, '18h', 'Green', 'R075', out))
    assert u.status == 'refused' and '41MHz holds no passed reconstruction' in u.reason, (u.status, u.reason)
    assert not any(os.path.exists(p) for p, _c in HS.wideband_plan(out, '18h', 'Green', 'R075'))


class TestHybridRecon(unittest.TestCase):
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
