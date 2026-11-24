#!/usr/bin/env python3
"""The masters deep stage `striation` (hybrid spec PR-6a): lib/masters/striation.py, the stage in lib/masters/deep_stages.py, its
config section masters_striation and the content tests of its three product classes.

Data-free (CI):
  1. registration: 'striation' in deep_stages.STAGE_ORDER / STAGES and masters.STAGES['deep'], not a default stage; the config
     section's keys, RULED names; the variants table equals production's;
  2. a synthetic band of 24 LST masters (64 x 64 SIN grids, zenith-centered, a fixed sky drifting 15 deg/h, striations along
     constant Dec injected in the target LST only): on the fitted pixels the model correlates with the injected striations
     (> 0.8) and the residual against the true sky falls below 0.6 of the uncorrected one; the model is zero on the bright
     pixels; pass 1 computed lazily equals pass 1 of a band
     whose LSTs were all loaded first;
  3. the selection gate: md5 mismatch refuses; a cell outside the selection has no variants;
  4. the content tests of striation_apparent / striation_pbcorr / striation_model pass on consistent files and fail on a
     tampered pixel, a wrong STRKIND or a model of zeros.
Cluster only (production's tree present; else UNTESTED-CLUSTER, not a failure):
  5. production's striation_restore.py (exopipe 4c7098c) run on the same synthetic band gives bit-identical corrected master and
     model for the target.
Run: python tests/test_masters_striation.py   (or under pytest)
"""
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

_IMPORT_ERROR = None
try:
    import numpy as np
    from astropy.io import fits
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import validate
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.masters import deep_stages as DS, striation as SR
    import exopipe.phase3.lib.masters.tests  # noqa: F401 (registers the content tests)
except Exception as e:                                    # pragma: no cover
    _IMPORT_ERROR = f'{e.__class__.__name__}: {e}'

PROD_REPO = '/lustre/gh/main/exopipe'
PROD_COMMIT = '4c7098c'
BAND, VAR, TARGET = '59MHz', 'Taper_R075', '04h'
N, CD, BEAM = 64, 2.0, 6.0                      # pixels, deg per pixel, beam FWHM deg (3 px)
SOURCES = [(30.0, 40.0, 50.0), (75.0, 20.0, 30.0), (120.0, 55.0, 40.0), (200.0, 10.0, 25.0), (300.0, 45.0, 35.0)]   # RA, Dec, Jy


def _header(lst):
    h = fits.Header()
    h['NAXIS'] = 2; h['NAXIS1'] = N; h['NAXIS2'] = N
    h['CTYPE1'] = 'RA---SIN'; h['CTYPE2'] = 'DEC--SIN'; h['CRVAL1'] = 15.0 * lst; h['CRVAL2'] = 37.23
    h['CRPIX1'] = N / 2 + 1; h['CRPIX2'] = N / 2 + 1; h['CDELT1'] = -CD; h['CDELT2'] = CD
    h['BMAJ'] = BEAM; h['BMIN'] = BEAM; h['BPA'] = 0.0
    h['DATE-OBS'] = '2024-12-18T02:00:00'; h['DATE-END'] = '2025-01-11T03:00:00'; h['INTTIME'] = 72000.0
    return h


def _true_sky(h):
    """(apparent sky without striations, PB) of one LST grid: point sources x PB + a smooth background; NaN beyond the horizon."""
    from astropy.wcs import WCS
    w = WCS(h); yy, xx = np.indices((N, N)); ra, de = w.all_pix2world(xx, yy, 0)
    ok = np.isfinite(ra) & np.isfinite(de)
    rad = np.hypot(yy - N / 2, xx - N / 2) * CD
    pb = np.where(ok, np.clip(np.cos(np.radians(np.minimum(rad, 89.0))), 0, 1) ** 1.5, 0.0)
    sky = np.where(ok, 0.5 + 0.2 * np.cos(np.radians(np.nan_to_num(de))), 0.0)
    sig = BEAM / CD / 2.3548
    for r0, d0, f0 in SOURCES:
        x0, y0 = w.all_world2pix([r0], [d0], 0)
        if np.isfinite(x0[0]):
            sky = sky + f0 * np.exp(-((xx - x0[0]) ** 2 + (yy - y0[0]) ** 2) / (2 * sig ** 2))
    rng = np.random.default_rng(int(h['CRVAL1'] * 10))
    app = np.where(ok, sky * pb + 0.05 * rng.standard_normal((N, N)), np.nan)
    app[ok & (app == 0)] = 1e-6
    return app, pb, ok


def _stripes(h, ok):
    """earth-fixed striations: a pattern of true Declination (period 8 deg), constant along RA."""
    from astropy.wcs import WCS
    yy, xx = np.indices((N, N)); _ra, de = WCS(h).all_pix2world(xx, yy, 0)
    return np.where(ok, 0.6 * np.sin(2 * np.pi * np.nan_to_num(de) / 8.0), 0.0)


def _make_band(root):
    truth = {}
    for i in range(24):
        lst = f'{i:02d}h'; h = _header(i); app, pb, ok = _true_sky(h)
        if lst == TARGET:
            truth['sky'], truth['stripes'] = app.copy(), _stripes(h, ok)
            app = app + truth['stripes']
        d = os.path.join(root, f'{lst}_{BAND}', 'coadd_final'); os.makedirs(d)
        fits.writeto(SR.master_path(root, lst, BAND, VAR), app.astype(np.float32), h)
        fits.writeto(SR.pb_path(root, lst, BAND, VAR), pb.astype(np.float32), h)
    return truth


def _written_stub(d, c, corr, model):
    """The three products as plain FITS (no header) for content_test calls."""
    out = {}
    for kind, arr in (('apparent', corr), ('pbcorr', np.where(c.pb > 0.1, corr / np.where(c.pb > 0, c.pb, np.nan), np.nan)),
                      ('model', model)):
        out[kind] = os.path.join(d, f'stub_{kind}.fits')
        fits.writeto(out[kind], arr.astype(np.float32), overwrite=True)
    return out


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    raise AssertionError(f'{fn.__name__} did not raise {exc.__name__}')


# ------------------------------------------------------------------------------------------------------------------ 1 registration
def test_registration_and_config():
    assert 'striation' in DS.STAGE_ORDER and DS.STAGES['striation'] is DS.striation
    assert 'striation' in CFG.masters.STAGES['deep'] and 'striation' not in CFG.masters.DEFAULT_STAGES['deep']
    s = CFG.masters_striation
    assert (s.WIN, s.PASS1_WIN, s.TAG, s.ARCHIVE_PBCORR_TOL) == (8, 4, 'ssub-rst8', 1e-3)
    assert s.REF_CEILING == {'Taper_R075': 1.126, 'Taper_R0': 1.224}
    assert set(s.RULED) <= set(k for k in dir(s) if k.isupper())
    assert SR.VARIANTS['Taper_R075'][2] == 'Taper_Robust-0.75' and SR.VARIANTS['NoTaper_R0'][2] == 'NoTaper_Robust+0.0'


# ------------------------------------------------------------------------------------------------------------- 2 synthetic band
def test_synthetic_band():
    d = tempfile.mkdtemp(prefix='str_band_', dir=os.environ.get('TMPDIR') or None)
    try:
        truth = _make_band(d)
        B = SR.Band(d, BAND, VAR, pass1_win=4, win=8)
        corr, model, cov = B.correct(TARGET); c = B.cell(TARGET)
        assert sorted(B.cells) == [f'{i:02d}h' for i in (0, 1, 2, 3, 4, 5, 6, 7, 8)], sorted(B.cells)   # L-4..L+4 only
        fit = c.ok & cov & (model != 0)                   # the pixels the model was fitted on (zone and coverage gaps excluded)
        before = float(np.std((c.d - truth['sky'])[fit])); after = float(np.std((corr - truth['sky'])[fit]))
        r = float(np.corrcoef(model[fit], truth['stripes'][fit])[0, 1])
        # measured 2026-11-18: correlation 0.89, residual ratio 0.45; a model blind to the striations gives ~0 and ~1.0
        assert r > 0.8 and after < 0.6 * before, (r, before, after)
        zone = c.ok & (np.abs(np.nan_to_num(c.d)) > 5 * c.rms)
        assert zone.any() and np.all(model[zone] == 0), 'model must be zero on the bright pixels'
        # lazy pass 1 equals pass 1 of a band with every LST loaded first (pass 1 at n reads only n-1, n, n+1)
        B2 = SR.Band(d, BAND, VAR, pass1_win=4, win=8)
        for i in range(24):
            B2.cell(f'{i:02d}h')
        for n in ('01h', '07h'):
            assert np.array_equal(B.pass1(n), B2.pass1(n), equal_nan=True), n     # NaN beyond the horizon
        # the pass-1 disk cache returns the same array and is keyed by the inputs
        cache = os.path.join(d, 'cache')
        B3 = SR.Band(d, BAND, VAR, cache_dir=cache, pass1_win=4, win=8); a = B3.pass1('02h')
        B4 = SR.Band(d, BAND, VAR, cache_dir=cache, pass1_win=4, win=8); b = B4.pass1('02h')
        assert B4.pass1_from_cache == 1 and np.array_equal(a, b, equal_nan=True)
        os.utime(SR.master_path(d, '03h', BAND, VAR), ns=(1, 1))     # an input changed: the cached result is not reused
        B5 = SR.Band(d, BAND, VAR, cache_dir=cache, pass1_win=4, win=8); B5.pass1('02h')
        assert B5.pass1_from_cache == 0
        B6 = SR.Band(d, BAND, VAR, cache_dir=cache, pass1_win=4, win=8)
        k6 = B6._p1_key('02h')                                         # the key reads no pixels and names the code
        assert B6.cells == {} and SR.CODE_SHA, B6.cells
        real_sha = SR.CODE_SHA
        try:
            SR.CODE_SHA = 'other'
            assert B6._p1_key('02h') != k6, 'a code change must change the pass-1 cache key'
        finally:
            SR.CODE_SHA = real_sha
        assert not [f for f in os.listdir(cache) if '.part' in f], os.listdir(cache)
        # trim keeps memory to the window: cells, pass-1 arrays and coordinate pairs outside it are dropped
        keep = {SR.nb(TARGET, k) for k in range(-4, 5)}
        B.trim(keep)
        assert set(B.cells) <= keep and set(B.p1) <= keep and all(set(k) <= keep for k in B._coord)
        B.trim({TARGET})
        assert set(B.cells) == {TARGET} and not B.p1 and not B._coord
        # gates that cannot measure refuse: no pinned ceiling for the variant, off-plane level not measurable
        info = dict(index_before=1.5, index_after=0.9, coverage=1.0, offplane_ratio=None, src_flux_ratio=1.0)
        fails = SR.content_test(c, {k: v for k, v in _written_stub(d, c, corr, model).items()}, info, {})
        assert any('no pinned' in f for f in fails) and any('off-plane level not measurable' in f for f in fails), fails
        # a missing parents catalog refuses (production passed it as 1.0 over 0 sources)
        _raises(SR.StriationRefusal, SR.measures, c, corr, model, cov, os.path.join(d, 'absent.gaul.csv'))
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------------ 3 selection
def test_selection_gate():
    d = tempfile.mkdtemp(prefix='str_sel_', dir=os.environ.get('TMPDIR') or None)
    real = DS._cfg
    try:
        sel = os.path.join(d, 'selection.csv')
        with open(sel, 'w') as f:
            f.write('lst,band,variant\n04h,59MHz,Taper_R075\n04h,59MHz,Taper_R0\n08h,82MHz,Taper_R0\n')
        cfg = types.SimpleNamespace(hybrid=types.SimpleNamespace(
            STRIATION_SELECTION=sel, STRIATION_SELECTION_MD5=hashlib.md5(open(sel, 'rb').read()).hexdigest()))
        DS._cfg = lambda: cfg
        S = DS.striation_selection()
        assert S[('04h', '59MHz')] == ['Taper_R075', 'Taper_R0'] and ('05h', '59MHz') not in S
        with open(sel, 'a') as f:                                     # changed after it was pinned
            f.write('05h,59MHz,Taper_R0\n')
        assert 'not the pinned' in _raises(Refusal, DS.striation_selection)
        os.remove(sel)
        assert 'md5 None' in _raises(Refusal, DS.striation_selection)
    finally:
        DS._cfg = real
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------ 4 content tests
def _write(p, arr, **cards):
    h = fits.Header()
    for k, v in cards.items():
        h[k] = v
    fits.writeto(p, arr.astype(np.float32), h, overwrite=True)


def test_content_tests():
    d = tempfile.mkdtemp(prefix='str_ct_', dir=os.environ.get('TMPDIR') or None)
    try:
        rng = np.random.default_rng(1)
        master = rng.standard_normal((32, 32)) + 5.0; pb = np.full((32, 32), 0.5); pb[:4] = 0.05
        model = 0.1 * rng.standard_normal((32, 32)); app = master - model
        pbc = np.where(pb > 0.1, app / pb, np.nan)
        base = '59MHz_I_deep_Taper_Robust-0.75_shflux_dewarped_aligned_coadd_ssub-rst8_LST04h_20241218-20250111_N020'
        pbase = '59MHz_I_deep_Taper_Robust-0.75_shflux_pbcorr_dewarped_aligned_coadd_ssub-rst8_LST04h_20241218-20250111_N020'
        P = {k: os.path.join(d, n) for k, n in (('master', 'm.fits'), ('pb', 'pb.fits'), ('apparent', base + '.fits'),
                                                  ('pbcorr', pbase + '.fits'), ('model', base + '_striationmodel.fits'))}
        _write(P['master'], master); _write(P['pb'], pb)
        _write(P['apparent'], app, PRODNAME=os.path.basename(P['apparent']), STRKIND='apparent')
        _write(P['pbcorr'], pbc, PRODNAME=os.path.basename(P['pbcorr']), STRKIND='pbcorr')
        _write(P['model'], model, PRODNAME=os.path.basename(P['model']), STRKIND='model')
        ctx = dict(cell='04h_59MHz', master=P['master'], pb=P['pb'])

        def ok(cls, path, **kw):
            res = validate.run(cls, path, **{**ctx, **kw})
            return [r for r in res if not r['passed']]
        assert ok('striation_apparent', P['apparent'], model=P['model']) == []
        assert ok('striation_pbcorr', P['pbcorr'], apparent=P['apparent']) == []
        assert ok('striation_model', P['model']) == []
        bad = app.copy(); bad[10, 10] += 1.0                                   # a tampered pixel
        _write(P['apparent'], bad, PRODNAME=os.path.basename(P['apparent']), STRKIND='apparent')
        assert [r['name'] for r in ok('striation_apparent', P['apparent'], model=P['model'])] == \
            ['striation_apparent.striation_apparent_arithmetic']
        assert [r['name'] for r in ok('striation_pbcorr', P['pbcorr'], apparent=P['apparent'])] == \
            ['striation_pbcorr.striation_pbcorr_sibling']
        _write(P['model'], np.zeros_like(model), PRODNAME=os.path.basename(P['model']), STRKIND='apparent')
        assert {r['name'] for r in ok('striation_model', P['model'])} == {'striation_model.striation_model_header',
                                                                       'striation_model.striation_model_finite'}
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------- 5 production parity (cluster)
def test_production_parity():
    if not os.path.isdir(os.path.join(PROD_REPO, '.git')):
        print('UNTESTED-CLUSTER: production tree absent')
        return
    src = subprocess.run(['git', '-C', PROD_REPO, 'show', f'{PROD_COMMIT}:exopipe/phase3/striation_restore.py'],
                         stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
    d = tempfile.mkdtemp(prefix='str_prod_', dir=os.environ.get('TMPDIR') or None)
    try:
        _make_band(d)
        mod_path = os.path.join(d, 'striation_restore_prod.py')
        with open(mod_path, 'w') as f:
            f.write(src)
        os.environ['P3_COADD_ROOT'] = d
        spec = importlib.util.spec_from_file_location('striation_restore_prod', mod_path)
        prod = importlib.util.module_from_spec(spec); spec.loader.exec_module(prod)
        PB = prod.Band(BAND, VAR); PB.pass1(); pc, pm, pcov = PB.correct(TARGET)
        B = SR.Band(d, BAND, VAR, pass1_win=4, win=8); vc, vm, vcov = B.correct(TARGET)
        assert np.array_equal(pc, vc, equal_nan=True) and np.array_equal(pm, vm) and np.array_equal(pcov, vcov)
        print(f'  production {PROD_COMMIT} and v2 bit-identical on the synthetic band (target {TARGET}; v2 loaded '
              f'{len(B.cells)} of 24 LSTs)')
    finally:
        os.environ.pop('P3_COADD_ROOT', None)
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------- 6 the stage
def _forced_fail(path, **ctx):
    return False, None, 'forced failure (test of the stage)'


class _Cfg:
    """The real CFG with some sections replaced (the stage reads _cfg().masters_striation / hybrid / paths)."""
    def __init__(self, real, **over):
        self._real, self._over = real, over

    def __getattr__(self, k):
        return self._over[k] if k in self._over else getattr(self._real, k)


def _section(sec, **over):
    return types.SimpleNamespace(**{**{k: getattr(sec, k) for k in dir(sec) if k.isupper()}, **over})


def test_stage():
    import argparse
    from exopipe.phase3.core import record as R
    d = tempfile.mkdtemp(prefix='str_stage_', dir=os.environ.get('TMPDIR') or None)
    real = DS._cfg
    try:
        root = os.path.join(d, 'coadd_sb'); os.makedirs(root); _make_band(root)
        arch = os.path.join(d, 'archive', 'Coadd', 'Run_T', TARGET, BAND); os.makedirs(arch)
        long_ = SR.VARIANTS[VAR][2]
        an = f'{BAND}_I_deep_{long_}_shflux_pbcorr_dewarped_aligned_coadd_LST{TARGET}_20241218-20250111_N020.fits'
        with fits.open(SR.master_path(root, TARGET, BAND, VAR), memmap=False) as hm, \
                fits.open(SR.pb_path(root, TARGET, BAND, VAR), memmap=False) as hp:
            m, pb = hm[0].data.astype(np.float64), hp[0].data.astype(np.float64)
            fits.writeto(os.path.join(arch, an), np.where(pb > 0.1, m / np.where(pb > 0, pb, 1), np.nan).astype(np.float32),
                         hm[0].header)
        par = os.path.join(d, 'parents'); os.makedirs(par)
        with open(SR.parents_csv(par, TARGET, BAND), 'w') as f:
            f.write('# a,b,c,d,ra,e,dec,f,g,h,flux\n')
            for ra, de, fl in SOURCES:
                f.write(f'0,0,0,0,{ra},0,{de},0,0,0,{fl}\n')
        sel = os.path.join(d, 'selection.csv')
        with open(sel, 'w') as f:
            f.write(f'lst,band,variant\n{TARGET},{BAND},{VAR}\n')
        sel_md5 = hashlib.md5(open(sel, 'rb').read()).hexdigest()
        from exopipe.phase3.core import manifest as MFS
        pins = {'test-union': MFS.write_manifest(par, os.path.join(par, MFS.MANIFEST))[1]}   # the digest gate admits `par`

        def cfg(**ms):
            return _Cfg(CFG, masters_striation=_section(CFG.masters_striation, PARENTS=par, **ms),
                        hybrid=types.SimpleNamespace(STRIATION_SELECTION=sel, STRIATION_SELECTION_MD5=sel_md5),
                        paths=_section(CFG.paths, PHASE3_ARCHIVE_ROOT=os.path.join(d, 'archive'), ARCHIVE_RUN='Run_T',
                                       COADD_SB_ROOT=root, PARENT_UNION_MANIFEST_MD5=pins))

        def unit(cell, out, **ms):
            DS._cfg = lambda: cfg(**ms)
            CFG.authorize([], out=out)
            run = R.Run('masters', ['test'], out, CFG, config_sections=()); run.commit = 'abc1234'
            a = argparse.Namespace(deep_installed_root=None, work=os.path.join(d, 'work'))
            try:
                DS.striation(cell, a, run, out, {})
            except Exception:                              # noqa: BLE001 - the unit records it; the test reads the record
                pass
            return run.units[-1]

        out = os.path.join(d, 'out'); cell = f'{TARGET}_{BAND}'
        u = unit(cell, out)
        info = u.meta.get('striation', {}).get(VAR, {})
        assert u.status == 'done', (u.status, u.reason, info)
        files = sorted(os.listdir(os.path.join(out, cell)))
        assert len(files) == 3 and all('ssub-rst8' in f for f in files), files
        assert len(u.outputs) == 3 and all(t['passed'] for t in u.tests), u.tests
        assert info['content_test'] == 'PASS' and info['n_src'] >= 1
        assert {os.path.basename(p) for p in u.inputs} >= {os.path.basename(SR.master_path(root, '00h', BAND, VAR)),
                                                           os.path.basename(SR.pb_path(root, '08h', BAND, VAR))}
        md5s = {f: hashlib.md5(open(os.path.join(out, cell, f), 'rb').read()).hexdigest() for f in files}
        pa = argparse.Namespace(deep_installed_root=None, work=os.path.join(d, 'work'))
        lines, gates = DS._plan_simple('striation')(cell, pa, os.path.join(d, 'out_plan'), set())   # dry run: clean plan
        assert gates == [] and len(lines) == 1 and 'ssub' not in lines[0] and VAR in lines[0], (lines, gates)
        lines, gates = DS._plan_simple('striation')(cell, pa, out, set())                          # products exist: gated
        assert len(gates) == 1 and 'no silent overwrite' in gates[0], gates
        os.rename(SR.pb_path(root, '08h', BAND, VAR), SR.pb_path(root, '08h', BAND, VAR) + '.x')    # an input missing: gated
        lines, gates = DS._plan_simple('striation')(cell, pa, os.path.join(d, 'out_plan'), set())
        os.rename(SR.pb_path(root, '08h', BAND, VAR) + '.x', SR.pb_path(root, '08h', BAND, VAR))
        assert len(gates) == 1 and 'missing' in gates[0] and '08h' in gates[0], gates
        lines, gates = DS._plan_simple('striation')(f'05h_{BAND}', pa, os.path.join(d, 'out_plan'), set())
        assert gates == [] and lines[0].startswith('no input') and 'not in the striation selection' in lines[0], (lines, gates)
        u = unit(cell, out)                                    # rerun: refuses, the earlier products untouched
        assert u.status == 'refused' and 'no silent overwrite' in u.reason, (u.status, u.reason)
        assert {f: hashlib.md5(open(os.path.join(out, cell, f), 'rb').read()).hexdigest() for f in files} == md5s
        out2 = os.path.join(d, 'out2')                         # a failed content test: refused, products renamed *.FAILED
        u = unit(cell, out2, REF_CEILING={VAR: 0.0})
        assert u.status == 'refused' and 'content test FAILED' in u.reason, (u.status, u.reason)
        assert sorted(os.listdir(os.path.join(out2, cell))) == sorted(f + '.FAILED' for f in files)
        out4 = os.path.join(d, 'out4')                         # a REGISTERED content test fails: refused, renamed *.FAILED
        validate.content_test('striation_model')(_forced_fail)
        try:
            u = unit(cell, out4)
        finally:
            validate._TESTS['striation_model'] = [t for t in validate._TESTS['striation_model'] if t[1] is not _forced_fail]
        assert u.status == 'refused' and 'content test(s) FAILED' in u.reason and '_forced_fail' in u.reason, (u.status, u.reason)
        assert sorted(os.listdir(os.path.join(out4, cell))) == sorted(f + '.FAILED' for f in files)
        out5 = os.path.join(d, 'out5')                         # a write fails after the first product: renamed, not left
        from exopipe.phase3.core import products as P
        real_wsh, calls = P.write_standard_header, []

        def wsh(path, **kw):
            calls.append(path)
            if len(calls) == 2:
                raise OSError('simulated write failure')
            return real_wsh(path, **kw)
        P.write_standard_header = wsh
        try:
            u = unit(cell, out5)
        finally:
            P.write_standard_header = real_wsh
        assert u.status == 'error' and 'simulated write failure' in u.reason, (u.status, u.reason)
        assert sorted(os.listdir(os.path.join(out5, cell))) == sorted(os.path.basename(p) + '.FAILED' for p in calls), \
            os.listdir(os.path.join(out5, cell))
        u = unit(cell, os.path.join(d, 'out6'), REF_CEILING={})   # no pinned ceiling for the variant: refused
        assert u.status == 'refused' and 'no pinned striation-index ceiling' in u.reason, (u.status, u.reason)
        Bk = [b for k, b in DS._STRIATION_BANDS.items() if k[1:3] == (BAND, VAR)]
        assert len(Bk) == 1 and set(Bk[0].cells) <= {SR.nb(TARGET, k) for k in range(-4, 5)}, sorted(Bk[0].cells)
        u = unit(f'05h_{BAND}', os.path.join(d, 'out3'))       # not in the selection: no input
        assert u.status == 'no-input' and 'not in the striation selection' in u.reason, (u.status, u.reason)
        assert u.meta.get('parents') is None
        with open(SR.parents_csv(par, TARGET, BAND), 'a') as f:   # the catalog changed after its MANIFEST.md5: refused, dry run gated
            f.write(f'0,0,0,0,10.0,0,20.0,0,0,0,1.0\n')
        u = unit(cell, os.path.join(d, 'out7'))
        assert u.status == 'refused' and 'parent union digest gate: M4' in u.reason and not u.outputs, (u.status, u.reason)
        assert not os.path.exists(os.path.join(d, 'out7', cell)) or os.listdir(os.path.join(d, 'out7', cell)) == []
        lines, gates = DS._plan_simple('striation')(cell, pa, os.path.join(d, 'out_plan'), set())
        assert len(gates) == 1 and 'digest gate: M4' in gates[0], gates
        print(f'  stage: done ({info["index_before"]:.3f} -> {info["index_after"]:.3f}), rerun refused, failed test renamed, '
              f'unselected cell no-input, changed parents catalog refused')
    finally:
        DS._cfg = real
        DS._STRIATION_BANDS.clear()
        shutil.rmtree(d, ignore_errors=True)


TESTS = [test_registration_and_config, test_synthetic_band, test_selection_gate, test_content_tests, test_production_parity,
         test_stage]

try:                                                      # pytest collects TestMastersStriation, not the plain functions
    import unittest

    class TestMastersStriation(unittest.TestCase):
        pass
    for _t in TESTS:
        setattr(TestMastersStriation, _t.__name__, (lambda f: lambda self: (self.skipTest(f'UNTESTED-IMPORT: {_IMPORT_ERROR}')
                                                                            if _IMPORT_ERROR else f()))(_t))
    for _t in TESTS:
        globals().pop(_t.__name__, None)
except Exception:                                         # pragma: no cover
    pass


def main():
    if _IMPORT_ERROR is not None:
        print(f'UNTESTED-IMPORT: cannot import dependencies in this interpreter ({_IMPORT_ERROR}) - run under the pipeline env')
        return 1
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f'PASS  {t.__name__}')
        except Exception as e:
            failed += 1
            print(f'FAIL  {t.__name__}: {e.__class__.__name__}: {str(e)[:600]}')
    print(f'{len(TESTS) - failed}/{len(TESTS)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
