#!/usr/bin/env python3
"""The hybrid route, pull request PR-6b (plan/specs/hybrid.md section 4): `exopipe3 mosaic --generation v3`, the FEATHER3 mosaics
(lib/mosaic/v3.py, tests_v3.py; band_products product Vclean).

Data-free (CI; the engine, the PB derivation and the renderer are stand-ins - their arithmetic is production's, proven by the cluster
parity below):
  1. config: the input patterns equal production's globs (STG Vclean, S2B), the variants, products and default striation choice;
  2. names: every map and render name of a Blue job equals production's FEATHER3 file name; the engine's link names and weighting map
     per job (coadd_jysr._paths: Stokes V takes the Robust 0 map);
  3. job selection: plain, then striation-corrected (the selection's pairs only), then Vclean; --striation with / without / both;
     --variants; unknown scope, product, variant, striation or stage refused;
  4. the plain stage: the weighting maps derived per LST from the robust's Taper master, with production's header (STG); a plan
     refusal and a failed derivation refuse; the tile list tests pass, and fail on 23 tiles and on a changed weighting map;
  5. the striation-corrected and Stokes V stages: the selected LSTs (all for Stokes V) take the archived master, the others the plain
     one, the weighting maps are the plain job's; a missing corrected master, a master with an excluded night, and a missing plain
     stage refuse;
  6. the co-add: the engine is called per domain on a link tree under its own names with the job's tag and product; outputs placed
     with their md5s; a MISSING tile refuses, a guard skip is recorded (mosaic.TILE_GUARD_ACTION skip-and-record) or refuses
     ('refuse'); the render is called with the job's product and tag and refuses on another tile list;
  7. the CLI: --generation v3 needs --out; v3 refuses --lsts, --weightings, --workers 2; --striation refuses in v2 and v5.
With healpy (the cluster): band_products renders a Stokes V map under production's name with its own PSF map's beam.
Cluster parity (not here): Blue (plain Taper R-0.75 and R0, striation-corrected Taper R-0.75, Stokes V) against lwacalim09's FEATHER3
working tree: weighting maps whole-file md5, HEALPix maps and renders data md5.
Run: python tests/test_mosaic_v3.py   (or under pytest)
"""
import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np
from astropy.io import fits

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from exopipe.phase3.config import CFG  # noqa: E402
from exopipe.phase3.core import exclusions, record, validate  # noqa: E402
from exopipe.phase3.lib.masters import pb_apply  # noqa: E402
from exopipe.phase3.lib.mosaic import band_products as BP  # noqa: E402
from exopipe.phase3.lib.mosaic import tests_v3 as T3  # noqa: E402
from exopipe.phase3.lib.mosaic import v3  # noqa: E402

LSTS = [f'{h:02d}h' for h in range(24)]
J_R075, J_R0 = v3.Job('I', 'Taper_R075'), v3.Job('I', 'Taper_R0')
J_SSUB, J_V = v3.Job('I', 'Taper_R075', ssub=True), v3.Job('Vclean', 'Taper_R0')


def _tmp(prefix):
    return tempfile.mkdtemp(prefix=f'exopipe3_v3_{prefix}', dir=os.environ.get('TMPDIR') or None)


def _md5f(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return str(e)
    raise AssertionError(f'{fn.__name__} did not raise {exc.__name__}')


def _img(path, nights=('2025-08-09', '2025-10-29'), **cards):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h = fits.Header()
    for k, v in dict(PBCORR=True, BMAJ=0.1, BMIN=0.09, RESTFRQ=7.5e7, PBMODEL='m', PBCHROM='c', PBRESID='r',
                     CRVAL1=10.0, CDELT1=-0.03, RADESYS='FK5').items():
        h[k] = v
    for k, v in cards.items():
        if v is None:
            del h[k]
        else:
            h[k] = v
    for n in nights:
        h.add_history(f'COADD constituent {n}')
    fits.PrimaryHDU(np.ones((4, 4), np.float32), header=h).writeto(path, overwrite=True)
    return path


def _hpx(path, sig, **cards1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdu = fits.BinTableHDU.from_columns([fits.Column(name='SIGNAL', format='E', array=np.asarray(sig, np.float32))])
    for k, v in cards1.items():
        hdu.header[k] = v
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)
    return path


@contextlib.contextmanager
def _patched(*triples):
    old = [(o, a, getattr(o, a)) for o, a, _v in triples]
    for o, a, val in triples:
        setattr(o, a, val)
    try:
        yield
    finally:
        for o, a, val in old:
            setattr(o, a, val)


@contextlib.contextmanager
def _fake_module(name, mod):
    pkg, _, attr = name.rpartition('.')
    pm = sys.modules[pkg]
    had_m, old_m = name in sys.modules, sys.modules.get(name)
    had_a, old_a = hasattr(pm, attr), getattr(pm, attr, None)
    sys.modules[name] = mod
    setattr(pm, attr, mod)
    try:
        yield mod
    finally:
        if had_m:
            sys.modules[name] = old_m
        else:
            sys.modules.pop(name, None)
        if had_a:
            setattr(pm, attr, old_a)
        else:
            delattr(pm, attr)


@contextlib.contextmanager
def _no_content_tests():
    with _patched((validate, 'run', lambda cls, path, **ctx: [])):
        yield


def _args(**k):
    a = dict(scopes=None, products=None, variants=None, striation=None, stages=None, from_stage=None, until_stage=None, work=None)
    a.update(k)
    return types.SimpleNamespace(**a)


class _Ex:
    def __init__(self, bad=()):
        self.bad = set(bad)

    def excluded_constituents(self, lst, scope, dates):
        return sorted(d for d in dates if (lst, d) in self.bad)


def _unit(out, st, fn, scope, job, *a):
    run = record.Run('mosaic', ['test'], out, CFG)
    with run.unit(v3.unit_name(st, scope, job), stage=st, scope=scope, job=job.label) as u:
        fn(u, scope, job, out, *a)
    return u


# ---- 1. config --------------------------------------------------------------------------------------------------------------------

def test_config_equals_production():
    C = CFG.mosaic_v3
    # STG 'Vclean': f'{band}_V_deep_Taper_Robust+0.0_vclean_*pbcorr_dewarped_aligned_coadd_LST{lst}_*.fits'
    assert C.VCLEAN_PATTERN.format(scope='Blue', lst='04h') == 'Blue_V_deep_Taper_Robust+0.0_vclean_*pbcorr_dewarped_aligned_coadd_LST04h_*.fits'
    # S2B: f'{band}_I_deep_{LONG[variant]}_shflux*_pbcorr_dewarped_aligned_coadd_ssub-rst8_LST{lst}_*.fits'
    assert C.SSUB_PATTERN.format(scope='78MHz', var='Taper_Robust-0.75', tag=CFG.masters_striation.TAG, lst='15h') == \
        '78MHz_I_deep_Taper_Robust-0.75_shflux*_pbcorr_dewarped_aligned_coadd_ssub-rst8_LST15h_*.fits'
    assert C.VARIANTS == ('Taper_R075', 'Taper_R0') and C.PRODUCTS == ('I', 'Vclean') and C.VCLEAN_VARIANT == 'Taper_R0'
    assert C.STRIATION == 'both' and C.REQUIRE_N_LSTS == 24 and len(C.SCOPES) == 19 and set(C.RENDER_SCOPES) == set(C.SCOPES)
    assert {CFG.mosaic.VARIANT_TOKENS[v] for v in C.VARIANTS} == {'Taper_Robust-0.75', 'Taper_Robust+0.0'}


# ---- 2. names ---------------------------------------------------------------------------------------------------------------------

# lwacalim09:/fast/gh/main/phase3/mosaic_v3/{healpix,renders} (the FEATHER3 working tree), Blue
PROD_NAMES = {
    J_R075: ['healpix_I_deep_Taper_R075_Blue_jysr_nside2048.fits', 'healpix_PSF_I_deep_Taper_R075_Blue_nside2048.fits',
             'healpix_I_deep_Taper_R075_Blue_jybeam_nside2048.fits', 'healpix_PBeff_I_deep_Taper_R075_Blue_nside2048.fits',
             'Blue_Taper_R075_jysr_Mollweide.fits', 'Blue_Taper_R075_jysr_RADec.fits', 'Blue_Taper_R075_jybeam_Mollweide.fits',
             'Blue_Taper_R075_jybeam_RADec.fits'],
    J_SSUB: ['healpix_I_deep_Taper_R075_Blue_ssub-rst8_jysr_nside2048.fits', 'healpix_PSF_I_deep_Taper_R075_Blue_ssub-rst8_nside2048.fits',
             'healpix_I_deep_Taper_R075_Blue_ssub-rst8_jybeam_nside2048.fits', 'healpix_PBeff_I_deep_Taper_R075_Blue_ssub-rst8_nside2048.fits',
             'Blue_Taper_R075_ssub-rst8_jysr_Mollweide.fits', 'Blue_Taper_R075_ssub-rst8_jysr_RADec.fits',
             'Blue_Taper_R075_ssub-rst8_jybeam_Mollweide.fits', 'Blue_Taper_R075_ssub-rst8_jybeam_RADec.fits'],
    J_V: ['healpix_V_deep_Taper_R0_clean_Blue_jysr_nside2048.fits', 'healpix_PSF_V_deep_Taper_R0_clean_Blue_nside2048.fits',
          'healpix_V_deep_Taper_R0_clean_Blue_jybeam_nside2048.fits', 'healpix_PBeff_V_deep_Taper_R0_clean_Blue_nside2048.fits',
          'Blue_V_deep_Taper_R0_clean_jysr_Mollweide.fits', 'Blue_V_deep_Taper_R0_clean_jysr_RADec.fits',
          'Blue_V_deep_Taper_R0_clean_jybeam_Mollweide.fits', 'Blue_V_deep_Taper_R0_clean_jybeam_RADec.fits']}


def test_names_equal_production():
    for job, want in PROD_NAMES.items():
        got = [os.path.basename(p) for st in ('coadd', 'render') for p, _c in v3.outputs_of(st, '/o', 'Blue', job)]
        assert got == want, (job, got)
    assert (J_R075.engine_name('Blue'), J_V.engine_name('Blue')) == ('I_deep_Taper_R075_pbcorr_Blue.fits',
                                                                     'V_deep_Taper_R0_clean_pbcorr_Blue.fits')
    assert (J_R075.robust, J_SSUB.robust, J_R0.robust, J_V.robust) == ('R075', 'R075', 'R0', 'R0')
    assert (J_SSUB.pb_job(), J_V.pb_job(), J_R0.pb_job()) == (J_R075, J_R0, J_R0)
    st = v3.outputs_of('stage', '/o', 'Blue', J_R075)
    assert st[0] == ('/o/Blue_I_Taper_R075/stage/tiles.json', 'mosaic_v3_tiles') and len(st) == 25
    assert st[5] == ('/o/Blue_I_Taper_R075/stage/pb/04h/PB_deep_R075_Blue.fits', 'mosaic_weight_pb')
    assert v3.outputs_of('stage', '/o', 'Blue', J_V) == [('/o/Blue_Vclean/stage/tiles.json', 'mosaic_v3_tiles')]
    assert v3.unit_name('coadd', 'Blue', J_SSUB) == 'v3-coadd.Blue_I_Taper_R075_ssub-rst8'
    _raises(ValueError, v3.Job, 'Vclean', 'Taper_R0', ssub=True)
    _raises(ValueError, v3.Job, 'Q', 'Taper_R0')


# ---- 3. job selection -------------------------------------------------------------------------------------------------------------

SEL = {('04h', 'Blue'): ['Taper_R075'], ('13h', 'Blue'): ['Taper_R075'], ('15h', '78MHz'): ['Taper_R0', 'Taper_R075']}


def test_select_units():
    u = v3.select_units(_args(scopes='Blue'), SEL)
    jobs = []
    for _s, _sc, j in u:
        if j not in jobs:
            jobs.append(j)
    assert jobs == [J_R075, J_R0, J_SSUB, J_V], jobs
    assert [s for s, _sc, j in u if j == J_V] == list(v3.STAGES)
    assert [j for _s, _sc, j in v3.select_units(_args(scopes='78MHz', striation='with', products='I', stages='stage'), SEL)] == \
        [v3.Job('I', 'Taper_R075', ssub=True), v3.Job('I', 'Taper_R0', ssub=True)]
    assert [j for _s, _sc, j in v3.select_units(_args(scopes='Red', stages='coadd'), SEL)] == [v3.Job('I', 'Taper_R075'),
                                                                                                v3.Job('I', 'Taper_R0'), J_V]
    assert [j for _s, _sc, j in v3.select_units(_args(scopes='Blue', striation='without', variants='Taper_R0', stages='stage'), SEL)] \
        == [J_R0, J_V]
    assert [(s, j) for s, _sc, j in v3.select_units(_args(scopes='Blue', products='Vclean', from_stage='coadd'), SEL)] == \
        [('coadd', J_V), ('render', J_V)]
    for k, val, msg in (('scopes', 'Purple', '--scopes'), ('products', 'Q', '--products'), ('variants', 'NoTaper_R0', '--variants'),
                        ('striation', 'maybe', '--striation'), ('stages', 'caveat', 'unknown v3 stage')):
        assert msg in _raises(ValueError, v3.select_units, _args(**{k: val}), SEL), k
    assert v3.selected_lsts(SEL, 'Blue', 'Taper_R075') == ['04h', '13h'] and v3.selected_lsts(SEL, 'Blue', 'Taper_R0') == []


# ---- 4. the plain stage -----------------------------------------------------------------------------------------------------------

def _plan(masters, refusals=()):
    tiles = {lst: {'master': p, 'hdr': {'dates': ['2025-08-09', '2025-10-29'], 'BMAJ': 0.1, 'BMIN': 0.09}} for lst, p in masters.items()}
    return types.SimpleNamespace(refusals=list(refusals), tiles=tiles, masters=dict(masters),
                                 admitted=sorted(f'{l}:{d}' for l in masters for d in ('2025-08-09', '2025-10-29')))


def _masters(d, job, scope='Blue'):
    tok = CFG.mosaic.VARIANT_TOKENS[job.variant]
    return {lst: _img(os.path.join(d, 'A', lst, scope, f'{scope}_I_deep_{tok}_shflux_pbcorr_dewarped_aligned_coadd_LST{lst}_'
                                                         f'20250809-20251029_N002.fits')) for lst in LSTS}


def _fake_pb(calls, fail=()):
    def wideband_pb(scope, lst, master_path, **k):
        calls.append((scope, lst, master_path))
        if lst in fail:
            raise RuntimeError('library map missing')
        return np.full((4, 4), 0.5, np.float32), {'used_dates': ['2025-08-09', '2025-10-29'], 'chromatic': 'bs-sqbp-v1',
                                                  'residual_pb': 'residual_pb_v2-perband'}
    return wideband_pb


def _stage_plain(d, out, job, fail=(), refusals=()):
    ms = _masters(d, job)
    calls = []
    with _patched((v3, 'plan_plain', lambda scope, j: _plan(ms, refusals)), (pb_apply, 'wideband_pb', _fake_pb(calls, fail))):
        u = _unit(out, 'stage', v3.stage_unit, 'Blue', job, {})
    return u, ms, calls


def test_plain_stage_and_tile_tests():
    d = _tmp('st_')
    try:
        out = os.path.join(d, 'out')
        u, ms, calls = _stage_plain(d, out, J_R075)
        assert u.status == 'done', (u.status, u.reason)
        assert [c[1] for c in calls] == LSTS and all(c[2] == ms[c[1]] for c in calls)
        P = v3.paths(out, 'Blue', J_R075)
        h = fits.getheader(P['pb']['04h'])
        assert (h['PBSCOPE'], h['PBROBUST'], h['PBNDATE'], h['PBCHROM'], h['PBRESID']) == ('Blue', '-0.75', 2, 'bs-sqbp-v1',
                                                                                         'residual_pb_v2-perband')
        assert h['PBFOR'] == os.path.basename(ms['04h'])[:60] and 'never divide' in str(h['COMMENT'])
        assert (h['CRVAL1'], h['CDELT1'], h['RADESYS'], h['BITPIX']) == (10.0, -0.03, 'FK5', -32)
        assert [k for k in h if k in ('CTYPE1', 'EQUINOX')] == []            # only the master's grid cards present are copied
        t = json.load(open(P['tiles']))
        assert [r['lst'] for r in t['tiles']] == LSTS and t['tiles'][4]['pb_md5'] == _md5f(P['pb']['04h'])
        assert t['tiles'][4]['pbfor'] == ms['04h'] and t['tiles'][4]['kind'] == 'plain' and t['job'] == 'I_Taper_R075'
        assert [r['passed'] for r in validate.run('mosaic_v3_tiles', P['tiles'], scope='Blue', job='I_Taper_R075')] == [True, True]
        assert all(o['class'] in ('mosaic_v3_tiles', 'mosaic_weight_pb') for o in u.outputs) and len(u.outputs) == 25
        t2 = dict(t, tiles=t['tiles'][:23])
        json.dump(t2, open(P['tiles'], 'w'))
        assert T3.tiles_complete(P['tiles'])[0] is False
        json.dump(t, open(P['tiles'], 'w'))
        with open(P['pb']['09h'], 'ab') as f:
            f.write(b'\0' * 2880)
        assert T3.tiles_on_disk(P['tiles'])[0] is False
        u, _ms, _c = _stage_plain(d, out, J_R075, fail=('07h',))
        assert u.status == 'refused' and 'G5 07h' in u.reason and not os.path.exists(P['tiles']), (u.status, u.reason)
        u, _ms, _c = _stage_plain(d, out, J_R075, refusals=[('G1', '05h: night 2025-08-09 excluded')])
        assert u.status == 'refused' and 'G1 05h' in u.reason, u.reason
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---- 5. the striation-corrected and Stokes V stages ---------------------------------------------------------------------------------

def _archive(d, kind, lsts, nights=('2025-08-09', '2025-10-29'), **cards):
    out = {}
    for lst in lsts:
        if kind == 'ssub':
            n = f'Blue_I_deep_Taper_Robust-0.75_shflux_pbcorr_dewarped_aligned_coadd_ssub-rst8_LST{lst}_20250809-20251029_N002.fits'
            out[lst] = _img(os.path.join(d, 'A', lst, 'Blue', n), nights, PBMODEL=None, PBCHROM=None, PBRESID=None, **cards)
        else:
            n = f'Blue_V_deep_Taper_Robust+0.0_vclean_shflux_pbcorr_dewarped_aligned_coadd_LST{lst}_20250809-20251029_N002.fits'
            out[lst] = _img(os.path.join(d, 'A', lst, 'Blue', n), nights, **cards)
    return out


def test_substituted_stages():
    d = _tmp('sub_')
    try:
        out = os.path.join(d, 'out')
        sel = {('04h', 'Blue'): ['Taper_R075'], ('13h', 'Blue'): ['Taper_R075']}
        env = lambda ex=None: _patched((CFG.mosaic, 'INPUT_COADD_RUN', os.path.join(d, 'A')),          # noqa: E731
                                       (exclusions, 'load', lambda *a, **k: ex or _Ex()))
        with env(), _no_content_tests():
            u = _unit(out, 'stage', v3.stage_unit, 'Blue', J_SSUB, sel)
        assert u.status == 'refused' and 'no stage record for Blue I_Taper_R075' in u.reason, u.reason
        u, ms, _c = _stage_plain(d, out, J_R075)
        assert u.status == 'done'
        plain = json.load(open(v3.paths(out, 'Blue', J_R075)['tiles']))['tiles']
        with env():
            u = _unit(out, 'stage', v3.stage_unit, 'Blue', J_SSUB, sel)
        assert u.status == 'refused' and '04h: 0 archived ssub masters' in u.reason, u.reason
        ss = _archive(d, 'ssub', ['04h', '13h'])
        with env():
            u = _unit(out, 'stage', v3.stage_unit, 'Blue', J_SSUB, sel)
        assert u.status == 'done', (u.status, u.reason)
        t = json.load(open(v3.paths(out, 'Blue', J_SSUB)['tiles']))
        assert [r['kind'] for r in t['tiles']].count('ssub') == 2 and t['tiles'][4]['path'] == ss['04h']
        assert t['tiles'][4]['plain_path'] == ms['04h'] and t['tiles'][5]['path'] == ms['05h']
        assert all(r['pb'] == p['pb'] and r['pb_md5'] == p['pb_md5'] for r, p in zip(t['tiles'], plain))
        assert u.meta['n_substituted'] == 2 and t['plain_job'] == 'I_Taper_R075'
        assert [r['passed'] for r in validate.run('mosaic_v3_tiles', v3.paths(out, 'Blue', J_SSUB)['tiles'], scope='Blue',
                                                  job=J_SSUB.label)] == [True, True]
        with env(_Ex({('13h', '2025-10-29')})):
            u = _unit(out, 'stage', v3.stage_unit, 'Blue', J_SSUB, sel)
        assert u.status == 'refused' and 'G1 13h' in u.reason and '2025-10-29' in u.reason, u.reason
        # Stokes V: every LST its cleaned-V master, the Robust 0 weighting maps
        _archive(d, 'vclean', LSTS[:23])
        with env():
            u = _unit(out, 'stage', v3.stage_unit, 'Blue', J_V, {})
        assert u.status == 'refused' and 'no stage record for Blue I_Taper_R0' in u.reason, u.reason
        assert _stage_plain(d, out, J_R0)[0].status == 'done'
        with env():
            u = _unit(out, 'stage', v3.stage_unit, 'Blue', J_V, {})
        assert u.status == 'refused' and '23h: 0 archived vclean masters' in u.reason, u.reason
        vv = _archive(d, 'vclean', ['23h'], PBCORR=False)
        with env():
            u = _unit(out, 'stage', v3.stage_unit, 'Blue', J_V, {})
        assert u.status == 'refused' and 'G8 23h: PBCORR False' in u.reason, u.reason
        _archive(d, 'vclean', ['23h'])
        with env():
            u = _unit(out, 'stage', v3.stage_unit, 'Blue', J_V, {})
        assert u.status == 'done', (u.status, u.reason)
        t = json.load(open(v3.paths(out, 'Blue', J_V)['tiles']))
        r0 = json.load(open(v3.paths(out, 'Blue', J_R0)['tiles']))['tiles']
        assert all(r['kind'] == 'vclean' for r in t['tiles']) and t['tiles'][23]['path'] == vv['23h']
        assert all(r['pb'] == p['pb'] and os.path.basename(r['pb']) == 'PB_deep_Blue.fits' for r, p in zip(t['tiles'], r0))
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---- 6. coadd and render ----------------------------------------------------------------------------------------------------------

def _fake_engine(calls, skip=(), missing=()):
    m = types.ModuleType('fake_coadd_jysr')
    m.BEAM_OVERRIDE, m.SKIP_CELLS = {'x': 1}, {'y'}

    def coadd_jysr(lsts, variant, band, root=None, workroot=None, last_it=None, out_dir=None, tag='', product='I', domain='jysr'):
        job = v3.Job(product, variant, ssub=bool(tag))
        calls.append(dict(lsts=list(lsts), variant=variant, band=band, tag=tag, product=product, domain=domain,
                          override=dict(m.BEAM_OVERRIDE)))
        for lst in lsts:
            assert os.path.exists(os.path.join(root, lst, job.engine_name(band)))
            assert os.path.exists(os.path.join(root, lst, 'PB_deep_R075_Blue.fits' if job.robust == 'R075' else 'PB_deep_Blue.fits'))
            print(f'[{lst}] MISSING' if lst in missing else f'[{lst}] CORRUPT-FLAT (sigma=0) -- skipped' if lst in skip
                  else f'[{lst}] added (omega_zen=1e-5 sr)', flush=True)
        stem = f'{job.stem}_{band}'
        names = ([f'healpix_{stem}{tag}_jybeam_nside2048.fits', f'healpix_PBeff_{stem}{tag}_nside2048.fits'] if domain == 'jybeam'
                 else [f'healpix_{stem}{tag}_jysr_nside2048.fits', f'healpix_PSF_{stem}{tag}_nside2048.fits'])
        for i, n in enumerate(names):
            _hpx(os.path.join(out_dir, n), np.arange(12) + i + (10 if domain == 'jybeam' else 0), FREQ=7.0e7)
        return None, None
    m.coadd_jysr = coadd_jysr
    return m


def _fake_build(calls):
    def build(variant, hpx_dir, suffix, out, bands=(), moll_W=0, car_cdelt=0, dec_cut=None, product='I'):
        calls.append(dict(variant=variant, suffix=suffix, bands=list(bands), product=product, dec_cut=dec_cut,
                          links=sorted(os.listdir(hpx_dir))))
        label = variant if product == 'I' else 'V_deep_Taper_R0_clean'
        for b in bands:
            for proj in ('Mollweide', 'RADec'):
                fits.PrimaryHDU(np.zeros((2, 2), np.float32)).writeto(os.path.join(out, f'{b}_{label}{suffix}_{proj}.fits'))
    return build


def test_coadd_and_render_units():
    d = _tmp('co_')
    try:
        out, work = os.path.join(d, 'out'), os.path.join(d, 'work')
        sel = {('04h', 'Blue'): ['Taper_R075']}
        assert _stage_plain(d, out, J_R075)[0].status == 'done'
        _archive(d, 'ssub', ['04h'])
        with _patched((CFG.mosaic, 'INPUT_COADD_RUN', os.path.join(d, 'A')), (exclusions, 'load', lambda *a, **k: _Ex())):
            assert _unit(out, 'stage', v3.stage_unit, 'Blue', J_SSUB, sel).status == 'done'
        for job in (J_R075, J_SSUB):
            calls = []
            with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', _fake_engine(calls)), _no_content_tests():
                u = _unit(out, 'coadd', v3.coadd_unit, 'Blue', job, work)
            assert u.status == 'done', (u.status, u.reason)
            assert [(c['domain'], c['tag'], c['product'], c['override']) for c in calls] == \
                [(dm, job.tag, 'I', {}) for dm in ('jysr', 'jybeam')] and calls[0]['lsts'] == LSTS
            P = v3.paths(out, 'Blue', job)
            for k in ('jysr', 'psf', 'jybeam', 'pbeff'):
                assert os.path.exists(P[k]) and u.meta['md5'][k] == _md5f(P[k]), k
            assert os.listdir(work) == [] and u.meta['tiles_md5'] == _md5f(P['tiles'])
            assert {o['ctx']['scope'] for o in u.outputs} == {'Blue'} and u.outputs[0]['ctx']['tag'] == job.tag
        calls = []
        with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', _fake_engine(calls, skip=('13h',))), _no_content_tests():
            u = _unit(out, 'coadd', v3.coadd_unit, 'Blue', J_R075, work)
        assert u.status == 'done' and any('G9' in n and '13h' in n for n in u.notes), (u.status, u.notes)
        assert u.outputs[0]['ctx']['tiles'][13]['verdict'] == 'CORRUPT-FLAT'
        with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', _fake_engine(calls, skip=('13h',))), _no_content_tests(), \
                _patched((CFG.mosaic, 'TILE_GUARD_ACTION', 'refuse')):
            u = _unit(out, 'coadd', v3.coadd_unit, 'Blue', J_R075, work)
        assert u.status == 'refused' and 'G9' in u.reason and not os.path.exists(v3.paths(out, 'Blue', J_R075)['jysr'])
        with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', _fake_engine(calls, missing=('02h',))), _no_content_tests():
            u = _unit(out, 'coadd', v3.coadd_unit, 'Blue', J_R075, work)
        assert u.status == 'refused' and 'G3/G5' in u.reason and '02h' in u.reason, u.reason
        with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', _fake_engine(calls)), _no_content_tests():
            assert _unit(out, 'coadd', v3.coadd_unit, 'Blue', J_R075, work).status == 'done'
        # render
        bc = []
        with _patched((BP, 'build', _fake_build(bc))), _no_content_tests():
            u = _unit(out, 'render', v3.render_unit, 'Blue', J_SSUB, work)
        assert u.status == 'done', (u.status, u.reason)
        assert [(c['suffix'], c['product'], c['bands']) for c in bc] == [('_ssub-rst8_jysr', 'I', ['Blue']),
                                                                         ('_ssub-rst8_jybeam', 'I', ['Blue'])]
        assert bc[0]['links'] == sorted(os.path.basename(v3.paths(out, 'Blue', J_SSUB)[k]) for k in ('jysr', 'jybeam', 'psf'))
        assert bc[0]['dec_cut'] == CFG.mosaic_render.RENDER_DEC_CUT_DEG
        for p, _c in v3.outputs_of('render', out, 'Blue', J_SSUB):
            assert os.path.exists(p), p
        assert os.listdir(work) == []
        _stage_plain(d, out, J_R075)                              # a new tile list: the R075 coadd must be redone first
        with open(v3.paths(out, 'Blue', J_R075)['tiles'], 'a') as f:
            f.write(' ')
        with _patched((BP, 'build', _fake_build(bc))), _no_content_tests():
            u = _unit(out, 'render', v3.render_unit, 'Blue', J_R075, work)
        assert u.status == 'refused' and 'another tile list' in u.reason, (u.status, u.reason)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---- 7. the CLI -------------------------------------------------------------------------------------------------------------------

def test_cli_dispatch():
    from exopipe.phase3.config import ConfigRefusal
    from exopipe.phase3.lib.mosaic import v5
    from exopipe.phase3.steps import mosaic as SM
    ns = lambda: types.SimpleNamespace(out_root='/nonexistent')            # noqa: E731
    assert 'needs --out' in _raises(ConfigRefusal, SM.default_out, _args(generation='v3'))
    seen = []
    with _patched((v3, 'run', lambda args, run: seen.append('v3') or 0), (v5, 'run', lambda args, run: seen.append('v5') or 0),
                  (SM, '_start_gates', lambda run: None)):
        assert SM.run(_args(generation='v3', products='Vclean', variants='Taper_R0', striation='with'), ns()) == 0 and seen == ['v3']
        for k, val in (('lsts', '04h'), ('weightings', 'R075'), ('hybrid_out', '/h'), ('jobs', 'x'), ('domains', 'jysr')):
            e = _raises(ValueError, SM.run, _args(generation='v3', **{k: val}), ns())
            assert '--' + k.replace('_', '-') in e, (k, e)
        e = _raises(ValueError, SM.run, _args(generation='v3', workers=2), ns())
        assert '--workers 2' in e and 'disjoint --scopes' in e
        assert '--striation' in _raises(ValueError, SM.run, _args(generation='v5', striation='with', weightings=None, hybrid_out=None,
                                                                  lsts=None, jobs=None, domains=None, list_jobs=False,
                                                                  keep_stage=False), ns())
        assert '--striation is a --generation v3 option' in _raises(
            ValueError, SM.run, _args(generation='v2', striation='both', weightings=None, hybrid_out=None, jobs=None,
                                      list_jobs=False), ns())
        assert seen == ['v3']
    import argparse
    p = argparse.ArgumentParser()
    SM.add_arguments(p)
    a = p.parse_args(['--generation', 'v3', '--striation', 'without', '--products', 'I,Vclean', '--stages', 'stage,coadd'])
    assert (a.generation, a.striation, a.products) == ('v3', 'without', 'I,Vclean')


def test_render_vclean_names():
    """With healpy: band_products renders a Stokes V map under production's name, its beam from the V PSF map."""
    try:
        import healpy  # noqa: F401
    except ImportError:
        print('  (healpy absent: render of a Stokes V map not run)')
        return
    d = _tmp('bp_')
    try:
        n = 16
        npix = 12 * n * n
        _hpx(os.path.join(d, 'healpix_V_deep_Taper_R0_clean_Blue_jysr_nside16.fits'), np.ones(npix), BUNIT='Jy/sr', NSIDE=n,
             ORDERING='RING', PIXTYPE='HEALPIX')
        _hpx(os.path.join(d, 'healpix_PSF_V_deep_Taper_R0_clean_Blue_jysr_nside2048.fits'), np.ones(npix))
        seen = []
        with _patched((BP, '_beam_arcmin', lambda hpx_dir, variant, b, suffix='', pre=None: seen.append(pre) or 9.0)):
            os.rename(os.path.join(d, 'healpix_V_deep_Taper_R0_clean_Blue_jysr_nside16.fits'),
                      os.path.join(d, 'healpix_V_deep_Taper_R0_clean_Blue_jysr_nside2048.fits'))
            BP.build('Taper_R0', d, '_jysr', os.path.join(d, 'r'), bands=['Blue'], moll_W=360, car_cdelt=1.0, dec_cut=-30.0,
                     product='Vclean')
        assert seen == ['V_deep_Taper_R0_clean']
        for proj in ('Mollweide', 'RADec'):
            h = fits.getheader(os.path.join(d, 'r', f'Blue_V_deep_Taper_R0_clean_jysr_{proj}.fits'))
            assert h['VARIANT'] == 'V_deep_Taper_R0_clean' and h['DECMIN'] == -30.0
        _raises(ValueError, BP.build, 'Taper_R0', d, '_jysr', d, bands=['Blue'], product='Q')
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'{name}: ok')
    print('all PR-6b v3 tests passed')
