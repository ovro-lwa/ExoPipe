#!/usr/bin/env python3
"""The hybrid route, pull request PR-7a (plan/specs/hybrid.md section 4): `exopipe3 mosaic --generation v5` (lib/mosaic/v5.py, fill.py,
caveat.py, tests_v5.py) and the tilefill band field from a v5 run (steps/hybrid --mosaic-out).

Data-free (CI; the engine, the fill, the caveat map and the renderer are stand-ins - their arithmetic is production's, proven by the
cluster parity below):
  1. config: the fill and caveat limits equal production's literals (MF 31-34, 192-193; NL 1852-1857); the v5 keys;
  2. unit selection: stage order, caveat only for hybrid.CAVEAT_SCOPES, unknown stage / scope / weighting refused, --from / --until;
  3. the stage: per LST 'work' where --hybrid-out holds the cell's hybrid record, else 'pinned'; a changed tile (md5), a filled tile,
     a missing weighting map, a tile source refusal and fewer than 24 tiles refuse (HG14); the dry run computes no md5; the tile list
     content tests pass, and fail on 23 tiles and on a changed tile;
  4. the co-add: the engine is called once per domain on a link tree under its own names, tag _lsfill; outputs placed with their md5s;
     a tile the engine did not add refuses and nothing is placed;
  5. the fill: outputs placed only when both fill tests pass, md5s recorded; a FAIL leaves no file; fill_cards and footprint_kept on
     synthetic maps (each broken card fails);
  6. caveat and render units place their outputs from the stand-ins; caveat_rows; domain_cards wants the tagged run's own PSFMAP;
     the tile list chain (coadd -> caveat / fill -> render: another tiles.json refuses); the caveat refuses other weights than the
     coadd's; a changed weighting map refuses the coadd; a failed content test removes the unit's outputs (v5.run); the md5 cache;
     a tagged map name reads as its scope (coverage threshold);
  7. tilefill --mosaic-out: no done fill record refuses; a changed map, another tile list, or tiles that are not this run's unfilled
     masters refuse (HG13); the band field is made from the v5 pair under a file name of its own;
  8. the CLI: --generation v5 needs --out; v5-only flags refuse in v2; v2-only flags refuse in v5.
With healpy (the cluster): the renderer's beam lookup reads the tagged PSF map (production 14e8dba).
Cluster parity (not here): Blue Taper R-0.75, all tiles pinned - the unfilled co-add against MQ11, the caveat map against V5, the filled
maps and renders against V5 (data md5).
Run: python tests/test_mosaic_v5.py   (or under pytest)
"""
import contextlib
import csv
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
from exopipe.phase3.core import record, validate  # noqa: E402
from exopipe.phase3.core.record import Refusal  # noqa: E402
from exopipe.phase3.lib.hybrid import tilefill as TF  # noqa: E402
from exopipe.phase3.lib.mosaic import fill as MF  # noqa: E402
from exopipe.phase3.lib.mosaic import tests as MT  # noqa: E402
from exopipe.phase3.lib.mosaic import tests_v5 as T5  # noqa: E402
from exopipe.phase3.lib.mosaic import v5  # noqa: E402
from exopipe.phase3.steps import hybrid as HS  # noqa: E402

LSTS = [f'{h:02d}h' for h in range(24)]
TILE = 'Blue_I_deep_Taper_Robust-0.75_shflux_pbcorr_dewarped_aligned_coadd_lsrecon_lsfill_LST{lst}_20250809-20251029_N008.fits'


def _tmp(prefix):
    return tempfile.mkdtemp(prefix=f'exopipe3_v5_{prefix}', dir=os.environ.get('TMPDIR') or None)


def _md5f(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return str(e)
    raise AssertionError(f'{fn.__name__} did not raise {exc.__name__}')


def _img(path, data=None, **cards):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h = fits.Header()
    for k, v in cards.items():
        h[k] = v
    fits.PrimaryHDU(data if data is not None else np.ones((4, 4), np.float32), header=h).writeto(path, overwrite=True)
    return path


def _hpx(path, sig, cards0=None, **cards1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdu = fits.BinTableHDU.from_columns([fits.Column(name='SIGNAL', format='E', array=np.asarray(sig, np.float32))])
    for k, v in cards1.items():
        hdu.header[k] = v
    p0 = fits.PrimaryHDU()
    for k, v in (cards0 or {}).items():
        p0.header[k] = v
    fits.HDUList([p0, hdu]).writeto(path, overwrite=True)
    return path


@contextlib.contextmanager
def _patched(*triples):
    """(object, attribute, value) set for the block, restored after."""
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
    """A stand-in for module `name` (sys.modules and its package's attribute), restored after."""
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


def _run(out):
    return record.Run('mosaic', ['test'], out, CFG)


# ---- 1. config --------------------------------------------------------------------------------------------------------------------

def test_config_equals_production():
    F, K, V = CFG.mosaic_fill, CFG.mosaic_caveat, CFG.mosaic_v5
    assert (F.LS_MAX, F.HP_MAX, F.HP_ABS_FRAC, F.TEST_NSIDE, F.LS_SMOOTH_DEG, F.HP_SMOOTH_DEG, F.INNER_APOD_MIN) == \
        (0.02, 0.03, 0.5, 256, 30.0, 5.0, 0.9)                                                   # MF 31-34, 85-97
    assert (F.JB_SMOOTH_DEG, F.JB_BRIGHT_PERCENTILE, F.JB_PBEFF_MIN, F.JB_MED_TOL, F.JB_SPREAD) == (5.0, 70.0, 0.2, 0.01, 0.06)  # MF 137-193
    assert (K.NSIDE, K.PBEFF_MIN, K.PBEFF_MED_MAX, K.PBEFF_P99_MAX, K.PBEFF_MIN_PIXELS) == (128, 0.2, 0.01, 0.05, 1000)  # NL 1759, 1852-1857
    assert V.TAG == '_lsfill' and V.REQUIRE_N_LSTS == 24 and V.VARIANT_OF == {'R075': 'Taper_R075', 'R0': 'Taper_R0'}
    assert set(CFG.hybrid.CAVEAT_SCOPES) <= set(V.SCOPES) and len(V.SCOPES) == 19


# ---- 2. unit selection ------------------------------------------------------------------------------------------------------------

def _args(**k):
    a = dict(scopes=None, weightings=None, stages=None, from_stage=None, until_stage=None, hybrid_out=None, work=None)
    a.update(k)
    return types.SimpleNamespace(**a)


def test_select_units():
    u = v5.select_units(_args(scopes='Blue,18MHz', weightings='R075'))
    assert u == [(s, 'Blue', 'R075') for s in v5.STAGES] + [(s, '18MHz', 'R075') for s in ('stage', 'coadd', 'fill', 'render')], u
    assert v5.select_units(_args(scopes='Full', stages='fill,coadd')) == [('coadd', 'Full', 'R075'), ('fill', 'Full', 'R075'),
                                                                         ('coadd', 'Full', 'R0'), ('fill', 'Full', 'R0')]
    assert [x[0] for x in v5.select_units(_args(scopes='Blue', weightings='R0', from_stage='caveat', until_stage='fill'))] == ['caveat', 'fill']
    assert 'unknown v5 stage' in _raises(ValueError, v5.select_units, _args(stages='conform'))
    assert 'unknown scope' in _raises(ValueError, v5.select_units, _args(scopes='Purple'))
    assert 'unknown weighting' in _raises(ValueError, v5.select_units, _args(weightings='R05'))


# ---- 3. the stage -----------------------------------------------------------------------------------------------------------------

def _tree(d, n=24, filled=()):
    """24 unfilled Blue R075 tiles under d/Q with their weighting maps under d/M; a stand-in tilefill_source naming them."""
    q, m = os.path.join(d, 'Q'), os.path.join(d, 'M')
    tiles = {}
    for lst in LSTS[:n]:
        p = _img(os.path.join(q, TILE.format(lst=lst)), BMAJ=0.1, BMIN=0.09, **({'TPFILL': True} if lst in filled else {}))
        tiles[lst] = (p, _md5f(p))
        _img(os.path.join(m, lst, 'Blue', 'PB_deep_R075_Blue.fits'))
    return tiles


def _fake_source(tiles, calls, refuse=()):
    def src(out_root, lst, scope, w, rows, mode, plan_only=False, need_pb=True):
        calls.append((lst, mode, plan_only, need_pb))
        if lst in refuse:
            raise TF.TilefillRefusal(f'{lst}: refused by the source')
        if lst not in tiles:
            raise TF.TilefillRefusal(f'{lst}: no installed master')
        p, m = tiles[lst]
        return dict(source=mode, src={'pbcorr': (p, os.path.basename(p), m)}, nights=['2025-08-09', '2025-10-29'],
                    nights_from=os.path.join(out_root, 'rec.json') if mode == 'work' else '')
    return src


def _stage_env(d, tiles, calls, refuse=()):
    return _patched((HS, 'tilefill_source', _fake_source(tiles, calls, refuse)), (HS, '_manifest_rows', lambda T: {}),
                    (CFG.hybrid, 'ARCHIVE_MASTERS', os.path.join(d, 'M')))


def test_resolve_tiles():
    d = _tmp('stage_')
    try:
        tiles = _tree(d)
        ho = os.path.join(d, 'HO')
        for lst in ('03h', '17h'):                                       # two cells rebuilt in the hybrid work run
            os.makedirs(os.path.join(ho, f'{lst}_Blue'))
            open(os.path.join(ho, f'{lst}_Blue', 'R075_hybrid.json'), 'w').write('{}')
        calls = []
        with _stage_env(d, tiles, calls):
            t, probs = v5.resolve_tiles('Blue', 'R075', ho)
            assert not probs and len(t) == 24, probs
            assert {x['lst'] for x in t if x['source'] == 'work'} == {'03h', '17h'} and all(c[2] and c[3] is False for c in calls)
            assert t[0]['pb'] == os.path.join(d, 'M', '00h', 'Blue', 'PB_deep_R075_Blue.fits') and t[0]['pb_md5'] and t[0]['BMAJ'] == 0.1
            t, probs = v5.resolve_tiles('Blue', 'R075', None)            # no hybrid work run: all pinned
            assert not probs and all(x['source'] == 'pinned' for x in t)
            with open(tiles['05h'][0], 'ab') as f:                       # a changed tile
                f.write(b'\0' * 2880)
            t, probs = v5.resolve_tiles('Blue', 'R075', None)
            assert len(probs) == 1 and '05h' in probs[0] and 'md5' in probs[0], probs
            calls.clear()
            t, probs = v5.resolve_tiles('Blue', 'R075', None, plan_only=True)      # the dry run computes no md5
            assert not probs and len(t) == 24 and 'pb_md5' not in t[0]
            os.remove(os.path.join(d, 'M', '07h', 'Blue', 'PB_deep_R075_Blue.fits'))
            t, probs = v5.resolve_tiles('Blue', 'R075', None, plan_only=True)
            assert len(probs) == 1 and 'weighting map' in probs[0], probs
        tiles = _tree(_tmp('stage2_'), filled=('11h',))
        with _stage_env(os.path.dirname(os.path.dirname(tiles['00h'][0])), tiles, [], refuse=('12h',)):
            t, probs = v5.resolve_tiles('Blue', 'R075', None)
            assert len(probs) == 2 and any('filled master' in p for p in probs) and any('refused by the source' in p for p in probs), probs
        shutil.rmtree(os.path.dirname(os.path.dirname(tiles['00h'][0])), ignore_errors=True)
        tiles = _tree(_tmp('stage3_'), n=23)
        with _stage_env(os.path.dirname(os.path.dirname(tiles['00h'][0])), tiles, []):
            t, probs = v5.resolve_tiles('Blue', 'R075', None)
            assert len(t) == 23 and len(probs) == 1 and '23h' in probs[0], probs
        shutil.rmtree(os.path.dirname(os.path.dirname(tiles['00h'][0])), ignore_errors=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _stage(d, out, tiles, refuse=()):
    with _stage_env(d, tiles, [], refuse):
        run = _run(out)
        with run.unit(v5.unit_name('stage', 'Blue', 'R075'), stage='stage', scope='Blue', weighting='R075') as u:
            v5.stage_unit(u, 'Blue', 'R075', out, None)
    return u


def test_stage_unit_and_tile_tests():
    d = _tmp('stu_')
    try:
        tiles = _tree(d)
        out = os.path.join(d, 'out')
        u = _stage(d, out, tiles)
        assert u.status == 'done' and u.meta['sources'] == {'work': 0, 'pinned': 24}, (u.status, u.reason)
        assert u.admitted == ['2025-08-09', '2025-10-29']
        p = v5.paths(out, 'Blue', 'R075')['tiles']
        assert all(r['passed'] for r in validate.run('mosaic_v5_tiles', p, scope='Blue', weighting='R075'))
        t = json.load(open(p))
        t['tiles'] = t['tiles'][:23]
        bad = os.path.join(d, 'bad.json')
        json.dump(t, open(bad, 'w'))
        r = {x['name']: x['passed'] for x in validate.run('mosaic_v5_tiles', bad, scope='Blue', weighting='R075')}
        assert r == {'mosaic_v5_tiles.tiles_complete': False, 'mosaic_v5_tiles.tiles_on_disk': True}, r
        with open(tiles['09h'][0], 'ab') as f:
            f.write(b'\0' * 2880)
        r = {x['name']: x['passed'] for x in validate.run('mosaic_v5_tiles', p, scope='Blue', weighting='R075')}
        assert r['mosaic_v5_tiles.tiles_on_disk'] is False, r
        u = _stage(d, out, tiles)                                        # the stage now refuses, and leaves no tile list
        assert u.status == 'refused' and 'HG14' in u.reason and '09h' in u.reason and not os.path.exists(p), (u.status, u.reason)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---- 4. the co-add ----------------------------------------------------------------------------------------------------------------

def _fake_engine(calls, skip=()):
    m = types.ModuleType('fake_coadd_jysr')
    m.BEAM_OVERRIDE, m.SKIP_CELLS = {'x': 1}, {'y'}

    def coadd_jysr(lsts, variant, band, root=None, workroot=None, last_it=None, out_dir=None, tag='', product='I', domain='jysr'):
        calls.append(dict(lsts=list(lsts), variant=variant, band=band, tag=tag, domain=domain, override=dict(m.BEAM_OVERRIDE),
                          nside=m.NSIDE))
        for lst in lsts:
            assert os.path.exists(os.path.join(root, lst, f'I_deep_{variant}_pbcorr_{band}.fits'))
            assert os.path.exists(os.path.join(root, lst, 'PB_deep_R075_Blue.fits'))
            print(f'[{lst}] CORRUPT-FLAT (sigma=0) -- skipped' if lst in skip else f'[{lst}] added (omega_zen=1e-5 sr)', flush=True)
        stem = f'I_deep_{variant}_{band}'
        names = ([f'healpix_{stem}{tag}_jybeam_nside{m.NSIDE}.fits', f'healpix_PBeff_{stem}{tag}_nside{m.NSIDE}.fits'] if domain == 'jybeam'
                 else [f'healpix_{stem}{tag}_jysr_nside{m.NSIDE}.fits', f'healpix_PSF_{stem}{tag}_nside{m.NSIDE}.fits'])
        for i, n in enumerate(names):
            _hpx(os.path.join(out_dir, n), np.arange(12) + i + (10 if domain == 'jybeam' else 0), FREQ=7.0e7)
            print(f'WROTE {os.path.join(out_dir, n)}', flush=True)
        return None, None
    m.coadd_jysr = coadd_jysr
    return m


@contextlib.contextmanager
def _no_content_tests():
    with _patched((validate, 'run', lambda cls, path, **ctx: [])):
        yield


def _unit(out, st, fn, *a):
    run = _run(out)
    with run.unit(v5.unit_name(st, 'Blue', 'R075'), stage=st, scope='Blue', weighting='R075') as u:
        fn(u, 'Blue', 'R075', out, *a)
    return u


def _staged(d):
    tiles = _tree(d)
    out = os.path.join(d, 'out')
    assert _stage(d, out, tiles).status == 'done'
    return out


def test_coadd_unit():
    d = _tmp('co_')
    try:
        out = _staged(d)
        work = os.path.join(d, 'work')
        calls = []
        with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', _fake_engine(calls)), _no_content_tests():
            u = _unit(out, 'coadd', v5.coadd_unit, work)
        assert u.status == 'done', (u.status, u.reason)
        assert [c['domain'] for c in calls] == ['jysr', 'jybeam'] and all(c['tag'] == '_lsfill' and c['override'] == {} for c in calls)
        assert calls[0]['lsts'] == LSTS and calls[0]['variant'] == 'Taper_R075' and calls[0]['nside'] == CFG.mosaic_coadd.NSIDE
        P = v5.paths(out, 'Blue', 'R075')
        for k in ('jysr', 'psf', 'jybeam', 'pbeff'):
            assert os.path.exists(P[f'unfilled_{k}']) and u.meta['md5'][k] == _md5f(P[f'unfilled_{k}']), k
        assert os.path.basename(P['unfilled_jysr']) == 'healpix_I_deep_Taper_R075_Blue_lsfill_jysr_nside2048.fits'
        assert os.listdir(work) == [] and not [f for f in os.listdir(v5.unit_dir(out, 'Blue', 'R075')) if f.startswith('.tmp')]
        assert u.meta['tiles_md5'] == _md5f(P['tiles'])
        pb = json.load(open(P['tiles']))['tiles'][4]['pb']
        with open(pb, 'ab') as f:                                        # a weighting map changed since the stage
            f.write(b'\0' * 2880)
        with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', _fake_engine(calls)), _no_content_tests():
            u = _unit(out, 'coadd', v5.coadd_unit, work)
        assert u.status == 'refused' and '04h PB' in u.reason, (u.status, u.reason)
        _staged(d)                                                       # stage again (the new PB md5)
        calls.clear()
        with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', _fake_engine(calls, skip=('13h',))), _no_content_tests():
            u = _unit(out, 'coadd', v5.coadd_unit, work)
        assert u.status == 'refused' and 'HG14' in u.reason and '13h' in u.reason, (u.status, u.reason)
        assert not os.path.exists(P['unfilled_jysr']) and os.listdir(work) == []
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---- 5. the fill ------------------------------------------------------------------------------------------------------------------

def _cards(src, s=0.71):
    return dict(TPFILL=True, TPDONOR='LWA1 LFSS', TPDONMHZ=70, TPSCALE=s, TPLLO=CFG.hybrid_tilefill.L_LO, TPLHI=CFG.hybrid_tilefill.L_HI,
                TPSRC=os.path.basename(src), TPSRCMD5=_md5f(src))


def _fake_fill(result='PASS', jb_result='PASS'):
    def fill_file(src, out):
        with fits.open(src, memmap=False) as h:
            sig = np.asarray(h[1].data['SIGNAL']) + 1.0
        c = _cards(src)
        _hpx(out, sig, c, FREQ=7.0e7, **c)
        return dict(src=src, src_md5=_md5f(src), out=out, out_md5=_md5f(out), donor_mhz=70, s=0.71, test={'result': result}, result=result)

    def fill_jybeam_file(jb, ju, jf, psf, pbeff, out):
        with fits.open(jb, memmap=False) as h:
            sig = np.asarray(h[1].data['SIGNAL']) + 2.0
        c = dict(_cards(jb), TPJBCONV='jybeam/jysr measured')
        _hpx(out, sig, c, FREQ=7.0e7, **c)
        return dict(src=jb, src_md5=_md5f(jb), out=out, out_md5=_md5f(out), identity_rel=0.5, k_median=1.0, k_p05=0.98, k_p95=1.02, n_bright=9, result=jb_result)
    return fill_file, fill_jybeam_file


def _coadded(d):
    out = _staged(d)
    with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', _fake_engine([])), _no_content_tests():
        assert _unit(out, 'coadd', v5.coadd_unit, os.path.join(d, 'work')).status == 'done'
    return out


def test_fill_unit():
    d = _tmp('fill_')
    try:
        out = _coadded(d)
        P = v5.paths(out, 'Blue', 'R075')
        ff, fj = _fake_fill()
        with _patched((MF, 'fill_file', ff), (MF, 'fill_jybeam_file', fj)), _no_content_tests():
            u = _unit(out, 'fill', v5.fill_unit)
        assert u.status == 'done', (u.status, u.reason)
        assert u.meta['md5']['filled_jysr'] == _md5f(P['filled_jysr']) and u.meta['md5']['unfilled_jysr'] == _md5f(P['unfilled_jysr'])
        assert {t['name'] for t in u.tests} == {'HG19.fill_jysr', 'HG19.fill_jybeam'} and all(t['passed'] for t in u.tests)
        for bad in (('FAIL', 'PASS'), ('PASS', 'FAIL')):
            ff, fj = _fake_fill(*bad)
            with _patched((MF, 'fill_file', ff), (MF, 'fill_jybeam_file', fj)), _no_content_tests():
                u = _unit(out, 'fill', v5.fill_unit)
            assert u.status == 'refused' and 'HG19' in u.reason, (bad, u.status, u.reason)
            hd = os.path.dirname(P['filled_jysr'])
            assert os.listdir(hd) == [], os.listdir(hd)                  # nothing placed, no temporary left
        with open(P['unfilled_jybeam'], 'ab') as f:                      # an unfilled map changed since the co-add
            f.write(b'\0' * 2880)
        ff, fj = _fake_fill()
        with _patched((MF, 'fill_file', ff), (MF, 'fill_jybeam_file', fj)), _no_content_tests():
            u = _unit(out, 'fill', v5.fill_unit)
        assert u.status == 'refused' and 'changed since the coadd' in u.reason, (u.status, u.reason)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_fill_card_tests():
    d = _tmp('fc_')
    try:
        un = _hpx(os.path.join(d, 'u.fits'), [1.0, 2.0, np.nan, 4.0], FREQ=7.0e7)
        fb = _hpx(os.path.join(d, 'ub.fits'), [1.0, 2.0, np.nan, 4.0], FREQ=7.0e7)
        c = _cards(un, s=0.71)
        f = _hpx(os.path.join(d, 'f.fits'), [1.5, 2.5, np.nan, 4.5], c, FREQ=7.0e7, **c)
        TF.bind()
        ok = lambda fn, p, **k: fn(p, **k)[0]  # noqa: E731
        assert TF.donor_mhz(7.0e7) == 70 and ok(T5.fill_cards, f, unfilled=un, domain='jysr') and ok(T5.footprint_kept, f, unfilled=un)
        for k, v in (('TPDONMHZ', 74), ('TPSCALE', 0.0), ('TPLLO', 7), ('TPSRC', 'x.fits'), ('TPSRCMD5', '0' * 32)):
            c2 = dict(c, **{k: v})
            g = _hpx(os.path.join(d, f'g_{k}.fits'), [1.5, 2.5, np.nan, 4.5], c2, FREQ=7.0e7, **c2)
            assert not ok(T5.fill_cards, g, unfilled=un, domain='jysr'), k
        g = _hpx(os.path.join(d, 'g_prod.fits'), [1.5, 2.5, np.nan, 4.5], dict(c, PRODNAME='x'), FREQ=7.0e7, **c)
        assert not ok(T5.fill_cards, g, unfilled=un, domain='jysr')      # no PRODNAME before conform
        g = _hpx(os.path.join(d, 'g_mask.fits'), [1.5, 2.5, 3.0, 4.5], c, FREQ=7.0e7, **c)
        assert not ok(T5.footprint_kept, g, unfilled=un)                 # the footprint grew
        g = _hpx(os.path.join(d, 'g_same.fits'), [1.0, 2.0, np.nan, 4.0], c, FREQ=7.0e7, **c)
        assert not ok(T5.footprint_kept, g, unfilled=un)                 # nothing filled
        cb = dict(_cards(fb, s=0.71), TPJBCONV='x')
        jb = _hpx(os.path.join(d, 'healpix_jybeam.fits'), [1.5, 2.5, np.nan, 4.5], cb, FREQ=7.0e7, **cb)
        assert ok(T5.fill_cards, jb, unfilled=fb, filled_jysr=f, domain='jybeam')
        cb2 = dict(cb, TPSCALE=0.7)
        jb2 = _hpx(os.path.join(d, 'jb2.fits'), [1.5, 2.5, np.nan, 4.5], cb2, FREQ=7.0e7, **cb2)
        assert not ok(T5.fill_cards, jb2, unfilled=fb, filled_jysr=f, domain='jybeam')
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---- 6. caveat, render, and the shared map tests ----------------------------------------------------------------------------------

def test_caveat_and_render_units():
    from exopipe.phase3.lib.mosaic import caveat as CV
    d = _tmp('cr_')
    try:
        out = _coadded(d)
        P = v5.paths(out, 'Blue', 'R075')
        seen = {}

        def cav(tiles, variant, scope, out_dir, pbeff_mosaic, nside=None):
            seen.update(n=len(tiles), variant=variant, pbeff=pbeff_mosaic, kinds={t['kind'] for t in tiles})
            mp, tp = (os.path.join(out_dir, x) for x in CV.names(variant, scope, 128))
            _hpx(mp, np.zeros(12))
            with open(tp, 'w') as f:
                f.write('lst,kind\n')
            return dict(map=mp, table=tp, n_hybrid=len(tiles), n_taper_only=0, taper_only=[], test={'verdict': 'pass'})
        with _patched((CV, 'caveat_map', cav)), _no_content_tests(), \
                _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', _fake_engine([])):
            u = _unit(out, 'caveat', v5.caveat_unit)
        assert u.status == 'done' and seen == dict(n=24, variant='Taper_R075', pbeff=P['unfilled_pbeff'], kinds={'hybrid'}), (u.reason, seen)
        assert os.path.exists(P['caveat_map']) and os.path.basename(P['caveat_table']) == 'caveats_Taper_R075_Blue_lsfill.csv'
        ff, fj = _fake_fill()
        with _patched((MF, 'fill_file', ff), (MF, 'fill_jybeam_file', fj)), _no_content_tests():
            assert _unit(out, 'fill', v5.fill_unit).status == 'done'
        bp = types.ModuleType('fake_band_products')
        built = []

        def build(variant, hpx_dir, suffix, outd, bands=(), moll_W=0, car_cdelt=0, dec_cut=None):
            built.append((variant, suffix, tuple(bands), moll_W, round(car_cdelt * 60, 6), dec_cut, sorted(os.listdir(hpx_dir))))
            for b in bands:
                for s in ('Mollweide', 'RADec'):
                    _img(os.path.join(outd, f'{b}_{variant}{suffix}_{s}.fits'))
        bp.build = build
        with _fake_module('exopipe.phase3.lib.mosaic.band_products', bp), _no_content_tests():
            u = _unit(out, 'render', v5.render_unit, os.path.join(d, 'work'))
        assert u.status == 'done', (u.status, u.reason)
        assert [b[1] for b in built] == ['_lsfill_jysr', '_lsfill_jybeam'] and built[0][3:6] == (12000, 1.5, -30.0), built
        assert built[0][6] == sorted(v5.coadd_names('Blue', 'R075')[k] for k in ('jysr', 'jybeam', 'psf'))
        assert all(os.path.exists(p) for p, _c in v5.outputs_of('render', out, 'Blue', 'R075'))
        assert os.path.basename(P['render_jysr_mollweide']) == 'Blue_Taper_R075_lsfill_jysr_Mollweide.fits'
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_caveat_rows_and_domain_cards():
    d = _tmp('cv_')
    try:
        p = os.path.join(d, 'c.csv')
        cols = ['lst', 'kind', 'LSCAVEAT'] + [f'LSLVL{i}' for i in range(1, 6)]
        with open(p, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows([dict(lst=lst, kind='hybrid') for lst in LSTS])
        assert T5.caveat_rows(p, n_tiles=24)[0] and not T5.caveat_rows(p, n_tiles=23)[0]
        cc = CFG.mosaic_coadd
        cards = dict(BUNIT='Jy/beam', PBCORR=False, ALIGNED=False, FEATHER=f'PB {cc.PB_FEATHER_START}->{cc.PB_FLOOR} cosine weight taper',
                     WEIGHTS='PB^2/sigma_beam^2 (per-beam, point-source-optimal)', PSFMAP='healpix_PSF_I_deep_Taper_R075_Blue_nside2048.fits')
        jb = _hpx(os.path.join(d, 'healpix_I_deep_Taper_R075_Blue_lsfill_jybeam_nside2048.fits'), np.zeros(12), **cards)
        assert not MT.domain_cards(jb, domain='jybeam', tag='_lsfill')[0]  # a tagged map naming the v2 Taper mosaic's psfmap fails
        jb = _hpx(jb, np.zeros(12), **dict(cards, PSFMAP='healpix_PSF_I_deep_Taper_R075_Blue_lsfill_nside2048.fits'))
        assert MT.domain_cards(jb, domain='jybeam', tag='_lsfill')[0]      # its own psfmap passes
        jb0 = _hpx(os.path.join(d, 'healpix_I_deep_Taper_R075_Blue_jybeam_nside2048.fits'), np.zeros(12), **cards)
        assert MT.domain_cards(jb0, domain='jybeam')[0]                    # the untagged v2 maps unchanged
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_tile_list_chain_and_caveat_weights():
    from exopipe.phase3.lib.mosaic import caveat as CV
    d = _tmp('ch_')
    try:
        out = _coadded(d)
        P = v5.paths(out, 'Blue', 'R075')

        def cav(tiles, variant, scope, out_dir, pbeff_mosaic, nside=None):
            mp, tp = (os.path.join(out_dir, x) for x in CV.names(variant, scope, 128))
            _hpx(mp, np.zeros(12))
            open(tp, 'w').write('lst,kind\n')
            return dict(map=mp, table=tp, n_hybrid=24, n_taper_only=0, taper_only=[], test={'verdict': 'pass'})
        eng = _fake_engine([])
        with _patched((CV, 'caveat_map', cav), (CFG.mosaic_coadd, 'PB_FLOOR', 0.2)), _no_content_tests(), \
                _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', eng):
            u = _unit(out, 'caveat', v5.caveat_unit)
        assert u.status == 'refused' and 'weights' in u.reason and not os.path.exists(P['caveat_map']), (u.status, u.reason)
        ff, fj = _fake_fill()
        with _patched((MF, 'fill_file', ff), (MF, 'fill_jybeam_file', fj)), _no_content_tests():
            u = _unit(out, 'fill', v5.fill_unit)
        assert u.status == 'done' and u.meta['tiles_md5'] == _md5f(P['tiles']), (u.status, u.reason)
        t = json.load(open(P['tiles']))                                  # the stage run again with another hybrid work run
        t['hybrid_out'] = '/elsewhere'
        json.dump(t, open(P['tiles'], 'w'), indent=1)
        with _patched((CV, 'caveat_map', cav)), _no_content_tests(), _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', eng):
            u = _unit(out, 'caveat', v5.caveat_unit)
        assert u.status == 'refused' and 'another tile list' in u.reason, (u.status, u.reason)
        bp = types.ModuleType('fake_band_products')
        bp.build = lambda *a, **k: None
        with _fake_module('exopipe.phase3.lib.mosaic.band_products', bp), _no_content_tests():
            u = _unit(out, 'render', v5.render_unit, os.path.join(d, 'work'))
        assert u.status == 'refused' and 'another tile list' in u.reason, (u.status, u.reason)
        with _patched((MF, 'fill_file', ff), (MF, 'fill_jybeam_file', fj)), _no_content_tests():
            u = _unit(out, 'fill', v5.fill_unit)
        assert u.status == 'refused' and 'another tile list' in u.reason, (u.status, u.reason)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_failed_content_test_removes_outputs():
    d = _tmp('ft_')
    try:
        out = _staged(d)
        P = v5.paths(out, 'Blue', 'R075')
        failing = lambda cls, path, **ctx: [{'name': f'{cls}.x', 'passed': cls != 'mosaic_pbeff', 'value': 0, 'detail': ''}]  # noqa: E731
        run = _run(out)
        with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', _fake_engine([])), _patched((validate, 'run', failing)):
            v5.run(_args(stages='coadd', scopes='Blue', weightings='R075', work=os.path.join(d, 'work')), run)
        r = record.load_record(out, v5.STEP, v5.unit_name('coadd', 'Blue', 'R075'))
        assert r['status'] == 'refused' and 'mosaic_pbeff.x' in r['reason'], (r['status'], r['reason'])
        assert not any(os.path.exists(P[f'unfilled_{k}']) for k in ('jysr', 'psf', 'jybeam', 'pbeff'))
        run = _run(out)
        with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', _fake_engine([])), _no_content_tests():
            v5.run(_args(stages='coadd', scopes='Blue', weightings='R075', work=os.path.join(d, 'work')), run)
        assert all(os.path.exists(P[f'unfilled_{k}']) for k in ('jysr', 'psf', 'jybeam', 'pbeff'))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_md5_cache_and_scope_of():
    from exopipe.phase3.lib.install import steps_common as ISC
    d = _tmp('m5_')
    try:
        p = _img(os.path.join(d, 'a.fits'))
        n = []
        real = ISC.md5
        with _patched((ISC, 'md5', lambda q: n.append(q) or real(q))):
            assert v5._md5(p) == v5._md5(p) == _md5f(p) and len(n) == 1   # hashed once in the process
            with open(p, 'ab') as f:
                f.write(b'\0' * 2880)
            assert v5._md5(p) == _md5f(p) and len(n) == 2                 # a changed file is hashed again
    finally:
        shutil.rmtree(d, ignore_errors=True)
    for b in ('healpix_I_deep_Taper_R075_Blue_lsfill_jysr_nside2048.fits', 'healpix_PSF_I_deep_Taper_R0_18MHz_lsfill_nside2048.fits',
              'healpix_I_deep_Taper_R075_Blue_jysr_nside2048.fits'):
        assert MT._scope_of(f'/x/{b}') in ('Blue', '18MHz'), (b, MT._scope_of(f'/x/{b}'))


# ---- 7. tilefill --mosaic-out -----------------------------------------------------------------------------------------------------

def test_tilefill_mosaic_out():
    d = _tmp('mo_')
    try:
        mo, w = os.path.join(d, 'mos'), os.path.join(d, 'w')
        assert 'no done v5 fill unit' in _raises(TF.TilefillRefusal, HS.v5_mosaic_pair, mo, 'Blue', 'R075', w, {})
        P = v5.paths(mo, 'Blue', 'R075')
        _hpx(P['unfilled_jysr'], np.zeros(12))
        _hpx(P['filled_jysr'], np.ones(12), TPFILL=True)
        os.makedirs(os.path.dirname(P['tiles']))
        json.dump(dict(tiles=[dict(lst=lst, source='pinned', md5=f'm{lst}') for lst in LSTS]), open(P['tiles'], 'w'))
        run = _run(mo)
        with run.unit(v5.unit_name('fill', 'Blue', 'R075')) as u:
            u.meta['md5'] = {'filled_jysr': _md5f(P['filled_jysr']), 'unfilled_jysr': _md5f(P['unfilled_jysr'])}
            u.meta['tiles_md5'] = _md5f(P['tiles'])
            u.add_output(P['filled_jysr'], 'csv_table', test=False)
        want = {lst: f'm{lst}' for lst in LSTS}

        def src(out_root, lst, scope, w_, rows, mode, plan_only=False, need_pb=True):
            assert plan_only and need_pb is False
            return dict(src={'pbcorr': ('p', 'n', want[lst] if mode == 'pinned' else 'rebuilt')})
        with _patched((HS, 'tilefill_source', src)):
            assert HS.v5_mosaic_pair(mo, 'Blue', 'R075', w, {})[:2] == (P['filled_jysr'], P['unfilled_jysr'])
            want['06h'] = 'other'                                        # the installed master of 06h is not the mosaic's tile
            assert '06h' in _raises(TF.TilefillRefusal, HS.v5_mosaic_pair, mo, 'Blue', 'R075', w, {})
            want['06h'] = 'm06h'
            os.makedirs(os.path.join(w, '09h_Blue'))                     # this run rebuilt 09h; the mosaic used the pinned tile
            open(os.path.join(w, '09h_Blue', 'R075_hybrid.json'), 'w').write('{}')
            assert '09h' in _raises(TF.TilefillRefusal, HS.v5_mosaic_pair, mo, 'Blue', 'R075', w, {})
            shutil.rmtree(os.path.join(w, '09h_Blue'))
        got = []

        def bi(unf, fil, out):
            got.append((unf, fil))
            _img(out)
            return dict(out=out, out_md5=_md5f(out), filled=fil, filled_md5=_md5f(fil), unfilled=unf, unfilled_md5=_md5f(unf), s=1.0,
                        donor_mhz=70, result='PASS')
        HS._FIELDS.clear()
        with _patched((TF, 'band_increment', bi), (HS, 'tilefill_source', src)):
            r = HS.tilefill_field(w, 'Blue', 'R075', {}, mo)
        assert r['result'] == 'PASS' and got == [(P['unfilled_jysr'], P['filled_jysr'])], r
        assert os.path.basename(r['out']).startswith('field_Blue_R075_v5-') and r['out'].endswith('.fits'), r['out']   # its own name
        with open(P['unfilled_jysr'], 'ab') as f:
            f.write(b'\0' * 2880)
        with _patched((HS, 'tilefill_source', src)):
            assert 'HG13' in _raises(TF.TilefillRefusal, HS.v5_mosaic_pair, mo, 'Blue', 'R075', w, {})
        HS._FIELDS.clear()
        with _patched((TF, 'band_increment', bi), (HS, 'tilefill_source', src)):
            r = HS.tilefill_field(w, 'Blue', 'R075', {}, mo)
        assert r['result'] == 'FAIL' and 'HG13' in r['error'] and len(got) == 1, r
    finally:
        HS._FIELDS.clear()
        shutil.rmtree(d, ignore_errors=True)


# ---- 8. the CLI -------------------------------------------------------------------------------------------------------------------

def test_cli_dispatch():
    from exopipe.phase3.config import ConfigRefusal
    from exopipe.phase3.steps import mosaic as SM
    assert 'needs --out' in _raises(ConfigRefusal, SM.default_out, _args(generation='v5'))
    a = _args(generation='v2', weightings='R075', jobs=None, list_jobs=False)
    assert '--generation v5 option' in _raises(ValueError, SM.run, a, types.SimpleNamespace(out_root='/nonexistent'))
    seen = []
    with _patched((v5, 'run', lambda args, run: seen.append(args.generation) or 0), (SM, '_start_gates', lambda run: None)):
        assert SM.run(_args(generation='v5'), types.SimpleNamespace(out_root='/nonexistent')) == 0 and seen == ['v5']
        for k, val in (('list_jobs', True), ('lsts', '04h'), ('variants', 'NoTaper_R0'), ('jobs', 'x'), ('keep_stage', True)):
            e = _raises(ValueError, SM.run, _args(generation='v5', **{k: val}), types.SimpleNamespace(out_root='/nonexistent'))
            assert '--' + k.replace('_', '-') in e and seen == ['v5'], (k, e)
    import argparse
    p = argparse.ArgumentParser()
    SM.add_arguments(p)
    ns = p.parse_args(['--generation', 'v5', '--weightings', 'R0', '--hybrid-out', '/x', '--stages', 'stage,coadd'])
    assert (ns.generation, ns.weightings, ns.hybrid_out) == ('v5', 'R0', '/x')
    p = argparse.ArgumentParser()
    HS.add_arguments(p)
    assert p.parse_args(['--mosaic-out', '/m']).mosaic_out == '/m'


def test_beam_lookup_tagged():
    try:
        import healpy  # noqa: F401
    except ImportError:
        print('  (no healpy: the tagged beam lookup is checked on the cluster)')
        return
    from exopipe.phase3.lib.mosaic import band_products as BP
    d = _tmp('bm_')
    try:
        om = 1.1331 * np.radians(0.1) ** 2
        _hpx(os.path.join(d, 'healpix_PSF_I_deep_Taper_R075_Blue_lsfill_nside2048.fits'), np.full(12, om), BUNIT='sr', NSIDE=1,
             ORDERING='RING', PIXTYPE='HEALPIX', INDXSCHM='IMPLICIT')
        assert abs(BP._beam_arcmin(d, 'Taper_R075', 'Blue', '_lsfill_jysr') - 6.0) < 1e-6
        assert BP._beam_arcmin(d, 'Taper_R075', 'Blue', '_jysr') == BP.BEAMS['Blue']      # untagged: no such PSF map, the table
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith('test_') and callable(f):
            f()
            n += 1
            print(f'ok {k}', flush=True)
    print(f'{n} passed')
