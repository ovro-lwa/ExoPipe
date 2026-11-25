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
  7. tilefill --mosaic-out: no done conform record refuses (a done fill record is not enough); the conformed pair is read; a changed
     map, another tile list, or tiles that are not this run's unfilled masters refuse (HG13); the band field is made from the v5 pair
     under a file name of its own;
  8. the CLI: --generation v5 needs --out; v5-only flags refuse in v2; v2-only flags refuse in v5.
With healpy (the cluster): the renderer's beam lookup reads the tagged PSF map (production 14e8dba).
Cluster parity (not here): Blue Taper R-0.75, all tiles pinned - the unfilled co-add against MQ11, the caveat map against V5, the filled
maps and renders against V5 (data md5).
PR-7b (lib/mosaic/pkcorr.py, content.py; stages pkcorr and content):
  9. config: mosaic_content and mosaic_pkcorr equal MT / MP literals; the Dec nodes, RA bases and file names are MP's;
 10. inject_round on a synthetic tile: one source at its node, peak and integral of the restoring beam, the tile itself unchanged, no
     source where PB <= 0.12 or the pixel is blank; a second round from the same tile cache holds only its own source; curve_map is
     MP's interpolation; the map test without a curve; the catalog reader on header-only catalogs and a path with '[';
 11. the pkcorr and content units: inputs at the coadd's md5s, the coadd's engine weights, the tile list chain; outputs placed from
     the stand-ins; an injection refusal, a failed test_pkcorr, a failed TEST 1, and TEST 1 reading a file that is not the tile
     list's or skipping a tile (TEST 2) each leave the unit not done; no parent catalog refuses;
 12. the file tests of the curve, the recovery rows, the PKCORR JSON and the content report (each broken file fails).
Cluster parity (not here): TEST 1 values = V5 tables/summary.csv rows; the PKCORR map = the archived V5 map (SIGNAL bit-identical).
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


def _hpx_cards(npix):
    """The HEALPix cards of the pipeline's map writers (coadd_jysr 257, tilefill 141, caveat 120, pkcorr 241); NSIDE and INDXSCHM
    when the pixel count is a full sky. The header validator checks a HEALPix map on its HDU 1 table (decision C-08)."""
    c = dict(PIXTYPE='HEALPIX', ORDERING='RING', COORDSYS='C')
    k = int(round((npix / 12.0) ** 0.5))
    if k > 0 and 12 * k * k == npix:
        c.update(NSIDE=k, INDXSCHM='IMPLICIT')
    return c


def _hpx(path, sig, cards0=None, **cards1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdu = fits.BinTableHDU.from_columns([fits.Column(name='SIGNAL', format='E', array=np.asarray(sig, np.float32))])
    for k, v in dict(_hpx_cards(len(hdu.data)), **cards1).items():
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


def _pinned_union(par):
    """(section, key, value) pinning the scratch parent union `par`: its MANIFEST.md5 written now, its md5 the one pin of the digest
    gate (core/manifest.verify_union)."""
    from exopipe.phase3.core import manifest as MFS
    return (CFG.paths, 'PARENT_UNION_MANIFEST_MD5', {'test-union': MFS.write_manifest(par, os.path.join(par, MFS.MANIFEST))[1]})


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
    assert v5.STAGES == ('stage', 'coadd', 'pkcorr', 'caveat', 'fill', 'render', 'content', 'conform', 'tables')    # spec 3.1
    per = [s for s in v5.STAGES if s != 'tables']
    assert u == [(s, 'Blue', 'R075') for s in per] + [(s, '18MHz', 'R075') for s in per if s != 'caveat'] + [('tables', None, None)], u
    assert v5.select_units(_args(scopes='Blue', weightings='R075', stages='tables')) == [('tables', None, None)]
    assert v5.select_units(_args(scopes='Blue', weightings='R0', stages='content,conform')) == [('content', 'Blue', 'R0'),
                                                                                               ('conform', 'Blue', 'R0')]
    assert v5.select_units(_args(scopes='Full', stages='fill,coadd')) == [('coadd', 'Full', 'R075'), ('fill', 'Full', 'R075'),
                                                                         ('coadd', 'Full', 'R0'), ('fill', 'Full', 'R0')]
    assert [x[0] for x in v5.select_units(_args(scopes='Blue', weightings='R0', from_stage='caveat', until_stage='fill'))] == ['caveat', 'fill']
    assert 'unknown v5 stage' in _raises(ValueError, v5.select_units, _args(stages='push'))
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
            rd = os.path.join(ho, '_records', 'hybrid')                  # the work run refused 09h, has 10h running
            os.makedirs(rd)
            for name, st in (('wideband.09h_Blue_R075', 'refused'), ('hybrid.10h_Blue_R075', 'running'),
                             ('hybrid.03h_Blue_R075', 'done'), ('wideband.09h_Blue_R0', 'refused'),
                             ('wideband.17h_Blue_R075', 'refused'), ('hybrid.17h_Blue_R075', 'done'),   # a stale work hybrid
                             ('tilefill.12h_Blue_R075', 'done')):                                       # a pinned tilefill
                json.dump(dict(status=st, reason='synthetic'), open(os.path.join(rd, name + '.json'), 'w'))
            json.dump(dict(status='done', staged_from={}), open(os.path.join(rd, 'stack.11h_Blue_R075.json'), 'w'))   # an input
            t, probs = v5.resolve_tiles('Blue', 'R075', ho)
            assert [p[:13] for p in probs] == ['09h: refused ', '10h: unfinish', '17h: refused '], probs
            assert all('REFUSED_INSTALLED' in p for p in probs) and len(t) == 21, probs
            keep = {(x, 'Blue', 'R075'): 'keep' for x in ('09h', '10h', '17h')}
            with _patched((CFG.hybrid_unit, 'REFUSED_INSTALLED', keep)):
                t, probs = v5.resolve_tiles('Blue', 'R075', ho)
            assert len(probs) == 1 and probs[0].startswith('10h: unfinished'), probs       # keep rules a refused unit only
            kept = {x['lst']: x for x in t if x['caveat']}
            assert sorted(kept) == ['09h', '17h'] and all(x['source'] == 'pinned' and x['caveat'].startswith('kept by ruling')
                                                          for x in kept.values()), t
            assert {x['lst'] for x in t if x['source'] == 'work'} == {'03h'}, t
            shutil.rmtree(os.path.join(ho, '_records'))
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
            _hpx(os.path.join(out_dir, n), np.arange(12) + i + (10 if domain == 'jybeam' else 0), FREQ=7.0e7,
                 **({} if i else {'BUNIT': 'Jy/beam' if domain == 'jybeam' else 'Jy/sr'}))     # the engine's BUNIT (coadd_jysr 290)
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


def test_caveat_placement_refusal():
    """Placement P4 fix round, the handler of caveat_unit: a placement refusal of the caveat TABLE is the unit's refusal with the
    placement message and neither product stays - (1) its attributes refused (atomic_many places neither); (2) its rename fails
    after the map was renamed into place (atomic_many's phase 2 is not all-or-none): the unit itself removes the placed map
    before the refusal is recorded. No temporary file is left either way."""
    from exopipe.phase3.core import placement as PL
    from exopipe.phase3.lib.mosaic import caveat as CV
    import errno
    d = _tmp('cpr_')
    try:
        out = _coadded(d)
        P = v5.paths(out, 'Blue', 'R075')
        cdir = os.path.dirname(P['caveat_map'])

        def cav(tiles, variant, scope, out_dir, pbeff_mosaic, nside=None):      # the stand-in writes as caveat_map does
            mp, tp = (os.path.join(out_dir, x) for x in CV.names(variant, scope, 128))
            with PL.atomic_many([mp, tp]) as tmp:
                _hpx(tmp[mp], np.zeros(12))
                with open(tmp[tp], 'w') as f:
                    f.write('lst,kind\n')
            return dict(map=mp, table=tp, n_hybrid=len(tiles), n_taper_only=0, taper_only=[], test={'verdict': 'pass'})
        foreign = next((g for g in (0, 1, 2, 3, 4, 65534) if g not in os.getgroups() and g != os.getegid()), None)
        real_ta, real_replace = PL.target_attrs, os.replace
        cases = []
        if os.geteuid() != 0 and foreign is not None:
            cases.append(('attributes', (PL, 'target_attrs', lambda dst, like=None: dict(real_ta(dst, like), gid=foreign)
                                         if dst == P['caveat_table'] else real_ta(dst, like))))
        else:
            print('  UNTESTED (attribute refusal of the caveat table): running as root or a member of every candidate group')

        seen = []

        def table_rename_fails(src, dst):            # the map is renamed into place before the table's rename fails
            seen.append((dst, os.path.exists(P['caveat_map'])))
            if dst == P['caveat_table']:
                raise OSError(errno.EIO, 'Input/output error')
            return real_replace(src, dst)
        cases.append(('rename', (os, 'replace', table_rename_fails)))
        for what, patch in cases:
            seen.clear()
            with _patched((CV, 'caveat_map', cav), patch), _no_content_tests(), \
                    _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', _fake_engine([])):
                u = _unit(out, 'caveat', v5.caveat_unit)
            assert os.replace is real_replace and PL.target_attrs is real_ta
            assert u.status == 'refused' and u.reason.startswith('placement:') and P['caveat_table'] in u.reason, (what, u.status, u.reason)
            if what == 'rename':
                mine = [x for x in seen if x[0] in (P['caveat_map'], P['caveat_table'])]      # (the unit record is placed too)
                assert mine == [(P['caveat_map'], False), (P['caveat_table'], True)], seen     # the map WAS placed
                assert f"already placed: {[P['caveat_map']]}" in u.reason, u.reason
            else:
                assert 'none of the 2 products placed' in u.reason, u.reason
            assert not os.path.exists(P['caveat_map']) and not os.path.exists(P['caveat_table']), (what, os.listdir(cdir))
            assert os.listdir(cdir) == [], (what, os.listdir(cdir))
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
        assert 'no done v5 conform unit' in _raises(TF.TilefillRefusal, HS.v5_mosaic_pair, mo, 'Blue', 'R075', w, {})
        P = v5.paths(mo, 'Blue', 'R075')
        cj = os.path.join(P['conform'], 'Blue', 'I', 'healpix', ARCH_JYSR)            # the conformed pair (the archived names)
        cu = os.path.join(P['conform'], 'unfilled', 'Blue', 'I', 'healpix', ARCH_JYSR)
        _hpx(cu, np.zeros(12))
        _hpx(cj, np.ones(12), TPFILL=True)
        _hpx(P['filled_jysr'], np.ones(12), TPFILL=True)                              # the fill stage's work file: not read
        os.makedirs(os.path.dirname(P['tiles']))
        json.dump(dict(tiles=[dict(lst=lst, source='pinned', md5=f'm{lst}') for lst in LSTS]), open(P['tiles'], 'w'))
        run = _run(mo)
        with run.unit(v5.unit_name('fill', 'Blue', 'R075')) as u:                     # a done fill unit is not enough
            u.add_output(P['filled_jysr'], 'csv_table', test=False)
        assert 'no done v5 conform unit' in _raises(TF.TilefillRefusal, HS.v5_mosaic_pair, mo, 'Blue', 'R075', w, {})
        with run.unit(v5.unit_name('conform', 'Blue', 'R075')) as u:
            u.meta['products'] = {'jysr': {'path': cj, 'md5': _md5f(cj)}, 'unfilled_jysr': {'path': cu, 'md5': _md5f(cu)}}
            u.meta['tiles_md5'] = _md5f(P['tiles'])
            u.add_output(cj, 'csv_table', test=False)
        want = {lst: f'm{lst}' for lst in LSTS}

        def src(out_root, lst, scope, w_, rows, mode, plan_only=False, need_pb=True):
            assert plan_only and need_pb is False
            return dict(src={'pbcorr': ('p', 'n', want[lst] if mode == 'pinned' else 'rebuilt')})
        with _patched((HS, 'tilefill_source', src)):
            assert HS.v5_mosaic_pair(mo, 'Blue', 'R075', w, {}) == (cj, cu, _md5f(cj), _md5f(cu))
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
        assert r['result'] == 'PASS' and got == [(cu, cj)], r
        assert os.path.basename(r['out']).startswith('field_Blue_R075_v5-') and r['out'].endswith('.fits'), r['out']   # its own name
        with open(cu, 'ab') as f:
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
        e = _raises(ValueError, SM.run, _args(generation='v5', workers=4), types.SimpleNamespace(out_root='/nonexistent'))
        assert '--workers 4' in e and 'disjoint --scopes' in e and seen == ['v5'], e       # refused, never silently serial
        assert SM.run(_args(generation='v5', workers=1), types.SimpleNamespace(out_root='/nonexistent')) == 0 and seen == ['v5'] * 2
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


# ---- 9-12. PR-7b: pkcorr and content -----------------------------------------------------------------------------------------------

def test_config_pkcorr_content_equal_production():
    from exopipe.phase3.lib.mosaic import pkcorr as PK
    K, T = CFG.mosaic_pkcorr, CFG.mosaic_content
    assert (K.S_INJ, K.DEC_START, K.DEC_STOP, K.DEC_STEP, K.RA_STEP, K.ROUND_OFF, K.INJ_PB_MIN) == (50.0, -30.0, 89.0, 2.0, 47.0, 3.0,
                                                                                                     0.12)     # MP 32-35, 78
    assert (K.PEAK_RADIUS_FWHM, K.PEAK_RADIUS_MIN_DEG, K.OMEGA_RADIUS_DEG) == (0.5, 0.05, 0.2)                  # MP 89-93
    assert (K.ROUNDS, K.MIN_ROUNDS, K.REC_MIN, K.REC_MAX, K.MAP_TOL) == (5, 3, 0.5, 1.2, 1e-6)                 # MP 148-158, 167
    assert (T.N_SOURCES, T.DEC_MIN, T.ISOLATION_BEAMS, T.APERTURE_BEAMS) == (200, -30.0, 5.0, 2.0)             # MT 24, 27-48, 114, 202
    assert (T.MIN_SOURCES, T.MED_TOL, T.FRAC5_MIN) == (20, 0.01, 0.90) and not hasattr(T, 'PROV_BYTES')        # MT 137; TEST 2: v5
    assert T.PARENTS is None and CFG.paths.PARENT_UNION == '/lustre/gh/main/phase3/deep_parent_catalogs_union_20261118T0130'  # MT 201; B15
    assert T.RULED == {'DEC_MIN': 'R-GREGG-20261016-MOSAIC-DEC-MIN-30'} and K.RULED == {'DEC_START': 'R-GREGG-20261016-MOSAIC-DEC-MIN-30'}
    decs, ra = PK.nodes()
    assert np.array_equal(decs, np.arange(-30.0, 89.0, 2.0)) and len(decs) == 60
    assert np.array_equal(ra, np.array([(i * 47.0) % 360.0 for i in range(60)]))
    assert PK.names('Taper_R075', 'Blue', '_lsfill', 2048) == {
        'map': 'healpix_PKCORR_I_deep_Taper_R075_Blue_lsfill_nside2048.fits', 'recovery': 'peak_recovery_Taper_R075_Blue_lsfill.csv',
        'curve': 'peak_corr_curve_Taper_R075_Blue_lsfill.csv', 'json': 'pkcorr_Taper_R075_Blue_lsfill.json'}


def _sin_tile(path, data, ra0=330.0, dec0=30.0, cd=0.25, **cards):
    n = data.shape[0]
    return _img(path, data, CTYPE1='RA---SIN', CTYPE2='DEC--SIN', CRVAL1=ra0, CRVAL2=dec0, CRPIX1=n / 2, CRPIX2=n / 2,
                CDELT1=-cd, CDELT2=cd, CUNIT1='deg', CUNIT2='deg', BMAJ=2.0, BMIN=1.5, BPA=30.0, **cards)


def _paths_module():
    m = types.ModuleType('fake_coadd_jysr')

    def _paths(root, lst, variant, band, product='I'):
        return (os.path.join(root, lst, f'I_deep_{variant}_pbcorr_{band}.fits'), os.path.join(root, lst, 'PB_deep_R075_Blue.fits'))
    m._paths = _paths
    return m


def test_inject_round():
    from exopipe.phase3.lib.mosaic import pkcorr as PK
    d = _tmp('inj_')
    try:
        root, dst = os.path.join(d, 'root'), os.path.join(d, 'dst')
        base = np.zeros((64, 64), np.float32)
        lsts = ['00h', '01h', '02h']
        pbs = {'00h': np.ones((64, 64), np.float32), '01h': np.full((64, 64), 0.12, np.float32), '02h': np.ones((64, 64), np.float32)}
        for lst in lsts:
            dat = base.copy()
            if lst == '02h':
                dat[:] = np.nan                                          # a blank tile: nothing injected
            _sin_tile(os.path.join(root, lst, 'I_deep_Taper_R075_pbcorr_Blue.fits'), dat)
            _img(os.path.join(root, lst, 'PB_deep_R075_Blue.fits'), pbs[lst])
        before = _md5f(os.path.join(root, '00h', 'I_deep_Taper_R075_pbcorr_Blue.fits'))
        with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', _paths_module()):
            tiles = PK.load_tiles(root, 'Blue', 'Taper_R075', lsts)
            n = PK.inject_round(tiles, dst, 0.0)
            n2 = PK.inject_round(tiles, os.path.join(d, 'dst2'), 3.0)      # round 1 from the same cache
        assert sorted(tiles) == lsts and not tiles['00h'][3].flags.writeable and float(np.nansum(tiles['00h'][3])) == 0.0
        i30 = 30                                                         # node 30: Dec +30, RA 30 x 47 mod 360 = 330 (the tile center)
        assert n[i30] == 1 and n.sum() == 1, (np.nonzero(n), n.sum())    # 01h: PB <= 0.12; 02h: blank; other nodes off the 16-deg tile
        assert _md5f(os.path.join(root, '00h', 'I_deep_Taper_R075_pbcorr_Blue.fits')) == before
        out = fits.getdata(os.path.join(dst, '00h', 'I_deep_Taper_R075_pbcorr_Blue.fits'))
        assert out.dtype.kind == 'f' and out.dtype.itemsize == 4
        fx, fn = 2.0 / 0.25, 1.5 / 0.25                                  # FWHM in pixels (BMAJ 2, BMIN 1.5 deg; 0.25 deg pixels)
        assert abs(out[31, 31] - 50.0) < 1e-3 and out.max() == out[31, 31], (out[31, 31], out.max())   # on the reference pixel
        integral = 50.0 * np.pi / (4 * np.log(2)) * fx * fn
        assert abs(out.sum() / integral - 1) < 1e-3, (out.sum(), integral)
        assert float(np.nansum(fits.getdata(os.path.join(dst, '01h', 'I_deep_Taper_R075_pbcorr_Blue.fits')))) == 0.0
        assert os.path.islink(os.path.join(dst, '00h', 'PB_deep_R075_Blue.fits'))
        assert os.readlink(os.path.join(dst, '00h', 'PB_deep_R075_Blue.fits')) == os.path.realpath(
            os.path.join(root, '00h', 'PB_deep_R075_Blue.fits'))
        out2 = fits.getdata(os.path.join(d, 'dst2', '00h', 'I_deep_Taper_R075_pbcorr_Blue.fits'))
        assert n2[i30] == 1 and abs(out2.sum() / integral - 1) < 1e-3 and out2.max() < 50.0 + 1e-3, (n2[i30], out2.sum())  # one source
        assert out2[31, 31] < out[31, 31] - 1                            # moved 3 deg in RA: not the round-0 source
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _hpx_read(path):
    return np.asarray(fits.getdata(path, 1)['SIGNAL'], np.float32).ravel()


def _fake_run_band(calls, verdict='pass', refuse=None):
    from exopipe.phase3.lib.mosaic import pkcorr as PK

    def run_band(root, band, variant, out, lsts, rounds, tag, mos_c, psf, scratch, nside, record_dir=None):
        calls.append(dict(root=root, band=band, variant=variant, lsts=list(lsts), rounds=rounds, tag=tag, nside=nside, n=len(mos_c),
                          links=sorted(os.listdir(os.path.join(root, lsts[0]))), scratch=os.path.isdir(scratch)))
        if refuse:
            raise PK.InjectionRefusal(refuse)
        nm = PK.names(variant, band, tag, nside)
        P = {k: os.path.join(out, f) for k, f in nm.items()}
        _hpx(P['map'], np.full(12, 0.9), NSIDE=nside, ORDERING='RING', BUNIT='', SINJ=50.0, NROUNDS=rounds)
        for k in ('curve', 'recovery'):
            open(P[k], 'w').write('x\n')
        res = dict(rounds=rounds, rec_median=0.9, rec_min=0.9, rec_max=0.9, ra_scatter_median_mad=0.01,
                   map=os.path.join(record_dir, nm['map']), test_pkcorr=dict(verdict=verdict, nodes=60, missing_or_few_rounds=[],
                                                                            out_of_range=[], max_map_err=0.0))
        json.dump(res, open(P['json'], 'w'))
        return P, res
    return run_band


def test_pkcorr_unit():
    from exopipe.phase3.lib.mosaic import pkcorr as PK
    d = _tmp('pk_')
    try:
        with _patched((CFG.mosaic_coadd, 'NSIDE', 1)):
            out = _coadded(d)
            work = os.path.join(d, 'work')
            P = v5.paths(out, 'Blue', 'R075')
            calls = []
            eng = _fake_engine([])
            with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', eng), _no_content_tests(), \
                    _patched((PK, 'run_band', _fake_run_band(calls)), (PK, 'read_map', _hpx_read)):
                u = _unit(out, 'pkcorr', v5.pkcorr_unit, work)
            assert u.status == 'done', (u.status, u.reason)
            c = calls[0]
            assert (c['band'], c['variant'], c['rounds'], c['tag'], c['nside'], c['n'], c['lsts']) == (
                'Blue', 'Taper_R075', 5, '_lsfill', 1, 12, LSTS) and c['scratch'], c
            assert c['links'] == ['I_deep_Taper_R075_pbcorr_Blue.fits', 'PB_deep_R075_Blue.fits'], c['links']
            for k in ('map', 'curve', 'recovery', 'json'):
                assert os.path.exists(P[f'pkcorr_{k}']) and u.meta['md5'][k] == _md5f(P[f'pkcorr_{k}']), k
            assert os.path.basename(P['pkcorr_map']) == 'healpix_PKCORR_I_deep_Taper_R075_Blue_lsfill_nside1.fits'
            assert json.load(open(P['pkcorr_json']))['map'] == P['pkcorr_map']
            assert os.listdir(work) == [] and not [f for f in os.listdir(v5.unit_dir(out, 'Blue', 'R075')) if f.startswith('.tmp')]
            assert u.meta['tiles_md5'] == _md5f(P['tiles']) and u.meta['pkcorr']['rounds'] == 5
            # an injection refusal, then a failed test_pkcorr: the unit does not end done and nothing stays in place (v5.run)
            with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', eng), _no_content_tests(), \
                    _patched((PK, 'run_band', _fake_run_band([], refuse='tiles not added')), (PK, 'read_map', _hpx_read)):
                u = _unit(out, 'pkcorr', v5.pkcorr_unit, work)
            assert u.status == 'refused' and 'tiles not added' in u.reason and not os.path.exists(P['pkcorr_map']), (u.status, u.reason)
            with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', eng), _no_content_tests(), \
                    _patched((PK, 'run_band', _fake_run_band([], verdict='fail')), (PK, 'read_map', _hpx_read)):
                v5.run(_args(stages='pkcorr', scopes='Blue', weightings='R075', work=work), _run(out))
            r = record.load_record(out, v5.STEP, v5.unit_name('pkcorr', 'Blue', 'R075'))
            assert r['status'] != 'done' and 'test_pkcorr' in str(r['reason']), (r['status'], r['reason'])
            assert not any(os.path.exists(P[f'pkcorr_{k}']) for k in ('map', 'curve', 'recovery', 'json'))
            # the clean map changed since the coadd; other engine weights than the coadd's
            with open(P['unfilled_jysr'], 'ab') as f:
                f.write(b'\0' * 2880)
            with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', eng), _no_content_tests(), \
                    _patched((PK, 'run_band', _fake_run_band([])), (PK, 'read_map', _hpx_read)):
                u = _unit(out, 'pkcorr', v5.pkcorr_unit, work)
            assert u.status == 'refused' and 'changed since the coadd' in u.reason, (u.status, u.reason)
            out = _coadded(d)
            with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', eng), _no_content_tests(), \
                    _patched((PK, 'run_band', _fake_run_band([])), (PK, 'read_map', _hpx_read),
                             (CFG.mosaic_coadd, 'PB_FEATHER_START', float(CFG.mosaic_coadd.PB_FEATHER_START) + 0.05)):
                u = _unit(out, 'pkcorr', v5.pkcorr_unit, work)
            assert u.status == 'refused' and 'weights now' in u.reason, (u.status, u.reason)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _report(n=24, med=1.002, fr=0.95, nsrc=150, links=True, files=None):
    pv = [dict(lst=x, file=(files or {}).get(x, f'/x/{x}.fits'), pb=f'/x/pb_{x}.fits', md5='0' * 32, symlink=links) for x in LSTS[:n]]
    r = dict(map='m', variant='Taper_R075', band='Blue', n_tiles=n, n_sources=nsrc, median_ratio=med, p05=0.97, p95=1.03,
             frac_within_5pct=fr, provenance=pv, result='PASS' if abs(med - 1) <= 0.01 and fr >= 0.9 and nsrc >= 20 and links else 'FAIL')
    return r


def test_content_unit():
    from exopipe.phase3.lib.mosaic import content as CT
    d = _tmp('ct_')
    try:
        par = os.path.join(d, 'parents')
        os.makedirs(par)
        open(os.path.join(par, 'master_R075_00h_Blue.gaul.csv'), 'w').write('# none\n')
        with _patched((CFG.mosaic_coadd, 'NSIDE', 1), (CFG.mosaic_content, 'PARENTS', par), _pinned_union(par)):
            out = _coadded(d)
            work = os.path.join(d, 'work')
            P = v5.paths(out, 'Blue', 'R075')
            listed = json.load(open(P['tiles']))['tiles']
            good = [dict(lst=r['lst'], file=os.path.realpath(r['path']), pb=os.path.realpath(r['pb']), symlink=True) for r in listed]
            calls = []

            def fake_run(maps, root, variant, band, parents, med=1.002, read=good):
                calls.append(([os.path.basename(m) for m in maps.values()], variant, band, list(maps), parents, sorted(os.listdir(root))))
                out = {}
                for dom in maps:
                    r = _report(med=med if dom == 'jybeam' else 1.002)
                    r.pop('provenance')
                    out[dom] = dict(r, domain=dom, read=[dict(x) for x in read])
                return out
            eng = _fake_engine([])
            with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', eng), _patched((CT, 'run', fake_run)):
                u = _unit(out, 'content', v5.content_unit, work)
            assert u.status == 'done', (u.status, u.reason)
            assert len(calls) == 1 and calls[0][3] == ['jysr', 'jybeam']                     # one reading of the tiles, both domains
            assert calls[0][0] == [os.path.basename(P['unfilled_jysr']), os.path.basename(P['unfilled_jybeam'])]
            assert calls[0][1:3] == ('Taper_R075', 'Blue') and calls[0][4] == par and calls[0][5] == LSTS
            rep = json.load(open(P['content']))
            assert rep['jysr']['provenance_matches_tiles'] and os.path.basename(P['content']) == 'content_I_deep_Taper_R075_Blue_lsfill.json'
            assert [x['md5'] for x in rep['jybeam']['provenance']] == [r['md5'] for r in listed] and 'read' not in rep['jysr']
            assert os.listdir(work) == []
            # TEST 1 fails in Jy/beam; TEST 2: a PB that is not the listed one, a tile TEST 1 skipped
            wrong_pb = [dict(x, pb='/elsewhere/pb.fits') if x['lst'] == '05h' else x for x in good]
            for kw, name in ((dict(med=1.03), 'TEST1'), (dict(read=wrong_pb), 'TEST2'), (dict(read=good[1:]), 'TEST2')):
                bad = (lambda kw: lambda *a: fake_run(*a, **kw))(kw)
                with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', eng), _patched((CT, 'run', bad)):
                    v5.run(_args(stages='content', scopes='Blue', weightings='R075', work=work), _run(out))
                r = record.load_record(out, v5.STEP, v5.unit_name('content', 'Blue', 'R075'))
                assert r['status'] != 'done' and name in str(r['reason']) and not os.path.exists(P['content']), (name, r['status'], r['reason'])
            with _patched((CFG.mosaic_content, 'PARENTS', os.path.join(d, 'none'))), \
                    _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', eng), _patched((CT, 'run', fake_run)):
                u = _unit(out, 'content', v5.content_unit, work)
            assert u.status == 'refused' and 'no parent catalog' in u.reason, (u.status, u.reason)
            # the digest gate: a catalog changed after its MANIFEST.md5 (M4), a file the manifest does not list (M3), an unpinned
            # manifest (M2), no manifest (M1) - each refuses before TEST 1 runs
            cat = os.path.join(par, 'master_R075_00h_Blue.gaul.csv')
            for change, check in ((lambda: open(cat, 'a').write('# changed\n'), 'M4'),
                                  (lambda: open(os.path.join(par, 'master_R075_01h_Blue.gaul.csv'), 'w').write('# new\n'), 'M3'),
                                  (lambda: open(os.path.join(par, 'MANIFEST.md5'), 'a').write('\n'), 'M2'),
                                  (lambda: os.remove(os.path.join(par, 'MANIFEST.md5')), 'M1')):
                change()
                del calls[:]
                with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', eng), _patched((CT, 'run', fake_run)):
                    u = _unit(out, 'content', v5.content_unit, work)
                assert u.status == 'refused' and f'digest gate: {check}' in u.reason and not calls, (check, u.status, u.reason)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_curve_map_and_catalog_reader():
    try:
        import healpy as hp
    except ImportError:                                                          # CI has no healpy: the catalog reader only
        hp = None
    from exopipe.phase3.lib.mosaic import content as CT
    from exopipe.phase3.lib.mosaic import pkcorr as PK
    cd, cm = [-30.0, 0.0, 40.0, 88.0], [0.80, 0.85, 0.90, 0.95]
    for nside in (4, 16) if hp else ():
        th, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        assert np.array_equal(PK.curve_map(cd, cm, nside), np.interp(90.0 - np.degrees(th), cd, cm))   # MP 129-130
    d = _tmp('cmap_')
    try:
        mp = os.path.join(d, 'pk.fits')
        if hp is None:
            print('  (healpy absent: curve_map and the map-without-curve test not run)')
        else:
            with _patched((CFG.mosaic_coadd, 'NSIDE', 1)):
                _hpx(mp, PK.curve_map(cd, cm, 1).astype(np.float32), NSIDE=1, ORDERING='RING', BUNIT='', SINJ=50.0, NROUNDS=5)
                ok, err, msg = T5.pkcorr_map_is_curve(mp, rounds=5)                # no curve: no comparison, no TypeError
                assert ok and err is None and 'no curve' in msg, (ok, err, msg)
        par = os.path.join(d, 'cats[1]')                                         # a '[' in the directory name
        os.makedirs(par)
        open(os.path.join(par, 'master_R075_00h_Blue.gaul.csv'), 'w').write('# header only\n\n')
        assert CT._sources(par, 'Blue', 200, 0.3).shape == (0, 3)                  # no row: empty, not an IndexError
        row = lambda ra, dec, pk: ','.join(['x'] * 4 + [str(ra), 'x', str(dec)] + ['x'] * 3 + [str(pk)]) + '\n'  # noqa: E731
        open(os.path.join(par, 'master_R075_01h_Blue.gaul.csv'), 'w').write(
            row(10.0, 20.0, 5.0) + row(10.5, 20.0, 4.0) + row(30.0, 20.0, 3.0) + row(30.0, -40.0, 9.0))
        S = CT._sources(par, 'Blue', 200, 0.3)                                     # 0.5 deg < 5 beams: the fainter neighbor dropped
        assert S.tolist() == [[10.0, 20.0, 5.0], [30.0, 20.0, 3.0]], S.tolist()   # Dec -40 < DEC_MIN + 3 beams
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_audit_mosaic_caveat_table():
    """The archive audit reads a HEALPix table without a SIGNAL column (the caveat map: LEVEL, TAPERONLY) and still finds an
    all-NaN map; a SIGNAL map is read from SIGNAL."""
    from astropy.io import fits
    from exopipe.phase3.lib.archive.push import audit_mosaic
    d = _tmp('audcav_')
    try:
        name = 'Blue_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_feather_lsfill_caveat_allsky_20241218-20260419_N024.fits'
        cards = dict(PBCORR=True, NSTACK=24, **{'DATE-OBS': '2024-12-18T00:00:00', 'DATE-END': '2026-04-19T00:00:00'})
        def write(cols, path):
            hdu = fits.BinTableHDU.from_columns([fits.Column(name=k, format='E', array=v) for k, v in cols])
            for k, v in cards.items():
                hdu.header[k] = v
            fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)
        ok = os.path.join(d, 'ok')
        os.makedirs(ok)
        write([('LEVEL', np.array([0, 1, 2] * 4, np.float32)), ('TAPERONLY', np.zeros(12, np.float32))], os.path.join(ok, name))
        assert audit_mosaic(os.path.join(ok, name)) == [], audit_mosaic(os.path.join(ok, name))
        nan = os.path.join(d, 'nan')
        os.makedirs(nan)
        write([('LEVEL', np.full(12, np.nan, np.float32)), ('TAPERONLY', np.zeros(12, np.float32))], os.path.join(nan, name))
        assert audit_mosaic(os.path.join(nan, name)) == ['no finite pixel'], audit_mosaic(os.path.join(nan, name))
        sig = os.path.join(d, 'sig')
        os.makedirs(sig)
        write([('TAPERONLY', np.full(12, np.nan, np.float32)), ('SIGNAL', np.ones(12, np.float32))], os.path.join(sig, name))
        assert audit_mosaic(os.path.join(sig, name)) == [], audit_mosaic(os.path.join(sig, name))   # SIGNAL wins over column 1
        oth = os.path.join(d, 'oth')                                        # any other table: refused, not read vacuously
        os.makedirs(oth)
        write([('PIXEL', np.arange(12, dtype=np.float32)), ('VALUE', np.full(12, np.nan, np.float32))], os.path.join(oth, name))
        assert audit_mosaic(os.path.join(oth, name)) == ["no SIGNAL or LEVEL column ['PIXEL', 'VALUE']"], audit_mosaic(os.path.join(oth, name))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_pkcorr_and_content_file_tests():
    d = _tmp('ft7b_')
    try:
        decs = np.arange(-30.0, 89.0, 2.0)

        def curve(path, rows):
            with open(path, 'w') as f:
                f.write('dec,rec_median,rec_mad,n,n_rounds\n')
                for dec, med, nr in rows:
                    f.write(f'{dec:.1f},{med:.4f},0.0100,{nr},{nr}\n')
            return path
        good = curve(os.path.join(d, 'c.csv'), [(x, 0.85, 5) for x in decs])
        assert T5.pkcorr_curve_nodes(good, rounds=5)[0]
        assert not T5.pkcorr_curve_nodes(curve(os.path.join(d, 'c1.csv'), [(x, 0.85, 5) for x in decs[1:]]), rounds=5)[0]
        assert not T5.pkcorr_curve_nodes(curve(os.path.join(d, 'c2.csv'), [(x, 0.85, 2 if x == 0 else 5) for x in decs]), rounds=5)[0]
        assert not T5.pkcorr_curve_nodes(curve(os.path.join(d, 'c3.csv'), [(x, 1.25 if x == 40 else 0.85, 5) for x in decs]), rounds=5)[0]
        rp = os.path.join(d, 'r.csv')
        open(rp, 'w').write('round,dec,ra,ntile,rec\n' + ''.join(f'{r},-30.0,{r * 3:.2f},2,0.8500\n' for r in range(5)))
        assert T5.pkcorr_recovery_rounds(rp, rounds=5)[0] and not T5.pkcorr_recovery_rounds(rp, rounds=6)[0]
        jp = os.path.join(d, 'p.json')
        json.dump(dict(rounds=5, map='/m.fits', rec_median=0.85, test_pkcorr=dict(verdict='pass')), open(jp, 'w'))
        assert T5.pkcorr_report_pass(jp, map_path='/m.fits', rounds=5)[0]
        assert not T5.pkcorr_report_pass(jp, map_path='/other.fits', rounds=5)[0] and not T5.pkcorr_report_pass(jp, rounds=4)[0]
        json.dump(dict(rounds=5, map='/m.fits', test_pkcorr=dict(verdict='fail')), open(jp, 'w'))
        assert not T5.pkcorr_report_pass(jp, map_path='/m.fits', rounds=5)[0]
        cp = os.path.join(d, 'content.json')
        for rep, ok in ((_report(), True), (_report(med=1.011), False), (_report(fr=0.89), False), (_report(nsrc=19), False),
                        (_report(links=False), False), (_report(n=23), False)):
            rep['provenance_matches_tiles'] = True
            json.dump({'jysr': rep, 'jybeam': rep}, open(cp, 'w'))
            assert T5.content_passes(cp, n_tiles=24)[0] is ok, (rep['median_ratio'], rep['frac_within_5pct'], rep['n_sources'], ok)
        rep = _report()
        rep['provenance_matches_tiles'] = False
        json.dump({'jysr': dict(_report(), provenance_matches_tiles=True), 'jybeam': rep}, open(cp, 'w'))
        assert not T5.content_passes(cp, n_tiles=24)[0]
        json.dump({'jysr': dict(_report(), provenance_matches_tiles=True)}, open(cp, 'w'))
        assert not T5.content_passes(cp, n_tiles=24)[0]                  # a domain missing
    finally:
        shutil.rmtree(d, ignore_errors=True)



# ---- PR-7c: conform and tables ----------------------------------------------------------------------------------------------------

ARCH_JYSR = 'Blue_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_jysr_feather_lsfill_allsky_20241218-20260419_N024.fits'
ARCH_NAMES = {   # the archived V5 Blue Taper R-0.75 products (mosaic/Run_HYBRID_20241218-20260419/Blue/I/<sub>/), listed 2026-11-18
    ('healpix', 'Blue_I_deep_Taper_Robust-0.75_coadd_lsrecon_mosaic_feather_lsfill_pbmap_allsky_20241218-20260419_N024.fits'),
    ('healpix', 'Blue_I_deep_Taper_Robust-0.75_coadd_lsrecon_mosaic_jybeam_feather_lsfill_allsky_20241218-20260419_N024.fits'),
    ('healpix', 'Blue_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_feather_lsfill_caveat_allsky_20241218-20260419_N024.fits'),
    ('healpix', 'Blue_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_feather_lsfill_pkcorr_allsky_20241218-20260419_N024.fits'),
    ('healpix', 'Blue_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_feather_lsfill_psfmap_allsky_20241218-20260419_N024.fits'),
    ('healpix', 'Blue_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_jysr_feather_lsfill_allsky_20241218-20260419_N024.fits'),
    ('mollweide', 'Blue_I_deep_Taper_Robust-0.75_coadd_lsrecon_mosaic_jybeam_feather_lsfill_molrender_allsky_20241218-20260419_N024.fits'),
    ('mollweide', 'Blue_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_jysr_feather_lsfill_molrender_allsky_20241218-20260419_N024.fits'),
    ('radec', 'Blue_I_deep_Taper_Robust-0.75_coadd_lsrecon_mosaic_jybeam_feather_lsfill_carrender_allsky_20241218-20260419_N024.fits'),
    ('radec', 'Blue_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_jysr_feather_lsfill_carrender_allsky_20241218-20260419_N024.fits')}


def test_conform_names_and_rows():
    from astropy.time import Time
    from exopipe.phase3.lib.mosaic import conform_v5 as CF
    P = v5.paths('/o', 'Blue', 'R075')
    t0, t1 = Time('2024-12-18T03:04:12.100').mjd, Time('2026-04-19T07:03:34.360').mjd
    nm = CF.names('Blue', 'R075', P, True, t0, t1, 24)
    assert {(sub, n) for k, (_s, sub, n, _p) in nm.items() if sub != 'unfilled'} == ARCH_NAMES, sorted(nm)
    assert (nm['unfilled_jysr'][1:3], nm['unfilled_jybeam'][1:3]) == (('unfilled', nm['jysr'][2]), ('unfilled', nm['jybeam'][2]))
    assert nm['unfilled_jysr'][0] == P['unfilled_jysr'] and nm['unfilled_jybeam'][0] == P['unfilled_jybeam']
    assert nm['psfmap'][0] == P['unfilled_psf'] and nm['pbmap'][0] == P['unfilled_pbeff'] and nm['jysr'][0] == P['filled_jysr']
    assert [k for k, v in nm.items() if v[3]] == ['jysr', 'jysr_mollweide', 'jysr_radec', 'psfmap', 'pkcorr', 'caveat',
                                                  'unfilled_jysr']   # PBCORR
    assert 'caveat' not in CF.names('18MHz', 'R0', v5.paths('/o', '18MHz', 'R0'), False, t0, t1, 24)
    assert CF.names('18MHz', 'R0', v5.paths('/o', '18MHz', 'R0'), False, t0, t1, 24)['jysr'][2] == \
        '18MHz_I_deep_Taper_Robust+0.0_pbcorr_coadd_lsrecon_mosaic_jysr_feather_lsfill_allsky_20241218-20260419_N024.fits'
    # the table rows of production's formats (archived V5 tables, Blue Taper R-0.75)
    fm = {'donor_mhz': 74, 's': 1.0522906851499552, 'test': {'large_scale_rel_filled': 0.0011883698366225858,
          'large_scale_rel_unfilled': 0.2925097595796858, 'highpass_rel': 0.01025648869260899,
          'highpass_abs_over_map_rms': 0.06163105975005245, 'result': 'PASS'}}
    assert ','.join(CF.tpfill_row('X', fm)[1:]) == '74,1.05229,0.0011883698366225858,0.2925097595796858,0.01025648869260899,0.06163105975005245,PASS'
    d = _tmp('cfrows_')
    try:
        mk = lambda dom, med, p05, p95: dict(map=f'/w/healpix_I_deep_Taper_R075_Blue_lsfill_{dom}_nside2048.fits', variant='Taper_R075',  # noqa: E731
                                              band='Blue', domain=dom, n_tiles=24, n_sources=200, median_ratio=med, p05=p05, p95=p95,
                                              frac_within_5pct=1.0, result='PASS')
        cj = os.path.join(d, 'c.json')
        json.dump({'jysr': mk('jysr', 0.9979303613242603, 0.981671171058413, 1.0156599100788055),
                   'jybeam': mk('jybeam', 0.9982441914025694, 0.9815171146845747, 1.0160787733402987)}, open(cj, 'w'))
        assert [','.join(r) for r in CF.summary_rows(cj)] == [
            'healpix_I_deep_Taper_R075_Blue_lsfill_jybeam_nside2048.fits,Taper_R075,Blue,jybeam,24,200,0.9982441914025694,0.9815171146845747,'
            '1.0160787733402987,1.0,PASS',
            'healpix_I_deep_Taper_R075_Blue_lsfill_jysr_nside2048.fits,Taper_R075,Blue,jysr,24,200,0.9979303613242603,0.981671171058413,'
            '1.0156599100788055,1.0,PASS']
        keys = [CF.summary_sort(['', v, b, dom]) for v in ('Taper_R0', 'Taper_R075') for b in ('Blue', '82MHz', '18MHz') for dom in ('jysr', 'jybeam')]
        assert sorted(keys)[:3] == ['Taper_R075_18MHz_lsfill_jybeam.json', 'Taper_R075_18MHz_lsfill_jysr.json',
                                    'Taper_R075_82MHz_lsfill_jybeam.json']           # R-0.75 first; sub-bands before Blue
        cv = os.path.join(d, 'caveats_Taper_R075_Blue_lsfill.csv')
        open(cv, 'w').write('lst,kind,staged,sigma_apparent\n00h,hybrid,/q/T00.fits,0.27\n01h,taper_only,,0.3\n')
        head, rows = CF.caveat_rows(cv, 'Blue')
        assert head == ['lst', 'kind', 'staged', 'sigma_apparent'] and rows[0][2] == CF.prov_file('00h', 'Blue', 'T00.fits') and rows[1][2] == ''
        assert CF.prov_file('00h', 'Blue', 'T00.fits') == os.path.join(CFG.hybrid.ARCHIVE_HYBRID, '00h', 'Blue', 'T00.fits')
        a = [['Taper_R0', 'Blue', '-30.0'], ['Taper_R075', 'Blue', '-30.0'], ['Taper_R075', 'Blue', '-28.0'], ['Taper_R075', '18MHz', '-30.0']]
        m = CF.merge(a, [['Taper_R075', 'Blue', '-30.0', 'new']], lambda x: (x[0], x[1]), CF.curve_sort)
        assert m == [['Taper_R075', '18MHz', '-30.0'], ['Taper_R075', 'Blue', '-30.0', 'new'], ['Taper_R0', 'Blue', '-30.0']], m
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _time_cards(i):
    from astropy.time import Time
    t0 = 60662.0 + 20 * i
    return {'MJD-OBS': t0, 'MJD-MID': t0 + 5, 'DATE-END': Time(t0 + 10 + i, format='mjd').isot, 'INTTIME': 1000.0 + i,
            'GITTAG': f'g{i}', 'BMAJ': 0.1 + 0.001 * i, 'BMIN': 0.09 + 0.001 * i}


def test_conform_membership():
    from astropy.time import Time
    from exopipe.phase3.lib.mosaic import conform_v5 as CF
    d = _tmp('cfmem_')
    try:
        rng = np.random.default_rng(3)
        tiles, sig = [], []
        for i, lst in enumerate(('02h', '00h', '01h')):
            data = rng.normal(0, 1 + i, (12, 12)).astype(np.float32)
            pb = np.full((12, 12), 0.5, np.float32)
            p = _img(os.path.join(d, f'T{lst}.fits'), data, **_time_cards(i))
            tiles.append(dict(lst=lst, path=p, pb=_img(os.path.join(d, f'PB{lst}.fits'), pb), prodname=f'N{lst}.fits'))
            app = (np.nan_to_num(data) * pb).ravel()
            sig.append(1.4826 * np.median(np.abs(app - np.median(app))))
        rows, t0, t1, itt, beam = CF.membership(tiles, 'Blue', 0.1)
        assert [r['FILE'] for r in rows] == [CF.prov_file(x, 'Blue', f'N{x}.fits') for x in ('00h', '01h', '02h')]   # LST order
        assert np.allclose([r['WEIGHT'] for r in rows], [1 / sig[1] ** 2, 1 / sig[2] ** 2, 1 / sig[0] ** 2], rtol=1e-6)
        assert [r['P1RUN'] for r in rows] == ['g1', 'g2', 'g0'] and t0 == 60662.0
        assert abs(t1 - Time(_time_cards(2)['DATE-END']).mjd) < 1e-9 and itt == 3003.0 and beam == (0.101, 0.091, 0.0)
        # the build's guards: a flat tile, fewer than 100 pixels above the PB floor, no DATE-END
        _img(tiles[0]['path'], np.full((12, 12), 2.0, np.float32), **_time_cards(0))
        assert 'flat or corrupt' in _raises(CF.ConformRefusal, CF.membership, tiles, 'Blue', 0.1)
        _img(tiles[0]['path'], rng.normal(0, 1, (12, 12)).astype(np.float32), **_time_cards(0))
        assert 'finite apparent pixels' in _raises(CF.ConformRefusal, CF.membership, tiles, 'Blue', 0.6)
        c = _time_cards(0)
        c.pop('DATE-END')
        _img(tiles[0]['path'], rng.normal(0, 1, (12, 12)).astype(np.float32), **c)
        assert 'R-07' in _raises(CF.ConformRefusal, CF.membership, tiles, 'Blue', 0.1)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _chain(d):
    """A whole v5 unit (Blue R075, NSIDE 1) through stage .. content with stand-ins whose files the conform can stamp: tiles with noise
    and time cards, maps with their BUNIT, a realistic PKCORR curve, a caveat table, a fill test of the archived columns."""
    from exopipe.phase3.lib.mosaic import caveat as CV
    from exopipe.phase3.lib.mosaic import content as CT
    from exopipe.phase3.lib.mosaic import pkcorr as PK
    rng = np.random.default_rng(5)
    tiles = {}
    for i, lst in enumerate(LSTS):
        p = _img(os.path.join(d, 'Q', TILE.format(lst=lst)), rng.normal(0, 1, (12, 12)).astype(np.float32), **_time_cards(i))
        tiles[lst] = (p, _md5f(p))
        _img(os.path.join(d, 'M', lst, 'Blue', 'PB_deep_R075_Blue.fits'), np.full((12, 12), 0.5, np.float32))
    out = os.path.join(d, 'out')
    assert _stage(d, out, tiles).status == 'done'
    P = v5.paths(out, 'Blue', 'R075')
    work = os.path.join(d, 'work')
    eng = _fake_engine([])
    with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', eng), _no_content_tests():
        assert _unit(out, 'coadd', v5.coadd_unit, work).status == 'done'
    for k, unit in (('psf', 'sr'), ('pbeff', '')):          # the engine's cards on its maps (HDU 1)
        with fits.open(P[f'unfilled_{k}'], mode='update', memmap=False) as h:
            h[1].header['BUNIT'] = unit
    co = record.load_record(out, v5.STEP, v5.unit_name('coadd', 'Blue', 'R075'))
    co['meta']['md5'].update(psf=_md5f(P['unfilled_psf']), pbeff=_md5f(P['unfilled_pbeff']))
    json.dump(co, open(record.record_path(out, v5.STEP, v5.unit_name('coadd', 'Blue', 'R075')), 'w'))

    def run_band(root, band, variant, outd, lsts, rounds, tag, mos_c, psf, scratch, nside, record_dir=None):
        nm = PK.names(variant, band, tag, nside)
        Q = {k: os.path.join(outd, f) for k, f in nm.items()}
        _hpx(Q['map'], np.full(12, 0.9), NSIDE=nside, ORDERING='RING', BUNIT='', SINJ=50.0, NROUNDS=rounds)
        with open(Q['curve'], 'w', newline='') as f:
            f.write('dec,rec_median,rec_mad,n,n_rounds\r\n-30.0,0.8492,0.0768,5,5\r\n-28.0,0.85,0.07,5,5\r\n')
        open(Q['recovery'], 'w').write('x\n')
        res = dict(rounds=rounds, rec_median=0.9, rec_min=0.9, rec_max=0.9, ra_scatter_median_mad=0.01, map=os.path.join(record_dir, nm['map']),
                   test_pkcorr=dict(verdict='pass', nodes=60, missing_or_few_rounds=[], out_of_range=[], max_map_err=0.0))
        json.dump(res, open(Q['json'], 'w'))
        return Q, res

    def cav(tl, variant, scope, out_dir, pbeff_mosaic, nside=None):
        mp, tp = (os.path.join(out_dir, x) for x in CV.names(variant, scope, 128))
        _hpx(mp, np.zeros(12 * 128 ** 2, np.float32), NSIDE=128)   # a full NSIDE-128 map, as its name says (caveat.py:120)
        with open(tp, 'w', newline='') as f:
            f.write('lst,kind,staged,sigma_apparent\r\n' + ''.join(f'{t["lst"]},hybrid,{t["path"]},0.2\r\n' for t in tl))
        return dict(map=mp, table=tp, n_hybrid=len(tl), n_taper_only=0, taper_only=[], test={'verdict': 'pass'})

    def fill_file(src, outp):
        sig = np.asarray(fits.getdata(src, 1)['SIGNAL']) + 1.0
        c = _cards(src)
        _hpx(outp, sig, c, FREQ=7.0e7, BUNIT='Jy/sr', FEATHER=True, NSIDE=1, **c)
        return dict(src=src, src_md5=_md5f(src), out=outp, out_md5=_md5f(outp), donor_mhz=70, s=0.711111,
                    test={'large_scale_rel_filled': 0.001, 'large_scale_rel_unfilled': 0.3, 'highpass_rel': 0.01,
                          'highpass_abs_over_map_rms': 0.06, 'result': 'PASS'}, result='PASS')

    def fill_jybeam_file(jb, ju, jf, psf, pbeff, outp):
        sig = np.asarray(fits.getdata(jb, 1)['SIGNAL']) + 2.0
        c = dict(_cards(jb), TPJBCONV='jybeam/jysr measured')
        _hpx(outp, sig, c, FREQ=7.0e7, BUNIT='Jy/beam', NSIDE=1, PSFMAP=os.path.basename(psf), **c)
        return dict(src=jb, src_md5=_md5f(jb), out=outp, out_md5=_md5f(outp), identity_rel=0.5, k_median=1.0, k_p05=0.98, k_p95=1.02,
                    n_bright=9, result='PASS')
    bp = types.ModuleType('fake_band_products')

    def build(variant, hpx_dir, suffix, outd, bands=(), moll_W=0, car_cdelt=0, dec_cut=None):
        for b in bands:
            for sfx in ('Mollweide', 'RADec'):
                _img(os.path.join(outd, f'{b}_{variant}{suffix}_{sfx}.fits'), BUNIT='Jy/sr' if 'jysr' in suffix else 'Jy/beam')
    bp.build = build

    def ct_run(maps, root, variant, band, parents):
        listed = json.load(open(P['tiles']))['tiles']
        read = [dict(lst=r['lst'], file=os.path.realpath(r['path']), pb=os.path.realpath(r['pb']), symlink=True) for r in listed]
        out_ = {}
        for dom, map_path in maps.items():
            r = _report()
            r.pop('provenance')
            r.update(map=map_path, domain=dom, n_sources=200, median_ratio=0.998, frac_within_5pct=1.0, read=read)
            out_[dom] = r
        return out_
    par = os.path.join(d, 'parents')
    os.makedirs(par, exist_ok=True)
    open(os.path.join(par, 'master_R075_00h_Blue.gaul.csv'), 'w').write('# none\n')
    with _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', eng), _no_content_tests(), _fake_module(
            'exopipe.phase3.lib.mosaic.band_products', bp), _patched(
            (CV, 'caveat_map', cav), (MF, 'fill_file', fill_file), (MF, 'fill_jybeam_file', fill_jybeam_file), (PK, 'run_band', run_band),
            (PK, 'read_map', _hpx_read), (CT, 'run', ct_run), (CFG.mosaic_content, 'PARENTS', par), _pinned_union(par)):
        for st, fn, a in (('pkcorr', v5.pkcorr_unit, (work,)), ('caveat', v5.caveat_unit, ()), ('fill', v5.fill_unit, ()),
                          ('render', v5.render_unit, (work,)), ('content', v5.content_unit, (work,))):
            u = _unit(out, st, fn, *a)
            assert u.status == 'done', (st, u.status, u.reason)
    return out, tiles


def _arch_tables(root, blue_r075=None):
    """A stand-in archived V5 tables directory: every scope of mosaic_v5.SCOPES in both weightings, all PASS (production's row order)."""
    from exopipe.phase3.lib.mosaic import conform_v5 as CF
    td = os.path.join(root, 'tables')
    os.makedirs(td, exist_ok=True)
    V5 = CFG.mosaic_v5
    vb = [(V5.VARIANT_OF[w], sc) for w in V5.WEIGHTINGS for sc in V5.SCOPES]
    rows = [[f'healpix_I_deep_{v}_{b}_lsfill_{dom}_nside2048.fits', v, b, dom, '24', '200', '0.999', '0.98', '1.02', '1.0', 'PASS']
            for v, b in vb for dom in ('jybeam', 'jysr')]
    CF._write(os.path.join(td, 'summary.csv'), CF.SUMMARY_COLS, sorted(rows, key=CF.summary_sort))
    rows = [[v, b, dec, '0.85', '0.07', '5', '5'] for v, b in vb for dec in ('-30.0', '-28.0')]
    CF._write(os.path.join(td, 'pkcorr_curves.csv'), CF.CURVE_COLS, sorted(rows, key=CF.curve_sort))
    for w in V5.WEIGHTINGS:
        tok = 'Robust-0.75' if w == 'R075' else 'Robust+0.0'
        rows = [[f'{sc}_I_deep_Taper_{tok}_pbcorr_coadd_lsrecon_mosaic_jysr_feather_lsfill_tpfill_allsky_20241218-20251029_N024.fits', '74',
                 '1.05', '0.001', '0.29', '0.01', '0.06', 'PASS'] for sc in V5.SCOPES]
        CF._write(os.path.join(td, f'summary_tpfill_Taper_{w}.csv'), CF.TPFILL_COLS, sorted(rows, key=lambda x: x[0]))
    for w in V5.WEIGHTINGS:
        for sc in CFG.hybrid.CAVEAT_SCOPES:
            CF._write(os.path.join(td, f'caveats_{V5.VARIANT_OF[w]}_{sc}_lsfill.csv'), ('lst', 'kind', 'staged', 'sigma_apparent'),
                      [['00h', 'hybrid', f'/a/{sc}', '0.1']])
    CF._write(os.path.join(td, 'taper_only.csv'), CF.TAPER_ONLY_COLS, [])
    return td


def test_conform_and_tables_units():
    from exopipe.phase3.lib.mosaic import conform_v5 as CF
    d = _tmp('cf_')
    try:
        arun = os.path.join(d, 'ARCH', 'Run_HYBRID_20241218-20260419')
        with _patched((CFG.mosaic_coadd, 'NSIDE', 1), (CFG.hybrid_tilefill, 'FILLED_MOSAIC_RUN', arun)):
            out, tiles = _chain(d)
            P = v5.paths(out, 'Blue', 'R075')
            eng = _fake_engine([])

            def conform(miss=(), src=None):
                with _stage_env(d, src or tiles, []), _fake_module('exopipe.phase3.lib.mosaic.coadd_jysr', eng), \
                        _patched((CF, 'classic_missing', lambda scope: list(miss))):
                    return _unit(out, 'conform', v5.conform_unit)
            u = conform()
            assert u.status == 'done', (u.status, u.reason)
            assert not [t for t in u.tests if not t['passed']] and sum(t['cls'] == 'mosaic_v5_product' for t in u.tests) >= 12, u.tests
            staged = {k: x['path'] for k, x in u.meta['products'].items()}
            assert len(staged) == 12 and all(os.path.dirname(x).startswith(os.path.join(P['conform'], 'Blue', 'I')) for k, x in staged.items()
                                             if not k.startswith('unfilled'))
            assert os.path.dirname(staged['unfilled_jysr']) == os.path.join(P['conform'], 'unfilled', 'Blue', 'I', 'healpix')
            assert os.path.basename(staged['unfilled_jysr']) == os.path.basename(staged['jysr'])
            with fits.open(staged['unfilled_jybeam'], memmap=False) as h:
                assert 'TPFILL' not in h[0].header and h[1].header['PSFMAP'] == os.path.basename(staged['psfmap'])
                assert np.array_equal(h[1].data['SIGNAL'], fits.getdata(P['unfilled_jybeam'], 1)['SIGNAL'])
            n0 = os.path.basename(staged['jysr'])
            assert n0.startswith('Blue_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_jysr_feather_lsfill_allsky_') and n0.endswith('_N024.fits')
            with fits.open(staged['jysr'], memmap=False) as h:
                h0, h1, pv = h[0].header, h[1].header, h['PROVENANCE'].data
                assert h0['PRODNAME'] == n0 == h1['PRODNAME'] and h0['NSTACK'] == 24 and 'THERMNOI' not in h0 and h0['BUNIT'] == 'Jy/sr'
                assert h0['PROVBAS'].startswith('Coadd/') and h0['PROVBAS'].endswith('(Gregg 2026-10-31)') and h0['TPFILL'] is True
                assert list(pv['FILE']) == [CF.prov_file(x, 'Blue', TILE.format(lst=x)) for x in LSTS]
                assert ' '.join(h0['HISTORY']).count('Mosaic Run_HYBRID_20241218-20260419: 24 constituent masters') == 1
                assert np.array_equal(h[1].data['SIGNAL'], fits.getdata(P['filled_jysr'], 1)['SIGNAL'])
            with fits.open(staged['jybeam'], memmap=False) as h:
                assert h[1].header['PSFMAP'] == os.path.basename(staged['psfmap']) and h[0].header['PBCORR'] is False
            for dom in ('jysr', 'jybeam'):                                    # renders carry their filled map's fill cards
                h1 = fits.getheader(P[f'filled_{dom}'], 1)
                tp = [k for k in ('TPFILL', 'TPDONOR', 'TPDONMHZ', 'TPSCALE', 'TPLLO', 'TPLHI', 'TPJBCONV') if k in h1]
                assert 'TPFILL' in tp, tp
                for r in ('mollweide', 'radec'):
                    h0 = fits.getheader(staged[f'{dom}_{r}'], 0)
                    assert all(h0[k] == h1[k] and h0.comments[k] == h1.comments[k] for k in tp), (dom, r, tp)
            assert u.meta['tables']['tpfill'] == [n0, '70', '0.71111', '0.001', '0.3', '0.01', '0.06', 'PASS']
            # refusals: a classic Taper map not on Lustre (HG16); the tiles resolved now differ (HG13); a source changed since its stage;
            # a failed test in an upstream record
            u = conform(miss=('R0',))
            assert u.status == 'refused' and 'HG16' in u.reason and not os.path.exists(P['conform']), (u.status, u.reason)
            u = conform(src=dict(tiles, **{'05h': (tiles['05h'][0], 'f' * 32)}))
            assert u.status == 'refused' and 'HG13' in u.reason and '05h' in u.reason, (u.status, u.reason)
            rp = record.record_path(out, v5.STEP, v5.unit_name('pkcorr', 'Blue', 'R075'))
            keep = open(rp).read()
            r = json.loads(keep)
            r['tests'][0]['passed'] = False
            json.dump(r, open(rp, 'w'))
            u = conform()
            assert u.status == 'refused' and 'failed test' in u.reason, (u.status, u.reason)
            open(rp, 'w').write(keep)
            with open(P['render_jysr_radec'], 'ab') as f:
                f.write(b'\0' * 2880)
            u = conform()
            assert u.status == 'refused' and 'changed since its stage' in u.reason, (u.status, u.reason)
            with fits.open(P['render_jysr_radec'], memmap=False) as h:      # back to its recorded bytes
                h.writeto(P['render_jysr_radec'] + '.x')
            os.replace(P['render_jysr_radec'] + '.x', P['render_jysr_radec'])
            rn = record.load_record(out, v5.STEP, v5.unit_name('render', 'Blue', 'R075'))
            rn['meta']['md5']['jysr_radec'] = _md5f(P['render_jysr_radec'])
            json.dump(rn, open(record.record_path(out, v5.STEP, v5.unit_name('render', 'Blue', 'R075')), 'w'))
            assert conform().status == 'done'
            # tables: the archived rows of the other units unchanged, Blue R075 replaced; a second pass over its own output is a no-op
            td0 = _arch_tables(arun)

            def tables():
                run = _run(out)
                with run.unit(v5.TABLES_UNIT, stage='tables') as u:
                    v5.tables_unit(u, out)
                return u
            u = tables()
            assert u.status == 'done', (u.status, u.reason)
            td = v5.tables_dir(out)
            cav = sorted(f'caveats_{CFG.mosaic_v5.VARIANT_OF[w]}_{sc}_lsfill.csv' for w in CFG.mosaic_v5.WEIGHTINGS
                         for sc in CFG.hybrid.CAVEAT_SCOPES)
            assert len(cav) == 14 and sorted(os.listdir(td)) == sorted(cav + ['pkcorr_curves.csv', 'summary.csv', 'summary_tpfill_Taper_R0.csv',
                                                                             'summary_tpfill_Taper_R075.csv', 'taper_only.csv'])
            for name in cav:                                                  # the other 13 caveat tables: the archived bytes
                if name != 'caveats_Taper_R075_Blue_lsfill.csv':
                    assert open(os.path.join(td, name), 'rb').read() == open(os.path.join(td0, name), 'rb').read(), name
            for name in ('summary.csv', 'pkcorr_curves.csv', 'summary_tpfill_Taper_R075.csv', 'summary_tpfill_Taper_R0.csv'):
                a, b = CF._read(os.path.join(td0, name))[1], CF._read(os.path.join(td, name))[1]
                blue = (lambda x: 'Blue' in x[0] and 'Robust-0.75' in x[0]) if 'tpfill' in name else \
                    (lambda x: 'Taper_R075' in x and 'Blue' in x)
                assert [x for x in a if not blue(x)] == [x for x in b if not blue(x)] and len(a) == len(b), name
                assert [a.index(x) for x in a if blue(x)] == [b.index(x) for x in b if blue(x)], name     # same place in the order
            assert [x for x in CF._read(os.path.join(td, 'summary_tpfill_Taper_R075.csv'))[1] if x[0].startswith('Blue')][0][0] == n0
            assert CF._read(os.path.join(td, 'caveats_Taper_R075_Blue_lsfill.csv'))[1][3][2] == CF.prov_file('03h', 'Blue', TILE.format(lst='03h'))
            assert open(os.path.join(td, 'summary.csv'), 'rb').read().count(b'\r\n') == 77
            for name in os.listdir(td):
                shutil.copy2(os.path.join(td, name), os.path.join(td0, name))
            before = {n: open(os.path.join(td0, n), 'rb').read() for n in os.listdir(td)}
            assert tables().status == 'done'
            assert {n: open(os.path.join(td, n), 'rb').read() for n in os.listdir(td)} == before
            # a missing archived caveat table, of another scope or of the conformed one, refuses
            for name in ('caveats_Taper_R0_Full_lsfill.csv', 'caveats_Taper_R075_Blue_lsfill.csv'):
                os.rename(os.path.join(td0, name), os.path.join(td0, name + '.x'))
                u = tables()
                assert u.status == 'refused' and name in u.reason and 'is missing' in u.reason and not os.path.exists(td), (u.status, u.reason)
                os.rename(os.path.join(td0, name + '.x'), os.path.join(td0, name))
            # completeness and PASS: a scope missing from the archived table, a FAIL row
            h, rows = CF._read(os.path.join(td0, 'summary.csv'))
            CF._write(os.path.join(td0, 'summary.csv'), h, [x for x in rows if x[2] != '18MHz' or x[1] != 'Taper_R0'])
            u = tables()
            assert u.status == 'refused' and 'no jybeam row of Taper_R0 18MHz' in u.reason and not os.path.exists(td), (u.status, u.reason)
            CF._write(os.path.join(td0, 'summary.csv'), h, [x[:-1] + (['FAIL'] if x[2] == '23MHz' else [x[-1]]) for x in rows])
            u = tables()
            assert u.status == 'refused' and 'FAIL' in u.reason, (u.status, u.reason)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_classic_missing_skips_quarantine_runs():
    """HG16: a classic Taper Jy/sr map with a PASS row counts only in a live mosaic run; a quarantine run's copy does not."""
    from exopipe.phase3.lib.mosaic import conform_v5 as CF
    d = _tmp('hg16q_')
    try:
        arun = os.path.join(d, 'Run_HYBRID_20241218-20260419')
        os.makedirs(arun)

        def put(run):
            for w, v in CFG.mosaic_v5.VARIANT_OF.items():
                tok = f"{v.split('_')[0]}_{'Robust-0.75' if v.endswith('R075') else 'Robust+0.0'}"
                hp_ = os.path.join(d, run, 'Blue', 'I', 'healpix')
                os.makedirs(hp_, exist_ok=True)
                open(os.path.join(hp_, f'Blue_I_deep_{tok}_pbcorr_coadd_lsrecon_mosaic_jysr_feather_allsky_20241218-20251029_N024.fits'),
                     'w').close()
            os.makedirs(os.path.join(d, run, 'tables'), exist_ok=True)
            with open(os.path.join(d, run, 'tables', 'summary.csv'), 'w') as f:
                f.write('file,variant,band,domain,result\n' + ''.join(f'x,{v},Blue,jysr,PASS\n' for v in CFG.mosaic_v5.VARIANT_OF.values()))
        with _patched((CFG.hybrid_tilefill, 'FILLED_MOSAIC_RUN', arun)):
            put('Run_FEATHER3_QUARANTINE_20261109')
            assert sorted(CF.classic_missing('Blue')) == sorted(CFG.mosaic_v5.VARIANT_OF)
            put('Run_FEATHER3_20261109')
            assert CF.classic_missing('Blue') == []
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_stamp_products_killed_and_duplicates():
    """stamp_products (lib/mosaic/gen_common): (1) a stamp KILLED mid-write leaves its temporary file (core/placement's
    <dst>.<pid>.<8 hex digits>.tmp) and the rerun's _clear removes it - and not another product's temporary file; (2) two
    items naming one final path: identical, one product; different, refused before anything is written."""
    import glob
    import subprocess
    from exopipe.phase3.core import placement, products
    from exopipe.phase3.lib.mosaic import gen_common as G
    d = _tmp('stamp')
    try:
        src, src2 = _img(os.path.join(d, 'src', 'a.fits')), _img(os.path.join(d, 'src', 'b.fits'), np.zeros((4, 4), np.float32))
        dst = os.path.join(d, 'out', 'mollweide', 'a.fits')
        os.makedirs(os.path.dirname(dst))
        other = os.path.join(d, 'out', 'mollweide', 'b.fits.123.0123abcd.tmp')       # another product's temporary file
        open(other, 'w').close()
        kw = dict(scope='deep', stokes='I', t0=60000.0, t1=60001.0, itt=0, beam=None, thermnoi=None, psf_name='p')
        child = (f'import os, sys\nsys.path.insert(0, {REPO!r})\n'
                 'from exopipe.phase3.core import products\nfrom exopipe.phase3.lib.mosaic import gen_common as G\n'
                 'products.write_standard_header = lambda *a, **k: os._exit(9)\n'
                 f'G.stamp_products([("jybeam", {src!r}, "mollweide", {dst!r}, False, {{"PRODNAME": "x"}})], [], [], **{kw!r})\n')
        p = subprocess.run([sys.executable, '-c', child], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        left = glob.glob(placement.temp_glob(glob.escape(dst)))
        assert p.returncode == 9 and len(left) == 1 and not os.path.exists(dst), (p.returncode, p.stdout[-400:], left)
        G._clear([dst])
        assert not glob.glob(placement.temp_glob(glob.escape(dst))) and os.path.exists(other), os.listdir(os.path.dirname(dst))
        it = ('jybeam', src, 'mollweide', dst, False, {'PRODNAME': 'x'})
        it2 = ('jybeam', src2, 'mollweide', dst, False, {'PRODNAME': 'x'})
        noop = lambda *a, **k: None       # noqa: E731 - the header writers are not under test here
        with _patched((products, 'write_standard_header', noop), (products, 'append_provenance_table', noop)):
            out = G.stamp_products([it, it], [], [], **kw)
            assert out == [(dst, src, 'jybeam')] and _md5f(dst) != _md5f(src2), out
            assert sorted(os.listdir(os.path.dirname(dst))) == ['a.fits', os.path.basename(other)]
            m0 = _md5f(dst)
            _raises(Refusal, G.stamp_products, [it, it2], [], [], **kw)
            assert _md5f(dst) == m0 and sorted(os.listdir(os.path.dirname(dst))) == ['a.fits', os.path.basename(other)]
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
