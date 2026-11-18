#!/usr/bin/env python3
"""The hybrid route, pull request PR-5 (plan/specs/hybrid.md section 4): the stage tilefill (lib/hybrid/tilefill.py, the content-test
classes of lib/hybrid/tests_tilefill.py, steps/hybrid.tilefill_*).

Data-free (CI):
  1. the constants bound from the config (production MF 25-29, 192-199; HMF 41-45);
  2. the identity and compact-source checks on synthetic maps: an exact increment passes, a changed pixel fails; a bright source
     changed by 5% fails the compact check, a smooth large-scale increment passes it;
  3. the record content tests: a record of the rule passes every test; each rule broken alone fails its own test; the compact check
     failing passes only on a ruled caveat unit that recorded the ruled text;
  4. card_superset (D7): a file lacking a card key of the file it replaces fails;
  5. the sources: work without a hybrid record = no input; pinned refuses a unit rebuilt in --out; pinned finds the unfilled source
     its installed master's TPSRCMD5 names in a pinned manifest, takes the nights from the production record of those sources (none =
     refused), refuses (HG13) when the source changed, refuses a night now excluded and two installed files of a kind; the file a
     product replaces is found by kind, not by name; the dry run's plan reads no md5 and no PB map; the mosaic pair takes the
     first unfilled run holding the twin and refuses 0 or 2 filled maps;
  5a. exclusion per member for wideband and Full units (a night excluded in one member only is no refusal), per night for subbands;
  5b. the band field is made once per --out and reused while unchanged; a changed filled mosaic remakes it, a changed unfilled
     mosaic records FAIL (HG13); a record without a compact-source block fails tilefill_compact;
  6. the plan: the filled names are the masters' product names, the CLEAN residual loses 'lsfill'; pinned re-makes no CLEAN residual.
Cluster parity (not here): 18h 46 MHz Robust -0.75, band field and fill from the pinned 11-11 sources and from the PR-4 work outputs,
against the archived filled masters (data md5), run by hand on lwacalim07.
Run: python tests/test_hybrid_tilefill.py   (or under pytest)
"""
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
from exopipe.phase3.core.record import NoInput, Refusal  # noqa: E402
from exopipe.phase3.lib.hybrid import tests_tilefill as TT  # noqa: E402
from exopipe.phase3.lib.hybrid import tilefill as TF  # noqa: E402
from exopipe.phase3.steps import hybrid as HS  # noqa: E402

NAME = '46MHz_I_deep_Taper_Robust-0.75_shflux_mdewarp{pb}_dewarped_aligned_coadd_lsrecon_lsfill{cr}_LST18h_20250405-20251014_N015.fits'


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    raise AssertionError(f'{fn.__name__} did not raise {exc.__name__}')


def _tmp(prefix):
    return tempfile.mkdtemp(prefix=prefix, dir=os.environ.get('TMPDIR') or None)


def test_bind_equals_production():
    g = TF.bind()
    want = dict(LFSS_MHZ=(35, 38, 40, 45, 50, 60, 70, 74, 80), LMAX=256, APOD=3.0, L_LO=8, L_HI=12, TILE_RT_NSIDE=256, TILE_RT_MAX=0.02,
                COMPACT_K=30, COMPACT_MED_MAX=0.005, COMPACT_MAX=0.02, COMPACT_SIG=10.0,
                TPCAVEAT_TEXT='compact check exceeded at 03h rim, Cygnus X (Gregg 11-11)',
                TPCAVEAT_UNITS=frozenset({('03h', '55MHz', 'R0', 'pbcorr'), ('03h', '59MHz', 'R0', 'pbcorr')}),
                REPAIR_KEYS=('P2RUN', 'SCALE_A', 'FLUXREF', 'ASTREF', 'P1RUN', 'CALBP', 'MDWSTAT'))
    bad = {k: (g[k], v) for k, v in want.items() if g[k] != v}
    assert not bad, bad
    assert g['LFSS_DIR'] == os.path.join(CFG.hybrid_recon.LFSS_DIR, 'healpix')
    assert TF.donor_mhz(46.0e6) == 45 and TF.donor_mhz(82.0e6) == 80 and TF.donor_mhz(18.0e6) == 35


def _sky(seed=1, n=200):
    r = np.random.default_rng(seed)
    old = r.normal(0, 1.0, (n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    for i in range(40):                                   # 40 significant compact sources (contrast >> 10 x the robust rms)
        y, x = r.integers(10, n - 10, 2)
        old += (50 + 5 * i) * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 2.0)
    old[:3] = np.nan                                      # a blanked strip
    return old, yy, xx


def test_identity_and_compact():
    TF.bind()
    old, yy, xx = _sky()
    inc = 0.5 + 1e-3 * yy                                 # a smooth large-scale increment
    new = (old + inc).astype(np.float32).astype(np.float64)
    assert TF._identity(new, old, inc) <= 1.0
    bad = new.copy(); bad[100, 100] += 1.0
    assert TF._identity(bad, old, inc) > 1.0
    c = TF._compact(new, old)
    assert c['ok'] and c['n'] == 30, c                    # the 30 brightest significant maxima, unchanged contrast
    o = np.where(np.isfinite(old), old, -np.inf)
    y, x = np.unravel_index(np.argmax(o), o.shape)
    hit = new.copy(); hit[y, x] += 0.05 * (old[y, x] - np.nanmedian(old))     # the brightest source changed by ~5%
    c2 = TF._compact(hit, old)
    assert not c2['ok'] and c2['max'] > 0.02, c2


def _record(path, lst='18h', scope='46MHz', w='R075', **over):
    t = dict(identity_rel=0.12, blanking_unchanged=True, compact=dict(n=30, median=1e-4, max=1e-3, ok=True), caveat=None,
             header_valid=True, out='/nonexistent', out_md5='x')
    rec = dict(lst=lst, scope=scope, weighting=w, source='work', result='PASS',
               pbcorr=dict(t, roundtrip=dict(n_pix=290000, rel=1.4e-5, ok=True)), plain=dict(t))
    for k, v in over.items():
        kind, key = k.split('__')
        rec[kind][key] = v
    with open(path, 'w') as f:
        json.dump(rec, f)
    return path


def test_record_content_tests():
    d = _tmp('tf_rec_')
    try:
        p = _record(os.path.join(d, 'r.json'))
        for fn in (TT.tilefill_identity, TT.tilefill_blanking, TT.tilefill_compact, TT.tilefill_roundtrip):
            assert fn(p)[0], fn.__name__
        assert TT.tilefill_record_unit(p, lst='18h', scope='46MHz', weighting='R075')[0]
        assert not TT.tilefill_record_unit(p, lst='19h', scope='46MHz', weighting='R075')[0]
        assert not TT.tilefill_identity(_record(p, plain__identity_rel=1.5))[0]
        assert not TT.tilefill_blanking(_record(p, pbcorr__blanking_unchanged=False))[0]
        assert not TT.tilefill_roundtrip(_record(p, pbcorr__roundtrip=dict(n_pix=290000, rel=0.05, ok=False)))[0]
        assert not TT.tilefill_roundtrip(_record(p, pbcorr__roundtrip=dict(n_pix=50, rel=1e-5, ok=False)))[0]
        failc = dict(n=30, median=1e-3, max=0.08, ok=False)
        assert not TT.tilefill_compact(_record(p, pbcorr__compact=failc))[0]                     # not a ruled caveat unit
        txt = CFG.hybrid_tilefill.TPCAVEAT_TEXT
        ruled = dict(lst='03h', scope='55MHz', w='R0')
        assert TT.tilefill_compact(_record(p, pbcorr__compact=failc, pbcorr__caveat=txt, **ruled))[0]
        assert not TT.tilefill_compact(_record(p, pbcorr__compact=failc, **ruled))[0]           # the caveat was not recorded
        assert not TT.tilefill_compact(_record(p, plain__compact=failc, plain__caveat=txt, **ruled))[0]   # ruled for pbcorr only
        assert not TT.tilefill_compact(_record(p, pbcorr__caveat=txt, **ruled))[0]             # a caveat on a passing check
        assert not TT.tilefill_compact(_record(p, plain__compact=None))[0]                      # no compact block is no pass
        assert not TT.tilefill_compact(_record(p, plain__compact=dict(median=0.0, max=0.0)))[0]    # nor one without n and ok
        assert not TT.tilefill_product_md5(_record(p))[0]                                       # the products are not there
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _fits(path, data=None, **cards):
    h = fits.Header()
    for k, v in cards.items():
        h[k] = v
    fits.PrimaryHDU(np.zeros((4, 4), np.float32) if data is None else data, h).writeto(path, overwrite=True)
    return path


def test_card_superset():
    d = _tmp('tf_cards_')
    try:
        old = _fits(os.path.join(d, 'old.fits'), P2RUN='a', MDWSTAT='applied', TPFILL=True)
        assert TT.card_superset(_fits(os.path.join(d, 'new.fits'), P2RUN='a', MDWSTAT='applied', TPFILL=True, GITTAG='x'), old)[0]
        ok, n, why = TT.card_superset(_fits(os.path.join(d, 'new2.fits'), P2RUN='a', TPFILL=True), old)
        assert not ok and n == 1 and 'MDWSTAT' in why, why
        assert TT.card_superset(old, None)[0]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_sources_and_plan():
    d = _tmp('tf_src_')
    try:
        out = os.path.join(d, 'w')
        os.makedirs(os.path.join(out, '18h_46MHz'))
        assert 'no hybrid record' in _raises(NoInput, _source, d, out, {}, 'work')
        # pinned: installed filled masters naming their unfilled sources by md5; the production record of those sources lists the nights
        arch, q = os.path.join(d, 'H', '18h', '46MHz'), os.path.join(d, 'Q')
        os.makedirs(arch); os.makedirs(q)
        rows, md = {}, {}
        for kind, pb in (('plain', ''), ('pbcorr', '_pbcorr')):
            n = NAME.format(pb=pb, cr='')
            src = _fits(os.path.join(q, n), PRODNAME=n)
            md[kind] = rows[src] = hashlib.md5(open(src, 'rb').read()).hexdigest()
            _fits(os.path.join(arch, n), PRODNAME=n, TPFILL=True, TPSRCMD5=md[kind])
        cr = _fits(os.path.join(arch, NAME.format(pb='', cr='_cleanres').replace('_lsfill_cleanres_', '_cleanres_')))
        _fits(os.path.join(arch, 'PBdiffuse_deep_46MHz.fits'), data=np.ones((4, 4), np.float32))
        assert 'nights are unknown' in _raises(TF.TilefillRefusal, _source, d, out, rows, 'pinned')     # no production record
        rj = os.path.join(d, 'R1', '18h_46MHz', 'R075_hybrid.json')
        os.makedirs(os.path.dirname(rj))
        with open(rj, 'w') as f:
            json.dump(dict(product=dict(md5=md['plain']), product_pbcorr=dict(md5=md['pbcorr']),
                           span=dict(nights=[dict(date='2025-04-08'), dict(date='2025-04-05')])), f)
        old = (CFG.hybrid.ARCHIVE_HYBRID, CFG.hybrid.LSRECON_RUNS)
        rj2 = os.path.join(d, 'R2', '18h_46MHz', 'R075_hybrid.json')        # a second run carrying the same record: one product
        os.makedirs(os.path.dirname(rj2))
        shutil.copy(rj, rj2)
        assert _source(d, out, rows, 'pinned', runs=('R1', 'R2'))['nights'] == ['2025-04-05', '2025-04-08']
        with open(rj2, 'w') as f:                                             # ... but one that disagrees on the nights refuses
            json.dump(dict(product=dict(md5=md['plain']), product_pbcorr=dict(md5=md['pbcorr']), span=dict(nights=[dict(date='2025-04-08')])), f)
        assert '2 distinct' in _raises(TF.TilefillRefusal, _source, d, out, rows, 'pinned', runs=('R1', 'R2'))
        s = _source(d, out, rows, 'pinned')
        assert s['source'] == 'pinned' and set(s['src']) == {'plain', 'pbcorr'}, s
        assert all(s['src'][k][0] in rows and s['src'][k][2] == md[k] for k in s['src']) and s['pb'].shape == (4, 4)
        assert s['nights'] == ['2025-04-05', '2025-04-08'] and rj in s['reads'], s
        assert s['replaces']['plain'] == os.path.join(arch, NAME.format(pb='', cr='')) and s['replaces']['cleanres'] == cr, s['replaces']
        assert '2025-04-08' in _raises(TF.TilefillRefusal, _source, d, out, rows, 'pinned', excluded=('2025-04-08',))
        plan = HS.tilefill_plan(out, '18h', '46MHz', 'R075', s)
        assert [c for _p, c in plan] == ['hybrid_filled_master', 'hybrid_filled_master', 'hybrid_tilefill_record'], plan
        assert os.path.basename(plan[1][0]) == NAME.format(pb='_pbcorr', cr='') and '/tilefill/' in plan[0][0]
        with open(s['src']['plain'][0], 'ab') as f:                         # the pinned source changed
            f.write(b'\0' * 2880)
        assert 'HG13' in _raises(TF.TilefillRefusal, _source, d, out, rows, 'pinned')
        sp = _source(d, out, rows, 'pinned', plan_only=True)                # the dry run computes no md5 and reads no PB map
        assert sp['pb'] is None and sp['source'] == 'pinned'
        _fits(os.path.join(arch, NAME.format(pb='', cr='').replace('N015', 'N016')), PRODNAME='x')     # two installed of a kind
        assert 'more than one' in _raises(TF.TilefillRefusal, _source, d, out, rows, 'pinned')
        # a unit rebuilt in --out is filled from the work run, never pinned
        os.remove(os.path.join(arch, NAME.format(pb='', cr='').replace('N015', 'N016')))
        with open(os.path.join(out, '18h_46MHz', 'R075_hybrid.json'), 'w') as f:
            f.write('{}')
        assert 'rebuilt in --out' in _raises(Refusal, _source, d, out, rows, 'pinned')
        # the work plan: the CLEAN residual renamed without lsfill
        sw = dict(source='work', src={'plain': ('/a', NAME.format(pb='', cr='')), 'pbcorr': ('/b', NAME.format(pb='_pbcorr', cr='')),
                                      'cleanres': ('/c', NAME.format(pb='', cr='_cleanres'))})
        plan = HS.tilefill_plan(out, '18h', '46MHz', 'R075', sw)
        assert [c for _p, c in plan][2] == 'hybrid_cleanres_renamed'
        assert os.path.basename(plan[2][0]) == NAME.format(pb='', cr='_cleanres').replace('_lsfill_cleanres_', '_cleanres_')
        assert (CFG.hybrid.ARCHIVE_HYBRID, CFG.hybrid.LSRECON_RUNS) == old
    finally:
        shutil.rmtree(d, ignore_errors=True)


class _Excl:
    path = None

    def __init__(self, bad):
        self.bad = set(bad)

    def excluded_constituents(self, lst, scope, dates):
        return sorted(x for x in dates if x in self.bad)


def _source(d, out, rows, mode, excluded=(), plan_only=False, runs=('R1',)):
    """tilefill_source with hybrid.ARCHIVE_HYBRID and hybrid.LSRECON_RUNS pointed at the test tree d and a stub exclusion read
    (`excluded` = the dates it excludes), all restored after the call."""
    from exopipe.phase3.core import exclusions
    old = (CFG.hybrid.ARCHIVE_HYBRID, CFG.hybrid.LSRECON_RUNS, exclusions.for_tier)
    setattr(CFG.hybrid, 'ARCHIVE_HYBRID', os.path.join(d, 'H'))
    setattr(CFG.hybrid, 'LSRECON_RUNS', {k: os.path.join(d, k) for k in runs})
    exclusions.for_tier = lambda tier, fresh=False: _Excl(excluded)
    try:
        return HS.tilefill_source(out, '18h', '46MHz', 'R075', rows, mode, plan_only=plan_only)
    finally:
        setattr(CFG.hybrid, 'ARCHIVE_HYBRID', old[0]); setattr(CFG.hybrid, 'LSRECON_RUNS', old[1]); exclusions.for_tier = old[2]


def test_field_reuse():
    """The band field is made once per --out and reused while its record passed and its inputs and file are unchanged; a changed
    filled mosaic remakes it; a changed unfilled mosaic refuses (HG13) and records FAIL; no temporary file is left."""
    d = _tmp('tf_fld_')
    calls = []

    def fake(unf, fil, out):
        calls.append(out)
        _fits(out, data=np.full((4, 4), len(calls), np.float32))
        return dict(out=out, out_md5=_md5f(out), filled=fil, filled_md5=_md5f(fil), unfilled=unf, unfilled_md5=_md5f(unf), s=1.0,
                    donor_mhz=50, result='PASS')
    old = (HS.mosaic_pair, TF.band_increment)
    try:
        fil, unf = _fits(os.path.join(d, 'F.fits')), _fits(os.path.join(d, 'U.fits'))
        rows = {unf: _md5f(unf)}
        HS.mosaic_pair = lambda scope, w, T: (fil, unf)
        TF.band_increment = fake
        out = os.path.join(d, 'w')
        HS._FIELDS.clear()
        r1 = HS.tilefill_field(out, '46MHz', 'R075', rows)
        fd = os.path.join(out, '_tilefill_fields')
        assert r1['result'] == 'PASS' and r1['out'] == os.path.join(fd, 'field_46MHz_R075.fits') and len(calls) == 1, r1
        assert sorted(os.listdir(fd)) == ['field_46MHz_R075.fits', 'field_46MHz_R075.json'], os.listdir(fd)
        HS._FIELDS.clear()
        assert HS.tilefill_field(out, '46MHz', 'R075', rows)['out_md5'] == r1['out_md5'] and len(calls) == 1      # reused
        with open(fil, 'ab') as f:
            f.write(b'\0' * 2880)
        HS._FIELDS.clear()
        assert HS.tilefill_field(out, '46MHz', 'R075', rows)['result'] == 'PASS' and len(calls) == 2              # remade
        with open(unf, 'ab') as f:
            f.write(b'\0' * 2880)
        HS._FIELDS.clear()
        r4 = HS.tilefill_field(out, '46MHz', 'R075', rows)
        assert r4['result'] == 'FAIL' and 'HG13' in r4['error'] and len(calls) == 2, r4
        with open(os.path.join(fd, 'field_46MHz_R075.json')) as f:
            assert json.load(f)['result'] == 'FAIL'
        assert sorted(os.listdir(fd)) == ['field_46MHz_R075.fits', 'field_46MHz_R075.json'], os.listdir(fd)
    finally:
        HS.mosaic_pair, TF.band_increment = old
        HS._FIELDS.clear()
        shutil.rmtree(d, ignore_errors=True)


def test_excluded_in_unit():
    """Exclusion of a hybrid unit: a subband unit by its nights; a wideband or Full unit per member, from the member stack records
    (a night excluded in one member is absent from that member only); a member without a stack record refuses."""
    d = _tmp('tf_exc_')
    try:
        def stack(b, dates, dirty=()):
            p = os.path.join(d, f'13h_{b}', 'R075_stack.json')
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'w') as f:
                json.dump(dict(nights=[dict(date=x) for x in dates], dirty_nights_dropped=list(dirty)), f)
        ex = _Excl(('2025-05-07',))
        bad, reads = HS.excluded_in_unit(ex, d, '13h', '50MHz', 'R075', ['2025-05-06', '2025-05-07'])
        assert bad == {'50MHz': ['2025-05-07']} and reads == [], bad
        for b in CFG.hybrid.WIDEBAND_MEMBERS['Green']:
            stack(b, ['2025-05-06'] if b == '50MHz' else ['2025-05-06', '2025-05-07'])
        ex = _ExclBy({('50MHz', '2025-05-07')})
        bad, reads = HS.excluded_in_unit(ex, d, '13h', 'Green', 'R075', ['2025-05-06', '2025-05-07'])
        assert bad == {} and len(reads) == 5, (bad, reads)              # the night held only by members that admit it
        stack('50MHz', ['2025-05-06', '2025-05-07'])
        assert HS.excluded_in_unit(ex, d, '13h', 'Green', 'R075', [])[0] == {'50MHz': ['2025-05-07']}
        stack('50MHz', ['2025-05-06', '2025-05-07'], dirty=['2025-05-07'])  # dropped as dirty by the stack: not admitted
        assert HS.excluded_in_unit(ex, d, '13h', 'Green', 'R075', [])[0] == {}
        assert 'no stack record' in _raises(TF.TilefillRefusal, HS.excluded_in_unit, ex, d, '13h', 'Full', 'R075', [])   # Red, Blue
    finally:
        shutil.rmtree(d, ignore_errors=True)


class _ExclBy(_Excl):
    """Excludes the (subband, date) pairs given."""
    def excluded_constituents(self, lst, scope, dates):
        return sorted(x for x in dates if (scope, x) in self.bad)


def _md5f(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def _mosaic(path, tpfill):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = fits.BinTableHDU.from_columns([fits.Column(name='SIGNAL', format='E', array=np.zeros(12, np.float32))])
    if tpfill:
        t.header['TPFILL'] = True
    fits.HDUList([fits.PrimaryHDU(), t]).writeto(path, overwrite=True)


def test_mosaic_pair():
    d = _tmp('tf_mos_')
    try:
        mos, q13, q11 = (os.path.join(d, x) for x in ('MOS', 'Q13', 'Q11'))
        T = types.SimpleNamespace(FILLED_MOSAIC_RUN=mos, UNFILLED_MOSAIC_RUNS=(q13, q11), FILLED_MOSAIC_GLOB=CFG.hybrid_tilefill.FILLED_MOSAIC_GLOB)
        rel = 'Green/I/healpix/Green_I_deep_Taper_Robust-0.75_pbcorr_coadd_lsrecon_mosaic_jysr_feather_lsfill_allsky_20241218-20260419_N024.fits'
        assert 'filled Jy/sr mosaics' in _raises(TF.TilefillRefusal, HS.mosaic_pair, 'Green', 'R075', T)
        _mosaic(os.path.join(mos, rel), True)
        assert 'no unfilled twin' in _raises(TF.TilefillRefusal, HS.mosaic_pair, 'Green', 'R075', T)
        _mosaic(os.path.join(q11, rel), False)
        assert HS.mosaic_pair('Green', 'R075', T) == (os.path.join(mos, rel), os.path.join(q11, rel))
        _mosaic(os.path.join(q13, rel), False)                              # the lost-sky twin comes first
        assert HS.mosaic_pair('Green', 'R075', T)[1] == os.path.join(q13, rel)
        _mosaic(os.path.join(q13, rel), True)                               # a twin that is itself filled
        assert 'TPFILL' in _raises(TF.TilefillRefusal, HS.mosaic_pair, 'Green', 'R075', T)
        _mosaic(os.path.join(mos, rel.replace('N024', 'N023')), True)       # two filled maps
        assert '2 filled' in _raises(TF.TilefillRefusal, HS.mosaic_pair, 'Green', 'R075', T)
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith('test_') and callable(f):
            f(); n += 1
            print(f'ok {k}', flush=True)
    print(f'{n} passed')
