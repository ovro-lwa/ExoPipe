#!/usr/bin/env python3
"""The hybrid route, pull request PR-5b (plan/specs/hybrid.md section 4): `archive push --class hybrid` (lib/archive/hybrid.py, the
content-test class hybrid_archive_copy of lib/archive/tests.py) and the diffuse PB-map header rule (pb_apply.pbdiffuse_header).

Data-free (CI); a synthetic hybrid --out (a done tilefill unit, its products, its reconstruction record, its diffuse PB map) and a
MIRROR archive root under $TMPDIR, the exclusion read stubbed:
  1. the config: HYBRID_RETIRED_KEYS = the ruled card lists of pb_apply; pbdiffuse_header leaves no SH* or master card;
  2. an empty archive: every file of a work unit is published (filled master, beam-corrected sibling, CLEAN residual,
     reconstruction and its split, diffuse PB map, full PB map from the masters archive); the copies carry the sources' md5, the
     applied manifest lists them, the content audit passes; a second plan keeps all;
  3. installed files of another night span with other pixels: replaced (D2) - the old ones in the round quarantine with a manifest
     row (path, dest, md5), the new ones in, D7 superset passes; an installed file of the same name and pixels but another header
     is kept; an installed product kind the rebuild no longer makes is retired into the quarantine;
  4. gates, each refusing the unit before any copy: unit not done; record without nights; nights differ from the unit record's;
     a night now excluded; a source changed since its stage; a product whose PRODNAME is not its file name; a diffuse PB map
     carrying a master card or a full PB map carrying an SH* card (HG12 / D6); a replacement lacking a card key of the file it
     replaces (D7) - while a PB map lacking only retired keys passes; two installed files of a kind; a quarantine destination that
     exists (G4) leaves no .part behind;
  5. a pinned unit pushes only the tilefill outputs and retires nothing;
  6. master.write_diffuse_pb: a clean PB-map header (no SH* or master card, no PRODNAME, the repair cards kept); a current map kept;
     a map with other values or a master card rewritten; Full's left to the full stage.
Cluster parity (not here): the PR-5 parity products of 18h 46 MHz Robust -0.75 pushed into a scratch mirror holding the installed
unfilled masters, and the plan replay of the 11-13 lost-sky tile swap; run by hand on lwacalim07.
Run: python tests/test_archive_hybrid.py   (or under pytest)
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
from exopipe.phase3.core.record import Refusal  # noqa: E402
from exopipe.phase3.lib.archive import common as K  # noqa: E402
from exopipe.phase3.lib.archive import hybrid as AH  # noqa: E402
from exopipe.phase3.lib.archive import tests as AT  # noqa: E402
from exopipe.phase3.lib.masters import pb_apply as PBA  # noqa: E402

TAPER = '46MHz_I_deep_Taper_Robust-0.75_shflux_mdewarp{pb}_dewarped_aligned_coadd_lsrecon{fill}{cr}_LST18h_20250405-20251014_N{n}.fits'
RECON = '46MHz_I_deep_NoTaper_Robust-0.75_shflux_mdewarp{pb}_dewarped_aligned_coadd_lsrecon_LST18h_20250405-20251014_N{n}.fits'
NIGHTS = ['2025-04-05', '2025-05-07', '2025-10-14']
REPAIR = dict(P2RUN='r', SCALE_A=1.0, MDWSTAT='applied')
SH = dict(SHSCALE=1.0, SHERR=0.1)
MASTER = dict(FLUXFIT='x', IMGRMS=1.0, THERMNOI=1.0)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    raise AssertionError(f'{fn.__name__} did not raise {exc.__name__}')


def _md5(p):
    with open(p, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def _fits(path, value=1.0, prodname=True, **cards):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h = fits.Header()
    if prodname:
        h['PRODNAME'] = os.path.basename(path) if prodname is True else prodname
    for k, v in cards.items():
        h[k] = v
    fits.PrimaryHDU(np.full((4, 4), value, np.float32), h).writeto(path, overwrite=True)
    return path


def _names(n='015'):
    return dict(plain=TAPER.format(pb='', fill='_lsfill', cr='', n=n), pbcorr=TAPER.format(pb='_pbcorr', fill='_lsfill', cr='', n=n),
                cleanres=TAPER.format(pb='', fill='', cr='_cleanres', n=n), recon=RECON.format(pb='', n=n),
                recon_pbcorr=RECON.format(pb='_pbcorr', n=n))


class _Excl:
    path = None

    def __init__(self, bad=()):
        self.bad = set(bad)

    def excluded_constituents(self, lst, scope, dates):
        return sorted(x for x in dates if x in self.bad)


class _U:
    def __init__(self):
        self.outputs, self.notes = [], []

    def add_output(self, path, cls, **ctx):
        self.outputs.append((path, cls, ctx))

    def note(self, s):
        self.notes.append(s)


def _unit(d, source='work', cleanres=True, value=2.0, nights=NIGHTS, admitted=None, status='done', with_nights=True):
    """A done tilefill unit of 18h 46MHz R075 in the hybrid --out d/out; returns the out root."""
    out = os.path.join(d, 'out')
    wd = os.path.join(out, '18h_46MHz')
    n = _names()
    rec = dict(result='PASS', source=source, lst='18h', scope='46MHz', weighting='R075')
    for k in ('plain', 'pbcorr'):
        p = _fits(os.path.join(wd, 'tilefill', n[k]), value, TPFILL=True, GITTAG='new', **REPAIR)
        rec[k] = dict(out=p, out_md5=_md5(p), result='PASS')
    if cleanres:
        p = _fits(os.path.join(wd, 'tilefill', n['cleanres']), value, **REPAIR)
        rec['cleanres'] = dict(out=p, name=n['cleanres'], out_md5=_md5(p), result='PASS')
    if with_nights:
        rec['nights'] = list(nights)
        rec['nights_from'] = os.path.join(wd, 'R075_hybrid.json')
    if source == 'work':
        rr = {}
        for k, key in (('recon', 'product'), ('recon_pbcorr', 'product_pbcorr')):
            p = _fits(os.path.join(wd, f'I_deep_NoTaper_R075_lsrecon{"_pbcorr" if k == "recon_pbcorr" else ""}_46MHz.fits'), value,
                      prodname=n[k], LSRECON=True, **REPAIR)
            rr[key] = dict(path=p, prodname=n[k], md5=_md5(p))
        with open(os.path.join(wd, 'R075_lsrecon.json'), 'w') as f:
            json.dump(rr, f)
        _fits(os.path.join(wd, 'PBdiffuse_deep_46MHz.fits'), 0.5, prodname=False, BUNIT='', PBMODEL='m')
    with open(os.path.join(wd, 'R075_tilefill.json'), 'w') as f:
        json.dump(rec, f)
    os.makedirs(os.path.join(out, '_records', 'hybrid'), exist_ok=True)
    with open(os.path.join(out, '_records', 'hybrid', 'tilefill.18h_46MHz_R075.json'), 'w') as f:
        json.dump(dict(status=status, admitted=list(nights if admitted is None else admitted)), f)
    return out


class _Env:
    """hybrid.ARCHIVE_MASTERS -> d/M (with the cell's full PB map), the exclusion read stubbed; ctx on the mirror root d/root."""

    def __init__(self, d, excluded=(), pb_cards=None):
        from exopipe.phase3.core import exclusions
        self.d, self.excluded = d, excluded
        self.E = exclusions
        self.old = (CFG.hybrid.ARCHIVE_MASTERS, exclusions.for_tier)
        _fits(os.path.join(d, 'M', '18h', '46MHz', 'PB_deep_R075_46MHz.fits'), 0.9, prodname=False, BUNIT='',
              **(pb_cards or {}))
        self.ctx = types.SimpleNamespace(archive_root=os.path.join(d, 'root'), algo='md5', stamp='20261118T000000Z', dry=False,
                                          apply=True, guard_writes=lambda paths, what: K.Ctx.guard_writes(None, paths, what))

    def __enter__(self):
        setattr(CFG.hybrid, 'ARCHIVE_MASTERS', os.path.join(self.d, 'M'))
        self.E.for_tier = lambda tier, fresh=False: _Excl(self.excluded)
        return self

    def __exit__(self, *a):
        setattr(CFG.hybrid, 'ARCHIVE_MASTERS', self.old[0])
        self.E.for_tier = self.old[1]

    def adir(self):
        return AH.archive_dir(self.ctx, '18h', '46MHz')

    def plan(self, out):
        return AH.hybrid_plan(self.ctx, out, '18h', '46MHz', 'R075')

    def apply(self, out, rnd='T1'):
        rows, _r, notes = self.plan(out)
        u = _U()
        AH.apply_unit(self.ctx, u, rows, rnd)
        return rows, u, notes


def _tmp(prefix):
    return tempfile.mkdtemp(prefix=prefix, dir=os.environ.get('TMPDIR') or None)


def _audit(u):
    """Every hybrid_archive_copy output passes its three content tests; returns their count."""
    n = 0
    for path, cls, ctx in u.outputs:
        if cls != 'hybrid_archive_copy':
            continue
        for t in (AT.copy_checksum, AT.copy_header, AT.copy_card_superset):
            ok, _v, why = t(path, **ctx)
            assert ok, (t.__name__, path, why)
        n += 1
    return n


def test_config():
    R = CFG.archive_push.HYBRID_RETIRED_KEYS
    assert tuple(R['pb']) == PBA.PB_MAP_SH_CARDS, R['pb']
    assert tuple(R['pbdiffuse']) == PBA.PB_MAP_SH_CARDS + PBA.PBDIFFUSE_MASTER_CARDS, R['pbdiffuse']
    h = fits.Header()
    for k, v in {**SH, **MASTER, 'BUNIT': 'Jy/beam'}.items():
        h[k] = v
    assert PBA.pbdiffuse_header_problems(h)
    PBA.pbdiffuse_header(h)
    assert not PBA.pbdiffuse_header_problems(h), PBA.pbdiffuse_header_problems(h)
    assert CFG.archive_push.CLASSES['hybrid']['enabled'] and CFG.archive_push.CLASSES['hybrid']['existing'] == 'replace-quarantine'


def test_publish_keep():
    d = _tmp('ah_pub_')
    try:
        out = _unit(d)
        with _Env(d) as E:
            rows, u, _n = E.apply(out)
            acts = {r['kind']: r['action'] for r in rows}
            assert acts == {k: 'publish' for k in ('plain', 'pbcorr', 'cleanres', 'recon', 'recon_pbcorr', 'pbdiffuse', 'pb')}, acts
            for r in rows:
                assert os.path.exists(r['dst']) and _md5(r['dst']) == _md5(r['src']) == r['md5'], r
                assert not os.path.exists(r['dst'] + '.part')
            assert _audit(u) == 7
            man = os.path.join(AH.archive_dir(E.ctx), 'MANIFEST_20261118T000000Z.csv')
            ok, _v, why = AT.rows_transferred(man, paths=[r['dst'] for r in rows])
            assert ok, why
            assert not os.path.exists(AH.quarantine_dir(E.ctx, 'T1'))
            rows2, _r, _n = E.plan(out)
            assert {r['action'] for r in rows2} == {'keep'}, [(r['kind'], r['action']) for r in rows2]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_replace_keep_same_pixels_retire():
    d = _tmp('ah_rep_')
    try:
        out = _unit(d, cleanres=False)
        with _Env(d) as E:
            a = E.adir()
            old = _names('014')
            olds = {k: _fits(os.path.join(a, old[k]), 1.0, **REPAIR) for k in ('plain', 'pbcorr', 'cleanres', 'recon', 'recon_pbcorr')}
            # the diffuse map installed with the same pixels and an older header (a retired master card): kept
            _fits(os.path.join(a, 'PBdiffuse_deep_46MHz.fits'), 0.5, prodname=False, BUNIT='', PBMODEL='old', HISTORY='h')
            # the full PB map installed with other pixels and the SH cards it lost since: replaced (retired keys)
            _fits(os.path.join(a, 'PB_deep_R075_46MHz.fits'), 0.8, prodname=False, BUNIT='', **SH)
            rows, u, notes = E.apply(out, 'T2')
            acts = {r['kind']: r['action'] for r in rows}
            assert acts == dict(plain='replace', pbcorr='replace', recon='replace', recon_pbcorr='replace', pbdiffuse='keep',
                                pb='replace', cleanres='retire'), acts
            q = AH.quarantine_dir(E.ctx, 'T2')
            for k, p in olds.items():
                assert not os.path.exists(p), p
                assert os.path.exists(os.path.join(q, '18h', '46MHz', os.path.basename(p))), (k, p)
            assert os.path.exists(os.path.join(q, '18h', '46MHz', 'PB_deep_R075_46MHz.fits'))
            qman = os.path.join(q, AH.QUARANTINE_MANIFEST)
            ok, _v, why = AT.rows_quarantined(qman)
            assert ok, why
            assert _audit(u) == 5
            assert any('same pixels' in n for n in notes) and any('no cleanres' in n for n in notes), notes
            new = _names()
            for k in ('plain', 'pbcorr', 'recon', 'recon_pbcorr'):
                assert os.path.exists(os.path.join(a, new[k])), k
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_gates():
    d = _tmp('ah_gate_')
    try:
        with _Env(d) as E:
            for kw, want in ((dict(status='refused'), 'not done'), (dict(with_nights=False), 'nights'),
                             (dict(admitted=NIGHTS[:2]), 'admitted')):
                shutil.rmtree(os.path.join(d, 'out'), ignore_errors=True)
                why = _raises(Refusal, E.plan, _unit(d, **kw))
                assert want in why, (kw, why)
            shutil.rmtree(os.path.join(d, 'out'))
            out = _unit(d)
            E.excluded = ('2025-05-07',)
            assert 'excluded' in _raises(Refusal, E.plan, out)
            E.excluded = ()
            tf = json.load(open(os.path.join(out, '18h_46MHz', 'R075_tilefill.json')))
            _fits(tf['pbcorr']['out'], 3.0, TPFILL=True, **REPAIR)
            assert 'changed since' in _raises(Refusal, E.plan, out)
            shutil.rmtree(out)
            out = _unit(d)
            rr = json.load(open(os.path.join(out, '18h_46MHz', 'R075_lsrecon.json')))
            p = rr['product']['path']
            _fits(p, 2.0, prodname='46MHz_wrong.fits', LSRECON=True)
            rr['product']['md5'] = _md5(p)
            json.dump(rr, open(os.path.join(out, '18h_46MHz', 'R075_lsrecon.json'), 'w'))
            assert 'PRODNAME' in _raises(Refusal, E.plan, out)
            shutil.rmtree(out)
            out = _unit(d)
            _fits(os.path.join(out, '18h_46MHz', 'PBdiffuse_deep_46MHz.fits'), 0.5, prodname=False, BUNIT='', FLUXFIT='x')
            assert 'HG12' in _raises(Refusal, E.plan, out)
            _fits(os.path.join(out, '18h_46MHz', 'PBdiffuse_deep_46MHz.fits'), 0.5, prodname=False, BUNIT='')
        with _Env(d, pb_cards=SH) as E:
            assert 'HG12' in _raises(Refusal, E.plan, out)
        with _Env(d) as E:
            a = E.adir()
            old = _fits(os.path.join(a, _names('014')['plain']), 1.0, TPFILL=True, EXTRA='k', **REPAIR)
            why = _raises(Refusal, E.plan, out)
            assert 'D7' in why and 'EXTRA' in why, why
            os.remove(old)
            _fits(os.path.join(a, _names('014')['plain']), 1.0, **REPAIR)
            _fits(os.path.join(a, _names('013')['plain']), 1.0, **REPAIR)
            assert 'more than one' in _raises(Refusal, E.plan, out)
            shutil.rmtree(a)
            p = _fits(os.path.join(a, _names('014')['plain']), 1.0, **REPAIR)
            rows, _r, _n = E.plan(out)
            q = os.path.join(AH.quarantine_dir(E.ctx, 'T3'), '18h', '46MHz', os.path.basename(p))
            _fits(q, 1.0)
            why = _raises(Refusal, AH.apply_unit, E.ctx, _U(), rows, 'T3')
            assert 'G4' in why, why
            assert os.path.exists(p) and not [f for f in os.listdir(a) if f.endswith('.part')], os.listdir(a)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_pinned():
    d = _tmp('ah_pin_')
    try:
        out = _unit(d, source='pinned')
        with _Env(d) as E:
            a = E.adir()
            _fits(os.path.join(a, _names()['recon']), 1.0, LSRECON=True)
            rows, _r, _n = E.plan(out)
            assert sorted(r['kind'] for r in rows) == ['cleanres', 'pbcorr', 'plain'], rows
            assert {r['action'] for r in rows} == {'publish'}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_write_diffuse_pb():
    from exopipe.phase3.lib.hybrid import master as M
    M.bind()
    d = _tmp('ah_pbd_')
    old = M.diffuse_pb_map
    vals = {'v': 0.5}
    M.diffuse_pb_map = lambda run_dir, lst, scope, v='R075': (np.full((4, 4), vals['v']), 'test map / smearing term')
    try:
        wd = os.path.join(d, '18h_46MHz')
        p = _fits(os.path.join(wd, 'recon.fits'), 1.0, BUNIT='Jy/beam', **SH, **MASTER, **REPAIR)
        json.dump(dict(product=dict(path=p)), open(os.path.join(wd, 'R075_lsrecon.json'), 'w'))
        pbd = M.write_diffuse_pb(d, '18h', '46MHz', 'R075')
        h = fits.getheader(pbd)
        assert not PBA.pbdiffuse_header_problems(h) and 'PRODNAME' not in h and h['P2RUN'] == 'r', repr(h)
        ino = os.stat(pbd).st_ino
        assert M.write_diffuse_pb(d, '18h', '46MHz', 'R075') == pbd and os.stat(pbd).st_ino == ino      # current: kept
        vals['v'] = 0.6
        M.write_diffuse_pb(d, '18h', '46MHz', 'R075')
        assert os.stat(pbd).st_ino != ino and float(fits.getdata(pbd)[0, 0]) == np.float32(0.6)          # stale values: rewritten
        ino = os.stat(pbd).st_ino
        with fits.open(pbd, mode='update', memmap=False) as f:
            f[0].header['THERMNOI'] = 1.0
        M.write_diffuse_pb(d, '18h', '46MHz', 'R075')
        assert os.stat(pbd).st_ino != ino and 'THERMNOI' not in fits.getheader(pbd)                    # bad header: rewritten
        assert M.write_diffuse_pb(d, '18h', 'Full', 'R075').endswith('18h_Full/PBdiffuse_deep_Full.fits')
        assert not os.path.exists(os.path.join(d, '18h_Full'))                                           # Full: the full stage's
    finally:
        M.diffuse_pb_map = old
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('PASS', name)
