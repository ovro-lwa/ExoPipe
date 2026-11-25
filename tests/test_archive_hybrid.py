#!/usr/bin/env python3
"""The hybrid route, pull request PR-5b (plan/specs/hybrid.md section 4): `archive push --class hybrid` (lib/archive/hybrid.py, the
content-test class hybrid_archive_copy of lib/archive/tests.py) and the diffuse PB-map header rule
(lib/hybrid/common.pbdiffuse_header).

Data-free (CI); a synthetic hybrid --out (a done tilefill unit, its products, its reconstruction record, its diffuse PB map) and a
MIRROR archive root under $TMPDIR, the exclusion read stubbed:
  1. the config: HYBRID_RETIRED_KEYS = the ruled card lists of pb_apply and lib/hybrid/common; pbdiffuse_header leaves no
     SH* or master card (LSSTKADM included);
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
  5b. a unit the work run did not finish (its stage records; steps.hybrid.unit_state, refused before a done tilefill): unfinished
     refuses; refused with nothing installed refuses as a gap; refused with installed products refuses until
     hybrid_unit.REFUSED_INSTALLED rules it: keep (keep rows) or quarantine (retire rows, moved into the round quarantine with
     manifest rows, no push manifest); run_hybrid makes it a push unit of its own;
  6. master.write_diffuse_pb: a clean PB-map header (no SH* or master card, no PRODNAME, the repair cards kept); a current map kept;
     a map with other values or a master card rewritten; Full's left to the full stage.
  7. the README unit (lib/archive/readme.py): the changed units in words; a plan stages the installed README + one section
     (content test archive_readme; a note line starting with UPDATE is no second section); an apply refuses before its first unit
     without the run's README or an --update-note citing a ledger row; an apply describes what the archive holds and the README
     does not name (this push's manifest, the round's retire rows; production-format manifests ignored), swaps the README in (the
     old one quarantined as README/<stamp>/README.txt, in no push manifest; content tests archive_readme_copy: checksum, history
     kept); a rerun without --resume adds nothing; a push whose README unit stopped is described by the next push of the round;
     a whole-name match (..._R0 is not named by ..._R075).
Cluster parity (not here): the PR-5 parity products of 18h 46 MHz Robust -0.75 pushed into a scratch mirror holding the installed
unfilled masters, and the plan replay of the 11-13 lost-sky tile swap; run by hand on lwacalim07.
Run: python tests/test_archive_hybrid.py   (or under pytest)
"""
import hashlib
import csv
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
from exopipe.phase3.lib.hybrid import common as HC  # noqa: E402
from exopipe.phase3.lib.masters import pb_apply as PBA  # noqa: E402

TAPER = '46MHz_I_deep_Taper_Robust-0.75_shflux_mdewarp{pb}_dewarped_aligned_coadd_lsrecon{fill}{cr}_LST18h_20250405-20251014_N{n}.fits'
RECON = '46MHz_I_deep_NoTaper_Robust-0.75_shflux_mdewarp{pb}_dewarped_aligned_coadd_lsrecon_LST18h_20250405-20251014_N{n}.fits'
NIGHTS = ['2025-04-05', '2025-05-07', '2025-10-14']
REPAIR = dict(P2RUN='r', SCALE_A=1.0, MDWSTAT='applied')
SH = dict(SHSCALE=1.0, SHERR=0.1)
MASTER = dict(FLUXFIT='x', IMGRMS=1.0, THERMNOI=1.0, LSSTKADM='82MHz')   # LSSTKADM: a Robust -0.75 master's admission card


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
        pbd = _fits(os.path.join(wd, 'PBdiffuse_deep_46MHz.fits'), 0.5, prodname=False, BUNIT='', PBMODEL='m')
        rec['pbdiffuse'] = 'in memory: the fill\'s PB-source description'        # the fill's own key (not the map's record)
        rec['pbdiffuse_map'] = dict(path=pbd, md5=_md5(pbd))
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
        self.old = (CFG.hybrid.ARCHIVE_MASTERS, exclusions.for_tier, CFG.hybrid_full.PBDIFFUSE_WEIGHTING)
        _fits(os.path.join(d, 'M', '18h', '46MHz', 'PB_deep_R075_46MHz.fits'), 0.9, prodname=False, BUNIT='',
              **(pb_cards or {}))
        self.ctx = types.SimpleNamespace(archive_root=os.path.join(d, 'root'), algo='md5', stamp='20261118T000000Z', dry=False,
                                          apply=True, guard_writes=lambda paths, what, allow=None: K.Ctx.guard_writes(None, paths, what, allow))

    def __enter__(self):
        setattr(CFG.hybrid, 'ARCHIVE_MASTERS', os.path.join(self.d, 'M'))
        setattr(CFG.hybrid_full, 'PBDIFFUSE_WEIGHTING', 'R075')        # the fixture's unit owns the cell's diffuse map
        self.E.for_tier = lambda tier, fresh=False: _Excl(self.excluded)
        return self

    def __exit__(self, *a):
        setattr(CFG.hybrid, 'ARCHIVE_MASTERS', self.old[0])
        self.E.for_tier = self.old[1]
        setattr(CFG.hybrid_full, 'PBDIFFUSE_WEIGHTING', self.old[2])

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
    assert tuple(R['pbdiffuse']) == PBA.PB_MAP_SH_CARDS + HC.PBDIFFUSE_MASTER_CARDS, R['pbdiffuse']
    h = fits.Header()
    for k, v in {**SH, **MASTER, 'BUNIT': 'Jy/beam'}.items():
        h[k] = v
    assert HC.pbdiffuse_header_problems(h)
    HC.pbdiffuse_header(h)
    assert not HC.pbdiffuse_header_problems(h), HC.pbdiffuse_header_problems(h)
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


def test_replace_same_pixels_retire():
    d = _tmp('ah_rep_')
    try:
        out = _unit(d, cleanres=False)
        with _Env(d) as E:
            a = E.adir()
            old = _names('014')
            olds = {k: _fits(os.path.join(a, old[k]), 1.0, **REPAIR) for k in ('plain', 'pbcorr', 'cleanres', 'recon', 'recon_pbcorr')}
            # the diffuse map installed with the same pixels and an older header: replaced (only a byte-identical file is kept)
            _fits(os.path.join(a, 'PBdiffuse_deep_46MHz.fits'), 0.5, prodname=False, BUNIT='', PBMODEL='old', HISTORY='h')
            # the full PB map installed with other pixels and the SH cards it lost since: replaced (retired keys)
            _fits(os.path.join(a, 'PB_deep_R075_46MHz.fits'), 0.8, prodname=False, BUNIT='', **SH)
            rows, u, notes = E.apply(out, 'T2')
            acts = {r['kind']: r['action'] for r in rows}
            assert acts == dict(plain='replace', pbcorr='replace', recon='replace', recon_pbcorr='replace', pbdiffuse='replace',
                                pb='replace', cleanres='retire'), acts
            q = AH.quarantine_dir(E.ctx, 'T2')
            for k, p in olds.items():
                assert not os.path.exists(p), p
                assert os.path.exists(os.path.join(q, '18h', '46MHz', os.path.basename(p))), (k, p)
            for n in ('PB_deep_R075_46MHz.fits', 'PBdiffuse_deep_46MHz.fits'):
                assert os.path.exists(os.path.join(q, '18h', '46MHz', n)), n
            qman = os.path.join(q, AH.QUARANTINE_MANIFEST)
            ok, _v, why = AT.rows_quarantined(qman)
            assert ok, why
            assert _audit(u) == 6
            assert any('no cleanres' in n for n in notes), notes
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
            _fits(os.path.join(out, '18h_46MHz', 'PBdiffuse_deep_46MHz.fits'), 0.5, prodname=False, BUNIT='', PBMODEL='m2')
            assert 'changed since' in _raises(Refusal, E.plan, out)      # the map's recorded checksum (PR-5b review)
            shutil.rmtree(out)
            out = _unit(d)
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
            _fits(os.path.join(a, _names('014')['recon']), 1.0, LSRECON=True)   # two reconstructions: not this unit's kinds
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
        assert not HC.pbdiffuse_header_problems(h) and 'PRODNAME' not in h and h['P2RUN'] == 'r', repr(h)
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



def test_pbdiffuse_owner():
    d = _tmp('ah_own_')
    try:
        out = _unit(d)
        with _Env(d) as E:
            CFG.hybrid_full.PBDIFFUSE_WEIGHTING = 'R0'                  # another weighting owns the map: this unit does not push it
            rows, _r, _n = E.plan(out)
            assert 'pbdiffuse' not in {r['kind'] for r in rows}, rows
            CFG.hybrid_full.PBDIFFUSE_WEIGHTING = 'R075'
            tp = os.path.join(out, '18h_46MHz', 'R075_tilefill.json')
            rec = json.load(open(tp))
            rec.pop('pbdiffuse_map')
            json.dump(rec, open(tp, 'w'))
            try:
                E.plan(out)
                raise AssertionError('no refusal')
            except Refusal as e:
                assert 'no diffuse PB map recorded' in str(e), e
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_swap_and_copy_failures():
    d = _tmp('ah_fail_')
    old_commit, old_copy = K.commit_part, K.verified_copy
    try:
        out = _unit(d)
        with _Env(d) as E:
            a = E.adir()
            olds = {k: _fits(os.path.join(a, _names('014')[k]), 1.0, **REPAIR) for k in ('plain', 'pbcorr')}
            # a copy fails part way (OSError after its .part began): every .part of the push removed, nothing moved
            n = {'c': 0}

            def bad_copy(src, dst, *x, **kw):
                n['c'] += 1
                if n['c'] == 2:
                    open(dst + '.part', 'w').close()
                    raise OSError(28, 'No space left on device')
                return old_copy(src, dst, *x, **kw)
            K.verified_copy = bad_copy
            try:
                E.apply(out, 'F1')
                raise AssertionError('no failure')
            except OSError:
                pass
            K.verified_copy = old_copy
            assert not [f for f in os.listdir(a) if f.endswith('.part')], os.listdir(a)
            assert all(os.path.exists(p) for p in olds.values())
            # a swap stops part way: the written files recorded, the unswapped .part removed, a refusal naming the state
            m = {'c': 0}

            def bad_commit(part, dst, replace=False):
                m['c'] += 1
                if m['c'] == 2:
                    raise OSError(5, 'Input/output error')
                return old_commit(part, dst, replace=replace)
            K.commit_part = bad_commit
            try:
                E.apply(out, 'F2')
                raise AssertionError('no refusal')
            except Refusal as e:
                assert 'the swap stopped' in str(e) and '1 written' in str(e), e
            K.commit_part = old_commit
            assert not [f for f in os.listdir(a) if f.endswith('.part')], os.listdir(a)
            pman = os.path.join(AH.archive_dir(E.ctx), f'MANIFEST_{E.ctx.stamp}.csv')
            assert len(list(csv.DictReader(open(pman)))) == 1
            # a rerun (another round) keeps what was written and completes the rest
            rows, u, _n = E.apply(out, 'F3')
            acts = [r['action'] for r in rows]
            assert acts.count('keep') == 1 and 'retire' not in acts, acts
            assert not [f for f in os.listdir(a) if f.endswith('.part')]
            assert all(os.path.exists(r['dst']) for r in rows)
    finally:
        K.commit_part, K.verified_copy = old_commit, old_copy
        shutil.rmtree(d, ignore_errors=True)


def _records(out, unit, staged=False, **stages):
    d = os.path.join(out, '_records', 'hybrid')
    os.makedirs(d, exist_ok=True)
    for s, st in stages.items():
        with open(os.path.join(d, f'{s}.{unit}.json'), 'w') as f:
            json.dump(dict(status=st, reason=f'{s} {st} (synthetic)', **({'staged_from': {}} if staged else {})), f)


def test_refused_units():
    from exopipe.phase3.steps import hybrid as SH
    d = _tmp('ah_ref_')
    old = dict(CFG.hybrid_unit.REFUSED_INSTALLED)
    try:
        out = os.path.join(d, 'out')
        _records(out, '18h_46MHz_R075', stack='done', recon='refused')
        _records(out, '18h_50MHz_R075', stack='done', recon='done', hybrid='running')
        _records(out, '18h_55MHz_R075', stack='done', recon='refused', hybrid='done', tilefill='done')   # a stale done tilefill
        _records(out, '19h_46MHz_R075', stack='done', recon='done', hybrid='done', tilefill='done')
        _records(out, '18h_69MHz_R075', staged=True, stack='done')      # a member stack staged from production: an input
        _records(out, '18h_64MHz_R075', stack='done')                   # a stack the run made, recon not started: unfinished
        _records(out, '18h_59MHz_R075', stack='done', recon='error', hybrid='done', tilefill='done')    # a rerun that crashed
        S = {k: SH.unit_state(v) for k, v in SH.work_states(out).items()}
        assert S[('18h', '46MHz', 'R075')][:3] == ('refused', 'recon', 'refused'), S
        assert S[('18h', '50MHz', 'R075')][:3] == ('unfinished', 'hybrid', 'running'), S
        assert S[('18h', '55MHz', 'R075')][:2] == ('refused', 'recon') and S[('19h', '46MHz', 'R075')][0] == 'done', S
        assert S[('18h', '69MHz', 'R075')][0] == 'input' and S[('18h', '64MHz', 'R075')][:2] == ('unfinished', 'stack'), S
        assert S[('18h', '59MHz', 'R075')][:3] == ('unfinished', 'recon', 'error'), S
        assert SH.unit_state(SH.work_states(out)[('18h', '50MHz', 'R075')], through='hybrid')[:2] == ('unfinished', 'hybrid')
        assert SH.unit_state({'tilefill': ('done', '', False)}, through='hybrid')[0] == 'none'          # a pinned tilefill unit
        assert set(SH.work_states(out, lsts={'19h'})) == {('19h', '46MHz', 'R075')}
        with _Env(d) as E:
            st = S[('18h', '46MHz', 'R075')]
            assert 'unfinished' in _raises(Refusal, AH.refused_rows, E.ctx, '18h', '50MHz', 'R075', S[('18h', '50MHz', 'R075')])
            assert 'a gap' in _raises(Refusal, AH.refused_rows, E.ctx, '18h', '46MHz', 'R075', st)
            a = E.adir()
            olds = {k: _fits(os.path.join(a, _names('014')[k]), 1.0, **REPAIR) for k in ('plain', 'pbcorr', 'cleanres')}
            pbm = _fits(os.path.join(a, 'PB_deep_R075_46MHz.fits'), 0.8, prodname=False, BUNIT='')
            msg = _raises(Refusal, AH.refused_rows, E.ctx, '18h', '46MHz', 'R075', st)
            assert 'REFUSED_INSTALLED' in msg and 'stay silently' in msg and _names('014')['plain'] in msg, msg
            CFG.hybrid_unit.REFUSED_INSTALLED[('18h', '46MHz', 'R075')] = 'keep'
            rows, notes = AH.refused_rows(E.ctx, '18h', '46MHz', 'R075', st)
            assert sorted((r['kind'], r['action'], r['dst'], r['md5']) for r in rows) == sorted(
                (k, 'keep', p, _md5(p)) for k, p in olds.items()), rows
            assert notes and notes[0].startswith('keep by ruling') and 'recon' in notes[0], notes
            CFG.hybrid_unit.REFUSED_INSTALLED[('18h', '46MHz', 'R075')] = 'quarantine'
            rows, notes = AH.refused_rows(E.ctx, '18h', '46MHz', 'R075', st)
            assert {(r['kind'], r['action'], r['replaces']) for r in rows} == {(k, 'retire', p) for k, p in olds.items()}, rows
            u = _U()
            AH.apply_unit(E.ctx, u, rows, 'T9')
            q = AH.quarantine_dir(E.ctx, 'T9')
            assert all(not os.path.exists(p) and os.path.exists(os.path.join(q, '18h', '46MHz', os.path.basename(p)))
                       for p in olds.values())
            ok, _v, why = AT.rows_quarantined(os.path.join(q, AH.QUARANTINE_MANIFEST))
            assert ok, why
            assert os.path.exists(pbm) and not os.path.exists(os.path.join(AH.archive_dir(E.ctx), f'MANIFEST_{E.ctx.stamp}.csv'))
            assert [c for _p, c, _x in u.outputs] == ['quarantine_manifest'], u.outputs
            names = []
            args = types.SimpleNamespace(staging_root=out, round='T9', _stages=['transfer'], lsts=None, update_note=[])
            E.ctx.tier = 'deep'
            # the README gate refuses an apply before its first unit: no README, then no --update-note citing a ledger row
            msg = _raises(Refusal, AH.run_hybrid, E.ctx, args, None, lambda name, body, describe: names.append(name))
            assert 'no README.txt' in msg and 'nothing pushed' in msg and names == [], msg
            with open(os.path.join(AH.archive_dir(E.ctx), 'README.txt'), 'w') as f:
                f.write('test README\n')
            msg = _raises(Refusal, AH.run_hybrid, E.ctx, args, None, lambda name, body, describe: names.append(name))
            assert 'needs --update-note' in msg and names == [], msg
            old_fl = CFG.paths.FIX_LEDGER
            try:
                CFG.paths.FIX_LEDGER, args.update_note = _ledger(d, 'TEST-REFUSED-20261119'), ['test (TEST-REFUSED-20261119)']
                AH.run_hybrid(E.ctx, args, None, lambda name, body, describe: names.append(name))
                assert names == [f'push_hybrid_18h_{b}MHz_R075' for b in ('46', '50', '55', '59', '64')], names
                names.clear()
                AH.run_hybrid(E.ctx, types.SimpleNamespace(**{**vars(args), 'lsts': '19h'}), None,
                              lambda name, body, describe: names.append(name))
                assert names == [], names                   # 19h: done by its records, but no tilefill output to push here
            finally:
                CFG.paths.FIX_LEDGER = old_fl
    finally:
        CFG.hybrid_unit.REFUSED_INSTALLED.clear()
        CFG.hybrid_unit.REFUSED_INSTALLED.update(old)
        shutil.rmtree(d, ignore_errors=True)


def test_kept_refill():
    """A unit the work run refused, kept by ruling (hybrid_unit.REFUSED_INSTALLED keep), whose tilefill record is done with source
    pinned: the push replaces its filled masters with the re-filled ones and keeps its other installed products; without the ruling,
    or with a work-source tilefill record (stale), it is not a refill."""
    from exopipe.phase3.steps import hybrid as SH

    class _UM(_U):
        def __init__(self):
            super().__init__()
            self.meta, self.inputs = {}, []

        def add_input(self, p, extra=None):
            self.inputs.append(p)
    d = _tmp('ah_kref_')
    old = dict(CFG.hybrid_unit.REFUSED_INSTALLED)
    try:
        out = _unit(d, source='pinned', cleanres=False)
        _records(out, '18h_46MHz_R075', stack='done', recon='refused', hybrid='no-input')
        st = SH.unit_state(SH.work_states(out)[('18h', '46MHz', 'R075')])
        assert st[:2] == ('refused', 'recon'), st
        tp = os.path.join(out, '18h_46MHz', 'R075_tilefill.json')
        new = {k: json.load(open(tp))[k]['out_md5'] for k in ('plain', 'pbcorr')}
        with _Env(d) as E:
            a = E.adir()
            olds = {k: _fits(os.path.join(a, _names()[k]), 1.0, **REPAIR) for k in ('plain', 'pbcorr', 'cleanres')}
            olds_md5 = {k: _md5(p) for k, p in olds.items()}
            assert not AH.kept_refill(out, '18h', '46MHz', 'R075', st)                 # no ruling
            CFG.hybrid_unit.REFUSED_INSTALLED[('18h', '46MHz', 'R075')] = 'keep'
            assert AH.kept_refill(out, '18h', '46MHz', 'R075', st)
            with open(tp) as f:
                tf = json.load(f)
            with open(tp, 'w') as f:
                json.dump({**tf, 'source': 'work'}, f)
            assert 'stale' in _raises(Refusal, AH.kept_refill, out, '18h', '46MHz', 'R075', st)       # a work fill of a refused unit
            with open(tp, 'w') as f:
                json.dump(tf, f)
            up = os.path.join(out, '_records', 'hybrid', 'tilefill.18h_46MHz_R075.json')
            with open(up) as f:
                ur = json.load(f)
            with open(up, 'w') as f:
                json.dump({**ur, 'status': 'refused', 'reason': 'HG13 synthetic'}, f)
            assert 'HG13 synthetic' in _raises(Refusal, AH.kept_refill, out, '18h', '46MHz', 'R075', st)   # a re-fill that failed
            with open(up, 'w') as f:
                json.dump(ur, f)
            with open(os.path.join(AH.archive_dir(E.ctx), 'README.txt'), 'w') as f:
                f.write('test README\n')
            old_fl, res = CFG.paths.FIX_LEDGER, {}

            def ufn(name, body, describe):
                if name == 'push_hybrid_18h_46MHz_R075':
                    u = _UM()
                    res[name] = (u, body(u))
            try:
                CFG.paths.FIX_LEDGER = _ledger(d, 'TEST-KEPT-20261120')
                args = types.SimpleNamespace(staging_root=out, round='T7', _stages=['transfer'], lsts=None,
                                             update_note=['test (TEST-KEPT-20261120)'])
                E.ctx.tier = 'deep'
                AH.run_hybrid(E.ctx, args, None, ufn)
            finally:
                CFG.paths.FIX_LEDGER = old_fl
            u, rows = res['push_hybrid_18h_46MHz_R075']
            assert sorted((r['kind'], r['action']) for r in rows) == [('cleanres', 'keep'), ('pbcorr', 'replace'), ('plain', 'replace')], rows
            assert {k: _md5(olds[k]) for k in ('plain', 'pbcorr')} == new and _md5(olds['cleanres']) == olds_md5['cleanres']
            q = AH.quarantine_dir(E.ctx, 'T7')
            assert all(_md5(os.path.join(q, '18h', '46MHz', os.path.basename(olds[k]))) == olds_md5[k] for k in ('plain', 'pbcorr'))
            assert u.meta['kept'] == [dict(path=olds['cleanres'], md5=olds_md5['cleanres'])], u.meta['kept']
            assert any('pinned tilefill' in n and 'pbcorr, plain pushed' in n for n in u.notes), u.notes
            assert _audit(u) == 2
    finally:
        CFG.hybrid_unit.REFUSED_INSTALLED.clear()
        CFG.hybrid_unit.REFUSED_INSTALLED.update(old)
        shutil.rmtree(d, ignore_errors=True)


class _RU:
    """A push unit as lib/archive/readme reads it (core.record.Unit's attributes)."""

    def __init__(self, name, status='done', counts=None, notes=(), inputs=(), manifests=()):
        self.name, self.status, self.reason = name, status, ''
        self.meta = {'counts': dict(counts or {})}
        self.notes, self.inputs = list(notes), {p: {} for p in inputs}
        self.outputs = [{'path': m, 'class': 'archive_manifest'} for m in manifests]

    def add_input(self, p, extra=None):
        self.inputs[p] = extra or {}

    def add_output(self, path, cls, **ctx):
        self.outputs.append({'path': path, 'class': cls, 'ctx': ctx})

    def note(self, s):
        self.notes.append(s)


def _ledger(d, lid):
    p = os.path.join(d, 'fix_ledger.csv')
    with open(p, 'w', newline='') as f:
        f.write(f'date,fix_id,files,description,authorized_by,status\r\n2026-11-19T07:00Z,{lid},,test only,,done\r\n')
    return p


def test_readme():
    from exopipe.phase3.core.record import NoInput
    from exopipe.phase3.lib.archive import readme as R
    assert R.summary('hybrid', ['push_hybrid_18h_46MHz_R075', 'push_hybrid_19h_46MHz_R075', 'push_hybrid_21h_46MHz_R075',
                                'push_hybrid_18h_Green_R0']) == '46MHz R075 at 18h-19h, 21h; Green R0 at 18h'
    assert R.summary('mosaic_v5', ['push_mosaic_v5_Green_R0', 'push_mosaic_v5_tables']) == 'Green R0; tables'
    d = _tmp('ahr_')
    old_ledger = CFG.paths.FIX_LEDGER
    try:
        CFG.paths.FIX_LEDGER = _ledger(d, 'TEST-README-20261119')
        ctx = types.SimpleNamespace(archive_root=os.path.join(d, 'root'), algo='md5', stamp='20261119T000000Z', dry=False, apply=False,
                                    manifest_dir=lambda kind: os.path.join(d, 'plans', kind),
                                    guard_writes=lambda paths, what, allow=None: K.Ctx.guard_writes(None, paths, what, allow))
        adir, qdir = AH.archive_dir(ctx), AH.quarantine_dir(ctx, 'T5')
        os.makedirs(adir)
        base = b'HYBRID NoTaper DEEP MASTERS (test)\nline two\n'
        with open(os.path.join(adir, 'README.txt'), 'wb') as f:
            f.write(base)
        run = types.SimpleNamespace(commit='abc1234', out_root=os.path.join(d, 'out'), step='archive')
        rec = os.path.join(d, 'work', '_records', 'hybrid', 'tilefill.18h_46MHz_R075.json')
        os.makedirs(os.path.dirname(rec))
        with open(rec, 'w') as f:
            json.dump(dict(status='done', commit='feed123'), f)
        seen = {}

        def unit_fn(name, body, describe=None):
            u = _RU(name)
            try:
                seen[name] = describe(body(u))
            except (Refusal, NoInput) as e:
                u.status, u.reason = ('refused' if isinstance(e, Refusal) else 'no-input'), str(e)
                seen[name] = f'{u.status.upper()} {e}'
            return u
        units = [_RU('push_hybrid_18h_46MHz_R075', counts=dict(replace=3, keep=1), inputs=[rec]),
                 _RU('push_hybrid_19h_46MHz_R075', counts=dict(keep=4)),                                # changed nothing
                 _RU('push_hybrid_19h_59MHz_R0', status='refused'),
                 _RU('push_hybrid_20h_59MHz_R0', counts=dict(retire=2), notes=['quarantine by ruling (hybrid_unit.REFUSED_INSTALLED): '
                                                                               'refused at recon; installed a, b'])]
        wd = os.path.join(d, 'work')
        args = types.SimpleNamespace(_stages=['manifest'], update_note=[])
        push = lambda us, a=args: R.push_readme(ctx, a, run, unit_fn, 'hybrid', adir, qdir, AH.QUARANTINE_MANIFEST, us, 'T5', wd)  # noqa
        flat = lambda b: ' '.join((b.decode() if isinstance(b, bytes) else b).split())                                         # noqa
        # a plan: this job's planned changes; nothing written into the archive
        u = push(units)
        assert u.status == 'done' and seen['push_hybrid_readme'].startswith('README section appended'), seen
        staged = next(o['path'] for o in u.outputs if o['class'] == 'archive_readme')
        ok, n, why = AT.readme_staged(staged, **next(o['ctx'] for o in u.outputs if o['class'] == 'archive_readme'))
        assert ok and n == 1, why
        txt = open(staged).read()
        assert txt.startswith(base.decode()) and '\n\nUPDATE T5: ' + R.NO_NOTE in txt, txt
        assert '2 unit(s) changed: 46MHz R075 at 18h; 59MHz R0 at 20h' in flat(txt), txt
        assert 'feed123' in txt and 'abc1234' in txt and 'Files: 3 replaced, 0 published, 2 retired, 1 kept' in flat(txt)
        assert 'By ruling: 20h_59MHz_R0: quarantine by ruling' in flat(txt)
        assert 'Not pushed in this job (1 unit(s)' in flat(txt) and '19h_59MHz_R0' in txt
        assert all(len(x) <= CFG.archive_push.README_WIDTH for x in txt.splitlines())
        assert any(o['class'] == 'csv_table' for o in u.outputs) and open(os.path.join(adir, 'README.txt'), 'rb').read() == base
        # the staged test: one section; a note line starting with UPDATE is no second section; a doubled section fails
        hdr = dict(prefix_md5=hashlib.md5(base).hexdigest(), prefix_bytes=len(base), round='T5')
        for body_, want in ((b'\nUPDATE T5: why\nUPDATE T4 was wrong (X)\n', True), (b'\nUPDATE T5: a\n\nUPDATE T5: a\n', False),
                            (b'\nwhy\n', False), (b'\nUPDATE T5: no newline', False)):
            with open(os.path.join(d, 'st.txt'), 'wb') as f:
                f.write(base + body_)
            assert AT.readme_staged(os.path.join(d, 'st.txt'), **hdr)[0] is want, body_
        assert R.named('push_hybrid_20h_59MHz_R0', 'a push_hybrid_20h_59MHz_R0.') and not R.named(
            'push_hybrid_20h_59MHz_R0', 'push_hybrid_20h_59MHz_R075 and x_push_hybrid_20h_59MHz_R0')
        # the gate before the first unit of an apply: the README and a note citing a ledger row; a plan or dry run passes
        ctx.apply, args._stages = True, ['transfer']
        for note, why in (([], 'none given'), (['why without an id'], 'no ledger ID')):
            args.update_note = note
            try:
                R.precheck(ctx, args, 'hybrid', adir)
                raise AssertionError('precheck passed')
            except Refusal as e:
                assert why in str(e) and 'nothing pushed' in str(e), e
        u = push(units)
        assert u.status == 'refused' and '--update-note' in u.reason, u.reason
        args.update_note = ['Q1 rebuild (TEST-README-20261119)']
        assert R.precheck(ctx, args, 'hybrid', adir) is None
        # an apply describes what the archive holds: this push's manifest (a production-format manifest is ignored) and the
        # retire rows of the round's quarantine manifest; the units' own counts are not trusted
        pman = os.path.join(adir, f'MANIFEST_{ctx.stamp}.csv')

        def product_push(man, unit, acts, retire_unit=None, nret=0):
            K.write_csv(man, CFG.archive_push.HYBRID_PUSH_COLUMNS, [
                dict(unit=unit, kind=f'k{i}', src='', dst=os.path.join(adir, f'{unit}_{i}.fits'), bytes='', md5='', action=a,
                     replaces='', written_at='x') for i, a in enumerate(acts)])
            if nret:
                with K.CsvAppender(os.path.join(qdir, AH.QUARANTINE_MANIFEST), CFG.archive_push.HYBRID_QUARANTINE_COLUMNS) as wq:
                    for i in range(nret):
                        wq.write(dict(unit=retire_unit, kind=f'k{i}', path=os.path.join(adir, f'{retire_unit}_{i}.fits'),
                                      dest=os.path.join(qdir, f'{retire_unit}_{i}.fits'), bytes='', md5='', replaced_by='',
                                      moved_at='x'))
        os.makedirs(qdir)
        with open(os.path.join(adir, 'MANIFEST_20261030T000000Z.csv'), 'w') as f:
            f.write('unit,src,dst,md5,bytes\nold,a,b,c,1\n')
        product_push(pman, 'push_hybrid_18h_46MHz_R075', ['replace'] * 3 + ['keep'], 'push_hybrid_20h_59MHz_R0', 2)
        u = push(units)
        assert u.status == 'done', u.reason
        new = open(os.path.join(adir, 'README.txt'), 'rb').read()
        q = os.path.join(qdir, 'README', '20261119T000000Z', 'README.txt')
        assert new.startswith(base) and b'UPDATE T5: Q1 rebuild (TEST-README-20261119)' in new and open(q, 'rb').read() == base
        assert 'Files: 3 replaced, 0 published, 2 retired, 1 kept' in flat(new) and 'MANIFEST_20261119T000000Z.csv' in flat(new)
        assert 'MANIFEST_20261030T000000Z.csv' not in flat(new) and 'push_hybrid_20h_59MHz_R0_1.fits' in flat(new), new
        cp = [o for o in u.outputs if o['class'] == 'archive_readme_copy']
        assert len(cp) == 1 and cp[0]['ctx']['replaces'] == q
        for t in (AT.copy_checksum, AT.readme_history_kept):
            ok, _v, why = t(cp[0]['path'], **cp[0]['ctx'])
            assert ok, (t.__name__, why)
        ok, _v, why = AT.rows_quarantined(os.path.join(qdir, AH.QUARANTINE_MANIFEST), paths=[os.path.join(adir, 'README.txt')])
        assert ok, why
        with open(pman) as f:                                       # the README is in no push manifest (every push replaces it)
            assert {r['unit'] for r in csv.DictReader(f)} == {'push_hybrid_18h_46MHz_R075'}
        assert not any(o['class'] == 'archive_manifest' for o in u.outputs)
        # a rerun without --resume after a complete push: nothing undescribed, no section
        ctx.stamp = '20261119T010000Z'
        u = push(units)
        assert u.status == 'no-input' and 'stays as installed' in u.reason and open(os.path.join(adir, 'README.txt'), 'rb').read() == new
        # a push whose README unit stopped: the next push of the round describes it, though this job changed nothing
        product_push(os.path.join(adir, 'MANIFEST_20261119T020000Z.csv'), 'push_hybrid_21h_59MHz_R0', ['replace'],
                     'push_hybrid_22h_59MHz_R0', 1)
        ctx.stamp = '20261119T030000Z'
        u = push([units[1]])
        two = open(os.path.join(adir, 'README.txt'), 'rb').read()
        assert u.status == 'done' and two.startswith(new) and two.count(b'\nUPDATE T5: ') == 2, u.reason
        sec = flat(two[len(new):])
        assert '2 unit(s) changed: 59MHz R0 at 21h-22h' in sec and '2 unit(s) applied by an earlier job' in sec, sec
        assert 'Files: 1 replaced, 0 published, 1 retired, 0 kept' in sec and 'push_hybrid_20h_59MHz_R0_0' not in sec, sec
        assert os.path.exists(os.path.join(qdir, 'README', '20261119T030000Z', 'README.txt'))
        # no README: the gate and the unit refuse; a push of no unit has no README unit
        os.remove(os.path.join(adir, 'README.txt'))
        try:
            R.precheck(ctx, args, 'hybrid', adir)
            raise AssertionError('precheck passed')
        except Refusal as e:
            assert 'no README.txt' in str(e), e
        u = push(units)
        assert u.status == 'refused' and 'no README.txt' in u.reason, u.reason
        assert push([None]) is None
    finally:
        CFG.paths.FIX_LEDGER = old_ledger
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('PASS', name)
