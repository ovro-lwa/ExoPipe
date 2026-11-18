#!/usr/bin/env python3
"""The hybrid route, pull request PR-7d (plan/specs/hybrid.md section 4): `archive push --class mosaic_v5` (lib/archive/mosaic_v5.py,
the content-test class mosaic_v5_archive_copy of lib/archive/tests.py).

Data-free (CI); a synthetic v5 --out (a done conform unit of Blue Taper R-0.75 with four products and the unfilled twins, its tile
list, a done v5-tables record) and a MIRROR archive root under $TMPDIR holding the 24 filled masters, the exclusion read stubbed:
  1. the config: the class enabled in 'replace-quarantine' mode, its quarantine prefix, the push and audit dispatch;
  2. an empty run: every product is published under <run>/Blue/I/<sub>/ (the unfilled twins are not pushed); the copies carry the
     sources' md5, the applied manifest lists them, the content audit passes; a second plan keeps all;
  3. installed files of another night span with other pixels: replaced (D2) - the old ones in the round quarantine
     mosaic/Run_HYBRID_QUARANTINE_<round>/ with a manifest row, the new ones in; an installed file of the same name and other bytes
     is replaced too;
  4. gates, each refusing the unit before any copy: record not done; a failed test; the tile list changed; a night now excluded;
     a product changed since conform; a PRODNAME that is not the file name; HG20 (a master not installed, its TPSRCMD5 or TPMOSMD5
     not the mosaic's); D7 (a replacement lacking a card key of the file it replaces, HDU 1 included); two installed files of a
     kind; a quarantine destination that exists (G4) leaves no .part behind;
  5. the tables unit: refused while the scope's products are not installed; then published, a changed table replaced into the
     round quarantine's tables/; a table changed since the tables stage, or a record without md5s, refuses;
  6. run_mosaic_v5: without --apply each unit writes its plan manifest and nothing reaches the run.
Run: python tests/test_archive_mosaic_v5.py   (or under pytest)
"""
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
from exopipe.phase3.core.record import Refusal  # noqa: E402
from exopipe.phase3.lib.archive import common as K  # noqa: E402
from exopipe.phase3.lib.archive import mosaic_v5 as AM  # noqa: E402
from exopipe.phase3.lib.archive import tests as AT  # noqa: E402
from exopipe.phase3.lib.mosaic import v5  # noqa: E402

LSTS = tuple(f'{h:02d}h' for h in range(24))
SPAN = '20241218-20260419_N024'
PRE = 'Blue_I_deep_Taper_Robust-0.75_'
PRODUCTS = {   # kind: (subdirectory, name less the span, HDU layout, BUNIT, PBCORR)
    'jysr': ('healpix', PRE + 'pbcorr_coadd_lsrecon_mosaic_jysr_feather_lsfill_allsky', 'hpx', 'Jy/sr', True),
    'jybeam': ('healpix', PRE + 'coadd_lsrecon_mosaic_jybeam_feather_lsfill_allsky', 'hpx', 'Jy/beam', False),
    'pbmap': ('healpix', PRE + 'coadd_lsrecon_mosaic_feather_lsfill_pbmap_allsky', 'hpx', '', False),
    'jysr_mollweide': ('mollweide', PRE + 'pbcorr_coadd_lsrecon_mosaic_jysr_feather_lsfill_molrender_allsky', 'img', 'Jy/sr', True)}
UNFILLED = ('unfilled_jysr', 'unfilled_jybeam')
MASTER = 'Blue_I_deep_Taper_Robust-0.75_shflux_pbcorr_dewarped_aligned_coadd_lsrecon_lsfill_LST{lst}_20241218-20251029_N008.fits'


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return str(e)
    raise AssertionError(f'{fn.__name__} did not raise {exc.__name__}')


def _md5(p):
    with open(p, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def _product(path, layout, bunit, pbcorr, value=1.0, extra0=None, extra1=None, prodname=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    name = prodname or os.path.basename(path)
    h0 = fits.Header()
    for k, v in dict(PRODNAME=name, NSTACK=24, PBCORR=pbcorr, **{'DATE-OBS': '2024-12-18T00:00:00', 'DATE-END': '2026-04-19T00:00:00'},
                     TIMEFIX='v2-mosaic-constituents', **(extra0 or {})).items():
        h0[k] = v
    if layout == 'img':
        h0['BUNIT'] = bunit
        fits.PrimaryHDU(np.full((4, 4), value, np.float32), h0).writeto(path, overwrite=True)
        return path
    t = fits.BinTableHDU.from_columns([fits.Column(name='SIGNAL', format='E', array=np.full(12, value, np.float32))])
    for k, v in dict(PRODNAME=name, BUNIT=bunit, NSIDE=1, **(extra1 or {})).items():
        t.header[k] = v
    fits.HDUList([fits.PrimaryHDU(header=h0), t]).writeto(path, overwrite=True)
    return path


class _Excl:
    path = None


class _U:
    def __init__(self):
        self.outputs, self.notes, self.meta, self.inputs = [], [], {}, []

    def add_output(self, path, cls, **ctx):
        self.outputs.append((path, cls, ctx))

    def add_input(self, p, extra=None):
        self.inputs.append(p)

    def note(self, s):
        self.notes.append(s)


def _record(out, unit, status='done', meta=None, tests=(), outputs=(), inputs=None):
    p = os.path.join(out, '_records', v5.STEP, f'{unit}.json')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        json.dump(dict(status=status, meta=meta or {}, tests=list(tests), outputs=list(outputs), inputs=dict(inputs or {})), f)
    return p


class _Env:
    """A v5 --out d/out with a done conform unit of Blue R075 and the mirror archive d/root holding the 24 filled masters filled from
    its Jy/sr product; the exclusion read and steps.hybrid.excluded_in_unit stubbed (bad: {LST: dates})."""

    def __init__(self, d, value=2.0, bad=None):
        from exopipe.phase3.core import exclusions
        from exopipe.phase3.steps import hybrid as SH
        self.d, self.bad = d, dict(bad or {})
        self.E, self.SH = exclusions, SH
        self.old = (exclusions.for_tier, SH.excluded_in_unit)
        self.out = os.path.join(d, 'out')
        self.root = os.path.join(d, 'root')
        self.ctx = types.SimpleNamespace(archive_root=self.root, algo='md5', stamp='20261118T000000Z', dry=False, apply=True,
                                          tier='deep', manifest_dir=self._mdir,
                                          guard_writes=lambda paths, what, allow=None: K.Ctx.guard_writes(None, paths, what, allow))
        P = self.P = v5.paths(self.out, 'Blue', 'R075')
        self.prods = {}
        for kind, (sub, stem, lay, bu, pbc) in PRODUCTS.items():
            self.prods[kind] = _product(os.path.join(P['conform'], 'Blue', 'I', sub, f'{stem}_{SPAN}.fits'), lay, bu, pbc, value)
        for kind in UNFILLED:
            sub, stem, lay, bu, pbc = PRODUCTS[kind[len('unfilled_'):]]
            self.prods[kind] = _product(os.path.join(P['conform'], 'unfilled', 'Blue', 'I', sub, f'{stem}_{SPAN}.fits'), lay, bu, pbc, 0.5)
        os.makedirs(os.path.dirname(P['tiles']), exist_ok=True)
        self.tiles = [dict(lst=lst, kind='hybrid', source='pinned', md5=f'tile{lst}', nights=['2025-01-01'],
                           nights_from=os.path.join(d, 'hyb', f'{lst}_Blue', 'R075_hybrid.json')) for lst in LSTS]
        with open(P['tiles'], 'w') as f:
            json.dump(dict(tiles=self.tiles), f)
        ah = os.path.join(CFG.paths.PHASE3_ARCHIVE_ROOT, os.path.relpath(CFG.hybrid.ARCHIVE_HYBRID, CFG.paths.PHASE3_ARCHIVE_ROOT))
        self.prov = [os.path.join(ah, lst, 'Blue', MASTER.format(lst=lst)) for lst in LSTS]
        self.conform()
        self.masters()

    def _mdir(self, name):
        p = os.path.join(self.d, 'plans', name)
        os.makedirs(p, exist_ok=True)
        return p

    def conform(self, status='done', tests=(), drop=()):
        meta = dict(tiles_md5=_md5(self.P['tiles']), prov_files=self.prov,
                    products={k: dict(path=p, md5=_md5(p)) for k, p in self.prods.items() if k not in drop})
        return _record(self.out, v5.unit_name('conform', 'Blue', 'R075'), status, meta, tests)

    def masters(self, jysr_md5=None, skip=(), src=None):
        jn, jm = os.path.basename(self.prods['jysr']), jysr_md5 or _md5(self.prods['jysr'])
        for t, f in zip(self.tiles, self.prov):
            a = AM.archived(self.ctx, f)
            if t['lst'] in skip:
                if os.path.exists(a):
                    os.remove(a)
                continue
            os.makedirs(os.path.dirname(a), exist_ok=True)
            h = fits.Header()
            h['TPSRCMD5'] = (src or {}).get(t['lst'], t['md5'])
            h['TPMOSAIC'] = jn
            h['TPMOSMD5'] = jm
            fits.PrimaryHDU(np.zeros((2, 2), np.float32), h).writeto(a, overwrite=True)

    def tables(self, names=('summary.csv', 'taper_only.csv'), md5=True, units=('Blue_R075',)):
        td = v5.tables_dir(self.out)
        os.makedirs(td, exist_ok=True)
        outs, m = [], {}
        from exopipe.phase3.lib.mosaic import conform_v5 as CF
        for n in names:
            p = os.path.join(td, n)
            CF._write(p, CF.SUMMARY_COLS if n == 'summary.csv' else CF.TAPER_ONLY_COLS,
                      [['x', 'Taper_R075', 'Blue', 'jysr'] + [''] * (len(CF.SUMMARY_COLS) - 5) + ['PASS']] if n == 'summary.csv' else [])
            outs.append(dict(path=p, **{'class': 'mosaic_v5_table'}))
            m[n] = _md5(p)
        ins = {}                                     # the installed tables the stage merged from (v5.tables_unit archived())
        for n in names:
            dst = os.path.join(AM.run_dir(self.ctx), 'tables', n)
            if os.path.exists(dst):
                ins[os.path.join(CFG.hybrid_tilefill.FILLED_MOSAIC_RUN, 'tables', n)] = {'md5': _md5(dst)}
        return _record(self.out, v5.TABLES_UNIT, 'done', dict(units=list(units), **(dict(md5=m) if md5 else {})), (), outs, ins)

    def __enter__(self):
        self.E.for_tier = lambda tier, fresh=False: _Excl()
        self.SH.excluded_in_unit = lambda excl, rundir, lst, scope, w, nights: ({scope: self.bad[lst]} if lst in self.bad else {}, [])
        return self

    def __exit__(self, *a):
        self.E.for_tier, self.SH.excluded_in_unit = self.old

    def plan(self):
        return AM.scope_plan(self.ctx, self.out, 'Blue', 'R075')[0]

    def apply(self, rows, rnd='T1'):
        from exopipe.phase3.lib.archive import hybrid as AH
        rd, qd = AM.run_dir(self.ctx), AM.quarantine_dir(self.ctx, rnd)
        u = _U()
        AH.apply_unit(self.ctx, u, rows, rnd, cls='mosaic_v5', adir=rd, qdir=qd, qman_name=AM.QUARANTINE_MANIFEST,
                      copy_class="mosaic_v5_archive_copy")
        return u

    def dst(self, kind):
        sub = PRODUCTS[kind][0]
        return os.path.join(AM.run_dir(self.ctx), 'Blue', 'I', sub, os.path.basename(self.prods[kind]))


def _tmp(prefix):
    return tempfile.mkdtemp(prefix=prefix, dir=os.environ.get('TMPDIR') or None)


def _audit(u):
    n = 0
    for path, cls, ctx in u.outputs:
        if cls != 'mosaic_v5_archive_copy':
            continue
        for t in (AT.v5_copy_checksum, AT.v5_copy_header, AT.v5_copy_card_superset):
            ok, _v, why = t(path, **ctx)
            assert ok, (t.__name__, path, why)
        n += 1
    return n


def test_config():
    c = CFG.archive_push.CLASSES['mosaic_v5']
    assert c['enabled'] and c['existing'] == 'replace-quarantine' and c['tiers'] == ('deep',)
    assert CFG.archive_push.MOSAIC_V5_QUARANTINE_PREFIX == 'mosaic/Run_HYBRID_QUARANTINE_'
    from exopipe.phase3.lib.archive import push as PU
    assert PU.AUDIT_CLASSES['mosaic_v5'] == 'archive_manifest'
    assert os.path.relpath(AM.run_dir(types.SimpleNamespace(archive_root='/R')), '/R') == \
        os.path.relpath(CFG.hybrid_tilefill.FILLED_MOSAIC_RUN, CFG.paths.PHASE3_ARCHIVE_ROOT)
    assert AM.kind_key(PRE + 'x_allsky_20241218-20260419_N024.fits') == AM.kind_key(PRE + 'x_allsky_20250101-20260101_N023.fits')


def test_publish_keep():
    d = _tmp('am_pub_')
    try:
        with _Env(d) as E:
            rows = E.plan()
            assert {r['kind']: r['action'] for r in rows} == {k: 'publish' for k in PRODUCTS}, rows
            assert {r['dst'] for r in rows} == {E.dst(k) for k in PRODUCTS}
            u = E.apply(rows)
            for r in rows:
                assert os.path.exists(r['dst']) and _md5(r['dst']) == _md5(r['src']) == r['md5'] and not os.path.exists(r['dst'] + '.part')
            assert _audit(u) == len(PRODUCTS)
            assert not os.path.exists(os.path.join(AM.run_dir(E.ctx), 'unfilled'))           # the unfilled twins stay in --out
            man = os.path.join(AM.run_dir(E.ctx), 'MANIFEST_20261118T000000Z.csv')
            ok, _v, why = AT.rows_transferred(man, paths=[r['dst'] for r in rows])
            assert ok, why
            assert not os.path.exists(AM.quarantine_dir(E.ctx, 'T1'))
            assert {r['action'] for r in E.plan()} == {'keep'}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_replace():
    d = _tmp('am_rep_')
    try:
        with _Env(d) as E:
            old = {}
            for kind in ('jysr', 'pbmap'):          # another night span, other pixels
                sub, stem, lay, bu, pbc = PRODUCTS[kind]
                old[kind] = _product(os.path.join(os.path.dirname(E.dst(kind)), f'{stem}_20241218-20260301_N023.fits'), lay, bu, pbc, 9.0)
                with fits.open(old[kind], mode='update', memmap=False) as h:
                    h[0].header['NSTACK'] = 23
            sub, stem, lay, bu, pbc = PRODUCTS['jybeam']   # the same name, other bytes
            old['jybeam'] = _product(E.dst('jybeam'), lay, bu, pbc, 7.0)
            rows = E.plan()
            acts = {r['kind']: r['action'] for r in rows}
            assert acts == dict(jysr='replace', pbmap='replace', jybeam='replace', jysr_mollweide='publish'), acts
            assert {r['kind']: r['replaces'] for r in rows if r['action'] == 'replace'} == old
            u = E.apply(rows)
            assert _audit(u) == 4
            qd = AM.quarantine_dir(E.ctx, 'T1')
            for kind, p in old.items():
                q = os.path.join(qd, os.path.relpath(p, AM.run_dir(E.ctx)))
                assert os.path.exists(q) and (kind == 'jybeam' or not os.path.exists(p)), (kind, q)
            with open(os.path.join(qd, AM.QUARANTINE_MANIFEST), newline='') as f:
                qrows = list(csv.DictReader(f))
            assert sorted(r['kind'] for r in qrows) == ['jybeam', 'jysr', 'pbmap'] and all(r['md5'] for r in qrows), qrows
            assert {r['action'] for r in E.plan()} == {'keep'}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_gates():
    d = _tmp('am_gate_')
    try:
        with _Env(d) as E:
            E.conform(status='refused')
            assert 'not done' in _raises(Refusal, E.plan)
            E.conform(tests=[dict(name='product_conformant', passed=False)])
            assert 'failed content test' in _raises(Refusal, E.plan)
            E.conform(drop=('jysr',))
            assert 'no Jy/sr product' in _raises(Refusal, E.plan)
            E.conform()
            with open(E.P['tiles'], 'a') as f:
                f.write(' ')
            assert 'tile list' in _raises(Refusal, E.plan)
            E.conform()
            E.bad = {'07h': ['2025-01-01']}
            assert 'now excluded' in _raises(Refusal, E.plan)
            E.bad = {}
            with fits.open(E.prods['pbmap'], mode='update', memmap=False) as h:
                h[0].header['X'] = 1
            assert 'changed since conform' in _raises(Refusal, E.plan)
            _product(E.prods['pbmap'], 'hpx', '', False, 2.0, prodname='other.fits')
            E.conform()
            assert 'PRODNAME' in _raises(Refusal, E.plan)
            _product(E.prods['pbmap'], 'hpx', '', False, 2.0)
            E.conform()
            E.masters(skip=('05h',))
            assert 'HG20' in _raises(Refusal, E.plan) and 'not installed' in _raises(Refusal, E.plan)
            E.masters(jysr_md5='0' * 32)
            assert 'HG20' in _raises(Refusal, E.plan) and 'TPMOSMD5' in _raises(Refusal, E.plan)
            E.masters(src={'11h': 'other'})
            assert 'TPSRCMD5' in _raises(Refusal, E.plan)
            E.masters()
            E.plan()                                                    # every gate clear again
            # D7: the installed file holds a card key (HDU 1) the new one lacks
            sub, stem, lay, bu, pbc = PRODUCTS['jybeam']
            _product(os.path.join(os.path.dirname(E.dst('jybeam')), f'{stem}_20241218-20260301_N023.fits'), lay, bu, pbc, 9.0,
                     extra1={'TPJBCONV': 1.0})
            assert 'D7' in _raises(Refusal, E.plan) and '1:TPJBCONV' in _raises(Refusal, E.plan)
            _product(os.path.join(os.path.dirname(E.dst('jybeam')), f'{stem}_20241218-20260302_N023.fits'), lay, bu, pbc, 9.0)
            assert 'more than one installed' in _raises(Refusal, E.plan)
            shutil.rmtree(os.path.dirname(E.dst('jybeam')))
            # G4: a quarantine destination that exists; nothing copied, no .part left
            old = _product(E.dst('pbmap'), 'hpx', '', False, 5.0)
            q = os.path.join(AM.quarantine_dir(E.ctx, 'T1'), os.path.relpath(old, AM.run_dir(E.ctx)))
            os.makedirs(os.path.dirname(q))
            open(q, 'w').write('x')
            rows = E.plan()
            assert 'G4' in _raises(Refusal, E.apply, rows)
            assert not os.path.exists(E.dst('jysr')) and not [f for f in os.listdir(os.path.dirname(E.dst('pbmap'))) if f.endswith('.part')]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_tables():
    d = _tmp('am_tab_')
    try:
        with _Env(d) as E:
            E.tables()
            assert 'not installed' in _raises(Refusal, AM.tables_plan, E.ctx, E.out)
            E.apply(E.plan())
            rows, _r = AM.tables_plan(E.ctx, E.out)
            assert {r['kind'] for r in rows} == {'table'} and {r['action'] for r in rows} == {'publish'}, rows
            rd = AM.run_dir(E.ctx)
            assert {r['dst'] for r in rows} == {os.path.join(rd, 'tables', n) for n in ('summary.csv', 'taper_only.csv')}
            u = E.apply(rows)
            assert _audit(u) == 2
            assert {r['action'] for r in AM.tables_plan(E.ctx, E.out)[0]} == {'keep'}
            with open(os.path.join(rd, 'tables', 'taper_only.csv'), 'a') as f:      # another push changed the installed table:
                f.write('a,b,c\r\n')                                                # refused until the stage merges again
            assert 'another push changed it' in _raises(Refusal, AM.tables_plan, E.ctx, E.out)
            E.tables()                                                               # merged from the installed table: replaced
            rows = AM.tables_plan(E.ctx, E.out)[0]
            assert {r['dst'].split('/')[-1]: r['action'] for r in rows} == {'summary.csv': 'keep', 'taper_only.csv': 'replace'}
            E.apply(rows, rnd='T2')
            assert os.path.exists(os.path.join(AM.quarantine_dir(E.ctx, 'T2'), 'tables', 'taper_only.csv'))
            with open(os.path.join(v5.tables_dir(E.out), 'summary.csv'), 'a') as f:
                f.write('\r\n')
            assert 'changed since the tables stage' in _raises(Refusal, AM.tables_plan, E.ctx, E.out)
            E.tables()
            cp = os.path.join(E.out, '_records', v5.STEP, v5.unit_name('conform', 'Blue', 'R075') + '.json')
            old = open(cp).read()
            cr = json.loads(old)
            cr['meta']['tables'] = {'content': [os.path.join(E.out, 'content_new.json'), 'f' * 32]}     # conform ran again since
            open(cp, 'w').write(json.dumps(cr))
            assert 'earlier conform' in _raises(Refusal, AM.tables_plan, E.ctx, E.out)
            open(cp, 'w').write(old)
            assert 'waived failure is never pushed' in _raises(Refusal, AM._done, dict(status='done', tests=[
                dict(name='TEST 1', passed=False, waived='R-X')]), cp)
            E.tables(md5=False)
            assert 'no md5 recorded' in _raises(Refusal, AM.tables_plan, E.ctx, E.out)
            E.tables(units=())
            assert 'lists no conform unit' in _raises(Refusal, AM.tables_plan, E.ctx, E.out)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_run_plan_only():
    d = _tmp('am_run_')
    try:
        with _Env(d) as E:
            E.tables()
            E.ctx.apply = False
            seen = {}

            def unit_fn(name, body, describe=None):
                u = _U()
                try:
                    seen[name] = (describe(body(u)), u)
                except Refusal as e:
                    seen[name] = ('REFUSED ' + str(e)[:60], u)
            args = types.SimpleNamespace(staging_root=E.out, round='T1', _stages=('manifest',))
            AM.run_mosaic_v5(E.ctx, args, None, unit_fn)
            assert list(seen) == ['push_mosaic_v5_Blue_R075', 'push_mosaic_v5_tables'], seen
            assert seen['push_mosaic_v5_Blue_R075'][0] == 'publish 4', seen
            assert seen['push_mosaic_v5_tables'][0].startswith('REFUSED the tables describe products not installed'), seen
            man = seen['push_mosaic_v5_Blue_R075'][1].outputs[0][0]
            assert man.startswith(os.path.join(d, 'plans', 'push_mosaic_v5')) and os.path.exists(man)
            assert not os.path.exists(AM.run_dir(E.ctx))
            assert 'staging-root' in _raises(ValueError, AM.v5_units, os.path.join(d, 'none'))
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for f in fns:
        f()
        print('PASS', f.__name__)
    print(f'{len(fns)} passed')
