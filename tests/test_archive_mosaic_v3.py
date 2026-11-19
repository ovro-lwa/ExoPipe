#!/usr/bin/env python3
"""The hybrid route, pull request PR-6e (plan/specs/hybrid.md section 4): `archive push --class mosaic_v3` (lib/archive/mosaic_v3.py,
the content-test class mosaic_v3_archive_copy of lib/archive/tests.py).

Data-free (CI); a synthetic v3 --out (done conform units of Blue I_Taper_R075 and Blue Vclean with their products, tile lists over 24
tile masters, a done v3-tables record) and a MIRROR archive root under $TMPDIR, the exclusion read stubbed:
  1. the config: the class enabled in 'replace-quarantine' mode, its quarantine prefix, the push and audit dispatch; the run is
     mosaic_v3.ARCHIVE_RUN;
  2. an empty run: every product is published under <run>/Blue/I/<sub>/ (Stokes V under <run>/Blue/V/<sub>/); the copies carry the
     sources' md5, the applied manifest lists them, the content audit passes; a second plan keeps all;
  3. installed files of another night span with other pixels: replaced (D2) - the old ones in the round quarantine
     mosaic/Run_FEATHER3_QUARANTINE_<round>/ with a manifest row, the new ones in; an installed file of the same name and other
     bytes is replaced too;
  4. gates, each refusing the unit before any copy: record not done; a failed test; no Jy/sr product; the tile list changed; a night
     now excluded; HG13 (a tile master replaced or gone since the stage); a product changed since conform; a PRODNAME that is not
     the file name; a product outside <scope>/<stokes>; D7 (a replacement lacking a card key of the file it replaces, HDU 1
     included); two installed files of a kind; a quarantine destination that exists (G4) leaves no .part behind;
  5. the tables unit: refused while the scopes' products are not installed; then published, a changed table replaced into the
     round quarantine's tables/; a table changed since the tables stage, a table made from an earlier conform, or a record without
     md5s refuses;
  6. run_mosaic_v3: without --apply each unit writes its plan manifest and nothing reaches the run.
Run: python tests/test_archive_mosaic_v3.py   (or under pytest)
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
from exopipe.phase3.lib.archive import mosaic_v3 as AM  # noqa: E402
from exopipe.phase3.lib.archive import tests as AT  # noqa: E402
from exopipe.phase3.lib.mosaic import v3  # noqa: E402

LSTS = tuple(f'{h:02d}h' for h in range(24))
SPAN = '20241218-20251029_N024'
J_I, J_V = v3.Job('I', 'Taper_R075'), v3.Job('Vclean', 'Taper_R0')
PRE = {J_I: 'Blue_I_deep_Taper_Robust-0.75_', J_V: 'Blue_V_deep_Taper_Robust+0.0_vclean_'}
PRODUCTS = {   # kind: (subdirectory, name between the prefix and the span, HDU layout, BUNIT, PBCORR)
    'jysr': ('healpix', 'pbcorr_coadd_mosaic_jysr_feather_allsky', 'hpx', 'Jy/sr', True),
    'jybeam': ('healpix', 'coadd_mosaic_jybeam_feather_allsky', 'hpx', 'Jy/beam', False),
    'pbmap': ('healpix', 'coadd_mosaic_feather_pbmap_allsky', 'hpx', '', False),
    'jysr_mollweide': ('mollweide', 'pbcorr_coadd_mosaic_jysr_feather_molrender_allsky', 'img', 'Jy/sr', True)}


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
    for k, v in dict(PRODNAME=name, NSTACK=24, PBCORR=pbcorr, **{'DATE-OBS': '2024-12-18T00:00:00', 'DATE-END': '2025-10-29T00:00:00'},
                     **(extra0 or {})).items():
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

    def __init__(self, bad):
        self.bad = bad

    def excluded_constituents(self, lst, scope, dates):
        return [x for x in self.bad.get(lst, []) if x in dates]


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
    p = os.path.join(out, '_records', v3.STEP, f'{unit}.json')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        json.dump(dict(status=status, meta=meta or {}, tests=list(tests), outputs=list(outputs), inputs=dict(inputs or {})), f)
    return p


class _Env:
    """A v3 --out d/out with done conform units of Blue I_Taper_R075 and Blue Vclean, each over 24 tile masters in d/masters, and the
    mirror archive d/root; the exclusion read stubbed (bad: {LST: dates})."""

    def __init__(self, d, value=2.0, bad=None):
        from exopipe.phase3.core import exclusions
        self.d, self.bad = d, dict(bad or {})
        self.E = exclusions
        self.old = exclusions.for_tier
        self.out = os.path.join(d, 'out')
        self.root = os.path.join(d, 'root')
        self.ctx = types.SimpleNamespace(archive_root=self.root, algo='md5', stamp='20261119T000000Z', dry=False, apply=True,
                                          tier='deep', manifest_dir=self._mdir,
                                          guard_writes=lambda paths, what, allow=None: K.Ctx.guard_writes(None, paths, what, allow))
        self.P, self.prods, self.tiles = {}, {}, {}
        for j, st in ((J_I, 'I'), (J_V, 'V')):
            P = self.P[j] = v3.paths(self.out, 'Blue', j)
            self.prods[j] = {k: _product(os.path.join(P['conform'], 'Blue', st, sub, f'{PRE[j]}{stem}_{SPAN}.fits'), lay, bu, pbc, value)
                             for k, (sub, stem, lay, bu, pbc) in PRODUCTS.items()}
            tiles = []
            for lst in LSTS:
                m = os.path.join(d, 'masters', j.label, f'{lst}.fits')
                os.makedirs(os.path.dirname(m), exist_ok=True)
                fits.PrimaryHDU(np.full((2, 2), float(LSTS.index(lst)), np.float32)).writeto(m, overwrite=True)
                tiles.append(dict(lst=lst, kind='plain', path=m, md5=_md5(m), nights=['2025-01-01', '2025-02-01']))
            self.tiles[j] = tiles
            os.makedirs(os.path.dirname(P['tiles']), exist_ok=True)
            with open(P['tiles'], 'w') as f:
                json.dump(dict(scope='Blue', job=j.label, tiles=tiles), f)
            self.conform(j)

    def _mdir(self, name):
        p = os.path.join(self.d, 'plans', name)
        os.makedirs(p, exist_ok=True)
        return p

    def conform(self, j=J_I, status='done', tests=(), drop=(), tables=None):
        P = self.P[j]
        meta = dict(tiles_md5=_md5(P['tiles']), prov_files=[t['path'] for t in self.tiles[j]],
                    products={k: dict(path=p, md5=_md5(p)) for k, p in self.prods[j].items() if k not in drop},
                    tables=tables or {'content': [P['content'], 'c' * 32]})
        return _record(self.out, v3.unit_name('conform', 'Blue', j), status, meta, tests)

    def tables(self, names=('summary.csv', 'selection_20261022.csv'), md5=True, conformed=('Blue I_Taper_R075', 'Blue Vclean')):
        from exopipe.phase3.lib.mosaic import conform_v3 as CF
        from exopipe.phase3.lib.mosaic import conform_v5 as CF5
        td = v3.tables_dir(self.out)
        os.makedirs(td, exist_ok=True)
        outs, m = [], {}
        for n in names:
            p = os.path.join(td, n)
            if n == 'summary.csv':
                CF5._write(p, CF.SUMMARY_COLS, [['x', 'Taper_R075', 'Blue', 'jysr'] + [''] * (len(CF.SUMMARY_COLS) - 5) + ['PASS']])
            else:
                with open(p, 'w', newline='') as f:            # a carried table: production's bytes (LF)
                    f.write('lst,band,variant\n04h,Blue,Taper_R075\n')
            outs.append(dict(path=p, **{'class': 'mosaic_v3_table'}))
            m[n] = _md5(p)
        ins = {self.P[j]['content']: {'md5': 'c' * 32} for j in (J_I, J_V)}   # each conform's content report (its record's md5)
        for n in names:                                   # the installed tables the stage merged from (v3.tables_unit archived())
            dst = os.path.join(AM.run_dir(self.ctx), 'tables', n)
            if os.path.exists(dst):
                ins[os.path.join(CFG.mosaic_v3.ARCHIVE_RUN, 'tables', n)] = {'md5': _md5(dst)}
        return _record(self.out, v3.TABLES_UNIT, 'done', dict(conformed=list(conformed), **(dict(md5=m) if md5 else {})), (), outs, ins)

    def __enter__(self):
        self.E.for_tier = lambda tier, fresh=False: _Excl(self.bad)
        return self

    def __exit__(self, *a):
        self.E.for_tier = self.old

    def plan(self, j=J_I):
        return AM.scope_plan(self.ctx, self.out, 'Blue', j)[0]

    def apply(self, rows, rnd='T1'):
        from exopipe.phase3.lib.archive import hybrid as AH
        rd, qd = AM.run_dir(self.ctx), AM.quarantine_dir(self.ctx, rnd)
        u = _U()
        AH.apply_unit(self.ctx, u, rows, rnd, cls='mosaic_v3', adir=rd, qdir=qd, qman_name=AM.QUARANTINE_MANIFEST,
                      copy_class='mosaic_v3_archive_copy')
        return u

    def dst(self, kind, j=J_I):
        return os.path.join(AM.run_dir(self.ctx), 'Blue', 'V' if j == J_V else 'I', PRODUCTS[kind][0],
                            os.path.basename(self.prods[j][kind]))


def _tmp(prefix):
    return tempfile.mkdtemp(prefix=prefix, dir=os.environ.get('TMPDIR') or None)


def _audit(u):
    n = 0
    for path, cls, ctx in u.outputs:
        if cls != 'mosaic_v3_archive_copy':
            continue
        for t in (AT.v3_copy_checksum, AT.v3_copy_header, AT.v3_copy_card_superset):
            ok, _v, why = t(path, **ctx)
            assert ok, (t.__name__, path, why)
        n += 1
    return n


def test_config():
    c = CFG.archive_push.CLASSES['mosaic_v3']
    assert c['enabled'] and c['existing'] == 'replace-quarantine' and c['tiers'] == ('deep',)
    assert CFG.archive_push.MOSAIC_V3_QUARANTINE_PREFIX == 'mosaic/Run_FEATHER3_QUARANTINE_'
    assert CFG.archive_push.MOSAIC_V3_RETIRED_KEYS == ()
    from exopipe.phase3.lib.archive import push as PU
    assert PU.AUDIT_CLASSES['mosaic_v3'] == 'archive_manifest'
    assert os.path.relpath(AM.run_dir(types.SimpleNamespace(archive_root='/R')), '/R') == \
        os.path.relpath(CFG.mosaic_v3.ARCHIVE_RUN, CFG.paths.PHASE3_ARCHIVE_ROOT) == 'mosaic/Run_FEATHER3_20241218-20251029'
    assert 'archive:mosaic_v3' not in CFG.paths.PRODUCTION_WRITES_ALLOWED            # no ruling opens the production archive to it
    js = AM.all_jobs()
    assert list(js) == list(CFG.mosaic_v3.SCOPES) and [j.label for j in js['Blue']] == [
        'I_Taper_R075', 'I_Taper_R0', 'I_Taper_R075_ssub-rst8', 'I_Taper_R0_ssub-rst8', 'Vclean']


def test_publish_keep():
    d = _tmp('am3_pub_')
    try:
        with _Env(d) as E:
            for j in (J_I, J_V):
                rows = E.plan(j)
                assert {r['kind']: r['action'] for r in rows} == {k: 'publish' for k in PRODUCTS}, rows
                assert {r['dst'] for r in rows} == {E.dst(k, j) for k in PRODUCTS}
                u = E.apply(rows)
                for r in rows:
                    assert os.path.exists(r['dst']) and _md5(r['dst']) == _md5(r['src']) == r['md5']
                    assert not os.path.exists(r['dst'] + '.part')
                assert _audit(u) == len(PRODUCTS)
                assert {r['action'] for r in E.plan(j)} == {'keep'}
            man = os.path.join(AM.run_dir(E.ctx), 'MANIFEST_20261119T000000Z.csv')
            ok, _v, why = AT.rows_transferred(man, paths=[E.dst(k, j) for k in PRODUCTS for j in (J_I, J_V)])
            assert ok, why
            assert not os.path.exists(AM.quarantine_dir(E.ctx, 'T1'))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_replace():
    d = _tmp('am3_rep_')
    try:
        with _Env(d) as E:
            old = {}
            for kind in ('jysr', 'pbmap'):          # another night span, other pixels
                sub, stem, lay, bu, pbc = PRODUCTS[kind]
                old[kind] = _product(os.path.join(os.path.dirname(E.dst(kind)), f'{PRE[J_I]}{stem}_20241218-20250827_N023.fits'), lay,
                                     bu, pbc, 9.0)
            sub, stem, lay, bu, pbc = PRODUCTS['jybeam']   # the same name, other bytes
            old['jybeam'] = _product(E.dst('jybeam'), lay, bu, pbc, 7.0)
            rows = E.plan()
            acts = {r['kind']: r['action'] for r in rows}
            assert acts == dict(jysr='replace', pbmap='replace', jybeam='replace', jysr_mollweide='publish'), acts
            assert {r['kind']: r['replaces'] for r in rows if r['action'] == 'replace'} == old
            u = E.apply(rows)
            assert _audit(u) == 4
            qd = AM.quarantine_dir(E.ctx, 'T1')
            assert qd == os.path.join(E.root, 'mosaic', 'Run_FEATHER3_QUARANTINE_T1')
            for kind, p in old.items():
                q = os.path.join(qd, os.path.relpath(p, AM.run_dir(E.ctx)))
                assert q.startswith(os.path.join(qd, 'Blue', 'I', PRODUCTS[kind][0])), q            # production's layout
                assert os.path.exists(q) and (kind == 'jybeam' or not os.path.exists(p)), (kind, q)
            with open(os.path.join(qd, AM.QUARANTINE_MANIFEST), newline='') as f:
                qrows = list(csv.DictReader(f))
            assert sorted(r['kind'] for r in qrows) == ['jybeam', 'jysr', 'pbmap'] and all(r['md5'] for r in qrows), qrows
            assert {r['action'] for r in E.plan()} == {'keep'}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_gates():
    d = _tmp('am3_gate_')
    try:
        with _Env(d) as E:
            E.conform(status='refused')
            assert 'not done' in _raises(Refusal, E.plan)
            E.conform(tests=[dict(name='product_conformant', passed=False)])
            assert 'failed content test' in _raises(Refusal, E.plan)
            E.conform(drop=('jysr',))
            assert 'no Jy/sr product' in _raises(Refusal, E.plan)
            E.conform()
            with open(E.P[J_I]['tiles'], 'a') as f:
                f.write(' ')
            assert 'tile list' in _raises(Refusal, E.plan)
            E.conform()
            E.bad = {'07h': ['2025-02-01']}
            assert 'now excluded' in _raises(Refusal, E.plan) and '07h' in _raises(Refusal, E.plan)
            E.bad = {}
            m = E.tiles[J_I][5]['path']                                   # HG13: a tile master replaced since the stage
            keep = open(m, 'rb').read()
            open(m, 'ab').write(b'\0' * 2880)
            assert 'HG13' in _raises(Refusal, E.plan) and '05h' in _raises(Refusal, E.plan)
            E.ctx.dry = True                                              # a dry run checks existence only
            assert {r['action'] for r in AM.scope_plan(E.ctx, E.out, 'Blue', J_I, checksum=False)[0]} == {'publish'}
            E.ctx.dry = False
            os.remove(m)
            assert 'HG13' in _raises(Refusal, AM.scope_plan, E.ctx, E.out, 'Blue', J_I, checksum=False)
            open(m, 'wb').write(keep)
            with fits.open(E.prods[J_I]['pbmap'], mode='update', memmap=False) as h:
                h[0].header['X'] = 1
            assert 'changed since conform' in _raises(Refusal, E.plan)
            _product(E.prods[J_I]['pbmap'], 'hpx', '', False, 2.0, prodname='other.fits')
            E.conform()
            assert 'PRODNAME' in _raises(Refusal, E.plan)
            _product(E.prods[J_I]['pbmap'], 'hpx', '', False, 2.0)
            good = E.prods[J_I]['pbmap']                                  # a Stokes I product under <scope>/V
            E.prods[J_I]['pbmap'] = _product(good.replace('/Blue/I/', '/Blue/V/'), 'hpx', '', False, 2.0)
            E.conform()
            assert 'is not under' in _raises(Refusal, E.plan) and '/Blue/I' in _raises(Refusal, E.plan)
            os.remove(E.prods[J_I]['pbmap'])
            E.prods[J_I]['pbmap'] = good
            E.conform()
            E.plan()                                                      # every gate clear again
            # D7: the installed file holds a card key (HDU 1) the new one lacks
            sub, stem, lay, bu, pbc = PRODUCTS['jybeam']
            _product(os.path.join(os.path.dirname(E.dst('jybeam')), f'{PRE[J_I]}{stem}_20241218-20250827_N023.fits'), lay, bu, pbc, 9.0,
                     extra1={'TPJBCONV': 1.0})
            assert 'D7' in _raises(Refusal, E.plan) and '1:TPJBCONV' in _raises(Refusal, E.plan)
            _product(os.path.join(os.path.dirname(E.dst('jybeam')), f'{PRE[J_I]}{stem}_20241218-20250828_N023.fits'), lay, bu, pbc, 9.0)
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
    d = _tmp('am3_tab_')
    try:
        with _Env(d) as E:
            E.tables()
            assert 'not installed' in _raises(Refusal, AM.tables_plan, E.ctx, E.out)
            E.apply(E.plan(J_I))
            assert 'Blue Vclean' in _raises(Refusal, AM.tables_plan, E.ctx, E.out)          # the Stokes V products not yet in
            E.apply(E.plan(J_V))
            rows, _r = AM.tables_plan(E.ctx, E.out)
            assert {r['kind'] for r in rows} == {'table'} and {r['action'] for r in rows} == {'publish'}, rows
            rd = AM.run_dir(E.ctx)
            assert {r['dst'] for r in rows} == {os.path.join(rd, 'tables', n) for n in ('summary.csv', 'selection_20261022.csv')}
            u = E.apply(rows)
            assert _audit(u) == 2
            assert {r['action'] for r in AM.tables_plan(E.ctx, E.out)[0]} == {'keep'}
            with open(os.path.join(rd, 'tables', 'summary.csv'), 'a') as f:            # another push changed the installed table:
                f.write('a,b,c\r\n')                                                   # refused until the stage merges again
            assert 'another push changed it' in _raises(Refusal, AM.tables_plan, E.ctx, E.out)
            E.tables()                                                                  # merged from the installed table: replaced
            rows = AM.tables_plan(E.ctx, E.out)[0]
            assert {r['dst'].split('/')[-1]: r['action'] for r in rows} == {'summary.csv': 'replace', 'selection_20261022.csv': 'keep'}
            E.apply(rows, rnd='T2')
            assert os.path.exists(os.path.join(AM.quarantine_dir(E.ctx, 'T2'), 'tables', 'summary.csv'))
            with open(os.path.join(v3.tables_dir(E.out), 'summary.csv'), 'a') as f:
                f.write('\r\n')
            assert 'changed since the tables stage' in _raises(Refusal, AM.tables_plan, E.ctx, E.out)
            E.tables()
            E.conform(J_V, tables={'content': [os.path.join(E.out, 'content_new.json'), 'f' * 32]})   # conform ran again since
            assert 'earlier conform' in _raises(Refusal, AM.tables_plan, E.ctx, E.out)
            E.conform(J_V)
            E.tables(md5=False)
            assert 'no md5 recorded' in _raises(Refusal, AM.tables_plan, E.ctx, E.out)
            E.tables(conformed=())
            assert 'lists no conform unit' in _raises(Refusal, AM.tables_plan, E.ctx, E.out)
            E.tables(conformed=('Blue I_NoTaper',))
            assert 'not a mosaic_v3 job' in _raises(Refusal, AM.tables_plan, E.ctx, E.out)
            ok, _v, why = AT.v3_table_columns(os.path.join(rd, 'tables', 'selection_20261022.csv'))
            assert ok and 'carried' in why, why
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_run_plan_only():
    d = _tmp('am3_run_')
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
            AM.run_mosaic_v3(E.ctx, args, None, unit_fn)
            assert list(seen) == ['push_mosaic_v3_Blue_I_Taper_R075', 'push_mosaic_v3_Blue_Vclean', 'push_mosaic_v3_tables'], seen
            assert seen['push_mosaic_v3_Blue_I_Taper_R075'][0] == 'publish 4', seen
            assert seen['push_mosaic_v3_tables'][0].startswith('REFUSED the tables describe products not installed'), seen
            man = seen['push_mosaic_v3_Blue_I_Taper_R075'][1].outputs[0][0]
            assert man.startswith(os.path.join(d, 'plans', 'push_mosaic_v3')) and os.path.exists(man)
            assert not os.path.exists(AM.run_dir(E.ctx))
            assert 'staging-root' in _raises(ValueError, AM.v3_units, os.path.join(d, 'none'))
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for f in fns:
        f()
        print('PASS', f.__name__)
    print(f'{len(fns)} passed')
