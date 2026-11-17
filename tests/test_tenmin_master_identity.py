#!/usr/bin/env python3
"""THE 10-MINUTE MASTER IDENTITY in v2's cumulative build (lib/serve/sublst_build.master_identity, gate G13; Gregg 2026-11-17,
R-GREGG-20261117-TENMIN-MASTER-IDENTITY-BEFORE-RELEASE1; production exopipe_ops fa669f0, sublst_cumulative.sh:102-108).

Real detections: the frozen 10-minute grouping inputs (refactor_wt_20261102/F_grouping/frozen_inputs/tenmin), Stokes I of the
adjacent LSTs 04h and 05h (their skies overlap; sources repeat there). They are grouped for real by design B (sublst_build.batch_grouping) and served
as production's per-LST builds serve them: one served identity per design-B identity AND LST, each named per LST (the J-name join
that split one source across LSTs). Then
  T1 served_identity_splits finds exactly the design-B identities served under two names (an independent count from the fixture)
     and names no unlinked detection and no key under two identities;
  T2 design B (the default): the outcome test REFUSES that build (G13) and rewrites nothing;
  T3 production's grouping (the parity file): batch_grouping + production's LIVE sublst_master_identity.py (run in place, as the
     stage runs it) merge them; G13 passes; every design-B identity is one served name, the J-name (source_id) of its row with the
     smallest pos_err_arcsec; every column other than the identity fields is unchanged, row for row; the record lists every
     served identity that was merged;
  T4 refusals (design B): a served member detection the grouping does not hold; a (source_id, band) under two identities; no
     present table for a served Stokes; a present table changed since G8 read it.
Cluster-only (the frozen inputs on Lustre; the common frame reads production's astrom_frame). Run:
  NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=... TMPDIR=/fast/... python3 tests/test_tenmin_master_identity.py [test names]
"""
import collections
import csv
import hashlib
import os
import shutil
import sys
import tempfile
import time
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
FROZEN = '/lustre/gh/main/phase3/refactor_analysis/refactor_wt_20261102/F_grouping/frozen_inputs/tenmin'
SCRATCH = os.path.join(tempfile.gettempdir(), 'exopipe3_tests_tenmin_master_identity')
LSTS = ('04h', '05h')
ST, OTHER = 'I', 'V'        # Stokes I: sources repeat across the two LSTs (Stokes V of 04h + 05h has none)
WB = ('Red', 'Green', 'Blue')
csv.field_size_limit(1 << 30)


class Skip(Exception):
    pass


def _rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def _wcsv(p, cols, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cols))
        w.writeheader()
        w.writerows(rows)


class Patch:
    """setattr for a with-block, restored after."""

    def __init__(self, *triples):
        self.triples, self.old = triples, []

    def __enter__(self):
        for obj, name, val in self.triples:
            self.old.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)
        return self

    def __exit__(self, *a):
        for obj, name, val in reversed(self.old):
            setattr(obj, name, val)


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _SB():
    from exopipe.phase3.lib.serve import sublst_build
    return sublst_build


_WORLD = {}


def world():
    """The pristine build B0 (master_catalog.csv + pertie_detections.csv of the frozen 04h + 05h Stokes I detections; served
    catalogs as production's per-LST builds serve them) and its design-B grouping G0 (batch_grouping, for real). Built once.
    Returns dict(B0, G0, gid, served, rows_of_b, split_b, n_served)."""
    if _WORLD:
        return _WORLD
    if not os.path.isdir(FROZEN):
        raise Skip(f'{FROZEN} is not mounted (cluster-only test)')
    SB = _SB()
    B0 = os.path.join(SCRATCH, 'B0')
    shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(B0)
    mc = [r for r in _rows(os.path.join(FROZEN, 'master_catalog.csv')) if r['stokes'] == ST and r['lst'][:3] in LSTS]
    keep = {r['cand_id'] for r in mc}
    pt = [r for r in _rows(os.path.join(FROZEN, 'pertie_detections.csv')) if r['cand_id'] in keep]
    with open(os.path.join(FROZEN, 'master_catalog.csv'), newline='') as f:
        mcols = csv.DictReader(f).fieldnames
    with open(os.path.join(FROZEN, 'pertie_detections.csv'), newline='') as f:
        pcols = csv.DictReader(f).fieldnames
    _wcsv(os.path.join(B0, 'master_catalog.csv'), mcols, mc)
    _wcsv(os.path.join(B0, 'pertie_detections.csv'), pcols, pt)
    G0 = os.path.join(SCRATCH, 'G0')
    t0 = time.time()
    tabs = SB.batch_grouping(B0, G0, 'test_tenmin_master_identity')
    print(f'  world: {len(mc)} Stokes {ST} detections of {"+".join(LSTS)} grouped in {time.time() - t0:.1f} s: {tabs[ST][1]}')
    assert tabs[OTHER][1]['detections'] == 0 and not _rows(tabs[OTHER][0]), f'Stokes {OTHER}: a header-only table'
    gid = {r['cand_id']: r['identity_id'] for r in _rows(tabs[ST][0])}
    assert set(gid) == keep, 'the grouping holds exactly the build\'s detections'
    det = {r['cand_id']: r for r in mc}
    # served catalogs: one served identity per (design-B identity, LST), named per LST; one row per (served identity, band)
    sid_of = {}
    for c in sorted(gid):
        k = (gid[c], det[c]['lst'][:3])
        if k not in sid_of:
            sid_of[k] = f'J{k[1]}{len(sid_of):05d}'
    members = collections.defaultdict(list)      # (tier, band) -> member rows
    surv = collections.defaultdict(dict)         # (tier, band) -> {source_id: row}
    for c in sorted(gid):
        r = det[c]
        sid = sid_of[(gid[c], r['lst'][:3])]
        T = 'wb' if r['band'] in WB else 'sb'
        source_id = f'{sid}_{r["band"]}'
        members[(T, r['band'])].append(dict(cand_id=c, source_id=source_id, band=r['band'], identity_id=sid, snr=r['snr']))
        pe = float(r['pos_stat_maj_arcsec'])
        s = surv[(T, r['band'])].get(source_id)
        if s is None or pe < float(s['pos_err_arcsec']):
            surv[(T, r['band'])][source_id] = dict(source_id=source_id, band=r['band'], identity_id=sid, pos_err_arcsec=f'{pe:.1f}',
                                                   identity_bands='', lst_cells=r['lst'][:3], best_cand_id=c)
    bands_of = collections.defaultdict(set)
    for (T, band), d in surv.items():
        for s in d.values():
            bands_of[s['identity_id']].add(band)
    pages = collections.defaultdict(list)
    for (T, band), d in surv.items():
        for s in d.values():
            s['identity_bands'] = '+'.join(sorted(bands_of[s['identity_id']]))
            pages[T].append(dict(page_id=f'{s["source_id"]}_{s["lst_cells"]}', source_id=s['source_id'], band=band,
                                 identity_id=s['identity_id']))
        _wcsv(os.path.join(B0, f'source_catalogs_{ST}_{T}', f'{band}_sources_survivors.csv'),
              ['source_id', 'band', 'identity_id', 'pos_err_arcsec', 'identity_bands', 'lst_cells', 'best_cand_id'],
              sorted(d.values(), key=lambda x: x['source_id']))
        _wcsv(os.path.join(B0, f'source_catalogs_{ST}_{T}', f'{band}_source_members.csv'),
              ['cand_id', 'source_id', 'band', 'identity_id', 'snr'], members[(T, band)])
    for T, rr in pages.items():
        _wcsv(os.path.join(B0, 'source_catalogs', f'pages_input_{ST}_{T}.csv'), ['page_id', 'source_id', 'band', 'identity_id'],
              sorted(rr, key=lambda x: x['page_id']))
    served_of_b = collections.defaultdict(set)
    for (b, L), sid in sid_of.items():
        served_of_b[b].add(sid)
    split_b = {b for b, s in served_of_b.items() if len(s) > 1}
    if not split_b:
        raise AssertionError('fixture: no design-B identity has detections in both LSTs - the test would prove nothing')
    print(f'  world: {len(sid_of)} served identities, {len(split_b)} design-B identities served under two names')
    _WORLD.update(B0=B0, G0=G0, gid=gid, sid_of=sid_of, served_of_b=served_of_b, split_b=split_b, n_served=len(sid_of))
    return _WORLD


def fresh(name):
    """A copy of the pristine build to work on."""
    W = world()
    B = os.path.join(SCRATCH, name)
    shutil.rmtree(B, ignore_errors=True)
    shutil.rmtree(B + _SB().MASTER_GROUPING, ignore_errors=True)
    shutil.copytree(W['B0'], B)
    return B


def _sha1(p):
    with open(p, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def _tree_digest(B):
    out = {}
    for dp, _dn, fn in os.walk(B):
        for f in fn:
            out[os.path.relpath(os.path.join(dp, f), B)] = _sha1(os.path.join(dp, f))
    return out


def unit(name):
    from exopipe.phase3.core import record
    run = record.Run('serve', [name], os.path.join(SCRATCH, 'out'), _cfg())
    return run.unit(f'sublst_{name}_build', stage='build', tier='sublst')


def present_meta(G):
    from exopipe.phase3.lib.catalog import delta_build as DB
    p = os.path.join(G, f'grouping_{ST}', 'detections.csv')
    _r, _m, sha1, sha256 = DB.read_grouping(p)
    return {ST: dict(path=p, sha1=sha1, sha256=sha256)}


def run_mi(B, name, present=None, production=False):
    """master_identity on B in a unit of its own: design B (present = u.meta['grouping_present']) or production's grouping."""
    C, SB = _cfg(), _SB()
    plan = types.SimpleNamespace(run_name='Run_SLSTB', parity=production)
    pat = [(C.catalog_grouping, 'GROUPING', 'production'), (C.catalog_grouping, 'MULTIPLES_RULE', 'production')] \
        if production else []
    with Patch(*pat):
        with unit(name) as u:
            if present is not None:
                u.meta['grouping_present'] = present
            SB.master_identity(plan, B, u, os.path.join(SCRATCH, f'{name}.log'))
    return u


# ---------------------------------------------------------------------------------------------------------------------

def test_splits_found():
    W, SB = world(), _SB()
    r = SB.served_identity_splits(W['B0'], ST, W['gid'])
    want = sorted(sorted(W['served_of_b'][b]) for b in W['split_b'])
    assert r['splits'] == want, (len(r['splits']), len(want), r['splits'][:3], want[:3])
    assert not r['unlinked'] and not r['twice'] and r['identities'] == W['n_served'], (r['unlinked'][:3], r['twice'][:3])
    assert r['members'] == len(W['gid']), (r['members'], len(W['gid']))


def test_designb_refuses_split():
    W = world()
    B = fresh('B_designb')
    before = _tree_digest(B)
    u = run_mi(B, 'designb', present=present_meta(W['G0']))
    assert u.status == 'refused' and f'G13 master identity, Stokes {ST}' in u.reason and \
        f'{len(W["split_b"])} sets of served identities' in u.reason, (u.status, u.reason)
    assert _tree_digest(B) == before, 'the outcome test rewrote the build'
    # the same build with one served identity per design-B identity passes: rename every served identity to its design-B one
    Bm = fresh('B_designb_merged')
    sid2b = {sid: b for (b, _L), sid in W['sid_of'].items()}
    for dp, _dn, fn in os.walk(Bm):
        for f in fn:
            p = os.path.join(dp, f)
            hdr = open(p).readline().strip().split(',')
            if f.endswith('.csv') and 'identity_id' in hdr:
                rr = _rows(p)
                for r in rr:
                    r['identity_id'] = sid2b[r['identity_id']]
                _wcsv(p, hdr, rr)
    u = run_mi(Bm, 'designb_merged', present=present_meta(W['G0']))
    assert u.status != 'refused', u.reason
    o = u.meta['master_identity']['outcome'][ST]
    assert o['splits'] == 0 and o['identities'] == len(W['served_of_b']) and o['members'] == len(W['gid']), o
    assert any(t['name'] == f'G13_master_identity_{ST}' and t['passed'] for t in u.tests), u.tests


def test_production_merges():
    W, SB = world(), _SB()
    B = fresh('B_prod')
    before = {p: _rows(os.path.join(B, p)) for p in _tree_digest(B) if p.endswith('.csv')}
    u = run_mi(B, 'prod', production=True)
    assert u.status != 'refused', u.reason
    assert any(n.startswith('MASTER-IDENTITY OK') for n in u.notes), u.notes
    mi = u.meta['master_identity']
    assert mi['how'].startswith('production grouping') and mi['grouping'] == B + SB.MASTER_GROUPING, mi
    o = mi['outcome'][ST]
    assert o['splits'] == o['unlinked'] == o['twice'] == 0 and o['members'] == len(W['gid']), o
    outs = {(x['class'], os.path.basename(x['path'])) for x in u.outputs}
    assert ('grouping_table', 'detections.csv') in outs and ('master_identity_record', f'master_identity_{ST}.csv') in outs, outs
    assert all(t['passed'] for t in u.tests), [t for t in u.tests if not t['passed']]
    # the stage's own grouping of B is the fixture's (batch: the same detections, the same partition)
    g2 = {r['cand_id']: r['identity_id'] for r in _rows(os.path.join(B + SB.MASTER_GROUPING, f'grouping_{ST}', 'detections.csv'))}
    part = lambda g: sorted(sorted(c for c in g if g[c] == i) for i in set(g.values()))
    assert part(g2) == part(W['gid']), 'the stage\'s grouping of B partitions its detections as the fixture\'s does'
    # every design-B identity is one served name: the source_id of its row with the smallest pos_err_arcsec
    rows, err = collections.defaultdict(list), {}
    for p, old in before.items():
        if p.endswith('_sources_survivors.csv'):
            for r in old:
                rows[r['identity_id']].append(r)
    new_of = {}
    for p in before:
        if p.endswith('_sources_survivors.csv'):
            for r in _rows(os.path.join(B, p)):
                new_of.setdefault(r['source_id'], r['identity_id'])
    for b, sids in W['served_of_b'].items():
        rr = [r for s in sids for r in rows[s]]
        names = {new_of[r['source_id']] for r in rr}
        assert len(names) == 1, (b, sorted(sids), names)
        if len(sids) > 1:
            want = sorted((float(r['pos_err_arcsec']), r['source_id']) for r in rr)[0][1]
            assert names == {want}, (b, names, want)
            err[b] = want
        else:
            assert names == set(sids), (b, names, sids)      # an identity that joins no other keeps its name
    # row for row, every other column unchanged
    for p, old in before.items():
        new = _rows(os.path.join(B, p))
        assert len(new) == len(old), p
        for a, c in zip(old, new):
            assert {k: v for k, v in a.items() if k not in ('identity_id', 'identity_bands', 'identity_tiers')} == \
                {k: v for k, v in c.items() if k not in ('identity_id', 'identity_bands', 'identity_tiers')}, (p, a, c)
    rec = _rows(os.path.join(B, 'source_catalogs', f'master_identity_{ST}.csv'))
    assert {r['identity_id'] for r in rec} == {s for b in W['split_b'] for s in W['served_of_b'][b]}, len(rec)
    assert {r['master_identity'] for r in rec} == set(err.values()), 'the record names every master'
    # the outcome test, independently, on the rewritten build
    r = SB.served_identity_splits(B, ST, W['gid'])
    assert not r['splits'] and r['identities'] == len(W['served_of_b']), (len(r['splits']), r['identities'])
    # the record's content test fails a build where a superseded identity is left
    from exopipe.phase3.core import validate
    recp = os.path.join(B, 'source_catalogs', f'master_identity_{ST}.csv')
    assert all(t['passed'] for t in validate.run('master_identity_record', recp, stokes=ST))
    old = next(x['identity_id'] for x in rec if x['identity_id'] != x['master_identity'])
    d = os.path.join(B, f'source_catalogs_{ST}_sb')
    p = sorted(f for f in os.listdir(d) if f.endswith('_sources_survivors.csv'))[0]
    rr = _rows(os.path.join(d, p))
    rr.append(dict(rr[0], source_id='Jleft_over', identity_id=old))
    _wcsv(os.path.join(d, p), list(rr[0]), rr)
    t = validate.run('master_identity_record', recp, stokes=ST)
    assert not t[0]['passed'] and f'superseded identity {old} still in the build' in t[0]['detail'], t


def test_refusals():
    W = world()
    pm = present_meta(W['G0'])
    # a served member detection the grouping does not hold
    B = fresh('B_unlinked')
    d = os.path.join(B, f'source_catalogs_{ST}_sb')
    p = sorted(f for f in os.listdir(d) if f.endswith('_source_members.csv'))[0]
    rr = _rows(os.path.join(d, p))
    rr.append(dict(rr[0], cand_id=f'{ST}9_05h_2099-01-01T00_00MHz_9999'))
    _wcsv(os.path.join(d, p), list(rr[0]), rr)
    u = run_mi(B, 'unlinked', present=pm)
    assert u.status == 'refused' and 'served member detections are not in the grouping' in u.reason, u.reason
    # a (source_id, band) under two identities
    B = fresh('B_twice')
    p = sorted(f for f in os.listdir(os.path.join(B, f'source_catalogs_{ST}_sb')) if f.endswith('_sources_survivors.csv'))[0]
    rr = _rows(os.path.join(B, f'source_catalogs_{ST}_sb', p))
    rr.append(dict(rr[0], identity_id='Jsomeone_else'))
    _wcsv(os.path.join(B, f'source_catalogs_{ST}_sb', p), list(rr[0]), rr)
    u = run_mi(B, 'twice', present=pm)
    assert u.status == 'refused' and 'keys under two identities' in u.reason, u.reason
    # no present table for a served Stokes
    u = run_mi(fresh('B_nopresent'), 'nopresent', present={})
    assert u.status == 'refused' and f'Stokes {ST} is served but no present design-B grouping table' in u.reason, u.reason
    # a present table changed since G8 read it
    u = run_mi(fresh('B_changed'), 'changed', present={ST: dict(pm[ST], sha1='0' * 40)})
    assert u.status == 'refused' and 'changed since G8 read it' in u.reason, u.reason


TESTS = [test_splits_found, test_designb_refuses_split, test_production_merges, test_refusals]


def main(argv):
    want = set(argv[1:])
    t0 = time.time()
    failed, skipped = [], []
    try:
        for t in TESTS:
            if want and t.__name__ not in want:
                continue
            t1 = time.time()
            try:
                t()
                print(f'PASS {t.__name__} ({time.time() - t1:.1f} s)')
            except Skip as e:
                skipped.append(t.__name__)
                print(f'SKIP {t.__name__}: {e}')
            except Exception:
                import traceback
                failed.append(t.__name__)
                print(f'FAIL {t.__name__}\n{traceback.format_exc()}')
    finally:
        if not os.environ.get('EXOPIPE3_TEST_KEEP'):
            shutil.rmtree(SCRATCH, ignore_errors=True)
    n = len([t for t in TESTS if not want or t.__name__ in want])
    passed = n - len(failed) - len(skipped)
    print((f'{len(failed)} failed: {failed}; ' if failed else '') + f'{passed} of {n} passed'
          + (f', {len(skipped)} skipped: {skipped}' if skipped else '') + f' ({time.time() - t0:.1f} s)')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
