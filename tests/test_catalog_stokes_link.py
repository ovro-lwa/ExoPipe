#!/usr/bin/env python3
"""The cross-Stokes link (lib/catalog/stokes_link.py; R-GREGG-20261124-CROSS-STOKES-LINK-TEST-APPROVED) and its readers, on
synthetic data (no data files):

  astrometry   core/astrometry.offset_pct_profile = 100 x the profile's mass within u (grouping._profile_mass, numerical);
               p_chance_ellipse = 1 - exp(-rho pi u^2 sqrt(det S)), 0 at u = 0, rising with u
  names        grouping.display_names(reserved=): a name the other Stokes holds is never reused (short -> long form; a long
               form also held -> a letter suffix); without reserved names nothing changes
  link         on synthetic identity moments: a close I/V pair is linked (P >= 0.97, u <= 4), a far one is not; an I identity
               with two V identities near it is ONE component of three, marked ambiguous and named by its best-localized
               member; the counts; every pair records an offset percentile in 0-100 and a chance-coincidence probability in 0-1
  stage        run_link on grouping directories on disk: the content test stokes_link (L1-L4) passes on the grouping's own link
               and on a served copy (publish); a shared display name, a stale table and a corrupted merge table are caught or
               refused; identity_merges.csv follows the 'into' chain to the live ID
  readers      partner_names and the localization wrap (P3_LOC_LINKS: a linked identity localized on both Stokes' detections,
               a detection once; an ambiguous component pools nothing; a missing table refuses); add_link_columns (the
               transient list's link columns; an unknown (Stokes, name) refuses)
  banner       lib/serve/stokes_banner.py: the partner pages of a plan page; the banner of a linked, an ambiguous and an
               unlinked identity and of a run without a link table; inserted after <body> of a source page only

Run: env PYTHONPATH=$PWD NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py
     /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_catalog_stokes_link.py
Writes only under $TMPDIR/exopipe3_tests_stokes_link (removed at the end)."""
import csv
import json
import math
import os
import shutil
import sys
import tempfile
import types

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np  # noqa: E402

from exopipe.phase3.core import astrometry as A  # noqa: E402
from exopipe.phase3.lib.catalog import grouping as G  # noqa: E402
from exopipe.phase3.lib.catalog import grouping_stage as GS  # noqa: E402
from exopipe.phase3.lib.catalog import stokes_link as SL  # noqa: E402
from exopipe.phase3.lib.catalog import tests_grouping as TG  # noqa: E402
from exopipe.phase3.lib.catalog.wraps import source_localization as WL  # noqa: E402
from exopipe.phase3.lib.serve import stokes_banner as SB  # noqa: E402

T = os.path.join(tempfile.gettempdir(), 'exopipe3_tests_stokes_link')
FAILS, N = [], [0]


def check(what, ok, detail=''):
    N[0] += 1
    if ok:
        print('PASS ', what)
    else:
        FAILS.append(f'{what}: {detail}')
        print('FAIL ', what, detail)


def rcsv(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def wcsv(p, cols, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cols), extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def test_astrometry():
    prof = G.Params().prof_ws
    d = max(abs(A.offset_pct_profile(u, prof) - 100.0 * G._profile_mass(prof, u)) for u in (0.2, 0.7, 1.5, 2.5, 4.0))
    check('astrometry: offset_pct_profile = 100 x the profile mass within u (numerical integral), to 1e-4 percent', d < 1e-4, d)
    rho, det = 2e-8, 400.0 ** 2
    v = [A.p_chance_ellipse(rho, u, det) for u in (0.0, 1.0, 2.0)]
    check('astrometry: p_chance_ellipse = 1 - exp(-rho pi u^2 sqrt(det S)), 0 at u = 0, rising',
          v[0] == 0.0 and abs(v[1] - (1 - math.exp(-rho * math.pi * 400.0))) < 1e-12 and v[2] > v[1], v)


def test_names():
    ra, dec = np.array([80.0, 120.0]), np.array([30.0, 40.0])
    n0, c0 = G.display_names(ra, dec)
    n1, c1 = G.display_names(ra, dec, reserved={n0[0]})
    longf = G._jname_long(80.0, 30.0)
    n2, c2 = G.display_names(ra, dec, reserved={n0[0], longf})
    check('names: no reserved names - unchanged; a reserved short name -> the long form; a reserved long form too -> a '
          'letter suffix; the other position untouched',
          n1[0] == longf and n2[0] == longf + 'a' and n1[1] == n0[1] == n2[1] and n0[0] != longf and c0 == 0,
          (n0, n1, n2))


def moments(ids, ra, dec, err, lst='05h', band='46MHz'):
    n = len(ids)
    cov = np.zeros((n, 2, 2))
    cov[:, 0, 0] = cov[:, 1, 1] = np.asarray(err, float) ** 2
    return dict(ids=np.asarray(ids), ra=np.asarray(ra, float), dec=np.asarray(dec, float), cov=cov,
                lst=np.full(n, lst), band=np.full(n, band), n_det=np.ones(n, int), snr=np.full(n, 8.0))


def field():
    """I: GTI-1 at (80, 30), GTI-2 at (82, 32), GTI-3 at (84, 34); V: GTV-1 5 arcsec from GTI-1, GTV-2 and GTV-3 4 arcsec either
    side of GTI-2 (GTV-3 the best-localized), GTV-4 one degree from GTI-3"""
    a = 4.0 / 3600.0
    mI = moments(['GTI-0000001', 'GTI-0000002', 'GTI-0000003'], [80.0, 82.0, 84.0], [30.0, 32.0, 34.0], [20.0, 20.0, 20.0])
    mV = moments(['GTV-0000001', 'GTV-0000002', 'GTV-0000003', 'GTV-0000004'],
                 [80.0 + 5.0 / 3600.0 / math.cos(math.radians(30)), 82.0, 82.0, 84.0], [30.0, 32.0 + a, 32.0 - a, 35.0],
                 [20.0, 25.0, 10.0, 20.0])
    nI = {'GTI-0000001': 'J0520+3000', 'GTI-0000002': 'J0528+3200', 'GTI-0000003': 'J0536+3400'}
    nV = {'GTV-0000001': 'J0520+3000a', 'GTV-0000002': 'J0528+3200a', 'GTV-0000003': 'J0528+3200b', 'GTV-0000004': 'J0536+3500'}
    return mI, mV, nI, nV


def test_link():
    mI, mV, nI, nV = field()
    rows, pairs, c = SL.link(mI, mV, nI, nV, '10min')
    got = sorted((p['I_identity_id'], p['V_identity_id']) for p in pairs)
    check('link: the close pair and both V identities near GTI-2 are linked, the far one is not',
          got == [('GTI-0000001', 'GTV-0000001'), ('GTI-0000002', 'GTV-0000002'), ('GTI-0000002', 'GTV-0000003')], got)
    check('link: every linked pair passes P >= the identity level within the cap, with an offset percentile in 0-100 and a '
          'chance-coincidence probability in 0-1',
          all(p['p'] >= c['p_threshold'] and p['u'] <= c['u_cap'] and 0 <= p['offset_pct'] <= 100 and 0 <= p['p_chance'] <= 1
              for p in pairs), pairs)
    by = {(r['stokes'], r['identity_id']): r for r in rows}
    a, b = by[('I', 'GTI-0000001')], by[('I', 'GTI-0000002')]
    check('link: a one-to-one link is a component of two, not ambiguous; the one-to-many link is ONE component of three, '
          'marked ambiguous and named by its best-localized member (GTV-0000003)',
          a['link_size'] == 2 and a['link_ambiguous'] == 0 and b['link_size'] == 3 and b['link_ambiguous'] == 1
          and b['link_id'] == 'GTV-0000003' and b['link_name'] == 'J0528+3200b'
          and by[('V', 'GTV-0000002')]['link_id'] == 'GTV-0000003', rows)
    check('link: a one-to-one component is named by its best-localized member, Stokes I first on a tie (both 20 arcsec)',
          a['link_id'] == 'GTI-0000001' and by[('V', 'GTV-0000001')]['link_name'] == 'J0520+3000', a)
    check('link: the counts', c['linked_pairs'] == 3 and c['components'] == 2 and c['ambiguous_components'] == 1
          and c['component_sizes'] == {'2': 1, '3': 1} and c['linked_identities_I'] == 2 and c['linked_identities_V'] == 3, c)
    return mI, mV, nI, nV


def write_grouping(out, S, m, names, merged=()):
    """grouping_<S>/ with one detection per identity (detections.csv, identities.csv, state_next) and its det"""
    d = GS.group_dir(out, S)
    ids = [str(x) for x in m['ids']]
    rows = [dict(cand_id=f'{S}_05h_{k}', identity_id=i, identity_name=names[i]) for k, i in enumerate(ids)]
    wcsv(os.path.join(d, GS.DETECTIONS), ('cand_id', 'identity_id', 'identity_name'), rows)
    wcsv(os.path.join(d, GS.IDENTITIES), ('identity_id', 'identity_name'), [dict(identity_id=i, identity_name=names[i])
                                                                             for i in ids])
    st = {i: dict(level='identity', status='live', into='') for i in ids}
    for a, b in merged:
        st[a] = dict(level='identity', status='merged', into=b, last_run='R2')
    os.makedirs(os.path.join(d, 'state_next'), exist_ok=True)
    with open(GS.state_path(os.path.join(d, 'state_next'), '10min', S), 'w') as f:
        json.dump(dict(ids=st), f)
    return dict(cand_id=[r['cand_id'] for r in rows], ra=m['ra'], dec=m['dec'], C=m['cov'], snr=m['snr'], lst=m['lst'],
                band=m['band'])


def test_stage(mI, mV, nI, nV):
    out = os.path.join(T, 'grouping')
    det = {'I': write_grouping(out, 'I', mI, nI, merged=[('GTI-0000009', 'GTI-0000008'), ('GTI-0000008', 'GTI-0000002')]),
           'V': write_grouping(out, 'V', mV, nV)}
    paths, c = SL.run_link(det, out, '10min')
    bad = TG.link_failures(paths['links'])
    check('stage: run_link writes the link; the content test stokes_link (L1-L4) passes on it', not bad and c['linked_pairs'] == 3,
          (bad, c))
    mg = {r['identity_id']: r['into_id'] for r in rcsv(os.path.join(SL.link_dir(out), SL.MERGES))}
    check("stage: identity_merges.csv follows the 'into' chain to the live ID (GTI-0000009 -> GTI-0000008 -> GTI-0000002)",
          mg == {'GTI-0000009': 'GTI-0000002', 'GTI-0000008': 'GTI-0000002'} and c['merged_ids'] == {'I': 2, 'V': 0}, (mg, c))
    dest = os.path.join(T, 'served_run')
    lp = SL.publish(out, dest)
    check('stage: a served copy (publish) passes the same content test (its tables are the grouping it names)',
          lp == os.path.join(dest, 'stokes_link', 'stokes_links.csv') and not TG.link_failures(lp), TG.link_failures(lp))
    # a corrupted merge table
    mp = os.path.join(dest, 'stokes_link', SL.MERGES)
    keep = open(mp).read()
    with open(mp, 'a') as f:
        f.write('I,GTI-0000007,GTI-0000099,R3\n')
    bad = TG.link_failures(lp)
    check('stage: L4 catches a merged ID standing for no live identity', any(b.startswith('L4') for b in bad), bad)
    open(mp, 'w').write(keep)
    # a stale table
    p = os.path.join(GS.group_dir(out, 'V'), GS.DETECTIONS)
    keep = open(p).read()
    open(p, 'a').write('')
    with open(p, 'a') as f:
        f.write('V_05h_9,GTV-0000009,J0600+0000\n')
    try:
        SL.current_links(out)
        got = 'current'
    except SL.LinkRefused as e:
        got = str(e)
    check('stage: a link computed from another grouping table REFUSES (current_links)', 'another V grouping table' in got, got)
    check('stage: L1 catches the changed table', any(b.startswith('L1') for b in TG.link_failures(paths['links'])),
          TG.link_failures(paths['links']))
    open(p, 'w').write(keep)
    # a shared display name
    out2 = os.path.join(T, 'grouping_shared')
    nV2 = dict(nV, **{'GTV-0000004': 'J0536+3400'})
    det2 = {'I': write_grouping(out2, 'I', mI, nI), 'V': write_grouping(out2, 'V', mV, nV2)}
    try:
        SL.run_link(det2, out2, '10min')
        got = 'linked'
    except SL.LinkRefused as e:
        got = str(e)
    check('stage: a display name held in both Stokes REFUSES the link', 'held in both Stokes' in got, got)
    # one Stokes grouped: nothing to link (grouping_stage.link_unit), no link written, the unit not refused
    out3 = os.path.join(T, 'grouping_one')
    write_grouping(out3, 'I', mI, nI)

    class U:
        meta, notes = {}, []

        def note(self, s):
            self.notes.append(s)

        def refuse(self, s):
            raise AssertionError(s)

        def add_output(self, *a, **k):
            raise AssertionError('an output with one Stokes grouped')
    u = U()
    r = GS.link_unit(u, out3, '10min')
    check('stage: with one Stokes grouped the grouping unit links nothing, writes no link and says so',
          r == (None, None) and u.meta['stokes_link'] is None and not os.path.exists(SL.link_dir(out3))
          and 'nothing to link' in ' '.join(u.notes) and SL.grouped_stokes(out3) == {'I'}, (r, u.notes))
    return out, dest


def test_readers(out, dest):
    lp = os.path.join(dest, 'stokes_link', 'stokes_links.csv')
    pn = SL.partner_names(lp)
    check('readers: partner_names - the one-to-one link pools, the ambiguous component pools nothing',
          pn.get('J0520+3000') == ['J0520+3000a'] and pn.get('J0520+3000a') == ['J0520+3000']
          and 'J0528+3200' not in pn and 'J0528+3200a' not in pn, pn)
    U = os.path.join(T, 'U')
    data = {'J0520+3000': ([{'name': 'J0520+3000', 'stokes': 'I'}], [{'cand_id': 'I1'}, {'cand_id': 'X'}]),
            'J0520+3000a': ([{'name': 'J0520+3000a', 'stokes': 'V'}], [{'cand_id': 'V1'}, {'cand_id': 'X'}]),
            'J0528+3200': ([{'name': 'J0528+3200'}], [{'cand_id': 'I2'}]),
            'J0528+3200a': ([{'name': 'J0528+3200a'}], [{'cand_id': 'V2'}])}
    mod = types.SimpleNamespace(detections=lambda ident, U_: data.get(ident, ([], [])))
    f = WL.link_detections(mod)
    old = os.environ.pop(SL.LOC_ENV, None)
    try:
        plain = f('J0520+3000', U)
        os.environ[SL.LOC_ENV] = lp
        s1, d1 = f('J0520+3000', U)
        s2, d2 = f('J0528+3200', U)
        os.environ[SL.LOC_ENV] = os.path.join(T, 'missing.csv')
        try:
            f('J0520+3000', U)
            got = 'ran'
        except RuntimeError as e:
            got = str(e)
    finally:
        os.environ.pop(SL.LOC_ENV, None)
        if old is not None:
            os.environ[SL.LOC_ENV] = old
    check('readers: the localization wrap - unset: unchanged; a linked identity takes its partner\'s rows and detections (a '
          'shared detection once); an ambiguous one keeps its own; a missing table refuses',
          plain == data['J0520+3000'] and [r['name'] for r in s1] == ['J0520+3000', 'J0520+3000a']
          and [d['cand_id'] for d in d1] == ['I1', 'X', 'V1'] and (s2, d2) == data['J0528+3200'] and 'REFUSING' in got,
          (plain, s1, d1, s2, d2, got))
    tl = os.path.join(T, 'TRANSIENT_LIST.csv')
    wcsv(tl, ('source_id', 'stokes', 'identity_id'), [dict(source_id='J0520+3000', stokes='I', identity_id='J0520+3000'),
                                                     dict(source_id='J0528+3200a', stokes='V', identity_id='J0528+3200a'),
                                                     dict(source_id='J0536+3500', stokes='V', identity_id='J0536+3500')])
    body = open(tl).read()
    open(tl, 'w').write('# DEEP TRANSIENT LIST - test\n# Rows per tier (one row per tier, Stokes and source): I_wb 1\n' + body)
    n, nl = SL.add_link_columns(tl, out, lp)
    lines = open(tl).read().splitlines()
    check('readers: add_link_columns keeps the list\'s comment header and names the link columns after it',
          lines[0] == '# DEEP TRANSIENT LIST - test' and lines[1].startswith('# Rows per tier') and
          lines[2].startswith('# Cross-Stokes link') and lines[3].startswith('source_id,stokes,identity_id,stokes_link_id'),
          lines[:4])
    open(tl, 'w').write('\n'.join(ln for ln in lines if not ln.startswith('#')) + '\n')
    r = rcsv(tl)
    check('readers: add_link_columns - the link id, name, ambiguous flag and the other Stokes\' names; an unlinked identity '
          'is its own link',
          (n, nl) == (3, 2) and (r[0]['stokes_link_id'], r[0]['stokes_link_name'], r[0]['stokes_linked_names']) ==
          ('GTI-0000001', 'J0520+3000', 'J0520+3000a') and r[1]['stokes_link_ambiguous'] == '1'
          and r[1]['stokes_linked_names'] == 'J0528+3200' and (r[2]['stokes_link_id'], r[2]['stokes_linked_names']) ==
          ('GTV-0000004', ''), r)
    wcsv(tl, ('source_id', 'stokes', 'identity_id'), [dict(source_id='J0000+0000', stokes='I', identity_id='J0000+0000')])
    try:
        SL.add_link_columns(tl, out, lp)
        got = 'added'
    except SL.LinkRefused as e:
        got = str(e)
    check('readers: add_link_columns refuses a row whose (Stokes, name) is no identity of the grouping', 'REFUSING' in got, got)


class _P:
    def __init__(self, tier, family, stokes, kind, port):
        self.tier, self.family, self.stokes, self.kind, self.port = tier, family, stokes, kind, port


def test_banner(dest):
    pages = [_P('sublst', 'review', 'V', 'single', 8787), _P('sublst', 'review', 'I', 'single', 8788),
             _P('sublst', 'review', 'V', 'multi', 8789), _P('sublst', 'review', 'I', 'multi', 8790),
             _P('snapshot', 'review', 'V', '10s', 8779)]
    pr = SB.partners_of(pages[1], pages)
    check('banner: the partner pages of the 10-minute I page are the V pages, singles first; a snapshot page has none',
          pr == [dict(port=8787, stokes='V', kind='single'), dict(port=8789, stokes='V', kind='multi')]
          and SB.partners_of(pages[4], pages) == [], pr)
    page = types.SimpleNamespace(entry={'argv': ['--run', dest], SB.ENTRY_KEY: pr})
    h1 = SB.html_for(page, 'J0520+3000', ['GTI-0000001'])
    h2 = SB.html_for(page, 'J0528+3200', ['GTI-0000002'])
    h3 = SB.html_for(page, 'J0536+3400', ['GTI-0000003'])
    check('banner: a linked identity names the other Stokes\' identity, links both V pages and gives P, the offset, its '
          'percentile and the chance-coincidence probability',
          'Stokes V J0520+3000a' in h1 and 'http://localhost:8787/source/J0520%2B3000a' in h1
          and 'http://localhost:8789/source/J0520%2B3000a' in h1 and 'P = ' in h1 and 'offset percentile' in h1
          and 'chance-coincidence probability' in h1 and 'AMBIGUOUS' not in h1, h1)
    check('banner: an identity of a one-to-many component is marked AMBIGUOUS with every member listed and both pairs',
          'AMBIGUOUS' in h2 and 'J0528+3200a' in h2 and 'J0528+3200b' in h2 and h2.count('P = ') == 2, h2)
    check('banner: an unlinked identity says so', 'Not linked to a Stokes V identity' in h3, h3)
    page2 = types.SimpleNamespace(entry={'argv': ['--run', os.path.join(T, 'nowhere')], SB.ENTRY_KEY: pr})
    check('banner: a run without a link table says so (no silent omission)', 'no link table' in
          SB.html_for(page2, 'J0520+3000', ['GTI-0000001']))
    txt = '<html><head></head><body class="x"><div>page</div></body></html>'
    a = SB.insert(page, '/source/J0520%2B3000', txt, {'J0520+3000': 'GTI-0000001'})
    b = SB.insert(page, '/', txt, {'J0520+3000': 'GTI-0000001'})
    c = SB.insert(page, '/source/J0900%2B0900', txt, {'J0520+3000': 'GTI-0000001'})
    d = SB.insert(types.SimpleNamespace(entry={'argv': ['--run', dest]}), '/source/J0520%2B3000', txt,
                  {'J0520+3000': 'GTI-0000001'})
    check('banner: inserted right after <body> of a source page only; not on the index, a page without a stable ID, or a '
          'page whose entry names no partner pages',
          a.startswith('<html><head></head><body class="x"><div class="stokes-link"') and b == txt and c == txt and d == txt,
          (a[:120], b == txt, c == txt, d == txt))


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_astrometry()
        test_names()
        mI, mV, nI, nV = test_link()
        out, dest = test_stage(mI, mV, nI, nV)
        test_readers(out, dest)
        test_banner(dest)
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


def test_all():
    assert main() == 0


if __name__ == '__main__':
    sys.exit(main())
