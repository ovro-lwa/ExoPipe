#!/usr/bin/env python3
"""The cross-Stokes link (lib/catalog/stokes_link.py; R-GREGG-20261124-CROSS-STOKES-LINK-TEST-APPROVED) and its readers, on
synthetic data (no data files):

  astrometry   core/astrometry.offset_pct_profile = 100 x the profile's mass within u (grouping._profile_mass, numerical);
               p_chance_ellipse = 1 - exp(-rho pi u^2 sqrt(det S)), 0 at u = 0, rising with u; identity_means and
               identity_association (the link's positions and test) = design B's arithmetic (grouping.offset_en, _inv2, _quad,
               _areal) bit for bit
  names        grouping.display_names(reserved=): a name the other Stokes holds is never reused (short -> long form; a long
               form also held -> a letter suffix); without reserved names nothing changes
  link         on synthetic identity moments: a close I/V pair is linked (P >= 0.97, u <= 4), a far one is not; an I identity
               with two V identities near it is ONE component of three, marked ambiguous and named by its best-localized
               member; the counts; every pair records an offset percentile in 0-100 and a chance-coincidence probability in 0-1;
               the test's P and cap are config catalog_grouping P_IDENTITY and U_CAP, and a Params override changes both the
               decision and the recorded values
  stage        run_link on grouping directories on disk: the content test stokes_link (L1-L4) passes on the grouping's own link
               and on a served copy (publish); a shared display name, a stale table and a corrupted merge table are caught or
               refused; identity_merges.csv follows the 'into' chain to the live ID, and records a chain ending at a RETIRED
               identity as 'retired' (valid history: L4 passes, read_merges leaves it out); a chain that cycles or ends at an
               ID the state never issued REFUSES; L4 catches a merge row the state does not hold and a wrong end status
  versions     stokes_link is a symbolic link switched atomically to a new version per run; the current version and
               KEEP_SUPERSEDED before it kept, older ones deleted; a stokes_link left as a plain directory (the earlier layout)
               is adopted as a version; publish switches the served copy the same way (the old version kept); a version
               directory outside the parent refuses
  readers      partner_names and the localization wrap (P3_LOC_LINKS: a linked identity localized on both Stokes' detections,
               a detection once; an ambiguous component pools nothing; a missing table refuses); add_link_columns (the
               transient list's link columns; an unknown (Stokes, name) refuses)
  banner       lib/serve/stokes_banner.py: the partner pages of a plan page; the banner of a linked, an ambiguous and an
               unlinked identity and of a run without a link table; the P and cap it prints are the link's (its summary);
               inserted after <body> of a source page only
  one-Stokes   tier_sublst._link_env: a view standing on one Stokes needs no link - {} with a note in the record and the log

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
    # identity_means / identity_association against design B's own functions, on random detections
    rng = np.random.default_rng(7)
    n = 400
    ra, dec = rng.uniform(0, 360, n), rng.uniform(-20, 85, n)
    lab = rng.integers(0, 60, n)
    ra = (ra[lab] * 0 + rng.uniform(0, 360, 60)[lab] + rng.normal(0, 0.01, n)) % 360
    dec = rng.uniform(-20, 85, 60)[lab] + rng.normal(0, 0.01, n)
    L = rng.normal(0, 30, (n, 2, 2))
    C = np.einsum('nij,nkj->nik', L, L) + 25.0 * np.eye(2)[None]
    snr = rng.uniform(5, 50, n) * rng.choice([-1, 1], n)
    m = A.identity_means(ra, dec, C, snr, lab)
    ids, inv = np.unique(lab, return_inverse=True)
    order = np.lexsort((-np.abs(snr), inv))
    first = np.ones(n, bool); first[1:] = inv[order][1:] != inv[order][:-1]
    anc = np.empty(len(ids), int); anc[inv[order][first]] = order[first]
    dx, dy = G.offset_en(ra[anc[inv]], dec[anc[inv]], ra, dec)
    Ci = G._inv2(C)
    AA = np.zeros((len(ids), 2, 2)); b = np.zeros((len(ids), 2))
    np.add.at(AA, inv, Ci); np.add.at(b, inv, np.einsum('nij,nj->ni', Ci, np.c_[dx, dy]))
    cov = G._inv2(AA); mm = np.einsum('nij,nj->ni', cov, b)
    rde = dec[anc] + mm[:, 1] / 3600.0
    rra = (ra[anc] + mm[:, 0] / 3600.0 / np.cos(np.radians(dec[anc]))) % 360
    check('astrometry: identity_means = design B\'s inverse-covariance mean (grouping.offset_en, _inv2), bit for bit',
          np.array_equal(m['ra'], rra) and np.array_equal(m['dec'], rde) and np.array_equal(m['cov'], cov)
          and np.array_equal(m['anchor'], anc) and np.array_equal(m['n_det'], np.bincount(inv)))
    P = G.Params()
    k = len(ids)
    j = rng.permutation(k)
    rho = rng.uniform(1e-9, 1e-7, k)
    a = A.identity_association(m['ra'], m['dec'], m['cov'], m['ra'][j] + 0.003, m['dec'][j] - 0.002, m['cov'][j], rho,
                               P.prof_ws, P.floor, 0.97, 4.0)
    ex, ey = G.offset_en(m['ra'], m['dec'], m['ra'][j] + 0.003, m['dec'][j] - 0.002)
    q, detS = G._quad(np.c_[ex, ey], m['cov'] + m['cov'][j] + (P.floor ** 2) * np.eye(2)[None])
    odds = G._areal(q, detS, P.prof_ws) / rho
    pr = odds / (1.0 + odds)
    check('astrometry: identity_association = design B\'s identity-level test (grouping.offset_en, _quad, _areal), bit for bit',
          np.array_equal(a['u'], np.sqrt(q)) and np.array_equal(a['p'], pr) and np.array_equal(a['detS'], detS)
          and np.array_equal(a['offset_arcsec'], np.hypot(ex, ey))
          and np.array_equal(a['associated'], (np.sqrt(q) <= 4.0) & (pr >= 0.97)))


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
    from exopipe.phase3.config import CFG
    g = CFG.catalog_grouping
    check('link: the test\'s P and cap are config catalog_grouping P_IDENTITY and U_CAP (recorded in the counts)',
          (c['p_threshold'], c['u_cap']) == (float(g.P_IDENTITY), float(g.U_CAP)), (c['p_threshold'], c['u_cap']))
    P = G.Params()
    P.u_cap = 0.05
    _r, pairs2, c2 = SL.link(mI, mV, nI, nV, '10min', P)
    P = G.Params()
    P.p_thr = 1.0 - 1e-15
    _r, pairs3, c3 = SL.link(mI, mV, nI, nV, '10min', P)
    check('link: a Params override of the cap or of P changes the decision and the recorded values (one source: the Params)',
          (len(pairs2), c2['u_cap']) == (0, 0.05) and len(pairs3) < 3 and c3['p_threshold'] == 1.0 - 1e-15,
          (len(pairs2), c2['u_cap'], len(pairs3), c3['p_threshold']))
    return mI, mV, nI, nV


def write_grouping(out, S, m, names, merged=(), retired=()):
    """grouping_<S>/ with one detection per identity (detections.csv, identities.csv, state_next) and its det; merged:
    (ID, into) pairs in the state; retired: IDs retired in the state (in no table)"""
    d = GS.group_dir(out, S)
    ids = [str(x) for x in m['ids']]
    rows = [dict(cand_id=f'{S}_05h_{k}', identity_id=i, identity_name=names[i]) for k, i in enumerate(ids)]
    wcsv(os.path.join(d, GS.DETECTIONS), ('cand_id', 'identity_id', 'identity_name'), rows)
    wcsv(os.path.join(d, GS.IDENTITIES), ('identity_id', 'identity_name'), [dict(identity_id=i, identity_name=names[i])
                                                                             for i in ids])
    st = {i: dict(level='identity', status='live', into='') for i in ids}
    for a in retired:
        st[a] = dict(level='identity', status='retired', into='', last_run='R2')
    for a, b in merged:
        st[a] = dict(level='identity', status='merged', into=b, last_run='R2')
    os.makedirs(os.path.join(d, 'state_next'), exist_ok=True)
    with open(GS.state_path(os.path.join(d, 'state_next'), '10min', S), 'w') as f:
        json.dump(dict(ids=st), f)
    return dict(cand_id=[r['cand_id'] for r in rows], ra=m['ra'], dec=m['dec'], C=m['cov'], snr=m['snr'], lst=m['lst'],
                band=m['band'])


def test_stage(mI, mV, nI, nV):
    out = os.path.join(T, 'grouping')
    merged = [('GTI-0000009', 'GTI-0000008'), ('GTI-0000008', 'GTI-0000002'), ('GTI-0000007', 'GTI-0000006')]
    det = {'I': write_grouping(out, 'I', mI, nI, merged=merged, retired=['GTI-0000006']),
           'V': write_grouping(out, 'V', mV, nV)}
    paths, c = SL.run_link(det, out, '10min')
    bad = TG.link_failures(paths['links'])
    check('stage: run_link writes the link; the content test stokes_link (L1-L4) passes on it, a chain ending at a retired '
          'identity included', not bad and c['linked_pairs'] == 3, (bad, c))
    mg = {r['identity_id']: (r['into_id'], r['into_status']) for r in rcsv(os.path.join(SL.link_dir(out), SL.MERGES))}
    check("stage: identity_merges.csv follows the 'into' chain to the live ID (GTI-0000009 -> GTI-0000008 -> GTI-0000002) and "
          "records a chain ending at a retired identity as retired (GTI-0000007 -> GTI-0000006)",
          mg == {'GTI-0000009': ('GTI-0000002', 'live'), 'GTI-0000008': ('GTI-0000002', 'live'),
                 'GTI-0000007': ('GTI-0000006', 'retired')}
          and c['merged_ids'] == {'I': 3, 'V': 0} and c['merged_ids_retired'] == {'I': 1, 'V': 0}, (mg, c))
    check('stage: read_merges (the label carrying\'s reader) maps a merged ID to its live survivor only; the one whose '
          'survivor retired is left out (a label for it gets the 409)',
          SL.read_merges(out) == {'GTI-0000009': 'GTI-0000002', 'GTI-0000008': 'GTI-0000002'}, SL.read_merges(out))
    for what, mrg, ret, want in (('a chain that cycles', [('GTI-0000005', 'GTI-0000004'), ('GTI-0000004', 'GTI-0000005')], [],
                                  'is broken'),
                                 ('a chain ending at an ID the state never issued', [('GTI-0000005', 'GTI-0000099')], [],
                                  'never issued')):
        ob = os.path.join(T, 'grouping_broken')
        shutil.rmtree(ob, ignore_errors=True)
        db = {'I': write_grouping(ob, 'I', mI, nI, merged=mrg, retired=ret), 'V': write_grouping(ob, 'V', mV, nV)}
        try:
            SL.run_link(db, ob, '10min')
            got = 'linked'
        except SL.LinkRefused as e:
            got = str(e)
        check(f'stage: {what} REFUSES the link (merge history broken)', want in got and 'REFUSING' in got
              and not os.path.lexists(SL.link_dir(ob)), got)
    dest = os.path.join(T, 'served_run')
    lp = SL.publish(out, dest)
    check('stage: a served copy (publish) passes the same content test (its tables are the grouping it names)',
          lp == os.path.join(dest, 'stokes_link', 'stokes_links.csv') and not TG.link_failures(lp), TG.link_failures(lp))
    # a corrupted merge table (the served copy; L4 checks it against the grouping's state)
    mp = os.path.join(dest, 'stokes_link', SL.MERGES)
    keep = open(mp).read()
    with open(mp, 'a') as f:
        f.write('I,GTI-0000003,GTI-0000002,live,R3\n')
    bad = TG.link_failures(lp)
    check('stage: L4 catches a merge row of an ID the state does not hold as merged', any(b.startswith('L4') for b in bad), bad)
    open(mp, 'w').write(keep.replace('GTI-0000006,retired', 'GTI-0000006,live'))
    bad = TG.link_failures(lp)
    check('stage: L4 catches a wrong end status (a retired survivor recorded live)', any(b.startswith('L4') for b in bad), bad)
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


def test_versions(mI, mV, nI, nV):
    out = os.path.join(T, 'grouping_versions')
    det = {'I': write_grouping(out, 'I', mI, nI), 'V': write_grouping(out, 'V', mV, nV)}
    os.makedirs(SL.link_dir(out))                       # the earlier layout: a plain directory
    with open(os.path.join(SL.link_dir(out), 'old.txt'), 'w') as f:
        f.write('the earlier link\n')
    seen = []
    for k in range(4):
        paths, _c = SL.run_link(det, out, '10min')
        seen.append(os.path.realpath(SL.link_dir(out)))
    vers = sorted(os.listdir(os.path.join(out, SL.VERSIONS)))
    cur = os.path.basename(seen[-1])
    check('versions: stokes_link is a relative symbolic link to the newest version; each run made a new one',
          os.path.islink(SL.link_dir(out)) and not os.path.isabs(os.readlink(SL.link_dir(out))) and len(set(seen)) == 4
          and cur == vers[-1] and not TG.link_failures(paths['links']), (os.readlink(SL.link_dir(out)), seen, vers))
    adopted = [v for v in os.listdir(os.path.join(out, SL.VERSIONS)) if v.endswith('_adopted')]
    check(f'versions: the current version and {SL.KEEP_SUPERSEDED} before it are kept, older ones deleted (the adopted '
          f'plain directory of the earlier layout among them, being the oldest)',
          len(vers) == 1 + SL.KEEP_SUPERSEDED and [os.path.basename(x) for x in seen[-3:]] == vers and not adopted, vers)
    o2 = os.path.join(T, 'grouping_adopt')
    d2 = {'I': write_grouping(o2, 'I', mI, nI), 'V': write_grouping(o2, 'V', mV, nV)}
    os.makedirs(SL.link_dir(o2))
    with open(os.path.join(SL.link_dir(o2), 'old.txt'), 'w') as f:
        f.write('the earlier link\n')
    SL.run_link(d2, o2, '10min')
    adopted = [v for v in os.listdir(os.path.join(o2, SL.VERSIONS)) if v.endswith('_adopted')]
    check('versions: a stokes_link left as a plain directory is adopted as a version (its files kept), then switched',
          os.path.islink(SL.link_dir(o2)) and len(adopted) == 1
          and os.path.exists(os.path.join(o2, SL.VERSIONS, adopted[0], 'old.txt')), adopted)
    check('versions: no temporary switch link is left behind',
          not [n for n in os.listdir(out) + os.listdir(o2) if '.switch.' in n])
    dest = os.path.join(T, 'served_versions')
    SL.publish(out, dest)
    v1 = os.path.realpath(SL.link_dir(dest))
    SL.run_link(det, out, '10min')
    lp = SL.publish(out, dest)
    v2 = os.path.realpath(SL.link_dir(dest))
    check('versions: publish switches the served copy to a new version (the previous one kept) and the copy passes L1-L4',
          os.path.islink(SL.link_dir(dest)) and v1 != v2 and os.path.isdir(v1) and not TG.link_failures(lp), (v1, v2))
    try:
        SL.switch_to(dest, os.path.join(T, 'grouping_adopt'))
        got = 'switched'
    except SL.LinkRefused as e:
        got = str(e)
    check('versions: switching to a directory that is not a version of the parent REFUSES', 'not a version directory' in got,
          got)
    # a reader holding the old path while the link switches reads one complete version
    p_old = os.path.join(v2, SL.LINKS)
    SL.run_link(det, out, '10min')
    SL.publish(out, dest)
    check('versions: a reader of the previous version still reads it after a switch (kept: KEEP_SUPERSEDED >= 1)',
          os.path.exists(p_old) and rcsv(p_old) == rcsv(os.path.join(SL.link_dir(dest), SL.LINKS)))


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
    pt, uc = SL.link_test_values(SL.read_summary(dest))
    check('banner: an unlinked identity says so, with the P and the cap the link was computed with (its summary)',
          'Not linked to a Stokes V identity' in h3 and f'P &ge; {pt:g} within {uc:g} standard deviations' in h3, h3)
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


def test_banner_override(mI, mV, nI, nV):
    """the banner prints the values of the link it reads: a link computed with a Params override (P 0.99, cap 3.5)"""
    out = os.path.join(T, 'grouping_override')
    det = {'I': write_grouping(out, 'I', mI, nI), 'V': write_grouping(out, 'V', mV, nV)}
    P = G.Params()
    P.p_thr, P.u_cap = 0.99, 3.5
    SL.run_link(det, out, '10min', P)
    dest = os.path.join(T, 'served_override')
    SL.publish(out, dest)
    page = types.SimpleNamespace(entry={'argv': ['--run', dest], SB.ENTRY_KEY: [dict(port=8787, stokes='V', kind='single')]})
    h = SB.html_for(page, 'J0536+3400', ['GTI-0000003'])
    check('banner: the P and the cap printed are the link\'s own (a link computed with P 0.99 and cap 3.5 prints them)',
          'P &ge; 0.99 within 3.5 standard deviations' in h and SL.link_test_values(SL.read_summary(dest)) == (0.99, 3.5), h)


def test_one_stokes_view():
    from exopipe.phase3.lib.catalog import tier_sublst as TS
    said = []
    ctx = types.SimpleNamespace(grouping_mode=types.SimpleNamespace(design_b=True), grouping_dir=os.path.join(T, 'nowhere'),
                                log=types.SimpleNamespace(say=said.append))
    notes = []
    s = types.SimpleNamespace(u=types.SimpleNamespace(note=notes.append))
    gI = os.path.join(T, 'grouping', 'grouping_I', 'detections.csv')
    r = TS._link_env(ctx, os.path.join(T, 'view_05h'), {'I': (gI, 'x' * 40)}, s)
    check('one-Stokes: a view standing on one Stokes needs no link - {} (no refusal), noted in the record and the log',
          r == {} and len(notes) == 1 and 'nothing to link' in notes[0] and said == notes, (r, notes, said))
    r2 = TS._link_env(ctx, os.path.join(T, 'build_05h'), None, s)
    check('one-Stokes: a build on a grouping of one Stokes (or none) needs no link either - {}, noted',
          r2 == {} and len(notes) == 2 and 'nothing to link' in notes[1], (r2, notes))


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_astrometry()
        test_names()
        mI, mV, nI, nV = test_link()
        out, dest = test_stage(mI, mV, nI, nV)
        test_versions(mI, mV, nI, nV)
        test_readers(out, dest)
        test_banner(dest)
        test_banner_override(mI, mV, nI, nV)
        test_one_stokes_view()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


def test_all():
    assert main() == 0


if __name__ == '__main__':
    sys.exit(main())
