#!/usr/bin/env python3
"""STANDING TEST - the combined snapshot view's pages p8986-p8989 and its false-positive filter step (rows 10 and 11 of
refactor_20261012/plan/CUTOVER_REMAINING_20261124.md; lib/serve/snapshot_view.py serve stages, lib/serve/fp_filters.py).
Data-free: a synthetic site file and synthetic views under a fresh scratch directory ($EXOPIPE3_TEST_SCRATCH, else the system
temp directory), removed at exit. Plain asserts; run standalone or under pytest.

  1. config: serve.PAGES family comb = 8986 V 10-s, 8987 I 10-s, 8988 V 1-min, 8989 I 1-min with snapshot_serve.sh's files, hub
     keys '<S> 10-sec-comb' / '<S> 1-min-comb'; the hides' constants protected and ruled.
  2. the serve plan (plan.make --family comb): the view under the snapshot root (parity: under --out), no --source, the member
     root; argv with the OTHER comb pages as siblings (never the batch pages 8779-8782), in snapshot_serve.sh's order; the
     registry entry's P3_CUTOUTS_TIER / P3_PAGE_ID; the hub --set in snapshot_serve.sh's key order; the exclusion gate refuses
     on comb pages like on the snapshot pages.
  3. stage selection: no selection, --from alone or --until naming a build stage = the build stages (stop at ready); the serve
     stages only when named (--stages, or --until naming one); --from hides alone refused; the batch serve's TESS step between
     the page context and the bright-source filter. The serve order: a build resume plans no page (no port or label gate);
     G26 without --label; the serve hold (G29) refuses before the hides; hold and go checked once, before the hides. The TESS
     step: master_catalog.csv an input (refused when absent), the orbit table fingerprinted after the tool's refresh.
  4. fp_filters on a synthetic four-page view: the listed rows only, the page's own tier (an empty tier counts), the first
     page's row of a detection on two pages, the flood rule's three edges (>= 100x and >= 50: flood; 60x: not; 49 against 0:
     not), the held alignment cell, the CSV / json form (line ends, key order), the records; the refusals (files exist, an
     excluded cell-band, nothing listed, no queue, a listed directory without detections.csv); the install (the json renamed
     before the CSV, a half-installed pair refused and untouched, nothing left after a failure); the content test's negatives.
"""
import atexit
import csv
import io
import json
import os
import shutil
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
_base = os.environ.get('EXOPIPE3_TEST_SCRATCH') or tempfile.gettempdir()
os.makedirs(_base, exist_ok=True)
T = tempfile.mkdtemp(prefix='snapview_pages_', dir=_base)
atexit.register(shutil.rmtree, T, True)
# a synthetic site (data-free: the same on the cluster and in CI)
for d in ('exopipe_ops', 'clv15'):
    os.makedirs(os.path.join(T, 'site', d), exist_ok=True)
SITE = os.path.join(T, 'site', 'site.py')
with open(SITE, 'w') as f:
    f.write(f'PYTHON = {{"default": {sys.executable!r}}}\n'
            f'LEGACY_CODE_ROOTS = [{os.path.join(T, "site", "clv15")!r}, {os.path.join(T, "site", "exopipe_ops")!r}]\n')
os.environ['EXOPIPE3_SITE'] = SITE

from exopipe.phase3.config import CFG                                 # noqa: E402
from exopipe.phase3.core.record import Refusal                        # noqa: E402
from exopipe.phase3.lib.serve import exclusion_gate, fp_filters, plan as planmod, register, snapshot_copy, snapshot_view  # noqa
from exopipe.phase3.lib.serve import tests as serve_tests            # noqa: E402,F401  (registers class fp_filters)
from exopipe.phase3.core import validate                              # noqa: E402

S = CFG.serve
VIEW = 'Run_SNAPCOMB_20990101'
PASSED = []


def ok(name):
    PASSED.append(name)
    print('PASS', name)


def args(**kw):
    a = dict(tier='snapshot', family='comb', run_label=VIEW, label='Run_SNAPCOMB_20990101 COMBINED VIEW (test)', pages=None,
             stokes=None, port_offset=0, forced_tag=None, source=None, restart=None, stop=None, lsts=None, stages=None,
             from_stage=None, until=None, view_omit=None)
    a.update(kw)
    return types.SimpleNamespace(**a)


# ---- 1. config ----------------------------------------------------------------------------------------------------------
def test_config():
    comb = [(k, c) for (t, k), c in S.PAGES.items() if t == 'snapshot' and c['family'] == 'comb']
    want = [(8986, 'V', '10s', 'queue_V_10s.txt', 'source_catalogs_V_10s', 'source_catalogs_loc_10s', ('V', '10-sec-comb')),
            (8987, 'I', '10s', 'queue_I_10s.txt', 'source_catalogs_I_10s', 'source_catalogs_loc_10s', ('I', '10-sec-comb')),
            (8988, 'V', '1min', 'queue_V_1min.txt', 'source_catalogs_V_1min', 'source_catalogs_loc_1min', ('V', '1-min-comb')),
            (8989, 'I', '1min', 'queue_I_1min.txt', 'source_catalogs_I_1min', 'source_catalogs_loc_1min', ('I', '1-min-comb'))]
    got = [(c['port'], c['stokes'], c['kind'], c['queue'], c['catalog'], c['loc_dir'], c['hub']) for _, c in comb]
    assert got == want, got
    assert all(c['loc_offsets'] and c['labels'] == 'judgements' and c['twin'] is None for _, c in comb)
    for k in ('SNAPVIEW_FLOOD_FACTOR', 'SNAPVIEW_FLOOD_MIN', 'SNAPVIEW_ALIGN_FAILED', 'SNAPVIEW_ALIGN_HELD', 'SNAPVIEW_FP_ALIGN',
              'SNAPVIEW_FP_FLOOD'):
        assert k in S.PROTECTED_KEYS, k
    assert S.SNAPVIEW_FLOOD_FACTOR == 100 and S.SNAPVIEW_FLOOD_MIN == 50
    assert S.SNAPVIEW_FP_ALIGN['default_off'] == S.SNAPVIEW_FP_ALIGN['states'][:1]
    assert S.SNAPVIEW_FP_FLOOD['default_off'] == S.SNAPVIEW_FP_FLOOD['states'][:1]
    ok('config: four comb pages, hub keys, protected hides constants')


# ---- 2. the serve plan ----------------------------------------------------------------------------------------------------
def test_plan():
    out = os.path.join(T, 'out')
    p = planmod.make(args(), out)
    assert p.family == 'comb' and p.run_name == VIEW and p.source is None, (p.family, p.run_name, p.source)
    assert p.parity and p.U == os.path.join(out, 'snapshot', VIEW), p.U
    assert p.member_root == planmod.run_root('snapshot'), p.member_root
    assert [pg.port for pg in p.pages] == [8986, 8987, 8988, 8989]
    utc = '2099-01-01 00:00 UT'
    a = p.argv(p.pages[0], utc)
    sib = [a[i + 1] for i, x in enumerate(a) if x == '--sibling']
    U = p.U
    assert sib == [f'8987:{U}/source_catalogs_I_10s:Stokes I 10-s page', f'8988:{U}/source_catalogs_V_1min:Stokes V 1-min page',
                   f'8989:{U}/source_catalogs_I_1min:Stokes I 1-min page'], sib
    flags = [x for x in a[2:] if x.startswith('--')]
    assert flags == ['--run', '--queue', '--catalog-dir', '--port', '--default-fp-reason', '--loc-dir', '--loc-offsets', '--title',
                     '--sibling', '--sibling', '--sibling', '--lists-dir'], flags     # --lists-dir: a parity plan's lists
    assert a[a.index('--queue') + 1] == os.path.join(U, 'queue_V_10s.txt')
    assert a[a.index('--loc-dir') + 1] == os.path.join(U, 'source_catalogs_loc_10s')
    assert args().label in a[a.index('--title') + 1]
    for pg, ct in zip(p.pages, ('snap', 'snap', '1min', '1min')):
        e = register.page_entry(p, pg, utc)
        assert e['id'] == f'p{pg.port}' and e['legacy_port'] == pg.port and '--port' not in e['argv']
        assert e['env'].get('P3_CUTOUTS_TIER') == ct and e['env'].get('P3_PAGE_ID') == f'p{pg.port}', e['env']
        h = p.hub_entry(pg, utc)
        assert h[:2] == pg.c['hub'] and [x.split('=')[0] for x in h[2]] == ['port', 'run', 'generation', 'served', 'live'], h
        assert h[2][0] == f'port={pg.port}' and h[2][1] == f'run={VIEW}' and h[2][4] == 'live=1', h
    # the batch pages keep their own siblings
    pb = planmod.make(args(family='review', run_label='Run_SNAPB1_20990101'), out)
    ab = pb.argv(pb.pages[0], utc)
    assert [x.split(':')[0] for i, x in enumerate(ab) if i and ab[i - 1] == '--sibling'] == ['8780', '8781', '8782']
    # --source: the view is its own review copy
    try:
        planmod.make(args(source='/somewhere/Run_SNAPCOMB_20990101'), out)
        raise AssertionError('--source with --family comb was accepted')
    except Refusal as e:
        assert '--source' in str(e)
    # the exclusion gate keys the comb pages as the snapshot pages (refuses the same way)
    assert exclusion_gate.gate_key('snapshot', 'comb') == 'snapshot' == exclusion_gate.gate_key('snapshot', 'review')
    assert exclusion_gate.refuses('snapshot', 'comb') == exclusion_gate.refuses('snapshot', 'review')
    assert p.first_page().port_cfg == 8986 and pb.first_page().port_cfg == 8779, (p.first_page(), pb.first_page())
    ok('plan: comb pages, siblings, entries with cutouts, hub order, no --source, exclusion gate key, first page of the family')


# ---- 3. stages ----------------------------------------------------------------------------------------------------------
def test_stages():
    sel, B, SV = snapshot_view.select_stages, list(snapshot_view.BUILD_STAGES), list(snapshot_view.SERVE_STAGES)
    assert sel(args()) == B
    # the serve stages only when named (--stages, or --until naming one): --from or --until alone stops at ready
    assert sel(args(from_stage='build')) == B[1:] and sel(args(until='localize')) == B[:4]
    assert sel(args(from_stage='hides', until='verify')) == SV and sel(args(until='hides')) == B + ['hides']
    assert sel(args(stages='hides,start')) == ['hides', 'start']
    try:
        sel(args(from_stage='hides'))
        raise AssertionError('--from hides without --until selected nothing and was not refused')
    except Refusal as e:
        assert '--until' in str(e), e
    st = list(snapshot_copy.STAGES)
    assert st.index('page-context') < st.index('tess') < st.index('bright-filter'), st
    assert [s for s, _ in snapshot_copy.STEP_STAGES] == ['scintillation', 'page-context', 'tess', 'bright-filter', 'page-index']
    assert set(snapshot_view.SERVE_STAGES) - {'hides'} <= set(snapshot_copy.STAGES)
    assert 'tess_coverage' in __import__('exopipe.phase3.lib.serve.scripts', fromlist=['x']).SCRIPTS
    ok('stages: build by default and with --from / --until alone; serve only when named; --from hides alone refused; TESS '
       'after the page context')


def test_serve_order():
    """A build resume plans no page (so no port or label gate refuses it); G29 and G27 come before the hides write anything;
    the batch serve's stages are not checked twice."""
    from exopipe.phase3.core.record import Run
    from exopipe.phase3.lib.serve import flow
    out, calls = os.path.join(T, 'order'), []
    saved = (snapshot_view.gate_runs, snapshot_view.serve_plan, snapshot_view.describe, snapshot_view.hides_stage,
             snapshot_copy.run, snapshot_copy.hold_or_no_go, flow.require_done, S.SNAPSHOT_SERVE_HOLD)
    real_plan, real_hold = saved[1], saved[5]
    try:
        snapshot_view.gate_runs = lambda plan, st: calls.append(('gate_runs', tuple(st)))
        snapshot_view.serve_plan = lambda a, r: calls.append(('serve_plan',)) or real_plan(a, r)
        snapshot_view.describe = lambda plan, st, sp, a: calls.append(('describe', tuple(st), sp is None))
        snapshot_view.run(args(from_stage='filtercat', label=None), types.SimpleNamespace(out_root=out, dry_run=True))
        assert calls == [('gate_runs', ('filtercat', 'localize', 'ready')),
                         ('describe', ('filtercat', 'localize', 'ready'), True)], calls
        calls.clear()
        try:
            snapshot_view.run(args(from_stage='hides', until='verify', label=None), types.SimpleNamespace(out_root=out,
                                                                                                         dry_run=False))
            raise AssertionError('a serve without --label was not refused')
        except Refusal as e:
            assert 'G26' in str(e) and calls == [('serve_plan',)], (e, calls)
        # the serve hold: refused before the hides; nothing later runs
        calls.clear()
        flow.require_done = lambda *a, **k: None
        snapshot_view.hides_stage = lambda r, p, sp: calls.append(('hides',))
        snapshot_copy.run = lambda a, r, sp, rest, held_checked=False: calls.append(('copy', tuple(rest), held_checked))
        hold = os.path.join(T, 'SERVE_HOLD')
        open(hold, 'w').write('test hold\n')
        S.SNAPSHOT_SERVE_HOLD = hold
        run_ = Run('serve', ['test'], os.path.join(out, 'held'), CFG, config_sections=None)
        snapshot_view.run(args(stages='hides,gate-copy,start', port_offset=10000), run_)
        assert calls == [('serve_plan',)] and [(u.name, u.status) for u in run_.units] == [
            (f'snapshot_{VIEW}_hold', 'refused')], (calls, [(u.name, u.status, u.reason) for u in run_.units])
        # no hold: the hold / go check once, then the hides, then the batch stages without a second check
        calls.clear()
        snapshot_copy.hold_or_no_go = lambda r, sp: calls.append(('hold_or_no_go',)) or False
        run_ = Run('serve', ['test'], os.path.join(out, 'free'), CFG, config_sections=None)
        snapshot_view.run(args(stages='hides,gate-copy,start', port_offset=10000), run_)
        assert calls == [('serve_plan',), ('hold_or_no_go',), ('hides',), ('copy', ('gate-copy', 'start'), True)], calls
    finally:
        (snapshot_view.gate_runs, snapshot_view.serve_plan, snapshot_view.describe, snapshot_view.hides_stage, snapshot_copy.run,
         snapshot_copy.hold_or_no_go, flow.require_done, S.SNAPSHOT_SERVE_HOLD) = saved
    assert real_hold is snapshot_copy.hold_or_no_go
    ok('serve order: a build resume plans no page; G26 refuses a serve without --label; the serve hold refuses before the hides; '
       'hold and go checked once, before the hides')


def test_tess_step():
    """The batch serve's TESS step: the view's master_catalog.csv an input (refused when absent, before the tool); the orbit
    table fingerprinted as the tool left it (after its refresh)."""
    import hashlib
    from exopipe.phase3.core.record import Run
    from exopipe.phase3.lib.pages import common as pcommon
    from exopipe.phase3.lib.serve import scripts
    DST = os.path.join(T, 'tess_view')
    os.makedirs(DST)
    table = os.path.join(os.path.dirname(scripts.path('tess_coverage')), 'TESS_orbit_times.csv')
    os.makedirs(os.path.dirname(table), exist_ok=True)
    open(table, 'w').write('Sector,Orbit,Start,End\n1,1,old,old\n')
    fresh = 'Sector,Orbit,Start,End\n1,1,2099-01-01,2099-01-14\n'
    cid = '05h_2099-01-02_23MHz_V_10s_20990102_030000'
    calls = []

    def fake_run(name, a=(), **k):
        calls.append(name)
        open(table, 'w').write(fresh)                 # the tool's refresh
        with open(os.path.join(DST, 'tess_coverage.csv'), 'w') as f:
            f.write(f'# tess_coverage.py (stand-in)\ncand_id,status,sector,camera,ccd\n{cid},not observed,,,\n')
        return types.SimpleNamespace(ok=True, rc=0, reason='', tail='wrote tess_coverage')
    saved = (scripts.run, pcommon.interpreter_or_none, CFG.pages_deep.TESS_REFRESH_IN_READER)
    plan = types.SimpleNamespace(U=DST, run_name='Run_SNAPTESS')
    try:
        scripts.run, pcommon.interpreter_or_none = fake_run, (lambda key: sys.executable)
        CFG.pages_deep.TESS_REFRESH_IN_READER = True
        r0 = Run('serve', ['test'], os.path.join(T, 'tess_out0'), CFG, config_sections=None)
        u = snapshot_copy.tess_stage(r0, plan)
        assert u.status == 'refused' and 'master_catalog.csv' in u.reason and not calls, (u.status, u.reason, calls)
        with open(os.path.join(DST, 'master_catalog.csv'), 'w') as f:
            f.write(f'cand_id,lst,date,band,stokes\n{cid},05h,2099-01-02,23MHz,V\n')
        r1 = Run('serve', ['test'], os.path.join(T, 'tess_out1'), CFG, config_sections=None)
        u = snapshot_copy.tess_stage(r1, plan)
        assert u.status == 'done' and calls == ['tess_coverage'], (u.status, u.reason, calls)
        fp = u.inputs[os.path.abspath(table)]
        assert fp['sha1'] == hashlib.sha1(fresh.encode()).hexdigest(), 'the orbit table was fingerprinted before the refresh'
        assert os.path.abspath(os.path.join(DST, 'master_catalog.csv')) in u.inputs
    finally:
        scripts.run, pcommon.interpreter_or_none, CFG.pages_deep.TESS_REFRESH_IN_READER = saved
    ok('TESS step: master_catalog.csv an input (absent: refused before the tool); the orbit table fingerprinted after the '
       'refresh')


# ---- 4. fp_filters -------------------------------------------------------------------------------------------------------
PAGES = [dict(id='p8986', queue='queue_V_10s.txt', catalog='source_catalogs_V_10s', kind='10s', stokes='V'),
         dict(id='p8987', queue='queue_I_10s.txt', catalog='source_catalogs_I_10s', kind='10s', stokes='I'),
         dict(id='p8988', queue='queue_V_1min.txt', catalog='source_catalogs_V_1min', kind='1min', stokes='V'),
         dict(id='p8989', queue='queue_I_1min.txt', catalog='source_catalogs_I_1min', kind='1min', stokes='I')]
DF = ['cand_id', 'lst', 'date', 'band', 'ra', 'dec', 'snr', 'stokes', 'tier']


def _det(i, lst, night, band, stokes, tier, ra=None):
    return dict(cand_id=f'{lst}_{night}_{band}_{stokes}_{tier}_{i:05d}', lst=lst, date=night + 'T01:00:00', band=band,
                ra=f'{ra if ra is not None else 10 + i * 1e-3:.5f}', dec='20.00000', snr='6.1', stokes=stokes, tier=tier)


def make_view(root):
    """A four-page view. V 10-s sources: 'flood' (120 detections 04h 2099-01-05 55MHz; 1 on each of 3 other nights = base 1),
    'sixty' (60 on 05h 2099-01-05 41MHz against base 1: 60x, not a flood), 'small' (49 on 06h 2099-01-05 69MHz, no other night:
    under the minimum), 'held' (2 on 23h 2025-08-27 82MHz: the held alignment cell), 'mixed' (a 1-min row and an empty-tier row in
    a 10-s page directory), 'unlisted' (200 on 04h 2099-01-06 55MHz, not in the queue). I 10-s: 'twice' (the first 3 of the flood
    rows again, same key: counted once, the V page's row kept). 1-min pages: one plain source each."""
    V = os.path.join(root, VIEW)
    srcs = {p['id']: [] for p in PAGES}
    def add(pid, sid, dets, listed=True):
        rd = f'{pid}_pages/{sid}'
        os.makedirs(os.path.join(V, rd), exist_ok=True)
        with open(os.path.join(V, rd, 'detections.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, DF + ['extra'])
            w.writeheader()
            for d in dets:
                w.writerow(dict(d, extra='x'))
        srcs[pid].append((sid, rd, listed))
    nights = ['2099-01-02', '2099-01-03', '2099-01-04']
    flood = [_det(i, '04h', '2099-01-05', '55MHz', 'V', '10s') for i in range(120)]
    base = [_det(900 + k, '04h', n, '55MHz', 'V', '10s') for k, n in enumerate(nights)]
    add('p8986', 'Jflood', flood + base)
    add('p8986', 'Jsixty', [_det(i, '05h', '2099-01-05', '41MHz', 'V', '10s') for i in range(60)]
        + [_det(900 + k, '05h', n, '41MHz', 'V', '10s') for k, n in enumerate(nights)])
    add('p8986', 'Jsmall', [_det(i, '06h', '2099-01-05', '69MHz', 'V', '10s') for i in range(49)])
    add('p8986', 'Jheld', [_det(i, '23h', '2025-08-27', '82MHz', 'V', '10s') for i in range(2)])
    add('p8986', 'Jmixed', [_det(0, '07h', '2099-01-05', '55MHz', 'V', '1min'), _det(1, '07h', '2099-01-05', '55MHz', 'V', '')])
    add('p8986', 'Junlisted', [_det(i, '04h', '2099-01-06', '55MHz', 'V', '10s') for i in range(200)], listed=False)
    add('p8987', 'Jtwice', [dict(d, stokes='V') for d in flood[:3]])
    add('p8988', 'Jv1', [_det(0, '08h', '2099-01-05', '55MHz', 'V', '1min')])
    add('p8988', 'Jshort', [])
    with open(os.path.join(V, 'p8988_pages', 'Jshort', 'detections.csv'), 'a') as f:
        f.write('short_1,08h\n')       # a short row: its missing fields None in DictReader
    add('p8989', 'Ji1', [_det(0, '09h', '2099-01-05', '55MHz', 'I', '1min')])
    for p in PAGES:
        os.makedirs(os.path.join(V, p['catalog']), exist_ok=True)
        with open(os.path.join(V, p['queue']), 'w') as f:
            f.write('# a header line\n\n' + ''.join(f'{s}\n' for s, _, l in srcs[p['id']] if l))
        with open(os.path.join(V, p['catalog'], f'{p["stokes"]}_{p["kind"]}_sources_survivors.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['source_id', 'identity_id', 'review_dir'])
            for s, rd, _ in srcs[p['id']]:
                w.writerow([s + '_raw', s, rd])
    return V


class NoEx:
    def is_excluded(self, lst, band, date):
        return False


class Ex:
    def is_excluded(self, lst, band, date):
        return (lst, band, date) == ('05h', '41MHz', '2099-01-05')


def test_fp_filters():
    V = make_view(os.path.join(T, 'view'))
    out, rec = os.path.join(T, 'fp_out'), os.path.join(T, 'fp_rec')
    os.makedirs(out)
    res = fp_filters.build(V, out, PAGES, records=rec, ex=NoEx(), log=lambda m: None)
    c = res['counts']
    # distinct: V 10-s 123 + 63 + 49 + 2 + 1 = 238 (the unlisted 200 not read, the 1-min row of a 10-s page dropped; the I page's
    # 3 rows are duplicates); 1-min pages 2
    # + the short row of p8988 (no date, no position: keyed by its raw strings)
    assert c['distinct'] == 241, c
    assert c['flood_hours'] == 1 and c['flood'] == 120 and c['align_held'] == 2 and c['align_failed'] == 0, c
    assert c['rows'] == 122 and res['missing'] == 0, c
    assert res['pages']['p8986']['detections'] == 123 + 63 + 49 + 2 + 1, res['pages']['p8986']   # the 1-min row dropped
    with open(res['csv'], 'rb') as f:
        raw = f.read()
    first, rest = raw.split(b'\n', 1)
    assert first.startswith(b'# frame-alignment states and flood hours at 100x') and b'\r' not in first
    assert rest.startswith(b'cand_id,ra,dec,fp_c_align,fp_c_flood\r\n'), rest[:80]
    rows = list(csv.reader(io.StringIO(rest.decode())))[1:]
    F0, A1 = S.SNAPVIEW_FP_FLOOD['states'][0], S.SNAPVIEW_FP_ALIGN['states'][1]
    assert sum(1 for r in rows if r[4] == F0) == 120 and sum(1 for r in rows if r[3] == A1) == 2
    assert all(r[0].startswith('04h_2099-01-05_55MHz_V_10s') for r in rows if r[4]), 'a non-flood detection marked flood'
    assert rows[0][1] == '10.00000', rows[0]                      # the position as written in detections.csv (raw)
    meta = json.load(open(res['json']))
    assert [list(k) for k in meta['cats']] == [['key', 'stokes', 'states', 'default_off', 'label', 'help']] * 2
    assert meta['metrics'] == [] and '1 flood hours, 120 detections, 1 nights' in meta['cats'][1]['help']
    assert '(2 detections in this view)' in meta['cats'][0]['help']
    assert open(res['json']).read().startswith('{\n "cats": [\n  {\n   "key": "fp_c_align"')
    with open(os.path.join(rec, 'flood_list_K100.csv')) as f:
        fl = list(csv.reader(f))
    assert fl == [fp_filters.FLOOD_HEADER, ['snapshot', 'V', '10s', '04h', '2099-01-05', '55MHz', '120', '1', '120.0']], fl
    with open(os.path.join(rec, 'flood_dets_p8987.csv')) as f:
        d = list(csv.DictReader(f))
    assert len(d) == 3 and d[0]['page'] == 'p8987' and d[0]['default_listed'] == '1' and d[0]['night'] == '2099-01-05'
    with open(os.path.join(rec, 'flood_dets_p8988.csv')) as f:
        d = [r for r in csv.DictReader(f) if r['cand_id'] == 'short_1']
    # production's round trip through the CSV: None written as '', the date as str(None)[:10]
    assert len(d) == 1 and d[0]['lst'] == '08h' and d[0]['night'] == 'None' and d[0]['ra'] == '' and d[0]['band'] == '', d
    for r in validate.run('fp_filters', res['csv'], rows=c['rows']):
        assert r['passed'], r
    ok('fp_filters: listed rows, own tier, first page kept, flood edges 120x / 60x / 49, held cell, file form, records')

    # refusals
    for what, fn in (('exists', lambda: fp_filters.build(V, out, PAGES, ex=NoEx(), log=lambda m: None)),
                     ('excluded', lambda: fp_filters.build(V, os.path.join(T, 'fp_ex'), PAGES, ex=Ex(), log=lambda m: None)),
                     ('no queue', lambda: fp_filters.build(V, os.path.join(T, 'fp_nq'), PAGES[:1] + [dict(PAGES[1], queue='q.txt')],
                                                           ex=NoEx(), log=lambda m: None))):
        os.makedirs(os.path.join(T, 'fp_ex'), exist_ok=True)
        os.makedirs(os.path.join(T, 'fp_nq'), exist_ok=True)
        try:
            fn()
            raise AssertionError(f'{what}: not refused')
        except fp_filters.HidesRefusal as e:
            assert {'exists': 'not overwritten', 'excluded': 'excluded cell-bands', 'no queue': 'no queue'}[what] in str(e), e
    assert not os.listdir(os.path.join(T, 'fp_ex')) and not os.listdir(os.path.join(T, 'fp_nq')), 'a refusal wrote files'
    E = os.path.join(T, 'empty')
    for p in PAGES:
        os.makedirs(os.path.join(E, VIEW, p['catalog']), exist_ok=True)
        open(os.path.join(E, VIEW, p['queue']), 'w').write('# nothing listed\n')
    os.makedirs(os.path.join(T, 'fp_empty'))
    try:
        fp_filters.build(os.path.join(E, VIEW), os.path.join(T, 'fp_empty'), PAGES, ex=NoEx(), log=lambda m: None)
        raise AssertionError('an empty view was not refused')
    except fp_filters.HidesRefusal as e:
        assert 'no listed detection' in str(e)
    assert not os.listdir(os.path.join(T, 'fp_empty'))
    # a listed source whose page directory lost its detections.csv refuses before anything is written; counted when allowed
    os.remove(os.path.join(V, 'p8988_pages', 'Jv1', 'detections.csv'))
    os.makedirs(os.path.join(T, 'fp_miss'))
    try:
        fp_filters.build(V, os.path.join(T, 'fp_miss'), PAGES, ex=NoEx(), log=lambda m: None)
        raise AssertionError('a listed page directory without detections.csv was not refused')
    except fp_filters.HidesRefusal as e:
        assert 'no detections.csv' in str(e) and 'p8988_pages/Jv1' in str(e), e
    assert not os.listdir(os.path.join(T, 'fp_miss'))
    r2 = fp_filters.build(V, os.path.join(T, 'fp_miss'), PAGES, ex=NoEx(), log=lambda m: None, refuse_missing=False)
    assert r2['missing'] == 1 and r2['pages']['p8988']['missing_first'] == ['p8988_pages/Jv1'], r2['pages']['p8988']
    ok('fp_filters refusals: files exist, excluded cell-band, no queue, nothing listed, a listed page directory without '
       'detections.csv (nothing written); with refuse_missing=False it is counted')

    # the install: both .part files tested, the json renamed before the CSV; a half-installed pair refused, never overwritten;
    # a failure leaves no .part file and no json without its CSV
    meta = json.load(open(res['json']))
    good = [list(r) for r in rows]
    cnt = dict(align_failed=0, align_held=2, flood=120)
    order, real_replace = [], os.replace

    def rec_replace(a, b):
        order.append(os.path.basename(b))
        return real_replace(a, b)
    W = os.path.join(T, 'fp_write')
    os.makedirs(W)
    open(os.path.join(W, 'fp_filters.csv.part'), 'w').write('left by a dead attempt\n')
    notes = []
    fp_filters.os.replace = rec_replace
    try:
        fp_filters.write_files(W, good, meta, cnt, log=notes.append)
    finally:
        fp_filters.os.replace = real_replace
    assert order == ['fp_filters.json', 'fp_filters.csv'], order
    assert sorted(os.listdir(W)) == ['fp_filters.csv', 'fp_filters.json'] and any('earlier attempt' in n for n in notes), notes
    assert open(os.path.join(W, 'fp_filters.csv'), 'rb').read() == open(res['csv'], 'rb').read()
    for keep in ('fp_filters.csv', 'fp_filters.json'):
        H = os.path.join(T, 'fp_half_' + keep.split('.')[1])
        os.makedirs(H)
        shutil.copy(os.path.join(W, keep), H)
        for fn in (lambda: fp_filters.build(V, H, PAGES, ex=NoEx(), log=lambda m: None),
                   lambda: fp_filters.write_files(H, good, meta, cnt, log=lambda m: None)):
            try:
                fn()
                raise AssertionError(f'a half-installed pair ({keep} alone) was not refused')
            except fp_filters.HidesRefusal as e:
                assert 'HALF-INSTALLED' in str(e) and keep in str(e), e
        assert os.listdir(H) == [keep], os.listdir(H)
    F = os.path.join(T, 'fp_fail')
    os.makedirs(F)
    try:
        fp_filters.write_files(F, good + [['x_1', '1.0', '2.0', 'no such state', '']], meta, cnt, log=lambda m: None)
        raise AssertionError('a file failing its content test was installed')
    except fp_filters.HidesRefusal as e:
        assert 'content test' in str(e), e
    assert not os.listdir(F), os.listdir(F)

    def fail_csv(a, b):
        if b.endswith('.csv'):
            raise OSError('simulated rename failure')
        return real_replace(a, b)
    fp_filters.os.replace = fail_csv
    try:
        fp_filters.write_files(F, good, meta, cnt, log=lambda m: None)
        raise AssertionError('a failed CSV rename was not raised')
    except OSError as e:
        assert 'simulated' in str(e)
    finally:
        fp_filters.os.replace = real_replace
    assert not os.listdir(F), os.listdir(F)
    ok('fp_filters install: json renamed before the CSV; a dead attempt\'s .part removed with a note; CSV or json alone refused '
       'as HALF-INSTALLED (build and write), untouched; a content-test failure or a failed CSV rename leaves nothing')

    # the content test's negatives
    def bad(name, edit_csv=None, edit_json=None):
        d = os.path.join(T, 'neg_' + name)
        os.makedirs(d)
        shutil.copy(res['csv'], d)
        shutil.copy(res['json'], d)
        if edit_csv:
            txt = open(os.path.join(d, 'fp_filters.csv'), newline='').read()
            open(os.path.join(d, 'fp_filters.csv'), 'w', newline='').write(edit_csv(txt))
        if edit_json:
            m = json.load(open(os.path.join(d, 'fp_filters.json')))
            edit_json(m)
            json.dump(m, open(os.path.join(d, 'fp_filters.json'), 'w'))
        ok_, n, det = fp_filters.check_files(os.path.join(d, 'fp_filters.csv'))
        assert not ok_, (name, det)
    lines = lambda t: t.split('\r\n')
    bad('state', edit_csv=lambda t: t.replace(F0, 'flood', 1))
    bad('nostate', edit_csv=lambda t: '\r\n'.join(lines(t) + ['x_1,1.0,2.0,,', '']))
    bad('dup', edit_csv=lambda t: '\r\n'.join(lines(t)[:-1] + [lines(t)[2], '']))
    bad('header', edit_csv=lambda t: t.replace(',fp_c_flood\r\n', ',fp_c_other\r\n', 1))
    bad('default_off', edit_json=lambda m: m['cats'][0].__setitem__('default_off', ['no such state']))
    bad('json', edit_json=lambda m: m.__setitem__('cats', []))
    ok('fp_filters content test: wrong state, row without a state, repeated key, header, default_off, no category')


def main():
    test_config()
    test_plan()
    test_stages()
    test_serve_order()
    test_tess_step()
    test_fp_filters()
    print(f'ALL {len(PASSED)} CHECKS PASSED')


if __name__ == '__main__':
    main()
