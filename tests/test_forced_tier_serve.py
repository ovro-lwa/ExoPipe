#!/usr/bin/env python3
"""STANDING TEST - the 10-minute and deep forced-photometry pages p9120-p9123 (serve --tier snapshot --family forced-tier: family
T's <run>.new installed, its page inputs, one registration) and the pinned pages p9411-p9414, p9300, p9301 (--family pinned);
row 12 of refactor_20261012/plan/CUTOVER_REMAINING_20261124.md. Data-free: a synthetic site file and synthetic runs under a fresh
scratch directory ($EXOPIPE3_TEST_SCRATCH, else the system temp directory), removed at exit. Plain asserts. (The planned entries
against the LIVE registry and hub are tests/test_serve.py test_production_readonly, on the cluster.)

  1. config: the ten pages (ports, families, hub keys, run / queue / labels / code / env / title of the pinned ones), their
     exclusion-gate keys refusing, the forced-tier server environment, the install names.
  2. plan: the forced-tier runs (production root; parity under <out>/review, where family T writes under the same --out); argv
     in the forced pages' order with the three siblings in the live words; the title from the run's queue header (samples,
     generation from the run name; G26 refuses another header, Stokes or tier, a missing queue, another run name); the hub
     button's key order and generation text. Pinned: their own code (or the tier's), trailing --filter-defaults off, their own
     environment ({} for the deep samples) in the registry entry.
  3. readiness: no <run>.new = NO-INPUT; no family T record, one of another review root, one not done, one without the queue
     output or with a failed test = not ready; a done record naming the queue = ready, with its site.
  4. the install: check_page's refusals before anything moves (not ready, files missing, an install pending, the kept name taken,
     the exclusion gate); a failure after staging (the App probe, the page inputs, the second gate) puts <run>.new back; a
     success swaps with the served run's judgements copied into the new run, the served run kept, the stage directory gone; a
     first install only renames.
  5. the exclusion gate's keyer for family T rows (10-minute window and deep cand_ids, columns) and its refusal of an unkeyed
     served event; the content tests of the TESS rows and window times held to the served candidates.
  6. dispatch: steps/serve.py FAMILIES and _module; pinned refuses a parity serve.
"""
import atexit
import csv
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
T = tempfile.mkdtemp(prefix='forced_tier_serve_', dir=_base)
atexit.register(shutil.rmtree, T, True)
for d in ('exopipe_ops', 'clv15'):
    os.makedirs(os.path.join(T, 'site', d), exist_ok=True)
SITE = os.path.join(T, 'site', 'site.py')
with open(SITE, 'w') as f:
    f.write(f'PYTHON = {{"default": {sys.executable!r}}}\n'
            f'LEGACY_CODE_ROOTS = [{os.path.join(T, "site", "clv15")!r}, {os.path.join(T, "site", "exopipe_ops")!r}]\n')
os.environ['EXOPIPE3_SITE'] = SITE

from exopipe.phase3.config import CFG                                  # noqa: E402
from exopipe.phase3.core import validate                               # noqa: E402
from exopipe.phase3.core.record import Refusal, Run                    # noqa: E402
from exopipe.phase3.lib.pages import tests as page_tests_mod           # noqa: E402,F401  (registers tess_coverage, ...)
from exopipe.phase3.lib.serve import exclusion_gate, forced_tier, pinned, plan as planmod, register  # noqa: E402
from exopipe.phase3.steps import serve as serve_step                    # noqa: E402

S = CFG.serve
UTC = '2099-01-01 00:00 UT'
PASSED = []


def ok(name):
    PASSED.append(name)
    print('PASS', name)


def args(**kw):
    a = dict(tier='snapshot', family='forced-tier', run_label=None, label=None, pages=None, stokes=None, port_offset=0,
             forced_tag=None, source=None, restart=None, stop=None, lsts=None, stages=None, from_stage=None, until=None)
    a.update(kw)
    return types.SimpleNamespace(**a)


def refused(fn, *a, want=None, **kw):
    try:
        fn(*a, **kw)
    except Refusal as e:
        assert want is None or want in str(e), (want, str(e))
        return str(e)
    raise AssertionError(f'{fn.__name__} did not refuse')


HEADER = '# queue_name: forced photometry (HWJ + 10 pc + LPT), Stokes {S}, {tier} tier\n'


def make_run(D, S_='V', tier='10min', header=None, label=None):
    """A synthetic family T run tree: the queue with its header, the catalogue views, the fits / association tables, cards,
    one served event."""
    os.makedirs(os.path.join(D, f'source_catalogs_{S_}'), exist_ok=True)
    os.makedirs(os.path.join(D, f'source_catalogs_{S_}_loc'), exist_ok=True)
    os.makedirs(os.path.join(D, 'cards'), exist_ok=True)
    with open(os.path.join(D, f'queue_{S_}.txt'), 'w') as f:
        f.write(header if header is not None else HEADER.format(S=S_, tier=tier))
        f.write('# run: test\nHD_1\n')
    for x in ('forced_fits.csv', 'forced_assoc.csv', 'forced_assoc_identity.csv'):
        with open(os.path.join(D, x), 'w') as f:
            f.write('cand_id\n')
    with open(os.path.join(D, 'cards', 'lines.json'), 'w') as f:
        f.write('{}')
    date = '2025-09-03T02' if tier == '10min' else '2025-09-03'
    ev = os.path.join(D, 'candidates', S_, f'HD_1_18MHz_{tier}')
    os.makedirs(ev, exist_ok=True)
    with open(os.path.join(ev, 'detections.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lst', 'date', 'band', 'stokes', 'tier', 'cand_id'])
        w.writerow(['21h', date, '18MHz', S_, tier, f'F_21h_{date}_18MHz_{tier}_HWJ_HD_1'])
    if label is not None:
        os.makedirs(os.path.join(D, 'judgements'), exist_ok=True)
        with open(os.path.join(D, 'judgements', 'judgements.jsonl'), 'w') as f:
            f.write(label)


# ---- 1. config ----------------------------------------------------------------------------------------------------------
def test_config():
    ft = [(k, c) for (t, k), c in S.PAGES.items() if t == 'snapshot' and c['family'] == 'forced-tier']
    got = [(c['port'], c['stokes'], c['kind'], c['queue'], c['catalog'], c['hub']) for _, c in ft]
    assert got == [(9120, 'V', '10min', 'queue_V.txt', 'source_catalogs_V', ('V', '10-min-forced')),
                   (9121, 'I', '10min', 'queue_I.txt', 'source_catalogs_I', ('I', '10-min-forced')),
                   (9122, 'V', 'deep', 'queue_V.txt', 'source_catalogs_V', ('V', '1-hour-forced')),
                   (9123, 'I', 'deep', 'queue_I.txt', 'source_catalogs_I', ('I', '1-hour-forced'))], got
    assert all(c['loc_offsets'] and c['labels'] == 'judgements' and c['twin'] is None for _, c in ft)
    pin = {(t, c['port']): c for (t, k), c in S.PAGES.items() if c['family'] == 'pinned'}
    assert sorted(pin) == [('deep', 9411), ('deep', 9412), ('sublst', 9300), ('sublst', 9301), ('sublst', 9413),
                           ('sublst', 9414)], sorted(pin)
    assert pin[('deep', 9411)]['run'] == 'Run_P3S5' and pin[('deep', 9411)]['env'] == {} \
        and tuple(pin[('deep', 9411)]['extra']) == ('--filter-defaults', 'off')
    assert pin[('sublst', 9413)]['run'] == 'Run_SLSTB' and 'code_frozen/SLSTB_' in pin[('sublst', 9413)]['code'] \
        and 'env' not in pin[('sublst', 9413)]
    assert pin[('sublst', 9300)]['run'] == pin[('sublst', 9301)]['run'] == 'Run_SLSTB_GRPB_PREVIEW_20261026'
    assert pin[('sublst', 9300)]['code'] == S.GROUPB_PREVIEW_CODE and pin[('sublst', 9300)]['title'].endswith('p8790') \
        and pin[('sublst', 9301)]['title'].endswith('p8789')
    assert all(c.get('hub') is None for c in pin.values())
    for t, fam in (('snapshot', 'forced-tier'), ('deep', 'pinned'), ('sublst', 'pinned')):
        assert exclusion_gate.refuses(t, fam), (t, fam)
    assert set(S.SERVER_ENV['snapshot-forced-tier']) == set(S.SERVER_ENV['snapshot-forced'])
    assert S.FORCED_TIER_SCINT_TIER == {'10min': 'tenmin', 'deep': 'deep'}
    assert S.FORCED_TIER_PARENTS == {'10min': 'Run_SLSTB', 'deep': 'Run_P3S5'}
    ok('config: four forced-tier pages and six pinned pages, gate keys refusing, environment, install names')


# ---- 2. plan --------------------------------------------------------------------------------------------------------------
def test_plan():
    out = os.path.join(T, 'out_plan')
    p = planmod.make(args(), out)
    assert p.parity and [pg.port for pg in p.pages] == [9120, 9121, 9122, 9123]
    names = [pg.run_name for pg in p.pages]
    assert names == ['Run_SNAPFORCED_V10min_gen5', 'Run_SNAPFORCED_I10min_gen5', 'Run_SNAPFORCED_Vdeep_gen5',
                     'Run_SNAPFORCED_Ideep_gen5'], names
    R = os.path.join(out, CFG.forced_snapshot.REVIEW_SUBDIR_UNDER_OUT)
    assert all(pg.root == os.path.join(R, pg.run_name) for pg in p.pages)
    assert all(pg.root_src == os.path.join(CFG.paths.SNAPSHOT_REVIEW_ROOT, pg.run_name) for pg in p.pages)
    pg = p.pages[0]
    make_run(pg.root, 'V', '10min')
    p.forced_after = {o.key: o.root for o in p.pages}
    a = p.argv(pg, UTC)
    flags = [x for x in a[2:] if x.startswith('--')]
    assert flags == ['--run', '--queue', '--catalog-dir', '--port', '--loc-offsets', '--default-fp-reason', '--sibling',
                     '--sibling', '--sibling', '--title', '--lists-dir'], flags
    sib = [a[i + 1] for i, x in enumerate(a) if x == '--sibling']
    assert sib == [f'9121:{R}/Run_SNAPFORCED_I10min_gen5/source_catalogs_I:forced Stokes I 10-minute page',
                   f'9122:{R}/Run_SNAPFORCED_Vdeep_gen5/source_catalogs_V:forced Stokes V deep (1-hour) page',
                   f'9123:{R}/Run_SNAPFORCED_Ideep_gen5/source_catalogs_I:forced Stokes I deep (1-hour) page'], sib
    title = a[a.index('--title') + 1]
    assert title == ('Snapshot-chain FORCED photometry (HWJ + 10 pc + LPT), Stokes V, 10-minute windows - each event fitted in '
                     'its own image (Gaussian >= 7 sigma, template below), localized, associated (target / background source '
                     '/ ambiguous); no reference tie at this tier (Run_SNAPFORCED_V10min_gen5) - served 2099-01-01 00:00 UT'), title
    h = p.hub_entry(pg, UTC, pg.root)
    assert h[:2] == ('V', '10-min-forced') and [x.split('=')[0] for x in h[2]] == ['run', 'generation', 'served', 'live', 'port']
    kv = dict(x.split('=', 1) for x in h[2])
    assert kv['generation'] == ('forced photometry at the HWJ + 10 pc + LPT targets, 10min tier, generation-5 tier cutouts, '
                                'Stokes V events') and kv['port'] == '9120' and kv['run'] == pg.run_name, kv
    e = register.page_entry(p, pg, UTC, pg.root)
    assert e['id'] == 'p9120' and e['env'] == {k: v for k, v in p.server_env_extra(pg).items() if k.startswith('P3_')}
    # G26: the words come from the run itself
    for hdr, want in (('# queue_name: forced photometry (HWJ), Stokes I, 10min tier\n', 'not the family T queue header'),
                      ('# queue_name: forced photometry (HWJ), Stokes V, deep tier\n', 'not the family T queue header'),
                      ('# something else\n', 'not the family T queue header')):
        make_run(pg.root, 'V', '10min', header=hdr)
        refused(planmod.forced_tier_words, pg, pg.root, want=want)
    refused(planmod.forced_tier_words, pg, os.path.join(T, 'nowhere', pg.run_name), want='no queue')
    other = os.path.join(T, 'renamed', 'Run_SNAPFORCED_V10min_gen6')
    make_run(other)
    refused(planmod.forced_tier_words, pg, other, want='not forced_tier_pages.RUN_NAME')
    make_run(pg.root, 'V', '10min')
    assert planmod.forced_tier_words(pg, pg.root) == ('HWJ + 10 pc + LPT', '5')
    # family T's own content test reads the same header (an LPT sample in the parentheses: f06a293's header)
    from exopipe.phase3.lib.forced import tests_tier_pages
    for hdr in (HEADER.format(S='V', tier='10min'), '# queue_name: forced photometry (HWJ + 10 pc), Stokes I, deep tier\n'):
        assert tests_tier_pages._QH.match(hdr), hdr
    # production plan: the runs under the snapshot review root
    pp = planmod.make(args(), CFG.paths.SNAPSHOT_REVIEW_ROOT)
    assert not pp.parity and all(x.root == x.root_src for x in pp.pages)
    # pinned
    for tier in ('deep', 'sublst'):
        q = planmod.make(args(tier=tier, family='pinned'), os.path.join(T, 'out_pin'))
        for x in q.pages:
            U = x.root
            a = q.argv(x, UTC)
            code = x.c.get('code') or S.SERVER_ARGV0[tier].format(ops=q.ops)     # deep: relative, run from the ops root
            assert a[1] == code, (x.key, a[1])
            assert a[a.index('--run') + 1] == U and os.path.basename(U) == x.c['run']
            assert a[a.index('--title') + 1] == x.c['title']
            if x.c.get('extra'):
                i = a.index('--filter-defaults')
                assert a[i:i + 2] == ['--filter-defaults', 'off'] and a[i + 2] == '--lists-dir', a[i:]
            env = q.server_env_extra(x)
            assert env == ({} if 'env' in x.c else q.server_env_extra()), (x.key, env)
    ok('plan: forced-tier runs, argv with the live siblings and title words, hub button, G26 on the run\'s own header; pinned '
       'code, trailing arguments and environment')


# ---- 3. readiness -------------------------------------------------------------------------------------------------------
def write_record(out, root, q, status='done', test_passed=True, site='/site'):
    os.makedirs(os.path.join(out, '_records', 'forced'), exist_ok=True)
    rec = dict(unit='T_runs', status=status, reason='x' if status != 'done' else None,
               meta=dict(site=site, review_root=root, verify_line='TIER-CHAIN-OK test'),
               outputs=[{'path': q, 'class': 'forced_tier_run'}] if q else [],
               tests=[dict(name='forced_tier_run.tier_run', path=q, passed=test_passed)] if q else [])
    with open(os.path.join(out, '_records', 'forced', 'T_runs.json'), 'w') as f:
        json.dump(rec, f)


def test_ready():
    out = os.path.join(T, 'out_ready')
    p = planmod.make(args(pages='forced-tier-V-10min'), out)
    pg = p.pages[0]
    root = forced_tier.install_root(pg)
    assert root == os.path.join(out, 'review')
    okk, text, _ = forced_tier.ready(pg)
    assert not okk and text.startswith('NO-INPUT: '), text
    new = forced_tier.new_tree(pg)
    make_run(new)
    q = os.path.join(new, 'queue_V.txt')
    okk, text, _ = forced_tier.ready(pg)
    assert not okk and 'no family T record' in text, text
    write_record(out, '/elsewhere', q)
    okk, text, _ = forced_tier.ready(pg)
    assert not okk and 'built into /elsewhere' in text, text
    write_record(out, root, q, status='refused')
    assert not forced_tier.ready(pg)[0]
    write_record(out, root, None)
    assert 'wrote no' in forced_tier.ready(pg)[1]
    write_record(out, root, q, test_passed=False)
    assert 'failed' in forced_tier.ready(pg)[1]
    write_record(out, root, q, site='/the/site')
    okk, text, site = forced_tier.ready(pg)
    assert okk and site == '/the/site' and 'TIER-CHAIN-OK' in text, (okk, text, site)
    assert forced_tier.record_outs(root)[-1] == out
    ok('readiness: NO-INPUT without <run>.new; the family T record of this review root, done, with its queue output passed')


# ---- 4. the install -----------------------------------------------------------------------------------------------------
def test_install():
    out = os.path.join(T, 'out_install')
    p = planmod.make(args(pages='forced-tier-V-10min'), out)
    pg = p.pages[0]
    root, new, live = forced_tier.install_root(pg), forced_tier.new_tree(pg), pg.root
    run = Run('serve', ['test'], out, CFG, config_sections=None)
    calls = []
    saved = (forced_tier.page_output, forced_tier.servers.app_probe, forced_tier.page_inputs, forced_tier._gate)
    forced_tier.page_output = lambda u, pg_, path, cls, what=None, **ctx: calls.append(('test', cls, os.path.basename(path)))
    forced_tier.servers.app_probe = lambda plan, pg_, D, unit=None: calls.append(('probe', D)) or {'n': 1}
    fail = {}

    def inputs(u, plan, pg_, D):
        calls.append(('inputs', D))
        assert os.path.basename(D) == pg_.run_name and os.path.basename(os.path.dirname(D)) == f'{pg_.run_name}.install'
        if fail.get('inputs'):
            raise Refusal('inputs failed (test)')
        with open(os.path.join(D, 'scintillation_index.csv'), 'w') as f:
            f.write('x')

    def gate(u, plan, pg_, D, when):
        calls.append(('gate', when))
        if fail.get('gate') == when:
            raise Refusal(f'G17 serve-time exclusion gate {when}: hit (test)')
    forced_tier.page_inputs, forced_tier._gate = inputs, gate
    try:
        with run.unit('t_install', stage='install') as u:
            # refusals before anything moves
            refused(forced_tier.check_page, u, p, pg, '20990101T000000Z', want='NO-INPUT')
            make_run(new, label='{"new": 1}\n')
            refused(forced_tier.check_page, u, p, pg, '20990101T000000Z', want='not ready')
            write_record(out, root, os.path.join(new, 'queue_V.txt'))
            os.remove(os.path.join(new, 'forced_fits.csv'))
            refused(forced_tier.check_page, u, p, pg, '20990101T000000Z', want='lacks forced_fits.csv')
            make_run(new, label='{"new": 1}\n')
            os.mkdir(forced_tier.stage_dir(pg))
            refused(forced_tier.check_page, u, p, pg, '20990101T000000Z', want='did not finish')
            os.rmdir(forced_tier.stage_dir(pg))
            make_run(live, label='{"old": 1}\n')
            os.mkdir(forced_tier.keep_path(pg, '20990101T000000Z'))
            refused(forced_tier.check_page, u, p, pg, '20990101T000000Z', want='kept name')
            os.rmdir(forced_tier.keep_path(pg, '20990101T000000Z'))
            fail['gate'] = 'on the new build'
            refused(forced_tier.check_page, u, p, pg, '20990101T000000Z', want='hit')
            assert os.path.isdir(new) and not os.path.exists(forced_tier.stage_dir(pg))
            # failures after staging put <run>.new back
            for k, v in (('gate', 'on the staged run after its page inputs'), ('inputs', True)):
                fail.clear()
                fail[k] = v
                refused(forced_tier.check_page, u, p, pg, '20990101T000000Z')
                assert os.path.isdir(new) and not os.path.exists(forced_tier.stage_dir(pg)), k
                with open(os.path.join(live, 'judgements', 'judgements.jsonl')) as f:
                    assert f.read() == '{"old": 1}\n'
            # the swap: the served run's judgements copied into the new run (rsync --update: the newer file wins)
            fail.clear()
            os.utime(os.path.join(new, 'judgements', 'judgements.jsonl'), (1, 1))
            with open(os.path.join(live, 'judgements', 'extra.jsonl'), 'w') as f:
                f.write('{"extra": 1}\n')
            D, exp, notes, kept = forced_tier.check_page(u, p, pg, '20990101T000000Z')
            assert D == live and kept == forced_tier.keep_path(pg, '20990101T000000Z') and exp == {'n': 1}
            assert not os.path.exists(new) and not os.path.exists(forced_tier.stage_dir(pg))
            assert os.path.exists(os.path.join(live, 'scintillation_index.csv'))        # the staged run, now served
            assert open(os.path.join(live, 'judgements', 'judgements.jsonl')).read() == '{"old": 1}\n'
            assert open(os.path.join(live, 'judgements', 'extra.jsonl')).read() == '{"extra": 1}\n'
            assert not os.path.exists(os.path.join(kept, 'scintillation_index.csv'))
            assert [c[0] for c in calls if c[0] in ('probe', 'inputs')][-2:] == ['probe', 'inputs']
            assert ('test', 'forced_tier_run', 'queue_V.txt') in calls
            # a first install (no served run) only renames
            p2 = planmod.make(args(pages='forced-tier-I-deep'), out)
            pg2 = p2.pages[0]
            make_run(forced_tier.new_tree(pg2), 'I', 'deep')
            write_record(out, root, os.path.join(forced_tier.new_tree(pg2), 'queue_I.txt'))
            D2, _e, _n, kept2 = forced_tier.check_page(u, p2, pg2, '20990101T000000Z')
            assert kept2 is None and os.path.isdir(D2) and not os.path.exists(forced_tier.new_tree(pg2))
            u.note('test')
    finally:
        forced_tier.page_output, forced_tier.servers.app_probe, forced_tier.page_inputs, forced_tier._gate = saved
    ok('install: refusals before anything moves; a failure after staging puts <run>.new back; the swap keeps the served '
       'run and its judgements (rsync --update); a first install only renames')


# ---- 5. gate keyer and content tests --------------------------------------------------------------------------------------
class _Ex:
    def __init__(self, bad):
        self.bad = bad

    def is_excluded(self, lst, band, date):
        return (lst, date, band) in self.bad


def test_gate_and_tests():
    k = exclusion_gate.keys_forced_tier
    assert k({'cand_id': 'F_21h_2025-09-03T02_18MHz_10min_HWJ_HD_1'}) == [('21h', '2025-09-03', '18MHz')]
    assert k({'cand_id': 'F_05h_2025-01-06_Red_deep_10pc_GJ_1'}) == [('05h', '2025-01-06', 'Red')]
    assert k({'lst': '13h', 'date': '2025-04-10T02', 'band': 'Blue'}) == [('13h', '2025-04-10', 'Blue')]
    assert k({'source_id': 'HD_1'}) == []
    D = os.path.join(T, 'gate_run')
    make_run(D)
    s = exclusion_gate.scan_forced_tier_run(D, ex=_Ex({('21h', '2025-09-03', '18MHz')}))
    assert len(s.hits) == 1 and s.hits[0][2:] == ('21h', '2025-09-03', '18MHz'), s.hits
    assert not exclusion_gate.scan_forced_tier_run(D, ex=_Ex(set())).hits
    with open(os.path.join(D, 'candidates', 'V', 'HD_1_18MHz_10min', 'detections.csv'), 'a') as f:
        f.write(',,,V,10min,X_1\n')
    refused(exclusion_gate.scan_forced_tier_run, D, ex=_Ex(set()), want='without an (LST, night, band) key')
    # TESS rows and window times held to the served candidates (no master_catalog.csv)
    R = os.path.join(T, 'tess_run')
    make_run(R)
    cid = 'F_21h_2025-09-03T02_18MHz_10min_HWJ_HD_1'
    tc = os.path.join(R, 'tess_coverage.csv')
    with open(tc, 'w') as f:
        f.write('# tess_coverage.py test\ncand_id,stokes,lst,date,band,status,sector,camera,ccd\n'
                f'{cid},V,21h,2025-09-03,18MHz,not observed,,,\n')
    assert all(r['passed'] for r in validate.run('tess_coverage', tc, written=1, left_out=0, served='candidates'))
    bad = validate.run('tess_coverage', tc, written=1, left_out=1, served='candidates')
    assert not all(r['passed'] for r in bad), bad
    ft = os.path.join(R, 'frame_times.csv')
    cols = page_tests_mod.FRAME_TIMES_COLUMNS
    with open(ft, 'w', newline='') as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        row = {c: '' for c in cols}
        row.update(stokes='V', lst='21h', band='18MHz', date='2025-09-03T02', date_obs='2025-09-03T06:22:33',
                   date_end='2025-09-03T06:32:34')
        w.writerow(row)
    assert all(r['passed'] for r in validate.run('tenmin_frame_times', ft, windows=1, needed=1, served='candidates'))
    assert not all(r['passed'] for r in validate.run('tenmin_frame_times', ft, windows=1, needed=2, served='candidates'))
    ok('exclusion gate keys family T rows per night and refuses an unkeyed served event; TESS rows and window times held to '
       'the served candidates')


# ---- 6. dispatch ----------------------------------------------------------------------------------------------------------
def test_dispatch():
    assert 'forced-tier' in serve_step.FAMILIES and 'pinned' in serve_step.FAMILIES
    assert serve_step._module(planmod.make(args(), os.path.join(T, 'o1'))) is forced_tier
    q = planmod.make(args(tier='deep', family='pinned'), os.path.join(T, 'o2'))
    assert serve_step._module(q) is pinned and q.parity
    refused(pinned.run, args(), None, q, ('register',), want='no parity serve')
    assert forced_tier.STAGES == ('install', 'hub', 'verify') and pinned.STAGES == ('register', 'verify')
    ok('dispatch: families forced-tier and pinned, their stages; pinned refuses a parity serve')


def main():
    for t in (test_config, test_plan, test_ready, test_install, test_gate_and_tests, test_dispatch):
        t()
    print(f'{len(PASSED)} passed')


if __name__ == '__main__':
    main()
