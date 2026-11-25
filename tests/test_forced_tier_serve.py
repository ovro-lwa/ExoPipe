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
     the exclusion gate); then whole serves (install_stage through flow.register_pages, the parity registry <out>/two_port
     written by register.prepare / write, the server and its build stubbed) with a failure injected at each point: BEFORE the
     swap (the stage directory, the rename into it, the first judgements sync, the journal test, the loader check, the server
     check inside the lock, the swap's rename; an OSError of one page's inputs refuses that page only, the other registered)
     the page is served as before and <run>.new is back; AFTER the swap (the second sync, the journal re-test,
     register.write) the install record names the step, the page's unit ends 'installed, not registered: <step>' (status
     error, never refused), and the next serve - no <run>.new - resumes and registers it (the record removed); a success
     keeps the served run and its judgements (rsync --update) and registers the new run with the staged build's title; a
     first install only renames.
  5. the exclusion gate's keyer for family T rows (10-minute window and deep cand_ids, columns), its refusal of an unkeyed
     served event, and its one pass over the run (each file listed and opened once); the content tests of the TESS rows and
     window times held to the served candidates.
  6. dispatch: steps/serve.py FAMILIES and _module; pinned refuses a parity serve; every family's plan words from its module's
     PLAN_HOOK (plan.family_hook: a family without one refuses - no fallback to the tier's words).
  7. pinned: a publish marker standing on its run (Run_P3S5 / Run_SLSTB: the deep and sub-LST publishers' runs) refuses the
     registration, 'publish in progress on <run>, retry after', and the marker stays; the App probe of a pinned page runs
     with exactly the code directory, cwd and environment of its registry entry.
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
from exopipe.phase3.lib.serve import exclusion_gate, flow, forced_tier, pinned, plan as planmod, register, servers  # noqa: E402
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
        refused(forced_tier.forced_tier_words, pg, pg.root, want=want)
    refused(forced_tier.forced_tier_words, pg, os.path.join(T, 'nowhere', pg.run_name), want='no queue')
    other = os.path.join(T, 'renamed', 'Run_SNAPFORCED_V10min_gen6')
    make_run(other)
    refused(forced_tier.forced_tier_words, pg, other, want='not forced_tier_pages.RUN_NAME')
    make_run(pg.root, 'V', '10min')
    assert forced_tier.forced_tier_words(pg, pg.root) == ('HWJ + 10 pc + LPT', '5')
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
    qs = q if isinstance(q, list) else [q] if q else []
    os.makedirs(os.path.join(out, '_records', 'forced'), exist_ok=True)
    rec = dict(unit='T_runs', status=status, reason='x' if status != 'done' else None,
               meta=dict(site=site, review_root=root, verify_line='TIER-CHAIN-OK test'),
               outputs=[{'path': x, 'class': 'forced_tier_run'} for x in qs],
               tests=[dict(name='forced_tier_run.tier_run', path=x, passed=test_passed) for x in qs])
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
KEEP_UTC = '20990101T000000Z'


def mark(D, text):
    with open(os.path.join(D, 'MARK'), 'w') as f:
        f.write(text)


def marked(D):
    try:
        with open(os.path.join(D, 'MARK')) as f:
            return f.read()
    except OSError:
        return None


class Harness:
    """The install's surroundings stubbed: the page's content tests (the judgement journal's injectable), the App probe, the
    page inputs and the gate on the run (each injectable), the server check (the n-th call injectable), the server's loader
    check in register.prepare and register.write (injectable; real otherwise, in the parity registry <out>/two_port), the
    build wait (refuses: no server here - a registered page's unit ends refused 'BUILD-WAIT'), os.mkdir / os.rename of the
    install's paths and the judgements sync (injectable; rsync otherwise)."""

    def __init__(self):
        self.fail, self.calls, self.checks = {}, [], 0

    def __enter__(self):
        H = self
        self.saved = (forced_tier.page_output, servers.app_probe, forced_tier.page_inputs, forced_tier._gate,
                      forced_tier._sync_judgements, register.check_server, register.prepare, register.write,
                      register.wait_built, os.mkdir, os.rename)
        sync0, prep0, write0, mkdir0, rename0 = (forced_tier._sync_judgements, register.prepare, register.write, os.mkdir,
                                                 os.rename)

        def page_output(u, pg_, path, cls, what=None, **ctx):
            H.calls.append(('test', cls, os.path.basename(path), what))
            if cls == 'judgement_journal' and H.fail.get('journal') and H.fail['journal'] in (what or ''):
                raise Refusal(f'the content test of {path} (class judgement_journal) failed: carry_consistent (test)')

        def probe(plan, pg_, D, unit=None):
            H.calls.append(('probe', D))
            return {'n': 1}

        def inputs(u, plan, pg_, D):
            H.calls.append(('inputs', D))
            assert os.path.basename(D) == pg_.run_name and os.path.basename(os.path.dirname(D)) == f'{pg_.run_name}.install'
            if H.fail.get('inputs') == pg_.key:
                raise OSError(5, 'Input/output error (test)', os.path.join(D, 'scintillation_index.csv'))
            for x in list(S.FORCED_TIER_INPUTS_REQUIRED) + ['frame_times.csv']:
                with open(os.path.join(D, x), 'w') as f:
                    f.write('x')

        def gate(u, plan, pg_, D, when, after='nothing installed'):
            H.calls.append(('gate', when))
            if H.fail.get('gate') == when:
                raise Refusal(f'G17 serve-time exclusion gate {when}: hit (test) - {after}')

        def sync(src, dst):
            first = '.install' in dst
            if H.fail.get('sync') == ('first' if first else 'second'):
                raise Refusal(f'judgements copy {src}/judgements/ -> {dst}/judgements/ failed (rsync exit 23: test)')
            return sync0(src, dst)

        def check_server(ports, d):
            H.checks += 1
            if H.fail.get('check') == H.checks:
                raise Refusal(f'G16 two-port server: test refusal (call {H.checks})')
            return {}

        def prepare(entries, d):
            if H.fail.get('prepare'):
                raise Refusal("registration refused by the server's loader (nothing written): test")
            return prep0(entries, d)

        def write(prep):
            if H.fail.get('write'):
                raise OSError(5, 'Input/output error (test)', prep['registry'])
            return write0(prep)

        def wait_built(*a, **k):
            raise Refusal('BUILD-WAIT (test): no two-port server here')

        def mkdir(path, *a, **k):
            if H.fail.get('mkdir') and str(path).endswith('.install'):
                raise OSError(28, 'No space left on device (test)', path)
            return mkdir0(path, *a, **k)

        def rename(src, dst, *a, **k):
            f = H.fail.get('rename')
            if f and f(str(src), str(dst)):
                raise OSError(5, 'Input/output error (test)', src)
            return rename0(src, dst, *a, **k)
        (forced_tier.page_output, servers.app_probe, forced_tier.page_inputs, forced_tier._gate, forced_tier._sync_judgements,
         register.check_server, register.prepare, register.write, register.wait_built, os.mkdir, os.rename) = (
            page_output, probe, inputs, gate, sync, check_server, prepare, write, wait_built, mkdir, rename)
        return self

    def __exit__(self, *exc):
        (forced_tier.page_output, servers.app_probe, forced_tier.page_inputs, forced_tier._gate, forced_tier._sync_judgements,
         register.check_server, register.prepare, register.write, register.wait_built, os.mkdir, os.rename) = self.saved
        return False


def install_setup(name, pages='forced-tier-V-10min', live=True):
    """A parity review root under <T>/inst_<name> with, per page, a served run (MARK old, journal {"old": 1}) and its new build
    <run>.new (MARK new, journal {"new": 1}) that family T's record names; the parity registry directory."""
    out = os.path.join(T, f'inst_{name}')
    os.makedirs(os.path.join(out, S.TWO_PORT_PARITY_DIR), exist_ok=True)
    p = planmod.make(args(pages=pages), out)
    qs = []
    for pg in p.pages:
        if live:
            make_run(pg.root, pg.stokes, pg.kind, label='{"old": 1}\n')
            mark(pg.root, 'old')
        new = forced_tier.new_tree(pg)
        make_run(new, pg.stokes, pg.kind, label='{"new": 1}\n')
        mark(new, 'new')
        qs.append(os.path.join(new, f'queue_{pg.stokes}.txt'))
    write_record(out, forced_tier.install_root(p.pages[0]), qs)
    return out


def serve_once(out, pages='forced-tier-V-10min'):
    p = planmod.make(args(pages=pages), out)
    run = Run('serve', ['test'], out, CFG, config_sections=None)
    forced_tier.install_stage(run, p, flow.new_state(), UTC)
    return p, {x.name: x for x in run.units}


def kept_runs(pg):
    return sorted(x for x in os.listdir(forced_tier.install_root(pg)) if x.startswith(pg.run_name + '.before_'))


def registered(out, pid='p9120'):
    try:
        with open(register.files(os.path.join(out, S.TWO_PORT_PARITY_DIR))[0]) as f:
            return next((e for e in json.load(f)['pages'] if e['id'] == pid), None)
    except OSError:
        return None


def as_before(pg, out, why=''):
    """Served as before: the served run untouched, <run>.new back, no stage directory, no kept run, no install record, nothing
    registered."""
    new = forced_tier.new_tree(pg)
    assert marked(pg.root) == 'old' and marked(new) == 'new', (why, marked(pg.root), marked(new))
    assert open(os.path.join(pg.root, 'judgements', 'judgements.jsonl')).read() == '{"old": 1}\n', why
    assert not os.path.lexists(forced_tier.stage_dir(pg)), why
    assert not os.path.lexists(forced_tier.install_record_path(pg)) and not kept_runs(pg), (why, kept_runs(pg))
    assert registered(out, f'p{pg.port}') is None, why


def test_install_checks():
    """check_page's refusals before anything moves."""
    out = os.path.join(T, 'out_install')
    p = planmod.make(args(pages='forced-tier-V-10min'), out)
    pg = p.pages[0]
    root, new, live = forced_tier.install_root(pg), forced_tier.new_tree(pg), pg.root
    run = Run('serve', ['test'], out, CFG, config_sections=None)
    with Harness() as H, run.unit('t_install', stage='install') as u:
        inst = forced_tier.Install(p, KEEP_UTC)
        refused(forced_tier.check_page, u, p, pg, inst, want='NO-INPUT')
        make_run(new, label='{"new": 1}\n')
        refused(forced_tier.check_page, u, p, pg, inst, want='not ready')
        write_record(out, root, os.path.join(new, 'queue_V.txt'))
        os.remove(os.path.join(new, 'forced_fits.csv'))
        refused(forced_tier.check_page, u, p, pg, inst, want='lacks forced_fits.csv')
        make_run(new, label='{"new": 1}\n')
        os.mkdir(forced_tier.stage_dir(pg))
        refused(forced_tier.check_page, u, p, pg, inst, want='did not finish')
        os.rename(new, new + '.away')
        refused(forced_tier.check_page, u, p, pg, inst, want='did not finish')     # before readiness: no NO-INPUT
        os.rename(new + '.away', new)
        os.rmdir(forced_tier.stage_dir(pg))
        make_run(live, label='{"old": 1}\n')
        os.mkdir(forced_tier.keep_path(pg, KEEP_UTC))
        refused(forced_tier.check_page, u, p, pg, inst, want='kept name')
        os.rmdir(forced_tier.keep_path(pg, KEEP_UTC))
        H.fail['gate'] = 'on the new build'
        refused(forced_tier.check_page, u, p, pg, inst, want='hit')
        assert os.path.isdir(new) and not os.path.exists(forced_tier.stage_dir(pg)) and not inst.staged
        H.fail.clear()
        D, exp, _notes = forced_tier.check_page(u, p, pg, inst)
        assert D == os.path.abspath(live) and exp == {'n': 1} and inst.staged == {pg.key: os.path.join(
            forced_tier.stage_dir(pg), pg.run_name)} and p.title_from[pg.key] == inst.staged[pg.key]
        assert not os.path.exists(new) and os.path.isdir(inst.staged[pg.key])
        assert [c[0] for c in H.calls if c[0] in ('probe', 'inputs')][-2:] == ['probe', 'inputs']
        assert ('test', 'forced_tier_run', 'queue_V.txt') in [c[:3] for c in H.calls]
        assert any(c[1] == 'judgement_journal' and 'first judgements sync' in c[3] for c in H.calls if c[0] == 'test')
        inst.unstage(u, pg, 'test')
        assert os.path.isdir(new) and not os.path.exists(forced_tier.stage_dir(pg)) and not p.title_from
        u.note('test')
    ok('install checks: refusals before anything moves (an unfinished install first, before readiness); a staged build put '
       'back by unstage')


def test_install_before_swap():
    """A failure at each point before the swap: served as before, <run>.new back, the page refused."""
    run_name = 'Run_SNAPFORCED_V10min_gen5'
    cases = [
        ('mkdir', dict(mkdir=True), 'creating'),
        ('rename', dict(rename=lambda a, b: a.endswith(run_name + '.new')), 'staging'),
        ('sync1', dict(sync='first'), 'judgements copy'),
        ('journal1', dict(journal='first judgements sync'), 'judgement_journal'),
        ('prepare', dict(prepare=True), "server's loader"),
        ('check2', dict(check=2), 'G16 two-port server: test refusal (call 2)'),
        ('swap', dict(rename=lambda a, b: a.endswith(f'{run_name}.install/{run_name}') and b.endswith(run_name)),
         'the served run put back'),
    ]
    for name, fail, want in cases:
        out = install_setup(f'pre_{name}')
        with Harness() as H:
            H.fail.update(fail)
            p, U = serve_once(out)
        pg = p.pages[0]
        pu, iu = U[pg.unit], U['snapshot_forced-tier_install']
        assert pu.status == 'refused' and want in (pu.reason or '') + ' '.join(iu.notes) + (iu.reason or ''), \
            (name, pu.status, pu.reason, iu.status, iu.reason)
        assert iu.status in ('refused', 'no-input'), (name, iu.status, iu.reason)
        as_before(pg, out, name)
    # an OSError of one page's inputs refuses that page only: the other page is installed and registered
    pages = 'forced-tier-V-10min,forced-tier-I-10min'
    out = install_setup('pre_oserror', pages)
    with Harness() as H:
        H.fail['inputs'] = 'forced-tier-V-10min'
        p, U = serve_once(out, pages)
    v, i = p.pages
    assert U['snapshot_forced-tier_install'].status == 'done', U['snapshot_forced-tier_install'].reason
    assert U[v.unit].status == 'refused' and 'OSError on the staged run' in U[v.unit].reason, U[v.unit].reason
    as_before(v, out, 'oserror')
    e = registered(out, 'p9121')
    assert e is not None and e['argv'][e['argv'].index('--run') + 1] == os.path.abspath(i.root) and marked(i.root) == 'new'
    assert U[i.unit].status == 'refused' and 'BUILD-WAIT' in U[i.unit].reason, U[i.unit].reason    # registered, no server
    assert not os.path.lexists(forced_tier.install_record_path(i)) and len(kept_runs(i)) == 1
    ok(f'install before the swap: {len(cases)} injected failures (stage directory, rename into it, first judgements sync, '
       'journal test, loader check, server check in the lock, the swap rename) leave the page served as before with '
       '<run>.new back; an OSError of one page\'s inputs refuses that page only')


def test_install_after_swap():
    """A failure at each point after the swap: the run is installed, its record names the step, its unit ends 'installed, not
    registered: <step>' (status error), and the next serve resumes from the step without a <run>.new."""
    for name, fail, step in (('sync2', dict(sync='second'), 'second judgements sync'),
                             ('journal2', dict(journal='second judgements sync'), 'journal re-test'),
                             ('write', dict(write=True), 'register')):
        out = install_setup(f'post_{name}')
        with Harness() as H:
            H.fail.update(fail)
            p, U = serve_once(out)
        pg = p.pages[0]
        pu, iu = U[pg.unit], U['snapshot_forced-tier_install']
        want = f'installed, not registered: {step}'
        assert pu.status == 'error' and want in pu.reason, (name, pu.status, pu.reason)
        assert iu.status == 'error' and f'installed, not registered: page {pg.key}: {step} (' in iu.reason, \
            (name, iu.status, iu.reason)
        rec = json.load(open(forced_tier.install_record_path(pg)))
        kept = os.path.join(forced_tier.install_root(pg), kept_runs(pg)[0])
        assert rec['step'] == step and rec['failed'] and rec['kept'] == kept and rec['run'] == os.path.abspath(pg.root), rec
        assert marked(pg.root) == 'new' and marked(kept) == 'old' and not os.path.lexists(forced_tier.new_tree(pg))
        assert not os.path.lexists(forced_tier.stage_dir(pg)) and registered(out) is None
        # a label written in the kept run after the failure (the page served it until the swap)
        with open(os.path.join(kept, 'judgements', 'late.jsonl'), 'w') as f:
            f.write('{"late": 1}\n')
        if name == 'write':           # the server check refusing before anything: the page stays 'installed, not registered'
            with Harness() as H:
                H.fail['check'] = 1
                p, U = serve_once(out)
            assert U[pg.unit].status == 'error' and 'not resumed' in U[pg.unit].reason, U[pg.unit].reason
            assert os.path.exists(forced_tier.install_record_path(pg))
        if name == 'journal2':        # resumed and failing again: still installed, not registered, the record updated
            with Harness() as H:
                H.fail['journal'] = 'resumed'
                p, U = serve_once(out)
            assert U[pg.unit].status == 'error' and 'installed, not registered: journal re-test' in U[pg.unit].reason
            rec2 = json.load(open(forced_tier.install_record_path(pg)))
            assert rec2['step'] == 'journal re-test' and 'carry_consistent' in rec2['failed'] \
                and rec2['updated'] >= rec['updated'], rec2
        with Harness() as H:
            p, U = serve_once(out)
        iu = U['snapshot_forced-tier_install']
        assert iu.status == 'done', (name, iu.status, iu.reason)
        assert any('RESUMING' in n for n in iu.notes) and any('install record' in n and 'removed' in n for n in iu.notes)
        assert not os.path.lexists(forced_tier.install_record_path(pg)), name
        e = registered(out)
        assert e is not None and e['argv'][e['argv'].index('--run') + 1] == os.path.abspath(pg.root), name
        assert pg.run_name in e['title'] and U[pg.unit].status == 'refused' and 'BUILD-WAIT' in U[pg.unit].reason
        late = os.path.join(pg.root, 'judgements', 'late.jsonl')
        assert os.path.exists(late) == (step == 'second judgements sync'), (name, os.path.exists(late))
        assert any(c[1] == 'judgement_journal' and 'resumed' in c[3] for c in H.calls if c[0] == 'test') \
            == (step != 'register'), name
    ok('install after the swap: an injected failure of the second judgements sync, the journal re-test or register.write '
       'leaves the run installed with its install record; its unit ends "installed, not registered: <step>" (error, never '
       'refused); the next serve resumes from the step without a <run>.new and registers it')


def test_install_success():
    out = install_setup('ok')
    pg0 = planmod.make(args(pages='forced-tier-V-10min'), out).pages[0]
    live = pg0.root
    new = forced_tier.new_tree(pg0)
    os.utime(os.path.join(new, 'judgements', 'judgements.jsonl'), (1, 1))     # the served journal is newer: it wins
    with open(os.path.join(live, 'judgements', 'extra.jsonl'), 'w') as f:
        f.write('{"extra": 1}\n')
    with Harness() as H:
        p, U = serve_once(out)
    pg = p.pages[0]
    iu = U['snapshot_forced-tier_install']
    assert iu.status == 'done' and U[pg.unit].status == 'refused' and 'BUILD-WAIT' in U[pg.unit].reason, (iu.reason,)
    kept = os.path.join(forced_tier.install_root(pg), kept_runs(pg)[0])
    assert marked(live) == 'new' and marked(kept) == 'old' and not os.path.exists(new)
    assert not os.path.exists(forced_tier.stage_dir(pg)) and not os.path.exists(forced_tier.install_record_path(pg))
    assert os.path.exists(os.path.join(live, 'scintillation_index.csv'))
    assert open(os.path.join(live, 'judgements', 'judgements.jsonl')).read() == '{"old": 1}\n'
    assert open(os.path.join(live, 'judgements', 'extra.jsonl')).read() == '{"extra": 1}\n'
    e = registered(out)
    assert e['argv'][e['argv'].index('--run') + 1] == os.path.abspath(live) and pg.run_name in e['title']
    assert iu.meta['installed'][pg.key]['kept'] == kept and not p.title_from
    whens = [c[3] for c in H.calls if c[0] == 'test' and c[1] == 'judgement_journal']
    assert len(whens) == 2 and 'first judgements sync' in whens[0] and 'second judgements sync' in whens[1], whens
    # a first install (no served run) only renames; no second sync
    out = install_setup('first', 'forced-tier-I-deep', live=False)
    with Harness() as H:
        p, U = serve_once(out, 'forced-tier-I-deep')
    pg = p.pages[0]
    assert U['snapshot_forced-tier_install'].status == 'done' and marked(pg.root) == 'new' and not kept_runs(pg)
    assert U['snapshot_forced-tier_install'].meta['installed'][pg.key]['kept'] is None and registered(out, 'p9123')
    ok('install: the swap keeps the served run, its judgements copied (rsync --update), the new run registered with the '
       'staged build\'s title and its record removed; a first install only renames')


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
    # one pass over the run: the file list made once, each file opened once (the served events read in the same pass)
    import builtins
    lists, opened = [], []
    saved = exclusion_gate.snapshot_run_files, builtins.open
    exclusion_gate.snapshot_run_files = lambda r: lists.append(r) or saved[0](r)

    def counting_open(path, *a, **kw):
        if str(path).startswith(D + os.sep):
            opened.append(os.path.relpath(path, D))
        return saved[1](path, *a, **kw)
    builtins.open = counting_open
    try:
        refused(exclusion_gate.scan_forced_tier_run, D, ex=_Ex(set()), want='without an (LST, night, band) key')
    finally:
        exclusion_gate.snapshot_run_files, builtins.open = saved
    assert lists == [D] and len(opened) == len(set(opened)) and \
        os.path.join('candidates', 'V', 'HD_1_18MHz_10min', 'detections.csv') in opened, (lists, opened)
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
    ok('exclusion gate keys family T rows per night, refuses an unkeyed served event, in one pass over the run; TESS rows '
       'and window times held to the served candidates')


# ---- 6. dispatch ----------------------------------------------------------------------------------------------------------
def test_dispatch():
    assert 'forced-tier' in serve_step.FAMILIES and 'pinned' in serve_step.FAMILIES
    assert serve_step._module(planmod.make(args(), os.path.join(T, 'o1'))) is forced_tier
    q = planmod.make(args(tier='deep', family='pinned'), os.path.join(T, 'o2'))
    assert serve_step._module(q) is pinned and q.parity
    refused(pinned.run, args(), None, q, ('register',), want='no parity serve')
    assert forced_tier.STAGES == ('install', 'hub', 'verify') and pinned.STAGES == ('register', 'verify')
    # every family's plan words from its own module's PLAN_HOOK; none falls back to the tier's
    from exopipe.phase3.lib.serve import deep, forced, jupiter, snapshot_copy, sublst_build
    for (tier, fam), mod in (({('snapshot', 'forced-tier'): forced_tier, ('snapshot', 'forced'): forced,
                               ('snapshot', 'review'): snapshot_copy, ('snapshot', 'comb'): snapshot_copy,
                               ('deep', 'review'): deep, ('deep', 'pinned'): pinned, ('sublst', 'pinned'): pinned,
                               ('sublst', 'review'): sublst_build, ('jupiter', 'jupiter'): jupiter}).items()):
        assert planmod.family_module(tier, fam) is mod and planmod.family_hook(tier, fam) is mod.PLAN_HOOK, (tier, fam)
    assert isinstance(forced_tier.PLAN_HOOK, forced_tier.ForcedTierHook) and isinstance(pinned.PLAN_HOOK, pinned.PinnedHook)
    assert {t for t, _f in planmod.FAMILY_MODULES} >= {t for t, _k in S.PAGES}, 'a tier of serve.PAGES without a family module'
    assert {(t, c['family']) for (t, _k), c in S.PAGES.items()} <= set(planmod.FAMILY_MODULES), 'a family without a module'
    refused(planmod.family_hook, 'snapshot', 'no-such-family', want='G26')
    planmod.FAMILY_MODULES[('snapshot', 'hookless')] = 'register'          # a module without PLAN_HOOK
    try:
        refused(planmod.family_hook, 'snapshot', 'hookless', want='no PLAN_HOOK')
        import copy
        p = planmod.make(args(), os.path.join(T, 'o3'))
        pg = copy.copy(p.pages[0])
        pg.family = 'hookless'
        for fn, a in ((p.argv, (pg, UTC)), (p.title, (pg, UTC)), (p.code, (pg,)), (p.server_env_extra, (pg,)),
                      (p.siblings, (pg,))):
            refused(fn, *a, want='no PLAN_HOOK')
    finally:
        del planmod.FAMILY_MODULES[('snapshot', 'hookless')]
    ok('dispatch: families forced-tier and pinned, their stages; pinned refuses a parity serve; each family\'s plan words '
       'from its module\'s PLAN_HOOK, a family without one refused')


# ---- 7. pinned: publish markers it does not own, the App probe's setup ---------------------------------------------------
def test_pinned_hold_and_probe():
    out = os.path.join(T, 'out_pinned')
    os.makedirs(os.path.join(out, S.TWO_PORT_PARITY_DIR), exist_ok=True)
    q = planmod.make(args(tier='deep', family='pinned'), out)
    pg = q.pages[0]
    U = os.path.join(T, 'pin_runs', pg.c['run'])
    os.makedirs(U, exist_ok=True)
    register.set_hold(U, 'publish in progress (test)')
    hold = register.hold_file(U)
    assert pinned.hold_refusal(U).startswith(f'publish in progress on {U}, retry after')
    saved = register.check_server
    register.check_server = lambda *a, **k: {}
    try:
        run = Run('serve', ['test'], out, CFG, config_sections=None)

        def prepare(u):
            return [(pg, U, register.page_entry(q, pg, UTC, U), {'n': 1}, [])], {}
        flow.register_pages(run, q, prepare, 'deep_pinned_register', stage='register', inc_log=False, holds='refuse')
    finally:
        register.check_server = saved
    ru = {x.name: x for x in run.units}['deep_pinned_register']
    assert ru.status == 'refused' and f'publish in progress on {U}, retry after' in ru.reason, (ru.status, ru.reason)
    assert os.path.exists(hold) and registered(out, f'p{pg.port}') is None
    got = {}
    saved = pinned.flow.register_pages
    pinned.flow.register_pages = lambda *a, **k: got.update(k) or {}
    try:
        pinned.register_stage(run, q, flow.new_state(), UTC)
    finally:
        pinned.flow.register_pages = saved
    assert got.get('holds') == 'refuse', got
    # the App probe: exactly the code directory, cwd and environment of the registry entry
    from exopipe.phase3.core import legacy
    seen = []

    class Res:
        rc, tail = 0, '{"n": 2}'
    saved = legacy.run
    legacy.run = lambda script, a, **k: seen.append((a, k)) or Res()
    try:
        for tier in ('deep', 'sublst'):
            qq = planmod.make(args(tier=tier, family='pinned'), out)
            for x in qq.pages:
                e = register.page_entry(qq, x, UTC, x.root)
                code, cwd, env = servers.probe_setup(qq, x)
                assert code == os.path.dirname(e['code']) == x.cwd() == cwd, (x.key, code, e['code'], cwd)
                assert env == dict(S.TWO_PORT_SERVER_ENV, **e['env']), (x.key, env)
                assert servers.app_probe(qq, x, x.root)['n'] == 2
                a, k = seen[-1]
                assert a[0] == code and k['cwd'] == cwd and k['env'] == env, (x.key, a[0], k)
    finally:
        legacy.run = saved
    assert len(seen) == 6
    ok('pinned: a publish marker of a run it does not own refuses the registration ("publish in progress on <run>, retry '
       'after") and stands; the App probe of each pinned page runs with its entry\'s code directory, cwd and environment')


def main():
    for t in (test_config, test_plan, test_ready, test_install_checks, test_install_before_swap, test_install_after_swap,
              test_install_success, test_gate_and_tests, test_dispatch, test_pinned_hold_and_probe):
        t()
    print(f'{len(PASSED)} passed')


if __name__ == '__main__':
    main()
