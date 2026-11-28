#!/usr/bin/env python
"""Every tier's roots name the cutout generation the SERVED pages read, checked against a fixture of the served files.

The fixture tests/fixtures/served_generations.json is written by `--write-fixture` from the files named below (no other file is
read, nothing recursive, no memory map): the hub's page settings serve/registered_pages.json (sha256 and modification time
recorded; registry.json is generated from it) - each forced page's --run and every argument naming a generation, the planets
page's root -; the generation-naming fields of the served runs' page records (the 10-minute / 1-hour site
site_hwj10pc_tiers_gen5 and the planets site site_planets_SUN_gen5; the 10-s and 1-min runs have none, so the last FT-START
line of each run's build log, which records its evaluations, P3_PHOT_ROOT and P3_FORCED_STAGE1); and one stage-1 npz per
served snapshot generation with its minute_basis member.

Data free (CI): the config is compared with the fixture. Asserts:
  - the page table: v2's forced pages (serve.PAGES, families forced and forced-tier) are the registry's ports, each with the
    Stokes and cadence its run names;
  - per cadence, the served runs' generation is photometry.GENERATIONS[CADENCE_TIER[cadence]], and every root and run name
    derived from it is the served one: photometry.CUTOUT_ROOTS / STAGE1_DIR / TIER_FORCED_ROOTS, the five forced_snapshot keys
    (= the 10-s / 1-min build logs' paths), the forced wide-area test's runs and work root, the email's forced runs, the tier
    pages' evaluation names (= the page record's), and the minute rule of each served snapshot generation (= the rule its
    stage-1 minute_basis gives);
  - the planets page: PLANETS_PAGE_GENERATION is the served planets site's, and its per-night inputs (steps_common.
    planets_roots: SolarSystem cutouts and stage 1) and its tier forced root are the page record's; switching the key alone
    moves the per-night inputs and the served-root gate, not the tier root (planets_gen6_stage34_20261127.sh:66-67);
  - every config key naming a generation is either DERIVED from photometry.GENERATIONS or listed in SEPARATE with the reason
    it cannot follow it; every DERIVED key equals its derivation;
  - every key naming gen5 is in GEN5_SERVED (read by a served page) or GEN5_STEP (read by a production step, not served),
    each with its reader's file:line, and that line reads the key; a served entry's value is the served page's input;
  - falsification: each kind of disagreement fails, named.
On a node with the served files: they are read again and compared with the fixture - any difference in a recorded field
fails, named (a registry whose bytes changed with identical fields is a note; `--write-fixture` refreshes the fixture).
Run: python tests/test_served_generation_roots.py [--write-fixture]   (pytest: each test_* fails on a failed check)
"""
import datetime
import functools
import hashlib
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from exopipe.phase3.config import CFG                    # noqa: E402

FIXTURE = os.path.join(REPO, 'tests', 'fixtures', 'served_generations.json')
# ---- the served files the fixture is written from (read only these)
REGISTRY = '/lustre/gh/main/phase3/serve/registered_pages.json'
TIER_RECORD = '/lustre/gh/main/phase3/snapshot_forced/site_hwj10pc_tiers_gen5/page_record.json'   # p9120-p9123's site
PLANETS_RECORD = 'page_record.json'                     # under the registry's planets root
RUN_LOG = '/lustre/gh/main/exopipe_ops/snapshot_forced_pages_run_{run}.log'   # the 10-s / 1-min runs' build logs
STAGE1_NPZ = '/lustre/gh/main/phase3/snapshot_forced/stage1_{generation}/18h_2025-04-11/forced_HWJ_18MHz_I.npz'
FORCED_RUN_RE = re.compile(r'^Run_SNAPFORCED_([IV])(10s|1min|10min|deep)_(gen\d+)$')
CADENCE_TIER = {'10s': 'snapshot', '1min': 'snapshot', '10min': 'sublst', 'deep': 'deep'}   # the review's map
GEN_RE = re.compile(r'(?<![0-9A-Za-z])gen\d+(?![0-9A-Za-z])')
_GEN5 = re.compile(r'(?<![0-9A-Za-z])gen5(?![0-9A-Za-z])')

# ---- THE KEYS THAT NAME A GENERATION BUT ARE NOT DERIVED FROM photometry.GENERATIONS: why each cannot follow it
SEPARATE = {
    'photometry.GENERATIONS': 'the source: tier -> the served cutout generation',
    'photometry.PLANETS_PAGE_GENERATION': 'a served reader that differs: the planets page (/s/planets/) serves gen5 per-night '
                                          'inputs while the snapshot pages serve gen6, until PLANETS-GEN6-DONE',
    'photometry_campaign.GENERATION': 'the generation a campaign WRITES: never the served one (CP-GEN refuses it equal)',
    'photometry_cutouts.STAMP_REFERENCE_ROOT': 'the stamp method\'s reference, generation 6 as approved and tested (Gregg '
                                               '2026-11-21, R-GREGG-20261121-THREE-DECISIONS-APPROVED item 2): a served '
                                               'switch does not re-validate the method on another generation',
    'photometry_fluxscale.GEN5_ROOT': 'check (b) compares a new generation with generation 5 by definition',
    'serve.FORCED_GENERATIONS': 'a table with one row per generation (title words, samples), not a copy of the served one',
    'forced_wide_search.MINUTE_RULE_OF_GENERATION': 'a table with one row per generation (its stage 1\'s minute rule)',
    'forced_tier_pages.EVAL_DIRS': 'MIRROR of the in-place page script\'s literals (tier_forced_page_20261027.py:37-38): what '
                                   'the script reads, which a GENERATIONS switch does not change (PROTECTED)',
    'forced_tier_pages.SCRIPT_EVAL_NAME': 'MIRROR of the script\'s ev_dir :37-38 (PROTECTED)',
    'forced_tier_pages.FT_PRODUCTION': 'MIRROR of planets_forced_page_20261026.py:61, the script\'s default (PROTECTED)',
    'forced_tier_pages.PHOT_PRODUCTION': 'MIRROR of planets_forced_page_20261026.py:54, the script\'s default (PROTECTED)',
    'forced_tier_pages.RUN_NAME': 'MIRROR of the script\'s run_name :394-395: the chain names its runs itself',
    'forced_tier_pages.SITE_NAME': 'the served site directory of p9120-p9123 (a new build takes --tier-site, T-G35)',
    'forced_snapshot.FROZEN_MASTERS': 'the Masters\' generation 4 (FREEZE-SNAPSHOT-MASTERS-GEN4), not a cutout generation',
    'masters_snapshot.FROZEN_ROOT': 'the Masters\' generation 4, not a cutout generation',
    'pages_snapshot.MASTERS_ROOT': 'the Masters\' generation 4, not a cutout generation',
}

# ---- THE KEYS THAT MAY NAME gen5, split by who reads them: key -> (what reads it, the v2 reader 'file:line', the name that
#      line reads). A served entry also names the fixture input it must equal (served_inputs). An entry whose line does not
#      read it, a gen5 key in neither list, or an entry naming no gen5 fails.
TP, SC, SCU = ('exopipe/phase3/lib/forced/tier_pages.py', 'exopipe/phase3/lib/photometry/steps_common.py',
               'exopipe/phase3/lib/photometry/steps_cutouts.py')
GEN5_SERVED = {
    'photometry.GENERATIONS[deep]': ('p9122 / p9123: the generation of their evaluations', TP + ':110', 'GENERATIONS'),
    'photometry.GENERATIONS[sublst]': ('p9120 / p9121: the generation of their evaluations', TP + ':110', 'GENERATIONS'),
    'photometry.CUTOUT_ROOTS[deep]': ('p9122 / p9123: the tier cutouts (page record inputs.phot)', TP + ':175', 'CUTOUT_ROOTS'),
    'photometry.CUTOUT_ROOTS[sublst]': ('p9120 / p9121: the tier cutouts (inputs.phot)', TP + ':175', 'CUTOUT_ROOTS'),
    'photometry.TIER_FORCED_ROOTS[HWJ]': ('p9120-p9123: forced_tiers_gen5 (inputs.tiers)', TP + ':182', 'TIER_FORCED_ROOTS'),
    'photometry.TIER_FORCED_ROOTS[10pc]': ('p9120-p9123: forced_tiers_gen5 (inputs.tiers)', TP + ':182', 'TIER_FORCED_ROOTS'),
    'photometry.TIER_FORCED_ROOTS[SolarSystem]': ('/s/planets/: its 10-minute and deep inputs (inputs.tiers)', SC + ':238',
                                                  'TIER_FORCED_ROOTS'),
    'photometry.PLANETS_PAGE_GENERATION': ('/s/planets/: its per-night cutouts and stage 1 (inputs.phot, inputs.stage1)',
                                           SC + ':233', 'PLANETS_PAGE_GENERATION'),
    'forced_tier_pages.EVAL_DIRS[10min,sb]': ('p9120 / p9121 (inputs.evals 10min|sb)', TP + ':190', 'EVAL_DIRS'),
    'forced_tier_pages.EVAL_DIRS[10min,wb]': ('p9120 / p9121 (inputs.evals 10min|wb)', TP + ':190', 'EVAL_DIRS'),
    'forced_tier_pages.EVAL_DIRS[deep,sb]': ('p9122 / p9123 (inputs.evals deep|sb)', TP + ':190', 'EVAL_DIRS'),
    'forced_tier_pages.EVAL_DIRS[deep,wb]': ('p9122 / p9123 (inputs.evals deep|wb)', TP + ':190', 'EVAL_DIRS'),
    'forced_tier_pages.SCRIPT_EVAL_NAME': ('p9120-p9123: the evaluation names the page script reads (inputs.evals)',
                                           TP + ':198', 'SCRIPT_EVAL_NAME'),
    'forced_tier_pages.FT_PRODUCTION': ('p9120-p9123 (inputs.tiers)', TP + ':238', 'FT_PRODUCTION'),
    'forced_tier_pages.PHOT_PRODUCTION': ('p9120-p9123 (inputs.phot)', TP + ':238', 'PHOT_PRODUCTION'),
    'forced_tier_pages.RUN_NAME': ('p9120-p9123: the registry\'s --run', TP + ':95', 'RUN_NAME'),
    'forced_tier_pages.SITE_NAME': ('p9120-p9123: the site holding their page record', TP + ':147', 'SITE_NAME'),
}
GEN5_STEP = {
    'photometry.TIER_FORCED_ROOTS[LPT]': ('cutmeasure writes LPT\'s tier forced root (the next tier page build reads it, '
                                          'tier_pages.py:182); no served page shows LPT at 10 minutes or 1 hour (the served '
                                          'site is hwj10pc)', SCU + ':1527', 'TIER_FORCED_ROOTS'),
    'photometry_fluxscale.GEN5_ROOT': ('check (b) of a snapshot cutouts unit: the new generation against generation 5',
                                       SCU + ':1052', 'GEN5_ROOT'),
    'serve.FORCED_GENERATIONS{gen5}': ('V2 of a gen5 evaluation: the samples generation 5 measured (and G26 of a gen5 '
                                        'snapshot serve, lib/serve/forced.py:208; the served snapshot runs are gen6)',
                                       'exopipe/phase3/lib/photometry/tests.py:1230', 'FORCED_GENERATIONS'),
    'forced_wide_search.MINUTE_RULE_OF_GENERATION{gen5}': ('W-MINUTE: a wide-area search of gen5 cutouts with no stage-1 npz',
                                                           'exopipe/phase3/lib/forced/wide_stages.py:410',
                                                           'MINUTE_RULE_OF_GENERATION'),
}

FAILED = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else '  %s' % (detail,)), flush=True)
    if not ok:
        FAILED.append(name)


def case(fn):
    """A test function: its checks run and print as under main(); under pytest it fails when any of them failed."""
    @functools.wraps(fn)
    def run(*a, **k):
        n = len(FAILED)
        fn(*a, **k)
        new = FAILED[n:]
        assert not new, '%d check(s) failed: %s' % (len(new), '; '.join(new))
    return run


# ---------------------------------------------------------------------------------------------------------------------
# the served files -> the fixture's fields (only the files named above; json and np.load without a memory map)
# ---------------------------------------------------------------------------------------------------------------------

def _file_id(path):
    with open(path, 'rb') as f:
        b = f.read()
    st = os.stat(path)
    return b, {'path': path, 'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b),
               'mtime_utc': datetime.datetime.utcfromtimestamp(st.st_mtime).strftime('%Y-%m-%dT%H:%M:%SZ')}


def _gen_fields(v, path=''):
    """{dotted path: value} of every string in ``v`` that names a generation."""
    out = {}
    if isinstance(v, str):
        if GEN_RE.search(v):
            out[path] = v
    elif isinstance(v, dict):
        for k, x in v.items():
            out.update(_gen_fields(x, '%s.%s' % (path, k) if path else str(k)))
    elif isinstance(v, (list, tuple)):
        for i, x in enumerate(v):
            out.update(_gen_fields(x, '%s[%d]' % (path, i)))
    return out


def read_served():
    """The fixture's content from the served files: {registry, page_records, run_logs, stage1_minute_basis}."""
    import numpy as np
    b, reg = _file_id(REGISTRY)
    pages = json.loads(b)['pages']
    lines = b.decode().splitlines()
    line_of = {}
    for i, ln in enumerate(lines, 1):
        m = re.match(r'\s*"id":\s*"([^"]+)"', ln)
        if m:
            line_of.setdefault(m.group(1), i)
    forced, planets = {}, None
    for e in pages:
        argv = e.get('argv') or []
        run = argv[argv.index('--run') + 1] if '--run' in argv else None
        if run and FORCED_RUN_RE.match(os.path.basename(run)):
            forced[e['id'].lstrip('p')] = {'line': line_of.get(e['id']), 'run': run,
                                           'generation_args': [a for a in argv if GEN_RE.search(a)]}
        if e.get('id') == 'planets':
            planets = {'line': line_of.get('planets'), 'root': e.get('root')}
    reg.update(forced=forced, planets=planets)
    recs = {}
    for name, path in (('tier pages', TIER_RECORD), ('planets', os.path.join(planets['root'], PLANETS_RECORD))):
        b, rid = _file_id(path)
        rid['site'] = os.path.basename(os.path.dirname(path))
        rid['fields'] = {'inputs.' + k: v for k, v in _gen_fields(json.loads(b).get('inputs') or {}).items()}
        recs[name] = rid
    logs = {}
    for v in sorted(forced.values(), key=lambda x: x['run']):
        run = os.path.basename(v['run'])
        if CADENCE_TIER[FORCED_RUN_RE.match(run).group(2)] != 'snapshot':
            continue
        path = RUN_LOG.format(run=run)
        with open(path) as f:
            starts = [(i, ln) for i, ln in enumerate(f, 1) if ' FT-START %s ' % run in ln]
        i, ln = starts[-1]
        ev = re.search(r' eval=(\S+) \+ (\S+) ', ln)
        fields = {'eval_sb': ev.group(1), 'eval_wb': ev.group(2)}
        for k in ('P3_PHOT_ROOT', 'P3_FORCED_STAGE1'):
            fields[k] = re.search(r' %s=(\S+)' % k, ln).group(1)
        logs[run] = {'path': path, 'line': i, 'fields': fields}
    gens = sorted({FORCED_RUN_RE.match(os.path.basename(v['run'])).group(3) for v in forced.values()
                   if CADENCE_TIER[FORCED_RUN_RE.match(os.path.basename(v['run'])).group(2)] == 'snapshot'}
                  | {gen_of_site(recs['planets']['site'])})
    basis = {}
    for g in gens:
        p = STAGE1_NPZ.format(generation=g)
        with np.load(p, allow_pickle=False) as z:
            basis[g] = {'path': p, 'minute_basis': str(z['minute_basis']) if 'minute_basis' in z.files else ''}
    return {'registry': reg, 'page_records': recs, 'run_logs': logs, 'stage1_minute_basis': basis}


def gen_of_site(name):
    m = re.search(r'_(gen\d+)$', name)
    return m.group(1) if m else None


def write_fixture():
    fx = read_served()
    fx = dict({'about': 'the served generations, written by tests/test_served_generation_roots.py --write-fixture from the '
                        'files it names (serve/registered_pages.json, the served runs\' page records or build logs, one '
                        'stage-1 npz per served snapshot generation); the test compares the config with it, and with the '
                        'live files where they are readable',
               'written_utc': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}, **fx)
    with open(FIXTURE, 'w') as f:
        json.dump(fx, f, indent=1, sort_keys=True)
        f.write('\n')
    print('wrote %s' % FIXTURE)


def load_fixture():
    with open(FIXTURE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------------------
# the config against the fixture
# ---------------------------------------------------------------------------------------------------------------------

def served_runs(fx):
    """{(Stokes, cadence): run name} of the registry's forced pages."""
    out = {}
    for v in fx['registry']['forced'].values():
        S, cad, _ = FORCED_RUN_RE.match(os.path.basename(v['run'])).groups()
        out[(S, cad)] = os.path.basename(v['run'])
    return out


def served_generation(fx, cadence):
    gens = {FORCED_RUN_RE.match(r).group(3) for (_, c), r in served_runs(fx).items() if c == cadence}
    return gens.pop() if len(gens) == 1 else None


def _entry(k):
    return ','.join(map(str, k)) if isinstance(k, tuple) else str(k)


def gen_keys(rx=GEN_RE):
    """{'section.KEY[entry]' (or '{key}' for a dict key): value} of every config value matching ``rx``."""
    out = {}

    def walk(v, path):
        if isinstance(v, str):
            if rx.search(v):
                out[path] = v
        elif isinstance(v, (list, tuple, set, frozenset)):
            for x in v:
                walk(x, path)
        elif isinstance(v, dict):
            for k, x in v.items():
                if isinstance(k, str) and rx.search(k):
                    out[path + '{%s}' % k] = k
                walk(x, path + '[%s]' % _entry(k))
    for n, sec in sorted(CFG.sections().items()):
        for k, v in sec.items():
            walk(v, '%s.%s' % (n, k))
    return out


def _base(key):
    return re.split(r'[\[{]', key, 1)[0]


def served_inputs(fx):
    """{GEN5_SERVED key: (config value, the served input it must equal)}."""
    ph, t = CFG.photometry, CFG.forced_tier_pages
    from exopipe.phase3.lib.forced import tier_pages as TPM
    from exopipe.phase3.lib.photometry import steps_common as K
    tr = fx['page_records']['tier pages']['fields']
    pr = fx['page_records']['planets']['fields']
    pl = K.planets_roots()
    runs = served_runs(fx)
    out = {'photometry.GENERATIONS[deep]': (ph.GENERATIONS['deep'], served_generation(fx, 'deep')),
           'photometry.GENERATIONS[sublst]': (ph.GENERATIONS['sublst'], served_generation(fx, '10min')),
           'photometry.CUTOUT_ROOTS[deep]': (ph.CUTOUT_ROOTS['deep'], tr.get('inputs.phot')),
           'photometry.CUTOUT_ROOTS[sublst]': (ph.CUTOUT_ROOTS['sublst'], tr.get('inputs.phot')),
           'photometry.TIER_FORCED_ROOTS[HWJ]': (ph.TIER_FORCED_ROOTS['HWJ'], tr.get('inputs.tiers')),
           'photometry.TIER_FORCED_ROOTS[10pc]': (ph.TIER_FORCED_ROOTS['10pc'], tr.get('inputs.tiers')),
           'photometry.TIER_FORCED_ROOTS[SolarSystem]': (pl['planets page tier forced root'], pr.get('inputs.tiers')),
           'photometry.PLANETS_PAGE_GENERATION': ((ph.PLANETS_PAGE_GENERATION, pl['planets page stage 1'],
                                                   os.path.dirname(pl['planets page cutouts'])),
                                                  (gen_of_site(fx['page_records']['planets']['site']), pr.get('inputs.stage1'),
                                                   pr.get('inputs.phot'))),
           'forced_tier_pages.SCRIPT_EVAL_NAME': (sorted(t.SCRIPT_EVAL_NAME.format(tier=ti, wb='wb_' if k == 'wb' else '')
                                                         for ti, k in t.EVAL_DIRS),
                                                  sorted(os.path.basename(v) for k, v in tr.items()
                                                         if k.startswith('inputs.evals.'))),
           'forced_tier_pages.FT_PRODUCTION': (t.FT_PRODUCTION, tr.get('inputs.tiers')),
           'forced_tier_pages.PHOT_PRODUCTION': (t.PHOT_PRODUCTION, tr.get('inputs.phot')),
           'forced_tier_pages.RUN_NAME': (sorted(name for _, _, name in TPM.run_names(t)),
                                          sorted(r for (S, c), r in runs.items() if CADENCE_TIER[c] != 'snapshot')),
           'forced_tier_pages.SITE_NAME': (t.SITE_NAME, fx['page_records']['tier pages']['site'])}
    for (ti, k), v in t.EVAL_DIRS.items():
        out['forced_tier_pages.EVAL_DIRS[%s,%s]' % (ti, k)] = (v, tr.get('inputs.evals.%s|%s' % (ti, k)))
    return out


def problems(fx):
    """Every disagreement of the config with the served files (the fixture), as 'key: got, want' lines (empty = none)."""
    from exopipe.phase3.lib.forced import tier_pages as TPM
    from exopipe.phase3.lib.forced import tier_snapshot as TS
    from exopipe.phase3.lib.forced import wide_jobs as WJ
    from exopipe.phase3.lib.photometry import steps_common as K
    ph, cc = CFG.photometry, CFG.photometry_campaign
    bad = []

    def want(key, got, exp):
        if got != exp:
            bad.append('%s: %r, the served files need %r' % (key, got, exp))
    want('photometry.CADENCE_TIER', ph.CADENCE_TIER, CADENCE_TIER)
    runs = served_runs(fx)
    for cad, tier in CADENCE_TIER.items():
        g = served_generation(fx, cad)
        want('photometry.GENERATIONS[%s]' % tier, ph.GENERATIONS.get(tier), g)
        want('photometry.CUTOUT_ROOTS[%s]' % tier, ph.CUTOUT_ROOTS.get(tier), cc.CUTOUT_ROOT_FMT.format(generation=g))
    gs = served_generation(fx, '10s')
    want('the 10-s and 1-min pages serve one generation', served_generation(fx, '1min'), gs)
    want('photometry.STAGE1_DIR', ph.STAGE1_DIR, cc.STAGE1_DIR_FMT.format(generation=gs))
    fs = CFG.forced_snapshot
    for run, lg in sorted(fx['run_logs'].items()):     # the 10-s / 1-min runs were built from these (FT-START)
        f = lg['fields']
        for key, got in (('EVAL_SB', f['eval_sb']), ('EVAL_WB', f['eval_wb']), ('PHOT_ROOT', f['P3_PHOT_ROOT']),
                         ('STAGE1_ROOT', f['P3_FORCED_STAGE1'])):
            want('forced_snapshot.%s (%s, %s:%d)' % (key, run, os.path.basename(lg['path']), lg['line']), getattr(fs, key), got)
    want('forced_snapshot.PHOT_ROOT = photometry.CUTOUT_ROOTS[snapshot]', fs.PHOT_ROOT, ph.CUTOUT_ROOTS['snapshot'])
    want('forced_snapshot.STAGE1_ROOT = <OUT_ROOT[snapshot]>/<STAGE1_DIR>', fs.STAGE1_ROOT,
         os.path.join(ph.OUT_ROOT['snapshot'], ph.STAGE1_DIR))
    roots = TS.generation_roots(gs)
    for k in TS.GENERATION_KEYS:
        want('forced_snapshot.' + k, getattr(fs, k), roots[k])
    tr = fx['page_records']['tier pages']['fields']
    for (ti, k) in sorted(CFG.forced_tier_pages.EVAL_DIRS):
        want('tier_pages.eval_name(%s, %s) of the served samples' % (ti, k),
             TPM.eval_name(ti, k, CFG.forced_tier_pages.SCRIPT_SAMPLES),
             os.path.basename(tr.get('inputs.evals.%s|%s' % (ti, k)) or ''))
    for key, (got, exp) in sorted(served_inputs(fx).items()):
        want(key + ' (the served page\'s input)', got, exp)
    want('photometry.CUTOUT_ROOTS: deep and sublst one root (the tier cutouts)', ph.CUTOUT_ROOTS['sublst'], ph.CUTOUT_ROOTS['deep'])
    w = CFG.forced_wide
    for S, cad in w.AMBIGUOUS_RUNS:
        want('forced_wide run (%s %s)' % (S, cad), WJ.run_name(S, cad), runs.get((S, cad)))
    for S, cad in w.RUNS:
        if CADENCE_TIER.get(cad) != 'snapshot':
            bad.append('forced_wide.RUNS (%s %s): not a snapshot cadence' % (S, cad))
    want('forced_wide.WORK_ROOT', os.path.basename(w.WORK_ROOT), 'forced_wide_' + gs)
    ws = CFG.forced_wide_search
    want('forced_wide_search.MINUTE_RULE (None: the inputs\' rule, W-MINUTE)', ws.MINUTE_RULE, None)
    for g, mb in sorted(fx['stage1_minute_basis'].items()):
        want('forced_wide_search.MINUTE_RULE_OF_GENERATION[%s] = the rule of its stage-1 minute_basis %r (%s)'
             % (g, mb['minute_basis'] or None, mb['path']), ws.MINUTE_RULE_OF_GENERATION.get(g),
             ws.MINUTE_BASIS_RULE.get(mb['minute_basis']))
        want('the stage-1 npz of %s is under <OUT_ROOT[snapshot]>/<STAGE1_DIR_FMT>' % g,
             os.path.dirname(os.path.dirname(mb['path'])),
             os.path.join(ph.OUT_ROOT['snapshot'], cc.STAGE1_DIR_FMT.format(generation=g)))
    n = CFG.notify
    want('notify.FORCED_TIERS cadences', sorted(set(n.FORCED_TIERS.values())), sorted(CADENCE_TIER))
    for (S, cad), run in sorted(runs.items()):
        want('notify.FORCED_RUN (%s %s)' % (S, cad), n.FORCED_RUN.format(
            stokes=S, tier=cad, generation=ph.GENERATIONS[ph.CADENCE_TIER[cad]]), run)
    want('photometry.PLANETS_PAGE_GENERATION', ph.PLANETS_PAGE_GENERATION,
         gen_of_site(os.path.basename(fx['registry']['planets']['root'])))
    want('the served-root gate holds the planets page\'s per-night inputs',
         {r for _, r in K.served_roots()} >= {v for w_, v in K.planets_roots().items() if w_ != 'planets page tier forced root'},
         True)
    der = {'%s.%s' % (nm, k) for nm, _, k, _, _ in CFG.derived_keys()}
    loose = sorted(k for k in gen_keys() if _base(k) not in der and _base(k) not in SEPARATE)
    if loose:
        bad.append('keys naming a generation, neither DERIVED from photometry.GENERATIONS nor SEPARATE: %s' % ', '.join(loose))
    for nm, sec, k, _, fn in CFG.derived_keys():
        want('%s.%s = its derivation' % (nm, k), getattr(sec, k), fn())
    unlisted = sorted(set(gen_keys(_GEN5)) - set(GEN5_SERVED) - set(GEN5_STEP))
    if unlisted:
        bad.append('keys naming gen5 in neither GEN5_SERVED nor GEN5_STEP: %s' % ', '.join(unlisted))
    return bad


def reader_problems(entries):
    """Entries ({key: (why, 'file:line', name)}) whose named line does not read the key's name."""
    bad = []
    for key, (_, where, name) in sorted(entries.items()):
        f, ln = where.rsplit(':', 1)
        try:
            with open(os.path.join(REPO, f)) as fh:
                lines = fh.read().splitlines()
            line = lines[int(ln) - 1]
        except (OSError, IndexError, ValueError) as e:
            bad.append('%s: reader %s unreadable (%s)' % (key, where, e))
            continue
        if name not in line or name not in key:
            bad.append('%s: its reader %s does not read %s (%r)' % (key, where, name, line.strip()[:120]))
    return bad


# ---------------------------------------------------------------------------------------------------------------------
# the tests
# ---------------------------------------------------------------------------------------------------------------------

@case
def test_page_table():
    fx = load_fixture()
    pages = {v['port']: (v['stokes'], v['kind']) for (_, _), v in CFG.serve.PAGES.items()
             if v.get('family') in ('forced', 'forced-tier')}
    reg = {int(p): FORCED_RUN_RE.match(os.path.basename(v['run'])).groups()[:2] for p, v in fx['registry']['forced'].items()}
    check('serve.PAGES: the forced and forced-tier pages are the registry\'s %d ports, each with the Stokes and cadence its run '
          'names (fixture of %s, %s)' % (len(reg), fx['registry']['path'], fx['registry']['mtime_utc']), pages == reg,
          (pages, reg))
    check('the registry serves one generation per cadence (gen6 for 10 s and 1 min, gen5 for 10 minutes and deep)',
          [served_generation(fx, c) for c in ('10s', '1min', '10min', 'deep')] == ['gen6', 'gen6', 'gen5', 'gen5'])


@case
def test_served():
    fx = load_fixture()
    bad = problems(fx)
    check('every tier\'s roots, run names and generation keys follow the served files (the fixture)', not bad, bad)
    ph = CFG.photometry
    check('the snapshot tier is gen6: photometry_gen6, stage1_gen6, eval_gen6, minute rule midpoint (its stage-1 minute_basis)',
          (ph.GENERATIONS['snapshot'], os.path.basename(ph.CUTOUT_ROOTS['snapshot']), ph.STAGE1_DIR,
           os.path.basename(CFG.forced_snapshot.EVAL_SB), CFG.forced_wide_search.MINUTE_BASIS_RULE.get(
               fx['stage1_minute_basis']['gen6']['minute_basis'])) == ('gen6', 'photometry_gen6', 'stage1_gen6', 'eval_gen6',
                                                                        'midpoint'))
    check('the deep and 10-minute tiers are gen5 (the tier cutouts); gen5\'s stage 1 has no minute_basis: filename',
          [os.path.basename(ph.CUTOUT_ROOTS[t]) for t in ('deep', 'sublst')] == ['photometry_gen5'] * 2
          and fx['stage1_minute_basis']['gen5']['minute_basis'] == '')
    src = open(os.path.join(REPO, 'exopipe', 'phase3', 'config', 'steps', 'photometry.py')).read().splitlines()
    line = [ln for ln in src if ln.strip().startswith('PLANETS_PAGE_GENERATION =')]
    check('the planets line says "switch at PLANETS-GEN6-DONE"', len(line) == 1 and 'switch at PLANETS-GEN6-DONE' in line[0],
          line)


@case
def test_gen5_readers():
    g5 = gen_keys(_GEN5)
    both = sorted(set(GEN5_SERVED) & set(GEN5_STEP))
    check('GEN5_SERVED and GEN5_STEP are disjoint', not both, both)
    stale = sorted(k for k in list(GEN5_SERVED) + list(GEN5_STEP) if k not in g5)
    check('every GEN5_SERVED / GEN5_STEP entry names gen5 in the config (a key switched leaves the list)', not stale, stale)
    bad = reader_problems(dict(GEN5_SERVED, **GEN5_STEP))
    check('every entry\'s named reader (file:line) reads it: %d served, %d production-step entries'
          % (len(GEN5_SERVED), len(GEN5_STEP)), not bad, bad)
    sep = sorted(k for k in SEPARATE if not any(_base(g) == k for g in gen_keys()))
    check('every SEPARATE key names a generation (no stale justification)', not sep, sep)
    der = {'%s.%s' % (nm, k) for nm, _, k, _, _ in CFG.derived_keys()}
    check('SEPARATE and DERIVED are disjoint', not (der & set(SEPARATE)), sorted(der & set(SEPARATE)))


@case
def test_one_key_switch():
    """photometry.GENERATIONS[snapshot] switches every snapshot root and run name; PLANETS_PAGE_GENERATION the planets
    page's per-night inputs and the served-root gate, not its tier root."""
    from exopipe.phase3.lib.forced import wide_jobs as WJ
    from exopipe.phase3.lib.photometry import steps_common as K
    ph = CFG.photometry
    tier_root = ph.TIER_FORCED_ROOTS[ph.PLANET_SAMPLE]
    try:
        CFG.set('photometry.GENERATIONS[snapshot]', 'gen7', source='test')
        fs, n = CFG.forced_snapshot, CFG.notify
        got = (os.path.basename(ph.CUTOUT_ROOTS['snapshot']), ph.STAGE1_DIR, os.path.basename(fs.EVAL_SB),
               os.path.basename(fs.EVAL_WB), os.path.basename(fs.ZONE_FLAGS), os.path.basename(fs.STAGE1_ROOT),
               os.path.basename(fs.PHOT_ROOT), os.path.basename(CFG.forced_wide.WORK_ROOT), WJ.run_name('I', '10s'),
               n.FORCED_RUN.format(stokes='V', tier='1min', generation=ph.GENERATIONS[ph.CADENCE_TIER['1min']]),
               WJ.run_name('I', 'deep'), os.path.basename(ph.CUTOUT_ROOTS['deep']))
        rec = sorted({o[0] for o in CFG.overrides if str(o[3]).startswith('derived from ')})
        check('one --set photometry.GENERATIONS[snapshot]=gen7 moves every snapshot root and run name (each recorded as an '
              'override "derived from photometry.GENERATIONS"), and no tier one: %s' % (got,),
              got == ('photometry_gen7', 'stage1_gen7', 'eval_gen7', 'eval_wb_gen7', 'zones_recomputed_gen7.csv', 'stage1_gen7',
                      'photometry_gen7', 'forced_wide_gen7', 'Run_SNAPFORCED_I10s_gen7', 'Run_SNAPFORCED_V1min_gen7',
                      'Run_SNAPFORCED_Ideep_gen5', 'photometry_gen5')
              and 'forced_snapshot.EVAL_SB' in rec and 'photometry.CUTOUT_ROOTS' in rec, (got, rec))
    finally:
        CFG.reset()
    try:
        CFG.set('forced_snapshot.EVAL_SB', '/x/eval_mine', source='test')
        CFG.set('photometry.GENERATIONS[snapshot]', 'gen7', source='test')
        check('... a derived key the run set itself keeps its value (EVAL_SB), the others follow (EVAL_WB)',
              CFG.forced_snapshot.EVAL_SB == '/x/eval_mine' and CFG.forced_snapshot.EVAL_WB.endswith('eval_wb_gen7'))
    finally:
        CFG.reset()
    check('... reset restores the served values', os.path.basename(CFG.forced_snapshot.EVAL_SB) == 'eval_gen6'
          and ph.STAGE1_DIR == 'stage1_gen6')
    before = K.planets_roots()
    try:
        CFG.set('photometry.PLANETS_PAGE_GENERATION', 'gen6', source='test')
        after = K.planets_roots()
        served = {r for _, r in K.served_roots()}
        check('PLANETS-GEN6-DONE is one key: PLANETS_PAGE_GENERATION=gen6 moves the planets page\'s cutouts to '
              'photometry_gen6/SolarSystem and its stage 1 to stage1_gen6_planets_orbit, both in the served-root gate; its tier '
              'forced root stays %s (planets_gen6_stage34_20261127.sh:66-67)' % os.path.basename(tier_root),
              after['planets page cutouts'] == os.path.join(CFG.photometry_campaign.CUTOUT_ROOT_FMT.format(generation='gen6'),
                                                            'SolarSystem')
              and os.path.basename(after['planets page stage 1']) == 'stage1_gen6_planets_orbit'
              and after['planets page tier forced root'] == tier_root == before['planets page tier forced root']
              and {after['planets page cutouts'], after['planets page stage 1']} <= served
              and before['planets page stage 1'] not in served, (before, after))
        hit = K.served_root_of(os.path.join(after['planets page stage 1'], '18h_2025-04-11'))
        check('... the gate names a path under the new planets stage 1 as served (C-INPLACE / PHOT-ADHOC protect it)',
              hit is not None and hit[1] == after['planets page stage 1'], hit)
    finally:
        CFG.reset()


def falsify(fx, label, sets, expect, patch=None):
    """Each override in ``sets`` (dotted key -> value) - or ``patch`` applied to a copy of the fixture - makes problems()
    name ``expect``."""
    import copy
    try:
        for k, v in sets.items():
            CFG.set(k, v, source='test')
        f2 = copy.deepcopy(fx)
        if patch:
            patch(f2)
        bad = problems(f2)
        check('falsification: %s fails, named' % label, any(expect in b for b in bad), bad)
    finally:
        CFG.reset()


@case
def test_falsification():
    fx = load_fixture()
    g5 = CFG.photometry_campaign.CUTOUT_ROOT_FMT.format(generation='gen5')
    g6 = CFG.photometry_campaign.CUTOUT_ROOT_FMT.format(generation='gen6')
    falsify(fx, 'the snapshot cutout root alone back to gen5', {'photometry.CUTOUT_ROOTS[snapshot]': g5},
            'photometry.CUTOUT_ROOTS[snapshot]')
    falsify(fx, 'the snapshot generation back to gen5', {'photometry.GENERATIONS[snapshot]': 'gen5'},
            'photometry.GENERATIONS[snapshot]')
    falsify(fx, 'the deep cutout root at gen6 (which holds no tier cutouts)', {'photometry.CUTOUT_ROOTS[deep]': g6},
            'photometry.CUTOUT_ROOTS[deep]')
    falsify(fx, 'stage 1 back to stage1_gen5', {'photometry.STAGE1_DIR': 'stage1_gen5'}, 'photometry.STAGE1_DIR')
    falsify(fx, 'a forced_snapshot key of gen5 (not the build log\'s)',
            {'forced_snapshot.EVAL_WB': '/lustre/gh/main/phase3/snapshot_forced/eval_wb_gen5'}, 'forced_snapshot.EVAL_WB')
    falsify(fx, 'a 10-s build log naming another stage 1', {}, 'forced_snapshot.STAGE1_ROOT',
            lambda f: f['run_logs']['Run_SNAPFORCED_I10s_gen6']['fields'].update(P3_FORCED_STAGE1='/x/stage1_gen5'))
    falsify(fx, 'the wide-area minute rule set to gen5\'s', {'forced_wide_search.MINUTE_RULE': 'filename'},
            'forced_wide_search.MINUTE_RULE')
    falsify(fx, 'a gen6 stage 1 without minute_basis (its rule would be filename)', {},
            'MINUTE_RULE_OF_GENERATION[gen6]', lambda f: f['stage1_minute_basis']['gen6'].update(minute_basis=''))
    falsify(fx, 'the tier pages\' evaluations of another generation', {}, 'tier_pages.eval_name(deep, sb)',
            lambda f: f['page_records']['tier pages']['fields'].update(
                {'inputs.evals.deep|sb': '/lustre/gh/main/phase3/snapshot_forced/eval_deep_gen4_hwj10pc'}))
    falsify(fx, 'the 10-minute pages served from gen6 runs', {}, 'photometry.GENERATIONS[sublst]',
            lambda f: [v.update(run=v['run'].replace('10min_gen5', '10min_gen6')) for v in f['registry']['forced'].values()])
    falsify(fx, 'the planets page switched before PLANETS-GEN6-DONE', {'photometry.PLANETS_PAGE_GENERATION': 'gen6'},
            'photometry.PLANETS_PAGE_GENERATION')
    falsify(fx, 'the planets page record naming another tier root', {}, 'photometry.TIER_FORCED_ROOTS[SolarSystem]',
            lambda f: f['page_records']['planets']['fields'].update({'inputs.tiers': '/x/tiers_gen6_planets_orbit'}))
    falsify(fx, 'an unlisted key naming gen5', {'forced_snapshot.REVIEW_SUBDIR_UNDER_OUT': 'review_gen5'},
            'forced_snapshot.REVIEW_SUBDIR_UNDER_OUT')
    falsify(fx, 'an unlisted key naming gen6 (neither DERIVED nor SEPARATE)', {'forced_snapshot.REVIEW_SUBDIR_UNDER_OUT':
                                                                             'review_gen6'}, 'nor SEPARATE: forced_snapshot.REVIEW_SUBDIR_UNDER_OUT')
    bad = reader_problems({'photometry_fluxscale.GEN5_ROOT': ('x', SCU + ':1', 'GEN5_ROOT')})
    check('falsification: an entry whose named line does not read it fails, named', bad and 'does not read' in bad[0], bad)
    check('falsification: the overrides are undone', not problems(fx), problems(fx))


PROVENANCE = ('sha256', 'bytes', 'mtime_utc', 'line')   # a change of these alone is a note, not a failure


def live_diff(fx, live):
    """([(field path, fixture, live)] of the recorded fields that differ, [the same of provenance fields only])."""
    diff = []

    def cmp(path, a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                cmp(path + [k], a.get(k), b.get(k))
        elif a != b:
            diff.append(('/'.join(map(str, path)), a, b))
    cmp([], {k: fx.get(k) for k in live}, live)
    hard = [d for d in diff if d[0].rsplit('/', 1)[-1] not in PROVENANCE]
    return hard, [d for d in diff if d not in hard]


@case
def test_live():
    """The served files read again (where this node has them) equal the fixture, field for field."""
    import copy
    if not os.path.exists(REGISTRY):
        print('UNTESTED the live comparison (%s absent here): the config was compared with the fixture only' % REGISTRY)
        return
    fx, live = load_fixture(), read_served()
    hard, notes = live_diff(fx, live)
    for d in notes:
        print('NOTE  %s: fixture %r, live %r (provenance; the recorded fields are compared)' % d)
    check('the served files read now equal the fixture in every recorded field (%s; %d page records, %d build logs, %d '
          'stage-1 npz) - a difference names the file and field; --write-fixture refreshes it'
          % (REGISTRY, len(live['page_records']), len(live['run_logs']), len(live['stage1_minute_basis'])), not hard,
          ['%s: fixture %r, live %r' % d for d in hard])
    f2 = copy.deepcopy(fx)
    f2['page_records']['tier pages']['fields']['inputs.phot'] = '/lustre/pipeline/exopipe/phase3/photometry_gen6'
    f2['stage1_minute_basis']['gen6']['minute_basis'] = ''
    h2, _ = live_diff(f2, live)
    check('falsification: a fixture differing from the live files in a recorded field fails, naming each field',
          sorted(d[0] for d in h2) == ['page_records/tier pages/fields/inputs.phot', 'stage1_minute_basis/gen6/minute_basis'],
          h2)
    f3 = copy.deepcopy(fx)
    f3['registry']['sha256'] = '0' * 64
    h3, n3 = live_diff(f3, live)
    check('... a registry whose bytes changed with every recorded field equal is a note, not a failure',
          not h3 and [d[0] for d in n3 if d[0] == 'registry/sha256'] == ['registry/sha256'], (h3, n3))


def main():
    if '--write-fixture' in sys.argv[1:]:
        write_fixture()
        return 0
    for t in (test_page_table, test_served, test_gen5_readers, test_one_key_switch, test_falsification, test_live):
        try:
            t()
        except AssertionError:
            pass                                       # printed by check(); counted in FAILED
    print('%s: %d check(s) failed' % ('FAILED' if FAILED else 'all checks passed', len(FAILED)))
    return 1 if FAILED else 0


if __name__ == '__main__':
    sys.exit(main())
