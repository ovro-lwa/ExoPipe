#!/usr/bin/env python
"""Every tier's roots name the cutout generation the SERVED pages read, checked against a fixture of the served files.

The fixture exopipe/phase3/config/served_generations.json (photometry.SERVED_GENERATIONS_FILE: production's served-root gate
SERVED-GEN reads it too, steps_common.gate_served_generations) is written by `--write-fixture` from the files named below (no
other file is read, nothing recursive, no memory map): the hub's page settings serve/registered_pages.json (sha256 and modification time
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
  - the planets page: PLANETS_PAGE_GENERATION (derived: GENERATIONS[snapshot], since PLANETS-GEN6-DONE) is the generation
    its page record's per-night inputs name (not the site's name: site_planets_SUN_gen5 holds the gen6 page), and its
    per-night inputs (steps_common.planets_roots: SolarSystem cutouts and stage 1), its tier cubes (CUTOUT_ROOTS[deep]) and
    its tier forced root are the page record's; a snapshot switch moves the per-night inputs and the served-root gate, not
    the tier roots (planets_gen6_stage34_20261127.sh:66-67), and refuses SERVED-GEN while the page has not followed;
  - every config key naming a generation is either derived (config.Derived, computed when read; photometry.GENERATIONS among
    its inputs) or listed in SEPARATE with the reason it cannot follow it; every input a derived key lists changes it (the
    inputs table, INPUT_CHANGE); a derived key is not settable; photometry.GENERATIONS is protected and ruled;
  - SERVED-GEN: the served-root gate refuses a config whose generations disagree with the fixture (parity replay exempt);
  - every key naming gen5 is in GEN5_SERVED (read by a served page) or GEN5_STEP (read by a production step, not served),
    each with its reader's file:line, and that line reads the key; a served entry's value is the served page's input;
  - falsification: each kind of disagreement fails, named.
On a node with the served files: they are read again and compared with the fixture - any difference in a recorded field
fails, named (a registry whose bytes changed with identical fields is a note; `--write-fixture` refreshes the fixture).
Run: python tests/test_served_generation_roots.py [--write-fixture]   (pytest: each test_* fails on a failed check)
"""
import contextlib
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

FIXTURE = os.path.join(REPO, 'exopipe', 'phase3', 'config', 'served_generations.json')   # photometry.SERVED_GENERATIONS_FILE
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
    # (photometry.PLANETS_PAGE_GENERATION was here - 'the planets page serves gen5 per-night inputs while the snapshot pages
    # serve gen6' - until PLANETS-GEN6-DONE, 2026-11-28 05:29:23Z; since then it is DERIVED from GENERATIONS['snapshot'])
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
    'photometry.CUTOUT_ROOTS[deep]': ('p9122 / p9123: the tier cutouts (page record inputs.phot); /s/planets/: its tier cubes '
                                      '(inputs.phot_tiers)', TP + ':175', 'CUTOUT_ROOTS'),
    'photometry.CUTOUT_ROOTS[sublst]': ('p9120 / p9121: the tier cutouts (inputs.phot)', TP + ':175', 'CUTOUT_ROOTS'),
    'photometry.TIER_FORCED_ROOTS[HWJ]': ('p9120-p9123: forced_tiers_gen5 (inputs.tiers)', TP + ':182', 'TIER_FORCED_ROOTS'),
    'photometry.TIER_FORCED_ROOTS[10pc]': ('p9120-p9123: forced_tiers_gen5 (inputs.tiers)', TP + ':182', 'TIER_FORCED_ROOTS'),
    'photometry.TIER_FORCED_ROOTS[SolarSystem]': ('/s/planets/: its 10-minute and deep inputs (inputs.tiers; the served-root '
                                                  'gate holds it)', SC + ':340', 'TIER_FORCED_ROOTS'),
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
    'forced_wide_search.MINUTE_RULE_OF_GENERATION{gen5}': ('W-MINUTE: the run\'s rule of a wide-area search of gen5 cutouts '
                                                           '(each unit checks its cells\' stage-1 minute_basis against it)',
                                                           'exopipe/phase3/lib/forced/wide_stages.py:385',
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
                  | ({planets_generation(recs['planets']['fields'])} - {None}))
    basis = {}
    for g in gens:
        p = STAGE1_NPZ.format(generation=g)
        with np.load(p, allow_pickle=False) as z:
            basis[g] = {'path': p, 'minute_basis': str(z['minute_basis']) if 'minute_basis' in z.files else ''}
    return {'registry': reg, 'page_records': recs, 'run_logs': logs, 'stage1_minute_basis': basis}


def planets_generation(fields):
    """The generation of the planets page's per-night inputs as its page record names them (``fields``: the fixture's
    inputs.* of the planets record): stage 1 <OUT_ROOT[snapshot]>/<photometry.PLANETS_STAGE1_FMT> and cutouts
    <photometry_campaign.CUTOUT_ROOT_FMT>; None unless both name one and the same. Not the site's name: the gen6 page was
    swapped into site_planets_SUN_gen5 (planets_gen6_stage34_20261127.sh:64-77)."""
    ph, cc = CFG.photometry, CFG.photometry_campaign
    gens = set()
    for k, fmt in (('inputs.stage1', os.path.join(ph.OUT_ROOT['snapshot'], ph.PLANETS_STAGE1_FMT)),
                   ('inputs.phot', cc.CUTOUT_ROOT_FMT)):
        pre, _, post = fmt.partition('{generation}')
        m = re.fullmatch(re.escape(pre) + r'(gen\d+)' + re.escape(post), str(fields.get(k) or '').rstrip('/'))
        gens.add(m.group(1) if m else None)
    return gens.pop() if len(gens) == 1 else None


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
    """{'section.KEY[entry]' (or '{key}' for a dict key): value} of every config value matching ``rx``. A path into this
    checkout (package data, e.g. masters_build.NAMED_OUT_OF_MASTER) is matched without the checkout's own location, which
    may name a generation itself (a worktree .../planets_gen6_20261128/wt)."""
    out = {}

    def walk(v, path):
        if isinstance(v, str):
            if rx.search(v[len(REPO):] if v.startswith(REPO + os.sep) else v):
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
    tr = fx['page_records']['tier pages']['fields']
    pr = fx['page_records']['planets']['fields']
    runs = served_runs(fx)
    out = {'photometry.GENERATIONS[deep]': (ph.GENERATIONS['deep'], served_generation(fx, 'deep')),
           'photometry.GENERATIONS[sublst]': (ph.GENERATIONS['sublst'], served_generation(fx, '10min')),
           'photometry.CUTOUT_ROOTS[deep]': (ph.CUTOUT_ROOTS['deep'], tr.get('inputs.phot')),
           'photometry.CUTOUT_ROOTS[sublst]': (ph.CUTOUT_ROOTS['sublst'], tr.get('inputs.phot')),
           'photometry.TIER_FORCED_ROOTS[HWJ]': (ph.TIER_FORCED_ROOTS['HWJ'], tr.get('inputs.tiers')),
           'photometry.TIER_FORCED_ROOTS[10pc]': (ph.TIER_FORCED_ROOTS['10pc'], tr.get('inputs.tiers')),
           'photometry.TIER_FORCED_ROOTS[SolarSystem]': (ph.TIER_FORCED_ROOTS[ph.PLANET_SAMPLE], pr.get('inputs.tiers')),
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
        want('photometry.CUTOUT_ROOTS[snapshot] (%s: P3_PHOT_ROOT)' % run, ph.CUTOUT_ROOTS['snapshot'], f['P3_PHOT_ROOT'])
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
    pr = fx['page_records']['planets']['fields']
    want('photometry.PLANETS_PAGE_GENERATION (= GENERATIONS[snapshot]; the planets page record\'s inputs.stage1 and '
         'inputs.phot name one generation)', ph.PLANETS_PAGE_GENERATION, planets_generation(pr))
    pl = K.planets_roots()
    want('the planets page\'s per-night stage 1 (steps_common.planets_roots; inputs.stage1)', pl['planets page stage 1'],
         pr.get('inputs.stage1'))
    want('the planets page\'s per-night cutouts\' generation root (steps_common.planets_roots; inputs.phot)',
         os.path.dirname(pl['planets page cutouts']), pr.get('inputs.phot'))
    want('the planets page\'s tier cubes = photometry.CUTOUT_ROOTS[deep] (inputs.phot_tiers)', ph.CUTOUT_ROOTS['deep'],
         pr.get('inputs.phot_tiers'))
    try:
        want('the served-root gate holds the planets page\'s per-night inputs',
             {r for _, r in K.served_roots()} >= set(K.planets_roots().values()), True)
    except K.Refusal as e:
        bad.append('the served-root gate refuses: %s' % e)
    der = {'%s.%s' % (nm, k) for nm, _, k, d in CFG.derived_keys() if 'photometry.GENERATIONS' in d.inputs}
    loose = sorted(k for k in gen_keys() if _base(k) not in der and _base(k) not in SEPARATE)
    if loose:
        bad.append('keys naming a generation, neither DERIVED from photometry.GENERATIONS nor SEPARATE: %s' % ', '.join(loose))
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
    from exopipe.phase3.lib.photometry import steps_common as K
    fx = load_fixture()
    bad = problems(fx)
    check('every tier\'s roots, run names and generation keys follow the served files (the fixture)', not bad, bad)
    check('the fixture is the one production\'s served-root gate reads (photometry.SERVED_GENERATIONS_FILE, protected)',
          os.path.realpath(K.served_fixture_path()) == os.path.realpath(FIXTURE)
          and CFG.protected('photometry.SERVED_GENERATIONS_FILE'), K.served_fixture_path())
    ph = CFG.photometry
    check('the snapshot tier is gen6: photometry_gen6, stage1_gen6, eval_gen6, minute rule midpoint (its stage-1 minute_basis)',
          (ph.GENERATIONS['snapshot'], os.path.basename(ph.CUTOUT_ROOTS['snapshot']), ph.STAGE1_DIR,
           os.path.basename(CFG.forced_snapshot.EVAL_SB), CFG.forced_wide_search.MINUTE_BASIS_RULE.get(
               fx['stage1_minute_basis']['gen6']['minute_basis'])) == ('gen6', 'photometry_gen6', 'stage1_gen6', 'eval_gen6',
                                                                        'midpoint'))
    check('the deep and 10-minute tiers are gen5 (the tier cutouts); the fixture records one stage-1 minute basis, gen6\'s: '
          'no served page reads a gen5 stage 1 since PLANETS-GEN6-DONE (the planets page reads stage1_gen6_planets_orbit)',
          [os.path.basename(ph.CUTOUT_ROOTS[t]) for t in ('deep', 'sublst')] == ['photometry_gen5'] * 2
          and sorted(fx['stage1_minute_basis']) == ['gen6'], sorted(fx['stage1_minute_basis']))
    src = open(os.path.join(REPO, 'exopipe', 'phase3', 'config', 'steps', 'photometry.py')).read().splitlines()
    line = [ln for ln in src if ln.strip().startswith('PLANETS_PAGE_GENERATION =')]
    d = CFG.derived('photometry.PLANETS_PAGE_GENERATION')
    check('PLANETS-GEN6-DONE: the planets page follows the snapshot generation - PLANETS_PAGE_GENERATION is one line, Derived '
          'from photometry.GENERATIONS alone (= GENERATIONS[snapshot] = gen6); its tier forced root stays '
          'tiers_<GENERATIONS[deep]>_planets_orbit (gen5)', len(line) == 1 and 'Derived((\'photometry.GENERATIONS\',)' in line[0]
          and d is not None and d.inputs == ('photometry.GENERATIONS',)
          and ph.PLANETS_PAGE_GENERATION == ph.GENERATIONS['snapshot'] == 'gen6'
          and os.path.basename(ph.TIER_FORCED_ROOTS[ph.PLANET_SAMPLE]) == 'tiers_gen5_planets_orbit', (line, d and d.inputs))


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
    der = {'%s.%s' % (nm, k) for nm, _, k, _ in CFG.derived_keys()}
    check('SEPARATE and DERIVED are disjoint', not (der & set(SEPARATE)), sorted(der & set(SEPARATE)))


@contextlib.contextmanager
def raw(*triples):
    """(section, KEY, value) set on the section class itself, bypassing CFG.set (a protected input such as photometry.OUT_ROOT,
    a test fixture path), restored as it was (vars: a Derived key's descriptor, not its value)."""
    old = [(sec, k, vars(sec)[k]) for sec, k, _ in triples]
    try:
        for sec, k, v in triples:
            setattr(sec, k, v)
        yield
    finally:
        for sec, k, v in old:
            setattr(sec, k, v)


@contextlib.contextmanager
def parity():
    """As a --parity replay into a non-production --out (cli.py: CFG.parity_replay): a protected key may be overridden."""
    CFG.parity_replay = True
    try:
        yield
    finally:
        CFG.reset()
        CFG.parity_replay = False


def fixture_copy(edit):
    """The served registry's fixture with ``edit`` applied, written to a scratch file: its path."""
    import copy
    import tempfile
    fx = copy.deepcopy(load_fixture())
    edit(fx)
    fd, path = tempfile.mkstemp(prefix='served_generations_', suffix='.json', dir=os.environ.get('EXOPIPE3_TEST_SCRATCH'))
    with os.fdopen(fd, 'w') as f:
        json.dump(fx, f)
    return path


def planets_record_of(gen):
    """A fixture edit (for fixture_copy / falsify): the planets page record's per-night inputs of generation ``gen`` -
    inputs.stage1 <OUT_ROOT[snapshot]>/<PLANETS_STAGE1_FMT>, inputs.phot <CUTOUT_ROOT_FMT> (gen5: the record before
    PLANETS-GEN6-DONE); its tier inputs unchanged."""
    def edit(f):
        ph = CFG.photometry
        f['page_records']['planets']['fields'].update({
            'inputs.stage1': os.path.join(ph.OUT_ROOT['snapshot'], ph.PLANETS_STAGE1_FMT.format(generation=gen)),
            'inputs.phot': CFG.photometry_campaign.CUTOUT_ROOT_FMT.format(generation=gen)})
    return edit


def refusal(fn, *a, **k):
    from exopipe.phase3.config import ConfigRefusal
    from exopipe.phase3.core.record import Refusal
    try:
        fn(*a, **k)
    except (Refusal, ConfigRefusal) as e:
        return str(e)
    return None


@case
def test_one_key_switch():
    """photometry.GENERATIONS[snapshot] switches every snapshot root and run name and the planets page's per-night inputs
    (a protected key: a --parity replay here), not the planets page's tier root (GENERATIONS[deep]); PLANETS_PAGE_GENERATION
    is derived from it, so a snapshot switch the planets page has not followed refuses SERVED-GEN."""
    from exopipe.phase3.lib.forced import wide_jobs as WJ
    from exopipe.phase3.lib.photometry import steps_common as K
    ph = CFG.photometry
    tier_root = ph.TIER_FORCED_ROOTS[ph.PLANET_SAMPLE]
    e = refusal(CFG.set, 'photometry.GENERATIONS[snapshot]', 'gen7', source='test')
    check('photometry.GENERATIONS is PROTECTED and RULED (%s): a --set refuses outside a --parity replay, whatever '
          '--ruling-for says (protected() is checked before the ruling)' % ', '.join(CFG.ruled('photometry.GENERATIONS')),
          e is not None and 'PROTECTED' in e and CFG.protected('photometry.GENERATIONS')
          and CFG.ruled('photometry.GENERATIONS') == ('R-GREGG-20261118-GEN6-WITH-LPTS-SAMPLE-WIRED',), e)
    with parity():
        CFG.set('photometry.GENERATIONS[snapshot]', 'gen7', source='test')
        fs, n = CFG.forced_snapshot, CFG.notify
        got = (os.path.basename(ph.CUTOUT_ROOTS['snapshot']), ph.STAGE1_DIR, os.path.basename(fs.EVAL_SB),
               os.path.basename(fs.EVAL_WB), os.path.basename(fs.ZONE_FLAGS), os.path.basename(fs.STAGE1_ROOT),
               os.path.basename(fs.PHOT_ROOT), os.path.basename(CFG.forced_wide.WORK_ROOT), WJ.run_name('I', '10s'),
               n.FORCED_RUN.format(stokes='V', tier='1min', generation=ph.GENERATIONS[ph.CADENCE_TIER['1min']]),
               WJ.run_name('I', 'deep'), os.path.basename(ph.CUTOUT_ROOTS['deep']), ph.PLANETS_PAGE_GENERATION,
               os.path.relpath(K.planets_roots()['planets page cutouts'], os.path.dirname(ph.CUTOUT_ROOTS['snapshot'])),
               os.path.basename(K.planets_roots()['planets page stage 1']),
               os.path.basename(ph.TIER_FORCED_ROOTS[ph.PLANET_SAMPLE]))
        rec = [(o[0], str(o[4]).split(':', 1)[0]) for o in CFG.overrides]
        check('one photometry.GENERATIONS[snapshot]=gen7 (parity replay) moves every snapshot root and run name and the '
              'planets page\'s per-night inputs, computed when read (the one override recorded is the input), and no tier '
              'one (the planets page\'s tier forced root stays): %s' % (got,),
              got == ('photometry_gen7', 'stage1_gen7', 'eval_gen7', 'eval_wb_gen7', 'zones_recomputed_gen7.csv', 'stage1_gen7',
                      'photometry_gen7', 'forced_wide_gen7', 'Run_SNAPFORCED_I10s_gen7', 'Run_SNAPFORCED_V1min_gen7',
                      'Run_SNAPFORCED_Ideep_gen5', 'photometry_gen5', 'gen7', 'photometry_gen7/SolarSystem',
                      'stage1_gen7_planets_orbit', 'tiers_gen5_planets_orbit')
              and rec == [('photometry.GENERATIONS[snapshot]', 'parity replay')], (got, rec))
        e = refusal(K.served_roots)
        check('... the served-root gate takes the parity replay (SERVED-GEN exempts it, the derived planets page key too: '
              'writes confined to --out)', e is None, e)
        e = refusal(CFG.set, 'forced_snapshot.EVAL_SB', '/x/eval_mine', source='test')
        check('... a derived key is not settable (EVAL_SB): refused, naming its inputs - no frozen value beside re-derived '
              'siblings', e is not None and 'derived' in e and 'photometry_evaluate.EVAL_DIR' in e
              and CFG.forced_snapshot.EVAL_SB.endswith('eval_gen7'), e)
    check('... reset restores the served values', os.path.basename(CFG.forced_snapshot.EVAL_SB) == 'eval_gen6'
          and ph.STAGE1_DIR == 'stage1_gen6' and ph.PLANETS_PAGE_GENERATION == 'gen6' and not CFG.overrides)
    pl = K.planets_roots()
    served = {r for _, r in K.served_roots()}
    g5 = K.planets_roots('gen5')
    e = refusal(CFG.set, 'photometry.PLANETS_PAGE_GENERATION', 'gen5', source='test')
    check('PLANETS-GEN6-DONE (2026-11-28 05:29:23Z): PLANETS_PAGE_GENERATION is derived (a --set refuses, naming '
          'photometry.GENERATIONS) = gen6; the planets page\'s cutouts photometry_gen6/SolarSystem and stage 1 '
          'stage1_gen6_planets_orbit are in the served-root gate and stage1_gen5_planets_orbit is not under any served root; '
          'its tier forced root stays %s (GENERATIONS[deep]; planets_gen6_stage34_20261127.sh:66-67)' % os.path.basename(tier_root),
          e is not None and 'derived' in e and 'photometry.GENERATIONS' in e and ph.PLANETS_PAGE_GENERATION == 'gen6'
          and pl['planets page cutouts'] == os.path.join(CFG.photometry_campaign.CUTOUT_ROOT_FMT.format(generation='gen6'),
                                                         'SolarSystem')
          and os.path.basename(pl['planets page stage 1']) == 'stage1_gen6_planets_orbit'
          and {pl['planets page cutouts'], pl['planets page stage 1']} <= served and tier_root in served
          and os.path.basename(tier_root) == 'tiers_gen5_planets_orbit'
          and K.served_root_of(os.path.join(g5['planets page stage 1'], '18h_2025-04-11')) is None, (e, pl))
    hit = K.served_root_of(os.path.join(pl['planets page stage 1'], '18h_2025-04-11'))
    check('... the gate names a path under the planets stage 1 as served (C-INPLACE / PHOT-ADHOC protect it)',
          hit is not None and hit[1] == pl['planets page stage 1'], hit)
    hit = K.served_root_of(os.path.join(g5['planets page cutouts'], 'Green'))
    check('... the gen5 SolarSystem per-night cutouts are no planets input, but they stay under a served root: photometry_gen5 '
          'holds the tier cutouts (CUTOUT_ROOTS[deep] / [sublst])', hit is not None
          and hit[1] == CFG.photometry.CUTOUT_ROOTS['deep'] and 'planets' not in hit[0], hit)
    old = fixture_copy(planets_record_of('gen5'))
    try:
        with raw((ph, 'SERVED_GENERATIONS_FILE', old)):
            e = refusal(K.served_roots)
            e2 = refusal(K.served_cutout_root, 'snapshot')
        check('a fixture whose planets page serves gen5 (its record before PLANETS-GEN6-DONE) against GENERATIONS[snapshot] gen6: '
              'the served-root gate refuses (SERVED-GEN), naming the derived key - a snapshot switch the page has not followed '
              'fails closed (served_cutout_root too)', e is not None and 'SERVED-GEN' in e
              and 'PLANETS_PAGE_GENERATION = gen6 (derived: GENERATIONS[snapshot]), the planets page serves gen5' in e
              and e2 is not None and 'SERVED-GEN' in e2, (e, e2))
    finally:
        os.remove(old)
        CFG.reset()


# every input a derived key may list -> a changed value of it (test_derived_inputs: each must change every derived key that
# lists it; an input missing here fails)
INPUT_CHANGE = {
    'photometry.GENERATIONS': lambda d: {t: 'gen7' for t in d},
    'forced_snapshot.GENERATION': lambda v: 'gen7',
    'photometry.OUT_ROOT': lambda d: {t: v + '_moved' for t, v in d.items()},
    'photometry_evaluate.EVAL_DIR': lambda d: {k: 'moved_' + v for k, v in d.items()},
    'photometry_evaluate.ZONE_FLAGS_FILE': lambda v: 'moved_' + v,
    'photometry_campaign.STAGE1_DIR_FMT': lambda v: 'moved_' + v,
    'photometry_campaign.CUTOUT_ROOT_FMT': lambda v: v + '_moved',
    'photometry.TIER_FORCED_ROOT_FMT': lambda v: v + '_moved',
    'photometry.TIER_FORCED_ROOT_SAMPLES': lambda v: tuple(v) + ('MOVED',),
    'photometry.PLANET_SAMPLE': lambda v: v + 'Moved',
    'photometry.PLANETS_TIERS_FMT': lambda v: v + '_moved',
    'forced_wide.WORK_ROOT_FMT': lambda v: v + '_moved',
}


@case
def test_derived_inputs():
    """The derived keys (config.Derived: computed when read from the inputs each declares - the only list, and the only
    values its function receives): every input changes every key that lists it; none is settable; an input's own gate
    (PROTECTED, RULED, VALIDATE, check_out) applies on its --set; and SERVED-GEN refuses a generation the fixture does not
    serve."""
    from exopipe.phase3.lib.photometry import steps_common as K
    table = {'%s.%s' % (nm, k): d.inputs for nm, _, k, d in CFG.derived_keys()}
    print('INFO the derived keys and their inputs:\n  ' + '\n  '.join('%s <- %s' % (k, ', '.join(v))
                                                                      for k, v in sorted(table.items())))
    missing = sorted({i for v in table.values() for i in v} - set(INPUT_CHANGE))
    check('every input of a derived key has a changed value in INPUT_CHANGE (%d derived keys, %d inputs)'
          % (len(table), len({i for v in table.values() for i in v})), not missing, missing)
    same = []
    for key, inputs in sorted(table.items()):
        for i in inputs:
            if i not in INPUT_CHANGE:
                continue
            before = CFG.get(key)
            sn, ik = i.split('.', 1)
            sec = CFG._section(sn)
            with raw((sec, ik, INPUT_CHANGE[i](getattr(sec, ik)))):
                after = CFG.get(key)
            if after == before or CFG.get(key) != before:
                same.append('%s did not change %s (%r)' % (i, key, after))
    check('each input override changes each derived key that lists it, and is undone (%d pairs)'
          % sum(len(v) for v in table.values()), not same, same)
    sets = []
    for key, inputs in sorted(table.items()):
        e = refusal(CFG.set, key + ('[snapshot]' if isinstance(CFG.get(key), dict) else ''), '/x', source='test')
        if not (e and 'derived' in e and all(i in e for i in inputs)):
            sets.append('%s: %s' % (key, e))
    CFG.reset()
    check('a derived key is not settable: --set refuses, naming every input to set instead (%d keys)' % len(table),
          not sets and not CFG.overrides, sets)
    with parity():
        e2 = refusal(CFG.set, 'photometry.GENERATIONS[snapshot]', 'six', source='test')
        g = CFG.photometry.GENERATIONS['snapshot']
    check('VALIDATE at --set: a generation that is not gen<N> refuses (GENERATIONS under parity; forced_snapshot.GENERATION '
          'below) and leaves the value (and the derived PLANETS_PAGE_GENERATION) as it was', e2 is not None
          and 'does not name a generation' in e2 and g == 'gen6' and CFG.photometry.PLANETS_PAGE_GENERATION == 'gen6', e2)
    import tempfile
    out = tempfile.mkdtemp(prefix='derived_out_', dir=os.environ.get('EXOPIPE3_TEST_SCRATCH'))
    try:
        e = refusal(CFG.set, 'forced_snapshot.GENERATION', 'gen5', source='test')
        CFG.authorize([], out=out)
        e4 = refusal(CFG.set, 'forced_snapshot.GENERATION', 'photometry_gen5', source='test')
        CFG.set('forced_snapshot.GENERATION', 'gen5', source='test')
        st1 = CFG.forced_snapshot.STAGE1_ROOT
        e3 = refusal(CFG.check_out, CFG.photometry.OUT_ROOT['snapshot'])
        check('a ruled input (forced_snapshot.GENERATION): refused into a production --out without --ruling-for; into a '
              'non-production --out a value that is not gen<N> refuses (VALIDATE), gen5 moves the derived roots (%s), and '
              'check_out refuses when the final --out is a production root' % os.path.basename(st1), e is not None
              and 'carries a ruling' in e and e4 is not None and 'does not name a generation' in e4
              and st1.endswith('stage1_gen5') and e3 is not None and 'production root' in e3, (e, e4, e3))
    finally:
        CFG.reset()
        CFG.authorize([])
        os.rmdir(out)
    e = refusal(K.gate_served_generations)
    check('SERVED-GEN: the config agrees with the fixture (snapshot gen6, deep and sublst gen5, planets page gen6)', e is None, e)
    g5 = fixture_copy(lambda f: [v.update(run=v['run'].replace('10s_gen6', '10s_gen5').replace('1min_gen6', '1min_gen5'))
                                 for v in f['registry']['forced'].values()])
    mixed = fixture_copy(lambda f: [v.update(run=v['run'].replace('V10s_gen6', 'V10s_gen5'))
                                    for v in f['registry']['forced'].values()])
    try:
        with raw((CFG.photometry, 'SERVED_GENERATIONS_FILE', g5)):
            e = refusal(K.gate_served_generations)
            e2 = refusal(K.served_cutout_root, 'snapshot')
            with parity():
                CFG.set('photometry.GENERATIONS[snapshot]', 'gen5', source='test')
                e3 = refusal(K.gate_served_generations)
        check('SERVED-GEN: a fixture serving gen5 snapshot pages against GENERATIONS[snapshot] gen6 refuses, naming both '
              '(served_cutout_root too); a parity replay of GENERATIONS[snapshot]=gen5 passes', e is not None
              and 'GENERATIONS[snapshot] = gen6, the registry serves gen5' in e and e2 is not None and 'SERVED-GEN' in e2
              and e3 is None, (e, e2, e3))
        with raw((CFG.photometry, 'SERVED_GENERATIONS_FILE', mixed)):
            e = refusal(K.gate_served_generations)
        check('SERVED-GEN: a fixture serving the snapshot tier from two generations refuses', e is not None
              and 'two generations' in e, e)
        with raw((CFG.photometry, 'SERVED_GENERATIONS_FILE', g5 + '.absent')):
            e = refusal(K.gate_served_generations)
        check('SERVED-GEN: an unreadable fixture refuses', e is not None and 'unreadable' in e, e)
    finally:
        os.remove(g5)
        os.remove(mixed)


def falsify(fx, label, sets, expect, patch=None, inputs=()):
    """Each override in ``sets`` (dotted key -> value; protected keys as a --parity replay, ruled ones into a non-production
    --out) - and each raw (section, KEY, value) in ``inputs``, or ``patch`` applied to a copy of the fixture - makes
    problems() name ``expect``."""
    import copy
    import tempfile
    out = tempfile.mkdtemp(prefix='falsify_out_', dir=os.environ.get('EXOPIPE3_TEST_SCRATCH'))
    try:
        CFG.authorize([], out=out)
        with parity(), raw(*inputs):
            for k, v in sets.items():
                CFG.set(k, v, source='test')
            f2 = copy.deepcopy(fx)
            if patch:
                patch(f2)
            bad = problems(f2)
        check('falsification: %s fails, named' % label, any(expect in b for b in bad), bad)
    finally:
        CFG.reset()
        CFG.authorize([])
        os.rmdir(out)


@case
def test_falsification():
    fx = load_fixture()
    ph, cc = CFG.photometry, CFG.photometry_campaign
    falsify(fx, 'the cutout root format moved (photometry_campaign.CUTOUT_ROOT_FMT, an input of CUTOUT_ROOTS)',
            {'photometry_campaign.CUTOUT_ROOT_FMT': '/x/photometry_{generation}'}, 'photometry.CUTOUT_ROOTS[snapshot]')
    falsify(fx, 'the snapshot generation back to gen5', {'photometry.GENERATIONS[snapshot]': 'gen5'},
            'photometry.GENERATIONS[snapshot]')
    falsify(fx, 'the deep generation at gen6 (which holds no tier cutouts)', {'photometry.GENERATIONS[deep]': 'gen6'},
            'photometry.CUTOUT_ROOTS[deep]')
    falsify(fx, 'stage 1 back to stage1_gen5 (photometry_campaign.STAGE1_DIR_FMT, an input of STAGE1_DIR)',
            {'photometry_campaign.STAGE1_DIR_FMT': 'stage1_gen5'}, 'photometry.STAGE1_DIR')
    falsify(fx, 'the forced_snapshot keys of gen5 (forced_snapshot.GENERATION; not the build log\'s)',
            {'forced_snapshot.GENERATION': 'gen5'}, 'forced_snapshot.EVAL_WB')
    falsify(fx, 'an evaluation name format moved (photometry_evaluate.EVAL_DIR[wb], an input of forced_snapshot.EVAL_WB)',
            {'photometry_evaluate.EVAL_DIR[wb]': 'eval_wb_moved_{tag}'}, 'forced_snapshot.EVAL_WB')
    falsify(fx, 'the snapshot root moved (photometry.OUT_ROOT, protected: an input of the stage-1 and evaluation roots)', {},
            'forced_snapshot.STAGE1_ROOT', inputs=[(ph, 'OUT_ROOT', dict(ph.OUT_ROOT, snapshot='/x/snapshot_forced'))])
    falsify(fx, 'the wide-area work root format moved (forced_wide.WORK_ROOT_FMT)',
            {'forced_wide.WORK_ROOT_FMT': '/x/forced_wide_moved_{generation}'}, 'forced_wide.WORK_ROOT')
    falsify(fx, 'the tier forced root format moved (photometry.TIER_FORCED_ROOT_FMT)',
            {'photometry.TIER_FORCED_ROOT_FMT': '/x/forced_tiers_{generation}'}, 'photometry.TIER_FORCED_ROOTS[HWJ]')
    falsify(fx, 'the planets tier root format moved (photometry.PLANETS_TIERS_FMT)',
            {'photometry.PLANETS_TIERS_FMT': '/x/tiers_{generation}_planets_orbit'},
            'photometry.TIER_FORCED_ROOTS[SolarSystem]')
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
    falsify(fx, 'the planets page record of gen5 (before PLANETS-GEN6-DONE) against the snapshot generation gen6', {},
            'photometry.PLANETS_PAGE_GENERATION', planets_record_of('gen5'))
    old = fixture_copy(planets_record_of('gen5'))
    try:
        falsify(fx, '... which the served-root gate refuses too (SERVED-GEN)', {}, 'SERVED-GEN', planets_record_of('gen5'),
                inputs=[(ph, 'SERVED_GENERATIONS_FILE', old)])
    finally:
        os.remove(old)
    falsify(fx, 'the planets page record naming gen6 cutouts and a gen5 stage 1 (no one generation)', {},
            'photometry.PLANETS_PAGE_GENERATION', lambda f: f['page_records']['planets']['fields'].update(
                {'inputs.stage1': os.path.join(ph.OUT_ROOT['snapshot'], 'stage1_gen5_planets_orbit')}))
    falsify(fx, 'the planets page\'s tier cubes of another generation (inputs.phot_tiers)', {}, 'inputs.phot_tiers',
            lambda f: f['page_records']['planets']['fields'].update({'inputs.phot_tiers': cc.CUTOUT_ROOT_FMT.format(
                generation='gen6')}))
    falsify(fx, 'the planets page record naming another tier root', {}, 'photometry.TIER_FORCED_ROOTS[SolarSystem]',
            lambda f: f['page_records']['planets']['fields'].update({'inputs.tiers': '/x/tiers_gen6_planets_orbit'}))
    falsify(fx, 'an unlisted key naming gen5', {'forced_snapshot.REVIEW_SUBDIR_UNDER_OUT': 'review_gen5'},
            'forced_snapshot.REVIEW_SUBDIR_UNDER_OUT')
    falsify(fx, 'an unlisted key naming gen6 (neither DERIVED nor SEPARATE)', {'forced_snapshot.REVIEW_SUBDIR_UNDER_OUT':
                                                                             'review_gen6'}, 'nor SEPARATE: forced_snapshot.REVIEW_SUBDIR_UNDER_OUT')
    bad = reader_problems({'photometry_fluxscale.GEN5_ROOT': ('x', SCU + ':1', 'GEN5_ROOT')})
    check('falsification: an entry whose named line does not read it fails, named', bad and 'does not read' in bad[0], bad)
    check('falsification: the overrides are undone', not problems(fx) and not CFG.overrides
          and cc.CUTOUT_ROOT_FMT.endswith('photometry_{generation}'), problems(fx))


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
    for t in (test_page_table, test_served, test_gen5_readers, test_one_key_switch, test_derived_inputs, test_falsification,
              test_live):
        try:
            t()
        except AssertionError:
            pass                                       # printed by check(); counted in FAILED
    print('%s: %d check(s) failed' % ('FAILED' if FAILED else 'all checks passed', len(FAILED)))
    return 1 if FAILED else 0


if __name__ == '__main__':
    sys.exit(main())
