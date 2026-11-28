#!/usr/bin/env python
"""Every tier's roots name the cutout generation the SERVED registry names (data free: no file is read but this repository's
config). The registry below is the hub's page settings as served on 2026-11-27 (serve/registered_pages.json, written
2026-11-27 11:40 UTC; registry.json is generated from it), one entry per forced page with its file:line. Asserts:
  - the page table: v2's forced pages (serve.PAGES, families forced and forced-tier) are the registry's ports, each with the
    registry's Stokes and cadence;
  - per tier, the generation of the served runs (the run name's _gen<N>) is photometry.GENERATIONS[tier], and
    photometry.CUTOUT_ROOTS[tier] is that generation's root (photometry_campaign.CUTOUT_ROOT_FMT): snapshot (10 s, 1 min)
    gen6, deep and 10 minutes (sub-LST) gen5;
  - the snapshot tier's other keys follow it: photometry.STAGE1_DIR, the five forced_snapshot generation keys
    (tier_snapshot.generation_roots), the forced wide-area test's runs, work root and minute rule, and the email's forced
    runs (notify.FORCED_RUN with FORCED_RUN_GENERATION); the 10-minute and deep pages' keys (forced_tier_pages RUN_NAME,
    PHOT_PRODUCTION) follow the tier cutouts;
  - the planets page (/s/planets/) is named as served now (photometry.PLANETS_PAGE_GENERATION, gen5), and its config line says
    'switch at PLANETS-GEN6-DONE';
  - no config key names gen5 unless it is in ALLOWED_GEN5, each entry naming the served gen5 page or reader that justifies it;
  - falsification: a snapshot root set back to gen5, a deep root set to gen6, a forced_snapshot key, a wide-area run or a
    minute rule of the other generation, and an unlisted gen5 key each fail, named.
Run: python tests/test_served_generation_roots.py
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from exopipe.phase3.config import CFG                    # noqa: E402

# ---- THE SERVED REGISTRY, 2026-11-27 (/lustre/gh/main/phase3/serve/registered_pages.json): port -> (Stokes, cadence, the run
#      its --run argument serves). Change it only from that file, citing the lines.
SERVED_FORCED = {
    8775: ('V', '1min', 'Run_SNAPFORCED_V1min_gen6'),      # :778-783
    8776: ('I', '1min', 'Run_SNAPFORCED_I1min_gen6'),      # :813-818
    8785: ('I', '10s', 'Run_SNAPFORCED_I10s_gen6'),        # :848-853
    8786: ('V', '10s', 'Run_SNAPFORCED_V10s_gen6'),        # :883-888
    9120: ('V', '10min', 'Run_SNAPFORCED_V10min_gen5'),    # :223-228
    9121: ('I', '10min', 'Run_SNAPFORCED_I10min_gen5'),    # :257-262
    9122: ('V', 'deep', 'Run_SNAPFORCED_Vdeep_gen5'),      # :291-296
    9123: ('I', 'deep', 'Run_SNAPFORCED_Ideep_gen5'),      # :325-330
}
SERVED_PLANETS_ROOT = 'site_planets_SUN_gen5'   # :114-116 (static site 'planets', /s/planets/); switch at PLANETS-GEN6-DONE
PHOT_TIER = {'10s': 'snapshot', '1min': 'snapshot', '10min': 'sublst', 'deep': 'deep'}   # cadence -> photometry tier
MINUTE_RULE_OF = {'gen5': 'filename',     # gen5's stage 1 keyed a minute block by the file-name minute
                  'gen6': 'midpoint'}     # gen6's by the integration midpoint (its npz minute_basis
                                          # integration-midpoint-dateobs-v1); forced_wide_search.py:35 FW_MINUTE_RULE

# ---- THE KEYS THAT MAY NAME gen5, each with the served page or reader that reads it. A key outside this list naming gen5 fails.
ALLOWED_GEN5 = {
    'photometry.GENERATIONS[deep]': 'the deep tier cutouts the 1-hour pages p9122 / p9123 serve',
    'photometry.GENERATIONS[sublst]': 'the 10-minute tier cutouts the pages p9120 / p9121 serve',
    'photometry.CUTOUT_ROOTS[deep]': 'p9122 / p9123 (site_hwj10pc_tiers_gen5/page_record.json phot = photometry_gen5)',
    'photometry.CUTOUT_ROOTS[sublst]': 'p9120 / p9121 (the same page record)',
    'photometry.PLANETS_PAGE_GENERATION': '/s/planets/ (site_planets_SUN_gen5) until PLANETS-GEN6-DONE',
    'photometry.TIER_FORCED_ROOTS[HWJ]': 'forced_tiers_gen5, the tier pages p9120-p9123 (tier_forced_page_20261027.py)',
    'photometry.TIER_FORCED_ROOTS[10pc]': 'forced_tiers_gen5, the tier pages p9120-p9123',
    'photometry.TIER_FORCED_ROOTS[LPT]': 'forced_tiers_gen5, the tier pages p9120-p9123',
    'photometry.TIER_FORCED_ROOTS[SolarSystem]': 'tiers_gen5_planets_orbit, the planets page\'s 10-minute and deep inputs '
                                                 '(planets_gen6_stage34_20261127.sh:67 keeps it)',
    'photometry_fluxscale.GEN5_ROOT': 'not served: the flux-scale check\'s read-only comparison of generation 6 against 5 (a '
                                      'reader the gen5 per-night delete retires)',
    'forced_tier_pages.EVAL_DIRS[10min,sb]': 'p9120 / p9121: the 10-minute evaluations of the gen5 tier cutouts',
    'forced_tier_pages.EVAL_DIRS[10min,wb]': 'p9120 / p9121',
    'forced_tier_pages.EVAL_DIRS[deep,sb]': 'p9122 / p9123',
    'forced_tier_pages.EVAL_DIRS[deep,wb]': 'p9122 / p9123',
    'forced_tier_pages.EVAL_NAME': 'p9120-p9123',
    'forced_tier_pages.SCRIPT_EVAL_NAME': 'p9120-p9123',
    'forced_tier_pages.FT_PRODUCTION': 'p9120-p9123 (forced_tiers_gen5)',
    'forced_tier_pages.PHOT_PRODUCTION': 'p9120-p9123 (the gen5 tier cutouts)',
    'forced_tier_pages.RUN_NAME': 'p9120-p9123 (Run_SNAPFORCED_<S><tier>_gen5)',
    'forced_tier_pages.SITE_NAME': 'p9120-p9123 (site_hwj10pc_tiers_gen5)',
    'forced_wide.RUN_GENERATION[10min]': 'the queues of p9120 / p9121, which the ambiguous-target list reads',
    'forced_wide.RUN_GENERATION[deep]': 'the queues of p9122 / p9123',
    'notify.FORCED_RUN_GENERATION[10min]': 'p9120 / p9121, the email\'s 10-minute forced events',
    'notify.FORCED_RUN_GENERATION[deep]': 'p9122 / p9123, the email\'s deep forced events',
    'serve.FORCED_GENERATIONS{gen5}': 'a table keyed by generation (the words and samples of each generation\'s forced runs, '
                                      'V2 of its evaluations); it names no root',
}

_GEN5 = re.compile(r'gen5(?!\d)')
FAILED = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else '  %s' % (detail,)))
    if not ok:
        FAILED.append(name)


def gen_of(run):
    m = re.search(r'_(gen\d+)$', run)
    return m.group(1) if m else None


def served_generation(cadence):
    """The one generation the registry serves at a cadence (None if the registry is mixed)."""
    gens = {gen_of(r) for (_, c, r) in SERVED_FORCED.values() if c == cadence}
    return gens.pop() if len(gens) == 1 else None


def _entry(k):
    return ','.join(map(str, k)) if isinstance(k, tuple) else str(k)


def gen5_keys():
    """{'section.KEY[entry]' (or '{key}' for a dict key): value} of every config value naming gen5."""
    out = {}

    def walk(v, path):
        if isinstance(v, str):
            if _GEN5.search(v):
                out[path] = v
        elif isinstance(v, (list, tuple, set, frozenset)):
            for x in v:
                walk(x, path)
        elif isinstance(v, dict):
            for k, x in v.items():
                if isinstance(k, str) and _GEN5.search(k):
                    out[path + '{%s}' % k] = k
                walk(x, path + '[%s]' % _entry(k))
    for n, sec in sorted(CFG.sections().items()):
        for k, v in sec.items():
            walk(v, '%s.%s' % (n, k))
    return out


def problems():
    """Every key whose value disagrees with the served registry, as 'key: got, want' lines (empty = all follow it)."""
    from exopipe.phase3.lib.forced import tier_snapshot as TS
    from exopipe.phase3.lib.forced import wide_jobs as WJ
    ph, cc = CFG.photometry, CFG.photometry_campaign
    bad = []

    def want(key, got, exp):
        if got != exp:
            bad.append('%s: %r, the served registry needs %r' % (key, got, exp))
    for cad, tier in PHOT_TIER.items():
        g = served_generation(cad)
        want('photometry.GENERATIONS[%s]' % tier, ph.GENERATIONS.get(tier), g)
        want('photometry.CUTOUT_ROOTS[%s]' % tier, ph.CUTOUT_ROOTS.get(tier), cc.CUTOUT_ROOT_FMT.format(generation=g))
    gs = served_generation('10s')
    want('the 10-s and 1-min pages serve one generation', served_generation('1min'), gs)
    want('photometry.STAGE1_DIR', ph.STAGE1_DIR, cc.STAGE1_DIR_FMT.format(generation=gs))
    roots = TS.generation_roots(gs)
    for k in TS.GENERATION_KEYS:
        want('forced_snapshot.' + k, getattr(CFG.forced_snapshot, k), roots[k])
    t = CFG.forced_tier_pages
    for S, cad, run in SERVED_FORCED.values():
        if PHOT_TIER[cad] != 'snapshot':
            want('forced_tier_pages.RUN_NAME (%s %s)' % (S, cad), t.RUN_NAME.format(S=S, tier=cad), run)
    want('forced_tier_pages.PHOT_PRODUCTION', t.PHOT_PRODUCTION, ph.CUTOUT_ROOTS['deep'])
    want('photometry.CUTOUT_ROOTS: deep and sublst one root (the tier cutouts)', ph.CUTOUT_ROOTS['sublst'], ph.CUTOUT_ROOTS['deep'])
    served = {(S, cad): run for S, cad, run in SERVED_FORCED.values()}
    w = CFG.forced_wide
    for S, cad in w.AMBIGUOUS_RUNS:
        want('forced_wide run (%s %s)' % (S, cad), WJ.run_name(S, cad), served.get((S, cad)))
    for S, cad in w.RUNS:
        if PHOT_TIER.get(cad) != 'snapshot':
            bad.append('forced_wide.RUNS (%s %s): not a snapshot cadence' % (S, cad))
    want('forced_wide.WORK_ROOT', os.path.basename(w.WORK_ROOT), 'forced_wide_' + gs)
    want('forced_wide_search.MINUTE_RULE', CFG.forced_wide_search.MINUTE_RULE, MINUTE_RULE_OF.get(gs))
    n = CFG.notify
    tiers = set(n.FORCED_TIERS.values())
    want('notify.FORCED_TIERS cadences', sorted(tiers), sorted(PHOT_TIER))
    for (S, cad), run in sorted(served.items()):
        if cad in n.FORCED_RUN_GENERATION:
            want('notify.FORCED_RUN (%s %s)' % (S, cad),
                 n.FORCED_RUN.format(stokes=S, tier=cad, generation=n.FORCED_RUN_GENERATION[cad]), run)
        else:
            bad.append('notify.FORCED_RUN_GENERATION: no entry for %s' % cad)
    want('photometry.PLANETS_PAGE_GENERATION', ph.PLANETS_PAGE_GENERATION, gen_of(SERVED_PLANETS_ROOT))
    unlisted = sorted(set(gen5_keys()) - set(ALLOWED_GEN5))
    if unlisted:
        bad.append('keys naming gen5 that no served page reads (not in ALLOWED_GEN5): %s' % ', '.join(unlisted))
    return bad


def test_page_table():
    pages = {v['port']: (v['stokes'], v['kind']) for (_, _), v in CFG.serve.PAGES.items()
             if v.get('family') in ('forced', 'forced-tier')}
    check('serve.PAGES: the forced and forced-tier pages are the registry\'s 8 ports, each with its Stokes and cadence',
          pages == {p: (S, c) for p, (S, c, _) in SERVED_FORCED.items()}, pages)
    check('the registry serves one generation per cadence (gen6 for 10 s and 1 min, gen5 for 10 minutes and deep)',
          [served_generation(c) for c in ('10s', '1min', '10min', 'deep')] == ['gen6', 'gen6', 'gen5', 'gen5'])


def test_served():
    bad = problems()
    check('every tier\'s roots and every gen5 key follow the served registry', not bad, bad)
    ph = CFG.photometry
    check('the snapshot tier is gen6: photometry_gen6, stage1_gen6, eval_gen6',
          (ph.GENERATIONS['snapshot'], os.path.basename(ph.CUTOUT_ROOTS['snapshot']), ph.STAGE1_DIR,
           os.path.basename(CFG.forced_snapshot.EVAL_SB)) == ('gen6', 'photometry_gen6', 'stage1_gen6', 'eval_gen6'))
    check('the deep and 10-minute tiers are gen5 (the tier cutouts)',
          [os.path.basename(ph.CUTOUT_ROOTS[t]) for t in ('deep', 'sublst')] == ['photometry_gen5'] * 2)
    src = open(os.path.join(REPO, 'exopipe', 'phase3', 'config', 'steps', 'photometry.py')).read().splitlines()
    line = [ln for ln in src if ln.strip().startswith('PLANETS_PAGE_GENERATION =')]
    check('the planets line says "switch at PLANETS-GEN6-DONE"', len(line) == 1 and 'switch at PLANETS-GEN6-DONE' in line[0],
          line)
    stale = sorted(k for k in ALLOWED_GEN5 if k not in gen5_keys())
    check('ALLOWED_GEN5 holds only keys that name gen5 (a key switched leaves the list)', not stale, stale)


def falsify(label, sets, expect):
    """Each override in ``sets`` (dotted key -> value) makes problems() name ``expect``."""
    try:
        for k, v in sets.items():
            CFG.set(k, v, source='test')
        bad = problems()
        check('falsification: %s fails, named' % label, any(expect in b for b in bad), bad)
    finally:
        CFG.reset()


def test_falsification():
    g5 = CFG.photometry_campaign.CUTOUT_ROOT_FMT.format(generation='gen5')
    g6 = CFG.photometry_campaign.CUTOUT_ROOT_FMT.format(generation='gen6')
    falsify('the snapshot cutout root back to gen5', {'photometry.CUTOUT_ROOTS[snapshot]': g5}, 'photometry.CUTOUT_ROOTS[snapshot]')
    falsify('the snapshot generation back to gen5', {'photometry.GENERATIONS[snapshot]': 'gen5'}, 'photometry.GENERATIONS[snapshot]')
    falsify('the deep cutout root at gen6 (which holds no tier cutouts)', {'photometry.CUTOUT_ROOTS[deep]': g6},
            'photometry.CUTOUT_ROOTS[deep]')
    falsify('stage 1 back to stage1_gen5', {'photometry.STAGE1_DIR': 'stage1_gen5'}, 'photometry.STAGE1_DIR')
    falsify('a forced_snapshot key of gen5', {'forced_snapshot.EVAL_WB': '/lustre/gh/main/phase3/snapshot_forced/eval_wb_gen5'},
            'forced_snapshot.EVAL_WB')
    falsify('the wide-area 10-s runs of gen5', {'forced_wide.RUN_GENERATION[10s]': 'gen5'}, 'forced_wide run (I 10s)')
    falsify('the wide-area minute rule of gen5', {'forced_wide_search.MINUTE_RULE': 'filename'}, 'forced_wide_search.MINUTE_RULE')
    falsify('the email\'s 1-min forced runs of gen5', {'notify.FORCED_RUN_GENERATION[1min]': 'gen5'}, 'notify.FORCED_RUN (I 1min)')
    falsify('the planets page switched before PLANETS-GEN6-DONE', {'photometry.PLANETS_PAGE_GENERATION': 'gen6'},
            'photometry.PLANETS_PAGE_GENERATION')
    falsify('an unlisted key naming gen5', {'forced_snapshot.REVIEW_SUBDIR_UNDER_OUT': 'review_gen5'},
            'forced_snapshot.REVIEW_SUBDIR_UNDER_OUT')
    check('falsification: the overrides are undone', not problems(), problems())


def main():
    test_page_table()
    test_served()
    test_falsification()
    print('%s: %d check(s) failed' % ('FAILED' if FAILED else 'all checks passed', len(FAILED)))
    return 1 if FAILED else 0


if __name__ == '__main__':
    sys.exit(main())
