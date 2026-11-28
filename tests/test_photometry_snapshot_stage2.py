#!/usr/bin/env python
"""The snapshot forced photometry from stage 1 to the served pages for a cutout generation: stage 2 (the evaluation), the
event pages (exopipe3 forced --tier snapshot, family C) and the serve switch (serve --tier snapshot --family forced). Data free:
synthetic stage-1 directories (empty files named as stage-1 files), a synthetic exclusion list and synthetic unit records.
Asserts:
  - V2 (lib/photometry/tests.threshold_classes_check): thresholds.csv holds one row per (sample, Stokes, tier) of the samples
    the evaluation's cutout generation measured (serve.FORCED_GENERATIONS: gen5 HWJ + 10pc, 8 rows; gen6 with LPT, 12;
    restricted to the evaluation's --samples when its record names them), never derived from the files present; the
    generation comes from what names it (given, the stage-1 root, the evaluation's directory), an unknown or disagreeing one
    refuses; under gen6 the 8 rows refuse with the four LPT classes named (the engine records no dropped class); a row of
    another class, a duplicate, a missing column and a --samples sample the generation did not measure refuse, each named;
    through the content test a missing column is a recorded failure;
  - the evaluation over a campaign's stage 1 (steps_snapshot._measure_refusals, refused_for, done_evidence): per band kind, a
    cell without the kind's cutouts (SM-NOBAND), not run for a gap (SM-HELD outcome 'gap') or whose measurement refused
    (SM-DONE) is listed refused for that kind only; a cell held for its cutouts (SM-HELD 'held') or stopped by the campaign
    (CP-PIN) is not listed and has no done evidence (SE-DONE refuses the evaluation); a done kind unit is done evidence; a
    refused full-band measure.<cell> refuses both kinds (reserve :142-144, unchanged);
  - the generation of family C (lib/forced/tier_snapshot): generation_roots('gen6') = the forced_snapshot keys as configured
    (the served generation since 2026-11-27): stage1_gen6, eval_gen6, eval_wb_gen6, zones_recomputed_gen6.csv and
    photometry_gen6; gen5 names the retired generation's roots;
    --snap-generation sets forced_snapshot.GENERATION (the five keys are derived from it: not settable themselves) as an
    override of that source - ruled: into a non-production --out only, check_out refusing a production one; the served
    generation is a no-op - and refuses a mix with --snap-eval-sb/-wb, --snap-zone-flags or a --set of the key; a bad tag
    refuses; G-GEN: an evaluation made from another stage 1 (its
    evaluate record, else its sibling stage1_<tag>) is named, one made from the panels' stage 1 is not, one of unknown
    provenance (neither) is named;
  - the serve switch (lib/serve): the title and hub text of a forced run come from serve.FORCED_GENERATIONS of the run's tag -
    gen5 in the words of the live generation-5 entries (snapshot_forced_gen5_serve_20261026.sh:57, :72), gen6 in the words
    of gen6_forced_serve_SWITCH_DRAFT.sh:55, :71 (HWJ + 10 pc + LPT); another tag refuses (G26); a refresh names the run it
    serves; readiness: an exopipe3 forced verify record done, servable, not partial and naming the served review copy is
    ready; refused, partial, not servable or of another review copy is not; no record and no log is not ready;
  - the build content test (lib/forced/tests_snapshot.queue_header) reads generation 5's queue header and the current
    build's, which names the measured samples (HWJ + 10pc + LPT).
  - the page inputs of a forced page (lib/serve/forced.paired_view, page_context, check_page; the per-panel QA block, row 15,
    and production's page runner, exopipe_ops bd07263): the paired view is the one run the registry entries
    serve.FORCED_PAGE_CONTEXT_VIEW_IDS serve (a missing entry, two runs, a run without a search summary or without
    frame_counts.json refuse); the switch runs, from /tmp and in production's order, scintillation_index.py RUN --tier
    snapshot, bright_source_filter_precompute.py RUN --tier forced --workers 6, page_context_snapshot.py RUN --served-from VIEW
    --workers 4 and tess/tess_coverage.py --run RUN --from-candidates (site PYTHON tess), after G26, readiness, the served-run
    probe (a run a registry entry serves is refused before anything changes), the exclusion gate and the App probe; each
    content test runs once; no TESS interpreter (before any tool), a failed tool (the later ones not run), a missing output or
    a failed content test refuses the page (in a real registration unit: a refused item, the unit and the other page done);
    readiness finds the verify record under the run's own --out (<out>/review/<name>); the pointer test fails on a missing
    pointer, one naming the run itself or another view, and a view without its count table; the sidecar test fails on a panel
    without its sidecar, a sidecar naming another view (the served generation-5 state) and a run without
    scintillation_index.csv. TESS runs through the shared helper (lib/serve/tess.coverage: its interpreter key, --no-refresh in
    a parity run, a stale orbit table refused before any tool, the counts from the call's own summary line, none refusing);
    its content test tess_rows (class tess_coverage) passes on a forced run and fails on counts that disagree with the rows or
    the run's master_catalog.csv, a cand_id outside that catalogue and no catalogue; failing, it refuses the page only.
Scratch: a fresh directory under $EXOPIPE3_TEST_SCRATCH, else the system temp directory, removed at exit.
Run: python tests/test_photometry_snapshot_stage2.py
"""
import argparse
import csv
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

from exopipe.phase3.config import CFG, ConfigRefusal                 # noqa: E402
from exopipe.phase3.core import record                              # noqa: E402
from exopipe.phase3.core.record import Refusal                      # noqa: E402
from exopipe.phase3.lib.photometry import steps_snapshot as SS      # noqa: E402
from exopipe.phase3.lib.photometry import tests as T                # noqa: E402

FAILS = []
SB = [b for b in CFG.photometry.SNAPSHOT_BANDS if 'MHz' in b]
WB = [b for b in CFG.photometry.SNAPSHOT_BANDS if 'MHz' not in b]
CELL_A, CELL_B = '00h_2025-09-03', '10h_2025-01-07'


def check(label, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + label + ('' if cond else '  ' + str(detail)[-1500:]), flush=True)
    if not cond:
        FAILS.append(label)


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except (Refusal, ConfigRefusal) as e:
        return str(e)
    return None


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').close()


def th_rows(samples, stokes=('I', 'V'), tiers=('10s', '1min')):
    return [dict(sample=s, stokes=st, tier=t) for s in samples for st in stokes for t in tiers]


# ---------------------------------------------------------------------------------------------------------------------
def test_v2_classes(d):
    check('config: the snapshot samples measured are HWJ, 10pc and LPT (photometry.SAMPLES_MEASURED)',
          tuple(CFG.photometry.SAMPLES_MEASURED['snapshot']) == ('HWJ', '10pc', 'LPT'), CFG.photometry.SAMPLES_MEASURED)
    G = CFG.serve.FORCED_GENERATIONS
    check('config: each generation\'s samples (serve.FORCED_GENERATIONS): gen5 HWJ and 10pc, gen6 HWJ, 10pc and LPT; each a '
          'measured sample and each named in its title text',
          T.generation_samples('gen5') == ('HWJ', '10pc') and T.generation_samples('gen6') == ('HWJ', '10pc', 'LPT')
          and T.generation_samples('gen7') is None
          and all(set(e[2]) <= set(CFG.photometry.SAMPLES_MEASURED['snapshot'])
                  and e[0].replace(' ', '').split('+') == list(e[2]) for e in G.values()), G)
    cols = ['sample', 'stokes', 'tier', 'thr_survey_gauss']
    want = T.threshold_classes('gen6')
    check('V2 classes of gen6: its samples x 2 Stokes x 2 tiers (12), from the configuration alone',
          len(want) == 12 and {k[0] for k in want} == {'HWJ', '10pc', 'LPT'}, sorted(want))
    check('V2 classes of gen5: 8 (no LPT); restricted to --samples HWJ,10pc, gen6 has the same 8; an unknown generation none',
          T.threshold_classes('gen5') == {k for k in want if k[0] != 'LPT'} == T.threshold_classes('gen6', ('HWJ', '10pc'))
          and T.threshold_classes('gen7') == set())
    ok, n, det = T.threshold_classes_check(th_rows(('HWJ', '10pc', 'LPT')), cols, 'gen6')
    check('V2 gen6: 12 rows, one per class, pass', ok and n == 12 and 'generation gen6' in det, det)
    ok, n, det = T.threshold_classes_check(th_rows(('HWJ', '10pc')), cols, 'gen5')
    check('V2 gen5: 8 rows, one per class of HWJ and 10pc, pass (generation 5 never measured LPT)', ok and n == 8, det)
    ok, n, det = T.threshold_classes_check(th_rows(('HWJ', '10pc')), cols, 'gen6')
    check('V2 gen6: 8 rows refuse - the four LPT classes named (the engine records no dropped class)',
          not ok and det.count('no row of LPT/') == 4 and det.count('no row of') == 4, det)
    ok, n, det = T.threshold_classes_check(th_rows(('HWJ', '10pc', 'LPT')), cols, 'gen5')
    check('V2 gen5: an LPT row refuses, named (not a class of generation 5)',
          not ok and 'a row of LPT/I/10s, not a configured class' in det, det)
    ok, n, det = T.threshold_classes_check(th_rows(('HWJ', '10pc')), cols, 'gen7')
    check('V2: an unknown generation refuses, named', not ok and 'unknown generation gen7' in det, det)
    ok, n, det = T.threshold_classes_check(th_rows(('HWJ', '10pc')), cols, 'gen6', samples=('HWJ', '10pc'))
    check('V2 gen6: 8 rows pass when the evaluation\'s record names --samples HWJ,10pc', ok and n == 8, det)
    ok, n, det = T.threshold_classes_check(th_rows(('HWJ', '10pc', 'LPT')), cols, 'gen6', samples=('HWJ', '10pc'))
    check('V2: a row of a sample outside the evaluation\'s --samples refuses, named',
          not ok and 'a row of LPT/I/10s, not a configured class' in det, det)
    ok, n, det = T.threshold_classes_check(th_rows(('HWJ', '10pc', 'LPT', 'SolarSystem')), cols, 'gen6')
    check('V2: a row of a sample not configured refuses, named', not ok and 'a row of SolarSystem/I/10s' in det, det)
    ok, n, det = T.threshold_classes_check(th_rows(('HWJ', '10pc', 'LPT')) + th_rows(('HWJ',), ('V',), ('1min',)), cols,
                                           'gen6')
    check('V2: a duplicated class refuses', not ok and '2 rows of HWJ/V/1min' in det, det)
    ok, n, det = T.threshold_classes_check(th_rows(('HWJ', '10pc', 'LPT'))[1:], cols, 'gen6')
    check('V2: one missing class refuses, named', not ok and 'no row of HWJ/I/10s' in det, det)
    ok, n, det = T.threshold_classes_check([], cols, 'gen6')
    check('V2: an empty thresholds.csv refuses, all 12 classes missing', not ok and det.count('no row of') == 5, det)
    ok, n, det = T.threshold_classes_check(th_rows(('HWJ', '10pc')), cols, 'gen5', samples=('HWJ', 'LPT'))
    check('V2: --samples naming a sample the generation did not measure refuses', not ok and 'LPT' in det, det)
    ok, n, det = T.threshold_classes_check([dict(sample=s, stokes=st) for s, st, _ in want], ['sample', 'stokes'], 'gen6')
    check('V2: a thresholds.csv without its tier column refuses, named', not ok and 'no tier column' in det, det)
    # the generation of an evaluation: from what names it, never from the files present
    g = T.eval_generation
    check('generation: from the stage-1 root the evaluate unit records (stage1_gen6), from the directory (eval_wb_gen5, '
          'eval_gen6), or given',
          g('/x/eval_20261125', dict(stage1='/x/stage1_gen6'))[0] == 'gen6' and g('/x/eval_wb_gen5', {})[0] == 'gen5'
          and g('/x/eval_gen6/', {})[0] == 'gen6' and g('/x/eval_T1', dict(generation='gen5'))[0] == 'gen5',
          [g('/x/eval_20261125', dict(stage1='/x/stage1_gen6')), g('/x/eval_wb_gen5', {})])
    r = [g('/x/eval_wb_20261007', dict(stage1='/x/stage1')), g('/x/eval_gen5', dict(stage1='/x/stage1_gen6')),
         g('/x/eval_gen5', dict(generation='gen7')), g('/x/eval_gen5_planets_SUN', {})]
    check('generation: none named (a date tag, stage1), named twice differently, an unknown one given, or a planet '
          'evaluation refuse', r[0][0] is None and 'unknown generation' in r[0][1] and r[1][0] is None
          and 'disagree' in r[1][1] and r[2][0] is None and 'unknown generation gen7' in r[2][1] and r[3][0] is None, r)
    ev = os.path.join(d, 'eval_nocol')          # through the content test: a recorded failure, not an exception
    os.makedirs(ev, exist_ok=True)
    with open(os.path.join(ev, 'thresholds_grid.csv'), 'w') as f:
        f.write('sample,stokes,tier\n')
    with open(os.path.join(ev, 'thresholds.csv'), 'w') as f:
        f.write('sample,stokes\nHWJ,V\n')
    with open(os.path.join(ev, 'cells.csv'), 'w') as f:
        f.write('cell\n10h_2025-01-07\n')
    try:
        ok, n, det = T.ev3(ev, stage1=os.path.join(d, 'stage1_gen6'))
        r = (ok, det)
    except Exception as x:      # noqa: BLE001
        r = (None, repr(x))
    check('V2 content test (E3_exceedance_or_V2_rows): a missing column is a recorded failure, not an exception',
          r[0] is False and 'no tier column' in r[1], r)
    try:
        ok, n, det = T.ev5(ev, stage1=os.path.join(d, 'stage1_gen6'))
        r = (ok, det)
    except Exception as x:      # noqa: BLE001
        r = (None, repr(x))
    check('S content test: a cells.csv without its kind column is a recorded failure', r[0] is False and 'no kind column' in r[1], r)


# ---------------------------------------------------------------------------------------------------------------------
def write_rec(out, name, status, reason='', meta=None):
    record.atomic_write_json(record.record_path(out, 'photometry', name),
                             dict(unit=name, status=status, reason=reason, meta=dict(meta or {})))


def test_campaign_refusals(d):
    out = os.path.join(d, 'campaign')
    stage1 = os.path.join(out, 'stage1_gen6')
    os.makedirs(stage1)
    sbu = lambda c: SS.unit_name(c, list(SS.kind_bands('sb')))       # noqa: E731
    wbu = lambda c: SS.unit_name(c, list(SS.kind_bands('wb')))       # noqa: E731
    c = ['00h_2025-01-0%d' % i for i in range(1, 9)]
    write_rec(out, sbu(c[0]), 'done', meta=dict(bands_done=SB, outcome=None))
    write_rec(out, wbu(c[0]), 'no-input', 'SM-NOBAND: no wideband - every wideband unit of %s is terminal and none wrote a '
              'cutout: cutouts.%s.Red: no cutout' % (c[0], c[0]), meta=dict(outcome='no wideband'))
    write_rec(out, sbu(c[1]), 'refused', 'SM-HELD: the stage 1 of %s (subbands) is not run - a gate refused the inputs of its '
              'cutouts units (gaps, never forced): cutouts.%s.41MHz: refused C-FLUXSCALE' % (c[1], c[1]),
              meta=dict(outcome='gap'))
    write_rec(out, wbu(c[1]), 'done', meta=dict(bands_done=WB))
    write_rec(out, sbu(c[2]), 'refused', 'SM-HELD: the stage 1 of %s (subbands) waits for its cutouts units - cutouts.%s.41MHz: '
              'no record' % (c[2], c[2]), meta=dict(outcome='held'))
    write_rec(out, sbu(c[3]), 'refused', 'SM-DONE: 2 REFUSED manifest rows; (work kept: /x)')
    write_rec(out, sbu(c[4]), 'refused', 'CP-PIN: the engine code changed since the campaign started: x - not started')
    write_rec(out, sbu(c[5]), 'no-input', 'SM-NOBAND: no subband - every subband unit ...', meta=dict(outcome='no subband'))
    write_rec(out, 'measure.%s' % c[6], 'refused', 'SM-DONE: bands refused')     # a full-band unit (not a campaign's)
    sb = SS._measure_refusals(stage1, c, 'sb')
    wb = SS._measure_refusals(stage1, c, 'wb')
    check('campaign stage 1, subbands: a gap, a refused measurement, no subband cutout and a refused full-band unit are listed '
          'refused; held, stopped (CP-PIN), done and unrecorded cells are not',
          sorted(sb) == sorted([c[1], c[3], c[5], c[6]]), sb)
    check('campaign stage 1, widebands: only no wideband cutout and the full-band refusal are listed (the subband outcome '
          'never refuses the widebands)', sorted(wb) == sorted([c[0], c[6]]), wb)
    check('campaign refusal reasons name the outcome (generation 5\'s "no cutouts to measure"; the gen6 driver\'s "held: input '
          'gate refused"; "stage 1 failed")',
          wb[c[0]].startswith('no cutouts to measure: no wideband') and sb[c[1]].startswith('held: input gate refused')
          and sb[c[3]].startswith('stage 1 failed') and sb[c[5]].startswith('no cutouts to measure: no subband'), (sb, wb))
    check('the full-band behavior is unchanged: kind None lists only the refused measure.<cell>',
          sorted(SS._measure_refusals(stage1, c)) == [c[6]], SS._measure_refusals(stage1, c))
    ev = {cl: SS.done_evidence(stage1, cl, 'sb') for cl in (c[0], c[2], c[4])}
    check('done evidence: a done subband unit is; a held one (SM-HELD held) and a stopped one (CP-PIN) are not - the '
          'evaluation refuses SE-DONE', ev[c[0]][0] and not ev[c[2]][0] and not ev[c[4]][0], ev)
    check('done evidence of the widebands where the subbands are a gap: the done wideband unit',
          SS.done_evidence(stage1, c[1], 'wb')[0], SS.done_evidence(stage1, c[1], 'wb'))
    saved = SS.refused_list
    SS.refused_list = lambda kind: []
    try:
        ref, path = SS.refused_for('sb', {}, stage1, c, lifted=set())
    finally:
        SS.refused_list = saved
    check('refused_for lists the campaign refusals for the evaluated kind', sorted(ref) == sorted(sb) and path is None, ref)
    txt = SS.refused_text(ref)
    check('the engine\'s refused list has one cell,reason line per cell (no comma inside a reason)',
          txt.startswith('cell,reason\n') and all(ln.count(',') == 1 for ln in txt.splitlines()), txt)


# ---------------------------------------------------------------------------------------------------------------------
def snap_args(**k):
    a = dict(snap_generation=None, snap_eval_sb=None, snap_eval_wb=None, snap_zone_flags=None)
    a.update(k)
    return argparse.Namespace(**a)


def test_generation(d):
    from exopipe.phase3.lib.forced import tier_snapshot as TS
    s = CFG.forced_snapshot
    g6 = TS.generation_roots('gen6')
    check('generation_roots(gen6) = the forced_snapshot keys as configured (the served generation since 2026-11-27)',
          all(g6[k] == getattr(s, k) for k in TS.GENERATION_KEYS), {k: (g6[k], getattr(s, k)) for k in TS.GENERATION_KEYS})
    root = CFG.photometry.OUT_ROOT['snapshot']
    check('generation_roots(gen6): eval_gen6, eval_wb_gen6, zones_recomputed_gen6.csv, stage1_gen6 under the snapshot root; '
          'the generation-6 cutouts', g6 == dict(EVAL_SB=os.path.join(root, 'eval_gen6'),
                                                 EVAL_WB=os.path.join(root, 'eval_wb_gen6'),
                                                 ZONE_FLAGS=os.path.join(root, 'zones_recomputed_gen6.csv'),
                                                 STAGE1_ROOT=os.path.join(root, 'stage1_gen6'),
                                                 PHOT_ROOT=CFG.photometry_campaign.CUTOUT_ROOT_FMT.format(generation='gen6')), g6)
    check('a bad generation tag refuses', raises(TS.generation_roots, '../x') is not None)
    try:
        r = raises(TS.apply_generation, snap_args(snap_generation='gen6', snap_eval_sb='/x'))
        check('--snap-generation with --snap-eval-sb refuses', r is not None and '--snap-eval-sb' in r, r)
        r = raises(CFG.set, 'forced_snapshot.STAGE1_ROOT', '/elsewhere')
        check('a --set of a generation root itself refuses: it is derived (config.Derived), set its input '
              'forced_snapshot.GENERATION', r is not None and 'derived' in r and 'forced_snapshot.GENERATION' in r, r)
        CFG.authorize([], out=d)
        CFG.set('forced_snapshot.GENERATION', 'gen5')
        r = raises(TS.apply_generation, snap_args(snap_generation='gen6'))
        check('--snap-generation with a --set of forced_snapshot.GENERATION refuses', r is not None
              and 'forced_snapshot.GENERATION' in r, r)
        CFG.reset()
        g5 = TS.generation_roots('gen5')
        check('generation_roots(gen5): the retired generation\'s eval_gen5, stage1_gen5 and photometry_gen5 (not the served '
              'roots)', g5['STAGE1_ROOT'] == os.path.join(root, 'stage1_gen5') and g5['EVAL_SB'] == os.path.join(root, 'eval_gen5')
              and os.path.basename(g5['PHOT_ROOT']) == 'photometry_gen5' and all(g5[k] != getattr(s, k) for k in TS.GENERATION_KEYS),
              g5)
        TS.apply_generation(snap_args(snap_generation='gen6'))
        check('--snap-generation gen6, the served generation, is a no-op (no override)', CFG.overrides == [], CFG.overrides)
        CFG.authorize([], out=None)
        r = raises(TS.apply_generation, snap_args(snap_generation='gen5'))
        check('--snap-generation gen5 into a production --out refuses: forced_snapshot.GENERATION is ruled (--ruling-for, or a '
              'non-production --out)', r is not None and 'forced_snapshot.GENERATION' in r and 'ruling' in r
              and CFG.overrides == [], r)
        CFG.authorize([], out=d)
        TS.apply_generation(snap_args(snap_generation='gen5'))
        got = {k: getattr(s, k) for k in TS.GENERATION_KEYS}
        src = {o[0]: o[3] for o in CFG.overrides}
        check('--snap-generation gen5 (a non-production --out) sets forced_snapshot.GENERATION, recorded as an override of '
              'source --snap-generation; the five keys follow it', got == g5 and src == {'forced_snapshot.GENERATION':
                                                                                          '--snap-generation'}, (got, src))
        r = raises(CFG.check_out, CFG.photometry.OUT_ROOT['snapshot'])
        check('... and refuses at check_out when the run\'s final --out is a production root', r is not None
              and 'forced_snapshot.GENERATION' in r and 'production root' in r, r)
        roots = TS.roots_env(s)
        check('the children read generation 5\'s cutouts and stage 1 (P3_PHOT_ROOT, P3_FORCED_STAGE1)',
              roots == {'P3_PHOT_ROOT': g5['PHOT_ROOT'], 'P3_FORCED_STAGE1': g5['STAGE1_ROOT']}, roots)
    finally:
        CFG.reset()
        CFG.authorize([])
    # G-GEN: evaluation_stage1 / generation_mismatch
    base = os.path.join(d, 'gen')
    for x in ('stage1_gen5', 'stage1_gen6', 'eval_gen6', 'eval_wb_gen6', 'v2/eval_gen6', 'v2/eval_wb_gen6'):
        os.makedirs(os.path.join(base, x))
    record.atomic_write_json(record.record_path(os.path.join(base, 'v2'), 'photometry', 'evaluate.sb.gen6'),
                             dict(unit='evaluate.sb.gen6', status='done', meta=dict(stage1=os.path.join(base, 'stage1_gen5'))))
    st, how = TS.evaluation_stage1(os.path.join(base, 'eval_wb_gen6'))
    check('an evaluation without a record was made from its sibling stage1_<tag>', st == os.path.join(base, 'stage1_gen6'), how)
    st, how = TS.evaluation_stage1(os.path.join(base, 'v2', 'eval_gen6'))
    check('an evaluation with an evaluate record was made from the record\'s stage 1', st == os.path.join(base, 'stage1_gen5'),
          how)

    class C:
        pass
    c = C()
    # the panels' stage 1 is a derived key (not settable): a stand-in config names it
    c.cfg = argparse.Namespace(forced_snapshot=argparse.Namespace(STAGE1_ROOT=os.path.join(base, 'stage1_gen6')))
    try:
        c.ev_sb, c.ev_wb = os.path.join(base, 'eval_gen6'), os.path.join(base, 'eval_wb_gen6')
        check('G-GEN: both evaluations made from the panels\' stage 1 - nothing named', TS.generation_mismatch(c) == [],
              TS.generation_mismatch(c))
        c.ev_sb = os.path.join(base, 'v2', 'eval_gen6')
        mm = TS.generation_mismatch(c)
        check('G-GEN: an evaluation made from generation 5\'s stage 1 under generation 6\'s panels is named',
              len(mm) == 1 and 'stage1_gen5' in mm[0], mm)
        c.ev_sb = os.path.join(base, 'v2', 'eval_wb_gen6')
        mm = TS.generation_mismatch(c)
        check('G-GEN: an evaluation of unknown stage-1 provenance (no record, no sibling stage1_<tag>) is named',
              len(mm) == 1 and 'stage 1 unknown' in mm[0] and 'v2/eval_wb_gen6' in mm[0], mm)
        c.ev_sb = os.path.join(base, 'synthetic_evaluation')
        mm = TS.generation_mismatch(c)
        check('G-GEN: an evaluation not named eval_<tag> is of unknown provenance, named',
              len(mm) == 1 and 'stage 1 unknown' in mm[0], mm)
    finally:
        CFG.reset()


# ---------------------------------------------------------------------------------------------------------------------
GEN5_TITLE = ('Snapshot FORCED photometry (HWJ + 10 pc), Stokes V, 10-second frames - pixel-distance screen, fitted, '
              'reference-tied, localized, associated (target / background source / ambiguous) - generation-5 cutouts from the '
              'current cubes (Run_SNAPFORCED_V10s_gen5) - served UTC')
GEN6_TITLE = ('Snapshot FORCED photometry (HWJ + 10 pc + LPT), Stokes V, 10-second frames - pixel-distance screen, fitted, '
              'reference-tied, localized, associated (target / background source / ambiguous) - generation-6 cutouts from the '
              'rescaled cubes (Master flux scale) (Run_SNAPFORCED_V10s_gen6) - served UTC')
GEN5_HUB = ('forced photometry at HWJ + 10 pc targets, generation-5 cutouts from the current cubes, Stokes I 1min events: '
            'pixel-distance screen, fitted, reference-tied, localized, associated')
GEN6_HUB = ('forced photometry at HWJ + 10 pc + LPT targets, generation-6 cutouts from the rescaled cubes (Master flux scale), '
            'Stokes I 1min events: pixel-distance screen, fitted, reference-tied, localized, associated')


def serve_ns(tag):
    return argparse.Namespace(tier='snapshot', run_label=None, port_offset=0, pages=None, stokes=None, lsts=None, label=None,
                              source=None, family='forced', forced_tag=tag, restart=None, stop=None)


def test_serve(d):
    from exopipe.phase3.lib.serve import forced
    from exopipe.phase3.lib.serve import plan as planmod
    out = os.path.join(d, 'serve_out')
    site = CFG.site                     # a site without a file (CI): the legacy root and the interpreter the plan names
    saved = {k: getattr(site, k, None) for k in ('LEGACY_CODE_ROOTS', 'PYTHON')}
    if not any(r.rstrip('/').endswith('/exopipe_ops') for r in (getattr(site, 'LEGACY_CODE_ROOTS', None) or [])):
        site.LEGACY_CODE_ROOTS = list(getattr(site, 'LEGACY_CODE_ROOTS', None) or []) + [os.path.join(d, 'exopipe_ops')]
    if 'default' not in (getattr(site, 'PYTHON', None) or {}):
        site.PYTHON = dict(getattr(site, 'PYTHON', None) or {}, default=sys.executable)
    try:
        serve_checks(d, out, forced, planmod)
    finally:
        for k, v in saved.items():
            setattr(site, k, v)


def serve_checks(d, out, forced, planmod):
    texts = {}
    for tag in ('gen5', 'gen6'):
        p = planmod.make(serve_ns(tag), out)
        pg = {x.key: x for x in p.pages}
        v10, i1 = pg['forced-V-10s'], pg['forced-I-1min']
        texts[tag] = (p.title(v10, 'UTC'), dict(kv.split('=', 1) for kv in p.hub_entry(i1, 'UTC')[2])['generation'])
    check('forced title and hub text of gen5 = the live generation-5 entries\' words', texts['gen5'] == (GEN5_TITLE, GEN5_HUB),
          texts['gen5'])
    check('forced title and hub text of gen6 = the generation-6 switch\'s words (HWJ + 10 pc + LPT, rescaled cubes)',
          texts['gen6'] == (GEN6_TITLE, GEN6_HUB), texts['gen6'])
    p = planmod.make(serve_ns('gen7'), out)
    r = raises(p.title, p.pages[0], 'UTC')
    check('a tag without serve.FORCED_GENERATIONS text refuses (G26) - title', r is not None and 'G26' in r, r)
    r = raises(p.hub_entry, p.pages[0], 'UTC')
    check('a tag without serve.FORCED_GENERATIONS text refuses (G26) - hub', r is not None and 'G26' in r, r)
    p = planmod.make(serve_ns('gen6'), out)
    v10 = [x for x in p.pages if x.key == 'forced-V-10s'][0]
    t = p.title(v10, 'UTC', root='/r/Run_SNAPFORCED_V10s_gen5')
    check('the title names the run it serves (a refresh of a gen5 page under another tag: gen5 words, its run name)',
          t == GEN5_TITLE, t)
    # the hub button names the run the page's entry serves (as the title), G26 per page
    import json
    from exopipe.phase3.lib.serve import flow, register
    i1 = [x for x in p.pages if x.key == 'forced-I-1min'][0]
    rd = register.registry_dir(p)
    os.makedirs(rd, exist_ok=True)
    with open(register.files(rd)[0], 'w') as f:
        json.dump({'pages': [{'id': f'p{i1.port}', 'argv': ['--run', '/r/Run_SNAPFORCED_I1min_gen5']}]}, f)
    h = dict(kv.split('=', 1) for kv in p.hub_entry(i1, 'UTC')[2])
    check('hub entry: the run name and generation of the run the page\'s entry serves (gen5 under a gen6 plan)',
          h['run'] == 'Run_SNAPFORCED_I1min_gen5' and h['generation'] == GEN5_HUB, h)
    h = dict(kv.split('=', 1) for kv in p.hub_entry(i1, 'UTC', root='/r/Run_SNAPFORCED_I1min_gen6')[2])
    check('hub entry: a root given wins (the dry run: the run to be registered)', h['run'] == 'Run_SNAPFORCED_I1min_gen6'
          and h['generation'] == GEN6_HUB, h)
    ns = serve_ns(None)
    ns.forced_refresh_siblings = True
    pn = planmod.make(ns, out)
    i1n = [x for x in pn.pages if x.key == 'forced-I-1min'][0]
    h = dict(kv.split('=', 1) for kv in pn.hub_entry(i1n, 'UTC')[2])
    check('hub entry of a plan without a tag (a sibling refresh): the registered run, no G26', h['run'] ==
          'Run_SNAPFORCED_I1min_gen5', h)
    os.remove(register.files(rd)[0])
    # the switch: a page whose registry entry refuses (G26) is refused alone; the others' sibling links are made without it
    got, seen = [], []
    sv = (flow.register_pages, forced.paired_view, forced.check_page, register.page_entry)
    flow.register_pages = lambda run, plan, prepare, name, **k: got.append(prepare) or {}
    forced.paired_view = lambda d=None: '/view'
    forced.check_page = lambda u, plan, pg: (pg.root, {}, [])

    def entry(plan, pg, utc, D):
        seen.append((pg.key, sorted(plan.forced_after)))
        if pg.key == 'forced-I-10s':
            raise Refusal('G26 test')
        return {'id': f'p{pg.port}', 'argv': []}
    register.page_entry = entry
    try:
        forced.switch(None, p, flow.new_state(), 'UTC')
        items, refused = got[0](_Unit())
    finally:
        flow.register_pages, forced.paired_view, forced.check_page, register.page_entry = sv
    keys = sorted(x.key for x in p.pages)
    check('switch: a page whose entry refuses (G26) is refused alone, the other three registered with sibling links made '
          'without it', sorted(it[0].key for it in items) == sorted(k for k in keys if k != 'forced-I-10s')
          and list(refused) == ['forced-I-10s'] and all('forced-I-10s' not in a for k_, a in seen[-3:]), (items, refused, seen))
    # readiness (a run name production never built: since 2026-11-27 Run_SNAPFORCED_V10s_gen6 has a real FT-READY log)
    W = os.path.join(d, 'work')
    D = os.path.join(d, 'review', 'Run_SNAPFORCED_V10s_gen99')

    def verify(status='done', **meta):
        m = dict(review=D, servable=True, partial=False, verify_line='FORCED-VERIFY-OK 3 events')
        m.update(meta)
        record.atomic_write_json(record.record_path(W, 'forced', 'C_verify_Run_SNAPFORCED_V10s_gen99'),
                                 dict(unit='C_verify_Run_SNAPFORCED_V10s_gen99', status=status, reason='', meta=m))
    try:
        CFG.set('forced_snapshot.WORK_ROOT', W)
        ok, why = forced.ready('Run_SNAPFORCED_V10s_gen99', D)
        check('ready: no exopipe3 record and no production log - not ready, both named', not ok and 'no record' in why
              and 'no log' in why, why)
        verify()
        ok, why = forced.ready('Run_SNAPFORCED_V10s_gen99', D)
        check('ready: the family-C verify record done, servable, for this review copy', ok and 'FORCED-VERIFY-OK' in why, why)
        ok, why = forced.ready('Run_SNAPFORCED_V10s_gen99', os.path.join(d, 'other'))
        check('not ready: the record verified another review copy', not ok and 'verified the review copy' in why, why)
        verify(partial=True)
        check('not ready: a partial run (an hour or cell restriction)', not forced.ready('Run_SNAPFORCED_V10s_gen99', D)[0])
        verify(servable=False)
        check('not ready: a run without association (not servable)', not forced.ready('Run_SNAPFORCED_V10s_gen99', D)[0])
        verify(status='refused')
        ok, why = forced.ready('Run_SNAPFORCED_V10s_gen99', D)
        check('not ready: a refused verify record', not ok and 'refused' in why, why)
        # a run built under its own --out X: its record under X, its review copy X/review/<name>
        X = os.path.join(d, 'own_out')
        DX = os.path.join(X, CFG.forced_snapshot.REVIEW_SUBDIR_UNDER_OUT, 'Run_SNAPFORCED_V10s_gen99')
        record.atomic_write_json(record.record_path(X, 'forced', 'C_verify_Run_SNAPFORCED_V10s_gen99'),
                                 dict(unit='C_verify_Run_SNAPFORCED_V10s_gen99', status='done', reason='',
                                      meta=dict(review=DX, servable=True, partial=False, verify_line='FORCED-VERIFY-OK 2')))
        ok, why = forced.ready('Run_SNAPFORCED_V10s_gen99', DX)
        check('ready: the verify record under the run\'s own --out (review copy <out>/review/<name>), not only WORK_ROOT',
              ok and X in why and 'FORCED-VERIFY-OK 2' in why, why)
    finally:
        CFG.reset()


def test_queue_header(d):
    from exopipe.phase3.lib.forced import tests_snapshot as TSN
    got = {}
    for name, text in (('gen5', 'HWJ + 10 pc'), ('build', 'HWJ + 10pc + LPT'), ('bad', 'HWJ + 10 pc), (x')):
        run = os.path.join(d, 'qh_' + name)
        os.makedirs(run)
        with open(os.path.join(run, 'queue.txt'), 'w') as f:
            f.write('# forced photometry (%s), Stokes I, 1min tier only, clean screen, S/N (V: |S/N|) >= Gaussian survey '
                    'threshold; eval /e/eval_gen6 + /e/eval_wb_gen6\nJ0409+4640\n' % text)
        got[name] = TSN.queue_header(run)
    want = ('I', '1min', '/e/eval_gen6', '/e/eval_wb_gen6', ['J0409+4640'])
    check('the build content test reads the queue header of generation 5 (HWJ + 10 pc) and of the current build, whose '
          'samples text is the measured samples joined (HWJ + 10pc + LPT, snapshot_forced_panels.py:475)',
          got['gen5'] == want and got['build'] == want, got)
    check('a queue header whose samples text is malformed is no header', got['bad'] is None, got['bad'])


class _Res:
    def __init__(self, ok=True, tail=''):
        self.ok, self.tail, self.reason, self.rc = ok, tail, '' if ok else 'rc 1', 0 if ok else 1


class _Unit:
    """A stand-in registration unit: add_output runs the class's content tests once and records them, refuse raises Refusal
    and refuse_item records the item, as core.record.Unit does."""
    def __init__(self):
        self.outputs, self.notes, self.tests, self.refused_items, self.meta = [], [], [], [], {}

    def add_output(self, path, cls, **ctx):
        from exopipe.phase3.core import validate
        self.outputs.append((path, cls, ctx))
        self.tests += [dict(r, cls=cls, path=path) for r in validate.run(cls, path, **ctx)]

    def add_input(self, path):
        pass

    def refuse(self, reason):
        raise Refusal(reason)

    def refuse_item(self, path, reason, gate=None):
        self.refused_items.append((path, reason, gate))

    def check(self, name, passed, value=None, detail='', cls=None, path=None):
        self.tests.append(dict(name=name, passed=bool(passed), value=value, detail=detail, cls=cls, path=path))

    def note(self, n):
        self.notes.append(n)


def pctx_tree(d, name='Run_SNAPFORCED_V10s_gen6', view_name='Run_SNAPCOMB_20261116'):
    """A synthetic paired view (its own frame counts) and a forced run with one panel, its sidecar and its pointer."""
    import gzip
    import json
    V = os.path.join(d, 'review', view_name)
    R = os.path.join(d, 'review', name)
    os.makedirs(V, exist_ok=True)
    with open(os.path.join(V, 'combined_view_summary.json'), 'w') as f:
        json.dump({'batches': []}, f)
    held = ['I_10s', 'V_10s']
    with open(os.path.join(V, 'frame_counts.json'), 'w') as f:
        json.dump(dict(builder='page_context_snapshot.py', view=V, cells=[CELL_A], held=held,
                       hist={st: {'23MHz': {'0': 5, '1': 1}} for st in held}), f)
    for st in held:
        with gzip.open(os.path.join(V, f'frame_counts_{st}.csv.gz'), 'wt') as f:
            f.write('cell,band,ts,n\n%s,23MHz,20250903_095708,1\n' % CELL_A)
    folder = os.path.join(R, 'candidates', 'V', 'J0409+4640_23MHz')
    os.makedirs(folder, exist_ok=True)
    png = os.path.join(folder, 'rank0000_%s_23MHz20250903095708.png' % CELL_A)
    touch(png)
    lines = ['transient-search view of the counts and populations below: ' + view_name,
             'global noise V: 1.0 mJy (50 pctile of band, all admitted 10s frames of the transient-search view %s)' % view_name,
             'global noise I: 2.0 mJy (50 pctile of band, all admitted 10s frames of the transient-search view %s)' % view_name,
             'served detections in this frame, Stokes I (n_i_epoch): 1  (83 pctile of band, all admitted frames)',
             'served detections in this frame, Stokes V (n_v_epoch): 1  (83 pctile of band, all admitted frames)',
             "scintillation of the hour and band: index 0.100 (40 pctile of the band's hours)",
             "beam epoch/hour: 1.00 (50 pctile of band; reference = median beam of the hour's frames)"]
    with open(png[:-4] + '.page.json', 'w') as f:
        json.dump(dict(ctx_lines=lines, builder='page_context_snapshot.py', code='x',
                       frame=[CELL_A, '23MHz', 'V', '10s', '20250903_095708']), f)
    with open(os.path.join(R, 'frame_counts.json'), 'w') as f:
        json.dump(dict(builder='page_context_snapshot.py', view=V), f)
    with open(os.path.join(R, 'scintillation_index.csv'), 'w') as f:
        f.write('lst,date,band,scint_index,n_sources,n_frames,pct_in_band,state\n00h,2025-09-03,23MHz,0.1000,50,200,40.0,other\n')
    with open(os.path.join(R, 'scintillation_index.json'), 'w') as f:
        json.dump(dict(tier='snapshot', default_hidden=False, hour_bands=1, worst_pct=90.0, forced_run=True,
                       states=['worst 10% of its band', 'other', 'not measured'], counts={'23MHz': {'other': 1}}), f)
    # the page's rows (two cand_ids, one with no epoch time: left out by the TESS tool), the bright-source filter, the TESS table
    cids = ['F_%s_23MHz_20250903_095708_J0409+4640' % CELL_A, 'F_%s_23MHz_20250903_095718_J0409+4640' % CELL_A]
    with open(os.path.join(folder, 'detections.csv'), 'w') as f:
        f.write('cand_id,stokes,lst,date,band,tier,ts,ra,dec\n' + ''.join(
            '%s,V,00h,2025-09-03,23MHz,10s,%s,62.25,46.67\n' % (c, c.split('_')[4] + '_' + c.split('_')[5]) for c in cids))
    bsf_npz(R)
    with open(os.path.join(R, 'tess_coverage.csv'), 'w') as f:
        f.write('# tess_coverage.py 2026-11-24T00:00:00Z; orbit table cached; tess-point; window = the radio epoch\n'
                'cand_id,stokes,lst,date,band,ra,dec,epoch_start,epoch_end,status,sector,camera,ccd,column,row\r\n'
                '%s,V,00h,2025-09-03,23MHz,62.25,46.67,2025-09-03T09:57:08,2025-09-03T09:57:18,simultaneous,96,1,2,100,200\r\n'
                % cids[0])
    with open(os.path.join(R, 'master_catalog.csv'), 'w') as f:      # a forced run's catalogue: the page's cand_ids
        f.write('cand_id,stokes,lst,date,band\n' + ''.join('%s,V,00h,2025-09-03,23MHz\n' % c for c in cids))
    return V, R, png


# tess_coverage.py's summary line for <run>/tess_coverage.csv (:286-290), as the fake tool prints it: run, written, left out
TESS_LINE = ('%s/tess_coverage.csv: 2 detections (frame times the cell noise tables (10s, 1min)); orbit table cached; V '
             'simultaneous: 1; V no epoch time: 1; written %d (left out, page says not computed: %d)')


def bsf_npz(R, tier='forced', stokes='V', cadence='10s'):
    """A bright_source_filter_precompute.py npz of one detection, one cell and one control, its parent catalog in R."""
    import json
    import numpy as np
    cat = os.path.join(R, 'master_R075_00h_23MHz.gaul.csv')
    touch(cat)
    one = {'det_cand': np.array(['x']), 'det_ra': np.zeros(1), 'det_dec': np.zeros(1), 'det_stokes': np.array(['V']),
           'det_cell': np.zeros(1, int), 'det_sin_el': np.ones(1), 'dfront_n': np.zeros(1, int), 'det_peak': np.ones(1),
           'det_psr': np.array([''])}
    one.update({k: np.zeros(1, int) for k in ('ctl_cell', 'cfront_n')}, ctl_sin_el=np.ones(1))
    one.update({k: np.zeros(1) for k in ('cell_lst', 'cell_band', 'cell_beam_deg', 'cell_freq', 'cell_ep_I', 'cell_ep_V',
                                         'cell_cap_sr', 'cell_n_draw', 'cell_n_src')}, cell_bandname=np.array(['23MHz']))
    one.update({k: np.zeros(0) for k in ('pair_snr', 'pair_sep', 'dfront_snr', 'dfront_sep', 'dfront_flux', 'cfront_snr',
                                         'cfront_sep')}, pair_det=np.zeros(0, int))
    meta = dict(tier=tier, stokes=stokes, cadence=cadence, catalogs={'00h_23MHz': os.path.realpath(cat)},
                catalog_mtimes={'00h_23MHz': os.path.getmtime(cat)})
    with open(os.path.join(R, 'bright_source_filter.npz'), 'wb') as f:
        np.savez_compressed(f, meta=np.array(json.dumps(meta)), **one)


def test_forced_page_context(d):
    import json
    from exopipe.phase3.core import validate
    from exopipe.phase3.lib.serve import exclusion_gate, forced, register, scripts, servers
    d = os.path.join(d, 'pctx')
    V, R, png = pctx_tree(d)
    # the paired view: the one run of the four registry entries
    reg = os.path.join(d, 'registry')
    os.makedirs(reg)

    def entries(runs):
        with open(register.files(reg)[0], 'w') as f:
            json.dump({'pages': [{'id': i, 'argv': ['--run', r]} for i, r in runs.items() if r]}, f)
    ids = CFG.serve.FORCED_PAGE_CONTEXT_VIEW_IDS
    entries({i: V for i in ids})
    check('paired view: the one run the snapshot pages\' registry entries serve', forced.paired_view(reg) == os.path.realpath(V))
    other = os.path.join(d, 'review', 'Run_SNAPCOMB_20261101')
    os.makedirs(other)
    entries(dict({i: V for i in ids}, **{ids[-1]: other}))
    r = raises(forced.paired_view, reg)
    check('paired view: entries naming two runs refuse', r is not None and 'no single' in r, r)
    entries(dict({i: V for i in ids}, **{ids[0]: None}))
    r = raises(forced.paired_view, reg)
    check('paired view: a missing entry refuses', r is not None and 'no single' in r, r)
    entries({i: other for i in ids})
    r = raises(forced.paired_view, reg)
    check('paired view: a run without a search summary refuses (the tool refuses it too)',
          r is not None and 'combined_view_summary' in r, r)
    touch(os.path.join(other, 'combined_view_summary.json'))
    r = raises(forced.paired_view, reg)
    check('paired view: a run without frame_counts.json refuses before any step (production\'s runner, exopipe_ops bd07263)',
          r is not None and 'frame_counts.json' in r and 'page inputs not built' in r, r)
    # the content tests
    def tests(path=R, **ctx):
        return {t['name'].split('.')[1]: t for t in validate.run('snapshot_forced_page_context', path, **ctx)}
    t = tests(view=V)
    check('content tests: a forced run whose pointer names the paired view and whose panel has its sidecar from it passes',
          all(x['passed'] for x in t.values()) and set(t) == {'frame_counts_pointer', 'sidecar_per_panel'}, t)
    check('content tests: the pointer naming another view than the paired one fails',
          not tests(view=other)['frame_counts_pointer']['passed'])
    fj = os.path.join(R, 'frame_counts.json')
    keep = open(fj).read()
    os.remove(fj)
    t = tests(view=V)
    check('content tests: no frame_counts.json fails (production\'s page code falls back silently, review_filters.py:262-271)',
          not t['frame_counts_pointer']['passed'], t['frame_counts_pointer'])
    with open(fj, 'w') as f:
        json.dump(dict(builder='page_context_snapshot.py', view=R), f)
    check('content tests: a pointer naming the run itself fails', not tests()['frame_counts_pointer']['passed'])
    with open(fj, 'w') as f:
        f.write(keep)
    tab = os.path.join(V, 'frame_counts_V_10s.csv.gz')
    os.rename(tab, tab + '.x')
    t = tests(view=V)['frame_counts_pointer']
    check('content tests: a view without a held Stokes/tier\'s count table fails', not t['passed'] and 'V_10s' in t['detail'],
          t)
    os.rename(tab + '.x', tab)
    sc = png[:-4] + '.page.json'
    obj = json.load(open(sc))
    lines = list(obj['ctx_lines'])
    obj['ctx_lines'][0] = 'transient-search view of the counts and populations below: Run_SNAPCOMB_20261107'
    with open(sc, 'w') as f:
        json.dump(obj, f)
    t = tests(view=V)['sidecar_per_panel']
    check('content tests: a sidecar naming another view than the pointer\'s fails (the served generation-5 state)',
          not t['passed'] and '20261107' in t['detail'], t)
    obj['ctx_lines'] = lines
    with open(sc, 'w') as f:
        json.dump(obj, f)
    sp = os.path.join(R, 'scintillation_index.csv')
    os.rename(sp, sp + '.x')
    check('content tests: no scintillation_index.csv fails', not tests(view=V)['sidecar_per_panel']['passed'])
    os.rename(sp + '.x', sp)
    os.rename(sc, sc + '.x')
    check('content tests: a panel without its sidecar fails', not tests(view=V)['sidecar_per_panel']['passed'])
    os.rename(sc + '.x', sc)
    # the TESS table (tess_coverage.py --from-candidates) against the run's master_catalog.csv and the tool's own counts:
    # content test tess_rows (class tess_coverage), as lib/serve/tess.coverage records it
    tp, mc = os.path.join(R, 'tess_coverage.csv'), os.path.join(R, 'master_catalog.csv')
    tkeep, mkeep = open(tp).read(), open(mc).read()

    def tess(**kw):
        return validate.run('tess_coverage', tp, **dict(dict(written=1, left_out=1), **kw))[0]
    t = tess()
    check('TESS content test on a forced run: one row written, one cand_id left out, of its master_catalog.csv (the page\'s '
          'cand_ids) passes', t['passed'] and t['value'] == 1, t)
    for what, kw, txt, cat, key in (
            ('the tool\'s written count not the rows', dict(written=2), tkeep, mkeep, 'the tool wrote 2'),
            ('a left-out count not the catalogue\'s cand_ids without a row', dict(left_out=0), tkeep, mkeep, 'left out 0'),
            ('a cand_id outside the run\'s master_catalog.csv', {}, tkeep.replace('095708_J0409', '095700_J0409'), mkeep,
             'not in master_catalog.csv'),
            ('no master_catalog.csv', {}, tkeep, None, 'to test the rows against')):
        with open(tp, 'w') as f:
            f.write(txt)
        if cat is None:
            os.remove(mc)
        t = tess(**kw)
        check('TESS content test on a forced run: %s fails' % what, not t['passed'] and key in str(t['detail']), t)
        with open(tp, 'w') as f:
            f.write(tkeep)
        with open(mc, 'w') as f:
            f.write(mkeep)
    bp = os.path.join(R, 'bright_source_filter.npz')
    b = {x['name'].split('.')[1]: x for x in validate.run('bright_source_filter', bp, tier='forced', stokes='V', cadence='10s')}
    check('bright-source filter content tests on the forced tier\'s npz (tier forced, the page\'s Stokes and cadence) pass',
          all(x['passed'] for x in b.values()) and set(b) == {'arrays_consistent', 'parents_current'}, b)
    # page_context: the four page inputs in production's form and order (bd07263), then the files and the content tests
    calls = []
    fail = {}
    tess_counts = [1, 1]          # written, left out on the fake tool's summary line; empty: it prints none

    def fake_run(name, args=(), cwd=None, python='default', env=None, **k):
        a = [str(x) for x in args]
        calls.append((name, a, cwd, python, (env or {}).get('NUMPY_MADVISE_HUGEPAGE')))
        tail = 'last line of ' + name
        if name == 'tess_coverage':
            tail = TESS_LINE % (a[1], *tess_counts) if tess_counts else 'Traceback: x'
        return _Res(ok=name not in fail, tail=tail)
    plan = argparse.Namespace(parity=False, out=d, forced_view=V)
    pg = argparse.Namespace(key='forced-V-10s', run_name=os.path.basename(R), root_src=R, root=R, stokes='V', kind='10s')
    from exopipe.phase3.lib.pages import common as pcommon, tess_orbits
    from exopipe.phase3.lib.serve import tess as stess
    saved = (scripts.run, forced.paired_view, pcommon.interpreter_or_none, tess_orbits.stale_for_reader, stess.orbit_table)
    scripts.run = fake_run
    stess.orbit_table = lambda: os.path.join(d, 'TESS_orbit_times.csv')     # beside the tool, which CI's site lacks
    pcommon.interpreter_or_none = lambda key: sys.executable      # the site's TESS interpreter need not be on this host
    try:
        u = _Unit()
        forced.page_context(u, plan, pg, R, V)
        S = CFG.serve
        cw = S.SNAPSHOT_STEP_CWD
        want = [('scintillation_index', [R, '--tier', S.SNAPSHOT_FILTER_TIER], cw, 'default', '0'),
                ('bright_source_filter', [R, '--tier', 'forced', '--workers', '6'], cw, 'default', '0'),
                ('page_context_snapshot', [R, '--served-from', V, '--workers', '4'], cw, 'default', '0'),
                ('tess_coverage', ['--run', R, '--from-candidates'], cw, CFG.pages_deep.TESS_PYTHON, None)]
        check('page inputs: scintillation_index.py RUN --tier snapshot, bright_source_filter_precompute.py RUN --tier forced '
              '--workers 6, page_context_snapshot.py RUN --served-from VIEW --workers 4, tess/tess_coverage.py --run RUN '
              '--from-candidates (lib/serve/tess.coverage: site PYTHON[pages_deep.TESS_PYTHON]), all from /tmp: production\'s '
              'runner, its form and order (exopipe_ops bd07263)', calls == want and cw == '/tmp', calls)
        check('page inputs: the TESS table (the tool\'s own counts), the scintillation table, the bright-source filter (tier '
              'forced, the page\'s Stokes and cadence) and the run (its view) recorded, each tool noted',
              u.outputs == [(os.path.join(R, 'tess_coverage.csv'), 'tess_coverage', {'written': 1, 'left_out': 1}),
                            (os.path.join(R, 'scintillation_index.csv'), 'scintillation_index', {'tier': S.SNAPSHOT_FILTER_TIER}),
                            (os.path.join(R, 'bright_source_filter.npz'), 'bright_source_filter',
                             {'tier': 'forced', 'stokes': 'V', 'cadence': '10s'}),
                            (R, 'snapshot_forced_page_context', {'view': V})] and len(u.notes) == 4
              and u.meta['tess_coverage_pages'] == {pg.key: {'written': 1, 'left_out': 1}}, (u.outputs, u.notes, u.meta))
        names = [t['name'] for t in u.tests]
        check('page inputs: each content test runs once (through add_output only), all passed',
              names and len(names) == len(set(names)) and all(t['passed'] for t in u.tests)
              and {'snapshot_forced_page_context.frame_counts_pointer', 'snapshot_forced_page_context.sidecar_per_panel',
                   'bright_source_filter.arrays_consistent', 'bright_source_filter.parents_current',
                   'tess_coverage.tess_rows', 'scintillation_index.table_matches_definition'}
              <= set(names), names)
        del calls[:]
        pcommon.interpreter_or_none = lambda key: None
        r = raises(forced.page_context, _Unit(), plan, pg, R, V)
        check('page inputs: no TESS interpreter on this host refuses the page before any tool runs',
              r is not None and "PYTHON['tess']" in r and calls == [], (r, calls))
        pcommon.interpreter_or_none = lambda key: sys.executable
        fail['bright_source_filter'] = 1
        del calls[:]
        r = raises(forced.page_context, _Unit(), plan, pg, R, V)
        check('page inputs: a failed bright_source_filter_precompute.py refuses the page; the page context and TESS do not run',
              r is not None and 'bright_source_filter_precompute.py' in r and 'old entry kept' in r
              and [c[0] for c in calls] == ['scintillation_index', 'bright_source_filter'], (r, calls))
        fail.clear()
        fail['tess_coverage'] = 1
        r = raises(forced.page_context, _Unit(), plan, pg, R, V)
        check('page inputs: a failed tess_coverage.py refuses the page', r is not None and 'tess_coverage.py' in r
              and 'old entry kept' in r, r)
        fail.clear()
        tess_counts[:] = []
        r = raises(forced.page_context, _Unit(), plan, pg, R, V)
        check('page inputs: no summary line for the run\'s tess_coverage.csv in the tool\'s output refuses the page',
              r is not None and 'no summary line' in r and 'old entry kept' in r, r)
        tess_counts[:] = [1, 1]
        fp = os.path.join(R, 'bright_source_filter.npz')
        os.rename(fp, fp + '.x')
        r = raises(forced.page_context, _Unit(), plan, pg, R, V)
        check('page inputs: no bright_source_filter.npz after the tools refuses the page',
              r is not None and 'bright_source_filter.npz' in r and 'lacks' in r, r)
        os.rename(fp + '.x', fp)
        fp = os.path.join(R, 'tess_coverage.csv')
        os.rename(fp, fp + '.x')
        u = _Unit()
        r = raises(forced.page_context, u, plan, pg, R, V)
        check('page inputs: no tess_coverage.csv after the tool refuses the page (its content test), the file a refused item',
              r is not None and 'tess_rows' in r and [x[0] for x in u.refused_items] == [fp], (r, u.refused_items))
        os.rename(fp + '.x', fp)
        # a parity serve: --no-refresh (it never writes the shared orbit table), the table reaching now; a stale one refuses first
        pplan = argparse.Namespace(parity=True, out=d, forced_view=V)
        tess_orbits.stale_for_reader = lambda table, now=None: None
        del calls[:]
        u = _Unit()
        forced.page_context(u, pplan, pg, R, V)
        check('page inputs, parity serve: tess_coverage.py with --no-refresh, the orbit table recorded as reaching now',
              calls[-1][:2] == ('tess_coverage', ['--run', R, '--from-candidates', '--no-refresh'])
              and 'serve.tess_orbit_table_reaches_now' in [t['name'] for t in u.tests], (calls[-1], [t['name'] for t in u.tests]))
        tess_orbits.stale_for_reader = lambda table, now=None: 'ends 2026-01-01, before now'
        del calls[:]
        r = raises(forced.page_context, _Unit(), pplan, pg, R, V)
        check('page inputs, parity serve: an orbit table that does not reach now refuses the page before any tool runs',
              r is not None and 'orbit table' in r and 'ends 2026-01-01' in r and calls == [], (r, calls))
        tess_orbits.stale_for_reader = saved[3]
        bsf_npz(R, tier='snapshot')
        u = _Unit()
        r = raises(forced.page_context, u, plan, pg, R, V)
        check('page inputs: a bright-source filter failing its content test (meta tier snapshot) refuses the page, the npz '
              'recorded as a refused item', r is not None and 'arrays_consistent' in r
              and [x[0] for x in u.refused_items] == [os.path.join(R, 'bright_source_filter.npz')], (r, u.refused_items))
        bsf_npz(R)
        tess_counts[:] = [2, 1]
        u = _Unit()
        r = raises(forced.page_context, u, plan, pg, R, V)
        check('page inputs: a TESS table failing its content test (the tool says written 2, the file has 1 row) refuses the '
              'page, the table recorded as a refused item', r is not None and 'tess_rows' in r
              and [x[0] for x in u.refused_items] == [os.path.join(R, 'tess_coverage.csv')], (r, u.refused_items))
        tess_counts[:] = [1, 1]
        sj = os.path.join(R, 'scintillation_index.json')
        sk = open(sj).read()
        with open(sj, 'w') as f:
            f.write(sk.replace('"default_hidden": false', '"default_hidden": true'))
        r = raises(forced.page_context, _Unit(), plan, pg, R, V)
        check('page context: a scintillation table failing its content test refuses the page',
              r is not None and 'scintillation_index.table_matches_definition' in r, r)
        with open(sj, 'w') as f:
            f.write(sk)
        fail['page_context_snapshot'] = 1
        r = raises(forced.page_context, _Unit(), plan, pg, R, V)
        check('page context: a failed page_context_snapshot.py refuses the page', r is not None and
              'page_context_snapshot.py' in r and 'old entry kept' in r, r)
        fail.clear()
        os.remove(fj)
        r = raises(forced.page_context, _Unit(), plan, pg, R, V)
        check('page context: no frame_counts.json after the tools refuses the page',
              r is not None and 'frame_counts.json' in r, r)
        with open(fj, 'w') as f:
            f.write(keep)
        os.rename(sc, sc + '.x')
        u = _Unit()
        r = raises(forced.page_context, u, plan, pg, R, V)
        check('page context: a failed content test refuses the page, its run recorded as a refused item',
              r is not None and 'sidecar_per_panel' in r and [x[0] for x in u.refused_items] == [R], (r, u.refused_items))
        os.rename(sc + '.x', sc)
        # in a real registration unit: one page's failed content test leaves the unit (and the other pages) done
        from exopipe.phase3.core.record import Run
        V2, R2, _ = pctx_tree(os.path.join(d, 'two'))
        pg2 = argparse.Namespace(**dict(vars(pg), root_src=R2, root=R2))
        run = Run('serve', ['test'], os.path.join(d, 'out'), CFG, config_sections=None)
        with run.unit('snapshot_forced_TEST_register', stage='switch', tier='snapshot') as ru:
            os.rename(sc, sc + '.x')
            r = raises(forced.page_context, ru, plan, pg, R, V)
            os.rename(sc + '.x', sc)
            r2 = raises(forced.page_context, ru, plan, pg2, R2, V2)
        check('page context in a real unit: the failing page refused, the other page passes, the unit ends done',
              r is not None and r2 is None and ru.status == 'done' and [x['path'] for x in ru.refused_items] == [R],
              (r, r2, ru.status, ru.reason, ru.refused_items))
        # check_page: what only reads first (G26, readiness, files, already served), then the gate, the App probe, the QA block
        order = []
        sv = (forced.ready, forced.missing_files, forced.page_context, exclusion_gate.scan_snapshot_run,
              exclusion_gate.refuses, servers.app_probe, forced.served_by)
        forced.ready = lambda *a, **k: order.append('ready') or (True, 'ok')
        forced.missing_files = lambda *a, **k: []
        exclusion_gate.scan_snapshot_run = lambda D: order.append('gate') or argparse.Namespace(hits=[], summary=lambda: 'none')
        exclusion_gate.refuses = lambda *a: True
        forced.served_by = lambda plan_, D: order.append('served?') or []

        def pctx_stop(u, plan, pg, D, view):
            order.append(('pctx', D, view))
            raise Refusal('stop here')
        forced.page_context = pctx_stop
        servers.app_probe = lambda *a, **k: order.append('probe') or {}
        try:
            raises(forced.check_page, _Unit(), plan, pg)
            check('check_page: readiness, the served-run probe, the exclusion gate and the App probe before the QA block, '
                  'which runs on the run with the paired view', order == ['ready', 'served?', 'gate', 'probe', ('pctx', R, V)],
                  order)
            del order[:]
            forced.served_by = lambda plan_, D: order.append('served?') or ['p8786']
            r = raises(forced.check_page, _Unit(), plan, pg)
            check('check_page: a run a registry entry serves already is refused before anything changes (no gate, probe or '
                  'QA block: never rewritten in place)', r is not None and 'served already (p8786)' in r and 'in place' in r
                  and order == ['ready', 'served?'], (r, order))
            del order[:]
            pgx = argparse.Namespace(**dict(vars(pg), root_src=os.path.join(d, 'Run_SNAPFORCED_V10s_gen99')))
            r = raises(forced.check_page, _Unit(), plan, pgx)
            check('check_page: G26 (a tag without serve.FORCED_GENERATIONS text) refuses the page before anything else',
                  r is not None and 'G26' in r and order == [], (r, order))
        finally:
            (forced.ready, forced.missing_files, forced.page_context, exclusion_gate.scan_snapshot_run,
             exclusion_gate.refuses, servers.app_probe, forced.served_by) = sv
        # served_by reads the plan's registry
        rd = os.path.join(d, 'reg_served')
        os.makedirs(rd)
        with open(register.files(rd)[0], 'w') as f:
            json.dump({'pages': [{'id': 'p8786', 'argv': ['--run', R]}, {'id': 'p8785', 'argv': ['--run', V]}]}, f)
        sv = register.registry_dir
        register.registry_dir = lambda plan_: rd
        try:
            check('served_by: the registry entries whose --run is the run', forced.served_by(plan, R) == ['p8786']
                  and forced.served_by(plan, os.path.join(d, 'x')) == [], forced.served_by(plan, R))
        finally:
            register.registry_dir = sv
    finally:
        scripts.run, forced.paired_view, pcommon.interpreter_or_none, tess_orbits.stale_for_reader, stess.orbit_table = saved


def main():
    base = os.environ.get('EXOPIPE3_TEST_SCRATCH') or None
    d = tempfile.mkdtemp(prefix='test_photometry_snapshot_stage2_', dir=base)
    # a synthetic exclusion list (the content tests and the serve's G17 read one; CI has no production file): one row, a
    # cell no check uses
    from exopipe.phase3.core import exclusions
    dd = os.path.join(d, 'data_decisions.csv')
    with open(dd, 'w') as f:
        f.write('lst,subband,date,verdict,reason\n23h,82MHz,2024-01-01,exclude,synthetic\n')
    CFG.paths.DATA_DECISIONS = dd
    exclusions._CACHE.clear()
    try:
        test_v2_classes(d)
        test_campaign_refusals(d)
        test_generation(d)
        test_serve(d)
        test_queue_header(d)
        test_forced_page_context(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print('%d check(s) failed: %s' % (len(FAILS), FAILS) if FAILS else 'all checks passed', flush=True)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
