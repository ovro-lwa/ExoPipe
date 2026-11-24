#!/usr/bin/env python
"""The snapshot forced photometry from stage 1 to the served pages for a cutout generation: stage 2 (the evaluation), the
event pages (exopipe3 forced --tier snapshot, family C) and the serve switch (serve --tier snapshot --family forced). Data free:
synthetic stage-1 directories (empty files named as stage-1 files), a synthetic exclusion list and synthetic unit records.
Asserts:
  - V2 (lib/photometry/tests.threshold_classes_check): thresholds.csv holds one row per (sample, Stokes, tier) of the samples
    with a stage-1 file of the evaluated kind in the evaluated cells - LPT's four classes with its files (12 rows), 8 rows
    without them; a missing class, a class of a sample without stage 1 and a duplicated class fail, each named; Full, a band
    of the other kind and an excluded (cell, band) call for no class; an evaluation with no stage-1 file fails;
  - the evaluation over a campaign's stage 1 (steps_snapshot._measure_refusals, refused_for, done_evidence): per band kind, a
    cell without the kind's cutouts (SM-NOBAND), not run for a gap (SM-HELD outcome 'gap') or whose measurement refused
    (SM-DONE) is listed refused for that kind only; a cell held for its cutouts (SM-HELD 'held') or stopped by the campaign
    (CP-PIN) is not listed and has no done evidence (SE-DONE refuses the evaluation); a done kind unit is done evidence; a
    refused full-band measure.<cell> refuses both kinds (reserve :142-144, unchanged);
  - the generation of family C (lib/forced/tier_snapshot): generation_roots('gen5') = the forced_snapshot keys as configured
    (the served generation), gen6 names stage1_gen6, eval_gen6, eval_wb_gen6, zones_recomputed_gen6.csv and photometry_gen6;
    --snap-generation sets the five keys as overrides of that source and refuses a mix with --snap-eval-sb/-wb,
    --snap-zone-flags or a --set of those keys; a bad tag refuses; G-GEN: an evaluation made from another stage 1 (its
    evaluate record, else its sibling stage1_<tag>) is named, one made from the panels' stage 1 is not;
  - the serve switch (lib/serve): the title and hub text of a forced run come from serve.FORCED_GENERATIONS of the run's tag -
    gen5 in the words of the live generation-5 entries (snapshot_forced_gen5_serve_20261026.sh:57, :72), gen6 in the words
    of gen6_forced_serve_SWITCH_DRAFT.sh:55, :71 (HWJ + 10 pc + LPT); another tag refuses (G26); a refresh names the run it
    serves; readiness: an exopipe3 forced verify record done, servable, not partial and naming the served review copy is
    ready; refused, partial, not servable or of another review copy is not; no record and no log is not ready;
  - the build content test (lib/forced/tests_snapshot.queue_header) reads generation 5's queue header and the current
    build's, which names the measured samples (HWJ + 10pc + LPT).
  - the per-panel QA block of a forced page (lib/serve/forced.paired_view, page_context, check_page; content tests
    snapshot_forced_page_context): the paired view is the one run the registry entries serve.FORCED_PAGE_CONTEXT_VIEW_IDS
    serve (a missing entry, two runs or a run without a search summary refuse); the switch runs scintillation_index.py RUN
    --tier snapshot (cwd SNAPSHOT_STEP_CWD), then page_context_snapshot.py RUN --served-from VIEW --workers N, after the
    exclusion gate and before the App probe; a failed tool, a missing frame_counts.json / scintillation_index.csv / .json or a
    failed content test refuses the page; the pointer test fails on a missing pointer, one naming the run itself or another
    view, and a view without its count table; the sidecar test fails on a panel without its sidecar, a sidecar naming another
    view (the served generation-5 state) and a run without scintillation_index.csv.
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

from exopipe.phase3.config import CFG                                # noqa: E402
from exopipe.phase3.core import exclusions, record                  # noqa: E402
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
    except Refusal as e:
        return str(e)
    return None


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').close()


def stage1_files(stage1, cell, samples, bands):
    for smp in samples:
        for b in bands:
            for st in ('I', 'V'):
                touch(os.path.join(stage1, cell, 'forced_%s_%s_%s.npz' % (smp, b, st)))


def th_rows(samples, stokes=('I', 'V'), tiers=('10s', '1min')):
    return [dict(sample=s, stokes=st, tier=t) for s in samples for st in stokes for t in tiers]


# ---------------------------------------------------------------------------------------------------------------------
def test_v2_classes(d):
    stage1 = os.path.join(d, 'stage1_gen6')
    dd = os.path.join(d, 'data_decisions.csv')
    with open(dd, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lst', 'subband', 'date', 'verdict', 'reason'])
        w.writerow(['10h', SB[3], '2025-01-07', 'exclude', 'synthetic'])
    ex = exclusions.Exclusions(dd, scope='snapshot')
    check('config: the snapshot samples measured are HWJ, 10pc and LPT (photometry.SAMPLES_MEASURED)',
          tuple(CFG.photometry.SAMPLES_MEASURED['snapshot']) == ('HWJ', '10pc', 'LPT'), CFG.photometry.SAMPLES_MEASURED)
    stage1_files(stage1, CELL_A, ('HWJ', '10pc', 'LPT'), SB[:2] + WB[:1])
    stage1_files(stage1, CELL_B, ('HWJ', '10pc'), SB[:2])
    stage1_files(stage1, CELL_B, ('LPT',), [SB[3]])          # LPT only in an excluded (cell, band): no LPT class from B
    touch(os.path.join(stage1, CELL_A, 'forced_SolarSystem_%s_I.npz' % SB[0]))     # not a snapshot sample: no class
    touch(os.path.join(stage1, CELL_A, 'forced_HWJ_Full_I.npz'))
    cells_a = [dict(cell=CELL_A, kind='sb'), dict(cell=CELL_B, kind='sb')]
    cells_b = [dict(cell=CELL_B, kind='sb')]
    want = T.threshold_classes(cells_a, stage1, ex)
    check('V2 classes: three samples x 2 Stokes x 2 tiers with LPT stage 1 in an evaluated cell (12)',
          len(want) == 12 and {k[0] for k in want} == {'HWJ', '10pc', 'LPT'}, sorted(want))
    ok, n, det = T.threshold_classes_check(th_rows(('HWJ', '10pc', 'LPT')), cells_a, stage1, ex)
    check('V2: 12 rows, one per class, pass', ok and n == 12, det)
    ok, n, det = T.threshold_classes_check(th_rows(('HWJ', '10pc')), cells_a, stage1, ex)
    check('V2: the former 8 rows fail when LPT has stage 1 - the four LPT classes named',
          not ok and det.count('no row of LPT/') == 4 and 'LPT/I/10s' in det, det)
    ok, n, det = T.threshold_classes_check(th_rows(('HWJ', '10pc')), cells_b, stage1, ex)
    check('V2: 8 rows pass when LPT stage 1 lies only in an excluded (cell, band) (never reaches the histograms)', ok, det)
    ok, n, det = T.threshold_classes_check(th_rows(('HWJ', '10pc', 'LPT')), cells_b, stage1, ex)
    check('V2: a row of a sample without stage 1 in the evaluated cells fails, named',
          not ok and 'a row of LPT/I/10s' in det, det)
    ok, n, det = T.threshold_classes_check(th_rows(('HWJ', '10pc', 'LPT')) + th_rows(('HWJ',), ('V',), ('1min',)),
                                           cells_a, stage1, ex)
    check('V2: a duplicated class fails', not ok and '2 rows of HWJ/V/1min' in det, det)
    ok, n, det = T.threshold_classes_check(th_rows(('HWJ', '10pc', 'LPT'))[1:], cells_a, stage1, ex)
    check('V2: one missing class fails, named', not ok and 'no row of HWJ/I/10s' in det, det)
    wb = T.threshold_classes([dict(cell=CELL_A, kind='wb'), dict(cell=CELL_B, kind='wb')], stage1, ex)
    check('V2 classes of a wideband evaluation: only wideband files count (Full never; cell B has none)',
          len(wb) == 12 and all(v.startswith(CELL_A + ' ' + WB[0]) for v in wb.values()), wb)
    ok, n, det = T.threshold_classes_check([], [dict(cell='11h_2025-01-01', kind='sb')], stage1, ex)
    check('V2: an evaluation whose cells have no stage-1 file fails', not ok and 'no class expected' in det, det)


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
    g5 = TS.generation_roots('gen5')
    check('generation_roots(gen5) = the forced_snapshot keys as configured (the served generation)',
          all(g5[k] == getattr(s, k) for k in TS.GENERATION_KEYS), {k: (g5[k], getattr(s, k)) for k in TS.GENERATION_KEYS})
    g6 = TS.generation_roots('gen6')
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
        CFG.set('forced_snapshot.STAGE1_ROOT', '/elsewhere')
        r = raises(TS.apply_generation, snap_args(snap_generation='gen6'))
        check('--snap-generation with a --set of a generation key refuses', r is not None and 'STAGE1_ROOT' in r, r)
        CFG.reset()
        TS.apply_generation(snap_args(snap_generation='gen6'))
        got = {k: getattr(s, k) for k in TS.GENERATION_KEYS}
        src = {o[0]: o[3] for o in CFG.overrides}
        check('--snap-generation gen6 sets the five keys, recorded as overrides of source --snap-generation',
              got == g6 and all(src.get('forced_snapshot.' + k) == '--snap-generation' for k in TS.GENERATION_KEYS),
              (got, src))
        roots = TS.roots_env(s)
        check('the children read generation 6\'s cutouts and stage 1 (P3_PHOT_ROOT, P3_FORCED_STAGE1)',
              roots == {'P3_PHOT_ROOT': g6['PHOT_ROOT'], 'P3_FORCED_STAGE1': g6['STAGE1_ROOT']}, roots)
    finally:
        CFG.reset()
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
    c.cfg = CFG
    try:
        CFG.set('forced_snapshot.STAGE1_ROOT', os.path.join(base, 'stage1_gen6'))
        c.ev_sb, c.ev_wb = os.path.join(base, 'eval_gen6'), os.path.join(base, 'eval_wb_gen6')
        check('G-GEN: both evaluations made from the panels\' stage 1 - nothing named', TS.generation_mismatch(c) == [],
              TS.generation_mismatch(c))
        c.ev_sb = os.path.join(base, 'v2', 'eval_gen6')
        mm = TS.generation_mismatch(c)
        check('G-GEN: an evaluation made from generation 5\'s stage 1 under generation 6\'s panels is named',
              len(mm) == 1 and 'stage1_gen5' in mm[0], mm)
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
    # readiness
    W = os.path.join(d, 'work')
    D = os.path.join(d, 'review', 'Run_SNAPFORCED_V10s_gen6')

    def verify(status='done', **meta):
        m = dict(review=D, servable=True, partial=False, verify_line='FORCED-VERIFY-OK 3 events')
        m.update(meta)
        record.atomic_write_json(record.record_path(W, 'forced', 'C_verify_Run_SNAPFORCED_V10s_gen6'),
                                 dict(unit='C_verify_Run_SNAPFORCED_V10s_gen6', status=status, reason='', meta=m))
    try:
        CFG.set('forced_snapshot.WORK_ROOT', W)
        ok, why = forced.ready('Run_SNAPFORCED_V10s_gen6', D)
        check('ready: no exopipe3 record and no production log - not ready, both named', not ok and 'no record' in why
              and 'no log' in why, why)
        verify()
        ok, why = forced.ready('Run_SNAPFORCED_V10s_gen6', D)
        check('ready: the family-C verify record done, servable, for this review copy', ok and 'FORCED-VERIFY-OK' in why, why)
        ok, why = forced.ready('Run_SNAPFORCED_V10s_gen6', os.path.join(d, 'other'))
        check('not ready: the record verified another review copy', not ok and 'verified the review copy' in why, why)
        verify(partial=True)
        check('not ready: a partial run (an hour or cell restriction)', not forced.ready('Run_SNAPFORCED_V10s_gen6', D)[0])
        verify(servable=False)
        check('not ready: a run without association (not servable)', not forced.ready('Run_SNAPFORCED_V10s_gen6', D)[0])
        verify(status='refused')
        ok, why = forced.ready('Run_SNAPFORCED_V10s_gen6', D)
        check('not ready: a refused verify record', not ok and 'refused' in why, why)
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
    def __init__(self):
        self.outputs, self.notes = [], []

    def add_output(self, path, cls, **ctx):
        self.outputs.append((path, cls, ctx))

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
    return V, R, png


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
    # page_context: the two tools in production's form and order, then the files and the content tests
    calls = []
    fail = {}

    def fake_run(name, args=(), cwd=None, **k):
        calls.append((name, [str(a) for a in args], cwd))
        return _Res(ok=name not in fail, tail='last line of ' + name)
    plan = argparse.Namespace(parity=False, out=d, forced_view=V)
    pg = argparse.Namespace(key='forced-V-10s', run_name=os.path.basename(R), root_src=R, root=R, stokes='V')
    saved = (scripts.run, forced.paired_view)
    scripts.run = fake_run
    try:
        u = _Unit()
        notes = forced.page_context(u, plan, pg, R, V)
        S = CFG.serve
        want = [('scintillation_index', [R, '--tier', S.SNAPSHOT_FILTER_TIER], S.SNAPSHOT_STEP_CWD),
                ('page_context_snapshot', [R, '--served-from', V, '--workers', str(S.SNAPSHOT_PAGE_CONTEXT_WORKERS)], None)]
        check('page context: scintillation_index.py RUN --tier snapshot, then page_context_snapshot.py RUN --served-from VIEW '
              '--workers N (production\'s hand steps, its form and order)', calls == want, calls)
        check('page context: the scintillation table and the run recorded (classes scintillation_index, '
              'snapshot_forced_page_context with its view)',
              u.outputs == [(os.path.join(R, 'scintillation_index.csv'), 'scintillation_index', {'tier': S.SNAPSHOT_FILTER_TIER}),
                            (R, 'snapshot_forced_page_context', {'view': V})] and len(notes) == 2, (u.outputs, notes))
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
        r = raises(forced.page_context, _Unit(), plan, pg, R, V)
        check('page context: a failed content test refuses the page', r is not None and 'sidecar_per_panel' in r, r)
        os.rename(sc + '.x', sc)
        # check_page: the QA block after the exclusion gate, before the App probe
        order = []
        sv = (forced.ready, forced.missing_files, forced.page_context, exclusion_gate.scan_snapshot_run,
              exclusion_gate.refuses, servers.app_probe)
        forced.ready = lambda *a, **k: (True, 'ok')
        forced.missing_files = lambda *a, **k: []
        exclusion_gate.scan_snapshot_run = lambda D: order.append('gate') or argparse.Namespace(hits=[], summary=lambda: 'none')
        exclusion_gate.refuses = lambda *a: True
        forced.page_context = lambda u, plan, pg, D, view: order.append(('pctx', D, view)) or []

        def probe(*a, **k):
            order.append('probe')
            raise Refusal('stop here')
        servers.app_probe = probe
        try:
            raises(forced.check_page, _Unit(), plan, pg)
        finally:
            (forced.ready, forced.missing_files, forced.page_context, exclusion_gate.scan_snapshot_run,
             exclusion_gate.refuses, servers.app_probe) = sv
        check('check_page: the QA block on the served run with the paired view, after the exclusion gate and before the App '
              'probe', order == ['gate', ('pctx', R, V), 'probe'], order)
    finally:
        scripts.run, forced.paired_view = saved


def main():
    base = os.environ.get('EXOPIPE3_TEST_SCRATCH') or None
    d = tempfile.mkdtemp(prefix='test_photometry_snapshot_stage2_', dir=base)
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
