#!/usr/bin/env python3
"""The catalog step's grouping (item F.3): lib/catalog/grouping_stage.py, lib/catalog/tests_grouping.py (content tests),
lib/catalog/wraps/builder_grouping.py (the builder with the grouping), the 10-minute build wiring (tier_sublst._build) and the
`--tier sublst --stages group` unit. The deep chain's stage is tested in tests/test_catalog-deep.py
(test_grouping_design_b_in_the_chain).

  mode         the defaults are the ruled ones (design-b, one-rule, multiples-level); a disagreeing pair and an unknown value
               refuse; the parity file holds production's two lines; every code site that switches is marked WAVE2 GROUPING
  frozen       the stage on the FROZEN subsets of the approved inputs (F_grouping/frozen_inputs: 10-minute I/V, deep I/V):
               content tests pass, identity and multiples partitions = the research code's (golden), a regroup carries every
               ID with no event, an incremental append carries the early IDs and records every change
  defects      the content tests catch: a detection missing / twice / with another fingerprint, an identity row off its
               members, a multiples link off its group, a member flag off its group, two identities in a deep V group, an ID
               of another format, an ID changed without its move event, a re-issued ID, a state off its table
  runner       the patches apply exactly once to the live builder and to every frozen 10-minute builder; a missing patch text
               or a census change refuses; on a stub builder carrying the patch points, PASSTHROUGH writes byte-identical files
               and DESIGN-B forms the grouping's identities with the one multiples rule; a stale grouping refuses
  sublst       the 10-minute build stage (tier_sublst._build) runs the builder through the runner with the grouping under
               design B and the deep-omission decision table that grouping recorded, refuses a selected row the grouping does
               not hold, a grouping run that recorded no table and a table changed since (S-G28), runs it in place under
               production's mode
  record       one_rule_record against synthetic builder outputs: rescued multiples, singles dropped by the sigma-stack test
               and by the repeat rungs, a single kept by a cross-match rung
  carry        carry_judgements: agreeing pages carried unchanged, a disagreement a CONFLICT not carried (kept with both
               verdicts for the page, merge_conflicts.jsonl), an unreached judgement reported, a non-empty journal refused
  cli          `exopipe3 catalog --tier sublst --stages group --group-input <frozen 10-minute subset> --group-omission
               <a table>`: the first run needs --group-new-state; a second run carries every ID; under production's mode the
               unit is no-input
  omission     THE DEEP OMISSION BEFORE THE GROUPING (S-G28; production's order, sublst_cumulative.sh:99-107): the OMIT cells'
               rows are absent from the grouping, which equals on every field the grouping of the input with those rows removed
               by hand; the dropped rows are recorded; the record names the table (what the builds read); no table, a missing
               one and one without an LST grouped refuse; a table changed after the grouping makes the builds' reading refuse
  naming       THE NAMING RULE (grouping_stage.naming_order) through the 10-minute unit on the frozen I and V subsets: asked
               V,I or I,V, the unit groups I first; a name both Stokes would take stays I's short name and V takes the long
               form; no name is held in both; the cross-Stokes link is written (S-G27 not hit); an unchanged dataset gives the
               same names and IDs run after run and in either order asked; each run switches the link to a new version, keeping
               the current one and KEEP_SUPERSEDED before it
Run: NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py taskset -c $(bash
     /lustre/gh/main/exopipe_ops/node_free_cores.sh) /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3
     tests/test_catalog_grouping_stage.py [names]
Writes only under /tmp/exopipe3_tests_grouping_stage (removed at the end); reads the frozen inputs and the builders read-only.
"""
import collections
import csv
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from exopipe.phase3.lib.catalog import grouping as G  # noqa: E402
from exopipe.phase3.lib.catalog import grouping_stage as GS  # noqa: E402
from exopipe.phase3.lib.catalog import tests_grouping as TG  # noqa: E402
from grouping_stub_builder import PATCHABLE_BUILDER  # noqa: E402
from frozen_stub_builder import FROZEN_BUILDER  # noqa: E402

T = '/tmp/exopipe3_tests_grouping_stage'
FROZEN = '/lustre/gh/main/phase3/refactor_analysis/refactor_wt_20261102/F_grouping/frozen_inputs'
LIVE_BUILDER = '/lustre/gh/main/exopipe_ops/build_source_catalogs_v3.py'
FROZEN_BUILDERS = '/lustre/gh/main/phase3/sublst/code_frozen/*/exopipe_ops/build_source_catalogs_v3.py'
PY = sys.executable
FAILS, N = [], [0]
ALL_LSTS = [f'{h:02d}h' for h in range(24)]
OMISSION_COLS = ['lst', 'band', 'night', 'stokes', 'gates', 'verdict', 'reason']


def _bgm():
    from exopipe.phase3.lib.catalog.wraps import builder_grouping as BG
    return BG


def cells_of(*paths):
    """{cell: n} of the rows of csv files (production-form cand_ids; a cand_id that names no cell is left out - the grouping
    then refuses it)"""
    n = collections.Counter()
    for p in paths:
        for r in rcsv(p)[1]:
            k = _bgm().cell_of(r.get('cand_id'), r.get('stokes'))
            if k:
                n[k] += 1
    return n


def omission_dir(d, omit=(), lsts=ALL_LSTS, cover=(), cells=None):
    """d/deep_omission.csv (the omit stage's --out layout): one KEEP decision per LST of ``lsts`` plus an OMIT decision per
    (lst, band, night, stokes) of ``omit``, and its coverage record d/deep_omission_cells.csv (the omit stage's
    write_cells): the cells of the rows of the csv files ``cover`` (the detections rated) plus ``cells``; returns d"""
    rows = [dict(lst=L, band='73MHz', night='2000-01-01', stokes='I', gates='beam', verdict='KEEP', reason='test') for L in lsts]
    rows += [dict(lst=k[0], band=k[1], night=k[2], stokes=k[3], gates='beam', verdict='OMIT', reason='test') for k in omit]
    os.makedirs(d, exist_ok=True)
    wcsv(f'{d}/deep_omission.csv', OMISSION_COLS, rows)
    n = cells_of(*cover)
    n.update(cells or {})
    _bgm().write_cells(f'{d}/deep_omission.csv', n)
    return d


def sha256(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({str(detail)[:700]})'), flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def rcsv(p):
    with open(p, newline='') as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)


def wcsv(p, cols, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cols), extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def inputs(tier):
    if tier == '10min':
        return f'{FROZEN}/tenmin/master_catalog.csv', f'{FROZEN}/tenmin/pertie_detections.csv'
    return f'{FROZEN}/deep/master_catalog.csv', f'{FROZEN}/deep/frame_beam_bmaj.csv'


# =====================================================================================================================
def test_mode():
    from exopipe.phase3.config import CFG
    g = CFG.catalog_grouping
    m = GS.mode()
    check('mode: the ruled defaults (design-b, one-rule, multiples-level)', (m.grouping, m.rule, m.rungs, m.design_b) ==
          ('design-b', 'one-rule', 'multiples-level', True), m)
    check('mode: GROUPING, MULTIPLES_RULE and REPEAT_RUNGS are RULED (10-26 design; 10-27 one chain; the refactor chat\'s 11-04 '
          'decisions under the 10-16 delegation: no multiples tables on any tier, repeat rungs at the multiples level)',
          'R-GREGG-20261026-GROUPING-DESIGN-B-APPROVED-IDENTITY-097-MULTIPLES-085-DEEPV-IDENTITY-ONLY-PREVIEW-THEN-REFACTOR'
          in str(CFG.ruled('catalog_grouping.GROUPING'))
          and 'R-GREGG-20261027-HUB-BOTH-LISTENERS-MULTIPLES-CHAIN-FORCED-EVERY-TIER' in str(CFG.ruled('catalog_grouping.MULTIPLES_RULE'))
          and 'R-REFACTOR-20261104-NO-MULTIPLES-TABLES-ANY-TIER' in str(CFG.ruled('catalog_grouping.MULTIPLES_RULE'))
          and CFG.ruled('catalog_grouping.REPEAT_RUNGS') == ('R-REFACTOR-20261104-REPEAT-RUNGS-MULTIPLES-LEVEL',),
          [CFG.ruled(f'catalog_grouping.{k}') for k in ('GROUPING', 'MULTIPLES_RULE', 'REPEAT_RUNGS')])
    keep = (g.GROUPING, g.MULTIPLES_RULE, g.REPEAT_RUNGS)
    try:
        for a, b, c, want in (('production', 'one-rule', 'multiples-level', 'must agree'),
                              ('design-b', 'production', 'multiples-level', 'must agree'),
                              ('design-c', 'one-rule', 'multiples-level', 'not one of'),
                              ('design-b', 'one-rule', 'sometimes', 'not one of')):
            g.GROUPING, g.MULTIPLES_RULE, g.REPEAT_RUNGS = a, b, c
            try:
                GS.mode()
                got = 'accepted'
            except GS.StageRefusal as e:
                got = str(e)
            check(f'mode: {a} / {b} / {c} refuses ({want})', want in got, got)
        g.GROUPING, g.MULTIPLES_RULE, g.REPEAT_RUNGS = 'production', 'production', 'multiples-level'
        check("mode: production / production (the parity file) is accepted, design_b False", GS.mode().design_b is False)
    finally:
        g.GROUPING, g.MULTIPLES_RULE, g.REPEAT_RUNGS = keep
    lines = [ln.split('#', 1)[0].strip() for ln in open(f'{REPO}/exopipe/phase3/parity/production_20261013.cfg')]
    check('parity file: production\'s grouping and its two multiples chains',
          "catalog_grouping.GROUPING = 'production'" in lines and "catalog_grouping.MULTIPLES_RULE = 'production'" in lines)
    sites = {'lib/catalog/deep_chain.py': 3, 'lib/catalog/tier_sublst.py': 2, 'lib/catalog/incremental.py': 1}
    bad = {f: open(f'{REPO}/exopipe/phase3/{f}').read().count('WAVE2 GROUPING') for f in sites}
    check('code sites marked WAVE2 GROUPING (deep group / associate / fork, sub-LST build, incremental gate)',
          all(bad[f] >= n for f, n in sites.items()), bad)


# =====================================================================================================================
def run_stage(tier, S, det, out, state, label, mode=None, new=False, inp=None):
    return GS.run_group(det, tier, S, out, state, label, dict(master_catalog=inp), group_mode=mode, new_state=new,
                        check=lambda pp: TG.failures(pp, tier, S, inp))


def test_frozen():
    for tier, S in (('10min', 'I'), ('10min', 'V'), ('deep', 'I'), ('deep', 'V')):
        mc, side = inputs(tier)
        det = GS.load_det(tier, S, mc, side)
        out, state = f'{T}/frozen/{tier}_{S}/run1', f'{T}/frozen/{tier}_{S}/state'
        try:
            run_stage(tier, S, det, out, state, 'r1', inp=mc)
            refused = ''
        except GS.StageRefusal as e:
            refused = str(e)
        check(f'{tier} {S}: no state and no --group-new-state REFUSES (IDs never restart at 1 silently)', 'new=True' in refused,
              refused)
        paths, summ, res = run_stage(tier, S, det, out, state, 'r1', new=True, inp=mc)
        gold = np.load(f'{FROZEN}/golden/{tier}_{S}.npz')
        check(f'{tier} {S}: identities and multiples = the research code (golden)',
              G.same_partition(res.identity, gold['ident']) and G.same_partition(res.multiples, gold['fork']),
              summ['counts'])
        bad = TG.failures(paths, tier, S, mc)
        check(f'{tier} {S}: content tests pass on the products ({summ["counts"]["identities"]} identities, '
              f'{summ["counts"]["multiples"]} multiples)', not bad, bad[:3])
        check(f'{tier} {S}: the state is saved only now, and equals the table',
              os.path.exists(GS.state_path(state, tier, S)) and not TG.state_failures(GS.state_path(state, tier, S), tier, S,
                                                                                     paths['detections']))
        if tier == 'deep' and S == 'V':
            _, mr = rcsv(paths['multiples'])
            check('deep V: the multiples level is off (one identity per multiples-level group)',
                  all(r['n_identities'] == '1' for r in mr))
        p2, s2, r2 = run_stage(tier, S, det, f'{T}/frozen/{tier}_{S}/run2', state, 'r2', mode='regroup', inp=mc)
        same = (list(res.identity_id) == list(r2.identity_id)) and (list(res.multiples_id) == list(r2.multiples_id))
        ev = [json.loads(ln)['event'] for ln in open(p2['events'])]
        check(f'{tier} {S}: a regroup of the same detections carries every ID, no ID event', same and ev == ['run'], ev[:4])
        check(f'{tier} {S}: the regroup run keeps the previous state beside its products (T6 reads it)',
              os.path.exists(p2.get('state_previous', '')) and not TG.table_failures(p2['detections'], tier, S, mc))
    # incremental append (10-minute I): the early 75% of the nights, then every night
    mc, side = inputs('10min')
    det = GS.load_det('10min', 'I', mc, side)
    nights = sorted(set(det['night'].tolist()))
    early = np.isin(det['night'], nights[:int(len(nights) * 0.75)])
    sub = {k: (v[early] if isinstance(v, np.ndarray) and len(v) == len(early) else v) for k, v in det.items()}
    state = f'{T}/incr/state'
    p1, _, r1 = run_stage('10min', 'I', sub, f'{T}/incr/run1', state, 'early', new=True, inp=None)
    p2, s2, r2 = run_stage('10min', 'I', det, f'{T}/incr/run2', state, 'all', mode='incremental', inp=mc)
    old = dict(zip(r1.cand_id.tolist(), r1.identity_id.tolist()))
    moved = {e['cand_id'] for e in r2.events if e.get('event') == 'move' and e.get('level') == 'identity'}
    changed = [c for c, i in zip(r2.cand_id.tolist(), r2.identity_id.tolist()) if c in old and old[c] != i and c not in moved]
    check('incremental append: the early detections keep their identity IDs unless a move event says otherwise; T6 passes',
          not changed and not TG.table_failures(p2['detections'], '10min', 'I', mc),
          (len(changed), s2['mode'], s2['merges'], s2['splits']))


# =====================================================================================================================
def _copy_products(src, dst):
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    return dst


def test_defects():
    mc, side = inputs('10min')
    base = f'{T}/frozen/10min_V/run2/grouping_V'
    if not os.path.isdir(base):
        check('defects: needs test_frozen first', False)
        return
    tier, S = '10min', 'V'

    def mutate(name, fn, want, fname=GS.DETECTIONS, inp=mc, tier=tier, S=S, src=base):
        d = _copy_products(src, f'{T}/defects/{name}/grouping_{S}')
        cols, rows = rcsv(f'{d}/{fname}')
        cols, rows = fn(cols, rows)
        wcsv(f'{d}/{fname}', cols, rows)
        bad = TG.table_failures(f'{d}/{GS.DETECTIONS}', tier, S, inp)
        check(f'defects: {name} -> {want}', any(b.startswith(want) for b in bad), bad[:3])

    mutate('a detection dropped', lambda c, r: (c, r[1:]), 'T2')
    mutate('a detection twice', lambda c, r: (c, r + r[:1]), 'T1')
    mutate('another fingerprint', lambda c, r: (c, [dict(x, snr=repr(float(x['snr']) * 1.5)) if i == 0 else x
                                                    for i, x in enumerate(r)]), 'T2')

    def off_count(c, r):
        r[0] = dict(r[0], n_det=str(int(r[0]['n_det']) + 1))
        return c, r
    mutate('an identity row off its members', off_count, 'T3', fname=GS.IDENTITIES)

    def off_link(c, r):
        k = next(i for i, x in enumerate(r) if int(x['n_identities']) >= 2)
        r[k] = dict(r[k], identity_ids=r[k]['identity_ids'].split(';')[0])
        return c, r
    mutate('a multiples link off its group', off_link, 'T4', fname=GS.MULTIPLES)

    def off_flag(c, r):
        k = next(i for i, x in enumerate(r) if x['multiple'] == '1')
        r[k] = dict(r[k], multiple='0')
        return c, r
    mutate('a member flag off its group', off_flag, 'T4')
    mutate('an ID of another format', lambda c, r: (c, [dict(x, identity_id=x['identity_id'].replace('GTV', 'GDV'))
                                                        if x['identity_id'] == r[0]['identity_id'] else x for x in r]), 'T5')

    # deep V: two identities in one group
    dv = f'{T}/frozen/deep_V/run2/grouping_V'
    mcd, _ = inputs('deep')

    def two_in_group(c, r):
        a, b = r[0], next(x for x in r if x['identity_id'] != r[0]['identity_id'])
        m0 = a['multiples_id']
        return c, [dict(x, multiples_id=m0, multiples_name=a['multiples_name']) if x['identity_id'] == b['identity_id'] else x
                   for x in r]
    if os.path.isdir(dv):
        mutate('two identities in one deep V group', two_in_group, 'T4', inp=None, tier='deep', S='V', src=dv)

    # T6: an identity ID changed without its move event; a re-issued (retired) ID
    def silent_move(c, r):
        prev = json.load(open(f'{base}/state_previous/grouping_10min_V_state.json'))
        x0 = next(x for x in r if x['cand_id'] in prev['assign'])
        other = next(x['identity_id'] for x in r if x['identity_id'] != x0['identity_id'])
        return c, [dict(x, identity_id=other) if x['cand_id'] == x0['cand_id'] else x for x in r]
    mutate('an ID changed without its move event', silent_move, 'T6')
    d = _copy_products(base, f'{T}/defects/reissued/grouping_V')
    pp = f'{d}/state_previous/grouping_10min_V_state.json'
    P = json.load(open(pp))
    _, rows = rcsv(f'{d}/{GS.DETECTIONS}')
    P['ids'][rows[0]['identity_id']]['status'] = 'retired'
    json.dump(P, open(pp, 'w'))
    bad = TG.carried_failures(pp, rows, f'{d}/{GS.EVENTS}')
    check('defects: an ID retired in the previous state re-issued -> T6', any('never reused' in b for b in bad), bad[:2])
    # the state off its table
    st = f'{T}/frozen/10min_V/state/grouping_10min_V_state.json'
    sd = f'{T}/defects/state'
    os.makedirs(sd, exist_ok=True)
    S_ = json.load(open(st))
    c0 = next(iter(S_['assign']))
    other = next(v[0] for v in S_['assign'].values() if v[0] != S_['assign'][c0][0])
    S_['assign'][c0][0] = other
    json.dump(S_, open(f'{sd}/grouping_10min_V_state.json', 'w'))
    bad = TG.state_failures(f'{sd}/grouping_10min_V_state.json', '10min', 'V', f'{base}/{GS.DETECTIONS}')
    check('defects: a state assignment off its table -> S2', any(b.startswith('S2') for b in bad), bad[:2])


# =====================================================================================================================
def _stub_world(d):
    """a builder input of two bands (Stokes I) and its grouping table: identity A (one multiple: windows w1, w2, below 7
    sigma), identity B (one window, 6.5 sigma, promoted only by REP in production's eyes), identity C (one window, failing
    the sigma-stack test), identity D (one window, 9 sigma), identity E in two bands of one window (8 + 6 sigma)."""
    os.makedirs(d, exist_ok=True)
    rows = []

    def add(cid, band, lst, date, ra, dec, snr, sstack='0.5'):
        rows.append(dict(cand_id=cid, band=band, lst=lst, date=date, stokes='I', ra=repr(ra), dec=repr(dec), snr=repr(snr),
                         sstack=sstack))
    add('I_05h_A1', '46MHz', '05h', '2030-01-01', 80.0, 30.0, 6.2)
    add('I_05h_A2', '46MHz', '05h', '2030-01-02', 80.001, 30.0, 6.4)
    add('I_05h_B1', '46MHz', '05h', '2030-01-03', 81.0, 31.0, 6.5, sstack='')
    add('I_05h_B2', '46MHz', '05h', '2030-01-03', 81.0004, 31.0, 6.1, sstack='')
    add('I_05h_C1', '46MHz', '05h', '2030-01-01', 82.0, 32.0, 7.5, sstack='2.0')
    add('I_05h_D1', '46MHz', '05h', '2030-01-02', 83.0, 33.0, 9.0)
    add('I_05h_E1', '46MHz', '05h', '2030-01-02', 84.0, 34.0, 8.0)
    add('I_05h_E2', '50MHz', '05h', '2030-01-02', 84.0005, 34.0, 6.0, sstack='')
    wcsv(f'{d}/selected_I_sb.csv', list(rows[0]), rows)
    ident = {'A': ('GTI-0000001', 'J0520+3000a', 'MTI-0000001', '1', '2'), 'B': ('GTI-0000002', 'J0524+3100a', 'MTI-0000002', '0', '1'),
             'C': ('GTI-0000003', 'J0528+3200a', 'MTI-0000003', '0', '1'), 'D': ('GTI-0000004', 'J0532+3300a', 'MTI-0000004', '0', '1'),
             'E': ('GTI-0000005', 'J0536+3400a', 'MTI-0000005', '0', '1')}   # display names unlike the stub's own J-names
    tab = []
    for r in rows:
        k = r['cand_id'].split('_')[2][0]
        i, nm, m, flag, mult = ident[k]
        tab.append(dict(cand_id=r['cand_id'], stokes='I', band=r['band'], lst=r['lst'], date=r['date'],
                        key=f"{r['lst']}|{r['date']}", night=r['date'], ra=r['ra'], dec=r['dec'], snr=r['snr'], identity_id=i,
                        identity_name=nm, multiples_id=m, multiples_name=nm, multiplicity=mult, multiple=flag))
    wcsv(f'{d}/grouping_I/detections.csv', GS.DET_COLS, tab)
    open(f'{d}/v_deep_images.csv', 'w').write('lst,scope,stokes,date\n')
    return rows, tab


def _bg():
    import importlib.util
    spec = importlib.util.spec_from_file_location('bg', GS.builder_runner())
    bg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bg)
    return bg


def _stub_sha1(builder):
    return _bg().normalized_sha1(builder, open(builder).read())


def _run_runner(d, out, mode, table=None, builder=None, accepted=None, table_sha1=None, provenance=False):
    env = dict(os.environ, EXOPIPE3_GROUPING_MODE=mode, EXOPIPE3_GROUPING_TIER='deep', EXOPIPE3_GROUPING_RUNGS='multiples-level',
               EXOPIPE3_GROUPING_BUILDER_SHA1=','.join(accepted if accepted is not None else [_stub_sha1(builder)]))
    if table:
        env['EXOPIPE3_GROUPING_TABLE'] = table
    if table_sha1:
        env['EXOPIPE3_GROUPING_TABLE_SHA1'] = table_sha1
    if provenance:
        env['EXOPIPE3_GROUPING_PROVENANCE'] = '1'
    p = subprocess.run([PY, GS.builder_runner(), builder, 'selected_I_sb.csv', out, 'v_deep_images.csv'], cwd=d, env=env,
                       capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout + p.stderr


def _dir_digest(d):
    return {f: hashlib.sha1(open(os.path.join(d, f), 'rb').read()).hexdigest() for f in sorted(os.listdir(d))}


def test_runner():
    import importlib.util
    spec = importlib.util.spec_from_file_location('bg', GS.builder_runner())
    bg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bg)
    files = [LIVE_BUILDER] + sorted(glob.glob(FROZEN_BUILDERS))
    bad = []
    for f in files:
        try:
            out, names = bg.patch(open(f).read())
            compile(out, f, 'exec')
        except Exception as e:                  # noqa: BLE001 - the failure reported
            bad.append(f'{f}: {e}')
    check(f'runner: every patch applies exactly once and compiles on the live builder and {len(files) - 1} frozen 10-minute '
          f'builders', not bad, bad[:2])
    src = open(LIVE_BUILDER).read()
    for what, mod in (('a patch text missing', src.replace("        if SSTACK_GATE and len({(m['lst'], m['date']) for m in mem}) == 1:\n",
                                                           "        if SSTACK_GATE and len({m['date'] for m in mem}) == 1:\n")),
                      ('a census change (a new use of SSTACK_GATE)', src + '\nif SSTACK_GATE: pass\n')):
        try:
            bg.patch(mod)
            got = 'patched'
        except bg.Refused as e:
            got = str(e)
        check(f'runner: {what} REFUSES (re-port)', 'REFUSING' in got and 're-port' in got, got[:200])
    # the stub builder, unpatched vs passthrough vs design B
    d = f'{T}/runner'
    shutil.rmtree(d, ignore_errors=True)
    rows, tab = _stub_world(d)
    sys.path.insert(0, d)
    open(f'{d}/_stubcat.py', 'w').write(STUBCAT)
    b = f'{d}/build_source_catalogs_v3.py'
    open(b, 'w').write(PATCHABLE_BUILDER.lstrip())
    p = subprocess.run([PY, b, 'selected_I_sb.csv', 'out_plain', 'v_deep_images.csv'], cwd=d, capture_output=True, text=True)
    rc, txt = _run_runner(d, 'out_pass', 'passthrough', builder=b)
    check('runner PASSTHROUGH on the stub builder: byte-identical files to the builder alone',
          p.returncode == 0 and rc == 0 and _dir_digest(f'{d}/out_plain') == _dir_digest(f'{d}/out_pass')
          and 'GROUPING-BUILDER done' in txt, (p.returncode, rc, txt[-300:]))
    rc, txt = _run_runner(d, 'out_b', 'design-b', table=f'{d}/grouping_I/detections.csv', builder=b)
    check('runner DESIGN-B ran (machine lines; every one of the 8 input rows checked against the grouping)',
          rc == 0 and 'GROUPING-BUILDER start' in txt and 'GROUPING-BUILDER done' in txt and '8 input rows checked' in txt, txt[-500:])
    sv = {r['source_id'] + '|' + r['band']: r for p_ in glob.glob(f'{d}/out_b/*_sources_survivors.csv') for r in rcsv(p_)[1]}
    mem = {r['cand_id']: r['source_id'] for p_ in glob.glob(f'{d}/out_b/*_source_members.csv') for r in rcsv(p_)[1]}
    ss = {r['cand_id'] for p_ in glob.glob(f'{d}/out_b/*_sigma_stack_singleton_removed.csv') for r in rcsv(p_)[1]}
    names = {t['cand_id']: t['identity_name'] for t in tab}
    check('design B: every member file row is named by its identity', all(mem[c] == names[c] for c in mem), mem)
    check('design B: the multiple A (two windows, below 7 sigma, unpromoted) survives the floor and the sigma-stack gate',
          'J0520+3000a|46MHz' in sv and sv['J0520+3000a|46MHz']['multiple'] == '1' and sv['J0520+3000a|46MHz']['multi_integration'] == '1'
          and sv['J0520+3000a|46MHz']['multiples_identities'] == 'J0520+3000a', sorted(sv))
    check('design B: the single B (one window, 6.5 sigma; two detections of one image: the stub\'s REP) is NOT promoted by the '
          'repeat rung - the multiples level decides repeats', 'J0524+3100a|46MHz' not in sv, sorted(sv))
    check('design B: the single C fails the sigma-stack test and is removed (recorded)', 'I_05h_C1' in ss and
          'J0528+3200a|46MHz' not in sv, sorted(ss))
    check('design B: the single D at 9 sigma survives; E keeps its 8-sigma band (identity across two bands)',
          'J0532+3300a|46MHz' in sv and 'J0536+3400a|46MHz' in sv and sv['J0536+3400a|46MHz']['identity_bands'] == '46MHz+50MHz'
          and sv['J0536+3400a|46MHz']['identity_id'] == 'J0536+3400a', sorted(sv))
    # stale grouping: a selected row the grouping does not hold, then one held with another S/N
    T_ = rcsv(f'{d}/grouping_I/detections.csv')[1]
    wcsv(f'{d}/stale1.csv', GS.DET_COLS, T_[1:])
    rc, txt = _run_runner(d, 'out_s1', 'design-b', table=f'{d}/stale1.csv', builder=b)
    check('runner: a builder row the grouping does not hold REFUSES (exit 2, stale grouping)', rc == 2 and 'stale' in txt,
          txt[-300:])
    wcsv(f'{d}/stale2.csv', GS.DET_COLS, [dict(T_[0], snr='9.99')] + T_[1:])
    rc, txt = _run_runner(d, 'out_s2', 'design-b', table=f'{d}/stale2.csv', builder=b)
    check('runner: a row grouped with another fingerprint REFUSES', rc == 2 and 'stale' in txt, txt[-300:])
    check('covers(): the stage pre-check names both', len(GS.covers(f'{d}/stale1.csv', f'{d}/selected_I_sb.csv')) == 1
          and len(GS.covers(f'{d}/stale2.csv', f'{d}/selected_I_sb.csv')) == 1)
    # F7 (review 2026-11-04): the builder pin - the accepted path-normalized sha1s (config catalog_grouping.BUILDER_SHA1)
    from exopipe.phase3.config import CFG
    live = _bg().normalized_sha1(LIVE_BUILDER, open(LIVE_BUILDER).read())
    frozen = {_bg().normalized_sha1(f, open(f).read()) for f in glob.glob(FROZEN_BUILDERS) if '.refused' not in f}
    check('F7: the config pin holds the live builder and every frozen 10-minute tree\'s builder (path-normalized; 7 versions)',
          live in CFG.catalog_grouping.BUILDER_SHA1 and frozen <= set(CFG.catalog_grouping.BUILDER_SHA1)
          and len(set(CFG.catalog_grouping.BUILDER_SHA1)) == 7, (live, sorted(frozen - set(CFG.catalog_grouping.BUILDER_SHA1))))
    rc, txt = _run_runner(d, 'out_pin', 'design-b', table=f'{d}/grouping_I/detections.csv', builder=b,
                          accepted=list(CFG.catalog_grouping.BUILDER_SHA1))
    check('F7: a builder not in the pin REFUSES before anything runs (re-port), even with every patch text in place',
          rc == 2 and 're-port' in txt and 'path-normalized sha1' in txt and not os.path.exists(f'{d}/out_pin'), txt[-300:])
    rc, txt = _run_runner(d, 'out_nopin', 'design-b', table=f'{d}/grouping_I/detections.csv', builder=b, accepted=[])
    check('F7: no accepted list REFUSES', rc == 2 and 'BUILDER_SHA1 is empty' in txt, txt[-300:])
    # F2: the table's sha1 - the runner reads the table once and refuses another one than the stage checked
    rc, txt = _run_runner(d, 'out_sha', 'design-b', table=f'{d}/grouping_I/detections.csv', builder=b, table_sha1='0' * 40)
    check('F2: a grouping table whose sha1 is not the one the stage checked REFUSES (a regroup in between)',
          rc == 2 and 'changed after the check' in txt, txt[-300:])
    sha = GS.table_digest(f'{d}/grouping_I/detections.csv')
    rc, txt = _run_runner(d, 'out_prov', 'design-b', table=f'{d}/grouping_I/detections.csv', builder=b, table_sha1=sha,
                          provenance=True)
    prov = json.load(open(f'{d}/out_prov/grouping_table.json')) if os.path.exists(f'{d}/out_prov/grouping_table.json') else {}
    check('F2/F3: the checked sha1 passes, and the provenance names the table and its sha1 beside the catalogues',
          rc == 0 and prov.get('table_sha1') == sha and prov.get('rows_checked') == 8, (rc, prov, txt[-200:]))


STUBCAT = r'''
import csv, os
def rcsv(p):
    with open(p, newline='') as f:
        rd = csv.DictReader(f)
        return list(rd.fieldnames or []), list(rd)
def wcsv(p, cols, rows):
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cols), extrasaction='ignore'); w.writeheader(); w.writerows(rows)
'''


# =====================================================================================================================
def test_sublst_build():
    """tier_sublst._build under design B: the runner with the LST's grouping; a selected row the grouping does not hold
    refuses (S-G18); production's mode runs the builder in place."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.catalog import tier_sublst as TS
    d = f'{T}/sublst'
    shutil.rmtree(d, ignore_errors=True)
    U, ops, gdir = f'{d}/U', f'{d}/ops', f'{d}/grouping'
    rows, tab = _stub_world(U)
    prod = {r['cand_id']: f'I_05h_{r["date"]}T03_{r["band"]}_{i + 1}' for i, r in enumerate(rows)}   # name their cells
    for p in (f'{U}/selected_I_sb.csv', f'{U}/grouping_I/detections.csv'):
        cols, rr = rcsv(p)
        wcsv(p, cols, [dict(r, cand_id=prod[r['cand_id']]) for r in rr])
    sel_cells = cells_of(f'{U}/selected_I_sb.csv')
    os.makedirs(f'{gdir}/grouping_I')
    shutil.move(f'{U}/grouping_I/detections.csv', f'{gdir}/grouping_I/detections.csv')
    os.makedirs(ops)
    open(f'{ops}/_stubcat.py', 'w').write(STUBCAT)
    # the stub with the frozen statistics' patch points: an LST-wide build under design B writes the frozen record
    open(f'{ops}/build_source_catalogs_v3.py', 'w').write(FROZEN_BUILDER.lstrip())
    open(f'{ops}/build_localizations.py', 'w').write(
        "import os, sys\nout = os.environ['P3_LOC_OUT']; os.makedirs(out, exist_ok=True)\n"
        "open(out + '/localizations.csv', 'w').write('identity_id,ra,dec\\nJ0532+3300a,83.0,33.0\\n')\nprint('localized')\n")
    open(f'{ops}/emit_rejected_record.py', 'w').write("open('rejected.csv', 'w').write('cand_id\\n'); print('0 rejected rows')\n")
    open(f'{U}/frame_times.csv', 'w').write('key\nx\n')
    os.makedirs(f'{U}/pertie')
    open(f'{U}/pertie/pertie_detections.csv', 'w').write('cand_id\n')

    class Ctx:
        pass
    ctx = Ctx()
    ctx.U, ctx.dry, ctx.builds, ctx.pt, ctx.lst = U, False, ('noveto',), f'{U}/pertie', '05h'
    ctx.night = None                                # an LST-wide pass (a night pass's build is the delta build, tested apart)
    ctx.py = {'default': PY}
    ctx.opsc = lambda name: os.path.join(ops, name)
    ctx.log = TS.Log(f'{U}/catalogs.log')
    ctx.env = lambda **extra: {k: str(v) for k, v in extra.items()}
    ctx.grouping_mode = GS.mode()
    ctx.grouping_dir = gdir
    ctx.grouping_table = lambda S: os.path.join(gdir, f'grouping_{S}', 'detections.csv')
    keep_pin, keep_fpin = CFG.catalog_grouping.BUILDER_SHA1, CFG.catalog_grouping.FROZEN_BUILDER_SHA1
    CFG.catalog_grouping.BUILDER_SHA1 = tuple(keep_pin) + (_stub_sha1(f'{ops}/build_source_catalogs_v3.py'),)
    CFG.catalog_grouping.FROZEN_BUILDER_SHA1 = tuple(keep_fpin) + (_stub_sha1(f'{ops}/build_source_catalogs_v3.py'),)

    omt = f'{omission_dir(f"{d}/omit", cells=sel_cells)}/deep_omission.csv'   # KEEP only: the builder reads the selection

    def record(status='done', sha=None, finished='2030-01-01T00:00:00Z', name='Run_SLSTB.group', omission=True, S='I',
               table=None):
        t = ctx.grouping_table(S)
        tb = table or omt
        r = dict(unit=name, status=status, finished=finished, reason='' if status == 'done' else 'a refusal',
                 meta=dict(stage='group', tier='sublst', stokes=[S],
                           tables={} if status != 'done' else {S: dict(path=t, sha1=sha or GS.table_digest(t))}))
        if omission:
            r['meta']['omission'] = dict(path=tb, sha256=sha256(tb), cells_path=_bgm().cells_path(tb),
                                         cells_sha256=sha256(_bgm().cells_path(tb)))
        os.makedirs(f'{gdir}/_records/catalog', exist_ok=True)
        json.dump(r, open(f'{gdir}/_records/catalog/{name}.json', 'w'))
    record()

    def run_build():
        s = TS.Stage(ctx, 'build', TS.DryUnit())
        plan = []
        try:
            TS._build(ctx, s, plan, 'noveto')
            return 'done', s, plan
        except Exception as e:                  # noqa: BLE001 - the refusal reported
            return f'{type(e).__name__}: {e}', s, plan
    st, s, plan = run_build()
    surv = {r['source_id'] for p in glob.glob(f'{U}/source_catalogs_noveto_I_sb/*_sources_survivors.csv') for r in rcsv(p)[1]}
    check('sub-LST build under design B: the runner ran with the grouping (machine line recorded), identities as sources',
          st == 'done' and 'GROUPING-BUILDER done' in json.dumps(s.u.meta.get('grouping_builder')) and 'J0520+3000a' in surv
          and any('builder_grouping.py' in ln for ln in plan), (st, s.u.meta.get('grouping_builder'), surv))
    check('sub-LST build: the grouping table and its sha1 recorded on the unit; the provenance beside the catalogues',
          s.u.meta.get('grouping_tables', {}).get('I', {}).get('sha1') == GS.table_digest(ctx.grouping_table('I'))
          and os.path.exists(f'{U}/source_catalogs_noveto_I_sb/grouping_table.json'), s.u.meta.get('grouping_tables'))
    check('sub-LST build under design B writes the frozen record of the per-hour delta build (builder_frozen.json)',
          os.path.exists(f'{U}/source_catalogs_noveto_I_sb/builder_frozen.json'),
          sorted(os.listdir(f'{U}/source_catalogs_noveto_I_sb')) if os.path.isdir(f'{U}/source_catalogs_noveto_I_sb') else None)
    frz = json.load(open(f'{U}/source_catalogs_noveto_I_sb/builder_frozen.json')) \
        if os.path.exists(f'{U}/source_catalogs_noveto_I_sb/builder_frozen.json') else {}
    cov = _bgm().cells_path(omt)
    check('sub-LST build (S-G28): the unit records the decision table its grouping recorded (path, sha256, coverage record)',
          s.u.meta.get('grouping_omission') == dict(path=omt, sha256=sha256(omt), cells_path=cov, cells_sha256=sha256(cov)),
          s.u.meta.get('grouping_omission'))
    check('sub-LST build (S-G28): the frozen record names the decision table (path, sha256, coverage sha256; nothing dropped)',
          (frz.get('omission') or {}).get('sha256') == sha256(omt) and frz['omission'].get('path') == omt
          and frz['omission'].get('cells_sha256') == sha256(cov) and frz['omission'].get('dropped') == 0, frz.get('omission'))
    for what, setup, want in (('its grouping run recorded no deep-omission decision table',
                               lambda: record(omission=False), 'recorded no deep-omission decision table'),
                              ('the decision table changed since its grouping run recorded it',
                               lambda: (record(), open(omt, 'a').write('05h,46MHz,2030-01-01,I,beam,OMIT,late\n')),
                               'changed since')):
        setup()
        shutil.rmtree(f'{U}/source_catalogs_noveto_I_sb', ignore_errors=True)
        st, s, plan = run_build()
        check(f'S-G28: sub-LST build REFUSES when {what}', 'S-G28' in st and want in st and 'REFUSING' in st, st)
    # FAIL CLOSED (finding 2): a selected row of a cell the table does not decide REFUSES, the cell named
    miss = ('05h', '46MHz', '2030-01-03', 'I')
    omission_dir(f'{d}/omit', cells={k: v for k, v in sel_cells.items() if k != miss})
    record()
    shutil.rmtree(f'{U}/source_catalogs_noveto_I_sb', ignore_errors=True)
    st, s, plan = run_build()
    check('S-G28: a selected row of a cell the decision table does not decide REFUSES (never kept), naming the cell',
          'S-G28' in st and '05h/46MHz/2030-01-03/I' in st and 'have no verdict' in st
          and not os.path.isdir(f'{U}/source_catalogs_noveto_I_sb'), st)
    # finding 5: every selected row of the key in OMIT cells - nothing built, no catalog, no frozen record; recorded by name
    omission_dir(f'{d}/omit', omit=list(sel_cells), cells=sel_cells)
    record()
    os.makedirs(f'{U}/source_catalogs_noveto_I_sb', exist_ok=True)          # an earlier catalog directory goes
    st, s, plan = run_build()
    part = f'{U}/detection_record_parts/builder_noveto_I_sb.csv'
    prt = rcsv(part)[1] if os.path.exists(part) else []
    un = (s.u.meta.get('omission_unbuilt') or {}).get('noveto_I_sb', '')
    check('S-G28: a key whose every selected row is in an OMIT cell builds nothing (no catalog directory, no frozen record, '
          'no builder run) and the unit names it: "all N selected rows in OMIT cells"; the record part holds every row with '
          'the omission outcome',
          st == 'done' and not os.path.isdir(f'{U}/source_catalogs_noveto_I_sb') and f'all {len(rows)} selected rows in OMIT '
          f'cells of {omt}' in un and not (s.u.meta.get('grouping_builder') or {}).get('noveto_I_sb')
          and sorted(r['cand_id'] for r in prt) == sorted(prod.values())
          and {r['builder_outcome'] for r in prt} == {_bgm().OMIT_OUTCOME}, (st, un, plan, prt[:2]))
    # finding 7: the Stokes I and V groupings recorded different tables - the build REFUSES
    omission_dir(f'{d}/omit', cells=sel_cells)
    other = f'{omission_dir(f"{d}/omit_v", cells=sel_cells, lsts=["05h"])}/deep_omission.csv'
    record()
    os.makedirs(f'{gdir}/grouping_V', exist_ok=True)
    shutil.copy(ctx.grouping_table('I'), ctx.grouping_table('V'))
    wcsv(f'{U}/selected_V_sb.csv', ['cand_id', 'band', 'lst', 'date', 'stokes', 'ra', 'dec', 'snr', 'sstack'],
         [dict(cand_id=f'V_05h_2030-01-01T03_46MHz_99', band='46MHz', lst='05h', date='2030-01-01', stokes='V', ra='1',
               dec='1', snr='7', sstack='')])
    record(name='Run_SLSTV.group', S='V', table=other)
    st, s, plan = run_build()
    check('S-G28: the Stokes I and V groupings filtered by different decision tables REFUSE the build (both named)',
          'S-G28' in st and 'different deep-omission decision tables' in st and omt in st and other in st, st)
    os.remove(f'{U}/selected_V_sb.csv')
    os.remove(f'{gdir}/_records/catalog/Run_SLSTV.group.json')
    shutil.rmtree(f'{gdir}/grouping_V')
    record()
    # F2: the build reads only a table a DONE group-unit record recorded, with the sha1 it has now
    tab0 = open(ctx.grouping_table('I')).read()

    def case_refused():
        record('refused', finished='2030-01-02T00:00:00Z', name='Run_X.group')

    def case_changed():
        os.remove(f'{gdir}/_records/catalog/Run_X.group.json')
        open(ctx.grouping_table('I'), 'a').write('\n')

    def case_none():
        open(ctx.grouping_table('I'), 'w').write(tab0)
        shutil.rmtree(f'{gdir}/_records')
    for what, setup, want in (('the latest grouping run refused', case_refused, 'ended refused'),
                              ('the table changed since its run recorded it', case_changed,
                               'changed since its grouping run recorded it'),
                              ('no record of a grouping run', case_none, 'no record of a 10-minute grouping run')):
        setup()
        shutil.rmtree(f'{U}/source_catalogs_noveto_I_sb', ignore_errors=True)
        st, s, plan = run_build()
        check(f'F2: sub-LST build REFUSES when {what} (S-G18; S-G28 when the stage\'s one reading of the omission meets it '
              f'first)', ('S-G18' in st or 'S-G28' in st) and want in st, st)
    record()
    tab2 = rcsv(ctx.grouping_table('I'))[1]
    wcsv(ctx.grouping_table('I'), GS.DET_COLS, tab2[1:])
    record()                                              # a done run recorded this (smaller) table
    shutil.rmtree(f'{U}/source_catalogs_noveto_I_sb', ignore_errors=True)
    st, s, plan = run_build()
    check('sub-LST build: a selected row the grouping does not hold REFUSES (S-G18), naming the remedy',
          'S-G18' in st and '--group-lst-dir' in st, st)
    keep = (CFG.catalog_grouping.GROUPING, CFG.catalog_grouping.MULTIPLES_RULE)
    try:
        CFG.catalog_grouping.GROUPING = CFG.catalog_grouping.MULTIPLES_RULE = 'production'
        ctx.grouping_mode = GS.mode()
        st, s, plan = run_build()
        surv = {r['source_id'] for p in glob.glob(f'{U}/source_catalogs_noveto_I_sb/*_sources_survivors.csv') for r in rcsv(p)[1]}
        check("sub-LST build under production's mode (the parity file): the builder in place, no runner, its own sources",
              st == 'done' and not s.u.meta.get('grouping_builder') and not any('builder_grouping' in ln for ln in plan)
              and surv and not (surv & {'J0520+3000a'}), (st, surv))
    finally:
        CFG.catalog_grouping.GROUPING, CFG.catalog_grouping.MULTIPLES_RULE = keep
        CFG.catalog_grouping.BUILDER_SHA1, CFG.catalog_grouping.FROZEN_BUILDER_SHA1 = keep_pin, keep_fpin


# =====================================================================================================================
def test_one_rule_record():
    d = f'{T}/record'
    shutil.rmtree(d, ignore_errors=True)
    rows, tab = _stub_world(d)
    cat = f'{d}/source_catalogs_I_sb'
    # production's builder outcomes (production's own sources): A1+A2 one source below 7 unpromoted; B1+B2 one source promoted
    # REP; C1 sigma-stack removed; D1 survivor; E1 survivor, E2 below 7 promoted MB
    wcsv(f'{cat}/46MHz_sigma_stack_singleton_removed.csv', ['cand_id'], [dict(cand_id='I_05h_C1')])
    mem = [('SA', 'I_05h_A1'), ('SA', 'I_05h_A2'), ('SB', 'I_05h_B1'), ('SB', 'I_05h_B2'), ('SD', 'I_05h_D1'), ('SE', 'I_05h_E1')]
    wcsv(f'{cat}/46MHz_source_members.csv', ['source_id', 'band', 'cand_id'], [dict(source_id=s, band='46MHz', cand_id=c) for s, c in mem])
    wcsv(f'{cat}/50MHz_source_members.csv', ['source_id', 'band', 'cand_id'], [dict(source_id='SE2', band='50MHz', cand_id='I_05h_E2')])
    wcsv(f'{cat}/46MHz_sources_master.csv', ['source_id', 'band', 'lst_cells', 'best_snr', 'promotion'],
         [dict(source_id='SA', band='46MHz', lst_cells='05h', best_snr='6.4', promotion=''),
          dict(source_id='SB', band='46MHz', lst_cells='05h', best_snr='6.5', promotion='REP'),
          dict(source_id='SD', band='46MHz', lst_cells='05h', best_snr='9.0', promotion=''),
          dict(source_id='SE', band='46MHz', lst_cells='05h', best_snr='8.0', promotion='')])
    wcsv(f'{cat}/50MHz_sources_master.csv', ['source_id', 'band', 'lst_cells', 'best_snr', 'promotion'],
         [dict(source_id='SE2', band='50MHz', lst_cells='05h', best_snr='6.0', promotion='MB')])
    out, cnt = GS.one_rule_record(f'{d}/grouping_I/detections.csv', [cat], [f'{d}/selected_I_sb.csv'], 'I')
    v = {r['cand_id']: (r['served_today'], r['served_rule']) for r in out}
    check('one_rule_record: the multiple A is rescued (removed today below 7 sigma)', v['I_05h_A1'] == (0, 1)
          and v['I_05h_A2'] == (0, 1), v)
    check('one_rule_record: the single B, kept today by REP only, is dropped (repeat rungs follow the multiples level)',
          v['I_05h_B1'] == (1, 0) and v['I_05h_B2'] == (1, 0), v)
    check('one_rule_record: C stays removed (sigma-stack test), D served, E served with its MB-promoted band',
          v['I_05h_C1'] == (0, 0) and v['I_05h_D1'] == (1, 1) and v['I_05h_E1'] == (1, 1) and v['I_05h_E2'] == (1, 1), v)
    check('one_rule_record: the counts', cnt.get('rescued') == 2 and cnt.get('dropped') == 2, cnt)


# =====================================================================================================================
def test_carry():
    d = f'{T}/carry'
    shutil.rmtree(d, ignore_errors=True)
    rows, tab = _stub_world(d)
    jr = [dict(action='classify', scope='source', source_id='OLD_A1', judge_confidence='FP', judge_fp_reason='leakage',
               judge_fp_reason_2='', judge_note='n1'),
          dict(action='classify', scope='source', source_id='OLD_A2', judge_confidence='FP', judge_fp_reason='leakage',
               judge_fp_reason_2=''),
          dict(action='classify', scope='source', source_id='OLD_E1', judge_confidence='FP', judge_fp_reason='bilobe',
               judge_fp_reason_2=''),
          dict(action='classify', scope='source', source_id='OLD_E2', judge_confidence='REAL', judge_fp_reason='',
               judge_fp_reason_2=''),
          dict(action='classify', scope='source', source_id='OLD_GONE', judge_confidence='FP', judge_fp_reason='leakage',
               judge_fp_reason_2=''),
          dict(action='classify', scope='source', source_id='OLD_D', judge_confidence='FP', judge_fp_reason='leakage',
               judge_fp_reason_2=''),
          dict(action='retract', source_id='OLD_D')]
    os.makedirs(f'{d}/old')
    with open(f'{d}/old/judgements.jsonl', 'w') as f:
        for r in jr:
            f.write(json.dumps(r) + '\n')
    det = {r['cand_id']: dict(cand_id=r['cand_id'], band=r['band'], lst=r['lst'], date=r['date'], ra=float(r['ra']),
                              dec=float(r['dec'])) for r in rows}
    # OLD_A2's detection RE-NUMBERED by a re-search (another cand_id, the same window and position within the radius): the
    # carry matches by window and position (the deep carry's rule), never by cand_id alone
    members = {'OLD_A1': [det['I_05h_A1']], 'OLD_A2': [dict(det['I_05h_A2'], cand_id='I_05h_A2_RENUMBERED', ra=80.0013)],
               'OLD_E1': [det['I_05h_E1']], 'OLD_E2': [det['I_05h_E2']],
               'OLD_GONE': [dict(det['I_05h_D1'], date='2030-01-09')], 'OLD_D': [det['I_05h_D1']]}
    cnt = GS.carry_judgements(f'{d}/old/judgements.jsonl', members, f'{d}/grouping_I/detections.csv', f'{d}/new', 'Run_TEST',
                              carried_from_run='Run_OLD')
    rep = {r['new_source'] or r['old_sources']: r for r in rcsv(f'{d}/new/carry_report.csv')[1]}
    new = {json.loads(ln)['source_id']: json.loads(ln) for ln in open(f'{d}/new/judgements.jsonl')}
    check('carry: two old pages that agree carry UNCHANGED to their identity (FP/leakage), naming both',
          'J0520+3000a' in new and new['J0520+3000a']['judge_fp_reason'] == 'leakage'
          and new['J0520+3000a']['carried_from_source'] == 'OLD_A1;OLD_A2' and new['J0520+3000a']['group_identity_id'] == 'GTI-0000001'
          and 'original note: n1' in new['J0520+3000a']['judge_note'], new.get('J0520+3000a'))
    check('carry: disagreeing pages are a CONFLICT, reported and not carried', 'J0536+3400a' not in new
          and rep.get('J0536+3400a', {}).get('outcome') == 'CONFLICT - not carried', rep.get('J0536+3400a'))
    check('carry: a judgement reaching no detection is reported; a retracted one is not carried',
          rep.get('OLD_GONE', {}).get('outcome') == 'old judgement reached no new detection' and 'J0532+3300a' not in new, rep)
    check('carry: the counts', cnt['carried'] == 1 and cnt['conflicts'] == 1 and cnt['unmatched'] == 1, cnt)
    mc = [json.loads(ln) for ln in open(f'{d}/new/{GS.MERGE_CONFLICTS}')]
    check('carry: the CONFLICT is kept with BOTH verdicts for the page (merge_conflicts.jsonl: shown and marked AMBIGUOUS, '
          'never one picked; R-GREGG-20261124-CROSS-STOKES-LINK-TEST-APPROVED)',
          len(mc) == 1 and mc[0]['source_id'] == 'J0536+3400a' and mc[0]['group_identity_id'] == 'GTI-0000005'
          and mc[0]['origin'] == 'carry' and [(v['source'], v['judge_confidence'], v['judge_fp_reason']) for v in
                                              mc[0]['verdicts']] == [('OLD_E1', 'FP', 'bilobe'), ('OLD_E2', 'REAL', '')]
          and cnt['ambiguous_kept'] == 1, (mc, cnt))
    try:
        GS.carry_judgements(f'{d}/old/judgements.jsonl', members, f'{d}/grouping_I/detections.csv', f'{d}/new', 'Run_TEST')
        got = 'written'
    except GS.StageRefusal as e:
        got = str(e)
    check('carry: refuses to write over a non-empty journal (it only seeds)', 'only seeds' in got, got)
    # the next build: the unsettled conflict is carried, both verdicts listed again (never dropped by a rebuild)
    members2 = {'J0520+3000a': [det['I_05h_A1'], det['I_05h_A2']], 'J0536+3400a': [det['I_05h_E1'], det['I_05h_E2']]}
    cnt2 = GS.carry_judgements(f'{d}/new/judgements.jsonl', members2, f'{d}/grouping_I/detections.csv', f'{d}/new2',
                               'Run_TEST2', carried_from_run='Run_TEST')
    mc2 = [json.loads(ln) for ln in open(f'{d}/new2/{GS.MERGE_CONFLICTS}')]
    new2 = {json.loads(ln)['source_id'] for ln in open(f'{d}/new2/judgements.jsonl')}
    check('carry: an UNSETTLED merge conflict is carried to the next build with both verdicts, the page still unjudged',
          len(mc2) == 1 and mc2[0]['source_id'] == 'J0536+3400a' and mc2[0].get('carried_conflict_from') == 'J0536+3400a'
          and sorted((v['source'], v['judge_confidence']) for v in mc2[0]['verdicts']) == [('OLD_E1', 'FP'), ('OLD_E2', 'REAL')]
          and 'J0536+3400a' not in new2 and cnt2['conflicts_carried'] == 1, (mc2, cnt2))
    # a reviewer judges the page: the conflict is settled and not carried again
    with open(f'{d}/new2/judgements.jsonl', 'a') as f:
        f.write(json.dumps(dict(action='classify', scope='source', source_id='J0536+3400a', judge_confidence='REAL',
                                judge_fp_reason='', judge_fp_reason_2='', ts='2030-02-01T00:00:00Z')) + '\n')
    cnt3 = GS.carry_judgements(f'{d}/new2/judgements.jsonl', members2, f'{d}/grouping_I/detections.csv', f'{d}/new3',
                               'Run_TEST3', carried_from_run='Run_TEST2')
    mc3 = [json.loads(ln) for ln in open(f'{d}/new3/{GS.MERGE_CONFLICTS}') if ln.strip()]
    new3 = {json.loads(ln)['source_id']: json.loads(ln) for ln in open(f'{d}/new3/judgements.jsonl')}
    check('carry: a conflict settled by a verdict judged on the page is not carried; that verdict is',
          not mc3 and new3.get('J0536+3400a', {}).get('judge_confidence') == 'REAL' and cnt3['conflicts_carried'] == 0,
          (mc3, cnt3))
    try:
        GS.carry_judgements(f'{d}/old/judgements.jsonl', {'OLD_A1': ['I_05h_A1']}, f'{d}/grouping_I/detections.csv',
                            f'{d}/new_ids_only', 'Run_TEST')
        got = 'carried'
    except GS.StageRefusal as e:
        got = str(e)
    check('carry: an old member without its fingerprint (a bare cand_id) REFUSES (cand_ids are re-numbered by a re-search)',
          'fingerprint' in got, got)
    # the content tests of serve's journal class on the carried journal (production's report columns)
    from exopipe.phase3.lib.serve import tests as ST
    r1 = ST.parses_append_only(f'{d}/new/judgements.jsonl')
    r2 = ST.carry_consistent(f'{d}/new/judgements.jsonl')
    check("carry: the carried journal passes serve's judgement_journal content tests (parses, carry_consistent)",
          r1[0] and r2[0], (r1, r2))


def test_serve_carry():
    """serve --tier sublst under design B (lib/serve/journals.sublst_carry_journals / sublst_carry_between, wired in
    sublst_build.carry_swap_stage): the served run's journals CARRIED into the build by member detections (the page ids change
    to the identities' names), the lines journalled between the carry and the swap carried and appended, a retraction among
    them carried as a retraction"""
    from exopipe.phase3.lib.serve import journals
    d = f'{T}/serve_carry'
    shutil.rmtree(d, ignore_errors=True)
    rows, tab = _stub_world(f'{d}/w')
    old, B = f'{d}/Run_OLD', f'{d}/Run_NEW.build'
    # the served run (production's grouping): two pages, J0520+3000 (A1, A2: identity_id set) and J0532+3300 (D1)
    os.makedirs(f'{old}/source_catalogs_I')
    wcsv(f'{old}/source_catalogs_I/46MHz_sources_survivors.csv', ['source_id', 'identity_id', 'band', 'best_cand_id'],
         [dict(source_id='J0520+3000', identity_id='J0520+3000', band='46MHz', best_cand_id='I_05h_A2'),
          dict(source_id='J0532+3300', identity_id='', band='46MHz', best_cand_id='I_05h_D1')])
    byc = {r['cand_id']: r for r in rows}
    wcsv(f'{old}/source_catalogs_I/46MHz_source_members.csv', ['source_id', 'band', 'cand_id', 'det_ra', 'det_dec', 'det_date'],
         [dict(source_id=s_, band='46MHz', cand_id=c, det_ra=byc[c]['ra'], det_dec=byc[c]['dec'], det_date=byc[c]['date'])
          for s_, c in (('J0520+3000', 'I_05h_A1'), ('J0520+3000', 'I_05h_A2'), ('J0532+3300', 'I_05h_D1'))])
    j0 = [dict(action='classify', scope='source', source_id='J0520+3000', judge_confidence='FP', judge_fp_reason='leakage',
               judge_fp_reason_2='')]
    os.makedirs(f'{old}/judgements_I')
    with open(f'{old}/judgements_I/judgements.jsonl', 'w') as f:
        f.write(''.join(json.dumps(r) + '\n' for r in j0))
    # the build (design B): the per-tier catalogue the builder wrote with the grouping
    os.makedirs(f'{B}/source_catalogs_I_sb')
    wcsv(f'{B}/source_catalogs_I_sb/46MHz_sources_survivors.csv', ['source_id', 'identity_id', 'band', 'best_cand_id',
                                                                   'group_identity_id'],
         [dict(source_id='J0520+3000a', identity_id='J0520+3000a', band='46MHz', best_cand_id='I_05h_A2',
               group_identity_id='GTI-0000001'),
          dict(source_id='J0532+3300a', identity_id='J0532+3300a', band='46MHz', best_cand_id='I_05h_D1',
               group_identity_id='GTI-0000004')])
    wcsv(f'{B}/source_catalogs_I_sb/46MHz_source_members.csv', ['source_id', 'band', 'cand_id', 'det_ra', 'det_dec', 'det_date'],
         [dict(source_id=s_, band='46MHz', cand_id=c, det_ra=byc[c]['ra'], det_dec=byc[c]['dec'], det_date=byc[c]['date'])
          for s_, c in (('J0520+3000a', 'I_05h_A1'), ('J0520+3000a', 'I_05h_A2'), ('J0532+3300a', 'I_05h_D1'))])
    carried, marks = journals.sublst_carry_journals(old, B)
    new = [json.loads(l) for l in open(f'{B}/judgements_I/judgements.jsonl')]
    check('serve carry (design B): the served page J0520+3000 judgement reaches the identity J0520+3000a (by member detections)',
          carried.get('judgements_I', {}).get('carried') == 1 and len(new) == 1 and new[0]['source_id'] == 'J0520+3000a'
          and new[0]['carried_from_source'] == 'J0520+3000' and marks == {'judgements_I': 1}, (carried, new, marks))
    with open(f'{old}/judgements_I/judgements.jsonl', 'a') as f:      # journalled on the old server after the carry
        f.write(json.dumps(dict(action='classify', scope='source', source_id='J0532+3300', judge_confidence='REAL',
                                judge_fp_reason='', judge_fp_reason_2='')) + '\n')
        f.write(json.dumps(dict(action='retract', source_id='J0520+3000')) + '\n')
    os.rename(B, f'{d}/Run_NEW')
    app = journals.sublst_carry_between(old, f'{d}/Run_NEW', marks)
    new = [json.loads(l) for l in open(f'{d}/Run_NEW/judgements_I/judgements.jsonl')]
    last = {}
    for r in new:
        if r.get('action') == 'retract':
            last.pop(r['source_id'], None)
        else:
            last[r['source_id']] = r
    check('serve carry (design B): the between lines are carried and appended - the new judgement of J0532+3300 reaches '
          'J0532+3300a, the retraction of J0520+3000 retracts J0520+3000a',
          set(last) == {'J0532+3300a'} and last['J0532+3300a']['judge_confidence'] == 'REAL'
          and app.get('judgements_I', {}).get('retracted') == 1, (app, new))
    from exopipe.phase3.lib.serve import tests as ST
    r2 = ST.carry_consistent(f'{d}/Run_NEW/judgements_I/judgements.jsonl')
    check("serve carry: the report and the journal agree (serve's carry_consistent content test)", r2[0], r2)


# =====================================================================================================================
def test_refused_grouping():
    """review 2026-11-04 F2 (the reviewer's probe, RV/refused_probe/probe.py): a grouping run whose content tests REFUSE leaves
    the table every build reads and the state exactly as they were - its products go to grouping_<S>.refused_<UTC>; a run
    holding the state's lock makes a second one refuse"""
    mc, side = inputs('10min')
    det = GS.load_det('10min', 'V', mc, side)
    out = f'{T}/refused'
    shutil.rmtree(out, ignore_errors=True)
    p1, _s1, _r1 = run_stage('10min', 'V', det, out, out, 'probe.group', new=True, inp=mc)
    sha1 = GS.table_digest(f'{out}/grouping_V/detections.csv')
    st1 = open(GS.state_path(out, '10min', 'V')).read()
    try:
        GS.run_group(det, '10min', 'V', out, out, 'probe.group2', {}, check=lambda pp: ['T3 forced failure (probe)'])
        got = 'not refused'
    except GS.StageRefusal as e:
        got = str(e)
    refused = glob.glob(f'{out}/grouping_V.refused_*')
    check('F2: a refused grouping leaves the canonical table (sha1) and the state unchanged; its products are kept aside',
          'NOT advanced' in got and GS.table_digest(f'{out}/grouping_V/detections.csv') == sha1
          and open(GS.state_path(out, '10min', 'V')).read() == st1 and len(refused) == 1
          and not glob.glob(f'{out}/grouping_V.tmp.*'), (got, refused))
    with GS._Lock(GS.state_lock_path(out, '10min', 'V'), 'the test holder'):
        try:
            GS.run_group(det, '10min', 'V', out, out, 'probe.group3', {})
            got = 'not refused'
        except GS.StageRefusal as e:
            got = str(e)
    check('F2: a second grouping of the same state while one holds its lock REFUSES at once, nothing written',
          'held by another grouping run' in got and GS.table_digest(f'{out}/grouping_V/detections.csv') == sha1
          and open(GS.state_path(out, '10min', 'V')).read() == st1, got)
    p3, _s3, _r3 = run_stage('10min', 'V', det, out, out, 'probe.group4', mode='regroup', inp=mc)
    check('F2: after the lock is released a run succeeds; the previous table is kept as grouping_V.superseded_*',
          len(glob.glob(f'{out}/grouping_V.superseded_*')) == 1 and p3['detections_sha1'] == GS.table_digest(p3['detections']))


# =====================================================================================================================
def test_carry_between_window():
    """review 2026-11-04 F1 (the reviewer's probe, RV/carry_probe/probe.py): the lines journalled between the carry and the swap
    are carried by RE-DERIVING each identity they reach from the FULL old journal, in order. (1) classify / retract / classify
    of one old page ends judged as the last action says; (2) a label in the window that disagrees with a carried label (another
    old page reaching the same identity) is a CONFLICT row and a retract - never an override; (3) the same two labels in one
    seed carry: the same CONFLICT (the outcome does not depend on when a label was made); (4) an agreeing label re-carries
    with both pages named; (5) an identity judged on the new page is never overridden"""
    from exopipe.phase3.lib.serve import tests as ST
    D = f'{T}/carry_window'
    shutil.rmtree(D, ignore_errors=True)
    old = {'J0520+3000': [dict(cand_id='I1_05h_2030-01-01T03_46MHz_0001', band='46MHz', lst='05h', date='2030-01-01T03',
                               ra=80.0, dec=30.0)],
           'J0520+3001': [dict(cand_id='I1_05h_2030-01-02T03_46MHz_0002', band='46MHz', lst='05h', date='2030-01-02T03',
                               ra=80.0, dec=30.0)]}
    new = [dict(cand_id='I1_05h_2030-01-01T03_46MHz_0001', band='46MHz', lst='05h', date='2030-01-01T03', ra=80.0, dec=30.0,
                name='J0520+3000a', id='GTI-0000001'),
           dict(cand_id='I1_05h_2030-01-02T03_46MHz_0002', band='46MHz', lst='05h', date='2030-01-02T03', ra=80.0, dec=30.0,
                name='J0520+3000a', id='GTI-0000001')]

    def cl(sid, conf, reason='', note=''):
        return json.dumps(dict(action='classify', scope='source', source_id=sid, judge_confidence=conf, judge_fp_reason=reason,
                               **({'judge_note': note} if note else {})))

    def rt(sid):
        return json.dumps(dict(action='retract', scope='source', source_id=sid))

    def final(path):
        return {k: (v.get('judge_confidence'), v.get('judge_fp_reason'), 'carried_from_run' in v)
                for k, v in GS._latest_state(path).items()}

    def carry(case, seed, between, own=None):
        d = f'{D}/{case}'
        os.makedirs(d)
        open(f'{d}/old.jsonl', 'w').write(''.join(x + '\n' for x in seed))
        GS.carry_judgements(f'{d}/old.jsonl', old, new, f'{d}/new', 'NEW', carried_from_run='OLD', lines=list(seed))
        if own:                                    # a judgement made on the new page itself
            with open(f'{d}/new/judgements.jsonl', 'a') as f:
                f.write(own + '\n')
        full = list(seed) + list(between)
        open(f'{d}/old.jsonl', 'w').write(''.join(x + '\n' for x in full))
        c = GS.carry_judgements(f'{d}/old.jsonl', old, new, f'{d}/new', 'NEW', carried_from_run='OLD', lines=full, append=True,
                                since=len(seed))
        rep = rcsv(f'{d}/new/carry_report.csv')[1]
        ok_ = ST.carry_consistent(f'{d}/new/judgements.jsonl')
        return c, final(f'{d}/new/judgements.jsonl'), rep, ok_
    c, f, rep, ok_ = carry('c1', [], [cl('J0520+3000', 'real'), rt('J0520+3000'), cl('J0520+3000', 'FP', 'leakage')])
    check('F1 (1): classify / retract / classify in the window ends judged as the last action says (FP / leakage)',
          f == {'J0520+3000a': ('FP', 'leakage', True)} and c['carried'] == 1 and c['retracted'] == 0 and ok_[0], (c, f, ok_))
    c, f, rep, ok_ = carry('c2', [cl('J0520+3001', 'real')], [cl('J0520+3000', 'FP', 'leakage')])
    last = [r for r in rep if r['new_source'] == 'J0520+3000a'][-1]
    check('F1 (2): a label in the window that disagrees with a carried label is a CONFLICT row and a retract, never an '
          'override', f == {} and last['outcome'] == 'CONFLICT - not carried' and c['conflicts'] == 1 and c['retracted'] == 1
          and ok_[0], (c, f, rep, ok_))
    c, f, rep, ok_ = carry('c3', [cl('J0520+3001', 'real'), cl('J0520+3000', 'FP', 'leakage')], [])
    check('F1 (3): the same two labels in one seed carry give the same CONFLICT (timing-independent)',
          f == {} and [r['outcome'] for r in rep] == ['CONFLICT - not carried'] and ok_[0], (f, rep, ok_))
    c, f, rep, ok_ = carry('c4', [cl('J0520+3001', 'FP', 'leakage')], [cl('J0520+3000', 'FP', 'leakage')])
    nj = [json.loads(l) for l in open(f'{D}/c4/new/judgements.jsonl')]
    check('F1 (4): an agreeing label in the window re-carries the identity, naming both old pages',
          f == {'J0520+3000a': ('FP', 'leakage', True)} and nj[-1]['carried_from_source'] == 'J0520+3000;J0520+3001'
          and ok_[0], (f, nj[-1], ok_))
    own = json.dumps(dict(action='classify', scope='source', source_id='J0520+3000a', judge_confidence='real',
                          judge_fp_reason=''))
    c, f, rep, ok_ = carry('c5', [cl('J0520+3001', 'FP', 'leakage')], [cl('J0520+3000', 'FP', 'sidelobe')], own=own)
    check('F1 (5): an identity judged on the new page is never overridden by the window carry (reported)',
          f == {'J0520+3000a': ('real', '', False)} and c['judged_on_new_page'] == 1
          and rep[-1]['outcome'] == 'not carried: judged on the new page' and ok_[0], (c, f, rep[-1:], ok_))
    # F1 residual (CUTOVER_READINESS 2026-11-04): a note changed in the window with the verdict unchanged re-carries the
    # identity (the carried note quotes the new note); the same label sent again with the same note writes nothing
    c, f, rep, ok_ = carry('c8', [cl('J0520+3000', 'FP', 'leakage', 'first note')],
                           [cl('J0520+3000', 'FP', 'leakage', 'second note')])
    n8 = [json.loads(l) for l in open(f'{D}/c8/new/judgements.jsonl')]
    check('F1 residual: a note changed in the window with the verdict unchanged is carried (the last record quotes it)',
          c['carried'] == 1 and c['unchanged'] == 0 and len(n8) == 2 and n8[-1]['judge_note'].endswith(
              'original note: second note') and f == {'J0520+3000a': ('FP', 'leakage', True)} and ok_[0], (c, n8, ok_))
    c, f, rep, ok_ = carry('c9', [cl('J0520+3000', 'FP', 'leakage', 'same note')],
                           [cl('J0520+3000', 'FP', 'leakage', 'same note')])
    n9 = [json.loads(l) for l in open(f'{D}/c9/new/judgements.jsonl')]
    check('F1 residual: the same verdict and the same note sent again in the window write nothing (unchanged)',
          c['carried'] == 0 and c['unchanged'] == 1 and len(n9) == 1 and ok_[0], (c, n9, ok_))
    # the review server keys a page's state by (source_id, presentation_id): a judgement made on a re-queue presentation (p2)
    # carries as p1 (production's carry tools: deep_carry_judgements.py:106), and the retract a conflict writes names p1
    p2 = json.dumps(dict(json.loads(cl('J0520+3001', 'real')), presentation_id='p2'))
    c, f, rep, ok_ = carry('c6', [p2], [])
    n6 = [json.loads(l) for l in open(f'{D}/c6/new/judgements.jsonl')]
    c, f, rep, ok_ = carry('c7', [p2], [cl('J0520+3000', 'FP', 'leakage')])
    n7 = [json.loads(l) for l in open(f'{D}/c7/new/judgements.jsonl')]
    check('carry: a judgement of presentation p2 carries as p1; the retract of a conflict names p1 (the server\'s state key)',
          [r.get('presentation_id') for r in n6] == ['p1'] and [(r.get('action'), r.get('presentation_id')) for r in n7]
          == [('classify', 'p1'), ('retract', 'p1')], (n6, n7))


# =====================================================================================================================
def test_grouping_mix():
    """review 2026-11-04 F3 and the per-hour form of it: serve --tier sublst under design B refuses a cumulative whose LST
    builds do not all stand on the PRESENT grouping table - a directory that names it passes; one that names another table
    passes only when the gate itself certifies it from the directory's own record (lib/catalog/delta_build.standing);
    an LST built without the design-B grouping is named - lib/serve/sublst_build.grouping_mix. PR 51 F3: a directory that
    names the present table is certified too (its population checked); F4: a detection the record lists as removed before
    the builder is an accepted absence, an unlisted one refuses"""
    import hashlib
    from exopipe.phase3.lib.serve import sublst_build as SB
    from exopipe.phase3.lib.catalog import delta_build as DB
    from exopipe.phase3.lib.catalog.wraps import builder_grouping as BG
    R = f'{T}/mix'
    shutil.rmtree(R, ignore_errors=True)
    head = 'cand_id,stokes,band,lst,identity_id,identity_name,multiples_id,multiples_name,multiplicity,multiple,date'

    def table(name, dets):
        """dets: (cand_id, lst, identity_id); Stokes I, 46MHz"""
        p = f'{R}/{name}/grouping_I/detections.csv'
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w', newline='') as f:
            f.write('\n'.join([head] + [f'{c},I,46MHz,{l},{g},J{g},M{g},J{g},1,0,2030-01-01T00' for c, l, g in dets]) + '\n')
        return p, DB.read_grouping(p)
    old = [('a1', '05h', 'G1'), ('a2', '05h', 'G2'), ('b1', '06h', 'G3'), ('c1', '07h', 'G4')]
    t_old, g_old = table('old', old)
    t_now, g_now = table('now', old + [('c2', '07h', 'G5')])            # a new night's detection at 07h

    def lst(L, g=None, built=None, bind=True, vetoed=(), removed=None):
        """an all-nights catalog directory of LST L built under grouping g, from the detections of ``built`` (its record);
        ``removed``: the record's detections removed before the builder ({cand_id: reason}; None = no such field)"""
        d = f'{R}/Run_SLST{L[:2]}B/source_catalogs_I_sb'
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        if g is None:
            return d
        if built is not None:
            Tb, mg = g[0], g[1]
            rec = dict(version=2, grouping_table=dict(path='x', sha1=g[2]), vetoed=sorted(vetoed),
                       rows={c: [Tb[c]['identity_id'], BG.group_digest_of(mg[Tb[c]['multiples_id']])] for c in built
                             if c not in vetoed})
            if removed is not None:
                rec['removed'] = removed
            json.dump(rec, open(f'{d}/builder_frozen.json', 'w'))
        prov = dict(table='/x/grouping_I/detections.csv', table_sha1=g[2])
        if built is not None and bind:
            prov['frozen_record_sha256'] = hashlib.sha256(open(f'{d}/builder_frozen.json', 'rb').read()).hexdigest()
        json.dump(prov, open(f'{d}/grouping_table.json', 'w'))
        return d
    orig = SB.lst_run
    SB.lst_run = lambda l: f'{R}/Run_SLST{l[:2]}B'
    calls = []

    def present(S):
        calls.append(S)
        return t_now, g_now
    try:
        lst('05h', g_now, built=['a1', 'a2'])
        lst('06h', g_now, built=['b1'])
        bad, tables, how = SB.grouping_mix(['05h', '06h'], present)
        check('F3 gate: two LST builds on the present grouping table pass, each certified here too (the table read once)',
              not bad and tables == {'I': {g_now[2]: ['05h', '06h']}} and [c['lst'] for c in how['certified']] == ['05h', '06h']
              and calls == ['I']
              and how['present']['I']['sha1'] == g_now[2] and how['present']['I']['sha256'] == g_now[3], (bad, tables, how))
        lst('05h', g_now, built=['a1'])
        bad, tables, how = SB.grouping_mix(['05h', '06h'], present)
        check('review 51 F3: a directory that names the present table but whose record lacks a detection of the hour refuses '
              '(no skip on its own provenance)',
              len(bad) == 1 and bad[0][0] == '05h' and 'names the present grouping table' in bad[0][1]
              and '1 are not in the record (a2)' in bad[0][1] and [c['lst'] for c in how['certified']] == ['06h'], bad)
        lst('05h', g_now, built=['a1'], bind=False)
        bad, tables, how = SB.grouping_mix(['05h', '06h'], present)
        check('review 51 F3: ... and one naming the present table whose record is not bound by sha256 refuses',
              len(bad) == 1 and bad[0][0] == '05h' and 'names no sha256' in bad[0][1], bad)
        lst('05h', g_now, built=['a1'], removed={'a2': 'selected_all_ephem_removed.csv: moon 0.120 deg < 0.25 deg'})
        bad, tables, how = SB.grouping_mix(['05h', '06h'], present)
        c5 = [c for c in how['certified'] if c['lst'] == '05h']
        check('review 51 F4: a detection of the hour the record lists as removed before the builder is an accepted absence',
              not bad and len(c5) == 1 and c5[0]['detections'] == 1, (bad, how))
        lst('05h', g_now, built=['a1'], removed={'zz': 'selected_all_removed.csv: below the S/N floor'})
        bad, tables, how = SB.grouping_mix(['05h', '06h'], present)
        check('review 51 F4: an absence the record does not list refuses (a listed removal of another detection excuses '
              'nothing)', len(bad) == 1 and bad[0][0] == '05h' and '(a2)' in bad[0][1], bad)
        lst('05h', g_now, built=['a1'], removed=['a2'])
        bad, tables, how = SB.grouping_mix(['05h', '06h'], present)
        check('review 51 F4: a removed list that is not a mapping refuses',
              len(bad) == 1 and bad[0][0] == '05h' and 'not a mapping' in bad[0][1], bad)
        lst('05h', g_now, built=['a1', 'a2'])
        lst('06h', g_old, built=['b1'])
        bad, tables, how = SB.grouping_mix(['05h', '06h'], present)
        c = ([c for c in how['certified'] if c['lst'] == '06h'] or [{}])[0]
        check('gate: an LST built under an earlier table whose hour the present table leaves as it was is certified here',
              not bad and len(how['certified']) == 2 and c['lst'] == '06h' and c['key'] == 'I_sb' and c['detections'] == 1
              and c['record'] == 'builder_frozen.json' and c['table_to']['sha1'] == g_now[2] and len(c['record_sha256']) == 64
              and tables == {'I': {g_now[2]: ['05h'], g_old[2]: ['06h']}}, (bad, how))
        lst('07h', g_old, built=['c1'])
        bad, tables, how = SB.grouping_mix(['05h', '06h', '07h'], present)
        check('gate: an LST whose hour gained a detection in the present table refuses, with both directions counted',
              len(bad) == 1 and bad[0][0] == '07h' and 'does not stand' in bad[0][1] and '1 are not in the record (c2)' in bad[0][1]
              and "of the record's 1, 0 are not among them" in bad[0][1] and len(how['certified']) == 2, bad)
        lst('06h', g_old, built=['b1'], bind=False)
        bad, tables, how = SB.grouping_mix(['05h', '06h'], present)
        check('gate: a record its directory\'s provenance does not name by sha256 certifies nothing',
              len(bad) == 1 and bad[0][0] == '06h' and 'names no sha256' in bad[0][1]
              and [c['lst'] for c in how['certified']] == ['05h'], bad)
        lst('06h', g_old)
        bad, tables, how = SB.grouping_mix(['05h', '06h'], present)
        check('gate: an LST built under another table without a record refuses (the cumulative joins LST runs by identity name)',
              len(bad) == 1 and bad[0][0] == '06h' and g_old[2][:12] in bad[0][1] and g_now[2][:12] in bad[0][1]
              and 'missing or unreadable' in bad[0][1], bad)
        lst('06h', g_old)
        lst('05h', g_old)
        bad, tables, how = SB.grouping_mix(['05h', '06h'], present)
        check('gate: LSTs that agree with each other on an EARLIER table do not pass by agreeing (the present table decides)',
              len(bad) == 2 and tables == {'I': {g_old[2]: ['05h', '06h']}}, bad)
        lst('05h', g_now, built=['a1', 'a2'])
        lst('08h')
        bad, tables, how = SB.grouping_mix(['05h', '08h'], present)
        check('F3 gate: an LST built without the design-B grouping (no provenance) is named',
              len(bad) == 1 and bad[0][0] == '08h' and 'no grouping_table.json' in bad[0][1], bad)

        def refused(S):
            raise GS.StageRefusal('the latest 10-minute grouping run of Stokes I ended refused - REFUSING')
        bad, tables, how = SB.grouping_mix(['05h', '06h'], refused)
        check('gate: a present table that cannot be read refuses once per Stokes',
              len(bad) == 1 and bad[0][0] == '*' and 'cannot be read' in bad[0][1] and how['present'] == {}, bad)
        # review of PR 51: F6 a malformed table or record is a refusal, not a crash; F5 a missing directory of an hour and
        # key the present table holds detections of refuses; F10 the population follows the configured split
        def broken(S):
            raise ValueError('not valid UTF-8')
        bad, tables, how = SB.grouping_mix(['05h', '06h'], broken)
        check('review 51: a present table that raises ValueError refuses (no traceback)',
              len(bad) == 1 and bad[0][0] == '*' and 'not valid UTF-8' in bad[0][1], bad)
        lst('06h', g_old, built=['b1'])
        p = f'{R}/Run_SLST06B/source_catalogs_I_sb/builder_frozen.json'
        json.dump(dict(version=2, grouping_table=dict(path='x', sha1=g_old[2]), vetoed=[], rows=[['b1']]), open(p, 'w'))
        json.dump(dict(table='/x', table_sha1=g_old[2], frozen_record_sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest()),
                  open(f'{R}/Run_SLST06B/source_catalogs_I_sb/grouping_table.json', 'w'))
        bad, tables, how = SB.grouping_mix(['05h', '06h'], present)
        check('review 51: a bound record of the wrong form refuses (no traceback)',
              len(bad) == 1 and bad[0][0] == '06h' and 'does not stand' in bad[0][1]
              and [c['lst'] for c in how['certified']] == ['05h'], bad)
        lst('06h', g_now, built=['b1'])
        shutil.rmtree(f'{R}/Run_SLST07B', ignore_errors=True)
        bad, tables, how = SB.grouping_mix(['05h', '06h', '07h'], present)
        check('review 51: an LST whose run lacks the directory of a key the present table holds detections of refuses',
              len(bad) == 1 and bad[0][0] == '07h' and 'source_catalogs_I_sb is missing' in bad[0][1], bad)
        bad, tables, how = SB.grouping_mix(['05h', '06h', '09h'], present)
        check('review 51: ... an hour the present table holds no detection of needs no directory', not bad, bad)
        check('review 51: the key classes follow the configured split (a band of neither class is in no key)',
              DB.key_class('sb')('46MHz') and not DB.key_class('sb')('Red') and DB.key_class('wb')('Red')
              and DB.key_class('wb')('Full') and not DB.key_class('wb')('46MHz') and not DB.key_class('wb')('Odd'))
        # review 51 F4: what a builder record lists as removed before the builder - the table's detections the build did not
        # read that the selection's or the ephemeris filter's removal table of a run directory names, with the reason
        Ua, Un = f'{R}/U_all', f'{R}/U_night'
        for d, name, rows in ((Ua, 'selected_all_removed.csv', [('a2', 'below the S/N floor'), ('zz', 'not in the table')]),
                              (Un, 'selected_all_ephem_removed.csv', [('c2', 'moon 0.120 deg < 0.25 deg'), ('a1', 'read')])):
            os.makedirs(d, exist_ok=True)
            with open(f'{d}/{name}', 'w', newline='') as f:
                f.write('cand_id,snr,removal_reason\n' + ''.join(f'{c},5.0,{r}\n' for c, r in rows))
        rm = BG.removed_before_builder([Ua, Un], {'a1', 'b1', 'c1'}, g_now[0])
        check('review 51 F4: removed_before_builder lists the table\'s unread detections a removal table names, with the reason '
              '(a detection the build read, or one not in the table, is not listed)',
              rm == {'a2': 'selected_all_removed.csv: below the S/N floor',
                     'c2': 'selected_all_ephem_removed.csv: moon 0.120 deg < 0.25 deg'}, rm)
        check('review 51 F4: ... and nothing when no run directory holds a removal table',
              BG.removed_before_builder([f'{R}/Run_SLST05B'], {'a1'}, g_now[0]) == {})
        env0 = GS.builder_env(t_now, '10min', sha1=g_now[2])
        env1 = GS.builder_env(t_now, '10min', sha1=g_now[2], frozen='delta', frozen_record='/x/builder_frozen.json',
                              removed_dirs=[Ua, Un])
        old_env = os.environ.get(BG.REMOVED_DIRS_ENV)
        try:
            os.environ[BG.REMOVED_DIRS_ENV] = env1[BG.REMOVED_DIRS_ENV]
            dirs1 = BG.removed_dirs('selected_I_sb.csv')
            os.environ[BG.REMOVED_DIRS_ENV] = env0[BG.REMOVED_DIRS_ENV]
            dirs0 = BG.removed_dirs(f'{Ua}/selected_I_sb.csv')
        finally:
            if old_env is None:
                os.environ.pop(BG.REMOVED_DIRS_ENV, None)
            else:
                os.environ[BG.REMOVED_DIRS_ENV] = old_env
        check('review 51 F4: builder_env always sets the removal directories (empty = the input\'s own directory; a delta '
              'build names the all-nights run and its nights)',
              env0[BG.REMOVED_DIRS_ENV] == '' and dirs0 == [Ua] and dirs1 == [Ua, Un], (env0, dirs0, dirs1))
    finally:
        SB.lst_run = orig


# =====================================================================================================================
def test_identity_writeback():
    """the deep identity stage under design B (grouping_stage.identity_writeback): the four identity columns from the grouping
    per Stokes into every tier file and the union; a row not named by its design-B identity REFUSES; the identity_table content
    test checks the design-B identity (and catches a wrong row)"""
    from exopipe.phase3.lib.catalog import tests as CT
    d = f'{T}/identity'
    shutil.rmtree(d, ignore_errors=True)
    rows, tab = _stub_world(f'{d}/w')
    nm = {t['cand_id']: t['identity_name'] for t in tab}
    cols = ['rank', 'source_id', 'identity_id', 'identity_bands', 'band', 'best_cand_id', 'best_snr']
    wcsv(f'{d}/run/source_catalogs_I_sb/46MHz_sources_survivors.csv', cols,
         [dict(rank='rank0001', source_id=nm[c], identity_id=nm[c], identity_bands='46MHz', band='46MHz', best_cand_id=c,
               best_snr='9') for c in ('I_05h_A2', 'I_05h_D1', 'I_05h_E1')])
    wcsv(f'{d}/run/source_catalogs_I_sb/50MHz_sources_survivors.csv', cols,
         [dict(rank='rank0001', source_id=nm['I_05h_E2'], identity_id=nm['I_05h_E2'], identity_bands='50MHz', band='50MHz',
               best_cand_id='I_05h_E2', best_snr='6')])
    wcsv(f'{d}/run/source_catalogs_I_wb/Red_sources_survivors.csv', cols,
         [dict(rank='rank0001', source_id=nm['I_05h_D1'], identity_id=nm['I_05h_D1'], identity_bands='Red', band='Red',
               best_cand_id='I_05h_D1', best_snr='9')])
    union = []
    for p in sorted(glob.glob(f'{d}/run/source_catalogs_I_*/*_sources_survivors.csv')):
        union += [dict(r, stokes='I') for r in rcsv(p)[1]]
    for b in ('46MHz', '50MHz', 'Red'):
        wcsv(f'{d}/run/source_catalogs/{b}_sources_survivors.csv', cols + ['stokes'], [r for r in union if r['band'] == b])
    tier = set(glob.glob(f'{d}/run/source_catalogs_I_*/*_sources_survivors.csv'))
    uni = set(glob.glob(f'{d}/run/source_catalogs/*_sources_survivors.csv'))
    tables = {'I': f'{d}/w/grouping_I/detections.csv'}
    cnt = GS.identity_writeback(tier, uni, tables)
    rd = {(r['source_id'], r['band']): r for p in tier for r in rcsv(p)[1]}
    e = rd[(nm['I_05h_D1'], 'Red')]
    check('identity (design B): the columns written from the grouping (D in 46MHz sb + Red wb: one identity, both tiers)',
          e['identity_id'] == nm['I_05h_D1'] and e['identity_bands'] == '46MHz+Red' and e['identity_tiers'] == 'I_sb+I_wb'
          and e['identity_n'] == '2' and cnt['identities'] == {'I': 3} and cnt['cross_tier'] == {'I': 1}, (e, cnt))
    ur = {(r['source_id'], r['band']): r for p in uni for r in rcsv(p)[1]}
    check('identity (design B): the union rows carry the same columns', ur[(nm['I_05h_D1'], '46MHz')]['identity_tiers'] ==
          'I_sb+I_wb', ur)
    res = [CT.identity_columns(p, grouping='design-b', tables=tables) for p in sorted(tier | uni)]
    check('identity_table content test (design B) passes on every written file', all(r[0] for r in res), res)
    p0 = f'{d}/run/source_catalogs_I_sb/46MHz_sources_survivors.csv'
    cols_, rr = rcsv(p0)
    rr[0]['identity_id'] = 'J9999+9999'
    wcsv(p0, cols_, rr)
    r = CT.identity_columns(p0, grouping='design-b', tables=tables)
    check('identity_table content test (design B) catches a row not named by its identity', not r[0], r)
    try:
        GS.identity_writeback(tier, uni, tables)
        got = 'written'
    except GS.StageRefusal as e_:
        got = str(e_)
    check('identity (design B): a row not named by its design-B identity REFUSES the stage', 'not named by their design-B '
          'identity' in got, got)


# =====================================================================================================================
def test_tenmin_input():
    """the 10-minute grouping input (grouping_stage.assemble_tenmin_input): every LST's master_catalog.csv and window tie
    concatenated - a published run's layout (pertie_detections.csv at the top) or a catalogue directory's (pertie/) given by
    --group-lst-dir; a cand_id in two LSTs and an LST outside --lsts refuse"""
    d = f'{T}/tenmin_input'
    shutil.rmtree(d, ignore_errors=True)
    mc_cols = ['cand_id', 'lst', 'stokes', 'snr']
    ca, cb = 'I1_05h_2030-01-01T05_46MHz_0001', 'I1_06h_2030-01-01T06_46MHz_0001'
    wcsv(f'{d}/run05/master_catalog.csv', mc_cols, [dict(cand_id=ca, lst='05h', stokes='I', snr='8')])
    wcsv(f'{d}/run05/pertie_detections.csv', ['cand_id', 'sys_arcmin'], [dict(cand_id=ca, sys_arcmin='0.5')])
    wcsv(f'{d}/cat06/master_catalog.csv', mc_cols, [dict(cand_id=cb, lst='06h', stokes='I', snr='9')])
    wcsv(f'{d}/cat06/pertie/pertie_detections.csv', ['cand_id', 'sys_arcmin'], [dict(cand_id=cb, sys_arcmin='0.4')])
    om = GS.omission_of(omission_dir(f'{d}/keep', cover=[f'{d}/run05/master_catalog.csv', f'{d}/cat06/master_catalog.csv']))
    mc, pt, runs = GS.assemble_tenmin_input(['05h', '06h'], f'{d}/out', lst_dirs={'05h': f'{d}/run05', '06h': f'{d}/cat06'},
                                            omission=om)
    check('10-minute input: both layouts concatenated in LST order (published run; catalog directory)',
          [r['cand_id'] for r in rcsv(mc)[1]] == [ca, cb] and [r['cand_id'] for r in rcsv(pt)[1]] == [ca, cb],
          (rcsv(mc)[1], rcsv(pt)[1]))
    # S-G28: the OMIT cells' rows (production-form cand_ids) are absent from both inputs and recorded
    c1, c2, c3 = 'I_05h_2030-01-01T05_46MHz_0001', 'I_05h_2030-01-02T05_46MHz_0002', 'V1_05h_2030-01-01T05_46MHz_0003'
    wcsv(f'{d}/run07/master_catalog.csv', mc_cols, [dict(cand_id=c, lst='05h', stokes=c[0], snr='8') for c in (c1, c2, c3)])
    wcsv(f'{d}/run07/pertie_detections.csv', ['cand_id', 'sys_arcmin'], [dict(cand_id=c, sys_arcmin='0.5') for c in (c1, c2, c3)])
    omx = GS.omission_of(omission_dir(f'{d}/omit', omit=[('05h', '46MHz', '2030-01-01', 'I')],
                                      cover=[f'{d}/run07/master_catalog.csv']))
    mc, pt, _r = GS.assemble_tenmin_input(['05h'], f'{d}/out3', lst_dirs={'05h': f'{d}/run07'}, omission=omx)
    dr = rcsv(f'{d}/out3/grouping_input/{GS.OMISSION_DROPPED}')[1]
    check('10-minute input (S-G28): the OMIT cell\'s rows (that LST, band, night and Stokes) are absent from the catalog and '
          'the window tie, the other Stokes and nights kept; the dropped rows recorded per file',
          [r['cand_id'] for r in rcsv(mc)[1]] == [c2, c3] and [r['cand_id'] for r in rcsv(pt)[1]] == [c2, c3]
          and sorted((r['cand_id'], r['lst'], r['band'], r['night'], r['stokes']) for r in dr)
          == [(c1, '05h', '46MHz', '2030-01-01', 'I')] * 2 and sum(omx.dropped.values()) == 2,
          (rcsv(mc)[1], dr, omx.dropped))
    for what, kw, text in (('no decision table', dict(omission=None), 'no deep-omission decision table'),
                           ('a decision table that does not decide the LST\'s cells', dict(omission=GS.omission_of(
                               omission_dir(f'{d}/k6', lsts=['06h']))), '05h/46MHz/2030-01-01/I, 05h/46MHz/2030-01-01/V'),
                           ('a decision table without one night\'s cells', dict(omission=GS.omission_of(omission_dir(
                               f'{d}/k7', cover=[f'{d}/run07/master_catalog.csv'], lsts=['05h']))), None)):
        if text is None:                          # the coverage without the night 2030-01-02 (a night newer than the table)
            k7 = f'{d}/k7/deep_omission.csv'
            _bgm().write_cells(k7, {k: 1 for k in cells_of(f'{d}/run07/master_catalog.csv') if k[2] != '2030-01-02'})
            kw, text = dict(omission=GS.omission_of(f'{d}/k7')), 'night(s) 05h 2030-01-02'
        try:
            GS.assemble_tenmin_input(['05h'], f'{d}/out4', lst_dirs={'05h': f'{d}/run07'}, **kw)
            got = 'assembled'
        except GS.StageRefusal as e:
            got = str(e)
        check(f'10-minute input (S-G28): {what} REFUSES, never "no omission" or kept, naming what is missing',
              'S-G28' in got and 'REFUSING' in got and text in got, got)
    wcsv(f'{d}/cat06/master_catalog.csv', mc_cols, [dict(cand_id=ca, lst='06h', stokes='I', snr='9')])
    for what, kw in (('a cand_id in two LSTs', dict(lst_dirs={'05h': f'{d}/run05', '06h': f'{d}/cat06'})),
                     ('an LST outside --lsts', dict(lst_dirs={'05h': f'{d}/run05', '07h': f'{d}/cat06'}))):
        try:
            GS.assemble_tenmin_input(['05h', '06h'] if '07h' not in kw['lst_dirs'] else ['05h'], f'{d}/out2', omission=om, **kw)
            got = 'assembled'
        except GS.StageRefusal as e:
            got = str(e)
        check(f'10-minute input: {what} REFUSES', 'REFUSING' in got, got)


# =====================================================================================================================
def cli(args, timeout=1800):
    env = dict(os.environ, PYTHONPATH=REPO)
    p = subprocess.run([PY, '-m', 'exopipe.phase3'] + args, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def test_cli():
    out = f'{T}/cli/grouping_sublst'
    common = ['catalog', '--tier', 'sublst', '--stages', 'group', '--group-input', f'{FROZEN}/tenmin', '--out', out,
              '--cores', 'none', '--group-stokes', 'V', '--group-omission',
              omission_dir(f'{T}/cli/omit', cover=[f'{FROZEN}/tenmin/master_catalog.csv', f'{FROZEN}/tenmin/pertie_detections.csv'])]
    rc, txt = cli(common)
    check('cli: the first 10-minute grouping without --group-new-state REFUSES (exit 2)', rc == 2 and 'new=True' in txt,
          txt[-400:])
    rc, txt = cli(common + ['--group-new-state'])
    recp = sorted(glob.glob(f'{out}/_records/catalog/Run_SLSTB.group.json'))
    rec = json.load(open(recp[0])) if recp else {}
    check('cli: --group-new-state groups the cumulative (exit 0; unit done; content tests passed)',
          rc == 0 and rec.get('status') == 'done' and rec.get('tests') and all(t['passed'] for t in rec['tests']), txt[-400:])
    tab, sha = GS.recorded_table(out, 'V')
    check('cli: the record holds the table and its sha1 - what the LST builds check (F2)',
          rec.get('meta', {}).get('tables', {}).get('V', {}).get('sha1') == sha == GS.table_digest(f'{out}/grouping_V/detections.csv'),
          rec.get('meta', {}).get('tables'))
    ids0 = {r['cand_id']: r['identity_id'] for r in rcsv(f'{out}/grouping_V/detections.csv')[1]}
    rc, txt = cli(common)
    ids1 = {r['cand_id']: r['identity_id'] for r in rcsv(f'{out}/grouping_V/detections.csv')[1]}
    check('cli: a second run (incremental, the same detections) carries every identity ID', rc == 0 and ids0 == ids1, txt[-300:])
    # F2: two 10-minute groupings at once - the second REFUSES on the unit lock, writing nothing
    sha_before = GS.table_digest(f'{out}/grouping_V/detections.csv')
    with GS._Lock(os.path.join(out, GS.UNIT_LOCK), 'the test holder'):
        rc, txt = cli(common)
    rec = json.load(open(recp[0])) if recp else {}
    check('cli (F2): a second 10-minute grouping while one holds the lock REFUSES (exit 2), the table unchanged',
          rc == 2 and rec.get('status') == 'refused' and 'held by another grouping run' in (rec.get('reason') or '')
          and GS.table_digest(f'{out}/grouping_V/detections.csv') == sha_before, (rc, rec.get('reason')))
    try:
        GS.recorded_table(out, 'V')
        got = 'accepted'
    except GS.StageRefusal as e:
        got = str(e)
    check('cli (F2): after that refusal the builds refuse until a grouping run ends done', 'ended refused' in got, got)
    rc, txt = cli(common)
    check('cli (F2): the next run (lock free) ends done and the builds accept its table again',
          rc == 0 and GS.recorded_table(out, 'V')[1] == GS.table_digest(f'{out}/grouping_V/detections.csv'), txt[-300:])
    rc, txt = cli(common + ['--set', 'catalog_grouping.GROUPING=production', '--set', 'catalog_grouping.MULTIPLES_RULE=production'])
    rec = json.load(open(recp[0])) if recp else {}
    check("cli: under production's mode the unit is no-input (exit 3), naming the parity file",
          rc == 3 and rec.get('status') == 'no-input' and 'no grouping stage' in (rec.get('reason') or ''), (rc, rec.get('reason')))


def test_omission():
    """S-G28 through the unit on the frozen 10-minute subset (production-form cand_ids): the grouping after the deep omission
    equals the grouping of the input with the OMIT cells' rows removed by hand, on every field"""
    d = f'{T}/omission'
    src = f'{FROZEN}/tenmin'
    import re
    cid = re.compile(r'^([IV])\w*?_(\d\dh)_(\d{4}-\d\d-\d\d)T\d\d_([^_]+)_\d+$')   # sublst_deep_omission_drop.py:43

    def key(r):
        m = cid.match(r['cand_id'])
        return (m.group(2), m.group(4), m.group(3), r.get('stokes') or m.group(1))
    mcols, mrows = rcsv(f'{src}/master_catalog.csv')
    pcols, prows = rcsv(f'{src}/pertie_detections.csv')
    cells = collections.Counter(key(r) for r in mrows if r['stokes'] == 'V')
    omit = [k for k, _n in sorted(cells.items(), key=lambda kv: (-kv[1], kv[0]))[:2]]   # the two fullest V cells
    gone = {r['cand_id'] for r in mrows if key(r) in omit}
    os.makedirs(f'{d}/hand')
    wcsv(f'{d}/hand/master_catalog.csv', mcols, [r for r in mrows if r['cand_id'] not in gone])
    wcsv(f'{d}/hand/pertie_detections.csv', pcols, [r for r in prows if r['cand_id'] not in gone])
    fz = [f'{src}/master_catalog.csv', f'{src}/pertie_detections.csv']
    keep, om = omission_dir(f'{d}/keep', cover=fz), omission_dir(f'{d}/omit', omit=omit, cover=fz)
    base = ['catalog', '--tier', 'sublst', '--stages', 'group', '--cores', 'none', '--group-stokes', 'V', '--group-new-state']
    rc_o, txt_o = cli(base + ['--group-input', src, '--out', f'{d}/o', '--group-omission', om])
    rc_h, txt_h = cli(base + ['--group-input', f'{d}/hand', '--out', f'{d}/h', '--group-omission', keep])
    to = rcsv(f'{d}/o/grouping_V/detections.csv')[1] if rc_o == 0 else []
    th = rcsv(f'{d}/h/grouping_V/detections.csv')[1] if rc_h == 0 else []
    wantV = {r['cand_id'] for r in mrows if r['stokes'] == 'V'} - gone
    check(f'omission: the {len(gone)} detections of the two OMIT cells {omit} are absent from the grouping; every other V '
          f'detection of the input is in it', rc_o == 0 and len(gone) > 0 and {r['cand_id'] for r in to} == wantV,
          (rc_o, txt_o[-400:]))
    check('omission: the grouping after the deep omission equals the grouping of the hand-filtered input on every field of '
          'every row', rc_h == 0 and to == th and len(to) == len(wantV), (rc_h, txt_h[-300:], len(to), len(th)))
    rec = json.load(open(f'{d}/o/_records/catalog/Run_SLSTB.group.json')) if rc_o == 0 else {}
    mo = rec.get('meta', {}).get('omission', {})
    dr = rcsv(mo['record'])[1] if mo.get('record') and os.path.exists(mo['record']) else []
    check('omission: the record names the table and its coverage record (paths, sha256) and the dropped rows per file; the '
          'drop record lists them',
          mo.get('path') == f'{om}/deep_omission.csv' and mo.get('sha256') == sha256(f'{om}/deep_omission.csv')
          and mo.get('cells_path') == f'{om}/deep_omission_cells.csv'
          and mo.get('cells_sha256') == sha256(f'{om}/deep_omission_cells.csv')
          and mo.get('omit_keys') == 2 and mo.get('dropped') == {'master_catalog.csv': len(gone), 'pertie_detections.csv':
          sum(1 for r in prows if r['cand_id'] in gone)} and {r['cand_id'] for r in dr} == gone, (mo, len(dr)))
    got = GS.recorded_omission(f'{d}/o', 'V').ref() if rc_o == 0 else None
    check('omission: recorded_omission (what every build reads) returns that table, read once (an Omission)',
          got == dict(path=mo.get('path'), sha256=mo.get('sha256'), cells_path=mo.get('cells_path'),
                      cells_sha256=mo.get('cells_sha256')), got)
    for what, args, text in (('no --group-omission', [], 'no deep-omission decision table'),
                             ('a missing table', ['--group-omission', f'{d}/nowhere'], 'missing or unreadable'),
                             ('a table without the input\'s LSTs', ['--group-omission', omission_dir(f'{d}/k23', lsts=['23h'])],
                              'have no verdict'),
                             ('a table without its coverage record', ['--group-omission', f'{d}/nocov'], 'coverage record')):
        if args and 'nocov' in args[-1]:
            os.makedirs(f'{d}/nocov', exist_ok=True)
            shutil.copy(f'{keep}/deep_omission.csv', f'{d}/nocov/deep_omission.csv')
        rc, txt = cli(base + ['--group-input', src, '--out', f'{d}/x'] + args)
        check(f'omission: {what} REFUSES (exit 2, S-G28), never "no omission"', rc == 2 and 'S-G28' in txt and text in txt,
              (rc, txt[-300:]))
    with open(f'{om}/deep_omission.csv', 'a') as f:
        f.write('05h,46MHz,2025-01-01,V,beam,OMIT,changed after the grouping\n')
    try:
        GS.recorded_omission(f'{d}/o', 'V')
        got = 'accepted'
    except GS.StageRefusal as e:
        got = str(e)
    check('omission: a table changed after the grouping recorded it makes the builds\' reading REFUSE (S-G28)',
          'S-G28' in got and 'changed since' in got, got)


def _names(out, S):
    return {r['cand_id']: (r['identity_id'], r['identity_name']) for r in rcsv(f'{out}/grouping_{S}/detections.csv')[1]}


def test_naming():
    from exopipe.phase3.lib.catalog import stokes_link as SL
    check('naming: naming_order - both Stokes in either order: I first reserving nothing, V reserving (run_group default: '
          'I\'s new names); one Stokes: as asked, the default reservation',
          GS.naming_order(['V', 'I']) == (['I', 'V'], {'I': frozenset()}) == GS.naming_order(['I', 'V'])
          and GS.naming_order(['V']) == (['V'], {}) and GS.naming_order(['I']) == (['I'], {}))
    outs, keep = {}, omission_dir(f'{T}/naming/omit', cover=[f'{FROZEN}/tenmin/master_catalog.csv',
                                                            f'{FROZEN}/tenmin/pertie_detections.csv'])
    for k, order in (('a', 'V,I'), ('b', 'I,V')):
        out = outs[k] = f'{T}/naming/{k}'
        rc, txt = cli(['catalog', '--tier', 'sublst', '--stages', 'group', '--group-input', f'{FROZEN}/tenmin', '--out', out,
                       '--cores', 'none', '--group-stokes', order, '--group-new-state', '--group-omission', keep])
        rec = json.load(open(f'{out}/_records/catalog/Run_SLSTB.group.json'))
        check(f'naming: --group-stokes {order} groups both and links them (exit 0, unit done, no S-G27)',
              rc == 0 and rec.get('status') == 'done' and isinstance(rec.get('meta', {}).get('stokes_link'), dict)
              and os.path.islink(SL.link_dir(out)) and not TG.link_failures(os.path.join(SL.link_dir(out), SL.LINKS)),
              (rc, rec.get('status'), rec.get('reason'), txt[-300:]))
    a = {S: _names(outs['a'], S) for S in ('I', 'V')}
    b = {S: _names(outs['b'], S) for S in ('I', 'V')}
    nI, nV = {v[1] for v in a['I'].values()}, {v[1] for v in a['V'].values()}
    check('naming: no display name is held in both Stokes', not (nI & nV), sorted(nI & nV)[:3])
    check('naming: the same dataset gives the same identity IDs and names whichever order the Stokes were asked',
          a == b, [S for S in a if a[S] != b[S]])
    # the collision rule: V grouped alone, reserving nothing, would take some of I's names; in the unit I keeps them and V takes
    # the long form (I grouped alone reserving nothing equals the unit's I: I's names come from I's groupings only)
    mc, side = inputs('10min')
    free = {}
    for S in ('I', 'V'):
        o = f'{T}/naming/free_{S}'
        GS.run_group(GS.load_det('10min', S, mc, side), '10min', S, o, o, 'free', dict(master_catalog=mc), new_state=True,
                     reserved_names=frozenset())
        free[S] = _names(o, S)
    coll = {v[1] for v in free['V'].values()} & {v[1] for v in free['I'].values()}
    movedV = {c for c, v in free['V'].items() if v[1] in coll}
    check(f'naming: grouped alone and reserving nothing, I and V would share {len(coll)} display names (the case the rule '
          f'decides; non-zero on this subset)', len(coll) > 0, len(coll))
    check('naming: in the unit I keeps its names exactly (I keeps the short name)', a['I'] == free['I'],
          sum(1 for c in a['I'] if a['I'][c] != free['I'].get(c)))
    longV = [c for c in movedV if a['V'][c][1] == G._jname_long(*_pos_of(outs['a'], 'V', a['V'][c][0]))]
    check('naming: every V identity whose name I holds takes the long form; every other V name is unchanged',
          len(longV) == len(movedV) and all(a['V'][c] == free['V'][c] for c in free['V'] if c not in movedV),
          (len(longV), len(movedV), sum(1 for c in free['V'] if c not in movedV and a['V'][c] != free['V'][c])))
    # run after run on an unchanged dataset (incremental, the state carried): the same names; the link switched to a new version
    vers0 = sorted(os.listdir(os.path.join(outs['a'], SL.VERSIONS)))
    for k in range(3):
        rc, txt = cli(['catalog', '--tier', 'sublst', '--stages', 'group', '--group-input', f'{FROZEN}/tenmin', '--out',
                       outs['a'], '--cores', 'none', '--group-stokes', 'V,I', '--group-omission', keep])
        same = all(_names(outs['a'], S) == a[S] for S in ('I', 'V'))
        if rc != 0 or not same:
            break
    vers = sorted(os.listdir(os.path.join(outs['a'], SL.VERSIONS)))
    cur = os.path.basename(os.path.realpath(SL.link_dir(outs['a'])))
    check('naming: three more runs on the unchanged dataset give the same IDs and names each time (exit 0)', rc == 0 and same,
          (rc, txt[-300:]))
    check(f'naming: each run switched the link to a new version; the current one and {SL.KEEP_SUPERSEDED} before it are kept, '
          f'the older ones deleted',
          len(vers0) == 1 and vers0[0] not in vers and len(vers) == 1 + SL.KEEP_SUPERSEDED and cur == vers[-1], (vers0, vers, cur))


def _pos_of(out, S, iid):
    """(ra, dec) of identity iid in grouping_<S>/identities.csv (the position its display name is made from)"""
    for r in rcsv(f'{out}/grouping_{S}/identities.csv')[1]:
        if r['identity_id'] == iid:
            return float(r['name_ra']), float(r['name_dec'])
    raise KeyError(iid)


# =====================================================================================================================
ORDER = ['test_mode', 'test_frozen', 'test_defects', 'test_runner', 'test_sublst_build', 'test_one_rule_record', 'test_carry',
         'test_serve_carry', 'test_carry_between_window', 'test_refused_grouping', 'test_grouping_mix', 'test_identity_writeback',
         'test_tenmin_input', 'test_cli', 'test_omission', 'test_naming']


def main(names):
    t0 = time.time()
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        for n in names or ORDER:
            t = time.time()
            try:
                globals()[n]()
            except Exception as e:              # noqa: BLE001 - a crash is a failure of that test
                import traceback
                traceback.print_exc()
                check(f'{n} crashed: {type(e).__name__}: {e}', False)
            print(f'  ({n}: {time.time() - t:.1f} s)', flush=True)
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed in {time.time() - t0:.0f} s'
          + (f'; FAILED: {FAILS}' if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    os.chdir(REPO)
    sys.exit(main(sys.argv[1:]))
