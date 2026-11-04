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
               design B, refuses a selected row the grouping does not hold, runs it in place under production's mode
  record       one_rule_record against synthetic builder outputs: rescued multiples, singles dropped by the sigma-stack test
               and by the repeat rungs, a single kept by a cross-match rung
  carry        carry_judgements: agreeing pages carried unchanged, a disagreement a CONFLICT not carried, an unreached judgement
               reported, a non-empty journal refused
  cli          `exopipe3 catalog --tier sublst --stages group --group-input <frozen 10-minute subset>`: the first run needs
               --group-new-state; a second run carries every ID; under production's mode the unit is no-input
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

T = '/tmp/exopipe3_tests_grouping_stage'
FROZEN = '/lustre/gh/main/phase3/refactor_analysis/refactor_wt_20261102/F_grouping/frozen_inputs'
LIVE_BUILDER = '/lustre/gh/main/exopipe_ops/build_source_catalogs_v3.py'
FROZEN_BUILDERS = '/lustre/gh/main/phase3/sublst/code_frozen/*/exopipe_ops/build_source_catalogs_v3.py'
PY = sys.executable
FAILS, N = [], [0]


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
    check('mode: GROUPING and MULTIPLES_RULE are RULED (10-26 design; 10-27 one chain)',
          'R-GREGG-20261026-GROUPING-DESIGN-B-APPROVED-IDENTITY-097-MULTIPLES-085-DEEPV-IDENTITY-ONLY-PREVIEW-THEN-REFACTOR'
          in str(CFG.ruled('catalog_grouping.GROUPING'))
          and 'R-GREGG-20261027-HUB-BOTH-LISTENERS-MULTIPLES-CHAIN-FORCED-EVERY-TIER' in str(CFG.ruled('catalog_grouping.MULTIPLES_RULE')))
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


def _run_runner(d, out, mode, table=None, builder=None):
    env = dict(os.environ, EXOPIPE3_GROUPING_MODE=mode, EXOPIPE3_GROUPING_TIER='deep', EXOPIPE3_GROUPING_RUNGS='multiples-level')
    if table:
        env['EXOPIPE3_GROUPING_TABLE'] = table
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
    os.makedirs(f'{gdir}/grouping_I')
    shutil.move(f'{U}/grouping_I/detections.csv', f'{gdir}/grouping_I/detections.csv')
    os.makedirs(ops)
    open(f'{ops}/_stubcat.py', 'w').write(STUBCAT)
    open(f'{ops}/build_source_catalogs_v3.py', 'w').write(PATCHABLE_BUILDER.lstrip())
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
    ctx.U, ctx.dry, ctx.builds, ctx.pt = U, False, ('noveto',), f'{U}/pertie'
    ctx.py = {'default': PY}
    ctx.opsc = lambda name: os.path.join(ops, name)
    ctx.log = TS.Log(f'{U}/catalogs.log')
    ctx.env = lambda **extra: {k: str(v) for k, v in extra.items()}
    ctx.grouping_mode = GS.mode()
    ctx.grouping_dir = gdir
    ctx.grouping_table = lambda S: os.path.join(gdir, f'grouping_{S}', 'detections.csv')

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
    tab2 = rcsv(ctx.grouping_table('I'))[1]
    wcsv(ctx.grouping_table('I'), GS.DET_COLS, tab2[1:])
    shutil.rmtree(f'{U}/source_catalogs_noveto_I_sb')
    st, s, plan = run_build()
    check('sub-LST build: a selected row the grouping does not hold REFUSES (S-G18), naming the remedy',
          'S-G18' in st and 'group the cumulative again' in st, st)
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
    try:
        GS.carry_judgements(f'{d}/old/judgements.jsonl', members, f'{d}/grouping_I/detections.csv', f'{d}/new', 'Run_TEST')
        got = 'written'
    except GS.StageRefusal as e:
        got = str(e)
    check('carry: refuses to write over a non-empty journal (it only seeds)', 'only seeds' in got, got)
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
def cli(args, timeout=1800):
    env = dict(os.environ, PYTHONPATH=REPO)
    p = subprocess.run([PY, '-m', 'exopipe.phase3'] + args, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def test_cli():
    out = f'{T}/cli/grouping_sublst'
    common = ['catalog', '--tier', 'sublst', '--stages', 'group', '--group-input', f'{FROZEN}/tenmin', '--out', out,
              '--cores', 'none', '--group-stokes', 'V']
    rc, txt = cli(common)
    check('cli: the first 10-minute grouping without --group-new-state REFUSES (exit 2)', rc == 2 and 'new=True' in txt,
          txt[-400:])
    rc, txt = cli(common + ['--group-new-state'])
    recp = sorted(glob.glob(f'{out}/_records/catalog/Run_SLSTB.group.json'))
    rec = json.load(open(recp[0])) if recp else {}
    check('cli: --group-new-state groups the cumulative (exit 0; unit done; content tests passed)',
          rc == 0 and rec.get('status') == 'done' and rec.get('tests') and all(t['passed'] for t in rec['tests']), txt[-400:])
    ids0 = {r['cand_id']: r['identity_id'] for r in rcsv(f'{out}/grouping_V/detections.csv')[1]}
    rc, txt = cli(common)
    ids1 = {r['cand_id']: r['identity_id'] for r in rcsv(f'{out}/grouping_V/detections.csv')[1]}
    check('cli: a second run (incremental, the same detections) carries every identity ID', rc == 0 and ids0 == ids1, txt[-300:])
    rc, txt = cli(common + ['--set', 'catalog_grouping.GROUPING=production', '--set', 'catalog_grouping.MULTIPLES_RULE=production'])
    rec = json.load(open(recp[0])) if recp else {}
    check("cli: under production's mode the unit is no-input (exit 3), naming the parity file",
          rc == 3 and rec.get('status') == 'no-input' and 'no grouping stage' in (rec.get('reason') or ''), (rc, rec.get('reason')))


# =====================================================================================================================
ORDER = ['test_mode', 'test_frozen', 'test_defects', 'test_runner', 'test_sublst_build', 'test_one_rule_record', 'test_carry',
         'test_serve_carry', 'test_identity_writeback', 'test_cli']


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
