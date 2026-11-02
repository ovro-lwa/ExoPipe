"""The deep search's search-tier gate at 2.0 x the cell median (production exopipe 702dd68; ledger
R-GREGG-20261030-SEARCH-TIER-GATE-2X-CELL-MEDIAN-FASTER-OPTION).

The engine's own code is executed: each block of lib/search/deep_search.run_deep_cell under test is cut from the source and run
in a namespace of inputs (as tests/test_search_b30.py does for the DURATION block), and production's block (git show of
exopipe 702dd68 and of its parent, read only) is run beside it where production has one.
Part 1, the gate on synthetic tables: the threshold is 2.0 x np.median per Stokes (strictly above refuses; an even count's
median is the mean of the middle two); each Stokes on its own values; fewer than SEARCH_TIER_MIN values skip that Stokes; inf
refused, blank / nan never tested; an admitted night (D-ADMIT) tested after the threshold, never in it; the refusal row
lst, scope, date, Stokes, rms, threshold, rule, attempt - the rule before the attempt tag, the tag LAST (the readers' contract);
P3_SEARCH_TIER_RATIO reaches the engine and the rule column; the branch's block = production 702dd68's on every synthetic case.
Part 2, the gate on every installed cell's recorded stage-1 rms (read only; skipped without the co-add root): the branch refuses
exactly what production 702dd68 refuses in every cell, the refusal rows byte for byte; on the inputs of 2026-11-02 (their
digest) V 437 -> 442 and I 674 -> 762 (the 2026-10-30 ledger's counts) and EQ Peg's 21h 2025-08-01 V images at 41 / 59 / 69 /
73 MHz admitted (refused by the old rule); the wideband member admission's mirror (wideband_admission.tier_failures) and the
D-ADMIT build record (build_stats.search_tier_stats) give the engine's refusals and thresholds in every cell.
Part 3, the refusal-file readers and the wiring: the content test accepts the 8 columns and the 7 of rows written before
2026-10-30 (one file may hold both, as production's drop-patched attempts of 2026-10-30 do), refuses another count, a bad rule
and an rms not above its threshold; COLUMNS.csv names the 8 columns; config search_detect / search_sublst SEARCH_TIER_RATIO 2.0,
ruled, SEARCH_TIER_SIGMA no longer a key; the deep step, the 10-minute tier and the frames engine pass P3_SEARCH_TIER_RATIO; no
parity-file line (production's value is the ruled one); production's own attempts of 2026-10-30 pass (read only).
Run: python tests/test_search_tier_table.py
"""
import csv
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.config import CFG                                # noqa: E402

T = '/tmp/exopipe3_tests_search_tier_table'
PROD_REPO = '/lustre/gh/main/exopipe'                                # production (read only: git show)
PROD_PATH = 'exopipe/phase3/phase3_deep.py'
COADD = '/data07/gh/phase3/coadd_sb'                                 # the installed cells (read only)
COLLECT = '/lustre/gh/main/phase3_collect'                           # production's attempts (read only)
LEDGER = '/lustre/gh/main/clv15/fix_ledger.csv'
RULING = 'R-GREGG-20261030-SEARCH-TIER-GATE-2X-CELL-MEDIAN-FASTER-OPTION'
INPUTS_20261102 = (456, 'f2a2d029783c51984dfb32a4f744d4fac21e7658')  # the 456 stage-1 rms tables the counts were measured on
FAILS, N = [], [0]


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({str(detail)[:600]})'))
    if not ok:
        FAILS.append(name)


def branch_src():
    return open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_search.py')).read()


def prod_src(rev):
    return subprocess.run(['git', '-C', PROD_REPO, 'show', f'{rev}:{PROD_PATH}'], check=True, stdout=subprocess.PIPE,
                          universal_newlines=True).stdout


def cut(src, start, end, include_end=True):
    """The dedented lines of ``src`` from the one line that starts (stripped) with ``start`` to the first after it that starts
    with ``end`` (included or not)."""
    ln = src.split('\n')
    i = [k for k, x in enumerate(ln) if x.strip().startswith(start)]
    assert len(i) == 1, (start, len(i))
    j = next(k for k in range(i[0] + 1, len(ln)) if ln[k].strip().startswith(end))
    return textwrap.dedent('\n'.join(ln[i[0]:j + (1 if include_end else 0)]))


def gate_block(src):
    """The search-tier gate: from the stage-1 rms tables to the refusal write and its message."""
    return cut(src, "_rmsI = {r['date']: float(r['i_rms_mjy'])", 'except FileNotFoundError as _e:', include_end=False)


class _Adm:
    @staticmethod
    def values(adm, col):
        from exopipe.phase3.lib.search import admitted as ADS
        return ADS.values(adm, col)


def run_gate(code, rows, refdir, lst='05h', scope='78MHz', ratio=2.0, adm=None, dates=None, attempt='20300101T000000'):
    """Execute a gate block on residuals_provisional ``rows``: ({(date, Stokes): (rms, thr)}, refusal rows written, printed)."""
    os.makedirs(refdir, exist_ok=True)
    p = os.path.join(refdir, f'{lst}_{scope}_searchtier_excluded.csv')
    if os.path.exists(p):
        os.remove(p)
    said = []
    ns = dict(np=np, os=os, _prov=rows, _kd=lambda d: d, cadence='deep', stokes='IV', lst=lst, scope=scope,
              dates=sorted(dates if dates is not None else {r['date'] for r in rows if r.get('date')}), _refdir=refdir,
              _attempt=attempt, _ADS=_Adm, _adm=adm or {}, _unm_tier=set(), print=lambda *a, **k: said.append(' '.join(map(str, a))),
              _enough=lambda n, need, what, count=True: n >= need, SEARCH_TIER_RATIO=ratio, SEARCH_TIER_SIGMA=5.0,
              SEARCH_TIER_MIN=5)
    exec(code, ns)
    tier = {(d, st): (r, thr) for d, st, r, thr in ns['_tier']}
    return tier, (open(p).read().splitlines() if os.path.exists(p) else []), said


def inputs_digest():
    h = hashlib.sha1()
    fs = sorted(glob.glob(os.path.join(COADD, '*', 'residuals_provisional.csv')))
    for f in fs:
        h.update(os.path.basename(os.path.dirname(f)).encode() + b'\0')
        h.update(open(f, 'rb').read() + b'\0')
    return len(fs), h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# part 1: the gate on synthetic tables
# ---------------------------------------------------------------------------------------------------------------------
def test_gate_synthetic():
    br, pr = gate_block(branch_src()), gate_block(prod_src('702dd68'))
    rows = [dict(date='2025-01-%02d' % i, i_rms_mjy='100', v_rms_mjy='50') for i in range(1, 11)]
    rows += [dict(date='2025-02-01', i_rms_mjy='200.0', v_rms_mjy='99.9'),     # exactly 2 x the I median: searched
             dict(date='2025-02-02', i_rms_mjy='200.1', v_rms_mjy='50'),       # just above it in I only
             dict(date='2025-02-03', i_rms_mjy='100', v_rms_mjy='inf'),        # inf: tested and refused, never in the median
             dict(date='2025-02-04', i_rms_mjy='nan', v_rms_mjy='')]           # nan / blank: never tested
    d = os.path.join(T, 'gate')
    tier, written, said = run_gate(br, rows, d)
    check('gate: refused = the I night above 2 x the I median (200.1 > 200) and the V night of inf; 2 x the median exactly '
          'is searched; per Stokes (the I outlier keeps its V search); nan / blank never tested',
          set(tier) == {('2025-02-02', 'I'), ('2025-02-03', 'V')} and tier[('2025-02-02', 'I')][1] == 200.0
          and tier[('2025-02-03', 'V')][1] == 100.0, tier)
    rr = [x.split(',') for x in written]
    check('gate: the refusal row is lst, scope, date, Stokes, rms, threshold, rule, attempt - the rule ratio2xmedian before '
          'the attempt tag, the tag LAST', len(rr) == 2 and all(len(x) == 8 and x[6] == 'ratio2xmedian'
                                                               and x[-1] == '20300101T000000' for x in rr)
          and rr[0][:6] == ['05h', '78MHz', '2025-02-02', 'I', '200.1', '200.0'] and rr[1][4] == 'inf', written)
    check('gate: the message names the rule', any('stage-1 rms > 2 x the cell median of that Stokes' in s for s in said), said)
    tp, wp, _s = run_gate(pr, rows, os.path.join(T, 'gate_prod'))
    check('gate: the branch block = production 702dd68\'s (refusals, thresholds, rows)', tp == tier and wp == written, (tp, wp))
    ev = [dict(date='n%d' % i, i_rms_mjy=str(x), v_rms_mjy='1') for i, x in enumerate((10, 20, 30, 40, 50, 1000))]
    te, we, _s = run_gate(br, ev, os.path.join(T, 'gate_even'))
    check('gate: an even count\'s median is the mean of the middle two (np.median: 35 -> threshold 70, not 80)',
          set(te) == {('n5', 'I')} and te[('n5', 'I')][1] == 70.0 and we[0].split(',')[5] == '70.0', (te, we))
    few = [dict(date='m%d' % i, i_rms_mjy='10', v_rms_mjy=('5' if i < 4 else '')) for i in range(6)]
    few.append(dict(date='m9', i_rms_mjy='10', v_rms_mjy='999'))
    tf, _w, said = run_gate(br, few, os.path.join(T, 'gate_few'))
    check('gate: fewer than SEARCH_TIER_MIN values skips that Stokes only (V: 5 values with the outlier, 4 without it)',
          ('m9', 'V') in tf and not any(s == 'I' for _d, s in tf), tf)
    few2 = [r for r in few if r['date'] != 'm9']
    tf2, _w, said2 = run_gate(br, few2, os.path.join(T, 'gate_few2'))
    check('gate: 4 V values - the V gate skipped (loudly), the I gate run', not tf2 and any('SEARCH-TIER GATE (V) SKIPPED' in s
                                                                                         for s in said2), said2)
    adm = {'2025-03-01': dict(stage2_i_rms_mjy='300', stage2_v_rms_mjy='60')}
    ta, wa, _s = run_gate(br, rows, os.path.join(T, 'gate_adm'), adm=adm, dates=[r['date'] for r in rows] + ['2025-03-01'])
    check('gate (D-ADMIT): an admitted night is tested against the cell\'s threshold, never part of it (I 300 > 200 refused, '
          'V 60 searched; the thresholds unchanged)', ta.get(('2025-03-01', 'I')) == (300.0, 200.0)
          and ('2025-03-01', 'V') not in ta and {k: v for k, v in ta.items() if k[0] != '2025-03-01'} == tier, ta)
    t25, w25, s25 = run_gate(br, rows, os.path.join(T, 'gate_25'), ratio=2.5)
    check('gate: another ratio is the rule\'s value everywhere (2.5: threshold 250 in I, rule ratio2.5xmedian, the message)',
          set(t25) == {('2025-02-03', 'V')} and all(x.split(',')[6] == 'ratio2.5xmedian' for x in w25)
          and any('> 2.5 x the cell median' in s for s in s25), (t25, w25))
    pr_ = subprocess.run([sys.executable, '-c', 'import sys; sys.path.insert(0, %r); from exopipe.phase3.lib.search import '
                          'deep_search as D; print(D.SEARCH_TIER_RATIO)' % REPO], env=dict(os.environ, P3_SEARCH_TIER_RATIO='2.5'),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=900)
    check('engine: P3_SEARCH_TIER_RATIO reaches deep_search.SEARCH_TIER_RATIO', pr_.stdout.strip().splitlines()[-1:] == ['2.5'],
          pr_.stdout[-300:])
    from exopipe.phase3.lib.search import deep_search as D
    check('engine: SEARCH_TIER_RATIO 2.0 by default (production\'s P3_SEARCH_TIER_RATIO default)', D.SEARCH_TIER_RATIO == 2.0)
    src = branch_src()
    check('engine: the threshold line is production 702dd68\'s, and no other line takes the old MAD threshold',
          '_thr = SEARCH_TIER_RATIO * _med                # Gregg 2026-10-30 (was _med + SEARCH_TIER_SIGMA * _mad)' in src
          and '_med + SEARCH_TIER_SIGMA * _mad\n' not in src and 'sigma=SEARCH_TIER_SIGMA' not in src
          and '_wbp = dict(ratio=SEARCH_TIER_RATIO,' in src)


# ---------------------------------------------------------------------------------------------------------------------
# part 2: the gate on every installed cell's recorded stage-1 rms
# ---------------------------------------------------------------------------------------------------------------------
def test_gate_replay():
    cells = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(COADD, '*', 'residuals_provisional.csv')))
    if not cells:
        print(f'SKIP the real-data replay (no {COADD})')
        return
    from exopipe.phase3.lib.search import wideband_admission as W
    from exopipe.phase3.lib.masters import build_stats as BS
    codes = {'branch': gate_block(branch_src()), 'prod': gate_block(prod_src('702dd68')), 'old': gate_block(prod_src('702dd68^'))}
    res = {k: {} for k in codes}
    mirror_bad, stats_bad = [], []
    for c in cells:
        lst, scope = c.split('_', 1)
        rows = list(csv.DictReader(open(os.path.join(COADD, c, 'residuals_provisional.csv'))))
        for k, code in codes.items():
            res[k][c] = run_gate(code, rows, os.path.join(T, 'replay', k), lst=lst, scope=scope, attempt='REPLAY')[:2]
        tier = res['branch'][c][0]
        tf = W.tier_failures(rows, 2.0, 5)
        mirror = {(d, st): v for st, x in tf.items() for d, v in x.items()}
        if mirror != tier:
            mirror_bad.append(c)
        bs = BS.search_tier_stats(rows, 2.0, 5)
        for (d, st), (r, thr) in tier.items():
            if bs[st].get('thr') != thr:
                stats_bad.append((c, st))
    diff = [c for c in cells if res['branch'][c] != res['prod'][c]]
    check(f'replay ({len(cells)} cells): the branch refuses exactly what production 702dd68 refuses in every cell, the refusal '
          f'rows byte for byte', not diff, diff[:5])
    check('replay: the wideband member admission\'s mirror (tier_failures, ratio 2.0) gives the engine\'s refusals and '
          'thresholds in every cell', not mirror_bad, mirror_bad[:5])
    check('replay: the D-ADMIT build record\'s search-tier threshold (build_stats, ratio 2.0) is the engine\'s in every cell',
          not stats_bad, stats_bad[:5])
    cnt = {k: {st: sum(1 for c in cells for (_d, s) in res[k][c][0] if s == st) for st in ('I', 'V')} for k in codes}
    eq = {b: (('2025-08-01', 'V') in res['old'].get('21h_' + b, ({},))[0], ('2025-08-01', 'V') in res['branch'].get('21h_' + b, ({},))[0])
          for b in ('41MHz', '59MHz', '69MHz', '73MHz')}
    print(f'      replay counts: old rule (production 702dd68^) V {cnt["old"]["V"]} I {cnt["old"]["I"]}; branch V {cnt["branch"]["V"]} '
          f'I {cnt["branch"]["I"]}; EQ Peg 21h 2025-08-01 V refused (old, branch): {eq}')
    dig = inputs_digest()
    if dig == INPUTS_20261102:
        check('replay on the inputs of 2026-11-02: V refused 437 -> 442, I 674 -> 762 (the 2026-10-30 ledger\'s counts)',
              (cnt['old']['V'], cnt['branch']['V'], cnt['old']['I'], cnt['branch']['I']) == (437, 442, 674, 762), cnt)
        check('replay on the inputs of 2026-11-02: EQ Peg 21h 2025-08-01 V at 41 / 59 / 69 / 73 MHz refused by the old rule, '
              'searched under 2 x the median (69 MHz: the -8.4 sigma burst)', all(o and not n for o, n in eq.values()), eq)
    else:
        print(f'      NOTE the stage-1 rms tables changed since 2026-11-02 ({dig} != {INPUTS_20261102}): the counts are '
              f'reported, not asserted; the per-cell agreement with production above is the test')


# ---------------------------------------------------------------------------------------------------------------------
# part 3: the refusal-file readers and the wiring
# ---------------------------------------------------------------------------------------------------------------------
def _refusal_file(name, lines):
    rd = os.path.join(T, 'att_ref', 'transients', 'Run_TEST', 'refusals')
    os.makedirs(rd, exist_ok=True)
    p = os.path.join(rd, name)
    with open(p, 'w') as f:
        f.write(''.join(x + '\n' for x in lines))
    return p


def test_readers_and_wiring():
    from exopipe.phase3.lib.search import tests as ST, deep_search as D, tier_sublst as TS
    from exopipe.phase3.core import validate
    new = '05h,78MHz,2025-01-02,V,120.5,100.0,ratio2xmedian,20301030T000000'
    old = '05h,78MHz,2025-01-01,I,500.0,410.2,20261008T030052'
    nm = '05h_78MHz_searchtier_excluded.csv'
    res = {k: ST.parses(_refusal_file(nm, v)) for k, v in (('new', [new]), ('old', [old]), ('both', [old, new]))}
    check('refusal_record: the 8-column row, the 7 of a row before 2026-10-30, and a file holding both pass',
          all(r[0] for r in res.values()), {k: r[2] for k, r in res.items()})
    bad = {'6 columns': '05h,78MHz,2025-01-02,V,120.5,20301030T000000',
           'a bad rule': new.replace('ratio2xmedian', 'mad5'),
           'an rms not above its threshold': new.replace('120.5', '99.9'),
           'an unreadable rms': new.replace('120.5', 'x')}
    got = {k: ST.parses(_refusal_file(nm, [old, v])) for k, v in bad.items()}
    check('refusal_record: a row of another count, a bad rule, an rms not above its threshold or unreadable FAILS',
          not any(r[0] for r in got.values()), {k: r[2] for k, r in got.items()})
    check('readers: the attempt tag is the last column in both forms (r[-1]), the Stokes the fourth (r[3]); refused_dates '
          'reads both', [x.split(',')[-1] for x in (old, new)] == ['20261008T030052', '20301030T000000']
          and ST.refused_dates(os.path.join(T, 'att_ref', 'transients', 'Run_TEST'), 'V')[0] >= {'2025-01-02'}
          and ST.refused_dates(os.path.join(T, 'att_ref', 'transients', 'Run_TEST'), 'I')[0] >= {'2025-01-01'})
    r = validate.run('refusal_record', _refusal_file(nm, [old, new]))
    check('refusal_record through the registry', r and all(x['passed'] for x in r), r)
    cols = dict(D.REFUSAL_COLUMNS)
    check('COLUMNS.csv names the 8 columns of a searchtier row; the 10-minute tier\'s column table agrees',
          cols['_searchtier_excluded.csv'] == 'lst,scope,date,stokes,stage1_rms,threshold,rule,attempt'
          and TS.REFUSAL_COLUMNS['searchtier'] == 8 and ST.REFUSAL_KINDS['searchtier'] == (8, 'row')
          and ST.REFUSAL_EARLIER_NCOL == {'searchtier': 7})
    sd, ss = CFG.search_detect, CFG.search_sublst
    check('config: search_detect and search_sublst SEARCH_TIER_RATIO = 2.0, ruled by the 2026-10-30 ledger row; '
          'SEARCH_TIER_SIGMA is no key', sd.SEARCH_TIER_RATIO == 2.0 == ss.SEARCH_TIER_RATIO
          and sd.RULED.get('SEARCH_TIER_RATIO') == RULING == ss.RULED.get('SEARCH_TIER_RATIO')
          and not hasattr(sd, 'SEARCH_TIER_SIGMA') and not hasattr(ss, 'SEARCH_TIER_SIGMA')
          and 'SEARCH_TIER_SIGMA' not in sd.RULED and 'SEARCH_TIER_SIGMA' not in ss.RULED)
    check('the ruling ID is a fix_id of the ledger', any(r_.get('fix_id') == RULING for r_ in csv.DictReader(open(LEDGER))))
    try:
        CFG.set('search_detect.SEARCH_TIER_SIGMA', '5.0')
        ok = False
    except KeyError as e:
        ok = 'unknown config key' in str(e) and 'SEARCH_TIER_SIGMA' in str(e)
    finally:
        CFG.reset()
    check('config: --set search_detect.SEARCH_TIER_SIGMA refuses (no such key: the old rule cannot be asked for silently)', ok)
    CFG.reset()
    CFG.authorize([], out=T)
    CFG.set('search_detect.SEARCH_TIER_RATIO', '2.5')
    CFG.set('search_sublst.SEARCH_TIER_RATIO', '3.0')
    try:
        from exopipe.phase3.lib.search import deep_step as DST
        env = DST.engine_env(f'{T}/c', f'{T}/a', f'{T}/f', 'X', f'{T}/ar', f'{T}/tmp')
        env10 = TS.engine_env({'label': 'SLST09B', 'arm': 'B', 'lst': '09h'}, f'{T}/att')
        from exopipe.phase3.lib.frames import deep_engine as DE
        vals = DE.engine_values('deep')
        vals10 = DE.engine_values('sublst')
    finally:
        CFG.reset()
    desrc = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'frames', 'deep_engine.py')).read()
    check('wiring: the deep step passes P3_SEARCH_TIER_RATIO from search_detect (and no P3_SEARCH_TIER_SIGMA)',
          env.get('P3_SEARCH_TIER_RATIO') == '2.5' and 'P3_SEARCH_TIER_SIGMA' not in env, {k: v for k, v in env.items() if 'TIER' in k})
    check('wiring: the 10-minute tier passes P3_SEARCH_TIER_RATIO from search_sublst (and no P3_SEARCH_TIER_SIGMA)',
          env10.get('P3_SEARCH_TIER_RATIO') == '3.0' and 'P3_SEARCH_TIER_SIGMA' not in env10,
          {k: v for k, v in env10.items() if 'TIER' in k})
    check('wiring: the frames engine reads SEARCH_TIER_RATIO per tier and passes P3_SEARCH_TIER_RATIO',
          vals['search_tier_ratio'] == 2.5 and vals10['search_tier_ratio'] == 3.0
          and "s['P3_SEARCH_TIER_RATIO'] = repr(v['search_tier_ratio'])" in desrc and 'SEARCH_TIER_SIGMA' not in desrc,
          (vals.get('search_tier_ratio'), vals10.get('search_tier_ratio')))
    par = open(os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg')).read()
    check('parity: no SEARCH_TIER line (production\'s value, 2.0 x the median since 702dd68, is the ruled one)',
          'SEARCH_TIER' not in par)
    # production's own attempts of 2026-10-30 (read only): 8-column rows, and drop-patched attempts holding both forms
    refs = sorted(glob.glob(os.path.join(COLLECT, 'p3s5_collect_*', 'attempts', '*_2026103*', 'transients', 'Run_P3S5', 'refusals',
                                         '*_searchtier_excluded.csv')))
    if refs:
        forms, bad_ = set(), []
        for p in refs:
            forms |= {len(x) for x in csv.reader(open(p)) if x}
            ok_, _v, det = ST.parses(p)
            if not ok_:
                bad_.append((p, det))
        check(f'production\'s searchtier records of 2026-10-30 ({len(refs)} files; column counts {sorted(forms)}) pass '
              f'refusal_record', not bad_ and forms == {7, 8}, bad_[:3])
    else:
        print(f'SKIP production\'s attempts of 2026-10-30 (none under {COLLECT})')


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        for fn in (test_gate_synthetic, test_gate_replay, test_readers_and_wiring):
            try:
                fn()
            except Exception as e:
                import traceback
                traceback.print_exc()
                check(f'{fn.__name__} ran to the end', False, f'{type(e).__name__}: {e}')
    finally:
        if not os.environ.get('EXOPIPE3_TEST_KEEP'):
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
