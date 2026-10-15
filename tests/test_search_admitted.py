"""D-ADMIT (a) slice 6: the deep search with a cell's admitted nights (lib/search/admitted.py and the WAVE2 D-ADMIT lines of
lib/search/deep_search.run_deep_cell; plan/wave2/DADMIT_DESIGN.md "Slices 5 and 6 as they will be written").

Part 1, the helpers on synthetic inputs: the reading passed to the engine (EXOPIPE3_ADMITTED_NIGHTS or config; an unknown
value refuses); which nights the search accepts (admitted into this master, every product carrying the admission's cards;
none at 10 minutes or under production's reading); E10 / E11 from admissions.csv (a FAIL refuses, as for the build's
records); the admitted nights' own gate values; residual dates and populations (the same objects when no night is
admitted - production's calls unchanged); each admitted night's sigma stack over the cell's searched nights plus itself,
never another admitted night.
Part 2, the engine source: every population the engine computes leaves the admitted nights out, and every insertion is
marked WAVE2 D-ADMIT.
Part 3 (end to end, about 15 minutes, no GPU): a copy of 05h_78MHz with two member nights taken out of its records and its
master's constituent list and admitted again (tests/test_masters_admit6.py's copy), searched by run_deep_cell in a child in
incremental mode for one admitted night (x2 flow refinement off, v110 off): under production's reading the membership gate
refuses the cell; under the ruled reading the night is accepted, passes the gates against the cell's own nights, is
searched with its own sigma stack over the cell's 20 searched nights plus itself (not the other admitted night), written
beside the cell's map.
Run: python tests/test_search_admitted.py [--quick]   (--quick: parts 1 and 2 only)
"""
import ast
import csv
import glob
import inspect
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.core import env as ENV                          # noqa: E402
ENV.legacy()
from exopipe.phase3.config import CFG                               # noqa: E402
from exopipe.phase3.core.record import Refusal                      # noqa: E402
from exopipe.phase3.lib.masters import admit as A                   # noqa: E402
from exopipe.phase3.lib.search import admitted as ADS               # noqa: E402

T = '/tmp/exopipe3_tests_search_admitted'
FAILS, N = [], [0]
CELL, LST, SC = '05h_78MHz', '05h', '78MHz'


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'), flush=True)
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------------------------------------------------
# part 1: the helpers
# ---------------------------------------------------------------------------------------------------------------------

def _cell(root):
    """root/<CELL>: the installed I master linked, two admitted nights' I and V sub products (small FITS with the cards) and
    their admissions.csv rows."""
    from astropy.io import fits
    from exopipe.phase3.lib.masters import admit_stages as AS
    src, cd = os.path.join(CFG.paths.COADD_SB_ROOT, CELL), os.path.join(root, CELL)
    os.makedirs(os.path.join(cd, 'coadd_final'))
    os.makedirs(os.path.join(cd, 'subtracted'))
    os.symlink(os.path.join(src, 'coadd_final', f'I_deep_Taper_R075_{SC}.fits'),
               os.path.join(cd, 'coadd_final', f'I_deep_Taper_R075_{SC}.fits'))
    cdg, md5 = A.master_identity(cd)
    rows = []
    for d, ng, eg in (('2026-10-20', 'pass', 'pass'), ('2026-10-21', 'pass', 'pass')):
        for st in ('I', 'V'):
            h = fits.Header()
            h['ADMITTED'] = True
            h['ADMITREF'] = cdg[:32]
            fits.writeto(os.path.join(cd, 'subtracted', f'{d}_{st}_sub_{SC}.fits'), np.zeros((4, 4), np.float32), h)
        r = dict.fromkeys(AS.ADMISSION_COLS, '')
        r.update(date=d, verdict='admitted', reference_constdig=cdg, reference_md5=md5, night_image=ng, epoch_gate=eg,
                 qa3_rms_mjy='12.50', stage2_i_rms_mjy='3.2500', stage2_v_rms_mjy='', utc='2026-10-22T00:00:00Z')
        rows.append(r)
    AS.write_admissions(cd, rows)
    return cd


def test_policy():
    old = os.environ.pop(ADS.POLICY_ENV, None)
    try:
        a = ADS.policy()
        os.environ[ADS.POLICY_ENV] = 'absent'
        b = ADS.policy()
        os.environ[ADS.POLICY_ENV] = 'admited'
        try:
            ADS.policy()
            c = 'no refusal'
        except Refusal as e:
            c = str(e)
    finally:
        os.environ.pop(ADS.POLICY_ENV, None)
        if old is not None:
            os.environ[ADS.POLICY_ENV] = old
    check('the engine\'s reading: config without the variable (ruled \'admitted\'), the variable when the step passes it, an '
          'unknown value refuses', a == 'admitted' and b == 'absent' and 'ADMITTED_NIGHTS' in c, (a, b, c))
    from exopipe.phase3.lib.search import deep_step as DS
    from exopipe.phase3.lib.frames import deep_engine as FE
    src = inspect.getsource(DS.engine_env) + inspect.getsource(FE.child_env)
    check('the search step and the frames step pass masters_admit.ADMITTED_NIGHTS to the engine child',
          src.count('_ADS.POLICY_ENV') == 2 and src.count('masters_admit.ADMITTED_NIGHTS') == 2)
    from exopipe.phase3.steps import archive, frames, install, masters, search
    miss = [m.__name__ for m in (archive, frames, install, masters, search) if 'masters_admit' not in m.CONFIG]
    check('every step that reads the switch declares masters_admit (its config hash moves with the value)', not miss, miss)


def test_nights(cd):
    from astropy.io import fits
    os.environ[ADS.POLICY_ENV] = 'admitted'
    try:
        ok, bad = ADS.nights(cd, SC, 'deep')
        check('two admitted nights with their cards are accepted', sorted(ok) == ['2026-10-20', '2026-10-21'] and not bad,
              (sorted(ok), bad))
        ok10, bad10 = ADS.nights(cd, SC, '10min')
        check('the 10-minute cadence accepts none (the snapshot chat\'s tier: through the deep record once agreed)',
              ok10 == {} and bad10 == {})
        p = os.path.join(cd, 'subtracted', f'2026-10-21_V_sub_{SC}.fits')
        with fits.open(p, mode='update') as h:
            h[0].header['ADMITREF'] = 'another'
        ok, bad = ADS.nights(cd, SC, 'deep')
        check('a product whose ADMITREF is not the master\'s CONSTDIG: the night is refused, named', sorted(ok) == ['2026-10-20']
              and 'ADMITREF' in bad.get('2026-10-21', ''), (sorted(ok), bad))
        os.environ[ADS.POLICY_ENV] = 'absent'
        check('production\'s reading: no night accepted', ADS.nights(cd, SC, 'deep') == ({}, {}))
    finally:
        os.environ.pop(ADS.POLICY_ENV, None)


def test_unreadable_and_same_state(cd):
    """MINOR 10 (eighth review): an unreadable admitted product refuses its night by name, not the engine; same_state_v."""
    p = os.path.join(cd, 'subtracted', f'2026-10-20_I_sub_{SC}.fits')
    keep = open(p, 'rb').read()
    open(p, 'wb').write(b'not a FITS file')
    os.environ[ADS.POLICY_ENV] = 'admitted'
    try:
        try:
            ok, bad = ADS.nights(cd, SC, 'deep')
            r = None
        except Exception as e:                       # noqa: BLE001 - the failure this test guards against
            ok, bad, r = {}, {}, f'{type(e).__name__}: {e}'
    finally:
        os.environ.pop(ADS.POLICY_ENV, None)
        open(p, 'wb').write(keep)
    check('an unreadable admitted product refuses its night by name (E15), the engine goes on', r is None
          and '2026-10-20' not in ok and 'unreadable' in bad.get('2026-10-20', ''), (r, sorted(ok), bad))
    adm = {'a': dict(v_products='refused (B-03 same-state): x'), 'b': dict(v_products='admitted'),
           'c': dict(v_products='no V image'), 'd': dict(v_products='refused (alignment: the V image ...)')}
    check('same_state_v: only an admitted night whose V the same-state rule refused (a member\'s refusals_input.csv rule)',
          ADS.same_state_v(adm) == {'a'}, ADS.same_state_v(adm))


def test_unit_outputs():
    """MINOR 7 (eighth review): an admitted night's own sigma stacks are outputs of the search unit (class noise_map)."""
    from exopipe.phase3.lib.search import deep_step as DS
    att = os.path.join(T, 'outputs_att')
    sd = os.path.join(att, 'transients', 'Run_X', 'sigstack')
    os.makedirs(sd)
    for name in (f'{CELL}_I_sigstack.fits', f'{CELL}_I_sigstack_2026-10-20.fits', f'{CELL}_V_sigstack_2026-10-20.fits'):
        open(os.path.join(sd, name), 'w').close()
    outs = DS.outputs_of(DS.Item(CELL, 'I', 'bulk'), {}, dict(attempt=att, label='X'))
    got = sorted(os.path.basename(p) for p, cls, _c in outs if cls == 'noise_map')
    check('the search unit\'s outputs: the cell\'s map and each admitted night\'s own stack of its Stokes',
          got == [f'{CELL}_I_sigstack.fits', f'{CELL}_I_sigstack_2026-10-20.fits'], got)


def test_values():
    adm = {'a': dict(night_image='pass', epoch_gate='pass', qa3_rms_mjy='12.5', stage2_i_rms_mjy='3.25', stage2_v_rms_mjy=''),
           'b': dict(night_image='FAIL', epoch_gate='pass', qa3_rms_mjy='nan', stage2_i_rms_mjy='x', stage2_v_rms_mjy='4'),
           'c': dict(night_image='pass', epoch_gate='FAIL', qa3_rms_mjy='', stage2_i_rms_mjy='', stage2_v_rms_mjy='')}
    check('E10 / E11 from admissions.csv: a FAIL refuses (the build records\' rule)', ADS.gate_failures(adm) == ({'b'}, {'c'}),
          ADS.gate_failures(adm))
    check('an admitted night\'s own values; blank, nan or unreadable are absent (the gate\'s unmeasured rule applies)',
          ADS.values(adm, 'qa3_rms_mjy') == {'a': 12.5} and ADS.values(adm, 'stage2_i_rms_mjy') == {'a': 3.25}
          and ADS.values(adm, 'stage2_v_rms_mjy') == {'b': 4.0})
    check('no admitted night: every helper returns production\'s values', ADS.gate_failures({}) == (set(), set())
          and ADS.values({}, 'qa3_rms_mjy') == {} and ADS.without(['x', 'y'], {}) == ['x', 'y'])


def test_residuals_and_stacks():
    from exopipe.phase3.lib.search import deep_search as E
    dates = ['m1', 'm2', 'm3']
    check('residual dates: the same list object when no admitted night is searched (production\'s call unchanged)',
          ADS.residual_dates(dates, dates, {}) is dates and ADS.residual_dates(dates, ['m4'], {'a': {}}) is dates)
    check('incremental with an admitted night: its searched dates plus the cell\'s nights its stack needs',
          ADS.residual_dates(['a'], ['m1', 'm2'], {'a': {}}) == ['a', 'm1', 'm2'])
    r = {d: np.full((2, 2), i, float) for i, d in enumerate(['m1', 'm2', 'a', 'b'])}
    check('only(): the same mapping when it holds no other date', ADS.only(r, ['m1', 'm2', 'a', 'b', 'z']) is r)
    lz = E._Lazy()
    calls = []
    for d in r:
        dict.__setitem__(lz, d, (lambda _d=d: calls.append(_d) or r[_d]))
    sub = ADS.only(lz, ['m1', 'a'])
    check('only() of a streaming mapping keeps its loaders (nothing loaded) and its kind', isinstance(sub, E._Lazy)
          and sorted(dict.keys(sub)) == ['a', 'm1'] and not calls, (type(sub), sorted(dict.keys(sub)), calls))
    built, written = [], []
    out = ADS.stacks(['a', 'm1', 'b'], ['m1', 'm2', 'b'], {'a': {}, 'b': {}}, r,
                     lambda rr: built.append(sorted(dict.keys(rr))) or len(rr), lambda s, d: written.append((d, s)))
    check('each admitted night\'s stack: the cell\'s searched nights plus itself, never another admitted night; each written',
          built == [['a', 'm1', 'm2'], ['b', 'm1', 'm2']] and out == {'a': 3, 'b': 3} and written == [('a', 3), ('b', 3)],
          (built, out, written))
    check('no admitted night searched: no stack built', ADS.stacks(['m1'], ['m1', 'm2'], {}, r, lambda rr: 1 / 0,
                                                                  lambda s, d: 1 / 0) == {})


# ---------------------------------------------------------------------------------------------------------------------
# part 2: the engine source
# ---------------------------------------------------------------------------------------------------------------------

def test_engine_source():
    from exopipe.phase3.lib.search import deep_search as E
    src = inspect.getsource(E.run_deep_cell)
    lines = [ln for ln in src.splitlines() if '_adm' in ln or '_ADS' in ln or '_rdates' in ln or '_ssA_' in ln]
    unmarked = [ln.strip() for ln in lines if 'WAVE2 D-ADMIT' not in ln and not ln.strip().startswith('lambda')]
    check('every engine line that reads the admitted nights is marked WAVE2 D-ADMIT (a lambda continues a marked call)',
          len(lines) >= 25 and not unmarked, (len(lines), unmarked))
    want = {
        'the membership gate reads admitted nights': "_ADV.verify(base, policy=_ADS.policy())",
        'E10 night-image verdicts of admitted nights': "_ngf = set(_ngf) | _ADS.gate_failures(_adm)[0]",
        'E11 epoch verdicts of admitted nights': "_egf = set(_egf) | _ADS.gate_failures(_adm)[1]",
        'the I sanity gate: own value after the median': "_wr = {**_wr, **_ADS.values(_adm, 'qa3_rms_mjy')}",
        'the search-tier gate: own value after the threshold': "_rm = {**_rm, **_ADS.values(_adm, f'stage2_{_st.lower()}_rms_mjy')}",
        'E15 accepts an admitted night': "and _kd(d) not in _adm:",
        'the beam gate median: the cell\'s nights': "_fin = sorted(v for d, v in _areas.items() if d not in _adm)",
        'the V sanity median: the cell\'s nights': "and _d not in _adm)",
        'the members\' I stack: the cell\'s nights': "_sigma_stack(_ADS.only(resid_I, _ADS.without(dates, _adm))",
        'the members\' V stack: the cell\'s nights': "_sigma_stack(_ADS.only(resid_V, _ADS.without(dates, _adm))",
        'an admitted night\'s I gate uses its own stack': "run_identification_i(rI, master_I, hdr, beam, _ssA_I.get(d, sstack_I)",
    }
    miss = [k for k, v in want.items() if v not in src]
    check('the engine applies each rule at its gate', not miss, miss)
    # order: the median / threshold is taken BEFORE the admitted night's value joins the lookup
    i_med, i_wr = src.find('_med = _fin[len(_fin) // 2]'), src.find("_wr = {**_wr, **_ADS.values(_adm, 'qa3_rms_mjy')}")
    i_thr, i_rm = src.find('_thr = _med + SEARCH_TIER_SIGMA * _mad'), src.find("_rm = {**_rm, **_ADS.values(")
    check('an admitted night\'s value joins the lookup after the population\'s median / threshold is taken (it never moves '
          'them)', 0 < i_med < i_wr and 0 < i_thr < i_rm, (i_med, i_wr, i_thr, i_rm))
    from exopipe.phase3.lib.frames import deep_seam
    try:
        deep_seam.propose(inspect.getsource(E))
        seam = 'ok'
    except ValueError as e:
        seam = str(e)
    check('the frames seam still applies to the engine (its anchors found once each)', seam == 'ok', seam)


# ---------------------------------------------------------------------------------------------------------------------
# part 3: end to end on a copy of 05h_78MHz
# ---------------------------------------------------------------------------------------------------------------------

_DRIVER = r'''
import json, os, sys
sys.path.insert(0, sys.argv[1])
from exopipe.phase3.core import env
env.legacy()
from exopipe.phase3.lib.search import deep_search as E
fail = json.loads(sys.argv[6]) if len(sys.argv) > 6 else None
if fail:                         # force this night's Stokes I residual to fail (the builder then drops it, B-30 counted)
    for _name in ('_resid_task', '_resid_task_spill'):
        _f0 = getattr(E, _name)
        def _w(d, _f=_f0):
            if d == fail and E._RESID_CTX.get('tag') == 'I':
                return d, None
            return _f(d)
        setattr(E, _name, _w)
out = {'ok': False}
try:
    n = E.run_deep_cell(sys.argv[2], sys.argv[3], label='ADMITTEST', included_dates=json.loads(sys.argv[4]), stokes='IV')
    out = {'ok': True, 'n': n}
except Exception as e:
    out = {'ok': False, 'error': f'{type(e).__name__}: {e}'}
try:                             # the run's B-30 events (kind, key, n, detail): a gate's own record of its population
    from exopipe.phase3.core import fallback as _FB
    _FB._drain()
    out['fallbacks'] = [[e[2], e[1], e[3], e[4]] for e in _FB.EVENTS]
except Exception as e:
    out['fallbacks_error'] = f'{type(e).__name__}: {e}'
json.dump(out, open(sys.argv[5], 'w'))
'''


def strip_constituents(path, nights):
    """The master copy without the HISTORY 'COADD constituent <date>' lines of ``nights`` (the pixels and every other card
    unchanged): the nights are then neither constituents nor in weights.csv - a night newer than its master."""
    from astropy.io import fits
    with fits.open(path) as h:
        data, hdr = h[0].data, h[0].header
        hist = [str(x) for x in hdr.get('HISTORY', [])]
        keep = [x for x in hist if not (x.startswith('COADD constituent') and x.split()[-1] in nights)]
        del hdr['HISTORY']
        for x in keep:
            hdr['HISTORY'] = x
        tmp = path + '.part'
        fits.writeto(tmp, data, hdr)
    os.remove(path)
    os.replace(tmp, path)
    return len(hist) - len(keep)


def engine(root, work, label_root, policy, dates, fail_i=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith('P3_')}
    env.update(PYTHONPATH=REPO, P3_COADD_ROOT=root, P3_FLOW_ROOT=work, P3_STAGING_ROOT=label_root,
               P3_IRESID_ROOT=os.path.join(label_root, 'iresid'), P3_VRESID_ROOT=os.path.join(label_root, 'vresid'),
               P3_REFINE_OFF='1', P3_ARCHIVE_WRITE='0', P3_V110_OFF='1', P3_MASTER_GEN='ivw-mean-v3',
               P3_FLOW_LOCK=os.path.join(work, '.flow.lock'))
    env[ADS.POLICY_ENV] = policy
    if fail_i:                                       # production's handling of a failed residual: dropped, counted
        env.update(DEEP_NPROC_RESID='1', EXOPIPE3_FALLBACK_POLICY=json.dumps({'fallbacks.SEARCH_RESIDUAL_FALLBACK': 'fallback'}))
    res = os.path.join(label_root, f'result_{policy}.json')
    os.makedirs(label_root, exist_ok=True)
    p = subprocess.run(['nice', sys.executable, '-c', _DRIVER, REPO, LST, SC, json.dumps(dates), res]
                       + ([json.dumps(fail_i)] if fail_i else []), env=env, capture_output=True, text=True, timeout=5400)
    log = p.stdout + p.stderr
    open(os.path.join(label_root, f'engine_{policy}.log'), 'w').write(log)
    return (json.load(open(res)) if os.path.exists(res) else {'ok': False, 'error': f'rc {p.returncode}'}), log


def test_end_to_end():
    from astropy.io import fits
    sys.path.insert(0, os.path.join(REPO, 'tests'))
    import test_masters_admit6 as M6
    M6.T = os.path.join(T, 'admit6')
    os.makedirs(M6.T, exist_ok=True)
    src = os.path.join(CFG.paths.COADD_SB_ROOT, CELL)
    w = list(csv.DictReader(open(os.path.join(src, 'coadd_final', 'weights.csv'))))
    flows = os.path.join(src, 'flow')
    cands = sorted(r['date'] for r in w if r.get('in_master') == '1' and r.get('qa3_guard') == 'pass'
                   and os.path.exists(os.path.join(flows, f'{r["date"]}_scaled_flow.npy')))
    nights = cands[-2:]
    wt, out = os.path.join(M6.T, 'wt'), os.path.join(M6.T, 'out')
    cd = M6.copy_without(set(nights), wt)
    common = ['--out', out, '--deep-installed-root', wt]
    rc1, t1 = M6.cli('masters', '--stages', 'admit-prep', *common)
    for d in nights:
        for f in (f'{d}_scaled_flow.npy', f'{d}_scaled_dewarp.fits'):
            shutil.copy2(os.path.join(flows, f), os.path.join(cd, A.ADMIT_DIR, 'flow', f))
            os.utime(os.path.join(cd, A.ADMIT_DIR, 'flow', f))
    rc2, t2 = M6.cli('flow', '--stages', 'admit', '--out', out)          # the pairs carry the cell tree's paths
    rc3, t3 = M6.cli('masters', '--stages', 'admit', *common, '--set', 'masters_admit.DC2_MEDIAN_WITHOUT_RECORD=1.0')
    rows = A.admitted_rows(cd)
    check('setup: the two nights admitted into the copy (admit-prep, the flow step\'s admit, admit)',
          (rc1, rc2, rc3) == (0, 0, 0) and sorted(rows) == nights, ((rc1, rc2, rc3), sorted(rows), t2[-300:], t3[-300:]))
    if sorted(rows) != nights:
        return
    # the rest of what a search reads, linked read only: the members' products, leave-one-out masters, PB maps; the
    # admitted nights have no leave-one-out master (they are not in the master)
    for sub in ('subtracted', os.path.join('coadd_final', 'loo')):
        os.makedirs(os.path.join(cd, sub), exist_ok=True)
        for f in sorted(os.listdir(os.path.join(src, sub))):
            if f[:10] not in nights and not os.path.exists(os.path.join(cd, sub, f)):
                os.symlink(os.path.join(src, sub, f), os.path.join(cd, sub, f))
    for f in os.listdir(os.path.join(src, 'coadd_final')):
        if f.startswith('PB_deep') and not os.path.exists(os.path.join(cd, 'coadd_final', f)):
            os.symlink(os.path.join(src, 'coadd_final', f), os.path.join(cd, 'coadd_final', f))
    n = strip_constituents(os.path.join(cd, 'coadd_final', f'I_deep_Taper_R075_{SC}.fits'), set(nights))
    check('the copy\'s master lists neither night as a constituent; its pixels unchanged (the admission still its own)',
          n == 2 and sorted(A.admitted_rows(cd)) == nights, (n, sorted(A.admitted_rows(cd))))
    work, att = os.path.join(T, 'work'), os.path.join(T, 'attempt')
    res, log = engine(wt, work, att, 'absent', [nights[0]])
    check("production's reading: the membership gate refuses the cell (the admitted nights are absent nights)",
          not res['ok'] and 'membership rulings' in res.get('error', '') and 'ABSENT' in res.get('error', ''),
          (res, log[-600:]))
    res, log = engine(wt, work, att, 'admitted', [nights[0]])
    check('ruled reading: the engine runs (incremental, one admitted night)', res['ok'], (res, log[-1500:]))
    pop = [r['date'] for r in w if r['date'] not in nights]
    m = re.search(r'(\d+) admitted night\(s\) \[([^\]]*)\]', log)
    check('it names both admitted nights (judged against the cell\'s own nights)', m and m.group(1) == '2', log[-800:])
    check('E15 accepts the admitted night (no input-provenance refusal of it)',
          not re.search(rf'INPUT-PROVENANCE GATE[^\n]*{nights[0]}', log), log[-800:])
    ss = sorted(os.listdir(os.path.join(att, 'transients', 'Run_ADMITTEST', 'sigstack'))) if os.path.isdir(
        os.path.join(att, 'transients', 'Run_ADMITTEST', 'sigstack')) else []
    own = [f for f in ss if f.endswith(f'_sigstack_{nights[0]}.fits')]
    check('the admitted night\'s own sigma stacks (I and V) are written beside the cell\'s map', len(own) == 2, ss)
    if own:
        h = fits.getheader(os.path.join(att, 'transients', 'Run_ADMITTEST', 'sigstack', own[0]))
        check('the stack records which night it was made for', h.get('ADMITTED') == nights[0], dict(h).get('ADMITTED'))
    # the stack's nights: the cell's nights the gates kept (its record's nights less those a refusing gate dropped -
    # exclusions, the night-image / epoch / 10x sanity / provenance / beam gates; the 3x I sanity and search-tier gates only
    # skip a search), never the other admitted night
    from exopipe.phase3.core import exclusions as EX
    ex = EX.load()
    rd = os.path.join(att, 'transients', 'Run_ADMITTEST', 'refusals')
    dropped = set()
    for kind in ('nightimage', 'epochgate', 'sanity', 'stale_input', 'beam'):
        f = os.path.join(rd, f'{LST}_{SC}_{kind}_excluded.csv')
        if os.path.exists(f):
            dropped |= {ln.split(',')[2] for ln in open(f) if ln.count(',') >= 3}
    want = sorted(d for d in pop if not ex.is_excluded(LST, SC, d) and d not in dropped)
    got = {m.group(1): m.group(3).split(',') for m in re.finditer(
        r'05h 78MHz ([IV]): admitted night (\S+): own sigma stack over \d+ of the cell\'s searched nights \+ itself: (\S*)', log)
        if m.group(2) == nights[0]}
    check('the I stack: exactly the cell\'s nights the gates kept, plus the night itself (not the other admitted night)',
          got.get('I') == want and nights[1] not in got.get('I', []), (got.get('I'), want, sorted(dropped)))
    vwant = [d for d in want if os.path.exists(os.path.join(cd, 'subtracted', f'{d}_V_sub_{SC}.fits'))]
    check('the V stack: those of them with a V product', got.get('V') == vwant, (got.get('V'), vwant))
    # the candidates of the admitted night were judged / annotated with its own stack
    led = list(csv.DictReader(open(os.path.join(att, 'transients', 'Run_ADMITTEST', 'v_deep_ledger.csv'))))
    bad, n = [], 0
    for st in ('I', 'V'):
        p_ = os.path.join(att, 'transients', 'Run_ADMITTEST', 'sigstack', f'{LST}_{SC}_{st}_sigstack_{nights[0]}.fits')
        if not os.path.exists(p_):
            continue
        ssd = np.squeeze(fits.getdata(p_)).astype(float)
        for r in led:
            if r['stokes'] == st and r['date'] == nights[0] and r.get('sstack') not in (None, ''):
                n += 1
                if abs(float(r['sstack']) - round(float(ssd[int(float(r['pix_y'])), int(float(r['pix_x']))]), 2)) > 0.005:
                    bad.append((st, r['pix_x'], r['pix_y'], r['sstack']))
    check('every gate-ledger row of the admitted night carries the value of ITS OWN stack at its pixel', n > 0 and not bad,
          (n, bad[:3]))
    print(f'  (end to end: {len(pop)} nights in the cell\'s record; the log is {att}/engine_admitted.log)')
    A0, B0 = nights
    # MAJOR 1 (eighth review): a night whose I residual fails leaves both searches - and both of the admitted night's stacks
    fail_n = want[0]
    att1 = os.path.join(T, 'attempt_fail')
    res, log1 = engine(wt, work, att1, 'admitted', [A0], fail_i=fail_n)
    st1 = {m.group(1): m.group(3).split(',') for m in re.finditer(
        r"05h 78MHz ([IV]): admitted night (\S+): own sigma stack over \d+ of the cell's searched nights \+ itself: (\S*)", log1)
        if m.group(2) == A0}
    check('MAJOR (eighth review): a night whose I residual failed (production: dropped, counted) is in neither of the admitted '
          'night\'s stacks - the V epochs follow the I-filtered dates, as production\'s',
          res['ok'] and st1.get('I') == [d for d in want if d != fail_n] and st1.get('V') == [d for d in vwant if d != fail_n],
          (res, {k: len(v) for k, v in st1.items()}, fail_n in st1.get('V', []), log1[-600:]))
    # ... and its production half (ninth review: the stacks above come out right with the population filter alone): with no
    # admitted night searched, a night whose I residual failed gets no V residual, so no V search - production's dates
    m1, m2 = want[1], want[0]
    att5 = os.path.join(T, 'attempt_fail_members')
    res5, log5 = engine(wt, work, att5, 'admitted', [m1, m2], fail_i=m2)
    ip5 = os.path.join(att5, 'transients', 'Run_ADMITTEST', 'v_deep_images.csv')
    img5 = [(r['stokes'], r['date']) for r in csv.DictReader(open(ip5))] if os.path.exists(ip5) else []
    vgate = [ev[3] for ev in res5.get('fallbacks', []) if 'Stokes V sanity gate, nights with a V frame rms' in ev[3]]
    check('the production path (ninth review): members searched, one night\'s I residual failing - its V epochs follow the '
          'I-filtered dates: the V sanity gate\'s population is the other night\'s V frame alone (the gate\'s own record: 1 '
          'night), that night searched in I and V, the failed night in neither',
          res5['ok'] and len(vgate) == 1 and ': 1 (needs' in vgate[0] and ('I', m1) in img5 and ('V', m1) in img5
          and not [x for x in img5 if x[1] == m2],
          (res5.get('error'), vgate, res5.get('fallbacks_error'), img5, log5[-500:]))
    # MAJOR 2 (eighth review): an admitted night whose V the same-state rule refused - I searched, V skipped (not refused by E15)
    import io as _io
    aside = os.path.join(T, 'aside_v')
    os.makedirs(aside, exist_ok=True)
    vprods = sorted(glob.glob(os.path.join(cd, 'subtracted', f'{A0}_V_*')))
    adm_p = os.path.join(cd, 'coadd_final', A.ADMISSIONS)
    adm_keep = open(adm_p, newline='').read()
    rows_ = list(csv.DictReader(_io.StringIO(adm_keep)))
    for r in rows_:
        if r['date'] == A0:
            r['v_products'] = 'refused (B-03 same-state): forced (test)'
    with open(adm_p, 'w', newline='') as f:
        dw = csv.DictWriter(f, fieldnames=list(rows_[0].keys()))
        dw.writeheader()
        dw.writerows(rows_)
    for q in vprods:
        os.replace(q, os.path.join(aside, os.path.basename(q)))
    att2 = os.path.join(T, 'attempt_vsame')
    try:
        res, log2 = engine(wt, work, att2, 'admitted', [A0])
    finally:
        for q in vprods:
            os.replace(os.path.join(aside, os.path.basename(q)), q)
        open(adm_p, 'w', newline='').write(adm_keep)
    img2 = list(csv.DictReader(open(os.path.join(att2, 'transients', 'Run_ADMITTEST', 'v_deep_images.csv')))) if os.path.exists(
        os.path.join(att2, 'transients', 'Run_ADMITTEST', 'v_deep_images.csv')) else []
    check('MAJOR (eighth review): an admitted night whose V image the same-state rule refused keeps its I search (E15 skips '
          'its V, as for a member with its refusals_input.csv row) and has no V search',
          res['ok'] and not re.search(rf'INPUT-PROVENANCE GATE[^\n]*{A0}', log2)
          and [r['stokes'] for r in img2 if r['date'] == A0] == ['I'], (res, [(r['stokes'], r['date']) for r in img2],
                                                                       log2[-600:]))
    # MINOR 6 (eighth review): an admitted night is appended alone
    att3 = os.path.join(T, 'attempt_mixed')
    res, log3 = engine(wt, work, att3, 'admitted', [A0, want[0]])
    check('an incremental list holding an admitted night and another night is refused (an admitted night is appended alone)',
          not res['ok'] and 'appended alone' in res.get('error', ''), (res, log3[-400:]))
    # a member's results do not change because nights were admitted (the reviewer's gap): the same member night searched
    # with the admitted nights' products and record present, and with them moved out of the copy
    m = want[0]

    def member_rows(att_):
        rd_ = os.path.join(att_, 'transients', 'Run_ADMITTEST')
        out_ = {}
        for name in ('master_catalog.csv', 'v_deep_ledger.csv', 'v_deep_images.csv'):
            p_ = os.path.join(rd_, name)
            rows__ = list(csv.DictReader(open(p_))) if os.path.exists(p_) else []
            out_[name] = sorted(json.dumps(r, sort_keys=True) for r in rows__ if r.get('date') == m)
        return out_
    attm1 = os.path.join(T, 'attempt_member_with')
    res1, logm1 = engine(wt, work, attm1, 'admitted', [m])
    moved = sorted(glob.glob(os.path.join(cd, 'subtracted', f'{A0}_*')) + glob.glob(os.path.join(cd, 'subtracted', f'{B0}_*'))
                   + [adm_p])
    aside2 = os.path.join(T, 'aside_adm')
    os.makedirs(aside2, exist_ok=True)
    for q in moved:
        os.replace(q, os.path.join(aside2, os.path.basename(q)))
    attm0 = os.path.join(T, 'attempt_member_without')
    try:
        res0, logm0 = engine(wt, work, attm0, 'admitted', [m])
    finally:
        for q in moved:
            os.replace(os.path.join(aside2, os.path.basename(q)), q)
    r1, r0 = member_rows(attm1), member_rows(attm0)
    check('a member night\'s catalogue, gate-ledger and image rows are the same with the admitted nights present and with '
          'them taken out of the cell (no population of its gates or stack holds an admitted night)',
          res1['ok'] and res0['ok'] and r1 == r0 and r1['v_deep_images.csv'],
          (res1, res0, {k: (len(r1[k]), len(r0[k])) for k in r1}, logm1[-300:], logm0[-300:]))


def main(argv):
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_policy()
        cd = _cell(os.path.join(T, 'unit'))
        test_nights(cd)
        test_unreadable_and_same_state(cd)
        test_unit_outputs()
        test_values()
        test_residuals_and_stacks()
        test_engine_source()
        if '--quick' not in argv:
            test_end_to_end()
    finally:
        if not os.environ.get('EXOPIPE3_TEST_KEEP'):
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
