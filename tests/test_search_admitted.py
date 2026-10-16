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
import collections
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


def test_b02_units():
    """B-02 step 3a: removal_rules' search helpers - the switch, a product's time block, the daytime gate on frames whose times
    put the Sun up or down or cannot be read, the track hit; every engine line of them marked WAVE2 B-02."""
    from astropy.io import fits
    from exopipe.phase3.lib.search import removal_rules as RR
    old = os.environ.pop(RR.RULES_ENV, None)
    try:
        ok_cfg = RR.search_rules() == CFG.search_detect.REMOVAL_RULES == 'once-in-search'
        os.environ[RR.RULES_ENV] = 'production'
        ok_env = RR.search_rules() == 'production'
        bad = []
        for v_ in ('Production', ''):                # a typo, an empty value: refused (RuntimeError - a refusal in the child)
            os.environ[RR.RULES_ENV] = v_
            try:
                RR.search_rules()
            except RuntimeError as e_:
                bad.append('REFUSING' in str(e_))
        ok_bad = bad == [True, True]
    finally:
        os.environ.pop(RR.RULES_ENV, None)
        if old is not None:
            os.environ[RR.RULES_ENV] = old
    check("search_detect.REMOVAL_RULES: the ruled 'once-in-search' by default, the step's value through the engine env, an "
          "unknown or empty value refused", ok_cfg and ok_env and ok_bad, (ok_cfg, ok_env, bad))
    d = os.path.join(T, 'b02_units')
    os.makedirs(d, exist_ok=True)

    def prod(name, **cards):
        h = fits.Header()
        for k_, v_ in cards.items():
            h[k_.replace('_', '-')] = v_
        p_ = os.path.join(d, name)
        fits.writeto(p_, np.zeros((4, 4), np.float32), h, overwrite=True)
        return p_
    night = prod('night.fits', DATE_OBS='2025-01-11T08:00:00.000', DATE_END='2025-01-11T09:00:00.000')   # local midnight (PST)
    day = prod('day.fits', DATE_OBS='2025-06-21T20:00:00.000', DURATION=3600.0)                          # local 13:00 (PDT)
    none_ = prod('none.fits')
    check('product_times: DATE-END, else DATE-OBS + DURATION, else None', RR.product_times(night)[1].startswith('2025-01-11T09')
          and RR.product_times(day)[1].startswith('2025-06-21T21:00') and RR.product_times(none_) is None,
          (RR.product_times(night), RR.product_times(day), RR.product_times(none_)))
    got = RR.daytime_frames({('I', '05h', '78MHz', '2025-01-11'): night, ('V', '05h', '78MHz', '2025-06-21'): day,
                             ('I', '05h', '78MHz', '2025-02-02'): none_}, quar={})
    check('daytime_frames: the night frame passes, the Sun-up frame is refused in its Stokes with the conformance\'s words, '
          'a frame without times is refused', set(got) == {('V', '2025-06-21'), ('I', '2025-02-02')}
          and got[('V', '2025-06-21')].startswith('daytime: Sun') and got[('I', '2025-02-02')] == RR.DAYTIME_UNREADABLE, got)
    dusk = prod('dusk.fits', DATE_OBS='2025-06-22T03:40:00.000', DATE_END='2025-06-22T04:40:00.000')   # the Sun to -4.7 deg
    kd = ('I', '05h', '78MHz', '2025-06-22')
    tw_q = RR.daytime_frames({kd: dusk}, quar={('05h', '2025-06-22'): {'78'}})
    tw_n = RR.daytime_frames({kd: dusk}, quar={})
    check('daytime_frames: twilight refused in a band the solar audit quarantines that night, admitted elsewhere',
          list(tw_q) == [('I', '2025-06-22')] and tw_q[('I', '2025-06-22')].startswith('twilight in a solar-audit QUARANTINE')
          and tw_n == {}, (tw_q, tw_n))
    olda = os.environ.pop(RR.AUDIT_ENV, None)
    try:
        os.environ[RR.AUDIT_ENV] = CFG.search_tables.SOLAR_AUDIT          # the real audit, named as the step names it
        real = RR.daytime_frames({('I', '05h', '78MHz', '2025-01-11'): night})
        os.environ[RR.AUDIT_ENV] = os.path.join(d, 'no_such_audit.csv')
        try:
            RR.daytime_frames({('I', '05h', '78MHz', '2025-01-11'): night})
            miss = None
        except RuntimeError as e_:
            miss = str(e_)
    finally:
        os.environ.pop(RR.AUDIT_ENV, None)
        if olda is not None:
            os.environ[RR.AUDIT_ENV] = olda
    check('daytime_frames: the quarantine bands from the audit the step names (EXOPIPE3_SOLAR_AUDIT); without that file the '
          'cell refuses (RuntimeError with the conformance\'s words)', real == {} and miss is not None and 'REFUSING' in miss,
          (real, miss))
    wcsv = os.path.join(d, 'weights.csv')
    with open(wcsv, 'w') as fh:
        fh.write('date,in_master\n2024-12-18,1\n2024-13-45,1\n2024-12-19,0\n')
    sk1, sk2 = [], []
    pts1 = RR.jupiter_track('05h', wcsv, skipped=sk1)
    pts2 = RR.jupiter_track('05h', os.path.join(d, 'no_weights.csv'), skipped=sk2)
    check('jupiter_track: the co-added nights it cannot place are named to the caller (one bad date; no weights.csv: every '
          'night), the others placed; a night not in the master not tracked', [p_[2] for p_ in pts1] == ['2024-12-18']
          and len(sk1) == 1 and sk1[0].startswith('2024-13-45') and pts2 == [] and len(sk2) == 1, (pts1, sk1, sk2))
    from exopipe.phase3.lib.search import tests as ST
    check('the search step\'s content tests know the new veto reason (with a night) and refusal kind (6 columns, the Stokes of '
          'column 4)', ST.known_veto_reason('jupiter-track:2024-12-30') and not ST.known_veto_reason('jupiter-track:')
          and not ST.known_veto_reason('jupiter-track:x') and ST.REFUSAL_KINDS.get('daytime') == (6, 'row'),
          ST.REFUSAL_KINDS.get('daytime'))
    pts = [(80.0, 22.0, '2024-12-30'), (90.0, 23.0, '2025-01-20')]
    check('jupiter_track_hit: the first co-added night within the band\'s radius (Blue 0.5 deg), none outside',
          RR.jupiter_track_hit(80.3, 22.1, '78MHz', pts) == '2024-12-30' and RR.jupiter_track_hit(85.0, 22.0, '78MHz', pts) is None
          and RR.jupiter_track_hit(84.0, 22.0, 'Red', pts) == '2024-12-30',
          (RR.jupiter_track_hit(80.3, 22.1, '78MHz', pts), RR.jupiter_track_hit(84.0, 22.0, 'Red', pts)))
    from exopipe.phase3.lib.search import deep_search as DSE
    src = inspect.getsource(DSE).splitlines()
    unmarked = [ln for ln in src if re.search(r'_RR\b|_b02\b|_jtrack\b|_jth\b|_jskip\b|_dfr\b|_day\b|_dw\b|_sk\b', ln)
                and 'WAVE2 B-02' not in ln and not ln.strip().startswith('#') and not ln.strip().startswith('if _jtrack else')]
    check('every engine line of the search\'s removal rules is marked WAVE2 B-02 (a continuation excepted)', not unmarked,
          unmarked[:4])
    D_ = '2024-12-1'
    fr_ = RR.daytime_candidate_frames('05h', '78MHz', '/b', [D_ + '8', D_ + '9'],
                                      {'I': {D_ + '8': 1, D_ + '9': 1, '2024-12-20': 1}, 'V': {D_ + '8': 1}},
                                      {'I': {D_ + '9'}, 'V': set()})
    check('the daytime gate judges each Stokes\' searched nights with a residual, none another gate refused, none of the '
          'nights a stack only needs', sorted(fr_) == [('I', '05h', '78MHz', D_ + '8'), ('V', '05h', '78MHz', D_ + '8')]
          and fr_[('I', '05h', '78MHz', D_ + '8')] == '/b/subtracted/2024-12-18_I_sub_78MHz.fits', fr_)
    trk = [ln for ln in src if 'jupiter_track_hit(' in ln and 'WAVE2 B-02' in ln]
    check('the engine tests the track at the catalogue row\'s 5-decimal position in both chains (as candidates.write_candidates '
          'rounds; pinned in the source - the effect is under 1e-5 deg)', len(trk) == 2
          and all(ln.count('round(float(') == 2 and ln.count(', 5)') == 2 for ln in trk), trk)
    srcj = inspect.getsource(DSE)
    ix_a = [m.start() for m in re.finditer(r"reason='ateam', pass_ms=1\)\)", srcj)]
    ix_t = [m.start() for m in re.finditer(r"reason=f'jupiter-track:\{_jth\}'", srcj)]
    ix_c = [m.start() for m in re.finditer(r'\n\s+cut = False\n', srcj)]
    check('the track sits right after the A-team veto in both chains (the V gating loop, the I finals) and before the I table',
          len(ix_a) == 2 and len(ix_t) == 2 and len(ix_c) == 1 and ix_a[0] < ix_t[0] < ix_a[1] < ix_t[1] < ix_c[0],
          (ix_a, ix_t, ix_c))


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
from exopipe.phase3.lib.search import removal_rules as _RR
_b02 = json.loads(os.environ.get('TEST_B02') or 'null')
_calls = {'daytime_frames': [], 'jupiter_track': []}
if _b02 is not None:             # B-02 step 3a: the frames the daytime gate refuses and the track points, given (deterministic);
    _real_day, _real_jt = _RR.daytime_frames, _RR.jupiter_track   # each call recorded, the real rule run on its arguments too

    def _day(frames, quar=None):
        real = _real_day(frames, quar)
        _calls['daytime_frames'].append({'frames': sorted([list(k), p] for k, p in frames.items()),
                                         'real': sorted([list(k), v] for k, v in real.items())})
        if _b02.get('real'):                     # record only: the real rule's verdicts
            return real
        return {(st, d): 'daytime: Sun 5.0 to 10.0 deg over the frame (test)' for st, d in _b02.get('day', [])
                if any(k[0] == st and k[3] == d for k in frames)}

    def _jt(lst, weights_path, nights=None, skipped=None):
        sk = []
        real = _real_jt(lst, weights_path, nights, skipped=sk)
        _calls['jupiter_track'].append({'lst': lst, 'weights': weights_path, 'n_real': len(real), 'skipped': sk})
        if skipped is not None:                  # the nights the track cannot place, to the engine's B-30 site: the real ones,
            skipped.extend(sk if _b02.get('real') else _b02.get('skip', []))   # or those given
        if _b02.get('real'):
            return real
        return [tuple(x) for x in _b02.get('track', [])]
    _RR.daytime_frames, _RR.jupiter_track = _day, _jt
if os.environ.get('TEST_OPEN_JUPITER'):    # B-02 test device: the per-night Jupiter veto opened (both runs of a pair); I only -
    if 'V' in (os.environ.get('TEST_STOKES') or 'IV'):   # the V chain takes a zero radius as 'no Jupiter' and fails on it
        raise SystemExit('TEST_OPEN_JUPITER is a device of the Stokes I pair only')
    E.JUPITER_VETO_DEG = {k: 0.0 for k in E.JUPITER_VETO_DEG}
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
    n = E.run_deep_cell(sys.argv[2], sys.argv[3], label='ADMITTEST', included_dates=json.loads(sys.argv[4]),
                        stokes=os.environ.get('TEST_STOKES') or 'IV')
    out = {'ok': True, 'n': n}
except Exception as e:
    out = {'ok': False, 'error': f'{type(e).__name__}: {e}'}
out['b02_calls'] = _calls
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


def engine(root, work, label_root, policy, dates, fail_i=None, rules=None, b02=None, stokes=None, open_jupiter=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith('P3_')}
    env.update(PYTHONPATH=REPO, P3_COADD_ROOT=root, P3_FLOW_ROOT=work, P3_STAGING_ROOT=label_root,
               P3_IRESID_ROOT=os.path.join(label_root, 'iresid'), P3_VRESID_ROOT=os.path.join(label_root, 'vresid'),
               P3_REFINE_OFF='1', P3_ARCHIVE_WRITE='0', P3_V110_OFF='1', P3_MASTER_GEN='ivw-mean-v3',
               P3_FLOW_LOCK=os.path.join(work, '.flow.lock'))
    env[ADS.POLICY_ENV] = policy
    if rules:                                        # B-02: the search's removal rules (else the config's, as the step passes)
        env['EXOPIPE3_REMOVAL_RULES'] = rules
    if b02 is not None:
        env['TEST_B02'] = json.dumps(b02)
    if stokes:                                       # one Stokes per run, as the search step runs the engine (else 'IV')
        env['TEST_STOKES'] = stokes
    if open_jupiter:
        env['TEST_OPEN_JUPITER'] = '1'
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
    res, log = engine(wt, work, att, 'admitted', [nights[0]], b02={'real': True})   # B-02 hooks record only
    check('ruled reading: the engine runs (incremental, one admitted night)', res['ok'], (res, log[-1500:]))
    fr0 = ((res.get('b02_calls') or {}).get('daytime_frames') or [{}])[0].get('frames', [])
    check("B-02: an admitted night's search judges its own frames only (I and V) - not the cell's other nights, whose "
          "residuals it builds for its stack", res['ok'] and fr0 and {x[0][3] for x in fr0} == {nights[0]}
          and {x[0][0] for x in fr0} == {'I', 'V'}, [x[0] for x in fr0][:6])
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
    test_b02_end_to_end(wt, work, want[:3])


def test_b02_end_to_end(wt, work, b3):
    """B-02 step 3a: production's rules against 'once-in-search' on three member nights (``b3``) of the copy ``wt``, one
    Stokes per run as the search step runs the engine (the harness's 'IV' call skips a night's V search with its refused I
    frame - production's per-date loop - which no step or fleet call meets). Each run is seeded with the served search's
    cell sigma stack (the map an incremental search reads, so a search of a few nights gates as the full search did and
    detections reach the end of each chain). Given: the daytime gate refuses one night's frame; the Jupiter track holds one
    point, on a detection that reached the track's place in its chain (past the A-team veto) under production's rules. In
    this cell no I detection of these nights gets that far but Jupiter's own, which the per-night Jupiter veto removes
    first, so the I pair runs with that veto opened in both runs (a test device; the track then meets Jupiter itself)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from exopipe.phase3.lib.search import removal_rules as RR, tests as ST
    ssrc = os.path.join(CFG.calibrate.SERVED_RUN['deep'], 'sigstack')
    cdir = os.path.join(wt, CELL)
    W = WCS(fits.getheader(os.path.join(cdir, 'coadd_final', f'I_deep_Taper_R075_{SC}.fits'))).celestial

    def sky(r):                                      # the catalogue's position of a ledger row: its pixel's, 5 decimals
        s_ = W.pixel_to_world(int(float(r['pix_x'])), int(float(r['pix_y'])))
        return round(float(s_.ra.deg), 5), round(float(s_.dec.deg), 5)

    def b02_run(name, st, rules, b02, open_jupiter):
        att_ = os.path.join(T, name)
        shutil.rmtree(att_, ignore_errors=True)      # a fresh attempt (the engine appends to its ledgers)
        sg = os.path.join(att_, 'transients', 'Run_ADMITTEST', 'sigstack')
        os.makedirs(sg, exist_ok=True)
        shutil.copy2(os.path.join(ssrc, f'{CELL}_{st}_sigstack.fits'), sg)
        res_, log_ = engine(wt, work, att_, 'admitted', b3, rules=rules, b02=b02, stokes=st, open_jupiter=open_jupiter)
        rd_ = os.path.join(att_, 'transients', 'Run_ADMITTEST')
        out_ = {}
        for name_ in ('v_deep_ledger.csv', 'v_deep_images.csv'):
            p_ = os.path.join(rd_, name_)
            out_[name_] = [r for r in csv.DictReader(open(p_)) if r.get('date') in b3] if os.path.exists(p_) else []
        dayf = os.path.join(rd_, 'refusals', f'{CELL}_daytime_excluded.csv')
        out_['day'] = open(dayf).read().splitlines() if os.path.exists(dayf) else []
        out_['att'] = att_
        return res_, log_, out_

    def reach(r):                                    # past the A-team veto: v110 (V), the table, the row line or a final
        return r['final'] == '1' or r['veto_reason'] in ('table', 'row-line') or r['veto_reason'].startswith('v110:')
    key = lambda r: (r['date'], r['ra'], r['dec'])
    cols = ('n_ident5', 'n_after_morph', 'n_after_mstruct', 'n_after_jupiter', 'n_final', 'n_raw_floor')
    early = {'I': 'mstruct_I', 'V': 'mstruct'}       # a gate before the A-team veto in each chain
    for st, night, night2, oj in (('I', '2024-12-30', '2024-12-31', True), ('V', '2025-01-04', '2025-01-05', False)):
        dev = ' (the per-night Jupiter veto opened in both runs)' if oj else ''
        resp, logp, rp = b02_run(f'attempt_b02_{st}_production', st, 'production', {'day': [], 'track': []}, oj)
        cand = sorted((r for r in rp['v_deep_ledger.csv'] if r['stokes'] == st and reach(r)),
                      key=lambda r: (r['final'] != '1', r['date']))
        if not (resp['ok'] and cand):
            check(f"B-02 setup ({st}): a detection of the nights past the A-team veto under production's rules{dev}", False,
                  (resp.get('error'), len(rp['v_deep_ledger.csv']), logp[-300:]))
            continue
        # the track point: inside the band's radius at the target's catalogue position (its pixel's), outside it at the source
        # finder's centroid (the ledger's ra/dec) - a track tested at the centroid misses it (the first review's MAJOR)
        R_ = RR.jupiter_track_radius(SC)
        tgt = ra0 = de0 = None
        for r in cand:
            (px, py), (cx, cy) = sky(r), (float(r['ra']), float(r['dec']))
            ux, uy = ((px - cx + 180.0) % 360.0 - 180.0) * np.cos(np.radians(py)), py - cy
            dl = float(np.hypot(ux, uy))
            if dl < 2e-3:
                continue
            P = (px + ux / dl * (R_ - dl / 2) / np.cos(np.radians(py)), py + uy / dl * (R_ - dl / 2))
            if RR.jupiter_track_hit(px, py, SC, [(*P, 'x')]) and not RR.jupiter_track_hit(cx, cy, SC, [(*P, 'x')]):
                tgt, (ra0, de0) = r, (float(P[0]), float(P[1]))
                break
        if tgt is None:
            check(f"B-02 setup ({st}): a detection past the A-team veto whose pixel and centroid are 0.002 deg apart or more{dev}",
                  False, [(r['date'], r['ra'], r['dec'], sky(r)) for r in cand[:5]])
            continue
        dn = next(d for d in b3 if d != tgt['date'])     # the daytime frame: another night's, of this Stokes
        # a second point on a row an earlier gate removed, far from the first: the row keeps its reason (the track's place)
        erow = next((r for r in rp['v_deep_ledger.csv'] if r['stokes'] == st and r['veto_reason'] == early[st]
                     and r['date'] != dn and RR.jupiter_track_hit(*sky(r), SC, [(ra0, de0, 'x')]) is None
                     and RR.jupiter_track_hit(ra0, de0, SC, [(*sky(r), 'x')]) is None), None)
        pts = [(ra0, de0, night)] + ([(*sky(erow), night2)] if erow else [])
        reso, logo, ro = b02_run(f'attempt_b02_{st}_once', st, 'once-in-search',
                                 {'day': [[st, dn]], 'track': [list(x) for x in pts]}, oj)
        lp = {key(r): r for r in rp['v_deep_ledger.csv'] if r['stokes'] == st}
        lo = {key(r): r for r in ro['v_deep_ledger.csv'] if r['stokes'] == st}
        ip = {r['date']: r for r in rp['v_deep_images.csv'] if r['stokes'] == st}
        io = {r['date']: r for r in ro['v_deep_images.csv'] if r['stokes'] == st}
        zeros = ('0', '0', '', '', '0', '0') if st == 'I' else ('0',) * 6
        check(f"B-02 step 3a ({st}): under 'once-in-search' the frame the daytime gate refuses is not searched (its image row "
              f"of zeros, no ledger row, one row in refusals/<cell>_daytime_excluded.csv); under 'production' it is{dev}",
              reso['ok'] and any(k[0] == dn for k in lp) and not any(k[0] == dn for k in lo) and dn in io
              and tuple(io[dn][c] for c in cols) == zeros and len(ro['day']) == 1
              and ro['day'][0].startswith(f'{LST},{SC},{dn},{st},') and not rp['day'],
              (reso.get('error'), dn, io.get(dn), ro['day'], rp['day'], logo[-300:]))
        # the track: every detection past the A-team veto within the band's radius of a point (at its catalogue position), of
        # any night searched under both, vetoed, naming the first point's night; the row an earlier gate removed keeps it
        hit = {k: RR.jupiter_track_hit(*sky(r), SC, pts) for k, r in lp.items()}
        want_v = {k for k, r in lp.items() if k[0] != dn and reach(r) and hit[k]}
        got_v = {k for k, r in lo.items() if r['veto_reason'].startswith('jupiter-track')}
        ok_v = bool(want_v) and want_v == got_v and all(
            lo[k]['veto_reason'] == f'jupiter-track:{hit[k]}' and lo[k]['final'] == '0' for k in got_v)
        ek = key(erow) if erow else None
        check(f"B-02 step 3a ({st}): a track point on a row an earlier gate removed ({early[st]}) leaves its reason - the track "
              f"sits after that gate{dev}", erow is not None and lo.get(ek, {}).get('veto_reason') == early[st],
              (ek, lo.get(ek, {}).get('veto_reason') if ek else None))
        check(f"B-02 step 3a ({st}): the track point sits inside the radius at the target's catalogue position and outside it at "
              f"its centroid, and the target is vetoed - the track is tested at the catalogue position{dev}",
              key(tgt) in got_v, (key(tgt), lo.get(key(tgt), {}).get('veto_reason')))
        # the hooks: none under production; once each under once-in-search, on what the search reads
        cp, co = resp.get('b02_calls', {}), reso.get('b02_calls', {})
        want_fr = sorted([[st, LST, SC, d], os.path.join(cdir, 'subtracted', f'{d}_{st}_sub_{SC}.fits')] for d in b3)
        dcall = (co.get('daytime_frames') or [{}])[0]
        jcall = (co.get('jupiter_track') or [{}])[0]
        check(f"B-02 ({st}): under 'production' the search calls neither rule; under 'once-in-search' the daytime gate once, on "
              f"the Stage-1 products of the searched Stokes and nights (the real rule on them: all dark), the track once, on the "
              f"cell's weights.csv (the real track: every co-added night placed){dev}",
              cp == {'daytime_frames': [], 'jupiter_track': []} and len(co.get('daytime_frames', [])) == 1
              and len(co.get('jupiter_track', [])) == 1 and dcall.get('frames') == want_fr and dcall.get('real') == []
              and all(os.path.exists(x[1]) for x in want_fr)
              and jcall.get('weights') == os.path.join(cdir, 'coadd_final', 'weights.csv') and jcall.get('n_real', 0) > 0
              and jcall.get('skipped') == [], (cp, dcall, jcall))
        # the refused frame still archived; the search step's content tests on the once-in-search outputs (its verify)
        arch = os.path.join(ro['att'], 'iresid' if st == 'I' else 'vresid', f'{LST}_{SC}', f'{dn}_{st}_redo_resid_{SC}.fits')
        rdo = os.path.join(ro['att'], 'transients', 'Run_ADMITTEST')
        for sub in ('iresid', 'vresid'):                 # the step's layout: <attempt>/Run_<label>/<sub> (harness: <attempt>/<sub>)
            if os.path.isdir(os.path.join(ro['att'], sub)):
                os.makedirs(os.path.join(ro['att'], 'Run_ADMITTEST'), exist_ok=True)
                os.symlink(os.path.join(ro['att'], sub), os.path.join(ro['att'], 'Run_ADMITTEST', sub))
        ct = [ST.complete(os.path.join(rdo, 'v_deep_ledger.csv'), stokes=st, cell=CELL, tier='deep'),
              ST.parses(os.path.join(rdo, 'refusals', f'{CELL}_daytime_excluded.csv')),
              ST.one_row_per_searched_epoch(os.path.join(rdo, 'v_deep_images.csv'), stokes=st, cell=CELL, tier='deep')]
        check(f"B-02 step 3a ({st}): the refused frame is still archived, and the search step's content tests pass on the "
              f"'once-in-search' outputs (gate ledger, the daytime refusal record, image ledger){dev}",
              os.path.exists(arch) and all(t_[0] for t_ in ct), (os.path.exists(arch), [t_[2][:300] for t_ in ct]))
        # every other ledger and image row of the nights searched under both is production's, but for the counts of the
        # vetoed: a gated V row's n_v_epoch counts what its epoch's gates kept (an A-team veto's rule), the image row's
        # n_final what the night kept to its end (V: after the gates - every vetoed row; I: the finals)
        nv = collections.Counter(k[0] for k in got_v)
        nf = collections.Counter(k[0] for k in got_v if st == 'V' or lp[k]['final'] == '1')
        diff = []
        for k, r in lp.items():
            if k[0] == dn or k in got_v:
                continue
            exp = dict(r)
            if st == 'V' and r['n_v_epoch'] not in ('', None):
                exp['n_v_epoch'] = str(int(r['n_v_epoch']) - nv[k[0]])
            if lo.get(k) != exp:
                diff.append((k, {c: (exp[c], lo[k][c]) for c in exp if exp[c] != lo[k][c]} if k in lo else 'missing'))
        extra = [k for k in lo if k not in lp]
        idiff = []
        for d, r in ip.items():
            if d == dn:
                continue
            exp = dict(r)
            if r['n_final'] not in ('', None):
                exp['n_final'] = str(int(r['n_final']) - nf[d])
            if io.get(d) != exp:
                idiff.append((d, exp, io.get(d)))
        check(f"B-02 step 3a ({st}): the track vetoes every detection past the A-team veto within the band's radius of its "
              f"point, 'jupiter-track:<night>' (final 0), under 'once-in-search' only; every other ledger and image row is "
              f"production's (their counts less the vetoed){dev}", ok_v and not diff and not extra and not idiff,
              (tgt['date'], tgt['veto_reason'] or 'final', sorted(want_v), sorted(got_v), diff[:3], extra[:3], idiff[:2]))
        check(f"B-02 ({st}): the search's log names the daytime refusal under 'once-in-search' only",
              'DAYTIME GATE (B-02)' in logo and 'DAYTIME GATE' not in logp, (logo[-300:], logp[-300:]))
    # the B-30 site: a co-added night the track cannot place refuses the search under the ruled policy, naming the site
    ress, logs, _rs = b02_run('attempt_b02_track_skip', 'I', 'once-in-search',
                              {'day': [], 'track': [], 'skip': ['2024-12-25 (test: no position)']}, False)
    check("B-02 step 3a: a co-added night the Jupiter track cannot place refuses the search (B-30, ruled 'refuse'), naming "
          "fallbacks.SEARCH_JUPITER_TRACK_FALLBACK and the night", not ress['ok']
          and 'SEARCH_JUPITER_TRACK_FALLBACK' in ress.get('error', '') and '2024-12-25' in ress.get('error', ''),
          (ress.get('error'), logs[-300:]))

def main(argv):
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_policy()
        cd = _cell(os.path.join(T, 'unit'))
        test_nights(cd)
        test_unreadable_and_same_state(cd)
        test_unit_outputs()
        test_b02_units()
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
