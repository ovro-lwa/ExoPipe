"""Per-hour refusals of the deep search with a named reason (a deep append, included_dates; batch unchanged).

R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS (2), "Approved.": frozen statistics from earlier nights judge a new night. A thin
population is the recorded outcome - the skip the whole-cell search made, counted under fallbacks.SEARCH_GATE_SKIP_PERHOUR_FALLBACK
(R-BEAM-GATE-FIVE-NIGHT-FLOOR-WAIVED-THIN-CELLS-SEARCHED, Gregg 2026-09-11 "Waive the floor. I want a complete Masters set."); a
frozen value that is MISSING refuses.

Checks: (a) the skip key, its ruling, default and parity value, and the engine's cell-median gates passing the per-hour flag;
(b) the V sanity gate's recorded population (gate_population 'v_sanity'): written by a whole-cell search, thin told from missing
by frozen_v_sanity (no row, an 'applied' row without a frame_noise.csv median, another factor or minimum, an inconsistent row
refuse; 'below-minimum' is returned), read before the gate's try (a refusal is never caught), the content test; (c) the wideband
member mirror: a searched member night without the value a running gate judges it by is refused in that Stokes, a gate below its
minimum judges nobody, 18 MHz and a member the image does not hold are not judged, a record without perhour is the batch record.
Real data (read only, the production co-add root; skipped without it): on three wideband cells the per-hour record and verdict
equal the batch ones for every measured night, the added refusals are exactly the unmeasured nights; the census of unmeasured
present member nights over every subband cell is printed.
Run: python tests/test_search_perhour_refusals.py
"""
import csv
import glob
import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.core import env as ENV                          # noqa: E402
ENV.legacy()
from exopipe.phase3.config import CFG                               # noqa: E402
from exopipe.phase3.lib.search import gate_population as GP         # noqa: E402
from exopipe.phase3.lib.search import wideband_admission as W       # noqa: E402

T = '/tmp/exopipe3_tests_perhour_refusals'
FAILS, N = [], [0]
P = dict(ratio=2.0, tier_min=5, sanity_x_both=10.0, sanity_x_i=3.0, sanity_min=5, beam_ratio=1.5, beam_min=5)
KP = 'fallbacks.SEARCH_GATE_SKIP_PERHOUR_FALLBACK'


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def _refuses(f):
    try:
        f()
        return 'no refusal'
    except RuntimeError as e:
        return str(e)


def _src():
    return open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_search.py')).read()


def test_skip_key():
    from exopipe.phase3.core import fallback
    ids = CFG.ruled(KP) or ()
    ids = (ids,) if isinstance(ids, str) else tuple(ids)
    par = open(os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg')).read()
    check('(a) fallbacks.SEARCH_GATE_SKIP_PERHOUR_FALLBACK: a fallback key, default fallback, ruled by the beam-floor waiver, set to '
          'fallback in the parity file', KP in fallback.all_keys() and CFG.get(KP) == 'fallback'
          and 'R-BEAM-GATE-FIVE-NIGHT-FLOOR-WAIVED-THIN-CELLS-SEARCHED' in ids
          and re.search(rf"^{re.escape(KP)} = 'fallback'$", par, re.M) is not None, (CFG.get(KP), ids))
    src = _src()
    check('(a) the per-hour flag is a deep append only; the I sanity, search-tier and V sanity gates pass it, the frozen beam '
          'and V skips count under it; the whole-cell expressions are unchanged',
          "_ph = cadence == 'deep' and bool(included_dates)" in src
          and "Stokes I sanity gate, nights with a qa3 rms', perhour=_ph)" in src
          and "with a stage-1 rms', count=_st in stokes, perhour=_ph)" in src
          and "count=bool(resid_V), perhour=_ph)" in src
          and "f'frozen population', perhour=True)" in src
          and "f'attempt\\'s recorded V population', perhour=True)" in src
          and "_ok_v = _enough(len(_vf), SANITY_MIN_NIGHTS," in src
          and "_vmed = _vb if _vb is not None else _vf[len(_vf) // 2]" in src
          and src.count("fallback.site('fallbacks.SEARCH_GATE_SKIP_PERHOUR_FALLBACK'") == 1
          and 'recorded no V noise-rule median - the gate is skipped' not in src)


def test_v_sanity_record():
    d = os.path.join(T, 'v')
    os.makedirs(d)
    nights = [f'2025-01-{i:02d}' for i in range(1, 8)]
    from exopipe.phase3.lib.search import deep_search as DS
    x, mn = DS.SANITY_I_X, DS.SANITY_MIN_NIGHTS        # what the engine records and reads with
    miss = _refuses(lambda: GP.frozen_v_sanity(d, '10h', 'Blue', x, mn))
    GP.append(d, [GP.row('10h', 'Blue', 'v_sanity', nights[:3], None, x, mn, 'below-minimum', 'search', '20261101T000000')])
    GP.append(d, [GP.row('10h', 'Green', 'v_sanity', nights, 0.012, x, mn, 'applied', 'search', '20261101T000000')])
    GP.append(d, [GP.row('10h', 'Red', 'v_sanity', nights, None, x, mn, 'below-minimum', 'search', '20261101T000000')])
    thin = GP.frozen_v_sanity(d, '10h', 'Blue', x, mn)
    check('(b) the thin outcome is a recorded value: below-minimum with its n is returned; no row refuses (thin cannot be told '
          'from missing)', thin['outcome'] == 'below-minimum' and thin['n'] == 3 and thin['median'] is None
          and 'cannot be told from a missing value' in miss and 'REFUSING' in miss, (thin, miss))
    check('(b) an applied row (the base should carry the frame_noise.csv median), a below-minimum row with n at the minimum, '
          'another factor and another minimum refuse',
          'inconsistent' in _refuses(lambda: GP.frozen_v_sanity(d, '10h', 'Green', x, mn))
          and 'inconsistent' in _refuses(lambda: GP.frozen_v_sanity(d, '10h', 'Red', x, mn))
          and 'REFUSING' in _refuses(lambda: GP.frozen_v_sanity(d, '10h', 'Blue', x + 1, mn))
          and 'REFUSING' in _refuses(lambda: GP.frozen_v_sanity(d, '10h', 'Blue', x, mn + 1)))
    from exopipe.phase3.core import validate
    from exopipe.phase3.lib.search import tests as _st      # noqa: F401 (registers the content tests)
    one = os.path.join(T, 'v1')
    os.makedirs(one)
    GP.append(one, [GP.row('10h', 'Blue', 'beam', nights, 0.25, 1.5, 5, 'applied', 'search', 't'),
                    GP.row('10h', 'Blue', 'v_sanity', nights[:3], None, x, mn, 'below-minimum', 'search', 't')])
    bad = os.path.join(T, 'v2')
    os.makedirs(bad)
    GP.append(bad, [GP.row('10h', 'Blue', 'v_sanity', nights[:3], 0.01, x, mn, 'applied', 'search', 't')])
    r1 = validate.run('gate_population', os.path.join(one, GP.FILE), cell='10h_Blue')
    r2 = validate.run('gate_population', os.path.join(bad, GP.FILE), cell='10h_Blue')
    check('(b) content test gate_population: passes a beam and a v_sanity row; fails a v_sanity row applied below its minimum',
          r1 and all(z['passed'] for z in r1) and r2 and not all(z['passed'] for z in r2), (r1, r2))
    src = _src()
    i_b0 = src.find("_vb0 = _RR.base_vsanmed(run_root) if (_b02 and included_dates) else None")
    i_read = src.find('_vfz = _GP.frozen_v_sanity(run_root, lst, scope, SANITY_I_X, SANITY_MIN_NIGHTS)')
    i_try = src.find('    try:\n', i_read)
    i_vr = src.find('        _vr = {}\n', i_read)
    i_row = src.find("_vsrow = _GP.row(lst, scope, 'v_sanity', _vfn, _vf[len(_vf) // 2] if _ok_v else None,", i_vr)
    i_exc = src.find("fallback.site('fallbacks.SEARCH_SANITY_FALLBACK', f'{lst} {scope}: the V sanity gate could not run", i_row)
    i_wr = src.find('    else:', i_exc)
    check('(b) the engine: an append reads its base\'s V median ONCE and, without one, the base\'s V row, both BEFORE the gate\'s '
          'try (a refusal is never caught); a whole-cell deep search with a V search builds the row inside the gate (median '
          '_vf[len // 2] when applied, the recorded below-minimum otherwise) and writes it AFTER the try, only when the gate ran '
          'to its end (a write failure never drops the gate\'s refusals)', 0 < i_b0 < i_read < i_try < i_vr < i_row < i_exc < i_wr
          and "if _b02 and included_dates and resid_V and _vb0 is None:" in src and "_vb = _vb0" in src
          and "if cadence == 'deep' and not included_dates and resid_V:" in src
          and src.find('_GP.append(run_root, [_vsrow])', i_wr) == src.find('_GP.append(run_root, [_vsrow])') > i_wr
          and src.count('_RR.base_vsanmed(run_root)') == 1, (i_b0, i_read, i_try, i_vr, i_row, i_exc, i_wr))


def _rows():
    days = [f'2025-01-{i:02d}' for i in range(1, 8)]
    w = [dict(date=x, qa3_rms_mjy='' if x == days[5] else str(10.0 + i)) for i, x in enumerate(days)]
    p = [dict(date=x, i_rms_mjy=str(5.0 + i), v_rms_mjy='nan' if x == days[4] else str(3.0 + i)) for i, x in enumerate(days)]
    adm = {'2025-02-01': dict(qa3_rms_mjy='12', stage2_i_rms_mjy='6', stage2_v_rms_mjy='')}
    return days, w, p, adm


def test_wideband_synthetic():
    days, w, p, adm = _rows()
    un = W.unmeasured_nights(w, p, adm, [days[0], days[4], days[5], '2025-02-01'], P)
    check('(c) unmeasured: a night without qa3 (both Stokes), without a V stage-1 rms (V), an admitted night without its '
          'admissions.csv stage-2 V rms (V); a measured night none',
          sorted(un) == [days[4], days[5], '2025-02-01'] and sorted(un[days[5]]) == ['I', 'V'] and list(un[days[4]]) == ['V']
          and list(un['2025-02-01']) == ['V'] and 'weights.csv' in un[days[5]]['I'][0]
          and 'residuals_provisional.csv' in un[days[4]]['V'][0] and 'admissions.csv' in un['2025-02-01']['V'][0], un)
    thin = W.unmeasured_nights(w[:4], p[:4], {}, [days[0], days[3]], P)
    check('(c) a gate below its minimum judges nobody: no night is unmeasured by it', thin == {}, thin)
    rec = dict(present=set(days) | {'2025-02-01'}, both={}, i_only={}, v_only={}, tier={'I': {}, 'V': {}})
    recs = {'59MHz': dict(rec, unmeasured=un), '18MHz': dict(rec, unmeasured=un), '64MHz': dict(rec, unmeasured=un),
            '55MHz': dict(rec)}
    img = {n: {'I': ({'59MHz', '18MHz', '55MHz'}, []), 'V': ({'59MHz', '18MHz', '55MHz'}, [])} for n in days + ['2025-02-01']}
    ref = W.refusals('11h', 'Green', days + ['2025-02-01'], recs, img)
    who = {k: sorted({m for m, _ in v}) for k, v in ref.items()}
    check('(c) refusals(): the unmeasured night is refused in that Stokes by its member, with the named reason; 18 MHz and a '
          'member the image does not hold (64 MHz) are not judged; the batch record (55 MHz, no unmeasured) adds nothing',
          who == {(days[4], 'V'): ['59MHz'], (days[5], 'I'): ['59MHz'], (days[5], 'V'): ['59MHz'], ('2025-02-01', 'V'): ['59MHz']}
          and all(r.startswith('member night unmeasured') for v in ref.values() for _, r in v), ref)


def _real_cells():
    root = CFG.paths.COADD_SB_ROOT
    return root if os.path.isdir(os.path.join(root, '11h_59MHz')) else None


def test_wideband_real():
    root = _real_cells()
    if root is None:
        print('SKIP  real data: no production co-add root')
        return
    from exopipe.common.phase2_config import SUBBAND_BANDS
    from exopipe.phase3.lib.masters import night_image_gate as NG, epoch_gate as EG, coadd as C
    from exopipe.phase3.lib.search import admitted as ADS
    excl = C.load_exclusions()
    rd = dict(night_failed=NG.failed_dates, epoch_failed=EG.failed_dates, same_state_v=lambda c: set(),
              excluded=lambda l, m, n: C.is_excluded(l, m, n, excl), beam_area=W.beam_area)
    tot_un, tot_n = 0, 0
    for lst, scope in (('11h', 'Green'), ('16h', 'Red'), ('10h', 'Blue')):
        recs_b, recs_p = {}, {}
        for m in W.members(scope, SUBBAND_BANDS):
            cd = os.path.join(root, f'{lst}_{m}')
            adm = ADS.nights(cd, m, 'deep')
            recs_b[m] = W.member_record(cd, m, lst, P, rd, admitted=adm)
            recs_p[m] = W.member_record(cd, m, lst, P, rd, admitted=adm, perhour=recs_b[m]['present'])
        same = all({k: v for k, v in recs_p[m].items() if k != 'unmeasured'} == recs_b[m] and 'unmeasured' not in recs_b[m]
                   for m in recs_b)
        nights = sorted(set().union(*(r['present'] for r in recs_b.values())))
        img = {n: {st: (set(recs_b), []) for st in ('I', 'V')} for n in nights}   # every member held: the records alone decide
        ref_b = W.refusals(lst, scope, nights, recs_b, img)
        ref_p = W.refusals(lst, scope, nights, recs_p, img)
        kept = {k: [x for x in v if not x[1].startswith('member night unmeasured')] for k, v in ref_p.items()}
        kept = {k: v for k, v in kept.items() if v}
        added = {(n, st, m) for (n, st), v in ref_p.items() for m, r in v if r.startswith('member night unmeasured')}
        want = {(n, st, m) for m, r in recs_p.items() if m != W.EXCEPTION for n, d in r['unmeasured'].items() for st in d}
        tot_un += len({(n, st) for n, st, _ in added})
        tot_n += 2 * len(nights)
        check(f'real {lst} {scope} (read only): the per-hour member records equal the batch ones but for "unmeasured"; every '
              f'batch refusal stands with the same reasons; the per-hour verdict adds exactly the unmeasured member nights '
              f'({len(added)} of them, {len(nights)} nights)', same and kept == ref_b and added == want,
              (same, sorted(set(kept) ^ set(ref_b))[:4], sorted(added ^ want)[:4]))
    print(f'      real: {tot_un} of {tot_n} night-Stokes of the three cells refused per hour for an unmeasured member night')
    # under fallbacks.SEARCH_SANITY_FALLBACK 'fallback' the member's own search searches an unmeasured epoch, counted: the
    # mirror admits it and counts the same site (review of pull request 27, finding 3)
    from exopipe.phase3.core import fallback as FB
    lst, scope = '10h', 'Blue'                     # 20 unmeasured member night-Stokes above
    pol, counted = FB.policy, []
    site = FB.site
    FB.policy = lambda k: 'fallback'
    FB.site = lambda k, detail, n=1: counted.append((k, n))
    try:
        recs_f = {}
        for m in W.members(scope, SUBBAND_BANDS):
            cd = os.path.join(root, f'{lst}_{m}')
            adm = ADS.nights(cd, m, 'deep')
            b = W.member_record(cd, m, lst, P, rd, admitted=adm)
            recs_f[m] = (b, W.member_record(cd, m, lst, P, rd, admitted=adm, perhour=b['present']))
    finally:
        FB.policy, FB.site = pol, site
    check(f"real {lst} {scope} under SEARCH_SANITY_FALLBACK 'fallback': no member night is refused for being unmeasured; the "
          f"site is counted once per member with unmeasured nights ({len(counted)} members)",
          counted and all(not p_['unmeasured'] for _b, p_ in recs_f.values())
          and all(k == W.UNMEASURED_KEY for k, _n in counted), counted)
    # census over every subband cell: present member nights an append would refuse for a value the member's tables miss
    from exopipe.phase3.core import cells as CC
    subs = sorted({m for g in ('Red', 'Green', 'Blue') for m in SUBBAND_BANDS[g]})
    cnt, cells_with, n_present = {}, [], 0
    for lst in sorted({c.split('_', 1)[0] for c in CC.all_cells()}):
        for m in subs:
            cd = os.path.join(root, f'{lst}_{m}')
            wp, pp = os.path.join(cd, 'coadd_final', 'weights.csv'), os.path.join(cd, 'residuals_provisional.csv')
            if not (os.path.exists(wp) and os.path.exists(pp)):
                continue
            adm = ADS.nights(cd, m, 'deep')[0]
            prod = {os.path.basename(x)[:10] for x in glob.glob(os.path.join(cd, 'subtracted', f'*_I_sub_{m}.fits'))}
            present = sorted(prod | set(adm))
            n_present += len(present)
            un = W.unmeasured_nights(list(csv.DictReader(open(wp))), list(csv.DictReader(open(pp))), adm, present, P)
            for n, d in un.items():
                for st, rs in d.items():
                    for r in rs:
                        k = re.sub(r'no (I|V) stage-1', 'no <S> stage-1', r)
                        cnt[(k, st)] = cnt.get((k, st), 0) + 1
            if un:
                cells_with.append(f'{lst}_{m}')
    print(f'      census: {n_present} present member nights in the subband cells; unmeasured (reason, Stokes): '
          f'{sorted(cnt.items())}; cells: {cells_with[:12]}{" ..." if len(cells_with) > 12 else ""} ({len(cells_with)})')
    check('real census ran over the subband cells', n_present > 0, n_present)


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_skip_key()
        test_v_sanity_record()
        test_wideband_synthetic()
        test_wideband_real()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
