"""Wideband member admission (R-GREGG-20261023-WIDEBAND-MEMBER-SEARCH-ADMISSION; lib/search/wideband_admission.py and its call in
lib/search/deep_search.run_deep_cell).

Synthetic: the group membership; the per-Stokes search tier (2.0 x the cell median since 2026-10-30, production exopipe 702dd68 -
median + 5 x 1.4826 MAD before; inf tested and refused, fewer than min_n nights = no refusal, admitted nights tested outside the
population); the sanity gates (10x both Stokes, 3x I only, the engine's
upper-middle median); member_record on a temporary cell (presence from subtracted/, a stale residuals table refuses, gates through
injected readers, D-ADMIT, B-03 same-state V, products without a beam, the beam gate over the nights the earlier gates leave);
refusals() judging only the members the image holds (the 18 MHz allowance) and the image's own reasons.
Engine: the switch, the config key, its environment mapping and parity value, the refusal kind, the V side outside the sanity try.
Real data (read only, skipped without the production co-add root): 11h Green 2024-12-23 refused in I and V by 59 MHz; 11h Green, 16h
Red and 13h Red refuse every census night that stands under today's search tier (the census of 2026-10-23 judged the tier by the
MAD rule: a night refused only by it stands when the member's rms is above 2 x its cell median); the review's image cases (04h Red 2024-12-20 incomplete - no 36 MHz; 06h Red 2024-12-24
not refused for its excluded 18 MHz, which the image does not hold; 11h Green 2025-01-23 incomplete - no 46 MHz).
Run: python tests/test_search_wbmember.py
"""
import csv
import glob
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.lib.search import wideband_admission as W          # noqa: E402

T = '/tmp/exopipe3_tests_search_wbmember_%d' % os.getpid()
PROD_COADD = '/data07/gh/phase3/coadd_sb'
CENSUS = ('/lustre/gh/main/phase3/refactor_analysis/bright_source_contamination_20261020/wideband_member_qa_20261023/'
          'member_admission.csv')
P = dict(ratio=2.0, tier_min=5, sanity_x_both=10.0, sanity_x_i=3.0, sanity_min=5, beam_ratio=1.5, beam_min=5)
FAILS, N = [], [0]


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else '  -- %s' % (detail,)))
    if not ok:
        FAILS.append(name)


def write(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(rows)


def fits_stub(path, beam=(0.1, 0.1)):
    import numpy as np
    from astropy.io import fits
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h = fits.Header(); h['BMAJ'], h['BMIN'] = beam
    fits.writeto(path, np.zeros((2, 2), np.float32), h, overwrite=True)


def readers(ng=(), eg=(), excl=(), ssv=(), areas=None):
    return dict(night_failed=lambda fd: set(ng), epoch_failed=lambda fd, b: set(eg), excluded=lambda l, m, n: (m, n) in set(excl),
                same_state_v=lambda c: set(ssv), beam_area=lambda c, m, l, d: (areas or {}).get(d, 0.01))


def main():
    from exopipe.common.phase2_config import SUBBAND_BANDS
    try:
        # ---- membership
        check('members: Green = its five subbands', W.members('Green', SUBBAND_BANDS) == ('41MHz', '46MHz', '50MHz', '55MHz', '59MHz'))
        check('members: Full = the fifteen subbands', len(W.members('Full', SUBBAND_BANDS)) == 15)
        check('members: a subband scope has none', W.members('55MHz', SUBBAND_BANDS) == ())
        # ---- search tier
        rows = [dict(date='2025-01-%02d' % i, i_rms_mjy=str(100 + i), v_rms_mjy=str(50 + (i % 3))) for i in range(1, 11)]
        rows.append(dict(date='2025-01-20', i_rms_mjy='900', v_rms_mjy='51'))
        rows.append(dict(date='2025-01-21', i_rms_mjy='inf', v_rms_mjy='nan'))
        tf = W.tier_failures(rows, 2.0, 5)
        check('tier: the outlier and inf are refused in I only; nan never', set(tf['I']) == {'2025-01-20', '2025-01-21'} and tf['V'] == {}, tf)
        import numpy as np
        v = np.array([float(r['i_rms_mjy']) for r in rows[:11]])
        thr = float(2.0 * np.median(v))
        check('tier: threshold = 2.0 x np.median (the engine\'s SEARCH_TIER_RATIO x _med, production exopipe 702dd68)',
              tf['I']['2025-01-20'][1] == thr)
        ev = [dict(date='e%d' % i, i_rms_mjy=str(x), v_rms_mjy='1') for i, x in enumerate((10, 20, 30, 40, 50, 1000))]
        check('tier: an even count\'s median is the mean of the middle two, as np.median (threshold 70, not 80)',
              W.tier_failures(ev, 2.0, 5)['I'] == {'e5': (1000.0, 70.0)}, W.tier_failures(ev, 2.0, 5))
        check('tier: fewer than min_n finite nights refuses nothing', W.tier_failures(rows[:3] + rows[10:12], 2.0, 5) == {'I': {}, 'V': {}})
        tfa = W.tier_failures(rows[:11], 2.0, 5, extra={'I': {'adm': 950.0}})
        check('tier: an admitted night is tested against the record\'s threshold, not in it', 'adm' in tfa['I'] and tfa['I']['adm'][1] == thr)
        # ---- sanity
        wr = [dict(date='d%02d' % i, qa3_rms_mjy=str(100 + i)) for i in range(9)] + [dict(date='big', qa3_rms_mjy='2000'),
                                                                                      dict(date='mid', qa3_rms_mjy='400')]
        b, i_ = W.sanity_failures(wr, 10.0, 3.0, 5)
        fin = sorted(float(r['qa3_rms_mjy']) for r in wr)
        check('sanity: 10x refuses both, 3x I only, median = the upper middle (_fin[n // 2])', set(b) == {'big'} and set(i_) == {'mid'}
              and b['big'][1] == fin[len(fin) // 2], (b, i_))
        check('sanity: fewer than min_n nights refuses nothing', W.sanity_failures(wr[:4], 10.0, 3.0, 5) == ({}, {}))
        # ---- member_record on a temporary cell
        cell = os.path.join(T, '11h_59MHz')
        nights = ['2025-01-%02d' % i for i in range(1, 11)]
        write(os.path.join(cell, 'coadd_final', 'weights.csv'), ['date', 'weight', 'qa3_rms_mjy'],
              [[d, '1', '100'] for d in nights] + [['2025-01-20', '1', '400']])
        write(os.path.join(cell, 'residuals_provisional.csv'), ['date', 'i_rms_mjy', 'v_rms_mjy'],
              [[d, '100', '50'] for d in nights] + [['2025-01-20', '900', '50']])
        for d in nights + ['2025-01-20']:
            fits_stub(os.path.join(cell, 'subtracted', f'{d}_I_sub_59MHz.fits'))
            if d != '2025-01-05':
                fits_stub(os.path.join(cell, 'subtracted', f'{d}_V_sub_59MHz.fits'), beam=(0.1, 0.1) if d != '2025-01-06' else (0.0, 0.1))
        areas = {d: 0.01 for d in nights + ['2025-01-20']}
        areas['2025-01-07'] = 0.03
        rec = W.member_record(cell, '59MHz', '11h', P, readers(ng=['2025-01-02'], eg=['2025-01-03'], excl=[('59MHz', '2025-01-04')],
                                                              ssv=['2025-01-05'], areas=areas))
        check('member: presence from subtracted/', rec['present'] == set(nights + ['2025-01-20']))
        check('member: night-image, epoch gate and data_decisions refuse both Stokes',
              rec['both'].get('2025-01-02') == ['night-image gate'] and rec['both'].get('2025-01-03') == ['epoch gate']
              and rec['both'].get('2025-01-04') == ['data_decisions exclusion'], rec['both'])
        check('member: B-03 same-state - no V product refuses V only', '2025-01-05' in rec['v_only'] and '2025-01-05' not in rec['both'])
        check('member: a product without a positive beam refuses both (input-provenance)',
              any('V product without a positive beam' in r for r in rec['both'].get('2025-01-06', [])), rec['both'].get('2025-01-06'))
        check('member: the I sanity gate (3x qa3) refuses I only', '2025-01-20' in rec['i_only'] and set(rec['tier']['I']) == {'2025-01-20'})
        check('member: the beam gate refuses an area > 1.5 x the median of the nights the gates leave',
              any(r.startswith('beam gate') for r in rec['both'].get('2025-01-07', [])), rec['both'].get('2025-01-07'))
        try:
            W.member_record(cell, '59MHz', '11h', P, dict(readers(), beam_area=lambda c, m, l, d: None))
            check('member: a night without a readable beam REFUSES the member', False)
        except RuntimeError as e:
            check('member: a night without a readable beam REFUSES the member', 'REFUSING' in str(e) and '59MHz' in str(e), e)
        write(os.path.join(cell, 'residuals_provisional.csv'), ['date', 'i_rms_mjy', 'v_rms_mjy'], [[d, '100', '50'] for d in nights])
        try:
            W.member_record(cell, '59MHz', '11h', P, readers())
            check('member: a residuals table of another generation REFUSES', False)
        except RuntimeError as e:
            check('member: a residuals table of another generation REFUSES', 'stale' in str(e) and 'REFUSING' in str(e), e)
        os.remove(os.path.join(cell, 'residuals_provisional.csv'))
        try:
            W.member_record(cell, '59MHz', '11h', P, readers())
            check('member: missing records REFUSE naming the member', False)
        except RuntimeError as e:
            check('member: missing records REFUSE naming the member', 'REFUSING' in str(e) and '59MHz' in str(e), e)

        def _nf(fd):
            raise FileNotFoundError('night_image_gate.csv')
        write(os.path.join(cell, 'residuals_provisional.csv'), ['date', 'i_rms_mjy', 'v_rms_mjy'],
              [[d, '100', '50'] for d in nights] + [['2025-01-20', '900', '50']])
        try:
            W.member_record(cell, '59MHz', '11h', P, dict(readers(), night_failed=_nf))
            check('member: a missing gate record REFUSES naming the member', False)
        except RuntimeError as e:
            check('member: a missing gate record REFUSES naming the member', '59MHz' in str(e) and 'REFUSING' in str(e), e)
        adm = ({'2025-02-01': dict(night_image='PASS', epoch_gate='FAIL', stage2_i_rms_mjy='101', stage2_v_rms_mjy='50', qa3_rms_mjy='100')},
               {'2025-02-03': 'admitted night: no I product'})
        rec2 = W.member_record(cell, '59MHz', '11h', P, readers(), admitted=adm)
        check('D-ADMIT: an accepted admitted night is present, its verdicts apply; a refused one refuses',
              '2025-02-01' in rec2['present'] and rec2['both'].get('2025-02-01') == ['epoch gate']
              and rec2['both'].get('2025-02-03') == ['admitted night refused by the search'], rec2['both'])
        # ---- refusals: the image's set and its own reasons
        good = dict(present={'n1', 'n2'}, both={}, i_only={}, v_only={}, tier={'I': {}, 'V': {}})
        bad18 = dict(present={'n1'}, both={'n1': ['epoch gate']}, i_only={}, v_only={}, tier={'I': {}, 'V': {}})
        recs = {'18MHz': bad18, '23MHz': good, '27MHz': good, '32MHz': good, '36MHz': good}
        imgs = {'n1': {'I': ({'23MHz', '27MHz', '32MHz', '36MHz'}, []), 'V': ({'18MHz', '23MHz', '27MHz', '32MHz', '36MHz'}, [])},
                'n2': {'I': ({'23MHz', '27MHz', '32MHz'}, ['stacked from an incomplete subband set (missing 36MHz)']),
                       'V': (None, ['x.fits cannot show its subbands'])}}
        ref = W.refusals('06h', 'Red', ['n1', 'n2'], recs, imgs)
        check('refusals: a failing member the image does not hold (18 MHz allowance) is not judged', ('n1', 'I') not in ref, ref)
        check('refusals: a failing 18 MHz member the image DOES hold never refuses (the one exception, Gregg 2026-10-23)',
              ('n1', 'V') not in ref, ref)
        bad23 = dict(present={'n1'}, both={'n1': ['epoch gate']}, i_only={}, v_only={}, tier={'I': {}, 'V': {}})
        ref2 = W.refusals('12h', 'Red', ['n1'], dict(recs, **{'23MHz': bad23}), imgs)
        check('refusals: a failing 23 MHz member refuses both Stokes', ('n1', 'I') in ref2 and ('n1', 'V') in ref2, ref2)
        check('refusals: the image\'s own reasons refuse that Stokes', ref.get(('n2', 'I'), [])[:1] == [('Red', 'stacked from an incomplete '
              'subband set (missing 36MHz)')] and ('n2', 'V') in ref, ref)
        check('refusals: no reason holds a comma', all(',' not in r for v in ref.values() for _, r in v))
        # ---- the engine
        pr = subprocess.run([sys.executable, '-c', 'import sys; sys.path.insert(0, %r); from exopipe.phase3.lib.search import '
                             'deep_search as D; print(D.WIDEBAND_MEMBER_ADMISSION)' % REPO],
                            env=dict(os.environ, P3_WIDEBAND_MEMBER_ADMISSION='0'), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True, timeout=900)
        check('engine: P3_WIDEBAND_MEMBER_ADMISSION=0 switches the rule off', pr.stdout.strip().splitlines()[-1:] == ['False'], pr.stdout[-400:])
        from exopipe.phase3.lib.search import deep_search as D
        check('engine: on by default', D.WIDEBAND_MEMBER_ADMISSION is True)
        src = open(D.__file__).read()
        i_try = src.index("        fallback.site('fallbacks.SEARCH_SANITY_FALLBACK', f'{lst} {scope}: the V sanity gate could not run ({_e})')")
        check('engine: the V side is applied AFTER the V sanity try (never dropped by its fallback)',
              src.index('    skip_V_dates |= {d for d in _skipV_wb if d in resid_V}') > i_try and 'skip_I_dates.update(_wbI)' in src)
        check('engine: deep cadence and wideband scopes only', "if cadence == 'deep' and WIDEBAND_MEMBER_ADMISSION and scope in _WBA.GROUPS:" in src)
        check('engine: COLUMNS.csv names the new refusal file', any(x == '_wbmember_excluded.csv' for x, _ in D.REFUSAL_COLUMNS))
        from exopipe.phase3.lib.search import tests as ST
        check('content tests: the wbmember kind is known (7 columns, the Stokes in column 4)', ST.REFUSAL_KINDS.get('wbmember') == (7, 'row'))
        from exopipe.phase3.config import CFG
        sd = CFG.search_detect
        check('config: search_detect.WIDEBAND_MEMBER_ADMISSION = True, ruled', sd.WIDEBAND_MEMBER_ADMISSION is True
              and sd.RULED.get('WIDEBAND_MEMBER_ADMISSION') == 'R-GREGG-20261023-WIDEBAND-MEMBER-SEARCH-ADMISSION')
        ds_src = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_step.py')).read()
        check('config -> engine: deep_step.engine_env passes P3_WIDEBAND_MEMBER_ADMISSION',
              "'P3_WIDEBAND_MEMBER_ADMISSION': '1' if sd.WIDEBAND_MEMBER_ADMISSION else '0'" in ds_src)
        par = open(os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg')).read()
        check('parity: production_20261013.cfg sets the rule off (production has none)', 'search_detect.WIDEBAND_MEMBER_ADMISSION = False' in par)
        led = open('/lustre/gh/main/clv15/fix_ledger.csv', newline='').read()
        check('the ruling ID is in the fix ledger', 'R-GREGG-20261023-WIDEBAND-MEMBER-SEARCH-ADMISSION' in led)
        # ---- real data, read only
        if os.path.isdir(os.path.join(PROD_COADD, '11h_59MHz')):
            from exopipe.phase3.lib.masters import night_image_gate as NG, epoch_gate as EG, coadd as C
            excl = C.load_exclusions()
            rd = dict(night_failed=NG.failed_dates, epoch_failed=EG.failed_dates, same_state_v=lambda c: set(),
                      excluded=lambda l, m, n: C.is_excluded(l, m, n, excl), beam_area=W.beam_area)

            def cell_refusals(lst, scope, nights):
                recs = {m: W.member_record(os.path.join(PROD_COADD, '%s_%s' % (lst, m)), m, lst, P, rd)
                        for m in W.members(scope, SUBBAND_BANDS)}
                sci = {r['date']: r.get('sci_dir', '') for r in csv.DictReader(open(os.path.join(PROD_COADD, '%s_%s' % (lst, scope),
                                                                                             'coadd_final', 'weights.csv')))}
                # a night a later rebuild dropped from the cell (04h Red 2024-12-20 since the 2026-11-08 rebuild): its Phase 2 Science dir
                p2 = lambda n: (sorted(glob.glob('/lustre/pipeline/exopipe/phase2/%s/%s/Science_*' % (lst, n))) or [''])[-1]
                imgs = {n: W.image_findings(lst, scope, sci.get(n) or p2(n), n, scope in ('Red', 'Full') and (lst, '18', n) in excl)
                        for n in nights}
                return W.refusals(lst, scope, nights, recs, imgs)

            ref = cell_refusals('11h', 'Green', ['2024-12-23', '2025-01-23'])
            r59 = [r for m, r in ref.get(('2024-12-23', 'I'), []) if m == '59MHz'] + [r for m, r in ref.get(('2024-12-23', 'V'), []) if m == '59MHz']
            check('real 11h Green 2024-12-23: refused in I and V by 59MHz', ('2024-12-23', 'I') in ref and ('2024-12-23', 'V') in ref
                  and any(r.startswith('search tier (I') for r in r59) and any(r.startswith('search tier (V') for r in r59), r59)
            check('real 11h Green 2025-01-23: the image lacks 46 MHz - refused', any('missing 46MHz' in r for _, r in ref.get(('2025-01-23', 'I'), [])),
                  ref.get(('2025-01-23', 'I')))
            ref = cell_refusals('04h', 'Red', ['2024-12-20'])
            check('real 04h Red 2024-12-20: the image lacks 36 MHz - refused', any('missing 36MHz' in r for _, r in ref.get(('2024-12-20', 'I'), [])),
                  ref.get(('2024-12-20', 'I')))
            ref = cell_refusals('06h', 'Red', ['2024-12-24'])
            check('real 06h Red 2024-12-24: the excluded 18 MHz is not in the image - not refused for it',
                  not any(m == '18MHz' for k, v in ref.items() for m, _ in v if k[0] == '2024-12-24'), ref)
            if os.path.exists(CENSUS):
                import re
                import numpy as np

                def tier_fails(lst, m, st, night):
                    # today's search tier on the member's own table, independently of the module: the night's stage-1 rms of that
                    # Stokes above 2.0 x numpy's median of the cell's positive finite values (at least 5; production 702dd68)
                    rows_m = list(csv.DictReader(open(os.path.join(PROD_COADD, '%s_%s' % (lst, m), 'residuals_provisional.csv'))))
                    col = 'i_rms_mjy' if st == 'I' else 'v_rms_mjy'
                    vals = {r['date']: float(r[col]) for r in rows_m if r.get(col) not in (None, '', 'nan')}
                    pop = [x for x in vals.values() if np.isfinite(x) and x > 0]
                    return len(pop) >= 5 and vals.get(night, 0) > 2.0 * float(np.median(pop))

                def stands(r):
                    # a census refusal stands when a failing member other than 18 MHz fails a gate other than the search tier, or
                    # still fails the search tier under today's rule (the census of 2026-10-23 judged it by the MAD rule)
                    for seg in r['reasons'].split(' | '):
                        m, _, rs = seg.partition(': ')
                        kinds = {x.strip() for x in re.sub(r'\([^)]*\)', '', rs).split(',')}
                        if m != '18MHz' and (kinds - {'search tier'} or tier_fails(r['lst'], m, r['stokes'], r['date'])):
                            return True
                    return False
                for lst, scope in (('11h', 'Green'), ('16h', 'Red'), ('13h', 'Red')):
                    rows_ = [r for r in csv.DictReader(open(CENSUS)) if r['lst'] == lst and r['group'] == scope]
                    # the census (2026-10-23 morning) judged 18 MHz like the other members; the 18 MHz exception keeps a night whose
                    # ONLY failing member is 18 MHz (Gregg 2026-10-23 "No! The one exception to the exclusion rule is 18 MHz")
                    cen = {(r['date'], r['stokes']) for r in rows_ if r['failing_members'] != '18MHz' and stands(r)}
                    freed = {(r['date'], r['stokes']): set(r['failing_members'].split(';')) for r in rows_
                             if r['failing_members'] != '18MHz' and not stands(r)}
                    only18 = {(r['date'], r['stokes']) for r in rows_ if r['failing_members'] == '18MHz'}
                    ref = cell_refusals(lst, scope, sorted({r['date'] for r in rows_}))
                    print('      real %s %s: census refusals %d, standing under the 2 x median search tier %d, freed by it %s'
                          % (lst, scope, len(cen) + len(freed), len(cen), sorted(freed)))
                    check('real %s %s: every census refusal that stands under the 2 x median search tier is refused (18 MHz-only '
                          'failures excepted)' % (lst, scope), cen <= set(ref), sorted(cen - set(ref)))
                    check('real %s %s: a census night the 2 x median search tier frees is not refused for the tier by its census '
                          'members' % (lst, scope),
                          not any(m_ in ms and r_.startswith('search tier') for k, ms in freed.items() for m_, r_ in ref.get(k, [])),
                          {k: ref.get(k) for k in freed if k in ref})
                    check('real %s %s: no night whose only failing member is 18 MHz is refused' % (lst, scope), not (only18 & set(ref)), sorted(only18 & set(ref)))
        else:
            print('SKIP real-data checks (no %s)' % PROD_COADD)
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print('%d of %d checks passed' % (N[0] - len(FAILS), N[0]))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
