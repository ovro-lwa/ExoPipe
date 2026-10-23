"""Wideband member admission (R-GREGG-20261023-WIDEBAND-MEMBER-SEARCH-ADMISSION; lib/search/wideband_admission.py and its call in
lib/search/deep_search.run_deep_cell).

Synthetic: the group membership; the per-Stokes search tier (median + sigma x 1.4826 MAD, inf tested and refused, fewer than min_n
nights = no refusal, admitted nights tested outside the population); the sanity gates (10x both Stokes, 3x I only, the engine's
upper-middle median); member_record on a temporary cell (presence from subtracted/, a stale residuals table refuses, gates through
injected readers, D-ADMIT, B-03 same-state V, products without a beam, the beam gate over the nights the earlier gates leave);
refusals() judging only the members the image holds (the 18 MHz allowance) and the image's own reasons.
Engine: the switch, the config key, its environment mapping and parity value, the refusal kind, the V side outside the sanity try.
Real data (read only, skipped without the production co-add root): 11h Green 2024-12-23 refused in I and V by 59 MHz; 11h Green, 16h
Red and 13h Red refuse every census night; the review's image cases (04h Red 2024-12-20 incomplete - no 36 MHz; 06h Red 2024-12-24
not refused for its excluded 18 MHz, which the image does not hold; 11h Green 2025-01-23 incomplete - no 46 MHz).
Run: python tests/test_search_wbmember.py
"""
import csv
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
P = dict(sigma=5.0, tier_min=5, sanity_x_both=10.0, sanity_x_i=3.0, sanity_min=5, beam_ratio=1.5, beam_min=5)
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
        tf = W.tier_failures(rows, 5.0, 5)
        check('tier: the outlier and inf are refused in I only; nan never', set(tf['I']) == {'2025-01-20', '2025-01-21'} and tf['V'] == {}, tf)
        import numpy as np
        v = np.array([float(r['i_rms_mjy']) for r in rows[:11]])
        thr = float(np.median(v) + 5.0 * 1.4826 * np.median(np.abs(v - np.median(v))))
        check('tier: threshold = np.median + 5 x 1.4826 MAD (the engine\'s)', abs(tf['I']['2025-01-20'][1] - thr) < 1e-9)
        check('tier: fewer than min_n finite nights refuses nothing', W.tier_failures(rows[:3] + rows[10:12], 5.0, 5) == {'I': {}, 'V': {}})
        tfa = W.tier_failures(rows[:11], 5.0, 5, extra={'I': {'adm': 950.0}})
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
        check('refusals: the same member in the V image refuses V', ('n1', 'V') in ref and ref[('n1', 'V')] == [('18MHz', 'epoch gate')], ref)
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
                imgs = {n: W.image_findings(lst, scope, sci.get(n, ''), n, scope in ('Red', 'Full') and (lst, '18', n) in excl)
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
                for lst, scope in (('11h', 'Green'), ('16h', 'Red'), ('13h', 'Red')):
                    cen = {(r['date'], r['stokes']) for r in csv.DictReader(open(CENSUS)) if r['lst'] == lst and r['group'] == scope}
                    ref = cell_refusals(lst, scope, sorted({d for d, _ in cen}))
                    check('real %s %s: every census refusal is refused' % (lst, scope), cen <= set(ref), sorted(cen - set(ref)))
        else:
            print('SKIP real-data checks (no %s)' % PROD_COADD)
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print('%d of %d checks passed' % (N[0] - len(FAILS), N[0]))
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
