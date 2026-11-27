"""The bright-source table's noise in the deep search (rules b02-3d; R-REFACTOR-20261016-B02-CONVENTIONS (3): "The bright-source
table is applied with the inputs it was derived with"): the median over the cell's gate-ledger rows of |peak_jy / global_snr|,
the rows harvested as the table derivation harvests them (ops/p3s3_curve_derive.py load_harvest), per Stokes, over the cell's
admitted nights - not the frame noise (the uniform rms the V noise rule keeps).

Checks, data-free:
  harvest       removal_rules.table_noise_rms takes a row as the derivation does and leaves out each kind of row it leaves out
                (|S/N| below the floor, a mid-hour elevation below its floor, no global S/N, no peak, no position, unparsable
                values); a negative row counts by its absolute ratio; ledger_rms keeps the cell's rows only
  population    table_noise: the cell's admitted nights, their rows' median; an admitted night outside the cell (D-ADMIT) with
                those rows plus its own; an append with its base's rows (base_population: the base's ledger rows of its
                population nights) plus its own
  recompute     the frame_noise content test (lib/search/tests.medians_and_cards) recomputes every recorded TBLNOISE from the
                run directory's gate ledger: a consistent record passes; a ledger row changed, a row added, a recorded value
                changed and the old statistic (the frame noise median) are each refused
  engine        the engine computes the table's noise after every night is extracted and before the table stage, from the
                rows its stages write, and refuses when the ledger it writes differs from those rows
Real derivation (read only; skipped where exopipe_ops is not mounted): ops/p3s3_curve_derive.load_harvest over the same
ledger, with the floors its deep singles tables were derived with, gives the same median.
Run: python tests/test_search_table_noise.py
"""
import csv
import math
import os
import re
import shutil
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
if os.path.isfile('/lustre/gh/main/exopipe_ops/exopipe3_site.py'):       # absent on a data-free machine: no site file is loaded
    os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

import numpy as np                                                      # noqa: E402

from exopipe.phase3.config import CFG                                   # noqa: E402
from exopipe.phase3.lib.search import removal_rules as RR, tests as ST  # noqa: E402

T = f'/tmp/exopipe3_tests_search_table_noise_{os.getpid()}'
OPS = '/lustre/gh/main/exopipe_ops'
N, FAILS = [0], []
LST, SC = '05h', '78MHz'
D = ['2024-12-18', '2024-12-19', '2024-12-20', '2024-12-21', '2024-12-22']


def check(name, ok, detail=None):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name)
    if not ok:
        FAILS.append(name)
        if detail is not None:
            print('      ', str(detail)[:1500])


def row(st, d, x, **k):
    """A gate-ledger row of the cell whose |peak / global S/N| is x (global S/N 8: an exact quotient), 6 sigma, near zenith."""
    r = dict.fromkeys(ST.GLED_COLS, '')
    r.update(lst=LST, scope=SC, stokes=st, date=d, ra='82.5', dec='37.0', snr='6.0', peak_jy=repr(8 * x), global_snr='8.0',
             pix_x='10', pix_y='20', veto_reason='mstruct', final='0')
    r.update(k)
    return r


def test_harvest():
    c = CFG.catalog_conform
    el = lambda dec: RR.derivation_elevation(LST, 82.5, dec)
    check('the harvest floors are the derivation\'s: |S/N| 5 (its --floor), 20 deg at mid-hour (gate_inputs.DEEP_ELEV_FLOOR_DEG '
          'before 2026-10-07), each with the B-02 conventions ruling',
          c.DEEP_TABLE_NOISE_SNR_FLOOR == 5.0 and c.DEEP_TABLE_NOISE_ELEV_FLOOR_DEG == 20.0
          and c.RULED.get('DEEP_TABLE_NOISE_SNR_FLOOR') == 'R-REFACTOR-20261016-B02-CONVENTIONS'
          and c.RULED.get('DEEP_TABLE_NOISE_ELEV_FLOOR_DEG') == 'R-REFACTOR-20261016-B02-CONVENTIONS')
    dz = 37.2398 - 70.0                              # 20 deg elevation due south at mid-hour: dec = lat - 70
    check('derivation_elevation: 90 deg at the mid-hour zenith, the south meridian at the latitude minus the zenith angle, '
          'equal to the conformance\'s mid-hour elevation within 1e-9 deg',
          abs(el(37.2398) - 90.0) < 1e-6 and abs(el(dz) - 20.0) < 1e-9
          and all(abs(RR.derivation_elevation(LST, ra, de) - RR.mid_hour_elevation(LST, ra, de)) < 1e-9
                  for ra, de in ((10.0, 50.0), (82.5, -10.0), (200.0, 80.0), (300.0, 0.0))),
          (el(37.2398), el(dz)))
    take = {'plain': row('I', D[0], 0.012), 'negative': row('I', D[0], -0.012, snr='-6.0', global_snr='-8.0'),
            'at the S/N floor': row('I', D[0], 0.012, snr='5.0'), 'at 20.0001 deg': row('I', D[0], 0.012, dec=repr(dz + 1e-4)),
            'a table cut': row('I', D[0], 0.012, veto_reason='table'), 'a final': row('I', D[0], 0.012, veto_reason='', final='1')}
    drop = {'S/N 4.99': row('I', D[0], 0.012, snr='4.99'), 'S/N -4.99': row('I', D[0], 0.012, snr='-4.99'),
            'below 20 deg': row('I', D[0], 0.012, dec=repr(dz - 1e-4)), 'no global S/N': row('I', D[0], 0.012, global_snr=''),
            'global S/N 0': row('I', D[0], 0.012, global_snr='0'), 'peak 0': row('I', D[0], 0.012, peak_jy='0.0'),
            'no S/N': row('I', D[0], 0.012, snr=''), 'no ra': row('I', D[0], 0.012, ra=''), 'dec nan': row('I', D[0], 0.012, dec='nan'),
            'peak text': row('I', D[0], 0.012, peak_jy='x')}
            # a 25 deg floor would drop 'at 20.0001 deg': the singles tables' derivations ran at 20
    got_t = {k: RR.table_noise_rms(r, LST) for k, r in take.items()}
    got_d = {k: RR.table_noise_rms(r, LST) for k, r in drop.items()}
    check('table_noise_rms: rows the derivation harvests give |peak / global S/N| (a negative row by its absolute ratio; a '
          'vetoed row, a table cut and a final alike); each row it leaves out gives None',
          all(v == 0.012 for v in got_t.values()) and all(v is None for v in got_d.values()), (got_t, got_d))
    by = RR.ledger_rms([row('I', D[0], 0.010), row('V', D[1], 0.005), row('I', D[0], 0.012), row('I', D[0], 0.5, lst='06h'),
                        row('I', D[0], 0.5, scope='82MHz'), row('I', D[2], 0.5, snr='1.0')], LST, SC)
    check('ledger_rms: per Stokes and night, in ledger order; rows of another cell and rows not harvested left out',
          by == {'I': {D[0]: [0.010, 0.012]}, 'V': {D[1]: [0.005]}}, by)


def write_attempt(att, lrows, nz, adm, cell, base=None, framed=None, tamper=None):
    """A run directory with the ledger rows, the frame noise record the engine would write from them, and the frames."""
    from astropy.io import fits
    rd = os.path.join(att, 'transients', 'Run_T')
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, 'v_deep_ledger.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=ST.GLED_COLS)
        w.writeheader()
        w.writerows(lrows)
    t = RR.table_noise(RR.ledger_rms(lrows, LST, SC), {st: set(v) for st, v in nz.items()}, adm, cell, base=base)
    rows = RR.frame_noise_rows(LST, SC, sorted({d for v in nz.values() for d in v}), nz, adm, t, None, 'A1', framed=framed)
    if tamper:
        rows = tamper(rows)
    with open(os.path.join(rd, RR.FRAME_NOISE), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(RR.FRAME_NOISE_COLS)
        w.writerows(rows)
    for r in rows:
        fp = ST.frame_path(att, 'T', LST, SC, r[3], r[2])
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        h = fits.Header()
        if r[4]:
            h['IMGRMS'] = float(r[4])
        fits.writeto(fp, np.zeros((4, 4), np.float32), h, overwrite=True)
    with open(os.path.join(rd, 'v_deep_images.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(ST.GIMG_COLS)
        w.writerows([[LST, SC, r[2], r[3], 0, 0, '', '', 0, 0, 25.0] for r in rows])
    return rd, t, rows


def test_population_and_recompute():
    nz = {'I': {D[0]: 0.009, D[1]: 0.011, D[2]: 0.013, D[3]: 0.020, D[4]: 0.015}, 'V': {D[0]: 0.004, D[1]: 0.006}}
    adm = {'I': {D[0], D[1], D[2], D[3]}, 'V': {D[0], D[1]}}
    cell = {D[0], D[1], D[2], D[4]}                  # D[3]: admitted from outside the cell (D-ADMIT); D[4] not admitted
    lrows = [row('I', D[0], 0.010), row('I', D[0], 0.016), row('I', D[1], 0.012), row('I', D[2], 0.014), row('I', D[3], 0.030),
             row('I', D[4], 0.500), row('I', D[1], 0.700, snr='3.0'), row('V', D[0], 0.005), row('V', D[1], 0.007),
             row('V', D[1], 0.009)]
    rd, t, rows = write_attempt(os.path.join(T, 'a1'), lrows, nz, adm, cell)
    pop = [0.010, 0.016, 0.012, 0.014]
    check('table_noise: the population is the cell\'s admitted nights, its median over their harvested rows (not the frame '
          'noise); the D-ADMIT night with those rows plus its own; a night not admitted carries the population\'s median; '
          'each Stokes its own',
          t['I']['population'] == [D[0], D[1], D[2]] and t['I']['n'] == 4 and t['I']['median'] == float(np.median(pop))
          and t['I']['night'][D[3]] == float(np.median(pop + [0.030])) and t['I']['night'][D[4]] == float(np.median(pop))
          and t['V']['median'] == float(np.median([0.005, 0.007, 0.009])) and t['I']['median'] != float(np.median([0.009, 0.011, 0.013])),
          t)
    fnp = os.path.join(rd, RR.FRAME_NOISE)
    ok0 = ST.medians_and_cards(fnp)
    check('the frame_noise content test recomputes every TBLNOISE from the run directory\'s gate ledger: the engine\'s record '
          'passes', ok0[0], ok0)
    led = os.path.join(rd, 'v_deep_ledger.csv')
    keep_l, keep_f = open(led).read(), open(fnp).read()

    def refused(mut_l=None, mut_f=None, words=None):
        try:
            if mut_l:
                open(led, 'w').write(mut_l(keep_l))
            if mut_f:
                open(fnp, 'w').write(mut_f(keep_f))
            r_ = ST.medians_and_cards(fnp)
            return (not r_[0]) and (words is None or words in str(r_[2])), r_[2][:200]
        finally:
            open(led, 'w').write(keep_l)
            open(fnp, 'w').write(keep_f)
    f = {'a ledger row\'s peak changed': refused(lambda x: x.replace(repr(8 * 0.010), repr(8 * 0.020), 1), words='tblnoise'),
         'a harvested row added': refused(lambda x: x + ','.join(row('I', D[1], 0.040)[k] for k in ST.GLED_COLS) + '\r\n',
                                          words='tblnoise'),
         'a recorded TBLNOISE changed': refused(mut_f=lambda x: x.replace(repr(t['V']['median']), repr(t['V']['median'] * 1.01))),
         'the frame noise median recorded (the old statistic)': refused(
             mut_f=lambda x: x.replace(repr(t['I']['median']), repr(0.011)), words='tblnoise')}
    check('the content test refuses a ledger row changed, a harvested row added, a recorded TBLNOISE changed, and the frame '
          'noise median recorded in its place', all(v[0] for v in f.values()), f)
    # an append: its base's rows (the base's population nights) plus its own; the base's rows stay its population
    base = RR.base_population(rd)
    check('base_population: the base\'s harvested ledger rows of its population nights per Stokes (not its frame noise); no '
          'FRAME_NOISE is no base', base == {'I': [0.010, 0.016, 0.012, 0.014], 'V': [0.005, 0.007, 0.009]}
          and RR.base_population(os.path.join(T, 'a1')) is None, base)
    nz2 = {'I': {'2025-01-05': 0.02, '2025-01-06': 0.02}, 'V': {}}
    t2 = RR.table_noise(RR.ledger_rms([row('I', '2025-01-05', 0.030), row('I', '2025-01-05', 0.032)], LST, SC),
                        {st: set(v) for st, v in nz2.items()}, {'I': {'2025-01-05'}, 'V': set()}, set(), base=base)
    check('table_noise of an append: the appended admitted night with the base\'s rows plus its own; a frame not admitted '
          'with the base\'s median', t2['I']['night'] == {'2025-01-05': float(np.median(pop + [0.030, 0.032])),
                                                         '2025-01-06': float(np.median(pop))} and t2['I']['n'] == 4, t2)
    # the append's record, written as the engine writes it (the base's rows and record kept, the night appended), passes
    rows2 = RR.frame_noise_rows(LST, SC, ['2025-01-05'], {'I': {'2025-01-05': 0.02}}, {'I': {'2025-01-05'}}, t2, None, 'A2')
    with open(fnp, 'a', newline='') as fh:
        csv.writer(fh).writerows(rows2)
    with open(led, 'a', newline='') as fh:
        csv.DictWriter(fh, fieldnames=ST.GLED_COLS).writerows([row('I', '2025-01-05', 0.030), row('I', '2025-01-05', 0.032)])
    from astropy.io import fits
    fits.writeto(ST.frame_path(os.path.join(T, 'a1'), 'T', LST, SC, '2025-01-05', 'I'), np.zeros((4, 4), np.float32),
                 fits.Header([('IMGRMS', 0.02)]), overwrite=True)
    with open(os.path.join(rd, 'v_deep_images.csv'), 'a', newline='') as fh:
        csv.writer(fh).writerow([LST, SC, 'I', '2025-01-05', 0, 0, '', '', 0, 0, 25.0])
    ok2 = ST.medians_and_cards(fnp)
    check('the content test on an appended attempt: the base\'s rows keep the base\'s median, the appended night the median '
          'of the base\'s rows and its own', ok2[0], ok2)
    s = RR.frame_noise_summary(fnp, 'I')
    check('frame_noise_summary: the table noise the population rows carry (one value)',
          s['tblnoise_median_jy'] == float(np.median(pop)) and s['population'] == 3, s)


def test_engine_source():
    src = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_search.py')).read()
    i_tn = [m.start() for m in re.finditer(r'_RR\.table_noise\(', src)]
    i_close = src.find("_ahead_id.close(f'{cell_work}/ident_ahead.json')")
    i_istage = src.find('    if i_pending:\n')
    i_use = [m.start() for m in re.finditer(r'_b02_tnoise\(\'[IV]\', d_?\)', src)]
    i_led = src.find("_append_locked(f'{run_root}/v_deep_ledger.csv'")
    i_chk = src.find('the gate ledger this run writes differs from the rows its table noise was the median')
    check('the engine computes the table\'s noise once, after every night is extracted (the main loop closed) and before the '
          'table stage, from the ledger rows so far plus one row per pending I and V candidate; every table call after it',
          len(i_tn) == 1 and 0 < i_close < i_tn[0] < i_istage and i_use and min(i_use) > i_tn[0]
          and "for d_, front, *_x in i_pending for c in front" in src and "for d_, keptv, *_x in v_pending for c in keptv" in src,
          (i_tn, i_close, i_istage, i_use[:2]))
    check('the engine refuses, before writing its ledger, when the rows it writes differ from those its table noise was taken '
          'over', 0 < i_chk < i_led, (i_chk, i_led))
    check('RULES_VERSION names the change (an append onto an attempt of the earlier table noise refuses: deep_step\'s rules '
          'check)', RR.RULES_VERSION == 'b02-3d')


def test_real_derivation():
    p = os.path.join(OPS, 'p3s3_curve_derive.py')
    if not os.path.isfile(p):
        print('SKIP  the real derivation (exopipe_ops not mounted)')
        return
    lrows = [row('I', D[0], 0.010), row('I', D[0], 0.016), row('I', D[1], -0.012, snr='-7.5', global_snr='-8.0'),
             row('I', D[2], 0.014, dec=repr(37.2398 - 70.0 + 1e-3)), row('I', D[2], 0.9, dec=repr(37.2398 - 70.0 - 1e-3)),
             row('I', D[2], 0.9, snr='4.9'), row('I', D[2], 0.9, global_snr=''), row('V', D[0], 0.005), row('V', D[1], 0.007)]
    ad = os.path.join(T, 'deriv', 'att')
    os.makedirs(os.path.join(ad, 'transients', 'Run_TNTEST'), exist_ok=True)
    with open(os.path.join(ad, 'transients', 'Run_TNTEST', 'v_deep_ledger.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=ST.GLED_COLS)
        w.writeheader()
        w.writerows(lrows)
    child = ("import json, sys\n"
             "sys.path[:0] = [sys.argv[3], '/lustre/gh/main/clv15']\n"
             "import p3s3_curve_derive as PCD\n"
             "PCD.PS.LABEL = 'TNTEST'\n"
             "PCD.PS.published_attempts = lambda stokes_list: [dict(cell=sys.argv[2], dir=sys.argv[1])]\n"
             "H = PCD.load_harvest(['V', 'I'], 5.0)\n"
             "print('HARVEST ' + json.dumps({k[2]: dict(g=v['g'], rms=v['rms']) for k, v in H.items()}))\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith('P3') and k != 'PYTHONPATH'}
    env.update(P3_DEEP_MIN_ELEVATION='20.0', P3S_LABEL='TNTEST')   # the floor the installed singles tables were derived with
    import json
    import subprocess
    pr = subprocess.run([sys.executable, '-c', child, ad, f'{LST}_{SC}', OPS], cwd=T, env=env, capture_output=True, text=True,
                        timeout=600)              # production's own package path (the derivation imports exopipe.phase3 of main)
    out = [ln for ln in pr.stdout.splitlines() if ln.startswith('HARVEST ')]
    if pr.returncode or not out:
        check('the derivation\'s own load_harvest runs (read only)', False, (pr.returncode, pr.stderr[-800:]))
        return
    H = json.loads(out[-1][8:])
    mine = RR.ledger_rms(lrows, LST, SC)
    got = {st: RR.median_rms([x for v in mine[st].values() for x in v]) for st in mine}
    want = {st: v['g'] for st, v in H.items()}
    check('the derivation\'s own load_harvest (exopipe_ops p3s3_curve_derive.py, read only, in a child with production\'s '
          'package) at its tables\' floors gives the median the search records, per Stokes, over the same rows',
          got == want and all(sorted(H[st]['rms']) == sorted(x for v in mine[st].values() for x in v) for st in ('I', 'V')),
          (got, want))


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_harvest()
        test_population_and_recompute()
        test_engine_source()
        test_real_derivation()
    finally:
        if not os.environ.get('EXOPIPE3_TEST_KEEP'):
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
