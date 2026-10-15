"""D-ADMIT (a), slice 1: the build's recorded statistics (lib/masters/build_stats.py, <cell>/coadd_final/build_stats.json).

Each statistic must be the number its gate computes: stage 2 as coadd.detect_outliers_mad (and the thresholds the build
applied, exactly), the beam gate's upper middle element over the images the co-add beam gate reads in the Science dirs the
build used, the Stokes I sanity gate's upper middle element, the search tier's numpy median with 1.4826 x MAD; the record
holds the settings it used and its content test re-derives everything with them and fails on every kind of edit (the
2026-10-15 review's list). The co-add beam gate, refactored onto beam_gate_image / beam_area, gives the old loop's areas.
The on-demand stage writes the record beside the records of the cell tree it is given (a copy here), refuses a production
root, an existing record, a population holding a night excluded now and a membership violation.
Run: python tests/test_masters_dadmit.py
"""
import contextlib
import csv
import io
import json
import os
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
from exopipe.phase3.lib.masters import build_stats as BS            # noqa: E402
from exopipe.phase3.lib.masters import coadd as co                  # noqa: E402

FAILS, N = [], [0]
T = '/tmp/exopipe3_tests_dadmit'
ROOT = CFG.paths.COADD_SB_ROOT


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def copy_records(cell, dst_root):
    """A copy of a cell's small records (what build_stats and membership.verify read) under dst_root/<cell>/."""
    src, dst = os.path.join(ROOT, cell), os.path.join(dst_root, cell)
    os.makedirs(os.path.join(dst, 'coadd_final'), exist_ok=True)
    for rel in ('residuals_provisional.csv', 'coadd_final/weights.csv', 'coadd_final/stage2_verdict.csv',
                'coadd_final/night_image_gate.csv', 'coadd_final/epoch_gate.csv'):
        if os.path.exists(os.path.join(src, rel)):
            shutil.copy2(os.path.join(src, rel), os.path.join(dst, rel))
    return dst


def test_formulas():
    rng = np.random.default_rng(15)
    vals = list(rng.normal(100, 10, 9)) + [float('nan'), 400.0]
    dates = [f'2025-01-{i + 1:02d}' for i in range(len(vals))]
    rows = [dict(i_rms_mjy=v, v_rms_mjy=v / 4) for v in vals]
    st = BS.stage2_stats(rows, float(co.OUTLIER_SIGMA), float(co.LOW_OUTLIER_SIGMA))
    _out, thr = co.detect_outliers_mad(dates, vals)
    check('stage 2: the high threshold is coadd.detect_outliers_mad\'s (finite values only)',
          st['I']['gate'] and abs(st['I']['thr_hi'] - thr) < 1e-9 and st['I']['n'] == 10, (st['I'], thr))
    few = BS.stage2_stats(rows[:4], 3.0, 4.0)
    v4 = np.array(vals[:4])
    m4 = float(np.median(v4))
    check('stage 2: fewer than 5 finite values - no gate, the population recorded (n, median, MAD), no threshold',
          few['I'] == {'n': 4, 'gate': False, 'median': m4, 'mad': float(np.median(np.abs(v4 - m4)))}, few['I'])
    areas = {'a': 1.0, 'b': 4.0, 'c': 2.0, 'd': 3.0, 'e': 9.0, 'f': 6.0}
    bs = BS.beam_stats(areas, 1.5, 5)
    check('beam gate: the upper middle element of the sorted areas and the nights above 1.5 x it',
          bs['median_area'] == 4.0 and bs['outliers'] == ['e'] and BS.beam_stats(dict(list(areas.items())[:4]), 1.5, 5)['gate']
          is False, bs)
    w = [dict(qa3_rms_mjy=x) for x in ('10', '20', '30', '40', '', 'nan', '-5', '50', '60')]
    ss = BS.sanity_stats(w, 5)
    check('sanity: the upper middle element of the finite positive qa3 values', ss['qa3_median'] == 40.0 and ss['n'] == 6, ss)
    prov = [dict(i_rms_mjy=x, v_rms_mjy=x) for x in (1.0, 2.0, 3.0, 4.0, 10.0, 6.0)]
    tt = BS.search_tier_stats(prov, 5.0, 5)['I']
    v = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 10.0])
    med, mad = float(np.median(v)), float(np.median(np.abs(v - np.median(v)))) * 1.4826
    check('search tier: numpy median and 1.4826 x MAD (not the upper middle element)',
          abs(tt['median'] - med) < 1e-12 and abs(tt['thr'] - (med + 5.0 * mad)) < 1e-12 and med == 3.5, (tt, med, mad))


def test_drift():
    src = open(os.path.join(REPO, 'exopipe/phase3/lib/search/deep_search.py')).read()
    check('drift: the search\'s qa3 discard is build_stats.X_DISCARD x the cell median',
          f'_wr.get(_kd(d), 0) > {BS.X_DISCARD} * _med' in src)
    import inspect
    check('drift: detect_outliers_mad\'s minimum count is build_stats.STAGE2_MIN',
          f'if len(valid) < {BS.STAGE2_MIN}:' in inspect.getsource(co.detect_outliers_mad))


def _old_beam_areas(lst, fl, dates_list):
    """coadd_beam_gate's area loop as it was before beam_gate_image / beam_area (2026-10-13 D1 rule), pinned here."""
    from astropy.io import fits
    out = {}
    for d in dates_list:
        sci = co.find_latest_science_dir(lst, d)
        img = None
        if sci:
            with contextlib.redirect_stdout(io.StringIO()):
                img = (co.find_subband_alignment_image(sci, fl[:-3]) if fl.endswith('MHz')
                       else co.find_wideband_alignment_image(sci, fl, lst, d))
        if img:
            try:
                h = fits.getheader(img)
                out[d] = float(h['BMAJ']) * float(h['BMIN'])
            except Exception:
                pass
    return out


def test_beam_gate_refactor():
    for cell in ('23h_23MHz', '11h_Full', '01h_18MHz'):
        lst, sc = cell.split('_')
        nights = sorted({r['date'] for r in csv.DictReader(open(os.path.join(ROOT, cell, BS.BUILD_NIGHTS)))})
        with contextlib.redirect_stdout(io.StringIO()):
            _o, new = co.coadd_beam_gate(lst, sc, nights)
        check(f'beam gate refactor ({cell}): beam_gate_image / beam_area give the old loop\'s areas', new == _old_beam_areas(lst, sc, nights),
              (new, cell))


def _edits(rec):
    """(name, edit) pairs of the 2026-10-15 review: each must fail the content test."""
    n0, n1 = sorted(rec['beam_gate']['nights'])[:2]

    def e(f):
        def g(s):
            s = json.loads(json.dumps(s))
            f(s)
            return s
        return g

    def drop_night(s):
        del s['beam_gate']['nights'][n0]
        s['beam_gate']['n'] -= 1

    def swap(s):
        a, b = s['beam_gate']['nights'][n0], s['beam_gate']['nights'][n1]
        s['beam_gate']['nights'][n0], s['beam_gate']['nights'][n1] = dict(b), dict(a)

    def missing_img(s):
        s['beam_gate']['nights'][n0]['image'] = '/nonexistent/x.fits'
        s['beam_gate']['nights'][n0]['area'] *= 3
    return [('drop a night from the beam nights', e(drop_night)),
            ('a night\'s area set to null', e(lambda s: s['beam_gate']['nights'][n0].update(area=None))),
            ('a night\'s image missing, its area tripled', e(missing_img)),
            ('two nights\' images and areas swapped', e(swap)),
            ('the sources pointed at another cell', e(lambda s: s['sources'].update(weights='../23h_27MHz/coadd_final/weights.csv'))),
            ('the cell changed', e(lambda s: s.update(cell='23h_27MHz'))),
            ('the version changed', e(lambda s: s.update(version=1))),
            ('the applied threshold copy raised by 0.09 per cent', e(lambda s: s['stage2_recorded_thr'].update(I=[t * 1.0009 for t in s['stage2_recorded_thr']['I']]))),
            ('the applied threshold copy emptied', e(lambda s: s['stage2_recorded_thr'].update(I=[]))),
            ('a night renamed', e(lambda s: s.update(nights=['1999-01-01'] + s['nights'][1:]))),
            ('the search tier block deleted', e(lambda s: s.pop('search_tier'))),
            ('a setting removed', e(lambda s: s['settings'].pop('SEARCH_TIER_SIGMA'))),
            ('the search tier threshold edited', e(lambda s: s['search_tier']['V'].update(thr=s['search_tier']['V']['thr'] * 1.01))),
            ('a beam area doubled', e(lambda s: s['beam_gate']['nights'][n0].update(area=s['beam_gate']['nights'][n0]['area'] * 2)))]


def test_record_and_recheck():
    for cell in ('23h_23MHz', '11h_Red'):
        lst, sc = cell.split('_')
        cd = copy_records(cell, os.path.join(T, 'copies'))
        st = BS.compute(cd, lst, sc, co, CFG, 'stage')
        rec_thr = st['stage2_recorded_thr']
        check(f'real {cell}: the applied stage-2 thresholds equal the re-derived ones exactly',
              all(rec_thr[s] == [st['stage2'][s]['thr_hi']] for s in ('I', 'V')), (rec_thr, st['stage2']))
        check(f'real {cell}: sources relative to the cell, the nights of its record, the settings it used',
              st['sources'] == {'stage2': 'coadd_final/stage2_verdict.csv', 'residuals_provisional': 'residuals_provisional.csv',
                                'weights': 'coadd_final/weights.csv'} and st['nights'] == sorted(st['beam_gate']['nights'])
              and set(st['settings']) == set(BS.SETTINGS), (st['sources'], st['settings']))
        p = BS.write(cd, st)
        good = BS.recheck(p, co)
        check(f'real {cell}: the content test passes on the record', not good, good)
        bad = []
        for name, f in _edits(st):
            json.dump(f(st), open(p, 'w'))
            if not BS.recheck(p, co):
                bad.append(name)
        check(f'real {cell}: the content test fails on each of the review\'s {len(_edits(st))} edits', not bad, bad)
        s2 = json.loads(json.dumps(st))
        s2['settings']['SEARCH_TIER_SIGMA'] = 4.0
        s2['search_tier'] = BS.search_tier_stats(list(csv.DictReader(open(os.path.join(cd, BS.BUILD_NIGHTS)))), 4.0,
                                                 st['settings']['SEARCH_TIER_MIN'])
        json.dump(s2, open(p, 'w'))
        check(f'real {cell}: a record made with other settings passes - it is tested against its own settings, not today\'s',
              not BS.recheck(p, co), BS.recheck(p, co))
    cd = copy_records('01h_18MHz', os.path.join(T, 'copies'))
    st = BS.compute(cd, '01h', '18MHz', co, CFG, 'stage')
    check('real 01h_18MHz (built before the 2026-10-13 image rule): the beam block says today\'s rule disagrees with the build',
          st['beam_gate']['agree'] is False and '2024-12-27' in st['beam_gate']['outliers']
          and '2024-12-27' not in st['beam_gate']['recorded_outliers'], st['beam_gate'].get('outliers'))


def test_wiring():
    from exopipe.phase3.lib.masters import deep_stages as DS
    import inspect
    src = inspect.getsource(DS.build)
    check('the build writes the record (basis build) only under masters_build.WRITE_BUILD_STATS, after the co-add, before the '
          'outputs are tested', 0 < src.index('if _cfg().masters_build.WRITE_BUILD_STATS:') < src.index('for p, cls in _build_outputs(')
          and "co, _cfg(), 'build')" in src and 'coadd_final/build_stats.json' in DS.BUILD_PRODUCT_PATHS)
    prod = open(os.path.join(REPO, 'exopipe/phase3/parity/production_20261013.cfg')).read()
    check('production values write no record (parity file)', 'masters_build.WRITE_BUILD_STATS = False' in prod
          and CFG.masters_build.WRITE_BUILD_STATS is True)
    from exopipe.phase3.lib.install import steps_deep as ISD
    check('install carries build_stats.json with the cell\'s records', ISD.BUILD_STATS in ISD.CARRIED_RECORDS)


def cli(*args):
    p = subprocess.run([sys.executable, '-m', 'exopipe.phase3', 'masters'] + list(args), cwd=REPO, env=dict(os.environ),
                       capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


def _rec(out, cell):
    p = f'{out}/_records/masters/build-stats_{cell}.json'
    return json.load(open(p)) if os.path.exists(p) else {}


def test_stage():
    """`masters --stages build-stats` on a copy of an installed cell: the record beside the copy's records, equal to
    build_stats.compute; refusals and no input."""
    from exopipe.phase3.lib.masters import deep_stages as DS
    check('build-stats then admit-prep: the last deep stages, in no default; the module\'s order is the config\'s',
          DS.STAGE_ORDER == tuple(CFG.masters.STAGES['deep']) and DS.STAGE_ORDER[-2:] == ('build-stats', 'admit-prep')
          and not {'build-stats', 'admit-prep'} & set(CFG.masters.DEFAULT_STAGES['deep']), DS.STAGE_ORDER)
    cell, lst, sc = '23h_23MHz', '23h', '23MHz'
    wt = os.path.join(T, 'wt')
    cd = copy_records(cell, wt)
    out = f'{T}/out'
    dest = f'{cd}/coadd_final/{BS.FILE}'
    common = ['--tier', 'deep', '--cells', cell, '--stages', 'build-stats', '--cores', 'none', '--out', out]
    rc, txt = cli(*common, '--deep-installed-root', wt, '--dry-run')
    check('dry run: plans the record beside the copy\'s records, no gate, writes nothing',
          rc == 0 and dest in txt and 'GATE' not in txt and not os.path.exists(dest), txt[-500:])
    rc, txt = cli(*common)
    r0 = _rec(out, cell)
    check('the default tree (production) is refused until cutover', rc == 2 and 'production' in (r0.get('reason') or ''),
          (r0.get('reason'), txt[-300:]))
    rc, txt = cli(*common, '--deep-installed-root', wt)
    rec = _rec(out, cell)
    got = json.load(open(dest)) if os.path.exists(dest) else None
    want = json.loads(json.dumps(BS.compute(cd, lst, sc, co, CFG, 'stage')))
    check('the stage (exit 0) writes the record beside the records, equal to build_stats.compute (basis stage)',
          rc == 0 and rec.get('status') == 'done' and got == want, (txt[-500:], rec.get('reason')))
    names = {t['name']: t['passed'] for t in rec.get('tests', [])}
    check('its content test, the population check and membership.verify passed',
          names.get('build_stats.build_stats_rederived') is True and names.get('masters_build_stats.cell_membership') is True
          and names.get('masters_build_stats.population_not_excluded') is True, names)
    import argparse
    from exopipe.phase3.core import record as R

    class _Run:                                   # the real resume_check (core.record), as Run.resume_check calls it
        def resume_check(self, unit, inputs, admitted):
            return R.resume_check(out, 'masters', unit, inputs=inputs, admitted=admitted)
    rp = f'{out}/_records/masters/build-stats_{cell}.json'
    rc_ = json.load(open(rp))
    rc_['commit'] = str(rc_.get('commit')).replace('-dirty', '')      # this scratch record: as if from a clean tree
    json.dump(rc_, open(rp, 'w'))
    same = DS._resume_ok(_Run(), f'build-stats_{cell}', 'build-stats', cell, out, argparse.Namespace(deep_installed_root=wt))
    wt2 = os.path.join(T, 'wt2')
    shutil.copytree(wt, wt2)
    other = DS._resume_ok(_Run(), f'build-stats_{cell}', 'build-stats', cell, out, argparse.Namespace(deep_installed_root=wt2))
    check('build-stats --resume (real resume_check): the same tree skips - its inputs are the records and every beam image, '
          'as recorded -; another cell tree re-runs', same[0] is True and other == (False, 'input set changed')
          and len(rec.get('inputs', {})) > 3, (same, other, len(rec.get('inputs', {}))))
    rc, _ = cli(*common, '--deep-installed-root', wt)
    r2 = _rec(out, cell)
    rc3, _ = cli(*common, '--deep-installed-root', wt, '--fresh')
    r3 = _rec(out, cell)
    check('an existing record refuses without --fresh; --fresh recomputes it', rc == 2 and 'fresh' in r2.get('reason', '')
          and rc3 == 0 and r3.get('status') == 'done', (r2.get('reason'), r3.get('reason')))
    got['basis'] = 'build'
    json.dump(got, open(dest, 'w'))
    rc, _ = cli(*common, '--deep-installed-root', wt)
    r4 = _rec(out, cell)
    rc4, _ = cli(*common, '--deep-installed-root', wt, '--fresh')
    aside = sorted(f for f in os.listdir(f'{cd}/coadd_final') if f.startswith('build_stats.superseded_'))
    old_basis = [json.load(open(f'{cd}/coadd_final/{f}')).get('basis') for f in aside]
    left = [f for f in os.listdir(f'{cd}/coadd_final') if '.part' in f]
    check('a record of either basis refuses without --fresh; --fresh replaces it and moves the old one aside, no partial file',
          rc == 2 and "'build'" in r4.get('reason', '') and rc4 == 0 and json.load(open(dest))['basis'] == 'stage'
          and 'build' in old_basis and not left, (r4.get('reason'), rc4, aside, old_basis, left))
    cd7 = copy_records('07h_18MHz', wt)                    # 2025-10-14: excluded since its build, in the population
    rc, _ = cli('--tier', 'deep', '--cells', '07h_18MHz', '--stages', 'build-stats', '--cores', 'none', '--out', out,
                '--deep-installed-root', wt)
    r5 = _rec(out, '07h_18MHz')
    check('a population holding a night excluded now refuses (07h_18MHz 2025-10-14, a non-member)',
          rc == 2 and '2025-10-14' in r5.get('reason', '') and not os.path.exists(f'{cd7}/coadd_final/{BS.FILE}'), r5.get('reason'))
    cdm = copy_records('11h_Red', wt)
    w = list(csv.DictReader(open(f'{cdm}/coadd_final/weights.csv')))
    s2 = {r['date']: r for r in csv.DictReader(open(f'{cdm}/coadd_final/stage2_verdict.csv'))}
    out_n = [r for r in w if r.get('in_master') == '0' and str(s2.get(r['date'], {}).get('outlier')) in ('True', '1')]
    if out_n:
        out_n[0]['in_master'] = '1'                          # a stage-2 outlier put in the master: a membership violation
        with open(f'{cdm}/coadd_final/weights.csv', 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(w[0].keys()))
            wr.writeheader()
            wr.writerows(w)
        rc, _ = cli('--tier', 'deep', '--cells', '11h_Red', '--stages', 'build-stats', '--cores', 'none', '--out', out,
                    '--deep-installed-root', wt)
        r6 = _rec(out, '11h_Red')
        check('a membership violation refuses', rc == 2 and 'membership.verify' in r6.get('reason', ''), r6.get('reason'))
    else:
        check('a membership violation refuses (no stage-2 outlier in 11h_Red to misplace: skipped)', True)
    os.makedirs(f'{T}/empty/{cell}')
    rc, _ = cli(*common[:-1], f'{T}/out_empty', '--deep-installed-root', f'{T}/empty')
    r7 = _rec(f'{T}/out_empty', cell)
    check('no input (exit 3) without the build records', rc == 3 and r7.get('status') == 'no-input'
          and 'residuals_provisional.csv' in r7.get('reason', ''), r7.get('reason'))


def test_review2():
    """The second review's cases (2026-10-15): an extra key or a boolean written as 1 fails the content test; a night
    without a readable beam does not; a population below a gate's minimum is recorded (n, median, MAD); a record made
    after the build carries the build's own OUTLIER_SIGMA, read back from the threshold it applied, so a later ruling of
    the sigma does not orphan old cells; another stage-2 metric or records of two generations give no record."""
    from exopipe.phase3.lib.masters import build_stats as BS2
    cd = copy_records('23h_23MHz', os.path.join(T, 'review2'))
    st = BS2.compute(cd, '23h', '23MHz', co, CFG, 'stage')
    p = BS2.write(cd, st)
    s = json.load(open(p))
    s['extra'] = 1
    json.dump(s, open(p, 'w'))
    extra = BS2.recheck(p, co)
    s.pop('extra')
    s['beam_gate']['gate'] = 1
    json.dump(s, open(p, 'w'))
    gate1 = BS2.recheck(p, co)
    check('review 2: an extra key and a boolean written as 1 fail the content test', extra and gate1, (extra, gate1))
    real_area = co.beam_area
    n0 = sorted(st['beam_gate']['nights'])[0]
    img0 = st['beam_gate']['nights'][n0]['image']
    co.beam_area = lambda img: None if img == img0 else real_area(img)
    try:
        st2 = BS2.compute(cd, '23h', '23MHz', co, CFG, 'stage')
        p2 = BS2.write(cd, st2)
        ok2 = BS2.recheck(p2, co)
    finally:
        co.beam_area = real_area
    check('review 2: a night whose image has no readable beam is skipped, as the gate skips it (the record passes)',
          st2['beam_gate']['nights'][n0]['area'] is None and not ok2, ok2)
    few = BS2.stage2_stats([dict(i_rms_mjy=x, v_rms_mjy=x) for x in (1.0, 2.0, 4.0)], 3.0, 4.0)['I']
    tier = BS2.search_tier_stats([dict(i_rms_mjy=x, v_rms_mjy=x) for x in (1.0, 2.0, 4.0)], 5.0, 5)['I']
    check('review 2: below a gate\'s minimum the population is recorded all the same (n, median, MAD; no threshold)',
          few == dict(n=3, gate=False, median=2.0, mad=1.0) and tier['gate'] is False and tier['median'] == 2.0
          and 'thr' not in tier, (few, tier))
    s2_rows = BS2.stage2_source(cd)[1]
    check('review 2: the build\'s OUTLIER_SIGMA read back from the threshold it applied (3, as production)',
          BS2.build_outlier_sigma(s2_rows, 9.9) == float(co.OUTLIER_SIGMA), BS2.build_outlier_sigma(s2_rows, 9.9))
    old_sigma = co.OUTLIER_SIGMA
    co.OUTLIER_SIGMA = 4.0                     # as if the sigma were ruled to 4 after the build
    try:
        st3 = BS2.compute(cd, '23h', '23MHz', co, CFG, 'stage')
        p3 = BS2.write(cd, st3)
        ok3 = BS2.recheck(p3, co)
    finally:
        co.OUTLIER_SIGMA = old_sigma
    check('review 2: with a new sigma in force, a record made after the build carries the build\'s and passes',
          st3['settings']['OUTLIER_SIGMA'] == 3.0 and not ok3, (st3['settings']['OUTLIER_SIGMA'], ok3))
    w = list(csv.DictReader(open(os.path.join(cd, BS2.BUILD_NIGHTS))))
    with open(os.path.join(cd, BS2.BUILD_NIGHTS), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(w[0].keys()))
        wr.writeheader()
        wr.writerows(w[:-1])
    try:
        BS2.compute(cd, '23h', '23MHz', co, CFG, 'stage')
        gen = 'no refusal'
    except BS2.NoRecord as e:
        gen = str(e)
    check('review 2: records of two generations (a night missing from residuals_provisional.csv) give no record',
          'generation' in gen, gen)


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_formulas()
        test_drift()
        test_beam_gate_refactor()
        test_record_and_recheck()
        test_wiring()
        test_stage()
        test_review2()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
