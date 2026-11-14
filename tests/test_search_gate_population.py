"""The beam gate's frozen population (lib/search/gate_population.py; R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS (2)).

Checks: the record's rows and reader (one content per cell and gate; a thin cell is the recorded outcome 'below-minimum', told
apart from no record); frozen_beam refuses a missing record, another factor or minimum, a base searched with the gate off, an
inconsistent row and a population holding a night excluded now; the engine reads the record on the append path only and
writes it on the whole-cell path only, its whole-cell gate lines unchanged. On real records (read only; the published attempt of
10h_Blue Stokes I copied to scratch, the cell's products read where they are): the backfill reconstructs the attempt's
population and reproduces its beam refusals; the per-hour verdict of every night the whole-cell search judged equals that
search's verdict; the median equals the engine's own expression over the same areas; the content test passes the row and
fails a tampered one; a tampered refusal record makes the backfill refuse.
Run: python tests/test_search_gate_population.py
"""
import csv
import glob
import os
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

T = '/tmp/exopipe3_tests_gate_population'
FAILS, N = [], [0]
CELL, STOKES = '10h_Blue', 'I'


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


def test_record_and_reader():
    d = os.path.join(T, 'unit')
    os.makedirs(d)
    no = lambda _n: False
    miss = _refuses(lambda: GP.frozen_beam(d, '10h', 'Blue', 1.5, 5, no))
    nights = [f'2025-01-{i:02d}' for i in range(1, 8)]
    GP.append(d, [GP.row('10h', 'Blue', 'beam', nights, 0.25, 1.5, 5, 'applied', 'search', '20261101T000000')])
    GP.append(d, [GP.row('10h', 'Green', 'beam', nights[:3], None, 1.5, 5, 'below-minimum', 'search', '20261101T000000')])
    GP.append(d, [GP.row('10h', 'Red', 'beam', [], None, 1.5, 5, 'off', 'search', '20261101T000000')])
    fz = GP.frozen_beam(d, '10h', 'Blue', 1.5, 5, no)
    thin = GP.frozen_beam(d, '10h', 'Green', 1.5, 5, no)
    check('the record round-trips the median exactly; a thin cell reads as the recorded outcome below-minimum (n 3), a missing '
          'record refuses naming the backfill', fz['median'] == 0.25 and fz['n'] == 7 and fz['nights'] == nights
          and thin['outcome'] == 'below-minimum' and thin['n'] == 3 and thin['median'] is None
          and 'holds no frozen beam-gate population' in miss and GP.BACKFILL_STAGE in miss, (fz, thin, miss))
    check('frozen_beam refuses another factor, another minimum, a base searched with the gate off, a population night excluded '
          'now', all('REFUSING' in x for x in (
              _refuses(lambda: GP.frozen_beam(d, '10h', 'Blue', 1.4, 5, no)),
              _refuses(lambda: GP.frozen_beam(d, '10h', 'Blue', 1.5, 4, no)),
              _refuses(lambda: GP.frozen_beam(d, '10h', 'Red', 1.5, 5, no)),
              _refuses(lambda: GP.frozen_beam(d, '10h', 'Blue', 1.5, 5, lambda n: n == nights[2])))))
    GP.append(d, [GP.row('10h', 'Blue', 'beam', nights, 0.25, 1.5, 5, 'applied', 'backfill', '20261102T000000')])
    same = GP.read(d, '10h', 'Blue', 'beam')
    GP.append(d, [GP.row('10h', 'Blue', 'beam', nights, 0.26, 1.5, 5, 'applied', 'search', '20261103T000000')])
    two = _refuses(lambda: GP.read(d, '10h', 'Blue', 'beam'))
    check('rows of the same content (another tag, another basis) read as one; rows of different content refuse',
          same['median'] == 0.25 and 'rows of different content' in two, (same, two))
    bad = os.path.join(T, 'unit_bad')
    os.makedirs(bad)
    with open(os.path.join(bad, GP.FILE), 'w', newline='') as fh:
        csv.writer(fh).writerows([GP.COLS, ['10h', 'Blue', 'beam', '7', '0.25', '1.5', '5', 'applied', ';'.join(nights[:6]),
                                            'search', 't']])
    check('frozen_beam refuses an inconsistent row (n 7, 6 nights listed)',
          'inconsistent' in _refuses(lambda: GP.frozen_beam(bad, '10h', 'Blue', 1.5, 5, no)))
    check('judge: applied refuses area > factor x median; below-minimum judges none',
          GP.judge(fz, {'a': 0.375, 'b': 0.376}, ['a', 'b']) == ['b'] and GP.judge(thin, {'a': 9.0}, ['a']) == [])


def test_engine_wiring():
    src = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_search.py')).read()
    check('the engine reads the frozen population on the deep append path only (cadence deep and included_dates) and writes '
          'it on the whole-cell path only; the whole-cell gate is the unchanged expression',
          "if cadence == 'deep' and included_dates:" in src and '_GP.frozen_beam(run_root, lst, scope' in src
          and "if cadence == 'deep' and not included_dates:" in src and "_append_locked(f'{run_root}/{_GP.FILE}'" in src
          and "_bok = _enough(len(_fin), BEAM_GATE_MIN_NIGHTS, f'{lst} {scope}: beam gate, nights with a beam area')" in src
          and "_bmed = _fin[len(_fin) // 2] if _bfz is None else _bfz['median']" in src
          and "if _areas.get(d, 0) > _thr * _bmed]" in src)
    check('an append with the gate off (P3_BEAM_GATE_OFF=1) reads the base row and refuses unless the base ran with it off',
          "if os.environ.get('P3_BEAM_GATE_OFF') == '1' and cadence == 'deep' and included_dates:" in src
          and "if _bfz is None or _bfz['outcome'] != 'off':" in src)


def _copy_attempt():
    from exopipe.phase3.lib.search import deep_step as D
    att = D.resolve_published(CELL, STOKES, os.path.join(T, 'no_out'))
    if not att:
        return None, None
    rd = glob.glob(os.path.join(att, 'transients', 'Run_*'))[0]
    dst = os.path.join(T, os.path.basename(att.rstrip('/')))
    drd = os.path.join(dst, 'transients', os.path.basename(rd))
    os.makedirs(os.path.join(drd, 'refusals'))
    for p in glob.glob(os.path.join(rd, 'refusals', f'{CELL}_*_excluded.csv')) + [os.path.join(rd, 'v_deep_images.csv')]:
        shutil.copy2(p, os.path.join(drd, 'refusals' if '/refusals/' in p else '', os.path.basename(p)))
    return att, dst


def test_real_backfill_and_verdicts():
    from exopipe.phase3.core import exclusions, validate
    from exopipe.phase3.lib.search import admitted as ADS, wideband_admission as WBA
    from exopipe.phase3.lib.search import tests as _st      # noqa: F401 (registers the content tests)
    att, dst = _copy_attempt()
    if att is None:
        check(f'a published attempt of {CELL} {STOKES}', False, 'none resolved')
        return
    lst, scope = CELL.split('_', 1)
    cd = os.path.join(CFG.paths.COADD_SB_ROOT, CELL)
    adm = set(ADS.nights(cd, scope, 'deep')[0])
    ex = exclusions.load()
    area = lambda d: WBA.beam_area(cd, scope, lst, d)
    fac, mn = float(CFG.search_detect.BEAM_GATE_RATIO), int(CFG.search_detect.BEAM_GATE_MIN_NIGHTS)
    r, rep = GP.backfill_attempt(dst, CELL, STOKES, cd, fac, mn, area, adm, lambda d: ex.is_excluded(lst, scope, d))
    check(f'backfill of the published {os.path.basename(att)} (copy): accepted', r is not None, rep.get('refused'))
    if r is None:
        return
    drd = glob.glob(os.path.join(dst, 'transients', 'Run_*'))[0]
    GP.append(drd, [r])
    fz = GP.read(drd, lst, scope, 'beam')
    with open(os.path.join(drd, 'refusals', f'{CELL}_beam_excluded.csv'), newline='') as fh:
        batch = {x[2] for x in csv.reader(fh) if x}
    areas = {d: area(d) for d in fz['nights']}
    fin = sorted(areas.values())                                          # the engine's whole-cell expression
    per_hour = {d for d in fz['nights'] if GP.judge(fz, areas, [d])}      # each judged night alone, as an append judges it
    check(f'per-hour verdict = the whole-cell verdict for every night it judged ({fz["n"]} nights, {len(batch)} refused); the '
          f'frozen median = the engine\'s _fin[len // 2]', per_hour == batch - adm and fz['median'] == fin[len(fin) // 2]
          and fz['outcome'] == 'applied', (sorted(per_hour), sorted(batch), fz['median'], fin[len(fin) // 2]))
    res = validate.run('gate_population', os.path.join(drd, GP.FILE), cell=CELL)
    check('content test gate_population passes the backfilled row', res and all(x['passed'] for x in res), res)
    tp = os.path.join(T, 'tampered')
    shutil.copytree(dst, tp)
    trd = glob.glob(os.path.join(tp, 'transients', 'Run_*'))[0]
    rows = list(csv.reader(open(os.path.join(trd, GP.FILE), newline='')))
    rows[1][7], rows[1][4] = 'below-minimum', ''
    with open(os.path.join(trd, GP.FILE), 'w', newline='') as fh:
        csv.writer(fh).writerows(rows)
    res2 = validate.run('gate_population', os.path.join(trd, GP.FILE), cell=CELL)
    bp = os.path.join(trd, 'refusals', f'{CELL}_beam_excluded.csv')
    br = list(csv.reader(open(bp, newline='')))
    br[0][3] = f'{float(br[0][3]) + 0.5:.2f}'
    with open(bp, 'w', newline='') as fh:
        csv.writer(fh).writerows(br)
    r3, rep3 = GP.backfill_attempt(tp, CELL, STOKES, cd, fac, mn, area, adm, lambda d: ex.is_excluded(lst, scope, d))
    check('the content test fails a row whose outcome contradicts the recorded beam refusals; a beam refusal whose ratio the '
          'reconstruction does not reproduce makes the backfill refuse', res2 and not all(x['passed'] for x in res2)
          and r3 is None and 'does not reproduce' in rep3.get('refused', ''), (res2, rep3.get('refused')))
    # nights refused after the gate (daytime, source extraction) stay in its population (review of pull request 26, finding 1)
    bf = lambda at, a=adm: GP.backfill_attempt(at, CELL, STOKES, cd, fac, mn, area, a, lambda d: ex.is_excluded(lst, scope, d))
    searched = sorted(set(fz['nights']) - batch)
    for gate, extra in (('daytime', []), ('extraction_I', ['2026-11-01T00:00:00Z'])):
        dp = os.path.join(T, f'after_{gate}')
        shutil.copytree(dst, dp)
        prd = glob.glob(os.path.join(dp, 'transients', 'Run_*'))[0]
        os.remove(os.path.join(prd, GP.FILE))
        night, tag = searched[len(searched) // 2], r[10]
        ip = os.path.join(prd, 'v_deep_images.csv')
        rows = list(csv.DictReader(open(ip, newline='')))
        with open(ip, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(x for x in rows if not (x['date'] == night and x['stokes'] == STOKES))
        ra, repa = bf(dp)
        with open(os.path.join(prd, 'refusals', f'{CELL}_{gate}_excluded.csv'), 'a', newline='') as fh:
            csv.writer(fh).writerow([lst, scope, night, STOKES, f'{gate}, synthetic', tag, *extra])
        rb, repb = bf(dp)
        check(f'a night the attempt refused for {gate} (after the beam gate) stays in its population: the same row; without '
              f'that record it is unplaced', ra is None and 'do not place' in repa.get('refused', '')
              and rb is not None and rb[3:9] == r[3:9], (repa.get('refused'), repb.get('refused')))
    # admitted nights: an engine without admissions kept them in the population (finding 2) - never a silent choice
    one = searched[0]
    rk, repk = bf(dst, adm | {one})
    check(f'with a night admitted now ({one}): the row is the reading that reproduces the records, or the backfill refuses '
          f'when both readings do with different rows', (rk is not None and rk[3:9] == r[3:9]
                                                        and repk.get('admitted_reading', '').startswith(('kept', 'either')))
          or (rk is None and 'cannot be told' in repk.get('refused', '')), (repk.get('admitted_reading'), repk.get('refused')))


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_record_and_reader()
        test_engine_wiring()
        test_real_backfill_and_verdicts()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
