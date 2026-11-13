"""The frozen gate populations of the 10-minute search (lib/search/gate_population.py, "the 10-minute cadence";
R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS (1) and (2)).

Data-free. Checks: the record of the search-tier gate per Stokes, the beam gate and the V sanity gate round-trips and is read
frozen (frozen_frames): a thin cell is the recorded outcome 'below-minimum', told apart from no record; no record, another
factor or minimum, an 'off' row, an inconsistent row and a population holding a frame of a night excluded now refuse. The
verdict of a new night's windows is the frozen median's, whatever the windows' own values are. The earlier frames' refusal
rows are carried under the append's tag (carried_refusals): only the rows of the attempt's newest search (the newest tag
over EVERY refusal file of the cell, drop-patch rows aside), only the frames the caller keeps, every field but the tag
unchanged; a newest tag that is not older than the append's refuses. The engine: a whole-cell 10-minute search writes the
rows and reads none; an append reads them, reads the stage-1 rms of its own windows only, and writes no row.
Run: python tests/test_search_gate_population_tenmin.py
"""
import csv
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

from exopipe.phase3.lib.search import gate_population as GP         # noqa: E402

T = '/tmp/exopipe3_tests_gate_population_tenmin'
FAILS, N = [], [0]
LST, BAND = '02h', '55MHz'
CELL = f'{LST}_{BAND}'
BASE_TAG, DROP_TAG, APPEND_TAG = '20261101T000000', '20261102T000000stdrop', '20261113T000000'


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


def _frames(nights):
    return [f'{n}T{w:02d}' for n in nights for w in range(6)]


def test_record_and_reader():
    d = os.path.join(T, 'unit')
    os.makedirs(d)
    no = lambda _k: False
    what = 'SEARCH-TIER GATE (I)'
    miss = _refuses(lambda: GP.frozen_frames(d, LST, BAND, 'search_tier_I', 2.0, 5, no, what))
    frames = _frames(['2024-12-27', '2024-12-29'])
    med = 0.1 + 0.2                                             # not exactly representable: the round trip must be exact
    GP.append(d, [GP.row(LST, BAND, 'search_tier_I', frames, med, 2.0, 5, 'applied', 'search', BASE_TAG),
                  GP.row(LST, BAND, 'search_tier_V', frames[:3], None, 2.0, 5, 'below-minimum', 'search', BASE_TAG),
                  GP.row(LST, BAND, 'v_sanity', frames, 0.75, 3.0, 5, 'applied', 'search', BASE_TAG)])
    fz = GP.frozen_frames(d, LST, BAND, 'search_tier_I', 2.0, 5, no, what)
    thin = GP.frozen_frames(d, LST, BAND, 'search_tier_V', 2.0, 5, no, 'SEARCH-TIER GATE (V)')
    vs = GP.frozen_frames(d, LST, BAND, 'v_sanity', 3.0, 5, no, 'V SANITY GATE')
    check('the search-tier, thin and V sanity rows read back: the median exactly, the thin cell as below-minimum (3 frames, no '
          'median), and no record refuses naming the whole-cell search',
          fz['median'] == med and fz['n'] == 12 and fz['nights'] == sorted(frames) and fz['outcome'] == 'applied'
          and thin['outcome'] == 'below-minimum' and thin['n'] == 3 and thin['median'] is None
          and vs['median'] == 0.75 and vs['factor'] == 3.0
          and 'no frozen population' in miss and 'search the cell whole again' in miss, (fz, thin, vs, miss))
    f1 = _refuses(lambda: GP.frozen_frames(d, LST, BAND, 'search_tier_I', 2.5, 5, no, what))
    f2 = _refuses(lambda: GP.frozen_frames(d, LST, BAND, 'search_tier_I', 2.0, 6, no, what))
    f3 = _refuses(lambda: GP.frozen_frames(d, LST, BAND, 'search_tier_I', 2.0, 5, lambda k: k[:10] == '2024-12-29', what))
    f4 = _refuses(lambda: GP.frozen_frames(d, LST, '59MHz', 'search_tier_I', 2.0, 5, no, what))
    check('another factor, another minimum, a population frame of a night excluded now and another cell\'s record refuse',
          'mix the rules' in f1 and 'mix the rules' in f2 and 'excludes now' in f3 and '2024-12-29T00' in f3
          and 'no frozen population' in f4, (f1, f2, f3, f4))
    bad = os.path.join(T, 'bad')
    os.makedirs(bad)
    GP.append(bad, [GP.row(LST, BAND, 'search_tier_I', [], None, 2.0, 5, 'off', 'search', BASE_TAG),
                    GP.row(LST, BAND, 'search_tier_V', frames, None, 2.0, 5, 'below-minimum', 'search', BASE_TAG),
                    GP.row(LST, BAND, 'v_sanity', frames[:2], 0.5, 3.0, 5, 'applied', 'search', BASE_TAG)])
    b1 = _refuses(lambda: GP.frozen_frames(bad, LST, BAND, 'search_tier_I', 2.0, 5, no, what))
    b2 = _refuses(lambda: GP.frozen_frames(bad, LST, BAND, 'search_tier_V', 2.0, 5, no, what))
    b3 = _refuses(lambda: GP.frozen_frames(bad, LST, BAND, 'v_sanity', 3.0, 5, no, what))
    GP.append(bad, [GP.row(LST, BAND, 'v_sanity', frames[:2], 0.6, 3.0, 5, 'applied', 'search', BASE_TAG)])
    b4 = _refuses(lambda: GP.frozen_frames(bad, LST, BAND, 'v_sanity', 3.0, 5, no, what))
    check('an off row, 12 frames recorded below a minimum of 5, a median applied with 2 frames, and two rows of different '
          'content refuse', 'never off' in b1 and 'inconsistent' in b2 and 'inconsistent' in b3 and 'different content' in b4,
          (b1, b2, b3, b4))
    new = {'2025-10-30T00': 2 * med, '2025-10-30T01': 2 * med * (1 + 1e-12), '2025-10-30T02': 50.0, '2025-10-30T03': 0.0}
    got = GP.judge(fz, new, sorted(new) + ['2025-10-30T04'])
    check('the new night\'s windows are judged by the frozen median alone: above 2.0 x it refused, exactly at it and '
          'unmeasured kept (its own values, one of them 50 x larger, move nothing); a thin row refuses none',
          got == ['2025-10-30T01', '2025-10-30T02'] and GP.judge(thin, new, sorted(new)) == [], got)


def _write(refdir, gate, rows):
    with open(os.path.join(refdir, f'{CELL}_{gate}_excluded.csv'), 'w', newline='') as fh:
        csv.writer(fh, lineterminator='\n').writerows(rows)


def test_carried_refusals():
    rd = os.path.join(T, 'refusals')
    os.makedirs(rd)
    old, base = '20261008T000000', BASE_TAG
    tier = [[LST, BAND, '2024-12-27T03', 'I', '0.4', '0.3', old],                                    # an older search's row
            [LST, BAND, '2024-12-27T03', 'I', '412.0', '300.5', 'ratio2xmedian', base],
            [LST, BAND, '2024-12-27T04', 'V', '9.1', '8.0', 'ratio2xmedian', base],
            [LST, BAND, '2024-12-29T00', 'I', '700.0', '300.5', 'ratio2xmedian', base],
            [LST, BAND, '2024-12-31T01', 'I', '650.0', '300.5', 'ratio2xmedian', DROP_TAG],            # a drop patch's row
            ['03h', BAND, '2024-12-27T03', 'I', '1.0', '0.5', 'ratio2xmedian', base]]                  # another cell's row
    _write(rd, 'searchtier', tier)
    _write(rd, 'sanity_I', [[LST, BAND, '2024-12-29', '99.0', '5.0', base]])
    check('the attempt\'s newest search tag is taken over every refusal file of the cell, drop-patch rows aside',
          GP.newest_search_tag(rd, CELL) == base and GP.newest_search_tag(rd, '09h_18MHz') is None,
          GP.newest_search_tag(rd, CELL))
    got = GP.carried_refusals(rd, CELL, 'searchtier', lambda k: k[:10] != '2024-12-29', APPEND_TAG)
    check('carried: the rows of the newest search for the frames kept, every field but the tag unchanged; the older '
          'search\'s row, the drop patch\'s row, the frame not kept and the other cell\'s row are left',
          got == [tier[1][:-1] + [APPEND_TAG], tier[2][:-1] + [APPEND_TAG]], got)
    # a later search of another gate file moves the attempt's newest tag: the search-tier rows of the earlier tag are no longer
    # the attempt's current rows (the readers drop them), so none is carried
    _write(rd, 'beam', [[LST, BAND, '2024-12-27T00', '1.80', '1.5', '20261105T000000']])
    check('a later search that recorded no search-tier refusal leaves none to carry',
          GP.carried_refusals(rd, CELL, 'searchtier', lambda k: True, APPEND_TAG) == [])
    late = _refuses(lambda: GP.carried_refusals(rd, CELL, 'searchtier', lambda k: True, '20261105T000000'))
    check('a newest tag that is not older than the append\'s refuses; a gate file that does not exist carries nothing',
          'not older' in late and GP.carried_refusals(rd, CELL, 'wbmember', lambda k: True, APPEND_TAG) == [], late)
    ext = [[LST, BAND, '2024-12-27T03', 'I', 'why', '20261109T000000', '2026-11-09T00:00:00Z']]
    _write(rd, 'extraction_I', ext)
    check('the tag of a source-extraction row is its sixth field',
          GP.refusal_tag('extraction_I', ext[0]) == '20261109T000000' and GP.newest_search_tag(rd, CELL) == '20261109T000000')


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_record_and_reader()
        test_carried_refusals()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
