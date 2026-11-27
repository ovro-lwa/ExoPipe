"""The deep search's archive frames never carry fewer standard cards than the frames they supersede (the refactor chat's
decision (3) of 2026-11-27, case (b): archived search frames' cards are read - the BMAJ gate through gate_inputs.deep_target_beam,
the archive push / audit content tests, the pages' cutouts - so every archive frame the search writes is the superseded frame's
full header updated with the search's own cards, written with its header validated; a frame with no predecessor validates or
refuses).

Checks, data-free (a scratch archive root; frames 16 x 16 px with a standard header built by core.products):
  quarantined   the predecessor in this round's quarantine Run (the pre-search quarantine moved it), equal pixels: the archive
                frame is its header updated with the search's cards - the cards only it had (ASTREF, FLUXREF, SCALE_A, P1RUN,
                P2RUN, CALBP) carried with its values, the search's cards with the search's values, the work frame's COMMENT
                lines (decision (5)), the work frame's pixels bit for bit, a HISTORY line naming the carried cards; the content
                test passes
  differ        a predecessor with other pixels: the same header rule, the search's pixels
  in place      a frame already at the destination (written earlier in this launch) is the predecessor before the round
  none          no predecessor: the work frame's own header, written when it validates; refused (nothing placed, no temporary
                file) when it does not
  older round   a frame only in an older quarantine round (20261004): never a predecessor (decision (6a)); the frame validates
                on its own
  search-owned  a predecessor's search-owned cards the search does not write (LOOEXCL, PBFILE) dropped and named in a HISTORY
                line, its stale IMGRMS replaced; THERMNOI, the frame's noise card (ruling (1) of 2026-10-16), carried when the
                search writes none (its I frames) (decision (6b))
  owned set     SEARCH_OWNED_CARDS derived from the writers' source: the literal cards of the frame writers and the standard
                header's cards of the arguments the frame writer computes itself (not those read from the epoch's Stage-1
                product), less THERMNOI - over every committed version of production's deep frame writer and v2's writer, checked
                in as tests/fixtures/search_owned_cards.json (fix round 2 (8)); CI checks the constant and v2's writer against
                the fixture, a run beside production's repositories re-derives the history and compares it, commit for commit
  record        the engine's archive record (P3_ARCHIVE_RECORD): each archive frame with the predecessor it merged, its cards,
                the carried and dropped cards (decision (7)); the content test reads it - a frame replaced in place by its
                recorded cards, a frame without a row fails, a predecessor other than the recorded one fails
  wcs           a predecessor holding coordinate cards the search's frame lacks: refused, every card and the cell named, the
                destination as it was (decision (8): per (cell, Stokes) unit)
  test          the content test (search_frame copy_identical) refuses a copy lacking a card of its predecessor (the plain copy
                v2 made before), a copy whose pixels differ, a copy lacking the work frame's COMMENT lines, a copy dropping a
                search-owned card without naming it
  case (a)      not taken: the readers of archived copies' cards (the deep step's content tests, the removal rules' beam)
  step          the deep step hands the engine the predecessor roots - the archive root, this launch's quarantine round only,
                and in the window the production archive in place - and the record's path; the engine no longer copies the work
                frame as is; the round from the run's fixed launch (run_launch), never per-stage state
  separate run  --stages quarantine in one run, --stages detect in another (fix round 2 (2)): the second run finds the frame
                the first moved into the launch's round and carries its repair cards
  register      the deep step registers each archive frame for the content tests: the attempt frame's PRODNAME (over CONTINUE
                cards) read whole, the archive frame found, with the attempt frame and the engine's record
Run: python tests/test_search_archive_header.py
"""
import inspect
import json
import os
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
if os.path.isfile('/lustre/gh/main/exopipe_ops/exopipe3_site.py'):       # absent on a data-free machine: no site file is loaded
    os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

import warnings                                                         # noqa: E402
import numpy as np                                                      # noqa: E402
from astropy.io import fits                                             # noqa: E402
from astropy.wcs import WCS                                             # noqa: E402

from exopipe.phase3.config import CFG                                   # noqa: E402
from exopipe.phase3.core import products as prod                        # noqa: E402
from exopipe.phase3.core import validate                                # noqa: E402
from exopipe.phase3.lib.search import deep_search as DSE                # noqa: E402
from exopipe.phase3.lib.search import deep_step as DS                   # noqa: E402
from exopipe.phase3.lib.frames import tests as FT                       # noqa: E402

warnings.filterwarnings('ignore', module='astropy.wcs')
T = f'/tmp/exopipe3_tests_search_archive_header_{os.getpid()}'
ARCH, WORK = os.path.join(T, 'archive'), os.path.join(T, 'attempt', 'Run_T', 'vresid', '03h_18MHz')
RUN, QRUN = 'Run_20260909_000000', 'Run_SEARCH_QUARANTINE_20261127'
OLD_ONLY = {'ASTREF': '2024-12-29', 'FLUXREF': '2024-12-29', 'SCALE_A': 1.0123, 'P1RUN': 'Run_20260331_171539',
            'P2RUN': 'Science_20260525_220814', 'CALBP': 'bcal_20241218.bcal'}
N, FAILS = [0], []


def check(name, ok, detail=None):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name)
    if not ok:
        FAILS.append(name)
        if detail is not None:
            print('      ', str(detail)[:1500])


def header(date, imgrms, gittag, stokes='V', thermnoi=0.2, extra=None):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---SIN', 'DEC--SIN']
    w.wcs.crval = [52.0, 37.0]
    w.wcs.crpix = [8.0, 8.0]
    w.wcs.cdelt = [-0.5, 0.5]
    w.wcs.cunit = ['deg', 'deg']
    h = w.to_header()
    h['BUNIT'] = ('Jy/beam', 'Brightness unit')
    h['TELESCOP'] = ('OVRO-LWA', 'Telescope')
    ex = {'LOOMSTR': (False, 'Differenced against leave-one-out master'), 'MSTRFILE': ('m.fits', 'Master actually subtracted'),
          'CADENCE': ('deep', 'Frame cadence'), 'RUNLABEL': ('T', 'Search run label')}
    ex.update(extra or {})
    prod.write_standard_header(h, scope='18MHz', stokes=stokes, start=f'{date}T04:30:00', end=f'{date}T05:29:41',
                               inttime_s=3581.0, time_basis='phase2', beam=(1.0, 0.8, 10.0), ebeam=(1.1, 1.1, 0.0),
                               thermnoi=thermnoi, imgrms=imgrms, pbcorr=False, dewarped=True, aligned=True, alignref='a.fits',
                               msub=True, msubfile='m.fits', ssub=True, extra=ex, validate=False)
    h['PRODNAME'] = prod.build_name('18MHz', stokes, 'deep', 'Taper', 0.0, proc=['dewarped', 'aligned', 'msub', 'ssub'],
                                    lst='03h', epoch=f'{date}T04:30:00')
    h['GITTAG'] = (gittag, 'pipeline commit')
    return h


COMMENT = 'search frame: a COMMENT line of the work frame, kept on the archive copy'
REC = os.path.join(T, 'archive_record.csv')
PROD_REPO = '/lustre/gh/main/exopipe'                                          # production's code (git; read only)
PROD_DEEP = PROD_REPO + '/exopipe/phase3/phase3_deep.py'                       # production's writer of the archived frames
PROD_PRODUCTS = PROD_REPO + '/exopipe/common/phase3_products.py'
# decision (8) of fix round 2: the search-owned set is derived from every committed version of production's deep frame writer -
# each tracked file named *phase3_deep.py of these repositories (production's code and the ops tree's copies of it), each commit
# of its history (git log --follow, then git show per commit; no directory walk) - plus v2's writer; checked in as a fixture
WRITER_REPOS = (PROD_REPO, '/lustre/gh/main/exopipe_ops')
WRITER_NAME = 'phase3_deep.py'
WRITER_FNS = ('_archive_frame_header', '_write_pbcorr_sibling', '_write_archive_frame')     # the archive frame writers
FIXTURE = os.path.join(REPO, 'tests', 'fixtures', 'search_owned_cards.json')


def work_frame(date, value, **k):
    h = header(date, 0.21, 'e1a4e65', **k)
    h.add_comment(COMMENT)
    p = os.path.join(WORK, f'{date}_V_redo_resid_18MHz.fits')
    os.makedirs(WORK, exist_ok=True)
    d = np.full((16, 16), value, np.float32)
    d[3, 4] = np.nan
    fits.writeto(p, d, h, overwrite=True)
    return p, h


def archived(where, date, value, extra=None, after=None):
    """The frame production archived: the same standard header with the cards header_repair stamped (OLD_ONLY), the earlier
    search's IMGRMS and GITTAG; ``after``: cards set last (over the standard header's)."""
    h = header(date, 0.21, '634c659', extra=extra)
    for k, v in OLD_ONLY.items():
        h[k] = (v, 'stamped on the archive copy')
    for k, v in (after or {}).items():
        h[k] = v
    rel = os.path.join('03h', date, RUN, '18MHz', 'V', 'deep', h['PRODNAME'])
    p = os.path.join(where, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    d = np.full((16, 16), value, np.float32)
    d[3, 4] = np.nan
    fits.writeto(p, d, h, overwrite=True)
    return p, os.path.join(ARCH, rel)


def same_pixels(a, b):
    with fits.open(a, memmap=False) as x, fits.open(b, memmap=False) as y:
        return np.array_equal(np.ascontiguousarray(x[0].data).view(np.uint8), np.ascontiguousarray(y[0].data).view(np.uint8))


def hdr(p):
    with fits.open(p, memmap=False) as h:
        return h[0].header.copy()


def publish(work):
    return DSE._archive_frame_publish(work, '03h', '18MHz', os.path.basename(work)[:10], 'V', 'deep', '03h 18MHz')


def record_rows():
    import csv
    with open(REC, newline='') as fh:
        return {r['archive']: r for r in csv.DictReader(fh)}


def cardset(p):
    h = hdr(p)
    return {k for k in h if k not in ('COMMENT', 'HISTORY', '') and not DSE._STRUCT_CARD.match(k)}


def test_branches():
    os.environ.update(P3_ARCHIVE_ROOT=ARCH, P3_ARCHIVE_RUN=RUN, P3_ARCHIVE_RECORD=REC,
                      P3_ARCHIVE_PREDECESSOR_ROOTS=os.pathsep.join([ARCH, os.path.join(ARCH, QRUN)]))
    os.environ.pop('P3_ARCHIVE_WRITE', None)
    # quarantined predecessor, equal pixels
    w1, h1 = work_frame('2024-12-18', 1.5)
    q1, dst1 = archived(os.path.join(ARCH, QRUN), '2024-12-18', 1.5)
    out = publish(w1)
    a = hdr(dst1)
    carried_ok = all(a.get(k) == v for k, v in OLD_ONLY.items())
    hw1 = hdr(w1)                                # the work frame as written (FITS float formatting)
    own_ok = all(str(a.get(k)) == str(hw1.get(k)) for k in hw1 if k not in ('COMMENT', 'HISTORY', ''))
    hist = ' '.join(str(x) for x in a['HISTORY']) if 'HISTORY' in a else ''
    check('predecessor in the round\'s quarantine Run (equal pixels): the archive frame carries its cards ASTREF FLUXREF '
          'SCALE_A P1RUN P2RUN CALBP with their values, every card of the search\'s frame with the search\'s value (GITTAG '
          'e1a4e65, not 634c659), the work frame\'s pixels bit for bit, a HISTORY line naming the carried cards',
          out == dst1 and carried_ok and own_ok and a['GITTAG'] == 'e1a4e65' and same_pixels(dst1, w1)
          and 'carried from the superseded archive frame: ASTREF FLUXREF SCALE_A P1RUN P2RUN CALBP' in hist
          and os.stat(dst1).st_mtime_ns == os.stat(w1).st_mtime_ns,
          ({k: a.get(k) for k in OLD_ONLY}, a.get('GITTAG'), hist))
    com = [str(x) for x in a['COMMENT']] if 'COMMENT' in a else []
    check('the work frame\'s COMMENT lines are on the archive frame (decision (5)), once', com.count(COMMENT) == 1, com)
    r = validate.run('search_frame', dst1, copy_of=w1, archive_record=REC, stokes='V')
    want = ('search_frame.header_full', 'search_frame.copy_identical', 'search_frame.prodname')
    got = {x['name']: x['passed'] for x in r}
    check('the archive frame passes the search_frame content tests that read its header and its copy (header_full, '
          'copy_identical with the engine\'s record, prodname)', all(got.get(k) for k in want), got)
    # a predecessor with other pixels
    w2, _ = work_frame('2024-12-19', 2.5)
    q2, dst2 = archived(os.path.join(ARCH, QRUN), '2024-12-19', 9.0)
    publish(w2)
    a2 = hdr(dst2)
    check('a predecessor with other pixels: the same header rule, the search\'s pixels',
          all(a2.get(k) == v for k, v in OLD_ONLY.items()) and a2['GITTAG'] == 'e1a4e65' and same_pixels(dst2, w2)
          and not same_pixels(dst2, q2))
    # in place: a frame already at the destination comes before the quarantine Run
    w3, _ = work_frame('2024-12-20', 3.5)
    ip, dst3 = archived(ARCH, '2024-12-20', 3.5, extra={'INPLACE': ('yes', 'only on the in-place frame')})
    archived(os.path.join(ARCH, QRUN), '2024-12-20', 3.5, extra={'QRUNONLY': ('yes', 'only on the quarantined frame')})
    publish(w3)
    a3 = hdr(dst3)
    check('a frame already at the destination (written earlier in the launch) is the predecessor, before the round',
          a3.get('INPLACE') == 'yes' and 'QRUNONLY' not in a3 and same_pixels(dst3, w3))
    # no predecessor: valid -> written; invalid -> refused, nothing placed
    w4, h4 = work_frame('2024-12-21', 4.5)
    publish(w4)
    dst4 = os.path.join(ARCH, '03h', '2024-12-21', RUN, '18MHz', 'V', 'deep', h4['PRODNAME'])
    a4 = hdr(dst4)
    check('no predecessor: the work frame\'s own header, validated, written',
          all(str(a4.get(k)) == str(hdr(w4).get(k)) for k in hdr(w4) if k not in ('COMMENT', 'HISTORY', ''))
          and not any(k in a4 for k in OLD_ONLY) and same_pixels(dst4, w4))
    w5, h5 = work_frame('2024-12-22', 5.5)
    with fits.open(w5, mode='update', memmap=False) as h:
        del h[0].header['DATE-OBS']
        del h[0].header['BMAJ']
    dst5 = os.path.join(ARCH, '03h', '2024-12-22', RUN, '18MHz', 'V', 'deep', h5['PRODNAME'])
    try:
        publish(w5)
        r5 = 'not refused'
    except RuntimeError as e:
        r5 = str(e)
    left = [x for x in os.listdir(os.path.dirname(dst5))] if os.path.isdir(os.path.dirname(dst5)) else []
    check('no predecessor and a header that fails validation: refused, nothing placed, no temporary file left',
          'validation FAILED' in r5 and not os.path.exists(dst5) and not left, (r5[:200], left))
    # a predecessor holding coordinate cards the search's frame lacks
    w6, _ = work_frame('2024-12-23', 6.5)
    q6, dst6 = archived(os.path.join(ARCH, QRUN), '2024-12-23', 6.5,
                        extra={'EQUINOX': (2000.0, 'equinox'), 'SPECSYS': ('TOPOCENT', 'spectral reference frame')})
    with fits.open(w6, memmap=False) as h:
        lacks = 'EQUINOX' not in h[0].header and 'SPECSYS' not in h[0].header   # (WCS.to_header writes LONPOLE in some astropy versions)
    try:
        publish(w6)
        r6 = 'not refused'
    except RuntimeError as e:
        r6 = str(e)
    check('a predecessor holding coordinate cards (EQUINOX, SPECSYS) the search\'s frame lacks: refused, every card and the '
          'cell named, with REFUSING (the engine child refuses this (cell, Stokes) unit only), nothing placed',
          lacks and 'coordinate card(s) EQUINOX SPECSYS' in r6 and r6.startswith('03h 18MHz 2024-12-23 ') and 'REFUSING' in r6
          and not os.path.exists(dst6) and dst6 not in record_rows(), r6[:200])
    # fix round 2 (9): RADECSYS, the observer's place (OBSGEO-*) and the time family (DATE-OBS / MJD-OBS / MJDREF ...) are
    # never carried and never silently dropped
    w6b, _ = work_frame('2024-12-28', 6.75)
    new_only = {'RADECSYS': ('FK5', 'old name of RADESYS'), 'OBSGEO-X': (-2409261.7339, '[m]'),
                'OBSGEO-Y': (-4477916.6, '[m]'), 'OBSGEO-Z': (3839351.6, '[m]'), 'DATE-AVG': ('2024-12-28T05:00:00', ''),
                'MJD-AVG': (60672.208, ''), 'TIMEUNIT': ('s', '')}
    q6b, dst6b = archived(os.path.join(ARCH, QRUN), '2024-12-28', 6.75, extra=new_only)
    try:
        publish(w6b)
        r6b = 'not refused'
    except RuntimeError as e:
        r6b = str(e)
    fam = ('DATE-OBS', 'DATE-BEG', 'DATE-AVG', 'DATE-END', 'MJD-OBS', 'MJD-BEG', 'MJD-AVG', 'MJD-END', 'MJD-MID', 'MJDREF',
           'MJDREFI', 'MJDREFF', 'JDREF', 'DATEREF', 'TIMESYS', 'TREFPOS', 'TIMEUNIT', 'RADECSYS', 'RADESYS', 'OBSGEO-X',
           'OBSGEO-Y', 'OBSGEO-Z', 'OBSGEO-B', 'OBSGEO-L', 'OBSGEO-H')
    check('fix round 2 (9): a predecessor holding RADECSYS, OBSGEO-X/Y/Z and time cards (DATE-AVG, MJD-AVG, TIMEUNIT) the '
          'search\'s frame lacks: refused, each named, nothing placed; the refused set covers RADECSYS, OBSGEO-*, MJD-OBS and '
          'the DATE-OBS / MJDREF time family (and not a search card such as IMGRMS or MSTRMJD)',
          all(k in r6b for k in new_only) and 'REFUSING' in r6b and not os.path.exists(dst6b)
          and all(DSE._WCS_CARD.match(k) for k in fam) and not any(DSE._WCS_CARD.match(k) for k in ('IMGRMS', 'MSTRMJD',
                                                                                                    'DATE', 'MJDOBS')),
          (r6b[:300], [k for k in fam if not DSE._WCS_CARD.match(k)]))
    # a frame only in an older quarantine round: never a predecessor (the engine reads only the roots the deep step gives)
    w7, h7 = work_frame('2024-12-24', 7.5)
    archived(os.path.join(ARCH, 'Run_SEARCH_QUARANTINE_20261004'), '2024-12-24', 7.5,
             extra={'OLDROUND': ('yes', 'only on the older round\'s frame')})
    dst7 = publish(w7)
    a7 = hdr(dst7)
    check('a frame only in an older quarantine round (20261004) is never the predecessor (decision (6a)): the work frame\'s own '
          'header, validated; the record names no predecessor',
          'OLDROUND' not in a7 and not any(k in a7 for k in OLD_ONLY) and record_rows()[dst7]['predecessor'] == '', a7.get('OLDROUND'))
    # search-owned cards of the predecessor the search does not write; THERMNOI, the frame's noise card
    w8, h8 = work_frame('2024-12-26', 8.5, thermnoi=None)          # as the search's I frames: no THERMNOI of its own
    q8, dst8 = archived(os.path.join(ARCH, QRUN), '2024-12-26', 8.5,
                        extra={'LOOEXCL': ('2024-12-26', 'Epoch excluded from the LOO master')},
                        after={'PBFILE': ('PB_old.fits', 'PB map applied'), 'IMGRMS': (0.99, 'the earlier search\'s'),
                               'THERMNOI': (0.31, 'stamped on the archive copy')})
    publish(w8)
    a8 = hdr(dst8)
    hist8 = [str(x) for x in a8['HISTORY']] if 'HISTORY' in a8 else []
    row8 = record_rows().get(dst8, {})
    check('a predecessor\'s search-owned cards the search does not write (LOOEXCL, PBFILE) are dropped and named in a HISTORY '
          'line; its IMGRMS is the search\'s (0.21, not 0.99); THERMNOI - the frame\'s noise card (ruling (1) of 2026-10-16) - is '
          'carried (0.31) when the search writes none, as on its I frames; the record lists both (decision (6b))',
          'LOOEXCL' not in a8 and 'PBFILE' not in a8 and abs(float(a8['IMGRMS']) - 0.21) < 1e-9 and float(a8['THERMNOI']) == 0.31
          and 'THERMNOI' not in h8 and DSE.dropped_history('LOOEXCL') in hist8 and DSE.dropped_history('PBFILE') in hist8
          and len(DSE.dropped_history('MSTRFILE')) <= 72 and row8.get('dropped') == 'LOOEXCL PBFILE' and 'THERMNOI' in row8.get('carried', '').split()
          and row8.get('predecessor') == q8, (hist8, row8))
    r8 = FT.copy_identical(dst8, copy_of=w8, archive_record=REC)
    check('... and the content test passes it from the record (the superset check exempts the search-owned cards)', r8[0], r8)
    return w1, q1, dst1, w2, (w3, dst3), (w4, dst4), (w8, q8, dst8)


def _rewrite(src, dst, keep):
    with fits.open(src, memmap=False) as h:
        hd, d = h[0].header, np.array(h[0].data)
        out = fits.Header([c for c in hd.cards if keep(c) and not DSE._STRUCT_CARD.match(c.keyword)])
    fits.writeto(dst, d, out, overwrite=True)
    return dst


def test_content_test(w1, q1, dst1, w2, inplace, nopred, owned):
    plain = os.path.join(T, 'plain_copy.fits')
    shutil.copy(w1, plain)                       # what v2 archived before: the work frame as is
    r = FT.copy_identical(plain, copy_of=w1, predecessor=q1)
    r2 = FT.copy_identical(dst1, copy_of=w2)
    check('copy_identical refuses a copy lacking cards of its predecessor (the plain copy v2 made before) and a copy whose '
          'pixels differ; passes the merged frame', not r[0] and 'superseded' in r[2] and not r2[0] and 'pixels' in r2[2]
          and FT.copy_identical(dst1, copy_of=w1, predecessor=q1)[0], (r, r2))
    nocom = _rewrite(dst1, os.path.join(T, 'no_comment.fits'), lambda c: not (c.keyword == 'COMMENT' and str(c.value) == COMMENT))
    r3 = FT.copy_identical(nocom, copy_of=w1, predecessor=q1)
    w8, q8, dst8 = owned
    unnamed = _rewrite(dst8, os.path.join(T, 'unnamed_drop.fits'), lambda c: not (c.keyword == 'HISTORY' and 'dropped' in str(c.value)))
    r4 = FT.copy_identical(unnamed, copy_of=w8, predecessor=q8)
    check('copy_identical refuses a copy lacking the work frame\'s COMMENT line and a copy dropping a search-owned card of its '
          'predecessor without naming it in a HISTORY line', not r3[0] and 'COMMENT' in r3[2] and not r4[0]
          and 'without a HISTORY line' in r4[2], (r3, r4))
    rows = record_rows()
    (w3, dst3), (w4, dst4) = inplace, nopred
    check('the engine\'s record (P3_ARCHIVE_RECORD): one row per archive frame written, with the predecessor it merged (the '
          'quarantined frame; the frame in place, by its own path; none), that predecessor\'s cards, the carried cards',
          rows[dst1]['predecessor'] == q1 and set(rows[dst1]['predecessor_cards'].split()) == cardset(q1)
          and rows[dst1]['carried'] == 'ASTREF FLUXREF SCALE_A P1RUN P2RUN CALBP' and rows[dst1]['work_frame'] == w1
          and rows[dst3]['predecessor'] == dst3 and 'INPLACE' in rows[dst3]['predecessor_cards'].split()
          and rows[dst4]['predecessor'] == '', {k: rows[k] for k in (dst1, dst3, dst4)})
    r5 = FT.copy_identical(dst3, copy_of=w3, archive_record=REC)
    empty = os.path.join(T, 'empty_record.csv')
    open(empty, 'w').write(','.join(DSE.ARCHIVE_RECORD_COLS) + '\n')
    r6 = FT.copy_identical(dst1, copy_of=w1, archive_record=empty)
    r7 = FT.copy_identical(dst1, copy_of=w1, archive_record=REC, predecessor=os.path.join(ARCH, 'other.fits'))
    r8 = FT.copy_identical(dst4, copy_of=w4, archive_record=REC)
    check('the content test reads the record: a frame replaced in place passes on its recorded cards (the file is the new '
          'frame); a frame with no row fails; a predecessor other than the recorded one fails; no predecessor passes',
          r5[0] and 'as recorded' in r5[2] and not r6[0] and 'no row' in r6[2] and not r7[0] and 'recorded' in r7[2]
          and r8[0] and 'no superseded frame' in r8[2], (r5, r6, r7, r8))
    with fits.open(q1, mode='update', memmap=False) as h:          # the quarantined file changed after the merge
        h[0].header['LATECARD'] = (1, 'added after the merge')
    r9 = FT.copy_identical(dst1, copy_of=w1, archive_record=REC)
    check('... and a recorded card list that no longer matches the predecessor file fails', not r9[0] and 'differ' in r9[2], r9)


def _key(sl):
    sl = getattr(sl, 'value', sl) if type(sl).__name__ == 'Index' else sl         # py38 wraps a subscript in ast.Index
    import ast
    return sl.value if isinstance(sl, ast.Constant) and isinstance(sl.value, str) else None


def _writer_scan(src, present=None):
    """(literal cards, computed arguments) of a deep-search frame writer's source: every literal card name the archive frame
    writers (WRITER_FNS, those the version has) assign (h['X'] = ..., extra['X'] = ..., dict literals' keys), and the
    write_standard_header arguments whose value derives from nothing read from the epoch's Stage-1 product (hs: the names
    assigned from it, to a fixed point) - but the frame's identity (scope, stokes), extra (scanned above) and validate.
    ``present``: a list that receives the writer functions the source has."""
    import ast
    body = ast.parse(src).body
    fns = {n.name: n for n in body if isinstance(n, ast.FunctionDef)}
    lit, computed = set(), set()
    for name in WRITER_FNS:
        if name not in fns:
            continue
        if present is not None:
            present.append(name)
        for n in ast.walk(fns[name]):
            if isinstance(n, ast.Assign):
                lit.update(_key(t.slice) for t in n.targets if isinstance(t, ast.Subscript) and _key(t.slice))
            if isinstance(n, ast.Dict):
                lit.update(k.value for k in n.keys if isinstance(k, ast.Constant) and isinstance(k.value, str))
    f = fns.get('_archive_frame_header')
    if f is None:
        return lit, computed
    names = lambda node: {x.id for x in ast.walk(node) if isinstance(x, ast.Name)}            # noqa: E731
    taint = {'hs'}
    while True:
        more = set()
        for n in ast.walk(f):
            if isinstance(n, ast.Assign) and names(n.value) & taint:
                for t in n.targets:
                    more |= names(t)
        if more <= taint:
            break
        taint |= more
    for n in ast.walk(f):
        if isinstance(n, ast.Call) and getattr(n.func, 'attr', None) == 'write_standard_header':
            computed |= {kw.arg for kw in n.keywords
                         if kw.arg not in ('scope', 'stokes', 'extra', 'validate') and not names(kw.value) & taint}
    return lit, computed


def _git(repo, *a):
    import subprocess
    return subprocess.run(['git', '-C', repo] + list(a), capture_output=True, text=True, check=True).stdout


def _history(repo, path):
    """[(commit, date, path at that commit)] of a tracked file, newest first (git log --follow)."""
    revs, cur = [], None
    for ln in _git(repo, 'log', '--format=C %H %cs', '--follow', '--name-only', '--', path).splitlines():
        if ln.startswith('C '):
            cur = ln.split()[1:]
        elif ln.strip():
            revs.append((cur[0], cur[1], ln.strip()))
    return revs


def _writer_versions():
    """Every committed version of production's deep frame writer (WRITER_REPOS, WRITER_NAME), scanned: one row per (repo,
    file, commit) with the writer functions it has, its literal cards and its computed arguments."""
    rows = []
    for repo in WRITER_REPOS:
        for path in sorted(f for f in _git(repo, 'ls-files').splitlines() if os.path.basename(f).endswith(WRITER_NAME)):
            for h, d, at in _history(repo, path):
                fns = []
                lit, comp = _writer_scan(_git(repo, 'show', f'{h}:{at}'), present=fns)
                rows.append(dict(repo=repo, path=path, commit=h, date=d, at=at, writers=fns, literal=sorted(lit),
                                 computed=sorted(comp)))
    return rows


def _products_versions():
    """Every committed version of production's standard header writer (phase3_products.write_standard_header): the card
    names (string constants of FITS keyword form) in its body."""
    import ast
    import re
    card = re.compile(r'^[A-Z][A-Z0-9_-]{1,7}$')
    rel = os.path.relpath(PROD_PRODUCTS, PROD_REPO)
    out = []
    for h, d, at in _history(PROD_REPO, rel):
        fns = {n.name: n for n in ast.parse(_git(PROD_REPO, 'show', f'{h}:{at}')).body if isinstance(n, ast.FunctionDef)}
        w = fns.get('write_standard_header')
        out.append(dict(commit=h, date=d, cards=sorted({n.value for n in ast.walk(w) if isinstance(n, ast.Constant)
                                                        and isinstance(n.value, str) and card.match(n.value)}) if w else None))
    return out


def _derive(versions):
    """The search-owned set of scanned writer versions (and v2's writer): their literal cards and the standard header's cards
    of their computed arguments, less THERMNOI."""
    lit, comp = _writer_scan(inspect.getsource(DSE))
    for v in versions:
        lit |= set(v['literal'])
        comp |= set(v['computed'])
    return (lit | _std_cards(comp)) - {'THERMNOI'}, lit, comp


def write_fixture():
    versions = _writer_versions()
    owned, lit, comp = _derive(versions)
    fx = dict(about='SEARCH_OWNED_CARDS (lib/search/deep_search.py), derived by tests/test_search_archive_header.py '
                    '--write-fixture from every committed version of production\'s deep frame writer (sources) and v2\'s writer '
                    '(v2_writer); the standard header\'s cards of the computed arguments from core.products.write_standard_header '
                    '(= phase3_products.write_standard_header; products: its committed versions)',
              set=sorted(owned), less={'THERMNOI': 'ruling (1) of 2026-10-16: "THERMNOI stays the frame\'s noise card"'},
              literal=sorted(lit), computed=sorted(comp), v2_writer='exopipe/phase3/lib/search/deep_search.py',
              writer_functions=list(WRITER_FNS), sources=versions, products=_products_versions())
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    with open(FIXTURE, 'w') as fh:
        json.dump(fx, fh, indent=1, sort_keys=True)
        fh.write('\n')
    print(f'wrote {FIXTURE}: {len(owned)} cards from {len(versions)} writer versions')


def _std_cards(computed):
    """The standard header's cards of the computed arguments: those that vanish or change when one is left at its default."""
    sig = inspect.signature(prod.write_standard_header)
    base = dict(scope='18MHz', stokes='V', start='2024-12-18T04:30:00', end='2024-12-18T05:29:41', inttime_s=3581.0,
                time_basis='phase2', beam=(1.0, 0.8, 10.0), ebeam=(1.1, 1.1, 0.0), thermnoi=0.2, imgrms=0.21, pbcorr=False,
                dewarped=True, aligned=True, alignref='a.fits', msub=True, msubfile='m.fits', ssub=True, p2run='p2', calbp='c',
                p1run='p1', validate=False)

    def cards(kw):
        h = fits.Header()
        prod.write_standard_header(h, **kw)
        return {k: str(h[k]) for k in h if k not in ('COMMENT', 'HISTORY', '')}
    full, out = cards(base), set()
    for a in computed:
        less = cards(dict(base, **{a: sig.parameters[a].default}))
        out |= {k for k in full if less.get(k) != full[k]}
    return out


def _no_doc(fn):
    import ast
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant):
        fn.body = fn.body[1:]
    return ast.dump(fn)


def test_search_owned_set():
    """decision (6b), fix round 2 (8): the search-owned cards are derived from the writers' source (not a hand list) - every
    committed version of production's deep frame writer and v2's writer - checked in as tests/fixtures/search_owned_cards.json.
    Data-free (CI): the constant equals the fixture, v2's writer and the fixture's recorded scans give the fixture's set. With
    production's repositories here: the history is re-derived and equals the fixture, scan for scan."""
    import ast
    with open(FIXTURE) as fh:
        fx = json.load(fh)
    fset = set(fx['set'])
    lit, comp = _writer_scan(inspect.getsource(DSE))
    v2 = (lit | _std_cards(comp)) - {'THERMNOI'}
    rec, _, _ = _derive(fx['sources'])
    check('SEARCH_OWNED_CARDS equals the fixture tests/fixtures/search_owned_cards.json; v2\'s writer alone gives that set; the '
          'fixture\'s recorded scans of production\'s writer versions give it too (the frame writers\' literal cards and the '
          'standard header\'s cards of the arguments the frame writer computes itself, less THERMNOI - the frame\'s noise card, '
          'ruling (1) of 2026-10-16)',
          fset == DSE.SEARCH_OWNED_CARDS and v2 == fset and rec == fset and 'thermnoi' in comp
          and set(fx['less']) == {'THERMNOI'} and len(fx['sources']) > 0,
          (sorted(fset ^ DSE.SEARCH_OWNED_CARDS), sorted(v2 ^ fset), sorted(rec ^ fset)))
    if not all(os.path.isdir(os.path.join(r, '.git')) for r in WRITER_REPOS) or not os.path.isfile(PROD_PRODUCTS):
        print('UNTESTED the re-derivation from production\'s history (' + ', '.join(WRITER_REPOS) + ' absent here): the '
              'fixture\'s recorded scans only')
        return
    versions = _writer_versions()
    owned, _, _ = _derive(versions)
    key = lambda v: (v['repo'], v['path'], v['commit'])                       # noqa: E731
    have, want = {key(v): v for v in versions}, {key(v): v for v in fx['sources']}
    diff = sorted(set(have) ^ set(want)) + [k for k in have if k in want and have[k] != want[k]]
    w = [v for v in versions if v['writers']]
    extra = sorted(owned - DSE.SEARCH_OWNED_CARDS)
    print(f'  history: {len(versions)} committed writer versions ({len(w)} with an archive frame writer, '
          f'{min(v["date"] for v in w)} .. {max(v["date"] for v in w)}); cards beyond the constant: {extra or "none"}')
    check('the search-owned set re-derived from every committed version of production\'s deep frame writer (git log --follow '
          'and git show per commit of each *phase3_deep.py tracked in ' + ', '.join(WRITER_REPOS) + ') plus v2\'s writer equals '
          'the fixture, version for version (a new commit refreshes the fixture: --write-fixture)',
          owned == fset and not diff, (extra, sorted(fset - owned), diff[:5]))
    pv = _products_versions()
    newest = set(pv[0]['cards'] or ())
    with open(PROD_PRODUCTS) as fh:
        pf = next(n for n in ast.parse(fh.read()).body if isinstance(n, ast.FunctionDef) and n.name == 'write_standard_header')
    vf = next(n for n in ast.parse(inspect.getsource(prod)).body
              if isinstance(n, ast.FunctionDef) and n.name == 'write_standard_header')
    check('production\'s write_standard_header is v2\'s (the standard header\'s cards of the computed arguments); no committed '
          'version of it names a card the newest lacks; the fixture lists those versions',
          _no_doc(pf) == _no_doc(vf) and all(set(v['cards'] or ()) <= newest for v in pv)
          and [v['commit'] for v in pv] == [v['commit'] for v in fx['products']],
          [(v['commit'][:7], sorted(set(v['cards'] or ()) - newest)) for v in pv if not set(v['cards'] or ()) <= newest])


def test_case_a_does_not_apply():
    """Case (a) - leave the archived frame when the search's pixels equal it - holds only when nothing reads a search-stamped
    card from the archived copy. These readers exist, so case (b) is the one implemented, for every frame."""
    from exopipe.phase3.lib.search import removal_rules as RR
    so = inspect.getsource(DS.side_outputs)
    rms = inspect.getsource(FT.rms_cards)
    mu = inspect.getsource(FT.master_used)
    rr = inspect.getsource(RR)
    check('case (a) does not apply: the deep step registers each archive copy as a search_frame product, whose content tests '
          'read its IMGRMS / THERMNOI (rms_cards) and LOOMSTR / MSTRFILE (master_used); the removal rules read the archived '
          'frame\'s BMAJ (gate_inputs.deep_target_beam: the archive, then the newest quarantine round)',
          "'search_frame'" in so and "'copy_of'" in so and 'IMGRMS' in rms and 'THERMNOI' in rms and 'LOOMSTR' in mu
          and 'deep_target_beam' in rr)


def test_step():
    src = inspect.getsource(DSE._archive_frame_publish)
    check('the engine no longer copies the work frame as is: _archive_frame_publish writes the merged header '
          '(no placement.copy)', 'placement.copy(' not in src and '_archive_write_merged(' in src)
    mirror, parch = os.path.join(T, 'mirror'), os.path.join(T, 'prod_archive')
    for a, r in ((mirror, '20261127'), (parch, '20261004'), (parch, '20261008_retired_psfmaster'), (parch, '20261113')):
        os.makedirs(os.path.join(a, 'Run_SEARCH_QUARANTINE_' + r), exist_ok=True)
    open(os.path.join(parch, 'Run_SEARCH_QUARANTINE_20261201'), 'w').close()       # a file, not a round
    CFG.paths.PHASE3_ARCHIVE_ROOT = parch
    from exopipe.phase3.core import legacy
    real = legacy.interpreter
    legacy.interpreter = lambda name: sys.executable      # data-free: no site file names the engine's interpreters
    try:
        L = (1795737600.0, '20261127')                    # the run's launch (run_launch, fixed in ctx at the run's start)
        win = DS.archive_predecessor_roots(dict(archive_root=mirror, launch=L))
        live = DS.archive_predecessor_roots(dict(archive_root=parch, launch=L))
        env = DS.engine_env('/x', '/x', '/x', 'X', '/x/arch', '/x/tmp', pred_roots=win, archive_record='/x/a/archive_record.csv')
    finally:
        legacy.interpreter = real
        CFG.reset()
    q = lambda a, r: os.path.join(a, 'Run_SEARCH_QUARANTINE_' + r)                  # noqa: E731
    sd, sq = inspect.getsource(DS.stage_detect), inspect.getsource(DS.stage_quarantine)
    sp = inspect.getsource(DS.archive_predecessor_roots)
    check('the deep step\'s predecessor roots (decision (6a), fix round 2 (2)): the run\'s archive root (the frame in place), '
          'then this launch\'s own quarantine round only (never 20261004 .. 20261113), then - in the window\'s mirror only - the '
          'production archive in place (read only); the round from the run\'s fixed launch (run_launch), never from a stage\'s '
          'state (no f[\'quarantine_round\']); handed to the engine with the record\'s path (P3_ARCHIVE_RECORD, decision (7))',
          win == [mirror, q(mirror, '20261127'), parch] and live == [parch, q(parch, '20261127')]
          and env.get('P3_ARCHIVE_PREDECESSOR_ROOTS') == os.pathsep.join(win)
          and env.get('P3_ARCHIVE_RECORD') == '/x/a/archive_record.csv'
          and 'archive_predecessor_roots(ctx)' in sd and 'run_launch(ctx)' in sq and 'run_launch(ctx)' in sp
          and "quarantine_round']" not in sd + sq + sp, (win, live))


def test_separate_runs():
    """Fix round 2 (2): the predecessor roots take this launch's round from the run's fixed launch (the launch file, read once
    at the run's start), never from per-stage state. `--stages quarantine` in one run, then `--stages detect,...` in a separate
    run: the second run finds the frame the first moved into the launch's round, and its repair cards are carried. Data-free:
    the production branch of the quarantine stage on scratch roots (the archive, records and ops roots under the test
    directory; the membership precheck, which runs production's code, stubbed)."""
    import types
    from exopipe.phase3.lib.archive import quarantine as Q
    sa, srec, sops = (os.path.join(T, 'sep_' + x) for x in ('archive', 'records', 'ops'))
    for d in (sa, srec, sops):
        os.makedirs(d, exist_ok=True)
    date = '2024-12-27'
    real_pre = Q.membership_precheck
    keep_env = {k: os.environ.get(k) for k in ('P3_ARCHIVE_ROOT', 'P3_ARCHIVE_PREDECESSOR_ROOTS')}
    try:
        CFG.paths.PHASE3_ARCHIVE_ROOT, CFG.paths.ARCHIVE_RUN, CFG.paths.OPS_ROOT = sa, RUN, sops
        CFG.archive.RECORDS_ROOT = srec
        CFG.paths.PRODUCTION_WRITES = True                   # the production branch of stage_quarantine: scratch roots only
        CFG.search_run.PRODUCTION_WORK_ROOT = os.path.join(T, 'sep_work')
        prev, _ = archived(sa, date, 1.5)                     # production's archive frame, from before the launch
        os.utime(prev, (time.time() - 86400, time.time() - 86400))
        launch = round(time.time(), 3)
        Q.begin_launch(_Unit(), epoch=repr(launch), apply=True)
        rnd = Q.round_for(launch, 'launch-date')
        Q.membership_precheck = lambda cell, unit=None, log=None: (True, 'stubbed (data-free test)')
        item = types.SimpleNamespace(cell='03h_18MHz', lst='03h', scope='18MHz', stokes='V', name='03h_18MHz_V', mode='fresh')
        # run 1: --stages quarantine
        ctx1 = dict(out=os.path.join(T, 'sep_out1'), t0=time.time(), archive_root=sa, production=True)
        u1 = _Unit()
        DS.stage_quarantine(u1, item, ctx1, {})
        moved = os.path.join(sa, f'Run_SEARCH_QUARANTINE_{rnd}', os.path.relpath(prev, sa))
        moved_ok = os.path.exists(moved) and not os.path.exists(prev)
        # run 2: --stages detect,... - a new run, nothing carried over from run 1 but the launch file
        ctx2 = dict(out=os.path.join(T, 'sep_out2'), t0=time.time() + 5, archive_root=sa, production=True)
        roots = DS.archive_predecessor_roots(ctx2)
        os.environ.update(P3_ARCHIVE_ROOT=sa, P3_ARCHIVE_PREDECESSOR_ROOTS=os.pathsep.join(roots))
        w, _ = work_frame(date, 1.5)
        dst = os.path.join(sa, os.path.relpath(prev, sa))
        found = DSE._archive_predecessor(dst)
        out = publish(w)
        a = hdr(dst)
    finally:
        Q.membership_precheck = real_pre
        for k, v in keep_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        CFG.reset()
    hist = ' '.join(str(x) for x in a['HISTORY']) if 'HISTORY' in a else ''
    check('a separate run (--stages quarantine, then --stages detect in a new run): the second run\'s predecessor roots name '
          'the launch\'s round from the launch file; the frame the first run moved there is found and its repair cards (ASTREF '
          'FLUXREF SCALE_A P1RUN P2RUN CALBP) are carried onto the archive frame, named in a HISTORY line',
          moved_ok and ctx2['launch'] == (launch, rnd)
          and roots == [sa, os.path.join(sa, f'Run_SEARCH_QUARANTINE_{rnd}')] and found == moved and out == dst
          and all(a.get(k) == v for k, v in OLD_ONLY.items()) and 'carried from the superseded archive frame' in hist,
          (u1.notes[-2:], roots, found, {k: a.get(k) for k in OLD_ONLY}))


class _Unit:
    def __init__(self):
        self.meta, self.notes, self.outputs = {}, [], []

    def check(self, *a, **k):
        pass

    def note(self, m):
        self.notes.append(m)

    def add_output(self, p, cls, **k):
        self.outputs.append((p, cls))


def test_registration():
    """The deep step registers each archive frame its detect wrote for the content tests (side_outputs), found by the attempt
    frame's PRODNAME. A deep PRODNAME is longer than one card holds: astropy writes it over CONTINUE cards, and the header-only
    reader returned the first card's text ending in '&' - no archive frame was ever found, so the content tests (copy_identical
    with the predecessor's cards) never ran on one (the row-64 parity runs: 0 of 288 archive frames registered)."""
    import time
    from exopipe.phase3.core.record import fits_primary_cards
    from types import SimpleNamespace
    att, mirror, pred = os.path.join(T, 'reg_attempt'), os.path.join(T, 'reg_mirror'), os.path.join(T, 'reg_prod')
    date = '2024-12-21'
    h = header(date, 0.21, 'e1a4e65')
    wd = os.path.join(att, 'Run_T', 'vresid', '03h_18MHz')
    os.makedirs(wd, exist_ok=True)
    work = os.path.join(wd, f'{date}_V_redo_resid_18MHz.fits')
    fits.writeto(work, np.zeros((16, 16), np.float32), h, overwrite=True)
    name = h['PRODNAME']
    raw, whole = fits_primary_cards(work).get('PRODNAME'), fits_primary_cards(work, continued=True).get('PRODNAME')
    check('a deep PRODNAME spans CONTINUE cards: the header-only reader returns it whole with continued=True (as astropy reads '
          'it) and, unchanged for every other caller, the first card\'s text ending in \'&\' without',
          len(name) > 68 and whole == name and str(raw).endswith('&') and raw != name, (name, raw, whole))
    t0 = time.time() - 60
    rel = os.path.join('03h', date, DS._cfg().paths.ARCHIVE_RUN, '18MHz', 'V', 'deep', name)
    for root in (mirror, pred):
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        shutil.copyfile(work, os.path.join(root, rel))
    item = SimpleNamespace(cell='03h_18MHz', lst='03h', scope='18MHz', stokes='V', mode='full')
    ctx = dict(flow_root=os.path.join(T, 'reg_flow'), out=os.path.join(T, 'reg_out'), archive_root=mirror)
    rec = os.path.join(att, DS.ARCHIVE_RECORD)
    f = dict(attempt=att, label='T', t_detect=t0, archive_pred_roots=[mirror, pred], archive_record=rec)
    got = [(dst, cls, tc) for _p, cls, tc, dst in DS.side_outputs(item, ctx, f) if cls == 'search_frame']
    check('the deep step registers the archive frame its detect wrote (found by the whole PRODNAME) for the content tests, '
          'with the attempt frame it copies and the engine\'s record of the frame it superseded (never a second search of '
          'the roots, decision (7))',
          got == [(os.path.join(mirror, rel), 'search_frame', {'stokes': 'V', 'copy_of': work, 'archive_record': rec})], got)


def main():
    if '--write-fixture' in sys.argv[1:]:
        write_fixture()
        return 0
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    keep = {k: os.environ.get(k) for k in ('P3_ARCHIVE_ROOT', 'P3_ARCHIVE_RUN', 'P3_ARCHIVE_PREDECESSOR_ROOTS', 'P3_ARCHIVE_WRITE',
                                           'P3_ARCHIVE_RECORD')}
    try:
        w1, q1, dst1, w2, inplace, nopred, owned = test_branches()
        test_content_test(w1, q1, dst1, w2, inplace, nopred, owned)
        test_search_owned_set()
        test_case_a_does_not_apply()
        test_step()
        test_separate_runs()
        test_registration()
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if not os.environ.get('EXOPIPE3_TEST_KEEP'):
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
