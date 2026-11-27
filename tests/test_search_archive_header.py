"""The deep search's archive frames never carry fewer standard cards than the frames they supersede (the refactor chat's
decision (3) of 2026-11-27, case (b): archived search frames' cards are read - the BMAJ gate through gate_inputs.deep_target_beam,
the archive push / audit content tests, the pages' cutouts - so every archive frame the search writes is the superseded frame's
full header updated with the search's own cards, written with its header validated; a frame with no predecessor validates or
refuses).

Checks, data-free (a scratch archive root; frames 16 x 16 px with a standard header built by core.products):
  quarantined   the predecessor in this round's quarantine Run (the pre-search quarantine moved it), equal pixels: the archive
                frame is its header updated with the search's cards - the cards only it had (ASTREF, FLUXREF, SCALE_A, P1RUN,
                P2RUN, CALBP, THERMNOI) carried with its values, the search's cards with the search's values, the work frame's
                pixels bit for bit, a HISTORY line naming the carried cards; the content test passes
  differ        a predecessor with other pixels: the same header rule, the search's pixels
  in place      a frame already at the destination (written earlier in this round) is the predecessor before the quarantine Run
  none          no predecessor: the work frame's own header, written when it validates; refused (nothing placed, no temporary
                file) when it does not
  wcs           a predecessor holding a coordinate card the search's frame lacks: refused, the destination as it was
  test          the content test (search_frame copy_identical) refuses a copy lacking a card of its predecessor (the plain copy
                v2 made before) and a copy whose pixels differ
  case (a)      not taken: the readers of archived copies' cards (the deep step's content tests, the removal rules' beam)
  step          the deep step hands the engine the predecessor roots: the archive root, its quarantine rounds newest first, and
                in the window the production archive and its rounds; the engine no longer copies the work frame as is
  register      the deep step registers each archive frame for the content tests: the attempt frame's PRODNAME (over CONTINUE
                cards) read whole, the archive frame found, with the attempt frame and the predecessor
Run: python tests/test_search_archive_header.py
"""
import inspect
import os
import shutil
import sys

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


def work_frame(date, value, **k):
    h = header(date, 0.21, 'e1a4e65', **k)
    p = os.path.join(WORK, f'{date}_V_redo_resid_18MHz.fits')
    os.makedirs(WORK, exist_ok=True)
    d = np.full((16, 16), value, np.float32)
    d[3, 4] = np.nan
    fits.writeto(p, d, h, overwrite=True)
    return p, h


def archived(where, date, value, extra=None):
    """The frame production archived: the same standard header with the cards header_repair stamped (OLD_ONLY), the earlier
    search's IMGRMS and GITTAG."""
    h = header(date, 0.21, '634c659', extra=extra)
    for k, v in OLD_ONLY.items():
        h[k] = (v, 'stamped on the archive copy')
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


def test_branches():
    os.environ.update(P3_ARCHIVE_ROOT=ARCH, P3_ARCHIVE_RUN=RUN,
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
    r = validate.run('search_frame', dst1, copy_of=w1, predecessor=q1, stokes='V')
    want = ('search_frame.header_full', 'search_frame.copy_identical', 'search_frame.prodname')
    got = {x['name']: x['passed'] for x in r}
    check('the archive frame passes the search_frame content tests that read its header and its copy (header_full, '
          'copy_identical with the predecessor, prodname)', all(got.get(k) for k in want), got)
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
    check('a frame already at the destination (written earlier in the round) is the predecessor, before the quarantine Run',
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
    # a predecessor holding a coordinate card the search's frame lacks
    w6, _ = work_frame('2024-12-23', 6.5)
    q6, dst6 = archived(os.path.join(ARCH, QRUN), '2024-12-23', 6.5, extra={'EQUINOX': (2000.0, 'equinox')})
    with fits.open(w6, memmap=False) as h:
        lacks = 'EQUINOX' not in h[0].header
    try:
        publish(w6)
        r6 = 'not refused'
    except RuntimeError as e:
        r6 = str(e)
    check('a predecessor holding a coordinate card (EQUINOX) the search\'s frame lacks: refused, nothing placed',
          lacks and 'coordinate card EQUINOX' in r6 and not os.path.exists(dst6), r6[:200])
    return w1, q1, dst1, w2


def test_content_test(w1, q1, dst1, w2):
    plain = os.path.join(T, 'plain_copy.fits')
    shutil.copy(w1, plain)                       # what v2 archived before: the work frame as is
    r = FT.copy_identical(plain, copy_of=w1, predecessor=q1)
    r2 = FT.copy_identical(dst1, copy_of=w2)
    check('copy_identical refuses a copy lacking cards of its predecessor (the plain copy v2 made before) and a copy whose '
          'pixels differ; passes the merged frame', not r[0] and 'superseded' in r[2] and not r2[0] and 'pixels' in r2[2]
          and FT.copy_identical(dst1, copy_of=w1, predecessor=q1)[0], (r, r2))


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
        win = DS.archive_predecessor_roots(dict(archive_root=mirror))
        live = DS.archive_predecessor_roots(dict(archive_root=parch))
        env = DS.engine_env('/x', '/x', '/x', 'X', '/x/arch', '/x/tmp', pred_roots=win)
    finally:
        legacy.interpreter = real
        CFG.reset()
    q = lambda a, r: os.path.join(a, 'Run_SEARCH_QUARANTINE_' + r)                  # noqa: E731
    check('the deep step\'s predecessor roots: the run\'s archive root, then its quarantine rounds newest first (this launch\'s '
          'round first; retired PSF-master rounds and non-directories left out), then - in the window\'s mirror only - the '
          'production archive and its rounds (read only); handed to the engine',
          win == [mirror, q(mirror, '20261127'), parch, q(parch, '20261113'), q(parch, '20261004')]
          and live == [parch, q(parch, '20261113'), q(parch, '20261004')]
          and env.get('P3_ARCHIVE_PREDECESSOR_ROOTS') == os.pathsep.join(win), (win, live))


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
    f = dict(attempt=att, label='T', t_detect=t0, archive_pred_roots=[mirror, pred])
    got = [(dst, cls, tc) for _p, cls, tc, dst in DS.side_outputs(item, ctx, f) if cls == 'search_frame']
    check('the deep step registers the archive frame its detect wrote (found by the whole PRODNAME) for the content tests, '
          'with the attempt frame it copies and the frame it superseded',
          got == [(os.path.join(mirror, rel), 'search_frame', {'stokes': 'V', 'copy_of': work,
                                                               'predecessor': os.path.join(pred, rel)})], got)


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    keep = {k: os.environ.get(k) for k in ('P3_ARCHIVE_ROOT', 'P3_ARCHIVE_RUN', 'P3_ARCHIVE_PREDECESSOR_ROOTS', 'P3_ARCHIVE_WRITE')}
    try:
        w1, q1, dst1, w2 = test_branches()
        test_content_test(w1, q1, dst1, w2)
        test_case_a_does_not_apply()
        test_step()
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
