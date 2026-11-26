#!/usr/bin/env python3
"""Data-free test (CI): the dewarp step's MDWSTAT card on the leave-one-out (LOO) masters, gate G-R5.

Production's Release 1 header repair wrote MDWSTAT on every archived LOO master with its cell's value - the value of the
cell's central masters - when the LOO master is its parent's and the value agrees with its MDWARP card
(exopipe_ops/header_repair_release1.py:1077-1090, mdw_cards :830-863; DONE-20261114-HEADER-REPAIR-LATE-DEEP-CLASSES-449-
CELLS). The v2 dewarp step gives the value by the same rule, with the LOO-to-parent link of lib/masters/loo_link - the one
rule the masters' T3 and the search's D-G16 call too (lib/dewarp/steps_common.loo_files, loo_problems; steps_cell.record,
stamp_mdwstat, applied_mdwstat, loo_source_gate, loo_gate). Checked here on small synthetic cells in a temporary directory
(TMPDIR), nothing else read:
  1. loo_files: the files of coadd_final/loo that coadd._LOO_RE names, sorted; none without loo/
  2. loo_problems: a clean cell passes; a LOO master without PARENTDG is linked by COADDOP (the search's rule); a foreign
     parent digest, a wrong NCOADD, a wrong LOOEXCL, a missing parent master, an MDWARP disagreeing with the value (or, on
     an apply's source, with the parent's), a subband combination (LSWBSB without MDWARP) and a name that is not a LOO
     master each give one problem naming the file
  3. the masters' count rule (loo_link.expected, the want of lib/masters/tests.py loo_master.count_per_stokes; the range
     masters_build.LOO_N_RANGE from the config): one LOO master too few, one too many, LOO masters of a parent outside
     the range, a parent in the range with none, a coadd_final without loo/ - each refuses; a parent outside the range
     with none passes; a parent without NCOADD refuses; the record stage refuses before any write; the PASS detail counts
     what was checked
  4. one LOO-to-parent rule (loo_link): on every combination of the cards, T3's per-LOO verdict and the D-G16 preview's
     verdict are those of their code before the helper (kept verbatim here), and whole T3 runs give the same result; G-R5's
     digest verdict is the search's on every combination
  5. the record stage of a not-accepted cell stamps its masters AND its LOO masters with one value, header valid, data
     bytes unchanged; meta.mdwstat keeps the masters (the install's currency set) and the LOO masters (loo_files) apart;
     each is an mdwstat_master output; then the applied=0 row
  6. no in-place rewrite: every file is staged to a temporary sibling and verified, then all are placed by ONE
     core.placement.replace_many call (masters and LOO masters together); a reader holding a file keeps the old inode; a
     failure staging the k-th LOO master, or a staged copy failing its check, leaves every original byte-identical and no
     temporary file behind
  7. G-R5 refusal: one LOO master of another master refuses the record stage before any write - no file carries the card,
     every file keeps its md5, no applied=0 row
  8. G-R5 on an accepted cell's apply source (loo_source_gate): a stale LOO master refuses before the resample - the apply
     is never called, no generation on disk
  9. the apply's record of an accepted cell: its LOO masters must carry 'applied' (recorded in loo_files); a LOO master
     without the card refuses 'dewarp.mdwstat', one without MDWARP refuses G-R5
 10. the install's currency rule (lib/install/mdwstat) covers meta.mdwstat.loo_files as the masters: a LOO master of other
     bytes or card, one missing, one not stamped, a record without loo_files, a size change and an installed LOO master's
     card each refuse
 11. the card comment is the archive's ('Master dewarp status', release1_cards.COMMENTS): the stamped 80-character card
     equals the one the Release 1 header repair writes

Run standalone:  python tests/test_dewarp_loo_mdwstat.py
"""
import glob
import hashlib
import os
import shutil
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

import numpy as np                                            # noqa: E402
from astropy.io import fits                                   # noqa: E402

from exopipe.phase3.config import CFG                         # noqa: E402
from exopipe.phase3.core import record as R                   # noqa: E402
from exopipe.phase3.lib.dewarp import steps_cell as SCL       # noqa: E402
from exopipe.phase3.lib.dewarp import steps_common as SC      # noqa: E402
from exopipe.phase3.lib.dewarp import tests as _T             # noqa: E402,F401  (registers class mdwstat_master)

N = 32
LST, SCOPE = '05h', '32MHz'
CELL = f'{LST}_{SCOPE}'
DIG = hashlib.md5(b'constituents 2025-01-01 2025-01-02 2025-01-03').hexdigest()
NIGHTS = ('2025-01-01', '2025-01-02')
ROW = {'lst': LST, 'scope': SCOPE, 'status': 'REFERENCE ONLY: held-out improvement 3.0 arcsec < 5; not applied',
       'anchor_off_med_as': '12.0', 'cv_resid_med_as': '9.0', 'med_before': '2.0', 'med_after': '1.5', 'n_anchors': '40',
       'measured_at': '2026-10-13T00:00:00Z'}
WANT = 'refused: held-out gain 3.0 arcsec < 5'
RESULTS = []


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + ('' if cond else f'  [{detail}]'), flush=True)


def header(**cards):
    """The standard header (validate_header level 'standard': beam, DATE-OBS, DURATION, INTTIME, a frequency)."""
    h = fits.Header()
    for k, v in (('CTYPE1', 'RA---SIN'), ('CTYPE2', 'DEC--SIN'), ('CRVAL1', 180.0), ('CRVAL2', 40.0), ('CRPIX1', N / 2 + 1),
                 ('CRPIX2', N / 2 + 1), ('CDELT1', -0.25), ('CDELT2', 0.25), ('CUNIT1', 'deg'), ('CUNIT2', 'deg'),
                 ('BMAJ', 1.0), ('BMIN', 1.0), ('BPA', 0.0), ('FLUXBAKE', True), ('FLUXVER', 'fluxcorr-v2-20261001-test'),
                 ('SHSCALE', 1.0), ('DATE-OBS', '2025-01-01T00:00:00'), ('DURATION', 3600.0), ('INTTIME', 3600.0)):
        h[k] = v
    h['RESTFRQ'] = float(CFG.instrument.SUBBAND_FREQ_HZ.get(SCOPE, {}).get('center', 3.2e7))   # Hz
    for k, v in cards.items():
        h[k] = v
    return h


def write(path, seed, **cards):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.random.default_rng(seed).normal(0, 1, (N, N)).astype(np.float32)
    fits.writeto(path, data, header(**cards), overwrite=True)


def make_cell(base, mdwarp=False, token=''):
    """<base>/<CELL>/coadd_final: three masters, a PB map, and per night an I and a V LOO master; loo/ also holds a file
    that is not a LOO master. ``token``: the generation's file-name token ('_mdewarp' in a master-dewarp generation)."""
    fd = os.path.join(base, CELL, 'coadd_final')
    extra = {'MDWARP': True} if mdwarp else {}
    for i, n in enumerate(('I_deep_Taper_R075', 'I_deep_Taper_R075_pbcorr', 'V_deep_Taper_R0')):
        write(os.path.join(fd, f'{n}{token}_{SCOPE}.fits'), i, CONSTDIG=DIG, NCOADD=len(NIGHTS), **extra)
    write(os.path.join(fd, f'PB_deep{token}_{SCOPE}.fits'), 9)
    for j, d in enumerate(NIGHTS):
        for st, tw in (('I', 'Taper_R075'), ('V', 'Taper_R0')):
            write(os.path.join(fd, 'loo', f'{d}_{st}_loo_{tw}{token}_{SCOPE}.fits'), 10 + j, CONSTDIG=DIG, PARENTDG=DIG,
                  LOOEXCL=d, NCOADD=len(NIGHTS) - 1, **extra)
    open(os.path.join(fd, 'loo', 'README.txt'), 'w').write('not a LOO master\n')
    return fd


def data_md5(path):
    """md5 of the data section: the bytes after the primary header's END block (header-only change => same md5)."""
    with open(path, 'rb') as f:
        while True:
            blk = f.read(2880)
            if not blk:
                return None
            if any(blk[i:i + 80].startswith(b'END ') or blk[i:i + 80] == b'END' + b' ' * 77 for i in range(0, 2880, 80)):
                break
        return hashlib.md5(f.read()).hexdigest()


def setcards(path, **cards):
    with fits.open(path, mode='update', memmap=False) as hl:
        for k, v in cards.items():
            if v is None:
                del hl[0].header[k]
            else:
                hl[0].header[k] = v


def seed_record(out):
    """The screen record the record stage replaces its row in: the cell routed (applied=1), as a re-rooted run's copy."""
    from exopipe.phase3.lib.install import install as INS
    sdir = INS._screen_dir(os.path.join(out, 'archive'), CFG.paths.ARCHIVE_RUN)
    INS._screen_row(sdir, dict(lst=LST, scope=SCOPE, applied=1, status='OK', n_anchors=40, vlssr_med_before_arcmin='2.0',
                               vlssr_med_after_arcmin='1.0', heldout_resid_med_arcsec='5.0', anchor_off_med_arcsec='20.0',
                               screen_p50_arcsec='1.0', screen_p95_arcsec='2.0', screen_file='', screen_md5='',
                               measured_at='2026-10-01T00:00:00Z', written_utc='2026-10-01T00:00:00Z'))
    return os.path.join(sdir, 'mdewarp_screens.csv')


def roots_for(out, rebuild):
    roots = SC.Roots(types.SimpleNamespace(out=out, work=None), types.SimpleNamespace(out_root=out, run_id='t'))
    roots.rebuild = rebuild
    return roots


def test_loo_files(tmp):
    fd = make_cell(os.path.join(tmp, 'files'))
    got = [os.path.basename(p) for p in SC.loo_files(fd)]
    want = sorted(f'{d}_{st}_loo_{tw}_{SCOPE}.fits' for d in NIGHTS for st, tw in (('I', 'Taper_R075'), ('V', 'Taper_R0')))
    check('loo_files: the four LOO masters of coadd_final/loo, sorted; not README.txt, not a master', got == want, str(got))
    check('loo_files: none for a coadd_final without loo/', SC.loo_files(os.path.join(tmp, 'files', 'nowhere')) == [])
    check('mdwstat_files (the masters, the install\'s currency set) still lists no LOO master',
          not any('/loo/' in p for p in SC.mdwstat_files(types.SimpleNamespace(rebuild=os.path.join(tmp, 'files')), CELL)))


def test_rule(tmp):
    base = os.path.join(tmp, 'rule')
    fd = make_cell(base)
    loo = SC.loo_files(fd)
    i1 = [p for p in loo if '_I_loo_' in p][0]
    v2 = [p for p in loo if '_V_loo_' in p][-1]
    def P(value=WANT, files=None):
        return SC.loo_problems(fd, value, files)[0]
    p, chk = SC.loo_problems(fd, WANT)
    check('G-R5 rule: a clean cell - every LOO master its parent\'s, no MDWARP, a refused value - no problem; the PASS '
          'detail counts the LOO masters checked (2 I, 2 V) and each parent of the count rule',
          p == [] and chk.startswith('4 leave-one-out master(s) checked (2 I, 2 V) against 2 parent(s)')
          and f'I_deep_Taper_R075_{SCOPE}.fits NCOADD 2: 2 of 2 present' in chk
          and f'V_deep_Taper_R0_{SCOPE}.fits NCOADD 2: 2 of 2 present' in chk, f'{p} {chk}')
    setcards(i1, PARENTDG=None)
    check('G-R5 rule: a build before PARENTDG with its parent\'s COADDOP is linked (the search\'s rule, D-G16)', P() == [],
          str(P()))
    setcards(i1, COADDOP='median')
    p = P()
    check('G-R5 rule: a build before PARENTDG with another COADDOP than its parent\'s - one problem (the search refuses it)',
          len(p) == 1 and 'COADDOP' in p[0] and os.path.basename(i1) in p[0], str(p))
    setcards(i1, COADDOP=None, PARENTDG='0' * 32)
    p = P()
    check('G-R5 rule: a foreign parent digest - one problem naming the file and both digests',
          len(p) == 1 and os.path.basename(i1) in p[0] and '0' * 32 in p[0] and DIG in p[0], str(p))
    setcards(i1, PARENTDG=DIG, NCOADD=len(NIGHTS))
    p = P()
    check('G-R5 rule: NCOADD other than the parent\'s - 1 - one problem (T3\'s rule)',
          len(p) == 1 and 'NCOADD' in p[0] and os.path.basename(i1) in p[0], str(p))
    setcards(i1, NCOADD=len(NIGHTS) - 1, LOOEXCL='2025-01-03')
    p = P()
    check('G-R5 rule: LOOEXCL other than the night the name excludes - one problem',
          len(p) == 1 and 'LOOEXCL' in p[0] and os.path.basename(i1) in p[0], str(p))
    setcards(i1, LOOEXCL=NIGHTS[0])
    vm = os.path.join(fd, f'V_deep_Taper_R0_{SCOPE}.fits')
    keep = vm + '.keep'
    os.rename(vm, keep)
    p = P()
    os.rename(keep, vm)
    check('G-R5 rule: a missing parent master - one problem per LOO master of it (the two V ones)',
          len(p) == 2 and all('V_deep_Taper_R0' in x and 'is not in' in x for x in p), str(p))
    setcards(v2, MDWARP=True)
    p = P()
    check("G-R5 rule: a LOO master carrying MDWARP T cannot take a 'refused' value - one problem",
          len(p) == 1 and 'MDWARP' in p[0] and os.path.basename(v2) in p[0], str(p))
    p = P('applied')
    check("G-R5 rule: 'applied' needs MDWARP T on every LOO master - the three without it",
          len(p) == 3 and os.path.basename(v2) not in ' '.join(p), str(p))
    p = P(None)
    check("G-R5 rule on an apply's source (no value): a LOO master whose MDWARP is not its parent's - one problem",
          len(p) == 1 and 'MDWARP' in p[0] and os.path.basename(v2) in p[0] and "parent's" in p[0], str(p))
    setcards(v2, MDWARP=None, LSWBSB='18MHz,23MHz')
    p = P()
    check('G-R5 rule: a combination of subband products (LSWBSB without MDWARP) takes no master\'s value - one problem',
          len(p) == 1 and 'LSWBSB' in p[0], str(p))
    setcards(v2, LSWBSB=None)
    p = P(WANT, loo + [os.path.join(fd, 'loo', 'README.txt')])
    check('G-R5 rule: a file that is not a LOO master name - one problem', len(p) == 1 and 'README.txt' in p[0], str(p))


def test_need(tmp):
    """The masters' own count rule (lib/masters/tests.py loo_master.count_per_stokes, via loo_link.expected): a parent of
    NCOADD n in masters_build.LOO_N_RANGE has exactly n LOO masters, any other parent none."""
    from exopipe.phase3.lib.masters import loo_link as LL
    lo, hi = CFG.masters_build.LOO_N_RANGE
    check('count rule: the range is the masters config\'s (masters_build.LOO_N_RANGE), read by loo_link',
          LL.n_range() == (lo, hi) and LL.needed(lo) and LL.needed(hi) and not LL.needed(lo - 1) and not LL.needed(hi + 1),
          str((lo, hi)))
    check('count rule: loo_link.expected is NCOADD in the range, else 0 (count_per_stokes\'s want)',
          [LL.expected(n) for n in (lo - 1, lo, hi, hi + 1)] == [0, lo, hi, 0])
    import inspect
    from exopipe.phase3.lib.masters import tests as MT
    check('count rule: the masters\' count test (loo_master.count_per_stokes) takes its want from loo_link.expected',
          'LL.expected(n)' in inspect.getsource(MT.count_per_stokes))
    p0, chk0 = SC.loo_problems(make_cell(os.path.join(tmp, 'need_ok')), WANT)
    check(f'count rule: {len(NIGHTS)} LOO masters per parent of NCOADD {len(NIGHTS)} pass, the detail counting them',
          p0 == [] and f'I_deep_Taper_R075_{SCOPE}.fits NCOADD {len(NIGHTS)}: {len(NIGHTS)} of {len(NIGHTS)} present'
          in chk0, f'{p0} {chk0}')
    # one too few: the V parent (NCOADD 2) keeps 1 of its 2
    fd = make_cell(os.path.join(tmp, 'need_few'))
    os.remove([p for p in SC.loo_files(fd) if '_V_loo_' in p][0])
    p = SC.loo_problems(fd, WANT)[0]
    check('count rule: ONE LOO master too few (V parent NCOADD 2, 1 present) refuses, naming the parent, the want and the count',
          len(p) == 1 and p[0].startswith(f'V_deep_Taper_R0_{SCOPE}.fits: NCOADD 2 is in masters_build.LOO_N_RANGE')
          and 'wants 2 leave-one-out master(s)' in p[0] and p[0].endswith('holds 1'), str(p))
    # one too many: a third I LOO master, linked correctly (its own night as LOOEXCL), beside a parent of NCOADD 2
    fd = make_cell(os.path.join(tmp, 'need_many'))
    src = [p for p in SC.loo_files(fd) if '_I_loo_' in p][0]
    extra = os.path.join(os.path.dirname(src), f'2025-01-03_I_loo_Taper_R075_{SCOPE}.fits')
    shutil.copy2(src, extra)
    setcards(extra, LOOEXCL='2025-01-03')
    p = SC.loo_problems(fd, WANT)[0]
    check('count rule: ONE LOO master too many (I parent NCOADD 2, 3 present, each linked) refuses',
          len(p) == 1 and p[0].startswith(f'I_deep_Taper_R075_{SCOPE}.fits: NCOADD 2') and p[0].endswith('holds 3'), str(p))
    # a parent outside the range keeps no LOO master: its LOO masters (linked: NCOADD = the parent's - 1) refuse
    fd = make_cell(os.path.join(tmp, 'need_out'))
    for m in SC.master_files(fd):
        setcards(m, NCOADD=hi + 1)
    for q in SC.loo_files(fd):
        setcards(q, NCOADD=hi)
    p = SC.loo_problems(fd, WANT)[0]
    check(f'count rule: parents above the range (NCOADD {hi + 1}) with {len(NIGHTS)} LOO masters each refuse (want 0)',
          len(p) == 2 and all('is outside masters_build.LOO_N_RANGE' in x and 'wants 0' in x for x in p), str(p))
    # none present: one parent, then both (no loo/)
    fd = make_cell(os.path.join(tmp, 'need'))
    vl = [p for p in SC.loo_files(fd) if '_V_loo_' in p]
    for p in vl:
        os.remove(p)
    p, chk = SC.loo_problems(fd, WANT)
    check('count rule: a parent in the range (V NCOADD 2) with no LOO master refuses, naming the parent and the range',
          len(p) == 1 and f'V_deep_Taper_R0_{SCOPE}.fits' in p[0] and 'LOO_N_RANGE' in p[0] and 'holds 0 (none)' in p[0], str(p))
    shutil.rmtree(os.path.join(fd, 'loo'))
    p = SC.loo_problems(fd, WANT)[0]
    check('count rule: a coadd_final without loo/ refuses for both parents (I and V)', len(p) == 2, str(p))
    for m in SC.master_files(fd):
        setcards(m, NCOADD=lo - 1)
    p, chk = SC.loo_problems(fd, WANT)
    check(f'count rule: parents below the range (NCOADD {lo - 1}) with no LOO master pass, the detail says so',
          p == [] and '0 of 0 present' in chk and chk.startswith('0 leave-one-out master(s) checked'), f'{p} {chk}')
    for m in SC.master_files(fd):
        setcards(m, NCOADD=hi + 1)
    check(f'count rule: parents above the range (NCOADD {hi + 1}) with no LOO master pass', SC.loo_problems(fd, WANT)[0] == [])
    setcards(os.path.join(fd, f'I_deep_Taper_R075_{SCOPE}.fits'), NCOADD=None)
    p = SC.loo_problems(fd, WANT)[0]
    check('count rule: a parent without NCOADD refuses (its count cannot be decided)',
          len(p) == 1 and 'no integer NCOADD' in p[0], str(p))
    # the record stage refuses on the count rule before any write
    out = os.path.join(tmp, 'out_need')
    csvp = seed_record(out)
    before = open(csvp).read()
    rebuild = os.path.join(tmp, 'promoted_need')
    fd = make_cell(rebuild)
    for p in [p for p in SC.loo_files(fd) if '_I_loo_' in p]:
        os.remove(p)
    md5s = {p: SC.md5(p) for p in _all_files(fd)}
    with R.Run('dewarp', ['test'], out, CFG, dry_run=True).unit(CELL) as u:
        SCL.record(u, roots_for(out, rebuild), CELL, ROW, None)
    check('count rule: the record stage of a cell whose I parent lost its LOO masters refuses G-R5 before any write',
          u.status == 'refused' and (u.reason or '').startswith(f'G-R5: {CELL}:') and 'holds 0 (none)' in u.reason
          and all(SC.md5(p) == m for p, m in md5s.items()) and open(csvp).read() == before,
          f'{u.status}: {(u.reason or "")[:300]}')


def _old_t3_bad(hh, cdig, n):
    """deep_verify.t3's per-LOO verdict before loo_link (v2 921c3ea, deep_verify.py:136), verbatim."""
    return hh.get('PARENTDG') != cdig or hh.get('NCOADD') != n - 1


def _old_dg16(lh, mh):
    """deep_cells.preview's D-G16 verdict before loo_link (v2 921c3ea, deep_cells.py:348-356), verbatim."""
    pd, cd = str(lh.get('PARENTDG', '') or ''), str(mh.get('CONSTDIG', '') or '')
    if pd and cd and pd != cd:
        return f'LOO STALE (PARENTDG {pd[:8]} != CONSTDIG {cd[:8]}): the engine REFUSES (D-G16)'
    elif not pd and lh.get('COADDOP') != mh.get('COADDOP'):
        return 'LOO STALE (legacy, COADDOP differs): the engine REFUSES (D-G16)'
    else:
        return 'loo'


def _grid():
    """Every combination of the cards the rules read: (LOO header, parent header)."""
    import itertools
    other = hashlib.md5(b'another parent').hexdigest()
    A = object()                                 # the card absent
    for pdg, cdg, lop, pop, lnc, pnc in itertools.product((A, '', None, DIG, other), (A, '', DIG, other),
                                                          (A, 'mean', 'median'), (A, 'mean'), (A, 1, 2), (A, 2, 3)):
        lh, mh = {}, {}
        for h, k, v in ((lh, 'PARENTDG', pdg), (mh, 'CONSTDIG', cdg), (lh, 'COADDOP', lop), (mh, 'COADDOP', pop),
                        (lh, 'NCOADD', lnc), (mh, 'NCOADD', pnc)):
            if v is not A:
                h[k] = v
        yield lh, mh


def _outcome(f, *a):
    try:
        return ('value', bool(f(*a)))
    except Exception as e:
        return ('raises', type(e).__name__)


def test_link_equivalence(tmp):
    from exopipe.phase3.lib.frames import deep_cells
    from exopipe.phase3.lib.masters import loo_link as LL
    grid = list(_grid())
    t3_diff = [(lh, mh) for lh, mh in grid
               if _outcome(_old_t3_bad, lh, mh.get('CONSTDIG'), mh.get('NCOADD'))
               != _outcome(lambda a, b: LL.digest_problem(a, b, legacy=False) or LL.ncoadd_problem(a, b), lh, mh)]
    check(f'one rule: T3\'s per-LOO verdict through loo_link equals its earlier inline code on all {len(grid)} card '
          f'combinations (exceptions included)', not t3_diff, str(t3_diff[:2]))
    d = os.path.join(tmp, 'dg16')
    os.makedirs(d)
    mp, lp = os.path.join(d, 'master.fits'), os.path.join(d, 'loo.fits')
    open(mp, 'w').close()
    open(lp, 'w').close()
    night = '2025-01-01'

    class FakeCell:
        lst, scope, final = LST, SCOPE, os.path.join(d, 'no_such_cell', 'coadd_final')
        hdr = {}

        def epoch_dates(self):
            return [night]

        def weights(self):
            return []

        def in_master(self):
            return {night: True}

        def master(self, st):
            return mp

        def loo(self, dd, st):
            return lp

        def header(self, p):
            return self.hdr[p]

    class NoExcl:
        def is_excluded(self, *a):
            return False
    c = FakeCell()
    dg_diff, gr_diff = [], []
    for lh, mh in grid:
        c.hdr = {lp: lh, mp: mh}
        try:
            got = deep_cells.preview(c, 'I', NoExcl())['loo'][night]
        except Exception as e:
            got = f'raises {type(e).__name__}'
        old = _old_dg16(lh, mh)
        if got != old:
            dg_diff.append((lh, mh, got, old))
        if bool(LL.digest_problem(lh, mh)) != (old != 'loo'):
            gr_diff.append((lh, mh, old))
    check(f'one rule: the D-G16 preview through loo_link gives its earlier verdict and wording on all {len(grid)} card '
          f'combinations', not dg_diff, str(dg_diff[:2]))
    check(f'one rule: G-R5\'s digest verdict is the search\'s (D-G16) on all {len(grid)} combinations - G-R5 never treats '
          f'as current what the search refuses, nor the reverse', not gr_diff, str(gr_diff[:2]))
    _t3_runs(tmp)


def _old_t3(gen_dir):
    """deep_verify.t3 before loo_link (v2 921c3ea, deep_verify.py:119-150), verbatim but for the module references."""
    from exopipe.phase3.lib.masters.deep_verify import _ctx, _cfg, _weights, sname
    c = _ctx(gen_dir)
    R, G, sc = c['R'], c['G'], c['sc']
    vs = _cfg().masters_verify
    lo, hi = _cfg().masters_build.LOO_N_RANGE
    w = _weights(R)
    inm = sorted(d for d, x in w.items() if x.get('in_master') == '1')
    parts, ok = [], True
    for st in vs.T3_STOKES:
        mvar = 'I_deep_Taper_R075' if st == 'I' else 'V_deep_Taper_R0'
        mh = fits.getheader(f'{G}/coadd_final/{sname(f"{mvar}_{sc}.fits", sc)}')
        n, cdig = mh.get('NCOADD'), mh.get('CONSTDIG')
        loos = sorted(glob.glob(f'{G}/coadd_final/loo/*_{st}_loo_*.fits'))
        lbad = []
        for p in loos:
            hh = fits.getheader(p)
            if hh.get('PARENTDG') != cdig or hh.get('NCOADD') != n - 1:
                lbad.append(os.path.basename(p))
        need = (lo <= n <= hi)
        if st == 'I':
            good = not lbad and (not need or len(loos) == n == len(inm))
        else:
            good = not lbad and (not need or len(loos) == n)
        ok = ok and good
        parts.append(f'{st}: NCOADD {n}, in-master nights {len(inm)}, LOO masters {len(loos)} (needed: {need}); '
                     f'not derivatives: {lbad[:3]}')
    return ok, len(inm), '; '.join(parts)


def _t3_runs(tmp):
    """Whole T3 runs, old and new, on small generations: the same (ok, nights, detail) or the same exception."""
    from exopipe.phase3.lib.masters import deep_verify as DV
    cases = {'clean': {}, 'foreign PARENTDG': {'PARENTDG': '0' * 32}, 'no PARENTDG': {'PARENTDG': None},
             'NCOADD wrong': {'NCOADD': 5}, 'one LOO missing': 'drop', 'N above the range': 'big',
             'parent without NCOADD': 'nonco', 'no CONSTDIG anywhere': 'nodig'}
    diff, kinds = [], set()
    for k, (name, how) in enumerate(cases.items()):
        cell = os.path.join(tmp, f't3_{k}', CELL)
        gen = os.path.join(cell, 'gen')
        fd = os.path.join(gen, 'coadd_final')
        os.makedirs(os.path.join(cell, 'coadd_final'))
        with open(os.path.join(cell, 'coadd_final', 'weights.csv'), 'w') as f:
            f.write('date,in_master\n' + ''.join(f'{d},1\n' for d in NIGHTS))
        n = 25 if how == 'big' else len(NIGHTS)
        for st, m in (('I', 'I_deep_Taper_R075'), ('V', 'V_deep_Taper_R0')):
            cards = {'CONSTDIG': DIG, 'NCOADD': n}
            if how == 'nonco':
                cards.pop('NCOADD')
            if how == 'nodig':
                cards.pop('CONSTDIG')
            write(os.path.join(fd, DV.sname(f'{m}_{SCOPE}.fits', SCOPE)), 1, **cards)
            for j, d in enumerate(NIGHTS):
                if how == 'drop' and st == 'V' and j == 1:
                    continue
                lc = {'PARENTDG': DIG, 'NCOADD': n - 1, 'LOOEXCL': d}
                if how == 'nodig':
                    lc.pop('PARENTDG')
                if isinstance(how, dict) and st == 'I' and j == 0:
                    lc.update({x: v for x, v in how.items() if v is not None})
                    for x in [x for x, v in how.items() if v is None]:
                        lc.pop(x)
                write(os.path.join(fd, 'loo', f'{d}_{st}_loo_{m[7:]}_{SCOPE}.fits'), 2, **lc)
        a, b = _outcome(lambda g: _old_t3(g), gen), _outcome(lambda g: DV.t3(g), gen)
        ra = _old_t3(gen) if a[0] == 'value' else a
        rb = DV.t3(gen) if b[0] == 'value' else b
        kinds.add(ra[0] if a[0] == 'value' else 'raises')
        if ra != rb:
            diff.append((name, ra, rb))
    check(f'one rule: whole T3 runs through loo_link equal the earlier T3 on {len(cases)} generations (passing, failing and '
          f'raising ones)', not diff and kinds == {True, False, 'raises'}, f'{diff} {kinds}')


def _all_files(fd):
    return sorted(glob.glob(os.path.join(fd, '*.fits')) + glob.glob(os.path.join(fd, 'loo', '*')))


def test_record_stage(tmp):
    out = os.path.join(tmp, 'out_ok')
    csvp = seed_record(out)
    rebuild = os.path.join(tmp, 'promoted_ok')
    fd = make_cell(rebuild)
    masters, loo = SC.mdwstat_files(types.SimpleNamespace(rebuild=rebuild), CELL), SC.loo_files(fd)
    dmd5 = {p: data_md5(p) for p in masters + loo}
    pb = os.path.join(fd, f'PB_deep_{SCOPE}.fits')
    pb_md5 = SC.md5(pb)
    with R.Run('dewarp', ['test'], out, CFG, dry_run=True).unit(CELL) as u:
        SCL.record(u, roots_for(out, rebuild), CELL, ROW, None)
    check('record stage: done', u.status == 'done', f'{u.status}: {(u.reason or "")[:300]}')
    got = {os.path.basename(p): fits.getheader(p).get('MDWSTAT') for p in masters + loo}
    check(f'record stage: the 3 masters and the 4 LOO masters carry the one value {WANT!r}',
          len(got) == 7 and set(got.values()) == {WANT}, str(got))
    from exopipe.common.validate_fits_headers import validate_header
    check('record stage: every stamped LOO master keeps a valid standard header',
          all(validate_header(p, level='standard').ok for p in loo))
    same = [p for p in masters + loo if data_md5(p) == dmd5[p]]
    check('record stage: the data bytes of all 7 stamped files are unchanged (md5 of the data section)',
          len(same) == 7, f'{len(same)} of 7')
    check('record stage: the PB map is not stamped, byte-identical', SC.md5(pb) == pb_md5)
    md = u.meta.get('mdwstat') or {}
    check('record stage: meta.mdwstat keeps the masters (files, the install\'s currency set) and the LOO masters '
          '(loo_files) apart, each with the md5 the stamp left',
          md.get('value') == WANT and [x['path'] for x in md.get('files', [])] == masters
          and [x['path'] for x in md.get('loo_files', [])] == loo
          and all(x['md5'] == SC.md5(x['path']) for x in md['files'] + md['loo_files']), str(md)[:300])
    outs = sorted(o['path'] for o in u.outputs if o['class'] == 'mdwstat_master')
    check('record stage: every stamped master and LOO master is an mdwstat_master output', outs == sorted(masters + loo),
          str(outs)[:200])
    t = [x for x in u.tests if x['name'] == 'dewarp.loo_mdwstat']
    check('record stage: the G-R5 check passed with the 4 LOO masters, its detail counting what was checked',
          len(t) == 1 and t[0]['passed'] and t[0]['value'] == '4 LOO master(s)'
          and t[0]['detail'].startswith('4 leave-one-out master(s) checked (2 I, 2 V)'), str(t))
    from exopipe.phase3.lib.dewarp.steps_cell import _cards
    from exopipe.phase3.lib.masters.release1_cards import COMMENTS
    card = fits.Card('MDWSTAT', WANT, COMMENTS['MDWSTAT']).image.encode('ascii')
    with open(masters[0], 'rb') as f:
        got_cards = [c for c in _cards(f.read(2880 * 4)) if c[:8] == b'MDWSTAT ']
    check("record stage: the stamped 80-character card is the one the Release 1 header repair writes (comment 'Master "
          "dewarp status', release1_cards.COMMENTS)", CFG.dewarp.MDWSTAT_COMMENT == COMMENTS['MDWSTAT'] ==
          'Master dewarp status' and got_cards == [card], f'{got_cards} {card}')
    rows = [r for r in SC.read_table(csvp) if (r['lst'], r['scope']) == (LST, SCOPE)]
    check('record stage: then the applied=0 row', len(rows) == 1 and rows[0]['applied'] == '0', str(rows))


def test_refusal(tmp):
    out = os.path.join(tmp, 'out_refused')
    csvp = seed_record(out)
    before = open(csvp).read()
    rebuild = os.path.join(tmp, 'promoted_refused')
    fd = make_cell(rebuild)
    bad = [p for p in SC.loo_files(fd) if '_V_loo_' in p][0]
    setcards(bad, PARENTDG='f' * 32)
    md5s = {p: SC.md5(p) for p in _all_files(fd)}
    with R.Run('dewarp', ['test'], out, CFG, dry_run=True).unit(CELL) as u:
        SCL.record(u, roots_for(out, rebuild), CELL, ROW, None)
    check('G-R5 refusal: a LOO master of another master refuses the record stage, the gate and the file named',
          u.status == 'refused' and (u.reason or '').startswith(f'G-R5: {CELL}:') and os.path.basename(bad) in u.reason
          and 'no MDWSTAT written, no applied=0 row' in u.reason, f'{u.status}: {(u.reason or "")[:300]}')
    check('G-R5 refusal: no half-stamped file - no master or LOO master carries MDWSTAT, every file keeps its md5',
          not any('MDWSTAT' in fits.getheader(p) for p in md5s if p.endswith('.fits'))
          and all(SC.md5(p) == m for p, m in md5s.items()), '')
    check('G-R5 refusal: no applied=0 row (the screen record unchanged: the cell stays routed)', open(csvp).read() == before)


def _temps(fd):
    return glob.glob(os.path.join(fd, '*.tmp')) + glob.glob(os.path.join(fd, 'loo', '*.tmp'))


def test_staging(tmp):
    """No in-place rewrite: temporary siblings, verified, then ONE replace_many; a failure leaves every original."""
    from exopipe.phase3.core import placement
    out = os.path.join(tmp, 'out_stage')
    seed_record(out)
    rebuild = os.path.join(tmp, 'promoted_stage')
    fd = make_cell(rebuild)
    masters, loo = SC.mdwstat_files(types.SimpleNamespace(rebuild=rebuild), CELL), SC.loo_files(fd)
    md5s = {p: SC.md5(p) for p in _all_files(fd)}
    real_stage = SCL._stage_mdwstat
    for k in (1, 3):                                   # the k-th LOO master's staging fails (a partial copy written)
        seen = []

        def failing(src, tmpn, value, comment):
            if '/loo/' in src:
                seen.append(src)
                if len(seen) == k:
                    with open(tmpn, 'wb') as f:
                        f.write(b'partial')
                    raise OSError(28, 'No space left on device (injected)')
            return real_stage(src, tmpn, value, comment)
        SCL._stage_mdwstat = failing
        try:
            with R.Run('dewarp', ['test'], out, CFG, dry_run=True).unit(CELL) as u:
                SCL.record(u, roots_for(out, rebuild), CELL, ROW, None)
        finally:
            SCL._stage_mdwstat = real_stage
        check(f'staging: a failure on the LOO master {k} of 4 refuses (dewarp.mdwstat, the file named) with every original '
              f'byte-identical and no temporary file left (no half-stamped cell)',
              u.status == 'refused' and 'dewarp.mdwstat' in (u.reason or '') and os.path.basename(loo[k - 1]) in u.reason
              and 'none of the 7 files changed' in u.reason and all(SC.md5(p) == m for p, m in md5s.items())
              and not _temps(fd), f'{u.status}: {(u.reason or "")[:200]} {_temps(fd)}')

    def grows(src, tmpn, value, comment):              # a staged copy that fails its check (size changed)
        r = real_stage(src, tmpn, value, comment)
        if src == loo[-1]:
            with open(tmpn, 'ab') as f:
                f.write(b'\0' * 2880)
        return r
    SCL._stage_mdwstat = grows
    try:
        with R.Run('dewarp', ['test'], out, CFG, dry_run=True).unit(CELL) as u:
            SCL.record(u, roots_for(out, rebuild), CELL, ROW, None)
    finally:
        SCL._stage_mdwstat = real_stage
    check('staging: a staged copy failing its read-back check (the last LOO master, size changed) refuses with every '
          'original byte-identical and no temporary file left', u.status == 'refused' and 'failed its check' in (u.reason or '')
          and all(SC.md5(p) == m for p, m in md5s.items()) and not _temps(fd), f'{u.status}: {(u.reason or "")[:200]}')
    calls, real_rm = [], placement.replace_many

    def spy(pairs):
        pairs = list(pairs)
        calls.append(pairs)
        return real_rm(pairs)
    placement.replace_many = spy
    held = open(masters[0], 'rb')                      # a concurrent reader holding the old file
    ino = os.fstat(held.fileno()).st_ino
    try:
        with R.Run('dewarp', ['test'], out, CFG, dry_run=True).unit(CELL) as u:
            SCL.record(u, roots_for(out, rebuild), CELL, ROW, None)
        old_hdr = held.read(2880 * 4)
    finally:
        placement.replace_many = real_rm
        held.close()
    from exopipe.phase3.core import tempnames
    tail = tempnames.tail('mdwstat')
    check('staging: done; the 3 masters and the 4 LOO masters placed together by ONE core.placement.replace_many call, each '
          'from a temporary sibling in its own directory', u.status == 'done' and len(calls) == 1
          and [d for _s, d in calls[0]] == masters + loo
          and all(os.path.dirname(s_) == os.path.dirname(d) and tail.search(os.path.basename(s_)) for s_, d in calls[0]),
          f'{u.status}: {(u.reason or "")[:200]} {calls}'[:400])
    check('staging: a reader holding a master keeps the old inode and the old bytes (no MDWSTAT); the path is a new file',
          os.stat(masters[0]).st_ino != ino and b'MDWSTAT ' not in old_hdr and b'MDWSTAT ' in open(masters[0], 'rb').read(11520))
    md = u.meta.get('mdwstat') or {}
    check('staging: the md5 recorded (taken from the write) is the md5 of each placed file; data sections unchanged',
          all(x['md5'] == SC.md5(x['path']) and x['size'] == os.path.getsize(x['path'])
              and x['mtime_ns'] == os.stat(x['path']).st_mtime_ns for x in md.get('files', []) + md.get('loo_files', []))
          and len(md.get('files', [])) == 3 and len(md.get('loo_files', [])) == 4 and not _temps(fd), str(md)[:300])


def test_source_gate(tmp):
    """G-R5 on an accepted cell's apply source, before the resample (steps_cell.apply -> loo_source_gate). The apply imports
    lib/dewarp/master_dewarp, whose import needs PyBDSF (phase2.cascade_dewarp), absent from the CI image: a stand-in module
    with the two names the apply reaches before the gate (_gen_root, and apply_saved, which must never be called) takes its
    place for this test, and the real entry (if any) is put back after."""
    import exopipe.phase3.lib.dewarp as pkg
    name = 'exopipe.phase3.lib.dewarp.master_dewarp'
    MD = types.ModuleType(name)
    MD.NO_RESIDUALS = False
    held = (sys.modules.get(name), getattr(pkg, 'master_dewarp', None))
    sys.modules[name] = MD
    pkg.master_dewarp = MD
    src = os.path.join(tmp, 'source')
    fd = make_cell(src)
    stale = SC.loo_files(fd)[2]
    setcards(stale, PARENTDG='e' * 32)
    out = os.path.join(tmp, 'out_source')
    roots = roots_for(out, src)
    scr = os.path.join(tmp, 'screen.fits')
    write(scr, 3)
    meas = types.SimpleNamespace(screen=lambda scope: scr, table=os.path.join(tmp, 'table.csv'), dir=tmp)
    run = types.SimpleNamespace(fresh=False)
    called = []
    keep = (SCL.screen_key_gate, SCL.pb_weights_gate, SCL._screen_nanchor)
    MD.apply_saved = lambda *a, **k: called.append(a) or 0
    MD._gen_root = lambda working, rebuild, lst, scope: os.path.join(src, CELL)
    SCL.screen_key_gate = lambda *a: (None, None, None)     # G-A2's screen, G-A12 and the PB weights: tested in test_dewarp
    SCL.pb_weights_gate = lambda *a: None
    SCL._screen_nanchor = lambda path: 40
    try:
        with R.Run('dewarp', ['test'], out, CFG, dry_run=True).unit(CELL) as u:
            SCL.apply(u, run, roots, CELL, meas, dict(ROW, n_anchors='40'))
    finally:
        SCL.screen_key_gate, SCL.pb_weights_gate, SCL._screen_nanchor = keep
        if held[0] is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = held[0]
        if held[1] is not None:
            pkg.master_dewarp = held[1]
        elif getattr(pkg, 'master_dewarp', None) is MD:
            del pkg.master_dewarp
    gen = roots.gen_of(CELL)
    check('G-R5 on the apply source: an accepted cell with a stale LOO master refuses before the resample - the apply never '
          'called, no gen_mdewarp_rbf on disk, the gate and the file named',
          u.status == 'refused' and (u.reason or '').startswith(f'G-R5: {CELL}:') and os.path.basename(stale) in u.reason
          and 'no generation written' in u.reason and not called and not os.path.exists(gen)
          and os.path.basename(gen) == 'gen_mdewarp_rbf', f'{u.status}: {(u.reason or "")[:300]} {gen}')
    t = [x for x in u.tests if x['name'] == 'dewarp.loo_source']
    check("G-R5 on the apply source: recorded as the failed check 'dewarp.loo_source'", len(t) == 1 and not t[0]['passed'],
          str(t))


def test_install_currency(tmp):
    """lib/install/mdwstat: the record's loo_files are a currency set as its files (the masters)."""
    import json
    from exopipe.phase3.lib.install import mdwstat as MW
    out = os.path.join(tmp, 'out_inst')
    seed_record(out)
    rebuild = os.path.join(tmp, 'promoted_inst')
    fd = make_cell(rebuild)
    with R.Run('dewarp', ['test'], out, CFG, dry_run=True).unit(CELL) as u:
        SCL.record(u, roots_for(out, rebuild), CELL, ROW, None)
    md = u.meta['mdwstat']
    gen = os.path.join(rebuild, CELL)

    def choice(m):
        p = R.record_path(out, 'dewarp', CELL)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w') as f:
            json.dump({'status': 'done', 'meta': {'mdwstat': m}}, f)
        return MW.choose(CELL, {}, 'no table', out)
    ch = choice(md)
    rel = {os.path.relpath(x['path'], gen): x['md5'] for x in md['files'] + md['loo_files']}
    src = MW.Source('no table', [ch])
    check('install currency: the record\'s loo_files are read (4 LOO masters) and the staged generation is current',
          ch.source == MW.RECORD and len(ch.loo_files) == 4 and src.staged_problems(CELL, rel) == []
          and '4 stamped leave-one-out masters' in ch.notes[-1], f'{ch.notes} {src.staged_problems(CELL, rel)}')
    lo = os.path.relpath(md['loo_files'][1]['path'], gen)
    p = src.staged_problems(CELL, dict(rel, **{lo: '0' * 32}))
    check('install currency: a LOO master of other bytes than the record stamped (its card or data changed) refuses, named',
          len(p) == 1 and os.path.basename(lo) in p[0] and '0' * 32 in p[0], str(p))
    p = src.staged_problems(CELL, {k: v for k, v in rel.items() if k != lo})
    check('install currency: a LOO master the record stamped but not staged refuses', len(p) == 1
          and 'stamped by the record but not installed' in p[0], str(p))
    extra = os.path.join('coadd_final', 'loo', f'2025-01-03_I_loo_Taper_R075_{SCOPE}.fits')
    p = src.staged_problems(CELL, dict(rel, **{extra: '1' * 32}))
    check('install currency: a LOO master the record did not stamp refuses', len(p) == 1
          and 'is not a leave-one-out master the record stamped' in p[0], str(p))
    ch0 = choice({k: v for k, v in md.items() if k != 'loo_files'})
    p = MW.Source('no table', [ch0]).staged_problems(CELL, rel)
    check('install currency: a record without loo_files (stamped no LOO master) is stale for a generation with LOO masters',
          len(p) == 4 and all('is not a leave-one-out master the record stamped' in x for x in p), str(p))
    lst = {os.path.basename(x['path']): (x['size'], x['mtime_ns']) for x in md['loo_files']}
    mst = {os.path.basename(x['path']): (x['size'], x['mtime_ns']) for x in md['files']}
    n1 = os.path.basename(lo)
    pr, de = MW.stat_problems(ch, mst, (), lst)
    pr2, _de2 = MW.stat_problems(ch, mst, (), dict(lst, **{n1: (lst[n1][0] + 2880, lst[n1][1])}))
    check('install currency: the plan\'s size/mtime pre-check covers the LOO masters (current: none; a size change: '
          'refused, named)', pr == [] and de == {} and len(pr2) == 1 and n1 in pr2[0], f'{pr} {de} {pr2}')
    inputs = {x['path']: {'md5': x['md5']} for x in md['files'] + md['loo_files']}
    irec = {'status': 'done', 'meta': {'gen': gen, 'mdwstat_source': {'source': MW.RECORD, 'value': WANT}},
            'inputs': inputs}
    ok = MW.installed_problems(ch, irec, gen)
    setcards(md['loo_files'][0]['path'], MDWSTAT='refused: something else')
    bad = MW.installed_problems(ch, irec, gen)
    check('install currency: installed_problems covers the installed LOO masters (inputs and card); a LOO master whose '
          'card differs from the record\'s value refuses, named', ok == [] and len(bad) == 1
          and 'installed leave-one-out master' in bad[0] and os.path.basename(md['loo_files'][0]['path']) in bad[0],
          f'{ok} {bad}')


def test_apply_record(tmp):
    gen = os.path.join(tmp, 'gen', CELL, 'gen_mdewarp_rbf')
    fd = make_cell(os.path.dirname(gen), mdwarp=True, token='_mdewarp')
    os.rename(os.path.join(os.path.dirname(gen), CELL), gen + '.tmp')
    os.makedirs(os.path.dirname(gen), exist_ok=True)
    os.rename(gen + '.tmp', gen)
    fd = os.path.join(gen, 'coadd_final')
    files = SC.master_files(fd) + SC.loo_files(fd)
    for p in files:
        setcards(p, MDWSTAT='applied')
    out = os.path.join(tmp, 'out_apply')
    run = R.Run('dewarp', ['test'], out, CFG, dry_run=True)
    with run.unit(CELL) as u:
        SCL.applied_mdwstat(u, CELL, gen)
    md = u.meta.get('mdwstat') or {}
    check("apply record: an accepted cell's LOO masters carry 'applied', recorded in loo_files apart from the masters",
          u.status == 'done' and md.get('value') == 'applied' and len(md.get('files', [])) == 3
          and [x['path'] for x in md.get('loo_files', [])] == SC.loo_files(fd), f'{u.status}: {(u.reason or "")[:200]}')
    lo = SC.loo_files(fd)[1]
    setcards(lo, MDWSTAT=None)
    with run.unit(CELL) as u2:
        SCL.applied_mdwstat(u2, CELL, gen)
    check("apply record: a LOO master without the card refuses ('dewarp.mdwstat', the file named)",
          u2.status == 'refused' and 'dewarp.mdwstat' in (u2.reason or '') and lo in u2.reason, (u2.reason or '')[:200])
    setcards(lo, MDWSTAT='applied', MDWARP=None)
    with run.unit(CELL) as u3:
        SCL.applied_mdwstat(u3, CELL, gen)
    check("apply record: a LOO master without MDWARP cannot carry 'applied' - refused G-R5",
          u3.status == 'refused' and (u3.reason or '').startswith('G-R5') and os.path.basename(lo) in u3.reason,
          (u3.reason or '')[:200])


def main():
    tmp = tempfile.mkdtemp(prefix='exopipe3_test_dewarp_loo_mdwstat_')
    try:
        test_loo_files(tmp)
        test_rule(tmp)
        test_need(tmp)
        test_link_equivalence(tmp)
        test_record_stage(tmp)
        test_staging(tmp)
        test_refusal(tmp)
        test_source_gate(tmp)
        test_apply_record(tmp)
        test_install_currency(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    bad = [r for r in RESULTS if not r[1]]
    print(f'\n{len(RESULTS) - len(bad)} of {len(RESULTS)} checks passed', flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
