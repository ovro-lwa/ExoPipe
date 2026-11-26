#!/usr/bin/env python3
"""Data-free test (CI): the dewarp step's MDWSTAT card on the leave-one-out (LOO) masters, gate G-R5.

Production's Release 1 header repair wrote MDWSTAT on every archived LOO master with its cell's value - the value of the
cell's central masters - when the LOO master is its parent's (parent digest, LOOEXCL) and the value agrees with its MDWARP
card (exopipe_ops/header_repair_release1.py:1077-1090, mdw_cards :830-863; DONE-20261114-HEADER-REPAIR-LATE-DEEP-CLASSES-449-
CELLS). The v2 dewarp step applies the same rule (lib/dewarp/steps_common.loo_files, loo_mdwstat_problems; steps_cell.record,
applied_mdwstat, loo_gate). Checked here on small synthetic cells in a temporary directory (TMPDIR), nothing else read:
  1. loo_files: the files of coadd_final/loo that coadd._LOO_RE names, sorted; none without loo/
  2. loo_mdwstat_problems: a clean cell passes; the CONSTDIG fallback of builds before PARENTDG passes; a foreign parent
     digest, a wrong LOOEXCL, a missing parent master, an MDWARP disagreeing with the value, a subband combination (LSWBSB
     without MDWARP) and a name that is not a LOO master each give one problem naming the file
  3. the record stage of a not-accepted cell stamps its masters AND its LOO masters with one value, header valid, data
     bytes unchanged; meta.mdwstat keeps the masters (the install's currency set) and the LOO masters (loo_files) apart;
     each is an mdwstat_master output; then the applied=0 row
  4. G-R5 refusal: one LOO master of another master refuses the record stage before any write - no file carries the card,
     every file keeps its md5, no applied=0 row
  5. the apply's record of an accepted cell: its LOO masters must carry 'applied' (recorded in loo_files); a LOO master
     without the card refuses 'dewarp.mdwstat', one without MDWARP refuses G-R5

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
        write(os.path.join(fd, f'{n}{token}_{SCOPE}.fits'), i, CONSTDIG=DIG, **extra)
    write(os.path.join(fd, f'PB_deep{token}_{SCOPE}.fits'), 9)
    for j, d in enumerate(NIGHTS):
        for st, tw in (('I', 'Taper_R075'), ('V', 'Taper_R0')):
            write(os.path.join(fd, 'loo', f'{d}_{st}_loo_{tw}{token}_{SCOPE}.fits'), 10 + j, CONSTDIG=DIG, PARENTDG=DIG,
                  LOOEXCL=d, **extra)
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
    check('G-R5 rule: a clean cell - every LOO master its parent\'s, no MDWARP, a refused value - no problem',
          SC.loo_mdwstat_problems(fd, WANT) == [], str(SC.loo_mdwstat_problems(fd, WANT)))
    setcards(i1, PARENTDG=None)
    check('G-R5 rule: a build before PARENTDG links by its CONSTDIG (the parent\'s, copied with its header)',
          SC.loo_mdwstat_problems(fd, WANT) == [], str(SC.loo_mdwstat_problems(fd, WANT)))
    setcards(i1, PARENTDG='0' * 32)
    p = SC.loo_mdwstat_problems(fd, WANT)
    check('G-R5 rule: a foreign parent digest - one problem naming the file and both digests',
          len(p) == 1 and os.path.basename(i1) in p[0] and '0' * 32 in p[0] and DIG in p[0], str(p))
    setcards(i1, PARENTDG=DIG, LOOEXCL='2025-01-03')
    p = SC.loo_mdwstat_problems(fd, WANT)
    check('G-R5 rule: LOOEXCL other than the night the name excludes - one problem',
          len(p) == 1 and 'LOOEXCL' in p[0] and os.path.basename(i1) in p[0], str(p))
    setcards(i1, LOOEXCL=NIGHTS[0])
    vm = os.path.join(fd, f'V_deep_Taper_R0_{SCOPE}.fits')
    keep = vm + '.keep'
    os.rename(vm, keep)
    p = SC.loo_mdwstat_problems(fd, WANT)
    os.rename(keep, vm)
    check('G-R5 rule: a missing parent master - one problem per LOO master of it (the two V ones)',
          len(p) == 2 and all('V_deep_Taper_R0' in x and 'is not in' in x for x in p), str(p))
    setcards(v2, MDWARP=True)
    p = SC.loo_mdwstat_problems(fd, WANT)
    check("G-R5 rule: a LOO master carrying MDWARP T cannot take a 'refused' value - one problem",
          len(p) == 1 and 'MDWARP' in p[0] and os.path.basename(v2) in p[0], str(p))
    p = SC.loo_mdwstat_problems(fd, 'applied')
    check("G-R5 rule: 'applied' needs MDWARP T on every LOO master - the three without it",
          len(p) == 3 and os.path.basename(v2) not in ' '.join(p), str(p))
    setcards(v2, MDWARP=None, LSWBSB='18MHz,23MHz')
    p = SC.loo_mdwstat_problems(fd, WANT)
    check('G-R5 rule: a combination of subband products (LSWBSB without MDWARP) takes no master\'s value - one problem',
          len(p) == 1 and 'LSWBSB' in p[0], str(p))
    setcards(v2, LSWBSB=None)
    p = SC.loo_mdwstat_problems(fd, WANT, loo + [os.path.join(fd, 'loo', 'README.txt')])
    check('G-R5 rule: a file that is not a LOO master name - one problem', len(p) == 1 and 'README.txt' in p[0], str(p))


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
    check('record stage: the G-R5 check passed with the 4 LOO masters', len(t) == 1 and t[0]['passed']
          and t[0]['value'] == '4 LOO master(s)', str(t))
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
        test_record_stage(tmp)
        test_refusal(tmp)
        test_apply_record(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    bad = [r for r in RESULTS if not r[1]]
    print(f'\n{len(RESULTS) - len(bad)} of {len(RESULTS)} checks passed', flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
