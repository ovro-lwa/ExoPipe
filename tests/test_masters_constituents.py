"""THE MEMBERSHIP CHECK AGAINST THE CONSTITUENT LIST (refactor FOLD_IN rows of 2026-10-26, the deep chat's 'SILENT DROP' and
'MORE SILENT DROPS' findings): lib/masters/constituents.py, the content test master.members_equal_constituents and the build's
record of member nights without a Phase 2 image (lib/masters/deep_stages.record_missing_images).

A master's constituents must equal the cell's recorded members for its product: weights.csv in_master 1, minus (V and the I
variants) the nights recorded as refused - B-03 refusals (refusals_input.csv), VARIANT-GATE HISTORY lines, and the member
nights with no Phase 2 image of the product (the build's record, else found again in the Phase 2 tree). Each check fails if
a correct master is refused, or if a master holding a non-member or a refused night, a silent drop (with or without the
night's image), a wrong NCOADD or an unknown membership passes. Synthetic cells under /tmp/exopipe3_tests_masters_constituents;
READ-ONLY checks on installed production cells (lwacalim07 /data07: 23h_23MHz - the cell of the masters parity P1 - and
08h_18MHz, the finding's silent drop), skipped when absent.
Run: python tests/test_masters_constituents.py
"""
import csv
import hashlib
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from astropy.io import fits                                          # noqa: E402
from exopipe.phase3.core import env as _env                          # noqa: E402
_env.legacy()
from exopipe.phase3.config import CFG                                # noqa: E402
from exopipe.phase3.core import record, validate                     # noqa: E402
from exopipe.phase3.lib.masters import constituents as CN            # noqa: E402
from exopipe.phase3.lib.masters import deep_stages as DS             # noqa: E402

T = '/tmp/exopipe3_tests_masters_constituents'
FAILS, N = [], [0]
LST, SC_ = '05h', '64MHz'
CELL = f'{LST}_{SC_}'
D = ['2025-01-02', '2025-01-03', '2025-01-05', '2025-01-07', '2025-01-09']      # D[4]: not a member (in_master 0)
INSTALLED = CFG.paths.COADD_SB_ROOT


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'), flush=True)
    if not ok:
        FAILS.append(name)


IMAGES = {'survey': ('I', '64MHz-I-Taper-3600s-Robust-0.75-image_dewarped.fits'),
          'Taper_R0': ('I', '64MHz-I-Taper-3600s-Robust-0-image_dewarped.fits'),
          'NoTaper_R0': ('I', '64MHz-I-NoTaper-3600s-Robust-0-image_dewarped.fits'),
          'NoTaper_R075': ('I', '64MHz-I-NoTaper-3600s-Robust-0.75-image_dewarped.fits'),
          'V': ('V', '64MHz-V-Taper-3600s-Robust-0-image_dewarped.fits')}


def phase2_night(root, date, skip=()):
    """A Phase 2 night with every product's image (empty files: the finders read names) except ``skip``; a skipped NoTaper R0
    keeps only its .pbcorr file (the 08h / 09h 18 MHz case of the finding). Returns the Science dir."""
    sci = os.path.join(root, LST, date, 'Science_20260101')
    for prod, (st, name) in IMAGES.items():
        d = os.path.join(sci, SC_, st, 'deep')
        os.makedirs(d, exist_ok=True)
        if prod in skip:
            if prod == 'NoTaper_R0':
                open(os.path.join(d, name.replace('-image_dewarped.fits', '-image.pbcorr.fits')), 'w').close()
            continue
        open(os.path.join(d, name), 'w').close()
    return sci


def master(path, dates, history=(), ncoadd=None):
    h = fits.Header()
    h['NCOADD'] = len(dates) if ncoadd is None else ncoadd
    h['CONSTDIG'] = hashlib.md5(','.join(sorted(set(dates))).encode()).hexdigest()
    for d in dates:
        h['HISTORY'] = f'COADD constituent {d}'
    for x in history:
        h['HISTORY'] = x
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.writeto(path, np.zeros((4, 4), np.float32), h, overwrite=True)
    return path


def cell(name, skips=None, in_master=None, sci_missing=()):
    """A synthetic cell <T>/<name>/<CELL>: weights.csv over D (in_master 1 for D[0:4] unless ``in_master`` says otherwise),
    each night's Science dir in a Phase 2 tree of its own (``skips``: {date: products without an image}; ``sci_missing``:
    dates whose weights.csv row has no Science dir). Returns the cell directory."""
    base = os.path.join(T, name)
    cd = os.path.join(base, CELL)
    os.makedirs(os.path.join(cd, 'coadd_final'))
    im = in_master or {d: (d != D[4]) for d in D}
    with open(os.path.join(cd, 'coadd_final', 'weights.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'weight', 'qa3_scale', 'qa3_tilt', 'qa3_nsrc', 'qa3_guard', 'qa3_rms_mjy', 'sci_dir', 'in_master',
                    'flux_dc', 'qa3_scale_aligned', 'qa3_window'])
        for d in D:
            sci = '' if d in sci_missing else phase2_night(os.path.join(base, 'phase2'), d, (skips or {}).get(d, ()))
            w.writerow([d, '1.0', '1.0', '0', '50', 'pass', '10', sci, int(im[d]), '1.0', '1.0', 'inside'])
    return cd


def fp(cd, stem):
    return os.path.join(cd, 'coadd_final', f'{stem}_{SC_}.fits')


def masters_all(cd, members, override=None, history=None):
    """Every master of the cell with ``members`` as constituents unless ``override`` / ``history`` say otherwise."""
    for stem in CN.PRODUCT_OF:
        master(fp(cd, stem), (override or {}).get(stem, members), (history or {}).get(stem, ()))


def violations(path, cd, no_image=None):
    return CN.check(path, cd, no_image=no_image)[0]


def test_rule():
    M = D[:4]
    # 1. every member in every master
    cd = cell('normal')
    masters_all(cd, M)
    v, infos = CN.check_cell(cd)
    check('every master holding exactly the members passes (no-image nights found again in the Phase 2 tree)', not v and
          len(infos) == 5 and all(i['constituents'] == 4 for i in infos.values()), v)
    check('... and with the build\'s record (no member without an image)', not CN.check_cell(cd, no_image={})[0])
    # 2. the finding's case: a member night with only a .pbcorr NoTaper R0 image, left out of that master
    cd = cell('no_image', skips={D[2]: ('NoTaper_R0',)})
    masters_all(cd, M, override={'I_deep_NoTaper_R0': [d for d in M if d != D[2]]})
    v, info = CN.check(fp(cd, 'I_deep_NoTaper_R0'), cd)
    check('a member night without a Phase 2 image of the variant, left out: passes, the night named as no image', not v and
          info['no_image'] == [D[2]] and 'no Phase 2 NoTaper_R0 image' in info['refused'][D[2]], (v, info.get('refused')))
    check('... with the build\'s record naming it: passes', not violations(fp(cd, 'I_deep_NoTaper_R0'), cd, [D[2]]))
    v = violations(fp(cd, 'I_deep_NoTaper_R0'), cd, [])
    check('... with a build record that names no such night: REFUSED (a silent drop)', len(v) == 1 and 'silent drop' in v[0]
          and D[2] in v[0] and 'not recorded as without a Phase 2 image' in v[0], v)
    noimg, unj = CN.missing_images_cell(cd, SC_)
    check('missing_images_cell finds exactly that night and product',
          {k: x for k, x in noimg.items() if x} == {'I_deep_NoTaper_R0': [D[2]]} and not any(unj.values()), noimg)
    # 3. a silent drop of a night whose image exists (an alignment failure), then the same night recorded by the variant gate
    cd = cell('silent')
    masters_all(cd, M, override={'I_deep_Taper_R0': M[:3]})
    v = violations(fp(cd, 'I_deep_Taper_R0'), cd)
    check('a member left out although its image exists, with no record: REFUSED', len(v) == 1 and 'silent drop' in v[0]
          and 'its Phase 2 image exists' in v[0] and M[3] in v[0], v)
    master(fp(cd, 'I_deep_Taper_R0'), M[:3], [f'VARIANT-GATE refused {M[3]}: divergent input (|max| 2e+05)'])
    check('the same night refused by the variant gate (HISTORY): passes', not violations(fp(cd, 'I_deep_Taper_R0'), cd))
    # 4. B-03 same-state refusals (refusals_input.csv)
    cd = cell('b03')
    masters_all(cd, M, override={'V_deep_Taper_R0': [d for d in M if d != D[3]],
                                 'I_deep_Taper_R0': [d for d in M if d != D[3]]})
    v = violations(fp(cd, 'V_deep_Taper_R0'), cd) + violations(fp(cd, 'I_deep_Taper_R0'), cd)
    check('V and Taper R0 left out without a B-03 record: REFUSED (both)', len(v) == 2, v)
    with open(os.path.join(cd, 'refusals_input.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['lst', 'scope', 'date', 'product', 'image', 'reason'])
        w.writerow([LST, SC_, D[3], 'V', 'x.fits', 'raw V on a dewarped survey image'])
        w.writerow([LST, SC_, D[3], 'I_Taper_R0', 'y.fits', 'raw I_Taper_R0 on a dewarped survey image'])
    check('... with the B-03 refusals recorded: both pass',
          not violations(fp(cd, 'V_deep_Taper_R0'), cd) and not violations(fp(cd, 'I_deep_Taper_R0'), cd))
    # 5. a master holding a non-member, a refused night, a wrong NCOADD; the survey master; no record at all
    cd = cell('wrong')
    masters_all(cd, M)
    master(fp(cd, 'I_deep_NoTaper_R075'), M + [D[4]])
    v = violations(fp(cd, 'I_deep_NoTaper_R075'), cd)
    check('a master holding a night that is not a member (in_master 0): REFUSED', any('not members' in x for x in v), v)
    master(fp(cd, 'I_deep_NoTaper_R075'), M, [f'VARIANT-GATE refused {M[1]}: divergent input (x)'])
    v = violations(fp(cd, 'I_deep_NoTaper_R075'), cd)
    check('a master holding a night its own HISTORY records as refused: REFUSED', any('although recorded as refused' in x
                                                                                     for x in v), v)
    master(fp(cd, 'I_deep_NoTaper_R075'), M, ncoadd=3)
    v = violations(fp(cd, 'I_deep_NoTaper_R075'), cd)
    check('NCOADD differing from the constituent lines: REFUSED', any('NCOADD' in x for x in v), v)
    master(fp(cd, 'I_deep_Taper_R075'), M[:3])
    v = violations(fp(cd, 'I_deep_Taper_R075'), cd)
    check('the survey master missing a member: REFUSED (no reason can excuse it)', len(v) == 1 and 'silent drop' in v[0], v)
    os.remove(os.path.join(cd, 'coadd_final', 'weights.csv'))
    v = violations(fp(cd, 'I_deep_Taper_R0'), cd)
    check('no weights.csv: REFUSED (membership unknown)', len(v) == 1 and 'membership unknown' in v[0], v)
    cd = cell('nosci', sci_missing=(D[1],))
    masters_all(cd, M)
    v = violations(fp(cd, 'V_deep_Taper_R0'), cd)
    check('a member without a Science directory cannot be looked up: REFUSED', any('no Science directory' in x for x in v), v)


def test_content_test_and_layouts():
    M = D[:4]
    cd = cell('ct', skips={D[2]: ('NoTaper_R0',)})
    masters_all(cd, M, override={'I_deep_NoTaper_R0': [d for d in M if d != D[2]]})

    def ct(path, **ctx):
        return {r['name']: r for r in validate.run('master', path, **ctx)}['master.members_equal_constituents']
    r = ct(fp(cd, 'I_deep_NoTaper_R0'), cell=CELL, no_image=[D[2]])
    check('content test: the build\'s record kept with the output passes', r['passed'], r['detail'])
    r = ct(fp(cd, 'I_deep_NoTaper_R0'), cell=CELL, no_image=[])
    check('content test: a record without the night fails (the unit is refused)', not r['passed'], r['detail'])
    r = ct(fp(cd, 'I_deep_NoTaper_R0'))
    check('content test without context (exopipe3 verify): the night found again, passes', r['passed'] and 'found again'
          in r['detail'], r['detail'])
    # the scaled generation keeps the cell's records (weights.csv, refusals_input.csv in the cell directory)
    g = master(os.path.join(cd, 'gen_shflux', 'coadd_final', f'I_deep_NoTaper_R0_shflux_{SC_}.fits'), [d for d in M if d != D[2]])
    r = ct(g, cell=CELL)
    check('content test: the gen_shflux master is checked against its cell\'s weights.csv', r['passed'], r['detail'])
    master(g, M[:2])
    check('content test: a gen_shflux master missing a member fails', not ct(g, cell=CELL)['passed'])
    # the variant-only rebuild stamps every night it leaves out in HISTORY (no_image=[] is its record)
    vr = master(os.path.join(cd, 'variants_rebuilt', f'I_deep_NoTaper_R0_{SC_}.fits'), [d for d in M if d != D[2]],
                [f'VARIANT-GATE refused {D[2]}: divergent input (no NoTaper_R0 input)'])
    r = ct(vr, cell=CELL, no_image=[])
    check('content test: the variant-only rebuild\'s HISTORY record of a missing input passes', r['passed'], r['detail'])
    # the build's registration of its outputs (deep_stages.register_build_outputs, called by build after the co-add): the
    # no-image record in the unit and every master output's content test with it, through UnitRecord.add_output
    run = record.Run('masters', ['test'], os.path.join(T, 'run'), CFG, dry_run=True)

    def mine(u):
        return {os.path.basename(t['path']): t for t in u.tests if t['name'] == 'master.members_equal_constituents'}
    with run.unit(f'build_{CELL}') as u:
        DS.register_build_outputs(u, CELL, cd, SC_)
    res, ctxs = mine(u), {os.path.basename(o['path']): o.get('ctx') for o in u.outputs if o['class'] == 'master'}
    check('build: the no-image night is recorded in the unit (meta + note); every master output passes with it, the record '
          'kept as its context', u.meta.get('no_image') == {'I_deep_NoTaper_R0': [D[2]]}
          and any('NO IMAGE (recorded refusal)' in n for n in u.notes) and len(res) == 5
          and all(t['passed'] for t in res.values()) and ctxs.get(f'I_deep_NoTaper_R0_{SC_}.fits', {}).get('no_image') == [D[2]],
          (u.meta.get('no_image'), {k: t['detail'][:80] for k, t in res.items() if not t['passed']}))
    master(fp(cd, 'I_deep_Taper_R0'), M[:3])                     # a silent drop: D[3] has its Taper R0 image
    with run.unit(f'build_{CELL}_silent') as u:
        DS.register_build_outputs(u, CELL, cd, SC_)
    res = mine(u)
    failed = {t['path'] for t in u.failed_tests() if t['name'] == 'master.members_equal_constituents'}
    check('build: a master that silently drops a member fails its content test, which refuses the build unit (the other '
          'masters pass)', u.status == 'refused' and failed == {os.path.abspath(fp(cd, 'I_deep_Taper_R0'))}
          and all(t['passed'] for k, t in res.items() if k != f'I_deep_Taper_R0_{SC_}.fits'),
          (u.status, sorted(failed)))
    master(fp(cd, 'I_deep_Taper_R0'), M)
    cd2 = cell('ct_nosci', sci_missing=(D[0],))
    with run.unit(f'build_{CELL}_nosci') as u:
        DS.record_missing_images(u, cd2, SC_)
    check('build: a member without a Science directory refuses the build', u.status == 'refused'
          and 'Science directory' in u.reason, u.reason[:160])
    # a stale copy of the co-add finders (FINDER_PINS no longer the coadd.py text) is never used to look images up
    check('the finder copy is current (FINDER_PINS = lib/masters/coadd.py)', CN.finder_copy_current() == [],
          CN.finder_copy_current())
    saved = dict(CN._PINS)
    CN._PINS['bad'] = ['find_subband_variant_image']
    try:
        with run.unit(f'build_{CELL}_stale') as u:
            DS.record_missing_images(u, cd, SC_)
        r = ct(fp(cd, 'I_deep_NoTaper_R0'))
    finally:
        CN._PINS.clear()
        CN._PINS.update(saved)
    check('a stale finder copy refuses the build\'s record and fails the content test (no lookup with it)',
          u.status == 'refused' and 're-port core.inputs' in u.reason and not r['passed'] and 're-port' in r['detail'],
          (u.reason[:120], r['detail'][:120]))


def test_real_cells():
    """Read only: the installed 23h_23MHz cell (the masters parity cell P1; installed 2026-10-13 from the rebuild P1 compared
    with) passes in every mode; 08h_18MHz's silent drop (2025-01-01, NoTaper R0) is identified as a missing image."""
    for cellname in ('23h_23MHz', '08h_18MHz'):
        cd = os.path.join(INSTALLED, cellname)
        if not os.path.isfile(os.path.join(cd, 'coadd_final', 'weights.csv')):
            print(f'SKIP  {cd} not on this node', flush=True)
            continue
        sc = cellname.split('_', 1)[1]
        noimg, unj = CN.missing_images_cell(cd, sc)
        v1, infos = CN.check_cell(cd)
        v2, _ = CN.check_cell(cd, no_image=noimg)
        if cellname == '23h_23MHz':
            check('real (read-only) 23h_23MHz: all five masters hold exactly the 11 members, found again and as the build '
                  'records it', not v1 and not v2 and len(infos) == 5 and not any(noimg.values())
                  and all(i['constituents'] == i['members'] == 11 for i in infos.values()), (v1, v2, noimg))
        else:
            v3, _ = CN.check_cell(cd, no_image={})
            check('real (read-only) 08h_18MHz: the finding\'s dropped night is a missing NoTaper R0 image (passes once '
                  'recorded); unrecorded it is refused', not v1 and not v2
                  and {k: x for k, x in noimg.items() if x} == {'I_deep_NoTaper_R0': ['2025-01-01']}
                  and len(v3) == 1 and 'silent drop' in v3[0], (v1, v2, noimg, v3))


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_rule()
        test_content_test_and_layouts()
        test_real_cells()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
