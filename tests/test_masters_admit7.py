"""D-ADMIT (a) slice 5: membership read with a cell's admitted nights (lib/masters/admit.verify / membership; plan/wave2/
DADMIT_DESIGN.md "Slices 5 and 6 as they will be written").

A cell tree made from the records of the installed 05h_78MHz (weights.csv, the stage-2, night-image and epoch records; the
masters and PB maps linked, read only) with a Stokes I residual NAME for every recorded night (membership.verify reads only the names
in subtracted/) and one more for a night no record names: membership.verify reports that night ABSENT from the co-add
record - production's sentence, which admit.absent_violation must equal exactly. With an admissions.csv row admitting the
night into this master the ruled reading ('admitted') moves the violation to the reports ('admitted, rebuild pending');
production's reading ('absent'), a row into another master (another CONSTDIG, or the same CONSTDIG with other pixels - a
re-baked or re-dewarped master keeps its CONSTDIG), a refused row and a second absent night keep it; every other violation
stays. Then the call sites: no branch module calls membership.verify but admit.py (a static check); the masters
installed-cell check, the class installed_cell and membership_record content tests, and the archive quarantine's precheck
(production's membership code in a child on this cell, its violations read by the parent) read the admitted night as the
ruling says, and as production under 'absent'.
Run: python tests/test_masters_admit7.py
"""
import ast
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
from exopipe.phase3.core.record import Refusal                      # noqa: E402
from exopipe.phase3.lib.masters import admit as A                   # noqa: E402
from exopipe.phase3.lib.masters import membership as MB             # noqa: E402

T = '/tmp/exopipe3_tests_admit7'
FAILS, N = [], [0]
CELL, LST, SC = '05h_78MHz', '05h', '78MHz'
NEW = '2026-10-20'                                   # a night no record of the cell names (in the future: never excluded)


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def make_cell(root, extra=(NEW,)):
    """root/<CELL>: the installed cell's membership records, its masters and PB maps linked (read only), an I residual
    name per recorded night and per ``extra`` night."""
    src, dst = os.path.join(CFG.paths.COADD_SB_ROOT, CELL), os.path.join(root, CELL)
    os.makedirs(os.path.join(dst, 'coadd_final'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'subtracted'), exist_ok=True)
    for rel in ('coadd_final/weights.csv', 'coadd_final/stage2_verdict.csv', 'coadd_final/night_image_gate.csv',
                'coadd_final/epoch_gate.csv', 'residuals_provisional.csv'):
        if os.path.exists(os.path.join(src, rel)):
            shutil.copy2(os.path.join(src, rel), os.path.join(dst, rel))
    for rel in (f'coadd_final/I_deep_Taper_R075_{SC}.fits', f'coadd_final/V_deep_Taper_R0_{SC}.fits',
                f'coadd_final/PB_deep_{SC}.fits', f'coadd_final/PB_deep_R075_{SC}.fits'):   # what verify_products reads
        os.symlink(os.path.join(src, rel), os.path.join(dst, rel))
    nights = [r['date'] for r in csv.DictReader(open(os.path.join(dst, 'coadd_final', 'weights.csv')))]
    for d in list(nights) + list(extra):
        open(os.path.join(dst, 'subtracted', f'{d}_I_sub_{SC}.fits'), 'w').close()
    return dst


def admission(cd, date, verdict='admitted', constdig=None, md5=None):
    from exopipe.phase3.lib.masters import admit_stages as AS
    p = os.path.join(cd, 'coadd_final', A.ADMISSIONS)
    if os.path.exists(p):
        os.remove(p)
    cdg, m5 = A.master_identity(cd)
    row = dict.fromkeys(AS.ADMISSION_COLS, '')
    row.update(date=date, verdict=verdict, reason='' if verdict == 'admitted' else 'test', utc='2026-10-21T00:00:00Z',
               reference_constdig=cdg if constdig is None else constdig, reference_md5=m5 if md5 is None else md5)
    AS.write_admissions(cd, [row])


def test_the_reading(cd):
    base = MB.verify(cd)
    want = A.absent_violation(LST, SC, NEW)
    check('membership.verify reports the unrecorded night with production\'s sentence, exactly admit.absent_violation',
          want in base, [v for v in base if NEW in v])
    check('the installed records give no other violation (the copy is the installed cell\'s)', base == [want], base)
    v, rep = A.verify(cd, policy='admitted')
    check('no admissions.csv: the violation stays, nothing reported', v == [want] and not rep, (v, rep))
    admission(cd, NEW, constdig='0' * 32)
    v, rep = A.verify(cd, policy='admitted')
    check('a row admitting the night into a master of other nights (CONSTDIG): the violation stays', v == [want] and not rep,
          (v, rep))
    admission(cd, NEW, md5='0' * 32)
    v, rep = A.verify(cd, policy='admitted')
    check('a row with the master\'s CONSTDIG but other pixels (a re-baked / re-dewarped master keeps its CONSTDIG): the '
          'violation stays', v == [want] and not rep, (v, rep))
    admission(cd, NEW, verdict='refused')
    v, rep = A.verify(cd, policy='admitted')
    check('a refused row: the violation stays', v == [want] and not rep, (v, rep))
    admission(cd, NEW)
    v, rep = A.verify(cd, policy='admitted')
    check("ruled reading: the admitted night leaves the violations, reported 'admitted, rebuild pending'",
          v == [] and len(rep) == 1 and NEW in rep[0] and 'rebuild pending' in rep[0], (v, rep))
    v, rep = A.verify(cd)
    check("the config default is the ruled reading ('admitted')", CFG.masters_admit.ADMITTED_NIGHTS == 'admitted'
          and v == [] and len(rep) == 1, (CFG.masters_admit.ADMITTED_NIGHTS, v, rep))
    v, rep = A.verify(cd, policy='absent')
    check("production's reading: the admitted night is an absent night (a violation, nothing reported)",
          v == [want] and not rep, (v, rep))
    open(os.path.join(cd, 'subtracted', f'2026-10-21_I_sub_{SC}.fits'), 'w').close()
    v, rep = A.verify(cd, policy='admitted')
    check('a second absent night, not admitted, stays a violation', v == [A.absent_violation(LST, SC, '2026-10-21')]
          and len(rep) == 1, (v, rep))
    os.remove(os.path.join(cd, 'subtracted', f'2026-10-21_I_sub_{SC}.fits'))
    wp = os.path.join(cd, 'coadd_final', 'weights.csv')
    keep = open(wp).read()
    w = list(csv.DictReader(open(wp)))
    mem = next(r for r in w if r.get('in_master') == '1' and r.get('qa3_guard') != 'reference-identity')
    mem['qa3_scale'] = '9.0'
    with open(wp, 'w', newline='') as f:
        dw = csv.DictWriter(f, fieldnames=list(w[0].keys()))
        dw.writeheader()
        dw.writerows(w)
    v, rep = A.verify(cd, policy='admitted')
    check('every other violation stays (a member outside the stage-3 window)', len(v) == 1 and mem['date'] in v[0]
          and 'IN the master' in v[0] and len(rep) == 1, (v, rep))
    open(wp, 'w').write(keep)
    try:
        A.admitted_policy('admited')
        bad = 'no refusal'
    except Refusal as e:
        bad = str(e)
    check('an unknown masters_admit.ADMITTED_NIGHTS refuses', 'ADMITTED_NIGHTS' in bad, bad)
    par = open(os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg')).read()
    check("the parity values file sets production's reading ('absent')", "masters_admit.ADMITTED_NIGHTS = 'absent'" in par)


def test_call_sites(cd):
    """The checks that run on a cell tree here, under both readings."""
    from exopipe.phase3.lib.install import tests as IT
    from exopipe.phase3.lib.masters import deep_stages as DS, tests as MT
    from exopipe.phase3.config import CFG as C
    admission(cd, NEW)
    root = os.path.dirname(cd)
    got = {}
    for pol in ('admitted', 'absent'):
        C.masters_admit.ADMITTED_NIGHTS = pol
        try:
            got[pol] = dict(
                installed=[v for v in DS.installed_cell_violations(root, CELL) if 'ABSENT' in v],
                t8=IT.ic_membership(cd)[0],
                record=MT.verify(os.path.join(cd, 'coadd_final', 'weights.csv'))[0])
        finally:
            C.masters_admit.ADMITTED_NIGHTS = 'admitted'
    check('ruled reading: the masters installed-cell check, install T8 (installed_cell membership) and the class '
          'membership_record test pass the admitted night', got['admitted'] == dict(installed=[], t8=True, record=True),
          got['admitted'])
    check('production\'s reading: each of them fails on it, as production does', got['absent'] == dict(
        installed=[A.absent_violation(LST, SC, NEW)], t8=False, record=False), got['absent'])


def test_quarantine_precheck(cd):
    """The archive quarantine's G7 precheck runs production's membership code in a child: on this cell it fails with the
    absent-night violation; the parent passes it under the ruled reading only."""
    from exopipe.phase3.lib.archive import quarantine as Q

    class _Q:                                        # the cell roots the child searches: this test's tree only
        PRECHECK_CELL_ROOTS = (os.path.dirname(cd),)
        PRECHECK_CELL_MARKER = 'coadd_final/weights.csv'
    real = Q._q
    Q._q = lambda: _Q
    try:
        admission(cd, NEW)
        CFG.masters_admit.ADMITTED_NIGHTS = 'absent'
        try:
            ok_abs, msg_abs = Q.membership_precheck(CELL)
        finally:
            CFG.masters_admit.ADMITTED_NIGHTS = 'admitted'
        ok_adm, msg_adm = Q.membership_precheck(CELL)
    finally:
        Q._q = real
    check("archive precheck, production's reading: production's child refuses the cell on the absent night",
          not ok_abs and msg_abs.startswith('PRECHECK-FAIL') and NEW in msg_abs, msg_abs)
    check('archive precheck, ruled reading: the same child output is read with the admitted night - passes, reported',
          ok_adm and msg_adm.startswith('PRECHECK-OK') and 'rebuild pending' in msg_adm, msg_adm)


def test_quarantine_apply(cd):
    """MAJOR (eighth review): production's frame-quarantine helper re-runs production's membership check and refuses a cell
    holding an admitted night; under the ruled reading such a cell's moves are run by the helper's port, under production's by
    the helper (its own refusal, as production)."""
    import inspect
    from exopipe.phase3.lib.archive import quarantine as Q

    class _Q:
        PRECHECK_CELL_ROOTS = (os.path.dirname(cd),)
        PRECHECK_CELL_MARKER = 'coadd_final/weights.csv'
    real = Q._q
    Q._q = lambda: _Q
    try:
        admission(cd, NEW)
        got = Q.admitted_cell(CELL)
        CFG.masters_admit.ADMITTED_NIGHTS = 'absent'
        try:
            prod = Q.admitted_cell(CELL)
        finally:
            CFG.masters_admit.ADMITTED_NIGHTS = 'admitted'
        os.remove(os.path.join(cd, 'coadd_final', A.ADMISSIONS))
        none = Q.admitted_cell(CELL)
    finally:
        Q._q = real
    src = inspect.getsource(Q.quarantine_for_search)
    check('the quarantine apply: a cell holding an admitted night (ruled reading) takes the port, every other cell - and every '
          'cell under production\'s reading - production\'s helper', got == [NEW] and prod == [] and none == []
          and 'if production and not admitted:' in src and 'admitted = admitted_cell(cell)' in src, (got, prod, none))


def test_no_other_caller():
    """No branch module calls membership.verify but admit.py (and membership.py itself): every step reads admitted nights
    through the one entry point. (The quarantine precheck's child template is a string - production's code in a child; its
    parent reads the child's violations with admit.membership.)"""
    bad = []
    for p in glob.glob(os.path.join(REPO, 'exopipe', '**', '*.py'), recursive=True):
        rel = os.path.relpath(p, REPO)
        if rel.endswith(('lib/masters/admit.py', 'lib/masters/membership.py')):
            continue
        for n in ast.walk(ast.parse(open(p).read())):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'verify':
                v = n.func.value
                txt = ast.dump(v)
                if 'membership' in txt or (isinstance(v, ast.Name) and v.id in ('MB', '_MB', 'mb')):
                    bad.append(f'{rel}:{n.lineno}')
    check('no branch module calls membership.verify but admit.verify', not bad, bad)


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        cd = make_cell(os.path.join(T, 'a'))
        test_the_reading(cd)
        test_call_sites(cd)
        test_quarantine_precheck(cd)
        test_quarantine_apply(cd)
        test_no_other_caller()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
