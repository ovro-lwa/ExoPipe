"""D-ADMIT (a), slice 4 part 3 (plan/wave2/DADMIT_DESIGN.md "Slice 4 part 3"): the masters admit stage end to end, and
admission_reproduces.

On a copy of 05h_78MHz (built 2026-10-08 with today's flux rules - a cell built before them, such as 13h_78MHz of
2026-09-10, carries the old rule in its members: +0.27% and +0.31% at bright peaks there; no master dewarp: the members'
stored flows apply - no GPU) two member nights are taken out of
every record of the build (a consistent build without them) and admitted again: `masters --stages admit-prep`, the flow
step's admit stage with the stored flows copied in (fresh: nothing solved), then `masters --stages admit` with the guard
median DECISIONS_PENDING item 10 recommends (1.0, set for this test only - the config default refuses). The admitted
epoch must equal the member's installed epoch (its installed residual + the installed master) at the master's bright
compact sources: median |ratio - 1| <= REPRODUCE_TOL (1e-3, Gregg's tolerance), the measured values printed - where they
differ by design (the first DC read on another image; the guard median) the test reports by how much. The admitted
residual's identity (epoch - residual - master) holds; admissions.csv passes its content test; a second run admits
nothing (never twice); without a guard median the stage refuses.
Run: python tests/test_masters_admit6.py
"""
import calendar
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.core import env as ENV                          # noqa: E402
ENV.legacy()
from exopipe.phase3.config import CFG                               # noqa: E402
from exopipe.phase3.core.record import Refusal                      # noqa: E402
from exopipe.phase3.lib.masters import admit as A                   # noqa: E402

T = '/tmp/exopipe3_tests_admit6'
FAILS, N = [], [0]
CELL, LST, SC = '05h_78MHz', '05h', '78MHz'
REPRODUCE_TOL = 1e-3


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def copy_without(nights, root):
    """A copy of the installed cell's records, masters and provisional masters under root/<cell>/ in which ``nights`` -
    members - are out of every record of the build, the stage-2 record carries the thresholds that population gives, the
    population record is recomputed (and passes its test), and the build's records are dated one day before the earliest
    Science dir of those nights (admit.build_time: they qualify as nights Phase 2 first had after the build)."""
    from exopipe.phase3.lib.masters import build_stats as BS, coadd as co
    src, dst = os.path.join(CFG.paths.COADD_SB_ROOT, CELL), os.path.join(root, CELL)
    os.makedirs(os.path.join(dst, 'coadd_final'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'coadd_provisional'), exist_ok=True)
    for rel in [r for r, _c in A.BUILD_RECORDS] + ['coadd_final/stage2_verdict.csv', f'coadd_final/I_deep_Taper_R075_{SC}.fits',
                                                   f'coadd_final/V_deep_Taper_R0_{SC}.fits',
                                                   f'coadd_provisional/I_deep_Taper_R075_{SC}.fits',
                                                   f'coadd_provisional/V_deep_Taper_R0_{SC}.fits']:
        if os.path.exists(os.path.join(src, rel)):
            shutil.copy2(os.path.join(src, rel), os.path.join(dst, rel))
    for rel, col in list(A.BUILD_RECORDS) + [('coadd_final/stage2_verdict.csv', 'date')]:
        p = os.path.join(dst, rel)
        if not os.path.exists(p):
            continue
        rows = list(csv.reader(open(p, newline='')))
        i = rows[0].index(col) if col and rows and col in rows[0] else 2
        keep = [r for r in rows if not (len(r) > i and r[i].strip() in nights)]
        with open(p, 'w', newline='') as f:
            csv.writer(f).writerows(keep)
    s2p, rows2 = BS.stage2_source(dst)
    st2 = BS.stage2_stats(rows2, float(co.OUTLIER_SIGMA), float(co.LOW_OUTLIER_SIGMA))
    rows = list(csv.DictReader(open(s2p, newline='')))
    for r in rows:
        r['i_thresh_mjy'], r['v_thresh_mjy'] = repr(st2['I']['thr_hi']), repr(st2['V']['thr_hi'])
    with open(s2p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    BS.write(dst, BS.compute(dst, LST, SC, co, CFG, 'stage'))
    assert not BS.recheck(os.path.join(dst, 'coadd_final', BS.FILE), co)
    first = min(calendar.timegm(time.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S'))
                for d in nights for x in glob.glob(f'{co.PHASE2_ROOT}/{LST}/{d}/Science_*')
                for m in [re.match(A.SCIENCE_DIR, os.path.basename(x))] if m)
    for rel in (A.BUILD_NIGHTS, 'coadd_final/weights.csv'):
        os.utime(os.path.join(dst, rel), (first - 86400, first - 86400))
    return dst


def cli(step, *a, timeout=3600):
    p = subprocess.run([sys.executable, '-m', 'exopipe.phase3', step, '--tier', 'deep', '--cells', CELL, '--cores', 'none']
                       + list(a), cwd=REPO, env=dict(os.environ), capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def rec(out, step, unit):
    p = os.path.join(out, '_records', step, f'{unit}.json')
    return json.load(open(p)) if os.path.exists(p) else {}


def bright_ratio(e, m):
    """Median and count of e / m at the master's bright compact sources - the selection of coadd._epoch_content_arrays."""
    from scipy.ndimage import maximum_filter
    ok = np.isfinite(e) & np.isfinite(m) & (m != 0)
    v = m[ok]
    rr = 1.4826 * np.median(np.abs(v - np.median(v)))
    mf = np.nan_to_num(m)
    pk = (mf == maximum_filter(mf, size=9)) & (mf > 20 * rr) & (mf < 50)
    yy, xx = np.nonzero(pk)
    h, w = m.shape
    keep = np.hypot(xx - w / 2, yy - h / 2) / (w / 2) < 0.9
    yy, xx = yy[keep], xx[keep]
    r = e[yy, xx] / m[yy, xx]
    r = r[np.isfinite(r)]
    return float(np.median(r)), float(np.median(np.abs(r - 1))), len(r)


def test_admit_end_to_end():
    from astropy.io import fits
    from exopipe.phase3.core import validate
    w = list(csv.DictReader(open(os.path.join(CFG.paths.COADD_SB_ROOT, CELL, 'coadd_final', 'weights.csv'))))
    flows = os.path.join(CFG.paths.COADD_SB_ROOT, CELL, 'flow')
    cands = sorted(r['date'] for r in w if r.get('in_master') == '1' and r.get('qa3_guard') == 'pass'
                   and os.path.exists(os.path.join(flows, f'{r["date"]}_scaled_flow.npy')))
    nights = cands[-2:]
    wt, out = os.path.join(T, 'wt'), os.path.join(T, 'out')
    cd = copy_without(set(nights), wt)
    common = ['--out', out, '--deep-installed-root', wt]
    rc, txt = cli('masters', '--stages', 'admit-prep', *common)
    pr = rec(out, 'masters', f'admit-prep_{CELL}')
    check('admit-prep prepares the two nights made new', rc == 0 and sorted(d['date'] for d in pr['meta']['admit']['dates'])
          == nights, (rc, pr.get('reason'), txt[-400:]))
    for d in nights:                                   # the members' stored flows: fresh, so the flow step solves nothing
        for f in (f'{d}_scaled_flow.npy', f'{d}_scaled_dewarp.fits'):
            shutil.copy2(os.path.join(flows, f), os.path.join(cd, A.ADMIT_DIR, 'flow', f))
            t = time.time()
            os.utime(os.path.join(cd, A.ADMIT_DIR, 'flow', f), (t, t))
    rc, txt = cli('flow', '--stages', 'admit', '--out', out)      # the pairs carry the cell tree's paths
    fr = rec(out, 'flow', f'admit_{CELL}')
    check('the flow step\'s admit stage takes the stored flows (nothing solved, no GPU) and is done',
          rc == 0 and fr.get('status') == 'done' and not (fr.get('meta') or {}).get('admit', {}).get('recomputed'),
          (rc, fr.get('reason'), txt[-400:]))
    sel = _select('admit', nights[0], wt, out)
    check('--epochs (admit): before the admission a prepared night selects its cell', sel == [CELL], sel)
    rc, txt = cli('masters', '--stages', 'admit', *common)
    r0 = rec(out, 'masters', f'admit_{CELL}')
    check('without a guard median (recorded or ruled) the stage refuses - nothing substituted (item 10)',
          rc == 2 and 'item 10' in r0.get('reason', '') and not glob.glob(os.path.join(cd, 'subtracted', '*')),
          (rc, r0.get('reason')))
    rc, txt = cli('masters', '--stages', 'admit', *common, '--set', 'masters_admit.DC2_MEDIAN_WITHOUT_RECORD=1.0')
    r1 = rec(out, 'masters', f'admit_{CELL}')
    adm = os.path.join(cd, 'coadd_final', A.ADMISSIONS)
    rows = list(csv.DictReader(open(adm))) if os.path.exists(adm) else []
    check('with the recommended guard median (1.0, test only) both nights are admitted, one admissions.csv row each',
          rc == 0 and r1.get('status') == 'done' and sorted(r['date'] for r in rows) == nights
          and all(r['verdict'] == 'admitted' for r in rows), (rc, r1.get('reason'), [(r['date'], r['verdict'], r['reason'])
                                                                                       for r in rows], txt[-500:]))
    tests = {t['name']: t['passed'] for t in r1.get('tests', [])}
    check('every product and admissions.csv pass their content tests (identity, own beam, cards; the admission record)',
          any(k.startswith('admitted_epoch') for k in tests) and any(k.startswith('admissions') for k in tests)
          and all(tests.values()), {k: v for k, v in tests.items() if not v} or sorted(tests)[:6])
    inst = os.path.join(CFG.paths.COADD_SB_ROOT, CELL)
    m = np.squeeze(fits.getdata(os.path.join(inst, 'coadd_final', f'I_deep_Taper_R075_{SC}.fits'))).astype(np.float64)
    res = {}
    for d in nights:
        pa = os.path.join(cd, 'subtracted', f'{d}_I_epoch_{SC}.fits')
        if not os.path.exists(pa):
            res[d] = (float('nan'), float('nan'), 0)
            continue
        e_adm = np.squeeze(fits.getdata(pa)).astype(np.float64)
        pe = os.path.join(inst, 'subtracted', f'{d}_I_epoch_{SC}.fits')        # the member's installed epoch
        e_mem = (np.squeeze(fits.getdata(pe)).astype(np.float64) if os.path.exists(pe) else
                 np.squeeze(fits.getdata(os.path.join(inst, 'subtracted', f'{d}_I_sub_{SC}.fits'))).astype(np.float64) + m)
        res[d] = bright_ratio(e_adm, e_mem)
    print('  admission_reproduces (admitted / member epoch at the master\'s bright compact sources): '
          + '; '.join(f'{d}: median {v[0]:.6f}, median |r-1| {v[1]:.2e}, {v[2]} sources' for d, v in res.items()))
    check(f'admission_reproduces: each admitted epoch equals the member\'s at bright peaks within {REPRODUCE_TOL:g}',
          all(abs(v[0] - 1) <= REPRODUCE_TOL and v[2] >= 20 for v in res.values()), res)
    skip = {'HISTORY', 'COMMENT', '', 'ADMITTED', 'ADMITREF', 'GITTAG'}     # GITTAG: the code commit that wrote it
    hdiff = {}
    for d in nights:
        for st in ('I', 'V'):
            for kind in ('epoch', 'sub', 'clean'):
                a = fits.getheader(os.path.join(cd, 'subtracted', f'{d}_{st}_{kind}_{SC}.fits'))
                b = fits.getheader(os.path.join(inst, 'subtracted', f'{d}_{st}_{kind}_{SC}.fits'))
                dk = sorted(k for k in set(a) | set(b) if k not in skip and a.get(k) != b.get(k))
                if dk:
                    hdiff[f'{d}_{st}_{kind}'] = dk[:6]
    check('every admitted product\'s header equals the member\'s card for card (the epochs stamped as the archive '
          'publish stamps them) but ADMITTED / ADMITREF / HISTORY and the code commit (GITTAG)', not hdiff, hdiff)
    for stage, extra in (('admit', ['--set', 'masters_admit.DC2_MEDIAN_WITHOUT_RECORD=1.0']), ('admit-prep', [])):
        rp = os.path.join(out, '_records', 'masters', f'{stage}_{CELL}.json')
        r_ = json.load(open(rp))
        r_['commit'] = 'abc1234'                     # as from a clean tree (a dirty-tree record is never resumed)
        json.dump(r_, open(rp, 'w'))
        rc, txt = cli('masters', '--stages', stage, *common, *extra, '--resume')
        r2 = rec(out, 'masters', f'{stage}_{CELL}')
        check(f'--resume after the admission: {stage} is skipped and its done record kept (seventh review: its own rows '
              f'no longer change its identity)', rc == 0 and 'skipped' in txt and r2.get('status') == 'done'
              and r2.get('run_id') == r_.get('run_id'), (rc, txt[-500:], r2.get('status'), r2.get('reason')))
    sel = _select('admit', nights[0], wt, out)
    check('--epochs (admit): an admitted night selects no cell, and the refusal says why for each named night',
          isinstance(sel, str) and 'no cell selected' in sel and 'admitted or refused already' in sel, sel)
    rc, txt = cli('masters', '--stages', 'admit', *common, '--set', 'masters_admit.DC2_MEDIAN_WITHOUT_RECORD=1.0')
    check('a second run admits nothing (a night is never admitted twice)', rc == 3, (rc, txt[-300:]))
    ok = validate.run('admissions', adm)
    check('admissions.csv content test, run on its own', ok and all(t['passed'] for t in ok), ok)
    return nights



def _select(stage, night, wt, out):
    """deep_stages.select_cells for --epochs <LST>_<night> and one admission stage: the selected cells, or the refusal."""
    import argparse
    from exopipe.phase3.lib.masters import deep_stages as DS, input_resolver as IR
    a = argparse.Namespace(cells=CELL, lsts=None, scopes=None, exclude_cells=None, epochs=f'{LST}_{night}',
                           stages_resolved=[stage], deep_installed_root=wt, out=out)
    try:
        with IR.read_only():
            return DS.select_cells(a, out)
    except Refusal as e:
        return str(e)


def _stack_night():
    """(the functions on the stack, the night admit_stage is on) - for patches that act inside one night only."""
    import sys as _sys
    f, names, night = _sys._getframe(1), [], None
    while f:
        names.append(f.f_code.co_name)
        if f.f_code.co_name == 'admit_stage' and night is None:
            night = f.f_locals.get('d')
        f = f.f_back
    return names, night


def _stash(cd, tag):
    """The test copy's admission outcome moved aside (inside the test's scratch) so its nights qualify again."""
    dst = os.path.join(T, 'stash', tag)
    os.makedirs(dst, exist_ok=True)
    for p in glob.glob(os.path.join(cd, 'subtracted', '20*')) + glob.glob(os.path.join(cd, 'coadd_final', A.ADMISSIONS)):
        os.replace(p, os.path.join(dst, os.path.basename(p)))


def _run_masters(wt, out, stages):
    """The masters step's deep stages in-process through deep_stages.run (its read-only and resume rules), --out ``out``
    - the unit record of the last stage run."""
    import argparse
    from exopipe.phase3.core import record as R
    from exopipe.phase3.lib.masters import deep_stages as DS
    CFG.authorize([], out=out)
    run = R.Run('masters', ['test'], out, CFG, config_sections=())
    run.commit = 'abc1234'
    a = argparse.Namespace(cells=CELL, lsts=None, scopes=None, exclude_cells=None, epochs=None, stages_resolved=list(stages),
                           out=out, deep_installed_root=wt, reference=None, dry_run=False, resume=False, fresh=False)
    DS.run(a, run)
    return run.units[-1]


def _plan(wt, out):
    """admit_stages.plan_admit (the dry-run plan): (lines, gates)."""
    import argparse
    from exopipe.phase3.lib.masters import admit_stages as AS, input_resolver as IR
    a = argparse.Namespace(deep_installed_root=wt, epochs=None)
    with IR.read_only():
        return AS.plan_admit(CELL, a, out, set())


def _run_admit(wt, out):
    """The admit stage in-process (as the masters step calls it) - its unit record."""
    import argparse
    from exopipe.phase3.core import record as R
    from exopipe.phase3.lib.masters import admit_stages as AS
    CFG.authorize([], out=out)
    run = R.Run('masters', ['test'], out, CFG, config_sections=())
    run.commit = 'abc1234'
    args = argparse.Namespace(deep_installed_root=wt, epochs=None, fresh=False, resume=False)
    try:
        AS.admit_stage(CELL, args, run, out, {})
    except Exception as e:                           # noqa: BLE001 - the unit records it; the test reads the record
        print(f'  (admit_stage raised {type(e).__name__}: {e})')
    return run.units[-1]


def test_failure_paths(nights):
    """The admit stage's failure paths in-process on the end-to-end copy (sixth review: only the all-pass path was tested):
    a night failing the epoch check, a night without a V image, a V image the same-state rule refuses, a failure in the
    second night (the first night's row is already written) and the re-run, an interruption between the products and the
    row (the re-run moves the orphans aside and admits the night again), a night re-imaged since admit-prep, a stale flow,
    products of unknown origin, admit-prep's inputs changed (and the dry-run plan saying so) - and from the seventh review:
    the same-state refusal not written into the cell's build record (--out = the cell tree), an error inside the flux
    measurement or the night-image check retried (no row), a V image the build cannot align, a flow failing its own content
    test, the cell lock, the content tests on a changed beam and an orphan - each outcome a recorded refusal or a done unit,
    never an error but where one was forced, and never a product without a row."""
    import sys as _sys
    from astropy.io import fits
    from exopipe.phase3.core import inputs as CI
    from exopipe.phase3.lib.masters import admit_stages as AS, coadd as co
    wt, out = os.path.join(T, 'wt'), os.path.join(T, 'out')
    cd = os.path.join(wt, CELL)
    n1, n2 = nights
    saved = CFG.masters_admit.DC2_MEDIAN_WITHOUT_RECORD
    CFG.masters_admit.DC2_MEDIAN_WITHOUT_RECORD = 1.0                   # the recommendation of item 10, test only

    def rows():
        return {r['date']: r for r in csv.DictReader(open(os.path.join(cd, 'coadd_final', A.ADMISSIONS)))} \
            if os.path.exists(os.path.join(cd, 'coadd_final', A.ADMISSIONS)) else {}

    def prods(d):
        return sorted(os.path.basename(p) for p in glob.glob(os.path.join(cd, 'subtracted', f'{d}_*')))

    def gates(u):
        return {os.path.basename(r['path']): r['gate'] for r in u.refused_items}

    def in_night(d):
        f = _sys._getframe(2)
        return f.f_code.co_name == 'admit_stage' and f.f_locals.get('d') == d
    try:
        # (1) the epoch check fails for night 2
        _stash(cd, 'happy')
        real = co._epoch_content_arrays
        co._epoch_content_arrays = lambda e, m, r, st: ((False, 'forced FAIL (test)', 0.5, 30) if in_night(n2)
                                                        else real(e, m, r, st))
        try:
            u = _run_admit(wt, out)
        finally:
            co._epoch_content_arrays = real
        R_ = rows()
        check('a night failing the epoch check: a "refused" row, every planned product refused by name, none published; '
              'the other night admitted; the unit done (not an error)',
              u.status == 'done' and R_[n2]['verdict'] == 'refused' and 'epoch-image' in R_[n2]['reason']
              and R_[n1]['verdict'] == 'admitted' and not prods(n2) and set(gates(u).values()) == {'admit.epoch'}
              and len(gates(u)) == 6, (u.status, u.reason, gates(u), prods(n2)))
        # (2) no V image for night 2; (3) the same-state rule refuses night 2's V
        for tag, patch_cand in (('nov', True), ('vsame', False)):
            _stash(cd, tag)
            rv, rc_, rm = co.find_stokes_v_deep_image, CI.stokes_v_image, CI.state_mismatch
            n2dirs = [x[1] for x in co.discover_dates(LST, SC[:-3]) if x[0] == n2]
            co.find_stokes_v_deep_image = lambda sd, f: None if sd in n2dirs else rv(sd, f)
            if patch_cand:
                CI.stokes_v_image = lambda sd, f: None if sd in n2dirs else rc_(sd, f)
            else:
                CI.state_mismatch = lambda sd, f, p_, prod: ('forced mismatch (test)' if sd in n2dirs and prod == 'V'
                                                              else rm(sd, f, p_, prod))
            try:
                u = _run_admit(wt, out)
            finally:
                co.find_stokes_v_deep_image, CI.stokes_v_image, CI.state_mismatch = rv, rc_, rm
            R_ = rows()
            ok = (u.status == 'done' and R_[n2]['verdict'] == 'admitted' and not [p for p in prods(n2) if '_V_' in p]
                  and len([p for p in prods(n2) if '_I_' in p]) == 3)
            if patch_cand:
                check('a night without a V image: admitted with its I products only (the build\'s rule), "no V image" '
                      'recorded, the unit done', ok and R_[n2]['v_products'] == 'no V image' and not gates(u),
                      (u.status, u.reason, R_[n2]['v_products'], prods(n2)))
            else:
                check('a V image the same-state rule refuses: its three V products refused by name with the reason, '
                      'recorded on the row; I admitted; the unit done', ok and 'same-state' in R_[n2]['v_products']
                      and set(gates(u).values()) == {'admit.v_same_state'} and len(gates(u)) == 3,
                      (u.status, u.reason, R_[n2]['v_products'], gates(u)))
        # (4) a failure in night 2: night 1's row is already written; the re-run admits night 2
        _stash(cd, 'crash')
        rb = co.beam_gate_image
        co.beam_gate_image = lambda l, f, d, sci=None: ((_ for _ in ()).throw(RuntimeError('forced error (test)'))
                                                        if d == n2 and in_night(n2) else rb(l, f, d, sci=sci))
        try:
            u = _run_admit(wt, out)
        finally:
            co.beam_gate_image = rb
        R1 = rows()
        u2 = _run_admit(wt, out)
        R2 = rows()
        check('a failure partway: the night already done has its row (written per night), the failed one none; the '
              're-run admits the failed night and leaves the done one alone',
              u.status == 'error' and set(R1) == {n1} and R1[n1]['verdict'] == 'admitted' and len(prods(n1)) == 6
              and u2.status == 'done' and set(R2) == {n1, n2} and R2[n2]['verdict'] == 'admitted',
              (u.status, sorted(R1), u2.status, u2.reason, sorted(R2)))
        # (5) interrupted between the products and the row: orphans moved aside, the night admitted again
        _stash(cd, 'orphan')
        rw = AS.write_admissions
        hits = []

        def boom(cd_, rows_):
            if not hits:
                hits.append(1)
                raise OSError('forced interruption before the row (test)')
            return rw(cd_, rows_)
        AS.write_admissions = boom
        try:
            u = _run_admit(wt, out)
        finally:
            AS.write_admissions = rw
        orphans, R_mid = prods(n1), rows()
        u2 = _run_admit(wt, out)
        R2 = rows()
        aside = glob.glob(os.path.join(cd, 'subtracted', '_superseded', '*', f'{n1}_I_epoch_{SC}.fits'))
        check('an interruption between the products and the row leaves orphan products (ADMITTED, no row) - and the '
              're-run moves them aside (never deleted), admits the night again and both nights get rows',
              u.status == 'error' and len(orphans) == 6 and R_mid == {} and u2.status == 'done' and aside
              and R2.get(n1, {}).get('verdict') == 'admitted' and R2.get(n2, {}).get('verdict') == 'admitted'
              and len(prods(n1)) == 6, (u.status, len(orphans), sorted(R_mid), u2.status, u2.reason, aside, sorted(R2)))
        # (6) night 2 re-imaged since admit-prep: refused by name; night 1 admitted
        _stash(cd, 'reimage')
        rd = co.discover_dates
        newroot = os.path.join(T, 'reimaged', LST, n2, 'Science_20991231_000000')

        def disc(l, f):
            out_ = []
            for d, sd, img in rd(l, f):
                if d == n2:
                    dst = os.path.join(newroot, os.path.relpath(img, sd))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if not os.path.exists(dst):
                        shutil.copy2(img, dst)
                    out_.append((d, newroot, dst))
                else:
                    out_.append((d, sd, img))
            return out_
        co.discover_dates = disc
        try:
            u = _run_admit(wt, out)
        finally:
            co.discover_dates = rd
        R_ = rows()
        check('a night re-imaged since admit-prep: its products refused by name ("re-imaged since"), none published, no row '
              '(it is retried after admit-prep); the other night admitted; the unit done',
              u.status == 'done' and n2 not in R_ and not prods(n2) and set(gates(u).values()) == {'admit.reimaged'}
              and R_.get(n1, {}).get('verdict') == 'admitted', (u.status, u.reason, gates(u), sorted(R_)))
        # (7) a stale flow (older than its survey image: the build's freshness gate)
        _stash(cd, 'stale')
        fp = os.path.join(cd, A.ADMIT_DIR, 'flow', f'{n2}_scaled_flow.npy')
        t0 = os.path.getmtime(fp)
        os.utime(fp, (1e9, 1e9))
        try:
            u = _run_admit(wt, out)
        finally:
            os.utime(fp, (t0, t0))
        R_ = rows()
        check('a flow older than its survey image (the build\'s freshness gate): the night refused by name, no row, the '
              'unit done', u.status == 'done' and n2 not in R_ and set(gates(u).values()) == {'admit.flow'},
              (u.status, u.reason, gates(u)))
        # (8) products of unknown origin
        _stash(cd, 'unknown')
        junk = os.path.join(cd, 'subtracted', f'{n2}_I_sub_{SC}.fits')
        fits.writeto(junk, np.zeros((2, 2), np.float32))
        u = _run_admit(wt, out)
        R_ = rows()
        check('products no admission made: never replaced - the night refused by name, the file kept',
              u.status == 'done' and n2 not in R_ and set(gates(u).values()) == {'admit.exists'} and os.path.exists(junk),
              (u.status, u.reason, gates(u)))
        os.remove(junk)
        # (9) admit-prep's inputs changed since it ran
        _stash(cd, 'changed')
        rp_ = os.path.join(cd, A.BUILD_NIGHTS)
        orig = open(rp_, 'rb').read()                 # restored byte for byte (seventh review: text mode lost the CRLFs)
        t0 = os.path.getmtime(rp_)
        with open(rp_, 'ab') as f:
            f.write(b'\n')
        try:
            u = _run_admit(wt, out)
            plan_lines, plan_gates = _plan(wt, out)
        finally:
            open(rp_, 'wb').write(orig)
            os.utime(rp_, (t0, t0))
        check('admit-prep\'s inputs changed since it ran: the unit refused before any night, nothing written; the dry-run '
              'plan says the same (one definition of the gates)',
              u.status == 'refused' and 'changed since it ran' in (u.reason or '') and not rows()
              and any('changed since it ran' in g for g in plan_gates), (u.status, u.reason, plan_gates))
        ok_lines, ok_gates = _plan(wt, out)
        check('the dry-run plan of the restored copy: no gate, both nights with their I and V products',
              not ok_gates and sum('{I,V}' in ln for ln in ok_lines) == 2, (ok_gates, ok_lines))
        # (10) MAJOR (seventh review): with --out = the cell tree root, night 2's V refused by the same-state rule is NOT
        #      written into the cell's build record (refusals_input.csv) - the admission's own row holds the refusal - and
        #      the night is admitted with its I products, the stage run as the masters step runs it (deep_stages.run)
        _stash(cd, 'readonly')
        ri = os.path.join(cd, 'refusals_input.csv')
        before = open(ri, 'rb').read() if os.path.exists(ri) else None
        if not os.path.isdir(os.path.join(wt, '_records')):
            shutil.copytree(os.path.join(out, '_records'), os.path.join(wt, '_records'))
        rm = CI.state_mismatch
        n2dirs = [x[1] for x in co.discover_dates(LST, SC[:-3]) if x[0] == n2]
        CI.state_mismatch = lambda sd, f, p_, prod: ('forced mismatch (test)' if sd in n2dirs and prod == 'V'
                                                      else rm(sd, f, p_, prod))
        try:
            u = _run_masters(wt, wt, ['admit'])
        finally:
            CI.state_mismatch = rm
        R_ = rows()
        after = open(ri, 'rb').read() if os.path.exists(ri) else None
        check('MAJOR (seventh review): with --out = the cell tree root the same-state refusal is not written into the cell\'s '
              'build record; the night is admitted with its I products, the refusal on its row',
              after == before and u.status == 'done' and R_.get(n2, {}).get('verdict') == 'admitted'
              and 'same-state' in R_.get(n2, {}).get('v_products', '') and not [p for p in prods(n2) if '_V_' in p],
              (u.status, u.reason, after is None, R_.get(n2, {}).get('v_products')))
        # (11) MAJOR (seventh review): an error inside the flux-scale measurement (disk full) is no verdict - the night is
        #      refused as an item (retried), no row; the next run admits it
        for tag, mod, name in (('err_flux', co, 'measure_flux_scale'), ('err_gate', None, 'evaluate')):
            _stash(cd, tag)
            from exopipe.phase3.lib.masters import night_image_gate as NG
            mod = mod or NG
            real_f = getattr(mod, name)
            where = '_night_flux_dc' if name == 'measure_flux_scale' else 'final_night'

            def failing(*a, _f=real_f, _w=where, **k):
                names, night = _stack_night()
                if _w in names and night == n2:
                    raise OSError(28, 'No space left on device (test)')
                return _f(*a, **k)
            setattr(mod, name, failing)
            try:
                u = _run_admit(wt, out)
            finally:
                setattr(mod, name, real_f)
            R1 = rows()
            u2 = _run_admit(wt, out)
            R2 = rows()
            check(f'MAJOR (seventh review): an error inside {"the flux-scale measurement" if name != "evaluate" else "the night-image check"} '
                  f'is no verdict: the night refused as an item (retried), no row; the next run admits it',
                  u.status == 'done' and n2 not in R1 and set(gates(u).values()) == {'admit.error'}
                  and R1.get(n1, {}).get('verdict') == 'admitted' and u2.status == 'done'
                  and R2.get(n2, {}).get('verdict') == 'admitted', (u.status, u.reason, gates(u), sorted(R1), u2.status,
                                                                     sorted(R2)))
        # (12) a V image the build cannot align (process_variant_image returns False): I admitted, V refused by name
        _stash(cd, 'valign')
        rpv = co.process_variant_image

        def no_v(*a, **k):
            names, night = _stack_night()
            if names[1:2] == ['admit_stage'] and night == n2 and a and '-V-' in os.path.basename(str(a[0])):
                return False
            return rpv(*a, **k)
        co.process_variant_image = no_v
        try:
            u = _run_admit(wt, out)
        finally:
            co.process_variant_image = rpv
        R_ = rows()
        check('a V image the build cannot align: the night admitted with its I products (the build\'s rule), its V products '
              'refused by name, the reason on the row; the unit done (seventh review: it errored on every run)',
              u.status == 'done' and R_.get(n2, {}).get('verdict') == 'admitted'
              and R_[n2]['v_products'].startswith('refused (alignment') and set(gates(u).values()) == {'admit.v_alignment'}
              and len(gates(u)) == 3 and len([p for p in prods(n2) if '_I_' in p]) == 3
              and not [p for p in prods(n2) if '_V_' in p], (u.status, u.reason, gates(u), R_.get(n2, {}).get('v_products')))
        # (13) the flow's own content test (admit-prep's inputs unchanged, the flow newer than its input)
        _stash(cd, 'flowtest')
        fp = os.path.join(cd, A.ADMIT_DIR, 'flow', f'{n2}_scaled_flow.npy')
        keep = fp + '.keep'
        shutil.copy2(fp, keep)
        np.save(fp, np.zeros((4, 4, 2), np.float32))
        try:
            u = _run_admit(wt, out)
        finally:
            os.replace(keep, fp)
            t = time.time()
            os.utime(fp, (t, t))
        R_ = rows()
        check('a flow failing its own content test (admit-prep\'s inputs unchanged, newer than its input): the night '
              'refused by name, no row; the other admitted', u.status == 'done' and n2 not in R_
              and set(gates(u).values()) == {'admit.flow'} and R_.get(n1, {}).get('verdict') == 'admitted',
              (u.status, u.reason, gates(u)))
        # (14) the cell lock: a second admission run on the cell is refused, nothing written
        _stash(cd, 'lock')
        lk = AS.cell_lock(os.path.join(cd, A.ADMIT_DIR))
        try:
            u = _run_admit(wt, out)
        finally:
            lk.close()
        check('the cell lock: another admit run holds it - this one refused before any night, nothing written',
              u.status == 'refused' and 'holds' in (u.reason or '') and not rows(), (u.status, u.reason))
        # (15) the content tests: an admitted product whose beam differs from its own source fails admitted_header; an
        #      ADMITTED product without a row fails admissions_record (the orphan scan)
        _stash(cd, 'content')
        u = _run_admit(wt, out)
        from exopipe.phase3.core import validate
        pi = os.path.join(cd, 'subtracted', f'{n1}_I_sub_{SC}.fits')
        keep = pi + '.keep'
        shutil.copy2(pi, keep)
        with fits.open(pi, mode='update') as h:
            h[0].header['BMAJ'] = float(h[0].header['BMAJ']) * 1.5
        try:
            res = {t['name']: t['passed'] for t in validate.run('admitted_epoch', pi)}
        finally:
            os.replace(keep, pi)
        orphan = os.path.join(cd, 'subtracted', f'2099-01-01_I_sub_{SC}.fits')
        shutil.copy2(pi, orphan)
        try:
            res2 = validate.run('admissions', os.path.join(cd, 'coadd_final', A.ADMISSIONS))
        finally:
            os.remove(orphan)
        check('content tests: a beam that differs from its own source fails admitted_header; an ADMITTED product with no '
              'row fails the admissions record (the orphan scan)',
              u.status == 'done' and res.get('admitted_epoch.admitted_header') is False
              and not all(t['passed'] for t in res2) and any('2099-01-01' in str(t.get('detail')) for t in res2),
              (u.status, res, [(t['name'], t['passed'], str(t.get('detail'))[:120]) for t in res2]))
    finally:
        CFG.masters_admit.DC2_MEDIAN_WITHOUT_RECORD = saved


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        nights = test_admit_end_to_end()
        test_failure_paths(nights)
    finally:
        if not os.environ.get('EXOPIPE3_TEST_KEEP'):            # set to inspect the admission copy afterwards
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
