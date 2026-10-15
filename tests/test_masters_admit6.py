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
import inspect
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
    """The test copy's admission outcome cleared so its nights qualify again: this test's own scratch only - the products
    and admissions.csv of the scenario before, and the files the stage moved aside in its earlier runs (a scenario checks
    its own before the next one starts) - removed, so the test's disk stays that of one scenario (2026-10-15: kept, they
    took 29 GB of the node's root filesystem)."""
    for p in glob.glob(os.path.join(cd, 'subtracted', '20*')) + glob.glob(os.path.join(cd, 'coadd_final', A.ADMISSIONS)):
        os.remove(p)
    for d in (os.path.join(cd, 'subtracted'), os.path.join(cd, A.ADMIT_DIR, 'prep_final'),
              os.path.join(cd, A.ADMIT_DIR, 'sub_native'), os.path.join(cd, A.ADMIT_DIR, 'prep')):
        shutil.rmtree(os.path.join(d, '_superseded'), ignore_errors=True)


def _run_masters(wt, out, stages, resume=False):
    """The masters step's deep stages in-process through deep_stages.run (its read-only and resume rules), --out ``out``
    - the unit record of the last stage run."""
    import argparse
    from exopipe.phase3.core import record as R
    from exopipe.phase3.lib.masters import deep_stages as DS
    CFG.authorize([], out=out)
    run = R.Run('masters', ['test'], out, CFG, config_sections=())
    run.commit = 'abc1234'
    a = argparse.Namespace(cells=CELL, lsts=None, scopes=None, exclude_cells=None, epochs=None, stages_resolved=list(stages),
                           out=out, deep_installed_root=wt, reference=None, dry_run=False, resume=resume, fresh=False)
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


# Broad try/excepts in the co-add functions an admission runs (the admit stage and admit-prep, traced 2026-10-15 with
# sys.setprofile on this test's end-to-end copy), by function: what each does with the exception. 'caught': it turns the
# exception into a value and reports it (coadd._caught - the watch makes the night an error); 'raise' re-raises;
# 'fallback': fallback.site (the B-30 policy: refuse or count); 'problem': returns the problem for its caller to refuse on;
# 'parse': tries the next header key (the last resort is a fallback site); 'cosmetic': a header comment or the derived
# MJD-MID card (the products' header test compares every card with the member's).
INVENTORY = {'_night_flux_dc': ['caught'], 'measure_flux_scale': ['caught', 'caught'], 'process_variant_image': ['caught'],
             'beam_area': ['caught'], 'final_night': ['caught'], 'save_epoch_residual': ['raise'],
             '_epoch_beam_source_values': ['problem'], '_grid_refused_dates': ['fallback'], '_off_grid': ['fallback'],
             'build_striation_model': ['fallback'], 'epoch_time_header': ['fallback', 'cosmetic', 'cosmetic'],
             'get_midpoint_time': ['parse'], 'prep_night': [], 'measure_scale2': [], 'subtract_night': [], 'regrid_image': [],
             'get_uniform_rms': [], 'load_image': [], 'save_image': [], 'compute_residual_rms': [], '_epoch_content_arrays': [],
             'epoch_standard_header': [], 'beam_gate_image': []}


def inventory_problems(co):
    """[problem] where coadd.py's broad excepts in the INVENTORY functions differ from it (a new or changed one)."""
    import ast
    src = inspect.getsource(co)
    tree = ast.parse(src)
    out = []
    seen = set()
    for fn in tree.body:
        if not (isinstance(fn, ast.FunctionDef) and fn.name in INVENTORY):
            continue
        seen.add(fn.name)
        got = []
        for h in ast.walk(fn):
            if isinstance(h, ast.ExceptHandler) and (h.type is None or ast.get_source_segment(src, h.type) in
                                                      ('Exception', 'BaseException')):
                body = '\n'.join(ast.get_source_segment(src, b) or '' for b in h.body)
                got.append('caught' if '_caught(' in body else 'raise' if any(isinstance(b, ast.Raise) and b.exc is None
                                                                              for b in h.body)
                           else 'fallback' if 'fallback.site(' in body else 'problem' if body.lstrip().startswith('return {},')
                           else 'parse' if body.strip() == 'continue' and fn.name == 'get_midpoint_time'
                           else 'cosmetic' if fn.name == 'epoch_time_header' and body.strip() == 'pass' else body[:60])
        if sorted(got) != sorted(INVENTORY[fn.name]):
            out.append((fn.name, got, INVENTORY[fn.name]))
    out += [(name, 'not found in coadd.py') for name in INVENTORY if name not in seen]
    return out


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
            rec1 = json.load(open(os.path.join(out, '_records', 'masters', f'admit_{CELL}.json')))
            u2 = _run_masters(wt, out, ['admit'], resume=True)        # eighth review: --resume retries it, never skips
            R2 = rows()
            check(f'MAJOR (seventh, eighth review): an error inside {"the flux-scale measurement" if name != "evaluate" else "the night-image check"} '
                  f'is no verdict: the night refused as an item, no row, not in the unit\'s finished nights; a --resume run '
                  f'retries it (not skipped) and admits it',
                  u.status == 'done' and n2 not in R1 and set(gates(u).values()) == {'admit.error'}
                  and R1.get(n1, {}).get('verdict') == 'admitted' and rec1.get('admitted') == [n1]
                  and u2.status == 'done' and R2.get(n2, {}).get('verdict') == 'admitted',
                  (u.status, u.reason, gates(u), sorted(R1), rec1.get('admitted'), u2.status, u2.reason, sorted(R2)))
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
        # (16) MAJOR (eighth review): no admission into a cell without the provisional masters its stage-2 values need
        from exopipe.phase3.lib.masters import admit_stages as AS2
        pdir = os.path.join(cd, 'coadd_provisional')
        for st_, v_ in AS2.PROV:
            pp = os.path.join(pdir, f'{v_}_{SC}.fits')
            os.replace(pp, pp + '.held')
            try:
                why_ = AS2.population_problem(cd, CELL, co)
            finally:
                os.replace(pp + '.held', pp)
            check(f'MAJOR (eighth review): without the provisional {st_} master the cell admits no night (its stage-2 {st_} '
                  f'value - the search-tier gate\'s - could not be measured)', why_ and f'provisional {st_} master' in why_, why_)
        # (17) MAJOR (ninth review): every broad try/except of the co-add on an admission's path is known - the inventory of
        #      the functions the admit stage and admit-prep run (scratchpad trace, 2026-10-15) - and each one that turns an
        #      exception into a value reports it (coadd._caught); a new one fails here until it is classified
        inv = inventory_problems(co)
        check('every broad try/except in the co-add functions on an admission\'s path re-raises, refuses through the fallback '
              'policy (B-30), returns a problem for its caller, is header parsing or cosmetics, or reports to the error watch '
              '(the six that turn an exception into a value)', not inv, inv)
        # (18) MINOR (eighth review): a record of admissions into another master counts for nothing and is moved aside
        adm_p = os.path.join(cd, 'coadd_final', A.ADMISSIONS)
        rr = list(csv.DictReader(open(adm_p, newline='')))
        for r_ in rr:
            r_['reference_md5'] = '0' * 32
        with open(adm_p, 'w', newline='') as f_:
            dw_ = csv.DictWriter(f_, fieldnames=list(rr[0].keys()))
            dw_.writeheader()
            dw_.writerows(rr)
        before = A.admitted_nights(cd)
        u18 = _run_admit(wt, out)
        R18 = rows()
        sup = glob.glob(os.path.join(cd, 'coadd_final', 'admissions.superseded_*.csv'))
        check('MINOR (eighth review): rows made into another master block no night (never twice is per master); the stage '
              'moves that record aside (kept) and admits the nights into this master again',
              before == set() and u18.status == 'done' and sup and all(R18.get(d, {}).get('verdict') == 'admitted' for d in nights),
              (sorted(before), u18.status, u18.reason, sup, sorted(R18)))
        # (19) MAJOR (ninth review): an elevation transform failing for every source (an IERS table that cannot be refreshed)
        #      is swallowed per source inside measure_flux_scale - the scale then NaN, once a refusal row for good. Now an error:
        #      no row, the night retried, and the next run admits it
        _stash(cd, 'err_altaz')
        ra_ = co.AltAz

        class _NoAltAz:                              # astropy's own error when the IERS-A table is stale and cannot be fetched
            def __init__(self, *a, **k):
                raise ValueError('interpolating from IERS_Auto using predictive values that are more than 30.0 days old (test)')
        co.AltAz = _NoAltAz
        try:
            u19 = _run_admit(wt, out)
        finally:
            co.AltAz = ra_
        R19 = rows()
        u19b = _run_admit(wt, out)
        R19b = rows()
        check('MAJOR (ninth review): an elevation transform failing for every source is an error, not "flux scale not '
              'measurable": both nights refused as items (admit.error, the caught exception named), no row; the next run '
              'admits them', u19.status in ('done', 'refused') and not R19 and set(gates(u19).values()) == {'admit.error'}
              and any('IERS_Auto' in (r.get('reason') or '') for r in u19.refused_items)
              and all(R19b.get(d, {}).get('verdict') == 'admitted' for d in nights),
              (u19.status, u19.reason, gates(u19), sorted(R19), [r.get('reason', '')[:120] for r in u19.refused_items][:2],
               u19b.status, sorted(R19b)))
        # (20) the same swallowed only in the second-pass scale (after the alignment passed): an error at the second check
        _stash(cd, 'err_scale2')
        rms2 = co.measure_scale2

        def scale2_no_altaz(*a, **k):
            co.AltAz = _NoAltAz
            try:
                return rms2(*a, **k)
            finally:
                co.AltAz = ra_
        co.measure_scale2 = scale2_no_altaz
        try:
            u20 = _run_admit(wt, out)
        finally:
            co.measure_scale2, co.AltAz = rms2, ra_
        R20 = rows()
        check('an exception swallowed in the second-pass scale (the night aligned and checked) is an error too: no row, no '
              'product published, retried', not R20 and set(gates(u20).values()) == {'admit.error'}
              and not any(prods(d) for d in nights), (u20.status, gates(u20), sorted(R20), [prods(d) for d in nights]))
        u20b = _run_admit(wt, out)
        check('... and the next run admits both nights', all(rows().get(d, {}).get('verdict') == 'admitted' for d in nights),
              (u20b.status, u20b.reason, sorted(rows())))
        # (21) admit-prep (ninth review, MINOR): an error while a night is prepared is retried (not in the unit's finished
        #      nights); a night refused by a verdict on its own image (off the grid) is finished - --resume skips the unit
        _stash(cd, 'prep')
        co.AltAz = _NoAltAz
        try:
            u21 = _run_masters(wt, out, ['admit-prep'])
        finally:
            co.AltAz = ra_
        r21 = json.load(open(os.path.join(out, '_records', 'masters', f'admit-prep_{CELL}.json')))
        check('admit-prep: an error while a night is prepared (an elevation transform failing) refuses it as an item '
              '(admit-prep.error) and leaves it out of the unit\'s finished nights (retried)',
              set(gates(u21).values()) == {'admit-prep.error'} and not r21.get('admitted'),
              (u21.status, u21.reason, gates(u21), r21.get('admitted')))
        real_off = AS2.off_grid
        AS2.off_grid = lambda d, img, ts, cut: (1, 1) if d == n2 else real_off(d, img, ts, cut)
        try:
            u21b = _run_masters(wt, out, ['admit-prep'])
            r21b = json.load(open(os.path.join(out, '_records', 'masters', f'admit-prep_{CELL}.json')))
            u21c = _run_masters(wt, out, ['admit-prep'], resume=True)
            r21c = json.load(open(os.path.join(out, '_records', 'masters', f'admit-prep_{CELL}.json')))
        finally:
            AS2.off_grid = real_off
        check('MINOR (ninth review): a night off the master\'s grid is a verdict on its own image - in the unit\'s finished '
              'nights with the prepared one, so --resume skips the unit (its prep files and the flow record stay current)',
              set(gates(u21b).values()) == {'admit-prep.wrong_grid'} and r21b.get('admitted') == sorted(nights)
              and r21c.get('run_id') == r21b.get('run_id'),
              (u21b.status, gates(u21b), r21b.get('admitted'), u21c.status, getattr(u21c, 'reason', ''),
               r21c.get('run_id') == r21b.get('run_id')))
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
