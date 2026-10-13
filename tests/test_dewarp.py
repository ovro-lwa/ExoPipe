#!/usr/bin/env python3
"""STANDING TEST - `exopipe3 dewarp` (steps/dewarp.py, lib/dewarp/steps_*.py, lib/dewarp/tests.py, _cfgdraft_dewarp.py).

Synthetic part (a fresh directory /tmp/exopipe3_tests_dewarp/<pid>, removed at the end; every step command runs with --out in
it, the apply/verify/record run with --parity):
  - acceptance function, stage grammar, config protection (in-place keys protected, RULED IDs in the fix ledger)
  - gates: G-D0 (in-place values), G-D0b (the content-test child's production modules vs the branch; a modified copy refuses),
    G-D5 (excluded constituent, against the real data_decisions.csv, read only), G-D8 (threads), G-W (production roots)
  - content tests dewarp_table and dewarp_screen on synthetic files (pass and fail cases)
  - `exopipe3 dewarp --stages stage,measure` on a synthetic LST (staging md5-verified, phase3_master_dewarp.run_lst in place;
    no PyBDSF needed: no anchor-candidate master exists, so every row is a table outcome). Run through cli.main IN PROCESS
    with cascade_dewarp.load_vlssr_catalog stubbed: the real parse of FullVLSSCatalog.text takes ~95 s and the catalogue is
    never used when no anchor master exists (the real load is NOT covered here)
  - `exopipe3 dewarp --stages apply,verify,record --measurement M --parity` on five synthetic cells: accepted (cubic apply in
    place, D1-D6 by check_mdewarp_gen.py through core.legacy.run, the child's code checked), not accepted (applied=0 row,
    install's class screen_table + inline row check), accepted-but-apply-refused (G-R2), not routed (G-D2), input carries
    MDWARP (G-D7)
  - `exopipe3 verify dewarped_master <gen>` takes its context from the run record (scratch mode: the generation's sidecars are
    untouched); a corrupted product fails D3; G-A3 refuses a second apply; --fresh removes and records the old generation
Read-only part (production products, never written): dewarp_table on two production measurement tables, dewarp_screen on a
production screen, the light dewarped_master tests on the installed 12h_Red generation (skipped when absent).

Run standalone:  python tests/test_dewarp.py      (py38_orca_nkosogor; EXOPIPE3_SITE set)
"""
import csv
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')
SCRATCH = os.path.join('/tmp', 'exopipe3_tests_dewarp', str(os.getpid()))
PY = sys.executable
LST = '05h'
N = 128
CONSTITUENT = '2030-01-01'       # a night no data_decisions row names

import numpy as np                                          # noqa: E402
from astropy.io import fits                                 # noqa: E402

from exopipe.phase3.core import env as _env                # noqa: E402
_env.legacy()
from exopipe.phase3.config import CFG, ConfigRefusal        # noqa: E402
from exopipe.phase3.lib.dewarp import steps_common as SC    # noqa: E402
from exopipe.phase3.lib.dewarp import steps_measure as SM   # noqa: E402
from exopipe.phase3.lib.dewarp import tests as T            # noqa: E402

RESULTS = []


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f'  ({detail})' if detail else ''), flush=True)


def cli(*args, timeout=600):
    env = dict(os.environ, PYTHONPATH=REPO)
    t0 = time.time()
    p = subprocess.run([PY, '-m', 'exopipe.phase3'] + list(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       universal_newlines=True, env=env, cwd='/tmp', timeout=timeout)
    return p.returncode, p.stdout, round(time.time() - t0, 1)


def cli_in_process(*args):
    """cli.main in this process with cascade_dewarp.load_vlssr_catalog stubbed (see the module docstring); every config
    override and thread variable it sets is undone afterwards."""
    import io
    from contextlib import redirect_stdout, redirect_stderr
    from exopipe.phase2 import cascade_dewarp as CD
    from exopipe.phase3 import cli as CLI
    saved_env = {k: os.environ.get(k) for k in SC.THREAD_VARS + ('NUMEXPR_NUM_THREADS', 'BDSF_NCORES')}
    orig = CD.load_vlssr_catalog
    CD.load_vlssr_catalog = lambda path, name: {'stub': True, 'path': path}
    buf = io.StringIO()
    t0 = time.time()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            rc = CLI.main(list(args))
    finally:
        CD.load_vlssr_catalog = orig
        CFG.reset()
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return rc, buf.getvalue(), round(time.time() - t0, 1)


def rec(out, unit):
    p = os.path.join(out, '_records', 'dewarp', f'{unit}.json')
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------------------------------------------------------------------------------------------------------------
# synthetic products
# ---------------------------------------------------------------------------------------------------------------------

def header(scope, **cards):
    h = fits.Header()
    for k, v in (('CTYPE1', 'RA---SIN'), ('CTYPE2', 'DEC--SIN'), ('CRVAL1', 180.0), ('CRVAL2', 40.0), ('CRPIX1', N / 2 + 1),
                 ('CRPIX2', N / 2 + 1), ('CDELT1', -0.25), ('CDELT2', 0.25), ('CUNIT1', 'deg'), ('CUNIT2', 'deg'),
                 ('BMAJ', 1.0), ('BMIN', 1.0), ('BPA', 0.0), ('FLUXBAKE', True), ('FLUXVER', 'fluxcorr-v2-20261001-test'),
                 ('SHSCALE', 1.0)):
        h[k] = v
    h['HISTORY'] = f'COADD constituent {CONSTITUENT}'
    for k, v in cards.items():
        h[k] = v
    return h


def sky(seed, shape=(N, N)):
    rng = np.random.default_rng(seed)
    d = rng.normal(0, 1.0, shape).astype(np.float32)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    for _ in range(16):
        y0, x0 = rng.uniform(26, shape[0] - 26), rng.uniform(26, shape[1] - 26)
        d += (200.0 * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * 1.5 ** 2))).astype(np.float32)
    return d


def pb_map():
    yy, xx = np.mgrid[:N, :N]
    return np.exp(-((yy - N / 2) ** 2 + (xx - N / 2) ** 2) / (2 * 40.0 ** 2)).astype(np.float32)


def write(path, data, hdr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.writeto(path, data, hdr, overwrite=True)


def make_generation(base, cell, seed, bad_shape=False, mdwarp=False):
    """A cell generation like a promoted rebuild: coadd_final (4 I + 1 V plains, 2 PB maps, 5 pbcorr siblings, one LOO
    master) and subtracted (one I and one V residual)."""
    lst, s = cell.split('_', 1)
    cf = os.path.join(base, cell, 'coadd_final')
    pb = pb_map()
    extra = {'MDWARP': True} if mdwarp else {}
    plains = {}
    for i, name in enumerate((f'I_deep_Taper_R075_{s}', f'I_deep_Taper_R0_{s}', f'I_deep_NoTaper_R075_{s}',
                              f'I_deep_NoTaper_R0_{s}', f'V_deep_Taper_R0_{s}')):
        d = sky(seed + i)
        plains[name] = d
        write(os.path.join(cf, name + '.fits'), d, header(s, **extra))
    write(os.path.join(cf, f'PB_deep_{s}.fits'), pb, header(s))
    write(os.path.join(cf, f'PB_deep_R075_{s}.fits'), pb, header(s))
    for name, d in plains.items():
        sib = np.where(pb > 0.1, d / pb, np.nan).astype(np.float32)
        write(os.path.join(cf, name.replace(f'_{s}', f'_pbcorr_{s}', 1) + '.fits'), sib, header(s, **extra))
    write(os.path.join(cf, 'loo', f'2025-01-01_I_loo_Taper_R075_{s}.fits'), sky(seed + 10), header(s))
    sub = os.path.join(base, cell, 'subtracted')
    write(os.path.join(sub, f'2025-01-01_I_sub_{s}.fits'), sky(seed + 11), header(s))
    write(os.path.join(sub, f'2025-01-01_V_sub_{s}.fits'), sky(seed + 12), header(s))
    if bad_shape:
        write(os.path.join(sub, f'2025-01-02_I_sub_{s}.fits'), sky(seed + 13, (64, 64)), header(s))
    return os.path.join(base, cell)


def sibling_names(s):
    return [f'I_deep_Taper_R075_pbcorr_{s}', f'I_deep_Taper_R0_pbcorr_{s}', f'I_deep_NoTaper_R075_pbcorr_{s}',
            f'I_deep_NoTaper_R0_pbcorr_{s}', f'V_deep_Taper_R0_pbcorr_{s}']


def table_row(scope, status, **kw):
    from exopipe.phase3.lib.dewarp.tests import md_cols
    r = {c: '' for c in md_cols()}
    r.update(lst=LST, scope=scope, ref='-', status=status, n_written=0, measured_at='2026-10-13T00:00:00Z')
    r.update({k: v for k, v in kw.items()})
    return r


def write_table(path, rows):
    from exopipe.phase3.lib.dewarp.tests import md_cols
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=md_cols())
        w.writeheader()
        w.writerows(rows)


def save_screen(mdir, scope, nanch, master):
    from astropy.wcs import WCS
    from exopipe.phase2 import cascade_dewarp as CD
    sx = np.full((N, N), 1.0, np.float32)
    sy = np.full((N, N), -2.0, np.float32)
    CD._save_warp_screen(sx, sy, WCS(fits.getheader(master)).celestial, scope, os.path.join(mdir, LST), 'rbf',
                         'multiquadric', nanch, logging.getLogger('test_dewarp'))
    return os.path.join(mdir, LST, 'maps', f'{scope}_warp_screen.fits')


# ---------------------------------------------------------------------------------------------------------------------
# unit-level tests
# ---------------------------------------------------------------------------------------------------------------------

def test_acceptance_and_stages():
    A = SC.accepted
    check('accepted: OK and gain 5.0 accepted', A({'status': 'OK', 'anchor_off_med_as': '18.0', 'cv_resid_med_as': '13.0'}, 5.0))
    check('accepted: gain 4.9 not accepted', not A({'status': 'OK', 'anchor_off_med_as': '17.9', 'cv_resid_med_as': '13.0'}, 5.0))
    check('accepted: REFERENCE ONLY not accepted',
          not A({'status': 'REFERENCE ONLY: x', 'anchor_off_med_as': '30', 'cv_resid_med_as': '1'}, 5.0))
    check('accepted: missing row / unparsable not accepted',
          not A(None, 5.0) and not A({'status': 'OK', 'anchor_off_med_as': '', 'cv_resid_med_as': '1'}, 5.0))
    check('stages: default all five', SC.parse_stages(None) == list(SC.STAGES))
    check('stages: measure implies stage', SC.parse_stages('measure') == ['stage', 'measure'])
    check('stages: kept in pipeline order', SC.parse_stages('record,apply') == ['apply', 'record'])
    for bad in ('stage', 'bogus'):
        try:
            SC.parse_stages(bad)
            check(f'stages: {bad!r} refused', False)
        except ValueError:
            check(f'stages: {bad!r} refused', True)
    st = SC.render_not_accepted(None)
    check('applied=0 status of a cell without a row says not measured', 'status not measured' in st and '? -> ?' in st, st)
    long = SC.render_not_accepted({'status': 'REFUSED regression: ' + 'x' * 200, 'anchor_off_med_as': '1',
                                   'cv_resid_med_as': '2'})
    check('applied=0 status cut to 120 characters', len(long) == 120)


def test_config():
    c = CFG.dewarp
    check('config: dewarp section loads from the draft', len(c.items()) >= 70, f'{len(c.items())} keys')
    missing = [k for k in SC.IN_PLACE if k not in c.PROTECTED_KEYS]
    check('config: every in-place key is in PROTECTED_KEYS', not missing, str(missing))
    try:
        CFG.set('dewarp.HELDOUT_MIN_GAIN_AS', '4.0')
        check('config: override of an in-place key refused', False)
    except ConfigRefusal:
        check('config: override of an in-place key refused', True)
    finally:
        CFG.reset()
    from exopipe.phase3.core import ledger
    ids = sorted({x for v in c.RULED.values() for x in ((v,) if isinstance(v, str) else v)})
    absent = [i for i in ids if not ledger.has_id(i, CFG.paths.FIX_LEDGER)]
    check('config: every RULED ledger ID is a fix_ledger row', not absent, f'{len(ids)} IDs; absent {absent}')
    check('config: shared roots from paths (no own WORKING_ROOT / SCREEN_RECORD_RUN)',
          not hasattr(c, 'WORKING_ROOT') and not hasattr(c, 'SCREEN_RECORD_RUN'))


def test_gates(tmp):
    bad, n = SC.gate_in_place(import_heavy=True)
    check('G-D0: in-place values equal the section (real modules)', not bad, f'{n} values; {bad[:2]}')
    bad, sums = SC.gate_child_modules()
    check("G-D0b: the content-test child's production modules equal the branch's", not bad and len(sums) == 6,
          f'{len(sums)} modules; {bad[:1]}')
    # a modified copy of the production repository refuses
    fake = os.path.join(tmp, 'fake_repo')
    for prod_rel, _ in SC.CHILD_MODULES:
        src = os.path.join(SC.child_repo(), prod_rel)
        dst = os.path.join(fake, prod_rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
    p = os.path.join(fake, 'exopipe/phase3/phase3_master_dewarp.py')
    open(p, 'a').write('\n# local edit\n')
    script = os.path.join(tmp, 'check_mdewarp_gen.py')
    text = open(SC.content_check_script()).read().replace("'/lustre/gh/main/exopipe'", repr(fake))
    open(script, 'w').write(text)
    bad, _ = SC.gate_child_modules(script)
    check('G-D0b: a production module differing from the branch refuses', len(bad) == 1 and 'phase3_master_dewarp' in bad[0],
          bad[0][:120] if bad else 'no violation')
    # G-D5 against the real exclusion list (23h 18MHz 2025-08-18 is excluded since 2026-10-12)
    m = os.path.join(tmp, 'gd5', '23h_18MHz', 'coadd_final', 'I_deep_Taper_R075_18MHz.fits')
    h = header('18MHz')
    h['HISTORY'] = 'COADD constituent 2025-08-18'
    write(m, np.zeros((8, 8), np.float32), h)
    adm, bad, by, fp = SM.admitted('23h', [('18MHz', os.path.dirname(m), [m, m])])
    check('G-D5: a measured master holding an excluded night refuses', len(bad) == 1 and '2025-08-18' in bad[0],
          f'admitted {adm}')
    check('G-D5: the admitted set keeps the other constituents', adm == [f'18MHz:{CONSTITUENT}'], str(adm))
    # G-D8
    saved = {k: os.environ.pop(k, None) for k in SC.THREAD_VARS}
    try:
        check('G-D8: thread variables unset refuse the measurement', len(SC.gate_threads()) == 3)
        for k in SC.THREAD_VARS:
            os.environ[k] = '1'
        check('G-D8: single-thread BLAS passes', SC.gate_threads() == [])
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    # G-W: the production roots refuse while PRODUCTION_WRITES is False
    roots = SC.Roots(types.SimpleNamespace(out=None, work=None),
                     types.SimpleNamespace(out_root=CFG.dewarp.GEN_ROOT, run_id='dewarp_test'))
    bad = SC.gate_write_roots(roots, list(SC.STAGES))
    check('G-W: production measure/gen/archive/record roots refuse', len(bad) == 4, f'{len(bad)} violations')
    roots = SC.Roots(types.SimpleNamespace(out=tmp, work=None), types.SimpleNamespace(out_root=tmp, run_id='x'))
    check('G-W: a --out under /tmp passes', SC.gate_write_roots(roots, list(SC.STAGES)) == [])


def test_table_and_screen(tmp):
    from exopipe.phase3.core import cells
    t = os.path.join(tmp, 'tt', 'tables', f'table_{LST}.csv')

    def rows(**over):
        out = []
        for s in cells.scopes():
            out.append(table_row(s, 'REFUSED no generation root'))
        out[2] = table_row(cells.scopes()[2], 'OK', med_before='2.0', med_after='1.0', n_before=40, n_after=40,
                           n_anchors=50, anchor_off_med_as='20.0', cv_resid_med_as='15.1')
        out[3] = table_row(cells.scopes()[3], 'REFERENCE ONLY: held-out improvement 5.0 arcsec < 5; not applied',
                           med_before='2.0', med_after='1.5', n_before=40, n_after=40, n_anchors=50,
                           anchor_off_med_as='20.0', cv_resid_med_as='15.0')
        for i, r in over.items():
            out[int(i[1:])] = r
        return out

    write_table(t, rows())
    ok, v, d = T.table_rows(t)
    check('dewarp_table: a complete table passes (OK at 4.9 written gain, REFERENCE ONLY at 5.0: rounding tolerance)', ok, d)
    write_table(t, rows(i2=table_row(cells.scopes()[2], 'OK', med_before='2.0', med_after='1.0', n_before=40, n_after=40,
                                     n_anchors=50, anchor_off_med_as='20.0', cv_resid_med_as='15.5')))
    ok, v, d = T.table_rows(t)
    check("dewarp_table: an OK row with written (b') gain 4.5 fails", not ok, d)
    write_table(t, rows()[:-1])
    ok, v, d = T.table_rows(t)
    check('dewarp_table: a missing scope fails', not ok and 'no row' in d, d)
    write_table(t, rows(i0=table_row(cells.scopes()[0], 'DONE whatever')))
    ok, v, d = T.table_rows(t)
    check('dewarp_table: an unknown outcome fails', not ok, d)
    write_table(t, rows(i2=table_row(cells.scopes()[2], 'OK', med_before='1.0', med_after='2.0', n_before=40, n_after=40,
                                     n_anchors=50, anchor_off_med_as='30.0', cv_resid_med_as='1.0')))
    ok, v, d = T.table_rows(t)
    check('dewarp_table: an OK row whose VLSSr median got worse fails', not ok and 'med_after' in d, d)
    # screen
    s = cells.scopes()[2]
    master = os.path.join(tmp, 'tt_in', f'{LST}_{s}', 'coadd_final', f'I_deep_Taper_R075_{s}.fits')
    write(master, sky(1), header(s))
    write_table(t, rows())
    scr = save_screen(os.path.join(tmp, 'tt'), s, 50, master)
    ok, v, d = T.screen_record(scr, table=t, master=master)
    check('dewarp_screen: a screen matching its row and master passes', ok, d)
    write_table(t, rows(i2=table_row(s, 'OK', med_before='2.0', med_after='1.0', n_before=40, n_after=40, n_anchors=51,
                                     anchor_off_med_as='20.0', cv_resid_med_as='15.1')))
    ok, v, d = T.screen_record(scr, table=t, master=master)
    check("dewarp_screen: NANCHOR differing from the final row fails", not ok and 'NANCHOR' in d, d)
    write_table(t, rows())
    small = os.path.join(tmp, 'tt_in', 'small.fits')
    write(small, np.zeros((64, 64), np.float32), header(s))
    ok, v, d = T.screen_record(scr, table=t, master=small)
    check('dewarp_screen: a shape differing from the measured master fails', not ok and 'shape' in d, d)


# ---------------------------------------------------------------------------------------------------------------------
# end-to-end through the CLI
# ---------------------------------------------------------------------------------------------------------------------

def build_round(tmp):
    from exopipe.phase3.core import cells
    from exopipe.phase3.lib.dewarp import master_dewarp as MD
    promoted, check_root, working = (os.path.join(tmp, x) for x in ('promoted', 'rebuild', 'working'))
    os.makedirs(working)
    ops = os.path.join(tmp, 'ops')
    cells_ = {'27MHz': {}, '32MHz': {}, '36MHz': {'bad_shape': True}, '41MHz': {}, '46MHz': {'mdwarp': True}}
    for i, (s, kw) in enumerate(cells_.items()):
        make_generation(promoted, f'{LST}_{s}', 100 * (i + 1), **kw)
        os.makedirs(os.path.join(check_root, f'{LST}_{s}'))
        open(os.path.join(check_root, f'{LST}_{s}', 'REBUILD_CONTENT_OK'), 'w').write(
            f'{promoted}/{LST}_{s}\nT1-T8 passed 2026-10-13T00:00:00Z (synthetic)\n')
    rl = os.path.join(ops, CFG.dewarp.ROUTE_LIST)
    os.makedirs(os.path.dirname(rl))
    open(rl, 'w').write('lst,scope\n' + ''.join(f'{LST},{s}\n' for s in ('27MHz', '32MHz', '36MHz', '46MHz')))
    # the measurement: final rows + saved screens (constant integer shift: the cubic resample is a pure shift)
    mdir = os.path.join(tmp, 'meas')
    m27 = os.path.join(promoted, f'{LST}_27MHz', 'coadd_final', 'I_deep_Taper_R075_27MHz.fits')
    scr27 = save_screen(mdir, '27MHz', 50, m27)
    save_screen(mdir, '36MHz', 40, os.path.join(promoted, f'{LST}_36MHz', 'coadd_final', 'I_deep_Taper_R075_36MHz.fits'))
    with fits.open(m27) as h:
        d0, hdr = np.squeeze(h[0].data).astype(np.float32), h[0].header.copy()
    with fits.open(scr27) as h:
        sx, sy = h['SCREEN_X'].data.astype(np.float64), h['SCREEN_Y'].data.astype(np.float64)
    m1, n1, p1 = MD.vlssr_offsets(MD.resample(d0, sx, sy), hdr)
    rows = []
    for s in cells.scopes():
        if s == '27MHz':
            rows.append(table_row(s, 'OK', ref='VLSSr', med_before=f'{m1 + 0.5:.3f}', n_before=n1, p90_before='9.0',
                                  med_after=f'{m1:.3f}', n_after=n1, p90_after=f'{p1:.3f}', n_match=60, n_anchors=50,
                                  anchor_off_med_as='20.0', fit_resid_med_as='1.0', cv_resid_med_as='5.0',
                                  cv_resid_p90_as='8.0', warp_p50_as='2012.5', warp_p95_as='2012.5'))
        elif s == '36MHz':
            rows.append(table_row(s, 'OK', ref='27MHz (master-dewarped)', med_before='2.0', n_before=40, med_after='1.0',
                                  n_after=40, n_anchors=40, anchor_off_med_as='20.0', fit_resid_med_as='1.0',
                                  cv_resid_med_as='5.0', warp_p50_as='1.0', warp_p95_as='2.0'))
        elif s == '32MHz':
            rows.append(table_row(s, 'REFERENCE ONLY: held-out improvement 3.0 arcsec < 5; not applied',
                                  med_before='2.0', n_before=40, med_after='1.5', n_after=40, n_anchors=40,
                                  anchor_off_med_as='12.0', cv_resid_med_as='9.0'))
        else:
            rows.append(table_row(s, 'REFUSED no generation root'))
    write_table(os.path.join(mdir, 'tables', f'table_{LST}.csv'), rows)
    return dict(promoted=promoted, check=check_root, working=working, ops=ops, meas=mdir, n1=n1)


def common_sets(r):
    return ['--set', f'dewarp.REBUILD_ROOT={r["promoted"]}', '--set', f'dewarp.REBUILD_CHECK_ROOT={r["check"]}',
            '--set', f'paths.COADD_SB_ROOT={r["working"]}', '--set', f'paths.OPS_ROOT={r["ops"]}', '--cores', 'none']


def test_end_to_end(tmp):
    r = build_round(tmp)
    check('synthetic measurement: >= 10 VLSSr peak matches after the shift', r['n1'] >= 10, f'n={r["n1"]}')
    work = os.path.join(tmp, 'work')
    # --- A: measure only
    outA = os.path.join(tmp, 'outA')
    rc, txt, s = cli_in_process('dewarp', '--lsts', LST, '--stages', 'stage,measure', '--threads', '1', '--out', outA,
                                '--work', work, *common_sets(r))
    ra = rec(outA, LST)
    check('measure: exit 0 (staging md5-verified, run_lst in place, table content-tested)',
          rc == 0 and ra and ra['status'] == 'done', f'rc {rc}, {s} s; ' + (ra['reason'] if ra else txt[-300:]))
    if ra:
        t = [o for o in ra['outputs'] if o['class'] == 'dewarp_table']
        check('measure: the table is a fresh <out>/measure/<run_id>/tables/table_05h.csv with 19 rows',
              len(t) == 1 and '/measure/dewarp_' in t[0]['path'] and len(SC.read_table(t[0]['path'])) == 19)
        check('measure: admitted night set and exclusion fingerprint recorded',
              ra['admitted'] and ra['meta'].get('exclusions', {}).get('sha1'))
        check('measure: node-local staging removed', not glob.glob(os.path.join(work, 'dewarp_stage_*')))
    # --- B: apply, verify, record on five cells (parity guard on)
    outB = os.path.join(tmp, 'outB')
    cells_b = ','.join(f'{LST}_{x}' for x in ('27MHz', '32MHz', '36MHz', '41MHz', '46MHz'))
    rc, txt, s = cli('dewarp', '--cells', cells_b, '--stages', 'apply,verify,record', '--measurement', r['meas'], '--out',
                     outB, '--work', work, '--parity', *common_sets(r))
    check('cells: exit 2 (three cells refused by gates, two done)', rc == 2, f'rc {rc}, {s} s')
    c27, c32, c36 = rec(outB, f'{LST}_27MHz'), rec(outB, f'{LST}_32MHz'), rec(outB, f'{LST}_36MHz')
    gen = os.path.join(outB, 'gen', f'{LST}_27MHz', 'gen_mdewarp_rbf')
    ok27 = c27 and c27['status'] == 'done'
    check('accepted cell: done, install kind mdewarp', ok27 and c27['meta'].get('install') == {'kind': 'mdewarp', 'gen': gen},
          (c27['reason'] if c27 else txt[-400:]))
    if c27:
        names = {t['name']: t for t in c27['tests']}
        for n in ('dewarped_master.D1_D6', 'dewarped_master.cubic', 'dewarped_master.applied_screen',
                  'dewarped_master.products', 'dewarped_master.D4_tested_pixels', 'dewarped_master.D1',
                  'dewarped_master.D3', 'dewarped_master.D5', 'dewarped_master.D6', 'dewarp.G-D0b_child_code'):
            check(f'accepted cell: {n} passed', n in names and names[n]['passed'], str(names.get(n, {}).get('detail'))[:160])
        check('accepted cell: the child code fingerprint is in the record', len(c27['meta'].get('legacy_code') or {}) >= 6)
        src = sum(len(glob.glob(os.path.join(r['promoted'], f'{LST}_27MHz', d, '**', '*.fits'), recursive=True))
                  for d in ('coadd_final', 'subtracted'))
        have = glob.glob(os.path.join(gen, '**', '*.fits'), recursive=True)
        check('accepted cell: one generation file per source file', len(have) == src, f'{len(have)} of {src}')
        check('accepted cell: siblings recomputed (5) with the _mdewarp token',
              all(os.path.isfile(os.path.join(gen, 'coadd_final', n.replace('_27MHz', '_mdewarp_27MHz') + '.fits'))
                  for n in sibling_names('27MHz')))
        check('accepted cell: INSTALL_CONTENT_OK and APPLIED_SCREEN.json beside the generation',
              all(os.path.isfile(os.path.join(os.path.dirname(gen), f)) for f in (SC.CONTENT_OK, SC.APPLIED_SCREEN)))
    csvp = os.path.join(outB, 'archive', 'mdewarp_screens', CFG.paths.ARCHIVE_RUN, 'mdewarp_screens.csv')
    rows = SC.read_table(csvp) if os.path.isfile(csvp) else []
    check('not-accepted cell: done, install kind rebuild',
          c32 and c32['status'] == 'done' and (c32['meta'].get('install') or {}).get('kind') == 'rebuild',
          c32['reason'] if c32 else '')
    check('not-accepted cell: one applied=0 row (install class screen_table + inline row check)',
          [(x['lst'], x['scope'], x['applied']) for x in rows] == [(LST, '32MHz', '0')]
          and rows[0]['status'].startswith('rebuilt 2026-10-08') and c32
          and {t['name'] for t in c32['tests'] if t['passed']} >= {'screen_table.rows', 'dewarp.record_row'},
          str([t['name'] for t in c32['tests']]) if c32 else str(rows))
    check('accepted cell whose apply refused: refused G-R2, no applied=0 row, no generation',
          c36 and c36['status'] == 'refused' and 'G-R2' in c36['reason']
          and not os.path.exists(os.path.join(outB, 'gen', f'{LST}_36MHz', 'gen_mdewarp_rbf')),
          c36['reason'][:120] if c36 else '')
    c41, c46 = rec(outB, f'{LST}_41MHz'), rec(outB, f'{LST}_46MHz')
    check('not routed: refused G-D2', c41 and c41['status'] == 'refused' and 'G-D2' in c41['reason'])
    check('input carries MDWARP: refused G-D7', c46 and c46['status'] == 'refused' and 'G-D7' in c46['reason'])
    glog = os.path.join(outB, '_records', '_parity_guard_refusals.log')
    check('parity guard: no write outside --out/--work', not os.path.exists(glog) or not open(glog).read().strip())
    if not ok27:
        return
    # --- C: verify alone (context from the record, scratch mode)
    side = {f: os.stat(os.path.join(os.path.dirname(gen), f)).st_mtime_ns for f in (SC.CONTENT_OK, SC.APPLIED_SCREEN)}
    rc, txt, s = cli('verify', 'dewarped_master', gen)
    check('verify: the generation passes every dewarped_master test (record context)', rc == 0, f'rc {rc}, {s} s')
    check('verify: scratch mode leaves the generation sidecars untouched',
          side == {f: os.stat(os.path.join(os.path.dirname(gen), f)).st_mtime_ns for f in side})
    # --- E: a corrupted product fails D3
    victim = os.path.join(gen, 'subtracted', '2025-01-01_I_sub_mdewarp_27MHz.fits')
    with fits.open(victim, mode='update') as h:
        h[0].data[N // 2, N // 2] += 1.0
    rc, txt, s = cli('verify', 'dewarped_master', gen)
    check('verify: a corrupted product fails D3', rc == 2 and 'D3' in txt, f'rc {rc}, {s} s')
    # --- D: a second apply without --fresh refuses (G-A3); --fresh removes and records the old generation
    rc, txt, s = cli('dewarp', '--cells', f'{LST}_27MHz', '--stages', 'apply', '--measurement', r['meas'], '--out', outB,
                     *common_sets(r))
    c = rec(outB, f'{LST}_27MHz')
    check('G-A3: a written generation is never overwritten', rc == 2 and c and 'G-A3' in c['reason'], c['reason'][:100] if c else '')
    rc, txt, s = cli('dewarp', '--cells', f'{LST}_27MHz', '--stages', 'apply', '--measurement', r['meas'], '--out', outB,
                     '--fresh', *common_sets(r))
    c = rec(outB, f'{LST}_27MHz')
    check('--fresh: the old generation removed (recorded), the apply re-run, not installable without verify',
          rc == 0 and c and (c['meta'].get('removed_previous') or {}).get('files', 0) > 0 and c['meta'].get('install') is None
          and not os.path.exists(os.path.join(os.path.dirname(gen), SC.CONTENT_OK)), f'rc {rc}')


def test_stale_measurement(tmp):
    """G-A11: a measurement older than the rebuild's T1-T8 stamp is refused (re-uses the synthetic round of test_end_to_end)."""
    r = dict(promoted=os.path.join(tmp, 'promoted'), check=os.path.join(tmp, 'rebuild'), working=os.path.join(tmp, 'working'),
             ops=os.path.join(tmp, 'ops'), meas=os.path.join(tmp, 'meas'))
    okf = os.path.join(r['check'], f'{LST}_32MHz', 'REBUILD_CONTENT_OK')
    if not os.path.isfile(okf):
        check('G-A11: synthetic round present', False)
        return
    open(okf, 'w').write(f'{r["promoted"]}/{LST}_32MHz\nT1-T8 passed 2026-10-14T00:00:00Z (synthetic, after the measurement)\n')
    out = os.path.join(tmp, 'outG')
    rc, txt, s = cli('dewarp', '--cells', f'{LST}_32MHz', '--stages', 'record', '--measurement', r['meas'], '--out', out,
                     *common_sets(r))
    c = rec(out, f'{LST}_32MHz')
    check('G-A11: a measurement older than the rebuild is refused (no applied=0 row written)',
          rc == 2 and c and 'G-A11' in c['reason'] and not os.path.exists(os.path.join(out, 'archive')),
          f'rc {rc}, {s} s; ' + (c['reason'][:100] if c else txt[-200:]))


def test_real_products():
    M = '/fast/gh/main/phase3/mdewarp_rbf_20261008/from_lwacalim09'
    for t in (f'{M}/tables/table_12h.csv', f'{M}/tables/table_23h.csv'):
        if os.path.isfile(t):
            ok, v, d = T.table_rows(t)
            check(f'real (read-only): dewarp_table {os.path.basename(t)}', ok, f'{v}')
    scr = f'{M}/12h/maps/Red_warp_screen.fits'
    if os.path.isfile(scr):
        ok, v, d = T.screen_record(scr)
        check('real (read-only): dewarp_screen 12h Red', ok, d[:120])
    gen = '/lustre/gh/main/phase3/mdewarp_rbf_gen_20261007/12h_Red/gen_mdewarp_rbf'
    if os.path.isdir(gen):
        for fn in (T.applied_screen, T.products, T.d4_tested_pixels):
            ok, v, d = fn(gen)
            check(f'real (read-only): dewarped_master.{fn.__name__} 12h_Red', ok, str(d)[:120])


def main():
    t0 = time.time()
    os.makedirs(SCRATCH)
    try:
        test_acceptance_and_stages()
        test_config()
        test_gates(SCRATCH)
        test_table_and_screen(SCRATCH)
        test_end_to_end(SCRATCH)
        test_stale_measurement(SCRATCH)
        test_real_products()
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
        try:
            os.rmdir(os.path.dirname(SCRATCH))
        except OSError:
            pass
    bad = [r for r in RESULTS if not r[1]]
    print(f'\n{len(RESULTS) - len(bad)} of {len(RESULTS)} checks passed in {time.time() - t0:.0f} s', flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
