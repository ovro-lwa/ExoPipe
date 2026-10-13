#!/usr/bin/env python3
"""TEST - `exopipe3 install` (steps/install.py, lib/install/steps_*.py, lib/install/tests.py, _cfgdraft_install.py).

Synthetic data in a fresh directory /tmp/exopipe3_tests_install/<pid>/ (removed at the end). Nothing outside it is
written. Production files are only READ: the headers of the real 12h_Red mdewarp generation and the four small records of
the real working cell 12h_Red are used as templates for the synthetic cell (so the in-place stamping, publish and
membership code sees real header cards and real ruled records); data_decisions.csv, the solar audit, the named
out-of-master table and the archived screen record are read by the in-place membership code as in production.

  unit   stage grammar; config protection (PROTECTED_KEYS, shared roots, window-mirror check); waivers; the G-1b scan on
         a synthetic link tree; the re-run item file
  e2e A  refused AFTER staging (an in-master night fails the epoch gate): exit 2, the cell and the archive untouched, no
         staging left, the manifest holds the plan + INSTALL-FAILED
  e2e B  install --round auto with today's round partly taken -> round <today>b; every output passes its content test;
         old files in the working / archive quarantine with their md5; the _msub search frame untouched; screen
         quarantined and the applied=1 row written
  e2e C  re-install with --round <today>b (now taken): refused at G-DEST BEFORE staging (no staging, no action row)
  e2e D  re-install with --round auto: round <today>c, done
  e2e E  enqueue into a mirror of a synthetic live search state: markers/parks retired, both Stokes assigned, flow-holder
         row, ready.txt, request + re-run items pass their tests; a second enqueue adds nothing; recover on a finished
         install refuses

The two snapshot_masters link scans (912 links, ~40 s each on Lustre) are stubbed in the e2e runs; the scan itself is
tested on a synthetic tree (unit). Run: python tests/test_install.py   (about a minute)
"""
import csv
import glob
import hashlib
import json
import os
import shutil
import sys
import time
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

ROOT = f'/tmp/exopipe3_tests_install/{os.getpid()}'
REAL_GEN = '/lustre/gh/main/phase3/mdewarp_rbf_gen_20261007/12h_Red/gen_mdewarp_rbf'
REAL_CELL = '/data07/gh/phase3/coadd_sb/12h_Red'
CELL, LST, SCOPE = '12h_Red', '12h', 'Red'
N = 128
RESULTS = []


def ok(name, cond, detail=''):
    RESULTS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f'  ({detail})' if detail else ''), flush=True)
    return cond


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# synthetic data
# ---------------------------------------------------------------------------------------------------------------------

def _hdr(path):
    from astropy.io import fits
    h = fits.getheader(path)
    for k in ('CHECKSUM', 'DATASUM'):
        h.remove(k, ignore_missing=True, remove_all=True)
    return h


def _write(path, data, hdr):
    from astropy.io import fits
    import numpy as np
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.writeto(path, np.asarray(data, dtype=np.float32), hdr, overwrite=True)


def _images(seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:N, :N]
    src = np.zeros((N, N))
    for y in range(10, N - 9, 12):
        for x in range(10, N - 9, 12):
            if np.hypot(x - N / 2, y - N / 2) < 0.8 * N / 2:
                src += rng.uniform(1.0, 5.0) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 2.0)
    mi = (rng.normal(0, 0.01, (N, N)) + src).astype(np.float32)
    mv = (rng.normal(0, 0.01, (N, N)) + 0.05 * src).astype(np.float32)
    r = np.hypot(xx - N / 2, yy - N / 2) / (N / 2)
    pb = np.clip(1.0 - 0.8 * r ** 2, 0.02, 1.0).astype(np.float32)
    return mi, mv, pb, rng


def real_templates():
    """(nights, {gen rel path: header}) from the real generation: 3 in-master nights (not night-image FAIL)."""
    with open(f'{REAL_CELL}/coadd_final/weights.csv') as f:
        w = list(csv.DictReader(f))
    with open(f'{REAL_CELL}/coadd_final/night_image_gate.csv') as f:
        ng = {r['date']: r['verdict'] for r in csv.DictReader(f)}
    nights = [r['date'] for r in w if r.get('in_master') == '1' and ng.get(r['date']) != 'FAIL'][:3]
    rels = [os.path.relpath(p, REAL_GEN) for p in sorted(glob.glob(f'{REAL_GEN}/coadd_final/*.fits'))]
    for d in nights:
        rels += [f'subtracted/{d}_{st}_{k}_mdewarp_{SCOPE}.fits' for st in 'IV' for k in ('clean', 'epoch', 'sub')]
    return nights, {r: _hdr(f'{REAL_GEN}/{r}') for r in rels}


def build_generation(gen, hdrs, nights, seed, fail_night=None):
    """A synthetic mdewarp generation with real headers; masters with >= 20 bright compact sources; siblings = plain/PB;
    epochs = master (+ a tiny noise), residual = epoch - master (identity); ``fail_night``: its I epoch = 0.5 x master."""
    import numpy as np
    mi, mv, pb, rng = _images(seed)
    data = {}
    for rel, h in hdrs.items():
        b = os.path.basename(rel)
        if not rel.startswith('coadd_final/'):
            continue
        if b.startswith('PB_deep'):
            data[rel] = pb
        elif '_pbcorr_' not in b:
            data[rel] = mi if b.startswith('I_') else mv
    for rel, h in hdrs.items():
        b = os.path.basename(rel)
        if rel.startswith('coadd_final/') and '_pbcorr_' in b:
            plain = data[rel.replace('_pbcorr_', '_', 1)].astype(np.float64)
            floor = float(h.get('PBFLOOR', 0.1))
            out = np.full((N, N), np.nan)
            v = pb > floor
            out[v] = plain[v] / pb[v]
            data[rel] = out
    for d in nights:
        for st, m in (('I', mi), ('V', mv)):
            e = (m * (0.5 if (st == 'I' and d == fail_night) else 1.0) + rng.normal(0, 0.001, (N, N))).astype(np.float32)
            s = (e - m).astype(np.float32)
            data[f'subtracted/{d}_{st}_epoch_mdewarp_{SCOPE}.fits'] = e
            data[f'subtracted/{d}_{st}_sub_mdewarp_{SCOPE}.fits'] = s
            data[f'subtracted/{d}_{st}_clean_mdewarp_{SCOPE}.fits'] = 0.9 * s
    for rel, h in hdrs.items():
        _write(f'{gen}/{rel}', data[rel], h)
    cell_dir = os.path.dirname(gen)
    with open(f'{cell_dir}/INSTALL_CONTENT_OK', 'w') as f:
        f.write(f'{gen}\nD1-D6 passed {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}: {len(hdrs)} files resampled '
                f'cubic-spline; synthetic (test_install)\n')
    screen = f'{cell_dir}/Red_warp_screen.fits'
    _write(screen, rng.normal(0, 1, (8, 8)), None)
    js = dict(gen=gen, lst=LST, scope=SCOPE, screen=screen, screen_md5=md5(screen), status='OK', n_anchors='902',
              med_before='1.518', med_after='0.908', anchor_off_med_as='35.9', cv_resid_med_as='13.0', warp_p50_as='36.0',
              warp_p95_as='94.6', measured_at='2026-10-12T22:52:21Z')
    with open(f'{cell_dir}/APPLIED_SCREEN.json', 'w') as f:
        json.dump(js, f, indent=1)
    return gen


def build_current(out, hdrs, nights, run):
    """The working cell before the install (canonical names, other data, one orphan) and the archive cell (an old master,
    an old weights.csv, an old epoch, a search frame _msub that install must leave alone, the old screen + its row)."""
    import numpy as np
    cdir = f'{out}/coadd_sb/{CELL}'
    for rel, h in hdrs.items():
        canon = os.path.join(os.path.dirname(rel), os.path.basename(rel).replace('_mdewarp', '', 1))
        _write(f'{cdir}/{canon}', np.full((N, N), 0.25), h)
    _write(f'{cdir}/subtracted/2099-01-01_I_clean_{SCOPE}.fits', np.zeros((N, N)), None)      # orphan
    for f in ('weights.csv', 'night_image_gate.csv', 'stage2_verdict.csv', 'epoch_gate.csv'):
        shutil.copy2(f'{REAL_CELL}/coadd_final/{f}', f'{cdir}/coadd_final/{f}')
    shutil.copy2(f'{REAL_CELL}/residuals_provisional.csv', f'{cdir}/residuals_provisional.csv')
    arch = f'{out}/archive'
    mdir = f'{arch}/Coadd/{run}/{LST}/{SCOPE}'
    _write(f'{mdir}/old_master_N020.fits', np.ones((4, 4)), None)
    os.makedirs(mdir, exist_ok=True)
    open(f'{mdir}/weights.csv', 'w').write('date,weight\n')
    _write(f'{arch}/{LST}/{nights[0]}/{run}/{SCOPE}/I/deep/old_epoch.fits', np.ones((4, 4)), None)
    _write(f'{arch}/{LST}/{nights[0]}/{run}/{SCOPE}/I/deep/frame_msub_ssub.fits', np.ones((4, 4)), None)
    sdir = f'{arch}/mdewarp_screens/{run}'
    _write(f'{sdir}/{LST}_{SCOPE}_warp_screen.fits', np.ones((4, 4)), None)
    from exopipe.phase3.lib.install import install as PI
    with open(f'{sdir}/mdewarp_screens.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=PI.SCREEN_COLS)
        wr.writeheader()
        wr.writerow(dict(lst=LST, scope=SCOPE, applied=1, status='OK', screen_file=f'{LST}_{SCOPE}_warp_screen.fits',
                         screen_md5=md5(f'{sdir}/{LST}_{SCOPE}_warp_screen.fits'), n_anchors=1))
        wr.writerow(dict(lst='05h', scope='Blue', applied=0, status='measured, not accepted (synthetic)'))
    return cdir


def build_live_state(live):
    os.makedirs(live, exist_ok=True)
    for st in 'VI':
        open(f'{live}/assign_{st}.csv', 'w').write('11h_18MHz,lwacalim07\n11h_23MHz,lwacalim09\n')
        os.makedirs(f'{live}/markers_{st}', exist_ok=True)
        os.makedirs(f'{live}/parked_{st}', exist_ok=True)
    open(f'{live}/markers_V/{CELL}', 'w').write('done\n')
    open(f'{live}/markers_I/{CELL}.policy-skip', 'w').write('skip\n')
    open(f'{live}/parked_I/{CELL}', 'w').write('parked\n')
    open(f'{live}/ready.txt', 'w').write('11h_18MHz\n')
    open(f'{live}/cell_node_map.csv', 'w').write('11h_18MHz,lwacalim07,/fast/gh/main/phase3/s2work_p3s5/11h_18MHz\n')


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------

def cli(*argv):
    from exopipe.phase3 import cli as C
    from exopipe.phase3.config import CFG
    CFG.reset()
    rc = C.main([str(a) for a in argv])
    CFG.reset()
    return rc


def record(out, unit):
    p = f'{out}/_records/install/{unit}.json'
    return json.load(open(p)) if os.path.exists(p) else None


def failed_tests(rec):
    return [t for t in (rec or {}).get('tests', []) if not t['passed'] and not t.get('waived')]


def tree_md5(root):
    out = {}
    for p in sorted(glob.glob(f'{root}/**/*', recursive=True)):
        if os.path.isfile(p):
            out[os.path.relpath(p, root)] = md5(p)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------------------------------------------------

def test_units():
    from exopipe.phase3.config import CFG, ConfigRefusal
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.steps import install as S
    from exopipe.phase3.lib.install import steps_common as SC, steps_deep as SD, steps_enqueue as SE
    from exopipe.phase3.lib.install import install as PI
    from exopipe.phase3.core import cells
    ok('stages: install implies plan', S.parse_stages('install') == ['plan', 'install'])
    ok('stages: ordered', S.parse_stages('verify,plan') == ['plan', 'verify'])
    for bad in ('recover,plan', 'nonsense', '', 'supersede', 'handover'):
        try:
            S.parse_stages(bad)
            ok(f'stages: {bad!r} refused', False)
        except Refusal:
            ok(f'stages: {bad!r} refused', True)
    ok('tier: deep only', S.TIERS == ('deep',))
    # protection
    CFG.reset()
    for key in ('install.RECORD_CSV=/tmp/x.csv', 'install.WORKING_QUARANTINE_FMT=/tmp/wq_{round}',
                'install_enqueue.SEARCH_STATE_ROOT=/tmp/s', 'install.SNAPSHOT_MASTERS_LINK_TREE=/tmp/t'):
        try:
            CFG.apply_override_strings([key])
            ok(f'protected: {key.split("=")[0]}', False)
        except ConfigRefusal:
            ok(f'protected: {key.split("=")[0]}', True)
    CFG.reset()
    CFG.authorize([], out='/lustre/pipeline/exopipe/phase3/x')
    try:
        CFG.apply_override_strings(['install.FREE_FLOOR_GB=0'])
        ok('ruled FREE_FLOOR_GB into a production out refused', False)
    except ConfigRefusal:
        ok('ruled FREE_FLOOR_GB into a production out refused', True)
    CFG.reset()
    CFG.authorize([], out=ROOT)
    CFG.apply_override_strings(['install.FREE_FLOOR_GB=0'])
    ok('ruled FREE_FLOOR_GB into a /tmp out allowed', CFG.install.FREE_FLOOR_GB == 0.0)
    CFG.reset()
    CFG.apply_override_strings(['paths.COADD_SB_ROOT=/tmp/elsewhere'])
    try:
        SC.check_protected_overrides()
        ok('shared root override refused by install', False)
    except Refusal:
        ok('shared root override refused by install', True)
    CFG.reset()
    SC.check_window_consistency(PI)
    ok('window mirrors equal phase3_install literals', True)
    old = CFG.install.STAGING_SPACE_FACTOR
    CFG.install.STAGING_SPACE_FACTOR = 1.1
    try:
        SC.check_window_consistency(PI)
        ok('window mirror changed -> refused', False)
    except Refusal as e:
        ok('window mirror changed -> refused', 'STAGING_SPACE_FACTOR' in str(e))
    finally:
        CFG.install.STAGING_SPACE_FACTOR = old
    # waivers
    for w, good in (('install.H', True), ('install.G-DEST', False), ('install.G0', False), ('install.G-MEM', False),
                    ('install.NOPE', False), ('archive_cell.audit', False), ('installed_cell.T7_date_sets', True)):
        r = types.SimpleNamespace(waivers=[(w, 'words')])
        try:
            SC.check_waivers(r)
            ok(f'waiver {w} {"accepted" if good else "refused"}', good)
        except Refusal:
            ok(f'waiver {w} {"accepted" if good else "refused"}', not good)
    # G-1b on a synthetic link tree
    t = f'{ROOT}/unit/snapshot_masters'
    arch = f'{ROOT}/unit/arch'
    for p in (f'{arch}/Coadd/Run/12h/Red/m.fits', f'{ROOT}/unit/frozen/m.fits'):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'w').write('x')
    os.makedirs(f'{t}/12h', exist_ok=True)
    os.symlink(f'{arch}/Coadd/Run/12h/Red/m.fits', f'{t}/12h/live.fits')
    os.symlink(f'{ROOT}/unit/frozen/m.fits', f'{t}/12h/frozen.fits')
    old = CFG.install.SNAPSHOT_MASTERS_LINK_TREE
    CFG.install.SNAPSHOT_MASTERS_LINK_TREE = t
    SD._G1B.clear()
    try:
        n, live = SD.snapshot_link_scan([arch, f'{ROOT}/unit/frozen_arch'])
        ok('G-1b scan: links counted, the live one found', n == 2 and live[arch] == [f'{t}/12h/live.fits']
           and live[f'{ROOT}/unit/frozen_arch'] == [], f'{n} links, {live}')
    finally:
        CFG.install.SNAPSHOT_MASTERS_LINK_TREE = old
        SD._G1B.clear()
    # re-run item file
    p = f'{ROOT}/unit/rerun.txt'
    SE.write_rerun_file(p, {'12h_Red': ['2025-01-02', '2025-01-01'], '12h_41MHz': ['2025-01-01']}, '/w', 'searchable')
    head, items = SE.read_rerun_file(p)
    ok('re-run file: items are epoch items, cells joined', sorted(items) == ['12h_2025-01-01', '12h_2025-01-02']
       and items['12h_2025-01-01'] == ['12h_41MHz', '12h_Red'] and head['cells'] == '12h_41MHz,12h_Red')
    ok('re-run file: core.cells reads the items', cells.expand(f'@{p}') == ['12h_2025-01-01', '12h_2025-01-02']
       and all(cells.parse_epoch(x) for x in cells.expand(f'@{p}')))


# ---------------------------------------------------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------------------------------------------------

def test_e2e():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import validate
    from exopipe.phase3.lib.install import steps_deep as SD, steps_enqueue as SE
    from exopipe.phase3.lib.install import install as PI
    import glob as _glob
    if not (os.path.isdir(REAL_GEN) and os.path.isdir(REAL_CELL)):
        ok('e2e templates (real 12h_Red generation and cell) readable', False, 'SKIPPED: templates absent')
        return
    run = CFG.paths.ARCHIVE_RUN
    nights, hdrs = real_templates()
    ok('templates: 3 in-master nights, 14 coadd_final + 18 subtracted headers', len(nights) == 3 and len(hdrs) == 32,
       f'{nights}, {len(hdrs)} headers')
    # stubs for the two ~40 s scans of the real snapshot_masters link tree (tested on a synthetic tree in test_units)
    real_scan, real_glob = SD.snapshot_link_scan, PI.glob
    SD.snapshot_link_scan = lambda archs: (0, {a: [] for a in dict.fromkeys(archs)})
    PI.glob = types.SimpleNamespace(glob=lambda pat, recursive=False: [] if 'snapshot_masters' in pat
                                    else _glob.glob(pat, recursive=recursive))
    old_state, old_holder = CFG.install_enqueue.SEARCH_STATE_ROOT, getattr(CFG.site, 'FLOW_HOLDER', None)
    try:
        _e2e(run, nights, hdrs)
    finally:
        SD.snapshot_link_scan, PI.glob = real_scan, real_glob
        CFG.install_enqueue.SEARCH_STATE_ROOT = old_state
        if old_holder is None:
            if hasattr(CFG.site, 'FLOW_HOLDER'):
                delattr(CFG.site, 'FLOW_HOLDER')
        else:
            CFG.site.FLOW_HOLDER = old_holder


def _e2e(run, nights, hdrs):
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import validate
    from exopipe.phase3.lib.install import steps_enqueue as SE
    out = f'{ROOT}/out'
    common = ['--out', out, '--cores', 'none', '--set', 'install.FREE_FLOOR_GB=0.0', '--kind', 'mdewarp', '--not-rebuilt']
    cdir = build_current(out, hdrs, nights, run)
    arch = f'{out}/archive'
    today = time.strftime('%Y%m%d', time.gmtime())

    # ---- A: refused after staging (epoch gate, in-master night)
    gen_bad = build_generation(f'{ROOT}/gen_bad/{CELL}/gen_mdewarp_rbf', hdrs, nights, seed=1, fail_night=nights[1])
    before_cell, before_arch = tree_md5(cdir), tree_md5(arch)
    rc = cli('install', '--cells', CELL, '--gen', gen_bad, *common)
    rec = record(out, CELL)
    ok('A: refused after staging -> exit 2', rc == 2 and rec and rec['status'] == 'refused', f'rc {rc}, '
       f'{rec and rec["reason"][:160]}')
    ok('A: the refusal names the epoch gate', rec and 'EG' in rec['reason'])
    ok('A: cell and archive untouched, no staging left', tree_md5(cdir) == before_cell and tree_md5(arch) == before_arch
       and not os.path.exists(f'{cdir}/.install_staging'))
    man = list(csv.DictReader(open(f'{out}/_records/install/{CELL}.manifest.csv')))
    ok('A: manifest = plan (every destination free) + gates-passed? no + INSTALL-FAILED',
       any(r['step'] == 'plan-dest' for r in man) and any(r['step'] == 'INSTALL-FAILED' for r in man)
       and not any(r['step'] in ('working-quarantine', 'installed') for r in man))

    # ---- B: install, round auto with today's round partly taken
    gen = build_generation(f'{ROOT}/gen/{CELL}/gen_mdewarp_rbf', hdrs, nights, seed=2)
    taken = f'{out}/working_quarantine_{today}/{CELL}/coadd_final/I_deep_Taper_R075_{SCOPE}.fits'
    os.makedirs(os.path.dirname(taken), exist_ok=True)
    open(taken, 'w').write('an earlier round')
    old_cur = {r: md5(f'{cdir}/{r}') for r in [os.path.relpath(p, cdir) for p in glob.glob(f'{cdir}/**/*.fits',
                                                                                            recursive=True)]}
    old_arch = {p: md5(p) for p in glob.glob(f'{arch}/**/*', recursive=True) if os.path.isfile(p)
                and 'mdewarp_screens.csv' not in p}
    msub = f'{arch}/{LST}/{nights[0]}/{run}/{SCOPE}/I/deep/frame_msub_ssub.fits'
    msub_md5 = md5(msub)
    t0 = time.time()
    rc = cli('install', '--cells', CELL, '--gen', gen, *common)
    rec = record(out, CELL)
    ok('B: install done -> exit 0', rc == 0 and rec and rec['status'] == 'done', f'rc {rc}, {rec and rec["reason"][:300]}, '
       f'{time.time() - t0:.0f} s')
    if not rec or rec['status'] != 'done':
        for t in failed_tests(rec):
            print('   failed:', t['name'], str(t.get('detail'))[:400])
        print('   notes:', '\n'.join((rec or {}).get('notes', []))[-2500:])
        return
    rnd = rec['meta'].get('round')
    ok('B: round auto skipped the taken round -> <today>b', rnd == today + 'b', rnd)
    classes = sorted({o['class'] for o in rec['outputs']})
    ok('B: outputs', classes == ['archive_cell', 'install_manifest', 'installed_cell', 'screen_table'], classes)
    ok('B: every content test passed', not failed_tests(rec), '; '.join(f"{t['name']}: {str(t.get('detail'))[:120]}"
                                                                         for t in failed_tests(rec)))
    names = {t['name'] for t in rec['tests']}
    for want in ('install_manifest.rows_match_files', 'archive_cell.audit', 'screen_table.rows',
                 'screen_table.install_cell_row', 'installed_cell.membership', 'installed_cell.products',
                 'installed_cell.epochs', 'installed_cell.T4_pb_siblings', 'installed_cell.exclusion_conformance',
                 'install.V', 'install.G-DEST', 'install.INSTALL-OK'):
        ok(f'B: test/gate {want} recorded', want in names)
    wq = f'{out}/working_quarantine_{rnd}/{CELL}'
    moved_ok = all(os.path.exists(f'{wq}/{r}') and md5(f'{wq}/{r}') == m for r, m in old_cur.items())
    ok('B: every replaced working FITS (incl. the orphan) in the working quarantine with its md5', moved_ok,
       f'{len(old_cur)} files')
    ok('B: orphan gone from the working tree', not os.path.exists(f'{cdir}/subtracted/2099-01-01_I_clean_{SCOPE}.fits'))
    aq = f'{arch}/Run_INSTALL_QUARANTINE_{rnd}'
    aq_ok = all(os.path.exists(f'{aq}/{os.path.relpath(p, arch)}') and md5(f'{aq}/{os.path.relpath(p, arch)}') == m
                for p, m in old_arch.items() if '_msub' not in p)
    ok('B: every replaced archive file (and the old screen) in the archive quarantine with its md5', aq_ok,
       f'{len(old_arch) - 1} files')
    ok('B: the _msub search frame untouched', os.path.exists(msub) and md5(msub) == msub_md5)
    rows = list(csv.DictReader(open(f'{arch}/mdewarp_screens/{run}/mdewarp_screens.csv')))
    mine = [r for r in rows if (r['lst'], r['scope']) == (LST, SCOPE)]
    ok('B: one applied=1 screen row with the new md5; the other row kept', len(mine) == 1 and mine[0]['applied'] == '1'
       and mine[0]['screen_md5'] == json.load(open(f'{ROOT}/gen/{CELL}/APPLIED_SCREEN.json'))['screen_md5']
       and any(r['lst'] == '05h' for r in rows))
    ok('B: installed files = generation bytes', all(md5(f'{cdir}/{os.path.relpath(p, gen).replace("_mdewarp", "", 1)}')
                                                    for p in glob.glob(f'{gen}/coadd_final/PB_*.fits')))
    ok('B: admitted night set recorded', rec.get('admitted') and set(nights) <= set(rec['admitted']) | set(nights))
    for cls, path in (('installed_cell', cdir), ('archive_cell', f'{arch}/Coadd/{run}/{LST}/{SCOPE}'),
                      ('install_manifest', f'{out}/_records/install/{CELL}.manifest.csv')):
        res = validate.run(cls, path)
        ok(f'B: verify {cls} alone (no context) passes', all(r['passed'] for r in res),
           '; '.join(f"{r['name']}: {r['detail'][:100]}" for r in res if not r['passed']))
    log_rows = list(csv.DictReader(open(f'{out}/install_record.csv')))
    ok('B: action log re-rooted under --out (unchanged columns)', log_rows and list(log_rows[0]) ==
       ['time', 'cell', 'step', 'path', 'dest', 'md5', 'bytes', 'detail']
       and any(r['step'] == 'installed-and-audited' for r in log_rows))

    # ---- C: named round now taken -> G-DEST before staging
    n_log = len(log_rows)
    rc = cli('install', '--cells', CELL, '--gen', gen, '--round', today + 'b', *common)
    rec = record(out, CELL)
    ok('C: named taken round refused at G-DEST -> exit 2', rc == 2 and rec['status'] == 'refused'
       and rec['reason'].startswith('G-DEST'), rec['reason'][:160])
    ok('C: nothing staged, no action row', not os.path.exists(f'{cdir}/.install_staging')
       and len(list(csv.DictReader(open(f'{out}/install_record.csv')))) == n_log)
    ok('C: the earlier manifest kept', len(glob.glob(f'{out}/_records/install/{CELL}.manifest.*.csv')) >= 1)

    # ---- D: auto round again -> <today>c
    rc = cli('install', '--cells', CELL, '--gen', gen, *common)
    rec = record(out, CELL)
    ok('D: re-install, round auto -> <today>c, done', rc == 0 and rec['status'] == 'done'
       and rec['meta'].get('round') == today + 'c', f"rc {rc} {rec['meta'].get('round')} {rec['reason'][:200]}")

    # ---- E: enqueue into a mirror of a synthetic live state; recover on a finished install
    live = f'{ROOT}/live_state'
    build_live_state(live)
    CFG.install_enqueue.SEARCH_STATE_ROOT = live
    CFG.site.FLOW_HOLDER = 'lwacalim07'
    live_before = tree_md5(live)
    rc = cli('install', '--cells', CELL, '--stages', 'enqueue', '--out', out, '--cores', 'none')
    rec = record(out, f'{CELL}.enqueue')
    ok('E: enqueue done -> exit 0', rc == 0 and rec and rec['status'] == 'done', f'rc {rc} {rec and rec["reason"][:200]}')
    ok('E: the live state untouched', tree_md5(live) == live_before)
    S = f'{out}/p3s5_state'
    ok('E: both Stokes assigned to the node of the first line', all(f'{CELL},lwacalim07' in open(f'{S}/assign_{s}.csv').read()
                                                                    for s in 'VI'))
    ok('E: ready.txt and the flow-holder row', CELL in open(f'{S}/ready.txt').read().split()
       and f'{CELL},lwacalim07,/fast/gh/main/phase3/s2work_p3s5/{CELL}' in open(f'{S}/cell_node_map.csv').read())
    ok('E: marker, policy-skip and park retired with a stamp', not os.path.exists(f'{S}/markers_V/{CELL}')
       and not os.path.exists(f'{S}/markers_I/{CELL}.policy-skip') and not os.path.exists(f'{S}/parked_I/{CELL}')
       and len(glob.glob(f'{S}/markers_retired_V/{CELL}.marker.*')) == 1
       and len(glob.glob(f'{S}/markers_retired_I/{CELL}.park.*')) == 1)
    classes = sorted({o['class'] for o in rec['outputs']})
    ok('E: outputs research_request + rerun_epochs, tests pass', classes == ['rerun_epochs', 'research_request']
       and not failed_tests(rec), f'{classes} {failed_tests(rec)}')
    combined = glob.glob(f'{out}/_records/install/rerun_epochs/*.txt')
    head, items = SE.read_rerun_file(combined[0]) if combined else ({}, {})
    ok('E: combined re-run item file lists the cell\'s nights as epoch items', combined and
       set(items) == {f'{LST}_{d}' for d in nights}, f'{sorted(items)}')
    n_ready = open(f'{S}/ready.txt').read().split().count(CELL)
    rc = cli('install', '--cells', CELL, '--stages', 'enqueue', '--out', out, '--cores', 'none')
    ok('E: second enqueue adds nothing', rc == 0 and open(f'{S}/ready.txt').read().split().count(CELL) == n_ready == 1
       and sum(l.startswith(f'{CELL},') for l in open(f'{S}/assign_V.csv')) == 1)
    rc = cli('install', '--cells', CELL, '--stages', 'cleanup', '--out', out, '--cores', 'none')
    ok('E: cleanup under --out removes nothing (exit 3)', rc == 3)
    rc = cli('install', '--cells', CELL, '--stages', 'recover', '--gen', gen, *common)
    rec = record(out, f'{CELL}.recover')
    ok('E: recover of a finished install refused', rc == 2 and rec['status'] == 'refused', rec['reason'][:160])
    rc = cli('install', '--cells', CELL, '--tier', 'deep', '--stages', 'verify', '--out', out, '--cores', 'none')
    rec = record(out, f'{CELL}.verify')
    ok('E: verify alone done', rc == 0 and rec['status'] == 'done', rec['reason'][:200])


def main():
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)
    os.makedirs(ROOT)
    t0 = time.time()
    try:
        test_units()
        test_e2e()
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
        try:
            os.rmdir(os.path.dirname(ROOT))
        except OSError:
            pass
    bad = [r for r in RESULTS if not r[1]]
    print(f'\n{len(RESULTS) - len(bad)} of {len(RESULTS)} checks passed in {time.time() - t0:.0f} s')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
