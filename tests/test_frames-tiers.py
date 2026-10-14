#!/usr/bin/env python3
"""Tests of `exopipe3 frames --tier sublst|snapshot` (unit frames-tiers). Plain asserts, no pytest:

    NUMPY_MADVISE_HUGEPAGE=0 EXOPIPE3_SITE=/lustre/gh/main/exopipe_ops/exopipe3_site.py \
        /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_frames-tiers.py [--quick] [--real]

Synthetic data only, in a fresh /tmp/exopipe3_tests_frames-tiers/ (removed at the end). The in-place clv15 engines
(phase3_sublst_wideband.py, phase3_sublst_tenmin_tree.py, phase3_snapshot_regrid.py, phase3_snapshot_cube.py,
phase3_snapshot_frame_movie.py, phase3_snapshot_flow_movie.py) run for real on that synthetic data through
core.legacy.run, every step command under --parity. Production files are only READ: the frozen deep records and the
membership-veto pin (to choose real nights for the synthetic sub-LST tree, whose engine reads them), the 18h_18MHz snapshot
Master header (the regrid target grid), data_decisions.csv. --quick skips the engine runs; --real adds read-only content
tests on production products and a dry run on a real LST and a real night cell.
"""
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

import numpy as np                                                    # noqa: E402

from exopipe.phase3.core import env                                   # noqa: E402

env.legacy()
from exopipe.phase3.config import CFG, ConfigRefusal                  # noqa: E402
from exopipe.phase3.core import validate, cells, exclusions, ledger   # noqa: E402
from exopipe.phase3.lib.frames import tier_sublst as TS               # noqa: E402
from exopipe.phase3.lib.frames import tier_snapshot as TN             # noqa: E402
from exopipe.phase3.lib.frames import tests_sublst as XS              # noqa: E402
from exopipe.phase3.lib.frames import tests_snapshot as XN            # noqa: E402

T = '/tmp/exopipe3_tests_frames-tiers'
PY = sys.executable
QUICK = '--quick' in sys.argv
REAL = '--real' in sys.argv
LST = '09h'
RESULTS = []


def check(name, cond, detail=''):
    RESULTS.append((name, bool(cond), detail))
    print(('PASS ' if cond else 'FAIL ') + name + (f'  ({detail})' if detail and not cond else ''), flush=True)


def exopipe3(*argv, timeout=600):
    """Run `python -m exopipe.phase3 frames ...` in a child (the CLI as a user runs it). Returns (rc, output)."""
    e = dict(os.environ, PYTHONPATH=REPO)
    p = subprocess.run([PY, '-m', 'exopipe.phase3', 'frames'] + list(argv) + ['--cores', 'none'], cwd=REPO, env=e,
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def records(out, prefix=''):
    recs = {}
    for p in glob.glob(os.path.join(out, '_records', 'frames', f'{prefix}*.json')):
        r = json.load(open(p))
        recs[r['unit']] = r
    return recs


def fits_write(path, data, hdr):
    from astropy.io import fits
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h = fits.Header()
    for k, v in hdr.items():
        h[k] = v
    fits.PrimaryHDU(np.asarray(data, dtype=np.float32), h).writeto(path, overwrite=True)


# =====================================================================================================================
# 1. config, stage grammar, items
# =====================================================================================================================

def test_config():
    for sec in ('frames_sublst', 'frames_snapshot'):
        s = getattr(CFG, sec)
        for k in s.PROTECTED_KEYS:
            check(f'config {sec}.{k} protected key exists', hasattr(s, k))
        for k, v in (s.RULED or {}).items():
            for lid in (v if isinstance(v, (list, tuple)) else (v,)):
                check(f'config {sec}.{k} ruling {lid} in fix_ledger', ledger.has_id(lid, CFG.paths.FIX_LEDGER))
        for eng, key in s.INTERPRETER.items():
            check(f'config {sec} interpreter {eng} -> site key {key}', key in ('default', 'flow', 'py36'))
    for dead in ('frames_tiers', 'frames_stack', 'frames_armA', 'frames_wideband', 'frames_tree', 'frames_stage',
                 'frames_regrid', 'frames_cube', 'frames_movies'):
        check(f'config: no section {dead} (decisions 6, 7)', dead not in CFG.sections())
    try:
        CFG.set('frames_sublst.WINDOWS', '7')
        check('protected engine literal refuses an override', False)
    except ConfigRefusal:
        check('protected engine literal refuses an override', True)
    finally:
        CFG.reset()


def test_movie_minimums():
    """B-30 (Gregg 2026-10-14): the frame-movie engine's two minimum counts are mirrored (protected) and drift-checked, and a
    band the engine made no movie of for too few staged flowed frames is COUNTED (fallbacks.FRAME_MOVIE_SKIP_FALLBACK)."""
    import re as _re
    from exopipe.phase3.core import fallback
    src = open(os.path.join(CFG.paths.CLV15_ROOT, 'phase3_snapshot_frame_movie.py')).read()
    m1 = _re.search(r'    if len\(fl\) < (\d+):\n        print\("too few staged flowed frames for", sb\)', src)
    m2 = _re.search(r'use = ok if ok\.sum\(\) >= (\d+) else', src)
    m3 = _re.search(r'    if ok\.sum\(\) < (\d+):   # recorded, not silent', src)
    S = CFG.frames_snapshot
    check('frame-movie minimums mirror the engine (MIN_FRAMES_PER_BAND; MOVIE_BACKGROUND_MIN_ADMITTED and its BACKGROUND line)',
          m1 and m2 and m3 and int(m1.group(1)) == S.MIN_FRAMES_PER_BAND and int(m2.group(1)) == S.MOVIE_BACKGROUND_MIN_ADMITTED
          and int(m3.group(1)) == S.MOVIE_BACKGROUND_MIN_ADMITTED, (m1 and m1.group(1), m2 and m2.group(1), m3 and m3.group(1)))
    lines = ['13h_2025-04-07 18MHz: 412 frames, rms 0.02', 'too few staged flowed frames for 23MHz',
             'too few staged flowed frames for 82MHz',
             'BACKGROUND 27MHz: 6 admitted frames < 10 - background from all 40 frames with data', 'TOTAL 17 movies']
    mk = fallback.mark()
    few, bg = TN.movie_skips('13h_2025-04-07', lines)
    cnts = fallback.since(mk)[1]
    mk = fallback.mark()
    none = TN.movie_skips('13h_2025-04-07', lines[:1] + lines[-1:])
    cnt0 = fallback.since(mk)[1]
    check('a band without a movie (2) and a movie with the all-frames background (1) are counted; none when every band is '
          'normal', few == ['23MHz', '82MHz'] and bg == ['27MHz'] and cnts.get('fallbacks.FRAME_MOVIE_SKIP_FALLBACK') == 2
          and cnts.get('fallbacks.FRAME_MOVIE_BACKGROUND_FALLBACK') == 1 and none == ([], []) and not cnt0, (few, bg, cnts, none))
    src_tn = open(TN.__file__).read()
    call = src_tn.find('\n        movie_skips(cell, lines)\n')         # the call statement, not the definition
    check('the movies stage calls movie_skips on the lines the engine call wrote (review 2026-10-14 m5)',
          call > 0 and src_tn.rfind('res, lines = engine(', 0, call) > src_tn.rfind('\ndef ', 0, call), call)


def test_engine_flags():
    from exopipe.phase3.core.record import Refusal
    fm = os.path.join(CFG.paths.CLV15_ROOT, 'phase3_snapshot_frame_movie.py')
    cu = os.path.join(CFG.paths.CLV15_ROOT, 'phase3_snapshot_cube.py')
    d = TN.engine_defaults(cu)
    check('engine defaults read from the source (cube --quant = QUANT = 8.0)', d.get('--quant') == 8.0, str(d))
    S = CFG.frames_snapshot
    check('config = engine defaults -> production command line (no extra argument)',
          TN.engine_flags(fm, [('--k', 'STRETCH_K', S.STRETCH_K), ('--size', 'MOVIE_WIDTH_PX', S.MOVIE_WIDTH_PX),
                               ('--fps', 'MOVIE_FPS', S.MOVIE_FPS), ('--crf', 'MOVIE_CRF', S.MOVIE_CRF)]) == []
          and TN.engine_flags(cu, [('--quant', 'CUBE_QUANT', S.CUBE_QUANT)]) == [])
    try:
        TN.engine_flags(fm, [('--k', 'STRETCH_K', 11.0)])
        check('a config value != the engine default without an override refuses (drift)', False)
    except Refusal as e:
        check('a config value != the engine default without an override refuses (drift)', 'drift' in str(e), str(e))
    try:
        CFG.set('frames_snapshot.MOVIE_CRF', '20')
        got = TN.engine_flags(fm, [('--crf', 'MOVIE_CRF', CFG.frames_snapshot.MOVIE_CRF)])
        check('an overridden value is passed to the engine', got == ['--crf', '20'], str(got))
    except ConfigRefusal as e:
        check('an overridden value is passed to the engine', False, str(e))
    finally:
        CFG.reset()


def test_stage_grammar():
    class A:
        stages = None
        from_stage = None
        until_stage = None
    a = A()
    S = CFG.frames_sublst
    check('sublst per-hour default = stack', TS.requested_stages(a, S.ORDER, S.STAGES['per-hour']) == ['stack'])
    a.stages = 'verify,tree,wideband'
    check('stages run in the tier order', TS.requested_stages(a, S.ORDER, ()) == ['wideband', 'tree', 'verify'])
    a.stages, a.from_stage = None, 'tree'
    check('--from cuts the default list', TS.requested_stages(a, S.ORDER, S.STAGES['intermittent']) == ['tree', 'verify'])
    a.stages, a.from_stage = 'armA', None
    try:
        TS.requested_stages(a, S.ORDER, ())
        check('armA is not a frames stage any more (masters loo)', False)
    except ValueError:
        check('armA is not a frames stage any more (masters loo)', True)
    a.stages = None
    check('snapshot default = stage,regrid', TS.requested_stages(a, CFG.frames_snapshot.ORDER,
                                                                 CFG.frames_snapshot.DEFAULT_STAGES) == ['stage', 'regrid'])


def test_items():
    class A:
        epochs = '09h_2025-01-01,09h_2025-01-02'
        cells = None
        lsts = None
        scopes = None
    by, lsts = TS.parse_items(A())
    check('sublst epochs -> nights of an LST', by == {'09h': ['2025-01-01', '2025-01-02']} and lsts == [])
    a = A()
    a.epochs, a.cells = None, '18h_2025-04-11_36MHz,18h_2025-04-11_18MHz'
    it = TN.parse_items(a)
    check('snapshot night cells with subbands merge per cell', it == [('18h_2025-04-11', '18h', '2025-04-11',
                                                                       ['18MHz', '36MHz'])], str(it))
    a.cells, a.epochs = None, '18h_2025-04-11'
    it = TN.parse_items(a)
    check('snapshot epoch = every subband', len(it[0][3]) == len(CFG.frames_snapshot.SUBBANDS))
    a.epochs = '18h_2025-04-11_36MHz'
    try:
        TN.parse_items(a)
        check('an epoch item takes no subband suffix', False)
    except ValueError:
        check('an epoch item takes no subband suffix', True)


# =====================================================================================================================
# 2. sub-LST: synthetic LST, the real wideband and tree engines, content tests (b)
# =====================================================================================================================

def usable_nights(n=3):
    from phase3_tier_rules import membership_vetoed
    ok = None
    for sb in ('18MHz', '23MHz', 'Red'):
        rows = list(csv.DictReader(open(XS._frozen_weights(LST, sb))))
        good = {r['date'] for r in rows if str(r.get('in_master', '1')).strip() != '0' and float(r.get('weight') or 0) > 0
                and not membership_vetoed(LST, sb, r['date']) and not exclusions.is_excluded(LST, sb, r['date'])}
        ok = good if ok is None else ok & good
    return sorted(ok)[:n]


def build_sublst(root, nights):
    """Window 0 of two Red members (18MHz, 23MHz), three nights, both Stokes: stack, _pb, _nframes, arm-A and arm-B
    residuals with the cards the engines write; loo and refine logs for every Red member (production evidence)."""
    rng = np.random.default_rng(7)
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    pb = np.clip(1.0 - ((xx - 32) ** 2 + (yy - 32) ** 2) / 2500.0, 0.0, 1.0).astype(np.float32)
    for sb, fc in (('18MHz', 1.05), ('23MHz', 1.06)):
        for ni, night in enumerate(nights):
            for S in ('I', 'V'):
                base = f'sublst_{LST}_{night}_{sb}_{S}_w0'
                d = os.path.join(root, LST, f'{LST}_{night}', sb, S)
                h = {'CTYPE1': 'RA---SIN', 'CTYPE2': 'DEC--SIN', 'CRVAL1': 135.0, 'CRVAL2': 37.2, 'CRPIX1': 33, 'CRPIX2': 33,
                     'CDELT1': -1.0, 'CDELT2': 1.0, 'BUNIT': 'JY/BEAM', 'DATE-OBS': f'{night}T15:00:00.0',
                     'DATE-END': f'{night}T15:09:50.0', 'LSTHOUR': LST, 'LSTWIN': 0, 'NFRAMES': 58, 'BMAJ': 0.3 + 0.01 * ni,
                     'BMIN': 0.2, 'BPA': 10.0 + ni, 'SHSCALE': 0.98528, 'FLUXABS': 1.034, 'FLUXCORR': fc, 'FLUXVER': 'v2',
                     'FLUXBAKE': True, 'FLUXDC': 0.93, 'PRODUCT': 'sublst-v0-test'}
                img = rng.normal(0, 1.0 + 0.5 * (sb == '23MHz'), (n, n))
                fits_write(os.path.join(d, base + '.fits'), img, h)
                fits_write(os.path.join(d, base + '_pb.fits'), pb, dict(h, BUNIT=''))
                fits_write(os.path.join(d, base + '_nframes.fits'), np.full((n, n), 58.0), h)
                rh = dict(h, SUBTRACT='leave-one-out window Master', LOONCOAD=2,
                          LOOMSTR=f'sublst_master_{LST}_{sb}_{S}_w0.fits', JUPMASK=25.0, LSTBIN=0, DURATION=590.0)
                fits_write(os.path.join(d, base + '_resid_loo.fits'), img * 0.1, rh)
                dx = os.path.join(root, f'{LST}_x2', f'{LST}_{night}', sb, S)
                fits_write(os.path.join(dx, base + '_resid_loo.fits'), img * 0.1 + 0.01,
                           dict(rh, REFINE='x2 flow to the subtracted LOO window Master'))
    time.sleep(1.1)
    for sb in cells.band_members('Red'):
        lg = os.path.join(root, LST, 'logs')
        os.makedirs(lg, exist_ok=True)
        for S in ('I', 'V'):
            with open(os.path.join(lg, f'loo_{LST}_{sb}_{S}.log'), 'w') as f:
                f.write(f'SUBLST-LOO {LST} {sb} {S} windows=1 residuals=3 refused_night=0\n')
        xl = os.path.join(root, f'{LST}_x2', 'logs')
        os.makedirs(xl, exist_ok=True)
        with open(os.path.join(xl, f'{LST}_{sb}.log'), 'w') as f:
            f.write(f'SUBLST-REFINE {LST} {sb} aligned=3 refused=0 refused_rule=0 refused_night=0\n')


def test_sublst_engines():
    nights = usable_nights()
    check('three real 09h nights usable in the frozen records (18MHz, 23MHz, Red)', len(nights) == 3, str(nights))
    if len(nights) < 3:
        return
    root = os.path.join(T, 'sublst')
    build_sublst(root, nights)
    common = ['--tier', 'sublst', '--lsts', LST, '--scopes', '18MHz,23MHz,Red', '--sublst-loo-evidence', 'log',
              '--out', root, '--parity']
    # gate: arm A evidence 'record' (default) and no masters loo record -> wideband refused
    rc, out = exopipe3('--tier', 'sublst', '--lsts', LST, '--scopes', '18MHz,23MHz,Red', '--stages', 'wideband',
                       '--out', root, '--parity')
    recs = records(root, f'{LST}_wideband')
    check('wideband refuses without the masters loo record (decision 6)', rc == 2 and
          recs.get(f'{LST}_wideband_Red', {}).get('status') == 'refused', out[-400:])
    shutil.rmtree(os.path.join(root, '_records'), ignore_errors=True)
    if QUICK:
        return
    t0 = time.time()
    rc, out = exopipe3(*common, '--stages', 'wideband,tree,verify')
    recs = records(root)
    st = {k: v['status'] for k, v in recs.items()}
    check('sublst wideband,tree,verify exit 0', rc == 0, f'rc {rc}; {st}; {out[-1500:]}')
    check('wideband Red done', st.get(f'{LST}_wideband_Red') == 'done', str(st))
    for b in ('18MHz', '23MHz', 'Red'):
        check(f'tree {b} done', st.get(f'{LST}_tree_{b}') == 'done', str(st))
    check('verify (content test (b)) done', st.get(f'{LST}_verify') == 'done', str(recs.get(f'{LST}_verify', {}).get('reason')))
    wb = recs.get(f'{LST}_wideband_Red', {})
    outs = [o['path'] for o in wb.get('outputs', [])]
    check('wideband wrote 3 nights x 2 Stokes x 5 products', len(outs) == 30, str(len(outs)))
    check('wideband outputs all passed sublst_wideband', all(t['passed'] for t in wb.get('tests', [])) and wb.get('tests'))
    check('every engine call recorded with its argv (core.legacy.run)', len(wb.get('meta', {}).get('legacy_calls', [])) == 3)
    rp = os.path.join(root, LST, f'{LST}_{nights[0]}', 'Red', 'I', f'sublst_{LST}_{nights[0]}_Red_I_w0.fits')
    from astropy.io import fits
    h = fits.getheader(rp)
    check('wideband members and weights from the engine', h.get('WBMEMB') == '18MHz,23MHz' and h.get('WIDEBAND') == 'Red')
    fc = os.path.join(root, CFG.frames_sublst.TREE_SUBDIR, f'{LST}_Red', 'frames.csv')
    rows = list(csv.DictReader(open(fc))) if os.path.exists(fc) else []
    check('tree Red frames.csv: 3 nights x 2 Stokes', len(rows) == 6, str(len(rows)))
    print(f'    (sub-LST engine runs: {time.time() - t0:.1f} s)')
    # negative content tests on the products just made
    r = validate.run('sublst_wideband', rp)
    check('sublst_wideband passes on the engine output', all(x['passed'] for x in r), str(r))
    with fits.open(rp, mode='update') as hl:
        hl[0].data[30:34, 30:34] += 5.0
    r = validate.run('sublst_wideband', rp)
    check('sublst_wideband fails on altered pixels', not all(x['passed'] for x in r))
    victim = glob.glob(os.path.join(root, LST, f'{LST}_{nights[1]}', '23MHz', 'V', '*_resid_loo.fits'))[0]
    os.remove(victim)
    ok, n, det = XS.residual_set(os.path.join(root, LST))
    check('content test (b) R2 fails when an allowed arm-A residual is missing', not ok and 'arm A' in det, det[:300])
    link = rows[0]['sub_path'] if rows else None
    if link:
        os.remove(rows[0]['src'])
        r = validate.run('tenmin_tree', fc)
        check('tenmin_tree fails when a linked arm-B residual vanishes', not all(x['passed'] for x in r))
    # S-G8: an incomplete refine refuses wideband
    with open(os.path.join(root, f'{LST}_x2', 'logs', f'{LST}_27MHz.log'), 'w') as f:
        f.write(f'SUBLST-REFINE {LST} 27MHz aligned=2 refused=1 refused_rule=0 refused_night=0\nREFUSED-NIGHT 27MHz w0 x: no flow returned\n')
    rc, out = exopipe3(*common, '--stages', 'wideband', '--fresh')
    check('S-G8: an unapproved refine refusal refuses wideband', rc == 2 and 'S-G8' in out, out[-600:])
    rc, out = exopipe3('--tier', 'sublst', '--lsts', LST, '--scopes', 'Red', '--sublst-loo-evidence', 'log', '--out', root,
                       '--parity', '--stages', 'tree')
    check('S-G8: tree B of a group refuses while a member\'s refine is incomplete (nocat:236-247 before :264)',
          rc == 2 and 'S-G8' in out and records(root).get(f'{LST}_tree_Red', {}).get('status') == 'refused', out[-600:])
    os.utime(os.path.join(root, f'{LST}_x2', 'logs', f'{LST}_18MHz.log'), (1, 1))
    rc, out = exopipe3('--tier', 'sublst', '--lsts', LST, '--scopes', '18MHz', '--sublst-loo-evidence', 'log', '--out', root,
                       '--parity', '--stages', 'tree')
    check('S-G8: tree B refuses on a refine log older than the window stacks', rc == 2 and 'older than' in out, out[-600:])


# =====================================================================================================================
# 3. snapshot: synthetic Phase-1 night cell, stage -> regrid -> (simulated flow) -> cube -> movies, real engines
# =====================================================================================================================

CELL, SLST, SDATE = '18h_2030-06-21', '18h', '2030-06-21'


def build_phase1(p1, nframes=12):
    """One Phase-1 run of 18MHz frames, new era (Clean = I in snapshots_clean, Pilot = V in snapshots), 64 px at 2 deg
    (off the 1508-px Master grid, so the regrid runs), half of them fpack-compressed (.fz) so funpack runs."""
    rd = os.path.join(p1, SLST, SDATE, 'Run_20300622_000000', '18MHz')
    rng = np.random.default_rng(3)
    fpack = os.path.join(os.path.dirname(CFG.site.PYTHON['py36']), 'fpack')
    tss = []
    for i in range(nframes):
        t = 8 * 3600 + 10 * i
        ts = f'20300621_{t // 3600:02d}{t % 3600 // 60:02d}{t % 60:02d}'
        tss.append(ts)
        dob = f'2030-06-21T{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}.0'
        for kind, sub in (('Clean', 'snapshots_clean'), ('Pilot', 'snapshots')):
            p = os.path.join(rd, sub, f'18MHz-{kind}-Snapshot-{ts}-image.fits')
            h = {'CTYPE1': 'RA---SIN', 'CTYPE2': 'DEC--SIN', 'CRVAL1': 270.0, 'CRVAL2': 37.24, 'CRPIX1': 32.5, 'CRPIX2': 32.5,
                 'CDELT1': -2.0, 'CDELT2': 2.0, 'CUNIT1': 'deg', 'CUNIT2': 'deg', 'BUNIT': 'JY/BEAM', 'DATE-OBS': dob,
                 'BMAJ': 0.3, 'BMIN': 0.25, 'BPA': 5.0, 'WSCNVIS': 0 if i == 5 else 1000}
            fits_write(p, rng.normal(0, 1, (64, 64)), h)
            if i % 2:
                subprocess.run([fpack, '-D', '-Y', p], check=True)           # -> .fz, original removed
    return tss


def simulate_flow(stage, scratch, tss):
    """What `flow --tier snapshot` leaves: <raw>_flowed.fits beside each staged frame (FLOWED, FLOWREF) and flow fields
    flow_<sb>_<ts>.npy (every 8th pixel) in <scratch>/<cell>/flow."""
    from astropy.io import fits
    sd = os.path.join(stage, '18MHz')
    fd = CFG.frames_snapshot.FLOW_DIR.format(scratch=scratch, cell=CELL)
    os.makedirs(fd, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(sd, '*-image.fits'))):
        with fits.open(p) as hl:
            d, h = hl[0].data.copy(), hl[0].header.copy()
        h['FLOWED'] = (True, 'synthetic flow')
        h['FLOWREF'] = 'I_deep_Taper_R0_18MHz.fits'
        fits.PrimaryHDU(d, h).writeto(p[:-5] + '_flowed.fits', overwrite=True)
    n = 1508 // 8 + 1
    for ts in tss:
        np.save(os.path.join(fd, f'flow_18MHz_{ts}.npy'), (np.random.default_rng(1).normal(0, 0.05, (n, n, 2))).astype(np.float16))
    return fd


def test_snapshot_engines():
    p1, scratch, out = os.path.join(T, 'phase1'), os.path.join(T, 'scratch'), os.path.join(T, 'snapout')
    tss = build_phase1(p1)
    common = ['--tier', 'snapshot', '--cells', f'{CELL}_18MHz', '--snap-discovery', 'integration',
              '--set', f'paths.PHASE1_ROOT={p1}', '--set', 'frames_snapshot.MIN_FREE_GB[search]=0',
              '--work', scratch, '--out', out, '--parity']
    rc, o = exopipe3(*common, '--stages', 'stage')
    check('N-G3: 24 frames < frames_snapshot.MIN_FRAMES[search] = 1000 refuses the stage', rc == 2 and 'N-G3' in o, o[-500:])
    shutil.rmtree(os.path.join(scratch, CFG.frames_snapshot.STAGE_SUBDIR), ignore_errors=True)
    shutil.rmtree(os.path.join(out, '_records'), ignore_errors=True)
    common += ['--set', 'frames_snapshot.MIN_FRAMES[search]=4']
    if QUICK:
        rc, o = exopipe3(*common, '--stages', 'stage')
    else:
        rc, o = exopipe3(*common, '--stages', 'stage,regrid')
    recs = records(out)
    stg = recs.get(f'{CELL}_stage', {})
    check('stage done', stg.get('status') == 'done', f'rc {rc}: {stg.get("reason")} {o[-800:]}')
    sd = os.path.join(scratch, CFG.frames_snapshot.STAGE_SUBDIR, CELL, '18MHz')
    names = sorted(f for f in os.listdir(sd) if f.endswith('-image.fits')) if os.path.isdir(sd) else []
    check('staged 24 decompressed frames, no .fz left', len(names) == 24 and not glob.glob(os.path.join(sd, '*.f[sz]')))
    meta = stg.get('meta', {}).get('subbands', {}).get('18MHz', {})
    check('stage record: I/V counts and the WSCNVIS = 0 frames (sets for flow)', meta.get('I') == 12 and meta.get('V') == 12
          and len(meta.get('wscnvis0', [])) == 2, str(meta)[:300])
    check('snapshot_stage passes', all(x['passed'] for x in validate.run('snapshot_stage', sd)))
    rc, o = exopipe3(*common, '--stages', 'stage')
    check('re-staging without --fresh keeps an intact done staging (skipped-valid, record kept)',
          rc == 0 and 'skipped' in o and records(out).get(f'{CELL}_stage', {}).get('status') == 'done', o[-300:])
    if QUICK:
        return
    rg = recs.get(f'{CELL}_regrid', {})
    check('regrid done (64-px pilot-like frames onto the own Master grid)', rg.get('status') == 'done',
          f'{rg.get("reason")} {o[-800:]}')
    check('snapshot_regrid passes', all(x['passed'] for x in validate.run('snapshot_regrid', sd)))
    fd = simulate_flow(os.path.join(scratch, CFG.frames_snapshot.STAGE_SUBDIR, CELL), scratch, tss)
    t0 = time.time()
    rc, o = exopipe3(*common, '--stages', 'cube,movies')
    recs = records(out)
    cu, mv = recs.get(f'{CELL}_cube', {}), recs.get(f'{CELL}_movies', {})
    check('cube done (write + verify + ship + MANIFEST)', cu.get('status') == 'done', f'{cu.get("reason")} {o[-1500:]}')
    dest = os.path.join(out, 'snapshot_cubes', CELL)
    man = open(os.path.join(dest, 'MANIFEST.txt')).read().splitlines() if os.path.exists(os.path.join(dest, 'MANIFEST.txt')) else []
    check('MANIFEST: 2 cube lines copy-OK -> OK, COMPLETE cubes=2 subbands=1', len(man) == 3 and
          all('copy-OK' in l and l.endswith('-> OK') for l in man[:2]) and man[2].startswith(f'COMPLETE {CELL} cubes=2 subbands=1'),
          str(man)[:400])
    check('cube outputs passed snapshot_cube', cu.get('tests') and all(t['passed'] for t in cu.get('tests', [])))
    check('movies done (frame movies + flow movie)', mv.get('status') == 'done', f'{mv.get("reason")} {o[-1500:]}')
    mvo = [os.path.basename(x['path']) for x in mv.get('outputs', [])]
    check('frame movie of 18MHz and movies.log; flow_18MHz.mp4 shipped', f'frames_{CELL}_18MHz_I.mp4' in mvo and
          'movies.log' in mvo and 'flow_18MHz.mp4' in mvo, str(mvo))
    check('flow_movie passes with the flow dir', all(x['passed'] for x in validate.run(
        'flow_movie', os.path.join(out, 'snapshot_fleet_out', CELL, 'flow', 'flow_18MHz.mp4'), flow_dir=fd)))
    print(f'    (snapshot cube + movie engine runs: {time.time() - t0:.1f} s)')
    fmv = os.path.join(out, 'snapshot_fleet_out', CELL, 'flow', 'flow_18MHz.mp4')
    flog = os.path.join(out, 'snapshot_fleet_out', CELL, 'flow', 'movie_18MHz.log')
    ftxt = open(flog).read() if os.path.exists(flog) else ''
    check('flow movie log shipped without the legacy.run header, pixel scale of the own grid (M31)',
          not ftxt.lstrip().startswith('=====') and 'pixel scale 4.6020 arcmin' in ftxt, ftxt[:300])
    mlog = open(os.path.join(out, 'snapshot_movies', CELL, 'movies.log')).read()
    check('movies.log as production writes it (no legacy.run header line)', '=====' not in mlog, mlog[:200])
    with open(flog, 'w') as f:
        f.write(ftxt.replace('pixel scale 4.6020', 'pixel scale 2.9400'))
    check('flow_movie fails when its pixel scale is not the header frame\'s |CDELT2| (M31)',
          not all(x['passed'] for x in validate.run('flow_movie', fmv, flow_dir=fd)))
    with open(flog, 'w') as f:
        f.write(ftxt)
    rc, o = exopipe3(*common, '--stages', 'movies')
    mv = records(out).get(f'{CELL}_movies', {})
    check('a re-run keeps earlier shipped flow movies (cp -n) and refuses that item by name',
          mv.get('status') == 'done' and any('cp -n' in r['reason'] for r in mv.get('refused_items', [])),
          f'{mv.get("status")} {mv.get("refused_items")} {o[-400:]}')
    for p in glob.glob(os.path.join(sd, '*Clean-Snapshot-*-image_flowed.fits')):
        os.remove(p)
    rc, o = exopipe3(*common, '--stages', 'movies', '--fresh')
    mv = records(out).get(f'{CELL}_movies', {})
    check('no flowed Stokes I frame: the flow movie item is refused (no 4.602 arcmin fallback)',
          any('no flowed Stokes I frame' in r['reason'] for r in mv.get('refused_items', [])),
          f'{mv.get("status")} {mv.get("refused_items")} {o[-400:]}')
    rc, o = exopipe3('--tier', 'snapshot', '--cells', f'{CELL}_23MHz', '--snap-discovery', 'integration',
                     '--set', f'paths.PHASE1_ROOT={p1}', '--set', 'frames_snapshot.MIN_FREE_GB[search]=0',
                     '--work', os.path.join(T, 'scratch0'), '--out', os.path.join(T, 'snapout0'), '--parity', '--stages', 'stage')
    st0 = records(os.path.join(T, 'snapout0')).get(f'{CELL}_stage', {})
    check('N-G2: a cell with no Phase-1 frame refuses the stage (cube pass :61)', rc == 2 and st0.get('status') == 'refused'
          and 'N-G2' in st0.get('reason', ''), f'{rc} {st0.get("reason")} {o[-300:]}')
    cube = os.path.join(dest, f'{CELL}_18MHz_I_flowcube.fits')
    with open(os.path.join(dest, 'MANIFEST.txt')) as f:
        txt = f.read()
    with open(os.path.join(dest, 'MANIFEST.txt'), 'w') as f:
        f.write(txt.replace('copy-OK', 'COPY-MISMATCH', 1))
    check('snapshot_cube fails when its MANIFEST line is not copy-OK', not all(x['passed'] for x in validate.run('snapshot_cube', cube)))
    os.remove(os.path.join(sd, names[0]))
    rc, o = exopipe3(*common, '--stages', 'stage')
    check('a staging that no longer matches its record refuses a re-stage without --fresh', rc == 2 and 'earlier run' in o,
          o[-300:])
    rc, o = exopipe3(*common, '--stages', 'cube')
    check('cube refuses on a staging whose file set changed since its record', rc == 2, o[-300:])


# =====================================================================================================================
# 4. read-only checks on production products (--real)
# =====================================================================================================================

def test_real():
    SL = CFG.paths.SUBLST_ROOT
    for cls, p in (('sublst_wideband', f'{SL}/09h/09h_2025-01-01/Red/I/sublst_09h_2025-01-01_Red_I_w0.fits'),
                   ('sublst_wideband', f'{SL}/09h_x2/09h_2025-01-01/Blue/V/sublst_09h_2025-01-01_Blue_V_w4_resid_loo.fits'),
                   ('tenmin_tree', f'{SL}/tenmin_root_x2/09h_55MHz/frames.csv'),
                   ('tenmin_tree', f'{SL}/tenmin_root_x2/09h_55MHz/bin_masters/PB_bin0_55MHz.fits'),
                   ('tenmin_tree', f'{SL}/tenmin_root_x2/01h_18MHz/NO_WINDOWS.txt'),
                   ('snapshot_cube', f'{CFG.paths.SNAPSHOT_CUBES_ROOT}/18h_2025-04-11/MANIFEST.txt'),
                   ('snapshot_cube', f'{CFG.paths.SNAPSHOT_CUBES_ROOT}/18h_2025-04-11/18h_2025-04-11_36MHz_V_flowcube.fits'),
                   ('frame_movie', f'{CFG.frames_snapshot.MOVIES_ROOT}/18h_2025-04-11/movies.log'),
                   ('flow_movie', f'{CFG.paths.SNAPSHOT_FLEET_OUT}/18h_2025-10-15/flow/flow_46MHz.mp4')):
        r = validate.run(cls, p)
        check(f'real {cls} {os.path.basename(p)}', all(x['passed'] for x in r), str(r)[:300])
    # a flow movie shipped before the M31 fix (clv15 98595b3, 2026-10-13 01:40) used the 18 MHz pixel at 36 MHz
    r = validate.run('flow_movie', f'{CFG.paths.SNAPSHOT_FLEET_OUT}/18h_2025-04-11/flow/flow_36MHz.mp4')
    check('real pre-M31 flow movie 18h_2025-04-11 36MHz fails the scale test', not all(x['passed'] for x in r)
          and 'pixel-scale' in str(r), str(r)[:300])
    rc, o = exopipe3('--tier', 'sublst', '--lsts', '09h', '--stages', 'tree,verify', '--scopes', '55MHz', '--dry-run')
    check('dry run on real LST 09h (tree, verify)', rc == 0 and 'UNIT 09h_tree_55MHz' in o, o[-500:])
    rc, o = exopipe3('--tier', 'snapshot', '--cells', '18h_2025-04-11_36MHz', '--stages', 'stage,regrid',
                     '--work', os.path.join(T, 'dry'), '--out', os.path.join(T, 'dry', 'out'), '--dry-run')
    check('dry run on real night cell 18h_2025-04-11 (36MHz): discovery 690 frames', '36MHz: 690 frames' in o, o[-800:])


def main():
    t0 = time.time()
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_config()
        test_movie_minimums()
        test_engine_flags()
        test_stage_grammar()
        test_items()
        test_sublst_engines()
        test_snapshot_engines()
        if REAL:
            test_real()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    bad = [r for r in RESULTS if not r[1]]
    print(f'\n{len(RESULTS) - len(bad)} of {len(RESULTS)} checks passed in {time.time() - t0:.1f} s')
    for r in bad:
        print('FAILED:', r[0], r[2][:500])
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
