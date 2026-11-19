#!/usr/bin/env python3
"""The hybrid route, pull request PR-2 (plan/specs/hybrid.md section 4): the stack stage (lib/hybrid/stack.py, warp.py, common.py,
the content tests in lib/hybrid/tests.py and steps/hybrid.stack_unit).

Data-free (CI):
  1. the stack rebuild test (HG5): exact pass, the flow-recovered limits, the hybrid-scale admission of a flow-recovered high-band
     NoTaper stack, and refusals (limits missed, a NaN-mask difference, flows not recomputed);
  2. the alignment offsets: peak_offsets on synthetic sources (0 px on the same frame, the shift otherwise) and the acceptance rule;
  3. the hybrid filter G: 1 below hybrid.HYB_U_IN wavelengths, 0 above HYB_U_OUT, the cosine between; lowpass leaves a constant
     interior unchanged and never wraps around;
  4. membership (HG1): weights.csv members equal to the master's 'COADD constituent' nights and NCOADD; a variant-gate refusal and a
     night without an image of the variant leave the night out; a mismatch or an excluded night refuses;
  5. the Phase 1 file choice: the newest Phase 1 run older than the night's Phase 2 science run; none or several refuse (HG4);
  6. the unit: a variant a gate refuses is a refused item (its files, with the gate) while the other variant is written and tested;
     both refused = the unit refused; the content tests of the four product classes pass on good files and fail on bad ones.
Cluster parity (not here): plan/specs/hybrid.md PR-2 targets, run by hand on lwacalim07 (the node holding hybrid_stack.COADD_SB).
Run: python tests/test_hybrid_stack.py   (or under pytest)
"""
import csv
import json
import os
import shutil
import sys
import tempfile
import time
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from astropy.io import fits  # noqa: E402

from exopipe.phase3.config import CFG  # noqa: E402
from exopipe.phase3.core import record  # noqa: E402
from exopipe.phase3.lib.hybrid import common as HC  # noqa: E402
from exopipe.phase3.lib.hybrid import stack as HST  # noqa: E402
from exopipe.phase3.lib.hybrid import warp as HW  # noqa: E402
from exopipe.phase3.lib.install import steps_common as SC  # noqa: E402
from exopipe.phase3.steps import hybrid as HS  # noqa: E402


def _tmp(tag):
    return tempfile.mkdtemp(prefix=f'hy2_{tag}_', dir=os.environ.get('TMPDIR') or None)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc as e:
        return e
    raise AssertionError(f'{fn.__name__} did not raise {exc.__name__}')


def _gauss(N, srcs, fwhm=3.0):
    y, x = np.mgrid[0:N, 0:N]
    s = fwhm / 2.3548
    return sum(a * np.exp(-0.5 * (((y - y0) / s) ** 2 + ((x - x0) / s) ** 2)) for y0, x0, a in srcs)


def _cell(d, dates, flow_age_s):
    """A co-add cell directory: coadd_final/weights.csv and flow files flow_age_s newer than weights.csv."""
    cf = os.path.join(d, 'coadd_final')
    os.makedirs(cf)
    os.makedirs(os.path.join(d, 'flow'))
    with open(os.path.join(cf, 'weights.csv'), 'w') as f:
        f.write('date,in_master,weight,qa3_scale,flux_dc,qa3_guard,sci_dir\n')
        for dt in dates:
            f.write(f'{dt},1,1.0,1.0,,,/x/Science_20250101_000000\n')
    t = time.time() - 10 * 86400
    os.utime(os.path.join(cf, 'weights.csv'), (t, t))
    for dt in dates:
        p = os.path.join(d, 'flow', f'{dt}_scaled_flow.npy')
        np.save(p, np.zeros((2, 2, 2)))
        os.utime(p, (t + flow_age_s, t + flow_age_s))
    with open(os.path.join(cf, 'weights.csv')) as f:
        rows = list(csv.DictReader(f))
    return d, cf, rows


# ------------------------------------------------------------------------------------------------------------- 1 rebuild test
def test_rebuild_verdicts():
    rng = np.random.default_rng(1)
    N = 96
    master = _gauss(N, [(48, 48, 50.0), (30, 60, 20.0)]) + rng.normal(0, 1.0, (N, N))
    master[:3, :] = np.nan
    d = _tmp('rb')
    try:
        wd, cf, rows = _cell(os.path.join(d, 'old'), ['2025-01-01', '2025-01-02'], 0)
        r = HST.test_stack_rebuild(master.copy(), master, wd, cf, rows)
        assert r['verdict'] == 'pass (exact)', r
        # a 1.5% gain error: misses the exact limits; flows not recomputed -> refuses HG5
        e = _raises(HST.StackRefusal, HST.test_stack_rebuild, master * 1.015, master, wd, cf, rows)
        assert e.gate == 'HG5' and 'FAILED' in str(e), e
        # flows recomputed after the build (> FLOW_RECOVERED_AGE_S newer): a 0.1% ratio passes, a 1% ratio does not
        wd2, cf2, rows2 = _cell(os.path.join(d, 'rec'), ['2025-01-01', '2025-01-02'], CFG.hybrid_stack.FLOW_RECOVERED_AGE_S + 60)
        r = HST.test_stack_rebuild(master * 1.001, master, wd2, cf2, rows2)
        assert r['verdict'] == 'pass (flow-recovered cell)' and r['flows_recomputed_after_build'], r
        _raises(HST.StackRefusal, HST.test_stack_rebuild, master * 1.01, master, wd2, cf2, rows2)
        # the same 1% mismatch of a high-band NoTaper stack (admit_hdr): admitted at hybrid scales, recorded for the recon stage
        hdr = fits.Header({'CDELT1': -0.0625, 'CDELT2': 0.0625, 'CRPIX1': N / 2 + 1, 'CRPIX2': N / 2 + 1})
        r = HST.test_stack_rebuild(master * 1.01, master, wd2, cf2, rows2, admit_hdr=hdr)
        assert r['verdict'].startswith('admitted at hybrid scales') and 'lowpassed_rms' in r['hybrid_scale_admission'], r
        assert r['verdict'] in HST.VERDICTS
        # not admitted when the flows were not recomputed, nor with a NaN-mask difference
        _raises(HST.StackRefusal, HST.test_stack_rebuild, master * 1.01, master, wd, cf, rows, admit_hdr=hdr)
        bad = master * 1.0
        bad[50, 50] = np.nan
        _raises(HST.StackRefusal, HST.test_stack_rebuild, bad, master, wd2, cf2, rows2, admit_hdr=hdr)
        # a reference-identity night has no flow: with only such nights, not flow-recovered
        rows3 = [dict(r, qa3_guard='reference-identity') for r in rows2]
        _raises(HST.StackRefusal, HST.test_stack_rebuild, master * 1.001, master, wd2, cf2, rows3)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------- 2 offsets
def test_peak_offsets_and_rule():
    rng = np.random.default_rng(2)
    N = 200
    src = [(100, 100, 200.0), (60, 100, 200.0), (140, 100, 200.0), (100, 60, 200.0), (100, 140, 200.0), (70, 70, 200.0),
           (130, 130, 200.0)]                       # 40 px apart or more: one source per 25-px search box
    B = _gauss(N, src) + rng.normal(0, 1.0, (N, N))
    d0 = HW.peak_offsets(B.copy(), B)
    assert d0.size == 7 and np.all(d0 == 0), d0
    A = _gauss(N, [(y, x + 2, a) for y, x, a in src]) + rng.normal(0, 1.0, (N, N))
    d2 = HW.peak_offsets(A, B)
    assert d2.size == 7 and np.allclose(d2, 2.0), d2
    assert HW.offsets_pass(d0) and not HW.offsets_pass(d2)
    assert not HW.offsets_pass(d0[:CFG.hybrid_stack.WARP_MIN_SOURCES - 1])      # too few sources
    s = HW.offsets_summary(np.array([]))
    assert s == dict(n=0, median_px=None, p90_px=None), s


# ------------------------------------------------------------------------------------------------------------- 3 filter
def test_hybrid_filter_and_lowpass():
    N, cd = 256, 0.25
    G = HC.hybrid_filter(N, cd)
    u = np.hypot(np.fft.fftfreq(N)[:, None], np.fft.rfftfreq(N)[None, :]) / np.radians(cd)
    assert np.all(G[u <= CFG.hybrid.HYB_U_IN] == 1.0) and np.all(G[u >= CFG.hybrid.HYB_U_OUT] == 0.0)
    mid = (u > CFG.hybrid.HYB_U_IN) & (u < CFG.hybrid.HYB_U_OUT)
    assert mid.any() and np.all((G[mid] > 0) & (G[mid] < 1))
    um = 0.5 * (CFG.hybrid.HYB_U_IN + CFG.hybrid.HYB_U_OUT)
    k = np.argmin(np.abs(u - um))
    assert abs(G.ravel()[k] - 0.5 * (1 + np.cos(np.pi * (u.ravel()[k] - CFG.hybrid.HYB_U_IN) /
                                                 (CFG.hybrid.HYB_U_OUT - CFG.hybrid.HYB_U_IN)))) < 1e-12
    # a source at one edge does not appear at the opposite edge (linear convolution, no wrap-around)
    a = np.zeros((N, N))
    a[N // 2, 2] = 1000.0
    lp = HC.lowpass(a, cd)
    assert abs(lp[N // 2, N - 3]) < 1e-6 * abs(lp[N // 2, 2]), (lp[N // 2, N - 3], lp[N // 2, 2])
    ok = np.ones((N, N), bool)
    hdr = {'CRPIX1': N / 2 + 1, 'CRPIX2': N / 2 + 1, 'CDELT1': -0.5, 'CDELT2': 0.5}   # corner 1.58 rad from the center: off the sky
    inner = HC.sky_interior(ok, hdr)
    assert inner[N // 2, N // 2] and not inner[0, 0]


# ------------------------------------------------------------------------------------------------------------- 4 membership
class _Excl:
    def __init__(self, bad):
        self.bad = set(bad)

    def excluded_constituents(self, lst, scope, dates):
        return sorted(d for d in dates if (lst, scope, d) in self.bad)


def _master(path, dates, ncoadd=None, gate_refused=()):
    h = fits.Header()
    h['NCOADD'] = len(dates) if ncoadd is None else ncoadd
    for dt in dates:
        h.add_history(f'COADD constituent {dt}')
    for dt in gate_refused:
        h.add_history(f'VARIANT-GATE refused {dt}')
    fits.writeto(path, np.zeros((4, 4), np.float32), h, overwrite=True)


def test_members_of():
    d = _tmp('mem')
    orig = HST._CFP
    try:
        sb = os.path.join(d, 'coadd_sb')
        cell = os.path.join(sb, '18h_46MHz')
        dates = ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04']
        _cell(cell, dates, 0)
        mfile = HST.cell_paths(sb, '18h', '46MHz', 'R075')[2]
        noimg = {'2025-01-04'}
        from exopipe.phase3.lib.masters import coadd as realCFP
        # the Phase 2 finder answers per night through the row's sci_dir (2025-01-04 has no image of the variant)
        stub = types.SimpleNamespace(find_subband_variant_image=lambda sci, f, v: None if any(n in sci for n in noimg) else 'img',
                                     _constituent_dates=realCFP._constituent_dates)
        HST._CFP = lambda: stub
        with open(os.path.join(cell, 'coadd_final', 'weights.csv'), 'w') as f:
            f.write('date,in_master,weight,qa3_scale,flux_dc,qa3_guard,sci_dir\n')
            for dt in dates:
                f.write(f'{dt},1,1.0,1.0,,,/x/{dt}/Science_20250101_000000\n')
        # the master: 2025-01-02 refused by the variant gate; 2025-01-04 has no image -> constituents 01, 03
        _master(mfile, ['2025-01-01', '2025-01-03'], gate_refused=['2025-01-02'])
        rows, refused = HST.members_of(sb, '18h', '46MHz', 'R075', excl=_Excl([]))
        assert [r['date'] for r in rows] == ['2025-01-01', '2025-01-03'], rows
        assert refused == ['2025-01-02', '2025-01-04 (no NoTaper_R075 image)'], refused
        # an excluded constituent refuses (HG1)
        e = _raises(HST.StackRefusal, HST.members_of, sb, '18h', '46MHz', 'R075', excl=_Excl([('18h', '46MHz', '2025-01-03')]))
        assert e.gate == 'HG1' and 'excluded' in str(e), e
        # the master holds a night that is not a member / NCOADD disagrees: refuses (HG1)
        _master(mfile, ['2025-01-01', '2025-01-03', '2025-01-05'], gate_refused=['2025-01-02'])
        e = _raises(HST.StackRefusal, HST.members_of, sb, '18h', '46MHz', 'R075', excl=_Excl([]))
        assert e.gate == 'HG1' and '!= master constituents' in str(e), e
        _master(mfile, ['2025-01-01', '2025-01-03'], ncoadd=3, gate_refused=['2025-01-02'])
        _raises(HST.StackRefusal, HST.members_of, sb, '18h', '46MHz', 'R075', excl=_Excl([]))
        # no built cell (weights.csv or the master missing): a named refusal (HG1), not an exception
        os.remove(mfile)
        e = _raises(HST.StackRefusal, HST.members_of, sb, '18h', '46MHz', 'R075', excl=_Excl([]))
        assert e.gate == 'HG1' and 'does not exist' in str(e), e
        e = _raises(HST.StackRefusal, HST.members_of, sb, '13h', '46MHz', 'R075', excl=_Excl([]))
        assert e.gate == 'HG1' and 'weights.csv does not exist' in str(e), e
    finally:
        HST._CFP = orig
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------- 5 Phase 1 file
def test_p1file_choice():
    d = _tmp('p1')
    try:
        for run in ('Run_20250102_010000', 'Run_20250105_010000', 'Run_20250109_010000'):
            dd = os.path.join(d, '18h', '2025-01-01', run, '46MHz', 'I', 'deep')
            os.makedirs(dd)
            for k in ('dirty', 'psf'):
                open(os.path.join(dd, f'46MHz-I-Deep-NoTaper-Robust-0.75-{k}-x.fits'), 'w').close()
            open(os.path.join(dd, '46MHz-I-Deep-NoTaper-Robust-0.75-dirty-x.pbcorr.fits'), 'w').close()
        r = {'date': '2025-01-01', 'sci_dir': '/p2/18h/2025-01-01/Science_20250106_000000'}
        got = HST.p1file(d, '18h', '46MHz', 'NoTaper-Robust-0.75', r, 'dirty')
        assert '/Run_20250105_010000/' in got and not got.endswith('.pbcorr.fits'), got
        r0 = dict(r, sci_dir='/p2/18h/2025-01-01/Science_20250101_000000')        # every Phase 1 run is newer: none
        e = _raises(HST.StackRefusal, HST.p1file, d, '18h', '46MHz', 'NoTaper-Robust-0.75', r0, 'dirty')
        assert e.gate == 'HG4', e
        e = _raises(HST.StackRefusal, HST.p1file, d, '18h', '46MHz', 'NoTaper-Robust-0.75', r, 'residual')
        assert e.gate == 'HG4' and '0 Phase 1 files' in str(e), e
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------- 7 screen lock and cache
def test_screen_lock_and_cache():
    import fcntl
    import subprocess
    import exopipe.phase2 as P2
    assert not issubclass(HW.ScreenLockRefusal, RuntimeError)       # the night-drop rule (except RuntimeError) never catches it
    d = _tmp('lock')
    # a stand-in for exopipe.phase2.cascade_dewarp (its PyBDSF import is not in CI): the pair finders only; the cache and lock paths
    # tested here derive nothing
    key = 'exopipe.phase2.cascade_dewarp'
    saved = (sys.modules.get(key), getattr(P2, 'cascade_dewarp', None))
    CD = types.ModuleType(key)
    old_wait = CFG.hybrid_stack.SCREEN_LOCK_WAIT_S
    holder = None
    try:
        raw, dew = os.path.join(d, 'raw.fits'), os.path.join(d, 'raw_dewarped.fits')
        for p in (raw, dew):
            fits.writeto(p, np.zeros((8, 8), np.float32))
        CD.find_deep_pb_image = CD.find_deep_pb_image_dewarped = lambda *a: None
        CD.find_deep_raw_image, CD.find_deep_raw_image_dewarped = (lambda *a: raw), (lambda *a: dew)
        CD.apply_warp = None
        sys.modules[key] = CD
        P2.cascade_dewarp = CD
        cache = os.path.join(d, 'cache')
        os.makedirs(cache)
        out = os.path.join(cache, '2025-01-01_13h_78MHz_warp_screen_derived.fits')
        sx = np.full((8, 8), 0.25, np.float32)
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(sx, name='SCREEN_X'), fits.ImageHDU(-sx, name='SCREEN_Y')]).writeto(out)
        rec = dict(pair_fingerprint=HW._pair_fp(sorted({raw, dew})))
        with open(out + '.json', 'w') as f:
            json.dump(rec, f)
        holder = subprocess.Popen([sys.executable, '-c', 'import fcntl, os, sys, time; fd = os.open(sys.argv[1], os.O_CREAT | '
                                   'os.O_WRONLY); fcntl.flock(fd, fcntl.LOCK_EX); print("held", flush=True); time.sleep(600)',
                                   out + '.lock'], stdout=subprocess.PIPE)
        assert holder.stdout.readline().strip() == b'held'
        # a cached screen of this exact pair is used while another process holds the lock
        a, b, r = HW.derive_warp_screen(d, '2025-01-01', '13h', '78MHz', cache)
        assert np.allclose(a, 0.25) and np.allclose(b, -0.25) and r == rec
        # the pair changed and a live process holds the lock: refused after the wait (never a RuntimeError)
        os.utime(dew, ns=(10 ** 18, 10 ** 18))
        CFG.hybrid_stack.SCREEN_LOCK_WAIT_S = 0
        e = _raises(HW.ScreenLockRefusal, HW.derive_warp_screen, d, '2025-01-01', '13h', '78MHz', cache)
        assert 'held the lock' in str(e), e
        # the holder is killed: the kernel releases its lock (a killed derivation leaves no lock behind)
        holder.kill()
        holder.wait()
        fd = os.open(out + '.lock', os.O_WRONLY)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.close(fd)
    finally:
        if holder is not None and holder.poll() is None:
            holder.kill()
            holder.wait()
        if saved[0] is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = saved[0]
        if saved[1] is None:
            delattr(P2, 'cascade_dewarp')
        else:
            P2.cascade_dewarp = saved[1]
        CFG.hybrid_stack.SCREEN_LOCK_WAIT_S = old_wait
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------- 6 unit + content tests
def _fake_stack(fail, dropped=None, partial=(), lock=(), nan=()):
    """A stack_variant stand-in: refuses the variants in `fail` (gate HG5; those in `partial` after writing one stack and a
    temporary), raises the screen-lock refusal for those in `lock`, else writes valid files beside a fake master; `dropped`:
    {variant: [dates left out of its stacks]}; `nan`: variants whose dirty stack is all NaN (fails its content test)."""
    def fn(lst, band, v, workdir, prep, screens, read=None):
        assert os.path.isdir(prep) and os.listdir(prep) == []
        if v in lock:
            raise HW.ScreenLockRefusal(f'2025-01-02 {lst} {band}: another process has held the lock - REFUSING')
        if v in partial:
            open(f'{workdir}/{v}_dirty_stack.fits', 'w').close()
            open(f'{workdir}/{v}_stack.json.part', 'w').close()
        if v in fail:
            raise HST.StackRefusal(f'{lst} {band} {v}: test_stack_rebuild FAILED - REFUSING', 'HG5')
        mfile = HST.cell_paths(CFG.hybrid_stack.COADD_SB, lst, band, v)[2]
        mh = fits.getheader(mfile)
        img = (_gauss(64, [(32, 32, 10.0)]) + 0.5).astype(np.float32)
        nights = [dict(date=f'2025-01-0{i}', weight=1.0, dc=1.0) for i in (1, 2, 3)]
        rec = dict(lst=lst, band=band, weighting=v, nights=nights, dirty_nights_dropped=list((dropped or {}).get(v, [])),
                   nights_dropped_no_screen={},
                   **({'nights_stacked': [n['date'] for n in nights if n['date'] not in (dropped or {}).get(v, [])]}
                      if (dropped or {}).get(v) else {}),
                   test_stack_rebuild=dict(verdict='pass (exact)', max_abs_diff=0.0, rms_diff=0.0, master_peak=1.0, master_rms=1.0))
        if v in HC.TAPER_OF.values():
            HC.write_atomic(f'{workdir}/{v}_residual_stack.fits', img, HC.diag_header(mh, 'residual'))
        else:
            for n in ('dirty_stack', 'dirty_halfA', 'dirty_halfB'):
                HC.write_atomic(f'{workdir}/{v}_{n}.fits', img * np.nan if v in nan and n == 'dirty_stack' else img,
                                HC.diag_header(mh, n))
            HC.write_atomic(f'{workdir}/{v}_psf_stack.fits', img / img.max(), HC.diag_header(mh, 'psf'))
            os.makedirs(f'{workdir}/inputs', exist_ok=True)
            rec['inputs'] = {}
            for src in HST.input_copies(CFG.hybrid_stack.COADD_SB, lst, band, v):
                shutil.copy2(src, f'{workdir}/inputs/{os.path.basename(src)}')
                rec['inputs'][os.path.basename(src)] = dict(installed=src, md5=SC.md5(src))
        with open(f'{workdir}/{v}_stack.json', 'w') as f:
            json.dump(rec, f)
        return rec
    return fn


def _fake_cell(sb):
    cf = os.path.join(sb, '18h_46MHz', 'coadd_final')
    os.makedirs(cf)
    h = fits.Header({'CTYPE1': 'RA---SIN', 'CTYPE2': 'DEC--SIN', 'CRVAL1': 270.0, 'CRVAL2': 37.0, 'CRPIX1': 33.0, 'CRPIX2': 33.0,
                     'CDELT1': -0.5, 'CDELT2': 0.5})
    h['PRODNAME'] = 'x'
    for v in ('NoTaper_R075', 'NoTaper_R0', 'Taper_R075', 'Taper_R0'):
        fits.writeto(os.path.join(cf, f'I_deep_{v}_46MHz.fits'), np.zeros((64, 64), np.float32), h, overwrite=True)
    for p in ('PB_deep_R075_46MHz.fits', 'PB_deep_46MHz.fits'):
        fits.writeto(os.path.join(cf, p), np.ones((64, 64), np.float32), overwrite=True)


def _run_unit(out, work, w):
    run = record.Run('hybrid', ['test'], out, CFG)
    with run.unit(f'stack.18h_46MHz_{w}', stage='stack', lst='18h', scope='46MHz', weighting=w) as u:
        HS.stack_unit(u, '18h', '46MHz', w, out, work)
    return run.units[-1]


def test_stack_unit_items_and_content_tests():
    d = _tmp('unit')
    old = (CFG.hybrid_stack.COADD_SB, HST.stack_variant)
    try:
        sb = os.path.join(d, 'coadd_sb')
        _fake_cell(sb)
        CFG.hybrid_stack.COADD_SB = sb
        out, work = os.path.join(d, 'out'), os.path.join(d, 'work')
        os.makedirs(out)
        # both variants written: done, every content test passed (four classes)
        HST.stack_variant = _fake_stack(fail=())
        u = _run_unit(out, work, 'R075')
        assert u.status == 'done', (u.status, u.reason, [t for t in u.tests if not t['passed']])
        classes = {o['class'] for o in u.outputs}
        assert classes == {'hybrid_stack_image', 'hybrid_psf_stack', 'hybrid_stack_record', 'hybrid_stack_input'}, classes
        assert len(u.outputs) == 5 + 3 + 2, [o['path'] for o in u.outputs]       # R075: 5 files + 3 distinct inputs/; T075: 2
        assert all(t['passed'] for t in u.tests), [t for t in u.tests if not t['passed']]
        assert u.admitted == ['2025-01-01', '2025-01-02', '2025-01-03']
        # the NoTaper variant refused: its 8 planned files are refused items (gate HG5); the Taper one is done
        HST.stack_variant = _fake_stack(fail=('R075',))
        u = _run_unit(out, work, 'R075')
        assert u.status == 'done' and '8 planned item(s) refused' in u.reason, (u.status, u.reason)    # 5 files + 3 inputs/
        assert {r['gate'] for r in u.refused_items} == {'HG5'}
        assert not os.path.exists(os.path.join(out, '18h_46MHz', 'R075_dirty_stack.fits'))   # its earlier output removed first
        # a refusal after the variant wrote a stack and a temporary: both removed (no stale file beside a refusal)
        HST.stack_variant = _fake_stack(fail=('R075',), partial=('R075',))
        u = _run_unit(out, work, 'R075')
        left = sorted(f for f in os.listdir(os.path.join(out, '18h_46MHz')) if f.startswith('R075_'))
        assert left == [], left
        # a screen lock held too long refuses the variant (HG4); it never drops a night
        HST.stack_variant = _fake_stack(fail=(), lock=('T075',))
        u = _run_unit(out, work, 'R075')
        assert u.status == 'done' and {r['gate'] for r in u.refused_items} == {'HG4'} and len(u.refused_items) == 2, u.refused_items
        # admitted nights: those the written stacks hold (a night dropped from every stack is not admitted)
        HST.stack_variant = _fake_stack(fail=(), dropped={'R075': ['2025-01-02'], 'T075': ['2025-01-02']})
        u = _run_unit(out, work, 'R075')
        assert u.status == 'done' and u.admitted == ['2025-01-01', '2025-01-03'], (u.status, u.admitted)
        HST.stack_variant = _fake_stack(fail=(), dropped={'R075': ['2025-01-02']})
        assert _run_unit(out, work, 'R075').admitted == ['2025-01-01', '2025-01-02', '2025-01-03']     # the Taper stack holds it
        # both refused: the unit is refused
        HST.stack_variant = _fake_stack(fail=('R0', 'T0'))
        u = _run_unit(out, work, 'R0')
        assert u.status == 'refused' and 'every planned item refused' in u.reason, (u.status, u.reason)
        # a failed content test refuses the variant and removes its files (a stack record on disk is a passed stack)
        HST.stack_variant = _fake_stack(fail=(), nan=('R075',))
        u = _run_unit(out, work, 'R075')
        assert u.status == 'done' and {r['gate'] for r in u.refused_items} == {'content'} and len(u.refused_items) == 8, u.refused_items
        left = sorted(f for f in os.listdir(os.path.join(out, '18h_46MHz')) if f.startswith(('R075_', 'T075_')))
        assert left == ['T075_residual_stack.fits', 'T075_stack.json'], left
        assert u.admitted == ['2025-01-01', '2025-01-02', '2025-01-03']                               # the Taper stack's nights
        # content tests fail on bad files
        from exopipe.phase3.core import validate
        wdir = os.path.join(out, '18h_46MHz')
        bad = os.path.join(d, 'bad.fits')
        fits.writeto(bad, np.full((64, 64), np.nan, np.float32), fits.Header({'PRODNAME': 'x'}), overwrite=True)
        m = HST.cell_paths(sb, '18h', '46MHz', 'R075')[2]
        res = {r['name']: r['passed'] for r in validate.run('hybrid_stack_image', bad, master=m)}
        assert res == {'hybrid_stack_image.master_grid': False, 'hybrid_stack_image.finite_nonzero': False,
                       'hybrid_stack_image.intermediate_not_product': False}, res
        psf2 = os.path.join(d, 'psf2.fits')
        fits.writeto(psf2, np.full((64, 64), 0.5, np.float32), fits.getheader(os.path.join(wdir, 'T075_residual_stack.fits')))
        assert not {r['name']: r['passed'] for r in validate.run('hybrid_psf_stack', psf2, master=m)}['hybrid_psf_stack.peak_one']
        recp = os.path.join(d, 'rec.json')
        with open(os.path.join(wdir, 'T075_stack.json')) as f:
            rec = json.load(f)
        rec['test_stack_rebuild']['verdict'] = 'pass (something else)'
        rec['nights_dropped_no_screen'] = {'2025-01-01': 'x', '2025-01-02': 'y'}
        with open(recp, 'w') as f:
            json.dump(rec, f)
        r = validate.run('hybrid_stack_record', recp, lst='18h', band='46MHz', weighting='T075')[0]
        assert not r['passed'] and 'rebuild verdict' in r['detail'] and 'without a screen' in r['detail'], r
        rec['nights_dropped_no_screen'], rec['nights_raw_admitted'] = {}, ['2025-05-03']   # unaligned nights: ruled cells only
        rec['test_stack_rebuild']['verdict'] = 'pass (exact)'
        for cell, ok in ((('18h', '46MHz'), False), (('13h', '46MHz'), True)):
            rec['lst'], rec['band'] = cell
            with open(recp, 'w') as f:
                json.dump(rec, f)
            r = validate.run('hybrid_stack_record', recp)[0]
            assert ('outside hybrid_stack.RAW_NIGHT_ADMIT' not in r['detail']) is ok, (cell, r)
        inp = os.path.join(wdir, 'inputs', 'PB_deep_46MHz.fits')
        r = validate.run('hybrid_stack_input', inp, md5='0' * 32, installed='x')[0]
        assert not r['passed'], r
    finally:
        CFG.hybrid_stack.COADD_SB, HST.stack_variant = old
        shutil.rmtree(d, ignore_errors=True)



def test_copy_inputs_from_archive():
    """copy_inputs: the inputs/ copies come from the archived masters and PB maps (their headers: the repair cards, no SH card), each
    refused unless its pixels equal the co-add build file's; a current copy is kept; a missing or ambiguous archived master refuses."""
    d = _tmp('cpin')
    try:
        sb = os.path.join(d, 'coadd_sb')
        _fake_cell(sb)
        cf = os.path.join(sb, '18h_46MHz', 'coadd_final')
        am = os.path.join(d, 'masters')
        a = os.path.join(am, '18h', '46MHz')
        os.makedirs(a)
        for rob in ('Robust-0.75', 'Robust+0.0'):
            for pb in ('', 'pbcorr_'):
                n = f'46MHz_I_deep_NoTaper_{rob}_shflux_mdewarp_{pb}dewarped_aligned_coadd_LST18h_20250405-20251014_N015.fits'
                fits.writeto(os.path.join(a, n), np.zeros((64, 64), np.float32) + (1 if pb else 0),
                             fits.Header({'PRODNAME': n, 'P1RUN': 'r1', 'CALBP': 'bp'}))
        for p in ('PB_deep_R075_46MHz.fits', 'PB_deep_46MHz.fits'):
            fits.writeto(os.path.join(a, p), np.ones((64, 64), np.float32), fits.Header({'P1RUN': 'r1'}))
        ind, read = os.path.join(d, 'w', 'inputs'), set()
        got = HST.copy_inputs(ind, sb, '18h', '46MHz', 'R0', read, archive_masters=am)
        assert sorted(got) == ['I_deep_NoTaper_R075_46MHz.fits', 'I_deep_NoTaper_R0_46MHz.fits', 'PB_deep_46MHz.fits',
                               'PB_deep_R075_46MHz.fits'], got
        for n, rec in got.items():
            assert rec['installed'].startswith(a) and fits.getheader(os.path.join(ind, n))['P1RUN'] == 'r1', (n, rec)
            assert rec['md5'] == SC.md5(rec['installed'])
        assert got['I_deep_NoTaper_R0_46MHz.fits']['installed'].endswith('_Robust+0.0_shflux_mdewarp_dewarped_aligned_coadd_'
                                                                          'LST18h_20250405-20251014_N015.fits')
        assert len(read) == 8, read
        ino = os.stat(os.path.join(ind, 'PB_deep_46MHz.fits')).st_ino
        HST.copy_inputs(ind, sb, '18h', '46MHz', 'R075', set(), archive_masters=am)
        assert os.stat(os.path.join(ind, 'PB_deep_46MHz.fits')).st_ino == ino                          # current copy kept
        fits.writeto(os.path.join(cf, 'PB_deep_46MHz.fits'), np.full((64, 64), 0.9, np.float32), overwrite=True)
        for want, prep in (('pixels differ', None),
                           ('archived masters', lambda: os.remove(os.path.join(a, '46MHz_I_deep_NoTaper_Robust-0.75_shflux_mdewarp_'
                                                                                 'dewarped_aligned_coadd_LST18h_20250405-20251014_N015.fits')))):
            if prep:
                fits.writeto(os.path.join(cf, 'PB_deep_46MHz.fits'), np.ones((64, 64), np.float32), overwrite=True)
                prep()
            try:
                HST.copy_inputs(os.path.join(d, 'w2', 'inputs'), sb, '18h', '46MHz', 'R075', set(), archive_masters=am)
                raise AssertionError(f'no refusal ({want})')
            except HST.StackRefusal as e:
                assert want in str(e), (want, e)
        assert not [f for f in os.listdir(ind) if f.endswith('.part')]
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted((k, v) for k, v in dict(globals()).items() if k.startswith('test_') and callable(v)):
        try:
            fn()
            print('PASS', name, flush=True)
        except Exception as e:                                # pragma: no cover
            fails += 1
            import traceback
            traceback.print_exc()
            print('FAIL', name, e, flush=True)
    print(f'{fails} failed')
    sys.exit(1 if fails else 0)
