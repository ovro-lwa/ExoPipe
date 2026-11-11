#!/usr/bin/env python3
"""The ported 10-s / 1-minute snapshot engine (exopipe/phase3/lib/snapshot_engine/; Gregg 2026-11-11: the faster snapshot code
lands in the refactor): its speed-ups change no product (REVIEW.md item 8).

The acceptance of every speed-up is a PARITY RUN ON REAL HOURS against the pinned production files and the fleet's shipped
products (exopipe_ops/refactor_20261012/plan/STAGE_MAP_20261111.md section 8). These tests are the standing, data-free guard of
the same property on synthetic inputs, so CI catches a change that breaks it:

  1. row medians: _nanmedian_rows equals np.nanmedian(a, axis=1), value for value and dtype for dtype, on NaN-free rows, sparse
     NaNs, all-NaN rows (the first row included: numpy's result dtype depends on it), infinities, signed zeros, ties, odd and
     short rows, float32 and float64;
  2. detect(): every DETECT_THREADS value gives the candidate rows, the per-image noise rows, the admitted flags and the flood
     record of the serial loop, on a cube with sources, a blank frame, a frame without a noise estimate, noise outliers and a
     flooded frame; Stokes I and V;
  3. highpass_blocks(): identical bytes for every DETECT_THREADS value;
  4. admission_table(): identical rows and flags for every DETECT_THREADS value (the night rule is stubbed: no ephemeris here);
  5. the cutout target lists: a missing DEFAULT list refuses (production skipped it: a copy of the engine without the lists
     wrote the planets only), in _targets() and in check_default_lists(), which the searches call before any work;
  6. the wideband accumulation: the per-instant order (P3_SNAP_GROUP_THREADS > 0) gives the cubes, weight sums, member counts
     and member rows of production's loop, bit for bit, for several chunk sizes and thread counts, with screens, NaN regions,
     refused and unmatched frames, a blank frame and a frame on the wrong grid; with and without a Full group;
  7. write_cutouts(): the same cutout cubes and tables for every DETECT_THREADS value (fixed targets on and off the frame edge);
  8. the frame movies: a subband's binned cube is the same for every P3_SNAP_MOVIE_THREADS value;
  9. cluster only (the clv15 repository present): every ported file's pinned blob is the blob of the pin commit, and the files
     production has changed since the pin are listed in ONE failure that names what to port. Without clv15 this prints
     UNTESTED-PRODUCTION and does not fail.
Run: python tests/test_snapshot_engine.py   (or under pytest). On the cluster run it ALSO under the engine's own interpreter
(site PYTHON['py36'], numpy 1.19.2): numpy's result dtype for an all-NaN first row differs between numpy versions, and a row
median with the wrong rule is caught only where the engine actually runs (mutation-checked 2026-11-11: results applied out
of order, a refused frame left admitted, the wrong dtype rule - each fails here).
"""
import importlib
import os
import subprocess
import sys
import tempfile
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'snapshot_engine')
sys.path.insert(0, REPO)

_IMPORT_ERROR = None
try:
    import numpy as np
    sys.path.insert(0, ENGINE)                 # the engine files are scripts that import each other by name, as in clv15
    import phase3_snapshot_search as SS
    import phase3_snapshot_cutouts as SC
    import phase3_snapshot_group_search as GS
    from exopipe.phase3.lib import snapshot_engine as PIN
except Exception as e:                                    # pragma: no cover
    _IMPORT_ERROR = f'{e.__class__.__name__}: {e}'


def _nanmedian_rows():
    """The function from the light-curve module, without importing the module (it imports two unported clv15 modules)."""
    src = open(os.path.join(ENGINE, 'phase3_snapshot_lightcurves.py')).read()
    start, end = src.index('def _nanmedian_rows_fast(a):'), src.index('\ndef git_hash')
    ns = {'np': np}
    exec(src[start:end], ns)
    assert ns['_fast_median_ok']() is True, f'the fast row median is switched off under numpy {np.__version__} (its probe differs)'
    return ns['_nanmedian_rows']


def _same(x, y):
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def test_row_medians_equal_numpy():
    f = _nanmedian_rows()
    rng = np.random.RandomState(7)
    k = 1656                                             # the engine's stamp: 41 x 41 minus the central 5 x 5
    cases = []
    for dt in (np.float32, np.float64):
        a = rng.standard_normal((400, k)).astype(dt)
        cases.append(('no NaN', a))
        b = a.copy(); b[rng.rand(*b.shape) < 0.001] = np.nan
        cases.append(('sparse NaN', b))
        c = a.copy(); c[5] = np.nan; c[17, :800] = np.nan; c[30, 1:] = np.nan; c[31, :-1] = np.nan
        cases.append(('all-NaN row inside; odd and even counts', c))
        d = a.copy(); d[0] = np.nan; d[9, ::2] = np.nan
        cases.append(('first row all NaN', d))
        e = a.copy(); e[:, :] = np.nan
        cases.append(('everything NaN', e))
        g = a.copy(); g[3, 10] = np.inf; g[4, 11] = -np.inf; g[6, :3] = -0.0; g[7] = 0.0; g[8, :828] = -0.0; g[8, 828:] = 0.0
        cases.append(('infinities and signed zeros', g))
        cases.append(('ties', np.round(a * 3).astype(dt)))
        cases.append(('odd row length', a[:, :1655].copy()))
        m = a[:, :599].copy(); m[2, 5] = np.nan
        cases.append(('short rows (numpy small path)', m))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            med = np.nanmedian(d, axis=1)
        cases.append(('deviations from a first-row-NaN median', np.abs(d - med[:, None])))
    bad = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name, a in cases:
            if not _same(np.nanmedian(a, axis=1), f(a)):
                bad.append(f'{name} ({a.dtype})')
    assert not bad, f'_nanmedian_rows differs from np.nanmedian on: {bad}'


def _cube(stokes):
    """A small cube with what detect() branches on: point sources, a blank (non-admitted) frame, an all-NaN frame (no noise
    estimate), two noisy frames (noise outliers), one frame full of sources (flood) and NaN borders."""
    rng = np.random.RandomState(11 if stokes == 'I' else 12)
    nt, ny, nx = 48, 160, 160
    cube = rng.standard_normal((nt, ny, nx)).astype(np.float32)
    cube[:, :6, :] = np.nan; cube[:, :, -6:] = np.nan
    for t, y, x, amp in ((3, 40, 50, 9.0), (3, 90, 120, -8.0), (10, 70, 70, 12.0), (11, 70, 71, 11.0), (30, 100, 30, 7.5)):
        cube[t, y - 1:y + 2, x - 1:x + 2] += np.float32(amp)
    cube[20] = np.nan                                     # no noise estimate
    cube[25] *= np.float32(6.0); cube[26] *= np.float32(5.0)      # noise outliers
    ys, xs = np.mgrid[20:140:8, 20:140:8]
    cube[40, ys, xs] += np.float32(15.0)                  # flood
    admitted = np.ones(nt, dtype=bool); admitted[7] = False
    meta = [{'ts': '20241219_06%02d%02d' % (t // 6, (t % 6) * 10), 'date_obs': '2024-12-19T06:%02d:%02d.0' % (t // 6, (t % 6) * 10)}
            for t in range(nt)]
    stats = admitted.copy(); stats[20] = False             # the engine keeps blank frames out of the hour median (stats_ok)
    model = SS.chunked_median(cube[stats])
    mask = np.ones((ny, nx), dtype=bool); mask[:10] = False
    masks = [mask, mask.copy()]
    bin_of = np.array([0] * (nt // 2) + [1] * (nt - nt // 2))
    return cube, model, admitted, meta, masks, bin_of


def _detect(threads, stokes):
    SS.DETECT_THREADS = threads
    try:
        cube, model, admitted, meta, masks, bin_of = _cube(stokes)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cands, rows, adm, info = SS.detect(cube, model, admitted, meta, 5.0, stokes, masks, bin_of, flood_min_abs=20)
        return repr(cands), repr([list(r.items()) for r in rows]), adm.tobytes(), repr(sorted(info.items()))
    finally:
        SS.DETECT_THREADS = 1


def test_detect_is_the_serial_loop_for_every_thread_count():
    for stokes in ('I', 'V'):
        ref = _detect(1, stokes)
        assert 'no-noise-estimate' in ref[1] and 'noise-outlier' in ref[1] and "'flood'" in ref[1] and "'admission'" in ref[1], \
            f'the synthetic cube no longer exercises every refusal ({stokes})'
        assert len(ref[0]) > 100, 'no candidates in the synthetic cube'
        for n in (2, 3, 8):
            got = _detect(n, stokes)
            for what, a, b in zip(('candidates', 'noise rows', 'admitted', 'flood record'), ref, got):
                assert a == b, f'detect() with DETECT_THREADS={n} differs from the serial loop in {what} (Stokes {stokes})'


def test_highpass_is_identical_for_every_thread_count():
    rng = np.random.RandomState(3)
    blocks = rng.standard_normal((9, 200, 200)).astype(np.float32)
    blocks[:, :15, :] = np.nan; blocks[4] = np.nan; blocks[6, 50:60, 50:60] = np.nan
    SS.DETECT_THREADS = 1
    ref = SS.highpass_blocks(blocks)
    try:
        for n in (2, 4):
            SS.DETECT_THREADS = n
            assert ref.tobytes() == SS.highpass_blocks(blocks).tobytes(), f'highpass_blocks differs with DETECT_THREADS={n}'
    finally:
        SS.DETECT_THREADS = 1


def test_admission_is_identical_for_every_thread_count():
    rng = np.random.RandomState(5)
    v = rng.standard_normal((30, 120, 120)).astype(np.float32)
    v[4] *= np.float32(8.0); v[9] = np.nan
    meta = [{'ts': '20241219_0600%02d' % t, 'date_obs': '', 'bmaj': 0.4 + (0.5 if t == 12 else 0.0), 'bmin': 0.3} for t in range(30)]
    keep = SS.night_verdict, SS.sun_state
    SS.night_verdict = lambda ts, lst, date, sb: 'sun-up' if ts.endswith('03') else ''      # no ephemeris in a data-free test
    SS.sun_state = lambda ts: (-30.0, 'night', 0.0)
    try:
        out = []
        for n in (1, 2, 6):
            SS.DETECT_THREADS = n
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                rows, adm, vrms = SS.admission_table(v, None, meta, ctx=('04h', '2024-12-19', '18MHz'))
            out.append((repr([list(r.items()) for r in rows]), adm.tobytes(), vrms.tobytes()))
        assert out[0] == out[1] == out[2], 'admission_table differs between DETECT_THREADS values'
        assert "('beam_refused', 1)" in out[0][0] and "'sun-up'" in out[0][0] and "('rms_outlier', 1)" in out[0][0]
    finally:
        SS.night_verdict, SS.sun_state = keep
        SS.DETECT_THREADS = 1


def test_a_missing_default_target_list_refuses():
    keep = SC.SAMPLES_DIR, SC._TARGETS, os.environ.pop('P3_SNAP_CUTOUT_SAMPLES', None)
    with tempfile.TemporaryDirectory() as d:
        try:
            SC.SAMPLES_DIR, SC._TARGETS = d, None
            try:
                SC._targets()
            except RuntimeError as e:
                assert 'REFUSED' in str(e) and 'P3_SNAP_SAMPLES_DIR' in str(e)
            else:
                raise AssertionError('a missing default target list was skipped instead of refused')
            for fn in ('Hot_Warm_Jupiters.csv', '10pc.csv'):          # both present: read, no refusal
                with open(os.path.join(d, fn), 'w') as f:
                    f.write('name,ra,dec\nA,10.0,20.0\n')
            SC._TARGETS = None
            assert [t[1] for t in SC._targets()] == ['A', 'A']
        finally:
            SC.SAMPLES_DIR, SC._TARGETS = keep[0], keep[1]
            if keep[2] is not None:
                os.environ['P3_SNAP_CUTOUT_SAMPLES'] = keep[2]


def _accum_world(full):
    """Synthetic member frames for accumulate(): two groups on different grids (or one grid and a Full group)."""
    rng = np.random.RandomState(21)
    nt = 30
    red, green = list(GS.GROUPS['Red']), list(GS.GROUPS['Green'])[:2]
    # frames of 80 px or more: central_rms needs 1,000 finite pixels in the central quarter (smaller frames are all dropped)
    shape_of = {sb: (80, 80) for sb in red}
    shape_of.update({sb: ((80, 80) if full else (96, 96)) for sb in green})
    frames, sb_frames, sb_match, sb_adm, sb_screens, sb_dsec = {}, {}, {}, {}, {}, {}
    for k, sb in enumerate(red + green):
        fl = []
        for j in range(nt):
            key = '%s:%d' % (sb, j)
            d = rng.standard_normal(shape_of[sb]).astype(np.float32) * np.float32(1.0 + 0.3 * k)
            d[:3, :] = np.nan
            if (k + j) % 11 == 0:
                d[38:42, 38:42] = np.nan
            frames[key] = d
            fl.append(('20241219_06%02d%02d' % (j // 6, (j % 6) * 10 + k % 3), key))
        frames['%s:%d' % (sb, 4)] = np.full(shape_of[sb], np.nan, np.float32) if k == 1 else frames['%s:%d' % (sb, 4)]   # blank
        if k == 2:
            frames['%s:%d' % (sb, 9)] = rng.standard_normal((12, 12)).astype(np.float32)                                # wrong grid
        sb_frames[sb] = fl
        m = np.arange(nt)
        m[(np.arange(nt) + k) % 13 == 0] = -1                                    # unmatched instants
        sb_match[sb] = m
        sb_adm[sb] = {} if k == 3 else {ts: (j + k) % 7 != 0 for j, (ts, _) in enumerate(fl)}       # {} = every frame eligible
        sx = (rng.standard_normal(shape_of[sb]) * 0.3).astype(np.float32) if k % 2 else np.zeros(shape_of[sb], np.float32)
        sb_screens[sb] = (sx, np.zeros(shape_of[sb], np.float32))
        sb_dsec[sb] = (np.arange(nt, dtype=float), {key: 'dob-' + key for _, key in fl})
    group_of = {sb: g for g, sbs in GS.GROUPS.items() for sb in sbs}
    grid_of = {'Red': (80, 80), 'Green': (80, 80) if full else (96, 96)}
    if full:
        grid_of['Full'] = (80, 80)
    ref_list = sb_frames[red[0]]
    return dict(nt=nt, frames=frames, sb_frames=sb_frames, sb_match=sb_match, sb_adm=sb_adm, sb_screens=sb_screens,
                grid_of=grid_of, group_of=group_of, ref_list=ref_list, sb_dsec=sb_dsec, ref_dob=sb_dsec[red[0]][1])


def _accumulate(w, full, per_instant, threads, chunk):
    cubes = {g: np.zeros((w['nt'],) + w['grid_of'][g], dtype=np.float32) for g in w['grid_of']}
    wsums = {g: np.zeros(w['nt'], dtype=np.float64) for g in w['grid_of']}
    nsubs = {g: np.zeros(w['nt'], dtype=int) for g in w['grid_of']}
    keep = GS.read_frame, GS.GROUP_THREADS, GS.log.level
    GS.read_frame = lambda key: (w['frames'][key], None)
    GS.GROUP_THREADS = threads
    GS.log.setLevel(100)                                         # the wrong-grid warning is expected
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rows = GS.accumulate(w['nt'], w['sb_frames'], w['sb_match'], w['sb_adm'], w['sb_screens'], w['grid_of'], w['group_of'],
                                 full, w['ref_list'], w['sb_dsec'], w['ref_dob'], cubes, wsums, nsubs, 8, chunk, 0.0, per_instant)
    finally:
        GS.read_frame, GS.GROUP_THREADS = keep[0], keep[1]
        GS.log.setLevel(keep[2])
    return ([(g, cubes[g].tobytes(), wsums[g].tobytes(), nsubs[g].tobytes()) for g in sorted(cubes)], rows)


def test_wideband_accumulation_per_instant_equals_production_loop():
    for full in (False, True):
        w = _accum_world(full)
        ref = _accumulate(w, full, False, 0, 24)
        counts = np.frombuffer(ref[0][-1][3], dtype=int)             # Red's member count per instant
        assert len(ref[1]) > 100 and counts.min() < counts.max() <= 5, 'the synthetic frames no longer drop members at some instants'
        for chunk in (7, 24, 10 ** 6):
            for threads in (1, 3, 8):
                got = _accumulate(w, full, True, threads, chunk)
                prod = ref if chunk == 24 else _accumulate(w, full, False, 0, chunk)
                assert got[0] == prod[0], f'cubes / weight sums / member counts differ (Full {full}, chunk {chunk}, threads {threads})'
                assert got[1] == prod[1], f'member rows differ or are in another order (Full {full}, chunk {chunk}, threads {threads})'


def _cutout_tree(threads, root, lists):
    from astropy.io import fits
    rng = np.random.RandomState(31)
    nt, n = 9, 256
    hdr = fits.Header()
    for k, v in (('NAXIS', 2), ('NAXIS1', n), ('NAXIS2', n), ('CTYPE1', 'RA---SIN'), ('CTYPE2', 'DEC--SIN'), ('CRVAL1', 60.0),
                 ('CRVAL2', 37.24), ('CRPIX1', n / 2 + 0.5), ('CRPIX2', n / 2 + 0.5), ('CDELT1', -0.25), ('CDELT2', 0.25),
                 ('BMAJ', 0.45), ('BMIN', 0.33), ('BPA', 10.0)):
        hdr[k] = v
    cube = rng.standard_normal((nt, n, n)).astype(np.float32); cube[:, :20, :] = np.nan
    model = np.median(cube, axis=0).astype(np.float32)
    meta = [{'ts': '20241219_0600%02d' % (t * 6), 'date_obs': '2024-12-19T06:00:%02d.0' % (t * 6)} for t in range(nt)]
    admitted = np.ones(nt, dtype=bool); admitted[3] = False
    screen = np.zeros((n, n), np.float32)
    keep = os.environ.get('P3_SNAP_CUTOUT_SAMPLES'), SC._TARGETS, SS.DETECT_THREADS, SS.lst_deg_at
    os.environ['P3_SNAP_CUTOUT_SAMPLES'] = 'T=' + lists
    SC._TARGETS, SS.DETECT_THREADS = None, threads
    SS.lst_deg_at = lambda sec: 60.0                               # the field is at the zenith at the cell's mid-time
    try:
        nw = SC.write_cutouts(cube, model, meta, admitted, hdr, screen, screen, '04h', '2024-12-19', '18MHz', 'I',
                              noise_rows=[{'ts': m['ts'], 'global_rms_jyb': '1.0'} for m in meta], root=root)
    finally:
        SC._TARGETS, SS.DETECT_THREADS, SS.lst_deg_at = keep[1], keep[2], keep[3]
        if keep[0] is None:
            os.environ.pop('P3_SNAP_CUTOUT_SAMPLES', None)
        else:
            os.environ['P3_SNAP_CUTOUT_SAMPLES'] = keep[0]
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            with fits.open(os.path.join(d, f), memmap=False) as h:
                cards = [repr(c) for c in h[0].header.cards if c.keyword != 'DATE-CRE']
                out[os.path.relpath(os.path.join(d, f), root)] = (h[0].data.tobytes(), cards, h[1].data.tobytes(), repr(h[1].columns))
    return nw, out


def test_cutouts_are_identical_for_every_thread_count():
    with tempfile.TemporaryDirectory() as d:
        lists = os.path.join(d, 'targets.csv')
        with open(lists, 'w') as f:
            f.write('name,ra,dec\n')
            for i in range(11):                                   # inside the frame, near its edges, and far outside it
                f.write('obj%d,%.3f,%.3f\n' % (i, 60.0 + (i - 5) * 6.5, 37.24 + ((i * 7) % 11 - 5) * 5.0))
            f.write('far,200.0,-40.0\n')
        ref = _cutout_tree(1, os.path.join(d, 'r1'), lists)
        assert ref[0] >= 5 and len(ref[1]) == ref[0], f'the synthetic targets no longer give cutouts ({ref[0]})'
        for n in (2, 4, 16):
            got = _cutout_tree(n, os.path.join(d, 'r%d' % n), lists)
            assert got[0] == ref[0] and sorted(got[1]) == sorted(ref[1]), f'other cutout files with DETECT_THREADS={n}'
            for k in ref[1]:
                assert got[1][k] == ref[1][k], f'cutout {k} differs with DETECT_THREADS={n}'


def test_check_default_lists_refuses_before_any_work():
    keep = SC.SAMPLES_DIR, os.environ.pop('P3_SNAP_CUTOUT_SAMPLES', None)
    with tempfile.TemporaryDirectory() as d:
        try:
            SC.SAMPLES_DIR = d
            with open(os.path.join(d, 'Hot_Warm_Jupiters.csv'), 'w') as f:
                f.write('name,ra,dec\nA,10.0,20.0\n')
            try:
                SC.check_default_lists()
            except RuntimeError as e:
                assert 'REFUSED' in str(e) and '10pc.csv' in str(e)
            else:
                raise AssertionError('one default list missing: check_default_lists did not refuse')
            with open(os.path.join(d, '10pc.csv'), 'w') as f:
                f.write('name,ra,dec\nB,11.0,21.0\n')
            SC.check_default_lists()                               # both present: no refusal
            os.environ['P3_SNAP_CUTOUT_SAMPLES'] = 'X=/nonexistent/list.csv'
            SC.check_default_lists()                               # an ad-hoc set is not checked here, as before
        finally:
            SC.SAMPLES_DIR = keep[0]
            os.environ.pop('P3_SNAP_CUTOUT_SAMPLES', None)
            if keep[1] is not None:
                os.environ['P3_SNAP_CUTOUT_SAMPLES'] = keep[1]


def test_movie_frames_binned_identically_for_every_thread_count():
    from astropy.io import fits
    try:
        import phase3_snapshot_frame_movie as FM
    except Exception as e:                                         # matplotlib missing: reported, not failed
        print(f'UNTESTED-IMPORT: phase3_snapshot_frame_movie ({e.__class__.__name__}: {e})')
        return
    rng = np.random.RandomState(41)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, '18MHz'))
        for j in range(14):
            h = fits.Header(); h['DATE-OBS'] = '2024-12-19T06:%02d:%02d.000' % (j // 6, (j % 6) * 10)
            x = rng.standard_normal((64, 64)).astype(np.float32); x[:5] = np.nan
            fits.writeto(os.path.join(d, '18MHz', '18MHz-Clean-Snapshot-20241219_06%02d%02d-image_flowed.fits' % (j // 6, (j % 6) * 10)), x, h)
        keep = FM.a, FM.MOVIE_THREADS
        try:
            FM.a = FM.parse_args(['04h_2024-12-19', os.path.join(d, 'out'), '--frames-root', d, '--size', '32'])
            out = []
            for n in (1, 2, 5):
                FM.MOVIE_THREADS = n
                ts, cube = FM.load_staged('18MHz')
                out.append((ts, cube.shape, cube.tobytes()))
            assert out[0] == out[1] == out[2], 'load_staged differs between P3_SNAP_MOVIE_THREADS values'
            assert out[0][1] == (14, 32, 32)
        finally:
            FM.a, FM.MOVIE_THREADS = keep


def test_production_pin():
    """Cluster only. The pin is honest (each blob is the pin commit's) and production's later changes to a ported file are
    reported in one failure that names them: moving the pin is a port round."""
    repo = PIN.CLV15_REPO
    if not os.path.isdir(os.path.join(repo, '.git')):
        print(f'UNTESTED-PRODUCTION: {repo} is not here - the pin is checked on the cluster')
        return

    def git(*a):
        return subprocess.run(['git', '-C', repo] + list(a), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              universal_newlines=True).stdout.strip()
    wrong = [f for f, b in PIN.PIN_BLOBS.items() if git('rev-parse', f'{PIN.PIN_COMMIT}:{f}') != b]
    assert not wrong, f'PIN_BLOBS does not match clv15 {PIN.PIN_COMMIT[:7]} for: {wrong}'
    missing = [f for f in PIN.PIN_BLOBS if not os.path.exists(os.path.join(ENGINE, f))]
    assert not missing, f'pinned files missing from lib/snapshot_engine/: {missing}'
    drift = []
    for f, b in sorted(PIN.PIN_BLOBS.items()):
        live = git('hash-object', f)
        if live and live != b:
            n = len(git('log', '--oneline', f'{PIN.PIN_COMMIT}..HEAD', '--', f).splitlines())
            drift.append(f'{f} ({n} commit(s) since the pin{"" if n else "; uncommitted change"})')
    assert not drift, ('production changed ported engine files since the pin (clv15 %s): %s - port them and move the pin'
                       % (PIN.PIN_COMMIT[:7], '; '.join(drift)))


TESTS = [test_row_medians_equal_numpy, test_detect_is_the_serial_loop_for_every_thread_count,
         test_highpass_is_identical_for_every_thread_count, test_admission_is_identical_for_every_thread_count,
         test_a_missing_default_target_list_refuses, test_check_default_lists_refuses_before_any_work,
         test_wideband_accumulation_per_instant_equals_production_loop, test_cutouts_are_identical_for_every_thread_count,
         test_movie_frames_binned_identically_for_every_thread_count, test_production_pin]


def main():
    if _IMPORT_ERROR is not None:
        print(f'UNTESTED-IMPORT: cannot import dependencies in this interpreter ({_IMPORT_ERROR}) - run under the pipeline env')
        return 1
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f'PASS  {t.__name__}')
        except Exception as e:
            failed += 1
            print(f'FAIL  {t.__name__}: {e.__class__.__name__}: {str(e)[:400]}')
    print(f'{len(TESTS) - failed}/{len(TESTS)} passed (numpy {np.__version__})')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
