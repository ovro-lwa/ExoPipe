#!/usr/bin/env python3
"""Data-free tests of `flow --tier sublst --stages refine-nonmember` (lib/flow/sublst_nonmember_refine.py): the
x2-aligned residual of ONE night that is not in the window Masters, against the installed window Masters.

The tree is the synthetic one of tests/test_sublst_nonmember.py (a night's window stacks with the records of its stack
stage, installed window Masters with their weights records, an exclusion list), made square, with the names of the member
nights' stacks next to it (the independent model of the verdict counts nights by the stack files' names). The aligner
(phase2/align_images.py, a GPU child) is replaced by a function that writes the displacement field it is told to: the
unit's own arithmetic, its header, its log lines, its gates and the batch stage's verdict run for real.

Run: python3 tests/test_sublst_refine_nonmember.py   (also collected by pytest)
"""
import argparse
import contextlib
import glob
import io
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

_IMPORT_ERROR = None
try:
    import numpy as np
    from astropy.io import fits

    import test_sublst_nonmember as F
    if F._IMPORT_ERROR is not None:
        raise F._IMPORT_ERROR
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import fitsio, legacy, record
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.flow import steps_common as C
    from exopipe.phase3.lib.flow import steps_sublst as SS
    from exopipe.phase3.lib.flow import sublst_nonmember_refine as R
    from exopipe.phase3.lib.flow import tests as FT
    from exopipe.phase3.steps import flow as FLOW
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e
    if __name__ != '__main__':          # under pytest: a collection error that names the missing module
        raise

N = 16                                  # pixels per side of the synthetic images (the x2 arithmetic needs square images)
LST, NEW, CELL = 'LST', 'NEW', 'CELL'   # replaced below by the fixture module's values
if _IMPORT_ERROR is None:
    LST, NEW, CELL = F.LST, F.NEW, F.CELL


@contextlib.contextmanager
def stage(flow=(0.0, 0.0), rc=0, drop=(), listed=True, approved=(), **tree):
    """A synthetic root ready for the stage. flow: the (dx, dy) displacement in x2 pixels the stand-in aligner returns for
    every window; rc: its exit code; drop: windows it returns no field for; listed: the night is on the non-member nights
    list; approved: [(subband, window)] rows of the approved alignment refusals. Yields a namespace: t (the Tree), calls
    (one dict per aligner call), scratch, args, run(), unit(run, sb), log(sb), x2(sb, S, k)."""
    with contextlib.ExitStack() as st:
        st.enter_context(F.Patch((F, 'SHAPE', (N, N))))
        t = st.enter_context(F.Tree(**tree))
        for p in glob.glob(os.path.join(t.root, LST, CELL, '*', '*', f'sublst_{CELL}_*_w[0-9].fits')):
            with fits.open(p, mode='update') as h:          # the celestial cards the x2 images carry (core.residuals)
                h[0].header['CTYPE1'], h[0].header['CTYPE2'] = 'RA---SIN', 'DEC--SIN'
                h[0].header['CRVAL1'], h[0].header['CRVAL2'] = 135.0, 37.0
                h[0].header['CDELT1'], h[0].header['CDELT2'] = -0.1, 0.1
                h[0].header['CRPIX1'], h[0].header['CRPIX2'] = N / 2 + 1.0, N / 2 + 1.0
        frozen = {sb: [(d, 0.95, 1, w) for d, w in zip(F.MEMBERS, F.WEIGHTS)] for sb in t.sbs}
        li = st.enter_context(F.ListInputs(t, frozen, listed=[(LST, NEW)] if listed else []))
        for sb in t.sbs:                # the member nights' stacks, by name only (never read by this stage)
            for S in t.stokes:
                for n in F.MEMBERS:
                    d = os.path.join(t.root, LST, f'{LST}_{n}', sb, S)
                    os.makedirs(d, exist_ok=True)
                    for k in t.windows:
                        open(os.path.join(d, f'sublst_{LST}_{n}_{sb}_{S}_w{k}.fits'), 'w').close()
        scratch = os.path.join(t.dir, 'scratch')
        lock = os.path.join(t.dir, 'gpu0.lock')
        open(lock, 'w').close()
        appr = os.path.join(t.dir, 'approved.csv')
        with open(appr, 'w') as f:
            f.write('lst,subband,night,window,approval\n'
                    + ''.join(f'{LST},{sb},{NEW},{k},synthetic approval\n' for sb, k in approved))
        calls = []

        def aligner(target, args=(), python='default', env=None, log=None, unit=None, **kw):
            args = list(args)
            lines = [ln.split() for ln in open(args[args.index('--pairs') + 1]).read().splitlines() if ln.strip()]
            calls.append(dict(target=target, args=args, python=python, env=dict(env or {}), pairs=lines, log=log))
            with open(log, 'a') as f:
                f.write(f'  Batch: {len(lines)} pairs\n')
                for i, (ep, rf, wd) in enumerate(lines, 1):
                    f.write(f'  [{i}/{len(lines)}] {os.path.basename(ep)}\n')
                    k = int(re.search(r'_w(\d)_up2\.fits$', ep).group(1))
                    assert os.path.dirname(ep) == os.path.dirname(rf) == wd
                    n2 = fits.getheader(ep)['NAXIS1']
                    assert n2 == 2 * N == fits.getheader(rf)['NAXIS1']
                    if k in drop:
                        f.write('    ERROR: synthetic failure of this pair\n')
                        continue
                    fl = np.zeros((n2, n2, 2), np.float32)
                    fl[..., 0], fl[..., 1] = flow
                    np.save(ep[:-5] + '_flow.npy', fl)
            return types.SimpleNamespace(rc=rc, wall_s=0.01, tail='synthetic aligner output')

        rules = types.ModuleType('phase3_tier_rules')       # clv15's veto list is not data-free: no night vetoed here
        rules.membership_vetoed = lambda lst, sb, night: False
        had = sys.modules.get('phase3_tier_rules')
        sys.modules['phase3_tier_rules'] = rules
        st.callback(lambda: sys.modules.pop('phase3_tier_rules') if had is None
                    else sys.modules.__setitem__('phase3_tier_rules', had))
        st.enter_context(F.Patch(
            (legacy, 'run', aligner), (SS, 'drift_guard', lambda note: None),
            (C, 'required_gpu_locks', lambda gpu, gate, h=None: [lock]),
            (fitsio, 'scratch_root', lambda work=None: scratch),
            (legacy, 'interpreter', lambda name='default': sys.executable),
            (CFG.flow_refine, 'WEIGHTS_CSV', os.path.join(li.wdir, '{lst}', '{sb}', 'weights.csv')),
            (CFG.flow_refine, 'APPROVED_REFUSALS_CSV', appr)))
        args = argparse.Namespace(gpus='0', fresh=False, supersede=None, work=None, sublst_stack_root=t.root, resume=False,
                                  workers=None, sublst_jobs=None, sublst_deepflow_jobs=None, sublst_windows=None)

        def run(sbs=None, dry=False, **over):
            a = argparse.Namespace(**dict(vars(args), **over))
            r = record.Run('flow', ['test'], t.root, CFG, dry_run=dry, fresh=bool(a.fresh), resume=bool(a.resume))
            its = [(LST, sb, [NEW]) for sb in (sbs or t.sbs)]
            if dry:
                return r, R.plan_stage(a, r, its, t.root)
            R.run_stage(a, r, its, t.root)
            return r

        def unit(r, sb):
            hits = [u for u in r.units if u.name == R.unit_name(LST, sb, NEW)]
            assert len(hits) == 1, [(u.name, u.status) for u in r.units]
            return hits[0]

        x2root = os.path.join(t.root, f'{LST}_x2')
        yield types.SimpleNamespace(
            t=t, li=li, calls=calls, scratch=scratch, args=args, run=run, unit=unit, x2root=x2root,
            log=lambda sb: os.path.join(x2root, 'logs', f'{LST}_{sb}.log'),
            x2=lambda sb, S, k: FT.residual_path(x2root, LST, NEW, sb, S, k),
            on_disk=lambda: sorted(glob.glob(os.path.join(x2root, f'{LST}_*', '*', '*', '*_resid_loo.fits'))))


def read(path):
    with fits.open(path, memmap=False) as h:
        return np.asarray(h[0].data), h[0].header.copy()


# =====================================================================================================================
# the arithmetic and the header
# =====================================================================================================================

def nyquist_halved(a):
    """What the x2 round trip (upsample, then decimate) returns for an image, written with numpy.fft, not the unit's
    scipy.fft: the image with its Fourier components on the Nyquist row and column halved. The x2 image is the REAL part
    of a zero-padded Fourier plane that holds the Nyquist frequency on one side only, so it keeps half of it; an image
    with no power there (every real stack: several pixels per beam) comes back unchanged."""
    n = a.shape[0]
    w = np.ones((n, n))
    w[n // 2, :] = w[:, n // 2] = 0.5
    return np.real(np.fft.ifft2(np.fft.fft2(a) * w))


def test_x2_arithmetic():
    rng = np.random.RandomState(1)
    a = rng.normal(3.0, 1.0, (N, N))
    up = R.fft_upsample2(a, 2)
    assert up.shape == (2 * N, 2 * N)
    assert np.allclose(up[::2, ::2], a, atol=1e-12)                          # the x2 image passes through the samples
    assert np.allclose(R.fft_upsample2(np.full((N, N), 2.5), 2), 2.5, atol=1e-12)        # the flux scale is kept
    assert np.allclose(R.fft_decimate2(up, 2), nyquist_halved(a), atol=1e-12)
    assert not np.allclose(nyquist_halved(a), a, atol=1e-3)                  # white noise has power at Nyquist ...
    w0 = np.ones((N, N))
    w0[N // 2, :] = w0[:, N // 2] = 0.0
    band = np.real(np.fft.ifft2(np.fft.fft2(a) * w0))                        # ... this image has none: the exact inverse
    assert np.allclose(R.fft_decimate2(R.fft_upsample2(band, 2), 2), band, atol=1e-12)
    d = a.copy()
    d[5, 5] = np.nan                                                         # no data in the stack: zero in the x2 image
    assert np.allclose(R.fft_decimate2(R.fft_upsample2(d, 2), 2), nyquist_halved(np.nan_to_num(d)), atol=1e-12)
    master = rng.normal(3.0, 0.2, (N, N))
    master[0, 0] = np.nan
    zero = np.zeros((2 * N, 2 * N, 2), np.float32)
    r = R.x2_residual(d, R.fft_upsample2(d, 2), zero, master, 2, 0.999)
    assert r.dtype == np.float32 and np.isnan(r[5, 5]) and np.isnan(r[0, 0]) and int(np.isnan(r).sum()) == 2
    v = np.isfinite(r)
    assert np.allclose(r[v], (nyquist_halved(np.nan_to_num(d)) - master)[v], atol=1e-6)  # no displacement
    half = zero.copy()
    half[..., 0] = 1.0                                                       # one x2 pixel in x = half a native pixel
    rs = R.x2_residual(d, R.fft_upsample2(d, 2), half, master, 2, 0.999)
    # the footprint rule: a pixel whose x2 source footprint is not valid to 0.999 is no data - the pixel without data
    # spreads to its neighbor in x, and the column that reads beyond the image edge is no data too
    assert np.isnan(rs[5, 5]) and np.isnan(rs[5, 4]) and np.all(np.isnan(rs[:, -1])) and np.isfinite(rs[5, 6])
    assert R.flow_median_px(half) == 0.5 and R.flow_median_px(zero) == 0.0
    # a displacement of two x2 pixels moves a compact, band-limited source by exactly one native pixel:
    # output[y, x] = input[y, x + 1] (the convention of lib.masters.coadd.apply_flow_field)
    n = 32
    yy, xx = np.mgrid[:n, :n]
    blob = np.exp(-((yy - 16.0) ** 2 + (xx - 15.0) ** 2) / (2 * 2.0 ** 2))
    two = np.zeros((2 * n, 2 * n, 2), np.float32)
    two[..., 0] = 2.0
    rb = R.x2_residual(blob, R.fft_upsample2(blob, 2), two, np.zeros((n, n)), 2, 0.999)
    assert np.all(np.isnan(rb[:, -1])) and np.allclose(rb[:, :-1], blob[:, 1:], atol=1e-6)
    assert np.unravel_index(np.nanargmax(rb), rb.shape) == (16, 14) and R.flow_median_px(two) == 1.0


def test_residual_header():
    sh = fits.Header()
    sh['BUNIT'] = 'Jy/beam'
    sh['DATE-OBS'], sh['DATE-END'] = '2024-02-01T10:00:00.000', '2024-02-01T10:00:50.155'
    sh['NONMEMB'] = (True, 'night outside the deep co-add')
    sh['BINMSTR'] = 'a card the residual must not carry'
    mh = fits.Header()
    mh['JUPMASK'] = 25.15
    h = R.residual_header(sh, mh, '/x/09h_masters/55MHz/I/sublst_master_09h_55MHz_I_w3.fits', 3, 8, 'abc1234', 0.06543219)
    # production's cards in production's order (clv15/phase3_sublst_refine.py:225-244), after the stack's own
    assert list(h.keys()) == ['BUNIT', 'DATE-OBS', 'DATE-END', 'NONMEMB', 'SUBTRACT', 'LOONCOAD', 'LOOMSTR', 'CODE',
                              'LSTBIN', 'DURATION', 'REFINE', 'REFREF', 'FLOWMEDP', 'JUPMASK'], list(h.keys())
    assert h['SUBTRACT'] == 'leave-one-out window Master'
    assert h.comments['SUBTRACT'] == 'sub-LST arm B: x2-aligned to that Master first'
    assert h['LOONCOAD'] == 8 and h.comments['LOONCOAD'] == 'weighted nights in the leave-one-out Master'
    assert h['LOOMSTR'] == 'sublst_master_09h_55MHz_I_w3.fits' and h['CODE'] == 'abc1234'
    assert h.comments['CODE'] == R.CODE_COMMENT and h['LSTBIN'] == 3 and h['DURATION'] == 50.2
    assert h['REFINE'] == 'x2 flow to the subtracted LOO window Master' and h.comments['REFINE'] == 'Gregg 2026-09-25'
    assert h['REFREF'] == 'LOO window Master, this night left out, Stokes I'
    assert h['FLOWMEDP'] == 0.0654 and h['JUPMASK'] == 25.15
    assert h['NONMEMB'] is True and 'BINMSTR' not in h and 'BINMSTR' in sh                # the stack's header is not changed


def test_windows_to_align():
    def item(S, k, refuse=None):
        return dict(S=S, k=k, refuse=refuse)
    a = dict(items=[item('I', 0), item('V', 0), item('I', 1, 'N-7: no installed window Master'), item('V', 1),
                    item('V', 2), item('I', 3), item('V', 3, 'N-7: no installed window Master')])
    ks, items, left = R.windows_to_align(a)
    assert ks == [0, 3] and set(items) == {('I', 0), ('V', 0), ('I', 1), ('V', 1), ('V', 2), ('I', 3), ('V', 3)}
    assert left == {('I', 1): 'no-master', ('V', 3): 'no-master',
                    ('V', 1): 'i-window',          # the Stokes I window has no Master: production refuses the V window
                    ('V', 2): 'i-stack'}, left     # the NIGHT has no Stokes I stack of the window: gate N-11
    # the lines of windows left out for every night end as the approvals judge accepts them
    # (exopipe_ops/sublst_refine_approved.py:27), and the V line is production's (refine.py:139-140)
    judge = r"REFUSED-WINDOW 18MHz [IV] w\d+: .*arm A's rule, Gregg 2026-10-06\)$"
    for (S, k), kind in (('I', 1), 'no-master'), (('V', 3), 'no-master'), (('V', 1), 'i-window'):
        assert re.match(judge, R.window_line('18MHz', S, k, kind)), R.window_line('18MHz', S, k, kind)
    assert R.window_line('18MHz', 'V', 1, 'i-window') == ("REFUSED-WINDOW 18MHz V w1: no Stokes I flow to register V (Stokes "
                                                         "I window refused; arm A's rule, Gregg 2026-10-06)")
    script = '/lustre/gh/main/exopipe_ops/sublst_refine_approved.py'       # where the judge exists: its patterns, in place
    if os.path.exists(script):
        text = open(script).read()
        assert "arm A's rule, Gregg 2026-10-06\\)$" in text and "(\\S+): no flow returned$" in text


def test_aligner_errors_and_log_merge():
    text = ('  Batch: 3 pairs\n  [1/3] a\n   ok\n  [2/3] b\n    ERROR: out of memory\n  [3/3] c\n   Time: 3 s\n'
            '  [9/3] d\n  ERROR after an unknown pair\n')
    assert R.aligner_errors(text, [0, 2, 5]) == [(2, 'ERROR: out of memory'), (None, 'ERROR after an unknown pair')]
    assert R.aligner_errors('ERROR before any pair', [0]) == [(None, 'ERROR before any pair')]
    assert R.aligner_errors(None, [0]) == []
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix='sublst_refine_nonmember_')
    try:
        log = os.path.join(d, 'logs', '09h_55MHz.log')
        new = ['ALIGNED 09h 55MHz w0 2024-02-09 median_flow=0.100 px', 'ALIGNED 09h 55MHz w1 2024-02-09 median_flow=0.200 px']
        assert R.log_gate(log, '09h', '55MHz') is None                       # no log yet: the unit writes the first one
        final, markers = R.merge_log(log, '09h', '55MHz', '2024-02-09', new)
        assert final == 'SUBLST-REFINE 09h 55MHz aligned=2 refused=0 refused_rule=0 refused_night=0' and not markers
        assert open(log).read().splitlines() == new + [final] and os.listdir(os.path.dirname(log)) == ['09h_55MHz.log']
        # a batch run's log: its lines of the OTHER nights stay byte for byte, in place; the counts are the whole log's
        batch = ["REFUSED-WINDOW 55MHz I w5: 1 weighted night(s) of 3 (< 2; arm A's rule, Gregg 2026-10-06)",
                 "REFUSED-WINDOW 55MHz V w5: 1 weighted night(s) of 3 (< 2; arm A's rule, Gregg 2026-10-06)",
                 'ALIGNED 09h 55MHz w0 2024-02-01 median_flow=0.300 px',
                 'ALIGNER-ERROR 55MHz w1 2024-02-01:     ERROR: reference_sky is all zeros',
                 'REFUSED-NIGHT 55MHz w1 2024-02-01: no flow returned',
                 'REFUSED-NIGHT 55MHz I w0 2024-02-09: no co-add night weight',          # true before the night was listed
                 'REFUSED-RESIDUAL 55MHz I w2 2024-02-02: 1 weighted night(s) in its leave-one-out Master (< 2; per-night '
                 'rule, Gregg 2026-10-07)',
                 'ALIGNED 09h 55MHz w0 2024-02-02 median_flow=0.400 px', 'a line of bytes \udcff\udcfe kept as they are',
                 'SUBLST-REFINE 09h 55MHz aligned=2 refused=1 refused_rule=2 refused_night=1']
        with open(log, 'w', encoding='utf-8', errors='surrogateescape') as f:
            f.write('\n'.join(batch) + '\n')
        raw = open(log, 'rb').read()
        assert R.log_gate(log, '09h', '55MHz') is None
        window = [R.window_line('55MHz', 'I', 5, 'no-master'), R.window_line('55MHz', 'V', 4, 'no-master')]
        final, markers = R.merge_log(log, '09h', '55MHz', '2024-02-09', window + new)
        others = [ln for ln in batch[:-1] if '2024-02-09' not in ln]
        got = open(log, encoding='utf-8', errors='surrogateescape').read().splitlines()
        assert got == others + [window[1]] + new + [final], got              # w5 I had its line already; w4 V is new
        assert final == 'SUBLST-REFINE 09h 55MHz aligned=4 refused=1 refused_rule=3 refused_night=1' and not markers
        assert raw.split(b'SUBLST-REFINE')[0].replace(b'REFUSED-NIGHT 55MHz I w0 2024-02-09: no co-add night weight\n', b'') \
            == open(log, 'rb').read()[:len(raw.split(b'SUBLST-REFINE')[0]) - len(b'REFUSED-NIGHT 55MHz I w0 2024-02-09: no co-add night weight\n')]
        # the same night again replaces its lines (a second run is not counted twice), with one window without a flow
        again = [new[0], 'REFUSED-NIGHT 55MHz w1 2024-02-09: no flow returned']
        final, _ = R.merge_log(log, '09h', '55MHz', '2024-02-09', again)
        assert final == 'SUBLST-REFINE 09h 55MHz aligned=3 refused=2 refused_rule=3 refused_night=1', final
        got = open(log, encoding='utf-8', errors='surrogateescape').read().splitlines()
        assert got == others + [window[1]] + again + [final]
        # the judge's count: refused=M with M 'REFUSED-NIGHT <sb> w<k> <night>: no flow returned' lines
        assert sum(1 for ln in got if re.match(r'REFUSED-NIGHT 55MHz w(\d) (\S+): no flow returned$', ln)) == 2
        # a night marked incomplete: its lines go, the log has NO final line, and no other night's merge writes one
        final, markers = R.merge_log(log, '09h', '55MHz', '2024-02-09', [], marker='earlier residuals moved aside')
        got = open(log, encoding='utf-8', errors='surrogateescape').read().splitlines()
        assert final is None and markers == [f'{R.INCOMPLETE} 55MHz 2024-02-09: earlier residuals moved aside']
        assert got == others + [window[1]] + markers and R.log_gate(log, '09h', '55MHz') is None
        final, markers = R.merge_log(log, '09h', '55MHz', '2024-02-10', ['ALIGNED 09h 55MHz w0 2024-02-10 median_flow=0.000 px'])
        assert final is None and len(markers) == 1 and '2024-02-09' in markers[0]
        assert not any(ln.startswith('SUBLST-REFINE') for ln in open(log, encoding='utf-8', errors='surrogateescape'))
        final, markers = R.merge_log(log, '09h', '55MHz', '2024-02-09', new)  # the marked night's own unit, run again
        assert final == 'SUBLST-REFINE 09h 55MHz aligned=5 refused=1 refused_rule=3 refused_night=1' and not markers
        assert os.listdir(os.path.dirname(log)) == ['09h_55MHz.log']         # no temporary file left
        # N-13: a log without a final line and without a marker is a batch refine's, running or failed
        with open(log, 'w') as f:
            f.write(batch[2] + '\n')
        assert 'N-13' in R.log_gate(log, '09h', '55MHz') and R.BATCH_STAGE in R.log_gate(log, '09h', '55MHz')
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_by_night():
    its = [('09h', '18MHz', ['2024-02-01']), ('09h', '73MHz', ['2024-02-01', '2024-02-02']), ('10h', '18MHz', ['2024-02-01'])]
    assert R.by_night(its) == [('09h', '2024-02-01', ['18MHz', '73MHz']), ('09h', '2024-02-02', ['73MHz']),
                               ('10h', '2024-02-01', ['18MHz'])]
    try:
        R.by_night([('09h', '18MHz', None)])
    except Refusal as e:
        assert 'named nights' in str(e) and R.BATCH_STAGE in str(e), e
    else:
        raise AssertionError('an item without a named night was not refused')


# =====================================================================================================================
# the unit
# =====================================================================================================================

def test_unit_writes_the_residuals():
    with stage() as s:
        t = s.t
        r = s.run()
        for sb in t.sbs:
            u = s.unit(r, sb)
            assert u.status == 'done', (u.status, u.reason)
            assert len(u.outputs) == 8 and not u.refused_items and u.tests and all(x['passed'] for x in u.tests), u.tests
            assert {x['name'] for x in u.tests} == {'sublst_x2_residual.allowed_and_complete', 'sublst_x2_residual.cards'}
            assert u.admitted == [f'{NEW}/{sb}']
            for S in t.stokes:
                for k in t.windows:
                    d = t.data[(sb, S, k)]
                    stack64, master64 = d['stack'].astype(np.float64), d['master'].astype(np.float64)
                    want = np.where(np.isfinite(stack64) & np.isfinite(master64),      # no displacement
                                    nyquist_halved(np.nan_to_num(stack64)) - master64, np.nan)
                    got, h = read(s.x2(sb, S, k))
                    assert got.dtype == np.dtype('>f4') and np.array_equal(np.isnan(got), np.isnan(want))
                    v = np.isfinite(want)
                    assert np.isnan(got[0, 0]) and np.isnan(got[5, 5]) and np.allclose(got[v], want[v], atol=2e-6)
                    assert h['LOONCOAD'] == 3 and h['LOOMSTR'] == os.path.basename(t.master_paths(sb, S, k)[0])
                    assert h['JUPMASK'] == 30.52 and h['LSTBIN'] == k and h['DURATION'] == 50.2 and h['FLOWMEDP'] == 0.0
                    assert h['CODE'] == r.commit[:68] and h.comments['CODE'] == R.CODE_COMMENT and 'BINMSTR' not in h
                    sh = fits.getheader(t.window(sb, S, k))
                    for key in ('DATE-OBS', 'DATE-END', 'LSTHOUR', 'LSTWIN', 'NFRAMES', 'NONMEMB', 'FLUXBAKE', 'BUNIT'):
                        assert h[key] == sh[key], key                        # the stack's own time block and flux cards
            # production's log lines, in window order, and the final line the frames tier's refine gate reads
            assert open(s.log(sb)).read().splitlines() == [
                f'ALIGNED {LST} {sb} w{k} {NEW} median_flow=0.000 px' for k in t.windows] + [
                f'SUBLST-REFINE {LST} {sb} aligned=4 refused=0 refused_rule=0 refused_night=0']
            m = u.meta['refine']
            assert m['rc'] == 0 and m['residuals'] == 8 and m['windows'] == list(t.windows) and m['log'] == s.log(sb)
            assert set(m['timing_s']) == {'gates', 'x2_images', 'aligner_and_lock', 'residuals', 'verdict'}
            assert u.meta['verdict']['expected'] == 8 and u.meta['verdict']['missing'] == 0
        # ONE aligner process per unit, every window of the subband in its pairs file, the Stokes I images only
        assert len(s.calls) == len(t.sbs)
        for call, sb in zip(s.calls, t.sbs):
            assert call['target'] == R.ALIGN and os.path.isfile(R.ALIGN) and call['python'] == 'flow'
            assert call['args'][0] == '--pairs' and call['args'][2:] == ['--no_qa', '--save_flow']
            assert [os.path.basename(p[0]) for p in call['pairs']] == [f'ep_{NEW}_w{k}_up2.fits' for k in t.windows]
            assert [os.path.basename(p[1]) for p in call['pairs']] == [f'ref_{NEW}_w{k}_up2.fits' for k in t.windows]
            assert call['env']['CUDA_VISIBLE_DEVICES'] == '0' and call['pairs'][0][2].startswith(s.scratch)
            assert f'{LST}_{sb}_{NEW}_nonmember' in call['pairs'][0][2]
        assert not glob.glob(os.path.join(s.scratch, '**', '*.*'), recursive=True)        # the scratch files are gone
        assert not glob.glob(os.path.join(s.scratch, '*', '*', '*'))                      # and the unit's directories
        # no other night's residual, and nothing under the stack tree
        assert len(s.on_disk()) == 16 and all(f'{LST}_{NEW}' in p for p in s.on_disk())
        assert not t.residuals()


def test_the_displacement_is_applied_to_both_stokes():
    with stage(flow=(1.0, 0.0)) as s:                                        # one x2 pixel = half a native pixel in x
        t = s.t
        r = s.run(sbs=['18MHz'])
        assert s.unit(r, '18MHz').status == 'done', s.unit(r, '18MHz').reason
        for S in t.stokes:
            for k in t.windows:
                got, h = read(s.x2('18MHz', S, k))
                assert h['FLOWMEDP'] == 0.5
                d = t.data[('18MHz', S, k)]
                stack64, master64 = d['stack'].astype(np.float64), d['master'].astype(np.float64)
                # the footprint rule: the night's pixel without data spreads to its neighbor in x, the last column reads
                # beyond the image; the Master's pixel without data stays
                assert np.isnan(got[5, 4]) and np.isnan(got[5, 5]) and np.isnan(got[0, 0]) and np.all(np.isnan(got[:, -1]))
                assert int(np.isnan(got).sum()) == N + 3, int(np.isnan(got).sum())
                want = R.x2_residual(stack64, R.fft_upsample2(stack64, 2), _const_flow(1.0, 0.0), master64, 2,
                                     float(CFG.flow_refine.X2_VALID_FOOTPRINT))
                assert np.array_equal(got.astype(np.float32), want, equal_nan=True), (S, k)   # Stokes V by the I field
                still = np.where(np.isfinite(stack64) & np.isfinite(master64),
                                 nyquist_halved(np.nan_to_num(stack64)) - master64, np.nan)
                both = np.isfinite(got) & np.isfinite(still)
                assert both.sum() > 100 and np.nanmax(np.abs(got[both] - still[both])) > 0.5, (S, k)   # and it moved
        assert open(s.log('18MHz')).read().splitlines()[0] == f'ALIGNED {LST} 18MHz w0 {NEW} median_flow=0.500 px'
    # a displacement of a whole native pixel carries data INTO the night's pixel without data: the arithmetic (production's)
    # allows it, the content test (no finite residual pixel where the stack has no data) refuses the unit, and a refused
    # unit leaves no residual and no log line behind
    with stage(flow=(2.0, 0.0)) as s:
        d = s.t.data[('18MHz', 'I', 0)]
        stack64, master64 = d['stack'].astype(np.float64), d['master'].astype(np.float64)
        got = R.x2_residual(stack64, R.fft_upsample2(stack64, 2), _const_flow(2.0, 0.0), master64, 2,
                            float(CFG.flow_refine.X2_VALID_FOOTPRINT))
        assert np.isfinite(got[5, 5]) and np.isnan(got[5, 4])
        r = s.run(sbs=['18MHz'])
        u = s.unit(r, '18MHz')
        assert u.status == 'refused' and 'G-C8' in u.reason, (u.status, u.reason)
        assert 'finite pixels where the stack has no data' in u.reason, u.reason
        assert not s.on_disk() and not u.outputs and not os.path.exists(s.log('18MHz'))
        assert any('every residual this unit wrote was removed' in n for n in u.notes), u.notes
        assert u.failed_tests()                                              # the tests stay in the record


def _const_flow(dx, dy):
    fl = np.zeros((2 * N, 2 * N, 2), np.float32)
    fl[..., 0], fl[..., 1] = dx, dy
    return fl


def test_refusals_before_anything_is_written():
    # a night that IS a member of an installed window Master: the batch chain's work, never this unit's - no-input (Gregg
    # 2026-11-13: the per-hour chain goes on)
    with stage() as s:
        t = s.t
        for S in t.stokes:
            for k in range(6):
                t.write_master('18MHz', S, k, rows=list(zip(F.MEMBERS, F.WEIGHTS)) + [(NEW, 0.4)])
        r = s.run()
        u = s.unit(r, '18MHz')
        assert u.status == 'no-input' and 'IS in the installed window Master' in u.reason, (u.status, u.reason)
        assert "'sublst-lst'" in u.reason and len(u.meta['master_member']) == 12, u.meta
        assert s.unit(r, '73MHz').status == 'done'                           # the other subband is not held back
        assert len(s.calls) == 1 and not os.path.exists(s.log('18MHz')) and len(s.on_disk()) == 8
    # an excluded night: no aligner, no file
    with stage(excluded=[(LST, '18MHz', NEW)]) as s:
        r = s.run(sbs=['18MHz'])
        u = s.unit(r, '18MHz')
        assert u.status == 'refused' and 'excluded' in u.reason and not s.calls and not s.on_disk(), u.reason
    # no installed window Master of the subband
    with stage() as s:
        import shutil
        shutil.rmtree(os.path.join(s.t.root, f'{LST}_masters', '18MHz'))
        r = s.run(sbs=['18MHz'])
        u = s.unit(r, '18MHz')
        assert u.status == 'refused' and 'N-6' in u.reason and not s.calls and not s.on_disk(), u.reason
    # a selection outside the night's job plan owes nothing
    with stage() as s:
        r = s.run(sbs=['41MHz'])
        assert [(u.name, u.status) for u in r.units] == [(f'{R.STAGE}_{LST}_{NEW}', 'no-input')], r.units
        assert not s.calls and not s.on_disk()
    # N-12: a member night's window stacks are gone, so the verdict's count of weighted nights is not the Master's:
    # refused before the aligner, not after it
    with stage() as s:
        for S in s.t.stokes:
            os.remove(os.path.join(s.t.root, LST, f'{LST}_{F.MEMBERS[0]}', '18MHz', S, f'sublst_{LST}_{F.MEMBERS[0]}_18MHz_{S}_w0.fits'))
        r = s.run(sbs=['18MHz'])
        u = s.unit(r, '18MHz')
        assert u.status == 'refused' and 'N-12' in u.reason and 'I w0: 3 in the Master' in u.reason, u.reason
        assert not s.calls and not s.on_disk() and not os.path.exists(s.log('18MHz'))
    # N-13: the subband's refine log is a batch refine's, without its final line
    with stage() as s:
        os.makedirs(os.path.dirname(s.log('18MHz')))
        with open(s.log('18MHz'), 'w') as f:
            f.write(f'ALIGNED {LST} 18MHz w0 {F.MEMBERS[0]} median_flow=0.100 px\n')
        r = s.run(sbs=['18MHz'])
        u = s.unit(r, '18MHz')
        assert u.status == 'refused' and 'N-13' in u.reason and not s.calls and not s.on_disk(), u.reason
        assert open(s.log('18MHz')).read().count('\n') == 1
    # the gates of the GPU work and the option of the batch stage: no aligner, no file
    for patch, over, word in (((fitsio, 'scratch_root', _refuse('no node-local scratch root')), {}, 'scratch root'),
                              ((legacy, 'interpreter', _refuse("no interpreter 'flow'")), {}, 'interpreter'),
                              ((C, 'required_gpu_locks', _refuse('G-C11: no lock')), {}, 'no lock'),
                              (None, dict(sublst_windows='0,1'), '--sublst-windows'),
                              (None, dict(gpus=''), 'G-C11')):
        with stage() as s:
            with F.Patch(*([patch] if patch else []) + [(C, 'site_gpu_ids', lambda: [])]):
                r = s.run(sbs=['18MHz'], **over)
                _, rc = s.run(sbs=['18MHz'], dry=True, **over)
            u = s.unit(r, '18MHz')
            assert u.status == 'refused' and word in u.reason and rc == 2, (word, u.status, u.reason, rc)
            assert not s.calls and not s.on_disk() and not os.path.exists(s.log('18MHz'))
    # "before anything is MOVED": a gate that refuses a second run with --fresh --supersede leaves the earlier residuals,
    # the log and the superseded area as they were
    with stage() as s:
        r = s.run(sbs=['18MHz'])
        assert s.unit(r, '18MHz').status == 'done'
        first = {p: open(p, 'rb').read() for p in s.on_disk()}
        log = open(s.log('18MHz')).read()
        for patch, over in (((fitsio, 'scratch_root', _refuse('no node-local scratch root')), {}),
                            ((legacy, 'interpreter', _refuse("no interpreter 'flow'")), {}),
                            (None, dict(sublst_windows='0,1'))):
            with F.Patch(*([patch] if patch else [])):
                r = s.run(sbs=['18MHz'], fresh=True, supersede='TAG9', **over)
            assert s.unit(r, '18MHz').status == 'refused', s.unit(r, '18MHz').reason
            assert {p: open(p, 'rb').read() for p in s.on_disk()} == first and open(s.log('18MHz')).read() == log
            assert not os.path.exists(os.path.join(s.t.root, '_superseded')) and len(s.calls) == 1
        r = s.run(sbs=['18MHz'], fresh=True)                                 # --fresh without its tag: also before the GPU
        assert s.unit(r, '18MHz').status == 'refused' and '--supersede' in s.unit(r, '18MHz').reason and len(s.calls) == 1


def _refuse(text):
    def f(*a, **k):
        raise Refusal(text)
    return f


def test_a_night_in_every_subband_master_ends_no_input():
    # a night IN the window Masters of every selected subband: the stack stage wrote no plan for it, so the stage ends
    # no-input for the night, never refused (review of pull request 25, finding 1)
    with stage() as s:
        t = s.t
        for sb in t.sbs:
            for S in t.stokes:
                for k in range(6):
                    t.write_master(sb, S, k, rows=list(zip(F.MEMBERS, F.WEIGHTS)) + [(NEW, 0.4)])
        r = s.run()
        assert [(u.name, u.status) for u in r.units] == [(f'{R.STAGE}_{LST}_{NEW}', 'no-input')], \
            [(u.name, u.status, u.reason) for u in r.units]
        assert 'IS in the installed window Master' in r.units[0].reason and not s.calls and len(s.on_disk()) == 0


def test_earlier_residuals_need_fresh_and_supersede():
    with stage() as s:
        r = s.run(sbs=['18MHz'])
        assert s.unit(r, '18MHz').status == 'done'
        first = {p: open(p, 'rb').read() for p in s.on_disk()}
        r = s.run(sbs=['18MHz'])                                             # G-C5: never over earlier residuals
        u = s.unit(r, '18MHz')
        assert u.status == 'refused' and 'G-C5' in u.reason and '--fresh' in u.reason, u.reason
        assert len(s.calls) == 1 and {p: open(p, 'rb').read() for p in s.on_disk()} == first
        r = s.run(sbs=['18MHz'], fresh=True, supersede='TAG1')
        assert s.unit(r, '18MHz').status == 'done', s.unit(r, '18MHz').reason
        moved = sorted(glob.glob(os.path.join(s.t.root, '_superseded', 'TAG1', f'{LST}_x2', f'{LST}_{NEW}', '18MHz', '*',
                                              '*_resid_loo.fits')))
        assert len(moved) == 8 and sorted(open(p, 'rb').read() for p in moved) == sorted(first.values())
        assert len(s.on_disk()) == 8                                         # the superseded ones are not under <LST>_x2
        # the log is merged, not replaced and not doubled: the night's lines once, the counts of one run
        assert open(s.log('18MHz')).read().splitlines() == [
            f'ALIGNED {LST} 18MHz w{k} {NEW} median_flow=0.000 px' for k in s.t.windows] + [
            f'SUBLST-REFINE {LST} 18MHz aligned=4 refused=0 refused_rule=0 refused_night=0']
        assert set(os.listdir(os.path.join(s.x2root, 'logs'))) == {'_leases', f'{LST}_18MHz.log'}
        assert not glob.glob(os.path.join(s.x2root, 'logs', '_leases', '*', 'holders', '*'))       # the lease was given back
        # a run that fails AFTER it moved the earlier residuals aside: none of its own stays, and the log says the night
        # is incomplete (no final line: the frames tier's refine gate refuses the subband) until the unit has run again
        real, n = fitsio.write_fits, []

        def failing(path, *a, **k):
            n.append(path)
            if len(n) == 3:
                raise OSError('synthetic failure of the third write')
            return real(path, *a, **k)

        with F.Patch((fitsio, 'write_fits', failing)):
            r = s.run(sbs=['18MHz'], fresh=True, supersede='TAG2')
        u = s.unit(r, '18MHz')
        assert u.status == 'error' and 'synthetic failure' in u.reason, (u.status, u.reason)
        assert not s.on_disk() and not u.outputs and not glob.glob(os.path.join(s.x2root, '**', '.*tmp*'), recursive=True)
        assert len(glob.glob(os.path.join(s.t.root, '_superseded', 'TAG2', f'{LST}_x2', f'{LST}_{NEW}', '18MHz', '*',
                                          '*_resid_loo.fits'))) == 8
        lines = open(s.log('18MHz')).read().splitlines()
        assert len(lines) == 1 and lines[0].startswith(f'{R.INCOMPLETE} 18MHz {NEW}: ') and 'TAG2' in lines[0], lines
        assert not glob.glob(os.path.join(s.scratch, '**', '*.*'), recursive=True)
        assert not glob.glob(os.path.join(s.x2root, 'logs', '_leases', '*', 'holders', '*'))
        r = s.run(sbs=['18MHz'])                                             # nothing to move now: no --fresh needed
        assert s.unit(r, '18MHz').status == 'done', s.unit(r, '18MHz').reason
        assert len(s.on_disk()) == 8 and open(s.log('18MHz')).read().splitlines()[-1] == \
            f'SUBLST-REFINE {LST} 18MHz aligned=4 refused=0 refused_rule=0 refused_night=0'
        assert R.INCOMPLETE not in open(s.log('18MHz')).read()


def test_a_failure_leaves_no_residual():
    with stage(rc=1) as s:                                                   # the aligner: nothing was written or moved yet
        r = s.run(sbs=['18MHz'])
        u = s.unit(r, '18MHz')
        assert u.status == 'error' and 'aligner exit 1' in (u.reason or ''), (u.status, u.reason)
        assert not s.on_disk() and not glob.glob(os.path.join(s.scratch, '**', '*.*'), recursive=True)
        assert not os.path.exists(s.log('18MHz'))                            # the subband's log is not touched
    assert 1 not in tuple(CFG.flow_refine.ALIGNER_OK_RC) and 0 in tuple(CFG.flow_refine.ALIGNER_OK_RC)
    with stage() as s:                                                       # with earlier residuals: they stay in place
        r = s.run(sbs=['18MHz'])
        first = {p: open(p, 'rb').read() for p in s.on_disk()}
        log = open(s.log('18MHz')).read()
        with F.Patch((legacy, 'run', lambda *a, **k: types.SimpleNamespace(rc=1, wall_s=0.0, tail='out of memory'))):
            r = s.run(sbs=['18MHz'], fresh=True, supersede='TAG3')
        assert s.unit(r, '18MHz').status == 'error', s.unit(r, '18MHz').reason
        assert {p: open(p, 'rb').read() for p in s.on_disk()} == first and open(s.log('18MHz')).read() == log
        assert not os.path.exists(os.path.join(s.t.root, '_superseded'))
        assert not glob.glob(os.path.join(s.scratch, '**', '*.*'), recursive=True)
    with stage() as s:                                                       # a write fails in one worker thread
        real, n = fitsio.write_fits, []

        def failing(path, *a, **k):
            n.append(path)
            if len(n) == 5:
                raise OSError('synthetic failure of the fifth write')
            return real(path, *a, **k)

        with F.Patch((fitsio, 'write_fits', failing)):
            r = s.run(sbs=['18MHz'])
        u = s.unit(r, '18MHz')
        assert u.status == 'error' and 'synthetic failure' in u.reason and len(n) >= 5, (u.status, u.reason, len(n))
        assert not s.on_disk() and not u.outputs                             # the residuals written before it are removed
        assert not glob.glob(os.path.join(s.x2root, '**', '.*tmp*'), recursive=True)
        assert not os.path.exists(s.log('18MHz')) and not glob.glob(os.path.join(s.scratch, '**', '*.*'), recursive=True)


def test_a_window_without_a_flow():
    with stage(drop=(2,)) as s:                                              # the aligner returns no field for window 2
        r = s.run(sbs=['18MHz'])
        u = s.unit(r, '18MHz')
        assert u.status == 'refused' and 'G-C7' in u.reason, (u.status, u.reason)
        # a refused unit leaves no residual and no final line for the frames tier's refine gate; its lines are its notes
        assert not s.on_disk() and not u.outputs and not os.path.exists(s.log('18MHz'))
        assert f'ALIGNER-ERROR 18MHz w2 {NEW}: ERROR: synthetic failure of this pair' in u.notes
        assert f'REFUSED-NIGHT 18MHz w2 {NEW}: no flow returned' in u.notes
    with stage(drop=(2,), approved=[('18MHz', 2)]) as s:                     # the same, with exactly that pair approved
        # after a batch run whose log holds the lines of the other nights: they stay, and the counts are the whole log's
        batch = [f'ALIGNED {LST} 18MHz w{k} {n} median_flow=0.100 px' for n in F.MEMBERS for k in s.t.windows]
        os.makedirs(os.path.dirname(s.log('18MHz')))
        with open(s.log('18MHz'), 'w') as f:
            f.write('\n'.join(batch + [f'SUBLST-REFINE {LST} 18MHz aligned={len(batch)} refused=0 refused_rule=0 '
                                       f'refused_night=0']) + '\n')
        r = s.run(sbs=['18MHz'])
        u = s.unit(r, '18MHz')
        assert u.status == 'done' and len(u.outputs) == 6 and len(u.refused_items) == 2, (u.status, u.reason)
        assert u.meta['refine']['rc'] == 3 and u.meta['verdict']['approved_pairs'] == [f'{NEW} w2']
        assert not os.path.exists(s.x2('18MHz', 'I', 2)) and not os.path.exists(s.x2('18MHz', 'V', 2))
        lines = open(s.log('18MHz')).read().splitlines()
        assert lines[:len(batch)] == batch
        assert lines[len(batch):] == [
            f'ALIGNER-ERROR 18MHz w2 {NEW}: ERROR: synthetic failure of this pair',
            f'ALIGNED {LST} 18MHz w0 {NEW} median_flow=0.000 px', f'ALIGNED {LST} 18MHz w1 {NEW} median_flow=0.000 px',
            f'REFUSED-NIGHT 18MHz w2 {NEW}: no flow returned', f'ALIGNED {LST} 18MHz w3 {NEW} median_flow=0.000 px',
            f'SUBLST-REFINE {LST} 18MHz aligned={len(batch) + 3} refused=1 refused_rule=0 refused_night=0'], lines[len(batch):]
        assert u.meta['refine']['final'] == lines[-1]


def test_resume_skips_a_unit_valid_by_its_record():
    with stage() as s:
        r = s.run(sbs=['18MHz'])
        assert s.unit(r, '18MHz').status == 'done'
        first = {p: (open(p, 'rb').read(), os.stat(p).st_mtime_ns) for p in s.on_disk()}
        log = open(s.log('18MHz')).read()
        # the record's own check (fingerprints, code, config, outputs) by a stand-in: this checkout may have local edits
        with F.Patch((record, 'resume_check', lambda *a, **k: (True, 'stand-in'))):
            r = s.run(sbs=['18MHz'], resume=True)
            _, rc = s.run(sbs=['18MHz'], resume=True, dry=True)
            assert s.unit(r, '18MHz').status == 'skipped-valid' and rc == 0, (s.unit(r, '18MHz').status, rc)
            assert len(s.calls) == 1 and open(s.log('18MHz')).read() == log
            assert {p: (open(p, 'rb').read(), os.stat(p).st_mtime_ns) for p in s.on_disk()} == first
            assert record.load_record(s.t.root, 'flow', R.unit_name(LST, '18MHz', NEW))['status'] == 'done'
            # an input the record does not name (a window stack of the night written since): not skipped, and the unit
            # then refuses over its earlier residuals instead of replacing them unasked
            assert R.resume_valid(r, s.t.root, LST, NEW, '18MHz', (), R.unit_name(LST, '18MHz', NEW))[0]
            extra = os.path.join(s.t.dir, 'a_new_input.txt')
            open(extra, 'w').close()
            ok, why = R.resume_valid(r, s.t.root, LST, NEW, '18MHz', (extra,), R.unit_name(LST, '18MHz', NEW))
            assert not ok and 'an input the record does not name' in why, why
            os.remove(s.x2('18MHz', 'V', 3))                                 # the residuals on disk are not the record's
            ok, why = R.resume_valid(r, s.t.root, LST, NEW, '18MHz', (), R.unit_name(LST, '18MHz', NEW))
            assert not ok and 'residuals on disk' in why, why
            r = s.run(sbs=['18MHz'], resume=True)
            u = s.unit(r, '18MHz')
            assert u.status == 'refused' and 'G-C5' in u.reason and len(s.calls) == 1, (u.status, u.reason)
    with stage() as s:                                                       # the record's check says no: the unit runs
        r = s.run(sbs=['18MHz'])
        with F.Patch((record, 'resume_check', lambda *a, **k: (False, 'stand-in: config changed'))):
            r = s.run(sbs=['18MHz'], resume=True)
        u = s.unit(r, '18MHz')
        assert u.status == 'refused' and 'G-C5' in u.reason, (u.status, u.reason)


def test_dry_run_plans_and_writes_nothing():
    with stage() as s:
        before = sorted(glob.glob(os.path.join(s.t.dir, '**', '*'), recursive=True))
        r, rc = s.run(dry=True)
        assert rc == 0 and not s.calls and not r.units
        assert sorted(glob.glob(os.path.join(s.t.dir, '**', '*'), recursive=True)) == before
        for S in s.t.stokes:
            for k in range(6):
                s.t.write_master('18MHz', S, k, rows=list(zip(F.MEMBERS, F.WEIGHTS)) + [(NEW, 0.4)])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _, rc = s.run(dry=True)
        assert rc == 0 and not s.calls, rc                                   # a member night: no-input, not a refusal
        assert 'NO-INPUT - ' in out.getvalue() and 'IS in the installed window Master' in out.getvalue(), out.getvalue()


# =====================================================================================================================
# the stage in the flow step, and the verdict's model
# =====================================================================================================================

def test_stage_wiring():
    assert R.STAGE == 'refine-nonmember' and R.STAGE in SS.STAGES and R.STAGE in FLOW.STAGES['sublst']
    assert R.STAGE not in CFG.flow.DEFAULT_STAGES['sublst']                  # on request only
    assert list(SS.STAGES).index('refine') < list(SS.STAGES).index(R.STAGE) < list(SS.STAGES).index('check')
    with stage() as s:
        a = argparse.Namespace(**dict(vars(s.args), flow_stages=('refine', R.STAGE), lsts=None, cells=None,
                                      epochs=CELL, nights=None, scopes='18MHz', exclude_cells=None))
        r = record.Run('flow', ['test'], s.t.root, CFG)
        try:
            SS.run(a, r)
        except Refusal as e:
            assert 'both write the x2 residuals' in str(e), e
        else:
            raise AssertionError('refine and refine-nonmember in one invocation were not refused')
        assert not s.calls and not s.on_disk()
        # through the step itself: --stages as the command line gives it, resolved by the flow step
        a = argparse.Namespace(**dict(vars(s.args), tier='sublst', stages=R.STAGE, from_stage=None, until_stage=None,
                                      lsts=None, cells=None, epochs=CELL, nights=None, scopes='18MHz', exclude_cells=None))
        r = record.Run('flow', ['test'], s.t.root, CFG)
        assert FLOW.run(a, r) is None and a.flow_stages == [R.STAGE]
        assert [(u.name, u.status) for u in r.units] == [(R.unit_name(LST, '18MHz', NEW), 'done')], r.units
    ns = argparse.Namespace(tier='sublst', stages=None, from_stage=None, until_stage=None)
    assert R.STAGE not in FLOW.resolve_stages(ns)                            # in no default
    ns.stages = f'check,{R.STAGE},refine'
    assert FLOW.resolve_stages(ns) == ['refine', R.STAGE, 'check']           # always in pipeline order
    ns.stages, ns.from_stage = None, R.STAGE
    assert R.STAGE not in FLOW.resolve_stages(ns)                            # --from never adds a stage the list leaves out


def test_the_model_counts_listed_nights_without_a_frozen_row():
    with stage() as s:                                                       # listed, no frozen row: weight 0
        wts, _ = FT._weights(LST, '18MHz')
        assert wts == dict(zip(F.MEMBERS, F.WEIGHTS), **{NEW: 0.0}), wts
    with stage(listed=False) as s:                                           # not listed: unknown to the model
        wts, _ = FT._weights(LST, '18MHz')
        assert NEW not in wts and wts == dict(zip(F.MEMBERS, F.WEIGHTS)), wts
        r = s.run(sbs=['18MHz'])                 # the verdict would refuse its residuals: refused before the aligner
        u = s.unit(r, '18MHz')
        assert u.status == 'refused' and 'N-12' in u.reason and 'does not owe' in u.reason, (u.status, u.reason)
        assert not s.calls and not s.on_disk()


TESTS = [test_x2_arithmetic, test_residual_header, test_windows_to_align, test_aligner_errors_and_log_merge,
         test_by_night, test_unit_writes_the_residuals, test_the_displacement_is_applied_to_both_stokes,
         test_refusals_before_anything_is_written, test_a_night_in_every_subband_master_ends_no_input, test_earlier_residuals_need_fresh_and_supersede,
         test_a_failure_leaves_no_residual, test_a_window_without_a_flow, test_resume_skips_a_unit_valid_by_its_record,
         test_dry_run_plans_and_writes_nothing,
         test_stage_wiring, test_the_model_counts_listed_nights_without_a_frozen_row]


def main(argv):
    if _IMPORT_ERROR is not None:
        print(f'UNTESTED-IMPORT: {type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}')
        return 2
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f'PASS {fn.__name__}')
        except Exception:  # noqa: BLE001
            import traceback
            failed += 1
            print(f'FAIL {fn.__name__}')
            traceback.print_exc()
    print(f'{len(TESTS) - failed} passed, {failed} failed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
