#!/usr/bin/env python3
"""STANDING TEST - the size / axis-ratio rule at every S/N (lib/search/size_rule.py; Gregg 2026-10-22 "All approved",
R-GREGG-20261022-ALL-APPROVED-SIZE-GATE-READMISSION-SUBAGENTS; spec exopipe_ops/refactor_20261012/plan/specs/fp_gates_v2.md).
Synthetic frames and the repo's two calibration products; no production file is written.

  measure      a noiseless beam fits back to 1.00 beam, axis 1.00; a source twice the beam to 2.00 (the run's own fitter)
  products     the calibration tables read: both repo products load with the md5 config pins, calibrate I and V x low and
               high at 5-30 sigma, and equal the 2026-10-22 analysis products; malformed / incomplete / other-md5 refuse
  snr          the S/N dependence: linear in log S/N between the calibrated S/N, clamped below 5 and above 30; thresholds fall
               with S/N (deep I low: 2.99 at 7 sigma, the 1.5 floor from ~15); a missing Stokes / class refuses
  floor        max(1.5, q99): never below the ratified 1.5 floors, at every S/N, Stokes, band class, in both tiers
  verdict      the rule at high S/N (the floors) and at low S/N (the envelope); |S/N|; no fit / no S/N keeps; an unknown band
               name refuses (no silent low-band calibration); the deep Full band under FULL_BAND_SIZE 'off' (R-REFACTOR-20261104-
               AXIS-RULE-ALONE-FULLBAND-SIZE-OFF-FROM-PULSARS): no removal by size there (its excess still computed), the axis
               test applied, every other band and 'rule' unchanged
  injection    end to end on beam-correlated noise: the fixed 1.5 / 1.5 rule loses 7-sigma point sources, the envelope rule
               built from the same injections keeps >= 97.5% at every S/N
  domain       injections only inside the search's domain; isolated zeros allowed, > 10% refused
  annulus      the S/N's noise is measured outside the fitted source (an extended source keeps its S/N)
  stamp        the frame stamp and its edge fallback (6 beams), off the frame: no measurement
  engine       engine_test (deep_search._size_test's rule branch): columns recorded (size_role 'gate'); a 4-beam source at S/N
               7.6 removed that the fixed rule (above 8 sigma only) keeps; Stokes V fitted whole (production: the PyBDSF
               component); a point source kept; production's record-only flags above 8 sigma; an epoch without a beam refuses;
               the deep Full band under 'off': a 3-beam source kept by size (size_role marked ':full_size_off ...', its flag
               saying why) that every other band and 'rule' remove, an elongated one removed by its axis ratio
  early        every candidate row (spec section 4): a Stokes I row below the local-SNR floor judged as the fork's member pool
               ('reject_local_snr:<verdict>' when removed - never a fork member), every other early removal recorded only
               (size_role 'record', its reason kept); the engine's source calls them on every early path; in the deep Full band
               under 'off' the pool is judged by axis only and every row's role is marked
  plumbing     the rule's values and environment (unknown rule, missing md5 refuse; production sets only the rule), the floors
               (a SIZE_FLOOR_BEAMS / AXIS_REMOVE other than 1.5 refuses under the rule, in the step and the engine), the
               product's imaging configuration (10-minute arm A, REFINE_X2 on, another master generation refuse), the config
               keys and pins, the parity file's production values, the steps' environments; search_detect.FULL_BAND_SIZE
               ('off', ruled; no parity line - production has no rule) handed to the deep engine child only
  records      the run's size_rule.json: written under the rule only, one rule per run (an append under another or of another
               version or Full-band size test refuses), one rule per catalog (a mix refuses; another than configured
               refuses), read from a ledger header (size-rule-1's too); a size-rule-2 record (no full_band_size) read as 'rule'
  downstream   the catalog builder's 7-8 sigma gate replaced (P3_SIZE_GATE_7_8 = inf); the conformance's fixed SIZE step
               contradicts the rule (refused); its fixed AXIS step under catalog_conform.CONFORM_AXIS_UNDER_RULE: 'replace'
               (the ruled default, R-REFACTOR-20261104-AXIS-RULE-ALONE-FULLBAND-SIZE-OFF-FROM-PULSARS) - P3_CONFORM_AXIS=0, the
               production script's switch required (refused without it), any axis removal refused; 'second_gate'
               (production's behavior, the parity file's value) - removed in production's words and recorded, never refused;
               each removed row counted once (the fork's tag copies); size_axis under both; final rows' findings (a Full-band
               final above its size excess passes under 'off', by its axis alone)
  ledger       the gate-ledger content test: GLED_COLS_SIZE (size_role) on every row with the run's record, 'gate' rows' removals
               the rule's own verdicts, 'record' rows an earlier gate's, the member pool's reasons; the deep Full band under
               'off': every row of it marked, none other, none removed by size; a size-rule-1 ledger read by its own checks
  catalog      the 10-minute catalog's search jobs (one rule, the configured one: another refuses at S-G7), the incremental
               deep path's ledger header: the builder's 7-8 sigma gate replaced under the rule, the axis step per the key
  patch        the pending production patch (docs/pending_production_patch_p3s2_conform_axis.diff) applied to a scratch copy of
               production's p3s2_conform.py: it applies cleanly, carries the switch (size_rule.conform_switch), default on, its
               preamble's patched md5 is the result's
  carry        the per-instant carry stays within its instant, and nothing carries from the deep Full band's size test
Run standalone:  python tests/test_size_rule.py [test_name ...]
"""
import csv
import json
import math
import os
import shutil
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

import numpy as np                                                  # noqa: E402
from scipy.ndimage import gaussian_filter                           # noqa: E402
from exopipe.phase3.lib.search import size_rule as SZ               # noqa: E402

T = '/tmp/exopipe3_tests_size_rule'
BMAJ, BMIN, BPA, CD = 0.207, 0.184, 161.5, 0.0767                  # a deep 36 MHz frame (Run_P3S5): 2.70 px beam
ANA = '/lustre/gh/main/phase3/refactor_analysis/bright_source_contamination_20261020'
DEEP = os.path.join(REPO, 'exopipe', 'common', 'size_envelope_deep.json')
SUBLST = os.path.join(REPO, 'exopipe', 'common', 'size_envelope_sublst.json')


def _close(a, b, tol, what):
    assert abs(a - b) <= tol, '%s: %r not within %r of %r' % (what, a, tol, b)


def _refuses(fn, what):
    try:
        fn()
    except SZ.RuleRefusal as e:
        assert 'REFUSING' in str(e), (what, str(e))
        return str(e)
    raise AssertionError(f'{what}: did not refuse')


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def _noise(rng, Rn, bpix):
    """Beam-correlated noise (interferometric noise is correlated on the beam scale; white noise barely moves the fit)."""
    return gaussian_filter(rng.normal(0.0, 1.0, (2 * Rn + 41, 2 * Rn + 41)), bpix / 2.3548)[20:-20, 20:-20]


# ---------------------------------------------------------------------------------------------------------------------------
def test_measure():
    bpix = BMAJ / CD
    R = SZ.stamp_radius(bpix)
    size, axis, err, peak = SZ.measure(10.0 * SZ.beam_image(BMAJ, BMIN, BPA, CD, R), R, bpix, CD, BMAJ, BMIN)
    _close(size, 1.0, 0.005, 'noiseless beam size'); _close(axis, 1.0, 0.01, 'noiseless beam axis'); _close(peak, 10.0, 0.01, 'peak')
    R2 = 3 * SZ.stamp_radius(bpix)                                    # the fitter takes its own stamp from the center
    size2, axis2, _, _ = SZ.measure(SZ.beam_image(2 * BMAJ, 2 * BMIN, BPA, CD, R2), R2, bpix, CD, BMAJ, BMIN)
    _close(size2, 2.0, 0.02, 'twice the beam'); _close(axis2, 1.0, 0.02, 'twice the beam, axis')


def test_products():
    C = _cfg()
    for path, sec, ref in ((DEEP, C.search_detect, os.path.join(ANA, 'deep', 'size_cal_v2.json')),
                           (SUBLST, C.search_sublst, os.path.join(ANA, 'tenmin', 'size_cal_v2.json'))):
        assert SZ.resolve(sec.SIZE_ENVELOPE) == path, (sec.SIZE_ENVELOPE, path)
        env = SZ.Envelope.load(path, sec.SIZE_ENVELOPE_MD5).require()          # the config's pin
        assert env.md5 == sec.SIZE_ENVELOPE_MD5 == SZ.md5_file(path)
        assert env.label() == f'envelope:{os.path.basename(path)}:{sec.SIZE_ENVELOPE_MD5[:12]}', env.label()
        for st in SZ.STOKES:
            for cl in SZ.CLASSES:
                for k in SZ.QUANTILE_KEYS:
                    pts = env.points(st, cl, k)
                    assert [p[0] for p in pts] == [float(x) for x in SZ.SNR_GRID], (st, cl, k, pts)
                assert all(env.cal['%s_%s_%g' % (st, cl, s)]['n'] == 720 for s in SZ.SNR_GRID), (st, cl)
        if os.path.exists(ref):                                      # the 2026-10-22 analysis product, byte for byte
            assert SZ.md5_file(ref) == env.md5, ref
    _refuses(lambda: SZ.Envelope.load(DEEP, '0' * 32), 'another md5')
    _refuses(lambda: SZ.Envelope.load(os.path.join(T, 'absent.json')), 'a missing product')
    os.makedirs(T, exist_ok=True)
    bad = os.path.join(T, 'bad.json')
    open(bad, 'w').write('{not json')
    _refuses(lambda: SZ.Envelope.load(bad), 'an unreadable product')
    cal = json.load(open(DEEP))
    _refuses(lambda: SZ.Envelope({k: v for k, v in cal.items() if not k.startswith('V_high')}, 'x').require(),
             'a product without V high')
    _refuses(lambda: SZ.Envelope(dict({k: v for k, v in cal.items() if not k.startswith('I_low_')}, I_low_5=cal['I_low_5']),
                                 'x').require(), 'a product with one I low S/N')
    nan = dict(cal, I_low_7=dict(cal['I_low_7'], eff_q99=float('nan')))
    _refuses(lambda: SZ.Envelope(nan, 'x').require(), 'a non-finite percentile')


def test_snr():
    env = SZ.Envelope.load(DEEP).require()
    q = {s: env.cal['I_low_%g' % s]['eff_q99'] for s in SZ.SNR_GRID}
    _close(env.value('I', 'low', 7, 'eff_q99'), q[7], 1e-12, 'grid point 7')
    mid = math.sqrt(7 * 8)                                            # linear in log S/N: the geometric midpoint
    _close(env.value('I', 'low', mid, 'eff_q99'), (q[7] + q[8]) / 2, 1e-12, 'geometric midpoint of 7 and 8')
    _close(env.value('I', 'low', 3.0, 'eff_q99'), q[5], 1e-12, 'clamped below 5')
    _close(env.value('I', 'low', 99.0, 'eff_q99'), q[30], 1e-12, 'clamped above 30')
    ts7, ta7 = env.thresholds('I', 'low', 7.0)
    _close(ts7, 2.9883, 1e-3, 'deep I low size threshold at 7 sigma'); _close(ta7, 4.1101, 1e-3, 'deep I low axis threshold at 7')
    seq = [env.thresholds('I', 'low', s)[0] for s in (5, 6, 7, 8, 10, 12, 15, 20, 30)]
    assert all(a >= b for a, b in zip(seq, seq[1:])), f'the size threshold falls with S/N: {seq}'
    assert seq[-3:] == [1.5, 1.5, 1.5] and seq[5] > 1.5, f'the 1.5 floor binds from 15 sigma (I low): {seq}'
    # the classes and tiers differ (each tier calibrated in its own frames)
    t10 = SZ.Envelope.load(SUBLST).require()
    assert env.thresholds('I', 'high', 7)[0] != env.thresholds('I', 'low', 7)[0]
    assert t10.thresholds('I', 'low', 7)[0] != env.thresholds('I', 'low', 7)[0]
    try:
        SZ.Envelope({'I_low_5': dict(eff_q99=2.0, axis_q99=2.0)}).value('V', 'low', 7, 'eff_q99')
        raise AssertionError('a missing calibration must refuse')
    except KeyError:
        pass


def test_floor():
    for path in (DEEP, SUBLST):
        env = SZ.Envelope.load(path).require()
        for st in SZ.STOKES:
            for cl in SZ.CLASSES:
                for snr in np.exp(np.linspace(math.log(2), math.log(200), 60)):
                    ts, ta = env.thresholds(st, cl, float(snr))
                    assert ts >= 1.5 and ta >= 1.5, (path, st, cl, snr, ts, ta)
                assert env.thresholds(st, cl, 30.0) == (1.5, 1.5), (path, st, cl, env.thresholds(st, cl, 30.0))
    tiny = SZ.Envelope({f'{st}_{cl}_{s}': dict(eff_q99=1.1, axis_q99=1.2) for st in 'IV' for cl in ('low', 'high')
                        for s in (5, 30)}).require()
    assert tiny.thresholds('I', 'low', 5) == (1.5, 1.5), 'the floors bind where the envelope is narrower'


def _cal():
    pts = {5: (4.0, 4.5), 10: (1.4, 1.6), 30: (1.1, 1.15)}
    return SZ.Envelope({'I_low_%d' % s: dict(eff_q99=e, axis_q99=a) for s, (e, a) in pts.items()})


def test_verdict():
    env = _cal()
    rec = dict(eff=1.6, axis=1.0, snr=30.0)
    v, sx, ax = SZ.size_verdict(env, 'I', '18MHz', rec)
    assert v == 'gaussfit-size' and sx > 1, (v, sx)
    v, sx, _ = SZ.size_verdict(env, 'I', '18MHz', dict(rec, snr=5.0))
    assert v == '' and sx < 1, (v, sx)
    v, _, ax = SZ.size_verdict(env, 'I', 'Red', dict(eff=1.0, axis=1.7, snr=-40.0))     # |S/N|; Red is low band
    assert v == 'axis-ratio' and ax > 1, (v, ax)
    assert SZ.size_verdict(env, 'I', '18MHz', None)[0] == '', 'failed fit keeps'
    assert SZ.size_verdict(env, 'I', '18MHz', dict(rec, snr=float('nan')))[0] == '', 'no S/N keeps'
    assert SZ.band_class('Red') == 'low' and SZ.band_class('Green') == 'high' and SZ.band_class('41MHz') == 'high'
    assert SZ.band_class('36MHz') == 'low' and SZ.band_class('Full') == 'high'
    for bad in ('37MHz', 'red', '', None, 'Blue2'):          # a typo or an uncalibrated band never takes the low-band envelope
        _refuses(lambda: SZ.band_class(bad), f'band {bad!r}')
        _refuses(lambda: SZ.judge(env, 'I', bad, rec), f'judging in band {bad!r}')
    assert {SZ.band_class(b) for b in ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz', 'Red')} == {'low'}
    j = SZ.judge(env, 'I', '18MHz', rec)
    assert j['size_verdict'] == 'gaussfit-size' and j['size_excess'] > 1 and j['size_thr'] == 1.5, j
    assert SZ.judge(env, 'I', '18MHz', None)['size_verdict'] == 'fit-failed'
    assert SZ.judge(env, 'I', '18MHz', dict(rec, snr=float('nan')))['size_verdict'] == 'no-snr'
    assert SZ.judge(env, 'I', '18MHz', dict(rec, eff=1.0))['size_verdict'] == 'point-like'
    # the deep Full band (R-REFACTOR-20261104-AXIS-RULE-ALONE-FULLBAND-SIZE-OFF-FROM-PULSARS): under full_band 'off' no removal by
    # size there - its excess still computed and recorded - while the axis test applies; every other band, and 'rule', as before
    on, off = SZ.Envelope.load(DEEP).require(), SZ.Envelope.load(DEEP, full_band='off').require()
    assert on.full_band == 'rule' and off.full_band == 'off', 'the product alone judges every band alike'
    wide, long_, both = dict(eff=3.0, axis=1.0, snr=12.0), dict(eff=1.0, axis=3.5, snr=12.0), dict(eff=3.0, axis=3.5, snr=12.0)
    for st in SZ.STOKES:                                     # 12 sigma: size limits 1.5-1.94, axis 1.5-2.02 in every class
        for band in ('Full', '82MHz', '41MHz', 'Green', 'Blue', 'Red', '36MHz', '18MHz'):
            v, sx, ax = SZ.size_verdict(on, st, band, wide)
            assert v == 'gaussfit-size' and sx > 1, (st, band, v, sx)
            assert SZ.size_verdict(off, st, band, wide) == (('', sx, ax) if band == 'Full' else (v, sx, ax)), (st, band)
            assert SZ.size_verdict(off, st, band, long_)[0] == 'axis-ratio', (st, band)   # the axis test in every band
            assert SZ.size_verdict(off, st, band, both)[0] == ('axis-ratio' if band == 'Full' else 'gaussfit-size'), (st, band)
    j, j0 = SZ.judge(off, 'I', 'Full', wide), SZ.judge(on, 'I', 'Full', wide)
    assert j['size_verdict'] == 'point-like' and j0['size_verdict'] == 'gaussfit-size', (j, j0)
    assert {k: v for k, v in j.items() if k != 'size_verdict'} == {k: v for k, v in j0.items() if k != 'size_verdict'}, \
        'the threshold and the excess recorded alike'
    assert not SZ.size_tested(off, 'Full') and SZ.size_tested(on, 'Full') and SZ.size_tested(off, 'Green')
    assert SZ.FULL_BAND == 'Full' and SZ.band_class('Full') == 'high'
    for bad in ('', 'Off', 'on', 'remove', 'replace'):
        _refuses(lambda: SZ.full_band_mode(bad), f'FULL_BAND_SIZE {bad!r}')
        _refuses(lambda: SZ.Envelope(env.cal, full_band=bad), f'an envelope with full_band {bad!r}')
    assert [SZ.full_band_mode(m) for m in SZ.FULL_BAND_MODES] == ['off', 'rule']


def test_injection():
    rng = np.random.default_rng(3)
    bpix = BMAJ / CD
    R, Rn = SZ.stamp_radius(bpix), SZ.rms_radius(bpix)
    beam = SZ.beam_image(BMAJ, BMIN, BPA, CD, R)
    rows, fixed = [], {}
    for snr in (7, 10, 30):
        lost = 0
        for _ in range(250):
            big = _noise(rng, Rn, bpix)
            rms = SZ.annulus_rms(big, Rn, bpix)
            big[Rn - R:Rn + R + 1, Rn - R:Rn + R + 1] += snr * rms * beam
            rec = SZ.measure_detection(big, Rn, bpix, CD, BMAJ, BMIN)
            if rec is None:
                rows.append(dict(st='I', cls='low', snr=snr, fit=0)); continue
            rows.append(dict(st='I', cls='low', snr=snr, fit=1, size=rec['size'], axis=rec['axis'], eff=rec['eff'],
                             snr_meas=rec['snr']))
            lost += (rec['eff'] > 1.5) or (rec['axis'] > 1.5)
        fixed[snr] = lost / 250.0
    cal = SZ.envelope_from_injections(rows)
    assert set(cal) == {'I_low_7', 'I_low_10', 'I_low_30'}, sorted(cal)
    assert cal['I_low_30']['size_q99'] < 1.25, cal['I_low_30']
    assert fixed[7] > 0.05, 'on beam-correlated noise the fixed rule loses point sources at 7 sigma: %r' % fixed
    env = SZ.Envelope(cal)
    for snr in (7, 10, 30):
        vs = [SZ.size_verdict(env, 'I', '18MHz', dict(eff=o['eff'], axis=o['axis'], snr=o['snr_meas']))[0] for o in rows
              if o['snr'] == snr and o['fit']]
        frac = sum(v != '' for v in vs) / float(len(vs))
        assert frac <= 0.025, 'envelope rule at S/N %d flags %.3f' % (snr, frac)


def test_domain():
    # 00h LST at mid-hour: RA 7.5 deg transits; the north celestial pole is always at elevation = latitude
    assert SZ.in_domain(7.5, 37.0, 7.5) and SZ.in_domain(0.0, 89.9, 7.5)
    assert not SZ.in_domain(187.5, 37.0, 7.5), 'below the horizon'
    assert not SZ.in_domain(float('nan'), 10.0, 7.5), 'off the image (no sky)'
    assert not SZ.in_domain(350.85, 58.0, 350.0), 'within 2 deg of Cas A'
    z = np.ones((21, 21)); z[3, 4] = 0.0
    assert SZ.stamp_ok(z), 'isolated zeros are allowed'
    z[:5, :] = 0.0
    assert not SZ.stamp_ok(z), 'more than 10% zeros: refused'
    z = np.ones((21, 21)); z[0, 0] = np.nan
    assert not SZ.stamp_ok(z)


def test_annulus():
    bpix = BMAJ / CD; Rn = SZ.rms_radius(bpix)
    noise = _noise(np.random.default_rng(5), Rn, bpix)
    rms0 = SZ.annulus_rms(noise, Rn, bpix, 1.0)
    src = 20 * rms0 * SZ.beam_image(3 * BMAJ, 3 * BMIN, BPA, CD, Rn)      # 3 beams wide
    rec = SZ.measure_detection(noise + src, Rn, bpix, CD, BMAJ, BMIN)
    assert rec and abs(rec['size'] - 3.0) < 0.3, rec
    assert 0.8 < rec['snr'] / 20.0 < 1.25, 'S/N %.1f for a 20-sigma 3-beam source' % rec['snr']
    fixed = 20 * rms0 / SZ.annulus_rms(noise + src, Rn, bpix, 1.0)        # the old fixed 2-6 beam annulus
    assert fixed < rec['snr'], 'the fixed annulus lowers the S/N of an extended source: %.1f vs %.1f' % (fixed, rec['snr'])


def _frame(rng, n=400, sources=()):
    bpix = BMAJ / CD
    img = gaussian_filter(rng.normal(0.0, 1.0, (n, n)), bpix / 2.3548)
    img /= np.std(img)
    for x, y, amp, scale in sources:
        R = int(6 * scale * bpix)
        img[y - R:y + R + 1, x - R:x + R + 1] += amp * SZ.beam_image(scale * BMAJ, scale * BMIN, BPA, CD, R)
    return img


def _frame_ell(rng, x, y, amp, smaj, smin, n=400):
    """As _frame, with one ELONGATED source at (x, y): FWHM smaj x BMAJ by smin x BMIN at the beam's PA."""
    bpix = BMAJ / CD
    img = gaussian_filter(rng.normal(0.0, 1.0, (n, n)), bpix / 2.3548)
    img /= np.std(img)
    R = int(6 * max(smaj, smin) * bpix)
    img[y - R:y + R + 1, x - R:x + R + 1] += amp * SZ.beam_image(smaj * BMAJ, smin * BMIN, BPA, CD, R)
    return img


def test_stamp():
    bpix = BMAJ / CD
    img = np.zeros((300, 300)); img[150, 150] = np.nan
    big, R_ = SZ.frame_stamp(img, 150, 150, bpix)
    assert R_ == SZ.rms_radius(bpix) and big.shape == (2 * R_ + 1,) * 2 and np.isfinite(big).all(), (R_, big.shape)
    big, R_ = SZ.frame_stamp(img, SZ.edge_radius(bpix) + 1, 150, bpix)     # near the left edge: the 6-beam stamp
    assert R_ == SZ.edge_radius(bpix), R_
    assert SZ.frame_stamp(img, 3, 150, bpix) == (None, None), 'off the frame: no stamp'
    assert SZ.measure_at(img, 3, 150, BMAJ, BMIN, CD) is None


def test_engine():
    env = SZ.Envelope.load(DEEP).require()

    def frame(amp, scale):                  # one source at (200, 200) on the same beam-correlated noise (seed 11)
        return _frame(np.random.default_rng(11), sources=((200, 200, amp, scale),))
    from exopipe.phase3.lib.search import deep_search as DS
    # a 4-beam source at S/N 7.6: removed by the rule (2.99 beams allowed there); production never tests below 8 sigma
    img = frame(7.5, 4.0)
    c = dict(pix_x=200, pix_y=200, local_snr=7.5)
    sb, v = SZ.engine_test(c, 'I', '36MHz', img, BMAJ, BMIN, CD, env)
    assert v == 'remove' and c['size_verdict'] == 'gaussfit-size' and c['size_excess'] > 1 and c['size_role'] == 'gate', c
    assert c['size_rule'] == env.label() and set(SZ.GLED_COLS_SIZE) <= set(c), c
    assert abs(c['size_beams'] - 4.0) < 0.5 and abs(c['size_snr'] - 7.6) < 0.2 and abs(c['size_thr'] - 2.985) < 0.01, c
    c0 = dict(pix_x=200, pix_y=200, local_snr=7.5)
    assert DS._size_test(c0, 7.5, BMAJ, CD, 'I', img, BMAJ / CD, BMIN) == (c0.get('size_beams'), ''), \
        "production's rule never removes below 8 sigma"
    c1 = dict(pix_x=200, pix_y=200, local_snr=7.5)
    assert DS._size_test(c1, 7.5, BMAJ, CD, 'I', img, BMAJ / CD, BMIN, rule=env, band='36MHz') == (sb, v) and c1 == c, \
        'deep_search._size_test with the rule = engine_test'
    # a 3-beam source at S/N 11: Stokes V on one Gaussian over the whole source (production read the PyBDSF component)
    img = frame(12.0, 3.0)
    cv = dict(pix_x=200, pix_y=200, local_snr=12.0, bmaj_fit_deg=BMAJ, bmaj_fit_err_deg=0.0, bmin_fit_deg=BMIN)
    assert DS._size_test(dict(cv), 12.0, BMAJ, CD, 'V', img, BMAJ / CD, BMIN)[1] == '', \
        "production's Stokes V reads the PyBDSF component (beam-sized here), not the frame"
    cr = dict(cv)
    assert DS._size_test(cr, 12.0, BMAJ, CD, 'V', img, BMAJ / CD, BMIN, rule=env, band='36MHz')[1] == 'remove', cr
    assert abs(cr['size_beams'] - 2.9) < 0.3, cr
    # a bright point source: kept as point-like
    img = frame(40.0, 1.0)
    c = dict(pix_x=200, pix_y=200, local_snr=40.0)
    assert SZ.engine_test(c, 'V', '64MHz', img, BMAJ, BMIN, CD, env)[1] == '' and c['size_verdict'] == 'point-like', c
    assert abs(c['size_beams'] - 1.0) < 0.05 and c['size_excess'] < 1 and c['axis_excess'] < 1 and c['size_thr'] == 1.5, c
    # 1.3 beams at S/N 9: kept, and production's record-only flag above 8 sigma
    img = frame(9.0, 1.3)
    c = dict(pix_x=200, pix_y=200, local_snr=9.0)
    assert SZ.engine_test(c, 'I', 'Red', img, BMAJ, BMIN, CD, env)[1] == 'flag' and c['size_verdict'] == 'point-like', c
    assert c['size_flag'].startswith('size > 1.3 beams (point-like at its S/N'), c
    c = dict(pix_x=200, pix_y=200, local_snr=7.0)
    assert SZ.engine_test(c, 'I', 'Red', img, BMAJ, BMIN, CD, env)[1] == '' and 'size_flag' not in c, 'no flag at 8 sigma or below'
    # no source to fit, a stamp off the frame: kept, recorded; an epoch without a beam refuses
    c = dict(pix_x=200, pix_y=200, local_snr=12.0)           # an empty frame: the fit has no noise annulus - no S/N, kept
    assert SZ.engine_test(c, 'I', 'Red', np.zeros((400, 400)), BMAJ, BMIN, CD, env)[1] == '' \
        and c['size_verdict'] in ('fit-failed', 'no-snr') and c['size_excess'] == '', c
    c = dict(pix_x=2, pix_y=200, local_snr=12.0)
    assert SZ.engine_test(c, 'I', 'Red', img, BMAJ, BMIN, CD, env) == ('', '') and c['size_verdict'] == 'fit-failed', c
    _refuses(lambda: SZ.engine_test(dict(pix_x=200, pix_y=200), 'I', 'Red', img, None, BMIN, CD, env), 'no epoch beam')
    # the deep Full band under FULL_BAND_SIZE 'off' (R-REFACTOR-20261104-AXIS-RULE-ALONE-FULLBAND-SIZE-OFF-FROM-PULSARS): a 3-beam
    # source at S/N 12, removed by size in every other band and under 'rule', is KEPT there - its row marked, its flag saying why
    off = SZ.Envelope.load(DEEP, full_band='off').require()
    img = frame(12.0, 3.0)
    for band, e_, want in (('Full', off, 'flag'), ('Full', env, 'remove'), ('82MHz', off, 'remove'), ('41MHz', off, 'remove'),
                           ('Green', off, 'remove'), ('Blue', off, 'remove'), ('Red', off, 'remove'), ('18MHz', off, 'remove')):
        for st in SZ.STOKES:
            c = dict(pix_x=200, pix_y=200, local_snr=12.0)
            assert SZ.engine_test(c, st, band, img, BMAJ, BMIN, CD, e_)[1] == want, (st, band, e_.full_band, c)
            marked = band == 'Full' and e_.full_band == 'off'
            assert c['size_role'] == (f'gate{SZ.ROLE_SEP}{SZ.FULL_OFF_NOTE}' if marked else 'gate') and c['size_excess'] > 1 \
                and c['size_verdict'] == ('point-like' if marked else 'gaussfit-size'), (st, band, c)
    c = dict(pix_x=200, pix_y=200, local_snr=12.0)
    assert SZ.engine_test(c, 'I', 'Full', img, BMAJ, BMIN, CD, off) == (c['size_beams'], 'flag') and abs(c['size_beams'] - 3.0) < 0.4
    assert c['size_flag'] == f'size > 1.3 beams (kept: {SZ.FULL_OFF_NOTE}; the envelope\'s limit {c["size_thr"]:.2f})', c
    c1 = dict(pix_x=200, pix_y=200, local_snr=12.0)
    assert DS._size_test(c1, 12.0, BMAJ, CD, 'I', img, BMAJ / CD, BMIN, rule=off, band='Full') == (c['size_beams'], 'flag') \
        and c1 == c, 'deep_search._size_test with the rule = engine_test, in the Full band too'
    img = frame(9.0, 1.3)                   # within its size limit there: production's record-only flag in its usual words
    c = dict(pix_x=200, pix_y=200, local_snr=9.0)
    assert SZ.engine_test(c, 'I', 'Full', img, BMAJ, BMIN, CD, off)[1] == 'flag' and c['size_excess'] < 1 \
        and c['size_flag'].startswith('size > 1.3 beams (point-like at its S/N') and c['size_role'].startswith('gate:'), c
    # ... an elongated source (3 x 1 beams) at S/N 12 there is removed by its axis ratio: the axis test applies in the Full band
    ell = _frame_ell(np.random.default_rng(11), 200, 200, 12.0, 3.0, 1.0)
    for band, e_, want in (('Full', off, 'remove-axis'), ('Full', env, 'remove'), ('64MHz', off, 'remove')):
        c = dict(pix_x=200, pix_y=200, local_snr=12.0)
        assert SZ.engine_test(c, 'I', band, ell, BMAJ, BMIN, CD, e_)[1] == want and c['axis_excess'] > 1 \
            and c['size_excess'] > 1, (band, e_.full_band, c)


def test_early():
    """Every candidate row (spec section 4). The multiples fork's member pool - the deep tier's Stokes I rows below the
    local-SNR floor ('reject_local_snr') whose LEDGER S/N (round(local_snr, 2), as deep_search._gled_row writes it and
    exopipe_ops/multiples_fork.py reads it) lies in catalog_fork.MEMBER_SNR, 5 <= |S/N| < 7 - is JUDGED: removed by the rule a
    row is 'reject_local_snr:<verdict>' (never a member nor a calibration row), kept it stays 'reject_local_snr'. Every other
    early removal is RECORDED with its own reason: the local-SNR rows outside the window (the floor removed them; never
    members - the verification of 2026-11-03: 962 of the 1,192 rows of the engine test's nights), every 10-minute row (no fork
    reads a 10-minute ledger), the other identification removals, the master-structure gate, Stokes V."""
    env = SZ.Envelope.load(DEEP).require()
    big = _frame(np.random.default_rng(11), sources=((200, 200, 7.5, 4.0),))      # the 4-beam source test_engine removes
    pt = _frame(np.random.default_rng(11), sources=((200, 200, 7.5, 1.0),))       # a point source at the same S/N
    C = _cfg()
    win = SZ.member_window('deep', from_env=False)
    assert win == tuple(float(x) for x in C.catalog_fork.MEMBER_SNR) == (5.0, 7.0), win
    assert SZ.member_window('10min') is None and SZ.member_window('10min', from_env=False) is None, 'no pool at 10 minutes'
    c = dict(pix_x=200, pix_y=200, local_snr=6.5)
    r = SZ.engine_early(c, 'reject_local_snr', 'I', '36MHz', big, BMAJ, BMIN, CD, env, win)
    assert r == 'reject_local_snr:gaussfit-size' and c['size_role'] == 'gate' and c['size_verdict'] == 'gaussfit-size', (r, c)
    assert c['size_excess'] > 1 and 'size_flag' not in c and abs(c['size_beams'] - 4.0) < 0.5, c
    c = dict(pix_x=200, pix_y=200, local_snr=6.5)
    r = SZ.engine_early(c, 'reject_local_snr', 'I', '36MHz', pt, BMAJ, BMIN, CD, env, win)
    assert r == 'reject_local_snr' and c['size_role'] == 'gate' and c['size_verdict'] == 'point-like', (r, c)
    # the window, on the ledger's rounded S/N: 4.996 is the ledger's 5.0 (in), 4.994 its 4.99 (out), 6.996 its 7.0 (out), the
    # sign ignored; outside it, or with no window (10 minutes), the same source is recorded and its reason kept
    for snr, pool in ((4.996, True), (4.994, False), (5.0, True), (6.994, True), (6.996, False), (7.0, False), (-5.5, True),
                      (-4.2, False), (3.1, False), ('', False), (None, False)):
        for w in ((win, None) if pool else (win,)):
            c = dict(pix_x=200, pix_y=200, local_snr=snr)
            r = SZ.engine_early(c, 'reject_local_snr', 'I', '36MHz', big, BMAJ, BMIN, CD, env, w)
            want_gate = pool and w is not None
            assert (r, c['size_role']) == (('reject_local_snr:gaussfit-size', 'gate') if want_gate
                                           else ('reject_local_snr', 'record')), (snr, w, r, c['size_role'])
            assert c['size_verdict'] == 'gaussfit-size' and set(SZ.GLED_COLS_SIZE) <= set(c), c   # measured either way
    assert SZ.ledger_snr(dict(local_snr=4.996, snr=9.9)) == 5.0 and SZ.ledger_snr(dict(snr='6.123')) == 6.12
    assert SZ.ledger_snr(dict(local_snr='x')) is None and not SZ.in_member_pool(None, win)
    for st, reason in (('I', 'reject_merged'), ('I', 'reject_sigma_stack'), ('I', 'reject_bright_prox'), ('V', 'reject_local_snr'),
                       ('V', 'reject_bilobe'), ('V', 'reject_merged')):
        c = dict(pix_x=200, pix_y=200, local_snr=6.5)
        r = SZ.engine_early(c, reason, st, '36MHz', big, BMAJ, BMIN, CD, env, win)
        assert r == reason and c['size_role'] == 'record' and c['size_verdict'] == 'gaussfit-size', (st, reason, r, c)
        assert c['size_rule'] == env.label() and set(SZ.GLED_COLS_SIZE) <= set(c) and abs(c['size_beams'] - 4.0) < 0.5, c
    c = dict(pix_x=200, pix_y=200, local_snr=9.0)                 # the master-structure gate's rows: recorded, no flags
    assert SZ.engine_record(c, 'I', 'Red', big, BMAJ, BMIN, CD, env) == 'gaussfit-size' and c['size_role'] == 'record' \
        and 'size_flag' not in c, c
    c = dict(pix_x=2, pix_y=200, local_snr=6.5)                   # off the frame: judged as not measured, the reason kept
    assert SZ.engine_early(c, 'reject_local_snr', 'I', 'Red', big, BMAJ, BMIN, CD, env, win) == 'reject_local_snr' \
        and c['size_verdict'] == 'fit-failed' and c['size_role'] == 'gate' and 'size_beams' not in c, c
    _refuses(lambda: SZ.engine_early(dict(pix_x=200, pix_y=200), 'reject_merged', 'I', 'Red', big, None, BMIN, CD, env, win),
             'an early row of an epoch without a beam')
    _refuses(lambda: SZ.engine_early(dict(pix_x=200, pix_y=200, local_snr=6.0), 'reject_local_snr', 'I', 'Red2', big, BMAJ, BMIN,
                                     CD, env, win), 'an unknown band')
    # the deep Full band under FULL_BAND_SIZE 'off': its member pool judged by the axis ratio alone (the 4-beam source stays in
    # the pool, an elongated one leaves it), and every row of the band - judged or recorded - carries the mark
    off = SZ.Envelope.load(DEEP, full_band='off').require()
    ell = _frame_ell(np.random.default_rng(11), 200, 200, 12.0, 3.0, 1.0)      # 3 x 1 beams, measured at S/N ~12 (axis limit 2.0)
    FG, FR = f'gate{SZ.ROLE_SEP}{SZ.FULL_OFF_NOTE}', f'record{SZ.ROLE_SEP}{SZ.FULL_OFF_NOTE}'
    c = dict(pix_x=200, pix_y=200, local_snr=6.5)
    r = SZ.engine_early(c, 'reject_local_snr', 'I', 'Full', big, BMAJ, BMIN, CD, off, win)
    assert r == 'reject_local_snr' and c['size_role'] == FG and c['size_verdict'] == 'point-like' and c['size_excess'] > 1, (r, c)
    c = dict(pix_x=200, pix_y=200, local_snr=6.5)
    r = SZ.engine_early(c, 'reject_local_snr', 'I', 'Full', ell, BMAJ, BMIN, CD, off, win)
    assert r == 'reject_local_snr:axis-ratio' and c['size_role'] == FG and c['axis_excess'] > 1, (r, c)
    for band, e_, want, role in (('Full', env, 'reject_local_snr:gaussfit-size', 'gate'),           # under 'rule'
                                 ('82MHz', off, 'reject_local_snr:gaussfit-size', 'gate'),          # the other bands unchanged
                                 ('Green', off, 'reject_local_snr:gaussfit-size', 'gate')):
        c = dict(pix_x=200, pix_y=200, local_snr=6.5)
        assert SZ.engine_early(c, 'reject_local_snr', 'I', band, big, BMAJ, BMIN, CD, e_, win) == want \
            and c['size_role'] == role, (band, e_.full_band, c)
    for st, reason, snr in (('I', 'reject_merged', 6.5), ('V', 'reject_local_snr', 6.5), ('I', 'reject_local_snr', 4.2)):
        c = dict(pix_x=200, pix_y=200, local_snr=snr)
        assert SZ.engine_early(c, reason, st, 'Full', big, BMAJ, BMIN, CD, off, win) == reason and c['size_role'] == FR \
            and c['size_verdict'] == 'point-like', (st, reason, c)
    c = dict(pix_x=200, pix_y=200, local_snr=9.0)
    assert SZ.engine_record(c, 'I', 'Full', big, BMAJ, BMIN, CD, off) == 'point-like' and c['size_role'] == FR, c
    assert SZ.role_base(FG) == 'gate' and SZ.role_base(FR) == 'record' and SZ.role_base('gate') == 'gate'
    assert SZ.role_full_off(FG) and SZ.role_full_off(FR) and not SZ.role_full_off('gate') and not SZ.role_full_off('')
    # the window the step hands the deep engine child (MEMBER_ENV), read back there; a malformed one refuses
    from exopipe.phase3.lib.search import deep_step, tier_sublst
    keep = os.environ.pop(SZ.MEMBER_ENV, None)
    try:
        e = deep_step.size_rule_env()
        assert e[SZ.MEMBER_ENV] == '5.0,7.0', e
        assert SZ.MEMBER_ENV not in tier_sublst.engine_env(dict(label='SLST10B', arm='B', lst='10h'), T, T), '10 minutes: none'
        os.environ[SZ.MEMBER_ENV] = '5.0,7.0'
        assert SZ.member_window('deep') == (5.0, 7.0) and SZ.member_window('10min') is None
        for bad in ('7.0,5.0', '5.0', 'x,y', '0,7'):
            os.environ[SZ.MEMBER_ENV] = bad
            _refuses(lambda: SZ.member_window('deep'), f'the member window {bad!r}')
    finally:
        os.environ.pop(SZ.MEMBER_ENV, None)
        if keep is not None:
            os.environ[SZ.MEMBER_ENV] = keep
    # the engine: every early path calls the rule, only under it; the window read once (none at 10 minutes); the floors
    # checked before anything is written
    src = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_search.py')).read()
    for call in ("_SZ.engine_early(c, _rsn, 'I', scope, rI, *_szb, cd, _szr, _szm)",
                 "_SZ.engine_early(c, _rsn, 'V', scope, rV, *_szb, cd, _szr)", "_SZ.engine_record(c, 'I', scope, rI, *_szb, cd, _szr)"):
        assert src.count(call) == 1, call
        i = src.index(call)
        assert 'if _szr is not None:' in src[i - 200:i], call
    assert src.count('_szm = _SZ.member_window(cadence) if _szr is not None else None') == 1
    assert src.index("_SZ.check_floors(SIZE_FLOOR_BEAMS, AXIS_REMOVE") < src.index('_SZ.write_run_record(run_root, _szr)')
    # the gate-ledger content test knows the member pool's reasons, and they keep the identification stage's exemption
    from exopipe.phase3.lib.search import tests as ST
    assert all(ST.known_veto_reason(SZ.MEMBER_REASON + v) for v in SZ.REMOVING)
    assert not ST.known_veto_reason(SZ.MEMBER_REASON + 'point-like') and (SZ.MEMBER_REASON + 'x').startswith('reject_')


def test_plumbing():
    C = _cfg()
    _refuses(lambda: SZ.rule_value('fixed'), 'an unknown rule')
    _refuses(lambda: SZ.rule_value(''), 'an empty rule')
    assert SZ.step_env('production', '', '') == {SZ.RULE_ENV: 'production'}
    e = SZ.step_env('envelope', C.search_detect.SIZE_ENVELOPE, C.search_detect.SIZE_ENVELOPE_MD5)
    assert e == {SZ.RULE_ENV: 'envelope', SZ.ENVELOPE_ENV: DEEP, SZ.ENVELOPE_MD5_ENV: C.search_detect.SIZE_ENVELOPE_MD5}, e
    _refuses(lambda: SZ.step_env('envelope', C.search_detect.SIZE_ENVELOPE, ''), 'no md5 pinned')
    _refuses(lambda: SZ.step_env('envelope', C.search_detect.SIZE_ENVELOPE, C.search_sublst.SIZE_ENVELOPE_MD5),
             "the 10-minute pin on the deep product")
    keep = {k: os.environ.get(k) for k in (SZ.RULE_ENV, SZ.ENVELOPE_ENV, SZ.ENVELOPE_MD5_ENV)}
    try:
        for k in keep:
            os.environ.pop(k, None)
        assert SZ.engine_rule('deep').md5 == C.search_detect.SIZE_ENVELOPE_MD5, 'no step environment: the cadence\'s config'
        assert SZ.engine_rule('10min').md5 == C.search_sublst.SIZE_ENVELOPE_MD5
        os.environ.update(SZ.step_env('production', '', ''))
        assert SZ.engine_rule('deep') is None
        os.environ.update({SZ.RULE_ENV: 'envelope', SZ.ENVELOPE_ENV: SUBLST, SZ.ENVELOPE_MD5_ENV: ''})
        _refuses(lambda: SZ.engine_rule('10min'), 'the engine without a pinned md5')
        os.environ[SZ.RULE_ENV] = 'enveloppe'
        _refuses(lambda: SZ.engine_rule('deep'), 'the engine with an unknown rule')
    finally:
        for k, v in keep.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    # config: ruled defaults, pins protected, ruling IDs
    for sec in (C.search_detect, C.search_sublst):
        assert sec.SIZE_RULE == 'envelope' and sec.RULED['SIZE_RULE'] == SZ.RULING, sec
        assert {'SIZE_ENVELOPE', 'SIZE_ENVELOPE_MD5'} <= set(sec.PROTECTED_KEYS), sec.PROTECTED_KEYS
    par = open(os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg')).read().splitlines()
    assert "search_detect.SIZE_RULE = 'production'" in par and "search_sublst.SIZE_RULE = 'production'" in par
    # the steps hand the rule to the engine child
    from exopipe.phase3.lib.search import deep_step, tier_sublst
    os.makedirs(T, exist_ok=True)
    env = deep_step.engine_env(T, T, T, 'X', T, T)
    assert env[SZ.RULE_ENV] == 'envelope' and env[SZ.ENVELOPE_ENV] == DEEP, {k: v for k, v in env.items() if 'SIZE' in k}
    e2 = tier_sublst.engine_env(dict(label='SLST10B', arm='B', lst='10h'), T, T)
    assert e2[SZ.RULE_ENV] == 'envelope' and e2[SZ.ENVELOPE_ENV] == SUBLST, e2
    assert 'lib/search/size_rule.py' in tier_sublst.ENGINE_SOURCES
    pins = {n: (ok, d) for n, p, ok, d, r in deep_step.pinned_inputs('I')}
    assert pins['size envelope (the size / axis rule)'][0], pins
    # the deep Full band's size test (search_detect.FULL_BAND_SIZE, R-REFACTOR-20261104-AXIS-RULE-ALONE-FULLBAND-SIZE-OFF-FROM-
    # PULSARS): 'off', ruled; no parity line (production has no rule: its fixed gate never reads it); the deep engine child gets
    # it (FULL_BAND_ENV), the 10-minute child none (no Full band there: 'rule'); production's environment sets only its rule
    assert C.search_detect.FULL_BAND_SIZE == 'off' and C.ruled('search_detect.FULL_BAND_SIZE') == (SZ.PULSAR_DECISION,)
    assert not hasattr(C.search_sublst, 'FULL_BAND_SIZE') and SZ.FULL_BAND not in C.search_sublst.BANDS
    assert not [ln for ln in par if 'FULL_BAND_SIZE' in ln.split('#', 1)[0]], 'no parity line for FULL_BAND_SIZE'
    assert "catalog_conform.CONFORM_AXIS_UNDER_RULE = 'second_gate'" in par, "production's axis step in the parity file"
    assert env[SZ.FULL_BAND_ENV] == 'off' and SZ.FULL_BAND_ENV not in e2, (env.get(SZ.FULL_BAND_ENV), e2.get(SZ.FULL_BAND_ENV))
    assert SZ.FULL_BAND_ENV not in SZ.step_env('production', '', '')
    assert SZ.configured_record('deep')['full_band_size'] == 'off' and SZ.configured_record('10min')['full_band_size'] == 'rule'
    from exopipe.phase3.core.record import Refusal as Refusal_
    saved_fb, saved_rule = C.search_detect.FULL_BAND_SIZE, C.search_detect.SIZE_RULE
    keep = {k: os.environ.pop(k, None) for k in (SZ.RULE_ENV, SZ.ENVELOPE_ENV, SZ.ENVELOPE_MD5_ENV, SZ.FULL_BAND_ENV)}
    try:
        assert SZ.engine_rule('deep').full_band == 'off' and SZ.engine_rule('10min').full_band == 'rule', 'no step: the config'
        os.environ[SZ.FULL_BAND_ENV] = 'rule'                  # the step's value wins in the child
        assert SZ.engine_rule('deep').full_band == 'rule' and SZ.engine_rule('10min').full_band == 'rule'
        os.environ[SZ.FULL_BAND_ENV] = 'of'
        _refuses(lambda: SZ.engine_rule('deep'), 'the deep engine with an unknown FULL_BAND_SIZE')
        assert SZ.engine_rule('10min').full_band == 'rule', 'the 10-minute engine never reads it'
        os.environ.pop(SZ.FULL_BAND_ENV)
        C.search_detect.FULL_BAND_SIZE = 'rule'
        assert deep_step.engine_env(T, T, T, 'X', T, T)[SZ.FULL_BAND_ENV] == 'rule'
        assert SZ.configured_record('deep')['full_band_size'] == 'rule'
        C.search_detect.FULL_BAND_SIZE = 'partly'
        try:
            deep_step.engine_env(T, T, T, 'X', T, T)
            raise AssertionError('deep engine_env with an unknown FULL_BAND_SIZE must refuse')
        except Refusal_ as e_:
            assert 'FULL_BAND_SIZE' in str(e_) and 'REFUSING' in str(e_), str(e_)
        C.search_detect.SIZE_RULE = 'production'               # production: no rule, the key never read
        assert SZ.FULL_BAND_ENV not in deep_step.engine_env(T, T, T, 'X', T, T)
    finally:
        C.search_detect.FULL_BAND_SIZE, C.search_detect.SIZE_RULE = saved_fb, saved_rule
        for k, v in keep.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    # the floors (finding 6): under the rule SIZE_FLOOR_BEAMS / AXIS_REMOVE must be its 1.5 / 1.5 - a changed value refuses
    # (it would be ignored); production keeps its own fixed rule with them
    dp, dm = C.search_detect.SIZE_ENVELOPE, C.search_detect.SIZE_ENVELOPE_MD5
    assert SZ.step_env('envelope', dp, dm, floors=(1.5, 1.5)) == e
    for fl in ((1.6, 1.5), (1.5, 1.4), ('x', 1.5), (None, 1.5)):
        _refuses(lambda: SZ.step_env('envelope', dp, dm, floors=fl), f'the floors {fl} under the rule')
    assert SZ.step_env('production', '', '', floors=(1.6, 1.4)) == {SZ.RULE_ENV: 'production'}
    # the product's imaging configuration (finding 7b): the 10-minute product is set B's (B and its injection copy BI), no
    # x2 refinement, the ivw-mean-v3 masters; the deep product Run_P3S5's
    fr10, frd = C.search_sublst.SIZE_ENVELOPE_FRAMES, C.search_detect.SIZE_ENVELOPE_FRAMES
    SZ.check_frames(fr10, tier_sublst.size_rule_frames('B'))
    SZ.check_frames(fr10, tier_sublst.size_rule_frames('BI'))
    SZ.check_frames(frd, deep_step.size_rule_frames())
    _refuses(lambda: SZ.check_frames(fr10, tier_sublst.size_rule_frames('A')), 'the 10-minute product on arm A (bin masters)')
    _refuses(lambda: SZ.check_frames(fr10, dict(tier_sublst.size_rule_frames('B'), REFINE_X2=True)), 'with REFINE_X2 on')
    _refuses(lambda: SZ.check_frames(fr10, dict(tier_sublst.size_rule_frames('B'), MASTER_GEN='ivw-mean-v2')),
             'another master generation')
    _refuses(lambda: SZ.check_frames(fr10, {'ARM': 'B'}), 'a search naming no REFINE_X2 / MASTER_GEN')
    _refuses(lambda: SZ.check_frames({}, {'ARM': 'B'}), 'a product without its recorded configuration')
    _refuses(lambda: SZ.check_frames(frd, dict(deep_step.size_rule_frames(), REFINE_X2=False)), 'deep without refinement')
    assert {'SIZE_ENVELOPE_FRAMES'} <= set(C.search_detect.PROTECTED_KEYS) & set(C.search_sublst.PROTECTED_KEYS)
    # ... in the steps: the deep preflight's gate / engine_env and the 10-minute preflight / engine_env refuse
    from exopipe.phase3.core.record import Refusal
    saved = [(C.search_detect, 'SIZE_FLOOR_BEAMS'), (C.search_detect, 'REFINE_X2'), (C.search_sublst, 'AXIS_REMOVE')]
    saved = [(sec, k, getattr(sec, k)) for sec, k in saved]
    try:
        for sec, k, v in ((C.search_detect, 'SIZE_FLOOR_BEAMS', 1.6), (C.search_detect, 'REFINE_X2', False)):
            setattr(sec, k, v)
            try:
                deep_step.engine_env(T, T, T, 'X', T, T)
                raise AssertionError(f'deep engine_env with {k}={v!r} under the rule must refuse')
            except Refusal as e_:
                assert 'REFUSING' in str(e_), str(e_)
            setattr(sec, k, dict((x[1], x[2]) for x in saved)[k])
        try:
            tier_sublst.engine_env(dict(label='SLST10A', arm='A', lst='10h'), T, T)
            raise AssertionError('the 10-minute engine_env on arm A under the rule must refuse')
        except Refusal as e_:
            assert 'imaging configuration' in str(e_) and "ARM = 'A'" in str(e_), str(e_)
        C.search_sublst.AXIS_REMOVE = 1.4
        try:
            tier_sublst.engine_env(dict(label='SLST10B', arm='B', lst='10h'), T, T)
            raise AssertionError('the 10-minute engine_env with AXIS_REMOVE 1.4 under the rule must refuse')
        except Refusal as e_:
            assert 'AXIS_REMOVE' in str(e_), str(e_)
    finally:
        for sec, k, v in saved:
            setattr(sec, k, v)
    src = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_search.py')).read()
    assert src.count('rule=_szr, band=scope)   # WAVE2 SIZE-RULE') == 2, 'both Stokes call the rule'
    assert '_szr = _SZ.engine_rule(cadence)' in src and '_SZ.write_run_record(run_root, _szr)' in src


def test_records():
    shutil.rmtree(T, ignore_errors=True)
    env = SZ.Envelope.load(DEEP, full_band='off').require()          # as the deep engine child applies the configured rule
    rd = os.path.join(T, 'att1', 'transients', 'Run_X')
    os.makedirs(rd)
    assert SZ.write_run_record(rd, None)['rule'] == 'production' and not os.path.exists(os.path.join(rd, SZ.RUN_RECORD)), \
        'production writes nothing'
    rec = SZ.write_run_record(rd, env)
    assert rec == SZ.read_run_record(rd) and rec['rule'] == 'envelope' and rec['md5'] == env.md5 and rec['ruling'] == SZ.RULING
    assert rec['version'] == SZ.RECORD_VERSION == 'size-rule-3' and rec['full_band_size'] == 'off', rec
    assert SZ.run_record(None)['full_band_size'] == '', 'production has no rule'
    assert SZ.write_run_record(rd, env) == rec, 'an append under the same rule'
    _refuses(lambda: SZ.write_run_record(rd, None), 'production into a run searched under the rule')
    other = SZ.Envelope(dict(env.cal), source=SUBLST, md5=SZ.md5_file(SUBLST), full_band='off')
    _refuses(lambda: SZ.write_run_record(rd, other), 'another product into the run')
    rule_env = SZ.Envelope.load(DEEP, full_band='rule').require()
    w = _refuses(lambda: SZ.write_run_record(rd, rule_env), "an append with the Full band's size test on into a run with it off")
    assert SZ.FULL_BAND_WORDS['off'] in w and SZ.FULL_BAND_WORDS['rule'] in w, w
    rd2 = os.path.join(T, 'att2', 'transients', 'Run_X')
    os.makedirs(rd2)
    open(os.path.join(rd2, 'v_deep_ledger.csv'), 'w').write('lst\n')       # searched before, under production
    _refuses(lambda: SZ.write_run_record(rd2, env), 'the rule into a run production searched')
    rd3 = os.path.join(T, 'att3', 'transients', 'Run_X')
    os.makedirs(rd3)
    open(os.path.join(rd3, SZ.RUN_RECORD), 'w').write('{"rule": "envelope"}')
    _refuses(lambda: SZ.read_run_record(rd3), 'a malformed record')
    assert SZ.one_rule([rd, rd]) == rec
    assert SZ.one_rule([rd2])['rule'] == 'production' and SZ.one_rule([]) == SZ.run_record(None)
    _refuses(lambda: SZ.one_rule([rd, rd2]), 'a mix of rules')
    _refuses(lambda: SZ.one_rule([rd2], want=rec), 'production attempts under the configured rule')
    assert SZ.one_rule([rd], want=SZ.configured_record('deep')) == rec
    assert SZ.ledger_rule(os.path.join(rd2, 'v_deep_ledger.csv'))['rule'] == 'production'
    from exopipe.phase3.lib.search import deep_search as DS
    lp = os.path.join(T, 'ledger.csv')
    with open(lp, 'w', newline='') as f:
        csv.writer(f).writerow(DS.GLED_COLS + DS.GLED_COLS_SIZE)
    assert SZ.ledger_rule(lp)['rule'] == 'envelope' and SZ.ledger_rule(os.path.join(T, 'none.csv'))['rule'] == 'production'
    assert SZ.ledger_rule(lp)['version'] == SZ.RECORD_VERSION == 'size-rule-3'
    with open(lp, 'w', newline='') as f:                    # a size-rule-1 search's ledger (8957c42: no size_role)
        csv.writer(f).writerow(DS.GLED_COLS + SZ.GLED_COLS_SIZE_V1)
    assert SZ.ledger_rule(lp)['rule'] == 'envelope' and SZ.ledger_rule(lp)['version'] == 'size-rule-1'
    # one rule per run and per catalog includes the record's version (what the engine wrote on each row)
    rd4 = os.path.join(T, 'att4', 'transients', 'Run_X')
    os.makedirs(rd4)
    json.dump(dict(rec, version='size-rule-1'), open(os.path.join(rd4, SZ.RUN_RECORD), 'w'))
    _refuses(lambda: SZ.write_run_record(rd4, env), 'an append of size-rule-3 onto a size-rule-1 run')
    w = _refuses(lambda: SZ.one_rule([rd, rd4]), 'two versions of the rule in one catalog')
    assert 'size-rule-1' in w and 'size-rule-3' in w, w
    _refuses(lambda: SZ.one_rule([rd4], want=rec), 'a size-rule-1 run where the configuration applies size-rule-3')
    assert SZ.same_rule(SZ.run_record(None), dict(SZ.run_record(None), version='size-rule-1')), 'production has no version'
    # a size-rule-2 record (3aa89e0, before full_band_size) is read as 'rule' - the size test in every band, what it applied -
    # and refused where the configuration or another attempt applies size-rule-3
    rd5 = os.path.join(T, 'att5', 'transients', 'Run_X')
    os.makedirs(rd5)
    json.dump({**{k: v for k, v in rec.items() if k != 'full_band_size'}, 'version': 'size-rule-2'},
              open(os.path.join(rd5, SZ.RUN_RECORD), 'w'))
    r5 = SZ.read_run_record(rd5)
    assert r5['full_band_size'] == 'rule' and r5['version'] == 'size-rule-2' and set(r5) == set(rec), r5
    _refuses(lambda: SZ.write_run_record(rd5, env), 'an append of size-rule-3 onto a size-rule-2 run')
    w = _refuses(lambda: SZ.one_rule([rd5], want=SZ.configured_record('deep')), 'size-rule-2 where size-rule-3 is configured')
    assert 'size-rule-2' in w and 'size-rule-3' in w, w
    for bad in (dict(rec, full_band_size='partly'),                                   # an unknown Full-band test
                {k: v for k, v in rec.items() if k != 'full_band_size'}):             # size-rule-3 without it
        json.dump(bad, open(os.path.join(rd5, SZ.RUN_RECORD), 'w'))
        _refuses(lambda: SZ.read_run_record(rd5), f'a malformed record {bad}')
    # one rule per catalog includes the deep Full band's size test: attempts with it off and on never mix, and attempts with it
    # on refuse where the configuration has it off
    rd6 = os.path.join(T, 'att6', 'transients', 'Run_X')
    os.makedirs(rd6)
    r6 = SZ.write_run_record(rd6, rule_env)
    assert r6['full_band_size'] == 'rule' and not SZ.same_rule(r6, rec), r6
    w = _refuses(lambda: SZ.one_rule([rd, rd6]), "attempts with the Full band's size test off and on in one catalog")
    assert SZ.FULL_BAND_WORDS['off'] in w and SZ.FULL_BAND_WORDS['rule'] in w, w
    _refuses(lambda: SZ.one_rule([rd6], want=SZ.configured_record('deep')), "attempts with it on, the configuration's off")
    assert SZ.configured_record('deep')['full_band_size'] == 'off' and SZ.one_rule([rd6], want=r6) == r6
    # the append check of the search step (one attempt, one rule)
    from exopipe.phase3.lib.search import deep_step
    assert deep_step.base_size_refusal(os.path.join(T, 'att1'), 'X') is None
    why = deep_step.base_size_refusal(os.path.join(T, 'att2'), 'X')
    assert why and 'search_detect.SIZE_RULE=production' in why and '--fresh' in why, why


def _axis_mode(mode):
    """Set catalog_conform.CONFORM_AXIS_UNDER_RULE for a test; returns the previous value (restore with it)."""
    C = _cfg()
    old = C.catalog_conform.CONFORM_AXIS_UNDER_RULE
    C.catalog_conform.CONFORM_AXIS_UNDER_RULE = mode
    return old


def _removed_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['cand_id', 'removal_reason']); w.writeheader(); w.writerows(rows)
    return path


SWITCH_LINE = "AXIS_STEP = os.environ.get('P3_CONFORM_AXIS', '1') != '0'\n"


def test_downstream():
    from exopipe.phase3.lib.search import removal_rules as RR
    C = _cfg()
    assert SZ.catalog_env(dict(SZ.run_record(None), rule='envelope')) == {'P3_SIZE_GATE_7_8': 'inf'}
    assert SZ.catalog_env(SZ.run_record(None)) == {}
    assert float(SZ.GATE_7_8_OFF) == float('inf') and not (5.0 > float(SZ.GATE_7_8_OFF)), 'the builder\'s gate never acts'
    AX = dict(cand_id='a', removal_reason='axis ratio 2.4 x beam > 2.0 above 8.0 sigma (Gregg 2026-09-05)')
    SZR = dict(cand_id='b', removal_reason='gaussfit size 2.70 beams > 2.5 above 8.0 sigma (Gregg 2026-09-04)')
    RL = dict(cand_id='c', removal_reason='row-line: >= 5 finals of this epoch on one pixel row (Gregg 2026-09-05)')
    rows = [AX, SZR, RL]
    assert [r['cand_id'] for r in SZ.fixed_rule_removals(rows)] == ['a', 'b']
    assert [r['cand_id'] for r in SZ.fixed_rule_removals(rows, axis_step=True)] == ['b'], 'the axis step as the second gate'
    os.makedirs(T, exist_ok=True)
    p = _removed_csv(os.path.join(T, 'master_catalog_removed.csv'), rows)
    why = SZ.fixed_removals_in([p], 'conform')
    assert why and why.startswith('conform: 2 row(s)') and 'REFUSING' in why, why
    assert SZ.fixed_removals_in([os.path.join(T, 'none.csv')]) is None
    # each row once (finding 9): the multiples fork keeps every conformance run's removals under its tag beside the last
    # run's master_catalog_removed.csv - the same rows in two or three files
    tags = [_removed_csv(os.path.join(T, f'master_catalog_removed_{t}.csv'), rows) for t in ('A', 'B')]
    why = SZ.fixed_removals_in([p] + tags, 'multiples fork V')
    assert why.startswith('multiples fork V: 2 row(s)'), why
    assert len(SZ.removed_rows([p] + tags)) == 3 and [f for f, _r in SZ.removed_rows([p] + tags)] == ['master_catalog_removed.csv'] * 3
    # the conformance's AXIS step under the rule (catalog_conform.CONFORM_AXIS_UNDER_RULE): RULED 'replace' (the rule alone,
    # R-REFACTOR-20261104-AXIS-RULE-ALONE-FULLBAND-SIZE-OFF-FROM-PULSARS, on Gregg's delegation R-GREGG-20261104-AXIS-THRESHOLD-
    # DECIDE-FROM-PULSARS); 'second_gate' is production's behavior, the parity file's value
    env_rec, prod_rec = dict(SZ.run_record(None), rule='envelope'), SZ.run_record(None)
    assert C.catalog_conform.CONFORM_AXIS_UNDER_RULE == 'replace' == SZ.conform_axis_mode(), 'the ruled default'
    assert C.ruled('catalog_conform.CONFORM_AXIS_UNDER_RULE') == (SZ.PULSAR_DECISION,) \
        and not C.protected('catalog_conform.CONFORM_AXIS_UNDER_RULE')
    assert SZ.PULSAR_DECISION == 'R-REFACTOR-20261104-AXIS-RULE-ALONE-FULLBAND-SIZE-OFF-FROM-PULSARS' \
        and SZ.PULSAR_DELEGATION == 'R-GREGG-20261104-AXIS-THRESHOLD-DECIDE-FROM-PULSARS'
    assert SZ.AXIS_STEP_ORIGIN in SZ.AXIS_STEP_BASIS['second_gate'] and SZ.AXIS_STEP_SUPERSEDED_BY in SZ.AXIS_STEP_BASIS['second_gate']
    assert "production's behavior" in SZ.AXIS_STEP_BASIS['second_gate'] and 'pending Gregg' not in ''.join(SZ.AXIS_STEP_BASIS.values())
    assert SZ.PULSAR_DECISION in SZ.AXIS_STEP_BASIS['replace'] and SZ.PULSAR_DELEGATION in SZ.AXIS_STEP_BASIS['replace']
    src_cfg = open(os.path.join(REPO, 'exopipe', 'phase3', 'config', 'steps', 'catalog.py')).read()
    assert 'Supersedes ... the 2.0 axis value of' in src_cfg and 'Approved - remove above 8' in src_cfg, 'both rulings quoted'
    assert 'You should decide, based on the data' in src_cfg, "Gregg's delegation quoted"
    # RULED: into a production --out an override needs --ruling-for (a fix-ledger row; refused without it), into a
    # non-production --out it is recorded as such
    from exopipe.phase3.config import ConfigRefusal
    auth, prod = dict(C._auth), os.path.join(C.paths.PRODUCTION_ROOTS[0], 'size_rule_test_run')
    try:
        C.authorize([], out=prod)
        try:
            C.set('catalog_conform.CONFORM_AXIS_UNDER_RULE', "'second_gate'")
            raise AssertionError('the ruled key into a production --out must refuse without --ruling-for')
        except ConfigRefusal as e_:
            assert SZ.PULSAR_DECISION in str(e_) and '--ruling-for' in str(e_), str(e_)
        assert C.catalog_conform.CONFORM_AXIS_UNDER_RULE == 'replace'
        C.authorize([f'catalog_conform.CONFORM_AXIS_UNDER_RULE={SZ.PULSAR_DECISION}'], out=prod)
        C.set('catalog_conform.CONFORM_AXIS_UNDER_RULE', "'second_gate'")
        assert C.catalog_conform.CONFORM_AXIS_UNDER_RULE == 'second_gate' \
            and C.overrides[-1][4] == f'ruling:{SZ.PULSAR_DECISION}', C.overrides[-1:]
        C.reset()
        C.authorize([], out=os.path.join(T, 'scratch_out'))
        C.set('catalog_conform.CONFORM_AXIS_UNDER_RULE', "'second_gate'")
        assert C.overrides[-1][4].startswith('non-production out:'), C.overrides[-1:]
    finally:
        C.reset()
        C._auth = auth
    assert C.catalog_conform.CONFORM_AXIS_UNDER_RULE == 'replace'
    _refuses(lambda: SZ.conform_axis_mode('second gate'), 'an unknown CONFORM_AXIS_UNDER_RULE')
    pax = _removed_csv(os.path.join(T, 'axis_only.csv'), [AX, RL])
    assert SZ.conform_removals([p], prod_rec) == (None, None), "production's attempts: its removals stand"
    assert SZ.conform_env(prod_rec) == {} and SZ.conform_axis_step(prod_rec) and SZ.conform_axis_step('production')
    # 'replace' (the default): P3_CONFORM_AXIS=0 to production's script; an axis removal means the switch did not act
    assert SZ.conform_env(env_rec) == {SZ.CONFORM_AXIS_ENV: '0'} and not SZ.conform_axis_step(env_rec)
    assert SZ.conform_env(prod_rec) == {}, "production's attempts keep the step"
    why, sg = SZ.conform_removals([pax], env_rec, 'S-G13 conformance')
    assert why and 'S-G13 conformance: 1 row(s)' in why and 'P3_CONFORM_AXIS=0' in why and why.endswith('REFUSING'), why
    why, sg = SZ.conform_removals([_removed_csv(os.path.join(T, 'rl.csv'), [RL])], env_rec)
    assert why is None and sg == dict(mode='replace', basis=SZ.AXIS_STEP_BASIS['replace'], rows=0, examples=[]), (why, sg)
    why, sg = SZ.conform_removals([p], env_rec, 'conform')
    assert why and 'conform: 2 row(s)' in why and sg is None, why          # the size step's removal and the axis step's
    old = _axis_mode('second_gate')
    try:                    # 'second_gate' (production's behavior): the axis step stands, its removals recorded, never refused;
        pax_a = _removed_csv(os.path.join(T, 'axis_only_A.csv'), [AX, RL])          # its tag copy: the row counted once
        why, sg = SZ.conform_removals([pax, pax_a], env_rec, 'multiples fork I')
        assert why is None and sg['mode'] == 'second_gate' and sg['basis'] == SZ.AXIS_STEP_BASIS['second_gate'] \
            and sg['rows'] == 1, (why, sg)
        assert SZ.conform_env(env_rec) == {} and SZ.conform_axis_step(env_rec)
        why, sg = SZ.conform_removals([p], env_rec, 'conform')                     # the size step still contradicts
        assert why and 'conform: 1 row(s)' in why and 'gaussfit size' in why and sg is None, why
        _axis_mode('bogus')
        _refuses(lambda: SZ.conform_env(env_rec), 'a configured CONFORM_AXIS_UNDER_RULE outside the two values')
    finally:
        _axis_mode(old)
    # the production switch (CONFORM_AXIS_PATCH): read from the script's source, default on; 'replace' refuses without it
    no_sw = os.path.join(T, 'p3s2_conform_today.py')
    open(no_sw, 'w').write("import os\nSIZE_STEP = os.environ.get('P3_CONFORM_SIZE', '1') != '0'\nAXIS_REMOVE = 2.0\n")
    sw = os.path.join(T, 'p3s2_conform_patched.py')
    open(sw, 'w').write("import os\nSIZE_STEP = os.environ.get('P3_CONFORM_SIZE', '1') != '0'\n" + SWITCH_LINE)
    off = os.path.join(T, 'p3s2_conform_default_off.py')
    open(off, 'w').write("import os\nAXIS_STEP = os.environ.get('P3_CONFORM_AXIS', '0') != '0'\n")
    assert SZ.conform_switch(sw) and not SZ.conform_switch(no_sw) and not SZ.conform_switch(off)
    w = SZ.conform_axis_refusal(env_rec, no_sw, 'multiples fork V')          # 'replace', the default, needs the switch
    assert w and w.startswith('multiples fork V:') and SZ.CONFORM_AXIS_PATCH in w and w.endswith('REFUSING'), w
    assert SZ.PULSAR_DECISION in w and 'CONFORM_AXIS_UNDER_RULE=second_gate' in w and '--ruling-for' in w, w
    assert SZ.conform_axis_refusal(env_rec, off) and SZ.conform_axis_refusal(env_rec, os.path.join(T, 'absent.py'))
    assert SZ.conform_axis_refusal(env_rec, sw) is None and SZ.conform_axis_refusal(prod_rec, no_sw) is None
    old = _axis_mode('second_gate')
    try:
        assert SZ.conform_axis_refusal(env_rec, no_sw) is None, "'second_gate' needs no switch"
    finally:
        _axis_mode(old)
    fin = [dict(final='1', size_verdict='point-like', size_excess='0.9', axis_excess='0.5'),
           dict(final='1', size_verdict='point-like', size_excess='0.9', axis_excess='1.01'),
           dict(final='1', size_verdict='gaussfit-size', size_excess='1.2', axis_excess='0.5'),
           dict(final='1', size_verdict='', size_excess='', axis_excess=''),
           dict(final='1', size_verdict='fit-failed', size_excess='', axis_excess=''),
           dict(final='0', size_verdict='axis-ratio', size_excess='0.5', axis_excess='1.3'),
           dict(final='1', size_verdict='point-like', size_excess='0.2', axis_excess='0.2', size_role='record')]
    got = [(fin.index(r), rule) for r, rule, _w in SZ.final_findings(fin)]
    assert got == [(1, 'axis'), (2, 'size'), (3, 'size'), (6, 'size')], got
    # ... in the deep Full band under FULL_BAND_SIZE 'off' (its size_role marked) a final passes by its axis alone - its size
    # excess recorded, not applied; the mark belongs to the Full band only and, with the run's record, where the record says
    FG = f'gate{SZ.ROLE_SEP}{SZ.FULL_OFF_NOTE}'
    full = [dict(final='1', scope='Full', size_verdict='point-like', size_excess='1.6', axis_excess='0.5', size_role=FG),   # A
            dict(final='1', scope='Full', size_verdict='point-like', size_excess='1.6', axis_excess='1.2', size_role=FG),   # B
            dict(final='1', scope='Full', size_verdict='point-like', size_excess='0.6', axis_excess='0.5', size_role='gate'),   # C
            dict(final='1', scope='Green', size_verdict='point-like', size_excess='0.6', axis_excess='0.5', size_role=FG),  # D
            dict(final='1', scope='Full', size_verdict='gaussfit-size', size_excess='1.6', axis_excess='0.5', size_role=FG),  # E
            dict(final='1', scope='Full', size_verdict='point-like', size_excess='0.6', axis_excess='0.5',
                 size_role=f'record{SZ.ROLE_SEP}{SZ.FULL_OFF_NOTE}')]                                                       # F
    for fb, want in ((None, [(1, 'axis'), (3, 'size'), (4, 'size'), (5, 'size')]),
                     ('off', [(1, 'axis'), (2, 'size'), (3, 'size'), (4, 'size'), (5, 'size')]),
                     ('rule', [(0, 'size'), (1, 'size'), (3, 'size'), (4, 'size'), (5, 'size')])):
        got = [(full.index(r), rule) for r, rule, _w in SZ.final_findings(full, full_band=fb)]
        assert got == want, (fb, got)
    assert SZ.final_findings([full[0]], full_band='off') == [] and SZ.final_findings([dict(full[0], size_role='gate')],
                                                                                    full_band='rule')[0][1] == 'size'
    # the conformance's size_axis (the deep check, the port): production removes above its fixed limits; under the rule the
    # size step flags, the axis step flags ('replace', the default) or removes as the second gate ('second_gate')
    r = dict(snr='9.0', size_beams='2.7', axis_norm='2.4', axis_ratio='', fwhm_maj_arcmin='')
    AXW = 'axis ratio 2.4 x beam > 2.0 above 8.0 sigma (Gregg 2026-09-05)'
    keep = RR._cc().SIZE_STEP
    try:
        RR._cc().SIZE_STEP = dict(keep, deep=True)
        cols, why = RR.size_axis(r)
        assert why.startswith('gaussfit size 2.70 beams > 2.5'), why
        cols, why = RR.size_axis(r, rule='envelope')
        assert why == '' and cols['size_flag'].startswith('size > 2.5') and cols['axis_flag'].startswith('axis > 2.0'), cols
        old = _axis_mode('second_gate')
        try:
            cols, why = RR.size_axis(r, rule='envelope')
            assert why == AXW and cols['size_flag'].startswith('size > 2.5 beams (kept'), (cols, why)
        finally:
            _axis_mode(old)
    finally:
        RR._cc().SIZE_STEP = keep
    cols, why = RR.size_axis(r)
    assert why == AXW, "production's attempts keep the step under 'replace'"
    cols, why = RR.size_axis(r, rule='envelope')
    assert why == '' and cols['axis_norm'] == 2.4 and cols['axis_flag'] == ('axis > 2.0 x beam (kept: point-like at its '
                                                                           'S/N by the size / axis rule)'), (cols, why)
    cols, why = RR.size_axis(dict(r, axis_norm='1.7'), rule='envelope')
    assert why == '' and cols['axis_flag'] == 'axis 1.5-2.0 x beam', cols
    old = _axis_mode('second_gate')
    try:
        cols, why = RR.size_axis(r)
        assert why == AXW, why
        cols, why = RR.size_axis(r, rule='envelope')
        assert why == AXW and cols['axis_norm'] == 2.4, (cols, why)
    finally:
        _axis_mode(old)
    try:
        RR.size_axis(r, rule='fixed')
        raise AssertionError('an unknown rule must refuse')
    except RuntimeError:
        pass
    # the port's environment check (finding 8): P3_CONFORM_AXIS must be what the axis step asked for
    saved = {k: os.environ.get(k) for k in ('P3_CONFORM_AXIS',)}
    try:
        os.environ.pop('P3_CONFORM_AXIS', None)
        RR._refuse_other_modes(False, True)
        RR._refuse_other_modes(False, False)
        os.environ['P3_CONFORM_AXIS'] = '0'
        RR._refuse_other_modes(False, False)
        try:
            RR._refuse_other_modes(False, True)
            raise AssertionError("P3_CONFORM_AXIS=0 with the axis step on must refuse")
        except SystemExit as e_:
            assert 'P3_CONFORM_AXIS' in str(e_), str(e_)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    src = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'removal_rules.py')).read()
    assert "rule = _SZ.ledger_rule(f'{U}/v_deep_ledger.csv')['rule']" in src and 'size_step=size_step, rule=rule)' in src, \
        "the port judges by the merged attempts' rule"
    # the per-instant content test still reads its two size templates whole from size_axis (one f-string each)
    from exopipe.phase3.lib.catalog import tests_per_instant as TPI
    assert TPI.reason_kind('axis ratio 2.4 x beam > 2.0 above 8.0 sigma (Gregg 2026-09-05)') == ('axis ratio', 'own')
    assert TPI.reason_kind('gaussfit size 2.70 beams > 2.5 above 8.0 sigma (Gregg 2026-09-04)') == ('size', 'own')


def test_ledger():
    from exopipe.phase3.lib.search import deep_search as DS, tests as ST
    assert ST.GLED_COLS_SIZE == SZ.GLED_COLS_SIZE == DS.GLED_COLS_SIZE, 'one column list'
    assert ST.GLED_COLS_SIZE_V1 == SZ.GLED_COLS_SIZE_V1 == SZ.GLED_COLS_SIZE[:7] and SZ.GLED_COLS_SIZE[-1] == 'size_role'
    env = SZ.Envelope.load(DEEP).require()
    shutil.rmtree(T, ignore_errors=True)
    rd = os.path.join(T, 'att', 'transients', 'Run_X')
    os.makedirs(rd)
    SZ.write_run_record(rd, env)
    cols = DS.GLED_COLS + DS.GLED_COLS_SIZE

    def row(**kw):
        r = dict.fromkeys(cols, '')
        r.update(lst='05h', scope='23MHz', stokes='V', date='2025-01-01', ra='10.0', dec='20.0', snr='7.5', mstruct='1.0',
                 final='0', size_rule=env.label(), size_snr='7.5', size_thr='2.99', axis_thr='4.11', size_excess='0.4',
                 axis_excess='0.3', size_verdict='point-like', size_role='gate')
        r.update(kw)
        return r

    def write(rows, header=cols, tier='deep', rdir=None):
        p = os.path.join(rdir or rd, 'v_deep_ledger.csv')
        with open(p, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
        return ST.size_rule_rows(p, header, [dict((k, r.get(k, '')) for k in header) for r in rows], tier=tier)
    unmeasured = dict(size_snr='', size_thr='', axis_thr='', size_excess='', axis_excess='', size_verdict='fit-failed')
    ok = [row(final='1'), row(veto_reason='mstruct'), row(veto_reason='gaussfit-size', size_verdict='gaussfit-size',
                                                         size_excess='1.3'),
          row(veto_reason='axis-ratio', size_verdict='axis-ratio', axis_excess='1.1'),
          row(stokes='I', veto_reason='reject_local_snr', snr='6.0'),                      # the fork's member pool, kept
          row(stokes='I', veto_reason='reject_local_snr:gaussfit-size', snr='5.0', size_verdict='gaussfit-size', size_excess='1.4'),
          row(stokes='I', veto_reason='reject_local_snr:axis-ratio', snr='-6.99', size_verdict='axis-ratio', axis_excess='1.2'),
          row(stokes='I', veto_reason='reject_local_snr', snr='4.99', size_role='record', size_verdict='gaussfit-size',
              size_excess='1.6'),                                                          # below the pool: recorded only
          row(stokes='V', veto_reason='reject_local_snr', snr='6.0', size_role='record'),     # Stokes V: no member pool
          row(stokes='I', veto_reason='mstruct_I', size_role='record', size_verdict='gaussfit-size', size_excess='1.3'),
          row(veto_reason='reject_merged', size_role='record', **unmeasured)]
    assert write(ok) == [], write(ok)
    bad = [([row(veto_reason='gaussfit-size')], "a 'gaussfit-size' row the rule did not remove"),
           ([row(veto_reason='mstruct', size_verdict='axis-ratio')], 'the rule removed it, the ledger says mstruct'),
           ([row(final='1', axis_excess='1.2')], 'a final the rule removes'),
           ([row(final='1', size_verdict='')], 'a final the rule never judged'),
           ([row(final='1', size_rule='envelope:other.json:0123')], 'another product than the record names'),
           ([row(veto_reason='reject_merged', size_role='record', size_verdict='')], 'a candidate row neither judged nor recorded'),
           ([row(veto_reason='mstruct', size_role='')], 'a row without its role'),
           ([row(stokes='I', veto_reason='reject_local_snr', snr='4.2')], 'a local-SNR row below the pool judged as the pool'),
           ([row(stokes='I', veto_reason='reject_local_snr:gaussfit-size', snr='4.2', size_verdict='gaussfit-size',
                 size_excess='1.4')], 'a floor removal attributed to the size rule (below the member window)'),
           ([row(stokes='I', veto_reason='reject_local_snr', snr='6.0', size_role='record')], 'a pool row only recorded'),
           ([row(stokes='I', veto_reason='reject_local_snr', snr='7.0')], 'the window is lo <= |S/N| < hi: 7.0 is outside'),
           ([row(final='1', size_role='record')], "a final only 'recorded'"),
           ([row(veto_reason='gaussfit-size', size_verdict='gaussfit-size', size_role='record')], "a 'record' row the rule removed"),
           ([row(veto_reason='', final='1', size_role='record')], "a 'record' row without an earlier gate"),
           ([row(stokes='I', veto_reason='reject_local_snr:gaussfit-size')], 'a member-pool removal the rule did not make'),
           ([row(stokes='V', veto_reason='reject_local_snr:gaussfit-size', size_verdict='gaussfit-size')],
            'a member-pool reason on a Stokes V row'),
           ([row(stokes='I', veto_reason='reject_local_snr', size_verdict='gaussfit-size')],
            "the rule removes a member-pool row whose reason stays 'reject_local_snr'")]
    for rows_, what in bad:
        assert write(rows_), what
    # the 10-minute tier has no member pool: every local-SNR row is recorded; one judged as the pool refuses
    assert write([row(stokes='I', veto_reason='reject_local_snr', snr='6.0', size_role='record')], tier='sublst') == []
    assert write([row(stokes='I', veto_reason='reject_local_snr', snr='6.0')], tier='sublst'), '10 minutes: no member pool'
    assert write([row(stokes='I', veto_reason='reject_local_snr:axis-ratio', snr='6.0', size_verdict='axis-ratio',
                      axis_excess='1.2')], tier='sublst'), '10 minutes: no member-pool removal'
    assert write([row(final='1')], header=DS.GLED_COLS), "the record says envelope, the header lacks the rule's columns"
    # the deep Full band (R-REFACTOR-20261104-AXIS-RULE-ALONE-FULLBAND-SIZE-OFF-FROM-PULSARS): under this run's record
    # (full_band_size 'rule') its rows are as every band's; under a record of 'off' EVERY row of it carries the mark, no other
    # row does, and none is removed by size (its size excess recorded, never applied; the axis test applies)
    FG, FR = f'gate{SZ.ROLE_SEP}{SZ.FULL_OFF_NOTE}', f'record{SZ.ROLE_SEP}{SZ.FULL_OFF_NOTE}'
    assert SZ.read_run_record(rd)['full_band_size'] == 'rule'
    assert write([row(scope='Full', final='1'), row(scope='Full', veto_reason='gaussfit-size', size_verdict='gaussfit-size',
                                                    size_excess='1.3')]) == []
    assert write([row(scope='Full', final='1', size_role=FG)]), "the mark where the run's record says 'rule'"
    assert write([row(scope='Full', final='1', size_excess='1.2')]), "a Full-band final above its size limit under 'rule'"
    rdf = os.path.join(T, 'attf', 'transients', 'Run_X')
    os.makedirs(rdf)
    assert SZ.write_run_record(rdf, SZ.Envelope.load(DEEP, full_band='off').require())['full_band_size'] == 'off'
    okf = [row(scope='Full', final='1', size_role=FG, size_excess='1.6'),                # kept by size: excess recorded
           row(scope='Full', veto_reason='axis-ratio', size_verdict='axis-ratio', size_excess='1.4', axis_excess='1.2',
               size_role=FG),                                                            # removed by the axis test
           row(scope='Full', stokes='I', veto_reason='reject_local_snr', snr='6.0', size_excess='1.5', size_role=FG),   # the pool
           row(scope='Full', stokes='I', veto_reason='reject_local_snr:axis-ratio', snr='6.0', size_verdict='axis-ratio',
               axis_excess='1.3', size_role=FG),
           row(scope='Full', stokes='I', veto_reason='reject_local_snr', snr='4.2', size_role=FR),                   # below it
           row(scope='Full', veto_reason='mstruct', size_role=FR, size_excess='2.0'),
           row(final='1'), row(veto_reason='gaussfit-size', size_verdict='gaussfit-size', size_excess='1.3')]       # 23 MHz
    assert write(okf, rdir=rdf) == [], write(okf, rdir=rdf)
    badf = [([row(scope='Full', final='1')], "a Full-band row without the mark under 'off'"),
            ([row(scope='Full', veto_reason='mstruct', size_role='record')], "a recorded Full-band row without the mark"),
            ([row(scope='Full', veto_reason='gaussfit-size', size_verdict='gaussfit-size', size_excess='1.3', size_role=FG)],
             "a Full-band removal by size under 'off'"),
            ([row(scope='Full', stokes='I', veto_reason='reject_local_snr:gaussfit-size', snr='6.0', size_verdict='gaussfit-size',
                  size_excess='1.4', size_role=FG)], "a Full-band member-pool removal by size under 'off'"),
            ([row(final='1', size_role=FG)], 'the mark on a subband row'),
            ([row(scope='Green', final='1', size_role=FG)], 'the mark on a wideband row other than Full'),
            ([row(scope='Full', final='1', size_role=FG, axis_excess='1.1')], 'a marked Full-band final above its axis limit'),
            ([row(scope='Full', final='1', size_role='gate:another note')], 'another mark'),
            ([row(scope='Full', final='1', size_role=FR)], "a marked Full-band final only 'recorded'")]
    for rows_, what in badf:
        assert write(rows_, rdir=rdf), what
    assert write([row(scope='Full', final='1', size_role=FG)], tier='sublst', rdir=rdf), 'the 10-minute tier: no Full-band mark'
    # a size-rule-2 run (3aa89e0: its record without full_band_size) is read by its own semantics - the size test in every band
    rd2 = os.path.join(T, 'att2', 'transients', 'Run_X')
    os.makedirs(rd2)
    json.dump({**{k: v for k, v in SZ.run_record(env).items() if k != 'full_band_size'}, 'version': 'size-rule-2'},
              open(os.path.join(rd2, SZ.RUN_RECORD), 'w'))
    assert write([row(scope='Full', final='1'), row(veto_reason='gaussfit-size', size_verdict='gaussfit-size', size_excess='1.3',
                                                    scope='Full')], rdir=rd2) == []
    assert write([row(scope='Full', final='1', size_role=FG)], rdir=rd2), 'size-rule-2 never marked a row'
    json.dump(dict(SZ.read_run_record(rd), version='size-rule-1'), open(os.path.join(rd, SZ.RUN_RECORD), 'w'))
    assert write([row(final='1')]), "size-rule-3 columns, the record says size-rule-1"
    v1 = DS.GLED_COLS + SZ.GLED_COLS_SIZE_V1                  # a size-rule-1 ledger is read by its own checks (judged rows only)
    assert write([row(final='1'), row(veto_reason='reject_local_snr', size_verdict='', size_rule='')], header=v1) == []
    assert write([row(veto_reason='gaussfit-size')], header=v1), 'size-rule-1: a removal the rule did not make'
    assert ST.complete.__doc__ and 'every row' in ST.complete.__doc__
    os.remove(os.path.join(rd, SZ.RUN_RECORD))
    assert write([row(final='1')]), "the header carries the rule's columns, no record"
    assert write([row(final='1')], header=DS.GLED_COLS) == [], 'production: nothing to check'


def test_catalog_paths():
    """The 10-minute catalog reads its search jobs' records (one rule, the configured one: a mix or another refuses, S-G7);
    the incremental deep path reads the merged ledger's header; both give the builder P3_SIZE_GATE_7_8=inf under the rule,
    refuse production's fixed size removals and treat its axis step per catalog_conform.CONFORM_AXIS_UNDER_RULE."""
    import types
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.catalog import tier_sublst as CS, incremental as INC
    from exopipe.phase3.lib.search import deep_search as DS
    env = SZ.Envelope.load(SUBLST).require()
    shutil.rmtree(T, ignore_errors=True)
    src = os.path.join(T, 'search_B')
    for job in ('14h_18MHz_I', '14h_18MHz_V', '14h_Red_I'):
        rd = os.path.join(src, job, 'transients', 'Run_SLST14B')
        os.makedirs(rd)
        SZ.write_run_record(rd, env)                      # the engine records its rule before it writes the ledger
        open(os.path.join(rd, 'v_deep_ledger.csv'), 'w').write(','.join(DS.GLED_COLS + DS.GLED_COLS_SIZE) + '\n')
    os.makedirs(os.path.join(src, '14h_23MHz_I', 'transients', 'Run_SLST14B'))      # no ledger: not a completed search job

    def ctx():
        o = types.SimpleNamespace(src=src, jobs_src=src, lst='14h', label='SLST14B', _size=None)   # jobs_src: the search jobs
                                                                                    # (= src except in a per-hour night pass)
        o.search_jobs = lambda: CS.Ctx.search_jobs(o)
        return o
    o = ctx()
    assert len(o.search_jobs()) == 3 and CS.Ctx.size_rule(o)['md5'] == env.md5, o.search_jobs()
    assert SZ.catalog_env(CS.Ctx.size_rule(o)) == {'P3_SIZE_GATE_7_8': 'inf'}
    os.remove(os.path.join(src, '14h_Red_I', 'transients', 'Run_SLST14B', SZ.RUN_RECORD))   # one job of production's
    try:
        CS.Ctx.size_rule(ctx())
        raise AssertionError('jobs of two rules must refuse')
    except Refusal as e:
        assert 'S-G7' in str(e) and 'size rules' in str(e), str(e)
    for job in ('14h_18MHz_I', '14h_18MHz_V'):                # every job production's, the configuration the rule's (finding 5:
        os.remove(os.path.join(src, job, 'transients', 'Run_SLST14B', SZ.RUN_RECORD))   # as the deep check, never silently)
    try:
        CS.Ctx.size_rule(ctx())
        raise AssertionError("production's jobs under the configured rule must refuse")
    except Refusal as e:
        assert 'S-G7' in str(e) and 'the configuration applies envelope' in str(e), str(e)
    C = _cfg()
    keep = C.search_sublst.SIZE_RULE
    try:                                                      # ... configured as production (the parity file): theirs
        C.search_sublst.SIZE_RULE = 'production'
        assert CS.Ctx.size_rule(ctx())['rule'] == 'production' and SZ.catalog_env(CS.Ctx.size_rule(ctx())) == {}
    finally:
        C.search_sublst.SIZE_RULE = keep
    U = os.path.join(T, 'U')
    os.makedirs(U)
    ic = types.SimpleNamespace(out=U, label='P3S5', parents='')
    open(os.path.join(U, 'v_deep_ledger.csv'), 'w').write(','.join(DS.GLED_COLS) + '\n')
    assert 'P3_SIZE_GATE_7_8' not in INC.stage_env(ic, 'build'), 'production: the builder keeps its 1.7'
    open(os.path.join(U, 'v_deep_ledger.csv'), 'w').write(','.join(DS.GLED_COLS + DS.GLED_COLS_SIZE) + '\n')
    assert INC.stage_env(ic, 'build')['P3_SIZE_GATE_7_8'] == 'inf' and INC.stage_env(ic, 'fork')['P3_SIZE_GATE_7_8'] == 'inf'
    assert 'P3_SIZE_GATE_7_8' not in INC.stage_env(ic, 'conform')
    rm = _removed_csv(os.path.join(U, 'master_catalog_removed.csv'),
                      [dict(cand_id='I1', removal_reason='axis ratio 2.4 x beam > 2.0 above 8.0 sigma (Gregg 2026-09-05)')])
    rs = _removed_csv(os.path.join(U, 'size_removed.csv'),
                      [dict(cand_id='I2', removal_reason='gaussfit size 2.70 beams > 2.5 above 8.0 sigma (Gregg 2026-09-04)')])
    unit = types.SimpleNamespace(meta={}, notes=[])
    unit.note = unit.notes.append
    try:
        INC._fixed_size_refusal(U, [rs], 'conform')
        raise AssertionError("production's fixed size removal under the rule must refuse")
    except Refusal as e:
        assert 'conform: 1 row(s)' in str(e), str(e)
    real = INC._ops(INC.C().STAGE_SCRIPTS['conform'])
    # 'replace' (the ruled default): P3_CONFORM_AXIS=0 to the conform and fork stages; the production script's switch required
    # before the stage runs; an axis-step removal refuses
    assert INC.stage_env(ic, 'conform')['P3_CONFORM_AXIS'] == '0' == INC.stage_env(ic, 'fork')['P3_CONFORM_AXIS']
    assert 'P3_CONFORM_AXIS' not in INC.stage_env(ic, 'build')
    try:
        INC._axis_switch(U, 'conform')
        assert SZ.conform_switch(real), 'without the switch the replayed conform must refuse'
    except Refusal as e:
        assert not SZ.conform_switch(real) and SZ.CONFORM_AXIS_PATCH in str(e), str(e)
    try:
        INC._fixed_size_refusal(U, [rm], 'conform')
        raise AssertionError("under 'replace' an axis-step removal must refuse")
    except Refusal as e:
        assert 'P3_CONFORM_AXIS=0' in str(e), str(e)
    old = _axis_mode('second_gate')
    try:                                                    # 'second_gate' (production's behavior): recorded, not refused
        INC._fixed_size_refusal(U, [rm], 'conform', unit)
        assert unit.meta['size_rule_second_gate']['conform']['rows'] == 1 and 'second gate' in unit.notes[0], \
            (unit.meta, unit.notes)
        assert INC.stage_env(ic, 'conform').get('P3_CONFORM_AXIS') is None and INC._axis_switch(U, 'conform') is None
    finally:
        _axis_mode(old)
    open(os.path.join(U, 'v_deep_ledger.csv'), 'w').write(','.join(DS.GLED_COLS) + '\n')
    INC._fixed_size_refusal(U, [rm, rs], 'conform')        # production's attempts: its own removals stand, under 'replace' too
    assert 'P3_CONFORM_AXIS' not in INC.stage_env(ic, 'conform') and INC._axis_switch(U, 'conform') is None


PROD_CONFORM = '/lustre/gh/main/exopipe_ops/p3s2_conform.py'        # read only


def test_patch():
    """The pending production patch (size_rule.CONFORM_AXIS_PATCH): it applies cleanly to a scratch copy of production's
    p3s2_conform.py as it is today (git apply; production itself never written), the result carries the switch the 'replace'
    stages require (size_rule.conform_switch, default on) and differs from production in the patch's lines only; the switch's
    reading is production's own idiom (P3_CONFORM_SIZE). Byte-identity of the outputs with the switch unset, and the rows it
    keeps when off: tests/test_removal_rules.py test_pending_patch (Run_P3S5 cells, lwacalim07). Once production carries it,
    the check is that it does."""
    import re
    import subprocess
    patch = os.path.join(REPO, SZ.CONFORM_AXIS_PATCH)
    text = open(patch).read()
    assert 'R-AXIS-RATIO-REMOVAL-ABOVE-8SIGMA' in text and 'multiples_fork.py' in text and 'git apply' in text, 'its preamble'
    assert SZ.PULSAR_DECISION in text and SZ.PULSAR_DELEGATION in text, 'the preamble names the decision and the delegation'
    if SZ.conform_switch(PROD_CONFORM):
        print('  (production p3s2_conform.py already carries the switch: nothing to apply)')
        return
    d = os.path.join(T, 'patch')
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    shutil.copy(PROD_CONFORM, os.path.join(d, 'p3s2_conform.py'))
    r = subprocess.run(['git', 'apply', '--check', patch], cwd=d, capture_output=True, text=True)
    assert r.returncode == 0, f'the pending patch no longer applies to production\'s p3s2_conform.py (re-base it): {r.stderr}'
    subprocess.run(['git', 'apply', patch], cwd=d, check=True, capture_output=True)
    new = os.path.join(d, 'p3s2_conform.py')
    assert SZ.conform_switch(new) and not SZ.conform_switch(PROD_CONFORM), 'the patched copy carries the switch, default on'
    a, b = open(PROD_CONFORM).read().splitlines(), open(new).read().splitlines()
    import difflib
    gone = [x[1:] for x in difflib.unified_diff(a, b, lineterm='', n=0) if x.startswith('-') and not x.startswith('---')]
    added = [x[1:] for x in difflib.unified_diff(a, b, lineterm='', n=0) if x.startswith('+') and not x.startswith('+++')]
    assert gone == ['n_axis = n_aflag = 0', '        if r[\'axis_norm\'] > AXIS_REMOVE:', '        if r[\'axis_norm\'] > AXIS_FLAG:'], gone
    assert SWITCH_LINE.strip() in added and len(added) == 11, added
    m = re.search(r'patched md5 ([0-9a-f]{32})', text)
    assert m and SZ.md5_file(new) == m.group(1), ('the preamble\'s patched md5', m and m.group(1), SZ.md5_file(new))
    import ast as _ast
    _ast.parse(open(new).read(), new)                       # it compiles


def test_carry():
    got = SZ.carry_instant([('J1', 'n1', '50MHz', 'point-like'), ('J1', 'n1', '50MHz', 'gaussfit-size'),
                            ('J1', 'n2', '50MHz', ''), ('J2', 'n1', '50MHz', 'point-like')])
    assert got == {('J1', 'n1')}, got
    # nothing carries from the deep Full band's size test (R-REFACTOR-20261104-AXIS-RULE-ALONE-FULLBAND-SIZE-OFF-FROM-PULSARS):
    # under FULL_BAND_SIZE 'off' the band has no size verdict, and under 'rule' its size removals stay its own - they never take
    # the subbands' detections of the instant; its axis removals, and every other band's, carry as the per-instant rule says
    got = SZ.carry_instant([('J1', 'n1', 'Full', 'gaussfit-size'), ('J1', 'n1', '41MHz', 'point-like'),
                            ('J2', 'n1', 'Full', 'axis-ratio'), ('J3', 'n1', 'Green', 'gaussfit-size'),
                            ('J4', 'n1', 'Red', 'gaussfit-size'), ('J5', 'n1', 'Full', 'point-like')])
    assert got == {('J2', 'n1'), ('J3', 'n1'), ('J4', 'n1')}, got
    assert not SZ.carries('Full', 'gaussfit-size') and SZ.carries('Full', 'axis-ratio') and SZ.carries('82MHz', 'gaussfit-size')
    assert not SZ.carries('Full', 'point-like') and not SZ.carries('82MHz', '') and not SZ.carries('82MHz', 'fit-failed')
    # ... and the rule's wiring holds no other carry: carry_instant / carries are called nowhere (the deferred carry), and the
    # engine judges one band per run (run_deep_cell's cell is one (LST, band)); no module but size_rule reads another band's
    # size verdict
    import ast as _ast
    calls = []
    for dp, dn, fn in os.walk(os.path.join(REPO, 'exopipe')):
        dn[:] = [x for x in dn if x != '__pycache__']
        for f in fn:
            if f.endswith('.py') and f != 'size_rule.py':
                src = open(os.path.join(dp, f)).read()
                for n in _ast.walk(_ast.parse(src)):
                    if ((isinstance(n, _ast.Attribute) and n.attr in ('carry_instant', 'carries'))
                            or (isinstance(n, _ast.ImportFrom) and {a.name for a in n.names} & {'carry_instant', 'carries'})):
                        calls.append(f'{os.path.relpath(os.path.join(dp, f), REPO)}:{n.lineno}')
    assert not calls, (f'the per-instant carry is wired ({calls}): check that it never carries the deep Full band\'s size '
                       f'verdicts (size_rule.carries) and update this test')


TESTS = [test_measure, test_products, test_snr, test_floor, test_verdict, test_injection, test_domain, test_annulus, test_stamp,
         test_engine, test_early, test_plumbing, test_records, test_downstream, test_ledger, test_catalog_paths, test_patch,
         test_carry]


def main(argv):
    names = set(argv[1:])
    fails = 0
    from exopipe.phase3.core import env as _env
    _env.apply()
    _env.legacy()
    try:
        for t in TESTS:
            if names and t.__name__ not in names:
                continue
            t1 = time.time()
            try:
                t()
                print(f'PASS {t.__name__} ({time.time() - t1:.1f} s)', flush=True)
            except Exception:
                fails += 1
                print(f'FAIL {t.__name__}\n{traceback.format_exc()}', flush=True)
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print('%d of %d passed' % (len([t for t in TESTS if not names or t.__name__ in names]) - fails,
                               len([t for t in TESTS if not names or t.__name__ in names])))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
