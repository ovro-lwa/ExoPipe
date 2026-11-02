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
  verdict      the rule at high S/N (the floors) and at low S/N (the envelope); |S/N|; no fit / no S/N keeps
  injection    end to end on beam-correlated noise: the fixed 1.5 / 1.5 rule loses 7-sigma point sources, the envelope rule
               built from the same injections keeps >= 97.5% at every S/N
  domain       injections only inside the search's domain; isolated zeros allowed, > 10% refused
  annulus      the S/N's noise is measured outside the fitted source (an extended source keeps its S/N)
  stamp        the frame stamp and its edge fallback (6 beams), off the frame: no measurement
  engine       engine_test (deep_search._size_test's rule branch): columns recorded; a 4-beam source at S/N 7.6 removed that
               the fixed rule (above 8 sigma only) keeps; Stokes V fitted whole (production: the PyBDSF component); a point
               source kept; production's record-only flags above 8 sigma; an epoch without a beam refuses
  plumbing     the rule's values and environment (unknown rule, missing md5 refuse; production sets only the rule), the config
               keys and pins, the parity file's production values, the steps' environments
  records      the run's size_rule.json: written under the rule only, one rule per run (an append under another refuses),
               one rule per catalogue (a mix refuses; another than configured refuses), read from a ledger header
  downstream   the catalogue builder's 7-8 sigma gate replaced (P3_SIZE_GATE_7_8 = inf), production's fixed conformance
               removals flagged as contradictions, the conformance's size_axis keeps under the rule, final rows' findings
  ledger       the gate-ledger content test: GLED_COLS_SIZE accepted with the run's record, the rule's verdicts and reasons agree
  catalog      the 10-minute catalogue's search jobs (one rule, a mix refuses), the incremental deep path's ledger header: the
               builder's 7-8 sigma gate replaced under the rule, production's fixed removals refused
  carry        the per-instant carry stays within its instant
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
    j = SZ.judge(env, 'I', '18MHz', rec)
    assert j['size_verdict'] == 'gaussfit-size' and j['size_excess'] > 1 and j['size_thr'] == 1.5, j
    assert SZ.judge(env, 'I', '18MHz', None)['size_verdict'] == 'fit-failed'
    assert SZ.judge(env, 'I', '18MHz', dict(rec, snr=float('nan')))['size_verdict'] == 'no-snr'
    assert SZ.judge(env, 'I', '18MHz', dict(rec, eff=1.0))['size_verdict'] == 'point-like'


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
    assert v == 'remove' and c['size_verdict'] == 'gaussfit-size' and c['size_excess'] > 1, c
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
    src = open(os.path.join(REPO, 'exopipe', 'phase3', 'lib', 'search', 'deep_search.py')).read()
    assert src.count('rule=_szr, band=scope)   # WAVE2 SIZE-RULE') == 2, 'both Stokes call the rule'
    assert '_szr = _SZ.engine_rule(cadence)' in src and '_SZ.write_run_record(run_root, _szr)' in src


def test_records():
    shutil.rmtree(T, ignore_errors=True)
    env = SZ.Envelope.load(DEEP).require()
    rd = os.path.join(T, 'att1', 'transients', 'Run_X')
    os.makedirs(rd)
    assert SZ.write_run_record(rd, None)['rule'] == 'production' and not os.path.exists(os.path.join(rd, SZ.RUN_RECORD)), \
        'production writes nothing'
    rec = SZ.write_run_record(rd, env)
    assert rec == SZ.read_run_record(rd) and rec['rule'] == 'envelope' and rec['md5'] == env.md5 and rec['ruling'] == SZ.RULING
    assert SZ.write_run_record(rd, env) == rec, 'an append under the same rule'
    _refuses(lambda: SZ.write_run_record(rd, None), 'production into a run searched under the rule')
    other = SZ.Envelope(dict(env.cal), source=SUBLST, md5=SZ.md5_file(SUBLST))
    _refuses(lambda: SZ.write_run_record(rd, other), 'another product into the run')
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
    # the append check of the search step (one attempt, one rule)
    from exopipe.phase3.lib.search import deep_step
    assert deep_step.base_size_refusal(os.path.join(T, 'att1'), 'X') is None
    why = deep_step.base_size_refusal(os.path.join(T, 'att2'), 'X')
    assert why and 'search_detect.SIZE_RULE=production' in why and '--fresh' in why, why


def test_downstream():
    from exopipe.phase3.lib.search import removal_rules as RR
    assert SZ.catalog_env(dict(SZ.run_record(None), rule='envelope')) == {'P3_SIZE_GATE_7_8': 'inf'}
    assert SZ.catalog_env(SZ.run_record(None)) == {}
    assert float(SZ.GATE_7_8_OFF) == float('inf') and not (5.0 > float(SZ.GATE_7_8_OFF)), 'the builder\'s gate never acts'
    rows = [dict(cand_id='a', removal_reason='axis ratio 2.4 x beam > 2.0 above 8.0 sigma (Gregg 2026-09-05)'),
            dict(cand_id='b', removal_reason='gaussfit size 2.70 beams > 2.5 above 8.0 sigma (Gregg 2026-09-04)'),
            dict(cand_id='c', removal_reason='row-line: >= 5 finals of this epoch on one pixel row (Gregg 2026-09-05)')]
    assert [r['cand_id'] for r in SZ.fixed_rule_removals(rows)] == ['a', 'b']
    os.makedirs(T, exist_ok=True)
    p = os.path.join(T, 'master_catalog_removed.csv')
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['cand_id', 'removal_reason']); w.writeheader(); w.writerows(rows)
    why = SZ.fixed_removals_in([p], 'conform')
    assert why and why.startswith('conform: 2 row(s)') and 'REFUSING' in why, why
    assert SZ.fixed_removals_in([os.path.join(T, 'none.csv')]) is None
    fin = [dict(final='1', size_verdict='point-like', size_excess='0.9', axis_excess='0.5'),
           dict(final='1', size_verdict='point-like', size_excess='0.9', axis_excess='1.01'),
           dict(final='1', size_verdict='gaussfit-size', size_excess='1.2', axis_excess='0.5'),
           dict(final='1', size_verdict='', size_excess='', axis_excess=''),
           dict(final='1', size_verdict='fit-failed', size_excess='', axis_excess=''),
           dict(final='0', size_verdict='axis-ratio', size_excess='0.5', axis_excess='1.3')]
    got = [(fin.index(r), rule) for r, rule, _w in SZ.final_findings(fin)]
    assert got == [(1, 'axis'), (2, 'size'), (3, 'size')], got
    # the conformance's size_axis: production removes above its fixed limits; under the rule it keeps (and flags)
    r = dict(snr='9.0', size_beams='2.7', axis_norm='2.4', axis_ratio='', fwhm_maj_arcmin='')
    keep = RR._cc().SIZE_STEP
    try:
        RR._cc().SIZE_STEP = dict(keep, deep=True)
        cols, why = RR.size_axis(r)
        assert why.startswith('gaussfit size 2.70 beams > 2.5'), why
        cols, why = RR.size_axis(r, rule='envelope')
        assert why == '' and cols['size_flag'].startswith('size > 2.5') and cols['axis_flag'].startswith('axis > 2.0'), cols
    finally:
        RR._cc().SIZE_STEP = keep
    cols, why = RR.size_axis(r)
    assert why.startswith('axis ratio 2.4 x beam > 2.0 above 8.0 sigma'), why
    cols, why = RR.size_axis(r, rule='envelope')
    assert why == '' and cols['axis_norm'] == 2.4, (cols, why)
    try:
        RR.size_axis(r, rule='fixed')
        raise AssertionError('an unknown rule must refuse')
    except RuntimeError:
        pass
    # the per-instant content test still reads its two size templates whole from size_axis (one f-string each)
    from exopipe.phase3.lib.catalog import tests_per_instant as TPI
    assert TPI.reason_kind('axis ratio 2.4 x beam > 2.0 above 8.0 sigma (Gregg 2026-09-05)') == ('axis ratio', 'own')
    assert TPI.reason_kind('gaussfit size 2.70 beams > 2.5 above 8.0 sigma (Gregg 2026-09-04)') == ('size', 'own')


def test_ledger():
    from exopipe.phase3.lib.search import deep_search as DS, tests as ST
    assert ST.GLED_COLS_SIZE == SZ.GLED_COLS_SIZE == DS.GLED_COLS_SIZE, 'one column list'
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
                 axis_excess='0.3', size_verdict='point-like')
        r.update(kw)
        return r

    def write(rows, header=cols):
        p = os.path.join(rd, 'v_deep_ledger.csv')
        with open(p, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
        return ST.size_rule_rows(p, header, [dict((k, r.get(k, '')) for k in header) for r in rows])
    ok = [row(final='1'), row(veto_reason='mstruct'), row(veto_reason='gaussfit-size', size_verdict='gaussfit-size',
                                                         size_excess='1.3'),
          row(veto_reason='axis-ratio', size_verdict='axis-ratio', axis_excess='1.1'), row(veto_reason='reject_local_snr',
                                                                                          size_verdict='', size_rule='')]
    assert write(ok) == [], write(ok)
    assert write([row(veto_reason='gaussfit-size')]), "a 'gaussfit-size' row the rule did not remove"
    assert write([row(veto_reason='mstruct', size_verdict='axis-ratio')]), 'the rule removed it, the ledger says mstruct'
    assert write([row(final='1', axis_excess='1.2')]), 'a final the rule removes'
    assert write([row(final='1', size_verdict='')]), 'a final the rule never judged'
    assert write([row(final='1', size_rule='envelope:other.json:0123')]), 'another product than the record names'
    assert write([row(final='1')], header=DS.GLED_COLS), "the record says envelope, the header lacks the rule's columns"
    os.remove(os.path.join(rd, SZ.RUN_RECORD))
    assert write([row(final='1')]), "the header carries the rule's columns, no record"
    assert write([row(final='1')], header=DS.GLED_COLS) == [], 'production: nothing to check'


def test_catalog_paths():
    """The 10-minute catalogue reads its search jobs' records (one rule; a mix refuses); the incremental deep path reads the
    merged ledger's header; both give the builder P3_SIZE_GATE_7_8=inf under the rule and refuse production's fixed removals."""
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
        o = types.SimpleNamespace(src=src, lst='14h', label='SLST14B', _size=None)
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
    U = os.path.join(T, 'U')
    os.makedirs(U)
    ic = types.SimpleNamespace(out=U, label='P3S5', parents='')
    open(os.path.join(U, 'v_deep_ledger.csv'), 'w').write(','.join(DS.GLED_COLS) + '\n')
    assert 'P3_SIZE_GATE_7_8' not in INC.stage_env(ic, 'build'), 'production: the builder keeps its 1.7'
    open(os.path.join(U, 'v_deep_ledger.csv'), 'w').write(','.join(DS.GLED_COLS + DS.GLED_COLS_SIZE) + '\n')
    assert INC.stage_env(ic, 'build')['P3_SIZE_GATE_7_8'] == 'inf' and INC.stage_env(ic, 'fork')['P3_SIZE_GATE_7_8'] == 'inf'
    assert 'P3_SIZE_GATE_7_8' not in INC.stage_env(ic, 'conform')
    rm = os.path.join(U, 'master_catalog_removed.csv')
    with open(rm, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['cand_id', 'removal_reason']); w.writeheader()
        w.writerow(dict(cand_id='I1', removal_reason='axis ratio 2.4 x beam > 2.0 above 8.0 sigma (Gregg 2026-09-05)'))
    try:
        INC._fixed_size_refusal(U, [rm], 'conform')
        raise AssertionError("production's fixed axis removal under the rule must refuse")
    except Refusal as e:
        assert 'conform: 1 row(s)' in str(e), str(e)
    open(os.path.join(U, 'v_deep_ledger.csv'), 'w').write(','.join(DS.GLED_COLS) + '\n')
    INC._fixed_size_refusal(U, [rm], 'conform')            # production's attempts: its own removal stands


def test_carry():
    got = SZ.carry_instant([('J1', 'n1', False), ('J1', 'n1', True), ('J1', 'n2', False), ('J2', 'n1', False)])
    assert got == {('J1', 'n1')}, got


TESTS = [test_measure, test_products, test_snr, test_floor, test_verdict, test_injection, test_domain, test_annulus, test_stamp,
         test_engine, test_plumbing, test_records, test_downstream, test_ledger, test_catalog_paths, test_carry]


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
