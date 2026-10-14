"""B-30 (a) fallback sites of the search engine (lib/search/deep_search.py, search_v.py, deep_step.py; core/fallback.py)
and the ruled ephemeris time ("ephemerides at frame mid-time", search_detect.EPHEMERIS_TIME).

Per helper site, on synthetic inputs: under the ruled policy the site refuses naming its key (a count-only site counts);
under the production policy 'fallback' the function returns what production returns, counted. The sites inside
run_deep_cell (PRODNAME, pbcorr sibling, beam fit, epochs without a flow, weights.csv, sanity gates, DURATION) need a real
cell: they are covered by the registry test (every key named by a site) and the search parity replay.
Run: python tests/test_search_b30.py
"""
import json
import os
import shutil
import sys
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from astropy.io import fits                                          # noqa: E402
from exopipe.phase3.config import CFG                                # noqa: E402
from exopipe.phase3.core import fallback                             # noqa: E402

T = '/tmp/exopipe3_tests_search_b30'
FAILS, N = [], [0]


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def outcome(fn):
    try:
        return fn(), None
    except fallback.FallbackRefused as e:
        return None, str(e)


def under(key, pol, fn):
    """Run fn() with ``key`` at policy ``pol``; returns (result, refusal text, (refusals, counts) recorded)."""
    CFG.reset()
    CFG.authorize([], out=T)
    CFG.set(key, pol)
    m = fallback.mark()
    r, ref = outcome(fn)
    st = fallback.since(m)
    CFG.reset()
    return r, ref, st


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    from exopipe.phase3.lib.search import deep_search as DS, search_v as SV, deep_step as DST
    try:
        # ---- garbage pixels: a per-item drop, counted under the ruled default -------------------------------------
        a = np.zeros((8, 8), np.float32)
        a[1, 1], a[2, 2] = 1e9, np.nan
        p = f'{T}/zap.fits'
        fits.writeto(p, a)
        CFG.reset()
        m = fallback.mark()
        r = DS._load(p)
        check('zapped garbage pixels: counted (n pixels) under the ruled default, the array as production',
              fallback.since(m) == ([], {'fallbacks.SEARCH_ZAP_PIXELS_FALLBACK': 1}) and r[1, 1] == 0 and r[2, 2] == 0
              and CFG.fallbacks.SEARCH_ZAP_PIXELS_FALLBACK == 'fallback', fallback.since(m))

        # ---- ephemeris time: the 00:00 default refuses; the ruled frame mid-time ------------------------------------
        k = 'fallbacks.SEARCH_EPHEMERIS_TIME_FALLBACK'
        r, ref, st = under(k, 'refuse', lambda: DS._body_skycoord('moon', None, '2025-08-16'))
        check('ephemeris without an epoch time refuses under the ruled policy', ref and k in ref, ref)
        r, ref, st = under(k, 'refuse', lambda: DS._body_skycoord('moon', 'not-a-time', '2025-08-16'))
        check('ephemeris with an unparsable epoch time refuses', ref and 'unparsable' in ref, ref)
        r, ref, st = under(k, 'fallback', lambda: DS._body_skycoord('moon', None, '2025-08-16'))
        want = DS._body_skycoord('moon', '2025-08-16T00:00:00', '2025-08-16')
        check('ephemeris fallback: 00:00 UTC of the date, counted', ref is None and st[1] == {k: 1}
              and r.separation(want).arcsec < 1e-6, (ref, st))
        sub = f'{T}/base/subtracted'
        os.makedirs(sub)
        h = fits.Header()
        h['DATE-OBS'], h['DURATION'] = '2025-08-16T05:00:00', 3600.0
        fits.writeto(f'{sub}/2025-08-16_I_sub_41MHz.fits', np.zeros((4, 4), np.float32), h)
        os.environ.pop('EXOPIPE3_EPHEMERIS_TIME', None)
        check('production ephemeris time: the epoch DATE-OBS, unchanged',
              DS._eph_time(f'{T}/base', '2025-08-16', '41MHz', '2025-08-16T05:00:00') == '2025-08-16T05:00:00')
        os.environ['EXOPIPE3_EPHEMERIS_TIME'] = 'frame-mid'
        te = DS._eph_time(f'{T}/base', '2025-08-16', '41MHz', '2025-08-16T05:00:00')
        none = DS._eph_time(f'{T}/base', '2025-08-17', '41MHz', '2025-08-17T05:00:00')
        os.environ.pop('EXOPIPE3_EPHEMERIS_TIME', None)
        check('ruled ephemeris time: the frame mid-time (DATE-OBS 05:00 + DURATION 3600 s -> 05:30); no frame -> None',
              te.startswith('2025-08-16T05:30:00') and none is None, (te, none))
        env = DST.engine_env(f'{T}/c', f'{T}/a', f'{T}/f', 'X', f'{T}/ar', f'{T}/tmp')
        check('engine_env passes search_detect.EPHEMERIS_TIME to the engine child (ruled default frame-mid)',
              env.get('EXOPIPE3_EPHEMERIS_TIME') == 'frame-mid' == CFG.search_detect.EPHEMERIS_TIME,
              env.get('EXOPIPE3_EPHEMERIS_TIME'))

        # ---- master structure map: beam <= 0, destripe failure -----------------------------------------------------
        mv = np.random.default_rng(3).normal(0, 1, (32, 32)).astype(np.float64)
        k = 'fallbacks.SEARCH_BEAM_FALLBACK'
        r, ref, st = under(k, 'refuse', lambda: DS._mstruct_maps(mv, f'{T}/nomaster.fits', 0.0))
        check('structure map with no beam refuses under the ruled policy', ref and k in ref, ref)
        k2 = 'fallbacks.SEARCH_MSTRUCT_DESTRIPE_FALLBACK'
        r, ref, st = under(k2, 'refuse', lambda: DS._mstruct_maps(mv, f'{T}/nomaster.fits', 3.0))
        check('structure map whose destripe fails refuses under the ruled policy', ref and k2 in ref, ref)
        r, ref, st = under(k2, 'fallback', lambda: DS._mstruct_maps(mv, f'{T}/nomaster.fits', 3.0))
        check('destripe failure under fallback: the raw map twice, counted', ref is None and st[1] == {k2: 1}
              and r[0] is r[1], (ref, st))

        # ---- residuals that fail in the fork workers: the site in the parent ---------------------------------------
        saved = DS._residual
        DS._residual = lambda epoch, master, flow, hdr, cd: (_ for _ in ()).throw(ValueError('bad')) if epoch is None \
            else np.asarray(epoch) - master
        ep = {'d1': np.ones((4, 4)), 'd2': None, 'd3': np.ones((4, 4))}
        call = lambda: DS._build_residuals_mp(ep, np.zeros((4, 4)), {d: None for d in ep}, None, 1.0, list(ep), 2, tag='t')
        k = 'fallbacks.SEARCH_RESIDUAL_FALLBACK'
        r, ref, st = under(k, 'refuse', call)
        check('a residual that failed in a worker refuses in the parent under the ruled policy', ref and k in ref, ref)
        r, ref, st = under(k, 'fallback', call)
        check('residual failure under fallback: the epoch dropped as production, counted', ref is None
              and sorted(r) == ['d1', 'd3'] and st[1] == {k: 1}, (ref, st))
        DS._residual = saved

        # ---- search_v: beam model, frequency --------------------------------------------------------------------
        k = 'fallbacks.SEARCH_PB_MODEL_FALLBACK'
        stub = types.ModuleType('exopipe.phase2.extractor_pb_75')
        stub.BEAM_PATH, stub.BeamModel = '/nonexistent/beam.h5', object
        real = sys.modules.get('exopipe.phase2.extractor_pb_75')
        sys.modules['exopipe.phase2.extractor_pb_75'] = stub

        def model():
            SV._BEAM_MODEL, SV._BEAM_MODEL_TRIED = None, False
            return SV._get_beam_model()
        r, ref, st = under(k, 'refuse', model)
        check('no beam model refuses under the ruled policy', ref and k in ref, ref)
        r, ref, st = under(k, 'fallback', model)
        check('no beam model under fallback: None (the analytic response), counted', ref is None and r is None
              and st[1] == {k: 1}, (ref, st))
        if real is None:
            sys.modules.pop('exopipe.phase2.extractor_pb_75', None)
        else:
            sys.modules['exopipe.phase2.extractor_pb_75'] = real
        SV._BEAM_MODEL, SV._BEAM_MODEL_TRIED = None, False
        k = 'fallbacks.SEARCH_FREQ_FALLBACK'
        r, ref, st = under(k, 'refuse', lambda: SV._get_freq_mhz(fits.Header(), '/x/41MHz-I-deep.fits'))
        check('no frequency card refuses under the ruled policy', ref and k in ref, ref)
        r, ref, st = under(k, 'fallback', lambda: SV._get_freq_mhz(fits.Header(), '/x/41MHz-I-deep.fits'))
        check('no frequency card under fallback: label + 2.3 MHz, counted', ref is None and abs(r - 43.3) < 1e-9
              and st[1] == {k: 1}, (r, st))
        hh = fits.Header()
        hh['RESTFRQ'] = 41.0e6
        m = fallback.mark()
        check('a frequency card is not a fallback', SV._get_freq_mhz(hh, '/x/41MHz.fits') == 41.0
              and fallback.since(m) == ([], {}))

        # ---- the engine child reports a refusal (a site) as refused, exit 2 ----------------------------------------
        spec = f'{T}/spec.json'
        json.dump({'label': 'X', 'stokes': 'V', 'cadence': 'deep', 'lst': '23h', 'scope': '23MHz', 'cell': '23h_23MHz',
                   'markers': [], 'result': f'{T}/result.json'}, open(spec, 'w'))
        saved = DS.run_deep_cell
        DS.run_deep_cell = lambda *a, **kw: (_ for _ in ()).throw(fallback.FallbackRefused('B-30 fallback site x: y'))
        rc = DST._child(spec)
        res = json.load(open(f'{T}/result.json'))
        DS.run_deep_cell = saved
        check('engine child: a Refusal (a fallback site) is refused, exit 2', rc == 2 and res['status'] == 'refused', res)
    finally:
        CFG.reset()
        os.environ.pop('EXOPIPE3_EPHEMERIS_TIME', None)
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
