"""B-30 (a) fallback sites of the masters modules (lib/masters/coadd.py, pb_apply.py, vclean_masters.py; core/fallback.py).

Per site, on synthetic files: under the ruled policy 'refuse' the site raises FallbackRefused naming its key and input;
under the production policy 'fallback' the function returns what production returns and the site is counted once.
A check fails if a site is missing, refuses under 'fallback', changes the production result, or fires on a good input.
Run: python tests/test_masters_b30.py
"""
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from astropy.io import fits                                          # noqa: E402
from exopipe.phase3.config import CFG                                # noqa: E402
from exopipe.phase3.core import fallback                             # noqa: E402
from exopipe.phase3.lib.masters import coadd as C                    # noqa: E402
from exopipe.phase3.lib.masters import pb_apply as PB                # noqa: E402
from exopipe.phase3.lib.masters import vclean_masters as VC          # noqa: E402

T = '/tmp/exopipe3_tests_masters_b30'
FAILS, N = [], [0]


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def both(key, fn, prod_ok, label):
    """Run fn() under 'refuse' (must raise naming key) and under 'fallback' (prod_ok(result) and one count)."""
    CFG.reset()
    try:
        fn()
        refused, msg = False, 'no refusal'
    except fallback.FallbackRefused as e:
        refused, msg = key in str(e), str(e)
    check(f'{label}: refuses under the ruled policy, naming {key}', refused, msg[:200])
    CFG.authorize([], out=T)
    CFG.set(key, 'fallback')
    m = fallback.mark()
    try:
        r, err = fn(), None
    except Exception as e:                    # noqa: BLE001
        r, err = None, e
    refs, counts = fallback.since(m)
    check(f'{label}: production behaviour under fallback, counted once', err is None and not refs
          and counts.get(key) == 1 and prod_ok(r), (err, refs, counts))
    CFG.reset()
    return r


def wfits(p, data, **cards):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    h = fits.Header()
    for k, v in cards.items():
        h[k.replace('_', '-') if k in ('DATE_OBS', 'DATE_END') else k] = v
    fits.writeto(p, np.asarray(data, np.float32), h, overwrite=True)
    return p


def wcs_cards(n=8):
    return dict(CTYPE1='RA---SIN', CTYPE2='DEC--SIN', CRVAL1=150.0, CRVAL2=37.0, CRPIX1=n / 2, CRPIX2=n / 2,
                CDELT1=-1.0, CDELT2=1.0, CUNIT1='deg', CUNIT2='deg')


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    saved = {}
    try:
        # ---- coadd.epoch_time_header: an unreadable per-epoch source ---------------------------------------------
        tmpl = fits.Header()
        tmpl['DATE-OBS'] = '2024-01-01T00:00:00'
        r = both('fallbacks.EPOCH_TIME_SOURCE_FALLBACK', lambda: C.epoch_time_header(tmpl, f'{T}/nope.fits'),
                 lambda h: 'DATE-OBS' not in h, 'epoch_time_header')
        good = wfits(f'{T}/epoch.fits', np.zeros((4, 4)), DATE_OBS='2025-08-16T09:36:31')
        m = fallback.mark()
        h = C.epoch_time_header(tmpl, good)
        check('epoch_time_header: a readable source is not a fallback', h['DATE-OBS'] == '2025-08-16T09:36:31'
              and fallback.since(m) == ([], {}))

        # ---- coadd.stamp_master_span: no time source for a night / no constituent span ---------------------------
        master = wfits(f'{T}/master.fits', np.zeros((4, 4)), DATE_OBS='2024-01-01T00:00:00')
        src = wfits(f'{T}/src_0817.fits', np.zeros((4, 4)), DATE_OBS='2025-08-17T09:00:00', DATE_END='2025-08-17T10:00:00',
                    INTTIME=3600.0)
        saved['ts'] = C._phase2_time_source
        C.MASTER_SPAN_SOURCE = 'phase2-glob'                                                # production's finder, stubbed
        C._phase2_time_source = lambda lst, d, scope: src if d == '2025-08-17' else None
        both('fallbacks.MASTER_SPAN_FALLBACK', lambda: C.stamp_master_span(master, '23h', '23MHz', ['2025-08-16', '2025-08-17']),
             lambda r: fits.getheader(master)['DATE-OBS'].startswith('2025-08-17T09:00:00'), 'stamp_master_span (night without a time source)')
        m = fallback.mark()
        C.stamp_master_span(master, '23h', '23MHz', ['2025-08-17'])
        check('stamp_master_span: every night with its time block is not a fallback', fallback.since(m) == ([], {})
              and fits.getheader(master)['NSTACK'] == 1)
        C._phase2_time_source = saved.pop('ts')
        # MASTER_SPAN_SOURCE 'constituent' (B-30 skeptic MAJOR 1): each night's OWN image - a raw-failsafe name with a
        # 2949 s integration the production globs cannot match - gives the span; a night without its image refuses
        saved['span'] = C.MASTER_SPAN_SOURCE
        C.MASTER_SPAN_SOURCE = 'constituent'
        raw = wfits(f'{T}/41MHz-I-Taper-2949s-Robust-0.75-20250502_123513-image.fits', np.zeros((4, 4)),
                    DATE_OBS='2025-05-02T12:35:13', DATE_END='2025-05-02T13:24:22', INTTIME=2949.0)
        m = fallback.mark()
        C.stamp_master_span(master, '19h', '41MHz', ['2025-05-02', '2025-08-17'], sources={'2025-05-02': raw, '2025-08-17': src})
        h = fits.getheader(master)
        check('constituent span: each night\'s own image (raw-failsafe name included), no site',
              fallback.since(m) == ([], {}) and h['DATE-OBS'].startswith('2025-05-02T12:35:13')
              and h['DATE-END'].startswith('2025-08-17T10:00:00') and abs(h['INTTIME'] - 6549.0) < 1e-6, dict(h))
        both('fallbacks.MASTER_SPAN_FALLBACK', lambda: C.stamp_master_span(master, '19h', '41MHz', ['2025-05-02', '2025-05-03'],
                                                                             sources={'2025-05-02': raw}),
             lambda r: True, 'constituent span (a night without its image)')
        C.MASTER_SPAN_SOURCE = saved.pop('span')

        # ---- coadd._select_reference: a previous master without COADDOP ------------------------------------------
        wd = f'{T}/wd/23h_23MHz'
        wfits(f'{wd}/coadd_final/I_deep_Taper_R075_23MHz.fits', np.zeros((4, 4)))           # no COADDOP card
        saved['mode'], saved['res'] = C.REF_MODE, C._resolve_reference_entry
        C.REF_MODE = 'master'
        C._resolve_reference_entry = lambda lst, fl, d, dates, workdir: ('FIXED', d, None)
        both('fallbacks.REFERENCE_MASTER_FALLBACK', lambda: C._select_reference('23h', '23MHz', wd, []),
             lambda r: r == ('FIXED', C.REFERENCE_DATES.get('23h'), None), '_select_reference (previous master without COADDOP)')
        C.REF_MODE, C._resolve_reference_entry = saved.pop('mode'), saved.pop('res')

        # ---- coadd._grid_refused_dates: an unreadable post-cutover image -----------------------------------------
        late = '2026-01-01'
        both('fallbacks.GRID_HEADER_FALLBACK', lambda: C._grid_refused_dates([(late, None, f'{T}/missing.fits')], 'x'),
             lambda r: r == set(), '_grid_refused_dates (unreadable header)')

        # ---- pb_apply.subband_deep_pb: no 60-min map, six bins -----------------------------------------------------
        pbd = f'{T}/pb'
        for k in range(6):
            wfits(PB._bin_path('41MHz', k, pbd), np.full((8, 8), 0.5 + 0.1 * k), **wcs_cards())
        PB._DEEP_CACHE.pop('41MHz', None)

        def deep():
            PB._DEEP_CACHE.pop('41MHz', None)
            return PB.subband_deep_pb('41MHz', pbd)
        both('fallbacks.PB_DEEP_MAP_FALLBACK', deep, lambda r: np.allclose(r[0], 0.75), 'subband_deep_pb (no deep map)')
        wfits(PB._deep_path('41MHz', pbd), np.full((8, 8), 0.75), **wcs_cards())
        m = fallback.mark()
        deep()
        check('subband_deep_pb: a present deep map is not a fallback', fallback.since(m) == ([], {}))

        # ---- pb_apply.window_model_pb: a library map without PBTREF ----------------------------------------------
        saved['interp'], saved['cache'] = PB._beam_interp, os.environ.get('P3_PB_WINDOW_CACHE')
        PB._beam_interp = lambda: (lambda pts: np.ones(len(pts)))
        os.environ['P3_PB_WINDOW_CACHE'] = ''

        def window():
            PB._WIN_CACHE.clear()
            return PB.window_model_pb('41MHz', 0.0, 10.0, pbd)
        both('fallbacks.PB_REFERENCE_TIME_FALLBACK', window, lambda r: r[0].shape == (8, 8), 'window_model_pb (no PBTREF)')
        PB._beam_interp = saved.pop('interp')
        c = saved.pop('cache')
        if c is None:
            os.environ.pop('P3_PB_WINDOW_CACHE', None)
        else:
            os.environ['P3_PB_WINDOW_CACHE'] = c

        # ---- pb_apply.wideband_pb: Full with no positive weight; a subband master without RESTFRQ -----------------
        mp = wfits(f'{T}/m41.fits', np.zeros((8, 8)), BMAJ=0.5)                               # BMAJ, no RESTFRQ
        saved.update(cd=PB.constituent_dates, sw=PB.subband_weights, cw=PB._canonical_full_wcs, tf=PB.THREE_FACTOR,
                     rp=PB.RESIDUAL_PB, cg=PB.chromatic_gain, rg=PB.residual_band_gain)
        from astropy.wcs import WCS
        fh = fits.Header(wcs_cards())
        fh['NAXIS1'], fh['NAXIS2'] = 8, 8
        PB.constituent_dates = lambda p: ['2025-08-16']
        PB._canonical_full_wcs = lambda pb_dir=None: (WCS(fh), fh)
        PB.THREE_FACTOR, PB.RESIDUAL_PB = False, ''
        PB.subband_weights = lambda lst, dates, category='deep': ({sb: 0.0 for sb in PB.cfg.ALL_SUBBANDS}, [], dates)
        both('fallbacks.PB_WEIGHTS_FALLBACK', lambda: PB.wideband_pb('Full', '23h', mp, pbd),
             lambda r: r[0] is not None and not np.any(r[0]), 'wideband_pb Full (no member with a positive weight)')
        PB.THREE_FACTOR = True
        PB.subband_weights = lambda lst, dates, category='deep': ({'41MHz': 1.0}, dates, [])
        PB.chromatic_gain = lambda hdr, shape, nu, bm, dnu_hz=None: (np.ones(shape), 'c')
        PB.residual_band_gain = lambda hdr, shape, band: (np.ones(shape), 'r')
        PB._DEEP_CACHE.pop('41MHz', None)
        both('fallbacks.PB_RESTFRQ_FALLBACK', lambda: PB.wideband_pb('41MHz', '23h', mp, pbd),
             lambda r: np.allclose(r[0], 0.75), 'wideband_pb three-factor (master without RESTFRQ)')
        mp2 = wfits(f'{T}/m41b.fits', np.zeros((8, 8)), BMAJ=0.5, RESTFRQ=41.0e6)
        m = fallback.mark()
        PB.wideband_pb('41MHz', '23h', mp2, pbd)
        check('wideband_pb: a master with RESTFRQ is not a fallback', fallback.since(m) == ([], {}))
        (PB.constituent_dates, PB.subband_weights, PB._canonical_full_wcs, PB.THREE_FACTOR, PB.RESIDUAL_PB,
         PB.chromatic_gain, PB.residual_band_gain) = (saved.pop('cd'), saved.pop('sw'), saved.pop('cw'), saved.pop('tf'),
                                                      saved.pop('rp'), saved.pop('cg'), saved.pop('rg'))

        # ---- vclean_masters.pick_i_master: the preferred robust corrupt --------------------------------------------
        cf = f'{T}/cf'
        wfits(f'{cf}/I_deep_Taper_R075_41MHz.fits', np.full((4, 4), 1e9))                   # corrupt (> 1e6)
        wfits(f'{cf}/I_deep_Taper_R0_41MHz.fits', np.ones((4, 4)))
        both('fallbacks.VCLEAN_I_MASTER_FALLBACK', lambda: VC.pick_i_master(cf, '41MHz'),
             lambda r: r.endswith('I_deep_Taper_R0_41MHz.fits'), 'pick_i_master (preferred robust corrupt)')
        wfits(f'{cf}/I_deep_Taper_R075_41MHz.fits', np.ones((4, 4)))
        m = fallback.mark()
        check('pick_i_master: the preferred robust sane is not a fallback',
              VC.pick_i_master(cf, '41MHz').endswith('R075_41MHz.fits') and fallback.since(m) == ([], {}))
    finally:
        CFG.reset()
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
