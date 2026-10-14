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
        # a CACHED six-bin mean is the fallback at every use (skeptic MINOR 3: only the first use was counted)
        CFG.authorize([], out=T)
        CFG.set('fallbacks.PB_DEEP_MAP_FALLBACK', 'fallback')
        m = fallback.mark()
        PB._DEEP_CACHE.pop('41MHz', None)
        PB.subband_deep_pb('41MHz', pbd)
        PB.subband_deep_pb('41MHz', pbd)                                   # cache hit
        check('subband_deep_pb: a cached six-bin mean is counted at every use', fallback.since(m)[1] ==
              {'fallbacks.PB_DEEP_MAP_FALLBACK': 2}, fallback.since(m))
        CFG.reset()
        try:
            PB.subband_deep_pb('41MHz', pbd)
            hit_refused = False
        except fallback.FallbackRefused:
            hit_refused = True
        check('subband_deep_pb: a cached six-bin mean refuses under the ruled policy', hit_refused)
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
        # the memory cache and the node-local disk cache (skeptic MINOR 3: a cached window built at the default time was
        # used silently - the disk cache was read before the site, and its file name does not record PBTREF)
        CFG.authorize([], out=T)
        CFG.set('fallbacks.PB_REFERENCE_TIME_FALLBACK', 'fallback')
        PB._WIN_CACHE.clear()
        PB.window_model_pb('41MHz', 0.0, 10.0, pbd)
        m = fallback.mark()
        PB.window_model_pb('41MHz', 0.0, 10.0, pbd)                        # memory-cache hit
        check('window_model_pb: a memory-cached window without PBTREF is counted at every use',
              fallback.since(m)[1] == {'fallbacks.PB_REFERENCE_TIME_FALLBACK': 1}, fallback.since(m))
        CFG.reset()
        cdir = f'{T}/wincache'
        os.makedirs(cdir, exist_ok=True)
        np.save(f'{cdir}/41MHz_000.0_010.0_n3_{os.path.basename(os.path.dirname(os.path.abspath(pbd)))}.npy',
                np.ones((8, 8), np.float32))
        os.environ['P3_PB_WINDOW_CACHE'] = cdir
        PB._WIN_CACHE.clear()
        try:
            PB.window_model_pb('41MHz', 0.0, 10.0, pbd)
            disk_refused = False
        except fallback.FallbackRefused:
            disk_refused = True
        os.environ['P3_PB_WINDOW_CACHE'] = ''
        check('window_model_pb: a disk-cached window refuses before use when the library map has no PBTREF', disk_refused)
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
        # ---- minimum counts (Gregg 2026-10-14): the co-add beam gate's skip below 5 readable beam areas is COUNTED ---------
        K = 'fallbacks.MASTERS_BEAM_GATE_SKIP_FALLBACK'
        bdir = f'{T}/beam'
        os.makedirs(bdir, exist_ok=True)
        imgs = {}
        for i in range(6):
            imgs[f'2025-01-0{i + 1}'] = f'{bdir}/img{i}.fits'
            wfits(imgs[f'2025-01-0{i + 1}'], np.zeros((4, 4)), BMAJ=0.25, BMIN=0.20)
        orig = (C.find_latest_science_dir, C.find_subband_alignment_image)
        C.find_latest_science_dir = lambda lst, d: d
        C.find_subband_alignment_image = lambda sci, sb: imgs.get(sci)
        try:
            CFG.reset()
            m = fallback.mark()
            r4 = C.coadd_beam_gate('04h', '41MHz', sorted(imgs)[:4])
            c4 = fallback.since(m)[1].get(K)
            m = fallback.mark()
            r5 = C.coadd_beam_gate('04h', '41MHz', sorted(imgs)[:5])
            c5 = fallback.since(m)[1].get(K)
        finally:
            C.find_latest_science_dir, C.find_subband_alignment_image = orig
        check('co-add beam gate: 4 readable beam areas -> skipped and counted once (production result unchanged); 5 -> runs, '
              'nothing counted', r4[0] == [] and len(r4[1]) == 4 and c4 == 1 and r5[0] == [] and not c5, (r4, c4, r5, c5))
        # ---- the PB weights' missing Phase-2 noise files: every night counted; a wideband sibling above the limit refuses -
        from exopipe.phase3.core import record
        from exopipe.phase3.lib.masters import deep_stages as DS
        K2 = 'fallbacks.PB_QA_MISSING_FALLBACK'
        run = record.Run('masters', ['test'], f'{T}/pbqa_out', CFG, dry_run=True)

        def gate(sc, nd, nm, frac, name):
            fd = f'{T}/pbqa_{name}'
            os.makedirs(fd, exist_ok=True)
            wfits(f'{fd}/I_deep_Taper_R075_pbcorr_{sc}.fits', np.zeros((4, 4)), PBNDATE=nd, PBMISS=nm)
            rows = DS.pb_qa_missing(fd, sc, ['I_deep_Taper_R075', 'V_deep_Taper_R0'])
            m = fallback.mark()
            with run.unit(f'pbqa_{name}') as u:
                DS.pb_qa_gate(u, f'12h_{sc}', sc, rows, frac)
                u.add_output(f'{fd}/I_deep_Taper_R075_pbcorr_{sc}.fits', 'csv_table', test=False)
            return u.status, u.reason, fallback.since(m)[1].get(K2), rows
        st1 = gate('73MHz', 20, 10, 0.25, 'sub')                   # over the limit: a subband's weights cancel
        st2 = gate('Blue', 24, 2, 0.25, 'wb_under')
        st3 = gate('Blue', 20, 10, 0.25, 'wb_over')
        st4 = gate('Blue', 20, 10, 1.0, 'wb_production')
        st5 = gate('Blue', 30, 0, 0.25, 'wb_none')
        # the stage gates BEFORE do_cell writes anything, from the beam masters' constituents (review 2026-10-14 m4 / m5)
        orig_qa = (PB.QA_ROOT, PB.constituent_dates)
        qa = f'{T}/qa_root'
        for dte in ('2025-01-01', '2025-01-02', '2025-01-03'):
            os.makedirs(f'{qa}/12h/{dte}/Science_x/QA', exist_ok=True)
            open(f'{qa}/12h/{dte}/Science_x/QA/{dte}_12h_thermal_noise.csv', 'w').write('category,freq,weight\ndeep,73.0,1.0\n')
        PB.QA_ROOT = qa
        PB.constituent_dates = lambda mp: ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04']
        try:
            fdq = f'{T}/pbqa_inputs'
            wfits(f'{fdq}/I_deep_Taper_R075_Blue.fits', np.zeros((2, 2)))
            pre = DS.pb_qa_inputs(fdq, '12h', 'Blue')
        finally:
            PB.QA_ROOT, PB.constituent_dates = orig_qa
        check('pb_qa_inputs: the beam master\'s nights split by their noise file (3 with, 1 without), before any write',
              pre == [('I_deep_Taper_R075', 3, 1)], pre)
        # the stage itself (review 2026-10-14 m5): a wideband cell over the limit refuses and do_cell - the writer of the PB
        # maps and siblings - is never called; at the limit's other side the stage reaches do_cell. Stubbed: the record
        # chain, the in-place module setup (tested in test_masters-deep) and do_cell; the config's PB library and residual
        # table point at empty stand-ins (record.fingerprint hashes every input).
        from exopipe.phase3.lib.masters import deep_inplace as IP
        from exopipe.phase3.lib.masters import pbcorr_masters as PCM

        class _Sec:
            def __init__(self, base, **kw):
                self._b, self._kw = base, kw

            def __getattr__(self, k):
                return self._kw[k] if k in self._kw else getattr(self._b, k)

        class _Cfg:
            def __init__(self, base, sec):
                self._b, self._sec = base, sec

            def __getattr__(self, k):
                return self._sec if k == 'masters_pbmaps' else getattr(self._b, k)

        class _Reached(Exception):
            pass

        calls = []

        def _do_cell(out, cell):
            calls.append(cell)
            raise _Reached(cell)

        os.makedirs(f'{T}/pblib_empty', exist_ok=True)
        open(f'{T}/residual_stub.csv', 'w').write('band,gain\n')
        saved = (DS.require_stage, DS.inplace, DS._cfg, IP.check_import_env, IP.apply_attrs, IP.check_mirrors, PCM.do_cell,
                 PB.QA_ROOT, PB.constituent_dates)
        DS.require_stage = lambda *a: None
        DS.inplace = lambda out: None
        DS._cfg = lambda: _Cfg(CFG, _Sec(CFG.masters_pbmaps, PB_LIBRARY=f'{T}/pblib_empty',
                                         PB_RESIDUAL_V2_TABLE=f'{T}/residual_stub.csv'))
        IP.check_import_env = lambda: None
        IP.apply_attrs = lambda *a, **k: []
        IP.check_mirrors = lambda *a, **k: None
        PCM.do_cell = _do_cell
        PB.QA_ROOT = qa                                    # QA files for 2025-01-01..03; 2025-01-04..05 have none
        try:
            res = {}
            for name, dates in (('over', ['2025-01-01', '2025-01-02', '2025-01-04', '2025-01-05']),   # 2 of 4 missing
                                ('under', ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'])):  # 1 of 4 = the limit
                out = f'{T}/pbmaps_stage_{name}'
                if os.path.isdir(out):
                    shutil.rmtree(out)
                wfits(f'{out}/12h_Blue/coadd_final/I_deep_Taper_R075_Blue.fits', np.zeros((2, 2)))
                PB.constituent_dates = lambda mp, _d=dates: list(_d)
                srun = record.Run('masters', ['test'], out, CFG, dry_run=True)
                del calls[:]
                try:
                    DS.pbmaps('12h_Blue', None, srun, out, {})
                    reached = False
                except _Reached:
                    reached = True
                res[name] = (srun.units[-1].status, srun.units[-1].reason, list(calls), reached,
                             sorted(os.listdir(f'{out}/12h_Blue/coadd_final')))
        finally:
            (DS.require_stage, DS.inplace, DS._cfg, IP.check_import_env, IP.apply_attrs, IP.check_mirrors, PCM.do_cell,
             PB.QA_ROOT, PB.constituent_dates) = saved
        st, why, cl, reached, files = res['over']
        check('pbmaps stage: a wideband cell missing 2 of 4 noise files refuses BEFORE do_cell (never called; nothing written)',
              st == 'refused' and 'PB_QA_MISSING_MAX_FRAC' in why and not cl and not reached
              and files == ['I_deep_Taper_R075_Blue.fits'], res['over'])
        st, why, cl, reached, files = res['under']
        check('pbmaps stage: 1 of 4 missing (at the 25% limit, not above) passes the gate and reaches do_cell',
              cl == ['12h_Blue'] and reached, res['under'])
        check('PB weights: missing noise files counted per cell (10 / 2 / 10 / 10 / 0); a subband never refused (10 of 30); a '
              'wideband refused only above masters_pbmaps.PB_QA_MISSING_MAX_FRAC (10 of 30 at 0.25), not at 2 of 26 or under '
              'production\'s 1.0', st1[:1] == ('done',) and st1[2] == 10 and st2[0] == 'done' and st2[2] == 2
              and st3[0] == 'refused' and 'PB_QA_MISSING_MAX_FRAC' in st3[1] and st3[2] == 10
              and st4[0] == 'done' and st4[2] == 10 and st5[0] == 'done' and not st5[2] and st1[3] == [('I_deep_Taper_R075', 20, 10)],
              (st1, st2, st3, st4, st5))
    finally:
        CFG.reset()
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
