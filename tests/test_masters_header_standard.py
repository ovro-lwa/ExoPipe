"""The output standard at write, writers step 1 (plan/wave2/OUTPUT_WRITERS_DESIGN.md): the Stage-1 carry of THERMNOI and P2RUN
(lib/masters/thermnoi.py, coadd.night_standard_cards, masters_build.HEADER_STANDARD).

  (1) thermnoi.measure equals what Phase 2's own stamping writes (phase2_science_agg._stamp_subband_thermnoi with the
      aggregation's deep_rms, _stamp_wideband_thermnoi), on synthetic Phase 2 directories: a subband with a dewarped and a raw
      V image (the dewarped one measured), with a raw one only, with none, with an empty file (none: Phase 2 stamps nothing);
      a wideband with dewarped, raw and PB-corrected V products (the dewarped one), with the dewarped one unreadable (the raw
      one), with none;
      a wideband Phase 2 did not form (its log) - none; candidates() in the order measure() walks;
  (2) coadd.night_standard_cards: THERMNOI in the product's units (the measurement / the flat flux factor the pixels were
      divided by) with Phase 2's comment, and P2RUN; pixels without a single factor, or a night without a V image that
      measures, lose a THERMNOI card the template carried and are named in HISTORY; a Phase 2 I card other than the
      measurement is named in HISTORY, an equal one is not; the record and note_thermnoi; applied_flat_scale; shflux.bake and
      rebake_file scale THERMNOI and IMGRMS with the pixels under 'full' only;
  (3) the wiring: masters_build.HEADER_STANDARD 'full' (ruled: its ledger IDs are rows of the fix ledger), the parity file
      'production', the attribute the build sets (deep_inplace), the build refusing an unknown value, subtract_night
      stamping only under 'full';
  (4) real data, read only: on archived nights the measurement equals the Phase 2 card (a subband and a wideband night), a
      night whose images carry no card gets a value, a night whose only V file is empty gets none.
Run: python tests/test_masters_header_standard.py
"""
import csv
import inspect
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

import numpy as np                                                  # noqa: E402
from astropy.io import fits                                         # noqa: E402

from exopipe.phase3.core import env as ENV                          # noqa: E402
ENV.legacy()
from exopipe.phase3.config import CFG                               # noqa: E402
from exopipe.phase3.lib.masters import thermnoi as TN               # noqa: E402

T = '/tmp/exopipe3_tests_header_standard'
FAILS, N = [], [0]
N0 = 256


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'\n      {str(detail)[:900]}'), flush=True)
    if not ok:
        FAILS.append(name)


def img(path, sigma, seed):
    """A small image of Gaussian noise of ``sigma`` (each file its own, so the file measured is told by its value)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = np.random.default_rng(seed).normal(0.0, sigma, (N0, N0)).astype(np.float32)
    fits.PrimaryHDU(d).writeto(path, overwrite=True)
    return path


def p2():
    from exopipe.phase2 import phase2_science_agg as P2
    return P2


# ---------------------------------------------------------------------------------------------------------------------
# (1) the measurement against Phase 2's own stamping
# ---------------------------------------------------------------------------------------------------------------------

SB, TAG = '50MHz', '50MHz-{S}-Taper-3591s-Robust-{R}-20250101_040422-image{X}.fits'


def subband_run(root, v_files):
    """A Science directory with the subband's I deep images and the given V deep files ({suffix: sigma or None (empty)})."""
    sci = os.path.join(root, 'Science_20260101_000000')
    d = os.path.join(sci, SB)
    for k, x in enumerate(('_dewarped', '', '.pbcorr', '.pbcorr_dewarped')):
        img(os.path.join(d, 'I', 'deep', TAG.format(S='I', R='0.75', X=x)), 0.5, 10 + k)
    os.makedirs(os.path.join(d, 'V', 'deep'), exist_ok=True)
    for k, (x, sg) in enumerate(v_files.items()):
        p = os.path.join(d, 'V', 'deep', TAG.format(S='V', R='0', X=x))
        if sg is None:
            open(p, 'w').close()
        else:
            img(p, sg, 20 + k)
    return sci


def phase2_subband(sci):
    """What Phase 2 stamps on the subband's I dewarped image: the aggregation's deep_rms (the first dewarped V deep Taper
    image, else the first raw one; a value only when it measures), then _stamp_subband_thermnoi - on this directory."""
    P2 = p2()
    sb_dir = os.path.join(sci, SB)
    v = (P2._find_subband_image(sb_dir, 'V', 'deep', 'Taper', '', require_dewarped=True)
         or P2._find_subband_image(sb_dir, 'V', 'deep', 'Taper', '', require_dewarped=False))
    deep_rms = {}
    if v:
        rms = P2.get_inner_rms(v[0])
        if np.isfinite(rms) and rms > 0:
            deep_rms[SB] = rms
    P2._stamp_subband_thermnoi(sci, deep_rms, {})
    return fits.getheader(os.path.join(sb_dir, 'I', 'deep', TAG.format(S='I', R='0.75', X='_dewarped'))).get('THERMNOI')


WB = 'Red'
WTAG = 'Red_{S}_deep_Taper_Robust-{R}{X}_20250101_LST02h.fits'


def wideband_run(root, v_files):
    sci = os.path.join(root, 'Science_20260101_000000')
    d = os.path.join(sci, WB)
    img(os.path.join(d, 'I', 'deep', WTAG.format(S='I', R='0.75', X='_dewarped')), 0.5, 30)
    img(os.path.join(d, 'I', 'deep', WTAG.format(S='I', R='0.75', X='')), 0.5, 31)
    os.makedirs(os.path.join(d, 'V', 'deep'), exist_ok=True)
    for k, (x, sg) in enumerate(v_files.items()):
        p = os.path.join(d, 'V', 'deep', WTAG.format(S='V', R='0', X=x))
        if sg is None:
            open(p, 'w').close()
        else:
            img(p, sg, 40 + k)
    return sci


def phase2_wideband(sci):
    p2()._stamp_wideband_thermnoi(sci)
    return fits.getheader(os.path.join(sci, WB, 'I', 'deep', WTAG.format(S='I', R='0.75', X='_dewarped'))).get('THERMNOI')


def test_measure():
    cases = (('subband: a dewarped and a raw V image - the dewarped one', 'sub_both', subband_run,
              {'_dewarped': 0.040, '': 0.060, '.pbcorr': 0.090}, phase2_subband, SB, '_dewarped'),
             ('subband: a raw V image only - the raw one', 'sub_raw', subband_run, {'': 0.060, '.pbcorr': 0.090},
              phase2_subband, SB, ''),
             ('subband: no V image - none', 'sub_none', subband_run, {}, phase2_subband, SB, None),
             ('subband: an empty V file only (the 0-byte Phase 2 products) - none', 'sub_empty', subband_run, {'': None},
              phase2_subband, SB, None),
             ('wideband: dewarped, raw and PB-corrected V products - the dewarped one', 'wb_all', wideband_run,
              {'_dewarped': 0.030, '': 0.050, '_pbcorr': 0.110}, phase2_wideband, WB, '_dewarped'),
             ('wideband: the dewarped V product unreadable - the raw one', 'wb_bad', wideband_run,
              {'_dewarped': None, '': 0.050, '_pbcorr': 0.110}, phase2_wideband, WB, ''),
             ('wideband: no V product - none', 'wb_none', wideband_run, {}, phase2_wideband, WB, None))
    def phase2_wideband_skipped(sci):              # Phase 2 with the band in skip_bands (an incomplete subband set)
        p2()._stamp_wideband_thermnoi(sci, skip_bands={'Red', 'Full'})
        return fits.getheader(os.path.join(sci, WB, 'I', 'deep', WTAG.format(S='I', R='0.75', X='_dewarped'))).get('THERMNOI')
    cases += (('wideband: Phase 2 did not form the band (its log: SKIPPED, an incomplete subband set) though a product '
               'exists - none', 'wb_skipped', wideband_run, {'_dewarped': 0.030, '': 0.050}, phase2_wideband_skipped, WB, None),)
    for desc, name, build, vf, stamp, band, want in cases:
        root = os.path.join(T, name)
        shutil.rmtree(root, ignore_errors=True)
        sci = build(root, vf)
        if name == 'wb_skipped':
            with open(os.path.join(sci, 'phase2_02h_2025-01-01_Science_20260101_000000.log'), 'w') as fh:
                fh.write("2026-01-01 00:00:00,000 - [PHASE2] - WARNING - Wideband bands SKIPPED (incomplete subband set): "
                         "['Full', 'Red']\n")
            TN.phase2_skipped.cache_clear()
        val, vimg, why = TN.measure(sci, band)
        if band == WB and want is not None:
            cand = TN.candidates(sci, band)
            check(f'{desc}: candidates() lists the V products in the order measure() walks them (not PB-corrected, then '
                  f'dewarped), and the one measured is the first of them that measures',
                  cand and os.path.basename(cand[-1]).endswith('_pbcorr_20250101_LST02h.fits') and vimg in cand
                  and all(not TN._measures(p2().get_inner_rms(c)) for c in cand[:cand.index(vimg)]), (cand, vimg))
        card = stamp(sci)                     # Phase 2's own stamping, on the same directory (after the measurement)
        if want is None:
            check(f'{desc}: measure gives none, with its reason, and Phase 2 stamps nothing', val is None and why
                  and card is None, (val, vimg, why, card))
        else:
            bn = os.path.basename(vimg)
            right = ('_dewarped' in bn) if want == '_dewarped' else ('dewarped' not in bn and 'pbcorr' not in bn)
            check(f"{desc}: measure equals Phase 2's stamped card, from that file", val is not None and val == card and right,
                  (val, card, bn))


# ---------------------------------------------------------------------------------------------------------------------
# (2) night_standard_cards
# ---------------------------------------------------------------------------------------------------------------------

def test_cards():
    from exopipe.phase3.lib.masters import coadd as co
    root = os.path.join(T, 'cards')
    shutil.rmtree(root, ignore_errors=True)
    sci = subband_run(root, {'_dewarped': 0.040})
    src = os.path.join(sci, SB, 'I', 'deep', TAG.format(S='I', R='0.75', X='_dewarped'))
    val = TN.measure(sci, SB)[0]
    tmpl = fits.Header()
    tmpl['THERMNOI'] = (0.5, 'the template night')
    del co.THERMNOI_RECORD[:]
    out = co.night_standard_cards(tmpl, sci, SB, src, flux_scale=0.8)
    hist = '\n'.join(str(x) for x in out.get('HISTORY', []))
    check("night_standard_cards: THERMNOI the night's measurement in the product's units - divided by the flux factor its "
          "pixels were divided by (0.8 here) - with Phase 2's comment (the template's replaced), the division in HISTORY, "
          "P2RUN its Science run", out['THERMNOI'] == float(round(val / 0.8, 9)) and out.comments['THERMNOI'] == TN.COMMENT_SUBBAND
          and out['P2RUN'] == 'Science_20260101_000000' and f'THERMNOI {val} Jy/beam measured on' in hist
          and 'flux scale 0.800000' in hist, (out.get('THERMNOI'), val, hist))
    check("the card comment is Phase 2's for the band: a subband's ('matching V Taper inner rms'), a wideband's I images' "
          "('matching V product inner rms')", TN.comment_for('50MHz') == '[Jy/beam] Noise (matching V Taper inner rms)'
          and TN.comment_for('Full') == '[Jy/beam] Noise (matching V product inner rms)')
    out = co.night_standard_cards(tmpl, sci, SB, src, flux_scale=None)
    hist = '\n'.join(str(x) for x in out.get('HISTORY', []))
    check('pixels without a single flux factor (an affine scale): no THERMNOI (the template\'s dropped), HISTORY says why',
          'THERMNOI' not in out and 'no single flux factor' in hist, (out.get('THERMNOI'), hist))
    with fits.open(src, mode='update') as h:
        h[0].header['THERMNOI'] = val
    out = co.night_standard_cards(fits.Header(), sci, SB, src, flux_scale=1.0)
    check('an I card equal to the measurement: no HISTORY about the card', 'the Phase 2 card on' not in
          '\n'.join(str(x) for x in out.get('HISTORY', [])), out.get('HISTORY'))
    with fits.open(src, mode='update') as h:
        h[0].header['THERMNOI'] = round(val * 0.9, 9)
    out = co.night_standard_cards(fits.Header(), sci, SB, src, flux_scale=1.0)
    hist = '\n'.join(str(x) for x in out.get('HISTORY', []))
    check('an I card other than the measurement (a stale Phase 2 card): THERMNOI the measurement, the card named in HISTORY',
          out['THERMNOI'] == val and 'the Phase 2 card on' in hist and str(round(val * 0.9, 9)) in hist.replace('\n', ''), hist)
    root2 = os.path.join(T, 'cards_none')
    shutil.rmtree(root2, ignore_errors=True)
    sci2 = subband_run(root2, {'': None})
    out = co.night_standard_cards(tmpl, sci2, SB, None, flux_scale=1.0)
    hist = '\n'.join(str(x) for x in out.get('HISTORY', []))
    check('a night without a V image that measures: no THERMNOI (the template\'s card dropped), HISTORY says why, P2RUN '
          'still stamped', 'THERMNOI' not in out and 'THERMNOI none:' in hist and out['P2RUN'] == 'Science_20260101_000000',
          (out.get('THERMNOI'), hist))
    rec = list(co.THERMNOI_RECORD)
    check('every call is recorded (sci_dir, band, THERMNOI or None, the V image, why) for the unit record',
          len(rec) == 5 and rec[0][2] == float(round(val / 0.8, 9)) and rec[1][2] is None and rec[-1][2] is None
          and rec[-1][4], rec)

    class _U:
        def __init__(self):
            self.notes = []

        def note(self, t):
            self.notes.append(t)
    from exopipe.phase3.lib.masters import admit_stages as AS
    co.THERMNOI_RECORD.extend(rec[-1:])            # a B-05 pass repeats a night: named once
    u = _U()
    AS.note_thermnoi(u, co, '2025-01-01')
    check('note_thermnoi: each night without THERMNOI named once on the unit record, with why; the record cleared',
          len(u.notes) == 2 and all('no THERMNOI for Science_20260101_000000 50MHz' in n for n in u.notes)
          and not co.THERMNOI_RECORD, u.notes)
    flat = np.full((4, 4), 0.5)
    aff = flat + np.arange(16).reshape(4, 4) * 1e-3
    check('applied_flat_scale: the product of flat factors (a factor |s| <= 0.1 divides by 1, as the division does); None for '
          'an affine or a missing map', co.applied_flat_scale(flat, np.full((4, 4), 2.0)) == 1.0
          and co.applied_flat_scale(flat, np.full((4, 4), 0.05)) == 0.5 and co.applied_flat_scale(aff, flat) is None
          and co.applied_flat_scale(None, flat) is None)


def test_bake():
    """shflux.bake: under 'full' a noise card follows its pixels (THERMNOI and IMGRMS x SHSCALE x FLUXCORR, rounded as Phase 2
    rounds, a HISTORY line); under 'production' both cards are kept as they were (production's bake)."""
    from exopipe.phase3.lib.masters import coadd as co, deep_stages as DS, flux_correction as FC, shflux as SH
    row = DS.scale_row('13h', '78MHz')
    corr_t = FC.load_table(CFG.masters_fluxscale.FLUXCORR_TABLE)
    f = float(row['scale']) * float(corr_t['78MHz']['corr'])
    h0 = fits.Header()
    h0['THERMNOI'] = (0.048456518, TN.COMMENT_SUBBAND)
    h0['IMGRMS'] = (0.1234567891, 'the image rms')
    d = np.ones((8, 8), np.float32)
    keep = co.HEADER_STANDARD
    try:
        co.HEADER_STANDARD = 'production'
        _d, hp = SH.bake(d, h0.copy(), '78MHz', row, corr_t)
        co.HEADER_STANDARD = 'full'
        _d, hf = SH.bake(d, h0.copy(), '78MHz', row, corr_t)
    finally:
        co.HEADER_STANDARD = keep
    hist = '\n'.join(str(x) for x in hf.get('HISTORY', []))
    check("bake under 'production': THERMNOI and IMGRMS kept (production's bake)", hp['THERMNOI'] == 0.048456518
          and hp['IMGRMS'] == 0.1234567891 and 'noise cards' not in '\n'.join(str(x) for x in hp.get('HISTORY', [])))
    check("bake under 'full': THERMNOI and IMGRMS x SHSCALE x FLUXCORR with the pixels, comments kept, a HISTORY line",
          hf['THERMNOI'] == float(round(0.048456518 * float(row['scale']) * float(corr_t['78MHz']['corr']), 9))
          and hf['IMGRMS'] == float(round(0.1234567891 * float(row['scale']) * float(corr_t['78MHz']['corr']), 9))
          and hf.comments['THERMNOI'] == TN.COMMENT_SUBBAND and 'noise cards THERMNOI, IMGRMS x' in hist,
          (hf.get('THERMNOI'), hf.get('IMGRMS'), f, hist))
    # the flux re-bake (rebake_file) moves the pixels by s_new / s_old x FLUXCORR: the noise cards follow them under 'full'
    s_old = round(float(row['scale']) * 1.02, 6)
    fr = s_old and (float(row['scale']) / s_old) * float(corr_t['78MHz']['corr'])
    got = {}
    for std in ('production', 'full'):
        src = os.path.join(T, f'rebake_{std}.fits')
        hd = fits.Header()
        hd['SHSCALE'] = s_old
        hd['THERMNOI'] = (0.048456518, TN.COMMENT_SUBBAND)
        hd['IMGRMS'] = (0.1234567891, 'the image rms')
        fits.PrimaryHDU(np.ones((8, 8), np.float32), hd).writeto(src, overwrite=True)
        try:
            co.HEADER_STANDARD = std
            r = SH.rebake_file(src, src, '78MHz', s_old, dict(row, window=row.get('window') or 'dec 20-60 (test row)'), corr_t,
                               write=True)
        finally:
            co.HEADER_STANDARD = keep
        got[std] = (r['status'], fits.getheader(src))
    hp, hf = got['production'][1], got['full'][1]
    check("rebake_file under 'full': THERMNOI and IMGRMS x the re-bake factor with the pixels; under 'production' kept",
          got['production'][0] == got['full'][0] == 'REBAKED' and hp['THERMNOI'] == 0.048456518 and hp['IMGRMS'] == 0.1234567891
          and hf['THERMNOI'] == float(round(0.048456518 * fr, 9)) and hf['IMGRMS'] == float(round(0.1234567891 * fr, 9)),
          (got['production'][0], got['full'][0], hp.get('THERMNOI'), hf.get('THERMNOI'), fr))


# ---------------------------------------------------------------------------------------------------------------------
# (3) the wiring
# ---------------------------------------------------------------------------------------------------------------------

def test_wiring():
    from exopipe.phase3.lib.masters import coadd as co, deep_inplace as IP, deep_stages as DS
    b = CFG.masters_build
    ids = CFG.ruled('masters_build.HEADER_STANDARD')
    with open(CFG.paths.FIX_LEDGER, newline='', encoding='utf-8', errors='replace') as f:
        rd = csv.reader(f)
        h = next(rd)
        led = {r[h.index('fix_id')] for r in rd if len(r) > h.index('fix_id')}
    check("masters_build.HEADER_STANDARD is 'full', ruled, and its ledger IDs are rows of the fix ledger",
          b.HEADER_STANDARD == 'full' and ids and set(ids) <= led, (b.HEADER_STANDARD, ids, sorted(set(ids or ()) - led)))
    par = open(os.path.join(REPO, 'exopipe', 'phase3', 'parity', 'production_20261013.cfg')).read()
    check("the parity file keeps production's headers (masters_build.HEADER_STANDARD = 'production')",
          "\nmasters_build.HEADER_STANDARD = 'production'\n" in par)
    src = inspect.getsource(IP)
    check('the build sets coadd.HEADER_STANDARD from the key (deep_inplace attribute map)',
          "'HEADER_STANDARD': 'attr:coadd.HEADER_STANDARD'" in src and co.HEADER_STANDARDS == ('full', 'production'))
    s = inspect.getsource(DS.inplace)
    check('the build refuses an unknown value (a typo never builds production\'s headers silently)',
          'b.HEADER_STANDARD not in co.HEADER_STANDARDS' in s and 'raise Refusal' in s.split('b.HEADER_STANDARD not in')[1][:200])
    sn = inspect.getsource(co.subtract_night)
    check("subtract_night stamps the night's cards under 'full' only, after the second-pass guard, with the flat factor its "
          "pixels are divided by, on the header every product of the night is written with",
          "if HEADER_STANDARD == 'full':" in sn and 'epoch_hdr = night_standard_cards(epoch_hdr, sci_dir, fl, align_img_path,' in sn
          and 'flux_scale=applied_flat_scale(first_scale, scale2_full)' in sn
          and sn.index('scale2-degenerate') < sn.index('night_standard_cards(') < sn.index('save_epoch_residual('))
    from exopipe.phase3.lib.masters import admit_stages as AS
    fin = inspect.getsource(co.run_final_lst_freq)
    ad = inspect.getsource(AS)
    check("both writers pass the night's first-pass scale map: the build (params['scale_map']) and the admission "
          "(pa['scale_map'])", "first_scale=params['scale_map']" in fin and "first_scale=pa['scale_map']" in ad)
    bs = inspect.getsource(DS.build)
    ai = inspect.getsource(AS.admit_inputs)
    check("the V images a THERMNOI may be measured on (thermnoi.candidates) are inputs of the build's and the admission's units "
          "under 'full'; the nights without one are named (note_thermnoi)", 'TN.candidates(_sd, sc)' in bs
          and 'TN.candidates(sd, sc)' in ai and 'note_thermnoi(u, co,' in bs and 'note_thermnoi(u, co, d)' in ad)
    import subprocess
    sci = subband_run(os.path.join(T, 'ephem'), {'_dewarped': 0.040})
    code = ("import sys; sys.path.insert(0, %r); from astropy.coordinates import solar_system_ephemeris as S; e0 = S.get(); "
            "from exopipe.phase3.lib.masters import thermnoi as TN; v = TN.measure(%r, '50MHz')[0]; "
            "print(e0, S.get(), v is not None)") % (REPO, sci)
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=600)
    check("a measurement leaves astropy's solar-system ephemeris as it was (Phase 2's import switches it to de432s)",
          r.returncode == 0 and r.stdout.split()[-3:] == ['builtin', 'builtin', 'True'], (r.stdout[-300:], r.stderr[-300:]))


# ---------------------------------------------------------------------------------------------------------------------
# (4) real data, read only
# ---------------------------------------------------------------------------------------------------------------------

def _night(cell, date):
    """(sci_dir, the I alignment image) of an installed night (its weights.csv row; the build's finders)."""
    from exopipe.phase3.lib.masters import coadd as co
    w = os.path.join(CFG.paths.COADD_SB_ROOT, cell, 'coadd_final', 'weights.csv')
    if not os.path.exists(w):
        return None, None
    r = next((x for x in csv.DictReader(open(w)) if x['date'] == date), None)
    if r is None:
        return None, None
    lst, band = cell.split('_')
    fs = band.replace('MHz', '')
    i = (co.find_subband_alignment_image(r['sci_dir'], int(fs)) if co.is_subband(fs)
         else co.find_wideband_alignment_image(r['sci_dir'], fs, lst, date))
    return r['sci_dir'], i


def test_real():
    for cell, date, kind in (('09h_73MHz', '2024-12-18', 'card'), ('09h_Red', '2024-12-18', 'card'),
                             ('11h_Green', '2024-12-18', 'stale'), ('02h_55MHz', '2025-01-01', 'none')):
        sci, i = _night(cell, date)
        if not sci or not i:
            check(f'real night {cell} {date}: on this node', False, (sci, i))
            continue
        val, vimg, why = TN.measure(sci, cell.split('_')[1])
        card = fits.getheader(i).get('THERMNOI')
        if kind == 'card':
            check(f"real night {cell} {date}: the measurement equals the Phase 2 card ({card})", val == card, (val, card, vimg))
        elif kind == 'stale':
            vc = fits.getheader(vimg).get('THERMNOI') if vimg else None
            check(f"real night {cell} {date} (an I image re-made after Phase 2 stamped it): the measurement equals the V card "
                  f"({vc}), not the I card ({card})", val == vc and val != card, (val, vc, card))
        else:
            check(f'real night {cell} {date} (its only V deep file is empty): no THERMNOI, with the reason', val is None and why
                  and card is None, (val, vimg, why, card))
    # a night Phase 2 did not form Full on ('Wideband bands SKIPPED (incomplete subband set): ['Full', 'Green']') although the
    # bug fixed 2026-09-05 left a Full image (Red + Blue): Phase 2 stamped nothing there, and the measurement gives none
    sci, i = _night('00h_Full', '2025-09-02')
    if sci and i:
        val, vimg, why = TN.measure(sci, 'Full')
        check('real night 00h Full 2025-09-02 (Phase 2 skipped Full; a Full image exists, without a card): no THERMNOI, '
              'with the reason', fits.getheader(i).get('THERMNOI') is None and val is None and 'Phase 2 did not form Full' in why
              and 'Full' in TN.phase2_skipped(sci), (val, vimg, why))


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_measure()
        test_cards()
        test_bake()
        test_wiring()
        test_real()
    finally:
        if not os.environ.get('EXOPIPE3_TEST_KEEP'):
            shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
