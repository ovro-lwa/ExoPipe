"""B-03 (a), same-state (Gregg 2026-10-13; the same-state reading 2026-10-14): one input resolver for the masters build.

core.inputs' copy of the co-add's five image finders must choose exactly what the finders choose (randomized Phase 2
listings here; every real deep cell's nights in plan/parity_scripts/b03_resolver_equality.py). lib/masters/input_resolver
puts the copy in the co-add's place and applies the same-state rule: a night's V or I-variant image whose dewarp state
differs from its survey image is refused (the finder returns None) and recorded once in <cell>/refusals_input.csv; the
per-night product test and T6 accept the missing V products of exactly those nights; the record has a content test. Each
check fails if the copy chose another file, the rule refused a matching product or admitted a mismatched one, a refusal
was recorded twice or not at all, 'coadd-finders' did not restore the co-add's own finders, a changed finder was used, or
an unrefused night's missing V product passed.
Steps 3-4: a product on another grid than its night's I image is aligned from its own WCS (PRODUCT_GRID_MISMATCH 'regrid'; a
point source lands on its sky position, which the I image's mapping misses); a post-cutover survey image off the template
grid is regridded at every regrid decision, not refused (OFF_GRID_SURVEY 'regrid'); the four wrong-grid nights' cells refuse
until data_decisions.csv excludes the night.
Run: python tests/test_masters_b03.py
"""
import contextlib
import csv
import io
import os
import random
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from astropy.io import fits                                          # noqa: E402
from exopipe.phase3.config import CFG                                # noqa: E402
from exopipe.phase3.core import inputs as I                          # noqa: E402
from exopipe.phase3.core.record import Refusal                       # noqa: E402
from exopipe.phase3.lib.masters import coadd as co                   # noqa: E402
from exopipe.phase3.lib.masters import input_resolver as IR          # noqa: E402

T = '/tmp/exopipe3_tests_masters_b03'
FAILS, N = [], [0]
CANDIDATES_I = [f'{sb}-I-{tp}-3600s-Robust-{r}-image{sfx}.fits'
                for sb in ('23MHz',) for tp in ('Taper', 'NoTaper') for r in ('0.75', '0')
                for sfx in ('', '_dewarped', '_dewarped_v2', '-pbcorr', '_pbcorr_dewarped')]
CANDIDATES_I += ['23MHz-I-Taper-3600s-Robust-0.75-model.fits', '23MHz-I-Taper-3600s-Robust-0.75-dewarped-image.fits',
                 'Blue-I-Taper-3600s-Robust-0.75-image_dewarped.fits', 'Blue-I-Taper-3600s-Robust-0.75-image.fits',
                 'Blue-I-NoTaper-3600s-Robust-0-image.fits', 'Blue-I-Taper-3600s-Robust-0-image_dewarped.fits']
CANDIDATES_V = [f'23MHz-V-{tp}-3600s-Robust-{r}-image{sfx}.fits' for tp in ('Taper', 'NoTaper') for r in ('0.75', '0')
                for sfx in ('', '_dewarped', '-pbcorr')]


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def night(root, lst, date, files, sci='Science_20260101'):
    """A Phase 2 night <root>/<lst>/<date>/<sci>/<scope>/<S>/deep/<file> with empty files; returns the Science dir."""
    sd = os.path.join(root, lst, date, sci)
    for rel in files:
        p = os.path.join(sd, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'w').close()
    return sd


def quiet(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a)
    return r, buf.getvalue()


# ---------------------------------------------------------------------------------------------------------------------
# 1. the copy chooses what the finders choose (randomized listings)
# ---------------------------------------------------------------------------------------------------------------------

def test_copy_equals_finders():
    rng = random.Random(20261014)
    root = os.path.join(T, 'eq')
    bad, n = [], 0
    for k in range(300):
        files = ([f'23MHz/I/deep/{f}' for f in rng.sample(CANDIDATES_I[:20], rng.randint(0, 8))]
                 + [f'23MHz/V/deep/{f}' for f in rng.sample(CANDIDATES_V, rng.randint(0, 5))]
                 + [f'Blue/I/deep/{f}' for f in rng.sample(CANDIDATES_I[20:], rng.randint(0, 4))]
                 + [f'Blue/V/deep/{f.replace("23MHz", "Blue")}' for f in rng.sample(CANDIDATES_V, rng.randint(0, 3))])
        sd = night(root, '05h', f'2025-01-{k:03d}', files)
        pairs = [('survey sb', quiet(co.find_subband_alignment_image, sd, 23)[0], I.survey_subband(sd, 23)[0]),
                 ('survey wb', quiet(co.find_wideband_alignment_image, sd, 'Blue', '05h', 'x')[0], I.survey_wideband(sd, 'Blue')[0]),
                 ('V sb', co.find_stokes_v_deep_image(sd, '23'), I.stokes_v_image(sd, '23')),
                 ('V wb', co.find_stokes_v_deep_image(sd, 'Blue'), I.stokes_v_image(sd, 'Blue'))]
        for v in I.VARIANTS:
            pairs.append((f'{v} sb', co.find_subband_variant_image(sd, '23', v), I.variant_subband(sd, '23', v)))
            pairs.append((f'{v} wb', co.find_wideband_deep_images(sd, 'Blue', 'I', v), I.variant_wideband(sd, 'Blue', 'I', v)))
        for name, a, b in pairs:
            n += 1
            if a != b:
                bad.append((k, name, a, b))
    check(f'core.inputs chooses the finders\' file on 300 random listings ({n} lookups, both outcomes exercised)', not bad, bad[:3])
    check('finder pins hold for the co-add module', I.pin_mismatches(open(co.__file__).read()) == [])
    check('is_subband: config subbands = coadd.SUBBANDS', sorted(int(s[:-3]) for s in CFG.instrument.ALL_SUBBANDS) == sorted(co.SUBBANDS))


# ---------------------------------------------------------------------------------------------------------------------
# 2. the same-state rule through the installed wrappers
# ---------------------------------------------------------------------------------------------------------------------

S_DEW = '23MHz/I/deep/23MHz-I-Taper-3600s-Robust-0.75-image_dewarped.fits'
S_RAW = '23MHz/I/deep/23MHz-I-Taper-3600s-Robust-0.75-image.fits'
V_DEW = '23MHz/V/deep/23MHz-V-Taper-3600s-Robust-0-image_dewarped.fits'
V_RAW = '23MHz/V/deep/23MHz-V-Taper-3600s-Robust-0-image.fits'
R0_DEW = '23MHz/I/deep/23MHz-I-Taper-3600s-Robust-0-image_dewarped.fits'
R0_RAW = '23MHz/I/deep/23MHz-I-Taper-3600s-Robust-0-image.fits'
B_S_DEW = 'Blue/I/deep/Blue-I-Taper-3600s-Robust-0.75-image_dewarped.fits'
B_R0_RAW = 'Blue/I/deep/Blue-I-Taper-3600s-Robust-0-image.fits'


def test_same_state():
    root = os.path.join(T, 'p2')
    out = os.path.join(T, 'out')
    for d in (root, out):
        shutil.rmtree(d, ignore_errors=True)
    nights = {'2025-01-01': [S_DEW, V_RAW, R0_RAW],            # raw products on a dewarped survey -> refused
              '2025-01-02': [S_RAW, V_RAW, R0_RAW],            # all raw -> kept
              '2025-01-03': [S_RAW, V_DEW, R0_DEW],            # dewarped products on a raw survey -> refused
              '2025-01-04': [S_DEW, V_DEW, R0_DEW, B_S_DEW, B_R0_RAW]}   # matched subband; a raw wideband variant
    sd = {d: night(root, '05h', d, f) for d, f in nights.items()}
    cell = os.path.join(out, '05h_23MHz')
    os.makedirs(cell)
    os.makedirs(os.path.join(out, '05h_Blue'))
    orig_wd = co._update_workdir
    co._update_workdir(out)
    try:
        own = {n: getattr(co, n) for n in IR.FINDERS}
        # an unknown value, and the rule without the resolver, refuse
        for args, why in ((('core', 'same-state'), 'INPUT_RESOLVER'), (('core.inputs', 'strict'), 'RAW_INPUT_FLOW_RULE'),
                          (('coadd-finders', 'same-state'), 'needs')):
            try:
                IR.install(co, *args)
                ok = False
            except Refusal as e:
                ok = why in str(e)
            check(f'install{args} refuses ({why})', ok)
        IR.install(co, 'core.inputs', 'same-state')
        check('core.inputs: the five co-add finders are the wrappers', all(getattr(co, n) is IR._WRAPPERS[n] for n in IR.FINDERS))
        got = {d: (quiet(co.find_stokes_v_deep_image, s, '23')[0], quiet(co.find_subband_variant_image, s, '23', 'Taper_R0')[0])
               for d, s in sd.items()}
        check('same-state: raw V / variant on a dewarped survey refused (None)', got['2025-01-01'] == (None, None), got['2025-01-01'])
        check('same-state: raw V / variant on a raw survey kept', got['2025-01-02'] == (
            os.path.join(sd['2025-01-02'], V_RAW), os.path.join(sd['2025-01-02'], R0_RAW)), got['2025-01-02'])
        check('same-state: dewarped V / variant on a raw survey refused', got['2025-01-03'] == (None, None), got['2025-01-03'])
        check('same-state: dewarped products on a dewarped survey kept',
              got['2025-01-04'] == (os.path.join(sd['2025-01-04'], V_DEW), os.path.join(sd['2025-01-04'], R0_DEW)))
        wb, txt = quiet(co.find_wideband_deep_images, sd['2025-01-04'], 'Blue', 'I', 'Taper_R0')
        check('same-state: a raw wideband variant on a dewarped wideband survey refused, printed',
              wb is None and 'INPUT REFUSED 05h Blue 2025-01-04 I_Taper_R0' in txt, (wb, txt))
        sd5 = night(root, '05h', '2025-01-05', [S_DEW, B_S_DEW, 'Blue/V/deep/Blue-V-Taper-3600s-Robust-0-image.fits',
                                               '23MHz/I/deep/23MHz-I-NoTaper-3600s-Robust-0-image.fits',
                                               '23MHz/I/deep/23MHz-I-NoTaper-3600s-Robust-0.75-image_dewarped.fits'])
        r5 = (quiet(co.find_stokes_v_deep_image, sd5, 'Blue')[0], quiet(co.find_subband_variant_image, sd5, '23', 'NoTaper_R0')[0],
              quiet(co.find_subband_variant_image, sd5, '23', 'NoTaper_R075')[0])
        check('same-state: a raw wideband V and a raw NoTaper R0 refused on dewarped surveys; a dewarped NoTaper R0.75 kept',
              r5 == (None, None, os.path.join(sd5, '23MHz/I/deep/23MHz-I-NoTaper-3600s-Robust-0.75-image_dewarped.fits')), r5)
        lines = []
        for _ in range(3):
            lines.append(quiet(co.find_stokes_v_deep_image, sd5, 'Blue')[1])
        check('a refusal is printed once per process, not at every lookup', sum(x.count('INPUT REFUSED') for x in lines) == 0
              and r5[0] is None, lines)
        for _ in range(3):                                    # the co-add asks again (provisional, final, residuals)
            quiet(co.find_stokes_v_deep_image, sd['2025-01-01'], '23')
        rows = IR.read_refusals(cell)
        check('refusals_input.csv: one row per refused (night, product), never twice',
              sorted((r['date'], r['product']) for r in rows) == [('2025-01-01', 'I_Taper_R0'), ('2025-01-01', 'V'),
                                                                   ('2025-01-03', 'I_Taper_R0'), ('2025-01-03', 'V'),
                                                                   ('2025-01-05', 'I_NoTaper_R0')], rows)
        check('refusals_input.csv: the wideband cell\'s own record (05h_Blue) holds its refusals',
              sorted((r['date'], r['product']) for r in IR.read_refusals(os.path.join(out, '05h_Blue')))
              == [('2025-01-04', 'I_Taper_R0'), ('2025-01-05', 'V')], IR.read_refusals(os.path.join(out, '05h_Blue')))
        IR._WRITTEN.clear()
        IR._LOADED.clear()                                    # a new process: the file's rows are read, not repeated
        quiet(co.find_stokes_v_deep_image, sd['2025-01-01'], '23')
        check('refusals_input.csv: a later process does not repeat a row', len(IR.read_refusals(cell)) == 5)
        # the content test of the record
        from exopipe.phase3.core import validate
        orig_root = CFG.paths.PHASE2_ROOT
        CFG.authorize([], out=T)
        CFG.set('paths.PHASE2_ROOT', root)
        try:
            res = validate.run('input_refusals', IR.refusals_path(cell))
            check('input_refusals content test: every row holds on the Phase 2 tree', all(t['passed'] for t in res), res)
            open(os.path.join(sd['2025-01-01'], V_DEW), 'w').close()       # the night's V dewarped later
            res2 = validate.run('input_refusals', IR.refusals_path(cell))
            os.remove(os.path.join(sd['2025-01-01'], V_DEW))
            check('input_refusals content test fails when the finder now chooses another image',
                  not all(t['passed'] for t in res2) and 'now chooses' in res2[0]['detail'], res2)
        finally:
            CFG.reset()
        check('the PHASE2_ROOT override was taken back', CFG.paths.PHASE2_ROOT == orig_root)
        # the survey images: chosen as before, '[failsafe]' line as the co-add prints it
        a, ta = quiet(own['find_subband_alignment_image'], sd['2025-01-02'], 23)
        b, tb = quiet(co.find_subband_alignment_image, sd['2025-01-02'], 23)
        check('survey image: the same file and the same [failsafe] line as the co-add finder', a == b and ta == tb and '[failsafe]' in ta,
              (ta, tb))
        # survey-image-flow under the resolver: every product admitted
        IR.install(co, 'core.inputs', 'survey-image-flow')
        v = quiet(co.find_stokes_v_deep_image, sd['2025-01-01'], '23')[0]
        check('survey-image-flow (production rule) under the resolver: the raw V admitted', v == os.path.join(sd['2025-01-01'], V_RAW))
        IR.install(co, 'coadd-finders', 'survey-image-flow')
        check('coadd-finders: the co-add\'s own finders restored', all(getattr(co, n) is own[n] for n in IR.FINDERS))
        # a finder whose source changed refuses the copy
        keep = dict(I.FINDER_PINS)
        I.FINDER_PINS['find_stokes_v_deep_image'] = '0' * 40
        try:
            IR.install(co, 'core.inputs', 'same-state')
            ok = False
        except Refusal as e:
            ok = 'find_stokes_v_deep_image' in str(e) and 're-port' in str(e)
        finally:
            I.FINDER_PINS.clear()
            I.FINDER_PINS.update(keep)
        check('a co-add finder changed since the copy: install refuses, naming it', ok)
        check('after the refused install the co-add\'s own finders stay', all(getattr(co, n) is own[n] for n in IR.FINDERS))
    finally:
        IR.install(co, 'coadd-finders', 'survey-image-flow')
        orig_wd(os.environ.get('P3_COADD_ROOT') or T)


# ---------------------------------------------------------------------------------------------------------------------
# 3. the per-night checks accept exactly the refused V nights
# ---------------------------------------------------------------------------------------------------------------------

def test_night_checks():
    from exopipe.phase3.core import validate
    from exopipe.phase3.lib.masters import tests as MT
    cell = os.path.join(T, 'cells', '05h_23MHz')
    shutil.rmtree(cell, ignore_errors=True)
    fd, sd = os.path.join(cell, 'coadd_final'), os.path.join(cell, 'subtracted')
    os.makedirs(fd)
    os.makedirs(sd)
    nights = ['2025-01-01', '2025-01-02', '2025-01-03']
    with open(os.path.join(fd, 'weights.csv'), 'w') as f:
        f.write('date,weight,qa3_scale,qa3_tilt,qa3_nsrc,qa3_guard,qa3_rms_mjy,sci_dir,in_master,flux_dc,qa3_scale_aligned,qa3_window\n')
        for d in nights:
            f.write(f'{d},1,1,0,30,,10,/x,1,0,1,ok\n')
    for d in nights:
        for st in ('I', 'V'):
            if st == 'V' and d == '2025-01-02':
                continue                                       # the refused night: no V products
            for kind in ('epoch', 'sub', 'clean'):
                fits.writeto(os.path.join(sd, f'{d}_{st}_{kind}_23MHz.fits'), np.zeros((2, 2), np.float32))
    probe = os.path.join(sd, '2025-01-01_I_epoch_23MHz.fits')

    def nightprods():
        MT._CACHE.clear()
        return MT.every_admitted_night_has_products(probe)
    ok0 = nightprods()
    with open(IR.refusals_path(cell), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(IR.REFUSALS_HEADER)
        w.writerow(['05h', '23MHz', '2025-01-02', 'V', 'x-image.fits', 'raw V on a dewarped survey image'])
    ok1 = nightprods()
    with open(IR.refusals_path(cell), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(IR.REFUSALS_HEADER)
        w.writerow(['05h', '23MHz', '2025-01-02', 'I_Taper_R0', 'x-image.fits', 'raw variant'])
    ok2 = nightprods()
    check('per-night products: a missing V night fails without a refusal row', not ok0[0], ok0)
    check('per-night products: ... passes when the night\'s V input is recorded as refused', ok1[0] and 'V input refused' in ok1[2], ok1)
    check('per-night products: a variant refusal does not excuse the missing V products', not ok2[0], ok2)
    # T6 of the generation checks (the epoch content check stubbed: only the V exemption is tested here)
    from exopipe.phase3.lib.masters import deep_verify as DV
    gen = os.path.join(cell, 'gen_x')
    os.makedirs(os.path.join(gen, 'coadd_final'))
    os.makedirs(os.path.join(gen, 'subtracted'))
    tok = CFG.masters_fluxscale.TOKEN
    for m in ('I_deep_Taper_R075', 'V_deep_Taper_R0'):
        fits.writeto(os.path.join(gen, 'coadd_final', DV.sname(f'{m}_23MHz.fits', '23MHz')), np.zeros((2, 2), np.float32))
    for d in nights:
        for st in ('I', 'V'):
            if not (st == 'V' and d == '2025-01-02'):
                fits.writeto(os.path.join(gen, 'subtracted', f'{d}_{st}_epoch{tok}_23MHz.fits'), np.zeros((2, 2), np.float32))
    shutil.copy(os.path.join(fd, 'weights.csv'), os.path.join(gen, 'coadd_final', 'weights.csv'))
    from exopipe.phase3.lib.masters import coadd as CFP
    keep = CFP._epoch_content_arrays
    CFP._epoch_content_arrays = lambda *a: (True, '', 1.0, 20)
    try:
        r_var = DV.t6(gen)                                     # the record names a variant only
        with open(IR.refusals_path(cell), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(IR.REFUSALS_HEADER)
            w.writerow(['05h', '23MHz', '2025-01-02', 'V', 'x-image.fits', 'raw V on a dewarped survey image'])
        r_v = DV.t6(gen)
    finally:
        CFP._epoch_content_arrays = keep
    check('T6: an in-master night without its V epoch fails unless its V input was refused', not r_var[0] and r_v[0], (r_var, r_v))


# ---------------------------------------------------------------------------------------------------------------------
# 4. real cells (read only): the dry-run preview equals the measured state table
# ---------------------------------------------------------------------------------------------------------------------

def test_real_preview():
    table = '/lustre/gh/main/exopipe_ops/refactor_20261012/plan/parity/b03_input_states_20261014.csv'
    rows = list(csv.DictReader(open(table)))
    for cell in ('11h_Red', '23h_Green', '13h_69MHz', '23h_23MHz'):
        lst, sc = cell.split('_')
        fq = sc[:-3] if sc.endswith('MHz') else sc
        dates = quiet(co.discover_dates, lst, fq)[0]
        pv = IR.preview(lst, sc, dates)
        want = {}
        for r in rows:
            if r['lst'] == lst and r['scope'] == sc and r['state'] not in ('none', '') and r['state'] != r['survey']:
                want.setdefault(r['product'], []).append(r['date'])
        check(f'{cell}: the preview refuses the measured mismatches ({sum(len(v) for v in want.values())} night-products)',
              {k: sorted(v) for k, v in pv.items()} == {k: sorted(v) for k, v in want.items()}, (pv, want))


# ---------------------------------------------------------------------------------------------------------------------
# 5. step 3: a product on another grid than its night's I image is regridded from its own WCS
# ---------------------------------------------------------------------------------------------------------------------

def _wcs_image(path, n, cdelt, src=None):
    """An n x n image centred on (150, 40) deg with pixel size cdelt deg; a 1 Jy point source at sky position src."""
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---SIN', 'DEC--SIN']
    w.wcs.crval = [150.0, 40.0]
    w.wcs.crpix = [n / 2 + 0.5, n / 2 + 0.5]
    w.wcs.cdelt = [-cdelt, cdelt]
    h = w.to_header()
    h['BMAJ'], h['BMIN'], h['BPA'] = 2 * cdelt, 2 * cdelt, 0.0
    d = np.zeros((n, n), np.float32)
    if src is not None:
        x, y = w.all_world2pix(src[0], src[1], 0)
        d[int(round(float(y))), int(round(float(x)))] = 1.0
    fits.writeto(path, d, h, overwrite=True)
    return w, h


def test_product_grid():
    d = os.path.join(T, 'grid')
    os.makedirs(d, exist_ok=True)
    src = (152.0, 41.5)
    wt, th = _wcs_image(os.path.join(d, 'tmpl.fits'), 64, 0.5)              # the master grid
    _wcs_image(os.path.join(d, 'I.fits'), 48, 0.6)                          # the night's I image (pre-cutover grid)
    _wcs_image(os.path.join(d, 'V.fits'), 40, 0.75, src)                    # its V on a third grid
    _wcs_image(os.path.join(d, 'V_same.fits'), 48, 0.6, src)                # a V on the I image's grid
    i_map = co.precompute_regrid_mapping(os.path.join(d, 'I.fits'), th, (64, 64))
    xe, ye = (float(v) for v in wt.all_world2pix(src[0], src[1], 0))
    keep = co.PRODUCT_GRID_MISMATCH
    res = {}
    try:
        for mode in ('drop', 'regrid'):
            co.PRODUCT_GRID_MISMATCH = mode
            for v in ('V', 'V_same'):
                out = os.path.join(d, f'{v}_{mode}_aligned.fits')
                m = co._product_mapping(os.path.join(d, f'{v}.fits'), os.path.join(d, 'I.fits'), i_map)
                quiet(co.process_variant_image, os.path.join(d, f'{v}.fits'), None, np.ones((64, 64)), th, (64, 64), out, m)
                a = fits.getdata(out)
                yp, xp = np.unravel_index(int(np.nanargmax(a)), a.shape)
                res[(mode, v)] = (m is i_map, round(float(np.hypot(xp - xe, yp - ye)), 1))
    finally:
        co.PRODUCT_GRID_MISMATCH = keep
    check('drop (production): a V on a third grid gets the I image\'s mapping and lands off its sky position',
          res[('drop', 'V')][0] and res[('drop', 'V')][1] > 3, res)
    check('regrid: that V is regridded from its own WCS and lands on its sky position (within 1 px)',
          not res[('regrid', 'V')][0] and res[('regrid', 'V')][1] <= 1.0, res)
    # the grid test: geometric, not strict WCS equality (152 real products differ from their survey image by < 1e-12 px)
    wt2, _h = _wcs_image(os.path.join(d, 'I_noise.fits'), 48, 0.6)
    hn = fits.getheader(os.path.join(d, 'I.fits'))
    hn['CRVAL1'] += 1e-13
    fits.writeto(os.path.join(d, 'I_noise.fits'), np.zeros((48, 48), np.float32), hn, overwrite=True)
    hs = fits.getheader(os.path.join(d, 'I.fits'))
    hs['CRPIX1'] += 1.0
    fits.writeto(os.path.join(d, 'I_shift.fits'), np.zeros((48, 48), np.float32), hs, overwrite=True)
    g = (co._same_grid(os.path.join(d, 'V_same.fits'), os.path.join(d, 'I.fits')),
         co._same_grid(os.path.join(d, 'I_noise.fits'), os.path.join(d, 'I.fits')),
         co._same_grid(os.path.join(d, 'I_shift.fits'), os.path.join(d, 'I.fits')),
         co._same_grid(os.path.join(d, 'V.fits'), os.path.join(d, 'I.fits')))
    check('grid test: same WCS, or WCS equal but for numerical noise -> one grid; the same shape one pixel off, or another '
          'shape -> another grid', g == (True, True, False, False), g)
    check('regrid: a V on its I image\'s grid keeps the I mapping (production\'s case), on its position',
          res[('regrid', 'V_same')][0] and res[('regrid', 'V_same')][1] <= 1.0 and res[('drop', 'V_same')] == res[('regrid', 'V_same')], res)
    # the call sites: the provisional V is no longer dropped under 'regrid', and every final-stage product takes _product_mapping
    import ast
    tree = ast.parse(open(co.__file__).read())
    fn = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    calls = {}
    for name in ('run_provisional_lst_freq', 'run_final_lst_freq'):
        for node in ast.walk(fn[name]):
            if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'process_variant_image':
                last = node.args[-1] if len(node.args) >= 7 else None
                src0 = getattr(node.args[0], 'id', '')
                calls.setdefault(name, []).append((src0, isinstance(last, ast.Call) and getattr(last.func, 'id', '') == '_product_mapping'))
    prov = dict(calls['run_provisional_lst_freq'])
    fin = dict(calls['run_final_lst_freq'])
    check('provisional: the night\'s V goes through _product_mapping (the reference V keeps its own mapping)',
          prov.get('v_src') is True and prov.get('ref_v_src') is False, calls['run_provisional_lst_freq'])
    check('final: V, I Taper R0 and the NoTaper variants go through _product_mapping',
          fin.get('v_img') is True and fin.get('i_r0_img') is True and fin.get('_img') is True, calls['run_final_lst_freq'])
    txt = ast.get_source_segment(open(co.__file__).read(), fn['run_provisional_lst_freq'])
    check("provisional: the V DROP needs PRODUCT_GRID_MISMATCH != 'regrid'",
          "_v_shape != _i_shape and PRODUCT_GRID_MISMATCH != 'regrid':" in txt)


# ---------------------------------------------------------------------------------------------------------------------
# 6. step 4: an off-grid post-cutover survey image is regridded, not refused; the four wrong-grid nights' gate
# ---------------------------------------------------------------------------------------------------------------------

def test_off_grid():
    from exopipe.phase3.core import exclusions, fallback
    d = os.path.join(T, 'offgrid')
    os.makedirs(d, exist_ok=True)
    on, off = os.path.join(d, 'on.fits'), os.path.join(d, 'off.fits')
    fits.writeto(on, np.zeros((8, 8), np.float32), overwrite=True)
    fits.writeto(off, np.zeros((6, 6), np.float32), overwrite=True)
    bad = os.path.join(d, 'bad.fits')
    open(bad, 'w').write('not a fits file')
    post, pre = '2025-03-01', '2024-12-20'
    keep = co.OFF_GRID_SURVEY
    try:
        co.OFF_GRID_SURVEY = 'refuse'
        r_ref = (co._off_grid(post, off, (8, 8)), co._grid_refused_dates([(post, None, off), ('2025-03-02', None, on),
                                                                          ('2025-03-03', None, on)], 'test_a'))
        co.OFF_GRID_SURVEY = 'regrid'
        r_reg = (co._off_grid(post, off, (8, 8)), co._off_grid(post, on, (8, 8)), co._off_grid(pre, off, (8, 8)),
                 co._off_grid('MASTER', off, (8, 8)),
                 co._grid_refused_dates([(post, None, off), ('2025-03-02', None, on), ('2025-03-03', None, on)], 'test_b'))
        CFG.reset()
        try:
            quiet(co._off_grid, post, bad, (8, 8))
            refused = False
        except fallback.FallbackRefused as e:
            refused = 'GRID_HEADER_FALLBACK' in str(e)
        CFG.authorize([], out=T)
        CFG.set('fallbacks.GRID_HEADER_FALLBACK', 'fallback')
        m = fallback.mark()
        counted = (quiet(co._off_grid, post, bad, (8, 8))[0], fallback.since(m)[1].get('fallbacks.GRID_HEADER_FALLBACK'))
        CFG.reset()
    finally:
        co.OFF_GRID_SURVEY = keep
    check("refuse (production): the off-grid night is refused by grid and _off_grid says nothing",
          r_ref == (False, {post}), r_ref)
    check("regrid: the off-grid post-cutover night is regridded (True), the on-grid one not, pre-cutover and MASTER by their "
          "own paths; nothing refused by grid", r_reg == (True, False, False, False, set()), r_reg)
    check('regrid: an unreadable header is the B-30 grid-header site (refuses under the ruled policy; production counts it)',
          refused and counted == (False, 1), (refused, counted))
    # every regrid decision consults _off_grid
    import re
    src = open(co.__file__).read()
    tests_ = re.findall(r'\n( *)(ref_)?needs_regrid = (?:ref_)?date < CUTOVER_DATE and[^\n]*\n( *)([^\n]*)', src)
    ok_sites = [t for t in tests_ if '_off_grid(' in t[3]]
    prep_ref = '(ref_date < CUTOVER_DATE and post_dates) or _off_grid(ref_date, ref_img_path, target_shape)' in src
    check(f'every cutover regrid decision is followed by _off_grid ({len(ok_sites)} of {len(tests_)}; prep reference too)',
          len(tests_) == 8 and len(ok_sites) == 8 and prep_ref and src.count('CUTOVER_DATE and') == 9, (len(tests_), len(ok_sites)))
    # the wrong-grid gate, from the data: an ADMITTED post-cutover night off the cell's modal grid refuses the cell
    imgs = {}
    for k_, n_ in enumerate(['2025-03-01', '2025-03-02', '2025-03-03', '2025-03-04']):
        p_ = os.path.join(d, f'g{k_}.fits')
        fits.writeto(p_, np.zeros((6, 6) if k_ == 3 else (8, 8), np.float32), overwrite=True)
        imgs[n_] = p_
    pre_img = os.path.join(d, 'pre.fits')
    fits.writeto(pre_img, np.zeros((5, 5), np.float32), overwrite=True)
    entries = [(n_, '/x', p_) for n_, p_ in imgs.items()] + [('2024-12-20', '/x', pre_img)]
    off = IR.off_grid_nights(entries, co.CUTOVER_DATE)
    CFG.reset()
    g_on = IR.wrong_grid_refusal('05h', '23MHz', entries, co.CUTOVER_DATE)
    g_ex = IR.wrong_grid_refusal('05h', '23MHz', [x for x in entries if x[0] != '2025-03-04'], co.CUTOVER_DATE)
    CFG.authorize([], out=T)
    CFG.set('masters_build.OFF_GRID_SURVEY', 'refuse')
    g_prod = IR.wrong_grid_refusal('05h', '23MHz', entries, co.CUTOVER_DATE)
    CFG.reset()
    keep = co.OFF_GRID_SURVEY
    co.OFF_GRID_SURVEY = 'refuse'
    try:
        prod_set = quiet(co._grid_refused_dates, entries, 'test_c')[0]
    finally:
        co.OFF_GRID_SURVEY = keep
    check('wrong-grid gate: the off-grid nights are exactly the ones production\'s grid test refuses (a pre-cutover night is '
          'regridded by design, not counted)', off == [('2025-03-04', (6, 6), (8, 8))] and {x[0] for x in off} == prod_set,
          (off, prod_set))
    check('wrong-grid gate (regrid): the cell refuses while the off-grid night is admitted, naming it; once it is excluded '
          '(not among the admitted nights) the cell passes', g_on is not None and '2025-03-04 (6x6)' in g_on
          and 'data_decisions' in g_on and g_ex is None, (g_on, g_ex))
    check("wrong-grid gate: none under OFF_GRID_SURVEY 'refuse' (production: the grid refusal keeps the night out)", g_prod is None)
    # real data (read only): the admitted nights off their cell's grid today - the four ruled 'leave excluded' and four more
    expected = {('02h', 'Green', '2025-12-26'), ('12h', 'Full', '2025-04-11'), ('14h', 'Full', '2025-05-07'),
                ('19h', 'Full', '2025-04-05'), ('02h', '50MHz', '2025-12-26'), ('11h', 'Full', '2025-03-09'),
                ('11h', 'Full', '2025-04-07'), ('22h', 'Full', '2025-08-08')}
    found, admitted_off, ex = set(), set(), exclusions.load()
    before = lambda l_, s_, d_: (l_, s_, d_) not in expected and ex.is_excluded(l_, s_, d_)   # data_decisions before the
    for lst, sc in sorted({(l_, s_) for l_, s_, _n in expected} | {('23h', '23MHz')}):       # eight rows of 2026-10-16
        fq = sc[:-3] if sc.endswith('MHz') else sc
        allk = quiet(co.discover_dates, lst, fq)[0]
        found |= {(lst, sc, n_) for n_, _x, _m in IR.off_grid_nights([x for x in allk if not before(lst, sc, x[0])],
                                                                      co.CUTOVER_DATE)}
        adm = [x for x in allk if not ex.is_excluded(lst, sc, x[0])]
        admitted_off |= {(lst, sc, n_) for n_, _x, _m in IR.off_grid_nights(adm, co.CUTOVER_DATE)}
    check('real: with data_decisions as it stood before 2026-10-16 the gate finds the eight admitted off-grid nights (the four '
          'ruled "leave excluded" and the four the review found), and nothing in 23h 23MHz', found == expected,
          sorted(found ^ expected))
    check('real: since 2026-10-16 all eight are excluded in data_decisions (clv15 df3242b, Gregg\'s permission) - no admitted '
          'night is off its grid', not admitted_off and all(ex.is_excluded(*k) for k in expected), sorted(admitted_off))


# ---------------------------------------------------------------------------------------------------------------------
# 7. the review fixes (2026-10-15)
# ---------------------------------------------------------------------------------------------------------------------

def test_review_fixes():
    import ast
    import re
    from exopipe.phase3.lib.search import deep_search as DSR
    d = os.path.join(T, 'review')
    os.makedirs(d, exist_ok=True)
    # a product below its minimum after the rule: the build says so (instead of failing the content tests unexplained)
    cell = os.path.join(d, '23h_Green')
    os.makedirs(os.path.join(cell, 'coadd_final'))
    fits.writeto(os.path.join(cell, 'coadd_final', 'I_deep_Taper_R0_Green.fits'), np.zeros((2, 2), np.float32))
    short = IR.masters_below_minimum(cell, 'Green', {'V': ['2025-08-09'], 'I_Taper_R0': ['2025-08-09']}, 3, 2)
    check('a product master missing after same-state refusals is named (V), one still built is not (I Taper R0)',
          len(short) == 1 and short[0].startswith('V_deep_Taper_R0_Green not built') and 'fewer than 3' in short[0], short)
    # unknown grid-key values refuse
    class B:
        PRODUCT_GRID_MISMATCH, OFF_GRID_SURVEY = 'regird', 'refuse'
    try:
        IR.check_grid_values(B)
        ok = False
    except Refusal as e:
        ok = 'PRODUCT_GRID_MISMATCH' in str(e)
    B.PRODUCT_GRID_MISMATCH = 'drop'
    IR.check_grid_values(B)
    check('a misspelt PRODUCT_GRID_MISMATCH / OFF_GRID_SURVEY refuses (any other value would silently mean production)', ok)
    # the pins on edited copies of the module: a body edit, a decorator, a new finder
    t = open(co.__file__).read()
    body = t.replace("    for f in files:                                  # failsafe: RAW -> flow dewarps\n        if base(f) and 'dewarped' not in f:\n            return os.path.join(d, f)\n    return None\n\n\ndef find_10min_images",
                     "    for f in files:                                  # failsafe: RAW -> flow dewarps\n        if base(f):\n            return os.path.join(d, f)\n    return None\n\n\ndef find_10min_images", 1)
    deco = t.replace('\ndef find_stokes_v_deep_image(', '\n@functools.lru_cache(None)\ndef find_stokes_v_deep_image(')
    newf = t + '\n\ndef find_extra_image(sci_dir):\n    return None\n'
    pins = (I.pin_mismatches(t), I.pin_mismatches(body), I.pin_mismatches(deco), I.pin_mismatches(newf))
    check('pins: the module as is passes; an edited finder body, a decorator and a new finder each refuse, named',
          pins[0] == [] and body != t and pins[1] == ['find_stokes_v_deep_image'] and pins[2] == ['find_stokes_v_deep_image (decorated)']
          and pins[3] == ['find_extra_image (a new finder)'], pins)
    # the search: a V-refused night keeps its I search (the provenance gate skips only the V product, by the record)
    rec = os.path.join(d, 'srch')
    os.makedirs(rec)
    with open(os.path.join(rec, 'refusals_input.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(IR.REFUSALS_HEADER)
        w.writerow(['11h', 'Red', '2024-12-18', 'V', 'x.fits', 'raw V on a dewarped survey image'])
        w.writerow(['11h', 'Red', '2024-12-19', 'I_Taper_R0', 'y.fits', 'raw I_Taper_R0 on a dewarped survey image'])
    src = open(DSR.__file__).read()
    gate = src[src.index('_vref = _same_state_refused_v(base)'):src.index("print(f'  {lst} {scope}: INPUT-PROVENANCE GATE")]
    skip = gate.index("if _st == 'V' and _kd(d) in _vref and not os.path.exists(f'{ebase}/subtracted/{d}_V_sub_{scope}.fits'):")
    check('search: the record\'s V nights only (a variant refusal is not one), and none without a record',
          DSR._same_state_refused_v(rec) == {'2024-12-18'} and DSR._same_state_refused_v(d) == set())
    check('search provenance gate: a recorded V refusal with no V product skips the V check before the header read (the '
          'night stays for I; the V search has no frame of it)', 0 < skip < gate.index('_h = fits.getheader('), skip)
    # the variants stage does not stamp a same-state refusal as a variant-gate one
    check('variants stage: the record answers per night and product', co._same_state_refused(rec, '2024-12-19', 'I_Taper_R0')
          and not co._same_state_refused(rec, '2024-12-19', 'V') and not co._same_state_refused(d, '2024-12-18', 'V'))
    csrc = open(co.__file__).read()
    rv = csrc[csrc.index('def rebuild_variants_from_record('):csrc.index('def run_final_lst_freq(')]
    check('variants stage: a same-state refusal is skipped before it could be appended to the variant-gate refusals',
          0 < rv.index("if not img and _same_state_refused(workdir, date, f'I_{vk}'):") < rv.index("refused.append((date, f'no {vk} input'))"))
    # call sites: each _product_mapping gets the product and its night's I image; each _off_grid the survey image
    tree = ast.parse(csrc)
    pm, og = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == '_product_mapping':
            pm.append(tuple(ast.get_source_segment(csrc, a_) for a_ in node.args))
        if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == '_off_grid':
            og.append(ast.get_source_segment(csrc, node.args[1]))
    check('_product_mapping call sites: (the product, the night\'s I image, its mapping) - never the product twice',
          sorted(pm) == sorted([('v_src', 'align_img_path', 'regrid_coords'), ('v_img', '_i_img', "params['regrid']"),
                                ('i_r0_img', '_i_img', "params['regrid']"), ('_img', '_i_img', "params['regrid']")]), pm)
    check('_off_grid call sites: the survey image (or the variant image in the variants stage), never a regridded work path',
          sorted(set(og)) == sorted({'align_img_path', 'ref_date_entry[2]', 'ref_img_path', 'img'}) and len(og) == 9
          and not any('work' in x or 'regrid' in x for x in og), og)
    # install carries the record with its generation (quarantining the cell's earlier one)
    from exopipe.phase3.lib.install import steps_deep as ISD
    moved, recs = [], []

    class FakePI:
        @staticmethod
        def _move_to_wq(cell_, path, rel):
            moved.append((cell_, os.path.basename(path), rel))
            os.remove(path)

        @staticmethod
        def rec(cell_, step, path='', dest='', m='', nbytes='', detail=''):
            recs.append((step, os.path.basename(dest)))

        @staticmethod
        def md5(p_):
            return 'md5'
    rb, cd = os.path.join(d, 'rebuild', '11h_Red'), os.path.join(d, 'working', '11h_Red')
    for x in (rb, cd):
        os.makedirs(x)
    open(os.path.join(cd, 'refusals_input.csv'), 'w').write('old\n')
    shutil.copy(os.path.join(rec, 'refusals_input.csv'), os.path.join(rb, 'refusals_input.csv'))

    class Plan:
        rebuild_root, cdir, cell = rb, cd, '11h_Red'
    got = ISD.carry_input_refusals(FakePI, Plan)
    check('install: the rebuild\'s record replaces the working cell\'s (the old one quarantined, the copy logged)',
          got == os.path.join(cd, 'refusals_input.csv') and moved == [('11h_Red', 'refusals_input.csv', 'refusals_input.csv')]
          and recs == [('cell-file', 'refusals_input.csv')] and open(got).read() == open(os.path.join(rb, 'refusals_input.csv')).read(),
          (got, moved, recs))
    Plan.rebuild_root = None
    check('install: nothing happens without a rebuild root (an install of the working tree\'s own generation)',
          ISD.carry_input_refusals(FakePI, Plan) is None and len(moved) == 1)
    # status: a night whose V input was refused is complete without a V frame
    from exopipe.phase3.lib.status import steps_coverage as COV
    row = dict(excluded=0, epoch_archived_I=1, frame_archived_I=1, frame_archived_V='refused', master_exists=1, residual=1,
               partial_exclusion=0)
    check("status: 'refused' V frame counts as complete (ok), a missing one does not",
          COV.census_class(row) == 'ok' and COV.census_class(dict(row, frame_archived_V=0)) == 'recover-local')
    # the flow step installs the resolver and the off-grid rule as the build does
    from exopipe.phase3.lib.flow import steps_deep as FSD
    CFG.reset()
    notes = []
    CO = FSD.coadd_module(None, notes.append)
    flow_ok = (CO.find_stokes_v_deep_image is IR._WRAPPERS['find_stokes_v_deep_image'] and CO.OFF_GRID_SURVEY == 'regrid'
               and any('B-03' in n for n in notes))
    IR.install(co, 'coadd-finders', 'survey-image-flow')
    co.OFF_GRID_SURVEY = 'refuse'
    check('flow step: coadd_module installs the resolver and sets OFF_GRID_SURVEY under the ruled values, with a note', flow_ok, notes)
    # the dry run's upper bound on 23h Green (read only): V keeps 3 admitted nights, exactly the minimum before any check
    exr = __import__('exopipe.phase3.core.exclusions', fromlist=['load']).load()
    adm = [x for x in quiet(co.discover_dates, '23h', 'Green')[0] if not exr.is_excluded('23h', 'Green', x[0])]
    left = IR.nights_left('23h', 'Green', adm)
    check('dry-run counts on 23h Green: each product keeps 3 of 5 admitted nights (the V master then needs all 3 in the master)',
          len(adm) == 5 and left == {'V': 3, 'I_Taper_R0': 3, 'I_NoTaper_R0': 3, 'I_NoTaper_R075': 3}, (len(adm), left))


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_copy_equals_finders()
        test_same_state()
        test_night_checks()
        test_real_preview()
        test_product_grid()
        test_off_grid()
        test_review_fixes()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
