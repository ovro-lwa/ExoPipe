"""B-03 (a), same-state (Gregg 2026-10-13; the same-state reading 2026-10-14): one input resolver for the masters build.

core.inputs' copy of the co-add's five image finders must choose exactly what the finders choose (randomized Phase 2
listings here; every real deep cell's nights in plan/parity_scripts/b03_resolver_equality.py). lib/masters/input_resolver
puts the copy in the co-add's place and applies the same-state rule: a night's V or I-variant image whose dewarp state
differs from its survey image is refused (the finder returns None) and recorded once in <cell>/refusals_input.csv; the
per-night product test and T6 accept the missing V products of exactly those nights; the record has a content test. Each
check fails if the copy chose another file, the rule refused a matching product or admitted a mismatched one, a refusal
was recorded twice or not at all, 'coadd-finders' did not restore the co-add's own finders, a changed finder was used, or
an unrefused night's missing V product passed.
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
        for _ in range(3):                                    # the co-add asks again (provisional, final, residuals)
            quiet(co.find_stokes_v_deep_image, sd['2025-01-01'], '23')
        rows = IR.read_refusals(cell)
        check('refusals_input.csv: one row per refused (night, product), never twice',
              sorted((r['date'], r['product']) for r in rows) == [('2025-01-01', 'I_Taper_R0'), ('2025-01-01', 'V'),
                                                                   ('2025-01-03', 'I_Taper_R0'), ('2025-01-03', 'V')], rows)
        IR._WRITTEN.clear()
        IR._LOADED.clear()                                    # a new process: the file's rows are read, not repeated
        quiet(co.find_stokes_v_deep_image, sd['2025-01-01'], '23')
        check('refusals_input.csv: a later process does not repeat a row', len(IR.read_refusals(cell)) == 4)
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


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_copy_equals_finders()
        test_same_state()
        test_night_checks()
        test_real_preview()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
