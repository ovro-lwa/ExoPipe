"""D-ADMIT, DECISIONS_PENDING item 7: admission into a master-dewarped cell (lib/masters/admit.mdwarp_screen and the admit
stage's gates, provisional-master resample, product cards and admissions column).

A night is aligned to the dewarped master directly by its flow (the reference copy is the installed master), so no screen
touches its products; its stage-2 number is measured against the build's provisional masters, which the dewarp did not
resample, so the stage resamples them with the cell's archived screen first. The screen is the one of the archived screen
record whose md5, anchor count and measurement time are the master's (MDWNANC, MDWDATE): any other refuses.
Data-free: synthetic masters, screens and records under /tmp/exopipe3_tests_admit_mdwarp (removed at the end).
Run: python tests/test_masters_admit_mdwarp.py
"""
import csv
import hashlib
import inspect
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from astropy.io import fits                                         # noqa: E402

from exopipe.phase3.core import env as ENV                          # noqa: E402
ENV.legacy()
from exopipe.phase3.core.record import Refusal                      # noqa: E402
from exopipe.phase3.lib.dewarp import master_dewarp as MD           # noqa: E402
from exopipe.phase3.lib.masters import admit as A                   # noqa: E402
from exopipe.phase3.lib.masters import admit_stages as AS           # noqa: E402

FAILS, N = [], [0]
T = '/tmp/exopipe3_tests_admit_mdwarp'
CELL, SHAPE, WHEN = '07h_Red', (40, 48), '2026-09-09T00:00:00'
RCOLS = ['lst', 'scope', 'applied', 'status', 'n_anchors', 'vlssr_med_before_arcmin', 'vlssr_med_after_arcmin',
         'heldout_resid_med_arcsec', 'anchor_off_med_arcsec', 'screen_p50_arcsec', 'screen_p95_arcsec', 'screen_file',
         'screen_md5', 'measured_at', 'written_utc']


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def screen(d, nanc=37, name='07h_Red_screen.fits'):
    """A screen file as the dewarp install archives it: SCREEN_X / SCREEN_Y (pixels), NANCHOR on SCREEN_X."""
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    sx, sy = 0.3 + 0.01 * xx, -0.2 + 0.005 * yy
    hx = fits.ImageHDU(sx.astype(np.float32), name='SCREEN_X')
    hx.header['NANCHOR'] = nanc
    p = os.path.join(d, name)
    fits.HDUList([fits.PrimaryHDU(), hx, fits.ImageHDU(sy.astype(np.float32), name='SCREEN_Y')]).writeto(p)
    with open(p, 'rb') as f:
        return p, hashlib.md5(f.read()).hexdigest()


def record(d, rows):
    p = os.path.join(d, 'mdewarp_screens.csv')
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, RCOLS)
        w.writeheader()
        for r in rows:
            w.writerow({**dict.fromkeys(RCOLS, ''), **r})
    return p


def row(scr_name, md5, applied='1', nanc=37, when=WHEN):
    return dict(lst='07h', scope='Red', applied=applied, status='applied' if applied == '1' else 'held', n_anchors=nanc,
                screen_file=scr_name, screen_md5=md5, measured_at=when)


def master(d, var, mdwarp=True, nanc=37, when=WHEN, med1=1.2):
    """A master with the dewarp's cards (master_dewarp._apply), or none."""
    h = fits.Header()
    h['CONSTDIG'] = 'abc123'
    if mdwarp:
        for k, v in dict(MDWARP=(True, 'final master dewarp applied (Gregg 2026-08-28)'),
                         MDWMETH=('rbf-multiquadric', 'Phase-2 RBF dewarp'), MDWINTP=(MD.RESAMPLE_METHOD, 'resampling'),
                         MDWREF=('VLSSr', 'dewarp reference'), MDWNANC=(nanc, 'RBF anchors'), MDWCV=(10.5, 'cv'),
                         MDWMED0=(3.1, 'before'), MDWMED1=(med1, 'after'), MDWP50=(40.0, 'p50'), MDWP95=(90.0, 'p95'),
                         MDWDATE=(when, 'dewarp measured'), MDWSTAT=('applied', 'status')).items():
            h[k] = v
    p = os.path.join(d, f'{var}_Red.fits')
    fits.writeto(p, np.zeros(SHAPE, np.float32), h, overwrite=True)
    return p


def test_screen():
    d = os.path.join(T, 'screen')
    os.makedirs(d)
    sp, md5 = screen(d)
    mp = master(d, 'I_deep_Taper_R075')
    rp = record(d, [row('old_screen.fits', '0' * 32, when='2026-08-01T00:00:00'), row(os.path.basename(sp), md5)])
    s = A.mdwarp_screen(CELL, mp, rp)
    h = fits.getheader(mp)
    check('a dewarped master: the record\'s last row of the cell names its screen (an earlier measurement\'s row is not read)',
          s is not None and s['path'] == sp and s['md5'] == md5 and s['sx'].shape == SHAPE and s['sy'].shape == SHAPE,
          s and (s['path'], s['md5']))
    check('the screen\'s cards are the master\'s MDW* cards, values and comments',
          s is not None and set(s['cards']) == set(A.SCREEN_CARDS)
          and all(s['cards'][k] == (h[k], h.comments[k]) for k in A.SCREEN_CARDS), s and sorted(s['cards']))
    check('a master without MDWARP has no screen (None; nothing read)',
          A.mdwarp_screen(CELL, master(d, 'plain', mdwarp=False), os.path.join(d, 'missing.csv')) is None)
    check('screen_inputs (the stage\'s --resume identity): the record and the screen its last row names; [] without MDWARP',
          A.screen_inputs(CELL, mp, rp) == [rp, sp]
          and A.screen_inputs(CELL, master(d, 'plain', mdwarp=False), rp) == [], A.screen_inputs(CELL, mp, rp))
    for sub in ('r1', 'r2', 'r3', 'r4', 'm5', 'm6'):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    for sub in ('r2', 'r3', 'r4'):                              # the record names the screen in its own directory
        shutil.copy(sp, os.path.join(d, sub, os.path.basename(sp)))
    bad = {'no record': (mp, os.path.join(d, 'missing.csv'))}
    bad['the cell\'s last row not applied'] = (mp, record(d + '/r1', [row(os.path.basename(sp), md5),
                                                                      row(os.path.basename(sp), md5, applied='0')]))
    bad['the screen file missing'] = (mp, record(d + '/r2', [row('gone.fits', md5)]))
    bad['the screen md5 not the record\'s'] = (mp, record(d + '/r3', [row(os.path.basename(sp), 'f' * 32)]))
    bad['the screen\'s anchors not the record\'s'] = (mp, record(d + '/r4', [row(os.path.basename(sp), md5, nanc=36)]))
    bad['the master\'s MDWNANC not the record\'s'] = (master(d + '/m5', 'I_deep_Taper_R075', nanc=36), rp)
    bad['the master\'s MDWDATE not the record\'s measured_at'] = (master(d + '/m6', 'I_deep_Taper_R075',
                                                                         when='2026-10-01T00:00:00'), rp)
    check('screen_inputs names only files that exist (the gate refuses the rest): a missing screen leaves the record',
          A.screen_inputs(CELL, mp, bad['the screen file missing'][1]) == [bad['the screen file missing'][1]]
          and A.screen_inputs(CELL, mp, os.path.join(d, 'missing.csv')) == [])
    for what, (m, r) in bad.items():
        try:
            A.mdwarp_screen(CELL, m, r)
            got = 'accepted'
        except Refusal as e:
            got = str(e)
        check(f'refuses: {what}', got != 'accepted', got)


def _gates(cd, ents, rec):
    """admit_gates with every gate but the dewarp's satisfied; returns (the failures, the screen)."""
    saved = (AS.prep_changed, AS.guard_median, A.mdwarp_screen, A.reference_problems)
    from exopipe.phase3.lib.flow import steps_deep as FSD
    from exopipe.phase3.lib.masters import deep_stages as DS
    saved2 = (FSD.admit_signature, DS.bake_factor)
    orig = A.mdwarp_screen
    try:
        AS.prep_changed = lambda prep: []
        AS.guard_median = lambda cd_: (0.0, 'test')
        A.mdwarp_screen = lambda cell, mp, _rec: orig(cell, mp, rec)
        A.reference_problems = lambda cd_, sc, ents_: []
        FSD.admit_signature = lambda p: 'MASTER:sig'
        DS.bake_factor = lambda lst, sc: (1.0, 'test')
        fails = []
        out = AS.admit_gates(CELL, cd, {}, {'meta': {'admit': {'reference_signature': 'MASTER:sig'}}}, ents,
                             lambda g, why: fails.append((g, why)), write=False)
    finally:
        AS.prep_changed, AS.guard_median, A.mdwarp_screen, A.reference_problems = saved
        FSD.admit_signature, DS.bake_factor = saved2
    return fails, out[4]


def _ents(cd, **kw):
    out = []
    for st, var in (('I', 'I_deep_Taper_R075'), ('V', 'V_deep_Taper_R0')):
        p = master(os.path.join(cd, 'coadd_final'), var, **kw.get(st, {}))
        h = fits.getheader(p)
        out.append(dict(stokes=st, var=var, path=p, factor=1.0, constdig='abc123', mdwarp=bool(h.get('MDWARP')),
                        data=np.zeros(SHAPE, np.float32), header=h))
    return out


def _prov(cd, mdwarp=False, shape=SHAPE):
    for _st, v in AS.PROV:
        h = fits.Header()
        if mdwarp:
            h['MDWARP'] = True
        fits.writeto(os.path.join(cd, 'coadd_provisional', f'{v}_Red.fits'), np.zeros(shape, np.float32), h,
                     overwrite=True)


def test_gates():
    d = os.path.join(T, 'gates')
    os.makedirs(d)
    sp, md5 = screen(d)
    rec = record(d, [row(os.path.basename(sp), md5)])

    def cell(name, prov=None, **kw):
        cd = os.path.join(d, name)
        for sub in ('coadd_final', 'coadd_provisional'):
            os.makedirs(os.path.join(cd, sub))
        ents = _ents(cd, **kw)
        if prov is not None:
            _prov(cd, **prov)
        return cd, ents

    cd, ents = cell('ok', prov={})
    fails, scr = _gates(cd, ents, rec)
    check('a dewarped cell with its screen and un-warped provisional masters on the screen\'s grid passes; the screen returned',
          not fails and scr is not None and scr['md5'] == md5, fails)
    cd, ents = cell('plain', prov={}, I=dict(mdwarp=False), V=dict(mdwarp=False))
    fails, scr = _gates(cd, ents, rec)
    check('a cell without the dewarp: no screen, no failure', not fails and scr is None, fails)
    cases = {
        'the V master of another dewarp measurement (MDWMED1)': cell('vmix', prov={}, V=dict(med1=1.5)),
        'the V master not dewarped': cell('vplain', prov={}, V=dict(mdwarp=False)),
        'the I master not dewarped (its V master is)': cell('iplain', prov={}, I=dict(mdwarp=False)),
        'a provisional master carrying MDWARP (it would be resampled twice)': cell('pwarp', prov=dict(mdwarp=True)),
        'a provisional master of another shape than the screen': cell('pshape', prov=dict(shape=(40, 40))),
    }
    for what, (cd, ents) in cases.items():
        fails, _ = _gates(cd, ents, rec)
        check(f'admit.mdwarp refuses: {what}', any(g == 'admit.mdwarp' for g, _w in fails), fails)
    cd, ents = cell('norec', prov={})
    fails, _ = _gates(cd, ents, os.path.join(d, 'missing.csv'))
    check('admit.mdwarp refuses a dewarped cell the screen record does not list',
          [g for g, _w in fails] == ['admit.mdwarp'] and 'no applied row' in fails[0][1], fails)


def test_stage_wiring():
    """The admit stage: the screen's record and file are inputs declared in admit_inputs (its --resume identity: an input
    added after the gates made every --resume re-run the stage - GPU test 2026-11-13), the provisional masters are resampled
    with the screen before any night, every product gets the master's MDW* cards and ADMWARP after ADMITTED/ADMITREF, the row records the
    screen's md5 (ADMISSION_COLS screen_md5); the dry-run plan takes the same gates."""
    src = inspect.getsource(AS.admit_stage)
    i_g = src.index('admit_gates(cell, cd, prep, frec, ents, fail, write=True)')
    i_rs = src.index("MD.resample(prov_d[st], scr['sx'], scr['sy'])")
    i_loop = src.index('for d, sd, img in todo:')
    i_ref = src.index("hd['ADMITREF']")
    i_cards = src.index("for kc, vc in scr['cards'].items()")
    i_w = src.index("hd['ADMWARP']")
    check('admit_stage: gates, provisional resample, then the nights; cards after ADMITREF',
          i_g < i_rs < i_loop < i_ref < i_cards < i_w, (i_g, i_rs, i_loop, i_ref, i_cards, i_w))
    check('the screen inputs are declared with the stage\'s others (admit_inputs, the --resume identity), never added '
          'after the gates', 'A.screen_inputs(cell, A.master_path(cd), _cfg().masters_verify.DEWARP_SCREEN_RECORD)'
          in inspect.getsource(AS.admit_inputs) and 'u.add_input' not in src[i_g:])
    check('admissions.csv carries screen_md5, filled from the screen', 'screen_md5' in AS.ADMISSION_COLS
          and "screen_md5=scr['md5'] if scr else ''" in src)
    check('the dry-run plan takes the stage\'s gates (five values)',
          '_refs, med, med_src, _factor, _scr = admit_gates(' in inspect.getsource(AS.plan_admit))
    check('the old blanket refusal is gone', 'not written until the GPU test' not in inspect.getsource(AS.admit_gates))
    # the resample puts a provisional master in the dewarped master's frame: a source of the provisional master lands
    # where the dewarped master has it (master_dewarp.resample is the dewarp's own call)
    p = np.zeros(SHAPE)
    p[20, 24] = 1.0
    sx, sy = np.full(SHAPE, 2.0), np.full(SHAPE, -1.0)
    w = MD.resample(p, sx, sy)
    check('master_dewarp.resample moves a provisional master\'s source as the screen moved the master\'s '
          '(output[y, x] = input[y + sy, x + sx])', np.unravel_index(np.nanargmax(w), SHAPE) == (21, 22),
          np.unravel_index(np.nanargmax(w), SHAPE))


def test_content_tests():
    """admitted_epoch's admitted_cards (the header standard's own tests need a real product): a product of a dewarped cell carries the master's MDW* cards and ADMWARP 'flow'; a product of a cell
    without the dewarp carries none of them."""
    from exopipe.phase3.core import validate
    for name, warped in (('07h_Red', True), ('08h_Red', False)):
        cd = os.path.join(T, 'content', name)
        for sub in ('coadd_final', 'subtracted'):
            os.makedirs(os.path.join(cd, sub))
        mp = master(os.path.join(cd, 'coadd_final'), 'I_deep_Taper_R075', mdwarp=warped)
        mh = fits.getheader(mp)
        cards = {k: (mh[k], mh.comments[k]) for k in A.SCREEN_CARDS if k in mh}

        def product(fn, extra):
            h = fits.Header()
            h['ADMITTED'] = True
            h['ADMITREF'] = 'abc123'
            for k, v in extra.items():
                h[k] = v
            p = os.path.join(cd, 'subtracted', fn)
            fits.writeto(p, np.zeros(SHAPE, np.float32), h, overwrite=True)
            return [t['passed'] for t in validate.run('admitted_epoch', p) if t['name'].endswith('admitted_cards')] == [True]
        full = dict(cards, ADMWARP=('flow', 'test'))
        if warped:
            check('admitted_epoch passes a dewarped cell\'s product with the master\'s MDW* cards and ADMWARP',
                  product('2025-01-01_I_epoch_Red.fits', full))
            check('admitted_epoch fails one without the cards', not product('2025-01-02_I_epoch_Red.fits', {}))
            check('admitted_epoch fails one with another MDWDATE',
                  not product('2025-01-03_I_epoch_Red.fits', dict(full, MDWDATE=('2026-10-01T00:00:00', 'x'))))
        else:
            check('admitted_epoch passes a product without the dewarp in a cell without it',
                  product('2025-01-01_I_epoch_Red.fits', {}))
            check('admitted_epoch fails a product carrying MDW* cards in a cell without the dewarp',
                  not product('2025-01-02_I_epoch_Red.fits', dict(MDWARP=(True, 'x'), ADMWARP=('flow', 'x'))))


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    try:
        test_screen()
        test_gates()
        test_stage_wiring()
        test_content_tests()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
