"""Conformant names of the deep search frames (Gregg 2026-10-16 "Approved. Please make all filesnames conformant.";
plan/rename/RENAME_PLAN_DEEP_20261016.md, step 2): the archived frame header (lib/search/deep_search._archive_frame_header)
carries its residual's flux-scale and master-dewarp provenance - every SH*, FLUX* and MDW* card of the residual - and the
name tokens they back, as the epochs' names do (shflux: SHSCALE; mdewarp: MDWARP); the 10-minute cadence is unchanged;
the PB-corrected sibling's name follows; the name parses and round-trips in the grammar, and the ops finder
(gate_inputs.deep_target_beam, ops f3e8919) finds a frame so named.
Run: python tests/test_frame_names.py
"""
import importlib.util
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
from exopipe.phase3.core import products as P                       # noqa: E402
from exopipe.phase3.lib.search import deep_search as DSR            # noqa: E402

T = '/tmp/exopipe3_tests_frame_names'
FAILS, N = [], [0]
GATE_INPUTS = '/lustre/gh/main/exopipe_ops/gate_inputs.py'
PROV = {'SHSCALE': (0.97886, 'flux scale applied'), 'SHERR': (0.0276, 'sigma'), 'SHNSRC': (782, 'matches'),
        'SHSRCS': ('VLSSr dec +20..+60 deg', 'window'), 'SHMETHOD': ('VLSSr-dec20-60', 'method'), 'SHFREQ': (75410000.0, 'Hz'),
        'SHDATE': ('2026-10-01T19:29:21Z', 'measured'), 'FLUXBND': (0.905218, 'k'), 'FLUXABS': (1.034018, 'g'),
        'FLUXCORR': (0.936012, 'k x g'), 'FLUXVER': ('fluxcorr-v2-20261001', 'version'), 'FLUXBAKE': (True, 'baked'),
        'MDWARP': (True, 'final master dewarp applied'), 'MDWMETH': ('rbf-multiquadric', 'method'), 'MDWCV': (48.2, 'arcsec'),
        'MDWDATE': ('2026-10-09T18:08:35Z', 'measured')}


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({str(detail)[:400]})'), flush=True)
    if not ok:
        FAILS.append(name)


def wcs_header():
    h = fits.Header()
    for k, v in (('NAXIS', 2), ('NAXIS1', 64), ('NAXIS2', 64), ('CTYPE1', 'RA---SIN'), ('CTYPE2', 'DEC--SIN'),
                 ('CRVAL1', 105.0), ('CRVAL2', 37.2), ('CRPIX1', 32.5), ('CRPIX2', 32.5), ('CDELT1', -0.03125),
                 ('CDELT2', 0.03125), ('CUNIT1', 'deg'), ('CUNIT2', 'deg')):
        h[k] = v
    return h


def residual(base, d, stokes, scope, cards, lstbin=None):
    h = wcs_header()
    h['DATE-OBS'] = '2024-12-18T08:04:14.000'
    h['DATE-END'] = '2024-12-18T09:04:05.000'
    h['INTTIME'] = 3591.0
    h['BMAJ'], h['BMIN'], h['BPA'] = 0.40, 0.30, 10.0
    if lstbin is not None:
        h['LSTBIN'] = lstbin
    for k in cards:
        h[k] = PROV[k]
    p = os.path.join(base, 'subtracted', f'{d}_{stokes}_sub_{scope}.fits')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fits.PrimaryHDU(np.zeros((64, 64), np.float32), h).writeto(p, overwrite=True)
    return p


def frame_header(case, cards, cadence='deep', stokes='V', scope='18MHz'):
    base = os.path.join(T, case)
    shutil.rmtree(base, ignore_errors=True)
    d = '2024-12-18'
    residual(base, d, stokes, scope, cards, lstbin=3 if cadence == '10min' else None)
    mf = os.path.join(base, 'master.fits')
    mh = fits.Header()
    mh['NSTACK'], mh['COADDOP'], mh['MJD-MID'], mh['CONSTDIG'] = 9, 'ivw-mean-v3', 60700.0, 'abc123'
    fits.PrimaryHDU(np.zeros((4, 4), np.float32), mh).writeto(mf, overwrite=True)
    return DSR._archive_frame_header(wcs_header(), base, d, scope, stokes, 1.0, 0.4, 0.02, 0.03, mf, None,
                                     os.path.join(base, 'align.fits'), 'P3S5', '07h 18MHz', cadence=cadence)


def gate_inputs():
    sp = importlib.util.spec_from_file_location('gate_inputs_t', GATE_INPUTS)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def main():
    shutil.rmtree(T, ignore_errors=True)
    os.makedirs(T)
    all_cards = list(PROV)
    h = frame_header('dewarped_cell', all_cards)
    n = str(h['PRODNAME'])
    want = P.build_name('18MHz', 'V', 'deep', 'Taper', 0.0, proc=['dewarped', 'aligned', 'msub', 'ssub', 'shflux', 'mdewarp'],
                        lst='07h', epoch='2024-12-18T08:04:14.000')
    check("a deep frame of a dewarped, flux-scaled residual: PRODNAME carries shflux and mdewarp (canonical order, as the "
          "epoch's name)", n == want and '_Robust+0.0_shflux_mdewarp_dewarped_aligned_msub_ssub_LST07h_' in n, (n, want))
    miss = [k for k in all_cards if h.get(k) != PROV[k][0]]
    check("the frame carries every SH*, FLUX* and MDW* card of its residual, with its value", not miss, miss)
    g = P.parse_name(n)
    check('the name parses in the phase3 grammar and round-trips', g['grammar'] == 'phase3' and P.build_name(**g) == n, g)
    h2 = frame_header('plain_cell', [k for k in all_cards if not k.startswith('MDW')])
    n2 = str(h2['PRODNAME'])
    check('a residual with no master dewarp (MDWARP absent): shflux only, no MDW* card',
          '_Robust+0.0_shflux_dewarped_aligned_msub_ssub_' in n2 and not any(k.startswith('MDW') for k in h2), n2)
    h3 = frame_header('unscaled', [])
    n3 = str(h3['PRODNAME'])
    check('a residual without SHSCALE: no shflux token, no provenance card (the name states only what was applied)',
          '_Robust+0.0_dewarped_aligned_msub_ssub_' in n3 and not any(k in h3 for k in PROV), n3)
    h4 = frame_header('tenmin', all_cards, cadence='10min')
    n4 = str(h4['PRODNAME'])
    check('the 10-minute cadence is unchanged (its rename comes later): no new token, no residual provenance copied',
          'shflux' not in n4 and 'mdewarp' not in n4 and not any(k in h4 for k in PROV if k not in ('FLUXDC',)), n4)
    hi = frame_header('stokes_i', all_cards, stokes='I')
    ni = str(hi['PRODNAME'])
    check('Stokes I: Robust -0.75 with the same tokens', '_I_deep_Taper_Robust-0.75_shflux_mdewarp_dewarped_aligned_msub_ssub_'
          in ni, ni)
    sib = n.replace('_dewarped_', '_pbcorr_dewarped_', 1)       # the sibling rule (deep_search, the PB-corrected copy)
    gs = P.parse_name(sib)
    check('the PB-corrected sibling\'s name (the frame\'s with pbcorr) is canonical: shflux, mdewarp, pbcorr, dewarped',
          gs['grammar'] == 'phase3' and P.build_name(**gs) == sib and '_shflux_mdewarp_pbcorr_dewarped_' in sib, sib)
    # the ops finder finds the frame so named, ignores its sibling, and refuses an old-named frame beside it
    gi = gate_inputs()
    root = os.path.join(T, 'archive')
    dd = os.path.join(root, '07h', '2024-12-18', gi.ARUN, '18MHz', 'V', 'deep')
    os.makedirs(dd)
    hh = fits.Header()
    hh['BMAJ'] = 0.4321
    fits.PrimaryHDU(np.zeros((4, 4), np.float32), hh).writeto(os.path.join(dd, n))
    fits.PrimaryHDU(np.zeros((4, 4), np.float32), hh).writeto(os.path.join(dd, sib))
    gi.AROOT = root
    b = gi.deep_target_beam('07h', '18MHz', '2024-12-18', 'V')
    check('gate_inputs.deep_target_beam finds the frame by its conformant name (the sibling ignored)', b == 0.4321, b)
    fits.PrimaryHDU(np.zeros((4, 4), np.float32), hh).writeto(os.path.join(dd, n.replace('_shflux_mdewarp_', '_')))
    gi._DB.clear()
    try:
        gi.deep_target_beam('07h', '18MHz', '2024-12-18', 'V')
        refused = False
    except RuntimeError as e:
        refused = 'one frame per night' in str(e)
    check('an old-named frame beside the renamed one: the finder refuses (one frame per night)', refused)
    print(f'\n{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
