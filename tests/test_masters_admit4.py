"""D-ADMIT (a), slice 4 (plan/wave2/DADMIT_DESIGN.md): the admitted night's products in the installed generation.

So far: shflux.bake - the body of shflux.apply's loop as one function (an admitted night's products are baked with it) -
writes exactly what the loop wrote: the old body is pinned here and both are run on a real product's header with synthetic
pixels, compared byte for byte. apply itself is covered by the parity chains (the fluxscale stage's gen_shflux tree, P5).
Run: python tests/test_masters_admit4.py
"""
import filecmp
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
os.environ.setdefault('EXOPIPE3_SITE', '/lustre/gh/main/exopipe_ops/exopipe3_site.py')

from exopipe.phase3.core import env as ENV                          # noqa: E402
ENV.legacy()
from exopipe.phase3.config import CFG                               # noqa: E402

T = '/tmp/exopipe3_tests_admit4'
FAILS, N = [], [0]


def check(name, ok, detail=''):
    N[0] += 1
    print(('PASS  ' if ok else 'FAIL  ') + name + ('' if ok else f'  ({detail})'))
    if not ok:
        FAILS.append(name)


def _old_bake_body(d, hd, scope, row, corr_t):
    """shflux.apply's loop body as it was before shflux.bake (2026-10-15), pinned here."""
    from exopipe.phase3.lib.masters import flux_correction as FC
    scale = float(row['scale']); corr = float(corr_t[scope]['corr'])
    d = (np.asarray(d, dtype=np.float32) * np.float32(scale * corr))
    for _c, _v, _com in FC.cards_for(scope, corr_t, baked=True):
        hd[_c] = (_v, _com)
    _vlssr = str(row['srcs']).startswith('VLSSr')
    hd['SHSCALE'] = (scale, 'flux scale applied (VLSSr/RCB per LST, 2026-09-07)' if _vlssr
                     else 'SH12 absolute flux scale applied (Gregg 2026-08-28)')
    hd['SHERR'] = (float(row['err']), '1-sigma uncertainty of SHSCALE')
    hd['SHNSRC'] = (int(row['nsrc']), 'VLSSr matches in the window' if _vlssr else 'SH calibrators used (el > 50 deg)')
    hd['SHSRCS'] = (row['srcs'], 'scale reference (window)' if _vlssr else 'SH calibrators used')
    hd['SHMETHOD'] = ('VLSSr-dec20-60' if _vlssr else 'SH12-calibrators', 'flux scale method')
    hd['SHFREQ'] = (float(row['freq_hz']), f"model frequency Hz ({row['freq_src'][:40]})")
    hd['SHDATE'] = (row['measured_at'], 'flux scale measured')
    hd['HISTORY'] = (f"shflux: pixels x {scale:.5f} (VLSSr/RCB median ratio per LST, {row['srcs']}, N {row['nsrc']}; R-FLUX-SCALE-APPLY-VLSSR-HIGH-ELEVATION-APPROVED 2026-09-07)" if _vlssr else
                     f"shflux: pixels x {scale:.5f} (SH12 calibrators {row['srcs']}; I Taper R-0.75 pbcorr master; stage A of MASTER_SHFLUX_VLSSR_DEWARP_SPEC_20260828)")
    hd['HISTORY'] = f"fluxcorr: pixels also x FLUXCORR {corr:.5f} (k x g, {corr_t[scope]['version'][:24]})"
    return d, hd


def test_bake_equivalence():
    from astropy.io import fits
    from exopipe.phase3.lib.masters import deep_stages as DS, flux_correction as FC, shflux as SH
    lst, sc = '13h', '78MHz'
    row = DS.scale_row(lst, sc)
    corr_t = FC.load_table(CFG.masters_fluxscale.FLUXCORR_TABLE)
    hd0 = fits.getheader(os.path.join(CFG.paths.COADD_SB_ROOT, '13h_78MHz', 'coadd_provisional', 'I_deep_Taper_R075_78MHz.fits'))
    rng = np.random.default_rng(4)
    data = rng.normal(0, 1, (64, 64)).astype(np.float32)
    data[3, 5] = np.nan
    a = _old_bake_body(data, hd0.copy(), sc, row, corr_t)
    b = SH.bake(data, hd0.copy(), sc, row, corr_t)
    os.makedirs(T, exist_ok=True)
    pa, pb = os.path.join(T, 'old.fits'), os.path.join(T, 'new.fits')
    fits.writeto(pa, a[0], a[1], overwrite=True)
    fits.writeto(pb, b[0], b[1], overwrite=True)
    check('shflux.bake = the old loop body, byte for byte (13h 78 MHz scale row, a real header, synthetic pixels)',
          filecmp.cmp(pa, pb, shallow=False) and b[1]['FLUXBAKE'] is True, (row.get('scale'), corr_t.get(sc)))


def main():
    shutil.rmtree(T, ignore_errors=True)
    try:
        test_bake_equivalence()
    finally:
        shutil.rmtree(T, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
