#!/usr/bin/env python3
"""CANONICAL PB correction of ALL Phase-3 deep masters (Gregg ruling
2026-07-11: PB correction moves into the main pipeline; the time-integrated
PB library is the single canonical method per PB_CORRECTION_HANDOFF.md §3;
the snapshot pb_correct products are superseded and RE-DERIVED here).

For every wideband cell ($P3_COADD_ROOT/{lst}_{band}) and every
subband cell (/data07/gh/phase3/coadd_sb/{lst}_{sb}), for every sane deep
master variant (I Taper/NoTaper x R0/R075 + V Taper_R0):
  - derive the 60-min time-integrated PB (pb_apply.wideband_pb: QA-weighted
    subband combination for bands, stored deep map for subbands),
  - write {stem}_pbcorr_{scope}.fits (intrinsic Jy/beam; overwrites the
    superseded snapshot-corrected file if present),
  - write the cell's PB map ONCE as coadd_final/PB_deep_{scope}.fits
    (the mosaic handoff artifact).
Provenance: PBCORR/PBFLOOR/PBLIBV (pb_apply) + PBMETHOD/PBWINDOW/BEAMMOD.
Skips corrupt masters (maxabs > 1e6) and logs them.
Usage: python phase3_pbcorr_masters.py [wideband|subband|all]
"""
from exopipe.common.roots import coadd_root
import sys
_ARGV = list(sys.argv); sys.argv = ['pbmasters']
import os, re, glob
import numpy as np
from astropy.io import fits
from exopipe.phase3 import pb_apply as PB
sys.argv = _ARGV              # restore after import-protection clobber

# v2-rebuild campaign (Gregg close-out 2026-08-29): PB siblings must be
# regenerated for the rebuilt cell roots; override via the same env var the
# search fleet uses; NO wideband default since 2026-09-04
WB_ROOT = os.environ.get('P3_COADD_ROOT')   # NO default (Gregg 2026-09-04); coadd_root() at use
SB_ROOT = os.environ.get('P3_COADD_SB_ROOT', '/data07/gh/phase3/coadd_sb')
VARIANTS = ['I_deep_Taper_R0', 'I_deep_Taper_R075', 'I_deep_NoTaper_R0',
            'I_deep_NoTaper_R075', 'V_deep_Taper_R0']
BEAMMOD = 'OVRO-LWA_MROsoil_updatedheight.h5'

def do_cell(root, cell):
    lst, scope = cell.split('_', 1)
    cf = f'{root}/{cell}/coadd_final'
    if not os.path.isdir(cf):
        return 0, 0, []
    pb_map = None
    n_ok, n_skip, errs = 0, 0, []
    for var in VARIANTS:
        src = f'{cf}/{var}_{scope}.fits'
        if not os.path.exists(src):
            continue
        try:
            a = fits.getdata(src)
            mx = float(np.nanmax(np.abs(a)))
            if not np.isfinite(mx) or mx > 1e6:
                errs.append(f'{cell} {var}: CORRUPT maxabs={mx:.3g} — skipped')
                n_skip += 1
                continue
            if pb_map is None:
                pb_map, meta = PB.wideband_pb(scope, lst, src, category='deep')
                ph = fits.getheader(src)
                for k in ('PBCORR',):
                    ph.pop(k, None)
                ph['PBLIBV'] = ('v1-2026-07-04', 'PB library version')
                ph['PBMETHOD'] = ('time-integrated-library', 'canonical (2026-07-11)')
                ph['PBWINDOW'] = ('60min-deep', 'PB integration window')
                ph['BEAMMOD'] = (BEAMMOD, 'beam model')
                fits.writeto(f'{cf}/PB_deep_{scope}.fits',
                             pb_map.astype(np.float32), ph, overwrite=True)
            out = f'{cf}/{var}_pbcorr_{scope}.fits'
            PB.apply_pbcorr(src, pb_map, out, meta=meta)
            with fits.open(out, mode='update') as h:
                h[0].header['PBMETHOD'] = ('time-integrated-library',
                                           'canonical (2026-07-11); snapshot superseded')
                h[0].header['PBWINDOW'] = ('60min-deep', 'PB integration window')
                h[0].header['BEAMMOD'] = (BEAMMOD, 'beam model')
            n_ok += 1
        except Exception as e:
            errs.append(f'{cell} {var}: {e}')
    return n_ok, n_skip, errs

def main():
    MODE = _ARGV[1] if len(_ARGV) > 1 else 'all'
    tot_ok = tot_skip = 0
    all_errs = []
    cells = []
    if MODE in ('wideband', 'all'):
        cells += [(coadd_root(), c) for c in sorted(os.listdir(coadd_root()))
                  if re.match(r'^\d\dh_(Red|Green|Blue|Full)$', c)]
    if MODE in ('subband', 'all'):
        cells += [(SB_ROOT, c) for c in sorted(os.listdir(SB_ROOT))
                  if re.match(r'^\d\dh_\d+MHz$', c)]
    print(f'{len(cells)} cells ({MODE})', flush=True)
    for root, cell in cells:
        ok, sk, errs = do_cell(root, cell)
        tot_ok += ok; tot_skip += sk; all_errs += errs
        print(f'  {cell}: {ok} pbcorr' + (f' ({sk} corrupt-skipped)' if sk else ''),
              flush=True)
    for e in all_errs:
        print('  !!', e, flush=True)
    print(f'PBCORR-MASTERS-DONE ok={tot_ok} corrupt_skipped={tot_skip} '
          f'errors={len(all_errs) - tot_skip}', flush=True)


if __name__ == '__main__':
    main()
