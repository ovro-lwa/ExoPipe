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
  - write the cell's PB maps, ONE PER ROBUST WEIGHTING (PB step 2, 2026-10-01): coadd_final/PB_deep_{scope}.fits with the
    I Taper Robust 0 beam (Robust 0 products) and coadd_final/PB_deep_R075_{scope}.fits with the survey master's beam
    (Robust -0.75 products); each sibling divides by the map of its own robust weighting.
Provenance: PBCORR/PBFLOOR/PBLIBV (pb_apply) + PBMETHOD/PBWINDOW/BEAMMOD.
Skips corrupt masters (maxabs > 1e6) and logs them.
Usage: python phase3_pbcorr_masters.py [wideband|subband|all]
"""
from exopipe.phase3.config import coadd_root
import sys
_ARGV = list(sys.argv); sys.argv = ['pbmasters']
import os, re, glob
import numpy as np
from astropy.io import fits
from exopipe.phase3.lib.masters import pb_apply as PB
sys.argv = _ARGV              # restore after import-protection clobber

# v2-rebuild campaign (Gregg close-out 2026-08-29): PB siblings must be
# regenerated for the rebuilt cell roots; override via the same env var the
# search fleet uses; NO wideband default since 2026-09-04
WB_ROOT = os.environ.get('P3_COADD_ROOT')   # NO default (Gregg 2026-09-04); coadd_root() at use
SB_ROOT = os.environ.get('P3_COADD_SB_ROOT', '/data07/gh/phase3/coadd_sb')
VARIANTS = ['I_deep_Taper_R0', 'I_deep_Taper_R075', 'I_deep_NoTaper_R0',
            'I_deep_NoTaper_R075', 'V_deep_Taper_R0',
            # the cleaned Stokes V master (vclean_masters; the Stokes V mosaic input) divides by the same Robust 0 map: left
            # out until 2026-10-03, so PB step 2's map rewrite (2026-10-01) left 19 of its siblings stale by up to 5.6 per
            # cent (ledger RESULT-FLUX-REBAKE-CONTENT-TEST; Gregg "preamble approved")
            'V_deep_Taper_R0_clean']
BEAMMOD = 'OVRO-LWA_MROsoil_updatedheight.h5'

def do_cell(root, cell):
    lst, scope = cell.split('_', 1)
    cf = f'{root}/{cell}/coadd_final'
    if not os.path.isdir(cf):
        return 0, 0, []
    # ONE PB MAP PER ROBUST WEIGHTING (PB step 2, Gregg 2026-10-01 "go on step 2"): the bandwidth-smearing factor depends on
    # the image's beam, which the robust weighting sets (taper barely changes it). PB_deep_<scope>.fits keeps its meaning -
    # built with the I Taper Robust 0 master's beam, for every Robust 0 product (I Taper/NoTaper R0, V Taper R0); NEW
    # PB_deep_R075_<scope>.fits is built with the survey master's (I Taper Robust -0.75) beam for the Robust -0.75 products
    # (I Taper/NoTaper R075). The pair test is flat with the survey beam on the current catalogues (2026-10-01). A missing
    # beam source for a robust refuses that robust's siblings (never a fallback to the other map).
    maps = {}
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
            rob = 'R075' if var.endswith('R075') else 'R0'
            if rob not in maps:
                bsrc = f'{cf}/I_deep_Taper_{rob}_{scope}.fits'
                if not os.path.exists(bsrc):
                    raise RuntimeError(f'no {os.path.basename(bsrc)} to take the Robust {rob} beam from - REFUSING')
                if fits.getheader(bsrc).get('MDWARP'):
                    raise RuntimeError('master-dewarped cell: its PB maps must carry the cell\'s dewarp screen and are made '
                                       'by the dewarp install, not here - REFUSING')
                pb_map, meta = PB.wideband_pb(scope, lst, bsrc, category='deep')
                ph = fits.getheader(bsrc)
                for k in ('PBCORR',):
                    ph.pop(k, None)
                ph['PBLIBV'] = ('v1-2026-07-04', 'PB library version')
                ph['PBMETHOD'] = ('time-integrated-library', 'canonical (2026-07-11)')
                ph['PBWINDOW'] = ('60min-deep', 'PB integration window')
                ph['BEAMMOD'] = (BEAMMOD, 'beam model')
                ph['PBMODEL'] = (meta.get('pb_model', 'v1-2026-07-04'), 'factor 1: model PB library version')
                if meta.get('chromatic'):
                    ph['PBCHROM'] = (meta['chromatic'], 'factor 2: bandwidth-smearing term (a priori)')
                if meta.get('residual_pb'):
                    ph['PBRESID'] = (meta['residual_pb'], 'factor 3: measured residual gain per band')
                ph['PBROBUST'] = ('-0.75' if rob == 'R075' else '0', 'robust weighting whose beam sets factor 2')
                ph['PBBMAJ'] = (float(fits.getheader(bsrc)['BMAJ']), 'deg, beam FWHM used in factor 2')
                fits.writeto(f'{cf}/PB_deep_{"R075_" if rob == "R075" else ""}{scope}.fits',
                             pb_map.astype(np.float32), ph, overwrite=True)
                maps[rob] = (pb_map, meta)
            pb_map, meta = maps[rob]
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
    # P3_PBCORR_LSTS='00h,01h,...' (2026-09-07): restrict to those LST hours (parallel streams)
    _lsts = os.environ.get('P3_PBCORR_LSTS')
    if _lsts:
        _keep = set(_lsts.split(','))
        cells = [(r, c) for r, c in cells if c.split('_')[0] in _keep]
    print(f'{len(cells)} cells ({MODE}' + (f', LSTs {_lsts}' if _lsts else '') + ')', flush=True)
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
