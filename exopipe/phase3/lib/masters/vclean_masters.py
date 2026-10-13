#!/usr/bin/env python3
"""Cleaned Stokes-V deep masters — mosaic-input production chain.

PRODUCER -> CONSUMER CONTRACT (GC-PRE-4, Gregg ruling 2026-08-12):
produces V_deep_Taper_R0_clean(+pbcorr) masters — the critical input to the
Stokes V mosaicked products (PHASE3_ARCHITECTURE.md §7bis, ruling
2026-08-12d).

Per cell (Gregg via mosaic chat, 2026-07-11): raw V master ->
search_v_transients_v2 --clean_only (sidelobe + bilobe + striation
subtraction, APPARENT domain, using the matching I master for source
positions) -> divide by the cell's stored PB_deep map -> intrinsic cleaned-V
master.

Outputs per cell (coadd_final/):
  V_deep_Taper_R0_clean_{scope}.fits          (apparent, SKYSUB headers)
  V_deep_Taper_R0_clean_pbcorr_{scope}.fits   (intrinsic, PBMETHOD/PBWINDOW)

I master selection: Taper_R0 for wideband scopes, Taper_R075 for subbands
(the search convention); falls back to the other robust if the primary is
corrupt. Idempotent per cell: an existing pair is skipped only when the sibling passes its content test against the
cell's current PB_deep map (pb_apply.sibling_check); a failing sibling alone is regenerated (2026-10-03). skip_existing=False
rebuilds the pair.

ENGINE: the --clean_only machinery of exopipe/phase3/search_v_transients_v2
(the GC-PRE-1 re-absorbed PRODUCTION copy; main() Phases 2-4 + SKYSUB header
stamp), shelled as a subprocess exactly as the frozen production driver
(clv15/phase3_vclean_masters.py, now superseded by this module) did.
History: engine ran the 2026-07-11 cleaned-V pass 330/330 subband +
wideband cells; the frozen driver's cell-sweep semantics are preserved in
run_vclean().

Callable contract for the golden-cell harness
(exopipe_ops/golden_cells/bin/drive_vclean.py):
    build_vclean_cell(coadd_root, cell) -> (cell, status)
status 'OK' / 'skip (verified: ...)' / 'sibling regenerated (...)' on success (2026-10-03: the old 'skip (exists)' is
retired - an existing sibling is skipped only after its content test); any other string is a refusal
reason (caller decides exit semantics). No import-time side effects.

Usage (batch, mirrors the frozen driver):
    python vclean_masters.py [wideband|subband|all] [NPAR]
"""
from exopipe.phase3.config import coadd_root
import os
import re
import sys
import glob
import shutil
import tempfile
import subprocess

import numpy as np
from astropy.io import fits

WB_ROOT = None   # NO default (Gregg 2026-09-04): coadd_root() at point of use
SB_ROOT = '/data07/gh/phase3/coadd_sb'
TMP_ROOT = '/fast/gh/main/tmp'
SV2_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'search', 'search_v.py')
SV2_TIMEOUT_S = 7200


def sane(path):
    """A master is sane if it exists, loads, and |max| is finite and < 1e6 Jy."""
    if not os.path.exists(path):
        return False
    try:
        a = fits.getdata(path)
        m = float(np.nanmax(np.abs(a)))
        return np.isfinite(m) and m < 1e6
    except Exception:
        return False


def pick_i_master(coadd_final_dir, scope):
    """Matching I master for the clean pass: R0 first for wideband scopes,
    R075 first for subbands (search convention); fall back if corrupt."""
    order = (['Taper_R0', 'Taper_R075'] if not scope.endswith('MHz')
             else ['Taper_R075', 'Taper_R0'])
    for var in order:
        p = f'{coadd_final_dir}/I_deep_{var}_{scope}.fits'
        if sane(p):
            return p
    return None


def write_vclean_sibling(cf, scope):
    """Write V_deep_Taper_R0_clean_pbcorr_{scope}.fits = the cleaned master / the cell's STORED PB_deep map (PB > floor), with
    the PB provenance of the cell's V_deep_Taper_R0 sibling (same map, same run: PBMODEL/PBCHROM/PBRESID/PBNDATE) and the
    PBMETHOD/PBWINDOW/BEAMMOD cards phase3_pbcorr_masters.do_cell writes. The cleaning engine is not run."""
    from exopipe.phase3.lib.masters import pb_apply as PB
    from exopipe.phase3.lib.masters.pbcorr_masters import BEAMMOD
    out_ap = f'{cf}/V_deep_Taper_R0_clean_{scope}.fits'
    out_pb = f'{cf}/V_deep_Taper_R0_clean_pbcorr_{scope}.fits'
    ref = f'{cf}/V_deep_Taper_R0_pbcorr_{scope}.fits'
    meta = None
    if os.path.exists(ref):
        rh = fits.getheader(ref)
        meta = dict(pb_model=rh.get('PBMODEL', 'v1-2026-07-04'), chromatic=rh.get('PBCHROM'), residual_pb=rh.get('PBRESID'),
                    n_dates=rh.get('PBNDATE', -1), n_missing=rh.get('PBMISS'))
    pb_map = np.squeeze(fits.getdata(f'{cf}/PB_deep_{scope}.fits')).astype(np.float64)
    PB.apply_pbcorr(out_ap, pb_map, out_pb, meta=meta)
    with fits.open(out_pb, mode='update') as h:
        h[0].header['PBMETHOD'] = ('time-integrated-library', 'canonical (2026-07-11)')
        h[0].header['PBWINDOW'] = ('60min-deep', 'PB integration window')
        h[0].header['BEAMMOD'] = (BEAMMOD, 'beam model')
    ok, why = PB.sibling_check(out_pb, out_ap, pb_map)
    if not ok:
        raise RuntimeError(f'{out_pb}: written sibling fails its content test ({why})')
    return why


def build_vclean_cell(coadd_root, cell, tmp_root=None, skip_existing=True):
    """Build V_deep_Taper_R0_clean(+pbcorr) for one cell. Returns (cell, status).

    Refuses (status string, no product) when the raw V master, a sane I
    master, or the PB_deep map is missing — the PB map must exist first
    (phase3_pbcorr_masters / drive_pbcorr).
    """
    from exopipe.phase3.lib.masters import pb_apply as PB

    lst, scope = cell.split('_', 1)
    cf = f'{coadd_root}/{cell}/coadd_final'
    vp = f'{cf}/V_deep_Taper_R0_{scope}.fits'
    out_ap = f'{cf}/V_deep_Taper_R0_clean_{scope}.fits'
    out_pb = f'{cf}/V_deep_Taper_R0_clean_pbcorr_{scope}.fits'
    # CONTENT GUARD, not existence (CLAUDE.md 3; Gregg 2026-10-03 "preamble approved"): an existing pair is skipped only when
    # the sibling equals the cleaned master / the cell's CURRENT PB_deep map; otherwise only the sibling is regenerated. The
    # existence-only skip let PB step 2's map rewrite (2026-10-01) leave 19 siblings stale by up to 5.6 per cent.
    if skip_existing and os.path.exists(out_pb) and os.path.exists(out_ap):
        pbp = f'{cf}/PB_deep_{scope}.fits'
        if not os.path.exists(pbp):
            return cell, 'NO PB MAP'
        ok, why = PB.sibling_check(out_pb, out_ap, np.squeeze(fits.getdata(pbp)))
        if ok:
            return cell, 'skip (verified: sibling x PB == cleaned master)'
        if fits.getheader(f'{cf}/I_deep_Taper_R075_{scope}.fits').get('MDWARP'):
            return cell, (f'REFUSED stale sibling ({why}) in a master-dewarped cell: its PB maps are made by the dewarp '
                          f'install, not here')
        try:
            return cell, f'sibling regenerated (was: {why}; now: {write_vclean_sibling(cf, scope)})'
        except Exception as e:
            return cell, f'ERROR regenerating the sibling: {e}'
    if not sane(vp):
        return cell, 'NO SANE V MASTER'
    ip = pick_i_master(cf, scope)
    if ip is None:
        return cell, 'NO SANE I MASTER'
    pbp = f'{cf}/PB_deep_{scope}.fits'
    if not os.path.exists(pbp):
        return cell, 'NO PB MAP'
    tmp_root = tmp_root or TMP_ROOT
    os.makedirs(tmp_root, exist_ok=True)
    td = tempfile.mkdtemp(prefix=f'vclean_{cell}_', dir=tmp_root)
    try:
        env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
               'OPENBLAS_NUM_THREADS': '2'}
        # sv2 imports exopipe.* (GC-PRE-1 rewrite): repo root on PYTHONPATH.
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))))
        env['PYTHONPATH'] = repo_root + (
            os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        r = subprocess.run(
            [sys.executable, SV2_SCRIPT,
             '--v_image', vp, '--i_image', ip,
             '--clean_only', '--no_cutouts', '--out_dir', td],
            capture_output=True, text=True, timeout=SV2_TIMEOUT_S, env=env)
        res = glob.glob(f'{td}/*_residual_final.fits')
        if r.returncode != 0 or not res:
            return cell, f'sv2 FAILED rc={r.returncode}: {r.stderr[-200:]}'
        shutil.move(res[0], out_ap)
        write_vclean_sibling(cf, scope)     # one writer for build and repair (2026-10-03)
        return cell, 'OK'
    except Exception as e:
        return cell, f'ERROR {e}'
    finally:
        shutil.rmtree(td, ignore_errors=True)


def enumerate_cells(mode='all', wb_root=None, sb_root=None):
    """(root, cell) pairs for the batch sweep, frozen-driver semantics."""
    wb_root = wb_root or coadd_root()
    sb_root = sb_root or SB_ROOT
    cells = []
    if mode in ('wideband', 'all') and os.path.isdir(wb_root):
        cells += [(wb_root, c) for c in sorted(os.listdir(wb_root))
                  if re.match(r'^\d\dh_(Red|Green|Blue|Full)$', c)]
    if mode in ('subband', 'all') and os.path.isdir(sb_root):
        cells += [(sb_root, c) for c in sorted(os.listdir(sb_root))
                  if re.match(r'^\d\dh_\d+MHz$', c)]
    return cells


def _do_cell(args):
    root, cell = args
    return build_vclean_cell(root, cell)


def run_vclean(mode='all', npar=2, wb_root=None, sb_root=None):
    """Batch sweep across cells (mirror of the frozen driver's __main__ body)."""
    import multiprocessing as mp_
    cells = enumerate_cells(mode, wb_root, sb_root)
    print(f'{len(cells)} cells ({mode}), NPAR={npar}', flush=True)
    nok = nbad = 0
    with mp_.get_context('fork').Pool(npar) as pool:
        for cell, st in pool.imap_unordered(_do_cell, cells):
            ok = st == 'OK' or st.startswith('skip (verified') or st.startswith('sibling regenerated')
            nok += ok
            nbad += (not ok)
            print(f'  {cell}: {st}', flush=True)
    print(f'VCLEAN-MASTERS-DONE ok={nok} problems={nbad}', flush=True)
    return nok, nbad


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    npar = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    run_vclean(mode, npar)


if __name__ == '__main__':
    main()
