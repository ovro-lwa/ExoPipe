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
corrupt. Idempotent per cell (skips when the pbcorr output exists, unless
skip_existing=False).

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
status 'OK' / 'skip (exists)' on success; any other string is a refusal
reason (caller decides exit semantics). No import-time side effects.

Usage (batch, mirrors the frozen driver):
    python vclean_masters.py [wideband|subband|all] [NPAR]
"""
import os
import re
import sys
import glob
import shutil
import tempfile
import subprocess

import numpy as np
from astropy.io import fits

WB_ROOT = '/fast/gh/main/phase3/coadd'
SB_ROOT = '/data07/gh/phase3/coadd_sb'
TMP_ROOT = '/fast/gh/main/tmp'
SV2_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'search_v_transients_v2.py')
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


def build_vclean_cell(coadd_root, cell, tmp_root=None, skip_existing=True):
    """Build V_deep_Taper_R0_clean(+pbcorr) for one cell. Returns (cell, status).

    Refuses (status string, no product) when the raw V master, a sane I
    master, or the PB_deep map is missing — the PB map must exist first
    (phase3_pbcorr_masters / drive_pbcorr).
    """
    from exopipe.phase3 import pb_apply as PB

    lst, scope = cell.split('_', 1)
    cf = f'{coadd_root}/{cell}/coadd_final'
    vp = f'{cf}/V_deep_Taper_R0_{scope}.fits'
    out_ap = f'{cf}/V_deep_Taper_R0_clean_{scope}.fits'
    out_pb = f'{cf}/V_deep_Taper_R0_clean_pbcorr_{scope}.fits'
    if skip_existing and os.path.exists(out_pb):
        return cell, 'skip (exists)'
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
        repo_root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
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
        pb_map = fits.getdata(pbp).astype(np.float64)
        PB.apply_pbcorr(out_ap, pb_map, out_pb)
        with fits.open(out_pb, mode='update') as h:
            h[0].header['PBMETHOD'] = ('time-integrated-library',
                                       'canonical (2026-07-11)')
            h[0].header['PBWINDOW'] = ('60min-deep', 'PB integration window')
        return cell, 'OK'
    except Exception as e:
        return cell, f'ERROR {e}'
    finally:
        shutil.rmtree(td, ignore_errors=True)


def enumerate_cells(mode='all', wb_root=None, sb_root=None):
    """(root, cell) pairs for the batch sweep, frozen-driver semantics."""
    wb_root = wb_root or WB_ROOT
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
            ok = st in ('OK', 'skip (exists)')
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
