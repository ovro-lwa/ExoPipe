#!/usr/bin/env python3
"""
phase3_save_residuals.py — write the Phase 3 deep products CONFORMANTLY:
the flow-aligned NON-subtracted image and the sky-SUBTRACTED residual, I & V
(Taper / Robust-0 search variant), into the canonical Phase 3 date dirs.

Naming + header via phase3_products (build_name + write_standard_header — the
FULL 7-item standard, validated on write). Per-epoch time / THERMNOI / GITTAG /
provenance are sourced from the original Phase 2 deep image (B's _sub headers
carry the co-add date, not the epoch). Residual = ×2-upflow(epoch) − master −
striation, exactly as the search ran. Deterministically regenerable.
"""
from exopipe.common.roots import coadd_root
import os, sys, glob, re
_ARGV = list(sys.argv)
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from exopipe.phase3.phase3_deep import _load, fft_upsample2, fft_decimate2
from exopipe.phase3.coadd_full_pipeline import apply_flow_field, build_striation_model
from exopipe.phase3.phase3_pipeline import STRIATION_WINDOW_DEG
from exopipe.phase3.identify import fit_empirical_beam
from exopipe.common import phase3_products as pp
from exopipe.common import phase3_paths as P

LABEL = _ARGV[3] if len(_ARGV) > 3 else 'PROD'
ROOT = os.environ.get('P3_STAGING_ROOT', P.STAGING_ROOT)   # /data07 override


def _p2_original(lst, scope, date):
    """Original Phase 2 deep dewarped image -> source of the correct time
    block, THERMNOI and provenance for this epoch."""
    pat = f'{scope}_I_deep_Taper_Robust-0_dewarped_*_LST{lst}.fits'
    for arch in (f'/lustre/pipeline/exopipe/phase2/flow/{lst}/{date}',
                 f'/lustre/pipeline/exopipe/phase2/{lst}/{date}'):
        g = glob.glob(f'{arch}/*/{scope}/I/deep/{pat}')
        if g:
            return sorted(g)[-1]
    return None


def _residual(epoch, master, flow, hdr, cd):
    up = fft_decimate2(apply_flow_field(fft_upsample2(epoch), flow))
    invalid = (up == 0) | (master == 0) | ~np.isfinite(up)
    w = WCS(hdr).celestial
    sm, _ = build_striation_model(np.nan_to_num(up - master), w, invalid,
                                  ra_window_beams=STRIATION_WINDOW_DEG / cd,
                                  beam_pix=1.0)
    return up, np.where(invalid, np.nan, (up - master) - sm)


def _write(data, wcs_hdr, outdir, name, *, scope, stokes, src, ebeam,
           msubfile=None):
    """Write one product + stamp the full validated standard header."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fits.writeto(path, np.nan_to_num(data, nan=0.0).astype(np.float32),
                 wcs_hdr, overwrite=True)
    sh = fits.getheader(src) if src else {}
    p2run = None
    if src and 'Science_' in src:
        p2run = 'Science_' + src.split('Science_')[1].split('/')[0]
    pp.write_standard_header(
        path, scope=scope, stokes=stokes,
        start=sh.get('DATE-OBS'), end=sh.get('DATE-END'),
        inttime_s=sh.get('INTTIME'), time_basis='phase2',
        ebeam=ebeam, thermnoi=sh.get('THERMNOI'),
        imgrms=float(np.nanstd(data[data.shape[0] // 4:3 * data.shape[0] // 4,
                                    data.shape[1] // 4:3 * data.shape[1] // 4])),
        dewarped=True, aligned=True,
        msub=(msubfile is not None), msubfile=msubfile,
        alignref='deep co-add (P3 x2-upflow)', p2run=p2run, validate=True)
    return path


def save_cell(lst, scope):
    # P3_COADD_ROOT override (same convention as phase3_deep): subband cells
    # live at /data07/gh/phase3/coadd_sb, wideband at the rebuild_w2/rebuild_v2 roots.
    _croot = coadd_root()   # REQUIRED, no default (Gregg 2026-09-04)
    base = f'{_croot}/{lst}_{scope}'
    _irob = 'R0' if scope in ('Red', 'Green', 'Blue', 'Full') else 'R075'  # subbands co-add I at Robust+0.75
    mIp = f'{base}/coadd_final/I_deep_Taper_{_irob}_{scope}.fits'
    if not os.path.exists(mIp):
        print(f'{lst}_{scope}: no master ({_irob}), skip'); return 0
    mI = _load(mIp); mh = fits.getheader(mIp); cd = abs(mh['CDELT1'])
    em, en, ep, nm = fit_empirical_beam(mI)
    if not np.isfinite(em):
        print(f'{lst}_{scope}: corrupt master, skip'); return 0
    ebeam = (em * cd, en * cd, ep)
    mVp = f'{base}/coadd_final/V_deep_Taper_R0_{scope}.fits'
    mV = _load(mVp) if os.path.exists(mVp) else None
    # deep workdir (flow .npy) — P3_DEEP_ROOT lets a runner stage flows on fast
    # storage (NVMe) for the heavy per-epoch up2_flow reads; default = /data07.
    _deep_root = os.environ.get('P3_DEEP_ROOT', '/data07/gh/phase3/deep')
    cw = f'{_deep_root}/{lst}_{scope}'
    n = 0
    for c in sorted(glob.glob(f'{base}/subtracted/*_I_sub_{scope}.fits')):
        d = re.search(r'(\d{4}-\d\d-\d\d)_I_sub', os.path.basename(c)).group(1)
        fp = f'{cw}/ep_{d}_up2_flow.npy'
        if not os.path.exists(fp):
            continue
        flow = np.load(fp)
        src = _p2_original(lst, scope, d)
        dd = d.replace('-', '')
        hdr_wcs = mh.copy()
        for st, mm, mpath in (('I', mI, mIp), ('V', mV, mVp)):
            if mm is None:
                continue
            sub = f'{base}/subtracted/{d}_{st}_sub_{scope}.fits'
            if not os.path.exists(sub):
                continue
            up, res = _residual(_load(sub) + mm, mm, flow, hdr_wcs, cd)
            od = P.epoch_dir(ROOT, lst, d, LABEL, scope, st, 'deep', make=True)
            kw = dict(lst=lst, date_start=dd, date_end=dd, n_epochs=1)
            # search-variant robust: wideband I+V and subband V are Robust-0;
            # the SUBBAND I chain consumes Taper Robust-0.75 (the R075 master
            # subtracted above) — name the product accordingly.
            _rob = 0.75 if (st == 'I' and _irob == 'R075') else 0.0
            nameA = pp.build_name(scope, st, 'deep', 'Taper', _rob,
                                  proc=['dewarped', 'aligned'], **kw)
            nameS = pp.build_name(scope, st, 'deep', 'Taper', _rob,
                                  proc=['dewarped', 'msub'], **kw)
            _write(up, hdr_wcs, od, nameA, scope=scope, stokes=st, src=src,
                   ebeam=ebeam)
            _write(res, hdr_wcs, od, nameS, scope=scope, stokes=st, src=src,
                   ebeam=ebeam, msubfile=os.path.basename(mpath))
        n += 1
    print(f'{lst}_{scope}: wrote {n} epochs (aligned+msub, I+V, conformant) '
          f'-> {ROOT}/{lst}/<date>/P3_{LABEL}/{scope}/', flush=True)
    return n


if __name__ == '__main__':
    save_cell(_ARGV[1], _ARGV[2])
