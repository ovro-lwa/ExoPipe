#!/usr/bin/env python3
"""STAGE B of the ratified Master absolute-flux-scale + final-dewarp stage
(MASTER_SHFLUX_VLSSR_DEWARP_SPEC_20260828.md, Gregg 2026-08-28; implementation approved 2026-09-06):
the FINAL master dewarp measured on the Stokes I Taper Robust -0.75 masters and applied to ALL of a cell's products
(ruled: 'the derived correction applies to ALL the data').

METHOD = THE PHASE-2 MAIN DEWARP (Gregg 2026-10-01: "The final Master dewarp should use the same method a the main
dewarp RBF-based multiquadric dwarp."). This replaces the 2026-09-06 implementation - an optical flow of each master
against a synthetic VLSSr sky - which departed from the spec ("the usual Phase-2 cascade dewarp machinery ... RBF warp
model"). The functions are Phase 2's own (exopipe.phase2.cascade_dewarp), run on one LST's masters instead of one
night's images:
  anchor   the 73 MHz master (fallbacks 78, 69, 64 MHz, as Phase 2) vs VLSSr at 74 MHz: PyBDSF on the PB-corrected
           master, unresolved sources on both sides, match radius 5 arcmin, peak flux ratio 0.8-1.2;
  cascade  outward subband by subband (69 -> 18 MHz down, 78 -> 82 MHz up), each master matched to the dewarped
           catalogue of the last ACCEPTED master of its arm: radius 5 arcmin x (73/nu)^1.5, flux ratio 0.8-1.2; three
           consecutive refusals halt an arm (every later subband gets a NOT ATTEMPTED row);
  wideband each Red/Green/Blue/Full master as one more cascade step from the accepted subband master whose centre
           frequency (phase2_config.SUBBAND_FREQ_HZ) is nearest the wideband master's header frequency;
  screen   iterative 3-sigma outlier clip of the anchors, then a multiquadric RBF screen per axis (smoothing 1.5, shape
           parameter = mean anchor nearest-neighbour distance: phase2_config WARP_RBF_*), applied by bilinear
           resampling (cascade_dewarp.apply_warp; NaN outside the image). Per-LST extraction settings (elevation
           floors, island cap) are set exactly as run_cascade_dewarping sets them.
RUNS IN THE BASE ENVIRONMENT (PyBDSF + scipy; no GPU):
    python -m exopipe.phase3.phase3_master_dewarp <lst> [<lst> ...] [--root R] [--input-root I] [--apply [--apply-scopes 18MHz,23MHz]]

Gates (refuse, never warn): a master already carrying MDWARP is REFUSED (no dewarp is stacked on another); fewer than 10
RBF anchors after outlier rejection = REFUSED; the VLSSr peak match (the reference-quality gate's measurement,
coadd_full_pipeline._reference_astrom_ok) on the plain master before and after: fewer than 10 matches, or no improvement
of the median separation = REFUSED, and the cell stays unwarped and flagged; a product whose shape differs from the
screen refuses the cell's apply before anything is written. Recorded per cell, not gating: anchors matched and kept, the
median anchor offset before, the in-sample residual of the screen at its anchors and the 5-fold held-out residual (the
screen's error at anchors it was not fitted to), screen amplitude p50/p95 over the central 3/4 of the frame.
--apply writes every FITS product of the cell's generation (coadd_final incl. loo/, pbcorr siblings, PB map, and
subtracted/ unless --no-residuals) resampled with the one screen into <root>/<cell>/gen_mdewarp_rbf/ (new files, the
'mdewarp' token before the scope; MDWARP/MDWMETH/MDWREF/MDWNANC/MDWCV/MDWMED0/MDWMED1/MDWP50/MDWP95/MDWDATE cards +
HISTORY); coadd_final is never touched (installation is a separate, gated step). --input-root I reads a cell's
generation from I/<lst>_<scope>/ where that exists (reconstructed un-warped inputs of previously flow-dewarped cells).
One row per cell attempt in <root>/mdewarp_rbf_table.csv; screens (FITS) and Phase-2 diagnostics in MEAS_ROOT/<lst>/.
"""
import argparse, csv, glob, logging, os, sys, time
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
sys.path.insert(0, '/lustre/gh/main/exopipe')
from exopipe.common import pipeline_config as pc
from exopipe.common import phase2_config as cfg
from exopipe.phase2 import cascade_dewarp as CD

VLSSR_PEAKS = '/lustre/gh/calibration/pipeline/reference/surveys/vlssr_radecpeak_unresolved.txt'
MIN_ANCHORS = 10
_VL = None


def vlssr_offsets(data, hdr):
    """Median VLSSr separation (arcmin) of the 50 brightest central peaks, N matches (< 60 arcmin), p90.
    Port of coadd_full_pipeline._reference_astrom_ok's measurement (same peak finder, same boxes)."""
    global _VL
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from scipy.ndimage import maximum_filter, center_of_mass
    if _VL is None:
        vv = np.loadtxt(VLSSR_PEAKS, skiprows=1); _VL = SkyCoord(vv[:, 0] * u.deg, vv[:, 1] * u.deg)
    a = np.nan_to_num(np.asarray(data, dtype=float)); w = WCS(hdr).celestial
    n = a.shape[0]; q = n // 8; cen = a[q:-q, q:-q]
    mfl = maximum_filter(cen, size=25); py, px = np.where((cen == mfl) & (cen > 0))
    if not len(py): return float('nan'), 0, float('nan')
    fx = cen[py, px]; seps = []
    for i in np.argsort(fx)[::-1][:50]:
        y, x = int(py[i]) + q, int(px[i]) + q
        box = a[y - 4:y + 5, x - 4:x + 5]
        if box.size != 81 or not np.all(np.isfinite(box)) or box.max() <= 0: continue
        cy, cx = center_of_mass(np.maximum(box, 0))
        sky = w.pixel_to_world(x - 4 + cx, y - 4 + cy)
        d = float(sky.separation(_VL).arcmin.min())
        if d < 60: seps.append(d)
    if not seps: return float('nan'), 0, float('nan')
    return float(np.median(seps)), len(seps), float(np.percentile(seps, 90))


def _fix2d(hdr):
    h = hdr.copy()
    for k in list(h.keys()):
        if k.startswith(('NAXIS3', 'NAXIS4', 'CTYPE3', 'CTYPE4', 'CRVAL3', 'CRVAL4', 'CRPIX3', 'CRPIX4', 'CDELT3', 'CDELT4', 'CUNIT3', 'CUNIT4')): h.pop(k, None)
    h['NAXIS'] = 2; h['WCSAXES'] = 2
    return h


def _out_name(path, scope):
    b = os.path.basename(path); suf = f'_{scope}.fits'
    return b[:-len(suf)] + f'_mdewarp{suf}' if b.endswith(suf) else b.replace('.fits', '_mdewarp.fits')


MEAS_ROOT = '/fast/gh/main/phase3/mdewarp_rbf'   # screens + Phase-2 diagnostics per LST (analysis output, not a product)
NO_RESIDUALS = False
WIDEBANDS = ('Red', 'Green', 'Blue', 'Full')
OUT_GEN = 'gen_mdewarp_rbf'
COLS = ['lst', 'scope', 'ref', 'status', 'input', 'med_before', 'n_before', 'p90_before', 'med_after', 'n_after', 'p90_after',
        'n_match', 'n_anchors', 'anchor_off_med_as', 'fit_resid_med_as', 'cv_resid_med_as', 'cv_resid_p90_as',
        'warp_p50_as', 'warp_p95_as', 'n_written', 'note', 'seconds', 'measured_at']
LOG = logging.getLogger('phase3_master_dewarp')


def _gen_root(root, input_root, lst, scope):
    """The cell's generation root: <input_root>/<lst>_<scope> when that holds a coadd_final, else <root>/<lst>_<scope>."""
    for base in ([input_root] if input_root else []) + [root]:
        if os.path.isdir(f'{base}/{lst}_{scope}/coadd_final'):
            return f'{base}/{lst}_{scope}'
    return None


def _screen_residuals(anchors, wcs, sx, sy, cd_as, k=5, seed=0):
    """In-sample residual of the applied screen at its anchors, and the k-fold held-out residual (a screen with the
    production RBF settings fitted without one fold, evaluated at that fold), both in arcsec."""
    from scipy.interpolate import RBFInterpolator
    from scipy.ndimage import map_coordinates
    from scipy.spatial import cKDTree
    mx, my = wcs.all_world2pix(anchors['ra_lwa'], anchors['dec_lwa'], 0)
    tx, ty = wcs.all_world2pix(anchors['ra_cat'], anchors['dec_cat'], 0)
    dx, dy = mx - tx, my - ty
    fit = np.hypot(dx - map_coordinates(sx, [ty, tx], order=1), dy - map_coordinates(sy, [ty, tx], order=1)) * cd_as
    pts = np.column_stack([ty, tx]); n = len(dx); idx = np.random.default_rng(seed).permutation(n); cv = np.full(n, np.nan)
    for f in range(k):
        te = idx[f::k]; tr = np.setdiff1d(idx, te)
        if len(tr) < cfg.MIN_WARP_ANCHORS or not len(te):
            continue
        kw = dict(kernel=cfg.WARP_RBF_KERNEL, smoothing=cfg.WARP_RBF_SMOOTHING)
        if cfg.WARP_RBF_KERNEL in ('multiquadric', 'inverse_multiquadric', 'gaussian'):
            kw['epsilon'] = cfg.WARP_RBF_EPSILON or float(np.mean(cKDTree(pts[tr]).query(pts[tr], k=2)[0][:, 1]))
        px = RBFInterpolator(pts[tr], dx[tr], **kw)(pts[te]); py = RBFInterpolator(pts[tr], dy[tr], **kw)(pts[te])
        cv[te] = np.hypot(dx[te] - px, dy[te] - py) * cd_as
    return np.hypot(dx, dy) * cd_as, fit, cv


def _measure(lst, scope, groot, ref, vlssr, diag):
    """One master. ref None = the anchor step against VLSSr; else (label, catalogue, nominal MHz) of the accepted master
    it cascades from. Returns (row, screen or None, dewarped catalogue or None)."""
    row = {c: '' for c in COLS}
    row.update(lst=lst, scope=scope, ref='VLSSr' if ref is None else f'{ref[0]} (master-dewarped)', input=groot or '',
               n_written=0, measured_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    t0 = time.time()
    if groot is None:
        row['status'] = 'REFUSED no generation root'; return row, None, None
    plain = f'{groot}/coadd_final/I_deep_Taper_R075_{scope}.fits'; pbc = f'{groot}/coadd_final/I_deep_Taper_R075_pbcorr_{scope}.fits'
    miss = [p for p in (plain, pbc) if not os.path.exists(p)]
    if miss:
        row['status'] = f'REFUSED missing {os.path.basename(miss[0])}'; return row, None, None
    with fits.open(plain) as h:
        d0 = np.squeeze(h[0].data).astype(np.float32); hdr = h[0].header.copy()
    if hdr.get('MDWARP') or fits.getheader(pbc).get('MDWARP'):
        row['status'] = 'REFUSED input already carries MDWARP (a dewarp is never stacked on another)'; return row, None, None
    wcs = WCS(hdr).celestial; shape = d0.shape; beam = hdr.get('BMAJ')
    cd_as = abs(float(hdr.get('CDELT1') or hdr.get('CD1_1'))) * 3600.0
    m0, n0, p0 = vlssr_offsets(d0, hdr); row.update(med_before=f'{m0:.3f}', n_before=n0, p90_before=f'{p0:.3f}')
    if n0 < MIN_ANCHORS:
        row['status'] = f'REFUSED {n0} VLSSr peak matches < {MIN_ANCHORS} before dewarp'; return row, None, None
    df = CD.extract_sources(pbc, LOG)
    if ref is None:
        anchors = CD.cross_match_vlssr(df, vlssr, cfg.ANCHOR_MATCH_RADIUS, cfg.ANCHOR_FLUX_RATIO_MIN,
                                       cfg.ANCHOR_FLUX_RATIO_MAX, LOG, beam_deg=beam)
    else:
        fmhz = float(scope.replace('MHz', '')) if scope.endswith('MHz') else ref[2]
        r = cfg.CASCADE_MATCH_RADIUS_BASE * (cfg.ANCHOR_FREQ_MHZ / fmhz) ** cfg.CASCADE_MATCH_RADIUS_EXPONENT
        anchors = CD.cross_match_subbands(df, ref[1], r * u.arcmin, cfg.CASCADE_FLUX_RATIO_MIN,
                                          cfg.CASCADE_FLUX_RATIO_MAX, LOG, beam_deg=beam)
    if anchors is None:
        row.update(status=f'REFUSED fewer than {cfg.MIN_WARP_ANCHORS} matched anchors ({len(df)} sources extracted)',
                   seconds=f'{time.time() - t0:.0f}'); return row, None, None
    row['n_match'] = len(anchors['ra_lwa'])
    anchors = CD.reject_outlier_anchors(anchors, wcs, shape, LOG)
    if anchors is None or len(anchors['ra_lwa']) < MIN_ANCHORS:
        row.update(status=f'REFUSED fewer than {MIN_ANCHORS} anchors after outlier rejection', seconds=f'{time.time() - t0:.0f}')
        return row, None, None
    row['n_anchors'] = len(anchors['ra_lwa'])
    sx, sy = CD.compute_warp_screen(anchors, wcs, shape, LOG, method='rbf', rbf_kernel=cfg.WARP_RBF_KERNEL,
                                    rbf_smoothing=cfg.WARP_RBF_SMOOTHING, rbf_epsilon=cfg.WARP_RBF_EPSILON,
                                    snr_weighting=cfg.WARP_RBF_SNR_WEIGHTING)
    if sx is None:
        row.update(status='REFUSED RBF screen failed', seconds=f'{time.time() - t0:.0f}'); return row, None, None
    off, fit, cv = _screen_residuals(anchors, wcs, sx, sy, cd_as)
    amp = np.hypot(sx, sy) * cd_as; q = amp.shape[0] // 8; ac = amp[q:-q, q:-q]
    row.update(anchor_off_med_as=f'{np.median(off):.1f}', fit_resid_med_as=f'{np.median(fit):.1f}',
               cv_resid_med_as=f'{np.nanmedian(cv):.1f}', cv_resid_p90_as=f'{np.nanpercentile(cv, 90):.1f}',
               warp_p50_as=f'{np.percentile(ac, 50):.1f}', warp_p95_as=f'{np.percentile(ac, 95):.1f}')
    d1 = CD.apply_warp(d0, sx, sy)
    m1, n1, p1 = vlssr_offsets(d1, hdr); row.update(med_after=f'{m1:.3f}', n_after=n1, p90_after=f'{p1:.3f}')
    CD.save_diagnostics(anchors, sx, sy, wcs, shape, scope, diag, row['ref'], LOG)
    CD._save_warp_screen(sx, sy, wcs, scope, diag, 'rbf', cfg.WARP_RBF_KERNEL, row['n_anchors'], LOG)
    if not (np.isfinite(m1) and m1 < m0) or n1 < MIN_ANCHORS:
        row.update(status=f'REFUSED regression: median VLSSr sep {m0:.2f} -> {m1:.2f} arcmin (N {n0} -> {n1}); cell left unwarped',
                   seconds=f'{time.time() - t0:.0f}')
        return row, None, None
    row['status'] = 'OK'
    # catalogue for the next cascade step: PyBDSF on the dewarped PB-corrected master (Phase 2 step 9); if that extraction
    # fails the reference does NOT advance (Phase 2 would fall back to the pre-dewarp extraction)
    tmp = f'{diag}/{scope}_pbcorr_mdewarp_tmp.fits'
    with fits.open(pbc) as h:   # original header and array shape, as Phase 2 writes *_dewarped.fits (PyBDSF reads the frequency axis)
        fits.writeto(tmp, CD.apply_warp(np.squeeze(h[0].data).astype(np.float32), sx, sy).reshape(h[0].data.shape),
                     h[0].header, overwrite=True)
    cat = CD.extract_sources(tmp, LOG); os.remove(tmp)
    if cat.empty:
        row['note'] = 'dewarped re-extraction failed: the cascade reference does not advance past this master'; cat = None
    row['seconds'] = f'{time.time() - t0:.0f}'
    return row, (sx, sy), cat


def _apply(lst, scope, groot, root, screen, row):
    """Resample every FITS product of the cell's generation with the one screen into <root>/<cell>/gen_mdewarp_rbf/.
    Refuses before writing if the output generation exists or any product's shape differs from the screen."""
    sx, sy = screen
    out = f'{root}/{lst}_{scope}/{OUT_GEN}'
    if os.path.isdir(out) and os.listdir(out):
        raise RuntimeError(f'{out} already holds files - refusing to overwrite a written generation')
    files = []
    for sdir in ('coadd_final',) + (() if NO_RESIDUALS else ('subtracted',)):
        files += sorted(glob.glob(f'{groot}/{sdir}/**/*.fits', recursive=True))
    bad = []
    for p in files:
        h = fits.getheader(p)
        if (h.get('NAXIS2'), h.get('NAXIS1')) != sx.shape or h.get('MDWARP'):
            bad.append(os.path.basename(p))
    if bad:
        raise RuntimeError(f'{len(bad)} products differ from the screen shape or already carry MDWARP, e.g. {bad[:3]}')
    cards = dict(MDWARP=(True, 'final master dewarp applied (Gregg 2026-08-28)'),
                 MDWMETH=('rbf-multiquadric', 'Phase-2 RBF dewarp (Gregg 2026-10-01)'),
                 MDWREF=(row['ref'][:60], 'dewarp reference'),
                 MDWNANC=(int(row['n_anchors']), 'RBF anchors after outlier rejection'),
                 MDWCV=(float(row['cv_resid_med_as']), 'median 5-fold held-out anchor residual, arcsec'),
                 MDWMED0=(float(row['med_before']), 'median VLSSr sep before, arcmin'),
                 MDWMED1=(float(row['med_after']), 'median VLSSr sep after, arcmin'),
                 MDWP50=(float(row['warp_p50_as']), 'screen amplitude p50, arcsec'),
                 MDWP95=(float(row['warp_p95_as']), 'screen amplitude p95, arcsec'),
                 MDWDATE=(row['measured_at'], 'dewarp measured'))
    nw = 0
    for p in files:
        rel = os.path.relpath(p, groot); dst = f'{out}/{os.path.dirname(rel)}'; os.makedirs(dst, exist_ok=True)
        with fits.open(p) as h:   # header and array shape kept (frequency/Stokes axes stay for downstream PyBDSF)
            shp = h[0].data.shape; d = np.squeeze(h[0].data).astype(np.float32); hh = h[0].header.copy()
        w = CD.apply_warp(d, sx, sy).astype(np.float32).reshape(shp)
        for k, v in cards.items():
            hh[k] = v
        hh['HISTORY'] = (f'mdewarp: resampled with the RBF screen measured on I_deep_Taper_R075_{scope} vs {row["ref"]} '
                         f'(stage B of MASTER_SHFLUX_VLSSR_DEWARP_SPEC_20260828; method ruled 2026-10-01)')
        fits.writeto(f'{dst}/{_out_name(p, scope)}', w, hh, overwrite=False); nw += 1
    return nw


def run_lst(lst, root, input_root, apply, apply_scopes, table, vlssr):
    """Anchor, cascade down and up, then the wideband masters, for one LST. One table row per attempt."""
    CD.ANCHOR_EL_CUT = getattr(cfg, 'ANCHOR_MIN_ELEVATION_DEG', {}).get(lst)
    CD.ANCHOR_EAST_EL_CUT = getattr(cfg, 'ANCHOR_EAST_MIN_ELEVATION_DEG', {}).get(lst)
    cap = getattr(cfg, 'ANCHOR_MAX_ISLAND_BEAM_AREAS', {}); CD.ANCHOR_MAXPIX_BEAM_AREAS = cap.get(lst, cap.get('*'))
    diag = f'{MEAS_ROOT}/{lst}'; os.makedirs(diag, exist_ok=True)
    accepted = {}   # scope -> (label, dewarped catalogue, nominal MHz)

    def emit(row):
        new = not os.path.exists(table)
        with open(table, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{row['lst']} {row['scope']} vs {row['ref']}: {row['status']} | VLSSr median {row['med_before']} -> "
              f"{row['med_after']} arcmin (N {row['n_before']} -> {row['n_after']}); anchors {row['n_anchors']}/{row['n_match']}, "
              f"offset {row['anchor_off_med_as']}\" -> held-out {row['cv_resid_med_as']}\"; screen p50/p95 "
              f"{row['warp_p50_as']}/{row['warp_p95_as']}\"; written {row['n_written']} {row['note']}", flush=True)

    def step(scope, ref):
        groot = _gen_root(root, input_root, lst, scope)
        row, screen, cat = _measure(lst, scope, groot, ref, vlssr, diag)
        if row['status'] == 'OK':
            if cat is not None:
                accepted[scope] = (scope, cat, float(scope.replace('MHz', '')) if scope.endswith('MHz') else ref[2])
            if apply and (not apply_scopes or scope in apply_scopes):
                try:
                    row['n_written'] = _apply(lst, scope, groot, root, screen, row)
                except Exception as e:
                    row['status'] = f'APPLY REFUSED: {str(e)[:160]}'
        emit(row)
        return row['status'] == 'OK' and cat is not None

    anchor = next((c for c in CD.ANCHOR_FALLBACK if step(c, None)), None)
    if anchor is None:
        for sb in cfg.ALL_SUBBANDS + list(WIDEBANDS):
            if sb not in CD.ANCHOR_FALLBACK:
                emit({**{c: '' for c in COLS}, 'lst': lst, 'scope': sb, 'ref': '-', 'n_written': 0,
                      'status': f'NOT ATTEMPTED: every anchor candidate {CD.ANCHOR_FALLBACK} refused'})
        return
    a = cfg.ALL_SUBBANDS.index(anchor)
    for arm in (cfg.ALL_SUBBANDS[a - 1::-1] if a > 0 else [], cfg.ALL_SUBBANDS[a + 1:]):
        ref, fails = anchor, 0
        for i, sb in enumerate(arm):
            if fails >= cfg.MAX_CONSECUTIVE_FAILURES:
                for rest in arm[i:]:
                    emit({**{c: '' for c in COLS}, 'lst': lst, 'scope': rest, 'ref': '-', 'n_written': 0,
                          'status': f'NOT ATTEMPTED: arm halted after {fails} consecutive refusals'})
                break
            if step(sb, accepted[ref]):
                ref, fails = sb, 0
            else:
                fails += 1
    centres = {sb: cfg.SUBBAND_FREQ_HZ[sb]['center'] for sb in accepted if sb in cfg.SUBBAND_FREQ_HZ}
    for wb in WIDEBANDS:
        groot = _gen_root(root, input_root, lst, wb)
        p = f'{groot}/coadd_final/I_deep_Taper_R075_{wb}.fits' if groot else None
        if not centres or not p or not os.path.exists(p):
            emit({**{c: '' for c in COLS}, 'lst': lst, 'scope': wb, 'ref': '-', 'n_written': 0, 'input': groot or '',
                  'status': 'REFUSED no accepted subband master to cascade from' if centres else 'REFUSED no wideband master'})
            continue
        h = fits.getheader(p); f = float(h.get('CRVAL3') or h.get('RESTFRQ'))
        near = min(centres, key=lambda sb: abs(centres[sb] - f))
        step(wb, accepted[near])


def main():
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    ap = argparse.ArgumentParser()
    ap.add_argument('lsts', nargs='+')
    ap.add_argument('--root', default='/data07/gh/phase3/coadd_sb'); ap.add_argument('--input-root', default=None)
    ap.add_argument('--apply', action='store_true'); ap.add_argument('--apply-scopes', default='')
    ap.add_argument('--table', default=None); ap.add_argument('--no-residuals', action='store_true')
    a = ap.parse_args()
    global NO_RESIDUALS; NO_RESIDUALS = a.no_residuals
    vlssr = CD.load_vlssr_catalog(cfg.VLSSR_CATALOG, 'VLSSr')
    if vlssr is None:
        sys.exit('REFUSED: VLSSr catalogue failed to load')
    for lst in a.lsts:
        run_lst(lst, a.root, a.input_root, a.apply, set(filter(None, a.apply_scopes.split(','))),
                a.table or f'{a.root}/mdewarp_rbf_table.csv', vlssr)


if __name__ == '__main__':
    main()
