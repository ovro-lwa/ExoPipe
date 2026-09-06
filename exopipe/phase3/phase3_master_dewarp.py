#!/usr/bin/env python3
"""STAGE B of the ratified Master absolute-flux-scale + final-dewarp stage
(MASTER_SHFLUX_VLSSR_DEWARP_SPEC_20260828.md, Gregg 2026-08-28; implementation approved 2026-09-06):
the FINAL master dewarp measured on the cell's Stokes I Taper Robust -0.75 master and applied to ALL the
cell's products (ruled: 'the derived correction applies to ALL the data').

RUNS UNDER THE py310_orca ENVIRONMENT (JAX + image_plane_correction, GPU), like the Phase-2 cascade:
    conda run -n py310_orca python -m exopipe.phase3.phase3_master_dewarp <lst> <scope> [--root R] [--gen gen_shflux] [--apply]

Per cell:
  1 input master = <cell>/<gen>/coadd_final/I_deep_Taper_R075[_shflux]_<scope>.fits (the flux-scaled generation root
    when present, else the cell's current coadd_final) - non-PB-corrected, as the Phase-2 anchor step uses;
  2 BEFORE metric = the pipeline's VLSSr peak match (the reference-quality gate's measurement, coadd_full_pipeline
    _reference_astrom_ok): median separation (arcmin) of the 50 brightest central peaks from the nearest VLSSr
    source, and the number of matches within 60 arcmin; fewer than 10 matches = REFUSED (never a silent fallback);
  3 flow = image_plane_correction.flow.calcflow on the master against the VLSSR synthetic sky
    (cleaned=True, use_best_pb_model=True, catalog_path = vlssr_radecpeak_unresolved.txt, the Phase-2 flow default) - the same machinery
    and settings as the Phase-2 anchor step (flow_cascade_dewarp.run_anchor);
  4 AFTER metric = the same VLSSr match on the dewarped master; the gate REFUSES on regression
    (after median >= before median) and the cell stays unwarped and flagged;
  5 --apply: the ONE flow resamples every FITS product of the generation (coadd_final incl. loo/ and pbcorr
    siblings and the PB map, subtracted/ residuals and clean epochs) into <cell>/<gen>_mdewarp/ (or gen_mdewarp for the current generation; new files, the
    'mdewarp' token before the scope; MDWARP/MDWMED0/MDWMED1/MDWNANC/MDWP50/MDWP95 cards + HISTORY); the flow
    offsets are saved as <outgen>/mdewarp_flow_<scope>.npy with a diagnostic PNG.
  6 one row per cell in <root>/mdewarp_table.csv (before/after offsets, anchors, warp amplitude percentiles).
"""
import argparse, csv, glob, os, sys, time
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
sys.path.insert(0, '/lustre/gh/main/exopipe')
from exopipe.common import pipeline_config as pc

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


def run(lst, scope, root, gen, apply, table):
    import jax.numpy as jnp
    from image_plane_correction.flow import calcflow
    cell = f'{root}/{lst}_{scope}'
    # generation root: <cell>/<gen>/{coadd_final,subtracted} (stage A layout) or the cell itself (current generation)
    groot = f'{cell}/{gen}' if os.path.isdir(f'{cell}/{gen}/coadd_final') else cell
    gdir = f'{groot}/coadd_final'; sdir_in = f'{groot}/subtracted'
    gen_used = gen if groot != cell else 'current'
    cands = [f'{gdir}/I_deep_Taper_R075_shflux_{scope}.fits', f'{gdir}/I_deep_Taper_R075_{scope}.fits']
    img = next((p for p in cands if os.path.exists(p)), None)
    row = dict(lst=lst, scope=scope, gen=gen_used, status='', med_before='', n_before='', p90_before='', med_after='', n_after='',
               p90_after='', warp_p50_as='', warp_p95_as='', n_written=0, seconds='', measured_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    if not img:
        row['status'] = 'REFUSED no I Taper R-0.75 master'; return row
    t0 = time.time()
    with fits.open(img) as h: d0 = np.squeeze(h[0].data).astype(np.float32); hdr = h[0].header.copy()
    cd = abs(float(hdr.get('CDELT1') or hdr.get('CD1_1'))) * 3600.0
    m0, n0, p0 = vlssr_offsets(d0, hdr); row.update(med_before=f'{m0:.3f}', n_before=n0, p90_before=f'{p0:.3f}')
    if n0 < MIN_ANCHORS:
        row['status'] = f'REFUSED {n0} VLSSr anchors < {MIN_ANCHORS} before dewarp'; return row
    try:
        image, ref_sky, flow, dewarped = calcflow(img, cleaned=True, use_best_pb_model=True, qa=False, write=False,
                                                  catalog='VLSSR', catalog_path=VLSSR_PEAKS)[:4]   # same file as the Phase-2 flow default (flow_cascade_dewarp --catalog_path)
    except Exception as e:
        row['status'] = f'REFUSED calcflow failed: {str(e)[:120]}'; return row
    off = np.asarray(flow.offsets); amp = np.hypot(off[..., 0], off[..., 1]) * cd
    n = amp.shape[0]; q = n // 8; ac = amp[q:-q, q:-q]
    row.update(warp_p50_as=f'{np.percentile(ac, 50):.1f}', warp_p95_as=f'{np.percentile(ac, 95):.1f}')
    d1 = np.asarray(dewarped, dtype=np.float32)
    m1, n1, p1 = vlssr_offsets(d1, hdr); row.update(med_after=f'{m1:.3f}', n_after=n1, p90_after=f'{p1:.3f}')
    if not (np.isfinite(m1) and m1 < m0) or n1 < MIN_ANCHORS:
        row['status'] = f'REFUSED regression: median VLSSr sep {m0:.2f} -> {m1:.2f} arcmin (N {n0} -> {n1}); cell left unwarped'
        row['seconds'] = f'{time.time() - t0:.0f}'; return row
    row['status'] = 'OK'
    if apply:
        out = f'{cell}/{gen_used}_mdewarp' if gen_used != 'current' else f'{cell}/gen_mdewarp'; os.makedirs(out, exist_ok=True)
        np.save(f'{out}/mdewarp_flow_{scope}.npy', off.astype(np.float32))
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            plt.imsave(f'{out}/mdewarp_flow_{scope}.png', np.asarray(flow.to_rgb()))
        except Exception: pass
        cards = dict(MDWARP=(True, 'final master dewarp applied (Gregg 2026-08-28)'), MDWMED0=(m0, 'median VLSSr sep before, arcmin'),
                     MDWMED1=(m1, 'median VLSSr sep after, arcmin'), MDWNANC=(n1, 'VLSSr matches after'),
                     MDWP50=(float(row['warp_p50_as']), 'warp amplitude p50, arcsec'), MDWP95=(float(row['warp_p95_as']), 'warp amplitude p95, arcsec'),
                     MDWDATE=(row['measured_at'], 'dewarp measured'))
        nw = 0
        for sdir in (gdir, sdir_in):
            if not os.path.isdir(sdir): continue
            for p in sorted(glob.glob(f'{sdir}/**/*.fits', recursive=True)):
                rel = os.path.relpath(p, groot)
                dst = f'{out}/{os.path.dirname(rel)}'; os.makedirs(dst, exist_ok=True)
                with fits.open(p) as h: d = np.squeeze(h[0].data).astype(np.float32); hh = _fix2d(h[0].header)
                if d.shape != off.shape[:2]: continue
                fin = np.isfinite(d); fill = float(np.median(d[fin])) if fin.any() else 0.0
                dd = np.nan_to_num(d, nan=fill, posinf=fill, neginf=fill)
                w = np.asarray(flow.apply(jnp.array(dd, dtype=jnp.float32))).astype(np.float32)
                if not fin.all(): w[~fin] = np.nan
                for k, v in cards.items(): hh[k] = v
                hh['HISTORY'] = f'mdewarp: resampled with the cell flow measured on {os.path.basename(img)} vs VLSSr (stage B of MASTER_SHFLUX_VLSSR_DEWARP_SPEC_20260828)'
                fits.writeto(f'{dst}/{_out_name(p, scope)}', w, hh, overwrite=True); nw += 1
        row['n_written'] = nw
    row['seconds'] = f'{time.time() - t0:.0f}'
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lst', nargs='?'); ap.add_argument('scope', nargs='?'); ap.add_argument('--cells')
    ap.add_argument('--root', default='/data07/gh/phase3/coadd_sb'); ap.add_argument('--gen', default='gen_shflux')
    ap.add_argument('--apply', action='store_true'); ap.add_argument('--table', default=None)
    a = ap.parse_args()
    cells = [(a.lst, a.scope, a.root)] if a.lst else [tuple((l.split() + [a.root])[:3]) for l in open(a.cells) if l.strip()]
    table = a.table or f'{a.root}/mdewarp_table.csv'
    cols = ['lst', 'scope', 'gen', 'status', 'med_before', 'n_before', 'p90_before', 'med_after', 'n_after', 'p90_after', 'warp_p50_as', 'warp_p95_as', 'n_written', 'seconds', 'measured_at']
    new = not os.path.exists(table)
    for lst, scope, root in cells:
        row = run(lst, scope, root, a.gen, a.apply, table)
        with open(table, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            if new: w.writeheader(); new = False
            w.writerow(row)
        print(f"{lst} {scope} [{row['gen']}]: {row['status']} | VLSSr median {row['med_before']} -> {row['med_after']} arcmin (N {row['n_before']} -> {row['n_after']}); warp p50/p95 {row['warp_p50_as']}/{row['warp_p95_as']} arcsec; written {row['n_written']}; {row['seconds']} s", flush=True)


if __name__ == '__main__':
    main()
