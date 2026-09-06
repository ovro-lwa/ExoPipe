#!/usr/bin/env python3
"""STAGE A of the ratified Master absolute-flux-scale + final-dewarp stage
(MASTER_SHFLUX_VLSSR_DEWARP_SPEC_20260828.md, Gregg 2026-08-28; implementation approved
2026-09-06 'all approved'): Scaife & Heald 2012 absolute flux scale of every deep master.

Per cell (LST x band-or-scope), MEASURE:
  * image = the PB-corrected Stokes I Taper Robust -0.75 master (coadd_final/I_deep_Taper_R075_pbcorr_<scope>.fits;
    built with phase3_pbcorr_masters.do_cell when absent, which also writes the cell's PB_deep map);
  * calibrators = the six SH12 broadband calibrators (pipeline_config.CALIB_DATA) whose elevation at the
    cell's LST midpoint (LST_mid = LST hour + 0.5 h, OVRO +37.24) exceeds SH_MIN_ELEVATION_DEG = 50 (ruled);
  * fit = the chain's own 2-D Gaussian (phase3_catalog.gaussian_photometry, no hand-rolled fitter) at the
    catalog position; the fit counts only when peak / central rms >= 5 (standing rule: fits only >= 5 sigma);
  * model frequency: subband = cfg.SUBBAND_FREQ_HZ (band labels are nominal); wideband = the master's RESTFRQ
    card (the Phase-2 stack's effective frequency), recorded with its source;
  * scale s = inverse-variance weighted mean over qualifying sources of (model / measured); the per-source
    uncertainty is s_i x (rms / peak_i); with N >= 2 the reported error is the larger of the propagated error
    and the scatter / sqrt(N).
  * REFUSAL (ruled): no qualifying calibrator, or the fit fails / is below 5 sigma on a cell's ONLY qualifying
    calibrator -> the cell is refused and escalated (row 'REFUSED' in the table, no product written); a cell is
    never left silently unscaled and never borrows a scale.

APPLY (--apply): the scale is BAKED into the pixels of every product of the cell (ruled) as NEW files - the
current generation is never rewritten in place: <cell>/<outgen>/ mirrors coadd_final/ (masters, pbcorr
siblings, loo/) and subtracted/ (residuals, clean epochs); every file name gains the 'shflux' token before the
scope, headers gain SHSCALE, SHERR, SHNSRC, SHSRCS, SHFREQ, SHFRQSRC, SHDATE and a HISTORY line. PB_deep maps
(beam response, dimensionless) are copied unscaled. The V master and every other product inherit the
I-measured scale (ruled: the calibration chain is common). Consumers keep reading the current generation until
Gregg's assessment of the measured offsets (ratified caveat), so nothing downstream changes by running this.

Record: one row per cell appended to <root>/shflux_table.csv (scale, error, sources, effective frequency,
per-source measured and model peaks) - the archived fluxscale-table class of the spec.

Usage: phase3_master_shflux.py <lst> <scope> [--root R] [--apply] [--outgen coadd_final_shflux]
       phase3_master_shflux.py --cells <file: lst scope per line> [...]"""
import argparse, csv, glob, math, os, shutil, sys, time
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
sys.path.insert(0, '/lustre/gh/main/exopipe')
from exopipe.common import pipeline_config as pc
from exopipe.common import phase2_config as cfg
from exopipe.phase3.phase3_catalog import gaussian_photometry
from exopipe.phase3.verify_mosaic import model_flux

SH_MIN_ELEVATION_DEG = 50.0     # ruled 2026-08-28
FIT_MIN_SNR = 5.0               # standing rule: Gaussian fits only >= 5 sigma
OVRO_LAT_DEG = 37.2398
DEFAULT_ROOT = '/data07/gh/phase3/coadd_sb'
SCALED_DIRS = ('coadd_final', 'subtracted')      # product trees that carry Jy/beam pixels
UNSCALED_PREFIX = ('PB_deep',)                    # beam maps: copied, never scaled


def elevation_deg(lst, ra_deg, dec_deg):
    h = int(str(lst).rstrip('h')); zra = 15.0 * (h + 0.5)
    c = (math.sin(math.radians(dec_deg)) * math.sin(math.radians(OVRO_LAT_DEG))
         + math.cos(math.radians(dec_deg)) * math.cos(math.radians(OVRO_LAT_DEG)) * math.cos(math.radians(ra_deg - zra)))
    return 90.0 - math.degrees(math.acos(max(-1.0, min(1.0, c))))


def central_rms_jy(a):
    n = a.shape[0]; q = n // 4
    c = a[q:-q, q:-q]; c = c[np.isfinite(c)]
    if c.size < 100: return float('nan')
    return float(1.4826 * np.median(np.abs(c - np.median(c))))


def master_freq_hz(scope, hdr):
    if scope in cfg.SUBBAND_FREQ_HZ:
        v = cfg.SUBBAND_FREQ_HZ[scope]; v = v.get('center', v.get('centre')) if isinstance(v, dict) else v
        return float(v), 'cfg.SUBBAND_FREQ_HZ (subband centre)'
    f = hdr.get('RESTFRQ') or hdr.get('RESTFREQ') or hdr.get('CRVAL3')
    return (float(f), 'RESTFRQ (Phase-2 stack effective frequency)') if f else (float('nan'), 'none')


NO_BUILD = False   # --no-build: never write PB maps / pbcorr siblings (measurement-only runs)


def pbcorr_master(root, lst, scope):
    cf = f'{root}/{lst}_{scope}/coadd_final'
    p = f'{cf}/I_deep_Taper_R075_pbcorr_{scope}.fits'
    if not os.path.exists(p) and not NO_BUILD:
        from exopipe.phase3 import phase3_pbcorr_masters as PBM
        n_ok, n_skip, errs = PBM.do_cell(root, f'{lst}_{scope}')
        for e in errs: print('  pbcorr:', e)
    return p if os.path.exists(p) else None


def measure(lst, scope, root):
    """Returns a dict (the table row); status 'OK' or 'REFUSED <why>'."""
    row = dict(lst=lst, scope=scope, status='', scale='', err='', nsrc=0, srcs='', freq_hz='', freq_src='',
               rms_jy='', detail='', measured_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    p = pbcorr_master(root, lst, scope)
    if not p:
        row['status'] = 'REFUSED no PB-corrected I Taper R-0.75 master'; return row
    hdr = fits.getheader(p); a = np.squeeze(fits.getdata(p)).astype(float)
    w = WCS(hdr).celestial
    cd = abs(float(hdr.get('CDELT1') or hdr.get('CD1_1'))); bpix = float(hdr['BMAJ']) / cd
    rms = central_rms_jy(a); row['rms_jy'] = f'{rms:.5f}'
    f_hz, f_src = master_freq_hz(scope, hdr); row['freq_hz'] = f'{f_hz:.0f}'; row['freq_src'] = f_src
    if not np.isfinite(f_hz):
        row['status'] = 'REFUSED no model frequency'; return row
    qual = []
    for name, d in pc.CALIB_DATA.items():
        c = d['coords']; el = elevation_deg(lst, c.ra.deg, c.dec.deg)
        if el > SH_MIN_ELEVATION_DEG: qual.append((name, d, el))
    if not qual:
        row['status'] = 'REFUSED no SH calibrator above 50 deg at this LST'; return row
    used, notes = [], []
    for name, d, el in qual:
        c = d['coords']
        try: x, y = w.world_to_pixel_values(c.ra.deg, c.dec.deg); x, y = int(round(float(x))), int(round(float(y)))
        except Exception: notes.append(f'{name}: off-frame'); continue
        if not (0 <= x < a.shape[1] and 0 <= y < a.shape[0]): notes.append(f'{name}: off-frame'); continue
        # local peak search within one beam (the catalog position vs the master's residual astrometry)
        r = max(3, int(round(bpix)))
        st = np.nan_to_num(a[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1], nan=-np.inf)
        if st.size == 0: notes.append(f'{name}: blank'); continue
        iy, ix = np.unravel_index(int(np.argmax(st)), st.shape); y0, x0 = max(0, y - r) + int(iy), max(0, x - r) + int(ix)
        g = gaussian_photometry(np.nan_to_num(a), x0, y0, bpix, cd)
        if not g or not np.isfinite(g['peak_jy']) or g['peak_jy'] <= 0: notes.append(f'{name}: fit failed'); continue
        snr = g['peak_jy'] / rms if rms > 0 else float('inf')
        if snr < FIT_MIN_SNR: notes.append(f'{name}: {snr:.1f} sigma < {FIT_MIN_SNR:.0f}'); continue
        model = float(model_flux(d['coeffs'], f_hz / 1e6, d.get('scale', 'SH12')))
        s_i = model / g['peak_jy']; sig_i = s_i * (rms / g['peak_jy'])
        used.append((name, el, g['peak_jy'], model, s_i, sig_i, snr))
    if not used:
        row['status'] = 'REFUSED ' + ('; '.join(notes) or 'no usable calibrator'); row['detail'] = '; '.join(notes); return row
    wts = np.array([1.0 / u[5] ** 2 for u in used]); s = np.array([u[4] for u in used])
    scale = float(np.sum(wts * s) / np.sum(wts)); err = float(np.sqrt(1.0 / np.sum(wts)))
    if len(used) >= 2: err = max(err, float(np.std(s, ddof=1) / math.sqrt(len(used))))
    row.update(status='OK', scale=f'{scale:.5f}', err=f'{err:.5f}', nsrc=len(used),
               srcs=';'.join(u[0] for u in used),
               detail='; '.join(f'{u[0]} el{u[1]:.0f} meas {u[2]:.2f} model {u[3]:.2f} s {u[4]:.4f}+-{u[5]:.4f} ({u[6]:.0f} sigma)' for u in used)
                      + ((' | skipped: ' + '; '.join(notes)) if notes else ''))
    return row


def _scaled_name(path, scope):
    b = os.path.basename(path); suf = f'_{scope}.fits'
    return b[:-len(suf)] + f'_shflux{suf}' if b.endswith(suf) else b.replace('.fits', '_shflux.fits')


def apply(lst, scope, root, row, outgen):
    """Write the scaled generation: every product of coadd_final/ (incl. loo/) and subtracted/."""
    cell = f'{root}/{lst}_{scope}'; out = f'{cell}/{outgen}'; n = 0
    scale = float(row['scale'])
    for sub in SCALED_DIRS:
        src = f'{cell}/{sub}'
        if not os.path.isdir(src): continue
        for p in sorted(glob.glob(f'{src}/**/*.fits', recursive=True)):
            rel = os.path.relpath(p, cell); dst_dir = f'{out}/{os.path.dirname(rel)}'; os.makedirs(dst_dir, exist_ok=True)
            base = os.path.basename(p)
            if base.startswith(UNSCALED_PREFIX):
                shutil.copy2(p, f'{dst_dir}/{base}'); continue
            with fits.open(p) as h:
                d = h[0].data; hd = h[0].header.copy()
            d = (np.asarray(d, dtype=np.float32) * np.float32(scale))
            hd['SHSCALE'] = (scale, 'SH12 absolute flux scale applied (Gregg 2026-08-28)')
            hd['SHERR'] = (float(row['err']), '1-sigma uncertainty of SHSCALE')
            hd['SHNSRC'] = (int(row['nsrc']), 'SH calibrators used (el > 50 deg)')
            hd['SHSRCS'] = (row['srcs'], 'SH calibrators used')
            hd['SHFREQ'] = (float(row['freq_hz']), f"model frequency Hz ({row['freq_src'][:40]})")
            hd['SHDATE'] = (row['measured_at'], 'flux scale measured')
            hd['HISTORY'] = f"shflux: pixels x {scale:.5f} (SH12 calibrators {row['srcs']}; I Taper R-0.75 pbcorr master; stage A of MASTER_SHFLUX_VLSSR_DEWARP_SPEC_20260828)"
            fits.writeto(f'{dst_dir}/{_scaled_name(p, scope)}', d, hd, overwrite=True); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lst', nargs='?'); ap.add_argument('scope', nargs='?')
    ap.add_argument('--cells'); ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--apply', action='store_true'); ap.add_argument('--outgen', default='coadd_final_shflux')
    ap.add_argument('--table', default=None); ap.add_argument('--no-build', action='store_true')
    a = ap.parse_args()
    global NO_BUILD; NO_BUILD = a.no_build
    cells = [(a.lst, a.scope)] if a.lst else [tuple(l.split()[:2]) for l in open(a.cells) if l.strip()]
    table = a.table or f'{a.root}/shflux_table.csv'
    cols = ['lst', 'scope', 'status', 'scale', 'err', 'nsrc', 'srcs', 'freq_hz', 'freq_src', 'rms_jy', 'n_written', 'detail', 'measured_at']
    new = not os.path.exists(table)
    for lst, scope in cells:
        row = measure(lst, scope, a.root); row['n_written'] = 0
        if row['status'] == 'OK' and a.apply:
            row['n_written'] = apply(lst, scope, a.root, row, a.outgen)
        with open(table, 'a', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            if new: wtr.writeheader(); new = False
            wtr.writerow(row)
        print(f"{lst} {scope}: {row['status']} scale {row['scale']} +- {row['err']} (N {row['nsrc']} {row['srcs']}; {row['freq_src']}; written {row['n_written']}) | {row['detail'][:200]}", flush=True)


if __name__ == '__main__':
    main()
