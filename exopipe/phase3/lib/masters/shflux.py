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
current generation is never rewritten in place: <cell>/<outgen>/ (default gen_shflux) mirrors coadd_final/ (masters, pbcorr
siblings, loo/) and subtracted/ (residuals, clean epochs); every file name gains the 'shflux' token before the
scope, headers gain SHSCALE, SHERR, SHNSRC, SHSRCS, SHFREQ, SHFRQSRC, SHDATE and a HISTORY line. PB_deep maps
(beam response, dimensionless) are copied unscaled. The V master and every other product inherit the
I-measured scale (ruled: the calibration chain is common). Consumers keep reading the current generation until
Gregg's assessment of the measured offsets (ratified caveat), so nothing downstream changes by running this.

Record: one row per cell appended to <root>/shflux_table.csv (scale, error, sources, effective frequency,
per-source measured and model peaks) - the archived fluxscale-table class of the spec.

Usage: phase3_master_shflux.py <lst> <scope> [--root R] [--apply] [--outgen gen_shflux]
       phase3_master_shflux.py --cells <file: lst scope per line> [...]"""
import argparse, csv, glob, math, os, shutil, sys, time
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
sys.path.insert(0, '/lustre/gh/main/exopipe')
from exopipe.common import pipeline_config as pc
from exopipe.common import phase2_config as cfg
from exopipe.phase3.lib.search.candidates import gaussian_photometry
from exopipe.phase3.lib.mosaic.verify_mosaic import model_flux

SH_MIN_ELEVATION_DEG = 50.0     # ruled 2026-08-28
FIT_MIN_SNR = 5.0               # standing rule: Gaussian fits only >= 5 sigma
OVRO_LAT_DEG = 37.2398
DEFAULT_ROOT = '/data07/gh/phase3/coadd_sb'
SCALED_DIRS = ('coadd_final', 'subtracted')      # product trees that carry Jy/beam pixels
NO_RESIDUALS = False                              # --no-residuals: masters + LOO + pbcorr only (Gregg 2026-09-06: residuals baked at push time)
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
        from exopipe.phase3.lib.masters import pbcorr_masters as PBM
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


def apply(lst, scope, root, row, outgen, corr_t=None):
    """Write the scaled generation: every product of coadd_final/ (incl. loo/) and subtracted/.
    From 2026-10-01 (Gregg: "Yes, bake it into the pixels in the same re-bake. It was always my intention to have the flux
    scale correction applied to pixels.") the pixels carry SHSCALE x FLUXCORR (k x g of the flux-correction table) and the
    FLUX* cards with FLUXBAKE = True; without the table row for the scope the apply is REFUSED."""
    if corr_t is None or scope not in corr_t:
        raise RuntimeError(f'{lst} {scope}: apply without the flux-correction table row for {scope} - the correction is baked '
                           f'into the pixels (Gregg 2026-10-01); REFUSING')
    cell = f'{root}/{lst}_{scope}'; out = f'{cell}/{outgen}'; n = 0
    for sub in (('coadd_final',) if NO_RESIDUALS else SCALED_DIRS):
        src = f'{cell}/{sub}'
        if not os.path.isdir(src): continue
        # MIRROR, NOT OVERLAY (2026-09-10, INCIDENT-SHFLUX-OUTGEN-KEPT-STALE-LOO-MASTERS): an earlier
        # apply into the same <outgen> left files whose source no longer exists (leave-one-out masters
        # of epochs the rebuilt stack rejected); they were promoted and archived as generation-4
        # products and the search refuses them by digest. Anything under <outgen>/<sub> that this
        # apply will not (re)write is removed first, so the output is exactly the scaled source.
        _expect = set()
        for p in glob.glob(f'{src}/**/*.fits', recursive=True):
            rel = os.path.relpath(p, cell); b = os.path.basename(p)
            _expect.add(f'{out}/{os.path.dirname(rel)}/' + (b if b.startswith(UNSCALED_PREFIX) else _scaled_name(p, scope)))
        for p in glob.glob(f'{out}/{sub}/**/*.fits', recursive=True):
            if p not in _expect:
                os.remove(p); print(f'    purged stale {outgen} file (no source in {sub}/): {os.path.relpath(p, cell)}', flush=True)
        for p in sorted(glob.glob(f'{src}/**/*.fits', recursive=True)):
            rel = os.path.relpath(p, cell); dst_dir = f'{out}/{os.path.dirname(rel)}'; os.makedirs(dst_dir, exist_ok=True)
            base = os.path.basename(p)
            if base.startswith(UNSCALED_PREFIX):
                shutil.copy2(p, f'{dst_dir}/{base}'); continue
            with fits.open(p) as h:
                d = h[0].data; hd = h[0].header.copy()
            d, hd = bake(d, hd, scope, row, corr_t)
            fits.writeto(f'{dst_dir}/{_scaled_name(p, scope)}', d, hd, overwrite=True); n += 1
    return n


def bake(d, hd, scope, row, corr_t):
    """One product's pixels and header in the scaled generation (the body of apply's loop; D-ADMIT bakes an admitted night's
    products with it): pixels x SHSCALE x FLUXCORR as float32, the FLUX* cards (flux_correction.cards_for, FLUXBAKE True),
    the SH* cards of the scale-table row and the two HISTORY lines; under the output standard (coadd.HEADER_STANDARD 'full',
    which the masters step sets from masters_build.HEADER_STANDARD) the noise cards THERMNOI and IMGRMS scaled with the
    pixels, so they stay in the product's units. ``hd`` is changed in place and returned."""
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
    from exopipe.phase3.lib.masters import coadd as _co     # WAVE2 OUTPUT (writers step 1): under the output standard a
    if _co.HEADER_STANDARD == 'full':                        # noise card follows its pixels - THERMNOI and IMGRMS scaled by
        _nk = [k for k in ('THERMNOI', 'IMGRMS') if hd.get(k) is not None]   # the factor the pixels were (production: kept)
        for k in _nk:
            hd[k] = (float(round(float(hd[k]) * scale * corr, 9)), hd.comments[k])
        if _nk:
            hd['HISTORY'] = f"noise cards {', '.join(_nk)} x {scale * corr:.6f} with the pixels (output standard)"
    return d, hd


def _md5_bytes(b):
    import hashlib
    return hashlib.md5(b).hexdigest()


def rebake_file(src, dst, scope, s_old, srow, corr_t, write=False, tmp_dir=None, s_old_from_cell=False):
    """FLUX RE-BAKE of one product (Gregg 2026-10-01: "Re-measure and re-bake the flux scale" and "Yes, bake it into the
    pixels in the same re-bake"): pixels x (s_new / s_old) x k x g, where s_old is the per-LST VLSSr factor the pixels
    carry (SHSCALE), s_new the re-measured one (srow: the scale-table row) and k x g the flux-correction table's FLUXCORR
    for the scope. Read src, write dst (the same path for the working tree; the archive reads the quarantined copy).
    Gates (refuse, never warn): PB maps are never scaled; a product without SHSCALE, or whose SHSCALE differs from s_old,
    is REFUSED; a product already carrying FLUXBAKE is 'ALREADY' only when SHSCALE, FLUXCORR and FLUXVER equal the
    targets exactly, otherwise REFUSED. Verification at write time: the written file is read back and must equal
    old x factor exactly (float, NaN-aware) with the target cards before it replaces dst (os.replace from a .part in
    dst's own directory, so no reader ever sees a partial file).
    s_old_from_cell: the archived search frames (epoch - leave-one-out master - striation model) are made from the cell's
    scaled products but were written without the SH cards; for them only, a missing SHSCALE means the cell's s_old (recorded
    in the detail) and the re-baked frame gains the cards.
    Returns dict(status, factor, md5_old, md5_new, detail)."""
    import io
    from exopipe.phase3.lib.masters import flux_correction as FC
    out = dict(status='', factor='', md5_old='', md5_new='', detail='')
    if os.path.basename(src).startswith(UNSCALED_PREFIX):
        out['status'] = 'SKIP-PB'; return out
    s_new = float(srow['scale']); corr = float(corr_t[scope]['corr'])
    if write:
        raw = open(src, 'rb').read(); out['md5_old'] = _md5_bytes(raw)
        with fits.open(io.BytesIO(raw), memmap=False) as h:
            hd = h[0].header.copy(); d = h[0].data
    else:                                   # dry run: header only (the pixels are read when written)
        hd = fits.getheader(src); d = np.zeros(1, dtype=np.float32 if hd.get('BITPIX') == -32 else
                                                 np.float64 if hd.get('BITPIX') == -64 else np.int32)
    _note = ''
    if hd.get('SHSCALE') is None:
        if not s_old_from_cell:
            out.update(status='REFUSED', detail='no SHSCALE card: the scale in the pixels is unknown'); return out
        hd['SHSCALE'] = s_old; _note = 'SHSCALE card absent (search frame): the cell factor in force assumed; '
    if FC.pixels_carry_correction(hd):
        same = (abs(float(hd['SHSCALE']) - s_new) < 5e-6 and abs(float(hd.get('FLUXCORR', -1)) - corr) < 5e-7
                and str(hd.get('FLUXVER', '')) == corr_t[scope]['version'][:68])
        out.update(status='ALREADY' if same else 'REFUSED',
                   detail='FLUXBAKE with the target cards' if same else
                   f"FLUXBAKE already set with other values (SHSCALE {hd['SHSCALE']}, FLUXCORR {hd.get('FLUXCORR')})")
        return out
    if abs(float(hd['SHSCALE']) - s_old) > 5e-5:
        out.update(status='REFUSED', detail=f"SHSCALE {hd['SHSCALE']} != the LST's factor in force {s_old}"); return out
    if d is None or d.dtype.kind != 'f':
        out.update(status='REFUSED', detail=f'data not floating point ({None if d is None else d.dtype})'); return out
    f = (s_new / s_old) * corr; out['factor'] = f'{f:.7f}'
    new = (d * d.dtype.type(f)).astype(d.dtype)
    hd['SHSCALE'] = (round(s_new, 6), 'per-LST VLSSr flux scale (re-measured 2026-10-01, PB per robust)')
    hd['SHERR'] = (float(srow['err']), '1-sigma uncertainty of SHSCALE')
    hd['SHNSRC'] = (int(srow['nsrc']), 'VLSSr matches in the window')
    hd['SHSRCS'] = (f"VLSSr {srow['window']}"[:68], 'scale reference (window)')
    hd['SHDATE'] = (srow['measured_at'], 'flux scale measured')
    hd['SHMETHOD'] = ('VLSSr-dec20-60', 'flux scale method')
    for _c, _v, _com in FC.cards_for(scope, corr_t, baked=True):
        hd[_c] = (_v, _com)
    hd['HISTORY'] = (f"flux re-bake {time.strftime('%Y-%m-%d', time.gmtime())}: pixels x {f:.6f} = SHSCALE {s_old:.5f} -> "
                     f"{s_new:.5f} x FLUXCORR {corr:.5f} (k {float(corr_t[scope]['k']):.5f} x g {float(corr_t[scope]['g']):.5f}); "
                     f"Gregg 2026-10-01")
    if not write:
        out.update(status='DRY-RUN OK', detail=f'{_note}x {f:.6f}'); return out
    part = f'{tmp_dir or os.path.dirname(dst)}/.{os.path.basename(dst)}.rebake.part'
    fits.writeto(part, new, hd, overwrite=True, output_verify='silentfix')
    try:
        with fits.open(part, memmap=False) as h2:
            back = h2[0].data; bh = h2[0].header
            if back.shape != d.shape or not np.array_equal(back, d * d.dtype.type(f), equal_nan=True):
                raise RuntimeError('read-back pixels differ from old x factor')
            if not (FC.pixels_carry_correction(bh) and abs(float(bh['FLUXCORR']) - corr) < 5e-7
                    and abs(float(bh['SHSCALE']) - s_new) < 5e-6):
                raise RuntimeError('read-back cards differ from the targets')
        out['md5_new'] = _md5_bytes(open(part, 'rb').read())
        os.replace(part, dst)
    except Exception as e:
        if os.path.exists(part):
            os.remove(part)
        out.update(status='FAILED', detail=str(e)[:200]); return out
    out.update(status='REBAKED', detail=f'{_note}x {f:.6f}')
    return out


def rebake_cell(lst, scope, root, s_old, srow, corr_t, write=False, record=None):
    """Re-bake every product of the cell's canonical trees (SCALED_DIRS: coadd_final incl. loo/, subtracted/) in place.
    Returns the per-file rows (also appended to record)."""
    cell = f'{root}/{lst}_{scope}'; rows = []
    for sub in SCALED_DIRS:
        for p in sorted(glob.glob(f'{cell}/{sub}/**/*.fits', recursive=True)):
            r = rebake_file(p, p, scope, s_old, srow, corr_t, write=write)
            r.update(cell=f'{lst}_{scope}', path=p); rows.append(r)
    if record:
        new = not os.path.exists(record)
        with open(record, 'a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['cell', 'path', 'status', 'factor', 'md5_old', 'md5_new', 'detail'])
            if new:
                w.writeheader()
            w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lst', nargs='?'); ap.add_argument('scope', nargs='?')
    ap.add_argument('--cells'); ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--apply', action='store_true'); ap.add_argument('--outgen', default='gen_shflux')
    ap.add_argument('--table', default=None); ap.add_argument('--no-build', action='store_true'); ap.add_argument('--no-residuals', action='store_true')
    # --scale-table (R-FLUX-SCALE-APPLY-VLSSR-HIGH-ELEVATION-APPROVED, Gregg 2026-09-07): the per-LST factor
    # from the VLSSr ratio in the +20..+60 deg declination window (columns lst, scale, err, nsrc, window, ref)
    # REPLACES the SH12 measurement; one factor for every scope of the LST hour; the apply path is unchanged.
    ap.add_argument('--scale-table', default=None)
    # --fluxcorr-table (Gregg 2026-10-01): the k x g table baked with the scale (required by --apply and --rebake).
    # --rebake --old-scale-table T: re-bake the existing products in place from the factor in force (T) to --scale-table
    # x FLUXCORR; dry run unless --write; per-file record in --table.
    ap.add_argument('--fluxcorr-table', default=None); ap.add_argument('--rebake', action='store_true')
    ap.add_argument('--old-scale-table', default=None); ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    corr_t = None
    if a.fluxcorr_table:
        from exopipe.phase3.lib.masters import flux_correction as FC
        corr_t = FC.load_table(a.fluxcorr_table)
    if a.rebake:
        if not (a.scale_table and a.old_scale_table and corr_t and a.table):
            raise SystemExit('--rebake needs --scale-table, --old-scale-table, --fluxcorr-table and --table; REFUSING')
        from collections import Counter
        new_t = {r['lst']: r for r in csv.DictReader(open(a.scale_table))}
        old_t = {r['lst']: float(r['scale']) for r in csv.DictReader(open(a.old_scale_table))}
        cells = [(a.lst, a.scope)] if a.lst else [tuple(l.split()[:2]) for l in open(a.cells) if l.strip()]
        for lst, scope in cells:
            if lst not in new_t or lst not in old_t or scope not in corr_t:
                print(f'{lst} {scope}: REFUSED - no factor (new scale {lst in new_t}, old scale {lst in old_t}, '
                      f'fluxcorr {scope in corr_t})', flush=True)
                continue
            rows = rebake_cell(lst, scope, a.root, old_t[lst], new_t[lst], corr_t, write=a.write, record=a.table)
            print(f'{lst} {scope}: ' + ', '.join(f'{k} {v}' for k, v in sorted(Counter(r["status"] for r in rows).items())), flush=True)
        return
    scale_tab = None
    if a.scale_table:
        scale_tab = {r['lst']: r for r in csv.DictReader(open(a.scale_table))}
    global NO_BUILD, NO_RESIDUALS; NO_BUILD = a.no_build; NO_RESIDUALS = a.no_residuals
    cells = [(a.lst, a.scope)] if a.lst else [tuple(l.split()[:2]) for l in open(a.cells) if l.strip()]
    table = a.table or f'{a.root}/shflux_table.csv'
    cols = ['lst', 'scope', 'status', 'scale', 'err', 'nsrc', 'srcs', 'freq_hz', 'freq_src', 'rms_jy', 'n_written', 'detail', 'measured_at']
    new = not os.path.exists(table)
    for lst, scope in cells:
        if scale_tab is not None:
            t = scale_tab.get(lst)
            row = dict(lst=lst, scope=scope, status='OK' if t else 'REFUSED no factor for this LST in the scale table',
                       scale=(t['scale'] if t else ''), err=(t['err'] if t else ''), nsrc=(t['nsrc'] if t else 0),
                       srcs=(f"VLSSr {t['window']}" if t else ''), freq_hz=75.41e6,
                       freq_src=(t.get('ref', 'VLSSr 73 MHz subband, one factor per LST') if t else ''), rms_jy='',
                       detail=f'scale table {os.path.basename(a.scale_table)}', measured_at=(t['measured_at'] if t else ''))
        else:
            row = measure(lst, scope, a.root)
        row['n_written'] = 0
        if row['status'] == 'OK' and a.apply:
            row['n_written'] = apply(lst, scope, a.root, row, a.outgen, corr_t)
        with open(table, 'a', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            if new: wtr.writeheader(); new = False
            wtr.writerow(row)
        print(f"{lst} {scope}: {row['status']} scale {row['scale']} +- {row['err']} (N {row['nsrc']} {row['srcs']}; {row['freq_src']}; written {row['n_written']}) | {row['detail'][:200]}", flush=True)


if __name__ == '__main__':
    main()
