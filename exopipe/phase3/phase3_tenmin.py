#!/usr/bin/env python3
"""
phase3_tenmin.py — production 10-MIN detection tier (full independent step).

Validated recipe (tenmin_validation + tenmin_scurve), updated 2026-06-18 to the
mandated Phase 3 standards:
  - reference = SUB-LST MASTER built INLINE from the INCLUDED (outlier-rejected)
    epochs' aligned intervals — the SAME outlier set as deep's coadd_final
    (excludes outliers.csv), inverse-variance weighted with per-pixel 5-MAD clip.
    (No provisional master is used; provisional was all-dates.)
  - deep I master read from coadd_final/ (R0 wideband / R075 subband).
  - NO striation (no recovery gain, adds FP on the thermal-limited 10-min sky).
  - per-interval sigma_stack from the cross-epoch residual ensemble.
  - I/V identification + the validated morphology veto -> survey catalog.
  - EVERY delivered image (per-epoch aligned + msub, I & V, and the per-interval
    masters) written through phase3_products with the FULL standard header and
    the interval-MIDPOINT timestamp (lst_bin). Images go to PRODUCT_ROOT
    (/data07; /fast NVMe is full); the catalog stays on the normal staging root.

Per (lst, scope): for each Stokes, align every INCLUDED epoch's intervals
(flux-scale + saved flow), build the outlier-rejected sub-LST master per
interval, subtract -> residual, build that interval's sigma_stack, search each
epoch's residual, write candidates (cadence='10min', interval midpoint) and the
conformant image products.
"""
import os, sys, glob, re, csv, fcntl
_ARGV = list(sys.argv)        # capture before lab imports that reset sys.argv
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from exopipe.phase3.lib.masters import coadd as C
from exopipe.phase3.lib.search.identification import build_sigma_stack, run_identification_i, run_identification_v
from exopipe.phase3.core.fitting import fit_empirical_beam
from exopipe.phase3.tenmin_validation import setup_date_params, align_date_intervals
from exopipe.phase3.lib.search import veto as pv
from exopipe.phase3.lib.search import candidates as pc
from exopipe.phase3.core import products as pp
from exopipe.phase3 import config as P
from exopipe.common import phase2_config as cfg
from exopipe.phase3 import leakage_check as _LC
from exopipe.phase3.lib.search import search_v as sv2
from astropy.coordinates import SkyCoord as _SC
from scipy.ndimage import maximum_filter as _maxf

# A-team post-veto gate (adopted 2026-07-15, same stage as phase3_deep /
# phase3_i_deep): applied AFTER morphology veto, never as a pre-detection
# mask. Run search waves with P3_ATEAM_MASK_OFF=1 at this tier boundary.
_ATEAM_COORDS = _SC([sv2.CONFIG['A_TEAM'][k]['coord'].ra.deg
                     for k in sv2.CONFIG['A_TEAM']],
                    [sv2.CONFIG['A_TEAM'][k]['coord'].dec.deg
                     for k in sv2.CONFIG['A_TEAM']], unit='deg')
ATEAM_VETO_DEG = {'V': 2.0, 'I': 5.0}
# --- env-guarded side-by-side veto compare (Gregg 2026-06-21) ---
_ISRC = {}
def _isrc_for(master_i, bpix):
    k = id(master_i)
    if k not in _ISRC:
        ms = max(int(bpix*1.5), 5); lm = _maxf(master_i, size=ms)
        ys, xs = np.where((master_i == lm) & (master_i >= 5.0))
        _ISRC[k] = [{'pix_y':int(y),'pix_x':int(x),'flux':float(master_i[y,x]),
                     'app_flux':float(master_i[y,x])} for y,x in zip(ys,xs)]
    return _ISRC[k]
def _compare_dump(passed, rclean, master_i, hdr, em, cd, wcs, ot, ar_max,
                  lst, scope, date, stokes, idx):
    kept, _ = pv.apply_veto(passed, rclean, master_i, hdr, em, cd, wcs, ot, ar_max=ar_max)
    curids = {id(c) for c in kept}; isrc = _isrc_for(master_i, em)
    msx, msf = pv.bright_source_list(master_i)
    vr = float(np.median(np.abs(rclean - np.median(rclean)))*1.4826)
    out = f'/fast/gh/main/phase3/transients/Run_PROD_compare/tenmin_{lst}_{scope}_{stokes}.csv'
    newf = not os.path.exists(out)
    with open(out, 'a', newline='') as f:
        w = csv.writer(f)
        if newf: w.writerow(['lst','scope','stokes','date','interval','ra','dec',
                             'peak_jy','snr','current','newmethod','new_verdict'])
        for c in passed:
            cy, cx = int(c['pix_y']), int(c['pix_x']); pk = float(c['flux'])
            try: co = wcs.pixel_to_world(cx, cy); ra, dec = float(co.ra.deg), float(co.dec.deg)
            except Exception: ra, dec = 0.0, 0.0
            # NEW = current exclusion/shape (veto_reason rms=0, keeps A-team/sidelobe) + leakage_check
            r0 = pv.veto_reason(c, rclean, msx, msf, em, cd, ar_max, rms=0.0, dipmap=None)
            nv = r0 if r0 != 'keep' else _LC.transient_leakage_verdict(cy, cx, pk, rclean, master_i, isrc, em, cd, vr)[0]
            w.writerow([lst, scope, stokes, date, idx, round(ra,4), round(dec,4),
                        round(pk,3), round(float(c.get('snr', c.get('local_snr',0))),1),
                        'keep' if id(c) in curids else 'VETO',
                        'VETO' if nv in ('bright_mask','on_source','gaussfit','leakage','bilobe') else 'keep', nv])

MIN_EPOCHS = int(os.environ.get('P3_MIN_EPOCHS', '5'))
# ^ env-overridable (2026-07-17, Gregg ruling): 00h runs with P3_MIN_EPOCHS=2
#   (n_epochs=2 cell — include and flag, don't drop). Default 5 unchanged.
PRODUCT_ROOT = os.environ.get('P3_PRODUCT_ROOT', '/data07/gh/phase3')
# ^ env-overridable (2026-07-13) so staged nodes (01/02, no /data07) can run
#   the tier and sync products back to 07, same pattern as the subband fleet.

# ── 10-MIN GATE LEDGER (V-parity mandate, Gregg 2026-07-11/13: "record
# everything for posterity" — built into this tier BEFORE its production run).
# One row per identified candidate with the gate verdicts; one row per
# (image=lst,scope,stokes,date,interval) with counts at each gate. Appended
# under an fcntl lock next to master_catalog.csv in the run root.
GLED10_COLS = ['lst', 'scope', 'stokes', 'date', 'interval', 'ra', 'dec',
               'pix_x', 'pix_y', 'snr', 'peak_jy', 'morph_pass',
               'veto_reason', 'final']
GIMG10_COLS = ['lst', 'scope', 'stokes', 'date', 'interval', 'n_ident',
               'n_after_morph', 'n_final']


def _append_locked(path, cols, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lk = open(path + '.lock', 'w')
    fcntl.flock(lk, fcntl.LOCK_EX)
    try:
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(cols)
            w.writerows(rows)
    finally:
        fcntl.flock(lk, fcntl.LOCK_UN)
        lk.close()


def _gled10_row(lst, scope, stokes, d, idx, c, morph_pass, reason='', final=0):
    snr = c.get('local_snr', c.get('snr', ''))
    ra = dec = ''
    if c.get('ra') is not None:
        ra, dec = round(float(c['ra']), 5), round(float(c['dec']), 5)
    elif c.get('coord') is not None:
        ra = round(float(c['coord'].ra.deg), 5)
        dec = round(float(c['coord'].dec.deg), 5)
    return [lst, scope, stokes, d, idx, ra, dec,
            int(c.get('pix_x', -1)), int(c.get('pix_y', -1)),
            round(float(snr), 2) if snr != '' else '',
            round(float(c.get('flux', c.get('peak_jy', 0))), 3),
            int(morph_pass), reason, int(final)]
MAD_CLIP = 5.0                       # per-pixel master outlier clip


def _read_outliers(wd):
    """Residual-rms outlier dates from the deep pass (outliers.csv) — excluded
    so the 10-min reference matches deep's coadd_final included set."""
    p = f'{wd}/outliers.csv'
    out = set()
    if os.path.exists(p):
        with open(p) as f:
            for row in csv.DictReader(f):
                d = (row.get('date') or '').strip()
                if d:
                    out.add(d)
    return out


def _p2run(path):
    if path and 'Science_' in path:
        return 'Science_' + path.split('Science_')[1].split('/')[0]
    return None


def _interval_meta(sci_dir, scope, stokes, idx):
    """Phase 2 10-min image for this interval -> the correct time block,
    THERMNOI, provenance, and the interval MIDPOINT (for the filename token)."""
    for p, i in C.find_10min_images(sci_dir, scope, stokes):
        if i != idx:
            continue
        try:
            h = fits.getheader(p)
        except Exception:
            return None
        start, end = h.get('DATE-OBS'), h.get('DATE-END')
        mjdmid = h.get('MJD-MID')
        if mjdmid is not None:
            mid = Time(float(mjdmid), format='mjd', scale='utc')
        elif start and end:
            mid = Time(np.mean([Time(start).mjd, Time(end).mjd]),
                       format='mjd', scale='utc')
        elif start:
            mid = Time(start, scale='utc')
        else:
            mid = None
        return dict(path=p, start=start, end=end, inttime=h.get('INTTIME'),
                    thermnoi=h.get('THERMNOI'), mid=mid, p2run=_p2run(p))
    return None


def _det_header(hdr0, scope, em, cd, obs_time):
    """Detection header: frequency block (BDSF needs it) + per-epoch time."""
    if scope in cfg.WIDEBAND_BANDS_HZ:
        lo, hi = cfg.WIDEBAND_BANDS_HZ[scope]
    elif scope in cfg.SUBBAND_FREQ_HZ:
        d = cfg.SUBBAND_FREQ_HZ[scope]; lo, hi = d['lo'], d['hi']
    else:
        lo = hi = 5e7
    h = hdr0.copy()
    # 2-D products must not carry stray 3rd/4th-axis WCS keys (WCSAXES=2
    # + CTYPE3 makes wcslib/DS9 reject the whole WCS; found 2026-09-03 on
    # the battery vresid frames, ruled clean at source)
    for _ax in (3, 4):
        for _k in ('CTYPE', 'CRVAL', 'CRPIX', 'CDELT', 'CUNIT'):
            if f'{_k}{_ax}' in h:
                del h[f'{_k}{_ax}']
    if obs_time:
        h['DATE-OBS'] = obs_time
    h['RESTFRQ'] = h['CRVAL3'] = (lo + hi) / 2.0
    h['CTYPE3'] = 'FREQ'; h['CDELT3'] = max(hi - lo, 1.0); h['CRPIX3'] = 1.0
    h['CUNIT3'] = 'Hz'; h['BMAJ'] = h['BMIN'] = em * cd; h['BPA'] = 0.0
    return h


def _coadd_master(arrs):
    """Outlier-rejected sub-LST interval master from the INCLUDED epochs'
    aligned intervals: per-pixel MAD_CLIP-MAD reject + inverse-variance weighted
    mean (per-epoch weight from central MAD). Leave-in, same included set as
    deep -> 'the same outlier-rejected (INCOADD) generation as deep'."""
    stk = np.stack([np.where(np.isfinite(a) & (a != 0), a, np.nan).astype(np.float32)
                    for a in arrs])
    n = stk.shape[0]
    w = np.zeros(n)
    for i in range(n):
        s = stk[i]; h, wd = s.shape
        c = s[h // 3:2 * h // 3, wd // 3:2 * wd // 3]
        c = c[np.isfinite(c)]
        if c.size > 100:
            mad = 1.4826 * np.median(np.abs(c - np.median(c)))
            w[i] = 1.0 / mad ** 2 if mad > 0 else 0.0
    med = np.nanmedian(stk, axis=0)
    spread = 1.4826 * np.nanmedian(np.abs(stk - med), axis=0)
    keep = np.isfinite(stk) & (np.abs(stk - med) <= MAD_CLIP * spread)
    wexp = w[:, None, None]
    num = np.nansum(np.where(keep, stk * wexp, 0.0), axis=0)
    den = np.nansum(np.where(keep, wexp, 0.0), axis=0)
    return np.where(den > 0, num / den, 0.0).astype(np.float32), int((w > 0).sum())


def _write_product(data, wcs_hdr, lst, date, scope, stokes, idx, kind, meta,
                   ebeam, label, msubfile=None):
    """One per-epoch image product + the full validated standard header."""
    od = P.epoch_dir(PRODUCT_ROOT, lst, date, label, scope, stokes, '10min', make=True)
    proc = ['dewarped', 'aligned'] if kind == 'aligned' else ['dewarped', 'msub']
    name = pp.build_name(scope, stokes, '10min', 'Taper', 0.0, proc=proc,
                         lst=lst, lst_bin=idx, epoch=meta['mid'])
    path = os.path.join(od, name)
    fits.writeto(path, np.nan_to_num(data, nan=0.0).astype(np.float32),
                 wcs_hdr, overwrite=True)
    h, wd = data.shape
    inn = data[h // 4:3 * h // 4, wd // 4:3 * wd // 4]
    pp.write_standard_header(
        path, scope=scope, stokes=stokes,
        start=meta['start'], end=meta['end'], inttime_s=meta['inttime'],
        time_basis='phase2', ebeam=ebeam, thermnoi=meta['thermnoi'],
        imgrms=float(np.nanstd(inn)), dewarped=True, aligned=True,
        msub=(kind == 'msub'), msubfile=msubfile,
        alignref='sub-LST 10min co-add (P3)', lstbin=idx, p2run=meta['p2run'],
        validate=True)
    return path


def _write_master(master, wcs_hdr, lst, scope, stokes, idx, dr0, dr1, nep,
                  ebeam, label, start=None, end=None, inttime=None):
    """The per-interval sub-LST master, written conformantly (co-add product)."""
    od = P.coadd_dir(PRODUCT_ROOT, lst, label, dr0, dr1, scope, stokes,
                     '10min', make=True)
    name = pp.build_name(scope, stokes, '10min', 'Taper', 0.0, proc=['dewarped'],
                         lst=lst, lst_bin=idx, date_start=dr0, date_end=dr1,
                         n_epochs=nep)
    path = os.path.join(od, name)
    fits.writeto(path, np.nan_to_num(master).astype(np.float32), wcs_hdr,
                 overwrite=True)
    h, wd = master.shape
    inn = master[h // 4:3 * h // 4, wd // 4:3 * wd // 4]
    pp.write_standard_header(
        path, scope=scope, stokes=stokes, start=start, end=end,
        inttime_s=inttime, time_basis='phase2', ebeam=ebeam,
        imgrms=float(np.nanstd(inn)), dewarped=True, aligned=True, lstbin=idx,
        extra={'NSTACK': nep}, validate=True)
    return os.path.basename(path)


def run_tenmin_search(lst, scope, label='PROD', included_dates=None,
                      write_products=True):
    """Production 10-min search for one (lst, scope). Returns n candidates."""
    ctx = setup_date_params(lst, scope)
    fl = ctx['fl']; cd = ctx['cd']; wd = ctx['workdir']
    _irob = 'R0' if scope in ('Red', 'Green', 'Blue', 'Full') else 'R075'
    coadd_i = f'{wd}/coadd_final/I_deep_Taper_{_irob}_{fl}.fits'   # REPOINTED
    if not os.path.exists(coadd_i):
        print(f"  {lst} {fl}: no deep I master (coadd_final/{_irob}) — skip 10min")
        return 0
    master_i = C.load_image(coadd_i)[0]
    hdr0 = fits.getheader(coadd_i)
    em, en, ep, _nm = fit_empirical_beam(master_i)
    if not np.isfinite(em):
        print(f"  {lst} {fl}: corrupt deep master — skip 10min")
        return 0
    bpix = em; ar_max = max(2.0, 1.5 * em / max(en, 1e-3))
    ebeam = (em * cd, en * cd, ep)

    dates_all = [d for d, _sd, _img in ctx['dates']]
    sci_by_date = {d: sd for d, sd, _img in ctx['dates']}
    # ── 2-stage-QA night set (fixed 2026-06-27) ──────────────────────────────
    # The per-interval reference master is built from the SAME deep 2-stage-QA
    # good-date set (residuals_provisional.csv non-outliers) intersected, per
    # interval, with the 10-min per-interval 2-stage QA (residuals_provisional_
    # 10min.csv). This replaces the OLD single-pass outliers.csv, which left the
    # 10-min night set inconsistent with deep (deep-rejected nights leaked into
    # the 10-min reference -> cand-#11-class false positives). The SEARCH still
    # runs on every epoch; only the master (reference) is restricted to survivors.
    from exopipe.phase3.lib.masters.coadd import (load_exclusions, is_excluded,
                                     get_good_dates_for_interval)
    _excl = load_exclusions()
    base = set(included_dates) if included_dates else set(dates_all)
    incl_dates = [d for d in dates_all if d in base
                  and not is_excluded(lst, scope, d, _excl)]   # epochs to align+search
    deep_good = get_good_dates_for_interval(wd, interval=None)  # deep 2-stage QA good
    if deep_good is None:
        deep_good = set(incl_dates)
    root = pc.P.STAGING_ROOT
    run_root = pc.P.transients_run(root, label, make=True)
    gled_rows, gimg_rows = [], []
    nwritten = 0

    for stokes in ['I', 'V']:
        idxs = sorted({int(re.search(r'_int(\d+)_', p).group(1)) for p in
                       glob.glob(f'{wd}/coadd_provisional/{stokes}_10min_int*_{fl}.fits')})
        if not idxs:
            continue
        # align ALL intervals per date once (flux scale computed once/date)
        aligned_by_date = {}
        for date in incl_dates:
            try:
                al = align_date_intervals(ctx, date, scope, stokes, idxs)
            except ValueError as e:
                # e.g. 19h Full 2025-05-10: raw-failsafe epoch on the WRONG
                # GRID (Phase-2 anomaly, Full built on Green grid) — skip
                # the epoch, never the cell
                print(f'  [{lst} {scope}] {date}: GRID/ALIGN skip ({e})',
                      flush=True)
                continue
            if al:
                aligned_by_date[date] = al
        if len(aligned_by_date) < MIN_EPOCHS:
            continue
        dr0 = min(aligned_by_date).replace('-', '')
        dr1 = max(aligned_by_date).replace('-', '')

        for idx in idxs:
            arrs = {d: a[idx] for d, a in aligned_by_date.items() if idx in a}
            if len(arrs) < MIN_EPOCHS:
                # loud skip (2026-08-11: the silent continue hid the 22h
                # 18MHz bin5 threshold-skip; refusals/skips are never silent)
                print(f"  [tenmin] SKIP int{idx:02d}: only {len(arrs)} epochs "
                      f"< MIN_EPOCHS={MIN_EPOCHS} — no master/search for "
                      f"this interval")
                continue
            # interval metadata (time / THERMNOI / provenance) per epoch
            metas = {}
            for d in arrs:
                m = _interval_meta(sci_by_date[d], scope, stokes, idx)
                if m is not None and m['mid'] is not None:
                    metas[d] = m
            arrs = {d: a for d, a in arrs.items() if d in metas}
            if len(arrs) < MIN_EPOCHS:
                continue
            # master = QA survivors only: deep 2-stage-good AND this interval's
            # 2-stage-good (per-interval residual-rms QA). Relax gracefully if too
            # few survivors. The search below still runs on ALL arrs epochs.
            pig = get_good_dates_for_interval(wd, interval=idx)
            mset = set(arrs) & deep_good
            if pig is not None:
                mset &= set(pig)
            if len(mset) < MIN_EPOCHS:
                mset = set(arrs) & deep_good           # relax: drop per-interval QA
            if len(mset) < MIN_EPOCHS:
                mset = set(arrs)                        # last resort: all epochs
            sm, nused = _coadd_master([arrs[d] for d in arrs if d in mset])
            residuals = {d: np.where((a == 0) | (sm == 0) | ~np.isfinite(a),
                                     np.nan, a - sm) for d, a in arrs.items()}
            # per-interval sigma_stack from the cross-epoch residual ensemble
            tmp = []
            for d, r in residuals.items():
                tp = f'/fast/gh/main/tmp/ss10_{fl}_{stokes}_{idx}_{d}_{os.getpid()}.fits'
                fits.writeto(tp, np.nan_to_num(r).astype(np.float32), hdr0, overwrite=True)
                tmp.append(tp)
            sstack = build_sigma_stack(tmp) if len(tmp) >= MIN_EPOCHS else None
            for tp in tmp:
                os.remove(tp)

            # conformant per-interval master (date span + summed INTTIME)
            mfile = None
            if write_products:
                starts = [metas[d]['start'] for d in arrs if metas[d]['start']]
                ends = [metas[d]['end'] for d in arrs if metas[d]['end']]
                itt = sum(float(metas[d]['inttime']) for d in arrs
                          if metas[d]['inttime'] is not None)
                mhdr = _det_header(hdr0, scope, em, cd, None)
                mfile = _write_master(sm, mhdr, lst, scope, stokes, idx,
                                      dr0, dr1, nused, ebeam, label,
                                      start=min(starts) if starts else None,
                                      end=max(ends) if ends else None,
                                      inttime=itt or None)

            for date, r in residuals.items():
                meta = metas[date]
                obs_time = meta['start']
                hdr = _det_header(hdr0, scope, em, cd, obs_time)
                wcs = WCS(hdr).celestial
                rclean = np.nan_to_num(r, nan=0.0)

                if write_products:
                    _write_product(arrs[date], hdr, lst, date, scope, stokes,
                                   idx, 'aligned', meta, ebeam, label)
                    _write_product(rclean, hdr, lst, date, scope, stokes, idx,
                                   'msub', meta, ebeam, label, msubfile=mfile)

                if stokes == 'I':
                    passed = run_identification_i(rclean, master_i, hdr, bpix, sstack)
                else:
                    passed = run_identification_v(rclean, master_i, hdr, bpix, sstack)
                if not passed:
                    gimg_rows.append([lst, fl, stokes, date, idx, 0, 0, 0])
                    continue
                if os.environ.get('P3_COMPARE') and stokes == 'V':
                    _compare_dump(passed, rclean, master_i, hdr, em, cd, wcs,
                                  obs_time, ar_max, lst, date, scope, idx)
                    continue
                kept, _vetoed = pv.apply_veto(passed, rclean, master_i, hdr, em,
                                              cd, wcs, obs_time, ar_max=ar_max)
                n_morph = len(kept)
                gated = []
                for c in kept:
                    adeg = float(min(wcs.pixel_to_world(
                        int(c['pix_x']), int(c['pix_y']))
                        .separation(_ATEAM_COORDS).deg))
                    c['nearest_ateam'] = round(adeg, 2)
                    if adeg < ATEAM_VETO_DEG[stokes]:
                        c['veto_reason'] = 'ateam'
                        gled_rows.append(_gled10_row(lst, fl, stokes, date, idx,
                                                     c, 1, reason='ateam'))
                        continue
                    gated.append(c)
                kept = gated
                for c in kept:
                    gled_rows.append(_gled10_row(lst, fl, stokes, date, idx,
                                                 c, 1, final=1))
                for c in _vetoed:
                    gled_rows.append(_gled10_row(lst, fl, stokes, date, idx, c,
                                                 0, reason=c.get('veto_reason', '')))
                gimg_rows.append([lst, fl, stokes, date, idx, len(passed),
                                  n_morph, len(kept)])
                if kept:
                    pc.write_candidates(root, label, kept, rclean, master_i, hdr,
                                        lst, date, scope, stokes, '10min', em, cd,
                                        obs_time, interval=idx)
                    nwritten += len(kept)
            print(f"    {lst} {fl} {stokes} int{idx:02d}: "
                  f"{len(arrs)} epochs (master from {nused})", flush=True)
        del aligned_by_date
    _append_locked(f'{run_root}/v_10min_ledger.csv', GLED10_COLS, gled_rows)
    _append_locked(f'{run_root}/v_10min_images.csv', GIMG10_COLS, gimg_rows)
    print(f"  {lst} {fl} 10min: wrote {nwritten} candidates "
          f"({len(gled_rows)} gate-ledger rows, {len(gimg_rows)} image rows)",
          flush=True)
    return nwritten


if __name__ == '__main__':
    lst, scope = _ARGV[1], _ARGV[2]
    label = _ARGV[3] if len(_ARGV) > 3 else 'PROD'
    run_tenmin_search(lst, scope, label)
