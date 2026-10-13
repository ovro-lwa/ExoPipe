#!/usr/bin/env python3
"""
phase3_i_gate_ledger.py — Stokes-I deep-wideband search verification + gate ledger
(mirror of phase3_gate_ledger.py for I; Gregg's 2026-07-03 rulings baked in):
  - I channel = Taper R0.75 (residual-noise test winner; stage-B archive native)
  - sigma-stack REPLACED by mstruct-I: local rms of the DESTRIPED deep I master,
    normalized to clean sky (same construction as the V mstruct)
  - records EVERY gate quantity per raw >=5sigma I candidate so the veto sequence
    can be tuned OFFLINE per band and per FP-class (Low first, per the plan)
  - doubles as the BDSF rms=nan verification run (blank-to-NaN path exercised)

v4 (Gregg's I-veto spec, 2026-07-04, handoff §5.13): detection on BOTH SIGNS (±I)
at the 5-sigma write floor (V parity; legacy chain was positive-only at 7), and
EVERY gate quantity recorded, NONE applied:
  lst, date, band, ra, dec, sign, snr, mstruct_i, sstack (info-only), master_i,
  dist_b7_beams, dist_b70_deg, neigh_min_jy (opposite-extreme within 1 beam,
  sign-relative), bilobe (production params), gaussfit (|snr|>8), n_epoch_raw
Usage: python phase3_i_gate_ledger.py <LST> <SCOPE>
"""
from exopipe.phase3.config import coadd_root
import sys
_ARGV = list(sys.argv)
import os, re, glob, csv, fcntl
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from exopipe.phase3.phase3_deep import (_load, _master_structure_map, _build_residuals_mp,
                         _mstruct_validity_guard,
                         _qa_outlier_dates, NPROC_RESID, MIN_EPOCHS,
                         load_exclusions, is_excluded)
from exopipe.phase3.phase3_tenmin import _det_header
from exopipe.phase3.phase3_pipeline import STRIATION_WINDOW_DEG
from exopipe.phase3.coadd_full_pipeline import build_striation_model, get_midpoint_time
from exopipe.phase3.core.fitting import fit_empirical_beam
from exopipe.phase3 import search_v_transients_v2 as sv2
from exopipe.phase3 import search_i_transients as sit
from exopipe.phase3 import phase3_veto as pv
from exopipe.phase3.phase3_deep import _sigma_stack

OUT = '/data07/gh/phase3/transients/Run_I_GATES'
PSFM = os.environ.get('I_LEDGER_PSFMATCH') == '1'   # matched-chain mode
LEDGER = f'{OUT}/i_gate_ledger_psfmatch.csv' if PSFM else f'{OUT}/i_gate_ledger.csv'
WORKROOT = '/data07/gh/phase3/deep'
RESID_ARCHIVE = '/data07/gh/phase3/i_resid'   # refined residuals KEPT (Gregg
# 2026-07-04: archive the search-frame residuals so every image-dependent gate
# tunes offline; reverses the V-era reconstruct-on-the-fly policy now that
# /data07 has room; ~40 MB per epoch-image)


def _append(rows):
    os.makedirs(OUT, exist_ok=True)
    lk = open(LEDGER + '.lock', 'w'); fcntl.flock(lk, fcntl.LOCK_EX)
    new = not os.path.exists(LEDGER)
    with open(LEDGER, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['lst', 'date', 'band', 'ra', 'dec', 'pix_x', 'pix_y',
                        'sign', 'snr', 'peak_jy', 'local_rms_jy',
                        'mstruct_i', 'sstack', 'master_i_jy', 'dist_b7_beams',
                        'dist_b70_deg',
                        'bl_sep_beams', 'bl_partner_sig', 'bl_ratio', 'bilobe',
                        'gf_ar', 'gf_resid', 'gf_fwhm_beams', 'gaussfit',
                        'n_epoch_raw'])
        w.writerows(rows)


def bilobe_measurables(img, x, y, beam, grms, maxsep=4.0):
    """CONTINUOUS dipole quantities (Gregg 2026-07-04: record measurables, not
    flags, so the bilobe veto tunes offline forever). Extreme opposite-sign
    pixel within `maxsep` beams (production only looks to 1.5): returns
    (sep_beams, partner_sig, lesser/brighter ratio) or ('','','')."""
    h, w = img.shape
    W = int(round(maxsep * beam))
    if not (W <= x < w - W and W <= y < h - W) or grms <= 0:
        return '', '', ''
    box = img[y - W:y + W + 1, x - W:x + W + 1]
    if not np.isfinite(box).any():
        return '', '', ''
    cw = max(1, int(round(0.3 * beam)))
    cp = img[y - cw:y + cw + 1, x - cw:x + cw + 1]
    cp = cp[np.isfinite(cp)]
    if cp.size == 0:
        return '', '', ''
    camp = float(np.nanmax(cp)) / grms          # sign-relative frame: cand is +
    b = np.where(np.isfinite(box), box, 0)
    yy, xx = np.indices(box.shape)
    dist = np.hypot(yy - W, xx - W) / beam
    b = np.where((dist > 0.5) & (dist <= maxsep), b, 0)
    j = np.unravel_index(np.argmin(b), b.shape)
    pv = float(b[j])
    if pv >= 0:
        return '', '', ''
    part = abs(pv) / grms
    lesser, brighter = min(camp, part), max(camp, part)
    return (round(float(dist[j]), 2), round(part, 2),
            round(lesser / max(brighter, 1e-9), 3))
    fcntl.flock(lk, fcntl.LOCK_UN); lk.close()


def _mstruct_i(master, mp, beam):
    """Destriped-I-master local-rms map (falls back to raw on destripe failure).

    Validity comes from the MASTER for both maps (problem-ledger P12, Gregg-approved
    2026-08-18): the running-median striation model reproduces its input exactly on
    ~0.8% of pixels, and inferring validity from `!= 0` on the difference read those
    as unimaged and handed them the gate floor."""
    vmask = np.isfinite(master) & (master != 0)
    raw = _master_structure_map(master, beam, valid=vmask)
    try:
        hh = fits.getheader(mp); mwcs = WCS(hh).celestial
        _hm, _am, iwm, _ot, _c, _e = sv2.make_masks(hh, master.shape, fits_path=mp)
        sm, _ = build_striation_model(master, mwcs, iwm,
                                      ra_window_beams=20.0, beam_pix=beam)
    except Exception as e:
        print(f'    [mstruct-I] destripe failed ({e}); raw map used')
        return raw
    dif = master - sm
    n_leak = int((vmask & (dif == 0)).sum())
    if n_leak:
        print(f'    [mstruct-I] P12: {n_leak:,} imaged pixels '
              f'({100.0 * n_leak / max(int(vmask.sum()), 1):.2f}%) are reproduced exactly '
              f'by the striation model; validity taken from the master')
    rep = {}
    destriped = _master_structure_map(dif, beam, valid=vmask, report=rep)
    _mstruct_validity_guard(rep, vmask, mp, 'I')
    return destriped


_PSFM_CTX = None


def _psfm_worker(d):
    from exopipe.phase3.phase3_i_psfmatch_ab import matched_resid
    epoch, master, flows, hdr_ref, cd, bm, base, scope = _PSFM_CTX
    try:
        ee, ee2, eepa, _ = fit_empirical_beam(epoch[d])
        be = (ee if np.isfinite(ee) and ee > 0 else bm[0],
              ee2 if np.isfinite(ee2) and ee2 > 0 else bm[1],
              eepa if np.isfinite(eepa) else 0.0)
        sp = f'{base}/subtracted/{d}_I_sub_{scope}.fits'
        ot = get_midpoint_time(fits.getheader(sp), sp)
        r, bpe, fdil, dc2 = matched_resid(epoch[d], master, flows[d],
                                          hdr_ref, cd, be, bm, mfs_args=ot)
        return d, r.astype(np.float32), (be, bpe, fdil, dc2), None
    except Exception as e:
        return d, None, None, str(e)


def run(lst, scope):
    base = f'{coadd_root()}/{lst}_{scope}'
    cw = f'{WORKROOT}/{lst}_{scope}'
    mp = f'{base}/coadd_final/I_deep_Taper_R075_{scope}.fits'
    if not os.path.exists(mp):
        print(f'  {lst} {scope}: no R075 I master — skip'); return
    master = _load(mp); hdr0 = fits.getheader(mp); cd = abs(hdr0['CDELT1'])
    em, en, mpa, _ = fit_empirical_beam(master)
    beam = em if np.isfinite(em) and em > 0 else 7.0
    bm = (beam, en if np.isfinite(en) and en > 0 else beam,
          mpa if np.isfinite(mpa) else 0.0)
    ms = None if PSFM else _mstruct_i(master, mp, beam)

    dates = [re.search(r'(\d{4}-\d\d-\d\d)_I_sub_', os.path.basename(c)).group(1)
             for c in sorted(glob.glob(f'{base}/subtracted/*_I_sub_{scope}.fits'))]
    flows = {d: np.load(f'{cw}/ep_{d}_up2_flow.npy') for d in dates
             if os.path.exists(f'{cw}/ep_{d}_up2_flow.npy')}
    dates = [d for d in dates if d in flows]
    excl = load_exclusions()
    dates = [d for d in dates if not is_excluded(lst, scope, d, excl)]
    qa = _qa_outlier_dates(lst, scope)
    dates = [d for d in dates if d not in qa]
    if len(dates) < MIN_EPOCHS:
        print(f'  {lst} {scope}: only {len(dates)} usable epochs — skip'); return

    hdr_ref = _det_header(hdr0, scope, beam, cd, None)
    epoch = {d: _load(f'{base}/subtracted/{d}_I_sub_{scope}.fits') + master
             for d in dates}
    if PSFM:
        from exopipe.phase3.phase3_i_psfmatch_ab import gk
        from astropy.convolution import convolve_fft
        import multiprocessing as _mp
        cap = int(os.environ.get('I_LEDGER_MAX_EPOCHS', '0'))
        if cap and len(dates) > cap:
            print(f'  [psfm] {lst} {scope}: SAMPLING first {cap} of '
                  f'{len(dates)} usable epochs (I_LEDGER_MAX_EPOCHS)', flush=True)
            dates = dates[:cap]
            epoch = {d: epoch[d] for d in dates}
        global _PSFM_CTX
        _PSFM_CTX = (epoch, master, flows, hdr_ref, cd, bm, base, scope)
        resid, ebeams = {}, {}
        with _mp.get_context('fork').Pool(3) as pool:
            for d, r, meta, err in pool.imap_unordered(_psfm_worker, dates):
                if err:
                    print(f'  [psfm] {d}: FAILED {err}', flush=True)
                    continue
                resid[d] = r.astype(np.float64); ebeams[d] = meta
                print(f'  [psfm] {d}: dil {meta[2]:.3f} DC2 {meta[3]:.4f}',
                      flush=True)
        _PSFM_CTX = None
        med_be = (float(np.median([ebeams[d][0][0] for d in resid])),
                  float(np.median([ebeams[d][0][1] for d in resid])),
                  float(np.median([ebeams[d][0][2] for d in resid])))
        ms = _mstruct_i(convolve_fft(master, gk(*med_be), allow_huge=True,
                                     normalize_kernel=True), mp, beam)
        beam = float(np.hypot(bm[0], med_be[0]))   # detection beam = matched
    else:
        resid = _build_residuals_mp(epoch, master, flows, hdr_ref, cd, dates,
                                    NPROC_RESID)
    del epoch
    dates = [d for d in dates if d in resid]
    # sstack: INFO-ONLY column (never a gate; Gregg spec §5.13), from the epoch
    # residual ensemble — same construction as run_deep_cell's sstack_I
    cwk = f'{WORKROOT}/{lst}_{scope}'
    sstack = _sigma_stack(resid, cwk, hdr0, 'Iv4') if len(resid) >= MIN_EPOCHS else None

    h, w = master.shape
    rows = []
    for d in dates:
        rI = np.nan_to_num(resid[d], nan=0.0)
        # per-epoch time block: sit.make_masks computes the elevation mask at
        # DATE-OBS + INTTIME/2, and the MASTER header's INTTIME is the summed
        # stack (~8-22 h) -> mask evaluated at a bogus LST (18h Red: fully
        # masked center -> rms=nan -> zero candidates; found 2026-07-03).
        # Use the epoch's own DATE-OBS + single-block span.
        try:
            eh = fits.getheader(f'{base}/subtracted/{d}_I_sub_{scope}.fits')
            ot = eh.get('DATE-OBS')
            span = float(eh.get('INTTIME') or 3591.0)
        except Exception:
            ot, span = None, 3591.0
        hdr = _det_header(hdr0, scope, beam, cd, ot)
        hdr['INTTIME'] = span
        hdr['DURATION'] = span
        wcs = WCS(hdr).celestial
        # archive the refined residual (search frame) with a conformant name
        try:
            from exopipe.phase3.core import products as prod
            ad = f'{RESID_ARCHIVE}/{lst}_{scope}'
            os.makedirs(ad, exist_ok=True)
            rn = prod.build_name(scope, 'I', 'deep', 'Taper', -0.75,
                                 proc=['dewarped', 'aligned', 'msub', 'ssub'],
                                 lst=lst, epoch=ot)
            rp = f'{ad}/psfmatchv4_{d}_I_resid.fits' if PSFM else f'{ad}/{rn}'
            if not os.path.exists(rp):
                wh = wcs.to_header()
                fits.writeto(rp, resid[d].astype(np.float32), wh, overwrite=True)
                prod.write_standard_header(
                    rp, scope=scope, stokes='I',
                    start=eh.get('DATE-OBS'), end=eh.get('DATE-END'),
                    inttime_s=span, time_basis='phase2',
                    beam=(beam * cd, beam * cd, 0.0),
                    aligned=True, alignref=os.path.basename(mp),
                    msub=True, msubfile=mp, ssub=True, dewarped=True,
                    validate=True,
                    extra={'RUNLABEL': ('Run_I_GATES', 'I veto development'),
                           'EPOCH_ID': (f'{lst}_{d}', 'LST_date epoch')})
        except Exception as e:
            print(f'  [resid archive] {d}: {e}', flush=True)
        # ± detection at the 6-sigma write floor (Gregg spec §5.13 as amended);
        # no veto applied — every gate is a recorded column.
        try:
            horizon, ateam, badwcs, obs_time, _el = sit.make_masks(
                hdr, rI.shape, wcs)
            excl = horizon | ateam | badwcs
            rms0 = sit.get_uniform_rms(rI, horizon)
            if not np.isfinite(rms0) or rms0 <= 0:
                print(f'  {lst} {scope} {d}: rms={rms0} — mask/geometry problem',
                      flush=True)
                continue
            cands = []
            for sgn, img in ((1, rI), (-1, -rI)):
                cs = sit.extract_positive_peaks(
                    img.astype(np.float32), hdr, horizon, ateam, badwcs,
                    wcs, obs_time, rms0, beam)
                # 5-sigma RECORDING floor (one-way-door protection: a 5s ledger
                # contains every >=6s analysis as a filter; analyses/candidate
                # lists are quoted at Gregg's 6s+)
                sit.assign_local_snr(cs, img, excl, beam, 5.0)
                sit.merge_nearby(cs, beam, wcs)
                for c in cs:
                    if c['status'] == 'PASS':
                        c['sgn'] = sgn
                        cands.append(c)
        except Exception as e:
            print(f'  {lst} {scope} {d}: detection FAILED: {e}', flush=True)
            continue
        # bilobe global rms (production convention: central-box MAD)
        cb = rI[h // 3:2 * h // 3, w // 3:2 * w // 3]
        cb = cb[np.isfinite(cb)]
        grms = (1.4826 * float(np.median(np.abs(cb - np.median(cb))))
                if cb.size > 100 else 0.0)
        ar_max = max(2.0, 1.5 * beam / max(en if np.isfinite(en) and en > 0
                                           else beam, 1e-3))
        # bright-source distances (recorded, NOT applied)
        if not hasattr(run, '_bright'):
            b7 = sit.find_bright_sources(master, horizon, wcs, beam,
                                         sit.CONFIG['BRIGHT_MASK_FLUX_JY'])
            run._bright = (
                np.array([[s['pix_x'], s['pix_y']] for s in b7], float)
                if b7 else np.empty((0, 2)),
                np.array([[s['pix_x'], s['pix_y']] for s in b7
                          if s['flux'] >= sit.CONFIG['BRIGHT_EXCL_FLUX_JY']],
                         float) if b7 else np.empty((0, 2)))
        B7, B70 = run._bright
        for c in cands:
            x, y = int(c['pix_x']), int(c['pix_y'])
            sgn = c['sgn']
            snr = abs(float(c.get('local_snr', c.get('snr', 0))))
            lr = float(c.get('local_rms', np.nan))
            pk = abs(float(c.get('flux', 0.0)))
            simg = rI if sgn > 0 else -rI          # sign-relative frame
            d7 = (float(np.min(np.hypot(B7[:, 0] - x, B7[:, 1] - y))) / beam
                  if len(B7) else np.inf)
            d70 = (float(np.min(np.hypot(B70[:, 0] - x, B70[:, 1] - y))) * cd
                   if len(B70) else np.inf)
            # continuous dipole measurables (searched to 4 beams) + prod flag
            bsep, bsig, brat = bilobe_measurables(simg, x, y, beam, grms)
            bl = int(grms > 0 and
                     pv.bilobe_ghost_reason(simg, x, y, beam, grms, cd) != 'keep')
            # continuous shape measurables for snr>=6 (trigger tunes 6..N) + flag
            gar = gres = gfw = ''
            gf = 0
            if snr >= 6.0:
                g = pv._gfit(simg, x, y, beam)
                if g:
                    gar = round(g['ar'], 2); gres = round(g['resid'], 3)
                    gfw = round(g['fwhm'] / beam, 2)
                    gf = int(g['ar'] > ar_max or g['resid'] > pv.RESID_MAX)
            sky = wcs.pixel_to_world(x, y)
            rows.append([lst, d, scope,
                         round(float(sky.ra.deg), 5), round(float(sky.dec.deg), 5),
                         x, y, '+' if sgn > 0 else '-',
                         round(snr, 2),
                         round(sgn * pk, 4),
                         round(lr, 4) if np.isfinite(lr) else '',
                         round(float(ms[y, x]), 2),
                         round(float(sstack[y, x]), 2) if sstack is not None else '',
                         round(float(master[y, x]), 3),
                         round(d7, 2) if np.isfinite(d7) else '',
                         round(d70, 3) if np.isfinite(d70) else '',
                         bsep, bsig, brat, bl,
                         gar, gres, gfw, gf, len(cands)])
        print(f'  {lst} {scope} {d}: {len(cands)} raw +-I candidates (5s floor)',
              flush=True)
    _append(rows)
    print(f'  {lst} {scope}: {len(dates)} epochs -> {len(rows)} ledger rows', flush=True)


if __name__ == '__main__':
    run(_ARGV[1], _ARGV[2])
