#!/usr/bin/env python3
"""
phase3_gate_ledger.py — reconstruct once, record EVERY gate verdict per raw >=5sigma V
candidate, so the veto sequence + image tagging + per-gate loss tracking can be applied
OFFLINE (and re-tuned instantly). V-only deep-wideband (as phase3_mstruct_review), NO I,
NO sigma-stack. For each raw candidate records: snr(local), mstruct value, co-located
I/V at rb=0, bilobe flag (production params, per-epoch global rms), gaussfit flag.
Usage: python phase3_gate_ledger.py <LST> <SCOPE>
"""
from exopipe.common.roots import coadd_root
import sys
_ARGV = list(sys.argv)
import os, re, glob, csv, fcntl
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from exopipe.phase3.phase3_deep import (_load, _mstruct_maps, _build_residuals_mp, _det_header, _epoch_time,
                         _qa_outlier_dates, SIGMA_MSTRUCT, NPROC_RESID, MIN_EPOCHS,
                         load_exclusions, is_excluded, pick_i_master)
from exopipe.phase3.phase3_pipeline import run_identification_v
from exopipe.phase3.identify import fit_empirical_beam
from exopipe.phase3 import phase3_veto as pv

OUT = '/data07/gh/phase3/transients/Run_GATES'
LEDGER = f'{OUT}/gate_ledger.csv'
WORKROOT = '/data07/gh/phase3/deep'
GAUSSFIT_SNR = 8.0     # gaussfit only evaluated above this (SNR_GATE)


def _append(path, header, rows):
    os.makedirs(OUT, exist_ok=True)
    lk = open(path + '.lock', 'w'); fcntl.flock(lk, fcntl.LOCK_EX)
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        w.writerows(rows)
    fcntl.flock(lk, fcntl.LOCK_UN); lk.close()


def run(lst, scope):
    base = f'{coadd_root()}/{lst}_{scope}'
    cw = f'{WORKROOT}/{lst}_{scope}'
    mIp = pick_i_master(base, scope)   # Taper_R0, or sane R075 fallback (13h/14h)
    mVp = f'{base}/coadd_final/V_deep_Taper_R0_{scope}.fits'
    if not (mIp and os.path.exists(mVp)):
        print(f'  {lst} {scope}: missing master(s) — skip'); return
    master_I = _load(mIp); hdr0 = fits.getheader(mIp); cd = abs(hdr0['CDELT1'])
    master_V = _load(mVp)
    em, en, _, _ = fit_empirical_beam(master_I)
    beam = em if np.isfinite(em) and em > 0 else 7.0
    ar_max = max(2.0, 1.5 * beam / max(en if np.isfinite(en) and en > 0 else beam, 1e-3))
    ms_destr, _ = _mstruct_maps(master_V, mVp, beam)
    h, w = master_I.shape; rbI = 0   # I/V at rb=0 (exact pixel)

    dates = [re.search(r'(\d{4}-\d\d-\d\d)_V_sub_', os.path.basename(c)).group(1)
             for c in sorted(glob.glob(f'{base}/subtracted/*_V_sub_{scope}.fits'))]
    flows = {d: np.load(f'{cw}/ep_{d}_up2_flow.npy') for d in dates
             if os.path.exists(f'{cw}/ep_{d}_up2_flow.npy')}
    dates = [d for d in dates if d in flows]
    excl = load_exclusions()
    dates = [d for d in dates if not is_excluded(lst, scope, d, excl)]
    qa_out = _qa_outlier_dates(lst, scope)
    dates = [d for d in dates if d not in qa_out]
    if len(dates) < MIN_EPOCHS:
        print(f'  {lst} {scope}: only {len(dates)} usable epochs — skip'); return

    hdr_ref = _det_header(hdr0, scope, beam, cd, None)
    epoch_V = {d: _load(f'{base}/subtracted/{d}_V_sub_{scope}.fits') + master_V for d in dates}
    resid_V = _build_residuals_mp(epoch_V, master_V, flows, hdr_ref, cd, dates, NPROC_RESID)
    dates = [d for d in dates if d in resid_V]

    rows = []
    for d in dates:
        rV = np.nan_to_num(resid_V[d], nan=0.0)
        ot = _epoch_time(base, d, scope)
        hdr = _det_header(hdr0, scope, beam, cd, ot); wcs = WCS(hdr).celestial
        # global central-field rms (as apply_veto uses for bilobe)
        cb = rV[h // 3:2 * h // 3, w // 3:2 * w // 3]; cb = cb[np.isfinite(cb)]
        grms = float(1.4826 * np.median(np.abs(cb - np.median(cb)))) if cb.size > 100 else 0.0
        pV = run_identification_v(rV, master_I, hdr, beam, None, sigma_thresh=5.0)  # raw >=5s
        for c in pV:
            x, y = int(c['pix_x']), int(c['pix_y'])
            snr = abs(float(c.get('local_snr', c.get('snr', 0))))
            msv = float(ms_destr[y, x]) if ms_destr is not None else 1.0
            V = abs(float(c.get('flux', 0)))
            iv0 = (abs(float(master_I[y, x])) / V) if V > 0 else 0.0
            bilobe = int(grms > 0 and pv.bilobe_ghost_reason(rV, x, y, beam, grms, cd) != 'keep')
            gfail = 0
            if snr > GAUSSFIT_SNR:
                gf = pv._gfit(rV, x, y, beam)
                gfail = int(bool(gf) and (gf['ar'] > ar_max or gf['resid'] > pv.RESID_MAX))
            sky = wcs.pixel_to_world(x, y)
            rows.append([lst, d, scope, round(float(sky.ra.deg), 5), round(float(sky.dec.deg), 5),
                         round(snr, 2), round(msv, 2), round(iv0, 2), bilobe, gfail])
        print(f'  {lst} {scope} {d}: {len(pV)} raw >=5s candidates', flush=True)

    _append(LEDGER, ['lst', 'date', 'band', 'ra', 'dec', 'snr', 'mstruct', 'iv0',
                     'bilobe', 'gaussfit'], rows)
    print(f'  {lst} {scope}: {len(dates)} epochs -> {len(rows)} candidate rows', flush=True)


if __name__ == '__main__':
    run(_ARGV[1], _ARGV[2])
