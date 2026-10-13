#!/usr/bin/env python3
"""
phase3_v_c50.py — BLIND large-N Stokes-V completeness for any band/cell, and the
producer of the MEASURED C50 table the catalog reads (phase3_c50).

Usage:  python phase3_v_c50.py <Band> [<LST> <DATE>]
  Band only -> auto-picks the cleanest available cell (lowest V-sub inner rms).

Method (faithful to phase3_deep.run_deep_cell's V path, mstruct sole gate):
  ~N_PER_FRAME*N_FRAMES BOTH-SIGN injections per cell at UNIFORM-RANDOM positions
  over the SEARCH area only (horizon/A-team/<20deg masks) — BLIND: no knowledge of
  mI brightness or mV structure; the only non-mask constraint is a min separation
  between our own injections. inject -> reconstruct production V residual -> BDSF
  (both signs) -> local-SNR>=6 -> 20x I-leakage -> morphology veto -> mstruct gate.
  Recovery binned by LOCAL-rms SNR; raw 50% crossing (NO logistic fit). The final
  (mstruct) crossing is recorded as the band's measured C50.
"""
from exopipe.phase3.config import coadd_root
import sys
_ARGV = list(sys.argv)   # capture BEFORE heavy imports — phase3_deep clobbers sys.argv at import
import os, csv, copy, glob, re
from datetime import datetime, timezone
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from exopipe.phase3.phase3_deep import _load, _residual, _mstruct_maps, SIGMA_MSTRUCT
from exopipe.phase3.phase3_tenmin import _det_header
from exopipe.phase3.core.fitting import fit_empirical_beam
from exopipe.phase3.core.residuals import inner_rms, gaussian_stamp
from exopipe.phase3 import phase3_veto as pv
from exopipe.phase3 import search_v_transients_v2 as sv2
from exopipe.phase3 import phase3_c50

N_PER_FRAME, N_FRAMES = 150, 4
SNR_LO, SNR_HI = 3.0, 12.0
MIN_SEP = 80
WORKROOT = '/data07/gh/phase3/deep'
OUT = '/data07/gh/phase3/transients/Run_REDO2/scurve_inspect'
STAGES = ['bdsf', 'localsnr', 'runidv', 'veto', 'mstruct']
METHOD = 'blind, mstruct-sole-gate, raw 50% crossing'


def pick_cell(band):
    """Cleanest (lowest V-sub inner-rms) cell with a cached flow for this band."""
    best = None
    for cw in sorted(glob.glob(f'{WORKROOT}/*_{band}')):
        lst = os.path.basename(cw).split('_')[0]
        base = f'{coadd_root()}/{lst}_{band}'
        for fl in glob.glob(f'{cw}/ep_*_up2_flow.npy'):
            d = re.search(r'ep_(\d{4}-\d\d-\d\d)_', os.path.basename(fl)).group(1)
            vsub = f'{base}/subtracted/{d}_V_sub_{band}.fits'
            if not os.path.exists(vsub):
                continue
            try:
                rms = inner_rms(fits.getdata(vsub).squeeze())
            except Exception:
                continue
            if np.isfinite(rms) and rms > 0 and (best is None or rms < best[2]):
                best = (lst, d, float(rms))
    return best


def gate_chain(peaks, rV, mI, hdr, beam, cd, wcs, ot, blank, ms_destr, ar_max):
    cg = copy.deepcopy(peaks)
    # FINAL veto (Gregg 2026-06-30): write floor local-SNR>=5; NO I-leakage rule
    # (ROC: zero marginal FP value at high mult, severe injection loss at low mult);
    # mstruct + morphology veto carry the FP rejection.
    sv2.assign_local_snr(cg, rV, blank, beam, 5.0)            # local-SNR>=5 write floor
    s_localsnr = [c for c in cg if c.get('status') == 'PASS']
    s_runidv = s_localsnr                                     # leakage rule OFF
    kept, _ = pv.apply_veto(s_runidv, rV, mI, hdr, beam, cd, wcs, ot, ar_max=ar_max)
    if ms_destr is not None:                                 # mstruct SOLE gate
        s_mstruct = [c for c in kept
                     if abs(float(c.get('local_snr', c.get('snr', 0))))
                     >= SIGMA_MSTRUCT * max(float(ms_destr[int(c['pix_y']), int(c['pix_x'])]), 1.0)]
    else:
        s_mstruct = kept
    return dict(localsnr=s_localsnr, runidv=s_runidv, veto=kept, mstruct=s_mstruct)


def main():
    band = _ARGV[1]
    if len(_ARGV) >= 4:
        lst, date = _ARGV[2], _ARGV[3]
    else:
        pc = pick_cell(band)
        if pc is None:
            print(f"no cell found for {band}"); return
        lst, date, _ = pc
    print(f"[cell] {lst} {date} {band}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    base = f'{coadd_root()}/{lst}_{band}'
    cw = f'{WORKROOT}/{lst}_{band}'
    mIp = f'{base}/coadd_final/I_deep_Taper_R0_{band}.fits'
    mVp = f'{base}/coadd_final/V_deep_Taper_R0_{band}.fits'
    h0 = fits.getheader(mIp); cd = abs(h0['CDELT1'])
    mI = _load(mIp); mV = _load(mVp)
    em, en, _, _ = fit_empirical_beam(mI); beam = em if np.isfinite(em) and em > 0 else 7.0
    ar_max = max(2.0, 1.5 * beam / max(en if np.isfinite(en) and en > 0 else beam, 1e-3))
    ms_destr, _ = _mstruct_maps(mV, mVp, beam)
    hdr = _det_header(h0, band, beam, cd, f'{date}T10:00:00'); wcs = WCS(hdr).celestial
    ot = f'{date}T10:00:00'
    hor, at, bw, _a, _b, _c = sv2.make_masks(hdr, mV.shape, fits_path=mVp)
    searchable = ~(hor | at | bw); blank = hor | at | bw
    subV = _load(f'{base}/subtracted/{date}_V_sub_{band}.fits')
    epV0 = subV + mV; sig0 = inner_rms(subV)
    epI = _load(f'{base}/subtracted/{date}_I_sub_{band}.fits') + mI
    eb, ebi, ebp, _ = fit_empirical_beam(epI)
    ebeam = (eb * cd, ebi * cd, ebp) if np.isfinite(eb) else (beam * cd, beam * cd, 0.0)
    flow = np.load(f'{cw}/ep_{date}_up2_flow.npy'); n = epV0.shape[0]
    print(f"[setup] beam={beam:.1f}px ar_max={ar_max:.1f} sig0={sig0*1e3:.0f}mJy", flush=True)

    rows = []
    for fr in range(N_FRAMES):
        rng = np.random.default_rng(1000 + fr)
        inj = []; tries = 0
        while len(inj) < N_PER_FRAME and tries < 60000:
            tries += 1
            x = int(rng.integers(120, n - 120)); y = int(rng.integers(120, n - 120))
            if not searchable[y, x]:                          # BLIND: only the search masks
                continue
            if any((x - px) ** 2 + (y - py) ** 2 < MIN_SEP ** 2 for px, py, _, _ in inj):
                continue                                      # avoid self-blending only
            s = float(rng.uniform(SNR_LO, SNR_HI))
            sgn = 1 if (len(inj) % 2 == 0) else -1
            inj.append((x, y, s, sgn))
        ei = epV0.copy()
        for x, y, s, sgn in inj:
            ei += gaussian_stamp(ei.shape, x, y, sgn * s * sig0, ebeam, cd)
        rV = np.nan_to_num(_residual(ei, mV, flow, hdr, cd), nan=0.0)
        v_rms = float(sv2.get_uniform_rms(rV, hor))
        peaks = sv2.extract_v_peaks(rV.astype(np.float32), hdr, hor, at, bw, wcs, ot,
                                    sv2.CONFIG.get('CANDIDATE_SIGMA', 5.0), v_rms)
        st = gate_chain(peaks, rV, mI, hdr, beam, cd, wcs, ot, blank, ms_destr, ar_max)
        st = {'bdsf': peaks, **st}
        inj_c = [{'pix_x': x, 'pix_y': y, 'flux': s * sig0, 'status': 'PASS', 'snr': 0.0}
                 for x, y, s, sgn in inj]
        sv2.assign_local_snr(inj_c, rV, blank, beam, sigma_thresh=0.0,
                             inner_beams=2.0, outer_beams=10.0)
        pos = {s_: (np.array([(c['pix_x'], c['pix_y']) for c in st[s_]], float)
                    if st[s_] else np.empty((0, 2))) for s_ in STAGES}
        for (x, y, s, sgn), ic in zip(inj, inj_c):
            lr = float(ic.get('local_rms', np.nan))
            snr_local = s * sig0 / lr if np.isfinite(lr) and lr > 0 else np.nan
            row = dict(frame=fr, x=x, y=y, sign=('pos' if sgn > 0 else 'neg'),
                       snr_inner=round(s, 3),
                       snr_local=round(snr_local, 3) if np.isfinite(snr_local) else '')
            for s_ in STAGES:
                sep = (np.hypot(pos[s_][:, 0] - x, pos[s_][:, 1] - y).min()
                       if len(pos[s_]) else 1e9)
                row[s_] = int(sep <= 1.5 * beam)
            rows.append(row)
        print(f"[frame {fr}] inj={len(inj)} v_rms={v_rms*1e3:.0f}mJy peaks={len(peaks)} "
              f"bdsf={sum(r['bdsf'] for r in rows if r['frame']==fr)} "
              f"mstruct={sum(r['mstruct'] for r in rows if r['frame']==fr)}", flush=True)

    csv_path = f'{OUT}/V_c50_blind_{lst}_{date}_{band}.csv'
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['frame', 'x', 'y', 'sign', 'snr_inner',
                                           'snr_local'] + STAGES)
        w.writeheader(); [w.writerow(r) for r in rows]

    valid = [r for r in rows if r['snr_local'] != '']
    edges = np.arange(3.0, 12.01, 0.5); ctr = 0.5 * (edges[1:] + edges[:-1])

    def curve(subset, s_):
        s = np.array([r['snr_local'] for r in subset], float)
        rec = np.array([r[s_] for r in subset], float)
        return [((m := (s >= edges[i]) & (s < edges[i + 1])).sum(),
                 rec[m].mean() if m.sum() else np.nan) for i in range(len(edges) - 1)]

    def crossing(cur):
        f = np.array([c[1] for c in cur]); ok = np.isfinite(f); x = ctr[ok]; y = f[ok]
        for i in range(len(x) - 1):
            if y[i] < 0.5 <= y[i + 1]:
                return float(x[i] + (0.5 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
        return float(x[np.argmin(np.abs(y - 0.5))]) if len(x) else np.nan

    cur = {s_: curve(valid, s_) for s_ in STAGES}
    print(f"\n=== {lst} {date} {band} | N={len(valid)} blind injections "
          f"(rms_inner={sig0*1e3:.0f}mJy) ===")
    print(f"{'SNR_local':>9s} {'n':>4s} " + ' '.join(f'{s:>9s}' for s in STAGES))
    for i in range(len(ctr)):
        nn = cur['bdsf'][i][0]
        if nn == 0:
            continue
        print(f"{ctr[i]:9.2f} {nn:4d} " +
              ' '.join(f'{cur[s_][i][1]:9.2f}' if np.isfinite(cur[s_][i][1]) else f'{"--":>9s}'
                       for s_ in STAGES))
    print("\n50% crossing (local-rms SNR), raw interp:")
    for s_ in STAGES:
        print(f"  {s_:9s} C50 = {crossing(cur[s_]):.2f} sigma-local")
    final = crossing(cur['mstruct'])
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    phase3_c50.record(band, final, f'{lst}_{date}', len(valid), METHOD, stamp)
    print(f"\nRECORDED measured C50[{band}] = {final:.2f} sigma-local -> {phase3_c50.MEASURED_FILE}")
    print(f"wrote {csv_path}")


if __name__ == '__main__':
    main()
