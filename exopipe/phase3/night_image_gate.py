#!/usr/bin/env python3
"""NIGHT-IMAGE GATE (Gregg 2026-10-02: "Approved"; ledger R-NIGHT-IMAGE-GATE-BEFORE-BOTH-GATES).

One QA test per night-image, computed once, ahead of BOTH existing gates - the co-add membership QA and the search-admission
QA (phase3_deep: data_decisions drop, sanity gate, per-Stokes search tier, provenance) - which keep their own logic. A
night-image FAILS when
  * its correlation with the reference (the master the night is aligned to) at the best shift within MAX_SHIFT_DEG is below
    THRESHOLD, or
  * any |value| of the night image exceeds BLOWUP_JY (numerically blown up).
The correlation: both images BLOCK x BLOCK block-averaged, central RADIUS of the half-width (elevation above ~46 deg for
RADIUS 0.7, so the horizon is outside), each normalised to zero median and unit robust noise (clipped at +-50), then to unit
norm: independent of the flux scale BY CONSTRUCTION (a calibrator-scintillation scale error cannot fail a good night;
verified 2026-10-02, a night x0.25 / x1 / x4 gives the same value to six decimals). The shift search absorbs residual position
offsets (measured 0.31-0.88 deg on raw, dewarp-failed nights; nothing matches only beyond 3 deg).
Measured basis (ledger 2026-10-02, 8,827 night-cells of Run_20260909_000000): good in-master nights 0.88-0.99, population
median 0.98, regridded pre-cutover nights in masters median 0.989 with none below 0.5; 161 night-cells fail on 32 nights.
Record: coadd_final/night_image_gate.csv, one row per night (date, corr_zero, corr_best, best_shift_deg, maxabs_jy, verdict,
reason, method). The search refuses a cell whose record is missing (a gate refuses, it never skips)."""
import csv
import os

import numpy as np

THRESHOLD = 0.5
MAX_SHIFT_DEG = 3.0
BLOWUP_JY = 1e6
BLOCK = 4
RADIUS = 0.7
GATE_FILE = 'night_image_gate.csv'
COLS = ['date', 'corr_zero', 'corr_best', 'best_shift_deg', 'maxabs_jy', 'verdict', 'reason', 'method']


def _prep(a):
    a = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    H, W = a.shape; H4, W4 = H // BLOCK * BLOCK, W // BLOCK * BLOCK
    a = a[:H4, :W4].reshape(H4 // BLOCK, BLOCK, W4 // BLOCK, BLOCK).mean(axis=(1, 3))
    h, w = a.shape; yy, xx = np.mgrid[:h, :w]
    m = np.hypot(xx - w / 2, yy - h / 2) < RADIUS * (w / 2)
    a = np.where(m, a, 0.0); v = a[m]; med = np.median(v)
    s = 1.4826 * np.median(np.abs(v - med)) or 1.0
    a = np.where(m, np.clip((a - med) / s, -50, 50), 0.0)
    return a / (np.sqrt((a ** 2).sum()) or 1.0)


def correlation(night, ref, cdelt_deg):
    """(correlation at zero shift, best correlation within MAX_SHIFT_DEG, that shift in degrees). Same grid required."""
    night = np.squeeze(night); ref = np.squeeze(ref)
    if night.shape != ref.shape:
        raise ValueError(f'night {night.shape} and reference {ref.shape} are on different grids')
    a, b = _prep(night), _prep(ref)
    c = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b)))))
    h, w = c.shape; yy, xx = np.mgrid[:h, :w]
    off = np.hypot(yy - h // 2, xx - w // 2) * BLOCK * abs(float(cdelt_deg))
    cm = np.where(off <= MAX_SHIFT_DEG, c, -np.inf); iy, ix = np.unravel_index(int(np.argmax(cm)), c.shape)
    return float(c[h // 2, w // 2]), float(cm[iy, ix]), float(off[iy, ix])


def evaluate(night_aligned, ref, cdelt_deg, night_raw=None):
    """The gate verdict for one night-image. night_raw (the input before alignment) is used for the blown-up test when given."""
    raw = np.squeeze(night_raw if night_raw is not None else night_aligned)
    fin = np.isfinite(raw)
    maxabs = float(np.max(np.abs(raw[fin]))) if fin.any() else float('inf')
    z, best, sh = correlation(night_aligned, ref, cdelt_deg)
    reasons = []
    if not (maxabs < BLOWUP_JY):
        reasons.append(f'values up to {maxabs:.3g} Jy/beam (> {BLOWUP_JY:.0e})')
    if not (best >= THRESHOLD):
        reasons.append(f'correlation with the reference {best:.3f} < {THRESHOLD} (best shift within {MAX_SHIFT_DEG:g} deg)')
    return dict(corr_zero=f'{z:.4f}', corr_best=f'{best:.4f}', best_shift_deg=f'{sh:.2f}', maxabs_jy=f'{maxabs:.3g}',
                verdict='FAIL' if reasons else 'pass', reason='; '.join(reasons))


def write_record(final_dir, rows, method):
    """rows: {date: evaluate() dict or a dict with verdict/reason}. Written whole (atomic), never appended."""
    p = f'{final_dir}/{GATE_FILE}'; tmp = p + '.part'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore'); w.writeheader()
        for d in sorted(rows):
            w.writerow(dict(rows[d], date=d, method=method))
    os.replace(tmp, p)
    return p


def read_record(final_dir):
    """{date: row}; FileNotFoundError when the cell has no record (callers REFUSE)."""
    with open(f'{final_dir}/{GATE_FILE}') as f:
        return {r['date']: r for r in csv.DictReader(f)}


def failed_dates(final_dir):
    return {d for d, r in read_record(final_dir).items() if r['verdict'] == 'FAIL'}
