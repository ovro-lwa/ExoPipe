#!/usr/bin/env python3
"""EPOCH GATE (Gregg 2026-10-04, option B, named approval: "I think your proposed solution is entirely consistent with what I
state above. Dont force products for crappy nights. We want to find those nights and exclude them via approved QA.";
ledger R-EPOCH-CONTENT-GATE-REFUSES-NIGHT-FROM-COADD-AND-SEARCH).

A night whose EPOCH IMAGE fails its content test is refused from the co-add AND from both searches, and no per-epoch product
is published for it. The test is the archive's own (coadd_full_pipeline._epoch_content_arrays, one definition):
  identity  epoch - residual - master has a robust rms below 1e-3 of the master's (when an epoch file exists to compare);
  Stokes I  the median epoch/master at the master's bright compact sources lies in 0.8-1.25.
The night's verdict comes from Stokes I (the flux scale is the night's, the same for V; the V test can only check the
identity) and applies to the night in both Stokes. Where the generation holds no epoch file (a cell that was not rebuilt), the
epoch is the night's residual + the master, exactly as the co-add and the 2026-10-01 archive repair define it.
Why a gate: a night whose flux scale lies outside the co-add's correction window (first pass: no correction outside 0.3-3;
second pass: the cell median outside 0.6-1.6 x it) keeps an uncorrected epoch and residual, and the night-image gate cannot see
it (correlation is scale-free). Measured 2026-10-04: 24 night-cells fail, 20 of them searched in Run_P3S4.

Record: coadd_final/epoch_gate.csv, one row per night with an I residual: date, i_ratio, n_src, verdict (pass / FAIL),
reason, master_md5 (md5 of the I survey master's data the night was tested against - a record for another master is stale),
method. Readers: the co-add membership, the search admission, the archive publish, the install, the P3S5 patch; a reader
that needs the record and finds it missing or stale REFUSES (a gate never skips)."""
import csv
import glob
import hashlib
import os
import re

import numpy as np

GATE_FILE = 'epoch_gate.csv'
COLS = ['date', 'i_ratio', 'n_src', 'verdict', 'reason', 'master_md5', 'method']
_SUB_RE = re.compile(r'^(?P<date>\d{4}-\d{2}-\d{2})_I_sub(?:_shflux)?(?:_mdewarp)?_(?P<fl>[A-Za-z0-9]+)\.fits$')


def master_md5(master_path):
    from astropy.io import fits
    return hashlib.md5(np.ascontiguousarray(np.squeeze(fits.getdata(master_path)).astype(np.float32)).tobytes()).hexdigest()


def evaluate_cell(final_dir, sub_dir, fl):
    """{date: row} for every night with a Stokes I residual in sub_dir, tested against final_dir's I survey master."""
    from astropy.io import fits
    from exopipe.phase3.lib.masters import coadd as C
    mp = f'{final_dir}/I_deep_Taper_R075_{fl}.fits'
    m = np.squeeze(fits.getdata(mp)).astype(np.float64); mmd5 = master_md5(mp)
    rows = {}
    for p in sorted(glob.glob(f'{sub_dir}/*_I_sub_{fl}.fits')):
        g = _SUB_RE.match(os.path.basename(p))
        if not g:
            continue
        d = g['date']; r = np.squeeze(fits.getdata(p)).astype(np.float64)
        ep = f'{sub_dir}/{d}_I_epoch_{fl}.fits'
        if os.path.exists(ep):
            e = np.squeeze(fits.getdata(ep)).astype(np.float64); src = 'epoch file'
            ok, why, med, n = C._epoch_content_arrays(e, m, r, 'I')
        else:
            e = r + m; src = 'residual + master'
            ok, why, med, n = C._epoch_content_arrays(e, m, None, 'I')
        rows[d] = dict(i_ratio='' if med is None else f'{med:.4f}', n_src=n, verdict='pass' if ok else 'FAIL',
                       reason='' if ok else why, master_md5=mmd5, method=f'epoch content test ({src})')
    return rows


def write_record(final_dir, rows):
    p = f'{final_dir}/{GATE_FILE}'; tmp = p + '.part'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore'); w.writeheader()
        for d in sorted(rows):
            w.writerow(dict(rows[d], date=d))
    os.replace(tmp, p)
    return p


def read_record(final_dir):
    """{date: row}; FileNotFoundError when the cell has no record (callers that need it REFUSE)."""
    with open(f'{final_dir}/{GATE_FILE}') as f:
        return {r['date']: r for r in csv.DictReader(f)}


def failed_dates(final_dir, fl=None, check_master=True):
    """Dates the epoch gate refused. With check_master, REFUSES (RuntimeError) a record made against another I master."""
    rec = read_record(final_dir)
    if check_master and rec and fl:
        cur = master_md5(f'{final_dir}/I_deep_Taper_R075_{fl}.fits')
        stale = {r['master_md5'] for r in rec.values()} - {cur}
        if stale:
            raise RuntimeError(f'{final_dir}/{GATE_FILE} was made against another I master (md5 {sorted(stale)[0][:8]}, '
                               f'current {cur[:8]}) - stale record; REFUSING')
    return {d for d, r in rec.items() if r['verdict'] == 'FAIL'}


if __name__ == '__main__':   # evaluate + write the record of cells:  python -m exopipe.phase3.epoch_gate <root> <cell> [...]
    import sys
    root = sys.argv[1]
    for cell in sys.argv[2:]:
        fl = cell.split('_', 1)[1]; fd, sd = f'{root}/{cell}/coadd_final', f'{root}/{cell}/subtracted'
        try:
            rows = evaluate_cell(fd, sd, fl); write_record(fd, rows)
            bad = sorted(d for d, r in rows.items() if r['verdict'] == 'FAIL')
            print(f'EPOCH-GATE {cell}: {len(rows)} nights, {len(bad)} FAIL {bad}', flush=True)
        except Exception as e:
            print(f'EPOCH-GATE-ERROR {cell}: {e}', flush=True)
