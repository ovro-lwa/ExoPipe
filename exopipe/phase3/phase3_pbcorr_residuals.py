#!/usr/bin/env python3
"""Materialize time-integrated PB-corrected siblings of EVERY Phase-3
per-epoch residual image, written to the Lustre archive (Gregg directive
2026-07-27: pbcorr version of every residual in the Phase-3 directories;
snapshots excluded).

Groups (src -> dst sibling with '_pbcorr' after the Robust token):
  v_resid   /lustre/.../phase3/v_resid/{cell}/*.fits          (in place)
  i_resid   /data07/gh/phase3/i_resid/{cell}/*.fits           -> Lustre i_resid
  sb-deep   /data07/.../P3_PROD/{sb}/{S}/deep/*msub*.fits     -> Lustre same rel
  wb-10min  /data07/.../P3_PROD10/{band}/{S}/10min/*msub*.fits-> Lustre same rel
  sb-10min  /data07/.../P3_SB10/{sb}/{S}/10min/*msub*.fits    -> Lustre same rel

PB map = cell coadd_final/PB_deep_{scope}.fits (time-integrated, canonical),
floor 0.1, provenance via pb_apply.apply_pbcorr. Idempotent: existing dst
skipped. One worker per cell (PB map read once). Progress + error log to
/data07/gh/phase3/logs/pbcorr_residuals_20260727.log; summary ledger row.
Usage: phase3_pbcorr_residuals.py [nproc]
"""
import os, re, sys, glob, csv, datetime
from collections import defaultdict
from multiprocessing import Pool

_argv = list(sys.argv); sys.argv = ['pbres']
import numpy as np
from astropy.io import fits
from exopipe.phase3 import pb_apply as PB

D7 = '/data07/gh/phase3'
LU = '/lustre/pipeline/exopipe/phase3'
WB_PB = '/fast/gh/main/phase3/coadd/{cell}/coadd_final/PB_deep_{scope}.fits'
SB_PB = D7 + '/coadd_sb/{cell}/coadd_final/PB_deep_{scope}.fits'
LOG = D7 + '/logs/pbcorr_residuals_20260727.log'
WIDE = {'Red', 'Green', 'Blue', 'Full'}
NPROC = int(_argv[1]) if len(_argv) > 1 else 14
FILTER = _argv[2] if len(_argv) > 2 else None  # substring on dst path


def pb_path(cell):
    lst, scope = cell.split('_', 1)
    return (WB_PB if scope in WIDE else SB_PB).format(cell=cell, scope=scope)


def pbcorr_name(name):
    out = re.sub(r'(Robust[+-][\d.]+)', r'\1_pbcorr', name, count=1)
    return None if out == name or 'pbcorr' in name else out


def build_tasks():
    """Return {cell: [(src, dst), ...]}."""
    by_cell = defaultdict(list)

    def add(cell, src, dst_dir):
        new = pbcorr_name(os.path.basename(src))
        if not new:
            return
        dst = os.path.join(dst_dir, new)
        if FILTER and FILTER not in dst:
            return
        if not os.path.exists(dst):
            by_cell[cell].append((src, dst))

    for cd in sorted(glob.glob(f'{LU}/v_resid/*_*')):
        cell = os.path.basename(cd)
        for f in glob.glob(f'{cd}/*_LST*.fits'):
            add(cell, f, cd)
    for cd in sorted(glob.glob(f'{D7}/i_resid/*_*')):
        cell = os.path.basename(cd)
        for f in glob.glob(f'{cd}/*_LST*.fits'):
            add(cell, f, f'{LU}/i_resid/{cell}')
    pats = [('P3_PROD', 'deep'), ('P3_PROD10', '10min'), ('P3_SB10', '10min')]
    for label, cad in pats:
        for f in glob.glob(f'{D7}/[0-2][0-9]h/*/{label}/*/[IV]/{cad}/*msub*.fits'):
            parts = f.split('/')
            lst, scope = parts[4], parts[7]
            add(f'{lst}_{scope}', f, os.path.dirname(f).replace(D7, LU))
    return by_cell


def do_cell(item):
    cell, pairs = item
    pp = pb_path(cell)
    done, errs = 0, []
    if not os.path.exists(pp):
        return cell, 0, [f'NO-PB-MAP {pp} ({len(pairs)} files)']
    pb = np.squeeze(fits.getdata(pp)).astype(np.float32)
    for src, dst in pairs:
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            PB.apply_pbcorr(src, pb, dst)
            done += 1
        except Exception as e:
            errs.append(f'{os.path.basename(src)}: {str(e)[:70]}')
    return cell, done, errs


def main():
    by_cell = build_tasks()
    total = sum(len(v) for v in by_cell.values())
    print(f'{len(by_cell)} cells, {total} pbcorr siblings to create', flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    n_done, n_err = 0, 0
    with open(LOG, 'a') as lg, Pool(NPROC) as p:
        lg.write(f'=== start {datetime.datetime.utcnow()} total={total}\n')
        items = sorted(by_cell.items(), key=lambda kv: -len(kv[1]))
        for cell, done, errs in p.imap_unordered(do_cell, items):
            n_done += done; n_err += len(errs)
            lg.write(f'{cell}: {done} ok, {len(errs)} err\n')
            for e in errs:
                lg.write(f'  ERR {cell} {e}\n')
            lg.flush()
            print(f'{cell}: {done} ok ({n_done}/{total})', flush=True)
    row = (f'{datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")},'
           f'pbcorr-residuals,{D7}+v_resid,{LU},{n_done},,'
           f'time-integrated cell PB floor0.1 errs={n_err},new-products')
    with open(f'{D7}/logs/lustre_push_ledger.csv', 'a') as f:
        f.write(row + '\n')
    print(f'DONE {n_done} created, {n_err} errors — ledgered', flush=True)


if __name__ == '__main__':
    main()
