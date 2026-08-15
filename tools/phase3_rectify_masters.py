#!/usr/bin/env python3
"""
phase3_rectify_masters.py — RECTIFYING ARCHIVER for the deep master trees
(the 873 GB "unknown-fits" manifest class). Local masters keep their
legacy names (working set untouched); the ARCHIVE copies get conformant
grammar names + completed headers.

Legacy -> grammar mapping (per cell {lst}_{scope}/coadd_final/):
  {S}_deep_{Taper}_{R075|R0}[_clean][_pbcorr]_{scope}.fits
    -> {scope}_{S}_deep_{Taper}_Robust{+0.0|-0.75}[_vclean][_pbcorr]
       _dewarped_aligned_coadd_LST{lst}_{D0}-{D1}_N{nstack}.fits
  (D0/D1/N from the master's own DATE-OBS/DATE-END/NSTACK — the v2 time
  block is already correct post-repair; _clean maps to the vclean token)
  PB_deep_{scope}.fits -> {scope}_I_deep_Taper_Robust-0.75_pbmap_... (the
  PB map that pairs the cell; 'pbmap' token added to the grammar)
Destination: /lustre/pipeline/exopipe/phase3/coadd/Run_MASTERS1_{overall
range}/{lst}_{scope}/  (coadd sibling w/ Run dir per PHASE3_STRUCTURE).

Modes: --smoke {cell}  print the mapping table + write ONE cell; default
       (after Gregg sign-off) walks both trees. Ledger rows appended.
"""
import os, sys, csv, glob, re, shutil, argparse

_REAL_ARGV = list(sys.argv)           # capture BEFORE any clobbering
sys.path.insert(0, '/lustre/gh/main/clv15')
from astropy.io import fits
from phase3_products import build_name

ROOTS = [('/fast/gh/main/phase3/coadd', 'wb'),
         ('/data07/gh/phase3/coadd_sb', 'sb')]
DST = '/lustre/pipeline/exopipe/phase3/coadd/Run_MASTERS1_20241218-20250827'
LEDGER = '/data07/gh/phase3/logs/lustre_push_ledger.csv'
LEG = re.compile(r'^(I|V)_deep_(Taper|NoTaper)_(R075|R0)'
                 r'(_clean)?(_pbcorr)?_([A-Za-z0-9]+)\.fits$')
PBM = re.compile(r'^PB_deep_([A-Za-z0-9]+)\.fits$')


def map_one(cell, fname, hdr):
    lst, scope = cell.split('_', 1)
    d0 = hdr['DATE-OBS'][:10].replace('-', '')
    d1 = hdr['DATE-END'][:10].replace('-', '')
    n = int(hdr.get('NSTACK', 0) or 0)
    m = LEG.match(fname)
    if m:
        st, taper, rob, clean, pbc, sc = m.groups()
        assert sc == scope, (fname, cell)
        proc = ['dewarped', 'aligned', 'coadd']
        if clean:
            proc.append('vclean')
        if pbc:
            proc.append('pbcorr')
        return build_name(scope=scope, stokes=st, cadence='deep',
                          taper=taper, robust=(-0.75 if rob == 'R075' else 0.0),
                          proc=proc, lst=lst, date_start=d0, date_end=d1,
                          n_epochs=n)
    m = PBM.match(fname)
    if m:
        return build_name(scope=scope, stokes='I', cadence='deep',
                          taper='Taper', robust=-0.75,
                          proc=['pbmap'], lst=lst, date_start=d0,
                          date_end=d1, n_epochs=n)
    return None


def do_cell(root, cell, write=True):
    src = f'{root}/{cell}/coadd_final'
    rows = []
    for p in sorted(glob.glob(f'{src}/*.fits')):
        fn = os.path.basename(p)
        try:
            hdr = fits.getheader(p)
            new = map_one(cell, fn, hdr)
        except Exception as e:
            rows.append((fn, f'SKIP ({str(e)[:40]})'))
            continue
        if new is None:
            rows.append((fn, 'UNMAPPED — held for review'))
            continue
        rows.append((fn, new))
        if write:
            od = f'{DST}/{cell}'
            os.makedirs(od, exist_ok=True)
            dstf = f'{od}/{new}'
            if not (os.path.exists(dstf)
                    and os.path.getsize(dstf) == os.path.getsize(p)):
                shutil.copy2(p, dstf)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', default=None)
    a = ap.parse_args(_REAL_ARGV[1:])
    # parse-gate self-test: prove the flag actually reached argparse
    if '--smoke' in _REAL_ARGV:
        assert a.smoke, 'ARGPARSE GATE FAILED: --smoke given but not parsed'
    if a.smoke:
        root = ROOTS[0][0] if '_' in a.smoke and not a.smoke.endswith('MHz') \
            else ROOTS[1][0]
        rows = do_cell(root, a.smoke, write=True)
        print(f'--- mapping for {a.smoke} ---')
        for old, new in rows:
            print(f'  {old:42s} -> {new}')
        return
    ncell = nfile = 0
    for root, tag in ROOTS:
        for cd in sorted(os.listdir(root)):
            if not os.path.isdir(f'{root}/{cd}/coadd_final'):
                continue
            rows = do_cell(root, cd, write=True)
            ncell += 1
            nfile += sum(1 for _, n in rows if n and 'UNMAPPED' not in n
                         and 'SKIP' not in n)
            if ncell % 50 == 0:
                print(f'  {ncell} cells...', flush=True)
    with open(LEDGER, 'a') as f:
        f.write(f'{__import__("time").strftime("%Y-%m-%dT%H:%M:%S")},'
                f'masters-rectified,coadd+coadd_sb,{DST},{nfile},,'
                f'rectifying-copy,pushed-no-delete\n')
    print(f'RECTIFY-DONE {ncell} cells {nfile} files', flush=True)


if __name__ == '__main__':
    main()
