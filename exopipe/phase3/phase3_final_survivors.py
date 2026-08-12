#!/usr/bin/env python3
"""
phase3_final_survivors.py — STAGE 1 of the Run_FINAL build (offline, no reconstruction).

Applies the agreed veto sequence (PHASE3_HANDOFF.md §5.9; logic identical to
phase3_gate_analysis.sequence) to Run_GATES/gate_ledger.csv and writes the survivor
list with every ledger-derived column. Photometry (peak/int/sign/elevation/per-band)
is filled by STAGE 2 (phase3_final_reconstruct.py, one reconstruction pass) and the
final assembly (phase3_final_csv.py).

Sequence: mstruct -> tag(Low<=2/Int3-10/High>=11 by post-mstruct >=7s) -> I/V 20x rb0
          -> bilobe (Int/High only) -> gaussfit -> A-team 2deg -> Jupiter (Red/Full 5deg,
          Green/Blue 0.5deg)
          -> CONDITIONAL I/V 10x on BAD images (Gregg 2026-07-02, "Definition B"):
             an image whose FINAL >=7sigma count exceeds BAD_N7 gets a second, tighter
             co-located I/V pass (rb=0, >BAD_IV_THRESH rejected), single-pass (the bad
             tag is computed once, from the pre-gate final list). Measured: Red storm
             image dC50 +0.16sig / 6% faint loss; Green 0.00 / 0%; all 11 confirmed-
             transient rows have iv0<=2.9 -> untouched (Run_FINAL/ivloss/).

cand_id = rank{NNNN}_{lst}_{date}_{band}, rank by descending |local SNR| over ALL
survivors (Gregg 2026-07-02: rank# replaces the VDF prefix). Because rank follows
SNR, ranks 1..N7 are exactly the >=7sigma cutout set.

Repeats (union-find, 10 arcmin, production REPEAT_MATCH_ARCMIN) are grouped HERE so
the >=7sigma index can list every epoch of a candidate's repeat group — stage 2 then
writes all-sky frames for those epochs too, and stage 3 writes the per-candidate
allsky_images.txt pointer files.

Outputs (under /data07/gh/phase3/transients/Run_FINAL/):
  survivors_pre_phot.csv   all >=5sigma survivors + veto columns + repeat columns
  survivors_7s_index.csv   >=7sigma survivors + their repeat-group epoch list
  allsky_needed.csv        (lst,date,band) epochs requiring all-sky FITS
Usage: python phase3_final_survivors.py
"""
import sys
import os, csv
from collections import defaultdict
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from exopipe.phase3.phase3_gate_analysis import load, sequence, ATEAM, _jup

OUT = '/data07/gh/phase3/transients/Run_FINAL'
ATEAM_NAMES = ['CygA', 'CasA', 'TauA', 'VirA']
BAD_N7 = 10            # image is BAD if its final >=7sigma count exceeds this
BAD_IV_THRESH = 10.0   # tighter co-located I/V (rb=0) applied on BAD images
REPEAT_MATCH_ARCMIN = 10.0


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    byimg = defaultdict(list)
    for r in rows:
        byimg[(r['lst'], r['date'], r['band'])].append(r)

    out = []
    nbad, ncut = 0, 0
    for k in sorted(byimg):
        lst, date, band = k
        tag, st = sequence(byimg[k])
        fin = sorted(st['jupiter'], key=lambda c: -c['snr'])
        # conditional I/V 10x on bad images (Definition B, Gregg 2026-07-02)
        if sum(1 for c in fin if c['snr'] >= 7) > BAD_N7:
            nbad += 1
            keep = [c for c in fin if c['iv0'] <= BAD_IV_THRESH]
            ncut += len(fin) - len(keep)
            fin = keep
        nve = len(fin)                    # n_v_epoch = final >=5s survivors in this image
        for c in fin:
            co = SkyCoord(float(c['ra']) * u.deg, float(c['dec']) * u.deg)
            dat = co.separation(ATEAM).deg
            jp = _jup(date)
            jdeg = round(float(co.separation(jp).deg), 2) if jp is not None else ''
            out.append(dict(lst=lst, date=date, band=band,
                            ra=c['ra'], dec=c['dec'], snr=c['snr'],
                            mstruct=c['mstruct'], iv0=c['iv0'], bilobe=c['bilobe'],
                            gaussfit=c['gaussfit'], fp_class=tag, n_v_epoch=nve,
                            nearest_ateam=f'{ATEAM_NAMES[int(np.argmin(dat))]}:{dat.min():.2f}',
                            jupiter_deg=jdeg))

    # ---- rank by descending SNR (cand_id = rank####_lst_date_band) ----
    out.sort(key=lambda s: -s['snr'])
    for i, s in enumerate(out, 1):
        s['rank'] = i
        s['cand_id'] = f'rank{i:04d}_{s["lst"]}_{s["date"]}_{s["band"]}'

    # ---- repeats: union-find over survivor positions (10 arcmin) ----
    co = SkyCoord(np.array([float(s['ra']) for s in out]) * u.deg,
                  np.array([float(s['dec']) for s in out]) * u.deg)
    parent = list(range(len(out)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    i1, i2, _, _ = co.search_around_sky(co, REPEAT_MATCH_ARCMIN * u.arcmin)
    for a, b in zip(i1, i2):
        if a < b:
            ra_, rb_ = find(int(a)), find(int(b))
            if ra_ != rb_:
                parent[max(ra_, rb_)] = min(ra_, rb_)
    groups = defaultdict(list)
    for i in range(len(out)):
        groups[find(i)].append(i)
    memb_epochs = {}
    for root, members in groups.items():
        members.sort()                       # lowest index = best (rank order)
        gid = out[members[0]]['cand_id']
        eps = sorted({(out[i]['lst'], out[i]['date'], out[i]['band']) for i in members})
        lstdates = sorted({f"{out[i]['lst']}:{out[i]['date']}" for i in members})
        for i in members:
            out[i]['n_det'] = len(members)
            out[i]['repeat_group'] = gid
            out[i]['repeat_lst_dates'] = '|'.join(lstdates)
            memb_epochs[i] = eps

    cols = ['cand_id', 'rank', 'lst', 'date', 'band', 'ra', 'dec', 'snr',
            'mstruct', 'iv0', 'bilobe', 'gaussfit', 'fp_class', 'n_v_epoch',
            'nearest_ateam', 'jupiter_deg', 'n_det', 'repeat_group',
            'repeat_lst_dates']
    with open(f'{OUT}/survivors_pre_phot.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(out)

    # ---- >=7sigma index + all-sky needs (own epoch + repeat-group epochs) ----
    idx7, needed = [], set()
    for i, s in enumerate(out):
        if s['snr'] < 7:
            continue
        eps = memb_epochs[i]
        needed.update(eps)
        idx7.append(dict(cand_id=s['cand_id'], lst=s['lst'], date=s['date'],
                         band=s['band'], ra=s['ra'], dec=s['dec'], snr=s['snr'],
                         group_epochs='|'.join(f'{l}:{d}:{b}' for l, d, b in eps)))
    with open(f'{OUT}/survivors_7s_index.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(idx7[0].keys()))
        w.writeheader(); w.writerows(idx7)
    with open(f'{OUT}/allsky_needed.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['lst', 'date', 'band'])
        w.writerows(sorted(needed))

    ep7 = len({(r['lst'], r['date'], r['band']) for r in idx7})
    print(f'{len(out)} survivors >=5s ({len(idx7)} >=7s over {ep7} epoch-images)')
    print(f'conditional I/V {BAD_IV_THRESH:.0f}x fired on {nbad} bad images '
          f'(>{BAD_N7} final >=7s), removed {ncut} rows')
    print(f'all-sky needed for {len(needed)} epoch-images (incl. repeat-group epochs)')


if __name__ == '__main__':
    main()
