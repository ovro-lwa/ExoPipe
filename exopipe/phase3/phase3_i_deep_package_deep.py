#!/usr/bin/env python3
"""
phase3_i_deep_package_deep.py — DEEP review packaging (Gregg 2026-07-06):
a rank directory for EVERY source with best SNR > 8, plus every source with
n_det >= 2 (any SNR). Image-wise cutout engine (each all-sky frame opened
once, all its candidates cut together, Pool-parallel). Also builds the
shared all-sky gallery: Run_I_DEEP/allsky/ with symlinks to every matched
residual hosting a packaged detection + every involved cell master
(one browsing point, no 35 GB duplication).
Updates review_dir in I_deep_FINAL_transients.csv.
Usage: python phase3_i_deep_package_deep.py
"""
import sys
import os, csv, glob, shutil
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
from exopipe.phase3 import assemble as live
from exopipe.phase3 import phase3_i_deep_final as F
from exopipe.common import phase3_products as prod

ROOT = live.ROOT
CSV = f'{ROOT}/I_deep_FINAL_transients.csv'
CANDDIR = f'{ROOT}/candidates'
ALLSKY = f'{ROOT}/allsky'
SNR_MIN, NDET_MIN = 8.0, 2
NPROC = int(os.environ.get('PKG_NPROC', '8'))


def cluster(finals):
    """Same clustering as phase3_i_deep_final (0.5 deg, SNR-greedy)."""
    cs = SkyCoord([float(r['ra']) for r in finals] * u.deg,
                  [float(r['dec']) for r in finals] * u.deg)
    order = sorted(range(len(finals)),
                   key=lambda i: -abs(float(finals[i]['snr'])))
    group = [-1] * len(finals)
    for i in order:
        if group[i] >= 0:
            continue
        group[i] = i
        for j in np.where(cs[i].separation(cs).deg < F.MERGE_DEG)[0]:
            if group[j] < 0:
                group[j] = int(i)
    return group


_MASTER_CACHE = {}


def _master(lst, band):
    key = (lst, band)
    if key not in _MASTER_CACHE:
        mp = f'{F.COADD}/{lst}_{band}/coadd_final/I_deep_Taper_R075_{band}.fits'
        a = fits.getdata(mp).squeeze().astype(np.float32)
        _MASTER_CACHE[key] = (a, fits.getheader(mp), mp)
    return _MASTER_CACHE[key]


def _cut_arr(data, w, cd, ra, dec, out, hdr_keys):
    n = int(F.CUT_DEG / cd)
    c = Cutout2D(data, SkyCoord(ra * u.deg, dec * u.deg), (n, n), wcs=w,
                 mode='partial', fill_value=np.nan)
    hh = c.wcs.to_header()
    for k, v in hdr_keys.items():
        hh[k] = v
    fits.writeto(out, c.data.astype(np.float32), hh, overwrite=True)


_CTX = None


def _image_worker(item):
    """One (lst, band, date) image: open resid+sub once, cut all its
    packaged candidates (resid / master / stage-B aligned epoch)."""
    (lst, band, date), tasks = item
    try:
        rp = F.resid_path(lst, band, date)
        sp = f'{F.COADD}/{lst}_{band}/subtracted/{date}_I_sub_{band}.fits'
        master, mh, mp = _master(lst, band)
        cd = abs(mh['CDELT1'])
        rdat = fits.getdata(rp).squeeze() if rp else None
        rhdr = fits.getheader(rp) if rp else None
        sdat = fits.getdata(sp).squeeze().astype(np.float32)
        shdr = fits.getheader(sp)
        aligned = sdat + master
        wm = WCS(mh).celestial
        wr = WCS(rhdr).celestial if rhdr is not None else wm
        ep = shdr['DATE-OBS']
        kk = {k: rhdr[k] for k in ('BMAJ', 'DATE-OBS', 'DATE-END', 'INTTIME',
                                   'PSFMATCH', 'RENORM', 'SCALE2B')
              if rhdr is not None and k in rhdr}
        for ra, dec, d, tgt in tasks:
            if rdat is not None:
                n = prod.build_name(band, 'I', 'deep', 'Taper', -0.75,
                                    proc=['dewarped', 'aligned', 'psfmatch',
                                          'msub', 'ssub'],
                                    lst=lst, epoch=ep, target=tgt, cutout=True)
                _cut_arr(rdat, wr, cd, ra, dec, f'{d}/{n}', kk)
            n = prod.build_name(band, 'I', 'deep', 'Taper', -0.75,
                                proc=['dewarped', 'aligned', 'coadd'],
                                lst=lst, epoch=ep, target=tgt, cutout=True)
            if not os.path.exists(f'{d}/{n}'):
                _cut_arr(master, wm, cd, ra, dec, f'{d}/{n}',
                         {'NSTACK': mh.get('NSTACK', '')})
            n = prod.build_name(band, 'I', 'deep', 'Taper', -0.75,
                                proc=['dewarped', 'aligned'],
                                lst=lst, epoch=ep, target=tgt, cutout=True)
            _cut_arr(aligned, wm, cd, ra, dec, f'{d}/{n}',
                     {'DATE-OBS': shdr.get('DATE-OBS', '')})
        return (lst, band, date), len(tasks), None
    except Exception as e:
        return (lst, band, date), 0, str(e)


def main():
    rows = live.load_rows()
    finals = [r for r in rows if r.get('final7') == '1']
    # apply the ratified offline gates identically to the assembler (size
    # bound + 2-D no-guard dynamic-gate keys) so directories match the CSV
    finals = [r for r in finals if not F._oversize(r)]
    kp = f'{ROOT}/dyn2d_vetoed_keys.csv'
    if os.path.exists(kp):
        vk = {(k['lst'], k['date'], k['band'], k['pix_x'], k['pix_y'])
              for k in csv.DictReader(open(kp))}
        finals = [r for r in finals if (r['lst'], r['date'], r['band'],
                                        r['pix_x'], r['pix_y']) not in vk]
    group = cluster(finals)
    from collections import defaultdict
    clusters = defaultdict(list)
    for i, g in enumerate(group):
        clusters[g].append(i)

    srcs = list(csv.DictReader(open(CSV)))
    # map published cand_id -> cluster via the best detection's identity
    key2rank = {}
    for s in srcs:
        key2rank[(s['lst'], s['date'], s['band'],
                  round(float(s['ra']), 5), round(float(s['dec']), 5))] = s
    sel = {}   # cluster gid -> source row
    for g, mem in clusters.items():
        best = max(mem, key=lambda i: abs(float(finals[i]['snr'])))
        r = finals[best]
        s = key2rank.get((r['lst'], r['date'], r['band'],
                          round(float(r['ra']), 5), round(float(r['dec']), 5)))
        if s is None:
            continue
        if (float(s['snr']) > SNR_MIN or int(s['n_det']) >= NDET_MIN
                or s.get('xmatch') or s.get('body')):   # association (Gregg
            sel[g] = s                                  # correction 2026-07-06)
    print(f'{len(sel)} sources selected (snr>{SNR_MIN} or n_det>={NDET_MIN} '
          f'or xmatch/body) of {len(srcs)}')

    # build dirs + CSV artifacts + image-wise cutout task list
    tasks = defaultdict(list)
    for g, s in sel.items():
        d = f"{CANDDIR}/{s['cand_id']}"
        os.makedirs(d, exist_ok=True)
        s['review_dir'] = d
        mem = [finals[i] for i in clusters[g]]
        with open(f'{d}/detections.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(mem[0].keys()))
            w.writeheader()
            for m in sorted(mem, key=lambda r: -abs(float(r['snr']))):
                w.writerow(m)
        with open(f"{d}/{s['cand_id'].split('_')[0]}.reg", 'w') as fh:
            fh.write(f'fk5\ncircle({float(s["ra"]):.5f},'
                     f'{float(s["dec"]):.5f},3600")\n')
        arows, seen = [], set()
        for m in sorted(mem, key=lambda r: -abs(float(r['snr']))):
            lst, band, date = m['lst'], m['band'], m['date']
            rp = F.resid_path(lst, band, date)
            mp = f'{F.COADD}/{lst}_{band}/coadd_final/I_deep_Taper_R075_{band}.fits'
            sp = f'{F.COADD}/{lst}_{band}/subtracted/{date}_I_sub_{band}.fits'
            arows.append([lst, date, band, m['snr'], rp, mp, sp])
            if (lst, band, date) not in seen:
                seen.add((lst, band, date))
                tasks[(lst, band, date)].append(
                    (float(s['ra']), float(s['dec']), d,
                     s['cand_id'].split('_')[0]))
        with open(f'{d}/allsky_images.csv', 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['lst', 'date', 'band', 'snr', 'allsky_residual_fits',
                        'master_fits', 'stageB_sub_fits'])
            w.writerows(arows)

    print(f'{len(tasks)} images to cut; running Pool({NPROC})')
    import multiprocessing as mp_
    nerr = 0
    with mp_.get_context('fork').Pool(NPROC) as pool:
        for key, n, err in pool.imap_unordered(_image_worker,
                                               sorted(tasks.items())):
            if err:
                nerr += 1
                print(f'  FAIL {key}: {err}', flush=True)
    print(f'cutouts done ({nerr} image failures)')

    # ---- all-sky gallery: symlinks, one browsing point ----
    if os.environ.get('PKG_NO_ALLSKY') == '1':
        print('allsky gallery: SKIPPED (PKG_NO_ALLSKY)')
        _sweep_stale(sel)
        _finalize_csv(sel, srcs)
        return
    os.makedirs(f'{ALLSKY}/residuals', exist_ok=True)
    os.makedirs(f'{ALLSKY}/masters', exist_ok=True)
    cells = set()
    nlink = 0
    for (lst, band, date) in tasks:
        rp = F.resid_path(lst, band, date)
        if rp:
            ln = f'{ALLSKY}/residuals/{os.path.basename(rp)}'
            if not os.path.lexists(ln):
                os.symlink(rp, ln); nlink += 1
        cells.add((lst, band))
    for lst, band in sorted(cells):
        mp = f'{F.COADD}/{lst}_{band}/coadd_final/I_deep_Taper_R075_{band}.fits'
        ln = f'{ALLSKY}/masters/{lst}_{band}_I_deep_Taper_R075.fits'
        if not os.path.lexists(ln):
            os.symlink(mp, ln)
    print(f'allsky gallery: {nlink} residual links + {len(cells)} master links '
          f'-> {ALLSKY}/')

    _sweep_stale(sel)
    _finalize_csv(sel, srcs)


def _sweep_stale(sel):
    """Rank namespaces shift on every list rebuild; sweep rank dirs that are
    no longer referenced into a dated stale dir (never delete — Gregg may
    have annotations). Found the hard way 2026-07-08: 1,053 stale dirs from
    three same-day rebuilds."""
    import datetime
    valid = {s['review_dir'].rstrip('/').split('/')[-1] for s in sel.values()}
    stale_root = (f'{ROOT}/candidates_stale_'
                  f'{datetime.date.today().strftime("%Y%m%d")}')
    moved = 0
    for d in sorted(os.listdir(CANDDIR)):
        if d.startswith('rank') and d not in valid:
            os.makedirs(stale_root, exist_ok=True)
            shutil.move(f'{CANDDIR}/{d}', f'{stale_root}/{d}')
            moved += 1
    if moved:
        print(f'swept {moved} stale rank dirs -> {stale_root}')


def _finalize_csv(sel, srcs):
    """Update the review_dir column in the final CSV."""
    sel_ids = {s['cand_id']: s['review_dir'] for s in sel.values()}
    for s in srcs:
        s['review_dir'] = sel_ids.get(s['cand_id'], s.get('review_dir', ''))
    with open(CSV + '.tmp', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=F.COLS, extrasaction='ignore')
        w.writeheader()
        for s in srcs:
            w.writerow(s)
    os.replace(CSV + '.tmp', CSV)
    print(f'review_dir updated in {CSV}')


if __name__ == '__main__':
    main()
