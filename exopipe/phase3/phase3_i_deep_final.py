#!/usr/bin/env python3
"""
phase3_i_deep_final.py — DEFINITIVE Run_I_DEEP deliverables (Gregg spec
2026-07-06, mirroring the Stokes-V Run_FINAL packaging):

1. I_deep_FINAL_transients.csv — ONE RANK PER SOURCE (repeat detections
   merged by position, 0.5 deg): the row carries the source's BEST detection
   (highest |local SNR|, any band) + n_det (number of detections in the
   group) + repeat_lst_dates + best per-band fluxes across the group.
2. candidates/rankNNNN_<lst>_<date>_<band>/ review directories for the
   band-balanced top set (TOP_N best-ranked sources per Red, Green, Blue,
   union-merged; Full detections ride along), each containing:
     - detections.csv        every ledger row of the source (all repeats)
     - allsky_images.csv     full paths of ALL corresponding all-sky frames
                             (matched residual + master per detection)
     - cutout FITS           residual + master + stage-B aligned epoch,
                             per detection (3 deg box, standard names)
     - rankNNNN.reg          1-degree circle at the merged position
Usage: python phase3_i_deep_final.py [TOP_N]   (default 20)
"""
import sys
_ARGV = list(sys.argv)
import os, csv, glob, shutil
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord, AltAz
import astropy.units as u
from exopipe.phase3 import assemble as live
from exopipe.phase3 import phase3_final_csv as fc
from exopipe.common import phase3_products as prod

ROOT = live.ROOT
OUTCSV = f'{ROOT}/I_deep_FINAL_transients.csv'
CANDDIR = f'{ROOT}/candidates'
COADD = live.COADD
RESID = '/data07/gh/phase3/i_resid'
TOP_N = int(_ARGV[1]) if len(_ARGV) > 1 else 20
CUT_DEG = 5.0   # Gregg standard: 5x5 deg cutouts at all times (2026-07-11)
MERGE_DEG = live.REPEAT_DEG

COLS = ['cand_id', 'source_id', 'lst', 'date', 'band', 'ra', 'dec',
        'ra_hms', 'dec_dms',
        'gal_lat_deg', 'ecl_lat_deg', 'snr', 'peak_jy', 'int_jy', 'sign',
        'elevation_deg', 'gfit_fwhm_beams', 'gfit_ar', 'gfit_resid',
        'peak_jy_Red', 'peak_jy_Green', 'peak_jy_Blue', 'peak_jy_Full',
        'mstruct', 'assoc_m05_jy', 'bilobe', 'gaussfit', 'fp_class',
        'img_rms_mjy', 'n_i_image', 'nearest_ateam', 'jupiter_deg',
        'n_det', 'n_epochs', 'repeat_lst_dates',
        'c50_mjy', 'body', 'xmatch', 'xmatch_sep_arcmin', 'xmatch_sample',
        'xmatch_dist_pc', 'pos_flag', 'fc_star', 'img_floor', 'dc2',
        'peak_jy_meas', 'review_dir']


def resid_path(lst, band, date):
    g = glob.glob(f'{RESID}/{lst}_{band}/{band}_I_deep_Taper_Robust-0.75_'
                  f'*psfmatch_msub_ssub_LST{lst}_{date.replace("-","")}T*.fits')
    return g[0] if g else ''


def cutout(src_fits, ra, dec, out_path, size_deg=CUT_DEG):
    data = fits.getdata(src_fits).squeeze()
    hdr = fits.getheader(src_fits)
    w = WCS(hdr).celestial
    cd = abs(hdr.get('CDELT1') or hdr.get('CD1_1') or 0.05)
    n = int(size_deg / cd)
    co = SkyCoord(ra * u.deg, dec * u.deg)
    c = Cutout2D(data, co, (n, n), wcs=w, mode='partial', fill_value=np.nan)
    hh = c.wcs.to_header()
    for k in ('BMAJ', 'BMIN', 'BPA', 'BUNIT', 'DATE-OBS', 'DATE-END',
              'INTTIME', 'DURATION', 'PSFMATCH', 'RENORM', 'SCALE2B',
              'NSTACK', 'TIMEFIX'):
        if k in hdr:
            hh[k] = hdr[k]
    fits.writeto(out_path, c.data.astype(np.float32), hh, overwrite=True)


def _image_rms():
    """(lst,date,band) -> central residual rms (mJy) from the per-image CSVs
    (the I quality convention — Gregg 2026-07-06: a primary quality metric)."""
    rms = {}
    for p in ([f'{ROOT}/i_deep_images.csv', f'{ROOT}/i_deep_images_v1.csv']
              + sorted(glob.glob(f'{ROOT}/nodes/*/i_deep_images.csv'))):
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            rms.setdefault((r['lst'], r['date'], r['band']), r['rms_mjy'])
    return rms


SIZE_BOUND = {'Blue': 1.26, 'Green': 1.32, 'Red': 1.66, 'Full': 2.11}
XR_BASE = {'Blue': 3.0, 'Green': 3.5, 'Red': 5.0, 'Full': 3.5}  # arcmin, doc §3b
XR_BEAM_AMIN = {'Blue': 9.2, 'Green': 12.9, 'Red': 20.3, 'Full': 14.9}
# ECM-safe size bound (ratified Gregg 2026-07-07; physics-derived, x1.2
# empirical-beam-fit margin). Applied here OFFLINE to pre-gate ledger rows;
# the production driver now gates identically at search time.


def _oversize(r):
    return bool(r['gf_fwhm_beams']) and \
        float(r['gf_fwhm_beams']) > SIZE_BOUND[r['band']]


def main():
    rows = live.load_rows()
    finals_all = [r for r in rows if r.get('final7') == '1']
    finals = [r for r in finals_all if not _oversize(r)]
    # 2-D no-guard dynamic-gate re-filter for PRE-v2 ledger rows (ratified
    # Gregg 2026-07-07; keys from phase3_dyn2d_eval.py; post-v2 production
    # rows already carry the gate in final7)
    kp = f'{ROOT}/dyn2d_vetoed_keys.csv'
    if os.path.exists(kp):
        vk = {(k['lst'], k['date'], k['band'], k['pix_x'], k['pix_y'])
              for k in csv.DictReader(open(kp))}
        n0 = len(finals)
        finals = [r for r in finals if (r['lst'], r['date'], r['band'],
                                        r['pix_x'], r['pix_y']) not in vk]
        print(f'2-D dynamic gate re-filter: {n0} -> {len(finals)} detections')
    # IMFIT QUARANTINE (RATIFIED Gregg 2026-07-09): low-variability /
    # subtraction-residual class — on a fitted master counterpart >= 10x band
    # confusion, fitted |v_frac| < 0.25, AND residual amplitude < 2.5x the
    # misalignment (dipole) prediction (this arm protects real transients:
    # B0950 0/47, J0410 0/45 quarantined). Quarantined rows -> side list.
    tp = f'{ROOT}/imfit_triage.csv'
    if os.path.exists(tp):
        CONF = {'Red': 0.497, 'Green': 0.183, 'Blue': 0.092, 'Full': 0.183}
        qk, qrows = set(), []
        for m in csv.DictReader(open(tp)):
            band = m['band']
            if band not in CONF:
                continue
            try:
                Sm = float(m['S_master_jy']) if m['S_master_jy'] else np.nan
                v = float(m['v_frac']) if m['v_frac'] else np.nan
                Sr = float(m['S_resid_jy']) if m['S_resid_jy'] else np.nan
                dp = float(m['dipole_pred_jy']) if m['dipole_pred_jy'] else 0.0
            except ValueError:
                continue
            if not (np.isfinite(Sm) and Sm >= 10 * CONF[band]
                    and m['fit_ok_e'] == '1' and m['fit_ok_m'] == '1'):
                continue
            if not (np.isfinite(v) and abs(v) < 0.25):
                continue
            dpf = max(dp, 0.86 * Sm * 0.05)
            if np.isfinite(Sr) and abs(Sr) < 2.5 * dpf:
                qk.add((m['lst'], m['date'], m['band'], m['pix_x'], m['pix_y']))
                qrows.append(m)
        n0 = len(finals)
        finals = [r for r in finals if (r['lst'], r['date'], r['band'],
                                        r['pix_x'], r['pix_y']) not in qk]
        with open(f'{ROOT}/low_variability_residuals.csv', 'w',
                  newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(qrows[0].keys())
                               if qrows else ['none'])
            w.writeheader()
            w.writerows(qrows)
        print(f'imfit quarantine: {n0} -> {len(finals)} detections '
              f'({len(qrows)} -> low_variability_residuals.csv)')
    # side list: sources whose EVERY final detection fails the size bound
    # (consistently-extended variables — gated from the point-source list but
    # NOT buried; Gregg's "varying resolved source" class)
    lostd = [r for r in finals_all if _oversize(r)]
    if lostd:
        from exopipe.phase3 import phase3_i_deep_package_deep as _P  # cluster() reuse (no cycle:
        # package_deep imports this module lazily inside functions only)
        grp = _P.cluster(finals_all)
        from collections import defaultdict as _dd
        cl = _dd(list)
        for i, g in enumerate(grp):
            cl[g].append(i)
        rv = []
        for g, mem in cl.items():
            if all(_oversize(finals_all[i]) for i in mem):
                b = max(mem, key=lambda i: abs(float(finals_all[i]['snr'])))
                r = finals_all[b]
                rv.append([r['lst'], r['date'], r['band'], r['ra'], r['dec'],
                           r['snr'], r['gf_fwhm_beams'], len(mem)])
        with open(f'{ROOT}/resolved_variables.csv', 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['lst', 'date', 'band', 'ra', 'dec', 'best_snr',
                        'gf_fwhm_beams', 'n_det'])
            w.writerows(sorted(rv, key=lambda x: -float(x[5])))
        print(f'{len(rv)} consistently-extended sources -> '
              f'{ROOT}/resolved_variables.csv')
    img_rms = _image_rms()
    print(f'{len(finals)} final detections from '
          f'{len({(r["lst"], r["date"], r["band"]) for r in rows})} images')
    from collections import Counter, defaultdict
    per_img = Counter((r['lst'], r['date'], r['band']) for r in finals)

    cs = SkyCoord([float(r['ra']) for r in finals] * u.deg,
                  [float(r['dec']) for r in finals] * u.deg)
    order = sorted(range(len(finals)), key=lambda i: -abs(float(finals[i]['snr'])))
    group = [-1] * len(finals)
    for i in order:
        if group[i] >= 0:
            continue
        group[i] = i
        for j in np.where(cs[i].separation(cs).deg < MERGE_DEG)[0]:
            if group[j] < 0:
                group[j] = int(i)
    clusters = defaultdict(list)
    for i, g in enumerate(group):
        clusters[g].append(i)

    xcat, xnames, xsamples, xdists = fc.load_xcat()
    sources = []
    for g, mem in clusters.items():
        best = max(mem, key=lambda i: abs(float(finals[i]['snr'])))
        r = finals[best]
        dc2 = float(r['dc2']) if r['dc2'] else 1.0
        pk = float(r['peak_jy'])
        eps = sorted({f"{finals[i]['lst']}:{finals[i]['date']}" for i in mem})
        s = dict(lst=r['lst'], date=r['date'], band=r['band'],
                 ra=float(r['ra']), dec=float(r['dec']),
                 snr=abs(float(r['snr'])), sign=r['sign'],
                 peak_jy=round(pk * dc2, 4), peak_jy_meas=round(pk, 4),
                 int_jy='', mstruct=r['mstruct_i'],
                 assoc_m05_jy=r['assoc_m05_jy'], bilobe=r['bilobe'],
                 gaussfit=r['gaussfit'], gfit_fwhm_beams=r['gf_fwhm_beams'],
                 gfit_ar=r['gf_ar'], gfit_resid=r['gf_resid'],
                 jupiter_deg=r['jupiter_deg'], fc_star=r['fc_star'],
                 img_floor=r['floor'], dc2=r['dc2'], c50_mjy='',
                 n_det=len(mem), n_epochs=len(eps),
                 repeat_lst_dates='|'.join(eps),
                 n_i_image=per_img[(r['lst'], r['date'], r['band'])],
                 img_rms_mjy=img_rms.get((r['lst'], r['date'], r['band']), ''),
                 members=mem)
        lo, hi = live.FP_CLASS[r['band']]
        s['fp_class'] = ('Low' if s['n_i_image'] <= lo
                         else ('High' if s['n_i_image'] > hi else 'Int'))
        # best per-band flux across the whole group (dc2-corrected)
        bb = {}
        for i in mem:
            ri = finals[i]
            d2 = float(ri['dc2']) if ri['dc2'] else 1.0
            v = abs(float(ri['peak_jy'])) * d2
            b = ri['band']
            if b not in bb or v > bb[b]:
                bb[b] = v
        for b in ('Red', 'Green', 'Blue', 'Full'):
            s[f'peak_jy_{b}'] = round(bb[b], 4) if b in bb else ''
        c = SkyCoord(s['ra'] * u.deg, s['dec'] * u.deg)
        s['ra_hms'] = c.ra.to_string(unit=u.hourangle, sep=':', precision=1,
                                     pad=True)
        s['dec_dms'] = c.dec.to_string(sep=':', precision=0, alwayssign=True,
                                       pad=True)
        s['gal_lat_deg'] = round(float(c.galactic.b.deg), 2)
        s['ecl_lat_deg'] = round(float(c.barycentricmeanecliptic.lat.deg), 2)
        seps = [(nm, float(c.separation(co).deg)) for nm, co in live.ATEAM]
        nm, dd = min(seps, key=lambda t: t[1])
        s['nearest_ateam'] = f'{nm.replace(" ", "")}:{dd:.2f}'
        t, utc = live.epoch_mid(s['lst'], s['band'], s['date'])
        s['elevation_deg'] = (round(float(c.transform_to(
            AltAz(obstime=t, location=fc.OVRO)).alt.deg), 2)
            if t is not None else '')
        s['body'] = fc.body_label(s['ra'], s['dec'], utc)
        # Cross-match radius per COADD_ASTROMETRIC_ERROR.md (measured vs
        # VLSSr 2026-07-07/08): band base (~90th pct compact-source error)
        # + sigma_stat in quadrature, x1.5 below elevation 40 deg. 13h (all
        # bands, ~25' stale-flow warp, rebuild pending) and 11h Green (~16')
        # positions UNRELIABLE: pos_flag set, xmatch suppressed.
        _13h_ok = os.path.exists(
            '/fast/gh/main/phase3/coadd/.13h_astrom_ok_20260708')
        _11h_ok = os.path.exists(
            '/fast/gh/main/phase3/coadd/.11h_astrom_ok_20260708')
        s['pos_flag'] = ('13h-warp~25amin' if (s['lst'] == '13h' and
                                               not _13h_ok) else
                         '11hGreen~16amin' if ((s['lst'], s['band']) ==
                         ('11h', 'Green') and not _11h_ok) else '')
        if s['pos_flag']:
            s['xmatch'] = s['xmatch_sep_arcmin'] = ''
            s['xmatch_sample'] = s['xmatch_dist_pc'] = ''
            sources.append(s)
            continue
        sig_stat = 0.42 * XR_BEAM_AMIN[s['band']] / max(s['snr'], 5.0)
        rad = float(np.hypot(XR_BASE[s['band']], sig_stat))
        if s['elevation_deg'] != '' and s['elevation_deg'] < 40.0:
            rad *= 1.5
        i, d2d, _ = c.match_to_catalog_sky(xcat)
        smin = float(np.atleast_1d(d2d.arcmin)[0])
        if smin < rad:
            s['xmatch'], s['xmatch_sep_arcmin'] = xnames[int(i)], round(smin, 2)
            s['xmatch_sample'], s['xmatch_dist_pc'] = xsamples[int(i)], xdists[int(i)]
        else:
            s['xmatch'] = s['xmatch_sep_arcmin'] = ''
            s['xmatch_sample'] = s['xmatch_dist_pc'] = ''
        sources.append(s)

    sources.sort(key=lambda s: -s['snr'])
    # STABLE source IDs (Gregg 2026-07-10): registry-backed, rank-independent,
    # shared with the V catalog (phase3_source_ids)
    from exopipe.phase3.phase3_source_ids import assign_ids
    n_new = assign_ids(sources, origin='I_deep')
    print(f'source IDs: {n_new} newly minted (rest registry-matched)')
    for rank, s in enumerate(sources, 1):
        s['cand_id'] = f"rank{rank:04d}_{s['lst']}_{s['date']}_{s['band']}"

    # band-balanced review set: TOP_N per band by the source's best flux-band
    # detection in THAT band (Full rides along, no quota)
    review = {}
    if os.environ.get('I_FINAL_NO_PKG') != '1':   # CSV-only mode (packaging
        for b in ('Red', 'Green', 'Blue'):        # refreshed separately)
            has = [s for s in sources if s[f'peak_jy_{b}'] != '']
            for s in has[:TOP_N]:
                review[s['cand_id']] = s
    print(f'{len(sources)} unique sources; {len(review)} review directories')

    os.makedirs(CANDDIR, exist_ok=True)
    for s in review.values():
        d = f"{CANDDIR}/{s['cand_id']}"
        os.makedirs(d, exist_ok=True)
        s['review_dir'] = d
        mem = [finals[i] for i in s['members']]
        with open(f'{d}/detections.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(mem[0].keys()))
            w.writeheader()
            for m in sorted(mem, key=lambda r: -abs(float(r['snr']))):
                w.writerow(m)
        with open(f"{d}/{s['cand_id'].split('_')[0]}.reg", 'w') as fh:
            fh.write(f'fk5\ncircle({s["ra"]:.5f},{s["dec"]:.5f},3600")\n')
        arows = []
        seen_cut = set()
        for m in sorted(mem, key=lambda r: -abs(float(r['snr']))):
            lst, band, date = m['lst'], m['band'], m['date']
            rp = resid_path(lst, band, date)
            mp = f'{COADD}/{lst}_{band}/coadd_final/I_deep_Taper_R075_{band}.fits'
            sp = f'{COADD}/{lst}_{band}/subtracted/{date}_I_sub_{band}.fits'
            arows.append([lst, date, band, m['snr'], rp, mp, sp])
            key = (lst, band, date)
            if key in seen_cut:
                continue
            seen_cut.add(key)
            tgt = s['cand_id'].split('_')[0]
            try:
                if rp:
                    n = prod.build_name(band, 'I', 'deep', 'Taper', -0.75,
                                        proc=['dewarped', 'aligned', 'psfmatch',
                                              'msub', 'ssub'],
                                        lst=lst, epoch=fits.getheader(sp)['DATE-OBS'],
                                        target=tgt, cutout=True)
                    cutout(rp, s['ra'], s['dec'], f'{d}/{n}')
                n = prod.build_name(band, 'I', 'deep', 'Taper', -0.75,
                                    proc=['dewarped', 'aligned', 'coadd'],
                                    lst=lst, epoch=fits.getheader(sp)['DATE-OBS'],
                                    target=tgt, cutout=True)
                if not os.path.exists(f'{d}/{n}'):
                    cutout(mp, s['ra'], s['dec'], f'{d}/{n}')
                # stage-B aligned epoch = sub + master (cheap, faithful)
                sub = fits.getdata(sp).squeeze()
                mst = fits.getdata(mp).squeeze()
                tmp = f'{d}/.tmp_ep.fits'
                fits.writeto(tmp, (sub + mst).astype(np.float32),
                             fits.getheader(sp), overwrite=True)
                n = prod.build_name(band, 'I', 'deep', 'Taper', -0.75,
                                    proc=['dewarped', 'aligned'],
                                    lst=lst, epoch=fits.getheader(sp)['DATE-OBS'],
                                    target=tgt, cutout=True)
                cutout(tmp, s['ra'], s['dec'], f'{d}/{n}')
                os.remove(tmp)
            except Exception as e:
                print(f'  [cutout] {s["cand_id"]} {lst} {band} {date}: {e}')
        with open(f'{d}/allsky_images.csv', 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['lst', 'date', 'band', 'snr',
                        'allsky_residual_fits', 'master_fits',
                        'stageB_sub_fits'])
            w.writerows(arows)
        print(f'  {s["cand_id"]}: n_det={s["n_det"]} packaged')

    with open(OUTCSV + '.tmp', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction='ignore')
        w.writeheader()
        for s in sources:
            s.setdefault('review_dir', '')
            w.writerow(s)
    os.replace(OUTCSV + '.tmp', OUTCSV)
    print(f'{len(sources)} sources -> {OUTCSV}')
    print(f'review dirs -> {CANDDIR}/')


if __name__ == '__main__':
    main()
