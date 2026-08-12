#!/usr/bin/env python3
"""
phase3_final_csv.py — STAGE 3 of the Run_FINAL build: assemble the FINAL TRANSIENT CSV
(PHASE3_HANDOFF.md §5.9, locked columns) from the survivor list (stage 1, which also
assigns rank-based cand_ids and repeat groups) + the reconstruction photometry ledger
(stage 2). All >=5sigma survivors are included (filterable by snr); cutouts/all-sky
(>=7sigma) were written in stage 2.

  - photometry join   : snr-keyed nearest stage-2 BDSF detection per survivor
  - per-band flux     : peak_jy_{Red,Green,Blue,Full} from same-(lst,date) sibling-band
                        >=5sigma detections within the nu-scaled radius (3'x73/nu);
                        blank if not detected in that band
  - elevation         : AltAz at the epoch midpoint (recomputed here — the stage-2
                        recorded value used a wrong obs_time historically)
  - body              : solar-system ephemeris label (flag in place, never dropped)
  - cross-match       : 10pc/25pc/50pc/Hot_Warm_Jupiters/LPTs/flare_ucd/stars.csv,
                        nu-scaled radius -> xmatch, xmatch_sample, xmatch_dist_pc
  - c50_mjy           : measured C50 sigma (phase3_c50) x epoch inner rms
  - pointer files     : candidates/{cand_id}/allsky_images.txt — every all-sky frame
                        (V+I residual + I master) covering ANY detection of the
                        candidate's repeat group (Gregg 2026-07-02)

Usage: python phase3_final_csv.py
"""
import sys
import os, csv, glob
from collections import defaultdict
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, get_body, EarthLocation, AltAz
from astropy.time import Time
from exopipe.phase3.phase3_c50 import c50_sigma

OUT = '/data07/gh/phase3/transients/Run_FINAL'
FINAL = f'{OUT}/V_deep_FINAL_transients.csv'
BAND_CTR = {'Red': 27.0, 'Green': 50.0, 'Blue': 73.0, 'Full': 50.0}
XMATCH_R0_ARCMIN, XMATCH_NU0 = 3.0, 73.0
PHOT_MATCH_DEG = 0.25
CATS = {'10pc': '/lustre/gh/main/clv15/10pc.csv',
        '25pc': '/lustre/gh/main/clv15/25pc.csv',
        '50pc': '/lustre/gh/main/clv15/50pc.csv',
        'Hot_Warm_Jupiters': '/lustre/gh/main/clv15/Hot_Warm_Jupiters.csv',
        'LPTs': '/lustre/gh/main/clv15/LPTs.csv',
        'flare_ucd': '/lustre/gh/main/clv15/flare_ucd.csv',
        'stars': '/lustre/gh/main/clv15/stars.csv',
        # ATNF pulsar catalogue (VizieR B/psr, 2530 rows; added Gregg 2026-07-03
        # after PSR J2215+5135 was found at rank17 with a blank xmatch)
        'ATNF_psr': '/lustre/gh/main/clv15/pulsars_atnf.csv'}
EPH_BODIES = ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn',
              'uranus', 'neptune']
EPH_MATCH_DEG = 1.0
OVRO = EarthLocation(lat=37.2339 * u.deg, lon=-118.2817 * u.deg, height=1222 * u.m)
COLS = ['cand_id', 'source_id', 'lst', 'date', 'band', 'ra', 'dec', 'ra_hms', 'dec_dms',
        'gal_lat_deg', 'ecl_lat_deg',
        'snr', 'peak_jy', 'int_jy', 'sign', 'elevation_deg',
        'gfit_fwhm_beams', 'gfit_ar', 'gfit_resid',
        'peak_jy_Red', 'peak_jy_Green', 'peak_jy_Blue', 'peak_jy_Full',
        'mstruct', 'iv0', 'bilobe', 'gaussfit', 'fp_class', 'n_v_epoch',
        'nearest_ateam', 'jupiter_deg',
        'n_det', 'repeat_group', 'repeat_lst_dates',
        'c50_mjy', 'body', 'xmatch', 'xmatch_sep_arcmin', 'xmatch_sample',
        'xmatch_dist_pc']


def load_phot():
    det, ep = defaultdict(list), {}
    for p in glob.glob(f'{OUT}/phot/*_epochs.csv'):
        for r in csv.DictReader(open(p)):
            ep[(r['lst'], r['date'], r['band'])] = r
    for p in glob.glob(f'{OUT}/phot/*.csv'):
        if p.endswith('_epochs.csv'):
            continue
        for r in csv.DictReader(open(p)):
            r['ra'] = float(r['ra']); r['dec'] = float(r['dec'])
            det[(r['lst'], r['date'], r['band'])].append(r)
    return det, ep


def nearest(rows, ra, dec, max_deg, snr=None):
    pool = rows
    if snr is not None:
        same = [r for r in rows if abs(float(r['snr']) - snr) < 0.011]
        if same:
            pool = same
    best, bd = None, max_deg
    cosd = np.cos(np.radians(dec))
    for r in pool:
        d = np.hypot((r['ra'] - ra) * cosd, r['dec'] - dec)
        if d < bd:
            best, bd = r, d
    return best, bd


def load_xcat():
    ras, decs, names, samples, dists = [], [], [], [], []
    for tag, path in CATS.items():
        if not os.path.exists(path):
            print(f'  WARNING: missing catalog {path}'); continue
        for r in csv.DictReader(open(path)):
            try:
                if 'ra_current' in r and r.get('ra_current'):
                    ra, dec, nm = r['ra_current'], r['dec_current'], r['common_name']
                    dist = r.get('distance', '')
                else:
                    ra, dec, nm = r['ra'], r['dec'], r['name']; dist = ''
                ras.append(float(ra)); decs.append(float(dec))
                names.append(nm.strip()); samples.append(tag); dists.append(dist)
            except Exception:
                pass
    return (SkyCoord(np.array(ras) * u.deg, np.array(decs) * u.deg),
            names, samples, dists)


_EPH_CACHE = {}


def body_label(ra, dec, utc):
    if not utc:
        return ''
    co = SkyCoord(ra * u.deg, dec * u.deg)
    for b in EPH_BODIES:
        key = (b, utc)
        if key not in _EPH_CACHE:
            try:
                bp = get_body(b, Time(utc), OVRO)
                _EPH_CACHE[key] = SkyCoord(bp.ra, bp.dec)
            except Exception:
                _EPH_CACHE[key] = None
        bc = _EPH_CACHE[key]
        if bc is not None and bc.separation(co).deg < EPH_MATCH_DEG:
            return b
    return ''


def write_pointer_files(ep):
    """candidates/{cand_id}/allsky_images.txt — all all-sky frames covering any
    detection of the candidate's repeat group (V+I residual per epoch + I master
    per involved cell)."""
    n = 0
    for r in csv.DictReader(open(f'{OUT}/survivors_7s_index.csv')):
        cdir = f'{OUT}/candidates/{r["cand_id"]}'
        if not os.path.isdir(cdir):
            continue
        lines = []
        cells = set()
        for tok in r['group_epochs'].split('|'):
            l, d, b = tok.split(':')
            cells.add((l, b))
            ts = (ep.get((l, d, b), {}).get('utc_start', '') or '')
            ts = ts.replace('-', '').replace(':', '')[:15]
            if not ts:
                continue
            for stokes, rob in (('V', 'Robust+0.0'), ('I', 'Robust-0.75')):
                fn = (f'{b}_{stokes}_deep_Taper_{rob}_dewarped_aligned_msub_ssub_'
                      f'LST{l}_{ts}.fits')
                if os.path.exists(f'{OUT}/allsky/{fn}'):
                    lines.append(fn)
        for l, b in sorted(cells):
            g = glob.glob(f'{OUT}/allsky/{b}_I_deep_Taper_Robust*_dewarped_'
                          f'aligned_coadd_LST{l}_*.fits')
            lines.extend(sorted(os.path.basename(x) for x in g))
        with open(f'{cdir}/allsky_images.txt', 'w') as fh:
            fh.write(f'# all-sky frames in {OUT}/allsky/ covering every detection of '
                     f'{r["cand_id"]}\'s repeat group\n')
            fh.write('\n'.join(dict.fromkeys(lines)) + '\n')
        n += 1
    print(f'wrote {n} allsky_images.txt pointer files')


def main():
    surv = list(csv.DictReader(open(f'{OUT}/survivors_pre_phot.csv')))
    for s in surv:
        s['ra'] = float(s['ra']); s['dec'] = float(s['dec'])
        s['snr'] = float(s['snr'])
    det, ep = load_phot()
    print(f'{len(surv)} survivors | phot ledger: {sum(len(v) for v in det.values())} '
          f'detections over {len(ep)} epoch-images')

    nmiss, maxd = 0, 0.0
    for s in surv:
        key = (s['lst'], s['date'], s['band'])
        m, d = nearest(det.get(key, []), s['ra'], s['dec'], PHOT_MATCH_DEG,
                       snr=s['snr'])
        if m is None:
            nmiss += 1
            s.update(peak_jy='', int_jy='', sign='')
        else:
            maxd = max(maxd, d)
            s.update(peak_jy=m['peak_jy'], int_jy=m['int_jy'], sign=m['sign'])
        for b in ('Red', 'Green', 'Blue', 'Full'):
            col = f'peak_jy_{b}'
            if b == s['band']:
                s[col] = s.get('peak_jy', '')
                continue
            rad = XMATCH_R0_ARCMIN * (XMATCH_NU0 / BAND_CTR[b]) / 60.0
            mb, _ = nearest(det.get((s['lst'], s['date'], b), []),
                            s['ra'], s['dec'], rad)
            s[col] = mb['peak_jy'] if mb else ''
    print(f'photometry join: {nmiss} unmatched (of {len(surv)}), '
          f'max match dist {maxd*3600:.1f} arcsec')

    xcat, xnames, xsamples, xdists = load_xcat()
    for s in surv:
        key = (s['lst'], s['date'], s['band'])
        utc = ep.get(key, {}).get('utc_start', '')
        s['body'] = body_label(s['ra'], s['dec'], utc)
        if utc:
            it = float(ep[key].get('inttime_s') or 3600.0)
            aa = SkyCoord(s['ra'] * u.deg, s['dec'] * u.deg).transform_to(
                AltAz(obstime=Time(utc) + (it / 2) * u.s, location=OVRO))
            s['elevation_deg'] = round(float(aa.alt.deg), 2)
        else:
            s['elevation_deg'] = ''
        cs = c50_sigma(s['band'])
        ir = ep.get(key, {}).get('inner_rms_jy', '')
        s['c50_mjy'] = (round(cs * float(ir) * 1000.0, 1)
                        if cs and ir not in ('', None) else '')
        c = SkyCoord(s['ra'] * u.deg, s['dec'] * u.deg)
        s['ra_hms'] = c.ra.to_string(unit=u.hourangle, sep=':', precision=1, pad=True)
        s['dec_dms'] = c.dec.to_string(sep=':', precision=0, alwayssign=True, pad=True)
        # Galactic + Ecliptic latitude (Gregg 2026-07-02 — restored from older iterations)
        s['gal_lat_deg'] = round(float(c.galactic.b.deg), 2)
        s['ecl_lat_deg'] = round(float(c.barycentricmeanecliptic.lat.deg), 2)
        rad = XMATCH_R0_ARCMIN * (XMATCH_NU0 / BAND_CTR[s['band']])
        i, d2d, _ = c.match_to_catalog_sky(xcat)
        smin = float(np.atleast_1d(d2d.arcmin)[0])
        if smin < rad:
            s['xmatch'] = xnames[int(i)]
            s['xmatch_sep_arcmin'] = round(smin, 2)
            s['xmatch_sample'] = xsamples[int(i)]
            s['xmatch_dist_pc'] = xdists[int(i)]
        else:
            s['xmatch'] = s['xmatch_sep_arcmin'] = ''
            s['xmatch_sample'] = s['xmatch_dist_pc'] = ''

    # ---- Gaussian-fit shape columns for the >=7sigma set (Gregg 2026-07-03,
    # after rank14: fitted FWHM in beams + axis ratio + fit residual, measured
    # with the production fitter on the saved all-sky V frames; blank <7sigma
    # (no full-frame residuals archived for the faint tail)) ----
    from exopipe.phase3 import phase3_veto as pv
    from astropy.wcs import WCS
    from astropy.io import fits as _fits
    by_epoch = defaultdict(list)
    for s in surv:
        s['gfit_fwhm_beams'] = s['gfit_ar'] = s['gfit_resid'] = ''
        if s['snr'] >= 7:
            by_epoch[(s['lst'], s['date'], s['band'])].append(s)
    nfit = 0
    for (l, d, b), members in sorted(by_epoch.items()):
        e = ep.get((l, d, b), {})
        ts = (e.get('utc_start', '') or '').replace('-', '').replace(':', '')[:15]
        p = (f'{OUT}/allsky/{b}_V_deep_Taper_Robust+0.0_dewarped_aligned_'
             f'msub_ssub_LST{l}_{ts}.fits')
        if not ts or not os.path.exists(p):
            continue
        with _fits.open(p) as hd:
            img = np.nan_to_num(hd[0].data.squeeze().astype(float))
            hdr = hd[0].header
        w = WCS(hdr).celestial
        beam_px = hdr['BMAJ'] / abs(hdr['CDELT1'])
        for s in members:
            x, y = [int(round(v)) for v in
                    w.all_world2pix([[s['ra'], s['dec']]], 0)[0]]
            gf = pv._gfit(img, x, y, beam_px)
            if gf:
                s['gfit_fwhm_beams'] = round(gf['fwhm'] / beam_px, 2)
                s['gfit_ar'] = round(gf['ar'], 2)
                s['gfit_resid'] = round(gf['resid'], 3)
                nfit += 1
    print(f'gaussian-shape columns filled for {nfit} >=7sigma rows')

    surv.sort(key=lambda s: -s['snr'])
    from exopipe.phase3.phase3_source_ids import assign_ids
    _n_new = assign_ids(surv, origin='V_deep')
    print(f'source IDs: {_n_new} newly minted (rest registry-matched)')
    with open(FINAL, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction='ignore')
        w.writeheader()
        for s in surv:
            w.writerow(s)
    print(f'wrote {len(surv)} rows -> {FINAL}')

    write_pointer_files(ep)

    n7 = sum(1 for s in surv if s['snr'] >= 7)
    print(f'>=7sigma rows: {n7}')
    for nm in ('FI Cnc', 'UX Ari'):
        hits = [s for s in surv if nm.lower() in s['xmatch'].lower()]
        if hits:
            h = max(hits, key=lambda s: s['snr'])
            print(f'  {nm}: RECOVERED n_rows={len(hits)} best {h["snr"]}s '
                  f'{h["lst"]} {h["date"]} {h["band"]} n_det={h["n_det"]}')
        else:
            print(f'  {nm}: NOT FOUND in xmatch — CHECK')


if __name__ == '__main__':
    main()
