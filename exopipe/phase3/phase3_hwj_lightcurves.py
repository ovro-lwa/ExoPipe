#!/usr/bin/env python3
"""HWJ light curves + period search from Run_HWJ_PHOT1 photometry tables
(stage after the tier sweeps; Gregg spec 2026-07-27).

Per object x Stokes: gather every photometry.csv row (all cadences, all
bands), time-order -> {obj}/lightcurve_{S}.csv (one file per Stokes; band +
cadence columns keep tiers separable).

Period search (V priority, then I), per object x band, following the
all_planet_ls_perband.py conventions: Lomb-Scargle 0.3-40 d on the forced
src_jy series (weights 1/rms^2), 10-min points preferred (deep points used
when no 10-min exists for that epoch/band), Baluev FAP, day-alias flag.
ECM is narrow-band -> bands searched separately, never gated on multi-band.
Candidate = FAP < 0.01 and non-alias in >=1 band.

Outputs at run root: periodsearch/ls_results_{S}.csv (every obj x band) +
periodsearch/ls_candidates_{S}.csv (the flagged subset).

Usage: phase3_hwj_lightcurves.py [--stokes V I] [--root ...]
"""
import os, sys, csv, glob, argparse
from collections import defaultdict

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from astropy.timeseries import LombScargle

ROOT = '/data07/gh/phase3/photometry/Run_HWJ_PHOT1'
ALIAS = [1.0, 0.99727, 0.5, 0.49863, 0.33333, 2.0]

# --- transient-pipeline contamination criteria, applied as TAGS only ---
from astropy.coordinates import SkyCoord, get_body, EarthLocation
from astropy.time import Time
from astropy import units as u
OVRO = EarthLocation(lat=37.23977727*u.deg, lon=-118.2816667*u.deg,
                     height=1222*u.m)
ATEAM_SKY = SkyCoord([350.850, 299.867, 83.629, 187.704, 139.524]*u.deg,
                     [58.815, 40.734, 22.014, 12.391, -12.096]*u.deg)
ATEAM_DEG = {('V', 'deep'): 2.0, ('V', '10min'): 2.0, ('V', 'snapshot'): 2.0,
             ('I', 'deep'): 2.0, ('I', '10min'): 5.0, ('I', 'snapshot'): 5.0}
JUP_DEG = {'Red': 5.0, 'Full': 5.0, 'Green': 0.5, 'Blue': 0.5}
_COLOR = {**{f'{f}MHz': 'Red' for f in (18, 23, 27, 32, 36)},
          **{f'{f}MHz': 'Green' for f in (41, 46, 50, 55, 59)},
          **{f'{f}MHz': 'Blue' for f in (64, 69, 73, 78, 82)}}
_JUP_CACHE = {}


def _jupiter_sky(date):
    """Apparent Jupiter (distance stripped — the get_body separation
    parallax gotcha)."""
    if date not in _JUP_CACHE:
        jb = get_body('jupiter', Time(date + ' 00:00:00', scale='utc'), OVRO)
        _JUP_CACHE[date] = SkyCoord(jb.ra, jb.dec)
    return _JUP_CACHE[date]


def load_target_coords():
    out = {}
    import re as _re
    tcsv = '/lustre/gh/main/clv15/Hot_Warm_Jupiters.csv'
    for i, r in enumerate(csv.DictReader(open(tcsv, encoding='utf-8-sig')),
                          1):
        try:
            ra, dec = float(r['ra_current']), float(r['dec_current'])
        except (KeyError, ValueError):
            continue
        obj = f'hwj{i:03d}_' + _re.sub(r'[\s/]+', '_',
                                       r['common_name'].strip())
        out[obj] = SkyCoord(ra*u.deg, dec*u.deg)
    return out


def load_static_tags(root):
    tags = {}
    p = os.path.join(root, 'contaminants_summary.csv')
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            tags[r['obj']] = r
    return tags


def load_contam_registry(root):
    """{obj: SkyCoord array of registered contaminant positions}."""
    reg = {}
    p = os.path.join(root, 'contaminant_positions.csv')
    if os.path.exists(p):
        by = {}
        for r in csv.DictReader(open(p)):
            by.setdefault(r['obj'], []).append(
                (float(r['src_ra']), float(r['src_dec'])))
        for obj, pts in by.items():
            reg[obj] = SkyCoord([p[0] for p in pts]*u.deg,
                                [p[1] for p in pts]*u.deg)
    return reg


# nominal beam FWHM (arcmin) per band, measured from archive Taper headers
# 2026-07-28 (per-cell spread exists, esp. Red wb: ~13-28'; nominal is the
# tag threshold only — known_src_sep_arcmin carries the exact distance)
BEAM_ARCMIN = {'Red': 27.6, 'Green': 8.3, 'Blue': 6.2, 'Full': 8.5,
               '18MHz': 21.1, '23MHz': 17.5, '27MHz': 14.9, '32MHz': 13.0,
               '36MHz': 11.5, '41MHz': 10.2, '46MHz': 9.2, '50MHz': 8.5,
               '55MHz': 7.8, '59MHz': 7.3, '64MHz': 6.8, '69MHz': 6.4,
               '73MHz': 6.0, '78MHz': 5.6, '82MHz': 5.5}


# --- positional-association percentile (Gregg 2026-07-29) ---
# Empirical radial-offset percentiles from COADD_ASTROMETRIC_ERROR.md
# (deep co-adds, bright sources; 95-99% tail = resolved probes).
# Effective radius per anchor = sqrt(r_p^2 + (q_p*sigma_stat)^2) with
# q_p the Rayleigh quantile factor, sigma_stat = 0.42*FWHM/|SNR|;
# all anchors x1.5 below el 40 deg; subbands: color curve x(36/nu)^1.5
# below 36 MHz; Full uses the Green curve (8.5' vs 8.3' beam).
# CAVEAT: percentiles are measured on DEEP co-adds — for 10min/single-
# epoch rows they are a LOWER BOUND on the true scatter (doc section 3b).
ASSOC_P = [50., 68., 90., 95., 99.]
ASSOC_Q = [1.1774, 1.5096, 2.1460, 2.4477, 3.0349]   # Rayleigh quantiles
ASSOC_R = {'Blue':  [1.05, 1.29, 2.73, 6.0, 15.7],
           'Green': [1.35, 1.67, 3.14, 6.9, 16.6],
           'Red':   [2.05, 2.61, 4.70, 10.1, 18.8]}


def assoc_percentile(band, off, snr, el):
    """(pctile, class) for a fitted offset off (arcmin). pctile is the
    percentile of the expected real-association scatter at which the
    offset sits (piecewise-linear between anchors, 0' -> 0%); class:
    core <=68, wing <=95, tail <=99, outside >99."""
    if band in ASSOC_R:
        base, scale = ASSOC_R[band], 1.0
    elif band == 'Full':
        base, scale = ASSOC_R['Green'], 1.0
    else:
        try:
            nu = float(band.replace('MHz', ''))
        except ValueError:
            return '', ''
        col = 'Red' if nu <= 36 else ('Green' if nu <= 59 else 'Blue')
        base = ASSOC_R[col]
        scale = (36./nu)**1.5 if nu < 36 else 1.0
    if el is not None and el < 40:
        scale *= 1.5
    sig = 0.42 * BEAM_ARCMIN.get(band, 8.0) / max(abs(snr), 3.0)
    reff = [((r*scale)**2 + (q*sig)**2)**0.5 for r, q in zip(base, ASSOC_Q)]
    pts = [(0.0, 0.0)] + list(zip(reff, ASSOC_P))
    if off > reff[-1]:
        return '>99', 'outside'
    for (r0, p0), (r1, p1) in zip(pts, pts[1:]):
        if off <= r1:
            p = p0 + (p1 - p0) * (off - r0) / max(r1 - r0, 1e-9)
            break
    cls = 'core' if p <= 68 else ('wing' if p <= 95 else 'tail')
    return round(p, 1), cls


# Full is the R+G+B stack, so Red ghost structure enters it at Red's
# spatial scale (measured: Full ghost fits land median 2.5' from the
# same-epoch Red fit); a Full row coinciding with a same-epoch tagged Red
# fit within PROP_ARCMIN inherits tag_known_src=2 (Gregg 2026-07-28)
PROP_ARCMIN = 5.0


def tag_rows(obj, rows, sc, static, registry=None):
    """Append tag_ateam / tag_jupiter / tag_sidelobe75 and, when the
    planet has registered contaminant positions, known_src_sep_arcmin +
    tag_known_src (1 = fitted position within 1.0 x nominal beam of a
    registered source [Gregg 2026-07-28, widened from 0.5 to catch beam
    skirts]; 2 = Full-band fit coincident with a same-epoch directly-
    tagged Red fit — inherited Red contamination in the R+G+B stack)
    to each row dict. Tags only — nothing removed."""
    if sc is None:
        return rows
    asep = float(min(sc.separation(ATEAM_SKY).deg))
    tsl = (static or {}).get('tag_sidelobe75', '0')
    creg = (registry or {}).get(obj)
    for r in rows:
        if creg is not None:
            sep_txt, tag = '', 0
            try:
                fra, fdec = r.get('fit_ra'), r.get('fit_dec')
                if fra not in (None, '', 'None'):
                    fp = SkyCoord(float(fra)*u.deg, float(fdec)*u.deg)
                    s = float(min(fp.separation(creg).arcmin))
                    sep_txt = round(s, 2)
                    beam = BEAM_ARCMIN.get(r.get('band', ''), 8.0)
                    tag = int(s <= beam)
            except Exception:
                pass
            r['known_src_sep_arcmin'] = sep_txt
            r['tag_known_src'] = tag
        else:
            r['known_src_sep_arcmin'] = ''
            r['tag_known_src'] = 0
        r['assoc_pctile'], r['assoc_class'] = '', ''
        if r.get('cadence') != 'snapshot':
            try:
                off = float(r.get('fit_offset_arcmin'))
                snr = float(r.get('peak_snr') or 5.0)
                el = float(r.get('el_deg')) if r.get('el_deg') else None
                r['assoc_pctile'], r['assoc_class'] = assoc_percentile(
                    r.get('band', ''), off, snr, el)
            except (TypeError, ValueError):
                pass
        cad = r.get('cadence') or 'snapshot'
        rad = ATEAM_DEG.get((r.get('stokes', 'V'), cad), 2.0)
        r['tag_ateam'] = int(asep < rad)
        band = r.get('band', '')
        jrad = JUP_DEG.get(_COLOR.get(band, band))
        tj = 0
        if jrad and r.get('date'):
            try:
                tj = int(float(sc.separation(
                    _jupiter_sky(r['date'])).deg) < jrad)
            except Exception:
                tj = 0
        r['tag_jupiter'] = tj
        r['tag_sidelobe75'] = tsl
    if creg is not None:
        epoch = lambda r: (r.get('cadence'), r.get('lst'), r.get('date'),
                           r.get('bin'))
        red_tagged = {}
        for r in rows:
            if (r.get('band') == 'Red' and r.get('tag_known_src') == 1
                    and r.get('fit_ra') not in (None, '', 'None')):
                red_tagged.setdefault(epoch(r), []).append(
                    SkyCoord(float(r['fit_ra'])*u.deg,
                             float(r['fit_dec'])*u.deg))
        for r in rows:
            if (r.get('band') == 'Full' and r.get('tag_known_src') == 0
                    and r.get('fit_ra') not in (None, '', 'None')):
                for rc in red_tagged.get(epoch(r), []):
                    fp = SkyCoord(float(r['fit_ra'])*u.deg,
                                  float(r['fit_dec'])*u.deg)
                    if float(fp.separation(rc).arcmin) <= PROP_ARCMIN:
                        r['tag_known_src'] = 2
                        break
    return rows


def near_alias(P):
    return any(abs(P - a) / a < 0.03 for a in ALIAS)


def load_obj_rows(obj_dir, stokes):
    rows = []
    for pcsv in glob.glob(f'{obj_dir}/{stokes}/*/*/photometry.csv'):
        with open(pcsv) as f:
            for r in csv.DictReader(f):
                rows.append(r)
    rows.sort(key=lambda r: float(r.get('mjd_mid') or r.get('mjd') or 0))
    return rows


def write_lightcurve(obj_dir, stokes, rows):
    if not rows:
        return
    out = os.path.join(obj_dir, f'lightcurve_{stokes}.csv')
    cols = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, restval='')
        w.writeheader()
        w.writerows(rows)


def write_detections(obj_dir, stokes, rows):
    """Planet-level detections CSV (Gregg 2026-07-27): every >=5-sigma row
    across all cadences/bands, ordered by |peak_snr| descending. Snapshot
    rows (COLS differ slightly) are unioned on their shared columns."""
    dets = [r for r in rows if abs(float(r.get('peak_snr') or 0)) >= 5.0]
    out = os.path.join(obj_dir, f'detections_{stokes}.csv')
    if not dets:
        if os.path.exists(out):
            os.remove(out)
        return 0
    dets.sort(key=lambda r: -abs(float(r['peak_snr'])))
    cols = []
    for r in dets:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, restval='')
        w.writeheader()
        w.writerows(dets)
    return len(dets)


def ls_one(rows, band, cadence):
    """LS on one object/band/cadence series. Deep and 10-min are searched
    SEPARATELY (Gregg ruling 2026-08-05 — the old 10-min-preferred logic
    silently excluded deep rows and left deep series unsearched)."""
    use = [r for r in rows if r.get('band') == band
           and r.get('excluded', '0') == '0'
           and r.get('cadence') == cadence]
    if len(use) < 20:
        return None
    t = np.array([float(r['mjd_mid']) for r in use])
    v = np.array([float(r['src_jy']) for r in use])
    dv = np.array([max(float(r['rms_jy']), 1e-3) for r in use])
    if t.ptp() < 3:
        return None
    freq = np.linspace(1/40., 1/0.3, 40000)
    ls = LombScargle(t, v, dv)
    power = ls.power(freq)
    i = int(np.argmax(power))
    P = 1.0 / freq[i]
    fap = float(ls.false_alarm_probability(power[i], method='baluev'))
    return {'band': band, 'n_points': len(use),
            'cadence': cadence,
            'span_d': round(float(t.ptp()), 1),
            'best_period_d': round(P, 5), 'peak_power': round(float(power[i]), 4),
            'fap_baluev': fap, 'is_alias': int(near_alias(P)),
            'candidate': int(fap < 0.01 and not near_alias(P))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stokes', nargs='+', default=['V', 'I'])
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--detections-only', action='store_true',
                    help='only (re)write per-planet detections_{S}.csv')
    args = ap.parse_args()
    root = args.root
    ps_dir = os.path.join(root, 'periodsearch')
    os.makedirs(ps_dir, exist_ok=True)
    objs = sorted(d for d in os.listdir(root)
                  if d.startswith('hwj') and
                  os.path.isdir(os.path.join(root, d)))
    coords = load_target_coords()
    static = load_static_tags(root)
    registry = load_contam_registry(root)
    for stokes in args.stokes:
        # pass 1 — per-(cadence,band) frame-rms MAD outlier cutoffs (Gregg
        # 2026-07-29: corrupted frames carry blown-up rms — e.g. bad-night
        # snapshots at 10^2-10^5 x thermal — but scale-invariant SNR still
        # "detects" on them; omit modified-z > 3.5 high-rms rows from all
        # detections outputs; audit sidecars, nothing deleted from
        # photometry.csv)
        # snapshot rows are pre-filtered by stage A of the merge (cutoffs
        # from the FULL measurement set, snap_merge/snap_rms_cutoffs.csv);
        # the per-obj snapshot files hold >=5sig rows only, so recomputing
        # cutoffs from them would be biased — skip that cadence here.
        by_bc = {}
        for obj in objs:
            for r in load_obj_rows(os.path.join(root, obj), stokes):
                if (r.get('cadence') or 'snapshot') == 'snapshot':
                    continue
                try:
                    by_bc.setdefault((r.get('cadence'), r.get('band')),
                                     []).append(float(r['rms_jy']))
                except (KeyError, TypeError, ValueError):
                    pass
        rms_cut = {}
        cut_rows = []
        for k, v in by_bc.items():
            a = np.array(v)
            med = float(np.median(a))
            mad = float(np.median(np.abs(a - med)))
            cut = med + 3.5 * mad / 0.6745 if mad > 0 else float('inf')
            rms_cut[k] = cut
            cut_rows.append([k[0], k[1], len(v), round(med, 4),
                             round(mad, 4), round(cut, 4)])
        with open(os.path.join(root, f'rms_flag_cutoffs_{stokes}.csv'),
                  'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['cadence', 'band', 'n_rows', 'median_rms',
                        'mad', 'cutoff_rms'])
            w.writerows(sorted(cut_rows, key=lambda r: (str(r[0]), str(r[1]))))

        def rms_ok(r):
            try:
                return float(r['rms_jy']) <= rms_cut.get(
                    (r.get('cadence'), r.get('band')), float('inf'))
            except (KeyError, TypeError, ValueError):
                return True

        results, cands = [], []
        all_dets = []
        rms_omitted = []
        for obj in objs:
            od = os.path.join(root, obj)
            rows = load_obj_rows(od, stokes)
            rms_omitted += [r for r in rows if not rms_ok(r)
                            and abs(float(r.get('peak_snr') or 0)) >= 5.0]
            rows = [r for r in rows if rms_ok(r)]
            rows = tag_rows(obj, rows, coords.get(obj), static.get(obj),
                            registry)
            all_dets += [r for r in rows
                         if abs(float(r.get('peak_snr') or 0)) >= 5.0]
            nd = write_detections(od, stokes, rows)
            if args.detections_only:
                if nd:
                    print(f'{stokes} {obj}: {nd} detections', flush=True)
                continue
            write_lightcurve(od, stokes, rows)
            bands = sorted(set(r['band'] for r in rows))
            for band in bands:
                for cadence in ('deep', '10min'):
                    res = ls_one(rows, band, cadence)
                    if res:
                        res = {'obj': obj, **res}
                        results.append(res)
                        if res['candidate']:
                            cands.append(res)
            print(f'{stokes} {obj}: {len(rows)} pts, '
                  f'{len(bands)} bands', flush=True)
        if rms_omitted:
            with open(os.path.join(root,
                      f'detections_{stokes}_rms_omitted.csv'),
                      'w', newline='') as f:
                cols = []
                for r in rms_omitted:
                    for k in r.keys():
                        if k not in cols:
                            cols.append(k)
                w = csv.DictWriter(f, fieldnames=cols, restval='')
                w.writeheader()
                w.writerows(rms_omitted)
            print(f'{stokes}: rms-flag omitted {len(rms_omitted)} '
                  f'>=5sig rows (audit sidecar)', flush=True)
        # single run-root CSV per Stokes (Gregg 2026-07-28), all planets,
        # ordered by |peak_snr| descending, plus per-tier splits (Gregg
        # 2026-07-28); tier derived from cadence + band type
        all_dets.sort(key=lambda r: -abs(float(r['peak_snr'])))
        for r in all_dets:
            cad = r.get('cadence') or 'snapshot'
            if cad == 'snapshot':
                r['tier'] = 'snapshot'
            else:
                sb = 'MHz' in (r.get('band') or '')
                r['tier'] = f"{'sb' if sb else 'wb'}_{cad}"
        if all_dets:
            cols = []
            for r in all_dets:
                for k in r.keys():
                    if k not in cols:
                        cols.append(k)
            with open(os.path.join(root, f'detections_{stokes}.csv'),
                      'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=cols, restval='')
                w.writeheader()
                w.writerows(all_dets)
            print(f'{stokes}: run-root detections_{stokes}.csv '
                  f'{len(all_dets)} rows', flush=True)
            for tier in ('wb_deep', 'sb_deep', 'wb_10min', 'sb_10min',
                         'snapshot'):
                trows = [r for r in all_dets if r['tier'] == tier]
                tout = os.path.join(root, f'detections_{stokes}_{tier}.csv')
                if not trows:
                    if os.path.exists(tout):
                        os.remove(tout)
                    continue
                with open(tout, 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=cols, restval='')
                    w.writeheader()
                    w.writerows(trows)
                print(f'{stokes}:   {tier}: {len(trows)} rows', flush=True)
        if results:
            for name, data in [(f'ls_results_{stokes}.csv', results),
                               (f'ls_candidates_{stokes}.csv', cands)]:
                with open(os.path.join(ps_dir, name), 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                    w.writeheader()
                    w.writerows(data)
        print(f'{stokes}: {len(results)} obj-band series, '
              f'{len(cands)} LS candidates', flush=True)


if __name__ == '__main__':
    main()
