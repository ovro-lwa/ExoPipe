#!/usr/bin/env python3
"""candinv/tests.py — the investigation test arms.

Each arm takes the candidate + already-loaded discovery images and returns a
flat dict of OBJECTIVE measurements plus boolean/string flags. No arm declares
"real/fake"; the score arm (candinv/score.py) combines these into a labelled,
non-authoritative 0-10 corroboration score and Gregg judges.

Arms:
  morphology   T1  V source size/shape vs the Stokes-I empirical beam
  astrometry   T2  VLSS-anchored positional error budget (-> spectrum box)
  spectrum     T3  forced photometry across all bands (+ 10-min lightcurve)
  iv_coincide  A1  I/V residual coincidence; |V|<=I sanity (corrected physics)
  context          elevation, gal/ecl coords, #(>Nsigma), A-team/ephemeris, x-match
"""
import numpy as np
from astropy.coordinates import SkyCoord, AltAz, get_body
from astropy.time import Time
import astropy.units as u

from . import config as C
from . import imaging as I
from . import resolve as R
from . import refs
from . import photometry as PHOT
from exopipe.common.pipeline_config import OVRO_LOC
from exopipe.phase3.lib.search.candidates import ephemeris_label


# ── T1: morphology vs the SUBBAND empirical I beam (Gregg's method) ─────────
def morphology(cand, spec, vfit, imas, nominal_beam_deg, fallback_bpix):
    """Point-source test: imfit the V source vs an empirical beam from unresolved
    Stokes-I sources in the SAME band/subband (Gregg). Valid on a SUBBAND or on a
    Red/Green/Blue band (each ~20 MHz) — NEVER on Full (18-82 MHz synthesis),
    where the wideband beam would false-diagnose a narrow-band burst as resolved.

    Prefers the strongest detected subband (conformant I, has a beam header);
    otherwise, for a Red/Green/Blue candidate, tests in that band using the
    discovery V size + the band I master (beam injected, since bundle masters are
    beamless). Returns point_like=None only when the sole detection is in Full."""
    subs = [r for r in spec if r.get('is_subband') and r.get('v_detected')
            and r.get('v_fwhm_maj_deg') == r.get('v_fwhm_maj_deg')]  # finite
    if subs:
        best = max(subs, key=lambda r: abs(r.get('v_snr', 0)))
        scope = best['scope']
        vmaj = best['v_fwhm_maj_deg']
        vmin = best.get('v_fwhm_min_deg') or vmaj
        ipath = (R.prod_residual(cand.lst, cand.date, scope, 'I', cand.cadence,
                                 cand.iv, 'aligned')
                 or R.prod_residual(cand.lst, cand.date, scope, 'I', 'deep',
                                    None, 'aligned'))
        eb = (PHOT.empirical_beam_imfit(ipath, exclude_radec=(cand.ra, cand.dec),
                                        fallback_bpix=fallback_bpix)
              if ipath else None)
    elif cand.band in ('Red', 'Green', 'Blue'):
        # detected only in a wideband BAND (not Full) — test in that band
        if not (vfit and vfit.get('fwhm_maj_deg') == vfit.get('fwhm_maj_deg')):
            return {'point_like': None, 'note': 'band detection but no V size'}
        scope = cand.band
        vmaj = vfit['fwhm_maj_deg']
        vmin = vfit.get('fwhm_min_deg') or vmaj
        eb = (PHOT.empirical_beam_imfit(imas, exclude_radec=(cand.ra, cand.dec),
                                        beam=nominal_beam_deg,
                                        fallback_bpix=fallback_bpix)
              if imas else None)
    else:
        return {'point_like': None,
                'note': 'detected only in Full — not diagnosed (wideband synthesis)'}
    if not eb:
        return {'point_like': None, 'morph_subband': scope,
                'note': f'could not fit an empirical beam in {scope}'}
    out = {'morph_subband': scope, 'v_fwhm_maj_deg': vmaj, 'v_fwhm_min_deg': vmin,
           'ebeam_maj_deg': eb[0], 'ebeam_min_deg': eb[1], 'ebeam_n_src': eb[3],
           'maj_ratio': vmaj / max(eb[0], 1e-9),
           'min_ratio': vmin / max(eb[1], 1e-9)}
    out['point_like'] = bool(out['maj_ratio'] < 1.35 and out['min_ratio'] < 1.4)
    out['extended'] = bool(out['maj_ratio'] >= 1.5 or out['min_ratio'] >= 1.5)
    return out


# ── helper: light bright-source extractor (no BDSF dependency) ─────────────
def extract_bright(data, wcs, bpix, cd, nmax=60, fmin_sigma=8.0):
    rms = I.central_rms(data)
    if not np.isfinite(rms) or rms <= 0:
        return []
    thr = fmin_sigma * rms
    work = np.where(np.isfinite(data), data, -np.inf).copy()
    excl = max(8, int(round(1.5 * bpix)))
    srcs = []
    for _ in range(nmax * 4):
        iy, ix = np.unravel_index(np.argmax(work), work.shape)
        v = work[iy, ix]
        if v < thr:
            break
        work[max(0, iy - excl):iy + excl + 1,
             max(0, ix - excl):ix + excl + 1] = -np.inf
        fit = I.fit_source(data, ix, iy, bpix, cd, force_sign=1)
        if fit is None:
            continue
        ra, dec = I.pix_to_world(wcs, fit['x_fit'], fit['y_fit'])
        srcs.append((ra, dec, v))
        if len(srcs) >= nmax:
            break
    return srcs


# ── T2: astrometry vs VLSS (BDSF source list) ──────────────────────────────
def astrometry(ipath, freq_mhz, nominal_beam_deg):
    """Extract the Stokes-I source list with BDSF (imfit/BDSF-only rule) and
    match to VLSS -> systematic offset + scatter. Defines the spectrum box.
    ν-scaled match radius. BDSF beam override since bundle masters are beamless."""
    out = {'astrom_n_match': 0}
    try:
        import bdsf
        img = bdsf.process_image(ipath, thresh_pix=12.0, thresh_isl=5.0,
                                 adaptive_rms_box=True, quiet=True,
                                 beam=tuple(nominal_beam_deg))
        srcs = [(s.posn_sky_max[0], s.posn_sky_max[1], s.peak_flux_max)
                for s in img.sources if not np.isnan(s.posn_sky_max[0])]
    except Exception:
        srcs = []
    out['astrom_n_isrc'] = len(srcs)
    if len(srcs) < 3:
        out['astrom_err_arcmin'] = C.ASTROM_FLOOR_ARCMIN
        out['box_half_arcmin'] = C.ASTROM_FLOOR_ARCMIN
        return out
    srcs.sort(key=lambda s: -s[2])
    srcs = srcs[:500]
    ra_a = np.array([s[0] for s in srcs])
    dec_a = np.array([s[1] for s in srcs])
    vcoord, _vflux = refs.vlss_unresolved()
    # ν-scaled match radius (ionospheric offset grows toward low freq).
    base = 6.0  # arcmin at 74 MHz
    match_r = float(np.clip(base * (74.0 / max(freq_mhz, 10.0)) ** 1.0,
                            4.0, 30.0))
    # ONE vectorized catalog match (the per-source loop was a bottleneck).
    sc_all = SkyCoord(ra_a * u.deg, dec_a * u.deg)
    idx, sep2d, _ = sc_all.match_to_catalog_sky(vcoord)
    mask = sep2d.arcmin <= match_r
    out['astrom_n_match'] = int(mask.sum())
    out['astrom_match_radius_arcmin'] = match_r
    if mask.sum() >= 3:
        mv = vcoord[idx[mask]]
        cosd = np.cos(np.radians(dec_a[mask]))
        # R-D3-FIX (2026-08-16): ra_a comes from BDSF posn_sky_max, i.e. the
        # image's own CRVAL1 branch (negative for LST 12h-23h masters), while
        # mv.ra.deg is 0-360. Use the SkyCoord built above (which normalises)
        # AND wrap the difference before the cos(dec) scaling, so the RA 0/360
        # seam is covered too.
        sc_m = sc_all[mask]
        dra = (((sc_m.ra.deg - mv.ra.deg) + 180.0) % 360.0 - 180.0) * cosd * 60.0
        ddec = (sc_m.dec.deg - mv.dec.deg) * 60.0
        out['astrom_dra_arcmin'] = float(np.median(dra))
        out['astrom_ddec_arcmin'] = float(np.median(ddec))
        sys_off = float(np.hypot(np.median(dra), np.median(ddec)))
        scatter = float(np.hypot(I._mad(dra), I._mad(ddec)))
        out['astrom_sys_offset_arcmin'] = sys_off
        out['astrom_scatter_arcmin'] = scatter
        out['astrom_err_arcmin'] = float(np.hypot(sys_off, scatter))
        out['box_half_arcmin'] = max(C.ASTROM_FLOOR_ARCMIN,
                                     sys_off + 2.0 * scatter)
    else:
        out['astrom_err_arcmin'] = C.ASTROM_FLOOR_ARCMIN
        out['box_half_arcmin'] = C.ASTROM_FLOOR_ARCMIN
    return out


# ── forced photometry at a fixed sky position in one residual image ────────
def _forced(path, ra, dec, fallback_bpix, beam=None, box_half_arcmin=None):
    """imfit forced photometry within the astrometric-error box
    (candinv.photometry.forced_imfit). imfit non-convergence => non-detection."""
    if path is None:
        return None
    return PHOT.forced_imfit(path, ra, dec, box_half_arcmin=box_half_arcmin,
                             fallback_bpix=fallback_bpix, beam=beam)


def _detected(m):
    return bool(m and m.get('converged') and m.get('snr', 0) >= C.DETECT_SNR)


# ── T3: spectrum across bands (+ A1 per-band I/V) ──────────────────────────
def spectrum(cand, ra, dec, box_half_arcmin, fallback_bpix):
    """Forced V photometry (imfit, astrometric box) per subband + wideband;
    coincident I only where V is detected (for the per-band I/V ratio)."""
    rows = []
    for scope in R.spectrum_scopes(cand.cadence):
        vmsub, vsrc = R.v_spectrum_image(cand.lst, cand.date, scope,
                                         cand.cadence, cand.iv)
        v = _forced(vmsub, ra, dec, fallback_bpix, box_half_arcmin=box_half_arcmin)
        if v is None:
            continue
        fhz = C.scope_center_hz(scope)
        row = {'scope': scope, 'freq_mhz': (fhz / 1e6 if fhz else None),
               'is_subband': scope in C.SUBBANDS, 'v_source': vsrc,
               'v_peak': v['peak'], 'v_err': v.get('err'), 'v_snr': v['snr'],
               'v_sign': v['sign'], 'v_upper_limit': v.get('upper_limit'),
               'v_fwhm_maj_deg': v['fwhm_maj_deg'],
               'v_fwhm_min_deg': v.get('fwhm_min_deg'),
               'v_axis_ratio': v['axis_ratio'], 'v_ra': v['ra'], 'v_dec': v['dec'],
               'v_ra_err_arcmin': v.get('ra_err_arcmin'),
               'v_dec_err_arcmin': v.get('dec_err_arcmin'),
               'v_detected': _detected(v)}
        if row['v_detected']:
            imsub = R.prod_residual(cand.lst, cand.date, scope, 'I',
                                    cand.cadence, cand.iv, 'msub')
            i = _forced(imsub, ra, dec, fallback_bpix,
                        box_half_arcmin=box_half_arcmin)
            if i:
                row.update(i_peak=i['peak'], i_err=i.get('err'), i_snr=i['snr'],
                           i_detected=_detected(i))
                if _detected(i) and i['peak'] != 0:
                    row['vi_ratio'] = abs(v['peak']) / abs(i['peak'])
        rows.append(row)
    return rows


def summarize_spectrum(rows):
    sub = [r for r in rows if r.get('is_subband') and r.get('v_detected')]
    out = {'n_scope_v_detect': sum(1 for r in rows if r.get('v_detected')),
           'n_subband_v_detect': len(sub)}
    # independent bands (Red/Green/Blue) with a subband detection
    bands = set()
    for r in sub:
        b = C.SUBBAND_TO_BAND.get(r['scope'])
        if b:
            bands.add(b)
    # wideband detections also count as their band
    for r in rows:
        if r.get('v_detected') and r['scope'] in ('Red', 'Green', 'Blue'):
            bands.add(r['scope'])
    out['v_bands'] = sorted(bands)
    out['n_v_bands'] = len(bands)
    # highest-freq strong detection -> refined position
    strong = [r for r in rows if r.get('v_snr', 0) >= C.STRONG_SNR
              and r.get('is_subband') and r.get('v_ra') is not None]
    strong.sort(key=lambda r: r['freq_mhz'] or 0, reverse=True)
    if strong:
        out['refined_ra'] = strong[0]['v_ra']
        out['refined_dec'] = strong[0]['v_dec']
        out['refined_from'] = strong[0]['scope']
    return out


def tenmin_lightcurve(cand, ra, dec, fallback_bpix, scope=None,
                      box_half_arcmin=None):
    """Per-bin V SNR sequence at the candidate band -> within-epoch variability.
    Drives the T5 (10-second snapshot) escalation gate."""
    scope = scope or cand.band
    bins = R.prod_tenmin_bins(cand.lst, cand.date, scope, 'V', 'msub')
    lc = []
    for b, path in bins:
        f = _forced(path, ra, dec, fallback_bpix, box_half_arcmin=box_half_arcmin)
        if f:
            lc.append({'bin': b, 'snr': f['snr'], 'peak': f['peak'],
                       'sign': f['sign'], 'detected': _detected(f)})
    out = {'tenmin_scope': scope, 'tenmin_nbin': len(lc), 'lightcurve': lc}
    snrs = [p['snr'] for p in lc if np.isfinite(p['snr'])]
    if len(snrs) >= 2:
        mx, mn = max(snrs), min(snrs)
        out['tenmin_var_frac'] = float((mx - mn) / max(mx, 1e-9))
        out['tenmin_peak_bin'] = lc[int(np.argmax(snrs))]['bin']
        out['short_structure'] = bool(out['tenmin_var_frac'] >=
                                      C.SNAPSHOT_VAR_FRAC and mx >= C.DETECT_SNR)
    else:
        out['short_structure'] = False
    return out


# ── T4: multi-epoch search (gated by band corroboration) ───────────────────
def multi_epoch(cand, ra, dec, fallback_bpix, scope=None, cadence='deep',
                box_half_arcmin=None):
    """Forced V photometry at the position across ALL dates at this LST.
    Deep cadence (the sensitive integration). Multi-epoch repeat = strongest
    confirmation alongside multi-band."""
    scope = scope or cand.band
    rows = []
    for date in R.prod_dates_for_lst(cand.lst):
        vmsub = R.prod_residual(cand.lst, date, scope, 'V', cadence, None, 'msub')
        f = _forced(vmsub, ra, dec, fallback_bpix, box_half_arcmin=box_half_arcmin)
        if f is None:
            continue
        rows.append({'date': date, 'v_snr': f['snr'], 'v_peak': f['peak'],
                     'v_sign': f['sign'], 'detected': _detected(f),
                     'is_discovery': date == cand.date})
    ndet = sum(1 for r in rows if r['detected'])
    frac = ndet / len(rows) if rows else 0.0
    # Detection in (almost) EVERY epoch of a master-subtracted residual = a
    # steady source / imperfect subtraction, NOT a transient. Flag it.
    persistent = bool(len(rows) >= 4 and frac >= 0.8)
    out = {'epoch_scope': scope, 'n_epoch_searched': len(rows),
           'n_epoch_detect': ndet, 'frac_detect': round(frac, 3),
           'persistent': persistent, 'epochs': rows}
    return out


# ── A1: I/V coincidence on the discovery (wideband) residual ───────────────
def iv_coincidence(vfit, ires_path, ra, dec, fallback_bpix, beam=None,
                   box_half_arcmin=None):
    """Corrected physics: I >= |V|; 100%-CP => I = |V|. Real CP transient shows
    in BOTH residuals. Flag coincidence + unphysical |V|>I."""
    out = {'v_peak': vfit['peak'], 'v_abs': abs(vfit['peak'])}
    i = _forced(ires_path, ra, dec, fallback_bpix, beam=beam,
                box_half_arcmin=box_half_arcmin)
    if i:
        out.update(i_resid_peak=i['peak'], i_resid_snr=i['snr'])
        out['i_resid_detected'] = _detected(i)
        if _detected(i) and i['peak'] != 0:
            out['vi_ratio'] = abs(vfit['peak']) / abs(i['peak'])
            out['unphysical_vi'] = bool(out['vi_ratio'] > C.VI_RATIO_UNPHYSICAL)
        out['iv_coincident'] = bool(out.get('i_resid_detected'))
    else:
        out['i_resid_detected'] = None
        out['iv_coincident'] = None
    return out


# ── context: elevation, coords, peak count, ephemeris, cross-match ─────────
def _horizon_masked_rms_and_peaks(vdata, vwcs, vhdr, bpix, obs_time, nsigma):
    """#(>nsigma POSITIVE) peaks above the horizon (el>0), using inner rms.
    Masks the bright horizon ring that inflates the naive count."""
    # Sample a coarse grid of pixel->el to build an above-horizon mask cheaply.
    h, w = vdata.shape
    step = max(1, h // 256)
    ys = np.arange(0, h, step); xs = np.arange(0, w, step)
    gx, gy = np.meshgrid(xs, ys)
    ra, dec = vwcs.all_pix2world(gx, gy, 0)
    sc = SkyCoord(ra * u.deg, dec * u.deg)
    el = sc.transform_to(AltAz(obstime=obs_time, location=OVRO_LOC)).alt.deg
    # Upsample the coarse el mask back to full res via nearest repeat.
    elmask = el > 15.0
    full = np.repeat(np.repeat(elmask, step, axis=0), step, axis=1)[:h, :w]
    rms = I.central_rms(vdata)
    if not np.isfinite(rms) or rms <= 0:
        return -1, rms
    sig = np.where(full & np.isfinite(vdata), vdata, 0.0) / rms
    npk = 0
    excl = max(6, int(round(bpix)))
    masked = sig.copy()
    for _ in range(20000):
        iy, ix = np.unravel_index(np.argmax(masked), masked.shape)
        if masked[iy, ix] < nsigma:
            break
        npk += 1
        masked[max(0, iy - excl):iy + excl + 1,
               max(0, ix - excl):ix + excl + 1] = 0.0
    return npk, rms


def context(cand, vdata, vwcs, vhdr, bpix, obs_iso):
    coord = cand.skycoord
    out = {}
    t = Time(obs_iso) if obs_iso else None
    if t is not None:
        aa = coord.transform_to(AltAz(obstime=t, location=OVRO_LOC))
        out['elevation_deg'] = float(aa.alt.deg)
        out['azimuth_deg'] = float(aa.az.deg)
        out['low_elevation'] = bool(aa.alt.deg < C.ELEV_MIN_DEG)
    out['gal_l_deg'] = float(coord.galactic.l.deg)
    out['gal_b_deg'] = float(coord.galactic.b.deg)
    ec = coord.barycentrictrueecliptic
    out['ecl_lon_deg'] = float(ec.lon.deg)
    out['ecl_lat_deg'] = float(ec.lat.deg)
    name, sep = refs.nearest_ateam(coord)
    out['nearest_ateam'] = f'{name} {sep:.1f}deg'
    out['ateam_sep_deg'] = sep
    if t is not None:
        out['ephemeris_body'] = ephemeris_label(cand.ra, cand.dec, t)
        # Jupiter explicit (known V contaminant).
        try:
            jup = get_body('jupiter', t, OVRO_LOC)
            out['jupiter_sep_deg'] = float(coord.separation(
                SkyCoord(jup.ra, jup.dec)).deg)
        except Exception:
            pass
        npk, rms = _horizon_masked_rms_and_peaks(vdata, vwcs, vhdr, bpix, t, 7.0)
        out['n_peaks_gt7sigma'] = npk
        out['discovery_rms'] = float(rms)
    # cross-match (ν-scaled radius using candidate band center)
    fc = C.SUBBAND_FREQ_HZ.get(cand.band, {}).get('center')
    radius = 6.0 if not fc else float(np.clip(
        6.0 * (74.0 / (fc / 1e6)), 3.0, 20.0))
    out['xmatch_radius_arcmin'] = radius
    xm = {}
    for key in C.XMATCH_KEYS:
        hit = refs.crossmatch(coord, key, radius)
        if hit:
            xm[key] = {'name': hit[0], 'sep_arcmin': round(hit[1], 2),
                       'dist_pc': hit[2]}
    out['xmatch'] = xm
    return out
