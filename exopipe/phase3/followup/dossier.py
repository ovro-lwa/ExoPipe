#!/usr/bin/env python3
"""candinv/dossier.py — orchestrate the arms and assemble a conformant dossier.

investigate(cand) runs the always-on core (morphology, astrometry, spectrum,
I/V coincidence, 10-min lightcurve, context), then the corroboration-gated
escalations (T4 all-epoch; T5 10-second snapshots — gated on 10-min structure),
scores, and writes everything under
  {OUT_ROOT}/transients/Run_{OUT_LABEL}/candidates/{cand_id}/
via phase3_products (FITS) and the documented sidecar convention (CSV/json/reg).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from . import candidate as candmod
from . import imaging as I
from . import resolve as R
from . import tests as T
from . import io_products as IO
from . import score as SC
from . import snapshots as SNAP
from . import photometry as PHOT
from . import astrometry as AST
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from exopipe.common import phase3_paths as P


def _reticle(ax, x, y, gap=9, length=20, color='lime', lw=1.3):
    """Gap-centered reticle marking (x,y) WITHOUT covering the source pixel
    (standing feedback: never put a marker on top of the source)."""
    ax.plot([x - gap - length, x - gap], [y, y], color=color, lw=lw)
    ax.plot([x + gap, x + gap + length], [y, y], color=color, lw=lw)
    ax.plot([x, x], [y - gap - length, y - gap], color=color, lw=lw)
    ax.plot([x, x], [y + gap, y + gap + length], color=color, lw=lw)


def investigate(cand, escalate=True, write=True, verbose=True, snapshot=True):
    ev = {'candidate': {'rank': cand.rank, 'cand_id': cand.cand_id,
                        'ra': cand.ra, 'dec': cand.dec, 'band': cand.band,
                        'cadence': cand.cadence, 'lst': cand.lst,
                        'date': cand.date, 'iv': cand.iv,
                        'peak_snr_disc': cand.peak_snr, 'ndates': cand.ndates,
                        'bands_disc': cand.bands}}

    def log(*a):
        if verbose:
            print('  ', *a)

    # --- discovery products -------------------------------------------------
    vres = R.bundle_path(cand.rank, cand.lst, cand.band, cand.date,
                         cand.cadence, cand.iv, 'Vresidual')
    ires = R.bundle_path(cand.rank, cand.lst, cand.band, cand.date,
                         cand.cadence, cand.iv, 'Iresidual')
    imas = R.bundle_path(cand.rank, cand.lst, cand.band, cand.date,
                         cand.cadence, cand.iv, 'Imaster')
    ev['inputs'] = {'Vresidual': vres, 'Iresidual': ires, 'Imaster': imas}
    if vres is None:
        ev['error'] = 'discovery Vresidual not found in bundle'
        return ev

    vdata, vhdr, vwcs = I.load_image(vres)
    cd = I.cdelt_deg(vhdr)
    x0, y0 = I.world_to_pix(vwcs, cand.ra, cand.dec)

    # Beam bootstrap for imfit: HEADER-FIRST (subband-aware fix, Gregg
    # 2026-07-13 — the conformant P3_PROD residuals carry the empirical beam;
    # a fixed 3.2 px nominal is only right on post-cutover WIDEBAND grids and
    # badly undersizes subband beams). Fall back to the 3.2 px nominal only
    # for genuinely beamless bundle files. The SCIENTIFIC point-source test
    # still uses the per-subband imfit empirical beam in T1 morphology.
    if vhdr.get('BMAJ'):
        nominal_beam_deg = (float(vhdr['BMAJ']),
                            float(vhdr.get('BMIN', vhdr['BMAJ'])),
                            float(vhdr.get('BPA', 0.0)))
        bpix = nominal_beam_deg[0] / cd
    else:
        nominal_beam_deg = (3.2 * cd, 3.2 * cd, 0.0)
        bpix = 3.2
    ebeam_deg = nominal_beam_deg
    ev['nominal_beam_deg'] = nominal_beam_deg

    # --- V forced photometry on discovery (imfit-on-cutout) ----------------
    vm = PHOT.forced_imfit(vres, cand.ra, cand.dec, fallback_bpix=bpix,
                           beam=ebeam_deg)
    if vm is None:
        ev['error'] = 'discovery forced photometry failed'
        return ev
    vfit = {'peak': vm['peak'], 'err': vm.get('err'), 'snr': vm['snr'],
            'local_rms': vm['rms'],
            'ra': vm['ra'], 'dec': vm['dec'], 'converged': vm['converged'],
            'fwhm_maj_deg': vm['fwhm_maj_deg'], 'fwhm_min_deg': vm['fwhm_min_deg'],
            'axis_ratio': vm['axis_ratio'], 'pa_deg': vm['pa_deg'],
            'pos_err_arcmin': vm.get('pos_err_arcmin')}
    disc_ra, disc_dec = vm['ra'], vm['dec']
    ev['discovery_fit'] = vfit
    ar = vfit['axis_ratio']
    log(f'discovery V: peak={vm["peak"]:.3f} Jy snr={vm["snr"]:.1f} '
        f'conv={vm["converged"]} axis_ratio={ar if ar==ar else "n/a"}')

    # --- epoch times (for headers + elevation) ------------------------------
    times = R.epoch_times(cand.lst, cand.date, cand.band, cand.cadence)
    obs_iso = times[0]

    # --- T3 spectrum (imfit; box ~beam-floor — astrometric error is sub-beam for
    # dewarped data, and the precise error is reported by T2 below) ----------
    spec = T.spectrum(cand, disc_ra, disc_dec, None, bpix)
    ev['spectrum'] = spec
    ev['spectrum_summary'] = T.summarize_spectrum(spec)
    ra = ev['spectrum_summary'].get('refined_ra', disc_ra)
    dec = ev['spectrum_summary'].get('refined_dec', disc_dec)
    ev['refined_position'] = {'ra': ra, 'dec': dec,
                              'from': ev['spectrum_summary'].get('refined_from')}
    log(f'spectrum: {ev["spectrum_summary"]["n_v_bands"]} V band(s) '
        f'{ev["spectrum_summary"].get("v_bands")}')

    # strongest detected SUBBAND — used by morphology + escalation
    sub_det = [r for r in spec if r.get('is_subband') and r.get('v_detected')]
    best_sub = (max(sub_det, key=lambda r: abs(r.get('v_snr', 0)))['scope']
                if sub_det else None)

    # --- T2 astrometry: position from the highest-frequency STRONG detection;
    # error = statistical (imfit) + direction-dependent systematic LOWER LIMIT
    # from bright nearby anchors matched to VLSSr via the 73 MHz band. ---------
    ev['astrometry'] = _astrometry(cand, spec, vfit, disc_ra, disc_dec)
    aa = ev['astrometry']
    log(f'astrometry [{aa.get("position_band")}]: pos '
        f'RA {aa.get("pos_ra"):.4f} Dec {aa.get("pos_dec"):.4f}  '
        f'stat={aa.get("sigma_stat_arcmin")}\' sys>={aa.get("sigma_sys_lowerlimit_arcmin")}\' '
        f'total>={aa.get("total_pos_err_arcmin")}\'')

    # --- T1 morphology (imfit vs empirical I beam; subband or R/G/B, not Full)
    ev['morphology'] = T.morphology(cand, spec, vfit, imas, nominal_beam_deg, bpix)
    mo = ev['morphology']
    log(f'morphology [{mo.get("morph_subband","-")}]: point_like='
        f'{mo.get("point_like")} maj_ratio={mo.get("maj_ratio")}')

    # --- A1 I/V coincidence -------------------------------------------------
    ev['iv_coincidence'] = T.iv_coincidence(vfit, ires, ra, dec, bpix,
                                            beam=ebeam_deg, box_half_arcmin=None)
    log(f'I/V: coincident={ev["iv_coincidence"].get("iv_coincident")} '
        f'vi_ratio={ev["iv_coincidence"].get("vi_ratio")}')

    # --- 10-min lightcurve --------------------------------------------------
    ev['tenmin'] = T.tenmin_lightcurve(cand, ra, dec, bpix,
                                       box_half_arcmin=None)
    log(f'10-min: nbin={ev["tenmin"]["tenmin_nbin"]} '
        f'short_structure={ev["tenmin"].get("short_structure")}')

    # --- context ------------------------------------------------------------
    ev['context'] = T.context(cand, vdata, vwcs, vhdr, bpix, obs_iso)
    log(f'context: el={ev["context"].get("elevation_deg")} '
        f'#>7sig={ev["context"].get("n_peaks_gt7sigma")} '
        f'xmatch={list(ev["context"]["xmatch"].keys())}')

    # --- escalations --------------------------------------------------------
    ev['escalation'] = {}
    nbands = ev['spectrum_summary']['n_v_bands']
    # best_sub computed above (strongest detected subband; wideband deep is not
    # kept per-epoch, so the all-epoch search runs on the subband).
    # Gate T4 (all-epoch search — the strongest confirmation) on a STRONG
    # single-band detection or a discovery repeat. NEVER gate on multi-band:
    # ECM is narrowband, so requiring >=2 bands would skip exactly the coherent
    # sources we most want to confirm (Gregg / project_ecm_narrowband_kelt24).
    strong_det = any(r.get('v_snr', 0) >= C.STRONG_SNR for r in spec)
    disc_repeat = bool(cand.ndates and cand.ndates >= 2)
    if escalate and best_sub and (strong_det or disc_repeat or
                                  nbands >= C.ESCALATE_MIN_BANDS):
        why = ('strong-band' if strong_det else
               (f'discovery ndates={cand.ndates}' if disc_repeat else 'multi-band'))
        ev['escalation']['multi_epoch'] = f'run (subband {best_sub}; {why})'
        ev['multi_epoch'] = T.multi_epoch(cand, ra, dec, bpix, scope=best_sub,
                                          box_half_arcmin=None)
        log(f'T4 multi-epoch [{best_sub}; {why}]: '
            f'{ev["multi_epoch"]["n_epoch_detect"]}/'
            f'{ev["multi_epoch"]["n_epoch_searched"]} epochs detected '
            f'(frac={ev["multi_epoch"]["frac_detect"]})')
    else:
        ev['escalation']['multi_epoch'] = (
            f'not warranted (no strong band, ndates={cand.ndates})')
    # --- T5 snapshot: ALWAYS run (Gregg 2026-06-28) -------------------------
    # 10-second lightcurve + snapshot spectrum + peak-frame morphology, no longer
    # gated on 10-min structure. We drill into the PEAK 10-min bin of the
    # snapshot scope to bound the frame count (~75 frames) and keep it tractable.
    if escalate and snapshot:
        try:
            _run_snapshot(cand, ev, ra, dec, best_sub, spec, bpix, log)
        except Exception as e:
            ev['escalation']['snapshot'] = f'error: {e}'
            log('T5 snapshot error:', e)
    else:
        ev['escalation']['snapshot'] = 'skipped (--no-snapshot)'

    # --- ranking: P(astrophysical) -----------------------------------------
    ev['p_astro'] = SC.p_astrophysical(ev)
    log(f'P(astrophysical) = {ev["p_astro"]["p_astro"]}  '
        f'(logit {ev["p_astro"]["logit"]}) {ev["p_astro"]["components"]}')

    if write:
        ev['outputs'] = _write_dossier(cand, ev, vres, ires, imas, times,
                                       ebeam_deg, ra, dec, bpix)
    return ev


# ---------------------------------------------------------------------------
def _snapshot_scope(cand, spec, best_sub):
    """Subband to run the 10-second lightcurve on. Snapshots are per-subband
    only, and a run may hold frames for only SOME subbands, so we pick the first
    candidate scope that ACTUALLY HAS frames (Gregg 2026-06-28 — otherwise a
    candidate with snapshots in a different subband gets no lightcurve at all,
    e.g. ranks 26/43 where only 64MHz was populated). Preference order: strongest
    detected subband, then strongest |V SNR| subband, then the discovery band's
    constituent subbands (centre-out), then the band itself."""
    pref = []
    if best_sub:
        pref.append(best_sub)
    subs = sorted([r for r in spec if r.get('is_subband')
                   and r.get('v_snr') is not None],
                  key=lambda r: -abs(r.get('v_snr', 0)))
    pref += [r['scope'] for r in subs]
    if cand.band in C.SUBBANDS:
        pref.append(cand.band)
    consts = C.SUBBAND_BANDS.get(cand.band)
    if consts:                                   # centre-out
        order = sorted(range(len(consts)), key=lambda i: abs(i - len(consts) // 2))
        pref += [consts[i] for i in order]
    seen, ordered = set(), []
    for s in pref:
        if s and s not in seen:
            seen.add(s); ordered.append(s)
    # first scope that actually has frames; else the first preference (-> note)
    for s in ordered:
        if SNAP.has_frames(cand.lst, cand.date, s):
            return s
    return ordered[0] if ordered else (cand.band if cand.band in C.SUBBANDS
                                       else '55MHz')


def _ts_token(iso):
    """ISO time -> 'YYYYMMDD_HHMMSS' snapshot filename token."""
    return Time(iso).to_datetime().strftime('%Y%m%d_%H%M%S')


def _run_snapshot(cand, ev, ra, dec, best_sub, spec, bpix, log):
    """Always-on T5: localize to the peak 10-min bin of the snapshot scope, run
    the 10-second V lightcurve, then the snapshot spectrum + peak-frame
    morphology at the peak frame. Stores results in ev (FITS saved at write)."""
    from astropy.io import fits as _f
    sc = _snapshot_scope(cand, spec, best_sub)
    ev['snapshot_scope'] = sc
    # localize the burst to a 10-min bin. For a 10-min candidate the discovery
    # bin is authoritative; for a deep candidate find the peak 10-min bin (on the
    # snapshot scope, else the discovery band).
    if cand.cadence == '10min' and cand.iv is not None:
        peak_bin = cand.iv
    else:
        peak_bin = None
        for lcsc in (sc, cand.band):
            lc = T.tenmin_lightcurve(cand, ra, dec, bpix,
                                     scope=lcsc).get('lightcurve', [])
            if lc:
                peak_bin = max(lc, key=lambda p: abs(p['snr']))['bin']
                break
    # 10-min bin windows are SCOPE-INDEPENDENT (same wall-clock binning), so read
    # the window from any scope that has the bin (PROD residual or Phase-2 image).
    start = end = None
    if peak_bin is not None:
        for wsc in [cand.band, sc] + list(C.SUBBANDS):
            w = (R.prod_residual(cand.lst, cand.date, wsc, 'V', '10min',
                                 peak_bin, 'msub')
                 or R.phase2_subband_image(cand.lst, cand.date, wsc, 'V',
                                           '10min', peak_bin))
            if w:
                hh = _f.getheader(w)
                start, end = hh.get('DATE-OBS'), hh.get('DATE-END')
                if start:
                    break
    if not start:                       # deep-only fallback: whole-epoch window
        start, end, _ = R.epoch_times(cand.lst, cand.date, sc, 'deep')
    if not start:
        ev['escalation']['snapshot'] = f'no window to localize ({sc})'
        return
    snap = SNAP.snapshot_lightcurve(cand.lst, cand.date, sc, ra, dec, start, end,
                                    pad_s=C.SNAPSHOT_PAD_S)
    snap['peak_bin'] = peak_bin
    ev['snapshot'] = snap
    log(f'T5 snapshot [{sc}, bin{peak_bin}]: {len(snap.get("frames",[]))} frames, '
        f'peak {snap.get("peak_snr",0):.1f} at {snap.get("peak_time")}')
    if not snap.get('peak_time'):
        ev['escalation']['snapshot'] = f'run ({sc}); no frames in window'
        return
    peak_ts = _ts_token(snap['peak_time'])
    ev['snapshot']['peak_ts'] = peak_ts
    # instantaneous spectrum across all subbands at the peak 10-second frame
    try:
        ev['snapshot_spectrum'] = SNAP.snapshot_spectrum(
            cand.lst, cand.date, ra, dec, peak_ts, verbose=False)
    except Exception as e:
        log('T5 snapshot spectrum error:', e)
        ev['snapshot_spectrum'] = []
    # peak-frame point-source test (the rank-48 decider) in the strongest
    # snapshot-spectrum band, else the lightcurve scope
    morph_sc = sc
    ss = [r for r in ev.get('snapshot_spectrum', [])
          if r.get('V_snr') is not None]
    if ss:
        morph_sc = max(ss, key=lambda r: abs(r.get('V_snr', 0)))['scope']
    try:
        ev['snapshot_morphology'] = SNAP.snapshot_morphology(
            cand.lst, cand.date, morph_sc, ra, dec, peak_ts, verbose=False)
    except Exception as e:
        log('T5 snapshot morphology error:', e)
    ev['escalation']['snapshot'] = (
        f'run (lightcurve {sc} bin{peak_bin}; spectrum+morphology @ {peak_ts}, '
        f'morph band {morph_sc})')


# ---------------------------------------------------------------------------
def _astrometry(cand, spec, vfit, disc_ra, disc_dec):
    """Position from the highest-frequency STRONG detection + statistical and
    direction-dependent systematic (lower-limit) positional errors."""
    strong = [r for r in spec if r.get('is_subband')
              and r.get('v_snr', 0) >= C.STRONG_SNR and r.get('v_ra') is not None
              and r.get('v_fwhm_maj_deg') == r.get('v_fwhm_maj_deg')]
    strong.sort(key=lambda r: (r.get('freq_mhz') or 0), reverse=True)
    full_only = False
    if strong:
        chk = strong[0]
    else:
        wb = [r for r in spec if r['scope'] in ('Red', 'Green', 'Blue')
              and r.get('v_detected') and r.get('v_ra') is not None]
        wb.sort(key=lambda r: (r.get('freq_mhz') or 0), reverse=True)
        if wb:
            chk = wb[0]
        else:
            full_only = True
            chk = {'scope': cand.band, 'v_ra': disc_ra, 'v_dec': disc_dec,
                   'v_snr': vfit.get('snr'),
                   'v_fwhm_maj_deg': vfit.get('fwhm_maj_deg'),
                   'v_ra_err_arcmin': vfit.get('ra_err_arcmin'),
                   'v_dec_err_arcmin': vfit.get('dec_err_arcmin'),
                   'freq_mhz': (C.scope_center_hz(cand.band) or 50e6) / 1e6}
    cscope, pos_ra, pos_dec = chk['scope'], chk['v_ra'], chk['v_dec']
    # statistical positional error: use imfit's RETURNED per-axis errors (don't
    # compute); fall back to the 0.42*FWHM/SNR rule-of-thumb only if absent. Keep
    # RA and Dec SEPARATE (direction-dependent); combine to a single number only
    # at the very end.
    sra = chk.get('v_ra_err_arcmin')
    sdec = chk.get('v_dec_err_arcmin')
    if sra is None or not (sra == sra) or sdec is None or not (sdec == sdec):
        f = AST.stat_pos_error_arcmin(chk.get('v_fwhm_maj_deg'), chk.get('v_snr'))
        sra = sdec = f
    # detection-band aligned DEEP I image (dewarped, pre-sky-subtraction);
    # fall back to the highest constituent subband for a wideband detection.
    check_ip = R.prod_residual(cand.lst, cand.date, cscope, 'I', 'deep', None,
                               'aligned')
    used = cscope
    if not check_ip and cscope in C.SUBBAND_BANDS:
        for sb in reversed(C.SUBBAND_BANDS[cscope]):
            check_ip = R.prod_residual(cand.lst, cand.date, sb, 'I', 'deep',
                                       None, 'aligned')
            if check_ip:
                used = sb
                break
    ref73 = R.prod_residual(cand.lst, cand.date, '73MHz', 'I', 'deep', None,
                            'aligned')
    sysr = AST.local_systematic(
        SkyCoord(pos_ra * u.deg, pos_dec * u.deg), check_ip, ref73,
        dewarped=True, freq_mhz=chk.get('freq_mhz') or 50.0,
        transient_snr=chk.get('v_snr'))
    sig_sys = sysr.get('sys_lowerlimit_arcmin')
    sig_stat_2d = (float(np.hypot(sra, sdec))
                   if (sra == sra and sdec == sdec) else float('nan'))
    total = (float(np.hypot(sig_stat_2d, sig_sys))
             if (np.isfinite(sig_stat_2d) and sig_sys is not None
                 and np.isfinite(sig_sys)) else None)
    _r = lambda x: (round(x, 3) if x is not None and x == x else None)
    return {'position_band': cscope, 'check_image_scope': used,
            'pos_ra': float(pos_ra), 'pos_dec': float(pos_dec),
            'full_only': full_only,
            # per-axis statistical error (imfit), kept separate:
            'sigma_stat_ra_arcmin': _r(sra), 'sigma_stat_dec_arcmin': _r(sdec),
            # systematic dict carries per-axis offset + scatter components:
            'systematic': sysr,
            # single end-numbers (combined only here):
            'sigma_stat_arcmin': _r(sig_stat_2d),
            'sigma_sys_lowerlimit_arcmin': _r(sig_sys),
            'total_pos_err_arcmin': _r(total),
            'note': ('detected only in Full — VLSS cross-match may be unreliable'
                     if full_only else None)}


def _write_dossier(cand, ev, vres, ires, imas, times, ebeam, ra, dec, bpix):
    outdir = C.candidate_dir(cand.cand_id, make=True)
    beam_t = ev.get('nominal_beam_deg')
    out = {'dir': outdir}
    have_time = times[0] is not None

    # evidence + spectrum + reg (always)
    out['evidence'] = IO.write_json(outdir, cand, ev,
                                    inputs=[vres, ires, imas],
                                    params={'box_half_arcmin':
                                            ev['astrometry'].get('box_half_arcmin')})
    out['spectrum_csv'] = IO.write_spectrum_csv(outdir, cand, ev['spectrum'],
                                                inputs=[vres])
    entries = [(cand.ra, cand.dec, 'yellow', f'rank{cand.rank} disc'),
               (ra, dec, 'green', 'refined')]
    out['reg'] = IO.write_reg(outdir, cand, entries)

    # conformant FITS — all-sky (essential) + cutouts, for V/I residual + I master
    proc_res = ['dewarped', 'aligned', 'msub', 'ssub']
    proc_mas = ['dewarped', 'aligned', 'coadd']
    if have_time:
        try:
            out['allsky_V'] = IO.conform_allsky(
                vres, outdir, cand, cand.band, 'V', proc_res, times,
                beam=beam_t, ebeam=beam_t, dewarped=True, aligned=True,
                msub=True, ssub=True)
            out['cutout_V'] = IO.write_cutout(
                vres, outdir, cand, cand.band, 'V', proc_res, ra, dec, times,
                beam=beam_t, ebeam=beam_t, dewarped=True, aligned=True,
                msub=True, ssub=True)
            if ires:
                out['allsky_I'] = IO.conform_allsky(
                    ires, outdir, cand, cand.band, 'I', proc_res, times,
                    beam=beam_t, ebeam=beam_t, dewarped=True, aligned=True,
                    msub=True, ssub=True)
                out['cutout_I'] = IO.write_cutout(
                    ires, outdir, cand, cand.band, 'I', proc_res, ra, dec, times,
                    beam=beam_t, ebeam=beam_t, dewarped=True, aligned=True,
                    msub=True, ssub=True)
        except Exception as e:
            print('  [io] WARN conformant FITS write failed:', e)
    else:
        print('  [io] WARN no Phase-2 epoch time found — skipping conformant '
              'FITS (cannot stamp standard time block)')

    if ev.get('snapshot', {}).get('frames'):
        out['snapshot_csv'] = IO.write_snapshot_csv(outdir, cand, ev['snapshot'])
    if ev.get('snapshot_spectrum'):
        out['snapshot_spectrum_csv'] = IO.write_snapshot_spectrum_csv(
            outdir, cand, ev['snapshot_spectrum'])
    # conformant DEWARPED burst all-sky FITS for the peak 10-second frame (V+I),
    # in the morphology band (where V is strongest at peak) — Gregg: all-sky FITS
    # are essential, not just cutouts.
    snap = ev.get('snapshot', {})
    peak_ts = snap.get('peak_ts')
    if peak_ts:
        burst_scope = ev.get('snapshot_morphology', {}).get('scope') \
            or ev.get('snapshot_scope') or snap.get('scope')
        try:
            out['snapshot_burst_fits'] = SNAP.save_burst_allsky(
                cand, cand.lst, cand.date, burst_scope, ra, dec, [peak_ts],
                outdir, verbose=False)
        except Exception as e:
            print('  [io] WARN burst all-sky save failed:', e)

    out['snapshot_png'] = _snapshot_png(outdir, cand, ev)
    out['index_png'] = _index_png(outdir, cand, ev, vres, ires, imas, ra, dec, bpix)
    return out


def _snapshot_png(outdir, cand, ev):
    """Per-candidate 10-second snapshot figure (Gregg 2026-06-28 — one for EVERY
    candidate): left = the 10-sec V signed-SNR lightcurve with the peak marked;
    right = the instantaneous snapshot spectrum (V & I |SNR|) at the peak frame.
    Genuine no-data candidates get a labelled placeholder so the file always
    exists."""
    try:
        snap = ev.get('snapshot', {}) or {}
        frames = snap.get('frames', []) or []
        sspec = ev.get('snapshot_spectrum', []) or []
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        a = ax[0]
        if frames:
            t0 = frames[0]['mjd']
            tsec = [(f['mjd'] - t0) * 86400.0 for f in frames]
            ssnr = [f['snr'] * (f.get('sign') or 1) for f in frames]
            a.plot(tsec, ssnr, '.-', ms=4, lw=0.8)
            a.axhline(0, color='gray', lw=0.6)
            for s in (C.STRONG_SNR, -C.STRONG_SNR):
                a.axhline(s, ls='--', color='r', lw=0.6)
            if snap.get('peak_time'):
                for i, f in enumerate(frames):
                    if f['time'] == snap['peak_time']:
                        a.axvline(tsec[i], color='orange', lw=0.8, ls=':')
                        break
            a.set_xlabel('seconds from %s UTC' % frames[0]['time'][:19])
            a.set_ylabel('V signed SNR')
            a.set_title('10-sec V LC [%s]  peak %.1f @ %s  burst ~%ss'
                        % (snap.get('scope'), snap.get('peak_snr', 0) or 0,
                           str(snap.get('peak_time'))[11:19], snap.get('burst_dur_s')))
        else:
            a.axis('off')
            a.text(0.5, 0.5, 'no Phase-1 snapshot data\n%s'
                   % str(ev.get('escalation', {}).get('snapshot', ''))[:60],
                   ha='center', va='center', fontsize=11)
        a = ax[1]
        if sspec:
            fr = [r['freq_mhz'] for r in sspec]
            vs = [abs(r.get('V_snr', 0) or 0) for r in sspec]
            isn = [abs(r.get('I_snr', 0) or 0) for r in sspec]
            a.plot(fr, vs, 'o-', label='|V|')
            a.plot(fr, isn, 's--', alpha=0.6, label='|I|')
            a.axhline(C.STRONG_SNR, ls='--', color='gray', lw=0.6)
            a.set_xlabel('MHz'); a.set_ylabel('|SNR| at peak frame')
            a.set_title('snapshot spectrum @ %s' % snap.get('peak_ts', '-'))
            a.legend(fontsize=8)
        else:
            a.axis('off')
            a.text(0.5, 0.5, 'no snapshot spectrum', ha='center', va='center')
        fig.suptitle('rank%d  %s — 10-second snapshot' % (cand.rank, cand.cand_id),
                     fontsize=12)
        out = os.path.join(outdir, f'{cand.cand_id}_snapshot_lc.png')
        fig.tight_layout(); fig.savefig(out, dpi=90); plt.close(fig)
        return out
    except Exception as e:
        print('  [io] WARN snapshot png failed:', e)
        return None


def _index_png(outdir, cand, ev, vres, ires, imas, ra, dec, bpix):
    try:
        fig, ax = plt.subplots(2, 3, figsize=(15, 9))
        half = 80
        for j, (src, title) in enumerate([(vres, 'V residual'),
                                          (ires, 'I residual'),
                                          (imas, 'I master')]):
            a = ax[0][j]
            if src and os.path.exists(src):
                d, h, w = I.load_image(src)
                x, y = I.world_to_pix(w, ra, dec)
                x, y = int(round(x)), int(round(y))
                sub = d[max(0, y - half):y + half, max(0, x - half):x + half]
                fin = sub[np.isfinite(sub)]
                v = np.nanpercentile(np.abs(fin), 99) if fin.size else 1.0
                a.imshow(sub, origin='lower', cmap='RdBu_r', vmin=-v, vmax=v)
                _reticle(a, sub.shape[1] / 2, sub.shape[0] / 2)
            a.set_title(title); a.set_xticks([]); a.set_yticks([])
        # spectrum
        sp = [r for r in ev['spectrum'] if r.get('is_subband') and r.get('freq_mhz')]
        a = ax[1][0]
        if sp:
            fr = [r['freq_mhz'] for r in sp]
            sn = [r.get('v_snr', 0) for r in sp]
            a.plot(fr, sn, 'o-')
            a.axhline(C.DETECT_SNR, ls='--', color='gray')
            a.set_xlabel('MHz'); a.set_ylabel('V SNR'); a.set_title('Spectrum')
        # lightcurve
        a = ax[1][1]
        lc = ev.get('tenmin', {}).get('lightcurve', [])
        if lc:
            a.plot([p['bin'] for p in lc], [p['snr'] for p in lc], 'o-')
            a.axhline(C.DETECT_SNR, ls='--', color='gray')
            a.set_xlabel('10-min bin'); a.set_ylabel('V SNR')
            a.set_title('Lightcurve')
        # text panel
        a = ax[1][2]; a.axis('off')
        co = ev.get('p_astro', {})
        sm = ev.get('snapshot_morphology', {})
        txt = [f'{cand.cand_id}',
               f'P(astro)={co.get("p_astro")} (logit {co.get("logit")})',
               f'V bands: {ev["spectrum_summary"].get("v_bands")}',
               f'el={ev["context"].get("elevation_deg",0):.0f} deg',
               f'gal b={ev["context"].get("gal_b_deg",0):.1f}',
               f'I/V coincident={ev["iv_coincidence"].get("iv_coincident")}',
               f'morph point_like={ev.get("morphology",{}).get("point_like")}',
               f'snap morph point_like={sm.get("point_like")}',
               f'snap peak SNR={ev.get("snapshot",{}).get("peak_snr")}',
               f'xmatch={list(ev["context"]["xmatch"].keys())}']
        a.text(0.02, 0.95, '\n'.join(txt), va='top', fontsize=11, family='monospace')
        out = os.path.join(outdir, f'{cand.cand_id}_index.png')
        fig.suptitle(f'rank{cand.rank}  {cand.cand_id}', fontsize=13)
        fig.tight_layout()
        fig.savefig(out, dpi=90); plt.close(fig)
        return out
    except Exception as e:
        print('  [io] WARN index png failed:', e)
        return None
