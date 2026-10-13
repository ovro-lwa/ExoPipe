#!/usr/bin/env python3
"""
phase3_i_deep.py — PRODUCTION deep-wideband Stokes-I search
(ratified by Gregg 2026-07-05; PHASE3_HANDOFF.md §5.13/§5.14e).

Chain per (LST, band) cell, all epochs (data_decisions + two-stage-QA drops):
  R075 I master -> stage-B aligned epoch (subtracted/_I_sub + master) ->
  x2-flow refine (cached flows) -> PSF-MATCH: symmetric cross-convolution
  (D = G_m(x)epoch - G_e(x)master) -> 2nd flux-scale fit on the MATCHED pair
  (guarded: ns>=8, DC in [0.5,2]) -> per-pair dilution renorm -> destripe.
  Residuals archived with standard headers (PSFMATCH/RENORM/SCALE2B keys).

Per-epoch time block (the §5.12 INTTIME/elevation fix, ported from the
ledger): masks are built from the EPOCH's DATE-OBS + single-block INTTIME,
never the master's summed-stack INTTIME.

Detection: +-I (both signs) at the 5-sigma write floor; el>=25 + horizon +
A-team + invalid-WCS masks (sit.CONFIG). POSITIVE-ONLY reporting: the final
list is sign='+'; negative rows are recorded in the ledger, never final.

Gate sequence (§5.13 as ratified; EVERY quantity also recorded so the veto
retunes offline without reruns):
  1. mstruct-I  (destriped matched-I-master local rms; 5*max(ms,1))
  2. bilobe     (production params, unconditional per pack §3)
  3. gaussfit   (|snr|>8; ar/resid only, NO size cut — ECM rule)
  4. Jupiter ephemeris (Red/Full 5deg, Green/Blue 0.5deg; SkyCoord(jb.ra,jb.dec))
  5. 70 Jy / 1.5 deg bright exclusion (retained, upstream of the dynamic gate)
  6. DYNAMIC bright-source gate (Gregg 2026-07-05): per-image cliff at the
     DEFAULT radius 0.5 beams — veto if assoc. master local max >= Fc*, where
     Fc* = largest flux keeping >=7sigma positives <= floor+2; floor =
     off-source (assoc<2 Jy) + ratio-protected (peak > guard*assoc,
     guard = max(0.2, 2*f_img)); IMAGE DISCARD if floor > 10.
     The 2-D (Fc, radius) retune stays OFFLINE: assoc master local-max is
     recorded at ALL scan radii 0.5/1/1.5/2 beams per candidate, and the
     per-image scan (f_img/floor/Fc*/kept7 per radius) goes to the images CSV.
RECORDED, NEVER GATED: n_epoch_raw, sstack, dist_b7_beams, ateam_deg, sign.

Outputs (env-overridable for fleet path parity — no node needs /data07):
  P3_COADD_ROOT  masters+subtracted   (REQUIRED - no default since 2026-09-04)
  P3_FLOW_ROOT   cached x2 flows      (default /data07/gh/phase3/deep)
  P3_OUT         ledger + images CSV  (default /data07/gh/phase3/transients/Run_I_DEEP)
  P3_RESID_ROOT  residual archive     (default /data07/gh/phase3/i_resid)
Usage: python phase3_i_deep.py <LST> <SCOPE>
"""
from exopipe.common.roots import coadd_root
import sys
_ARGV = list(sys.argv)          # BEFORE phase3_deep import (argv clobber gotcha)
import os, re, glob, csv, fcntl, time
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_body
import astropy.units as u
from scipy.ndimage import maximum_filter
from astropy.convolution import convolve_fft

from exopipe.phase3.phase3_deep import _load, _sigma_stack, MIN_EPOCHS
from exopipe.phase3.phase3_i_psfmatch_ab import matched_resid, gk
from exopipe.phase3.phase3_i_gate_ledger import bilobe_measurables, _mstruct_i
from exopipe.phase3.phase3_tenmin import _det_header
from exopipe.phase3.identify import fit_empirical_beam
from exopipe.phase3.coadd_full_pipeline import (OVRO_LOC, load_exclusions, is_excluded,
                                 get_midpoint_time)
from exopipe.phase3 import search_i_transients as sit
from exopipe.phase3 import phase3_veto as pv
from exopipe.phase3.core import products as prod

COADD_ROOT = os.environ.get('P3_COADD_ROOT')   # NO default (Gregg 2026-09-04); use coadd_root() at point of use
FLOW_ROOT = os.environ.get('P3_FLOW_ROOT', '/data07/gh/phase3/deep')
OUT = os.environ.get('P3_OUT', '/data07/gh/phase3/transients/Run_I_DEEP')
RESID_ROOT = os.environ.get('P3_RESID_ROOT', '/data07/gh/phase3/i_resid')
LEDGER = f'{OUT}/i_deep_ledger.csv'
IMGCSV = f'{OUT}/i_deep_images.csv'
NPROC = int(os.environ.get('I_DEEP_NPROC', '3'))

SIGMA_MSTRUCT = 5.0                       # ratified k=5
GF_SNR_GATE = pv.SNR_GATE                 # gaussfit gate trigger (production, =8)
# ECM-safe SIZE BOUND on fitted FWHM/beam (ratified Gregg 2026-07-07):
# worst-case narrowband source = the band's LOWEST subband PSF through the
# matched chain: hypot(theta_min, kernel)/matched_beam, x1.20 empirical-beam-
# fit margin (Gregg). Measured theta_min (18/41/64 MHz Taper R075 restoring
# beams) and matched beams from the production archive. Gate applies wherever
# gfw is measured (snr>=6). Kills the Cygnus-Loop/resolved class; B0950/J0410
# verified source-level safe.
SIZE_BOUND = {'Blue': 1.26, 'Green': 1.32, 'Red': 1.66, 'Full': 2.11}
# SUBBAND scopes (tier 2026-07-08): within a subband there is no narrowband
# PSF inflation, so the physics bound collapses to the fit-scatter margin.
# PROVISIONAL 1.32 (=1.1 scatter x1.2 empirical-beam margin) — pending Gregg
# ratification; until then the subband gate uses this value (recorded either way).
SUBBAND_SIZE_BOUND = 1.32
_SUB_BAND = {'18': 'Red', '23': 'Red', '27': 'Red', '32': 'Red', '36': 'Red',
             '41': 'Green', '46': 'Green', '50': 'Green', '55': 'Green',
             '59': 'Green', '64': 'Blue', '69': 'Blue', '73': 'Blue',
             '78': 'Blue', '82': 'Blue'}


def _bandof(scope):
    """Wideband name for a scope ('73MHz' -> 'Blue'; wideband passes through)."""
    return _SUB_BAND.get(scope.replace('MHz', ''), scope)


def _sizebound(scope):
    return SIZE_BOUND[scope] if scope in SIZE_BOUND else SUBBAND_SIZE_BOUND
JUP_RAD = {'Red': 5.0, 'Green': 0.5, 'Blue': 0.5, 'Full': 5.0}
B70_MIN_DEG = 1.5                         # 70 Jy tier retained
ATEAM_VETO_DEG = 5.0                      # A-team gate (Gregg ADOPTED 2026-07-15,
                                          # pilot-measured excess ends ~4 deg;
                                          # handoff §5.15g) — post-mstruct stage
# DYNAMIC GATE v2 (ratified Gregg 2026-07-07): per-image 2-D cliff search over
# (Fc, radius), NO ratio guard. floor(r) = off-source count (assoc_r < 2 Jy);
# target T = min_r floor(r) + FC_MARGIN (deepest achievable anywhere in the
# plane); among all (Fc, r) meeting T, take MINIMUM masked sky area.
# Benchmarked on the 2026-07-07 top-25 review: 18/22 FPs caught, anchors
# source-level safe (B0950 50/56, J0410 45/47).
RADII = (0.25, 0.5, 1.0, 1.5, 2.0)        # search grid; assoc recorded at ALL
FC_GRID = np.arange(2.0, 12.01, 0.25)
FC_MARGIN = 2
DISCARD_FLOOR = 10                        # image discard if min_r floor > this
FINAL_SNR = 7.0                           # final-list reporting threshold

LEDGER_COLS = ['lst', 'date', 'band', 'ra', 'dec', 'pix_x', 'pix_y', 'sign',
               'snr', 'peak_jy', 'local_rms_jy',
               'mstruct_i', 'sstack', 'master_i_jy',
               'dist_b7_beams', 'dist_b70_deg', 'ateam_deg', 'jupiter_deg',
               'bl_sep_beams', 'bl_partner_sig', 'bl_ratio', 'bilobe',
               'gf_ar', 'gf_resid', 'gf_fwhm_beams', 'gaussfit',
               'assoc_m025_jy', 'assoc_m05_jy', 'assoc_m10_jy',
               'assoc_m15_jy', 'assoc_m20_jy',
               'f_img', 'fc_star', 'rb_star', 'floor', 'img_discard',
               'dyn_veto', 'pass_mstruct', 'final7', 'n_epoch_raw',
               'dc2', 'renorm']
IMG_COLS = (['lst', 'band', 'date', 'n_raw5', 'n_pos7_upstream', 'rms_mjy',
             'dil', 'dc2', 'img_discard', 'f_img', 'rb_star', 'fc_star',
             'area_pct', 'target']
            + [f'{k}_{t}' for t in ('025', '05', '10', '15', '20')
               for k in ('floor', 'fc', 'kept7')])

ATEAM = [(n, i['coord']) for n, i in sit.CONFIG['A_TEAM'].items()]


def _append(path, cols, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lk = open(path + '.lock', 'w')
    fcntl.flock(lk, fcntl.LOCK_EX)
    try:
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(cols)
            w.writerows(rows)
    finally:
        fcntl.flock(lk, fcntl.LOCK_UN)
        lk.close()


def _qa_outlier_dates(lst, scope):
    p = f'{coadd_root()}/{lst}_{scope}/residuals_provisional.csv'
    if not os.path.exists(p):
        return set()
    return {r['date'] for r in csv.DictReader(open(p))
            if str(r.get('outlier')).strip().lower() == 'true'}


_CTX = None


def _worker(d):
    """Build one matched residual: load + reconstruct epoch, fit its beam,
    x2 refine + cross-conv + 2nd flux fit + renorm + destripe."""
    master, flows, hdr_ref, cd, bm, base, scope = _CTX
    try:
        sp = f'{base}/subtracted/{d}_I_sub_{scope}.fits'
        ep = _load(sp) + master
        ee, ee2, eepa, _ = fit_empirical_beam(ep)
        be = (ee if np.isfinite(ee) and ee > 0 else bm[0],
              ee2 if np.isfinite(ee2) and ee2 > 0 else bm[1],
              eepa if np.isfinite(eepa) else 0.0)
        ot = get_midpoint_time(fits.getheader(sp), sp)
        flow = np.load(flows[d])       # lazy: x2 flows are large (Blue/Full)
        r, bpe, fdil, dc2 = matched_resid(ep, master, flow, hdr_ref, cd,
                                          be, bm, mfs_args=ot)
        return d, r.astype(np.float32), (be, bpe, fdil, dc2), None
    except Exception as e:
        return d, None, None, str(e)


def _archive_matched_resid(lst, scope, d, resid_d, ot, span, eh, mp, wcs,
                           bp_eff, fdil, dc2, cd,
                           runlabel=('Run_I_DEEP', 'production I deep search')):
    """Archive one matched residual (search frame), standard name + headers.
    Shared by the full search path and the residuals-only build mode
    (2026-08-21: psfmatch variant ruled the canonical I residual)."""
    try:
        ad = f'{RESID_ROOT}/{lst}_{scope}'
        os.makedirs(ad, exist_ok=True)
        rn = prod.build_name(scope, 'I', 'deep', 'Taper', -0.75,
                             proc=['dewarped', 'aligned', 'psfmatch',
                                   'msub', 'ssub'],
                             lst=lst, epoch=ot)
        rp = f'{ad}/{rn}'
        if not os.path.exists(rp):
            fits.writeto(rp, resid_d, wcs.to_header(), overwrite=True)
            prod.write_standard_header(
                rp, scope=scope, stokes='I',
                start=ot, end=eh.get('DATE-END') if eh else None,
                inttime_s=span, time_basis='phase2',
                beam=(bp_eff * cd, bp_eff * cd, 0.0),
                aligned=True, alignref=os.path.basename(mp),
                msub=True, msubfile=mp, ssub=True, dewarped=True,
                validate=True,
                extra={'RUNLABEL': runlabel,
                       'EPOCH_ID': (f'{lst}_{d}', 'LST_date epoch'),
                       'PSFMATCH': (True, 'symmetric cross-conv subtraction'),
                       'RENORM': (round(1.0 / fdil, 4),
                                  'per-pair point-src dilution corr'),
                       'SCALE2B': (round(dc2, 4),
                                   '2nd-pass matched-pair flux DC')})
    except Exception as e:
        print(f'  [resid archive] {lst} {scope} {d}: {e}', flush=True)


def _disk(R):
    Ri = max(int(np.ceil(R)), 1)
    yy, xx = np.mgrid[-Ri:Ri + 1, -Ri:Ri + 1]
    return Ri, np.hypot(yy, xx) <= R


def _assoc(master, x, y, Ri, disk):
    h, w = master.shape
    if not (Ri <= x < w - Ri and Ri <= y < h - Ri):
        return np.inf
    box = np.where(disk, master[y - Ri:y + Ri + 1, x - Ri:x + Ri + 1], np.nan)
    return float(np.nanmax(box))


def run_cell(lst, scope, residuals_only=False, included_dates=None):
    t0 = time.time()
    base = f'{coadd_root()}/{lst}_{scope}'
    cw = f'{FLOW_ROOT}/{lst}_{scope}'
    mp = f'{base}/coadd_final/I_deep_Taper_R075_{scope}.fits'
    if not os.path.exists(mp):
        print(f'  {lst} {scope}: no R075 I master — skip'); return 0
    master = _load(mp)
    if (master != 0).sum() < 0.1 * master.size:
        print(f'  {lst} {scope}: R075 master corrupt — skip'); return 0
    hdr0 = fits.getheader(mp); cd = abs(hdr0['CDELT1'])
    em, en, mpa, _ = fit_empirical_beam(master)
    bmaj = em if np.isfinite(em) and em > 0 else 7.0
    bm = (bmaj, en if np.isfinite(en) and en > 0 else bmaj,
          mpa if np.isfinite(mpa) else 0.0)
    ar_max = max(2.0, 1.5 * bm[0] / max(bm[1], 1e-3))

    dates = [re.search(r'(\d{4}-\d\d-\d\d)_I_sub_', os.path.basename(c)).group(1)
             for c in sorted(glob.glob(f'{base}/subtracted/*_I_sub_{scope}.fits'))]
    if included_dates:
        dates = [d for d in dates if d in included_dates]
    flows = {d: f'{cw}/ep_{d}_up2_flow.npy' for d in dates
             if os.path.exists(f'{cw}/ep_{d}_up2_flow.npy')}
    noflow = [d for d in dates if d not in flows]
    if noflow:
        print(f'  {lst} {scope}: {len(noflow)} epoch(s) missing x2 flow '
              f'(not searched): {noflow}')
    dates = [d for d in dates if d in flows]
    excl = load_exclusions()
    dates = [d for d in dates if not is_excluded(lst, scope, d, excl)]
    qa = _qa_outlier_dates(lst, scope)
    dates = [d for d in dates if d not in qa]
    if len(dates) < MIN_EPOCHS and not residuals_only:
        print(f'  {lst} {scope}: only {len(dates)} usable epochs — skip'); return 0
    if not dates:
        print(f'  {lst} {scope}: no usable epochs — skip'); return 0

    hdr_ref = _det_header(hdr0, scope, bm[0], cd, None)
    global _CTX
    _CTX = (master, flows, hdr_ref, cd, bm, base, scope)
    resid, meta = {}, {}
    import multiprocessing as mp_
    with mp_.get_context('fork').Pool(min(NPROC, len(dates))) as pool:
        for d, r, m, err in pool.imap_unordered(_worker, dates):
            if err:
                print(f'  [{lst} {scope}] {d}: matched chain FAILED: {err}',
                      flush=True)
                continue
            resid[d] = r; meta[d] = m
            print(f'  [{lst} {scope}] {d}: dil {m[2]:.3f} DC2 {m[3]:.4f}',
                  flush=True)
    _CTX = None
    dates = [d for d in dates if d in resid]
    if len(dates) < MIN_EPOCHS and not residuals_only:
        print(f'  {lst} {scope}: only {len(dates)} residuals — skip'); return 0
    if not dates:
        print(f'  {lst} {scope}: no residuals — skip'); return 0

    # mstruct-I on the MATCHED master (median epoch kernel), detection beam =
    # matched effective beam — identical construction to the ratified ledger
    med_be = (float(np.median([meta[d][0][0] for d in dates])),
              float(np.median([meta[d][0][1] for d in dates])),
              float(np.median([meta[d][0][2] for d in dates])))
    beam = float(np.hypot(bm[0], med_be[0]))
    ms = None if residuals_only else \
        _mstruct_i(convolve_fft(master, gk(*med_be), allow_huge=True,
                                normalize_kernel=True), mp, bmaj)

    # sstack: INFO-ONLY column (§5.13 RECORD-NEVER-GATE)
    wtmp = f'{OUT}/tmp_{lst}_{scope}'
    os.makedirs(wtmp, exist_ok=True)
    sstack = _sigma_stack(resid, wtmp, hdr0, 'Iprod') \
        if (len(resid) >= MIN_EPOCHS and not residuals_only) else None

    # master local maxima >= 3 Jy for f_img (recorded scintillation level)
    validm = np.isfinite(master) & (master != 0)
    if not residuals_only:
        mf = maximum_filter(np.where(validm, master, 0), size=int(2 * beam) + 1)
        spy, spx = np.where((master == mf) & (master >= 3.0) & validm)

    h, w = master.shape
    if not residuals_only:
        disks = {rb: _disk(rb * beam) for rb in RADII}
        # per-radius max-filtered master: masked-area computation for the 2-D
        # (Fc, radius) min-area choice (one pass per cell, reused by every epoch)
        mfmaps = {rb: maximum_filter(np.where(validm, master, 0),
                                     footprint=disks[rb][1]) for rb in RADII}
    nvalid = max(int(validm.sum()), 1)
    B7 = B70 = None
    led_rows, img_rows, nfinal = [], [], 0
    for d in dates:
        rI = np.nan_to_num(resid[d].astype(np.float64), nan=0.0)
        be, bp_eff, fdil, dc2 = meta[d]
        # per-epoch time block (§5.12 fix): epoch DATE-OBS + single-block span
        try:
            eh = fits.getheader(f'{base}/subtracted/{d}_I_sub_{scope}.fits')
            ot = eh.get('DATE-OBS'); span = float(eh.get('INTTIME') or 3591.0)
        except Exception:
            eh, ot, span = None, None, 3591.0
        hdr = _det_header(hdr0, scope, beam, cd, ot)
        hdr['INTTIME'] = span; hdr['DURATION'] = span
        wcs = WCS(hdr).celestial
        if residuals_only:
            # build + archive the canonical psfmatch residual, no search
            # (ruling 2026-08-21: psfmatch variant = the canonical I residual)
            _archive_matched_resid(lst, scope, d, resid[d], ot, span, eh, mp,
                                   wcs, bp_eff, fdil, dc2, cd,
                                   runlabel=('PSFM_BF_20260821',
                                             'psfmatch residual backfill'))
            nfinal += 1
            continue
        try:
            horizon, ateam, badwcs, obs_time, _el = sit.make_masks(
                hdr, rI.shape, wcs)
            excl_m = horizon | ateam | badwcs
            rms0 = sit.get_uniform_rms(rI, horizon)
            if not np.isfinite(rms0) or rms0 <= 0:
                print(f'  {lst} {scope} {d}: rms={rms0} — mask/geometry problem',
                      flush=True)
                continue
            cands = []
            for sgn, img in ((1, rI), (-1, -rI)):
                cs = sit.extract_positive_peaks(
                    img.astype(np.float32), hdr, horizon, ateam, badwcs,
                    wcs, obs_time, rms0, beam)
                sit.assign_local_snr(cs, img, excl_m, beam, 5.0)  # write floor
                sit.merge_nearby(cs, beam, wcs)
                for c in cs:
                    if c['status'] == 'PASS':
                        c['sgn'] = sgn
                        cands.append(c)
        except Exception as e:
            print(f'  {lst} {scope} {d}: detection FAILED: {e}', flush=True)
            continue
        if B7 is None:
            b7 = sit.find_bright_sources(master, horizon, wcs, beam,
                                         sit.CONFIG['BRIGHT_MASK_FLUX_JY'])
            B7 = (np.array([[s['pix_x'], s['pix_y']] for s in b7], float)
                  if b7 else np.empty((0, 2)))
            B70 = (np.array([[s['pix_x'], s['pix_y']] for s in b7
                             if s['flux'] >= sit.CONFIG['BRIGHT_EXCL_FLUX_JY']],
                            float) if b7 else np.empty((0, 2)))

        # archive the matched residual (search frame), standard name + headers
        _archive_matched_resid(lst, scope, d, resid[d], ot, span, eh, mp,
                               wcs, bp_eff, fdil, dc2, cd)

        # Jupiter ephemeris position (per-epoch; SkyCoord(jb.ra,jb.dec) — the
        # separation(get_body) barycentric-parallax gotcha)
        jup = None
        if obs_time is not None:
            try:
                jb = get_body('jupiter', obs_time, OVRO_LOC)
                jup = SkyCoord(jb.ra, jb.dec)
            except Exception:
                jup = None

        # bilobe global rms (production convention: central-box MAD)
        cb = rI[h // 3:2 * h // 3, w // 3:2 * w // 3]
        cb = cb[np.isfinite(cb)]
        grms = (1.4826 * float(np.median(np.abs(cb - np.median(cb))))
                if cb.size > 100 else 0.0)

        # ---- per-candidate measurables + upstream gates ----
        rec = []
        for c in cands:
            x, y = int(c['pix_x']), int(c['pix_y'])
            sgn = c['sgn']
            snr = abs(float(c.get('local_snr', c.get('snr', 0))))
            lr = float(c.get('local_rms', np.nan))
            pk = abs(float(c.get('flux', 0.0)))
            simg = rI if sgn > 0 else -rI
            sky = wcs.pixel_to_world(x, y)
            d7 = (float(np.min(np.hypot(B7[:, 0] - x, B7[:, 1] - y))) / beam
                  if len(B7) else np.inf)
            d70 = (float(np.min(np.hypot(B70[:, 0] - x, B70[:, 1] - y))) * cd
                   if len(B70) else np.inf)
            adeg = min(float(sky.separation(co).deg) for _, co in ATEAM)
            jdeg = float(sky.separation(jup).deg) if jup is not None else np.nan
            bsep, bsig, brat = bilobe_measurables(simg, x, y, beam, grms)
            bl = int(grms > 0 and
                     pv.bilobe_ghost_reason(simg, x, y, beam, grms, cd)
                     != 'keep')
            gar = gres = gfw = ''
            gf = 0
            if snr >= 6.0:
                g = pv._gfit(simg, x, y, beam)
                if g:
                    gar = round(g['ar'], 2); gres = round(g['resid'], 3)
                    gfw = round(g['fwhm'] / beam, 2)
                    gf = int(g['ar'] > ar_max or g['resid'] > pv.RESID_MAX)
            assoc = {rb: _assoc(master, x, y, *disks[rb]) for rb in RADII}
            pass_ms = int(snr >= SIGMA_MSTRUCT * max(float(ms[y, x]), 1.0))
            # upstream gate verdict (everything before the dynamic gate)
            oversz = (gfw != '' and float(gfw) > _sizebound(scope))
            upstream = (pass_ms
                        and not bl
                        and not (snr > GF_SNR_GATE and gf)
                        and not oversz
                        and not (np.isfinite(jdeg) and jdeg < JUP_RAD[_bandof(scope)])
                        and not (np.isfinite(adeg) and adeg < ATEAM_VETO_DEG)
                        and d70 > B70_MIN_DEG)
            rec.append(dict(x=x, y=y, sgn=sgn, snr=snr, lr=lr, pk=pk,
                            ra=float(sky.ra.deg), dec=float(sky.dec.deg),
                            d7=d7, d70=d70, adeg=adeg, jdeg=jdeg,
                            bsep=bsep, bsig=bsig, brat=brat, bl=bl,
                            gar=gar, gres=gres, gfw=gfw, gf=gf,
                            assoc=assoc, msv=float(ms[y, x]),
                            pass_ms=pass_ms, upstream=upstream))

        # ---- DYNAMIC GATE v2: per-image 2-D (Fc, radius) min-area cliff ----
        up7 = [r for r in rec if r['upstream'] and r['sgn'] > 0
               and r['snr'] >= FINAL_SNR]
        # f_img at 0.5 bm: recorded scintillation level (informational)
        Ri, disk = disks[0.5]
        fr = []
        for sx, sy in zip(spx, spy):
            if Ri <= sx < w - Ri and Ri <= sy < h - Ri:
                box = np.where(disk,
                               rI[sy - Ri:sy + Ri + 1, sx - Ri:sx + Ri + 1],
                               np.nan)
                fr.append(np.nanmax(np.abs(box)) / master[sy, sx])
        f_img = float(np.percentile(fr, 95)) if fr else np.nan
        scan = {}
        for rb in RADII:
            A = np.array([r['assoc'][rb] for r in up7])
            scan[rb] = dict(A=A, floor=int((A < 2.0).sum()) if len(A) else 0,
                            fc='', kept7='', area='')
        min_floor = min(s['floor'] for s in scan.values()) if up7 else 0
        T = min_floor + FC_MARGIN
        best = None
        for rb in RADII:
            A = scan[rb]['A']
            fc_ok = None
            for fc in FC_GRID[::-1]:
                if (int((A < fc).sum()) if len(A) else 0) <= T:
                    fc_ok = float(fc)
                    break
            if fc_ok is None:
                continue
            area = float(((mfmaps[rb] >= fc_ok) & validm).sum()) / nvalid
            scan[rb].update(fc=fc_ok, area=round(100 * area, 2),
                            kept7=int((A < fc_ok).sum()) if len(A) else 0)
            if best is None or area < best[0]:
                best = (area, rb, fc_ok)
        area_star, rb_star, fc_star = best if best is not None \
            else (1.0, RADII[-1], float(FC_GRID[0]))
        discard = int(min_floor > DISCARD_FLOOR)
        if discard:
            print(f'  {lst} {scope} {d}: IMAGE DISCARD (dyn-gate min floor '
                  f'{min_floor} > {DISCARD_FLOOR})', flush=True)

        # row-level verdicts at the chosen (Fc*, r*) — NO ratio guard
        up7_ids = {id(r) for r in up7}
        n_final = 0
        for r in rec:
            if id(r) in up7_ids:
                dyn = int(not (r['assoc'][rb_star] < fc_star))
            else:
                dyn = ''
            final = int(dyn == 0 and id(r) in up7_ids and not discard)
            n_final += final
            led_rows.append([
                lst, d, scope, round(r['ra'], 5), round(r['dec'], 5),
                r['x'], r['y'], '+' if r['sgn'] > 0 else '-',
                round(r['snr'], 2), round(r['sgn'] * r['pk'], 4),
                round(r['lr'], 4) if np.isfinite(r['lr']) else '',
                round(r['msv'], 2),
                round(float(sstack[r['y'], r['x']]), 2)
                if sstack is not None else '',
                round(float(master[r['y'], r['x']]), 3),
                round(r['d7'], 2) if np.isfinite(r['d7']) else '',
                round(r['d70'], 3) if np.isfinite(r['d70']) else '',
                round(r['adeg'], 2),
                round(r['jdeg'], 2) if np.isfinite(r['jdeg']) else '',
                r['bsep'], r['bsig'], r['brat'], r['bl'],
                r['gar'], r['gres'], r['gfw'], r['gf'],
                *[('' if not np.isfinite(r['assoc'][rb])
                   else round(r['assoc'][rb], 3)) for rb in RADII],
                round(f_img, 4) if np.isfinite(f_img) else '',
                fc_star, rb_star, scan[rb_star]['floor'], discard, dyn,
                r['pass_ms'], final, len(cands),
                round(dc2, 4), round(1.0 / fdil, 4)])
        nfinal += n_final

        cenr = rI[h // 3:2 * h // 3, w // 3:2 * w // 3]
        cenr = cenr[np.isfinite(cenr) & (cenr != 0)]
        rms_c = 1.4826 * float(np.median(np.abs(cenr - np.median(cenr)))) \
            if cenr.size else np.nan
        img_rows.append([lst, scope, d, len(cands), len(up7),
                         round(1e3 * rms_c, 1), round(fdil, 4), round(dc2, 4),
                         discard,
                         round(f_img, 4) if np.isfinite(f_img) else '',
                         rb_star, fc_star, round(100 * area_star, 2), T]
                        + [v for rb in RADII for v in
                           (scan[rb]['floor'], scan[rb]['fc'],
                            scan[rb]['kept7'])])
        print(f'  {lst} {scope} {d}: {len(cands)} raw±5σ | {len(up7)} pos≥7σ '
              f'upstream | {n_final} FINAL', flush=True)
        resid[d] = None    # free as we go

    _append(LEDGER, LEDGER_COLS, led_rows)
    _append(IMGCSV, IMG_COLS, img_rows)
    print(f'  {lst} {scope}: {len(dates)} epochs -> {len(led_rows)} ledger rows, '
          f'{nfinal} final candidates ({(time.time() - t0) / 60:.1f} min)',
          flush=True)
    return len(led_rows)


if __name__ == '__main__':
    run_cell(_ARGV[1], _ARGV[2])
