#!/usr/bin/env python3
"""
phase3_veto.py — the validated morphology/position veto, shared by the deep
(stage_search) and 10-min detection tiers so they apply identical logic.

This is the veto proven in the blind-injection S-curve comparison: it beats a
hard bright-source mask (deeper C50, higher asymptote — the mask permanently
blinds ~40% of source-dense sky). Gates, in order:
  >30 Jy bright mask (flux-scaled radius)
  Gaussian-fit shape (ONLY above SNR_GATE): axis-ratio / fit-resid / FWHM
  on-source proximity (1 beam to a >2 Jy source)
Failed Gaussian fit => KEEP (do not veto a real source as a ring — the bug we
fixed). ar cut is beam-relative: max(2.0, 1.5*beam_axis_ratio).

NO ephemeris gate: solar-system bodies are FLAGGED IN PLACE in the catalog
(a `body` label column), never dropped — Gregg's ruling. Jupiter DAM etc. stay
in the master candidate list.
"""
import numpy as np
from scipy.optimize import curve_fit

RESID_MAX, FWHM_HI, FWHM_LO, SNR_GATE = 0.7, 2.0, 0.45, 8.0
ONSRC_JY, MASK_JY = 2.0, 30.0
MASK_COEF_DEG = 0.2          # >30 Jy sidelobe mask radius = 0.2deg * sqrt(flux/30)
# A-team avoidance (Gregg 2026-06-26): the brightest, artifact-dominating sources
# (Cyg A, Cas A, Tau A, Vir A, ...) get a HARD 2-deg avoidance floor on top of the
# sqrt-flux sidelobe mask. Leaves the validated 30-300 Jy regime unchanged.
ATEAM_JY = 300.0             # A-team-class flux threshold [Jy]
ATEAM_RADIUS_DEG = 2.0       # hard avoidance radius for A-team-class sources [deg]


def bright_source_list(master, thresh_jy=ONSRC_JY, maxn=400):
    """Greedy peak list from the deep master for the veto's proximity tests."""
    mw = np.where(np.isfinite(master), master, -np.inf).copy()
    xy, f = [], []
    for _ in range(maxn):
        iy, ix = np.unravel_index(np.argmax(mw), mw.shape)
        if mw[iy, ix] < thresh_jy:
            break
        xy.append((ix, iy)); f.append(float(mw[iy, ix]))
        mw[max(0, iy - 6):iy + 6, max(0, ix - 6):ix + 6] = -np.inf
    return np.array(xy, float), np.array(f)


def _gfit(diff, x, y, beam, r=8):
    st = diff[y - r:y + r + 1, x - r:x + r + 1]
    if st.shape != (2 * r + 1, 2 * r + 1) or not np.isfinite(st).all():
        return None
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    def g(c, a, x0, y0, sx, sy, th, o):
        X, Y = c
        xr = (X - x0) * np.cos(th) + (Y - y0) * np.sin(th)
        yr = -(X - x0) * np.sin(th) + (Y - y0) * np.cos(th)
        return (a * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2)) + o).ravel()
    try:
        p, _ = curve_fit(g, (xx, yy), st.ravel(),
                         p0=[np.nanmax(st), 0, 0, beam / 2.3, beam / 2.3, 0, 0],
                         maxfev=8000)
    except Exception:
        return None          # failed fit => KEEP (do not veto as ring)
    mj, mn = abs(p[3]) * 2.355, abs(p[4]) * 2.355
    model = g((xx, yy), *p).reshape(st.shape)
    return dict(ar=max(mj, mn) / max(min(mj, mn), 0.1),
                resid=float(np.nansum((st - model) ** 2) / max(np.nansum(st ** 2), 1e-9)),
                fwhm=max(mj, mn))


# ── SIMPLE bilobe veto (Gregg 2026-06-25) ──────────────────────────────────
# A +/- dipole is flagged iff the candidate has an OPPOSITE-sign partner within
# SEP_HI beam where (a) the LESSER lobe is > LESSER_SIG sigma and (b) the lesser
# lobe is > LESSER_RATIO of the brighter lobe. No population gate, no PA gate.
# A real single-sign source has only a weak opposite sidelobe (low ratio) -> kept;
# genuinely bright real sources are removed by the bright-source flag, which runs
# BEFORE this. (Replaces the convoluted Part1/2/3 dipole-sea design.)
SEP_HI = 1.5           # opposite-sign partner must be within this many beams
LESSER_SIG = 5.0       # the fainter (lesser) lobe must exceed this many sigma
LESSER_RATIO = 0.25    # lesser lobe must be > this fraction of the brighter lobe
NEG_SIG = 4.0          # significance floor used by dipole_lib.find_true_dipoles
# ── PA strengthening (Gregg 2026-06-25): a nearby dipole sharing this candidate's
# +/- axis PA STRENGTHENS the veto but can NEVER remove it. Used ONLY to veto an
# ASYMMETRIC dipole (lesser>LESSER_SIG but <=LESSER_RATIO x brighter) that the
# amplitude rule alone would keep; a symmetric dipole is already vetoed regardless
# of PA. So PA can only ADD a veto, never spare. ──
PA_RAD_BEAM = 15.0     # search this many beams around the candidate for a same-PA dipole
PA_TOL = 20.0          # PA match tolerance [deg] (sky-frame, 180-wrapped)


def bilobe_ghost_reason(diff, x, y, beam, rms, cd, dipmap=None, wcs=None):
    """SIMPLE bilobe veto (Gregg 2026-06-25). A candidate is a bilobe iff it has an
    OPPOSITE-sign partner within SEP_HI beam such that:
      (1) the LESSER (fainter) lobe is > LESSER_SIG sigma, AND
      (2) the lesser lobe is > LESSER_RATIO of the brighter lobe.
    A real single-sign source has only a weak opposite sidelobe (ratio << 0.25) and
    is kept; bright real sources are removed by the bright-source flag, which runs
    BEFORE this. No population gate, no PA gate. (dipmap/wcs accepted but unused —
    kept for call-site compatibility.) Returns 'bilobe_ghost' or 'keep'."""
    if rms <= 0 or not (0 <= y < diff.shape[0] and 0 <= x < diff.shape[1]):
        return 'keep'
    w = int(round(SEP_HI * beam)); by0, bx0 = max(0, y - w), max(0, x - w)
    box = diff[by0:y + w + 1, bx0:x + w + 1]
    if box.size == 0 or not np.isfinite(box).any():
        return 'keep'
    # candidate sign + amplitude from a TIGHT central patch (~0.3 beam)
    cw = max(1, int(round(0.3 * beam)))
    cp = diff[max(0, y - cw):y + cw + 1, max(0, x - cw):x + cw + 1]
    cp = cp[np.isfinite(cp)]
    if cp.size == 0:
        return 'keep'
    cmax, cmin = float(np.nanmax(cp)), float(np.nanmin(cp))
    cand_pos = cmax >= -cmin
    cand_amp = (cmax if cand_pos else -cmin) / rms           # sigma, >0
    # OPPOSITE-sign partner = the extreme opposite pixel within SEP_HI beam
    if cand_pos:
        pj = np.unravel_index(np.nanargmin(box), box.shape); pval = float(box[pj])
        if pval >= 0:
            return 'keep'                                    # no opposite-sign partner
    else:
        pj = np.unravel_index(np.nanargmax(box), box.shape); pval = float(box[pj])
        if pval <= 0:
            return 'keep'
    py, px = by0 + pj[0], bx0 + pj[1]
    if np.hypot(py - y, px - x) / beam > SEP_HI:             # within 1.5 beams
        return 'keep'
    part_amp = abs(pval) / rms                               # sigma, >0
    lesser, brighter = min(cand_amp, part_amp), max(cand_amp, part_amp)
    if lesser <= LESSER_SIG:                                 # lesser lobe must be > 5 sigma
        return 'keep'
    if lesser > LESSER_RATIO * brighter:                     # lesser > 0.25 x brighter
        return 'bilobe_ghost'                                # symmetric dipole -> definite
    # ── PA strengthening (non-removing). The pair is ASYMMETRIC (lesser>5sig but
    # <=0.25 x brighter): keep UNLESS a nearby dipole shares its +/- axis PA, which
    # corroborates an iono/X-Y bilobe field. This only ADDS a veto; it can never
    # spare a candidate (symmetric dipoles already vetoed above, irrespective of PA).
    if dipmap is None or len(dipmap) == 0 or wcs is None:
        return 'keep'
    from exopipe.phase3.dipole_lib import lobe_centroid as _cen, sky_pa as _skypa, pa_diff as _padiff
    if cand_pos:
        cxc, cyc = _cen(diff, x, y, +1, beam); cxp, cyp = _cen(diff, px, py, -1, beam)
        cand_pa = _skypa(wcs, cxc, cyc, cxp, cyp)
    else:
        cxc, cyc = _cen(diff, x, y, -1, beam); cxp, cyp = _cen(diff, px, py, +1, beam)
        cand_pa = _skypa(wcs, cxp, cyp, cxc, cyc)
    dd = np.hypot(dipmap[:, 0] - y, dipmap[:, 1] - x)
    near = dipmap[(dd > 0.5 * beam) & (dd < PA_RAD_BEAM * beam)]
    for d in near:
        if _padiff(cand_pa, float(d[2])) < PA_TOL:
            return 'bilobe_ghost'                            # PA-corroborated asymmetric bilobe
    return 'keep'


def veto_reason(c, diff, msx, msf, beam, cd, ar_max, rms=0.0, dipmap=None, wcs=None):
    """Return 'keep' or the veto reason for one candidate dict."""
    x, y = int(c['pix_x']), int(c['pix_y'])
    if len(msx):
        d = np.hypot(msx[:, 0] - x, msx[:, 1] - y)
        for j in np.where(msf > MASK_JY)[0]:
            rdeg = MASK_COEF_DEG * (msf[j] / MASK_JY) ** 0.5
            if msf[j] >= ATEAM_JY:                     # A-team: hard 2-deg floor
                rdeg = max(rdeg, ATEAM_RADIUS_DEG)
            if d[j] < max(2 * beam, rdeg / cd):
                return 'bright_mask'
    if rms > 0 and bilobe_ghost_reason(diff, x, y, beam, rms, cd, dipmap=dipmap, wcs=wcs) != 'keep':
        return 'bilobe_ghost'
    if abs(c.get('local_snr', c.get('snr', 0))) > SNR_GATE:
        gf = _gfit(diff, x, y, beam)
        # NO SIZE restriction (Gregg 2026-06-24): the target V signal (ECM) is NARROWBAND,
        # so in a WIDEBAND image it appears with the PSF of its EMISSION frequency, not the
        # wideband synthesized beam — a low-band-edge narrowband source is genuinely larger
        # than the nominal beam. A FWHM cut (FWHM_HI/FWHM_LO) would veto exactly those real
        # narrowband sources. Keep only fit-quality (resid = ring/multi-peak) and
        # elongation (ar); a narrowband source is still round + fits a (larger) Gaussian.
        if gf and (gf['ar'] > ar_max or gf['resid'] > RESID_MAX):
            return 'gaussfit'
    # on_source veto REMOVED (Gregg 2026-06-21): the blanket "within 1 beam of a
    # >=2 Jy I source" cut is too aggressive at low nu (large beam, dense >2 Jy
    # sky) and kills real co-located flares (UX Ari / FI Cnc emit in I too).
    # On-source I->V leakage is handled by run_identification_v's V/I<5%
    # (I-counterpart > 20x|V|) rule, which keeps high-polarization real flares.
    return 'keep'


def apply_veto(cands, diff, master, hdr, beam, cd, wcs, obs_time, ar_max=None):
    """Filter candidates through the validated veto. Returns (kept, vetoed)
    where each vetoed candidate carries c['veto_reason']. `wcs`/`obs_time` are
    kept in the signature for a stable shared interface (deep + 10-min) but the
    veto no longer drops on ephemeris — solar-system bodies are labelled in the
    catalog instead."""
    from exopipe.phase3.identify import fit_empirical_beam
    if ar_max is None:
        em, en, _ep, _nm = fit_empirical_beam(master)
        ar_max = max(2.0, 1.5 * em / max(en, 1e-3))
    msx, msf = bright_source_list(master)
    # central-field MAD thermal noise (peak/rms basis; noise is beam-correlated)
    h, w = diff.shape; cd_box = diff[h // 3:2 * h // 3, w // 3:2 * w // 3]
    cd_box = cd_box[np.isfinite(cd_box)]
    rms = float(1.4826 * np.median(np.abs(cd_box - np.median(cd_box)))) if cd_box.size > 100 else 0.0
    # dipole field for the PA-strengthening arm (sky-frame PA needs wcs). Computed ONCE.
    dipmap = None
    if rms > 0:
        try:
            from exopipe.phase3.dipole_lib import find_true_dipoles
            dipmap = find_true_dipoles(diff, beam, sig=NEG_SIG, wcs=wcs)
        except Exception:
            dipmap = None
    kept, vetoed = [], []
    for c in cands:
        r = veto_reason(c, diff, msx, msf, beam, cd, ar_max, rms=rms, dipmap=dipmap, wcs=wcs)
        if r == 'keep':
            kept.append(c)
        else:
            c['veto_reason'] = r
            vetoed.append(c)
    return kept, vetoed
