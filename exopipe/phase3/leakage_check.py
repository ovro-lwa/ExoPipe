#!/usr/bin/env python3
"""
leakage_check.py — per-transient LOCAL-LEAKAGE / BILOBE consistency check
(Gregg 2026-06-20).

Physics: instrumental leakage and x-y-phase / ionospheric bilobes are NOT random
per source — in any region of the field ALL the Stokes-I sources leak with about
the SAME fractional leakage and SAME polarity (the leakage varies slowly, worst at
the edges, opposite edges opposite sign). A bilobe likewise repeats: every nearby
I source shows the same +/- separation and the same position angle. So a V
"transient" that merely reproduces the LOCAL I-source leakage signature is an
artifact; a real V source departs from it (no I counterpart, or V/I far above the
local leakage fraction).

TERMINOLOGY (Gregg 2026-06-20 — easily confused):
  * "leakage MAP"   = search_v_transients_v2 sidelobe+bilobe subtraction; removes
                      bilobes from x-y PHASE-RESIDUAL leakage (mostly STATIC).
  * "leakage SCREEN"= fit_leakage_screen (phase3_pipeline); a smooth low-order poly
                      removed during per-epoch residual creation. Different thing.
  * Static x-y bilobes are MOSTLY REMOVED already by per-epoch MASTER SUBTRACTION
    (they're in the master too). What master subtraction does NOT remove is the
    IONOSPHERIC bilobe — it is time-variable, so it is absent/smeared in the master
    and survives into the per-epoch residual. THOSE are what this check must catch.

So: RUN THIS ON THE PER-EPOCH RESIDUAL (epoch - master), not on the deep master.
The bilobe arm targets the surviving (ionospheric) bilobes; the on-source arm
targets any residual smooth leakage. It is a cheap per-candidate check: look only
at bright I sources within a few degrees, measure their (per-epoch) leakage
signature, and ask whether the candidate merely reproduces it.

Returns a verdict + the supporting numbers (never a bare boolean) so the human can
adjudicate — see [[feedback_label_speculation]].

THRESHOLDS are data-driven from leakage_freq.py (the smooth screen is ~0.2-1%,
worse at low freq); tune the constants below from leakage_freq.csv before trusting
the veto. Until tuned, this is a FLAG (report), not an auto-reject.
"""
import numpy as np

# --- tunable (set from leakage_freq.csv per-band leakage distribution) ---
NEIGH_RADIUS_DEG = 15.0    # local neighborhood; leakage screen is field-scale and
                           # bright I sources are sparse (~0.016/deg^2) so 6deg gave <4
MIN_NEIGH       = 4        # need >= this many bright I sources to define a signature
COLOC_BEAMS     = 1.5      # an I source this close == candidate sits on a source
LEAK_FRAC_BAND  = 2.5      # cand |V/I| within this x the local median -> consistent w/ leakage
REAL_FRAC_MIN   = 0.03     # |V/I| above this (3%) is too polarised to be leakage -> spare it
BILOBE_PA_TOL   = 25.0     # deg: candidate PA vs local bilobe PA
BILOBE_SEP_TOL  = 0.40     # fractional tolerance on the +/- separation
BILOBE_AMP_LO   = 0.5      # a TRUE bilobe needs a comparable opposite-sign partner:
                           # |partner| >= this x |candidate| (else it's a single real
                           # lobe, e.g. Jupiter's V — do NOT veto). Matches phase3_veto.
MIN_I_FLUX      = 5.0      # Jy: bright enough I source for a clean V/I


def _signed_peak(data, y, x, half=1):
    h, w = data.shape
    y1, y2 = max(0, y-half), min(h, y+half+1); x1, x2 = max(0, x-half), min(w, x+half+1)
    return float(np.nanmean(data[y1:y2, x1:x2]))


def local_leakage_signature(cy, cx, v_data, i_sources, beam_pix, cd, v_rms):
    """Measure the leakage signature from bright I sources near (cy,cx).
    Returns dict(on_frac_med, on_frac_sign, n, bilobe_pa, bilobe_sep_pix,
    lobe_frac_med) or None if too few neighbours."""
    rad_pix = NEIGH_RADIUS_DEG / cd
    on = []                      # signed on-source V/I
    pa_list, sep_list, lobe_frac = [], [], []
    sr = int(4*beam_pix); h, w = v_data.shape
    for s in i_sources:
        sy, sx = s['pix_y'], s['pix_x']
        if np.hypot(sy-cy, sx-cx) > rad_pix:
            continue
        iflux = s.get('app_flux', s.get('flux', 0.0))
        if iflux < MIN_I_FLUX:
            continue
        # on-source leakage (signed)
        on.append(_signed_peak(v_data, sy, sx)/iflux)
        # bilobe: brightest off-core |V| peak within 0.5-4 beam
        y1, y2 = max(0, sy-sr), min(h, sy+sr+1); x1, x2 = max(0, sx-sr), min(w, sx+sr+1)
        vs = v_data[y1:y2, x1:x2]; yy, xx = np.indices(vs.shape)
        dist = np.hypot(xx-(sx-x1), yy-(sy-y1))
        m = (dist < 0.5*beam_pix) | (dist > sr)
        va = np.where(m, 0.0, np.abs(vs))
        if va.max() > 4*v_rms:
            iy, ix = np.unravel_index(np.argmax(va), va.shape)
            dy, dx = (y1+iy)-sy, (x1+ix)-sx
            pa_list.append(np.degrees(np.arctan2(dx, dy)) % 180)   # axis (mod 180)
            sep_list.append(np.hypot(dy, dx))
            lobe_frac.append(abs(vs[iy, ix])/iflux)
    if len(on) < MIN_NEIGH:
        return None
    on = np.array(on)
    sig = dict(n=len(on), on_frac_med=float(np.median(np.abs(on))),
               on_frac_sign=float(np.sign(np.median(on))),
               on_frac_scatter=float(np.median(np.abs(on-np.median(on)))*1.4826))
    if len(pa_list) >= MIN_NEIGH:
        # circular median PA on the doubled angle (axis, period 180)
        a = np.radians(np.array(pa_list)*2)
        pa = (np.degrees(np.arctan2(np.median(np.sin(a)), np.median(np.cos(a))))/2) % 180
        sig.update(bilobe_pa=float(pa), bilobe_sep_pix=float(np.median(sep_list)),
                   lobe_frac_med=float(np.median(lobe_frac)), n_bilobe=len(pa_list))
    return sig


def transient_leakage_verdict(cy, cx, cand_peak, v_data, i_data, i_sources,
                              beam_pix, cd, v_rms):
    """Adjudicate one V candidate against the LOCAL I-source leakage signature.
    Returns (verdict, reason, diag). verdict in {'real','leakage','bilobe','unknown'}."""
    sig = local_leakage_signature(cy, cx, v_data, i_sources, beam_pix, cd, v_rms)
    diag = dict(sig=sig)
    if sig is None:
        return 'unknown', 'too few local I sources for a leakage signature', diag
    # nearest bright I source to the candidate
    best, bd = None, np.inf
    for s in i_sources:
        d = np.hypot(s['pix_y']-cy, s['pix_x']-cx)
        if d < bd and s.get('app_flux', s.get('flux', 0.0)) >= MIN_I_FLUX:
            bd, best = d, s
    diag['nearest_i_beams'] = round(bd/beam_pix, 2) if np.isfinite(bd) else None

    # (1) ON-SOURCE LEAKAGE: candidate sits on a bright I source and its V/I is
    #     consistent (sign + magnitude) with the LOCAL leakage fraction.
    if best is not None and bd <= COLOC_BEAMS*beam_pix:
        iflux = best.get('app_flux', best.get('flux', 0.0))
        cfrac = cand_peak/iflux if iflux > 0 else np.inf
        diag.update(cand_frac=round(float(cfrac), 4),
                    local_on_frac=round(sig['on_frac_med'], 4))
        if (abs(cfrac) < REAL_FRAC_MIN and np.sign(cfrac) == sig['on_frac_sign']
                and abs(cfrac) <= LEAK_FRAC_BAND*sig['on_frac_med']):
            return ('leakage',
                    f"on-source: V/I={cfrac*100:.2f}% matches local leakage "
                    f"{sig['on_frac_med']*100:.2f}% (sign {sig['on_frac_sign']:+.0f}, "
                    f"N={sig['n']}); I={iflux:.1f}Jy at {bd/beam_pix:.1f} beam", diag)

    # (2) BILOBE: candidate is an off-source lobe at the LOCAL bilobe PA & separation
    #     from a nearby I source, with V/I matching the local lobe fraction.
    if 'bilobe_pa' in sig and best is not None and np.isfinite(bd):
        dy, dx = cy-best['pix_y'], cx-best['pix_x']
        cpa = np.degrees(np.arctan2(dx, dy)) % 180
        csep = np.hypot(dy, dx)
        pa_off = min(abs(cpa-sig['bilobe_pa']), 180-abs(cpa-sig['bilobe_pa']))
        sep_off = abs(csep-sig['bilobe_sep_pix'])/max(sig['bilobe_sep_pix'], 1)
        diag.update(cand_pa=round(cpa, 1), local_bilobe_pa=round(sig['bilobe_pa'], 1),
                    pa_off=round(pa_off, 1), sep_off=round(sep_off, 2))
        # A TRUE bilobe must have a comparable OPPOSITE-SIGN partner (a real +/-
        # dipole), not just geometric position match. Search 0.5-4 beam for the
        # strongest opposite-sign V; require |partner| >= BILOBE_AMP_LO x |cand|.
        h, w = v_data.shape; sr = int(4*beam_pix)
        y1, y2 = max(0, cy-sr), min(h, cy+sr+1); x1, x2 = max(0, cx-sr), min(w, cx+sr+1)
        st = v_data[y1:y2, x1:x2]; yy, xx = np.indices(st.shape)
        dpix = np.hypot(xx-(cx-x1), yy-(cy-y1))
        m = (dpix < 0.5*beam_pix) | (dpix > sr)
        opp = (-np.sign(cand_peak) * st); opp[m] = -np.inf
        partner = float(np.max(opp)) if np.isfinite(opp).any() else 0.0
        has_partner = partner >= BILOBE_AMP_LO * abs(cand_peak)
        diag.update(partner_amp=round(partner, 3),
                    partner_ratio=round(partner/max(abs(cand_peak), 1e-3), 2))
        if (0.5*beam_pix < csep < 4*beam_pix and pa_off < BILOBE_PA_TOL
                and sep_off < BILOBE_SEP_TOL and has_partner):
            return ('bilobe',
                    f"bilobe: PA={cpa:.0f}deg vs local {sig['bilobe_pa']:.0f}deg "
                    f"(off {pa_off:.0f}), sep {csep:.1f}px vs {sig['bilobe_sep_pix']:.1f}px, "
                    f"partner {partner/max(abs(cand_peak),1e-3):.2f}x (N={sig.get('n_bilobe')})", diag)

    return 'real', 'departs from local leakage/bilobe signature', diag
