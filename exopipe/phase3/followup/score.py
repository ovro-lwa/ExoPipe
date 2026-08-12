#!/usr/bin/env python3
"""candinv/score.py — P(astrophysical) ranking (Gregg 2026-06-28).

Replaces the old additive 0-10 corroboration score. Candidates are ranked by a
single number, P(astrophysical) in [0,1], computed as a logistic over
TRANSPARENT, itemised log-odds contributions so the value stays auditable. It is
still a sorting aid — Gregg judges the truth — but the axis is now "how likely is
this astrophysical" rather than "how much corroboration".

The decisive PHYSICAL discriminators (they killed rank-48 and rank-28 despite
I>=|V|):
  * morphology vs the EMPIRICAL beam — RESOLVED => propagation/lensing artifact;
    the peak-frame (10-second) morphology test is weighted strongest.
  * I >= |V| sanity — |V|>I (deep or in a snapshot frame) is the RFI mode.
The population-level artifact signature is LOW ELEVATION (much of the top-50
clusters at low elevation). Corroboration (multi-band, multi-epoch repeats,
stellar cross-match, coincident I+V) pushes P up; steady/persistent residuals,
A-team proximity, ephemeris bodies and striation push it down.
"""
import math

from . import config as C


def _snapshot_unphysical(snap, sspec):
    """|V|>I in a snapshot frame is a clean RFI veto (rank-10). True if any
    snapshot-spectrum band shows a strong V with V/I above the unphysical
    ceiling."""
    for r in (sspec or []):
        if abs(r.get('V_snr', 0) or 0) >= C.STRONG_SNR and \
                (r.get('vi_ratio') or 0) > C.VI_RATIO_UNPHYSICAL:
            return True
    return False


def _snapshot_physical_burst(snap, sspec):
    """A coincident I+V, physical-ratio 10-second burst is strong positive
    evidence (FI Cnc / the rank-48 burst was physical even though resolved)."""
    if not snap or snap.get('peak_snr', 0) < C.STRONG_SNR:
        return False
    for r in (sspec or []):
        if abs(r.get('V_snr', 0) or 0) >= C.STRONG_SNR and \
                abs(r.get('I_snr', 0) or 0) >= C.DETECT_SNR and \
                (r.get('vi_ratio') or 99) <= C.VI_RATIO_UNPHYSICAL:
            return True
    return False


def p_astrophysical(ev):
    spec = ev.get('spectrum_summary', {})
    morph = ev.get('morphology', {})
    smorph = ev.get('snapshot_morphology', {})
    ivc = ev.get('iv_coincidence', {})
    ctx = ev.get('context', {})
    epochs = ev.get('multi_epoch', {})
    snap = ev.get('snapshot', {})
    sspec = ev.get('snapshot_spectrum', [])

    c = {}

    # ---- decisive physical discriminators ----------------------------------
    # deep/subband morphology vs empirical beam
    if morph.get('extended'):
        c['morphology'] = -2.5
    elif morph.get('point_like'):
        c['morphology'] = +1.2
    else:
        c['morphology'] = 0.0
    # peak-frame (10-second) morphology — the strongest point-source test
    if smorph.get('extended'):
        c['snapshot_morphology'] = -3.0
    elif smorph.get('point_like'):
        c['snapshot_morphology'] = +1.8
    else:
        c['snapshot_morphology'] = 0.0
    # I >= |V| sanity (deep or snapshot)
    if ivc.get('unphysical_vi') or _snapshot_unphysical(snap, sspec):
        c['unphysical_vi'] = -3.0
    else:
        c['unphysical_vi'] = 0.0
    # coincident I+V with a physical ratio = the reliable CP-transient signature,
    # scaled by the circular-polarization fraction (V/I up to ~1).
    vi = ivc.get('vi_ratio')
    if ivc.get('iv_coincident') and vi is not None and vi <= C.VI_RATIO_UNPHYSICAL:
        c['iv_coincident'] = round(1.6 * min(1.0, vi / 0.5), 3)
    elif ivc.get('iv_coincident'):
        c['iv_coincident'] = 0.5
    else:
        c['iv_coincident'] = 0.0
    # a clean physical 10-second burst (coincident I+V at peak frame)
    c['snapshot_burst'] = 1.0 if _snapshot_physical_burst(snap, sspec) else 0.0

    # ---- corroboration -----------------------------------------------------
    nb = spec.get('n_v_bands', 0) or 0
    c['multi_band'] = round(min(2.0, 0.8 * nb), 3)   # ECM narrowband: 1 band OK
    persistent = epochs.get('persistent')
    nep = epochs.get('n_epoch_detect') or 0
    if persistent:
        c['multi_epoch'] = -2.0                       # steady / subtraction residual
    else:
        c['multi_epoch'] = round(min(1.5, 0.6 * max(0, nep - 1)), 3)
    xm = ctx.get('xmatch', {}) or {}
    c['xmatch_stellar'] = 1.0 if any(k in xm for k in C.XMATCH_POSITIVE) else 0.0

    # ---- population-level artifact signatures ------------------------------
    el = ctx.get('elevation_deg')
    if el is None:
        c['elevation'] = 0.0
    elif el < C.ELEV_MIN_DEG:
        c['elevation'] = -1.5
    elif el < C.ELEV_GOOD_DEG:
        c['elevation'] = -0.5
    else:
        c['elevation'] = +0.5
    c['ateam'] = -2.0 if (ctx.get('ateam_sep_deg', 99) or 99) < 5.0 else 0.0
    c['ephemeris'] = -2.5 if ctx.get('ephemeris_body') else 0.0
    npk = ctx.get('n_peaks_gt7sigma', 0) or 0
    c['striation'] = -1.0 if npk > 100 else (-0.5 if npk > 50 else 0.0)

    logit = float(sum(c.values()))
    p = 1.0 / (1.0 + math.exp(-logit))
    # A-team HARD exclusion (Gregg 2026-06-28): on/within 2 deg of an A-team
    # source -> that source itself / near-field artifact -> P=0 regardless of how
    # point-like/multi-band it looks (the Crab at rank-1 otherwise scored 0.875).
    ateam_sep = ctx.get('ateam_sep_deg')
    ateam_veto = bool(ateam_sep is not None and ateam_sep < C.ATEAM_VETO_DEG)
    if ateam_veto:
        p = 0.0
    return {'p_astro': round(p, 3), 'logit': round(logit, 2), 'components': c,
            'ateam_veto': ateam_veto,
            'note': ('A-team hard veto (<%.0f deg) -> P=0; ' % C.ATEAM_VETO_DEG
                     if ateam_veto else '') +
                    'P(astrophysical) sort key; logistic over itemised '
                    'log-odds. Non-authoritative — Gregg judges.'}
