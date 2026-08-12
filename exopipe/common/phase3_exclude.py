#!/usr/bin/env python3
"""
phase3_exclude.py — the settled Phase 3 exclusion gates (Gregg, 2026-06-13).

Three layers, no hand-curated data_decisions.csv:

  1. CAL bandpass amplitude max-σ gate  (cal-level, per subband-epoch)
     Per-subband z-score of median bandpass gain across ALL cal tables, then
     per cal max_sigma = MAX over subbands of |gain-med|/std. Flag cal if
     max_sigma > 2.0 → exclude every (lst,date,subband) that used that cal
     table, FULL-DATE, until Phase 1 redo. The MAX (not mean) is essential:
     it catches localized single-band RFI lock (Cyg A / Cas A too low → faint
     Virgo A → solver locks onto horizon RFI at low freq) that a mean buries.
     Validated: 6/6 known-bad cals caught. (cal_phase_qa.py phase-jump check
     is a confirmatory diagnostic, not a gate.)

  2. 3x THERMAL-NOISE gate  (epoch-level, per band)
     Drop any epoch with THERMNOI > 3x the band median (catches massive peel
     divergence — which DOES explode the noise).

  3. TWO-STAGE OUTLIER REJECTION  (residual MAD, joint I/V)
     The existing provisional→MAD→final two-pass on residual rms.
"""
import numpy as np

CAL_MAXSIGMA_THRESH = 2.0
THERMAL_FACTOR = 3.0
OUTLIER_SIGMA = 3.0


# ── Layer 1: bandpass amplitude max-σ ────────────────────────────────
def flag_bad_cals(gain_df, thresh=CAL_MAXSIGMA_THRESH):
    """gain_df: cal_gain_summary (cols cal_date, cal_lst, freq_mhz,
    median_gain). Returns {(cal_date, cal_lst): max_sigma} for cals whose
    worst-subband gain z-score exceeds `thresh`."""
    med = gain_df.groupby('freq_mhz')['median_gain'].median()
    std = gain_df.groupby('freq_mhz')['median_gain'].std()
    bad = {}
    for (cd, cl), t in gain_df.groupby(['cal_date', 'cal_lst']):
        s = t.set_index('freq_mhz')['median_gain']
        devs = [abs(s[f] - med[f]) / std[f] for f in s.index
                if std.get(f, 0) > 0 and np.isfinite(s[f])]
        ms = max(devs) if devs else 0.0
        if ms > thresh:
            bad[(cd, cl)] = float(ms)
    return bad


def excluded_subband_epochs(gain_df, prov_df, thresh=CAL_MAXSIGMA_THRESH):
    """Map the bad-cal flags through provenance to the (lst, date, subband)
    that used those cal tables. Returns set of (lst, date, subband_str).
    FULL-DATE: any subband whose bp_table came from a flagged cal is out."""
    bad = flag_bad_cals(gain_df, thresh)
    bad_bp = set(gain_df[gain_df.apply(
        lambda r: (r['cal_date'], r['cal_lst']) in bad, axis=1)]['bp_table'])
    out = set()
    for _, r in prov_df.iterrows():
        if r.get('bp_table') in bad_bp:
            out.add((r['lst'], r['date'], r.get('subband', '')))
    return out, bad


# ── Layer 2: 3x thermal-noise gate ───────────────────────────────────
def thermal_outliers(epochs, factor=THERMAL_FACTOR):
    """epochs: list of dicts with 'date' and 'thermnoi'. Returns set of dates
    whose thermal noise exceeds factor x the median (massive peel divergence)."""
    tn = np.array([e.get('thermnoi') or np.nan for e in epochs], float)
    med = np.nanmedian(tn)
    if not np.isfinite(med) or med <= 0:
        return set()
    return {e['date'] for e, v in zip(epochs, tn)
            if np.isfinite(v) and v > factor * med}


# ── Layer 3: two-stage MAD outlier rejection ─────────────────────────
def mad_outliers(dates, rms_values, sigma=OUTLIER_SIGMA):
    """Provisional→MAD→final: flag dates whose residual rms is a >sigma MAD
    high outlier. (Port of coadd_full_pipeline.detect_outliers_mad.)"""
    vals = [(d, r) for d, r in zip(dates, rms_values)
            if r is not None and np.isfinite(r)]
    if len(vals) < 4:
        return set()
    r = np.array([v[1] for v in vals])
    med = np.median(r)
    mad = 1.4826 * np.median(np.abs(r - med))
    if mad <= 0:
        return set()
    thr = med + sigma * mad
    return {d for d, rv in vals if rv > thr}


def combined_excluded_dates(epochs, i_rms, v_rms, cal_excluded_dates):
    """Union of all three gates → dates excluded from co-add AND search.
    epochs: list of dicts (date, thermnoi). i_rms/v_rms: {date: rms}.
    cal_excluded_dates: set of dates flagged by layer 1. Joint I/V MAD."""
    dates = [e['date'] for e in epochs]
    excl = set(cal_excluded_dates)
    excl |= thermal_outliers(epochs)
    excl |= mad_outliers(dates, [i_rms.get(d) for d in dates])
    excl |= mad_outliers(dates, [v_rms.get(d) for d in dates])
    return excl
