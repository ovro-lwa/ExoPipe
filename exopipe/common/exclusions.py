"""Aggregator for the exopipe.common.exclusions capability (refactor step 3) — re-exports constituent modules."""
from exopipe.common.phase3_exclude import *  # noqa


# ---- 2026-09-05 (Gregg: 'the policy is to not form a wideband product with an excluded
# subband. Maintain that policy.' + dirty-level gate approved with R-REEXCLUDE-81-DIRTY-
# CORRUPT-APPROVED): the Phase-2 entry points for per-night subband exclusion.
def excluded_subbands_for_night(lst, date, subbands):
    """Subset of `subbands` (e.g. '73MHz') that data_decisions excludes for (lst, date).
    Delegates to the single data_decisions reader (coadd_full_pipeline.load_exclusions /
    is_excluded) so Phase 2 can never disagree with Phase 3."""
    from exopipe.phase3.coadd_full_pipeline import load_exclusions, is_excluded
    excl = load_exclusions()
    return {sb for sb in subbands if is_excluded(lst, sb, date, excl)}


DIRTY_RMS_GATE_FACTOR = 5.0   # Gregg 2026-09-05: dirty rms > 5x the healthy baseline = corrupt


def dirty_rms_baseline():
    """{(lst, subband): healthy dirty rms in mJy} from exopipe/common/dirty_rms_baseline.csv
    (seeded 2026-09-05 from the 2026-08-30 audit's HEALTHY-BASELINE rows; 11h/12h/23h)."""
    import csv, os
    p = os.path.join(os.path.dirname(__file__), 'dirty_rms_baseline.csv')
    out = {}
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            out[(r['lst'], r['subband'])] = float(r['healthy_dirty_rms_mjy'])
    return out


def dirty_central_rms_mjy(fits_path):
    """Central-quarter, 3-sigma-clipped rms of an image in mJy (the audit's statistic)."""
    import numpy as np
    from astropy.io import fits
    d = np.squeeze(fits.getdata(fits_path)).astype(float)
    n = d.shape[0]
    c = d[n // 4:3 * n // 4, n // 4:3 * n // 4]
    c = c[np.isfinite(c)]
    if c.size == 0:
        return float('nan')
    for _ in range(3):
        m, s = np.median(c), np.std(c)
        c = c[np.abs(c - m) < 3 * s]
    return float(np.std(c)) * 1e3


def dirty_gated_subbands(lst, subband_dirs, factor=DIRTY_RMS_GATE_FACTOR):
    """{subband: (rms_mjy, baseline_mjy)} for subbands whose Taper Robust-0.75 Stokes I
    DIRTY image has central rms > factor x the (lst, subband) healthy baseline.
    subband_dirs = {subband: phase1_subband_dir}. Subbands without a baseline entry or
    without an archived dirty image are NOT gated (logged by the caller)."""
    import glob
    base = dirty_rms_baseline()
    out = {}
    for sb, d in subband_dirs.items():
        b = base.get((lst, sb))
        if not b:
            continue
        g = [p for p in glob.glob(f'{d}/I/deep/{sb}-I-Deep-Taper-Robust-0.75-dirty-*.fits')
             if 'pbcorr' not in p]
        if not g:
            continue
        rms = dirty_central_rms_mjy(g[0])
        if rms == rms and rms > factor * b:
            out[sb] = (rms, b)
    return out
