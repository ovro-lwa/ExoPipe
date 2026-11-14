"""Source grouping, design B: the ONE grouping of every detection of a tier and Stokes (refactor item F; task B of
GROUPING_IMPLEMENTATION_BRIEF_20261026.md), run once on all bands, nights and LSTs before the fork and before every
multiplicity-dependent gate.

RULINGS
  Gregg 2026-10-26, R-GREGG-20261026-GROUPING-DESIGN-B-APPROVED-IDENTITY-097-MULTIPLES-085-DEEPV-IDENTITY-ONLY-PREVIEW-THEN-REFACTOR
  (verbatim: "All three grouping decisions are recommended according to your recommendations."): design B for the 10-minute
  and deep tiers, both Stokes; identities at P >= 0.97 with the 4-sigma cap everywhere; a nested, merge-only multiples level at
  P >= 0.85 for 10-minute I and V and deep I; deep V at the identity level only until the acceptance of 3+ groups is
  calibrated on scrambled skies (config catalog_grouping.P_MULTIPLES).
  Gregg 2026-10-27, R-GREGG-20261027-HUB-BOTH-LISTENERS-MULTIPLES-CHAIN-FORCED-EVERY-TIER ("Which ever approach is taken (e.g.
  dropping <7-sigma is fine), let's be consistent."): this module's multiples level is the ONE multiples-chain rule of every
  tier - a multiples-level group holding >= 2 distinct 10-minute windows (deep: >= 2 nights) is a multiple (multiplicity()).

THE PORT. The algorithm is the approved research code - agent B's round-4 group_B.py (/lustre/gh/main/phase3/refactor_analysis/
grouping_20261026/agent_B/group_B.py, md5 c0b652978ed7b7427946062858e4e8ad), its geometry (geom.py) and its error model
(harness.py v2b load() and cov_of, harness_deep.py) - with every constant moved to config catalog_grouping. The arithmetic and
the order of every operation are unchanged, so the partitions are the research code's EXACTLY: checked on the full inputs of
the approved preview (10-minute I and V) and on the research's deep I and V inputs; tests/test_catalog_grouping.py re-checks
it on frozen subsets. Changes that cannot alter a result: dead parameters and an unused helper dropped (probe_r, ref_of,
categories()); the frozen-window dict no longer shadows the group-seeding switch (in the research _run only groups=False
together with frozen labels behaved differently); an unknown band refuses instead of taking the Blue-class pixel scale.
The ONE change that alters a result (2026-11-14, AMEND-20261114-GROUPING-INCREMENTAL-CHANGES-ONLY-WHAT-NEW-DETECTIONS-TOUCH): in
the incremental mode a merge or a seeded group needs a component holding a new unit (below); the batch mode is unchanged.

WHAT IT DOES (one rule at two levels). A unit j against a component k, in k's tangent plane (east, north; arcsec):
  d = x_j - m_k, m_k the inverse-covariance mean of k's members; S = C_j + (sum C_i^-1)^-1 + FLOOR^2 I [+ the chromatic term in
  stage A]; u = sqrt(d^T S^-1 d); f = the areal density of the measured offset profile (Gaussian core + Student-t tail; in stage
  B plus an S/N-dependent outlier component); the odds against the background are W_k f / rho_j - rho_j the density of unrelated
  units measured by PROBES (pixel offsets in the unit's own frame), W_k the component's weight (stage A n_k + N0; stage B
  n_k - 1 + the MEASURED seed prior of the pair category: same night / same LST hour / other LST hour). j joins the most likely
  admissible component when P_bg = rho_j / (sum_k W_k f_jk + rho_j) <= 1 - P and u <= U_CAP, else it seeds a new component.
  STAGE A, per 10-minute window across bands: detections -> window-sources (chromatic term (kappa_j - kappa_k) x 75" with
  kappa = (36 MHz / nu)^2; cannot-link: no two detections of one image more than 0.5 beam apart).
  STAGE B, across windows, nights and LSTs: window-sources (at their inverse-covariance mean, with their best member's
  covariance) -> identities, brightest first; one window-source per window per component; groups of 3+ seeded by a sequential
  odds product; passes of leave-one-out re-assignment (hysteresis) and a unit-level merge test; a final re-assignment, so every
  member ends within U_CAP of its component's leave-one-out mean.
  MULTIPLES LEVEL: from the identities, merge-only passes at the looser P (group seeding among single-unit identities, then the
  merge test) - it only coarsens the identities, never re-partitions them.
  INCREMENTAL: frozen labels (the earlier nights) keep their components; only new units are assigned; components may still
  merge, so no identity seen before is ever split (group_catalog checks it and refuses). A window holding both frozen and new
  detections would be re-grouped in stage A, so group_catalog refuses that case (mode='regroup' instead). It changes only what
  a new unit touches (the grouping form of R-GREGG-20261113-SPEEDUP-CHECKIN-TEN-ANSWERS (3), "the builder runs only on the
  sources the new night touches"): at both levels a merge needs one of its two components to hold a new unit, and a seeded
  group needs a new member; the earlier multiples groups are restored by forced joins as before. So a run without a new
  detection changes nothing, and a new night changes only the components its own units join or bridge. (The research code's
  incremental run also merged and seeded among frozen components alone: on the 24 published 10-minute runs, an incremental run
  WITHOUT a new detection over a batch state merged 49 Stokes I identities - RESULT-20261113-SPEEDUP-TENMIN-
  GROUPING-NO-NEW-DETECTION-RUNS.)

KNOWN LIMITS (review 2026-11-02)
  - The incremental mode is for PURE APPENDS - whole new windows (new nights). A re-search re-numbers and re-measures the
    detections of old windows, so windows then hold both matched and new detections, the incremental mode refuses, and the run
    must regroup. Run_SLSTB 2026-10-25 vs 2026-10-31 (per Stokes x window): 846 of 3,488 windows changed. With exact fingerprints
    only, 365 windows of the new build mix both kinds (I 350, V 15) and 466 were re-measured whole (new windows under exact
    matching); with the tolerance match I 102 and V 424 windows mix both kinds (re-measured windows are now recognized in part).
    The deep tier re-searches per LST x band across all nights: a deep re-search is a regroup.
  - In the incremental mode the MULTIPLES level can split an earlier multiples group: two of its identities that gain detections
    in one new window cannot be joined again (one window-source per window per group; research group_B.py:639-645). The ruling
    protects identities, which never split (checked); a multiples-level split is recorded ('split' event, a 'move' per detection).
    An incremental run without a new detection changes nothing at either level (since 2026-11-14: merges and seeded groups need
    a new unit; before, its merge passes restarted from the frozen components and merged further - 10-minute V subset: 1
    multiples-level merge). Regroup on an unchanged input changes nothing.
  - The batch mode's stage B does not always reach a fixed point: on the 24 published 10-minute runs (Stokes I, 37,724
    window-sources) every pass after the third re-assigns ~500 window-sources, merges 34-52 components and changes the component
    of 48-71 window-sources, with no downward trend to pass 30 - re-assignment and merge undo each other (Stokes V closes at pass
    3; PREAMBLE-20261114-SPEEDUP-TENMIN-GROUPING-PASS-CAP-30-RUNS). PASSES = 3 (the research value) makes it a
    deterministic point of that cycle; an all-nights batch and an incremental run over the same detections therefore differ
    in those components (recorded by the regroup's events, not refused).
  - cand_ids are not stable across re-searches (production re-numbers them: exopipe_ops p3s3_merge.py:77-81; 10-25 vs 10-31:
    1,562 of 67,317 shared cand_ids name a detection more than 1 deg away). The state keeps each detection's fingerprint (window
    key, band, catalog ra, dec, S/N at full precision) and match errors; a detection is matched exactly or within tolerance
    (re-measured, possibly re-numbered: the IDs follow it), else it is NEW and the state's ABSENT; a cand_id beyond tolerance of
    its own record is 'reused'; candidates that are ambiguous within tolerance match none ('ambiguous'). The tolerance decides
    only the ID carry - and, in the incremental mode, which detections are frozen (a re-measured window is then not new).
    Regroup 2026-10-25 -> 10-31, exact only vs tolerance: tolerance matches I 7,382, V 4,632 (ambiguous: I 0, V 3 groups);
    retired identity IDs I 3,186 -> 149, V 5,964 -> 1,596; new identity IDs I 3,149 -> 126, V 8,334 -> 3,968; the identity
    labels of the grouping-B preview carry by ID either way (I 97/97, V 63/63).
  - grouping_inputs maps positions through astrom_frame under a process-wide environment flag (P3_TENMIN_PERTIE): SINGLE-THREAD
    use - its mappings are serialized by a lock, but another thread calling astrom_frame meanwhile could see the flag.
  - Fewer than 1000 nights per tier and Stokes (the (LST hour, night) code L x 1000 + N); more refuses.

GROUPING ERRORS ARE NOT LOCALIZATION ERRORS. The inverse-covariance mean of a component's members (independent errors) is used
here only to NAME it (display J-names). Combined positions and their errors stay the catalog's localization step (the preview:
the independent-error mean gives 2.4" for PSR J2215+5135's 986 detections, production's correlated-error localization 93").

API
  group(det, ...)                   identity labels (one int per detection; equal ints = one identity)
  group_fork(det, ...)              (identity labels, multiples labels); the multiples level nests the identities
  group_catalog(det, tier, stokes, state=None, run_label='', mode=None)   the catalog step's entry point: the operating point
                                    from config, batch / incremental / regroup grouping, stable IDs, display names, multiplicity
                                    and the record of every change - merge, split, move (every detection whose ID changes),
                                    remeasured / reused / ambiguous detections, retirement (a GroupingResult; its .state is
                                    the next run's state, .matched the detection matching)
  match_state(det, state)           which detections ARE the state's: exact (cand_id + fingerprint), else within tolerance
  GroupingState                     every ID issued so far (never reused), the assignments with each detection's fingerprint;
                                    save(dir) - one file, atomic; load(dir, tier, stokes, new=False) - no state REFUSES unless new
  fingerprints(det)                 (window key, band, catalog ra, dec, S/N) per detection
  multiplicity(det, labels, tier)   distinct windows (10-minute) or nights (deep) of each detection's group
  geometry and error model          frame_center, cdelt, lm, radec, shift_pixels, elevation_pa, detection_sigma, covariance,
                                    offset_en, mahalanobis
det is a dict of equal-length numpy arrays (lib/catalog/grouping_inputs.py builds it from a run's tables): ra, dec (deg, the
COMMON frame), C (n x 2 x 2 sky covariance, arcsec^2, east/north), snr, band, lst ('04h'), key (the window: lst|date; a 10-minute
date ends in 'T<bin>'), night, beam (FWHM, arcmin) and, for the IDs, cand_id (+ ra_raw, dec_raw for the fingerprints, and the
frame_generation string, recorded in the run summary). The algorithm reads nothing else.
"""
import collections
import copy
import json
import math
import os

import numpy as np
from scipy.spatial import cKDTree

DEG2 = 3600.0 ** 2                      # arcsec^2 per deg^2
CATS = ('SN', 'SL', 'OL', 'ALL')        # stage-B pair categories: same LST hour and night (another window); same LST hour, another
#                                         night; another LST hour; ALL other windows pooled
TIERS = ('10min', 'deep')
LEVELS = ('identity', 'multiples')
WIDEBAND = ('Red', 'Green', 'Blue', 'Full')
RESEARCH_MD5 = 'c0b652978ed7b7427946062858e4e8ad'   # the approved group_B.py (round 4) whose batch mode this module reproduces
INCREMENTAL_RULE = 'touched-only-20261114'          # the incremental mode's rule (module docstring): a merge or a seeded group needs
#                                                     a new unit; recorded in every run summary (absent: a run under the research rule)


class GroupingRefused(RuntimeError):
    """A grouping input or request the module refuses (named in the message); the catalog step refuses the unit."""


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


# =====================================================================================================================
# parameters
# =====================================================================================================================

class Params:
    """The constants of one grouping run: config catalog_grouping read when the run starts (so --set / --config overrides
    apply), then keyword overrides by attribute name (tests, diagnostics)."""

    def __init__(self, **over):
        g = _cfg().catalog_grouping
        self.p_thr = float(g.P_IDENTITY)
        self.u_cap = float(g.U_CAP)
        self.n0 = float(g.STAGE_A_PRIOR_N0)
        self.rnb_a = float(g.STAGE_A_RADIUS_DEG)
        self.rnb = float(g.STAGE_B_RADIUS_DEG)
        self.floor = float(g.STAGE_B_FLOOR_ARCSEC)
        self.passes = int(g.PASSES)
        self.hyst = float(g.HYSTERESIS)
        self.prof_co = tuple(float(x) for x in g.PROFILE_CO)
        self.prof_ws = tuple(float(x) for x in g.PROFILE_WS)
        self.sigma_r = float(g.CHROMATIC_SIGMA_ARCSEC)
        self.nu0 = float(g.CHROMATIC_NU0_MHZ)
        self.nu_wide = {str(k): float(v) for k, v in g.WIDEBAND_NU_MHZ.items()}
        self.probes = tuple(float(x) for x in g.PROBES_PX)
        self.probe_r = {str(k): float(v) for k, v in g.PROBE_RADIUS_PX.items()}
        self.probe_floor = float(g.PROBE_FLOOR_COUNT)
        self.seed_classes = tuple((int(a), int(b)) for a, b in g.SEED_PRIOR_CLASSES)
        self.seed_umax = float(g.SEED_PRIOR_UMAX)
        self.seed_nmin = int(g.SEED_PRIOR_NMIN)
        self.seed_pair_deg = float(g.SEED_PRIOR_PAIR_DEG)
        self.seed_r_clip = tuple(float(x) for x in g.SEED_PRIOR_R_CLIP)
        self.tail = bool(g.OUTLIER_TAIL)
        self.tail_snr = np.array(g.TAIL_SNR, float)
        self.tail_beyond3 = np.array(g.TAIL_BEYOND3, float)
        self.s_out = float(g.TAIL_SCALE)
        self.p_ws_3 = float(g.TAIL_PROFILE_BEYOND3)
        self.pooled = bool(g.POOLED_PRIOR)
        self.groups = bool(g.GROUP_SEEDING)
        self.u_link = float(g.GROUP_U_LINK)
        self.max_k = int(g.GROUP_MAX_K)
        self.merge_radius = float(g.MERGE_RADIUS_DEG)
        self.max_rounds = int(g.MULTIPLES_MAX_ROUNDS)
        self.frame_dec = float(g.FRAME_DEC_DEG)
        self.cdelt = {str(k): float(v) for k, v in g.CDELT_DEG.items()}
        self.gamma = float(g.ELEVATION_GAMMA)
        self.el_floor = float(g.ELEVATION_FLOOR_DEG)
        self.deep_a = float(g.DEEP_STAT_SCALE)
        self.deep_k = float(g.DEEP_SYS_SCALE)
        self.rho_scale = 1.0                            # multiplies every background (diagnostics only)
        self.band_class = {}
        for cls, members in _cfg().catalog.BAND_CLASS.items():
            for b in members:
                self.band_class[str(b)] = str(cls)
        for b in WIDEBAND:
            self.band_class[b] = b
        for k, v in over.items():
            if k.startswith('_') or not hasattr(self, k):
                raise TypeError(f'grouping: unknown parameter {k!r}')
            setattr(self, k, v)
        self.p_out_3 = math.exp(-9.0 / (2 * self.s_out ** 2))


def _params(P):
    return Params() if P is None else P


def _resolve(P=None, p_thr=None, u_cap=None, **over):
    """a private Params for one run: P (copied, never modified) or config, then keyword overrides, then p_thr / u_cap"""
    if P is None:
        P = Params(**over)
    else:
        P = copy.copy(P)
        for k, v in over.items():
            if k.startswith('_') or not hasattr(P, k):
                raise TypeError(f'grouping: unknown parameter {k!r}')
            setattr(P, k, v)
        P.p_out_3 = math.exp(-9.0 / (2 * P.s_out ** 2))
    if p_thr is not None:
        P.p_thr = float(p_thr)
    if u_cap is not None:
        P.u_cap = float(u_cap)
    return P


# =====================================================================================================================
# geometry and the error model (geom.py, harness.py v2b load()/cov_of, harness_deep.py)
# =====================================================================================================================
# FRAMES. Every 10-minute and deep frame of LST hour L is RA---SIN / DEC--SIN about (RA = (L + 0.5) x 15 deg, Dec = FRAME_DEC_DEG),
# CDELT by band class (CDELT_DEG). Probes and chance copies are built by pixel offsets in a detection's own frame, so they keep
# the frame geometry at every declination (Gregg 2026-10-26: "specify offsets/shifts in pixels rather than angular size").
# ELEVATION TERM (Gregg 2026-10-26: "include an elevation term"). Near the zenith a SIN image foreshortens a displacement along
# the local vertical by sin(elevation): an error constant in the image plane is sigma along the horizontal and sigma / sin(el)^GAMMA
# along the vertical (at the parallactic angle q of the window's mid time; deep: the hour's mid LST).

def band_class(band, P=None):
    """'Red' | 'Green' | 'Blue' | 'Full' of a band ('46MHz' -> 'Green'); an unknown band REFUSES (GroupingRefused)."""
    P = _params(P)
    c = P.band_class.get(str(band))
    if c is None:
        raise GroupingRefused(f'grouping: band {band!r} has no band class (catalog.BAND_CLASS) - REFUSING')
    return c


def cdelt(band, P=None):
    """pixel scale (deg) of each band's frames (array in, array out; a scalar in, a 0-d array out)"""
    P = _params(P)
    band = np.asarray(band)
    u, inv = np.unique(band, return_inverse=True)
    vals = np.array([P.cdelt[band_class(b, P)] for b in u.tolist()], float)
    return vals[inv].reshape(band.shape)


def frame_center(lst):
    """RA (deg) of the frames' tangent point of each detection's LST hour"""
    return np.array([(int(x[:2]) + 0.5) * 15.0 for x in lst])


def lm(ra, de, ra0, de0):
    a, d, a0, d0 = map(np.radians, (ra, de, ra0, de0))
    da = a - a0
    l = np.cos(d) * np.sin(da)
    m = np.sin(d) * np.cos(d0) - np.cos(d) * np.sin(d0) * np.cos(da)
    return l, m


def radec(l, m, ra0, de0):
    a0, d0 = np.radians(ra0), np.radians(de0)
    r2 = l ** 2 + m ** 2
    n = np.sqrt(np.where(r2 < 1, 1 - r2, np.nan))
    d = np.arcsin(m * np.cos(d0) + n * np.sin(d0))
    a = a0 + np.arctan2(l, n * np.cos(d0) - m * np.sin(d0))
    return np.degrees(a) % 360, np.degrees(d)


def shift_pixels(det, dx, dy=0.0, P=None):
    """sky positions of det moved by (dx, dy) pixels in each detection's own frame (NaN where pushed off the projection)"""
    P = _params(P)
    ra0 = frame_center(det['lst']); cd = np.radians(cdelt(det['band'], P))
    l, m = lm(det['ra'], det['dec'], ra0, P.frame_dec)
    return radec(l + dx * cd, m + dy * cd, ra0, P.frame_dec)


def window_mid_lst_deg(lst, key, tier):
    """LST (deg) of each detection's time: the 10-minute window's mid time (hour + (bin x 10 + 5) min), deep the hour's mid LST"""
    if tier == 'deep':
        return np.array([(int(x[:2]) + 0.5) * 15.0 for x in lst])
    if tier != '10min':
        raise GroupingRefused(f'grouping: tier {tier!r} is not one of {TIERS} - REFUSING')
    b = np.array([int(k.split('T')[-1]) for k in key]); h = np.array([int(x[:2]) for x in lst])
    return (h + (b * 10 + 5) / 60.0) * 15.0


def elevation_pa(det, tier, P=None):
    """(elevation deg, parallactic angle q deg - the position angle of the direction to the zenith, east of north)"""
    P = _params(P)
    lstw = window_mid_lst_deg(det['lst'], det['key'], tier)
    ha, d, ph = np.radians(lstw - det['ra']), np.radians(det['dec']), np.radians(P.frame_dec)
    el = np.degrees(np.arcsin(np.sin(d) * np.sin(ph) + np.cos(d) * np.cos(ph) * np.cos(ha)))
    q = np.degrees(np.arctan2(np.sin(ha), np.tan(ph) * np.cos(d) - np.sin(d) * np.cos(ha)))
    return el, q


def detection_sigma(stat, sys, tier, P=None):
    """the grouping sigma (arcsec, image plane) of each detection. 10-minute: hypot(pos_stat_maj_arcsec, the window tie's per-case
    systematic) - the validated core, NOT sublst_pertie.sys_arcsec (its window-position term is 2.3x too large for pair offsets).
    Deep: hypot(DEEP_STAT_SCALE x stat, DEEP_SYS_SCALE x pos_sys): the night-to-night jitter (the table systematic is the
    scatter against the sky)."""
    P = _params(P)
    stat = np.asarray(stat, float); sys = np.asarray(sys, float)
    if tier == 'deep':
        return np.hypot(P.deep_a * stat, P.deep_k * sys)
    if tier != '10min':
        raise GroupingRefused(f'grouping: tier {tier!r} is not one of {TIERS} - REFUSING')
    return np.hypot(stat, sys)


def covariance(det, tier, P=None):
    """each detection's 2x2 sky covariance (east, north; arcsec^2): sigma (det['sig']) along the horizontal, sigma / sin(el)^GAMMA
    along the vertical (el at least ELEVATION_FLOOR_DEG), plus the Master-frame term fsig^2 on both axes (det['fsig'], when
    present: astrom_frame.term, independent between detections - conservative within one (LST, band), exact across LSTs)"""
    P = _params(P)
    el, q = elevation_pa(det, tier, P)
    s = det['sig']; sv = s * np.sin(np.radians(np.maximum(el, P.el_floor))) ** (-P.gamma)
    qr = np.radians(q)
    uv = np.c_[np.sin(qr), np.cos(qr)]                    # unit vector toward the zenith in (east, north)
    uh = np.c_[np.cos(qr), -np.sin(qr)]                   # the horizontal
    C = (sv ** 2)[:, None, None] * uv[:, :, None] * uv[:, None, :] + (s ** 2)[:, None, None] * uh[:, :, None] * uh[:, None, :]
    if 'fsig' in det:
        C = C + (det['fsig'] ** 2)[:, None, None] * np.eye(2)[None]
    return C


def offset_en(ra1, de1, ra2, de2):
    """(east, north) offset (arcsec) of 2 from 1, in 1's tangent plane"""
    a1, d1, a2, d2 = map(np.radians, (ra1, de1, ra2, de2))
    cosc = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(a2 - a1)
    x = np.cos(d2) * np.sin(a2 - a1) / cosc
    y = (np.cos(d1) * np.sin(d2) - np.sin(d1) * np.cos(d2) * np.cos(a2 - a1)) / cosc
    return np.degrees(x) * 3600, np.degrees(y) * 3600


def mahalanobis(dx, dy, C):
    """sqrt(d^T C^-1 d) for arrays of offsets and 2x2 covariances"""
    a, b, c = C[:, 0, 0], C[:, 0, 1], C[:, 1, 1]
    det_ = a * c - b * b
    return np.sqrt((c * dx * dx - 2 * b * dx * dy + a * dy * dy) / det_)


# =====================================================================================================================
# design B (agent_B/group_B.py round 4, ported; every line keeps the research arithmetic)
# =====================================================================================================================

def _unit(ra, de):
    r, d = np.radians(ra), np.radians(de)
    return np.c_[np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)]


def _chord(deg):
    return 2 * np.sin(np.radians(deg) / 2)


def _inv2(A):
    a, b, c, d = A[..., 0, 0], A[..., 0, 1], A[..., 1, 0], A[..., 1, 1]
    det = a * d - b * c
    out = np.empty_like(A)
    out[..., 0, 0] = d / det; out[..., 0, 1] = -b / det; out[..., 1, 0] = -c / det; out[..., 1, 1] = a / det
    return out


def _quad(d, S):
    """d^T S^-1 d and det S for arrays (n,2) and (n,2,2)"""
    a, b, c = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
    det = a * c - b * b
    return (c * d[..., 0] ** 2 - 2 * b * d[..., 0] * d[..., 1] + a * d[..., 1] ** 2) / det, det


def _areal(q, detS, prof):
    """areal density (per arcsec^2) of the profile at q = u^2 for covariance S (det detS)"""
    t, s1, s2, nu = prof
    r = np.sqrt(detS)
    return ((1 - t) * np.exp(-q / (2 * s1 * s1)) / (2 * math.pi * s1 * s1 * r)
            + t * (1 + q / (nu * s2 * s2)) ** (-(nu + 2) / 2) / (2 * math.pi * s2 * s2 * r))


def _eps_of(snr, P):
    """the outlier fraction of a unit of S/N snr: target(s) interpolated in log-log at the bin centers, minus what the profile
    already holds beyond u = 3, as a weight of the outlier component N(0, TAIL_SCALE^2 S) (>= 0)"""
    tgt = np.exp(np.interp(np.log(np.maximum(snr, 1.0)), np.log(P.tail_snr), np.log(P.tail_beyond3)))
    return np.clip((tgt - P.p_ws_3) / (P.p_out_3 - P.p_ws_3), 0.0, 1.0)


def _areal_tail(q, detS, prof, eps, s_out):
    """the profile with an outlier component of weight eps: (1 - eps) prof + eps N(0, s_out^2 S)"""
    return (1 - eps) * _areal(q, detS, prof) + eps * np.exp(-q / (2 * s_out ** 2)) / (2 * math.pi * s_out ** 2 * np.sqrt(detS))


def _profile_mass(prof, umax):
    uu = np.linspace(0, umax, 20001); t, s1, s2, nu = prof
    pdf = (1 - t) * uu / s1 ** 2 * np.exp(-uu ** 2 / (2 * s1 ** 2)) + t * uu / s2 ** 2 * (1 + uu ** 2 / (nu * s2 ** 2)) ** (-(nu + 2) / 2)
    return float(np.trapz(pdf, uu))


def _offset1(ra1, de1, ra2, de2):
    """scalar offset_en: (east, north) arcsec of 2 from 1 in 1's tangent plane"""
    a1, d1, a2, d2 = math.radians(ra1), math.radians(de1), math.radians(ra2), math.radians(de2)
    cd1, sd1, cd2, sd2, c12 = math.cos(d1), math.sin(d1), math.cos(d2), math.sin(d2), math.cos(a2 - a1)
    cosc = sd1 * sd2 + cd1 * cd2 * c12
    return (math.degrees(cd2 * math.sin(a2 - a1) / cosc) * 3600, math.degrees((cd1 * sd2 - sd1 * cd2 * c12) / cosc) * 3600)


def _nu_of(b, P):
    """frequency (MHz) of a band: a wideband image at its WIDEBAND_NU_MHZ, a subband '<nu>MHz' at nu; anything else REFUSES"""
    if b in P.nu_wide:
        return P.nu_wide[b]
    try:
        if not str(b).endswith('MHz'):
            raise ValueError(b)
        return float(str(b)[:-3])
    except ValueError:
        raise GroupingRefused(f'grouping: band {b!r} is neither a wideband ({", ".join(P.nu_wide)}) nor <nu>MHz - REFUSING') from None


def _kappa_of(band, P):
    """the chromatic lever (NU0 / nu)^2 of each detection (a wideband image at its WIDEBAND_NU_MHZ)"""
    nu = np.array([_nu_of(b, P) for b in band])
    return (P.nu0 / nu) ** 2


def _probe_density(ra, de, lst, band, key, same_key, radius_px, stat, P):
    """units per arcsec^2 around each unit (ra, de in the frame of (lst, band)), counted among the units in discs of radius_px
    pixels at the PROBES pixel offsets along the frame's x axis (toward/away from the meridian); units of the SAME window key
    (same_key) or of OTHER windows. Off-image probes are dropped."""
    pts_ra, pts_de, pts_key = ra, de, key
    n = len(ra); Pv = _unit(pts_ra, pts_de); t = cKDTree(Pv)
    ra0 = frame_center(lst); l, m = lm(ra, de, ra0, P.frame_dec); cd = np.radians(cdelt(band, P))
    rdeg = radius_px * np.degrees(cd); rch = _chord(rdeg)
    C = np.full((len(P.probes), n), np.nan)
    order = np.argsort(pts_key, kind='stable'); sk = pts_key[order]
    for a, s in enumerate(P.probes):
        pa, pd = radec(l - np.sign(l) * s * cd, m, ra0, P.frame_dec)      # positive s: toward the meridian
        ok = np.isfinite(pa) & np.isfinite(pd)
        Q = _unit(pa[ok], pd[ok]); idx = np.nonzero(ok)[0]
        tot = t.query_ball_point(Q, rch[ok], return_length=True).astype(float)
        same = np.zeros(len(idx))
        ukeys, inv = np.unique(key[idx], return_inverse=True)            # same-key counts: brute force per key (keys are small)
        lo = np.searchsorted(sk, ukeys, 'left'); hi = np.searchsorted(sk, ukeys, 'right')
        for g in range(len(ukeys)):
            qi = np.nonzero(inv == g)[0]
            pj = order[lo[g]:hi[g]]
            if len(pj) == 0:
                continue
            dd = np.linalg.norm(Q[qi][:, None, :] - Pv[pj][None, :, :], axis=2)
            same[qi] = (dd < rch[idx[qi]][:, None]).sum(1)
        C[a, idx] = same if same_key else tot - same
    with np.errstate(all='ignore'):
        c = np.nanmedian(C, 0) if stat == 'median' else np.nanmean(C, 0)
    floor = P.probe_floor
    c = np.where(np.isfinite(c), c, floor)
    return np.maximum(c, floor) / (math.pi * rdeg ** 2) / DEG2


def _probe_lists(ra, de, lst, band, radius_px, pts, P):
    """for each PROBES offset: (query unit indices, point indices) of the points within radius_px pixels of the displaced units"""
    ra0 = frame_center(lst); l, m = lm(ra, de, ra0, P.frame_dec); cd = np.radians(cdelt(band, P))
    rdeg = radius_px * np.degrees(cd); rch = _chord(rdeg); out = []
    for s in P.probes:
        pa, pd = radec(l - np.sign(l) * s * cd, m, ra0, P.frame_dec)          # positive s: toward the meridian
        ok = np.isfinite(pa) & np.isfinite(pd); idx = np.nonzero(ok)[0]
        res = pts.query_ball_point(_unit(pa[ok], pd[ok]), rch[ok])
        ln = np.fromiter((len(r) for r in res), int, len(idx))
        out.append((idx, np.repeat(idx, ln), np.fromiter((j for r in res for j in r), int, ln.sum())))
    return out, rdeg


def _category_density(ra, de, lst, band, L, N, w, P):
    """per unit, the areal density (per arcsec^2) of unrelated units in each pair category, from the PROBES (pixel offsets):
    rho_L = units of the unit's LST hour in other windows (mean of the probes, 'same_lst' discs), split by the night's share of
    that LST hour's units into rho_SN (same night) and rho_SL (other nights); rho_OL = units of other LST hours (median of the
    probes, 'other_lst' discs). Floor PROBE_FLOOR_COUNT per disc."""
    floor = P.probe_floor
    n = len(ra); t = cKDTree(_unit(ra, de))
    lists, rL = _probe_lists(ra, de, lst, band, P.probe_r['same_lst'], t, P)
    CL = np.full((len(P.probes), n), np.nan)
    for a, (idx, ii, jj) in enumerate(lists):
        ok = (L[jj] == L[ii]) & (w[jj] != w[ii]); CL[a, idx] = np.bincount(ii[ok], minlength=n)[idx]
    lists, rO = _probe_lists(ra, de, lst, band, P.probe_r['other_lst'], t, P)
    CO = np.full((len(P.probes), n), np.nan)
    for a, (idx, ii, jj) in enumerate(lists):
        ok = L[jj] != L[ii]; CO[a, idx] = np.bincount(ii[ok], minlength=n)[idx]
    with np.errstate(all='ignore'):
        cl = np.nanmean(CL, 0); co = np.nanmedian(CO, 0)
    cl = np.maximum(np.where(np.isfinite(cl), cl, floor), floor) / (math.pi * rL ** 2) / DEG2
    co = np.maximum(np.where(np.isfinite(co), co, floor), floor) / (math.pi * rO ** 2) / DEG2
    # the night's share of its LST hour's units (other windows); the (LST hour, night) code L x 1000 + N assumes fewer than 1000
    # nights per tier and Stokes (_run refuses more)
    nl = np.bincount(L, minlength=24).astype(float); ln_ = L * 1000 + N
    u_ln, inv_ln, c_ln = np.unique(ln_, return_inverse=True, return_counts=True); nw = np.bincount(w).astype(float)
    share = (c_ln[inv_ln] - nw[w]) / np.maximum(nl[L] - nw[w], 1.0)
    half = 0.5 / np.maximum(nl[L], 1.0)                             # smoothed: an LST hour seen on one night still has an SL background
    share = np.clip(share, half, 1.0 - half)
    return np.vstack([cl * share, cl * (1 - share), co, cl + co])  # (4, n): SN, SL, OL, ALL


def _seed_prior(ura, ude, uC, L, N, w, ulst, uband, ndet, prof, P):
    """MEASURED seed prior per pair category and band-count class: the expected number of recurrences, in that category, of a
    window-source with m detections, from the catalog itself: r = (observed - chance) / (1 - chance), the real fraction of units
    with a partner of that category within u < SEED_PRIOR_UMAX (u with both units' covariances; chance from the units displaced
    by the PROBES pixel offsets, categories taken relative to the undisplaced unit); W0 = -ln(1 - r) / F(umax). Classes with fewer
    than SEED_PRIOR_NMIN units pool into the class below; W0 is made non-decreasing in m. Returns W0 (4 categories x classes),
    each unit's class and a table."""
    umax, nmin, classes = P.seed_umax, P.seed_nmin, P.seed_classes
    n = len(ura); t = cKDTree(_unit(ura, ude))
    ra0 = frame_center(ulst); l, m = lm(ura, ude, ra0, P.frame_dec); cd = np.radians(cdelt(uband, P))

    def has(qra, qde, ok):
        h = np.zeros((4, n), bool); idx = np.nonzero(ok)[0]
        res = t.query_ball_point(_unit(qra[idx], qde[idx]), _chord(P.seed_pair_deg))
        ln = np.fromiter((len(r) for r in res), int, len(idx))
        ii = np.repeat(idx, ln); jj = np.fromiter((j for r in res for j in r), int, ln.sum())
        c = np.where(L[jj] != L[ii], 2, np.where(N[jj] == N[ii], 0, 1)); keep = (w[jj] != w[ii]) & (ii != jj)
        ii, jj, c = ii[keep], jj[keep], c[keep]
        dx, dy = offset_en(qra[ii], qde[ii], ura[jj], ude[jj])
        q, _ = _quad(np.stack([dx, dy], -1), uC[ii] + uC[jj]); s = q < umax ** 2
        for k in range(3):
            h[k, ii[s & (c == k)]] = True
        h[3, ii[s]] = True
        return h
    obs = has(ura, ude, np.ones(n, bool)).astype(float); ch = np.zeros((4, n)); cnt = np.zeros(n)
    for s in P.probes:
        pa, pd = radec(l - np.sign(l) * s * cd, m, ra0, P.frame_dec); ok = np.isfinite(pa) & np.isfinite(pd)
        ch += has(np.where(ok, pa, 0), np.where(ok, pd, 0), ok); cnt += ok
    ch = ch / np.maximum(cnt, 1)
    F = _profile_mass(prof, umax)
    cl = [list(c) for c in classes]
    while len(cl) > 1 and ((ndet >= cl[-1][0]) & (ndet <= cl[-1][1])).sum() < nmin:
        cl[-2][1] = cl[-1][1]; cl.pop()
    cls = np.zeros(n, int); W0 = np.zeros((4, len(cl))); table = []
    rlo, rhi = P.seed_r_clip
    for g, (lo, hi) in enumerate(cl):
        s_ = (ndet >= lo) & (ndet <= hi); cls[s_] = g
        for k in range(4):
            o, c = obs[k, s_].mean(), ch[k, s_].mean(); r = min(max((o - c) / (1 - c), rlo), rhi)
            W0[k, g] = -math.log(1 - r) / F; table.append([CATS[k], lo, hi, int(s_.sum()), round(o, 4), round(c, 4), round(W0[k, g], 3)])
    W0 = np.maximum.accumulate(W0, axis=1)
    return W0, cls, table


class _Mixture:
    """components in their anchors' tangent planes: information A = sum C^-1, b = sum C^-1 x, member counts, chromatic lever
    (stage A), members per image (stage A) or windows held, LST-hour and (LST hour, night) counts and seed-prior classes (stage B)"""

    def __init__(self, ra, de, C, floor, P, n0=1.0, kappa=None, img=None, beam=None, slot=None, L=None, N=None, W0=None, cls=None,
                 eps=None):
        n = len(ra); self.n = n; self.eps = eps; self.pooled = True; self.sigma_r = P.sigma_r; self.s_out = P.s_out
        self.ra, self.de, self.C, self.Ci = ra, de, C, _inv2(C); self.V = _unit(ra, de)
        self.floor2 = floor ** 2; self.n0 = n0; self.kappa = kappa; self.img = img; self.beam = beam; self.slot = slot
        self.L, self.N, self.W0, self.cls = L, N, W0, cls
        self.w = 1.0 / np.trace(C, axis1=1, axis2=2) if kappa is not None else None
        self.ara = np.zeros(n); self.ade = np.zeros(n); self.A = np.zeros((n, 2, 2)); self.b = np.zeros((n, 2)); self.cnt = np.zeros(n, int)
        self.skw = np.zeros(n); self.sw = np.zeros(n)
        if W0 is not None:
            self.ccnt = np.zeros((n, W0.shape[1]), int); self.cl_ = [dict() for _ in range(n)]; self.cln = [dict() for _ in range(n)]
        self.mem = [set() for _ in range(n)]; self.slots = [dict() for _ in range(n)]
        self.lab = -np.ones(n, int); self.free = list(range(n - 1, -1, -1))

    def _x(self, j, k):
        return np.array(_offset1(self.ara[k], self.ade[k], self.ra[j], self.de[j]))

    def _acc(self, j, k, sg):
        x = self._x(j, k); self.A[k] += sg * self.Ci[j]; self.b[k] += sg * (self.Ci[j] @ x)
        if self.w is not None:
            self.skw[k] += sg * self.w[j] * self.kappa[j]; self.sw[k] += sg * self.w[j]

    def add(self, j, k):
        if self.cnt[k] == 0:
            self.ara[k], self.ade[k] = self.ra[j], self.de[j]
        self._acc(j, k, 1.0); self.cnt[k] += 1
        if self.W0 is not None:
            self.ccnt[k, self.cls[j]] += 1; a, b = int(self.L[j]), (int(self.L[j]), int(self.N[j]))
            self.cl_[k][a] = self.cl_[k].get(a, 0) + 1; self.cln[k][b] = self.cln[k].get(b, 0) + 1
        key = self.slot[j] if self.slot is not None else self.img[j]
        self.slots[k].setdefault(key, set()).add(j); self.mem[k].add(j); self.lab[j] = k

    def remove(self, j):
        k = self.lab[j]; self._acc(j, k, -1.0); self.cnt[k] -= 1
        if self.W0 is not None:
            self.ccnt[k, self.cls[j]] -= 1; a, b = int(self.L[j]), (int(self.L[j]), int(self.N[j]))
            self.cl_[k][a] -= 1; self.cln[k][b] -= 1
            if self.cl_[k][a] == 0:
                del self.cl_[k][a]
            if self.cln[k][b] == 0:
                del self.cln[k][b]
        key = self.slot[j] if self.slot is not None else self.img[j]
        s = self.slots[k][key]; s.discard(j)
        if not s:
            del self.slots[k][key]
        self.mem[k].discard(j); self.lab[j] = -1
        if self.cnt[k] == 0:
            self.A[k] = 0; self.b[k] = 0; self.skw[k] = 0; self.sw[k] = 0; self.free.append(k)
        return k

    def new(self, j):
        k = self.free.pop(); self.add(j, k); return k

    def rebuild(self):
        """recompute every component's sums from its members, anchors kept (no drift from repeated add/remove)"""
        live = np.nonzero(self.cnt > 0)[0]
        self.A[live] = 0; self.b[live] = 0; self.skw[live] = 0; self.sw[live] = 0
        for k in live:
            for j in self.mem[k]:
                self._acc(j, k, 1.0)

    def top(self, ks):
        c = self.ccnt[ks]; return c.shape[1] - 1 - np.argmax(c[:, ::-1] > 0, axis=1)

    def mean_cov(self, ks):
        Ai = _inv2(self.A[ks]); return np.einsum('nij,nj->ni', Ai, self.b[ks]), Ai

    def prior_odds(self, ks, ref, rho, top):
        """stage B: max over pair categories c of W_kc / rho_c(ref): W_kc = n_kc - 1 + W0[c, class] if k holds n_kc >= 1 units of
        category c relative to the unit `ref` (the evaluated unit, or the reference unit of a merged component), else 0"""
        Lr, Nr = int(self.L[ref]), int(self.N[ref])
        nL = np.array([self.cl_[k].get(Lr, 0) for k in ks]); nSN = np.array([self.cln[k].get((Lr, Nr), 0) for k in ks])
        n_c = np.vstack([nSN, nL - nSN, self.cnt[ks] - nL, self.cnt[ks]])
        Wc = np.where(n_c >= 1, n_c - 1 + self.W0[:, top], 0.0)
        Wc[3] = np.where(n_c[3] >= 2, n_c[3] - 1, 0.0) if self.pooled else 0.0   # pooled: the OTHER members only (a seed keeps its
        #                                                                           category prior), so evidence accumulates across categories
        return np.divide(Wc, rho[:, ref][:, None], out=np.zeros_like(Wc), where=Wc > 0).max(0)

    def evaluate(self, j, ks, prof, rho):
        """(odds against the background, u) of unit j for components ks"""
        m, Ai = self.mean_cov(ks)
        dx, dy = offset_en(self.ara[ks], self.ade[ks], self.ra[j], self.de[j])
        d = np.stack([dx, dy], axis=-1) - m
        S = self.C[j][None] + Ai + self.floor2 * np.eye(2)[None]
        if self.kappa is not None:
            S = S + ((self.sigma_r * (self.kappa[j] - self.skw[ks] / self.sw[ks])) ** 2)[:, None, None] * np.eye(2)[None]
        q, dS = _quad(d, S); f = _areal(q, dS, prof) if self.eps is None else _areal_tail(q, dS, prof, self.eps[j], self.s_out)
        if self.W0 is None:
            return (self.cnt[ks] + self.n0) * f / rho[j], np.sqrt(q)
        return self.prior_odds(ks, j, rho, np.maximum(self.top(ks), self.cls[j])) * f, np.sqrt(q)

    def evaluate_units(self, js, k, prof, rho):
        """(odds, u) of MANY units js against ONE component k (stage B; the vectorized twin of evaluate)"""
        js = np.asarray(js); m, Ai = self.mean_cov(np.array([k]))
        dx, dy = offset_en(np.full(len(js), self.ara[k]), np.full(len(js), self.ade[k]), self.ra[js], self.de[js])
        d = np.stack([dx, dy], axis=-1) - m[0]
        S = self.C[js] + Ai[0][None] + self.floor2 * np.eye(2)[None]
        q, dS = _quad(d, S); f = _areal(q, dS, prof) if self.eps is None else _areal_tail(q, dS, prof, self.eps[js], self.s_out)
        Lj, Nj = self.L[js], self.N[js]
        nL = np.array([self.cl_[k].get(int(a), 0) for a in Lj]); nSN = np.array([self.cln[k].get((int(a), int(b)), 0) for a, b in zip(Lj, Nj)])
        n_c = np.vstack([nSN, nL - nSN, np.full(len(js), self.cnt[k]) - nL, np.full(len(js), self.cnt[k])])
        c = self.ccnt[k]; topk = c.shape[0] - 1 - np.argmax(c[::-1] > 0)
        top = np.maximum(topk, self.cls[js])
        Wc = np.where(n_c >= 1, n_c - 1 + self.W0[:, top], 0.0)
        Wc[3] = np.where(n_c[3] >= 2, n_c[3] - 1, 0.0) if self.pooled else 0.0
        O = np.divide(Wc, rho[:, js], out=np.zeros_like(Wc), where=Wc > 0).max(0)
        return O * f, np.sqrt(q)

    def admissible(self, j, ks):
        """the cannot-link: stage B one unit per window; stage A no two detections of one image > 0.5 beam apart"""
        if self.slot is not None:
            return [k for k in ks if self.slot[j] not in self.slots[k]]
        out = []
        for k in ks:
            others = self.slots[k].get(self.img[j])
            if others:
                lim = math.cos(math.radians(0.5 * self.beam[j] / 60.0)); vj = self.V[j]
                if any(float(self.V[o] @ vj) < lim for o in others):
                    continue
            out.append(k)
        return out


def _assign_pass(M, order, cands_of, rho, prof, p_thr, u_cap, first, hyst, movable=None):
    moved = 0; othr = p_thr / (1 - p_thr)
    for j in order:
        if movable is not None and not movable[j]:
            continue
        old = -1 if first or M.lab[j] < 0 else M.remove(j)
        ks = M.admissible(j, cands_of(j))
        best = -1
        if ks:
            ks = np.fromiter(ks, int, len(ks)); O, u = M.evaluate(j, ks, prof, rho)
            ok = u <= u_cap
            if ok.any():
                ks, O = ks[ok], O[ok]
                b = int(np.argmax(O))
                if old >= 0 and M.cnt[old] > 0:
                    w_ = np.nonzero(ks == old)[0]
                    if len(w_) and O[b] < hyst * O[w_[0]]:
                        b = int(w_[0])
                if O.sum() >= othr:
                    best = int(ks[b])
        if best >= 0:
            M.add(j, best)
        else:
            best = M.new(j)
        moved += int(best != old)
    return moved


def _group_seed(M, order, nbrs, prof, rho, p_thr, u_cap, u_link, max_k, movable=None):
    """GROUP EVIDENCE (round 4): among units still ALONE in their components, grow groups of mutually consistent units in different
    windows (each new member the strongest candidate within u_link of the group's mean, up to max_k) and seed a group as one source
    when (1) its CHAIN ODDS - the product, in brightest-first order, of each member's odds against the members before it (the same
    W f / rho as every assignment: the evidence of k coincident units grows as ~ (f / rho)^(k-1) while chance falls as
    (rho x area)^(k-1)) - reach the threshold odds p/(1-p), and (2) every member is admitted given the others (leave-one-out odds
    >= p/(1-p), u <= U_CAP; members failing it are dropped). For a pair both conditions equal the ordinary pair rule, so only
    k >= 3 is new. movable (the incremental mode): a group is seeded only when it holds a new unit. Returns the number of groups
    seeded."""
    othr = p_thr / (1 - p_thr); rank = np.empty(M.n, int); rank[order] = np.arange(len(order)); formed = 0
    for j in order:
        k = M.lab[j]
        if k < 0 or M.cnt[k] != 1:
            continue
        cs = np.array([c for c in nbrs[j] if c != j and M.lab[c] >= 0 and M.cnt[M.lab[c]] == 1 and M.slot[c] != M.slot[j]], int)
        if len(cs) < 2:
            continue
        if movable is not None and not movable[j] and not movable[cs].any():
            continue                                      # the incremental mode: no new unit among them, no group
        O, U = M.evaluate_units(cs, k, prof, rho)          # the pair prefilter: partners within u_link of the anchor
        cs = cs[U <= min(u_link, u_cap)]
        if len(cs) < 2:
            continue
        grp = [j]
        while True:
            cand = np.array([c for c in cs if M.lab[c] != k and M.cnt[M.lab[c]] == 1 and M.slot[c] not in M.slots[k]], int)
            if len(cand) == 0:
                break
            O, U = M.evaluate_units(cand, k, prof, rho)
            ok = U <= min(u_link, u_cap)
            if not ok.any():
                break
            b = int(np.argmax(np.where(ok, O, -1.0)))       # the strongest consistent candidate joins (a pair's own odds may be < 1:
            c = int(cand[b]); M.remove(c); M.add(c, k); grp.append(c)   # the evidence is judged on the whole group below)
            if len(grp) >= max_k:
                break
        if len(grp) < 3:                                  # pairs: the ordinary rule already decided
            for c in grp[1:]:
                M.remove(c); M.new(c)
            continue
        while len(grp) >= 2:                              # leave-one-out admission of every member
            lo = np.empty(len(grp))
            for x, i in enumerate(grp):
                M.remove(i); o, u = M.evaluate(i, np.array([k]), prof, rho); M.add(i, k)
                lo[x] = o[0] if u[0] <= u_cap else -1.0
            if (lo >= othr).all():
                break
            x = int(np.argmin(lo)); i = grp.pop(x); M.remove(i); M.new(i)
        if len(grp) < 3:
            for c in grp:
                if M.cnt[M.lab[c]] > 1:
                    M.remove(c); M.new(c)
            continue
        gsr = sorted(grp, key=lambda i: rank[i])          # chain odds in brightest-first order
        for i in gsr:
            M.remove(i)
        k2 = M.new(gsr[0]); chain = 1.0
        for i in gsr[1:]:
            o, u = M.evaluate(i, np.array([k2]), prof, rho); chain *= o[0]; M.add(i, k2)
        if chain >= othr and (movable is None or movable[gsr].any()):
            formed += 1
        else:
            for i in gsr[1:]:
                M.remove(i); M.new(i)
    return formed


def _merge_pass(M, prof, p_thr, u_cap, rho, radius_deg, movable=None):
    """UNIT-LEVEL merge test: component l joins k when EVERY unit of l, taken alone, would be admitted to k (u <= U_CAP against k's
    mean with its covariance, odds >= P) and the two share no window. (A mean-against-mean test with independent errors is too
    strict: systematics correlated within a night or LST hour leave well-sampled pieces of one source many 'mean sigmas' apart
    while every member sits within 2 sigma of the other piece's mean - J0034-0721.) movable (the incremental mode: the new
    units): only a pair of which one component holds a new unit may merge."""
    live = np.nonzero(M.cnt > 0)[0]
    if len(live) < 2:
        return 0
    m, Ai = M.mean_cov(live)
    mra = M.ara[live] + m[:, 0] / 3600.0 / np.cos(np.radians(M.ade[live])); mde = M.ade[live] + m[:, 1] / 3600.0
    t = cKDTree(_unit(mra, mde)); pr = t.query_pairs(_chord(radius_deg), output_type='ndarray')
    if len(pr) == 0:
        return 0
    a, b = pr[:, 0], pr[:, 1]
    wa, wb = M.cnt[live[a]], M.cnt[live[b]]
    big = np.where(wa >= wb, a, b); sm = np.where(wa >= wb, b, a)
    kb, ksm = live[big], live[sm]
    if movable is not None:                               # the incremental mode: a merge needs a component holding a new unit
        hn = np.zeros(len(M.cnt), bool)
        l_ = M.lab[np.nonzero(movable)[0]]
        hn[l_[l_ >= 0]] = True
        keep = hn[kb] | hn[ksm]
        if not keep.any():
            return 0
        big, sm, kb, ksm = big[keep], sm[keep], kb[keep], ksm[keep]
    # prefilter: l's mean against k's mean with the covariance of l's loosest unit (each unit must pass the cap on its own)
    cmax = {}
    for l in set(ksm.tolist()):
        mem = np.fromiter(M.mem[l], int); cmax[l] = M.C[mem[np.argmax(np.trace(M.C[mem], axis1=1, axis2=2))]]
    Cm = np.array([cmax[l] for l in ksm])
    dx, dy = offset_en(mra[big], mde[big], mra[sm], mde[sm])
    q, _ = _quad(np.stack([dx, dy], -1), Ai[big] + Cm + M.floor2 * np.eye(2)[None])
    cand = np.nonzero(q <= u_cap ** 2)[0]
    othr = p_thr / (1 - p_thr); done = set(); merged = 0
    for o in cand[np.argsort(q[cand])]:
        k, l = int(kb[o]), int(ksm[o])
        if k in done or l in done:
            continue
        if set(M.slots[k]) & set(M.slots[l]):
            continue
        ok = True
        for j in M.mem[l]:
            O, u = M.evaluate(j, np.array([k]), prof, rho)
            if not (u[0] <= u_cap and O[0] >= othr):
                ok = False; break
        if not ok:
            continue
        for j in list(M.mem[l]):
            M.remove(j); M.add(j, k)
        done.add(k); done.add(l); merged += 1
    return merged


def _check_det(det):
    need = ('ra', 'dec', 'C', 'snr', 'band', 'lst', 'key', 'night', 'beam')
    miss = [k for k in need if k not in det]
    if miss:
        raise GroupingRefused(f'grouping: det lacks {miss} - REFUSING')
    n = len(det['ra'])
    bad = [k for k in need if len(det[k]) != n]
    if bad:
        raise GroupingRefused(f'grouping: det fields {bad} do not have {n} rows - REFUSING')
    if np.asarray(det['C']).shape != (n, 2, 2):
        raise GroupingRefused('grouping: det C must be n x 2 x 2 - REFUSING')
    for k in ('ra', 'dec', 'snr', 'beam'):
        if not np.isfinite(np.asarray(det[k], float)).all():
            raise GroupingRefused(f'grouping: det {k} has non-finite values - REFUSING')
    if not np.isfinite(np.asarray(det['C'], float)).all():
        raise GroupingRefused('grouping: det C has non-finite values - REFUSING')
    return n


def _run(det, P, frozen=None):
    """the identity grouping; returns (labels, info, state) - state lets group_fork continue from the identity components"""
    _check_det(det)
    p_thr, u_cap, n0, floor, passes, rnb = P.p_thr, P.u_cap, P.n0, P.floor, P.passes, P.rnb
    ra, de, C = det['ra'], det['dec'], det['C']; n = len(ra)
    snr = np.abs(det['snr'])
    ukey, wkey = np.unique(det['key'], return_inverse=True)
    _, bnd = np.unique(det['band'], return_inverse=True)
    img = wkey * 1000 + bnd                                # an image = (window, band): fewer than 1000 bands
    kap = _kappa_of(det['band'], P)
    if len(np.unique(det['night'])) >= 1000:              # the (LST hour, night) code of _category_density holds < 1000 nights
        raise GroupingRefused(f'grouping: {len(np.unique(det["night"]))} nights - the night code supports fewer than 1000 - REFUSING')
    fro = None if frozen is None else np.asarray(frozen)
    # ---------------- stage A: within each window, across bands ----------------
    rhoA = _probe_density(ra, de, det['lst'], det['band'], wkey, True, P.probe_r['window'], 'mean', P) * P.rho_scale
    MA = _Mixture(ra, de, C, 0.0, P, n0=n0, kappa=kap, img=img, beam=det['beam'])
    byw = np.argsort(wkey, kind='stable'); bounds = np.r_[0, np.cumsum(np.bincount(wkey))]
    for w in range(len(bounds) - 1):
        idx = byw[bounds[w]:bounds[w + 1]]
        if fro is not None and (fro[idx] >= 0).all():
            wgroups = {}
            for i in idx:                                  # a frozen window: its window-sources are its (window, label) groups
                k = wgroups.get(fro[i])
                if k is None:
                    wgroups[fro[i]] = MA.new(i)
                else:
                    MA.add(i, k)
            continue
        order = idx[np.lexsort((ra[idx], de[idx], -snr[idx]))]
        if len(idx) > 1:                                   # same-window neighbors within STAGE_A_RADIUS_DEG (performance only:
            Vw = _unit(ra[idx], de[idx])                   # farther components have no density at u <= U_CAP)
            nbw = cKDTree(Vw).query_ball_point(Vw, _chord(P.rnb_a))
            nbA = {int(idx[a]): idx[r] for a, r in enumerate(nbw)}
        else:
            nbA = {int(idx[0]): idx}

        def cands(j, nbA=nbA):
            return list({MA.lab[i] for i in nbA[int(j)] if MA.lab[i] >= 0 and i != j})
        for j in order:
            _assign_pass(MA, [j], cands, rhoA, P.prof_co, p_thr, u_cap, True, P.hyst)
        for p in range(passes):
            if _assign_pass(MA, order, cands, rhoA, P.prof_co, p_thr, u_cap, False, P.hyst) == 0:
                break
    labA = MA.lab.copy()
    # ---------------- window-sources ----------------
    ua, inv = np.unique(labA, return_inverse=True); nu = len(ua)
    Ci = _inv2(C)
    best = np.full(nu, -1); bdet = np.full(nu, np.inf); dets = np.linalg.det(C)
    for i in range(n):
        if dets[i] < bdet[inv[i]]:
            bdet[inv[i]] = dets[i]; best[inv[i]] = i
    # inverse-covariance mean in the best member's tangent plane
    dx, dy = offset_en(ra[best[inv]], de[best[inv]], ra, de)
    A = np.zeros((nu, 2, 2)); np.add.at(A, inv, Ci); bb = np.zeros((nu, 2)); np.add.at(bb, inv, np.einsum('nij,nj->ni', Ci, np.c_[dx, dy]))
    mo = np.einsum('nij,nj->ni', _inv2(A), bb)
    Ude = de[best] + mo[:, 1] / 3600.0; Ura = (ra[best] + mo[:, 0] / 3600.0 / np.cos(np.radians(de[best]))) % 360
    UC = C[best]; Uw = wkey[best]; Ulst = det['lst'][best]; Uband = det['band'][best]
    Ub = np.zeros(nu); np.maximum.at(Ub, inv, snr)
    ndetU = np.bincount(inv).astype(float)
    # ---------------- stage B: across windows ----------------
    _, nidx = np.unique(det['night'], return_inverse=True)
    UL = np.array([int(x[:2]) for x in Ulst]); UN = nidx[best]
    rhoB = _category_density(Ura, Ude, Ulst, Uband, UL, UN, Uw, P) * P.rho_scale
    W0, ucls, w0_table = _seed_prior(Ura, Ude, UC, UL, UN, Uw, Ulst, Uband, ndetU, P.prof_ws, P)
    MB = _Mixture(Ura, Ude, UC, floor, P, n0=n0, slot=Uw, L=UL, N=UN, W0=W0, cls=ucls, eps=_eps_of(Ub, P) if P.tail else None)
    nbrs = cKDTree(_unit(Ura, Ude)).query_ball_point(_unit(Ura, Ude), _chord(rnb))

    def candsB(j):
        return list({MB.lab[i] for i in nbrs[j] if MB.lab[i] >= 0})
    order = np.lexsort((Ura, Ude, -Ub))
    movable = None
    if fro is not None:
        ufro = np.full(nu, -1); ufro[inv] = fro
        movable = ufro < 0
        comp_of = {}
        for u_ in np.nonzero(~movable)[0]:                # frozen units seed their old components first
            k = comp_of.get(ufro[u_])
            if k is None:
                comp_of[ufro[u_]] = MB.new(u_)
            else:
                MB.add(u_, k)
    MB.pooled = P.pooled
    _assign_pass(MB, order, candsB, rhoB, P.prof_ws, p_thr, u_cap, True, P.hyst, movable)
    seeded = _group_seed(MB, order, nbrs, P.prof_ws, rhoB, p_thr, u_cap, P.u_link, P.max_k, movable) if P.groups else 0

    def canon():
        lo = np.full(nu, nu); np.minimum.at(lo, MB.lab, np.arange(nu)); return lo[MB.lab]
    hist = []; prev = canon()
    for p in range(passes):
        MB.rebuild()
        mv = _assign_pass(MB, order, candsB, rhoB, P.prof_ws, p_thr, u_cap, False, P.hyst, movable)
        mg = _merge_pass(MB, P.prof_ws, p_thr, u_cap, rhoB, P.merge_radius, movable)
        cur = canon(); ch = int((cur != prev).sum()); prev = cur; hist.append((mv, mg, ch))
        if ch == 0:
            break
    if hist and hist[-1][1] > 0:                          # a final re-assignment: every member within the cap after the last merge
        MB.rebuild(); _assign_pass(MB, order, candsB, rhoB, P.prof_ws, p_thr, u_cap, False, P.hyst, movable)
    lab = MB.lab[inv]
    info = dict(labA=labA, inv=inv, n_windowsources=nu, passes=hist, w0_table=w0_table, rhoA=rhoA, rhoB=rhoB, groups_seeded=seeded,
                Ura=Ura, Ude=Ude, UC=UC, Uw=Uw, labU=MB.lab.copy())
    return lab, info, dict(MB=MB, inv=inv, rhoB=rhoB, nu=nu, order=order, nbrs=nbrs, movable=movable)


def group(det, p_thr=None, u_cap=None, frozen=None, return_info=False, P=None, **over):
    """IDENTITY labels of det (one int per detection; the ints are component indices - only the partition is meaningful).
    p_thr / u_cap default to config catalog_grouping.P_IDENTITY / U_CAP; frozen = labels of already-grouped detections (-1 = new;
    the incremental mode); P (a Params) or keyword overrides of its attributes for diagnostics (e.g. rho_scale, tail)."""
    P = _resolve(P, p_thr, u_cap, **over)
    lab, info, _ = _run(det, P, frozen)
    return (lab, info) if return_info else lab


def group_fork(det, p_thr=None, p_fork='config', u_cap=None, frozen=None, frozen_fork=None, tier=None, stokes=None, return_info=False,
               P=None, **over):
    """NESTED MULTIPLES LEVEL: (identity labels, multiples labels). The identities are group(det, p_thr, u_cap) exactly; the
    multiples level continues from them with MERGE-ONLY passes at the looser p_fork (group seeding among single-unit identities,
    then the unit-level merge test: every unit of the smaller component admitted to the larger at p_fork, u <= u_cap against its
    mean, no shared window) until no merge is left - it only COARSENS the identities, never re-partitions them.
    p_fork: a probability, None (no multiples level: the multiples labels are the identity labels - deep V, ruled), or 'config'
    (config catalog_grouping.P_MULTIPLES[(tier, stokes)]; tier and stokes required).
    Incremental: frozen = the identity labels of the earlier detections (-1 = new; as group()); frozen_fork = their earlier
    MULTIPLES labels: identity components holding earlier detections of one earlier multiples group are joined first (unless
    they share a window), so neither level splits a group seen before (a shared window is the one exception: see group_catalog)."""
    P = _resolve(P, p_thr, u_cap, **over)
    if isinstance(p_fork, str):
        if p_fork != 'config' or tier is None or stokes is None:
            raise GroupingRefused("grouping: p_fork='config' needs tier and stokes - REFUSING")
        p_fork = operating_point(tier, stokes)[1]
    lab, info, st = _run(det, P, frozen)
    info = dict(info, forced_joins=0, fork_seeded=0, fork_rounds=[])
    if p_fork is None:
        fork = lab.copy()
        return (lab, fork, info) if return_info else (lab, fork)
    p_fork = float(p_fork)
    MB, inv, rhoB = st['MB'], st['inv'], st['rhoB']
    MB.rebuild()
    forced = 0
    if frozen_fork is not None:
        ff = np.asarray(frozen_fork); uff = np.full(st['nu'], -1); uff[inv] = ff
        by = {}
        for u_ in np.nonzero(uff >= 0)[0]:
            by.setdefault(uff[u_], set()).add(MB.lab[u_])
        for comps in by.values():
            comps = sorted(comps, key=lambda c: -MB.cnt[c])
            k = comps[0]
            for l in comps[1:]:
                if MB.cnt[l] == 0 or MB.lab[next(iter(MB.mem[l]))] == k:
                    continue
                if set(MB.slots[k]) & set(MB.slots[l]):
                    continue
                for j in list(MB.mem[l]):
                    MB.remove(j); MB.add(j, k)
                forced += 1
    seeded = 0
    if P.groups:                                          # group evidence at the multiples level, among single-unit identities only
        MB.rebuild(); seeded = _group_seed(MB, st['order'], st['nbrs'], P.prof_ws, rhoB, p_fork, P.u_cap, P.u_link, P.max_k,
                                           st['movable'])
    rounds = []
    for r in range(P.max_rounds):
        mg = _merge_pass(MB, P.prof_ws, p_fork, P.u_cap, rhoB, P.merge_radius, st['movable'])
        rounds.append(mg)
        if mg == 0:
            break
        MB.rebuild()
    fork = MB.lab[inv]
    info.update(forced_joins=forced, fork_seeded=seeded, fork_rounds=rounds)
    return (lab, fork, info) if return_info else (lab, fork)


def operating_point(tier, stokes):
    """(P identity, P multiples or None, U_CAP) of a tier and Stokes from config (the 2026-10-26 ruling)"""
    if tier not in TIERS or stokes not in ('I', 'V'):
        raise GroupingRefused(f'grouping: no operating point for tier {tier!r} Stokes {stokes!r} - REFUSING')
    g = _cfg().catalog_grouping
    pm = g.P_MULTIPLES.get((tier, stokes), 'missing')
    if pm == 'missing':
        raise GroupingRefused(f'grouping: config catalog_grouping.P_MULTIPLES has no entry ({tier!r}, {stokes!r}) - REFUSING')
    return float(g.P_IDENTITY), (None if pm is None else float(pm)), float(g.U_CAP)


def multiplicity(det, labels, tier):
    """per detection, the number of distinct windows (10-minute: key) or nights (deep) its group holds; a multiples-level
    group with at least config catalog_grouping.MULTIPLE_MIN of them is a MULTIPLE"""
    what, _ = _cfg().catalog_grouping.MULTIPLE_MIN[tier]
    col = {'windows': 'key', 'nights': 'night'}[what]
    labels = np.asarray(labels)
    ul, li = np.unique(labels, return_inverse=True)
    pairs = np.unique(np.c_[li, np.unique(np.asarray(det[col]), return_inverse=True)[1]], axis=0)
    cnt = np.bincount(pairs[:, 0], minlength=len(ul))
    return cnt[li]


def is_multiple(det, labels, tier):
    """per detection: True when its (multiples-level) group is a multiple (MULTIPLE_MIN windows / nights)"""
    _, nmin = _cfg().catalog_grouping.MULTIPLE_MIN[tier]
    return multiplicity(det, labels, tier) >= int(nmin)


def same_partition(a, b):
    """True when label arrays a and b define the same partition (up to relabeling)"""
    a = np.asarray(a); b = np.asarray(b)
    if len(a) != len(b):
        return False
    pairs = set(zip(a.tolist(), b.tolist()))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


# =====================================================================================================================
# stable identities: IDs fixed at creation, display names, incremental assignment, the record of every change
# =====================================================================================================================
# An identity (and a multiples-level group) gets an ID when it is first created - '{level}{tier}{stokes}-{seq}' (config
# ID_FORMAT; GTI-0000001 = the first 10-minute Stokes I identity) - and keeps it for good: IDs are never reused, never renumbered.
# WHICH DETECTION IS WHICH (match_state): production re-numbers candidates when a cell is re-searched (exopipe_ops
# p3s3_merge.py:77-81; Run_SLSTB 2026-10-25 vs 10-31: 1,562 of 67,317 shared cand_ids name a detection more than 1 deg away) and
# re-measures the detections it finds again. A new detection IS a state detection when it has the same cand_id and fingerprint
# (window key, band, catalog ra, dec, S/N at full precision: fingerprints()) - the fast path - or, failing that, the same window
# and band with the positions within max(0.5 x the larger fitted position error, 0.05 x the larger FWHM) and the S/N of one
# sign within a factor 1.3 (config ID_MATCH_*), mutually and uniquely ('remeasured'; it may carry another cand_id). Two or more
# candidates within tolerance on either side match none ('ambiguous'). Anything else is a NEW detection and the state's is
# ABSENT; a cand_id of the state beyond tolerance of its own record is 'reused' and never carries the old detection's IDs.
# Carrying IDs to a new partition (_carry, at each level): the HOME of a previous ID is the new group holding most of its
# detections; a group that is the home of several previous IDs keeps the OLDEST (lowest sequence number) and the others are
# recorded as merged into it (status 'merged', 'into' - labels attached to an absorbed ID follow 'into'); a group that is the
# home of none gets a new ID; a previous ID whose detections now lie in several groups is SPLIT ('split' event; refused at the
# identity level of an incremental run). EVERY detection whose ID changes gets a 'move' event (from, to, reason 'merge' |
# 'split' | 'moved'); an ID left without detections is 'retired'. The display NAME is production's J-name
# (build_source_catalogs_v3.jname) of the identity's mean position (inverse-covariance mean of its members, common frame),
# made unique within the catalog as production does (JHHMMSS+DDMMSS for a shared name; a letter suffix where even that is
# shared); names may change between runs - IDs do not.

def _jname_many(ra, dec):
    """production's jname (exopipe_ops build_source_catalogs_v3.jname), vectorized: JHHMM+DDMM with its '{:02.0f}' rounding"""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    if len(ra) == 0:
        return []
    c = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg)
    h, m = c.ra.hms.h, c.ra.hms.m; dd, dm = c.dec.dms.d, c.dec.dms.m
    return ['J{:02.0f}{:02.0f}{}{:02.0f}{:02.0f}'.format(a, abs(b), '+' if e >= 0 else '-', abs(x), abs(y))
            for a, b, e, x, y in zip(np.atleast_1d(h), np.atleast_1d(m), np.asarray(dec), np.atleast_1d(dd), np.atleast_1d(dm))]


def _jname_long(ra, dec):
    """JHHMMSS+DDMMSS (build_source_catalogs_v3.jname_long)"""
    ra, dec = float('%.5f' % ra), float('%.5f' % dec)
    ts = int(round((ra % 360.0) / 15.0 * 3600.0)) % (24 * 3600)
    h, r = divmod(ts, 3600); m, sec = divmod(r, 60)
    dd, r = divmod(int(round(abs(dec) * 3600.0)), 3600); dm, dsec = divmod(r, 60)
    return 'J%02d%02d%02d%s%02d%02d%02d' % (h, m, sec, '+' if dec >= 0 else '-', dd, dm, dsec)


def display_names(ra, dec):
    """unique display names of positions, in the given order: production's jname; a name shared by two or more becomes
    JHHMMSS+DDMMSS (production's unique_names); one still shared takes a letter suffix in order (production refuses there - a
    display name never stops the grouping). Returns (names, number of letter suffixes)."""
    short = _jname_many(ra, dec)
    n = collections.Counter(short)
    names = [_jname_long(a, d) if n[s] > 1 else s for s, a, d in zip(short, ra, dec)]
    c2, seen, out, clash = collections.Counter(names), collections.Counter(), [], 0
    for x in names:
        if c2[x] > 1:
            seen[x] += 1; out.append(x + _suffix(seen[x] - 1)); clash += 1
        else:
            out.append(x)
    return out, clash


def _suffix(i):
    s = ''
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = 'abcdefghijklmnopqrstuvwxyz'[r] + s
    return s


def mean_position(ra, dec, C, snr):
    """inverse-covariance mean (ra, dec deg) of members (common frame), in the brightest member's tangent plane - for NAMES only
    (independent errors: far too optimistic as a localization error, which stays the catalog's localization step)"""
    a = int(np.argmax(np.abs(snr)))
    dx, dy = offset_en(np.full(len(ra), ra[a]), np.full(len(ra), dec[a]), ra, dec)
    X = np.c_[dx, dy]; Ci = np.linalg.inv(C); A = Ci.sum(0); Cm = np.linalg.inv(A); m = Cm @ np.einsum('nij,nj->i', Ci, X)
    de_ = dec[a] + m[1] / 3600.0; ra_ = (ra[a] + m[0] / 3600.0 / math.cos(math.radians(dec[a]))) % 360
    return float(ra_), float(de_)


def format_id(level, tier, stokes, seq):
    g = _cfg().catalog_grouping
    return g.ID_FORMAT.format(level=g.ID_LEVEL_CODE[level], tier=g.ID_TIER_CODE[tier], stokes=stokes, seq=int(seq))


def _seq_of(i):
    return int(str(i).rsplit('-', 1)[-1])


FP_FIELDS = ('key', 'band', 'ra', 'dec', 'snr')
MATCH_FIELDS = ('err_arcsec', 'fwhm_arcmin')          # stored with each assignment for the tolerance match
STATE_FORMAT = 'exopipe3-grouping-state-2'


def fingerprints(det):
    """each detection's fingerprint - what identifies it beyond its cand_id, which production re-numbers on a re-search: (window
    key, band, catalog ra, catalog dec, S/N), the floats as their shortest exact repr (ra_raw / dec_raw, the catalog's own
    positions, when det has them - grouping_inputs keeps them - else ra / dec)"""
    ra = det['ra_raw'] if 'ra_raw' in det else det['ra']
    de = det['dec_raw'] if 'dec_raw' in det else det['dec']
    return [(str(k), str(b), repr(float(a)), repr(float(d)), repr(float(s)))
            for k, b, a, d, s in zip(det['key'], det['band'], ra, de, det['snr'])]


def match_errors(det):
    """per detection, (its own fitted position error, arcsec - det['stat'] = pos_stat_maj_arcsec, NaN when det has none; the
    frame's FWHM, arcmin - det['beam']), as stored for the tolerance match"""
    n = len(det['ra'])
    err = np.asarray(det['stat'], float) if 'stat' in det else np.full(n, np.nan)
    return err, np.asarray(det['beam'], float)


class GroupingState:
    """Everything a grouping run hands to the next one for a (tier, Stokes): every ID ever issued (never reused) with its status,
    and each detection's identity and multiples IDs with its fingerprint and its match errors.
      ids[ID] = {'level', 'created' (run label), 'status' ('live' | 'merged' | 'retired'), 'into' (merged: the surviving ID),
                 'last_run', 'n_det', 'name' (identity) / 'multiple' (multiples level)}
      assign[cand_id] = (identity ID, multiples ID, fingerprint, err, fwhm)   fingerprint = (key, band, ra, dec, snr) strings
                 (fingerprints()); err (arcsec) and fwhm (arcmin) as strings (match_errors()) for the tolerance match
      next_seq[level] = the next sequence number of that level
      runs = the summary of every run
    Persisted in ONE file, grouping_<tier>_<S>_state.json, replaced atomically (temporary file, fsync, rename): save(dir).
    load(dir, tier, stokes) refuses a directory without that file unless new=True (the first run), and new=True where one
    exists: IDs never restart at 1 silently."""

    def __init__(self, tier, stokes, ids=None, assign=None, next_seq=None, runs=None):
        if tier not in TIERS or stokes not in ('I', 'V'):
            raise GroupingRefused(f'grouping: no state for tier {tier!r} Stokes {stokes!r} - REFUSING')
        self.tier, self.stokes = tier, stokes
        self.ids = dict(ids or {})
        self.assign = dict(assign or {})
        self.next_seq = dict(next_seq or {lv: 1 for lv in LEVELS})
        self.runs = list(runs or [])

    def issue(self, level, run_label):
        i = format_id(level, self.tier, self.stokes, self.next_seq[level])
        self.next_seq[level] += 1
        if i in self.ids:
            raise GroupingRefused(f'grouping: ID {i} was issued before (state next_seq damaged) - REFUSING')
        self.ids[i] = dict(level=level, created=run_label, status='live', into='', last_run=run_label, n_det=0)
        return i

    def resolve(self, i):
        """the live ID an ID stands for now (following 'into' of merged IDs)"""
        seen = set()
        while self.ids.get(i, {}).get('status') == 'merged':
            if i in seen:
                raise GroupingRefused(f'grouping: merge cycle at {i} - REFUSING')
            seen.add(i); i = self.ids[i]['into']
        return i

    def path(self, directory):
        return os.path.join(directory, f'grouping_{self.tier}_{self.stokes}_state.json')

    def save(self, directory):
        """write the state as ONE file, atomically (a reader sees the old state or the new one, never a mix); returns its path"""
        os.makedirs(directory, exist_ok=True)
        p = self.path(directory)
        doc = dict(format=STATE_FORMAT, tier=self.tier, stokes=self.stokes, next_seq=self.next_seq, runs=self.runs, ids=self.ids,
                   fingerprint_fields=list(FP_FIELDS), match_fields=list(MATCH_FIELDS),
                   assign={c: [i, m] + list(fp) + [e, w] for c, (i, m, fp, e, w) in self.assign.items()})
        tmp = f'{p}.tmp.{os.getpid()}'
        with open(tmp, 'w') as f:
            json.dump(doc, f, sort_keys=True, separators=(',', ':'), default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        return p

    @classmethod
    def load(cls, directory, tier, stokes, new=False):
        """the saved state of (tier, stokes) in directory. No state there: REFUSES unless new=True (then an empty state - the
        first run); new=True where a state exists REFUSES too. A damaged or foreign state REFUSES."""
        st = cls(tier, stokes)
        p = st.path(directory)
        if not os.path.exists(p):
            if new:
                return st
            raise GroupingRefused(f'grouping: no grouping state {p} - the first run of {tier} {stokes} must say new=True (IDs never '
                                  f'restart at 1 silently) - REFUSING')
        if new:
            raise GroupingRefused(f'grouping: new=True but {p} exists - a new state would restart the IDs at 1 - REFUSING')
        with open(p) as f:
            d = json.load(f)
        if d.get('format') != STATE_FORMAT or d.get('tier') != tier or d.get('stokes') != stokes:
            raise GroupingRefused(f'grouping: {p} is not a {STATE_FORMAT} state of {tier} {stokes} (format {d.get("format")!r}, '
                                  f'{d.get("tier")} {d.get("stokes")}) - REFUSING')
        if tuple(d.get('fingerprint_fields') or ()) != FP_FIELDS or tuple(d.get('match_fields') or ()) != MATCH_FIELDS:
            raise GroupingRefused(f'grouping: {p} fields {d.get("fingerprint_fields")} + {d.get("match_fields")}, not '
                                  f'{list(FP_FIELDS)} + {list(MATCH_FIELDS)} - REFUSING')
        assign, nf = {}, len(FP_FIELDS)
        for c, v in d['assign'].items():
            if not (isinstance(v, list) and len(v) == 2 + nf + len(MATCH_FIELDS) and all(isinstance(x, str) for x in v)):
                raise GroupingRefused(f'grouping: {p}: the assignment of {c} is malformed ({v!r}) - REFUSING')
            assign[c] = (v[0], v[1], tuple(v[2:2 + nf]), v[2 + nf], v[3 + nf])
        st = cls(tier, stokes, ids=d['ids'], assign=assign, next_seq=d['next_seq'], runs=d.get('runs', []))
        for c, (i, m, *_) in assign.items():
            for x, lv in ((i, 'identity'), (m, 'multiples')):
                e = st.ids.get(x)
                if e is None or e.get('level') != lv or e.get('status') != 'live':
                    raise GroupingRefused(f'grouping: {p}: {c} is assigned to {x}, not a live {lv} ID - REFUSING')
        for lv in LEVELS:
            top = max((_seq_of(i) for i, e in st.ids.items() if e.get('level') == lv), default=0)
            if int(st.next_seq.get(lv, 0)) <= top:
                raise GroupingRefused(f'grouping: {p}: next_seq[{lv}] {st.next_seq.get(lv)} would re-issue ID {top} - REFUSING')
        return st


class GroupingResult:
    """The outcome of group_catalog for one (tier, Stokes): per detection the identity and multiples-level labels and IDs, the
    multiplicity, the display names, the record of this run (events, summary) and the next state."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def rows(self):
        """one dict per detection (det order): cand_id, identity_id, identity_name, multiples_id, multiples_name, multiplicity (the
        multiples group's windows; deep: nights), multiple (0/1)"""
        out = []
        for j in range(len(self.cand_id)):
            out.append(dict(cand_id=self.cand_id[j], identity_id=self.identity_id[j], identity_name=self.names[self.identity_id[j]],
                            multiples_id=self.multiples_id[j], multiples_name=self.names[self.multiples_id[j]],
                            multiplicity=int(self.multiplicity[j]), multiple=int(self.multiple[j])))
        return out


def _sep_arcsec(ra1, de1, ra2, de2):
    a1, d1, a2, d2 = map(np.radians, (ra1, de1, ra2, de2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(a1 - a2)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0))) * 3600.0


def _within(old_rec, fp_new, err_new, fwhm_new, g):
    """(within tolerance, separation arcsec, tolerance arcsec, |S/N| ratio) of a state record against a new detection: the same
    window key and band, separation <= max(POS_ERR_FRACTION x the larger fitted error, POS_FWHM_FRACTION x the larger FWHM),
    S/N of one sign within SNR_FACTOR"""
    fo = old_rec[2]
    if fo[0] != fp_new[0] or fo[1] != fp_new[1]:
        return False, float('nan'), float('nan'), float('nan')
    sep = float(_sep_arcsec(float(fo[2]), float(fo[3]), float(fp_new[2]), float(fp_new[3])))
    err = np.fmax(float(old_rec[3]), err_new); fw = np.fmax(float(old_rec[4]), fwhm_new)
    tol = float(np.fmax(g.ID_MATCH_POS_ERR_FRACTION * err, g.ID_MATCH_POS_FWHM_FRACTION * fw * 60.0))
    so, sn = float(fo[4]), float(fp_new[4])
    ratio = max(abs(so), abs(sn)) / min(abs(so), abs(sn)) if so * sn > 0 else float('inf')
    return bool(sep <= tol and ratio <= g.ID_MATCH_SNR_FACTOR), sep, tol, ratio


class Matching:
    """which detections of det ARE detections of the state (match_state): prev[j] = (identity ID, multiples ID) or None;
    matched {new cand_id: (state cand_id, 'exact' | 'tolerance')}; remeasured [(j, state cand_id, sep, tol, ratio)];
    reused [(j, state record)] - a cand_id of the state beyond tolerance of its own record; ambiguous [{'new', 'old', 'pairs'}];
    absent [state cand_ids matched by no detection]"""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def match_state(det, state, tolerance=None):
    """Match the detections of det to the state's: EXACT (the same cand_id with the same fingerprint) first; then, when tolerance
    (default config catalog_grouping.ID_MATCH_TOLERANCE), the leftovers of each window key and band by TOLERANCE (_within: a
    re-search re-measures, and may re-number, the same detection). A tolerance match must be MUTUAL AND UNIQUE: a new detection
    with one candidate whose only candidate it is. Every connected group of candidates with a second candidate on either side
    matches none and is recorded ('ambiguous'). A new detection whose cand_id the state holds with another fingerprint and
    beyond tolerance of that record is 'reused' (the cand_id names another detection now). Only the ID carry depends on this,
    never the partition (except which windows the incremental mode freezes)."""
    g = _cfg().catalog_grouping
    tolerance = bool(g.ID_MATCH_TOLERANCE) if tolerance is None else bool(tolerance)
    cand_id = np.asarray(det['cand_id']).astype(str); n = len(cand_id)
    fp = fingerprints(det); err, fwhm = match_errors(det)
    old = state.assign
    prev = [None] * n; matched = {}; used = set()
    for j, c in enumerate(cand_id.tolist()):
        a = old.get(c)
        if a is not None and tuple(a[2]) == fp[j]:
            prev[j] = (a[0], a[1]); matched[c] = (c, 'exact'); used.add(c)
    remeasured, ambiguous = [], []
    if tolerance:
        new_left, old_left = collections.defaultdict(list), collections.defaultdict(list)
        for j, c in enumerate(cand_id.tolist()):
            if c not in matched:
                new_left[(fp[j][0], fp[j][1])].append(j)
        for c, a in old.items():
            if c not in used:
                old_left[(a[2][0], a[2][1])].append(c)
        for kb in sorted(set(new_left) & set(old_left)):
            edges = []
            for j in new_left[kb]:
                for c in old_left[kb]:
                    ok, sep, tol, ratio = _within(old[c], fp[j], err[j], fwhm[j], g)
                    if ok:
                        edges.append((j, c, sep, tol, ratio))
            if not edges:
                continue
            dn = collections.Counter(e[0] for e in edges); do = collections.Counter(e[1] for e in edges)
            parent = {}

            def find(x):
                while parent.setdefault(x, x) != x:
                    parent[x] = parent[parent[x]]; x = parent[x]
                return x
            for j, c, *_ in edges:
                parent[find(('n', j))] = find(('o', c))
            comp = collections.defaultdict(list)
            for e in edges:
                comp[find(('n', e[0]))].append(e)
            for es in comp.values():
                if all(dn[j] == 1 and do[c] == 1 for j, c, *_ in es):          # one edge: a unique mutual match
                    j, c, sep, tol, ratio = es[0]
                    a = old[c]; prev[j] = (a[0], a[1]); matched[cand_id[j]] = (c, 'tolerance'); used.add(c)
                    remeasured.append((j, c, sep, tol, ratio))
                else:
                    ambiguous.append(dict(new=sorted({str(cand_id[j]) for j, *_ in es}), old=sorted({c for _, c, *_ in es}),
                                          pairs=[[str(cand_id[j]), c, round(sep, 2), round(tol, 2), round(ratio, 4)]
                                                 for j, c, sep, tol, ratio in sorted(es, key=lambda e: (e[0], e[1]))]))
    reused = []
    for j, c in enumerate(cand_id.tolist()):
        a = old.get(c)
        if a is None or matched.get(c) == (c, 'exact') or matched.get(c) == (c, 'tolerance'):
            continue
        if not (tolerance and _within(a, fp[j], err[j], fwhm[j], g)[0]):     # within tolerance but unmatched = ambiguous
            reused.append((j, a))
    absent = sorted(c for c in old if c not in used)
    return Matching(prev=prev, matched=matched, remeasured=remeasured, reused=reused, ambiguous=ambiguous, absent=absent,
                    tolerance=tolerance, fp=fp, err=err, fwhm=fwhm)


def _carry(level, labels, prev_ids, cand_id, snr, state, run_label, allow_split):
    """IDs for the groups of `labels` at one level, carried from the detections' previous IDs (prev_ids: each detection's ID in
    the state, '' = none - a new detection, or a cand_id the state knew with another fingerprint). Returns ({label: ID}, events).
    HOME of a previous ID: the new group holding most of its detections (a tie: the one holding its smallest cand_id - content
    only, never the internal label values). A group that is the home of no previous ID gets a new ID (brightest member |S/N|
    first, then the smallest cand_id); one that is the home of several keeps the OLDEST (lowest sequence) and the others are
    recorded as merged into it ('merge' event; state 'merged', 'into'). A previous ID whose detections lie in several groups is
    SPLIT: refused unless allow_split (the identity level of an incremental run), else a 'split' event (the ID, its home's ID,
    the pieces). EVERY detection whose ID changes gets a 'move' event: reason 'merge' (its ID was absorbed into the new one),
    'split' (it lies in a piece split off its ID that got a new ID), 'moved' (it joined another existing ID while its own ID
    lives on in its home)."""
    labels = np.asarray(labels)
    comps = collections.defaultdict(list)
    for j, c in enumerate(labels.tolist()):
        comps[c].append(j)
    pieces = collections.defaultdict(collections.Counter)      # previous ID -> {group: detections}
    first = collections.defaultdict(dict)                      # previous ID -> {group: its smallest cand_id there}
    for j, p in enumerate(prev_ids):
        if p:
            if state.ids.get(p, {}).get('status') != 'live' or state.ids[p].get('level') != level:
                raise GroupingRefused(f'grouping: {cand_id[j]} carries {p}, not a live {level} ID of the state - REFUSING')
            c = labels[j]
            pieces[p][c] += 1
            if c not in first[p] or cand_id[j] < first[p][c]:
                first[p][c] = cand_id[j]
    split = sorted((p for p, cnt in pieces.items() if len(cnt) > 1), key=_seq_of)
    if split and not allow_split:
        p = split[0]
        raise GroupingRefused(f'grouping: {len(split)} {level} ID(s) would be SPLIT by an incremental run (e.g. {p} into '
                              f'{len(pieces[p])} pieces) - the incremental mode never splits an identity; REFUSING')
    home = {}
    for p, cnt in pieces.items():
        mx = max(cnt.values())
        home[p] = min((c for c, v in cnt.items() if v == mx), key=lambda c: first[p][c])
    claims = collections.defaultdict(list)
    for p, c in home.items():
        claims[c].append(p)
    out, issued, events = {}, set(), []
    key = {c: (-float(np.max(np.abs(snr[js]))), min(cand_id[j] for j in js)) for c, js in comps.items()}
    for c in sorted(comps, key=lambda c: key[c]):
        ps = sorted(claims.get(c, []), key=_seq_of)
        if ps:
            out[c] = ps[0]
            if len(ps) > 1:
                for q in ps[1:]:
                    state.ids[q].update(status='merged', into=ps[0], last_run=run_label)
                events.append(dict(run=run_label, level=level, event='merge', id=ps[0], absorbed=ps[1:],
                                   n_det=len(comps[c]), n_det_previous={q: int(pieces[q][c]) for q in ps},
                                   n_det_new=int(sum(1 for j in comps[c] if not prev_ids[j]))))
        else:
            out[c] = state.issue(level, run_label)
            issued.add(out[c])
    for p in split:
        events.append(dict(run=run_label, level=level, event='split', id=p, home_id=out[home[p]],
                           pieces={out[c]: int(v) for c, v in sorted(pieces[p].items(), key=lambda cv: out[cv[0]])}))
    for j, p in enumerate(prev_ids):
        if not p:
            continue
        c = labels[j]; to = out[c]
        if to == p:
            continue
        reason = 'merge' if home[p] == c else ('split' if to in issued else 'moved')
        events.append(dict(run=run_label, level=level, event='move', cand_id=str(cand_id[j]), from_id=p, to_id=to, reason=reason))
    return out, events


def group_catalog(det, tier, stokes, state=None, run_label='', mode=None, P=None, match_tolerance=None):
    """The catalog step's entry point for one tier and Stokes: group every detection of det (all bands, nights, LSTs) at the
    ruled operating point, give each identity and multiples-level group its stable ID, name it, count its windows (nights) and
    record every change of an ID (events: 'merge', 'split', 'move' - every detection whose ID changes, 'remeasured' - a
    detection matched to the state's by tolerance, 'reused' - a cand_id of the state that names another detection now,
    'ambiguous' - candidates within tolerance that match none, 'retired'). det needs cand_id (unique) besides the fields group()
    reads, and holds EVERY detection of the tier and Stokes - the state's and the new ones. Which detections ARE the state's is
    match_state(): the same cand_id with the same fingerprint, else (match_tolerance, default config ID_MATCH_TOLERANCE) the same
    window and band within the position and S/N tolerance, mutual and unique; anything else is a new detection, and a state
    detection matched by none is absent (dropped from the assignments, n_det_state_absent); an ID left without detections
    retires. The partition never depends on the matching (batch, regroup); the incremental mode freezes the matched detections.
    state: a GroupingState (the IDs of earlier runs) or None (the first run: every ID new).
    mode: 'batch' (no state: every ID new - refused when the state holds IDs),
          'incremental' (default with a non-empty state; PURE APPENDS: the state's detections keep their identities - frozen -
          and only the new ones are assigned; a window holding both kinds REFUSES - a re-search forces 'regroup' - and an
          identity split refuses; the multiples level may split an earlier group, recorded),
          'regroup' (the whole partition recomputed, IDs carried by member detections - matched as above; every merge,
          split and move recorded).
    Returns a GroupingResult (its .state is the next run's state; the state passed in is not modified)."""
    p_thr, p_fork, u_cap = operating_point(tier, stokes)
    P = _resolve(P, p_thr, u_cap)
    n = _check_det(det)
    if 'cand_id' not in det or len(det['cand_id']) != n:
        raise GroupingRefused('grouping: det needs cand_id for the IDs - REFUSING')
    cand_id = np.asarray(det['cand_id']).astype(str)
    if len(set(cand_id.tolist())) != n:
        raise GroupingRefused('grouping: cand_id is not unique - REFUSING')
    snr = np.asarray(det['snr'], float)
    old = state if state is not None else GroupingState(tier, stokes)
    if (old.tier, old.stokes) != (tier, stokes):
        raise GroupingRefused(f'grouping: state of {old.tier} {old.stokes} given for {tier} {stokes} - REFUSING')
    st = GroupingState(tier, stokes, ids=json.loads(json.dumps(old.ids)), assign={}, next_seq=dict(old.next_seq),
                       runs=list(old.runs))
    if mode is None:
        mode = 'incremental' if old.assign else 'batch'
    if mode not in ('batch', 'incremental', 'regroup'):
        raise GroupingRefused(f'grouping: mode {mode!r} is not batch, incremental or regroup - REFUSING')
    if mode == 'batch' and old.ids:
        raise GroupingRefused('grouping: a batch run would re-issue every ID although the state holds IDs - use mode '
                              "'incremental' or 'regroup' (IDs are never reused) - REFUSING")
    mt = match_state(det, GroupingState(tier, stokes) if mode == 'batch' else old, match_tolerance)
    fp = mt.fp
    prev_i = [x[0] if x else '' for x in mt.prev]; prev_m = [x[1] if x else '' for x in mt.prev]
    reused = mt.reused
    frozen = frozen_fork = None
    if mode == 'incremental':
        is_old = np.array([bool(x) for x in prev_i], bool)
        keys = np.asarray(det['key'])
        mixed = sorted(set(keys[is_old].tolist()) & set(keys[~is_old].tolist()))
        if mixed:
            ms = set(mixed)
            nre = sum(1 for j, _ in reused if keys[j] in ms)
            pos_of = {c: j for j, c in enumerate(cand_id.tolist())}
            nam = sum(1 for a in mt.ambiguous for c in a['new'] if keys[pos_of[c]] in ms)
            raise GroupingRefused(f'grouping: {len(mixed)} window(s) hold both detections of the state and new ones (e.g. {mixed[0]}'
                                  + (f'; {nre} of the new ones reuse a cand_id of the state for another detection - a re-search' if nre
                                     else '') + (f'; {nam} are ambiguous within tolerance' if nam else '')
                                  + "): the incremental mode is for pure appends (new windows) - run mode 'regroup' - REFUSING")
        codes_i = {x: k for k, x in enumerate(sorted({x for x in prev_i if x}))}
        codes_m = {x: k for k, x in enumerate(sorted({x for x in prev_m if x}))}
        frozen = np.array([codes_i[x] if x else -1 for x in prev_i], int)
        frozen_fork = np.array([codes_m[x] if x else -1 for x in prev_m], int)
    ident, fork, info = group_fork(det, p_fork=p_fork, frozen=frozen, frozen_fork=frozen_fork, return_info=True, P=P)
    ident = np.asarray(ident); fork = np.asarray(fork)
    # nesting: the multiples level never splits an identity (by construction; checked)
    pairs = set(zip(ident.tolist(), fork.tolist()))
    if len({a for a, _ in pairs}) != len(pairs):
        raise GroupingRefused('grouping: the multiples level split an identity - REFUSING')
    events = []
    id_of, ev = _carry('identity', ident, prev_i, cand_id, snr, st, run_label, allow_split=(mode != 'incremental'))
    events += ev
    # the multiples level: in an incremental run an earlier multiples group can split where two of its identities now share a
    # window (one window-source per window per group wins; research group_B.py:639-645): allowed, recorded ('split', 'move')
    mid_of, ev = _carry('multiples', fork, prev_m, cand_id, snr, st, run_label, allow_split=True)
    events += ev
    identity_id = np.array([id_of[c] for c in ident.tolist()], dtype=object)
    multiples_id = np.array([mid_of[c] for c in fork.tolist()], dtype=object)
    for j, c, sep, tol, ratio in mt.remeasured:
        a = old.assign[c]
        events.append(dict(run=run_label, event='remeasured', cand_id=str(cand_id[j]), cand_id_old=c, fingerprint_old=list(a[2]),
                           fingerprint_new=list(fp[j]), sep_arcsec=round(sep, 3), tol_arcsec=round(tol, 3), snr_ratio=round(ratio, 4),
                           identity=identity_id[j], multiples=multiples_id[j]))
    for j, a in reused:
        m = mt.matched.get(str(cand_id[j]))
        events.append(dict(run=run_label, event='reused', cand_id=str(cand_id[j]), fingerprint_old=list(a[2]), fingerprint_new=list(fp[j]),
                           from_identity=a[0], from_multiples=a[1], to_identity=identity_id[j], to_multiples=multiples_id[j],
                           matched_to=m[0] if m else ''))
    pos_of = {c: j for j, c in enumerate(cand_id.tolist())}
    for a in mt.ambiguous:
        events.append(dict(run=run_label, event='ambiguous', new=a['new'], old=a['old'], pairs=a['pairs'],
                           old_identity=sorted({old.assign[c][0] for c in a['old']}),
                           new_identity=sorted({identity_id[pos_of[c]] for c in a['new']})))
    # retired: previous live IDs with no detection now
    present = set(identity_id.tolist()) | set(multiples_id.tolist())
    for i, e in st.ids.items():
        if e['status'] == 'live' and i not in present:
            e.update(status='retired', last_run=run_label)
            events.append(dict(run=run_label, level=e['level'], event='retired', id=i))
    # names: identities by their mean position; a multiples group by its brightest identity
    members = collections.defaultdict(list)
    for j, i in enumerate(identity_id.tolist()):
        members[i].append(j)
    order_ids = sorted(members, key=_seq_of)
    pos = [mean_position(det['ra'][members[i]], det['dec'][members[i]], det['C'][members[i]], snr[members[i]]) for i in order_ids]
    nm, clash = display_names([p[0] for p in pos], [p[1] for p in pos])
    names = dict(zip(order_ids, nm))
    best = {i: float(np.max(np.abs(snr[members[i]]))) for i in order_ids}
    mult = multiplicity(det, fork, tier)
    is_mult = mult >= int(_cfg().catalog_grouping.MULTIPLE_MIN[tier][1])
    mmembers = collections.defaultdict(set)
    for j, (i, m) in enumerate(zip(identity_id.tolist(), multiples_id.tolist())):
        mmembers[m].add(i)
    for m, ids_ in mmembers.items():
        names[m] = names[max(ids_, key=lambda i: (best[i], -_seq_of(i)))]
    for j, c in enumerate(cand_id.tolist()):
        st.assign[c] = (identity_id[j], multiples_id[j], fp[j], repr(float(mt.err[j])), repr(float(mt.fwhm[j])))
    ndet = collections.Counter(identity_id.tolist()); ndet.update(multiples_id.tolist())
    for i in present:
        st.ids[i].update(last_run=run_label, n_det=int(ndet[i]))
        if st.ids[i]['level'] == 'identity':
            st.ids[i]['name'] = names[i]
    for j, m in enumerate(multiples_id.tolist()):
        st.ids[m]['multiple'] = bool(is_mult[j])
    n_new = {lv: sum(1 for i in present if st.ids[i]['level'] == lv and st.ids[i]['created'] == run_label and i not in old.ids)
             for lv in LEVELS}
    absent = len(mt.absent) if mode != 'batch' else 0
    gen = det.get('frame_generation')
    gen_prev = next((r.get('frame_generation') for r in reversed(old.runs) if r.get('frame_generation') is not None), None)
    cnt = collections.Counter((e.get('level'), e['event']) for e in events)
    summary = dict(run=run_label, tier=tier, stokes=stokes, mode=mode, p_identity=p_thr, p_multiples=p_fork, u_cap=u_cap,
                   n_det=n, n_det_previous=int(sum(1 for x in prev_i if x)), n_det_state_absent=int(absent),
                   n_matched_exact=int(sum(1 for v in mt.matched.values() if v[1] == 'exact')),
                   n_matched_tolerance=len(mt.remeasured), n_cand_id_reused=len(reused),
                   n_ambiguous=dict(groups=len(mt.ambiguous), new=int(sum(len(a['new']) for a in mt.ambiguous)),
                                    old=int(sum(len(a['old']) for a in mt.ambiguous))),
                   match_tolerance=(dict(pos_err_fraction=_cfg().catalog_grouping.ID_MATCH_POS_ERR_FRACTION,
                                         pos_fwhm_fraction=_cfg().catalog_grouping.ID_MATCH_POS_FWHM_FRACTION,
                                         snr_factor=_cfg().catalog_grouping.ID_MATCH_SNR_FACTOR) if mt.tolerance else None),
                   identities=len(members), multiples_groups=len(mmembers),
                   multiples=int(len({m for m, f in zip(multiples_id.tolist(), is_mult.tolist()) if f})), new_ids=n_new,
                   merges={lv: cnt[(lv, 'merge')] for lv in LEVELS}, splits={lv: cnt[(lv, 'split')] for lv in LEVELS},
                   moves={lv: cnt[(lv, 'move')] for lv in LEVELS}, retired={lv: cnt[(lv, 'retired')] for lv in LEVELS},
                   name_suffixes=clash, window_sources=int(info['n_windowsources']), groups_seeded=int(info['groups_seeded']),
                   forced_joins=int(info['forced_joins']), frame_generation=gen,
                   frame_generation_changed=(None if gen is None or gen_prev is None else gen != gen_prev), research_md5=RESEARCH_MD5,
                   incremental_rule=INCREMENTAL_RULE if mode == 'incremental' else None)
    st.runs.append(summary)
    return GroupingResult(cand_id=cand_id, identity=ident, multiples=fork, identity_id=identity_id, multiples_id=multiples_id,
                          names=names, multiplicity=mult, multiple=is_mult, events=events, summary=summary, state=st, info=info,
                          matched=dict(mt.matched))


def append_events(path, result):
    """append this run's summary and every event (merge, split, move, remeasured, reused, ambiguous, retired) to a JSON-lines
    record (one object per line)"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(dict(result.summary, event='run'), sort_keys=True, default=str) + '\n')
        for e in result.events:
            f.write(json.dumps(e, sort_keys=True, default=str) + '\n')
