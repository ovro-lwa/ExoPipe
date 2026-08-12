#!/usr/bin/env python3
"""candinv/config.py — all paths and tunables for the investigation harness.

Every path is overridable by an environment variable (CANDINV_<NAME>) so the
harness is node-portable; the defaults target lwacalim07 where the PROD survey
products live.  The actual code (this package) lives on shared /lustre, so we
edit on any node and run where the data is.
"""
import os

from exopipe.common import phase2_config as cfg


def _env(name, default):
    return os.environ.get(f'CANDINV_{name}', default)


# --- Survey inputs (lwacalim07, node-local) --------------------------------
TRANSIENTS_DIR = _env('TRANSIENTS_DIR', '/fast/gh/main/phase3/transients')
TOP50_CSV = _env('TOP50_CSV',
                 os.path.join(TRANSIENTS_DIR, 'wideband_V_search',
                              'top50_V_transients.csv'))
# Gathered per-candidate FITS (discovery V residual + I residual + I master).
BUNDLE_DIR = _env('BUNDLE_DIR', '/data07/gh/top50_allsky_tmp')

# PROD per-epoch conformant tree (subband _msub spectra, 10-min bins, aligned I).
PROD_ROOT = _env('PROD_ROOT', '/data07/gh/phase3')
PROD_LABEL = _env('PROD_LABEL', 'PROD')
# REDO1 holds the WIDEBAND DEEP (Red/Green/Blue/Full) V+I deep aligned/msub that
# PROD lacks (project_rank48_and_prod_redo1). Source wideband-deep residuals from
# here; subbands + all 10-min from PROD. Same /data07 tree, different P3_ label.
REDO1_LABEL = _env('REDO1_LABEL', 'REDO1')

# Phase-2 archive (sky images for the I counterpart; epoch times for headers).
PHASE2_ARCHIVE = _env('PHASE2_ARCHIVE', '/lustre/pipeline/exopipe/phase2')
# Phase-2 NVMe staging — holds Dewarp_Diagnostics/maps/*_warp_screen.fits and
# the 10-second snapshots used by the T5 escalation.
PHASE2_FAST = _env('PHASE2_FAST', '/fast/gh/main/phase2')

# --- Output (conformant dossier home) --------------------------------------
# phase3_paths.candidate_dir(OUT_ROOT, OUT_LABEL, cand_id) ->
#   {OUT_ROOT}/transients/Run_{OUT_LABEL}/candidates/{cand_id}/
OUT_ROOT = _env('OUT_ROOT', '/fast/gh/main/phase3')
OUT_LABEL = _env('OUT_LABEL', 'PROD')
# candinv writes to a DEDICATED subdir of the run, NOT the shared `candidates/`
# tree (which the Phase-3 transient SEARCH fills with ~223k I*/V* detection dirs
# — candinv's 50 J*_rank* dossiers were getting buried there). Gregg 2026-06-28.
OUT_SUBDIR = _env('OUT_SUBDIR', 'candinv')


def candidate_dir(cand_id, make=False):
    """Dedicated candinv dossier dir: {OUT_ROOT}/transients/Run_{LABEL}/candinv/{id}/."""
    from exopipe.common import phase3_paths as _P
    d = os.path.join(_P.transients_run(OUT_ROOT, OUT_LABEL, make=make),
                     OUT_SUBDIR, cand_id)
    if make:
        os.makedirs(d, exist_ok=True)
    return d

# --- Reference catalogs -----------------------------------------------------
REF = '/lustre/gh/calibration/pipeline/reference'
VLSS_FULL = _env('VLSS_FULL', f'{REF}/surveys/FullVLSSCatalog.text')
# RA Dec Peak ascii, already filtered to unresolved sources — ideal astrometry refs.
VLSS_UNRESOLVED = _env('VLSS_UNRESOLVED',
                       f'{REF}/surveys/vlssr_radecpeak_unresolved.txt')
BEAM_H5 = _env('BEAM_H5', f'{REF}/beams/OVRO-LWA_MROsoil_updatedheight.h5')

# Target / contaminant catalogs (clv15 working dir).
CLV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Cross-match catalogs (all in clv15). Distance lists + the flare/UCD catalog
# (recovers FI Cnc / UX Ari), the exoplanet-host list (PRIMARY science targets),
# and the known long-period-transient list. Gregg 2026-06-28: cross-match ALL of
# these, not just the *pc lists.
TARGET_CSVS = {
    '10pc': os.path.join(CLV, '10pc.csv'),
    '25pc': os.path.join(CLV, '25pc.csv'),
    '50pc': os.path.join(CLV, '50pc.csv'),
    'flare_ucd': os.path.join(CLV, 'flare_ucd.csv'),   # flare stars / UCDs
    'hwj': os.path.join(CLV, 'Hot_Warm_Jupiters.csv'), # exoplanet hosts
    'lpt': os.path.join(CLV, 'LPTs.csv'),              # long-period transients
}
# Keys cross-matched in the context arm (atnf used only if the file is present).
XMATCH_KEYS = ['10pc', '25pc', '50pc', 'flare_ucd', 'hwj', 'lpt', 'atnf']
# Catalog matches that count as positive "known astrophysical source" evidence.
XMATCH_POSITIVE = ['10pc', '25pc', '50pc', 'flare_ucd', 'hwj', 'lpt']
# Optional extras — used only if present on disk (see context arm).
FLARE_STAR_CSV = _env('FLARE_STAR_CSV', os.path.join(CLV, 'flare_ucd.csv'))
ATNF_CSV = _env('ATNF_CSV', os.path.join(CLV, 'atnf_pulsars.csv'))

# --- Scope vocabulary -------------------------------------------------------
SUBBANDS = list(cfg.ALL_SUBBANDS)                  # 18MHz .. 82MHz
WIDEBANDS = ['Red', 'Green', 'Blue', 'Full']
ALL_SCOPES = SUBBANDS + WIDEBANDS
SUBBAND_BANDS = cfg.SUBBAND_BANDS                  # Red/Green/Blue -> [subbands]
SUBBAND_TO_BAND = cfg.SUBBAND_TO_BAND
SUBBAND_FREQ_HZ = cfg.SUBBAND_FREQ_HZ             # {scope: {lo,hi,center,bw}}
WIDEBAND_FREQ_HZ = {b: {'lo': lo, 'hi': hi, 'center': (lo + hi) / 2.0}
                    for b, (lo, hi) in cfg.WIDEBAND_BANDS_HZ.items()}


def scope_center_hz(scope):
    if scope in SUBBAND_FREQ_HZ:
        return SUBBAND_FREQ_HZ[scope]['center']
    if scope in WIDEBAND_FREQ_HZ:
        return WIDEBAND_FREQ_HZ[scope]['center']
    return None

# --- Tunables ---------------------------------------------------------------
# Per-band 50%-completeness sigma (phase3_catalog.C50_SIGMA); subbands -> 8.0.
C50_SIGMA = {'Red': 9.3, 'Green': 8.4, 'Blue': 7.5, 'Full': 8.0}
SUBBAND_C50_SIGMA = 8.0

DETECT_SNR = 6.0          # per-band forced-photometry detection threshold
STRONG_SNR = 7.0          # "strong" detection (matches blind-search FP gate)
ELEV_MIN_DEG = 20.0       # below this, beam/iono degrade — flag low-elevation
ASTROM_FLOOR_ARCMIN = 5.0 # min search-box half-size even if VLSS scatter is tiny
ASTROM_SCALE_NU = -1.5    # match-radius frequency scaling (ionospheric)

# SNR control ring (validated V recipe: peak / min-sector MAD over an annulus).
SNR_INNER_BEAMS = 2.0
SNR_OUTER_BEAMS = 10.0
SNR_N_SECTORS = 4
SNR_MIN_SECTOR_PIX = 400

# Escalation gates.
#   T4 (all-epoch search): corroboration-gated.
ESCALATE_MIN_BANDS = 2    # >= this many independent bands detected -> escalate
#   T5 (10-second snapshot search): now ALWAYS run for every candidate (Gregg
#   2026-06-28) — lightcurve + snapshot spectrum, no longer gated on 10-min
#   structure. SNAPSHOT_VAR_FRAC is retained only to LABEL short_structure in the
#   10-min report. The snapshot drills into the peak 10-min bin (bounded ~75
#   frames) to keep the always-on cost tractable.
SNAPSHOT_VAR_FRAC = 0.5   # 10-min lightcurve fractional variation -> short_structure
SNAPSHOT_PAD_S = 60.0     # window pad around the peak 10-min bin for the lightcurve

# --- A1: I/V coincidence (corrected physics, Gregg 2026-06-26) --------------
# Stokes I >= |V| ALWAYS; a 100%-CP source has I = |V|. A real CP transient
# therefore shows in BOTH the I residual and the V residual (comparable flux
# when highly polarized); the MOST reliable detections are seen in both.
#   * coincident I+V residual detection -> strong corroboration
#   * |V| <= I is required; V/I > 1 is UNPHYSICAL -> artifact flag
# (NOT a "leakage tracks bright I" story — that earlier framing was wrong.)
VI_RATIO_MAX = 1.0        # physical ceiling; allow small margin for noise below
VI_RATIO_UNPHYSICAL = 1.25  # V/I above this (beyond noise) -> artifact

# Region-file / marker conventions (per standing feedback).
REG_RADIUS_DEG = 1.0      # all DS9 circles use a 1-degree radius

# --- P(astrophysical) ranking (replaces the old additive 0-10 score) --------
# Gregg 2026-06-28: rank candidates by P(astrophysical) ONLY. Implemented in
# score.py as a logistic over transparent, itemised log-odds contributions.
# The morphology point-source test (vs the EMPIRICAL beam) and the I>=|V| sanity
# are the decisive physical discriminators (they killed rank-48/rank-28); low
# elevation is the population-level artifact signature.
MORPH_RESOLVED_RATIO = 1.5   # observed/empirical-PSF FWHM ratio -> RESOLVED (artifact)
MORPH_POINT_RATIO = 1.35     # below this in maj AND ~min -> point-like
ELEV_GOOD_DEG = 45.0         # above this elevation, beam/iono are benign
# A-team HARD exclusion (Gregg 2026-06-28): a candidate within this of an A-team
# source IS that source / its near field -> P(astro)=0 regardless of other
# evidence (caught the Crab at rank-1). Distinct from the softer 5-deg additive
# penalty that still applies in 2-5 deg.
ATEAM_VETO_DEG = 2.0


def subband_c50_sigma(scope):
    return C50_SIGMA.get(scope, SUBBAND_C50_SIGMA)
