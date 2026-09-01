"""
phase2_config.py  Configuration for OVRO-LWA Phase 2 Science Pipeline

All tunable parameters for the Phase 2 pipeline (post-imaging science
extraction). Input images come from the Celery imaging pipeline on Lustre;
all Phase 2 work happens on NVMe.
"""
import astropy.units as u

# Import shared observatory/hardware config from the main pipeline
from exopipe.common import pipeline_config

OVRO_LOC = pipeline_config.OVRO_LOC
BEAM_MODEL_H5 = pipeline_config.BEAM_MODEL_H5
CALIB_DATA = pipeline_config.CALIB_DATA
VLSSR_CATALOG = pipeline_config.VLSSR_CATALOG

# ----------------------------------------------------------------------
# Pipeline version (stamped into FITS headers as GITTAG)
# ----------------------------------------------------------------------
def _resolve_gittag():
    """Short git hash of the pipeline code, or a static fallback when
    the working copy is not a git checkout."""
    import os
    import subprocess
    try:
        out = subprocess.run(
            ['git', '-C', os.path.dirname(os.path.abspath(__file__)),
             'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return 'clv15'


PIPELINE_GITTAG = _resolve_gittag()

# ----------------------------------------------------------------------
# Directories
# ----------------------------------------------------------------------
CELERY_IMAGE_ROOT = '/lustre/pipeline/exopipe/phase1/'
PHASE2_WORKING_DIR = '/fast/gh/main/phase2/'

# Final archive location  single constant to change when we move to Lustre
PHASE2_ARCHIVE_DIR = '/fast/gh/main/phase2/'

# ----------------------------------------------------------------------
# OVRO-LWA Instrument Parameters
# ----------------------------------------------------------------------
N_ANTENNAS = 352
CHANNEL_BW_HZ = 23926.0                     # 23.926 kHz per channel
N_CHANNELS_TOTAL = 3072                       # Total correlator channels
N_CHANNELS_PER_SUBBAND = 192                  # Channels per subband
SUBBAND_BW_HZ = N_CHANNELS_PER_SUBBAND * CHANNEL_BW_HZ  # 4,593,792 Hz
TOTAL_BW_HZ = N_CHANNELS_TOTAL * CHANNEL_BW_HZ          # 73,500,672 Hz (B-05 fix 2026-08-11; old comment said 73,476,672)
FREQ_LO_HZ = 13398000.0                      # Lowest channel edge: 13.398 MHz
FREQ_HI_HZ = FREQ_LO_HZ + TOTAL_BW_HZ       # Highest channel edge: ~86.875 MHz
LWA_DT = 10.0310                              # Integration time [seconds]

# ----------------------------------------------------------------------
# Subband Ordering
# ----------------------------------------------------------------------
ALL_SUBBANDS = [
    '18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz',
    '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz'
]

SKIP_SUBBANDS = ['13MHz']  # Known unusable

ANCHOR_SUBBAND = '73MHz'

# Cascade arms  ordered outward from anchor
CASCADE_DOWN = [
    '69MHz', '64MHz', '59MHz', '55MHz', '50MHz', '46MHz',
    '41MHz', '36MHz', '32MHz', '27MHz', '23MHz', '18MHz'
]
CASCADE_UP = ['78MHz', '82MHz']

# Exact subband frequency boundaries [Hz], derived from correlator setup:
# 3072 channels × 23.926 kHz, starting at 13.398 MHz, 192 channels per subband.
_SUBBAND_LOWER_EDGES_HZ = [FREQ_LO_HZ + i * SUBBAND_BW_HZ for i in range(17)]

SUBBAND_FREQ_HZ = {}
for _i, _label in enumerate(['13MHz'] + ALL_SUBBANDS):
    _lo = _SUBBAND_LOWER_EDGES_HZ[_i]
    _hi = _lo + SUBBAND_BW_HZ
    _center = (_lo + _hi) / 2.0
    SUBBAND_FREQ_HZ[_label] = {'lo': _lo, 'hi': _hi, 'center': _center,
                                'bw': SUBBAND_BW_HZ}

# Maximum consecutive failures before halting a cascade arm
MAX_CONSECUTIVE_FAILURES = 3

# ----------------------------------------------------------------------
# Image Sizes (Phase 1 output changed on this date)
# ----------------------------------------------------------------------
# Before cutover: all subbands are 4096x4096, 0.03125 deg/pix
# After cutover: frequency-dependent sizes
IMAGE_SIZE_CUTOVER_DATE = '2024-12-27'

# Subband-to-band membership (determines pixel scale post-cutover)
SUBBAND_BANDS = {
    'Red':   ['18MHz', '23MHz', '27MHz', '32MHz', '36MHz'],
    'Green': ['41MHz', '46MHz', '50MHz', '55MHz', '59MHz'],
    'Blue':  ['64MHz', '69MHz', '73MHz', '78MHz', '82MHz'],
}

# Reverse lookup: subband -> band name
SUBBAND_TO_BAND = {}
for _band, _subs in SUBBAND_BANDS.items():
    for _sb in _subs:
        SUBBAND_TO_BAND[_sb] = _band

# Post-cutover image parameters per band
NEW_IMAGE_PARAMS = {
    'Blue':  {'npix': 3122, 'cdelt_deg': 0.037, 'freq_range': (64, 82)},
    'Green': {'npix': 2357, 'cdelt_deg': 0.049, 'freq_range': (41, 64)},
    'Red':   {'npix': 1507, 'cdelt_deg': 0.0767, 'freq_range': (18, 41)},
}

# ----------------------------------------------------------------------
# Cascade Dewarping Parameters
# ----------------------------------------------------------------------
# Anchor step (73 MHz vs VLSSr at 74 MHz)
ANCHOR_FREQ_MHZ = 73.0
ANCHOR_MATCH_RADIUS = 5.0 * u.arcmin
ANCHOR_FLUX_RATIO_MIN = 0.80
ANCHOR_FLUX_RATIO_MAX = 1.20

# Cascade steps (subband vs dewarped neighbor)
# Match radius scales as (ANCHOR_FREQ / freq)^exponent from the base value.
# Ionospheric refractive shift scales ~ν^-2 (TEC-driven), but cascade steps
# bridge adjacent subbands (small Δν), so ν^-1.5 is a good compromise.
CASCADE_MATCH_RADIUS_BASE = 5.0   # arcmin at ANCHOR_FREQ_MHZ
CASCADE_MATCH_RADIUS_EXPONENT = 1.5  # frequency scaling exponent
CASCADE_FLUX_RATIO_MIN = 0.80
CASCADE_FLUX_RATIO_MAX = 1.20

# Source extraction (BDSF)
BDSF_THRESH_PIX = 10.0
BDSF_THRESH_ISL = 5.0

# Warp anchor filtering
MIN_WARP_ANCHORS = 6
VLSS_MAX_SIZE_DEG = 75.0 / 3600.0   # Reject resolved VLSSr sources

# Outlier rejection on matched anchors
OUTLIER_CLIP_SIGMA = 3.0
OUTLIER_CLIP_ITERS = 3

# Dewarping methods  separate choices for anchor and cascade steps.
#   'rbf'      BDSF source extraction + scipy RBFInterpolator (configurable kernel)
#   'zernike'  BDSF source extraction + Zernike polynomial fit
#   'flow'     Brox optical flow via image_plane_correction (requires py310_orca)
ANCHOR_METHOD = 'rbf'           # Catalog ? 73 MHz (default: rbf, best at source positions)
CASCADE_METHOD = 'flow'         # Image ? adjacent subband (default: flow, best continuous interpolation)

# Conda environment for optical flow (flow method only)
FLOW_CONDA_ENV = 'py310_orca'

# RBF interpolation parameters (rbf method)
#   Kernels: 'thin_plate_spline', 'multiquadric', 'inverse_multiquadric',
#            'gaussian', 'cubic'
WARP_RBF_KERNEL = 'multiquadric'
WARP_RBF_SMOOTHING = 1.5       # RBF smoothing for deep (1hr) images
WARP_RBF_SMOOTHING_10MIN = 5.0 # Higher smoothing for 10min (noisier source positions)
WARP_RBF_EPSILON = None        # Shape parameter for multiquadric/inverse_multiquadric/
                               # gaussian kernels. None = auto (mean nearest-neighbor
                               # distance between anchors). Not used by thin_plate_spline,
                               # linear, cubic, quintic.
WARP_RBF_SNR_WEIGHTING = False # Per-anchor smoothing weighted by 1/flux² (SNR²).
                               # Bright sources constrained tightly; faint sources relaxed.

# Zernike interpolation parameters (zernike method)
WARP_ZERNIKE_ORDER = 20        # Max radial order (231 terms for order 20)
WARP_ZERNIKE_ORDER_10MIN = 15  # Lower order for 10min (noisier, fewer sources)

# ----------------------------------------------------------------------
# Wideband Stacking
# ----------------------------------------------------------------------
WIDEBAND_BANDS = {
    'Red':   (18, 41),
    'Green': (41, 64),
    'Blue':  (64, 85),
    'Full':  (18, 85),
}

# Exact band boundaries [Hz] from subband edges (for FITS headers)
WIDEBAND_BANDS_HZ = {
    'Red':   (SUBBAND_FREQ_HZ['18MHz']['lo'], SUBBAND_FREQ_HZ['36MHz']['hi']),
    'Green': (SUBBAND_FREQ_HZ['41MHz']['lo'], SUBBAND_FREQ_HZ['59MHz']['hi']),
    'Blue':  (SUBBAND_FREQ_HZ['64MHz']['lo'], SUBBAND_FREQ_HZ['82MHz']['hi']),
    'Full':  (SUBBAND_FREQ_HZ['18MHz']['lo'], SUBBAND_FREQ_HZ['82MHz']['hi']),
}

# ----------------------------------------------------------------------
# 3-Color PNGs
# ----------------------------------------------------------------------
THREE_COLOR_SPECTRAL_INDEX = -0.7
THREE_COLOR_HORIZON_MASK_DEG = 5.0

# ----------------------------------------------------------------------
# Email + Slack
# ----------------------------------------------------------------------
EMAIL_RECIPIENT = ["gh@astro.caltech.edu", "nkosogor@caltech.edu"]
SECRETS_FILE = "~/pipeline_cred.json"

# Slack webhook URL for QA alerts (set to None to disable)
# Create at: https://api.slack.com/messaging/webhooks
SLACK_WEBHOOK_URL = None  # e.g., "https://hooks.slack.com/services/T.../B.../..."

# ----------------------------------------------------------------------
# Anchor-extraction elevation floor, per LST [deg]
# ----------------------------------------------------------------------
# At galactic-plane LSTs BDSF wedges for hours on low-elevation diffuse
# emission that can never yield dewarp anchors (the 13h timeout family,
# diagnosed 2026-06-11). Pixels below this elevation are blanked before
# anchor extraction; the warp screen is already masked outside anchor
# coverage downstream. LSTs not listed get no cut.
ANCHOR_MIN_ELEVATION_DEG = {'13h': 20.0}

# EAST-side elevation floor (Gregg 2026-08-28): at plane-rising LSTs the
# Galactic emission climbs the EASTERN sky well above the ring floor
# (13h: inner plane at ~20-35 deg east elevation) and BDSF grinds ~24 min
# per 10-min interval fitting kJy of diffuse emission that can never be an
# anchor. Pixels are blanked when BOTH east of the meridian AND below this
# elevation; the western and high sky is untouched. LSTs not listed: no cut.
# ARMED for 13h (2026-08-30) under Gregg's standing conditional ruling of
# 2026-08-30 ("if the island-cap retries fail again -> switch to the
# east-side elevation cut"): the stage-3 island-cap retry pass timed out
# at the 10 h cap on all six April nights (04-05/07/08/09/10/11, log
# /fast/gh/main/phase3/retry13h_chain.log on lwacalim02). 45 deg is the
# value from Gregg's 2026-08-28 diagnosis (blanks the rising Galactic
# plane, measured 47% of field at 13h). 2026-08-30 Gregg: trial 35 deg
# instead of 45 deg for the stage-4 pass.
ANCHOR_EAST_MIN_ELEVATION_DEG = {'13h': 35.0}

# Island size cap for anchor extraction (Gregg approved 2026-08-28):
# BDSF maxpix_isl = this many BEAM AREAS (computed per image from the
# header beam). The plane-fitting time sink is BDSF fitting enormous
# diffuse islands (measured: single island, ~20 Gaussians, 3 kJy, ~24 min);
# islands above the cap are discarded BEFORE fitting. Anchors must be
# unresolved (astrometry contract), so nothing is lost anywhere on the sky.
# Validated on 13h first; on success extend to all LSTs ('*' key).
ANCHOR_MAX_ISLAND_BEAM_AREAS = {'13h': 20.0}
