"""Configuration of `exopipe3 notify` - the hour's one email (the Phase 2 report plus the Phase 3 highlights).

Per run: --set notify.KEY=VALUE or --config FILE; `exopipe3 config --step notify` lists the section.
"""
from exopipe.phase3.config import _Section


class notify(_Section):
    """`exopipe3 notify`. Gregg 2026-11-12: "one email at the end of Phase 3 that is close to identical to the current one
    in what is included, just with the transient and forced photometry highlights added"; transients: "Ideally I want real
    transients not junk" (the candidates that pass the filters below and every default filter of their served page, most
    significant first);
    targets of the Hot / Warm Jupiter and 10 pc samples: "Just detections!!". The recipients, the mail account and the Slack
    webhook are Phase 2's (exopipe/common/phase2_config.py): this section adds none."""
    SEND = False                                                 # False: the email is WRITTEN (<out>/notify/<item>.eml, .html), not sent.
                                                                 # True needs Gregg's go: the recipients are Phase 2's list
    TOP_N = 10                                                   # transient candidates listed per Stokes (V first, then I)
    OUT_ROOT = '/lustre/gh/main/exopipe_ops/status_reports'      # <out>/notify/<item>_highlights.html (one small file per hour)
    # The served review pages (two-port server): the registry gives each page's run directory, catalog directory and code.
    REGISTRY = '/lustre/gh/main/phase3/serve/registry.json'
    HUB_URL = 'http://localhost:8765'                            # links as the hub writes them (the reader tunnels to the review host)
    # Gregg 2026-11-14: "I only want V and I deep and 10-min transients in the email, as well as the forced photometry of
    # course." The 10-second and 1-minute pages are not listed.
    PAGES = {('deep', 'V'): 'p8766', ('deep', 'I'): 'p8768',     # (tier, Stokes) -> page id; hub layout of 2026-10-27
             ('10-minute', 'V'): 'p8787', ('10-minute', 'I'): 'p8788'}
    # THE FILTERS OF THE LIST (Gregg 2026-11-14: "Apply the following filters to *all* of these: SNR >7; elevation >35;
    # n_v_epoch (or n_i_epoch for Stokes I) < 90-percentile; beam size and axis ratio both < 1.75. Remove known pulsars
    # from the list. ... Set the bright source allowed sky loss to 5%."). They choose what the EMAIL lists and change no
    # catalog, page or archive. Each is evaluated by the served page's own code (lib/notify/page_rule.py), on top of the
    # page's default filters.
    SNR_MIN = 7.0                    # |S/N| of the detection above this
    ELEVATION_MIN_DEG = 35.0         # elevation of the detection at the middle of its LST hour above this, degrees
    N_EPOCH_MAX_PERCENTILE = 90      # detections of the page's Stokes in the detection's epoch image: at or below this
                                     # percentile breakpoint of the page's slider (the page's population of epoch images)
    SIZE_MAX_BEAMS = 1.75            # Gaussian-fit size below this, in beams
    AXIS_RATIO_MAX = 1.75            # Gaussian-fit major / minor axis ratio below this
    BRIGHT_SKY_LOSS = 0.05           # allowed mean sky loss of the page's bright-source filter, a fraction
    REMOVE_KNOWN_PULSARS = True      # a candidate the page credibly matches to a known pulsar is not listed
    SITE_LAT_DEG = 37.2398           # the pages' site latitude (their elevation formula), for the forced-photometry events
    # Gregg 2026-11-14, a further requirement: "at least half the candodates must be deep in each polarization".
    DEEP_MIN_FRACTION = 0.5          # of the listed candidates of each Stokes, at least this fraction is of the deep tier
    GAL_LAT_BOLD_DEG = 10.0          # Galactic latitude: shown in bold when |b| is below this, degrees
    # Ionosphere quality (Gregg 2026-11-14: "The ionospheric QA is poorly calibrated. Even the best night i have ever seen
    # is listed as 'Moderate (Structured)'"): the email gives each number's percentile among the processed hours instead
    # of Phase 2's fixed-threshold word. The hours' numbers are kept in <out>/notify/<this file>; an hour is read once.
    IONOSPHERE_POPULATION = 'ionosphere_population.json'
    # Forced photometry: the served event lists (events at or above each sample's detection threshold, with the
    # association verdict of each) under paths.SNAPSHOT_REVIEW_ROOT
    FORCED_RUN = 'Run_SNAPFORCED_{stokes}{tier}_gen5'
    FORCED_TIERS = {'deep': 'deep', '10-minute': '10min', '1-minute': '1min', '10-s': '10s'}
    SAMPLES = {'HWJ': 'Hot / Warm Jupiter', '10pc': '10 pc'}     # sample tag of the forced tables -> name in the email
    XM_SAMPLES = {'Hot_Warm_Jupiters': 'HWJ', '10pc': '10pc'}    # prefix of a catalog's xm_best -> sample tag
