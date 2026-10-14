"""Configuration of `exopipe3 forced` - every tunable of the command, one section per stage or tier.

Units folded here: forced. A key in its section's PROTECTED_KEYS is locked
(a mirror of in-place code, a pinned input or a ruling-fixed value: it changes only by a commit); RULED names the
ledger ID of the ruling behind a key. Per run: --set SECTION.KEY=VALUE or --config FILE; `exopipe3 config --step forced` lists the sections.
"""
from exopipe.phase3.config import _Section


# =====================================================================================================================
# unit forced (was exopipe/phase3/_cfgdraft_forced.py)
# Configuration of `exopipe3 forced`. Spec: exopipe_ops/refactor_20261012/plan/specs/forced.md section 3, as
# amended by the refactor chat's decisions (_CROSSCHECK.md top, 2026-10-13 ~04:30 UTC): the deep family-A measurement (frames,
# stage, measure, evaluate) belongs to `photometry`; `forced` keeps family B (catflux), family C (snapshot event panels) and the
# deep family-A `panels` stage, which reads photometry's evaluate record.
#
# Every value: unit in the comment, the ruling (ledger ID) where one set it, and the production file:line it describes
# (exopipe_ops / clv15 at HEAD of 2026-10-13 ~04:30 UTC). During the window every stage runs the production script IN PLACE
# (core/legacy.run, DESIGN 11.1): the values below that describe a literal of an in-place script are NOT read by that script.
# The step therefore refuses a --set/--config override of any key outside lib/forced/common.STEERING (an override would change
# nothing in the products), and tests/test_forced.py compares the in-place values here with the literals of the files they
# name (a test, never a runtime read).
#
# Shared roots are NOT repeated here (decision 3): paths.TRANSIENTS_ROOT (deep runs, catflux run dirs, deep page runs),
# paths.SNAPSHOT_REVIEW_ROOT (served snapshot runs), paths.SNAPSHOT_FLEET_OUT (noise tables, hour-median models),
# paths.COADD_SB_ROOT (deep working tree: the Stokes I fallback of catflux, coadd_root.txt of the snapshot pages),
# paths.CLV15_ROOT (target lists). The interpreters are site PYTHON keys (decision 2).
# =====================================================================================================================

class forced(_Section):
    """exopipe3 forced - shared settings: family B (deep catalogue forced flux, stage catflux), family A page stage (deep
    panels of photometry's evaluation), family C (snapshot forced event panels). Spec forced.md 1-2; decisions 5-6."""
    # stages per tier, pipeline order
    STAGES = {'deep': ('catflux', 'panels'),                                  # catflux: p3s3_pass.sh:100-101; panels: deep_forced_panels.py
              'snapshot': ('build', 'fit', 'tie', 'loc', 'assoc', 'render', 'verify')}   # snapshot_forced_pages_run_20261007.sh:29-62
    # default without --stages: catflux is what the publish pass (between catalog fork and union) and the per-hour chain run;
    # deep panels run only when named (intermittent, after a photometry evaluation)
    DEFAULT_STAGES = {'deep': ('catflux',), 'snapshot': ('build', 'fit', 'tie', 'loc', 'assoc', 'render', 'verify')}
    EPOCH_STAGES = {'deep': ('catflux',), 'snapshot': ('build', 'fit', 'tie', 'loc', 'assoc', 'render', 'verify')}
    TIERS_WITH_PRODUCT = ('deep', 'snapshot')    # sub-LST: no production forced chain (ledger 1513/1518 notes; spec Q15)
    # target samples, file names under paths.CLV15_ROOT (pinned inputs; R-FORCED-PHOTOMETRY-ELEMENTS-APPROVED-V-FAINT-PARENT-
    # ENTRIES (1513); B-58). Read in place by deep_forced_panels.py:51,70 and snapshot_forced_assoc.py:71 (LISTDIR)
    SAMPLE_FILES = {'HWJ': 'Hot_Warm_Jupiters.csv', '10pc': '10pc.csv'}
    # in-place scripts (names under the site legacy root 'exopipe_ops', core.env.legacy_root)
    INPLACE_SCRIPTS = {'catflux': 'p3s3_forced_phot.py', 'panels': 'deep_forced_panels.py',
                       'snapshot': 'snapshot_forced_panels.py', 'assoc': 'snapshot_forced_assoc.py',
                       'reserve_test': 'test_snapshot_forced_reserve.py'}
    # files the in-place scripts launch BY PATH in a grandchild (whose imports the child fingerprint may not see): their
    # sha1 enters the unit's code identity (DESIGN 11.5 'plus every script a step shells out to'); names under exopipe_ops
    INPLACE_GRANDCHILDREN = {'panels': ('build_i14_pages_std7.py', 'std7_panel.py', 'tess/tess_coverage.py'),
                             'loc': ('build_localizations.py', 'source_astrometry_panel.py', 'source_localization.py')}
    # interpreter per call: a key of the site file's PYTHON (decision 2). default = py38_orca_nkosogor (p3s3_pass.sh:13 PY8;
    # deep_forced_panels.py:364); py36 = the development env of the snapshot forced scripts (pages_run:20); tess = TESS env
    # (deep_forced_panels.py:370 TESS_PY); casa = the CASA imfit worker of the tie (P3_CASA_PYTHON, phase3_snapshot_refastro.py:72)
    PYTHON_KEY = {'catflux': 'default', 'panels': 'default', 'snapshot': 'py36', 'tess': 'tess', 'casa': 'casa'}
    # D5 (Gregg 2026-10-13 'Approve the template mirror fix and re-measure K'): the orientation of the template beam in the
    # IN-PLACE template fit the snapshot fit stage calls (clv15/phase3_template_fit.py:53-56 since clv15 44f0155: Jm = the
    # pixel scale matrix, no mirror). Recorded on every fit unit beside fitting.TEMPLATE_BEAM_ORIENTATION (the switch of
    # core/fitting); test_forced fails when the clv15 code no longer says this
    INPLACE_TEMPLATE_ORIENTATION = 'wcs'
    NODE_LOCAL_RE = r'^/(fast|data\d\d)(/|$)'    # node-local path prefixes (site NODE_LOCAL_ROOTS names the host)
    LOG_TAIL_CHARS = 4000                        # characters of a failed call's log quoted in the record
    PROTECTED_KEYS = ('SAMPLE_FILES',)
    RULED = {'SAMPLE_FILES': 'R-FORCED-PHOTOMETRY-ELEMENTS-APPROVED-V-FAINT-PARENT-ENTRIES',
             'INPLACE_TEMPLATE_ORIENTATION': 'FINDING-TEMPLATE-FIT-MIRRORED-BEAM-20261013'}


class forced_panels(_Section):
    """Deep family A, stage panels (deep_forced_panels.py in place; isolated page run directory on the host holding the
    searched run). The rows come from photometry's evaluate record (decision 6) or --deep-targets-csv."""
    # photometry's record of the evaluation (photometry.md 4.1: <photometry out>/_records/photometry/evaluate.json) and the
    # output the panels render (deep_forced_panels.py:199 receives it explicitly in production)
    EVAL_RECORD = ('photometry', 'evaluate')
    EVAL_TARGETS = 'targets_above_threshold.csv'
    BUILDER = '/lustre/gh/main/exopipe_ops/build_i14_pages_std7.py'   # the page builder run under the wrapper (:44)
    MERGE_ARCMIN = 1.0                           # arcmin; entries of one system join one page (deep_forced_panels.py:45)
    GAUSS_SNR = 7.0                              # the radio-kind rule the pages check (:46); equals fitting.GAUSS_SNR
    WRAPPER_VERSION = '2026-10-11c'              # page fingerprint component (:50)
    INFO_LINE_CHARS = 190                        # characters std7_panel prints of the FORCED TARGET line (:330)
    SRC_LINKS = ('iresid', 'vresid', 'parent_catalogs', 'v_deep_images.csv', 'v110_products', 'v_deep_ledger.csv',
                 'master_catalog.csv')           # linked from the searched run (:340-341)
    SRC_COPIES = ('page_noise_cache.json', 'source_catalogs/localizations.csv',
                  'source_catalogs/localization_offsets.csv')   # copied (:344)
    QUEUE = 'judgements/queue_forced.txt'        # written only when the content test passes (:411-415)
    VERIFY_OK = 'DEEP-FORCED-VERIFY-OK'          # :338, :416
    TESS_OK = 'tess_coverage_forced.csv written'  # :372 (TESS failure = annotation, never a refusal; ledger 1756)
    RULED = {'MERGE_ARCMIN': 'DEFECT-DEEP-FORCED-PANELS-TARGET-NAME-COLLISION',
             'GAUSS_SNR': 'RULING-TEMPLATE-FIT-BELOW-7-SIGMA'}


class forced_catflux(_Section):
    """Deep family B, stage catflux (p3s3_forced_phot.py in place; B-03, LANDED-INTERFACE-ITEMS-P3S3-PASS-V2): forced flux at
    each served source's catalogue position on the run's own search frame of its best detection, written into the tier
    survivors in place, between `catalog --until fork` and `catalog --from union` (p3s3_pass.sh:100-101; catalog D-G22)."""
    CATALOG_GLOB = 'source_catalogs_[VI]*'       # every per-(Stokes, tier) dir AND the served views V, I (p3s3_forced_phot.py:70)
    SURVIVORS_GLOB = '*_sources_survivors.csv'   # :72
    COLUMNS = ('forced_jy', 'forced_pixel_jy', 'forced_rms_jy', 'forced_method', 'forced_frame')   # :65-67, :86
    STAMP_RADIUS_MIN_PX = 6                      # px (:53)
    STAMP_RADIUS_BEAMS = 1.5                     # x BMAJ in px (:53)
    FWHM_TO_SIGMA = 2.3548                       # (:60)
    FRAME_CACHE = 40                             # memory-mapped frames kept (:36)
    DECIMALS = 5                                 # rounding of forced_jy / forced_pixel_jy / forced_rms_jy (:65)
    I_FALLBACK_SUBTRACTED = True                 # Stokes I: <COADD_SB_ROOT>/<cell>/subtracted/ frame when the run's own is
                                                 # absent (p3s2_frames.isub_path :69-79); B-06 open (default c = keep); E-11
    ROOTS_CSV = '/lustre/gh/main/exopipe_ops/s2search_state/roots.csv'   # P3S2_ROOTS_CSV passed explicitly (p3s2_frames.py:17 default)
    NO_FRAME = 'no frame / off-frame'            # forced_method of a row without a measurement (:86)
    DONE_RE = r'forced photometry: (\d+) measured, (\d+) without a frame'   # :90
    RECHECK_ROWS = 5                             # rows per file re-measured independently by the content test (seed below)
    RECHECK_SEED = 1
    RECHECK_TOL_JY = 1e-5                        # Jy; the written value is rounded to 5 decimals
    PROTECTED_KEYS = ('ROOTS_CSV',)
    RULED = {'I_FALLBACK_SUBTRACTED': 'LANDED-INTERFACE-ITEMS-P3S3-PASS-V2'}


class forced_snapshot(_Section):
    """Snapshot family C: forced event panels of one Stokes and one cadence (snapshot_forced_panels.py in place; launcher
    snapshot_forced_pages_run_20261007.sh, driven by snapshot_forced_reserve_20261007.sh R3)."""
    WORK_ROOT = '/fast/gh/main/phase3/snapshot_forced_pages'           # work runs RUN = <root>/<NAME> (pages_run:20, compute node /fast)
    REVIEW_SUBDIR_UNDER_OUT = 'review'           # review copies under a non-production --out: <out>/review/<NAME>
    # the evaluations the last served generation used (snapshot_forced_reserve_20261007.sh:35 EV/EVW, TAG 20261007); the
    # module defaults are .../eval and .../eval_wb (snapshot_forced_panels.py:41-42); the step always passes them explicitly
    EVAL_SB = '/lustre/gh/main/phase3/snapshot_forced/eval_20261007'
    EVAL_WB = '/lustre/gh/main/phase3/snapshot_forced/eval_wb_20261007'
    ZONE_FLAGS = '/lustre/gh/main/phase3/snapshot_forced/zones_recomputed_20261007.csv'   # reserve:58 FLAGS (test Z2)
    STAGE1_ROOT = '/lustre/gh/main/phase3/snapshot_forced/stage1'      # photometry step output (snapshot_forced_panels.py:40)
    PHOT_ROOT = '/lustre/pipeline/exopipe/phase3/photometry'           # the search's photometry cutouts (:240)
    MASTER_CATS = '/lustre/gh/main/exopipe_ops/snapshot_master_catalogs'   # pinned tie references (:888)
    FROZEN_MASTERS = '/lustre/pipeline/exopipe/phase3/snapshot_masters_gen4'  # pinned; FREEZE-SNAPSHOT-MASTERS-GEN4 (:1264)
    HOURS_SEARCHED_DIR = '/lustre/gh/main/exopipe_ops/snapshot_forced_state/done'   # photometry fleet markers read as data (:1206); Q12
    REFASTRO_SCRATCH_PRODUCTION = '/fast/gh/main/phase3/snapshot_scratch/refastro'  # P3_SNAP_REFASTRO_SCRATCH default
                                                 # (clv15/phase3_snapshot_refastro.py:74); production runs only
    CADENCES = ('10s', '1min')                   # one per run (SNAP-FORCED-ONE-TIER-PER-PAGE) (:1422)
    EPOCH_NAME = 'Run_SNAPFORCED_{S}{cadence}_E{epoch}'   # per-hour run name (--epochs without --snap-name); NEW
    REFUSE_FULL = True                           # INCIDENT-SNAP-FULL-CUTOUTS-EQUAL-RED-62-CELLS; B-82 (:66-69)
    MERGE_ARCMIN = 1.0                           # arcmin (:433)
    ASTROM_SNR = 7.0                             # |S/N|; equals fitting.GAUSS_SNR (:708)
    BLOCK_MIN_FRAMES = 3                         # admitted frames per 1-min block (:272)
    VI_NOTE_SIGMA = 5.0                          # combined sigma; note, never a gate (:45)
    TIE_LARGE_X_SYS = 2.0                        # display flag only (:1112)
    TIE_FRAME_WINDOW_S = 12                      # s; reference frames within 12 s of the event (:935)
    TIE_RECON_TOL_ARCMIN = 0.02                  # arcmin; used-reference reconstruction vs applied correction (:996-997)
    PANEL_EL_MIN_DEG = 30.0                      # deg; std7_panel.EL_MIN for snapshot panels (:363); B-40
    LIGHTCURVE_TICK_STEP = {'10s': 30, '1min': 5}   # points between tick labels (:138)
    FIG_THREADS = 6                              # astrometry-figure threads (:1098)
    FIG_TIMEOUT_S = 900                          # s per figure (:1095)
    NPROC = {'fit': 24, 'tie': 12, 'render': 12}  # processes, module defaults (:1423,1426,1429); --workers overrides
    VERIFY_PANEL_MIN_HEIGHT_PX = 1300            # px; a panel shorter has no light-curve strip (:1360)
    VERIFY_SNR_SAMPLE = (5, 1)                   # 5 events, random seed 1 (:1363)
    VERIFY_SNR_TOL = 0.01                        # |S/N| stage 1 vs evaluation (:1365)
    SKIP_FIELDS = ('cell', 'sample', 'target', 'band', 'stokes', 'tier', 'ts', 'snr', 'screen', 'skip_reason')   # :574
    FLOW_ENV = {'P3_SNAP_FLOW': '1'}             # the module's own setdefault (:34), passed explicitly (legacy.run clears P3_*)
    # success lines of the subcommands (the launcher's completion signals, pages_run:31-64)
    OK_LINES = {'build': 'FORCED-BUILD identities', 'fit': 'FORCED-FIT events', 'tie': 'FORCED-TIE ',
                'loc': 'FORCED-LOC ', 'assoc': 'FORCED-ASSOC identities', 'assoc_verify': 'FORCED-ASSOC-VERIFY-OK',
                'render': 'FORCED-RENDER complete', 'yellow': 'YELLOW-OK', 'served': 'SERVED-ZONES-OK',
                'verify': 'FORCED-VERIFY-OK', 'masters': 'master links'}
    # items the review host's localization writes, copied back to the work run (pages_run:38-41)
    LOC_COPY_BACK = ('source_catalogs', 'source_catalogs_{S}_loc', 'localization_figs', 'coadd_root.txt')
    PROTECTED_KEYS = ('MASTER_CATS', 'FROZEN_MASTERS')
    RULED = {'REFUSE_FULL': 'INCIDENT-SNAP-FULL-CUTOUTS-EQUAL-RED-62-CELLS', 'CADENCES': 'SNAP-FORCED-ONE-TIER-PER-PAGE',
             'MERGE_ARCMIN': 'R-FORCED-PAGES-ASTROMETRY-FILTERS-MERGE', 'ASTROM_SNR': 'RULING-TEMPLATE-FIT-BELOW-7-SIGMA',
             'VI_NOTE_SIGMA': 'R-FORCED-VI-NOTE-NEVER-GATE', 'TIE_LARGE_X_SYS': 'R-REFERENCES-AT-ALL-ELEVATIONS',
             'FROZEN_MASTERS': 'FREEZE-SNAPSHOT-MASTERS-GEN4', 'BLOCK_MIN_FRAMES': 'R-FORCED-PANELS-SERVED'}


class forced_assoc(_Section):
    """Snapshot family C, stage assoc (snapshot_forced_assoc.py in place; IMPLEMENTED-FORCED-ASSOCIATION-TARGET-VS-
    BACKGROUND-Q8A)."""
    RADIO_CATDIR = '/lustre/gh/main/catalogs/radio_surveys_20261006'   # pinned, sha256 checked (:70, :143); P3_RADIO_CATDIR
    CEIL_PCT = 99.9                              # percentile ceiling (:73); K_CEIL = 3.717 x error (:75); v2 staged (2128)
    LR_DECISIVE = 100.0                          # likelihood ratio (:76); NOT a ratified operating point (2024 D; B-86)
    MIN_S74_JY = 0.5                             # Jy at 73.8 MHz (:77)
    ALPHA = -0.8                                 # spectral index to 73.8 MHz (:79)
    GROUP_LINK_ARCSEC = 60.0                     # arcsec (:80)
    DENS_CONE_DEG = 10.0                         # deg (:81)
    DENS_MIN_COUNT = 10                          # entries (:82)
    NEAREST_OUT_DEG = 2.0                        # deg, record only (:83)
    SITE_LAT_DEG = 37.2398                       # deg (:84); DECLIM = SITE_LAT - 60 (:85); B-42 open
    NO_LABEL = ('Gaia_DR3_625453654702751872',)  # AD Leo: identities never labelled (:87, :95)
    LABELS = ('target', 'background source', 'ambiguous', 'unassociated', 'no radio position', 'not labelled')   # :97
    FILES = ('forced_assoc.csv', 'forced_assoc_identity.csv', 'forced_assoc_events.csv')   # :98-100
    PROTECTED_KEYS = ('RADIO_CATDIR',)
    RULED = {'NO_LABEL': 'IMPLEMENTED-RULING-D-20261008-FORCED-I-PAGES-SERVED-ADLEO-NOT-LABELLED',
             'CEIL_PCT': 'IMPLEMENTED-FORCED-ASSOCIATION-TARGET-VS-BACKGROUND-Q8A'}
