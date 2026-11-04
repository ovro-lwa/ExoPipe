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
              'snapshot': ('build', 'fit', 'tie', 'loc', 'assoc', 'render', 'verify',   # snapshot_forced_pages_run_20261007.sh:29-62
                           # family W, the wide-area burst test of the forced runs, and family N, the night-activity table (the
                           # astrometry figure v2's inputs; lib/forced/wide_stages.py): named only, one family per invocation
                           'wide_jobs', 'wide_search', 'wide_test', 'wide_publish', 'activity')}
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
    # THE GENERATION THE SNAPSHOT FORCED PAGES SERVE: generation 5 (served 2026-10-26 19:54-20:16 UT, hub registry
    # <S> 10-sec-forced / 1-min-forced = Run_SNAPFORCED_{I,V}{10s,1min}_gen5; driver snapshot_forced_gen5_stages_20261026.sh:41-44,
    # TAG gen5; the four runs' FT-START lines in snapshot_forced_pages_run_<run>.log record eval, P3_PHOT_ROOT and
    # P3_FORCED_STAGE1). The step passes every one to the child explicitly - the evaluations (build/verify --eval-sb/--eval-wb)
    # and the zone flags (served --flags) as arguments, the two roots as P3_PHOT_ROOT / P3_FORCED_STAGE1 (tier_snapshot.roots_env;
    # core.legacy.run clears P3_*, and the module defaults are generation 4's: snapshot_forced_panels.py:40-42, :246). A new
    # generation changes these five values; tests/test_forced checks them against the served runs' records.
    EVAL_SB = '/lustre/gh/main/phase3/snapshot_forced/eval_gen5'       # gen5 driver :42 EV
    EVAL_WB = '/lustre/gh/main/phase3/snapshot_forced/eval_wb_gen5'    # :42 EVW
    ZONE_FLAGS = '/lustre/gh/main/phase3/snapshot_forced/zones_recomputed_gen5.csv'   # :42 FLAGS - the served-zones test Z2
    STAGE1_ROOT = '/lustre/gh/main/phase3/snapshot_forced/stage1_gen5'   # :41 IN - stage 1 (P3_FORCED_STAGE1)
    PHOT_ROOT = '/lustre/pipeline/exopipe/phase3/photometry_gen5'        # :41 PHOT - the cutout generation (P3_PHOT_ROOT)
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
    # Exclusions downstream (Gregg 2026-10-14, the refactor chat's recommendation; the snapshot chat agreed): a selected event
    # on an excluded (LST, band, night) - core.exclusions, read now; a wideband excluded with any member subband - refuses the
    # build before any in-place call: the evaluation predates the exclusion and is redone first. 'build' = production (no gate)
    EXCLUDED_EVENTS = 'refuse'
    PROTECTED_KEYS = ('MASTER_CATS', 'FROZEN_MASTERS')
    RULED = {'REFUSE_FULL': 'INCIDENT-SNAP-FULL-CUTOUTS-EQUAL-RED-62-CELLS', 'CADENCES': 'SNAP-FORCED-ONE-TIER-PER-PAGE',
             'MERGE_ARCMIN': 'R-FORCED-PAGES-ASTROMETRY-FILTERS-MERGE', 'ASTROM_SNR': 'RULING-TEMPLATE-FIT-BELOW-7-SIGMA',
             'VI_NOTE_SIGMA': 'R-FORCED-VI-NOTE-NEVER-GATE', 'TIE_LARGE_X_SYS': 'R-REFERENCES-AT-ALL-ELEVATIONS',
             'FROZEN_MASTERS': 'FREEZE-SNAPSHOT-MASTERS-GEN4', 'BLOCK_MIN_FRAMES': 'R-FORCED-PANELS-SERVED',
             'EXCLUDED_EVENTS': 'R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS'}


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


# =====================================================================================================================
# FAMILY W - THE FORCED TARGETS' WIDE-AREA BURST TEST (v3), and FAMILY N - THE NIGHT-ACTIVITY TABLE: the two inputs of the
# astrometry figure v2 (exopipe_ops source_astrometry_panel.py - v2 since 7e6904a, 2026-11-04; source_astrometry_panel_v2.py
# is its alias - and astrometry_assoc.py; live 2026-11-03 for the forced runs of
# every tier and the deep tier) that analysis scripts made (phase3/refactor_analysis/review_items_20261031/forced_wide/ and
# figure_v2/night_activity_build.py). Ported, not run in place: the analysis directory is not a production tree. The in-place
# production modules the method calls (the forced evaluation's gates snapshot_forced_eval, the snapshot search's time and
# night rules phase3_snapshot_search, tier rule 3 phase3_tier_rules, the radio catalogues snapshot_forced_assoc, the beam table
# phase3_snapshot_astro) are imported in the step process (core/env.legacy), so the search stays gate-exact with stage 2.
# THE APPROVAL (Gregg 2026-11-03, on the v3 page: "v3 is not perfect, but it is approved. Proceed."; ledger
# R-GREGG-20261103-FORCED-WIDE-V3-APPROVED): the method of the forced_wide_v3.py docstring - its values are the defaults below,
# the method's keys RULED by that row (the candidates by flux capability, the taper test and its control lines, the
# centroid test, the evidence rules, v2's centering; the bursts and the jobs it tests). Implementation choices inside it are
# the refactor chat's under its delegation (R-GREGG-20261016-REFACTOR-DECIDES-BY-GOALS): V2_CTL_DRAW and CONTROL_ORDER are
# RULED by R-REFACTOR-20261104-FORCED-WIDE-CONTROLS-ORDER-INDEPENDENT; the Monte Carlo precision and its knife-edge flag
# (TAPER_CTL_MAX, BOOT_DRAW, BOOT_Z_SE, BOOT_MAX, KNIFE_EDGE_SE, KNIFE_EDGE_DRAWS) by its amendment
# R-REFACTOR-20261104-FORCED-WIDE-CONTROLS-ORDER-INDEPENDENT-AMENDED-PRECISION-KNIFE-EDGE. Spec: docs/specs/forced_wide.md.
# =====================================================================================================================

class forced_wide(_Section):
    """Family W: the forced targets' WIDE-AREA BURST TEST - which cataloged source, if any, makes a forced target's events.
    Stages wide_jobs (the targets and their bands; the forced events), wide_search (every >= 7-sigma burst in the target's
    gen5 cutouts on exactly the frames the forced photometry used), wide_test (the v3 analysis) and wide_publish
    (<run>/forced_wide_v3.csv, read by the astrometry figure v2: astrometry_assoc.V3_MAP). Inputs, outputs, method: spec
    docs/specs/forced_wide.md."""
    # ---- inputs: the forced runs (generation 5, served 2026-10-26: hub p8785 I 10 s, p8786 V 10 s, p8776 I 1 min, p8775 V 1 min)
    #      under paths.SNAPSHOT_REVIEW_ROOT (lwacalim07's /data07); --wide-runs DIR reads a copy holding the same run names
    RUN_NAME = 'Run_SNAPFORCED_{S}{cadence}_gen5'
    # (Stokes, cadence) of the runs TESTED - their events, their base jobs, the publication - in the order the job list reads them
    # (forced_ambiguous.py: I10s, V10s, I1min, V1min; a target's first row wins, so the order is part of the result). The
    # 10-minute and 1-hour forced runs have no wide test: their events are not single-frame bursts (forced_wide_jobs.py:72,
    # jobs_ext.py:12)
    RUNS = (('I', '10s'), ('V', '10s'), ('I', '1min'), ('V', '1min'))
    # the runs whose QUEUES give the ambiguous list (forced_ambiguous.py:28: all eight forced runs, this order): the extension jobs
    # take a (Stokes, identity)'s first row from all of them (jobs_ext.py:18-19), so a target with 10-s / 1-min events that only a
    # 10-minute or 1-hour run lists is found too; the base jobs keep the RUNS' rows (forced_wide_jobs.py:69). On gen5 the four
    # tier runs add no job (review F9, 2026-11-04: 55 (Stokes, identity) only in their rows, none with 10-s / 1-min events)
    AMBIGUOUS_RUNS = (('I', '10s'), ('V', '10s'), ('I', '1min'), ('V', '1min'), ('I', '10min'), ('V', '10min'), ('I', 'deep'),
                      ('V', 'deep'))
    # the cutouts searched: photometry.CUTOUT_ROOT (the served generation - one source of truth); --wide-cutout-root overrides
    CUTOUT_GLOB = '*_??h_????????T??????_{S}.fits'   # the per-cell 10-s cutouts of an object (not _10min_ / _deep_);
                                                 # forced_wide_search.py:251
    WORK_ROOT = '/lustre/gh/main/phase3/snapshot_forced/forced_wide_gen5'   # the family's production --out (jobs, bursts,
                                                 # results); beside stage1_gen5 / eval_gen5 (photometry.OUT_ROOT[snapshot])
    JOBS = 'jobs.json'                           # wide_jobs: the job list (base + extension jobs)
    EVENTS = 'events.csv'                        # wide_jobs: the forced events of the four runs (v3 load_events)
    BURSTS_DIR = 'bursts'                        # wide_search: <BURSTS_DIR>/<S>_<identity>_<band>.npz
    RESULTS = 'results_v3.csv'                   # wide_test: one row per Stokes x tier x target (forced_wide_v3.py:60)
    CANDIDATES = 'candidates_v3.csv'             # wide_test: one row per tested candidate
    # the file each forced run carries and the figure reads (source_astrometry_panel.load:119, its literal)
    PUBLISHED = 'forced_wide_v3.csv'
    # ---- THE JOBS (forced_ambiguous.py 2026-11-01, forced_wide_jobs.py, v3/jobs_ext.py)
    # AMBIGUOUS: the nearest cataloged background source (VLSSr / TGSS / GLEAM entry, S74 >= AMB_MIN_S74_JY) within
    # AMB_SEARCH_BEAMS Master beams of the target, the beam at the median frequency of the target's forced events
    AMB_MIN_S74_JY = 0.5                         # Jy at 74 MHz (forced_ambiguous.py:21)
    AMB_SEARCH_BEAMS = 3                         # Master beams (forced_ambiguous.py:44)
    # the Master beam (arcmin) at each subband frequency (MHz), the nearest key taken - ties to the first (forced_ambiguous.py:10)
    MASTER_BEAM_ARCMIN = {18: 20.7, 23: 17.0, 27: 14.5, 32: 12.7, 36: 11.2, 41: 9.9, 46: 9.0, 50: 8.3, 55: 7.6, 59: 7.1,
                          64: 6.6, 69: 6.2, 73: 5.9, 78: 5.5, 82: 5.3}
    WIDEBAND_MHZ = {'Red': 27, 'Green': 50, 'Blue': 78, 'Full': 46}   # a wideband event's frequency (forced_ambiguous.py:11)
    DEFAULT_MHZ = 46.0                           # no events / an unknown label (forced_ambiguous.py:13, :41)
    # BASE JOBS: ambiguous targets whose background lies within BASE_MAX_BEAMS beams; their background S = the nearest physical
    # source >= BASE_BG_MIN_S74_JY within BASE_BG_RADIUS_BEAMS beams (forced_wide_jobs.py:69, :86)
    BASE_MAX_BEAMS = 1.0
    BASE_BG_MIN_S74_JY = 0.5
    BASE_BG_RADIUS_BEAMS = 1.01
    # EXTENSION JOBS (v3, after Gregg's KELT-24 note 2026-11-02): targets with events that the base list left out but with a
    # physical source >= EXT_MIN_S74_JY within EXT_RADIUS_FWHM x the snapshot beam FWHM of their LOWEST event band (jobs_ext.py)
    EXT_MIN_S74_JY = 3.0
    EXT_RADIUS_FWHM = 2.0
    GROUP_ARCMIN = 1.5                           # catalog entries within this are one physical source (forced_wide_jobs.py:30)
    # A MERGED identity (several list targets within forced_snapshot.MERGE_ARCMIN; 26 rows, 21 distinct (Stokes, identity) in the
    # four gen5 runs) takes the list position of ONE member: 'identity' = the member the identity is named after, else the first in
    # (sample, target) order - the same on any copy of the runs, and exactly the point the searched cutouts are centered on (they
    # are the identity's own: TARGRA/TARGDEC = that member's position, in every cutout, to 0.00"); 'first' = production
    # (forced_ambiguous.py:40 next(iter(...))): the first member in snapshot_forced_assoc.build_targets' order, the glob order of
    # the run tree on its file system (/data07 for the served runs; a Lustre copy lists in another order). On gen5 'first' put 6 of
    # 205 jobs 7.30-32.19" (0.12-0.54') off their cutouts' TARGRA/TARGDEC - at most 0.05 of the frame FWHM - and swapped two
    # identities' positions. The member choice leaves the tests' statistics unchanged: the one gen5 verdict that moved with it
    # (Struve 2398 A, I 1 min) moved because the job change re-indexed one control of v2's capped random draw (V2_CTL_DRAW; review
    # F2, 2026-11-04). PROPOSED (refactor 2026-11-03): production's value is in the parity config
    TARGET_MEMBER = 'identity'
    # the order the forced events are read in, which is the order the analysis meets - and so LISTS and SUMS - a target's bands
    # (a candidate's bands_far / bands_near, the taper sums): 'sorted' = by directory name (frequency order), the same on any
    # copy; 'directory' = production: os.listdir order (forced_wide_v3.load_events on its Lustre copy v3/events_copy/ - Lustre
    # lists a directory in a fixed hash order of its names, so a Lustre copy lists as that one did; /data07 lists otherwise).
    # PROPOSED (refactor 2026-11-03): production's value is in the parity config
    EVENT_ORDER = 'sorted'
    # ---- run level
    # the caller environment: switches the in-place modules read at import or call time (the radio catalogues, tier rule 3's
    # state, the run pins, the solar audit, the curve tables) and the analysis scripts' own (P3_PHOT_ROOT, FW_RUNS,
    # FW_MINUTE_RULE). Any of them set refuses the run (W-ENV): the stage passes its roots as arguments
    ENV_FORBIDDEN = ('P3_RADIO_CATDIR', 'P3_DEEP_SEARCH_STATE', 'P3_COADD_RUN', 'P3_SNAP_RUN_PINS', 'P3_SOLAR_AUDIT',
                     'P3_SNAP_FIT_RAW', 'P3_TABLE_PREFIX', 'P3_TABLE_ELEV', 'P3_PHOT_ROOT', 'FW_RUNS', 'FW_MINUTE_RULE')
    PROTECTED_KEYS = ('PUBLISHED',)
    RULED = {k: 'R-GREGG-20261103-FORCED-WIDE-V3-APPROVED' for k in (
        'AMB_MIN_S74_JY', 'AMB_SEARCH_BEAMS', 'BASE_MAX_BEAMS', 'BASE_BG_MIN_S74_JY', 'BASE_BG_RADIUS_BEAMS', 'EXT_MIN_S74_JY',
        'EXT_RADIUS_FWHM', 'GROUP_ARCMIN')}


class forced_wide_search(_Section):
    """Family W, stage wide_search: every peak of |S/N| >= SNR_MIN, either sign, in each job's gen5 cutouts, on EXACTLY the
    frames the forced photometry used (forced_wide_search.py docstring; gate-exact with stage 2, snapshot_forced_eval)."""
    # FRAMES: 10 s = admitted and the evaluation's domain gate at the target (snapshot_forced_eval.gate_mask, in place); 1 min =
    # gen5's blocks: admitted frames of one minute, >= BLOCK_MIN_FRAMES, averaged. MINUTE_RULE: the minute key of gen5's stage 1,
    # which ran 2026-10-26 before the integration-midpoint rule entered stage 1 (exopipe_ops 40611cc, 2026-10-28): 'filename' =
    # the file-name minute ts[:13] (forced_wide_search.py:31); 'midpoint' = phase3_snapshot_search.integration_minute_key. A
    # generation measured with the midpoint rule changes this with photometry.CUTOUT_ROOT
    MINUTE_RULE = 'filename'
    BLOCK_MIN_FRAMES = 3                         # admitted frames per 1-minute block (forced_wide_search.py:190; stage 1's rule)
    # PEAKS: local extrema of |value| over 3 x 3 pixels, either sign (in a frame-minus-median cube a wandering source is a
    # dipole), more than EDGE_PX inside the cutout; candidates at >= PRE_SIGMA x the plane's robust rms, then S/N = value / the
    # forced definition's local rms (1.4826 x MAD of the STAMP_PX x STAMP_PX stamp, the central (2 STAMP_CORE_PX + 1)^2
    # excluded, >= STAMP_MIN_FRAC of the stamp inside the cutout); kept at |S/N| >= SNR_MIN (forced_wide_search.py:33)
    SNR_MIN = 7.0
    PRE_SIGMA = 4.0
    STAMP_PX = 41
    STAMP_CORE_PX = 2
    STAMP_MIN_FRAC = 0.5
    EDGE_PX = 2
    MIN_FINITE_PX = 100                          # a plane with fewer finite pixels is not searched (forced_wide_search.py:171)
    # POSITION: a 2-D quadratic in log|value| over the 3 x 3 (exact for a rotated elliptical Gaussian), accepted within
    # INTERP2D_CLIP_PX; else separate 1-D fits clipped to INTERP1D_CLIP_PX (forced_wide_search.py:131, :153)
    INTERP2D_CLIP_PX = 0.75
    INTERP1D_CLIP_PX = 0.5
    # PROCESSES: Lustre reads yield to the recorders (CLAUDE.md; forced_wide_search.py:281 refused more than 6)
    WORKERS = 6
    MAX_WORKERS = 6
    # the recorders' I/O guard (exopipe_ops io_guard_20261023.sh): before each cutout read a worker waits while the latest
    # guard round has a node above IO_GUARD_MAX_S, until the last two rounds are both at or below it (the rule of
    # forced_wide/yield_forced_wide.sh, cooperative: no signals, no process matching); polled every IO_GUARD_POLL_S. None = no
    # yield; a set but missing log = no yield, said in the plan and noted in every search unit's record
    IO_GUARD_LOG = '/lustre/gh/main/exopipe_ops/io_guard_20261023.log'
    IO_GUARD_MAX_S = 4.0
    IO_GUARD_POLL_S = 20
    # a cutout file whose read or search raised: 'refuse' = that band's npz is refused (item refusal W-FILE; the test then
    # refuses W-POOL until it is resolved); 'drop' = production: the file's peaks are dropped and its error is recorded in the
    # npz file table (forced_wide_search.py:238-240, 'all or nothing per file'). No gen5 file raised (565 npz, 2026-11-02)
    FILE_ERRORS = 'refuse'
    # MIRROR (protected): the site phase3_snapshot_search.sun_state computes the Sun at (lat deg, lon deg, height m) - its literal
    # (clv15 phase3_snapshot_search.py:651); the vectorized time cache must hold what sun_state computes (tests compare)
    SUN_SITE = (37.2398, -118.2817, 1222)
    NULLABLE_KEYS = ('IO_GUARD_LOG',)
    PROTECTED_KEYS = ('SUN_SITE',)
    RULED = {k: 'R-GREGG-20261103-FORCED-WIDE-V3-APPROVED' for k in (
        'SNR_MIN', 'PRE_SIGMA', 'STAMP_PX', 'STAMP_CORE_PX', 'STAMP_MIN_FRAC', 'EDGE_PX', 'BLOCK_MIN_FRAMES')}


class forced_wide_test(_Section):
    """Family W, stage wide_test: the v3 analysis (forced_wide_v3.py; Gregg 2026-11-03 "v3 is not perfect, but it is approved.
    Proceed.") - candidates by flux, the taper test calibrated on control lines, v2's centering test, the centroid, the verdict.
    Lengths in FWHM of the band (phase3_snapshot_astro.beam_fwhm_arcmin, in place) unless named in arcmin."""
    # AMPLITUDE: a source makes an event of value v only if v <= MU x F, F = S74 (REF_MHZ / nu)^SPECTRAL_INDEX; MU = the largest
    # burst / band-flux ratio measured around cataloged sources (v3/mag_bound.py, mag_bound_V.py): Stokes I 10 s p99.9 3.3-4.3
    # -> 4.0, 1 min p99.9 2.0-2.2 -> 2.5; Stokes V |V| / F_I within 0.6 FWHM of >= 5 Jy sources, 10 s max 2.4 -> 2.5, 1 min max
    # 1.8 -> 2.0 (forced_wide_v3.py:15-19, :80)
    MU = {('I', '10s'): 4.0, ('I', '1min'): 2.5, ('V', '10s'): 2.5, ('V', '1min'): 2.0}
    REF_MHZ = 74.0
    SPECTRAL_INDEX = 0.7
    CAPABLE_FRACTION = 0.5                       # CAPABLE in a band: able to make at least this fraction of its events there
    S_MIN_JY = 0.25                              # candidates: physical sources >= this at 74 MHz (forced_wide_v3.py:83)
    R_CAND = 2.0                                 # candidates within this x FWHM of the target in a band with events
    NEAR = 0.6                                   # the taper test from this distance, the centering test inside it
    CUT_HALF_ARCMIN = 148.0                      # the cutout half-width (arcmin): 300' cutouts with a 2' margin (v2, v3)
    CUT_EDGE_ARCMIN = 4.0                        # the taper test's bins, disc and control sources stay this far inside it
    # TAPER TEST: bursts in a strip +-STRIP about the line candidate -> target, binned every BIN; the near and far windows at
    # SIDE from the target, joined log-linearly and integrated over the target's disc (radius DISC; DISC_GRID x DISC_GRID
    # samples); bins nearer than MIN_BIN to the candidate are never used (forced_wide_v3.py:20-26, :82)
    DISC = 0.35
    STRIP = 0.35
    BIN = 0.1
    SIDE = (0.35, 0.65)
    MIN_BIN = 0.05
    DISC_GRID = 41
    # CONTROL LINES: cataloged sources of CTL_FLUX x the candidate's flux (>= CTL_MIN_JY) in every job of the same Stokes, one
    # per cell of 1 / CTL_KEY_PER_DEG deg (round(ra x 20) / 20), imaginary targets at the candidate's distance in CTL_DIRS
    # directions (a random rotation), clear of sources >= CTL_BRIGHT_JY by NEAR x the smallest FWHM and >= CTL_TARGET_FWHM x
    # the largest from that cutout's own target, in bands the cutout searched (>= half) (forced_wide_v3.py:27-30, :267-308)
    CTL_FLUX = (0.5, 2.0)
    CTL_MIN_JY = 3.0
    CTL_DIRS = 8
    # the most control lines per candidate. None = EVERY eligible line (the amended decision R-REFACTOR-20261104-FORCED-WIDE-
    # CONTROLS-ORDER-INDEPENDENT-AMENDED-PRECISION-KNIFE-EDGE (1), 2026-11-04: precision, not a choice of draw - the cap was a
    # compute bound, and it left the calibration to the draw of lines it kept); production's cap=900 (forced_wide_v3.py
    # control_lines) is in the parity config. On gen5 the cap shortened 486 of 1,267 calls (38%, all Stokes I); without it the
    # test evaluates 1.43 x the lines (cap_cost_census, 2026-11-04)
    TAPER_CTL_MAX = None
    CTL_KEY_PER_DEG = 20
    CTL_BRIGHT_JY = 1.0
    CTL_TARGET_FWHM = 1.0
    # matched lines: a prediction within 1/MATCH_FACTORS[0]-MATCH_FACTORS[0] x the target's (floors MATCH_FLOORS bursts), widened
    # to MATCH_FACTORS[1] when fewer than MATCH_MIN; calibrated only with >= MATCH_MIN lines (forced_wide_v3.py:311-330)
    MATCH_FACTORS = (3.0, 10.0)
    MATCH_FLOORS = (1.0, 3.0)
    MATCH_MIN = 15
    # the taper ACCOUNTS for the target's count when p >= P_CONSISTENT and ratio <= RATIO_CONSISTENT; it FALLS SHORT when
    # p <= P_EXCESS and ratio >= R_EXCESS; night by night: Spearman rho > 0 and p < P_NIGHT (forced_wide_v3.py:35, :430)
    P_CONSISTENT = 0.05
    RATIO_CONSISTENT = 2.0
    P_EXCESS = 0.05
    R_EXCESS = 3.0
    P_NIGHT = 0.01
    MIN_NOBS = 3                                 # bursts at the target below this: UNDECIDED (fewer than 3 ...)
    # PER NIGHT: the target's bursts (disc) against the candidate's spread - bursts at NIGHT_FAR from it (and beyond
    # NIGHT_CLEAR of the target) over those within NIGHT_CORE - over nights with >= NIGHT_MIN_CORE at it, >= NIGHT_MIN_NIGHTS
    # such nights (forced_wide_v3.py:353-375)
    NIGHT_CORE = 0.35
    NIGHT_FAR = (0.8, 2.0)
    NIGHT_CLEAR = 0.5
    NIGHT_MIN_CORE = 20
    NIGHT_MIN_NIGHTS = 5
    # CENTROID of the bursts in the target's disc: error = sqrt(stat^2 + max(CEN_FLOOR, CEN_NIGHT / sqrt(nights))^2), stat from a
    # bootstrap over nights (BOOTSTRAP resamples, >= BOOT_MIN_NIGHTS nights; else CEN_DET / sqrt(n)): per detection 0.2 Master
    # beam = 0.11 frame FWHM, correlated within a night; the pulsar-measured combined floor 0.05 Master beam = 0.028 frame FWHM
    # (pulsar systematics 2026-11-01; forced_wide_v3.py:38-41, :469-499)
    CEN_DET = 0.12
    CEN_FLOOR = 0.028
    CEN_NIGHT = 0.11
    BOOTSTRAP = 1000                             # resamples (BOOT_Z_SE None: all of them; else the first block)
    BOOT_MIN_NIGHTS = 5
    # THE BOOTSTRAP'S PRECISION (the amended decision (2), 2026-11-04). BOOT_Z_SE: after the first BOOTSTRAP resamples, more
    # until the Monte Carlo standard error of the centroid z is at most this - and of each candidate's z_c, a z_c above Z_C_NEAR
    # counted as Z_C_NEAR (the error of every z of the centroid is z x SE(stat^2) / (2 err^2), SE(stat^2) from the resamples'
    # own spread; estimated, so the stop is at BOOT_Z_SE / 1.1) - or BOOT_MAX resamples are drawn; None = production: exactly
    # BOOTSTRAP (forced_wide_v3.py:478)
    BOOT_Z_SE = 0.02
    BOOT_MAX = 1000000
    # the resamples' generator. 'target' = one per (target, tier): numpy default_rng of a stable hash (sha256) of (SEED, the
    # target's (Stokes, identity), tier), the nights drawn vectorized; 'job' = production: the job's generator (after the control
    # lines when CONTROL_ORDER is 'pool'), replicate by replicate (forced_wide_v3.py:477-479)
    BOOT_DRAW = 'target'
    # THE KNIFE-EDGE FLAG (the amended decision (3), 2026-11-04): a row's knife_edge column names every statistic that decides
    # its verdict or its named sources and lies within KNIFE_EDGE_SE of its Monte Carlo standard errors of a threshold - the
    # centroid z (Z_ON, Z_OFF; error from the bootstrap), a candidate's z_c (Z_C_ON, Z_C_NEAR; the same), its taper p
    # (P_CONSISTENT, P_EXCESS; the binomial within each control source over the matched lines), its taper ratio
    # (RATIO_CONSISTENT, R_EXCESS; the matched lines' spread about kappa within each control source), the count of each
    # matched-line window (MATCH_MIN; the binomial within each control source) - within sources because a salt re-draws a
    # line's direction, never its control source (validated against 30 global salts on real rows, wide_v3.SE_KIND). DECIDES =
    # moved alone to the threshold's other side, the statistic changes the verdict or the set of named sources. With the flag,
    # results_v3.csv (and so forced_wide_v3.csv) gains bootstrap_n, centroid_z_se, knife_edge (and mc_draws, mc_outcomes, mc_row,
    # below); candidates_v3.csv the errors. None = production: none of these columns. 4 (the refactor chat, 2026-11-04, from the
    # 20-salt flip check of the gen5 run: at 2 the flag marked 5 of the 7 rows that differ from the served files - all 7 flip
    # between draws - and 10 of the 12 rows seen flipping; at 4 all of them, 17 flags in 312 rows against 13)
    KNIFE_EDGE_SE = 4.0
    # THE KNIFE-EDGE ROWS' MAJORITY (the refactor chat's decision, 2026-11-04, under the same amendment): a flagged row's
    # published verdict and named set are the most frequent outcome of KNIFE_EDGE_DRAWS independent Monte Carlo draws of the
    # approved method (each a salt of its own in place of SEED: a stable hash of (SEED, the target's (Stokes, identity), k),
    # k < KNIFE_EDGE_DRAWS), with the fraction (mc_draws, e.g. 'BACKGROUND SOURCE (VLSSr J...) in 14 of 20 Monte Carlo draws'),
    # every outcome's count (mc_outcomes) and where the row's numbers come from (mc_row: 'default', 'draw k', 'tie'); a tie is
    # UNDECIDED, its mc_draws 'a tie of 20 Monte Carlo draws, 10 each: <outcome>; <outcome>'. Unflagged rows keep their single draw. None = production: the single draw (no draws); with KNIFE_EDGE_SE None
    # no row is flagged and nothing is drawn
    KNIFE_EDGE_DRAWS = 20
    # VERDICT: centroid z >= Z_OFF not centered on the target, < Z_ON centered; a candidate the centroid lies within Z_C_ON of
    # (and within CENTER_WITHIN FWHM of the target) is direct evidence; near candidates v2 cannot separate count when z_c <
    # Z_C_NEAR; NOT EXPLAINED / CONSISTENT need tests covering >= COVER of the bursts (forced_wide_v3.py:42-57, :513)
    Z_OFF = 3.0
    Z_ON = 2.0
    Z_C_ON = 2.0
    Z_C_NEAR = 3.0
    CENTER_WITHIN = 1.0
    COVER = 2.0 / 3.0
    SEED = 20261102                              # numpy default_rng seed, one generator per job (CONTROL_ORDER 'pool': the
                                                 # control lines; BOOT_DRAW 'job': then the bootstrap) and the salt of every
                                                 # stable hash (the canonical controls; BOOT_DRAW 'target': the bootstrap's)
    # THE CONTROLS' ORDER - the taper test's control lines and v2's control pairs. 'canonical' (the decision
    # R-REFACTOR-20261104-FORCED-WIDE-CONTROLS-ORDER-INDEPENDENT, the refactor chat under Gregg's delegation, 2026-11-04: "in a
    # pipeline fed new data daily, adding a target must not re-draw an unrelated target's calibration"): ORDER-INDEPENDENT -
    #   the eligible pool by stable identifiers, never by job or pool order: one source per cell, its host the job with the best
    #   score (bands searched, nearest the cutout center), a tie to the smallest (Stokes, identity), then source identity;
    #   each control line rotated by its own seed: line i of a source at (i + u) x 360 / CTL_DIRS deg, u = a stable hash
    #   (sha256) of (SEED, the target's (Stokes, identity), the control source's identity (name, ra, dec), i);
    #   the sources taken in the order of a stable hash of (SEED, target, source), each with its lines - the seeded selection a
    #   TAPER_CTL_MAX cap applies (keyed on the target's identity); without a cap the order only fixes the sums' order;
    #   so adding or moving a job changes another target's control lines only where that job adds lines to its pool (a new
    #   source; a source it hosts better; under a cap, the lines it displaces at the end of the order).
    # 'pool' = production (forced_wide_v3.py control_lines, forced_wide_analyze.py controls_for; the parity config): the
    # pool-ordered list shuffled by the job's generator, one rotation per source from the same generator (which then draws the
    # centroid's bootstrap), host ties to the first job in pool order
    CONTROL_ORDER = 'canonical'
    # v2's CENTERING TEST (forced_wide_analyze.py, the near candidates): the region sep/2 + V2_K x FWHM about the pair's
    # midpoint (controls placed with V2_K_MAX); per-detection weights 1 / sigma^2, sigma = V2_SYS_ARCSEC x
    # (nu / V2_SYS_REF_MHZ)^V2_SYS_INDEX (+) V2_STAT_FWHM x FWHM; pooled controls of V2_CTL_FLUX x S (>= V2_CTL_MIN_JY), their
    # region inside their cutout (V2_CUT_MARGIN_ARCMIN), clear of the target's region, no source above V2_CTL_BRIGHTER x C in it,
    # none within V2_VERY_BRIGHT_DEG of a >= V2_VERY_BRIGHT_JY source, one per cell of 1 / V2_KEY_PER_DEG deg, every one of them or
    # a capped draw (V2_CTL_DRAW); a verdict needs >= V2_MIN_NIGHTS nights and >= V2_CTL_MIN_N controls of >= V2_CTL_MIN_CELLS
    # nights; power >= V2_POWER_MIN, then z < V2_Z_BG and power - z >= V2_D_TG -> BACKGROUND SOURCE, z >= V2_Z_NOT -> NOT THE
    # BACKGROUND SOURCE
    V2_K = 0.35
    V2_K_MAX = 0.5
    V2_SYS_ARCSEC = 130.0                        # the pulsar-measured 10-s systematic at V2_SYS_REF_MHZ (v2 :11)
    V2_SYS_REF_MHZ = 50.0
    V2_SYS_INDEX = -0.78
    V2_STAT_FWHM = 0.06
    V2_CTL_FLUX = (0.33, 3.0)
    V2_CTL_MIN_JY = 0.5
    V2_CUT_MARGIN_ARCMIN = 10.0
    V2_CTL_BRIGHTER = 2.0
    V2_VERY_BRIGHT_JY = 20.0
    V2_VERY_BRIGHT_DEG = 1.0
    V2_VERY_BRIGHT_SELF_ARCMIN = 1.5             # the control source's own entries are not 'very bright neighbors'
    V2_SRC_PAD_ARCMIN = 2.0                      # a job's sources: within CUT_HALF_ARCMIN + this (>= 0.25 Jy)
    V2_VERY_BRIGHT_PAD_ARCMIN = 62.0             # and its very bright ones within CUT_HALF_ARCMIN + this
    V2_KEY_PER_DEG = 40
    # v2's CONTROL DRAW. None = EVERY eligible control pair: the refactor chat's decision under Gregg's delegation, 2026-11-04
    # ("use ALL eligible controls, with no random cap. The cap was a speed convenience, and it makes knife-edge verdicts depend
    # on sampling noise"; ledger R-REFACTOR-20261104-FORCED-WIDE-CONTROLS-ORDER-INDEPENDENT (1)). (cap, seed) = at most cap pairs, drawn with numpy
    # default_rng(seed) when more are eligible: production (forced_wide_analyze.py CTL_MAX = 300, default_rng(20261102); the
    # parity config). The draw takes indices of a list in pool order, so ANY change to the job pool re-draws it - review F2: one
    # control re-indexed by the member change moved 44 of production's 300 for Struve 2398 A (I, 1 min), whose call is BACKGROUND
    # SOURCE in 55% of 2,000 random draws of 300; with all 476 eligible pairs power - z = 3.003 under both member rules
    V2_CTL_DRAW = None
    V2_CTL_MIN_CELLS = 3
    V2_CTL_MIN_N = 10
    V2_MIN_NIGHTS = 5
    V2_POWER_MIN = 3.0
    V2_Z_BG = 2.0
    V2_D_TG = 3.0
    V2_Z_NOT = 3.0
    WORKERS = 6                                  # processes (fork; outputs identical for any number: one generator per job)
    # at most this many processes (W-WORKERS): the test's workers read the Lustre burst files (each near candidate's centering
    # loads its job's npz), and Lustre reads yield to the recorders - as forced_wide_search.MAX_WORKERS (review F3)
    MAX_WORKERS = 6
    # wide_publish: each named source's position, the source of that name among the physical sources (>= S_MIN_JY) within
    # this of the target (publish_v3_to_runs.py:16-27); a name not found refuses the publication (W-NAMED)
    NAMED_RADIUS_ARCMIN = 200.0
    NULLABLE_KEYS = ('V2_CTL_DRAW', 'TAPER_CTL_MAX', 'BOOT_Z_SE', 'KNIFE_EDGE_SE', 'KNIFE_EDGE_DRAWS')
    RULED = {**{k: 'R-GREGG-20261103-FORCED-WIDE-V3-APPROVED' for k in (
        'MU', 'REF_MHZ', 'SPECTRAL_INDEX', 'CAPABLE_FRACTION', 'S_MIN_JY', 'R_CAND', 'NEAR', 'CUT_HALF_ARCMIN', 'CUT_EDGE_ARCMIN',
        'DISC', 'STRIP', 'BIN', 'SIDE', 'MIN_BIN', 'DISC_GRID', 'CTL_FLUX', 'CTL_MIN_JY', 'CTL_DIRS', 'CTL_KEY_PER_DEG',
        'CTL_BRIGHT_JY', 'CTL_TARGET_FWHM', 'MATCH_FACTORS', 'MATCH_FLOORS', 'MATCH_MIN', 'P_CONSISTENT', 'RATIO_CONSISTENT',
        'P_EXCESS', 'R_EXCESS', 'P_NIGHT', 'MIN_NOBS', 'NIGHT_CORE', 'NIGHT_FAR', 'NIGHT_CLEAR', 'NIGHT_MIN_CORE',
        'NIGHT_MIN_NIGHTS', 'CEN_DET', 'CEN_FLOOR', 'CEN_NIGHT', 'BOOTSTRAP', 'BOOT_MIN_NIGHTS', 'Z_OFF', 'Z_ON', 'Z_C_ON',
        'Z_C_NEAR', 'CENTER_WITHIN', 'COVER', 'V2_K', 'V2_K_MAX', 'V2_SYS_ARCSEC', 'V2_SYS_REF_MHZ', 'V2_SYS_INDEX',
        'V2_STAT_FWHM', 'V2_CTL_FLUX', 'V2_CTL_MIN_JY', 'V2_CUT_MARGIN_ARCMIN', 'V2_CTL_BRIGHTER', 'V2_VERY_BRIGHT_JY',
        'V2_VERY_BRIGHT_DEG', 'V2_VERY_BRIGHT_SELF_ARCMIN', 'V2_SRC_PAD_ARCMIN', 'V2_VERY_BRIGHT_PAD_ARCMIN', 'V2_KEY_PER_DEG',
        'V2_CTL_MIN_CELLS', 'V2_CTL_MIN_N', 'V2_MIN_NIGHTS', 'V2_POWER_MIN', 'V2_Z_BG', 'V2_D_TG', 'V2_Z_NOT')},
             'V2_CTL_DRAW': 'R-REFACTOR-20261104-FORCED-WIDE-CONTROLS-ORDER-INDEPENDENT',
             'CONTROL_ORDER': 'R-REFACTOR-20261104-FORCED-WIDE-CONTROLS-ORDER-INDEPENDENT',
             **{k: 'R-REFACTOR-20261104-FORCED-WIDE-CONTROLS-ORDER-INDEPENDENT-AMENDED-PRECISION-KNIFE-EDGE' for k in (
                 'TAPER_CTL_MAX', 'BOOT_Z_SE', 'BOOT_MAX', 'BOOT_DRAW', 'KNIFE_EDGE_SE', 'KNIFE_EDGE_DRAWS')}}


class night_activity(_Section):
    """Family N, stage activity: EACH NIGHT'S IONOSPHERIC ACTIVITY at an LST - per (LST, night, band) the night's snapshot 10-s
    Stokes I detections over the median of that band's other searched nights; per (LST, night) the median over its searched
    bands and its rank (figure_v2/night_activity_build.py, 2026-10-31; Gregg 2026-10-31 "Perhaps it is just necessary to see if
    the detections are clustered by night?"). Read by the astrometry figure v2 of every tier (astrometry_assoc.night_act)."""
    # the input: one row per detection of every LISTED source of the served snapshot 10-s Stokes I view - the flood rule's input
    # (flood_nights_20261031/flood_collect.py: the page built as the two-port server builds it; Run_SNAPCOMB_20261026, page
    # p8987). Rebuild from each new combined view (--na-detections <its table>)
    DETECTIONS = '/lustre/gh/main/phase3/refactor_analysis/flood_nights_20261031/flood_dets_p8987.csv'
    STOKES = 'I'                                 # every input row must be Stokes I, tier 10 s (N-INPUT)
    TIER = '10s'
    # de-duplication key: the frame id and the position's first DEDUP_CHARS characters (night_activity_build.py:16)
    DEDUP_CHARS = 9
    RATIO_FORMAT = '%.3g'                        # the ratio and activity columns (night_activity_build.py:36, :49)
    # where the figure reads it: astrometry_assoc.NIGHT_ACT (default <OUT_ROOT>/night_activity_I10s.csv) and its nights table
    # beside it (NIGHT_ACT_NIGHTS: the file name replaced) - the production --out and the two names
    OUT_ROOT = '/lustre/gh/main/phase3/night_activity'
    FILES = ('night_activity_I10s.csv', 'night_activity_nights_I10s.csv')
    PROTECTED_KEYS = ('FILES',)
