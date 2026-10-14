"""Configuration of `exopipe3 jupiter` - every tunable of the command, one section per stage or tier.

Units folded here: jupiter. A key in its section's PROTECTED_KEYS is locked
(a mirror of in-place code, a pinned input or a ruling-fixed value: it changes only by a commit); RULED names the
ledger ID of the ruling behind a key. Per run: --set SECTION.KEY=VALUE or --config FILE; `exopipe3 config --step jupiter` lists the sections.
"""
from exopipe.phase3.config import _Section


# =====================================================================================================================
# unit jupiter (was exopipe/phase3/_cfgdraft_jupiter.py)
# Configuration of `exopipe3 jupiter`.
#
# Sections: jupiter (roots, bands, interpreters, environment), jupiter_measure (Phase-1 measurement), jupiter_v2 (V2 Master
# variants, geometry, Blue anchor, metrics, site), jupiter_occurrence (Nancay occurrence maps), jupiter_flag (Phase-1 Jupiter
# flag backfill), jupiter_serve (8800 / 8801).
#
# Conventions of this draft (spec plan/specs/jupiter.md section 3):
# - Every value: unit, ruling (ledger fix_id where one exists) and the production file:line it mirrors.
# - PROTECTED_KEYS: no --set / --config override (config._Config.protected honours a section's PROTECTED_KEYS).
# - RULED: {KEY: ledger fix_id} - an override needs --ruling-for or a non-production --out (config/__init__.py).
# - PINNED-INPUT keys (jupiter.SUBLST_ROOT, CUBES_ROOT, EPHEM_DIR, DEEP_WEIGHTS): an override is allowed only into a non-production --out
#   (parity against the pre-bake reserve); the step refuses it into a production --out (lib/jupiter/common.py).
# - MIRROR keys: the in-place module reads its OWN literal. lib/jupiter/mirrors.py compares every MIRROR key with the module
#   literal at step start (refuses when they differ) and refuses an override of a MIRROR key unless the in-place driver can
#   set that module global (the redirectable roots); production runs set nothing.
# Values re-read 2026-10-13 at clv15 HEAD (phase3_jupiter_measure.py last changed 91f5b87), exopipe_ops HEAD (jupiter_server.py
# 6b8a23a), exopipe 7965688 Phase-1 files (sha256 pinned below), V2 code files on disk (/lustre/gh/main/phase3/jupiter/v2/code,
# not a git repository: identified per run by sha256 in the run record).
# =====================================================================================================================

class jupiter(_Section):
    """exopipe3 jupiter - the Jupiter reflection program (cross tier). Roots, bands, interpreters, environment."""
    # ---- product root (production; --out replaces it; under paths.PRODUCTION_ROOTS '/lustre/gh/main/phase3')
    ROOT = '/lustre/gh/main/phase3/jupiter'                      # program root
    PRODUCTS = 'products'                                        # Phase-1 cells <ROOT>/products/<cell>/; MIRROR phase3_jupiter_measure.py:33 (OUT_DEFAULT)
    PRODUCT_LOGS = 'products/logs'                               # per-stage logs + run.log; jupiter_phase1_run.sh:9 (L)
    V2 = 'v2'                                                    # V2 tree <ROOT>/v2; jupv2_masters.py:48, jint_metrics.py:124, jsite_render.py:65
    FLAGGED = 'phase1_flagged'                                   # Q12 copies; jupiter_flag_qa.py:56 (JOVIAN_ROOT), launcher :32 (JFQ_DEST)
    EPHEM_SUBDIR = 'ephem'                                       # where `ephem` writes: <out>/ephem/jupiter_ephem_<date>.csv (jup_ephem_fetch.py:25 D + 'ephem')
    # ---- pinned inputs
    EPHEM_DIR = '/lustre/gh/main/phase3/jupiter/ephem'           # PINNED-INPUT; Phase-0 Horizons cache read by measure, 40 nights 2024-12-18..2025-11-01; MIRROR phase3_jupiter_measure.py:32 (EPH)
    EPHEM_NAME = 'jupiter_ephem_{date}.csv'                      # phase3_jupiter_measure.py:50
    WINDOW_CSV = '/lustre/gh/main/phase3/jupiter/phase0/nights.csv'   # the Jupiter window (224 cells >= 30 deg); jup_nights.py; --cells pattern universe
    MIN_EL_DEG = 30.0                                            # deg, Jupiter at mid-hour; Gregg 2026-10-04 (IR:1230); JUPITER-PROGRAM-PHASE0-GO (ledger 1772)
    SUBLST_ROOT = '/lustre/gh/main/phase3/sublst'                # PINNED-INPUT; MIRROR phase3_jupiter_measure.py:30 (SL), build_cell.py:38, jblue_build.py:35, jblue_test.py:24, jint_metrics.py:130, jsite_lib.py:24, jsite_test.py:72
    CUBES_ROOT = '/lustre/pipeline/exopipe/phase3/snapshot_cubes'   # PINNED-INPUT; MIRROR phase3_jupiter_measure.py:31 (CUBES), jsite_lib.py:380, jblue_build.py:92
    DEEP_WEIGHTS = '/lustre/pipeline/exopipe/phase3/Coadd/Run_20260909_000000/{lst}/{sb}/weights.csv'   # PINNED-INPUT; V2 night weights; MIRROR jupv2_masters.py:53 (WEIGHTS_CSV)
    # ---- code roots of the in-place engines (refactor window; the clv15 / exopipe_ops roots come from the site file)
    PRODUCTION_EXOPIPE_ROOT = '/lustre/gh/main/exopipe'          # the production package the in-place children import (phase3_jupiter_measure.py:307; launcher :81, :90)
    V2_CODE_ROOT = '/lustre/gh/main/phase3/jupiter/v2/code'      # V2 code tree (outside git, B-87 default (b))
    PHASE0_CODE_ROOT = '/lustre/gh/main/phase3/jupiter/phase0'   # jup_ephem_fetch.py, jup_nights.py
    MEASURE_MODULE = 'phase3_jupiter_measure.py'                 # under the site's clv15 legacy root
    SERVER_MODULE = 'jupiter_server.py'                          # under the site's exopipe_ops legacy root
    # ---- bands and bodies
    SUBBANDS = ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz',
                '64MHz', '69MHz', '73MHz', '78MHz', '82MHz')    # MIRROR phase3_jupiter_measure.py:35-36 (BANDS)
    WIDEBANDS = ('Red', 'Green', 'Blue')                         # 10-min tier only (x2 residuals); MIRROR phase3_jupiter_measure.py:37 (WIDE); Full not measured
    ANCHOR_SUBBANDS = ('41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz')   # decimetric reference; phase3_jupiter_measure.py:38 (BANDS[5:])
    BODIES = ('jupiter', 'io', 'europa', 'ganymede', 'callisto')   # MIRROR phase3_jupiter_measure.py:39
    CADENCES = ('10s', '1min', '10min')                          # jupiter_server.py:23 (TIERS); jsite_lib.py:34
    WINDOWS_PER_HOUR = 6                                         # 10-min windows; phase3_jupiter_measure.py:165, :173 (range(6))
    # ---- process environment of the in-place engines
    ENV = {'NUMPY_MADVISE_HUGEPAGE': '0'}                        # jupiter_phase1_run.sh:8; masters/run_all.sh:5; occurrence/run_all.sh:7; run_integration.sh:18
    ENV_V2_EXTRA = {'PYTHONDONTWRITEBYTECODE': '1'}              # run_integration.sh:18
    ENV_FORBIDDEN = ('P3_COADD_RUN', 'P3_DEEP_SEARCH_STATE', 'EXOPIPE_JUPITER_FLAG_QA', 'JFQ_')   # gate J12 (names ending '_' are prefixes): phase3_tier_rules.py:36, :46; jupiter_flag_qa.py:340, :522; launcher :32-37
    # interpreter per stage: names of site PYTHON keys ('default' = py38_orca_nkosogor = the launchers' P8; 'flow' = py310_orca = P10)
    INTERPRETER = {'measure.extract': 'flow', 'measure.fit': 'default', 'measure.assign': 'default', 'measure.test': 'flow',   # jupiter_phase1_run.sh:12-14
                   'v2.masters.select': 'default', 'v2.masters.test': 'default', 'v2.masters': 'flow',                     # masters/run_all.sh:7-8, :19-25
                   'v2.anchor': 'default', 'v2.default': 'flow',                                                           # run_integration.sh:22-23, :57, :63
                   'occurrence': 'flow', 'ephem': 'default', 'flag': 'default', 'serve': 'default'}                        # occurrence/run_all.sh:8; launcher :30; live 8800/8801 argv (py38)
    CASA_IMPORT = 'casatasks'                                    # gate J8: the fit interpreter must import it (followup/photometry.py:22-27 turns a failed import into unconverged fits)
    PROTECTED_KEYS = ('ROOT', 'PRODUCTS', 'PRODUCT_LOGS', 'V2', 'FLAGGED', 'EPHEM_SUBDIR', 'EPHEM_NAME', 'WINDOW_CSV',
                      'PRODUCTION_EXOPIPE_ROOT', 'V2_CODE_ROOT', 'PHASE0_CODE_ROOT', 'MEASURE_MODULE', 'SERVER_MODULE',
                      'SUBBANDS', 'WIDEBANDS', 'ANCHOR_SUBBANDS', 'BODIES', 'CADENCES', 'WINDOWS_PER_HOUR', 'ENV_FORBIDDEN')
    RULED = {'MIN_EL_DEG': 'JUPITER-PROGRAM-PHASE0-GO'}


class jupiter_measure(_Section):
    """Phase-1 measurement (clv15/phase3_jupiter_measure.py, called in place). MIRROR keys checked against the module."""
    STAGES = ('extract', 'fit', 'assign', 'test')                # jupiter_phase1_run.sh:13
    CROP_ROOT = '/fast/gh/main/phase3/jupiter_phase1'            # node-local crops <CROP_ROOT>/<cell>/crops/ referenced by the products' crop column; MIRROR :34 (SCRATCH_DEFAULT); --work replaces it
    FIT_SNR = 6.0                                                # x local rms: crop for imfit at 10 s / 1 min when |value| >= this; MIRROR :40
    DET_SNR = 7.0                                                # sigma: 'detected' = converged imfit with |S/N| >= this (CLAUDE.md 5); MIRROR :41, :372
    CUT_DEG = 2.5                                                # deg, cutout half-size (5 x 5 deg); MIRROR :42
    CROP_DEG = 1.5                                               # deg, imfit crop half-size; MIRROR :43
    ASSIGN_PCT = 95.0                                            # offset percentile operating point of the body assignment; MIRROR :44 (PCT)
    RMS_ANNULUS_PX = (10, 40)                                    # pixels, local rms annulus (exclusive); literal inside local_rms :82 (checked by parity, not by mirrors.py)
    RMS_MIN_PIXELS = 200                                         # finite annulus pixels needed, else rms NaN; literal :83
    WIDE_REP_SUBBAND = {'Red': '27MHz', 'Green': '50MHz', 'Blue': '73MHz'}   # error-model band of a wideband row; literal :376
    SIGMA_FLOOR_ARCMIN = 1e-3                                    # arcmin; literal :383
    UNCLEAR_RANK = {'moons_only': 2, 'jupiter_and_moon': 1}      # Gregg 2026-10-04 (IR:1248-1250); literal :397-398
    TENMIN_FIT_ALWAYS = ('anchor', 'Green', 'Blue')              # 10-min rows fitted regardless of S/N when Stokes I in these bands; literal :325
    JUPMASK_REQUIRED = ('armA_resid', 'x2_subband_resid', 'window_master')   # RULING-MOVING-JUPITER-MASK-IN-MASTERS (1783), DEFECT-SUBLST-ARM-B-LOO-WITHOUT-JUPITER-MASK (1843); :169-170, :177, :277-278
    TIME_BASIS = 'file-name'                                     # frame time = file-name timestamp as midpoint (:72-75); RULING-GREGG-20261008-FRAMETIME-RERUN-REJECTED-ACCEPT-ERROR-UNTIL-MAJOR-REDO (2074)
    BAKED_INPUT = 'refuse'                                       # gate J7: the engine divides by FLUXDC (valid only for UNBAKED stacks, spec 12.1); 'refuse' is the only value implemented
    TEST_SAMPLE = (0.01, 5)                                      # TEST1: fraction of 10-s rows rebuilt, minimum count; default args of stage_test :423, :436
    TEST_SEED = 20261005                                         # default arg of stage_test :423
    TEST1_TOL = 1e-3                                             # worst |rebuilt - stored| / local rms; literal :455
    TEST2_TOL = 0.05                                             # |median(1-min frame-weighted mean / 10-min value) - 1|; revised form (ledger 1852); literal :473
    TEST2_MIN_SNR = 3.0                                          # literal :470
    TEST3_TOL = 1e-9                                             # arcmin, stored position vs imfit record; literal :489
    TEST4_MAX_MISSING = 0.01                                     # fraction of frames with data allowed absent; literal :506
    PARALLEL_CELLS = 6                                           # cells at once on one node (--workers default); jupiter_phase1_run.sh:7 (NPAR)
    FIT_RESUME = 'append-skip'                                   # fit appends to fits_<tier>.jsonl and skips crops already fitted (:314-323); B-49 default; stale case refused by gate J10
    PROTECTED_KEYS = ('STAGES', 'BAKED_INPUT', 'TIME_BASIS', 'FIT_RESUME')
    RULED = {'TIME_BASIS': 'RULING-GREGG-20261008-FRAMETIME-RERUN-REJECTED-ACCEPT-ERROR-UNTIL-MAJOR-REDO',
             'JUPMASK_REQUIRED': ('RULING-MOVING-JUPITER-MASK-IN-MASTERS', 'DEFECT-SUBLST-ARM-B-LOO-WITHOUT-JUPITER-MASK'),
             'TEST2_TOL': 'DEFECT-10MIN-RUNNER-STACKS-COMPLETE-ON-ADMISSION-FILES-AND-JUPITER-TEST2-REVISED'}


class jupiter_v2(_Section):
    """Jupiter V2 (the V2 code tree in place; one cell built). Master variants, geometry, Blue anchor, metrics, site."""
    CELLS_SUPPORTED = ('05h_2024-12-29',)                        # gate J20: decision 8 (rebuild the other nights) not started (RULINGS-JUPITER-DECISIONS-2-12, ledger 1883/1970)
    ROOT_LABELS = ('prod', 'V0', 'V1', 'V2', 'V2b', 'V3')        # run_integration.sh:28-30
    VARIANTS = ('V0', 'V1', 'V2', 'V2b', 'V3')                   # MIRROR jupv2_masters.py:54
    RULED_ROOT = 'V2'                                            # RULING-JUPITER-MASTER-V2-4BEAM (1882), site default (1899); MIRROR jsite_render.py:74 (DEFAULT_ROOT)
    ROOT_ORDER = ('V2', 'prod', 'V0', 'V1', 'V2b', 'V3')         # site.json order; MIRROR jsite_render.py:75
    FAR_BEAMS = {'V2': 4.0, 'V2b': 3.0}                          # x tonight's median Stokes-I BMAJ; RULED V2 4.0 (1882); MIRROR jupv2_masters.py:55
    V3_BEAMS = 2.0                                               # x BMAJ_I, no mask inside this disk (comparison root); MIRROR :56
    ARTEFACT_ANNULI_DEG = ((0.2, 1.0), (1.0, 3.0))               # deg around Jupiter, master metrics; MIRROR jupv2_masters.py:61-62 (ANN_IN, ANN_OUT)
    INTEGRATION_S = 10.03102041                                  # s, one 10-s integration (pairing by DATE-OBS + half); IMPLEMENTED-Q1-V2-STACKS-PAIRED-BY-INTEGRATION-1229 (2012); MIRROR jupv2_masters.py:510, jblue_build.py:91, jsite_lib.py:379
    MASTERS_STAGES = ('select', 'raw', 'masters', 'resid', 'fit', 'red10', 'metrics', 'mastersum', 'test', 'figs', 'bright', 'imprints')   # masters/run_all.sh:12 (+ bright_ratio.py, imprint_contrast.py, run by hand)
    MASTERS_NPROC = {'raw': 10, 'resid': 14, 'fit': 20, 'red10': 10, 'masters': 12}   # masters/run_all.sh:21-24; jupv2_masters.py:1511 default 12
    MASTERS_TEST_FAILURE_CONTINUES = True                        # run_all.sh:19 runs figs after a failed test; the unit is still REFUSED (B-30 default)
    ANCHOR_MEMBERS = ('64MHz', '69MHz', '73MHz', '78MHz', '82MHz')   # Blue product members; IMPLEMENTED-RULING7-BLUE-ANCHOR-PRODUCT-1229 (1902); MIRROR jblue_build.py:36 (BLUE)
    ANCHOR_NPROC = 16                                            # run_integration.sh:57
    ANCHOR_TEST = {'nrefit': 12, 'seed': 20261006}               # jblue_test.py:80-81 (defaults)
    GATE = (3, 15.0, 95.0)                                       # (min references, radius deg, percentile); decision 5 / Q4 (2010); MIRROR jint_metrics.py:131 (GATE_MIN_REFS, GATE_RADIUS_DEG, GATE_PCT)
    WANDER_MIN_PAIRS = 3                                         # same-frame I/V pairs for a wander term; MIRROR jint_metrics.py:133
    PROVISIONAL_K_MAD = 3.0                                      # T_scat, T_blue = median + 3 x 1.4826 MAD; PROVISIONAL, not ruled (2010 notes); MIRROR jint_metrics.py:134 (K_MAD)
    REF_INPUTS = {'05h_2024-12-29': ('integrate/inputs/ref_offsets_05h_2024-12-29.csv',
                                     'integrate/inputs/window_summary_05h_2024-12-29.csv')}   # per-cell reference-tie inputs under <ROOT>/v2; jint_metrics.py:128, jsite_lib.py:25, jsite_test.py:73
    REFCHECK_SCRIPT = 'integrate/inputs/jup_refcheck_20241229.py'   # under <ROOT>/v2: the scratch analysis that made REF_INPUTS (cell hard-coded at its line 12)
    HZ_API = 'https://ssd.jpl.nasa.gov/api/horizons.api'         # MIRROR hzfetch.py:31 (API)
    HZ_SITE = '-118.2817,37.2398,1.222'                          # lon deg E, lat deg, height km; B-42 default (c) keep copies; MIRROR hzfetch.py:32 (SITE)
    HZ_GRID = {'2024-12-29': ('01:20', '10:40', '06:15', '07:25')}   # UTC: 1-min grid G0..G1, 10-s check C0..C1 (hzfetch.py:22 usage); one entry per built night
    RENDER_TIERS = ('10s', '1min', '10min')                      # jsite_render.py:2235
    RENDER_NPROC = 12                                            # run_integration.sh:79 (jsite_render.py:2234 default 8)
    STRETCH_SIGMA = {'I': (-5.0, 8.0), 'V': (-5.0, 5.0)}         # x local rms, panel grey scale; MIRROR jsite_render.py:82 (STRETCH)
    R95_OVER_SIGMA = 2.448                                       # zoom circle radius = 2.448 x sigma (95 %); MIRROR jsite_lib.py:36
    ZOOM_FRAME = (20.0, (0.5, 0.06), 6.0)                        # (cap arcmin, margin (arcmin, fraction), min side arcmin); IMPLEMENTED-JUPITER-ZOOM-FRAME-PER-INTEGRATION-1229 (2007); MIRROR jsite_render.py:954-956
    REGION_LEVELS = (0.10, 0.25)                                 # CML-Io moon-region contours DRAWN for display; MIRROR jsite_render.py:1326
    ROOT_NOTES = {   # verbatim page notes per root (run_integration.sh:36-40); the prod note is jsite_render.py:68 (PROD_NOTE)
        'V0': 'V0 = production masked leave-one-out Master rebuilt (10 s/1 min identical to prod); 10 min = arm A (no x2, the ruled 10-min definition). Comparison root, non-conformant: V2 is the ruled Master (Gregg 2026-10-06).',
        'V1': "V1 = every other 05h night, no Jupiter masks (literal 'not adjacent'); 10 min = arm A (no x2, the ruled 10-min definition). Comparison root: V2 is the ruled Master (Gregg 2026-10-06).",
        'V2': "V2 = the RULED subtraction Master (Gregg 2026-10-06: 'Go with V2, the 4-beam Master'): nights whose Jupiter lies > 4 BMAJ from tonight's, no masks; its nights per band in the limits; 10 min = arm A without x2 and the one-night 18-32 MHz Master (2025-10-14) - both RULED 2026-10-07 (decision 1, 'Your rec accepted').",
        'V2b': 'V2b = nights with Jupiter > 3 BMAJ, no masks (admits 2025-01-11 at 27 MHz); 10 min = arm A (no x2, the ruled 10-min definition). Comparison root: V2 is the ruled Master (Gregg 2026-10-06).',
        'V3': "V3 = V0's nights and masks, no mask within 2 BMAJ of tonight's Jupiter (= V1 there); 10 min = arm A (no x2, the ruled 10-min definition). Comparison root: V2 is the ruled Master (Gregg 2026-10-06)."}
    SCRATCH_ROOT = '/fast/gh/main/scratch'                       # production node-local V2 scratch on lwacalim08 (jupv2_masters.py:49, jblue_build.py:38, jsite_render.py:66, jint_content_summary.py:41, jsite_crop_archive.py:104, build_maps.py:39); --work replaces it
    SCRATCH = {'masters': 'jupv2_masters', 'anchor': 'jupv2_anchor', 'site': 'jupv2_site', 'integrate': 'jupv2_integrate',
               'provenance': 'jupv2_provenance', 'occurrence': 'jupv2_occurrence', 'refcheck': 'jupref_20241229'}   # sub-directories of the scratch root
    PROTECTED_KEYS = ('CELLS_SUPPORTED', 'ROOT_LABELS', 'MASTERS_STAGES', 'REF_INPUTS', 'REFCHECK_SCRIPT', 'ROOT_NOTES', 'SCRATCH')
    RULED = {'RULED_ROOT': ('RULING-JUPITER-MASTER-V2-4BEAM', 'IMPLEMENTED-RULING-JUPITER-MASTER-V2-4BEAM-SITE-DEFAULT'),
             'ROOT_ORDER': 'IMPLEMENTED-RULING-JUPITER-MASTER-V2-4BEAM-SITE-DEFAULT',
             'FAR_BEAMS': 'RULING-JUPITER-MASTER-V2-4BEAM',
             'ANCHOR_MEMBERS': 'IMPLEMENTED-RULING7-BLUE-ANCHOR-PRODUCT-1229',
             'INTEGRATION_S': 'IMPLEMENTED-Q1-V2-STACKS-PAIRED-BY-INTEGRATION-1229',
             'GATE': 'IMPLEMENTED-DECISION5-GATE-NO-REFERENCE-VETO-RULED-WANDER-PRODUCT-BEAM-1229',
             'ZOOM_FRAME': 'IMPLEMENTED-JUPITER-ZOOM-FRAME-PER-INTEGRATION-1229'}


class jupiter_occurrence(_Section):
    """Occurrence maps from the Nancay decametric catalogue (V2 code tree, occurrence/, in place)."""
    STAGES = ('fetch', 'validate', 'digitize', 'maps', 'figures', 'lookup', 'test')   # occurrence/run_all.sh:12-13 order
    NANCAY_DIR = '/lustre/gh/main/clv15/jovian'                  # Marques et al. 2017 obs.dat / em.dat; MIRROR nda_catalog.py:23 (JOV)
    LITERATURE = ('/lustre/gh/main/clv15/jovian/marques2017_AA604A17.pdf', '/lustre/gh/main/clv15/jovian/aa33586-18.pdf',
                  '/lustre/gh/main/clv15/ROSES_2022_SSO_proposal_v6.2_subset.pdf')   # RULING9-MARQUES2017-PDF-COPIED (1891); MIRROR digitize_literature.py:31-33 (MARQUES, ZARKA, PROPOSAL)
    BINS = (72, 5.0)                                             # bins per axis, deg per bin (CML x Io phase); MIRROR build_maps.py:40-41 (NB, BIN)
    BOOTSTRAP = (200, 20261006)                                  # resamples, seed; MIRROR build_maps.py:43-44 (B_BOOT, SEED)
    DE_VARIANTS = ('all', 'deq', 'dpos', 'dneg')                 # Earth-latitude splits; MIRROR build_maps.py:45
    MERGE_GAP_MIN = 10                                           # min, Marques 2017 App. A event merging; MIRROR nda_catalog.py:35
    DROP_SESSIONS = (1198,)                                      # MIRROR nda_catalog.py:26
    HZ_STEP = (10, 600)                                          # (min per step, days per chunk); MIRROR fetch_horizons.py:12-13 (STEP_MIN, CHUNK_DAYS)
    TDB_MINUS_UTC_S = 69.184                                     # s; MIRROR validate_geometry.py:26 (TDB_MINUS_UTC)
    FIG_DPI = 200                                                # MIRROR make_figures.py:17 (DPI)
    RESULTS = 'occurrence/validation/content_test_results.json'   # under <ROOT>/v2; jint_disclose.py:30
    DISCLOSURE = 'integrate/occurrence_disclosed_failures.json'   # under <ROOT>/v2; waiver record compared exactly; jint_disclose.py:31
    ACCEPTED_FAILURES = ('L1.IoD', 'L2.ALL_pct', 'L5.IoC.peak')  # of 92 gating checks; IMPLEMENTED-RULING2-OCCURRENCE-MAPS-ACCEPTED (1900) amended Q18 (2011)
    GATING_TESTS = 92                                            # count of gating tests in the accepted record (2011)
    PROTECTED_KEYS = ('STAGES', 'NANCAY_DIR', 'LITERATURE', 'RESULTS', 'DISCLOSURE', 'ACCEPTED_FAILURES', 'GATING_TESTS')
    RULED = {'ACCEPTED_FAILURES': ('IMPLEMENTED-RULING2-OCCURRENCE-MAPS-ACCEPTED', 'IMPLEMENTED-Q18-OCCURRENCE-L8-REPLACED-ACCEPTANCE-RE-RECORDED')}


class jupiter_flag(_Section):
    """Phase-1 Jupiter check of Stokes-V-flagged integrations, archive backfill (Phase-1 package modules in place)."""
    TOOL = '/lustre/gh/main/exopipe/exopipe/phase1/jupiter_flag_backfill.py'   # launcher :26
    QA_MODULE = '/lustre/gh/main/exopipe/exopipe/phase1/jupiter_flag_qa.py'    # launcher :27
    TOOL_SHA256 = '93bec3aa259d0d9035f9ecdf58ec852ef725c80343d89b76fac5aef2d9f71294'   # tested version; launcher :28 (equal on disk 2026-10-13); gate J30
    QA_SHA256 = '5d5df3cd87682fcef8c4ae96f3615017f9bfaec01417b63247413e495d4300ae'     # launcher :29; gate J30
    STATE_DIR = '/lustre/gh/main/exopipe_ops/jupiter_flag_backfill_state'      # window.csv, shard_<node>.csv, run_<node>.log, summary_<node>.csv, VERIFY_<stamp>.txt, APPROVED; launcher :33 (JFQ_STATE)
    STATE_SUBDIR = 'jupiter_flag_backfill_state'                 # the state directory under a non-production --out (parity)
    APPROVAL_FILE = 'APPROVED'                                   # APPROVED-PREAMBLES-1-4-20261007 (1993); launcher :69; B-52 default keeps the file gate (J31)
    LIMIT_GB_TOTAL = 2000.0                                      # GB copied per launch, split equally over the nodes (integer GB per shard); APPROVED-PREAMBLES-1-4-20261007 "<= 2 TB"; launcher :34, :85
    MAX_HOURS = 12.0                                             # h per shard (timeout --signal=TERM); APPROVED-PREAMBLES-1-4-20261007 "<= 12 h"; launcher :35, :90
    NPROC = 16                                                   # processes per shard and for verify; launcher :36
    THREAD_ENV = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}   # launcher :52, :90
    VERIFY_DROP = ('WARNING', 'datfix', 'FITSFixed')             # lines left out of the VERIFY file (launcher :53 grep -v -E)
    PHASE1_ROOT = '/lustre/pipeline/exopipe/phase1'              # = paths.PHASE1_ROOT; MIRROR jupiter_flag_backfill.py:48
    CLEAN_EMPTY_FRACTION = 0.1                                   # CLEAN-I file < 10 % of the run's median size = flagged; MIRROR jupiter_flag_backfill.py:49
    RMS_TOLERANCE = 0.02                                         # relative, recomputed statistic vs recorded; MIRROR jupiter_flag_backfill.py:50
    QUANT_TOL = 0.02                                             # MIRROR jupiter_flag_backfill.py:424
    TEST_VERSION = 'JFQ-1 2026-10-07'                            # MIRROR jupiter_flag_qa.py:50
    QA_BOX_HALF_PX = 512                                         # central 1024 x 1024 px (the flag's own box); MIRROR jupiter_flag_qa.py:51 (QA_BOX_HALF)
    SEARCH_BEAMS = 1.0                                           # x BMAJ around the ephemeris pixel; IMPLEMENTED-RULING-Q12-PHASE1-JUPITER-FLAG-CHECK (1996); MIRROR jupiter_flag_qa.py:52
    MASK_BEAMS = 3.0                                             # x BMAJ; (1996); MIRROR jupiter_flag_qa.py:53
    DETECT_SNR = 10.0                                            # |peak V| / sigma_V; (1996); MIRROR jupiter_flag_qa.py:54
    HALF_INT_S = 5.0                                             # s, midpoint = DATE-OBS + 5 s; MIRROR jupiter_flag_qa.py:55
    COPY_REASONS = ('jupiter',)                                  # (1996); OPEN-Q12-BRIGHTEST-BURSTS-NOT-ATTRIBUTED (1999) option (a) = today; MIRROR jupiter_flag_qa.py:57
    PROTECTED_KEYS = ('TOOL', 'QA_MODULE', 'TOOL_SHA256', 'QA_SHA256', 'STATE_DIR', 'APPROVAL_FILE')
    RULED = {'LIMIT_GB_TOTAL': 'APPROVED-PREAMBLES-1-4-20261007', 'MAX_HOURS': 'APPROVED-PREAMBLES-1-4-20261007',
             'SEARCH_BEAMS': 'IMPLEMENTED-RULING-Q12-PHASE1-JUPITER-FLAG-CHECK', 'MASK_BEAMS': 'IMPLEMENTED-RULING-Q12-PHASE1-JUPITER-FLAG-CHECK',
             'DETECT_SNR': 'IMPLEMENTED-RULING-Q12-PHASE1-JUPITER-FLAG-CHECK', 'COPY_REASONS': 'IMPLEMENTED-RULING-Q12-PHASE1-JUPITER-FLAG-CHECK'}


class jupiter_serve(_Section):
    """Jupiter servers: the step owns 8800 (Phase-1 page) and 8801 (V2 static site) in the window (crosscheck D15: tell serve)."""
    PORTS = {'phase1': 8800, 'v2': 8801}                         # 8800 IR:1244 (BUILT-JUPITER-PHASE1-PAGE-8800); 8801 the V2 site (live argv read 2026-10-13 on lwacalim07)
    BIND = '127.0.0.1'                                           # reached by ssh tunnel (IF-SRV-05); jupiter_server.py:313; live 8801 argv '--bind 127.0.0.1'
    PHASE1_CWD = '/lustre/gh/main/exopipe_ops'                   # cwd of the live 8800 process (pid 100589, started 2026-10-05 09:29:52)
    PHASE1_LOG = 'products/logs/server_8800.log'                 # under the out root (the live 8800 has no log file)
    V2_SITE_DIR = 'v2/site'                                      # under the out root; jsite_render.py:65; live 8801 argv '--directory /lustre/gh/main/phase3/jupiter/v2/site'
    V2_SERVER_LOG = 'v2/site/server_8801.log'                    # the live 8801 log (stdout+stderr)
    PHASE1_RETIRED = False                                       # decision 8 ("Recommendation accepted"): retire 8800 after the other 23 nights are rebuilt (RULINGS-JUPITER-DECISIONS-2-12; not started)
    PARITY_PORT_OFFSET_MIN = 1000                                # --port-offset is 0 or >= this (parity servers never take a live port)
    TOL_S = {'10s': 5.0, '1min': 1e-3, '10min': 1e-3}            # s, integration grouping; MIRROR jupiter_server.py:40 (TOL; M33 fix 6b8a23a keeps it)
    STOP_WAIT_S = (10.0, 0.5)                                    # s: wait for the old pid after SIGTERM, poll interval (never SIGKILL; still alive = refused, gate J40)
    READY_POLLS = (60, 0.5)                                      # polls, s between: HTTP 200 readiness after a start
    HTTP_TIMEOUT_S = 60.0                                        # s per GET of the served content test (8800 renders on request)
    PROTECTED_KEYS = ('PORTS', 'BIND', 'PHASE1_CWD', 'V2_SITE_DIR', 'V2_SERVER_LOG')
