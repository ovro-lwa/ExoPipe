"""Configuration of `exopipe3 dewarp` - every tunable of the command, one section per stage or tier.

Units folded here: dewarp. A key in its section's PROTECTED_KEYS is locked
(a mirror of in-place code, a pinned input or a ruling-fixed value: it changes only by a commit); RULED names the
ledger ID of the ruling behind a key. Per run: --set SECTION.KEY=VALUE or --config FILE; `exopipe3 config --step dewarp` lists the sections.
"""
from exopipe.phase3.config import _Section


# =====================================================================================================================
# unit dewarp (was exopipe/phase3/_cfgdraft_dewarp.py)
# Config draft of `exopipe3 dewarp` (refactor window; config.py auto-loads config/steps/*.py until the refactor chat folds this
# section into config.py at '# @@STEP-SECTIONS@@'). Source: plan/specs/dewarp.md section 3. Every value names its unit, its
# ruling (fix_id of clv15/fix_ledger.csv) and the production file:line it was taken from (exopipe main 03ca797, exopipe_ops
# 0c1580c; MD = exopipe/phase3/lib/dewarp/master_dewarp.py, same line numbers as the production phase3_master_dewarp.py).
#
# Shared roots are NOT in this section (refactor chat decision 3, 2026-10-13 ~04:30 UTC): the deep working tree is
# paths.COADD_SB_ROOT (was dewarp.WORKING_ROOT), the archive Run of the screen record is paths.ARCHIVE_RUN (was
# dewarp.SCREEN_RECORD_RUN), the archive root paths.PHASE3_ARCHIVE_ROOT, and the ops root paths.OPS_ROOT (ROUTE_LIST is relative
# to it). CONTENT_CHECK_SCRIPT is relative to core.env.legacy_root('exopipe_ops').
#
# PROTECTED_KEYS (config.py honours it; decision 4) holds two groups:
#   - pinned inputs and the product names the window forbids renaming (DESIGN 11.4);
#   - the keys whose values the in-place modules (phase3_master_dewarp, cascade_dewarp, phase2_config, pb_apply, membership,
#     check_mdewarp_gen) read from their OWN constants during the window - an override would not reach them (DESIGN 11.1).
#     steps/dewarp.py also refuses to start when one of these differs from the in-place constant (gate G-D0). At absorption into
#     lib/ the in-place code takes them as arguments and they move from PROTECTED_KEYS to RULED only.
# =====================================================================================================================

class dewarp(_Section):
    """`exopipe3 dewarp` - deep master dewarp per LST (B6, contract step 21; DESIGN 2 row 3).

    Phase-2 RBF multiquadric cascade measured on the Stokes I Taper Robust -0.75 masters of one LST against VLSSr
    (R-MASTER-DEWARP-RBF-MULTIQUADRIC), acceptance (a)+(b') (R-MASTER-DEWARP-ACCEPTANCE-A-AND-B-PRIME), cubic-spline apply of
    the SAVED screen to every FITS product of an accepted routed cell (R-MDEWARP-CUBIC-REAPPLY), content test D1-D6 (D7 on an
    exact re-apply). WINDOW: the keys after '---- read in place' in PROTECTED_KEYS are read by phase3_master_dewarp /
    cascade_dewarp / phase2_config / pb_apply / membership / check_mdewarp_gen from their own constants: no override (DESIGN
    11.1). Shared roots: paths.COADD_SB_ROOT (working tree), paths.ARCHIVE_RUN, paths.PHASE3_ARCHIVE_ROOT, paths.OPS_ROOT."""
    from exopipe.common import phase2_config as _p2

    # ---- roots (production). --out re-roots gen/, measure/ and archive/ under the parity root; scratch from --work/site ----
    # (the deep working tree = paths.COADD_SB_ROOT: MD apply --root, phase3_master_dewarp.py:435; dewarp_lst.sh:11 ROOT, :59)
    REBUILD_ROOT = '/fast/gh/main/phase3/rebuild_20261007_promoted'   # path; promoted rebuilds of the round = MD --input-root
    #                                                          and the first staging root (dewarp_lst.sh:11 P, :25, :59;
    #                                                          chain_cell.sh:9)
    REBUILD_CHECK_ROOT = '/fast/gh/main/phase3/rebuild_20261007'   # path; holds <cell>/REBUILD_CONTENT_OK (T1-T8 passed)
    #                                                          (chain_cell.sh:9, :23; phase3_install.py:223-225)
    GEN_ROOT = '/lustre/gh/main/phase3/mdewarp_rbf_gen_20261007'   # path; MDEWARP_OUT_ROOT (dewarp_lst.sh:15, :17; read at
    #                                                          phase3_master_dewarp.py:264); also the step's production
    #                                                          record root (<GEN_ROOT>/_records/dewarp/)
    MEASURE_ROOT = '/fast/gh/main/phase3/mdewarp_rbf_20261008/from_lwacalim09'   # path; the measurement root of the round
    #                                                          (dewarp_lst.sh:15 MEAS07). New runs write a fresh
    #                                                          <MEASURE_ROOT>/<run_id>/ (never overwrite a screen, ledger
    #                                                          FINDING-ASTROM-FRAME-SCREENS-REWRITTEN-IN-PLACE-FROZEN-GEN4-COPY-20261010)
    STAGE_INPUTS = True                                      # bool; measure node-local copies, not the working tree
    #                                                          (dewarp_lst.sh:24-30; PyBDSF writes <file>.pybdsf.log beside
    #                                                          its input)
    MEASURE_FILES = ('I_deep_Taper_R075_{scope}.fits',
                     'I_deep_Taper_R075_pbcorr_{scope}.fits')   # names (protected); dewarp_lst.sh:25, :28;
    #                                                             phase3_master_dewarp.py:176

    # ---- pinned reference inputs (protected) ----
    VLSSR_PEAKS = '/lustre/gh/calibration/pipeline/reference/surveys/vlssr_radecpeak_unresolved.txt'   # path; 43,173 lines;
    #                                                          VLSSr peak metric (phase3_master_dewarp.py:57, :72); A-02-rev
    VLSSR_PEAKS_SHA1 = '27cde1b61dee013c06a346e046d423d67f5c25e1'   # sha1 of VLSSR_PEAKS, measured 2026-10-13 (gate G-D3)
    VLSSR_CATALOG = _p2.VLSSR_CATALOG                        # path; '/lustre/gh/calibration/pipeline/reference/surveys/
    #                                                          FullVLSSCatalog.text' (pipeline_config.py:61 via
    #                                                          phase2_config.py:16); anchor catalogue (phase3_master_dewarp.py
    #                                                          :452); R-MASTER-DEWARP-RBF-MULTIQUADRIC
    VLSSR_CATALOG_SHA1 = '6ed0113e57c0a42eb1425a4b56d657e5b4da15e7'   # sha1 of VLSSR_CATALOG, measured 2026-10-13 (G-D3)

    # ---- routing: which rebuilt cells are sent to the LST dewarp (batch row B-11; default = production's behaviour) ----
    ROUTE_SOURCE = 'screen_record'     # 'screen_record' = the cell's LATEST row of the archived screen record says applied=1:
    #                                    PRODUCTION since exopipe_ops e16201c (2026-10-13 04:57 UTC, chain_cell.sh:38-44;
    #                                    ledger IMPL-D1-COADD-BEAM-GATE-READS-COADD-IMAGE-AND-REBUILD-ROUND-20261013
    #                                    'CHAIN FIX': the list disagreed with the record for 7 of 32 cells) = B-11 option (a)
    #                                    as the deep chat implemented it | 'accepted_list' = the 2026-10-07 round's hand
    #                                    list (production until e16201c; B-11 option c)
    ROUTE_LIST = 'membership_rebuild_20261007/accepted_rebuilt.csv'   # path relative to paths.OPS_ROOT (protected); `lst,scope`
    #                                    CSV, 65 rows on 2026-10-13 (17 uncommitted); read only with ROUTE_SOURCE
    #                                    'accepted_list' (then fingerprinted into every cell record; dewarp_driver.sh:5)

    # ---- the archived screen record (the ONE source of truth for "dewarped cell", DESIGN 8a) ----
    # (the archive Run = paths.ARCHIVE_RUN: dewarp_lst.sh:17 (P3_ARCHIVE_RUN), :84; chain_cell.sh:14; install_cell.sh:11;
    #  membership.py:255. The class of the record file is install's 'screen_table' - crosscheck 4.2)
    SCREEN_RECORD_DIR = 'mdewarp_screens'       # name (protected); phase3_install.py:131-132
    SCREEN_RECORD_FILE = 'mdewarp_screens.csv'  # name (protected); phase3_install.py:137; membership.py:255
    SCREEN_STATUS_MAX_CHARS = 120               # characters of the status column; dewarp_lst.sh:83; phase3_install.py:185;
    #                                             no ruling found
    SCREEN_NOT_ACCEPTED_STATUS = ('rebuilt 2026-10-08 (membership round); re-measured, not accepted: status {status}, '
                                  'held-out gain {anchor_off_med_as} -> {cv_resid_med_as} arcsec')   # text; dewarp_lst.sh:82-83;
    #                                             'not measured' / '?' when the row or a value is absent; round wording, no
    #                                             ruling (spec Q8)

    # ---- measurement method [in-place: phase2_config / cascade_dewarp / phase3_master_dewarp] ----
    ANCHOR_SUBBAND = _p2.ANCHOR_SUBBAND                      # '73MHz'; phase2_config.py:72; R-MASTER-DEWARP-RBF-MULTIQUADRIC
    ANCHOR_FALLBACK = ('73MHz', '78MHz', '69MHz', '64MHz')   # try order; cascade_dewarp.py:102-116 (computed; 82 MHz never an
    #                                                          anchor); contract 5.1.1 (B-04 RATIFIED 2026-08-11)
    ANCHOR_FREQ_MHZ = _p2.ANCHOR_FREQ_MHZ                    # 73.0 MHz; phase2_config.py:127
    ANCHOR_MATCH_RADIUS_ARCMIN = float(_p2.ANCHOR_MATCH_RADIUS.to_value('arcmin'))   # 5.0 arcmin; phase2_config.py:128
    ANCHOR_FLUX_RATIO = (_p2.ANCHOR_FLUX_RATIO_MIN, _p2.ANCHOR_FLUX_RATIO_MAX)       # (0.80, 1.20) LWA/VLSSr peak flux; :129-130
    CASCADE_MATCH_RADIUS_BASE_ARCMIN = _p2.CASCADE_MATCH_RADIUS_BASE    # 5.0 arcmin at 73 MHz; phase2_config.py:136
    CASCADE_MATCH_RADIUS_EXPONENT = _p2.CASCADE_MATCH_RADIUS_EXPONENT   # 1.5 (radius x (73 MHz / nu)^1.5); :137
    CASCADE_FLUX_RATIO = (_p2.CASCADE_FLUX_RATIO_MIN, _p2.CASCADE_FLUX_RATIO_MAX)    # (0.80, 1.20) x the centre below; :138-139
    CASCADE_FLUX_CENTER = _p2.CASCADE_FLUX_CENTER            # 'measured-median'; :146; R-CASCADE-FLUX-WINDOW-MEASURED-MEDIAN
    MAX_CONSECUTIVE_FAILURES = _p2.MAX_CONSECUTIVE_FAILURES  # 3 refused steps halt an arm; :94; R-MASTER-DEWARP-RBF-MULTIQUADRIC
    MIN_WARP_ANCHORS = _p2.MIN_WARP_ANCHORS                  # 6 matched anchors to fit, and per held-out training fold; :153;
    #                                                          phase3_master_dewarp.py:154, :210
    OUTLIER_CLIP_SIGMA = _p2.OUTLIER_CLIP_SIGMA              # 3.0 sigma anchor clip; :157 (cascade_dewarp.py:445)
    OUTLIER_CLIP_ITERS = _p2.OUTLIER_CLIP_ITERS              # 3 iterations; :158 (cascade_dewarp.py:411)
    WARP_RBF_KERNEL = _p2.WARP_RBF_KERNEL                    # 'multiquadric'; :173; R-MASTER-DEWARP-RBF-MULTIQUADRIC
    WARP_RBF_SMOOTHING = _p2.WARP_RBF_SMOOTHING              # 1.5 (deep); :174; smoothing-closed
    WARP_RBF_EPSILON = _p2.WARP_RBF_EPSILON                  # None = mean anchor nearest-neighbour distance (pixels); :176
    WARP_RBF_SNR_WEIGHTING = _p2.WARP_RBF_SNR_WEIGHTING      # False; :180
    ANCHOR_MIN_ELEVATION_DEG = dict(_p2.ANCHOR_MIN_ELEVATION_DEG)           # {'13h': 20.0} deg; :229; 2026-06-11 diagnosis
    #                                                                         (no ledger ID found)
    ANCHOR_EAST_MIN_ELEVATION_DEG = dict(_p2.ANCHOR_EAST_MIN_ELEVATION_DEG) # {'13h': 35.0} deg; :245; FIX-13H-EAST-CUT-NEVER-ARMED
    #                                                                         records 45 deg - open for Gregg (spec Q6)
    ANCHOR_MAX_ISLAND_BEAM_AREAS = dict(_p2.ANCHOR_MAX_ISLAND_BEAM_AREAS)   # {'13h': 20.0} beam areas; :254;
    #                                                                         R-ANCHOR-ISLAND-CAP-13H; batch row B-64
    MIN_ANCHORS = 10                 # anchors after outlier rejection AND VLSSr peak matches before/after;
    #                                  phase3_master_dewarp.py:58, :197, :216, :234; R-MASTER-DEWARP-RBF-MULTIQUADRIC
    HELDOUT_MIN_GAIN_AS = 5.0        # arcsec; acceptance (b') = median anchor offset - median 5-fold held-out residual;
    #                                  phase3_master_dewarp.py:59, :243-245, :416; dewarp_lst.sh:47; check_mdewarp_gen.py:88;
    #                                  R-MASTER-DEWARP-ACCEPTANCE-A-AND-B-PRIME
    CV_FOLDS = 5                     # held-out folds; phase3_master_dewarp.py:141, :152; R-MASTER-DEWARP-ACCEPTANCE-A-AND-B-PRIME
    CV_SEED = 0                      # numpy default_rng seed of the fold permutation; phase3_master_dewarp.py:141, :151
    VLSSR_BORDER_FRACTION = 0.125    # of the frame side left out at each edge (n // 8); phase3_master_dewarp.py:74; A-02-rev
    #                                  (port of coadd _reference_astrom_ok)
    VLSSR_PEAK_FILTER_PX = 25        # pixels, maximum-filter size of the peak finder; phase3_master_dewarp.py:75
    VLSSR_N_PEAKS = 50               # brightest central peaks; phase3_master_dewarp.py:78
    VLSSR_CENTROID_BOX_PX = 9        # pixels, 9 x 9 centre-of-mass box; phase3_master_dewarp.py:80
    VLSSR_MAX_SEP_ARCMIN = 60.0      # arcmin, nearest-VLSSr match limit; phase3_master_dewarp.py:85
    VLSSR_TAIL_PERCENTILE = 90       # percent (p90_before / p90_after columns); phase3_master_dewarp.py:87
    SCREEN_STATS_BORDER_FRACTION = 0.125   # screen amplitude p50/p95 over the central 3/4 of the frame;
    #                                        phase3_master_dewarp.py:226
    BDSF_NCORES = 2                  # PyBDSF processes per extraction (read by cascade_dewarp.py:186-187 from the environment;
    #                                  the step sets it for the measurement); measure_on_09.sh:13
    MEASURE_THREADS = 1              # OMP / OpenBLAS / MKL threads in the measurement (RBF solves); measure_on_09.sh:13
    #                                  (gate G-D8; the step also caps the loaded BLAS pools with threadpoolctl)

    # ---- apply ----
    RESAMPLE_METHOD = 'cubic-spline' # MDWINTP card; phase3_master_dewarp.py:103; membership.py:256 (same ruling, second
    #                                  constant); R-MDEWARP-CUBIC-REAPPLY
    RESAMPLE_ORDER = 3               # spline order (scipy map_coordinates); NaN area = cascade_dewarp.apply_warp's (order 1);
    #                                  phase3_master_dewarp.py:118-119; R-MDEWARP-CUBIC-REAPPLY
    APPLY_RESIDUALS = True           # resample subtracted/ too (NO_RESIDUALS = False, phase3_master_dewarp.py:124, :268; no
    #                                  launcher passes --no-residuals); the step passes it to MD as MD.NO_RESIDUALS
    GEN_DIR = 'gen_mdewarp_rbf'      # name (protected; no renames in the window); phase3_master_dewarp.py:126;
    #                                  phase3_install.py:6
    GEN_TOKEN = '_mdewarp'           # name (protected); token before _<scope>.fits; phase3_master_dewarp.py:98-100; inverse
    #                                  phase3_install.py:89-90
    PB_FLOOR = 0.1                   # PB value at or below which a recomputed sibling is NaN when its header has no PBFLOOR
    #                                  card; pb_apply.py:139; phase3_master_dewarp.py:319

    # ---- content test D1-D7 [in-place: check_mdewarp_gen.py] ----
    CONTENT_CHECK_SCRIPT = 'membership_rebuild_20261007/check_mdewarp_gen.py'   # path relative to
    #                                  core.env.legacy_root('exopipe_ops'); run in place through core.legacy.run
    #                                  (dewarp_lst.sh:64); protected
    D5_DECIMALS = 3                  # written VLSSr median equals the row's med_after at 3 decimals (arcmin);
    #                                  check_mdewarp_gen.py:162; D5 delegated by Gregg 2026-10-05 'Pick what ever you recommend
    #                                  here' (R-MDEWARP-CUBIC-REAPPLY)
    D5_SRC_TOL_ARCMIN = 0.02         # arcmin, median per-source separation difference vs the prior generation (re-apply only);
    #                                  check_mdewarp_gen.py:52, :157
    D5_MIN_COMMON_PEAKS = 10         # common VLSSr peaks in the re-apply comparison; check_mdewarp_gen.py:158
    D6_POOL_RANGE = (0.995, 1.005)   # pooled median of 'peak height kept'; check_mdewarp_gen.py:52, :130;
    #                                  APPROVED-RESUME-SPACE-D6-REVISION
    D6_PROD_TOL = 0.015              # per product |median - 1| allowed (or the s.e. multiple below, if larger);
    #                                  check_mdewarp_gen.py:52, :121; APPROVED-RESUME-SPACE-D6-REVISION
    D6_PROD_SE_MULT = 3              # bootstrap standard errors; check_mdewarp_gen.py:121; APPROVED-RESUME-SPACE-D6-REVISION
    D6_MIN_SOURCES = 20              # bright compact sources for the per-product test; check_mdewarp_gen.py:52, :119;
    #                                  APPROVED-RESUME-SPACE-D6-REVISION
    D6_PEAK_RMS_MULT = 30.0          # x robust rms (1.4826 x MAD) of the inner region; check_mdewarp_gen.py:64, :66;
    #                                  R-MDEWARP-CUBIC-REAPPLY
    D6_PEAK_FILTER_PX = 9            # pixels; check_mdewarp_gen.py:66
    D6_INNER_FRACTION = 0.35         # inner radius / frame side; check_mdewarp_gen.py:60
    D6_MIN_INNER_PIXELS = 1000       # finite inner pixels, else no sources; check_mdewarp_gen.py:62
    D6_BOX_HALF_PX = 2               # pixels, +-2 px box at the source and at its moved position; check_mdewarp_gen.py:67-68
    D6_BOOTSTRAP_N = 500             # draws; check_mdewarp_gen.py:120
    D6_BOOTSTRAP_SEED = 0            # check_mdewarp_gen.py:99
    D6_EXEMPT_NAME_PARTS = ('_clean_',)   # exempt from the PER-PRODUCT test only (pooled test keeps them);
    #                                       check_mdewarp_gen.py:116-119; R-D6-ICLEAN-EXEMPT-AND-4-CELLS
    D4_MIN_SIBLINGS = 5              # recomputed pbcorr siblings required; check_mdewarp_gen.py:146; no ruling
    #                                  (RI-ops-deep-masters-018: T4 wants 6; spec Q7)
    D4_MIN_TESTED_PIXELS = 1         # NOT in-place: the step's own addition - pb_apply.sibling_check passes when 0 pixels are
    #                                  tested (pkg-masters-post defect D3); pure fix, refuses only an empty sibling (G-A10)
    TABLE_GAIN_TOL_AS = 0.1          # arcsec; NOT in-place: the dewarp_table content test re-derives (b') from the WRITTEN
    #                                  medians, each rounded to 0.1 arcsec (phase3_master_dewarp.py:227-228), so the written
    #                                  difference may differ from the unrounded gain the status was decided on by up to 0.1
    #                                  (an OK row may read 4.9, a REFERENCE ONLY row 5.0). No ruling (step's own tolerance).

    # ---- runner ----
    MAX_PARALLEL_LSTS = 2            # LST dewarps at once; dewarp_driver.sh:16, q28_driver_20261010.sh:21,
    #                                  ref14_driver_20261011.sh:23; no ruling ("the round's limit"); read by the runner chain

    RULED = {'ANCHOR_SUBBAND': 'R-MASTER-DEWARP-RBF-MULTIQUADRIC',
             'WARP_RBF_KERNEL': 'R-MASTER-DEWARP-RBF-MULTIQUADRIC',
             'MIN_ANCHORS': 'R-MASTER-DEWARP-RBF-MULTIQUADRIC',
             'MAX_CONSECUTIVE_FAILURES': 'R-MASTER-DEWARP-RBF-MULTIQUADRIC',
             'OUTLIER_CLIP_SIGMA': 'R-MASTER-DEWARP-RBF-MULTIQUADRIC',
             'WARP_RBF_SMOOTHING': 'smoothing-closed',
             'CASCADE_FLUX_CENTER': 'R-CASCADE-FLUX-WINDOW-MEASURED-MEDIAN',
             'ANCHOR_MAX_ISLAND_BEAM_AREAS': 'R-ANCHOR-ISLAND-CAP-13H',
             'ANCHOR_EAST_MIN_ELEVATION_DEG': 'FIX-13H-EAST-CUT-NEVER-ARMED',
             'HELDOUT_MIN_GAIN_AS': 'R-MASTER-DEWARP-ACCEPTANCE-A-AND-B-PRIME',
             'CV_FOLDS': 'R-MASTER-DEWARP-ACCEPTANCE-A-AND-B-PRIME',
             'RESAMPLE_METHOD': 'R-MDEWARP-CUBIC-REAPPLY',
             'RESAMPLE_ORDER': 'R-MDEWARP-CUBIC-REAPPLY',
             'D5_DECIMALS': 'R-MDEWARP-CUBIC-REAPPLY',
             'D6_PEAK_RMS_MULT': 'R-MDEWARP-CUBIC-REAPPLY',
             'D6_POOL_RANGE': 'APPROVED-RESUME-SPACE-D6-REVISION',
             'D6_PROD_TOL': 'APPROVED-RESUME-SPACE-D6-REVISION',
             'D6_PROD_SE_MULT': 'APPROVED-RESUME-SPACE-D6-REVISION',
             'D6_MIN_SOURCES': 'APPROVED-RESUME-SPACE-D6-REVISION',
             'D6_EXEMPT_NAME_PARTS': 'R-D6-ICLEAN-EXEMPT-AND-4-CELLS',
             'VLSSR_BORDER_FRACTION': 'A-02-rev'}

    # pinned inputs and names (DESIGN 11.4) + the keys read IN PLACE during the window (module docstring); steps_common.IN_PLACE
    # lists the second group and test_dewarp checks that every one of them is here
    PROTECTED_KEYS = ('MEASURE_FILES', 'VLSSR_PEAKS', 'VLSSR_PEAKS_SHA1', 'VLSSR_CATALOG', 'VLSSR_CATALOG_SHA1', 'ROUTE_LIST',
                      'SCREEN_RECORD_DIR', 'SCREEN_RECORD_FILE', 'GEN_DIR', 'GEN_TOKEN', 'CONTENT_CHECK_SCRIPT',
                      # ---- read in place (window) ----
                 'ANCHOR_SUBBAND', 'ANCHOR_FALLBACK', 'ANCHOR_FREQ_MHZ', 'ANCHOR_MATCH_RADIUS_ARCMIN', 'ANCHOR_FLUX_RATIO',
                 'CASCADE_MATCH_RADIUS_BASE_ARCMIN', 'CASCADE_MATCH_RADIUS_EXPONENT', 'CASCADE_FLUX_RATIO',
                 'CASCADE_FLUX_CENTER', 'MAX_CONSECUTIVE_FAILURES', 'MIN_WARP_ANCHORS', 'OUTLIER_CLIP_SIGMA',
                 'OUTLIER_CLIP_ITERS', 'WARP_RBF_KERNEL', 'WARP_RBF_SMOOTHING', 'WARP_RBF_EPSILON', 'WARP_RBF_SNR_WEIGHTING',
                 'ANCHOR_MIN_ELEVATION_DEG', 'ANCHOR_EAST_MIN_ELEVATION_DEG', 'ANCHOR_MAX_ISLAND_BEAM_AREAS', 'MIN_ANCHORS',
                 'HELDOUT_MIN_GAIN_AS', 'CV_FOLDS', 'CV_SEED', 'VLSSR_BORDER_FRACTION', 'VLSSR_PEAK_FILTER_PX',
                 'VLSSR_N_PEAKS', 'VLSSR_CENTROID_BOX_PX', 'VLSSR_MAX_SEP_ARCMIN', 'VLSSR_TAIL_PERCENTILE',
                 'SCREEN_STATS_BORDER_FRACTION', 'RESAMPLE_METHOD', 'RESAMPLE_ORDER', 'PB_FLOOR', 'D5_DECIMALS',
                 'D5_SRC_TOL_ARCMIN', 'D5_MIN_COMMON_PEAKS', 'D6_POOL_RANGE', 'D6_PROD_TOL', 'D6_PROD_SE_MULT',
                 'D6_MIN_SOURCES', 'D6_PEAK_RMS_MULT', 'D6_PEAK_FILTER_PX', 'D6_INNER_FRACTION', 'D6_MIN_INNER_PIXELS',
                 'D6_BOX_HALF_PX', 'D6_BOOTSTRAP_N', 'D6_BOOTSTRAP_SEED', 'D6_EXEMPT_NAME_PARTS', 'D4_MIN_SIBLINGS')
    del _p2
