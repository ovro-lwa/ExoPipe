"""Config draft of the catalog-tiers unit (exopipe3 catalog --tier sublst | snapshot) - REFACTOR WINDOW ONLY.

config.py loads this file (``_load_drafts``) until the refactor chat folds the four sections into config.py. Values are the
production values at exopipe_ops 0c1580c / clv15 39bdfd3 (spec plan/specs/catalog.md section 3, copied verbatim, plus the
path templates and interpreter roles the tier modules need, each with its source line).

Binding of each key during the window (the step calls clv15 / exopipe_ops modules IN PLACE, DESIGN 11.1):
  [env:X]   passed to the in-place module as environment variable X
  [arg:--x] passed as that command-line argument (passed only when it differs from the module's own default, so the
            command line equals production's while the values agree)
  [lit:F:N] a literal in the in-place module F at line N: the tier module reads it at start (AST, no import) and REFUSES on
            a difference; a --set of a [lit] key REFUSES until the module is absorbed (it would change nothing)
  [step]    used by the tier module's own (ported) code
``PROTECTED_KEYS`` (honoured by config.py since 2337938) lists the keys no --set / --config may change: the pinned inputs of the
spec's lower-case ``protected`` tuples AND every [lit] key (an override of a value the in-place module holds as a literal would
change nothing and make the record lie, spec section 3). ``RULED`` names the ledger ID of each ruled key (checked IDs).
Shared roots are config.paths (SUBLST_ROOT, SNAPSHOT_FLEET_OUT, SNAPSHOT_REVIEW_ROOT, OPS_ROOT, CLV15_ROOT, ...; refactor chat
decision 3) - no key here repeats one. Interpreters are the site file's PYTHON keys default / flow / py36 (decision 2), named in
the tier modules, not here.
Ownership (refactor chat decision 6): sub-LST parent catalogues are built by `masters --tier sublst --stages sources` (catalog
preflight only checks them); snapshot per-tier localization is `serve`'s in the window; the fit-store wideband quarantine is
`archive quarantine-rows` (catalog's store stage refuses unmarked rows); sub-LST tests (a)/(b) are masters/frames classes.
Paths: 'ops' = /lustre/gh/main/exopipe_ops, 'clv' = /lustre/gh/main/clv15.

Shared values NOT repeated here (they live in the catalog-deep draft's sections, per spec section 3): catalog.TABLES_SUBLST,
catalog.FOOTPRINT_CAT_DIR['sublst'], catalog_conform.{TABLE_PREFIX,TABLE_ELEV,TABLE_RECORD_ONLY,SIZE_STEP}['sublst'],
catalog_fitpos.{NPROC['sublst'],SUBLST_K_ARG}, catalog_select.PARENT_EXCLUSION, catalog_localize.{CATS,COMMON_FRAME,
SNAPSHOT_TIERS,SNAPSHOT_ALLOW_MODEL_ERRORS}, fitting.K_TEMPLATE[('sublst', S)]. The tier modules read them from there and
REFUSE, naming the key, when one is missing.
"""
from exopipe.phase3.config import _Section


class catalog_sublst(_Section):
    """sub-LST catalogue chain (sublst_catalog_chain.sh, sublst_catalogs.sh stages, sublst_catalog_publish.sh part 2)."""
    # pinned inputs (spec 'protected') + every [lit] key (bound to the in-place literal at preflight)
    PROTECTED_KEYS = ('TIER_RULE3_PIN', 'TIER_RULE3_PIN_MD5', 'COADD_RECORDS_FROZEN', 'MEMBERSHIP_VETO_PIN',
                      'MEMBERSHIP_VETO_PIN_MD5', 'FLUX_V1_TABLE', 'TIER_RULE4_RANGE', 'ELEV_FLOOR_DEG', 'REJECTED_SYS_ARCSEC',
                      'REJECTED_BEAM_ARCSEC')
    # the sub-LST root is config.paths.SUBLST_ROOT (/lustre/gh/main/phase3/sublst; sublst_catalogs.sh:22)
    ARM_TREES = {'A': 'tenmin_root', 'B': 'tenmin_root_x2', 'BI': 'tenmin_root_x2_inj'}   # under paths.SUBLST_ROOT; sublst_catalogs.sh:20 [env:P3_TENMIN_ROOT]
    ARM = 'B'                                             # served arm; sublst_catalog_publish.sh:38, sublst_resume_from_catalogs.sh:21 [step]
    CAT_ROOT = '/fast/gh/main/phase3/sublst_catalogs'     # node-local catalogue dirs catalog_/collect_/state_<SFX>; sublst_catalogs.sh:25 [step]
    LABEL = 'SLST{hh}{arm}'                               # sublst_catalogs.sh:24 [env:P3S_LABEL]
    SFX_19H_LEGACY = True                                 # 19h names omit the LST (E3); sublst_catalogs.sh:26, sublst_catalog_chain.sh:9 [step]
    SEARCH_STATE = '{ops}/sublst_search_state_{arm}'      # done/claims/failed under paths.OPS_ROOT; sublst_catalog_chain.sh:10 [step] -> search run records
    EXPECTED_JOBS = (36, 2)                               # 36 jobs less 2 per subband with NO_WINDOWS.txt; sublst_catalog_chain.sh:17-18 [step]
    CHAIN_MEM_GB = 100                                    # GB free in /user.slice; sublst_catalog_chain.sh:27 [step] -> runner admission --mem-gb
    CHAIN_MEM_WAIT_S = 43200                              # s; sublst_catalog_chain.sh:27 [step]
    FLUX_V1_TABLE = '/lustre/gh/main/phase3/sublst/catalog_inputs/shflux/flux_correction_v1_20260911.csv'   # unbaked frames only; sublst_catalogs.sh:68 [env:P3_FLUX_CORR_TABLE] (Gregg 2026-10-07 Q10)
    FLUX_BAKE_POLICY = 'branch'                           # 'branch' = baked/unbaked/mixed-refuses (today; B-72 b) | 'baked-only' (B-72 a); sublst_catalogs.sh:58-70 [step]
    TABLES_REQUIRED_METRIC = 'image-pixel'                # every sublst_ table key, elevation bands present; sublst_catalogs.sh:182-183 [step] (Gregg 2026-09-30/10-01)
    TIER_RULE3_PIN = '/lustre/gh/main/phase3/deep_search_refusals/Run_P3S5_pin_20261008T025911Z/deep_search_pin.csv'   # phase3_tier_rules.py:37 [lit] (RULINGS-GREGG-20261008-TIER3-FREEZE-B2-RECHAIN-LABELS)
    TIER_RULE3_PIN_MD5 = '29a53b8355146311b2b05954e9478a98'   # phase3_tier_rules.py:38 [lit]
    COADD_RECORDS_FROZEN = '/lustre/gh/main/phase3/sublst/deep_coadd_records_frozen_20261008'   # phase3_tier_rules.py:45 [env:P3_COADD_RUN unset -> lit]
    MEMBERSHIP_VETO_PIN = '/lustre/gh/main/phase3/sublst/deep_coadd_records_frozen_20261008/membership_vetoes_pinned.csv'   # phase3_tier_rules.py:54 [lit]
    MEMBERSHIP_VETO_PIN_MD5 = '851b16d86fe059c8c8bf43b1afa72344'   # phase3_tier_rules.py:55 [lit]
    TIER_RULE4_RANGE = (0.33, 3.0)                        # qa3_scale outside -> night refused when degenerate-flatDC; phase3_tier_rules.py:151 [lit] (RULING-TIER-RULE-4-FAILED-FLUX-SCALE; B-21 open)
    ELEV_FLOOR_DEG = 25.0                                 # deg at the window mid-time; gate_inputs.py:94 [lit] (IMPLEMENTED-10MIN-ELEVATION-FLOOR-25-DEG; B-25 open)
    PERTIE_SHARDS = 16                                    # CASA imfit shards; sublst_catalogs.sh:46-49 [arg:--shards]
    PERTIE_WORKERS = 16                                   # few-frame stage workers; sublst_catalogs.sh:48-49 [arg:--workers]
    PERTIE_TEST_PIXELS = 40                               # reference fits re-measured in the window pixels (T3); sublst_catalog_chain.sh:37 [arg:--pixels]
    PERTIE_TEST_LIGHTCURVES = 5                           # few-frame light curves re-read from the cubes (T7); sublst_catalog_chain.sh:37 [arg:--lightcurves]
    ASTROM_FRAME_TEST_SCREENS = 8                         # screens checked against their pixels; sublst_catalog_chain.sh:39 [arg]
    SERVED_BUILD = 'noveto'                               # pages read source_catalogs_noveto_* and localizations_noveto; sublst_pages_prep.py:69, :104 [step]
    BUILDS = ('noveto', 'veto')                           # today both are built (B-24 open); sublst_catalog_chain.sh:30-31 [step]
    RUN_COMPARE = True                                    # old-vs-new comparison, failures logged only (B-24 open); sublst_catalog_chain.sh:41-43 [step]
    RUN_REFTIE = True                                     # reference tie annotation, output unread (ruling S1 open; speed-up 3); sublst_resume_from_tie.sh:25 [step]
    REFTIE_SHARDS = 16                                    # sublst_resume_from_tie.sh:25 [env:P3_REFASTRO_SHARDS]
    REFTIE_MEMCAP_GB = 250                                # GB: the tie starts only while /user.slice holds less; sublst_resume_from_tie.sh:12 (CAP) [step] (the script waits; the step refuses until runner admission exists, K7)
    COMPARE_DIR = '{root}/compare_{lst}/new_{arm}/catalog'   # small products copied for pages (root = paths.SUBLST_ROOT); sublst_catalog_chain.sh:44-52 [step]
    REJECTED_SYS_ARCSEC = 60.0                            # rejected-record systematic (M32 open, B-39 open); emit_rejected_record.py:23 [lit]
    REJECTED_BEAM_ARCSEC = {'Red': 787.0, 'Green': 518.0, 'Blue': 374.0, 'Full': 528.0}   # emit_rejected_record.py:24 [lit]
    # (PARENT_REBUILD_NPROC dropped: the parent build is masters --tier sublst --stages sources, decision 6 / crosscheck group 9)
    # ---- path templates and interpreters the tier module needs (added by catalog-tiers; values = the launchers' own)
    TENMIN_TREE = '{root}/{tree}'                         # P3_TENMIN_ROOT = SL/tenmin_root[_x2[_inj]]; sublst_catalogs.sh:20 [env:P3_TENMIN_ROOT]
    SEARCH_DIR = '{root}/search_{arm}'                    # SRC, the arm's search outputs <cell>_<S>/; sublst_catalogs.sh:27 [step]
    COADD_DIR = '{root}/coadd_{lst}'                      # sublst_catalogs.sh:29 [env:P3_COADD_ROOT]
    PARENT_DIR = '{root}/parent_catalogs_{lst}'           # sublst_catalogs.sh:29 [env:P3_PARENT_DIR]; sublst_parent_catalogs.py:16 writes here
    ROOTS_CSV = '{root}/catalog_inputs/roots_{lst}.csv'   # sublst_catalogs.sh:30, sublst_catalog_chain.sh:36 [env:P3S2_ROOTS_CSV]
    STACK_ROOT = '{root}/{lst}'                           # window stacks SL/<LST>; sublst_catalogs.sh:46-48 [arg:--stack-root]
    COMPARE_ROOT = '{root}/compare_{lst}'                 # sublst_catalog_chain.sh:42 (old/ and new_<arm>/ under it) [step]
    RULED = {'TIER_RULE4_RANGE': 'RULING-TIER-RULE-4-FAILED-FLUX-SCALE',
             'ELEV_FLOOR_DEG': 'IMPLEMENTED-10MIN-ELEVATION-FLOOR-25-DEG',
             'FLUX_BAKE_POLICY': 'IMPLEMENTED-Q10-DEEP-FLUX-SCALE-BAKED-AND-Q1-FRAME-TIME-IN-10MIN-STACKS',
             'TABLES_REQUIRED_METRIC': ('R-SUBLST-TABLES-GO-ELEVATION-BANDS',
                                        'RULINGS-GREGG-20261013-DECISIONS-1-7-AND-TENMIN-TABLES'),
             'TIER_RULE3_PIN': 'RULINGS-GREGG-20261008-TIER3-FREEZE-B2-RECHAIN-LABELS',
             'BUILDS': 'RULING-GREGG-20261008-PER-CASE-TIE-EVERYWHERE-SUBBAND-FALLBACK-B2-FINISHES-FIRST'}   # both builds today (B-24 open)


class catalog_sublst_tie(_Section):
    """window tie (sublst_pertie.py) - every constant of the module, and its md5-pinned tables (PROTECTED)."""
    # every key of this section is a [lit] of sublst_pertie.py / astrom_frame.py or a pinned table: none is overridable
    PROTECTED_KEYS = ('WIDEN_DEG', 'REF_TRY', 'REF_N', 'TIE_MIN_REFS', 'TIE_CONSISTENCY_K', 'TIE_SYS_FLOOR_ARCMIN', 'ACCEPT_SNR',
                      'ACCEPT_OFFSET_BEAMS', 'STAT_COEFF_SNR_FLOOR', 'ELEV_BANDS_DEG', 'FALLBACK_PCT', 'FALLBACK_MIN_N', 'FEW_Z',
                      'FEW_FRAC', 'MAX_SIGNAL_FRAMES', 'CUBE_ROOTS', 'CALIB_GRID_STEP_PCT', 'CALIB_MIN_N', 'PAIR_CAP_PER_REF',
                      'RHO_PAIR_CAP', 'RADIUS99_EDGES', 'WINERR_SNR_EDGES', 'ERR_MIN_N', 'PINNED_TABLES', 'ERRMODEL_FITS',
                      'ERRMODEL_FITS_SHA256', 'VLSSR_CAT', 'VLSSR_SHA256', 'ASTROM_FRAME_INPUT_DIR', 'ASTROM_FRAME_INPUT_MD5_FILE',
                      'ASTROM_FRAME_SCREEN_MANIFEST_MD5', 'ASTROM_FRAME_BAND_FALLBACK_PCT')
    WIDEN_DEG = (10.0, 15.0, 20.0, 25.0, 30.0)            # deg reference search radii; sublst_pertie.py:77 [lit] (RULING-GREGG-20261008-ASTROMETRY-MEASURE-DONT-MOVE-BOTH-TIERS item 2)
    REF_TRY = (8, 6, 6, 6, 6)                             # candidates fitted per radius; sublst_pertie.py:79 [lit]
    REF_N = 3                                             # references used per detection; sublst_pertie.py:80 [lit]
    TIE_MIN_REFS = 2                                      # fewer accepted -> untied; sublst_pertie.py:81 [lit]
    TIE_CONSISTENCY_K = 3.0                               # x combined statistical sigma x sqrt 2 (or one beam); sublst_pertie.py:82 [lit]
    TIE_SYS_FLOOR_ARCMIN = 0.05                           # arcmin; sublst_pertie.py:83 [lit]
    ACCEPT_SNR = 7.0                                      # sigma, accepted reference fit; sublst_pertie.py:84 [lit]
    ACCEPT_OFFSET_BEAMS = 1.0                             # beams; sublst_pertie.py:85 [lit]
    STAT_COEFF_SNR_FLOOR = (0.42, 5.0)                    # sublst_pertie.py:86 [lit] (Gregg 2026-09-28)
    ELEV_BANDS_DEG = ((25.0, 35.0), (35.0, 50.0), (50.0, 90.01))   # sublst_pertie.py:87 [lit] (Gregg 2026-09-27)
    FALLBACK_PCT = 90.0                                   # untied -> measured 90th percentile (ruling item 3); sublst_pertie.py:88 [lit]
    FALLBACK_MIN_N = 20                                   # tied detections per bin, else refuse; sublst_pertie.py:89 [lit]
    FEW_Z = 5.0                                           # sigma, a signal frame; sublst_pertie.py:536 [lit] (RULINGS-GREGG-20261009-ASTROMETRY-FOLLOWUPS-1-8 item 1)
    FEW_FRAC = 0.5                                        # signal fraction in the signal frames; sublst_pertie.py:537 [lit]
    MAX_SIGNAL_FRAMES = 6                                 # frames (one minute); sublst_pertie.py:538 [lit]
    CUBE_ROOTS = ('/lustre/gh/main/phase3/sublst/cubes_deepflow_18_23', '/lustre/pipeline/exopipe/phase3/snapshot_cubes')   # deep-flow root first (E4); sublst_pertie.py:539 [lit]
    CALIB_GRID_STEP_PCT = 0.5                             # percent; sublst_pertie.py:944 [lit] (list comprehension: not bound by the AST check)
    CALIB_MIN_N = 200                                     # sublst_pertie.py:945 [lit]
    PAIR_CAP_PER_REF = 10                                 # sublst_pertie.py:946 [lit]
    RHO_PAIR_CAP = 60                                     # sublst_pertie.py:1040 [lit]
    RADIUS99_EDGES = (10.0, 20.0)                         # peak |SNR| bins; sublst_pertie.py:1043 [lit] (RULINGS-GREGG-20261010-TENMIN-SANDWICH-AND-NEW-LSTS)
    WINERR_SNR_EDGES = (8.0, 9.0, 10.0, 11.0, 12.5, 15.0, 20.0, 30.0, 50.0)   # sublst_pertie.py:1050 [lit]
    ERR_MIN_N = 100                                       # fits per (class, SNR bin) for the derivation; sublst_pertie.py:1052 [lit]
    PINNED_TABLES = {'sublst_pertie_fallback.json': '915333b506488b3804736c6d6c24046e',          # sublst_pertie.py:90-91
                     'sublst_pertie_frame_fallback.json': 'f3c74d0e89faf50650dde80abfac4065',    # :540-541
                     'sublst_pertie_calib.json': '14cda064493058caf6e23213976523d2',             # :942-943
                     'sublst_pertie_winerr.json': '247999ec443bde8961edd5ac22312e81',            # :1032-1033
                     'sublst_pertie_frame_scale.json': '447c049efc1475e3ffb13a23ffc0a8ce',       # :1034-1035
                     'sublst_pertie_rho.json': '5b3827c09a7692998b573d308b226f4c',               # :1036-1037
                     'sublst_pertie_radius99.json': '10805bc8dcfc59a5256b89b0a827bea2'}          # :1041-1042; read beside the module (_HERE, :76) [lit]
    ERRMODEL_FITS = '/lustre/gh/main/phase3/astrom_frame_inputs_20261008/tenmin_window_fits_20261008.csv.gz'   # sublst_pertie.py:1044 [lit]
    ERRMODEL_FITS_SHA256 = 'ea15c200294c0b0bde0dddcf5727524d85f80aa0cb30080c846c1f027e170153'   # sublst_pertie.py:1045 [lit]
    VLSSR_CAT = '/lustre/gh/main/catalogs/radio_surveys_20261006/FullVLSSCatalog.text'   # sublst_pertie.py:1046 [lit]
    VLSSR_SHA256 = 'c101910cea4d5addee258d8ca4600d9d1f7451a82f171415554ff075780b19fe'   # sublst_pertie.py:1047 [lit]
    ASTROM_FRAME_INPUT_DIR = '/lustre/gh/main/phase3/astrom_frame_inputs_20261008'   # astrom_frame.py:59 [env:P3_ASTROM_FRAME_INPUTS unset -> lit default]
    ASTROM_FRAME_INPUT_MD5_FILE = 'astrom_frame.py:60 INPUT_MD5'   # md5 dict of the frozen inputs, read from the module [lit]
    ASTROM_FRAME_SCREEN_MANIFEST_MD5 = 'f00d29e0d32fc99526f11a88b6e5dc73'   # astrom_frame.py:74 [lit]
    ASTROM_FRAME_BAND_FALLBACK_PCT = 90.0                 # astrom_frame.py:77 [lit]
    RULED = {'WIDEN_DEG': 'RULING-GREGG-20261008-ASTROMETRY-MEASURE-DONT-MOVE-BOTH-TIERS',
             'FALLBACK_PCT': 'RULING-GREGG-20261008-ASTROMETRY-MEASURE-DONT-MOVE-BOTH-TIERS',
             'STAT_COEFF_SNR_FLOOR': 'R-ASSOCIATION-RADIUS-QUADRATURE',
             'ELEV_BANDS_DEG': 'R-SUBLST-TABLES-GO-ELEVATION-BANDS',
             'FEW_Z': 'RULINGS-GREGG-20261009-ASTROMETRY-FOLLOWUPS-1-8',
             'RADIUS99_EDGES': 'RULINGS-GREGG-20261010-TENMIN-SANDWICH-AND-NEW-LSTS'}


class catalog_sublst_veto(_Section):
    """master-leak veto re-derived for windows (leak_veto_sublst.py); its output is not served (B-24 open)."""
    PROTECTED_KEYS = ('REF_PEAK_MIN_JY', 'EL_BINS_DEG', 'DEEP_WEIGHTS_CSV')   # [lit]; the [env] keys are overridable
    ELEV_FLOOR_DEG = 30.0                                 # deg, affected side where the Master has >= MASTER_MIN_JY; leak_veto_sublst.py:33 [env:ELEV_FLOOR] (R-LEAKVETO-IN-GATED-PATH, Gregg 2026-09-22)
    ENVELOPE_PCT = 97.0                                   # percentile of reference (frame - Master)/Master; leak_veto_sublst.py:34 [env:ENVELOPE_PCT]
    REF_MIN = 8                                           # references needed, else refuse; leak_veto_sublst.py:35 [env:REF_MIN]
    MASTER_MIN_JY = 0.5                                   # Jy; leak_veto_sublst.py:36 [env:MASTER_MIN_JY]
    REF_PEAK_MIN_JY = 1.0                                 # Jy, reference parents; leak_veto_sublst.py:153 [lit] (inline literal: not bound by the AST check)
    EL_BINS_DEG = ((0, 18), (18, 23), (23, 28), (28, 35), (35, 45), (45, 90))   # leak_veto_sublst.py:38 [lit]
    K_ASTROM = 12                                         # references for the local astrometry column; leak_veto_sublst.py:39 [env:K_ASTROM]
    R_ASTROM_DEG = 10.0                                   # deg; leak_veto_sublst.py:39 [env:R_ASTROM_DEG]
    DEEP_WEIGHTS_CSV = '/lustre/pipeline/exopipe/phase3/Coadd/Run_20260909_000000/{lst}/{band}/weights.csv'   # LIVE deep record, not the frozen one (B-20); leak_veto_sublst.py:32 [lit] (f-string of argv: not bound by the AST check)
    RULED = {'ELEV_FLOOR_DEG': 'R-LEAKVETO-IN-GATED-PATH'}


class catalog_snapshot(_Section):
    """snapshot batch chain (snapshot_rechain_batch.sh steps 0-5 + run test; phase3_snapshot_chain.py constants)."""
    # pinned inputs (spec 'protected') + every [lit] constant of phase3_snapshot_chain.py / refastro / the run content test;
    # [arg] and [env] keys stay overridable (passed only when they differ from the module default)
    PROTECTED_KEYS = ('CHECKSOURCES', 'CURVE_CONSTANTS', 'GHOST_DATES', 'GHOST_OFFSET_DEG', 'GHOST_SIGMA_DEG',
                      'MIN_ELEVATION_DEG', 'ATEAM_ANCHOR_DEG', 'ATEAM_MIN_DEG', 'ATEAM_SLOPE', 'ATEAM_FIXED_DEG', 'ATEAM_FLUX_NPZ',
                      'EPHEM_DEG', 'JUPITER_BEAMS', 'GHOST_MATCH_PERCENTILE', 'NEG_LOBE', 'VHF_FLOOR', 'VHF_EVENT_VETO', 'ASYM',
                      'E_MAX', 'VETO_RECORD_SNR', 'E_PROMOTE', 'XM_PERCENTILE', 'INSTANT_BIN_S', 'REPEAT_CONSTRAIN_K',
                      'REPEAT_SKY_RATE_ARCSEC_S', 'CHECKSOURCE_MATCH_ARCMIN', 'ASTROM_FIT_SNR', 'NO_SOURCE_SNR',
                      'REJECT_NO_SOURCE_FIT', 'REJECT_UNCONSTRAINED_FIT', 'CLUSTER', 'GATE3D', 'GATE3D_NTHR', 'LC_CACHE_DEFAULT',
                      'VETO_PROCS_DEFAULT', 'REFASTRO_SHARDS_DEFAULT', 'REF_MAX_SEP_DEG', 'TEST_GATE3D_PCT', 'TEST_MEMBER_ORDER',
                      'PRE_BUNDLE_TIE_MODEL')
    # ---- driver (snapshot_rechain_batch.sh). Shared roots: paths.SNAPSHOT_FLEET_OUT (the chain's --proto-root,
    # snapshot_rechain_batch.sh:280), paths.CLV15_ROOT (--samples-dir, phase3_snapshot_chain.py:3329), paths.SNAPSHOT_REVIEW_ROOT
    # (the served run, :56), paths.OPS_ROOT
    RUN_PREFIX = 'Run_SNAP'                               # snapshot_rechain_batch.sh:49 [step]
    WORK_ROOT = '/fast/gh/main/phase3'                    # snapshot_rechain_<NAME>/<NAME>; snapshot_rechain_batch.sh:51 [step]
    BATCH_DIR = '/lustre/gh/main/exopipe_ops/snapshot_batch'   # stores, tie cache, claims, rechain_<NAME>; snapshot_rechain_batch.sh:51 [step]
    BATCH1_STORE = '/fast/gh/main/phase3/snapshot_batch1/fits_store_batch1.csv'   # on the chain host only; snapshot_rechain_batch.sh:55 [step] (host in site file)
    MASTER_CATALOGS = '/lustre/gh/main/exopipe_ops/snapshot_master_catalogs'   # snapshot_rechain_batch.sh:280 [arg:--master-catalogs]
    CURVE_CONSTANTS = '/lustre/gh/main/clv15/snapshot_curve_constants.json'   # snapshot_rechain_batch.sh:280 [arg:--curve-constants] (B-19 open)
    LIGHTCURVES_ROOT = '/lustre/gh/main/exopipe_ops/snapshot_lightcurves'   # snapshot_rechain_batch.sh:280 [arg:--lightcurves-root]
    VETO_RECORD_ROOT = '{run}/vetoed'                     # snapshot_rechain_batch.sh:354 [arg:--veto-record-root] (chain default /lustre/pipeline/exopipe/phase3/snapshot_vetoed, phase3_snapshot_chain.py:3356)
    DIRTY_CELLS = '/lustre/gh/main/exopipe_ops/snapshot_fleet_worklist_dirtyI_20260918.txt'   # phase3_snapshot_chain.py:3343 [arg:--dirty-cells] (B-77 open)
    DIRTY_I_FLOOR = 0.0                                   # sigma; 0 = a dirty cell refuses; phase3_snapshot_chain.py:3346 [arg:--dirty-i-floor]
    LIMIT_H = 1000                                        # h runaway guard (Gregg: no time stop); snapshot_rechain_batch.sh:68 [step]
    FAST_FREE_MIN_GB = 400                                # GB free on the chain host's /fast; snapshot_rechain_batch.sh:170 [step]
    PRE_BUNDLE_TIE_MODEL = 'stat0.42floor5_refh10'        # refused tie-cache model; snapshot_rechain_batch.sh:65 [step]
    ELIGIBLE_STATUSES = ('TO-PREFIT', 'PREFIT-DONE', 'PREFIT-STALE', 'WRITER-ACTIVE')   # snapshot_rechain_batch.sh:99 [step]
    PREFIT_RELEASE_WAIT = (45, 120)                       # tries x s for the prefit writer to let go (90 min); snapshot_rechain_batch.sh:116-122 [step]
    STORE_QUIET_MIN = 11                                  # min a store must be unwritten; snapshot_rechain_batch.sh:178 [step]
    STORE_QUIET_WAIT = (16, 120)                          # tries x s; snapshot_rechain_batch.sh:177-181 [step]
    LIST_PARALLEL = 12                                    # per-cell repair / prefit lists at a time; snapshot_rechain_batch.sh:248, :286 [step]
    PREFIT_MAX_ITER = 3                                   # snapshot_rechain_batch.sh:283, :306 [step] (Gregg 2026-10-07 Q6)
    FIT_PROCS_RESERVE = 2                                 # fitter processes = free cores - 2 per node; snapshot_rechain_batch.sh:222 [step]
    TIE_SHARDS = (30000, 16, 8)                           # >= 30,000 detections at the fit step -> 16 imfit shards, else 8; snapshot_rechain_batch.sh:322 [env:P3_REFASTRO_SHARDS]
    TIE_CELLS_PER_NODE = 3                                # snapshot_rechain_batch.sh:330 [step]
    CHAIN_REFASTRO_SHARDS = 24                            # snapshot_rechain_batch.sh:338 [env:P3_REFASTRO_SHARDS]
    VETO_PROCS = 24                                       # batch script 24 (byte-identical at any count, chain.py:2723); b2 ran 8 after ENOMEM (exopipe_ops 6a88359) [env:P3_VETO_PROCS] (question Q6 open)
    LC_CACHE_MAX = 4000                                   # tables of ~30.7 MB; snapshot_rechain_batch.sh:338 [env:P3_LC_CACHE_MAX]
    MEM_MODEL = {'base_gb': 60.0, 'gb_per_m_loaded': 2.2, 'gb_per_m_fitstep': 10.0, 'gb_per_table': 0.031, 'tables_per_cell': 30,
                 'min_gb': 250, 'max_gb': 390}         # snapshot_rechain_batch.sh:345-350 [step]
    MEM_WAIT_S = 28800                                    # s; snapshot_rechain_batch.sh:352 [step]
    SERVED_PORT_CHECK = 8779                              # the served run whose cells must be kept; snapshot_rechain_batch.sh:103 [step] (serve config owns the port)
    # ---- chain (phase3_snapshot_chain.py constants, all [lit] unless an argument is named)
    STORE_FLOOR = 7.0                                     # sigma; :98 [arg:--store-floor] (Gregg 2026-09-22 "Use floor 7")
    SEED_FLOOR = 8.0                                      # sigma; :101 [arg:--seed-floor]
    MIN_ELEVATION_DEG = 30.0                              # deg; :106 (Gregg 2026-09-09)
    ASSOC_FLOOR = 7.0                                     # sigma; :107 [arg:--assoc-floor]
    REPEAT_FLOOR = 7.0                                    # sigma; :113 [arg:--repeat-floor]
    SINGLETON_FLOOR = {'I': 10.0, 'V': 8.0}               # sigma; :114-115 [arg:--singleton-floor(-v)] (Gregg 2026-09-09, 2026-09-13)
    ATEAM_ANCHOR_DEG = 3.5                                # deg Tau A anchor; :130 (Gregg 2026-09-05)
    ATEAM_MIN_DEG = 6.0                                   # deg; :131 (Gregg 2026-09-20)
    ATEAM_SLOPE = 0.18                                    # (F / F_TauA)^0.18; :132
    ATEAM_FIXED_DEG = {'HyaA': 2.5, 'HerA': 2.5}          # deg; :133
    ATEAM_FLUX_NPZ = '/lustre/gh/calibration/pipeline/workingv17/primary_calibrator_flux_models.npz'   # :134
    EPHEM_DEG = 2.0                                       # deg Sun (and Jupiter floor) at each frame's time; :137 (Gregg 2026-10-07 item 26)
    JUPITER_BEAMS = 8.0                                   # beam FWHM; :142 (Gregg 2026-09-11)
    GHOST_DATES = ('2024-12-28',)                         # :167 (RULINGS-GREGG-20261007-STATUS-ANSWERS Q9; B-79)
    GHOST_OFFSET_DEG = (-20.745, 6.854)                   # (dl, dm) deg about the zenith; :168
    GHOST_SIGMA_DEG = 0.065                               # :169
    GHOST_MATCH_PERCENTILE = 95.0                         # :170
    NEG_LOBE = (5.0, 0.25, 1.5)                           # rms, ratio (ratified 2026-09-20), beams; :183, :190, :192
    VHF_FLOOR = 10.0                                      # sigma; :198 (Gregg 2026-09-14)
    VHF_EVENT_VETO = False                                # annotation only; :203
    ASYM = {'angle_deg': 45.0, 'half_deg': 45.0, 'r_beams': (0.5, 1.5), 'snap_px': 1e-6, 'd': 0.08, 'z': 3.0}   # :210-223 (Gregg 2026-09-14)
    E_MAX = 0.03                                          # label threshold only; :234
    VETO_RECORD_SNR = 30.0                                # sigma; :238
    E_PROMOTE = 0.5                                       # :239
    XM_PERCENTILE = 95.0                                  # :240
    INSTANT_BIN_S = 10.0                                  # s (instants keyed by file-name ts: D3, parity keeps it); :241
    REPEAT_CONSTRAIN_K = 3.0                              # :245 (Gregg 2026-09-15)
    REPEAT_SKY_RATE_ARCSEC_S = 15.0                       # :246
    CHECKSOURCES = '/lustre/gh/main/exopipe_ops/snapshot_checksources.csv'   # :250 (R-SNAP-CHECKSOURCE-GATE-LPT-AND-PULSAR-IMPLEMENTED; B-78)
    CHECKSOURCE_MATCH_ARCMIN = 15.0                       # :2865, :2871 (no ruling; B-78) (inline literal: not bound by the AST check)
    STRUCT = {'k': 10.0, 'box_fwhm': 2.0, 'box_min_px': 15}   # :263-266 [arg:--struct-k]
    ASTROM_FIT_SNR = 7.0                                  # :756 (CLAUDE.md 5) (inline: not bound)
    NO_SOURCE_SNR = 8.0                                   # :960 (Gregg 2026-09-07) (inline: not bound)
    REJECT_NO_SOURCE_FIT = True                           # :947 (inline: not bound)
    REJECT_UNCONSTRAINED_FIT = True                       # :959 (RATIFIED-AND-IMPLEMENTED-EVERY-UNCONSTRAINED-FRAME-REMOVED-SNAPSHOT-TIER; M29 open) (inline: not bound)
    CLUSTER = (10.0, 15.0)                                # deg, min annotation window; :1495-1496 (inline: not bound)
    GATE3D = {'ratio': 2.0, 'snr': 8.0, 'parent_scale_refs': 10, 'match_percentile': 95.0, 'self_parent_all_bands': False,
              'ratio_high': 5.0, 'alpha': -1.5, 'alpha_min_sb': 4}   # :1792-1829 (Gregg 2026-09-18, 2026-10-04, 2026-10-07 Q15)
    GATE3D_NTHR = ((0.0, 3.0, 1), (3.0, 5.0, 1), (5.0, 10.0, 3), (10.0, 20.0, 23), (20.0, 50.0, 83), (50.0, float('inf'), 135))   # PROVISIONAL; :1795
    LC_CACHE_DEFAULT = 200                                # tables when P3_LC_CACHE_MAX unset; :1851 (inline: not bound)
    VETO_PROCS_DEFAULT = 16                               # :2723 (inline: not bound)
    REFASTRO_SHARDS_DEFAULT = 8                           # phase3_snapshot_refastro.py:354
    REF_MAX_SEP_DEG = 10.0                                # phase3_snapshot_refastro.py:47 ("nearby (within 10 degrees)")
    FLOW_MODE = True                                      # P3_SNAP_FIT_RAW unset -> flow-registered frames; :78 [env:P3_SNAP_FIT_RAW]
    # ---- run content test (snapshot_rechain_content_test.py R1-R7 operating points)
    TEST_GATE3D_PCT = 95.0                                # snapshot_rechain_content_test.py:71
    TEST_MEMBER_ORDER = 'time, subband, Stokes'           # snapshot_rechain_content_test.py:71 (Gregg 2026-10-07 item 23)
    # ---- added by catalog-tiers (values = snapshot_rechain_batch.sh's own; each a [step] key)
    NODE_FAST_ROOT = '/fast/gh/main/phase3'               # NODE_FAST = <root>/rechain_<NAME> on every node (tie outputs, fit shards); snapshot_rechain_batch.sh:58
    APPROVAL_MARKER = '/lustre/gh/main/exopipe_ops/snapshot_rechain_{name}.APPROVED'   # production approval marker (read only; --snap-approval-marker); snapshot_rechain_batch.sh:53
    CODE_SHA256 = '/lustre/gh/main/exopipe_ops/snapshot_rechain_{name}.code_sha256'    # tested-code fingerprint (P-G3, window form); snapshot_rechain_batch.sh:53, :75-76
    RULED = {'STORE_FLOOR': 'R-SNAP-STORE-FLOOR-8-SIGMA',          # 8 -> 7 by Gregg 2026-09-22 "Use floor 7" (no own ledger row found)
             'SINGLETON_FLOOR': 'R-SNAP-SINGLETON-FLOOR-10-SIGMA',
             'GHOST_DATES': 'RULINGS-GREGG-20261007-STATUS-ANSWERS',
             'NEG_LOBE': 'RATIFIED-NEG-LOBE-RATIO-0-25-AND-STRUCTURE-GATE-K-10-AND-ENV-OVERRIDES-REMOVED',
             'STRUCT': 'RATIFIED-NEG-LOBE-RATIO-0-25-AND-STRUCTURE-GATE-K-10-AND-ENV-OVERRIDES-REMOVED',
             'PREFIT_MAX_ITER': 'RULINGS-GREGG-20261007-STATUS-ANSWERS',
             'EPHEM_DEG': 'RULINGS-GREGG-20261007-STATUS-ANSWERS',
             'GATE3D': 'RULINGS-GREGG-20261007-STATUS-ANSWERS',
             'CHECKSOURCES': 'R-SNAP-CHECKSOURCE-GATE-LPT-AND-PULSAR-IMPLEMENTED',
             'REJECT_UNCONSTRAINED_FIT': 'RATIFIED-AND-IMPLEMENTED-EVERY-UNCONSTRAINED-FRAME-REMOVED-SNAPSHOT-TIER'}
