"""REFACTOR WINDOW config draft of `exopipe3 masters --tier sublst|snapshot` (unit masters-tiers). Loaded by
config._load_drafts() until the refactor chat folds it into config.py at @@STEP-SECTIONS@@.

Source of every value: plan/specs/masters.md section 3 (classes masters_sublst, masters_snapshot), adapted to the DECISIONS OF
THE REFACTOR CHAT (plan/specs/_CROSSCHECK.md, 2026-10-13 ~04:30 UTC): shared roots come from CFG.paths (SUBLST_ROOT,
SNAPSHOT_CUBES_ROOT, COADD_SB_ROOT, TRANSIENTS_ROOT, PHASE3_ARCHIVE_ROOT, OPS_ROOT) - never a key of this draft; interpreters
are site PYTHON keys default/flow/py36/casa/tess only; protection by PROTECTED_KEYS / RULED; sub-LST wideband and content
test (b) belong to frames, sub-LST loo and sources to masters; the per-hour sub-LST night stacks have ONE wrapper
(lib/masters/tier_sublst.stack_night) and ONE section (masters_sublst) that frames imports.

Values are today's production values with the production file:line they come from. Line numbers cite clv15 39bdfd3 and
exopipe_ops 640f677 (2026-10-13):
  stack = clv15/phase3_sublst_stack.py     loo = clv15/phase3_sublst_loo.py     rules = clv15/phase3_tier_rules.py
  jupmask = clv15/phase3_jupiter_mask.py   nocat = exopipe_ops/sublst_run_lst_nocat.sh (the admit heredoc is nocat:82-186)
  prep = exopipe_ops/sublst_prep_lst.sh    catpub = exopipe_ops/sublst_catalog_publish.sh
  freeze = exopipe_ops/snapshot_masters_freeze.py   snapsearch = clv15/phase3_snapshot_search.py
  install = exopipe/phase3/phase3_install.py (main checkout)
"Ledger" IDs are fix_ids of clv15/fix_ledger.csv (checked by tests/test_masters-tiers.py with core.ledger.has_id).

Values these tiers read from the step owner's draft (_cfgdraft_masters-deep.py), NOT repeated here: masters.STAGES /
masters.DEFAULT_STAGES (stage order), masters.PYTHON_SITE_KEY (engine interpreters), masters_sources.SUBLST_NPROC (processes of sublst_parent_catalogs.py, catpub:65),
masters_pbmaps.PB_LIBRARY (the model-beam PB library, one name for both tiers).

PROTECTED_KEYS holds two kinds of key, both refused by config.py on --set / --config (exit 2):
  pins     frozen, checksummed inputs a ruling names (tier-rule pins, the frozen record and master trees, the manifest sha1)
  mirrors  values that are LITERALS inside the in-place clv15 / exopipe_ops code (no argument carries them): an override
           could be recorded but never applied, so it is refused until the module is absorbed into lib/ and reads config.
           The tier refuses to run when an in-place literal differs from its mirror (the source text is read:
           lib/masters/tier_sublst.MIRRORS, tier_snapshot.MIRRORS).
"""
from exopipe.phase3.config import _Section


class masters_sublst(_Section):
    """masters --tier sublst (10-minute windows; tier C): prep, admit, stack, loo, sources (wideband: frames). In place:
    clv15 phase3_sublst_{stack,loo}.py and exopipe_ops sublst_parent_catalogs.py run by core.legacy.run with the production
    command lines; the shell logic of sublst_prep_lst.sh, the admit heredoc and gates of sublst_run_lst_nocat.sh and the pcheck
    of sublst_catalog_publish.sh ported into lib/masters/tier_sublst.py. frames --tier sublst (per hour) imports
    tier_sublst.stack_night and reads THIS section for the night's window stacks."""
    # ---- input locations below the shared roots (paths.SUBLST_ROOT, paths.TRANSIENTS_ROOT) -------------------------------
    FROZEN_RECORDS_DIR = 'deep_coadd_records_frozen_20261008'   # <paths.SUBLST_ROOT>/<this>/<LST>/<sb>/weights.csv: frozen deep
                                                                # co-add records (stack:71; loo:36; rules:45; nocat:151); ruling B 2026-10-08
    DEEPFLOW_CUBES_DIR = 'cubes_deepflow_18_23'   # <paths.SUBLST_ROOT>/<this>: deep-Master re-flow cubes, 19h 18/23 MHz (nocat:76, :93)
    PREP_PARENTS_RUN = 'Run_P3S4'             # <paths.TRANSIENTS_ROOT>/<this>/parent_catalogs: footprint fallback and parent copies
                                              # (prep:43, :50); B-36 default (c)
    PREP_FOOTPRINT_DIR = '/fast/gh/fp_artifact_investigation/support/bdsf_catalogs_aug'   # August footprint catalogues (prep:40);
                                              # lwacalim07-local; B-36 default (c)
    # The frozen generation-4 masters the cubes were flowed to are masters_snapshot.FROZEN_ROOT (stack:74 MASTER_PATH; prep:29).
    # ---- pins read by the in-place tier rules (rules:37-55) ---------------------------------------------------------------
    MEMBERSHIP_VETO_PIN_MD5 = '851b16d86fe059c8c8bf43b1afa72344'   # md5 of <frozen records>/membership_vetoes_pinned.csv, 127 rows (rules:55)
    DEEP_SEARCH_PIN = '/lustre/gh/main/phase3/deep_search_refusals/Run_P3S5_pin_20261008T025911Z/deep_search_pin.csv'   # tier rule 3 pin (rules:37)
    DEEP_SEARCH_PIN_MD5 = '29a53b8355146311b2b05954e9478a98'      # (rules:38)
    # ---- the stacker (stack) and the admit predictor (nocat:82-186) ------------------------------------------------------
    WINDOW_MIN = 10.0                         # min, sidereal window length (stack:197 --window-min default); a mirror because
                                              # loo:82 hard-codes six windows (spec section 3 lists it as an argument)
    N_WINDOWS = 6                             # windows per LST hour (loo:82)
    BEAM_AREA_MAX = 1.5                       # x the night's median beam area, both Stokes (stack:64)
    NOISE_K = 5.0                             # robust sigma, per-Stokes noise outlier (stack:65; Gregg 2026-09-09)
    BLANK_FINITE_MIN = 0.5                    # finite-pixel fraction below = blank frame (stack:66)
    QA3_ADMIT_FLOOR = 0.1                     # deep record qa3_scale must exceed this (stack:232; nocat:160) - admission only since Q10
    TIER_RULE4_RANGE = (0.33, 3.0)            # tier rule 4: degenerate-flatDC with qa3_scale outside -> refused (rules:151); B-21 default (a)
    MIN_USABLE_FRAMES = 10                    # frames passing the blank, beam and night gates (stack:353; nocat:181)
    FLUXDC_WAIT_S = 14400                     # s, a V job waits this long for its night's I flux_dc record (stack:279, env
                                              # SUBLST_FLUXDC_WAIT_S); the unit runs I before V, so the wait never triggers (B-76 (b))
    PRODUCT_CARD = 'sublst-v0-test'           # PRODUCT card of every window (stack:450); B-29 default (b)
    # s, one 10-s integration as the stacker and the admit heredoc time frames (stack:77; nocat:94: frame time = DATE-OBS + this/2).
    # NOT instrument.INTEGRATION_S (10.031 s = phase2_config.LWA_DT): the two differ by 2.0e-5 s; the admit port uses this value
    # so its frame times (rounded to 0.1 s, floored to the second for the night rule) equal the stacker's (impl note, finding F1)
    INTEGRATION_S = 10.03102041
    # ---- window masters + leave-one-out (loo) ------------------------------------------------------------------------------
    LOO_MIN_WEIGHTED_NIGHTS = 2               # a window Master needs >= 2 nights and >= 2 weighted nights (loo:93; Gregg 2026-10-06)
    PER_NIGHT_MIN = 2                         # a residual's own LOO master needs >= 2 weighted nights (loo:140; Q11 2026-10-07)
    WINDOW_COADDOP = 'sublst-ivw-window-mean-v2-jupmask'   # COADDOP of every window Master (loo:122)
    JUPITER_MASK_RADIUS_BEAMS = 2.0           # x BMAJ, moving Jupiter mask per night in the Masters (jupmask:14)
    # The loo stage starts only when every job of the LST's admit plan has a stack record. Production stops the LST when ANY job
    # lacks its completion line after two catch-up passes - a refused stack job included (nocat:208-218: 'REFUSING: N stack jobs
    # without a completion line'). False = that behaviour. True = the spec's G-S05 reading ('every planned stack job done OR
    # refused'): window Masters are then built without the refused (night, subband). Open question for Gregg (impl note Q2).
    LOO_ADMITS_REFUSED_STACKS = False
    # ---- prep (port of sublst_prep_lst.sh) ---------------------------------------------------------------------------------
    PREP_WIDEBAND = ('Red', 'Green', 'Blue')  # wideband groups whose masters prep copies (SUBLST_PREP_WIDEBAND default, prep:22)
    PREP_FILES = ('I_deep_Taper_R0_{b}.fits', 'I_deep_Taper_R075_{b}.fits', 'V_deep_Taper_R0_{b}.fits', 'PB_deep_{b}.fits')   # from the frozen masters (prep:30)
    PREP_MIN_PARENTS = 15                     # parent catalogue files copied per LST, fewer refuses (prep:51)
    WEIGHTS_FOR_SEARCH = 'live-at-prep'       # B-20 default (b): prep copies the LIVE weights.csv / residuals_provisional.csv from
                                              # paths.COADD_SB_ROOT (prep:32-37); the only value implemented ((a) 'frozen' needs Gregg)
    # ---- orchestration -----------------------------------------------------------------------------------------------------
    # Interpreters of the in-place engines: masters.PYTHON_SITE_KEY['sublst-stack'|'sublst-loo'|'sources'] (step owner's section,
    # py36 / py36 / default = sublst_stack_pass.sh:10, nocat:28, catpub:21) - one name per value; this section adds test (a)'s:
    TEST_WINDOWS_PYTHON = 'py36'              # site PYTHON key of sublst_reserve_test_windows.py (catpub:21 PYD, :87)
    STACK_JOB_UNIT = 'two-jobs-I-then-V'      # B-76 default (b): the I and V stacker invocations stay separate; I runs first
    PARALLEL = dict(stack=4, loo=8)           # units at once in one process when --workers is not given
                                              # (sublst_stack_pass.sh:8 NPAR; nocat:230 xargs -P 8)
    # ---- content tests (lib/masters/tests_sublst.py) ----------------------------------------------------------------------
    # Production content test (a) (catpub:87): windows = frames x stated factors on a seeded sample of n (night, subband) cells.
    WINDOW_SAMPLE_TEST = dict(n=3, seed='')   # sublst_reserve_test_windows.py --n / --seed defaults (its :23)
    # Per-file tolerances. Window ones are those of test (a) (sublst_reserve_test_windows.py:67-68, :98, :112-115): frame time
    # 1e-3 s, MJD-MID 1e-8 d, pixel factor vs the Master's 1e-9, FLUXDC vs the I record 1e-12; noise_rel: NOISEMED vs
    # median(admission noise_jyb) / FLUXDC x B (the CSV keeps 6 significant digits). The LOO identity (new, pure): residual =
    # stack - (M W - w a) / (W - w) on pixels finite in stack and Master, within tol_rms x the residual's robust rms, on at
    # least frac_min of them (Jupiter-masked pixels differ by construction; measured on 09h 18 MHz: impl note).
    WINDOW_TEST_TOL = dict(time_s=1e-3, mjd_d=1e-8, factor=1e-9, fluxdc=1e-12, noise_rel=1e-5)
    LOO_IDENTITY = dict(frac_min=0.99, tol_rms=1e-3)
    PROTECTED_KEYS = (
        # pins
        'FROZEN_RECORDS_DIR', 'MEMBERSHIP_VETO_PIN_MD5', 'DEEP_SEARCH_PIN', 'DEEP_SEARCH_PIN_MD5',
        # mirrors of literals in the in-place code
        'WINDOW_MIN', 'N_WINDOWS', 'BEAM_AREA_MAX', 'NOISE_K', 'BLANK_FINITE_MIN', 'QA3_ADMIT_FLOOR', 'TIER_RULE4_RANGE',
        'MIN_USABLE_FRAMES', 'PRODUCT_CARD', 'INTEGRATION_S', 'LOO_MIN_WEIGHTED_NIGHTS', 'PER_NIGHT_MIN', 'WINDOW_COADDOP',
        'JUPITER_MASK_RADIUS_BEAMS')
    RULED = {'FROZEN_RECORDS_DIR': 'IMPLEMENTED-RULINGS-A-B-20261008-SUBLST-FLOWREF-AND-MEMBERSHIP-VETOES',
             'MEMBERSHIP_VETO_PIN_MD5': 'IMPLEMENTED-RULINGS-A-B-20261008-SUBLST-FLOWREF-AND-MEMBERSHIP-VETOES',
             'DEEP_SEARCH_PIN': 'RULINGS-GREGG-20261008-TIER3-FREEZE-B2-RECHAIN-LABELS',
             'DEEP_SEARCH_PIN_MD5': 'RULINGS-GREGG-20261008-TIER3-FREEZE-B2-RECHAIN-LABELS',
             'TIER_RULE4_RANGE': 'RULING-TIER-RULE-4-FAILED-FLUX-SCALE',
             'MIN_USABLE_FRAMES': 'IMPLEMENTED-SUBLST-JOB-BUILDER-RULE-REFUSALS-AND-DEEPFLOW-CUBES-20261011',
             'DEEPFLOW_CUBES_DIR': 'IMPLEMENTED-SUBLST-JOB-BUILDER-RULE-REFUSALS-AND-DEEPFLOW-CUBES-20261011',
             'PER_NIGHT_MIN': 'IMPLEMENTED-Q11-PER-NIGHT-LEAVE-ONE-OUT-RULE-AND-COUNTS',
             'JUPITER_MASK_RADIUS_BEAMS': 'RULING-MOVING-JUPITER-MASK-IN-MASTERS',
             'PRODUCT_CARD': 'IMPLEMENTED-Q10-DEEP-FLUX-SCALE-BAKED-AND-Q1-FRAME-TIME-IN-10MIN-STACKS',
             'INTEGRATION_S': 'IMPLEMENTED-Q10-DEEP-FLUX-SCALE-BAKED-AND-Q1-FRAME-TIME-IN-10MIN-STACKS'}


class masters_snapshot(_Section):
    """masters --tier snapshot (tier D): no master is BUILT (contract S-1.3a). verify-frozen checks the frozen generation-4
    masters (md5 against MANIFEST.csv), the 912 links the snapshot tier registers frames through, and the model-beam PB
    library the tier reads (masters_pbmaps.PB_LIBRARY; RULING-SNAPSHOT-TIER-PB-CORRECTION-MODEL-BEAM-ONLY). freeze is
    plan-only in the window. The registry is written to <out>/snapshot/ (default out paths.PHASE3_ARCHIVE_ROOT, a production
    root: a real run needs --out while paths.PRODUCTION_WRITES is False)."""
    LINK_ROOT = '/lustre/pipeline/exopipe/phase3/snapshot_masters'      # 456 slots x 2 links (freeze:21 LINKS; P3_SNAP_MASTERS)
    FROZEN_ROOT = '/lustre/pipeline/exopipe/phase3/snapshot_masters_gen4'   # freeze:23 FROZEN; stack:74; FREEZE-SNAPSHOT-MASTERS-GEN4
    FROZEN_MANIFEST_SHA1 = '19c6f36f512949d186ba4bccec1e7f41e82ecd6e'  # sha1 of <FROZEN_ROOT>/MANIFEST.csv, 1,824 rows + header (read 2026-10-13)
    N_SLOTS = 456                             # (LST, scope) slots: 24 LSTs x 19 scopes
    N_FROZEN_FILES = 1824                     # I/V Taper R0, I Taper R-0.75, PB_deep per slot (freeze:47-57)
    FROZEN_FILES = ('I_deep_Taper_R0_{b}.fits', 'V_deep_Taper_R0_{b}.fits', 'I_deep_Taper_R075_{b}.fits', 'PB_deep_{b}.fits')   # freeze:47-57
    LINKED_FILES = ('I_deep_Taper_R0_{b}.fits', 'V_deep_Taper_R0_{b}.fits')   # the registration links per slot (freeze:47-48)
    FORBIDDEN_LINK_TARGET = 'Coadd/'          # below paths.PHASE3_ARCHIVE_ROOT: no link may resolve into the deep archive (install:203-207 G-1b)
    PB_MODEL_FILES = ('PB_{sb}_bin{k}.fits', 'PB_{sb}_deep.fits')   # in masters_pbmaps.PB_LIBRARY; k = 0 .. PB_MODEL_BINS-1 (snapsearch:954)
    PB_MODEL_BINS = 6                         # 10-minute bins per LST hour (snapsearch:79 N_TIME_BINS)
    PB_BEAM_MODEL = 'OVRO-LWA_MROsoil_updatedheight.h5'   # PBBEAM card of every library map (model beam only, ruling 1231)
    PB_LIBRARY_VALUE_RANGE = (-0.01, 1.001)   # content test of every library map and window PB; measured 2026-10-13 over the 105 maps:
                                              # 0.0 .. 0.99987, all finite (the bounds of masters_pbmaps.PB_VALUE_RANGE)
    FREEZE_ALLOWED = False                    # the freeze stage writes production (freeze:66-105); plan only in the window
    MD5_WORKERS = 8                           # files hashed at once in verify-frozen (Lustre reads; operational, no product effect)
    # ---- snapshot bright-source PARENT catalogues (M21, exopipe_ops a670970 / 3ff6565; review 2026-10-13) -----------------
    # One recipe for all 456 (LST, scope) pairs: exopipe_ops/snapshot_parentcat_contract.py (pc) in place - the deep search's
    # builder functions (phase3_deep._load, phase3_tenmin._det_header, bdsf.process_image) on the frozen gen-4 R075 masters,
    # then snapshot_parentcat_sanitize + the .gen.json sidecar. Class snapshot_parent_catalog (tests_snapshot.py).
    PARENT_CATALOG_DIR = '/lustre/gh/main/exopipe_ops/snapshot_master_catalogs'   # the installed set the snapshot chain reads
                                              # (3ff6565; also forced_snapshot.MASTER_CATS, curves.MASTER_CATALOG_DIR,
                                              # catalog_snapshot.MASTER_CATALOGS name this root - one name at fold-in)
    PARENT_MASTER_FILE = 'I_deep_Taper_R075_{b}.fits'   # the parent master of every pair, in FROZEN_ROOT (pc:49; R-CAT5)
    PARENT_THRESH_PIX = 7.0                   # sigma (pc:34 THRESH_PIX; R-CAT3)
    PARENT_THRESH_ISL = 4.0                   # sigma (pc:34 THRESH_ISL)
    PARENT_ATROUS = False                     # no atrous module (pc:69 atrous_do)
    PARENT_ZAP_JYB = 1.0e6                    # Jy/beam: phase3_deep._load sets |x| above this and non-finite pixels to 0 before
                                              # the build and the sidecar digest (phase3_deep.py:311-312)
    PARENT_NPROC = 8                          # builder processes (pc:85 --nproc default; operational, no product effect)
    PROTECTED_KEYS = ('LINK_ROOT', 'FROZEN_ROOT', 'FROZEN_MANIFEST_SHA1', 'FREEZE_ALLOWED', 'PB_MODEL_BINS',
                      # mirrors of literals in snapshot_parentcat_contract.py / phase3_deep._load
                      'PARENT_MASTER_FILE', 'PARENT_THRESH_PIX', 'PARENT_THRESH_ISL', 'PARENT_ATROUS', 'PARENT_ZAP_JYB')
    RULED = {'FROZEN_ROOT': 'FREEZE-SNAPSHOT-MASTERS-GEN4',
             'FROZEN_MANIFEST_SHA1': 'FREEZE-SNAPSHOT-MASTERS-GEN4',
             'PB_MODEL_FILES': 'RULING-SNAPSHOT-TIER-PB-CORRECTION-MODEL-BEAM-ONLY',
             'PB_BEAM_MODEL': 'RULING-SNAPSHOT-TIER-PB-CORRECTION-MODEL-BEAM-ONLY',
             'PARENT_THRESH_PIX': 'REBUILT-SNAPSHOT-PARENT-CATALOGUES-ONE-RECIPE-20261013',
             'PARENT_THRESH_ISL': 'REBUILT-SNAPSHOT-PARENT-CATALOGUES-ONE-RECIPE-20261013',
             'PARENT_ATROUS': 'REBUILT-SNAPSHOT-PARENT-CATALOGUES-ONE-RECIPE-20261013',
             'PARENT_MASTER_FILE': 'REBUILT-SNAPSHOT-PARENT-CATALOGUES-ONE-RECIPE-20261013'}
