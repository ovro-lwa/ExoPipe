"""Config draft of `exopipe3 search --tier deep` (unit search-deep; REFACTOR WINDOW ONLY - config.py loads every _cfgdraft_*.py
until the refactor chat folds them into config.py at `# @@STEP-SECTIONS@@`, then this file is deleted).

Source: plan/specs/search.md section 3 (values = production defaults at exopipe main 03ca797, clv15 9b6525d, exopipe_ops 6b8a23a;
every comment names the unit, the ruling and the production file:line), adapted to the refactor chat's binding decisions of
2026-10-13 ~04:30 UTC (plan/specs/_CROSSCHECK.md top):
- the sections here hold the DEEP tier only. The spec's per-tier dicts are split: the sub-LST and snapshot values live ONCE in
  the search-tiers draft (sections search_sublst, search_snapshot); a deep value is a plain scalar here. The section names are
  <step> and <step>_<stage> (DESIGN 11.4) and read only by lib/search/deep_step.py and lib/search/tests.py.
- shared roots come from config.paths (COADD_SB_ROOT, ARCHIVE_RUN, PHASE3_ARCHIVE_ROOT, PHASE2_ROOT, OPS_ROOT, ...), never a key
  here; ops state and scripts are named RELATIVE to paths.OPS_ROOT / core.env.legacy_root('exopipe_ops').
- PROTECTED_KEYS (config.py refuses any override): pinned inputs (tables, null tables, md5s, roots.csv, master generation) and
  ENGINE LITERALS - values hard-coded inside the in-place engine (deep_search.py and the modules it imports) that no environment
  variable reaches. An override of an engine literal could not change the result, so it is refused until the engine is absorbed;
  lib/search/deep_step.engine_literals() checks at every start that each such config value equals the engine's own value and
  refuses the run when one differs (the config would be untrue).
- COMPARISON keys (archaeology / battery escapes) are ordinary keys; deep_step refuses a deviating value into a production --out.
- Interpreter keys are site PYTHON keys from the fixed vocabulary (default, flow, py36, casa, tess).
"""
from exopipe.phase3.config import _Section


class search(_Section):
    """exopipe3 search --tier deep: detection + gates (lib/search/deep_search.run_deep_cell called in place, one child process per
    unit) and the fitted-position sidecar. Sub-LST: section search_sublst; snapshot: search_snapshot (search-tiers draft)."""
    STAGES = ('preflight', 'quarantine', 'detect', 'fit', 'verify', 'publish')   # search.md 1.4 (deep); --stages/--from/--until
    CADENCE = 'deep'                                        # run_deep_cell cadence (deep_search.py:889-897)
    RUN_LABEL = 'P3S5'                                      # Run_<label> (p3s5_search_fleet.sh:20); --run overrides for one run
    STOKES_ORDER = ('V', 'I')                               # V first per cell: the V search computes the flows the I search reuses
                                                            # (p3s5_search_fleet.sh:254-256; p3s5_launch.sh:24)
    ENGINE_PYTHON = 'default'                               # site PYTHON key of the engine child (py38_orca_nkosogor; p3s5_search_fleet.sh:25)
    REFUSAL_MARKERS = ('REFUSING', 'REFUSED')               # a RuntimeError whose text holds one = gate refusal (exit 2); 28 of the 29 raise
                                                            # sites of deep_search.py use the words; :401 (flow batch rc) does not -> error
    UNACCOUNTED_EPOCHS = 'record'                           # 'record' | 'refuse': admitted epochs with neither an image row nor a refusal row
                                                            # (flow rc 2 drop :1086 = defect R1, swallowed residual task); 'refuse' = B-30 (a), Gregg
    RULED = {'UNACCOUNTED_EPOCHS': 'FIX-ZERO-PRODUCT-RUN-NOW-REFUSES-INSTEAD-OF-REPORTING-COMPLETION'}
    PROTECTED_KEYS = ('CADENCE',)


class search_inputs(_Section):
    """Where the deep search reads its inputs (no renames in the window, DESIGN 11.1). The cell root is paths.COADD_SB_ROOT;
    roots.csv is the fleet's cell -> root table and must agree with it."""
    ROOTS_CSV = 's2search_state/roots.csv'                  # under paths.OPS_ROOT; cell,,root (456 rows, all = paths.COADD_SB_ROOT;
                                                            # p3s5_search_fleet.sh:35,164)
    MASTER_GEN = 'ivw-mean-v3'                              # P3_MASTER_GEN: COADDOP required of the I and V masters (deep_search.py:968;
                                                            # p3s5_search_fleet.sh:198; R-TILT-REMOVED-NEXTGEN)
    ALLOW_OLD_MASTERS = False                               # COMPARISON; P3_ALLOW_OLD_MASTERS=1 (deep_search.py:971, 984, 1001)
    ARCHIVE_WRITE = True                                    # P3_ARCHIVE_WRITE (deep_search.py:855); the archive root is paths.PHASE3_ARCHIVE_ROOT
                                                            # in production and <out>/archive_mirror while paths.PRODUCTION_WRITES is False;
                                                            # the generation is paths.ARCHIVE_RUN (P3_ARCHIVE_RUN, p3s5_search_fleet.sh:198)
    ARCHIVE_MIRROR_SUBDIR = 'archive_mirror'                # window: <out>/<this> receives the archive copies (spec 10, K3)
    RULED = {'MASTER_GEN': 'R-TILT-REMOVED-NEXTGEN'}
    PROTECTED_KEYS = ('ROOTS_CSV', 'MASTER_GEN')


class search_detect(_Section):
    """The deep detection chain (run_deep_cell and the modules it calls). ENV keys are passed to the engine child as environment
    variables (spec 7.3, lib/search/deep_step.engine_env); the keys in PROTECTED_KEYS are ENGINE LITERALS (deep_step checks them
    against the engine at every start)."""
    # ---- environment of the engine child (overridable) --------------------------------------------------------------------
    ELEVATION_FLOOR_DEG = 25.0                              # deg; P3_DEEP_MIN_ELEVATION: V and I detection masks (deep_search.py:921,927;
                                                            # R-DEEP-SERVE-ELEVGATE-FLOOR25-INSEARCH-ELEV)
    MIN_EPOCHS = 1                                          # epochs; P3_DEEP_MIN_EPOCHS (deep_search.py:303); fewer = P3-POLICY-SKIP = no-input
                                                            # (R-GREGG-20261011-DEEP-MIN-EPOCHS-1 in IMPL-REFERENCE-NIGHT-MUST-BE-MASTER-NIGHT-AND-14-REBUILD)
    SIGMA_STACK_MIN_EPOCHS = 3                              # epochs; P3_MIN_EPOCHS: at deep it is read only by _sigma_stack (deep_search.py:296,551)
    SIZE_FLOOR_BEAMS = 1.5                                  # (Maj - 1 sigma)/epoch BMAJ above 8 sigma; P3_SIZE_FLOOR_BEAMS (deep_search.py:75;
                                                            # R-SIZE-AXIS-GATE-1.5-RATIFIED)
    AXIS_REMOVE = 1.5                                       # normalised axis ratio; P3_AXIS_REMOVE (deep_search.py:79)
    SANITY_X = 3.0                                          # x cell median, I (qa3) and V (frame rms); P3_SANITY_I_X (deep_search.py:77;
                                                            # R-SANITY-3X-AND-ROWLINE-BOTH-STOKES-INDEPENDENT)
    SEARCH_TIER_SIGMA = 5.0                                 # x 1.4826 MAD above the median, per Stokes; P3_SEARCH_TIER_SIGMA (deep_search.py:455)
    BEAM_GATE_RATIO = 1.5                                   # x median fitted-beam area; P3_BEAM_GATE_RATIO (deep_search.py:1353; R-BEAM-REFUSALS-STAND-1P5)
    REFINE_X2 = True                                        # False = P3_REFINE_OFF=1 (deep_search.py:454); deep refines (production)
    NPROC_RESID = 6                                         # processes; DEEP_NPROC_RESID (deep_search.py:304)
    FLOW_XLA_MEM_FRACTION = 0.5                             # DEEP_XLA_MEMFRAC (deep_search.py:378)
    FLOW_PYTHON = 'flow'                                    # site PYTHON key -> P3_FLOW_PYTHON (deep_search.py:385; T2B-MERGE-COMPLETE)
    FLOW_LOCK = '/fast/gh/main/phase3/.flow_batch.lock'     # P3_FLOW_LOCK: node-global flock, one GPU batch per node, SHARED with production on
                                                            # purpose (deep_search.py:389). A lock file, not a product (written by flock(1))
    # ---- COMPARISON keys (production value = the default; deep_step refuses a deviation into a production --out) -----------
    BEAM_GATE_ON = True                                     # False = P3_BEAM_GATE_OFF=1 (deep_search.py:1304)
    STRIATION_OFF_BATTERY = False                           # True = P3_STRIATION_OFF=1 (deep_search.py:453)
    SIGMA_STACK_REMOVES = False                             # True = P3_SSTACK_REMOVES=1 (phase3_pipeline.py:1456; CENSUS-SIGMA-STACK-REMOVED-REPEATERS)
    GATE_LEDGER_ONLY = False                                # True = P3_GATE_LEDGER_ONLY=1 (deep_search.py:132)
    STREAMING = True                                        # False = P3_STREAM_RESID=0 (deep_search.py:903-908; validated bit-identical 2026-09-12)
    # ---- ENGINE LITERALS (PROTECTED_KEYS; checked against the engine) ------------------------------------------------------
    I_MASK_FLOOR_DEG = 25.0                                 # deg; at deep the engine sets the I mask = ELEVATION_FLOOR_DEG (deep_search.py:927)
    DETECTION_FLOOR_SIGMA = {'V': 5.0, 'I': 7.0}            # local SNR; V: literal sigma_thresh=5.0 (deep_search.py:1761); I: search_i CONFIG
                                                            # SIGMA_THRESHOLD (search_i_transients.py:71; R-I-FLOOR-7-ADOPTED-FOR-NOW)
    V_BDSF_THRESH_PIX = 5.0                                 # sigma; search_v CONFIG CANDIDATE_SIGMA (search_v_transients_v2.py:58; contract 36)
    I_BDSF_THRESH_PIX = 5.0                                 # sigma; search_i CONFIG BDSF_THRESH_PIX (search_i_transients.py:72)
    I_BDSF_THRESH_ISL = 3.0                                 # sigma; search_i CONFIG BDSF_THRESH_ISL (search_i_transients.py:73)
    LOCAL_SNR_ANNULUS_BEAMS = (2.0, 10.0)                   # beams; search_i CONFIG LOCAL_SNR_INNER/OUTER_BEAMS (search_i_transients.py:81-82)
    LOCAL_SNR_MIN_PIX = 400                                 # px; search_i CONFIG LOCAL_SNR_MIN_PIX (search_i_transients.py:84)
    EXTRACTION_FAILURE = 'refuse'                           # deep: PyBDSF failure refuses that image (deep_search.py:1580 _xrefuse = cadence ==
                                                            # 'deep'; IMPL-DEEP-SEARCH-EXTRACTION-FAILURE-REFUSAL)
    MSTRUCT_K = 5.0                                         # |SNR| < K x max(mstruct, 1) removed (deep_search.py:56 SIGMA_MSTRUCT; contract 37)
    STRIATION_WINDOW_BEAMS = 8.0                            # beams (deep_search.py:57; R-STRIATION-WINDOW-APPROVED-8BEAMS)
    SIGMA_STACK_BUILD_MIN = 5                               # epochs; identification.MIN_EPOCHS_SIGMA (phase3_pipeline.py:112)
    VETO_SNR_GATE = 8.0                                     # sigma (phase3_veto.py:22 SNR_GATE)
    VETO_MASK_JY = 30.0                                     # Jy (phase3_veto.py:23)
    VETO_ONSRC_JY = 2.0                                     # Jy (phase3_veto.py:23)
    VETO_MASK_COEF_DEG = 0.2                                # deg x sqrt(S / 30 Jy) (phase3_veto.py:24)
    VETO_ATEAM_JY = 300.0                                   # Jy (phase3_veto.py:28; Gregg 2026-06-26)
    VETO_ATEAM_RADIUS_DEG = 2.0                             # deg (phase3_veto.py:29)
    VETO_BILOBE = dict(SEP_HI=1.5, LESSER_SIG=5.0, LESSER_RATIO=0.25, NEG_SIG=4.0, PA_RAD_BEAM=15.0, PA_TOL=20.0)
                                                            # beams/sigma/ratio/deg (phase3_veto.py:75-85; Gregg 2026-06-25)
    SIZE_FLAG_BEAMS = 1.3                                   # (deep_search.py:76)
    AXIS_FLAG = 1.3                                         # (deep_search.py:80)
    SIZE_SNR_GATE = 8.0                                     # sigma (deep_search.py:81)
    JUPITER_VETO_DEG = {'Red': 5.0, 'Green': 0.5, 'Blue': 0.5, 'Full': 5.0}   # deg by band class (deep_search.py:58; R-EPHEMERIS-REMOVAL-APPROVED-DOC)
    MOON_VETO_DEG = 0.25                                    # deg, every band (deep_search.py:106; RULINGS-GREGG-20261007-MOON-Q12-Q13)
    ATEAM_VETO_DEG = {'V': 3.0, 'I': 5.0}                   # deg (deep_search.py:66-67; R-ATEAM-NO-PREBLANK-EVER-ANNULUS-STUDY-OFF-CRITICAL-PATH)
    ROWLINE_MIN = 5                                         # finals on one pixel row +/- 1 removed (deep_search.py:78; R-SANITY-3X-AND-ROWLINE)
    MAX_V_PER_EPOCH = 10                                    # flag only, n_v_epoch (deep_search.py:107)
    REPEAT_MATCH_ARCMIN = 10.0                              # arcmin; repeat_group cross-match (phase3_catalog.py:44)
    C50_TABLE = '/data07/gh/phase3/transients/c50_measured.json'   # lwacalim07-local; c50_mjy blank when unreadable (phase3_c50.py:17; B-44)
    # inline literals of deep_search.py (no attribute): verified by a pattern on the engine source (deep_step.engine_literals)
    MSTRUCT_WINDOW_BEAMS = 4.0                              # beams; _master_structure_map(win_beams=4.0) (deep_search.py:186)
    GARBAGE_PIXEL_JY = 1.0e6                                # Jy/beam; |x| above zeroed by _load (deep_search.py:312; B-30)
    EMPIRICAL_BEAM_FALLBACK_PX = 7.0                        # px when fit_empirical_beam is NaN (deep_search.py:994; B-30)
    DURATION_FALLBACK_S = 3600.0                            # s; epoch DURATION when the product has none (deep_search.py:1604-1606; B-30)
    CENTRAL_RMS_HALF_PX = 512                               # px; frame rms box = min(512, h/4, w/4) (deep_search.py:1618, 1732)
    PB_FLOOR = 0.05                                         # pbcorr sibling PB floor (deep_search.py:810; contract 34)
    RULED = {'ELEVATION_FLOOR_DEG': 'R-DEEP-SERVE-ELEVGATE-FLOOR25-INSEARCH-ELEV',
             'MIN_EPOCHS': 'IMPL-REFERENCE-NIGHT-MUST-BE-MASTER-NIGHT-AND-14-REBUILD',
             'SIZE_FLOOR_BEAMS': ('R-SIZE-AXIS-GATE-1.5-RATIFIED', 'R-GATE-TIGHTENED-SIZE-1.5-AXIS-1.5-VALIDATED-ON-KNOWN-TRANSIENTS'),
             'AXIS_REMOVE': ('R-SIZE-AXIS-GATE-1.5-RATIFIED', 'R-GATE-TIGHTENED-SIZE-1.5-AXIS-1.5-VALIDATED-ON-KNOWN-TRANSIENTS'),
             'SANITY_X': 'R-SANITY-3X-AND-ROWLINE-BOTH-STOKES-INDEPENDENT',
             'SEARCH_TIER_SIGMA': 'R-SEARCH-TIER-THRESHOLD-5-SIGMA-MAD-LANDED-RESEARCH-WORKLIST-493',
             'BEAM_GATE_RATIO': 'R-BEAM-REFUSALS-STAND-1P5'}
    PROTECTED_KEYS = ('I_MASK_FLOOR_DEG', 'DETECTION_FLOOR_SIGMA', 'V_BDSF_THRESH_PIX', 'I_BDSF_THRESH_PIX', 'I_BDSF_THRESH_ISL',
                      'LOCAL_SNR_ANNULUS_BEAMS', 'LOCAL_SNR_MIN_PIX', 'EXTRACTION_FAILURE', 'MSTRUCT_K', 'STRIATION_WINDOW_BEAMS',
                      'SIGMA_STACK_BUILD_MIN', 'VETO_SNR_GATE', 'VETO_MASK_JY', 'VETO_ONSRC_JY', 'VETO_MASK_COEF_DEG', 'VETO_ATEAM_JY',
                      'VETO_ATEAM_RADIUS_DEG', 'VETO_BILOBE', 'SIZE_FLAG_BEAMS', 'AXIS_FLAG', 'SIZE_SNR_GATE', 'JUPITER_VETO_DEG',
                      'MOON_VETO_DEG', 'ATEAM_VETO_DEG', 'ROWLINE_MIN', 'MAX_V_PER_EPOCH', 'REPEAT_MATCH_ARCMIN', 'C50_TABLE',
                      'MSTRUCT_WINDOW_BEAMS', 'GARBAGE_PIXEL_JY', 'EMPIRICAL_BEAM_FALLBACK_PX', 'DURATION_FALLBACK_S',
                      'CENTRAL_RMS_HALF_PX', 'PB_FLOOR')


class search_tables(_Section):
    """The per-Stokes, per-band bright-parent table R(S) inside the deep search (lib/search/parent_curve_table.py)."""
    PARENT_TABLES_MD5 = '74197af84d25472a9f4f1c400276b08c'  # md5 of the table file THE ENGINE reads (parent_curve_table.PATH = <package root>/
                                                            # exopipe/common/parent_curve_tables.json; production main's copy has the same md5,
                                                            # read 2026-10-13, last commit 2a5f7d5); gate G2 refuses another file
    TABLE_ELEV = True                                       # P3_TABLE_ELEV (parent_curve_table.py:162; p3s5_search_fleet.sh:44)
    ELEV_BANDS_DEG = ((25.0, 35.0), (35.0, 50.0), (50.0, 90.0))   # P3_TABLE_ELEV_BANDS (parent_curve_table.py:117)
    XM_EXEMPT_ON = True                                     # P3_TABLE_XM_EXEMPT (parent_curve_table.py:90)
    EXTRA_TABLES = None                                     # P3_TABLE_EXTRA (parent_curve_table.py:20-24); unset in every deep search
    PULSAR_CSV = '/lustre/gh/main/clv15/pulsars_atnf.csv'   # read by parent_curve_table (:312-321); a MISSING file silently disables the
                                                            # pulsar exemption today (B-58: kept, recorded); a DIFFERENT file refuses (G2)
    PULSAR_CSV_MD5 = '0806d9b47ef07f8ecbc3f81fcf0f3f3b'
    # engine literals
    DEEP_V_FAINT_S = 20.0                                   # parent S at or below: deep V bins not applied (parent_curve_table.py:27;
                                                            # R-DEEP-V-FAINT-PARENT-TABLE-ENTRIES-DROPPED)
    XM_EXEMPT_SAMPLES = ()                                  # R-TABLE-EXEMPTION-PULSARS-ONLY (parent_curve_table.py:115)
    XM_EXEMPT_MAX_S = 20.0                                  # parent S (parent_curve_table.py:116)
    XM_SAMPLES_DIR = '/lustre/gh/main/clv15'                # sample CSVs, notes only (parent_curve_table.py:67)
    MID_HOUR_OFFSET_H = 0.5                                 # h; table elevation at LST + 0.5 h (deep_search.py:458-463)
    MID_HOUR_LAT_DEG = 37.2398                              # deg; _mid_hour_elev's own latitude (deep_search.py:461; B-42 keeps it)
    RULED = {'TABLE_ELEV': 'R-DEEP-SERVE-ELEVGATE-FLOOR25-INSEARCH-ELEV'}
    PROTECTED_KEYS = ('PARENT_TABLES_MD5', 'PULSAR_CSV', 'PULSAR_CSV_MD5', 'DEEP_V_FAINT_S', 'XM_EXEMPT_SAMPLES', 'XM_EXEMPT_MAX_S',
                      'XM_SAMPLES_DIR', 'MID_HOUR_OFFSET_H', 'MID_HOUR_LAT_DEG')


class search_fpgate(_Section):
    """The frozen v110 Stokes V false-positive gate, one subprocess per cell, run by the engine itself (deep_search.py:577-631).
    FROZEN numerics (FP-GATE-v110-FINAL-conservative-config-RULED): the constants inside fp_gate_v110/v103/v101 and
    artifact_template_v2-4 are not config."""
    ON = True                                               # COMPARISON when False: P3_V110_OFF=1 (deep_search.py:2009)
    REMOVES = ('FP',)                                       # ('FP',) = production; ('FP', 'marginal') = P3_V110_READING=marginal (COMPARISON,
                                                            # deep_search.py:2015-2016; R-V110-MARGINAL-RETAIN-AND-LABEL; R-V110-INCONCLUSIVE-KEEP-LABEL)
    TIMEOUT_S = 43200                                       # s; P3_V110_TIMEOUT (deep_search.py:615; INCIDENT-V110-STAGE-4H-TIMEOUT)
    NULL_TRIALS = '/fast/gh/fp_artifact_investigation/null_trials_v4/null_trials.csv'     # node-local (lwacalim07); missing -> v110 crashes ->
    NULL_TRIALS_MD5 = 'f79e2487b18c5227d6b65be45c327abb'    # the V cell is refused (fp_gate_v110.py:132,166-169); preflight refuses earlier
    PA_NULL = '/fast/gh/fp_artifact_investigation/null_pa_profile_v4/null_pa_profile.csv' # missing -> PA term silently dropped
    PA_NULL_MD5 = '157f7c2eb0c60bfd08497599abc2cfb7'        # (fp_gate_v110.py:199,243-244; B-68)
    PA_NULL_REQUIRED = False                                # B-68 / spec Q6 (Gregg): True refuses a V unit without PA_NULL; False = today
    # engine literals
    WRAPPER = '/lustre/gh/main/exopipe_ops/design/FP_PREDICTION/fp_gate_redo_wrapper.py'   # deep_search.py:578-579 _V110_WRAP
    PYTHON = 'default'                                      # site key whose interpreter = the engine's literal _V110_PY (deep_search.py:577)
    RULED = {'REMOVES': ('R-V110-MARGINAL-RETAIN-AND-LABEL', 'R-V110-INCONCLUSIVE-KEEP-LABEL'),
             'ON': 'FP-GATE-v110-FINAL-conservative-config-RULED', 'PA_NULL_REQUIRED': 'FP-GATE-v110-FINAL-conservative-config-RULED'}
    PROTECTED_KEYS = ('NULL_TRIALS', 'NULL_TRIALS_MD5', 'PA_NULL', 'PA_NULL_MD5', 'WRAPPER', 'PYTHON')


class search_fit(_Section):
    """In-search fitted positions (R-GREGG-20261012-FITTED-DETECTION-POSITIONS-REQUIRED): core/fitting.detection_position on the
    SEARCHED frame of every catalogue row, written as the sidecar fit_positions.csv (deep_fit_positions.py, computed in search)."""
    ON = True                                               # the fit stage is in search.STAGES; False skips it (recorded)
    LST_LON_DEG = -118.2817                                 # deg; LST at the frame's MJD-MID (deep_fit_positions.py:37,62 OVRO_LON)
    ON_CENTRE_PX = 0.01                                     # px (deep_fit_positions.py:147)
    ON_CENTRE_MAX_FRAC = 0.01                               # of fitted rows (deep_fit_positions.py:149)
    APPLY_AT = 'catalog-fitpos'                             # where ra/dec become the fitted position. 'catalog-fitpos' (window): catalog's fitpos
                                                            # stage refits after conformance as today (p3s3_pass.sh:48-52) and parity compares it
                                                            # with this sidecar; 'catalog-apply' (cutover): fitpos applies this sidecar; 'search' =
                                                            # B-31 (b), gates read the fit - Gregg. The deep search never rewrites master_catalog.csv
    NPROC = 16                                              # processes over frames (deep_fit_positions.py:82; FITPOS_NPROC p3s3_pass.sh:49)
    RULED = {'ON': 'R-GREGG-20261012-FITTED-DETECTION-POSITIONS-REQUIRED',
             'APPLY_AT': 'R-GREGG-20261012-FITTED-DETECTION-POSITIONS-REQUIRED'}


class search_run(_Section):
    """Roots, leases and hygiene of deep search units (p3s5_search_fleet.sh / p3s4_incremental_fleet.sh logic the step ports)."""
    FAST_MIN_FREE_GB = 60                                   # GB free on the work filesystem before a unit starts; a node condition: error
                                                            # (retried by the runner), never a refusal (p3s5_search_fleet.sh:52-56; INCIDENT-FAST-FULL)
    RETRIES = 6                                             # runner error retries (p3s5_search_fleet.sh:276); refusals are terminal
    RETRY_BACKOFF_S = 120                                   # s x attempt (p3s5_search_fleet.sh:289)
    WORK_SUBDIR = 'search_deep'                             # window: <--work or site SCRATCH_ROOT>/<this>/{attempts,flowroot,tmp}
    PRODUCTION_WORK_ROOT = '/fast/gh/main/phase3/s2work_p3s5'           # P3_FLOW_ROOT (p3s5_search_fleet.sh:32); used only when
    PRODUCTION_ATTEMPT_ROOT = '/fast/gh/main/phase3/p3s5_staging/attempts'   # paths.PRODUCTION_WRITES is True (p3s5_search_fleet.sh:30)
    COLLECT_ROOT = '/lustre/gh/main/phase3_collect/p3s5_collect_{host}'   # the step's production --out (p3s5_search_fleet.sh:33)
    REGISTER = 'p3s5_state/published_{stokes}.csv'          # under paths.OPS_ROOT: cell,host,attempt (p3s5_search_fleet.sh:238); read only in the
                                                            # window (per-hour mode finds the attempt an epoch is appended to)
    COLLECT_ROOTS = ('/lustre/gh/main/phase3_collect', '/data07/gh/phase3/transients')   # published-attempt resolution, newest first
                                                            # (p3s3_sources.py:26-41)
    COLLECT_LABEL = 'p3s5'                                  # <root>/<label>_collect_<node>/attempts/<cell>_<S>_<att> (p3s3_sources.py:17-18,38)
    QUARANTINE_ROUND = '20261004'                           # P3S5_QROUND (p3s5_launch.sh:22; p3s5_quarantine_frames.py:17)
    QUARANTINE_EPOCH = 'launch-file'                        # 'launch-file' = LAUNCH_EPOCH_FILE (p3s5_launch.sh:16, today); 'job-created' = this
                                                            # run's start (fixes the reuse half of defect D2 - deep chat's call, spec Q7)
    LAUNCH_EPOCH_FILE = 'p3s5_state/launch_epoch'           # under paths.OPS_ROOT: UTC epoch seconds, written once (p3s5_launch.sh:16)
    QUARANTINE_SCRIPT = 'p3s5_quarantine_frames.py'         # in legacy_root('exopipe_ops'); owned by archive (decision 6, group 10)
    CACHE_CURRENT_SCRIPT = 'p3s3_cache_current.py'          # in legacy_root('exopipe_ops'): flow-cache currency (p3s5_search_fleet.sh:96,109;
                                                            # FIX-P3S3-CACHE-CURRENT-MD5-MATCHES-SEARCH-NAN-ZEROING)
    FLOW_PARTIAL_SLACK = 2                                  # epochs; a cache with fewer than (epochs - 2) flows and no generation sidecar is
                                                            # partial (p3s5_search_fleet.sh:128,146-158)
    HYGIENE = 'both-stokes-done'                            # drop the cell's flow work dir and staged attempts once both Stokes of the cell
                                                            # are published in this run (p3s5_search_fleet.sh:297-299); 'never' keeps them
    # Production trees NOT in paths.PRODUCTION_ROOTS (blocker K3 / crosscheck P8): while paths.PRODUCTION_WRITES is False the deep path
    # refuses an --out or --work under any of them (gate G3); cli._guard_out covers only paths.PRODUCTION_ROOTS.
    PRODUCTION_TREES = ('/lustre/gh/main/phase3_collect', '/lustre/gh/main/exopipe_ops/p3s5_state',
                        '/lustre/gh/main/exopipe_ops/p3s4_state', '/lustre/gh/main/exopipe_ops/snapshot_fleet_out',
                        '/lustre/gh/main/exopipe_ops/snapshot_batch', '/lustre/gh/main/exopipe_ops/sublst_search_state_A',
                        '/lustre/gh/main/exopipe_ops/sublst_search_state_B', '/fast/gh/main/phase3', '/data07/gh/phase3',
                        '/lustre/gh/main/phase3', '/lustre/pipeline/exopipe')
    PROTECTED_KEYS = ('PRODUCTION_TREES', 'REGISTER', 'LAUNCH_EPOCH_FILE', 'QUARANTINE_SCRIPT', 'CACHE_CURRENT_SCRIPT')
