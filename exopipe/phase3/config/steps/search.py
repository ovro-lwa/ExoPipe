"""Configuration of `exopipe3 search` - every tunable of the command, one section per stage or tier.

Units folded here: search-deep, search-tiers. A key in its section's PROTECTED_KEYS is locked
(a mirror of in-place code, a pinned input or a ruling-fixed value: it changes only by a commit); RULED names the
ledger ID of the ruling behind a key. Per run: --set SECTION.KEY=VALUE or --config FILE; `exopipe3 config --step search` lists the sections.
"""
from exopipe.phase3.config import _Section


# =====================================================================================================================
# unit search-deep (was exopipe/phase3/_cfgdraft_search-deep.py)
# Configuration of `exopipe3 search --tier deep` (unit search-deep).
#
# Source: plan/specs/search.md section 3 (values = production defaults at exopipe main 03ca797, clv15 9b6525d, exopipe_ops 6b8a23a;
# every comment names the unit, the ruling and the production file:line), adapted to the refactor chat's binding decisions of
# 2026-10-13 ~04:30 UTC (plan/specs/_CROSSCHECK.md top):
# - the sections here hold the DEEP tier only. The spec's per-tier dicts are split: the sub-LST and snapshot values live ONCE in
#   the search-tiers draft (sections search_sublst, search_snapshot); a deep value is a plain scalar here. The section names are
#   <step> and <step>_<stage> (DESIGN 11.4) and read only by lib/search/deep_step.py and lib/search/tests.py.
# - shared roots come from config.paths (COADD_SB_ROOT, ARCHIVE_RUN, PHASE3_ARCHIVE_ROOT, PHASE2_ROOT, OPS_ROOT, ...), never a key
#   here; ops state and scripts are named RELATIVE to paths.OPS_ROOT / core.env.legacy_root('exopipe_ops').
# - PROTECTED_KEYS (the config machinery refuses any override): pinned inputs (tables, null tables, md5s, roots.csv, master generation) and
#   ENGINE LITERALS - values hard-coded inside the in-place engine (deep_search.py and the modules it imports) that no environment
#   variable reaches. An override of an engine literal could not change the result, so it is refused until the engine is absorbed;
#   lib/search/deep_step.engine_literals() checks at every start that each such config value equals the engine's own value and
#   refuses the run when one differs (the config would be untrue).
# - COMPARISON keys (archaeology / battery escapes) are ordinary keys; deep_step refuses a deviating value into a production --out.
# - Interpreter keys are site PYTHON keys from the fixed vocabulary (default, flow, py36, casa, tess).
# =====================================================================================================================

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
    # the cell-median gates' minimum counts (Gregg 2026-10-14: production's 5 each, kept; a skip is counted -
    # fallbacks.SEARCH_GATE_SKIP_FALLBACK - never refused): P3_SANITY_MIN_NIGHTS, P3_SEARCH_TIER_MIN, P3_BEAM_GATE_MIN_NIGHTS
    SANITY_MIN_NIGHTS = 5                                   # nights with a measured noise (I qa3 / V frame rms) for the sanity gates
    SEARCH_TIER_MIN = 5                                     # nights with a stage-1 rms, per Stokes, for the search-tier gate
    BEAM_GATE_MIN_NIGHTS = 5                                # nights with a beam area for the beam gate (Gregg 2026-09-11 floor waived)
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
    EPHEMERIS_TIME = 'frame-mid'                            # B-30 (a) 'ephemerides at frame mid-time': Jupiter / Moon vetoes and the
                                                            # candidates' body labels at the frame's own mid-time; 'date-obs' = production
                                                            # (its DATE-OBS, Card 7a) - the engine child reads EXOPIPE3_EPHEMERIS_TIME
                                                            # (deep_step.engine_env; the 10-minute tier: search_sublst.EPHEMERIS_TIME)
    RULED = {'EPHEMERIS_TIME': 'R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32',
             'ELEVATION_FLOOR_DEG': 'R-DEEP-SERVE-ELEVGATE-FLOOR25-INSEARCH-ELEV',
             'MIN_EPOCHS': 'IMPL-REFERENCE-NIGHT-MUST-BE-MASTER-NIGHT-AND-14-REBUILD',
             'SIZE_FLOOR_BEAMS': ('R-SIZE-AXIS-GATE-1.5-RATIFIED', 'R-GATE-TIGHTENED-SIZE-1.5-AXIS-1.5-VALIDATED-ON-KNOWN-TRANSIENTS'),
             'AXIS_REMOVE': ('R-SIZE-AXIS-GATE-1.5-RATIFIED', 'R-GATE-TIGHTENED-SIZE-1.5-AXIS-1.5-VALIDATED-ON-KNOWN-TRANSIENTS'),
             'SANITY_X': 'R-SANITY-3X-AND-ROWLINE-BOTH-STOKES-INDEPENDENT',
             'SEARCH_TIER_SIGMA': 'R-SEARCH-TIER-THRESHOLD-5-SIGMA-MAD-LANDED-RESEARCH-WORKLIST-493',
             'BEAM_GATE_RATIO': 'R-BEAM-REFUSALS-STAND-1P5',
             'SANITY_MIN_NIGHTS': 'R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS', 'SEARCH_TIER_MIN': 'R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS', 'BEAM_GATE_MIN_NIGHTS': 'R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS'}
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
    ENGINE_TMP_ROOT = '/tmp'                                # the engine child's TMPDIR root (a short e3_<cell>_<S>_* per unit): production
                                                            # sets no TMPDIR; PyBDSF's multiprocessing sockets need a path <= 107 bytes
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


# =====================================================================================================================
# unit search-tiers (was exopipe/phase3/_cfgdraft_search-tiers.py)
# Configuration of `exopipe3 search --tier sublst|snapshot` (unit search-tiers).
#
# Spec: /lustre/gh/main/exopipe_ops/refactor_20261012/plan/specs/search.md sections 3, 5.2-5.4, 7.3 (sub-LST and snapshot
# parts). Two sections, one per tier, named <step>_<tier>: `search_sublst` (the sub-LST search pass, its engine environment,
# its lease and its fit sidecar) and `search_snapshot` (the snapshot per-frame search, group search, completeness, ship).
# The deep tier's sections (search, search_inputs, search_detect, search_tables, search_fpgate, search_fit, search_run) are the
# step owner's (config/steps/search.py) and are NOT read by the tier modules: where the spec puts a sub-LST value in a
# per-tier dict of a deep section (search_inputs.SUBLST_TREE, search_run.FAST_MIN_FREE_GB['sublst'], ...), the value lives
# HERE once, and the refactor chat merges the two at fold-in (listed in plan/impl/search-tiers.md).
#
# Production sources cited (heads read 2026-10-13; review 2026-10-13 re-read them): exopipe_ops a40b088 (sublst_search_pass.sh
# = SP, sublst_run_lst_nocat.sh = NOCAT, snapshot_fleet_node_v9.sh = v9, sublst_fit_positions.py = SFP); clv15 a715a1b
# (phase3_snapshot_search.py = SS - its night rule is the deep daytime gate since a715a1b and is NOT mirrored here: the engine
# runs in place; phase3_snapshot_group_search.py = GS); branch lib/search/deep_search.py = DS (the moved phase3_deep.py, same
# line numbers).
# Ruling IDs are fix_ids of clv15/fix_ledger.csv.
#
# Conventions (binding decisions of the refactor chat, _CROSSCHECK.md top, core 2337938):
# - PROTECTED_KEYS (honoured by config/__init__.py: no --set/--config can change them) holds (a) the keys whose value is HARD-CODED in
#   an in-place engine (no argument, no environment variable reaches it: an override would make the config untrue;
#   tests/test_search-tiers.py checks each against the engine's constant) and (b) pinned inputs and their md5s.
# - RULED = {'KEY': '<fix_id of clv15/fix_ledger.csv>'} (every ID checked against the ledger by the tests).
# - Shared roots are CFG.paths.* (SUBLST_ROOT, SNAPSHOT_FLEET_OUT, PHASE3_ARCHIVE_ROOT, PHASE1_ROOT, CLV15_ROOT): the keys here
#   are sub-paths under them, never a second copy of a root. The clv15 engine directory is core.env.legacy_root('clv15').
# - Interpreters are SITE PYTHON keys from the fixed vocabulary default / flow / py36 / casa / tess; the child runs through
#   core.legacy.run, which refuses with the missing key's name.
# =====================================================================================================================

_SUBBANDS = ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz',
             '73MHz', '78MHz', '82MHz')


class search_sublst(_Section):
    """exopipe3 search --tier sublst: the 10-minute search (deep_search.run_deep_cell(cadence='10min') in place, one child
    process per (cell, Stokes) unit) on the sub-LST residual tree of one arm (port of exopipe_ops/sublst_search_pass.sh and
    the search half of sublst_run_lst_nocat.sh)."""
    STAGES = ('preflight', 'detect', 'fit', 'verify', 'publish')   # search.md 1.4 (sublst row): no quarantine (archive off)
    CADENCE = '10min'                       # run_deep_cell cadence; SP:47; DS:889-897
    ARM = 'B'                               # residual tree searched; set B = the production search set since 2026-09-26
                                            # (NOCAT:275 'sublst_search_pass.sh B'); A and BI selectable with --sublst-arm
    TREE_SUBDIRS = {'A': 'tenmin_root',     # P3_TENMIN_ROOT = paths.SUBLST_ROOT/<this>; SP:19
                    'B': 'tenmin_root_x2',  # SP:19
                    'BI': 'tenmin_root_x2_inj'}   # SP:20 (set B + injections, Gregg 2026-09-26)
    COADD_SUBDIR = 'coadd_{lst}'            # P3_COADD_ROOT = paths.SUBLST_ROOT/<this>, the md5-verified live copy; SP:29; B-20 (b)
    RUN_LABEL = 'SLST{hh}{arm}'             # Run_<label>; SP:23 (LABEL=SLST${LST%h}$ARM)
    PUBLISH_SUBDIR = 'search_{arm}'         # under the product root (default paths.SUBLST_ROOT; SP:24 OUT=.../sublst/search_$ARM);
                                            # one <cell>_<S>/ per unit
    PUBLISH_ITEMS = ('Run_{label}', 'transients', '{cell}.log', 'flow/{cell}_10min/catdir', 'flow/{cell}_10min/v110_out')
                                            # copied from the attempt, rsync -a semantics; SP:53-55 (the last two when present)
    WORK_SUBDIR = 'sublst_search/{arm}'     # attempt dirs <scratch>/<this>/<cell>_<S>_<UTC>/ (+ /flow); SP:25, :42-43
    ENGINE_TMP_ROOT = '/tmp'                # the engine child's TMPDIR root, as search_run.ENGINE_TMP_ROOT (SP sets no TMPDIR)
    KEEP_ATTEMPT_ON_SUCCESS = False         # SP:57 'rm -rf $att' after a published success; a failed attempt is always kept (SP:52)
    BANDS = _SUBBANDS + ('Red', 'Green', 'Blue')   # 15 subbands + the three wideband groups; no Full in this tier (B-82; NOCAT:271)
    SEARCH_ORDER = ('Blue', 'Green', 'Red', '82MHz', '78MHz', '73MHz', '69MHz', '64MHz', '59MHz', '55MHz', '50MHz', '46MHz',
                    '41MHz', '36MHz', '32MHz', '27MHz', '23MHz', '18MHz')   # unit order of a pass; NOCAT:271
    STOKES_ORDER = ('I', 'V')               # NOCAT:272 ('${LST}_$sb:I ${LST}_$sb:V'); I and V never share a work dir (SP:10)
    ENGINE_PYTHON = 'default'               # site PYTHON key; 'default' = /opt/devel/pipeline/envs/py38_orca_nkosogor (SP:23)
    REFUSAL_MARKERS = ('REFUSING', 'REFUSED')   # a RuntimeError whose text holds one = gate refusal (exit 2); search.md 5.1
                                            # (28 of the 29 raise sites of DS; :401 is an error)
    # ---- engine environment (search.md 7.3): every inherited P3_* variable is cleared, then exactly these are set ----------
    MASTER_GEN = 'ivw-mean-v3'              # P3_MASTER_GEN; SP:30; DS:968 (COADDOP required of the I and V masters)
    TABLE_PREFIX = 'sublst_search'          # P3_TABLE_PREFIX; SP:36: a prefix with no installed table = record-only bright-parent
                                            # table in the search (the sub-LST tables are applied by catalog)
    TABLE_RECORD_ONLY = True                # P3_TABLE_RECORD_ONLY=1; SP:30
    REFINE_X2 = False                       # False = P3_REFINE_OFF=1 (no per-window x2 deep-Master flow); SP:30 (ruled 2026-09-25)
    ARCHIVE_WRITE = False                   # False = P3_ARCHIVE_WRITE=0: a 10-minute search is never written to the archive; SP:30
    XLA_PREALLOCATE = False                 # XLA_PYTHON_CLIENT_PREALLOCATE=false; SP:29
    # values the engine reads from the environment with these defaults; set explicitly so config is the one source
    MIN_ELEVATION_DEG = 25.0                # deg; P3_TENMIN_MIN_ELEVATION (V mask, 10-min branch); DS:915; Gregg 2026-09-11
    MIN_EPOCHS = 3                          # windows; P3_MIN_EPOCHS; DS:296 (B-75); fewer = P3-POLICY-SKIP = unit no-input
    SIZE_FLOOR_BEAMS = 1.5                  # (Maj - 1 sigma)/BMAJ above 8 sigma; P3_SIZE_FLOOR_BEAMS; DS:75
    AXIS_REMOVE = 1.5                       # normalised axis ratio; P3_AXIS_REMOVE; DS:79
    SANITY_I_X = 3.0                        # x cell median qa3; P3_SANITY_I_X; DS:77
    SEARCH_TIER_SIGMA = 5.0                 # x 1.4826 MAD; P3_SEARCH_TIER_SIGMA; DS:455
    BEAM_GATE_RATIO = 1.5                   # x median beam area; P3_BEAM_GATE_RATIO; DS:1353
    SANITY_MIN_NIGHTS = 5                   # the sanity gates' minimum count; P3_SANITY_MIN_NIGHTS (Gregg 2026-10-14: 5 kept, skips counted)
    SEARCH_TIER_MIN = 5                     # FRAMES with a stage-1 rms per Stokes at 10 minutes; P3_SEARCH_TIER_MIN (5 kept, skips counted)
    BEAM_GATE_MIN_NIGHTS = 5                # the beam gate's minimum count; P3_BEAM_GATE_MIN_NIGHTS (5 kept, skips counted)
    V110_TIMEOUT_S = 43200                  # s; P3_V110_TIMEOUT; DS:615 (INCIDENT-V110-STAGE-4H-TIMEOUT)
    TABLE_ELEV = True                       # P3_TABLE_ELEV=1 (parent_curve_table.py:162 default); record-only here
    TABLE_ELEV_BANDS = '25-35,35-50,50-90'  # deg; P3_TABLE_ELEV_BANDS (parent_curve_table.py:117 default)
    TABLE_XM_EXEMPT = True                  # P3_TABLE_XM_EXEMPT=1 (parent_curve_table.py:90 default)
    EPHEMERIS_TIME = 'frame-mid'            # B-30 (a) 'ephemerides at frame mid-time', as search_detect.EPHEMERIS_TIME: Jupiter /
                                            # Moon vetoes and the candidates' body labels at the window frame's own mid-time;
                                            # 'date-obs' = production (EXOPIPE3_EPHEMERIS_TIME; tier_sublst.engine_env)
    # engine constants the content tests read (not settable during the window: no environment variable reaches them)
    I_MASK_FLOOR_DEG = 20.0                 # deg; the Stokes I detection mask keeps search_i's module floor at 10 min (the 10-min
                                            # branch sets only the V module; search_i.py:67; DS:911-917; defect R7, report-only, B-25)
    # ---- orchestration (ported from SP and NOCAT) --------------------------------------------------------------------------
    FAST_MIN_FREE_GB = 200                  # GB free on the attempt filesystem before a unit starts (SEARCH_FAST_MIN_GB; SP:40-41;
                                            # INCIDENT-FAST-FULL-SILENT-SHIPPING-FAILURES). A node condition, not a result: error
                                            # (retryable), never 'done' or 'refused'
    DISK_WAIT_S = 0                         # s to wait for that free space before the error (0 = check once, as SP)
    EPOCHS_MODE = 'incremental'             # --epochs <LST>_<date> (per hour, DESIGN 12.1): 'incremental' = run_deep_cell
                                            # included_dates = that night's windows, appended to a copy of the published attempt
                                            # (DS:1371-1380, Gregg 2026-09-13 "Proceed" for the same function); 'refuse' = spec G4
    # ---- the search slot as a lease (NOCAT:36-45: one 10-minute search pass at a time, mkdir slot, owner host:pid, reclaimed
    #      when the owner is gone; spec 5.2/S4 replaces it by a per-node memory lease) ----------------------------------------
    LEASE = 'node-memory'                   # 'node-memory' = per-node counting lease: at most floor((MemTotal - NODE_MEM_MARGIN_GB)
                                            # / MEM_RESERVE_GB) units on a node (spec search_run.MEM_RESERVE_GB['sublst'], answers
                                            # C4.4: ~100 GB per 10-minute search); 'lst-mutex' = today's slot: units of ONE LST
                                            # share it, another LST waits (NOCAT:40-45); 'none'
    MEM_RESERVE_GB = 100                    # GB per unit (answers C4.4)
    NODE_MEM_MARGIN_GB = 120                # GB of a node's memory never leased (new: production bounded memory by lanes per node,
                                            # NOCAT:32; the same margin as snapshot_cell_prefit.sh:41)
    LEASE_ROOT = ''                         # '' = <runner.JOBS_ROOT>/_leases (in a parity run: <out>/_leases)
    LEASE_POLL_S = 120                      # s between attempts to take the lease; NOCAT:43 (sleep 120)
    LEASE_WAIT_S = 0                        # s before a unit gives up waiting (error, retryable); 0 = wait as long as it takes (NOCAT)
    LEASE_HEARTBEAT_S = 60                  # s; the holder refreshes its lease file this often
    LEASE_STALE_S = 900                     # s; a holder whose heartbeat is older (or, on this host, whose pid is gone) is reclaimed
                                            # (NOCAT:41-42 reclaimed a slot whose owner process was gone, read over ssh)
    # ---- fit stage: the fitted position of every detection on its searched frame, as a SIDECAR (spec 1.4 row 4, K14). The
    #      catalogue keeps the peak position (B-31 default); catalog's fitpos stage stays the operative fit in the window -------
    FIT_FUNCTION = 'core'                   # 'core' = core.fitting.detection_position(tier='sublst'; K = fitting.K_TEMPLATE[sublst,S]);
                                            # 'legacy' = exopipe_ops deep_forced_radio.detection_position in place with the same K
                                            # (spec Q11: the two differ only in the observatory latitude, 37.23977727 vs 37.2398 deg)
    FIT_NPROC = 16                          # processes over frames; SFP:159 (--nproc default)
    FIT_LST_LON_DEG = -118.2817             # deg; LST at MJD-MID; SFP:54
    FIT_BLANKED = ('pos_sys_arcsec', 'pos_err_maj_arcsec', 'pos_err_min_arcsec')   # error model A: the deep systematic never
                                            # enters a 10-minute product; SFP:56 (ruled 2026-10-08/10)
    FIT_STAT_FLOOR = '0.1'                  # arcsec; a fitted statistical error printed '0.0' is written as this and counted; SFP:131-134
    FIT_ON_CENTRE_PX = 0.01                 # px; F4: fitted positions within this of a pixel centre ...; SFP:57
    FIT_ON_CENTRE_MAX_FRAC = 0.01           # ... at most this fraction of the fitted rows; SFP:57
    # ---- pinned inputs of the V search's frozen FP gate (v110 reads them; search.md G2, K6) ----------------------------------
    V110_NULL_TRIALS = '/fast/gh/fp_artifact_investigation/null_trials_v4/null_trials.csv'   # node-local; missing = v110 crashes =
    V110_NULL_TRIALS_MD5 = 'f79e2487b18c5227d6b65be45c327abb'                                 # cell refused (fp_gate_v110.py:132,166-169)
    V110_PA_NULL = '/fast/gh/fp_artifact_investigation/null_pa_profile_v4/null_pa_profile.csv'   # missing = PA term silently
    V110_PA_NULL_MD5 = '157f7c2eb0c60bfd08497599abc2cfb7'                                       # disabled (fp_gate_v110.py:199,243-244)
    PA_NULL_REQUIRED = False                # B-68 / Q6 (Gregg): True would refuse a V unit without PA_NULL; today the term is dropped
    RULED = {'MIN_ELEVATION_DEG': 'R-BATTERY-CARD-SECTION-9-APPROVED-10MIN-ELEVATION-25-VHF-LABEL-ONLY-RATIO-2',
             'MIN_EPOCHS': 'R-SEARCH-MIN-EPOCHS-3-FOR-THIS-RUN',
             'TABLE_PREFIX': 'CORRECTION-SUBLST-SEARCH-APPLIED-OLD-10MIN-TABLE-IN-SEARCH',
             'TABLE_RECORD_ONLY': 'CORRECTION-SUBLST-SEARCH-APPLIED-OLD-10MIN-TABLE-IN-SEARCH',
             'SIZE_FLOOR_BEAMS': 'R-SIZE-AXIS-GATE-1.5-RATIFIED', 'AXIS_REMOVE': 'R-SIZE-AXIS-GATE-1.5-RATIFIED',
             'SANITY_I_X': 'R-SANITY-3X-AND-ROWLINE-BOTH-STOKES-INDEPENDENT',
             'SEARCH_TIER_SIGMA': 'R-SEARCH-TIER-THRESHOLD-5-SIGMA-MAD-LANDED-RESEARCH-WORKLIST-493',
             'BEAM_GATE_RATIO': 'R-BEAM-REFUSALS-STAND-1P5',
             'REFINE_X2': 'R-SUBLST-19H-CARD-SECTION-9-ALL-YES',
             'EPHEMERIS_TIME': 'R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32',
             'SANITY_MIN_NIGHTS': 'R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS', 'SEARCH_TIER_MIN': 'R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS', 'BEAM_GATE_MIN_NIGHTS': 'R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS'}
    # engine-hard-coded (CADENCE: the tier IS the 10-minute cadence; I_MASK_FLOOR_DEG: search_i's module constant) and pinned
    # (MASTER_GEN: R-TILT-REMOVED-NEXTGEN; the trees; the v110 null tables and their md5s)
    PROTECTED_KEYS = ('CADENCE', 'I_MASK_FLOOR_DEG', 'MASTER_GEN', 'TREE_SUBDIRS', 'V110_NULL_TRIALS', 'V110_NULL_TRIALS_MD5',
                      'V110_PA_NULL', 'V110_PA_NULL_MD5')


class search_snapshot(_Section):
    """exopipe3 search --tier snapshot: one unit per snapshot cell <LST>_<night> (one LST hour of one night - the per-hour
    unit, DESIGN 12.1): clv15 phase3_snapshot_search.py per subband and phase3_snapshot_group_search.py per Stokes, run in place
    as subprocesses with production's arguments; completeness, night-rule refusal records, ship (port of the search half of
    exopipe_ops/snapshot_fleet_node_v9.sh)."""
    STAGES = ('preflight', 'subbands', 'widebands', 'completeness', 'verify', 'ship')   # search.md 1.4 (snapshot)
    OPTIONAL_STAGES = ('fit', 'tier1min')   # not in the default list; refused here during the window (fit = catalog's
                                            # store/prefit until cutover, K14; tier1min = snapshot_frametime_rerun.sh, not ported)
    SUBBANDS = _SUBBANDS                    # v9:47 (P3_FLEET_SUBBANDS default): every one is searched (unstaged ones fail fast,
                                            # as in v9, and their search_<sb>.log ships); completeness covers the STAGED ones
    GROUPS = ('Red', 'Green', 'Blue')       # wideband groups accounted for (v9:255; GS:64; Full discarded unless on one grid)
    ENGINE_PYTHON = 'py36'                  # site PYTHON key 'py36' = /opt/devel/pipeline/envs/development/bin/python (v9:44);
                                            # engines: core.env.legacy_root('clv15')/phase3_snapshot_{search,group_search}.py (v9:43)
    STAGE_SUBDIR = 'snapshot_stage'         # staged + flowed frames <scratch>/<this>/<cell>/<sb>/ (frames --tier snapshot, the
                                            # same name as frames_stage.STAGE_SUBDIR; v9:87); --snap-stage-root overrides
    WORK_SUBDIR = 'snapshot_out'            # node-local search output <scratch>/<this>/<cell>/ (v9:88), shipped to <out>/<cell>/
    MASTERS = '/lustre/pipeline/exopipe/phase3/snapshot_masters'   # P3_SNAP_MASTERS default (v9:54): <lst>_<sb>/coadd_final/...
    MASTER_FILE = 'I_deep_Taper_R0_{sb}.fits'   # the flow reference of a subband (v9:131, 147, 153)
    MODE = 'median'                         # --mode (v9:65; SS:1476): hour-median sky model
    THRESH_SIGMA = 5.0                      # --thresh, per-image global sigma (v9:65, :237; contract S2.1)
    NTHREADS = 8                            # --nthreads (v9:66, :237)
    NPAR = 4                                # subband searches at a time (v9:35, xargs -P)
    GROUPS_PARALLEL_STOKES = False          # False = V then I serially (v9:235-239); True = speed-up S5 (pure: two processes)
    FLOW_REQUIRED = True                    # P3_SNAP_FLOW=1 (v9:64, :236): searches the _flowed frames, refuses an unflowed subband
    FLOW_MAX_MHZ = 1000.0                   # MHz; P3_SNAP_FLOW_MAX_MHZ (SS:667 default)
    PAUSE_1MIN = False                      # P3_SNAP_PAUSE_1MIN (v9:81-84; pause lifted 2026-10-08 06:47 UTC): no 1-minute tier, and
                                            # the group search skipped (v9:235); --snap-pause-1min sets it
    CUTOUTS_ON = True                       # P3_SNAP_CUTOUTS (SS:1315 default '1'; photometry cutout cubes, a product of `photometry`)
    CUTOUT_SUBDIR = 'photometry'            # P3_SNAP_CUTOUT_ROOT = paths.PHASE3_ARCHIVE_ROOT/<this> (= phase3_snapshot_cutouts.
                                            # ROOT_DEFAULT); while paths.PRODUCTION_WRITES is False: <out>/<this>. Product class
                                            # photometry_cutout (owned by photometry; recorded by this step on write)
    SOLAR_AUDIT = '/lustre/pipeline/exopipe/phase3/QA/solar_contamination_audit.csv'   # P3_SOLAR_AUDIT (SS:419), pinned (G2)
    SOLAR_AUDIT_MD5 = 'f5e4ef1124230351603fba34018a1f4a'   # md5 read 2026-10-13 (spec search.md 3); a different file refuses
    # ---- gates ported from v9 (preflight / completeness) --------------------------------------------------------------------
    MIN_STAGED_FRAMES = 1000                # decompressed staged frames per cell (P3_FLEET_MIN_FRAMES; v9:199-202)
    OUT_MIN_FREE_GB = 50                    # GB free on the node-local output filesystem (new: v9's 800 GB (v9:95) covers the
                                            # staging, which is now the frames step's; the search writes CSVs and models only)
    EXCLUDED_STAGED_SUBBAND = 'accounted'   # a STAGED subband excluded in data_decisions.csv (SS exits 3, no output) counts as an
                                            # outcome (exclusions are absolute), as v9 since exopipe_ops a40b088 (v9:217-235; 07h
                                            # 2025-10-14 failed the guard twice); 'unaccounted' = v9 before it (SEARCH-INCOMPLETE)
    KEEP_FAILED_LOGS = True                 # v9 deletes OUT of an incomplete cell (v9:228, :267); its logs are kept under
                                            # <out>/_failed/<cell>_<UTC>/ first (new; evidence only, never read as a product)
    SHIP_POLICY = 'no-clobber'              # cp -n (v9:336-337); an existing DIFFERENT file refuses the unit (content test
                                            # ship.no_clobber) instead of being kept silently
    SHIP_PATTERNS = ('*.csv', '*.json', '*.txt', '*.log', 'model_*_median.fits')   # v9:336-337
    # ---- engine constants the content tests read; locked (hard-coded in clv15, no argument or environment variable) ----------
    MIN_ELEVATION_DEG = 30.0                # deg; SS:78 (R-SNAP-ELEVATION-FLOOR-30-DEG)
    INTEGRATION_S = 10.03102041             # s; SS:133 (R-FRAME-TIME-FROM-INTEGRATION-ALL-STAGES)
    TIER1MIN_TIME_BASIS = 'integration-midpoint-dateobs-v1'   # runinfo frame_time_1min; SS:134
    MIN_BLOCK_FRAMES = 3                    # admitted frames per 1-minute block; SS:1132
    PAIR_TOL_S = 1.0                        # s; wideband members of one integration (GS:68)
    NIGHT_RULE_STATUS = 'refused-night-rule'   # status of a nightrefused_<tag>.json record (SS:508; the rule itself -
                                            # night_verdict, a715a1b - is the engine's, never mirrored here)
    WIDEBAND_OUTCOMES = ('searched', 'night-refused', 'excluded', 'no-members', 'too-few-frames', 'not-run')   # GS:132-135
    RULED = {'MIN_STAGED_FRAMES': 'INCIDENT-FAST-FULL-SILENT-SHIPPING-FAILURES',
             'FLOW_REQUIRED': 'R-SNAP-FLOW-ALIGNMENT-AND-FLOW-MOVIE-ARE-PIPELINE-PRODUCTS-FLEET-V4',
             'EXCLUDED_STAGED_SUBBAND': 'FIX-FLEET-GUARD-EXCLUDED-SUBBANDS-20261013',
             'PAUSE_1MIN': 'PAUSE-WIDEBAND-10S-AND-1MIN-IN-THE-FLEET'}
    # engine-hard-coded constants and pinned inputs (no override can change them)
    PROTECTED_KEYS = ('MIN_ELEVATION_DEG', 'INTEGRATION_S', 'TIER1MIN_TIME_BASIS', 'MIN_BLOCK_FRAMES', 'PAIR_TOL_S',
                      'NIGHT_RULE_STATUS', 'WIDEBAND_OUTCOMES', 'GROUPS', 'SOLAR_AUDIT', 'SOLAR_AUDIT_MD5', 'MASTERS')
