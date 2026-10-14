"""Config draft of `exopipe3 search --tier sublst|snapshot` (unit search-tiers; refactor window: config.py loads it until
the refactor chat folds it in at `# @@STEP-SECTIONS@@`).

Spec: /lustre/gh/main/exopipe_ops/refactor_20261012/plan/specs/search.md sections 3, 5.2-5.4, 7.3 (sub-LST and snapshot
parts). Two sections, one per tier, named <step>_<tier>: `search_sublst` (the sub-LST search pass, its engine environment,
its lease and its fit sidecar) and `search_snapshot` (the snapshot per-frame search, group search, completeness, ship).
The deep tier's sections (search, search_inputs, search_detect, search_tables, search_fpgate, search_fit, search_run) are the
step owner's (_cfgdraft_search-deep.py) and are NOT read by the tier modules: where the spec puts a sub-LST value in a
per-tier dict of a deep section (search_inputs.SUBLST_TREE, search_run.FAST_MIN_FREE_GB['sublst'], ...), the value lives
HERE once, and the refactor chat merges the two at fold-in (listed in plan/impl/search-tiers.md).

Production sources cited (heads read 2026-10-13; review 2026-10-13 re-read them): exopipe_ops a40b088 (sublst_search_pass.sh
= SP, sublst_run_lst_nocat.sh = NOCAT, snapshot_fleet_node_v9.sh = v9, sublst_fit_positions.py = SFP); clv15 a715a1b
(phase3_snapshot_search.py = SS - its night rule is the deep daytime gate since a715a1b and is NOT mirrored here: the engine
runs in place; phase3_snapshot_group_search.py = GS); branch lib/search/deep_search.py = DS (the moved phase3_deep.py, same
line numbers).
Ruling IDs are fix_ids of clv15/fix_ledger.csv.

Conventions (binding decisions of the refactor chat, _CROSSCHECK.md top, core 2337938):
- PROTECTED_KEYS (honoured by config.py: no --set/--config can change them) holds (a) the keys whose value is HARD-CODED in
  an in-place engine (no argument, no environment variable reaches it: an override would make the config untrue;
  tests/test_search-tiers.py checks each against the engine's constant) and (b) pinned inputs and their md5s.
- RULED = {'KEY': '<fix_id of clv15/fix_ledger.csv>'} (every ID checked against the ledger by the tests).
- Shared roots are CFG.paths.* (SUBLST_ROOT, SNAPSHOT_FLEET_OUT, PHASE3_ARCHIVE_ROOT, PHASE1_ROOT, CLV15_ROOT): the keys here
  are sub-paths under them, never a second copy of a root. The clv15 engine directory is core.env.legacy_root('clv15').
- Interpreters are SITE PYTHON keys from the fixed vocabulary default / flow / py36 / casa / tess; the child runs through
  core.legacy.run, which refuses with the missing key's name.
"""
from exopipe.phase3.config import _Section

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
    V110_TIMEOUT_S = 43200                  # s; P3_V110_TIMEOUT; DS:615 (INCIDENT-V110-STAGE-4H-TIMEOUT)
    TABLE_ELEV = True                       # P3_TABLE_ELEV=1 (parent_curve_table.py:162 default); record-only here
    TABLE_ELEV_BANDS = '25-35,35-50,50-90'  # deg; P3_TABLE_ELEV_BANDS (parent_curve_table.py:117 default)
    TABLE_XM_EXEMPT = True                  # P3_TABLE_XM_EXEMPT=1 (parent_curve_table.py:90 default)
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
             'REFINE_X2': 'R-SUBLST-19H-CARD-SECTION-9-ALL-YES'}
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
