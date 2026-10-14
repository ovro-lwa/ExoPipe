"""Configuration of `exopipe3 flow` - every tunable of the command, one section per stage or tier.

Units folded here: flow. A key in its section's PROTECTED_KEYS is locked
(a mirror of in-place code, a pinned input or a ruling-fixed value: it changes only by a commit); RULED names the
ledger ID of the ruling behind a key. Per run: --set SECTION.KEY=VALUE or --config FILE; `exopipe3 config --step flow` lists the sections.
"""
from exopipe.phase3.config import _Section


# =====================================================================================================================
# unit flow (was exopipe/phase3/_cfgdraft_flow.py)
# Config draft of the `exopipe3 flow` sections (refactor window; folded into config.py by the refactor chat).
#
# Spec: exopipe_ops/refactor_20261012/plan/specs/flow.md section 3, adapted to the binding decisions at the top of
# plan/specs/_CROSSCHECK.md (2026-10-13 ~04:30 UTC):
# - masters owns the deep native-reference reconstruction and the epoch-gate carry: the spec's section flow_reference is NOT
#   here. The deep reference mode and fixed-table nights, CUTOVER_DATE, MIN_DATES_FOR_COADD, the reference astrometry gate,
#   FLOW_MAX_NAN_FRACTION and SAFE_SCALE_FLOOR are read from masters_build (one home); the stored-flow source cell from
#   masters' own working-cell rule (lib/masters/deep_stagein.working_cell, masters_stagein tables);
# - frames owns the snapshot flow movies: the spec's section flow_movie is NOT here (frames_movies);
# - shared roots come from config.paths (SUBLST_ROOT, SNAPSHOT_FLEET_OUT, SNAPSHOT_FLOW_ROOT, PHASE3_ARCHIVE_ROOT,
#   COADD_SB_ROOT); interpreters are site PYTHON 'default' and 'flow' only.
#
# Every value is today's production value with its unit, the ruling (ledger fix_id) or "no ruling found", and the source
# file:line (exopipe 03ca797, clv15 9b6525d, exopipe_ops 0c1580c). ``RULED`` and ``PROTECTED_KEYS`` are honoured by config.py.
# Values that copy a constant of an in-place module are checked against the live module at every run (the flow drift guard:
# a difference refuses unless the key was overridden explicitly; an override that cannot reach the module - one that runs in
# a child process - refuses).
# =====================================================================================================================

class flow(_Section):
    """exopipe3 flow - optical-flow alignment on the GPU (image_plane_correction Brox flow, JAX, site PYTHON['flow']).
    Deep per-night co-add flows (survey: part of a masters build), snapshot per-frame flows (per hour), sub-LST x2 refine
    (per hour / per LST). Engine parameters are the same in every tier today."""
    DEFAULT_STAGES = {'deep': ('stage-in', 'prep', 'solve', 'check'),
                      'snapshot': ('flow', 'ship'),
                      'sublst': ('refine', 'check')}   # spec flow.md 1.2 without 'reference' (masters stage-in) and 'movie'
                                                       #   (frames movies): _CROSSCHECK decision 6, groups 1 and 4
    # ---- Brox flow parameters (GC-PRE-2 / align_images defaults; no separate ruling found). Production passes none of them
    #      (each module's own default applies); the step passes them to batch_flow_align / phase3_snapshot_flow only when
    #      overridden, and refuses an override for sub-LST (the refine module shells align_images with its own defaults).
    PREPROCESS_WEIGHT = 1.5      # power-law preprocessing weight; batch_flow_align.py:42; phase3_snapshot_flow.py:54; align_images.py:125
    ALPHA = 1.3                  # Brox smoothness; batch_flow_align.py:39; phase3_snapshot_flow.py:55; align_images.py:127
    GAMMA = 150.0                # Brox gradient-constancy weight; batch_flow_align.py:40; phase3_snapshot_flow.py:56; align_images.py:129
    SCALE_FACTOR = 0.7           # pyramid scale factor; batch_flow_align.py:41; phase3_snapshot_flow.py:57; align_images.py:131
    STATS_FOV_FRACTION = 0.4     # radius of the statistics disc, x min(ny, nx) (report only); batch_flow_align.py:107;
                                 #   phase3_snapshot_flow.py:86; align_images.py:87 (used by the content tests' reports)
    JAX_COMPILATION_CACHE_DIR = '~/.cache/jax'   # persistent JIT cache; batch_flow_align.py:31-32; align_images.py:34-35;
                                 #   phase3_snapshot_flow.py sets none: the step exports it for the snapshot children (speed-up S2;
                                 #   the cache changes compile time only, never results)
    # ---- GPU process environment ----
    XLA_PREALLOCATE = 'false'    # XLA_PYTHON_CLIENT_PREALLOCATE; rebuild_cell.sh:26; snapshot_fleet_node_v9.sh:183; snapshot_cube_pass.sh:96;
                                 #   phase3_sublst_refine.py:97 (set inside the module for its aligner child)
    XLA_MEM_FRACTION = {'deep': None, 'snapshot': 0.45, 'sublst': 0.5}
                                 # GPU memory fraction per process (None = not set); deep: rebuild_cell.sh:26 sets none;
                                 #   snapshot: XLA_PYTHON_CLIENT_MEM_FRACTION, v9:183, cube:96; sublst: DEEP_XLA_MEMFRAC=0.5,
                                 #   sublst_refine_pass.sh:22 (module default 0.45, phase3_sublst_refine.py:98)
    GPU_LOCK_WAIT_S = 86400      # s, longest wait for a GPU lock held by this step (phase3_sublst_refine.py:192 and phase3_deep.py:390
                                 #   use flock -w 86400); lock files per (host, GPU) from the site file GPU_LOCKS. Every GPU child
                                 #   (deep solve, snapshot flow, sub-LST refine) refuses on a host/GPU the site names no lock for
                                 #   (the site file's rule: a GPU another chat holds stays unused until the site names its lock)
    RULED = {'GPU_LOCK_WAIT_S': 'FIXED-FLOW-BATCH-EXCLUSIVE-PER-NODE-GPU-RC-6'}


class flow_prep(_Section):
    """Deep: stage-in (stored-flow copy), prep (coadd.process_lst_freq in place), solve (batch_flow_align in a child under
    site PYTHON['flow']) and check. Refuses without the done `masters --tier deep --stages stage-in` record of the cell (the
    reference it aligns to is masters'). Reference mode and table, cut-over, the astrometry gate, the NaN gate and the safe
    scale floor are masters_build keys."""
    STAGE_IN_FLOWS = True        # copy the working cell's stored flows into <out>/<cell>/flow/ before prep, rsync -a semantics (mtimes
                                 #   kept: the freshness gate compares them); rebuild_cell.sh:44. Files masters stage-in already copied
                                 #   identically are not copied again. False = every night is recomputed (as --deep-flow-all)
    BATCH_TIMEOUT_S = 7200       # s, one batch_flow_align call; coadd_full_pipeline.py:4205 (subprocess timeout)
    MOVE_ASIDE_STALE = True      # before solve, move a stale pair's two outputs to <W>/stale_flow_<UTC>/ so batch_flow_align (which
                                 #   SKIPS a pair whose outputs exist, batch_flow_align.py:76-80) recomputes it. Pure fix of
                                 #   RI-ops-deep-masters-002 (today the stale flow stays and the provisional stage refuses the cell);
                                 #   False = production behaviour
    OUT_ROOT_GPU_LOCK = True     # hold <out>/.gpu_lock around the batch as coadd._run_batch_flow does (coadd_full_pipeline.py:4198-4202);
                                 #   the site GPU_LOCKS of this host and GPU (node-global, ledger 1095) are held as well (required)
    RULED = {'MOVE_ASIDE_STALE': 'RATIFY-COADD-FLOW-REFRESH-STAGE-AND-FRESHNESS-GATE'}


class flow_snapshot(_Section):
    """Snapshot per-frame flow to the multi-night Master (Gregg 2026-09-12; R-SNAP-FLOW-ALIGNMENT-AND-FLOW-MOVIE-ARE-PIPELINE-
    PRODUCTS-FLEET-V4). Fields ship to paths.SNAPSHOT_FLOW_ROOT/<cell>/ and the flow QA files (flow_<sb>.csv, .log) to
    paths.SNAPSHOT_FLEET_OUT/<cell>/flow/; with --out D both go under D (D/snapshot_flow/, D/snapshot_fleet_out/)."""
    MASTERS_ROOT = '/lustre/pipeline/exopipe/phase3/snapshot_masters'   # 456 links into snapshot_masters_gen4; v9:54 (P3_SNAP_MASTERS),
                                 #   cube:23 (CUBEPASS_MASTERS); FREEZE-SNAPSHOT-MASTERS-GEN4; RULINGS-GREGG-20261008-A-D-STACK-FLOWREF-
                                 #   MEMBERSHIP-GEN4-FORCED-I1MIN item C (ledger 2024). PROTECTED (frames_regrid.MASTERS is the same tree)
    MASTER_FILE = '{lst}_{sb}/coadd_final/I_deep_Taper_R0_{sb}.fits'   # v9:131, 147, 153, 186; cube:69, 85, 99
    STAGE_DIR = 'snapshot_stage/{cell}'      # staging under --work when the frames stage record names no root (production
                                             #   /fast/gh/main/phase3/snapshot_stage/<cell>, v9:87; frames_stage.STAGE_SUBDIR)
    WORK_FLOW_DIR = '{cell}/flow'            # flow work dir under --work (spec 4.2 <work>/<cell>/flow/; frames_cube.FLOW_DIR reads
                                             #   the same); the production fleet's is /fast/gh/main/phase3/snapshot_out/<cell>/flow (v9:88, 142)
    REQUIRE_STAGE_RECORD = True  # G-S2 (DESIGN 11.5): the frame set comes from the done record of `frames --tier snapshot --stages
                                 #   stage` (<cell>_stage under <out>/_records/frames/ or frames' own default root). False = the staging's
                                 #   .filelist (the fleet's discovery list, v9:106-111) or its listing is the frame set - for parity on a
                                 #   fleet staging before frames' snapshot tier exists; recorded in the unit
    LOWRES = 8                   # stored field = every 8th pixel; phase3_snapshot_flow.py:51 (module default; passed only when changed);
                                 #   also read by flow_movie :31, lightcurves :149-150, the cube FLOW extension
    HORIZON_ELEVATION_DEG = 10.0 # deg, horizon radius given to preprocess(); phase3_snapshot_flow.py:58 (passed only when changed)
    BLOCK_S = 0                  # s; 0 = per-frame flow; phase3_snapshot_flow.py:59 (block mode measured not to fix 82 MHz, ledger 1054;
                                 #   no launcher uses it; passed only when changed)
    BLANK_FINITE_FRACTION = 0.5  # a Stokes I frame is blank when its finite fraction is below this or every finite pixel is 0;
                                 #   phase3_snapshot_flow.py:99 (literal; the content test recomputes FLOWED=False legitimacy with it)
    FLOW_MAX_MHZ = 1000.0        # MHz; subbands above are not flowed (1000 = no cap); phase3_snapshot_search.py:662 (P3_SNAP_FLOW_MAX_MHZ,
                                 #   the search's matching switch); the 69 MHz cap is RULING PENDING (contract S-1.2)
    SHIP_ON_EXISTING = 'refuse'  # ship: a destination file with different bytes refuses (today `cp -n` keeps the old file silently,
                                 #   v9:336-341; B-49 default); identical bytes = already shipped
    UNREADABLE_HEADER_HAS_VIS = True   # B-23 default (c): a staged frame whose header cannot be read counts as "has visibilities"
                                       #   (expected to flow), as the fleet heredoc does (v9:164-169)
    EXCLUDED_SUBBAND = 'flow'    # 'flow' = production: an (LST, subband, night) that data_decisions excludes is still flowed (the fleet
                                 #   flows every staged subband; the snapshot search refuses it, phase3_snapshot_search.py:1492) and the
                                 #   exclusion is recorded (admitted night set empty); 'refuse' = refuse the unit (open question)
    RULED = {'MASTERS_ROOT': 'FREEZE-SNAPSHOT-MASTERS-GEN4',
             'FLOW_MAX_MHZ': 'MEASURED-BLOCK-FLOW-DOES-NOT-FIX-82MHZ-PER-BAND-FLOW-CAP-SWITCH-ADDED',
             'BLOCK_S': 'MEASURED-BLOCK-FLOW-DOES-NOT-FIX-82MHZ-PER-BAND-FLOW-CAP-SWITCH-ADDED',
             'SHIP_ON_EXISTING': 'R-SNAP-FLOW-ALIGNMENT-AND-FLOW-MOVIE-ARE-PIPELINE-PRODUCTS-FLEET-V4'}
    PROTECTED_KEYS = ('MASTERS_ROOT',)


class flow_refine(_Section):
    """Sub-LST x2 refine (arm B; Gregg 2026-09-25 'That must happen with the Master that will be subtracted';
    R-SUBLST-X2-FLOW-TO-THE-SUBTRACTED-MASTER): clv15/phase3_sublst_refine.py run IN PLACE as a child under site
    PYTHON['default'] (it shells py310_orca itself). Its rules are enforced inside the module; the values below document them
    and drive the step's own content test, which recomputes the allowed residual set independently (test (b) R1/R2).
    Window stacks are read from paths.SUBLST_ROOT/<LST>/ (masters sub-LST); residuals go to <out>/<LST>_x2/."""
    MASTERS_DIR = '{lst}_masters'    # arm-A window Masters LOOMSTR must name, under paths.SUBLST_ROOT; sublst_resume_from_x2.sh:60-72
    WORK_SUBDIR = 'sublst_x2_work'   # node-local work under --work, gpu<g>/ below it (production /fast/gh/main/phase3/sublst_x2_work,
                                     #   sublst_refine_pass.sh:14)
    WEIGHTS_CSV = '/lustre/gh/main/phase3/sublst/deep_coadd_records_frozen_20261008/{lst}/{sb}/weights.csv'
                                 # frozen deep co-add records; phase3_sublst_loo.py:36 (refine.py:72, 76); drift-guarded against the module;
                                 #   IMPLEMENTED-RULINGS-A-B-20261008-SUBLST-FLOWREF-AND-MEMBERSHIP-VETOES (ledger 2026). PROTECTED
    MEMBERSHIP_VETO_PIN = ('/lustre/gh/main/phase3/sublst/deep_coadd_records_frozen_20261008/membership_vetoes_pinned.csv',
                           '851b16d86fe059c8c8bf43b1afa72344')   # (path, md5); phase3_tier_rules.py:54-55 (drift-guarded); ruling B
                                                                 #   2026-10-08. PROTECTED
    WINDOWS = (0, 1, 2, 3, 4, 5)     # 10-min sidereal windows per LST hour; phase3_sublst_refine.py:78 (module default; passed only for a subset)
    MIN_NIGHTS_PER_WINDOW = 2        # nights with a stack, per Stokes; refine.py:129 (literal); RULING-10-SUBLST-REFINE-ARM-A-RULE-IMPLEMENTED
    MIN_WEIGHTED_NIGHTS_PER_WINDOW = 2   # nights with nonzero co-add weight, per Stokes; refine.py:129 (literal); same ruling
    MIN_LOO_WEIGHTED_NIGHTS = 2      # weighted nights in a residual's own leave-one-out Master; refine.py:165-166 (literal);
                                     #   IMPLEMENTED-Q11-PER-NIGHT-LEAVE-ONE-OUT-RULE-AND-COUNTS (ledger 1990)
    X2_VALID_FOOTPRINT = 0.999       # fraction of a native pixel's x2 footprint valid after the warp; refine.py:221 (literal, documented)
    ALIGNER_OK_RC = (0, 2)           # align_images.py exit codes the module accepts; refine.py:203 (literal, documented)
    FFT_WORKERS = 8                  # scipy.fft threads, REFINE_FFT_WORKERS in the child environment; refine.py:56
    JUPITER_MASK_RADIUS_BEAMS = 2.0  # x BMAJ, Jupiter masked per night in the LOO sums; phase3_jupiter_mask.py:14 (drift-guarded);
                                     #   RULING-MOVING-JUPITER-MASK-IN-MASTERS (ledger 1783)
    DURATION_FALLBACK_S = 600.0      # s, DURATION card when DATE-OBS/END unreadable (TEFFSEC first); refine.py:232-235 (literal; B-30)
                                     # (the six literals above are documented, never passed: an override refuses in refine)
    # P3_FLOW_LOCK (sublst_refine_pass.sh:22, /fast/gh/main/phase3/.sublst_refine_gpu<g>.lock) = the site GPU_LOCKS entry of
    #   (host, GPU); a host/GPU the site names no lock for is refused (G-C11) - review 2026-10-13 removed the FLOW_LOCK fallback
    APPROVED_REFUSALS_CSV = '/lustre/gh/main/exopipe_ops/sublst_refine_approved_refusals.csv'   # 9 rows, 2 (LST, subband);
                                     #   sublst_refine_approved.py:14; IMPLEMENTED-01H-PROCEED-WITHOUT-18MHZ-20251014-W0-2 (1679),
                                     #   APPLIED-02H-50MHZ-20250103-REFUSAL (1873); moves to the rulings registry (DESIGN 11.6). PROTECTED
    EXCLUDED_STACK = 'refuse'        # a window stack of an (LST, subband, night) that data_decisions excludes refuses the unit (README
                                     #   rule 6, C-09/U-1c). Production does not check (the stacker leaves excluded nights out, so this
                                     #   fires only on a stack older than an exclusion - deep requirement (d)); 'record' = production
    RULED = {'MIN_NIGHTS_PER_WINDOW': 'RULING-10-SUBLST-REFINE-ARM-A-RULE-IMPLEMENTED',
             'MIN_WEIGHTED_NIGHTS_PER_WINDOW': 'RULING-10-SUBLST-REFINE-ARM-A-RULE-IMPLEMENTED',
             'MIN_LOO_WEIGHTED_NIGHTS': 'IMPLEMENTED-Q11-PER-NIGHT-LEAVE-ONE-OUT-RULE-AND-COUNTS',
             'JUPITER_MASK_RADIUS_BEAMS': 'RULING-MOVING-JUPITER-MASK-IN-MASTERS'}
    PROTECTED_KEYS = ('WEIGHTS_CSV', 'MEMBERSHIP_VETO_PIN', 'APPROVED_REFUSALS_CSV')
