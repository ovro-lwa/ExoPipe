"""Configuration of `exopipe3 frames` - every tunable of the command, one section per stage or tier.

Units folded here: frames-deep, frames-tiers. A key in its section's PROTECTED_KEYS is locked
(a mirror of in-place code, a pinned input or a ruling-fixed value: it changes only by a commit); RULED names the
ledger ID of the ruling behind a key. Per run: --set SECTION.KEY=VALUE or --config FILE; `exopipe3 config --step frames` lists the sections.
"""
from exopipe.phase3.config import _Section


# =====================================================================================================================
# unit frames-deep (was exopipe/phase3/_cfgdraft_frames-deep.py)
# Configuration of `exopipe3 frames`, deep tier (unit frames-deep; spec plan/specs/frames.md section 3).
#
# Sections HERE: ``frames`` (the deep tier's stages and attempt root) and ``frames_residuals``
# (what only the deep frame-only rebuild decides). The sub-LST / snapshot sections (frames_sublst, frames_snapshot) are the
# frames-tiers unit's draft.
#
# Decisions of the refactor chat (2026-10-13 ~04:30 UTC, plan/specs/_CROSSCHECK.md top + group 14 + section 4.3), applied:
# - During the window `search` (stage detect) PRODUCES the deep frames; `frames --tier deep` is for frame-only rebuilds and
#   is never chained before search. Every engine value that search's config carries is read from search's sections - the
#   effective keys - and NOT duplicated here (lib/frames/deep_engine.engine_values looks each up by key name in search,
#   search_inputs, search_detect: MASTER_GEN, ALLOW_OLD_MASTERS, ARCHIVE_WRITE, ROOTS_CSV, STRIATION_OFF_BATTERY, STREAMING,
#   NPROC_RESID, MIN_EPOCHS, SIGMA_STACK_MIN_EPOCHS, SIGMA_STACK_BUILD_MIN, SANITY_X, SEARCH_TIER_SIGMA, BEAM_GATE_RATIO,
#   BEAM_GATE_ON, FLOW_XLA_MEM_FRACTION, FLOW_PYTHON, FLOW_LOCK, STRIATION_WINDOW_BEAMS, RUN_LABEL, STOKES_ORDER,
#   ENGINE_PYTHON, UNACCOUNTED_EPOCHS). The engine literals that decide a frame and that search does NOT carry are mirrored
#   below (frames_residuals, PROTECTED); if search ever carries the same key, engine_values refuses unless both agree.
# - Shared roots come from config.paths (PHASE3_ARCHIVE_ROOT, ARCHIVE_RUN, OPS_ROOT): no frames copy of them.
# - Protection by PROTECTED_KEYS / RULED (honoured by config/__init__.py); no lower-case markers.
# Values = production at the refactor baseline; file:line cites the production source (phase3_deep.py = the worktree's
# lib/search/deep_search.py, same line numbers).
# =====================================================================================================================

class frames(_Section):
    """exopipe3 frames - the deep tier's stages and attempt root (spec frames.md 1.2, 4). Sub-LST / snapshot stage orders
    and roots: frames_sublst, frames_snapshot (unit frames-tiers)."""
    STAGES = {'deep': ('residuals', 'verify')}     # default --stages of the deep tier, in run order (spec 1.2)
    OPTIONAL_STAGES = {'deep': ()}                 # stages that run only when named in --stages
    # Production attempt root of the deep frame build: one collect tree per node, one attempt directory
    # <cell>_<S>_<UTC> per unit run (p3s5_search_fleet.sh:33 COLLECT, :187-188, :223). With --out D the attempts go to
    # D/deep_attempts and the archive copies to D/archive (parity product map, DESIGN 11.1).
    DEEP_ATTEMPTS_ROOT = '/lustre/gh/main/phase3_collect/p3s5_collect_{host}/attempts'   # under paths.PRODUCTION_ROOTS


class frames_residuals(_Section):
    """Deep frame-only rebuild (`frames --tier deep`; B11, contract steps 26-35): the keys only this step decides. The
    engine (lib/search/deep_search.run_deep_cell(frames_only=True)) runs in a fresh child per unit (core/legacy.run); its
    gate and residual values are search's keys (module docstring of this draft)."""
    # Who produces the deep search frames during the refactor window (refactor chat 2026-10-13, crosscheck group 14; not a
    # ledger ruling): 'search' = `search --tier deep` (detect) is the producer, and `frames --tier deep` residuals units
    # refuse unless --deep-rebuild states that the run is a frame-only rebuild (so no chain runs frames before search).
    # 'frames' = after cutover, if the frames/search split is ruled (search would then read the frames from disk).
    DEEP_PRODUCER = 'search'
    # Missing x2 flows: 'compute' = the engine computes them on this node's GPU under the node flow lock (production,
    # phase3_deep.py:361-403; the one documented GPU exception of DESIGN 11.2, crosscheck decision 9); 'require' = a
    # missing flow refuses the unit (CPU hosts, or a cache supplied by a run on the GPU node). --deep-x2-flows overrides.
    X2_FLOWS_MISSING = 'compute'
    # Node-local flow / spill directory under the scratch root (--work or site SCRATCH_ROOT); production P3_FLOW_ROOT is
    # /fast/gh/main/phase3/s2work_p3s5 (p3s5_search_fleet.sh:32). A frames run never shares production's flow cache.
    WORK_SUBDIR = 's2work_frames'
    UNIT_WORKERS = 1                   # units (engine children) at once in one invocation (default --workers); production
                                       # lanes V 6 / I 4 per node are `run submit --nodes host:k` (p3s5_launch.sh:23-24)
    # ---- mirrors of engine literals that decide a frame and that search's config does not carry. No environment variable
    # reaches them, so they are PROTECTED until the engine is absorbed; the content tests and the dry-run preview read them.
    PB_FLOOR = 0.05                    # PB fraction; search-frame sibling pixels at or below are NaN; carried in no card
                                       # (phase3_deep.py:810; contract step 34) - NOT pb_apply.PB_FLOOR = 0.1 of master siblings
    CENTRAL_RMS_HALF_PX = 512          # px; THERMNOI / IMGRMS = mad_std of the central box of half-size min(512, N/4)
                                       # (phase3_deep.py:1617-1618, 1731-1732)
    SANITY_ALIGN_X = 10.0              # x cell median weights.csv qa3_rms_mjy: the epoch is removed, NO frame (phase3_deep.py:1159;
                                       # R-SEARCH-SANITY-GATE-10X, R-BEAM-AND-10X-REFUSED-EPOCHS-STAY-OUT-NO-FRAMES-OWED)
    GATE_MIN_NIGHTS = 5                # nights with a value, else the 10x sanity gate is skipped (silently, inventory E12) and the
                                       # beam gate loudly (floor waived) (phase3_deep.py:1156, 1345-1351)
    LOO_NCOADD_RANGE = (2, 20)         # nights (NCOADD of the own-Stokes central master): inside it an in-master epoch must
                                       # subtract its own LOO master (phase3_deep.py:1403; audit R-19)
    SIGSTACK_FLOOR = 1.0               # ratio; noise map = across-epoch MAD / clean-sky median, floored at 1 (phase3_pipeline.py
                                       # :1383-1390 = lib/search/identification.py build_sigma_stack)
    PROTECTED_KEYS = ('DEEP_PRODUCER', 'PB_FLOOR', 'CENTRAL_RMS_HALF_PX', 'SANITY_ALIGN_X', 'GATE_MIN_NIGHTS',
                      'LOO_NCOADD_RANGE', 'SIGSTACK_FLOOR')
    RULED = {'PB_FLOOR': 'LANDED-P3S3-PRODUCTS-667a56e',
             'SANITY_ALIGN_X': ('R-SEARCH-SANITY-GATE-10X', 'R-BEAM-AND-10X-REFUSED-EPOCHS-STAY-OUT-NO-FRAMES-OWED'),
             'GATE_MIN_NIGHTS': 'R-BEAM-GATE-FIVE-NIGHT-FLOOR-WAIVED-THIN-CELLS-SEARCHED',
             'LOO_NCOADD_RANGE': ('R-LOO-MASTERS-AND-REDO-SEQUENCE', 'R-LOO-PER-STOKES-RATIFIED'),
             'SIGSTACK_FLOOR': 'LANDED-P3S3-PRODUCTS-667a56e'}


# =====================================================================================================================
# unit frames-tiers (was exopipe/phase3/_cfgdraft_frames-tiers.py)
# Configuration of `exopipe3 frames --tier sublst|snapshot` (unit frames-tiers).
#
# Two sections, one per tier (crosscheck decision 7: name = <step>_<tier>): `frames_sublst` and `frames_snapshot`. Values =
# production at the refactor baseline; spec plan/specs/frames.md sections 3, 3.9, 5.2, 5.3, adapted to the binding decisions at
# the top of plan/specs/_CROSSCHECK.md (2026-10-13 ~04:30 UTC):
# - sub-LST arm A (phase3_sublst_loo.py) is owned by `masters --tier sublst --stages loo`: no arm-A keys here; wideband gates on
#   the masters loo record (LOO_EVIDENCE).
# - the per-hour sub-LST window stacks run through the ONE wrapper of lib/masters/tier_sublst.py with config section
#   `masters_sublst` (no stack keys here).
# - frames owns sub-LST wideband (and tree B, content test (b)) and the snapshot stage/regrid/cube/movies incl. the flow
#   movies (flow drops its `movie` stage).
# - shared roots come from CFG.paths (SUBLST_ROOT, SNAPSHOT_CUBES_ROOT, SNAPSHOT_FLEET_OUT, SNAPSHOT_FLOW_ROOT, OPS_ROOT,
#   CLV15_ROOT, PHASE1_ROOT); the frozen deep records from masters_sublst.FROZEN_RECORDS; the snapshot master links from
#   masters_snapshot.LINK_ROOT - never a second key for them here.
# - interpreters: site PYTHON keys default / flow / py36 only.
# - RULED and PROTECTED_KEYS (honoured by config/__init__.py). PROTECTED_KEYS also holds every value HARD-CODED in an in-place clv15
#   engine (no argument reaches it): an override could not change the product, so no override is accepted until the engine is
#   absorbed into lib/frames. Keys that reach an engine argument are passed only when they differ from the production
#   default, so a default run has exactly the production command line.
#
# Line numbers cite clv15 39bdfd3 and exopipe_ops 0c1580c: nocat = exopipe_ops/sublst_run_lst_nocat.sh, v9 =
# exopipe_ops/snapshot_fleet_node_v9.sh, cube pass = exopipe_ops/snapshot_cube_pass.sh, wideband / tree / regrid / cube /
# frame_movie / flow_movie = clv15/phase3_sublst_wideband.py, phase3_sublst_tenmin_tree.py, phase3_snapshot_regrid.py,
# phase3_snapshot_cube.py, phase3_snapshot_frame_movie.py, phase3_snapshot_flow_movie.py. Ledger IDs are fix_ids of
# clv15/fix_ledger.csv.
# =====================================================================================================================

_SUBBANDS = ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz',
             '73MHz', '78MHz', '82MHz')


class frames_sublst(_Section):
    """exopipe3 frames --tier sublst (lib/frames/tier_sublst.py): per-hour window stacks (through masters' wrapper),
    wideband windows (clv15/phase3_sublst_wideband.py in place), tree B (clv15/phase3_sublst_tenmin_tree.py in place),
    content test (b). The sub-LST root is the run's --out (production paths.SUBLST_ROOT), laid out as production:
    <root>/<LST>/, <root>/<LST>_x2/, <root>/<LST>_masters/, <root>/tenmin_root_x2/."""
    # ---- stages ------------------------------------------------------------------------------------------------------
    ORDER = ('stack', 'wideband', 'tree', 'searchframes', 'verify')   # run order of any --stages list (frames.md 1.2)
    # default --stages per operating mode (DESIGN 12): 'per-hour' = --epochs <LST>_<date> (one night): the night's window
    # stacks; the x2 refine (`flow --tier sublst`, GPU) follows, then `frames --stages wideband,tree,verify` for the night.
    # 'intermittent' = --lsts <LST> (every night): after `masters --tier sublst --stages loo` and the refine.
    STAGES = {'per-hour': ('stack',), 'intermittent': ('wideband', 'tree', 'verify')}
    # ---- wideband (S-G8 .. S-G10) --------------------------------------------------------------------------------------
    GROUPS = ('Red', 'Green', 'Blue')       # nocat:256 (WBS); Full not built; Gregg 2026-10-01 'Go, Red/Green/Blue first'
    WIDEBAND_MIN_MEMBERS = 2                # member subbands per (night, window, group); wideband:83 (engine literal)
    WIDEBAND_WEIGHT = 'inv_var_V_inner_std'  # 1 / nanstd(central half of each axis of the member's V window)^2, same for I and V; wideband:37-40, 79-82 (engine literal)
    WIDEBAND_PARALLEL = 12                  # (group, night) processes at once; nocat:258 (xargs -P 12)
    # arm A must be complete before wideband (nocat:232-234). 'record' = the done record of `masters --tier sublst --stages
    # loo` (unit loo_<LST>_<sb>) under this run's root (crosscheck decision 6); 'log' = production's own evidence, the final
    # 'SUBLST-LOO <LST> <sb> <S> windows=' line of <root>/<LST>/logs/loo_<LST>_<sb>_<S>.log (for parity runs on production
    # arm-A outputs that have no masters record). No ruling: a refactor seam choice.
    LOO_EVIDENCE = 'record'
    REFINE_APPROVED_SCRIPT = 'sublst_refine_approved.py'   # exopipe_ops script judging approved refine refusals (nocat:243); its rows: sublst_refine_approved_refusals.csv
    # ---- tree B (S-G11) ----------------------------------------------------------------------------------------------
    TREE_SUBDIR = 'tenmin_root_x2'          # <root>/tenmin_root_x2/<LST>_<band>/ (nocat:263)
    TREE_BANDS = _SUBBANDS + ('Red', 'Green', 'Blue')   # nocat:263-264 ($SBS $WBS)
    TREE_COLUMNS = ('key', 'date', 'bin', 'stokes', 'src', 'dc', 'has_flow', 'src_dewarped', 'pb_blank_frac',
                    'pb_ratio_median', 'bmaj_deg', 'bmin_deg', 'bpa_deg', 'n_inject', 'sub_path', 'fluxbake')   # tree:31-34 (engine literal)
    TREE_PARALLEL = 4                       # bands built at once: production runs ONE serial process for all 18 bands (nocat:263);
                                            # one process per band builds identical trees (each band is independent, tree:68-131)
    WINDOWS = 6                             # 10-min sidereal windows per LST hour (wideband:66, loo:82; engine literal)
    # ---- engines -------------------------------------------------------------------------------------------------------
    INTERPRETER = {'wideband': 'default',           # PYW = py38_orca_nkosogor, nocat:257
                   'tree': 'py36',                  # PY6 = development, nocat:263
                   'refine_approved': 'py36'}       # PY6, nocat:243
    RULED = {'GROUPS': 'IMPLEMENTED-10MIN-WIDEBAND-RGB-IN-PROGRESS'}
    PROTECTED_KEYS = ('WIDEBAND_MIN_MEMBERS', 'WIDEBAND_WEIGHT', 'TREE_COLUMNS', 'WINDOWS', 'REFINE_APPROVED_SCRIPT',
                      'ORDER', 'INTERPRETER')


class frames_snapshot(_Section):
    """exopipe3 frames --tier snapshot (lib/frames/tier_snapshot.py): stage (port of v9:92-122 / cube pass:39-64), regrid
    (clv15/phase3_snapshot_regrid.py), cube (clv15/phase3_snapshot_cube.py write + verify, ship + MANIFEST), movies
    (clv15/phase3_snapshot_frame_movie.py and phase3_snapshot_flow_movie.py). Item: a night cell <LST>_<date>[_<subband>]."""
    # ---- stages ------------------------------------------------------------------------------------------------------
    ORDER = ('stage', 'regrid', 'wideband', 'cube', 'movies')   # frames.md 1.2; 'wideband' is not offered in the window (B9)
    # default --stages: the snapshot chain is `frames --stages stage,regrid` -> `flow --tier snapshot` (GPU) ->
    # `frames --stages cube,movies` -> search ... on the node that holds the staging (crosscheck 4.7 'snapshot-cell').
    DEFAULT_STAGES = ('stage', 'regrid')
    CHAIN = 'search'                        # whose rules apply: 'search' = the fleet (v9), 'cube' = the cube pass; selects MIN_FREE_GB, MIN_FRAMES, OFF_GRID
    # ---- stage (N-G1 .. N-G3) ------------------------------------------------------------------------------------------
    DISCOVERY = 'legacy'                    # 'legacy' = clv15 phase3_snapshot_search.discover_frames (v9:106-111); 'integration' = core.inputs.phase1_frames (B-15 a, landed 3a7ef3b). Today's behaviour until Gregg answers B-15
    SUBBANDS = _SUBBANDS                    # v9:47 (P3_FLEET_SUBBANDS default); cube pass:25
    STAGE_SUBDIR = 'snapshot_stage'         # node-local staging <scratch>/snapshot_stage/<cell>/<sb>/ (v9:87)
    MIN_FREE_GB = {'search': 800, 'cube': 700}   # GB free on the staging filesystem; v9:95 (P3_FLEET_MINFREE_GB); cube pass:26; STALL-SNAPSHOT-CUBE-PASS-STOPPED-ON-FAST-DISK-SPACE
    COPY_PROCS = 8                          # parallel copies; v9:112 (xargs -P 8)
    FUNPACK_PROCS = 16                      # parallel funpack -S to <o>.tmp then rename; v9:113-114; SNAP-REDESIGN-STAGE-FS-FIX
    FUNPACK = None                          # funpack binary; None = site FUNPACK, else <dir of site PYTHON['py36']>/funpack (= v9:45)
    MIN_FRAMES = {'search': 1000, 'cube': 1}     # decompressed frames per cell; v9:199-201 (P3_FLEET_MIN_FRAMES; tested after the flow there, at stage end here: frames.md 11); cube pass:59 refuses only 0
    CLEAN_AFTER = False                     # remove the node-local staging at the end of this invocation (the chain's last frames step sets it)
    # ---- regrid (N-G4, N-G5) -------------------------------------------------------------------------------------------
    MASTER_FILE = '{links}/{lst}_{sb}/coadd_final/I_deep_Taper_R0_{sb}.fits'   # under masters_snapshot.LINK_ROOT; v9:135, :146, :153
    ON_GRID_NAXIS = 1508                    # px; a first frame of another NAXIS1 runs the regrid (an unreadable header counts as 1508, B-22); v9:129-130
    REGRID_NPROC = 8                        # v9:137 (--nproc 8)
    REGRID_ORDER = 1                        # bilinear, NaN outside; regrid:30 (engine literal)
    OFF_GRID = {'search': 'regrid', 'cube': 'refuse'}   # fleet regrids (v9:126-138); cube pass refuses frames off their own Master's grid (cube pass:66-81); B-22 default (c)
    # ---- cube (N-G6 .. N-G9) -------------------------------------------------------------------------------------------
    CUBE_QUANT = 8.0                        # RICE quantize level; cube:43; passed as --quant only when changed
    CUBE_FLOW_SCALE_PX = 1e-3               # px per int16 unit, raised to max|flow|/32000 per cube; cube:44, :150 (engine literal)
    CUBE_WRITE_PROCS = 4                    # (subband, Stokes) write+verify at once (NWRITE); cube pass:16
    CUBES_PER_SUBBAND = 2                   # I and V; COMPLETE at 2 x staged subbands; cube pass:113-117
    CUBE_PRODUCT = 'snapshot-flowcube-v1'   # PRODUCT card; cube:165 (engine literal)
    CUBE_WORK_SUBDIR = 'cube_pass'          # node-local cubes before shipping: <scratch>/cube_pass/<cell>/cubes (cube pass:38)
    # where the snapshot flow fields flow_<sb>_<ts>.npy are read (--snap-flow-dir overrides): the flow step's node-local
    # <scratch>/<cell>/flow (flow.md 4.2), else its shipped copy <paths.SNAPSHOT_FLOW_ROOT>/<cell> (cp -n of the same files)
    FLOW_DIR = '{scratch}/{cell}/flow'
    # ---- movies (N-G10) ----------------------------------------------------------------------------------------------
    MOVIE_SOURCE = 'staged'                 # 'staged' (fleet, v9:277-281) | 'cube' (cube pass:136; snapshot_movie_pass.sh)
    MOVIES_ROOT = '/lustre/gh/main/phase3/snapshot_movies'   # <root>/<cell>/frames_<cell>_<band>_I.mp4, movies.log (v9:278)
    FRAME_BANDS = _SUBBANDS + ('Red', 'Green', 'Blue', 'Full')   # R-FRAME-MOVIES-19-BANDS (Gregg 2026-09-27); frame_movie:42 (--bands all)
    STRETCH_K = 12.0                        # +/- K x median frame rms; frame_movie:28; R-FRAME-MOVIE-STRETCH-PM12
    MOVIE_WIDTH_PX = 754                    # frame_movie:30
    MOVIE_FPS = 24.0                        # frame_movie:32
    MOVIE_CRF = 26                          # frame_movie:32
    MIN_FRAMES_PER_BAND = 10                # fewer staged flowed frames = no movie for that band; frame_movie:56 (engine literal); a band
                                            # skipped so is COUNTED by the step (fallbacks.FRAME_MOVIE_SKIP_FALLBACK; Gregg 2026-10-14)
    MOVIE_BACKGROUND_MIN_ADMITTED = 10      # fewer admitted frames = the movie's background from every frame with data; frame_movie:110-112
                                            # (engine literals, both drift-checked); its 'BACKGROUND <sb>: ...' line (clv15 40381cb) is
                                            # COUNTED by the step (fallbacks.FRAME_MOVIE_BACKGROUND_FALLBACK)
    FLOW_MOVIES = True                      # flow_<sb>.mp4 + series per subband (v9:270-271); frames owns them (crosscheck decision 6)
    FLOW_MOVIE_DURATION_S = 10.0            # s (Gregg 2026-09-12); flow_movie:28
    FLOW_MOVIE_PROCS = 4                    # subbands at once; v9:271 (xargs -P 4)
    # ---- engines -------------------------------------------------------------------------------------------------------
    INTERPRETER = {'regrid': 'py36',        # PY = development, v9:44, :137
                   'cube_write': 'flow',    # FLOWPY = py310_orca (astropy >= 5.3), cube pass:110
                   'cube_verify': 'py36',   # PY, cube pass:111
                   'frame_movies': 'flow',  # FLOWPY, v9:279; cube pass:136
                   'flow_movies': 'flow'}   # FLOWPY, v9:271
    RULED = {'OFF_GRID': 'APPROVED-AND-BUILT-PREAMBLE-E-PILOT-CELLS-REGRID-AND-RAISED-DIRTY-I-FLOOR-CALIBRATION-PENDING', 'CUBE_QUANT': 'R-SNAP-CUBES-STEP1-BUILT-TEST-RUNNING',
             'FRAME_BANDS': 'R-FRAME-MOVIES-19-BANDS', 'STRETCH_K': 'R-FRAME-MOVIE-STRETCH-PM12',
             'MIN_FRAMES_PER_BAND': 'R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS', 'MOVIE_BACKGROUND_MIN_ADMITTED': 'R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS'}
    PROTECTED_KEYS = ('REGRID_ORDER', 'CUBE_FLOW_SCALE_PX', 'CUBE_PRODUCT', 'CUBES_PER_SUBBAND', 'MIN_FRAMES_PER_BAND',
                      'MOVIE_BACKGROUND_MIN_ADMITTED', 'ORDER', 'INTERPRETER')
