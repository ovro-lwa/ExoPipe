"""Config draft of `exopipe3 frames --tier sublst|snapshot` (unit frames-tiers; refactor window: config.py loads it until the
refactor chat folds it in at `# @@STEP-SECTIONS@@`).

Two sections, one per tier (crosscheck decision 7: name = <step>_<tier>): `frames_sublst` and `frames_snapshot`. Values =
production at the refactor baseline; spec plan/specs/frames.md sections 3, 3.9, 5.2, 5.3, adapted to the binding decisions at
the top of plan/specs/_CROSSCHECK.md (2026-10-13 ~04:30 UTC):
- sub-LST arm A (phase3_sublst_loo.py) is owned by `masters --tier sublst --stages loo`: no arm-A keys here; wideband gates on
  the masters loo record (LOO_EVIDENCE).
- the per-hour sub-LST window stacks run through the ONE wrapper of lib/masters/tier_sublst.py with config section
  `masters_sublst` (no stack keys here).
- frames owns sub-LST wideband (and tree B, content test (b)) and the snapshot stage/regrid/cube/movies incl. the flow
  movies (flow drops its `movie` stage).
- shared roots come from CFG.paths (SUBLST_ROOT, SNAPSHOT_CUBES_ROOT, SNAPSHOT_FLEET_OUT, SNAPSHOT_FLOW_ROOT, OPS_ROOT,
  CLV15_ROOT, PHASE1_ROOT); the frozen deep records from masters_sublst.FROZEN_RECORDS; the snapshot master links from
  masters_snapshot.LINK_ROOT - never a second key for them here.
- interpreters: site PYTHON keys default / flow / py36 only.
- RULED and PROTECTED_KEYS (honoured by config.py). PROTECTED_KEYS also holds every value HARD-CODED in an in-place clv15
  engine (no argument reaches it): an override could not change the product, so no override is accepted until the engine is
  absorbed into lib/frames. Keys that reach an engine argument are passed only when they differ from the production
  default, so a default run has exactly the production command line.

Line numbers cite clv15 39bdfd3 and exopipe_ops 0c1580c: nocat = exopipe_ops/sublst_run_lst_nocat.sh, v9 =
exopipe_ops/snapshot_fleet_node_v9.sh, cube pass = exopipe_ops/snapshot_cube_pass.sh, wideband / tree / regrid / cube /
frame_movie / flow_movie = clv15/phase3_sublst_wideband.py, phase3_sublst_tenmin_tree.py, phase3_snapshot_regrid.py,
phase3_snapshot_cube.py, phase3_snapshot_frame_movie.py, phase3_snapshot_flow_movie.py. Ledger IDs are fix_ids of
clv15/fix_ledger.csv.
"""
from exopipe.phase3.config import _Section

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
    MIN_FRAMES_PER_BAND = 10                # fewer staged flowed frames = no movie for that band; frame_movie:56 (engine literal)
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
             'FRAME_BANDS': 'R-FRAME-MOVIES-19-BANDS', 'STRETCH_K': 'R-FRAME-MOVIE-STRETCH-PM12'}
    PROTECTED_KEYS = ('REGRID_ORDER', 'CUBE_FLOW_SCALE_PX', 'CUBE_PRODUCT', 'CUBES_PER_SUBBAND', 'MIN_FRAMES_PER_BAND',
                      'ORDER', 'INTERPRETER')
