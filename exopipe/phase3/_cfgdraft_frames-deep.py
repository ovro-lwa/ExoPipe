"""Config draft of `exopipe3 frames`, deep tier (unit frames-deep; spec plan/specs/frames.md section 3). Loaded by
config._load_drafts during the refactor window; the refactor chat folds it into config.py at ``# @@STEP-SECTIONS@@``.

Sections HERE: ``frames`` (the deep tier's stages and attempt root) and ``frames_residuals``
(what only the deep frame-only rebuild decides). The sub-LST / snapshot sections (frames_tiers, frames_armA, ...) are the
frames-tiers unit's draft.

Decisions of the refactor chat (2026-10-13 ~04:30 UTC, plan/specs/_CROSSCHECK.md top + group 14 + section 4.3), applied:
- During the window `search` (stage detect) PRODUCES the deep frames; `frames --tier deep` is for frame-only rebuilds and
  is never chained before search. Every engine value that search's config carries is read from search's sections - the
  effective keys - and NOT duplicated here (lib/frames/deep_engine.engine_values looks each up by key name in search,
  search_inputs, search_detect: MASTER_GEN, ALLOW_OLD_MASTERS, ARCHIVE_WRITE, ROOTS_CSV, STRIATION_OFF_BATTERY, STREAMING,
  NPROC_RESID, MIN_EPOCHS, SIGMA_STACK_MIN_EPOCHS, SIGMA_STACK_BUILD_MIN, SANITY_X, SEARCH_TIER_SIGMA, BEAM_GATE_RATIO,
  BEAM_GATE_ON, FLOW_XLA_MEM_FRACTION, FLOW_PYTHON, FLOW_LOCK, STRIATION_WINDOW_BEAMS, RUN_LABEL, STOKES_ORDER,
  ENGINE_PYTHON, UNACCOUNTED_EPOCHS). The engine literals that decide a frame and that search does NOT carry are mirrored
  below (frames_residuals, PROTECTED); if search ever carries the same key, engine_values refuses unless both agree.
- Shared roots come from config.paths (PHASE3_ARCHIVE_ROOT, ARCHIVE_RUN, OPS_ROOT): no frames copy of them.
- Protection by PROTECTED_KEYS / RULED (honoured by config.py); no lower-case markers.
Values = production at the refactor baseline; file:line cites the production source (phase3_deep.py = the worktree's
lib/search/deep_search.py, same line numbers).
"""
from exopipe.phase3.config import _Section


class frames(_Section):
    """exopipe3 frames - the deep tier's stages and attempt root (spec frames.md 1.2, 4). Sub-LST / snapshot stage orders
    and roots: frames_tiers (unit frames-tiers)."""
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
