"""Configuration of `exopipe3 curves` - every tunable of the command, one section per stage or tier.

Units folded here: curves. A key in its section's PROTECTED_KEYS is locked
(a mirror of in-place code, a pinned input or a ruling-fixed value: it changes only by a commit); RULED names the
ledger ID of the ruling behind a key. Per run: --set SECTION.KEY=VALUE or --config FILE; `exopipe3 config --step curves` lists the sections.
"""
from exopipe.phase3.config import _Section


# =====================================================================================================================
# unit curves (was exopipe/phase3/_cfgdraft_curves.py)
# Config draft of `exopipe3 curves` (refactor window; config.py loads it until the refactor chat folds it in).
#
# Spec: /lustre/gh/main/exopipe_ops/refactor_20261012/plan/specs/curves.md section 3. Line numbers cite the production files
# as of clv15 6f71d28 / exopipe_ops 6b8a23a (2026-10-13):
#   LC = clv15/phase3_snapshot_lightcurves.py (extractor, last changed 98595b3)   SS = clv15/phase3_snapshot_search.py
#   RA = clv15/phase3_snapshot_refastro.py   CH = clv15/phase3_snapshot_chain.py
#   v9 = exopipe_ops/snapshot_fleet_node_v9.sh   BF = exopipe_ops/snapshot_lightcurve_backfill.sh
# "Ledger N" = line N of clv15/fix_ledger.csv.
# =====================================================================================================================

class curves(_Section):
    """`exopipe3 curves` - the snapshot light-curve product (snapshot contract section 3; preamble C approved 2026-09-17,
    APPROVED-PREAMBLE-C-SIX-CELL-LIGHT-CURVE-PRODUCT-TEST; kept: R-SNAP-KEEP-3D-GATE-AND-LIGHTCURVE-STEP 2026-09-22).
    Refactor window: the extraction runs IN PLACE - clv15/phase3_snapshot_lightcurves.py run as production runs it
    (v9:366-370) through core.legacy.run under the site interpreter 'py36' (Python 3.6.13, numpy 1.19.2, astropy 4.1).
    PROTECTED_KEYS mirror that module or describe the window-only behaviour: no override can change them (they change by a
    commit, together with the module, at absorption into lib/curves); the step also checks the module's own literals
    against these values before extracting (gate G13, lib/curves/lock.py).
    Keys in RULED (incl. every input/product path) can be overridden only into a --out outside every production root
    (paths.PRODUCTION_ROOTS and this step's own production trees, steps/curves.py _guard_roots: spec B2) or with
    --ruling-for curves.KEY=<ledger ID>.
    Shared roots read from config.paths (never a key of this section): SNAPSHOT_FLEET_OUT (search outputs: admission,
    runinfo, nightrefused; spec SEARCH_OUT_ROOT), SNAPSHOT_FLOW_ROOT (flow fields; spec FLOWTREE_ROOT, exported to the
    module as P3_SNAP_FLOWTREE), CLV15_ROOT (target-sample CSVs = the extractor's --samples-dir default, LC:257; spec
    SAMPLES_DIR), PHASE1_ROOT (window-locked: the module's own SS:72)."""
    PROTECTED_KEYS = ('TIERS', 'STAMP_PX', 'STAMP_CORE_PX', 'RMS_MAD_FACTOR', 'NBINS', 'PB_TIME_MODEL', 'REF_SNR',
                      'MIN_SCALE_REFS', 'SCALE_REF_KIND', 'V_TAKES_I_SCALE', 'V_SCALE_TOL', 'WCS_TOL_DEG', 'FLOW_LOWRES_PX',
                      'INFIELD_MARGIN_PX', 'ROW_KINDS', 'SAMPLES', 'ADMISSION_MISSING', 'NO_ADMITTED_SCALE',
                      'MISSING_SAMPLE_FILE', 'FRAME_LISTER', 'FRAME_TIME', 'PB_LIBRARY', 'STOKES_PER_PROCESS')

    # ---- tiers ------------------------------------------------------------------------------------------------------
    TIERS = ('snapshot',)                    # tiers with a light-curve product (contract S-3 only; critique_coverage.md:273-275; spec Q1)

    # ---- product definition [W][R] (clv15/phase3_snapshot_lightcurves.py) ---------------------------------------------
    STAMP_PX = 41                            # px, side of the local-rms stamp; LC:43, :124; ledger 1248 IMPLEMENTED-SNAPSHOT-LIGHT-CURVE-PRODUCT-EXTRACTOR-SIX-CELL-TEST
    STAMP_CORE_PX = 5                        # px, side of the central square left out of the stamp (|dx|,|dy| <= 2); LC:125; ledger 1248
    RMS_MAD_FACTOR = 1.4826                  # dimensionless, MAD -> Gaussian sigma; LC:155; ledger 1248
    NBINS = 6                                # count, 10-minute PB bins over the frame span; LC:44, :131; ledger 1231 RULING-SNAPSHOT-TIER-PB-CORRECTION-MODEL-BEAM-ONLY
    PB_TIME_MODEL = 'per-10min-bin'          # PB per bin, nearest library pixel (SS:949-964), not interpolated in time; LC:160-163; contract 3.2 'TO CHANGE' (open; step edges, ledger 1266)
    REF_SNR = 20.0                           # Master peak / island rms, floor for flux-scale references; LC:45, :169; ledger 1248
    MIN_SCALE_REFS = 3                       # count; fewer references -> FLUXSCAL = -1.0; LC:171
    SCALE_REF_KIND = 'master:S'              # row kind of flux-scale references (PyBDSF S_Code 'S'); LC:168; ledger 1262 (refs stay S)
    V_TAKES_I_SCALE = True                   # Stokes V FLUXSCAL/NREFSCAL = the same cell/subband Stokes I values; LC:173-193; ledger 2311 FIX-REFACTOR-M20-M22-M30-M31-M33-20261013
    V_SCALE_TOL = 1e-9                       # absolute, |FLUXSCAL(V) - I scale| allowed by the inline test and content test LC4; LC:239
    WCS_TOL_DEG = 1e-3                       # deg, |CRVAL1|, |CRVAL2| of every frame vs the first frame; LC:139; ledger 1248
    FLOW_LOWRES_PX = 8                       # px per flow-field cell; LC:149-150 (literal 8.0) = RA.LOWRES (RA:68)
    INFIELD_MARGIN_PX = 22                   # px = STAMP_PX // 2 + 2; rows nearer the frame edge are dropped; LC:115-116
    ROW_KINDS = ('master', 'target')         # rows written: every finite Master Gaussian (kind master:<S_Code>) + target samples; LC:56-87; ledgers 1245, 1262
    CHAIN_RUN_DIR = None                     # path or None [R]; production passes no --run-dir (v9:366-370, BF:68-72): served chain rows absent (contract S-3 (a), open, spec Q4); a value changes products (gate G14)
    SAMPLES = (('Hot_Warm_Jupiters', 'Hot_Warm_Jupiters.csv'), ('10pc', '10pc.csv'), ('25pc', '25pc.csv'),
               ('LPTs', 'LPTs.csv'), ('flare_ucd', 'flare_ucd.csv'), ('sydney_stars', 'sydney_stars.csv'),
               ('pulsars_atnf', 'pulsars_atnf.csv'))
                                             # (tag, file under paths.CLV15_ROOT) = CH.SAMPLES (CH:390-393), read by CH.load_samples (CH:1639-1655)

    # ---- fallbacks of today's extractor [W] (batch rows B-23 / B-30 / B-58 open: any other value needs Gregg's ruling) -
    ADMISSION_MISSING = 'admit'              # 'admit' | 'refuse': no admission_<sb>.csv, or a frame absent from it, counts as admitted; LC:96-100, :156 ('refuse' not available in the window: G13)
    NO_ADMITTED_SCALE = 'all-frames'         # 'all-frames' | 'refuse': no admitted frame -> row medians over every frame; LC:166 (counted per pair)
    MISSING_SAMPLE_FILE = 'skip'             # 'skip' | 'refuse': a missing sample file contributes no rows; CH:1643-1645; B-58 (counted per pair)
    FRAME_LISTER = 'legacy-discover-frames'  # 'legacy-discover-frames' (SS:206-300, flow_select=False: latest Run_* per file-name stamp) | 'core-inputs' (core/inputs.phase1_frames); B-15
    FRAME_TIME = 'file-name'                 # 'file-name' | 'date-obs': FRAMES.ts / epoch_s / bin from the file-name stamp; LC:129-131; R-FRAME-TIME-FROM-INTEGRATION-ALL-STAGES (ledger 1752) not applied here; B-16

    # ---- inputs and outputs [P] (production paths today; no renames, DESIGN 11.1) -------------------------------------
    PRODUCT_ROOT = '/lustre/gh/main/exopipe_ops/snapshot_lightcurves'          # <root>/<cell>/<cell>_<sb>_<S>_lightcurves.fits; v9:359, BF:28; chain --lightcurves-root (snapshot_rechain_batch.sh:280)
    MASTER_CATALOG_DIR = '/lustre/gh/main/exopipe_ops/snapshot_master_catalogs' # master_R075_<LST>_<sb>.gaul.csv; v9:369, BF:71; LC:58
    PB_LIBRARY = '/lustre/pipeline/exopipe/phase3/pb_library'                   # PB_<sb>_bin<k>.fits, k = 0..5; SS:74 (a constant: window-locked)
    FLEET_CLAIMS_ROOT = '/lustre/gh/main/exopipe_ops/snapshot_fleet_claims'     # <root>/<cell>/done written by the fleet at cell end; v9:38, :380
    SEARCH_DONE_EVIDENCE = 'fleet-claim-done'  # 'fleet-claim-done' (window) | 'search-record' (after search cutover: <SEARCH_RECORD_ROOT>/_records/search/<cell>.json status done); replaces BF:31-33 (grep of fleet logs for ' DONE <cell> runinfo=')
    SEARCH_RECORD_ROOT = None                # path or None: product root of `exopipe3 search --tier snapshot` whose records G1 reads in 'search-record' mode
    PLANNED_SUBBANDS = 'admission'           # planned = subbands with admission_<sb>.csv (BF:29-30, :47; snapshot_cell_prefit.sh:96-106); the fleet's count used every staged directory (v9:361; defect R10)
    SKIP_FLOW_UNREGISTERED = True            # runinfo_<sb>_I_median.json flow_registered False -> not planned (recorded); snapshot_cell_prefit.sh:99-101, snapshot_batch_audit.py:88-92; ledger 1421 FINDING-SNAP-LIGHTCURVES-ABSENT-FOR-NO-MASTER-SUBBANDS (open)

    # ---- content tests ------------------------------------------------------------------------------------------------
    LC3_TARGET_ROWS_NEED_FLUX = False        # True = spec LC3 as written ('every row has at least one finite FLUX value'); False = master rows only, target/chain rows without
                                             # flux reported: production I tables carry target rows inside the frame rectangle but outside the imaged disc (73 in
                                             # 09h_2024-12-24 36MHz I; 4 of 105 sampled tables; never a master row) - with True the step would refuse tables production ships
    SCALE_RECOMPUTE_RTOL = 1e-6              # relative, FLUXSCAL(I) vs median(median_flux / master_peak) recomputed from the stored float32 columns (spec: measured <= 7e-8 on 10 tables)

    # ---- orchestration (no product effect) ---------------------------------------------------------------------------
    PROCS = 8                                # subbands extracted at a time (each: an I process then a V process); v9:365 (xargs -P 8), BF:12 (NPAR 8)
    STOKES_PER_PROCESS = ('I', 'V')          # order within one subband: the Stokes I process, then the Stokes V process reading the I table (LC:180-192); v9:362-366 runs both concurrently
    WAIT_I_TABLE_S = 0                       # s, exported as P3_LC_WAIT_I_S (LC:183 default 3600 s); 0 because the I table is written first by the same subband thread
    SKIP_V_WITHOUT_I = True                  # no I table -> V refused at once (G10) instead of extracting V then refusing (same outcome; LC:180-187)
    STAGE_COPY_PROCS = 8                     # parallel copies when the step stages itself; v9:112, BF:55 (the stage stage waits for frames --tier snapshot, spec B7)
    STAGE_FUNPACK_PROCS = 16                 # parallel decompressions; v9:113-114, BF:56-57 (same)
    STAGE_MIN_FREE_GB = 300                  # GB free on the node-local stage root before staging one cell; BF:40 (fleet whole-cell guard 800 GB, v9:95, belongs to frames)
    KEEP_STAGE = False                       # remove a self-made stage after the unit; BF:83
    INCOMPLETE_STOPS_CHAIN = False           # a refused curves unit does not stop the cell's later chain steps (today the fleet writes DONE regardless, v9:379-380); B-30 / B-19
    LOG_TAIL_LINES = 40                      # lines of each extractor log kept in the run record (production never ships the extractor logs, v9:336 vs :370)
    STAGE_ROOT_AUTO = True                   # FRAME_SOURCE 'staged', no --stage-root and no 'stage' stage: use the frames snapshot staging of this node, <scratch root>/<frames_snapshot.STAGE_SUBDIR>/<cell> (v9:87); G4 checks it
    FRAME_SOURCE = 'archive'                 # 'archive': the extractor reads the Phase-1 archive frames itself (no --stage; clv15 e0785b7 read_frame,
                                             # bit-identical to staged frames, 10,129/10,129 rows; the production regeneration recipe, exopipe_ops 3035b47) -
                                             # no staging, any node; 'staged': the fleet recipe (v9:366-370), frames staged node-local first (G4).
                                             # --stage-root always means 'staged'.

    RULED = {
        'STAMP_PX': 'IMPLEMENTED-SNAPSHOT-LIGHT-CURVE-PRODUCT-EXTRACTOR-SIX-CELL-TEST',
        'STAMP_CORE_PX': 'IMPLEMENTED-SNAPSHOT-LIGHT-CURVE-PRODUCT-EXTRACTOR-SIX-CELL-TEST',
        'RMS_MAD_FACTOR': 'IMPLEMENTED-SNAPSHOT-LIGHT-CURVE-PRODUCT-EXTRACTOR-SIX-CELL-TEST',
        'REF_SNR': 'IMPLEMENTED-SNAPSHOT-LIGHT-CURVE-PRODUCT-EXTRACTOR-SIX-CELL-TEST',
        'WCS_TOL_DEG': 'IMPLEMENTED-SNAPSHOT-LIGHT-CURVE-PRODUCT-EXTRACTOR-SIX-CELL-TEST',
        'NBINS': 'RULING-SNAPSHOT-TIER-PB-CORRECTION-MODEL-BEAM-ONLY',
        'PB_LIBRARY': 'RULING-SNAPSHOT-TIER-PB-CORRECTION-MODEL-BEAM-ONLY',
        'SCALE_REF_KIND': 'MEASURED-SIX-CELL-LIGHT-CURVE-POPULATION-BURST-AND-MODULATION-STATISTICS-BY-BAND',
        'ROW_KINDS': ('APPROVED-PREAMBLE-C-SIX-CELL-LIGHT-CURVE-PRODUCT-TEST',
                      'MEASURED-SIX-CELL-LIGHT-CURVE-POPULATION-BURST-AND-MODULATION-STATISTICS-BY-BAND'),
        'CHAIN_RUN_DIR': 'APPROVED-PREAMBLE-C-SIX-CELL-LIGHT-CURVE-PRODUCT-TEST',
        'V_TAKES_I_SCALE': 'FIX-REFACTOR-M20-M22-M30-M31-M33-20261013',
        'V_SCALE_TOL': 'FIX-REFACTOR-M20-M22-M30-M31-M33-20261013',
        'PRODUCT_ROOT': 'APPROVED-PREAMBLE-C-SIX-CELL-LIGHT-CURVE-PRODUCT-TEST',
        'FLEET_CLAIMS_ROOT': 'R-SNAP-LIGHTCURVE-BACKFILL',
        'MASTER_CATALOG_DIR': 'IMPLEMENTED-SNAPSHOT-LIGHT-CURVE-PRODUCT-EXTRACTOR-SIX-CELL-TEST',
        'SKIP_FLOW_UNREGISTERED': 'FINDING-SNAP-LIGHTCURVES-ABSENT-FOR-NO-MASTER-SUBBANDS',
    }
