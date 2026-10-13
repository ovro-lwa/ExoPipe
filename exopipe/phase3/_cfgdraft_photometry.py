"""Config draft of `exopipe3 photometry` (refactor window; loaded by config._load_drafts until the refactor chat folds these
sections into config.py at `# @@STEP-SECTIONS@@`). Spec: exopipe_ops/refactor_20261012/plan/specs/photometry.md section 3,
with the refactor chat's decisions (_CROSSCHECK.md top, 2026-10-13 ~04:30 UTC): photometry owns the deep target photometry
(frames, stage, measure, evaluate - forced keeps catflux, the deep panels and the snapshot event panels); interpreters are the
site PYTHON keys 'default' (deep engines) and 'py36' (snapshot engines); shared roots come from config.paths.

Every value: unit in the name or comment, the ruling (ledger ID, clv15/fix_ledger.csv) where one set it, and the production
file:line it describes (exopipe_ops / clv15 HEAD of 2026-10-13). During the window the engines run IN PLACE
(core/legacy.run): a value marked MIRROR is a literal of the in-place engine, read here only by the step's gates and content
tests; the engine keeps its own copy. MIRROR keys and pinned inputs are in PROTECTED_KEYS, so no --set/--config can change
them (an override would change nothing in the products); tests/test_photometry.py compares each MIRROR value with the
engine's literal. At absorption the engine reads config and the protection of its MIRROR keys is lifted.

Shared values are referenced, not copied: paths.CLV15_ROOT (sample lists), paths.COADD_SB_ROOT (deep cell roots, the parent
check's P3_COADD_ROOT), paths.TRANSIENTS_ROOT (searched deep runs), paths.SNAPSHOT_FLEET_OUT (snapshot search outputs: noise
medians, runinfo), paths.PHASE3_ARCHIVE_ROOT (the snapshot photometry cutouts, sub-directory search_snapshot.CUTOUT_SUBDIR);
catalog_snapshot.MASTER_CATALOGS and CURVE_CONSTANTS (zone screen); fitting.* (radio positions).
"""
from exopipe.phase3.config import _Section


class photometry(_Section):
    """exopipe3 photometry - photometry of the input samples at their catalogue positions (deep, snapshot; sub-LST refused):
    samples, bands, stages, output roots, in-place engines and their process environments."""
    # ---- stages per tier, pipeline order (spec 1). Per hour (DESIGN 12.1, --epochs) the deep tier measures the epoch and the
    #      evaluation is a campaign step over a whole measurement (DESIGN 12.2 'photometry campaigns')
    STAGES = {'deep': ('frames', 'stage', 'measure', 'evaluate'), 'snapshot': ('measure', 'evaluate')}
    EPOCH_STAGES = {'deep': ('frames', 'stage', 'measure'), 'snapshot': ('measure',)}
    TIERS_WITH_ENGINE = ('deep', 'snapshot')     # sub-LST: no production engine (ledger 1513/1518 notes; the 19h pilot
                                                 # sublst_forced_phot.py is one-off) - refused, gate PHOT-SUBLST-NO-ENGINE
    # ---- samples (pinned inputs: Gregg's lists, read only). File names under paths.CLV15_ROOT; the engines read the same files
    #      by literal path (deep_forced_phot.py:87-88; phase3_snapshot_cutouts.py:33-35 via snapshot_forced_phot.py:52-57)
    SAMPLE_FILES = {'HWJ': 'Hot_Warm_Jupiters.csv', '10pc': '10pc.csv'}
    SAMPLE_SHA1 = {'HWJ': 'ede4fe33ec4e2686dfff9a0c69b54a8113b1a842',      # sha1 read 2026-10-13: 77 rows (list rebuilt 2026-09-26,
                   '10pc': '63b4dcf71b28bdefb2be0b19e7e5591f90ac89ea'}     # HWJ-LIST-CUT-TO-THE-ELEVATION-RULING-AND-HD-20782-REMOVED); 292 rows
    SAMPLE_COLUMNS = ('common_name', 'ra_current', 'dec_current')         # name, RA deg, Dec deg (current epoch); deep_forced_phot.py:89
    SAMPLE_ENCODING = 'utf-8-sig'                                          # deep_forced_phot.py:88 (10pc.csv carries a BOM)
    SAMPLES_MEASURED = ('HWJ', '10pc')                                     # MIRROR deep_forced_phot.py:87, snapshot_forced_phot.py:45,
                                                                           # deep_forced_eval.py:93 (R-FORCED-PHOTOMETRY-ELEMENTS-APPROVED-V-FAINT-PARENT-ENTRIES)
    # known lists nothing measures today (B-58): file name under paths.CLV15_ROOT, or None when omitted by ruling
    SAMPLES_NOT_MEASURED = {'25pc': '25pc.csv',                            # 4,277 rows; no engine reads it
                            'flare_ucd': 'flare_ucd.csv',                  # 7,626 rows, 13-star pinned supplement (FIX-FLARE-UCD-DROPPED-ROWS)
                            '50pc': None}                                  # omitted (R-P3S2-PAGE-STANDARDS; file renamed 50pc_omit.csv)
    CURRENT_LIST_ONLY = True        # MIRROR: only targets of the CURRENT lists are measured / evaluated (snapshot_forced_phot.py:52-57,
                                    # 135-143; snapshot_forced_eval.py:169-170; 2026-09-28)
    # ---- snapshot bands measured (one engine process per band; snapshot_forced_fleet.sh:19-21). No Full
    #      (INCIDENT-SNAP-FULL-CUTOUTS-EQUAL-RED-62-CELLS); the widebands added by INCIDENT-SNAPSHOT-FORCED-PHOTOMETRY-RAN-WITHOUT-WIDEBAND
    SNAPSHOT_BANDS = ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz',
                      '69MHz', '73MHz', '78MHz', '82MHz', 'Red', 'Green', 'Blue')
    # ---- product roots of production (refused for writing while paths.PRODUCTION_WRITES is False; both under
    #      paths.PRODUCTION_ROOTS). deep: run directories under it (production run1..run3, run_deep_forced_20260928.sh:10);
    #      snapshot: stage1/<cell>/ (snapshot_forced_fleet.sh:14) and eval_<tag>/, eval_wb_<tag>/ (reserve :151-162)
    OUT_ROOT = {'deep': '/fast/gh/main/phase3/deep_forced', 'snapshot': '/lustre/gh/main/phase3/snapshot_forced'}
    # ---- in-place engines (window): file names resolved in the site LEGACY_CODE_ROOTS (exactly one hit, else refused); each
    #      runs with the site interpreter named here (core/legacy.run; decision 2: default = py38_orca_nkosogor,
    #      run_deep_forced_20260928.sh:9; py36 = the development env, snapshot_forced_fleet.sh:13)
    ENGINE = {'frames': 'deep_forced_frames.py', 'measure_deep': 'deep_forced_phot.py', 'evaluate_deep': 'deep_forced_eval.py',
              'measure_snapshot': 'snapshot_forced_phot.py', 'evaluate_snapshot': 'snapshot_forced_eval.py'}
    ENGINE_PYTHON = {'deep': 'default', 'snapshot': 'py36'}
    # process environment added to core/legacy.run's (which already clears inherited P3_* and sets runner.ENV_DEFAULTS, i.e.
    # NUMPY_MADVISE_HUGEPAGE=0, snapshot_forced_fleet.sh:10). deep measure: production pinned every shard to ONE core
    # (run_deep_forced_20260928.sh:33 taskset -c <one core>), so each ran one BLAS/OpenMP thread; the thread caps reproduce that
    # without pinning. snapshot evaluate: OMP_NUM_THREADS=1 (snapshot_forced_reserve_20261007.sh:156)
    ENV = {'deep_frames': {}, 'deep_measure': {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
           'deep_evaluate': {}, 'snapshot_measure': {}, 'snapshot_evaluate': {'OMP_NUM_THREADS': '1'}}
    # caller-environment prefixes refused before any engine starts (gate PHOT-ENV): switches the in-place modules or their
    # launchers read (spec 7.2) that core/legacy.run does not clear (it clears P3_* only)
    ENV_FORBIDDEN_PREFIXES = ('P3S', 'DEEP_FORCED_', 'SF_', 'FORCED_TEST_', 'SUBLST_')
    LOG_TAIL_LINES = 40             # engine log lines kept in a refused unit's reason / record (orchestration only)
    PROTECTED_KEYS = ('SAMPLE_FILES', 'SAMPLE_SHA1', 'SAMPLE_COLUMNS', 'SAMPLE_ENCODING', 'SAMPLES_MEASURED',
                      'SAMPLES_NOT_MEASURED', 'CURRENT_LIST_ONLY', 'OUT_ROOT', 'ENGINE', 'ENV_FORBIDDEN_PREFIXES',
                      'TIERS_WITH_ENGINE')
    RULED = {'SAMPLES_MEASURED': 'R-FORCED-PHOTOMETRY-ELEMENTS-APPROVED-V-FAINT-PARENT-ENTRIES',
             'SAMPLE_FILES': 'HWJ-LIST-CUT-TO-THE-ELEVATION-RULING-AND-HD-20782-REMOVED',
             'SNAPSHOT_BANDS': ('INCIDENT-SNAP-FULL-CUTOUTS-EQUAL-RED-62-CELLS',
                                'INCIDENT-SNAPSHOT-FORCED-PHOTOMETRY-RAN-WITHOUT-WIDEBAND'),
             'SAMPLES_NOT_MEASURED': ('R-P3S2-PAGE-STANDARDS', 'FIX-FLARE-UCD-DROPPED-ROWS')}


class photometry_frames(_Section):
    """Deep stages frames (deep_forced_frames.py in place, on the host that reads the cell roots) and stage (port of
    run_deep_forced_20260928.sh:14-31: node-local copies of the inputs the measuring host cannot read)."""
    SRC_RUN = None                  # the SEARCHED deep run directory: required, no default (DEEP_FORCED_SRC_RUN,
                                    # deep_forced_frames.py:20; D7 fix exopipe_ops a8b8a9c); --src-run DIR or --run LABEL
    SRC_RUN_DIR = 'Run_{label}'     # --run LABEL -> <paths.TRANSIENTS_ROOT>/Run_<LABEL> (the path deep_forced_frames.py:20 names)
    PARENT_DIR = None               # the Robust -0.75 parent catalogue directory the deep table pass used: required, no default
                                    # (gate_inputs.py:19 P3_PARENT_DIR falls back to the stale Run_P3S4/parent_catalogs); --parent-dir
    ROOTS_CSV = '/lustre/gh/main/exopipe_ops/s2search_state/roots.csv'   # pinned input, 456 cells; MIRROR deep_forced_frames.py:21
    FRAME_SUBDIR = {'I': 'iresid', 'V': 'vresid'}                        # MIRROR deep_forced_frames.py:38
    FRAME_NAME_RE = r'^(\d{4}-\d\d-\d\d)_([IV])_redo_resid_(\w+)\.fits$'  # MIRROR deep_forced_frames.py:22 (pbcorr siblings are not frames)
    PB_NAME = {'I': 'PB_deep_R075_{band}.fits', 'V': 'PB_deep_{band}.fits'}   # one PB map per weighting (R-FLUX-SCALE-REMEASURE-
                                                                             # REBAKE-AND-PB-STEP2-RUNNING); MIRROR :29, :49
    WEIGHTS_NAME = 'weights.csv'                                         # MIRROR deep_forced_frames.py:29
    PARENT_NAME = 'master_R075_{lst}_{band}.gaul.csv'                    # MIRROR deep_forced_frames.py:32; gate_inputs.py:44
    MASTER_NAME = 'I_deep_Taper_R075_{band}.fits'                        # the master the parent's .gen.json must name; gate_inputs.py:45
    FRAME_LIST_COLUMNS = ('lst', 'band', 'date', 'stokes', 'path', 'pb_path', 'parent_path', 'weights_path', 'thermnoi')  # :52-53
    FRAMES_FILE = 'frames_07.csv'                # run_deep_forced_20260928.sh:12 (production name kept in the window)
    STAGED_FRAMES_FILE = 'frames_09.csv'         # run_deep_forced_20260928.sh:23
    STAGE_LIST_FILE = 'stage_list.txt'           # run_deep_forced_20260928.sh:22
    STAGE_COLUMNS = ('path', 'pb_path', 'weights_path', 'parent_path')   # run_deep_forced_20260928.sh:19
    STAGE_PREFIX_ROOTS = ('/data07',)            # inputs under these roots are staged when this host cannot read them (:20)
    STAGE_SUBDIR = 'photometry_deep_stage'       # under --work or the site SCRATCH_ROOT (production /fast/gh/main/phase3/deep_forced/stage)
    RSYNC = ('rsync', '-aL')                     # run_deep_forced_20260928.sh:27 (rsync -aL --files-from=<list> SRC DST)
    REFUSED_RE = r'^\s*REFUSED (\S+): (.*)$'     # the engine's per-cell refusal lines on stderr (deep_forced_frames.py:59-60)
    # content test sampling (photometry_frame_list): frames whose existence FL2 checks, cells whose master digest FL4
    # recomputes, V rows whose THERMNOI FL5 re-reads (orchestration only)
    FL2_EXISTENCE_ROWS = 500
    FL4_DIGEST_CELLS = 4
    FL5_THERMNOI_ROWS = 50
    PROTECTED_KEYS = ('ROOTS_CSV', 'FRAME_SUBDIR', 'FRAME_NAME_RE', 'PB_NAME', 'WEIGHTS_NAME', 'PARENT_NAME', 'MASTER_NAME',
                      'FRAME_LIST_COLUMNS', 'STAGE_COLUMNS')
    RULED = {'PB_NAME': 'R-FLUX-SCALE-REMEASURE-REBAKE-AND-PB-STEP2-RUNNING',
             'SRC_RUN': 'FIX-REFACTOR-DEFECTS-D2-D6-D7-D9-D10-AND-RULINGS-D1-D5-D8-20261013'}


class photometry_measure(_Section):
    """Measurement at the target and its ring controls (deep_forced_phot.py, snapshot_forced_phot.py in place). MIRROR keys
    describe the engines' literals (PROTECTED); SHARDS, BAND_PROCS, CUTOUT_SUBDIR and WORK_SUBDIR steer the step."""
    # ---- both tiers (identical values today; R-FORCED-PHOTOMETRY-ELEMENTS-APPROVED-V-FAINT-PARENT-ENTRIES = ledger 1513)
    N_CONTROLS = 12                 # MIRROR deep_forced_phot.py:58 (N_CTRL), snapshot_forced_phot.py:47 (NCTL)
    RING_BEAMS = 4.0                # beam FWHM; MIRROR deep :58 (R_CTRL_BEAMS), snapshot :48; snapshot falls back to the largest
                                    # radius 5 px inside the cutout (R-SNAPSHOT-STAT-ERROR-0.42-FLOOR-5-AND-OTHER-RULINGS)
    AVOID_BEAMS = 2.0               # beam FWHM; a control this close to a Master source > AVOID_JY is not used; MIRROR deep :58, snap :49
    AVOID_JY = 1.0                  # Jy/beam (Master peak); MIRROR deep :58, snapshot :49
    STAMP_PX = 41                   # pixels, rms stamp side; MIRROR deep :112 (d[yi-20:yi+21]), snapshot :46
    STAMP_CORE_PX = 5               # pixels, central square left out of the rms; MIRROR deep :112, snapshot :86
    RMS_MAD_FACTOR = 1.4826         # robust rms = factor x MAD; MIRROR deep :113, snapshot :103
    # ---- per tier where they differ
    EDGE_PX = {'deep': 25, 'snapshot': 5}          # px: deep target > 25 px inside the frame (:106); snapshot ring >= 5 px inside (:76)
    STAMP_MIN_PIXELS = {'deep': 200}               # deep: > 200 finite non-zero stamp pixels, else rms NaN; MIRROR deep :113
    MIN_STAMP_FRAC = {'snapshot': 0.5}             # snapshot: a partial stamp is used when >= 50 % lies inside the cutout; MIRROR :75
    MIN_ELEVATION_DEG = {'deep': {'I': 20.0, 'V': 19.5}}   # deg at the frame's MJD-MID; MIRROR deep_forced_phot.py:57
                                                   # (ELEVATION-RULING-KEEP-20-SUPERSEDES-B02; B-06/B-40 unanswered: today's values)
    SHARDS = {'deep': 32}                          # shards of the deep measurement (run_deep_forced_20260928.sh:33; no ruling - the
                                                   # launcher's choice); another value changes the blank-sky rows (the generator is
                                                   # seeded per shard, below), never a target or control row; --shards overrides
    RNG_SEED_BASE = {'deep': 20260928}             # blank-sky generator seed = base + shard; MIRROR deep :212
    N_BLANK = {'deep': 200}                        # blank-sky positions per admitted frame (4 x drawn, first 200 kept); MIRROR :58, :312-316
    BLANK_EDGE_PX = {'deep': 30}                   # px; MIRROR deep :312
    BLANK_SNR_COUNTS = {'deep': (3, 4, 5, 6, 7)}   # |S/N| levels counted in blank_NNN.csv; MIRROR deep :319
    FIT_SNR = {'deep': 5.0}                        # Gaussian fit (size/axis) only at |S/N| >= 5 (B-01); MIRROR deep :119
    FIT_FAILED_FLAG_SNR = {'deep': 7.0}            # 'fit-failed' written in gate_size at >= 7 sigma; MIRROR deep :124
    SIZE_GATE = {'deep': ((7.0, 8.0, 1.7), (8.0, float('inf'), 1.5))}   # (S/N from, to, limit in beams): flag, never a removal;
                                                   # R-FORCED-PHOTOMETRY-GATE-CALLS-20260926, R-FORCED-SIZEGATE-KEEP-MEASURE-AND-PANEL-
                                                   # JUDGEMENTS; MIRROR deep :127-129
    INJECT_SIGMA = {'deep': (5.0, 10.0)}           # x local rms, beam-shaped source added at the target in a local copy; MIRROR :276
    INJECT_HALF_PX = {'deep': 30}                  # px, half-size of the injection cutout (61 x 61); MIRROR deep :277-279
    PB_FLOOR = {'deep': 0.1}                       # flux_pbcorr written only where PB > 0.1; MIRROR deep :290
    ATEAM_SOURCES = {'deep': ('23h23m24s +58d48m54s', '19h59m28s +40d44m02s', '05h34m31s +22d00m52s',
                              '12h30m49s +12d23m28s', '09h18m05.7s -12d05m44s', '16h51m08.15s +04d59m33.3s')}
                                                   # Cas A, Cyg A, Tau A, Vir A, Hyd A, Her A; MIRROR deep :91-92 (B-42 unanswered)
    ATEAM_RADIUS_DEG = {'deep': {'I': 5.0, 'V': 3.0}}   # deg, flag radius (10.1-ratify, Card 6); MIRROR deep :59, deep_forced_eval.py:40
    MOON_RADIUS_DEG = {'deep': 0.25}               # deg; MIRROR deep :59, deep_forced_eval.py:40
    JUPITER_RADIUS_DEG = {'deep': {'Red': 5.0, 'Full': 5.0, 'Green': 0.5, 'Blue': 0.5}}   # deg by band class
                                                   # (R-JUPITER-TRACK-VETO-IMPLEMENTED-IN-THE-CONFORMANCE-STAGE-BOTH-TIERS); MIRROR :62, eval :43
    JUPITER_TRACK = {'deep': {'step_h': 0.25, 'min_el_deg': 20.0, 'lst_centre_offset_h': 0.5}}   # per co-added night (in_master
                                                   # 1): Jupiter at the cell's mid-LST on a 15-min grid above 20 deg; MIRROR :236-243
    STRUCT_RADIUS_BEAMS = {'deep': {'I': ((3.0, 3.5), (1.0, 3.0)),
                                    'V': ((30.0, 6.0), (10.0, 5.0), (3.0, 4.0), (1.0, 2.5))}}   # (Master peak Jy, radius in frame
                                                   # beams), brightest first; screen radius = max(table zone, this)
                                                   # (R-FORCED-SCREEN-STRUCTURE-RADIUS-IMPLEMENTED-RUN3); MIRROR deep :140
    PARENT_TABLES = {'deep': '/lustre/gh/main/exopipe/exopipe/common/parent_curve_tables.json'}   # pinned input (production copy,
                                                   # sha1 0df0f592 = the worktree copy 2026-10-13); zone tables through
                                                   # parent_curve_table.table_for (R-DEEP-V-FAINT-PARENT-TABLE-ENTRIES-DROPPED); MIRROR :64
    SOLAR_AUDIT = {'deep': '/lustre/pipeline/exopipe/phase3/QA/solar_contamination_audit.csv'}   # pinned input: QUARANTINE
                                                   # band-nights of the twilight refusal (M18 kept); MIRROR deep :66-69
    DAY_SAMPLES = {'deep': 5}                      # Sun tested at 5 instants DATE-OBS..DATE-END (R-DAYTIME-GATE-DEEP-CONFORM); MIRROR :185
    TWILIGHT_SUN_DEG = {'deep': {'rising': -12.0, 'setting': -18.0}}   # deg; twilight refusal only in a QUARANTINE band
                                                   # (R-NIGHT-RULE-HORIZON-PLUS-SOLAR-AUDIT); MIRROR deep :187, :192
    NOISE_GATE_X = {'deep': 3.0}                   # frame refused when its I qa3_rms_mjy / V THERMNOI > 3 x the cell median; MIRROR :232, :234
    NOISE_GATE_MIN_NIGHTS = {'deep': 5}            # the noise gate applies only with >= 5 values; MIRROR deep :232, :234
    SITE = {'deep': (37.2398, -118.2817, 1222.0)}  # lat deg, lon deg, height m of the deep engine (B-42 unanswered); MIRROR :55-56
    ADMISSION_REASONS = {'deep': (r'^excluded epoch \(data_decisions\)$', r'^daytime: Sun ',
                                  r'^twilight in a solar-audit QUARANTINE band ', r'^3x per-Stokes noise gate$')}
                                                   # the refusal texts of frames_NNN.csv; MIRROR deep :252-253, :191, :193
    # ---- snapshot only
    CUTOUT_SUBDIR = 'photometry'                   # cutout root = paths.PHASE3_ARCHIVE_ROOT/<this> (= snapshot_forced_phot.py:43 ROOT,
                                                   # phase3_snapshot_cutouts.py:32; search_snapshot.CUTOUT_SUBDIR); --cutout-root overrides
    BLOCK_MIN_FRAMES = {'snapshot': 3}             # admitted frames per wall-clock minute for a 1-minute block (IMPLEMENTED-SNAPSHOT-
                                                   # FORCED-PHOTOMETRY-STAGES-1-2-HWJ-10PC); MIRROR snapshot_forced_phot.py:187
    MINUTE_KEY = {'snapshot': 'file-name'}         # 1-minute blocks keyed by the file-name minute ts[:13] (phase3_snapshot_search.py
                                                   # :114-116); frame-time error accepted (IMPLEMENTED-FRAME-TIME-MINIMAL-PHASE3-AND-
                                                   # Q29-INSTALLED-20261007 'NOT installed: minute_key'; B-16 unanswered); MIRROR :60-62
    BEAM_SOURCE = {'snapshot': 'chain-table'}      # beam = phase3_snapshot_astro.beam_fwhm_arcmin, not the cutout header (finding
                                                   # 2026-09-28; B-29 unanswered); MIRROR snapshot_forced_phot.py:159-163
    BAND_PROCS = {'snapshot': 8}                   # band processes at once inside a cell (snapshot_forced_fleet.sh:12 PAR default;
                                                   # the per-cell prefit passes 4, snapshot_cell_prefit.sh:119); orchestration only
    WORK_SUBDIR = {'snapshot': 'photometry_snapshot_work'}   # under --work or the site SCRATCH_ROOT (production
                                                   # /fast/gh/main/phase3/snapshot_forced_work, snapshot_forced_fleet.sh:15)
    MANIFEST_NAME = 'forced_manifest.csv'          # merged manifest of a cell (snapshot_forced_fleet.sh:22 SF_MANIFEST default)
    MANIFEST_HEADER = 'file,status,ring_beams,n_ctl'   # snapshot_forced_fleet.sh:38 (header LF, body rows as the engine wrote them)
    # ---- content-test sampling (orchestration only; the tests recompute these rows / frames with code of their own)
    T4_FRAMES = 3                                  # photometry_table T4: frames per shard file re-measured
    T4_ROWS_PER_FRAME = 7                          # photometry_table T4: target rows per frame re-measured
    N4_TARGETS = 2                                 # photometry_snapshot_npz N4: targets per npz re-measured from the cutout
    N4_FRAMES = 5                                  # photometry_snapshot_npz N4: frames per target
    PROTECTED_KEYS = ('N_CONTROLS', 'RING_BEAMS', 'AVOID_BEAMS', 'AVOID_JY', 'STAMP_PX', 'STAMP_CORE_PX', 'RMS_MAD_FACTOR',
                      'EDGE_PX', 'STAMP_MIN_PIXELS', 'MIN_STAMP_FRAC', 'MIN_ELEVATION_DEG', 'RNG_SEED_BASE', 'N_BLANK',
                      'BLANK_EDGE_PX', 'BLANK_SNR_COUNTS', 'FIT_SNR', 'FIT_FAILED_FLAG_SNR', 'SIZE_GATE', 'INJECT_SIGMA',
                      'INJECT_HALF_PX', 'PB_FLOOR', 'ATEAM_SOURCES', 'ATEAM_RADIUS_DEG', 'MOON_RADIUS_DEG',
                      'JUPITER_RADIUS_DEG', 'JUPITER_TRACK', 'STRUCT_RADIUS_BEAMS', 'PARENT_TABLES', 'SOLAR_AUDIT',
                      'DAY_SAMPLES', 'TWILIGHT_SUN_DEG', 'NOISE_GATE_X', 'NOISE_GATE_MIN_NIGHTS', 'SITE',
                      'ADMISSION_REASONS', 'BLOCK_MIN_FRAMES', 'MINUTE_KEY', 'BEAM_SOURCE', 'MANIFEST_HEADER')
    RULED = {'N_CONTROLS': 'R-FORCED-PHOTOMETRY-ELEMENTS-APPROVED-V-FAINT-PARENT-ENTRIES',
             'RING_BEAMS': ('R-FORCED-PHOTOMETRY-ELEMENTS-APPROVED-V-FAINT-PARENT-ENTRIES',
                            'R-SNAPSHOT-STAT-ERROR-0.42-FLOOR-5-AND-OTHER-RULINGS'),
             'MIN_ELEVATION_DEG': 'ELEVATION-RULING-KEEP-20-SUPERSEDES-B02',
             'SIZE_GATE': ('R-FORCED-PHOTOMETRY-GATE-CALLS-20260926', 'R-FORCED-SIZEGATE-KEEP-MEASURE-AND-PANEL-JUDGEMENTS'),
             'STRUCT_RADIUS_BEAMS': 'R-FORCED-SCREEN-STRUCTURE-RADIUS-IMPLEMENTED-RUN3',
             'DAY_SAMPLES': 'R-DAYTIME-GATE-DEEP-CONFORM',
             'TWILIGHT_SUN_DEG': 'R-NIGHT-RULE-HORIZON-PLUS-SOLAR-AUDIT',
             'BLOCK_MIN_FRAMES': 'IMPLEMENTED-SNAPSHOT-FORCED-PHOTOMETRY-STAGES-1-2-HWJ-10PC',
             'MINUTE_KEY': 'IMPLEMENTED-FRAME-TIME-MINIMAL-PHASE3-AND-Q29-INSTALLED-20261007'}


class photometry_evaluate(_Section):
    """Evaluation of a measurement (deep_forced_eval.py, snapshot_forced_eval.py in place). MIRROR keys describe the engines'
    literals (PROTECTED); SURVEY_CELLS, KINDS, EVAL_PROCS and REFUSED_CELLS steer the step."""
    THRESHOLD_GRID = {'deep': (3.0, 9.0, 0.25),            # S/N from, to, step; MIRROR deep_forced_eval.py:39
                      'snapshot': (3.0, 59.95, 0.05)}      # IMPLEMENTED-FORCED-THRESHOLD-GRID-TO-SN-59.95-AND-NAN-NOTE-Q5; MIRROR :64
    HIST_EDGES = {'snapshot': (-30.0, 60.0, 0.01)}         # S/N histogram, values beyond it in the end bins; MIRROR :60, :74-78
    LIST_SNR = 4.0                                         # targets_ge4: |S/N| (I signed in snapshot) >= 4; MIRROR deep_forced_eval.py:115,
                                                           # snapshot_forced_eval.py:193
    THRESHOLD_METHOD = 'control-one-false-alarm'           # R-DEEP-FORCED-THRESHOLDS-ADOPTED (values re-derived per run;
                                                           # R-GREGG-20261011-KEEP-8767-THRESHOLDS-AT-RERUN)
    COMPLETENESS_T = {'deep': (4.0, 4.5, 5.0, 5.5, 6.0, 7.0), 'snapshot': (4.5, 5.0, 5.5, 6.0)}   # S/N; MIRROR deep :130, snap :288
    COMPLETENESS_A = {'deep': (4, 5, 6, 7, 8, 10), 'snapshot': (5, 6, 7, 8)}                       # sigma; MIRROR deep :132, snap :290
    SURVEY_CELLS = {'snapshot': 402}                       # cells of the snapshot survey (survey factor); snapshot_forced_eval.py:208
    KINDS = {'snapshot': ('sb', 'wb')}                     # subbands and widebands evaluated separately, never pooled
                                                           # (INCIDENT-SNAPSHOT-FORCED-PHOTOMETRY-RAN-WITHOUT-WIDEBAND); --kind :209
    EVAL_PROCS = {'snapshot': 1}                           # --nproc; outputs identical at any value (ledger 1987; production 24 / 12)
    EVAL_DIR = {'sb': 'eval_{tag}', 'wb': 'eval_wb_{tag}'}   # production names (eval_20261007, eval_wb_20261007)
    REFUSED_CELLS = {'snapshot': '/lustre/gh/main/exopipe_ops/snapshot_forced_refused_cells_20261007.csv'}   # pinned input
                                                           # 'cell,kinds,reason', 4 rows (DECISION-FORCED-CORRUPT-CUTOUT-CELLS-REFUSED)
    REFUSED_FILE = 'refused_cells_{kind}_{tag}.csv'        # per-kind 'cell,reason' list passed to the engine (reserve :138-147 names)
    ZONE_FLAGS_FILE = 'zones_recomputed_{tag}.csv'         # the Z1 recomputation of both kinds (reserve :159, --write)
    DEEP_SEARCH_PIN = {'snapshot': '/lustre/gh/main/phase3/deep_search_refusals/Run_P3S5_pin_20261008T025911Z/deep_search_pin.csv'}
                                                           # pinned input of tier rule 3 (RULINGS-GREGG-20261008-TIER3-FREEZE-B2-
                                                           # RECHAIN-LABELS); MIRROR phase3_tier_rules.py:37
    THRESHOLD_ROWS = {'deep': 4, 'snapshot': 8}            # rows of thresholds.csv: sample x Stokes (x tier for snapshot);
                                                           # deep_forced_eval.py:93-94; snapshot_forced_reserve_20261007.sh:161
    SNAPSHOT_FWHM_ARCMIN = {18: 44.9, 27: 29.9, 36: 22.4, 46: 17.6, 55: 16.3, 64: 13.6, 73: 11.9, 82: 10.6}   # arcmin at MHz,
                                                           # interpolated; MIRROR phase3_snapshot_astro.py:50 (FWHM_ARCMIN)
    SNAPSHOT_BAND_REP_MHZ = {'Red': 27, 'Green': 50, 'Blue': 73}   # MIRROR phase3_snapshot_astro.py:52 (BAND_REP_FREQ)
    SNAPSHOT_SITE_LAT_DEG = 37.2398                        # deg, the zone metric's phase centre; MIRROR phase3_snapshot_astro.py:54
    PNG_DPI = {'deep': 110}                                # exceedance.png (display product); MIRROR deep_forced_eval.py:185
    CONTENT_TEST_PROCS = 8                                 # processes of the Z1 / S content tests of a snapshot evaluation
                                                           # (production 24, reserve :170); orchestration only
    PROTECTED_KEYS = ('THRESHOLD_GRID', 'HIST_EDGES', 'LIST_SNR', 'THRESHOLD_METHOD', 'COMPLETENESS_T', 'COMPLETENESS_A',
                      'REFUSED_CELLS', 'DEEP_SEARCH_PIN', 'THRESHOLD_ROWS', 'SNAPSHOT_FWHM_ARCMIN',
                      'SNAPSHOT_BAND_REP_MHZ', 'SNAPSHOT_SITE_LAT_DEG', 'PNG_DPI')
    RULED = {'THRESHOLD_GRID': 'IMPLEMENTED-FORCED-THRESHOLD-GRID-TO-SN-59.95-AND-NAN-NOTE-Q5',
             'THRESHOLD_METHOD': ('R-DEEP-FORCED-THRESHOLDS-ADOPTED', 'R-GREGG-20261011-KEEP-8767-THRESHOLDS-AT-RERUN'),
             'KINDS': 'INCIDENT-SNAPSHOT-FORCED-PHOTOMETRY-RAN-WITHOUT-WIDEBAND',
             'REFUSED_CELLS': 'DECISION-FORCED-CORRUPT-CUTOUT-CELLS-REFUSED',
             'DEEP_SEARCH_PIN': 'RULINGS-GREGG-20261008-TIER3-FREEZE-B2-RECHAIN-LABELS',
             'SURVEY_CELLS': 'IMPLEMENTED-SNAPSHOT-FORCED-PHOTOMETRY-STAGES-1-2-HWJ-10PC'}


class photometry_cutouts(_Section):
    """The snapshot photometry cutout product: WRITTEN by `search --tier snapshot` (phase3_snapshot_cutouts.write_cutouts, in
    place; search_snapshot.CUTOUTS_ON / CUTOUT_SUBDIR steer it), content test owned here (class photometry_cutout). Every key
    is a MIRROR of phase3_snapshot_cutouts.py (REQUEST-PHOTOMETRY-CUTOUT-PRODUCT-HWJ-10PC-SOLARSYSTEM,
    IMPLEMENTED-PHOTOMETRY-CUTOUT-PRODUCT)."""
    CUT_DEG = 5.0                      # deg, cutout side (R-CUTOUT-5DEG-RERULED); MIRROR phase3_snapshot_cutouts.py:30
    MIN_EL_DEG = 20.0                  # deg at the cell's mid-time (the search domain is 30); MIRROR :31
    PLANETS = ('jupiter', 'saturn', 'uranus', 'neptune')   # SolarSystem sample at per-frame ephemeris positions; MIRROR :34
    PLANET_EPHEM_QUANTUM_MIN = 10      # minutes; ephemeris quantised to the frame's 10-minute key; MIRROR :61
    PLANET_SITE = (37.2398, -118.2817, 1183.0)   # lat deg, lon deg, height m of the planets (B-42 unanswered); MIRROR :66
    TIMES_COLUMNS = ('ts', 'date_obs', 'admitted', 'global_rms_jyb', 'ra_deg', 'dec_deg', 'target_x', 'target_y')   # :160-167
    HEADER_CARDS = ('OBJECT', 'SAMPLE', 'LST', 'DATE', 'BAND', 'STOKES', 'CELL', 'UTCSTART', 'TARGRA', 'TARGDEC', 'CUTDEG')  # :150-153
    PROTECTED_KEYS = ('CUT_DEG', 'MIN_EL_DEG', 'PLANETS', 'PLANET_EPHEM_QUANTUM_MIN', 'PLANET_SITE', 'TIMES_COLUMNS',
                      'HEADER_CARDS')
    RULED = {'CUT_DEG': 'R-CUTOUT-5DEG-RERULED', 'PLANETS': 'REQUEST-PHOTOMETRY-CUTOUT-PRODUCT-HWJ-10PC-SOLARSYSTEM'}


class photometry_phase2(_Section):
    """The DORMANT Phase 2 sample photometry and Phase 2 flux-scale QA (lib/photometry/cutout.py, solar_system_cutout.py,
    flux_check_cutout.py after the pure git mv 4bccd14). Phase 1/2 import them; exopipe3 does not run them in the window
    (B-33 unanswered: today's code kept). MIRROR keys of those modules, PROTECTED until their edit commit."""
    DETECTION_SIGMA = 5.0              # free-peak detection routing (B-03 interim); MIRROR cutout.py:135, solar_system_cutout.py:90
    MIN_ELEVATION_DEG = 30.0           # deg at DATE-OBS + DURATION/2 (no ruling found; B-40); MIRROR cutout.py:136
    FIT_SIGMA_THRESHOLD = 5.0          # imfit only at peak >= 5 sigma (B-01); MIRROR cutout.py:142
    CUTOUT_DEG = 2.0                   # deg; MIRROR cutout.py:141, flux_check_cutout.py:30
    DEFAULT_BEAM_ARCSEC = 300.0        # arcsec when BMAJ is absent; MIRROR cutout.py:140
    SOLAR_SYSTEM_BODIES = ('moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune')   # MIRROR
                                       # solar_system_cutout.py:84-87
    EPHEMERIS = 'de432s'               # first of the import-time chain de432s -> jpl -> builtin; MIRROR solar_system_cutout.py:43-53
    WIDEBAND_NOMINAL_MHZ = {'Red': 30.0, 'Green': 52.0, 'Blue': 74.0, 'Full': 51.5}   # MHz written for wideband rows; MIRROR
                                       # solar_system_cutout.py:622
    FLUXCHECK_SEARCH_RADIUS_DEG = 0.25 # deg; MIRROR flux_check_cutout.py:31
    FLUXCHECK_MIN_ELEVATION_DEG = 20.0 # deg at DATE-OBS (start); MIRROR flux_check_cutout.py:32
    FLUXCHECK_SITE = (37.2398, -118.2817, 1200.0)   # lat deg, lon deg, height m; MIRROR flux_check_cutout.py:33 (B-42)
    FLUXCHECK_PIVOT_MHZ = 150.0        # MHz; MIRROR flux_check_cutout.py:51
    # The SH calibrator coefficients (flux_check_cutout.py:39-46) stay in the module: they differ from
    # pipeline_config.CALIB_DATA for 5 of 6 calibrators (inventory pkg-misc 4.8); B-33 decides. Not copied here.
    PROTECTED_KEYS = ('DETECTION_SIGMA', 'MIN_ELEVATION_DEG', 'FIT_SIGMA_THRESHOLD', 'CUTOUT_DEG', 'DEFAULT_BEAM_ARCSEC',
                      'SOLAR_SYSTEM_BODIES', 'EPHEMERIS', 'WIDEBAND_NOMINAL_MHZ', 'FLUXCHECK_SEARCH_RADIUS_DEG',
                      'FLUXCHECK_MIN_ELEVATION_DEG', 'FLUXCHECK_SITE', 'FLUXCHECK_PIVOT_MHZ')
    RULED = {'FIT_SIGMA_THRESHOLD': 'B-01', 'DETECTION_SIGMA': 'B-03'}
