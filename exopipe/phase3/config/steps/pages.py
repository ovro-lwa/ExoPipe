"""Configuration of `exopipe3 pages` - every tunable of the command, one section per stage or tier.

Units folded here: pages. A key in its section's PROTECTED_KEYS is locked
(a mirror of in-place code, a pinned input or a ruling-fixed value: it changes only by a commit); RULED names the
ledger ID of the ruling behind a key. Per run: --set SECTION.KEY=VALUE or --config FILE; `exopipe3 config --step pages` lists the sections.
"""
from exopipe.phase3.config import _Section


# =====================================================================================================================
# unit pages (was exopipe/phase3/_cfgdraft_pages.py)
# Config draft of `exopipe3 pages` (refactor window; config.py loads it until the refactor chat folds it in).
#
# Spec: /lustre/gh/main/exopipe_ops/refactor_20261012/plan/specs/pages.md section 3, adapted to the binding decisions at the top
# of plan/specs/_CROSSCHECK.md (2026-10-13 ~04:30 UTC): sections are `pages` and `pages_<tier>` only; deep combined localization
# is catalog's (catalog_localize) - pages reads the tables and keeps the figures; pages owns the review_dir merge and its ONE
# config key (pages.REVIEW_DIR_MERGE; serve reads the merged files); shared roots come from CFG.paths; interpreters are the site
# PYTHON keys default / py36 / tess. Source lines checked against the files on disk 2026-10-13 ~04:00-04:40 UTC (exopipe_ops
# e46bd45, clv15 9dbbbb3). Abbreviations: build_i14 = ops/build_i14_pages_std7.py, tenmin = ops/build_tenmin_pages.py,
# snap_pages = clv15/phase3_snapshot_pages.py; "ledger N" = line N of clv15/fix_ledger.csv.
#
# PROTECTED_KEYS holds (a) product names frozen in the refactor window and pinned roots, and (b) MIRROR keys: constants that
# live INSIDE the in-place production modules (an override would not reach the module). Before a stage runs a module, the step
# reads every mirror constant from the module's SOURCE (lib/pages/mirror.py ANCHORS; nothing imported) and refuses on any
# difference (gate P3). The mirror keys become live parameters when the modules are absorbed into lib/pages.
# =====================================================================================================================

_PAGES_MIRROR = ('PNG_DPI', 'PANEL_CUT_DEG', 'PANEL_OBS_LAT_DEG', 'HOURS_EL_MIN_DEG', 'PANEL_TEXT_WRAP_CHARS',
                 'FORCED_READING', 'FRAME_CACHE', 'PREDICTED_PANELS', 'PRED_V_MATCH_ARCMIN', 'PRED_I_PARENT_MIN_JY',
                 'PRED_I_MIN_SEP_BEAMS', 'PRED_I_STAMPS', 'PRED_I_STAMP_HALF_BEAMS', 'PRED_I_QA_DISK_BEAMS', 'PRED_I_QA_R_MIN',
                 'PRED_I_PLACE_MIN_JY', 'I_TEMPLATE_NULLS', 'I_TEMPLATE_NULL_PCTL', 'I_TEMPLATE_MIN_NULLS',
                 'I_TEMPLATE_MARGINAL_FACTOR', 'I_TEMPLATE_SEED', 'XMATCH_RADIUS_ARCMIN', 'XMATCH_SAMPLES',
                 'XMATCH_SAMPLE_DIR', 'SKY_AREA_DEG2', 'VHF_ONLY', 'SUBLST_FIT_FLUX_SNR', 'PB_LABEL')


class pages(_Section):
    """exopipe3 pages - review pages of every tier (B16 [49]; contract steps 43, 49, 81; snapshot SC 5.2): stages, page sets,
    processes, page layout (M14), the review_dir merge (owned here; serve reads), record chain, and the renderer constants
    that live inside the in-place modules (mirror keys)."""
    # REQUIRE_CATALOG_RECORD is protected (review 2026-10-13): P2 across steps is not implemented in the window (production
    # catalogues carry no record), so an override to True would be a silent no-op - a key no override may pretend to set
    PROTECTED_KEYS = ('RUN_DIR', 'QUEUE_NAMES', 'INDEX_NAMES', 'PASS_T0_REQUIRED', 'REQUIRE_CATALOG_RECORD') + _PAGES_MIRROR
    # ---- what runs ------------------------------------------------------------------------------------------------
    # deep: catalog --stages localize writes the localization tables BEFORE pages (crosscheck decision 6, group 6)
    STAGES = {'deep': ('tess', 'input', 'build', 'merge'),                    # p3s3_pass.sh:58-63, 137-195
              'sublst': ('prep', 'input', 'build', 'index', 'place'),         # sublst_publish_lst.sh:26-38
              'snapshot': ('build', 'filtercat')}                             # snapshot_rechain_batch.sh:362-366
    ON_REQUEST_STAGES = {'deep': ('strat',), 'sublst': (), 'snapshot': ()}   # never in the default list (Gregg 2026-09-24 strat)
    PAGE_SETS = {'deep': ('V_wb', 'V_sb', 'I_wb', 'I_sb', 'V_fwb', 'V_fsb', 'I_fwb', 'I_fsb'),   # build order p3s3_pass.sh:170
                 'sublst': ('I_sb', 'V_sb', 'I_wb', 'V_wb')}                                     # render order sublst_pages_build.sh:14
    SUBLST_INDEX_ORDER = ('V_sb', 'I_sb', 'V_wb', 'I_wb')   # index order; sublst_pages_build.sh:20
    FORK_SETS = ('V_fwb', 'V_fsb', 'I_fwb', 'I_fsb')        # pages input by p3s3_tier_pages_input.py, raw queue for serve's multiples page; p3s3_pass.sh:172-176, 189
    RUN_DIR = {'deep': 'Run_{label}', 'sublst': 'Run_SLST{hh}B', 'snapshot': '{run}'}   # names unchanged in the window: p3s3_pass.sh:12; sublst_publish_lst.sh:16; snapshot_rechain_batch.sh:51
    # production product roots of the tiers whose root is not a shared CFG.paths root (deep = CFG.paths.TRANSIENTS_ROOT)
    PRODUCTION_OUT = {'sublst': '/fast/gh/main/phase3/sublst_pages',              # sublst_publish_lst.sh:16 (publish host node-local; 'place' copies it)
                      'snapshot': '/fast/gh/main/phase3/snapshot_rechain_{run}'}  # snapshot_rechain_batch.sh:51 (lwacalim05 node-local)
    QUEUE_NAMES = {'singles': 'queue_p3s3_{K}.txt',       # written as queue_i14.txt, renamed: p3s3_pass.sh:177, 190; p3t_pages_index.py:32
                   'singles_build': 'queue_i14.txt',      # the name build_i14 writes for a singles set (P3S3_QUEUE_NAME; p3s3_pass.sh:177)
                   'fork': 'queue_p3s3_{K}_raw.txt',      # p3s3_pass.sh:176 (read by p3s3_serve_multiples.sh)
                   'snapshot': ('queue_{S}.txt', 'queue_{S}_{tier}.txt'),   # snap_pages:608-615, 645-652
                   'strat': ('queue_strat_{S}.txt', 'queue_strat_{S}_map.csv')}   # deep_stratified_queue.py:92-100
    INDEX_NAMES = {'singles': '{band}_pages_index_{S}.csv', 'fork': '{band}_pages_index_{K}.csv'}   # build_i14:999-1004 (fork-index naming, M14 item 3)
    # ---- processes ------------------------------------------------------------------------------------------------
    DEEP_PAGE_SHARDS = 12        # processes per page set; fork_serve_p3s5_20261008.sh:32 (production); p3s3_pass.sh:182 default 6
    SUBLST_PAGE_SHARDS = {'I_sb': 10, 'V_sb': 3, 'I_wb': 2, 'V_wb': 2}   # processes; sublst_publish_lst.sh:31 (10, 3); sublst_pages_build.sh:14 (wb 2)
    SNAP_PAGES_NPROC = 24        # processes; snapshot_rechain_batch.sh:363 (module default 12, snap_pages:391)
    # ---- page content, layout, merge -------------------------------------------------------------------------------
    PAGE_SORT_KEY = 'snr'        # |snr| descending; P3_SORT_SNR=1 p3s3_pass.sh:184 (builder default peak_jy, build_i14:55)
    PAGE_CTX_IN_TEXT = True      # run-level lines in <png>.page.json, not in the image; build_i14:818 (P3_PAGES_CTX_IN_TEXT default '1')
    PAGE_FINGERPRINT_SKIP = True # an unchanged page is kept, renamed to its new rank, touched; build_i14:811-888. False (or --full) = the sidecars of the set's pages are removed first so every page renders
    # M14 (DEFECTS_MEDIUM M14, deferred by the deep chat to this command): 'shared' = today, candidates/I/<page id> for both Stokes
    # and every set (build_i14:796, 865) - the window default (DESIGN 11.1: no layout change); 'per-set' = candidates/<K>/<page id>
    # (lib/pages/m14.py), with DEEP_MERGE 'keyed'. Switching = amendment A-M14 at cutover, a full deep re-render, serve following.
    PANEL_DIR_LAYOUT = 'shared'
    DEEP_MERGE = 'in-place'      # 'in-place' = ops/p3s3_merge_review_dir.py un-keyed (p3s3_pass.sh:195); 'keyed' = lib/pages/m14.keyed_merge (M14 item 2)
    # THE review_dir merge switches, per tier (crosscheck decision 6, group 12: pages owns the merge and this ONE key; serve's
    # views read the merged files and serve's sub-LST serve-time merge calls lib/pages/merge.run with this key)
    REVIEW_DIR_MERGE = {'deep': {'P3S_MERGE_INDEX_ONLY': '', 'P3S_MERGE_STOKES_KEY': ''},       # p3s3_pass.sh:195; p3s3_serve_tier.sh:12 (unkeyed; D9 residual, M14 item 2)
                        'sublst': {'P3S_MERGE_INDEX_ONLY': '1', 'P3S_MERGE_STOKES_KEY': '1'}}   # sublst_serve.sh:52 (ledgers 1914, 1918)
    MASTER_ROOTS_MAP = '/lustre/gh/main/exopipe_ops/s2search_state/roots.csv'   # P3S2_ROOTS_CSV default, p3s2_frames.py:17 (456 rows, all CFG.paths.COADD_SB_ROOT)
    PASS_T0_REQUIRED = True      # build always gets P3S3_PASS_T0 (one value per invocation): without it build_i14:982-987 deletes the other Stokes' same-pass panels (ledger 1457)
    # inherited variables removed from the step's environment before any in-place call (core/legacy.run clears P3_* only; a stray
    # P3S_LABEL, P3S3_TIERS, P3S_MERGE_*, P3T_*, SUBLST_*, SNAP_SCALE_MODE would change a product); the step sets every variable a
    # stage needs
    LEGACY_ENV_STRIP_PREFIXES = ('P3', 'SUBLST_', 'SNAP_', 'TESS_', 'FORK_', 'FITPOS_')
    # ---- record chain, lease, per-hour use --------------------------------------------------------------------------
    REQUIRE_CATALOG_RECORD = False   # P2 across steps: refuse when an input catalogue is not the output of a done `catalog` record; False in the window (production catalogues carry no record yet)
    REQUIRE_STAGE_RECORDS = True     # P2 within the step: a stage reads only what a done record of its previous stage lists (this invocation or an earlier one, outputs unchanged)
    LEASE_WAIT_S = 0             # s; P9 one exopipe3-pages writer per run directory: wait this long for the lease, then refuse
    EPOCH_IMPLIES_RESUME = True  # --epochs (per-hour use): units whose inputs, code and config are unchanged are kept (record resume); build_i14's fingerprint skip keeps unchanged pages
    MIRROR_CHECK = True          # gate P3; False only for tests on stub code roots (an override needs a non-production --out or a ruling ID)
    # --from-run (deep parity/scratch): small files the stages rewrite are COPIED into <out>/Run_<label>, read-only trees LINKED
    # (spec section 2; as deep_forced_panels.py:338-346 builds its isolated run)
    FROM_RUN_COPY = ('source_catalogs', 'source_catalogs_V_wb', 'source_catalogs_V_sb', 'source_catalogs_I_wb',
                     'source_catalogs_I_sb', 'source_catalogs_V_fwb', 'source_catalogs_V_fsb', 'source_catalogs_I_fwb',
                     'source_catalogs_I_fsb', 'master_catalog.csv', 'v_deep_images.csv', 'v_deep_ledger.csv',
                     'frame_times.csv', 'page_noise_cache.json', 'page_noise_cache_cells.json')
    FROM_RUN_LINK = ('iresid', 'vresid', 'v110_products', 'parent_catalogs')
    # ---- MIRROR: renderer constants inside the in-place modules (PROTECTED_KEYS; checked by gate P3) -----------------
    PNG_DPI = 110                # dots per inch; build_i14:945; tenmin:391; snap_pages:371
    PANEL_CUT_DEG = 5.0          # deg, cutout side; std7_panel.py:28; snap_pages:65
    PANEL_OBS_LAT_DEG = 37.2398  # deg, latitude of the panel elevation text; std7_panel.py:29; tenmin:144
    # deg, 'hours searched' floor of the panel text: std7_panel.py:34 EL_MIN = gate_inputs.DEEP_ELEV_FLOOR_DEG (default '25.0',
    # gate_inputs.py:95) or TENMIN_ELEV_FLOOR_DEG 25.0 (gate_inputs.py:94, P3_CADENCE/P3_TIER = 10min) since ops e46bd45
    # 2026-10-13 04:20 UTC (M13(a), Gregg '(a)', ledger 2316; the spec's 20.0 predates it - caught by gate P3); snap_pages:450
    HOURS_EL_MIN_DEG = {'deep': 25.0, 'sublst': 25.0, 'snapshot': 30.0}
    PANEL_TEXT_WRAP_CHARS = 46   # characters; std7_panel.py:159
    FORCED_READING = 'bilinear'  # yellow number = bilinear pixel value at the reticle over the annulus rms; std7_panel.py:124-136
    FRAME_CACHE = {'deep': 14, 'sublst': 12}   # frames held per process; build_i14:189; tenmin:257
    PREDICTED_PANELS = {'deep': True, 'sublst': True, 'snapshot': False}   # build_i14 Predicted V/I cells; snap_pages:295-300
    PRED_V_MATCH_ARCMIN = 1.0    # arcmin, predicted-V cutout to page position; build_i14:132; tenmin:181
    PRED_I_PARENT_MIN_JY = 2.0   # Jy, template stamp parents; build_i14:260, 540; tenmin:216
    PRED_I_MIN_SEP_BEAMS = 2.5   # beams from the position; build_i14:260, 540
    PRED_I_STAMPS = (8, 40)      # (minimum, maximum) parent stamps; build_i14:269-271
    PRED_I_STAMP_HALF_BEAMS = 6.0   # beams; build_i14:249, 536
    PRED_I_QA_DISK_BEAMS = 3.0   # beams, split-half disk; build_i14:274, 549
    PRED_I_QA_R_MIN = 0.5        # split-half correlation; below it 'TEMPLATE FAILED QA'; build_i14:280, 552
    PRED_I_PLACE_MIN_JY = 1.0    # Jy, parents placed in the predicted map; build_i14:287, 559
    I_TEMPLATE_NULLS = (40, 40)  # (empty-sky, prediction-matched) nulls per frame; build_i14:595, 604
    I_TEMPLATE_NULL_PCTL = 95.0  # percentile of the nulls; build_i14:614
    I_TEMPLATE_MIN_NULLS = 20    # nulls needed, else 'no-template'; build_i14:614
    I_TEMPLATE_MARGINAL_FACTOR = 1.5   # x p95; build_i14:618
    I_TEMPLATE_SEED = 20260905   # rng seed of the nulls; build_i14:566
    XMATCH_RADIUS_ARCMIN = 15.0  # arcmin, panel footer cross-match; build_i14:722; tenmin:122 (DUP-6: the catalogue uses 1 beam)
    XMATCH_SAMPLES = {'deep': ('10pc', '25pc', '50pc', 'flare_ucd', 'pulsars_atnf', 'sydney_stars', 'Hot_Warm_Jupiters'),   # build_i14:35-36 (50pc absent = omitted)
                      'sublst': ('10pc', '25pc', '50pc', 'Hot_Warm_Jupiters', 'flare_ucd', 'LPTs', 'stars', 'pulsars_atnf', 'sydney_stars', 'ATNF_psr')}   # tenmin:52
    XMATCH_SAMPLE_DIR = '/lustre/gh/main/clv15'   # sample CSVs (= CFG.paths.CLV15_ROOT); build_i14:725; tenmin:53
    SKY_AREA_DEG2 = 41252.96     # deg^2, all-sky density of a sample; build_i14:737; tenmin:112
    VHF_ONLY = {'deep': ('identity', ('59MHz', '64MHz', '69MHz', '78MHz', '82MHz', 'Blue')),   # build_i14:153 (73 MHz absent)
                'sublst': ('band', ('59', '64', '69', '73', '78', '82'))}                  # tenmin:54, 372; label only; B-43 default keeps both
    SUBLST_FIT_FLUX_SNR = 5.0    # sigma; Gaussian flux when a fitted size exists, else the pixel value; tenmin:347
    PB_LABEL = {'deep': 'PB-corr peak', 'sublst': 'PB-corr value at position'}   # std7_panel.py:288 default; tenmin:367
    RULED = {'PAGE_SORT_KEY': 'R-P3S2-PAGE-STANDARDS',
             'PAGE_CTX_IN_TEXT': 'R-GREGG-20261011-PAGE-SKIP-DESIGN-AND-DEEP-FORCED-TEMPLATE-FIT',
             'PAGE_FINGERPRINT_SKIP': 'IMPL-INCREMENTAL-PAGES-STEP2-CONTEXT-IN-TEXT-SKIP-UNCHANGED',
             'PANEL_DIR_LAYOUT': 'FIX-REFACTOR-MEDIUM-DEFECTS-M1-M19-DEEP-20261013',
             'DEEP_MERGE': 'FIX-REFACTOR-MEDIUM-DEFECTS-M1-M19-DEEP-20261013',
             'REVIEW_DIR_MERGE': 'FIX-TENMIN-PAGES-STOKES-KEY-VIEWS-AND-REVIEW-DIR-MERGE',
             'MIRROR_CHECK': 'R-STANDARD-7CUTOUT-PANEL',
             'PANEL_CUT_DEG': 'R-STANDARD-7CUTOUT-PANEL',
             'HOURS_EL_MIN_DEG': ('R-GREGG-20261013-RULINGS-ON-PENDING-DECISIONS', 'R-SNAP-ELEVATION-FLOOR-30-DEG'),
             'PANEL_TEXT_WRAP_CHARS': 'PANEL-INFO-TEXT-RESTORED-TO-PREVIOUS-SIZE-WRAP-BACK-TO-46-CHARACTERS',
             'FORCED_READING': 'R12-SERVED-PANEL-YELLOW-READING-PIXEL-VALUE',
             'PREDICTED_PANELS': ('R-PREDICTED-MAPS-ARE-REVIEW-PRODUCTS', 'SNAP-INTERFACE-NO-PREDICTED-NO-SIDECARS'),
             'PRED_I_QA_R_MIN': 'R-PREDICTED-TEMPLATE-SHOWN-EVEN-IF-QA-FAILS',
             'I_TEMPLATE_NULLS': 'R-I-TEMPLATE-VERDICT-AS-PAGE-LABEL',
             'SUBLST_FIT_FLUX_SNR': 'R-PRIMARY-FLUX-IS-THE-GAUSSIAN-FIT-AND-THE-5-SIGMA-SWITCH-IS-MEASURED',
             'VHF_ONLY': 'R-BATTERY-CARD-SECTION-9-APPROVED-10MIN-ELEVATION-25-VHF-LABEL-ONLY-RATIO-2',
             'PB_LABEL': 'IMPLEMENTED-Q7-10MIN-PBCORR-LABEL-AND-REMEASURED-5SIGMA-SWITCH'}


class pages_deep(_Section):
    """Deep tier: pages input (port of the p3s3_pass.sh:137-169 heredoc; ops/p3s3_tier_pages_input.py in place for the fork
    sets), the build environment of the production pass, astrometry figures (drawn inside build_i14), TESS coverage
    (ops/tess/tess_coverage.py in place) and the stratified queue (ops/deep_stratified_queue.py in place)."""
    # COLUMNS, DROP_ROWS_WITHOUT_DATE_OR_LST and FIGURE_TIMEOUT_S describe fixed behaviour of the port / the in-place builder
    # that no code reads (review 2026-10-13): protected, so an override cannot pretend to change it (B-30 needs code)
    PROTECTED_KEYS = ('PASS_MS', 'TESS_ORBIT_REFRESH_DAYS', 'TESS_ORBIT_URL', 'TESS_INTEGRATION_S', 'STRAT_EL_BINS_DEG',
                      'STRAT_PARENT_BINS_BEAMS', 'STRAT_PARENT_SNR_MIN', 'COLUMNS', 'DROP_ROWS_WITHOUT_DATE_OR_LST',
                      'FIGURE_TIMEOUT_S')
    # ---- input (DUP-11: three behaviours kept 1:1 until a parity-tested merge) ----
    SINGLES_SETS = ('V_wb', 'I_wb', 'V_sb', 'I_sb')   # loop order of the heredoc; p3s3_pass.sh:144
    COLUMNS = 46                 # columns of a deep pages_input file (header of Run_P3S5 pages_input_V_wb.csv)
    DROP_ROWS_WITHOUT_DATE_OR_LST = True   # silently dropped today: p3s3_pass.sh:165; p3s3_tier_pages_input.py:26 (B-30 a = refuse); counted in the unit record
    PASS_MS = '1'                # literal column value; p3s3_pass.sh:152; p3s3_tier_pages_input.py:18 (mirror)
    ATOMIC_WRITE = {'singles': False, 'fork': True}   # heredoc writes in place (p3s3_pass.sh:166); tier script .new + rename (p3s3_tier_pages_input.py:28-30)
    FORK_PAGES_LABEL = 'multiples fork'   # label column of fork-set pages input; p3s3_pass.sh:175
    # ---- the production pass environment of every page stage (fork_serve_p3s5_20261008.sh:31; P3S_LABEL from --run, P3_PARENT_DIR
    # from --deep-parent-dir, NUMPY_MADVISE_HUGEPAGE from runner.ENV_DEFAULTS) ----
    PASS_ENV = {'P3_CADENCE': 'deep', 'P3_TABLE_ELEV': '1'}
    PARENT_DIR = None            # P3_PARENT_DIR of the pass (fork_serve_p3s5_20261008.sh:15, 31). None = REQUIRED from --deep-parent-dir: the figures' source_localization -> gate_inputs.parents read it (gate_inputs.py:19 falls back to the retired Run_P3S4 parents)
    # ---- localization tables (written by catalog --stages localize) and astrometry figures (drawn by build_i14) ----
    LOC_TABLES = ('source_catalogs/localizations.csv', 'source_catalogs/localization_offsets.csv')   # gate P4: present before build (p3s3_pass.sh:132-134 order)
    FIGURE_MIN_DETECTIONS = 2    # astrometry figure only for identities with n_det > 1; build_i14:959
    FIGURE_TIMEOUT_S = 600       # s per figure subprocess; build_i14:968
    FIGURE_FAILURE = 'record'    # deep swallows figure failures (build_i14:966-970): missing figures counted; 'refuse' = B-30 a
    # ---- TESS coverage (an annotation, never a gate) ----
    TESS_ENABLED = True          # tiers with tess_coverage.csv today: deep only (p3s3_pass.sh:62); HANDOFF-TESS-COVERAGE-SNAPSHOT-SUBLST-RUNS open (spec 12.8)
    TESS_IS_GATE = False         # a failure is recorded (unit no-input with the reason), never refused; p3s3_pass.sh:62-63
    TESS_STATUSES = ('simultaneous', 'not observed', 'no epoch time')   # tess_coverage.py:13-14, 120-140
    TESS_STOKES = 'I,V'          # tess_coverage.py:60 default
    TESS_ORBIT_REFRESH_DAYS = 7  # days; orbit table re-fetched when older (tess_coverage.py:44) (mirror; B-58)
    TESS_ORBIT_URL = 'https://tess.mit.edu/public/files/TESS_orbit_times.csv'   # tess_coverage.py:39 (mirror)
    TESS_INTEGRATION_S = 10.03102041   # s; tess_coverage.py:77 (mirror; = instrument.INTEGRATION_S)
    # ---- stratified judging queue (on request) ----
    STRAT_PER_STRATUM = None     # identities per stratum: REQUIRED (deep_stratified_queue.py:22); --deep-strat-per-stratum
    STRAT_SEED = 20260924        # deep_stratified_queue.py:23
    STRAT_EL_BINS_DEG = (30.0, 50.0)       # deg; deep_stratified_queue.py:76 (mirror)
    STRAT_PARENT_BINS_BEAMS = (1.0, 3.0)   # target-image beams; deep_stratified_queue.py:77 (mirror)
    STRAT_PARENT_SNR_MIN = 20.0  # parent peak / cell median frame rms; deep_stratified_queue.py:74 (mirror)
    RULED = {'TESS_IS_GATE': 'IMPL-TESS-COVERAGE-ON-PAGES-AND-LIGHTCURVE-SKILL',
             'TESS_STATUSES': 'IMPL-TESS-SIMULTANEOUS-ONLY-AND-SOURCE-HEADER-LAYOUT',
             'PASS_ENV': 'R-GREGG-20261013-RULINGS-ON-PENDING-DECISIONS'}


class pages_sublst(_Section):
    """Sub-LST page run (ops/sublst_pages_prep.py, p3t_pages_input.py, build_tenmin_pages.py, p3t_pages_index.py in place) and the
    publish wrapper's page part (sublst_publish_lst.sh, sublst_pages_build.sh). Product root of the inputs: CFG.paths.SUBLST_ROOT."""
    # LOO_MASTER and OLD_CHAIN_FLAG describe sublst_pages_prep.py in place (B-27, E22): protected, no code reads them (review)
    PROTECTED_KEYS = ('SERVED_CATALOG_TAG', 'REQUIRED_FILES', 'RUN_COLUMN', 'LOO_MASTER', 'OLD_CHAIN_FLAG')
    ARM = 'B'                    # catalogue arm served (cube-based chain, set B); sublst_publish_lst.sh:28, 31
    SEARCH_ROOT = '{root}/search_{arm}'   # SUBLST_SEARCH_ROOT; sublst_publish_lst.sh:31; sublst_pages_build.sh:8
    LOC_REQUIRED = '{root}/compare_{lst}/new_{arm}/catalog/localizations_noveto/localizations.csv'   # refusal P11; sublst_publish_lst.sh:26
    CATALOG_DIR = '{root}/compare_{lst}/new_{arm}/catalog'   # prep's source (sublst_pages_prep.py:31)
    PARENT_LINK = '{root}/parent_catalogs_{lst}'   # <P>/parent_catalogs -> this; sublst_publish_lst.sh:29
    SERVED_CATALOG_TAG = 'noveto'   # source_catalogs_noveto_<S>_<T> copied as the served catalogues; sublst_pages_prep.py:69 (mirror)
    REQUIRED_FILES = ('pertie/pertie_detections.csv', 'rejected.csv', 'rejected_index.npz', 'detection_ts.csv')   # refusals sublst_pages_prep.py:39-48 (P12, P13; mirror)
    RUN_COLUMN = 'P3T4'          # literal 'run' column of sub-LST pages input; p3t_pages_input.py:35 (mirror; label only)
    LOO_MASTER = 'stack-minus-armA-residual'   # master panel = window stack - arm-A residual; sublst_pages_prep.py:113-126 (B-27 default)
    OLD_CHAIN_FLAG = True        # source_class OLD-CHAIN where compare_<lst>/old exists (10h, 19h); sublst_pages_prep.py:32, 95
    WIPE_CANDIDATES = True       # rm -rf <P>/candidates before the shards; sublst_pages_build.sh:9 (no fingerprint skip in this tier)
    FRESH_RUN_DIR = True         # rm -rf <P> before prep; sublst_publish_lst.sh:27
    SHARD_FAILURE = 'record'     # a crashed shard is recorded; the LST is placed when every index passes (sublst_pages_build.sh:16-19; B-30 a = 'refuse')
    PLACE_TARGET = '{review_host}:/data07/gh/phase3/transients/{run}/'   # rsync -a --delete; sublst_publish_lst.sh:38 ({review_host} = the site file's REVIEW_HOST; no host name in config, README rule 2)
    PLACE_DELETE = True         # --delete (stale 19h test pages, 2026-09-30)
    CODE_PINNED = True           # the in-place code must be a verified frozen tree (FROZEN.ok + MANIFEST.sha256; sublst_code_root.sh:15-31)
    RULED = {'SERVED_CATALOG_TAG': 'R-SUBLST-FORCED-SKIP-V110-AND-LEAK-VETO',
             'PLACE_DELETE': 'R-NEW-CHAIN-TAKES-OVER-10MIN-PAGES-CUMULATIVE',
             'CODE_PINNED': 'INSTALLED-FROZEN-CODE-PER-10MIN-LAUNCH-20261011',
             'ARM': 'R-10MIN-TIER-NEW-CUBE-CHAIN-ADOPTED'}


class pages_snapshot(_Section):
    """Snapshot pages (clv15/phase3_snapshot_pages.py with clv15/phase3_snapshot_panels.py frame readers, in place) and the filter
    catalogue (ops/snapshot_filter_catalog.py). Interpreter: site PYTHON['py36'] (production: /opt/devel/pipeline/envs/development/
    bin/python, snapshot_rechain_batch.sh:50). Search outputs: CFG.paths.SNAPSHOT_FLEET_OUT; filter coadd root: CFG.paths.COADD_SB_ROOT."""
    PROTECTED_KEYS = ('MASTERS_ROOT', 'PANELS_TOP_N', 'FIT_NO_SOURCE_SNR', 'IMAP_CHUNK')
    MASTERS_ROOT = '/lustre/pipeline/exopipe/phase3/snapshot_masters_gen4'   # frozen Masters (pinned); snapshot_rechain_batch.sh:56
    FIT_STORE = '{ops}/snapshot_batch/rechain_{run}/fits_store.csv'   # + _template.csv; snapshot_rechain_batch.sh:264 ({ops} = CFG.paths.OPS_ROOT)
    FLOW_FRAMES = True           # panels in the flow-registered frame (P3_SNAP_FIT_RAW never set); snap_pages:54-56, 396-399
    PANELS_TOP_N = 10            # panels per identity per (Stokes, tier) + best of each other band; snap_pages:106 (mirror)
    FIT_NO_SOURCE_SNR = 8.0      # sigma, fit_no_source flag; snap_pages:560-561 (mirror)
    SCALE_MODE = 'residual'      # display limits from each card's own residual cutouts; snap_pages:363 (SNAP_SCALE_MODE default)
    KEEP_PANELS = False          # --resume of phase3_snapshot_pages.py; off in every launcher (snapshot_rechain_batch.sh:362-363)
    IMAP_CHUNK = 4               # panels per worker task; snap_pages:670 (mirror)
    PYTHON_KEY = 'py36'          # site PYTHON key of the snapshot scripts (crosscheck decision 2)
    RULED = {'FLOW_FRAMES': 'R-FORCED-PANEL-FITTED-POSITION-AND-RETICLE',
             'SCALE_MODE': 'R-SNAP-PANEL-SCALES-TIED-TO-THE-RESIDUALS',
             'PANELS_TOP_N': 'SNAP-INTERFACE-PANELS-TOP10-PER-BAND',
             'FIT_NO_SOURCE_SNR': 'R-SNAP-NO-SOURCE-FIT-CLASS-ROUTED'}
