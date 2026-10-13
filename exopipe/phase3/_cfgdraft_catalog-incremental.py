"""Config draft (refactor window) - unit catalog-incremental: `exopipe3 catalog --tier deep --incremental` (DESIGN 12.1, 12.3).

Loaded by config._load_drafts until the refactor chat folds it into config.py. ONE section, catalog_incremental. It holds
only what is NEW to the incremental path. Every value the full deep chain also uses is read from the step owner's
sections, so each value lives in one place (lib/catalog/incremental.py, `_deep()`):
  label            catalog.DEEP_RUN_LABEL (when --run is not given)
  run root         paths.TRANSIENTS_ROOT (shared root, _CROSSCHECK decision 3)
  stage env        P3_CONFORM_SIZE from catalog_conform.SIZE_STEP['deep'], P3_TABLE_ELEV from catalog_conform.TABLE_ELEV['deep'],
                   P3_NO_PARENT_EXCL from catalog_select.PARENT_EXCLUSION, P3_CADENCE=deep (fork_serve_p3s5_20261008.sh:31-32,
                   p3s3_pass.sh:42, :57)
  processes        catalog_fitpos.NPROC['deep'] (p3s3_pass.sh:49), catalog_fork.NPROC (fork_serve_p3s5_20261008.sh:32)
  fork Stokes      catalog_fork.STOKES (p3s3_pass.sh:90)
  union            catalog_identity.UNION_TIERS / UNION_KINDS (p3s3_pass.sh:111-113; the port is lib/catalog/deep_ported.py)
In-place scripts are named relative to the site legacy root 'exopipe_ops' (core.env.legacy_root) and run through
core.legacy.run with the site interpreter 'default' (_CROSSCHECK decisions 1-2).
Tags: [step] used by lib/catalog/incremental.py; [ast] read from the live file by AST, never copied; [arg]/[env] passed
to the in-place module.
"""
from exopipe.phase3.config import _Section


class catalog_incremental(_Section):
    """exopipe3 catalog --incremental: one epoch (<LST>_<date>, all bands, both Stokes) added to the cumulative deep catalogue
    with results identical to a full rebuild over the same epochs (lib/catalog/incremental.py explains how)."""
    # Keys that name the code which decides the products. An override would make the record lie about what ran (the
    # module is in place; change it by a commit after absorption).
    PROTECTED_KEYS = ('STAGE_SCRIPTS', 'BUILDER', 'BUILDER_FUNCS', 'BUILDER_CONSTS', 'ASTROM_MODULE', 'SSTACK_SOURCE',
                      'SOURCE_FILES', 'FORK_KEYS')
    # ---- scope -------------------------------------------------------------------------------------------------------
    TIERS = ('deep',)                    # tiers with an incremental path; sublst / snapshot refuse (NOT DONE, impl note) [step]
    STATE_DIR = '_incremental'                            # under --out: epoch blocks, association state, deltas (NEW) [step]
    PHASES_DEFAULT = ('epoch', 'associate')               # 'identity' runs after `exopipe3 forced` (full-chain order p3s3_pass.sh:96-136; gate D-G22) [step]
    REQUIRE_FORCED_BEFORE_IDENTITY = True                 # spec catalog.md 5.1 D-G22 (a done `forced` record newer than the associate record) [step]
    FORCED_RECORDS = '{out}/_records/forced'              # where `exopipe3 forced` writes its unit records for this run (core/record layout) [step]
    # ---- where an epoch's detections come from -----------------------------------------------------------------------
    # 'run'    = the epoch's rows of a deep run's MERGE OUTPUT (master_catalog_asrun.csv, written by p3s2_conform.py:49 as a byte
    #            copy of p3s3_merge.py's output) plus its ledger, images, refused rows and frames. The merge gate works per
    #            (lst, band, Stokes, date) key (p3s3_merge.py:113-140), so these rows ARE what an epoch-scoped merge accepts.
    # 'search' = the per-hour search output (`exopipe3 search --epochs`): refuses until that layout exists (impl note NOT DONE).
    EPOCH_SOURCE = 'run'
    SOURCE_RUN = None                                     # None = paths.TRANSIENTS_ROOT/Run_<label>
    SOURCE_FILES = {'merged': 'master_catalog_asrun.csv', 'refused': 'master_catalog_refused.csv',
                    'ledger': 'v_deep_ledger.csv', 'images': 'v_deep_images.csv',
                    'fitcache': 'fit_positions_cache.csv'}   # p3s3_merge.py:100-160, p3s2_conform.py:49, deep_fit_positions.py:118 [step]
    SEED_FIT_CACHE_FROM_SOURCE = True                     # copy the source run's fit-cache rows of the epoch into the epoch's fit cache.
                                                          # deep_fit_positions.py keys the cache by cand_id|frame realpath|mtime|ra|dec|snr
                                                          # (:54-55), so a hit IS the same fit. No product changes; saves the refit [step]
    # ---- order independence ------------------------------------------------------------------------------------------
    # The assembled merge output is ALL epoch blocks' rows sorted by this key (a total order: cand_ids are unique), so the
    # replayed chain sees the same bytes whatever order the epochs were added in. NEW (production: published-attempt order;
    # no ruling yet - the order affects only the order of equal-|SNR| sources, impl note section 3).
    CANONICAL_ROW_ORDER = ('date', 'lst', 'cand_id')
    # ---- the replay of the population-coupled stages (in place, exactly as p3s3_pass.sh runs them) -------------------
    REPLAY_STAGES_ASSOCIATE = ('conform', 'fitpos', 'select', 'ephem', 'split', 'build', 'fork')   # p3s3_pass.sh:42-94 [step]
    REPLAY_STAGES_IDENTITY = ('union', 'identity', 'localize', 'list')                             # p3s3_pass.sh:98-136 [step]
    CATALOG_KEYS = ('V_wb', 'I_wb', 'V_sb', 'I_sb')       # per-(Stokes, tier) builds, this order; p3s3_pass.sh:77 [step]
    FORK_KEYS = {'V_fwb': 'fork_V/selected_V_wb_fork.csv', 'V_fsb': 'fork_V/selected_V_sb_fork.csv',
                 'I_fwb': 'fork_I/selected_I_wb_fork.csv', 'I_fsb': 'fork_I/selected_I_sb_fork.csv'}   # the final fork build inputs; multiples_fork.py:483-484 [step]
    BUILD_WORKERS = 1                                     # concurrent catalogue builds; 1 = today's sequence (spec 11 speed-up 1: no product change) [step]
    PARENT_DIR = None                                     # P3_PARENT_DIR (parent snapshot): REQUIRED, never defaulted (B-59; gate_inputs.py:19
                                                          # would fall back to Run_P3S4); --inc-parents or the step's --parents DIR [env:P3_PARENT_DIR]
    STAGE_SCRIPTS = {'conform': 'p3s2_conform.py', 'fitpos': 'deep_fit_positions.py', 'select': 'select_v3.py',
                     'ephem': 'p3s2_ephem_filter.py', 'build': 'build_source_catalogs_v3.py', 'fork': 'multiples_fork.py',
                     'identity': 'p3s3_identity.py', 'localize': 'build_localizations.py',
                     'list': 'p3s2_transient_list.py'}    # under legacy_root('exopipe_ops'); p3s3_pass.sh:39-136 [step]
    STAGE_TIMEOUT_S = None                                # None = no time stop (Gregg: no time stop on production chains) [step]
    TRANSIENT_LIST_NAME = 'TRANSIENT_LIST_{label}_{utc}.csv'   # + _latest symlink; p3s3_pass.sh:136 [step]
    # ---- the association engine (lib/catalog/incremental.py): the LIVE builder's own functions, loaded by AST --------
    BUILDER = 'build_source_catalogs_v3.py'               # under legacy_root('exopipe_ops'); functions and constants below, read at every run [ast]
    BUILDER_FUNCS = ('_stat_as', '_elev', '_sys_det', '_sys_class', '_sstack_fails', 'jname', 'jname_long', 'unique_names',
                     'assoc_groups')                      # build_source_catalogs_v3.py:398-459, 494-550, 580-600, 620-625 [ast]
    BUILDER_CONSTS = ('K_ASSOC', 'OVRO_LAT', 'SYS_SCALE', 'SYS_PROVISIONAL_AS', 'THR')   # :127, :131, :483, :528-529 [ast]
    ASTROM_MODULE = 'p3s3_astrom.py'                      # under legacy_root('exopipe_ops'): the builder's _astrom (sys_arcmin, band_class) [in place]
    # The builder takes the sigma-stack threshold from the PRODUCTION package checkout (sys.path.insert(0,
    # '/lustre/gh/main/exopipe'), build_source_catalogs_v3.py:59, then search_i_transients.CONFIG, :613-617). That checkout is
    # not a site legacy root and has no paths key, so the file is named here; it is read by AST.
    SSTACK_SOURCE = '/lustre/gh/main/exopipe/exopipe/phase3/search_i_transients.py'   # CONFIG['SIGMA_THRESHOLD'] = 7.0 at :71 [ast]
    SSTACK_FALLBACK = 7.0                                 # the builder's own fallback when that module is not importable (:617; B-30)
    BEAM_DEFAULT_ARCSEC = 900.0                           # band without a master beam (deep; build_source_catalogs_v3.py:215; B-32)
    PAIR_SEARCH_PAD = 1.000001                            # KD-tree candidate radius x this; the exact builder test follows (no effect on results) [step]
    CROSSCHECK = True                                     # after each build: the builder's member partition and names must equal the
                                                          # engine's, else the unit REFUSES and the state is not advanced [step]
