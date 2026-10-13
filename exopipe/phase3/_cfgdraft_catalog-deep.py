"""REFACTOR WINDOW DRAFT (unit catalog-deep): the config sections of `exopipe3 catalog` shared by every tier, and the deep
ones. config.py loads this file (config._load_drafts) until the refactor chat folds it into config.py.

Source: plan/specs/catalog.md section 3 (values = production values at exopipe_ops 0c1580c / clv15 9b6525d / exopipe
03ca797; each line names its source file:line and its ruling). Paths named 'ops' = /lustre/gh/main/exopipe_ops,
'clv' = /lustre/gh/main/clv15.

Binding of each key during the refactor window (DESIGN 11.1: the step calls clv15 / exopipe_ops modules IN PLACE):
  [env:X]   passed to the in-place module as environment variable X (the module reads it today)
  [arg:--x] passed as that command-line argument
  [lit:F:N] a literal in the in-place module F at line N. lib/catalog/deep_binding.BINDINGS reads each such literal from
            the module's source (AST, nothing imported) at the start of every deep run and REFUSES when it differs from the
            config key (spec 1.3 item 5). The bound keys of these sections are in their PROTECTED_KEYS ("window-bound"): an
            override would change nothing while the module carries its own literal, and the record would lie. A bound key of
            a shared section (astrometry, fitting, paths) is refused by the step at its first unit.
  [step]    used by the step's own (ported) code (steps/catalog.py, lib/catalog/deep_*.py)
Protection uses the section metas config.py honours (refactor decisions 2026-10-13, _CROSSCHECK.md item 4): PROTECTED_KEYS
(no --set / --config) and RULED (override only with --ruling-for or into a non-production --out).
Shared roots are NOT repeated here (decision item 3): CFG.paths.TRANSIENTS_ROOT (deep run dirs), COADD_SB_ROOT (masters),
PHASE3_ARCHIVE_ROOT + ARCHIVE_RUN (searched frames), CLV15_ROOT (sample catalogues), OPS_ROOT (search state); the
exopipe_ops root of the in-place modules is core.env.legacy_root('exopipe_ops') and the interpreter is the site PYTHON key
catalog.DEEP_PYTHON. Parent catalogues are built ONLY by `exopipe3 masters --tier deep --stages sources` (decision item 6):
catalog preflight assembles the per-pass snapshot of the current ones and refuses naming that command.

Sections the TIER modules own (catalog_sublst, catalog_sublst_tie, catalog_sublst_veto, catalog_snapshot) are NOT here:
they belong to _cfgdraft_catalog-tiers.py. Per-tier dicts below carry the sub-LST / snapshot entries of the shared
sections (spec section 3), so the tier draft must not redefine these sections.
"""
from exopipe.phase3.config import _Section


class catalog(_Section):
    """exopipe3 catalog - roots, run labels and reference inputs shared by every stage and tier; deep chain orchestration."""
    PROTECTED_KEYS = ('SOLAR_AUDIT', 'PULSARS_CSV', 'XMATCH_FILES', 'TABLES_DEEP', 'TABLES_SUBLST',      # spec 3 (reference inputs)
                      'DEEP_ROOTS_CSV', 'DEEP_PARENT_CANDIDATES', 'DEEP_PARENT_DIR_DEFAULT', 'SITE_LAT_DEG',   # window-bound
                      'SITE_LON_DEG', 'SITE_HEIGHT_M', 'BAND_CLASS', 'COLLECT_ROOTS', 'DEEP_STAGING_ATTEMPTS')
    # ---- deep run (tier B13-B14)
    DEEP_RUN_LABEL = 'P3S5'                               # current deep generation label (default_out only; --run is required); ops/fork_serve_p3s5_20261008.sh:31 [env:P3S_LABEL]
    DEEP_LABEL_PREFIX = 'P3S'                             # labels that are deep runs: fitted-position stage runs only for them; ops/p3s3_pass.sh:48-52 [step]
    DEEP_FORK_SKIP_PREFIX = 'P3T'                         # labels whose fork is skipped (the retired 10-minute generation); ops/p3s3_pass.sh:89 [step]
    COLLECT_ROOTS = ('/lustre/gh/main/phase3_collect', '/data07/gh/phase3/transients')   # newest first; ops/p3s3_sources.py:26 [env:P3S3_COLLECT_ROOTS] (2026-09-14)
    DEEP_STATE_DIR = '{ops_root}/{label_lower}_state'     # published_<S>.csv + markers_<S>/ ({ops_root} = paths.OPS_ROOT); ops/p3s3_sources.py:41, ops/p3s3_stale_exclusions.py:12 [env:P3S3_STATE_DIR]; a non-production --out gets a COPY under <out>/_state/
    DEEP_STAGING_ATTEMPTS = '/fast/gh/main/phase3/p3s3_staging/attempts'  # node-local attempt roots the harvest pulls from; ops/p3s3_sources.py:42, ops/p3s3_harvest_published.sh:11 [step]
    DEEP_ROOTS_CSV = '/lustre/gh/main/exopipe_ops/s2search_state/roots.csv'   # cell -> co-add root; ops/p3s2_frames.py:17 [env:P3S2_ROOTS_CSV]; ops/p3s2_conform.py:130 [lit]
    DEEP_PARENT_SNAPSHOT = '/fast/gh/main/phase3/p3s5_parents_union_{utc}'  # per-pass parent snapshot (production); ops/fork_serve_p3s5_20261008.sh:15 [env:P3_PARENT_DIR]; non-production --out: <out>/_parents/p3s5_parents_union_{utc} (never lwacalim07 /fast)
    DEEP_PARENT_CANDIDATES = ('/lustre/gh/main/phase3_collect/p3s5_collect_lwacalim07/parent_catalogs',
                              '/lustre/gh/main/phase3_collect/p3s5_collect_lwacalim09/parent_catalogs',
                              '/data07/gh/phase3/transients/Run_P3S4/parent_catalogs')   # ops/p3s5_parent_union.py:12-13 [lit]
    DEEP_PARENT_DIR_DEFAULT = '/data07/gh/phase3/transients/Run_P3S4/parent_catalogs'   # ops/gate_inputs.py:19 module default; the step ALWAYS sets P3_PARENT_DIR (B-59) [lit]
    DEEP_PARENT_TEST_SAMPLE = 24                          # cells whose parent catalogue digest the parent_snapshot content test re-reads against the master (seeded sample; every cell is checked structurally) [step] (new: content test)
    SEARCH_QUARANTINE_FALLBACK = True                     # target beam read from Run_SEARCH_QUARANTINE_* when the frame moved (2026-10-08); ops/gate_inputs.py:144-153 (B-30, B-59: fallback kept, counted)
    INSTALL_LOCK = '{ops_root}/membership_rebuild_20261007/.install.lock'   # held from the parent snapshot to the fork I content test; ops/fork_serve_p3s5_20261008.sh:18-19, 38-39 [step] -> runner lease 'deep-install' (K7). Taken only by a run that writes a production root
    # ---- reference inputs (versioned paths; a missing enabled input refuses - B-58)
    SOLAR_AUDIT = '/lustre/pipeline/exopipe/phase3/QA/solar_contamination_audit.csv'   # daytime gate quarantine bands; ops/p3s2_conform.py:315 [lit] (R-DAYTIME-GATE-DEEP-CONFORM; B-35 open)
    PULSARS_CSV = '/lustre/gh/main/clv15/pulsars_atnf.csv'   # pulsar-source consistency; ops/p3s2_conform.py:363 (Gregg 2026-09-28, R-PULSAR-TABLE-EXEMPTION-PULSAR-MUST-BE-THE-PARENT)
    XMATCH_FILES = ('pulsars_atnf', '10pc', '25pc', 'Hot_Warm_Jupiters', 'flare_ucd', 'sydney_stars', 'LPTs')   # ops/source_localization.py:64 [lit]; 50 pc omitted by absent file (R-P3S2-PAGE-STANDARDS; contract 48)
    TABLES_DEEP = '/lustre/gh/main/exopipe/exopipe/common/parent_curve_tables.json'          # deep singles + multiples tables; ops/multiples_fork.py:57 [lit]; parent_curve_table.tables()
    TABLES_SUBLST = '/lustre/gh/main/exopipe/exopipe/common/parent_curve_tables_sublst.json' # sub-LST tables (frozen tree copy in production); ops/sublst_catalogs.sh:75 [env:P3_TABLE_EXTRA]
    FOOTPRINT_CAT_DIR = {'deep': '/fast/gh/fp_artifact_investigation/support/bdsf_catalogs_aug',          # lwacalim07 /fast, 345 files, 1.5 GB (2026-10-13); ops/select_v3.py:20 [env:P3_FOOTPRINT_CAT_DIR default] (B-36 open)
                         'sublst': '/lustre/gh/main/phase3/sublst/catalog_inputs/bdsf_catalogs_aug_{lst}'}  # 19h: no suffix; ops/sublst_catalogs.sh:26
    # ---- site copies the in-place modules carry (B-42 open; listed so the binding check sees every copy)
    SITE_LAT_DEG = {'conform': 37.2398, 'gate_inputs': 37.2398, 'builder': 37.2397, 'shuffle': 37.2397, 'localize': 37.2398,
                    'leak_veto': 37.2398, 'snapshot_chain': 37.2398, 'ephem': 37.2398, 'forced_radio': 37.2398, 'conform_mid_el': 37.2398}   # deg; ops/p3s2_conform.py:287 and :211, gate_inputs.py:96, build_source_catalogs_v3.py:131, p3s3_repeat_shuffle.py:15, source_localization.py:61, leak_veto_sublst.py:37, clv/phase3_snapshot_chain.py:394, p3s2_ephem_filter.py:11, deep_forced_radio.py:75 [lit]
    SITE_LON_DEG = -118.2817                              # deg; ops/p3s2_conform.py:287, deep_fit_positions.py:37, sublst_fit_positions.py:54, p3s2_ephem_filter.py:11 [lit]
    SITE_HEIGHT_M = {'conform': 1200.0, 'ephem': 1188.0, 'snapshot_chain': 1183.0}   # m; ops/p3s2_conform.py:287, p3s2_ephem_filter.py:11, clv/phase3_snapshot_chain.py:394 [lit]
    BAND_CLASS = {'Red': ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz'), 'Green': ('41MHz', '46MHz', '50MHz', '55MHz', '59MHz'),
                  'Blue': ('64MHz', '69MHz', '73MHz', '78MHz', '82MHz')}   # ops/p3s2_conform.py:25-26, select_v3.py:21-24, p3s2_ephem_filter.py:13-14 [lit] (Gregg 2026-10-07 Q14)
    # ---- deep chain orchestration (the step's own code; no product value)
    DEEP_STAGES = ('preflight', 'harvest', 'merge', 'conform', 'fitpos', 'select', 'associate', 'fork', 'union', 'identity',
                   'localize', 'list')                    # pipeline order (spec 1.2); ops/p3s3_pass.sh:38-136 + fork_serve_p3s5_20261008.sh:20-30 [step]
    DEEP_DEFAULT_UNTIL = 'fork'                           # a plain deep run stops after the fork: `exopipe3 forced` writes forced_* into the tier survivors in place (ops/p3s3_forced_phot.py:70-89) before union..list (`--from union`) [step]
    DEEP_ASSOC_KEYS = ('V_wb', 'I_wb', 'V_sb', 'I_sb')    # build order of the four primary catalogues; ops/p3s3_pass.sh:77 [step]
    DEEP_ASSOC_WORKERS = 1                                # associate keys built at a time (1 = today, serial: ops/p3s3_pass.sh:77-83); speed-up 11.1 (pure) allows 4 - memory of one build not measured [step]
    DEEP_LOG = 'incremental.log'                          # the pass log every stage appends to (people read it; the run record is the state); ops/p3s3_pass.sh [step]
    DEEP_PYTHON = 'default'                               # site PYTHON key (default/flow/py36/casa/tess) of every deep in-place call: py38_orca_nkosogor = PY8, ops/p3s3_pass.sh:13 [step; core.legacy.run]
    PARENTS_COMMAND = 'exopipe3 masters --tier deep --stages sources --cells {cells}'   # named by the preflight refusal when a cell has no catalogue on its current master (decision item 6) [step]
    BINDING_CHECK = True                                  # read every bound literal ([lit]) from the in-place modules at start and refuse on a mismatch (spec 1.3 item 5) [step]
    FORCED_STEP = 'forced'                                # the step whose done record (newer than every associate/fork record) `--from union` requires (D-G22) [step]


class catalog_merge(_Section):
    """harvest + merge gate + staleness gate (B13; ops/p3s3_harvest_published.sh, p3s3_merge.py, p3s3_stale_exclusions.py)."""
    HARVEST_STOKES = ('V', 'I')                           # ops/p3s3_pass.sh:38 [step]
    HARVEST_SSH_SLEEP_S = 1                               # s between remote rsyncs (connection-burst rule); ops/p3s3_harvest_published.sh:22 [step]
    HARVEST_DEST_ROOT = '/data07/gh/phase3/transients'    # where a pulled attempt lands (<root>/<label_lower>_collect_<node>/attempts/); ops/p3s3_harvest_published.sh:11 COLL [step]
    BACKING_CAP_DEG = 0.5                                 # deg, one-to-one nearest final-ledger row per (cell, Stokes, date); ops/p3s3_merge.py:21 [lit]; ops/p3s2_conform.py:95 (backing rule 2026-09-05; R-APPROVALS-20260904-2000UT)
    DUPLICATE_ROUND_DEG = 1e-4                            # deg, (cell, Stokes, date, round(ra,4), round(dec,4)) duplicate key; ops/p3s3_merge.py:137, p3s2_conform.py:474 (audit R-07)
    STALENESS_GATE = True                                 # refuse attempts that searched a now-excluded epoch; ops/p3s3_merge.py:24-33 (CORRECTION-ITEM6-...-STALE-ATTEMPTS-AFTER-EXCLUSION-EDITS, 2026-09-07)
    UNHARVESTED = 'warn'                                  # 'warn' (today, ops/p3s3_merge.py:34-37; counted in the record) | 'refuse' (B-30 a, not ruled) [step: the harvest stage refuses an unpullable attempt either way, D-G3]
    CADENCE_LABEL_REPAIR = {'P3T4': '10min'}              # rows of a 10-min generation relabelled; ops/p3s3_merge.py:145 [lit] (Gregg 2026-09-16 "do the second")
    COPY_CANDIDATE_DIRS = True                            # copytree of every attempt's candidates/<id>/; ops/p3s3_merge.py:53-58 (speed-up 62 waits; K10)
    ZERO_ACCEPTED_REFUSES = True                          # ops/p3s3_merge.py:159-160
    TEST_BACKING_KEYS = 40                                # (cell, Stokes, date) keys whose one-to-one backing the merged_record content test re-runs from the ledger (seeded sample) [step] (new: content test)
    PROTECTED_KEYS = ('BACKING_CAP_DEG', 'CADENCE_LABEL_REPAIR')   # window-bound
    RULED = {'BACKING_CAP_DEG': 'R-APPROVALS-20260904-2000UT',
             'STALENESS_GATE': 'CORRECTION-ITEM6-IS-EXCLUDED-CLAIM-WAS-WRONG-STALE-ATTEMPTS-AFTER-EXCLUSION-EDITS'}


class catalog_conform(_Section):
    """publish-side conformance (B13; ops/p3s2_conform.py) - deep, sub-LST and the fork's two calls."""
    PROTECTED_KEYS = ('SIZE_REMOVE_BEAMS', 'SIZE_FLAG_BEAMS', 'SIZE_SNR', 'NOMINAL_BEAM_ARCSEC', 'AXIS_REMOVE', 'AXIS_FLAG',
                      'SANITY_X', 'ROWLINE_MIN', 'DEEP_ELEV_FLOOR_DEG', 'JUPITER_TRACK_RADIUS_DEG')   # window-bound
    SIZE_STEP = {'deep': False, 'sublst': False, 'fork': False}   # module default True (ops/p3s2_conform.py:23); the pass passes 0: ops/p3s3_pass.sh:42, sublst_catalogs.sh:186, multiples_fork.py:133 [env:P3_CONFORM_SIZE=0]
    SIZE_REMOVE_BEAMS = 2.5                               # beams (nominal band beam), above SIZE_SNR; ops/p3s2_conform.py:22 [lit] (Gregg 2026-09-04 ~20:00 UT)
    SIZE_FLAG_BEAMS = 2.0                                 # beams; ops/p3s2_conform.py:22 [lit]
    SIZE_SNR = 8.0                                        # sigma, size/axis rules act above it; ops/p3s2_conform.py:22 [lit]
    NOMINAL_BEAM_ARCSEC = {'Red': 787.0, 'Green': 518.0, 'Blue': 374.0, 'Full': 528.0}   # arcsec; size_beams overwrite (D8, ruling pending: parity keeps it); ops/p3s2_conform.py:24 [lit]
    AXIS_REMOVE = 2.0                                     # x epoch-beam axis ratio, above SIZE_SNR; ops/p3s2_conform.py:105 [lit] (Gregg 2026-09-05)
    AXIS_FLAG = 1.5                                       # ops/p3s2_conform.py:105 [lit]
    SANITY_X = 3.0                                        # x cell median frame noise (I qa3_rms_mjy, V THERMNOI); ops/p3s2_conform.py:129 [lit] (Gregg 2026-09-05 ratified)
    SANITY_MIN_NIGHTS = 5                                 # nights needed for the median; ops/p3s2_conform.py:139, :157
    ROWLINE_MIN = 5                                       # finals on one pixel row +/-1; ops/p3s2_conform.py:129 [lit] (Gregg 2026-09-05)
    ROWLINE_MATCH_DEG = 0.05                              # deg, row-line member match; ops/p3s2_conform.py:505
    DEEP_ELEV_FLOOR_DEG = 25.0                            # deg at mid-hour (LST + 0.5 h); ops/p3s2_conform.py:186, gate_inputs.py:95 [lit; env:P3_DEEP_MIN_ELEVATION unset in production] (Gregg 2026-10-06 "3. (c)"; R-DEEP-SERVE-ELEVGATE-FLOOR25-INSEARCH-ELEV)
    DEEP_ELEV_EPOCH = 'mid-hour'                          # LST + 0.5 h, not the frame time (B-02 open); ops/p3s2_conform.py:187-189
    TABLE_MODE = {'deep': 'singles', 'sublst': 'singles', 'fork_validity': 'off', 'fork_table': 'multi'}   # ops/p3s2_conform.py:42 [lit: module default = deep], multiples_fork.py:133 [env:P3_CONFORM_TABLE]
    TABLE_ELEV = {'deep': True, 'sublst': True}           # elevation-band tables; ops/p3s2_conform.py:268 (deep on since 2026-10-07), fork_serve_p3s5_20261008.sh:31 and sublst_catalogs.sh:81 [env:P3_TABLE_ELEV=1] (Gregg 2026-10-04 "Yes on 1"; 2026-09-27 "1. go")
    TABLE_ELEV_BANDS = '25-35,35-50,50-90'                # deg; parent_curve_table.py:117 [env:P3_TABLE_ELEV_BANDS]
    TABLE_PREFIX = {'deep': '', 'sublst': 'sublst'}       # '' = cadence prefix; ops/sublst_catalogs.sh:81 [env:P3_TABLE_PREFIX] (RULINGS-GREGG-20261013-DECISIONS-1-7-AND-TENMIN-TABLES; ops 7042670)
    TABLE_RECORD_ONLY = False                             # ops/sublst_catalogs.sh:186 [env:P3_TABLE_RECORD_ONLY=0]; parent_curve_table.py:49
    TABLE_TARGET_BEAM = True                              # gate distance in the searched frame's beam; ops/p3s2_conform.py:234 [env:P3_GATE_TARGET_BEAM] (Gregg 2026-09-24)
    TABLE_XM_EXEMPT = True                                # parent_curve_table.py:90 [env:P3_TABLE_XM_EXEMPT] (R-TABLE-EXEMPTION-PULSARS-ONLY)
    JUPITER_TRACK_RADIUS_DEG = {'Red': 5.0, 'Full': 5.0, 'Green': 0.5, 'Blue': 0.5}   # deg; ops/p3s2_conform.py:288 [lit] (Gregg 2026-09-18, R-JUPITER-TRACK-VETO-IMPLEMENTED-IN-THE-CONFORMANCE-STAGE-BOTH-TIERS)
    JUPITER_TRACK_MIN_ALT_DEG = 20.0                      # deg, Jupiter below this at the cell's mid-hour is ignored; ops/p3s2_conform.py:303
    JUPITER_TRACK_STEP_H = 0.25                           # h, sampling of the night for the mid-hour; ops/p3s2_conform.py:299
    JUPITER_TRACK_NIGHTS = {'deep': 'in_master', 'sublst': 'all weighted'}   # ops/p3s2_conform.py:295-297
    DAYTIME_SAMPLES = (0.0, 0.25, 0.5, 0.75, 1.0)         # fractions of DATE-OBS..DATE-END; ops/p3s2_conform.py:342 (R-DAYTIME-GATE-DEEP-CONFORM, 2026-09-26/27)
    TWILIGHT_RISING_DEG = -12.0                           # deg Sun altitude, quarantined bands only; ops/p3s2_conform.py:346
    TWILIGHT_SETTING_DEG = -18.0                          # deg; ops/p3s2_conform.py:346
    PULSAR_ASSOC_K = 3.034                                # deep pulsar groups (sub-LST: k_pair at 99 per cent); ops/p3s2_conform.py:437 (R-ASSOCIATION-RADIUS-QUADRATURE)
    PULSAR_PRECUT_DEG = 1.5                               # deg pre-selection around each pulsar; ops/p3s2_conform.py:400
    PULSAR_STAT_COEFF = 0.42                              # x beam / max(|SNR|, PULSAR_SNR_FLOOR); ops/p3s2_conform.py:373 (Gregg 2026-09-28)
    PULSAR_SNR_FLOOR = 5.0                                # sigma; ops/p3s2_conform.py:373
    READMIT_V110_MARGINAL = {'deep': True, 'fork': False} # ops/p3s2_conform.py:68, :534-548 (R-V110-MARGINAL-RETAIN-AND-LABEL, Gregg 2026-09-04)
    RULED = {'DEEP_ELEV_FLOOR_DEG': 'R-DEEP-SERVE-ELEVGATE-FLOOR25-INSEARCH-ELEV',
             'JUPITER_TRACK_RADIUS_DEG': 'R-JUPITER-TRACK-VETO-IMPLEMENTED-IN-THE-CONFORMANCE-STAGE-BOTH-TIERS',
             'DAYTIME_SAMPLES': 'R-DAYTIME-GATE-DEEP-CONFORM', 'READMIT_V110_MARGINAL': 'R-V110-MARGINAL-RETAIN-AND-LABEL',
             'PULSAR_ASSOC_K': 'R-ASSOCIATION-RADIUS-QUADRATURE', 'TABLE_XM_EXEMPT': 'R-TABLE-EXEMPTION-PULSARS-ONLY'}


class catalog_fitpos(_Section):
    """fitted detection positions on the searched frame (R-GREGG-20261012-FITTED-DETECTION-POSITIONS-REQUIRED). Window-only stage:
    the target design fits inside `search` (B-31); K per tier/Stokes is fitting.K_TEMPLATE (bound to the in-place literal)."""
    PROTECTED_KEYS = ('ON_CENTRE_MAX_FRAC',)               # window-bound
    TIERS = ('deep', 'sublst')                            # ops/p3s3_pass.sh:48-52 (deep labels only), sublst_catalogs.sh:213-217 [step]
    NPROC = {'deep': 16, 'sublst': 16}                    # processes; ops/p3s3_pass.sh:49 (FITPOS_NPROC), sublst_catalogs.sh:215 (SUBLST_FITPOS_NPROC) [arg:--nproc]
    ON_CENTRE_PX = 0.01                                   # px, fitted position this close to a pixel centre counts; ops/deep_fit_positions.py:147, sublst_fit_positions.py:57
    ON_CENTRE_MAX_FRAC = 0.01                             # refuse above 1 per cent of fitted rows (chance 0.03 per cent); ops/deep_fit_positions.py:149 [lit]
    KEEP_DEEP_SYSTEMATIC = {'deep': True, 'sublst': False}   # sub-LST blanks pos_sys/pos_err (error model A, 2026-10-08/10); ops/sublst_fit_positions.py:56
    SUBLST_K_ARG = 'I=0.476,0.423 V=0.658,0.499'          # must equal fitting.K_TEMPLATE[('sublst', S)]; ops/sublst_catalogs.sh:213 [arg:--k] (R-GREGG-20261013-TENMIN-TEMPLATE-K-MEASURED)
    RULED = {'TIERS': 'R-GREGG-20261012-FITTED-DETECTION-POSITIONS-REQUIRED'}


class catalog_select(_Section):
    """selection (ops/select_v3.py), ephemeris filter (ops/p3s2_ephem_filter.py), split by Stokes and tier (ops/p3s3_pass.sh PYS heredoc)."""
    PROTECTED_KEYS = ('POLICY', 'RADOP_ARCMIN', 'BEAM_DEFAULT_ARCSEC', 'EPHEM_JUPITER_DEG')   # window-bound
    SNR_MIN = 5.0                                         # sigma, |SNR| kept; ops/select_v3.py:142
    COUNTERPART_SNR_MIN = 6.0                             # sigma, multi-band override index; ops/select_v3.py:159
    PARENT_EXCLUSION = False                              # no page-side parent exclusion; ops/p3s3_pass.sh:57, sublst_catalogs.sh:219 [env:P3_NO_PARENT_EXCL=1] (R-P3S2-PAGE-STANDARDS)
    POLICY = {'Red': (10.0, 2.0), 'Green': (5.0, 1.5), 'Blue': (None, None), 'Full': (10.0, 1.5)}   # (Jy floor, beams); still used by the builder's footprint (B-36); ops/select_v3.py:25-26 [lit]
    RADOP_ARCMIN = {'Red': 8.0, 'Green': 6.6, 'Blue': 1.3, 'Full': 7.8}   # arcmin; ops/select_v3.py:33 [lit]
    BEAM_DEFAULT_ARCSEC = 900.0                           # arcsec; ops/select_v3.py:40, build_source_catalogs_v3.py:215 [lit] (B-32: no ruling for deep)
    VERDICT_INPUT = 'master_catalog.csv'                  # gate verdicts joined from the conformant record (M8, ops f15c489); ops/p3s3_pass.sh:57 [arg]
    EPHEM_JUPITER_DEG = {'Red': 5.0, 'Green': 0.5, 'Blue': 0.5, 'Full': 5.0}   # deg; ops/p3s2_ephem_filter.py:12 [lit] (Card 7a)
    EPHEM_MOON_DEG = 0.25                                 # deg; ops/p3s2_ephem_filter.py:57
    EPHEM_STOKES = ('I',)                                 # only I rows without jupiter_deg; ops/p3s2_ephem_filter.py:49
    EPHEM_TIME_FALLBACK = 'lst-approx'                    # counted fallback when no header DATE-OBS; ops/p3s2_ephem_filter.py:38-39 (B-30)
    SPLIT = {'deep': {'wb': ('Red', 'Green', 'Blue', 'Full'), 'sb': 'MHz'}, 'sublst': {'wb': ('Red', 'Green', 'Blue', 'Full'), 'sb': 'MHz'}}   # 'MHz' in band = subband tier; ops/p3s3_pass.sh:72, sublst_catalogs.sh:226-228 [step]
    DROP_EMPTY_TIER_DIR = True                            # a tier with < 1 selected row: its catalogue dir removed (moved aside by the step); ops/p3s3_pass.sh:80 [step]
    RULED = {'PARENT_EXCLUSION': 'R-P3S2-PAGE-STANDARDS'}


class catalog_associate(_Section):
    """per-source catalogues: association, identity merge, E statistic, promotion, unique names (B14; ops/build_source_catalogs_v3.py).
    K_ASSOC, STAT_COEFF and SYS_SCALE are config.astrometry (the binding check asserts the builder's literals equal them)."""
    PROTECTED_KEYS = ('SYS_FLOOR_ARCSEC', 'RPT_PCT', 'SURVIVOR_SNR', 'E_TARGET', 'PRIORITY_SAMPLES', 'SIZE_GATE_7_8')   # window-bound
    STAT_SNR_FLOOR = 5.0                                  # sigma; statistical error 0.42 x beam / max(|SNR|, 5) for unfitted rows; ops/build_source_catalogs_v3.py:504
    SYS_FLOOR_ARCSEC = 0.0                                # arcsec, no floor (Gregg 2026-09-24); ops/build_source_catalogs_v3.py:492 [lit; env:P3_SYS_FLOOR_AS unset]
    SYS_PROVISIONAL_ARCSEC = {'Red': 120.0, 'Green': 60.0, 'Blue': 40.0, 'Full': 65.0}   # arcsec; env override path only, unset in production; ops/build_source_catalogs_v3.py:528
    RPT_PCT = 99.0                                        # percentile, repeat consistency = association; ops/build_source_catalogs_v3.py:493 [lit]
    SURVIVOR_SNR = 7.0                                    # sigma, survivors = |best_snr| >= this or promoted; ops/build_source_catalogs_v3.py:127 [lit] (contract 48)
    MASTER_LIST_SNR = 5.0                                 # sigma, <band>_sources_master.csv; ops/build_source_catalogs_v3.py:1136
    E_TARGET = 20.0                                       # expected chance load defining per-sample floors; ops/build_source_catalogs_v3.py:128 [lit] (contract 47)
    E_FLOOR_SEARCH = (5.0, 9.0, 0.01)                     # sigma start, stop, step; ops/build_source_catalogs_v3.py:925-927
    E_PROMOTE = 0.5                                       # promotion when the source's own E < this; ops/build_source_catalogs_v3.py:977, :995-997 (CARD_XMATCH_STATISTICS_20260906)
    XM_OFFSET_PCT_MAX = 99.0                              # percentile; ops/build_source_catalogs_v3.py:977, :982
    PRIORITY_SAMPLES = ('Hot_Warm_Jupiters', '10pc')      # ops/build_source_catalogs_v3.py:130 [lit] (Gregg 2026-09-06 'All approved')
    RPT_MB_SNR = (6.0, 7.0)                               # sigma band of the retained RPT / MB rungs; ops/build_source_catalogs_v3.py:985 (2026-08-24 'not rescinded'; B-69 open)
    SAMPLE_DEDUP_ARCSEC = 10.0                            # arcsec, nested distance samples; ops/build_source_catalogs_v3.py:243, :250
    FOOTPRINT_NMC = 4000                                  # Monte-Carlo points; ops/build_source_catalogs_v3.py:302
    FOOTPRINT_SEED = 20260828                             # ops/build_source_catalogs_v3.py:301
    SKY_DEG2 = 41253.0                                    # deg^2; ops/build_source_catalogs_v3.py:309
    REPEAT_ELEV_FLOOR_DEG = 20.0                          # deg, repeat-significance cap and M (M13: for Gregg; deep floor is 25); ops/build_source_catalogs_v3.py:850, :860
    REPEAT_EPOCH_AREA_DEG2 = 13570.0                      # deg^2 cap at 20 deg (finding: divided by the SUM of caps, open); ops/build_source_catalogs_v3.py:851
    SIDEREAL_RATE_ARCSEC_S = 15.041                       # arcsec/s; ops/build_source_catalogs_v3.py:333
    MI_CONSTRAIN_K = 3.0                                  # drift > 3 x pair statistical error (Gregg 2026-09-21 item 4); ops/build_source_catalogs_v3.py:379
    SSTACK_THRESHOLD_FALLBACK = 7.0                       # sigma; used only if search_i_transients.CONFIG is not importable; ops/build_source_catalogs_v3.py:615-617 (B-30)
    SIZE_GATE_7_8 = 1.7                                   # beams, unpromoted singles at 7-8 sigma; ops/build_source_catalogs_v3.py:1053 [lit; env:P3_SIZE_GATE_7_8 unset] (Gregg 2026-09-09; R-SIZE-AXIS-GATE-1.7-AT-7-8-SIGMA-UNPROMOTED-SINGLES)
    BEAM_STRICT = {'deep': False, 'sublst': True}         # missing master beam refuses (sub-LST) or 900 arcsec (deep); ops/build_source_catalogs_v3.py:209-213 [env:P3_BEAM_STRICT] (RULING-GREGG-20261010-TENMIN-RADIUS-PERBIN-AND-SIMBAD-BAR item 3)
    TIER_TAG = {'deep': 'deep', 'sublst': '10min'}        # multi-integration tier; ops/build_source_catalogs_v3.py:334 [env:P3_TIER] (deep: unset, inferred from the label)
    FLUX_CORR_TABLE = {'deep': '', 'sublst': 'by-bake'}   # '' = none (deep masters carry FLUXBAKE); sub-LST v1 table only for unbaked frames; ops/build_source_catalogs_v3.py:1090-1125 [env:P3_FLUX_CORR_TABLE]
    PAIR_DIAG = (10.0, 300.0, 1.665)                      # arcmin cluster, arcsec core, median->sigma; diagnostic only; ops/build_source_catalogs_v3.py:560-569
    RULED = {'SIZE_GATE_7_8': 'R-SIZE-AXIS-GATE-1.7-AT-7-8-SIGMA-UNPROMOTED-SINGLES', 'E_PROMOTE': 'R-XMATCH-STATISTICS-AND-PRIORITY-RUNG',
             'REPEAT_EPOCH_AREA_DEG2': 'R-REPEAT-SIGNIFICANCE-CARD-APPROVED'}


class catalog_fork(_Section):
    """the multiples fork (ops/multiples_fork.py; R-FORKED-CHAIN-V1-ON-TODAYS-DATA-CUSTOM-MULTIPLES-TABLES, R-MULTIPLES-FORK-INTO-PRODUCTION-PREAMBLE-APPROVED)."""
    PROTECTED_KEYS = ('TABLES_SOURCE_DIR',                 # spec 3 (calibrate's input)
                      'SNR_MIN', 'MEMBER_SNR', 'MATCH_BEAMS', 'N_RANDOM', 'NPROC', 'SHUFFLE_SEED', 'SHUFFLE_K')   # window-bound
    TIERS = ('deep',)                                     # P3T* labels skip it; ops/p3s3_pass.sh:89 [step]
    STOKES = ('V', 'I')                                   # V first (the install lock is released after the I content test); ops/p3s3_pass.sh:90 [env:P3S3_FORK_STOKES -> --stokes]
    SNR_MIN = {'I': 7.0, 'V': 5.0}                        # sigma, fork-position rows; ops/multiples_fork.py:52 [lit]
    MEMBER_SNR = (5.0, 7.0)                               # sigma, Stokes I members (REPEAT5); ops/multiples_fork.py:53 [lit]
    MATCH_BEAMS = 1.0                                     # beam widths, repeat class partner radius; ops/multiples_fork.py:54 [lit]
    N_RANDOM = 1500                                       # random calibration discs per band; ops/multiples_fork.py:55 [lit; env:FORK_N_RANDOM]
    NPROC = 10                                            # processes; ops/multiples_fork.py:56, fork_serve_p3s5_20261008.sh:32 [lit; env:FORK_NPROC]
    TABLES_SOURCE_DIR = '/fast/gh/main/phase3/multiples_tables_20261008'   # 'tables' subcommand only (lwacalim07 /fast; -> calibrate); ops/multiples_fork.py:58 [lit; env:FORK_TABLES_DIR] (B-66)
    SHUFFLE_SEED = 20260906                               # ops/p3s3_repeat_shuffle.py:15 [lit]
    SHUFFLE_K = 3.034                                     # ops/p3s3_repeat_shuffle.py:15 [lit] (K_ASSOC copy, DUP-1)
    ROWLINE_MIN_FROM = 'phase3_deep.ROWLINE_MIN'          # ops/multiples_fork.py:175
    RULED = {'TIERS': 'R-MULTIPLES-FORK-INTO-PRODUCTION-PREAMBLE-APPROVED',
             'SNR_MIN': 'R-FORKED-CHAIN-V1-ON-TODAYS-DATA-CUSTOM-MULTIPLES-TABLES'}


class catalog_identity(_Section):
    """union of the primary tiers (ops/p3s3_pass.sh PYU0, ported), cross-tier identity (ops/p3s3_identity.py)."""
    PROTECTED_KEYS = ('IDENTITY_K',)                       # window-bound
    UNION_TIERS = ('V_wb', 'V_sb', 'I_wb', 'I_sb')        # four primary tiers only, in this order (D9, ops 09995b5); ops/p3s3_pass.sh:111 [step]
    UNION_KINDS = ('sources_survivors', 'sources_master', 'source_members')   # ops/p3s3_pass.sh:113 [step]
    IDENTITY_K = 3.034                                    # ops/p3s3_identity.py:20 [lit] (R-CROSS-TIER-SINGLE-IDENTITY, R-ASSOCIATION-RADIUS-QUADRATURE)
    IDENTITY_ERR_DEFAULT_ARCSEC = 60.0                    # arcsec, used when pos_err_arcsec is blank or 0; ops/p3s3_identity.py:37 (B-30)
    IDENTITY_SKIP_DIRS = ('source_catalogs_I', 'source_catalogs_V')   # served views never read; ops/p3s3_identity.py:28-30
    RULED = {'IDENTITY_K': 'R-CROSS-TIER-SINGLE-IDENTITY',
             'UNION_TIERS': 'FIX-REFACTOR-DEFECTS-D2-D6-D7-D9-D10-AND-RULINGS-D1-D5-D8-20261013'}


class catalog_localize(_Section):
    """combined positions per identity (ops/build_localizations.py, source_localization.py) and the deep transient list."""
    PROTECTED_KEYS = ('RHO_EPOCH', 'RHO_FAR', 'LIST_TIERS', 'LIST_PORTS')   # window-bound
    RHO_EPOCH = 0.187                                     # same-epoch correlation; ops/source_localization.py:59 [lit; env:P3_RHO_EPOCH] (Gregg 2026-09-27)
    RHO_FAR = 0.017                                       # between epochs; ops/source_localization.py:60 [lit; env:P3_RHO_FAR]
    CATS = {'deep': 'source_catalogs_[IV]_*', 'sublst': 'source_catalogs_{build}_[IV]_[sw]b', 'snapshot': 'source_catalogs_[IV]_locin_{tier}'}   # ops/build_localizations.py:21, sublst_catalogs.sh:268, snapshot_localize_tiers.sh:73 [env:P3_LOC_CATS]
    COMMON_FRAME = {'deep': False, 'sublst': True, 'snapshot': False}   # ops/sublst_catalogs.sh:44 [env:P3_LOC_COMMON_FRAME] (RULING-GREGG-20261008-ASTROMETRY-MEASURE-DONT-MOVE-BOTH-TIERS item 4)
    ERRORS_RAISE = {'deep': False, 'sublst': True, 'snapshot': False}   # deep drops an identity whose localization fails (B-32 open); ops/build_localizations.py:27-34
    SNAPSHOT_TIERS = ('10s', '1min')                      # ops/snapshot_localize_tiers.sh:18 [step]
    SNAPSHOT_ALLOW_MODEL_ERRORS = False                   # ops/snapshot_localize_tiers.sh:61 [env:P3_LOC_ALLOW_MODEL_ERRORS] (Gregg 2026-10-07 item 24)
    LIST_TIERS = ('V_wb', 'V_sb', 'I_wb', 'I_sb', 'V_fwb', 'V_fsb', 'I_fwb', 'I_fsb')   # ops/p3s2_transient_list.py:30 [lit] (M15, ops 0c1580c)
    LIST_PORTS = {('V', False): 8766, ('I', False): 8768, ('V', True): 8772, ('I', True): 8773}   # must equal serve ports; ops/p3s2_transient_list.py:31 [lit]
    LIST_NAME = 'TRANSIENT_LIST_{label}_{utc}.csv'        # utc = %Y%m%dT%H%M; symlink TRANSIENT_LIST_{label}_latest.csv; ops/p3s3_pass.sh:136 [step]
