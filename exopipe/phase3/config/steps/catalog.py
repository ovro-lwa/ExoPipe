"""Configuration of `exopipe3 catalog` - every tunable of the command, one section per stage or tier.

Units folded here: catalog-deep, catalog-incremental, catalog-tiers. A key in its section's PROTECTED_KEYS is locked
(a mirror of in-place code, a pinned input or a ruling-fixed value: it changes only by a commit); RULED names the
ledger ID of the ruling behind a key. Per run: --set SECTION.KEY=VALUE or --config FILE; `exopipe3 config --step catalog` lists the sections.
"""
from exopipe.phase3.config import _Section


# =====================================================================================================================
# unit catalog-deep (was exopipe/phase3/_cfgdraft_catalog-deep.py)
# Unit catalog-deep: the config sections of `exopipe3 catalog` shared by every tier, and the deep ones.
#
# Source: plan/specs/catalog.md section 3 (values = production values at exopipe_ops 0c1580c / clv15 9b6525d / exopipe
# 03ca797; each line names its source file:line and its ruling). Paths named 'ops' = /lustre/gh/main/exopipe_ops,
# 'clv' = /lustre/gh/main/clv15.
#
# Binding of each key during the refactor window (DESIGN 11.1: the step calls clv15 / exopipe_ops modules IN PLACE):
#   [env:X]   passed to the in-place module as environment variable X (the module reads it today)
#   [arg:--x] passed as that command-line argument
#   [lit:F:N] a literal in the in-place module F at line N. lib/catalog/deep_binding.BINDINGS reads each such literal from
#             the module's source (AST, nothing imported) at the start of every deep run and REFUSES when it differs from the
#             config key (spec 1.3 item 5). The bound keys of these sections are in their PROTECTED_KEYS ("window-bound"): an
#             override would change nothing while the module carries its own literal, and the record would lie. A bound key of
#             a shared section (astrometry, fitting, paths) is refused by the step at its first unit.
#   [step]    used by the step's own (ported) code (steps/catalog.py, lib/catalog/deep_*.py)
# Protection uses the section metas the config machinery honours (refactor decisions 2026-10-13, _CROSSCHECK.md item 4): PROTECTED_KEYS
# (no --set / --config) and RULED (override only with --ruling-for or into a non-production --out).
# Shared roots are NOT repeated here (decision item 3): CFG.paths.TRANSIENTS_ROOT (deep run dirs), COADD_SB_ROOT (masters),
# PHASE3_ARCHIVE_ROOT + ARCHIVE_RUN (searched frames), CLV15_ROOT (sample catalogues), OPS_ROOT (search state); the
# exopipe_ops root of the in-place modules is core.env.legacy_root('exopipe_ops') and the interpreter is the site PYTHON key
# catalog.DEEP_PYTHON. Parent catalogues are built ONLY by `exopipe3 masters --tier deep --stages sources` (decision item 6):
# catalog preflight assembles the per-pass snapshot of the current ones and refuses naming that command.
#
# Sections the TIER modules own (catalog_sublst, catalog_sublst_tie, catalog_sublst_veto, catalog_snapshot) are NOT here:
# they belong to config/steps/catalog.py. Per-tier dicts below carry the sub-LST / snapshot entries of the shared
# sections (spec section 3), so the tier draft must not redefine these sections.
# =====================================================================================================================

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
    DEEP_STAGES = ('preflight', 'harvest', 'merge', 'pulsars', 'conform', 'fitpos', 'select', 'associate', 'fork', 'union',
                   'identity', 'localize', 'list')        # pipeline order (spec 1.2); ops/p3s3_pass.sh:38-136 + fork_serve_p3s5_20261008.sh:20-30 [step];
                                                          # 'pulsars' = B-02 step 4 (catalog_conform.MODE 'check' only; no-input under 'remove')
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
    UNHARVESTED = 'refuse'                                # B-30 (a): a published attempt the merge finds unharvested (its rows would be left out
                                                          # of the merged catalogue) refuses; 'warn' = production (ops/p3s3_merge.py:34-37, the
                                                          # rows dropped with a printed warning; counted in the record) [step: the harvest stage
                                                          # refuses an unpullable attempt and the merge a changed published list, D-G3]
    CADENCE_LABEL_REPAIR = {'P3T4': '10min'}              # rows of a 10-min generation relabelled; ops/p3s3_merge.py:145 [lit] (Gregg 2026-09-16 "do the second")
    COPY_CANDIDATE_DIRS = True                            # copytree of every attempt's candidates/<id>/; ops/p3s3_merge.py:53-58 (speed-up 62 waits; K10)
    ZERO_ACCEPTED_REFUSES = True                          # ops/p3s3_merge.py:159-160
    TEST_BACKING_KEYS = 40                                # (cell, Stokes, date) keys whose one-to-one backing the merged_record content test re-runs from the ledger (seeded sample) [step] (new: content test)
    PROTECTED_KEYS = ('BACKING_CAP_DEG', 'CADENCE_LABEL_REPAIR')   # window-bound
    RULED = {'BACKING_CAP_DEG': 'R-APPROVALS-20260904-2000UT',
             'STALENESS_GATE': 'CORRECTION-ITEM6-IS-EXCLUDED-CLAIM-WAS-WRONG-STALE-ATTEMPTS-AFTER-EXCLUSION-EDITS',
             'UNHARVESTED': 'R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32'}


class catalog_conform(_Section):
    """publish-side conformance (B13; ops/p3s2_conform.py) - deep, sub-LST and the fork's two calls."""
    PROTECTED_KEYS = ('SIZE_REMOVE_BEAMS', 'SIZE_FLAG_BEAMS', 'SIZE_SNR', 'NOMINAL_BEAM_ARCSEC', 'AXIS_REMOVE', 'AXIS_FLAG',
                      'SANITY_X', 'ROWLINE_MIN', 'DEEP_ELEV_FLOOR_DEG', 'JUPITER_TRACK_RADIUS_DEG')   # window-bound
    SIZE_STEP = {'deep': False, 'sublst': False, 'fork': False}   # module default True (ops/p3s2_conform.py:23); the pass passes 0: ops/p3s3_pass.sh:42, sublst_catalogs.sh:186, multiples_fork.py:133 [env:P3_CONFORM_SIZE=0]
    SIZE_REMOVE_BEAMS = 2.5                               # beams (nominal band beam), above SIZE_SNR; ops/p3s2_conform.py:22 [lit] (Gregg 2026-09-04 ~20:00 UT)
    SIZE_FLAG_BEAMS = 2.0                                 # beams; ops/p3s2_conform.py:22 [lit]
    SIZE_SNR = 8.0                                        # sigma, size/axis rules act above it; ops/p3s2_conform.py:22 [lit]
    NOMINAL_BEAM_ARCSEC = {'Red': 787.0, 'Green': 518.0, 'Blue': 374.0, 'Full': 528.0}   # arcsec; ops/p3s2_conform.py:24 [lit] - UNUSED since D8 (Gregg 2026-10-13 'Approved', ops 283a513): conformance keeps the search's own epoch-beam size_beams/axis_norm; a row without them gets FWHM / its searched frame's BMAJ (frame_beam_bmaj.csv); c50_mjy blanked on every conformed row (M6)
    AXIS_REMOVE = 2.0                                     # x epoch-beam axis ratio, above SIZE_SNR; ops/p3s2_conform.py:105 [lit] (Gregg 2026-09-05)
    AXIS_FLAG = 1.5                                       # ops/p3s2_conform.py:105 [lit]
    SANITY_X = 3.0                                        # x cell median frame noise (I qa3_rms_mjy, V THERMNOI); ops/p3s2_conform.py:129 [lit] (Gregg 2026-09-05 ratified)
    SANITY_MIN_NIGHTS = 5                                 # nights needed for the median; ops/p3s2_conform.py:139, :157
    ROWLINE_MIN = 5                                       # finals on one pixel row +/-1; ops/p3s2_conform.py:129 [lit] (Gregg 2026-09-05)
    ROWLINE_MATCH_DEG = 0.05                              # deg, row-line member match; ops/p3s2_conform.py:505
    DEEP_ELEV_FLOOR_DEG = 25.0                            # deg, at the time DEEP_ELEV_EPOCH names; ops/p3s2_conform.py:186, gate_inputs.py:95 [lit; env:P3_DEEP_MIN_ELEVATION unset in production] (Gregg 2026-10-06 "3. (c)"; R-DEEP-SERVE-ELEVGATE-FLOOR25-INSEARCH-ELEV)
    MODE = 'check'                                        # B-02 (a) steps 2 and 4: the publish conformance only CHECKS (lib/catalog/
                                                          # conform_check.check: every removal rule re-evaluated both ways, any finding
                                                          # refuses) after the pulsar stage (conform_check.pulsars); 'remove' = production's
                                                          # ops/p3s2_conform.py (Run_P3S5 until its full re-search; the parity file)
    DEEP_ELEV_EPOCH = 'frame-mid'                         # the frame's own mid-time (B-02 step 3b: the search's floor and the check;
                                                          # R-REFACTOR-20261016-B02-CONVENTIONS (2)); production 'mid-hour' = LST + 0.5 h,
                                                          # ops/p3s2_conform.py:187-189 (the parity file)
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
    RULED = {'MODE': 'R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32',
             'DEEP_ELEV_FLOOR_DEG': 'R-DEEP-SERVE-ELEVGATE-FLOOR25-INSEARCH-ELEV',
             'JUPITER_TRACK_RADIUS_DEG': 'R-JUPITER-TRACK-VETO-IMPLEMENTED-IN-THE-CONFORMANCE-STAGE-BOTH-TIERS',
             'DAYTIME_SAMPLES': 'R-DAYTIME-GATE-DEEP-CONFORM', 'READMIT_V110_MARGINAL': 'R-V110-MARGINAL-RETAIN-AND-LABEL',
             'PULSAR_ASSOC_K': 'R-ASSOCIATION-RADIUS-QUADRATURE', 'TABLE_XM_EXEMPT': 'R-TABLE-EXEMPTION-PULSARS-ONLY',
             'DEEP_ELEV_EPOCH': ('R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32',   # B-02 (a); the convention
                                 'R-REFACTOR-20261016-B02-CONVENTIONS')}                        # under the 2026-10-16 delegation


class catalog_fitpos(_Section):
    """fitted detection positions on the searched frame (R-GREGG-20261012-FITTED-DETECTION-POSITIONS-REQUIRED). Window-only stage:
    the target design fits inside `search` (B-31); K per tier/Stokes is fitting.K_TEMPLATE (bound to the in-place literal)."""
    PROTECTED_KEYS = ('ON_CENTRE_MAX_FRAC',)               # window-bound
    TIERS = ('deep', 'sublst')                            # ops/p3s3_pass.sh:48-52 (deep labels only), sublst_catalogs.sh:213-217 [step]
    NPROC = {'deep': 16, 'sublst': 16}                    # processes; ops/p3s3_pass.sh:49 (FITPOS_NPROC), sublst_catalogs.sh:215 (SUBLST_FITPOS_NPROC) [arg:--nproc]
    ON_CENTRE_PX = 0.01                                   # px, fitted position this close to a pixel centre counts; ops/deep_fit_positions.py:147, sublst_fit_positions.py:57
    ON_CENTRE_MAX_FRAC = 0.01                             # refuse above 1 per cent of fitted rows (chance 0.03 per cent); ops/deep_fit_positions.py:149 [lit]
    KEEP_DEEP_SYSTEMATIC = {'deep': True, 'sublst': False}   # sub-LST blanks pos_sys/pos_err (error model A, 2026-10-08/10); ops/sublst_fit_positions.py:56
    SUBLST_K_ARG = 'I=0.464,0.450 V=0.599,0.602'          # must equal fitting.K_TEMPLATE[('sublst', S)]; ops/sublst_catalogs.sh:216 (ac2d267) [arg:--k] (R-GREGG-20261013-TENMIN-TEMPLATE-K-MEASURED)
    # a fitted row (gaussian / template) whose written statistical error is not positive - e.g. 0.0 for a Gaussian with an absurd
    # fitted S/N (2026-10-14: 2 rows of Run_P3S5, S/N 26,889 and 47,189) - fails the fitted_catalog content test ('refuse': B-32,
    # each detection's fitted error) or is reported only ('report' = production: the builder and source_localization take 0.0)
    ZERO_STAT_ERROR = 'refuse'
    RULED = {'TIERS': 'R-GREGG-20261012-FITTED-DETECTION-POSITIONS-REQUIRED', 'ZERO_STAT_ERROR': 'R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32'}


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
    REPEAT_ELEV_FLOOR_DEG = {'deep': 25.0, 'sublst': 25.0}   # deg, repeat-significance covering-epoch count M, per-epoch cap and shuffle cap = the tier's ruled floor (M13(a), Gregg 2026-10-13 '(a)', ops e46bd45; was 20): ops/build_source_catalogs_v3.py:338 EFLOOR = gate_inputs.DEEP_ELEV_FLOOR_DEG / TENMIN_ELEV_FLOOR_DEG (bound via catalog_conform.DEEP_ELEV_FLOOR_DEG), p3s3_repeat_shuffle.py:19
    REPEAT_EPOCH_AREA_DEG2 = 11909.36                     # deg^2 = SKY_DEG2 x (1 - sin floor) / 2 at the 25-deg floor (13,572 at the old 20 deg); ops/build_source_catalogs_v3.py:339 _CAP_EL (e46bd45); the per-epoch area is still divided by the SUM of caps (finding (b), not ruled, open)
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


# =====================================================================================================================
# unit catalog-incremental (was exopipe/phase3/_cfgdraft_catalog-incremental.py)
# Unit catalog-incremental: `exopipe3 catalog --tier deep --incremental` (DESIGN 12.1, 12.3).
#
# ONE section, catalog_incremental. It holds
# only what is NEW to the incremental path. Every value the full deep chain also uses is read from the step owner's
# sections, so each value lives in one place (lib/catalog/incremental.py, `_deep()`):
#   label            catalog.DEEP_RUN_LABEL (when --run is not given)
#   run root         paths.TRANSIENTS_ROOT (shared root, _CROSSCHECK decision 3)
#   stage env        P3_CONFORM_SIZE from catalog_conform.SIZE_STEP['deep'], P3_TABLE_ELEV from catalog_conform.TABLE_ELEV['deep'],
#                    P3_NO_PARENT_EXCL from catalog_select.PARENT_EXCLUSION, P3_CADENCE=deep (fork_serve_p3s5_20261008.sh:31-32,
#                    p3s3_pass.sh:42, :57)
#   processes        catalog_fitpos.NPROC['deep'] (p3s3_pass.sh:49), catalog_fork.NPROC (fork_serve_p3s5_20261008.sh:32)
#   fork Stokes      catalog_fork.STOKES (p3s3_pass.sh:90)
#   union            catalog_identity.UNION_TIERS / UNION_KINDS (p3s3_pass.sh:111-113; the port is lib/catalog/deep_ported.py)
# In-place scripts are named relative to the site legacy root 'exopipe_ops' (core.env.legacy_root) and run through
# core.legacy.run with the site interpreter 'default' (_CROSSCHECK decisions 1-2).
# Tags: [step] used by lib/catalog/incremental.py; [ast] read from the live file by AST, never copied; [arg]/[env] passed
# to the in-place module.
# =====================================================================================================================

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


# =====================================================================================================================
# unit catalog-tiers (was exopipe/phase3/_cfgdraft_catalog-tiers.py)
# The catalog-tiers unit (exopipe3 catalog --tier sublst | snapshot).
#
# Values are the
# production values at exopipe_ops 0c1580c / clv15 39bdfd3 (spec plan/specs/catalog.md section 3, copied verbatim, plus the
# path templates and interpreter roles the tier modules need, each with its source line).
#
# Binding of each key during the window (the step calls clv15 / exopipe_ops modules IN PLACE, DESIGN 11.1):
#   [env:X]   passed to the in-place module as environment variable X
#   [arg:--x] passed as that command-line argument (passed only when it differs from the module's own default, so the
#             command line equals production's while the values agree)
#   [lit:F:N] a literal in the in-place module F at line N: the tier module reads it at start (AST, no import) and REFUSES on
#             a difference; a --set of a [lit] key REFUSES until the module is absorbed (it would change nothing)
#   [step]    used by the tier module's own (ported) code
# ``PROTECTED_KEYS`` (honoured by the config machinery since 2337938) lists the keys no --set / --config may change: the pinned inputs of the
# spec's lower-case ``protected`` tuples AND every [lit] key (an override of a value the in-place module holds as a literal would
# change nothing and make the record lie, spec section 3). ``RULED`` names the ledger ID of each ruled key (checked IDs).
# Shared roots are config.paths (SUBLST_ROOT, SNAPSHOT_FLEET_OUT, SNAPSHOT_REVIEW_ROOT, OPS_ROOT, CLV15_ROOT, ...; refactor chat
# decision 3) - no key here repeats one. Interpreters are the site file's PYTHON keys default / flow / py36 (decision 2), named in
# the tier modules, not here.
# Ownership (refactor chat decision 6): sub-LST parent catalogues are built by `masters --tier sublst --stages sources` (catalog
# preflight only checks them); snapshot per-tier localization is `serve`'s in the window; the fit-store wideband quarantine is
# `archive quarantine-rows` (catalog's store stage refuses unmarked rows); sub-LST tests (a)/(b) are masters/frames classes.
# Paths: 'ops' = /lustre/gh/main/exopipe_ops, 'clv' = /lustre/gh/main/clv15.
#
# Shared values NOT repeated here (they live in the catalog-deep draft's sections, per spec section 3): catalog.TABLES_SUBLST,
# catalog.FOOTPRINT_CAT_DIR['sublst'], catalog_conform.{TABLE_PREFIX,TABLE_ELEV,TABLE_RECORD_ONLY,SIZE_STEP}['sublst'],
# catalog_fitpos.{NPROC['sublst'],SUBLST_K_ARG}, catalog_select.PARENT_EXCLUSION, catalog_localize.{CATS,COMMON_FRAME,
# SNAPSHOT_TIERS,SNAPSHOT_ALLOW_MODEL_ERRORS}, fitting.K_TEMPLATE[('sublst', S)]. The tier modules read them from there and
# REFUSE, naming the key, when one is missing.
# =====================================================================================================================

class catalog_sublst(_Section):
    """sub-LST catalogue chain (sublst_catalog_chain.sh, sublst_catalogs.sh stages, sublst_catalog_publish.sh part 2)."""
    # pinned inputs (spec 'protected') + every [lit] key (bound to the in-place literal at preflight)
    PROTECTED_KEYS = ('TIER_RULE3_PIN', 'TIER_RULE3_PIN_MD5', 'COADD_RECORDS_FROZEN', 'MEMBERSHIP_VETO_PIN',
                      'MEMBERSHIP_VETO_PIN_MD5', 'FLUX_V1_TABLE', 'TIER_RULE4_RANGE', 'ELEV_FLOOR_DEG', 'REJECTED_SYS_ARCSEC',
                      'REJECTED_BEAM_ARCSEC')
    # the sub-LST root is config.paths.SUBLST_ROOT (/lustre/gh/main/phase3/sublst; sublst_catalogs.sh:22)
    ARM_TREES = {'A': 'tenmin_root', 'B': 'tenmin_root_x2', 'BI': 'tenmin_root_x2_inj'}   # under paths.SUBLST_ROOT; sublst_catalogs.sh:20 [env:P3_TENMIN_ROOT]
    ARM = 'B'                                             # served arm; sublst_catalog_publish.sh:38, sublst_resume_from_catalogs.sh:21 [step]
    CAT_ROOT = '/fast/gh/main/phase3/sublst_catalogs'     # node-local catalogue dirs catalog_/collect_/state_<SFX>; sublst_catalogs.sh:25 [step]
    LABEL = 'SLST{hh}{arm}'                               # sublst_catalogs.sh:24 [env:P3S_LABEL]
    SFX_19H_LEGACY = True                                 # 19h names omit the LST (E3); sublst_catalogs.sh:26, sublst_catalog_chain.sh:9 [step]
    SEARCH_STATE = '{ops}/sublst_search_state_{arm}'      # done/claims/failed under paths.OPS_ROOT; sublst_catalog_chain.sh:10 [step] -> search run records
    EXPECTED_JOBS = (36, 2)                               # 36 jobs less 2 per subband with NO_WINDOWS.txt; sublst_catalog_chain.sh:17-18 [step]
    CHAIN_MEM_GB = 100                                    # GB free in /user.slice; sublst_catalog_chain.sh:27 [step] -> runner admission --mem-gb
    CHAIN_MEM_WAIT_S = 43200                              # s; sublst_catalog_chain.sh:27 [step]
    FLUX_V1_TABLE = '/lustre/gh/main/phase3/sublst/catalog_inputs/shflux/flux_correction_v1_20260911.csv'   # unbaked frames only; sublst_catalogs.sh:68 [env:P3_FLUX_CORR_TABLE] (Gregg 2026-10-07 Q10)
    FLUX_BAKE_POLICY = 'branch'                           # 'branch' = baked/unbaked/mixed-refuses (today; B-72 b) | 'baked-only' (B-72 a); sublst_catalogs.sh:58-70 [step]
    TABLES_REQUIRED_METRIC = 'image-pixel'                # every sublst_ table key, elevation bands present; sublst_catalogs.sh:182-183 [step] (Gregg 2026-09-30/10-01)
    TIER_RULE3_PIN = '/lustre/gh/main/phase3/deep_search_refusals/Run_P3S5_pin_20261008T025911Z/deep_search_pin.csv'   # phase3_tier_rules.py:37 [lit] (RULINGS-GREGG-20261008-TIER3-FREEZE-B2-RECHAIN-LABELS)
    TIER_RULE3_PIN_MD5 = '29a53b8355146311b2b05954e9478a98'   # phase3_tier_rules.py:38 [lit]
    COADD_RECORDS_FROZEN = '/lustre/gh/main/phase3/sublst/deep_coadd_records_frozen_20261008'   # phase3_tier_rules.py:45 [env:P3_COADD_RUN unset -> lit]
    MEMBERSHIP_VETO_PIN = '/lustre/gh/main/phase3/sublst/deep_coadd_records_frozen_20261008/membership_vetoes_pinned.csv'   # phase3_tier_rules.py:54 [lit]
    MEMBERSHIP_VETO_PIN_MD5 = '851b16d86fe059c8c8bf43b1afa72344'   # phase3_tier_rules.py:55 [lit]
    TIER_RULE4_RANGE = (0.33, 3.0)                        # qa3_scale outside -> night refused when degenerate-flatDC; phase3_tier_rules.py:151 [lit] (RULING-TIER-RULE-4-FAILED-FLUX-SCALE; B-21 open)
    ELEV_FLOOR_DEG = 25.0                                 # deg at the window mid-time; gate_inputs.py:94 [lit] (IMPLEMENTED-10MIN-ELEVATION-FLOOR-25-DEG; B-25 open)
    PERTIE_SHARDS = 16                                    # CASA imfit shards; sublst_catalogs.sh:46-49 [arg:--shards]
    PERTIE_WORKERS = 16                                   # few-frame stage workers; sublst_catalogs.sh:48-49 [arg:--workers]
    PERTIE_TEST_PIXELS = 40                               # reference fits re-measured in the window pixels (T3); sublst_catalog_chain.sh:37 [arg:--pixels]
    PERTIE_TEST_LIGHTCURVES = 5                           # few-frame light curves re-read from the cubes (T7); sublst_catalog_chain.sh:37 [arg:--lightcurves]
    ASTROM_FRAME_TEST_SCREENS = 8                         # screens checked against their pixels; sublst_catalog_chain.sh:39 [arg]
    SERVED_BUILD = 'noveto'                               # pages read source_catalogs_noveto_* and localizations_noveto; sublst_pages_prep.py:69, :104 [step]
    BUILDS = ('noveto', 'veto')                           # today both are built (B-24 open); sublst_catalog_chain.sh:30-31 [step]
    RUN_COMPARE = True                                    # old-vs-new comparison, failures logged only (B-24 open); sublst_catalog_chain.sh:41-43 [step]
    RUN_REFTIE = True                                     # reference tie annotation, output unread (ruling S1 open; speed-up 3); sublst_resume_from_tie.sh:25 [step]
    REFTIE_SHARDS = 16                                    # sublst_resume_from_tie.sh:25 [env:P3_REFASTRO_SHARDS]
    REFTIE_MEMCAP_GB = 250                                # GB: the tie starts only while /user.slice holds less; sublst_resume_from_tie.sh:12 (CAP) [step] (the script waits; the step refuses until runner admission exists, K7)
    COMPARE_DIR = '{root}/compare_{lst}/new_{arm}/catalog'   # small products copied for pages (root = paths.SUBLST_ROOT); sublst_catalog_chain.sh:44-52 [step]
    REJECTED_SYS_ARCSEC = 60.0                            # rejected-record systematic (M32 open, B-39 open); emit_rejected_record.py:23 [lit]
    REJECTED_BEAM_ARCSEC = {'Red': 787.0, 'Green': 518.0, 'Blue': 374.0, 'Full': 528.0}   # emit_rejected_record.py:24 [lit]
    # (PARENT_REBUILD_NPROC dropped: the parent build is masters --tier sublst --stages sources, decision 6 / crosscheck group 9)
    # ---- path templates and interpreters the tier module needs (added by catalog-tiers; values = the launchers' own)
    TENMIN_TREE = '{root}/{tree}'                         # P3_TENMIN_ROOT = SL/tenmin_root[_x2[_inj]]; sublst_catalogs.sh:20 [env:P3_TENMIN_ROOT]
    SEARCH_DIR = '{root}/search_{arm}'                    # SRC, the arm's search outputs <cell>_<S>/; sublst_catalogs.sh:27 [step]
    COADD_DIR = '{root}/coadd_{lst}'                      # sublst_catalogs.sh:29 [env:P3_COADD_ROOT]
    PARENT_DIR = '{root}/parent_catalogs_{lst}'           # sublst_catalogs.sh:29 [env:P3_PARENT_DIR]; sublst_parent_catalogs.py:16 writes here
    ROOTS_CSV = '{root}/catalog_inputs/roots_{lst}.csv'   # sublst_catalogs.sh:30, sublst_catalog_chain.sh:36 [env:P3S2_ROOTS_CSV]
    STACK_ROOT = '{root}/{lst}'                           # window stacks SL/<LST>; sublst_catalogs.sh:46-48 [arg:--stack-root]
    COMPARE_ROOT = '{root}/compare_{lst}'                 # sublst_catalog_chain.sh:42 (old/ and new_<arm>/ under it) [step]
    RULED = {'TIER_RULE4_RANGE': 'RULING-TIER-RULE-4-FAILED-FLUX-SCALE',
             'ELEV_FLOOR_DEG': 'IMPLEMENTED-10MIN-ELEVATION-FLOOR-25-DEG',
             'FLUX_BAKE_POLICY': 'IMPLEMENTED-Q10-DEEP-FLUX-SCALE-BAKED-AND-Q1-FRAME-TIME-IN-10MIN-STACKS',
             'TABLES_REQUIRED_METRIC': ('R-SUBLST-TABLES-GO-ELEVATION-BANDS',
                                        'RULINGS-GREGG-20261013-DECISIONS-1-7-AND-TENMIN-TABLES'),
             'TIER_RULE3_PIN': 'RULINGS-GREGG-20261008-TIER3-FREEZE-B2-RECHAIN-LABELS',
             'BUILDS': 'RULING-GREGG-20261008-PER-CASE-TIE-EVERYWHERE-SUBBAND-FALLBACK-B2-FINISHES-FIRST'}   # both builds today (B-24 open)


class catalog_sublst_tie(_Section):
    """window tie (sublst_pertie.py) - every constant of the module, and its md5-pinned tables (PROTECTED)."""
    # every key of this section is a [lit] of sublst_pertie.py / astrom_frame.py or a pinned table: none is overridable
    PROTECTED_KEYS = ('WIDEN_DEG', 'REF_TRY', 'REF_N', 'TIE_MIN_REFS', 'TIE_CONSISTENCY_K', 'TIE_SYS_FLOOR_ARCMIN', 'ACCEPT_SNR',
                      'ACCEPT_OFFSET_BEAMS', 'STAT_COEFF_SNR_FLOOR', 'ELEV_BANDS_DEG', 'FALLBACK_PCT', 'FALLBACK_MIN_N', 'FEW_Z',
                      'FEW_FRAC', 'MAX_SIGNAL_FRAMES', 'CUBE_ROOTS', 'CALIB_GRID_STEP_PCT', 'CALIB_MIN_N', 'PAIR_CAP_PER_REF',
                      'RHO_PAIR_CAP', 'RADIUS99_EDGES', 'WINERR_SNR_EDGES', 'ERR_MIN_N', 'PINNED_TABLES', 'ERRMODEL_FITS',
                      'ERRMODEL_FITS_SHA256', 'VLSSR_CAT', 'VLSSR_SHA256', 'ASTROM_FRAME_INPUT_DIR', 'ASTROM_FRAME_INPUT_MD5_FILE',
                      'ASTROM_FRAME_SCREEN_MANIFEST_MD5', 'ASTROM_FRAME_BAND_FALLBACK_PCT')
    WIDEN_DEG = (10.0, 15.0, 20.0, 25.0, 30.0)            # deg reference search radii; sublst_pertie.py:77 [lit] (RULING-GREGG-20261008-ASTROMETRY-MEASURE-DONT-MOVE-BOTH-TIERS item 2)
    REF_TRY = (8, 6, 6, 6, 6)                             # candidates fitted per radius; sublst_pertie.py:79 [lit]
    REF_N = 3                                             # references used per detection; sublst_pertie.py:80 [lit]
    TIE_MIN_REFS = 2                                      # fewer accepted -> untied; sublst_pertie.py:81 [lit]
    TIE_CONSISTENCY_K = 3.0                               # x combined statistical sigma x sqrt 2 (or one beam); sublst_pertie.py:82 [lit]
    TIE_SYS_FLOOR_ARCMIN = 0.05                           # arcmin; sublst_pertie.py:83 [lit]
    ACCEPT_SNR = 7.0                                      # sigma, accepted reference fit; sublst_pertie.py:84 [lit]
    ACCEPT_OFFSET_BEAMS = 1.0                             # beams; sublst_pertie.py:85 [lit]
    STAT_COEFF_SNR_FLOOR = (0.42, 5.0)                    # sublst_pertie.py:86 [lit] (Gregg 2026-09-28)
    ELEV_BANDS_DEG = ((25.0, 35.0), (35.0, 50.0), (50.0, 90.01))   # sublst_pertie.py:87 [lit] (Gregg 2026-09-27)
    FALLBACK_PCT = 90.0                                   # untied -> measured 90th percentile (ruling item 3); sublst_pertie.py:88 [lit]
    FALLBACK_MIN_N = 20                                   # tied detections per bin, else refuse; sublst_pertie.py:89 [lit]
    FEW_Z = 5.0                                           # sigma, a signal frame; sublst_pertie.py:536 [lit] (RULINGS-GREGG-20261009-ASTROMETRY-FOLLOWUPS-1-8 item 1)
    FEW_FRAC = 0.5                                        # signal fraction in the signal frames; sublst_pertie.py:537 [lit]
    MAX_SIGNAL_FRAMES = 6                                 # frames (one minute); sublst_pertie.py:538 [lit]
    CUBE_ROOTS = ('/lustre/gh/main/phase3/sublst/cubes_deepflow_18_23', '/lustre/pipeline/exopipe/phase3/snapshot_cubes')   # deep-flow root first (E4); sublst_pertie.py:539 [lit]
    CALIB_GRID_STEP_PCT = 0.5                             # percent; sublst_pertie.py:944 [lit] (list comprehension: not bound by the AST check)
    CALIB_MIN_N = 200                                     # sublst_pertie.py:945 [lit]
    PAIR_CAP_PER_REF = 10                                 # sublst_pertie.py:946 [lit]
    RHO_PAIR_CAP = 60                                     # sublst_pertie.py:1040 [lit]
    RADIUS99_EDGES = (10.0, 20.0)                         # peak |SNR| bins; sublst_pertie.py:1043 [lit] (RULINGS-GREGG-20261010-TENMIN-SANDWICH-AND-NEW-LSTS)
    WINERR_SNR_EDGES = (8.0, 9.0, 10.0, 11.0, 12.5, 15.0, 20.0, 30.0, 50.0)   # sublst_pertie.py:1050 [lit]
    ERR_MIN_N = 100                                       # fits per (class, SNR bin) for the derivation; sublst_pertie.py:1052 [lit]
    PINNED_TABLES = {'sublst_pertie_fallback.json': '915333b506488b3804736c6d6c24046e',          # sublst_pertie.py:90-91
                     'sublst_pertie_frame_fallback.json': 'f3c74d0e89faf50650dde80abfac4065',    # :540-541
                     'sublst_pertie_calib.json': '14cda064493058caf6e23213976523d2',             # :942-943
                     'sublst_pertie_winerr.json': '247999ec443bde8961edd5ac22312e81',            # :1032-1033
                     'sublst_pertie_frame_scale.json': '447c049efc1475e3ffb13a23ffc0a8ce',       # :1034-1035
                     'sublst_pertie_rho.json': '5b3827c09a7692998b573d308b226f4c',               # :1036-1037
                     'sublst_pertie_radius99.json': '10805bc8dcfc59a5256b89b0a827bea2'}          # :1041-1042; read beside the module (_HERE, :76) [lit]
    ERRMODEL_FITS = '/lustre/gh/main/phase3/astrom_frame_inputs_20261008/tenmin_window_fits_20261008.csv.gz'   # sublst_pertie.py:1044 [lit]
    ERRMODEL_FITS_SHA256 = 'ea15c200294c0b0bde0dddcf5727524d85f80aa0cb30080c846c1f027e170153'   # sublst_pertie.py:1045 [lit]
    VLSSR_CAT = '/lustre/gh/main/catalogs/radio_surveys_20261006/FullVLSSCatalog.text'   # sublst_pertie.py:1046 [lit]
    VLSSR_SHA256 = 'c101910cea4d5addee258d8ca4600d9d1f7451a82f171415554ff075780b19fe'   # sublst_pertie.py:1047 [lit]
    ASTROM_FRAME_INPUT_DIR = '/lustre/gh/main/phase3/astrom_frame_inputs_20261008'   # astrom_frame.py:59 [env:P3_ASTROM_FRAME_INPUTS unset -> lit default]
    ASTROM_FRAME_INPUT_MD5_FILE = 'astrom_frame.py:60 INPUT_MD5'   # md5 dict of the frozen inputs, read from the module [lit]
    ASTROM_FRAME_SCREEN_MANIFEST_MD5 = 'f00d29e0d32fc99526f11a88b6e5dc73'   # astrom_frame.py:74 [lit]
    ASTROM_FRAME_BAND_FALLBACK_PCT = 90.0                 # astrom_frame.py:77 [lit]
    RULED = {'WIDEN_DEG': 'RULING-GREGG-20261008-ASTROMETRY-MEASURE-DONT-MOVE-BOTH-TIERS',
             'FALLBACK_PCT': 'RULING-GREGG-20261008-ASTROMETRY-MEASURE-DONT-MOVE-BOTH-TIERS',
             'STAT_COEFF_SNR_FLOOR': 'R-ASSOCIATION-RADIUS-QUADRATURE',
             'ELEV_BANDS_DEG': 'R-SUBLST-TABLES-GO-ELEVATION-BANDS',
             'FEW_Z': 'RULINGS-GREGG-20261009-ASTROMETRY-FOLLOWUPS-1-8',
             'RADIUS99_EDGES': 'RULINGS-GREGG-20261010-TENMIN-SANDWICH-AND-NEW-LSTS'}


class catalog_sublst_veto(_Section):
    """master-leak veto re-derived for windows (leak_veto_sublst.py); its output is not served (B-24 open)."""
    PROTECTED_KEYS = ('REF_PEAK_MIN_JY', 'EL_BINS_DEG', 'DEEP_WEIGHTS_CSV')   # [lit]; the [env] keys are overridable
    ELEV_FLOOR_DEG = 30.0                                 # deg, affected side where the Master has >= MASTER_MIN_JY; leak_veto_sublst.py:33 [env:ELEV_FLOOR] (R-LEAKVETO-IN-GATED-PATH, Gregg 2026-09-22)
    ENVELOPE_PCT = 97.0                                   # percentile of reference (frame - Master)/Master; leak_veto_sublst.py:34 [env:ENVELOPE_PCT]
    REF_MIN = 8                                           # references needed, else refuse; leak_veto_sublst.py:35 [env:REF_MIN]
    MASTER_MIN_JY = 0.5                                   # Jy; leak_veto_sublst.py:36 [env:MASTER_MIN_JY]
    REF_PEAK_MIN_JY = 1.0                                 # Jy, reference parents; leak_veto_sublst.py:153 [lit] (inline literal: not bound by the AST check)
    EL_BINS_DEG = ((0, 18), (18, 23), (23, 28), (28, 35), (35, 45), (45, 90))   # leak_veto_sublst.py:38 [lit]
    K_ASTROM = 12                                         # references for the local astrometry column; leak_veto_sublst.py:39 [env:K_ASTROM]
    R_ASTROM_DEG = 10.0                                   # deg; leak_veto_sublst.py:39 [env:R_ASTROM_DEG]
    DEEP_WEIGHTS_CSV = '/lustre/pipeline/exopipe/phase3/Coadd/Run_20260909_000000/{lst}/{band}/weights.csv'   # LIVE deep record, not the frozen one (B-20); leak_veto_sublst.py:32 [lit] (f-string of argv: not bound by the AST check)
    RULED = {'ELEV_FLOOR_DEG': 'R-LEAKVETO-IN-GATED-PATH'}


class catalog_snapshot(_Section):
    """snapshot batch chain (snapshot_rechain_batch.sh steps 0-5 + run test; phase3_snapshot_chain.py constants)."""
    # pinned inputs (spec 'protected') + every [lit] constant of phase3_snapshot_chain.py / refastro / the run content test;
    # [arg] and [env] keys stay overridable (passed only when they differ from the module default)
    PROTECTED_KEYS = ('CHECKSOURCES', 'CURVE_CONSTANTS', 'GHOST_DATES', 'GHOST_OFFSET_DEG', 'GHOST_SIGMA_DEG',
                      'MIN_ELEVATION_DEG', 'ATEAM_ANCHOR_DEG', 'ATEAM_MIN_DEG', 'ATEAM_SLOPE', 'ATEAM_FIXED_DEG', 'ATEAM_FLUX_NPZ',
                      'EPHEM_DEG', 'JUPITER_BEAMS', 'GHOST_MATCH_PERCENTILE', 'NEG_LOBE', 'VHF_FLOOR', 'VHF_EVENT_VETO', 'ASYM',
                      'E_MAX', 'VETO_RECORD_SNR', 'E_PROMOTE', 'XM_PERCENTILE', 'INSTANT_BIN_S', 'REPEAT_CONSTRAIN_K',
                      'REPEAT_SKY_RATE_ARCSEC_S', 'CHECKSOURCE_MATCH_ARCMIN', 'ASTROM_FIT_SNR', 'NO_SOURCE_SNR',
                      'REJECT_NO_SOURCE_FIT', 'REJECT_UNCONSTRAINED_FIT', 'CLUSTER', 'GATE3D', 'GATE3D_NTHR', 'LC_CACHE_DEFAULT',
                      'VETO_PROCS_DEFAULT', 'REFASTRO_SHARDS_DEFAULT', 'REF_MAX_SEP_DEG', 'TEST_GATE3D_PCT', 'TEST_MEMBER_ORDER',
                      'TEST_GHOST_LEAK_DEG', 'PRE_BUNDLE_TIE_MODEL')
    # ---- driver (snapshot_rechain_batch.sh). Shared roots: paths.SNAPSHOT_FLEET_OUT (the chain's --proto-root,
    # snapshot_rechain_batch.sh:280), paths.CLV15_ROOT (--samples-dir, phase3_snapshot_chain.py:3329), paths.SNAPSHOT_REVIEW_ROOT
    # (the served run, :56), paths.OPS_ROOT
    RUN_PREFIX = 'Run_SNAP'                               # snapshot_rechain_batch.sh:49 [step]
    WORK_ROOT = '/fast/gh/main/phase3'                    # snapshot_rechain_<NAME>/<NAME>; snapshot_rechain_batch.sh:51 [step]
    BATCH_DIR = '/lustre/gh/main/exopipe_ops/snapshot_batch'   # stores, tie cache, claims, rechain_<NAME>; snapshot_rechain_batch.sh:51 [step]
    BATCH1_STORE = '/fast/gh/main/phase3/snapshot_batch1/fits_store_batch1.csv'   # on the chain host only; snapshot_rechain_batch.sh:55 [step] (host in site file)
    MASTER_CATALOGS = '/lustre/gh/main/exopipe_ops/snapshot_master_catalogs'   # snapshot_rechain_batch.sh:280 [arg:--master-catalogs]
    CURVE_CONSTANTS = '/lustre/gh/main/clv15/snapshot_curve_constants.json'   # snapshot_rechain_batch.sh:280 [arg:--curve-constants] (B-19 open)
    LIGHTCURVES_ROOT = '/lustre/gh/main/exopipe_ops/snapshot_lightcurves'   # snapshot_rechain_batch.sh:280 [arg:--lightcurves-root]
    VETO_RECORD_ROOT = '{run}/vetoed'                     # snapshot_rechain_batch.sh:354 [arg:--veto-record-root] (chain default /lustre/pipeline/exopipe/phase3/snapshot_vetoed, phase3_snapshot_chain.py:3356)
    DIRTY_CELLS = '/lustre/gh/main/exopipe_ops/snapshot_fleet_worklist_dirtyI_20260918.txt'   # phase3_snapshot_chain.py:3343 [arg:--dirty-cells] (B-77 open)
    DIRTY_I_FLOOR = 0.0                                   # sigma; 0 = a dirty cell refuses; phase3_snapshot_chain.py:3346 [arg:--dirty-i-floor]
    LIMIT_H = 1000                                        # h runaway guard (Gregg: no time stop); snapshot_rechain_batch.sh:68 [step]
    FAST_FREE_MIN_GB = 400                                # GB free on the chain host's /fast; snapshot_rechain_batch.sh:170 [step]
    PRE_BUNDLE_TIE_MODEL = 'stat0.42floor5_refh10'        # refused tie-cache model; snapshot_rechain_batch.sh:65 [step]
    ELIGIBLE_STATUSES = ('TO-PREFIT', 'PREFIT-DONE', 'PREFIT-STALE', 'WRITER-ACTIVE')   # snapshot_rechain_batch.sh:99 [step]
    PREFIT_RELEASE_WAIT = (45, 120)                       # tries x s for the prefit writer to let go (90 min); snapshot_rechain_batch.sh:116-122 [step]
    STORE_QUIET_MIN = 11                                  # min a store must be unwritten; snapshot_rechain_batch.sh:178 [step]
    STORE_QUIET_WAIT = (16, 120)                          # tries x s; snapshot_rechain_batch.sh:177-181 [step]
    LIST_PARALLEL = 12                                    # per-cell repair / prefit lists at a time; snapshot_rechain_batch.sh:248, :286 [step]
    PREFIT_MAX_ITER = 3                                   # snapshot_rechain_batch.sh:283, :306 [step] (Gregg 2026-10-07 Q6)
    FIT_PROCS_RESERVE = 2                                 # fitter processes = free cores - 2 per node; snapshot_rechain_batch.sh:222 [step]
    TIE_SHARDS = (30000, 16, 8)                           # >= 30,000 detections at the fit step -> 16 imfit shards, else 8; snapshot_rechain_batch.sh:322 [env:P3_REFASTRO_SHARDS]
    TIE_CELLS_PER_NODE = 3                                # snapshot_rechain_batch.sh:330 [step]
    CHAIN_REFASTRO_SHARDS = 24                            # snapshot_rechain_batch.sh:338 [env:P3_REFASTRO_SHARDS]
    VETO_PROCS = 24                                       # batch script 24 (byte-identical at any count, chain.py:2723); b2 ran 8 after ENOMEM (exopipe_ops 6a88359) [env:P3_VETO_PROCS] (question Q6 open)
    LC_CACHE_MAX = 4000                                   # tables of ~30.7 MB; snapshot_rechain_batch.sh:338 [env:P3_LC_CACHE_MAX]
    MEM_MODEL = {'base_gb': 60.0, 'gb_per_m_loaded': 2.2, 'gb_per_m_fitstep': 10.0, 'gb_per_table': 0.031, 'tables_per_cell': 30,
                 'min_gb': 250, 'max_gb': 390}         # snapshot_rechain_batch.sh:345-350 [step]
    MEM_WAIT_S = 28800                                    # s; snapshot_rechain_batch.sh:352 [step]
    SERVED_PORT_CHECK = 8779                              # the served run whose cells must be kept; snapshot_rechain_batch.sh:103 [step] (serve config owns the port)
    # ---- chain (phase3_snapshot_chain.py constants, all [lit] unless an argument is named)
    STORE_FLOOR = 7.0                                     # sigma; :98 [arg:--store-floor] (Gregg 2026-09-22 "Use floor 7")
    SEED_FLOOR = 8.0                                      # sigma; :101 [arg:--seed-floor]
    MIN_ELEVATION_DEG = 30.0                              # deg; :106 (Gregg 2026-09-09)
    ASSOC_FLOOR = 7.0                                     # sigma; :107 [arg:--assoc-floor]
    REPEAT_FLOOR = 7.0                                    # sigma; :113 [arg:--repeat-floor]
    SINGLETON_FLOOR = {'I': 10.0, 'V': 8.0}               # sigma; :114-115 [arg:--singleton-floor(-v)] (Gregg 2026-09-09, 2026-09-13)
    ATEAM_ANCHOR_DEG = 3.5                                # deg Tau A anchor; :130 (Gregg 2026-09-05)
    ATEAM_MIN_DEG = 6.0                                   # deg; :131 (Gregg 2026-09-20)
    ATEAM_SLOPE = 0.18                                    # (F / F_TauA)^0.18; :132
    ATEAM_FIXED_DEG = {'HyaA': 2.5, 'HerA': 2.5}          # deg; :133
    ATEAM_FLUX_NPZ = '/lustre/gh/calibration/pipeline/workingv17/primary_calibrator_flux_models.npz'   # :134
    EPHEM_DEG = 2.0                                       # deg Sun (and Jupiter floor) at each frame's time; :137 (Gregg 2026-10-07 item 26)
    JUPITER_BEAMS = 8.0                                   # beam FWHM; :142 (Gregg 2026-09-11)
    GHOST_DATES = ('2024-12-28',)                         # :167 (RULINGS-GREGG-20261007-STATUS-ANSWERS Q9; B-79)
    GHOST_OFFSET_DEG = (-20.745, 6.854)                   # (dl, dm) deg about the zenith; :168
    GHOST_SIGMA_DEG = 0.065                               # :169
    GHOST_MATCH_PERCENTILE = 95.0                         # :170
    NEG_LOBE = (5.0, 0.25, 1.5)                           # rms, ratio (ratified 2026-09-20), beams; :183, :190, :192
    VHF_FLOOR = 10.0                                      # sigma; :198 (Gregg 2026-09-14)
    VHF_EVENT_VETO = False                                # annotation only; :203
    ASYM = {'angle_deg': 45.0, 'half_deg': 45.0, 'r_beams': (0.5, 1.5), 'snap_px': 1e-6, 'd': 0.08, 'z': 3.0}   # :210-223 (Gregg 2026-09-14)
    E_MAX = 0.03                                          # label threshold only; :234
    VETO_RECORD_SNR = 30.0                                # sigma; :238
    E_PROMOTE = 0.5                                       # :239
    XM_PERCENTILE = 95.0                                  # :240
    INSTANT_BIN_S = 10.0                                  # s (instants keyed by file-name ts: D3, parity keeps it); :241
    REPEAT_CONSTRAIN_K = 3.0                              # :245 (Gregg 2026-09-15)
    REPEAT_SKY_RATE_ARCSEC_S = 15.0                       # :246
    CHECKSOURCES = '/lustre/gh/main/exopipe_ops/snapshot_checksources.csv'   # :250 (R-SNAP-CHECKSOURCE-GATE-LPT-AND-PULSAR-IMPLEMENTED; B-78)
    CHECKSOURCE_MATCH_ARCMIN = 15.0                       # :2865, :2871 (no ruling; B-78) (inline literal: not bound by the AST check)
    STRUCT = {'k': 10.0, 'box_fwhm': 2.0, 'box_min_px': 15}   # :263-266 [arg:--struct-k]
    ASTROM_FIT_SNR = 7.0                                  # :756 (CLAUDE.md 5) (inline: not bound)
    NO_SOURCE_SNR = 8.0                                   # :960 (Gregg 2026-09-07) (inline: not bound)
    REJECT_NO_SOURCE_FIT = True                           # :947 (inline: not bound)
    REJECT_UNCONSTRAINED_FIT = True                       # :959 (RATIFIED-AND-IMPLEMENTED-EVERY-UNCONSTRAINED-FRAME-REMOVED-SNAPSHOT-TIER; M29 open) (inline: not bound)
    CLUSTER = (10.0, 15.0)                                # deg, min annotation window; :1495-1496 (inline: not bound)
    GATE3D = {'ratio': 2.0, 'snr': 8.0, 'parent_scale_refs': 10, 'match_percentile': 95.0, 'self_parent_all_bands': False,
              'ratio_high': 5.0, 'alpha': -1.5, 'alpha_min_sb': 4}   # :1792-1829 (Gregg 2026-09-18, 2026-10-04, 2026-10-07 Q15)
    GATE3D_NTHR = ((0.0, 3.0, 1), (3.0, 5.0, 1), (5.0, 10.0, 3), (10.0, 20.0, 23), (20.0, 50.0, 83), (50.0, float('inf'), 135))   # PROVISIONAL; :1795
    LC_CACHE_DEFAULT = 200                                # tables when P3_LC_CACHE_MAX unset; :1851 (inline: not bound)
    VETO_PROCS_DEFAULT = 16                               # :2723 (inline: not bound)
    REFASTRO_SHARDS_DEFAULT = 8                           # phase3_snapshot_refastro.py:354
    REF_MAX_SEP_DEG = 10.0                                # phase3_snapshot_refastro.py:47 ("nearby (within 10 degrees)")
    FLOW_MODE = True                                      # P3_SNAP_FIT_RAW unset -> flow-registered frames; :78 [env:P3_SNAP_FIT_RAW]
    # ---- run content test (snapshot_rechain_content_test.py R1-R7 operating points)
    TEST_GATE3D_PCT = 95.0                                # snapshot_rechain_content_test.py:71
    TEST_MEMBER_ORDER = 'time, subband, Stokes'           # snapshot_rechain_content_test.py:71 (Gregg 2026-10-07 item 23)
    TEST_GHOST_LEAK_DEG = 1.0                             # snapshot_rechain_content_test.py LEAK_DEG (R3, exopipe_ops 56810b9): a removed
                                                          # ghost leaks when a served row of its cand_id lies within this of its position
    # ---- added by catalog-tiers (values = snapshot_rechain_batch.sh's own; each a [step] key)
    NODE_FAST_ROOT = '/fast/gh/main/phase3'               # NODE_FAST = <root>/rechain_<NAME> on every node (tie outputs, fit shards); snapshot_rechain_batch.sh:58
    APPROVAL_MARKER = '/lustre/gh/main/exopipe_ops/snapshot_rechain_{name}.APPROVED'   # production approval marker (read only; --snap-approval-marker); snapshot_rechain_batch.sh:53
    CODE_SHA256 = '/lustre/gh/main/exopipe_ops/snapshot_rechain_{name}.code_sha256'    # tested-code fingerprint (P-G3, window form); snapshot_rechain_batch.sh:53, :75-76
    RULED = {'STORE_FLOOR': 'R-SNAP-STORE-FLOOR-8-SIGMA',          # 8 -> 7 by Gregg 2026-09-22 "Use floor 7" (no own ledger row found)
             'SINGLETON_FLOOR': 'R-SNAP-SINGLETON-FLOOR-10-SIGMA',
             'GHOST_DATES': 'RULINGS-GREGG-20261007-STATUS-ANSWERS',
             'NEG_LOBE': 'RATIFIED-NEG-LOBE-RATIO-0-25-AND-STRUCTURE-GATE-K-10-AND-ENV-OVERRIDES-REMOVED',
             'STRUCT': 'RATIFIED-NEG-LOBE-RATIO-0-25-AND-STRUCTURE-GATE-K-10-AND-ENV-OVERRIDES-REMOVED',
             'PREFIT_MAX_ITER': 'RULINGS-GREGG-20261007-STATUS-ANSWERS',
             'EPHEM_DEG': 'RULINGS-GREGG-20261007-STATUS-ANSWERS',
             'GATE3D': 'RULINGS-GREGG-20261007-STATUS-ANSWERS',
             'CHECKSOURCES': 'R-SNAP-CHECKSOURCE-GATE-LPT-AND-PULSAR-IMPLEMENTED',
             'REJECT_UNCONSTRAINED_FIT': 'RATIFIED-AND-IMPLEMENTED-EVERY-UNCONSTRAINED-FRAME-REMOVED-SNAPSHOT-TIER'}
