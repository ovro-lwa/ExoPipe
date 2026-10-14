"""Config draft of `exopipe3 calibrate` (unit calibrate; REFACTOR WINDOW ONLY - config.py loads every _cfgdraft_*.py until the
refactor chat folds them into config.py, then this file is deleted).

Source: plan/specs/calibrate.md section 3 (values = production at exopipe main 03ca797, exopipe_ops 0c1580c, clv15 9b6525d;
re-read 2026-10-13 ~05:00 UTC: every sha256 below was recomputed and equals the spec's). "ledger N" = line N of
clv15/fix_ledger.csv. Every RULED value is a fix_id of that ledger (tests/test_calibrate.py checks each one).

Sections defined HERE: tables (shared: the installed reference tables, read by every consumer), calibrate (shared keys of the
step), calibrate_bright_parent, calibrate_multiples, calibrate_scale, calibrate_fluxcorr, calibrate_template_k.
Not here (classes not implemented in this unit yet - see plan/impl/calibrate.md): calibrate_snapshot_curve,
calibrate_snapshot_astrom, calibrate_completeness.

Changes from the spec block (all pure, no product change):
- Constants that live INSIDE an in-place module and cannot be set by the step (S_EDGES, SEP_STEP_BW, SEP_MAX_BW, N_NULL,
  NULL_CAP_DEG, NULL_SEED, MIN_CANDIDATES_PER_BIN, SMOOTH_KERNEL_BINS, REPORT_RATIO, PASS_LEDGER_VERDICTS of
  p3s3_curve_derive / phase3_snapshot_curve; VLSS_MIN_JY, MATCH_FRAC, ISO_FRAC, MIN_SUBBANDS, PB_MIN, ALPHA, PL_MAX_SCATTER,
  VLSSR_CATALOG of vlssr_flux_check; INJECT_SNRS, START_OFFSET_MAX_BEAMS, MIN_SEP_BEAMS, BRIGHT_AVOID_* of the injection
  scripts) are NOT config keys during the window: a key the step cannot apply would make the config untrue. They are listed
  in the section docstrings with their file:line and become keys when the code is absorbed into lib/calibrate.
- Values the step DOES apply (argv / environment of the in-place call, its own ported arithmetic, its gates) are keys. Where
  the in-place module also holds the value as a literal (post_process's 20-candidate guard and monotone cap; scale_table_v4's
  window and rounding), the step checks the module against the key before using it (lock) and refuses on a difference.
- Shared roots come from config.paths (COADD_SB_ROOT, TRANSIENTS_ROOT, SUBLST_ROOT, OPS_ROOT, ...), never a key here.
"""
from exopipe.phase3.config import _Section


class tables(_Section):
    """Installed reference tables: path + sha256 + version, read by every consumer (search, catalog, forced, masters, fitting).
    PROTECTED (pinned inputs, DESIGN 11.4): no --set / --config can change a key. A new version comes ONLY from
    `exopipe3 calibrate <table> install` - a new versioned file, a patch of this section (and of every other section that
    mirrors the pointer, e.g. search_tables.PARENT_TABLES / _MD5, masters_fluxscale.*), and a draft ledger row - applied by a
    commit. Never edited in place. Paths are today's (no renames in the refactor window, DESIGN 11.1).
    *_INPUTS / *_DERIVATIONS keys name the measurement each installed table was made from, so `calibrate <table> verify`
    recomputes the installed table from them (content tests recompute, never trust)."""
    # ---- bright-parent + multiples tables, deep (the file also carries the 38 retired 10min_* sky-metric keys byte for byte)
    BRIGHT_PARENT_DEEP = '/lustre/gh/main/exopipe/exopipe/common/parent_curve_tables.json'   # file; ledger 650 R-2D-GATE-PER-POLARIZATION-PER-BAND-DEEP; pkg/phase3/parent_curve_table.py:13
    BRIGHT_PARENT_DEEP_SHA256 = 'b6341ca5dd0ef2ff1433a177df011271c63f3e4076932f95aa80123a44dc18c6'   # sha256 (re-read 2026-10-13); last install exopipe 2a5f7d5 (2026-10-08 16:53 UTC, ledger 2101)
    # derivations the installed deep keys were built from (verify recomputes post_process key by key; all read-only):
    BRIGHT_PARENT_DEEP_DERIVATIONS = {
        'all-elevation': ('/data07/gh/phase3/transients/Run_P3S4/deep_tables_pix_20261001T0510.json',        # ratio 3 (33 keys); deep_tables_pixel_20261001.py:23-25
                          '/data07/gh/phase3/transients/Run_P3S4/deep_tables_pix_r2V_20261001T0510.json'),   # ratio 2, V 18-36 MHz (5 keys); installed 2026-10-01 (ledger 1684)
        'elevation-bands': ('/fast/gh/main/phase3/deep_tables_elev_20261006/deep_elev_r3.json',               # 114 band keys; deep_tables_elev_20261006.py:27-28
                            '/fast/gh/main/phase3/deep_tables_elev_20261006/deep_elev_r2V.json'),             # installed 2026-10-07 (ledger 1821); lwacalim07 node-local
        'multiples': ('/fast/gh/main/phase3/multiples_tables_20261008/multi_r3_I.json',                       # 89 -multi_ keys; multiples_fork.py:586
                      '/fast/gh/main/phase3/multiples_tables_20261008/multi_r3_V.json',
                      '/fast/gh/main/phase3/multiples_tables_20261008/multi_r2V.json')}                       # installed 2026-10-08 (ledger 2101); lwacalim07 node-local
    BRIGHT_PARENT_SUBLST = '/lustre/gh/main/exopipe/exopipe/common/parent_curve_tables_sublst.json'   # file; ledger 1557 R-SUBLST-TABLES-GO-ELEVATION-BANDS, ledger 2297; ops/sublst_catalogs.sh:75 (read from the frozen code tree $XC)
    BRIGHT_PARENT_SUBLST_SHA256 = '3a208eb4909d899ddb6cd94f1d6903d2782e39be432aa85fa5a92f6346c2b1fd'   # sha256; installed 2026-10-01T23:51:40Z (file meta)
    BRIGHT_PARENT_SUBLST_DERIVATION = None   # the installed file's derivation (lwacalim08:/fast/gh/main/phase3/sublst_catalogs/catalog_10h_B/sublst_tables_20261001T2350.json) is node-local on lwacalim08: equals_derivation is not run on the installed sub-LST file (spec B3)
    SNAPSHOT_CURVE_CONSTANTS = '/lustre/gh/main/clv15/snapshot_curve_constants.json'   # file; ledger 626, ledger 1678; clv15/phase3_snapshot_chain.py:275
    SNAPSHOT_CURVE_CONSTANTS_SHA256 = '6ff5ec9237d50274a0f3002b888cbb0c993524aa48849344aba61920c7018cdd'
    PULSARS_ATNF = '/lustre/gh/main/clv15/pulsars_atnf.csv'   # file; pulsar exemption sample, ledger 1449 R-TABLE-GATE-PULSAR-PARENTS-KEPT; parent_curve_table.py:67, :312
    PULSARS_ATNF_SHA256 = '79629d6d430a0c4c05c2cffd3b5be5fd1879ef1ba08c3197ceab82c078a0aad2'
    # ---- absolute flux scale (per LST) and band correction (k x g), deep (sub-LST inherits through the masters' SHSCALE card)
    SCALE_TABLE = '/lustre/gh/main/exopipe_ops/pbstep2_20261001/vlssr_scale_table_v4_20261001.csv'   # file; ledger 768; rebuild_cell.sh:144 reads the byte-identical lwacalim07 /fast copy
    SCALE_TABLE_SHA256 = 'dfb7852df13c8e9ad68c24b9f50a11e743e270bd458dad644d17e9566f5cb891'   # = /fast/gh/main/phase3/shflux/vlssr_scale_table_v4_20261001.csv
    SCALE_TABLE_VERSION = 'v4-20261001'   # ledger 1711 RUN-FLUX-REMEASURE-REBAKE-STAGE2-3-RESULTS-AND-STAGE4-TOOLING
    SCALE_TABLE_PREVIOUS = '/fast/gh/main/phase3/shflux/vlssr_scale_table_v3_threefactor.csv'   # s_old of v4 (lwacalim07 node-local); run_rebake_20261001.sh:11
    SCALE_TABLE_INPUTS = {'ratios': '/fast/gh/main/phase3/shflux/vlssr_check_pbstep2_20261001/ratios_73MHz.csv',   # the 2026-10-01 VLSSr measurement (lwacalim07 node-local)
                          'previous': '/fast/gh/main/phase3/shflux/vlssr_scale_table_v3_threefactor.csv'}
    FLUXCORR_TABLE = '/lustre/gh/main/exopipe_ops/pbstep2_20261001/flux_correction_v2_20261001.csv'   # file; ledger 1705 R-FLUX-CORRECTION-BAKED-INTO-PIXELS; rebuild_cell.sh:145
    FLUXCORR_TABLE_SHA256 = 'e50df4530fe0437f5fe890353b95dd07fb7f93eb10c82f98f189da8386153c77'
    FLUXCORR_SHCELLS = '/lustre/gh/main/exopipe_ops/pbstep2_20261001/flux_correction_v2_20261001_shcells.csv'   # per-cell S&H factors of the fit (flux_correction.py:69-70)
    FLUXCORR_SHCELLS_SHA256 = 'f8fda3c6b5bda7a7aa5bd59676d53ea2c591df0fe7378378b1965db6fdb53de5'
    FLUXCORR_VERSION = 'fluxcorr-v2-20261001'   # = pkg/phase3/flux_correction.py:19 VERSION prefix = membership.py:251 REQUIRED_FLUXVER; ledger 1714
    FLUXCORR_INPUTS = {'stagea_glob': '/fast/gh/main/phase3/shflux/stageA_pbstep2_20261001/shflux_measure_*.csv',   # 444 OK cells (lwacalim07 node-local); run_stageA_20261001.sh:4
                       'band_vector': '/lustre/gh/main/exopipe_ops/pbstep2_20261001/band_vector_final_20261001.csv',
                       'scale_table': '/lustre/gh/main/exopipe_ops/pbstep2_20261001/vlssr_scale_table_v4_20261001.csv'}   # lst_rescale = its 'rescale' column (ledger 1714)
    BAND_VECTOR = '/lustre/gh/main/exopipe_ops/pbstep2_20261001/band_vector_final_20261001.csv'   # file; LOO power-law vector k_b; ledger 988
    BAND_VECTOR_SHA256 = '09e9aab0a8088597ddf5729bd379458f01bb361eca3b03cef83e648d4508347b'
    BAND_VECTOR_CONVERGENCE = '/fast/gh/main/phase3/shflux/bandvec_pbstep2_20261001/loo7/smooth_by_subband_powerlaw.csv'   # LOO medians WITH the final vector applied (worst 0.18 per cent; iterate.log 2026-10-01 21:13); lwacalim07 node-local
    FLUXCORR_TABLE_SUBLST_UNBAKED = '/lustre/gh/main/phase3/sublst/catalog_inputs/shflux/flux_correction_v1_20260911.csv'   # un-baked 10-minute LSTs only (none today); ops/sublst_catalogs.sh:68
    FLUXCORR_TABLE_SUBLST_UNBAKED_SHA256 = 'e7f9e109e99493c40ebb9fda29b066a4cd2949678cca8f6ae7bf45c7472418cb'
    C50_TABLE = '/data07/gh/phase3/transients/c50_measured.json'   # file (lwacalim07 node-local); Gregg 2026-06-30 "no hard-wired numbers" (pkg/phase3/phase3_c50.py:3-9, no ledger ID); phase3_c50.py:17
    C50_TABLE_SHA256 = 'f5f480daa6d32d03ff19202d93ac57efe74be439f2da58e905c40d57677bf942'
    # ---- snapshot astrometry calibrations: STAGED in staged_20261008_pertie_astrometry (live clv15 reads none; install waits for the batch-2 serve)
    SNAPSHOT_UNTIED_SYS = '/lustre/gh/main/exopipe_ops/staged_20261008_pertie_astrometry/snapshot_untied_sys_v2.json'   # ledger 2082 item 2; staged phase3_snapshot_astro.py:44-46
    SNAPSHOT_UNTIED_SYS_SHA256 = '82a43bafab8771a2db554c3749d2dfad509afe6932fd091be4fc6776a42c7ec1'
    SNAPSHOT_ASSOC_CALIB = '/lustre/gh/main/exopipe_ops/staged_20261008_pertie_astrometry/snapshot_assoc_calib_v1.json'   # ledger 2094 item 5; staged phase3_snapshot_astro.py:232-234
    SNAPSHOT_ASSOC_CALIB_SHA256 = 'a0e7a80a0ce0b0aadb402c7ec17db7be42f4a01d1fb0f9da1c96cb90c1a9360d'
    SNAPSHOT_FORCED_CEILING = '/lustre/gh/main/exopipe_ops/staged_20261008_pertie_astrometry/snapshot_forced_ceiling_v2.json'   # ledger 2110 item 7, ledger 2123 (B); staged snapshot_forced_assoc.py:87-89
    SNAPSHOT_FORCED_CEILING_SHA256 = 'c5db06f5705c4b2401a28cd3afb77b66dcc382e297d5f5443f33b04e2f3ccb60'   # = staged snapshot_forced_assoc.py:89 CEIL_CALIB_SHA256, re-pinned 2026-10-13 (exopipe_ops 3909fd2, localizer 43db654; result identical); was f9c711f7...
    # ---- sub-LST window-tie tables: md5 exactly as ops/sublst_pertie.py pins them
    SUBLST_PERTIE_TABLES = {'fallback': ('/lustre/gh/main/exopipe_ops/sublst_pertie_fallback.json', '915333b506488b3804736c6d6c24046e'),             # sublst_pertie.py:90-91
                            'frame_fallback': ('/lustre/gh/main/exopipe_ops/sublst_pertie_frame_fallback.json', 'f3c74d0e89faf50650dde80abfac4065'),   # :540-541
                            'calib': ('/lustre/gh/main/exopipe_ops/sublst_pertie_calib.json', '14cda064493058caf6e23213976523d2'),                   # :942-943
                            'winerr': ('/lustre/gh/main/exopipe_ops/sublst_pertie_winerr.json', '247999ec443bde8961edd5ac22312e81'),                 # :1032-1033
                            'frame_scale': ('/lustre/gh/main/exopipe_ops/sublst_pertie_frame_scale.json', '447c049efc1475e3ffb13a23ffc0a8ce'),       # :1034-1035
                            'rho': ('/lustre/gh/main/exopipe_ops/sublst_pertie_rho.json', '5b3827c09a7692998b573d308b226f4c'),                       # :1036-1037
                            'radius99': ('/lustre/gh/main/exopipe_ops/sublst_pertie_radius99.json', '10805bc8dcfc59a5256b89b0a827bea2')}             # :1041-1042; ledger 2123
    # ---- template K: the VALUES live in fitting.K_TEMPLATE / K_SNAPSHOT; these are their measurement records
    K_RECORDS = {'deep': '/fast/gh/main/scratch/kcal_d5_20261013',   # trials_*.csv (600 I, 300 V) of the re-measure after the template mirror fix (lwacalim07 node-local): K I 0.542/0.500, V 0.572/0.551; ledger 2322 IMPL-D5-DEEP-K-REMEASURED-AFTER-TEMPLATE-MIRROR-FIX-20261013 (superseded: exopipe_ops/deep_forced_kcal_20261011, mirrored template, ledger 2189)
                 'sublst': None,       # trials on lwacalim00:/fast (not on Lustre; ledger 2285): K cannot be recomputed here (spec B6)
                 'snapshot': None}     # template_fit_validation run_20261004b (path not found; clv15/phase3_template_fit.py:30-34 comment)
    PROTECTED_KEYS = ('BRIGHT_PARENT_DEEP', 'BRIGHT_PARENT_DEEP_SHA256', 'BRIGHT_PARENT_DEEP_DERIVATIONS', 'BRIGHT_PARENT_SUBLST',
                      'BRIGHT_PARENT_SUBLST_SHA256', 'BRIGHT_PARENT_SUBLST_DERIVATION', 'SNAPSHOT_CURVE_CONSTANTS',
                      'SNAPSHOT_CURVE_CONSTANTS_SHA256', 'PULSARS_ATNF', 'PULSARS_ATNF_SHA256', 'SCALE_TABLE', 'SCALE_TABLE_SHA256',
                      'SCALE_TABLE_VERSION', 'SCALE_TABLE_PREVIOUS', 'SCALE_TABLE_INPUTS', 'FLUXCORR_TABLE', 'FLUXCORR_TABLE_SHA256',
                      'FLUXCORR_SHCELLS', 'FLUXCORR_SHCELLS_SHA256', 'FLUXCORR_VERSION', 'FLUXCORR_INPUTS', 'BAND_VECTOR',
                      'BAND_VECTOR_SHA256', 'BAND_VECTOR_CONVERGENCE', 'FLUXCORR_TABLE_SUBLST_UNBAKED',
                      'FLUXCORR_TABLE_SUBLST_UNBAKED_SHA256', 'C50_TABLE', 'C50_TABLE_SHA256', 'SNAPSHOT_UNTIED_SYS',
                      'SNAPSHOT_UNTIED_SYS_SHA256', 'SNAPSHOT_ASSOC_CALIB', 'SNAPSHOT_ASSOC_CALIB_SHA256', 'SNAPSHOT_FORCED_CEILING',
                      'SNAPSHOT_FORCED_CEILING_SHA256', 'SUBLST_PERTIE_TABLES', 'K_RECORDS')
    RULED = {'BRIGHT_PARENT_DEEP': ('R-2D-GATE-PER-POLARIZATION-PER-BAND-DEEP', 'MULTIPLES-FORK-TESTED-AND-TABLES-INSTALLED'),
             'BRIGHT_PARENT_SUBLST': ('R-SUBLST-TABLES-GO-ELEVATION-BANDS', 'RULINGS-GREGG-20261013-DECISIONS-1-7-AND-TENMIN-TABLES'),
             'SCALE_TABLE': 'R-FLUX-SCALE-APPLY-VLSSR-HIGH-ELEVATION-APPROVED',
             'FLUXCORR_TABLE': 'R-FLUX-CORRECTION-BAKED-INTO-PIXELS',
             'BAND_VECTOR': 'METHOD-BAND-VECTOR-LEAVE-ONE-BAND-OUT-CONVERGENCE',
             'PULSARS_ATNF': 'R-TABLE-GATE-PULSAR-PARENTS-KEPT'}


class calibrate(_Section):
    """exopipe3 calibrate - shared keys (one section per table class follows). Intermittent (survey) step: never part of the
    per-hour chain; the per-hour steps read the INSTALLED tables through section `tables`."""
    OUT_ROOT = '/lustre/gh/main/phase3/calibrate'   # candidate / diff / record root (Lustre, never node-local); new (no production equivalent); under paths.PRODUCTION_ROOTS, so refused while PRODUCTION_WRITES is False (parity passes --out)
    VERSIONED_TABLES_ROOT = None   # where `install` puts the new versioned table file once PRODUCTION_WRITES is True; None = inside the install bundle (<out>/calibrate/<class>/<tier>/<version>/install/), the only place it can be during the window
    SERVED_RUN = {'deep': '/data07/gh/phase3/transients/Run_P3S5', 'sublst': '/data07/gh/phase3/transients/Run_SLSTB'}   # served lists read by diff; deep_tables_elev_20261006.py:36 (DRY_RUN_DIR default); sub-LST cumulative run (sublst_serve.sh)
    SERVED_CATALOG_TIERS = {'deep': ('', '_wb', '_sb', '_fwb', '_fsb'), 'sublst': ('',)}   # source_catalogs_<S><tier>/ read by diff; deep_tables_elev_20261006.py:129 read only '' - the fork tiers exist since 2026-10-08 (reporting only, no table change; spec Q7 default)
    JUDGEMENTS = '{run}/judgements_{S}/judgements.jsonl'   # deep_tables_elev_20261006.py:133
    DIFF_LIST_CSV = ('master_catalog_asrun.csv', 'master_catalog.csv')   # first that exists in the served run; deep_tables_elev_20261006.py:119
    DIFF_LEDGER_LABEL = {'deep': 'P3S5', 'sublst': None}   # run label of the published gate ledgers searched for in-search table removals (DRY_LEDGER_LABEL, deep_tables_elev_20261006.py:37); sub-LST: none (its search applies no table, search_tables.TABLE_RECORD_ONLY)
    DIFF_REQUIRED_FOR_INSTALL = True   # gate G22: install needs a done diff record of the same candidate sha256 (DESIGN 11.2 "install ... with a diff report on served and judged sources"); new gate (pure)
    INSTALL_RULING_REQUIRED = True     # gates G18/G22: install needs --install-ruling <ledger ID> (a fix_ledger row; ledger 1821 "named approval, source-removing"; staged-k/install.sh K_RULING.txt)
    LEDGER_DRAFT_COLUMNS = ('date', 'fix_id', 'files', 'description', 'authorized_by', 'status')   # = clv15/fix_ledger.csv header
    DERIVE_PROCESSES = 3     # concurrent in-place derivation sub-runs per unit; derive.sh:10-11 (2), multiples derive_conformant.sh:8-11 (3); --workers overrides
    THREADS = 1              # OMP_NUM_THREADS per sub-run; derive.sh:9, derive_conformant.sh:6
    OVRO_LAT_DEG_LEGACY = 37.2398   # deg; site latitude of every derivation and dry run today (deep_tables_elev_20261006.py:91; p3s2_conform.py:211, :278; gate_inputs.py:96); B-42 keeps it (core/astrometry uses 37.23977727)
    LEGACY_PYTHON = 'default'        # site PYTHON key of every in-place script (py38_orca_nkosogor: derive.sh, run_stageA_20261001.sh, run_band_vector_20261001.sh)
    # in-place scripts and modules, relative to the site legacy root 'exopipe_ops' (refactor window; absorbed into lib/calibrate at cutover)
    LEGACY_SCRIPTS = {'curve_derive': 'p3s3_curve_derive.py',
                      'parent_union': 'p3s5_parent_union.py',
                      'post_process': 'deep_tables_elev_20261006.py',      # imported in process for post_process only (pure)
                      'vlssr_check': 'vlssr_flux_check.py',
                      'scale_table': 'pbstep2_20261001/scale_table_v4.py',
                      'kcal_tenmin': 'staged_20261012_sublst_fitted_positions/tests/kcal_tenmin_inject.py',
                      'kinj_deep': 'deep_forced_kcal_20261013/kinj_deep.py'}   # the deep K harness after the mirror fix (exopipe_ops 554b721; reads the production deep_forced_radio)
    PROTECTED_KEYS = ('LEDGER_DRAFT_COLUMNS', 'LEGACY_SCRIPTS', 'OVRO_LAT_DEG_LEGACY')
    RULED = {'DIFF_REQUIRED_FOR_INSTALL': 'R-GREGG-20261013-REFACTOR-DESIGN-V2-APPROVED',
             'INSTALL_RULING_REQUIRED': 'R-DEEP-BRIGHT-SOURCE-TABLES-ELEVATION-BANDS-AND-FRAME-S'}


class calibrate_bright_parent(_Section):
    """Bright-parent (2-D, per Stokes and band) tables, deep and sub-LST: derivation = ops/p3s3_curve_derive.py IN PLACE (port of
    clv15/phase3_snapshot_curve.derive), post-processing = deep_tables_elev_20261006.post_process IN PLACE (the 20-candidate
    excess guard and monotone cap, locked against EXCESS_GUARD_CANDIDATES / MONOTONE_CAP below), key loops ported from
    deep_tables_elev_20261006.build (:60-87), deep_tables_pixel_20261001.build and sublst_install_tables.py (:49-64).
    Application keys (DEEP_V_FAINT_S, ELEV_BANDS_DEG, K_PSR) are the search's (section search_tables).
    Module literals NOT settable during the window (keys at absorption): S_EDGES ((1,1.5)...(200,1e6); phase3_snapshot_curve.py:43-44),
    SEP grid 0.25 bw to 40 bw (:45), N_NULL 150000 (:47), null cap 70 deg (random_cap :61), null seed 7 (p3s3_curve_derive.py:114),
    30 candidates per bin (:185), 3-bin boxcar (:188), report ratio 1.5 (:198), ledger verdicts reaching the table
    ('', curve, curve-flood, table; :250)."""
    FLOOR_SIGMA = 5.0          # sigma, |snr| floor of harvest rows (argv --floor); ledger 653, ledger 1240; derive.sh:10, sublst_catalogs.sh:90
    RATIO = {'deep': 3.0, 'sublst': 2.0}     # density-ratio threshold (env P3S3_CURVE_RATIO); deep ledger 653; sub-LST ledger 902, 980, 1240; derive.sh:10, sublst_catalogs.sh:89
    RATIO_LOW_V = 2.0          # deep Stokes V at RATIO_LOW_V_BANDS (second derivation, Stokes V only); ledger 653; derive.sh:11; deep_tables_elev_20261006.py:70
    RATIO_LOW_V_BANDS = ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz')   # deep_tables_elev_20261006.py:30; multiples_fork.py:59
    METRIC = {'deep': 'image-pixel', 'sublst': 'image-pixel'}   # env P3_GATE_METRIC; ledger 1684, ledger 1668; derive.sh:8, sublst_catalogs.sh:87 (module default 'sky', p3s3_curve_derive.py:52)
    ELEV_BANDS_DERIVE = {'deep': '20-25,25-35,35-50,50-90', 'sublst': '25-35,35-50,50-90'}   # deg, env P3_CURVE_ELEV_BANDS; ledger 1821, ledger 1557; derive.sh:9, sublst_catalogs.sh:89
    ELEV_BANDS_INSTALL = ((25.0, 35.0), (35.0, 50.0), (50.0, 90.0))   # deg; deep_tables_elev_20261006.py:31, sublst_install_tables.py:18
    ELEV_BANDS_REPORT_ONLY = {'deep': ((20.0, 25.0),), 'sublst': ()}   # deg; derived and reported, never installed; deep_tables_elev_20261006.py:31 (Gregg 2026-10-04 item 3), ledger 1669
    EXCESS_GUARD_CANDIDATES = 20   # candidates; a bin keeps its radius only if marginal_excess x n_candidates >= 20; ledger 1242; deep_tables_elev_20261006.py:48 (LOCK: the in-place post_process is checked against it)
    MONOTONE_CAP = True        # radius capped from the brightest bin down; ledger 1242; deep_tables_elev_20261006.py:49-51 (LOCK, as above)
    BY = 'subband'             # argv --by: one table per band label (15 subbands + 4 wideband groups); derive.sh:10
    STOKES = ('V', 'I')        # argv --stokes of the main derivation; derive.sh:10 (the low-ratio run is Stokes V only, :11)
    TARGET_BEAM = True         # env P3_GATE_TARGET_BEAM=1: separations in each detection's searched-frame beam; ledger R-GATE-SEPARATIONS-ANGULAR-AND-DEEP-V-REPAIR-AND-REFERENCE-FIT-TEST-APPROVED (1445); p3s3_curve_derive.py:47
    ELEV_FLOOR_DEG = {'deep': 25.0, 'sublst': 25.0}   # deg; deep via env P3_DEEP_MIN_ELEVATION (gate_inputs.py:95); sub-LST = the literal gate_inputs.TENMIN_ELEV_FLOOR_DEG (:94, LOCK); ledger 1905, ledger 1674
    BUILD_KEYS = {'deep': ('elevation-bands',), 'sublst': ('all-elevation', 'elevation-bands')}   # deep: the installed all-elevation tables are NOT replaced (deep_tables_elev_20261006.py:12-14; the running searches apply them); ('all-elevation', 'elevation-bands') = the 2026-10-01 + 2026-10-06 builds together; sub-LST: the whole file (sublst_install_tables.py:49-64)
    SUBLST_BANDS = ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz', 'Red', 'Green', 'Blue', 'Full')   # a band with no harvest installs an empty table; sublst_install_tables.py:15-17
    KEY_PREFIX = {'deep': 'deep', 'sublst': 'sublst'}   # table key prefix (env P3_TABLE_PREFIX for sub-LST); sublst_catalogs.sh:81
    CADENCE = {'deep': 'deep', 'sublst': '10min'}       # env P3_CADENCE of the derivation (floor, beam source); p3s3_curve_derive.py:33
    HARVEST_LABEL = {'deep': 'P3S5', 'sublst': 'SLST{hh}{arm}'}   # env P3S_LABEL; derive.sh:8; sublst_catalogs.sh:24 (LABEL=SLST${LST%h}$ARM); --run overrides
    HARVEST_STATE_DIR = {'deep': '/lustre/gh/main/exopipe_ops/p3s5_state',
                         'sublst': '/fast/gh/main/phase3/sublst_catalogs/state_{sfx}'}   # published_{V,I}.csv (mutable: frozen into the unit's inputs stage); p3s3_sources.py:41; sublst_catalogs.sh:25-27 (lwacalim08 node-local; sfx = arm for 19h else <lst>_<arm>)
    SUBLST_CATALOGS_ROOT = '/fast/gh/main/phase3/sublst_catalogs'   # node-local (lwacalim08) collect layout of the sub-LST catalogue chain; sublst_catalogs.sh:23 (F)
    SUBLST_ARM = 'B'           # the served arm (Run_SLSTB); sublst_catalogs.sh:20
    SUBLST_TENMIN_TREE = {'A': 'tenmin_root', 'B': 'tenmin_root_x2'}   # under paths.SUBLST_ROOT; sublst_catalogs.sh:20 (TREE)
    COLLECT_ROOTS = ('/lustre/gh/main/phase3_collect', '/data07/gh/phase3/transients')   # deep attempt directories (env P3S3_COLLECT_ROOTS); p3s3_sources.py:26
    PARENTS_DIR = {'deep': None, 'sublst': 'parent_catalogs_{lst}'}   # deep None = a fresh current-master union (ops/p3s5_parent_union.py, chain.sh:16); sub-LST: under paths.SUBLST_ROOT (sublst_catalogs.sh:29)
    PARENT_UNION_SOURCES = ('/lustre/gh/main/phase3_collect/p3s5_collect_lwacalim07/parent_catalogs',
                            '/lustre/gh/main/phase3_collect/p3s5_collect_lwacalim09/parent_catalogs',
                            '/data07/gh/phase3/transients/Run_P3S4/parent_catalogs')   # p3s5_parent_union.py:12-13 (LOCK: the literal list of the in-place script)
    EXCLUDED_ROWS = 'report'   # harvest rows on (LST, scope, date) excluded by data_decisions.csv: 'report' = kept and counted (today: not filtered, p3s3_curve_derive.py:83-104); 'refuse' = the derivation refuses while any exist (spec Q2 default = today's)
    PROTECTED_KEYS = ('EXCESS_GUARD_CANDIDATES', 'MONOTONE_CAP', 'PARENT_UNION_SOURCES')
    RULED = {'FLOOR_SIGMA': 'R-2D-TABLES-APPROVED-WITH-VALIDATION-CONDITION',
             'RATIO': ('R-2D-TABLES-APPROVED-WITH-VALIDATION-CONDITION', 'R-TENMIN-BRIGHT-PARENT-TABLES-DERIVATION-APPROVED-PARENTS-DEEP-MASTER-CATALOGUE'),
             'RATIO_LOW_V': 'R-2D-TABLES-APPROVED-WITH-VALIDATION-CONDITION',
             'METRIC': ('R-DEEP-TABLE-DISTANCE-IMAGE-PIXEL-INSTALLED-REPASS4', 'IMPLEMENTED-10MIN-TABLE-IMAGE-PIXEL-METRIC'),
             'ELEV_BANDS_DERIVE': ('R-DEEP-BRIGHT-SOURCE-TABLES-ELEVATION-BANDS-AND-FRAME-S', 'R-SUBLST-TABLES-GO-ELEVATION-BANDS'),
             'ELEV_BANDS_INSTALL': ('R-DEEP-BRIGHT-SOURCE-TABLES-ELEVATION-BANDS-AND-FRAME-S', 'R-SUBLST-TABLES-GO-ELEVATION-BANDS'),
             'ELEV_BANDS_REPORT_ONLY': 'MEASURED-10MIN-PIXEL-TABLE-WITH-ELEVATION-20-25-REPORT-ONLY',
             'EXCESS_GUARD_CANDIDATES': 'MEASURED-10-MINUTE-TABLE-DERIVATION-DRY-RUN-AND-THE-DEEP-POST-PROCESSING-ADOPTED',
             'MONOTONE_CAP': 'MEASURED-10-MINUTE-TABLE-DERIVATION-DRY-RUN-AND-THE-DEEP-POST-PROCESSING-ADOPTED',
             'TARGET_BEAM': 'R-GATE-SEPARATIONS-ANGULAR-AND-DEEP-V-REPAIR-AND-REFERENCE-FIT-TEST-APPROVED',
             'ELEV_FLOOR_DEG': ('R-DEEP-SERVE-ELEVGATE-FLOOR25-INSEARCH-ELEV', 'IMPLEMENTED-10MIN-ELEVATION-FLOOR-25-DEG'),
             'BUILD_KEYS': 'R-DEEP-BRIGHT-SOURCE-TABLES-ELEVATION-BANDS-AND-FRAME-S'}


class calibrate_multiples(_Section):
    """Multiples (repeat-class) tables, deep: the derivation of the /fast derive_conformant.sh / derive_V.sh drivers (in place:
    p3s3_curve_derive.py --multiplicity harvest --classes multi) and the key loop of ops/multiples_fork.py `tables` (:580-637,
    ported). Floor, metric, target beam, elevation floor, post-processing: as calibrate_bright_parent (deep)."""
    RATIO = {'I': 3.0, 'V': 3.0}   # ledger 702 + Gregg 2026-10-08 ("2. yes"); derive_conformant.sh:9; multiples_fork.py:602
    RATIO_LOW_V = 2.0              # Stokes V at calibrate_bright_parent.RATIO_LOW_V_BANDS; derive_conformant.sh:11; multiples_fork.py:59, :602
    CLASSES = 'multi'              # argv --classes; derive_conformant.sh:9
    MULTIPLICITY = 'harvest'       # argv --multiplicity: repeat class from the harvest (refused otherwise, multiples_fork.py:591)
    MATCH_BEAMS = 1.0              # beam widths (argv --match-beams), partner in another (LST, night) cell; multiples_fork.py:54
    EXCLUDE_CELLS = {'I': '/fast/gh/main/phase3/multiples_tables_20261008/excluded_cells_rebuild_list.txt',
                     'V': '/fast/gh/main/phase3/multiples_tables_20261008/excluded_cells_V.txt'}   # env P3_DERIVE_EXCLUDE_CELLS per Stokes (lwacalim07 node-local); derive_conformant.sh:5-6, derive_V.sh:5-6; multiples_fork.py:617
    ELEV_BANDS_DERIVE = '20-25,25-35,35-50,50-90'   # deg; derive_conformant.sh:6
    INSTALL_EMPTY_ALL_ELEVATION = True   # an all-elevation multiples table with no bin is installed empty (nothing removed); multiples_fork.py:609-611
    REPLACE_ALL_MULTI_KEYS = True  # a re-derivation replaces every '-multi_' key; multiples_fork.py:623-624
    CLASS_DEFINITION = ('repeat class: harvest rows that reached the table, partner in another (LST, night) cell within one beam width '
                        '(multiples fork, Gregg 2026-09-06 / 2026-10-08)')   # written on every key; multiples_fork.py:618-619
    RULED = {'RATIO': ('R-FORKED-CHAIN-V1-ON-TODAYS-DATA-CUSTOM-MULTIPLES-TABLES', 'MULTIPLES-FORK-TESTED-AND-TABLES-INSTALLED'),
             'MULTIPLICITY': 'R-MULTIPLES-FORK-INTO-PRODUCTION-PREAMBLE-APPROVED',
             'EXCLUDE_CELLS': 'MULTIPLES-FORK-TESTED-AND-TABLES-INSTALLED'}


class calibrate_scale(_Section):
    """Per-LST VLSSr absolute flux scale: measure = ops/vlssr_flux_check.py IN PLACE (environment below), table =
    ops/pbstep2_20261001/scale_table_v4.py IN PLACE (s_new = s_old / median ratio in the declination window).
    Module literals of vlssr_flux_check (keys at absorption): VLSSr catalogue path (:45), VLSS_MIN_JY 4.0, MATCH_FRAC 0.5,
    ISO_FRAC 0.2, MIN_SUBBANDS 10, PB_MIN 0.3, ALPHA -0.7 (:47), PL_MAX_SCATTER 0.02 (:48)."""
    CAT_RUN = 'Run_P3S4'           # env VLSSR_CAT_RUN: parent catalogues whose digests match the current masters; run_band_vector_20261001.sh:13 (module default Run_P3S3, vlssr_flux_check.py:36)
    PB_ROBUST = 'R075'             # env VLSSR_PB_ROBUST: PB map of the catalogues' weighting; vlssr_flux_check.py:37
    DEC_WINDOW_DEG = (20.0, 60.0)  # deg; ledger 768; scale_table_v4.py:19 (LOCK: the content test recomputes the table with it)
    ERR_MEDIAN_FACTOR = 1.2533     # error of the median = 1.2533 x std / sqrt(N) / median; scale_table_v4.py:29 (LOCK, as above)
    ROUND_DIGITS = {'scale': 5, 'err': 4, 'median_ratio': 5, 'rescale': 5}   # scale_table_v4.py:30 (LOCK, as above)
    REF_DATE_LITERAL = '2026-10-01'   # the date scale_table_v4.py:31 writes literally into 'ref'; a new version's rows carry the measure date instead (named parity amendment A3)
    LST_COUNT = 24                 # rows 00h-23h
    RULED = {'DEC_WINDOW_DEG': 'R-FLUX-SCALE-APPLY-VLSSR-HIGH-ELEVATION-APPROVED'}
    PROTECTED_KEYS = ('DEC_WINDOW_DEG', 'ERR_MEDIAN_FACTOR', 'ROUND_DIGITS', 'REF_DATE_LITERAL')


class calibrate_fluxcorr(_Section):
    """Band correction table k x g (lib/masters/flux_correction.py, in process: read_vector, sh_fit) and its inputs: stage-A
    SH12 per cell (lib/masters/shflux --no-build --table, in place) and the leave-one-band-out band vector (vlssr_flux_check.py
    in place + the ported heredoc of run_band_vector_20261001.sh:16-28)."""
    BANDVEC_TOL = 0.003            # fraction: every band's LOO median within 0.3 per cent; ledger 982; run_band_vector_20261001.sh:27
    BANDVEC_MAX_STEPS = 4          # run_band_vector_20261001.sh:11 (STEPS default; the 2026-10-01 vector needed 7 passes - STEPS was raised by hand, ledger 1714)
    BANDVEC_NORM_BAND = '73MHz'    # k = 1 at 73 MHz; run_band_vector_20261001.sh:21
    BANDVEC_LOO = True             # env VLSSR_LOO_BAND=1; ledger 988; run_band_vector_20261001.sh:13
    BANDVEC_START = None           # start vector; None = tables.BAND_VECTOR (the vector in force; the 2026-10-01 run started from 2026-09-11's, run_band_vector_20261001.sh:10)
    STAGEA_WORKERS = 8             # processes; run_stageA_20261001.sh:5 (--workers overrides)
    SUBBANDS = ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz')   # flux_correction.py:17
    WIDEBAND = ('Red', 'Green', 'Blue', 'Full')   # k = mean of member subbands; flux_correction.py:18, :26-27
    VERSION_TEMPLATE = ('fluxcorr-{version}-{yyyymmdd} (LOO power-law smoothing k_b at 73 MHz=1, PB per robust; global S&H factor g = robust '
                        'median over cells on the {scale_date} per-LST VLSSr scale; wideband k = mean of constituent subbands; BAKED INTO THE PIXELS)')   # flux_correction.py:19-21 (v2 string reproduced exactly: version 'v2', yyyymmdd '20261001', scale_date '2026-10-01')
    TABLE_DECIMALS = 6             # k_band, g_abs, fluxcorr written to 6 decimals (flux_correction.build_table :65-70); the fluxcorr_table
                                   # structure test derives its exact bounds from it: |fluxcorr - k x g| <= 0.5e-6 x (k + g + 1) (round6 of the
                                   # UNROUNDED product; v2 reaches 2.6e-7 of that scale), |k(wideband) - mean k(members)| <= 1e-6 (v2: 4e-7).
                                   # (Was TABLE_TOL 5e-7 = check_rebuild.py:95-96 CARD_TOL, a header-card tolerance never applied here.)
    RULED = {'BANDVEC_TOL': 'R-FLUX-SCALE-POWERLAW-SMOOTHNESS-THEN-SH-FIT', 'BANDVEC_LOO': 'METHOD-BAND-VECTOR-LEAVE-ONE-BAND-OUT-CONVERGENCE'}
    PROTECTED_KEYS = ('SUBBANDS', 'WIDEBAND', 'VERSION_TEMPLATE', 'TABLE_DECIMALS')


class calibrate_template_k(_Section):
    """Template-fit position error scale K per tier and Stokes (the values are fitting.K_TEMPLATE; this section measures them).
    summary = port of kinj_deep.summary (:91-130) over stored trials; sub-LST inject = kcal_tenmin_inject.py IN PLACE.
    Literals of the injection scripts (keys at absorption): S/N grid 5.0-6.5 (kinj_deep.py:24; kcal_tenmin_inject.py:24),
    forced start 0-0.5 beam (:9), >= 12 beams apart, >= 2 beams from |pixel| > 4 rms, > 30 deg elevation (:8)."""
    K_PERCENTILE = 68.0            # per cent: K = 68th percentile of |error| x S/N / FWHM per beam axis; ledger 2285; kinj_deep.py:104
    SNR_KEY = 'snr_true'           # (a) injected S/N on the uniform rms (the deep definition, ledger 2285); kinj_deep.py:105
    BOOTSTRAP_N = 400              # resamplings for the 16-84 per cent interval; kinj_deep.py:110
    BOOTSTRAP_SEED = 1             # kinj_deep.py:108
    K_BOUNDS = (0.0, 5.0)          # open interval; gate G16; sublst_fit_positions.py:148-151
    REPORT_DECIMALS = 3            # K is ratified to 3 decimals (fitting.K_TEMPLATE); kinj_deep.py:116
    # orientation of the template beam the trials were measured with (gate G17 compares it with fitting.TEMPLATE_BEAM_ORIENTATION):
    TRIALS_ORIENTATION = {'deep': 'wcs'}   # the stored trials of tables.K_RECORDS['deep'] (kcal_d5_20261013: fixed template clv15 44f0155, un-mirrored axis projection, deep_forced_kcal_20261013/kinj_deep.py:34; ledger 2322). The 2026-10-11 trials were 'legacy-mirrored' (ledger 2291)
    INJECT_ORIENTATION = {'deep': 'wcs', 'sublst': 'wcs'}   # the in-place harnesses after clv15 44f0155 and their own axis fixes (RULINGS-GREGG-20261013B-MIRROR-EXCLUSIONS-ALL-APPROVED: 'Approve the template mirror fix and re-measure K')
    DEEP_PLAN = '/lustre/gh/main/exopipe_ops/deep_forced_kcal_20261013/plan.csv'   # the 30 frames (10 cells) and seeds of the ratified deep runs (same plan 2026-10-11 and 2026-10-13; ledger 2322); copied into the inject stage
    DEEP_PER_FRAME = 30            # injections per frame (ledger 2322: '30 injections per frame as the ratified 2026-10-11 run'); kinj_deep.py --per-frame
    SUBLST_N_I = 30                # frames; kcal_tenmin_inject.py usage (:21)
    SUBLST_N_V = 15                # frames
    SUBLST_PER_FRAME = 20          # injections per frame
    SUBLST_SEED = 20261012
    SUBLST_TRIALS_FILE = 'trials.csv'   # the trials file kcal_tenmin_inject.py writes in its out dir (:163)
    # trial columns per tier: (status column, value meaning ok, major-axis error, minor-axis error); the adopted definition is
    # the FORCED start with the injected S/N in both tiers (deep kinj_deep.py:97, :105; sub-LST (a) of kcal_tenmin_inject.py:132, ledger 2285)
    TRIALS_COLUMNS = {'deep': ('tf_status', 'ok', 'err_maj_deg', 'err_min_deg'),
                      'sublst': ('a_status', 'ok', 'a_err_maj_deg', 'a_err_min_deg')}
    # where the production code that fits IN PLACE holds each tier's K (legacy root, file, literal): `template-k verify`
    # refuses when config fitting.K_TEMPLATE / K_SNAPSHOT differs from it (review 2026-10-13: a re-measure lands there first)
    INPLACE_K = {'deep': ('exopipe_ops', 'deep_forced_radio.py', 'K_TEMPLATE'),                        # :77 (554b721, ledger 2322)
                 'sublst': ('exopipe_ops', 'sublst_catalogs.sh', 'FITPOS_K'),                          # :216 (ac2d267, ledger 2327)
                 'snapshot': ('clv15', 'phase3_template_fit.py', 'K_MAJOR, K_MINOR, K_MAJOR_HIGH')}    # :42 (a715a1b, ledger 2330)
    RULED = {'K_PERCENTILE': 'R-GREGG-20261013-TENMIN-TEMPLATE-K-MEASURED', 'SNR_KEY': 'R-GREGG-20261013-TENMIN-TEMPLATE-K-MEASURED'}
    PROTECTED_KEYS = ('TRIALS_ORIENTATION', 'INJECT_ORIENTATION', 'TRIALS_COLUMNS', 'INPLACE_K')
