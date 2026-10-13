"""Config draft of `exopipe3 followup` (refactor window; config.py loads it until the refactor chat folds it in).

Spec: /lustre/gh/main/exopipe_ops/refactor_20261012/plan/specs/followup.md section 3 (values and lines checked again
2026-10-13 ~05:00 UTC against exopipe_ops 554b721, exopipe main ac3d841, clv15 cb371c9). Abbreviations of the source
files (all IN PLACE during the window, DESIGN 11.1):
  EP  = exopipe_ops/einstein_probe/ep_wxt_plan_fetch.py (e8338e6: pointing bound 6 h)
  TL  = exopipe_ops/tess/tess_lightcurve.py              HO = exopipe_ops/phase3_honing.py
  G2  = exopipe_ops/steep_catalog/stage2g2_lightcurves_p3.py   H4 = .../stage2h4_periodicity_win.py
  PG  = exopipe_ops/steep_catalog/stage2i_periodicity_page.py  SF = .../steep_fold.py   SD = .../source_diag_page.py
  FC  = exopipe/phase3/followup/config.py (package, imported in-process by `investigate` / `summary`)
"Ledger N" = line N of clv15/fix_ledger.csv.

Guards (config.py honours PROTECTED_KEYS and RULED; steps/followup.py honours `pinned_inputs`):
- PROTECTED_KEYS: (a) production roots (use --out instead) and (b) values that MIRROR a literal inside the in-place code:
  an override would be recorded but not applied, so no --set / --config can change them (spec 'WINDOW-LOCKED', gate G6).
  They move out of PROTECTED_KEYS key by key when the consumer is absorbed into lib/followup/. The step also checks, at
  start, that the in-place sources still carry the mirrored literals named in followup.MIRRORS (gate G6b).
- RULED: a ruling of Gregg's set the value AND the step passes it to the in-place code (argv or environment), so an
  override is applied: allowed with --ruling-for section.KEY=<ledger ID> or into a non-production --out.
- pinned_inputs (lower case = not a tunable): inputs a ruling or a delivered product names; the step refuses an override
  unless --out is outside every production root (paths.PRODUCTION_ROOTS and followup.PRODUCTION_ROOTS).
"""
from exopipe.phase3.config import _Section


class followup(_Section):
    """exopipe3 followup - values shared by the follow-up sub-operations (architecture 9; steps/followup.py)."""
    # Production roots the step writes in normal operation. While paths.PRODUCTION_WRITES is False the step refuses to write
    # under any of them (gate G7; cli._guard_out covers only paths.PRODUCTION_ROOTS, which lacks the first and the last two).
    PRODUCTION_ROOTS = ('/lustre/gh/main/catalogs/einstein_probe/wxt_plan',          # ep-plan (crontab argument; ledger 1766 IMPL-EP-WXT-PLAN-ARCHIVE)
                        '/fast/gh/main/phase3/transients',                          # investigate / summary (FC:18, :43; lwacalim07 node-local)
                        '/lustre/gh/main/phase3/periodicity5',                      # periodicity lightcurves (run_lc5.sh:4)
                        '/lustre/gh/main/exopipe_ops/steep_catalog/battery_work',   # periodicity search + merged table (run_p4_fleet.sh:7)
                        '/lustre/gh/main/exopipe_ops')                              # tess products (TL:7 usage: --out .../exopipe_ops/tess_<key>)
    # Exclusions in follow-up inputs. No follow-up code reads data_decisions.csv today (0 readers in followup/, HO, the
    # steep_catalog scripts, tess/, einstein_probe/). 'report' = today's behaviour plus, on every unit with nights, the
    # admitted night set and the excluded nights it used (core.exclusions). 'filter' does not exist in the window (the
    # in-place light-curve script globs every night of the generation, G2:36); it would change the pass-5 tables.
    EXCLUSION_MODE = 'report'
    # site.PYTHON keys (site vocabulary 'default' / 'tess'; paths live in the site file, DESIGN 11.4/11.7)
    PYTHON_KEY = 'default'       # py38_orca_nkosogor (run_p4_fleet.sh:18, run_lc5.sh:8, attic/hwj_candinv_run.sh:4; healpy 1.16.5 for EP)
    TESS_PYTHON_KEY = 'tess'     # /fast/gh/main/envs/tess/bin/python (Python 3.12), lwacalim07 only (TL:5)
    # The in-place scripts, relative to the site's legacy root 'exopipe_ops' (core.env.legacy_root); run as production runs
    # them (argv and environment), through core.legacy.run.
    SCRIPTS = {'ep_plan': 'einstein_probe/ep_wxt_plan_fetch.py',
               'tess': 'tess/tess_lightcurve.py',
               'hone': 'phase3_honing.py',
               'lightcurves': 'steep_catalog/stage2g2_lightcurves_p3.py',
               'search': 'steep_catalog/stage2h4_periodicity_win.py',
               'page': 'steep_catalog/stage2i_periodicity_page.py',
               'fold': 'steep_catalog/steep_fold.py',
               'source_page': 'steep_catalog/source_diag_page.py'}
    # Literals the in-place sources must still contain for the mirrored keys to be true (gate G6b, checked at step start
    # for the scripts the chosen op runs): {script key: ((config key, text that must occur in the source), ...)}.
    MIRRORS = {'ep_plan': (('followup_ep_plan.URL', "URL = 'https://ep.bao.ac.cn/ep/observation_plan/download_obsplan_fov'"),
                           ('followup_ep_plan.NSIDE', 'NSIDE = 64'),
                           ('followup_ep_plan.POINTING_MIN_MIN', 'timedelta(minutes=5) <= t1 - t0'),
                           ('followup_ep_plan.POINTING_MAX_H', 't1 - t0 <= timedelta(hours=6)'),
                           ('followup_ep_plan.FOOTPRINT_MIN_DEG2', 'if not 3000 <= area <= 4500'),
                           ('followup_ep_plan.MIN_SPAN_H', 'if span < 20'),
                           ('followup_ep_plan.DOWNLOAD_TIMEOUT_S', 'urlopen(URL, timeout=300)')),
               'hone': (('followup_hone.REPRO_TOL', 'REPRO_TOL = 0.45'),
                        ('followup_hone.STRIATION_WINDOW_BEAMS', 'STRIATION_WINDOW_BEAMS = 20.0'),
                        ('followup_hone.INPUT_ROOT', "P3ROOT = '/data07/gh/phase3'")),
               'lightcurves': (('followup_periodicity.LC_ELEV_MIN_DEG', 'keep = el >= 20.0'),),
               'search': (('followup_periodicity.SHORT_PERIOD_D', 'PMIN,PMAX=0.05,2.0'),
                          ('followup_periodicity.LONG_PERIOD_D', 'PMIN_L,PMAX_L=2.0,60.0'),
                          ('followup_periodicity.SHORT_NSPEC', 'NSPEC=6000'),
                          ('followup_periodicity.LONG_NSPEC', 'NSPEC_L=2000'),
                          ('followup_periodicity.ATEAM_AVOID_DEG', "ATEAM_DEG={'V':3.0,'I':5.0}[STOKES]"),
                          ('followup_periodicity.NULL_SEED', '_random.seed(7)')),
               'fold': (('followup_fold.PDM_BINS', 'def pdm(P, nb=25)'),
                        ('followup_fold.MAD_SIGMA', 'abs(f[i]-m_) > 5*mad'),
                        ('followup_fold.MAD_WINDOW_PHASE', '(d <= 0.02) & (m != i)'))}
    # Observatory location literals duplicated in the in-place scripts (B-42 keeps copies until answered)
    OVRO_LAT_DEG = 37.2398       # deg; TL:42, SF:15, H4:71, G2:28 (pipeline_config: 37.23977727)
    OVRO_LON_DEG = -118.2817     # deg; same lines
    OVRO_HEIGHT_M = 1222.0       # m; TL:42, SF:15
    PROTECTED_KEYS = ('PRODUCTION_ROOTS', 'EXCLUSION_MODE', 'PYTHON_KEY', 'TESS_PYTHON_KEY', 'SCRIPTS', 'MIRRORS',
                      'OVRO_LAT_DEG', 'OVRO_LON_DEG', 'OVRO_HEIGHT_M')


class followup_ep_plan(_Section):
    """ep-plan: archive of the Einstein Probe WXT 24-hour PLANNED footprint (Gregg 2026-10-03 'proceed with (c) only for now
    until the data release'; ledger 1766 IMPL-EP-WXT-PLAN-ARCHIVE). Consumer: EP, run as the crontab runs it (in place)."""
    ARCHIVE_ROOT = '/lustre/gh/main/catalogs/einstein_probe/wxt_plan'   # production root (crontab argument; ledger 1766)
    URL = 'https://ep.bao.ac.cn/ep/observation_plan/download_obsplan_fov'   # EP:27
    NSIDE = 64                       # HEALPix NSIDE of the stored cells; EP:28 (cell area 0.8393 deg^2)
    DOWNLOAD_TIMEOUT_S = 300         # s; EP:112
    POINTING_MIN_MIN = 5             # min, shortest accepted pointing; EP:53
    POINTING_MAX_H = 6.0             # h, longest accepted pointing; EP:53 (e8338e6). Gregg 2026-10-13 'Which ever you recommend'
                                     #   = 6 h (ledger 2316 R-GREGG-20261013-RULINGS-ON-PENDING-DECISIONS item 11; was the unruled 3 h)
    CELL_CENTRE_TOL_DEG = 1e-5       # deg, a cell centre off an NSIDE-64 centre refuses; EP:60, :94
    FOOTPRINT_MIN_DEG2 = 3000.0      # deg^2 per pointing; EP:65 (measured 3,818.8, EP:13)
    FOOTPRINT_MAX_DEG2 = 4500.0      # deg^2 per pointing; EP:65
    MIN_SPAN_H = 20.0                # h, plan first start to last end; EP:68
    KEEP_REFUSED_RAW = True          # refused download kept as refused_<tag>.json.gz; EP:118-125 (M19, exopipe_ops fd22433)
    RAW_FIRST_PER_UTC_DAY = True     # raw_<tag>.json.gz for the first passing download of each UTC day; EP:135-138
    # Schedule (documentation: the gh crontab on lwacalim07 holds it during the window, ledger 1766 'cron 41 */3' and the daily
    # audit 06:55; at cutover the two lines call `exopipe3 followup ep-plan --action fetch|audit`)
    FETCH_CRON = '41 */3 * * *'
    AUDIT_CRON = '55 6 * * *'
    PROTECTED_KEYS = ('ARCHIVE_ROOT', 'URL', 'NSIDE', 'DOWNLOAD_TIMEOUT_S', 'POINTING_MIN_MIN', 'POINTING_MAX_H',
                      'CELL_CENTRE_TOL_DEG', 'FOOTPRINT_MIN_DEG2', 'FOOTPRINT_MAX_DEG2', 'MIN_SPAN_H', 'KEEP_REFUSED_RAW',
                      'RAW_FIRST_PER_UTC_DAY', 'FETCH_CRON', 'AUDIT_CRON')


class followup_tess(_Section):
    """tess: TESS light curve of a star around an OVRO-LWA radio detection (the tess-light-curve skill, Gregg 2026-10-03 'The
    creation of TESS light curves should be a skill'; ledger 1756 IMPL-TESS-COVERAGE-ON-PAGES-AND-LIGHTCURVE-SKILL).
    Consumer: TL, run under site.PYTHON['tess'] (in place). Every value below is a literal of TL."""
    OUT_PARENT = '/lustre/gh/main/exopipe_ops'   # production parent; product dir = <OUT_PARENT>/<key> (TL:7 usage)
    ORBIT_TABLE = '/lustre/gh/main/exopipe_ops/tess/TESS_orbit_times.csv'   # TL:43 (refreshed in place by pages' tess_coverage.py
                                     #   at 7 days age; uncommitted edit in exopipe_ops today: the unit fingerprints it as an input)
    PRODUCTS = (('SPOC', 120), ('SPOC', 20), ('QLP', 200))   # (author, cadence s) downloaded; TL:133
    TIC_RADIUS_DEG = 0.003           # deg, TIC cone around the J2000 position; TL:110
    LS_MIN_FREQ_PER_D = 0.05         # 1/d (periods <= 20 d); TL:63
    LS_MAX_FREQ_PER_D = 24.0         # 1/d; TL:63
    TREND_HALF_WINDOW_D = 0.25       # d (0.5-d running median); TL:64
    EXCURSION_SIGMA = 5.0            # x residual scatter (1.4826 MAD); TL:66
    EXCURSION_MERGE_D = 0.05         # d, excursions closer than this merge; TL:67
    ZOOM_20S_HALF_D = 0.5            # d around the radio epoch for 20-s data; TL:156
    PLOT_ZOOM_HALF_D = 2.0           # d (4-day zoom panel); TL:169
    TESSCUT_SIZE_PX = 15             # px, own-aperture cutout; TL:142
    APERTURE_THRESHOLD = 3.0         # sigma, threshold aperture; TL:143
    BACKGROUND_THRESHOLD = 0.001     # background mask threshold; TL:143
    NET_TIMEOUT_S = 30               # s, exoplanet archive and TESScut sector service; TL:56, :90
    FIG_DPI = 110                    # TL:187
    # content test tess_lightcurve: BTJD range of a plausible TESS cadence (TESS science began BTJD ~1325, 2018-07-25)
    BTJD_RANGE_D = (1000.0, 5000.0)  # d (BJD_TDB - 2457000); spec section 6
    PROTECTED_KEYS = ('OUT_PARENT', 'ORBIT_TABLE', 'PRODUCTS', 'TIC_RADIUS_DEG', 'LS_MIN_FREQ_PER_D', 'LS_MAX_FREQ_PER_D',
                      'TREND_HALF_WINDOW_D', 'EXCURSION_SIGMA', 'EXCURSION_MERGE_D', 'ZOOM_20S_HALF_D', 'PLOT_ZOOM_HALF_D',
                      'TESSCUT_SIZE_PX', 'APERTURE_THRESHOLD', 'BACKGROUND_THRESHOLD', 'NET_TIMEOUT_S', 'FIG_DPI')


class followup_investigate(_Section):
    """investigate / summary: the candidate dossier (the 'candinv' harness absorbed 2026-08-12, commit 17787c8; design Gregg
    2026-06-26/28 in code comments). Consumer: exopipe.phase3.followup.* (package, in-process). The path keys marked ENV are
    passed to the package as CANDINV_<NAME> before it is first imported (FC:14-15, followup/_cfg_adapter.py, gate G13), so
    an override IS applied; every other key mirrors a literal of the package (PROTECTED_KEYS)."""
    # ---- inputs and outputs (ENV)
    TOP50_CSV = '/fast/gh/main/phase3/transients/wideband_V_search/top50_V_transients.csv'   # ENV; pinned input; FC:19-22
                                     #   (lwacalim07 node-local; 4,661 bytes, 2026-06-26)
    BUNDLE_DIR = '/data07/gh/top50_allsky_tmp'   # ENV; pinned input (B-41); FC:24 (lwacalim07; the only copy of the June
                                     #   wideband deep residuals)
    PROD_ROOT = '/data07/gh/phase3'  # ENV; FC:27 (the P3_PROD / P3_REDO1 per-epoch trees under it no longer exist)
    PROD_LABEL = 'PROD'              # ENV; FC:28
    REDO1_LABEL = 'REDO1'            # ENV; FC:32 (wideband deep from REDO1 first, resolve.py:48-54)
    PHASE2_ARCHIVE = '/lustre/pipeline/exopipe/phase2'   # ENV; FC:35
    PHASE2_FAST = '/fast/gh/main/phase2'                 # ENV; FC:38
    OUT_ROOT = '/fast/gh/main/phase3'                    # ENV; production root (with --out X the step passes X); FC:43
    OUT_LABEL = 'PROD'                                   # ENV; FC:44 -> <OUT_ROOT>/transients/Run_PROD/
    OUT_SUBDIR = 'candinv'                               # ENV; FC:48 (Gregg 2026-06-28: not the shared candidates/ tree)
    VLSS_UNRESOLVED = '/lustre/gh/calibration/pipeline/reference/surveys/vlssr_radecpeak_unresolved.txt'   # ENV; FC:64-65
    FLARE_STAR_CSV = '/lustre/gh/main/exopipe/exopipe/phase3/flare_ucd.csv'   # ENV; = FC:87 resolved in the PRODUCTION checkout
                                     #   (absent there, as in the branch: the CLV defect below)
    ATNF_CSV = '/lustre/gh/main/exopipe/exopipe/phase3/atnf_pulsars.csv'      # ENV; FC:88 (absent)
    # Catalogue directory of the cross-match: FC:69 CLV = dirname(dirname(this file)) = exopipe/phase3 since the 2026-08-12
    # absorption (in clv15/candinv it was clv15). None of the 6 catalogues exists there, every dossier's context.xmatch is
    # empty and score.xmatch_stellar is always 0. Today's behaviour is KEPT in the window (batch row C-33 / B-41 covers it);
    # the step writes a run-record note on every unit. TARGET_CSV_DIR is read only after absorption.
    TARGET_CSV_DIR = '/lustre/gh/main/clv15'   # dir of 10pc.csv, 25pc.csv, 50pc.csv, flare_ucd.csv, Hot_Warm_Jupiters.csv, LPTs.csv
    XMATCH_KEYS = ('10pc', '25pc', '50pc', 'flare_ucd', 'hwj', 'lpt', 'atnf')   # FC:83
    XMATCH_POSITIVE = ('10pc', '25pc', '50pc', 'flare_ucd', 'hwj', 'lpt')       # FC:85
    # ---- thresholds (literals in the in-place modules)
    DETECT_SNR = 6.0                 # sigma, per-band forced detection; FC:113
    STRONG_SNR = 7.0                 # sigma, strong detection (matches the blind-search FP gate); FC:114
    ELEV_MIN_DEG = 20.0              # deg, low-elevation flag and score; FC:115
    ELEV_GOOD_DEG = 45.0             # deg, benign elevation in the score; FC:157
    ESCALATE_MIN_BANDS = 2           # bands, T4 multi-epoch trigger; FC:127 (never the only trigger: dossier.py:165-172)
    SNAPSHOT_VAR_FRAC = 0.5          # fraction, labels short_structure only; FC:133 (Gregg 2026-06-28: T5 always runs)
    SNAPSHOT_PAD_S = 60.0            # s, pad around the peak 10-minute bin; FC:134
    VI_RATIO_UNPHYSICAL = 1.25       # |V|/I above this = artifact; FC:144 (Gregg 2026-06-26)
    REG_RADIUS_DEG = 1.0             # deg, every DS9 circle; FC:147 (standing feedback); content test followup_region
    MORPH_POINT_RATIO = 1.35         # observed / empirical FWHM below this = point-like; FC:156; copies tests.py:76, snapshots.py:331-332
    MORPH_RESOLVED_RATIO = 1.5       # at or above = resolved; FC:155; copies tests.py:77, snapshots.py:333-334
    ATEAM_VETO_DEG = 2.0             # deg, A-team hard veto P = 0; FC:162, score.py:125-128 (Gregg 2026-06-28)
    ATEAM_PENALTY_DEG = 5.0          # deg, soft -2.0 within; score.py:115
    NOMINAL_BEAM_PX = 3.2            # px, beam when the discovery header has no BMAJ; dossier.py:82-83 (Gregg 2026-07-13)
    SNAPSHOT_FALLBACK_SCOPE = '55MHz'   # dossier.py:240 (Gregg 2026-06-28)
    CUTOUT_HALF_PX = 80              # px (161 x 161 cutout); io_products.py:79; content test followup_fits
    PRODUCT_TAPER = 'Taper'          # name token of every dossier FITS; io_products.py:24
    PRODUCT_ROBUST = 0.0             # name token of every dossier FITS; io_products.py:25
    P2RUN = 'PROD'                   # header P2RUN card; io_products.py:43, :56
    FIG_DPI = 90                     # dossier.py:519, :579
    ASTROM_NEAR_DEG = 15.0           # deg, anchors around the candidate; astrometry.py:77
    ASTROM_SNR_FACTOR = 3.0          # anchor S/N floor = max(15, 3 x transient S/N); astrometry.py:77, :87
    ASTROM_MIN_SNR_FLOOR = 15.0      # sigma; astrometry.py:78
    ASTROM_VLSS_RADIUS_ARCMIN = 2.0  # arcmin; astrometry.py:78, :89
    ASTROM_BDSF_THRESH = (8.0, 5.0)  # PyBDSF thresh_pix, thresh_isl; astrometry.py:51
    ASTROM_MAX_SOURCES = 400         # astrometry.py:46
    ASTROM_MIN_ANCHORS = 3           # astrometry.py:99, :103 (the ruled N >= 10 / MAD <= 1.5 arcmin gate is not implemented: C-33)
    PERSISTENT_MIN_EPOCHS = 4        # epochs; tests.py:290
    PERSISTENT_MIN_FRAC = 0.8        # fraction detected; tests.py:290
    CONTEXT_ELEV_MASK_DEG = 15.0     # deg, horizon mask of the peak count; tests.py:331
    CONTEXT_PEAK_SIGMA = 7.0         # sigma, peak count; tests.py:376
    XMATCH_RADIUS_74MHZ_ARCMIN = 6.0 # arcmin at 74 MHz, x (74 / nu), clipped 3-20 arcmin; tests.py:380-382
    SNAP_NOMINAL_BEAM_PX = 3.5       # px; snapshots.py:205, :308
    SNAP_DEFAULT_CDELT_DEG = (0.04, 0.037)   # deg (spectrum, morphology); snapshots.py:204, :307
    SNAP_RADIUS_74MHZ_ARCMIN = 6.0   # arcmin x (74 / nu), clipped 2-30 arcmin; snapshots.py:197, :359
    SNAP_FUNPACK_RETRIES = 3         # snapshots.py:230
    SNAP_FUNPACK_TIMEOUT_S = 180     # s; snapshots.py:247
    SNAP_FPACK_ENV = 'development'   # conda env of funpack; snapshots.py:31
    SNAP_PHASE1_ROOT = '/lustre/pipeline/exopipe/phase1'   # snapshots.py:45 (duplicate of paths.PHASE1_ROOT)
    SNAP_SCRATCH_PARENT = '/fast/gh/main'   # node-local scratch parent of the T5 arm; snapshots.py:135, :183, :298, :382
    # P(astrophysical) log-odds (score.py:63-118; Gregg 2026-06-28 'rank by P(astrophysical) ONLY'); the sum is the logit
    SCORE_LOGODDS = {'morph_extended': -2.5, 'morph_point': 1.2, 'snap_morph_extended': -3.0, 'snap_morph_point': 1.8,
                     'unphysical_vi': -3.0, 'iv_coincident_max': 1.6, 'iv_coincident_vi_scale': 0.5,
                     'iv_coincident_other': 0.5, 'snapshot_burst': 1.0, 'multi_band_per_band': 0.8, 'multi_band_max': 2.0,
                     'multi_epoch_persistent': -2.0, 'multi_epoch_per_epoch': 0.6, 'multi_epoch_max': 1.5,
                     'xmatch_stellar': 1.0, 'elev_low': -1.5, 'elev_mid': -0.5, 'elev_good': 0.5, 'ateam_soft': -2.0,
                     'ephemeris': -2.5, 'striation_gt100': -1.0, 'striation_gt50': -0.5}
    # content test followup_dossier: |p_astro - 1/(1+exp(-logit))| allowed (score.py rounds p to 3 and the logit to 2 places)
    P_ASTRO_TOL = 0.002
    PROTECTED_KEYS = ('OUT_ROOT', 'TARGET_CSV_DIR', 'XMATCH_KEYS', 'XMATCH_POSITIVE', 'DETECT_SNR', 'STRONG_SNR',
                      'ELEV_MIN_DEG', 'ELEV_GOOD_DEG', 'ESCALATE_MIN_BANDS', 'SNAPSHOT_VAR_FRAC', 'SNAPSHOT_PAD_S',
                      'VI_RATIO_UNPHYSICAL', 'REG_RADIUS_DEG', 'MORPH_POINT_RATIO', 'MORPH_RESOLVED_RATIO', 'ATEAM_VETO_DEG',
                      'ATEAM_PENALTY_DEG', 'NOMINAL_BEAM_PX', 'SNAPSHOT_FALLBACK_SCOPE', 'CUTOUT_HALF_PX', 'PRODUCT_TAPER',
                      'PRODUCT_ROBUST', 'P2RUN', 'FIG_DPI', 'ASTROM_NEAR_DEG', 'ASTROM_SNR_FACTOR', 'ASTROM_MIN_SNR_FLOOR',
                      'ASTROM_VLSS_RADIUS_ARCMIN', 'ASTROM_BDSF_THRESH', 'ASTROM_MAX_SOURCES', 'ASTROM_MIN_ANCHORS',
                      'PERSISTENT_MIN_EPOCHS', 'PERSISTENT_MIN_FRAC', 'CONTEXT_ELEV_MASK_DEG', 'CONTEXT_PEAK_SIGMA',
                      'XMATCH_RADIUS_74MHZ_ARCMIN', 'SNAP_NOMINAL_BEAM_PX', 'SNAP_DEFAULT_CDELT_DEG',
                      'SNAP_RADIUS_74MHZ_ARCMIN', 'SNAP_FUNPACK_RETRIES', 'SNAP_FUNPACK_TIMEOUT_S', 'SNAP_FPACK_ENV',
                      'SNAP_PHASE1_ROOT', 'SNAP_SCRATCH_PARENT', 'SCORE_LOGODDS')
    pinned_inputs = ('TOP50_CSV', 'BUNDLE_DIR')
    # The forced fitter's own values (box 1.5 beams floor, 40 px cap, 6.0 px fallback beam, 3 x rms upper limit;
    # followup/photometry.py:48, 85-90, 160) belong to the `fitting` section (core owner; MOVE_PLAN row 14).


class followup_hone(_Section):
    """hone: time-frequency honing by image-plane stacking (architecture 9.2; ledger 382 PHASE3-HONING-92-IMPLEMENTED, 387
    HONING-STRIATION-ORDER-FIX). Consumer: HO (in place, subprocess). DEFAULT_BAND and N_CONTROLS are passed as arguments;
    the rest are literals of HO."""
    INPUT_ROOT = '/data07/gh/phase3'                       # HO:52 (its P3_* trees were deleted 2026-09-01: every candidate ends no-products)
    INPUT_TREES = ('P3_SB10', 'P3_PROD', 'P3_PROD10', 'P3_REDO1')   # searched in this order; HO:63
    INPUT_GLOB = '*msub*bin*.fits'                         # 10-minute Stokes V msub frames; HO:65
    DEFAULT_BAND = 'Blue'            # HO:271 (--band)
    N_CONTROLS = 8                   # control positions; HO:272 (--controls)
    REGION_HALF_PX = 256             # px (513 x 513 working region); HO:53
    STAMP_HALF_PX = 80               # px (161 x 161 stamp); HO:54; content test honing_stamp
    REPRO_TOL = 0.45                 # | |S/N own| - |S/N disc| | / |S/N disc|, same sign required; HO:55, :212-213 (ledger 382;
                                     #   architecture 9.2 asks position + flux + S/N: only S/N here, batch row C-33)
    CONTROL_OFFSETS_PX = ((55, 0), (-55, 0), (0, 55), (0, -55), (40, 40), (-40, 40), (40, -40), (-40, -40))   # px; HO:56-57
    STRIATION_WINDOW_BEAMS = 20.0    # beams; HO:47 (production search uses 8 beams since ledger 395; honing not re-ruled, C-33)
    NOISE_ANNULUS_PX = (10, 60)      # px, clipped-std annulus; HO:99
    VALUE_BOX_PX = 7                 # px, signed peak of |value| in a 7 x 7 box; HO:132
    BEAM_FALLBACK = (0.25, 0.03, 3.0, 7.0)   # BMAJ deg, CDELT deg, min beam px, beam px on exception; HO:112-115
    EMPIRICAL_P_PRINT = 0.111        # p <= 1/9 printed as "empirically significant" (print only); HO:301
    PROTECTED_KEYS = ('INPUT_ROOT', 'INPUT_TREES', 'INPUT_GLOB', 'REGION_HALF_PX', 'STAMP_HALF_PX', 'REPRO_TOL',
                      'CONTROL_OFFSETS_PX', 'STRIATION_WINDOW_BEAMS', 'NOISE_ANNULUS_PX', 'VALUE_BOX_PX', 'BEAM_FALLBACK',
                      'EMPIRICAL_P_PRINT')
    RULED = {'N_CONTROLS': 'PHASE3-HONING-92-IMPLEMENTED'}


class followup_periodicity(_Section):
    """periodicity: the 10-minute wideband periodicity search on catalogued positions, pass 5 (Gregg 2026-09-20/21; ledger 1362
    R-PERIODICITY-SCOPE-WIDEBAND-10MIN-CATALOGUED-POSITIONS-FOR-NOW, 1370 R-PERIODICITY-P4-LONG-WINDOW, 1374
    R-PERIODICITY-POINT-NOISE-GATE, 1375 R-PERIODICITY-ATEAM-JUPITER-AVOIDANCE, 1393 R-PERIODICITY-P5-FREE-POSITION-SERVED).
    Consumers (in place, subprocess): G2 (lightcurves), H4 (search), PG (page). This is NOT the ruled exoplanet-class
    periodicity of architecture 9.4 (known-period-first, Tier-2 forced photometry, all tiers): that design is pending.
    No period range is changed without Gregg's word (ledger 1267 PROCESS-ERROR-UNAUTHORISED-PERIOD-CAP-IN-THE-PERIODICITY-RUN-2)."""
    # ---- lightcurves (G2)
    LC_POSITIONS = '/lustre/gh/main/exopipe_ops/steep_catalog/battery_work/periodicity_positions.csv'   # pinned input; argv 1;
                                     #   run_lc5.sh:5 (114,556 positions; untracked file in exopipe_ops)
    LC_GENERATION = 'Run_20260909_000000'                  # env P3_GEN; G2:26 (a different generation is a new science product)
    LC_ARCHIVE_ROOT = '/lustre/pipeline/exopipe/phase3'    # env P3_ARCHIVE_ROOT; G2:27
    LC_ROOT = '/lustre/gh/main/phase3/periodicity5'        # production root of lightcurves = input of search; run_lc5.sh:4
    BANDS = ('Red', 'Green', 'Blue')                       # band axis (run_lc3.sh:5); ledger 1362 (Full, subbands, 1-h out of scope)
    LC_LANES = 8                                           # parallel (LST, band) units per process; run_lc5.sh:8 (xargs -P 8); parallelism only
    LC_ELEV_MIN_DEG = 20.0           # deg, position kept when above at the cell's mid-LST; G2:34
    LC_FREE_POSITION_MIN_SNR = 5.0   # free-position amplitude accepted at >= 5 x the frame's local noise; G2:117 (ledger 1393)
    # ---- search (H4); passed as environment exactly as run_p4_fleet.sh:12 does (POINT_NOISE_GATE = H4's own default 3)
    SEARCH_OUT = '/lustre/gh/main/exopipe_ops/steep_catalog/battery_work/periodicity_p5'   # production root; run_p4_fleet.sh:8
    ELEV_FLOOR_DEG = 40.0            # deg, per cell at mid-LST; env ELEV_FLOOR (run_p4_fleet.sh:12; H4 default 50, H4:71); ledger 1362
    MIN_BLOCKS = 100                 # 10-minute blocks per source; env MIN_BLOCKS (run_p4_fleet.sh:12, H4:71); ledger 1362
    MIN_NIGHTS_LONG = 20             # nights for the long pass; env MIN_NIGHTS_LONG (run_p4_fleet.sh:12, H4:49); ledger 1370
    FRAME_NOISE_GATE = 3.0           # x cell median frame noise; env FRAME_NOISE_GATE (run_p4_fleet.sh:12, H4:59; Gregg 2026-09-05)
    POINT_NOISE_GATE = 3.0           # x source median point noise in the cell; env POINT_NOISE_GATE (H4:70); ledger 1374
    NULL_N = 120                     # light curves sampled per cell-set group for the window null; env NULL_N (run_p4_fleet.sh:12, H4:50)
    NCPU = 28                        # processes; run_p4_fleet.sh:13 (parallelism only; capped by --cores / free cores)
    DEFAULT_JOBS = ('Red_I', 'Green_I', 'Blue_I')   # run_p4_fleet.sh:14 (P5_JOBS default; Stokes V needs --stokes V)
    SHORT_DURATIONS_D = (0.007, 0.014, 0.021, 0.042)   # d box durations; H4:39
    SHORT_PERIOD_D = (0.05, 2.0)     # d; H4:39; ledger 1370; never changed without Gregg (ledger 1267)
    SHORT_NSPEC = 6000               # trial periods (log grid); H4:39
    LONG_DURATIONS_D = (1.0, 2.0, 4.0)   # d; H4:47
    LONG_PERIOD_D = (2.0, 60.0)      # d, each source capped at half its own baseline; H4:47; ledger 1370 / 1267
    LONG_NSPEC = 2000                # H4:47
    ATEAM_AVOID_DEG = {'V': 3.0, 'I': 5.0}   # deg, source not searched; H4:127; ledger 1375
    JUPITER_AVOID_DEG = {'Red': 5.0, 'Green': 0.5, 'Blue': 0.5, 'Full': 5.0}   # deg, per block at its own time; H4:130; ledger 1375
    NULL_SEED = 7                    # random.seed of the null sample; H4:323
    SOURCES_CSV = '/lustre/pipeline/exopipe/phase3/Catalog/Run_20260909_000000/allsky/sources.csv'   # H4:73
    BRIGHT_41MHZ_JY = (15.0, 80.0)   # Jy, S-code sources in this 41 MHz peak range form the 'bright' set; H4:74-77
    # ---- page (PG); FRAME_NOISE_GATE, POINT_NOISE_GATE and ELEV_FLOOR are passed to it too (PG:174-176: panel = searched series)
    MERGED_DIR = '/lustre/gh/main/exopipe_ops/steep_catalog/battery_work'   # production root of allsky_periodicity_<S>.csv (pass 5,
                                     #   2026-09-21 03:04 UTC; the archived Catalog/.../allsky/allsky_periodicity_<S>.csv are pass 4)
    PAGE_SNR_FLOOR = 8.0             # PG:50 (env P3_SNR_FLOOR, never set by production: default)
    PAGE_WINZ_FLOOR = 5.0            # PG:56 (env P3_WINZ_FLOOR, default)
    PROTECTED_KEYS = ('LC_GENERATION', 'LC_ROOT', 'SEARCH_OUT', 'MERGED_DIR', 'LC_ELEV_MIN_DEG', 'LC_FREE_POSITION_MIN_SNR',
                      'SHORT_DURATIONS_D', 'SHORT_PERIOD_D', 'SHORT_NSPEC', 'LONG_DURATIONS_D', 'LONG_PERIOD_D', 'LONG_NSPEC',
                      'ATEAM_AVOID_DEG', 'JUPITER_AVOID_DEG', 'NULL_SEED', 'SOURCES_CSV', 'BRIGHT_41MHZ_JY', 'PAGE_SNR_FLOOR',
                      'PAGE_WINZ_FLOOR')
    RULED = {'LC_POSITIONS': 'R-PERIODICITY-SCOPE-WIDEBAND-10MIN-CATALOGUED-POSITIONS-FOR-NOW',
             'BANDS': 'R-PERIODICITY-SCOPE-WIDEBAND-10MIN-CATALOGUED-POSITIONS-FOR-NOW',
             'ELEV_FLOOR_DEG': 'R-PERIODICITY-SCOPE-WIDEBAND-10MIN-CATALOGUED-POSITIONS-FOR-NOW',
             'MIN_BLOCKS': 'R-PERIODICITY-SCOPE-WIDEBAND-10MIN-CATALOGUED-POSITIONS-FOR-NOW',
             'MIN_NIGHTS_LONG': 'R-PERIODICITY-P4-LONG-WINDOW',
             'POINT_NOISE_GATE': 'R-PERIODICITY-POINT-NOISE-GATE'}
    pinned_inputs = ('LC_POSITIONS',)


class followup_fold(_Section):
    """fold: barycentre, PDM period search, MAD-cleaned fold of one 10-minute light curve (Gregg 2026-09-19 'Proceed' on
    barycentring; ledger 1340 MEASURED-BARYCENTRED-ECLIPSE-PERIODS-J1810-AND-B1957, 1336 DELIVERED-J1810-1744-FOLD-MAD-CLEANED-
    AND-SMOOTHED-PROFILE). Consumer: SF (in place). Not the ruled folding of architecture 9.5 (association evidence, window
    function): that design is pending."""
    PDM_BINS = 25                    # phase bins; SF:20
    COARSE_STEP_D = 2e-6             # d, global grid 0.08 d .. max(0.4, 2.2 x P_cat); SF:26
    COARSE_MIN_D = 0.08              # d; SF:26
    FINE_HALF_D = 3e-4               # d around P_cat; SF:27
    FINE_STEP_D = 2e-7               # d; SF:27
    ECLIPSE_LEVEL = 0.5              # normalised flux below = in eclipse (phase-0 centring); SF:29
    MAD_WINDOW_PHASE = 0.02          # +- phase window of the local median; SF:34
    MAD_MIN_NEIGHBOURS = 5           # SF:35
    MAD_SIGMA = 5.0                  # x 1.4826 MAD exclusion; SF:36-37 (ledger 1336)
    PROFILE_BINS = 40                # SF:39
    PROTECTED_KEYS = ('PDM_BINS', 'COARSE_STEP_D', 'COARSE_MIN_D', 'FINE_HALF_D', 'FINE_STEP_D', 'ECLIPSE_LEVEL',
                      'MAD_WINDOW_PHASE', 'MAD_MIN_NEIGHBOURS', 'MAD_SIGMA', 'PROFILE_BINS')
