"""Config draft of `exopipe3 mosaic` (refactor window; config.py auto-loads _cfgdraft_*.py until the refactor chat folds these
sections into config.py at '# @@STEP-SECTIONS@@'). Source: plan/specs/mosaic.md section 3, with one change the computed task
made: the LST list is no longer a fixed range - it is the LST directories of the input master generation that exist
(mosaic.LSTS = None; deep chat 927a836: 'new mosaic step takes its LST list from the masters that exist').

Every value names its unit, its ruling (a fix_id of clv15/fix_ledger.csv; checked by tests/test_mosaic.py) and the production
file:line it was taken from. Production code: exopipe main 927a836 (CJ = exopipe/phase3/lib/mosaic/coadd_jysr.py, same lines
as the production mosaic_coadd_jysr.py; BP = lib/mosaic/band_products.py = make_band_products.py; VM =
lib/mosaic/verify_mosaic.py; CP = lib/mosaic/conformance_pass.py = mosaic_conformance_pass.py), exopipe_ops HEAD (STG =
mosaic_stage_from_lustre.py, ANC = mosaic_v2_flux_anchor_measure.py, run_mosaic_v2*.sh, run_mosaic_jobs.sh).

Protection (config.py honours both metas):
  RULED           a key a ruling of Gregg's set: override only with --ruling-for <section.KEY>=<ledger ID> or into a
                  non-production --out.
  PROTECTED_KEYS  (a) pinned inputs; (b) values the IN-PLACE code reads from its OWN literals during the window (the engine's
                  arithmetic literals, the ops scripts' ROOT / VARMAP / name patterns, the renderer's CAR extent, the anchor
                  script's photometry). An override could not reach them, so config refuses it; steps/mosaic.py also refuses
                  to start when a value here differs from the in-place literal (gate G0). Keys the step passes as arguments
                  or sets on the imported module (NSIDE excepted, see below) are ordinary tunables.
"""
from exopipe.phase3.config import _Section, instrument


class mosaic(_Section):
    """`exopipe3 mosaic` - all-sky HEALPix mosaics of the deep masters (PHASE3_ARCHITECTURE.md 7bis; mosaic v2, Gregg
    2026-09-26 'Case A'). Selection, inputs, refusals, lanes. Stage tunables: mosaic_coadd, mosaic_render, mosaic_qa,
    mosaic_conform. Intermittent (DESIGN 12.2): never part of the per-hour chain."""
    # --- inputs ------------------------------------------------------------------------------------------------------
    INPUT_COADD_RUN = '/lustre/pipeline/exopipe/phase3/Coadd/Run_20260909_000000'
    #   path; the master generation every tile is read from (PROTECTED: pinned input; the in-place stage script hard-codes
    #   the same ROOT, STG:23). Gregg 2026-09-26 'The correct Masters are on Lustre' (CASE-A-...); masters beside mosaics
    #   (R-MASTERS-BESIDE-MOSAICS).
    MASTER_PATTERNS = {'I': ('{scope}_I_deep_{var}_shflux_pbcorr_dewarped_aligned_coadd_LST{lst}_*.fits',
                             '{scope}_I_deep_{var}_shflux_mdewarp_pbcorr_dewarped_aligned_coadd_LST{lst}_*.fits')}
    #   glob patterns of the current pbcorr master under <INPUT_COADD_RUN>/<lst>/<scope>/ (both name forms: 664 of the 1,824
    #   masters carry _mdewarp_); exactly one match per (lst, scope, variant) or refuse (RUN-SUBBAND-MOSAICS-V2-LAUNCHED).
    #   STG:44-48 (PROTECTED: the stage script globs these literally).
    VARIANT_TOKENS = {'Taper_R075': 'Taper_Robust-0.75', 'Taper_R0': 'Taper_Robust+0.0',
                      'NoTaper_R075': 'NoTaper_Robust-0.75', 'NoTaper_R0': 'NoTaper_Robust+0.0'}
    #   short variant -> archive-name token; STG:27-28 VARMAP (PROTECTED).
    LSTS = None
    #   None = every LST directory (NNh) of INPUT_COADD_RUN, i.e. the masters that exist (deep chat 927a836; STG:31 lists the
    #   same directories, and every v2 launcher passes what was staged: run_mosaic_v2.sh:12, run_mosaic_v2_sub.sh:13,
    #   run_mosaic_jobs.sh:10). A tuple of LSTs restricts it (as --lsts does). The engine CLI default range(23) is never used.
    REQUIRE_N_LSTS = 24
    #   tiles per job; fewer refuses unless `--waive mosaic.inputs_complete --ruling "..."` (all 24 tiles: CASE-A-...;
    #   run_mosaic_v2_sub.sh:14 and run_mosaic_jobs.sh:11 refuse below 24).
    # --- selection ---------------------------------------------------------------------------------------------------
    VARIANTS = ('Taper_R075', 'Taper_R0', 'NoTaper_R075', 'NoTaper_R0')
    #   imaging variants built; four, no hierarchy (Gregg 2026-08-10, architecture 7bis); run_mosaic_v2.sh:15.
    WIDEBANDS = ('Red', 'Green', 'Blue', 'Full')                 # run_mosaic_v2.sh:16
    SCOPES = WIDEBANDS + tuple(instrument.ALL_SUBBANDS)
    #   19 scopes; the 15 sub-band mosaics are required (R-SUBBAND-MOSAICS-REQUIRED-AND-STRIATION-MASTER-TEST);
    #   run_mosaic_v2_sub.sh:12.
    PRODUCTS = ('I',)
    #   Stokes I only in v2 (CP:78; PUSHED-MOSAIC-V2-TAPER-WIDEBAND). Any other product refuses (no production input pattern).
    DOMAINS = ('jysr', 'jybeam')
    #   jysr = canonical Jy/sr deliverable + Omega_eff psfmap (Gregg 2026-07-11); jybeam = apparent Jy/beam + PB_eff map
    #   ('jybeam approved', R-JYBEAM-TOKEN-AND-MOSAIC-V2-PUSH-TAPER-ONLY); run_mosaic_v2.sh:16.
    JOB_ORDER_VARIANTS = ('NoTaper_R075', 'Taper_R075', 'NoTaper_R0', 'Taper_R0')
    #   scheduling order only (--list-jobs, lanes): 'the variant Gregg reads is ready first' (run_mosaic_v2_sub.sh:8, :22).
    # --- weighting PB (the masters are already PB-corrected; the PB is a weight and feather only, never divided) ------
    WEIGHT_PB_SOURCE = 'derive'
    #   'derive' = pb_apply.wideband_pb per tile from the Taper master of the variant's robust, written by the in-place stage
    #   script (STG:61-89; PB step 2, R-FLUX-SCALE-REMEASURE-REBAKE-AND-PB-STEP2-RUNNING). The only value implemented: using
    #   the cells' archived PB_deep[_R075] maps waits for the parity check P-PB (spec 10) to show they are bit-identical.
    WEIGHT_PB_REF_VARIANT = {'R075': 'Taper_R075', 'R0': 'Taper_R0'}
    #   master whose constituent nights and beam set the weighting PB of each robust; STG:63-64 (PROTECTED).
    # --- refusals and fallbacks (DESIGN 11.6) ------------------------------------------------------------------------
    MISSING_TILE_ACTION = 'refuse'
    #   a planned tile without its master or weighting PB: 'refuse' (run_mosaic_v2_sub.sh:14-19). The engine alone would
    #   print MISSING and continue (CJ:195-197); that behaviour is not offered (batch row B-45 pending; no tile is missing
    #   today, spec M1).
    TILE_GUARD_ACTION = 'skip-and-record'
    #   a tile with < mosaic_coadd.MIN_VALID_TILE_PIXELS valid pixels or CORRUPT-FLAT: skipped by the engine (CJ:211-221,
    #   guard 2026-07-18) and recorded in the coadd record and NSTACK; 'refuse' = refuse the job instead (B-45 option a).
    # --- layout and where it runs ------------------------------------------------------------------------------------
    BUILD_ROOT = '/fast/gh/main/phase3/mosaic_v2'
    #   path; the default --out = today's wideband build tree on lwacalim07 (run_mosaic_v2.sh:11; sub-bands were built in
    #   /fast/gh/main/phase3/mosaic_v2_sub on lwacalim09, run_mosaic_v2_sub.sh:11). Under paths.PRODUCTION_ROOTS: refused
    #   while paths.PRODUCTION_WRITES is False.
    LAYOUT = {'healpix': 'healpix', 'renders': 'renders', 'qa': 'qa', 'staging': 'archive_staging/mosaic', 'logs': 'logs'}
    #   sub-directories of --out = today's mosaic_v2 tree (run_mosaic_v2.sh:14, :24; CP:46-49). No renames in the window.
    MAX_JOBS_PER_NODE = 20
    #   concurrent co-add / census / stage processes (lanes) on one node (run_mosaic_v2_sub.sh:24: 20 lanes on lwacalim09;
    #   about 4 GB each). --workers N overrides; the lanes never exceed the cores this process may use / CORES_PER_JOB.
    CORES_PER_JOB = 2
    #   cores pinned per lane (taskset -c 2L,2L+1: run_mosaic_v2.sh:22, run_mosaic_v2_sub.sh:29).
    RENDER_MAX_PARALLEL = 4
    #   concurrent render processes ('four variants at a time, one core each', ~72-million-pixel Mollweide grid per process:
    #   run_mosaic_v2_renders.sh:5, :10-13).
    RULED = {'REQUIRE_N_LSTS': 'CASE-A-CONFIRMED-NOTAPER-R0-GAP-IS-CLOSED-AT-SOURCE-MOSAIC-V2-BUILDING',
             'SCOPES': 'R-SUBBAND-MOSAICS-REQUIRED-AND-STRIATION-MASTER-TEST',
             'DOMAINS': 'R-JYBEAM-TOKEN-AND-MOSAIC-V2-PUSH-TAPER-ONLY',
             'WEIGHT_PB_SOURCE': 'R-FLUX-SCALE-REMEASURE-REBAKE-AND-PB-STEP2-RUNNING'}
    PROTECTED_KEYS = ('INPUT_COADD_RUN', 'MASTER_PATTERNS', 'VARIANT_TOKENS', 'WEIGHT_PB_REF_VARIANT')


class mosaic_coadd(_Section):
    """Co-add stage: lib/mosaic/coadd_jysr.py (the production mosaic_coadd_jysr.py, moved by pure git mv), called in place.
    The keys NOT in PROTECTED_KEYS are module globals the step sets on the imported module before each call."""
    NSIDE = 2048
    #   HEALPix NSIDE (50,331,648 pixels); M-03/M-04 (architecture 7bis, DOC); CJ:34. PROTECTED although the engine reads a
    #   global: the in-place renderer, anchor script and VLSSr check hard-code 2048 (BP:41, :67; ANC:21; VM:32, :65).
    DEC_FLOOR_DEG = -40.0            # deg; pixels south of it are NaN; CJ:35 (DEC_FLOOR), applied CJ:243-244
    PB_FLOOR = 0.10                  # PB; tile weight 0 at or below; Gregg 2026-07-11 'floor PB > 0.1' (DOC); CJ:36 (PBCORR_FLOOR)
    PB_FEATHER_START = 0.15          # PB; cosine weight taper from here down to PB_FLOOR; Gregg 2026-07-17 (DOC, M-04);
    #                                  CJ:37 (PB_FEATHER), CJ:227-228
    COSZ_FLOOR = 0.05                # cos(zenith angle) floor in Omega_b = Omega_zen / cos z; CJ:38 (N_FLOOR), CJ:119
    COMMON_GRID_N_PIX = 4200         # px per side of the intermediate per-tile SIN grid; CJ:43 (GRID_N)
    COMMON_GRID_CDELT_DEG = 0.0283   # deg per px of that grid; CJ:43 (CDELT)
    # ---- in-place literals (PROTECTED_KEYS; G0 checks each against the engine source) ----
    PB_WEIGHT_EXPONENT = 2           # weight = PB**2 / sigma**2 per tile (M-04, DOC); CJ:222
    MIN_VALID_TILE_PIXELS = 100      # apparent pixels at PB > PB_FLOOR; fewer = tile skipped; CJ:211
    CORRUPT_FLAT_SIGMA_JYBEAM = 1e-4  # Jy/beam; apparent MAD sigma below = CORRUPT-FLAT, skipped (guard 2026-07-18); CJ:219
    COMMON_GRID_CRVAL2_DEG = 37.2158  # deg; grid-centre Dec = the masters' CRVAL2 37.215849167 (spec M4; NOT the site
    #                                   latitude 37.23977727); CJ:105
    COMMON_GRID_RA_OFFSET_H = 0.5    # h; grid-centre RA = 15 x (LST + 0.5) deg; CJ:105
    REPROJECT_ORDER = 'bilinear'     # value-preserving on Jy/sr (M-03, DOC); CJ:151, :153, :160, :173, :175, :230-237
    SIGNAL_PATH = 'native-sin-healpix'
    #   signal native -> common SIN grid -> HEALPix (two bilinear steps, CJ:151-153, :172-175); weights native -> HEALPix.
    MIN_FREQ_HZ = 1e6                # Hz; RESTFRQ / RESTFREQ / CRVAL3 accepted above this; CJ:129
    # ---- content test only (no production equivalent) ----
    CONTENT_MIN_COVERAGE = dict({'Blue': 0.998, 'Full': 0.987, 'Green': 0.986, 'Red': 0.967},
                                **{sb: None for sb in instrument.ALL_SUBBANDS})
    #   per scope: fraction of the HEALPix pixels at Dec >= DEC_FLOOR_DEG that must be finite in a full
    #   (REQUIRE_N_LSTS-tile) map. A property of the scope (its PB > PB_FLOOR footprint), identical over the 4 variants and
    #   both domains: measured 2026-10-13 on all 32 wideband v2 maps of lwacalim07:/fast/gh/main/phase3/mosaic_v2/healpix -
    #   Blue 0.99914, Full 0.98881, Green 0.98768, Red 0.96859 - and set ~0.0015 below. Sub-bands None = report only until
    #   measured on the 240 sub-band maps on lwacalim09 (parity case P0). Content test only; no product depends on it.
    RULED = {'DEC_FLOOR_DEG': 'DOC', 'PB_FLOOR': 'DOC', 'PB_FEATHER_START': 'DOC', 'COSZ_FLOOR': 'DOC'}
    PROTECTED_KEYS = ('NSIDE', 'PB_WEIGHT_EXPONENT', 'MIN_VALID_TILE_PIXELS', 'CORRUPT_FLAT_SIGMA_JYBEAM',
                      'COMMON_GRID_CRVAL2_DEG', 'COMMON_GRID_RA_OFFSET_H', 'REPROJECT_ORDER', 'SIGNAL_PATH', 'MIN_FREQ_HZ')


class mosaic_render(_Section):
    """Render stage: lib/mosaic/band_products.py (the production make_band_products.py, moved by pure git mv), in place."""
    SCOPES = ('Red', 'Green', 'Blue', 'Full')
    #   scopes rendered today (run_mosaic_v2_renders.sh was run on the wideband set; sub-band maps were never rendered).
    DOMAINS = ('jysr', 'jybeam')     # run_mosaic_v2_renders.sh:9
    MOLLWEIDE_WIDTH_PX = 12000       # px; image 6,000 x 12,000, Galactic, centre l = 0 deg; run_mosaic_v2_renders.sh:11 -> BP:90
    CAR_CDELT_ARCMIN = 1.5           # arcmin per px; image 5,400 x 14,400; BP:91
    BEAM_FALLBACK_ARCMIN = {'Red': 13.1, 'Green': 8.4, 'Blue': 6.0, 'Full': 7.4}
    #   arcmin; the July table, used only when the map's PSF file is absent or empty (BP:14, :39-41); set on the module
    #   (BEAMS) before each call. The render's beam source is recorded per file (content test mosaic_render.beam_card).
    # ---- in-place literals (PROTECTED_KEYS) ----
    CAR_DEC_MIN_DEG = -45.0          # deg; car_wcs default, BP:50 (build calls car_wcs(car_cdelt), BP:75)
    CAR_DEC_MAX_DEG = 90.0           # deg; BP:50
    RENDER_DEC_CUT_DEG = None
    #   deg; None = no cut (today). -38.7 = the 2026-07-15 render convention (batch row B-46, pending); no code path cuts.
    OMEGA_PER_FWHM2 = 1.1331         # Omega = 1.1331 x FWHM^2 for the render's scalar beam; BP:40
    PROTECTED_KEYS = ('CAR_DEC_MIN_DEG', 'CAR_DEC_MAX_DEG', 'RENDER_DEC_CUT_DEG', 'OMEGA_PER_FWHM2')


class mosaic_qa(_Section):
    """QA stage, measure only: calibrator aperture flux (exopipe_ops/mosaic_v2_flux_anchor_measure.py, in place) and VLSSr
    astrometry (lib/mosaic/verify_mosaic.astrometry_check, in place)."""
    APPLY_FLUX_ANCHOR = False
    #   R-MOSAIC-V2-FLUX-SCALE-MEASURED-NOT-CORRECTED (Gregg 2026-09-30 '8. Approved'): the mosaics keep the masters'
    #   VLSSr-based scale; no code path scales pixels. True refuses (no implementation exists, gate G11).
    ANCHOR_TABLE_NAME = 'flux_anchor_measured_{yyyymmdd}.csv'   # UTC date of the run; push_mosaic_v2_20260929.py:22
    VLSSR_NSRC = 400                 # brightest VLSSr sources matched; VM:55, :60-61 (argument)
    VLSSR_MIN_FLUX_JY = 8.0          # Jy; VM:55, :58 (argument)
    VLSSR_TABLE_NAME = 'vlssr_astrometry_{yyyymmdd}.csv'   # NEW measure-only table (no production equivalent)
    VLSSR_GATE_MAX_MEDIAN_OFFSET_ARCMIN = None
    #   arcmin; None = measure only (today; C-19 pending). When set: a job whose median VLSSr offset exceeds it is refused at
    #   conform (architecture 5.4.5 wants a refusing VLSSr gate; the threshold is not ruled).
    # ---- pinned input and in-place literals (PROTECTED_KEYS) ----
    VLSSR_CATALOG = '/lustre/gh/calibration/pipeline/reference/surveys/vlssr_radecpeak_unresolved.txt'   # path; VM:17
    CALIBRATOR_SCALE = 'SH12'        # only Scaife & Heald 2012 calibrators are measured; ANC:43
    CALIBRATOR_TABLE = 'exopipe.common.pipeline_config.CALIB_DATA'   # ANC:17, :42 (two-table question A10, not ours)
    ANCHOR_APERTURE_FWHM = 2.5       # x local effective FWHM; ANC:24
    ANCHOR_ANNULUS_FWHM = (3.5, 5.0)  # x FWHM, background annulus; ANC:31
    ANCHOR_PSF_DISC_DEG = 0.2        # deg, disc of the local median Omega_eff; ANC:26
    ANCHOR_SOURCES = ('3C196', '3C380')   # anchor = geometric mean of their ratios (July convention); ANC:55
    VLSSR_DEC_RANGE_DEG = (-25.0, 85.0)   # deg; VM:59
    VLSSR_CENTROID_DISC_DEG = 0.1    # deg (6 arcmin); VM:65
    RULED = {'APPLY_FLUX_ANCHOR': 'R-MOSAIC-V2-FLUX-SCALE-MEASURED-NOT-CORRECTED'}
    PROTECTED_KEYS = ('VLSSR_CATALOG', 'CALIBRATOR_SCALE', 'CALIBRATOR_TABLE', 'ANCHOR_APERTURE_FWHM', 'ANCHOR_ANNULUS_FWHM',
                      'ANCHOR_PSF_DISC_DEG', 'ANCHOR_SOURCES', 'VLSSR_DEC_RANGE_DEG', 'VLSSR_CENTROID_DISC_DEG')


class mosaic_conform(_Section):
    """Archive-staging stage: the 'wb2' branch of the production mosaic_conformance_pass.py (CP:43-55, 70-89, 92-131,
    155-276) ported to lib/mosaic/steps_conform.py as one function per job with arguments; same calls (build_name,
    shutil.copy2, write_standard_header, append_provenance_table). The archive push itself is `exopipe3 archive`."""
    VARIANTS = ('Taper_R075', 'Taper_R0')
    #   'push Taper only' - NoTaper held until the diverged NoTaper masters are rebuilt (R-JYBEAM-TOKEN-AND-MOSAIC-V2-PUSH-
    #   TAPER-ONLY); CP:73-74; push_mosaic_v2_20260929.py:44-46.
    SCOPES = ('Blue', 'Green', 'Red', 'Full')   # CP:51 (wb2); no sub-band v2 staging mode exists
    SUBDIR = 'wideband'              # CP:50
    BASE_PROC = ('pbcorr', 'coadd', 'mosaic', 'jysr', 'feather')
    #   name tokens of the jysr map (M-09, DOC; R-JYBEAM-...); CP:77
    APPARENT_DROP_PROC = ('jysr', 'pbcorr')   # jybeam and pbmap names carry neither; PBCORR=False (R-JYBEAM-...); CP:224
    CADENCE = 'deep'                 # CP:179
    LST_TOKEN = 'allsky'             # mosaic footprint token (M-09, DOC); CP:180
    TIME_BASIS = 'mosaic-constituents'   # written TIMEFIX = 'v2-mosaic-constituents'; CP:161
    HEADER_VALIDATE = False          # write_standard_header(validate=False) as today (CP:163; decision C-08 pending)
    THERMNOI_DEC_STRIP_DEG = (20.0, 55.0)
    #   deg; THERMNOI = cleaned-V mosaic zenith-strip MAD x median Omega_eff (CP:134-148). v2 has no cleaned-V mosaic, so
    #   THERMNOI is absent, exactly as today (PUSHED-MOSAIC-V2-TAPER-WIDEBAND).
    COPY_HEADER_KEYS = ('FLUXSCL', 'FEATHER', 'NSIDE', 'JY2K', 'BUNIT')   # map HDU 1 -> staged header (jysr; CP:197)
    COPY_HEADER_KEYS_APPARENT = ('FEATHER', 'NSIDE', 'BUNIT')             # jybeam / pbmap (CP:233)
    PROVBAS = ("Lustre Run_20260909_000000 current masters (symlink stage, mosaic_stage_from_lustre.py); "
               "beams from the masters' own headers (no repair ledger); mosaic v2 2026-09-26..29")   # verbatim CP:54-55
    BEAMNOTE = 'scalar beam = median constituent master header; per-pixel Omega_eff in the psfmap product'   # CP:184-186
    RULED = {'VARIANTS': 'R-JYBEAM-TOKEN-AND-MOSAIC-V2-PUSH-TAPER-ONLY',
             'BASE_PROC': ('DOC', 'R-JYBEAM-TOKEN-AND-MOSAIC-V2-PUSH-TAPER-ONLY'),
             'APPARENT_DROP_PROC': 'R-JYBEAM-TOKEN-AND-MOSAIC-V2-PUSH-TAPER-ONLY',
             'LST_TOKEN': 'DOC'}
