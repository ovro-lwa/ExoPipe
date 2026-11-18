"""Configuration of `exopipe3 mosaic` - every tunable of the command, one section per stage or tier.

Units folded here: mosaic. A key in its section's PROTECTED_KEYS is locked
(a mirror of in-place code, a pinned input or a ruling-fixed value: it changes only by a commit); RULED names the
ledger ID of the ruling behind a key. Per run: --set SECTION.KEY=VALUE or --config FILE; `exopipe3 config --step mosaic` lists the sections.
"""
from exopipe.phase3.config import _Section, instrument


# =====================================================================================================================
# unit mosaic (was exopipe/phase3/_cfgdraft_mosaic.py)
# Configuration of `exopipe3 mosaic`. Source: plan/specs/mosaic.md section 3, with one change the computed task
# made: the LST list is no longer a fixed range - it is the LST directories of the input master generation that exist
# (mosaic.LSTS = None; deep chat 927a836: 'new mosaic step takes its LST list from the masters that exist').
#
# Every value names its unit, its ruling (a fix_id of clv15/fix_ledger.csv; checked by tests/test_mosaic.py) and the production
# file:line it was taken from. Production code: exopipe main 927a836 (CJ = exopipe/phase3/lib/mosaic/coadd_jysr.py, same lines
# as the production mosaic_coadd_jysr.py; BP = lib/mosaic/band_products.py = make_band_products.py; VM =
# lib/mosaic/verify_mosaic.py; CP = lib/mosaic/conformance_pass.py = mosaic_conformance_pass.py), exopipe_ops HEAD (STG =
# mosaic_stage_from_lustre.py, ANC = mosaic_v2_flux_anchor_measure.py, run_mosaic_v2*.sh, run_mosaic_jobs.sh).
#
# Protection (the config machinery honours both metas):
#   RULED           a key a ruling of Gregg's set: override only with --ruling-for <section.KEY>=<ledger ID> or into a
#                   non-production --out.
#   PROTECTED_KEYS  (a) pinned inputs; (b) values the IN-PLACE code reads from its OWN literals during the window (the engine's
#                   arithmetic literals, the ops scripts' ROOT / VARMAP / name patterns, the renderer's CAR extent, the anchor
#                   script's photometry). An override could not reach them, so config refuses it; steps/mosaic.py also refuses
#                   to start when a value here differs from the in-place literal (gate G0). Keys the step passes as arguments
#                   or sets on the imported module (NSIDE excepted, see below) are ordinary tunables.
# =====================================================================================================================

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
    OMEGA_PER_FWHM2 = 1.1331         # Omega = 1.1331 x FWHM^2 for the render's scalar beam; BP:40
    PROTECTED_KEYS = ('CAR_DEC_MIN_DEG', 'CAR_DEC_MAX_DEG', 'OMEGA_PER_FWHM2')
    # ---- the render's southern edge (batch row B-46) ----
    RENDER_DEC_CUT_DEG = -30.0
    #   deg; a render shows declination >= this (Gregg 2026-10-16 "Use declination >= -30"; the July renders cut at -38.7, a
    #   convention recorded nowhere): band_products blanks every rendered pixel south of it, in both projections, and stamps
    #   DECMIN on the render. The HEALPix maps keep mosaic_coadd.DEC_FLOOR_DEG. None = no cut. Production's renders cut at
    #   -30 deg too since 2026-10-22 (exopipe fa05cf3).
    RULED = {'RENDER_DEC_CUT_DEG': 'R-GREGG-20261016-MOSAIC-DEC-MIN-30'}
    NULLABLE_KEYS = ('RENDER_DEC_CUT_DEG',)   # None = no cut


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


class mosaic_v5(_Section):
    """`exopipe3 mosaic --generation v5` - the hybrid mosaics (spec plan/specs/hybrid.md 2.8, 2.10-2.14; mosaic/Run_HYBRID): the
    24 UNFILLED hybrid masters of a scope and weighting, staged under their Taper names (NL stage_mosaic 1704, kind 'hybrid'), are
    co-added by the v2 engine (lib/mosaic/coadd_jysr, tag _lsfill) in Jy/sr and Jy/beam, then zero-spacing filled (MF fill_file,
    fill_jybeam_file) and rendered from the filled maps; the high-band scopes also get a caveat map. Unit of work: (stage, scope,
    weighting). Tiles come from the hybrid step's work run (--hybrid-out, its tilefill-source 'work' records) or, for a unit not
    rebuilt, from the pinned unfilled sources the installed filled masters' TPSRCMD5 names (pinned). Each tile's weighting map is the
    archived full PB of its weighting in hybrid.ARCHIVE_MASTERS (NL 1704-1709: its PB = 0.1 contour is the master's blanking edge; the
    hybrid archive's copies are byte-identical, PR-5b). --out is required (a work run of its own; the maps are 200 MB each)."""
    WEIGHTINGS = ('R075', 'R0')
    VARIANT_OF = {'R075': 'Taper_R075', 'R0': 'Taper_R0'}
    #   NL 1704: the hybrid masters are staged under their own Taper names (I_deep_Taper_<v>_pbcorr_<scope>.fits), so the mosaics
    #   carry the masters' identity
    SCOPES = mosaic.SCOPES
    #   the 19 scopes of Run_HYBRID (V5 tables/summary.csv: 38 Jy/sr + 38 Jy/beam rows)
    TAG = '_lsfill'
    #   release1_hybrid_mosaics_20261109.sh:72 (mosaic_coadd_jysr --tag _lsfill); the unfilled co-add carries it too: lsfill proves
    #   the zero spacings only with card TPFILL (PR-1 token-card rule)
    REQUIRE_N_LSTS = 24
    #   HG14 (L3308, Gregg: "a Stokes I Hybrid for all subbands and widebands for all LSTs for both R0 and R-0.75"): every LST a hybrid
    #   tile; hybrid_unit.RULED_TAPER_ONLY is empty, so a missing tile refuses the unit (no Taper-only stand-in)
    RENDER_SCOPES = mosaic.SCOPES
    #   V5 holds Mollweide and RA/Dec renders for every scope (19 x 2 weightings x 2 domains x 2 projections)
    SUFFIX_PROC = ('lsrecon', 'lsfill')
    #   MC SUFFIX_PROC v5 ({'_lsfill': ['lsrecon', 'lsfill']}): the name tokens conform adds to mosaic_conform.BASE_PROC; the
    #   archived Run_HYBRID names carry them (lsfill = the large scales include the zero spacings, L3427)
    PROV_FILE = '{archive_hybrid}/{lst}/{scope}/{name}'
    #   a PROVENANCE row's FILE: the archived hybrid master of the tile's name (MC 242 with MOSAIC_V5_PROV_FILE
    #   '.../Coadd/Run_NOTAPER_HYBRID_20261028/{lst}/{band}/{real}'): the mosaic is built from the UNFILLED tiles, whose filled
    #   namesakes the archive holds since 2026-11-11 (L3427); {archive_hybrid} = hybrid.ARCHIVE_HYBRID
    PROVBAS = ('{archive_hybrid} hybrid NoTaper masters (Taper + reconstructed large scales) before the zero-spacing fill, '
               'beam-corrected; symlink stage, exopipe3 mosaic --generation v5; weighting map = the full PB')
    #   MC 127-128 restated for v2 (production's names its staging tool, notaper_largescale stage-mosaic); {archive_hybrid} =
    #   hybrid.ARCHIVE_HYBRID relative to paths.PHASE3_ARCHIVE_ROOT
    PROVBAS_CAVEAT = '; 64-82 MHz / Blue / Full: level and small-scale tests recorded caveats - see the caveat map (Gregg 2026-10-31)'
    #   MC 133-134 for hybrid.CAVEAT_SCOPES, less 'Taper-only units (refused NoTaper stack) = plain Taper master': HG14 admits none
    BEAMNOTE = 'scalar beam = median constituent master header; per-pixel Omega_eff in the psfmap product'
    #   = mosaic_conform.BEAMNOTE (MC 257 takes the beams from the master headers; production's v5 text names a repair ledger)
    RULED = {'REQUIRE_N_LSTS': 'R-GREGG-20261108-MIN-HYBRID-EVERY-CELL-BOTH-ROBUSTS',
             'TAG': 'R-GREGG-20261111-KEEP-LSFILL-NAME-FOR-ZERO-SPACING-MOSAICS',
             'SUFFIX_PROC': 'R-GREGG-20261111-KEEP-LSFILL-NAME-FOR-ZERO-SPACING-MOSAICS'}
    PROTECTED_KEYS = ('VARIANT_OF', 'TAG', 'PROV_FILE')


class mosaic_caveat(_Section):
    """Stage `caveat` of mosaic v5 (NL caveat_map 1759, test_caveat_map 1835; Gregg 2026-10-31 "ok on caveats"): per pixel the
    weighted mean of the hybrid tiles' large-scale level / survey prior (LEVEL) and the weight fraction of Taper-only tiles
    (TAPERONLY, 0 while HG14 admits none), with the mosaic's own tile weights. Scopes: hybrid.CAVEAT_SCOPES (D4: one list).
    Declination bands: hybrid_recon.DEC_BANDS."""
    NSIDE = 128
    #   NL 1759 (nside=128)
    PBEFF_MIN = 0.2
    PBEFF_MED_MAX = 0.01
    PBEFF_P99_MAX = 0.05
    PBEFF_MIN_PIXELS = 1000
    #   NL 1852-1857 (test_caveat_map (4)): the caveat weights' effective PB against the mosaic's PB_eff (NSIDE 2048 averaged to
    #   NSIDE 128), where both exceed 0.2: median |difference| < 0.01, 99th percentile < 0.05, over more than 1000 pixels
    RULED = {'PBEFF_MED_MAX': 'R-GREGG-20261031-HIGH-BAND-CAVEATS-APPROVED-GO-ON-EVERYTHING',
             'PBEFF_P99_MAX': 'R-GREGG-20261031-HIGH-BAND-CAVEATS-APPROVED-GO-ON-EVERYTHING'}
    PROTECTED_KEYS = ('NSIDE',)


class mosaic_fill(_Section):
    """Stage `fill` of mosaic v5 (MF fill_file 106, test_arrays 85, fill_jybeam_file 141): the zero-spacing fill of the unfilled
    Jy/sr co-add from the LWA1 LFSS (the donor, l-blend and harmonic limits are hybrid_tilefill's keys: one copy for the mosaic and
    the per-tile fill) and its Jy/beam companion through the map's own conversion. Content tests on the written files."""
    TEST_NSIDE = 256
    LS_SMOOTH_DEG = 30.0
    HP_SMOOTH_DEG = 5.0
    INNER_APOD_MIN = 0.9
    LS_MAX = 0.02
    HP_MAX = 0.03
    HP_ABS_FRAC = 0.5
    #   MF 28-34, 85-103 (test_tpfill): at 30-deg FWHM inside the footprint interior (apodization weight > 0.9, NSIDE 256) median
    #   |filled - s LFSS| / median(s LFSS) <= 0.02; the change d = filled - unfilled is smooth: rms(d - smooth_5deg(d)) / rms(d)
    #   <= 0.03 and max |d - smooth_5deg(d)| <= 0.5 x the unfilled map's robust rms. Limits fixed on the July filled maps before
    #   any hybrid map was filled
    JB_SMOOTH_DEG = 5.0
    JB_BRIGHT_PERCENTILE = 70.0
    JB_PBEFF_MIN = 0.2
    JB_MED_TOL = 0.01
    JB_SPREAD = 0.06
    #   MF 137-197: the Jy/beam conversion K = Omega_eff x PB_eff x r, r = jybeam / (jysr x Omega_eff x PB_eff) at the bright pixels
    #   (|jysr| above its 70th percentile, PB_eff > 0.2) smoothed to 5 deg; test: identity to float32 precision, median
    #   jybeam / (jysr x K) within 0.01 of 1, 5-95 percentile spread <= 0.06
    RULED = {'LS_MAX': 'R-GREGG-20261110-TOTAL-POWER-FILL-HYBRID-MOSAICS-R075-THEN-R0',
             'HP_MAX': 'R-GREGG-20261110-TOTAL-POWER-FILL-HYBRID-MOSAICS-R075-THEN-R0',
             'HP_ABS_FRAC': 'R-GREGG-20261110-TOTAL-POWER-FILL-HYBRID-MOSAICS-R075-THEN-R0',
             'JB_MED_TOL': 'R-GREGG-20261111-KEEP-LSFILL-NAME-FOR-ZERO-SPACING-MOSAICS',
             'JB_SPREAD': 'R-GREGG-20261111-KEEP-LSFILL-NAME-FOR-ZERO-SPACING-MOSAICS'}
    PROTECTED_KEYS = ('TEST_NSIDE', 'LS_SMOOTH_DEG', 'HP_SMOOTH_DEG', 'INNER_APOD_MIN', 'JB_SMOOTH_DEG', 'JB_BRIGHT_PERCENTILE',
                      'JB_PBEFF_MIN')


class mosaic_content(_Section):
    """Stage `test` of mosaic v5 (MT = production mosaic_content_test.py: TEST 1 run 108, TEST 2 provenance 95; lib/mosaic/content.py):
    the unfilled co-add, in Jy/sr and Jy/beam, against the same apertures recomputed from its staged tiles with the co-add's weights
    (production's V5 tables/summary.csv rows are the unfilled maps' values)."""
    PARENTS = '/lustre/gh/main/phase3/deep_parent_catalogs_union_20261118T0130'
    #   MT 201 (--parents default): master_R075_<LST>_<scope>.gaul.csv, positions and peaks of the brightest sources only.
    #   CHANGE FROM PRODUCTION (2026-11-18, spec section 7 B15), with hybrid_recon.CATALOGS: one rebuilt chain is tested against one
    #   parent union, the one rebuilt on the current Masters. Production's MT reads deep_parent_catalogs_union_20261014T0203, stale on
    #   47 of 456 cells after the Release 1 master rebuilds (RESULT-20261118-DEEP-PARENT-UNION-REFRESH). A parity run against an
    #   archived content test sets this back to the 10-14 union (the PR-7b and PR-6c parities ran with it).
    N_SOURCES = 200
    DEC_MIN = -30.0
    ISOLATION_BEAMS = 5.0
    APERTURE_BEAMS = 2.0
    #   MT 24 (DEC_MIN), 27-48 (isolated by 5 beams, Dec >= DEC_MIN + 3 beams), 114 (R = 2 beams), 202 (--n 200)
    MIN_SOURCES = 20
    MED_TOL = 0.01
    FRAC5_MIN = 0.90
    #   MT 137: PASS when >= 20 sources, |median ratio - 1| <= 0.01 and >= 90 % of the sources within 5 % (MT 8-13). The content
    #   tests belong to the run plan Gregg approved 2026-10-22 (R-GREGG-20261022-MOSAIC-RUN-GO-52-WIDEBAND-AS-IS-LWACALIM09); the
    #   limits are production's literals (exopipe 93a0d02), not stated in a ruling: protected, not RULED
    #   TEST 2 (MT 103 hashed each tile's first 64 MB and compared it with nothing): the unit checks that TEST 1 read exactly its
    #   tile list's files, each at the full md5 it verified (lib/mosaic/content.py); no size to configure
    RULED = {'DEC_MIN': 'R-GREGG-20261016-MOSAIC-DEC-MIN-30'}
    #   Gregg 2026-10-16: "Use declination >= -30" (the products' southern limit)
    PROTECTED_KEYS = ('PARENTS', 'N_SOURCES', 'DEC_MIN', 'ISOLATION_BEAMS', 'APERTURE_BEAMS', 'MIN_SOURCES', 'MED_TOL', 'FRAC5_MIN')


class mosaic_pkcorr(_Section):
    """Stage `pkcorr` of mosaic v5 (MP = production mosaic_pkcorr.py; lib/mosaic/pkcorr.py): the injection-calibrated peak-recovery
    correction map of the Jy/sr mosaic (the point-source set: S_true = peak x Omega_eff / PKCORR). Each round adds S_INJ restoring-beam
    Gaussians to copies of the staged tiles (PB > INJ_PB_MIN) at Dec DEC_START..DEC_STOP every DEC_STEP, RA i x RA_STEP + round x
    ROUND_OFF, co-adds them with the real engine and differences against the unfilled co-add (production's --clean-map: the chain ran
    PKCORR before the fill, run_v5_pkcorr.sh and lostsky_hybrids_green_full_20261113.sh step 3)."""
    S_INJ = 50.0
    DEC_START = -30.0
    DEC_STOP = 89.0
    DEC_STEP = 2.0
    RA_STEP = 47.0
    ROUND_OFF = 3.0
    INJ_PB_MIN = 0.12
    PEAK_RADIUS_FWHM = 0.5
    PEAK_RADIUS_MIN_DEG = 0.05
    OMEGA_RADIUS_DEG = 0.2
    #   MP 32-35 (S_INJ, DECS = arange(-30, 89, 2), RA_BASE i x 47 mod 360, ROUND_OFF), 78 (PB > 0.12), 89-93 (Omega_eff = median PSF
    #   map within 0.2 deg; peak = max within max(0.5 FWHM_eff, 0.05 deg))
    ROUNDS = 5
    #   run_v5_pkcorr.sh (--rounds 5; MP default 5): 5 rounds sample the 15-deg LST tile period
    MIN_ROUNDS = 3
    REC_MIN = 0.5
    REC_MAX = 1.2
    MAP_TOL = 1e-6
    #   MP 148-158 (test_pkcorr): every node holds samples from >= min(3, rounds) rounds, every node's median recovery in 0.5-1.2,
    #   the map read back equals the curve within 1e-6. The point-source set is an archive deliverable by
    #   R-GREGG-20261023-MOSAIC-ALL-PRODUCTS-ON-LUSTRE-NOTAPER-ABOVE-TAPER-JYSR-AND-POINT-SOURCE-SET, and it is the 2026-07-27 PKCORR
    #   program (CORRECTION-POINT-SOURCE-SET-IS-THE-20260727-PROGRAM-PKCORR-NOT-ARCHIVED); the method's numbers are production's
    #   literals (exopipe 26ff18f, "method unchanged"), not stated in a ruling: protected; the ruling's gate is HG15 at conform
    RULED = {'DEC_START': 'R-GREGG-20261016-MOSAIC-DEC-MIN-30'}
    #   Gregg 2026-10-16: "Use declination >= -30" (MP 10: "-30 (the product limit)")
    PROTECTED_KEYS = ('S_INJ', 'DEC_START', 'DEC_STOP', 'DEC_STEP', 'RA_STEP', 'ROUND_OFF', 'INJ_PB_MIN', 'PEAK_RADIUS_FWHM',
                      'PEAK_RADIUS_MIN_DEG', 'OMEGA_RADIUS_DEG', 'ROUNDS', 'MIN_ROUNDS', 'REC_MIN', 'REC_MAX', 'MAP_TOL')
