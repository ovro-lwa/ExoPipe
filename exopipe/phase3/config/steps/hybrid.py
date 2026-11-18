"""Configuration of `exopipe3 hybrid` - the hybrid NoTaper route (stack, reconstruction, wideband, hybrid, Full, per-tile fill).

Units folded here: hybrid (spec plan/specs/hybrid.md, sections 3.1-3.4; ported in pull requests PR-1 to PR-5). A key in its
section's PROTECTED_KEYS is locked (a pinned input or a ruling-fixed value: it changes only by a commit); RULED names the ledger ID
of the ruling behind a key. Per run: --set SECTION.KEY=VALUE or --config FILE; `exopipe3 config --step hybrid` lists the sections.

Every value keeps its production value and names its source. Production code: exopipe main 4c7098c (NL = exopipe/phase3/
notaper_largescale.py, MF = exopipe/phase3/mosaic_tpfill.py); the per-tile fill driver exopipe_ops/hybrid_master_lsfill_20261111/
run_hybrid_master_lsfill_20261111.py (HMF). "Lnnnn" in the spec = row nnnn of clv15/fix_ledger.csv; here the fix_id is written.

Pull request PR-1 (2026-11-18) adds the sections and the step's gate skeleton only; the stages are ported by PR-2 (stack), PR-3
(recon, wideband), PR-4 (hybrid, full) and PR-5 (tilefill). The mosaic-generation sections (v3 / v5, PKCORR, content, caveat, fill)
and the striation stage of `masters` come with the pull requests that port those stages (PR-6, PR-7).
"""
from exopipe.phase3.config import _Section


class hybrid(_Section):
    """`exopipe3 hybrid` - selection, inputs and outputs of the hybrid route. Unit of work: (LST, scope, weighting), weighting
    R075 (Robust -0.75) or R0 (Robust 0); a cell is (LST, scope)."""
    SCOPES = ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz',
              '64MHz', '69MHz', '73MHz', '78MHz', '82MHz', 'Red', 'Green', 'Blue', 'Full')
    #   the 19 scopes of the route (15 subbands, 3 widebands, Full); every one of them in both weightings for every LST
    #   (R-GREGG-20261108-MIN-HYBRID-EVERY-CELL-BOTH-ROBUSTS: "I need as a minimum a Stokes I Hybrid for all subbands and widebands
    #   for all LSTs for both R0 and R-0.75")
    WEIGHTINGS = ('R075', 'R0')
    #   NL 66-68 VARIANT / TAPER_OF: R075 -> NoTaper Robust -0.75 with Taper residual stack T075; R0 -> NoTaper Robust 0 with T0
    RECON_GEN = 'lsrecon-v3-horizon-grad'
    #   NL 69: CLEAN, step and prior to the true horizon; the survey sets the constant and both gradient modes of the step
    HYB_U_IN = 30.0
    HYB_U_OUT = 60.0
    #   wavelengths, NL 71: the hybrid takes the reconstruction in full below HYB_U_IN (the Taper's inner Tukey taper,
    #   -taper-inner-tukey 30) and as a cosine to nothing at HYB_U_OUT (G of NL 1080)
    CAVEAT_SCOPES = ('64MHz', '69MHz', '73MHz', '78MHz', '82MHz', 'Blue', 'Full')
    #   NL 828: the scopes whose large-scale level test and hybrid small-scale test are recorded caveats, not refusals (cards +
    #   caveat map). ONE copy (spec D4): the mosaic stages read this key (production MC 131-132 copies it). Full: Gregg 2026-11-08
    STRIATION_SCOPES = ('59MHz', '64MHz', '69MHz', '73MHz', '78MHz', '82MHz', 'Blue')
    #   NL 1111: the scopes whose hybrid base is the striation-restored Taper master where the selection lists the unit
    STRIATION_SELECTION = '/lustre/pipeline/exopipe/phase3/mosaic/Run_FEATHER3_20241218-20251029/tables/selection_20261022.csv'
    #   NL 1109: the striation selection, 142 (LST, band, Taper variant) rows (pinned input: the hybrid stage refuses when its
    #   md5 is not STRIATION_SELECTION_MD5)
    STRIATION_SELECTION_MD5 = 'be3db1a94ccd0686ead4939b181043f8'
    #   md5 of STRIATION_SELECTION as archived, read 2026-11-18 (file mtime 2026-11-13 16:39 UTC, 10,828 bytes); PR-4 checks it
    #   against the selection the archived hybrids were built from
    STRIATION_RESTORED = '/lustre/gh/main/phase3/lsrecon/striation_restored_20261022'
    #   NL 1110: the striation-restored masters (with MANIFEST.csv), read by the hybrid stage
    WIDEBAND_MEMBERS = {'Red': ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz'),
                        'Green': ('41MHz', '46MHz', '50MHz', '55MHz', '59MHz'),
                        'Blue': ('64MHz', '69MHz', '73MHz', '78MHz', '82MHz')}
    #   NL 944-947: the members of each wideband; Full = all 15 subbands (made by the full stage, never by the wideband stage)
    ARCHIVE_MASTERS = '/lustre/pipeline/exopipe/phase3/Coadd/Run_20260909_000000'
    #   NL 949: the installed Taper masters every unit is built on (read only)
    ARCHIVE_HYBRID = '/lustre/pipeline/exopipe/phase3/Coadd/Run_NOTAPER_HYBRID_20261028'
    #   NL 1514: the archived hybrid masters, CLEAN residuals, reconstructions and PB maps (written only by `archive push`)
    WORK_ROOT = '/lustre/gh/main/phase3/lsrecon'
    #   the parent of the reconstruction work runs (production: --run-dir, no default, NL 44); a v2 work run is a new
    #   directory under it, never one of LSRECON_RUNS
    LSRECON_RUNS = {'R1': '/lustre/gh/main/phase3/lsrecon/Run_20261028_234240',
                    'R2': '/lustre/gh/main/phase3/lsrecon/Run_20261031_172532',
                    'RF': '/lustre/gh/main/phase3/lsrecon/Run_FULL_20261108',
                    'RS': '/lustre/gh/main/phase3/lsrecon/Run_20261026_190545',
                    'RT': '/lustre/gh/main/phase3/lsrecon/Run_20261028_043135'}
    #   the production work runs the archived Release 1 hybrids were made from (spec "Paths and terms"): R1 18-59 MHz, Red, Green;
    #   R2 64-82 MHz, Blue; RF Full; RS the NoTaper stacks and double-size PSFs; RT the Taper residual stacks. Read only: the
    #   parity references of PR-2 to PR-5; their md5 manifests come with the pull request that reads each run
    FFT_WORKERS = 12
    #   scipy.fft threads of the route's FFTs (NL 80: env FFT_WORKERS, default 12); the results do not depend on it
    RULED = {'SCOPES': 'R-GREGG-20261108-MIN-HYBRID-EVERY-CELL-BOTH-ROBUSTS',
             'WEIGHTINGS': 'R-GREGG-20261108-MIN-HYBRID-EVERY-CELL-BOTH-ROBUSTS',
             'RECON_GEN': 'R-GREGG-20261028-HYBRID-NOTAPER-RERUN-APPROVED-CLEAN-TO-TRUE-HORIZON-HYBRID-RESIDUAL',
             'CAVEAT_SCOPES': 'R-GREGG-20261031-HIGH-BAND-CAVEATS-APPROVED-GO-ON-EVERYTHING',
             'STRIATION_SCOPES': ('R-GREGG-20261031-HIGH-BAND-HYBRID-PLAN-AND-STRIATION-RESTORED-BASE-APPROVED',
                                  'R-GREGG-20261031-REBUILD-REPUSH-59MHZ-R075-ON-STRIATION-RESTORED-BASE'),
             'WIDEBAND_MEMBERS': 'R-GREGG-20261031-HIGH-BAND-CAVEATS-APPROVED-GO-ON-EVERYTHING'}
    PROTECTED_KEYS = ('STRIATION_SELECTION', 'STRIATION_SELECTION_MD5', 'STRIATION_RESTORED', 'ARCHIVE_MASTERS', 'ARCHIVE_HYBRID',
                      'LSRECON_RUNS')


class hybrid_stack(_Section):
    """Stage `stack` (spec 2.1; NL actions stack / fleet-stack): the NoTaper dirty and PSF stacks and the Taper residual stacks of a
    unit, rebuilt from the Phase 1 images of the master's constituent nights, with warp screens, the rebuild test and admission."""
    COADD_SB = '/data07/gh/phase3/coadd_sb'
    #   NL 62 (env P3_COADD_SB_ROOT): the per-cell co-add build directories (weights.csv, flows). NODE-LOCAL (lwacalim07 today):
    #   the stage refuses when the path is absent on the running node (gate HG0); no host name in code
    MDEWARP = '/lustre/pipeline/exopipe/phase3/mdewarp_screens/Run_20260909_000000'
    #   NL 63: the master dewarp screens of the installed masters
    PHASE1_ROOT = '/lustre/pipeline/exopipe/phase1'
    #   NL 182-187 (_p1file): {PHASE1_ROOT}/{lst}/{date}/Run_*/{band}/I/deep/{band}-I-Deep-{variant}-{kind}-*.fits, not .pbcorr;
    #   with more than one run, the newest Phase 1 run older than the night's Phase 2 science run
    WARP_MEDIAN_PX = 0.5
    WARP_P90_PX = 1.5
    WARP_MIN_SOURCES = 5
    #   NL 289, 350, 414: a derived warp screen (or a dewarped dirty image) is accepted when the bright-source offsets have a
    #   median <= 0.5 px and a 90th percentile <= 1.5 px over >= 5 sources; else the night cannot be put on the master's frame
    MAX_UNDERIVABLE = 1
    #   NL 418-420: at most ONE night per stack without a derivable screen or failing the alignment test is left out (recorded);
    #   more refuses the stack (gate HG4)
    MIN_NIGHTS_DROP = 2
    #   NL 424: the stack refuses when fewer than max(2, N - MIN_NIGHTS_DROP) nights with Phase 1 dirty images remain (N = the
    #   master's constituent nights; a night whose Phase 1 dirty image or PSF was never written is left out, recorded)
    REBUILD_MAX_ABS_JY = 0.01
    REBUILD_MAX_ABS_PEAK_FRAC = 1e-5
    REBUILD_RMS_JY = 1e-4
    REBUILD_RMS_FRAC = 1e-5
    #   NL 474 (test_stack_rebuild, exact limits): the rebuilt image stack equals the installed master when max |difference| <
    #   max(0.01 Jy/beam, 1e-5 x the master's peak), rms < max(1e-4 Jy/beam, 1e-5 x the master's rms) and the NaN masks agree
    FLOW_RECOVERED_AGE_S = 3600
    FLOW_RECOVERED_RATIO_TOL = 0.002
    FLOW_RECOVERED_MEDIAN_FRAC = 2e-3
    FLOW_RECOVERED_MAX_PEAK_FRAC = 0.10
    #   NL 477-482: a cell whose flows were recomputed after the build (every flow file more than FLOW_RECOVERED_AGE_S newer
    #   than weights.csv) passes when the global ratio is within 0.2% of 1, the median |difference| < 2e-3 of the master's rms,
    #   the max after the ratio < 10% of the master's peak, and the NaN masks agree
    ADMISSION = 'hybrid-scales'
    #   NL 391-393, 483-492 and recon_variant NL 887: a high-band (CAVEAT_SCOPES) NoTaper stack of a flow-recovered cell that
    #   misses the limits above is admitted when the rms of G (x) (stack - master) over the sky interior is below the unit's
    #   half-stack noise at 1 deg (recorded as LSSTKADM; checked in the recon stage). 'none' = no admission (refuse)
    SCREEN_CACHE_SUBDIR = 'hybrid_stack_screens'
    #   the derived warp screens (warp.derive_warp_screen) are cached per night and subband in <--work or the site SCRATCH_ROOT>/
    #   <this> (node-local; production: <--scratch>/derived_screens, NL 344, 410), shared by the cell's stacks; a cached screen is
    #   reused only while the raw / dewarped pair it was derived from is unchanged
    SCREEN_LOCK_WAIT_S = 7200
    #   s, how long a stack waits for another process deriving the same night's screen (NL 237-241), then refuses
    RULED = {'FLOW_RECOVERED_AGE_S': 'R-GREGG-20261026-FLOW-RECOVERED-CELLS-BUILT-ON-CURRENT-RECORD-ACCEPTED-FOR-PRODUCTION',
             'FLOW_RECOVERED_RATIO_TOL': 'R-GREGG-20261026-FLOW-RECOVERED-CELLS-BUILT-ON-CURRENT-RECORD-ACCEPTED-FOR-PRODUCTION',
             'ADMISSION': 'R-GREGG-20261031-ADMIT-HB-FLOW-RECOVERED-NOTAPER-STACKS-AS-HYBRIDS',
             'MAX_UNDERIVABLE': 'R-GREGG-20261103-23H-69MHZ-DROP-UNDERIVABLE-NIGHT-FROM-RECON-STACKS',
             'WARP_P90_PX': 'R-GREGG-20261104-HB-13H64-KEEP-23H69-TAPER-ONLY-13H78R0-DROP-NIGHT',
             'MIN_NIGHTS_DROP': 'R-GREGG-20261027-LSRECON-SCALE-BOUND-0P5-2P0-LEVEL-REFUSALS-STAY-GAPS-55MHZ-06H-R0-N-MINUS-1'}
    PROTECTED_KEYS = ('MDEWARP', 'PHASE1_ROOT')


PSF2X_NIGHT_LIST = 'night record (MNC MJD 60770.25, 25 bad antennas)'
#   18-59 MHz (10-26): the observation night's own antenna-health record; the night images used the July list (Q1: the effect is
#   measured on one low band before any remake is decided)
PSF2X_JULY_LIST = 'July record (MNC_MJD 61232.04, matched 61040.25, 46 bad antennas)'
#   64-82 MHz (10-31): the list the Phase 1 images of that night were made with (Gregg 10-31, "exact July antenna lists")


class hybrid_psf2x(_Section):
    """The double-size PSFs (spec 2.2; made outside the package): pinned, checksummed inputs (Q1, Gregg 2026-11-18 "just go with
    your recommendations", R-GREGG-20261118-HYBRID-Q1-Q3-GO-WITH-RECOMMENDATIONS). v2 does not re-image visibilities; the recon
    stage refuses when a file's md5 differs from its entry (gate HG6)."""
    REGISTRY = {
        ('18MHz', 'R075'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_18MHz_R075-psf.fits',
                          '760f65798816c497a77471d2f03b255f', PSF2X_NIGHT_LIST),
        ('18MHz', 'R0'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_18MHz_R0-psf.fits',
                          'b3bb0336fbf820ed181a29e94e160f00', PSF2X_NIGHT_LIST),
        ('23MHz', 'R075'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_23MHz_R075-psf.fits',
                          '9f034fc835dde35ec8c1af2e67a32a90', PSF2X_NIGHT_LIST),
        ('23MHz', 'R0'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_23MHz_R0-psf.fits',
                          'b4a77f3613592884ea1037198b698f8f', PSF2X_NIGHT_LIST),
        ('27MHz', 'R075'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_27MHz_R075-psf.fits',
                          '655d5e51bb50e2fa14cbef24480f1601', PSF2X_NIGHT_LIST),
        ('27MHz', 'R0'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_27MHz_R0-psf.fits',
                          'bcdde3b577a577b254ee56ace948b18c', PSF2X_NIGHT_LIST),
        ('32MHz', 'R075'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_32MHz_R075-psf.fits',
                          '7ed5c52c334fc674f568634963d58ed4', PSF2X_NIGHT_LIST),
        ('32MHz', 'R0'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_32MHz_R0-psf.fits',
                          '819c7897052a39100d251a6f88092a58', PSF2X_NIGHT_LIST),
        ('36MHz', 'R075'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_36MHz_R075-psf.fits',
                          '245f32a0eca93c01b361692e08b92d87', PSF2X_NIGHT_LIST),
        ('36MHz', 'R0'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_36MHz_R0-psf.fits',
                          '398aea30eb522a787c020173a303b0b2', PSF2X_NIGHT_LIST),
        ('41MHz', 'R075'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_41MHz_R075-psf.fits',
                          'f31bd1a8f9f663d115b0a7a89864b587', PSF2X_NIGHT_LIST),
        ('41MHz', 'R0'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_41MHz_R0-psf.fits',
                          '91868c7717ed7c602a09e0e0438cb1b4', PSF2X_NIGHT_LIST),
        ('46MHz', 'R075'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_46MHz_R075-psf.fits',
                          '2bf5cf37f19d7090883c4151b947cf80', PSF2X_NIGHT_LIST),
        ('46MHz', 'R0'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_46MHz_R0-psf.fits',
                          '8ddbf066f8d7bded1d8e15f3270fd290', PSF2X_NIGHT_LIST),
        ('50MHz', 'R075'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_50MHz_R075-psf.fits',
                          '88ef61e9244149549611119f7803aaba', PSF2X_NIGHT_LIST),
        ('50MHz', 'R0'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_50MHz_R0-psf.fits',
                          'b88cd11b6aa6491e5767ed5431bd286c', PSF2X_NIGHT_LIST),
        ('55MHz', 'R075'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_55MHz_R075-psf.fits',
                          'f63fb91bd85f6d474bfbd2fa7217e08d', PSF2X_NIGHT_LIST),
        ('55MHz', 'R0'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_55MHz_R0-psf.fits',
                          'de840eb0cb7e4dc296833584ba82996e', PSF2X_NIGHT_LIST),
        ('59MHz', 'R075'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_59MHz_R075-psf.fits',
                          '447c9da3314ddce86453cdd5aea1b9a5', PSF2X_NIGHT_LIST),
        ('59MHz', 'R0'): ('/lustre/gh/main/phase3/lsrecon/Run_20261026_190545/psf2x/PSF2xF_59MHz_R0-psf.fits',
                          '1e0e709170aded2d73d948e985f77d3f', PSF2X_NIGHT_LIST),
        ('64MHz', 'R075'): ('/lustre/gh/main/phase3/lsrecon/Run_20261031_094930/psf2x/PSF2xF_64MHz_R075-psf.fits',
                          '2af27ad658e2a1f9858aead06c43e47f', PSF2X_JULY_LIST),
        ('64MHz', 'R0'): ('/lustre/gh/main/phase3/lsrecon/Run_20261031_094930/psf2x/PSF2xF_64MHz_R0-psf.fits',
                          'd8e2b131c7931454f81a7ca29404b910', PSF2X_JULY_LIST),
        ('69MHz', 'R075'): ('/lustre/gh/main/phase3/lsrecon/Run_20261031_172532/psf2x/PSF2xF_69MHz_R075-psf.fits',
                          '02cdc086c5ccba0d3aef3cd778bc06aa', PSF2X_JULY_LIST),
        ('69MHz', 'R0'): ('/lustre/gh/main/phase3/lsrecon/Run_20261031_172532/psf2x/PSF2xF_69MHz_R0-psf.fits',
                          'adc37a3c53b4e8cfc788a53f2c0a7691', PSF2X_JULY_LIST),
        ('73MHz', 'R075'): ('/lustre/gh/main/phase3/lsrecon/Run_20261031_094930/psf2x/PSF2xF_73MHz_R075-psf.fits',
                          'b1b70223b3883cea52ae79ce1471ea4d', PSF2X_JULY_LIST),
        ('73MHz', 'R0'): ('/lustre/gh/main/phase3/lsrecon/Run_20261031_094930/psf2x/PSF2xF_73MHz_R0-psf.fits',
                          '0b10e0b8cb0ce2c9e821810d04ac6950', PSF2X_JULY_LIST),
        ('78MHz', 'R075'): ('/lustre/gh/main/phase3/lsrecon/Run_20261031_172532/psf2x/PSF2xF_78MHz_R075-psf.fits',
                          'a892b978303b8be55d08d5abdd63dab6', PSF2X_JULY_LIST),
        ('78MHz', 'R0'): ('/lustre/gh/main/phase3/lsrecon/Run_20261031_172532/psf2x/PSF2xF_78MHz_R0-psf.fits',
                          '4311b2939553fa2527140716fb3499b8', PSF2X_JULY_LIST),
        ('82MHz', 'R075'): ('/lustre/gh/main/phase3/lsrecon/Run_20261031_094930/psf2x/PSF2xF_82MHz_R075-psf.fits',
                          'a23a13bb85816bb753e7c9374efabade', PSF2X_JULY_LIST),
        ('82MHz', 'R0'): ('/lustre/gh/main/phase3/lsrecon/Run_20261031_094930/psf2x/PSF2xF_82MHz_R0-psf.fits',
                          'a410f5c33720c971f2f1add53bc12775', PSF2X_JULY_LIST),
    }
    #   {(band, weighting): (path, md5, antenna list)}: the 30 files production's reconstructions read (lsrecon RS/psf2x for
    #   18-59 MHz; R2/psf2x for 64-82 MHz, whose 64, 73 and 82 MHz files link into Run_20261031_094930/psf2x - the real path is
    #   pinned). md5 recomputed 2026-11-18 with plain reads, equal to every production .md5 sidecar (30/30).
    PROTECTED_KEYS = ('REGISTRY',)


class hybrid_recon(_Section):
    """Stages `recon` and `wideband` (spec 2.3, 2.4): the NoTaper dirty stack re-deconvolved in the image domain with a
    large-scale step toward the LWA1 survey prior."""
    SCALES = (0, 12, 24, 47)
    #   px, NL 73 (95 px made the Robust 0 CLEAN diverge, pilot 02h 46 MHz)
    NITER = 500000
    GAIN = 0.1
    MGAIN = 0.6
    BIAS = 0.6
    AUTOMASK = 3.0
    AUTOTHRESH = 0.5
    PATCH = 256
    MAXACT = 100000
    MAXLOOP = 20000
    #   NL 74: the multiscale CLEAN parameters
    MAXCYC = 400
    #   NL 76: the major-cycle cap
    CASCADE = (('standard', 1, 'horizon'), ('3x', 3, 'horizon'), ('elev10', 3, 'elev10'))
    #   NL 2209-2215 (fleet-recon): the tries of a unit refused by a convergence test, in order: (name, factor on NITER and
    #   MAXCYC, CLEAN region); each later try records the earlier ones (escalation); 'elev10' = CLEAN to 10 deg elevation
    #   (NL 891), card LSCLNRGN
    TAU = 0.15
    STEP_FWHM_DEG = 1.0
    CG_TOL = 1e-6
    CG_MAX = 3000
    #   NL 75: the large-scale step (prior weight, smoothing) and its conjugate-gradient solve
    PRIOR_SCALE_RANGE = (0.5, 2.0)
    PRIOR_RMS_FRAC = 0.6
    #   NL 639 (test_prior_fit): the prior's fitted scale strictly inside the range and the residual rms at most 0.6 of the data rms
    COMPACT_TOL = 0.10
    COMPACT_TOL_LOW = 0.20
    COMPACT_LOW_SCOPES = ('18MHz', 'Red')
    COMPACT_N_SOURCES = 200
    COMPACT_N_MIN = 50
    #   NL 785-820 (test_compact_sources): the 200 brightest catalogued sources, at least 50 measured, median new / master ratio
    #   within 10% of 1, 20% at 18 MHz and Red (Red holds 18 MHz)
    LEVEL_LIMITS = (0.5, 1.6)
    DEC_BANDS = ((40, 90), (10, 40), (-10, 10), (-20, -10), (-30, -20))
    #   NL 829-830: the large-scale level test per declination band: a refusal stays a gap, never forced (2026-10-27 (2));
    #   a recorded caveat in hybrid.CAVEAT_SCOPES (2026-10-31)
    CLEAN_REGION = 'horizon'
    #   NL 870: CLEAN, step and prior to the true horizon ('elev10' = the 10-degree cut of the earlier generation)
    LFSS_DIR = '/lustre/gh/main/catalogs/lwa1_lfss'
    #   NL 64: the LWA1 Low Frequency Sky Survey maps (the prior)
    CATALOGS = '/lustre/gh/main/phase3/deep_parent_catalogs_union_20261014T0203'
    #   NL 65: the parent catalogs test_compact_sources reads (positions and peaks of the brightest sources only; no digest
    #   check). PINNED to production's value for parity. NOTE 2026-11-18: this union is stale on 47 of 456 cells after the
    #   Release 1 master rebuilds (RESULT-20261118-DEEP-PARENT-UNION-REFRESH); the current union is
    #   deep_parent_catalogs_union_20261118T0130 - which one a v2 rebuild reads is the deep chat's call, recorded there
    RULED = {'PRIOR_SCALE_RANGE': 'R-GREGG-20261027-LSRECON-SCALE-BOUND-0P5-2P0-LEVEL-REFUSALS-STAY-GAPS-55MHZ-06H-R0-N-MINUS-1',
             'COMPACT_TOL_LOW': 'R-GREGG-20261026-LSRECON-18-59MHZ-FLEET-APPROVED-KEEP-MASTERS-WIDEBAND-HIGH-PRIORITY-SPLIT-PB-COADD-ARCHIVE',
             'CLEAN_REGION': 'R-GREGG-20261028-HYBRID-NOTAPER-RERUN-APPROVED-CLEAN-TO-TRUE-HORIZON-HYBRID-RESIDUAL',
             'CASCADE': 'R-GREGG-20261028-HYBRID-NOTAPER-RERUN-APPROVED-CLEAN-TO-TRUE-HORIZON-HYBRID-RESIDUAL',
             'LEVEL_LIMITS': ('R-GREGG-20261027-LSRECON-SCALE-BOUND-0P5-2P0-LEVEL-REFUSALS-STAY-GAPS-55MHZ-06H-R0-N-MINUS-1',
                              'R-GREGG-20261031-HIGH-BAND-CAVEATS-APPROVED-GO-ON-EVERYTHING')}
    PROTECTED_KEYS = ('LFSS_DIR', 'CATALOGS')


class hybrid_unit(_Section):
    """Stage `hybrid` (spec 2.5): hybrid master = Taper master (striation-restored where ruled) + G (x) (reconstruction - Taper),
    its beam-corrected sibling, CLEAN residual, PB maps; every product read back and tested."""
    OWN_BEAM_MAX = 1e-4
    #   NL 1300: the reconstruction re-restored with the Taper beam must reproduce itself to 1e-4 (gate HG8; production b3be251)
    IDENTITY = 1e-5
    #   NL 1220, 1247, 1259: a written product read back equals the array computed to 1e-5 of its largest value
    LARGE_REL = 1e-3
    #   NL 1220: the hybrid's large scales equal G (x) (reconstruction - Taper) smoothed, to 1e-3
    SMALL_LIMIT = 5e-3
    #   NL 831, 858, 1219 (HYB_SMALL_LIMIT): above HYB_U_OUT the hybrid's small scales are the Taper's, rel rms < 5e-3 (a
    #   recorded caveat in CAVEAT_SCOPES, card LSSMALL). Here, not in hybrid_recon (spec 3.3): the hybrid stage applies it
    STRIATION_IDENTITY = 1e-6
    #   NL 1134 (taper_base): restored + model equals the installed master to 1e-6 of its peak, i.e. the base is made from THIS
    #   master (gate HG3)
    RAW_NIGHT_CAVEAT = {('13h', '64MHz'): 'nights 2025-04-21, 05-02, 05-03, 05-05, 05-07 entered the reconstruction stacks '
                                          'undewarped (Phase 2 saved no screen; raw vs dewarped 0-1 px median, <= 2.2 px 90th pct '
                                          'at Robust -0.75, ~0.3 beam) - kept as a recorded caveat'}
    #   NL 1681-1682: cells kept with reconstructions whose stacks took nights undewarped (card LSRAWNT on their hybrids)
    RULED_TAPER_ONLY = {}
    #   NL 1679: units that enter the mosaics as the plain Taper master (empty since 2026-11-08: a hybrid for every cell); a
    #   refused unit refuses its mosaics (gate HG14)
    JUMP10 = 'record'
    #   NL 1206-1218: the jump across 10 deg elevation is recorded as a diagnostic, never a gate
    RULED = {'RAW_NIGHT_CAVEAT': 'R-GREGG-20261104-HB-13H64-KEEP-23H69-TAPER-ONLY-13H78R0-DROP-NIGHT',
             'RULED_TAPER_ONLY': 'R-GREGG-20261108-MIN-HYBRID-EVERY-CELL-BOTH-ROBUSTS',
             'JUMP10': 'R-GREGG-20261028-HYBRID-10DEG-JUMP-DIAGNOSTIC-NOT-GATE',
             'STRIATION_IDENTITY': 'R-GREGG-20261031-HIGH-BAND-HYBRID-PLAN-AND-STRIATION-RESTORED-BASE-APPROVED'}
    PROTECTED_KEYS = ()


class hybrid_full(_Section):
    """Stage `full` (spec 2.6; NL hybrid_full 1404): the Full hybrid from the cell's 15 passed subband hybrids, regridded onto the
    Taper Full grid, weighted by 1/THERMNOI^2, through G."""
    MEMBERS = ('18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz',
               '64MHz', '69MHz', '73MHz', '78MHz', '82MHz')
    #   NL 947: Full = all 15 subbands; fewer than 15 passed subband hybrids refuses (gate HG11)
    REGRID = 'bilinear'
    #   NL 1409: the subband hybrids sit on three grids and are regridded bilinearly (only scales above ~2 deg are kept)
    PBDIFFUSE_WEIGHTING = 'R0'
    #   the cell's diffuse PB map PBdiffuse_deep_Full.fits is the same for both weightings (the same subband maps and 1/THERMNOI^2
    #   weights; 18h checked 2026-11-18: weights equal) but carries its writer's header: only this weighting's unit writes it. R0 =
    #   the archived maps' writer (Run_NOTAPER_HYBRID_20261028 18h Full: BMAJ = the R0 product's; production wrote it from both)
    RULED = {'MEMBERS': 'R-GREGG-20261108-MIN-HYBRID-EVERY-CELL-BOTH-ROBUSTS'}
    PROTECTED_KEYS = ()


class hybrid_tilefill(_Section):
    """Stage `tilefill` (spec 2.14; MF fill_master_unit 306): the per-tile zero-spacing fill of the hybrid masters from the band
    fill field of the filled mosaic."""
    TILE_RT_NSIDE = 256
    TILE_RT_MAX = 0.02
    #   MF 192: round trip through the mosaic's tile conversion, median |back - field| / median |field| at NSIDE 256
    COMPACT_K = 30
    COMPACT_MED_MAX = 0.005
    COMPACT_MAX = 0.02
    COMPACT_SIG = 10.0
    #   MF 193, 199: the 30 brightest compact maxima of at least 10 x the robust rms: |change of peak - annulus median| / that
    #   contrast, median <= 0.005 and max <= 0.02
    TPCAVEAT_UNITS = (('03h', '55MHz', 'R0', 'pbcorr'), ('03h', '59MHz', 'R0', 'pbcorr'))
    #   HMF 41-44, 124-143: the two beam-corrected files on the ruled caveat (Cygnus X at the 03h rim; card TPCAVEAT)
    UNFILLED_INPUTS = None
    #   the manifest of the unfilled hybrid masters the archived fill was made from (HQ11 and HQ13/unfilled, keyed by TPSRCMD5);
    #   None until Q2 (spec section 8: where the kept intermediates live) is answered - the stage refuses without it
    RULED = {'TILE_RT_MAX': 'R-GREGG-20261111-HYBRID-MASTERS-ZERO-SPACING-FILL-PER-TILE',
             'COMPACT_SIG': 'DONE-20261111-HYBRID-MASTERS-ZERO-SPACING-FILL-INSTALLED',
             'TPCAVEAT_UNITS': 'DONE-20261111-HYBRID-MASTERS-ZERO-SPACING-FILL-INSTALLED'}
    PROTECTED_KEYS = ('TPCAVEAT_UNITS', 'UNFILLED_INPUTS')
