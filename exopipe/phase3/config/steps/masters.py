"""Configuration of `exopipe3 masters` - every tunable of the command, one section per stage or tier.

Units folded here: masters-deep, masters-tiers. A key in its section's PROTECTED_KEYS is locked
(a mirror of in-place code, a pinned input or a ruling-fixed value: it changes only by a commit); RULED names the
ledger ID of the ruling behind a key. Per run: --set SECTION.KEY=VALUE or --config FILE; `exopipe3 config --step masters` lists the sections.
"""
import os
from exopipe.phase3.config import _Section

_PHASE3 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # exopipe/phase3


# =====================================================================================================================
# unit masters-deep (was exopipe/phase3/_cfgdraft_masters-deep.py)
# Configuration of `exopipe3 masters` - the shared section `masters` (step owner) and the DEEP tier sections
# (unit masters-deep).
#
# Source of every value: plan/specs/masters.md section 3 (values = today's production values, file:line of the production
# code cited per key). The sub-LST and snapshot tier sections (masters_sublst, masters_snapshot) are NOT here: they are
# the tier agents' drafts.
#
# Conventions (refactor chat decisions 2026-10-13 ~04:30, _CROSSCHECK.md): UPPER-case keys only, with ``RULED`` (ledger ID of
# the ruling that set a key) and ``PROTECTED_KEYS`` (keys no --set / --config may change; the config machinery refuses them, exit 2).
# PROTECTED_KEYS holds (a) pinned inputs and ruling-fixed values, (b) every MIRROR key - a literal or a default argument bound
# inside the in-place code (lib/masters/*.py moved by pure git mv; exopipe_ops scripts): the value documents it, an override
# could not reach the code, so it is refused until the module is absorbed into lib/ and reads config - and (c) values whose
# alternatives are not implemented (masters_build.EPOCH_GATE_SOURCE, B-05). How each key reaches the in-place code (module
# attribute, environment at import or at call, call argument, or wrapper only) is a table in code:
# lib/masters/deep_inplace.WIRING. Shared roots come from config.paths (COADD_SB_ROOT, ARCHIVE_RUN, OPS_ROOT, PHASE2_ROOT,
# PHASE3_ARCHIVE_ROOT), never from a key of this draft.
# =====================================================================================================================

class masters(_Section):
    """exopipe3 masters - shared values: tiers, stage order, the deep generation root, interpreters per stage."""
    TIERS = ('deep', 'sublst', 'snapshot')
    # stage order per tier (spec 1.2). Deep: pbmaps BEFORE vclean (rebuild_cell.sh:137-140: the cleaned-V sibling divides by
    # PB_deep_<scope>); stage-in BEFORE the `flow` step (the flows align to the reference stage-in writes; the flow step copies
    # the stored flows and computes the stale ones - crosscheck 1.1 group 1). Sub-LST 'wideband' is the frames step's
    # (crosscheck 1.1 group 3; decision 6), not a masters stage.
    STAGES = {'deep': ('stage-in', 'build', 'pbmaps', 'vclean', 'fluxscale', 'verify', 'sources', 'variants'),
              'sublst': ('prep', 'admit', 'stack', 'loo', 'sources'),
              'snapshot': ('verify-frozen', 'freeze', 'sources')}   # sources: snapshot parent catalogues (M21; review 2026-10-13)
    # deep default = the build part of the chain 'deep-build-cell' (masters --stages stage-in -> flow -> masters (default)):
    # a single masters invocation cannot hold stage-in and build, the flow step runs between them
    DEFAULT_STAGES = {'deep': ('build', 'pbmaps', 'vclean', 'fluxscale', 'verify'),
                      'sublst': ('prep', 'admit', 'stack', 'loo'),
                      'snapshot': ('verify-frozen',)}
    # default product root of a deep build when --out is not given and the site file names no MASTERS_GEN_ROOT for this host:
    # production's scratch generation root (rebuild_cell.sh:19 R=); it is under paths.PRODUCTION_ROOTS, so writes refuse
    # while paths.PRODUCTION_WRITES is False (dry runs only)
    DEEP_GEN_ROOT = '/fast/gh/main/phase3/rebuild_20261007'
    # Python interpreter per stage, as site-file PYTHON keys (decision 2: default, flow, py36, casa, tess only):
    # py38_orca_nkosogor = 'default' (rebuild_cell.sh:24); /opt/devel/pipeline/envs/development = 'py36'
    # (sublst_run_lst_nocat.sh:29, sublst_stack_pass.sh:10)
    PYTHON_SITE_KEY = {'deep': 'default', 'sublst-stack': 'py36', 'sublst-loo': 'py36', 'sources': 'default'}
    FREE_FLOOR_GB = {'deep': 40}                       # GB free on the --out filesystem before a deep unit starts (chain_cell.sh:19-20; operational)
    # roots that hold production data but are not in paths.PRODUCTION_ROOTS (spec 12 item 3): the step refuses to write under
    # them (and under paths.OPS_ROOT) while paths.PRODUCTION_WRITES is False (deep parent catalogues)
    EXTRA_PRODUCTION_ROOTS = ('/lustre/gh/main/phase3_collect',)
    PROTECTED_KEYS = ('TIERS', 'STAGES', 'EXTRA_PRODUCTION_ROOTS')


class masters_stagein(_Section):
    """deep stage-in: the native (unscaled), un-warped reference written under the canonical names and the epoch-gate record
    carried in (port of rebuild_cell.sh:23, 36-43, 47-120 in lib/masters/deep_stagein.py; the flow copy :44 is the flow
    step's). The working cell is paths.COADD_SB_ROOT/<cell> unless REF_OVERRIDE_TABLE names another root."""
    REF_PIXEL_FACTOR_RANGE = (0.3, 3.0)   # SHSCALE x FLUXCORR of a baked reference must lie strictly inside (rebuild_cell.sh:70, 87, 107)
    # B-12 default (c): ref_override.csv (working-cell root) honoured, pinned by sha1
    REF_OVERRIDE_TABLE = '/lustre/gh/main/exopipe_ops/membership_rebuild_20261007/ref_override.csv'        # 8 rows; ROUND-RESTART-LIST-A-87-CELLS-FLEET-QUEUE-ALIGNED (ledger 2076); rebuild_cell.sh:23
    REF_OVERRIDE_TABLE_SHA1 = '76016a8b6b2517df3c77895cc4975a5de14b0434'
    # The reference resolver (lib/masters/deep_stagein.resolve_reference): same constituents -> previous generation -> refuse,
    # over pre-screen (un-warped) copies indexed by CONSTDIG. Search order: (label, directory or glob, layout); within a glob
    # the newest directory (by name) first; the first root with a copy wins. layout 'cell' = <dir>/<cell>/coadd_final/<var>_<sc>.fits,
    # 'flat' = <dir>/<cell>/<var>_<sc>.fits, 'archive' = <dir>/<LST>/<sc>/<sc>_<st>_deep_Taper_<rob>_*.fits (native only; exactly 1).
    REFERENCE_COPY_ROOTS = (
        ('predewarp-quarantine', '/lustre/gh/main/phase3/working_quarantine_20261004', 'cell'),             # rebuild_cell.sh:80-93
        ('superseded-archive', '/lustre/pipeline/exopipe/phase3/Coadd/Run_20260909_000000_SUPERSEDED_20260927', 'archive'),   # :94-104
        ('unwarped-copy', '/fast/gh/main/phase3/unwarped_ref_*', 'flat'),           # ref_override_I.csv's copies (lwacalim07-local); :53-79
        ('quarantine-copy', '/lustre/gh/main/phase3/working_quarantine_*', 'cell'), # every later working quarantine
    )
    PREVIOUS_GENERATION_REFERENCE = True  # Gregg 2026-10-13 '(A) is approved' (exopipe_ops e5c218e): no copy of the current constituents
                                          # -> the replaced master's pre-screen copy (its CONSTDIG = the current master's ALIGNREF)
    V_OPTIONAL_BRANCHES = ('unwarped-copy', 'quarantine-copy', 'previous-generation')   # no V copy = no V reference only here (rebuild_cell.sh:65-67, e5c218e); elsewhere refused
    # Production's hand tables: per-row CROSS-CHECKS of the resolver (a row must name the file chosen), not inputs; '' = off
    REF_OVERRIDE_I_TABLE = '/lustre/gh/main/exopipe_ops/membership_rebuild_20261007/ref_override_I.csv'    # R-GREGG-20261009-DEWARP-CASCADE-23HGREEN-3CELLS-FAST; rebuild_cell.sh:53-79
    REF_PREVGEN_TABLE = '/lustre/gh/main/exopipe_ops/membership_rebuild_20261007/ref_prevgen.csv'          # exopipe_ops e5c218e
    PREDEWARP_QUARANTINE = '/lustre/gh/main/phase3/working_quarantine_20261004'   # (= REFERENCE_COPY_ROOTS[0]; kept for the plan text)
    SUPERSEDED_ARCHIVE = '/lustre/pipeline/exopipe/phase3/Coadd/Run_20260909_000000_SUPERSEDED_20260927'   # (= REFERENCE_COPY_ROOTS[1])
    EPOCH_GATE_CARRY_DIR = '/lustre/gh/main/exopipe_ops/membership_rebuild_20261007/epoch_gate_carry'    # empty since 2026-10-10 (rebuild_cell.sh:41); read only when EPOCH_GATE_CARRY is 'carry'
    EPOCH_GATE_CARRY = 'none'                 # B-05 (a), Gregg 2026-10-13: no carried epoch-gate record - the build gates every night on its OWN master
                                              # (masters_build.EPOCH_GATE_SOURCE 'own-master'); 'carry' = production (rebuild_cell.sh:39-43: the carried record, else the cell's own)
    STRIP_CARD_PREFIXES = ('SH', 'FLUX')  # cards removed from a reconstructed native reference (rebuild_cell.sh:74, 91, 111)
    PROTECTED_KEYS = ('REF_OVERRIDE_TABLE', 'REF_OVERRIDE_TABLE_SHA1', 'REF_OVERRIDE_I_TABLE', 'REF_PREVGEN_TABLE',
                      'PREVIOUS_GENERATION_REFERENCE', 'V_OPTIONAL_BRANCHES')
    RULED = {'REF_OVERRIDE_TABLE': 'ROUND-RESTART-LIST-A-87-CELLS-FLEET-QUEUE-ALIGNED',
             'EPOCH_GATE_CARRY': 'R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32',
             'REF_OVERRIDE_I_TABLE': 'R-GREGG-20261009-DEWARP-CASCADE-23HGREEN-3CELLS-FAST',
             'PREVIOUS_GENERATION_REFERENCE': 'IMPL-REBUILD-REFERENCE-A-PREVIOUS-GENERATION-PRESCREEN-20261013',
             'REF_PREVGEN_TABLE': 'IMPL-REBUILD-REFERENCE-A-PREVIOUS-GENERATION-PRESCREEN-20261013'}


class masters_build(_Section):
    """deep build: provisional + final co-add (in place: lib/masters/coadd.py run_provisional_lst_freq / run_final_lst_freq)."""
    # admission
    CUTOVER_DATE = '2024-12-27'               # nights before are regridded; after, off the modal grid = refused (coadd_full_pipeline.py:79; R-WRONG-GRID-NIGHTS-LEAVE-EXCLUDED)
    FLOW_MAX_NAN_FRACTION = 0.5               # flow more than 50 per cent NaN refuses the night (coadd_full_pipeline.py:1830-1836; FINDING-FIVE-GOOD-NIGHTS-BAD-RESIDUALS-ARE-FLOW-FAILURES)
    ALLOW_STALE_FLOW = False                  # P3_ALLOW_STALE_FLOW bypass of the freshness gate - never set (coadd_full_pipeline.py:2176; RATIFY-COADD-FLOW-REFRESH-STAGE-AND-FRESHNESS-GATE)
    INPUT_RESOLVER = 'core.inputs'            # B-03 (a): the co-add's five image finders replaced by core.inputs' copy (lib/masters/input_resolver.py; the same file chosen,
                                              # pinned); 'coadd-finders' = production (coadd_full_pipeline.py:653-753, 2912-2946)
    RAW_INPUT_FLOW_RULE = 'same-state'        # B-03 (a), the same-state reading (Gregg 2026-10-14): a night's V / I-variant image whose dewarp state differs from the night's
                                              # survey image is refused (refusals_input.csv; needs INPUT_RESOLVER core.inputs); 'survey-image-flow' = production: every product
                                              # takes the survey image's flow (defect M9: 26 V / 40 / 35 / 26 variant night-cells raw on a dewarped survey, 1 each dewarped on raw)
    PRODUCT_GRID_MISMATCH = 'regrid'          # B-03 (a) "one regrid for every off-grid input": a night's V / I variant on another grid (shape) than its I image is regridded
                                              # from its own WCS in the provisional and final stage (coadd._product_mapping); 'drop' = production: the provisional stage drops
                                              # such a V (coadd_full_pipeline.py:2339-2351; FIX-COADD-V-FRAME-ON-DIFFERENT-GRID-DROPPED-LOUDLY) and the final stage hands every
                                              # product its I image's regrid mapping (was masters_build.PROVISIONAL_V_GRID_MISMATCH, a mirror)
    OFF_GRID_SURVEY = 'regrid'                # B-03 (a): a post-cutover night whose survey image is off the cell's modal grid is regridded (coadd._off_grid at every regrid
                                              # decision, the build and the flow step); 'refuse' = production (coadd._grid_refused_dates, refusals_grid.csv)
    WRONG_GRID_NIGHTS = (('02h', 'Green', '2025-12-26'), ('12h', 'Full', '2025-04-11'), ('14h', 'Full', '2025-05-07'),
                         ('19h', 'Full', '2025-04-05'))   # the Phase 2 images on a wrong grid (R-WRONG-GRID-NIGHTS-LEAVE-EXCLUDED 'Leave excluded.'): under
                                              # OFF_GRID_SURVEY 'regrid' a build or flow run of their cell refuses until data_decisions.csv excludes the night (Gregg's rows)
    # reference
    REF_MODE = 'master'                       # previous-generation I Taper R-0.75 master (coadd_full_pipeline.py:1607; R-ALIGNMENT-REFERENCE-IS-PREVIOUS-GENERATION-MASTER)
    REFERENCE_DATES = {'00h': '2025-08-09', '01h': '2024-12-21', '02h': '2024-12-21', '03h': '2024-12-21', '04h': '2024-12-21',
                       '05h': '2024-12-29', '06h': '2024-12-21', '07h': '2024-12-21', '08h': '2024-12-21', '09h': '2024-12-21',
                       '10h': '2024-12-21', '11h': '2024-12-28', '12h': '2024-12-29', '13h': '2025-05-03', '14h': '2025-05-03',
                       '15h': '2025-05-03', '16h': '2025-05-03', '17h': '2025-05-03', '18h': '2025-05-03', '19h': '2025-05-03',
                       '20h': '2025-05-03', '21h': '2025-08-05', '22h': '2025-08-05', '23h': '2025-09-06'}
                                              # fallback night only when no previous master exists (coadd_full_pipeline.py:90-132; R-05H-REFERENCE-20241229 et al.; B-61)
    REF_ASTROM_MAX_ARCMIN = 3.0               # arcmin, median VLSSr separation (coadd_full_pipeline.py:1441; A-02 / A-02-rev)
    REF_ASTROM_PIVOT_MHZ = 36.0               # MHz; below it the threshold scales by (36/nu)^1.5 (coadd_full_pipeline.py:1457)
    REF_ASTROM_EXPONENT = 1.5                 # (coadd_full_pipeline.py:1457)
    REF_ASTROM_MIN_ANCHORS = 10               # anchors; fewer = refused (coadd_full_pipeline.py:1520-1528; A-02-rev)
    REF_ASTROM_N_PEAKS = 50                   # brightest central peaks tested (coadd_full_pipeline.py:1509)
    REF_ASTROM_MATCH_MAX_ARCMIN = 60.0        # arcmin, a peak further than this from VLSSr is not counted (coadd_full_pipeline.py:1518)
    VLSSR_PEAKS = '/lustre/gh/calibration/pipeline/reference/surveys/vlssr_radecpeak_unresolved.txt'   # coadd_full_pipeline.py:1442
    # per-night flux scale (first pass flat DC; IMPL-COADD-FLUX-SCALE-ESTIMATOR-AT-PEAK-PIXEL; LANDED-FLAT-DC-FLUX-SCALE-V3)
    FLUXSCALE_ESTIMATOR = 'at_peak_pixel'     # applied scale measured on the flow-aligned night at the reference peak pixels (coadd_full_pipeline.py:921-980)
    FLUXSCALE_SNR_MIN = 10.0                  # sigma, reference peak S/N (coadd_full_pipeline.py:939)
    FLUXSCALE_PEAK_JY = (10.0, 200.0)         # Jy/beam, reference peak range (coadd_full_pipeline.py:940)
    FLUXSCALE_MIN_ELEV_DEG = 45.0             # deg (coadd_full_pipeline.py:948)
    FLUXSCALE_APERTURE_BEAMS = 0.75           # x beam, at least 3 px (coadd_full_pipeline.py:932)
    FLUXSCALE_MAXFILTER_BEAMS = 1.5           # x beam, at least 5 px (coadd_full_pipeline.py:933)
    FLUXSCALE_COMPACTNESS = 2.5               # flux above 0.3 x peak within the aperture / peak (coadd_full_pipeline.py:970-971)
    FLUXSCALE_MIN_SOURCES = 5                 # fewer = not measurable -> night refused (coadd_full_pipeline.py:976; R-FLUX-CORRECT-EVERY-ADMITTED-NIGHT)
    FLUXSCALE_IRLS = (3, 3.0)                 # iterations, clip at x MAD x 1.4826 (coadd_full_pipeline.py:989-1000)
    FLUXSCALE_TILT = False                    # P3_FLUXSCALE_TILT comparison switch - never set (R-TILT-REMOVED-NEXTGEN)
    DEGENERACY = dict(n_min=8, scale_range=(0.7, 1.4), tilt_max=0.3)        # first-pass guard -> qa3_guard 'degenerate-flatDC' (coadd_full_pipeline.py:2302, 2613, 3663)
    SECOND_PASS = dict(n_min=8, range_x_median=(0.6, 1.6), tilt_x_median=0.3, degenerate='cell-median')   # B-08 default (b) (coadd_full_pipeline.py:3956-3961)
    SAFE_SCALE_FLOOR = 0.1                    # |scale| below = 1.0 used (coadd_full_pipeline.py:1070, 1787, 3964)
    # stage 2 (provisional; CORRECTION-STAGE2-PSFMATCH-REINTRODUCED-20261007-REMOVED)
    STAGE2_METRIC = 'nonmatched-provisional-v1'   # membership.py:43
    STAGE2_MARGIN_FRAC = {'I': 0.15, 'V': 0.25}   # residual rms on the inner 70 / 50 per cent (coadd_full_pipeline.py:2399, 2406)
    STAGE2_CLIP_SIGMA = 4.0                       # sigma clip inside compute_residual_rms (coadd_full_pipeline.py:1102)
    OUTLIER_SIGMA = 3.0                       # sigma, high side (coadd_full_pipeline.py:140; C-03)
    LOW_OUTLIER_SIGMA = 4.0                   # sigma, low side (coadd_full_pipeline.py:141; R-LOW-OUTLIER-4SIGMA, R-GC5-ASYMMETRIC-GATE-WANTED)
    MAD_MIN_VALUES = 5                        # fewer finite values = no outliers at all (coadd_full_pipeline.py:1157-1158; silent today)
    BEAM_GATE_RATIO = 1.5                     # x set-median fitted-beam area (coadd_full_pipeline.py:1136; R-BEAM-GATE-APPROVED-AND-SHIPPED)
    BEAM_GATE_MIN_AREAS = 5                   # fewer readable areas = gate does nothing (coadd_full_pipeline.py:1134)
    BEAM_GATE_IMAGE = 'admitted-image'        # D1 (Gregg 2026-10-13 'Approved'; exopipe 24929f8, merged 622fed1): the beam of the image the co-add admits - latest
                                              # Science dir, Taper R-0.75, dewarped else the RAW failsafe; no image / unreadable header = that night unjudged (coadd.coadd_beam_gate)
    PROVISIONAL_SIGCLIP = (True, 4.0, 4)      # (clip, sigma, minimum images): provisional co-add clips (coadd_full_pipeline.py:1253-1254, 2371-2382); B-09 default (b)
    LEGACY_STAGE2_RECORD = 'accept-residuals_provisional'   # B-04 default (c): membership.stage2_record falls back (membership.py:155-184)
    # final stage membership (two-tier: products for every non-excluded night)
    EPOCH_GATE_SOURCE = 'own-master'          # B-05 (a), Gregg 2026-10-13: after the final co-add every night is tested against THIS build's master and the final
                                              # co-add reruns without the in-master FAIL nights (lib/masters/epoch_gate_passes.py; exopipe_ops/refactor_20261012/plan/wave2/B05_DESIGN.md);
                                              # 'carried-record' = production: coadd_final/epoch_gate.csv read with check_master=False (coadd_full_pipeline.py:3492-3499)
    EPOCH_GATE_MAX_PASSES = 3                 # passes of the final co-add at most (B-05 'a bounded number of passes'); the cell refuses when the last pass still
                                              # has a FAIL night in the master, and at any pass whose own record breaks membership (a removed night that passes
                                              # the master built without it: Gregg names it out of the master). 3 and the refusal: Gregg 2026-10-14 approved the
                                              # refactor chat's recommendation; production 1 (no rebuild pass)
    MASTER_SPAN_SOURCE = 'constituent'        # the master time span (DATE-OBS/-END, INTTIME, MJD-*) from each constituent's OWN image (the one the
                                              # co-add used) - found by the B-30 skeptic, Gregg 2026-10-14 "Approved."; 'phase2-glob' = production
                                              # (coadd_full_pipeline.py:428-449 globs miss raw-failsafe and non-3xxx-s images: 688 night-cells in
                                              # 221 of 456 cells, 19h 41MHz DATE-OBS 2025-04-21 for 2025-04-10, INTTIME 38870 s for 52171 s)
    SOLAR_AUDIT = '/lustre/pipeline/exopipe/phase3/QA/solar_contamination_audit.csv'   # membership.py:72; R-SUNLIT-NIGHTS-OUT-OF-MASTERS
    SOLAR_AUDIT_SHA1 = '24151ffaa1b7f11faa7adeab5d0f9cb6cfae8da4'
    NAMED_OUT_OF_MASTER = os.path.join(_PHASE3, 'named_out_of_master.csv')   # package data file in exopipe/phase3/, 1 row; membership.py:98; R-GREGG-20261011-17H18-0505-OPTION-B
    NAMED_OUT_OF_MASTER_SHA1 = '57f9e985fb2d6bacc3c822772261223375786f46'
    QA3_HI = 1.6                              # stage-3 window upper edge (membership.py:45; R-THIRD-STAGE-QA-FLUXALIGN, R-QA3-BOUNDS-PROCESS-APPROVED)
    QA3_LO = {'18MHz': 0.35, '23MHz': 0.55, '27MHz': 0.55}   # membership.py:46
    QA3_LO_DEFAULT = 0.7                      # membership.py:47 (32 and 36 MHz take 0.7: no separate ruling found, membership.py:21)
    NIGHT_GATE = dict(threshold=0.5, max_shift_deg=3.0, blowup_jy=1e6, block=4, radius=0.7)   # night_image_gate.py:24-28; R-NIGHT-IMAGE-GATE-BEFORE-BOTH-GATES
    MIN_DATES_FOR_COADD = 3                   # survey I and V masters (coadd_full_pipeline.py:84; no ruling found, ledger 2219)
    MIN_DATES_FOR_VARIANT = 2                 # I Taper R0, NoTaper R0/R-0.75 (coadd_full_pipeline.py:88; R-GREGG-20261012-VARIANTS-FROM-2-NIGHTS-23HGREEN)
    VARIANT_MAX_ABS_JY = 1e5                  # Jy/beam, absolute divergence gate (coadd_full_pipeline.py:2902; R-NONSURVEY-VARIANT-RECOVERY-CAMPAIGN-GO)
    VARIANT_MIN_FINITE = 0.99                 # finite-pixel fraction (coadd_full_pipeline.py:2902)
    VARIANT_REL_K = 3.0                       # x cell-median and band-wide variant/survey rms ratio (coadd_full_pipeline.py:2925; R-VARIANT-GATE-3X-WITHIN-CELL)
    VARIANT_BAND_MEDIANS = os.path.join(os.path.dirname(_PHASE3), 'common', 'variant_band_medians.json')   # package data, frozen census 2026-09-29 (coadd_full_pipeline.py:2984-2985)
    VARIANT_BAND_MEDIANS_SHA1 = 'c0e96b34dc7fe6916ba7c3147b1ae22f9c5180cb'
    VARIANT_FLUXDC_LEGACY = 'qa3_scale-0.3-3' # B-04 default (c): variant-only rebuild of records before 2026-10-02 (coadd_full_pipeline.py:3353-3354)
    VARIANTS_ONLY = ('Taper_R0', 'NoTaper_R0')   # default of the variants stage (coadd_full_pipeline.py:4363)
    COADDOP = 'ivw-mean-v3'                   # final masters: plain inverse-variance mean (LANDED-FLAT-DC-FLUX-SCALE-V3; R-LOO-MASTERS-AND-REDO-SEQUENCE)
    FLUXFIT = 'flat-dc'                       # FLUXFIT card of a v3 master (coadd_full_pipeline.py:1330-1331; R-TILT-REMOVED-NEXTGEN)
    LOO_N_RANGE = (2, 20)                     # nights per Stokes stack (coadd_full_pipeline.py:1212-1215; R-LOO-THRESHOLD-CORRECTED, R-LOO-PER-STOKES-RATIFIED)
    STRIATION_WINDOW_BEAMS = 8.0              # x BMAJ, RA running median (coadd_full_pipeline.py:3981, 4008; R-STRIATION-WINDOW-APPROVED-8BEAMS)
    EPOCH_CHECK = dict(identity_max=1e-3, peak_snr=20.0, peak_max_jy=50.0, radius=0.9, box_px=9, min_sources=20,
                       ratio=(0.8, 1.25))     # coadd_full_pipeline.py:3150-3175; R-EPOCH-CONTENT-GATE-REFUSES-NIGHT-FROM-COADD-AND-SEARCH
    JUPITER_MASK_DEEP = False                 # R-DEEP-MASTER-JUPITER-MASK-DEFERRED (ledger 1787): option kept, off
    ARCHIVE_WRITE = False                     # P3_ARCHIVE_WRITE=0: the co-add never publishes; archive is its own step (rebuild_cell.sh:26)
    FORCE_REBUILD = True                      # P3_FORCE_REBUILD=1 (rebuild_cell.sh:26): the .done guard never skips a build (FIX-STALE-DONE-SKIPGUARD-BLOCKED-MASTER-REBUILD)
    KEEP_PREP = False                         # P3_KEEP_PREP unset (coadd_full_pipeline.py:4029)
    PROTECTED_KEYS = ('ALLOW_STALE_FLOW', 'FLUXSCALE_TILT', 'SOLAR_AUDIT', 'SOLAR_AUDIT_SHA1', 'NAMED_OUT_OF_MASTER',
                      'NAMED_OUT_OF_MASTER_SHA1', 'ARCHIVE_WRITE', 'VARIANT_BAND_MEDIANS', 'VARIANT_BAND_MEDIANS_SHA1',
                      'FLOW_MAX_NAN_FRACTION', 'WRONG_GRID_NIGHTS',
                      'REF_ASTROM_PIVOT_MHZ', 'REF_ASTROM_EXPONENT', 'REF_ASTROM_MIN_ANCHORS', 'REF_ASTROM_N_PEAKS',
                      'REF_ASTROM_MATCH_MAX_ARCMIN', 'FLUXSCALE_ESTIMATOR', 'FLUXSCALE_SNR_MIN', 'FLUXSCALE_PEAK_JY',
                      'FLUXSCALE_MIN_ELEV_DEG', 'FLUXSCALE_APERTURE_BEAMS', 'FLUXSCALE_MAXFILTER_BEAMS',
                      'FLUXSCALE_COMPACTNESS', 'FLUXSCALE_MIN_SOURCES', 'FLUXSCALE_IRLS', 'DEGENERACY', 'SECOND_PASS',
                      'SAFE_SCALE_FLOOR', 'STAGE2_MARGIN_FRAC', 'STAGE2_CLIP_SIGMA', 'MAD_MIN_VALUES', 'BEAM_GATE_RATIO',
                      'BEAM_GATE_MIN_AREAS', 'BEAM_GATE_IMAGE', 'PROVISIONAL_SIGCLIP', 'LEGACY_STAGE2_RECORD',
                      'VARIANT_MAX_ABS_JY', 'VARIANT_MIN_FINITE', 'VARIANT_REL_K', 'VARIANT_FLUXDC_LEGACY', 'COADDOP',
                      'FLUXFIT', 'LOO_N_RANGE', 'STRIATION_WINDOW_BEAMS', 'EPOCH_CHECK', 'JUPITER_MASK_DEEP',
                      'FORCE_REBUILD')   # FORCE_REBUILD: False would re-enable the in-place .done skip guard (CLAUDE.md 3; the wrapper refuses a skip)
    RULED = {'CUTOVER_DATE': 'R-WRONG-GRID-NIGHTS-LEAVE-EXCLUDED',
             'INPUT_RESOLVER': 'R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32',
             'PRODUCT_GRID_MISMATCH': 'R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32',
             'OFF_GRID_SURVEY': 'R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32',
             'WRONG_GRID_NIGHTS': ('R-WRONG-GRID-NIGHTS-LEAVE-EXCLUDED', 'R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32'),
             'RAW_INPUT_FLOW_RULE': ('R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32', 'R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS'),
             'EPOCH_GATE_SOURCE': 'R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32',
             'EPOCH_GATE_MAX_PASSES': ('R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32', 'R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS'),
             'MASTER_SPAN_SOURCE': 'R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS',
             'FLOW_MAX_NAN_FRACTION': 'FINDING-FIVE-GOOD-NIGHTS-BAD-RESIDUALS-ARE-FLOW-FAILURES',
             'ALLOW_STALE_FLOW': 'RATIFY-COADD-FLOW-REFRESH-STAGE-AND-FRESHNESS-GATE',
             'REF_MODE': 'R-ALIGNMENT-REFERENCE-IS-PREVIOUS-GENERATION-MASTER',
             'REF_ASTROM_MAX_ARCMIN': 'A-02-rev', 'REF_ASTROM_MIN_ANCHORS': 'A-02-rev',
             'FLUXSCALE_ESTIMATOR': 'IMPL-COADD-FLUX-SCALE-ESTIMATOR-AT-PEAK-PIXEL',
             'FLUXSCALE_MIN_SOURCES': 'R-FLUX-CORRECT-EVERY-ADMITTED-NIGHT',
             'FLUXSCALE_TILT': 'R-TILT-REMOVED-NEXTGEN',
             'STAGE2_METRIC': 'CORRECTION-STAGE2-PSFMATCH-REINTRODUCED-20261007-REMOVED',
             'OUTLIER_SIGMA': 'C-03', 'LOW_OUTLIER_SIGMA': ('R-LOW-OUTLIER-4SIGMA', 'R-GC5-ASYMMETRIC-GATE-WANTED'),
             'BEAM_GATE_RATIO': 'R-BEAM-GATE-APPROVED-AND-SHIPPED',
             'BEAM_GATE_MIN_AREAS': 'R-GREGG-20261014-REFACTOR-DECISIONS-LIST-RECOMMENDATIONS',
             'BEAM_GATE_IMAGE': 'IMPL-D1-COADD-BEAM-GATE-READS-COADD-IMAGE-AND-REBUILD-ROUND-20261013',
             'SOLAR_AUDIT': 'R-SUNLIT-NIGHTS-OUT-OF-MASTERS',
             'NAMED_OUT_OF_MASTER': 'R-GREGG-20261011-17H18-0505-OPTION-B',
             'QA3_HI': 'R-QA3-BOUNDS-PROCESS-APPROVED', 'QA3_LO': 'R-QA3-BOUNDS-PROCESS-APPROVED',
             'NIGHT_GATE': 'R-NIGHT-IMAGE-GATE-BEFORE-BOTH-GATES',
             'MIN_DATES_FOR_VARIANT': 'R-GREGG-20261012-VARIANTS-FROM-2-NIGHTS-23HGREEN',
             'VARIANT_MAX_ABS_JY': 'R-NONSURVEY-VARIANT-RECOVERY-CAMPAIGN-GO',
             'VARIANT_REL_K': 'R-VARIANT-GATE-3X-WITHIN-CELL',
             'COADDOP': 'LANDED-FLAT-DC-FLUX-SCALE-V3',
             'LOO_N_RANGE': ('R-LOO-THRESHOLD-CORRECTED', 'R-LOO-PER-STOKES-RATIFIED'),
             'STRIATION_WINDOW_BEAMS': 'R-STRIATION-WINDOW-APPROVED-8BEAMS',
             'EPOCH_CHECK': 'R-EPOCH-CONTENT-GATE-REFUSES-NIGHT-FROM-COADD-AND-SEARCH',
             'JUPITER_MASK_DEEP': 'R-DEEP-MASTER-JUPITER-MASK-DEFERRED',
             'FORCE_REBUILD': 'FIX-STALE-DONE-SKIPGUARD-BLOCKED-MASTER-REBUILD'}


class masters_pbmaps(_Section):
    """deep pbmaps: PB_deep_<scope> (Robust 0 beam) and PB_deep_R075_<scope> (survey beam) + the pbcorr sibling of every
    sane variant (in place: pbcorr_masters.do_cell; pb_apply.wideband_pb / apply_pbcorr)."""
    PB_LIBRARY = '/lustre/pipeline/exopipe/phase3/pb_library'   # 105 maps: PB_<sb>_bin0-5 + PB_<sb>_deep (pb_apply.py:31; FIX-PB-LIBRARY-DEFAULT-WAS-NODE-LOCAL)
    PB_THREE_FACTOR = True                    # model x chromatic x residual (pb_apply.py:59; LANDED-THREE-FACTOR-PB-MODEL-CHROMATIC-RESIDUAL)
    PB_CHAN_AVG_HZ = 96000.0                  # Hz, channel averaging of the chromatic term (pb_apply.py:57)
    PB_CHROM_VERSION = 'bs-sqbp-v1'           # pb_apply.py:58
    PB_RESIDUAL_V2_TABLE = '/lustre/pipeline/exopipe/phase3/pb_library/residual_pb/residual_pb_v2_perband.csv'   # pb_apply.py:60
    # PB night weights come from paths.PHASE2_ROOT/<lst>/<date>/Science_*/QA/<date>_<lst>_thermal_noise.csv (pb_apply.QA_ROOT,
    # pb_apply.py:138, 316-320 - a literal checked equal to paths.PHASE2_ROOT by the step)
    PB_FLOOR = 0.1                            # PB below = NaN in a sibling (pb_apply.py:139)
    PBLIBV = 'v1-2026-07-04'                  # card (phase3_pbcorr_masters.py:79; pb_apply.py:419, 495-496)
    BEAM_MODEL = 'OVRO-LWA_MROsoil_updatedheight.h5'   # card BEAMMOD (phase3_pbcorr_masters.py:41)
    PB_VARIANTS = ('I_deep_Taper_R0', 'I_deep_Taper_R075', 'I_deep_NoTaper_R0', 'I_deep_NoTaper_R075',
                   'V_deep_Taper_R0', 'V_deep_Taper_R0_clean')   # phase3_pbcorr_masters.py:35-40 (IMPL-VCLEAN-PBCORR-SIBLINGS-STALE-PB)
    PB_MAP_NAMES = {'R075': 'PB_deep_R075_{scope}.fits', 'R0': 'PB_deep_{scope}.fits'}   # pb_apply.py:142-149 (R-FLUX-SCALE-REMEASURE-REBAKE-AND-PB-STEP2-RUNNING)
    CORRUPT_MAX_JY = 1e6                      # Jy/beam; a variant above is skipped (phase3_pbcorr_masters.py:63)
    SIBLING_TOL = dict(rel=1e-5, abs=1e-9, floor_tie=1e-6)   # sibling x PB = plain where PB > floor (pb_apply.py:455-473)
    SIBLINGS_AFTER_PBMAPS = 5                 # 5 variants before vclean; 6 in the scaled generation (check_rebuild.py:88)
    PB_VALUE_RANGE = (-0.01, 1.001)           # content test of every PB map: measured 2026-10-13 min -0.0058 (23h Green PB_deep), max 0.9938-0.9999 (library and deep maps)
    PB_QA_MISSING_MAX_FRAC = 0.25             # a WIDEBAND PB sibling missing the Phase-2 noise file of more than this fraction of its nights refuses
                                              # (deep_stages.pb_qa_gate; every missing night counted). PROPOSED 2026-10-14 by the refactor chat -
                                              # Gregg to confirm (not in RULED until he does); production 1.0 = no limit. Census 2026-10-14: no
                                              # wideband master misses a night; 4 subband masters (12h 69-82 MHz) miss 1-2 (<= 7.7%)
    PROTECTED_KEYS = ('PB_THREE_FACTOR', 'PB_CHROM_VERSION', 'PB_FLOOR', 'PBLIBV', 'PB_MAP_NAMES',
                      'CORRUPT_MAX_JY', 'SIBLING_TOL')
    RULED = {'PB_LIBRARY': 'FIX-PB-LIBRARY-DEFAULT-WAS-NODE-LOCAL',
             'PB_THREE_FACTOR': 'LANDED-THREE-FACTOR-PB-MODEL-CHROMATIC-RESIDUAL',
             'PB_MAP_NAMES': 'R-FLUX-SCALE-REMEASURE-REBAKE-AND-PB-STEP2-RUNNING',
             'PB_VARIANTS': 'IMPL-VCLEAN-PBCORR-SIBLINGS-STALE-PB'}


class masters_vclean(_Section):
    """deep vclean: cleaned Stokes V master + sibling (in place: vclean_masters.build_vclean_cell -> search_v.py --clean_only)."""
    TIMEOUT_S = 7200                          # s, engine subprocess (vclean_masters.py:60)
    THREADS = 2                               # OMP/MKL/OpenBLAS threads of the engine (vclean_masters.py:156)
    SANE_MAX_JY = 1e6                         # Jy/beam (vclean_masters.py:63-73)
    I_MASTER_PREFERENCE = {'band': ('Taper_R0', 'Taper_R075'), 'subband': ('Taper_R075', 'Taper_R0')}   # vclean_masters.py:75-84 (2026-07-11; R-ALIGNMENT-REF-R075-EVERYWHERE not applied here: B-50)
    ENGINE_ELEV_DEG = (20.0, 19.5)            # the clean engine's own elevation floors (B-50 open value; search_v.py)
    ENGINE_ATEAM_PREBLANK = True              # the clean engine still pre-blanks A-team (B-50 open value)
    PROTECTED_KEYS = ('THREADS', 'SANE_MAX_JY', 'I_MASTER_PREFERENCE', 'ENGINE_ELEV_DEG', 'ENGINE_ATEAM_PREBLANK')   # mirrors
    RULED = {'TIMEOUT_S': 'gc-pre-4-complete'}


class masters_fluxscale(_Section):
    """deep fluxscale: pixels x SHSCALE[LST] x FLUXCORR[scope] into gen_shflux/ (in place: shflux.apply)."""
    SCALE_TABLE = '/lustre/gh/main/exopipe_ops/pbstep2_20261001/vlssr_scale_table_v4_20261001.csv'   # Lustre copy; sha1 equal to the lwacalim07 /fast copy rebuild_cell.sh:144 reads (checked 2026-10-13)
    SCALE_TABLE_SHA1 = 'c854a9e548a1fcb45941f27e4f14bebf89da0606'   # R-FLUX-SCALE-APPLY-VLSSR-HIGH-ELEVATION-APPROVED; re-measured 2026-10-01
    FLUXCORR_TABLE = '/lustre/gh/main/exopipe_ops/pbstep2_20261001/flux_correction_v2_20261001.csv'   # rebuild_cell.sh:145; R-FLUX-CORRECTION-BAKED-INTO-PIXELS
    FLUXCORR_TABLE_SHA1 = '26640f8595ae8fea54aa8abdb8e1a0c1760a9b20'
    REQUIRED_FLUXVER = 'fluxcorr-v2-20261001' # membership.py:251
    OUTGEN = 'gen_shflux'                     # rebuild_cell.sh:145
    TOKEN = '_shflux'                         # name token of the scaled generation (promote_rebuild.py:12; check_rebuild.py:37-38; literal in shflux._scaled_name -> protected)
    SCALE_MODEL_FREQ_HZ = 75.41e6             # Hz, SHFREQ of the table path (phase3_master_shflux.py:345)
    SCALED_DIRS = ('coadd_final', 'subtracted')   # phase3_master_shflux.py:49
    UNSCALED_PREFIX = ('PB_deep',)            # PB maps copied, never scaled (phase3_master_shflux.py:51)
    CARD_TOL = dict(SHSCALE=5e-6, FLUXCORR=5e-7, s_old=5e-5)   # phase3_master_shflux.py:234-240; check_rebuild.py:95-96
    PROTECTED_KEYS = ('SCALE_TABLE', 'SCALE_TABLE_SHA1', 'FLUXCORR_TABLE', 'FLUXCORR_TABLE_SHA1', 'REQUIRED_FLUXVER',
                      'TOKEN')
    RULED = {'SCALE_TABLE': 'R-FLUX-SCALE-APPLY-VLSSR-HIGH-ELEVATION-APPROVED',
             'FLUXCORR_TABLE': 'R-FLUX-CORRECTION-BAKED-INTO-PIXELS'}


class masters_verify(_Section):
    """deep verify: content tests T1-T8 of the scaled generation (port of ops/membership_rebuild_20261007/check_rebuild.py)."""
    T1_TOL = 0.01                             # |median(new / (ref x v4 x v2)) - 1| (check_rebuild.py:59)
    T1_PEAKS = dict(snr=20.0, max_jy=50.0, box_px=9, radius=0.9)   # check_rebuild.py:55-57
    T3_STOKES = ('I', 'V')                    # today check_rebuild.py:70 tests I only (review item 17); V added as a pure check (spec 6)
    T4_SIBLINGS = 6                           # check_rebuild.py:88
    T8_IGNORE = ('RBF master dewarp',)        # verify_products violations ignored before the dewarp (check_rebuild.py:119)
    DEWARP_SCREEN_RECORD = '/lustre/pipeline/exopipe/phase3/mdewarp_screens/Run_20260909_000000/mdewarp_screens.csv'   # membership.py:255; CORRECTION-11-ACCEPTED-CELLS-NOT-DEWARPED-WAS-WRONG
    DEWARP_SCREEN_RECORD_SHA1 = '3f42ba7f84cab12b9cacd81520edee9cb3672331'   # 2026-10-13; it changes at every dewarp install (re-pin per run: the record stores the digest read)
    DEWARP_INTERP = 'cubic-spline'            # membership.py:256; R-MDEWARP-CUBIC-REAPPLY
    INTEGRITY = dict(max_abs_jy=1e5, min_finite=0.5)   # every master (ops/verify_master_generation.py:35, 60-98; LANDED-VERIFIER-ALL-VARIANTS-AND-VCLEAN)
    PROTECTED_KEYS = ('DEWARP_SCREEN_RECORD',)
    RULED = {'DEWARP_SCREEN_RECORD': 'CORRECTION-11-ACCEPTED-CELLS-NOT-DEWARPED-WAS-WRONG',
             'DEWARP_INTERP': 'R-MDEWARP-CUBIC-REAPPLY',
             'INTEGRITY': 'LANDED-VERIFIER-ALL-VARIANTS-AND-VCLEAN'}


class masters_sources(_Section):
    """sources: master source (parent) catalogue, PyBDSF (in place: ops/p3s5_parents.one deep; ops/sublst_parent_catalogs.py
    sub-LST). Shared by the deep and sub-LST tiers (per-tier dicts where they differ)."""
    THRESH_ISL = 4.0                          # sigma (p3s5_parents.py:41; sublst_parent_catalogs.py:33)
    THRESH_PIX = 5.0                          # sigma (same lines)
    CATALOG_TYPE = 'gaul'
    BEAM = {'deep': 'empirical', 'sublst': 'header'}   # deep: phase3_deep.fit_empirical_beam (p3s5_parents.py:34); sub-LST: BMAJ/CDELT1 (sublst_parent_catalogs.py:32)
    BEAM_FALLBACK_PX = 7.0                    # px, deep (p3s5_parents.py:35-36)
    BDSF_NCORES = 2                           # install_cell.sh:26
    DEEP_COLLECT = '/lustre/gh/main/phase3_collect/p3s5_collect_lwacalim07'   # P3S5_COLLECT (p3s5_parents.py:21); production root (masters.EXTRA_PRODUCTION_ROOTS): parity runs write <out>/parent_catalogs/
    SUBLST_NPROC = 8                          # processes per LST as production passes it (sublst_catalog_publish.sh:65; script default 6, sublst_parent_catalogs.py:15)
    REQUIRE_DEWARP_SCREEN = True              # DESIGN 11.2: refuse when the installed master lacks the screen its record says applied (consistency with install; pure)
    PROTECTED_KEYS = ('THRESH_ISL', 'THRESH_PIX', 'CATALOG_TYPE', 'BEAM', 'BEAM_FALLBACK_PX')   # mirrors (p3s5_parents.py, sublst_parent_catalogs.py literals)
    RULED = {'THRESH_ISL': 'FINDING-10MIN-PARENT-CATALOGUES-PREDATE-REBAKE-REBUILT',
             'THRESH_PIX': 'FINDING-10MIN-PARENT-CATALOGUES-PREDATE-REBAKE-REBUILT'}


# =====================================================================================================================
# unit masters-tiers (was exopipe/phase3/_cfgdraft_masters-tiers.py)
# Configuration of `exopipe3 masters --tier sublst|snapshot` (unit masters-tiers).
#
# Source of every value: plan/specs/masters.md section 3 (classes masters_sublst, masters_snapshot), adapted to the DECISIONS OF
# THE REFACTOR CHAT (plan/specs/_CROSSCHECK.md, 2026-10-13 ~04:30 UTC): shared roots come from CFG.paths (SUBLST_ROOT,
# SNAPSHOT_CUBES_ROOT, COADD_SB_ROOT, TRANSIENTS_ROOT, PHASE3_ARCHIVE_ROOT, OPS_ROOT) - never a key of this draft; interpreters
# are site PYTHON keys default/flow/py36/casa/tess only; protection by PROTECTED_KEYS / RULED; sub-LST wideband and content
# test (b) belong to frames, sub-LST loo and sources to masters; the per-hour sub-LST night stacks have ONE wrapper
# (lib/masters/tier_sublst.stack_night) and ONE section (masters_sublst) that frames imports.
#
# Values are today's production values with the production file:line they come from. Line numbers cite clv15 39bdfd3 and
# exopipe_ops 640f677 (2026-10-13):
#   stack = clv15/phase3_sublst_stack.py     loo = clv15/phase3_sublst_loo.py     rules = clv15/phase3_tier_rules.py
#   jupmask = clv15/phase3_jupiter_mask.py   nocat = exopipe_ops/sublst_run_lst_nocat.sh (the admit heredoc is nocat:82-186)
#   prep = exopipe_ops/sublst_prep_lst.sh    catpub = exopipe_ops/sublst_catalog_publish.sh
#   freeze = exopipe_ops/snapshot_masters_freeze.py   snapsearch = clv15/phase3_snapshot_search.py
#   install = exopipe/phase3/phase3_install.py (main checkout)
# "Ledger" IDs are fix_ids of clv15/fix_ledger.csv (checked by tests/test_masters-tiers.py with core.ledger.has_id).
#
# Values these tiers read from the step owner's draft (config/steps/masters.py), NOT repeated here: masters.STAGES /
# masters.DEFAULT_STAGES (stage order), masters.PYTHON_SITE_KEY (engine interpreters), masters_sources.SUBLST_NPROC (processes of sublst_parent_catalogs.py, catpub:65),
# masters_pbmaps.PB_LIBRARY (the model-beam PB library, one name for both tiers).
#
# PROTECTED_KEYS holds two kinds of key, both refused by the config machinery on --set / --config (exit 2):
#   pins     frozen, checksummed inputs a ruling names (tier-rule pins, the frozen record and master trees, the manifest sha1)
#   mirrors  values that are LITERALS inside the in-place clv15 / exopipe_ops code (no argument carries them): an override
#            could be recorded but never applied, so it is refused until the module is absorbed into lib/ and reads config.
#            The tier refuses to run when an in-place literal differs from its mirror (the source text is read:
#            lib/masters/tier_sublst.MIRRORS, tier_snapshot.MIRRORS).
# =====================================================================================================================

class masters_sublst(_Section):
    """masters --tier sublst (10-minute windows; tier C): prep, admit, stack, loo, sources (wideband: frames). In place:
    clv15 phase3_sublst_{stack,loo}.py and exopipe_ops sublst_parent_catalogs.py run by core.legacy.run with the production
    command lines; the shell logic of sublst_prep_lst.sh, the admit heredoc and gates of sublst_run_lst_nocat.sh and the pcheck
    of sublst_catalog_publish.sh ported into lib/masters/tier_sublst.py. frames --tier sublst (per hour) imports
    tier_sublst.stack_night and reads THIS section for the night's window stacks."""
    # ---- input locations below the shared roots (paths.SUBLST_ROOT, paths.TRANSIENTS_ROOT) -------------------------------
    FROZEN_RECORDS_DIR = 'deep_coadd_records_frozen_20261008'   # <paths.SUBLST_ROOT>/<this>/<LST>/<sb>/weights.csv: frozen deep
                                                                # co-add records (stack:71; loo:36; rules:45; nocat:151); ruling B 2026-10-08
    DEEPFLOW_CUBES_DIR = 'cubes_deepflow_18_23'   # <paths.SUBLST_ROOT>/<this>: deep-Master re-flow cubes, 19h 18/23 MHz (nocat:76, :93)
    PREP_PARENTS_RUN = 'Run_P3S4'             # <paths.TRANSIENTS_ROOT>/<this>/parent_catalogs: footprint fallback and parent copies
                                              # (prep:43, :50); B-36 default (c)
    PREP_FOOTPRINT_DIR = '/fast/gh/fp_artifact_investigation/support/bdsf_catalogs_aug'   # August footprint catalogues (prep:40);
                                              # lwacalim07-local; B-36 default (c)
    # The frozen generation-4 masters the cubes were flowed to are masters_snapshot.FROZEN_ROOT (stack:74 MASTER_PATH; prep:29).
    # ---- pins read by the in-place tier rules (rules:37-55) ---------------------------------------------------------------
    MEMBERSHIP_VETO_PIN_MD5 = '851b16d86fe059c8c8bf43b1afa72344'   # md5 of <frozen records>/membership_vetoes_pinned.csv, 127 rows (rules:55)
    DEEP_SEARCH_PIN = '/lustre/gh/main/phase3/deep_search_refusals/Run_P3S5_pin_20261008T025911Z/deep_search_pin.csv'   # tier rule 3 pin (rules:37)
    DEEP_SEARCH_PIN_MD5 = '29a53b8355146311b2b05954e9478a98'      # (rules:38)
    # ---- the stacker (stack) and the admit predictor (nocat:82-186) ------------------------------------------------------
    WINDOW_MIN = 10.0                         # min, sidereal window length (stack:197 --window-min default); a mirror because
                                              # loo:82 hard-codes six windows (spec section 3 lists it as an argument)
    N_WINDOWS = 6                             # windows per LST hour (loo:82)
    BEAM_AREA_MAX = 1.5                       # x the night's median beam area, both Stokes (stack:64)
    NOISE_K = 5.0                             # robust sigma, per-Stokes noise outlier (stack:65; Gregg 2026-09-09)
    BLANK_FINITE_MIN = 0.5                    # finite-pixel fraction below = blank frame (stack:66)
    QA3_ADMIT_FLOOR = 0.1                     # deep record qa3_scale must exceed this (stack:232; nocat:160) - admission only since Q10
    TIER_RULE4_RANGE = (0.33, 3.0)            # tier rule 4: degenerate-flatDC with qa3_scale outside -> refused (rules:151); B-21 default (a)
    MIN_USABLE_FRAMES = 10                    # frames passing the blank, beam and night gates (stack:353; nocat:181)
    FLUXDC_WAIT_S = 14400                     # s, a V job waits this long for its night's I flux_dc record (stack:279, env
                                              # SUBLST_FLUXDC_WAIT_S); the unit runs I before V, so the wait never triggers (B-76 (b))
    PRODUCT_CARD = 'sublst-v0-test'           # PRODUCT card of every window (stack:450); B-29 default (b)
    # s, one 10-s integration as the stacker and the admit heredoc time frames (stack:77; nocat:94: frame time = DATE-OBS + this/2).
    # NOT instrument.INTEGRATION_S (10.031 s = phase2_config.LWA_DT): the two differ by 2.0e-5 s; the admit port uses this value
    # so its frame times (rounded to 0.1 s, floored to the second for the night rule) equal the stacker's (impl note, finding F1)
    INTEGRATION_S = 10.03102041
    # ---- window masters + leave-one-out (loo) ------------------------------------------------------------------------------
    LOO_MIN_WEIGHTED_NIGHTS = 2               # a window Master needs >= 2 nights and >= 2 weighted nights (loo:93; Gregg 2026-10-06)
    PER_NIGHT_MIN = 2                         # a residual's own LOO master needs >= 2 weighted nights (loo:140; Q11 2026-10-07)
    WINDOW_COADDOP = 'sublst-ivw-window-mean-v2-jupmask'   # COADDOP of every window Master (loo:122)
    JUPITER_MASK_RADIUS_BEAMS = 2.0           # x BMAJ, moving Jupiter mask per night in the Masters (jupmask:14)
    # The loo stage starts only when every job of the LST's admit plan has a stack record. Production stops the LST when ANY job
    # lacks its completion line after two catch-up passes - a refused stack job included (nocat:208-218: 'REFUSING: N stack jobs
    # without a completion line'). False = that behaviour. True = the spec's G-S05 reading ('every planned stack job done OR
    # refused'): window Masters are then built without the refused (night, subband). Open question for Gregg (impl note Q2).
    LOO_ADMITS_REFUSED_STACKS = False
    # ---- prep (port of sublst_prep_lst.sh) ---------------------------------------------------------------------------------
    PREP_WIDEBAND = ('Red', 'Green', 'Blue')  # wideband groups whose masters prep copies (SUBLST_PREP_WIDEBAND default, prep:22)
    PREP_FILES = ('I_deep_Taper_R0_{b}.fits', 'I_deep_Taper_R075_{b}.fits', 'V_deep_Taper_R0_{b}.fits', 'PB_deep_{b}.fits')   # from the frozen masters (prep:30)
    PREP_MIN_PARENTS = 15                     # parent catalogue files copied per LST, fewer refuses (prep:51)
    WEIGHTS_FOR_SEARCH = 'live-at-prep'       # B-20 default (b): prep copies the LIVE weights.csv / residuals_provisional.csv from
                                              # paths.COADD_SB_ROOT (prep:32-37); the only value implemented ((a) 'frozen' needs Gregg)
    # ---- orchestration -----------------------------------------------------------------------------------------------------
    # Interpreters of the in-place engines: masters.PYTHON_SITE_KEY['sublst-stack'|'sublst-loo'|'sources'] (step owner's section,
    # py36 / py36 / default = sublst_stack_pass.sh:10, nocat:28, catpub:21) - one name per value; this section adds test (a)'s:
    TEST_WINDOWS_PYTHON = 'py36'              # site PYTHON key of sublst_reserve_test_windows.py (catpub:21 PYD, :87)
    STACK_JOB_UNIT = 'two-jobs-I-then-V'      # B-76 default (b): the I and V stacker invocations stay separate; I runs first
    PARALLEL = dict(stack=4, loo=8)           # units at once in one process when --workers is not given
                                              # (sublst_stack_pass.sh:8 NPAR; nocat:230 xargs -P 8)
    # ---- content tests (lib/masters/tests_sublst.py) ----------------------------------------------------------------------
    # Production content test (a) (catpub:87): windows = frames x stated factors on a seeded sample of n (night, subband) cells.
    WINDOW_SAMPLE_TEST = dict(n=3, seed='')   # sublst_reserve_test_windows.py --n / --seed defaults (its :23)
    # Per-file tolerances. Window ones are those of test (a) (sublst_reserve_test_windows.py:67-68, :98, :112-115): frame time
    # 1e-3 s, MJD-MID 1e-8 d, pixel factor vs the Master's 1e-9, FLUXDC vs the I record 1e-12; noise_rel: NOISEMED vs
    # median(admission noise_jyb) / FLUXDC x B (the CSV keeps 6 significant digits). The LOO identity (new, pure): residual =
    # stack - (M W - w a) / (W - w) on pixels finite in stack and Master, within tol_rms x the residual's robust rms, on at
    # least frac_min of them (Jupiter-masked pixels differ by construction; measured on 09h 18 MHz: impl note).
    WINDOW_TEST_TOL = dict(time_s=1e-3, mjd_d=1e-8, factor=1e-9, fluxdc=1e-12, noise_rel=1e-5)
    LOO_IDENTITY = dict(frac_min=0.99, tol_rms=1e-3)
    PROTECTED_KEYS = (
        # pins
        'FROZEN_RECORDS_DIR', 'MEMBERSHIP_VETO_PIN_MD5', 'DEEP_SEARCH_PIN', 'DEEP_SEARCH_PIN_MD5',
        # mirrors of literals in the in-place code
        'WINDOW_MIN', 'N_WINDOWS', 'BEAM_AREA_MAX', 'NOISE_K', 'BLANK_FINITE_MIN', 'QA3_ADMIT_FLOOR', 'TIER_RULE4_RANGE',
        'MIN_USABLE_FRAMES', 'PRODUCT_CARD', 'INTEGRATION_S', 'LOO_MIN_WEIGHTED_NIGHTS', 'PER_NIGHT_MIN', 'WINDOW_COADDOP',
        'JUPITER_MASK_RADIUS_BEAMS')
    RULED = {'FROZEN_RECORDS_DIR': 'IMPLEMENTED-RULINGS-A-B-20261008-SUBLST-FLOWREF-AND-MEMBERSHIP-VETOES',
             'MEMBERSHIP_VETO_PIN_MD5': 'IMPLEMENTED-RULINGS-A-B-20261008-SUBLST-FLOWREF-AND-MEMBERSHIP-VETOES',
             'DEEP_SEARCH_PIN': 'RULINGS-GREGG-20261008-TIER3-FREEZE-B2-RECHAIN-LABELS',
             'DEEP_SEARCH_PIN_MD5': 'RULINGS-GREGG-20261008-TIER3-FREEZE-B2-RECHAIN-LABELS',
             'TIER_RULE4_RANGE': 'RULING-TIER-RULE-4-FAILED-FLUX-SCALE',
             'MIN_USABLE_FRAMES': 'IMPLEMENTED-SUBLST-JOB-BUILDER-RULE-REFUSALS-AND-DEEPFLOW-CUBES-20261011',
             'DEEPFLOW_CUBES_DIR': 'IMPLEMENTED-SUBLST-JOB-BUILDER-RULE-REFUSALS-AND-DEEPFLOW-CUBES-20261011',
             'PER_NIGHT_MIN': 'IMPLEMENTED-Q11-PER-NIGHT-LEAVE-ONE-OUT-RULE-AND-COUNTS',
             'JUPITER_MASK_RADIUS_BEAMS': 'RULING-MOVING-JUPITER-MASK-IN-MASTERS',
             'PRODUCT_CARD': 'IMPLEMENTED-Q10-DEEP-FLUX-SCALE-BAKED-AND-Q1-FRAME-TIME-IN-10MIN-STACKS',
             'INTEGRATION_S': 'IMPLEMENTED-Q10-DEEP-FLUX-SCALE-BAKED-AND-Q1-FRAME-TIME-IN-10MIN-STACKS'}


class masters_snapshot(_Section):
    """masters --tier snapshot (tier D): no master is BUILT (contract S-1.3a). verify-frozen checks the frozen generation-4
    masters (md5 against MANIFEST.csv), the 912 links the snapshot tier registers frames through, and the model-beam PB
    library the tier reads (masters_pbmaps.PB_LIBRARY; RULING-SNAPSHOT-TIER-PB-CORRECTION-MODEL-BEAM-ONLY). freeze is
    plan-only in the window. The registry is written to <out>/snapshot/ (default out paths.PHASE3_ARCHIVE_ROOT, a production
    root: a real run needs --out while paths.PRODUCTION_WRITES is False)."""
    LINK_ROOT = '/lustre/pipeline/exopipe/phase3/snapshot_masters'      # 456 slots x 2 links (freeze:21 LINKS; P3_SNAP_MASTERS)
    FROZEN_ROOT = '/lustre/pipeline/exopipe/phase3/snapshot_masters_gen4'   # freeze:23 FROZEN; stack:74; FREEZE-SNAPSHOT-MASTERS-GEN4
    FROZEN_MANIFEST_SHA1 = '19c6f36f512949d186ba4bccec1e7f41e82ecd6e'  # sha1 of <FROZEN_ROOT>/MANIFEST.csv, 1,824 rows + header (read 2026-10-13)
    N_SLOTS = 456                             # (LST, scope) slots: 24 LSTs x 19 scopes
    N_FROZEN_FILES = 1824                     # I/V Taper R0, I Taper R-0.75, PB_deep per slot (freeze:47-57)
    FROZEN_FILES = ('I_deep_Taper_R0_{b}.fits', 'V_deep_Taper_R0_{b}.fits', 'I_deep_Taper_R075_{b}.fits', 'PB_deep_{b}.fits')   # freeze:47-57
    LINKED_FILES = ('I_deep_Taper_R0_{b}.fits', 'V_deep_Taper_R0_{b}.fits')   # the registration links per slot (freeze:47-48)
    FORBIDDEN_LINK_TARGET = 'Coadd/'          # below paths.PHASE3_ARCHIVE_ROOT: no link may resolve into the deep archive (install:203-207 G-1b)
    PB_MODEL_FILES = ('PB_{sb}_bin{k}.fits', 'PB_{sb}_deep.fits')   # in masters_pbmaps.PB_LIBRARY; k = 0 .. PB_MODEL_BINS-1 (snapsearch:954)
    PB_MODEL_BINS = 6                         # 10-minute bins per LST hour (snapsearch:79 N_TIME_BINS)
    PB_BEAM_MODEL = 'OVRO-LWA_MROsoil_updatedheight.h5'   # PBBEAM card of every library map (model beam only, ruling 1231)
    PB_LIBRARY_VALUE_RANGE = (-0.01, 1.001)   # content test of every library map and window PB; measured 2026-10-13 over the 105 maps:
                                              # 0.0 .. 0.99987, all finite (the bounds of masters_pbmaps.PB_VALUE_RANGE)
    FREEZE_ALLOWED = False                    # the freeze stage writes production (freeze:66-105); plan only in the window
    MD5_WORKERS = 8                           # files hashed at once in verify-frozen (Lustre reads; operational, no product effect)
    # ---- snapshot bright-source PARENT catalogues (M21, exopipe_ops a670970 / 3ff6565; review 2026-10-13) -----------------
    # One recipe for all 456 (LST, scope) pairs: exopipe_ops/snapshot_parentcat_contract.py (pc) in place - the deep search's
    # builder functions (phase3_deep._load, phase3_tenmin._det_header, bdsf.process_image) on the frozen gen-4 R075 masters,
    # then snapshot_parentcat_sanitize + the .gen.json sidecar. Class snapshot_parent_catalog (tests_snapshot.py).
    PARENT_CATALOG_DIR = '/lustre/gh/main/exopipe_ops/snapshot_master_catalogs'   # the installed set the snapshot chain reads
                                              # (3ff6565; also forced_snapshot.MASTER_CATS, curves.MASTER_CATALOG_DIR,
                                              # catalog_snapshot.MASTER_CATALOGS name this root - one name at fold-in)
    PARENT_MASTER_FILE = 'I_deep_Taper_R075_{b}.fits'   # the parent master of every pair, in FROZEN_ROOT (pc:49; R-CAT5)
    PARENT_THRESH_PIX = 7.0                   # sigma (pc:34 THRESH_PIX; R-CAT3)
    PARENT_THRESH_ISL = 4.0                   # sigma (pc:34 THRESH_ISL)
    PARENT_ATROUS = False                     # no atrous module (pc:69 atrous_do)
    PARENT_ZAP_JYB = 1.0e6                    # Jy/beam: phase3_deep._load sets |x| above this and non-finite pixels to 0 before
                                              # the build and the sidecar digest (phase3_deep.py:311-312)
    PARENT_NPROC = 8                          # builder processes (pc:85 --nproc default; operational, no product effect)
    PROTECTED_KEYS = ('LINK_ROOT', 'FROZEN_ROOT', 'FROZEN_MANIFEST_SHA1', 'FREEZE_ALLOWED', 'PB_MODEL_BINS',
                      # mirrors of literals in snapshot_parentcat_contract.py / phase3_deep._load
                      'PARENT_MASTER_FILE', 'PARENT_THRESH_PIX', 'PARENT_THRESH_ISL', 'PARENT_ATROUS', 'PARENT_ZAP_JYB')
    RULED = {'FROZEN_ROOT': 'FREEZE-SNAPSHOT-MASTERS-GEN4',
             'FROZEN_MANIFEST_SHA1': 'FREEZE-SNAPSHOT-MASTERS-GEN4',
             'PB_MODEL_FILES': 'RULING-SNAPSHOT-TIER-PB-CORRECTION-MODEL-BEAM-ONLY',
             'PB_BEAM_MODEL': 'RULING-SNAPSHOT-TIER-PB-CORRECTION-MODEL-BEAM-ONLY',
             'PARENT_THRESH_PIX': 'REBUILT-SNAPSHOT-PARENT-CATALOGUES-ONE-RECIPE-20261013',
             'PARENT_THRESH_ISL': 'REBUILT-SNAPSHOT-PARENT-CATALOGUES-ONE-RECIPE-20261013',
             'PARENT_ATROUS': 'REBUILT-SNAPSHOT-PARENT-CATALOGUES-ONE-RECIPE-20261013',
             'PARENT_MASTER_FILE': 'REBUILT-SNAPSHOT-PARENT-CATALOGUES-ONE-RECIPE-20261013'}
