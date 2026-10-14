"""Shared configuration sections - read by every `exopipe3` step (exopipe/phase3/config/__init__.py lists the step files).

instrument (OVRO-LWA constants, imported from exopipe/common/phase2_config.py), paths (data roots, the production-root guard,
pinned inputs) and PROTECTED, exclusions (rule parameters; data_decisions.csv itself is Gregg's), astrometry and fitting
(the shared position-error and fitting models), runner (multi-node jobs and the named chains), site (filled from the site
file named by EXOPIPE3_SITE). Edit a value here and it applies to every step; per run use --set / --config instead.
"""
import os

from exopipe.phase3.config import _Section


class instrument(_Section):
    """OVRO-LWA instrument constants used by Phase 3 (authoritative values live in common/phase2_config.py;
    architecture 0.1). Imported, not copied, so Phase 2 and Phase 3 cannot disagree."""
    from exopipe.common import phase2_config as _p2
    ALL_SUBBANDS = tuple(_p2.ALL_SUBBANDS)            # subband labels, nominal ('18MHz' ... '82MHz')
    SUBBAND_FREQ_HZ = dict(_p2.SUBBAND_FREQ_HZ)       # Hz, exact subband centres (label -> Hz)
    SUBBAND_BANDS = {k: tuple(v) for k, v in _p2.SUBBAND_BANDS.items()}   # band label -> member subbands
    INTEGRATION_S = _p2.LWA_DT                        # s, one integration
    del _p2


class paths(_Section):
    """Roots of inputs and products. A step's --out / --work flags override the product and scratch roots
    for one run (parity and scratch runs never write into these)."""
    PHASE1_ROOT = '/lustre/pipeline/exopipe/phase1'
    PHASE2_ROOT = '/lustre/pipeline/exopipe/phase2'
    PHASE3_ARCHIVE_ROOT = '/lustre/pipeline/exopipe/phase3'
    DATA_DECISIONS = '/lustre/gh/main/clv15/data_decisions.csv'    # Gregg's exclusions (read only; core/exclusions)
    FIX_LEDGER = '/lustre/gh/main/clv15/fix_ledger.csv'            # Gregg's ruling ledger (read only; --ruling-for IDs)
    REFERENCE_DIR = '/lustre/gh/main/exopipe/reference'            # small publishable reference CSVs
    # Product roots that hold PRODUCTION data. While PRODUCTION_WRITES is False (the refactor window, until each
    # step's parity run passes and Gregg approves its cutover) a step refuses to write under any of them: parity and
    # scratch runs pass --out to their own roots.
    PRODUCTION_ROOTS = ('/lustre/pipeline/exopipe/phase3', '/data07/gh/phase3', '/lustre/gh/main/phase3',
                        '/fast/gh/main/phase3',   # node-local production run roots (deep forced, snapshot forced work)
                        '/lustre/gh/main/phase3_collect',                     # deep search attempts (p3s5 collect)
                        '/lustre/gh/main/exopipe_ops/snapshot_fleet_out',     # snapshot per-cell outputs
                        # production search state (markers, claims, published lists, logs): never written by the branch
                        '/lustre/gh/main/exopipe_ops/p3s5_state', '/lustre/gh/main/exopipe_ops/p3s4_state',
                        '/lustre/gh/main/exopipe_ops/snapshot_batch', '/lustre/gh/main/exopipe_ops/sublst_search_state_A',
                        '/lustre/gh/main/exopipe_ops/sublst_search_state_B')
    PRODUCTION_WRITES = False
    # Config keys ('section.KEY') whose values are PINNED inputs - frozen, checksummed inputs a ruling names (e.g. a
    # frozen master tree, a tier-rule pin). They are protected like the keys in PROTECTED (DESIGN 11.4, 11.6).
    PINNED_INPUTS = ()
    # Canonical shared roots used by several steps (one name each - steps reference these, never their own copy).
    # Production roots of TODAY (no renames in the refactor window).
    COADD_SB_ROOT = '/data07/gh/phase3/coadd_sb'                 # deep working tree <LST>_<scope>/{coadd_final,subtracted,flow}
    TRANSIENTS_ROOT = '/data07/gh/phase3/transients'             # deep / sub-LST run dirs Run_<label>/
    ARCHIVE_RUN = 'Run_20260909_000000'                          # current Phase 3 archive generation under PHASE3_ARCHIVE_ROOT
    SUBLST_ROOT = '/lustre/gh/main/phase3/sublst'                # sub-LST products (coadd, stacks, search_B, code_frozen)
    SNAPSHOT_FLEET_OUT = '/lustre/gh/main/exopipe_ops/snapshot_fleet_out'   # snapshot per-cell outputs (production)
    SNAPSHOT_FLOW_ROOT = '/lustre/pipeline/exopipe/phase3/snapshot_flow'
    SNAPSHOT_CUBES_ROOT = '/lustre/pipeline/exopipe/phase3/snapshot_cubes'
    SNAPSHOT_REVIEW_ROOT = '/data07/gh/phase3/snapshot_search/review'       # served snapshot runs (Run_SNAP*)
    OPS_ROOT = '/lustre/gh/main/exopipe_ops'                     # job records, state files (not products)
    CLV15_ROOT = '/lustre/gh/main/clv15'


# Keys no --set / --config can change (DESIGN 11.4; critique_rulings 4.1): the exclusion file and every exclusion rule
# parameter, the ruling ledger, the production-root guard and the list of pinned inputs (and, through it, each pinned
# input). 'section.*' protects a whole section. Change them in this file, by a commit.
PROTECTED = ('paths.DATA_DECISIONS', 'paths.FIX_LEDGER', 'paths.PRODUCTION_WRITES', 'paths.PRODUCTION_ROOTS',
             'paths.PINNED_INPUTS', 'exclusions.*')


class exclusions(_Section):
    """Exclusion rule parameters (core/exclusions.py is the only reader of data_decisions.csv)."""
    SUBBAND_ONLY = ('18MHz',)      # subbands whose exclusion does NOT propagate to their band (Gregg 2026-06-29)
    PERBAND_TAG = '[PERBAND]'      # reason tag of a per-image row: excludes only its own scope (Gregg 2026-08-26)


class astrometry(_Section):
    """Position-uncertainty model (core/astrometry.py; COADD_ASTROMETRIC_ERROR.md 2026-07-07/08, ratified)."""
    # per-axis systematic scatter (arcmin) by elevation bin (deg) and band class (S3, scaled to the S2b per-axis MAD)
    SYS_ARCMIN = {(75, 90): {'Red': 1.9, 'Green': 1.3, 'Blue': 1.0},
                  (60, 75): {'Red': 1.9, 'Green': 1.4, 'Blue': 1.1},
                  (45, 60): {'Red': 2.2, 'Green': 1.5, 'Blue': 1.1},
                  (0, 45): {'Red': 2.6, 'Green': 1.9, 'Blue': 1.4}}
    SYS_SCALE = {('V', 'Red'): 1.5}      # class scale of the systematic (build_source_catalogs_v3.SYS_SCALE)
    STAT_COEFF = 0.42                    # statistical error = STAT_COEFF x beam FWHM / S/N, per axis (S3b)
    # Association criterion (exopipe_ops/build_source_catalogs_v3.K_ASSOC, whose comment follows verbatim; multiples_fork
    # parses the value out of that file's source text today - this is the value it reads):
    # ---- association criterion (Gregg 2026-09-05, R-ASSOCIATION-RADIUS-QUADRATURE):
    # two detections/sources associate when their separation is below
    #   K_ASSOC x sqrt(stat_i^2 + sys_i^2 + stat_j^2 + sys_j^2)
    # stat = 0.42 x beam FWHM / SNR per detection (SNR floored at 5; _stat_as), sys = the detection's
    # systematic astrometric error measured from the data (below), K_ASSOC = 3.034 =
    # the 99th percentile of the Rayleigh distribution of a 2-D Gaussian offset.
    # No per-band operating radius, no floor, no cap. A source with N detections
    # carries stat = sqrt(sum w^2 stat^2) / sum w with w = 1/(stat^2 + sys^2) per detection, and the systematic
    # averaged down with the same weights (2026-09-27: this comment said w = SNR^2 and 'its band's sys'; code is inverse variance).
    K_ASSOC = 3.034                      # dimensionless multiplier of the quadrature sum (99th percentile, Rayleigh)
    # B-32 (a), Gregg 2026-10-13: ONE localization estimator and ONE error source in every tier - applied to the in-place
    # exopipe_ops/source_localization.py by the import wrap (core/wrap.py, lib/catalog/wraps/source_localization.py)
    LOC_ESTIMATOR = 'ivw-sandwich'       # the inverse-variance mean with the sandwich error sqrt(w^T V w), a non-positive
                                         # variance refuses; 'production' = GLS (inv / pinv) outside the 10-minute tier
    LOC_ERROR_SOURCE = 'fitted'          # each detection's fitted statistical error (det_stat_err_arcsec), per detection the
                                         # fitted row over a fork-tier row, none -> refuse; 'production' = first row seen, else
                                         # 0.42 x beam / SNR. Served Run_P3S5 2026-10-14 07:50: every singles detection
                                         # carries it, the multiples-fork tiers none - 52% of identities would refuse
                                         # (plan/wave2/B32_DESIGN.md; Gregg to choose: DECISIONS_PENDING C.8b)
    RULED = {'K_ASSOC': 'R-ASSOCIATION-RADIUS-QUADRATURE', 'LOC_ESTIMATOR': 'R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32', 'LOC_ERROR_SOURCE': 'R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32'}


class fitting(_Section):
    """Source fits shared by search, catalog and forced in every tier (core/fitting.py). CLAUDE.md 5 (Gregg 2026-10-04):
    Gaussian fit (free size and shape) at or above GAUSS_SNR; below, a template fit (empirical beam fixed, amplitude and
    position free); positions carry their errors."""
    GAUSS_SNR = 7.0                      # sigma; Gaussian at or above, template below (CLAUDE.md 5)
    RADIO_SNR_FLOOR = 4.0                # sigma; radio positions computed at |S/N| >= this (deep_forced_radio)
    # Template error scale K (68th percentile of |error| x S/N / FWHM per beam axis), (major, minor), per tier and Stokes:
    # deep: re-measured 2026-10-13 with the fixed (WCS-oriented) template and an un-mirrored axis projection, same plan,
    # frames and seeds (exopipe_ops 554b721; D5 approved, RULINGS-GREGG-20261013B-MIRROR-EXCLUSIONS-ALL-APPROVED);
    # superseded: I 0.520/0.510 (ratified 2026-10-11), V 0.575/0.531 (adopted 2026-10-11) - measured with the mirrored beam
    K_TEMPLATE = {('deep', 'I'): (0.542, 0.500),
                  ('deep', 'V'): (0.572, 0.551),
                  # sub-LST: re-measured after the mirror fix (exopipe_ops ac2d267; injection errors projected on the true
                  # beam axes, 5370179); superseded I 0.476/0.423, V 0.658/0.499 (measured with the mirrored template)
                  ('sublst', 'I'): (0.464, 0.450),
                  ('sublst', 'V'): (0.599, 0.602)}
    # snapshot tier: the template module's own K, re-measured with the template no longer mirrored (clv15 a715a1b,
    # phase3_template_fit.py:42; 97,200 trials): major 0.683 (0.639 at >= 69 MHz), minor 0.737; superseded 0.60/0.81/0.86
    K_SNAPSHOT = (0.683, 0.737)
    K_SNAPSHOT_MAJOR_HIGH = 0.639
    K_SNAPSHOT_HIGH_MHZ = 69.0
    TEMPLATE_WIN_BEAMS = 0.8             # fit window radius, x FWHM(major)
    TEMPLATE_MAX_SHIFT_BEAMS = 0.5       # bound on the shift from the peak start, x FWHM(major)
    TEMPLATE_SEARCH_BEAMS = 1.0          # peak start sought within this radius of the given position, x FWHM(major)
    # Orientation of the template beam in pixel space. 'legacy-mirrored' reproduces the production template fit, whose
    # beam is the mirror image of the header beam (pixel +x taken as east although CDELT1 < 0; refactor finding D5,
    # 2026-10-13, put to Gregg by both chats). 'wcs' is the corrected beam; switching needs Gregg's ruling and a
    # re-measured K for every tier.
    TEMPLATE_BEAM_ORIENTATION = 'wcs'    # D5 approved 2026-10-13 04:20 UTC; production clv15 44f0155. Parity against products
                                         # made before 04:20 UTC sets fitting.TEMPLATE_BEAM_ORIENTATION=legacy-mirrored
                                         # (and the superseded K) into its non-production --out.
    RULED = {'GAUSS_SNR': 'RULING-TEMPLATE-FIT-BELOW-7-SIGMA',
             'K_TEMPLATE': ('RULINGS-GREGG-20261013B-MIRROR-EXCLUSIONS-ALL-APPROVED',
                            'R-GREGG-20261013-TENMIN-TEMPLATE-K-MEASURED'),
             'TEMPLATE_BEAM_ORIENTATION': 'RULINGS-GREGG-20261013B-MIRROR-EXCLUSIONS-ALL-APPROVED'}


class fallbacks(_Section):
    """B-30 (a), Gregg 2026-10-13: no silent fallbacks (core/fallback.py). One key per fallback site - a place where code
    substituted a default for missing, unreadable or degenerate input and carried on: 'refuse' (ruled) = the site refuses,
    naming itself and its input; 'fallback' (production) = production behaviour, counted in the unit record (field
    `fallbacks`). exopipe/phase3/parity/production_20261013.cfg sets every key to 'fallback'. One section for every step
    (a site in shared code - coadd.get_midpoint_time - serves several), so it is in every step's resume hash."""
    # an image with no DATE-OBS / DATE-BEG / MJD-OBS and no YYYYMMDD_HHMMSS in its name gets 2025-01-01T00:00 UTC
    # (lib/masters/coadd.py get_midpoint_time = coadd_full_pipeline.py:187-203: masters build epochs, search identification)
    IMAGE_TIME_FALLBACK = 'refuse'
    # a residual's per-epoch source header unreadable -> its time block dropped (coadd.epoch_time_header)
    EPOCH_TIME_SOURCE_FALLBACK = 'refuse'
    # master time span: a night without a time source / header / DATE-OBS / DATE-END left out, INTTIME
    # unparsable ignored, no span or a failed write -> the template epoch time (coadd.stamp_master_span)
    MASTER_SPAN_FALLBACK = 'refuse'
    # previous master present but unreadable / no COADDOP -> the fixed-table night becomes the
    # alignment reference (coadd._select_reference)
    REFERENCE_MASTER_FALLBACK = 'refuse'
    # unreadable image header -> the night passes the post-cutover grid gate unchecked (coadd._grid_refused_dates)
    GRID_HEADER_FALLBACK = 'refuse'
    # 60-min PB map absent -> the mean of the six 10-min bins (pb_apply.subband_deep_pb)
    PB_DEEP_MAP_FALLBACK = 'refuse'
    # PB map without PBTREF -> the window beam computed at 2025-01-01 (pb_apply)
    PB_REFERENCE_TIME_FALLBACK = 'refuse'
    # no member subband with a positive weight -> an unnormalized (zero) multi-grid PB (pb_apply.wideband_pb)
    PB_WEIGHTS_FALLBACK = 'refuse'
    # master without RESTFRQ -> member beams not scaled 1/nu in the bandwidth-smearing factor (pb_apply)
    PB_RESTFRQ_FALLBACK = 'refuse'
    # preferred I master missing or corrupt -> the V clean uses the other robust (vclean_masters.pick_i_master)
    VCLEAN_I_MASTER_FALLBACK = 'refuse'
    # B-32: a band without a Master -> the localization takes 900 arcsec as its beam (source_localization.beam_as, wrapped)
    LOC_MASTER_BEAM_FALLBACK = 'refuse'
    # B-32: a failing identity is swallowed by build_localizations.py (except in the 10-minute tier) - recorded (sticky)
    LOCALIZATION_FALLBACK = 'refuse'
    # install archive audit: no epoch_gate.csv -> no night taken as epoch-gate refused (install._archive_audit)
    INSTALL_AUDIT_EPOCH_GATE_FALLBACK = 'refuse'
    # |pixel| > 1e6 Jy/beam zapped to 0 (deep_search._load) - a per-item drop: ruled 'counted' (the ruling's "per-item
    # drops counted in the run record"), n = pixels
    SEARCH_ZAP_PIXELS_FALLBACK = 'fallback'
    # no beam (the I master's empirical-beam fit NaN, a structure-map beam <= 0) -> 7 pixels (deep_search)
    SEARCH_BEAM_FALLBACK = 'refuse'
    # destriping the V master failed -> the raw master structure map gates the search (deep_search._mstruct_maps)
    SEARCH_MSTRUCT_DESTRIPE_FALLBACK = 'refuse'
    # build_name refused -> the search frame is written with PRODNAME 'unknown' (deep_search frame header)
    SEARCH_PRODNAME_FALLBACK = 'refuse'
    # the cell's PB map missing -> the pbcorr sibling is not written, recorded as owed (deep_search)
    SEARCH_PB_SIBLING_FALLBACK = 'refuse'
    # an epoch without an alignment flow is not searched (deep_search.run_deep_cell; named after the data_decisions /
    # night-image / epoch gates, which refuse their epochs anyway), n = epochs
    SEARCH_EPOCH_FLOW_FALLBACK = 'refuse'
    # a residual that failed to build in a pool worker -> the epoch is not searched (deep_search, site in the parent),
    # n = epochs
    SEARCH_RESIDUAL_FALLBACK = 'refuse'
    # a sanity / search-tier gate could not measure an epoch -> the epoch escapes the gate (deep_search): the V gate could
    # not run, an epoch's V frame rms failed, an epoch of weights.csv without a qa3 rms, a tabled epoch (deep) or an
    # existing frame (10 min) without a stage-1 rms; n = epochs
    SEARCH_SANITY_FALLBACK = 'refuse'
    # coadd_final/weights.csv missing -> the search-frame sanity gate / the generation check / the input-provenance
    # gate's generation test / the leave-one-out masters silently not used (deep_search; one count per check skipped)
    SEARCH_WEIGHTS_FALLBACK = 'refuse'
    # an epoch frame without a (numeric) DURATION -> 600 / 3600 s (deep_search frame header)
    SEARCH_DURATION_FALLBACK = 'refuse'
    # no / unparsable epoch time -> Jupiter and Moon at 00:00 UTC of the date (deep_search._body_skycoord); the frame's
    # mid-time unreadable under search_detect / search_sublst.EPHEMERIS_TIME 'frame-mid' -> its DATE-OBS (_eph_time)
    SEARCH_EPHEMERIS_TIME_FALLBACK = 'refuse'
    # a body's position failing -> the candidate is not labelled with it (candidates.ephemeris_label, the `body` column)
    SEARCH_EPHEMERIS_LABEL_FALLBACK = 'refuse'
    # the V-clean engine (search_v.main --clean_only, run by masters V-clean): V and I masters without BMAJ and no --beam
    # -> 5 pixels (the template and striation sizes). Not sites: search_v.main's primary-beam correction (beam model,
    # frequency) - apply_pb_correction's pb / true_flux are read by nothing, so no product depends on it
    VCLEAN_BEAM_FALLBACK = 'refuse'
    # a pixel block whose world coordinates fail is left out of the striation model (coadd.build_striation_model - the
    # search, its residual savers - and search_v's copy in V-clean) - a per-item drop: ruled 'counted', n = rows
    STRIATION_WCS_BLOCK_FALLBACK = 'fallback'
    RULED = {k: 'R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32' for k in ('IMAGE_TIME_FALLBACK', 'EPOCH_TIME_SOURCE_FALLBACK', 'MASTER_SPAN_FALLBACK', 'REFERENCE_MASTER_FALLBACK', 'GRID_HEADER_FALLBACK', 'PB_DEEP_MAP_FALLBACK', 'PB_REFERENCE_TIME_FALLBACK', 'PB_WEIGHTS_FALLBACK', 'PB_RESTFRQ_FALLBACK', 'VCLEAN_I_MASTER_FALLBACK', 'SEARCH_ZAP_PIXELS_FALLBACK', 'SEARCH_BEAM_FALLBACK', 'SEARCH_MSTRUCT_DESTRIPE_FALLBACK', 'SEARCH_PRODNAME_FALLBACK', 'SEARCH_PB_SIBLING_FALLBACK', 'SEARCH_EPOCH_FLOW_FALLBACK', 'SEARCH_RESIDUAL_FALLBACK', 'SEARCH_SANITY_FALLBACK', 'SEARCH_WEIGHTS_FALLBACK', 'SEARCH_DURATION_FALLBACK', 'SEARCH_EPHEMERIS_TIME_FALLBACK', 'SEARCH_EPHEMERIS_LABEL_FALLBACK', 'VCLEAN_BEAM_FALLBACK', 'STRIATION_WCS_BLOCK_FALLBACK', 'LOC_MASTER_BEAM_FALLBACK', 'LOCALIZATION_FALLBACK', 'INSTALL_AUDIT_EPOCH_GATE_FALLBACK')}


class runner(_Section):
    """Defaults for `exopipe3 run` and for every step's process environment."""
    from exopipe.phase3.core import env as _env
    ENV_DEFAULTS = dict(_env.ENV_DEFAULTS)       # applied by exopipe3 before numpy loads (core/env.py explains why)
    del _env
    FREE_CORES_CMD = 'bash /lustre/gh/main/exopipe_ops/node_free_cores.sh'   # prints the recorder-free core list
    JOBS_ROOT = '/lustre/gh/main/exopipe_ops/jobs'                            # job records of `exopipe3 run`
    SSH_PREFIX_ENV = {'SSH_AUTH_SOCK': ''}                                     # ssh from lwacalim07 (CLAUDE.md 8)
    POLL_S = 30                                                                # s, --after / status polling
    # `run submit --code COMMIT`: every step process of the job runs from a read-only `git worktree add --detach` of
    # that commit, made once per commit under this root and reused (DESIGN 11.7; replaces the sub-LST frozen code).
    CODE_WORKTREES_ROOT = '/lustre/gh/main/exopipe_ops/jobs/_code'
    CODE_REPO = None             # repository the --code commits come from; None = the repository this package runs from
    # Named chains for `run submit --chain NAME --items ...` (DESIGN 11.3): each item runs through the steps in order,
    # on one node, and stops at its first non-zero exit (refused / no-input / error recorded per step); a retried item
    # resumes at its first step not done. A step is a command-line template: {item} is replaced by the item; a
    # template without {item} gets '--cells <item>' appended. Example (for when the steps exist):
    #   'deep-cell': ['masters --tier deep --cells {item} --stages build,vclean,fluxscale,pbmaps',
    #                 'dewarp --tier deep --cells {item}', 'install --tier deep --cells {item}',
    #                 'masters --tier deep --cells {item} --stages sources']
    # Named chains (DESIGN 11.3, 12): `exopipe3 run submit --chain NAME --items ...` runs each item through these step
    # commands in order on one node; an item stops at its first non-zero exit (2 refused / 3 no-input / other error).
    # {item} is the item. The per-hour chain is the unit a merged Phase 1-2-3 controller will run (DESIGN 12.3).
    CHILD_ENV_CLEAR_PREFIXES = ('P3_', 'P3S', 'P3T', 'SUBLST_', 'SNAP_', 'FORK_', 'CATPUB_', 'NOCAT_', 'DEEP_FORCED_',
                                'FT_', 'TENMIN_', 'GEN4_', 'BDSF_')   # production switch families never inherited by a child
    CHAIN_VARS = {'deep_run': 'P3S5'}      # {deep_run} in a chain template: the current deep transient run label
    CHAINS = {
        # per hour: one LST hour of one night, every tier, against the installed masters (item kind: epoch 04h_2025-10-14)
        # deep frames are made by the deep search engine (frames_residuals.DEEP_PRODUCER 'search'): no frames step here
        'epoch': ['search --tier deep --epochs {item}',
                  'photometry --tier deep --epochs {item}',
                  'catalog --tier deep --incremental --run {deep_run} --epochs {item}',
                  'forced --tier deep --run {deep_run} --epochs {item}',
                  'frames --tier sublst --epochs {item}',
                  'flow --tier sublst --epochs {item}',
                  'search --tier sublst --epochs {item}',
                  'catalog --tier sublst --incremental --epochs {item}',
                  'frames --tier snapshot --epochs {item}',
                  'flow --tier snapshot --epochs {item}',
                  'search --tier snapshot --epochs {item}',
                  'curves --tier snapshot --epochs {item}',
                  'catalog --tier snapshot --epochs {item}',
                  'forced --tier snapshot --epochs {item}',
                  'pages --run {deep_run} --epochs {item}',
                  'serve --run {deep_run} --epochs {item}',
                  'archive push --epochs {item}'],
        # survey build of one deep cell (item kind: cell 04h_Full); dewarp (per LST) and install follow separately
        'deep-build-cell': ['masters --tier deep --stages stage-in --cells {item}',
                            'flow --tier deep --cells {item}',
                            'masters --tier deep --stages build,pbmaps,vclean,fluxscale,verify --cells {item}'],
        # re-run every hour of a cell against its rebuilt + installed master (item kind: cell)
        'deep-rerun-cell': ['search --tier deep --cells {item}',
                            'catalog --tier deep --cells {item}'],
        # sub-LST masters of one LST (item kind: lst 09h)
        'sublst-lst': ['masters --tier sublst --lsts {item}',
                       'flow --tier sublst --lsts {item}',
                       'frames --tier sublst --lsts {item}',
                       'search --tier sublst --lsts {item}',
                       'catalog --tier sublst --lsts {item}'],
        # one snapshot cell (item kind: night cell 18h_2025-04-11)
        'snapshot-cell': ['frames --tier snapshot --cells {item}',
                          'flow --tier snapshot --cells {item}',
                          'search --tier snapshot --cells {item}',
                          'curves --tier snapshot --cells {item}',
                          'catalog --tier snapshot --cells {item}'],
    }


class site(_Section):
    """Filled from the site file (EXOPIPE3_SITE). Expected names (all optional; a step that needs one refuses
    with a clear message when it is unset): NODE_POOLS {pool: [host, ...]}, GPU_HOSTS [host, ...],
    NODE_LOCAL_ROOTS {host: [root, ...]}, PYTHON {'default': path, 'flow': path}, SCRATCH_ROOT {host: path}."""
    NODE_POOLS = {}
    GPU_HOSTS = []
    NODE_LOCAL_ROOTS = {}
    PYTHON = {}
    SCRATCH_ROOT = {}
    LEGACY_CODE_ROOTS = []      # refactor window: clv15 / exopipe_ops roots the steps call in place (DESIGN 11.1)
