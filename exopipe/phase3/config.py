"""Phase 3 configuration - EVERY tunable of every `exopipe3` step, in one file.

How to use
----------
- Read:      ``from exopipe.phase3.config import CFG``; ``CFG.search.V_FLOOR_SIGMA``.
- Inspect:   ``exopipe3 config`` lists every section; ``exopipe3 config --section search`` prints that section
             with its comments (unit and ruling of every value).
- Override:  per run, on any step: ``--set search.V_FLOOR_SIGMA=4.5`` (repeatable) or ``--config my_overrides.py``
             (a python file of ``section.KEY = value`` lines). Every override is written into the run record
             and the products' HISTORY.
- One entry: ``--set fitting.K_TEMPLATE[sublst,V]=(0.66,0.50)`` changes ONE entry of a dict (tuple keys written
             comma-separated, nested dicts with one bracket per level); a whole-dict override that omits any key
             of the current value is refused (it would silently delete the omitted entries).
- Protected: ``PROTECTED`` below (the exclusion file and rule parameters, the production-root guard, the ledger,
             pinned inputs) - no override can change them; change them in this file, by a commit.
- Ruled:     a section's ``RULED = {'KEY': 'ledger ID'}`` names the keys a ruling of Gregg's set. Overriding one needs
             ``--ruling-for section.KEY=<ledger ID>`` (the ID must be a row of ``paths.FIX_LEDGER``) or a ``--out``
             outside every production root; the override and its authority go into the run record.

Conventions
-----------
- One class per step (``masters``, ``search``, ...) plus shared sections (``instrument``, ``paths``,
  ``exclusions``, ``astrometry``, ``products``, ``qa``, ``runner``, ``site``).
- Every value carries its unit (in the name or the comment) and the ledger ID of the ruling that set it.
- Tables that are data (scale tables, bright-parent tables, dewarp acceptance) are referenced by versioned path,
  never edited in place.
- Host names, node pools, GPU hosts and python environments are NOT here: they live in the site file named by
  the environment variable ``EXOPIPE3_SITE`` (ops, not git), loaded into ``CFG.site`` below. No production
  module names a host (PHASE3_ARCHITECTURE.md 0.1).
- Library modules read ``CFG`` when they are imported; the CLI applies ``--set`` / ``--config`` overrides
  before it imports any step, so overrides reach every module.
"""
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import re


_META = ('RULED', 'PROTECTED_KEYS')      # UPPER_CASE section attributes that DESCRIBE tunables; never tunables themselves


class _Section:
    """Base class of a config section: UPPER_CASE class attributes are the tunables. ``RULED`` (optional) maps a key to
    the ledger ID(s) of the ruling that set it."""

    @classmethod
    def items(cls):
        return [(k, getattr(cls, k)) for k in sorted(vars(cls)) if k.isupper() and k not in _META]


# =====================================================================================================================
# Shared sections
# =====================================================================================================================

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
                        '/lustre/gh/main/exopipe_ops/snapshot_fleet_out')     # snapshot per-cell outputs
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


# ---------------------------------------------------------------------------------------------------------------------
# Data roots and the Phase 3 directory layout of the pre-refactor package, absorbed VERBATIM (refactor 2026-10-13,
# MOVE_PLAN.md 2.0) from exopipe/common/roots.py and exopipe/common/phase3_paths.py. Module-level names, not a section:
# they read the environment exactly as before (required_root / coadd_root at call time, STAGING_ROOT at import).
#
# From exopipe/common/roots.py - "Canonical data roots - REQUIRED from the environment, no defaults.
# Gregg 2026-09-04: the superseded sigma-clipped co-add tree (/fast/gh/main/phase3/coadd) was deleted and every module
# default that pointed at it was removed ("remove the defaults"). A run that does not state its co-add root explicitly
# must fail loudly here, never fall through to a stale generation." (R-COADD-ROOT-NO-DEFAULTS)
# ---------------------------------------------------------------------------------------------------------------------

class RootUnset(RuntimeError):
    pass


def required_root(var, purpose):
    v = os.environ.get(var)
    if not v:
        raise RootUnset(
            f"{var} is not set ({purpose}). No default exists: the retired "
            f"/fast/gh/main/phase3/coadd tree was deleted 2026-09-04 and module "
            f"defaults were removed on Gregg's order. Export {var} to the "
            f"current-generation cell root (wideband: rebuild_w2/rebuild_v2 "
            f"on the node's /fast; subband: /data07/gh/phase3/coadd_sb).")
    return v


def coadd_root():
    """Root holding <lst>_<scope>/coadd_final/ and <lst>_<scope>/subtracted/."""
    return required_root('P3_COADD_ROOT', 'co-add cell root: masters + subtracted epochs')


# From exopipe/common/phase3_paths.py - "canonical Phase 3 directory layout (PHASE3_STRUCTURE.md).
# Staging on /fast MIRRORS the final /lustre tree so promotion is a plain copy. One builder for every write site;
# nothing constructs paths ad hoc.
#     {root}/{LST}/{date}/P3_{label}/{scope}/{Stokes}/{cadence}/    per-epoch
#     {root}/{LST}/coadd/Run_{label}_{lo}-{hi}/{scope}/{Stokes}/{cadence}/   masters
#     {root}/transients/Run_{label}/...                            survey-wide blind search
#     {root}/samples/{sample}/{target}/                            targeted forced photometry
# scope in {18MHz...82MHz, Red, Green, Blue, Full}; Stokes in {I, V}; cadence in {deep, 10min}"

STAGING_ROOT = os.environ.get('P3_STAGING_ROOT',
                              '/data07/gh/phase3')  # conformant FINAL products (lwacalim07;
                                                # /fast too small). Flat working staging
                                                # (WORKDIR_BASE) stays on /fast for I/O.
                                                # P3_STAGING_ROOT: staged-node override
                                                # (lwacalim01/02 have no /data07).
ARCHIVE_ROOT = '/lustre/pipeline/exopipe/phase3'  # promotion target


def epoch_dir(root, lst, date, label, scope, stokes, cadence, make=False):
    """Per-epoch product dir: aligned+scaled variants + final subtracted images."""
    p = os.path.join(root, lst, date, f'P3_{label}', scope, stokes, cadence)
    if make:
        os.makedirs(p, exist_ok=True)
    return p


def coadd_dir(root, lst, label, date_lo, date_hi, scope, stokes, cadence,
              make=False):
    """Co-add Run dir (sibling of date dirs; spans dates; date range in name)."""
    run = f'Run_{label}_{date_lo}-{date_hi}'
    p = os.path.join(root, lst, 'coadd', run, scope, stokes, cadence)
    if make:
        os.makedirs(p, exist_ok=True)
    return p


def coadd_run_root(root, lst, label, date_lo, date_hi, make=False):
    """The co-add Run root (holds QA csvs spanning scopes/Stokes)."""
    p = os.path.join(root, lst, 'coadd', f'Run_{label}_{date_lo}-{date_hi}')
    if make:
        os.makedirs(p, exist_ok=True)
    return p


def transients_run(root, label, make=False):
    """Survey-wide blind-search Run (spans ALL LSTs+dates; de-siloed)."""
    p = os.path.join(root, 'transients', f'Run_{label}')
    if make:
        os.makedirs(p, exist_ok=True)
    return p


def candidate_dir(root, label, cand_id, make=False):
    p = os.path.join(transients_run(root, label), 'candidates', cand_id)
    if make:
        os.makedirs(p, exist_ok=True)
    return p


def sample_dir(root, sample, target, make=False):
    """Targeted forced-photometry per (sample, target). sample e.g.
    solar_system, 10pc, Hot_Warm_Jupiters, Local_Volume."""
    safe = target.replace(' ', '_')
    p = os.path.join(root, 'samples', sample, safe)
    if make:
        os.makedirs(p, exist_ok=True)
    return p


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
    RULED = {'K_ASSOC': 'R-ASSOCIATION-RADIUS-QUADRATURE'}


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
                  ('sublst', 'I'): (0.476, 0.423),   # Gregg 2026-10-13 'Use the measured 10-min K'
                  ('sublst', 'V'): (0.658, 0.499)}   #   (R-GREGG-20261013-TENMIN-TEMPLATE-K-MEASURED)
    # snapshot tier: the template module's own measured K (template_fit_validation run_20261004b): major 0.60
    # (0.81 at >= 69 MHz), minor 0.86
    K_SNAPSHOT = (0.60, 0.86)
    K_SNAPSHOT_MAJOR_HIGH = 0.81
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
                                'FT_', 'TENMIN_', 'GEN4_')   # production switch families never inherited by a child
    CHAIN_VARS = {'deep_run': 'P3S5'}      # {deep_run} in a chain template: the current deep transient run label
    CHAINS = {
        # per hour: one LST hour of one night, every tier, against the installed masters (item kind: epoch 04h_2025-10-14)
        'epoch': ['frames --tier deep --epochs {item}',
                  'search --tier deep --epochs {item}',
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


# =====================================================================================================================
# Step sections (one per command, in pipeline order). Populated from the code each step absorbs; every value keeps
# the default the production code had at the refactor baseline and names its ruling.
# =====================================================================================================================

# @@STEP-SECTIONS@@


# =====================================================================================================================
# Machinery: the CFG namespace, overrides, hash, printing. Nothing below is a tunable.
# =====================================================================================================================

class ConfigRefusal(ValueError):
    """An override the config refuses: a protected key, a ruled key without a ruling, an unknown ledger ID, a whole-dict
    override that omits entries. The CLI prints it and exits 2."""


class _Config:
    """The namespace of all sections: CFG.<section>.<KEY>."""

    def __init__(self, sections):
        self._sections = dict(sections)
        for name, sec in sections.items():
            setattr(self, name, sec)
        # [(dotted key, old value, new value, source, authority)]; authority is None for an unruled key, else
        # 'ruling:<ledger ID>', 'non-production out:<dir>', 'display only' or 'forwarded (checked by the step)'
        self.overrides = []
        self._undo = []              # [(section, key, previous value)] for reset()
        self._auth = {'rulings': {}, 'out': None, 'mode': 'run'}

    def sections(self):
        return dict(self._sections)

    def get(self, dotted):
        sec, key, path = _parse_key(dotted)
        v = getattr(self._section(sec), key)
        for toks in path:
            v = v[_entry_key(v, toks, dotted)]
        return v

    # ---- protected and ruled keys -------------------------------------------------------------------------------
    def protected(self, dotted):
        """True when no override may change ``section.KEY`` (PROTECTED, 'section.*', paths.PINNED_INPUTS)."""
        sec, key, _ = _parse_key(dotted)
        base = f'{sec}.{key}'
        pinned = tuple(getattr(self._sections.get('paths'), 'PINNED_INPUTS', ()) or ())
        own = tuple(getattr(self._sections.get(sec), 'PROTECTED_KEYS', ()) or ())   # a section's own protected keys
        return base in PROTECTED or f'{sec}.*' in PROTECTED or base in pinned or key in own

    def ruled(self, dotted):
        """The ledger ID(s) of the ruling that set ``section.KEY`` (from the section's RULED), or None."""
        sec, key, _ = _parse_key(dotted)
        r = (getattr(self._section(sec), 'RULED', None) or {}).get(key)
        if r is None:
            return None
        return tuple(r) if isinstance(r, (list, tuple)) else (r,)

    def production_root_of(self, path):
        """The production root (paths.PRODUCTION_ROOTS) that holds ``path``, or None."""
        real = os.path.realpath(path)
        for root in self.paths.PRODUCTION_ROOTS:
            r = os.path.realpath(root)
            if real == r or real.startswith(r.rstrip('/') + '/'):
                return root
        return None

    def authorize(self, rulings=(), out=None, mode='run'):
        """How ruled keys may be overridden in this process (the CLI calls it before applying overrides).

        rulings: ['section.KEY=LEDGER_ID', ...] (--ruling-for) or a dict; every ID must be a row of paths.FIX_LEDGER.
        out:     the run's product root; outside every production root, a ruled key may be overridden without a ruling.
        mode:    'run' (check), 'display' (`exopipe3 config`: shown, nothing written), 'forward' (`run submit`: the
                 overrides are forwarded to each step process, which checks them against its own --out)."""
        if isinstance(rulings, dict):
            items = list(rulings.items())
        else:
            items = []
            for r in rulings or ():
                if '=' not in r:
                    raise ConfigRefusal(f'--ruling-for expects SECTION.KEY=LEDGER_ID, got {r!r}')
                k, v = r.split('=', 1)
                items.append((k.strip(), v.strip()))
        from exopipe.phase3.core import ledger
        for k, lid in items:
            _parse_key(k)
            try:
                ok = ledger.has_id(lid, self.paths.FIX_LEDGER)
            except OSError as e:
                raise ConfigRefusal(f'--ruling-for {k}={lid}: the ledger {self.paths.FIX_LEDGER} is not readable ({e})')
            if not ok:
                raise ConfigRefusal(f'--ruling-for {k}={lid}: no row with fix_id {lid!r} in {self.paths.FIX_LEDGER}')
        self._auth = {'rulings': dict(items), 'out': out, 'mode': mode}

    def _authority(self, dotted, base):
        ids = self.ruled(base)
        if not ids:
            return None
        lid = self._auth['rulings'].get(dotted) or self._auth['rulings'].get(base)
        if lid:
            return f'ruling:{lid}'
        if self._auth['mode'] == 'display':
            return 'display only'
        if self._auth['mode'] == 'forward':
            return 'forwarded (checked by the step)'
        out = self._auth['out']
        if out and self.production_root_of(out) is None:
            return f'non-production out:{os.path.abspath(out)}'
        raise ConfigRefusal(
            f'{base} carries a ruling ({", ".join(ids)}): override it only with --ruling-for {base}=<ledger ID> '
            f'(a row of {self.paths.FIX_LEDGER}) or into a --out outside every production root '
            f'(this run writes to {out or "the step production root"})')

    def check_out(self, out):
        """Re-check, once the run's final --out is known, every ruled override that was allowed for a non-production
        out (a step whose default root depends on an override could otherwise move it into production)."""
        for o in self.overrides:
            if o[4] and o[4].startswith('non-production out:') and self.production_root_of(out) is not None:
                raise ConfigRefusal(f'{o[0]} carries a ruling and was overridden without one, but this run writes '
                                    f'under the production root {self.production_root_of(out)} ({out})')

    # ---- overrides ----------------------------------------------------------------------------------------------
    def set(self, dotted, value, source='--set'):
        """Override one tunable: 'section.KEY' (whole value) or 'section.KEY[entry]...' (one entry of a dict)."""
        sec, key, path = _parse_key(dotted)
        s = self._section(sec)
        if not key.isupper() or key in _META or not hasattr(s, key):
            raise KeyError(f'unknown config key {dotted!r} (exopipe3 config --section {sec} lists the keys)')
        base = f'{sec}.{key}'
        if self.protected(base) and not getattr(self, 'parity_replay', False):
            raise ConfigRefusal(f'{base} is PROTECTED (config.PROTECTED / paths.PINNED_INPUTS): no --set or --config '
                                f'can change it; it changes only by a commit to exopipe/phase3/config.py (a --parity run '
                                f'into a non-production --out may replay a historical value)')
        if self.protected(base):
            # PARITY REPLAY: a shadow of a production run must see the inputs that run saw (e.g. the exclusion list of
            # that moment). Allowed only under --parity (every write confined to the parity root by core/guard) and
            # a non-production --out; recorded with this authority so no product of such a run can be installed.
            authority = 'parity replay: protected key overridden inside a --parity run (writes confined to --out)'
        else:
            authority = self._authority(dotted, base)
        top = getattr(s, key)
        if path:
            new_top, old, new = _set_entry(top, path, value, dotted)
        else:
            old = top
            new = _coerce(value, old) if isinstance(value, str) else _like(value, old)
            if isinstance(old, dict):
                _refuse_omitted(old, new, dotted)
            new_top = new
        setattr(s, key, new_top)
        self._undo.append((s, key, top))
        self.overrides.append((dotted, old, new, source, authority))

    def reset(self):
        """Undo every override applied in this process (tests; `exopipe3 config`)."""
        while self._undo:
            s, key, prev = self._undo.pop()
            setattr(s, key, prev)
        self.overrides = []

    def ruled_overrides(self):
        """[{key, ruled_by, authority}] for every override of a ruled key (written into each run record)."""
        out = []
        for o in self.overrides:
            ids = self.ruled(o[0])
            if ids:
                out.append({'key': o[0], 'ruled_by': list(ids), 'authority': o[4]})
        return out

    def apply_override_strings(self, items, source='--set'):
        for it in items or []:
            if '=' not in it:
                raise ValueError(f'--set expects SECTION.KEY=VALUE, got {it!r}')
            k, v = it.split('=', 1)
            self.set(k.strip(), v.strip(), source)

    def apply_override_file(self, path):
        """A python file of `section.KEY = value` lines (literals only)."""
        for n, line in enumerate(open(path), 1):
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            if '=' not in line:
                raise ValueError(f'{path}:{n}: expected section.KEY = value')
            k, v = line.split('=', 1)
            self.set(k.strip(), v.strip(), f'{path}:{n}')

    def snapshot(self):
        return {name: {k: _jsonable(v) for k, v in sec.items()} for name, sec in self._sections.items()
                if name != 'site'}

    def hash(self, sections=None):
        snap = self.snapshot()
        if sections is not None:
            snap = {k: v for k, v in snap.items() if k in sections}
        return hashlib.sha1(json.dumps(snap, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def describe(self, section=None):
        if section is None:
            out = []
            for name, sec in self._sections.items():
                doc = (inspect.getdoc(sec) or '').splitlines()
                out.append(f'{name:14s} {len(sec.items()):4d} keys  {doc[0] if doc else ""}')
            return '\n'.join(out)
        sec = self._section(section)
        try:
            return inspect.getsource(sec)
        except (OSError, TypeError):
            return '\n'.join(f'{k} = {v!r}' for k, v in sec.items())

    def _section(self, name):
        if name not in self._sections:
            raise KeyError(f'unknown config section {name!r}; sections: {", ".join(self._sections)}')
        return self._sections[name]


def _split(dotted):
    sec, key, _ = _parse_key(dotted)
    return sec, key


_KEY_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)((?:\[[^\[\]]*\])*)\s*$')


def _parse_key(dotted):
    """'section.KEY[a,b][c]' -> ('section', 'KEY', [['a', 'b'], ['c']])."""
    m = _KEY_RE.match(dotted or '')
    if not m:
        raise KeyError(f'config key must be SECTION.KEY or SECTION.KEY[entry], got {dotted!r}')
    path = [[t.strip().strip('\'"') for t in sub.split(',')] for sub in re.findall(r'\[([^\[\]]*)\]', m.group(3))]
    return m.group(1), m.group(2), path


def _entry_key(d, toks, dotted):
    """The existing key of dict ``d`` written as ``toks`` ('deep,I' -> ('deep', 'I'); '75,90' -> (75, 90))."""
    if not isinstance(d, dict):
        raise KeyError(f'{dotted}: {type(d).__name__} has no entries (only dicts take [entry])')
    for k in d:
        if isinstance(k, tuple):
            if len(k) == len(toks) and all(str(x) == t for x, t in zip(k, toks)):
                return k
        elif len(toks) == 1 and str(k) == toks[0]:
            return k
    raise KeyError(f'{dotted}: no entry [{",".join(toks)}]; entries: '
                   + ', '.join('[' + (','.join(map(str, k)) if isinstance(k, tuple) else str(k)) + ']' for k in d))


def _set_entry(d, path, value, dotted):
    """Copy of dict ``d`` with the entry at ``path`` replaced; returns (new dict, old entry, new entry)."""
    k = _entry_key(d, path[0], dotted)
    new_d = dict(d)
    if len(path) > 1:
        new_d[k], old, new = _set_entry(d[k], path[1:], value, dotted)
        return new_d, old, new
    old = d[k]
    new = _coerce(value, old) if isinstance(value, str) else _like(value, old)
    if isinstance(old, dict):
        _refuse_omitted(old, new, dotted)
    new_d[k] = new
    return new_d, old, new


def _refuse_omitted(old, new, dotted):
    if not isinstance(new, dict):
        return
    missing = [k for k in old if k not in new]
    if missing:
        raise ConfigRefusal(f'whole-dict override of {dotted} omits {len(missing)} of its {len(old)} entries '
                            f'({", ".join(map(repr, missing[:6]))}{" ..." if len(missing) > 6 else ""}); give every '
                            f'entry, or change one entry with {dotted}[entry]=value')


def _like(v, old):
    """Normalise a parsed value to the shape of the current one (list -> tuple, int -> float), recursively."""
    if isinstance(old, bool) or v is None or old is None:
        return v
    if isinstance(old, float) and isinstance(v, int) and not isinstance(v, bool):
        return float(v)
    if isinstance(old, tuple) and isinstance(v, (list, tuple)):
        if len(v) == len(old):
            return tuple(_like(x, o) for x, o in zip(v, old))
        return tuple(v)
    if isinstance(old, dict) and isinstance(v, dict):
        return {k: (_like(x, old[k]) if k in old else x) for k, x in v.items()}
    return v


def _coerce(text, old):
    """Parse an override string to the type of the current value."""
    if isinstance(old, bool):
        t = text.lower()
        if t in ('1', 'true', 'yes', 'on'):
            return True
        if t in ('0', 'false', 'no', 'off'):
            return False
        raise ValueError(f'expected a boolean, got {text!r}')
    if isinstance(old, str) or old is None:
        try:
            v = ast.literal_eval(text)
            return v if (old is None or isinstance(v, str)) else text
        except (ValueError, SyntaxError):
            return text
    v = _like(ast.literal_eval(text), old)
    if type(v) is not type(old):
        raise ValueError(f'expected {type(old).__name__}, got {text!r}')
    return v


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_jsonable(x) for x in v]
    return repr(v)


def _load_site(cfg):
    path = os.environ.get('EXOPIPE3_SITE')
    if not path:
        return None
    cfg.site_sha1 = hashlib.sha1(open(path, 'rb').read()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location('exopipe3_site', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k in dir(mod):
        if k.isupper():
            setattr(cfg.site, k, getattr(mod, k))
    return path


def _load_drafts():
    """REFACTOR WINDOW ONLY: step sections written by the implementation agents as exopipe/phase3/_cfgdraft_*.py are
    loaded here until the refactor chat folds them into this file (then the drafts and this function are deleted)."""
    import glob as _g
    here = os.path.dirname(os.path.abspath(__file__))
    for path in sorted(_g.glob(os.path.join(here, '_cfgdraft_*.py'))):
        spec = importlib.util.spec_from_file_location('exopipe.phase3.' + os.path.basename(path)[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for k, v in vars(mod).items():
            if inspect.isclass(v) and issubclass(v, _Section) and v is not _Section and not k.startswith('_'):
                if k in globals() and globals()[k] is not v:
                    raise RuntimeError(f'config draft {path} redefines section {k!r}')
                globals()[k] = v


def _collect():
    secs = {}
    for name, obj in list(globals().items()):
        if inspect.isclass(obj) and issubclass(obj, _Section) and obj is not _Section and not name.startswith('_'):
            secs[name] = obj
    return secs


_load_drafts()
CFG = _Config(_collect())
SITE_FILE = _load_site(CFG)

# Sections every step reads, whatever it declares: they are always part of a step's resume hash (critique_rulings 6.5:
# a per-section hash must not miss the shared sections).
SHARED_SECTIONS = ('instrument', 'paths', 'exclusions', 'astrometry', 'fitting')


def declared_sections(mod):
    """The sections a step module declares in ``CONFIG`` (a tuple of section names), or None when it declares none.
    An unknown section name raises KeyError."""
    decl = getattr(mod, 'CONFIG', None)
    if decl is None:
        return None
    decl = (decl,) if isinstance(decl, str) else tuple(decl)
    unknown = [s for s in decl if s not in CFG.sections()]
    if unknown:
        raise KeyError(f'step {getattr(mod, "__name__", mod)} declares unknown config section(s) {unknown} in CONFIG; '
                       f'sections: {", ".join(CFG.sections())}')
    return decl


def step_sections(mod):
    """The sections whose values make a step's config hash (DESIGN 11.4): its CONFIG plus SHARED_SECTIONS, or None
    (every section) for a step that declares no CONFIG."""
    decl = declared_sections(mod)
    if decl is None:
        return None
    return tuple(dict.fromkeys(decl + SHARED_SECTIONS))
