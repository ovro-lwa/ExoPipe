"""exopipe3 search - transient detection and every ruled gate, with FITTED positions on the searched frame (deep, sub-LST,
snapshot).

    exopipe3 search --tier deep --cells 12h_Red --stokes V,I --out /fast/gh/refactor_scratch/parity/search/deep
    exopipe3 search --tier deep --epochs 04h_2025-10-14 --out ...                (per hour: one night, every band of 04h)
    exopipe3 search --tier deep --cells 12h_Red --nights 2025-05-10 --replay-attempt published --out ...   (incremental: append)
    exopipe3 search --tier deep --cells 12h_Red --stokes I --stages fit,verify --replay-attempt <attempt dir> --out ...
    exopipe3 search --tier sublst ... / --tier snapshot ...                      (lib/search/tier_sublst.py, tier_snapshot.py)

Per hour vs intermittent (DESIGN 12): per hour, `--epochs <LST>_<date>` searches ONE night of one LST hour (every band, or the
--cells / --scopes given) against the INSTALLED masters; the deep tier runs the engine's incremental mode (every admission gate
on the whole cell, only that epoch built and searched, its rows appended to a copy of the cell's published attempt). In bulk
(intermittent: the re-run of every hour of a cell after its master was rebuilt and installed) `--cells` / `--lsts` search
whole cells; `--nights D1,D2` with `--cells` runs the same incremental mode on chosen nights of whole cells (production's
re-admission fleet, p3s4_incremental_fleet.sh). Item kinds (core/cells): deep cell <LST>_<scope> (--cells), epoch <LST>_<date>
(--epochs). The unit of work is printed in the plan (--dry-run).

Deep tier (lib/search/deep_step.py). Unit: <cell>_<S> in bulk (12h_Red_V), <cell>_<S>_<date> per hour / incremental,
<cell>_<S>_replay_<attempt> for a read-only replay; Stokes V before I (search.STOKES_ORDER: the V search computes the flows the
I search reuses).
Inputs:  the cell under its roots.csv root (= paths.COADD_SB_ROOT, /data07/gh/phase3/coadd_sb/<cell>): coadd_final/
         I_deep_Taper_R075_<scope>.fits, V_deep_Taper_R0_<scope>.fits (COADDOP ivw-mean-v3), loo/, weights.csv,
         night_image_gate.csv, epoch_gate.csv, PB maps; residuals_provisional.csv; subtracted/<date>_{I,V}_sub_<scope>.fits;
         data_decisions.csv (core/exclusions); bright-parent tables, pulsar list, v110 null tables (md5-pinned, gate G2);
         per hour / incremental: the cell's published attempt (--replay-attempt DIR, or 'published' = the newest row of
         <out>/published_<S>.csv, else of the production register; default 'published').
Outputs (under --out; production names unchanged, DESIGN 11.1): attempts/<cell>_<S>_<UTC>/ = Run_P3S5/{iresid,vresid}/<cell>/
         <date>_<S>_redo_resid[_pbcorr]_<scope>.fits (class search_frame, owned by frames), transients/Run_P3S5/
         {master_catalog.csv (detections), v_deep_ledger.csv (gate_ledger), v_deep_images.csv (image_ledger),
         refusals/*_excluded.csv (refusal_record), sigstack/ (noise_map), candidates/ (candidate_stamps)}, <cell>.log,
         fit_positions.csv + fit_positions_summary.json (detection_positions: core/fitting.detection_position on the SEARCHED
         frame; peak-fallback fraction per frame); parent_catalogs/master_R075_<cell>.gaul.csv (master_catalog);
         v110_products/<cell>/V_top100_fp_gate_p3.csv (fp_gate_verdicts); archive_mirror/<LST>/<night>/Run_20260909_000000/...
         (archive copies; the production archive is never written while paths.PRODUCTION_WRITES is False);
         published_<S>.csv (register rows cell,host,attempt); quarantine_plan/<unit>.csv (frames production would move);
         replay: replay/<attempt>/{fit_positions.csv, fit_positions_summary.json, verify_report.csv}.
         Node-local scratch (--work, else the site SCRATCH_ROOT): search_deep/{attempts,flowroot,tmp}.
Stages:  preflight, quarantine, detect, fit, verify, publish (--stages / --from / --until; search.STAGES). detect = the engine
         lib/search/deep_search.run_deep_cell called IN PLACE in a fresh child process per unit (every inherited P3_* variable
         cleared; the engine environment built from config). fit and verify run on an existing attempt alone with
         --replay-attempt DIR (or 'published': the cell's attempt from the register), read only.
Failure modes (exit 2 = refused, the gate named in the record): every engine gate E1-E37 of the spec (a RuntimeError carrying
         REFUSING/REFUSED); preflight: a missing master / gate record, a superseded COADDOP, the membership rulings, stale
         search-tier table, a missing or stale flow cache on a node without a GPU (site GPU_HOSTS), pinned-input md5 (G2),
         --out/--work under a production tree (G3), config untrue about an engine literal, --inject (no deep injection path,
         Q9); per hour / incremental: no published attempt to extend; fit: a
         detection without its searched frame, a catalogue already fitted, > 1 % of fitted rows on a pixel centre; verify:
         any content test. Exit 3 = nothing to do (policy skip below search_detect.MIN_EPOCHS; per hour: the epoch is
         excluded or has no product); exit 1 = error (engine crash, flow batch rc not in (0, 2), node short of disk).

Sub-LST and snapshot tiers: lib/search/tier_sublst.py and tier_snapshot.py (their flags are prefixed --sublst- / --snap-).
"""
import importlib
import os
import socket

SUMMARY = 'detection + gates with fitted positions (deep, sub-LST, snapshot)'
TIERS = ('deep', 'sublst', 'snapshot')
SELECTORS = ('tier', 'cells', 'lsts', 'scopes', 'stokes', 'exclude_cells', 'run_label', 'epochs', 'nights')
WRITES = True

_TIER_MODULES = {'deep': 'exopipe.phase3.lib.search.deep_step',
                 'sublst': 'exopipe.phase3.lib.search.tier_sublst',
                 'snapshot': 'exopipe.phase3.lib.search.tier_snapshot'}
_DEEP_SECTIONS = ('search', 'search_inputs', 'search_detect', 'search_tables', 'search_fpgate', 'search_fit', 'search_run')
_SHARED = ('fitting', 'astrometry', 'exclusions', 'paths', 'instrument', 'runner')


def _config_sections():
    """CONFIG: the deep sections, every other search_* section present (the tier drafts), and the shared ones. Computed from
    the loaded config so the declaration never names a section that does not exist (config.declared_sections refuses one)."""
    from exopipe.phase3.config import CFG
    have = CFG.sections()
    tiers = tuple(s for s in have if s.startswith('search_') and s not in _DEEP_SECTIONS)
    return tuple(s for s in _DEEP_SECTIONS if s in have) + tiers + tuple(s for s in _SHARED if s in have)


CONFIG = _config_sections()


def _tier_module(tier, required=True):
    """The tier module, imported lazily. A tier module that does not exist yet is None at parser time (its flags are simply
    absent) and a refusal at run time; any other import error propagates (a broken tier module must be loud)."""
    name = _TIER_MODULES[tier]
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name != name:
            raise
        if required:
            from exopipe.phase3.core.record import Refusal
            raise Refusal(f'search --tier {tier}: {name} is not available yet (the tier module is written by its own unit)')
        return None


def add_arguments(p):
    g = p.add_argument_group('search (every tier)')
    g.add_argument('--stages', help='stages to run, comma-separated (default search.STAGES[tier]; deep: preflight, quarantine, '
                                    'detect, fit, verify, publish). --dry-run runs preflight only')
    g.add_argument('--from', dest='from_stage', metavar='STAGE', help='first stage to run (of the default list)')
    g.add_argument('--until', dest='until_stage', metavar='STAGE', help='last stage to run (of the default list)')
    g.add_argument('--inject', metavar='MANIFEST',
                   help='recorded injection search, never published or served (deep: REFUSED - no production injection-'
                        'through-search path exists, spec Q9; sub-LST / snapshot: see their tier flags)')
    g.add_argument('--inject-tag', metavar='TAG', help='sub-root name of the injected outputs (default: sha1 of the manifest, 12)')
    g.add_argument('--replay-attempt', metavar='DIR|published',
                   help="an existing attempt directory: deep bulk = run fit/verify on it read-only (outputs to --out); deep per "
                        "hour (--epochs) = the published attempt the epoch is appended to. 'published' = each cell's attempt "
                        "from the register (search_run.REGISTER)")
    for tier in TIERS:
        mod = _tier_module(tier, required=False)
        if mod is not None and hasattr(mod, 'add_arguments'):
            mod.add_arguments(p)


def _host():
    return socket.gethostname().split('.')[0]


def default_out(args):
    """The tier's production root (the collect root of today's fleets); refused for writes while paths.PRODUCTION_WRITES is
    False - pass --out."""
    tier = getattr(args, 'tier', None) or TIERS[0]
    mod = _tier_module(tier, required=False)
    if mod is not None and hasattr(mod, 'default_out'):
        return mod.default_out(args)
    from exopipe.phase3.config import CFG
    root = CFG.search_run.COLLECT_ROOT.get(tier)
    if root is None:
        return '/tmp'
    arm = getattr(args, 'sublst_arm', None) or getattr(CFG.search_inputs, 'SUBLST_ARM', 'B')
    return root.format(host=_host(), arm=arm)


def resolve_stages(args, stages, optional=()):
    """The stage list of this run: --stages (any of `stages` + `optional`, in pipeline order) or the default list cut by
    --from / --until. Unknown names raise ValueError (exit 1). Shared by the tier modules."""
    known = tuple(stages) + tuple(s for s in optional if s not in stages)
    if args.stages:
        want = [s.strip() for s in args.stages.split(',') if s.strip()]
        bad = [s for s in want if s not in known]
        if bad:
            raise ValueError(f'unknown stage(s) {bad}; known: {", ".join(known)}')
        order = {s: i for i, s in enumerate(known)}
        sel = sorted(dict.fromkeys(want), key=order.get)
    else:
        sel = list(stages)
    if args.from_stage:
        if args.from_stage not in sel:
            raise ValueError(f'--from {args.from_stage}: not among the stages {sel}')
        sel = sel[sel.index(args.from_stage):]
    if args.until_stage:
        if args.until_stage not in sel:
            raise ValueError(f'--until {args.until_stage}: not among the stages {sel}')
        sel = sel[:sel.index(args.until_stage) + 1]
    if getattr(args, 'dry_run', False):
        sel = [s for s in sel if s == 'preflight'] or ['preflight']
    return sel


def run(args, run):
    """Dispatch to the tier module. (The spec's gate G1 - template orientation must be 'legacy-mirrored' - is retired: D5 was
    approved 2026-10-13 04:20 UTC, RULINGS-GREGG-20261013B-MIRROR-EXCLUSIONS-ALL-APPROVED, and production's template fit is
    WCS-oriented since clv15 44f0155. The deep detect engine runs no template fit; the fit stage uses core/fitting with
    fitting.TEMPLATE_BEAM_ORIENTATION and records it.)"""
    mod = _tier_module(args.tier)
    return mod.run(args, run)
