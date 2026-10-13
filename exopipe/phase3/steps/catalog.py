"""exopipe3 catalog - candidate catalogues: harvest, merge, conformance, fitted positions, selection, association, multiples
fork, identity, localizations, names (deep); the sub-LST catalogue chain; the snapshot batch chain.

Use (per hour vs intermittent, DESIGN 12):
  per hour      exopipe3 catalog --incremental --epochs 04h_2025-10-14     add one hour's detections to the cumulative
                deep catalogue (lib/catalog/incremental.py); snapshot: the batch chain per LST x night
  intermittent  exopipe3 catalog --tier deep --run P3S5                     the full deep publish pass, preflight..fork
                exopipe3 forced --tier deep --run P3S5 ...                  forced photometry into the tier catalogues
                exopipe3 catalog --tier deep --run P3S5 --from union        union, identity, localizations, transient list
                exopipe3 catalog --tier sublst --lsts 05h ...  /  --tier snapshot --run Run_SNAP... --cells @CELLS
  unit of work  deep: the run (units <label>.<stage>, <label>.associate.<K>, <label>.fork.<S>); sub-LST: one LST of
                one arm; snapshot: the batch run. --cells on the deep tier is a TEST filter (needs --out outside the
                production roots).

Deep stages (--stages / --from / --until; default preflight..fork because `forced` runs between fork and union):
  preflight  per-pass parent-catalogue snapshot of every installed cell on its CURRENT master (ops/p3s5_parent_union.py);
             a cell without one REFUSES, naming `exopipe3 masters --tier deep --stages sources --cells ...` (parents are
             built only by masters); the install lease is held from here to the fork I content test
  harvest    the published search attempts (ops/p3s3_sources.py); pulls any not yet harvested (production only)
  merge      merge gate + staleness gate (ops/p3s3_merge.py, ops/p3s3_stale_exclusions.py)
  conform    publish-side conformance (ops/p3s2_conform.py, P3_CONFORM_SIZE=0, the parent snapshot)
  fitpos     fitted detection positions on the searched frames (ops/deep_fit_positions.py; P3S* labels)
  select     selection (ops/select_v3.py, verdicts from master_catalog.csv), ephemeris filter (ops/p3s2_ephem_filter.py),
             split by Stokes and tier (ported)
  associate  per catalogue key V_wb, I_wb, V_sb, I_sb: association, identity merge, E statistic, promotion, unique
             J-names (ops/build_source_catalogs_v3.py)
  fork       the multiples fork per Stokes, run + its content test (ops/multiples_fork.py run|check)
  union      the four primary tiers into source_catalogs/ (ported, D9: one row per (Stokes, source_id), 'stokes' column)
  identity   one identity per source across tiers and Stokes (ops/p3s3_identity.py)
  localize   combined positions per identity (ops/build_localizations.py)
  list       TRANSIENT_LIST_<label>_<UTC>.csv + _latest (ops/p3s2_transient_list.py)

Inputs (deep): the search state <paths.OPS_ROOT>/<label>_state (published_<S>.csv, markers_<S>/), the published attempts
  under the collect roots, the masters paths.COADD_SB_ROOT (lwacalim07), the parent catalogues masters --stages sources
  built, the archived searched frames, the bright-parent tables, data_decisions.csv (core.exclusions), the sample
  catalogues; and, for every stage, ONLY the files its predecessor stages' run records list (no glob decides an input;
  DESIGN 11.5). Every exopipe_ops module runs IN PLACE through core.legacy.run (site interpreter catalog.DEEP_PYTHON).
Outputs (deep, in the run dir = --out, production relative paths): master_catalog*.csv, v_deep_ledger.csv,
  v_deep_images.csv, selected_*.csv, source_catalogs_<K>/, fork_<S>/, source_catalogs_<S>_f{wb,sb}/, source_catalogs/
  (union, localizations), TRANSIENT_LIST_*.csv; records <out>/_records/catalog/<label>.<stage>.json and stage
  directories <out>/_records/catalog/<label>/stage_<stage>/ (fresh each run; previous products moved aside, never
  deleted). Product classes: parent_snapshot, harvest_manifest, merged_record, conformant_catalog, frame_table,
  fitted_catalog, selection, source_catalog, fork_catalog, union_catalog, identity_table, localization_table,
  transient_list (+ candidate_catalog; the sub-LST and snapshot classes come with their tier modules).
Failure modes: exit 2 = refused by a gate, named in the record: binding check (a config literal the in-place module does
  not carry), an override of a bound or protected key, a predecessor record not current (re-run from that stage), a
  stale file the in-place module would glob, parent snapshot incomplete (D-G1), unharvested attempts outside production
  (D-G3), merge gate (D-G4..D-G8), conformance (D-G9..D-G16), fitted positions (D-G17), selection (D-G18), builds
  (D-G19), fork (D-G20, D-G23), union..list (D-G21; D-G22 no forced record newer than the builds; partial --stokes).
  The chain stops at the first stage that does not end done. Exit 1 = error; 3 = nothing to do.
Writes outside --out: none when --out is outside the production roots (the search state is copied under
  <out>/_state/, the parent snapshot goes under <out>/_parents/); a production run also edits the search markers
  (staleness gate) and pulls unharvested attempts (harvest), as production does today.
Other tiers: sub-LST and snapshot run lib/catalog/tier_sublst.py / tier_snapshot.py (their flags are prefixed --sublst-,
  --snap-). In the refactor window the snapshot per-tier localization belongs to `serve`, the fit-store row quarantine to
  `archive`, the sub-LST and deep parent catalogues to `masters --stages sources` (refactor decisions 2026-10-13).
"""
import importlib
import sys

SUMMARY = 'harvest, merge, conformance, fitted positions, selection, association, fork, identity, localizations, names; ' \
          'sub-LST chain; snapshot chain'
TIERS = ('deep', 'sublst', 'snapshot')
SELECTORS = ('tier', 'run_label', 'lsts', 'cells', 'stokes', 'exclude_cells', 'epochs')
WRITES = True
_WANT_CONFIG = ('catalog', 'catalog_merge', 'catalog_conform', 'catalog_fitpos', 'catalog_select', 'catalog_associate',
                'catalog_fork', 'catalog_identity', 'catalog_localize', 'catalog_sublst', 'catalog_sublst_tie',
                'catalog_sublst_veto', 'catalog_snapshot', 'catalog_incremental', 'astrometry', 'fitting', 'exclusions',
                'paths', 'instrument', 'runner', 'site')


def _declared_config():
    """CONFIG = the sections this step reads that exist (the tier and incremental drafts are written by other units in
    the refactor window; a missing one must not break `exopipe3 catalog`)."""
    try:
        from exopipe.phase3.config import CFG
        have = CFG.sections()
    except Exception:
        return ('catalog',)
    return tuple(s for s in _WANT_CONFIG if s in have)


CONFIG = _declared_config()

_TIER_ERRORS = {}     # tier -> why its module could not be loaded (reported by run and --help)


def _tier_module(tier):
    try:
        return importlib.import_module(f'exopipe.phase3.lib.catalog.tier_{tier}')
    except Exception as e:                      # written by the catalog-tiers unit; may be absent or broken
        _TIER_ERRORS[tier] = f'{type(e).__name__}: {e}'
        return None


def _incremental_module():
    try:
        return importlib.import_module('exopipe.phase3.lib.catalog.incremental')
    except Exception as e:
        _TIER_ERRORS['incremental'] = f'{type(e).__name__}: {e}'
        return None


def select_stages(all_stages, args, default_until=None):
    """The stages to run, in pipeline order, from --stages (exact set) or --from/--until (contiguous range). With neither
    --until nor --stages the range ends at ``default_until`` when it starts at or before it, else at the last stage.
    An unknown stage name raises ValueError (exit 1). Shared by the tier modules (args.stages, args.from_stage,
    args.until_stage)."""
    all_stages = list(all_stages)
    stages, a, b = getattr(args, 'stages', None), getattr(args, 'from_stage', None), getattr(args, 'until_stage', None)
    if stages:
        if a or b:
            raise ValueError('--stages is an exact set: do not combine it with --from/--until')
        want = [s.strip() for s in stages.split(',') if s.strip()]
        unknown = [s for s in want if s not in all_stages]
        if unknown:
            raise ValueError(f'unknown stage(s) {unknown}; stages: {", ".join(all_stages)}')
        return [s for s in all_stages if s in want]
    for s in (a, b):
        if s and s not in all_stages:
            raise ValueError(f'unknown stage {s!r}; stages: {", ".join(all_stages)}')
    i = all_stages.index(a) if a else 0
    if b:
        j = all_stages.index(b)
    elif default_until and default_until in all_stages and i <= all_stages.index(default_until):
        j = all_stages.index(default_until)
    else:
        j = len(all_stages) - 1
    if j < i:
        raise ValueError(f'--until {b} comes before --from {a}')
    return all_stages[i:j + 1]


def add_arguments(p):
    g = p.add_argument_group('stages and modes (every tier)')
    g.add_argument('--stages', help='run exactly these stages, in pipeline order (comma list; see the stage list above)')
    g.add_argument('--from', dest='from_stage', metavar='STAGE',
                   help='first stage of a contiguous range; the predecessor records must be done and current (else exit 2)')
    g.add_argument('--until', dest='until_stage', metavar='STAGE', help='last stage of the range')
    g.add_argument('--incremental', action='store_true',
                   help='per-hour mode: add the --epochs (<LST>_<date>) to the cumulative deep catalogue '
                        '(lib/catalog/incremental.py; results equal a full rebuild over the same epochs)')
    d = p.add_argument_group('tier deep')
    d.add_argument('--parents', '--deep-parents', dest='deep_parents', default='snapshot', metavar='DIR|snapshot',
                   help="parent-catalogue directory passed to every stage as P3_PARENT_DIR (as the pass exports it); "
                        "'snapshot' (default) = assemble the per-pass snapshot of the current catalogues in preflight "
                        "(production: catalog.DEEP_PARENT_SNAPSHOT; otherwise under <out>/_parents/); a directory = use it "
                        "(e.g. the production pass's own snapshot in a parity run). Missing cells refuse")
    for tier in ('sublst', 'snapshot'):
        mod = _tier_module(tier)
        if mod is None or not hasattr(mod, 'add_arguments'):
            continue
        tg = p.add_argument_group(f'tier {tier} (lib/catalog/tier_{tier}.py)')
        try:
            mod.add_arguments(tg)
        except Exception as e:
            _TIER_ERRORS[tier] = f'add_arguments: {type(e).__name__}: {e}'
    inc = _incremental_module()
    if inc is not None and hasattr(inc, 'add_arguments'):
        try:
            inc.add_arguments(p.add_argument_group('--incremental (lib/catalog/incremental.py)'))
        except Exception as e:
            _TIER_ERRORS['incremental'] = f'add_arguments: {type(e).__name__}: {e}'
    if _TIER_ERRORS:
        p.epilog = 'not loaded: ' + '; '.join(f'{k}: {v}' for k, v in sorted(_TIER_ERRORS.items()))


def default_out(args):
    from exopipe.phase3.config import CFG
    if getattr(args, 'incremental', False):
        inc = _incremental_module()
        if inc is not None and hasattr(inc, 'default_out'):
            return inc.default_out(args)
    if args.tier == 'deep' or getattr(args, 'incremental', False):
        return f'{CFG.paths.TRANSIENTS_ROOT}/Run_{args.run_label or CFG.catalog.DEEP_RUN_LABEL}'
    mod = _tier_module(args.tier)
    if mod is not None and hasattr(mod, 'default_out'):
        return mod.default_out(args)
    raise ValueError(f'catalog --tier {args.tier}: no default product root (lib/catalog/tier_{args.tier}.py not loaded: '
                     f'{_TIER_ERRORS.get(args.tier, "no default_out")}); give --out')


def run(args, run):
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import cells, record
    if args.incremental:
        try:
            epochs = cells.expand_epochs(args.epochs) if args.epochs else None    # items <LST>_<date> (core.cells)
        except ValueError as e:
            print(f'error: {e}', file=sys.stderr)
            return record.EXIT_ERROR
        inc = _incremental_module()
        if inc is None or not hasattr(inc, 'run_incremental'):
            print(f'catalog --incremental: lib/catalog/incremental.py not available '
                  f'({_TIER_ERRORS.get("incremental", "no run_incremental")})', file=sys.stderr)
            return record.EXIT_ERROR
        return inc.run_incremental(args, run, epochs)
    if args.epochs:
        print('--epochs is the per-hour item kind: use it with --incremental', file=sys.stderr)
        return record.EXIT_ERROR
    if args.tier == 'deep':
        if not args.run_label:
            print('catalog --tier deep needs --run <label> (e.g. --run P3S5)', file=sys.stderr)
            return record.EXIT_ERROR
        try:
            stages = select_stages(CFG.catalog.DEEP_STAGES, args, CFG.catalog.DEEP_DEFAULT_UNTIL)
            from exopipe.phase3.lib.catalog import deep_chain
            return deep_chain.run(args, run, stages)
        except ValueError as e:
            print(f'error: {e}', file=sys.stderr)
            return record.EXIT_ERROR
    mod = _tier_module(args.tier)
    if mod is None or not hasattr(mod, 'run'):
        print(f'catalog --tier {args.tier}: lib/catalog/tier_{args.tier}.py not available '
              f'({_TIER_ERRORS.get(args.tier, "no run(args, run)")})', file=sys.stderr)
        return record.EXIT_ERROR
    return mod.run(args, run)


# the content tests of this step's classes register on import (core/validate.OWNERS names this module)
from exopipe.phase3.lib.catalog import tests as _tests   # noqa: E402,F401
