"""exopipe3 calibrate - versioned reference tables: derive, diff on served and judged sources, install, verify.

    exopipe3 calibrate <table> <action> [--tier T] [flags]

    exopipe3 calibrate all verify                                         every installed table this step implements
    exopipe3 calibrate bright-parent verify --tier deep                   installed deep file: pin, form, = its derivations
    exopipe3 calibrate bright-parent derive --tier deep --out /lustre/.../parity/calibrate --work /fast/gh/scratch
    exopipe3 calibrate bright-parent derive --tier deep --stages build --curves R3.json,R2V.json --version V --out ...
    exopipe3 calibrate bright-parent diff --tier deep --version V --out ...      (installed vs candidate on the served list)
    exopipe3 calibrate bright-parent install --tier deep --version V --install-ruling <ledger ID> --out ...
    exopipe3 calibrate bright-parent derive --tier sublst --lsts 10h --out ...   (on the node that holds the collect layout)
    exopipe3 calibrate multiples derive|diff|install|verify --tier deep ...
    exopipe3 calibrate scale derive --out ... ; exopipe3 calibrate fluxcorr derive --cells @cells.txt --out ...
    exopipe3 calibrate template-k derive --tier deep --stages summary --out ...   (K from the stored trials)

Use: INTERMITTENT (survey) only - a calibration is population-wide (DESIGN 12.2). Never part of the per-hour chain: the
per-hour steps read the INSTALLED tables through config section `tables`. Unit of work = one (table class, tier, action,
version); `exopipe3 run` never fans it out by item; sub-runs inside a unit run as parallel processes (--workers).

<table> (classes of plan/specs/calibrate.md 1.2):
    bright-parent  deep, sublst   2-D bright-parent tables (exopipe/common/parent_curve_tables.json deep_* keys;
                                  parent_curve_tables_sublst.json)
    multiples      deep           the -multi_ (repeat-class) keys of parent_curve_tables.json
    scale          deep           per-LST VLSSr absolute flux scale (vlssr_scale_table_v4_20261001.csv)
    fluxcorr       deep           band correction k x g (flux_correction_v2_20261001.csv) and the LOO band vector
    template-k     deep, sublst, snapshot   template-fit error scale K (fitting.K_TEMPLATE)
    all            verify only: every installed table above
    snapshot-curve, assoc-cp, untied-sys, forced-ceiling, completeness, pertie: NOT implemented in this unit (refused;
                                  plan/impl/calibrate.md section 6)
<action>:
    derive   sub-stages (--stages / --from / --until), each its own unit, record and FRESH directory:
             bright-parent / multiples: inputs (freeze the harvest list, parents union, exclusion count) -> curves
             (ops/p3s3_curve_derive.py in place) -> build (post_process + the production key loops); scale: measure
             (ops/vlssr_flux_check.py in place) -> table (scale_table_v4.py in place); fluxcorr: stageA (lib/masters/shflux
             per cell) -> bandvec (LOO iteration) -> table (flux_correction.sh_fit); template-k: inject (deep
             deep_forced_kcal_20261013/kinj_deep.py, sub-LST kcal_tenmin_inject.py, in place) -> summary (or the stored trials)
    diff     candidate vs installed: every table key; served and judged sources the change touches (the conformance table
             step replayed on the served list with both tables; flux factors per LST / band; K factors)
    install  gated (a done diff of the same candidate sha256, --install-ruling <ledger ID>, never a subset derivation):
             a NEW versioned file, a mirror at the production path under --out, a config patch (tables.* and every key that
             mirrors the pointer) and a draft ledger row - the production file, config.py and fix_ledger.csv are never written
    verify   content tests of the installed table (sha256 pin, form, recomputed from its recorded inputs) or of --candidate

Inputs:  installed tables (config `tables`, sha256-pinned); harvest list p3s5_state/published_{V,I}.csv (deep) or the
         sub-LST collect layout (lwacalim08 /fast); published attempts' v_deep_ledger.csv; parent catalogues of the current
         masters (/data07/gh/phase3/coadd_sb, collect parent_catalogs); served run (Run_P3S5 / Run_SLSTB: master_catalog
         asrun, source_catalogs_<S><tier>, judgements); VLSSr catalogue, Run_P3S4 parent catalogues and PB maps; stage-A
         masters; injection trials (deep_forced_kcal_20261011). Node-local inputs live on lwacalim07 (deep) / lwacalim08 (sub-LST).
Outputs (under --out; production names kept where a production product exists):
         <out>/calibrate/<table>/<tier>/<version>/{inputs,curves,build,measure,table,stageA,bandvec,inject,summary,diff,
         install,verify-*}/...; records <out>/_records/calibrate/<table>.<tier>.<action>.<version>.json.
         Product classes: calibration_inputs, bright_parent_curves, bright_parent_table, vlssr_ratios, scale_table,
         shflux_stagea, band_vector, fluxcorr_table, template_k_trials, template_k_record, calibration_diff, install_bundle,
         calibration_verify (content tests in lib/calibrate/tests.py; `exopipe3 verify <class> FILE`).
Failure modes (exit 2 = refused, the gate named): G3 a harvest cell without a current-master parent catalogue; G6/G7/G8 a
         derivation of the wrong form (metric, elevation bands, multiplicity, mixed metric, no sublst key); lock: the in-place
         post_process or parent-union list differs from config; G9/G10 the mirror is not base + candidate / another key family
         changed; G11 an LST without VLSSr matches; G12 the S&H fit (no OK cell, an LST without rescale); G13 band vector not
         converged; G15 a fluxcorr patch without REQUIRED_FLUXVER; G16 K outside (0, 5) or one Stokes; G17 K measured with
         another template orientation; G18/G22 install without --install-ruling, without a done diff of the same candidate,
         or into a production root (the window); G19 an installed table whose sha256 differs from its pin; G23 exclusion list
         unreadable (or harvest rows on excluded nights with EXCLUDED_ROWS='refuse'); G24 install of a subset derivation;
         TABLE-NOTES (sub-LST diff); a stage directory that exists (pass --fresh or --resume). Exit 3 = nothing to do.
         Exit 1 = error (an in-place script crashed: its log is in the stage directory).
Refactor window (DESIGN 11.1): in-place scripts run under the site interpreter through core.legacy.run; --out must be
outside every production root; --parity confines every write (this process and its children) to --out and --work.
"""
SUMMARY = 'versioned reference tables: derive, diff on served/judged sources, install, verify'
TIERS = ('deep', 'sublst', 'snapshot')
SELECTORS = ('tier', 'cells', 'lsts', 'stokes', 'exclude_cells', 'run_label')
WRITES = True

_SECTIONS = ('tables', 'calibrate', 'calibrate_bright_parent', 'calibrate_multiples', 'calibrate_scale', 'calibrate_fluxcorr',
             'calibrate_template_k', 'fitting', 'search_tables', 'masters_fluxscale', 'exclusions', 'paths', 'site')


def _config_sections():
    """CONFIG = the sections this step reads that exist in the loaded config (search_tables and masters_fluxscale are other
    units' drafts: read by diff / the pointer patch)."""
    from exopipe.phase3.config import CFG
    have = CFG.sections()
    return tuple(s for s in _SECTIONS if s in have)


CONFIG = _config_sections()


def add_arguments(p):
    from exopipe.phase3.lib.calibrate import common as K
    p.add_argument('table', choices=sorted(K.CLASSES) + list(K.NOT_IMPLEMENTED) + ['all'], help='table class (see above)')
    p.add_argument('action', choices=K.ACTIONS, help='derive, diff, install or verify')
    g = p.add_argument_group('calibrate (every class)')
    g.add_argument('--stages', help='derive sub-stages, comma-separated (default: every sub-stage of the class)')
    g.add_argument('--from', dest='from_stage', metavar='STAGE', help='first derive sub-stage')
    g.add_argument('--until', dest='until_stage', metavar='STAGE', help='last derive sub-stage')
    g.add_argument('--version', help='version label (default: derive = a new one - UTC stamp, or v<N+1> for scale/fluxcorr; '
                                     'later stages / diff / install = the newest version with a done record)')
    g.add_argument('--candidate', help='diff / install / verify: the candidate file (default: the build record of --version)')
    g.add_argument('--installed', help='diff / build: compare against this file instead of the installed table (read only)')
    g.add_argument('--install-ruling', metavar='LEDGER_ID', help="install: Gregg's ruling for this table (a fix_ledger row; G18/G22)")
    g.add_argument('--install-words', metavar='TEXT', help="install: Gregg's words (written into the draft ledger row)")
    b = p.add_argument_group('bright-parent / multiples')
    b.add_argument('--harvest-state', metavar='DIR', help='published_{V,I}.csv directory (default calibrate_bright_parent.'
                                                         'HARVEST_STATE_DIR; frozen into the version)')
    b.add_argument('--parents', metavar='DIR', help='parent catalogue directory (deep default: a fresh current-master union built '
                                                   'by p3s5_parent_union.py in node-local scratch --work)')
    b.add_argument('--curves', metavar='F[,F...]', help='build from these derivation JSONs (deep r3,r2V; multiples r3_I,r3_V,r2V; '
                                                       'sub-LST one) instead of this version\'s curves record')
    b.add_argument('--served-run', metavar='DIR', help='diff: the served run (default calibrate.SERVED_RUN[tier])')
    b.add_argument('--list', metavar='CSV', help='diff: the detection list replayed (default the served run\'s '
                                                 'master_catalog_asrun.csv, else master_catalog.csv; multiples: fork_<S>)')
    b.add_argument('--ledger-label', help='diff: run label of the published gate ledgers (default calibrate.DIFF_LEDGER_LABEL)')
    b.add_argument('--diff-mode', choices=('candidate', 'elevation'), default='candidate',
                   help="diff: 'candidate' = installed vs installed+candidate, both with elevations; 'elevation' = the "
                        "2026-10-06 replay: installed all-elevation gate vs installed + band tables")
    b.add_argument('--table-only', action='store_true', help='diff: compare the tables only, no served replay (the diff is then '
                                                             'refused unless --waive calibrate.diff.served --ruling ...)')
    s = p.add_argument_group('scale / fluxcorr')
    s.add_argument('--cat-run', help='measure: parent-catalogue run (default calibrate_scale.CAT_RUN)')
    s.add_argument('--ratios', metavar='CSV', help='scale table: ratios_73MHz.csv instead of this version\'s measure record')
    s.add_argument('--old-scale-table', metavar='CSV', help='scale table: the factor in the catalogues\' pixels (default tables.SCALE_TABLE)')
    s.add_argument('--band-vector-start', metavar='CSV', help='bandvec: start vector (default tables.BAND_VECTOR)')
    s.add_argument('--band-vector', metavar='CSV', help='fluxcorr table: this band vector instead of the bandvec record')
    s.add_argument('--stagea', metavar='GLOB', help='fluxcorr table: stage-A CSVs instead of the stageA record (quote the glob)')
    s.add_argument('--scale-candidate', metavar='CSV', help='fluxcorr table: per-LST rescale from this scale table (default '
                                                            'tables.SCALE_TABLE)')
    k = p.add_argument_group('template-k')
    k.add_argument('--trials', metavar='DIR|GLOB', help='summary: stored trials (default tables.K_RECORDS[tier]; sub-LST: the '
                                                        'inject record)')
    k.add_argument('--catalog-dir', metavar='DIR', help='sub-LST inject: the LST catalogue directory (kcal_tenmin_inject.py argv 1)')


def default_out(args):
    from exopipe.phase3.config import CFG
    return CFG.calibrate.OUT_ROOT


def _module(cls):
    import importlib
    from exopipe.phase3.lib.calibrate import common as K
    return importlib.import_module(f'exopipe.phase3.lib.calibrate.{K.CLASSES[cls][0]}')


def _stages(args, cls, tier):
    from exopipe.phase3.lib.calibrate import common as K
    known = list(K.CLASSES[cls][2][tier])
    if args.stages:
        want = [x.strip() for x in args.stages.split(',') if x.strip()]
        bad = [x for x in want if x not in known]
        if bad:
            raise ValueError(f'unknown sub-stage(s) {bad} of {cls} {tier}; known: {", ".join(known) or "(none)"}')
        sel = [x for x in known if x in want]
    else:
        sel = list(known)
    if args.from_stage:
        if args.from_stage not in sel:
            raise ValueError(f'--from {args.from_stage}: not among {sel}')
        sel = sel[sel.index(args.from_stage):]
    if args.until_stage:
        if args.until_stage not in sel:
            raise ValueError(f'--until {args.until_stage}: not among {sel}')
        sel = sel[:sel.index(args.until_stage) + 1]
    return known, sel


def _new_version(cls):
    from exopipe.phase3.lib.calibrate import common as K
    if cls in ('scale', 'fluxcorr'):
        from exopipe.phase3.lib.calibrate.fluxscale import next_version
        return next_version(cls)
    return K.utcstamp()


def _version(args, run, cls, tier, action, known, sel):
    """The version this invocation works on (see --version)."""
    from exopipe.phase3.lib.calibrate import common as K
    from exopipe.phase3.core.record import Refusal
    if args.version:
        return K.check_version(args.version)
    if action == 'derive':
        if sel[0] == known[0]:
            return _new_version(cls)
        prev = known[known.index(sel[0]) - 1]
        v, _ = K.newest_done(run.out_root, cls, tier, f'derive-{prev}')
        if v is not None:
            return v
        explicit = {'build': args.curves, 'table': args.ratios if cls == 'scale' else (args.stagea and args.band_vector),
                    'summary': args.trials or tier == 'deep'}.get(sel[0])
        if explicit:                    # the stage reads explicit inputs (e.g. production's own derivation): a new version
            return _new_version(cls)
        raise Refusal(f'derive --from {sel[0]}: no version of {cls} {tier} has a done {prev} record under {run.out_root} '
                      f'(give --version, or the explicit inputs of {sel[0]})')
    if args.candidate:
        return 'cand-' + K.sha256(args.candidate)[:12]
    if action == 'verify':
        return 'installed'
    last = known[-1] if known else None
    v, _ = K.newest_done(run.out_root, cls, tier, f'derive-{last}') if last else (None, None)
    if v is None:
        raise Refusal(f'{action}: no done derive-{last} record of {cls} {tier} under {run.out_root} (give --version, or '
                      f'--candidate)')
    return v


def _print_plan(ctx, plan, stage=None):
    import json
    print(json.dumps({'table': ctx.cls, 'tier': ctx.tier, 'action': ctx.action, 'version': ctx.version, 'out': ctx.out,
                      'stage': stage, 'plan': plan}, indent=1, default=str))


def _one(args, run, cls, tier, action):
    from exopipe.phase3.lib.calibrate import common as K
    from exopipe.phase3.core.record import Refusal
    if tier not in K.CLASSES[cls][1]:
        raise Refusal(f'calibrate {cls}: tier {tier} not supported (tiers {", ".join(K.CLASSES[cls][1])})')
    known, sel = _stages(args, cls, tier) if action == 'derive' else (list(K.CLASSES[cls][2][tier]), [])
    if action == 'derive' and not sel:
        raise Refusal(f'calibrate {cls} --tier {tier} derive: no derivation in this unit (spec B6: no trials source)')
    version = _version(args, run, cls, tier, action, known, sel)
    ctx = K.Ctx(args, run, cls, tier, action, version)
    mod = _module(cls)
    if action == 'derive':
        fns = {'bright_parent': {'inputs': mod.derive_inputs if hasattr(mod, 'derive_inputs') else None,
                                 'curves': getattr(mod, 'derive_curves', None), 'build': getattr(mod, 'derive_build', None)},
               'fluxscale': {'measure': getattr(mod, 'scale_measure', None), 'stageA': getattr(mod, 'stagea', None),
                             'bandvec': getattr(mod, 'bandvec', None),
                             'table': getattr(mod, 'scale_table' if cls == 'scale' else 'fc_table', None)},
               'template_k': {'inject': getattr(mod, 'inject', None), 'summary': getattr(mod, 'summary', None)}}[K.CLASSES[cls][0]]
        for st in sel:
            n_before = len(run.units)
            plan = fns[st](ctx)
            if run.dry_run:
                _print_plan(ctx, plan, st)
                continue
            new = run.units[n_before:]
            if new and new[-1].status not in ('done', 'skipped-valid'):
                ctx.say(f'stopped after {st}: {new[-1].status} ({new[-1].reason[:800]})')
                break
        return
    if action == 'verify':
        if hasattr(mod, 'verify') and cls == 'template-k':
            plan = mod.verify(ctx)
        else:
            plan = K.verify(ctx, mod.verify_targets(ctx))
    else:
        plan = getattr(mod, action)(ctx)
    if run.dry_run:
        _print_plan(ctx, plan)


def run(args, run):
    from exopipe.phase3.lib.calibrate import common as K
    from exopipe.phase3.core.record import Refusal
    if args.table in K.NOT_IMPLEMENTED:
        raise Refusal(f'calibrate {args.table}: not implemented in this unit yet (plan/impl/calibrate.md section 6); the '
                      f'installed file is pinned in config section tables')
    if args.table == 'all':
        if args.action != 'verify':
            raise Refusal("calibrate all: only 'verify' (every installed table)")
        for cls, tier in (('bright-parent', 'deep'), ('bright-parent', 'sublst'), ('scale', 'deep'), ('fluxcorr', 'deep'),
                          ('template-k', 'deep'), ('template-k', 'sublst'), ('template-k', 'snapshot')):
            try:
                _one(args, run, cls, tier, 'verify')
            except Refusal as e:
                u = run.unit(K.unit_name(cls, tier, 'verify', 'installed'))
                u.status, u.reason = 'refused', str(e)
                print(f'REFUSED {cls} {tier}: {e}')
        return 0 if run.dry_run else None
    _one(args, run, args.table, args.tier, args.action)
    return 0 if run.dry_run else None
