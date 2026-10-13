"""exopipe3 jupiter - the Jupiter reflection program (Jovian decametric bursts reflected off the Galilean moons; Gregg
2026-10-04: a dedicated Jupiter page for nights with Jupiter above 30 deg). Cross tier: reads the snapshot tier's flowed cubes
and the sub-LST tier's window stacks; every product row carries its own cadence (10s, 1min, 10min).

    exopipe3 jupiter measure --cells 05h_2024-12-29 --out /lustre/.../parity/jupiter/new --work /fast/... [--sublst-root RES]
    exopipe3 jupiter measure --cells @cells.txt --stages fit,assign,test --resume
    exopipe3 jupiter v2 --cells 05h_2024-12-29 --stages prepare,anchor,render --out P --work W
    exopipe3 jupiter flag-backfill --stages window,run --shard lwacalim01      (one shard; many nodes: exopipe3 run submit)
    exopipe3 jupiter flag-backfill --stages verify
    exopipe3 jupiter serve --stages phase1,v2 [--restart|--stop] [--port-offset 1000]
    exopipe3 jupiter ephem --dates 2025-11-02          (on a host that reaches JPL)
    exopipe3 jupiter occurrence | quarantine --cells C --qdir Q

Use: survey (intermittent, DESIGN 12.2) - run on demand over Jupiter-window cells; not part of the per-hour chain. Unit of work:
a Jupiter cell <LST>_<YYYY-MM-DD> (the night-cell grammar; --cells patterns match the Jupiter window jupiter.WINDOW_CSV, the
cells with Jupiter >= 30 deg at mid-hour); flag-backfill: a shard (node); occurrence: one run-wide unit; serve: a port.
During the refactor window every engine runs IN PLACE, unchanged (clv15/phase3_jupiter_measure.py, the V2 code tree,
exopipe_ops/jupiter_server.py, phase0/jup_ephem_fetch.py, exopipe/phase1/jupiter_flag_backfill.py): products and layout are
today's, no renames. A run into any --out other than the production root runs every engine child under the write guard
(core/guard.py) rooted at --out and --work.

Actions and stages (--stages a,b or --from/--until within the action; default every stage, except flag-backfill = window,run
and v2 = all but refcheck):
  measure        extract, fit, assign, test       units measure.<stage>.<cell>; --workers N cells at once (default 6)
  v2             geometry, masters, prepare, provenance, anchor, refcheck, metrics, render, summary   units v2.<stage>.<cell>
  occurrence     fetch, validate, digitize, maps, figures, lookup, test   (production root only; parity = verify)
  ephem          fetch                            units ephem.<date>
  flag-backfill  window, run, verify              units flag.window, flag.run.<node>, flag.verify
  serve          phase1 (8800), v2 (8801)         units serve.<port>
  quarantine     renders                          units quarantine.<cell>

Inputs
  measure   per cell: sub-LST admission tables, window stacks + arm-A LOO residuals, window Masters (windows without the night's
            own stack, JUPMASK only), x2 residuals (15 subbands + Red/Green/Blue) under jupiter.SUBLST_ROOT; the flowed cubes
            under jupiter.CUBES_ROOT; the Horizons cache jupiter.EPHEM_DIR; for assign the frozen deep co-add records of tier
            rule 4 (phase3_tier_rules); data_decisions.csv (gate J2).
  v2        the cell's measure products, the V2 tree (<ROOT>/v2), deep weights (jupiter.DEEP_WEIGHTS), the occurrence results
            and the disclosure record. flag-backfill: the Phase-1 archive (paths.PHASE1_ROOT). serve: the products / site.
Outputs (production layout under --out = jupiter.ROOT; records <out>/_records/jupiter/<unit>.json; product classes [..])
  measure   <out>/products/<cell>/: meas_<tier>.csv [jupiter_meas; the node-local crops <work>/<cell>/crops/ it names:
            jupiter_crop], cutouts_<tier>_<band>_<S>.npz [jupiter_cutouts], fits_<tier>.jsonl [jupiter_fits],
            assign_<tier>.csv + moons_<tier>.json [jupiter_assign], CONTENT_TEST.txt [jupiter_cell]; logs
            <out>/products/logs/<stage>_<cell>.log and run.log lines (FAILED / DONE / ALL-DONE) as the old launcher.
  v2        <out>/v2/nightgeom [jupiter_geom_raw, jupiter_geom], masters [jupiter_v2_masters], masters/derived [jupiter_v2_root],
            provenance [jupiter_crop_archive], anchor [jupiter_blue_anchor], integrate/inputs [jupiter_refcheck],
            integrate/<cell> [jupiter_metrics], site/<cell>/renders [jupiter_site_render], site/<cell>/CONTENT_TEST.txt
            [jupiter_site_summary]. occurrence: <out>/v2/occurrence [jupiter_occurrence]. quarantine: site/<cell>/<qdir>
            [jupiter_render_quarantine]. ephem: <out>/ephem/jupiter_ephem_<date>.csv [jupiter_ephem].
  flag      STATE (jupiter_flag.STATE_DIR, or <out>/jupiter_flag_backfill_state) window.csv [jupiter_window], shard_<node>.csv,
            summary_<node>.csv [csv_table], run_<node>.log, VERIFY_<UTC>.txt [jupiter_flag_verify]; copies under
            <out>/phase1_flagged/<LST>/<date>/<Run>/<sb>/ [jupiter_flagged_copy].
  serve     <out>/_served/jupiter_<port>.json [jupiter_served_page] (the served content test runs on it).
Failure modes (exit 2 = refused, the gate in the record; 3 = nothing to do; 1 = error; a refusal writes nothing)
  J1 a literal cell outside the Jupiter window (--waive J1 --ruling); J2 an excluded (LST, scope, night) with inputs; J3 no input
  (exit 3); J4 a residual without JUPMASK; J7 a baked input (FLUXBAKE: the measure divides by FLUXDC - every current stack is
  baked, so no new night can be measured until the owner fixes it); J8 CASA not importable; J9 a frozen deep record missing;
  J10 stale fits (--fresh --supersede TAG); fit on another node than the crops; J12 JFQ_* / P3_COADD_RUN / P3_DEEP_SEARCH_STATE /
  EXOPIPE_JUPITER_FLAG_QA set; J13 --scopes into production; J14 a production --out while paths.PRODUCTION_WRITES is False; a
  config key that disagrees with the in-place module literal, or an override the engine would ignore; J20 a v2 cell other than
  05h_2024-12-29; J22-J27 V2 refusals; a v2 stage that cannot be redirected into a non-production --out (masters, metrics,
  summary); J30 untested backfill code; J31 no APPROVED file for a copying shard; J32 a shard log without its exit line; J33 the
  copy budget; J34 the shard time limit; J35 verify failed; J40 a port held by another process / old pid alive after SIGTERM.
"""
import importlib

SUMMARY = 'the Jupiter reflection program'
TIERS = None
SELECTORS = ('cells', 'exclude_cells', 'scopes', 'stokes')
CONFIG = ('jupiter', 'jupiter_measure', 'jupiter_v2', 'jupiter_occurrence', 'jupiter_flag', 'jupiter_serve', 'paths', 'exclusions',
          'instrument', 'runner')
WRITES = True
ACTIONS = ('measure', 'v2', 'occurrence', 'ephem', 'flag-backfill', 'serve', 'quarantine')
_MODULE = {'measure': 'measure', 'v2': 'v2', 'occurrence': 'occurrence', 'ephem': 'ephem', 'flag-backfill': 'flag',
           'serve': 'serve', 'quarantine': 'quarantine'}


def add_arguments(p):
    p.add_argument('action', choices=ACTIONS, help='what to run (stages per action above)')
    g = p.add_argument_group('stages (until cli.py adds them centrally)')
    g.add_argument('--stages', help='comma list of the action\'s stages, run in the action\'s order')
    g.add_argument('--from', dest='stage_from', metavar='STAGE', help='first stage of a contiguous range')
    g.add_argument('--until', dest='stage_until', metavar='STAGE', help='last stage of a contiguous range')
    g.add_argument('--supersede', metavar='TAG', help='with --fresh: move existing outputs aside to _superseded/TAG/ with a '
                                                      'manifest (stale fits J10, existing V2 derived roots J22); never deletes')
    g = p.add_argument_group('jupiter')
    g.add_argument('--sublst-root', metavar='DIR', help='shorthand for --set jupiter.SUBLST_ROOT=DIR (the sub-LST input root, '
                                                        'e.g. the pre-bake reserve for the 2026-10-05 products); refused into a '
                                                        'production --out')
    g.add_argument('--labels', help='v2 render: product roots (default jupiter_v2.ROOT_LABELS)')
    g.add_argument('--hz-grid', metavar='G0,G1,C0,C1', help='v2 geometry: Horizons 1-min grid and 10-s check span, UTC HH:MM '
                                                            '(default jupiter_v2.HZ_GRID[date])')
    g.add_argument('--dates', help='ephem: nights to fetch, YYYY-MM-DD[,...]')
    g.add_argument('--nodes', help='flag-backfill: the shard nodes, comma list (default site JUPITER_FLAG_NODES)')
    g.add_argument('--shard', metavar='NODE', help='flag-backfill run: the shard to run (default this host); the runner item')
    g.add_argument('--budget-gb', type=float, help='flag-backfill run: total copy budget, split over the nodes '
                                                   '(default jupiter_flag.LIMIT_GB_TOTAL; ruled <= 2 TB)')
    g.add_argument('--max-hours', type=float, help='flag-backfill run: per-shard time limit (default jupiter_flag.MAX_HOURS; '
                                                   'ruled <= 12 h)')
    g.add_argument('--no-copy', action='store_true', help='flag-backfill run: records only, nothing copied (parity)')
    g.add_argument('--flagged-root', metavar='DIR', help='flag-backfill verify: the copy tree to verify, read only '
                                                         '(default <out>/phase1_flagged)')
    g.add_argument('--port-offset', type=int, default=0, help='serve: add to jupiter_serve.PORTS (0, or >= '
                                                              'jupiter_serve.PARITY_PORT_OFFSET_MIN for parity servers)')
    g.add_argument('--restart', action='store_true', help='serve: exact-pid restart of this step\'s server')
    g.add_argument('--stop', action='store_true', help='serve: exact-pid stop of this step\'s server')
    g.add_argument('--qdir', help='quarantine: directory inside the site cell (default quarantine_superseded_<UTC date>)')


def default_out(args):
    from exopipe.phase3.config import CFG
    return CFG.jupiter.ROOT


def _check_ruled_budget(args, roots):
    """--budget-gb / --max-hours carry Gregg's limits (APPROVED-PREAMBLES-1-4-20261007): above them only with a ruling."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core.record import Refusal
    F = CFG.jupiter_flag
    given = {r.split('=', 1)[0].strip() for r in (getattr(args, 'ruling_for', None) or [])}
    for flag, val, lim, key in (('--budget-gb', args.budget_gb, F.LIMIT_GB_TOTAL, 'jupiter_flag.LIMIT_GB_TOTAL'),
                                ('--max-hours', args.max_hours, F.MAX_HOURS, 'jupiter_flag.MAX_HOURS')):
        if val is not None and val > lim and key not in given:
            raise Refusal(f'{flag} {val:g} exceeds the approved limit {lim:g} (APPROVED-PREAMBLES-1-4-20261007); a larger value '
                          f'needs a new approval (--ruling-for jupiter_flag.LIMIT_GB_TOTAL=<ledger ID> / MAX_HOURS)')


def run(args, run):
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.jupiter import common as K
    from exopipe.phase3.lib.jupiter import mirrors
    if args.sublst_root:
        CFG.set('jupiter.SUBLST_ROOT', args.sublst_root, source='--sublst-root')
    K.check_environment()
    mirrors.check_overrides(CFG)
    roots = K.Roots(args, run.out_root, args.action, dry_run=run.dry_run)
    K.check_pinned_overrides(roots)
    _check_ruled_budget(args, roots)
    stages = K.select_stages(args.action, args)
    run.meta = {'action': args.action, 'stages': stages}
    mod = importlib.import_module(f'exopipe.phase3.lib.jupiter.{_MODULE[args.action]}')
    return mod.run(args, run, roots, stages)
