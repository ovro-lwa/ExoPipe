"""exopipe3 forced - forced photometry at DETECTED positions and the forced-photometry review products: the deep catalogue
forced flux (catflux), the pages of the deep HWJ / 10 pc evaluation (panels), the snapshot forced event panels.

During the refactor window every stage runs the production script IN PLACE through core.legacy.run (DESIGN 11.1): results
identical to today's. The deep HWJ / 10 pc MEASUREMENT (frame list, staging, 32 shards, evaluation) is
`exopipe3 photometry --tier deep` (refactor chat decision 6); this step renders its evaluation.

Families and stages
  B  deep catflux (default for --tier deep): forced flux at each served source's CATALOGUE position on the run's own search
     frame of its best detection - the forced_jy primary-flux columns of every deep page (p3s3_forced_phot.py in place,
     p3s3_pass.sh:100-101). Runs between `catalog --until fork` and `catalog --from union` (catalog gate D-G22).
  A  deep panels (--stages panels): review pages of the rows photometry's evaluation lists above the control thresholds
     (deep_forced_panels.py in place; isolated page run; served on 8798/8799 by serve).
  C  snapshot (--tier snapshot): event panels of one Stokes and one cadence per run - build, fit (Gaussian at or above 7
     sigma, template below), tie, loc (review host), assoc, render, verify (review host).
  sub-LST (--tier sublst): no production forced chain exists - ends NO-INPUT (exit 3).

Unit of work / item kinds: deep catflux = the run directory (all its catalogues); deep panels = one page run; snapshot = one
(Stokes, cadence) run NAME (units C_<stage>_<NAME>). Per hour (DESIGN 12.1, chain 'epoch'): --epochs <LST>_<date>
  deep      catflux over the run's catalogues as `catalog --incremental` left them (every row re-measured on the frame of its
            best detection; needs --run LABEL - the 'epoch' chain passes --run {deep_run}, runner.CHAIN_VARS);
  snapshot  one run per Stokes and cadence (default I,V x 10s,1min) named forced_snapshot.EPOCH_NAME, built from that hour's
            rows of the evaluations - partial, not servable; a combination without an event ends no-input.
Survey (DESIGN 12.2): catflux after a full publish pass; panels after a photometry evaluation; one snapshot run per (Stokes,
cadence) over the full evaluations.

Inputs
  catflux   <out> = the deep run directory (the catalog step's --out; default <paths.TRANSIENTS_ROOT>/Run_<LABEL>):
            source_catalogs_[VI]*/*_sources_survivors.csv, the run's own vresid/ and iresid/ frames, and for Stokes I rows
            without the run's frame the cell's subtracted/ frame under paths.COADD_SB_ROOT (B-06 open; host of that root).
  panels    the rows: --deep-eval-run DIR (photometry's run, record _records/photometry/evaluate.json -> its
            targets_above_threshold.csv) or --deep-targets-csv FILE; the searched run <paths.TRANSIENTS_ROOT>/Run_<LABEL>
            (--run, required: D7) whose frames, parents and ledgers the page run links; the target lists.
  snapshot  the photometry step's evaluations (--snap-eval-sb/--snap-eval-wb, default the served generation 20261007),
            stage-1 files and cutouts, the snapshot fleet's noise tables (paths.SNAPSHOT_FLEET_OUT), the frozen gen-4 masters,
            the master catalogues (tie references), the radio catalogues (association), the zone recomputation (render).
Outputs (production names and layout unchanged; records at <out>/_records/forced/<unit>.json)
  catflux   the survivors catalogues gain forced_jy, forced_pixel_jy, forced_rms_jy, forced_method, forced_frame (class
            catalog_forced_columns), rewritten in place in <out>.
  panels    <out> = the page run (default <paths.TRANSIENTS_ROOT>/<--deep-panels-run>): links to the searched run, page PNGs,
            source_catalogs/{pages_input,*_pages_index,*_sources_survivors}, radio_posmap.json, radio_render_record.jsonl,
            tess_coverage_forced.csv, judgements/queue_forced.txt only when the content test passes (class forced_pages).
  snapshot  <out> = the work root (default forced_snapshot.WORK_ROOT); work run <out>/<NAME>: queue.txt and the build tree
            (forced_event_build), forced_fits.csv (forced_fits), forced_ties.csv (forced_ties), forced_assoc_identity.csv
            (forced_assoc), missing_panels.csv and the served tree (forced_event_panels); review copy <review root>/<NAME>
            with source_catalogs/localizations.csv (forced_localization) and the served tree.
Failure modes (exit 2 = refused, the gate named in the record; exit 3 = nothing to do; exit 1 = error)
  G1 no --run / searched run missing (deep); G17-G20 panels refusals and page content test (no queue written); G21 catflux
  failure or the run's own frames missing (the publish chain stops, D10); G22 evaluation incomplete; G23 a Full row; G24 no
  cadence / two cadences; G25-G26 events not panellable (listed in skipped_events.csv; none panellable refuses); G27 fit;
  G28 tie reconstruction; G29 localization figures; G30-G31 association inputs / content test; G32-G33 render (association
  missing, panels missing); G35 a snapshot run name used twice; G36 render-host tests Y (yellow numbers) and Z2 (served
  zones); G37 an override of a value the in-place code carries as a literal (forced_*, fitting.*, astrometry.*); a stage on
  the wrong host (catflux and panels: the host holding the run's node-local data; loc and verify: the review host); a stage
  whose previous stage has no done record; a review copy under a production root while paths.PRODUCTION_WRITES is False.
"""
import importlib
import sys

SUMMARY = 'forced photometry at detected positions (catalogue forced flux), deep forced pages, snapshot forced event panels'
TIERS = ('deep', 'snapshot', 'sublst')
SELECTORS = ('tier', 'cells', 'stokes', 'run_label', 'epochs')
CONFIG = ('forced', 'forced_panels', 'forced_catflux', 'forced_snapshot', 'forced_assoc', 'fitting', 'astrometry',
          'exclusions', 'paths', 'runner')
WRITES = True
TIER_MODULES = {'deep': 'exopipe.phase3.lib.forced.tier_deep', 'snapshot': 'exopipe.phase3.lib.forced.tier_snapshot'}


def add_arguments(p):
    g = p.add_argument_group('stages (every tier)')
    g.add_argument('--stages', metavar='LIST',
                   help='stages to run, in pipeline order whatever the order given. deep: catflux (default) or panels '
                        '(one per invocation: they write into different directories); snapshot: '
                        'build,fit,tie,loc,assoc,render,verify (default)')
    g.add_argument('--from', dest='stage_from', metavar='STAGE', help='first stage of the selected list')
    g.add_argument('--until', dest='stage_until', metavar='STAGE', help='last stage of the selected list')
    for mod in TIER_MODULES.values():
        importlib.import_module(mod).add_arguments(p)


def default_out(args):
    """The production directory the selected stages write into (catflux: the deep run dir; panels: the page run dir;
    snapshot: the work root). Refused for writing while paths.PRODUCTION_WRITES is False."""
    tier = getattr(args, 'tier', 'deep')
    if tier == 'sublst':
        return '/tmp/exopipe3_forced_sublst'
    mod = importlib.import_module(TIER_MODULES[tier])
    if tier == 'deep':
        from exopipe.phase3.lib.forced import common
        try:
            stages = common.selected_stages(args, 'deep')
        except ValueError:
            stages = ['catflux']
        return mod.default_out(args, stages)
    return mod.default_out(args)


def run(args, run):
    from exopipe.phase3.core import record
    from exopipe.phase3.lib.forced import common
    if args.tier == 'sublst':
        raise record.NoInput('tier sublst: no production forced chain (sublst_forced_phot.py / sublst_forced_eval.py exist '
                             'but no launcher runs them; ledger 1513/1518 notes; spec forced.md 1.1, Q15)')
    common.check_overrides()
    try:
        rc = importlib.import_module(TIER_MODULES[args.tier]).run(args, run)
    except ValueError as e:
        print('error: %s' % e, file=sys.stderr)
        return record.EXIT_ERROR
    if rc is None and run.dry_run:
        return record.EXIT_DONE          # the plan was printed and no gate refused it
    return rc
