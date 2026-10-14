"""exopipe3 pages - review pages of every tier: panels, sidecars, page indexes, raw judging queues, TESS coverage, the
review_dir merge, the snapshot filter catalogue. Per hour (add one epoch's candidates: incremental) or full.

    exopipe3 pages --tier deep --run P3S5 --deep-parent-dir /fast/gh/main/phase3/p3s5_parents_union_<UTC> --out /fast/gh/p
    exopipe3 pages --tier deep --run P3S5 --epochs 04h_2025-10-14 ...      # per hour: unchanged pages kept (fingerprint skip)
    exopipe3 pages --tier deep --run P3S5 --sets V_wb,V_sb --full ...      # re-render every page of the Stokes V sets
    exopipe3 pages --tier deep --run P3S5 --cells 04h_Full --stages build  # re-render the pages with a panel in 04h_Full
    exopipe3 pages --tier sublst --lsts 14h --legacy-root ops=<frozen>/exopipe_ops --legacy-root clv15=<frozen>/clv15 ...
    exopipe3 pages --tier snapshot --run Run_SNAPB2R_20261007 --from-run <copy of the chain run> --out /fast/gh/p

Unit of work (one run record each, <out>/_records/pages/<unit>.json):
  deep      <label>:tess, <label>:input:<K>, <label>:build:<K>, <label>:merge (+ <label>:strat on request; <label>:stagein with
            --from-run); K = page set V_wb V_sb I_wb I_sb V_fwb V_fsb I_fwb I_fsb. Item for `run`: the run label (a run-wide
            step after catalog in the chains 'deep-publish' and 'epoch').
  sublst    <LST>:prep, <LST>:input, <LST>:build, <LST>:index:<K>, <LST>:place; item kind lst (--lsts), or the LSTs of --epochs.
  snapshot  <run>:build, <run>:filtercat (+ <run>:stagein with --from-run); item: the chain run (--run Run_SNAP...).

Inputs (written by catalog, search, masters; read in place, never renamed):
  deep      <U>=<out>/Run_<label>: source_catalogs_<K>/ (survivors, members, E_params.json), source_catalogs/ (union survivors;
            localizations.csv + localization_offsets.csv written by catalog --stages localize), master_catalog.csv,
            v_deep_images.csv, v_deep_ledger.csv, frame_times.csv, iresid/ vresid/ v110_products/ parent_catalogs/ links; the
            masters through pages.MASTER_ROOTS_MAP; the parent snapshot of the pass (--deep-parent-dir); XM sample CSVs.
  sublst    <CFG.paths.SUBLST_ROOT>/compare_<LST>/new_B/catalog/ (catalogues, localizations_noveto, pertie, rejected record),
            window stacks and arm-A residuals, search_B residuals, parent_catalogs_<LST>.
  snapshot  the chain run (source_catalogs, detections, queue.txt, chain_summary.json), the fit store, the snapshot fleet outputs
            (CFG.paths.SNAPSHOT_FLEET_OUT), the frozen masters (pages_snapshot.MASTERS_ROOT).
Outputs (product classes; content tests in lib/pages/tests.py):
  review_page           candidates/<I|K|S>/<page id>/rank<N>_<LST>_<date>_<band>.png (+ .page.json sidecar, deep)
  page_index            source_catalogs/<band>_pages_index_<S>.csv (singles), <band>_pages_index_<K>.csv (fork sets)
  review_queue          judgements/queue_p3s3_<K>.txt (singles; serve sections it), queue_p3s3_<K>_raw.txt (fork);
                        snapshot queue_<S>.txt, queue_<S>_<tier>.txt
  pages_input           source_catalogs/pages_input_<K>.csv (+ tenmin_selection_<K>.csv, sub-LST)
  astrometry_figure     localization_figs/<identity>.png (deep; drawn inside build_i14)
  tess_coverage         tess_coverage.csv (deep)
  survivors_review_dir  the review_dir column of source_catalogs/<band>_sources_survivors.csv (deep merge; serve reads it)
  sublst_page_run, snapshot_page_run, filter_catalog, strat_queue, pages_run_copy (--from-run stage-in)
Failure modes (exit 2 = refused, the gate named in the record): P1 no --run; P2 a stage's previous-stage record missing / not
  done / its outputs changed; P3 MIRROR drift (a constant inside an in-place module differs from config); P4 localization tables
  missing; P6 pages input failed; P7 a page shard failed; P9 another pages job holds the run directory; P10 review_dir merge
  refused; P11-P17 sub-LST (localizations, tie table, rejected record, selection, index, frozen code, place); P18-P22 snapshot
  (flow mode, duplicate or missing panel, run content test, filter catalogue); a content test failing. Exit 3 = every unit had
  nothing to do (e.g. a set with no pages-input row). TESS never refuses (recorded NO-INPUT: 'coverage not computed').
Per-hour vs intermittent: per hour (DESIGN 12.1, chain 'epoch'): --epochs <LST>_<date> after catalog --incremental; the deep
  build is incremental by construction (build_i14 keeps every page whose fingerprint is unchanged and renders new/changed pages),
  unchanged units are kept by the record (deep build units in the shared layout: all together or none, one pass start for every
  set rendered; a changed --deep-parent-dir / roots map, or a partial record, is never kept). Intermittent: the full pass over a
  rebuilt catalogue (--full forces every page).
Production code runs IN PLACE during the refactor window through core.legacy.run (build_i14_pages_std7.py, std7_panel.py,
  p3s3_tier_pages_input.py, p3s3_merge_review_dir.py, tess_coverage.py, deep_stratified_queue.py, sublst_pages_prep.py,
  p3t_pages_input.py, build_tenmin_pages.py, p3t_pages_index.py, phase3_snapshot_pages.py, snapshot_rechain_content_test.py,
  snapshot_filter_catalog.py); the deep singles pages input is a port of the p3s3_pass.sh heredoc. M14 (per-set panel
  directories, keyed review_dir merge, fork-index names) is carried behind pages.PANEL_DIR_LAYOUT / pages.DEEP_MERGE (default:
  today's shared layout). Interface: exopipe_ops/INTERFACE_REQUIREMENTS.md (pages/judging sections).
"""
import importlib

SUMMARY = 'review pages (panels, sidecars, indexes, raw queues), TESS coverage, review_dir merge; per hour or full'
TIERS = ('deep', 'sublst', 'snapshot')
SELECTORS = ('tier', 'run_label', 'cells', 'lsts', 'stokes', 'epochs')
CONFIG = ('pages', 'pages_deep', 'pages_sublst', 'pages_snapshot', 'runner', 'site')
WRITES = True


def _tier_mod(tier):
    return importlib.import_module(f'exopipe.phase3.lib.pages.tier_{tier}')


def add_arguments(p):
    g = p.add_argument_group('pages (every tier)')
    g.add_argument('--stages', help='stages to run, in the tier\'s fixed order (default pages.STAGES[tier]: deep '
                                    'tess,input,build,merge (+strat on request); sublst prep,input,build,index,place; '
                                    'snapshot build,filtercat)')
    g.add_argument('--from', dest='from_stage', metavar='STAGE', help='start at this stage')
    g.add_argument('--until', dest='until_stage', metavar='STAGE', help='stop after this stage')
    g.add_argument('--sets', help='page sets (deep, sub-LST; default pages.PAGE_SETS[tier]); catalogues are never narrowed')
    g.add_argument('--shards', type=int, help='processes per page set (default pages.DEEP_PAGE_SHARDS / SUBLST_PAGE_SHARDS / '
                                              'SNAP_PAGES_NPROC)')
    g.add_argument('--only-pages', metavar='FILE', help='deep: re-render exactly these page ids; no index or queue written '
                                                        '(P3_PAGES_ONLY_FILE, build_i14:49-54)')
    g.add_argument('--full', action='store_true', help='deep: render every page (the set\'s sidecars are removed first, so '
                                                       'the fingerprint skip keeps nothing)')
    g.add_argument('--from-run', metavar='DIR', help='parity/scratch: build the run under --out from this existing run '
                                                     '(deep: small files copied, read-only trees linked; snapshot: the chain '
                                                     'run copied)')
    g.add_argument('--legacy-root', action='append', default=[], metavar='NAME=PATH',
                   help='refactor window: the code root of an in-place call (ops, clv15); sub-LST needs a verified frozen '
                        'tree (P16); default the site LEGACY_CODE_ROOTS')
    for t in TIERS:
        _tier_mod(t).add_arguments(p)


def default_out(args):
    from exopipe.phase3.config import CFG
    tier = getattr(args, 'tier', None) or TIERS[0]
    if tier == 'deep':
        return CFG.paths.TRANSIENTS_ROOT
    out = CFG.pages.PRODUCTION_OUT[tier]
    return out.format(run=getattr(args, 'run_label', None) or 'RUN') if '{run}' in out else out


def run(args, run):
    from exopipe.phase3.lib.pages import common
    tier = args.tier
    if tier != 'deep' and (args.only_pages or args.full or args.cells):
        raise common.Refusal('--only-pages, --full and --cells are deep-tier options')
    if tier != 'deep' and getattr(args, 'stokes', None) and tier == 'snapshot':
        raise common.Refusal('--stokes does not narrow the snapshot pages (one chain run renders both Stokes)')
    with common.scrubbed_environ():
        return _tier_mod(tier).run(args, run)
