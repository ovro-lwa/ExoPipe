"""exopipe3 status - what is done, stale, missing, due or breaking a ruling, for every tier; QA tables and hub status
pages; every list it prints that names items is written as an item list usable as --cells/--epochs/--lsts @FILE.

    exopipe3 status                                          # the session report (CLAUDE.md section 1), tier deep
    exopipe3 status --coverage --gaps --emit gaps.txt        # deep coverage; the gap cells as an item list
    exopipe3 status --gaps --emit-kind epoch --emit hours.txt    # the gap hours (<LST>_<date>) for the per-hour chain
    exopipe3 status --rebuild-due --emit due.txt             # REPORT: cells whose admitted time grew >= 50% (C-10)
    exopipe3 status --stale record --records /fast/gh/refactor_scratch/parity     # stale run records
    exopipe3 status --stale search_attempt,master            # attempts / masters an exclusion edit made stale
    exopipe3 status --exclusion-conformance --tier snapshot  # served rows on excluded cell-bands
    exopipe3 status --served-diff P3S4 P3S5                  # what changes for the served list between two runs
    exopipe3 status --cell-load --run P3S5                   # per (Stokes, LST, night, band) load (contract step 50)
    exopipe3 status qa --stages solar,noise --out /lustre/gh/main/exopipe_refactor_parity/status
    exopipe3 status conformance --out P;  status pages --out P;  status ledger --out P;  status register A.csv ... --out P
    exopipe3 status eligibility --tier snapshot --no-store;  status batch-list --tier snapshot --base B --list-out L

Use: INTERMITTENT and on demand (DESIGN 12.2: status coverage and --rebuild-due are survey steps); `status --coverage
--epochs 04h_2025-10-14` reports one hour after the per-hour chain. Reads every step's run records, never their
markers.

ACTIONS and stages (`--stages a,b` picks stages of the action; unit of work = one stage, record
<out>/_records/status/<action>_<stage>_<tier>[_<8 hex of the selected items>].json):
  report (default)  summary (default when no report flag is given), coverage, gaps, exclusion-conformance, violations,
                    stale, served-diff, cell-load, rebuild-due - each chosen by its flag; a flag alone runs that stage only
  qa                bandpass, qa-table, solar, provenance, beam, noise, outliers (default qa-table,solar); tierless
  conformance       membership (deep masters against the membership rulings; whole tree)
  pages             nights, data-processing (hub status pages; --check: only the night-page content test)
  ledger            the Phase 1/2 inventory -> processing ledger (whole survey only)
  register          the register of ratified steps from audit-slice CSVs (positional AUDIT_CSV ...)
  eligibility       snapshot cells' eligibility for the batch chains (tier snapshot)
  batch-list        the next snapshot batch's cell list (tier snapshot; --base and --list-out required)

Inputs:  data_decisions.csv (core/exclusions, every stage; missing = refusal), the processing ledger, the deep working
         tree paths.COADD_SB_ROOT (<cell>/coadd_final/{weights.csv,I_deep_Taper_R075_<s>.fits,...}, subtracted/),
         Phase 1/2 trees (paths.PHASE1_ROOT, PHASE2_ROOT), the archive generation (paths.ARCHIVE_RUN), published search
         attempts (exopipe_ops p3s5_state), served runs (hub registry.json under serve.HUB_DIR; paths.TRANSIENTS_ROOT,
         SNAPSHOT_REVIEW_ROOT), run records (<root>/_records of status.REPORT_ROOT, status.RECORD_ROOTS, --records and
         every job's --out in runner.JOBS_ROOT), snapshot fleet state (paths.SNAPSHOT_FLEET_OUT, ops claims), Phase 2 QA
         CSVs, calibration tables. Production tools run IN PLACE (exopipe_ops / clv15 / package; `--help` of each stage's
         module lib/status/steps_*.py lists which and how).
Outputs (product classes; under --out, layout <out>/{status,QA,review_hub,clv15,snapshot,items}/):
         status_report       status/report_<tier>_<UTC>.{txt,json}  (summary)
         readjudication_list QA/readjudication_list_<YYYYMMDD>.csv   (summary; proposals for Gregg)
         coverage_table      QA/coverage_<tier>[_<hash>].csv        (coverage, gaps)
         rebuild_due         QA/rebuild_due_deep[_<hash>].csv       (rebuild-due; measure: config status_rebuild;
                                                                     a report - nothing consumes it, D-ADMIT)
         exclusion_conformance QA/exclusion_conformance_<tier>.csv + QA/coadd_exclusion_conformance.csv (deep archive)
         stale_table         status/stale_<tier>_<class>.csv        (stale)
         served_diff         status/served_diff_<A>_<B>_rows.csv (+ _summary.json / _summary.txt beside it)
         cell_load_review    status/cell_load_review_<label>.csv    (cell-load)
         item_list           items/<unit>.txt for every listing stage; --emit FILE copies the selected stage's list
         qa_table, solar_audit (candidate only: never the installed table), beam_audit, noise_plot,
         date_outlier_proposals, bandpass_phase_qa, provenance_audit (qa); membership_conformance, status_page
         (conformance); qa_night_pages, status_page (pages); processing_ledger (ledger); ratified_register, status_page
         (register); snapshot_eligibility (eligibility); snapshot_batch_list (batch-list)
         Defaults: report / eligibility / batch-list -> status.REPORT_ROOT (not a production root); qa / conformance /
         pages / ledger / register -> status.QA_DIR (a production root: refused in the refactor window - pass --out).
Failure modes (exit 2 = refused, the gate named; 3 = no input; 1 = error):
  S-G1  data_decisions.csv missing or unreadable          S-G2  a planned output under paths.PRODUCTION_ROOTS or in
  status.PROTECTED_OUTPUTS while paths.PRODUCTION_WRITES is False   S-G3  a planned output in status.NEVER_WRITE
  (data_decisions.csv, the installed solar audit), at any time; the solar stage also checks the installed table unchanged
  S-G5  a stage whose input is node-local to another host (site NODE_LOCAL_ROOTS; /data07 and /fast on lwacalim07)
  S-G6  conformance: the branch membership module differs from production (other than import lines)
  S-G7  served-diff: a run without served catalogues (exit 3) or two runs of different tiers (exit 2)
  S-G8  batch-list without --base (window)    S-G9  pages: the night-page content test (P1-P4, P5) fails
  S-G10 ledger with --cells/--lsts/--nights   S-G11 an empty root key   S-G12 qa provenance without --run
  S-G14 --strict: violations, breaches, gaps or stale items found (exit 2; default exit 0 as conformance_check.py)
  Every in-process call of a production tool runs under the write guard (core/guard): a write outside --out/--work/
  /tmp is REFUSED. A content test failing refuses the stage's unit. Reports never edit data_decisions.csv.
"""
import os
import shutil
import sys

SUMMARY = ('completion, coverage, gaps, exclusion conformance, stale products, rebuild due, served diffs, QA tables, '
           'hub status pages; emits item lists')
TIERS = ('deep', 'sublst', 'snapshot')
SELECTORS = ('tier', 'cells', 'lsts', 'nights', 'run_label', 'epochs')
CONFIG = ('status', 'status_report', 'status_rebuild', 'status_ledger', 'status_coverage', 'status_conformance',
          'status_qa', 'status_pages', 'status_snapshot', 'serve', 'runner', 'site')
WRITES = True

ACTIONS = ('report', 'qa', 'conformance', 'pages', 'ledger', 'register', 'eligibility', 'batch-list')
REPORT_STAGES = ('summary', 'coverage', 'gaps', 'exclusion-conformance', 'violations', 'stale', 'served-diff',
                 'cell-load', 'rebuild-due', 'patched')
STAGES = {'report': REPORT_STAGES,
          'qa': ('bandpass', 'qa-table', 'solar', 'provenance', 'beam', 'noise', 'outliers'),
          'conformance': ('membership',), 'pages': ('nights', 'data-processing'), 'ledger': ('ledger',),
          'register': ('register',), 'eligibility': ('eligibility',), 'batch-list': ('batch-list',)}
LISTING = ('coverage', 'gaps', 'exclusion-conformance', 'violations', 'stale', 'rebuild-due', 'patched', 'summary')
# batch-list flags -> config keys (the values decide the list: config status_snapshot, snapshot_batch_lists.py:70-76)
_BATCH_FLAGS = (('--max-new', 'MAX_NEW'), ('--max-gate-gb', 'MAX_GATE_GB'), ('--lc-cache', 'LC_CACHE'),
                ('--base-detections', 'BASE_DETECTIONS'), ('--base-loaded', 'BASE_LOADED'),
                ('--new-detections', 'NEW_DETECTIONS'), ('--new-loaded', 'NEW_LOADED'))


def add_arguments(p):
    p.add_argument('action', nargs='?', default='report', choices=ACTIONS, help='what to do (default report)')
    p.add_argument('extra', nargs='*', metavar='AUDIT_CSV', help='register: the rulings-audit slice CSVs')
    p.add_argument('--stages', help='stages of the action, comma-separated (see the list above)')
    g = p.add_argument_group('report')
    g.add_argument('--coverage', action='store_true', help='coverage table per tier (deep: per LST, scope, night)')
    g.add_argument('--gaps', action='store_true', help='the coverage rows neither excluded nor processed (implies '
                                                         '--coverage); listed')
    g.add_argument('--exclusion-conformance', action='store_true', help='products and served rows on excluded '
                                                                          'cell-bands')
    g.add_argument('--violations', action='store_true', help='deep cells failing the membership conformance (latest '
                                                               'table); listed for `masters`')
    g.add_argument('--stale', nargs='?', const='record', metavar='CLASS[,CLASS]',
                   help='stale items of the classes: record (default), search_attempt, master, master_coverage, '
                        'frame_time, prefit, sublst_generation, qa_night_pages, ledger')
    g.add_argument('--records', action='append', default=[], metavar='ROOT',
                   help='--stale record: also read ROOT/_records (repeatable)')
    g.add_argument('--stale-step', help='--stale record: only records of these steps (comma-separated)')
    g.add_argument('--served-diff', nargs='+', metavar='RUN', help='[A] B: what changes for the served list from A (default: '
                                                                    'the run the hub serves for the tier) to B')
    g.add_argument('--cell-load', action='store_true', help='per (Stokes, LST, night, band) detections and served sources '
                                                              'of the run (--run; contract step 50)')
    g.add_argument('--rebuild-due', action='store_true',
                   help='REPORT of the deep cells whose admitted observing time grew by >= status_rebuild.DUE_FRACTION '
                        'since their master build (C-10). Never a trigger (D-ADMIT, Gregg 2026-10-13: the user invokes '
                        'the rebuild, e.g. `exopipe3 masters --cells @due.txt`)')
    g.add_argument('--patched', action='store_true',
                   help='deep: the published search attempts that are patched Run_P3S4 results (B-01 b, Gregg 2026-10-13: '
                        'kept until the next full re-search) - their cells are the ones a full re-search must redo')
    g.add_argument('--emit', metavar='FILE', help='write the item list of the selected listing stage to FILE')
    g.add_argument('--emit-kind', choices=('cell', 'epoch', 'lst', 'night_cell', 'unit'),
                   help='item kind of the lists (default status.EMIT_KIND[tier]; stale lists keep their own kind)')
    g.add_argument('--strict', action='store_true', default=None,
                   help='exit 2 when a violation, breach, gap or stale item is found (default status.STRICT)')
    g.add_argument('--no-legacy-text', action='store_true', help="summary: leave out phase_status.py's text")
    g.add_argument('--fingerprint', choices=('read', 'write', 'record'),
                   help='summary: data_decisions fingerprint mode (default status_report.FINGERPRINT_MODE)')
    g.add_argument('--archive-root', help='exclusion-conformance: the co-add archive generation to audit (default '
                                          'status_conformance.MASTERS_ARCHIVE_ROOT); "none" skips the archive part')
    g = p.add_argument_group('conformance / pages / qa')
    g.add_argument('--round', help='conformance: the rebuild list (default status_conformance.ROUND_LIST)')
    g.add_argument('--check', action='store_true', help='pages: only the night-page content test on the hub folder '
                                                          '(--out\'s review_hub, else serve.HUB_DIR)')
    g.add_argument('--min-frac', type=float, help='qa outliers: status_qa.OUTLIER_MIN_FRAC for this run (recorded)')
    g.add_argument('--min-cells', type=int, help='qa outliers: status_qa.OUTLIER_MIN_CELLS for this run (recorded)')
    g.add_argument('--bp-shards', type=int, help='qa bandpass: status_qa.BANDPASS_SHARDS for this run (recorded)')
    g = p.add_argument_group('snapshot (eligibility, batch-list)')
    g.add_argument('--cells-file', nargs='+', help='eligibility: cell lists (default: the v9 worklist + shipped cells)')
    g.add_argument('--exclude-file', help='eligibility: marks column in_exclude')
    g.add_argument('--no-store', action='store_true', help='eligibility: skip the fit-store test')
    g.add_argument('--list-out', help='batch-list: the list written (required)')
    g.add_argument('--base', help='batch-list: base cells (required in the refactor window, S-G8)')
    g.add_argument('--order', help='batch-list: candidate order (default status_snapshot.ORDER_LIST)')
    for flag, key in _BATCH_FLAGS:
        g.add_argument(flag, type=int, help=f'batch-list: status_snapshot.{key} for this run (recorded)')
    g.add_argument('--all', action='store_true', help='batch-list: every eligible cell')


def default_out(args):
    from exopipe.phase3.config import CFG
    if getattr(args, 'action', 'report') in ('report', 'eligibility', 'batch-list'):
        return CFG.status.REPORT_ROOT
    return CFG.status.QA_DIR


# =====================================================================================================================
# helpers
# =====================================================================================================================

class _Ctx:
    def __init__(self, args, run):
        from exopipe.phase3.config import CFG
        from exopipe.phase3.lib.status import steps_common as K
        self.args, self.run, self.C, self.K = args, run, CFG, K
        self.out = run.out_root
        self.tier = args.tier or 'deep'
        self.utc = K.utc_stamp()
        self.allowed = [self.out] + ([args.work] if args.work else []) + ['/tmp']
        self.strict = CFG.status.STRICT if args.strict is None else bool(args.strict)
        self.emitted = {}           # stage -> item list path
        self.dry = run.dry_run

    def log(self, unit):
        p = self.K.check_write(self.K.place(self.out, 'status', f'logs/{unit}.log'))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def emit_kind(self, default=None):
        return self.args.emit_kind or default or self.C.status.EMIT_KIND[self.tier]

    def items_selected(self):
        """The selected items, expanded (@file read, patterns resolved): the unit-name hash of a narrowed run."""
        from exopipe.phase3.core import cells
        a = self.args
        out = []
        if a.cells:
            try:
                out += cells.expand(a.cells, universe=list(cells.all_cells()), kind='cell')
            except ValueError:
                out += cells.expand(a.cells)
        for spec, uni in ((a.lsts, list(cells.LSTS)), (a.nights, None), (a.epochs, None)):
            if spec:
                out += cells.expand(spec, universe=uni)
        return sorted(set(out))

    def deep_spec(self):
        """The deep-cell selection of this run for deep_universe: --cells, else the cells of the --lsts, else the cells of the
        LSTs of the --epochs (rows are then filtered to the epochs); None = every cell."""
        from exopipe.phase3.core import cells
        a = self.args
        if a.cells:
            return a.cells
        lsts = []
        if a.lsts:
            lsts = cells.expand(a.lsts, universe=list(cells.LSTS))
        elif a.epochs:
            lsts = sorted({cells.parse_epoch(e)[0] for e in cells.expand_epochs(a.epochs)})
        return ','.join(f'{l}_*' for l in lsts) if lsts else None

    def unit(self, action, stage, items=None):
        return self.K.unit_name(action, stage, self.tier if action in ('report', 'eligibility', 'batch-list',
                                                                       'conformance', 'pages') else 'none', items)

    def emit(self, u, stage, kind, items, table, qualifies, extra_head=None):
        """Write the stage's item list (product class item_list) and record it."""
        K = self.K
        path = K.place(self.out, 'items', f'{u.name}.txt')
        its = K.write_items(path, kind, items, table, qualifies, stage)
        if extra_head:
            with open(path) as f:
                body = f.read()
            K.write_text(path, ''.join(f'# {k}: {v}\n' for k, v in extra_head.items()) + body)
        u.add_output(path, 'item_list')
        self.emitted[stage] = path
        u.meta['items'] = {'path': path, 'kind': kind, 'n': len(its)}
        return its

    def strict_check(self, u, n, what):
        u.meta['findings'] = n
        if self.strict and n:
            u.check(f'S-G14.strict.{what}', False, value=n, detail=f'{n} {what} found (--strict)')


def _stage_list(args):
    if args.action == 'report':
        flagged = [s for s, on in (('coverage', args.coverage or args.gaps), ('gaps', args.gaps),
                                   ('exclusion-conformance', args.exclusion_conformance),
                                   ('violations', args.violations), ('stale', args.stale is not None),
                                   ('served-diff', bool(args.served_diff)), ('cell-load', args.cell_load),
                                   ('rebuild-due', args.rebuild_due), ('patched', args.patched)) if on]
        if args.stages:
            st = [s.strip() for s in args.stages.split(',') if s.strip()]
        else:
            st = flagged or ['summary']
        if 'gaps' in st and 'coverage' in st:
            st.remove('coverage')                     # gaps writes the coverage table and its gap list
        return st
    if args.stages:
        return [s.strip() for s in args.stages.split(',') if s.strip()]
    from exopipe.phase3.config import CFG
    if args.action == 'qa':
        return list(CFG.status_qa.STAGES_DEFAULT)
    return list(STAGES[args.action])


def _check_stages(args, stages):
    from exopipe.phase3.core.record import Refusal
    bad = [s for s in stages if s not in STAGES[args.action]]
    if bad:
        raise SystemExit(f'error: unknown stage(s) {bad} of `status {args.action}`; stages: {", ".join(STAGES[args.action])}')
    tier = args.tier or 'deep'
    only_deep = {'violations', 'rebuild-due', 'patched', 'membership', 'nights', 'data-processing'}
    for s in stages:
        if s in only_deep and tier != 'deep':
            if s == 'rebuild-due' and tier == 'snapshot':
                raise Refusal('rebuild-due --tier snapshot: the snapshot masters are the deep slot masters - run '
                              '`status --rebuild-due --tier deep`')
            if s == 'rebuild-due':
                raise Refusal('rebuild-due --tier sublst is not defined in the refactor window (no sub-LST build record '
                              'carries the admitted nights yet; implementation note, status.md)')
            raise Refusal(f'stage {s} is deep-only (--tier deep)')
    if args.action in ('eligibility', 'batch-list') and tier != 'snapshot':
        raise Refusal(f'`status {args.action}` is a snapshot-tier action (--tier snapshot)')
    if args.emit:
        listing = [s for s in stages if s in LISTING]
        if len(listing) != 1:
            raise SystemExit(f'error: --emit takes the list of ONE listing stage; selected: {listing or "none"}')


def _apply_flag_overrides(args):
    from exopipe.phase3.config import CFG
    for flag, key in (('min_frac', 'status_qa.OUTLIER_MIN_FRAC'), ('min_cells', 'status_qa.OUTLIER_MIN_CELLS'),
                      ('bp_shards', 'status_qa.BANDPASS_SHARDS')):
        v = getattr(args, flag, None)
        if v is not None:
            CFG.set(key, v, source=f'--{flag.replace("_", "-")}')
    for flag, key in _BATCH_FLAGS:
        v = getattr(args, flag[2:].replace('-', '_'), None)
        if v is not None:
            CFG.set(f'status_snapshot.{key}', v, source=flag)
    if args.fingerprint:
        CFG.set('status_report.FINGERPRINT_MODE', args.fingerprint, source='--fingerprint')


# =====================================================================================================================
# run
# =====================================================================================================================

def run(args, run):
    from exopipe.phase3.core.record import Refusal
    _apply_flag_overrides(args)
    stages = _stage_list(args)
    try:
        _check_stages(args, stages)
    except Refusal as e:
        if run.dry_run:
            print(f'PLAN: REFUSED - {e}')
            return 2
        raise
    ctx = _Ctx(args, run)
    if args.resume:
        print('status: --resume has no effect - a report is the state now, every stage is recomputed', file=sys.stderr)
    if run.dry_run:
        return _dry_run(ctx, stages)
    for s in stages:
        fn = globals()['_stage_' + s.replace('-', '_')]
        fn(ctx)
    if args.emit:
        st = next(s for s in stages if s in LISTING)
        src = ctx.emitted.get(st)
        if src:
            ctx.K.check_write(args.emit)
            os.makedirs(os.path.dirname(os.path.abspath(args.emit)), exist_ok=True)
            shutil.copyfile(src, args.emit)
            print(f'item list ({st}): {args.emit}')
    return None


# =====================================================================================================================
# dry run: the plan (inputs found, outputs, gates); writes nothing
# =====================================================================================================================

def _dry_run(ctx, stages):
    from exopipe.phase3.core.record import Refusal
    K, C, a = ctx.K, ctx.C, ctx.args
    rc = 0
    tierless = a.action in ('qa', 'ledger', 'register')
    print(f'PLAN status {a.action} tier {"none" if tierless else ctx.tier} -> out {ctx.out}; stages: {", ".join(stages)}')
    try:
        ex = K.exclusions()
        fp = ex.fingerprint()
        print(f'  gate S-G1 data_decisions: OK ({fp["path"]}, {fp["n_exclude_rows"]} exclude keys, sha1 {fp["sha1"][:12]})')
    except Refusal as e:
        print(f'  gate S-G1: REFUSED - {e}')
        return 2
    for s in stages:
        try:
            plan = _plan(ctx, s)
        except Refusal as e:
            print(f'  [{s}] REFUSED - {e}')
            rc = 2
            continue
        print(f'  [{s}] unit {plan["unit"]}')
        for p in plan.get('inputs', ()):
            print(f'      input  {"found  " if os.path.exists(p) else "MISSING"} {p}')
        for p, cls in plan.get('outputs', ()):
            try:
                K.check_write(p)
                g = ''
            except Refusal as e:
                g = f'   <- REFUSED: {e}'
                rc = 2
            print(f'      output {cls:24s} {p}{g}')
        for n in plan.get('notes', ()):
            print(f'      note   {n}')
        try:
            K.need_inputs([p for p in plan.get('need', ())], s)
            if plan.get('need'):
                print(f'      gate   S-G5/S-G11 inputs present on this host: OK')
        except Refusal as e:
            print(f'      gate   REFUSED - {e}')
            rc = 2
    if a.emit:
        print(f'  --emit {a.emit}: the item list of {[s for s in stages if s in LISTING][0]}')
    print('PLAN: ' + ('every stage can run' if rc == 0 else 'a gate refuses (exit 2)'))
    return rc


def _plan(ctx, s):
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.status import steps_coverage as COV, steps_qa as QA
    K, C, a = ctx.K, ctx.C, ctx.args
    out, tier = ctx.out, ctx.tier
    sel = ctx.items_selected()
    if s == 'summary':
        R = C.status_report
        return dict(unit=ctx.unit('report', s), inputs=[C.status.LEDGER, C.paths.DATA_DECISIONS, R.REGISTER,
                                                        K.expand(C.status.REGISTRY_FILE), R.DECISIONS_FINGERPRINT],
                    outputs=[(K.place(out, 'status', f'report_{tier}_{ctx.utc}.json'), 'status_report'),
                             (K.place(out, 'status', f'report_{tier}_{ctx.utc}.txt'), 'status_report'),
                             (K.place(out, 'QA', f'readjudication_list_{ctx.utc[:8]}.csv'), 'readjudication_list')],
                    notes=[f'fingerprint mode {R.FINGERPRINT_MODE}: ' + ('production file copied to <out>/status, never '
                                                                          'written' if R.FINGERPRINT_MODE != 'write'
                                                                          else 'production file rewritten'),
                           f'co-add roots: ' + ', '.join(f'{r} ({"present" if os.path.isdir(r) else "absent"})'
                                                         for r in R.COADD_ROOTS)])
    if s in ('coverage', 'gaps'):
        u = ctx.unit('report', 'gaps' if s == 'gaps' else 'coverage', sel)
        if tier == 'deep':
            cells = COV.deep_universe(ctx.deep_spec())
            need = COV.deep_inputs()
            notes = [f'{len(cells)} cells, {len({c.split("_")[0] for c in cells})} LSTs; archive columns '
                     f'{C.status_coverage.ARCHIVE_COLUMNS}, searched columns {C.status_coverage.SEARCHED_COLUMNS} '
                     f'(published attempts of Run_{C.status.SEARCH_LABEL["deep"]})']
        elif tier == 'sublst':
            need, notes = COV.sublst_inputs(), ['per (LST, scope); served LSTs from the hub registry']
        else:
            need, notes = COV.snapshot_inputs(), ['per night cell of the v9 worklist and the shipped cells']
        table = K.place(out, 'QA', f'coverage_{tier}' + (u[len(f'report_{s.replace("-", "")}_{tier}'):] or '') + '.csv')
        return dict(unit=u, inputs=need, need=need, notes=notes,
                    outputs=[(table, 'coverage_table'), (K.place(out, 'items', f'{u}.txt'), 'item_list')])
    if s == 'rebuild-due':
        from exopipe.phase3.lib.status import steps_rebuild as RB
        RB.check_source()
        u = ctx.unit('report', s, sel)
        cells = COV.deep_universe(ctx.deep_spec())
        need = [C.paths.COADD_SB_ROOT, C.paths.PHASE2_ROOT]
        return dict(unit=u, inputs=need, need=need,
                    outputs=[(K.place(out, 'QA', 'rebuild_due_deep' + u[len('report_rebuilddue_deep'):] + '.csv'),
                              'rebuild_due'), (K.place(out, 'items', f'{u}.txt'), 'item_list')],
                    notes=[f'{len(cells)} cells; DUE when T_new >= {C.status_rebuild.DUE_FRACTION} x T_build; '
                           f't = {C.status_rebuild.TIME_CARD} of the Phase 2 deep image (config status_rebuild); new '
                           f'nights from {C.status_rebuild.NEW_SET_SOURCE} (D-ADMIT admission records: '
                           f'{C.status_rebuild.ADMISSION_RECORD or "none yet"}); a report, never a rebuild trigger'])
    if s == 'patched':
        u = ctx.unit('report', s, sel)
        return dict(unit=u, inputs=[], need=[],
                    outputs=[(K.place(out, 'QA', 'patched_attempts_deep' + u[len('report_patched_deep'):] + '.csv'),
                              'patched_attempts'), (K.place(out, 'items', f'{u}.txt'), 'item_list')],
                    notes=[f'the published attempts of Run_{C.status.SEARCH_LABEL["deep"]} (p3s3_sources.published_attempts) '
                           f'whose folder holds a PATCH_RECORD.json (ops/p3s5_patch_cell.py): patched Run_P3S4 results, '
                           f'kept until the next full re-search (B-01 b)'])
    if s == 'exclusion-conformance':
        u = ctx.unit('report', s, sel)
        runs = sorted(K.served_runs(tier))
        run_dirs = [os.path.join(K.runs_root(tier), r) for r in runs]
        outs = [(K.place(out, 'QA', K.narrowed(u, C.status_conformance.EXCLCONF_TABLE.format(tier=tier))),
                 'exclusion_conformance'),
                (K.place(out, 'items', f'{u}.txt'), 'item_list')]
        need = list(run_dirs)
        arch = a.archive_root or C.status_conformance.MASTERS_ARCHIVE_ROOT
        if tier == 'deep':
            need.append(C.paths.COADD_SB_ROOT)
            if arch != 'none':
                outs.append((K.place(out, 'QA', C.status_conformance.MASTERS_EXCL_TABLE), 'exclusion_conformance'))
        return dict(unit=u, inputs=need + ([arch] if tier == 'deep' and arch != 'none' else []), need=need,
                    outputs=outs, notes=[f'served runs of the tier (hub registry): {", ".join(runs) or "none"}'])
    if s == 'violations':
        u = ctx.unit('report', s, sel)
        from exopipe.phase3.lib.status import steps_report as RP
        t = RP.membership_table(out)
        return dict(unit=u, inputs=[t or K.expand(C.status_conformance.LATEST)],
                    outputs=[(K.place(out, 'status', K.narrowed(u, 'violations_deep.csv')), 'stale_table'),
                             (K.place(out, 'items', f'{u}.txt'), 'item_list')])
    if s == 'stale':
        classes = _stale_classes(ctx)
        from exopipe.phase3.lib.status import steps_stale as ST
        notes = []
        if 'record' in classes:
            notes.append('record roots: ' + ', '.join(ST.record_roots(a.records)) or 'none with _records/')
        u = ctx.unit('report', 'stale_' + '_'.join(classes), sel)
        return dict(unit=u, notes=notes, outputs=[(K.place(out, 'status', K.narrowed(u, f'stale_{tier}_{"_".join(classes)}.csv')),
                                                   'stale_table'), (K.place(out, 'items', f'{u}.txt'), 'item_list')])
    if s == 'served-diff':
        A, B = _served_pair(ctx)
        u = ctx.unit('report', f'serveddiff_{os.path.basename(A)}_{os.path.basename(B)}')
        base = K.place(out, 'status', f'served_diff_{os.path.basename(A)}_{os.path.basename(B)}')
        return dict(unit=u, inputs=[A, B], need=[A, B],
                    outputs=[(base + '_rows.csv', 'served_diff'), (base + '_summary.json', '(summary)'),
                             (base + '_summary.txt', '(summary)')])
    if s == 'cell-load':
        label = a.run_label or C.status.SEARCH_LABEL.get(tier)
        rd = K.resolve_run(label, tier)
        return dict(unit=ctx.unit('report', f'cellload_{label}'), inputs=[os.path.join(rd, 'selected_all.csv')],
                    need=[rd], outputs=[(K.place(out, 'status', C.status.CELL_LOAD_FILE.format(label=label)),
                                         'cell_load_review')])
    if a.action == 'qa':
        if s == 'provenance' and not a.run_label:
            raise Refusal('S-G12: qa --stages provenance needs --run (the run whose served detections are audited)')
        rd = K.resolve_run(a.run_label, 'deep') if s == 'provenance' else None
        need = [p for p in QA.stage_inputs(s, rd)]
        notes = []
        if s == 'bandpass':
            notes.append(f'{len(QA.bandpass_tables())} tables, {C.status_qa.BANDPASS_SHARDS} shards, site PYTHON["casa"]')
        return dict(unit=ctx.unit('qa', s), inputs=need, need=need, outputs=QA.stage_outputs(s, out, ctx.utc), notes=notes)
    if s == 'membership':
        rl = a.round or C.status_conformance.ROUND_LIST
        return dict(unit=ctx.unit('conformance', s), inputs=[C.paths.COADD_SB_ROOT, rl], need=[C.paths.COADD_SB_ROOT],
                    outputs=[(K.place(out, 'QA', os.path.basename(C.status_conformance.LATEST)), 'membership_conformance'),
                             (K.place(out, 'QA', os.path.basename(C.status_conformance.HISTORY)), 'csv_table'),
                             (K.place(out, 'review_hub', os.path.basename(C.status_conformance.PAGE)), 'status_page')])
    if s == 'nights':
        return dict(unit=ctx.unit('pages', s, sel), inputs=[C.status.LEDGER, C.paths.COADD_SB_ROOT],
                    need=[C.paths.COADD_SB_ROOT],
                    outputs=[(K.place(out, 'review_hub', 'nights'), 'qa_night_pages')],
                    notes=['--check: content test only' if a.check else 'build_qa_pages.main in place, then check()'])
    if s == 'data-processing':
        return dict(unit=ctx.unit('pages', s), inputs=[C.status.LEDGER, C.paths.COADD_SB_ROOT],
                    need=[C.paths.COADD_SB_ROOT],
                    outputs=[(K.place(out, 'review_hub', C.status_pages.DATA_PAGE), 'status_page')])
    if s == 'ledger':
        if sel:
            raise Refusal('S-G10: the ledger is whole-survey only (no --cells/--lsts/--nights/--epochs)')
        return dict(unit=ctx.unit('ledger', s), inputs=[C.paths.PHASE1_ROOT, C.paths.PHASE2_ROOT],
                    need=[C.paths.PHASE1_ROOT, C.paths.PHASE2_ROOT],
                    outputs=[(K.place(out, 'clv15', os.path.basename(C.status.LEDGER)), 'processing_ledger')])
    if s == 'register':
        if not a.extra:
            raise Refusal('register needs the audit-slice CSVs (positional AUDIT_CSV ...)')
        return dict(unit=ctx.unit('register', s), inputs=list(a.extra), need=list(a.extra),
                    outputs=[(K.place(out, 'clv15', os.path.basename(C.status_report.REGISTER)), 'ratified_register'),
                             (K.place(out, 'review_hub', C.status_pages.REGISTER_PAGE), 'status_page')])
    if s == 'eligibility':
        return dict(unit=ctx.unit('eligibility', s), inputs=[C.paths.SNAPSHOT_FLEET_OUT] + list(a.cells_file or ()),
                    need=[C.paths.SNAPSHOT_FLEET_OUT],
                    outputs=[(K.place(out, 'snapshot', C.status_snapshot.ELIGIBILITY_TABLE), 'snapshot_eligibility')])
    if s == 'batch-list':
        if not a.base:
            raise Refusal('S-G8: batch-list needs --base FILE in the refactor window (the script would ssh to the review '
                          'host to read the served run\'s cells)')
        if not a.list_out:
            raise Refusal('batch-list needs --list-out FILE')
        return dict(unit=ctx.unit('batch-list', s), inputs=[a.base, a.order or K.expand(C.status_snapshot.ORDER_LIST)],
                    need=[a.base], outputs=[(a.list_out, 'snapshot_batch_list')])
    raise Refusal(f'no plan for stage {s}')


def _stale_classes(ctx):
    a = ctx.args
    classes = [c.strip() for c in (a.stale or 'record').split(',') if c.strip()]
    bad = [c for c in classes if c not in ctx.C.status.STALE_CLASSES]
    if bad:
        raise SystemExit(f'error: unknown stale class(es) {bad}; classes: {", ".join(ctx.C.status.STALE_CLASSES)}')
    return classes


def _served_pair(ctx):
    from exopipe.phase3.core.record import Refusal
    K, a = ctx.K, ctx.args
    sd = list(a.served_diff or ())
    if not sd or len(sd) > 2:
        raise Refusal('--served-diff takes [A] B')
    if len(sd) == 1:
        served = sorted(K.served_runs(ctx.tier))
        if len(served) != 1:
            raise Refusal(f'--served-diff B: the hub registry serves {len(served)} runs for tier {ctx.tier} '
                          f'({served}); give A explicitly')
        sd = [served[0]] + sd
    return K.resolve_run(sd[0], ctx.tier), K.resolve_run(sd[1], ctx.tier)


# =====================================================================================================================
# report stages
# =====================================================================================================================

def _stage_summary(ctx):
    import hashlib
    import time
    from exopipe.phase3.lib.status import steps_report as RP
    K, C, a, out, tier = ctx.K, ctx.C, ctx.args, ctx.out, ctx.tier
    name = ctx.unit('report', 'summary')
    with ctx.run.unit(name, tier=tier) as u:
        ex = K.exclusions()
        u.add_input(C.paths.DATA_DECISIONS)
        led_rows = K.read_csv(C.status.LEDGER) if os.path.exists(C.status.LEDGER) else []
        if os.path.exists(C.status.LEDGER):
            u.add_input(C.status.LEDGER)
        legacy = ''
        if not a.no_legacy_text:
            rc, legacy = RP.legacy_summary_text(out, ctx.allowed, unit=u, log=ctx.log(name))
            u.meta['legacy_rc'] = rc
        prev, prev_src = RP.previous_state(out, tier)
        state = RP.readjudication_state(led_rows)
        rrows = RP.readjudication_rows(led_rows, prev)
        rpath = K.place(out, 'QA', f'readjudication_list_{ctx.utc[:8]}.csv')
        K.write_csv(rpath, C.status_report.READJUDICATION_COLUMNS, rrows)
        fp = ex.fingerprint()
        led_sha = hashlib.sha1(open(C.status.LEDGER, 'rb').read()).hexdigest() if led_rows else None
        rep = {'tier': tier, 'written': RP.now_utc(), 'host': ctx.run.host,
               'node_allocation': dict(C.site.NODE_POOLS or {}),
               'phase_completion': {'ledger': C.status.LEDGER, 'sha1': led_sha, 'rows': len(led_rows),
                                    'counts': RP.ledger_counts(led_rows),
                                    'awaiting_p2': sum(1 for r in led_rows if r.get('p2') == 'none'),
                                    'age_h': (round((time.time() - os.path.getmtime(C.status.LEDGER)) / 3600, 1)
                                              if led_rows else None)},
               'data_decisions': {'path': fp['path'], 'sha1': fp['sha1'], 'n_exclude_keys': fp['n_exclude_rows'],
                                  'excluded_nights': len({(l, d) for (l, s, d) in ex.rows}),
                                  'mtime_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                             time.gmtime(os.path.getmtime(fp['path'])))},
               'masters': RP.masters_section(), 'served_runs': RP.served_section(),
               'readjudication': {'list': rpath, 'rows': len(rrows), 'compared_with': prev_src},
               'conformance': RP.conformance_section(out), 'rulings_not_enforced': RP.rulings_section(),
               'canon_check': RP.canon_check()}
        if C.status_report.FINGERPRINT_MODE == 'record':
            from exopipe.phase3.core import record as R
            old = R.load_record(out, 'status', name)
            rep['data_decisions']['changed_since_last_report'] = (None if not old else
                                                                  (old.get('meta') or {}).get('dd_sha1') != fp['sha1'])
        jpath = K.place(out, 'status', f'report_{tier}_{ctx.utc}.json')
        RP.report_json(jpath, rep)
        txt = [legacy.rstrip()] if legacy else []
        txt += [RP.TEXT_HEADER,
                f"[nodes] {rep['node_allocation']}",
                f"[ledger] {rep['phase_completion']['rows']} (LST, night) rows; counts {rep['phase_completion']['counts']}",
                f"[data_decisions] {fp['n_exclude_rows']} exclude keys, {rep['data_decisions']['excluded_nights']} "
                f"(LST, night)s with an exclude row; sha1 {fp['sha1'][:12]}",
                '[masters] ' + RP.masters_text(rep['masters']),
                f"[served] " + '; '.join(f"{p['stokes']} {p['page']} {p['run']} :{p['port']}"
                                        for p in rep['served_runs'].get('pages', [])),
                f"[re-adjudication] {len(rrows)} excluded (LST, night)s advanced since {prev_src} -> {rpath}",
                f"[conformance] {rep['conformance']}",
                f"[rulings not enforced] {rep['rulings_not_enforced'].get('problem_counts')}",
                '[canon check] ' + (rep['canon_check'].strip() or 'clean')]
        tpath = K.place(out, 'status', f'report_{tier}_{ctx.utc}.txt')
        K.write_text(tpath, '\n'.join(txt) + '\n')
        print('\n'.join(txt))
        u.meta.update(readjudication_state=state, dd_sha1=fp['sha1'], exclusions=fp)
        for p in (jpath, tpath):
            u.add_output(p, 'status_report')
        u.add_output(rpath, 'readjudication_list', ledger=C.status.LEDGER, ledger_sha1=led_sha, previous=prev,
                     previous_source=prev_src)
        ctx.emit(u, 'summary', 'epoch', [f"{r['lst']}_{r['date']}" for r in rrows], rpath, 'go_no_go=' +
                 C.status_report.READJUDICATION_GO.replace('|', '/'))


def _stage_coverage(ctx, gaps=False):
    from exopipe.phase3.lib.status import steps_coverage as COV
    K, C, a, out, tier = ctx.K, ctx.C, ctx.args, ctx.out, ctx.tier
    sel = ctx.items_selected()
    stage = 'gaps' if gaps else 'coverage'
    name = ctx.unit('report', stage, sel)
    with ctx.run.unit(name, tier=tier, stage=stage) as u:
        ex = K.exclusions()
        u.add_input(C.paths.DATA_DECISIONS)
        u.meta['exclusions'] = ex.fingerprint()
        if tier == 'deep':
            K.need_inputs(COV.deep_inputs(), stage)
            rows, cells = COV.deep_coverage(ctx.deep_spec(), ex, workers=a.workers)
            from exopipe.phase3.core import cells as CL
            if a.epochs:
                want = {'_'.join(CL.parse_epoch(e)[:2]) for e in CL.expand_epochs(a.epochs)}
                rows = [r for r in rows if f"{r['lst']}_{r['date']}" in want]
            if a.nights:
                want = set(CL.expand(a.nights))
                rows = [r for r in rows if r['date'] in want]
            cols = COV.DEEP_COLUMNS
            u.meta['cells'] = len(cells)
        elif tier == 'sublst':
            K.need_inputs(COV.sublst_inputs(), stage)
            rows, _ = COV.sublst_coverage(a.lsts or a.cells, workers=a.workers)
            cols = COV.SUBLST_COLUMNS
        else:
            K.need_inputs(COV.snapshot_inputs(), stage)
            rows, _ = COV.snapshot_coverage(a.cells, workers=a.workers)
            cols = COV.SNAPSHOT_COLUMNS
        suffix = name[len(f'report_{stage}_{tier}'):]
        table = K.place(out, 'QA', f'coverage_{tier}{suffix}.csv')
        K.write_csv(table, cols, rows)
        u.add_output(table, 'coverage_table')
        gap = [r for r in rows if str(r['gap']) == '1']
        kind = ctx.emit_kind()
        items = [_item(kind, r) for r in gap]
        ctx.emit(u, stage, kind, items, table, 'gap=1')
        u.meta['gaps'] = len(gap)
        print(f'coverage {tier}: {len(rows)} rows, {len(gap)} gaps -> {table}')
        if tier == 'deep':
            import collections
            print('  census: ' + ', '.join(f'{k} {n}' for k, n in
                                          sorted(collections.Counter(r['census_class'] for r in rows).items())))
        ctx.strict_check(u, len(gap), 'gaps')


def _item(kind, r):
    from exopipe.phase3.lib.status import steps_common as K
    return K.row_item(kind, r)


def _stage_gaps(ctx):
    _stage_coverage(ctx, gaps=True)


def _stage_rebuild_due(ctx):
    import collections
    from exopipe.phase3.lib.status import steps_coverage as COV, steps_rebuild as RB
    K, C, a, out = ctx.K, ctx.C, ctx.args, ctx.out
    sel = ctx.items_selected()
    name = ctx.unit('report', 'rebuild-due', sel)
    with ctx.run.unit(name, tier='deep', stage='rebuild-due') as u:
        ex = K.exclusions()
        u.add_input(C.paths.DATA_DECISIONS)
        K.need_inputs([C.paths.COADD_SB_ROOT, C.paths.PHASE2_ROOT], 'rebuild-due')
        cells = COV.deep_universe(ctx.deep_spec())
        rows = RB.rebuild_due(cells, ex, workers=a.workers)
        table = K.place(out, 'QA', 'rebuild_due_deep' + name[len('report_rebuilddue_deep'):] + '.csv')
        K.write_csv(table, RB.COLUMNS, rows)
        u.meta.update(measure=C.status_rebuild.__doc__, due_fraction=C.status_rebuild.DUE_FRACTION,
                      new_set_source=C.status_rebuild.NEW_SET_SOURCE, admission_record=C.status_rebuild.ADMISSION_RECORD,
                      trigger='none: a report (D-ADMIT, ledger 2335 - the user invokes the rebuild)',
                      exclusions=ex.fingerprint(),
                      counts=dict(collections.Counter(r['status'] for r in rows)),
                      total_build_s=round(sum(float(r['build_s'] or 0) for r in rows), 1),
                      total_new_s=round(sum(float(r['new_s'] or 0) for r in rows), 1))
        u.add_output(table, 'rebuild_due')
        due = [r['cell'] for r in rows if RB.is_due(r)]
        ctx.emit(u, 'rebuild-due', 'cell', due, table, 'status=due|never-built')
        print(f'rebuild-due (report only; new nights {RB.check_source()}): {len(rows)} cells; '
              + ', '.join(f'{k} {n}' for k, n in sorted(u.meta['counts'].items())) + f' -> {table}')
        for r in rows:
            if r['status'] in ('due', 'never-built', 'unmeasured'):
                print(f"  {r['cell']:10s} {r['status']:11s} new {float(r['new_s'] or 0) / 3600:7.2f} h of build "
                      f"{float(r['build_s'] or 0) / 3600:7.2f} h ({r['n_new']} new nights){'  ' + r['unmeasured'][:80] if r['unmeasured'] else ''}")


def patched_rows(S, want):
    """The rows of the patched-attempt table: every published attempt of the register S (ops/p3s3_sources) whose cell is in
    ``want`` (None = every cell), with its patch state from the attempt's own PATCH_RECORD.json."""
    from exopipe.phase3.lib.catalog import deep_ported as DP
    rows = []
    for a in sorted(S.published_attempts(['V', 'I']), key=lambda a: (a['cell'], a['stokes'])):
        if want is not None and a['cell'] not in want:
            continue
        src = DP.patched_from(a['dir'])
        lst, scope = a['cell'].split('_', 1)
        rows.append(dict(cell=a['cell'], lst=lst, scope=scope, stokes=a['stokes'], attempt=a['att'], dir=a['dir'],
                         patched='1' if src is not None else '0', patched_from=src or ''))
    return rows


def _stage_patched(ctx):
    from exopipe.phase3.lib.status import steps_coverage as COV, steps_inplace as IPL
    K, C, out, a = ctx.K, ctx.C, ctx.out, ctx.args
    sel = ctx.items_selected()
    name = ctx.unit('report', 'patched', sel)
    with ctx.run.unit(name, tier='deep', stage='patched') as u:
        if a.nights or a.epochs:
            u.refuse('status --patched selects cells (--cells, --lsts): a published attempt holds every night of its cell, '
                     'so --nights / --epochs select no attempt')
        spec = ctx.deep_spec()
        want = set(COV.deep_universe(spec)) if spec else None
        label = C.status.SEARCH_LABEL['deep']
        S = IPL.search_sources(label)                     # register and collect roots pinned from config, as the harvest
        for st in ('V', 'I'):
            reg = os.path.join(S.STATE, f'published_{st}.csv')
            if os.path.exists(reg):
                u.add_input(reg)                          # status --stale and --resume see the register move on
        rows = patched_rows(S, want)
        table = K.place(out, 'QA', 'patched_attempts_deep' + name[len('report_patched_deep'):] + '.csv')
        K.write_csv(table, PATCHED_COLUMNS, rows)
        n = {st: sum(1 for r in rows if r['stokes'] == st and r['patched'] == '1') for st in ('V', 'I')}
        u.meta.update(search_state=S.STATE, label=label, cells_spec=spec, published=len(rows), patched=n,
                      ruling='B-01 (b), R-GREGG-20261013-REFACTOR-SHEET-DADMIT-B01-B02-B03-B05-B30-B32')
        u.add_output(table, 'patched_attempts', label=label, state=S.STATE,
                     cells=sorted(want) if want is not None else None)
        cells = sorted({r['cell'] for r in rows if r['patched'] == '1'})
        ctx.emit(u, 'patched', 'cell', cells, table, 'patched=1')
        print(f'patched: {len(rows)} published attempts of Run_{label} ({S.STATE}); patched Run_P3S4 results: {n["V"]} V, '
              f'{n["I"]} I, in {len(cells)} cells -> {table}')


PATCHED_COLUMNS = ('cell', 'lst', 'scope', 'stokes', 'attempt', 'dir', 'patched', 'patched_from')


def _stage_exclusion_conformance(ctx):
    import collections
    from exopipe.phase3.lib.status import steps_coverage as COV, steps_exclconf as XC
    K, C, a, out, tier = ctx.K, ctx.C, ctx.args, ctx.out, ctx.tier
    sel = ctx.items_selected()
    name = ctx.unit('report', 'exclusion-conformance', sel)
    with ctx.run.unit(name, tier=tier, stage='exclusion-conformance') as u:
        ex = K.exclusions()
        u.add_input(C.paths.DATA_DECISIONS)
        u.meta['exclusions'] = ex.fingerprint()
        rows = []
        runs = sorted(K.served_runs(tier))
        run_dirs = [os.path.join(K.runs_root(tier), r) for r in runs]
        K.need_inputs(run_dirs, 'exclusion-conformance (served runs)')
        if tier == 'deep':
            K.need_inputs([C.paths.COADD_SB_ROOT], 'exclusion-conformance (working tree)')
            cells = COV.deep_universe(ctx.deep_spec())
            rows += XC.working_tree_audit(cells, ex, workers=a.workers)
            arch = a.archive_root or C.status_conformance.MASTERS_ARCHIVE_ROOT
            if arch != 'none':
                atab = K.place(out, 'QA', C.status_conformance.MASTERS_EXCL_TABLE)
                arc, arows, res = XC.archive_audit(atab, arch, unit=u, log=ctx.log(name))
                u.meta['archive_audit'] = {'root': arch, 'rc': arc, 'tail': res.tail[-800:]}
                u.add_output(atab, 'exclusion_conformance')
                for r in arows:
                    rows.append(dict(part='master-archive', run=os.path.basename(arch), file=r.get('product', ''),
                                     row_id=r.get('kind', ''), lst=r['lst'], date=r['date'], band=r['band'],
                                     verdict=r['verdict'], note=r.get('note', '')))
        srows, counts, errors = XC.served_runs_audit(tier, run_dirs, workers=a.workers)
        rows += srows
        u.meta['served'] = {f'{k[0]} {k[1]}': v for k, v in counts.items()}
        if errors:
            u.meta['unreadable_files'] = errors[:50]
        # a served CSV that cannot be read is not audited: the unit refuses, as the serve-time gate does
        # (snapshot_serve_exclusion_gate.py:83-85 "could not be read - nothing served")
        u.check('exclusion_conformance.served_files_readable', not errors, value=len(errors),
                detail=('; '.join(errors[:3]) if errors else 'every served CSV read'))
        table = K.place(out, 'QA', K.narrowed(name, C.status_conformance.EXCLCONF_TABLE.format(tier=tier)))
        K.write_csv(table, XC.COLUMNS, rows)
        u.add_output(table, 'exclusion_conformance')
        kind = ctx.emit_kind()
        items = [_item(kind, r) for r in rows if r['verdict'] in ('BREACH', 'STALE', 'UNKNOWN-EPOCH', 'EXCLUDED-ROW')]
        ctx.emit(u, 'exclusion-conformance', kind, items, table, 'verdict=BREACH|STALE|UNKNOWN-EPOCH|EXCLUDED-ROW')
        by = collections.Counter((r['part'], r['verdict']) for r in rows)
        print(f'exclusion-conformance {tier}: ' + (', '.join(f'{p} {v} {n}' for (p, v), n in sorted(by.items()))
                                                   or 'conformant') + f' -> {table}')
        n_bad = sum(n for (p, v), n in by.items() if v in ('BREACH', 'EXCLUDED-ROW'))
        ctx.strict_check(u, n_bad, 'breaches or served rows on excluded cell-bands')


def _stage_violations(ctx):
    from exopipe.phase3.lib.status import steps_report as RP, steps_stale as ST
    K, a, out = ctx.K, ctx.args, ctx.out
    sel = ctx.items_selected()
    name = ctx.unit('report', 'violations', sel)
    with ctx.run.unit(name, tier='deep', stage='violations') as u:
        from exopipe.phase3.lib.status import steps_coverage as COV
        K.exclusions()                                    # S-G1 in every stage
        u.add_input(ctx.C.paths.DATA_DECISIONS)
        spec = ctx.deep_spec()
        rows, src = RP.violations(out, set(COV.deep_universe(spec)) if spec else None)
        u.add_input(src)
        table = K.place(out, 'status', K.narrowed(name, 'violations_deep.csv'))
        K.write_csv(table, ST.COLUMNS, [dict(**{'class': 'violation'}, item=r['cell'], kind='cell', status='FAIL',
                                             reason=(r.get('first_violation') or 'membership violation')[:300],
                                             step='masters', root=src, detail=f"{r.get('n_violations')} violations")
                                        for r in rows])
        u.add_output(table, 'stale_table')
        ctx.emit(u, 'violations', 'cell', [r['cell'] for r in rows], table, 'status=FAIL')
        print(f'violations: {len(rows)} failing cells (from {src}) -> {table}')
        ctx.strict_check(u, len(rows), 'membership violations')


def _stage_stale(ctx):
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.status import steps_stale as ST, steps_exclconf as XC, steps_coverage as COV
    K, C, a, out, tier = ctx.K, ctx.C, ctx.args, ctx.out, ctx.tier
    classes = _stale_classes(ctx)
    sel = ctx.items_selected()
    name = ctx.unit('report', 'stale_' + '_'.join(classes), sel)
    with ctx.run.unit(name, tier=tier, stage='stale', classes=classes) as u:
        ex = K.exclusions()
        u.add_input(C.paths.DATA_DECISIONS)
        rows, head = [], {}
        for cls in classes:
            if cls == 'record':
                roots = ST.record_roots(a.records)
                u.meta['record_roots'] = roots
                steps = set(a.stale_step.split(',')) if a.stale_step else None
                got = ST.stale_records(roots, ex, steps)
                head['stale-class'] = 'record'
            elif cls == 'search_attempt':
                if tier != 'deep':
                    raise Refusal('--stale search_attempt is deep-only')
                got = ST.stale_search_attempts(C.status.SEARCH_LABEL['deep'])
            elif cls == 'master':
                if tier != 'deep':
                    raise Refusal('--stale master is deep-only')
                K.need_inputs([C.paths.COADD_SB_ROOT], 'stale master')
                got = [dict(item=r['lst'] + '_' + r['band'], kind='cell', status='stale',
                            reason=f"constituent {r['date']} excluded after the master was written ({r['note']})",
                            step='masters', root=r['file'], detail=r['verdict'])
                       for r in XC.working_tree_audit(COV.deep_universe(ctx.deep_spec()), ex, workers=a.workers)
                       if r['verdict'] == 'STALE']
            elif cls == 'master_coverage':
                if tier != 'deep':
                    raise Refusal('--stale master_coverage is deep-only')
                K.need_inputs([C.paths.COADD_SB_ROOT, C.paths.PHASE2_ROOT], 'stale master_coverage')
                got = []
                for cell, d, cd, remedy, det in XC.coverage_gaps(COV.deep_universe(ctx.deep_spec()), ex, workers=a.workers):
                    if remedy == 'unjudged':
                        got.append(dict(item=cell, kind='cell', status='unjudged', step='masters', root=cd, detail='',
                                        reason=f'the coverage of the cell\'s master cannot be judged: {det}'))
                        continue
                    how = ('Phase 2 first had it after the build: admit it (masters --stages admit-prep,admit) or rebuild '
                           'the cell\'s masters' if remedy == 'admit or rebuild' else
                           f'only a rebuild of the cell\'s masters takes it in (a lifted exclusion, or an admission refused: {det})')
                    got.append(dict(item=cell, kind='cell', status='stale', step='masters', root=cd, detail=d,
                                    reason=f'night {d} admissible now (Phase 2 data, not excluded) is in neither the '
                                           f'master\'s record nor its admissions - {how}'))
            elif cls == 'frame_time':
                got, src = ST.stale_frame_time(a.cells)
                u.add_input(src, extra={'kind': 'code'})
            elif cls == 'prefit':
                from exopipe.phase3.lib.status import steps_snapshot as SN
                res, tab = SN.run_eligibility(out, a.cells_file, None, True, u, ctx.log(name))
                if not res.ok:
                    raise Refusal(f'eligibility audit failed: {res.reason}')
                u.add_output(tab, 'snapshot_eligibility')
                got = ST.stale_prefit(tab)
            elif cls == 'sublst_generation':
                got = ST.stale_sublst_generation()
            elif cls == 'qa_night_pages':
                got = ST.stale_night_pages()
            elif cls == 'ledger':
                from exopipe.phase3.lib.status import steps_ledger as LG
                rc, text, fresh = LG.run_ledger(out, u, ctx.log(name), ctx.allowed)
                if rc != 0:
                    raise Refusal(f'fresh inventory failed (rc {rc}): {text[-300:]}')
                u.add_output(fresh, 'processing_ledger')
                got = LG.ledger_diff(fresh, C.status.LEDGER)
            for r in got:
                r['class'] = cls
            rows += got
        table = K.place(out, 'status', K.narrowed(name, f'stale_{tier}_{"_".join(classes)}.csv'))
        K.write_csv(table, ST.COLUMNS, rows)
        u.add_output(table, 'stale_table')
        kinds = sorted({r['kind'] for r in rows})
        if len(kinds) == 1:
            kind = kinds[0]
        elif rows:
            kind = a.emit_kind or 'unit'
            u.note(f'stale classes of different item kinds {kinds}: the item list holds the {kind} items only')
        else:
            kind = 'unit' if classes == ['record'] else ctx.emit_kind()
        items = [r['item'] for r in rows if r['status'] == 'stale' and r['kind'] == kind]
        ctx.emit(u, 'stale', kind, items, table, 'status=stale', extra_head=head or None)
        n = sum(1 for r in rows if r['status'] == 'stale')
        print(f'stale ({",".join(classes)}): {n} stale, {len(rows) - n} unfinished/unreadable -> {table}')
        for r in rows[:20]:
            print(f"  {r['class']:16s} {r['item']:28s} {r['status']:10s} {r['reason'][:90]}")
        ctx.strict_check(u, n, 'stale items')


def _stage_served_diff(ctx):
    import json
    from exopipe.phase3.lib.status import steps_serveddiff as SD
    K, out, tier = ctx.K, ctx.out, ctx.tier
    A, B = _served_pair(ctx)
    name = ctx.unit('report', f'serveddiff_{os.path.basename(A)}_{os.path.basename(B)}')
    with ctx.run.unit(name, tier=tier, stage='served-diff', A=A, B=B) as u:
        K.exclusions()
        rows, summary, inputs = SD.served_diff(A, B, tier)
        for p in inputs:
            u.add_input(p)
        base = K.place(out, 'status', f'served_diff_{os.path.basename(A)}_{os.path.basename(B)}')
        K.write_csv(base + '_rows.csv', SD.COLUMNS, rows)
        K.write_text(base + '_summary.json', json.dumps(summary, indent=1, sort_keys=True) + '\n')
        K.write_text(base + '_summary.txt', SD.summary_text(summary))
        u.add_output(base + '_rows.csv', 'served_diff')
        u.meta['summary'] = summary
        print(SD.summary_text(summary), end='')


def _stage_cell_load(ctx):
    from exopipe.phase3.lib.status import steps_report as RP
    K, C, a, out, tier = ctx.K, ctx.C, ctx.args, ctx.out, ctx.tier
    label = a.run_label or C.status.SEARCH_LABEL.get(tier)
    with ctx.run.unit(ctx.unit('report', f'cellload_{label}'), tier=tier, stage='cell-load', run=label) as u:
        K.exclusions()                                    # S-G1 in every stage
        rd = K.resolve_run(label, tier)
        K.need_inputs([rd], 'cell-load')
        rows, inputs = RP.cell_load(rd)
        for p in inputs:
            u.add_input(p)
        dest = K.place(out, 'status', C.status.CELL_LOAD_FILE.format(label=label))
        K.write_csv(dest, C.status.CELL_LOAD_COLUMNS, rows)
        u.add_output(dest, 'cell_load_review', run_dir=rd)
        print(f'cell-load {label}: {len(rows)} (Stokes, LST, night, band) rows -> {dest}')


# =====================================================================================================================
# qa, conformance, pages, ledger, register, snapshot
# =====================================================================================================================

def _qa_stage(ctx, s):
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.status import steps_qa as QA
    K, C, a, out = ctx.K, ctx.C, ctx.args, ctx.out
    name = ctx.unit('qa', s)
    with ctx.run.unit(name, stage=s) as u:
        ex = K.exclusions()
        dd_before = QA.sha1_of(C.paths.DATA_DECISIONS)
        u.add_input(C.paths.DATA_DECISIONS)
        rd = None
        if s == 'provenance':
            if not a.run_label:
                raise Refusal('S-G12: qa --stages provenance needs --run')
            rd = K.resolve_run(a.run_label, 'deep')
        K.need_inputs(QA.stage_inputs(s, rd), f'qa {s}')
        log = ctx.log(name)
        if s == 'qa-table':
            res, dest = QA.run_qa_table(out, u, log)
            if not res.ok:
                raise Refusal(f'build_qa_table.py failed: {res.reason}; {res.tail[-300:]}')
            u.add_output(dest, 'qa_table')
        elif s == 'solar':
            rc, text, cand, diff = QA.run_solar(out, ctx.utc, u, log, ctx.allowed)
            if rc != 0:
                raise Refusal(f'solar_contamination_audit.py rc {rc}: {text[-300:]}')
            u.add_output(cand, 'solar_audit')
            u.add_output(diff, 'csv_table', required_columns=list(QA.SOLAR_DIFF_COLUMNS))
        elif s == 'beam':
            res, d = QA.run_beam(out, u, log)
            if not res.ok:
                raise Refusal(f'beam_audit.py failed: {res.reason}')
            u.add_output(os.path.join(d, 'beam_audit_all.csv'), 'beam_audit')
            u.add_output(os.path.join(d, 'beam_audit_flagged.csv'), 'csv_table', required_columns=list(QA.BEAM_COLUMNS))
        elif s == 'noise':
            res, dest = QA.run_noise(out, u, log)
            if not res.ok:
                raise Refusal(f'qa_noise_vs_freq_plot.py failed: {res.reason}')
            u.add_output(dest, 'noise_plot')
            u.meta['tail'] = res.tail[-1500:]
        elif s == 'outliers':
            unreadable = []
            rows, files = QA.date_outliers(ex, unreadable)
            u.meta['residual_records'] = len(files)
            u.meta['existing_rule'] = QA.OUTLIER_EXISTING_RULE
            if unreadable:                                # S-G16: skipped as the tool skips them, counted and named
                u.meta['unreadable_residual_records'] = unreadable[:50]
                u.note(f'{len(unreadable)} residuals_provisional.csv files skipped (unreadable), as '
                       f'propagate_date_outliers.py:25-26')
            dest = K.place(out, 'QA', C.status_qa.OUTLIER_PROPOSALS)
            K.write_csv(dest, QA.OUTLIER_COLUMNS, rows)
            u.add_output(dest, 'date_outlier_proposals')
            print(f'outliers: {len(rows)} proposals for Gregg from {len(files)} cell records -> {dest}')
        elif s == 'bandpass':
            res, dest, tables = QA.run_bandpass(out, a.work, u, log)
            if not os.path.exists(dest):
                raise Refusal(f'bandpass: no shard produced a row ({len(tables)} tables)')
            # shard exit codes are in meta.bandpass.shard_rc; completeness is the content test (one row per listed table)
            u.add_output(dest, 'bandpass_phase_qa', tables_list=u.meta['bandpass']['list'])
        elif s == 'provenance':
            for script, rc, dest, tail in QA.run_provenance(out, rd, u, log, ctx.allowed):
                if rc != 0:
                    raise Refusal(f'{script} rc {rc}: {tail}')
                u.add_output(dest, 'provenance_audit')
        u.check('S-G3.data_decisions_untouched', QA.sha1_of(C.paths.DATA_DECISIONS) == dd_before,
                value=dd_before[:12], detail=C.paths.DATA_DECISIONS)


for _s in STAGES['qa']:
    globals()['_stage_' + _s.replace('-', '_')] = (lambda s: (lambda ctx: _qa_stage(ctx, s)))(_s)


def _stage_membership(ctx):
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.status import steps_hub as HB
    K, C, a, out = ctx.K, ctx.C, ctx.args, ctx.out
    if ctx.items_selected():
        raise Refusal('conformance runs over the whole working tree (as conformance_check.py); no --cells')
    with ctx.run.unit(ctx.unit('conformance', 'membership'), stage='membership') as u:
        K.exclusions()
        K.need_inputs([C.paths.COADD_SB_ROOT], 'conformance')
        drift = HB.membership_drift()
        if drift:
            raise Refusal('S-G6: branch membership module differs from production: ' + '; '.join(drift[:3]))
        rl = a.round or C.status_conformance.ROUND_LIST
        rc, text, latest, hist, page = HB.run_conformance(out, rl, u, ctx.log(u.name), ctx.allowed)
        if rc != 0:
            raise Refusal(f'conformance_check.main rc {rc}: {text[-400:]}')
        u.add_output(latest, 'membership_conformance')
        u.add_output(hist, 'csv_table', required_columns=list(HB.HISTORY_COLUMNS), min_rows=1)
        u.add_output(page, 'status_page')
        n = sum(1 for r in K.read_csv(latest) if r['status'] == 'FAIL')
        print(text.strip().splitlines()[-1] if text.strip() else f'conformance -> {latest}')
        ctx.strict_check(u, n, 'failing cells')


def _stage_nights(ctx):
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.status import steps_hub as HB
    K, C, a, out = ctx.K, ctx.C, ctx.args, ctx.out
    sel = a.epochs or a.nights or a.cells
    nights = []
    if sel:
        from exopipe.phase3.core import cells as CL
        nights = CL.expand(sel)
        for n in nights:
            CL.parse_epoch(n)
    with ctx.run.unit(ctx.unit('pages', 'nights', nights or None), stage='nights') as u:
        K.exclusions()
        K.need_inputs([C.paths.COADD_SB_ROOT], 'pages nights')
        if a.check:
            hub = K.place(out, 'review_hub', '').rstrip('/')
            hub = hub if os.path.isdir(os.path.join(hub, 'nights')) else K.hub_dir()
            u.add_output(hub, 'qa_night_pages')
            return
        rc, text, hub = HB.run_nights(out, nights, u, ctx.log(u.name), ctx.allowed)
        if rc != 0:
            raise Refusal(f'build_qa_pages.main rc {rc}: {text[-400:]}')
        if nights:
            missing = [n for n in nights if not os.path.exists(os.path.join(hub, 'nights', f'{n}.html'))]
            u.check('pages.requested_nights_written', not missing, value=len(nights) - len(missing),
                    detail=f'missing {missing[:3]}' if missing else 'every requested night page written')
            u.add_output(hub, 'qa_night_pages', test=False)
            u.note('narrowed run (--only): the full-hub content test P1-P4 applies to a full build; run '
                   '`status pages --check` on the complete hub')
        else:
            u.add_output(hub, 'qa_night_pages')
        print(text.strip().splitlines()[-1] if text.strip() else f'night pages -> {hub}')


def _stage_data_processing(ctx):
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.status import steps_hub as HB
    K, C, out = ctx.K, ctx.C, ctx.out
    with ctx.run.unit(ctx.unit('pages', 'data-processing'), stage='data-processing') as u:
        K.exclusions()
        K.need_inputs([C.paths.COADD_SB_ROOT], 'pages data-processing')
        rc, text, dest = HB.run_data_page(out, u, ctx.log(u.name), ctx.allowed)
        if rc != 0:
            raise Refusal(f'build_data_processing_page.main rc {rc}: {text[-400:]}')
        u.add_output(dest, 'status_page')
        print(text.strip().splitlines()[-1] if text.strip() else dest)


def _stage_ledger(ctx):
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.status import steps_ledger as LG
    K, C = ctx.K, ctx.C
    if ctx.items_selected():
        raise Refusal('S-G10: the ledger is whole-survey only (no --cells/--lsts/--nights/--epochs)')
    with ctx.run.unit(ctx.unit('ledger', 'ledger'), stage='ledger') as u:
        K.exclusions()
        u.add_input(C.paths.DATA_DECISIONS)
        K.need_inputs([C.paths.PHASE1_ROOT, C.paths.PHASE2_ROOT], 'ledger')
        rc, text, led = LG.run_ledger(ctx.out, u, ctx.log(u.name), ctx.allowed)
        if rc != 0:
            raise Refusal(f'inventory / ledger build rc {rc}: {text[-400:]}')
        u.add_output(led, 'processing_ledger')
        print(f'ledger -> {led}')


def _stage_register(ctx):
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.status import steps_hub as HB
    K, a = ctx.K, ctx.args
    with ctx.run.unit(ctx.unit('register', 'register'), stage='register') as u:
        K.exclusions()
        if not a.extra:
            raise Refusal('register needs the audit-slice CSVs (positional AUDIT_CSV ...)')
        K.need_inputs(list(a.extra), 'register')
        rc, text, reg, page = HB.run_register(ctx.out, a.extra, u, ctx.log(u.name), ctx.allowed)
        if rc != 0:
            raise Refusal(f'build_ratified_register.main rc {rc}: {text[-400:]}')
        u.add_output(reg, 'ratified_register')
        u.add_output(page, 'status_page')
        print(text.strip().splitlines()[-1] if text.strip() else reg)


def _stage_eligibility(ctx):
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.status import steps_snapshot as SN
    K, C, a = ctx.K, ctx.C, ctx.args
    with ctx.run.unit(ctx.unit('eligibility', 'eligibility'), stage='eligibility') as u:
        K.exclusions()
        K.need_inputs([C.paths.SNAPSHOT_FLEET_OUT], 'eligibility')
        res, dest = SN.run_eligibility(ctx.out, a.cells_file, a.exclude_file, a.no_store, u, ctx.log(u.name))
        if not res.ok:
            raise Refusal(f'snapshot_batch_audit.py failed: {res.reason}; {res.tail[-300:]}')
        u.add_output(dest, 'snapshot_eligibility')
        print(res.tail.strip().splitlines()[-1] if res.tail.strip() else dest)


def _stage_batch_list(ctx):
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.status import steps_snapshot as SN
    K, a = ctx.K, ctx.args
    with ctx.run.unit(ctx.unit('batch-list', 'batch-list'), stage='batch-list') as u:
        K.exclusions()
        if not a.base:
            raise Refusal('S-G8: batch-list needs --base FILE in the refactor window')
        if not a.list_out:
            raise Refusal('batch-list needs --list-out FILE')
        u.add_input(a.base)
        res = SN.run_batch_list(a.list_out, a, u, ctx.log(u.name))
        if not res.ok:
            raise Refusal(f'snapshot_batch_lists.py failed: {res.reason}; {res.tail[-300:]}')
        sha = next((ln.split('=', 1)[1] for ln in res.tail.splitlines() if ln.startswith('CELLS_FILE_SHA256=')), None)
        import hashlib
        u.check('batch_list.sha256_printed', sha == hashlib.sha256(open(a.list_out, 'rb').read()).hexdigest(),
                value=sha, detail='CELLS_FILE_SHA256 printed by the script = sha256 of the list')
        u.add_output(a.list_out, 'snapshot_batch_list', all=bool(a.all), max_new=int(ctx.C.status_snapshot.MAX_NEW),
                     base=os.path.abspath(a.base))
        print(res.tail.strip())
