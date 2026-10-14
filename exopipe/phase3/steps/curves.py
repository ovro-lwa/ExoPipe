"""exopipe3 curves - light-curve products (snapshot tier): every Master Gaussian and target position, in every 10-s frame.

    exopipe3 curves --cells 18h_2025-04-11 --out DIR                  # archive frames (default curves.FRAME_SOURCE), any node
    exopipe3 curves --epochs 18h_2025-04-11 --out DIR                 # per-hour chain item (DESIGN 12.1)
    exopipe3 curves --cells 18h_2025-04-11 --stage-root /fast/gh/main/phase3/snapshot_stage/18h_2025-04-11 --out DIR   # staged
    exopipe3 curves --cells 18h_2025-04-11_36MHz --stage-root DIR --out DIR --supersede redo1   # one subband, replace
    exopipe3 curves --cells '18h_*' --stage-root '/fast/stage/{cell}' --out DIR --resume       # several staged cells
    exopipe3 curves --cells 18h_2025-04-11 --dry-run --out /tmp/x                              # plan only

Unit of work: one snapshot night cell <LST>_<YYYY-MM-DD> (core.cells.parse_night_cell; an item <cell>_<sb> or --scopes
limits the subbands; an --epochs item <LST>_<date> is the snapshot cell of that hour). One run record per unit,
<out>/_records/curves/<cell>[_<sb>+...][_I].json; inside a unit the planned items are (subband, Stokes) pairs.
Tiers: snapshot only - the sub-LST per-source curves stay in `catalog --tier sublst` (they decide the tie), and the deep tier
has no light-curve product (forced tables and follow-up curves are `forced` / `photometry` / `followup`).

Frames: curves.FRAME_SOURCE 'archive' (default; the production regeneration recipe, exopipe_ops 3035b47): the extractor reads
the Phase-1 archive frames itself (clv15 e0785b7 read_frame; bit-identical to staged frames, 10,129/10,129 rows) - no
staging, any node. 'staged' (the fleet recipe, v9:366-370) or --stage-root: frames staged node-local first; without
--stage-root the staged cell is frames' staging <scratch root>/<frames_snapshot.STAGE_SUBDIR>/<cell> (curves.STAGE_ROOT_AUTO;
v9:87), checked by gate G4.
Per hour (DESIGN 12.1; chains 'epoch' and 'snapshot-cell' in runner.CHAINS): after `flow` and `search`:
`curves --tier snapshot --epochs <LST>_<date>`. Intermittent (12.2): backfill or regeneration of many cells,
`exopipe3 run submit ... -- curves --cells @list --resume`.

Inputs (production paths; shared roots from config.paths, the rest in `exopipe3 config --section curves`):
  own Masters        masters_snapshot.LINK_ROOT/<LST>_<sb>/coadd_final/I_deep_Taper_R0_<sb>.fits (header only: G17)
  frames             the Phase-1 archive (FRAME_SOURCE archive), or --stage-root DIR / frames' staging: <DIR>/<sb>/
                     decompressed frames as the fleet / `frames --tier snapshot` stage them; the extractor lists them itself
  search outputs     paths.SNAPSHOT_FLEET_OUT/<cell>/admission_<sb>.csv, runinfo_<sb>_<S>_median.json,
                     nightrefused_<sb>_<S>_median.json
  search finished    curves.FLEET_CLAIMS_ROOT/<cell>/done (window; curves.SEARCH_DONE_EVIDENCE)
  Master catalogues  curves.MASTER_CATALOG_DIR/master_R075_<LST>_<sb>.gaul.csv
  target samples     paths.CLV15_ROOT/{Hot_Warm_Jupiters,10pc,25pc,LPTs,flare_ucd,sydney_stars,pulsars_atnf}.csv
  flow fields        paths.SNAPSHOT_FLOW_ROOT/<cell>/flow_<sb>_<ts>.npy (nearest valid instant when a frame's own is absent)
  PB library         curves.PB_LIBRARY/PB_<sb>_bin<k>.fits (model beam per 10-minute bin)
  exclusions         paths.DATA_DECISIONS (core.exclusions)
Outputs:
  <out>/<cell>/<cell>_<sb>_<S>_lightcurves.fits   product class light_curve (content tests LC1-LC6, lib/curves/tests.py):
                     FLUX, RMS float32 [rows x frames] Jy/beam raw (no PB correction, no scale), PB [rows x 6], SOURCES,
                     FRAMES; header FLUXSCAL / NREFSCAL (Stokes V carries the Stokes I values). Default <out> =
                     curves.PRODUCT_ROOT (/lustre/gh/main/exopipe_ops/snapshot_lightcurves).
  <out>/_records/curves/<unit>.json   pairs and outcomes, inputs with fingerprints (incl. the in-place clv15 modules the
                     extractor processes imported), admitted subbands, fallback counts per pair, log tails, checks LC7/LC8
  <out>/_superseded/<TAG>/<cell>/     old tables + MANIFEST.csv (only with --supersede)
  scratch: <work>/curves_extract_*/ (extractor tables and logs; removed when the unit ends without error)

Failure modes (exit 2 = refused, with the gate named; item refusals are recorded per pair and the unit completes):
  unit refused   G1 search not finished (a searched subband/Stokes without runinfo or night-refusal record, or no fleet
                 'done'); G2 data_decisions.csv missing; G4 (staged frames only) stage missing or incomplete (subband directory, leftovers, count
                 != .filelist); G12 publish conflict (a differing table already published, no --supersede); G13 the in-place
                 module's literals differ from config (protected keys: no override exists); G14 CHAIN_RUN_DIR into a
                 production root without a ruling; G16 --stokes V alone; LC8 a published table of an excluded (LST,
                 subband, date); a table failing a content test; --out under a production tree while
                 paths.PRODUCTION_WRITES is False; site PYTHON['py36'] or the clv15 legacy root missing.
  pair refused   G4 (staged frames only) a frame of the table was not staged (archive copy read); G6 no Master catalogue; G7 no row inside the
                 frame; G8 frames do not share one WCS; G9 no flow field; G10 V without its Stokes I table / V scale
                 differs from I; G11 no frames of that Stokes; G12 conflict; G17 (archive frames only) the subband's
                 archive frames are off its own Master's grid - the fleet regrids them in the staging (pilot era,
                 v9:126-139), so only the fleet recipe (--stage-root) extracts them; a content test failed before
                 publishing (LC1-LC6 incl. the regeneration recipe's T1-T5, exopipe_ops 3035b47: MCATSHA present and
                 equal to the catalogue's sha256, Master rows = the catalogue's Gaussians inside the first frame, admitted
                 flags = the admission table, V scale = I; with --supersede also the replaced table's frame list and its
                 values at identical positions - light_curve.same_as_previous).
  not planned    (recorded) subbands not searched (no admission table), excluded (G2), flow-unregistered (G3).
  exit 3         no admission table in the cell (G15), or no planned subband. exit 1: an extractor error (log tail kept).
Fallbacks kept as today until Gregg rules (B-23, B-30, B-58), counted per pair: frames absent from the admission table are
admitted; no admitted frame -> scale from every frame; a missing sample file adds no rows; zero-visibility frames kept.

Refactor window (DESIGN 11.1; refactor decisions 1-2): clv15/phase3_snapshot_lightcurves.py runs IN PLACE with production's
arguments (v9:366-370; without --stage = exopipe_ops snapshot_lightcurves_regen_20261013.sh:35-36) through core.legacy.run
under the site interpreter 'py36'; curves.PROCS subbands at a time, each the Stokes I process then the Stokes V process (the
regeneration runs --stokes I,V in one process: its V pass takes the I scale from memory, here from the I table's header
written just before - the same FLUXSCAL / NREFSCAL, LC:187-200).
NOT YET: the `stage` stage (staging by this step itself, `--stages stage,extract,publish`) needs the `frames --tier snapshot`
stage function (spec B7) and refuses until it exists. --workers is ignored (cells run one after another).
"""
import os
import sys

from exopipe.phase3.lib.curves import tests as _content_tests  # noqa: F401  (registers the light_curve content tests)

SUMMARY = 'light-curve products (snapshot: every Master Gaussian and target position in every 10-s frame, per subband and Stokes)'
TIERS = ('snapshot',)
SELECTORS = ('tier', 'cells', 'epochs', 'scopes', 'stokes', 'exclude_cells')
CONFIG = ('curves', 'frames_snapshot', 'masters_snapshot', 'paths', 'exclusions', 'instrument', 'runner', 'site')
WRITES = True
STAGES = ('stage', 'extract', 'publish')
PYTHON = 'py36'                 # site interpreter of the in-place extractor (production: v9:44, BF:15; refactor decision 2)

# The only values of the protected fallback keys the in-place module implements (any other needs absorption + ruling).
_WINDOW_ONLY = {'ADMISSION_MISSING': 'admit', 'NO_ADMITTED_SCALE': 'all-frames', 'MISSING_SAMPLE_FILE': 'skip',
                'FRAME_LISTER': 'legacy-discover-frames', 'FRAME_TIME': 'file-name', 'PB_TIME_MODEL': 'per-10min-bin',
                'V_TAKES_I_SCALE': True, 'STOKES_PER_PROCESS': ('I', 'V')}


def add_arguments(p):
    g = p.add_argument_group('curves')
    g.add_argument('--stage-root', metavar='DIR',
                   help="existing node-local staged cell (<DIR>/<subband>/ frames, as `frames --tier snapshot --stages "
                        "stage,regrid` or the fleet stage them); '{cell}' in DIR is replaced per cell (needed for more "
                        'than one cell); nothing under DIR is deleted. Default: frames\' staging of this node, '
                        '<scratch root>/<frames_snapshot.STAGE_SUBDIR>/<cell> (curves.STAGE_ROOT_AUTO)')
    g.add_argument('--stages', default=None,
                   help="stages to run: 'extract,publish' (default) or 'stage,extract,publish' (self-staging; refuses "
                        'until the frames stage function exists, spec B7); declared here until cli.py adds --stages')
    g.add_argument('--keep-stage', action='store_true', default=None,
                   help='keep a stage this step made itself (default curves.KEEP_STAGE: removed after the unit)')
    g.add_argument('--supersede', metavar='TAG',
                   help='replace differing tables already published under <out>/<cell>/: the old files move to '
                        '<out>/_superseded/<TAG>/<cell>/ with a MANIFEST.csv; without it a differing table refuses (G12)')


def default_out(args):
    from exopipe.phase3.config import CFG
    return CFG.curves.PRODUCT_ROOT


# ---------------------------------------------------------------------------------------------------------------------
# argument and configuration gates
# ---------------------------------------------------------------------------------------------------------------------

def _stokes(spec):
    from exopipe.phase3.core.record import Refusal
    if spec is None:
        return ('I', 'V')
    toks = [s.strip().upper() for s in str(spec).split(',') if s.strip()]
    if set(toks) == {'V'}:
        raise Refusal('G16: --stokes V alone - Stokes V carries the Stokes I flux scale (contract 3.2); run I,V or I')
    if set(toks) == {'I', 'V'}:
        return ('I', 'V')
    if set(toks) == {'I'}:
        return ('I',)
    raise ValueError(f'--stokes {spec!r}: give I,V (default) or I')


def _archive(args):
    """True when the extractor reads the Phase-1 archive frames itself (curves.FRAME_SOURCE 'archive', no --stage-root)."""
    from exopipe.phase3.config import CFG
    return CFG.curves.FRAME_SOURCE == 'archive' and not args.stage_root


def _stages(args, auto=True):
    if args.stages:
        st = tuple(s.strip() for s in args.stages.split(',') if s.strip())
    else:
        st = ('extract', 'publish') if (args.stage_root or auto or _archive(args)) else STAGES
    unknown = [s for s in st if s not in STAGES]
    if unknown:
        raise ValueError(f'--stages: unknown {unknown}; stages are {", ".join(STAGES)}')
    if ('extract' in st) != ('publish' in st):
        raise ValueError('--stages: extract and publish run together (publish copies only the tables this invocation '
                         'extracted and tested; extract alone would leave unpublished tables in scratch)')
    if 'stage' in st and args.stage_root:
        raise ValueError('--stages stage with --stage-root: the stage exists already; drop one of them')
    if 'stage' in st and _archive(args):
        raise ValueError("--stages stage: curves.FRAME_SOURCE is 'archive' - the extractor reads the archive frames, "
                         "nothing to stage (give --stage-root or set curves.FRAME_SOURCE=staged for the fleet recipe)")
    if 'extract' in st and 'stage' not in st and not args.stage_root and not auto and not _archive(args):
        raise ValueError('--stages extract,publish needs --stage-root (curves.STAGE_ROOT_AUTO is False)')
    return st


def _stage_root(args, cell, scratch_root, n_cells):
    """The staged cell of ``cell``: --stage-root ('{cell}' replaced), else frames' node-local staging (STAGE_ROOT_AUTO)."""
    from exopipe.phase3.config import CFG
    if args.stage_root:
        if n_cells > 1 and '{cell}' not in args.stage_root:
            raise ValueError(f"--stage-root {args.stage_root} names ONE staged cell but {n_cells} cells are selected: "
                             f"write '{{cell}}' where the cell goes")
        return args.stage_root.replace('{cell}', cell)
    if not scratch_root:
        return None
    return os.path.join(scratch_root, CFG.frames_snapshot.STAGE_SUBDIR, cell)


def _override_keys(CFG):
    return [(o[0], o) for o in CFG.overrides]


def _guard_roots(CFG):
    """Product and input roots of this step that are PRODUCTION trees (spec B2: paths.PRODUCTION_ROOTS lacks them): the
    configured values and, for overridden keys, the original values."""
    C, P = CFG.curves, CFG.paths
    keys = {'curves.PRODUCT_ROOT': C.PRODUCT_ROOT, 'curves.MASTER_CATALOG_DIR': C.MASTER_CATALOG_DIR,
            'curves.FLEET_CLAIMS_ROOT': C.FLEET_CLAIMS_ROOT, 'curves.PB_LIBRARY': C.PB_LIBRARY,
            'paths.SNAPSHOT_FLEET_OUT': P.SNAPSHOT_FLEET_OUT, 'paths.SNAPSHOT_FLOW_ROOT': P.SNAPSHOT_FLOW_ROOT,
            'paths.CLV15_ROOT': P.CLV15_ROOT}
    roots = set(keys.values())
    for key, o in _override_keys(CFG):
        if key.split('[')[0] in keys and isinstance(o[1], str):
            roots.add(o[1])
    return sorted(r for r in roots if r)


def _under(path, roots):
    real = os.path.realpath(path)
    for r in roots:
        rr = os.path.realpath(r)
        if real == rr or real.startswith(rr.rstrip('/') + '/'):
            return r
    return None


def _production_out(CFG, out):
    """The production root holding ``out`` (paths.PRODUCTION_ROOTS or this step's production trees), or None."""
    return CFG.production_root_of(out) or _under(out, _guard_roots(CFG))


def _config_gates(CFG, out):
    """G13 (window-only values), ruled overrides into this step's production trees, G14 (chain rows), the production-write
    guard. Returns the list of refusals (dry-run prints them; a real run raises them). Overrides of PROTECTED_KEYS never
    get here: config.py refuses them."""
    C = CFG.curves
    problems = []
    for k, v in _WINDOW_ONLY.items():
        if getattr(C, k) != v:
            problems.append(f'G13: curves.{k} = {getattr(C, k)!r}: only {v!r} exists during the refactor window')
    prod = _production_out(CFG, out)
    if prod:
        for key, o in _override_keys(CFG):
            base = key.split('[')[0]
            if base.startswith('curves.') and CFG.ruled(base) and not str(o[4] or '').startswith('ruling:'):
                problems.append(f'{base} carries a ruling ({", ".join(CFG.ruled(base))}) and was overridden without one '
                                f'while --out {out} is under the production tree {prod} (DESIGN 11.4)')
        if C.CHAIN_RUN_DIR and not any(k.split('[')[0] == 'curves.CHAIN_RUN_DIR' and str(o[4] or '').startswith('ruling:')
                                       for k, o in _override_keys(CFG)):
            problems.append(f'G14: curves.CHAIN_RUN_DIR = {C.CHAIN_RUN_DIR!r} adds served chain rows (contract S-3 (a), '
                            f'open) into the production tree {prod} without --ruling-for curves.CHAIN_RUN_DIR=<ledger ID>')
        if not CFG.paths.PRODUCTION_WRITES:
            problems.append(f'--out {out} is under the production tree {prod} and paths.PRODUCTION_WRITES is False '
                            f'(refactor window): write parity/scratch output to your own root')
    return problems


def _stage(cell, plan, args):
    """The `stage` stage: copy + decompress the cell's Phase-1 frames to node-local scratch and regrid pilot-era frames, by
    calling the `frames --tier snapshot` stage function (spec section 1, B7). That function does not exist yet."""
    from exopipe.phase3.core.record import Refusal
    raise Refusal(f'G4/B7: this step cannot stage {cell} itself yet - the `frames --tier snapshot` stage function it calls '
                  f'does not exist; stage the cell with `exopipe3 frames --tier snapshot --stages stage,regrid --cells '
                  f'{cell}` (or the fleet) and pass --stage-root, or drop --stages stage')


def _archive_grid(cell, sbs):
    """{subband: G17 reason} for the subbands whose ARCHIVE frames are off their own Master's grid (lib/curves/workers.
    archive_grid_problem; the fleet regrids them in the staging, v9:126-139). A subband whose archive cannot be listed is
    left to the extractor (it raises, and the pair is an error, as production)."""
    from exopipe.phase3.lib.curves import workers as W
    out = {}
    for sb in sbs:
        try:
            why = W.archive_grid_problem(cell, sb)
        except Exception:
            why = None
        if why:
            out[sb] = why
    return out


def _environment(CFG, args, stages):
    """Run-level facts and refusals: interpreter, clv15 root, extractor script, G13 lock, scratch root."""
    from exopipe.phase3.core import env, fitsio, legacy, record
    from exopipe.phase3.lib.curves import lock, workers as W
    e = {'problems': [], 'python': None, 'clv15': None, 'script': None, 'code_files': [], 'scratch_root': None,
         'scratch_note': None}
    try:
        e['python'] = legacy.interpreter(PYTHON)
        if not os.access(e['python'], os.X_OK):
            e['problems'].append(f"site PYTHON[{PYTHON!r}] = {e['python']} is not executable")
    except record.Refusal as x:
        e['problems'].append(f'{x} - the extractor runs under the production interpreter (Python 3.6.13; v9:44, BF:15)')
    try:
        e['clv15'] = env.legacy_root('clv15')
        e['script'] = os.path.join(e['clv15'], W.SCRIPT)
        ok, probs, files = lock.check(e['clv15'], CFG.curves, CFG.paths)
        e['code_files'] = files
        e['problems'] += [f'G13: {p}' for p in probs]
    except KeyError as x:
        e['problems'].append(str(x.args[0] if x.args else x))
    try:
        e['scratch_root'] = fitsio.scratch_root(args.work)
    except record.Refusal as x:
        e['scratch_note'] = str(x)
        if 'extract' in stages:
            e['problems'].append(e['scratch_note'])
    return e


# ---------------------------------------------------------------------------------------------------------------------
# plan printing (--dry-run)
# ---------------------------------------------------------------------------------------------------------------------

def _print_plan(name, plan, out, stokes, stage_root, stages, g1, stage_issues, notes):
    from exopipe.phase3.lib.curves.plan import table_name
    p = print
    p(f'== curves unit {name}: cell {plan["cell"]}, Stokes {",".join(stokes)}, stages {",".join(stages)}')
    p(f'   search outputs {plan["search_dir"]}: {len(plan["searched"])} searched subbands')
    p(f'   G1 search finished: {"ok" if g1[0] else "REFUSE"} - {g1[1]}')
    p(f'   staged cell: {stage_root or "-"}')
    for sb, i in plan['subbands'].items():
        recs = ','.join(f'{st}:{"runinfo" if st in i["runinfo"] else "nightrefused" if st in i["nightrefused"] else "-"}'
                        for st in ('I', 'V'))
        extra = ''
        if i['status'] == 'planned':
            extra = (f' catalogue {"ok" if i["catalog_exists"] else "MISSING (G6: pairs refused)"}; flow fields '
                     f'{i["flow_files"]}' + (' (G9 likely)' if not i['flow_files'] else ''))
            if stage_root and os.path.isdir(stage_root):
                extra += f'; stage {stage_issues.get(sb, "ok")}'
        p(f'   {sb:6s} {i["status"]:17s} {recs:30s}{extra}{(" - " + i["reason"]) if i["reason"] else ""}')
    if plan['planned']:
        n_out = len(plan['planned']) * len(stokes)
        p(f'   outputs: {n_out} light_curve tables {os.path.join(out, plan["cell"], table_name(plan["cell"], "<sb>", "<S>"))}'
          f' for <sb> in {",".join(plan["planned"])}, <S> in {",".join(stokes)}')
    for n in notes:
        p(f'   {n}')


# ---------------------------------------------------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------------------------------------------------

def run(args, run):
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import exclusions, record
    from exopipe.phase3.lib.curves import items, plan as P
    C = CFG.curves
    try:
        stokes = _stokes(args.stokes)
        stages = _stages(args, auto=C.STAGE_ROOT_AUTO)
        if args.tier not in C.TIERS:
            raise ValueError(f'--tier {args.tier}: curves exists for {", ".join(C.TIERS)} only')
        units = items.select(args.cells, args.epochs, args.exclude_cells, args.scopes, CFG.paths.SNAPSHOT_FLEET_OUT)
        if args.stage_root and len(units) > 1 and '{cell}' not in args.stage_root:
            raise ValueError(f"--stage-root {args.stage_root} names ONE staged cell but {len(units)} cells are selected: "
                             f"write '{{cell}}' where the cell goes")
    except ValueError as e:
        print(f'error: {e}', file=sys.stderr)
        return record.EXIT_ERROR
    if not units:
        print('error: no cells - give --cells or --epochs (snapshot cells <LST>_<YYYY-MM-DD>[_<subband>])', file=sys.stderr)
        return record.EXIT_ERROR
    out = run.out_root
    E = _environment(CFG, args, stages)
    problems = _config_gates(CFG, out) + E['problems']
    if problems and not run.dry_run:
        raise record.Refusal(problems[0] + (f' (+{len(problems) - 1} more: ' + ' | '.join(problems[1:]) + ')'
                                            if len(problems) > 1 else ''))
    ex = exclusions.load()                                    # G2: a missing / unreadable file refuses (C-09)

    if run.dry_run:
        rc = record.EXIT_DONE
        for p_ in problems:
            print(f'REFUSE (run level): {p_}')
            rc = record.EXIT_REFUSED
        print(f'out {out}; scratch {E["scratch_root"] or E["scratch_note"]}; extractor {E["script"] or "-"} under '
              f'{E["python"] or "-"} (site PYTHON[{PYTHON!r}]); G13 module literals '
              f'{"match config" if not any(p.startswith("G13") for p in problems) else "DIFFER"}; PROCS {C.PROCS}')
        any_input = False
        for cell, sbs in units:
            name = items.unit_name(cell, sbs, stokes)
            pl = P.plan_cell(cell, sbs, stokes, ex, C, CFG.instrument.ALL_SUBBANDS)
            g1 = P.search_done(pl, C) if pl['searched'] else (False, 'G15: no admission table (cell not searched)')
            issues, notes, g17 = {}, [], {}
            sroot = (_stage_root(args, cell, E['scratch_root'], len(units))
                     if 'stage' not in stages and not _archive(args) else None)
            if _archive(args):
                g17 = _archive_grid(cell, [sb for sb in pl["planned"] if pl["subbands"][sb]["catalog_exists"]])
                notes.append('frames: the Phase-1 archive, read by the extractor (curves.FRAME_SOURCE archive; no staging)')
                notes += [f'{sb} pairs refused - {why}' for sb, why in g17.items()]
            elif 'stage' in stages:
                notes.append('REFUSE G4/B7: the stage stage needs the frames stage function (not yet)')
            elif not sroot:
                notes.append('REFUSE G4: no --stage-root and no scratch root to find frames\' staging')
            elif os.path.isdir(sroot):
                issues = P.stage_problems(sroot, pl)
            else:
                notes.append(f'REFUSE G4: staged cell {sroot} does not exist on this host')
            _print_plan(name, pl, out, stokes, sroot, stages, g1, issues, notes)
            if not pl['planned']:
                print('   NO-INPUT: no planned subband')
                continue
            any_input = True
            if (not g1[0] or issues or any(n.startswith('REFUSE') for n in notes)
                    or (g17 and all(sb in g17 for sb in pl['planned']))):       # every planned pair G17: unit refused
                rc = record.EXIT_REFUSED
        return rc if any_input or rc == record.EXIT_REFUSED else record.EXIT_NO_INPUT

    for cell, sbs in units:
        name = items.unit_name(cell, sbs, stokes)
        pl = P.plan_cell(cell, sbs, stokes, ex, C, CFG.instrument.ALL_SUBBANDS)
        if args.resume and not args.fresh:
            ok, why = run.resume_check(name, inputs=None, admitted=P.admitted(pl))
            if ok:
                u = run.unit(name, tier='snapshot', cell=cell)
                u.status, u.reason = 'skipped-valid', why
                print(f'curves {name}: skipped (resume: {why})', file=sys.stderr)
                continue
            print(f'curves {name}: not resumed ({why})', file=sys.stderr)
        with run.unit(name, tier='snapshot', cell=cell) as u:
            _run_unit(u, args, run, pl, stokes, stages, ex, E,
                      None if _archive(args) else _stage_root(args, cell, E['scratch_root'], len(units)))
    return None


def _run_unit(u, args, run, pl, stokes, stages, ex, E, stage_root):
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import fitsio, record, validate
    from exopipe.phase3.lib.curves import plan as P, publish as PUB, workers as W
    C = CFG.curves
    out = run.out_root
    cell = pl['cell']
    u.add_input(CFG.paths.DATA_DECISIONS, extra={'exclusions': ex.fingerprint()})
    u.meta.update({'stokes': list(stokes), 'stages': list(stages), 'stage_root': stage_root,
                   'frame_source': 'archive' if _archive(args) else 'staged',
                   'interpreter': E['python'], 'extractor': E['script'],
                   'subbands': {sb: {'status': i['status'], 'reason': i['reason']} for sb, i in pl['subbands'].items()},
                   'pairs': {}})
    u.set_admitted(P.admitted(pl))
    if not pl['searched']:
        raise record.NoInput(f'G15: {cell} has no admission table under {pl["search_dir"]} (not searched)')
    if not pl['planned']:
        raise record.NoInput(f'{cell}: no planned subband (' + ', '.join(
            f'{sb} {i["status"]}' for sb, i in pl['subbands'].items() if i['status'] != 'not-searched') + ')')
    ok, detail = P.search_done(pl, C)
    if not ok:
        raise record.Refusal(f'G1: {cell}: {detail}')
    u.note(f'G1: {detail}')
    for path, extra in P.inputs(pl, C, CFG.paths.PHASE1_ROOT, E['code_files']):
        u.add_input(path, extra=extra)
    dst = {}
    for sb in pl['planned']:
        for st in stokes:
            dst[(sb, st)] = u.plan(os.path.join(out, cell, P.table_name(cell, sb, st)), 'light_curve')
    pairs = u.meta['pairs']
    run_sbs = []
    for sb in pl['planned']:                                   # G6 checked in the plan (the module checks it too)
        if not pl['subbands'][sb]['catalog_exists']:
            for st in stokes:
                why = f'G6: no Master catalogue {pl["subbands"][sb]["catalog"]}'
                u.refuse_item(dst[(sb, st)], why, gate='G6')
                pairs[f'{sb}_{st}'] = {'status': 'refused', 'gate': 'G6', 'reason': why}
        else:
            run_sbs.append(sb)
    if _archive(args):                                          # G17: archive frames the fleet would have regridded
        for sb in run_sbs:                                      # the own Masters G17 reads (header): inputs of the unit
            if os.path.exists(W.master_file(pl['lst'], sb)):
                u.add_input(W.master_file(pl['lst'], sb))
        g17 = _archive_grid(cell, run_sbs)
        for sb, why in g17.items():
            for st in stokes:
                u.refuse_item(dst[(sb, st)], why, gate='G17')
                pairs[f'{sb}_{st}'] = {'status': 'refused', 'gate': 'G17', 'reason': why}
        run_sbs = [sb for sb in run_sbs if sb not in g17]
    if not run_sbs:
        return
    # ---- stage ------------------------------------------------------------------------------------------------------
    if 'stage' in stages:
        _stage(cell, pl, args)                                 # refuses (B7) until the frames stage function exists
    if not _archive(args):                                     # G4: the staged cell (fleet recipe); none to check for archive frames
        if not stage_root or not os.path.isdir(stage_root):
            raise record.Refusal(f'G4: staged cell {stage_root or "(none)"} does not exist on {run.host} (pass '
                                 f'--stage-root, or stage it with `exopipe3 frames --tier snapshot --stages stage,regrid '
                                 f'--cells {cell}`, or use curves.FRAME_SOURCE=archive)')
        issues = P.stage_problems(stage_root, dict(pl, planned=run_sbs))
        if issues:
            raise record.Refusal(f'G4: staged cell {stage_root} incomplete: ' + '; '.join(f'{sb}: {v}' for sb, v in
                                                                                        issues.items()))
    # ---- extract: the in-place extractor, PROCS subbands at a time (I then V each) ------------------------------------
    scratch = fitsio.scratch_dir('curves_extract', args.work)
    u.meta['scratch'] = scratch
    specs = [W.make_spec(C, CFG.paths, E['script'], PYTHON, cell, sb, stokes,
                         os.path.join(stage_root, sb) if stage_root else None, os.path.join(scratch, sb)) for sb in run_sbs]

    def progress(sb, res):
        txt = ', '.join(f'{s} {r.get("status")}' + (f' ({r.get("gate")})' if r.get('gate') else '')
                        for s, r in sorted(res.items()))
        print(f'curves {cell} {sb}: {txt}', file=sys.stderr)

    results = W.run_all(specs, C.PROCS, unit=u, on_done=progress)
    # ---- extractor code identity: every legacy module the children imported is an input of this unit (spec B4) -------
    recorded = {p: (fp.get('sha1') or '')[:12] for p, fp in u.inputs.items()}
    drift = []
    for f, h in sorted((u.meta.get('legacy_code') or {}).items()):
        if f in recorded:
            if recorded[f] and recorded[f] != h:
                drift.append(f'{f} (fingerprinted {recorded[f]}, a child imported {h})')
        else:
            u.add_input(f)
    if drift:
        u.check('curves.extractor_code_identity', False, value=len(drift),
                detail='in-place code changed during the unit: ' + '; '.join(drift[:4]))
    # ---- publish ------------------------------------------------------------------------------------------------------
    conflicts, n_error = [], 0
    fb_tot = {'frames_absent_from_admission': 0, 'scale_from_all_frames': 0, 'zero_vis_frames_kept': 0,
              'admission_table_missing': 0}
    published_i = {}
    for sb in run_sbs:
        res = results.get(sb, {})
        for st in stokes:
            key = f'{sb}_{st}'
            r = res.get(st) or {'status': 'error', 'reason': f'no outcome for Stokes {st}'}
            ent = {k: r.get(k) for k in ('status', 'gate', 'reason', 'rc', 'wall_s', 'log', 'unpublished_table')
                   if r.get(k) is not None}
            ent['log_tail'] = r.get('tail', [])
            pairs[key] = ent
            d = dst[(sb, st)]
            if r['status'] == 'refused':
                u.refuse_item(d, f"{r.get('gate')}: {r.get('reason')}", gate=r.get('gate'))
                continue
            if r['status'] != 'done':
                n_error += 1
                u.note(f"{key} error: {r.get('reason')}" + (f" | {r.get('traceback', '')[-1500:]}"
                                                             if r.get('traceback') else ''))
                continue
            facts = W.table_facts(r['path'], st, os.path.join(stage_root, sb) if stage_root else None,
                                  os.path.join(pl['search_dir'], f'admission_{sb}.csv'), CFG.paths.CLV15_ROOT, C.SAMPLES,
                                  cell=cell, sb=sb)
            ent['fallbacks'] = {k: facts[k] for k in ('frames_absent_from_admission', 'admission_table_missing',
                                                      'scale_from_all_frames', 'zero_vis_frames_kept',
                                                      'missing_sample_files')}
            for k in fb_tot:
                fb_tot[k] += int(facts[k])
            if facts['unstaged_frames']:
                why = (f"G4: {facts['unstaged_frames']} frames of the table have no staged file under "
                       f"{os.path.join(stage_root, sb)} (the extractor read the archive copy)")
                u.refuse_item(d, why, gate='G4')
                ent.update(status='refused', gate='G4', reason=why)
                continue
            ent['mixed_grid_frames'] = facts['mixed_grid_frames']    # reported (frames whose NAXIS1 != the first's)
            if st == 'V' and 'I' in stokes and sb not in published_i:
                why = (f'G10: the Stokes I table of this subband was not published ({pairs.get(sb + "_I", {}).get("status")}'
                       f'); V carries its scale and is not published alone')
                u.refuse_item(d, why, gate='G10')
                ent.update(status='refused', gate='G10', reason=why)
                continue
            # content-test context: the regeneration recipe's gates T1-T5 (exopipe_ops snapshot_lightcurves_regen_test.py)
            # on every table - MCATSHA required (T1), Master rows = the catalogue's Gaussians in the first frame (T2), admitted
            # flags = the admission table (T3); with --supersede the replaced table (T3/T4); V scale = I (T5, flux_scale)
            ctx = {'naxis': facts['naxis'], 'flowtree': CFG.paths.SNAPSHOT_FLOW_ROOT, 'frame_header': facts['frame_header'],
                   'require_mcatsha': True, 'admission': os.path.join(pl['search_dir'], f'admission_{sb}.csv'),
                   'master_catalog': pl['subbands'][sb]['catalog']}
            if args.supersede and os.path.exists(d) and not PUB.same_bytes(r['path'], d):
                ctx['previous'] = d
            if st == 'V':
                ctx['sibling_i'] = os.path.join(os.path.dirname(r['path']), P.table_name(cell, sb, 'I'))
            pre = validate.run('light_curve', r['path'], **ctx)
            failing = [t for t in pre if not t['passed'] and not run.waived(t['name'])]
            if failing:
                for t in failing:
                    u.check(t['name'] + ' (before publish)', False, value=t['value'], detail=t['detail'],
                            cls='light_curve', path=r['path'])
                why = 'content test failed before publish: ' + '; '.join(f"{t['name']} ({t['detail']})"
                                                                         for t in failing)[:600]
                u.refuse_item(d, why, gate='content-test')
                ent.update(status='refused', gate='content-test', reason=why)
                continue
            status, pdetail = PUB.publish(r['path'], d, out, cell, args.supersede,
                                          reason=f'exopipe3 curves {run.run_id}')
            ent['publish'] = status
            if status == 'conflict':
                conflicts.append(d)
                u.refuse_item(d, f'G12: {pdetail}', gate='G12')
                ent.update(status='refused', gate='G12', reason=pdetail)
                continue
            ctx_pub = dict(ctx)
            ctx_pub.pop('previous', None)
            if status == 'superseded':                   # the replaced table now lives in the supersede quarantine
                ctx_pub['previous'] = os.path.join(out, '_superseded', args.supersede, cell, os.path.basename(d))
            if st == 'V':
                ctx_pub['sibling_i'] = published_i.get(sb)
            u.add_output(d, 'light_curve', **ctx_pub)
            if st == 'I':
                published_i[sb] = d
            ent['status'] = 'published' if status != 'skipped-valid' else 'skipped-valid'
            if status == 'superseded':
                ent['superseded'] = pdetail
    u.meta['fallbacks'] = fb_tot
    # ---- unit checks ----------------------------------------------------------------------------------------------------
    written = {o['path'] for o in u.outputs}
    refused = {r['path'] for r in u.refused_items}
    n_pub = sum(1 for p in u.planned if p in written)
    n_ref = sum(1 for p in u.planned if p in refused)
    if n_error:
        u.note(f'LC7 curves.complete not evaluated: {n_error} pair(s) ended in an extractor error (the unit is an error)')
    else:
        unacc = [p for p in u.planned if p not in written and p not in refused]
        u.check('curves.complete', not unacc, value=f'{n_pub} published + {n_ref} refused of {len(u.planned)} planned',
                detail=('every planned (subband, Stokes) pair published and tested, or refused by a named gate'
                        if not unacc else f'{len(unacc)} planned pairs unaccounted: ' + ', '.join(unacc[:4])))
    bad = _excluded_products(out, cell, ex)
    u.check('curves.no_excluded_products', not bad, value=len(bad),
            detail=('no table of an excluded (LST, subband, date) under ' + os.path.join(out, cell)) if not bad else
            f'{len(bad)} tables of excluded cell-bands (retire them with `archive retire-excluded`): ' + ', '.join(
                os.path.basename(b) for b in bad[:6]))
    if conflicts:
        u.check('curves.publish_conflicts', False, value=len(conflicts),
                detail=f'G12: {len(conflicts)} differing tables already published (pass --supersede TAG): '
                       + ', '.join(os.path.basename(c) for c in conflicts[:4]))
    if C.INCOMPLETE_STOPS_CHAIN and u.refused_items:
        u.check('curves.incomplete_stops_chain', False, value=len(u.refused_items),
                detail=f'{len(u.refused_items)} pairs refused and curves.INCOMPLETE_STOPS_CHAIN is True')
    if n_error or u.failed_tests():
        u.note(f'scratch kept for diagnosis: {scratch}')
    else:
        fitsio.cleanup(scratch)


def _excluded_products(out, cell, ex):
    """LC8: tables under <out>/<cell>/ that belong to an excluded (LST, subband, date)."""
    from exopipe.phase3.lib.curves.tests import NAME_RE
    d = os.path.join(out, cell)
    bad = []
    try:
        names = sorted(os.listdir(d))
    except FileNotFoundError:
        return bad
    for n in names:
        m = NAME_RE.match(n)
        if m and ex.is_excluded(m['lst'], m['sb'], m['date']):
            bad.append(os.path.join(d, n))
    return bad
