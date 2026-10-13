"""`exopipe3 frames --tier deep`: frame-only rebuilds of the deep search frames (B11, contract steps 26-35).

Decision of the refactor chat (2026-10-13, crosscheck group 14): during the refactor window `search --tier deep` (stage
detect) PRODUCES the deep frames; this tier is for frame-only rebuilds and is never chained before search
(frames_residuals.DEEP_PRODUCER = 'search': a residuals unit refuses unless --deep-rebuild says the run is one). The
engine is run_deep_cell(frames_only=True) (lib/frames/deep_engine.py; the keyword is the search owner's commit,
lib/frames/deep_seam.py).

Item kinds:
- per hour: ``--epochs 07h_2025-10-15`` (core.cells epoch): that night against EACH band's installed master: one unit
  per (band cell, Stokes) with the night as the engine's incremental date list (every admission gate runs on the whole
  cell, only this night is built; phase3_deep.py:1371-1380). A night outside the master (weights.csv in_master 0) has no
  LOO master and subtracts the central master (phase3_deep.py:1394-1433). ``--scopes`` narrows the bands.
- bulk re-run after a master rebuild: ``--cells 07h_18MHz,07h_Red`` (or ``--lsts`` x ``--scopes``): one unit per (cell,
  Stokes) over every admitted epoch; ``--nights`` restricts it to those nights (incremental).
Stages: ``residuals`` (the engine in a child per unit) then ``verify`` (re-runs every content test on the outputs listed in
the residuals record - never a glob - or, with ``--deep-adopt``, registers and tests production attempts read-only).
"""
import hashlib
import os
import socket
import sys
import time

from exopipe.phase3.core.record import NoInput, Refusal

STAGES = ('residuals', 'verify')


def _cfg():
    from exopipe.phase3.config import CFG
    return CFG


def add_arguments(p):
    g = p.add_argument_group('tier deep (frame-only rebuilds; --cells bulk, --epochs per hour)')
    g.add_argument('--deep-rebuild', action='store_true',
                   help='state that this is a frame-only rebuild: required while frames_residuals.DEEP_PRODUCER is '
                        "'search' (the refactor window: `search --tier deep` produces the deep frames, frames is never "
                        'chained before it)')
    g.add_argument('--deep-x2-flows', choices=('compute', 'require'), default=None,
                   help="missing x2 flows: 'compute' them on this node's GPU under the node flow lock (production), or "
                        "'require' a complete cache in the work dir (refuse otherwise; CPU hosts). Default "
                        "frames_residuals.X2_FLOWS_MISSING")
    g.add_argument('--deep-adopt', metavar='ATTEMPT_DIRS',
                   help='verify stage: register production attempt directories <cell>_<S>_<UTC> (list or @file) in '
                        'records and run every content test on their frames, siblings, sigma stacks, refusal records '
                        '(read only)')


# ---------------------------------------------------------------------------------------------------------------------
# roots
# ---------------------------------------------------------------------------------------------------------------------

def _host():
    return socket.gethostname()


def attempts_root_production():
    return _cfg().frames.DEEP_ATTEMPTS_ROOT.format(host=_host())


def default_out(args):
    """Production: the node's collect tree (records under <collect>/_records/frames)."""
    return os.path.dirname(attempts_root_production())


def product_roots(args):
    """(attempts root, archive root): rebased under --out (parity / scratch), else production."""
    if args.out:
        return os.path.join(args.out, 'deep_attempts'), os.path.join(args.out, 'archive')
    return attempts_root_production(), _cfg().paths.PHASE3_ARCHIVE_ROOT


def _under_production(path):
    cfg = _cfg()
    real = os.path.realpath(path)
    for r in cfg.paths.PRODUCTION_ROOTS:
        rr = os.path.realpath(r)
        if real == rr or real.startswith(rr.rstrip('/') + '/'):
            return r
    return None


def flow_root(args):
    """The node-local flow / spill root (P3_FLOW_ROOT): <scratch root>/<WORK_SUBDIR>; refuses without a node-local root."""
    from exopipe.phase3.core import fitsio
    return os.path.join(fitsio.scratch_root(args.work), _cfg().frames_residuals.WORK_SUBDIR)


def check_production_writes(args, roots):
    """While paths.PRODUCTION_WRITES is False the step writes only under --out and node-local scratch outside every
    production root (production deep frames come from the production fleet during the window)."""
    cfg = _cfg()
    if cfg.paths.PRODUCTION_WRITES:
        return
    if not args.out:
        raise Refusal('frames writes only under --out while paths.PRODUCTION_WRITES is False (refactor window): pass '
                      '--out <your root> (products are rebased to <out>/deep_attempts and <out>/archive)')
    for r in roots:
        hit = _under_production(r)
        if hit:
            raise Refusal(f'{r} is under the production root {hit} and paths.PRODUCTION_WRITES is False')


# ---------------------------------------------------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------------------------------------------------

class Plan:
    """One unit: (cell, Stokes) over every admitted epoch (bulk) or over listed nights (per-hour / incremental)."""

    def __init__(self, cell, stokes, dates=None, mode='bulk', item=None):
        self.cell, self.stokes = cell, stokes
        self.lst, self.scope = cell.split('_', 1)
        self.dates = list(dates) if dates else None
        self.mode = mode
        self.item = item
        if not self.dates:
            self.name = f'{cell}_{stokes}'
        elif len(self.dates) == 1:
            self.name = f'{cell}_{stokes}_{self.dates[0]}'
        else:
            h = hashlib.sha1(','.join(sorted(self.dates)).encode()).hexdigest()[:8]
            self.name = f'{cell}_{stokes}_n{len(self.dates)}_{h}'

    @property
    def date(self):
        """The night of a per-hour unit, else None."""
        return self.dates[0] if self.mode == 'per-hour' else None


def _stokes(args):
    from exopipe.phase3.lib.frames.deep_engine import engine_values
    order = engine_values('deep')['stokes_order']          # V first: the V run writes the flows I reuses
    v = getattr(args, 'stokes', None)
    st = [s.strip() for s in v.split(',')] if v else list(order)
    bad = [s for s in st if s not in ('I', 'V')]
    if bad:
        raise ValueError(f'--stokes: {bad} (I and/or V)')
    return sorted(dict.fromkeys(st), key=lambda s: order.index(s) if s in order else 9)


def _scopes(args):
    from exopipe.phase3.core import cells
    allsc = cells.scopes()
    if not getattr(args, 'scopes', None):
        return list(allsc)
    out = cells.expand(args.scopes, universe=allsc, kind='scope')
    bad = [s for s in out if s not in allsc]
    if bad:
        raise ValueError(f'--scopes: unknown {bad}')
    return out


def build_plans(args):
    """The units of this invocation, in order (cell order, Stokes in search.STOKES_ORDER)."""
    from exopipe.phase3.core import cells
    from exopipe.phase3.lib.frames import deep_cells
    st = _stokes(args)
    excl_cells = set(cells.expand_cells(args.exclude_cells)) if getattr(args, 'exclude_cells', None) else set()
    plans = []
    if getattr(args, 'epochs', None):
        if getattr(args, 'cells', None) or getattr(args, 'lsts', None) or getattr(args, 'nights', None):
            raise ValueError('give --epochs (per hour: <LST>_<date>, every band) OR --cells/--lsts [--nights] (bulk), '
                             'not both')
        for lst, date in deep_cells.expand_epochs(args.epochs):
            for scope in _scopes(args):
                cell = f'{lst}_{scope}'
                if cell in excl_cells:
                    continue
                for s in st:
                    plans.append(Plan(cell, s, [date], 'per-hour', item=f'{lst}_{date}'))
        return plans
    if getattr(args, 'cells', None):
        cl = cells.expand_cells(args.cells)
    elif getattr(args, 'lsts', None):
        cl = [f'{cells.parse_lst(l)}_{s}' for l in cells.expand(args.lsts, universe=cells.LSTS, kind='LST')
              for s in _scopes(args)]
    else:
        raise ValueError('tier deep needs --cells (bulk re-run), --lsts [--scopes], or --epochs <LST>_<date> (per hour)')
    nights = [cells.parse_date(n) for n in cells.expand(args.nights, kind='night')] if getattr(args, 'nights', None) \
        else None
    for c in cl:
        if c in excl_cells:
            continue
        for s in st:
            plans.append(Plan(c, s, nights, 'incremental' if nights else 'bulk'))
    return plans


# ---------------------------------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------------------------------

class Ctx:
    """What every unit of one invocation shares."""

    def __init__(self, args, run, stages):
        from exopipe.phase3.lib.frames import deep_engine
        cfg = _cfg()
        fr = cfg.frames_residuals
        self.args, self.run, self.stages, self.cfg = args, run, stages, cfg
        self.v = deep_engine.engine_values('deep', cfg)
        self.host = _host()
        self.label = getattr(args, 'run_label', None) or self.v['run_label']
        self.attempts_root, self.archive_root = product_roots(args)
        self.archive_copy = self.v['archive_write'] if getattr(args, 'archive_copy', None) is None else args.archive_copy
        self.archive_run = getattr(args, 'archive_run', None) or self.v['archive_run']
        self.x2_mode = getattr(args, 'deep_x2_flows', None) or fr.X2_FLOWS_MISSING
        self.producer = fr.DEEP_PRODUCER
        try:
            self.flow_root, self.flow_root_error = flow_root(args), None
        except Refusal as e:
            self.flow_root, self.flow_root_error = None, str(e)
        self.engine_file = deep_engine.engine_source()
        self.engine_frames_only = deep_engine.engine_has_frames_only(self.engine_file)
        self.workers = max(1, int(getattr(args, 'workers', None) or fr.UNIT_WORKERS))
        self.notes = deep_engine.check_overrides(cfg)

    def producer_refusal(self):
        """deep-producer gate: while search produces the deep frames, only an explicit frame-only rebuild runs."""
        if self.producer == 'search' and not getattr(self.args, 'deep_rebuild', False):
            return ('deep-producer: during the refactor window `search --tier deep` (detect) produces the deep search '
                    'frames (refactor decision 2026-10-13, crosscheck group 14); frames --tier deep runs only as a '
                    'frame-only rebuild, never chained before search - pass --deep-rebuild to state that this run is one')
        return None

    def seam_refusal(self):
        if not self.engine_frames_only:
            return (f'engine-seam: run_deep_cell in {self.engine_file} has no frames_only keyword yet - the search '
                    f"owner's one reviewed commit, byte-neutral for the default call (proposed edit and AST proof: "
                    f'python -m exopipe.phase3.lib.frames.deep_seam --check)')
        return None

    def work_cell(self, cell):
        return os.path.join(self.flow_root, cell) if self.flow_root else None


def run(args, run, stages):
    """Tier deep of `exopipe3 frames` (called by steps/frames.py with the selected stages)."""
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f'tier deep stages are {", ".join(STAGES)}; got {unknown}')
    ctx = Ctx(args, run, stages)
    adopt = _adopt_dirs(args) if getattr(args, 'deep_adopt', None) else []
    plans = [] if (adopt and stages == ['verify'] and not getattr(args, 'cells', None)
                   and not getattr(args, 'epochs', None) and not getattr(args, 'lsts', None)) else build_plans(args)
    if not plans and not adopt:
        raise NoInput('no units selected')
    if args.dry_run:
        return print_plan(ctx, plans, adopt)
    # records are writes too: every real run needs a non-production --out while paths.PRODUCTION_WRITES is False (the
    # default out, the node's collect tree, is a production tree that cli._guard_out does not cover; spec B8)
    roots = ([ctx.attempts_root, ctx.archive_root] + ([ctx.flow_root] if ctx.flow_root else [])
             if 'residuals' in stages else []) + ([args.out] if args.out else [])
    try:
        check_production_writes(args, roots)
    except Refusal:
        run.no_records = True        # nothing may be written - not even the run summary under the production out
        raise
    for n in ctx.notes:
        print(f'frames deep: {n}', file=sys.stderr)
    if 'residuals' in stages:
        _parallel(ctx, plans, _residuals_unit)
    if 'verify' in stages:
        for p in plans:
            _guarded(_verify_unit, ctx, p)
        for d in adopt:
            _guarded(_adopt_unit, ctx, d)
    return None


def _guarded(fn, *a):
    try:
        fn(*a)
    except (Refusal, NoInput):
        pass                      # recorded by the unit (UnitRecord swallows them; defensive)
    except Exception as e:        # recorded by the unit as 'error'; the run's exit code says so
        print(f'frames deep: unit error {type(e).__name__}: {e}', file=sys.stderr)


def _parallel(ctx, plans, fn):
    if ctx.workers <= 1 or len(plans) <= 1:
        for p in plans:
            _guarded(fn, ctx, p)
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=ctx.workers) as ex:
        list(ex.map(lambda p: _guarded(fn, ctx, p), plans))


def _new_attempt(ctx, plan):
    """A fresh attempt directory <attempts>/<cell>_<S>_<UTC> (p3s5_search_fleet.sh:187-188; one per unit run)."""
    base = os.path.join(ctx.attempts_root, f'{plan.cell}_{plan.stokes}_{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}')
    p, n = base, 1
    while os.path.exists(p):
        n += 1
        p = f'{base}_{n}'
    os.makedirs(p)
    return p


def _check_pregates(u, gates):
    """Record every passing pre-gate as a check; raise at the first failing one (Refusal, or NoInput for 'noinput')."""
    first = None
    for g in gates:
        if g['ok']:
            u.check(f"pregate.{g['gate']}", True, detail=g['detail'])
        elif first is None:
            first = g
    if first is not None:
        if first['kind'] == 'noinput':
            raise NoInput(f"{first['gate']}: {first['detail']}")
        raise Refusal(f"{first['gate']}: {first['detail']}")


def _meta(ctx, plan):
    return dict(tier='deep', stage='residuals', cell=plan.cell, stokes=plan.stokes, mode=plan.mode,
                included_dates=plan.dates, item=plan.item, label=ctx.label, producer=ctx.producer,
                rebuild=bool(getattr(ctx.args, 'deep_rebuild', False)))


def _residuals_unit(ctx, plan):
    from exopipe.phase3.core import exclusions, fitsio
    from exopipe.phase3.lib.frames import deep_cells, deep_engine
    if ctx.args.resume and not ctx.args.fresh:
        # README rule 5: skip only when the record is done and inputs, admitted nights, code and config are unchanged and
        # every recorded output passes its content tests now (record.resume_check)
        try:
            c = deep_cells.Cell(plan.cell, deep_cells.coadd_root_of(plan.cell, getattr(ctx.args, 'coadd_root', None)))
            ok, why = ctx.run.resume_check(plan.name, c.input_paths(plan.stokes),
                                           exclusions.load().admitted(c.lst, c.scope, c.epoch_dates()))
        except Exception as e:
            ok, why = False, f'resume check failed: {e}'
        if ok:
            u = ctx.run.unit(plan.name, **_meta(ctx, plan))
            u.status, u.reason = 'skipped-valid', why
            print(f'frames deep {plan.name}: skipped (--resume): {why}', flush=True)
            return
        print(f'frames deep {plan.name}: not resumed ({why})', flush=True)
    u = ctx.run.unit(plan.name, **_meta(ctx, plan))
    with u:
        for n in ctx.notes:
            u.note(n)
        pr = ctx.producer_refusal()
        if pr:
            raise Refusal(pr)
        root = deep_cells.coadd_root_of(plan.cell, getattr(ctx.args, 'coadd_root', None))
        if not root:
            raise (NoInput if plan.mode == 'per-hour' else Refusal)(
                f"no co-add root for {plan.cell} in {ctx.v['roots_csv']} (and no --coadd-root)")
        if getattr(ctx.args, 'coadd_root', None):
            u.note(f'--coadd-root {ctx.args.coadd_root} (non-default input root)')
        c = deep_cells.Cell(plan.cell, root)
        excl = exclusions.load()
        if ctx.flow_root_error:
            raise Refusal(ctx.flow_root_error)
        gates = deep_cells.pregates(c, plan.stokes, 'deep', date=plan.date, work_cell=ctx.work_cell(plan.cell),
                                    x2_mode=ctx.x2_mode, host=ctx.host, excl=excl)
        _check_pregates(u, gates)
        sr = ctx.seam_refusal()
        if sr:
            raise Refusal(sr)
        dates = c.epoch_dates()
        admitted = excl.admitted(c.lst, c.scope, dates)
        for p in c.input_paths(plan.stokes):
            u.add_input(p)
        u.set_admitted(admitted)
        u.note(f'exclusions {excl.fingerprint()}')
        owed = [d for d in admitted if not plan.dates or d in plan.dates]
        attempt = _new_attempt(ctx, plan)
        spec = deep_engine.UnitSpec(tier='deep', cadence='deep', lst=c.lst, scope=c.scope, stokes=plan.stokes,
                                    label=ctx.label, included_dates=plan.dates, attempt=attempt, coadd_root=root,
                                    archive_root=ctx.archive_root, archive_run=ctx.archive_run,
                                    archive_copy=ctx.archive_copy, work_root=ctx.flow_root)
        for d in owed:
            u.plan(spec.frame_path(d, plan.stokes), 'search_frame')
            u.plan(spec.frame_path(d, plan.stokes, True), 'search_frame')
        u.meta.update(attempt=attempt, coadd_root=root, archive_root=ctx.archive_root, archive_run=ctx.archive_run,
                      archive_copy=ctx.archive_copy, flow_root=ctx.flow_root, engine=ctx.engine_file,
                      child_env=deep_engine.describe_env(deep_engine.child_env(spec, ctx.cfg)), owed=owed)
        print(f'frames deep {plan.name}: run_deep_cell(frames_only=True) on {len(owed)} owed epoch(s) -> {attempt}',
              flush=True)
        scratch = fitsio.scratch_dir(f'frames_{plan.name}', ctx.args.work)
        try:
            res = deep_engine.run_child(spec, scratch, unit=u, cfg=ctx.cfg)
        finally:
            fitsio.cleanup(scratch)
        u.meta['engine_result'] = {k: res.get(k) for k in ('status', 'rc', 'wall_s', 'engine_file')}
        register(u, ctx, plan, c, spec, res, owed, excl)


def frame_ctx(c, stokes, d):
    """Content-test context of a search frame of cell c (in-master flag and the own-Stokes master's NCOADD)."""
    inm = c.in_master()
    mp = c.master(stokes)
    nco = int(c.header(mp).get('NCOADD', 0) or 0) if os.path.exists(mp) else None
    return dict(stokes=stokes, in_master=inm.get(d, False), ncoadd=nco)


def _sibling_ctx(ctx, c, stokes, plain):
    pb = os.path.join(c.final, f'PB_deep_R075_{c.scope}.fits' if stokes == 'I' else f'PB_deep_{c.scope}.fits')
    return dict(stokes=stokes, pb_path=pb if os.path.exists(pb) else None, plain_path=plain, floor=ctx.v['pb_floor'],
                cell=c.cell)


def register(u, ctx, plan, c, spec, res, owed, excl):
    """Register what the engine wrote: every planned frame and sibling written (content-tested now) or refused by a
    named gate; archive copies, refusal records, the sigma stack; per-unit checks."""
    from exopipe.phase3.lib.frames import deep_engine, tests as T
    S = plan.stokes
    found = deep_engine.collect(spec)
    whole = {}
    for kind in deep_engine.WHOLE_EPOCH_KINDS:
        p = found['refusals'].get(kind)
        if p:
            for d, why in deep_engine.refusal_dates(p).items():
                whole.setdefault(d, (kind, why))
    pbmiss = deep_engine.pbcorr_missing_dates(found['pbcorr_missing'])
    work_cell = ctx.work_cell(plan.cell)
    written, removed, unaccounted = [], [], []
    for d in owed:
        fp, sp = spec.frame_path(d, S), spec.frame_path(d, S, True)
        if (d, S) in found['frames']:
            u.add_output(fp, 'search_frame', **frame_ctx(c, S, d))
            written.append(fp)
            if (d, S) in found['siblings']:
                u.add_output(sp, 'search_frame', **_sibling_ctx(ctx, c, S, fp))
                written.append(sp)
            elif d in pbmiss:
                u.refuse_item(sp, f'PB map {pbmiss[d]} missing: frame written, sibling not (pbcorr_missing record; '
                                  f'carried as today, B-30)', gate='pb-missing')
            continue
        if d in whole:
            kind, why = whole[d]
            reason = f'{kind} gate removed the epoch, no frame owed' + (f' ({why})' if why else '')
            u.refuse_item(fp, reason, gate=kind)
            u.refuse_item(sp, reason, gate=kind)
            removed.append(d)
            continue
        flowless = ctx.v['refine_x2'] and work_cell and not os.path.exists(os.path.join(work_cell, f'ep_{d}_up2_flow.npy'))
        unaccounted.append(d)
        why = ('the x2 flow batch left this epoch without a flow (align_images rc 2; inventory R1)' if flowless else
               'the residual task dropped the epoch without a record (inventory R2)')
        if ctx.v['unaccounted'] == 'record':
            reason = f'{why}: production drops it silently; carried and recorded (search.UNACCOUNTED_EPOCHS=record, B-30)'
            u.refuse_item(fp, reason, gate='unaccounted-epoch')
            u.refuse_item(sp, reason, gate='unaccounted-epoch')
    if unaccounted and ctx.v['unaccounted'] != 'record':
        raise Refusal(f'unaccounted-epoch: {unaccounted} have neither a frame nor a refusal row '
                      f'(search.UNACCOUNTED_EPOCHS={ctx.v["unaccounted"]!r})')
    extra = sorted(k[0] for k in found['frames'] if k[1] == S and k[0] not in owed)
    u.check('frames_only_for_owed_epochs', not extra, value=len(extra),
            detail=('frames for epochs not owed: ' + ', '.join(extra[:5])) if extra else 'every frame is an owed epoch')
    bad = sorted(k[0] for k in found['frames'] if excl.is_excluded(c.lst, c.scope, k[0][:10]))
    u.check('exclusion_conformance', not bad, value=len(bad),
            detail=('frames of excluded nights: ' + ', '.join(bad[:5])) if bad else 'no frame of an excluded night')
    nf = sum(1 for d in owed if (d, S) in found['frames'])
    u.check('frame_count', nf == len(owed) - len(removed) - len(unaccounted), value=nf,
            detail=f'{nf} frames = {len(owed)} owed - {len(removed)} removed by whole-epoch gates - '
                   f'{len(unaccounted)} unaccounted (recorded)')
    rep = deep_engine.reported_set(res, S)
    if rep is not None:
        scanned = {os.path.abspath(p) for p in written}
        u.check('engine_reported_set', rep == scanned, value=len(rep),
                detail='engine-reported frames equal the attempt scan' if rep == scanned else
                f'reported {len(rep)} != scanned {len(scanned)}')
    if spec.archive_copy:
        miss = []
        for p in written:
            dst = deep_engine.archive_path(p, spec)
            if dst and os.path.exists(dst):
                kw = (_sibling_ctx(ctx, c, S, os.path.join(os.path.dirname(p), os.path.basename(p).replace(
                    '_redo_resid_pbcorr_', '_redo_resid_', 1))) if '_redo_resid_pbcorr_' in p
                      else frame_ctx(c, S, os.path.basename(p)[:10]))
                u.add_output(dst, 'search_frame', copy_of=p, **kw)
            else:
                miss.append(os.path.basename(p))
        u.check('archive_copies', not miss, value=len(written) - len(miss),
                detail=('no archive copy of ' + ', '.join(miss[:4])) if miss else f'{len(written)} archive copies')
    frame_dirs = [spec.resid_dir('I'), spec.resid_dir('V')]
    for kind, p in sorted(found['refusals'].items()):
        u.add_output(p, 'refusal_record', tier='deep', stokes=S, cell=plan.cell)
        if kind in deep_engine.WHOLE_EPOCH_KINDS:
            ok, det = T.removed_without_frame(p, frame_dirs)
            u.check(f'removed_epochs_have_no_frame.{kind}', ok, detail=det, path=p)
    if found['pbcorr_missing']:
        u.add_output(found['pbcorr_missing'], 'frame_refusals')
    sp_ = found['sigstack'].get(S)
    if plan.mode == 'bulk':
        if sp_:
            u.add_output(sp_, 'noise_map')
        else:
            u.note(f"noise_map: none ({nf} residuals; attempted from {ctx.v['sigstack_min_epochs']} and built from "
                   f"{ctx.v['sigstack_build_min']} residuals with a finite inner rms; D-G21, not a refusal)")
    elif sp_:
        u.note(f'{os.path.basename(sp_)} in the attempt was computed by the engine from the listed nights only: not a '
               f'cell noise map, not registered')
    if found['search_refusals'] or found['extras']:
        u.note('unexpected search products in a frames_only attempt (not registered): '
               + ', '.join(os.path.basename(p) for p in list(found['search_refusals'].values()) + found['extras']))
    if unaccounted:
        u.note(f'unaccounted epochs recorded as item refusals (carried defects R1/R2, B-30): {unaccounted}')


# ---------------------------------------------------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------------------------------------------------

def _adopt_dirs(args):
    from exopipe.phase3.core import cells
    out = []
    for d in cells.expand(args.deep_adopt, kind='attempt'):
        d = os.path.abspath(d.rstrip('/'))
        if not os.path.isdir(d):
            raise ValueError(f'--deep-adopt: {d} is not a directory')
        out.append(d)
    return out


def _cell_or_none(cell, args):
    from exopipe.phase3.lib.frames import deep_cells
    root = deep_cells.coadd_root_of(cell, getattr(args, 'coadd_root', None))
    if root and os.path.isdir(os.path.join(root, cell)):
        return deep_cells.Cell(cell, root)
    return None


def _names_by_prodname(paths):
    from astropy.io import fits
    out = {}
    for p in paths:
        try:
            n = str(fits.getheader(p).get('PRODNAME', '') or '')
        except OSError:
            continue
        if n:
            out[n if n.endswith('.fits') else n + '.fits'] = p
    return out


def _frame_kw(ctx, c, stokes, path, by_prodname):
    """Content-test context of a recorded search_frame output (plain or sibling, attempt file or archive copy)."""
    from exopipe.phase3.lib.frames import tests as T
    b = os.path.basename(path)
    attempt_file = b[:4].isdigit()
    kw = {}
    if not attempt_file and b in by_prodname:
        kw['copy_of'] = by_prodname[b]
    src = kw.get('copy_of', path)
    if T.is_sibling(path):
        plain = os.path.join(os.path.dirname(src), os.path.basename(src).replace('_redo_resid_pbcorr_', '_redo_resid_', 1))
        kw.update(_sibling_ctx(ctx, c, stokes, plain) if c else dict(stokes=stokes, plain_path=plain,
                                                                      floor=ctx.v['pb_floor']))
    elif c:
        kw.update(frame_ctx(c, stokes, os.path.basename(src)[:10]))
    else:
        kw['stokes'] = stokes
    return kw


def _verify_unit(ctx, plan):
    from exopipe.phase3.core import record
    from exopipe.phase3.lib.frames import deep_cells
    u = ctx.run.unit(f'{plan.name}_verify', tier='deep', stage='verify', cell=plan.cell, stokes=plan.stokes,
                     mode=plan.mode, of_unit=plan.name)
    with u:
        rec = record.load_record(ctx.run.out_root, ctx.run.step, plan.name)
        if rec is None:
            raise NoInput(f'no residuals record {plan.name} under {ctx.run.out_root}/_records/{ctx.run.step}: nothing to '
                          f'verify (verify reads the outputs the residuals record lists, never a glob)')
        if rec.get('status') != 'done':
            raise NoInput(f"residuals unit {plan.name} is {rec.get('status')} ({str(rec.get('reason', ''))[:120]}): "
                          f"nothing to verify")
        outs = rec.get('outputs') or []
        attempt_frames = [o['path'] for o in outs if os.path.basename(o['path'])[:4].isdigit()
                          and o['class'] == 'search_frame']
        by = _names_by_prodname(attempt_frames)
        root = (rec.get('meta') or {}).get('coadd_root')          # the co-add root the residuals unit used
        c = (deep_cells.Cell(plan.cell, root) if root and os.path.isdir(os.path.join(root, plan.cell))
             else _cell_or_none(plan.cell, ctx.args))
        for o in outs:
            p, cls = o['path'], o['class']
            if not os.path.exists(p):
                u.check('output_present', False, detail=f'{p} missing', cls=cls, path=p)
                continue
            if cls == 'search_frame':
                u.add_output(p, cls, **_frame_kw(ctx, c, plan.stokes, p, by))
            elif cls == 'refusal_record':
                u.add_output(p, cls, tier='deep', stokes=plan.stokes, cell=plan.cell)
            else:
                u.add_output(p, cls)
        u.meta['attempt'] = (rec.get('meta') or {}).get('attempt')


def _adopt_unit(ctx, d):
    """--deep-adopt: a production attempt <cell>_<S>_<UTC> registered in a record and content-tested (read only)."""
    import re
    from exopipe.phase3.core import exclusions
    from exopipe.phase3.lib.frames import deep_engine
    b = os.path.basename(d)
    m = re.match(r'^(?P<cell>\d\dh_(?:\d+MHz|Red|Green|Blue|Full))_(?P<s>[IV])_(?P<tag>\d{8}T\d{6}(?:_\d+)?)$', b)
    name = f'{b}_adopt_verify' if not m else f"{m['cell']}_{m['s']}_adopt_{m['tag']}_verify"
    u = ctx.run.unit(name, tier='deep', stage='verify', adopted=d)
    with u:
        if not m:
            raise Refusal(f'{d}: not an attempt directory name <cell>_<S>_<YYYYMMDDTHHMMSS>')
        labels = [x for x in os.listdir(d) if x.startswith('Run_') and os.path.isdir(os.path.join(d, x))]
        if len(labels) != 1:
            raise Refusal(f'{d}: expected one Run_<label> directory, found {labels}')
        cell, S = m['cell'], m['s']
        lst, scope = cell.split('_', 1)
        spec = deep_engine.UnitSpec(tier='deep', cadence='deep', lst=lst, scope=scope, stokes=S, label=labels[0][4:],
                                    attempt=d)
        found = deep_engine.collect(spec)
        c = _cell_or_none(cell, ctx.args)
        frames = sorted((k[0], p) for k, p in found['frames'].items() if k[1] == S)
        if not frames:
            raise NoInput(f'{d}: no Stokes {S} frames in the attempt')
        for dd, p in frames:
            u.add_output(p, 'search_frame', **(frame_ctx(c, S, dd) if c else dict(stokes=S)))
            sp = found['siblings'].get((dd, S))
            if sp:
                u.add_output(sp, 'search_frame', **(_sibling_ctx(ctx, c, S, p) if c else
                                                    dict(stokes=S, plain_path=p, floor=ctx.v['pb_floor'])))
        frame_dirs = [spec.resid_dir('I'), spec.resid_dir('V')]
        from exopipe.phase3.lib.frames import tests as T
        for kind, p in sorted(list(found['refusals'].items()) + list(found['search_refusals'].items())):
            u.add_output(p, 'refusal_record', tier='deep', stokes=S, cell=cell)
            if kind in deep_engine.WHOLE_EPOCH_KINDS:
                ok, det = T.removed_without_frame(p, frame_dirs)
                u.check(f'removed_epochs_have_no_frame.{kind}', ok, detail=det, path=p)
        if found['sigstack'].get(S):
            u.add_output(found['sigstack'][S], 'noise_map')
        if found['pbcorr_missing']:
            u.add_output(found['pbcorr_missing'], 'frame_refusals')
        if c is not None:
            excl = exclusions.load()
            have = {dd for dd, _ in frames}
            whole = set()
            for kind in deep_engine.WHOLE_EPOCH_KINDS:
                p = found['refusals'].get(kind)
                if p:
                    whole |= set(deep_engine.refusal_dates(p))
            adm = excl.admitted(lst, scope, c.epoch_dates())
            missing = [x for x in adm if x not in have and x not in whole]
            bad = sorted(x for x in have if excl.is_excluded(lst, scope, x[:10]))
            u.set_admitted(adm)
            u.check('frame_count', not missing, value=len(have),
                    detail=(f'admitted epochs of the cell now with neither a frame nor a whole-epoch refusal row: '
                            f'{missing[:6]}' if missing else f'{len(have)} frames + {len(whole)} whole-epoch refusals '
                                                             f'cover the {len(adm)} admitted epochs of the cell now'))
            u.check('exclusion_conformance', not bad, value=len(bad),
                    detail=('frames of nights excluded now: ' + ', '.join(bad)) if bad else 'no frame of an excluded night')
        else:
            u.note(f'co-add cell {cell} not reachable from this node: frame count and exclusion checks not run')


# ---------------------------------------------------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------------------------------------------------

def print_plan(ctx, plans, adopt):
    """--dry-run: inputs found, pre-gates, the record-based epoch preview, planned outputs, the engine command. Writes
    nothing. Exit 0 when every unit would run, 2 when a gate would refuse one, 3 when nothing is to do."""
    from exopipe.phase3.core import exclusions, legacy
    from exopipe.phase3.lib.frames import deep_cells, deep_engine
    pr = print
    pr(f'frames --tier deep (dry run): {len(plans)} unit(s); stages {",".join(ctx.stages)}; run label {ctx.label}')
    prod = ctx.producer_refusal() if 'residuals' in ctx.stages and plans else None
    seam = ctx.seam_refusal() if 'residuals' in ctx.stages and plans else None
    pr(f'  deep frame producer {ctx.producer!r}' + (f' -> every residuals unit REFUSES ({prod})' if prod else
                                                   (' (frame-only rebuild stated: --deep-rebuild)'
                                                    if ctx.producer == 'search' else '')))
    pr(f'  engine {ctx.engine_file}: frames_only keyword {"present" if ctx.engine_frames_only else "ABSENT"}'
       + (f' -> every residuals unit REFUSES ({seam})' if seam else ''))
    pr(f'  attempts {ctx.attempts_root}/<cell>_<S>_<UTC>; archive copies '
       + (f'{ctx.archive_root}/<LST>/<night>/{ctx.archive_run}/<scope>/<S>/deep/<PRODNAME>' if ctx.archive_copy else 'OFF'))
    pr(f'  flow / spill root: {ctx.flow_root or "NONE - " + str(ctx.flow_root_error)}; x2 flows {ctx.x2_mode}; host '
       f'{ctx.host} ({"GPU" if ctx.host in (ctx.cfg.site.GPU_HOSTS or ()) else "no GPU"} per the site file); workers '
       f'{ctx.workers}')
    for n in ctx.notes:
        pr(f'  NOTE: {n}')
    if not ctx.args.out and not ctx.cfg.paths.PRODUCTION_WRITES and 'residuals' in ctx.stages:
        pr('  NOTE: no --out: a real run REFUSES while paths.PRODUCTION_WRITES is False (production deep frames come from '
           'the production fleet during the refactor window)')
    try:
        excl = exclusions.load()
    except Refusal as e:
        pr(f'  REFUSED: {e}')
        return 2
    n_ref = n_noin = 0
    shown_env = False
    for plan in plans:
        root = deep_cells.coadd_root_of(plan.cell, getattr(ctx.args, 'coadd_root', None))
        head = f'unit {plan.name} [{plan.mode}{" " + ",".join(plan.dates) if plan.dates else ""}]'
        if not root:
            pr(f'{head}: no co-add root for {plan.cell} -> {"NO-INPUT" if plan.mode == "per-hour" else "REFUSED"}')
            n_noin += plan.mode == 'per-hour'
            n_ref += plan.mode != 'per-hour'
            continue
        c = deep_cells.Cell(plan.cell, root)
        pr(f'{head}: co-add {c.base}')
        gates = deep_cells.pregates(c, plan.stokes, 'deep', date=plan.date, work_cell=ctx.work_cell(plan.cell),
                                    x2_mode=ctx.x2_mode, host=ctx.host, excl=excl)
        first = next((g for g in gates if not g['ok']), None)
        for g in gates:
            pr(f"    {'PASS' if g['ok'] else ('NO-INPUT' if g['kind'] == 'noinput' else 'REFUSE')} {g['gate']:13s} "
               f"{g['detail']}")
        if ctx.flow_root_error:
            pr(f'    REFUSE work-root     {ctx.flow_root_error}')
            first = first or {'kind': 'refuse'}
        if first is not None:
            if first['kind'] == 'noinput':
                n_noin += 1
            else:
                n_ref += 1
            continue
        pv = deep_cells.preview(c, plan.stokes, excl, included=plan.dates)
        pr(f"    epochs on disk {len(pv['dates'])}; excluded {pv['excluded']}; night-image FAIL {pv['nightimage']}; "
           f"epoch-gate FAIL {pv['epochgate']}; 10x sanity {pv['sanity']}")
        loo = pv['loo']
        nloo = sum(1 for x in loo.values() if x == 'loo')
        ncen = sum(1 for x in loo.values() if x.startswith('central'))
        bad = {d: x for d, x in loo.items() if x != 'loo' and not x.startswith('central')}
        pr(f"    owed frames (predicted; the provenance and beam gates are the engine's) {len(pv['owed'])}: "
           f"{nloo} against their LOO {plan.stokes} master, {ncen} against the central master (NCOADD {pv['ncoadd']})"
           + (f"; nights outside the master {[d for d in pv['owed'] if not pv['in_master'].get(d, False)]}"
              if ncen else ''))
        for d, x in bad.items():
            pr(f'    {d}: {x}')
        if plan.dates:
            for d in plan.dates:
                if d not in pv['owed']:
                    why = ('excluded' if d in pv['excluded'] else 'night-image FAIL' if d in pv['nightimage'] else
                           'epoch-gate FAIL' if d in pv['epochgate'] else '10x sanity' if d in pv['sanity'] else
                           'not an epoch of this cell')
                    pr(f'    listed night {d}: no frame ({why}; recorded as an item refusal)')
        if plan.dates and not pv['owed']:
            pr('    REFUSE all-listed    every listed night is removed by a whole-epoch gate: every planned item refused')
            n_ref += 1
        elif prod or seam or bad:
            pr(f"    REFUSE {'deep-producer' if prod else 'engine-seam' if seam else 'loo-master':13s} (see above)")
            n_ref += 1
        n = len(pv['owed'])
        pr(f"    planned: {2 * n} search_frame ({n} frames + {n} pbcorr siblings) in <attempt>/Run_{ctx.label}/"
           f"{'iresid' if plan.stokes == 'I' else 'vresid'}/{plan.cell}/"
           + (f'; {2 * n} archive copies' if ctx.archive_copy else '')
           + ('; noise_map (sigstack, >= 5 residuals)' if plan.mode == 'bulk' else '')
           + '; refusal_record CSVs of the gates that fire')
        if not shown_env and ctx.flow_root:
            spec = deep_engine.UnitSpec(tier='deep', cadence='deep', lst=c.lst, scope=c.scope, stokes=plan.stokes,
                                        label=ctx.label, included_dates=plan.dates, attempt='<attempt>', coadd_root=root,
                                        archive_root=ctx.archive_root, archive_run=ctx.archive_run,
                                        archive_copy=ctx.archive_copy, work_root=ctx.flow_root)
            try:
                s = deep_engine.child_env(spec, ctx.cfg)
                pr(f"    engine child (core.legacy.run): {legacy.interpreter(ctx.v['python'])} -u -m "
                   f"{deep_engine.CHILD_MODULE} <spec.json>  (cwd {deep_engine.worktree_root()}; log <attempt>/"
                   f"{c.cell}.log)")
                pr(f'    child env: {deep_engine.describe_env(s)}')
                shown_env = True
            except Refusal as e:
                pr(f'    REFUSE child-env     {e}')
                n_ref += 1
    if 'verify' in ctx.stages:
        pr(f'verify: re-tests the outputs listed in each residuals record under {ctx.run.out_root}/_records/frames/'
           + (f'; adopts {len(adopt)} attempt(s): ' + ', '.join(os.path.basename(a) for a in adopt[:6]) if adopt else ''))
    sys.stdout.flush()
    if n_ref:
        pr(f'dry run: {n_ref} unit(s) would be REFUSED, {n_noin} would have nothing to do', flush=True)
        return 2
    if plans and n_noin == len(plans):
        return 3
    return 0
