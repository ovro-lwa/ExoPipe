"""exopipe3 hybrid - the hybrid NoTaper route (spec plan/specs/hybrid.md): NoTaper stacks, the large-scale reconstruction, the
wideband reconstructions, the hybrid masters (Taper master + G (x) (reconstruction - Taper)), the Full hybrid and the per-tile
zero-spacing fill. Production: exopipe/phase3/notaper_largescale.py (NL) and mosaic_tpfill.py (MF) at exopipe main 4c7098c.

    exopipe3 hybrid --lsts 18h --scopes 46MHz --weightings R075 --stages stack --out W --work F
    exopipe3 hybrid --lsts '*' --scopes Blue --stages hybrid,full --out W

Use: survey (intermittent, DESIGN 12.2) - run on demand when masters are rebuilt; not part of the per-hour chain. Unit of work:
(stage, LST, scope, weighting) - unit name <stage>.<LST>_<scope>_<weighting>, e.g. stack.18h_46MHz_R075.

Stages, in order (--stages a,b or --from / --until; default all), and the scopes each one takes:
  stack      subbands               NoTaper dirty / PSF stacks and Taper residual stacks (PR-2)
  recon      subbands               the large-scale reconstruction (PR-3; needs the double-size PSF registry, Q1)
  wideband   Red, Green, Blue       the wideband reconstructions from the members' reconstructions (PR-3)
  hybrid     subbands + widebands   the hybrid master, beam-corrected sibling, CLEAN residual, PB maps (PR-4)
  full       Full                   the Full hybrid from the 15 subband hybrids (PR-4)
  tilefill   every scope            the per-tile zero-spacing fill of the hybrid masters (PR-5; needs the unfilled inputs, Q2)

PULL REQUEST PR-1: the selection, the unit list and the gates that need no stage code. Every unit of a stage not yet ported REFUSES
(exit 2) with the pull request that ports it; nothing is written but the unit records.
PULL REQUEST PR-2: the stack stage (lib/hybrid/stack.py, warp.py). A unit stack.<LST>_<band>_<w> writes, in <--out>/<LST>_<band>/,
the NoTaper stacks of weighting w (<w>_dirty_stack, _dirty_halfA, _dirty_halfB, _psf_stack, <w>_stack.json, inputs/ copies) and
the Taper residual stack of its Taper weighting (T075 for R075, T0 for R0: <t>_residual_stack, <t>_stack.json), as production's
`stack` action did for the four weightings. A variant a gate refuses (HG1 membership / exclusion, HG4 night drops, HG5 rebuild) is
a refused item of the unit - its files, with the gate - and the other variant goes on; both refused = the unit refused. Node-local
scratch: --work or the site SCRATCH_ROOT (the derived warp screens are cached there, hybrid_stack.SCREEN_CACHE_SUBDIR).

Gates here (each refuses; spec 3.4): HG0 the node-local co-add build directories (hybrid_stack.COADD_SB) are present on this node
(stack); HG6 the double-size PSF registry is pinned (recon, wideband); the striation selection's md5 equals the pinned one (hybrid); the
unfilled inputs are pinned (tilefill); HG20 every refused unit is listed and makes the exit non-zero (core/record.Run.exit_code);
a scope or weighting outside the route refuses before any unit runs; --out never is, lies in or contains a production work run or
an archive run, and is never the work root itself.
"""
import os

SUMMARY = 'hybrid NoTaper route: stacks, large-scale reconstruction, wideband, hybrid masters, Full, per-tile zero-spacing fill'
TIERS = None
SELECTORS = ('lsts', 'scopes')
CONFIG = ('hybrid', 'hybrid_stack', 'hybrid_psf2x', 'hybrid_recon', 'hybrid_unit', 'hybrid_full', 'hybrid_tilefill', 'paths',
          'exclusions', 'instrument', 'runner')
WRITES = True
STAGES = ('stack', 'recon', 'wideband', 'hybrid', 'full', 'tilefill')
PORTED_BY = {'stack': 'PR-2', 'recon': 'PR-3', 'wideband': 'PR-3', 'hybrid': 'PR-4', 'full': 'PR-4', 'tilefill': 'PR-5'}


def add_arguments(p):
    g = p.add_argument_group('stages')
    g.add_argument('--stages', help='comma list of stages, run in the route\'s order (default: all)')
    g.add_argument('--from', dest='stage_from', metavar='STAGE', help='first stage of a contiguous range')
    g.add_argument('--until', dest='stage_until', metavar='STAGE', help='last stage of a contiguous range')
    g = p.add_argument_group('hybrid')
    g.add_argument('--weightings', help='R075, R0 or both (default hybrid.WEIGHTINGS)')
    g.add_argument('--clean-region', choices=('horizon', 'elev10'),
                   help='recon: the CLEAN region of the first try (default hybrid_recon.CLEAN_REGION)')


def default_out(args):
    from exopipe.phase3.config import ConfigRefusal
    raise ConfigRefusal('exopipe3 hybrid needs --out: a work run of its own, e.g. a new directory under hybrid.WORK_ROOT (never '
                        'a production work run of hybrid.LSRECON_RUNS, an archive run, or a directory that contains one)')


def config_overrides(args):
    """Step flags that set config keys: the CLI applies them with --set's checks before the run record is made."""
    return [('hybrid_recon.CLEAN_REGION', args.clean_region, '--clean-region')] if args.clean_region else []


def _within(path, root):
    """path (a real path) is root or lies inside it."""
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def out_problems(out, cfg):
    """--out is a work run of its own: never a production work run the archived hybrids were made from (hybrid.LSRECON_RUNS)
    or an archive run, never inside one or above one (a directory that contains it), and never the work root hybrid.WORK_ROOT
    itself or a directory above it (the parent of the production work runs). An earlier v2 work run may be named again
    (--resume)."""
    o = os.path.realpath(out)
    held = dict(cfg.hybrid.LSRECON_RUNS, ARCHIVE_HYBRID=cfg.hybrid.ARCHIVE_HYBRID, ARCHIVE_MASTERS=cfg.hybrid.ARCHIVE_MASTERS)
    bad = []
    for k, v in sorted(held.items()):
        r = os.path.realpath(v)
        if _within(o, r):
            bad.append(f'--out {out} is inside {k} = {v} (read only)')
        elif _within(r, o):
            bad.append(f'--out {out} contains {k} = {v} (read only)')
    w = os.path.realpath(cfg.hybrid.WORK_ROOT)
    if _within(w, o):
        bad.append(f'--out {out} is the work root hybrid.WORK_ROOT = {cfg.hybrid.WORK_ROOT} or contains it: name a new run under it')
    return bad


def _csv(s):
    return [t.strip() for t in str(s).split(',') if t.strip()]


def select_stages(args):
    """The stages of this run in the route's order: --stages, or the --from / --until range, or all."""
    from exopipe.phase3.core.record import Refusal
    if args.stages and (args.stage_from or args.stage_until):
        raise Refusal('--stages and --from / --until are exclusive')
    if args.stages:
        want = _csv(args.stages)
        bad = [s for s in want if s not in STAGES]
        if bad:
            raise Refusal(f'unknown stage(s) {bad}; the stages are {list(STAGES)}')
        return [s for s in STAGES if s in want]
    for flag, s in (('--from', args.stage_from), ('--until', args.stage_until)):
        if s and s not in STAGES:
            raise Refusal(f'{flag} {s}: not a stage; the stages are {list(STAGES)}')
    i = STAGES.index(args.stage_from) if args.stage_from else 0
    j = STAGES.index(args.stage_until) if args.stage_until else len(STAGES) - 1
    if i > j:
        raise Refusal(f'--from {args.stage_from} comes after --until {args.stage_until}')
    return list(STAGES[i:j + 1])


def stage_scopes(stage, H):
    """The scopes a stage takes (H = the hybrid config section)."""
    subbands = tuple(s for s in H.SCOPES if s.endswith('MHz'))
    wide = tuple(w for w in H.SCOPES if w in H.WIDEBAND_MEMBERS)
    return {'stack': subbands, 'recon': subbands, 'wideband': wide, 'hybrid': subbands + wide, 'full': ('Full',),
            'tilefill': tuple(H.SCOPES)}[stage]


def plan_units(args, cfg):
    """[(stage, lst, scope, weighting)] in stage order, then LST, scope (route order) and weighting. Refuses a scope or
    weighting outside the route; a scope a stage does not take is skipped for that stage (a run of every stage over Blue is
    wideband, hybrid and tilefill)."""
    from exopipe.phase3.core import cells
    from exopipe.phase3.core.record import Refusal
    H = cfg.hybrid
    lsts = cells.expand(args.lsts, universe=cells.LSTS, kind='LST') if args.lsts else list(cells.LSTS)
    bad = [l for l in lsts if l not in cells.LSTS]
    if bad:
        raise Refusal(f'LST(s) {bad} are not of {cells.LSTS[0]}..{cells.LSTS[-1]}')
    scopes = cells.expand(args.scopes, universe=list(H.SCOPES), kind='scope') if args.scopes else list(H.SCOPES)
    bad = [s for s in scopes if s not in H.SCOPES]
    if bad:
        raise Refusal(f'scope(s) {bad} are not in the hybrid route (hybrid.SCOPES)')
    weightings = _csv(args.weightings) if args.weightings else list(H.WEIGHTINGS)
    bad = [w for w in weightings if w not in H.WEIGHTINGS]
    if bad:
        raise Refusal(f'weighting(s) {bad} are not of hybrid.WEIGHTINGS {list(H.WEIGHTINGS)}')
    units = []
    for st in select_stages(args):
        take = stage_scopes(st, H)
        for lst in sorted(lsts):
            for sc in [s for s in H.SCOPES if s in scopes and s in take]:
                for w in [x for x in H.WEIGHTINGS if x in weightings]:
                    units.append((st, lst, sc, w))
    return units


def _md5(path):
    """md5 of a small pinned input (ordinary reads, no memory map); None when it cannot be read."""
    import hashlib
    try:
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for blk in iter(lambda: f.read(1 << 20), b''):
                h.update(blk)
        return h.hexdigest()
    except OSError:
        return None


def stage_gate_problems(stage, cfg):
    """The configuration gates a stage refuses on before any unit's own gates (spec 3.4). [] = none."""
    out = []
    if stage == 'stack' and not os.path.isdir(cfg.hybrid_stack.COADD_SB):
        out.append(f'HG0: the co-add build directories hybrid_stack.COADD_SB = {cfg.hybrid_stack.COADD_SB} are not on this node '
                   f'(node-local; the stack rebuild test reads weights.csv and the flows there)')
    if stage in ('recon', 'wideband') and not cfg.hybrid_psf2x.REGISTRY:
        out.append('HG6: the double-size PSF registry hybrid_psf2x.REGISTRY is empty (question Q1 with Gregg: which antenna list '
                   'each band\'s PSF was made with)')
    if stage == 'hybrid':
        got = _md5(cfg.hybrid.STRIATION_SELECTION)
        if got != cfg.hybrid.STRIATION_SELECTION_MD5:
            out.append(f'the striation selection {cfg.hybrid.STRIATION_SELECTION} has md5 {got}, not the pinned '
                       f'hybrid.STRIATION_SELECTION_MD5 {cfg.hybrid.STRIATION_SELECTION_MD5}')
    if stage == 'tilefill' and not cfg.hybrid_tilefill.UNFILLED_INPUTS:
        out.append('the unfilled hybrid inputs are not pinned (hybrid_tilefill.UNFILLED_INPUTS is None; question Q2 with Gregg: '
                   'where the kept intermediates live)')
    return out


def run(args, run):
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core.record import NoInput, Refusal
    bad = out_problems(run.out_root, CFG)
    if bad:
        raise Refusal('; '.join(bad))
    units = plan_units(args, CFG)
    run.meta = {'stages': sorted({u[0] for u in units}, key=STAGES.index), 'units': len(units)}
    if not units:
        raise NoInput('no unit: the selected stages take none of the selected scopes')
    gates = {st: stage_gate_problems(st, CFG) for st in run.meta['stages']}
    if run.dry_run:
        for st, lst, sc, w in units:
            what = (f'refused: {"; ".join(gates[st])}' if gates[st] else
                    ', '.join(os.path.relpath(p, run.out_root) for p, _c in stack_plan(run.out_root, lst, sc, w)) if st == 'stack' else
                    f'not ported ({PORTED_BY[st]})')
            print(f'  plan {st}.{lst}_{sc}_{w}: {what}', flush=True)
        print(f'exopipe3 hybrid: DRY RUN, {len(units)} units, nothing written', flush=True)
        return 0
    for st, lst, sc, w in units:
        with run.unit(f'{st}.{lst}_{sc}_{w}', stage=st, lst=lst, scope=sc, weighting=w) as u:
            if gates[st]:
                u.refuse('; '.join(gates[st]) + ' - REFUSED')
            if st == 'stack':
                stack_unit(u, lst, sc, w, run.out_root, getattr(args, 'work', None))
                continue
            u.refuse(f'stage {st} is not ported yet ({PORTED_BY[st]} of plan/specs/hybrid.md) - REFUSED')
    return run.exit_code()


def _stack_class(path):
    b = os.path.basename(path)
    if os.path.basename(os.path.dirname(path)) == 'inputs':
        return 'hybrid_stack_input'
    return 'hybrid_stack_record' if b.endswith('_stack.json') else 'hybrid_psf_stack' if b.endswith('_psf_stack.fits') else 'hybrid_stack_image'


def stack_plan(out_root, lst, band, w):
    """[(path, product class)] a stack unit owes, NoTaper weighting w first, then its Taper weighting."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.hybrid import stack as HST
    from exopipe.phase3.lib.hybrid.common import TAPER_OF, inputs_dir
    workdir = os.path.join(out_root, f'{lst}_{band}')
    out = []
    for v in (w, TAPER_OF[w]):
        files = [os.path.join(workdir, f) for f in HST.outputs_of(v)]
        if v in TAPER_OF:
            files += [os.path.join(inputs_dir(workdir), os.path.basename(p))
                      for p in HST.input_copies(CFG.hybrid_stack.COADD_SB, lst, band, v)]
        out += [(p, _stack_class(p)) for p in files]
    return out


def stack_unit(u, lst, band, w, out_root, work):
    """Stage stack of one unit (spec 2.1): the NoTaper stacks of weighting w, then the Taper residual stack of its Taper weighting.
    A variant a gate refuses is a refused item (its files, with the gate); the other goes on. Each variant runs in a fresh node-local
    scratch directory, removed afterwards; its own earlier outputs in the work directory are removed first (no stale file stands
    beside a refusal)."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import fitsio
    from exopipe.phase3.lib.hybrid import stack as HST
    from exopipe.phase3.lib.hybrid.common import TAPER_OF
    workdir = os.path.join(out_root, f'{lst}_{band}')
    os.makedirs(workdir, exist_ok=True)
    screens = os.path.join(fitsio.scratch_root(work), CFG.hybrid_stack.SCREEN_CACHE_SUBDIR)
    plan = stack_plan(out_root, lst, band, w)
    nights = set()
    for v in (w, TAPER_OF[w]):
        own = [os.path.join(workdir, f) for f in HST.outputs_of(v)]
        files = [(p, c) for p, c in plan if p in own or (v in TAPER_OF and c == 'hybrid_stack_input')]
        for p, c in files:
            u.plan(p, c)
        for p in own:
            if os.path.lexists(p):
                os.remove(p)
        prep = fitsio.scratch_dir(f'hybrid_stack_{lst}_{band}_{v}', work)
        read = set()
        try:
            rec = HST.stack_variant(lst, band, v, workdir, prep, screens, read)
        except HST.StackRefusal as e:
            for p, _c in files:
                u.refuse_item(p, f'{e.gate}: {e}', gate=e.gate)
            continue
        finally:
            fitsio.cleanup(prep)
            for p in sorted(read):
                u.add_input(p)
        t = rec['test_stack_rebuild']
        u.check(f'hybrid_stack.rebuild_{v}', True, value=t['verdict'],
                detail=f"max |diff| {t['max_abs_diff']:.3g} Jy/beam, rms {t['rms_diff']:.3g} Jy/beam vs master peak "
                       f"{t['master_peak']:.4g} Jy/beam, rms {t['master_rms']:.4g} Jy/beam")
        nights.update(n['date'] for n in rec['nights'])
        master = HST.cell_paths(CFG.hybrid_stack.COADD_SB, lst, band, v)[2]
        for p, c in files:
            if c == 'hybrid_stack_input':
                i = rec['inputs'][os.path.basename(p)]
                u.add_output(p, c, md5=i['md5'], installed=i['installed'])
            elif c == 'hybrid_stack_record':
                u.add_output(p, c, lst=lst, band=band, weighting=v)
            else:
                u.add_output(p, c, master=master)
    u.set_admitted(sorted(nights))
