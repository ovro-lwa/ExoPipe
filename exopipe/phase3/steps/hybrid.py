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
PULL REQUEST PR-4: the hybrid and full stages (lib/hybrid/master.py). A unit hybrid.<LST>_<scope>_<w> reads the scope's
reconstruction in --out (no <w>_lsrecon(_wb).json = no input; a recon or wideband record that did not end done refuses it), the
installed Taper master of weighting w and, at a striation scope, the striation-restored base; it writes, in <--out>/<LST>_<scope>/,
the hybrid I_deep_Taper_<w>_lsrecon_lsfill_<scope>.fits, its beam-corrected sibling (..._lsfill_pbcorr_...), the CLEAN-residual
hybrid (..._lsfill_cleanres_...) and the record <w>_hybrid.json. The CLEAN residual needs the Taper residual stack: when the stack
stage refused it, that file is a refused item and the unit is done. A unit full.<LST>_Full_<w> combines the 15 subband hybrids in
--out (none = no input; a member without a passed hybrid refuses it) and writes PBdiffuse_deep_Full.fits in place of a CLEAN
residual.
PULL REQUEST PR-3: the recon and wideband stages (lib/hybrid/recon.py, wideband.py). A unit recon.<LST>_<band>_<w> reads its
stack (stage stack, same --out; no stack record = no input) and writes, in <--out>/<LST>_<band>/, the reconstruction
I_deep_NoTaper_<w>_lsrecon_<band>.fits and its split beam-corrected sibling (..._lsrecon_pbcorr_...), the work files
<w>_lsrecon_{model,clean_residual,compact_restored,final_residual}.fits and the record <w>_lsrecon.json; a unit refused by
test_clean_converged is retried along hybrid_recon.CASCADE. A unit wideband.<LST>_<Red|Green|Blue>_<w> combines the members'
reconstructions in --out into <--out>/<LST>_<scope>/ (the same names, record <w>_lsrecon_wb.json); a member without a passed
reconstruction refuses it. Every production test refuses (ReconRefusal), and the content tests re-read the files.
PULL REQUEST PR-2: the stack stage (lib/hybrid/stack.py, warp.py). A unit stack.<LST>_<band>_<w> writes, in <--out>/<LST>_<band>/,
the NoTaper stacks of weighting w (<w>_dirty_stack, _dirty_halfA, _dirty_halfB, _psf_stack, <w>_stack.json, inputs/ copies) and
the Taper residual stack of its Taper weighting (T075 for R075, T0 for R0: <t>_residual_stack, <t>_stack.json), as production's
`stack` action did for the four weightings. A variant a gate refuses (HG1 membership / exclusion, HG4 night drops, HG5 rebuild) is
a refused item of the unit - its files, with the gate - and the other variant goes on; both refused = the unit refused. Node-local
scratch: --work or the site SCRATCH_ROOT (the derived warp screens are cached there, hybrid_stack.SCREEN_CACHE_SUBDIR).

Gates here (each refuses; spec 3.4): HG0 the node-local co-add build directories (hybrid_stack.COADD_SB) are present on this node
(stack); HG6 the double-size PSF registry is pinned (recon, wideband) and every file's md5 equals its entry (recon); the striation
selection's md5 equals the pinned one (hybrid); the unfilled inputs are pinned (tilefill); HG20 every refused unit is listed and
makes the exit non-zero (core/record.Run.exit_code); a scope or weighting outside the route refuses before any unit runs; --out
never is, lies in or contains a production work run or an archive run, and is never the work root itself.
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
    """md5 of a pinned input (lib/install/steps_common.md5: ordinary reads, no memory map); None when it cannot be read."""
    from exopipe.phase3.lib.install.steps_common import md5
    try:
        return md5(path)
    except OSError:
        return None


def stage_gate_problems(stage, cfg):
    """The configuration gates a stage refuses on before any unit's own gates (spec 3.4). [] = none."""
    out = []
    if stage in ('stack', 'hybrid', 'full') and not os.path.isdir(cfg.hybrid_stack.COADD_SB):
        out.append(f'HG0: the co-add build directories hybrid_stack.COADD_SB = {cfg.hybrid_stack.COADD_SB} are not on this node '
                   f'(node-local; the stack rebuild test reads weights.csv and the flows there; the hybrid span reads a refused Taper '
                   f'stack\'s membership and a wideband master\'s weights.csv there)')
    if stage in ('recon', 'wideband') and not cfg.hybrid_psf2x.REGISTRY:
        out.append('HG6: the double-size PSF registry hybrid_psf2x.REGISTRY is empty (question Q1 with Gregg: which antenna list '
                   'each band\'s PSF was made with)')
    elif stage == 'recon':
        for (band, v), (path, md5, _ants) in sorted(cfg.hybrid_psf2x.REGISTRY.items()):
            got = _md5(path)
            if got != md5:
                out.append(f'HG6: the double-size PSF {band} {v} {path} has md5 {got}, not the pinned hybrid_psf2x.REGISTRY md5 {md5}')
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
        plans = {'stack': stack_plan, 'recon': recon_plan, 'wideband': wideband_plan, 'hybrid': hybrid_plan, 'full': hybrid_plan}
        for st, lst, sc, w in units:
            what = (f'refused: {"; ".join(gates[st])}' if gates[st] else
                    ', '.join(os.path.relpath(p, run.out_root) for p, _c in plans[st](run.out_root, lst, sc, w)) if st in plans else
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
            if st == 'recon':
                recon_unit(u, lst, sc, w, run.out_root)
                continue
            if st == 'wideband':
                wideband_unit(u, lst, sc, w, run.out_root)
                continue
            if st in ('hybrid', 'full'):
                hybrid_master_unit(u, lst, sc, w, run.out_root)
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
    scratch directory, removed afterwards; its own earlier outputs and their temporaries in the work directory are removed first,
    and whatever it wrote is removed when a gate refuses it (no stale file stands beside a refusal). The inputs/ copies are verified
    copies of installed files that the cell's two weightings share: a refusal removes only this weighting's temporaries of them.
    A lock on a derived warp screen held past hybrid_stack.SCREEN_LOCK_WAIT_S refuses the variant (HG4). The unit's admitted nights
    are the nights its written stacks hold (a night dropped from a stack is not admitted)."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import fitsio
    import glob
    from exopipe.phase3.lib.hybrid import stack as HST
    from exopipe.phase3.lib.hybrid.common import TAPER_OF, inputs_dir
    from exopipe.phase3.lib.hybrid.warp import ScreenLockRefusal
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
        temps = lambda: sorted(glob.glob(os.path.join(glob.escape(inputs_dir(workdir)), f'*.{v}.part')))
        _clear(own + temps())
        prep = fitsio.scratch_dir(f'hybrid_stack_{lst}_{band}_{v}', work)
        read = set()
        try:
            rec = HST.stack_variant(lst, band, v, workdir, prep, screens, read)
        except (HST.StackRefusal, ScreenLockRefusal) as e:
            gate = getattr(e, 'gate', 'HG4')
            _clear(own + temps())
            for p, _c in files:
                u.refuse_item(p, f'{gate}: {e}', gate=gate)
            continue
        finally:
            fitsio.cleanup(prep)
            for p in sorted(read):
                u.add_input(p)
        t = rec['test_stack_rebuild']
        u.check(f'hybrid_stack.rebuild_{v}', True, value=t['verdict'],
                detail=f"max |diff| {t['max_abs_diff']:.3g} Jy/beam, rms {t['rms_diff']:.3g} Jy/beam vs master peak "
                       f"{t['master_peak']:.4g} Jy/beam, rms {t['master_rms']:.4g} Jy/beam")
        vn = set(n['date'] for n in rec['nights']) - set(rec.get('dirty_nights_dropped') or ())
        master = HST.cell_paths(CFG.hybrid_stack.COADD_SB, lst, band, v)[2]
        n0 = len(u.failed_tests())
        for p, c in files:
            if c == 'hybrid_stack_input':
                i = rec['inputs'][os.path.basename(p)]
                u.add_output(p, c, md5=i['md5'], installed=i['installed'])
            elif c == 'hybrid_stack_record':
                u.add_output(p, c, lst=lst, band=band, weighting=v)
            else:
                u.add_output(p, c, master=master)
        bad = u.failed_tests()[n0:]
        if bad:                                   # a failed content test refuses the variant: its files go (a stack record on disk
            why = '; '.join(f"{t['name']} {t.get('detail', '')}"[:200] for t in bad)   # is a passed stack), and a bad inputs copy too
            _clear(own + temps() + sorted({t['path'] for t in bad if t.get('path') and t['path'] not in own}))
            for p, _c in files:
                u.refuse_item(p, f'content test(s) FAILED: {why}', gate='content')
            continue
        nights |= vn
    u.set_admitted(sorted(nights))


def recon_plan(out_root, lst, band, w):
    """[(path, product class)] a recon unit owes (NL 870-915): the reconstruction and its split beam-corrected sibling (the
    products), the four work files, the record."""
    wd = os.path.join(out_root, f'{lst}_{band}')
    return ([(os.path.join(wd, f'I_deep_NoTaper_{w}_lsrecon_{band}.fits'), 'hybrid_recon_master'),
             (os.path.join(wd, f'I_deep_NoTaper_{w}_lsrecon_pbcorr_{band}.fits'), 'hybrid_recon_pbcorr')]
            + [(os.path.join(wd, f'{w}_lsrecon_{k}.fits'), 'hybrid_recon_work')
               for k in ('model', 'clean_residual', 'compact_restored', 'final_residual')]
            + [(os.path.join(wd, f'{w}_lsrecon.json'), 'hybrid_recon_record')])


def wideband_plan(out_root, lst, scope, w):
    """[(path, product class)] a wideband unit owes (NL 954-1074)."""
    wd = os.path.join(out_root, f'{lst}_{scope}')
    return [(os.path.join(wd, f'I_deep_NoTaper_{w}_lsrecon_{scope}.fits'), 'hybrid_recon_master'),
            (os.path.join(wd, f'I_deep_NoTaper_{w}_lsrecon_pbcorr_{scope}.fits'), 'hybrid_recon_pbcorr'),
            (os.path.join(wd, f'{w}_lsrecon_compact_restored.fits'), 'hybrid_recon_work'),
            (os.path.join(wd, f'{w}_lsrecon_wb.json'), 'hybrid_wideband_record')]


def _clear(paths):
    """Remove a unit's own earlier outputs and their temporaries <path>.part and <path>.tmp.* (no stale file stands beside a
    new run or a refusal)."""
    import glob
    for p in paths:
        for q in [p, f'{p}.part'] + glob.glob(f'{glob.escape(p)}.tmp.*'):
            if os.path.lexists(q):
                os.remove(q)


def _upstream_problem(out_root, unit, paths):
    """Why an upstream unit's files on disk may not be used, else None: its v2 unit record (when there is one; inputs staged from
    production have none) must be done and must not list any of `paths` as a refused item."""
    from exopipe.phase3.lib.hybrid.common import unit_record
    r = unit_record(out_root, unit)
    if r is None:
        return None
    if r.get('status') != 'done':
        return f"unit {unit} ended {r.get('status')}: {str(r.get('reason'))[:200]}"
    want = {os.path.abspath(p) for p in paths}
    hit = [i for i in r.get('refused_items') or [] if i.get('path') in want]
    return f"unit {unit} refused {os.path.basename(hit[0]['path'])}: {hit[0]['reason'][:200]}" if hit else None


def _stack_inputs(wd, w):
    return ([os.path.join(wd, f'{w}_{k}.fits') for k in ('dirty_stack', 'dirty_halfA', 'dirty_halfB', 'psf_stack')]
            + [os.path.join(wd, f'{w}_stack.json')])


def recon_unit(u, lst, band, w, out_root):
    """Stage recon of one unit (spec 2.3): production's recon_variant on the unit's stack (stage stack, same --out), with the
    convergence cascade of hybrid_recon.CASCADE (NL 2209-2219: a unit refused by test_clean_converged is retried with NITER and
    MAXCYC times the factor and the try's CLEAN region; each later try records the earlier ones). No stack record = no input (its
    stack refused or was not run). Any other refusal, or a content test that fails, refuses the unit and removes what it wrote."""
    import json
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core.record import NoInput
    from exopipe.phase3.lib.hybrid import recon as REC
    from exopipe.phase3.lib.hybrid.common import inputs_dir
    from exopipe.phase3.lib.masters import pb_apply as PBA
    wd = os.path.join(out_root, f'{lst}_{band}')
    srec_path = os.path.join(wd, f'{w}_stack.json')
    if not os.path.exists(srec_path):
        raise NoInput(f'no stack record {srec_path} (stage stack of this unit refused or was not run)')
    why = _upstream_problem(out_root, f'stack.{lst}_{band}_{w}', _stack_inputs(wd, w))
    if why:
        u.refuse(f'its stack may not be used: {why} - REFUSED')
    plan = recon_plan(out_root, lst, band, w)
    for p, c in plan:
        u.plan(p, c)
    own = [p for p, _c in plan]
    psf2x = CFG.hybrid_psf2x.REGISTRY[(band, w)][0]
    g = REC.bind()
    master = os.path.join(inputs_dir(wd), f'I_deep_NoTaper_{w}_{band}.fits')
    pb = os.path.join(inputs_dir(wd), PBA.deep_pb_name(band, w))
    _clear(own)
    done = []
    try:
        base = CFG.hybrid_recon.CLEAN_REGION     # NL 2209: the cascade's 'horizon' tries take the configured region (--clean-region)
        tries = []
        for _name, f, region in CFG.hybrid_recon.CASCADE:     # a try equal to an earlier one is not run again
            t = (int(g['NITER'] * f), int(g['MAXCYC'] * f), base if region == 'horizon' else region)
            if t not in tries:
                tries.append(t)
        REC.READS.clear()
        for i, (ni, mc, cr) in enumerate(tries):
            try:
                rec = REC.recon_variant(lst, band, w, wd, psf2x, None, ni, mc, cr, escalation=done)
                break
            except REC.ReconRefusal as e:
                if 'test_clean_converged FAILED' not in str(e) or i == len(tries) - 1:
                    raise
                done.append(dict(niter=ni, maxcyc=mc, clean_region=cr, refused=str(e)[:300]))
                REC.log(lst, band, w, 'convergence refused at', ni, mc, cr, '- escalating')
    except REC.ReconRefusal as e:
        _clear(own)
        u.refuse(f'{e}')
    except BaseException:
        _clear(own)
        raise
    finally:
        for p in _stack_inputs(wd, w) + [master, pb, psf2x] + sorted(REC.READS):     # READS: sky models and the catalog
            if os.path.exists(p):
                u.add_input(p)
    n0 = len(u.failed_tests())
    for t in ('test_psf2x', 'test_compact_sources', 'test_large_scale_level'):
        v = rec[t].get('verdict')
        u.check(f'hybrid_recon.{t}', v in ('pass', 'caveat'), value=v, detail=json.dumps(rec[t], default=str)[:400])
    if done:
        u.note(f'convergence cascade: {len(done)} earlier tr{"y" if len(done) == 1 else "ies"} refused, '
               f'clean region {rec["regions"]["clean_region"]}')
    pbs = [(pb, band, 1.0)]
    for p, c in plan:
        if c == 'hybrid_recon_master':
            u.add_output(p, c, master=master, scope=band, weighting=w)
        elif c == 'hybrid_recon_pbcorr':
            u.add_output(p, c, apparent=plan[0][0], compact=os.path.join(wd, f'{w}_lsrecon_compact_restored.fits'), pbs=pbs)
        elif c == 'hybrid_recon_work':
            u.add_output(p, c, master=master)
        else:
            u.add_output(p, c, lst=lst, band=band, weighting=w)
    bad = u.failed_tests()[n0:]
    if bad:
        _clear(own)
        u.refuse('content test(s) FAILED: ' + '; '.join(f"{t['name']} {t.get('detail', '')}"[:200] for t in bad))
    with open(srec_path) as f:
        srec = json.load(f)
    u.set_admitted(sorted(set(n['date'] for n in srec['nights']) - set(srec.get('dirty_nights_dropped') or ())))


def _member_product(mwd, w):
    """The reconstruction a member's record names (combine_wideband reads that path); '' when the record is absent."""
    import json
    p = os.path.join(mwd, f'{w}_lsrecon.json')
    try:
        with open(p) as f:
            return json.load(f)['product']['path']
    except (OSError, ValueError, KeyError, TypeError):
        return ''


def wideband_unit(u, lst, scope, w, out_root):
    """Stage wideband of one unit (spec 2.4): production's combine_wideband over the members' reconstructions in --out. A member
    without a passed reconstruction refuses the unit (as production); a refusal or a failed content test removes what it wrote."""
    import json
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.hybrid import recon as REC
    from exopipe.phase3.lib.hybrid import wideband as WB
    from exopipe.phase3.lib.hybrid.common import inputs_dir
    from exopipe.phase3.lib.masters import pb_apply as PBA
    members = tuple(CFG.hybrid.WIDEBAND_MEMBERS[scope])
    plan = wideband_plan(out_root, lst, scope, w)
    for p, c in plan:
        u.plan(p, c)
    own = [p for p, _c in plan]
    _clear(own)
    for b in members:
        why = _upstream_problem(out_root, f'recon.{lst}_{b}_{w}', [os.path.join(out_root, f'{lst}_{b}', f'{w}_lsrecon.json')])
        if why:
            u.refuse(f'member {b} may not be used: {why} - REFUSED')
    REC.READS.clear()
    try:
        rec = WB.combine_wideband(lst, scope, w, out_root)
    except REC.ReconRefusal as e:
        _clear(own)
        u.refuse(f'{e}')
    except BaseException:
        _clear(own)
        raise
    finally:
        for b in members:
            mwd = os.path.join(out_root, f'{lst}_{b}')
            for p in [os.path.join(mwd, f'{w}_lsrecon.json'), os.path.join(mwd, f'{w}_stack.json'), _member_product(mwd, w),
                      os.path.join(mwd, f'{w}_lsrecon_compact_restored.fits'),
                      os.path.join(inputs_dir(mwd), PBA.deep_pb_name(b, w))]:
                if os.path.exists(p):
                    u.add_input(p)
        for p in sorted(REC.READS):               # the compact-source catalog
            u.add_input(p)
    # the archive masters combine_wideband read: the installed wideband master (the compact-source reference) and, where read, the
    # members' installed masters (the caveat scopes' subband reference; THERMNOI from another master of the cell)
    u.add_input(rec['installed_wideband_master'])
    g = REC.bind()
    for s in rec['subbands']:
        if scope in g['CAVEAT_SCOPES']:
            u.add_input(REC.installed_master(lst, s['band'], w, False)[0])
        if not str(s.get('thermnoi_source', '')).startswith('reconstruction'):
            ow = 'R0' if w == 'R075' else 'R075'
            for tp_, v_ in ((True, w), (False, ow), (True, ow)):
                try:
                    u.add_input(REC.installed_master(lst, s['band'], v_, tp_)[0])
                except REC.ReconRefusal:              # not reached by combine_wideband (an earlier master held THERMNOI)
                    pass
    sub0 = _member_product(os.path.join(out_root, f'{lst}_{members[0]}'), w)       # the product combine_wideband read
    pbs = [(os.path.join(inputs_dir(os.path.join(out_root, f'{lst}_{s["band"]}')), PBA.deep_pb_name(s['band'], w)), s['band'], s['weight'])
           for s in rec['subbands']]
    n0 = len(u.failed_tests())
    for p, c in plan:
        if c == 'hybrid_recon_master':
            u.add_output(p, c, master=sub0, scope=scope, weighting=w, wideband=True, members=list(members))
        elif c == 'hybrid_recon_pbcorr':
            u.add_output(p, c, apparent=plan[0][0], compact=plan[2][0], pbs=pbs)
        elif c == 'hybrid_recon_work':
            u.add_output(p, c, master=sub0)
        else:
            u.add_output(p, c, lst=lst, scope=scope, weighting=w, members=list(members))
    bad = u.failed_tests()[n0:]
    if bad:
        _clear(own)
        u.refuse('content test(s) FAILED: ' + '; '.join(f"{t['name']} {t.get('detail', '')}"[:200] for t in bad))
    nights = set()
    for b in members:
        with open(os.path.join(out_root, f'{lst}_{b}', f'{w}_stack.json')) as f:
            s = json.load(f)
        nights |= set(n['date'] for n in s['nights']) - set(s.get('dirty_nights_dropped') or ())
    u.set_admitted(sorted(nights))


def hybrid_plan(out_root, lst, scope, w):
    """[(path, product class)] a hybrid unit owes (NL 1274-1389): the hybrid, its beam-corrected sibling, its CLEAN residual, the
    record; Full (NL 1404-1504): no CLEAN residual (its subbands sit on three grids), and the cell's diffuse PB map from the unit of
    weighting hybrid_full.PBDIFFUSE_WEIGHTING only (one file for both weightings)."""
    from exopipe.phase3.config import CFG
    wd = os.path.join(out_root, f'{lst}_{scope}')
    out = [(os.path.join(wd, f'I_deep_Taper_{w}_lsrecon_lsfill_{scope}.fits'), 'hybrid_master_product'),
           (os.path.join(wd, f'I_deep_Taper_{w}_lsrecon_lsfill_pbcorr_{scope}.fits'), 'hybrid_master_product')]
    if scope == 'Full':
        if w == CFG.hybrid_full.PBDIFFUSE_WEIGHTING:
            out.append((os.path.join(wd, f'PBdiffuse_deep_{scope}.fits'), 'hybrid_pbdiffuse'))
    else:
        out.append((os.path.join(wd, f'I_deep_Taper_{w}_lsrecon_lsfill_cleanres_{scope}.fits'), 'hybrid_master_product'))
    return out + [(os.path.join(wd, f'{w}_hybrid.json'), 'hybrid_record')]


def _hybrid_kind(path):
    return 'pbcorr' if '_lsfill_pbcorr_' in path else 'cleanres' if '_lsfill_cleanres_' in path else 'hybrid'


def hybrid_master_unit(u, lst, scope, w, out_root):
    """Stage hybrid (spec 2.5) or full (2.6) of one unit: production's hybrid_unit / hybrid_full over the unit's reconstruction
    (stages recon / wideband) or the cell's 15 subband hybrids (full), in --out. No reconstruction record (no subband hybrid, for
    Full) = no input; a wideband or Full unit whose member holds no passed hybrid refuses. The CLEAN residual that a refused Taper
    residual stack costs is a refused item with the recorded reason (Gregg 2026-10-31, "ok on caveats"). Any other refusal, or a
    failed content test, refuses the unit and removes what it wrote. The unit's inputs are every file the stage read
    (master.READS) except its own outputs; its admitted nights are the hybrid's span (the union of its inputs' nights)."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core.record import NoInput
    from exopipe.phase3.lib.hybrid import master as M
    from exopipe.phase3.lib.hybrid import recon as REC
    from exopipe.phase3.lib.hybrid.stack import StackRefusal
    wd = os.path.join(out_root, f'{lst}_{scope}')
    members = tuple(CFG.hybrid_full.MEMBERS) if scope == 'Full' else tuple(CFG.hybrid.WIDEBAND_MEMBERS.get(scope, ()))
    hyb = lambda b: os.path.join(out_root, f'{lst}_{b}', f'{w}_hybrid.json')
    if scope == 'Full':
        if not any(os.path.exists(hyb(b)) for b in members):
            raise NoInput(f'no subband hybrid of {lst} {w} (stage hybrid refused or was not run)')
    else:
        rj = os.path.join(wd, f'{w}_lsrecon_wb.json' if members else f'{w}_lsrecon.json')
        if not os.path.exists(rj):
            raise NoInput(f'no reconstruction record {rj} (its stage refused or was not run)')
        why = _upstream_problem(out_root, f'{"wideband" if members else "recon"}.{lst}_{scope}_{w}', [rj])
        if why:
            u.refuse(f'its reconstruction may not be used: {why} - REFUSED')
    for b in members:
        if not os.path.exists(hyb(b)):
            u.refuse(f'{lst} {scope} {w}: {b} holds no passed hybrid ({hyb(b)}) - REFUSED')
        why = _upstream_problem(out_root, f'hybrid.{lst}_{b}_{w}', [hyb(b)])
        if why:
            u.refuse(f'member {b} may not be used: {why} - REFUSED')
    for b in members or (scope,):                 # the span reads each subband's stack records and its Taper residual stack
        why = _upstream_problem(out_root, f'stack.{lst}_{b}_{w}', [])
        if why:
            u.refuse(f'the stacks of {b} may not be used: {why} - REFUSED')
    plan = hybrid_plan(out_root, lst, scope, w)
    for p, c in plan:
        u.plan(p, c)
    own = [p for p, _c in plan]
    _clear(own)
    M.READS.clear()
    try:
        rec = M.hybrid_unit(lst, scope, w, out_root)
    except (REC.ReconRefusal, StackRefusal) as e:
        _clear(own)
        u.refuse(f'{e}')
    except BaseException:
        _clear(own)
        raise
    finally:
        outs = {os.path.abspath(p) for p, _c in plan}
        for p in sorted(M.READS):
            if os.path.abspath(p) not in outs and os.path.exists(p):
                u.add_input(p)
    n0 = len(u.failed_tests())
    for p, c in plan:
        if c == 'hybrid_master_product':
            kind = _hybrid_kind(p)
            if kind == 'cleanres' and rec.get('product_cleanres') is None:
                u.refuse_item(p, f"no CLEAN residual product: {rec.get('cleanres_skipped')}", gate='taper-residual-stack')
                continue
            u.add_output(p, c, scope=scope, weighting=w, kind=kind, taper=rec['taper_master'])
        elif c == 'hybrid_pbdiffuse':
            u.add_output(p, c, taper=rec['taper_master'])
        else:
            u.add_output(p, c, lst=lst, scope=scope, weighting=w)
    bad = u.failed_tests()[n0:]
    if bad:
        _clear(own)
        u.refuse('content test(s) FAILED: ' + '; '.join(f"{t['name']} {t.get('detail', '')}"[:200] for t in bad))
    if rec.get('caveats'):
        u.note(f"recorded caveats: small scales {rec['caveats']['small_scale_rel']:.3g}"
               f"{' (caveat)' if rec['caveats']['small_scale_caveat'] else ''}, level {(rec['caveats']['level'] or {}).get('verdict')}")
    u.set_admitted(sorted(n['date'] for n in rec['span']['nights']))
