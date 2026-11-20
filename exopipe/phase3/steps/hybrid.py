"""exopipe3 hybrid - the hybrid NoTaper route (spec plan/specs/hybrid.md): NoTaper stacks, the large-scale reconstruction, the
wideband reconstructions, the hybrid masters (Taper master + G (x) (reconstruction - Taper)), the Full hybrid and the per-tile
zero-spacing fill. Production: exopipe/phase3/notaper_largescale.py (NL) and mosaic_tpfill.py (MF) at exopipe main 4c7098c.

    exopipe3 hybrid --lsts 18h --scopes 46MHz --weightings R075 --stages stack --out W --work F
    exopipe3 hybrid --lsts '*' --scopes Blue --stages hybrid,full --out W
    exopipe3 hybrid --lsts 03h --scopes 55MHz --weightings R0 --stages tilefill --tilefill-source pinned --out W

Use: survey (intermittent, DESIGN 12.2) - run on demand when masters are rebuilt; not part of the per-hour chain. Unit of work:
(stage, LST, scope, weighting) - unit name <stage>.<LST>_<scope>_<weighting>, e.g. stack.18h_46MHz_R075.

Stages, in order (--stages a,b or --from / --until; default all), and the scopes each one takes:
  stack      subbands               NoTaper dirty / PSF stacks and Taper residual stacks (PR-2)
  recon      subbands               the large-scale reconstruction (PR-3; needs the double-size PSF registry, Q1)
  wideband   Red, Green, Blue       the wideband reconstructions from the members' reconstructions (PR-3)
  hybrid     subbands + widebands   the hybrid master, beam-corrected sibling, CLEAN residual, PB maps (PR-4)
  full       Full                   the Full hybrid from the 15 subband hybrids (PR-4)
  tilefill   every scope            the per-tile zero-spacing fill of the hybrid masters (PR-5)

PULL REQUEST PR-1: the selection, the unit list and the gates that need no stage code. Every unit of a stage not yet ported REFUSES
(exit 2) with the pull request that ports it; nothing is written but the unit records.
PULL REQUEST PR-5: the tilefill stage (lib/hybrid/tilefill.py, production MF 211-400 and the 11-11 fill driver). Per scope and
weighting, once per run, the band field: the LFSS donor increment that turns the unfilled Jy/sr mosaic of
hybrid_tilefill.UNFILLED_MOSAIC_RUNS into the filled one of FILLED_MOSAIC_RUN (identity <= 1, scale TPSCALE), written to
<--out>/_tilefill_fields/. A unit tilefill.<LST>_<scope>_<w> adds it, tile by tile, to the hybrid master and its beam-corrected
sibling - from the hybrid stage's products in --out (--tilefill-source work, the default; no <w>_hybrid.json = no input), or from
the unfilled sources the installed filled masters' TPSRCMD5 names in the pinned manifests (pinned; a unit rebuilt in --out refuses
it) - and writes <--out>/<LST>_<scope>/tilefill/<the masters' product names> and <w>_tilefill.json; in work mode also the CLEAN
residual under its name without 'lsfill'. Production's tests refuse (identity, blanking, compact sources, the beam-corrected round
trip); the ruled caveat units (hybrid_tilefill.TPCAVEAT_UNITS) are decided before the write. A unit whose data hold a night now
excluded for the subband holding it refuses (wideband and Full per member stack; pinned: the nights of the production record whose
products are the TPSRCMD5 sources); every unfilled input must still have its pinned or recorded md5 when read; a band field already
in --out is reused while its record passed and its file and mosaics are unchanged. The repair cards of the installed Taper
master are carried (D7), TPMOSAIC and TPFILLED are full names (D8), GITTAG is the writer's with a HISTORY line naming the build (D9).
The archive copy of the products is a separate class (PR-5b).
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
selection's md5 equals the pinned one (hybrid); HG13 the unfilled inputs are pinned (hybrid_tilefill.UNFILLED_INPUTS, each
manifest's sha256 equal; every unfilled master and mosaic read is a manifest row with its md5 re-computed) (tilefill); HG20 every refused unit is listed and
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
    g.add_argument('--tilefill-source', choices=('work', 'pinned'), default='work',
                   help='tilefill: fill the hybrid stage\'s unfilled masters in --out (work, the default; no hybrid record = no '
                        'input), or re-fill the installed filled masters from the pinned unfilled sources their TPSRCMD5 names '
                        '(pinned: a unit not rebuilt whose band field changed). Under work, a unit kept by ruling '
                        '(hybrid_unit.REFUSED_INSTALLED keep, no hybrid record) is filled as pinned')
    g.add_argument('--mosaic-out', metavar='DIR',
                   help='tilefill: take the band fields from the hybrid mosaics of the `exopipe3 mosaic --generation v5` work run DIR (its '
                        'done conform records: the conformed filled Jy/sr map and its unfilled twin at their recorded md5s, the '
                        'products the archive receives) instead of the archived ones '
                        '(hybrid_tilefill.FILLED_MOSAIC_RUN with the pinned unfilled twins)')


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
    if stage == 'tilefill':
        T = cfg.hybrid_tilefill
        if not T.UNFILLED_INPUTS:
            out.append('HG13: the unfilled hybrid inputs are not pinned (hybrid_tilefill.UNFILLED_INPUTS is empty)')
        else:
            try:
                _manifest_rows(T)
            except Exception as e:
                out.append(f'HG13: the pinned manifests of hybrid_tilefill.UNFILLED_INPUTS cannot be used: {e}')
        for p in (T.FILLED_MOSAIC_RUN,) + tuple(T.UNFILLED_MOSAIC_RUNS):
            if not os.path.isdir(p):
                out.append(f'HG13: the mosaic run {p} (hybrid_tilefill) is not there')
    return out


def run(args, run):
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core.record import NoInput, Refusal
    bad = out_problems(run.out_root, CFG)
    if (getattr(args, 'workers', None) or 1) > 1:       # FINDING-20261118-V2-HYBRID-STEP-IGNORES-WORKERS: a flag never silently ignored
        bad.append(f'--workers {args.workers}: exopipe3 hybrid runs its units one after another in one process (lib/hybrid keeps '
                   'per-process module state); run one process per disjoint --lsts group instead (unit records are per unit)')
    if bad:
        raise Refusal('; '.join(bad))
    units = plan_units(args, CFG)
    run.meta = {'stages': sorted({u[0] for u in units}, key=STAGES.index), 'units': len(units)}
    if not units:
        raise NoInput('no unit: the selected stages take none of the selected scopes')
    gates = {st: stage_gate_problems(st, CFG) for st in run.meta['stages']}
    if run.dry_run:
        from exopipe.phase3.lib.hybrid.recon import ReconRefusal
        from exopipe.phase3.lib.hybrid.tilefill import TilefillRefusal
        plans = {'stack': stack_plan, 'recon': recon_plan, 'wideband': wideband_plan, 'hybrid': hybrid_plan, 'full': hybrid_plan,
                 'tilefill': _tilefill_dry_plan(args.tilefill_source)}
        for st, lst, sc, w in units:
            if gates[st]:
                what = f'refused: {"; ".join(gates[st])}'
            else:
                try:
                    what = ', '.join(os.path.relpath(p, run.out_root) for p, _c in plans[st](run.out_root, lst, sc, w))
                except NoInput as e:                   # e.g. tilefill work: the hybrid stage of this run makes its input
                    what = f'no input (yet): {e}'
                except (Refusal, TilefillRefusal, ReconRefusal) as e:      # a unit whose inputs refuse (tilefill: its source)
                    what = f'refused: {e}'
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
            if st == 'tilefill':
                tilefill_unit(u, lst, sc, w, run.out_root, args.tilefill_source, getattr(args, 'mosaic_out', None))
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


# ---- stage tilefill (PR-5; spec 2.14, D7-D9, HG13) ---------------------------------------------------------------------------------

ROBUST_TOKEN = {'R075': 'Robust-0.75', 'R0': 'Robust+0.0'}       # the weighting's token in product names (HMF 39)
_FIELDS = {}                                                     # (out root, scope, weighting) -> the band field record, once per run
_ROWS = {}                                                       # the pinned manifests' rows, read once per run


def _manifest_rows(T):
    """{path: md5} of every row of the pinned input manifests (hybrid_tilefill.UNFILLED_INPUTS), each manifest's sha256 checked."""
    import csv
    import hashlib
    from exopipe.phase3.core.record import Refusal
    ident = tuple(sorted((k, tuple(v)) for k, v in T.UNFILLED_INPUTS.items()))
    if ident in _ROWS:
        return _ROWS[ident]
    rows = {}
    for key, (path, sha) in sorted(T.UNFILLED_INPUTS.items()):
        with open(path, 'rb') as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != sha:
            raise Refusal(f'HG13: the pinned manifest {key} {path} has sha256 {got}, not hybrid_tilefill.UNFILLED_INPUTS {sha}')
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                if r.get('md5'):
                    rows[r['path']] = r['md5']
    _ROWS[ident] = rows
    return rows


def _pinned(path, rows):
    """HG13: the md5 a pinned manifest lists for path, re-computed now; refuses when the file is not listed or has changed."""
    from exopipe.phase3.lib.hybrid.tilefill import TilefillRefusal
    if path not in rows:
        raise TilefillRefusal(f'HG13: {path} is not a pinned input (no row in hybrid_tilefill.UNFILLED_INPUTS) - REFUSING')
    got = _md5(path)
    if got != rows[path]:
        raise TilefillRefusal(f'HG13: {path} has md5 {got}, not its pinned md5 {rows[path]} - REFUSING')
    return got


def mosaic_pair(scope, w, T):
    """(filled, unfilled): the one filled Jy/sr mosaic of the scope and weighting in hybrid_tilefill.FILLED_MOSAIC_RUN and its
    unfilled twin, the same relative path in the first of hybrid_tilefill.UNFILLED_MOSAIC_RUNS that holds it (HMF job_inc; the
    11-13 lost-sky twins of Green and Full first). The filled map carries TPFILL, the unfilled one does not."""
    import glob
    from astropy.io import fits
    from exopipe.phase3.lib.hybrid.tilefill import TilefillRefusal
    pat = os.path.join(T.FILLED_MOSAIC_RUN, T.FILLED_MOSAIC_GLOB.format(scope=scope, robust=ROBUST_TOKEN[w]))
    rel = sorted(glob.glob(pat))
    if len(rel) != 1:
        raise TilefillRefusal(f'{scope} {w}: {len(rel)} filled Jy/sr mosaics match {pat} - REFUSING')
    filled = rel[0]
    twins = [os.path.join(r, os.path.relpath(filled, T.FILLED_MOSAIC_RUN)) for r in T.UNFILLED_MOSAIC_RUNS]
    have = [p for p in twins if os.path.exists(p)]
    if not have:
        raise TilefillRefusal(f'{scope} {w}: no unfilled twin of {os.path.basename(filled)} in {list(T.UNFILLED_MOSAIC_RUNS)} - REFUSING')
    unfilled = have[0]
    with fits.open(filled, memmap=False) as a, fits.open(unfilled, memmap=False) as b:
        if not a[1].header.get('TPFILL') or b[1].header.get('TPFILL'):
            raise TilefillRefusal(f'{scope} {w}: the filled / unfilled pair is not (TPFILL, no TPFILL) - REFUSING')
    return filled, unfilled


def v5_mosaic_pair(mosaic_out, scope, w, out_root, rows):
    """(filled, unfilled, filled md5, unfilled md5): the conformed filled Jy/sr map and its conformed unfilled twin of the done v5
    conform unit of (scope, w) in the mosaic work run mosaic_out (lib/mosaic/v5.py) - the products the archive receives, so the
    filled masters' TPMOSAIC and TPMOSMD5 name the archived mosaic as production's do (the fill stage's work files carry another name
    and header). Refuses (HG13) unless the record is done, both files and the tile list (stage/tiles.json) have its md5s now, and
    the mosaic was made from the unfilled masters this tilefill run fills: per LST, the source ('work' where this run
    holds the cell's hybrid record <out_root>/<LST>_<scope>/<w>_hybrid.json, else 'pinned') and the md5 of the beam-corrected master
    that tilefill_source gives now, as the v5 stage resolved them (rows: the pinned manifests)."""
    import json
    from exopipe.phase3.core.record import Refusal, NoInput, load_record
    from exopipe.phase3.lib.hybrid.tilefill import TilefillRefusal
    from exopipe.phase3.lib.mosaic import v5
    r = load_record(mosaic_out, v5.STEP, v5.unit_name('conform', scope, w))
    if r is None or r.get('status') != 'done':
        raise TilefillRefusal(f'{scope} {w}: no done v5 conform unit in --mosaic-out {mosaic_out} '
                              f'({"no record" if r is None else r.get("status")}) - REFUSING')
    P = v5.paths(mosaic_out, scope, w)
    prods = r['meta'].get('products') or {}
    for k in ('jysr', 'unfilled_jysr'):
        e = prods.get(k) or {}
        if not e.get('path') or not os.path.exists(e['path']) or _md5(e['path']) != e.get('md5'):
            raise TilefillRefusal(f'HG13: the conformed {k} map {e.get("path")} is missing or its md5 is not the v5 conform record\'s '
                                  f'{e.get("md5")} - REFUSING')
    if r['meta'].get('tiles_md5') is None or not os.path.exists(P['tiles']) or _md5(P['tiles']) != r['meta']['tiles_md5']:
        raise TilefillRefusal(f'HG13: {P["tiles"]} is not the tile list of the v5 conform record (its tiles_md5) - REFUSING')
    with open(P['tiles']) as f:
        tiles = {t['lst']: t for t in json.load(f)['tiles']}
    bad = [f'{lst} (no tile)' for lst in v5.LSTS if lst not in tiles]
    for lst, t in sorted(tiles.items()):
        mode = 'work' if os.path.exists(os.path.join(out_root, f'{lst}_{scope}', f'{w}_hybrid.json')) else 'pinned'
        try:
            want = tilefill_source(out_root, lst, scope, w, rows, mode, plan_only=True, need_pb=False)['src']['pbcorr'][2]
        except (TilefillRefusal, Refusal, NoInput) as e:
            bad.append(f'{lst} ({mode}: {e})')
            continue
        if t['source'] != mode or t['md5'] != want:
            bad.append(f'{lst} (this run: {mode} md5 {want}; the mosaic used {t["source"]} {t["md5"]})')
    if bad:
        raise TilefillRefusal(f'HG13: the v5 mosaic of {scope} {w} in {mosaic_out} was not made from this run\'s unfilled masters: '
                              f'{bad[:4]} - REFUSING')
    return prods['jysr']['path'], prods['unfilled_jysr']['path'], prods['jysr']['md5'], prods['unfilled_jysr']['md5']


def tilefill_field(out_root, scope, w, rows, mosaic_out=None):
    """The band fill field of (scope, w) in <out>/_tilefill_fields/ (MF band_increment; HG13: the unfilled mosaic is a pinned input,
    or with mosaic_out (--mosaic-out) the v5 work run's conformed unfilled twin at its conform record's md5; the field reproduces the
    filled mosaic),
    recorded beside it. Made once per --out: a field whose record passed, whose md5 equals
    the record's and whose mosaics still have the recorded md5s is reused (another process or an earlier run made it); otherwise it
    is made under a name of this process and renamed into place, so no process deletes a field another one reads. A refusal is
    recorded (result FAIL) and refuses every unit of the scope and weighting."""
    import json
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.hybrid import tilefill as TF
    key = (os.path.abspath(out_root), scope, w, os.path.abspath(mosaic_out) if mosaic_out else None)
    if key in _FIELDS:
        return _FIELDS[key]
    fd = os.path.join(out_root, '_tilefill_fields')
    os.makedirs(fd, exist_ok=True)
    import hashlib
    sfx = f'_v5-{hashlib.md5(os.path.realpath(mosaic_out).encode()).hexdigest()[:8]}' if mosaic_out else ''   # one file per field source
    out = os.path.join(fd, f'field_{scope}_{w}{sfx}.fits')
    rj = out[:-5] + '.json'
    TF.bind()
    old = None
    try:
        if mosaic_out:
            filled, unfilled, _fm, um = v5_mosaic_pair(mosaic_out, scope, w, out_root, rows)
        else:
            filled, unfilled = mosaic_pair(scope, w, CFG.hybrid_tilefill)
            um = _pinned(unfilled, rows)
        if os.path.exists(rj) and os.path.exists(out):
            with open(rj) as f:
                old = json.load(f)
        if (old and old.get('result') == 'PASS' and (old.get('out'), old.get('filled'), old.get('unfilled')) == (out, filled, unfilled)
                and old.get('unfilled_md5') == um and old.get('out_md5') == _md5(out) and old.get('filled_md5') == _md5(filled)):
            rec = old
        else:
            old = None
            tmp = os.path.join(fd, f'.field_{scope}_{w}{sfx}.{os.getpid()}.fits')
            _clear([tmp])
            rec = TF.band_increment(unfilled, filled, tmp)
            if rec['unfilled_md5'] != um:
                _clear([tmp])
                raise TF.TilefillRefusal(f'HG13: {unfilled} changed while it was read - REFUSING')
            os.replace(tmp, out)
            rec['out'] = out
    except (TF.TilefillRefusal, OSError, KeyError, ValueError) as e:
        old = None
        rec = dict(scope=scope, weighting=w, result='FAIL', error=f'{type(e).__name__}: {e}')
    if rec is not old:
        part = f'{rj}.{os.getpid()}.part'
        with open(part, 'w') as f:
            json.dump(rec, f, indent=1, default=str)
        os.replace(part, rj)
    _FIELDS[key] = rec
    return rec


DONE_STATES = ('done', 'skipped-valid')
REFUSED_STATES = ('refused', 'no-input')        # the run's own verdict that the unit cannot be made (a ruling may keep / quarantine)


def work_states(out_root, lsts=None, scope=None, w=None):
    """{(LST, scope, weighting): {stage: (status, reason, staged)}}: the unit records of the hybrid --out (core.record.record_path:
    <out>/_records/hybrid/<stage>.<LST>_<scope>_<w>.json, stage in STAGES), narrowed by name (lsts, scope, w) before any is read;
    staged = the record was staged from production (its staged_from key). A record that cannot be read is ('unreadable', why,
    False). {} when the run holds no records."""
    import json
    from exopipe.phase3.core.record import record_path
    d = os.path.dirname(record_path(out_root, 'hybrid', 'x'))
    got = {}
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else ():
        stage, _dot, key = name[:-len('.json')].partition('.')
        if not name.endswith('.json') or stage not in STAGES or key.count('_') < 2:
            continue
        lst, rest = key.split('_', 1)
        sc, wt = rest.rsplit('_', 1)
        if lsts is not None and lst not in lsts or scope is not None and sc != scope or w is not None and wt != w:
            continue
        try:
            with open(os.path.join(d, name)) as f:
                r = json.load(f)
            st = (str(r.get('status')), str(r.get('reason') or ''), 'staged_from' in r)
        except (OSError, ValueError) as e:
            st = ('unreadable', f'{type(e).__name__}: {e}', False)
        got.setdefault((lst, sc, wt), {})[stage] = st
    return got


def unit_state(stages, through='tilefill'):
    """(state, stage, status, reason) of one unit from its work_states entry, over its stage records up to `through`: 'tilefill'
    (the archive push: the filled masters) or 'hybrid' (the hybrid or full stage's unfilled masters: the v5 mosaic, made before
    tilefill, and tilefill's own work source). States, in this order:
      input       only a stack record, staged from production and done (a member input such as the 22 high-band Robust -0.75
                  Taper residual stacks staged into the Q1 run; not a unit the run builds);
      refused     a record refused or no-input (the first in STAGES order), even before a done later record;
      unfinished  a record neither done nor refused (running, error, interrupted, unreadable), or the last stage not yet run
                  (a stack made by the run whose recon has not started is unfinished, never an input);
      done        every record done (or skipped-valid) and the last stage (tilefill; hybrid or full) among them;
      none        no record up to `through` (a pinned tilefill unit seen by the v5 mosaic)."""
    upto = STAGES[:STAGES.index('full' if through == 'hybrid' else through) + 1]
    st = [(s, stages[s]) for s in upto if s in stages]
    if not st:
        return 'none', '', '', ''
    if set(stages) == {'stack'} and stages['stack'][2] and stages['stack'][0] in DONE_STATES:
        return 'input', 'stack', stages['stack'][0], stages['stack'][1]
    for s, (status, reason, _g) in st:
        if status in REFUSED_STATES:
            return 'refused', s, status, reason
    for s, (status, reason, _g) in st:
        if status not in DONE_STATES:
            return 'unfinished', s, status, reason
    ends = ('tilefill',) if through == 'tilefill' else ('hybrid', 'full')
    if any(s in ends for s, _x in st):
        return 'done', st[-1][0], st[-1][1][0], ''
    return 'unfinished', st[-1][0], st[-1][1][0], f'done; the stages after it, through {through}, have not run'


def _installed(lst, scope, w, root=None):
    """{'plain', 'pbcorr', 'cleanres', 'recon', 'recon_pbcorr': [paths]}: the installed hybrid products of the cell and weighting in
    root/<LST>/<scope>/ (default hybrid.ARCHIVE_HYBRID; the archive push passes its --archive-root's copy): filled master, filled
    beam-corrected sibling, CLEAN residual, NoTaper reconstruction and its beam-corrected split, whatever their night span - the files
    a new product of the kind replaces."""
    import glob
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core.products import parse_name
    a = os.path.join(root or CFG.hybrid.ARCHIVE_HYBRID, lst, scope)
    found = {'plain': [], 'pbcorr': [], 'cleanres': [], 'recon': [], 'recon_pbcorr': []}
    for p in sorted(glob.glob(os.path.join(a, f'{scope}_I_deep_Taper_{ROBUST_TOKEN[w]}_*_LST{lst}_*.fits'))):
        proc = set(parse_name(os.path.basename(p)).get('proc') or ())
        if 'lsrecon' in proc and 'cleanres' in proc:
            found['cleanres'].append(p)
        elif 'lsfill' in proc:
            found['pbcorr' if 'pbcorr' in proc else 'plain'].append(p)
    for p in sorted(glob.glob(os.path.join(a, f'{scope}_I_deep_NoTaper_{ROBUST_TOKEN[w]}_*_LST{lst}_*.fits'))):
        proc = set(parse_name(os.path.basename(p)).get('proc') or ())
        if 'lsrecon' in proc and not proc & {'lsfill', 'cleanres'}:
            found['recon_pbcorr' if 'pbcorr' in proc else 'recon'].append(p)
    return found


def _production_record(lst, scope, w, tpsrc):
    """(path, record): the production hybrid record (<run>/<LST>_<scope>/<w>_hybrid.json of hybrid.LSRECON_RUNS) whose products' md5s
    are the installed filled masters' TPSRCMD5 (plain and beam-corrected) - the record of their unfilled sources, with their nights.
    Several records naming the same product file and nights are one (the first in run-key order is returned); none, or records
    that disagree, refuse."""
    import json
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.hybrid.tilefill import TilefillRefusal
    hits = []
    for _k, run in sorted(CFG.hybrid.LSRECON_RUNS.items()):
        p = os.path.join(run, f'{lst}_{scope}', f'{w}_hybrid.json')
        if not os.path.exists(p):
            continue
        with open(p) as f:
            r = json.load(f)
        if ((r.get('product') or {}).get('md5'), (r.get('product_pbcorr') or {}).get('md5')) == (tpsrc['plain'], tpsrc['pbcorr']):
            hits.append((p, r))
    same = {(r['product'].get('path'), tuple(sorted(n['date'] for n in r['span']['nights']))) for _p, r in hits}
    if len(same) != 1:       # a run that carries another run's record (Run_FULL holds R1's) names the same product and nights
        raise TilefillRefusal(f'{lst} {scope} {w}: {len(hits)} production hybrid records in hybrid.LSRECON_RUNS name the unfilled sources '
                              f'{tpsrc} ({len(same)} distinct products and night lists): the installed masters\' nights are unknown - '
                              f'REFUSING')
    return hits[0]


def tilefill_source(out_root, lst, scope, w, rows, mode='work', plan_only=False, need_pb=True):
    """The unfilled masters a tilefill unit fills: dict(source, src = {kind: (path, product name, md5 it must have)}, pb, pb_how,
    nights, reads, replaces = {kind: the installed file a product of the kind replaces, or None}).
    'work' (spec D1): the hybrid stage's products in --out (record <w>_hybrid.json; its v2 unit record must be done; each file's md5
    must equal the recorded one); the diffuse PB of a subband or wideband computed as production's archive step wrote it
    (lib/hybrid/master.diffuse_pb_map, rounded through float32), Full's read from the file of the hybrid_full.PBDIFFUSE_WEIGHTING unit
    (where that unit refused and hybrid_unit.REFUSED_INSTALLED keeps it: the installed map, the kept cell's).
    'work' on a unit with no hybrid record in --out that hybrid_unit.REFUSED_INSTALLED keeps (refused in the work run, its installed
    products kept by ruling): as 'pinned' - its filled masters follow the band field, as the v5 mosaic takes its pinned tile
    (lib/mosaic/v5.resolve_tiles) and as v5_mosaic_pair checks it; without the refill the archived mosaic and that cell's fill differ
    (archive push --class mosaic_v5 refuses them, HG20).
    'pinned' (a unit this run did not rebuild, whose band field changed): the installed filled masters in hybrid.ARCHIVE_HYBRID, whose
    TPSRCMD5 names their unfilled sources in the pinned manifests (HG13), the installed diffuse PB map, and the nights of the
    production hybrid record of those sources; the CLEAN residual is not re-made.
    Both: a night now excluded in data_decisions.csv (deep read, made now) for the subband that holds it refuses (excluded_in_unit;
    no fill re-admits an excluded night).
    plan_only (the dry run): no md5 is re-computed and no PB map is read. need_pb=False (the v5 mosaic stage, which reads only the
    beam-corrected master): the diffuse PB map is neither looked up nor read (pb, pb_how None). mode = --tilefill-source. Raises TilefillRefusal,
    core.record.Refusal, or core.record.NoInput (work: no hybrid record in --out)."""
    import json
    import numpy as np
    from astropy.io import fits
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import exclusions
    from exopipe.phase3.core.record import NoInput, Refusal
    from exopipe.phase3.lib.hybrid import master as M
    from exopipe.phase3.lib.hybrid.tilefill import TilefillRefusal
    wd = os.path.join(out_root, f'{lst}_{scope}')
    hj = os.path.join(wd, f'{w}_hybrid.json')
    inst = _installed(lst, scope, w)
    many = {k: v for k, v in inst.items() if len(v) > 1 and k in ('plain', 'pbcorr', 'cleanres')}
    if many:
        raise TilefillRefusal(f'{lst} {scope} {w}: more than one installed hybrid master of a kind {many} - REFUSING')
    replaces = {k: (v[0] if v else None) for k, v in inst.items()}
    pb = None
    if mode == 'work' and not os.path.exists(hj) and CFG.hybrid_unit.REFUSED_INSTALLED.get((lst, scope, w)) == 'keep':
        mode = 'pinned'                 # a cell kept by ruling: re-filled from its installed masters' own unfilled sources (docstring)
    if mode == 'work':
        if not os.path.exists(hj):
            raise NoInput(f'no hybrid record {hj} (its stage refused or was not run; --tilefill-source pinned re-fills the installed '
                          f'masters)')
        why = _upstream_problem(out_root, f'{"full" if scope == "Full" else "hybrid"}.{lst}_{scope}_{w}', [hj])
        if why:
            raise Refusal(f'its hybrid may not be used: {why} - REFUSED')
        ws = work_states(out_root, lsts=(lst,), scope=scope, w=w).get((lst, scope, w))
        st = unit_state(ws, through='hybrid') if ws else ('none',)
        if st[0] not in ('done', 'none'):     # a stage before the hybrid rerun after it and refused / unfinished: a stale hybrid
            raise Refusal(f'its hybrid may not be used: {st[0]} in the work run at stage {st[1]} ({st[2]}: {st[3][:200]}) - REFUSED')
        with open(hj) as f:
            rec = json.load(f)
        reads, src = [hj], {}
        for kind, k in (('plain', 'product'), ('pbcorr', 'product_pbcorr'), ('cleanres', 'product_cleanres')):
            p = rec.get(k)
            if not p:
                continue
            if not plan_only and _md5(p['path']) != p['md5']:
                raise TilefillRefusal(f'{p["path"]}: md5 differs from the hybrid record {hj} - REFUSING')
            src[kind] = (p['path'], p['prodname'], p['md5'])
        if 'plain' not in src or 'pbcorr' not in src:
            raise TilefillRefusal(f'{hj}: no hybrid master or beam-corrected sibling - REFUSING')
        nights = sorted(n['date'] for n in rec['span']['nights'])
        if not need_pb:
            how = None
        elif scope == 'Full':
            pw = CFG.hybrid_full.PBDIFFUSE_WEIGHTING
            pbf = os.path.join(wd, f'PBdiffuse_deep_{scope}.fits')
            why = _upstream_problem(out_root, f'full.{lst}_{scope}_{pw}', [pbf])
            pws = work_states(out_root, lsts=(lst,), scope=scope, w=pw).get((lst, scope, pw)) if why and w != pw else None
            if (pws and unit_state(pws, through='hybrid')[0] == 'refused'
                    and CFG.hybrid_unit.REFUSED_INSTALLED.get((lst, scope, pw)) == 'keep'):
                # the cell's map-writing unit refused and is kept by ruling: its installed products stay the cell's, the diffuse map
                # among them (archive push --class hybrid pushes the map with the writer's unit only). The map's values depend on
                # neither the weighting nor the rebuild (the Q1 rebuild's 23 written Full maps equal the archived ones bit for bit,
                # 2026-11-19), so this weighting's unit divides by the installed map.
                pbf, why = os.path.join(CFG.hybrid.ARCHIVE_HYBRID, lst, scope, f'PBdiffuse_deep_{scope}.fits'), None
            if why or not os.path.exists(pbf):
                raise Refusal(f'the diffuse PB map {pbf} may not be used: {why or "missing"} - REFUSED')
            reads.append(pbf)
            how = pbf
            if not plan_only:
                with fits.open(pbf, memmap=False) as h:
                    pb = np.squeeze(h[0].data).astype(np.float64)
        else:
            how = 'in memory (lib/hybrid/master.diffuse_pb_map, rounded through float32)'
            if not plan_only:
                M.bind()
                a, how0 = M.diffuse_pb_map(out_root, lst, scope, w)
                pb = np.asarray(a).astype(np.float32).astype(np.float64)   # as the archived float32 map that production's fill read
                how = f'in memory: {how0} (rounded through float32)'
        source = 'work'
    else:
        if os.path.exists(hj):
            raise Refusal(f'{lst} {scope} {w} was rebuilt in --out ({hj}): its fill is --tilefill-source work - REFUSED')
        if not inst['plain'] or not inst['pbcorr']:
            raise TilefillRefusal(f'{lst} {scope} {w}: installed filled masters {inst} - not one each; REFUSING')
        src, tpsrc, reads = {}, {}, []
        for kind in ('plain', 'pbcorr'):
            p = inst[kind][0]
            with fits.open(p, memmap=False) as h:
                hh = h[0].header
            reads.append(p)
            if hh.get('PRODNAME') != os.path.basename(p) or not hh.get('TPFILL') or not hh.get('TPSRCMD5'):
                raise TilefillRefusal(f'{p}: PRODNAME, TPFILL or TPSRCMD5 missing or wrong - REFUSING')
            hit = [q for q, m in rows.items() if m == hh['TPSRCMD5']]
            if len(hit) != 1:
                raise TilefillRefusal(f'HG13: {p}: TPSRCMD5 {hh["TPSRCMD5"]} names {len(hit)} pinned inputs - REFUSING')
            if not plan_only:
                _pinned(hit[0], rows)
            src[kind] = (hit[0], os.path.basename(p), rows[hit[0]])
            tpsrc[kind] = hh['TPSRCMD5']
        rp, prec = _production_record(lst, scope, w, tpsrc)
        reads.append(rp)
        nights = sorted(n['date'] for n in prec['span']['nights'])
        pbf = os.path.join(CFG.hybrid.ARCHIVE_HYBRID, lst, scope, f'PBdiffuse_deep_{scope}.fits')
        if need_pb and not os.path.exists(pbf):
            raise TilefillRefusal(f'{pbf}: the installed diffuse PB map is missing - REFUSING')
        if need_pb:
            reads.append(pbf)
        how = pbf if need_pb else None
        if need_pb and not plan_only:
            with fits.open(pbf, memmap=False) as h:
                pb = np.squeeze(h[0].data).astype(np.float64)
        source = 'pinned'
    excl = exclusions.for_tier('deep', fresh=True)
    if getattr(excl, 'path', None):
        reads.append(excl.path)
    bad, sreads = excluded_in_unit(excl, out_root if mode == 'work' else os.path.dirname(os.path.dirname(rp)), lst, scope, w, nights)
    reads += sreads
    if bad:
        raise TilefillRefusal(f'{lst} {scope} {w}: the unfilled masters hold nights now excluded in data_decisions.csv {bad} - rebuild '
                              f'the hybrid first - REFUSING')
    return dict(source=source, src=src, pb=pb, pb_how=how, nights=nights, reads=reads, replaces=replaces,
                nights_from=hj if mode == 'work' else rp)


def excluded_in_unit(excl, rundir, lst, scope, w, nights):
    """({subband: [dates]}, [records read]): the nights a hybrid unit's data hold that the exclusions (excl) now exclude for the
    subband that holds them. A subband unit: its nights (the hybrid record's), under the subband's rule. A wideband or Full unit
    combines its members' reconstructions, each made from its own subband's admitted nights (a night excluded in one member is
    absent from that member only; production 11-13 lost-sky update): each member's NoTaper stack record <rundir>/<LST>_<member>/
    <w>_stack.json (its nights less the dirty nights dropped, as the stack unit admits them) under the member's rule; a member
    without one refuses (its nights are unknown). The Taper master is the master builder's (its own exclusion gate)."""
    import json
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.hybrid.tilefill import TilefillRefusal
    WB = CFG.hybrid.WIDEBAND_MEMBERS
    if scope not in WB and scope != 'Full':
        e = excl.excluded_constituents(lst, scope, nights)
        return ({scope: e} if e else {}), []
    members = sorted({b for k in WB for b in WB[k]}) if scope == 'Full' else list(WB[scope])
    bad, reads = {}, []
    for b in members:
        sp = os.path.join(rundir, f'{lst}_{b}', f'{w}_stack.json')
        if not os.path.exists(sp):
            raise TilefillRefusal(f'{sp}: no stack record of the member {b} - its nights are unknown; REFUSING')
        with open(sp) as f:
            sr = json.load(f)
        reads.append(sp)
        e = excl.excluded_constituents(lst, b, sorted(set(n['date'] for n in sr['nights']) - set(sr.get('dirty_nights_dropped') or ())))
        if e:
            bad[b] = e
    return bad, reads


def _tilefill_dry_plan(mode):
    def plan(out_root, lst, scope, w):
        """The dry run's plan of a tilefill unit (reads the pinned manifests, the hybrid record or the installed filled masters'
        headers; no md5 and no PB map)."""
        from exopipe.phase3.config import CFG
        s = tilefill_source(out_root, lst, scope, w, _manifest_rows(CFG.hybrid_tilefill), mode, plan_only=True)
        return tilefill_plan(out_root, lst, scope, w, s)
    return plan


def tilefill_plan(out_root, lst, scope, w, s):
    """[(path, product class)] a tilefill unit owes for source s (tilefill_source): in <--out>/<LST>_<scope>/tilefill/, the filled
    master and beam-corrected sibling under their product names, the renamed CLEAN residual (work source, where the hybrid stage made
    one), and the record <w>_tilefill.json beside the hybrid record."""
    from exopipe.phase3.lib.hybrid.tilefill import cleanres_name
    wd = os.path.join(out_root, f'{lst}_{scope}')
    td = os.path.join(wd, 'tilefill')
    out = [(os.path.join(td, s['src'][k][1]), 'hybrid_filled_master') for k in ('plain', 'pbcorr')]
    if s['source'] == 'work' and 'cleanres' in s['src']:
        out.append((os.path.join(td, cleanres_name(s['src']['cleanres'][1])), 'hybrid_cleanres_renamed'))
    return out + [(os.path.join(wd, f'{w}_tilefill.json'), 'hybrid_tilefill_record')]


def tilefill_unit(u, lst, scope, w, out_root, mode='work', mosaic_out=None):
    """Stage tilefill (spec 2.14) of one unit: production's band field + fill_master_unit + rename_cleanres (MF 211, 306, 374) and
    the ruled caveat (HMF 124-143). The band field of the scope and weighting is made once per --out; the unfilled masters come from
    tilefill_source. A gate, a failed test or an unreadable input refuses the unit (the run goes on) and removes what it wrote. The
    unit's inputs are every file read; its admitted nights are the unfilled masters' (checked against the exclusions now)."""
    import json
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.hybrid import master as M
    from exopipe.phase3.lib.hybrid import recon as REC
    from exopipe.phase3.lib.hybrid import tilefill as TF
    T = CFG.hybrid_tilefill
    TF.READS.clear(); M.READS.clear()
    expected = (TF.TilefillRefusal, REC.ReconRefusal, OSError, KeyError, ValueError)
    for _k, (path, _sha) in sorted(T.UNFILLED_INPUTS.items()):
        u.add_input(path)
    rows = _manifest_rows(T)
    try:
        s = tilefill_source(out_root, lst, scope, w, rows, mode)
        REC.bind()
        taper = REC.installed_master(lst, scope, w, taper=True)[0]
    except expected as e:
        u.refuse(f'{e}' if isinstance(e, (TF.TilefillRefusal, REC.ReconRefusal)) else f'{type(e).__name__}: {e} - REFUSED')
    for p in s['reads'] + [taper]:
        u.add_input(p)
    field = tilefill_field(out_root, scope, w, rows, mosaic_out)
    if field.get('result') != 'PASS':
        u.refuse(f'the band field of {scope} {w} refused: {field.get("error")}')
    for p in (field['out'], field['filled'], field['unfilled']):
        u.add_input(p)
    plan = tilefill_plan(out_root, lst, scope, w, s)
    for p, c in plan:
        u.plan(p, c)
    own = [p for p, _c in plan]
    _clear(own)
    os.makedirs(os.path.dirname(own[0]), exist_ok=True)
    TF.bind()
    outs = {k: p for k, (p, c) in zip(('plain', 'pbcorr'), plan) if c == 'hybrid_filled_master'}
    cres = [p for p, c in plan if c == 'hybrid_cleanres_renamed']
    try:
        rec = TF.fill_master_unit(s['src']['plain'][0], s['src']['pbcorr'][0], s['pb'], s['pb_how'], field['out'], outs, lst, scope, w,
                                  taper_master=taper, expect_md5={k: s['src'][k][2] for k in ('plain', 'pbcorr')})
        if rec['increment_md5'] != field['out_md5']:
            raise TF.TilefillRefusal(f'the band field {field["out"]} changed while this unit read it (md5 {rec["increment_md5"]}, '
                                     f'recorded {field["out_md5"]}) - REFUSING')
        rec['source'] = s['source']; rec['field'] = field
        rec['nights'] = s['nights']; rec['nights_from'] = s['nights_from']      # the archive push re-checks them (PR-5b)
        if cres:
            rec['cleanres'] = TF.rename_cleanres(s['src']['cleanres'][0], cres[0], taper_master=taper)
            if rec['cleanres']['src_md5'] != s['src']['cleanres'][2]:
                raise TF.TilefillRefusal(f'{s["src"]["cleanres"][0]}: md5 differs from the hybrid record when read - REFUSING')
        if s['source'] == 'work' and w == CFG.hybrid_full.PBDIFFUSE_WEIGHTING:
            # the cell's diffuse PB map, pushed beside a rebuilt unit's products by `archive push --class hybrid` (production wrote it at
            # archive time, NL 1644-1649), written and pushed by the hybrid_full.PBDIFFUSE_WEIGHTING unit only (as the full stage), so
            # its header never depends on which weighting ran first. Full's is the full stage's (not this unit's to remove).
            from exopipe.phase3.lib.install.steps_common import md5 as _md5
            M.bind()
            pbd = M.write_diffuse_pb(out_root, lst, scope, w)
            if scope != 'Full':
                own.insert(0, pbd)                                   # own[-1] stays the record
            rec['pbdiffuse_map'] = dict(path=pbd, md5=_md5(pbd))       # ('pbdiffuse' is the fill's PB-source description)
        rp = own[-1]
        with open(rp + '.part', 'w') as f:
            json.dump(rec, f, indent=1, default=str)
        os.replace(rp + '.part', rp)
    except expected as e:
        _clear(own)
        u.refuse(f'{e}' if isinstance(e, (TF.TilefillRefusal, REC.ReconRefusal)) else f'{type(e).__name__}: {e} - REFUSED')
    except BaseException:
        _clear(own)
        raise
    finally:
        outs_abs = {os.path.abspath(p) for p in own}
        for p in sorted(TF.READS | M.READS):
            if os.path.abspath(p) not in outs_abs and os.path.exists(p):
                u.add_input(p)
    bad_kinds = [k for k in ('pbcorr', 'plain', 'cleanres') if k in rec and rec[k].get('result') != 'PASS']
    if bad_kinds:
        _clear(own)
        u.refuse('production test(s) FAILED: ' + '; '.join(
            f'{k}: ' + json.dumps({x: rec[k].get(x) for x in ('identity_rel', 'blanking_unchanged', 'header_valid', 'header_errors', 'compact',
                                                             'roundtrip', 'caveat', 'data_identical', 'prodname_ok') if x in rec[k]},
                                  default=str)[:400] for k in bad_kinds) + ' - REFUSED')
    n0 = len(u.failed_tests())
    for p, c in plan:
        if c == 'hybrid_tilefill_record':
            u.add_output(p, c, lst=lst, scope=scope, weighting=w)
            continue
        kind = 'cleanres' if c == 'hybrid_cleanres_renamed' else 'pbcorr' if p == outs['pbcorr'] else 'plain'
        u.add_output(p, c, scope=scope, weighting=w, kind=kind, src=s['src'][kind][0], replaces=s['replaces'].get(kind))
    bad = u.failed_tests()[n0:]
    if bad:
        _clear(own)
        u.refuse('content test(s) FAILED: ' + '; '.join(f"{t['name']} {t.get('detail', '')}"[:200] for t in bad))
    if rec.get('pbdiffuse_map'):
        pbd = rec['pbdiffuse_map']['path']
        n1 = len(u.failed_tests())
        u.add_output(pbd, 'hybrid_pbdiffuse', taper=taper)
        bad = u.failed_tests()[n1:]
        if bad:
            _clear(own)
            u.refuse(f'{pbd}: content test(s) FAILED: ' + '; '.join(f"{t['name']} {t.get('detail', '')}"[:200] for t in bad))
    cav = [k for k in ('pbcorr', 'plain') if rec[k].get('caveat')]
    if cav:
        u.note(f'ruled caveat (TPCAVEAT) on {cav}: {TF.TPCAVEAT_TEXT}')
    u.note(f"source {s['source']}; field s {field['s']:.5f}, donor {field['donor_mhz']} MHz")
    u.set_admitted(s['nights'])
