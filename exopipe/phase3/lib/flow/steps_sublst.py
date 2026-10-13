"""`exopipe3 flow --tier sublst`: the sub-LST second alignment (x2 refine, arm B) - PER HOUR (one night: --epochs
<LST>_<date>, chain 'epoch') or per LST (chain 'sublst-lst': masters sub-LST -> flow -> frames -> search -> catalog).

Items: --lsts L (every subband), --cells L_<subband> (deep grammar: one subband over every night), --cells / --epochs
L_<date>[_<subband>] (that night's residuals only; the leave-one-out Masters still use every night, the module's --nights).
Units:
  refine_<LST>_<sb>[_<night>]  clv15/phase3_sublst_refine.py IN PLACE through core.legacy.run under site PYTHON['default'],
               exactly the command line of sublst_refine_pass.sh:22-23 (--root <SUBLST_ROOT>/<LST> --out <out>/<LST>_x2
               --work <work>/sublst_x2_work/gpu<g> --lst --subband [--nights] [--windows]) with CUDA_VISIBLE_DEVICES=<g>,
               DEEP_XLA_MEMFRAC=0.5, P3_FLOW_LOCK=<the GPU's lock>, REFINE_FFT_WORKERS; the module shells align_images.py in
               py310_orca itself (flock -w 86400). Then content test (b) is recomputed INDEPENDENTLY of the module (frozen
               weights + membership vetoes + the stack listing): R1 every residual allowed with its LOONCOAD, R2 every expected
               residual written, except EXACTLY the approved (night, window) rows (G-C7, structural - never from log text),
               and the cards (G-C9).
  check_<LST>_<sb>[_<night>]   the same verdict on residuals already on disk (no GPU); runs only when refine did not run in
               the same invocation.
"""
import hashlib
import os

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.flow import steps_common as C

STAGES = ('refine', 'check')
REFINE_SCRIPT = ('clv15', 'phase3_sublst_refine.py')   # (legacy root, file); tests replace it with a stand-in


def add_arguments(p):
    g = p.add_argument_group('tier sublst (x2 refine to the subtracted leave-one-out window Master; per hour / per LST)')
    g.add_argument('--sublst-windows', dest='sublst_windows', metavar='LIST',
                   help='windows to refine, comma list (default flow_refine.WINDOWS = all six; the module\'s --windows)')
    g.add_argument('--sublst-stack-root', dest='sublst_stack_root', metavar='DIR',
                   help='root holding <LST>/ window stacks and <LST>_masters/ (default paths.SUBLST_ROOT; parity mirrors)')


def default_out(args):
    return C.cfg().paths.SUBLST_ROOT


def _stack_base(args):
    return args.sublst_stack_root or C.cfg().paths.SUBLST_ROOT


# ---------------------------------------------------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------------------------------------------------

def items(args):
    """[(lst, sb, nights or None)] - nights restrict the residuals written (LOO still uses every night)."""
    from exopipe.phase3.core import cells
    allsb = list(C.cfg().instrument.ALL_SUBBANDS)
    want = {}

    def add(lst, sb, night):
        k = (lst, sb)
        if night is None:
            want[k] = None
        elif k not in want or want[k] is not None:
            want.setdefault(k, set()).add(night)

    for t in (cells.expand(args.cells, kind='item') if args.cells else []):
        try:
            lst, scope = cells.parse_cell(t)
            if cells.is_band(scope):
                raise Refusal(f'{t}: band scope - the x2 refine is per subband (wideband windows are built from the x2 '
                              f'subband residuals by `frames --tier sublst`)')
            add(lst, scope, None)
            continue
        except ValueError:
            pass
        lst, date, sb = C.parse_snapshot_item(t)
        for s in ([sb] if sb else allsb):
            add(lst, s, date)
    for e in (cells.expand_epochs(args.epochs) if args.epochs else []):
        lst, date = cells.parse_epoch(e)
        for s in allsb:
            add(lst, s, date)
    for lst in (cells.expand(args.lsts, universe=cells.LSTS, kind='LST') if args.lsts else []):
        for s in allsb:
            add(lst, s, None)
    if args.scopes:
        keep = set(cells.expand(args.scopes, universe=allsb, kind='subband'))
        want = {k: v for k, v in want.items() if k[1] in keep}
    if args.nights:
        ns = set(cells.expand(args.nights, kind='night'))
        want = {k: (ns if v is None else (v & ns)) for k, v in want.items()}
        want = {k: v for k, v in want.items() if v}
    if args.exclude_cells:
        drop = set(cells.expand(args.exclude_cells, kind='item'))
        want = {k: v for k, v in want.items() if f'{k[0]}_{k[1]}' not in drop}
    return [(l, s, (sorted(v) if v else None)) for (l, s), v in want.items()]


def unit_name(stage, lst, sb, nights):
    base = f'{stage}_{lst}_{sb}'
    if not nights:
        return base
    if len(nights) == 1:
        return f'{base}_{nights[0]}'
    return f'{base}_{len(nights)}n_{hashlib.sha1(",".join(nights).encode()).hexdigest()[:8]}'


# ---------------------------------------------------------------------------------------------------------------------
# gates and the independent verdict
# ---------------------------------------------------------------------------------------------------------------------

def drift_guard(note):
    """WEIGHTS_CSV, the veto pin and the Jupiter radius against the in-place clv15 modules (read in-process, no write)."""
    import phase3_jupiter_mask as JM          # clv15 (site LEGACY_CODE_ROOTS on sys.path, core.env.legacy)
    import phase3_sublst_loo as LOO
    import phase3_tier_rules as TR
    C.drift_guard([('flow_refine.WEIGHTS_CSV', LOO.WEIGHTS_CSV, None),
                   ('flow_refine.MEMBERSHIP_VETO_PIN', (TR.MEMBERSHIP_VETO_PIN, TR.MEMBERSHIP_VETO_PIN_MD5), None),
                   ('flow_refine.JUPITER_MASK_RADIUS_BEAMS', JM.RADIUS_BEAMS, None)], note)
    # the module's rule literals (refine.py:129, 165-166, 203, 221, 232-235): documented in flow_refine, never passed
    C.refuse_unreachable(tuple(f'flow_refine.{k}' for k in ('MIN_NIGHTS_PER_WINDOW', 'MIN_WEIGHTED_NIGHTS_PER_WINDOW',
                                                             'MIN_LOO_WEIGHTED_NIGHTS', 'X2_VALID_FOOTPRINT', 'ALIGNER_OK_RC',
                                                             'DURATION_FALLBACK_S')),
                         'phase3_sublst_refine.py')
    f = C.cfg().flow
    if (f.ALPHA, f.GAMMA, f.SCALE_FACTOR, f.PREPROCESS_WEIGHT) != (1.3, 150.0, 0.7, 1.5):
        raise Refusal('flow.ALPHA/GAMMA/SCALE_FACTOR/PREPROCESS_WEIGHT overridden: the sub-LST refine module shells '
                      'align_images.py with its own defaults and cannot take them during the refactor window')
    pin, md5 = C.cfg().flow_refine.MEMBERSHIP_VETO_PIN
    if not os.path.exists(pin) or hashlib.md5(open(pin, 'rb').read()).hexdigest() != md5:
        raise Refusal(f'membership-veto pin {pin} missing or altered (md5 {md5} expected; ruling B 2026-10-08)')


def excluded_stacks(model, lst, sb):
    from exopipe.phase3.core import exclusions
    E = exclusions.load()
    nights = sorted({n for nd in model['stacks'].values() for n in nd})
    return E.path, [n for n in nights if E.is_excluded(lst, sb, n)]


def verdict(u, run, args, lst, sb, nights, rc_child=None):
    """Content test (b) for (LST, subband[, nights]) on <out>/<LST>_x2, independently of the module; plans every expected
    residual, adds every residual on disk as an output (R1 + cards per file), judges the missing ones (G-C7)."""
    from exopipe.phase3.lib.flow import tests as T
    fr = C.cfg().flow_refine
    x2 = os.path.join(run.out_root, f'{lst}_x2')
    base = _stack_base(args)
    model = T.sublst_model(lst, sb, stack_root=os.path.join(base, lst), refresh=True)
    nset = set(nights) if nights else None
    expected = sorted(k for k in model['expected'] if nset is None or k[2] in nset)
    for S, k, n in expected:
        u.plan(T.residual_path(x2, lst, n, sb, S, k), 'sublst_x2_residual')
    have = T.residuals_on_disk(x2, lst, sb, nset)
    mroot = os.path.join(base, fr.MASTERS_DIR.format(lst=lst))
    for key, p in sorted(have.items()):
        u.add_output(p, 'sublst_x2_residual', masters_root=mroot, stack_root=os.path.join(base, lst))
    missing, appr = T.r2_missing(x2, lst, sb, nset, model)
    ok, waived, why = T.judge_missing(missing, appr)
    u.meta['verdict'] = {'expected': len(expected), 'on_disk': len(have), 'missing': len(missing),
                         'approved_pairs': sorted(f'{n} w{k}' for n, k in appr), 'refine_rc': rc_child}
    if missing and ok:
        for (S, k, n), p in missing.items():
            u.refuse_item(p, f'approved alignment refusal ({n} w{k}): {appr[(n, k)][:300]}', gate='G-C7')
        u.check('G-C7.approved_refusals', True, value=len(missing),
                detail=f'{why}; ruling rows of {fr.APPROVED_REFUSALS_CSV}')
    elif not ok:
        raise Refusal(f'G-C7: {why}')
    if not expected and not have:
        raise NoInput(f'no residual owed for {lst} {sb}' + (f' nights {nights}' if nights else '') + ' (every window refused '
                      f'by the >= 2 weighted nights rules, or no stack)')
    return model


# ---------------------------------------------------------------------------------------------------------------------
# refine
# ---------------------------------------------------------------------------------------------------------------------

def refine_unit(u, args, run, lst, sb, nights):
    from exopipe.phase3.core import fitsio, legacy
    from exopipe.phase3.lib.flow import tests as T
    c = C.cfg()
    fr = c.flow_refine
    u.meta.update(lst=lst, subband=sb, nights=nights)
    drift_guard(u.note)
    wp = fr.WEIGHTS_CSV.format(lst=lst, sb=sb)
    if not os.path.exists(wp):
        raise Refusal(f'G-C1: no night-weight record {wp} (ruling B 2026-10-08)')
    u.add_input(wp, extra={'role': 'frozen deep co-add night weights'})
    u.add_input(fr.MEMBERSHIP_VETO_PIN[0], extra={'role': 'membership-veto pin'})
    u.add_input(fr.APPROVED_REFUSALS_CSV, extra={'role': 'approved alignment refusals'})
    base = _stack_base(args)
    model = T.sublst_model(lst, sb, stack_root=os.path.join(base, lst), refresh=True)
    stacks = sorted(p for nd in model['stacks'].values() for p in nd.values())
    if not stacks:
        raise NoInput(f'no window stack of {lst} {sb} under {os.path.join(base, lst)}')
    C.add_listing_input(u, os.path.join(run.out_root, '_records', 'flow', 'inputs', f'{u.name}_stacks.json'), stacks,
                        'window stacks')
    dpath, excl = excluded_stacks(model, lst, sb)
    u.add_input(dpath, extra={'role': 'data_decisions'})
    nset = set(nights) if nights else None
    all_nights = sorted({n for nd in model['stacks'].values() for n in nd})
    u.set_admitted([n for n in all_nights if n not in excl])
    if excl:
        msg = f'G-C12: window stacks of nights excluded in data_decisions.csv: {excl}'
        if fr.EXCLUDED_STACK == 'refuse':
            raise Refusal(msg + ' (the stacks predate the exclusion; rebuild them with masters --tier sublst)')
        u.note(msg + ' (flow_refine.EXCLUDED_STACK=record: production behaviour)')
    # ---- G-C11: a GPU and the lock production holds on it (site GPU_IDS / GPU_LOCKS; sublst_refine_pass.sh:22 passes the
    #      per-GPU lock as P3_FLOW_LOCK, which the module flocks around align_images). Checked BEFORE anything is moved. ----
    gpus = C.parse_gpus(args.gpus) or C.site_gpu_ids()
    if not gpus:
        raise Refusal(f'G-C11: no GPU id for {C.host()} (--gpus, or GPU_IDS in the site file)')
    g = gpus[0]
    locks = C.required_gpu_locks(g, 'G-C11')
    lock = locks[0]
    if len(locks) > 1:
        u.note(f'site names {len(locks)} locks for GPU {g}; the module takes one (P3_FLOW_LOCK={lock})')
    # ---- G-C5: earlier residuals of this (LST, subband[, nights]) -------------------------------------------------------
    x2 = os.path.join(run.out_root, f'{lst}_x2')
    old = T.residuals_on_disk(x2, lst, sb, nset)
    log = os.path.join(x2, 'logs', f'{lst}_{sb}.log')
    if old:
        if not args.fresh:
            raise Refusal(f'G-C5: {len(old)} residual(s) of an earlier run of {lst} {sb} under {x2} - pass --fresh '
                          f'--supersede TAG to move them aside first (as sublst_resume_from_x2.sh:54-56 does by hand)')
        moves = C.supersede(sorted(old.values()), run.out_root, run.out_root, args.supersede, f'flow refine {u.name} --fresh')
        u.note(f'--fresh: {len(moves)} earlier residuals moved aside under {run.out_root}/_superseded/{args.supersede}/')
    if os.path.exists(log):
        # one run per log, as production's `> $L` (sublst_refine_pass.sh:23): the frames S-G8 gate and
        # sublst_refine_approved.py read the log's REFUSED lines against its LAST SUBLST-REFINE line, so a per-hour run
        # (--nights) must not append to an earlier run's log either; the earlier log is kept under logs/_previous/
        prev_dir = os.path.join(x2, 'logs', '_previous')
        os.makedirs(prev_dir, exist_ok=True)
        dst = os.path.join(prev_dir, f'{lst}_{sb}.log.{C.utcstamp()}')
        k = 1
        while os.path.lexists(dst):             # two runs in one second: never rename onto (and lose) an earlier log
            dst = os.path.join(prev_dir, f'{lst}_{sb}.log.{C.utcstamp()}.{k}')
            k += 1
        os.rename(log, dst)
    # ---- the child ------------------------------------------------------------------------------------------------------
    work = os.path.join(fitsio.scratch_root(args.work), fr.WORK_SUBDIR, f'gpu{g}')
    argv = ['--root', os.path.join(base, lst), '--out', x2, '--work', work, '--lst', lst, '--subband', sb]
    if nights:
        argv += ['--nights', ','.join(nights)]
    if args.sublst_windows:
        argv += ['--windows', args.sublst_windows]
    elif tuple(fr.WINDOWS) != (0, 1, 2, 3, 4, 5):
        argv += ['--windows', ','.join(str(w) for w in fr.WINDOWS)]
    env = {'CUDA_VISIBLE_DEVICES': g, 'DEEP_XLA_MEMFRAC': c.flow.XLA_MEM_FRACTION.get('sublst'), 'P3_FLOW_LOCK': lock,
           'REFINE_FFT_WORKERS': fr.FFT_WORKERS, 'PYTHONUNBUFFERED': '1'}
    env = {k: v for k, v in env.items() if v is not None}
    script = C.legacy_script(*REFINE_SCRIPT)
    u.add_input(script, extra={'role': 'code'})
    os.makedirs(os.path.dirname(log), exist_ok=True)
    u.meta['child_env'] = {k: str(v) for k, v in env.items()}
    res = legacy.run(script, argv, python='default', env=env, log=log, refusal_codes=(), refusal_text=(), unit=u)
    final = next((ln for ln in reversed((res.tail or '').splitlines()) if ln.startswith('SUBLST-REFINE ')), None)
    u.meta['refine'] = dict(rc=res.rc, wall_s=res.wall_s, log=log, final=final, lock=lock, gpu=g)
    if final:
        u.note(final)
    if res.rc == 1 and 'REFUSED' in (res.tail or ''):
        line = next((ln for ln in reversed(res.tail.splitlines()) if 'REFUSED' in ln), 'REFUSED')
        raise Refusal(f'phase3_sublst_refine.py: {line.strip()[:600]}')         # G-C1/G-C2/G-C5/G-C6 (the module's own)
    if res.rc not in (0, 3):
        raise RuntimeError(f'phase3_sublst_refine.py exit {res.rc}: {(res.tail or "")[-600:]}')
    verdict(u, run, args, lst, sb, nights, rc_child=res.rc)


def check_unit(u, args, run, lst, sb, nights):
    from exopipe.phase3.lib.flow import tests as T
    fr = C.cfg().flow_refine
    u.meta.update(lst=lst, subband=sb, nights=nights)
    wp = fr.WEIGHTS_CSV.format(lst=lst, sb=sb)
    if not os.path.exists(wp):
        raise Refusal(f'G-C1: no night-weight record {wp}')
    u.add_input(wp, extra={'role': 'frozen deep co-add night weights'})
    T.sublst_model(lst, sb, stack_root=os.path.join(_stack_base(args), lst), refresh=True)
    verdict(u, run, args, lst, sb, nights)


# ---------------------------------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------------------------------

def run(args, run):
    from exopipe.phase3.core import record
    stages = args.flow_stages
    try:
        its = items(args)
    except ValueError as e:
        raise Refusal(str(e))
    if not its:
        raise NoInput('no sub-LST item selected (give --lsts, --cells <LST>_<subband> or --epochs <LST>_<date>)')
    if run.dry_run:
        return plan(args, run, its, stages)
    for lst, sb, nights in its:
        refined = False
        if 'refine' in stages:
            name = unit_name('refine', lst, sb, nights)
            prev = record.load_record(run.out_root, 'flow', name)
            if run.resume and prev and prev.get('status') == 'done':
                # the window stacks re-listed now (a re-stacked or new night changes the leave-one-out Masters) and the
                # admitted nights recomputed from data_decisions now - never the record's own values
                from exopipe.phase3.lib.flow import tests as T
                model = T.sublst_model(lst, sb, stack_root=os.path.join(_stack_base(args), lst), refresh=True)
                stacks = sorted(p for nd in model['stacks'].values() for p in nd.values())
                ok, why = C.listing_current(prev, 'window stacks', stacks)
                if ok:
                    _p, excl = excluded_stacks(model, lst, sb)
                    nights_now = sorted({n for nd in model['stacks'].values() for n in nd})
                    ok, why = run.resume_check(name, inputs=None, admitted=[n for n in nights_now if n not in excl])
                if not ok:
                    print(f'flow sublst {name}: not resumed ({why})')
                if ok:
                    u = run.unit(name, tier='sublst')
                    u.status, u.reason = 'skipped-valid', why
                    print(f'flow sublst {name}: skipped ({why})')
                    continue
            try:
                with run.unit(name, tier='sublst', lst=lst, subband=sb) as u:
                    refine_unit(u, args, run, lst, sb, nights)
            except Exception as e:
                print(f'flow sublst {name}: error {type(e).__name__}: {e}')
            refined = True
        if 'check' in stages and not refined:
            name = unit_name('check', lst, sb, nights)
            try:
                with run.unit(name, tier='sublst', lst=lst, subband=sb) as u:
                    check_unit(u, args, run, lst, sb, nights)
            except Exception as e:
                print(f'flow sublst {name}: error {type(e).__name__}: {e}')
    return None


def plan(args, run, its, stages):
    from exopipe.phase3.lib.flow import tests as T
    c = C.cfg()
    fr = c.flow_refine
    rc = 0
    base = _stack_base(args)
    try:
        drift_guard(lambda s: print(f'[plan] {s}'))
    except Refusal as e:
        print(f'[plan] WOULD REFUSE - {e}')
        return 2
    gpus = C.parse_gpus(args.gpus) or C.site_gpu_ids()
    for lst, sb, nights in its:
        refusal = None
        wp = fr.WEIGHTS_CSV.format(lst=lst, sb=sb)
        x2 = os.path.join(run.out_root, f'{lst}_x2')
        model = T.sublst_model(lst, sb, stack_root=os.path.join(base, lst), refresh=True) if os.path.exists(wp) else None
        nset = set(nights) if nights else None
        print(f'[plan] unit {unit_name("refine" if "refine" in stages else "check", lst, sb, nights)}: weights '
              f'{"present" if model else "ABSENT"} ({wp})')
        if model is None:
            refusal = 'G-C1: no night-weight record'
        else:
            nst = sum(len(nd) for nd in model['stacks'].values())
            nwin = {k for k in model['wok'] if model['wok'][k]}
            exp = [k for k in model['expected'] if nset is None or k[2] in nset]
            have = T.residuals_on_disk(x2, lst, sb, nset)
            _, excl = excluded_stacks(model, lst, sb)
            appr = T.approved_rows(lst, sb)
            print(f'[plan]   {nst} window stacks under {os.path.join(base, lst)}; (Stokes, window) allowed {len(nwin)} of '
                  f'{len(model["wok"])}; expected residuals {len(exp)}; on disk now under {x2}: {len(have)}; approved '
                  f'refusal rows {sorted(appr)}; excluded-night stacks {excl}')
            if excl and fr.EXCLUDED_STACK == 'refuse':
                refusal = f'G-C12: stacks of excluded nights {excl}'
            if 'refine' in stages:
                if have and not run.fresh:
                    refusal = refusal or f'G-C5: {len(have)} earlier residuals under {x2} (needs --fresh --supersede TAG)'
                lock = '-'
                if not gpus:
                    refusal = refusal or f'G-C11: no GPU id for {C.host()}'
                else:
                    try:
                        lock = C.required_gpu_locks(gpus[0], 'G-C11')[0]
                    except Refusal as e:
                        refusal = refusal or str(e)
                print(f'[plan]   refine: {c.site.PYTHON.get("default")} clv15/phase3_sublst_refine.py --root '
                      f'{os.path.join(base, lst)} --out {x2} --work <work>/{fr.WORK_SUBDIR}/gpu{gpus[0] if gpus else "?"} '
                      f'--lst {lst} --subband {sb}' + (f' --nights {",".join(nights)}' if nights else '')
                      + f'; CUDA_VISIBLE_DEVICES={gpus[0] if gpus else "?"} DEEP_XLA_MEMFRAC='
                      f'{c.flow.XLA_MEM_FRACTION.get("sublst")} P3_FLOW_LOCK={lock}')
            print('[plan]   verdict: R1 (allowed + LOONCOAD), R2 (expected written except EXACTLY the approved rows: G-C7), '
                  'cards (G-C9) [sublst_x2_residual]')
        if refusal:
            print(f'[plan] {lst}_{sb}: WOULD REFUSE - {refusal}')
            rc = 2
    return rc
