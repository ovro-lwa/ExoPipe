"""exopipe3 flow --tier sublst --stages refine-nonmember - the x2-aligned 10-minute (sub-LST) residual of ONE night that
is NOT in the window Masters, against the INSTALLED window Masters (the per-hour arm B; the batch arm B is the stage
`refine`, clv15/phase3_sublst_refine.py in place).

Ruling: Gregg 2026-11-12 - nights that are not in a Master are searched as non-members against the frozen Masters
("Approved."). The arithmetic is production's for a night of weight 0 (phase3_sublst_refine.py since clv15 5b1fc5b): the
leave-one-out Master of such a night IS the window Master (the night added nothing to the sum or the weight), so the
image the night is aligned to and the image that is subtracted are both the window Master (Gregg 2026-09-25: the x2
alignment "must happen with the Master that will be subtracted"). Two differences from the batch stage:
  - the window Master is READ FROM ITS INSTALLED FILE (float32) instead of being recomputed in float64 from every night's
    stacks: per subband the unit reads the night's 12 window stacks and 12 installed Masters, no other night's stack;
  - the windows of the subband go to ONE aligner process (one line per window in its pairs file) instead of one
    process per window.

Words used here. Window stack, window Master, member, non-member: as in lib/masters/sublst_nonmember.py. x2 alignment:
the Stokes I stack and the Stokes I Master are upsampled by 2 (Fourier zero-padding), the aligner (phase2/align_images.py,
Brox optical flow on the GPU, its own default parameters) returns the displacement field that maps the stack onto the
Master, the field is applied to the upsampled stack of each Stokes (the I field registers V), and the result is brought
back to the native grid.

Unit of work: one (LST hour, subband, night), Stokes I and V: unit refine-nonmember_<LST>_<sb>_<night>, one per subband
of the night's job plan (the admission of the stack stage decides which subbands a night owes). Per window k:
    flow_k  = aligner( upsample2(stack I) -> upsample2(installed Master I) )
    r_k(S)  = decimate2( flow_k applied to upsample2(stack S) ) - installed Master S         (float32), S = I and V
    NaN where less than flow_refine.X2_VALID_FOOTPRINT of the pixel's x2 source footprint was valid, or where the Master
    has no data. The Jupiter mask of the Masters is in the installed pixels; the night's own residual keeps Jupiter.

Inputs (read only; fingerprinted in the unit record): as the arm-A unit (sublst_nonmember.assess: the night's plan and
stack with the evidence that the stack jobs completed, the installed window Masters and their weights records,
data_decisions.csv), under --sublst-stack-root (default paths.SUBLST_ROOT).
Outputs (production's names and cards, in production's order: SUBTRACT, LOONCOAD = the Master's weighted nights, LOOMSTR,
CODE, LSTBIN, DURATION, REFINE, REFREF, FLOWMEDP, JUPMASK = the Master's card; no BINMSTR):
  <out>/<LST>_x2/<LST>_<night>/<sb>/<S>/sublst_<LST>_<night>_<sb>_<S>_w<k>_resid_loo.fits      [sublst_x2_residual]
  <out>/<LST>_x2/logs/<LST>_<sb>.log    production's log lines (ALIGNED, REFUSED-WINDOW, REFUSED-NIGHT, ALIGNER-ERROR and
                                        the final SUBLST-REFINE line): the frames tier's refine gate reads this file
  <out>/_records/flow/logs/<unit>_aligner.log   the aligner's own output
The verdict on the outputs is the batch stage's (steps_sublst.verdict: every expected residual written except exactly
the approved rows, each allowed, with LOONCOAD equal to an independent count, and the cards).

Failure modes (exit 2 = refused; the gates N-1 to N-10 are sublst_nonmember.assess's and are checked before anything is
moved or written):
  N-1..N-10  as the arm-A unit; N-5 = the night IS a member of an installed window Master: the batch stage `refine`.
  G-C5   x2 residuals of the night and subband from an earlier run, without --fresh --supersede TAG.
  G-C7   a window whose flow did not come back (its residuals are not written; production: REFUSED-NIGHT) unless the
         missing (night, window) pairs are exactly the approved rows.
  G-C11  no GPU, or no lock for it, on this host.
Exit 1 (error): the aligner ended with an exit code outside flow_refine.ALIGNER_OK_RC (for example a GPU out of memory);
the residuals this unit wrote are removed.
"""
import glob
import os
import re

from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.flow import steps_common as C

STAGE = 'refine-nonmember'
BATCH_STAGE = 'flow --tier sublst --stages refine'
CODE_COMMENT = 'exopipe3 lib/flow/sublst_nonmember_refine.py'
# Threads over the windows of one unit, for the work before and after the aligner (the x2 images; the warps and the
# decimation). The same arithmetic per window in any order: scipy's FFT and map_coordinates run outside the interpreter
# lock (lwacalim07, scipy 1.10.1: four warps in four threads 2.0 s, one after the other 6.5 s). An hour has six windows.
WINDOW_THREADS = 6
# the repo's own copy of the aligner (byte-identical to the one clv15/phase3_sublst_refine.py:44 shells), never imported
ALIGN = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                     'phase2', 'align_images.py')


def unit_name(lst, sb, night):
    return f'{STAGE}_{lst}_{sb}_{night}'


# =====================================================================================================================
# the arithmetic and the header (clv15/phase3_sublst_refine.py for a night of weight 0)
# =====================================================================================================================

def fft_upsample2(a, workers):
    """Band-limited x2 upsample by Fourier zero-padding; refine.py:61-65 (the formula of core.residuals.fft_upsample2
    with scipy.fft on ``workers`` threads, as production's refine)."""
    import numpy as np
    import scipy.fft as sf
    a = np.nan_to_num(a)
    n = a.shape[0]
    F = sf.fftshift(sf.fft2(a, workers=workers))
    P = np.zeros((2 * n, 2 * n), complex)
    P[n // 2:n // 2 + n, n // 2:n // 2 + n] = F
    return np.real(sf.ifft2(sf.ifftshift(P), workers=workers)) * 4.0


def fft_decimate2(a, workers):
    """The Fourier plane of an x2 image cropped back; refine.py:67-70. The inverse of fft_upsample2 except on the Nyquist
    row and column, which come back halved (the real part of the x2 image keeps half of them); as production."""
    import numpy as np
    import scipy.fft as sf
    n = a.shape[0] // 2
    F = sf.fftshift(sf.fft2(a, workers=workers))
    Cc = F[n // 2:n // 2 + n, n // 2:n // 2 + n]
    return np.real(sf.ifft2(sf.ifftshift(Cc), workers=workers)) / 4.0


def x2_residual(d, up, flow, master, workers, footprint):
    """One residual (float32) on the native grid; refine.py:219-224 with the leave-one-out Master = ``master``.
    d: the window stack (float64, NaN = no data); up: fft_upsample2(d); flow: the x2 displacement field; master: the
    installed window Master of the same Stokes (float64 of its float32 pixels)."""
    import numpy as np
    from exopipe.phase3.lib.masters.coadd import apply_flow_field
    warped = fft_decimate2(apply_flow_field(up, flow), workers)
    vup = np.repeat(np.repeat(np.isfinite(d).astype(np.float64), 2, 0), 2, 1)
    n0, n1 = d.shape
    vd = apply_flow_field(vup, flow).reshape(n0, 2, n1, 2).mean(axis=(1, 3))
    return np.where((vd > footprint) & np.isfinite(master), warped - master, np.nan).astype(np.float32)


def flow_median_px(flow):
    """Median length of the x2 displacement, in native pixels (FLOWMEDP; refine.py:213)."""
    import numpy as np
    return float(np.median(np.hypot(flow[..., 0], flow[..., 1]))) / 2.0


def residual_header(stack_header, master_header, master_path, k, n_weighted, code, fmed):
    """The residual's header: the stack's own header (its own time block) plus the cards of refine.py:225-244, same
    keywords, values, comments and order. n_weighted = the Master's weighted nights (LOONCOAD; for a non-member none is
    left out). JUPMASK is the Master's card (the mask radius used when the Master was built)."""
    from exopipe.phase3.lib.masters import sublst_nonmember as NM
    rh = stack_header.copy()
    rh['SUBTRACT'] = ('leave-one-out window Master', 'sub-LST arm B: x2-aligned to that Master first')
    rh['LOONCOAD'] = (int(n_weighted), 'weighted nights in the leave-one-out Master')
    rh['LOOMSTR'] = (os.path.basename(master_path)[:68], 'arm-A window Master (all nights)')
    rh['CODE'] = (str(code)[:68], CODE_COMMENT)
    rh['LSTBIN'] = (int(k), 'sidereal window index (sub-LST)')
    rh['DURATION'] = (NM.duration_s(rh), 's, first frame start to last frame end')
    if 'BINMSTR' in rh:
        del rh['BINMSTR']
    rh['REFINE'] = ('x2 flow to the subtracted LOO window Master', 'Gregg 2026-09-25')
    rh['REFREF'] = ('LOO window Master, this night left out, Stokes I', 'flow reference')
    rh['FLOWMEDP'] = (round(fmed, 4), 'median |x2 flow|, native pixels')
    rh['JUPMASK'] = (master_header['JUPMASK'], 'arcmin, Jupiter masked per night in the leave-one-out Master')
    return rh


# =====================================================================================================================
# the unit
# =====================================================================================================================

def _read(path):
    import numpy as np
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        return np.asarray(h[0].data, dtype=np.float64), h[0].header.copy()


def windows_to_align(a):
    """From an arm-A assessment: (ks, items {(S, k): item}, left {(S, k): reason}). ks = the windows aligned: the night
    has a Stokes I stack there and its Stokes I Master is installed. A Stokes V window is written only with a Stokes I
    flow of the same window (the I flow registers V; refine.py:137-141)."""
    items = {(it['S'], it['k']): it for it in a['items']}
    ks = sorted(k for (S, k), it in items.items() if S == 'I' and not it['refuse'])
    left = {}
    for (S, k), it in sorted(items.items()):
        if it['refuse']:
            left[(S, k)] = 'no installed window Master'
        elif k not in ks:
            left[(S, k)] = 'no Stokes I flow to register V (no Stokes I stack or Master of the window)'
    return ks, items, left


def rotate_log(log):
    """One run per refine log (steps_sublst.refine_unit): an earlier log is kept under logs/_previous/."""
    if not os.path.exists(log):
        return
    prev_dir = os.path.join(os.path.dirname(log), '_previous')
    os.makedirs(prev_dir, exist_ok=True)
    dst = os.path.join(prev_dir, f'{os.path.basename(log)}.{C.utcstamp()}')
    k = 1
    while os.path.lexists(dst):
        dst = os.path.join(prev_dir, f'{os.path.basename(log)}.{C.utcstamp()}.{k}')
        k += 1
    os.rename(log, dst)


def aligner_errors(text, ks):
    """[(window or None, line)] - the aligner's ERROR lines with the pair they belong to ('[i/n]' progress lines;
    refine.py:199-205)."""
    out, cur = [], None
    for ln in (text or '').splitlines():
        q = re.match(r'\s*\[(\d+)/\d+\]', ln)
        if q:
            cur = ks[int(q.group(1)) - 1] if 0 < int(q.group(1)) <= len(ks) else None
        elif 'ERROR' in ln:
            out.append((cur, ln.strip()[:300]))
    return out


def assess(run, base, lst, night, sb):
    """(refusal or None, arm-A assessment or None) of one unit, without writing: the exclusion gate and
    sublst_nonmember.assess on the stack root ``base``."""
    from exopipe.phase3.lib.masters import sublst_nonmember as NM
    ex, excluded = NM._exclusion_gate(lst, sb, night)
    if excluded:
        return excluded, None
    a = NM.assess(run, base, lst, f'{lst}_{night}', sb, ('I', 'V'), NM.stack_evidence_kind(), ex)
    if a['refuse']:
        return a['refuse'].replace('`masters --tier sublst --stages loo` writes (unit loo_', f'`{BATCH_STAGE}` aligns (unit refine_'), a
    ks, _items, _left = windows_to_align(a)
    if not ks:
        return (f'N-7: {lst}_{night} {sb}: no window of the night has a Stokes I stack with an installed Stokes I window '
                f'Master ({len(a["items"])} window stack(s))'), a
    return None, a


def refine_unit(u, args, run, lst, sb, night, base, plan_files=()):
    import time
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    from exopipe.phase3.core import fitsio, legacy
    from exopipe.phase3.core.residuals import write_up_fits
    from exopipe.phase3.lib.flow import steps_sublst as SS
    from exopipe.phase3.lib.flow import tests as T
    from exopipe.phase3.lib.masters import sublst_nonmember as NM
    c = C.cfg()
    fr = c.flow_refine
    cell = f'{lst}_{night}'
    u.meta.update(lst=lst, subband=sb, nights=[night], stage=STAGE)
    t0 = time.time()
    timing = {}                                           # seconds per phase of this unit, in its record (meta 'refine')
    SS.drift_guard(u.note)
    # ---- the gates of the arm-A unit, on the same stack and the same installed Masters ---------------------------------
    why, a = assess(run, base, lst, night, sb)
    if a is not None:
        for p in NM._inputs(a, plan_files):
            u.add_input(p)
        for n in a['notes']:
            u.note(n)
    if why:
        raise Refusal(why)
    u.set_admitted([f'{night}/{sb}'])
    ks, items, left = windows_to_align(a)
    timing['gates'] = round(time.time() - t0, 1)
    # ---- G-C11: a GPU and the lock production holds on it, before anything is moved ------------------------------------
    gpus = C.parse_gpus(args.gpus) or C.site_gpu_ids()
    if not gpus:
        raise Refusal(f'G-C11: no GPU id for {C.host()} (--gpus, or GPU_IDS in the site file)')
    g = gpus[0]
    lock = C.required_gpu_locks(g, 'G-C11')[0]
    # ---- G-C5: earlier x2 residuals of this (LST, subband, night) ------------------------------------------------------
    x2 = os.path.join(run.out_root, f'{lst}_x2')
    old = T.residuals_on_disk(x2, lst, sb, {night})
    if old:
        if not args.fresh:
            raise Refusal(f'G-C5: {len(old)} x2 residual(s) of an earlier run of {cell} {sb} under {x2} - pass --fresh '
                          f'--supersede TAG to move them aside first')
        moves = C.supersede(sorted(old.values()), run.out_root, run.out_root, args.supersede, f'flow {u.name} --fresh')
        u.note(f'--fresh: {len(moves)} earlier residuals moved aside under {run.out_root}/_superseded/{args.supersede}/')
    log = os.path.join(x2, 'logs', f'{lst}_{sb}.log')
    os.makedirs(os.path.dirname(log), exist_ok=True)
    rotate_log(log)
    lines = []

    def say(line):
        lines.append(line)
        with open(log, 'a') as f:
            f.write(line + '\n')

    for (S, k), reason in sorted(left.items()):
        say(f'REFUSED-WINDOW {sb} {S} w{k}: {reason} (non-member night {night})')
    # ---- the x2 images of the night's Stokes I windows and of their Masters, one pairs file ---------------------------
    work = os.path.join(fitsio.scratch_root(args.work), fr.WORK_SUBDIR, f'gpu{g}', f'{lst}_{sb}_{night}_nonmember')
    os.makedirs(work, exist_ok=True)
    for f in glob.glob(os.path.join(work, '*')):          # this unit's own files of an attempt that did not finish
        os.remove(f)
    W = int(fr.FFT_WORKERS)
    held, pairs = {}, []

    def x2_images(k):
        it = items[('I', k)]
        d, hh = _read(it['stack'])
        m, mh = _read(it['master'])
        up = fft_upsample2(d, W)
        ep = os.path.join(work, f'ep_{night}_w{k}_up2.fits')
        rf = os.path.join(work, f'ref_{night}_w{k}_up2.fits')
        write_up_fits(up, hh, ep)
        write_up_fits(fft_upsample2(m, W), hh, rf)
        return (d, hh, m, mh, up), f'{ep} {rf} {work}'

    with ThreadPoolExecutor(max_workers=max(1, min(len(ks), WINDOW_THREADS))) as ex:
        for k, (h, pair) in zip(ks, ex.map(x2_images, ks)):               # in window order: the pairs file is the same
            held[k] = h
            pairs.append(pair)
    pp = os.path.join(work, f'pairs_{os.getpid()}.txt')
    with open(pp, 'w') as f:
        f.write('\n'.join(pairs) + '\n')
    timing['x2_images'] = round(time.time() - t0 - sum(timing.values()), 1)
    env = {'CUDA_VISIBLE_DEVICES': g, 'XLA_FLAGS': '--xla_gpu_enable_command_buffer=',
           'XLA_PYTHON_CLIENT_PREALLOCATE': c.flow.XLA_PREALLOCATE, 'PYTHONUNBUFFERED': '1'}
    if c.flow.XLA_MEM_FRACTION.get('sublst') is not None:
        env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = c.flow.XLA_MEM_FRACTION['sublst']
    alog = os.path.join(run.out_root, '_records', 'flow', 'logs', f'{u.name}_aligner.log')
    os.makedirs(os.path.dirname(alog), exist_ok=True)
    if os.path.exists(alog):
        os.remove(alog)
    u.add_input(ALIGN, extra={'role': 'code'})
    with C.hold_locks([lock], c.flow.GPU_LOCK_WAIT_S, u.note) as locks_held:
        res = legacy.run(ALIGN, ['--pairs', pp, '--no_qa', '--save_flow'], python='flow', env=env, log=alog,
                         refusal_codes=(), refusal_text=(), unit=u)
    try:
        with open(alog, errors='replace') as f:
            atext = f.read()
    except OSError:
        atext = res.tail or ''
    for k, ln in aligner_errors(atext, ks):
        say(f'ALIGNER-ERROR {sb} w{"?" if k is None else k} {night}: {ln}')
    u.meta['aligner'] = dict(rc=res.rc, wall_s=res.wall_s, log=alog, pairs=len(pairs), gpu=g, locks=locks_held,
                             env={k: str(v) for k, v in env.items()})
    written = []

    def clean_work():
        for f in glob.glob(os.path.join(work, '*')):
            os.remove(f)
        try:
            os.rmdir(work)                                # the unit's own directory, empty now
        except OSError:
            pass

    if res.rc not in tuple(fr.ALIGNER_OK_RC):
        say(f'REFUSED: flow batch rc={res.rc} for {sb} {night}: {(res.tail or "")[-300:]!r}')
        clean_work()
        raise RuntimeError(f'aligner exit {res.rc} for {cell} {sb} (log {alog}): {(res.tail or "")[-400:]}')
    timing['aligner_and_lock'] = round(time.time() - t0 - sum(timing.values()), 1)
    # ---- the residuals ------------------------------------------------------------------------------------------------
    n_ok = n_ref = 0
    medians = {}

    def window_residuals(k):
        """The residuals of window k (Stokes I, then V with the same flow); the median flow, or None without a flow."""
        d, hh, m, mh, up = held.pop(k)
        fp = os.path.join(work, f'ep_{night}_w{k}_up2_flow.npy')
        if not os.path.exists(fp):
            return None
        fl = np.load(fp)
        fmed = flow_median_px(fl)
        for S in ('I', 'V'):
            it = items.get((S, k))
            if it is None or (S, k) in left:
                continue
            if S == 'V':
                d, hh = _read(it['stack'])
                m, mh = _read(it['master'])
                up = fft_upsample2(d, W)
            res_px = x2_residual(d, up, fl, m, W, float(fr.X2_VALID_FOOTPRINT))
            rh = residual_header(hh, mh, it['master'], k, it['n_weighted'], run.commit, fmed)
            written.append(fitsio.write_fits(T.residual_path(x2, lst, night, sb, S, k), res_px, rh, overwrite=False))
        return fmed

    try:
        with ThreadPoolExecutor(max_workers=max(1, min(len(ks), WINDOW_THREADS))) as ex:
            futures = [ex.submit(window_residuals, k) for k in ks]       # every thread has ended when the block closes
        for k, fu in zip(ks, futures):
            fmed = fu.result()                                            # the first failure, raised here
            if fmed is None:
                say(f'REFUSED-NIGHT {sb} w{k} {night}: no flow returned')
                n_ref += 1
                continue
            medians[k] = round(fmed, 4)
            n_ok += 1
            say(f'ALIGNED {lst} {sb} w{k} {night} median_flow={fmed:.3f} px')
        written.sort()
    except BaseException:
        # a unit that fails while writing leaves NO residual of its own behind (the wideband and search stages take the
        # night's residuals from the directory): the files this process wrote, and its own unfinished temporary files
        tmp = [os.path.join(os.path.dirname(p), f'.{os.path.basename(p)}.tmp.{os.getpid()}')
               for S in ('I', 'V') for p in (T.residual_path(x2, lst, night, sb, S, k) for k in ks)]
        for p in written + tmp:
            if os.path.lexists(p):
                os.unlink(p)
        clean_work()
        raise
    clean_work()
    timing['residuals'] = round(time.time() - t0 - sum(timing.values()), 1)
    n_rule = len(left)
    final = f'SUBLST-REFINE {lst} {sb} aligned={n_ok} refused={n_ref} refused_rule={n_rule} refused_night=0'
    say(final)
    u.note(final)
    u.meta['refine'] = dict(rc=3 if n_ref else 0, log=log, final=final, lock=lock, gpu=g, windows=ks,
                            flow_median_px=medians, residuals=len(written), reference='installed window Master',
                            windows_left={f'{S} w{k}': r for (S, k), r in sorted(left.items())}, timing_s=timing)
    print(f'  {u.name}: {len(written)} residual(s) of {n_ok} window(s) against the installed window Masters'
          + (f', {n_ref} window(s) without a flow' if n_ref else ''))
    try:
        SS.verdict(u, run, args, lst, sb, [night], rc_child=3 if n_ref else 0)
    finally:
        timing['verdict'] = round(time.time() - t0 - sum(timing.values()), 1)


# =====================================================================================================================
# the stage (called by steps_sublst.run and steps_sublst.plan)
# =====================================================================================================================

def night_units(run, args, lst, night, subbands):
    """([subband] of the night's job plan in this selection, plan files, how). The plan is the stack stage's (admit
    record admit_<LST>_<night> of this run root): a subband the admission left out owes no residual and gets no unit."""
    from exopipe.phase3.lib.masters import sublst_nonmember as NM
    from exopipe.phase3.lib.masters import tier_sublst as TS
    jobs, files, how = NM.night_plan(run, TS.Opts.from_args(args), lst, night)
    cell = f'{lst}_{night}'
    owed = [sb for c_, sb, _st, _cr in (jobs or []) if c_ == cell]
    return [sb for sb in subbands if sb in owed], [f for f in files], how


def by_night(its):
    """[(lst, night, [subband])] of the step's items [(lst, sb, nights or None)]; an item without a named night refuses."""
    out = {}
    for lst, sb, nights in its:
        if not nights:
            raise Refusal(f'the {STAGE} stage takes named nights (--epochs <LST>_<date>): {lst} {sb} was given without one; '
                          f'the x2 residuals of every night of an LST are the batch stage `{BATCH_STAGE}`')
        for n in nights:
            out.setdefault((lst, n), []).append(sb)
    return [(lst, n, sbs) for (lst, n), sbs in sorted(out.items())]


def run_stage(args, run, its, base):
    for lst, night, sbs in by_night(its):
        try:
            units, files, how = night_units(run, args, lst, night, sbs)
        except Refusal as e:
            with run.unit(f'{STAGE}_{lst}_{night}', tier='sublst', lst=lst, stage=STAGE) as u:
                u.refuse(f'N-2: {e}')
            continue
        if not units:
            with run.unit(f'{STAGE}_{lst}_{night}', tier='sublst', lst=lst, stage=STAGE):
                raise NoInput(f'no stack job of the night\'s plan in this selection: nothing to align ({how})')
            continue
        for sb in units:
            name = unit_name(lst, sb, night)
            try:
                with run.unit(name, tier='sublst', lst=lst, subband=sb) as u:
                    refine_unit(u, args, run, lst, sb, night, base, files)
            except Exception as e:
                print(f'flow sublst {name}: error {type(e).__name__}: {e}')


def plan_stage(args, run, its, base):
    """Dry run: what each unit would do; returns 2 when a gate would refuse."""
    from exopipe.phase3.lib.flow import tests as T
    rc = 0
    gpus = C.parse_gpus(args.gpus) or C.site_gpu_ids()
    for lst, night, sbs in by_night(its):
        try:
            units, _files, how = night_units(run, args, lst, night, sbs)
        except Refusal as e:
            print(f'[plan] {STAGE} {lst}_{night}: WOULD REFUSE - N-2: {e}')
            rc = 2
            continue
        print(f'[plan] {STAGE} {lst}_{night}: {len(units)} unit(s) of {len(sbs)} selected subband(s) (plan: {how}); installed '
              f'window Masters under {os.path.join(base, C.cfg().flow_refine.MASTERS_DIR.format(lst=lst))}')
        x2 = os.path.join(run.out_root, f'{lst}_x2')
        for sb in units:
            try:
                why, a = assess(run, base, lst, night, sb)
                if not why and not gpus:
                    why = f'G-C11: no GPU id for {C.host()}'
                if not why:
                    C.required_gpu_locks(gpus[0], 'G-C11')
                old = T.residuals_on_disk(x2, lst, sb, {night})
                if not why and old and not run.fresh:
                    why = f'G-C5: {len(old)} earlier x2 residual(s) under {x2} (needs --fresh --supersede TAG)'
            except Refusal as e:
                why, a = str(e), None
            if why:
                print(f'[plan]   {unit_name(lst, sb, night)}: WOULD REFUSE - {why}')
                rc = 2
                continue
            ks, items, left = windows_to_align(a)
            n = sum(1 for (S, k) in items if (S, k) not in left)
            print(f'[plan]   {unit_name(lst, sb, night)}: {len(ks)} window(s) to one aligner process (GPU {gpus[0]}), {n} '
                  f'residual(s) to write' + (f'; left out: {sorted(f"{S} w{k}" for S, k in left)}' if left else ''))
    return rc
