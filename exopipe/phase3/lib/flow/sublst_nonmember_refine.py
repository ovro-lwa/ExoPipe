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
  <out>/<LST>_x2/logs/<LST>_<sb>.log    the subband's refine log, which the frames tier's refine gate and the approvals
                                        judge read. The unit MERGES into it (merge_log): the lines of the batch stage
                                        and of the other nights stay as they are, this night's lines (production's
                                        forms: ALIGNED, REFUSED-NIGHT, ALIGNER-ERROR) replace its earlier ones, and the
                                        final SUBLST-REFINE line is counted again over the whole log
  <out>/_records/flow/logs/<unit>_aligner.log   the aligner's own output
The verdict on the outputs is the batch stage's (steps_sublst.verdict: every expected residual written except exactly
the approved rows, each allowed, with LOONCOAD equal to an independent count, and the cards).

Order of work: every gate first; then the x2 images and the aligner on node scratch (nothing in the product tree is
touched while the unit waits for the GPU); then, under a lease of the subband's x2 directory, earlier residuals are
moved aside, the new ones are written and judged, and the log is merged. A unit that is refused or fails after it wrote
residuals removes them (as the arm-A unit), so a refused unit never leaves residuals for the wideband, tree and search
stages; when earlier residuals were moved aside the log carries the line NONMEMBER-INCOMPLETE <sb> <night> and no final
line (the frames tier's refine gate refuses the subband) until the unit has run again.
--resume skips a unit whose record is done and still describes what an assessment finds now.

Failure modes (exit 2 = refused; every gate is checked before anything is moved or written):
  N-1..N-10  as the arm-A unit; N-5 = the night IS a member of an installed window Master: the batch stage `refine` -
         checked first, the unit ends NO-INPUT (exit 3; Gregg 2026-11-13: a member night's 10-minute products stay the
         batch chain's), its record names the member windows (meta master_member).
  N-11   a Stokes V window stack of the night without the night's Stokes I stack of the window (nothing registers it).
  N-12   the residuals the unit would write are not the ones the verdict's independent model owes, or the weighted
         nights of an installed Master's weights record differ from the model's count (a member night's stacks moved
         away, or the frozen weights changed since the Masters were installed): the verdict would refuse the residuals.
  N-13   the subband's refine log has no final line (a batch refine is running, or failed): the unit does not merge
         into it.
  G-C5   x2 residuals of the night and subband from an earlier run, without --fresh --supersede TAG.
  G-C7   a window whose flow did not come back (its residuals are not written; production: REFUSED-NIGHT) unless the
         missing (night, window) pairs are exactly the approved rows.
  G-C8   a written residual fails its content test.
  G-C11  no GPU, or no lock for it, on this host; no node-local scratch root; no interpreter for the aligner.
  --sublst-windows is the batch stage's option: the unit aligns every window of the night and refuses it.
Exit 1 (error): the aligner ended with an exit code outside flow_refine.ALIGNER_OK_RC (for example a GPU out of memory);
nothing was moved or written in the product tree at that point.
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
XLA_MEM_FRACTION_DEFAULT = 0.45     # refine.py:101, when the config names none for the tier
INCOMPLETE = 'NONMEMBER-INCOMPLETE'  # log line of a night whose earlier residuals were moved aside and not replaced yet
LEASE_POLL_S = 2                    # s between attempts to take the subband's x2 lease (held for the write phase)
LEASE_WAIT_S = 3600                 # s before a unit gives up waiting for it (error, retryable)
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
    """From an arm-A assessment: (ks, items {(S, k): item}, left {(S, k): kind}). ks = the windows aligned: the night
    has a Stokes I stack there and its Stokes I Master is installed. A Stokes V window is written only with a Stokes I
    flow of the same window (the I flow registers V; refine.py:137-141). Kinds of a window that is left out:
      'no-master'  no installed window Master of this Stokes and window (the window Masters' stage wrote none);
      'i-window'   Stokes V whose Stokes I window has no installed Master (production: the V window is refused);
      'i-stack'    Stokes V whose Stokes I window has a Master but the NIGHT has no Stokes I stack there (gate N-11)."""
    items = {(it['S'], it['k']): it for it in a['items']}
    ks = sorted(k for (S, k), it in items.items() if S == 'I' and not it['refuse'])
    left = {}
    for (S, k), it in sorted(items.items()):
        if it['refuse']:
            left[(S, k)] = 'no-master'
        elif k not in ks:
            left[(S, k)] = 'i-window' if ('I', k) in items else 'i-stack'
    return ks, items, left


def window_line(sb, S, k, kind):
    """The log line of a window that is left out for EVERY night (production's REFUSED-WINDOW forms, refine.py:135-140,
    with the suffix exopipe_ops/sublst_refine_approved.py accepts: the window Masters' stage wrote no Master because the
    window has fewer than 2 weighted nights, arm A's rule)."""
    if kind == 'i-window':
        return (f"REFUSED-WINDOW {sb} V w{k}: no Stokes I flow to register V (Stokes I window refused; arm A's rule, Gregg "
                f"2026-10-06)")
    return (f"REFUSED-WINDOW {sb} {S} w{k}: no installed window Master (the window Masters' stage wrote none; arm A's rule, "
            f"Gregg 2026-10-06)")


def _log_lines(log):
    if not os.path.exists(log):
        return None
    with open(log, encoding='utf-8', errors='surrogateescape') as f:       # written back byte for byte
        return f.read().splitlines()


def _night_line(line, sb, night):
    """True for a log line about ONE night's residuals of the subband: ALIGNED, REFUSED-NIGHT (no flow returned; and the
    batch stage's 'no co-add night weight', which a night on the non-member list no longer has), ALIGNER-ERROR, and the
    unit's own marker."""
    b, n = re.escape(sb), re.escape(night)
    return bool(re.match(rf'(ALIGNED \S+ {b} w\d+ {n} |REFUSED-NIGHT {b} (?:[IV] )?w\d+ {n}: |ALIGNER-ERROR {b} w\S+ {n}: '
                         rf'|{INCOMPLETE} {b} {n}: )', line))


def log_gate(log, lst, sb):
    """N-13: None, or why the unit does not merge into the subband's refine log: the log exists without a final
    SUBLST-REFINE line and without a marker of this stage - a batch refine is writing it, or failed."""
    lines = _log_lines(log)
    if lines is None or any(ln.startswith(f'SUBLST-REFINE {lst} {sb} ') or ln.startswith(INCOMPLETE + ' ') for ln in lines):
        return None
    return (f'N-13: the refine log {log} has no final SUBLST-REFINE line: a batch refine of {lst} {sb} is running or '
            f'failed (`{BATCH_STAGE}`); the unit does not merge its lines into an unfinished log')


def merge_log(log, lst, sb, night, new_lines, marker=None):
    """Rewrite the subband's refine log with this night's lines replaced. Every other line is kept as it is (the batch
    stage's lines and the other nights'); the night's earlier lines and the earlier final line are dropped; ``new_lines``
    are added (a REFUSED-WINDOW line only when the log has none for that Stokes and window); the final SUBLST-REFINE
    line is counted again over the whole log with production's counters (refine.py:256: one per ALIGNED line, per
    'REFUSED-NIGHT ... no flow returned' line, per REFUSED-WINDOW line, per REFUSED-RESIDUAL line), so the approvals
    judge (exopipe_ops/sublst_refine_approved.py) finds refused=M with M such lines over all nights.
    With ``marker`` (a reason) the night is marked incomplete instead. A log with the marker of ANY night gets no final
    line: the frames tier's refine gate then refuses the subband until that night's unit has run again.
    Written to a temporary file and renamed. Returns (final line or None, the marker lines left in the log)."""
    kept = [ln for ln in (_log_lines(log) or [])
            if not ln.startswith(f'SUBLST-REFINE {lst} {sb} ') and not _night_line(ln, sb, night)]
    lines = list(kept)
    for ln in new_lines:
        q = re.match(r'(REFUSED-WINDOW \S+ [IV] w\d+): ', ln)
        if q and any(k.startswith(q.group(1) + ': ') for k in kept):
            continue
        lines.append(ln)
    if marker:
        lines.append(f'{INCOMPLETE} {sb} {night}: {marker}')
    markers = [ln for ln in lines if ln.startswith(INCOMPLETE + ' ')]
    final = None
    if not markers:
        def count(pat):
            return sum(1 for ln in lines if re.match(pat, ln))
        n_ok, n_ref = count(r'ALIGNED '), count(r'REFUSED-NIGHT \S+ w\d+ \S+: no flow returned')
        n_rule, n_night = count(r'REFUSED-WINDOW '), count(r'REFUSED-RESIDUAL ')
        final = f'SUBLST-REFINE {lst} {sb} aligned={n_ok} refused={n_ref} refused_rule={n_rule} refused_night={n_night}'
        lines.append(final)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    tmp = os.path.join(os.path.dirname(log), f'.{os.path.basename(log)}.tmp.{os.getpid()}')
    try:
        with open(tmp, 'w', encoding='utf-8', errors='surrogateescape') as f:
            f.write(''.join(ln + '\n' for ln in lines))
        os.replace(tmp, log)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return final, markers


def x2_lease(x2, lst, sb, unit):
    """The writer lease of ONE subband's x2 residuals and refine log (capacity 1 across processes and nodes): two nights
    of an hour move, write and merge one after the other. The search's Lease class (holder files under a mkdir lock,
    heartbeats, stale holders removed); the lease lives beside the log it guards, so every writer of this x2 directory
    finds the same one."""
    from exopipe.phase3.lib.search import tier_sublst as SE
    return SE.Lease(os.path.join(x2, 'logs', '_leases'), f'{lst}_{sb}', 'node-memory', capacity=1, poll_s=LEASE_POLL_S,
                    wait_s=LEASE_WAIT_S, info={'unit': unit, 'x2': x2})


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
    ks, items, left = windows_to_align(a)
    if not ks:
        return (f'N-7: {lst}_{night} {sb}: no window of the night has a Stokes I stack with an installed Stokes I window '
                f'Master ({len(a["items"])} window stack(s))'), a
    orphan = sorted(k for (S, k), kind in left.items() if kind == 'i-stack')
    if orphan:
        return (f'N-11: {lst}_{night} {sb}: Stokes V window stack(s) w{",".join(map(str, orphan))} without the night\'s '
                f'Stokes I stack of the window: no flow registers them (the stack stage writes both Stokes of a window)'), a
    # N-12: what the verdict will judge, before the GPU work and before anything is moved - the residuals this unit
    # writes against the independent model's (the stack listing of every night and the frozen weights NOW), and the
    # LOONCOAD each residual will carry (the Master's weights record) against the model's count
    from exopipe.phase3.lib.flow import tests as T
    model = T.sublst_model(lst, sb, stack_root=os.path.join(base, lst), refresh=True)
    write = {k2 for k2 in items if k2 not in left}
    owed = {(S, k) for (S, k, n) in model['expected'] if n == night}
    if write != owed:
        def show(x):
            return ', '.join(f'{S} w{k}' for S, k in sorted(x)) or 'none'
        return (f'N-12: {lst}_{night} {sb}: the unit would write residuals the verdict\'s model does not owe ({show(write - owed)}) '
                f'or not write ones it owes ({show(owed - write)}); the model counts the window stacks under '
                f'{os.path.join(base, lst)} with the frozen weights {model["weights_path"]}'), a
    bad = [(S, k, items[(S, k)]['n_weighted'], model['nloo'].get((S, k, night))) for S, k in sorted(write)
           if items[(S, k)]['n_weighted'] != model['nloo'].get((S, k, night))]
    if bad:
        S, k, n_m, n_i = bad[0]
        return (f'N-12: {lst}_{night} {sb}: {len(bad)} installed window Master(s) count other weighted nights than the stacks '
                f'on disk with the frozen weights, e.g. {S} w{k}: {n_m} in the Master\'s weights record, {n_i} from the '
                f'stack listing under {os.path.join(base, lst)} and {model["weights_path"]} - a member night\'s stacks were '
                f'moved away, or the weights changed since the Masters were installed; the verdict would refuse LOONCOAD'), a
    return None, a


def early_gates(args, run, lst, sb, night):
    """The gates after the assessment that need no arm-A result, each before anything is moved or written:
    (refusal or None, gpu, lock, scratch root, x2 root, log, earlier residuals)."""
    from exopipe.phase3.core import fitsio, legacy
    from exopipe.phase3.lib.flow import tests as T
    x2 = os.path.join(run.out_root, f'{lst}_x2')
    log = os.path.join(x2, 'logs', f'{lst}_{sb}.log')
    old = T.residuals_on_disk(x2, lst, sb, {night})
    gpus = C.parse_gpus(args.gpus) or C.site_gpu_ids()
    if not gpus:
        return f'G-C11: no GPU id for {C.host()} (--gpus, or GPU_IDS in the site file)', None, None, None, x2, log, old
    try:
        lock = C.required_gpu_locks(gpus[0], 'G-C11')[0]
        scratch = fitsio.scratch_root(args.work)
        legacy.interpreter('flow')
    except Refusal as e:
        why = str(e)
        return (why if why.startswith('G-C11') else f'G-C11: {why}'), gpus[0], None, None, x2, log, old
    why = None
    if getattr(args, 'sublst_windows', None):
        why = (f'--sublst-windows {args.sublst_windows} is the batch stage\'s option (`{BATCH_STAGE}`): the {STAGE} unit '
               f'aligns every window of the night')
    elif old and not args.fresh:
        why = (f'G-C5: {len(old)} x2 residual(s) of an earlier run of {lst}_{night} {sb} under {x2} - pass --fresh '
               f'--supersede TAG to move them aside first')
    elif old and not args.supersede:
        why = 'G-C5: --fresh would replace earlier residuals: give --supersede TAG (they are moved aside, never deleted)'
    else:
        why = log_gate(log, lst, sb)
    return why, gpus[0], lock, scratch, x2, log, old


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
    # ---- every gate, before anything is moved or written: the arm-A unit's on the same stack and installed Masters,
    #      N-11 and N-12, then the GPU, its lock, the scratch root, the aligner's interpreter, G-C5 and the log ----------
    member = None if NM._exclusion_gate(lst, sb, night)[1] else NM.master_member(base, lst, sb, night)
    if member:                                            # N-5 first: no-input, the batch chain's (Gregg 2026-11-13)
        u.meta['master_member'] = [list(x) for x in member]
        raise NoInput(NM.member_why(lst, night, sb, member))
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
    why, g, lock, scratch, x2, log, _old = early_gates(args, run, lst, sb, night)
    if why:
        raise Refusal(why)
    timing['gates'] = round(time.time() - t0, 1)
    # ---- the x2 images of the night's Stokes I windows and of their Masters, one pairs file ---------------------------
    work = os.path.join(scratch, fr.WORK_SUBDIR, f'gpu{g}', f'{lst}_{sb}_{night}_nonmember')
    os.makedirs(work, exist_ok=True)

    def clean_work():
        for f in glob.glob(os.path.join(work, '*')):
            os.remove(f)
        try:
            os.rmdir(work)                                # the unit's own directory, empty now
        except OSError:
            pass

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

    night_lines = []                                      # this night's lines for the subband's refine log
    try:
        with ThreadPoolExecutor(max_workers=max(1, min(len(ks), WINDOW_THREADS))) as ex:
            for k, (h, pair) in zip(ks, ex.map(x2_images, ks)):           # in window order: the pairs file is the same
                held[k] = h
                pairs.append(pair)
        pp = os.path.join(work, f'pairs_{os.getpid()}.txt')
        with open(pp, 'w') as f:
            f.write('\n'.join(pairs) + '\n')
        timing['x2_images'] = round(time.time() - t0 - sum(timing.values()), 1)
        frac = c.flow.XLA_MEM_FRACTION.get('sublst')
        env = {'CUDA_VISIBLE_DEVICES': g, 'XLA_FLAGS': '--xla_gpu_enable_command_buffer=',
               'XLA_PYTHON_CLIENT_PREALLOCATE': c.flow.XLA_PREALLOCATE, 'PYTHONUNBUFFERED': '1',
               'XLA_PYTHON_CLIENT_MEM_FRACTION': XLA_MEM_FRACTION_DEFAULT if frac is None else frac}
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
            night_lines.append(f'ALIGNER-ERROR {sb} w{"?" if k is None else k} {night}: {ln}')
        u.meta['aligner'] = dict(rc=res.rc, wall_s=res.wall_s, log=alog, pairs=len(pairs), gpu=g, locks=locks_held,
                                 env={k: str(v) for k, v in env.items()})
        if res.rc not in tuple(fr.ALIGNER_OK_RC):
            for ln in night_lines:
                u.note(ln)
            raise RuntimeError(f'aligner exit {res.rc} for {cell} {sb} (log {alog}): {(res.tail or "")[-400:]}')
        timing['aligner_and_lock'] = round(time.time() - t0 - sum(timing.values()), 1)
    except BaseException:
        clean_work()                                      # nothing was moved or written in the product tree
        raise
    # ---- the write phase, under the lease of the subband's x2 directory: earlier residuals aside, the new ones, the
    #      verdict, the log -------------------------------------------------------------------------------------------
    written, medians = [], {}
    n_ok = n_ref = 0
    marked = False

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

    lease = x2_lease(x2, lst, sb, u.name)
    try:
        lease.acquire(say=u.note)
        old = T.residuals_on_disk(x2, lst, sb, {night})           # again, now that no other unit of the subband writes
        why = log_gate(log, lst, sb)
        if old and not args.fresh:
            why = f'G-C5: {len(old)} x2 residual(s) of {cell} {sb} appeared under {x2} while the aligner ran'
        if why:
            raise Refusal(why)
        if old:
            # the log says so BEFORE the residuals move: from here to the merge below the subband has no final line
            merge_log(log, lst, sb, night, [], marker=f'earlier residuals moved aside (--supersede {args.supersede}) by '
                                                      f'{u.name}, the new ones not written and judged yet')
            marked = True
            moves = C.supersede(sorted(old.values()), run.out_root, run.out_root, args.supersede, f'flow {u.name} --fresh')
            u.note(f'--fresh: {len(moves)} earlier residuals moved aside under {run.out_root}/_superseded/{args.supersede}/')
        with ThreadPoolExecutor(max_workers=max(1, min(len(ks), WINDOW_THREADS))) as ex:
            futures = [ex.submit(window_residuals, k) for k in ks]       # every thread has ended when the block closes
        for k, fu in zip(ks, futures):
            fmed = fu.result()                                            # the first failure, raised here
            if fmed is None:
                night_lines.append(f'REFUSED-NIGHT {sb} w{k} {night}: no flow returned')
                n_ref += 1
                continue
            medians[k] = round(fmed, 4)
            n_ok += 1
            night_lines.append(f'ALIGNED {lst} {sb} w{k} {night} median_flow={fmed:.3f} px')
        written.sort()
        timing['residuals'] = round(time.time() - t0 - sum(timing.values()), 1)
        for ln in night_lines:
            u.note(ln)
        SS.verdict(u, run, args, lst, sb, [night], rc_child=3 if n_ref else 0)
        mine = {os.path.abspath(p) for p in written}
        failed = [t for t in u.failed_tests() if t.get('path') in mine]
        if failed:
            raise Refusal(f'G-C8: {len({t["path"] for t in failed})} of {len(written)} residual(s) failed the content test, '
                          f'e.g. {os.path.basename(failed[0]["path"])}: {failed[0]["name"]} '
                          f'({failed[0].get("detail") or failed[0].get("value")})')
        timing['verdict'] = round(time.time() - t0 - sum(timing.values()), 1)
        window_lines = [window_line(sb, S, k, kind) for (S, k), kind in sorted(left.items())]
        final, markers = merge_log(log, lst, sb, night, window_lines + night_lines)
    except BaseException as e:
        # a unit that is refused or fails here leaves NO residual of its own behind (the wideband, tree and search stages
        # take the night's residuals from the directory): the files this process wrote, and its own unfinished temporary
        # files; the tests stay in the record. The log keeps the marker when earlier residuals were moved aside.
        tmp = [os.path.join(os.path.dirname(p), f'.{os.path.basename(p)}.tmp.{os.getpid()}')
               for S in ('I', 'V') for p in (T.residual_path(x2, lst, night, sb, S, k) for k in ks)]
        gone = [p for p in written + tmp if os.path.lexists(p)]
        for p in gone:
            os.unlink(p)
        mine = {os.path.abspath(p) for p in written}
        u.outputs[:] = [o for o in u.outputs if o['path'] not in mine]
        if written:
            u.note(f'{type(e).__name__} after {len(written)} residual(s) were written: every residual this unit wrote was '
                   f'removed' + (f'; the refine log {log} carries the line {INCOMPLETE} {sb} {night} and no final line '
                                 f'until the unit has run again' if marked else ''))
        raise
    finally:
        lease.release()
        clean_work()
    if final is None:
        u.note(f'the refine log {log} has no final SUBLST-REFINE line: another night is marked incomplete ({markers[0]}); '
               f'the frames tier\'s refine gate refuses the subband until that night\'s unit has run again')
    else:
        u.note(final)
    u.meta['refine'] = dict(rc=3 if n_ref else 0, log=log, final=final, lock=lock, gpu=g, windows=ks,
                            flow_median_px=medians, residuals=len(written), reference='installed window Master',
                            windows_left={f'{S} w{k}': kind for (S, k), kind in sorted(left.items())},
                            night_lines=night_lines, lease_waited_s=round(lease.waited_s, 1), timing_s=timing)
    print(f'  {u.name}: {len(written)} residual(s) of {n_ok} window(s) against the installed window Masters'
          + (f', {n_ref} window(s) without a flow' if n_ref else ''))


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


def resume_valid(run, base, lst, night, sb, plan_files, name):
    """--resume: (True, why) when the unit's record is done and still describes what an assessment finds NOW: no gate
    refuses, every input an assessment names is in the record with the same fingerprint (a new window stack, a Master
    installed since, a changed exclusion list or plan all show), the admitted night, the code and the config are
    unchanged, the residuals on disk are exactly the record's outputs and each passes its content test."""
    from exopipe.phase3.core import record
    from exopipe.phase3.lib.flow import tests as T
    from exopipe.phase3.lib.masters import sublst_nonmember as NM
    prev = record.load_record(run.out_root, 'flow', name)
    if not prev or prev.get('status') != 'done':
        return False, 'no done record'
    try:
        why, a = assess(run, base, lst, night, sb)
    except Refusal as e:
        why, a = str(e), None
    if not why:
        new = sorted({os.path.abspath(p) for p in NM._inputs(a, plan_files)} - set(prev.get('inputs', {})))
        if new:
            why = f'an input the record does not name: {new[0]}'
    if not why:
        have = {os.path.abspath(p) for p in T.residuals_on_disk(os.path.join(run.out_root, f'{lst}_x2'), lst, sb,
                                                                 {night}).values()}
        if have != {os.path.abspath(o['path']) for o in prev.get('outputs', [])}:
            why = 'the residuals on disk are not the record\'s outputs'
    if not why:
        ok, why = run.resume_check(name, inputs=None, admitted=[f'{night}/{sb}'])
        if ok:
            return True, why
    print(f'flow sublst {name}: not resumed ({why})')
    return False, why


def run_stage(args, run, its, base):
    for lst, night, sbs in by_night(its):
        try:
            units, files, how = night_units(run, args, lst, night, sbs)
        except Refusal as e:
            with run.unit(f'{STAGE}_{lst}_{night}', tier='sublst', lst=lst, stage=STAGE) as u:
                u.refuse(str(e) if str(e).startswith('N-') else f'N-2: {e}')
            continue
        if not units:
            with run.unit(f'{STAGE}_{lst}_{night}', tier='sublst', lst=lst, stage=STAGE):
                raise NoInput(f'no stack job of the night\'s plan in this selection: nothing to align ({how})')
            continue
        for sb in units:
            name = unit_name(lst, sb, night)
            if run.resume and not run.fresh and resume_valid(run, base, lst, night, sb, files, name)[0]:
                u = run.unit(name, tier='sublst')
                u.status, u.reason = 'skipped-valid', 'record done; inputs, admitted night, code, config unchanged; ' \
                                                      'outputs pass their content tests'
                print(f'flow sublst {name}: skipped (--resume)')
                continue
            try:
                with run.unit(name, tier='sublst', lst=lst, subband=sb) as u:
                    refine_unit(u, args, run, lst, sb, night, base, files)
            except Exception as e:
                print(f'flow sublst {name}: error {type(e).__name__}: {e}')


def plan_stage(args, run, its, base):
    """Dry run: what each unit would do; returns 2 when a gate would refuse."""
    rc = 0
    for lst, night, sbs in by_night(its):
        try:
            units, _files, how = night_units(run, args, lst, night, sbs)
        except Refusal as e:
            print(f'[plan] {STAGE} {lst}_{night}: WOULD REFUSE - {e if str(e).startswith("N-") else "N-2: " + str(e)}')
            rc = 2
            continue
        print(f'[plan] {STAGE} {lst}_{night}: {len(units)} unit(s) of {len(sbs)} selected subband(s) (plan: {how}); installed '
              f'window Masters under {os.path.join(base, C.cfg().flow_refine.MASTERS_DIR.format(lst=lst))}')
        for sb in units:
            name = unit_name(lst, sb, night)
            if run.resume and not run.fresh and resume_valid(run, base, lst, night, sb, _files, name)[0]:
                print(f'[plan]   {name}: valid by its record, WOULD BE SKIPPED (--resume)')
                continue
            g = '?'
            from exopipe.phase3.lib.masters import sublst_nonmember as NM
            member = None if NM._exclusion_gate(lst, sb, night)[1] else NM.master_member(base, lst, sb, night)
            if member:
                print(f'[plan]   {name}: NO-INPUT - {NM.member_why(lst, night, sb, member)}')
                continue
            try:
                why, a = assess(run, base, lst, night, sb)
                if not why:
                    why, g = early_gates(args, run, lst, sb, night)[:2]
            except Refusal as e:
                why, a = str(e), None
            if why:
                print(f'[plan]   {name}: WOULD REFUSE - {why}')
                rc = 2
                continue
            ks, items, left = windows_to_align(a)
            n = sum(1 for (S, k) in items if (S, k) not in left)
            print(f'[plan]   {unit_name(lst, sb, night)}: {len(ks)} window(s) to one aligner process (GPU {g}), {n} '
                  f'residual(s) to write' + (f'; left out: {sorted(f"{S} w{k}" for S, k in left)}' if left else ''))
    return rc
