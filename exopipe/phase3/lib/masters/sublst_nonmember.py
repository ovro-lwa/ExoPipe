"""exopipe3 frames --tier sublst --stages nonmember - the 10-minute (sub-LST) residual of ONE night that is NOT in the
window Masters (the per-hour arm A; the batch arm A is `masters --tier sublst --stages loo`).

Ruling: Gregg 2026-11-12 - nights that are not in a Master are searched as non-members against the frozen Masters
("Approved."). The arithmetic is production's for a night of weight 0 (clv15/phase3_sublst_loo.py since clv15 5b1fc5b,
Gregg 2026-11-11: a listed night without a co-add weight "is subtracted against the Master of the weighted nights"), with
the Master READ FROM ITS INSTALLED FILE instead of being recomputed from every night's stack.

Words used here. Window stack: the image of one of the six 10-minute sidereal windows of an LST hour on one night (the
weighted mean of its 10-s frames with the deep flux scale baked in, Jy/beam; written by the stack stage). Window Master:
the night-weighted mean of one window over its member nights (Jy/beam; weight = the night's weight in the frozen deep
co-add record; written as float32 by the batch stage). Member (constituent): a night with a weight above 0 in the Master's
weights record. Non-member: any other night - one the record lists with weight 0 (rejected by the deep co-add, or listed
as a non-member when the Master was built) or one the record does not hold at all (a night newer than the Master).

Unit of work: one (LST hour, night, subband), Stokes I and V: unit nonmember_<LST>_<night>_<sb>, one per subband of the
night's job plan (the admission of the stack stage decides which subbands a night owes; this stage changes none of it).
Per window k and Stokes S of the night:
    residual = window stack - installed window Master         (float64 difference of the two float32 images, stored float32)
    NaN where the stack or the Master has no data. No other night's stack is read; no leave-one-out is needed, because the
    night is not in the Master. The Jupiter mask of the Masters (per member night, applied when the Master was built) is in
    the installed pixels already; the night's own residual keeps Jupiter, as in production.

Inputs (read only; fingerprinted in the unit record):
  the night's job plan: the done admit record admit_<LST>_<night> of this run root (step frames or masters) and its plan
      files <out>/_plans/sublst_jobs_<LST>_<night>[_deepflow].txt;
  the night's stack: <root>/<LST>/<LST>_<night>/<sb>/<S>/sublst_<LST>_<night>_<sb>_<S>_w<k>.fits with its _nframes and _pb
      siblings and admission_<LST>_<night>_<sb>_<S>.csv, and the evidence that the stack jobs completed
      (masters_sublst.NONMEMBER_STACK_EVIDENCE: 'record' = the done stack record stack_<LST>_<night>_<sb> of this run root;
      'log' = the stacker's own completion line in <root>/<LST>/logs/<LST>_<night>_<sb>_<S>.log, for parity runs on
      production stacks, which have no record);
  the installed window Masters: <root>/<LST>_masters/<sb>/<S>/sublst_master_<LST>_<sb>_<S>_w<k>.fits + _weights.csv
      (night, weight);
  data_decisions.csv through core.exclusions and the co-add's own reader (the two readers the stack stage uses).
Outputs (production's names; the header is the stack's own header - its own time block - plus production's residual
cards, in production's order: SUBTRACT, LOONCOAD = the Master's weighted nights, LOOMSTR, CODE, JUPMASK = the Master's
card, LSTBIN, DURATION; no BINMSTR):
  <root>/<LST>/<LST>_<night>/<sb>/<S>/sublst_<LST>_<night>_<sb>_<S>_w<k>_resid_loo.fits   [sublst_nonmember_residual]
  run records <out>/_records/<caller's step>/nonmember_<LST>_<night>_<sb>.json
The non-member mark: production's residual carries no card of its own for it. The mark is the stack's NONMEMB card
(written by the stacker, inherited here through the header copy) and, for this stage's products, the product class and
the Master's weights record, which does not weight the night. No card is invented here.

Failure modes (exit 2 = refused, the gate named in the record; nothing is written by a refused unit):
  N-1  the night is excluded for the subband in data_decisions.csv, or the two exclusion readers disagree (G-M01).
  N-2  no done admit record of the night under this run root (the stack stage has not run), or its plan files changed.
  N-3  a stack job of the night's plan is not complete by the evidence (G-S05: record not done, or no completion line).
  N-4  the night's stack is incomplete on disk: no admission record, the admission record and the window files disagree
       (a window missing or extra, NFRAMES), no window at all, a sibling (_nframes, _pb) missing, or the windows on disk
       are not those of the stack record / completion line.
  N-5  the night IS a member (weight above 0 in an installed window Master): its residual needs the leave-one-out path
       of the batch stage `masters --tier sublst --stages loo` (unit loo_<LST>_<sb>).
  N-6  no installed window Master of the (LST, subband, Stokes) at all: `masters --tier sublst --stages loo` writes them.
  N-7  the installed Master of ONE window is missing where the night has a stack: that window's residual is an item
       refusal (production's batch writes no residual for a window without a Master either); the unit completes with
       the other windows. When NO window of the night has a Master the unit is refused as a whole, before anything is
       moved or written.
  N-8  an installed Master cannot be subtracted: no weights record; its header disagrees with its weights record by the
       Master's own content test (tests_sublst.master_record_problems: COADDOP of another generation, LSTWIN, NCOADD,
       CONSTDIG, the HISTORY lists of constituents and of nights not in the Master, fewer than
       masters_sublst.LOO_MIN_WEIGHTED_NIGHTS weighted nights, no JUPMASK card); another image shape; or a constituent
       excluded in data_decisions.csv now (the Master must be rebuilt by the batch stage).
  N-9  the stack and the Master carry different flux scales (baked flag, pixel factor): production refuses such a window;
       or the stack's pixels do not carry the flux correction (FLUXBAKE is not True).
  N-10 the stack's own time block (DATE-OBS, DATE-END) cannot be read: no DURATION is assumed.
  N-11 residuals of an earlier run where this unit writes, without --fresh --supersede TAG (G-S01).
  N-12 a written residual fails its content test (class sublst_nonmember_residual: the pixel identity, exact; never
       waivable): every residual the unit wrote is removed and the unit is refused.
With --fresh --supersede TAG the earlier residuals are moved to <out>/_superseded/<TAG>/ (sha1 manifest) once every gate
N-1 to N-10 has passed; a unit that then fails while writing, or at N-12, leaves no residual of its own and the earlier
ones under _superseded/<TAG>/ (as every stage with --supersede).
Exit 3 (no input, never done): the plan holds no job of the night in this selection (the admission left every subband
out by rule; the reasons are in the admit record). Exit 1: an error (for example an unreadable FITS file).

Per-hour interface (frames --tier sublst imports this, as it imports tier_sublst.stack_night):
  nonmember_night(run, lst, night, sublst_root=None, subbands=None, stokes=('I', 'V'), workers=None, fresh=False,
                  supersede=None, resume=False, stack_planned=False) -> {'units', 'would_refuse'}
"""
import csv
import glob
import os
import re

from exopipe.phase3.core import record
from exopipe.phase3.core.record import NoInput, Refusal
from exopipe.phase3.lib.masters import tier_sublst as TS

STAGE = 'nonmember'
PRODUCT_CLASS = 'sublst_nonmember_residual'
BATCH_STAGE = 'masters --tier sublst --stages loo'      # writes the window Masters and the member (leave-one-out) residuals
STACK_STAGE = 'frames --tier sublst --stages stack'     # writes a night's window stacks (admit + stack)
STACK_EVIDENCE = ('record', 'log')
CODE_COMMENT = 'exopipe3 lib/masters/sublst_nonmember.py'


# =====================================================================================================================
# names
# =====================================================================================================================

def unit_name(lst, night, sb):
    return f'nonmember_{lst}_{night}_{sb}'


def master_products(root, lst, sb, S, k):
    """(window Master, its weights record) of window k under the installed <root>/<LST>_masters/ (the batch stage's names,
    clv15/phase3_sublst_loo.py:153-156)."""
    base = os.path.join(root, f'{lst}_masters', sb, S, f'sublst_master_{lst}_{sb}_{S}_w{k}')
    return base + '.fits', base + '_weights.csv'


def residual_path(window):
    """The arm-A residual of a window stack: the stack's name with _resid_loo (clv15/phase3_sublst_loo.py:185)."""
    return window[:-5] + '_resid_loo.fits'


def stack_evidence_kind():
    how = TS._sec().NONMEMBER_STACK_EVIDENCE
    if how not in STACK_EVIDENCE:
        raise Refusal(f'masters_sublst.NONMEMBER_STACK_EVIDENCE must be one of {STACK_EVIDENCE}, not {how!r}')
    return how


# =====================================================================================================================
# the arithmetic and the header (production's lines for a night of weight 0)
# =====================================================================================================================

def subtract(stack, master):
    """residual (float32) = stack - Master, formed in float64 from the two images as stored; NaN where either has no data.
    clv15/phase3_sublst_loo.py:165-168 for a night of weight 0: there the leave-one-out mean IS the window mean (the night
    added nothing to the sum or the weight) and the residual is NaN where the night or the mean has no data. The one
    difference: production subtracts the mean before it is rounded to float32 for the Master file; here the Master is the
    installed float32 file."""
    import numpy as np
    d = np.asarray(stack, dtype=np.float64)
    m = np.asarray(master, dtype=np.float64)
    if d.shape != m.shape:
        raise Refusal(f'N-8: window stack shape {d.shape} != window Master shape {m.shape}')
    return np.where(np.isfinite(d) & np.isfinite(m), d - m, np.nan).astype(np.float32)


def duration_s(header):
    """DURATION of a window (s): its last frame's end minus its first frame's start, from the stack's own time block
    (DATE-END - DATE-OBS, rounded to 0.1 s; clv15/phase3_sublst_loo.py:180). An unreadable time block refuses (N-10):
    production falls back to TEFFSEC or 600 s there; an epoch product carries its own time, so nothing is assumed here."""
    try:
        from astropy.time import Time
        return round((Time(header['DATE-END']) - Time(header['DATE-OBS'])).sec, 1)
    except Exception as e:
        raise Refusal(f'N-10: the stack\'s own time block (DATE-OBS, DATE-END) cannot be read ({type(e).__name__}: {e}) - '
                      f'no DURATION is assumed')


def residual_header(stack_header, master_header, master_path, k, n_weighted, code):
    """The residual's header: the stack's own header (its own time block) plus the cards of clv15/phase3_sublst_loo.py
    :169-184, same keywords, values, comments and order. n_weighted = the Master's weighted nights (LOONCOAD: the nights in
    the Master this night was subtracted against; for a non-member none is left out). code = this code's commit (CODE)."""
    rh = stack_header.copy()
    rh['SUBTRACT'] = ('leave-one-out window Master', 'sub-LST arm A1: cube-data Master, this night left out')
    rh['LOONCOAD'] = (int(n_weighted), 'nights in the leave-one-out Master')
    rh['LOOMSTR'] = (os.path.basename(master_path)[:68], 'central window Master (all nights)')
    rh['CODE'] = (str(code)[:68], CODE_COMMENT)
    rh['JUPMASK'] = (master_header['JUPMASK'], 'arcmin, Jupiter masked per night in the leave-one-out Master')
    # cards the 10-minute search reads: LSTBIN selects the window PB and names the product; BINMSTR must not be present
    rh['LSTBIN'] = (int(k), 'sidereal window index (sub-LST)')
    rh['DURATION'] = (duration_s(rh), 's, first frame start to last frame end')
    if 'BINMSTR' in rh:
        del rh['BINMSTR']
    return rh


def write_residual(item, code):
    """Read the night's window stack and the installed window Master of one item of assess(), write the residual."""
    import numpy as np
    from astropy.io import fits
    from exopipe.phase3.core import fitsio
    with fits.open(item['stack'], memmap=False) as h:
        d = np.asarray(h[0].data, dtype=np.float64)
        hh = h[0].header.copy()
    with fits.open(item['master'], memmap=False) as h:
        m = np.asarray(h[0].data, dtype=np.float64)
        mh = h[0].header.copy()
    r = subtract(d, m)
    rh = residual_header(hh, mh, item['master'], item['k'], item['n_weighted'], code)
    return fitsio.write_fits(item['resid'], r, rh, overwrite=False)


# =====================================================================================================================
# the night's plan and stack (the stack stage's admission and completion; nothing of it is changed here)
# =====================================================================================================================

def night_plan(run, opts, lst, night, planned_now=False):
    """(jobs [(cell, subband, [Stokes], cube root)], plan files, how) of ONE night: the plan of its admit record
    admit_<LST>_<night>, read with tier_sublst.load_plan (sha1 of the plan files checked). The admit record is under the
    step that stacked the night - frames (per hour, stack_night) or masters (--cells <LST>_<night> --stages admit,stack);
    the newest wins, as tier_sublst.stack_record does for the stack records."""
    key = f'{lst}_{night}'
    best = None
    for step in dict.fromkeys((run.step,) + TS.STACK_RECORD_STEPS):
        rec = record.load_record(run.out_root, step, f'admit_{key}')
        if rec is not None and (best is None or str(rec.get('finished', '')) > str(best[1].get('finished', ''))):
            best = (step, rec)
    try:
        return TS.load_plan(run, opts, lst, key, planned_now, step=best[0] if best else None)
    except Refusal as e:
        raise Refusal(f'N-2: no job plan of the night {key}: {e} (the night\'s stack comes first: `{STACK_STAGE} --epochs '
                      f'{key}`)')


def _completion(log, cell, sb, S):
    """The stacker's last completion line 'SUBLST <cell> <sb> <S> ... windows=<n>' in its log (production's own evidence
    of a complete stack job, sublst_stack_pass.sh:23), or None."""
    try:
        with open(log, errors='replace') as f:
            done = [l.strip() for l in f if re.match(rf'^SUBLST {re.escape(cell)} {re.escape(sb)} {S} ', l)]
    except OSError:
        return None
    return done[-1] if done else None


def stack_state(run, root, lst, cell, sb, stokes, how):
    """What the night's stack of one subband holds, and whether it is complete.
    Returns (problems [str], windows {Stokes: [(k, window path)]}, inputs [path], notes [str]). problems is empty only
    when, for every Stokes the plan owes (a Stokes whose stack job was refused and is admitted as refused by
    masters_sublst.LOO_ADMITS_REFUSED_STACKS is not owed, as in the batch stage):
      - the stack job completed by the evidence ``how`` ('record': tier_sublst.stack_gate, G-S05; 'log': the stacker's
        completion line), and
      - the admission record agrees with the window files (content test admission_matches_windows of class
        sublst_admission: per window the used frames = NFRAMES, a window file exactly where frames were used), at least
        one window exists, every window has its _nframes and _pb siblings, and the windows on disk are those the
        evidence names (the record's sublst_window outputs; the completion line's windows= count)."""
    from exopipe.phase3.lib.masters import tests_sublst as TST
    night = cell.split('_', 1)[1]
    problems, windows, inputs, notes = [], {}, [], []
    owed = list(stokes)
    rec = None
    if how == 'record':
        bad, refused = TS.stack_gate(run, lst, sb, [(cell, sb, list(stokes), None)])
        if bad:
            problems.append(f'N-3 (G-S05): stack job(s) not complete by their record: {bad}')
        if refused and not bad:
            notes.append(f'stack jobs refused and admitted as refused (LOO_ADMITS_REFUSED_STACKS): {refused} - no residual '
                         f'owed for them')
        owed = [S for S in stokes if (cell, S) not in refused]
        rec = TS.stack_record(run, lst, night, sb)
    for S in owed:
        P = TS.stack_products(root, lst, cell, sb, S)
        wins = sorted(glob.glob(os.path.join(P['dir'], f'sublst_{cell}_{sb}_{S}_w[0-9].fits')))
        windows[S] = [(int(w[:-5].rsplit('_w', 1)[1]), w) for w in wins]
        inputs += [P['admission']] + wins
        if how == 'log':
            line = _completion(P['log'], cell, sb, S)
            inputs.append(P['log'])
            if line is None:
                problems.append(f'N-3: no completion line "SUBLST {cell} {sb} {S} ..." in {P["log"]}')
            else:
                m = re.search(r'windows=(\d+)', line)
                if not m or int(m.group(1)) != len(wins):
                    problems.append(f'N-4: {len(wins)} window file(s) of Stokes {S} on disk, completion line: {line[:160]}')
        elif rec is not None and not problems:
            listed = sorted(o.get('path') for o in rec.get('outputs') or []
                            if o.get('class') == 'sublst_window' and os.path.dirname(str(o.get('path'))) == os.path.abspath(P['dir']))
            if listed != [os.path.abspath(w) for w in wins]:
                problems.append(f'N-4: the stack record lists {len(listed)} Stokes {S} window(s), {len(wins)} are on disk')
        if not os.path.isfile(P['admission']):
            problems.append(f'N-4: no admission record {P["admission"]}')
            continue
        try:
            ok, _value, detail = TST.admission_matches_windows(P['admission'])
        except Exception as e:                        # an unreadable admission record or window header: incomplete
            ok, detail = False, f'{type(e).__name__}: {e}'
        if not ok:
            problems.append(f'N-4: admission record and window files of Stokes {S} disagree: {str(detail)[:300]}')
        if not wins:
            problems.append(f'N-4: no window stack of Stokes {S} under {P["dir"]}')
        for w in wins:
            miss = [os.path.basename(w[:-5] + sfx) for sfx in ('_nframes.fits', '_pb.fits') if not os.path.isfile(w[:-5] + sfx)]
            if miss:
                problems.append(f'N-4: {os.path.basename(w)} lacks its sibling(s) {miss}')
    return problems, windows, inputs, notes


# =====================================================================================================================
# the installed window Masters
# =====================================================================================================================

def read_weights(path):
    """[(night, weight)] of a window Master's weights record (the batch stage's _weights.csv: columns night, weight -
    weight = the night's weight in the Master, 0 for a night that is listed and not a constituent)."""
    with open(path, newline='') as f:
        return [(r['night'], float(r['weight'])) for r in csv.DictReader(f)]


def master_problems(header, rows, k, ex, lst, sb):
    """Why an installed window Master cannot be subtracted (N-8), [] when it can: the Master's header against its own
    weights record, by the SAME function as the Master's content test (tests_sublst.master_record_problems: COADDOP,
    LSTWIN, NCOADD = the weighted nights, at least LOO_MIN_WEIGHTED_NIGHTS of them, CONSTDIG = md5 of the listed nights,
    HISTORY = the constituents and the nights not in the Master, a JUPMASK card), plus: at least PER_NIGHT_MIN weighted
    nights (the residual's LOONCOAD), and no constituent excluded now. The content test's comparison with the frozen deep
    record is the batch stage's: an installed Master stays the reference after the deep record has moved on."""
    from exopipe.phase3.lib.masters import tests_sublst as TST
    s = TS._sec()
    weighted = sorted(n for n, w in rows if w > 0)
    out = list(TST.master_record_problems(header, rows, k))
    if len(weighted) < int(s.PER_NIGHT_MIN):
        out.append(f'{len(weighted)} weighted night(s): fewer than the {s.PER_NIGHT_MIN} a residual needs (PER_NIGHT_MIN)')
    gone = [n for n in weighted if ex.is_excluded(lst, sb, n)]
    if gone:
        out.append(f'constituent night(s) {gone} are excluded in data_decisions.csv now (exclusion is absolute: the Master '
                   f'must be rebuilt)')
    return out


def flux_scale(header):
    """(baked flag, pixel factor rounded to 1e-6): the flux scale a 10-minute image carries (FLUXBAKE; SHSCALE x FLUXCORR),
    as clv15/phase3_sublst_loo.py:123 compares it between the stacks of one window. The window Master carries the cards
    of its first night's stack (loo:147-149), which production refused to differ from any other night's."""
    from exopipe.phase3.lib.masters import flux_correction as FC
    f = FC.pixel_factor(header)
    return (FC.pixels_carry_correction(header), None if f is None else round(f, 6))


# =====================================================================================================================
# assessment of one unit (read only: the plan of a dry run and the gates of a real run)
# =====================================================================================================================

def assess(run, root, lst, cell, sb, stokes, how, ex):
    """Everything one unit needs to know before it writes, without writing: {'refuse': the unit's refusal or None,
    'items': [{'S', 'k', 'stack', 'master', 'weights', 'resid', 'n_weighted', 'refuse', 'gate'}] one per window stack of the
    night (refuse/gate set for an item refusal, N-7), 'inputs': [path], 'notes': [str]}. ``ex``: the core Exclusions."""
    from astropy.io import fits
    night = cell.split('_', 1)[1]
    out = {'refuse': None, 'items': [], 'inputs': [], 'notes': []}
    problems, windows, inputs, notes = stack_state(run, root, lst, cell, sb, stokes, how)
    out['inputs'] += inputs
    out['notes'] += notes
    if problems:
        out['refuse'] = (f'the stack of {cell} {sb} is missing or incomplete (evidence: {how}): ' + '; '.join(problems[:6])
                         + f' - run `{STACK_STAGE} --epochs {cell}` first (with --fresh --supersede TAG over earlier '
                           f'products); no residual is written from a partial stack')
        return out
    member, bad_master, bad_scale, bad_time = [], [], [], []
    for S in sorted(windows):
        mdir = os.path.join(root, f'{lst}_masters', sb, S)
        if not glob.glob(os.path.join(mdir, f'sublst_master_{lst}_{sb}_{S}_w[0-9].fits')):
            out['refuse'] = (f'N-6: no installed window Master of {lst} {sb} Stokes {S} under {mdir}: the batch stage '
                             f'`{BATCH_STAGE}` writes them (unit loo_{lst}_{sb}); a night is never subtracted without '
                             f'its installed Master')
            return out
        for k, w in windows[S]:
            mp, wp = master_products(root, lst, sb, S, k)
            item = {'S': S, 'k': k, 'stack': w, 'master': mp, 'weights': wp, 'resid': residual_path(w), 'n_weighted': None,
                    'refuse': None, 'gate': None}
            out['items'].append(item)
            if not os.path.isfile(mp):
                item['gate'] = 'N-7 no window Master'
                item['refuse'] = (f'N-7: no installed window Master {mp} for window {k} (the batch stage `{BATCH_STAGE}` '
                                  f'writes it; it writes none for a window with fewer than '
                                  f'{TS._sec().LOO_MIN_WEIGHTED_NIGHTS} weighted nights): no residual for this window')
                continue
            out['inputs'] += [mp, wp]
            if not os.path.isfile(wp):
                bad_master.append(f'{os.path.basename(mp)}: no weights record {os.path.basename(wp)}')
                continue
            rows = read_weights(wp)
            mh = fits.getheader(mp, 0)
            wn = dict(rows).get(night)
            if wn is not None and wn > 0:
                member.append(f'w{k} Stokes {S} (weight {wn:g})')
                continue
            if wn is not None:
                out['notes'].append(f'w{k} Stokes {S}: the Master\'s record lists {night} with weight 0 (not a constituent)')
            bad = master_problems(mh, rows, k, ex, lst, sb)
            sh = fits.getheader(w, 0)
            if (sh.get('NAXIS1'), sh.get('NAXIS2')) != (mh.get('NAXIS1'), mh.get('NAXIS2')):
                bad.append(f'image {mh.get("NAXIS1")} x {mh.get("NAXIS2")} px, the stack {sh.get("NAXIS1")} x {sh.get("NAXIS2")}')
            if bad:
                bad_master.append(f'{os.path.basename(mp)}: ' + '; '.join(bad))
                continue
            try:
                fs, fm = flux_scale(sh), flux_scale(mh)
            except RuntimeError as e:                         # FLUXBAKE without FLUXCORR (flux_correction.pixel_factor)
                bad_scale.append(f'w{k} Stokes {S}: {e}')
                continue
            if fs != fm:
                bad_scale.append(f'w{k} Stokes {S}: stack (baked, pixel factor) {fs} != Master {fm}')
                continue
            if sh.get('FLUXBAKE') is not True:               # the residual inherits the card; its content test requires it
                bad_scale.append(f'w{k} Stokes {S}: the stack\'s pixels do not carry the flux correction (FLUXBAKE '
                                 f'{sh.get("FLUXBAKE")!r})')
                continue
            try:
                duration_s(sh)
            except Refusal as e:
                bad_time.append(f'{os.path.basename(w)}: {e}')
                continue
            item['n_weighted'] = sum(1 for _n, x in rows if x > 0)
    if member:
        out['refuse'] = (f'N-5: {cell} {sb}: the night IS a member of the installed window Master(s) - ' + ', '.join(member[:12])
                         + f': its residual needs the leave-one-out path (the Master without this night), which the batch '
                           f'stage `{BATCH_STAGE}` writes (unit loo_{lst}_{sb}); this stage serves only nights that are not '
                           f'in the Master (Gregg 2026-11-12)')
    elif bad_master:
        out['refuse'] = (f'N-8: installed window Master(s) of {lst} {sb} cannot be subtracted: ' + ' | '.join(bad_master[:4])
                         + f' - rebuild them with the batch stage `{BATCH_STAGE}`')
    elif bad_scale:
        out['refuse'] = (f'N-9: {cell} {sb}: the stack and the installed window Master carry different flux scales, or the '
                         f'stack\'s pixels do not carry the flux correction: '
                         + '; '.join(bad_scale[:4]) + ' (one flux scale per window, Gregg 2026-10-07; production refuses the '
                                                      'window, clv15/phase3_sublst_loo.py:123-125)')
    elif bad_time:
        out['refuse'] = '; '.join(bad_time[:4])
    return out


# =====================================================================================================================
# the unit and the stage
# =====================================================================================================================

def _exclusion_gate(lst, sb, night):
    """N-1, the stack stage's own checks on a night (tier_sublst.stage_admit and _stack_unit): the two exclusion readers
    agree on (LST, subband, night) (G-M01, raises Refusal) and the night is not excluded. Returns (core Exclusions,
    refusal or None)."""
    ex, _ = TS.exclusion_readers_agree(lst, sb, [night])
    if ex.is_excluded(lst, sb, night):
        return ex, (f'N-1: {lst} {sb} {night} is excluded in data_decisions.csv (core.exclusions): exclusion is absolute - '
                    f'no residual of an excluded night')
    return ex, None


def _earlier(root, lst, cell, sb, stokes):
    """Every arm-A residual of the night's (subband, Stokes) on disk now: where this unit writes. One left by an earlier
    run where this run writes none would be searched as current (clv15/phase3_sublst_loo.py:190-196 refuses it)."""
    return sorted(p for S in stokes for p in glob.glob(os.path.join(root, lst, cell, sb, S,
                                                                    f'sublst_{cell}_{sb}_{S}_w[0-9]_resid_loo.fits')))


def _inputs(a, plan_files):
    """The inputs of one unit, from its assessment: the night's window stacks and admission records (and stack logs under
    the 'log' evidence), the installed window Masters and their weights records, the plan files, data_decisions.csv."""
    return sorted(p for p in set(a['inputs']) | set(plan_files) | {TS.cfg().paths.DATA_DECISIONS} if os.path.exists(p))


def _unit(run, opts, root, lst, cell, sb, stokes, plan_files, how):
    night = cell.split('_', 1)[1]
    name = unit_name(lst, night, sb)
    if opts.resume and not opts.fresh:
        # --resume: the unit's inputs are what an assessment NOW names (a new window stack, a Master installed since, a
        # changed exclusion list all change the set or a fingerprint); a unit that would refuse now is never skipped
        try:
            ex, excluded = _exclusion_gate(lst, sb, night)
            now = None if excluded else assess(run, root, lst, cell, sb, stokes, how, ex)
        except Refusal:
            now = None
        if now is not None and not now['refuse'] and TS.resume_skip(run, opts, name, _inputs(now, plan_files),
                                                                    [f'{night}/{sb}']):
            return
    with run.unit(name, lst=lst, cell=cell, subband=sb, stokes=list(stokes), stage=STAGE, stack_evidence=how) as u:
        ex, excluded = _exclusion_gate(lst, sb, night)
        if excluded:
            u.refuse(excluded)
        u.set_admitted([f'{night}/{sb}'])
        a = assess(run, root, lst, cell, sb, stokes, how, ex)
        for p in _inputs(a, plan_files):
            u.add_input(p)
        for n in a['notes']:
            u.note(n)
        if a['refuse']:
            u.refuse(a['refuse'])
        if not any(it['refuse'] is None for it in a['items']):
            # nothing to write: refused as a whole and BEFORE earlier residuals are moved aside (a refused unit changes
            # nothing under the product root)
            u.refuse(f'N-7: {cell} {sb}: no window of the night has an installed window Master ({len(a["items"])} window '
                     f'stack(s)): ' + '; '.join(it['refuse'] for it in a['items'][:3]))
        TS.clear_earlier(run, opts, u, _earlier(root, lst, cell, sb, stokes), f'{cell} {sb} non-member residuals')
        for it in a['items']:
            u.plan(it['resid'], PRODUCT_CLASS)
        mroot = os.path.join(root, f'{lst}_masters')
        written = []
        try:
            for it in a['items']:
                if it['refuse']:
                    u.refuse_item(it['resid'], it['refuse'], gate=it['gate'])
                    continue
                written.append(write_residual(it, run.commit))
        except BaseException:
            # a unit that fails while writing leaves NO residual of its own behind (a partial set would be read by the
            # wideband and search stages as the night's residuals): only the files THIS process wrote a moment ago, and
            # its own unfinished temporary file (core.fitsio.write_fits writes .<name>.tmp.<pid>, then renames)
            tmp = [os.path.join(os.path.dirname(it['resid']), f'.{os.path.basename(it["resid"])}.tmp.{os.getpid()}')
                   for it in a['items']]
            for p in written + tmp:
                if os.path.lexists(p):
                    os.unlink(p)
            raise
        for p in written:
            u.add_output(p, PRODUCT_CLASS, master_root=mroot)
        mine = {os.path.abspath(p) for p in written}
        failed = [t for t in u.failed_tests() if t.get('path') in mine]
        if failed:
            # N-12: the wideband, tree and search stages take the night's residuals from the directory, so a residual
            # that failed its content test must not stay there, and no partial set either: every residual this unit
            # wrote is removed (the tests stay in the record) and the unit is refused
            for p in written:
                if os.path.lexists(p):
                    os.unlink(p)
            u.outputs[:] = [o for o in u.outputs if o['path'] not in mine]
            u.refuse(f'N-12: {len({t["path"] for t in failed})} of {len(written)} residual(s) failed the content test, e.g. '
                     f'{os.path.basename(failed[0]["path"])}: {failed[0]["name"]} ({failed[0].get("detail") or failed[0].get("value")}); '
                     f'every residual this unit wrote was removed')
        n_written = len(written)
        u.meta.update(residuals=n_written, windows_refused=len(u.refused_items))
        TS.say(f'  {name}: {n_written} residual(s) against the installed window Masters'
               + (f', {len(u.refused_items)} window(s) refused (no installed Master)' if u.refused_items else ''))


def stage_nonmember(run, opts, sel, root, planned_now=False):
    """The non-member residuals of the selected nights (each named: --epochs / --cells <LST>_<night>). One unit per
    (night, subband) of the night's plan. Returns the number of dry-run refusals."""
    how = stack_evidence_kind()
    bad = 0
    for lst in sel.lsts:
        if lst not in sel.explicit:
            raise Refusal(f'the {STAGE} stage takes named nights (--epochs <LST>_<date>): {lst} was given without one; the '
                          f'residuals of every night of an LST are the batch stage `{BATCH_STAGE}`')
        for cell in sel.cells_of(lst):
            night = cell.split('_', 1)[1]
            key = f'{lst}_{night}'
            try:
                jobs, files, plan_how = night_plan(run, opts, lst, night, planned_now)
            except Refusal as e:
                if run.dry_run:
                    TS.say(f'[sublst {STAGE} {key}] WOULD REFUSE: {e}')
                    bad += 1
                    continue
                with run.unit(f'{STAGE}_{key}', lst=lst, cell=cell, stage=STAGE) as u:
                    u.refuse(str(e))
                continue
            if jobs is None:
                TS.say(f'[sublst {STAGE} {key}] jobs: {plan_how}')
                continue
            units = [(c, sb, [S for S in st if S in sel.stokes]) for c, sb, st, _cr in jobs
                     if c == cell and sb in sel.subbands and sel.scope_ok(c, sb)]
            units = [x for x in units if x[2]]
            if not units:
                if run.dry_run:
                    TS.say(f'[sublst {STAGE} {key}] NO-INPUT: no job of the night\'s plan in this selection ({plan_how})')
                    continue
                with run.unit(f'{STAGE}_{key}', lst=lst, cell=cell, stage=STAGE):
                    raise NoInput(f'no stack job of the night\'s plan in this selection: nothing to subtract ({plan_how})')
                continue
            if run.dry_run:
                TS.say(f'[sublst {STAGE} {key}] {len(units)} unit(s) (plan: {plan_how}; stack evidence: {how}); installed '
                       f'window Masters under {os.path.join(root, lst + "_masters")}')
                for c, sb, st in units:
                    try:
                        ex, why = _exclusion_gate(lst, sb, night)
                        a = None if why else assess(run, root, lst, c, sb, st, how, ex)
                        why = why or a['refuse']
                        old = _earlier(root, lst, c, sb, st)
                        if (not why and old and opts.resume and not opts.fresh
                                and run.resume_check(unit_name(lst, night, sb), inputs=_inputs(a, files),
                                                     admitted=[f'{night}/{sb}'])[0]):
                            TS.say(f'    {unit_name(lst, night, sb)}: valid by its record, WOULD BE SKIPPED (--resume)')
                            continue                         # what the real run does (tier_sublst.resume_skip)
                        if not why and old and not (opts.fresh and opts.supersede):
                            why = f'N-11 (G-S01): {len(old)} residual(s) of an earlier run, e.g. {old[0]} (no --fresh --supersede)'
                    except Refusal as e:
                        a, why = None, str(e)
                    if why:
                        bad += 1
                        TS.say(f'    {unit_name(lst, night, sb)}: WOULD REFUSE: {why}')
                    else:
                        n_ref = sum(1 for it in a['items'] if it['refuse'])
                        TS.say(f'    {unit_name(lst, night, sb)} Stokes {",".join(st)}: {len(a["items"]) - n_ref} residual(s) '
                               f'to write' + (f', {n_ref} window(s) WOULD BE REFUSED (no installed Master)' if n_ref else ''))
                continue
            TS.pool_map(lambda x: _unit(run, opts, root, lst, x[0], x[1], x[2], files, how), units,
                        opts.workers or TS._sec().PARALLEL[STAGE], run, lambda x: unit_name(lst, night, x[1]))
    return bad


# =====================================================================================================================
# per-hour interface (frames --tier sublst imports this; keep the signature stable)
# =====================================================================================================================

def nonmember_night(run, lst, night, sublst_root=None, subbands=None, stokes=('I', 'V'), workers=None, fresh=False,
                    supersede=None, resume=False, stack_planned=False):
    """The arm-A residuals of ONE night that is not in the installed window Masters of its LST hour (DESIGN 12.1 per-hour
    work): one unit nonmember_<LST>_<night>_<sb> per subband of the night's plan, records under the caller's step.

    run          the caller's record.Run (its step names the records; its dry_run prints the plan and writes nothing)
    lst, night   '09h', '2024-12-29'
    sublst_root  the sub-LST product root (default run.out_root): holds <LST>/ (the night's stacks) and <LST>_masters/
    subbands     subset of instrument.ALL_SUBBANDS (default all); stokes ('I', 'V') or one of them
    workers      units at once (default masters_sublst.PARALLEL['nonmember'])
    fresh, supersede, resume   as the common flags (--fresh --supersede TAG moves earlier residuals; --resume by record)
    stack_planned  the caller runs the stack stage in the same invocation (a dry run then has no plan to show yet)
    Returns {'units': [unit names of this night in this run], 'would_refuse': dry-run refusals}. Raises record.Refusal (dry
    run too) when an in-place literal differs from its config mirror or a never-waivable test is waived - the start-up
    checks of `masters --tier sublst`, as tier_sublst.stack_night makes them (the Masters are the in-place loo's)."""
    from exopipe.phase3.core import cells as CL
    CL.parse_epoch(f'{lst}_{night}')
    TS.clear_p3_env()
    viol = TS.mirror_violations()
    if viol or TS.waiver_problem(run):
        raise Refusal(('config differs from the in-place code it mirrors: ' + '; '.join(viol)) if viol
                      else TS.waiver_problem(run))
    opts = TS.Opts(fresh=fresh, supersede=supersede, resume=resume, workers=workers)
    sel = TS.Selection.one_night(lst, night, subbands, stokes)
    bad = stage_nonmember(run, opts, sel, sublst_root or run.out_root, planned_now=bool(stack_planned))
    return {'units': [u.name for u in run.units if u.name.startswith(f'{STAGE}_{lst}_{night}')], 'would_refuse': bad}
