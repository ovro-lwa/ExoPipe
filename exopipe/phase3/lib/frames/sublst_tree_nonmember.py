"""exopipe3 frames --tier sublst --stages tree-nonmember - the entry of ONE night that is NOT in the window Masters in the
tree the 10-minute search reads (the per-hour form of the stage `tree`, which runs clv15/phase3_sublst_tenmin_tree.py in
place and rebuilds the tree of an (LST hour, band) from every night's residuals and window primary-beam maps).

Ruling: Gregg 2026-11-12 - nights that are not in a Master are searched as non-members against the frozen Masters
("Approved."). The tree of a band, <root>/tenmin_root_x2/<LST>_<band>/, is what the search reads:
  subtracted/<night>T<kk>_{I,V}_sub_<band>.fits   links to the x2-aligned window residuals (one epoch per night and window)
  frames.csv                                      one row per link (the frame record: beam, flux cards, the link)
  bin_masters/PB_bin<k>_<band>.fits               the window's primary-beam map: the co-add-weighted mean of the presented
                                                  nights' window primary-beam maps
What a night of co-add weight 0 adds to it, by production's builder (phase3_sublst_tenmin_tree.py:95-145):
  - two links per window that has both Stokes, and their two rows in frames.csv, in the builder's order (night, window,
    Stokes I then V);
  - nothing in the pixels of PB_bin<k>: the night's map enters the weighted mean with weight 0. Its card PBNNIGHT (nights
    averaged) counts the night, and its header is the window primary-beam header of the EARLIEST presented night, so a
    night earlier than every presented night of a window also replaces that header.
This unit writes exactly that, and nothing else: it reads the night's own residual headers and window primary-beam
headers, the band's frames.csv and the PB_bin maps of the night's windows; no other night's file. The result is the tree
the builder writes from the same inputs (frames.csv byte for byte, the same links, PB_bin files byte for byte).

Unit of work: one (LST hour, band, night): unit <LST>_tree-nonmember_<band>_<night>. It is idempotent: a night that is
presented already, with these residuals, writes nothing (so --resume needs no skip: the unit re-reads the tree, seconds).
It writes in the order links, frames.csv, PB_bin headers; a run interrupted after frames.csv leaves the night's rows with
PBNNIGHT one short, and the next run of the same unit completes exactly that. One writer per band's tree at a time: the
lease <tree>/.leases/<LST>_<band> (the Lease class of lib/search/tier_sublst.py), which the batch stage `tree` takes too.
Production's own chain (sublst_run_lst_nocat.sh) runs the builder without it: never run this unit on a tree that chain
is rebuilding.

Inputs (read only): <root>/<LST>_x2/<LST>_<night>/<band>/<S>/sublst_<LST>_<night>_<band>_<S>_w<k>_resid_loo.fits (S = I,
V), <root>/<LST>/<LST>_<night>/<band>/I/sublst_<LST>_<night>_<band>_I_w<k>_pb.fits, the frozen deep weights of the band,
the non-member nights list, data_decisions.csv, the refine log <root>/<LST>_x2/logs/<LST>_<band>.log.
Outputs: the band's tree as above [tenmin_tree]: the night's links, frames.csv (replaced atomically), PB_bin<k> of the
night's windows (replaced atomically when a card changes).

Failure modes (exit 2 = refused; every gate is checked before anything is written):
  T-1  the (LST hour, band, night) is excluded in data_decisions.csv.
  T-2  the night has a co-add weight above 0 in the band (a member changes the PB_bin pixels: the batch stage `tree`),
       or no weight at all (not in the frozen deep weights and not on the non-member nights list).
  T-3  no x2 residual of the night, or no window with both Stokes.
  T-4  a residual without LSTBIN or DURATION, or with BINMSTR (the builder's refusal).
  T-5  a presented window without the night's window primary-beam map.
  T-6  the band has no tree to add to (no frames.csv, a frames.csv without rows or with other columns, NO_WINDOWS.txt):
       the batch stage builds it.
  T-7  the night is presented already with OTHER rows or links (residuals of another root): the batch stage rebuilds.
  T-8  a window of the night without a PB_bin map, with a PB_bin that is not a weighted mean (only non-member nights so
       far: one more would change its pixels), of another shape, or whose PBNNIGHT is not the Stokes I rows of the window
       in frames.csv (one short is accepted when the night's rows are in already: the interrupted run above).
  S-G8 the x2 refine of the band is not complete (its log), as the batch stage.
"""
import csv
import glob
import io
import os
import re

STAGE = 'tree-nonmember'
BATCH_STAGE = 'frames --tier sublst --stages tree'
# the builder's literals (phase3_sublst_tenmin_tree.py:137-140); tests/test_sublst_tree_nonmember.py compares them with
# the script's text where the script exists
PBSRC_WEIGHTED = ('sub-LST window PB, night-weighted mean', 'phase3_sublst_tenmin_tree.py')
PBNNIGHT_COMMENT = 'nights averaged'
LEASE_SUBDIR = '.leases'    # <tree>/.leases/<LST>_<band>: beside the tree it guards, outside every '<LST>_*' listing
LEASE_POLL_S = 2            # s between attempts to take the band tree's writer lease (the unit holds it for seconds)
LEASE_WAIT_S = 900          # s before a unit gives up waiting for it (error, retryable); a batch build holds it ~1 min
# cards the FITS writer derives from the pixels: never compared (a window primary-beam header may have 4 axes or 64 bits)
STRUCTURAL = ('SIMPLE', 'BITPIX', 'EXTEND')


def unit_name(lst, band, night):
    return f'{lst}_{STAGE}_{band}_{night}'


def cell_dir(tree, lst, band):
    return os.path.join(tree, f'{lst}_{band}')


def night_residuals(x2, lst, night, band):
    """{window: {Stokes: path}} of the night's x2 residuals of the band, the paths spelled as the builder's own listing
    spells them (tree:64-67: its --resid-root joined with the night directory)."""
    pat = re.compile(r'sublst_%s_%s_%s_([IV])_w(\d+)_resid_loo\.fits$' % (re.escape(lst), re.escape(night), re.escape(band)))
    found = {}
    for p in glob.glob(os.path.join(x2, f'{lst}_{night}', band, '?', 'sublst_*_resid_loo.fits')):
        m = pat.search(os.path.basename(p))
        if m:
            found.setdefault(int(m.group(2)), {})[m.group(1)] = p
    return found


def weights(lst, band):
    """({night: co-add weight}, the frozen deep weights file) as the builder reads them (tree:95-104): the rows of the
    frozen record with a weight, and the listed non-member nights it lacks at weight 0."""
    from exopipe.phase3.lib.masters import tier_sublst as TS
    wp = TS.frozen_weights(lst, band)
    wts = {}
    if os.path.exists(wp):
        with open(wp, newline='') as f:
            for r in csv.DictReader(f):
                if r.get('weight') not in (None, ''):
                    wts[r['date']] = float(r['weight'])
    for n in TS.nonmember_nights(lst) - set(wts):
        wts[n] = 0.0
    return wts, wp


def row_line(cols, row):
    """One frames.csv line (with its line end) as the builder's csv.DictWriter writes it."""
    buf = io.StringIO()
    csv.DictWriter(buf, fieldnames=list(cols)).writerow(row)
    return buf.getvalue()


def night_rows(cols, sdir, band, night, found, ks):
    """[(sort key, csv line, link, residual)] of the night's presented windows, the builder's row (tree:119-128)."""
    from astropy.io import fits
    out = []
    for k in ks:
        key = '%sT%02d' % (night, k)
        for i, s in enumerate(('I', 'V')):
            src = found[k][s]
            h = fits.getheader(src)
            link = os.path.join(sdir, '%s_%s_sub_%s.fits' % (key, s, band))
            row = {'key': key, 'date': night, 'bin': k, 'stokes': s, 'src': src, 'dc': h.get('FLUXDC', ''),
                   'has_flow': 1, 'src_dewarped': '', 'pb_blank_frac': '', 'pb_ratio_median': '',
                   'bmaj_deg': h.get('BMAJ', ''), 'bmin_deg': h.get('BMIN', ''), 'bpa_deg': h.get('BPA', ''),
                   'n_inject': 0, 'sub_path': link, 'fluxbake': int(bool(h.get('FLUXBAKE', False)))}
            out.append(((night, k, i), row_line(cols, row), link, src))
    return out


def read_frames(path, cols):
    """(refusal or None, [(sort key, raw line with its line end, row dict)]) of an existing frames.csv; the raw lines are
    kept so that the rows of the other nights are written back byte for byte."""
    with open(path, newline='') as f:
        text = f.read()
    lines = text.splitlines(keepends=True)
    rows = list(csv.DictReader(io.StringIO(text, newline='')))
    if not lines or lines[0].rstrip('\r\n').split(',') != list(cols):
        return f'{path}: its columns are not the tree writer\'s ({", ".join(cols)})', []
    if len(lines) != len(rows) + 1:
        return f'{path}: {len(lines) - 1} line(s) for {len(rows)} row(s) (a field with a line break)', []
    out = []
    for line, r in zip(lines[1:], rows):
        try:
            out.append(((r['date'], int(r['bin']), ('I', 'V').index(r['stokes'])), line, r))
        except (KeyError, ValueError, TypeError):
            return f'{path}: a row without a usable date, bin and stokes: {line.strip()[:120]}', []
    return None, out


def _cards(header):
    """The cards as the FITS writer writes them (a comment too long for its card is cut there), the structural ones
    apart."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return [str(c) for c in header.cards if c.keyword not in STRUCTURAL and not c.keyword.startswith('NAXIS')]


def assess(lst, band, night, x2, root, tree, cols, excluded):
    """What the unit would do, without writing: dict(refuse, action ('present' or 'already presented'), ks, skipped,
    lines (the new frames.csv, or None when unchanged), links [(link, residual)] to create, pb {window: target header}
    to write, inputs, weight, n_rows)."""
    from astropy.io import fits
    a = dict(refuse=None, action=None, ks=[], skipped=[], lines=None, links=[], pb={}, inputs=[], weight=None, n_rows=0)
    cdir = cell_dir(tree, lst, band)
    sdir, bdir, fcsv = os.path.join(cdir, 'subtracted'), os.path.join(cdir, 'bin_masters'), os.path.join(cdir, 'frames.csv')
    what = f'{lst}_{night} {band}'
    if excluded:
        a['refuse'] = f'T-1: {what}: excluded in data_decisions.csv - never presented to the search'
        return a
    wts, wp = weights(lst, band)
    a['inputs'].append(wp)
    if night not in wts:
        a['refuse'] = (f'T-2: {what}: no co-add weight - the night has no weight in {wp} and is not on the non-member '
                       f'nights list (the builder refuses it too)')
        return a
    a['weight'] = wts[night]
    if wts[night] != 0.0:
        a['refuse'] = (f'T-2: {what}: the night has the co-add weight {wts[night]:g} in {wp} - a member night changes the '
                       f'pixels of the window primary-beam maps: `{BATCH_STAGE}` rebuilds the band\'s tree')
        return a
    found = night_residuals(x2, lst, night, band)
    ks = sorted(k for k, st in found.items() if set(st) == {'I', 'V'})
    a['skipped'] = ['%s w%d (only %s)' % (night, k, ''.join(sorted(st))) for k, st in sorted(found.items())
                    if set(st) != {'I', 'V'}]
    if not ks:
        a['refuse'] = (f'T-3: {what}: no window with an x2 residual of both Stokes under '
                       f'{os.path.join(x2, f"{lst}_{night}", band)} ({len(found)} window(s) with one Stokes)')
        return a
    a['ks'] = ks
    night_pb = {}
    for k in ks:
        for s in ('I', 'V'):
            h = fits.getheader(found[k][s])
            if 'BINMSTR' in h or 'LSTBIN' not in h or 'DURATION' not in h:
                a['refuse'] = f'T-4: {found[k][s]} lacks LSTBIN/DURATION or carries BINMSTR'
                return a
            a['inputs'].append(found[k][s])
        pbp = os.path.join(root, f'{lst}_{night}', band, 'I', 'sublst_%s_%s_%s_I_w%d_pb.fits' % (lst, night, band, k))
        if not os.path.exists(pbp):
            a['refuse'] = f'T-5: {what} w{k}: no window primary-beam map {pbp}'
            return a
        night_pb[k] = pbp
        a['inputs'].append(pbp)
    if os.path.exists(os.path.join(cdir, 'NO_WINDOWS.txt')) or not os.path.exists(fcsv):
        a['refuse'] = (f'T-6: {what}: the band has no tree to add a night to ({cdir}: '
                       f'{"NO_WINDOWS.txt" if os.path.exists(fcsv) else "no frames.csv"}) - `{BATCH_STAGE}` builds it')
        return a
    bad, have = read_frames(fcsv, cols)
    if bad or not have:
        a['refuse'] = f'T-6: {what}: {bad or fcsv + " has no rows"} - `{BATCH_STAGE}` builds the band\'s tree'
        return a
    new = night_rows(cols, sdir, band, night, found, ks)
    mine = [x for x in have if x[2]['date'] == night]
    others = [x for x in have if x[2]['date'] != night]
    if mine and [x[1] for x in mine] != [x[1] for x in new]:
        a['refuse'] = (f'T-7: {what}: the night is in {fcsv} already with other rows ({len(mine)} row(s), {len(new)} to '
                       f'present) - `{BATCH_STAGE}` rebuilds the band\'s tree')
        return a
    for _key, _line, link, src in new:
        if os.path.lexists(link):
            if not os.path.islink(link) or os.readlink(link) != src:
                a['refuse'] = (f'T-7: {what}: {link} exists and is not a link to {src} - `{BATCH_STAGE}` rebuilds the '
                               f'band\'s tree')
                return a
        else:
            a['links'].append((link, src))
    target = sorted(others + [(key, line, None) for key, line, _l, _s in new], key=lambda x: x[0])
    a['n_rows'] = len(target)
    if not mine:
        a['lines'] = [','.join(cols) + '\r\n'] + [x[1] for x in target]
    n_now = {k: sum(1 for x in have if x[0][1] == k and x[0][2] == 0) for k in ks}
    n_target = {k: sum(1 for x in target if x[0][1] == k and x[0][2] == 0) for k in ks}
    for k in ks:
        pb = os.path.join(bdir, 'PB_bin%d_%s.fits' % (k, band))
        if not os.path.exists(pb):
            a['refuse'] = (f'T-8: {what} w{k}: no {pb} - no earlier night presented this window, so its primary-beam map '
                           f'would hold non-member nights only: `{BATCH_STAGE}` builds it')
            return a
        h = fits.getheader(pb)
        hn = fits.getheader(night_pb[k])
        if h.get('PBSRC') != PBSRC_WEIGHTED[0]:
            a['refuse'] = (f'T-8: {what} w{k}: {pb} is not a weighted mean (PBSRC {h.get("PBSRC")!r}): one more non-member '
                           f'night changes its pixels - `{BATCH_STAGE}` rebuilds the band\'s tree')
            return a
        if (h.get('NAXIS1'), h.get('NAXIS2')) != (hn.get('NAXIS1'), hn.get('NAXIS2')):
            a['refuse'] = (f'T-8: {what} w{k}: {pb} is {h.get("NAXIS1")} x {h.get("NAXIS2")} pixels, the night\'s window '
                           f'primary-beam map {hn.get("NAXIS1")} x {hn.get("NAXIS2")}')
            return a
        # the map counts the Stokes I rows of its window; one short only when the night's rows are in already (a run of
        # this unit interrupted between frames.csv and this header)
        if h.get('PBNNIGHT') not in ((n_now[k], n_now[k] - 1) if mine else (n_now[k],)):
            a['refuse'] = (f'T-8: {what} w{k}: {pb} has PBNNIGHT {h.get("PBNNIGHT")!r}, {fcsv} has {n_now[k]} Stokes I '
                           f'row(s) of the window - `{BATCH_STAGE}` rebuilds the band\'s tree')
            return a
        first = min([x[0][0] for x in target if x[0][1] == k and x[0][2] == 0])
        if first == night:                # the builder takes the header of the earliest presented night (tree:117)
            want = hn.copy()
            want['PBSRC'] = PBSRC_WEIGHTED
        else:
            want = h.copy()
        want['PBNNIGHT'] = (n_target[k], PBNNIGHT_COMMENT)
        if _cards(want) != _cards(h):     # the writer sets BITPIX and NAXIS* from the pixels, as the builder's does
            a['pb'][k] = (pb, want)
    a['action'] = 'present' if (a['lines'] or a['links'] or a['pb']) else 'already presented'
    return a


def _replace_pb(path, header):
    """Rewrite one PB_bin map with a new header and the same pixels: a temporary file beside it, then one rename (a
    reader sees the old file or the new one, never a file being written)."""
    import numpy as np
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        data = np.asarray(h[0].data)
    tmp = os.path.join(os.path.dirname(path), '.%s.%d.tmp' % (os.path.basename(path), os.getpid()))
    try:
        fits.writeto(tmp, data, header, overwrite=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def apply(a, fcsv):
    """Write what assess found: the links, then frames.csv, then the PB_bin headers. A failure removes the links this
    call made; a rerun completes whatever is left (assess reads the tree as it is)."""
    made = []
    tmp = os.path.join(os.path.dirname(fcsv), '.frames.csv.%d.tmp' % os.getpid())
    try:
        if a['lines']:
            with open(tmp, 'w', newline='') as f:
                f.write(''.join(a['lines']))
        for link, src in a['links']:
            os.symlink(src, link)
            made.append(link)
        if a['lines']:
            os.replace(tmp, fcsv)
            made = []                      # the rows are in: the links stay
        for _k, (pb, header) in sorted(a['pb'].items()):
            _replace_pb(pb, header)
    except BaseException:
        for link in made:
            try:
                os.remove(link)
            except OSError:
                pass
        raise
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def tree_lease(tree, lst, band, unit):
    """The writer lease of ONE band's tree (capacity 1 across processes and nodes): two nights of an hour are added one
    after the other, and never while the batch stage rebuilds the band. The Lease class of lib/search/tier_sublst.py,
    kept beside the tree it guards (never under the search's lease root, a production path outside a parity run)."""
    from exopipe.phase3.lib.search import tier_sublst as SS
    return SS.Lease(os.path.join(tree, LEASE_SUBDIR), f'{lst}_{band}', 'node-memory', capacity=1, poll_s=LEASE_POLL_S,
                    wait_s=LEASE_WAIT_S, info={'unit': unit, 'tree': cell_dir(tree, lst, band)})


def stage(ctx, lst, nights):
    """Run (or plan) the stage for the named nights of an LST hour: one unit per band of TREE_BANDS in the selection that
    has an x2 residual of the night."""
    from exopipe.phase3.lib.frames import tier_sublst as F
    if not nights:
        F.unit_refused(ctx.run, f'{lst}_{STAGE}', f'the {STAGE} stage adds ONE named night to the tree: give --epochs '
                       f'<LST>_<date> (or --lsts with --nights); the tree of every night of an LST is `{BATCH_STAGE}`',
                       ctx.dry, ctx, tier=F.TIER)
        return
    S = ctx.S
    cols = tuple(S.TREE_COLUMNS)
    tree = ctx.tree_dir()
    bands = [b for b in S.TREE_BANDS if not ctx.scopes or b in ctx.scopes]
    for night in nights:
        base = ctx.input_base(f'{lst}_x2') if ctx.dry else None
        x2, root = ctx.x2_dir(lst, base), ctx.lst_dir(lst, base)
        todo = [b for b in bands if night_residuals(x2, lst, night, b)]
        if ctx.dry:
            F.plan(f'  STAGE {STAGE} {lst}_{night}: {len(todo)} band(s) with an x2 residual of the night under {x2} (of '
                   f'{len(bands)} selected); tree {tree}')
        if not todo:
            F.unit_refused(ctx.run, f'{lst}_{night}_{STAGE}', f'T-3: {lst}_{night}: no x2 residual of the night under {x2} '
                           f'in the {len(bands)} selected band(s) - `flow --tier sublst --stages refine-nonmember` writes '
                           f'them', ctx.dry, ctx, tier=F.TIER)
            continue
        ex = F.exclusions()
        if ctx.dry:
            for b in todo:
                name = unit_name(lst, b, night)
                a = assess(lst, b, night, x2, root, tree, cols, ex.is_excluded(lst, b, night))
                gate = [g for g in F.tree_refine_gate(ctx, lst, b, base) if not g[1]]
                why = a['refuse'] or (f'S-G8 x2 refine of {lst} {gate[0][0]} not complete ({gate[0][2]})' if gate else None)
                if why:
                    F.plan(f'    UNIT {name}: WOULD REFUSE - {why}')
                    ctx.would_refuse += 1
                else:
                    F.plan(f'    UNIT {name}: {a["action"]}: {len(a["ks"])} window(s), {len(a["links"])} link(s), '
                           f'{"frames.csv with " + str(a["n_rows"]) + " rows" if a["lines"] else "frames.csv unchanged"}, '
                           f'{len(a["pb"])} PB_bin header(s)')
                ctx.planned += 1
            continue

        def one(b, night=night, x2=x2, root=root, ex=ex):
            name = unit_name(lst, b, night)
            cdir = cell_dir(tree, lst, b)
            with ctx.run.unit(name, tier=F.TIER, stage=STAGE, lst=lst, band=b, night=night) as u:
                if not os.path.isdir(cdir):
                    u.refuse(f'T-6: {lst}_{night} {b}: the band has no tree to add a night to (no {cdir}) - '
                             f'`{BATCH_STAGE}` builds it')
                # every gate that reads the tree is checked under the lease: the tree is then what this unit adds to
                lease = tree_lease(tree, lst, b, name)
                lease.acquire(say=lambda m: F.say(f'frames: [{name}] {m}'))
                try:
                    a = assess(lst, b, night, x2, root, tree, cols, ex.is_excluded(lst, b, night))
                    for p in a['inputs']:
                        u.add_input(p)
                    u.set_admitted([] if a['refuse'] and a['refuse'].startswith('T-1') else [night])
                    if a['refuse']:
                        u.refuse(a['refuse'])
                    for sb, ok, why, _log in F.tree_refine_gate(ctx, lst, b):
                        if not ok:
                            u.refuse(f'S-G8 x2 refine of {lst} {sb} not complete ({why}): a night is presented only '
                                     f'after its refine')
                    apply(a, os.path.join(cdir, 'frames.csv'))
                finally:
                    lease.release()
                u.meta[STAGE] = dict(action=a['action'], windows=a['ks'], skipped=a['skipped'], weight=a['weight'],
                                     links_made=len(a['links']), frames_rows=a['n_rows'],
                                     frames_written=bool(a['lines']), pb_headers_written=sorted(a['pb']))
                u.note(f'{a["action"]}: {len(a["ks"])} window(s), {len(a["links"])} link(s) made, frames.csv '
                       f'{"written with " + str(a["n_rows"]) + " rows" if a["lines"] else "unchanged"}, '
                       f'{len(a["pb"])} PB_bin header(s) written' + (f'; skipped {"; ".join(a["skipped"])}' if a['skipped'] else ''))
                u.add_output(os.path.join(cdir, 'frames.csv'), 'tenmin_tree', lst=lst, band=b)
                for k in a['ks']:
                    u.add_output(os.path.join(cdir, 'bin_masters', 'PB_bin%d_%s.fits' % (k, b)), 'tenmin_tree', lst=lst,
                                 band=b)
            F.report(u)

        # a unit's error is in its record; the other bands go on (the batch stage's rule)
        F.parallel(one, todo, ctx.workers or S.TREE_PARALLEL)
