"""The pages of a night (design 2.3a of plan/PER_HOUR_GAP_DESIGNS_20261111.md, stage 3): a sub-LST page run made from an LST
hour's catalog VIEW of a night pass, in which only the blocks that changed are rendered.

Two parts, both used by lib/pages/tier_sublst.py; nothing here runs a production module.

THE STAGED CATALOG DIRECTORY (stage_catalog): what production's sublst_pages_prep.py reads as "the hour's catalog directory",
as a directory of links: the view root's tables and catalogs (lib/catalog/delta_build.view_root: detection table, image
table, selected rows, window-tie table, catalogs, localizations) and, from the hour's all-nights catalog directory, the
records a night pass does not write again (pages_sublst.NIGHT_ALL_NIGHTS_FILES: the rejected-source record, the frame
times of the served detections, the page filter file, the conformance read's record and log). The detections of the nights
appended since the all-nights build have no rows in those records until the next all-nights build.

THE BLOCKS (plan, reduced, rename, link_blocks, test, the record): build_tenmin_pages.py writes one image per selection
row, <run>/candidates/<Stokes>/<page id>/rank<5 digits>_<LST>_<night>_bin<k>_<band>.png; the rank (the row's place in its
selection by falling |S/N|) is in the file name only. What a block shows depends on: its selection row (every column), the
frames of its window (the search's residual images, the leave-one-out window Master), the set's E_params.json, the
detection's window-tie row and reference-tie row, the predicted-V products and the parent catalog the run links to, the
cross-match samples, and the code. A block of an EARLIER page run of the hour is carried (a hard link under the new rank's
name) when that run can vouch for all of them:
  its record (page_blocks.json) is complete and names the same code (code_digest: the frozen tree's files with the tree's
  own root normalized, so two trees frozen from one code agree), the same cross-match samples and the same search root;
  it names, for the set, the same all-nights catalog build (frozen_id: the earlier nights' detections, and with them their
  frames, are the ones the earlier run rendered);
  the set's E_params.json is byte-identical, both runs link to the same parent catalog directory and, for the block's
  (LST, band), to the same predicted-V directory, and both hold a reference-tie table or neither does;
  the block's selection row, window-tie row and reference-tie row are equal, and the earlier run holds the image;
  the files of the block's frame (frame_ids: the searched residual images of both Stokes, the predicted-V images of the
  frame, the parent catalog of the band, and the run's leave-one-out window Masters) have the names, sizes and
  modification times the earlier run recorded before it rendered.
The leave-one-out window Masters of the earlier run are carried on the same terms (link_masters: the window stack and the
leave-one-out residual a Master is made from are the files the earlier run recorded).
Every other row is rendered, from a REDUCED selection file (those rows: the full file's lines, in its order), and the
images are then renamed to the rank each row has in the full selection (rename). A set whose earlier run cannot vouch is
rendered whole from the full selection file, as an all-nights page run is.

The ranking rule and the image name are production's (build_tenmin_pages.py:329-340, :387, :422-425), mirrored in ranked()
and block_rel(); test() holds every page run to them (an image the mirror does not expect fails it).
"""
import csv
import errno
import glob
import hashlib
import json
import math
import os
import re
import shutil

RECORD = 'page_blocks.json'                    # in a page run: what its blocks were rendered from (written once test() passed)
VERSION = 1
ROW_KEY = ('cand_id', 'member_cand_id')        # (page id, member detection) of a selection row
SELECTION = os.path.join('source_catalogs', 'tenmin_selection_{K}.csv')
REDUCED_DIR = os.path.join('source_catalogs', '_night')     # the reduced selection files (same base names: the renderer
                                                            # reads its set from the name)
VIEW_ROOT_RECORD = 'hour_view.json'            # lib/catalog/delta_build.ROOT_RECORD
VIEW_RECORD = 'catalog_view.json'              # lib/catalog/delta_build.VIEW_RECORD
FROZEN_RECORD = 'builder_frozen.json'          # lib/catalog/delta_build.FROZEN_RECORD
PERTIE = os.path.join('pertie', 'pertie_detections.csv')
LIVE_ROOT = '/lustre/gh/main'                  # the live code root a frozen tree's rewrites point away from
CODE_ROOTS = ('clv15/', 'exopipe/', 'exopipe_ops/')     # the manifest's code files (the tree's own bookkeeping names its path)


class NightRefusal(Exception):
    """A night's page run that must not be made (its message ends with REFUSING)."""


def _sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _json(path):
    with open(path) as f:
        return json.load(f)


# ---- the staged catalog directory --------------------------------------------------------------------------------------
def stage_catalog(staging, view_root, files):
    """Make ``staging`` (removed first when it exists: this stage's own directory) = the catalog directory
    sublst_pages_prep.py is to read, as links: every table and catalog of the complete view root ``view_root``; from the
    all-nights catalog directory its record names, the files ``files`` and every selected_*.csv and pertie/ entry the view
    root does not hold. Returns dict(all_nights=<directory>, view=[names], carried=[names], absent=[names of ``files`` the
    all-nights directory does not hold])."""
    view_root = os.path.abspath(view_root)
    recp = os.path.join(view_root, VIEW_ROOT_RECORD)
    if not os.path.exists(recp):
        raise NightRefusal(f'{view_root} holds no {VIEW_ROOT_RECORD}: not a view root of a night pass - REFUSING')
    rec = _json(recp)
    if rec.get('complete') is not True:
        raise NightRefusal(f'{view_root}: the view root is not complete (its localizations are not in) - REFUSING')
    all_nights = rec['all_nights']
    if not os.path.isdir(all_nights):
        raise NightRefusal(f'{view_root} stands on the all-nights catalog directory {all_nights}, which is gone - REFUSING')
    pt = os.path.join(view_root, PERTIE)
    if not os.path.exists(pt):
        raise NightRefusal(f'{view_root} holds no window-tie table ({PERTIE}) - REFUSING')
    if os.path.lexists(staging):
        shutil.rmtree(staging)
    os.makedirs(os.path.join(staging, 'pertie'))
    out = dict(all_nights=all_nights, view=[], carried=[], absent=[])
    for n in sorted(os.listdir(view_root)):
        if n == 'pertie' or n == VIEW_ROOT_RECORD or n.startswith('_') or n.endswith('.log'):
            continue                                  # the record, the view stage's side directories and logs
        os.symlink(os.path.join(view_root, n), os.path.join(staging, n))
        out['view'].append(n)
    os.symlink(pt, os.path.join(staging, PERTIE))
    out['view'].append(PERTIE)
    apt = os.path.join(all_nights, 'pertie')
    for n in (sorted(os.listdir(apt)) if os.path.isdir(apt) else []):
        if not os.path.lexists(os.path.join(staging, 'pertie', n)):
            os.symlink(os.path.join(apt, n), os.path.join(staging, 'pertie', n))
            out['carried'].append(os.path.join('pertie', n))
    names = list(files) + sorted(os.path.basename(p) for p in glob.glob(os.path.join(all_nights, 'selected_*.csv')))
    for n in names:
        if os.path.lexists(os.path.join(staging, n)):
            continue
        if os.path.exists(os.path.join(all_nights, n)):
            os.symlink(os.path.join(all_nights, n), os.path.join(staging, n))
            out['carried'].append(n)
        else:
            out['absent'].append(n)
    return out


def frozen_id(cat_dir):
    """The all-nights catalog build a catalog directory stands on: the sha1 of its builder_frozen.json (an all-nights
    directory) or the frozen_record_sha1 its catalog_view.json names (a view); '' when it names neither (a build that
    predates the frozen statistics: no later page run can stand on its blocks)."""
    p = os.path.join(cat_dir, FROZEN_RECORD)
    if os.path.exists(p):
        return _sha1(p)
    p = os.path.join(cat_dir, VIEW_RECORD)
    if os.path.exists(p):
        return _json(p).get('frozen_record_sha1') or ''
    return ''


# ---- what a run was rendered with ---------------------------------------------------------------------------------------
def code_digest(tree):
    """sha1 over the code files a verified frozen tree's MANIFEST.sha256 lists (those under clv15/, exopipe/, exopipe_ops/),
    each file's bytes taken with the tree's own root replaced by the live root: two trees frozen from the same code give the
    same value. '' without a tree (code that is not pinned vouches for nothing)."""
    if not tree:
        return ''
    tree = os.path.abspath(tree)
    old, new = tree.encode(), LIVE_ROOT.encode()
    h = hashlib.sha1()
    with open(os.path.join(tree, 'MANIFEST.sha256')) as f:
        rels = sorted({ln.rstrip('\n')[66:] for ln in f if re.match(r'[0-9a-fA-F]{64}(  | \*)\S', ln)})
    n = 0
    for rel in rels:
        if not rel.startswith(CODE_ROOTS):
            continue
        with open(os.path.join(tree, rel), 'rb') as f:
            h.update(f'{rel}\t{hashlib.sha1(f.read().replace(old, new)).hexdigest()}\n'.encode())
        n += 1
    return h.hexdigest() if n else ''


def samples_id(sample_dir, samples):
    """The cross-match sample files the renderer reads ({sample: [size, mtime in ns]} of <sample_dir>/<sample>.csv; a
    sample without a file is omitted, as the renderer omits it)."""
    out = {}
    for s in samples:
        p = os.path.join(sample_dir, f'{s}.csv')
        if os.path.exists(p):
            st = os.stat(p)
            out[s] = [st.st_size, st.st_mtime_ns]
    return out


# ---- the files a frame's blocks are drawn from --------------------------------------------------------------------------
PRED_NAME = re.compile(r'VD_P3_(\d\dh)_([A-Za-z0-9]+?)_(\d{4}-\d\d-\d\dT\d\d)_')        # build_tenmin_pages.py:200
MASTER_REL = re.compile(r'^loo_masters/(\d\dh)_(\d{4}-\d\d-\d\d)/([^/]+)/([IV])/loo_master_w(\d+)\.fits$')


def frame_key(lst, night, k, band):
    return f'{lst}|{night}|{int(k)}|{band}'


def frames_of(rows):
    """{(lst, night, window, band)} of the rows the renderer renders (its frame key, build_tenmin_pages.py:341-344)."""
    return {(rows[i]['lst'], night, k, rows[i]['band']) for _rank, i, night, k in ranked(rows)}


def _stat(path, name=None):
    """[name, size, modification time in ns] of a file, or None when there is none."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [name or path, st.st_size, st.st_mtime_ns]


def _digest(stats):
    return hashlib.sha1(json.dumps(sorted(x for x in stats if x), separators=(',', ':')).encode()).hexdigest()[:20]


def masters_in(sub_root, lst, night, k, band):
    """Digest of the files sublst_pages_prep.py makes a frame's leave-one-out window Masters from (its lines 151-153: the
    night's window stack and its leave-one-out residual, both Stokes)."""
    st = []
    for S in ('I', 'V'):
        a = os.path.join(sub_root, lst, f'{lst}_{night}', band, S, f'sublst_{lst}_{night}_{band}_{S}_w{int(k)}.fits')
        st += [_stat(a), _stat(a[:-5] + '_resid_loo.fits')]
    return _digest(st)


def frame_ids(run, search_root, sub_root, keys):
    """{frame key: dict(searched, n_searched, masters_in, masters)} of the frames ``keys`` ({(lst, night, window, band)}) as
    the page run ``run`` reads them. Each value is a digest of [name, size, modification time] lists:
      searched    the search's residual images of the frame, both Stokes, with and without the primary-beam correction
                  (build_tenmin_pages.py:296-300; n_searched counts them), the predicted-V images of the frame in the
                  run's v110_products/<LST>_<band> (:199-203) and the band's parent catalog (:218);
      masters_in  masters_in() - what prep makes the frame's leave-one-out window Masters from;
      masters     the run's own leave-one-out window Masters of the frame (:307), named relative to the run."""
    rdirs, preds, parents, out = {}, {}, {}, {}
    for lst, night, k, band in sorted(keys):
        dk = f'{night}T{int(k):02d}'
        st, n = [], 0
        for S in ('I', 'V'):
            c = (lst, band, S)
            if c not in rdirs:
                rdirs[c] = sorted(glob.glob(os.path.join(search_root, f'{lst}_{band}_{S}', 'Run_*',
                                                         'iresid' if S == 'I' else 'vresid', f'{lst}_{band}')))
            for d in rdirs[c]:
                for pb in ('', 'pbcorr_'):
                    x = _stat(os.path.join(d, f'{dk}_{S}_redo_resid_{pb}{band}.fits'))
                    if x:
                        st.append(x)
                        n += 1
        c = (lst, band)
        if c not in preds:
            preds[c] = {}
            d = os.path.realpath(os.path.join(run, 'v110_products', f'{lst}_{band}'))
            try:
                ents = list(os.scandir(d))
            except OSError:
                ents = []
            for e in ents:
                m = PRED_NAME.search(e.name) if e.name.endswith('_predicted_cutout.fits') else None
                if m and (m.group(1), m.group(2)) == c:
                    x = _stat(os.path.join(d, e.name))
                    if x:
                        preds[c].setdefault(m.group(3), []).append(x)
            parents[c] = _stat(os.path.realpath(os.path.join(run, 'parent_catalogs', f'master_R075_{lst}_{band}.gaul.csv')))
        st += preds[c].get(dk, [])
        st.append(parents[c])
        rels = [os.path.join('loo_masters', f'{lst}_{night}', band, S, f'loo_master_w{int(k)}.fits') for S in ('I', 'V')]
        out[frame_key(lst, night, k, band)] = dict(
            searched=_digest(st), n_searched=n, masters_in=masters_in(sub_root, lst, night, k, band),
            masters=_digest([_stat(os.path.join(run, rel), rel) for rel in rels]))
    return out


def run_frames(run, sets, search_root, sub_root):
    """frame_ids of every frame the selections of ``sets`` in the page run name."""
    keys = set()
    for K in sets:
        keys |= frames_of(read_selection(os.path.join(run, SELECTION.format(K=K)))[2])
    return frame_ids(run, search_root, sub_root, keys)


def read_record(run):
    """The page run's block record, or None (none written, or unreadable)."""
    try:
        return _json(os.path.join(run, RECORD))
    except (OSError, ValueError):
        return None


def write_record(run, rec):
    p = os.path.join(run, RECORD)
    with open(p + '.tmp', 'w') as f:
        json.dump(rec, f, indent=1, sort_keys=True)
        f.write('\n')
    os.replace(p + '.tmp', p)
    return p


# ---- the renderer's rows, ranks and image names (mirror of build_tenmin_pages.py) ----------------------------------------
def read_selection(path):
    """(header line, data lines or None, rows) of a selection file: rows as the renderer reads them (csv.DictReader); the
    data lines (one per row, bytes kept) or None when a field spans lines or a line is blank (no reduced file can be cut)."""
    with open(path, newline='') as f:
        text = f.read()
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    lines = [ln[:-1] if ln.endswith('\r') else ln for ln in text.split('\n')]
    while lines and lines[-1] == '':
        lines.pop()
    head, data = (lines[0] if lines else ''), lines[1:]
    if len(data) != len(rows) or any(not ln.strip() for ln in data):
        data = None
    return head, data, rows


def _fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def ranked(rows):
    """[(rank, index into rows, night, bin)] of the rows build_tenmin_pages.py renders, in its page order (lines 329-340: a
    row whose date is not <night>T<digits> or whose snr is not finite is dropped; falling |snr|, equal values in file
    order; ranks count from 1)."""
    keep = []
    for i, r in enumerate(rows):
        d = r.get('date') or ''
        if 'T' not in d:
            continue
        night, b = d.split('T', 1)
        if not b.isdigit():
            continue
        s = abs(_fnum(r.get('snr')))
        if not math.isfinite(s):
            continue
        keep.append((i, night, int(b), s))
    keep.sort(key=lambda t: -t[3])
    return [(n + 1, i, night, k) for n, (i, night, k, _s) in enumerate(keep)]


def block_rel(row, rank, night, k):
    """The image of a selection row, relative to the page run (build_tenmin_pages.py:362, :387, :422-425)."""
    S = row.get('stokes') or 'V'
    sid = row.get('cand_id') or f"{row['lst']}_{row['date']}_{row['band']}_{rank:05d}"
    return os.path.join('candidates', S, sid, f"rank{rank:05d}_{row['lst']}_{night}_bin{k}_{row['band']}.png")


# ---- the plan of one set ------------------------------------------------------------------------------------------------
def _by_cand(path):
    """{cand_id: [row, ...]} of a table, or None when the run holds none."""
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            out.setdefault(r.get('cand_id', ''), []).append(r)
    return out


def _link_of(path):
    """Where a link of a page run points ('' when the run holds none)."""
    return os.path.realpath(path) if os.path.lexists(path) else ''


def run_vouches(base, ident):
    """'' when the earlier page run ``base`` can vouch for blocks at all under ``ident`` (dict(code, samples, search_root)
    of the present run), else the reason."""
    if not base:
        return 'no earlier page run of the hour was named'
    if not os.path.isdir(base):
        return f'the earlier page run {base} does not exist'
    rec = read_record(base)
    if rec is None:
        return f'the earlier page run {base} holds no {RECORD} (it was not rendered by this step, or not to its end)'
    if rec.get('version') != VERSION or rec.get('complete') is not True:
        return f'the record of the earlier page run {base} is of another version or not complete'
    if not ident.get('code'):
        return 'the code of this run is not a frozen tree (code that is not pinned vouches for nothing)'
    for k, what in (('code', 'another code'), ('samples', 'other cross-match sample files'), ('search_root', 'another search root')):
        if rec.get(k) != ident.get(k):
            return f'the earlier page run {base} was rendered with {what}'
    return ''


def plan(K, run, base, ident, frames=None):
    """What the set ``K`` of the page run ``run`` needs rendered and what the earlier run ``base`` (None: none) holds
    already; ``frames`` = run_frames() of ``run`` (without it no block is carried). Returns dict(set, selection, head,
    lines, rows, blocks=[(rank, index, relative image path)], fkeys=[the frame key of each block],
    render=[positions in blocks], link=[(position in blocks, the earlier run's relative image path)], whole=<the reason the
    whole set is rendered, or ''>, why={reason: rows rendered for it})."""
    sel = os.path.join(run, SELECTION.format(K=K))
    head, lines, rows = read_selection(sel)
    rk = ranked(rows)
    blocks = [(rank, i, block_rel(rows[i], rank, night, k)) for rank, i, night, k in rk]
    fkeys = [frame_key(rows[i]['lst'], night, k, rows[i]['band']) for _rank, i, night, k in rk]
    out = dict(set=K, selection=sel, head=head, lines=lines, rows=rows, blocks=blocks, fkeys=fkeys, render=[], link=[],
               whole='', why={})

    def whole(reason):
        out.update(whole=reason, render=list(range(len(blocks))), link=[])
        return out
    why = run_vouches(base, ident)
    if why:
        return whole(why)
    bframes = (read_record(base) or {}).get('frames') or {}
    if not frames or not bframes:
        return whole('the files of the frames were not recorded (by this run or by the earlier page run)')
    cols = next(csv.reader([head])) if head else []
    if any(c not in cols for c in ROW_KEY):
        return whole(f'the selection has no {" / ".join(ROW_KEY)} columns to name a block by')
    if lines is None:
        return whole('the selection file cannot be cut by lines (a field spans lines, or a line is blank)')
    if any(not rows[i].get('cand_id') for _rank, i, _rel in blocks):
        return whole('a row has no page id (the renderer names its block, and titles it, by its rank)')
    bsel = os.path.join(base, SELECTION.format(K=K))
    if not os.path.exists(bsel):
        return whole(f'the earlier page run holds no selection of {K}')
    bhead, _bl, brows = read_selection(bsel)
    if bhead != head:
        return whole('the selection\'s columns differ from the earlier page run\'s')
    cat, bcat = os.path.join(run, f'source_catalogs_{K}'), os.path.join(base, f'source_catalogs_{K}')
    fz = frozen_id(cat)
    if not fz or fz != frozen_id(bcat):
        return whole('the catalog does not stand on the all-nights build the earlier page run\'s stands on (or names none)')
    ep, bep = os.path.join(cat, 'E_params.json'), os.path.join(bcat, 'E_params.json')
    if os.path.exists(ep) != os.path.exists(bep) or (os.path.exists(ep) and _sha1(ep) != _sha1(bep)):
        return whole('E_params.json differs from the earlier page run\'s')
    if _link_of(os.path.join(run, 'parent_catalogs')) != _link_of(os.path.join(base, 'parent_catalogs')):
        return whole('the parent catalog directory differs from the earlier page run\'s')
    pt, bpt = _by_cand(os.path.join(run, 'pertie_detections.csv')), _by_cand(os.path.join(base, 'pertie_detections.csv'))
    tie, btie = _by_cand(os.path.join(run, 'tie_members.csv')), _by_cand(os.path.join(base, 'tie_members.csv'))
    if pt is None or bpt is None:
        return whole('a page run holds no window-tie table')
    if (tie is None) != (btie is None):
        return whole('one page run holds a reference-tie table and the other does not')
    key = lambda r: tuple(r.get(c) or '' for c in ROW_KEY)      # noqa: E731
    n_new, n_old, brow, bimg = {}, {}, {}, {}
    for r in rows:
        n_new[key(r)] = n_new.get(key(r), 0) + 1
    for brank, bi, bnight, bk in ranked(brows):
        kk = key(brows[bi])
        n_old[kk] = n_old.get(kk, 0) + 1
        brow[kk], bimg[kk] = brows[bi], block_rel(brows[bi], brank, bnight, bk)
    cells = {}

    def cell_same(r):
        c = f"{r.get('lst')}_{r.get('band')}"
        if c not in cells:
            rel = os.path.join('v110_products', c)
            cells[c] = _link_of(os.path.join(run, rel)) == _link_of(os.path.join(base, rel))
        return cells[c]
    for pos, (_rank, i, _rel) in enumerate(blocks):
        r = rows[i]
        kk = key(r)
        mid = r.get('member_cand_id') or r.get('cand_id') or ''
        if n_new[kk] != 1 or n_old.get(kk, 0) > 1:
            w = 'the page and member detection name more than one row'
        elif kk not in brow:
            w = 'the row is not in the earlier page run'
        elif brow[kk] != r:
            w = 'the row changed'
        elif pt.get(mid) != bpt.get(mid):
            w = 'the window-tie row changed'
        elif tie is not None and tie.get(mid) != btie.get(mid):
            w = 'the reference-tie row changed'
        elif not cell_same(r):
            w = 'the predicted-V directory of the cell differs'
        elif frames.get(fkeys[pos]) is None or frames[fkeys[pos]] != bframes.get(fkeys[pos]):
            w = 'a file of the frame changed (searched image, predicted image, parent catalog or window Master)'
        elif not os.path.isfile(os.path.join(base, bimg[kk])):
            w = 'the earlier page run holds no image of the row'
        else:
            out['link'].append((pos, bimg[kk]))
            continue
        out['render'].append(pos)
        out['why'][w] = out['why'].get(w, 0) + 1
    return out


def reduced(pl, run):
    """Write the reduced selection file of a plan (the rows to render: the full file's header and lines, in its order) under
    <run>/source_catalogs/_night/. Returns (path, [(reduced rank, relative image path as rendered, relative image path under
    the full selection's rank)]), or (None, []) when nothing is rendered. A plan that renders the whole set, or whose
    every row needs rendered, is not reduced: (the full selection, [])."""
    if not pl['render']:
        return None, []
    if pl['whole'] or len(pl['render']) == len(pl['blocks']):
        return pl['selection'], []
    final = {i: rel for _rank, i, rel in (pl['blocks'][pos] for pos in pl['render'])}
    idx = sorted(final)
    path = os.path.join(run, REDUCED_DIR, os.path.basename(pl['selection']))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.tmp', 'w', newline='') as f:
        f.write('\n'.join([pl['head']] + [pl['lines'][i] for i in idx]) + '\n')
    os.replace(path + '.tmp', path)
    _h, _l, rrows = read_selection(path)
    rr = ranked(rrows)
    if len(rr) != len(idx) or rrows != [pl['rows'][i] for i in idx]:
        raise NightRefusal(f'{path}: the reduced selection does not read back as the {len(idx)} rows to render - REFUSING')
    return path, [(r, block_rel(rrows[j], r, night, k), final[idx[j]]) for r, j, night, k in rr]


def rename(run, moves):
    """Give the images rendered from a reduced selection the names of the full selection (``moves`` of reduced()). A row's
    rank in the full selection is never below its rank in the reduced one and the order is the same, so the moves are made
    from the highest reduced rank down: no image is named over one still to move. Returns (moved, [the final relative
    paths of rows without an image: their frame has no searched image])."""
    moved, none = 0, []
    for _r, src, dst in sorted(moves, reverse=True):
        a, b = os.path.join(run, src), os.path.join(run, dst)
        if not os.path.isfile(a):
            none.append(dst)
            continue
        if a != b:
            if os.path.lexists(b):
                raise NightRefusal(f'{b} exists already: the rendered image {a} cannot take its name - REFUSING')
            os.makedirs(os.path.dirname(b), exist_ok=True)
            os.rename(a, b)
        moved += 1
    return moved, none


def _link(a, b):
    """Hard link, or a copy with the file's times where the two are on different file systems. Returns 'linked' / 'copied'."""
    try:
        os.link(a, b)
        return 'linked'
    except OSError as e:
        if e.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK):
            raise
    shutil.copy2(a, b + '.tmp')
    os.replace(b + '.tmp', b)
    return 'copied'


def link_blocks(base, run, pl):
    """Carry the plan's unchanged images from the earlier run under their names in this run. Returns {'linked': n, 'copied': n}."""
    n = dict(linked=0, copied=0)
    for pos, brel in pl['link']:
        a, b = os.path.join(base, brel), os.path.join(run, pl['blocks'][pos][2])
        if os.path.lexists(b):
            raise NightRefusal(f'{b} exists already: the earlier run\'s image {a} cannot be carried to it - REFUSING')
        os.makedirs(os.path.dirname(b), exist_ok=True)
        n[_link(a, b)] += 1
    return n


def link_masters(base, run, ident, sub_root):
    """Carry the earlier run's leave-one-out window Masters into ``run`` (before sublst_pages_prep.py, which writes only
    the Masters a run does not hold): every Master whose inputs (masters_in: the night's window stack and leave-one-out
    residual) are the files the earlier run recorded for the frame. Returns dict(carried, not_carried, why=<the reason
    nothing is carried, or ''>)."""
    out = dict(carried=0, not_carried=0, why=run_vouches(base, ident))
    if out['why']:
        return out
    bframes = (read_record(base) or {}).get('frames') or {}
    same = {}
    for d, _dirs, files in os.walk(os.path.join(base, 'loo_masters')):
        for f in sorted(files):
            a = os.path.join(d, f)
            rel = os.path.relpath(a, base)
            m = MASTER_REL.match(rel.replace(os.sep, '/'))
            if not m:
                out['not_carried'] += 1
                continue
            lst, night, band, _S, k = m.groups()
            fk = frame_key(lst, night, k, band)
            if fk not in same:
                same[fk] = fk in bframes and bframes[fk].get('masters_in') == masters_in(sub_root, lst, night, k, band)
            b = os.path.join(run, rel)
            if not same[fk] or os.path.lexists(b):
                out['not_carried'] += 1
                continue
            os.makedirs(os.path.dirname(b), exist_ok=True)
            _link(a, b)
            out['carried'] += 1
    return out


# ---- the content test of a page run's blocks ----------------------------------------------------------------------------
def test(run, sets, frames=None):
    """(ok, detail, counts): every image under <run>/candidates is the image of exactly one row of the selections of ``sets``
    under the rank and name the renderer gives it (no image of another rank, of a row no longer selected, or of an unknown
    name); two rows never share an image name; and, with ``frames`` (run_frames), every image is of a frame this module
    finds a searched image for (the renderer draws no block without one: an image without one means frame_ids no longer
    lists the files the renderer reads). counts: {K: dict(rows, images, without_image)}; a row without an image is one whose
    frame has no searched image (the renderer skips it) and is counted, not failed."""
    expected, dup, counts, fk_of = {}, [], {}, {}
    for K in sets:
        _h, _l, rows = read_selection(os.path.join(run, SELECTION.format(K=K)))
        counts[K] = dict(rows=0, images=0, without_image=0)
        for rank, i, night, k in ranked(rows):
            rel = block_rel(rows[i], rank, night, k)
            if rel in expected:
                dup.append(rel)
            expected[rel] = K
            fk_of[rel] = frame_key(rows[i]['lst'], night, k, rows[i]['band'])
            counts[K]['rows'] += 1
    have = {os.path.relpath(p, run) for p in glob.glob(os.path.join(run, 'candidates', '*', '*', '*'))}
    stray = sorted(have - set(expected))
    for rel, K in expected.items():
        counts[K]['images' if rel in have else 'without_image'] += 1
    blind = sorted(rel for rel in expected if rel in have and frames is not None
                   and not (frames.get(fk_of[rel]) or {}).get('n_searched'))
    ok = not stray and not dup and not blind
    n_img, n_none = sum(c['images'] for c in counts.values()), sum(c['without_image'] for c in counts.values())
    detail = (f'{n_img} images of {len(expected)} selection rows in {len(counts)} set(s), {n_none} row(s) without an image'
              + (f'; {len(stray)} image(s) no selection row names, e.g. {stray[:2]}' if stray else '')
              + (f'; {len(dup)} image name(s) shared by two rows, e.g. {dup[:2]}' if dup else '')
              + (f'; {len(blind)} image(s) of a frame without a searched image this module finds, e.g. {blind[:2]}' if blind else ''))
    return ok, detail, counts
