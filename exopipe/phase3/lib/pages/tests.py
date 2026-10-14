"""Content tests of every product class `exopipe3 pages` writes (registered with @content_test; core/validate discovers
this module). Each test reads the file's CONTENT; ``ctx`` (passed by the step when it writes the file) sharpens a test
(expected ids, the pass start, the page code, the layout); without ctx a test checks what the file itself and its run
directory allow (`exopipe3 verify <class> <path>`).

Classes: review_page, page_index, review_queue, pages_input, astrometry_figure, tess_coverage, survivors_review_dir,
sublst_page_run, snapshot_page_run, filter_catalog, strat_queue. (localization_table is catalog's class: catalog --stages localize
writes the tables, crosscheck decision 6.)
"""
import csv
import glob
import json
import math
import os
import re

from exopipe.phase3.core.validate import content_test
from exopipe.phase3.lib.pages import common

PANEL_SIZE_IN = (15.5, 8.8)        # std7_panel.render_block figsize (std7_panel.py:252); pixel size = this x pages.PNG_DPI
INDEX_COLUMNS = ('source_id', 'band', 'rank', 'review_dir', 'kind', 'date', 'ra', 'dec', 'flux_jy', 'xm_best', 'xm_all',
                 'xm_dist_pc', 'xm_sep_arcmin', 'xm_offset_pct', 'xm_p_chance')     # build_i14:794-805; p3t_pages_index.py:21-24
_PID_RE = re.compile(r'^(?P<j>J\S+|i14_\d{4,})_(?P<lst>\d\dh)_(?P<date>\d{4}-\d\d-\d\d(?:T\d+)?)_(?P<band>[A-Za-z0-9]+)$')
_INDEX_NAME = re.compile(r'^(?P<band>[^_]+)_pages_index_(?P<tag>[IV](?:_f[ws]b)?)\.csv$')


def _png_re(ctx):
    return common.server_png_re(ctx.get('roots'))


def _decode_png(path):
    """(width, height, mode) after a full CRC check of every chunk (PIL verify), then a header re-open."""
    from PIL import Image
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        return im.size[0], im.size[1], im.mode


def _expected_size():
    dpi = common.CFG().pages.PNG_DPI
    return int(round(PANEL_SIZE_IN[0] * dpi)), int(round(PANEL_SIZE_IN[1] * dpi))


def _tier_of_panel(name, png_re):
    m = png_re.match(name)
    if not m:
        return None
    if '_bin' in m.group(2):
        return 'sublst'
    tok = m.group(3)
    return 'snapshot' if re.search(r'\d{6,}', tok) or tok.endswith('1min') else 'deep'


# ---------------------------------------------------------------------------------------------------------------------
# review_page: one panel PNG of a page directory
# ---------------------------------------------------------------------------------------------------------------------

@content_test('review_page')
def decodes(path, **ctx):
    w, h, mode = _decode_png(path)
    want = _expected_size()
    ok = (w, h) == want and mode in ('RGBA', 'RGB')
    return ok, f'{w}x{h}', f'{mode}; render_block at {common.CFG().pages.PNG_DPI} dpi gives {want[0]}x{want[1]}'


@content_test('review_page')
def name_parses(path, **ctx):
    """The file name matches the server's PNG_RE (classify_transients.py:54-55, read in place) and agrees with its page
    directory: same band (deep, sub-LST: page id <J-name>_<LST>_<date>_<band>; snapshot: <J-name>_<band>[_1min])."""
    png_re = _png_re(ctx)
    name, page = os.path.basename(path), os.path.basename(os.path.dirname(path))
    m = png_re.match(name)
    if not m:
        return False, name, 'does not match classify_transients.PNG_RE (the server would not show it)'
    tier = ctx.get('tier') or _tier_of_panel(name, png_re)
    band_tok = m.group(3)
    if tier in ('deep', 'sublst'):
        pm = _PID_RE.match(page)
        if not pm:
            return False, page, 'page directory is not <J-name>_<LST>_<date>_<band> (or i14_<rank>_...)'
        if pm['band'] != band_tok:
            return False, f'{band_tok} vs {pm["band"]}', 'panel band differs from the page band'
        return True, page, tier
    base = page[:-5] if page.endswith('_1min') else page
    band = base.rsplit('_', 1)[-1] if '_' in base else ''
    ok = bool(band) and band_tok.startswith(band) and (page.endswith('_1min') == band_tok.endswith('1min'))
    return ok, page, f'{tier}: panel token {band_tok} vs directory band {band}'


@content_test('review_page')
def newest_is_served(path, **ctx):
    """This panel is the one the server shows for its (LST, night, band) - the NEWEST matching file of its directory
    (classify_transients.py:595-616) - and, given the pass start (ctx pass_t0), it was written or kept by this pass."""
    png_re = _png_re(ctx)
    d, name = os.path.dirname(path), os.path.basename(path)
    m = png_re.match(name)
    if not m:
        return False, name, 'no PNG_RE match'
    key = m.groups()[:3]
    mine = os.path.getmtime(path)
    rivals = []
    for fn in os.listdir(d):
        if fn == name:
            continue
        mm = png_re.match(fn)
        if mm and mm.groups()[:3] == key and not mm.group(4):
            rivals.append((os.path.getmtime(os.path.join(d, fn)), fn))
    newer = [fn for t, fn in rivals if t > mine]
    t0 = ctx.get('pass_t0')
    if t0 and mine < float(t0):
        return False, round(mine - float(t0), 1), f'written {round(float(t0) - mine)} s before the pass started'
    if newer:
        tier = ctx.get('tier') or _tier_of_panel(name, png_re)
        shared = (ctx.get('layout') or ('shared' if tier == 'deep' and
                                        os.path.basename(os.path.dirname(d)) == 'I' else 'per-set')) == 'shared'
        if shared:
            # M14, known and deferred (pages.PANEL_DIR_LAYOUT 'shared' = production layout in the window): another set's or
            # the other Stokes' later panel for the same key is the one served. Counted, not refused - refusing would refuse
            # production-identical output; strict under 'per-set'.
            return True, 'shadowed', f'{newer[0]} is newer for the same key: shadowed (M14, shared layout)'
        return False, len(rivals), f'{newer[0]} is newer for the same key: this panel is not the one served (M14)'
    return True, len(rivals), f'{len(rivals)} other file(s) for the key, all older'


@content_test('review_page')
def sidecar(path, **ctx):
    """Deep pages (pages.PAGE_CTX_IN_TEXT): <png>.page.json carries fp (sha1), code = sha1 of build_i14 + std7_panel +
    p3s3_astrom of the code that runs now (build_i14:819-820; a page of other code is stale), ctx_lines, rendered."""
    tier = ctx.get('tier') or _tier_of_panel(os.path.basename(path), _png_re(ctx))
    sc = path[:-4] + '.page.json'
    if tier != 'deep' or not common.CFG().pages.PAGE_CTX_IN_TEXT:
        return True, None, f'no sidecar expected ({tier})'
    if not os.path.exists(sc):
        return False, None, 'no <png>.page.json (deep pages carry their run-level lines and fingerprint in it)'
    j = json.load(open(sc))
    fp, code = str(j.get('fp') or ''), str(j.get('code') or '')
    if not re.match(r'^[0-9a-f]{40}$', fp):
        return False, fp[:12], 'fp is not a sha1'
    if not isinstance(j.get('ctx_lines'), list) or not all(isinstance(x, str) for x in j['ctx_lines']):
        return False, None, 'ctx_lines missing'
    want = ctx.get('code') or common.page_code(ctx.get('roots') or common.legacy_roots())
    if code != want:
        return False, code, f'page code {code} != current build_i14+std7_panel+p3s3_astrom {want} (stale page)'
    return True, code, f'rendered={j.get("rendered")}'


@content_test('review_page')
def own_set(path, **ctx):
    """Per-set layout (M14): the panel lives under candidates/<its set>/ (never another set's or Stokes' directory)."""
    K, layout = ctx.get('set'), ctx.get('layout') or common.CFG().pages.PANEL_DIR_LAYOUT
    if layout != 'per-set' or not K:
        return True, None, f'layout {layout}'
    sub = os.path.basename(os.path.dirname(os.path.dirname(path)))
    return sub == K, sub, f'expected candidates/{K}/'


# ---------------------------------------------------------------------------------------------------------------------
# page_index
# ---------------------------------------------------------------------------------------------------------------------

@content_test('page_index')
def index_rows(path, **ctx):
    """Columns; file name carries its Stokes (singles) or set tag (fork); every row of the file's band; page ids parse and
    carry the row's band; ranks unique; review_dir = <candidates dir>/<page id>; ids = the planned rows (ctx)."""
    m = _INDEX_NAME.match(os.path.basename(path))
    if not m:
        return False, os.path.basename(path), 'name is not <band>_pages_index_<S>[_f<t>b].csv'
    hdr, rows = common.read_csv(path)
    miss = [c for c in INDEX_COLUMNS if c not in hdr]
    if miss:
        return False, len(rows), f'missing columns {miss}'
    if not rows:
        return False, 0, 'no rows'
    bad = []
    ranks = set()
    sub = ctx.get('subdir')
    for r in rows:
        pm = _PID_RE.match(r['source_id'])
        if not pm or pm['band'] != r['band'] or r['band'] != m['band']:
            bad.append(f"id {r['source_id']} / band {r['band']}")
        if r['rank'] in ranks:
            bad.append(f"rank {r['rank']} twice")
        ranks.add(r['rank'])
        rd = r.get('review_dir') or ''
        if os.path.basename(rd) != r['source_id'] or not rd.startswith('candidates/'):
            bad.append(f'review_dir {rd}')
        elif sub and rd.split('/')[1] != sub:
            bad.append(f'review_dir {rd} not under candidates/{sub}/')
    want = ctx.get('expected_ids')
    if want is not None and set(want) != {r['source_id'] for r in rows}:
        bad.append(f'ids differ from the plan ({len(set(want) ^ {r["source_id"] for r in rows})} differ)')
    return not bad, len(rows), '; '.join(bad[:3]) or f'{len(rows)} rows, tag {m["tag"]}'


@content_test('page_index')
def dirs_have_panels(path, **ctx):
    """Every review_dir exists under the run directory and holds at least one panel the server reads (PNG_RE)."""
    png_re = _png_re(ctx)
    run = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    _, rows = common.read_csv(path)
    empty = []
    for r in rows:
        d = os.path.join(run, r.get('review_dir') or '')
        try:
            ok = any(png_re.match(fn) for fn in os.listdir(d))
        except OSError:
            ok = False
        if not ok:
            empty.append(r.get('review_dir'))
    return not empty, len(empty), (f'{len(empty)} of {len(rows)} page directories without a panel, e.g. {empty[:2]}'
                                   if empty else f'{len(rows)} page directories hold panels')


# ---------------------------------------------------------------------------------------------------------------------
# review_queue
# ---------------------------------------------------------------------------------------------------------------------

def _queue(path):
    head, ids = [], []
    for line in open(path):
        s = line.strip()
        if not s:
            continue
        (head if s.startswith('#') else ids).append(s)
    return head, ids


@content_test('review_queue')
def queue_protocol(path, **ctx):
    """The queue protocol (p3s3_stokes_queue.py reads it; IF-SRV-06): '# queue_name:' and '# selection:' header lines
    (a snapshot identity queue queue_<S>.txt carries '# run:' instead of the name, phase3_snapshot_pages.py:612-613);
    at least one id (IF-JDG-11); no id twice."""
    head, ids = _queue(path)
    name = os.path.basename(path)
    has_sel = any(h.startswith('# selection:') for h in head)
    has_name = any(h.startswith('# queue_name:') for h in head) or (
        re.match(r'^queue_[IV]\.txt$', name) and any(h.startswith('# run:') for h in head))
    dup = len(ids) - len(set(ids))
    ok = has_sel and bool(has_name) and ids and not dup
    return ok, len(ids), f'header name {bool(has_name)}, selection {has_sel}, {dup} repeated ids'


@content_test('review_queue')
def queue_order(path, **ctx):
    """The ids and their order: exactly the planned page ids in rank order (ctx expected_ids, or recomputed from the set's
    pages input for a raw deep queue, build_i14:1017-1018); a sub-LST queue in |snr| order (p3t_pages_index.py:14);
    a serve-sectioned deep singles queue: every id a page of the set (order is serve's sectioning)."""
    head, ids = _queue(path)
    want = ctx.get('expected_ids')
    run = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    m = re.match(r'^queue_p3s3_([IV]_f?[ws]b)(_raw)?\.txt$', os.path.basename(path))
    if want is None and m:
        pin = os.path.join(run, 'source_catalogs', f'pages_input_{m.group(1)}.csv')
        if os.path.exists(pin):
            from exopipe.phase3.lib.pages import panels
            rows = panels.ordered_rows(pin, common.CFG().pages.PAGE_SORT_KEY)
            sublst = rows and 'pass_ms' in rows[0] and (rows[0].get('run') == common.CFG().pages_sublst.RUN_COLUMN)
            if m.group(2):
                want = [r['_pid'] for r in rows]
            elif sublst:
                pid = {f"{r['_sid']}_{r['lst']}_{r['date']}_{r['band']}": abs(panels.fnum(r.get('snr'))) for r in rows}
                order_ok = all(pid.get(a, 0) >= pid.get(b, 0) for a, b in zip(ids, ids[1:]))
                unknown = [i for i in ids if i not in pid]
                return order_ok and not unknown, len(ids), f'|snr| order {order_ok}; {len(unknown)} ids not in pages input'
            else:
                known = {r['_pid'] for r in rows}
                unknown = [i for i in ids if i not in known]
                return not unknown, len(ids), (f'{len(unknown)} ids are not pages of the set, e.g. {unknown[:2]}'
                                               if unknown else 'serve-sectioned: every id a page of the set')
    if want is None:
        return True, len(ids), 'no plan to compare (standalone)'
    if ids == list(want):
        return True, len(ids), 'ids and order = the plan'
    return False, len(ids), (f'differs from the plan: {len(set(ids) ^ set(want))} ids differ'
                             + ('; same ids, other order' if set(ids) == set(want) else ''))


# ---------------------------------------------------------------------------------------------------------------------
# pages_input
# ---------------------------------------------------------------------------------------------------------------------

@content_test('stale_panels_manifest')
def stale_manifest_verifies(path, **ctx):
    """<run>/stale_panels_<pass start>/MANIFEST.sha256 (ops/p3s3_stale_panels.py, the deep 'stale' stage): every listed
    file lies in the quarantine folder with its listed sha256, and none of them is back at its path in the run (ctx run:
    the run directory; default the folder's parent) - nothing was lost, nothing deleted, nothing stale left served."""
    import hashlib
    q = os.path.dirname(path)
    run = ctx.get('run') or os.path.dirname(q)
    bad, n = [], 0
    for line in open(path):
        if '  ' not in line:
            continue
        want, rel = line.rstrip('\n').split('  ', 1)
        n += 1
        f = os.path.join(q, rel)
        if not os.path.isfile(f):
            bad.append(f'{rel}: not in the quarantine')
            continue
        h = hashlib.sha256()
        with open(f, 'rb') as fh:
            for blk in iter(lambda: fh.read(1 << 20), b''):
                h.update(blk)
        if h.hexdigest() != want:
            bad.append(f'{rel}: sha256 differs from the manifest')
        if os.path.exists(os.path.join(run, rel)):
            bad.append(f'{rel}: back in the run')
    return (n > 0 and not bad), n, (f'{n} files moved aside, all verify' if not bad else '; '.join(bad[:5]))


@content_test('pages_input')
def pages_input_rows(path, **ctx):
    """Deep: the 46 columns in the heredoc's order, one row per survivors row of the set (rows without date/LST dropped),
    every row of the set's Stokes; the file equals what the port (singles) or the fork rules recompute from the run's
    catalogues NOW. Sub-LST (p3t_pages_input.py): its columns, every row of the set's Stokes; tenmin_selection rows."""
    from exopipe.phase3.lib.pages import pages_input as PI
    name = os.path.basename(path)
    m = re.match(r'^(pages_input|tenmin_selection)_([IV]_f?[ws]b)\.csv$', name)
    if not m:
        return False, name, 'name is not pages_input_<K>.csv / tenmin_selection_<K>.csv'
    K = m.group(2)
    hdr, rows = common.read_csv(path)
    if not rows:
        return False, 0, 'no rows'
    if m.group(1) == 'tenmin_selection':
        miss = [c for c in ('cand_id', 'lst', 'date', 'band', 'ra', 'dec', 'snr') if c not in hdr]
        return not miss, len(rows), f'missing {miss}' if miss else f'{len(rows)} member detections'
    wrong_st = sum(1 for r in rows if r.get('stokes') != K[0])
    if wrong_st:
        return False, wrong_st, f'{wrong_st} rows not Stokes {K[0]}'
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    if rows[0].get('run') == common.CFG().pages_sublst.RUN_COLUMN or ctx.get('tier') == 'sublst':
        miss = [c for c in ('_sid', 'lst', 'date', 'band', 'ra', 'dec', 'snr', 'stokes') if c not in hdr]
        return not miss, len(rows), f'missing {miss}' if miss else f'sub-LST, {len(rows)} sources'
    if tuple(hdr) != PI.SINGLES_COLUMNS:
        return False, len(hdr), f'columns differ from the heredoc schema ({len(hdr)} vs {len(PI.SINGLES_COLUMNS)})'
    run_label = rows[0].get('run', '')
    if K in common.CFG().pages.FORK_SETS:
        exp, dropped = PI.fork_rows(run_dir, K, common.CFG().pages_deep.FORK_PAGES_LABEL, run_label)
    else:
        exp, dropped = PI.singles_rows(run_dir, K, run_label)
    got = [{k: (v if v is not None else '') for k, v in r.items()} for r in rows]
    exp = [{k: ('' if v is None else str(v)) for k, v in r.items()} for r in exp]
    if got != exp:
        n = sum(1 for a, b in zip(got, exp) if a != b) + abs(len(got) - len(exp))
        return False, n, f'{n} rows differ from the rows recomputed from the catalogues now ({len(got)} vs {len(exp)})'
    return True, len(rows), f'= recomputed from the catalogues ({dropped} rows without date/LST dropped)'


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------------------------------
# astrometry_figure, tess_coverage
# ---------------------------------------------------------------------------------------------------------------------

@content_test('astrometry_figure')
def figure_decodes(path, **ctx):
    w, h, mode = _decode_png(path)
    return w >= 200 and h >= 200, f'{w}x{h}', mode


@content_test('tess_coverage')
def tess_rows(path, **ctx):
    """The output contract of tess_coverage.py (:126-160): header comment line, one row per cand_id, status in the ruled
    vocabulary (simultaneous only), a simultaneous row carries sector/camera/CCD; with the run's master_catalog.csv beside
    it, the cand_ids are exactly that catalogue's detections of the wanted Stokes."""
    with open(path) as f:
        first = f.readline()
        rows = list(csv.DictReader(f))
    if not first.startswith('# tess_coverage.py'):
        return False, None, 'no tess_coverage.py header comment line'
    if not rows:
        return False, 0, 'no rows'
    allowed = set(common.CFG().pages_deep.TESS_STATUSES)
    bad = [r['cand_id'] for r in rows if r.get('status') not in allowed]
    bad += [r['cand_id'] for r in rows if r.get('status') == 'simultaneous' and not (r.get('sector') and r.get('camera')
                                                                                    and r.get('ccd'))]
    ids = [r['cand_id'] for r in rows]
    if len(ids) != len(set(ids)):
        bad.append('repeated cand_id')
    mc = ctx.get('master_catalog') or os.path.join(os.path.dirname(path), 'master_catalog.csv')
    detail = ''
    if os.path.exists(mc) and not bad:
        want = set(common.CFG().pages_deep.TESS_STOKES.split(','))
        with open(mc, newline='') as f:
            exp = {r['cand_id'] for r in csv.DictReader(f) if r.get('stokes') in want}
        diff = exp ^ set(ids)
        if diff:
            bad.append(f'{len(diff)} cand_ids differ from master_catalog.csv ({len(exp)} detections)')
        detail = f'= master_catalog.csv detections ({len(exp)})'
    return not bad, len(rows), '; '.join(map(str, bad[:3])) or detail or f'{len(rows)} rows'


# ---------------------------------------------------------------------------------------------------------------------
# survivors_review_dir (the merge writes the review_dir column of the union survivors)
# ---------------------------------------------------------------------------------------------------------------------

@content_test('survivors_review_dir')
def review_dir_links(path, **ctx):
    """Only review_dir changed (ctx pre_digest: sha1 of every other column before the merge); every non-empty review_dir
    is a page directory with panels whose page id carries the row's J-name and band; per-set layout: under the row's own
    singles set. Rows without a page are counted (value)."""
    png_re = _png_re(ctx)
    from exopipe.phase3.lib.pages import m14
    hdr, rows = common.read_csv(path)
    if 'review_dir' not in hdr:
        return False, None, 'no review_dir column'
    pre = ctx.get('pre_digest')
    if pre and common.digest_without(path, 'review_dir') != pre:
        return False, None, 'columns other than review_dir changed in the merge'
    run = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    layout = ctx.get('layout') or 'shared'
    bad, none, cache = [], 0, {}
    for r in rows:
        rd = r.get('review_dir') or ''
        if not rd:
            none += 1
            continue
        page = os.path.basename(rd.rstrip('/'))
        if rd not in cache:
            try:
                cache[rd] = any(png_re.match(fn) for fn in os.listdir(os.path.join(run, rd)))
            except OSError:
                cache[rd] = False
        if not cache[rd]:
            bad.append(f'{rd} holds no panel')
        elif not (page.startswith(r['source_id'] + '_') and page.endswith('_' + r.get('band', ''))):
            bad.append(f"{r['source_id']} {r.get('band')} -> {page}")
        if layout == 'per-set' and r.get('stokes') in ('I', 'V'):
            if not rd.startswith(f"candidates/{m14.set_of(r['stokes'], r.get('band', ''))}/"):
                bad.append(f'{rd} outside the row\'s own set')
    return not bad, none, ('; '.join(bad[:3]) + f' ({len(bad)} rows)') if bad else f'{len(rows) - none} linked, {none} without a page'


# ---------------------------------------------------------------------------------------------------------------------
# sub-LST page run, snapshot page run, filter catalogue, stratified queue
# ---------------------------------------------------------------------------------------------------------------------

@content_test('sublst_page_run')
def sublst_run(path, **ctx):
    """The prep products of a sub-LST page run (sublst_pages_prep.py): the window-tie table, the rejected record and the
    localizations present and non-empty; the copied files byte-equal to the catalogue's (ctx cat_dir, or the catalogue of
    the run's LST under CFG.paths.SUBLST_ROOT); every loo_master carries the LOOM card."""
    from exopipe.phase3.core.record import fits_primary_cards
    S = common.CFG().pages_sublst
    req = ('pertie_detections.csv', 'rejected.csv', 'rejected_index.npz', 'detection_ts.csv', 'master_catalog.csv',
           'v_deep_images.csv', 'source_catalogs/localizations.csv')
    miss = [f for f in req if not (os.path.isfile(os.path.join(path, f)) and os.path.getsize(os.path.join(path, f)) > 0)]
    if miss:
        return False, len(miss), f'missing or empty: {miss}'
    cat = ctx.get('cat_dir')
    m = re.search(r'Run_SLST(\d\d)', os.path.basename(path.rstrip('/')))
    if not cat and m:
        cat = S.CATALOG_DIR.format(root=common.CFG().paths.SUBLST_ROOT, lst=f'{m.group(1)}h', arm=S.ARM)
    diff = []
    if cat and os.path.isdir(cat):
        pairs = [('master_catalog.csv', 'master_catalog.csv'), ('v_deep_images.csv', 'v_deep_images.csv'),
                 ('pertie_detections.csv', 'pertie/pertie_detections.csv'), ('rejected.csv', 'rejected.csv'),
                 ('detection_ts.csv', 'detection_ts.csv')]
        for mine, src in pairs:
            a, b = os.path.join(path, mine), os.path.join(cat, src)
            if os.path.exists(b) and common.sha1_file(a, 40) != common.sha1_file(b, 40):
                diff.append(mine)
    nl, nobad = 0, []
    for p in glob.glob(os.path.join(path, 'loo_masters', '*', '*', '*', 'loo_master_w*.fits')):
        nl += 1
        if 'LOOM' not in fits_primary_cards(p):
            nobad.append(os.path.relpath(p, path))
    ok = not diff and not nobad and nl > 0
    return ok, nl, (f'copies differ from {cat}: {diff}; ' if diff else '') + (
        f'{len(nobad)} loo_masters without LOOM' if nobad else f'{nl} loo_masters with LOOM') + (
        '' if nl else '; no loo_master')


@content_test('snapshot_page_run')
def snapshot_run(path, **ctx):
    """R6 missing_panels.csv lists no panel; R2 every detections.csv carries finite positive pos_err_stat_arcsec and
    pos_err_arcsec; R7 every RA column of source_catalogs*/*.csv and detections.csv in [0, 360)
    (snapshot_rechain_content_test.py:56, 74-101); per-(Stokes, tier) views: n_det = n_det_tier."""
    mp = os.path.join(path, 'missing_panels.csv')
    if not os.path.exists(mp):
        return False, None, 'R6: no missing_panels.csv'
    miss = [l for l in open(mp).read().split('\n') if l.strip()]
    bad = [f'R6: {len(miss)} panels missing'] if miss else []
    dfiles = sorted(glob.glob(os.path.join(path, 'candidates', '*', '*', 'detections.csv')))
    if not dfiles:
        bad.append('no detections.csv')
    n_err = n_ra = n_out = 0
    for p in sorted(glob.glob(os.path.join(path, 'source_catalogs*', '*.csv'))) + dfiles:
        with open(p, newline='') as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames or []
            racols = [c for c in cols if c == 'ra' or c.startswith('ra_') or c.endswith('_ra')]
            isdet = p.endswith('detections.csv')
            if isdet and not ('pos_err_stat_arcsec' in cols and 'pos_err_arcsec' in cols):
                n_err += 1
            for r in rd:
                if isdet:
                    a, b = _f(r.get('pos_err_stat_arcsec')), _f(r.get('pos_err_arcsec'))
                    if a is None or b is None or a <= 0 or b <= 0:
                        n_err += 1
                for c in racols:
                    v = (r.get(c) or '').strip()
                    if not v:
                        continue
                    n_ra += 1
                    x = _f(v)
                    if x is None or not 0 <= x < 360:
                        n_out += 1
    if n_err:
        bad.append(f'R2: {n_err} detections without a finite positive own error')
    if n_out or not n_ra:
        bad.append(f'R7: {n_out} of {n_ra} RA values outside [0, 360)')
    n_view = 0
    for p in glob.glob(os.path.join(path, 'source_catalogs_[IV]_10s', '*_sources_survivors.csv')) + glob.glob(
            os.path.join(path, 'source_catalogs_[IV]_1min', '*_sources_survivors.csv')):   # the per-(Stokes, tier) views
        for r in common.read_csv(p)[1]:
            if r.get('n_det_tier') not in (None, '') and r.get('n_det') not in (None, '') and r['n_det'] != r['n_det_tier']:
                n_view += 1
    if n_view:
        bad.append(f'views: {n_view} rows with n_det != n_det_tier')
    return not bad, len(dfiles), '; '.join(bad) or f'{len(dfiles)} page directories; R2 R6 R7 pass'


FILTER_COLUMNS = ('cand_id', 'stokes', 'lst', 'band', 'date', 'snr', 'ra', 'dec', 'size_beams', 'axis_ratio',
                  'nearest_ateam', 'n_v_epoch')      # snapshot_filter_catalog.py:49-51


@content_test('filter_catalog')
def filter_rows(path, **ctx):
    """snapshot_filter_catalog.py's output contract: its columns, one row per cand_id of candidates/*/*/detections.csv,
    and coadd_root.txt beside it naming CFG.paths.COADD_SB_ROOT (snapshot_filter_catalog.py:55)."""
    hdr, rows = common.read_csv(path)
    if tuple(hdr) != FILTER_COLUMNS:
        return False, len(hdr), f'columns {hdr[:5]}... differ from the filter catalogue contract'
    run = os.path.dirname(os.path.abspath(path))
    dets = set()
    for p in glob.glob(os.path.join(run, 'candidates', '*', '*', 'detections.csv')):
        dets.update(r['cand_id'] for r in common.read_csv(p)[1])
    ids = [r['cand_id'] for r in rows]
    bad = []
    if len(ids) != len(set(ids)):
        bad.append('repeated cand_id')
    if set(ids) != dets:
        bad.append(f'{len(set(ids) ^ dets)} cand_ids differ from the pages\' detections ({len(dets)})')
    cr = os.path.join(run, 'coadd_root.txt')
    want = common.CFG().paths.COADD_SB_ROOT
    if not os.path.exists(cr) or open(cr).read().strip() != want:
        bad.append(f'coadd_root.txt does not name {want}')
    return not bad, len(rows), '; '.join(bad) or f'{len(rows)} detections'


@content_test('strat_queue')
def strat_rows(path, **ctx):
    """deep_stratified_queue.py's outputs: the queue protocol header, the map beside it listing the queue's ids in its
    order, strata labels from the configured bins, every id a page of the Stokes view's index."""
    head, ids = _queue(path)
    mp = path[:-4] + '_map.csv'
    if not os.path.exists(mp):
        return False, None, 'no _map.csv beside the queue'
    _, rows = common.read_csv(mp)
    bad = []
    if not any(h.startswith('# queue_name:') for h in head) or not any(h.startswith('# selection:') for h in head):
        bad.append('queue header')
    if [r['page'] for r in rows] != ids:
        bad.append('map pages != queue ids')
    ST = common.CFG().pages_deep
    e0, e1 = (int(x) for x in ST.STRAT_EL_BINS_DEG)
    p0, p1 = (int(x) for x in ST.STRAT_PARENT_BINS_BEAMS)
    labels = {f'{e}|{s}' for e in (f'el<{e0}', f'el{e0}-{e1}', f'el>={e1}')
              for s in (f'par<{p0}bm', f'par{p0}-{p1}bm', f'par>={p1}bm')}
    odd = [r['stratum'] for r in rows if r.get('stratum') not in labels]
    if odd:
        bad.append(f'{len(odd)} unknown strata, e.g. {odd[:2]}')
    m = re.match(r'^queue_strat_([IV])\.txt$', os.path.basename(path))
    if m:
        run = os.path.dirname(os.path.dirname(os.path.abspath(path)))
        known = set()
        for p in glob.glob(os.path.join(run, f'source_catalogs_{m.group(1)}', f'*_pages_index_{m.group(1)}.csv')):
            known.update(r['source_id'] for r in common.read_csv(p)[1])
        if known and not set(ids) <= known:
            bad.append(f'{len(set(ids) - known)} ids not pages of the Stokes {m.group(1)} view')
    return not bad and bool(ids), len(ids), '; '.join(bad) or f'{len(ids)} identities'


@content_test('pages_run_copy')
def run_copy(path, **ctx):
    """--from-run stage-in: a copied file equals its source (ctx source) and, for a CSV, has a header; a copied directory holds
    files; a linked tree resolves to an existing directory."""
    src = ctx.get('source')
    if os.path.islink(path):
        tgt = os.path.realpath(path)
        ok = os.path.isdir(tgt) and (not src or tgt == os.path.realpath(src))
        return ok, tgt, 'link resolves to the source tree' if ok else 'link does not resolve to the source tree'
    if os.path.isdir(path):
        n = sum(len(f) for _, _, f in os.walk(path))
        if src and os.path.isdir(src):
            m = sum(len(f) for _, _, f in os.walk(src))
            return n == m and n > 0, n, f'{n} files (source {m})'
        return n > 0, n, f'{n} files'
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False, None, 'missing or empty'
    if src and os.path.isfile(src) and common.sha1_file(src, 40) != common.sha1_file(path, 40):
        return False, None, 'differs from its source'
    if path.endswith('.csv'):
        with open(path, newline='') as f:
            hdr = next(csv.reader(f), [])
        return bool(hdr), len(hdr), f'{len(hdr)} columns'
    return True, os.path.getsize(path), 'copied'
