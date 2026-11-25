"""THE FALSE-POSITIVE FILTER STEP of a served snapshot view: <view>/fp_filters.csv + fp_filters.json, the two default hides of the
combined view's pages p8986-p8989 (serve --tier snapshot --family comb, stage 'hides', after VIEW-READY-FOR-HIDES and before the
pages are registered).

The files (exopipe_ops review_filters.py:80-92, the reader every review page runs): fp_filters.csv holds one row per hidden or
marked detection - cand_id, its position of record ra / dec (the detection's detections.csv columns, which the page keys by
(cand_id, ra and dec to 4 decimals of a degree)), then one column per category, a state of that category or empty; fp_filters.json
describes the categories {"cats": [{key, stokes, states, default_off, label, help}], "metrics": []}. A detection whose row is
absent or whose column is empty takes the category's LAST state. The reader skips lines starting '#'; a file it cannot read is
dropped silently by the page (review_filters.py:333-358), so this step content-tests both files before anything is installed.

Port of the refactor chat's production tools for the Release 1 view (review_items_20261031/hides_20261115/: run_hides.sh,
parallel_collect.py, newview_hides.py flood + write), which wrote the served Run_SNAPCOMB_20261116/fp_filters.*. THE FILTER CONTENT
IS THE SNAPSHOT CHAT'S and is ported unchanged (config serve.SNAPVIEW_ALIGN_*, SNAPVIEW_FLOOD_*, SNAPVIEW_FP_*; a filter that
hides sources needs Gregg's go):

  collect   per page of the view, in the pages' order (V 10-s, I 10-s, V 1-min, I 1-min = p8986-p8989): the LISTED sources (the
            queue's non-comment lines) and, for each survivors row of the page's catalog directory (*_sources_survivors.csv in
            sorted order) whose identity (identity_id, else source_id) is listed, the rows of <view>/<review_dir>/detections.csv of
            the page's own tier (a row with an empty tier counts) - exactly as classify_transients assembles a page (parallel_collect.py,
            "validated against the page builds before use"). serve.SNAPVIEW_HIDES_READERS threads read the files; the order is the
            input order.
  flood     detections de-duplicated across the four pages by (cand_id, ra and dec rounded to 4 decimals; the raw strings when
            not numbers), the first page's row kept; the load of a night-band = its distinct detections per (Stokes, tier, LST,
            night, band); its base = the median of the OTHER nights' loads at the same (Stokes, tier, LST, band), over every night
            seen at that (Stokes, tier, LST) - a night without detections in the band counts 0; a FLOOD HOUR is a night-band with
            load >= serve.SNAPVIEW_FLOOD_MIN and load >= serve.SNAPVIEW_FLOOD_FACTOR x max(base, 1) (Gregg 2026-10-31
            "recommendations approved"; "100x works" 2026-11-08). newview_hides.flood line for line.
  write     fp_c_align: state 0 (alignment failed, hidden by default) on the cells of serve.SNAPVIEW_ALIGN_FAILED, state 1 (re-searched,
            held; shown by default, Gregg 2026-11-03 "Only 10-30") on serve.SNAPVIEW_ALIGN_HELD, by (LST, night) and band; fp_c_flood:
            state 0 on the flood hours. One row per distinct detection with any state, in the de-duplicated order. The json holds the
            two categories of config (help texts with this view's counts) and no metric.

Refusals (gates refuse; nothing is written on a refusal): fp_filters.csv or .json already in the output directory (never
overwritten - production's 'REFUSED: ... exists - not overwritten'; one without the other is named HALF-INSTALLED); a page whose
queue or catalog directory is absent; no listed detection at all; a listed detection on an excluded cell-band (core.exclusions
read now, the snapshot gate's reader - production's check named the three excluded 23h 82 MHz cells literally; this reads the
whole list); the written files failing their content test (class fp_filters). Deliberate deviations: the exclusion check reads
every data_decisions row instead of production's three literal cells (stricter: it refuses, it never hides); the CSV's first
comment line names this module; the collected detections and the flood list are kept as the stage's records (production kept them
in the tools' dets/ directory, the cross-tier pair builder's input); flood-list rows of equal ratio follow their key (production's
order of ties was not reproducible); a listed source's page directory without detections.csv refuses (production's collector read
it as no detections); the json is renamed into place before the CSV (production: the CSV first), so an interrupted install never
leaves the page a CSV without its json.
"""
import collections
import csv
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor

from exopipe.phase3.core import placement

FP_FILE, FP_META = 'fp_filters.csv', 'fp_filters.json'     # review_filters.py:92
DET_FIELDS = ('cand_id', 'lst', 'date', 'band', 'ra', 'dec', 'snr', 'stokes', 'tier')   # parallel_collect.py F
DETS_HEADER = ['page', 'source', 'default_listed', 'stokes', 'tier', 'lst', 'night', 'band', 'cand_id', 'ra', 'dec', 'snr']
FLOOD_HEADER = ['tier', 'stokes', 'subtier', 'lst', 'night', 'band', 'detections', 'median_other_nights', 'ratio']


class HidesRefusal(Exception):
    pass


def _S():
    from exopipe.phase3.config import CFG
    return CFG.serve


def listed_ids(queue):
    """The listed identities of a page: the queue's non-empty lines not starting '#' (parallel_collect.py)."""
    with open(queue) as f:
        return {ln.strip() for ln in f if ln.strip() and not ln.startswith('#')}


def page_jobs(view, page):
    """[(identity, review_dir)] of a page: each survivors row of its catalog directory whose identity is listed, files in sorted
    order, rows in file order. ``page``: dict with queue, catalog (paths relative to the view), kind, stokes, id."""
    import glob
    q, cdir = os.path.join(view, page['queue']), os.path.join(view, page['catalog'])
    if not os.path.isfile(q):
        raise HidesRefusal(f'page {page["id"]}: no queue {q}')
    if not os.path.isdir(cdir):
        raise HidesRefusal(f'page {page["id"]}: no catalog directory {cdir}')
    listed = listed_ids(q)
    jobs = []
    for p in sorted(glob.glob(os.path.join(cdir, '*_sources_survivors.csv'))):
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                sid = (r.get('identity_id') or '').strip() or r['source_id']
                if sid in listed:
                    jobs.append((sid, r['review_dir']))
    return listed, jobs


def _read(view, tier, job):
    sid, rd = job
    p = os.path.join(view, rd, 'detections.csv')
    try:
        with open(p, newline='') as f:
            rows = [{k: d.get(k, '') for k in DET_FIELDS} for d in csv.DictReader(f)]
    except FileNotFoundError:
        return sid, None
    return sid, [d for d in rows if not (d.get('tier') and d['tier'] != tier)]


def collect(view, page, readers=None):
    """The page's listed detections: [{page, source, stokes, tier, lst, night, band, cand_id, ra, dec, snr}] in input order, plus
    {listed, review_dirs, missing}: the listed identities, the review directories read, those without a detections.csv."""
    listed, jobs = page_jobs(view, page)
    tier, out, missing = page['kind'], [], []
    with ThreadPoolExecutor(readers or _S().SNAPVIEW_HIDES_READERS) as ex:
        for (sid, rd), (_, dets) in zip(jobs, ex.map(lambda j: _read(view, tier, j), jobs, chunksize=64)):
            if dets is None:
                missing.append(rd)
                continue
            for d in dets:
                # production wrote these rows to flood_dets_<page>.csv and read them back: a short row's missing field (None)
                # became '' - and its date the text 'None' (str(None)[:10]) - before the flood rule and the write saw it
                g = {k: ('' if v is None else v) for k, v in d.items()}
                out.append(dict(page=page['id'], source=sid, stokes=g['stokes'], tier=g['tier'], lst=g['lst'],
                                night=str(d['date'])[:10], band=g['band'], cand_id=g['cand_id'], ra=g['ra'], dec=g['dec'],
                                snr=g['snr']))
    return out, dict(listed=len(listed), review_dirs=len(jobs), missing=missing)


def write_dets(path, dets):
    """newview_hides.collect's flood_dets_<page>.csv format (the cross-tier pair builder's snapshot input). Placed (core/
    placement.atomic: the existing file's group, mode and ACL, or the directory's rule for a new file)."""
    with placement.atomic(path) as tmp, open(tmp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(DETS_HEADER)
        for d in dets:
            w.writerow([d['page'], d['source'], 1, d['stokes'], d['tier'], d['lst'], d['night'], d['band'], d['cand_id'], d['ra'],
                        d['dec'], d['snr']])


def det_key(d):
    """(cand_id, ra, dec to 4 decimals) - the page's key of a detection; the raw strings when not numbers (newview_hides.flood)."""
    try:
        return (d['cand_id'], round(float(d['ra']), 4), round(float(d['dec']), 4))
    except ValueError:
        return (d['cand_id'], d['ra'], d['dec'])


def dedupe(pages_dets):
    """{key: detection} over the pages in order, the first page's row kept."""
    uniq = {}
    for dets in pages_dets:
        for d in dets:
            uniq.setdefault(det_key(d), d)
    return uniq


def flood(uniq, factor=None, n_min=None):
    """The flood hours: (set of (Stokes, tier, LST, night, band), load Counter, base dict) - newview_hides.flood."""
    S = _S()
    factor = S.SNAPVIEW_FLOOD_FACTOR if factor is None else factor
    n_min = S.SNAPVIEW_FLOOD_MIN if n_min is None else n_min
    load = collections.Counter((d['stokes'], d['tier'], d['lst'], d['night'], d['band']) for d in uniq.values())
    nights, bands = collections.defaultdict(set), collections.defaultdict(set)
    for (s, t, l, n, b) in load:
        nights[(s, t, l)].add(n)
        bands[(s, t, l)].add(b)
    base = {}
    for (s, t, l), ns in nights.items():
        for b in bands[(s, t, l)]:
            for n in ns:
                others = [load.get((s, t, l, n2, b), 0) for n2 in ns if n2 != n]
                base[(s, t, l, n, b)] = statistics.median(others) if others else 0
    fl = {k for k, v in load.items() if v >= n_min and v >= factor * max(base.get(k, 0), 1)}
    return fl, load, base


def write_flood_list(path, fl, load, base, factor=None):
    """newview_hides.py's flood_list_K<factor>.csv: the flood night-bands by falling ratio (detections over the median of the
    other nights, at least 1). Ties in the ratio follow the key (Stokes, tier, LST, night, band): production's tie order was its
    process's set order (string hashing is seeded per process), not reproducible - the same rows, ties ordered."""
    factor = _S().SNAPVIEW_FLOOD_FACTOR if factor is None else factor
    with placement.atomic(path) as tmp, open(tmp, 'w', newline='') as f:      # placed, as write_dets
        w = csv.writer(f)
        w.writerow(FLOOD_HEADER)
        for k in sorted(fl, key=lambda k: (-load[k] / max(base.get(k, 0), 1), k)):
            w.writerow(['snapshot', k[0], k[1], k[2], k[3], k[4], load[k], base.get(k, 0), round(load[k] / max(base.get(k, 0), 1), 1)])


def excluded_hits(uniq, ex=None):
    """{(lst, night, band): n} of the distinct listed detections on an excluded cell-band (core.exclusions read now, as the
    snapshot serve-time gate reads it: exclusion_gate.fresh_exclusions('snapshot'))."""
    from exopipe.phase3.lib.serve import exclusion_gate
    ex = ex or exclusion_gate.fresh_exclusions('snapshot')
    cache, hits = {}, collections.Counter()
    for d in uniq.values():
        k = (d['lst'], d['night'], d['band'])
        if k not in cache:
            cache[k] = bool(ex.is_excluded(k[0], k[2], k[1]))
        if cache[k]:
            hits[k] += 1
    return hits


def _cells(spec):
    return {tuple(k): set(v) for k, v in (spec or {}).items()}


def rows_and_meta(uniq, fl):
    """The CSV rows [[cand_id, ra, dec, align state, flood state]] and the json (newview_hides.write), and the counts."""
    S = _S()
    failed, held = _cells(S.SNAPVIEW_ALIGN_FAILED), _cells(S.SNAPVIEW_ALIGN_HELD)
    ca, cf = dict(S.SNAPVIEW_FP_ALIGN), dict(S.SNAPVIEW_FP_FLOOD)
    A0, A0b, F0 = ca['states'][0], ca['states'][1], cf['states'][0]
    rows, na, nh, nf = [], 0, 0, 0                 # rows; detections hidden as alignment-failed, held, in a flood hour
    for d in uniq.values():
        a = d['band'] in failed.get((d['lst'], d['night']), ())
        h = d['band'] in held.get((d['lst'], d['night']), ())
        f_ = (d['stokes'], d['tier'], d['lst'], d['night'], d['band']) in fl
        if a or h or f_:
            rows.append([d['cand_id'], d['ra'], d['dec'], A0 if a else (A0b if h else ''), F0 if f_ else ''])
            na += a
            nh += h
            nf += f_
    n_nights = len({k[3] for k in fl})             # nights with a flood hour
    ca['help'] = ca['help'].format(n_held=nh)
    cf['help'] = cf['help'].format(factor=S.SNAPVIEW_FLOOD_FACTOR, n_min=S.SNAPVIEW_FLOOD_MIN, n_hours=len(fl), n_det=nf,
                                   n_nights=n_nights)
    order = ('key', 'stokes', 'states', 'default_off', 'label', 'help')     # production's key order (the live json's)
    meta = {'cats': [{k: c[k] for k in order} for c in (ca, cf)], 'metrics': []}
    counts = dict(rows=len(rows), align_failed=na, align_held=nh, flood=nf, flood_hours=len(fl), flood_nights=n_nights,
                  distinct=len(uniq))
    return rows, meta, counts


def installed_refusal(out):
    """The refusal text when ``out`` holds either file already, else None: both (a view's hides are written once, before its
    pages are registered), or a HALF-INSTALLED pair - the CSV without its json, or the json without its CSV (an install
    stopped between its two renames); the page reads the filters only when both exist (review_filters.py:334-335), so either
    half shows no hides, silently. Never overwritten: whoever finds a half-installed pair decides what to remove."""
    c, j = (os.path.lexists(os.path.join(out, p)) for p in (FP_FILE, FP_META))
    if c and j:
        return (f'{os.path.join(out, FP_FILE)} and {FP_META} exist - not overwritten (a view\'s hides are written once, '
                f'before its pages are registered)')
    if c:
        return (f'HALF-INSTALLED hides: {os.path.join(out, FP_FILE)} exists without {FP_META} (the page shows no hides without '
                f'both files) - not overwritten; remove the CSV to rebuild both')
    if j:
        return (f'HALF-INSTALLED hides: {os.path.join(out, FP_META)} exists without {FP_FILE} (an install stopped between its '
                f'two renames; the page shows no hides) - not overwritten; remove the json to rebuild both')
    return None


def write_files(out, rows, meta, counts, log=print):
    """<out>/fp_filters.csv (the note line, header, rows; csv.writer's line ends, as production) and fp_filters.json (indent 1):
    both written as .part, both content-tested (check_files; a file the page cannot read is dropped by the page silently), then
    placed together (core/placement.replace_many: each .part given the directory's rule - both files are new here - where it
    lies, then renamed): the json renamed into place BEFORE the CSV (the CSV is the file the page keys its rows on, so no state
    ever shows a CSV without its json; production renamed the CSV first). A placement refusal of either .part renames neither;
    a failed CSV rename after the json's is a PlacementRefusal naming the json placed. On any failure the .part files are
    removed, and so is the json this call renamed into place when the CSV is not installed (nothing is left half-installed).
    Refuses when either file exists (installed_refusal); a .part left by an earlier attempt that died is this step's own
    intermediate, removed with a note."""
    S = _S()
    why = installed_refusal(out)
    if why:
        raise HidesRefusal(why)
    cp, jp = os.path.join(out, FP_FILE), os.path.join(out, FP_META)
    parts = (cp + '.part', jp + '.part')
    for p in parts:
        if os.path.lexists(p):
            log(f'fp_filters: removing {p} left by an earlier attempt that did not finish')
            os.remove(p)
    ours = None                            # the json .part's inode: the file this call renames into place
    try:
        with open(parts[0], 'x', newline='') as f:
            f.write(S.SNAPVIEW_FP_NOTE.format(factor=S.SNAPVIEW_FLOOD_FACTOR, n_failed=f'{counts["align_failed"]:,}',
                                              n_held=f'{counts["align_held"]:,}', n_flood=f'{counts["flood"]:,}'))
            w = csv.writer(f)
            w.writerow(['cand_id', 'ra', 'dec', 'fp_c_align', 'fp_c_flood'])
            w.writerows(rows)
        with open(parts[1], 'x') as f:
            json.dump(meta, f, indent=1)
        ok, n, detail = check_files(parts[0], parts[1])
        if not ok or n != len(rows):
            raise HidesRefusal(f'the written fp_filters fail their content test ({detail}; {n} rows read back, {len(rows)} '
                               f'written) - nothing installed')
        ours = os.stat(parts[1]).st_ino
        placement.replace_many([(parts[1], jp), (parts[0], cp)])
    except BaseException:
        for p in parts:
            if os.path.lexists(p):
                os.remove(p)
        if (ours is not None and not os.path.lexists(cp) and os.path.lexists(jp)
                and os.stat(jp).st_ino == ours):   # the CSV's rename failed: take back the json this call installed
            os.remove(jp)
        raise
    return cp, jp


def pages_of(plan_pages):
    """The page dicts collect reads, from serve.PAGES pages (lib/serve/plan.Page) of the combined view, in config order; id =
    p<production port> (the cross-tier tools key snapshot pages by it)."""
    return [dict(id=f'p{pg.port_cfg}', queue=pg.c['queue'], catalog=pg.c['catalog'], kind=pg.kind, stokes=pg.stokes)
            for pg in plan_pages]


def build(view, out, pages, records=None, ex=None, readers=None, log=print, refuse_missing=True):
    """The whole step: collect every page of ``view`` (collect_pages), then the filters from what was collected (filters_from)
    into ``out`` (the view itself when served; a scratch directory for a content test). ``records``: a directory for the collected
    detections and the flood list (flood_dets_<page>.csv, flood_list_K<factor>.csv). Returns a summary dict. Raises HidesRefusal,
    before anything is written into ``out``."""
    why = installed_refusal(out)               # before any read: a refusal costs nothing
    if why:
        raise HidesRefusal(why)
    per_page, info = collect_pages(view, pages, records=records, readers=readers, log=log, refuse_missing=refuse_missing)
    if not any(per_page):
        raise HidesRefusal(f'no listed detection on any page of {view} - nothing to filter (a view without detections is not served)')
    cp, jp, counts = filters_from(per_page, out, records=records, ex=ex, log=log)
    return dict(csv=cp, json=jp, pages=info, counts=counts, missing=sum(v['missing'] for v in info.values()))


def collect_pages(view, pages, records=None, readers=None, log=print, refuse_missing=True):
    """([detections of each page], {page id: counts}): collect over ``pages`` in order (production's four parallel_collect.py
    runs), each page's detections written to <records>/flood_dets_<page>.csv. Refuses when a listed source's page directory has no
    detections.csv (``refuse_missing``; production's collector read such a directory as no detections: the page would list a
    source whose detections the hides never saw)."""
    per_page, info = [], {}
    for pg in pages:
        dets, inf = collect(view, pg, readers)
        log(f'{pg["id"]}: {inf["listed"]:,} listed, {inf["review_dirs"]:,} review directories, {len(dets):,} detections'
            + (f', {len(inf["missing"])} review directories without detections.csv' if inf['missing'] else ''))
        per_page.append(dets)
        info[pg['id']] = dict(listed=inf['listed'], review_dirs=inf['review_dirs'], detections=len(dets),
                              missing=len(inf['missing']), missing_first=inf['missing'][:3])
        if records:
            os.makedirs(records, exist_ok=True)
            write_dets(os.path.join(records, f'flood_dets_{pg["id"]}.csv'), dets)
    missing = {p: v for p, v in info.items() if v['missing']}
    if missing and refuse_missing:
        raise HidesRefusal('listed sources whose page directory has no detections.csv: ' + '; '.join(
            f'{p}: {v["missing"]}, e.g. {v["missing_first"]}' for p, v in missing.items()) + ' - the view is incomplete; no fp_filters '
            'written')
    return per_page, info


def filters_from(per_page, out, records=None, ex=None, log=print):
    """(csv path, json path, counts): the filters from the collected detections of the pages in order (newview_hides.py write):
    de-duplicate (the first page's row kept), refuse a detection on an excluded cell-band, the flood hours (the flood list to
    <records>/flood_list_K<factor>.csv), the rows and the json, written into ``out``. The rows of production's flood_dets files
    read with csv.DictReader are valid input (production's own round trip)."""
    S = _S()
    uniq = dedupe(per_page)
    if not uniq:
        raise HidesRefusal('no listed detection on any page - nothing to filter (a view without detections is not served)')
    hits = excluded_hits(uniq, ex)
    if hits:
        raise HidesRefusal(f'the view lists {sum(hits.values())} detections on excluded cell-bands ({"; ".join(f"{k[0]} {k[1]} {k[2]}: {n}" for k, n in sorted(hits.items())[:5])}) '
                           f'- exclusion is total: re-build the view; no fp_filters written')
    fl, load, base = flood(uniq)
    if records:
        os.makedirs(records, exist_ok=True)
        write_flood_list(os.path.join(records, f'flood_list_K{S.SNAPVIEW_FLOOD_FACTOR}.csv'), fl, load, base)
    rows, meta, counts = rows_and_meta(uniq, fl)
    log(f'{len(uniq):,} distinct detections; flood night-bands {len(fl)} holding {counts["flood"]:,} detections on '
        f'{counts["flood_nights"]} nights; alignment failed {counts["align_failed"]:,}, held {counts["align_held"]:,}')
    cp, jp = write_files(out, rows, meta, counts, log=log)
    return cp, jp, counts


# ---- the content test (class fp_filters) -------------------------------------------------------------------------------------
def check_files(csv_path, json_path=None):
    """(ok, n rows, detail): the json's categories well formed (key fp_c_*, states, default_off within states, label, help),
    the CSV's header = cand_id, ra, dec + the json's category keys, every cell empty or a state of its category, no row without a
    state, no two rows with one key (cand_id, ra, dec to 4 decimals), ra / dec numbers, the json's metrics a list."""
    json_path = json_path or os.path.join(os.path.dirname(csv_path), FP_META)
    bad = []
    try:
        meta = json.load(open(json_path))
    except (OSError, ValueError) as e:
        return False, None, f'{json_path} unreadable ({e})'
    cats = meta.get('cats') or []
    if not isinstance(meta.get('metrics', []), list):
        bad.append('metrics is not a list')
    states = {}
    for c in cats:
        k = str(c.get('key', ''))
        if not k.startswith('fp_c_') or not c.get('states') or not c.get('label') or not c.get('help'):
            bad.append(f'category {k!r} without key fp_c_*, states, label or help')
            continue
        if any(s not in c['states'] for s in (c.get('default_off') or [])):
            bad.append(f'{k}: default_off names a state it does not have')
        states[k] = set(c['states'])
    if not states:
        bad.append('no category')
    with open(csv_path, newline='') as f:
        rd = csv.DictReader(l for l in f if not l.startswith('#'))
        hdr = rd.fieldnames or []
        if hdr[:3] != ['cand_id', 'ra', 'dec'] or set(hdr[3:]) != set(states):
            bad.append(f'header {hdr} is not cand_id, ra, dec + the categories {sorted(states)}')
        n, seen = 0, set()
        for r in rd:
            n += 1
            try:
                k = (r['cand_id'], round(float(r['ra']), 4), round(float(r['dec']), 4))
            except (TypeError, ValueError):
                bad.append(f'row {n}: ra / dec not numbers'); continue
            if not r['cand_id'] or k in seen:
                bad.append(f'row {n}: no cand_id or a repeated key {k}')
            seen.add(k)
            vals = [(c, r.get(c) or '') for c in states]
            if not any(v for _, v in vals):
                bad.append(f'row {n}: no state')
            bad += [f'row {n}: {c} = {v!r} is not one of its states' for c, v in vals if v and v not in states[c]]
            if len(bad) > 20:
                break
    if n == 0 and not bad:
        bad.append('no rows')
    return not bad, n, '; '.join(bad[:3]) or f'{n} rows, categories {", ".join(sorted(states))}'
