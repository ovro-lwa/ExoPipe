"""Per-Stokes catalogue views and unions served to the review servers (ports of the launchers' heredocs, byte-identical
output; D9: views are built per Stokes from the tier catalogues, page links keyed within the view's Stokes). The deep
union's review_dir column is written by `pages` (merge stage) and only READ here (deep_merge_current refuses without it).

  deep_stokes_view(U, S, review_dir)   p3s3_serve_stokes.sh:15-55 (PYV): the union rows of THIS Stokes (refuses a union
                                       without 'stokes', G2 / D9 09995b5) restricted to the (source_id, band) keys of the
                                       Stokes' wb/sb tier catalogues; members/master rows merged from those tiers;
                                       pages_index copied. review_dir='own-stokes' resolves each row's page link in the
                                       view's OWN Stokes index (the D9 residual; config serve.DEEP_VIEW_REVIEW_DIR)
  deep_multi_view(U, S, keep)          p3s3_serve_multiples.sh:86-131 (PYQ, fork branch): fork survivors of the queued
                                       sources, review_dir from the fork page indexes, members/master merged, indexes copied
  sublst_union(U)                      sublst_serve.sh:34-51 (PYU): the sb + wb survivors of both Stokes with a 'stokes'
                                       column; a row whose stokes is not its tier's REFUSES (G5)
  sublst_stokes_view(U, S)             sublst_serve.sh:75-109 (PYV): one row per (J-name, band) of the Stokes; refuses a
                                       union without 'stokes', a key held twice, a survivor missing from the union (G6)
  sublst_multi_view(B, S, QM)          sublst_cumulative.sh:82-94 (PYM): the Stokes view restricted to the J-names of the
                                       multi-integration queue
"""
import csv
import glob
import os
import shutil
from collections import Counter

from exopipe.phase3.core.record import Refusal


def _S():
    from exopipe.phase3.config import CFG
    return CFG.serve


def _write(out, fieldnames, rows, **kw):
    with open(out + '.new', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, **kw)
        w.writeheader()
        w.writerows(rows)
    os.replace(out + '.new', out)


def pages_merge_env(tier):
    """The environment of p3s3_merge_review_dir.py for ``tier`` from the `pages` section (the merge's one config key is
    pages' - decision 6 group 12; the draft named it REVIEW_DIR_MERGE_ENV, then REVIEW_DIR_MERGE)."""
    from exopipe.phase3.config import CFG
    sec = getattr(CFG, 'pages', None)
    for name in ('REVIEW_DIR_MERGE', 'REVIEW_DIR_MERGE_ENV'):
        v = getattr(sec, name, None) if sec is not None else None
        if isinstance(v, dict) and tier in v:
            return dict(v[tier])
    raise Refusal(f'config: the pages section has no review_dir merge environment for {tier} (pages.REVIEW_DIR_MERGE)')


# ---- deep -------------------------------------------------------------------------------------------------------------
def deep_merge_current(U):
    """The review_dir merge into the union survivors belongs to `pages` (refactor-chat decision 6, group 12: pages `merge`
    writes, serve reads and refuses when absent). (ok, detail): every <U>/source_catalogs/*_sources_survivors.csv carries a
    review_dir column, at least one row links a page, and every union file is newer than every page index
    (<band>_pages_index_*.csv) - an index written after the last merge means the merge of this pass has not run
    (p3s3_pass.sh:194-195 runs it after the last page build)."""
    surv = sorted(glob.glob(f'{U}/source_catalogs/*_sources_survivors.csv'))
    idx = sorted(glob.glob(f'{U}/source_catalogs/*_pages_index_*.csv'))
    if not surv:
        return False, f'no union survivors under {U}/source_catalogs (catalog union not run)'
    if not idx:
        return False, f'no page index under {U}/source_catalogs (pages not built)'
    nolink, linked, merged = [], 0, []
    for p in surv:
        with open(p) as f:
            rd = csv.DictReader(f)
            rows = list(rd)
            if not rows:            # header only: the merge leaves it untouched (p3s3_merge_review_dir.py:57 'continue')
                continue
            if 'review_dir' not in (rd.fieldnames or []):
                nolink.append(os.path.basename(p))
                continue
            merged.append(p)
            linked += sum(1 for r in rows if r.get('review_dir'))
    if nolink:
        return False, f'{len(nolink)} union files carry no review_dir column (pages merge absent), e.g. {nolink[0]}'
    if not linked:
        return False, 'no union row links a page (pages merge absent)'
    t_idx = max(os.path.getmtime(p) for p in idx)
    stale = [os.path.basename(p) for p in merged if os.path.getmtime(p) < t_idx]
    if stale:
        newest = max(idx, key=os.path.getmtime)
        return False, (f'{len(stale)} union files are older than the page index {os.path.basename(newest)} (the pages merge '
                       f'of this pass has not run), e.g. {stale[0]}')
    return True, f'{len(surv)} union files merged after the last page index; {linked} rows link a page'


def own_stokes_review_dirs(U, ST):
    """The page-link resolver of p3s3_merge_review_dir.py (unkeyed resolution order, p3s3_merge_review_dir.py:31-43,
    68-73) restricted to the view's OWN Stokes singles index <band>_pages_index_<ST>.csv (index files in sorted order).
    Returns resolve(row) -> review_dir or None."""
    exact, bbd, bb, base_only = {}, {}, {}, {}
    for p in sorted(glob.glob(f'{U}/source_catalogs/*_pages_index_{ST}.csv')):
        for r in csv.DictReader(open(p)):
            sid, rd = r['source_id'], r.get('review_dir', '')
            if not rd:
                continue
            exact.setdefault(sid, rd)
            if sid.startswith('J') and sid.count('_') >= 3:
                base, lst, date, band = sid.rsplit('_', 3)
                bbd.setdefault((base, band, date), rd)
                bb.setdefault((base, band), rd)
                base_only.setdefault(base, rd)

    def resolve(r):
        sid, band, date = r['source_id'], r.get('band', ''), r.get('best_date', '')
        rd = exact.get(sid) or bbd.get((sid, band, date)) or bb.get((sid, band))
        if not rd:
            g = sorted(glob.glob(f'{U}/candidates/I/{sid}_*_{band}'))
            g = [x for x in g if date and f'_{date}_' in x] or g
            if g:
                rd = os.path.relpath(g[0], U)
        return rd or base_only.get(sid)
    return resolve


def deep_stokes_view(U, ST, review_dir='union'):
    """Write <U>/source_catalogs_<ST>/. Returns {'rows': n, 'review_dir_changed': [(source_id, band, old, new)]}."""
    S = _S()
    V = f'{U}/source_catalogs_{ST}'
    os.makedirs(V, exist_ok=True)
    tiers = [f'{U}/source_catalogs_{ST}_{t}' for t in S.DEEP_SINGLES_TIERS]
    keep = set()
    for d in tiers:
        for p in glob.glob(f'{d}/*_sources_survivors.csv'):
            for r in csv.DictReader(open(p)):
                keep.add((r['source_id'], r['band']))
    resolve = own_stokes_review_dirs(U, ST) if review_dir == 'own-stokes' else None
    if review_dir not in ('union', 'own-stokes'):
        raise Refusal(f'serve.DEEP_VIEW_REVIEW_DIR={review_dir!r}: expected union or own-stokes')
    n, changed = 0, []
    for p in glob.glob(f'{U}/source_catalogs/*_sources_survivors.csv'):
        rr = list(csv.DictReader(open(p)))
        if rr and 'stokes' not in rr[0]:
            raise Refusal(f'G2 deep union without Stokes: {p}: union row carries no stokes column (built before the D9 fix) '
                          f'- REFUSED (p3s3_serve_stokes.sh:28-29)')
        rows = [r for r in rr if r['stokes'] == ST and (r['source_id'], r['band']) in keep]
        if resolve is not None:
            for r in rows:
                new = resolve(r)
                if new and new != r.get('review_dir', ''):
                    changed.append((r['source_id'], r['band'], r.get('review_dir', ''), new))
                    r['review_dir'] = new
        out = f'{V}/{os.path.basename(p)}'
        if not rows:
            if os.path.exists(out):
                os.remove(out)
            continue
        _write(out, list(rows[0].keys()), rows)
        n += len(rows)
    for kind in ('source_members', 'sources_master'):
        seen = {}
        for d in tiers:
            for p in glob.glob(f'{d}/*_{kind}.csv'):
                seen.setdefault(os.path.basename(p), []).append(p)
        for name, ps in seen.items():
            rows, hdr = [], None
            for p in ps:
                for r in csv.DictReader(open(p)):
                    hdr = hdr or list(r.keys())
                    rows.append(r)
            if hdr:
                _write(f'{V}/{name}', hdr, rows, extrasaction='ignore')
    for p in glob.glob(f'{U}/source_catalogs/*_pages_index_{ST}.csv'):
        shutil.copy2(p, f'{V}/{os.path.basename(p)}')
    return {'rows': n, 'review_dir_changed': changed, 'dir': V}


def deep_multi_view(U, ST, keep):
    """Write <U>/views/source_catalogs_<ST>_multi/ (fork mode). Returns {'rows': n, 'dir': V}."""
    S = _S()
    V = f'{U}/views/source_catalogs_{ST}_multi'
    os.makedirs(V, exist_ok=True)
    n = 0
    fdirs = [f'{U}/source_catalogs_{ST}_{ft}' for ft in S.DEEP_FORK_TIERS]
    src_files = sorted(sum((glob.glob(f'{d}/*_sources_survivors.csv') for d in fdirs), []))
    rd_exact, rd_bbd, rd_bb = {}, {}, {}
    for ip in glob.glob(f'{U}/source_catalogs/*_pages_index_{ST}_f*.csv'):
        for r in csv.DictReader(open(ip)):
            sid, rd = r['source_id'], r.get('review_dir', '')
            if not rd or sid.count('_') < 3:
                continue
            base, lst, date, band = sid.rsplit('_', 3)
            rd_exact.setdefault(sid, rd)
            rd_bbd.setdefault((base, band, date), rd)
            rd_bb.setdefault((base, band), rd)
    for p in src_files:
        rows = [r for r in csv.DictReader(open(p)) if (r['source_id'], r['band']) in keep]
        for r in rows:
            bc = r.get('best_cand_id', '')
            lst = bc.split('_')[1] if bc.count('_') >= 3 else ''
            r['review_dir'] = (rd_exact.get(f"{r['source_id']}_{lst}_{r.get('best_date', '')}_{r['band']}")
                               or rd_bbd.get((r['source_id'], r['band'], r.get('best_date', '')))
                               or rd_bb.get((r['source_id'], r['band'])) or r.get('review_dir', ''))
        out = f'{V}/{os.path.basename(p)}'
        if not rows:
            if os.path.exists(out):
                os.remove(out)
            continue
        _write(out, list(rows[0].keys()), rows)
        n += len(rows)
    for kind in ('source_members', 'sources_master'):
        seen_f = {}
        for d in fdirs:
            for p in glob.glob(f'{d}/*_{kind}.csv'):
                seen_f.setdefault(os.path.basename(p), []).append(p)
        for name, ps in seen_f.items():
            rows, hdr = [], None
            for p in ps:
                for r in csv.DictReader(open(p)):
                    if (r['source_id'], r['band']) in keep:
                        hdr = hdr or list(r.keys())
                        rows.append(r)
            if hdr:
                _write(f'{V}/{name}', hdr, rows, extrasaction='ignore')
    for p in glob.glob(f'{U}/source_catalogs/*_pages_index_{ST}_f*.csv'):
        shutil.copy2(p, f'{V}/{os.path.basename(p)}')
    return {'rows': n, 'dir': V}


# ---- sub-LST ----------------------------------------------------------------------------------------------------------
def sublst_union(U):
    """Write <U>/source_catalogs/<band>_sources_survivors.csv (both Stokes, 'stokes' column). Returns the band count."""
    S = _S()
    os.makedirs(f'{U}/source_catalogs', exist_ok=True)
    by = {}
    for ST in 'VI':
        for t in S.SUBLST_TIERS:
            for p in glob.glob(f'{U}/source_catalogs_{ST}_{t}/*_sources_survivors.csv'):
                by.setdefault(os.path.basename(p), []).append((ST, p))
    staged = []
    for name, ps in by.items():
        rows, hdr = [], []
        for ST, p in ps:
            for r in csv.DictReader(open(p)):
                if r.get('stokes', ST) != ST:
                    raise Refusal(f'G5 sub-LST union Stokes: {p} row {r["source_id"]} carries stokes={r["stokes"]} '
                                  f'(sublst_serve.sh:45)')
                r['stokes'] = ST
                rows.append(r)
                hdr += [k for k in r if k not in hdr]
        staged.append((name, hdr, rows))
    # production writes file by file and exits at the first refusing row; the refusal is checked for every file first
    # here, so a refusal leaves every union file as it was (the build is not swapped in either way: G11)
    for name, hdr, rows in staged:
        _write(f'{U}/source_catalogs/{name}', hdr, rows)
    return len(by)


def sublst_stokes_view(U, ST):
    """Write <U>/source_catalogs_<ST>/ (sub-LST). Returns {'rows': n, 'dir': V}."""
    S = _S()
    V = f'{U}/source_catalogs_{ST}'
    os.makedirs(V, exist_ok=True)
    tiers = [f'{U}/source_catalogs_{ST}_{t}' for t in S.SUBLST_TIERS]
    keep = set()
    for d in tiers:
        for p in glob.glob(f'{d}/*_sources_survivors.csv'):
            for r in csv.DictReader(open(p)):
                keep.add((r['source_id'], r['band']))
    views, got = [], set()
    for p in glob.glob(f'{U}/source_catalogs/*_sources_survivors.csv'):
        rows = list(csv.DictReader(open(p)))
        if rows and 'stokes' not in rows[0]:
            raise Refusal(f'G6 sub-LST view keys: {p} carries no stokes column (union not rebuilt) (sublst_serve.sh:87)')
        rows = [r for r in rows if r['stokes'] == ST and (r['source_id'], r['band']) in keep]
        dup = sorted(k for k, c in Counter((r['source_id'], r['band']) for r in rows).items() if c > 1)
        if dup:
            raise Refusal(f'G6 sub-LST view keys: {len(dup)} (J-name, band) keys held by two Stokes {ST} rows in {p}, '
                          f'e.g. {dup[0]} (sublst_serve.sh:89-90)')
        for r in rows:
            del r['stokes']
            got.add((r['source_id'], r['band']))
        views.append((p, rows))
    if keep - got:
        raise Refusal(f'G6 sub-LST view keys: {len(keep - got)} Stokes {ST} survivors rows missing from the union, e.g. '
                      f'{sorted(keep - got)[0]} (sublst_serve.sh:93)')
    n = 0
    for p, rows in views:
        out = f'{V}/{os.path.basename(p)}'
        if not rows:
            if os.path.exists(out):
                os.remove(out)
            continue
        _write(out, list(rows[0].keys()), rows)
        n += len(rows)
    for kind in ('source_members', 'sources_master'):
        for d in tiers:
            for p in glob.glob(f'{d}/*_{kind}.csv'):
                shutil.copy2(p, f'{V}/{os.path.basename(p)}')
    for p in glob.glob(f'{U}/source_catalogs/*_pages_index_{ST}.csv'):
        shutil.copy2(p, f'{V}/{os.path.basename(p)}')
    return {'rows': n, 'dir': V}


def sublst_multi_view(U, ST, QM):
    """Write <U>/source_catalogs_<ST>_multi/ from the Stokes view, restricted to the multi-integration queue's J-names
    (bare J-name, as sublst_cumulative.sh:84). Returns {'sources': n, 'dir': V}."""
    ids = {l.strip().split('_')[0] for l in open(QM) if l.startswith('J')}
    V = f'{U}/source_catalogs_{ST}_multi'
    os.makedirs(V, exist_ok=True)
    for p in glob.glob(f'{U}/source_catalogs_{ST}/*.csv'):
        rows = list(csv.DictReader(open(p)))
        if 'source_id' in (rows[0] if rows else {}):
            rows = [r for r in rows if r['source_id'].split('_')[0] in ids]
        if not rows:
            continue
        with open(f'{V}/{os.path.basename(p)}', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return {'sources': len(ids), 'dir': V}
