"""Deep item M14 (DEFECTS_MEDIUM M14; deferred by the deep chat to this command - ledger
FIX-REFACTOR-MEDIUM-DEFECTS-M1-M19-DEEP-20261013 - and carried ONLY in the new code; FOLD_IN 'OPEN - refactor owns').

Today (pages.PANEL_DIR_LAYOUT = 'shared', the window default): build_i14 writes every set's and both Stokes' panels into
candidates/I/<page id>/ and the review server shows the NEWEST panel per (LST, night, band) of a directory, so a page can
show another set's or the other Stokes' panel (Run_P3S5 2026-10-13: 1,373 of 12,910 directories hold more than one panel
for one key; 4 J-names share a directory between V and I); the deep review_dir merge runs without the Stokes key.

M14 (pages.PANEL_DIR_LAYOUT = 'per-set' with pages.DEEP_MERGE = 'keyed'):
  1. panel directory candidates/<K>/<page id> per page set K - build_i14 is run IN PLACE against a per-set OVERLAY run
     directory <U>/_pagesets/<K>/ whose entries are links to <U>'s, except candidates/I -> <U>/candidates/<K> (the builder
     hard-codes candidates/I, build_i14:796, 865); the index rows it writes (review_dir candidates/I/<page id>) are then
     rewritten to candidates/<K>/<page id>;
  2. the review_dir merge into the union survivors keyed by (Stokes, source_id): each row resolves only within the index
     files of ITS OWN singles set (K = <Stokes>_wb for a band, <Stokes>_sb for a subband); a key naming two pages of its set
     refuses (the Stokes-keyed rules of p3s3_merge_review_dir.py:9-22, 44-83, generalised to the per-set layout);
  3. fork-index naming: fork sets keep <band>_pages_index_<S>_f<t>b.csv (build_i14:999-1001); the keyed merge recognises
     those names and never takes a link from them (they serve the multiples page), where the in-place Stokes-keyed merge
     treats them as 'index naming no Stokes' and refuses (p3s3_merge_review_dir.py:34-35, 44).
Switching is amendment A-M14 (refactor chat, at cutover): it changes every deep page directory, needs a full deep re-render
(16,511 panels, ~2.5 h) and serve's views/review_dir checks to follow (spec 12.1).
"""
import csv
import glob
import os
import re

from exopipe.phase3.lib.pages import common

OVERLAY_DIR = '_pagesets'
_INDEX_RE = re.compile(r'^(?P<band>[^_]+)_pages_index_(?P<tag>[IV](?:_f[ws]b)?)\.csv$')
_NOT_LINKED = ('candidates', OVERLAY_DIR, '_records')
_PRECREATE = ('source_catalogs', 'judgements', 'localization_figs', 'candidates')


def set_of(stokes, band):
    """The singles page set of a (Stokes, band) row: <S>_sb for a subband (…MHz), <S>_wb for a band."""
    return f"{stokes}_{'sb' if 'MHz' in band else 'wb'}"


def overlay(U, K, dry_run=False):
    """The run directory build_i14 is given for set K in the per-set layout (point 1 of the module docstring). Links are
    (re)made for every entry of U at call time; the builder's run-level caches written by os.replace become overlay-local
    files (page_noise_cache*.json: internal, not products)."""
    ov = os.path.join(U, OVERLAY_DIR, K)
    if dry_run:
        return ov
    for d in _PRECREATE:
        os.makedirs(os.path.join(U, d), exist_ok=True)
    os.makedirs(os.path.join(U, 'candidates', K), exist_ok=True)
    os.makedirs(os.path.join(ov, 'candidates'), exist_ok=True)
    link = os.path.join(ov, 'candidates', 'I')
    _link(link, os.path.join(U, 'candidates', K))
    for name in sorted(os.listdir(U)):
        if name in _NOT_LINKED:
            continue
        dst = os.path.join(ov, name)
        if os.path.exists(dst) and not os.path.islink(dst) and os.path.isfile(dst) and name.startswith('page_noise_cache'):
            continue                                  # an overlay-local cache written by the builder (see docstring)
        _link(dst, os.path.join(U, name))
    return ov


def _link(dst, target):
    if os.path.islink(dst):
        if os.readlink(dst) == target:
            return
        os.remove(dst)
    elif os.path.exists(dst):
        raise common.Refusal(f'M14 overlay: {dst} exists and is not a link (the overlay holds links only)')
    os.symlink(target, dst)


def rewrite_index(paths, K):
    """Rewrite review_dir candidates/I/<page id> -> candidates/<K>/<page id> in the set's index files (point 1)."""
    n = 0
    for p in paths:
        if not os.path.exists(p):
            continue
        hdr, rows = common.read_csv(p)
        for r in rows:
            rd = r.get('review_dir') or ''
            if rd.startswith('candidates/I/'):
                r['review_dir'] = f'candidates/{K}/' + rd[len('candidates/I/'):]
                n += 1
        with open(p + '.new', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            w.writeheader()
            w.writerows(rows)
        os.replace(p + '.new', p)
    return n


def index_files(U):
    """{(band, tag): path} of every page index of the run; tag = 'I'/'V' (singles) or '<S>_f<t>b' (fork)."""
    out, odd = {}, []
    for p in sorted(glob.glob(os.path.join(U, 'source_catalogs', '*_pages_index*.csv'))):
        m = _INDEX_RE.match(os.path.basename(p))
        if m:
            out[(m['band'], m['tag'])] = p
        else:
            odd.append(os.path.basename(p))
    return out, odd


def keyed_merge(U, dry_run=False, index_only=False):
    """Point 2: review_dir into <U>/source_catalogs/<band>_sources_survivors.csv keyed by (Stokes, source_id) within each
    row's own singles set. Returns dict(matched, without, files, refused_keys). Refuses (nothing written) on: a survivors row
    without Stokes I/V, a repeated (J-name, band, Stokes) row, a key naming two pages of its set after separating by the
    row's own page id, an index file whose name is neither singles nor fork."""
    idx, odd = index_files(U)
    bad = [f'index files with an unrecognised name: {odd[:3]}'] if odd else []
    sk = {}
    for (band, tag), p in sorted(idx.items()):
        if '_f' in tag:
            continue                                  # fork indexes serve the multiples page (point 3)
        K = set_of(tag, band)
        for r in csv.DictReader(open(p, newline='')):
            sid, rd = r.get('source_id', ''), r.get('review_dir', '')
            if not rd:
                continue
            sk.setdefault(('x', sid, K), []).append(rd)
            if sid.startswith('J') and sid.count('_') >= 3:
                base, lst, date, b = sid.rsplit('_', 3)
                for k in (('d', base, b, date), ('b', base, b), ('n', base)):
                    sk.setdefault(k + (K,), []).append(rd)

    def one(k, own=None):
        v = sk.get(k)
        if v and len(set(v)) > 1 and own:
            v = [x for x in v if os.path.basename(x) == own] or v
        if v and len(set(v)) > 1:
            bad.append(f'{k[1:]} names {len(set(v))} pages: {sorted(set(v))[:2]}')
        return v[0] if v else None

    def first(k):
        v = sk.get(k)
        return v[0] if v else None

    seen, out, n, miss = set(), [], 0, 0
    for cur in sorted(glob.glob(os.path.join(U, 'source_catalogs', '*_sources_survivors.csv'))):
        hdr, rows = common.read_csv(cur)
        if not rows:
            continue
        for r in rows:
            sid, band, date, S = r['source_id'], r.get('band', ''), r.get('best_date', ''), r.get('stokes', '')
            k = (sid, band, S)
            if S not in ('I', 'V') or k in seen:
                bad.append(f'survivors row {k} ' + ('repeated' if k in seen else 'has no Stokes I/V'))
                continue
            seen.add(k)
            K = set_of(S, band)
            c = (r.get('best_cand_id') or '').split('_')
            own = f'{sid}_{c[1]}_{date}_{band}' if len(c) >= 5 and c[0][:1] == S and c[2] == date and c[3] == band else None
            rd = one(('x', sid, K)) or one(('d', sid, band, date, K), own) or first(('b', sid, band, K))
            if not rd and not index_only:
                g = sorted(glob.glob(os.path.join(U, 'candidates', K, f'{sid}_*_{band}')))
                g = [x for x in g if date and f'_{date}_' in x] or g
                if g:
                    rd = os.path.relpath(g[0], U)
            rd = rd or first(('n', sid, K))
            if rd:
                r['review_dir'] = rd
                n += 1
            else:
                miss += 1
                if index_only:
                    r['review_dir'] = ''
        if 'review_dir' not in hdr:
            hdr.append('review_dir')
        out.append((cur, hdr, rows))
    if bad:
        raise common.Refusal(f'P10 keyed review_dir merge (M14): {len(bad)} keys not unique or without Stokes (nothing '
                             f'written), e.g. ' + '; '.join(sorted(set(bad))[:3]))
    if not dry_run:
        for cur, hdr, rows in out:
            with open(cur + '.new', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=hdr)
                w.writeheader()
                w.writerows(rows)
            os.replace(cur + '.new', cur)
    return dict(matched=n, without=miss, files=[c for c, _, _ in out])
