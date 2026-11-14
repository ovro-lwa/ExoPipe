"""PER-HOUR 10-minute catalog build, the night's delta: the file work around the builder (Gregg 2026-11-13, answer 3 of
R-GREGG-20261113-SPEEDUP-CHECKIN-TEN-ANSWERS: "the builder runs only on the sources the new night touches, with its statistics
frozen from the last all-nights build, and writes a delta rather than a whole new catalog").

The builder itself runs through lib/catalog/wraps/builder_grouping.py in its mode `delta` (FROZEN STATISTICS there). This module
holds what lib/catalog/tier_sublst._build_delta does before and after that run, as plain functions on directories so that they
are tested without a chain (tests/test_catalog_delta_build.py):

  assemble(build_dir, frozen_from, night_dirs, keys, tag)
      the delta build directory: per key the all-nights selected_<K>.csv with the selected rows of every appended night after
      it; the all-nights v_deep_images.csv (the wrap refuses another image table than the record's); the window-tie table and,
      for the veto build, the leak-veto tables, each the all-nights rows plus the nights'.
  content_test(cat_dir, night_cands, vetoed=())
      the stage's content test on one delta catalog directory.

RULES. The all-nights rows keep their bytes and their order; the appended rows follow SORTED BY cand_id, whatever the order of the
night directories, so a delta does not depend on the order in which nights arrived (ruling (2) of
R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS). A detection named twice (a night that is already in the all-nights build, or a
night directory given twice) REFUSES: a night is appended once. A table whose header differs from the all-nights table's REFUSES.
Nothing here judges a detection: no row is dropped, changed or re-verdicted.
"""
import csv
import glob
import json
import os
import shutil

SELECTED = 'selected_{key}.csv'
IMAGES = 'v_deep_images.csv'
PERTIE = os.path.join('pertie', 'pertie_detections.csv')
LEAKVETO = 'leakveto'
DELTA_RECORD = 'builder_delta.json'            # written by the builder wrap in its mode `delta`
FROZEN_RECORD = 'builder_frozen.json'          # written by the builder wrap in its mode `record` (the all-nights build)


class DeltaRefusal(Exception):
    """A night's delta build that must not run (its message ends with REFUSING)."""


def _lines(path):
    """(header line, data lines, line terminator, cand_id per data line) of a CSV table with one record per line."""
    with open(path, newline='') as f:
        text = f.read()
    term = '\r\n' if '\r\n' in text else '\n'
    lines = [ln[:-1] if ln.endswith('\r') else ln for ln in text.split('\n')]
    while lines and lines[-1] == '':
        lines.pop()
    if not lines:
        raise DeltaRefusal(f'{path} is empty (no header) - REFUSING')
    recs = list(csv.reader(lines))
    if len(recs) != len(lines):
        raise DeltaRefusal(f'{path}: {len(lines)} lines but {len(recs)} records (a field spans lines): the delta build appends '
                           f'whole lines - REFUSING')
    head = recs[0]
    if 'cand_id' not in head:
        raise DeltaRefusal(f'{path} has no cand_id column - REFUSING')
    i = head.index('cand_id')
    short = [n for n, r in enumerate(recs[1:], 2) if len(r) != len(head)]
    if short:
        raise DeltaRefusal(f'{path}: line {short[0]} has another number of fields than the header ({len(short)} such lines) - '
                           f'REFUSING')
    return lines[0], lines[1:], term, [r[i] for r in recs[1:]]


def merge(base, nights, out, what):
    """Write ``out`` = the table ``base`` (bytes of its lines and their order kept) followed by the rows of the tables ``nights``
    sorted by cand_id. Returns dict(base=rows, nights={path: rows}, appended=[cand_id, ...]). Refuses on a header that differs
    or a cand_id named twice."""
    head, rows, term, ids = _lines(base)
    seen = dict.fromkeys(ids, base)
    if len(seen) != len(ids):
        raise DeltaRefusal(f'{what}: {base} names a detection twice - REFUSING')
    add, per = [], {}
    for p in nights:
        h, r, _t, i = _lines(p)
        if h != head:
            raise DeltaRefusal(f'{what}: the header of {p} differs from the all-nights table\'s ({base}): tables of another '
                               f'code version are not appended - REFUSING')
        for c, ln in zip(i, r):
            if c in seen:
                raise DeltaRefusal(f'{what}: detection {c} of {p} is already in {seen[c]}: a night is appended once, and a '
                                   f'night that is in the all-nights build is not appended - REFUSING')
            seen[c] = p
            add.append((c, ln))
        per[p] = len(r)
    add.sort()
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    tmp = out + '.part'
    with open(tmp, 'w', newline='') as f:
        f.write(term.join([head] + rows + [ln for _c, ln in add]) + term)
    os.replace(tmp, out)
    return dict(base=len(rows), nights=per, appended=[c for c, _ln in add])


def _pertie_of(d):
    for p in (os.path.join(d, PERTIE), os.path.join(d, 'pertie_detections.csv')):
        if os.path.exists(p):
            return p
    raise DeltaRefusal(f'no window-tie table (pertie/pertie_detections.csv) in {d} - REFUSING')


def assemble(build_dir, frozen_from, night_dirs, keys, tag):
    """Make the delta build directory ``build_dir`` (removed first when it exists: it is this stage's own side directory) from
    the all-nights catalog directory ``frozen_from`` and the catalog directories ``night_dirs`` of the nights appended since.
    ``keys``: the split keys to assemble (those with a selected file in the all-nights directory are written; a key the
    all-nights directory does not hold REFUSES when a night holds rows of it: there is no frozen record to judge them by).
    ``tag``: 'noveto' or 'veto' (the veto build also gets the merged leak-veto tables).
    Returns dict(selected={key: merge result}, pertie=merge result, leakveto={file: merge result}, vetoed=[cand_id, ...])."""
    frozen_from = os.path.abspath(frozen_from)
    nights = [os.path.abspath(d) for d in night_dirs]
    if len(set(nights)) != len(nights):
        raise DeltaRefusal(f'a night directory is named twice ({nights}) - REFUSING')
    if frozen_from in nights:
        raise DeltaRefusal(f'the all-nights catalog directory {frozen_from} is named as a night directory - REFUSING')
    if os.path.lexists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)
    out = dict(selected={}, leakveto={}, vetoed=[])
    for k in keys:
        name = SELECTED.format(key=k)
        base = os.path.join(frozen_from, name)
        have = [os.path.join(d, name) for d in nights if os.path.exists(os.path.join(d, name))]
        if not os.path.exists(base):
            n = sum(len(_lines(p)[1]) for p in have)
            if n:
                raise DeltaRefusal(f'{frozen_from} holds no {name} but the appended nights hold {n} selected row(s) of that key: '
                                   f'the all-nights build made no catalog (and no frozen record) for it - build all nights '
                                   f'again - REFUSING')
            continue
        out['selected'][k] = merge(base, have, os.path.join(build_dir, name), name)
    img = os.path.join(frozen_from, IMAGES)
    if not os.path.exists(img):
        raise DeltaRefusal(f'{img} is missing: the delta build reads the all-nights image table - REFUSING')
    shutil.copyfile(img, os.path.join(build_dir, IMAGES))
    out['pertie'] = merge(_pertie_of(frozen_from), [_pertie_of(d) for d in nights], os.path.join(build_dir, PERTIE),
                          'window-tie table')
    if tag == 'veto':
        base_lv = os.path.join(frozen_from, LEAKVETO)
        if not glob.glob(os.path.join(base_lv, 'leakveto_*.csv')):
            raise DeltaRefusal(f'{base_lv} holds no leak-veto table: the all-nights veto build has none to stand on - REFUSING')
        names = sorted({os.path.basename(p) for d in [frozen_from] + nights
                        for p in glob.glob(os.path.join(d, LEAKVETO, 'leakveto_*.csv'))})
        for n in names:
            src = [os.path.join(d, LEAKVETO, n) for d in [frozen_from] + nights if os.path.exists(os.path.join(d, LEAKVETO, n))]
            dst = os.path.join(build_dir, LEAKVETO, n)
            # a cell only a night has a table for: the night's first table is the base (nothing frozen to keep in order)
            out['leakveto'][n] = merge(src[0], src[1:], dst, f'leak-veto table {n}')
            with open(dst, newline='') as f:
                out['vetoed'] += [r['cand_id'] for r in csv.DictReader(f) if r.get('veto')]
    return out


def content_test(cat_dir, night_cands, vetoed=()):
    """The stage's content test on one delta catalog directory (source_catalogs_<tag>_<K> of a night):
      - builder_delta.json is there and names the touched identities;
      - every source of the master tables is a source of a touched identity (no untouched source is in a delta);
      - every selected detection of the night (``night_cands``) is a member of exactly ONE source of the delta, or is in a
        removed table of the delta, or (veto build) is vetoed by the merged leak-veto tables (``vetoed``) and then a member
        of none.
    Returns (ok, detail, counts)."""
    night_cands, vetoed = set(night_cands), set(vetoed)
    rec = os.path.join(cat_dir, DELTA_RECORD)
    if not os.path.exists(rec):
        return False, f'no {DELTA_RECORD} in {cat_dir}', {}
    with open(rec) as f:
        dl = json.load(f)
    touched = {(b, s) for t in dl.get('touched', []) for b, s, _n in t.get('sources', [])}
    member_of, sources, untouched = {}, 0, []
    for p in sorted(glob.glob(os.path.join(cat_dir, '*_source_members.csv'))):
        band = os.path.basename(p)[:-len('_source_members.csv')]
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                member_of.setdefault(r['cand_id'], set()).add((band, r['source_id']))
    for p in sorted(glob.glob(os.path.join(cat_dir, '*_sources_master.csv'))):
        band = os.path.basename(p)[:-len('_sources_master.csv')]
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                sources += 1
                if (band, r['source_id']) not in touched:
                    untouched.append(f'{band} {r["source_id"]}')
    removed = set()
    for p in glob.glob(os.path.join(cat_dir, '*_removed.csv')):
        with open(p, newline='') as f:
            removed |= {r.get('cand_id', '') for r in csv.DictReader(f)}
    twice = sorted(c for c in night_cands if len(member_of.get(c, ())) > 1)
    back = sorted(c for c in night_cands & vetoed if c in member_of)
    lost = sorted(c for c in night_cands - vetoed if c not in member_of and c not in removed)
    counts = dict(sources=sources, touched_identities=len(dl.get('touched', [])), new_detections=dl.get('new_detections'),
                  night_selected=len(night_cands), night_members=len([c for c in night_cands if c in member_of]),
                  night_removed=len([c for c in night_cands if c not in member_of and c in removed]),
                  night_vetoed=len(night_cands & vetoed), untouched_sources=len(untouched), in_two_sources=len(twice),
                  vetoed_but_member=len(back), in_neither=len(lost))
    ok = not (untouched or twice or back or lost)
    detail = (f'{sources} source(s) of {counts["touched_identities"]} touched identities; of the night\'s {len(night_cands)} '
              f'selected detection(s) {counts["night_members"]} are members of a delta source, {counts["night_removed"]} are in a '
              f'removed table, {counts["night_vetoed"]} are leak-vetoed')
    if not ok:
        detail += (f'; FAILED: sources of no touched identity {untouched[:5]}, detections in two sources {twice[:5]}, vetoed '
                   f'detections that are members {back[:5]}, detections in neither a source nor a removed table {lost[:5]}')
    return ok, detail, counts
