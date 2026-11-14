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
import collections
import csv
import glob
import hashlib
import io
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


def _merged(base, nights, what):
    """(text, result) of merge, nothing written."""
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
    return (term.join([head] + rows + [ln for _c, ln in add]) + term,
            dict(base=len(rows), nights=per, appended=[c for c, _ln in add]))


def merge(base, nights, out, what):
    """Write ``out`` = the table ``base`` (bytes of its lines and their order kept) followed by the rows of the tables ``nights``
    sorted by cand_id. Returns dict(base=rows, nights={path: rows}, appended=[cand_id, ...]). Refuses on a header that differs
    or a cand_id named twice."""
    text, res = _merged(base, nights, what)
    _put(out, text)
    return res


def merge_keyed(base, nights, out, cols, what):
    """merge for a table without a cand_id (the image table: one row per LST, band, Stokes and window): ``out`` = the lines
    of ``base`` followed by the rows of ``nights`` sorted by their ``cols`` values. A row whose ``cols`` values an earlier
    table holds REFUSES (a window is listed once). Returns dict(base=rows, appended=rows)."""
    def read(p):
        with open(p, newline='') as f:
            text = f.read()
        term = '\r\n' if '\r\n' in text else '\n'
        lines = [ln[:-1] if ln.endswith('\r') else ln for ln in text.split('\n')]
        while lines and lines[-1] == '':
            lines.pop()
        recs = list(csv.reader(lines))
        if not lines or len(recs) != len(lines) or any(c not in recs[0] for c in cols):
            raise DeltaRefusal(f'{what}: {p} is empty, holds a field that spans lines, or lacks one of the columns {cols} - '
                               f'REFUSING')
        ix = [recs[0].index(c) for c in cols]
        return lines[0], lines[1:], term, [tuple(r[i] for i in ix) for r in recs[1:]]
    head, rows, term, keys = read(base)
    seen, add = dict.fromkeys(keys, base), []
    if len(seen) != len(keys):
        raise DeltaRefusal(f'{what}: {base} lists a row twice by {cols} - REFUSING')
    for p in nights:
        h, r, _t, k = read(p)
        if h != head:
            raise DeltaRefusal(f'{what}: the header of {p} differs from the all-nights table\'s ({base}) - REFUSING')
        for key, ln in zip(k, r):
            if key in seen:
                raise DeltaRefusal(f'{what}: the row {key} of {p} is already in {seen[key]} - REFUSING')
            seen[key] = p
            add.append((key, ln))
    add.sort()
    _put(out, term.join([head] + rows + [ln for _k, ln in add]) + term)
    return dict(base=len(rows), appended=len(add))


def _pertie_of(d):
    for p in (os.path.join(d, PERTIE), os.path.join(d, 'pertie_detections.csv')):
        if os.path.exists(p):
            return p
    raise DeltaRefusal(f'no window-tie table (pertie/pertie_detections.csv) in {d} - REFUSING')


def _plan(frozen_from, night_dirs, keys, tag):
    """The delta build directory of assemble as TEXTS, nothing written: dict(selected={key: (text, merge result)},
    images=path, pertie=(text, result), leakveto={file: (text, result)}, vetoed=[cand_id, ...], digest={key: sha1}).
    ``digest[key]`` names everything a delta build of the key reads from the directory: the key's selected rows, the
    window-tie table, the image table and (veto build) the leak-veto tables. Two delta builds with one digest, one frozen
    record and one grouping read the same input."""
    frozen_from = os.path.abspath(frozen_from)
    nights = [os.path.abspath(d) for d in night_dirs]
    if len(set(nights)) != len(nights):
        raise DeltaRefusal(f'a night directory is named twice ({nights}) - REFUSING')
    if frozen_from in nights:
        raise DeltaRefusal(f'the all-nights catalog directory {frozen_from} is named as a night directory - REFUSING')
    P = dict(selected={}, leakveto={}, vetoed=[])
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
        P['selected'][k] = _merged(base, have, name)
    P['images'] = os.path.join(frozen_from, IMAGES)
    if not os.path.exists(P['images']):
        raise DeltaRefusal(f'{P["images"]} is missing: the delta build reads the all-nights image table - REFUSING')
    P['pertie'] = _merged(_pertie_of(frozen_from), [_pertie_of(d) for d in nights], 'window-tie table')
    if tag == 'veto':
        base_lv = os.path.join(frozen_from, LEAKVETO)
        if not glob.glob(os.path.join(base_lv, 'leakveto_*.csv')):
            raise DeltaRefusal(f'{base_lv} holds no leak-veto table: the all-nights veto build has none to stand on - REFUSING')
        names = sorted({os.path.basename(p) for d in [frozen_from] + nights
                        for p in glob.glob(os.path.join(d, LEAKVETO, 'leakveto_*.csv'))})
        for n in names:
            src = [os.path.join(d, LEAKVETO, n) for d in [frozen_from] + nights if os.path.exists(os.path.join(d, LEAKVETO, n))]
            # a cell only a night has a table for: the night's first table is the base (nothing frozen to keep in order)
            P['leakveto'][n] = _merged(src[0], src[1:], f'leak-veto table {n}')
            P['vetoed'] += [r['cand_id'] for r in csv.DictReader(io.StringIO(P['leakveto'][n][0], newline='')) if r.get('veto')]
    sha = lambda t: hashlib.sha1(t.encode()).hexdigest()      # noqa: E731
    common = ([f'tag:{tag}', f'images:{_sha1(P["images"])}', f'pertie:{sha(P["pertie"][0])}']
              + [f'leakveto:{n}:{sha(t)}' for n, (t, _r) in sorted(P['leakveto'].items())])
    P['digest'] = {k: sha('\n'.join([f'key:{k}', f'selected:{sha(t)}'] + common)) for k, (t, _r) in P['selected'].items()}
    return P


def inputs(frozen_from, night_dirs, keys, tag):
    """What a delta build of each key would read, WITHOUT writing a build directory: {key: dict(digest, base=all-nights rows,
    appended=[cand_id of every appended selected row])} (the digest is assemble's)."""
    P = _plan(frozen_from, night_dirs, keys, tag)
    return {k: dict(digest=P['digest'][k], base=r['base'], appended=r['appended']) for k, (_t, r) in P['selected'].items()}


def assemble(build_dir, frozen_from, night_dirs, keys, tag):
    """Make the delta build directory ``build_dir`` (removed first when it exists: it is this stage's own side directory) from
    the all-nights catalog directory ``frozen_from`` and the catalog directories ``night_dirs`` of the nights appended since.
    ``keys``: the split keys to assemble (those with a selected file in the all-nights directory are written; a key the
    all-nights directory does not hold REFUSES when a night holds rows of it: there is no frozen record to judge them by).
    ``tag``: 'noveto' or 'veto' (the veto build also gets the merged leak-veto tables).
    Returns dict(selected={key: merge result}, pertie=merge result, leakveto={file: merge result}, vetoed=[cand_id, ...],
    digest={key: what the key's build reads (see _plan)})."""
    P = _plan(frozen_from, night_dirs, keys, tag)
    if os.path.lexists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)
    for k, (text, _r) in P['selected'].items():
        _put(os.path.join(build_dir, SELECTED.format(key=k)), text)
    shutil.copyfile(P['images'], os.path.join(build_dir, IMAGES))
    _put(os.path.join(build_dir, PERTIE), P['pertie'][0])
    for n, (text, _r) in P['leakveto'].items():
        _put(os.path.join(build_dir, LEAKVETO, n), text)
    return dict(selected={k: r for k, (_t, r) in P['selected'].items()}, pertie=P['pertie'][1],
                leakveto={n: r for n, (_t, r) in P['leakveto'].items()}, vetoed=P['vetoed'], digest=P['digest'])


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


# =====================================================================================================================
# THE CATALOG VIEW of an LST hour (per-hour page append, design 2.3a of PER_HOUR_GAP_DESIGNS_20261111.md, stages 1 and 2)
#
# A delta holds only the sources of the touched identities. Pages, the cumulative product and the email need the hour's WHOLE
# catalog as it stands now: the all-nights tables with every row of a touched identity replaced by the newest delta's rows (a
# delta is cumulative since the all-nights build). view_catalog writes that directory for one key, in the layout of an
# all-nights catalog directory, so every reader of a catalog directory reads a view unchanged. A key without a delta is
# viewed as the all-nights tables themselves, and only under a CERTIFICATE (certificate, below) that the present grouping
# touches none of its detections.
#
# RULES. Which row a delta replaces is decided by DETECTIONS and the present grouping table, never by names: a row is
# replaced when the identity that now holds its detection (a source row's best detection, a member row's or a removed
# detection's own) is a touched identity. A delta row of an identity that is not touched REFUSES. Rows of untouched identities
# keep every value; a survivors table is put back in the builder's order (falling |best S/N|; equal values keep the all-nights
# order, a delta row after them) and its `rank` and `review_dir` are numbered again as the builder numbers them
# (build_source_catalogs_v3.py:1103-1108), in the survivors table and in the master table's `review_dir`. A view of a
# catalog with no touched identity is byte-identical to the all-nights catalog (tested). Nothing here judges a detection.
# =====================================================================================================================
VIEW_RECORD = 'catalog_view.json'              # in a view's catalog directory: what it was made from
PROVENANCE = 'grouping_table.json'             # the builder runner's provenance (wraps/builder_grouping.PROVENANCE)
SURVIVORS, MASTER = '_sources_survivors.csv', '_sources_master.csv'
CARRIED = ('E_params.json', 'README_source_catalogs.txt')    # the all-nights build's, carried: the frozen statistics in force


def _sha1(path):
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def read_grouping(table):
    """({cand_id: row}, {multiples_id: [rows]}, sha1, sha256) of a grouping table. ONE read: digests and rows are the same
    bytes."""
    with open(table, 'rb') as f:
        data = f.read()
    T, mg = {}, {}
    for r in csv.DictReader(io.StringIO(data.decode('utf-8'), newline='')):
        if r['cand_id'] in T:
            raise DeltaRefusal(f'{table}: cand_id {r["cand_id"]} twice - REFUSING')
        T[r['cand_id']] = r
        mg.setdefault(r['multiples_id'], []).append(r)
    return T, mg, hashlib.sha1(data).hexdigest(), hashlib.sha256(data).hexdigest()


def certificate(record, grouping, table_path, lst='', key=''):
    """Whether the build that wrote ``record`` (builder_frozen.json of an all-nights build, or builder_delta.json of a delta
    build: both hold, per input detection, the identity and the content digest of the multiples-level group it was built
    under) is UNTOUCHED by the grouping ``grouping`` (read_grouping's result for ``table_path``): every detection of the
    record is in the table, under the recorded identity, in a group whose digest (its detections with their identity,
    identity name, window, multiplicity and group name: wraps/builder_grouping.group_digest_of) is the recorded one. This is
    the builder's own test of a touched identity (its mode `delta`) made without running it; for an untouched build every
    catalog row is the row a build under this table writes. A table that holds FURTHER detections of the hour is not this
    function's business: a night's detections reach a catalog only through a delta build.
    Returns the certificate, a dict: ok, the record's path and sha1, both tables' digests, the hour and key, the counts, and up
    to 20 differing detections. A caller treats anything but ok True as 'build the delta or refuse', never as a pass."""
    from exopipe.phase3.lib.catalog.wraps import builder_grouping as BG
    T, mg, sha1, sha256 = grouping
    with open(record, 'rb') as f:
        data = f.read()
    rec = json.loads(data.decode())
    rows = rec.get('rows')
    out = dict(version=1, ok=False, lst=lst, key=key, record=os.path.abspath(record),
               record_sha1=hashlib.sha1(data).hexdigest(),
               table_from=dict(rec.get('grouping_table') or {}),
               table_to=dict(path=os.path.abspath(table_path), sha1=sha1, sha256=sha256),
               detections=len(rows or ()), differing=0, examples=[])
    if not isinstance(rows, dict):
        out['why'] = 'the record holds no per-detection grouping (rows): written before the certificate existed - build again'
        return out
    digest, bad = {}, []
    for c in sorted(rows):
        was_id, was_dg = rows[c]
        e = T.get(c)
        if e is None:
            bad.append((c, 'not in the table'))
            continue
        m = e['multiples_id']
        if m not in digest:
            digest[m] = BG.group_digest_of(mg[m])
        if e['identity_id'] != was_id:
            bad.append((c, f'identity {was_id} -> {e["identity_id"]}'))
        elif digest[m] != was_dg:
            bad.append((c, f'multiples-level group of {e["identity_id"]} changed (members, a name or a multiplicity)'))
    out.update(ok=not bad, differing=len(bad), examples=[list(b) for b in bad[:20]])
    if bad:
        out['why'] = f'{len(bad)} of {len(rows)} detection(s) are grouped otherwise now'
    return out


def _read(path):
    """(header, rows, line terminator) of a CSV table. REFUSES a table that does not come back byte-for-byte when its parsed
    rows are written again: a view rewrites tables, and must not change a value it does not mean to change."""
    with open(path, newline='') as f:
        text = f.read()
    term = '\r\n' if '\r\n' in text else '\n'
    recs = list(csv.reader(io.StringIO(text, newline='')))
    if not recs:
        raise DeltaRefusal(f'{path} is empty (no header) - REFUSING')
    if _text(recs[0], recs[1:], term) != text:
        raise DeltaRefusal(f'{path}: its rows do not write back to the same bytes (quoting or line ends this module does not '
                           f'reproduce): not merged into a view - REFUSING')
    if any(len(r) != len(recs[0]) for r in recs[1:]):
        raise DeltaRefusal(f'{path}: a row has another number of fields than the header - REFUSING')
    return recs[0], recs[1:], term


def _text(head, rows, term):
    b = io.StringIO(newline='')
    w = csv.writer(b, lineterminator=term)
    w.writerow(head)
    w.writerows(rows)
    return b.getvalue()


def _put(path, text):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path + '.part', 'w', newline='') as f:
        f.write(text)
    os.replace(path + '.part', path)


def _det_col(head, path):
    for c in ('best_cand_id', 'cand_id'):
        if c in head:
            return head.index(c)
    raise DeltaRefusal(f'{path}: neither a best_cand_id nor a cand_id column: a table this module cannot place by detection - '
                       f'REFUSING')


def _order_snr(head, rows, path):
    """rows in the builder's order: falling |best_snr|, equal values in the order given (a stable sort)."""
    i = head.index('best_snr')
    try:
        return sorted(rows, key=lambda r: -abs(float(r[i])))
    except ValueError:
        raise DeltaRefusal(f'{path}: a best_snr that is not a number - REFUSING')


def view_catalog(out, all_cat, delta_cat, grouping, table_path, cert=None):
    """Write the catalog directory ``out`` of one key: the all-nights catalog directory ``all_cat`` with the rows of the
    touched identities replaced by those of the delta catalog directory ``delta_cat`` (None: the key has no delta; then
    ``cert`` must be an ok certificate of the all-nights build's frozen record against this grouping, and the tables are the
    all-nights tables). ``grouping``: read_grouping(table_path): the table the delta was built with (checked by sha1), or
    a later table under which ``cert``, an ok certificate of the delta's record, says the delta build is untouched.
    Returns the view's record (also written as catalog_view.json; the provenance grouping_table.json names the present
    table and carries the certificate or the delta record's digest)."""
    T, _mg, sha1, sha256 = grouping
    if os.path.lexists(out):
        shutil.rmtree(out)
    touched, replaces, dl, dl_sha = set(), set(), None, ''
    if delta_cat is not None:
        p = os.path.join(delta_cat, DELTA_RECORD)
        if not os.path.exists(p):
            raise DeltaRefusal(f'{delta_cat} holds no {DELTA_RECORD}: not a delta build - REFUSING')
        dl_sha = _sha1(p)
        with open(p) as f:
            dl = json.load(f)
        if dl['grouping_table']['sha1'] != sha1 and not (cert and cert.get('ok') is True and cert['table_to']['sha1'] == sha1
                                                         and cert['record_sha1'] == dl_sha):
            raise DeltaRefusal(f'{delta_cat} was built with the grouping table {dl["grouping_table"]["sha1"][:12]}, the view is '
                               f'asked for under {sha1[:12]} ({table_path}) and no certificate says that table leaves the '
                               f'delta build untouched: build the delta again - REFUSING')
        frz = os.path.join(all_cat, FROZEN_RECORD)
        if not os.path.exists(frz) or _sha1(frz) != dl['frozen_record_sha1']:
            raise DeltaRefusal(f'{delta_cat} does not stand on the all-nights build {all_cat} (its {FROZEN_RECORD} is not the '
                               f'one the delta names) - REFUSING')
        touched = {t['identity_id'] for t in dl['touched']}
        replaces = {(b, n) for t in dl['touched'] for b, n in t['replaces']}
    else:
        if not (cert and cert.get('ok') is True and cert['table_to']['sha1'] == sha1
                and os.path.exists(os.path.join(all_cat, FROZEN_RECORD))
                and cert['record_sha1'] == _sha1(os.path.join(all_cat, FROZEN_RECORD))):
            raise DeltaRefusal(f'{all_cat}: no delta and no certificate that the grouping {sha1[:12]} leaves this build '
                               f'untouched: build its delta - REFUSING')
    names = sorted({os.path.basename(p) for d in (all_cat, delta_cat) if d for p in glob.glob(os.path.join(d, '*.csv'))})
    tabs, counts, dropped = {}, dict(rows_kept=0, rows_delta=0, rows_replaced=0), set()
    for name in names:
        a, d = os.path.join(all_cat, name), (os.path.join(delta_cat, name) if delta_cat else '')
        ha, ra, ta = _read(a) if os.path.exists(a) else (None, [], None)
        hd, rd, td = _read(d) if d and os.path.exists(d) else (None, [], None)
        if ha and hd and ha != hd:
            raise DeltaRefusal(f'{name}: the delta\'s header differs from the all-nights table\'s: tables of another code '
                               f'version are not merged - REFUSING')
        head, term = ha or hd, ta or td
        i = _det_col(head, name)
        keep = []
        for r in ra:
            e = T.get(r[i])
            if e is None:
                raise DeltaRefusal(f'{name}: detection {r[i]} of the all-nights table is not in the grouping table - a stale '
                                   f'grouping - REFUSING')
            if e['identity_id'] in touched:
                counts['rows_replaced'] += 1
                if 'best_cand_id' in head and 'source_id' in head and 'band' in head:
                    dropped.add((r[head.index('band')], r[head.index('source_id')]))
            else:
                keep.append(r)
        for r in rd:
            e = T.get(r[i])
            if e is None or e['identity_id'] not in touched:
                raise DeltaRefusal(f'{name}: the delta row of detection {r[i]} is of no touched identity - REFUSING')
        counts['rows_kept'] += len(keep)
        counts['rows_delta'] += len(rd)
        tabs[name] = [head, keep + rd, term]
    if dropped != replaces:
        raise DeltaRefusal(f'{all_cat}: the source rows the grouping table replaces ({len(dropped)}) are not the ones the delta '
                           f'record names ({len(replaces)}; only here {sorted(dropped - replaces)[:3]}, only in the record '
                           f'{sorted(replaces - dropped)[:3]}) - REFUSING')
    # the builder's order and numbering: survivors by falling |best S/N| with rank and review_dir; the master's review_dir
    review = {}
    for name, (head, rows, term) in tabs.items():
        if not name.endswith(SURVIVORS):
            continue
        rows = tabs[name][1] = _order_snr(head, rows, name)
        ir, iv, ib, isrc = (head.index(c) for c in ('rank', 'review_dir', 'band', 'source_id'))
        seen = set()
        for n, r in enumerate(rows, 1):
            if r[isrc] in seen:
                raise DeltaRefusal(f'{name}: source {r[isrc]} twice after the merge - REFUSING')
            seen.add(r[isrc])
            r[ir] = f'rank{n:04d}'
            r[iv] = f'candidates/{r[ib]}/{r[ir]}_{r[ib]}_{r[isrc]}'
            review[(name[:-len(SURVIVORS)], r[isrc])] = r[iv]
    for name, (head, rows, term) in tabs.items():
        if not name.endswith(MASTER):
            continue
        rows = tabs[name][1] = _order_snr(head, rows, name)
        iv, isrc = head.index('review_dir'), head.index('source_id')
        if len({r[isrc] for r in rows}) != len(rows):
            raise DeltaRefusal(f'{name}: a source twice after the merge - REFUSING')
        for r in rows:
            r[iv] = review.get((name[:-len(MASTER)], r[isrc]), '')
    for name, (head, rows, term) in tabs.items():
        _put(os.path.join(out, name), _text(head, rows, term))
    for name in CARRIED:
        if os.path.exists(os.path.join(all_cat, name)):
            shutil.copyfile(os.path.join(all_cat, name), os.path.join(out, name))
    rec = dict(version=1, all_nights=os.path.abspath(all_cat), delta=os.path.abspath(delta_cat) if delta_cat else None,
               delta_record_sha1=dl_sha, grouping_table=dict(path=os.path.abspath(table_path), sha1=sha1, sha256=sha256),
               # the all-nights build this directory stands on (the pages of a night compare it with an earlier page run's)
               frozen_record_sha1=dl['frozen_record_sha1'] if dl else cert['record_sha1'],
               touched_identities=len(touched), sources_replaced=len(dropped), tables=len(tabs), **counts,
               kind='delta' if delta_cat else 'certificate', certificate=cert)
    prov = {}
    pa = os.path.join(delta_cat or all_cat, PROVENANCE)
    if os.path.exists(pa):
        with open(pa) as f:
            prov = json.load(f)
    # the provenance the cumulative's one-table gate reads: the present table, and how this directory stands on it
    prov.update(table=os.path.abspath(table_path), table_sha1=sha1,
                view=dict(kind=rec['kind'], all_nights=rec['all_nights'], delta_record_sha1=dl_sha, certificate=rec['certificate']))
    for name, obj in ((PROVENANCE, prov), (VIEW_RECORD, rec)):
        _put(os.path.join(out, name), json.dumps(obj, indent=1, sort_keys=True) + '\n')
    return rec


def view_test(view_cat, all_cat, delta_cat, grouping):
    """The content test of one view catalog directory, made from the files again (not from view_catalog's own lists):
      - every row of the view is an all-nights row whose detection is of an untouched identity, or a delta row, each exactly
        once, with every value equal except a survivors table's `rank` and `review_dir` and a master table's `review_dir`;
      - no all-nights row of an untouched identity and no delta row is missing;
      - a survivors table is in falling |best S/N| order, its ranks count 1, 2, ... and its review_dir is the builder's form;
        a master row's review_dir is its survivors row's, or empty;
      - a member detection is in one source of its table only.
    Returns (ok, detail, counts)."""
    T = grouping[0]
    touched = set()
    if delta_cat:
        with open(os.path.join(delta_cat, DELTA_RECORD)) as f:
            touched = {t['identity_id'] for t in json.load(f)['touched']}
    bad, counts = [], collections.Counter()

    def table(p):
        if not p or not os.path.exists(p):
            return None, []
        with open(p, newline='') as f:
            rd = csv.reader(f)
            return next(rd), list(rd)

    names = sorted({os.path.basename(p) for d in (view_cat, all_cat, delta_cat) if d for p in glob.glob(os.path.join(d, '*.csv'))})
    for name in names:
        hv, rv = table(os.path.join(view_cat, name))
        ha, ra = table(os.path.join(all_cat, name))
        hd, rd = table(os.path.join(delta_cat, name) if delta_cat else '')
        if hv is None:
            bad.append(f'{name}: not in the view')
            continue
        i = hv.index('best_cand_id') if 'best_cand_id' in hv else hv.index('cand_id')
        free = {hv.index(c) for c in ('rank', 'review_dir') if c in hv and name.endswith((SURVIVORS, MASTER))}
        strip = lambda r: tuple(v for k, v in enumerate(r) if k not in free)      # noqa: E731
        want = collections.Counter(strip(r) for r in ra if T[r[i]]['identity_id'] not in touched)
        want.update(strip(r) for r in rd)
        got = collections.Counter(strip(r) for r in rv)
        if got != want:
            bad.append(f'{name}: {sum((got - want).values())} row(s) of the view are neither an untouched all-nights row nor '
                       f'a delta row, {sum((want - got).values())} such row(s) are missing')
        counts['rows'] += len(rv)
        counts['rows_from_delta'] += len(rd)
        if name.endswith(SURVIVORS) and rv:
            s, ir, iv, ib, isrc = (hv.index(c) for c in ('best_snr', 'rank', 'review_dir', 'band', 'source_id'))
            v = [abs(float(r[s])) for r in rv]
            if any(a < b for a, b in zip(v, v[1:])):
                bad.append(f'{name}: not in falling |best S/N| order')
            if any(r[ir] != f'rank{n:04d}' or r[iv] != f'candidates/{r[ib]}/rank{n:04d}_{r[ib]}_{r[isrc]}'
                   for n, r in enumerate(rv, 1)):
                bad.append(f'{name}: a rank or review_dir that is not the builder\'s numbering')
            hm, rm = table(os.path.join(view_cat, name[:-len(SURVIVORS)] + MASTER))
            rev = {r[isrc]: r[iv] for r in rv}
            if hm is None or any(r[hm.index('review_dir')] != rev.get(r[hm.index('source_id')], '') for r in rm):
                bad.append(f'{name}: the master table\'s review_dir is not the survivors table\'s')
            counts['survivors'] += len(rv)
        if 'cand_id' in hv and 'source_id' in hv:
            c = collections.Counter(r[hv.index('cand_id')] for r in rv)
            if any(n > 1 for n in c.values()):
                bad.append(f'{name}: a member detection in two rows')
            counts['members'] += len(rv)
    ok = not bad
    detail = (f'{counts["rows"]} rows in {len(names)} tables ({counts["rows_from_delta"]} from the delta of {len(touched)} touched '
              f'identities), {counts["survivors"]} survivors, {counts["members"]} member rows' + ('' if ok else '; FAILED: ' + '; '.join(bad[:6])))
    return ok, detail, dict(counts, tables=len(names), touched_identities=len(touched), failures=len(bad))


# =====================================================================================================================
# THE STATE OF AN LST HOUR under a grouping table, its VIEW ROOT and the pass's INDEX (design 2.3a, stages 1 and 2)
#
# After a night's incremental grouping run every LST hour of the grouping is in one of these states, per build tag and key:
#   all-nights  no night was appended for the key and the all-nights build is certified untouched: its catalog stands;
#   delta       a delta build stands: it read exactly the present input (INPUTS_RECORD digest), stands on the present
#               all-nights build (frozen record sha1) and is certified untouched by the present table;
#   build       neither: a delta build is owed (tier_sublst._build_delta runs it, for the pass's own hour and for others);
#   no-record   the all-nights build wrote no frozen record: nothing can be certified or built - build all nights again;
#   empty       the key has no selected row at all: no catalog.
# The decision is made from CONTENT at every pass (key_state), never from a pointer: an earlier pass's index only names
# the directories worth checking. A VIEW ROOT is a directory in the layout of an all-nights catalog directory for one hour
# (view_root): the catalogs of every key as view_catalog writes them, the selected rows, the window-tie table, the
# detection table and the image table of all nights, and the localizations (the all-nights rows of every name no touched
# identity carries, the rest computed again on the view: loc_side, loc_merge). An hour whose every key is `all-nights` has
# no view root: its all-nights directory is its view. The INDEX (hour_views.json in the night's directory) lists every
# hour's root, state digest and certificates under the pass's grouping tables.
# NOT in a view root: the rejected-source record, detection_ts.csv, the table filter file, the conformance records.
# =====================================================================================================================
INPUTS_RECORD = 'delta_inputs.json'            # beside a delta catalog once its content test passed: what the build read
ROOT_RECORD = 'hour_view.json'                 # in a view root: what it was made from; complete only when `complete` is true
INDEX = 'hour_views.json'                      # in a night pass's directory
CAT = 'source_catalogs_{tag}_{key}'
DETECTIONS = 'master_catalog.csv'
IMAGE_KEY = ('lst', 'scope', 'stokes', 'date')
LOC_DIR = 'localizations_{tag}'
LOC_TABLES = ('localizations.csv', 'localization_offsets.csv')
MEMBERS = '_source_members.csv'
STANDS = ('all-nights', 'delta')
SELECTED_ALL = 'selected_all.csv'                 # the selection before its split by key: the cumulative product reads it per hour


def _json(path):
    with open(path) as f:
        return json.load(f)


def _dump(path, obj):
    _put(path, json.dumps(obj, indent=1, sort_keys=True) + '\n')


def write_inputs(cat_dir, lst, key, tag, frozen_from, night_dirs, digest):
    """Record beside a delta catalog what its build read (after its content test passed)."""
    _dump(os.path.join(cat_dir, INPUTS_RECORD),
          dict(version=1, lst=lst, key=key, tag=tag, all_nights=os.path.abspath(frozen_from),
               nights=sorted(os.path.abspath(d) for d in night_dirs), digest=digest))


def key_state(lst, key, tag, frozen_from, inp, candidates, grouping, table_path):
    """The state of one key of one LST hour under the grouping ``grouping`` (read_grouping(table_path)); see the section
    text. ``inp``: inputs(...)[key]; ``candidates``: delta catalog directories to check, best first. Returns a dict: kind
    (all-nights | delta | build | no-record | empty), lst, key, tag, all_nights (the key's all-nights catalog directory),
    appended, digest; for a standing kind also dir (delta), record, record_sha1 and certificate; else why."""
    cat = os.path.join(os.path.abspath(frozen_from), CAT.format(tag=tag, key=key))
    st = dict(lst=lst, key=key, tag=tag, all_nights=cat, appended=len(inp['appended']), digest=inp['digest'])
    frz = os.path.join(cat, FROZEN_RECORD)
    if not os.path.exists(frz):
        if not inp['base'] and not inp['appended'] and not os.path.isdir(cat):
            return dict(st, kind='empty')
        return dict(st, kind='no-record', why=f'the all-nights build wrote no {FROZEN_RECORD} in {cat} (it predates the '
                                               f'record or made no catalog of this key): build all nights of {lst} again')
    fsha = _sha1(frz)
    if inp['appended']:
        why = f'{len(inp["appended"])} appended selected row(s)'
    else:
        c = certificate(frz, grouping, table_path, lst, key)
        if c['ok']:
            return dict(st, kind='all-nights', record=frz, record_sha1=fsha, certificate=c)
        why = f'no appended row, and the all-nights build is not untouched: {c.get("why")}'
    for d in candidates:
        rec, ir = os.path.join(d, DELTA_RECORD), os.path.join(d, INPUTS_RECORD)
        if not (os.path.exists(rec) and os.path.exists(ir)):
            continue
        i = _json(ir)
        if (i.get('digest'), i.get('lst'), i.get('key'), i.get('tag')) != (inp['digest'], lst, key, tag):
            continue
        if _json(rec).get('frozen_record_sha1') != fsha:
            continue
        c = certificate(rec, grouping, table_path, lst, key)
        if c['ok']:
            return dict(st, kind='delta', dir=os.path.abspath(d), record=rec, record_sha1=c['record_sha1'], certificate=c)
    return dict(st, kind='build', record=frz, record_sha1=fsha,
                why=why + '; no delta build of this input stands under the grouping table')


def state_digest(states):
    """One digest of an hour's standing states ({(tag, key): key_state}): equal digests = the same view, file for file."""
    lines = sorted(f'{t}|{k}|{st["kind"]}|{st.get("record_sha1", "")}|{st["digest"]}' for (t, k), st in states.items())
    return hashlib.sha1('\n'.join(lines).encode()).hexdigest()


def _selected_all(root, frozen_from, nights, asm, keys):
    """The view root's selected_all.csv = the all-nights table followed by the appended nights' rows (merge), when the
    all-nights directory holds one; None when no directory holds one. REFUSES when only some of the directories hold it, and
    when its detections are not those of the root's keyed selected tables (every selected row is in exactly one key's table:
    an equality when every key of ``keys`` has a table in the root, else the keyed rows must all be in it and the others are
    counted as outside_keys)."""
    base = os.path.join(frozen_from, SELECTED_ALL)
    have = [d for d in nights if os.path.exists(os.path.join(d, SELECTED_ALL))]
    if not os.path.exists(base):
        if have:
            raise DeltaRefusal(f'{frozen_from} holds no {SELECTED_ALL} but the night directory {have[0]} does: the all-nights '
                               f'directory is not a whole catalog directory - REFUSING')
        return None
    lack = [d for d in nights if d not in have]
    if lack:
        raise DeltaRefusal(f'{lack[0]} holds no {SELECTED_ALL} (the all-nights directory {frozen_from} holds one): the '
                           f'night\'s selection is not whole - REFUSING')
    out = os.path.join(root, SELECTED_ALL)
    m = merge(base, [os.path.join(d, SELECTED_ALL) for d in have], out, 'selection table')
    ids = set(_lines(out)[3])
    keyed = [c for k in asm['selected'] for c in _lines(os.path.join(root, SELECTED.format(key=k)))[3]]
    kset = set(keyed)
    lost, extra = sorted(kset - ids), sorted(ids - kset)
    whole = all(k in asm['selected'] for k in keys)
    if len(kset) != len(keyed) or lost or (whole and extra):
        raise DeltaRefusal(f'{out}: its {len(ids)} detection(s) are not those of the keyed selected tables ({len(keyed)} row(s), '
                           f'{len(keyed) - len(kset)} in two tables; {len(lost)} keyed row(s) not in it, e.g. {lost[:3]}; '
                           f'{len(extra)} of its rows in no keyed table, e.g. {extra[:3]}) - REFUSING')
    return dict(base=m['base'], appended=len(m['appended']), outside_keys=len(extra))


def view_root(root, lst, frozen_from, night_dirs, states, groupings, tags, keys):
    """Write the VIEW ROOT ``root`` of one LST hour (removed first when it exists) from its all-nights catalog directory, the
    catalog directories of the nights appended since, and the standing ``states`` ({(tag, key): key_state}; any other kind
    REFUSES). ``groupings``: {Stokes: (read_grouping result, table path)}. Every catalog must pass view_test or the root
    REFUSES. Returns the root's record (also written as hour_view.json, with complete false: finish_root sets it once the
    localizations are in)."""
    bad = sorted(f'{t}_{k} {st["kind"]}' for (t, k), st in states.items() if st['kind'] not in STANDS)
    if bad:
        raise DeltaRefusal(f'{lst}: no view while a build is owed or cannot be made ({bad}) - REFUSING')
    frozen_from = os.path.abspath(frozen_from)
    nights = [os.path.abspath(d) for d in night_dirs]
    asm = assemble(root, frozen_from, nights, keys, 'veto' if 'veto' in tags else 'noveto')
    img = merge_keyed(os.path.join(frozen_from, IMAGES), [os.path.join(d, IMAGES) for d in nights if os.path.exists(os.path.join(d, IMAGES))],
                      os.path.join(root, IMAGES), IMAGE_KEY, 'image table')
    det = None
    if os.path.exists(os.path.join(frozen_from, DETECTIONS)):
        m = merge(os.path.join(frozen_from, DETECTIONS),
                  [os.path.join(d, DETECTIONS) for d in nights if os.path.exists(os.path.join(d, DETECTIONS))],
                  os.path.join(root, DETECTIONS), 'detection table')
        det = dict(base=m['base'], appended=len(m['appended']))
    sel_all = _selected_all(root, frozen_from, nights, asm, keys)
    cats = {}
    for (tag, key), st in sorted(states.items()):
        g, table = groupings[key[0]]
        out = os.path.join(root, CAT.format(tag=tag, key=key))
        d = st['dir'] if st['kind'] == 'delta' else None
        rec = view_catalog(out, st['all_nights'], d, g, table, cert=st['certificate'])
        ok, detail, counts = view_test(out, st['all_nights'], d, g)
        if not ok:
            raise DeltaRefusal(f'{out}: the view fails its content test: {detail} - REFUSING')
        cats[f'{tag}_{key}'] = dict(kind=rec['kind'], delta=d, record_sha1=st['record_sha1'], digest=st['digest'],
                                    detail=detail, **counts)
    rec = dict(version=1, complete=False, lst=lst, all_nights=frozen_from, nights=nights, tags=list(tags),
               state_digest=state_digest(states), inputs_digest=view_inputs_digest(frozen_from, nights), catalogs=cats, images=img, detections=det,
               selected={k: dict(base=r['base'], appended=len(r['appended'])) for k, r in asm['selected'].items()},
               selected_all=sel_all, window_tie=dict(base=asm['pertie']['base'], appended=len(asm['pertie']['appended'])), localizations={})
    _dump(os.path.join(root, ROOT_RECORD), rec)
    return rec


def loc_side(root, tag, cats):
    """The side directory for the localizations a view root owes for one build tag: <root>/_loc/<tag>, holding, for every
    catalog directory of the root under the glob ``cats``, the survivors rows of the NAMES to localize again and their
    bands' member tables. The names: the present name of every touched identity and the name of every source a delta
    replaced, over all the keys (a name's position is combined over every key and both Stokes, so it is computed on the
    view, never on one delta). Returns (side directory, sorted names, survivors rows written)."""
    dirs = sorted(glob.glob(os.path.join(root, cats)))
    names = set()
    for d in dirs:
        vr = _json(os.path.join(d, VIEW_RECORD))
        if vr['kind'] == 'delta':
            for t in _json(os.path.join(vr['delta'], DELTA_RECORD))['touched']:
                names.add(t['identity_name'])
                names.update(n for _b, n in t['replaces'])
    side = os.path.join(root, '_loc', tag)
    if os.path.lexists(side):
        shutil.rmtree(side)
    os.makedirs(side)
    n = 0
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, '*' + SURVIVORS))):
            head, rows, term = _read(p)
            i = head.index('identity_id') if 'identity_id' in head else head.index('source_id')
            keep = [r for r in rows if r[i] in names]
            if not keep:
                continue
            o = os.path.join(side, os.path.basename(d))
            _put(os.path.join(o, os.path.basename(p)), _text(head, keep, term))
            mp = p[:-len(SURVIVORS)] + MEMBERS
            if os.path.exists(mp):
                shutil.copyfile(mp, os.path.join(o, os.path.basename(mp)))
            n += len(keep)
    return side, sorted(names), n


def loc_merge(root, tag, names, all_loc, new_loc, cats):
    """Write <root>/localizations_<tag>: the all-nights tables (directory ``all_loc``) without the rows of ``names``, plus
    the rows computed again (directory ``new_loc``), in the localization script's order (by identity; a stable sort, so an
    identity's detection offsets keep their order). REFUSES a new row of a name that was not asked for, a header that
    differs, and a row left for a name that is no survivor of the view. Returns {table: counts}."""
    names = set(names)
    alive = set()
    for p in glob.glob(os.path.join(root, cats, '*' + SURVIVORS)):
        head, rows, _t = _read(p)
        i = head.index('identity_id') if 'identity_id' in head else head.index('source_id')
        alive.update(r[i] for r in rows)
    out, res = os.path.join(root, LOC_DIR.format(tag=tag)), {}
    for name in LOC_TABLES:
        a, n = os.path.join(all_loc, name), os.path.join(new_loc, name)
        ha, ra, ta = _read(a) if os.path.exists(a) else (None, [], None)
        hn, rn, tn = _read(n) if os.path.exists(n) else (None, [], None)
        if ha is None and hn is None:
            continue
        if ha and hn and ha != hn:
            raise DeltaRefusal(f'{name} ({tag}): the header computed now differs from the all-nights table\'s ({a}): tables of '
                               f'another code version are not merged - REFUSING')
        head, term = ha or hn, ta or tn
        i = head.index('identity_id')
        stray = sorted({r[i] for r in rn if r[i] not in names})
        if stray:
            raise DeltaRefusal(f'{name} ({tag}): rows of {stray[:3]} were computed but not asked for - REFUSING')
        keep = [r for r in ra if r[i] not in names]
        rows = sorted(keep + rn, key=lambda r: r[i])
        dead = sorted({r[i] for r in rows} - alive)
        if dead:
            raise DeltaRefusal(f'{name} ({tag}): rows of {dead[:3]} ({len(dead)} names), which are no survivors of the view - '
                               f'REFUSING')
        _put(os.path.join(out, name), _text(head, rows, term))
        res[name] = dict(kept=len(keep), new=len(rn), dropped=len(ra) - len(keep), identities=len({r[i] for r in rows}))
    return res


def finish_root(root, localizations):
    """Mark a view root complete: its record gains the localization counts ({tag: loc_merge result})."""
    p = os.path.join(root, ROOT_RECORD)
    rec = _json(p)
    rec.update(complete=True, localizations=localizations)
    _dump(p, rec)
    return rec


def view_inputs_digest(frozen_from, night_dirs):
    """One digest of the tables a view root merges besides the keyed ones (whose content the states' digests name): the image
    table, the detection table and the whole selection of the all-nights directory and of every night directory."""
    lines = []
    for d in [os.path.abspath(frozen_from)] + [os.path.abspath(x) for x in night_dirs]:
        for name in (IMAGES, DETECTIONS, SELECTED_ALL):
            p = os.path.join(d, name)
            lines.append(f'{d}|{name}|{_sha1(p) if os.path.exists(p) else "-"}')
    return hashlib.sha1('\n'.join(lines).encode()).hexdigest()


def root_carries(root, digest, inputs_digest, states, groupings):
    """(True, '') when an earlier pass's view root ``root`` may be carried: root_stands for ``digest``, made from the same
    image, detection and selection tables (view_inputs_digest), and every catalog of ``states`` ({(tag, key): key_state})
    passes view_test again from its files under the present ``groupings`` ({Stokes: read_grouping result}); else (False,
    why). Its completion flag alone never carries it."""
    if not root_stands(root, digest):
        return False, 'not a complete root of these states'
    if _json(os.path.join(root, ROOT_RECORD)).get('inputs_digest') != inputs_digest:
        return False, 'made from other image, detection or selection tables'
    for (tag, key), st in sorted(states.items()):
        ok, detail, _c = view_test(os.path.join(root, CAT.format(tag=tag, key=key)), st['all_nights'],
                                   st['dir'] if st['kind'] == 'delta' else None, groupings[key[0]])
        if not ok:
            return False, f'{tag}_{key}: {detail}'
    return True, ''


def root_stands(root, digest):
    """True when ``root`` is a complete view root of exactly the standing states ``digest`` names (state_digest)."""
    p = os.path.join(root, ROOT_RECORD)
    if not os.path.exists(p):
        return False
    rec = _json(p)
    return rec.get('complete') is True and rec.get('state_digest') == digest


def read_indexes(night_dirs):
    """[(night directory, its index)] of the night directories that hold one, newest written first."""
    got = []
    for d in dict.fromkeys(os.path.abspath(x) for x in night_dirs):
        p = os.path.join(d, INDEX)
        if os.path.exists(p):
            got.append((d, _json(p)))
    return sorted(got, key=lambda x: str(x[1].get('written', '')), reverse=True)


def index_candidates(indexes, lst, tag, key):
    """Delta catalog directories earlier passes listed as standing for (hour, tag, key), newest index first, each once."""
    out = {}
    for _d, ix in indexes:
        st = ((ix.get('hours') or {}).get(lst) or {}).get('keys', {}).get(f'{tag}_{key}') or {}
        if st.get('kind') == 'delta' and st.get('dir'):
            out.setdefault(st['dir'], None)
    return list(out)


def index_roots(indexes, lst):
    """View roots earlier passes listed for the hour, newest index first, each once."""
    out = {}
    for _d, ix in indexes:
        e = (ix.get('hours') or {}).get(lst) or {}
        if e.get('root') and e.get('kind') == 'view':
            out.setdefault(e['root'], None)
    return list(out)
