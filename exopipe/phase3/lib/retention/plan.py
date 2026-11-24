"""The retention plan (dry run), its manifest, and the delete that re-checks it (config section retention).

ITEMS. The raw tree retention.RAW_ROOT is <subband>/<UT date>/<UT hour>/<data entries>. One item = one (subband, UT date, UT
hour) directory. list_tree() lists the subband, date and hour directories by NAME (os.scandir, one level each; never a walk of
the tree); a name that does not fit the layout is reported, never an item.
DATA ENTRIES of an item (list_item, names and entry types only): a regular file named <YYYYMMDD_HHMMSS>_... (a .ms.tar set, a
flagmeta.bin), or a directory named so whose name matches retention.MS_DIR_RE (an MS: older nights store each 10-s set as an
MS directory). The entry's start time is the one in its name.
STATUS of an item (classify), from the production records and the active holds (lib/retention/table.py), first that applies:
  held          an active hold of its subband overlaps its UT hour: never deleted, whatever its expiry (not listed);
  unprocessed   no production record of its subband overlaps its UT hour (not listed): it never expires;
  unrecognized  its directory holds an entry that is not a data entry (another directory, a link, another name): never
                deleted; the plan names the entries;
  empty         no data entry;
  partial       some of its entries start at a time no production record covers: never deleted;
  retained      every entry is covered and the expiry - the latest covering production + RETENTION_DAYS - is still ahead;
  expired       every entry is covered and the expiry has passed.
Only an item with a production record and no hold is listed (one os.scandir of its directory, names only).
THE DRY RUN (make_plan, write_plan) writes <out>/plans/RETENTION_TABLE_<UTC>.csv (every item: status, entries, production,
expiry, records, holds) and, when items are expired, <out>/manifests/DRYRUN_RETENTION_<UTC>.csv with its sidecar .json: the
oldest-expiring retention.MAX_ITEMS_PER_MANIFEST expired items and every path the delete would remove - each file (kind file),
each link inside an MS (link) and each directory of an MS, the MS itself included (dir) - with its size (bytes; 0 for a
directory or link), modification time (ns) and inode from os.lstat (an MS is listed directory by directory: os.scandir and
os.lstat inside that MS only) and, with checksum 'content', the sha256 of each file's bytes; per item the file count, total
bytes and an identity sha256 over its paths (relative to the item), kinds, sizes, mtimes and inodes. The sidecar holds the
manifest's sha256, the raw root, the retention days, the checksum kind, the totals, and the records the plan read: the real
path of its --out and the size and sha256 of the retention table and of the holds file as read (records_digests).
A plan with --dry-run lists and classifies only: no lstat, no checksum, nothing written (make_plan stat=False).
THE DELETE (preflight, then delete_item per item) refuses before it deletes anything unless the manifest was planned from the
records the delete reads (the same --out; RT-MANIFEST), those records only GREW since (the table's and the holds file's first
bytes still hash to the planned digests: append-only, RT-RECORDS; rows appended since are judged by re-classifying), and
EVERY item of the manifest still holds exactly the paths listed with the same kind, size, mtime and inode (and content sha256
for a content manifest), is still expired by the records now and is not held now (RT-MATCH, RT-NOT-EXPIRED, RT-HELD); then,
item by item, it unlinks the listed
files and links (each re-checked by lstat right before), removes the listed directories deepest first with os.rmdir (an empty
directory only: never shutil.rmtree), writes every removed path to <out>/manifests/APPLIED_RETENTION_<UTC>.csv (fsynced per
item), and removes the emptied hour directory (and its date directory when that is empty).
"""
import csv
import hashlib
import io
import json
import os
import re
import stat
import time

from exopipe.phase3.core.record import Refusal
from exopipe.phase3.lib.retention import table as T

STATUSES = ('expired', 'retained', 'partial', 'empty', 'unrecognized', 'held', 'unprocessed')
PLAN_COLUMNS = ('item', 'subband', 'date', 'hour', 'path', 'status', 'n_entries', 'produced_utc', 'expiry_utc', 'record_ids',
                'hold_ids', 'in_manifest', 'note')
MANIFEST_COLUMNS = ('kind', 'item', 'path', 'bytes', 'mtime_ns', 'inode', 'content_sha256', 'identity_sha256', 'n_files',
                    'produced_utc', 'expiry_utc', 'record_ids')
APPLIED_COLUMNS = ('item', 'kind', 'path', 'bytes', 'mtime_ns', 'inode', 'deleted_utc')
PATH_KINDS = ('file', 'link', 'dir')            # manifest rows of the paths a delete removes (besides the 'item' rows)
MANIFEST_KIND = 'retention_dryrun_manifest'
HOUR_S = 3600


def _c():
    return T.cfg()


def stamp(t=None):
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(time.time() if t is None else t))


def sha256_file(path, block=8 * 2 ** 20):
    """sha256 of a file's bytes, read in blocks with ordinary reads (no memory map)."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(block)
            if not b:
                return h.hexdigest()
            h.update(b)


def identity(rows):
    """sha256 over the sorted lines rel<TAB>kind<TAB>bytes<TAB>mtime_ns<TAB>inode of an item's paths [{rel (the path relative
    to the item directory), kind, bytes, mtime_ns, inode}]."""
    lines = sorted('%s\t%s\t%d\t%d\t%d' % (r['rel'], r['kind'], int(r['bytes']), int(r['mtime_ns']), int(r['inode']))
                   for r in rows)
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# the tree, by name
# ---------------------------------------------------------------------------------------------------------------------

def _dirs(path):
    """(directory names, other names) of one directory, by its entries' types (os.scandir; d_type, no stat on Lustre)."""
    dirs, other = [], []
    with os.scandir(path) as it:
        for e in it:
            (dirs if e.is_dir(follow_symlinks=False) else other).append(e.name)
    return sorted(dirs), sorted(other)


def list_tree(root, nights=None, subbands=None):
    """(items, odd): items [{item, subband, date, hour, path, t0, t1}] of the tree under ``root`` (t0, t1: the UT hour, s);
    odd: paths of names that do not fit the layout. ``nights`` / ``subbands``: sets that restrict the listing."""
    c = _c()
    if not os.path.isdir(root):
        raise Refusal('RT-ROOT: the raw-data root %s is not a directory' % root)
    items, odd = [], []
    sbs, rest = _dirs(root)
    odd += [os.path.join(root, n) for n in rest]
    for sb in sbs:
        if not re.match(c.SUBBAND_RE, sb):
            odd.append(os.path.join(root, sb) + '/')
            continue
        if subbands and sb not in subbands:
            continue
        dates, rest = _dirs(os.path.join(root, sb))
        odd += [os.path.join(root, sb, n) for n in rest]
        for date in dates:
            try:
                if not re.match(c.DATE_RE, date):
                    raise ValueError(date)
                d0 = T.parse_utc(date)       # a name of the pattern that is no date (2026-13-45) is odd, never a crash
            except ValueError:
                odd.append(os.path.join(root, sb, date) + '/')
                continue
            if nights and date not in nights:
                continue
            hours, rest = _dirs(os.path.join(root, sb, date))
            odd += [os.path.join(root, sb, date, n) for n in rest]
            for hh in hours:
                if not re.match(c.HOUR_RE, hh) or int(hh) > 23:
                    odd.append(os.path.join(root, sb, date, hh) + '/')
                    continue
                t0 = d0 + int(hh) * HOUR_S
                items.append(dict(item='%s/%s/%s' % (sb, date, hh), subband=sb, date=date, hour=hh,
                                  path=os.path.join(root, sb, date, hh), t0=t0, t1=t0 + HOUR_S))
    return items, odd


def _stamp_time(m):
    return T.parse_utc('%s-%s-%sT%s:%s:%sZ' % (m.group(1)[:4], m.group(1)[4:6], m.group(1)[6:], m.group(2)[:2],
                                              m.group(2)[2:4], m.group(2)[4:]))


def list_item(path):
    """(entries [(name, start time s, kind 'file' | 'ms')], odd names) of one item directory, by name and entry type
    (os.scandir; no stat): a regular file named <YYYYMMDD_HHMMSS>_..., or a directory named so and matching MS_DIR_RE, is a
    data entry; anything else is odd (a directory gets a trailing '/', a link '@')."""
    c = _c()
    entries, odd = [], []
    with os.scandir(path) as it:
        for e in it:
            m = re.match(c.STAMP_RE, e.name)
            if e.is_symlink():
                odd.append(e.name + '@')
                continue
            isdir = e.is_dir(follow_symlinks=False)
            kind = 'ms' if isdir and re.search(c.MS_DIR_RE, e.name) else 'file' if e.is_file(follow_symlinks=False) else None
            if not m or kind is None:
                odd.append(e.name + ('/' if isdir else ''))
                continue
            try:
                entries.append((e.name, _stamp_time(m), kind))
            except ValueError:
                odd.append(e.name + ('/' if isdir else ''))
    return sorted(entries), sorted(odd)


def _matches(sbset, sb):
    return sbset is None or sb in sbset


def classify(item, records, holds, now, days=None):
    """The item's status and what decides it (module docstring): {status, n_entries, entries, produced, expiry, record_ids,
    hold_ids, note}. Lists the item's directory only when a production record overlaps it and no hold does."""
    days = _c().RETENTION_DAYS if days is None else days
    sb, t0, t1 = item['subband'], item['t0'], item['t1']
    out = dict(status=None, n_entries=None, entries=[], produced=None, expiry=None, record_ids=[], hold_ids=[], note='')
    hh = [h for h in holds if _matches(h['subbands'], sb) and h['start'] < t1 and h['end'] > t0]
    if hh:
        out.update(status='held', hold_ids=[h['hold_id'] for h in hh])
        return out
    recs = [r for r in records if _matches(r['subbands'], sb) and r['start'] < t1 and r['end'] > t0]
    if not recs:
        out['status'] = 'unprocessed'
        return out
    entries, odd = list_item(item['path'])
    out.update(n_entries=len(entries), entries=entries)
    if odd:
        out.update(status='unrecognized', note='%d entries that are not data entries: %s' % (len(odd), ', '.join(odd[:5])))
        return out
    if not entries:
        out['status'] = 'empty'
        return out
    used, uncovered, produced = set(), 0, None
    for _name, t, _kind in entries:
        cov = [r for r in recs if r['start'] <= t < r['end']]
        if not cov:
            uncovered += 1
            continue
        best = max(cov, key=lambda r: r['produced'])
        used.add(best['record_id'])
        produced = best['produced'] if produced is None else max(produced, best['produced'])
    out['record_ids'] = sorted(used)
    if uncovered:
        out.update(status='partial', note='%d of %d entries start at a time no production record covers' % (
            uncovered, len(entries)))
        return out
    out.update(produced=produced, expiry=produced + days * 86400)
    out['status'] = 'expired' if out['expiry'] <= now else 'retained'
    return out


def _row(item_path, p, st, checksum):
    if stat.S_ISREG(st.st_mode):
        kind = 'file'
    elif stat.S_ISDIR(st.st_mode):
        kind = 'dir'
    elif stat.S_ISLNK(st.st_mode):
        kind = 'link'
    else:
        raise Refusal('RT-MATCH: %s is neither a file, a directory nor a link' % p)
    return dict(rel=os.path.relpath(p, item_path), path=p, kind=kind, bytes=st.st_size if kind == 'file' else 0,
                mtime_ns=st.st_mtime_ns, inode=st.st_ino,
                content_sha256=sha256_file(p) if checksum == 'content' and kind == 'file' else '')


def stat_entries(item_path, entries, checksum='identity'):
    """[{rel, path, kind, bytes, mtime_ns, inode, content_sha256}] of every path the delete of these data entries [(name, t,
    kind)] removes: a file entry by one os.lstat (it must still be a regular file, RT-MATCH otherwise); an MS entry by an
    os.lstat of the MS directory and of every path inside it (os.scandir of each of its directories: a walk inside that MS
    only). content_sha256 of files only for checksum 'content'."""
    out = []
    for name, _t, kind in entries:
        p = os.path.join(item_path, name)
        st = os.lstat(p)
        if kind == 'file':
            if not stat.S_ISREG(st.st_mode):
                raise Refusal('RT-MATCH: %s is no longer a regular file' % p)
            out.append(_row(item_path, p, st, checksum))
            continue
        if not stat.S_ISDIR(st.st_mode):
            raise Refusal('RT-MATCH: %s is no longer a directory' % p)
        out.append(_row(item_path, p, st, checksum))
        todo = [p]
        while todo:
            d = todo.pop()
            with os.scandir(d) as it:
                for e in sorted(it, key=lambda x: x.name):
                    r = _row(item_path, e.path, os.lstat(e.path), checksum)
                    out.append(r)
                    if r['kind'] == 'dir':
                        todo.append(e.path)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the dry run
# ---------------------------------------------------------------------------------------------------------------------

class Plan:
    """The classified tree: items (with their status), odd paths, the manifest batch [(item, cls, paths)], counts."""

    def __init__(self):
        self.items, self.odd, self.batch, self.n_expired = [], [], [], 0
        self.root, self.now, self.days, self.checksum, self.records = None, None, None, None, None

    def counts(self):
        by = {}
        for it, cls in self.items:
            by[cls['status']] = by.get(cls['status'], 0) + 1
        return by


def prefix_digest(path, n=None):
    """(bytes, sha256) of the first ``n`` bytes of ``path`` (all of it when n is None); a missing file is (0, sha256 of
    nothing). With n larger than the file: (its size, None) - the file shrank."""
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            data = f.read() if n is None else f.read(n)
    except FileNotFoundError:
        data = b''
    if n is not None and len(data) < n:
        return len(data), None
    h.update(data)
    return len(data), h.hexdigest()


def records_digests(out):
    """{'root': real path of <out>, 'table': [bytes, sha256], 'holds': [bytes, sha256]}: what a plan read its records from."""
    return dict(root=os.path.realpath(out), table=list(prefix_digest(T.table_path(out))),
                holds=list(prefix_digest(T.holds_path(out))))


def make_plan(root, out, now=None, nights=None, subbands=None, checksum=None, max_items=None, stat=True):
    """Classify every item of the tree under ``root`` against <out>'s production records and holds at ``now`` and stat the
    paths of the oldest-expiring ``max_items`` expired items (the manifest batch). With stat=False (a dry run of the plan)
    the batch is chosen but not statted (its paths None): no lstat, no checksum. Writes nothing."""
    c = _c()
    checksum = checksum or c.CHECKSUM
    if checksum not in ('identity', 'content'):
        raise Refusal("RT-CHECKSUM: retention.CHECKSUM / --checksum is %r, not 'identity' or 'content'" % checksum)
    max_items = c.MAX_ITEMS_PER_MANIFEST if max_items is None else int(max_items)
    pl = Plan()
    pl.root, pl.now, pl.days, pl.checksum = os.path.realpath(root), time.time() if now is None else now, c.RETENTION_DAYS, \
        checksum
    pl.records = records_digests(out)        # before the read: a row appended in between is judged at the delete
    records, holds = T.read_records(out), T.read_holds(out)
    items, pl.odd = list_tree(root, nights, subbands)
    for it in items:
        pl.items.append((it, classify(it, records, holds, pl.now, pl.days)))
    expired = sorted([(cls['expiry'], it['date'], it['hour'], it['subband'], it, cls) for it, cls in pl.items
                      if cls['status'] == 'expired'], key=lambda x: x[:4])
    pl.n_expired = len(expired)
    for _e, _d, _h, _s, it, cls in expired[:max_items]:
        pl.batch.append((it, cls, stat_entries(it['path'], cls['entries'], checksum) if stat else None))
    return pl


def _write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in columns})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _ids(xs):
    return ' '.join(xs)


def write_plan(pl, out, when=None):
    """Write the plan table (every item) and, when the batch is not empty, the dry-run manifest and its sidecar. Returns
    (plan csv, manifest csv or None, sidecar json or None)."""
    c = _c()
    st = stamp(when)
    in_batch = {it['item'] for it, _cls, _f in pl.batch}
    rows = []
    for it, cls in sorted(pl.items, key=lambda x: (x[0]['date'], x[0]['hour'], x[0]['subband'])):
        rows.append(dict(item=it['item'], subband=it['subband'], date=it['date'], hour=it['hour'], path=it['path'],
                         status=cls['status'], n_entries='' if cls['n_entries'] is None else cls['n_entries'],
                         produced_utc=T.iso(cls['produced']) if cls['produced'] is not None else '',
                         expiry_utc=T.iso(cls['expiry']) if cls['expiry'] is not None else '',
                         record_ids=_ids(cls['record_ids']), hold_ids=_ids(cls['hold_ids']),
                         in_manifest='yes' if it['item'] in in_batch else '', note=cls['note']))
    for p in pl.odd:
        rows.append(dict(item='', path=p, status='unrecognized', note='a name outside the layout '
                         '<subband>/<YYYY-MM-DD>/<HH>/ (never an item)'))
    plan_csv = os.path.join(out, c.PLAN_DIR, 'RETENTION_TABLE_%s.csv' % st)
    _write_csv(plan_csv, PLAN_COLUMNS, rows)
    if not pl.batch:
        return plan_csv, None, None
    if any(paths is None for _it, _cls, paths in pl.batch):
        raise Refusal('RT-MANIFEST: a plan made without stat (a dry run) has no manifest to write')
    mrows, items, nfiles, ndirs, nbytes = [], [], 0, 0, 0
    for it, cls, paths in pl.batch:
        b = sum(r['bytes'] for r in paths)
        nf = sum(r['kind'] != 'dir' for r in paths)
        ident = identity(paths)
        mrows.append(dict(kind='item', item=it['item'], path=it['path'], bytes=b, identity_sha256=ident, n_files=nf,
                          produced_utc=T.iso(cls['produced']), expiry_utc=T.iso(cls['expiry']),
                          record_ids=_ids(cls['record_ids'])))
        for r in paths:
            mrows.append(dict(kind=r['kind'], item=it['item'], path=r['path'], bytes=r['bytes'], mtime_ns=r['mtime_ns'],
                              inode=r['inode'], content_sha256=r['content_sha256']))
        items.append(dict(item=it['item'], n_files=nf, n_dirs=len(paths) - nf, bytes=b, identity_sha256=ident))
        nfiles += nf
        ndirs += len(paths) - nf
        nbytes += b
    man_csv = os.path.join(out, c.MANIFEST_DIR, 'DRYRUN_RETENTION_%s.csv' % st)
    _write_csv(man_csv, MANIFEST_COLUMNS, mrows)
    side = dict(kind=MANIFEST_KIND, version=1, csv=os.path.basename(man_csv), csv_sha256=sha256_file(man_csv),
                raw_root=pl.root, retention_days=pl.days, checksum=pl.checksum, evaluated_utc=T.iso(pl.now),
                written_utc=T.iso(time.time()), n_items=len(items), n_files=nfiles, n_dirs=ndirs, bytes=nbytes,
                n_expired_in_tree=pl.n_expired, n_expired_not_in_manifest=pl.n_expired - len(items),
                plan_csv=os.path.basename(plan_csv), records=pl.records, items=items)
    man_json = os.path.splitext(man_csv)[0] + '.json'
    tmp = man_json + '.part'
    with open(tmp, 'w') as f:
        json.dump(side, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, man_json)
    return plan_csv, man_csv, man_json


# ---------------------------------------------------------------------------------------------------------------------
# the delete
# ---------------------------------------------------------------------------------------------------------------------

def read_manifest(man_csv):
    """(sidecar dict, {item: {'row': item row, 'paths': [file, link and dir rows]}}) of a dry-run manifest, refused (RT-MANIFEST) unless its
    sidecar is a retention dry-run sidecar whose csv_sha256 is the CSV's sha256 now and whose counts match the rows."""
    man_json = os.path.splitext(man_csv)[0] + '.json'
    try:
        with open(man_json) as f:
            side = json.load(f)
    except (OSError, ValueError) as e:
        raise Refusal('RT-MANIFEST: the sidecar %s of the manifest cannot be read (%s)' % (man_json, e))
    if side.get('kind') != MANIFEST_KIND or side.get('csv') != os.path.basename(man_csv):
        raise Refusal('RT-MANIFEST: %s is not the dry-run sidecar of %s' % (man_json, man_csv))
    try:
        now_sha = sha256_file(man_csv)
    except OSError as e:
        raise Refusal('RT-MANIFEST: %s cannot be read (%s)' % (man_csv, e))
    if now_sha != side.get('csv_sha256'):
        raise Refusal('RT-MANIFEST: %s changed since its dry run (sha256 %s, the sidecar says %s)' % (
            man_csv, now_sha[:12], str(side.get('csv_sha256'))[:12]))
    items = {}
    with open(man_csv, newline='') as f:
        rd = csv.DictReader(f)
        if tuple(rd.fieldnames or ()) != MANIFEST_COLUMNS:
            raise Refusal('RT-MANIFEST: %s has the columns %s, not %s' % (man_csv, rd.fieldnames, list(MANIFEST_COLUMNS)))
        for r in rd:
            e = items.setdefault(r['item'], {'row': None, 'paths': []})
            if r['kind'] == 'item':
                e['row'] = r
            elif r['kind'] in PATH_KINDS:
                e['paths'].append(r)
            else:
                raise Refusal('RT-MANIFEST: %s has a row of kind %r' % (man_csv, r['kind']))
    nf = sum(r['kind'] != 'dir' for e in items.values() for r in e['paths'])
    if any(e['row'] is None for e in items.values()) or len(items) != side.get('n_items') or nf != side.get('n_files'):
        raise Refusal('RT-MANIFEST: %s: %d items and %d files, the sidecar says %s and %s (or a file row without its item '
                      'row)' % (man_csv, len(items), nf, side.get('n_items'), side.get('n_files')))
    return side, items


def preflight(man_csv, root, out, now=None):
    """Every check of the delete, before anything is deleted: (sidecar, items) when the manifest still matches the files and
    every item is still expired and not held; Refusal (RT-MANIFEST, RT-MATCH, RT-NOT-EXPIRED, RT-HELD) otherwise, naming up to
    ten problems. Reads only."""
    side, items = read_manifest(man_csv)
    root = os.path.realpath(root)
    if os.path.realpath(side.get('raw_root') or '') != root:
        raise Refusal('RT-MANIFEST: the manifest is of the tree %s, this delete is of %s (retention.RAW_ROOT)' % (
            side.get('raw_root'), root))
    rec = side.get('records') or {}
    if rec.get('root') != os.path.realpath(out):
        raise Refusal('RT-MANIFEST: the manifest was planned from the records of %s, this delete reads those of %s: plan '
                      'again with --out %s' % (rec.get('root') or 'an unknown --out', os.path.realpath(out), out))
    now = time.time() if now is None else now
    probs, codes = [], []
    for what, path in (('table', T.table_path(out)), ('holds', T.holds_path(out))):
        try:
            n, sha = (rec.get(what) or [None, None])
            got = prefix_digest(path, int(n))[1]
        except (TypeError, ValueError, OSError) as x:
            got, sha = None, 'unreadable (%s)' % x
        if got is None or got != sha:
            # appended rows are fine (judged below by re-classifying every item); a rewritten or shortened file is not
            probs.append('the %s %s is not the one the plan read plus appended rows (its first %s bytes do not hash to the '
                         'planned %s): plan again' % (what, path, n, str(sha)[:12]))
            codes.append('RT-RECORDS')
    records, holds = T.read_records(out), T.read_holds(out)
    for key in sorted(items):
        e = items[key]
        parts = key.split('/')
        path = os.path.join(root, *parts) if len(parts) == 3 else None
        try:
            t0 = T.parse_utc(parts[1]) + int(parts[2]) * HOUR_S if path else None
        except (ValueError, IndexError):
            t0 = None
        if path is None or t0 is None or os.path.realpath(e['row']['path']) != path:
            probs.append('%s: item path %s is not <root>/%s' % (key, e['row']['path'], key))
            codes.append('RT-MANIFEST')
            continue
        sb, date, hh = parts
        it = dict(item=key, subband=sb, date=date, hour=hh, path=path, t0=t0)
        it['t1'] = it['t0'] + HOUR_S
        if not os.path.isdir(path):
            probs.append('%s: the directory is gone' % key)
            codes.append('RT-MATCH')
            continue
        cls = classify(it, records, holds, now)
        if cls['status'] == 'held':
            probs.append('%s: held now (%s) - a held range is never deleted' % (key, ', '.join(cls['hold_ids'])))
            codes.append('RT-HELD')
            continue
        if cls['status'] != 'expired':
            probs.append('%s: %s now, not expired%s' % (key, cls['status'], (' (%s)' % cls['note']) if cls['note'] else ''))
            # retained / unprocessed: the records changed (a re-processing restarts the clock); unrecognized / partial /
            # empty: the directory's entries changed since the dry run
            codes.append('RT-NOT-EXPIRED' if cls['status'] in ('retained', 'unprocessed') else 'RT-MATCH')
            continue
        try:
            now_paths = stat_entries(path, cls['entries'], side.get('checksum') or 'identity')
        except (OSError, Refusal) as x:
            probs.append('%s: %s' % (key, x))
            codes.append('RT-MATCH')
            continue
        want = {os.path.relpath(r['path'], e['row']['path']): r for r in e['paths']}
        here = {r['rel']: r for r in now_paths}
        if set(want) != set(here):
            probs.append('%s: %d paths now, %d in the manifest (%d new, %d gone: %s)' % (
                key, len(here), len(want), len(set(here) - set(want)), len(set(want) - set(here)),
                ', '.join(sorted(set(here) ^ set(want))[:3])))
            codes.append('RT-MATCH')
            continue
        bad = [k for k, r in sorted(here.items())
               if (r['kind'], str(r['bytes']), str(r['mtime_ns']), str(r['inode'])) !=
               (want[k]['kind'], want[k]['bytes'], want[k]['mtime_ns'], want[k]['inode'])
               or (side.get('checksum') == 'content' and r['content_sha256'] != want[k]['content_sha256'])]
        if bad or identity(now_paths) != e['row']['identity_sha256']:
            probs.append('%s: %d paths changed since the dry run (%s)' % (key, len(bad), ', '.join(bad[:3]) or 'identity'))
            codes.append('RT-MATCH')
            continue
        e['now'] = now_paths
        e['it'] = it
    if probs:
        raise Refusal('%s: the manifest does not hold now - NOTHING deleted; %d problem(s): %s%s' % (
            '/'.join(dict.fromkeys(codes)), len(probs), '; '.join(probs[:10]), ' ...' if len(probs) > 10 else ''))
    return side, items


def ruling_problem(row, side, c=None):
    """None when the ledger row (core/ledger.rows) is Gregg's go for THIS manifest, else why not: authorized_by begins with
    the word retention.RULING_AUTHORIZED_BY, status begins with retention.RULING_STATUS_PREFIX (any case), and the row's text
    quotes the manifest's sha256 - in full or its first RULING_SHA_MIN_HEX hex digits or more (its file name alone does not
    count: a manifest regenerated or edited under the same name must not match). Code cannot prove who wrote a row: the guard
    is procedural (no session writes an R-GREGG row without Gregg's words) plus these checks."""
    c = c or _c()
    who = (row.get('authorized_by') or '').split()
    if not who or who[0].rstrip(',;:') != c.RULING_AUTHORIZED_BY:
        return 'authorized_by is %r, not %s' % (row.get('authorized_by'), c.RULING_AUTHORIZED_BY)
    if not (row.get('status') or '').strip().upper().startswith(c.RULING_STATUS_PREFIX.upper()):
        return 'status is %r, not %s...' % (row.get('status'), c.RULING_STATUS_PREFIX)
    sha = str(side.get('csv_sha256') or '').lower()
    text = ' '.join(str(v) for k, v in row.items() if k not in ('authorized_by', 'status')).lower()
    n = int(c.RULING_SHA_MIN_HEX)
    if len(sha) != 64 or not any(sha.startswith(h) for h in re.findall(r'(?<![0-9a-f])[0-9a-f]{%d,64}(?![0-9a-f])' % n, text)):
        return 'its text does not quote this manifest\'s sha256 (%s..., at least %d hex digits)' % (sha[:n], n)
    return None


def delete_item(e, applied_csv):
    """Remove the paths of one preflighted item: unlink its files and links, each re-checked by lstat right before (same kind,
    size, mtime, inode), then os.rmdir its MS directories deepest first (each still a directory of the same inode; an rmdir
    removes an empty directory only); append every removed path to the applied manifest (fsynced); then remove the emptied
    hour directory and its date directory when that is empty. Returns (files, bytes, note). A path that changed since the
    preflight refuses (RT-MATCH): the paths before it are gone and recorded."""
    rows, n, b = [], 0, 0
    leaves = [r for r in e['now'] if r['kind'] != 'dir']
    dirs = sorted((r for r in e['now'] if r['kind'] == 'dir'), key=lambda r: (-r['rel'].count(os.sep), r['rel']))
    try:
        for r in leaves:
            st = os.lstat(r['path'])
            if _row(e['it']['path'], r['path'], st, 'identity')['kind'] != r['kind'] or \
                    (st.st_size if r['kind'] == 'file' else 0, st.st_mtime_ns, st.st_ino) != (r['bytes'], r['mtime_ns'],
                                                                                            r['inode']):
                raise Refusal('RT-MATCH: %s changed after the preflight - not deleted; the delete stops here' % r['path'])
            os.unlink(r['path'])
            rows.append(dict(item=e['it']['item'], kind=r['kind'], path=r['path'], bytes=r['bytes'], mtime_ns=r['mtime_ns'],
                             inode=r['inode'], deleted_utc=T.iso(time.time())))
            n, b = n + (r['kind'] != 'dir'), b + r['bytes']
        for r in dirs:                   # deepest first; an emptied directory's mtime changed, its inode did not
            st = os.lstat(r['path'])
            if not stat.S_ISDIR(st.st_mode) or st.st_ino != r['inode']:
                raise Refusal('RT-MATCH: %s is no longer the directory of the manifest - the delete stops here' % r['path'])
            try:
                os.rmdir(r['path'])
            except OSError as x:
                raise Refusal('RT-MATCH: %s holds a path the manifest does not list (%s) - kept; the delete stops here' % (
                    r['path'], x))
            rows.append(dict(item=e['it']['item'], kind='dir', path=r['path'], bytes=0, mtime_ns=r['mtime_ns'],
                             inode=r['inode'], deleted_utc=T.iso(time.time())))
    finally:
        _append_rows(applied_csv, APPLIED_COLUMNS, rows)      # every removed path, also when a later one refused
    try:
        os.rmdir(e['it']['path'])
    except OSError as x:
        return n, b, 'the hour directory is kept: %s' % x      # an entry appeared after the preflight: never removed here
    try:
        os.rmdir(os.path.dirname(e['it']['path']))
    except OSError:
        pass                                                   # the date directory still holds other hours
    return n, b, ''


def _append_rows(path, columns, rows):
    """Append rows (one write, fsynced) to a CSV that gets its header when it is created (lib/retention/table.append_row)."""
    if not rows:
        return
    T.append_row(path, columns, rows[0])
    if len(rows) > 1:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator='\n')
        for r in rows[1:]:
            w.writerow([r.get(c, '') for c in columns])
        data = buf.getvalue().encode()
        fd = os.open(path, os.O_WRONLY | os.O_APPEND)
        try:
            while data:
                data = data[os.write(fd, data):]
            os.fsync(fd)
        finally:
            os.close(fd)
