"""Content tests of the files `exopipe3 retention` writes (lib/retention/). Each reads the file's rows; context keys are optional,
so `exopipe3 verify <class> <file>` runs them alone.

  retention_plan       plan_rows           every row's status is one of plan.STATUSES; a row in the manifest is expired; an
                                           expired or retained row names its production records and expiry; a held row its
                                           holds; a protected row its do-not-delete cells
  retention_manifest   manifest_consistent the sidecar's sha256 is the CSV's; per item, its files' bytes sum to the item's bytes,
                                           its files and links number n_files, every path is inside the item directory and the
                                           identity sha256 over its paths is the item's; the totals are the sidecar's (the
                                           delete's preflight compares the rows with the tree); no item lies in a do-not-delete
                                           cell of data_decisions.csv as read now (lib/retention/protect.py; any subband)
  retention_records    records_parse       every production record parses (times, subbands, end after start)
  retention_holds      holds_parse         every hold and release row parses; every release names an earlier hold
  retention_applied    paths_gone          every removed path of the applied manifest is gone from the tree; with ctx item and
                                           n_paths, that item has exactly n_paths rows
"""
import csv
import os

from exopipe.phase3.core.validate import content_test


@content_test('retention_plan')
def plan_rows(path, **ctx):
    from exopipe.phase3.lib.retention import plan as P
    bad, n = [], 0
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        if tuple(rd.fieldnames or ()) != P.PLAN_COLUMNS:
            return False, 0, 'columns %s, not %s' % (rd.fieldnames, list(P.PLAN_COLUMNS))
        for r in rd:
            n += 1
            st = r['status']
            if st not in P.STATUSES:
                bad.append('%s: status %r' % (r['item'] or r['path'], st))
            elif r['in_manifest'] and st != 'expired':
                bad.append('%s: in the manifest but %s' % (r['item'], st))
            elif st in ('expired', 'retained') and not (r['record_ids'] and r['expiry_utc']):
                bad.append('%s: %s without a production record' % (r['item'], st))
            elif st == 'held' and not r['hold_ids']:
                bad.append('%s: held without a hold' % r['item'])
            elif st == 'protected' and not r['note'].startswith('do not delete: '):
                bad.append('%s: protected without its do-not-delete cells' % r['item'])
    if bad:
        return False, len(bad), '%d of %d rows fail: %s' % (len(bad), n, '; '.join(bad[:4]))
    return True, n, '%d rows' % n


@content_test('retention_manifest')
def manifest_consistent(path, **ctx):
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.retention import plan as P
    try:
        side, items = P.read_manifest(path)
    except Refusal as e:
        return False, None, str(e)
    bad, nbytes = [], 0
    for key, e in sorted(items.items()):
        top = e['row']['path']
        paths = [dict(rel=os.path.relpath(r['path'], top), kind=r['kind'], bytes=int(r['bytes']), mtime_ns=int(r['mtime_ns']),
                      inode=int(r['inode'])) for r in e['paths']]
        b = sum(r['bytes'] for r in paths)
        nf = sum(r['kind'] != 'dir' for r in paths)
        nbytes += b
        if str(b) != e['row']['bytes'] or str(nf) != e['row']['n_files']:
            bad.append('%s: %d files of %d bytes, the item row says %s files of %s bytes' % (
                key, nf, b, e['row']['n_files'], e['row']['bytes']))
        elif any(not r['path'].startswith(top.rstrip('/') + '/') for r in e['paths']):
            bad.append('%s: a path outside the item directory' % key)
        elif P.identity(paths) != e['row']['identity_sha256']:
            bad.append('%s: identity sha256 differs from its paths' % key)
        elif side.get('checksum') == 'content' and any(len(r['content_sha256']) != 64 for r in e['paths']
                                                         if r['kind'] == 'file'):
            bad.append('%s: a content manifest without a sha256 per file' % key)
    if nbytes != side.get('bytes'):
        bad.append('the items hold %d bytes, the sidecar says %s' % (nbytes, side.get('bytes')))
    try:
        P.refuse_do_not_delete(items)
    except Refusal as e:
        bad.append(str(e))
    if bad:
        return False, len(bad), '; '.join(bad[:4])
    return True, side.get('n_files'), '%d items, %d files, %.3f TB' % (len(items), side.get('n_files'), nbytes / 1e12)


@content_test('retention_records')
def records_parse(path, **ctx):
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.retention import table as T
    try:
        recs = T.read_records(os.path.dirname(path)) if os.path.basename(path) == T.cfg().TABLE_NAME else None
    except Refusal as e:
        return False, None, str(e)
    if recs is None:
        return False, None, 'not a table named retention.TABLE_NAME (%s)' % T.cfg().TABLE_NAME
    return bool(recs), len(recs), '%d production records' % len(recs)


@content_test('retention_holds')
def holds_parse(path, **ctx):
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.retention import table as T
    if os.path.basename(path) != T.cfg().HOLDS_NAME:
        return False, None, 'not a file named retention.HOLDS_NAME (%s)' % T.cfg().HOLDS_NAME
    try:
        active = T.read_holds(os.path.dirname(path))
    except Refusal as e:
        return False, None, str(e)
    with open(path, newline='') as f:
        n = sum(1 for _ in csv.DictReader(f))
    return n > 0, len(active), '%d rows, %d active holds' % (n, len(active))


@content_test('retention_applied')
def paths_gone(path, item=None, n_paths=None, **ctx):
    from exopipe.phase3.lib.retention import plan as P
    left, n, mine = [], 0, 0
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        if tuple(rd.fieldnames or ()) != P.APPLIED_COLUMNS:
            return False, 0, 'columns %s, not %s' % (rd.fieldnames, list(P.APPLIED_COLUMNS))
        for r in rd:
            n += 1
            mine += r['item'] == item
            if os.path.lexists(r['path']):
                left.append(r['path'])
    if left:
        return False, len(left), '%d of %d removed paths still exist: %s' % (len(left), n, ', '.join(left[:3]))
    if item is not None and n_paths is not None and mine != int(n_paths):
        return False, mine, '%s: %d rows, %s paths removed' % (item, mine, n_paths)
    return n > 0, n, '%d paths gone' % n
