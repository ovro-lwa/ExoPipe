"""Content tests of the product classes `exopipe3 archive` writes (spec archive.md section 6). Each test reads the FILE and
checks its content against the archive as it is now (files present, sizes, checksums, a re-derived plan); context keys are
optional, so `exopipe3 verify <class> <file>` runs them alone.

  archive_manifest         rows_transferred        push / publish manifest: every row's file where the row says it is, with its
                                                   size and checksum (planned rows: the source; transferred rows: the destination)
  quarantine_manifest      rows_quarantined        planned rows: source present with its size (and checksum), destination free,
                                                   same filesystem; moved rows: quarantined copy present with its size and
                                                   checksum, source gone or holding a different file (its replacement)
  archive_sync_record      archive_equals_attempt  the in-place sync plan re-derived now equals the recorded plan; a closing
                                                   record must be all 'keep'
  retirement_audit         no_excluded_product_left  re-scan with core.exclusions: plan = the recorded files, applied = none left;
                                                   exclusion list unchanged (else STALE)
  retirement_audit         no_refused_night_frame_left  (retire-refused) re-plan: plan = recorded rows, applied = none left
  row_quarantine_manifest  rows_moved_with_manifest  MANIFEST.sha256 verifies; each quarantined store copy has its sha256_before;
                                                   the quarantined rows are wideband rows without the mark; with ctx fresh=True
                                                   the live store equals rows_after / sha256_after
  restore_manifest         rows_restored           restored rows: the file is back with the manifest md5 (or a later quarantine
                                                   of the same path explains its absence); planned rows: the copy is intact
  archive_plan             plan_current            a plan of an in-place script (supersede, quarantine-rows): structure, and every
                                                   recorded input unchanged since (else STALE)
  archive_mosaic_copy      class_audit             push_mosaic_v2_20260929.py:79-107 rules on one archived mosaic file
"""
import os

from exopipe.phase3.core.validate import content_test


def _K():
    from exopipe.phase3.lib.archive import common
    return common


def _fmt(problems, n, what):
    if problems:
        return False, len(problems), f'{len(problems)} of {n} {what} fail: ' + '; '.join(problems[:4])
    return True, n, f'{n} {what} pass'


# ---------------------------------------------------------------------------------------------------------------------
# archive_manifest
# ---------------------------------------------------------------------------------------------------------------------

@content_test('archive_manifest')
def rows_transferred(path, since=None, paths=None, checksum=True, **ctx):
    K = _K()
    cols, rows = K.read_csv(path)
    if set(('src', 'dst', 'bytes', 'md5')) <= set(cols):
        dialect = 'push'
    elif set(K.sec().PUBLISH_COLUMNS) <= set(cols):
        dialect = 'publish'
    else:
        return False, None, f'header {cols} is neither (src,dst,bytes,md5) nor archive.PUBLISH_COLUMNS'
    sc, dc = ('src', 'dst') if dialect == 'push' else ('source', 'dest')
    if since and 'written_at' in cols:
        rows = [r for r in rows if (r.get('written_at') or '') >= since]
    if paths is not None:
        rows = [r for r in rows if r[dc] in set(paths)]
    last = {}
    for i, r in enumerate(rows):
        last[r[dc]] = i
    if dialect == 'push':
        dup = len(rows) - len(last)
        if dup:
            return False, dup, f'{dup} duplicate destination(s)'
    rows = [r for i, r in enumerate(rows) if last[r[dc]] == i]       # append-only publish manifests: the latest row holds
    algo = 'sha256' if any(len(r.get('md5') or '') == 64 for r in rows[:1]) else 'md5'
    problems = []
    if dialect == 'push':
        there = [os.path.exists(r[dc]) for r in rows]
        if any(there) and not all(there):
            return False, sum(there), f'partial push: {sum(there)} of {len(rows)} destinations exist'
        transferred = bool(rows) and all(there)
    for r in rows:
        if dialect == 'push':
            done = transferred
        else:
            done = r['action'] == 'keep' or bool((r.get('written_at') or '').strip())
        f = r[dc] if done else r[sc]
        if not f or not os.path.exists(f):
            problems.append(f"{'destination' if done else 'source'} missing: {f}")
            continue
        if str(r.get('bytes', '')).strip() and os.path.getsize(f) != int(r['bytes']):
            problems.append(f"{f}: size {os.path.getsize(f)} != {r['bytes']}")
            continue
        if checksum and (r.get('md5') or '').strip() and K.checksum(f, algo) != r['md5']:
            problems.append(f'{f}: {algo} differs from the manifest')
            continue
        if dialect == 'publish' and not done:
            if r['action'] == 'publish' and os.path.lexists(r[dc]):
                problems.append(f'{r[dc]}: planned publish but the destination exists now')
            if r['action'] == 'replace' and not os.path.exists(r[dc]):
                problems.append(f'{r[dc]}: planned replace but the destination is gone')
    return _fmt(problems, len(rows), 'manifest rows')


# ---------------------------------------------------------------------------------------------------------------------
# quarantine_manifest
# ---------------------------------------------------------------------------------------------------------------------

def _norm_quarantine_rows(cols, rows):
    """(algo, [(src, dst, size, sum, moved_at or None, moved)]) for the three manifest dialects in use."""
    if 'original_path' in cols:                                     # snapshot_frametime_ship.sh MANIFEST.csv: all moved
        return 'sha256', [(r['original_path'], r['quarantine_path'], r.get('bytes'), r.get('sha256'), None, True) for r in rows]
    if 'quarantine_path' in cols:                                   # retire-excluded DRYRUN / APPLIED
        algo = 'md5'
        return algo, [(r['path'], r['quarantine_path'], r.get('size_bytes'), r.get('md5'), r.get('moved_at') or None,
                       bool((r.get('moved_at') or '').strip())) for r in rows]
    if 'dest' in cols and 'path' in cols:                           # p3s5_quarantine_frames.py / p3s5_archive_sync.py
        return 'md5', [(r['path'], r['dest'], r.get('bytes'), r.get('md5'), r.get('moved_at') or None,
                        bool((r.get('moved_at') or '').strip())) for r in rows]
    return None, None


@content_test('quarantine_manifest')
def rows_quarantined(path, since=None, paths=None, checksum=True, algo=None, **ctx):
    K = _K()
    cols, rows = K.read_csv(path)
    a, norm = _norm_quarantine_rows(cols, rows)
    if norm is None:
        return False, None, f'header {cols} is not a quarantine manifest'
    algo = algo or a
    if since:
        norm = [x for x in norm if not x[5] or (x[4] or '') >= since]
    if paths is not None:
        ps = set(paths)
        norm = [x for x in norm if x[0] in ps]
    last = {}
    for i, x in enumerate(norm):
        last[x[1]] = i
    superseded = len(norm) - len(last)
    norm = [x for i, x in enumerate(norm) if last[x[1]] == i]        # a destination reused by a later row: the later row holds
    problems, rolled = [], 0
    for src, dst, size, s, moved_at, moved in norm:
        if os.path.basename(src) != os.path.basename(dst):
            problems.append(f'{dst}: name differs from the source {os.path.basename(src)}')
            continue
        if not moved:
            if not os.path.exists(src):
                problems.append(f'planned source missing: {src}')
                continue
            if str(size or '').strip() and os.path.getsize(src) != int(size):
                problems.append(f'{src}: size {os.path.getsize(src)} != {size}')
                continue
            if os.path.lexists(dst):
                problems.append(f'G4: planned destination exists: {dst}')
                continue
            if K.sec().SAME_FILESYSTEM_ONLY and not K.same_filesystem(src, dst):
                problems.append(f'G3: {src} and {dst} on different filesystems')
                continue
            if checksum and (s or '').strip() and K.checksum(src, algo) != s:
                problems.append(f'{src}: {algo} differs from the plan')
            continue
        if not os.path.exists(dst):
            if os.path.exists(src) and (s or '').strip() and checksum and K.checksum(src, algo) == s:
                rolled += 1                                         # put back by a rollback (restore manifest)
                continue
            problems.append(f'quarantined copy missing: {dst}')
            continue
        if str(size or '').strip() and os.path.getsize(dst) != int(size):
            problems.append(f'{dst}: size {os.path.getsize(dst)} != {size}')
            continue
        if checksum and (s or '').strip() and K.checksum(dst, algo) != s:
            problems.append(f'{dst}: {algo} differs from the manifest')
            continue
        if os.path.exists(src) and os.stat(src).st_ino == os.stat(dst).st_ino:
            problems.append(f'{src}: still the same file as its quarantined copy (hard link?)')
    ok, v, d = _fmt(problems, len(norm), 'rows')
    extra = ([f'{superseded} row(s) superseded by a later row with the same destination'] if superseded else []) + \
            ([f'{rolled} row(s) rolled back (file back at its path with the manifest checksum)'] if rolled else [])
    return ok, v, d + ('; ' + '; '.join(extra) if extra else '')


# ---------------------------------------------------------------------------------------------------------------------
# archive_sync_record
# ---------------------------------------------------------------------------------------------------------------------

class _Roots:
    def __init__(self, root, run):
        self.archive_root, self.gen_run = root, run


@content_test('archive_sync_record')
def archive_equals_attempt(path, **ctx):
    from exopipe.phase3.lib.archive import sync as SY
    K = _K()
    rec = K.read_json(path)
    for k in ('kind', 'cell', 'stokes', 'attempt', 'archive_root', 'run', 'actions'):
        if k not in rec:
            return False, None, f'record lacks {k!r}'
    if not os.path.isdir(rec['attempt']):
        return False, None, f"attempt directory {rec['attempt']} is gone"
    S = SY.legacy_module()
    acts = [list(x) for x in SY.plan(S, _Roots(rec['archive_root'], rec['run']), rec['attempt'], rec['cell'], rec['stokes'])]
    if rec['kind'] == 'closing':
        bad = [x for x in acts if x[0] != 'keep']
        return (not bad), len(bad), (f'{len(bad)} action(s) other than keep, e.g. {bad[0]}' if bad else
                                     f'{len(acts)} archive frames equal the attempt frames in pixels')
    same = acts == rec['actions']
    c = SY.counts([tuple(x) for x in acts])
    return same, len(acts), ('plan re-derived now equals the record: ' + ', '.join(f'{k} {v}' for k, v in c.items())
                             if same else f'plan changed since the record ({len(acts)} actions now, '
                                          f'{len(rec["actions"])} recorded)')


# ---------------------------------------------------------------------------------------------------------------------
# retirement_audit
# ---------------------------------------------------------------------------------------------------------------------

def _stale(rec):
    from exopipe.phase3.core import exclusions
    now = exclusions.load().fingerprint()
    was = rec.get('exclusions') or {}
    if (now.get('sha1'), now.get('n_exclude_rows')) != (was.get('sha1'), was.get('n_exclude_rows')):
        return f"STALE: exclusion list changed since the record (sha1 {str(was.get('sha1'))[:12]} -> {now['sha1'][:12]}, " \
               f"{was.get('n_exclude_rows')} -> {now['n_exclude_rows']} exclude rows)"
    return None


@content_test('retirement_audit')
def no_excluded_product_left(path, **ctx):
    from exopipe.phase3.core import exclusions
    from exopipe.phase3.lib.archive import retire as R
    K = _K()
    rec = K.read_json(path)
    kind = rec.get('kind')
    if kind not in ('excluded', 'report'):
        return True, None, f'n/a (kind {kind})'
    st = _stale(rec)
    if st:
        return False, None, st
    ex = exclusions.load()
    if kind == 'report':
        _, rows = R.report_rows(rec['tier'], rec['cell'], ex)
        now, was = {r['path'] for r in rows}, set(rec.get('paths') or [])
        return now == was, len(now), (f'{len(now)} file(s) of excluded keys, as reported' if now == was else
                                      f'report changed: {len(now - was)} new, {len(was - now)} gone')
    filt = rec.get('filter') or {}
    keys = {tuple(k) for k in rec.get('keys') or []}
    not_excl = [k for k in keys if not ex.is_excluded(k[0], k[2], k[1])]
    if not_excl:
        return False, len(not_excl), f'{len(not_excl)} recorded key(s) no longer excluded, e.g. {not_excl[0]}'
    cells, nights = filt.get('cells'), filt.get('nights')
    excl = set(filt.get('exclude_cells') or [])

    def keep(lst, date, scope):
        c = f'{lst}_{scope}'
        return (cells is None or c in cells) and c not in excl and (nights is None or date in nights)
    found = R.scan_excluded(rec['archive_root'], rec['run'], tuple(rec.get('cadences') or ()), ex, lsts=filt.get('lsts'),
                            keep=keep)
    now = {p for v in found.values() for p in v}
    if rec.get('state') == 'applied':
        left = {p for k, v in found.items() if k in keys for p in v}
        return (not left), len(left), (f'0 files of the {len(keys)} retired key(s) left under the generation Run' if not left
                                       else f'{len(left)} file(s) of retired keys left, e.g. {sorted(left)[0]}')
    was = set(rec.get('paths') or [])
    return now == was, len(now), (f'{len(now)} file(s) of {len(found)} excluded key(s) found now = the plan' if now == was
                                  else f'plan changed: {len(now - was)} file(s) not in the plan, {len(was - now)} planned '
                                       f'file(s) gone')


@content_test('retirement_audit')
def no_refused_night_frame_left(path, **ctx):
    from exopipe.phase3.lib.archive import sync as SY
    K = _K()
    rec = K.read_json(path)
    if rec.get('kind') != 'refused':
        return True, None, f"n/a (kind {rec.get('kind')})"
    if not os.path.isdir(rec['attempt']):
        return False, None, f"attempt directory {rec['attempt']} is gone"
    S = SY.legacy_module()
    rows, excl = SY.refused_rows(S, _Roots(rec['archive_root'], rec['run']), rec['attempt'], rec['cell'], rec['stokes'],
                                 checksum=False)
    now = {r['path'] for r in rows}
    if rec.get('state') == 'applied':
        return (not now), len(now), ('every _msub file of the (cell, Stokes) is named by the attempt or excluded' if not now
                                     else f'{len(now)} frame(s) of refused nights left, e.g. {sorted(now)[0]}')
    was = set(rec.get('rows') or [])
    return now == was, len(now), (f'{len(now)} refused-night frame(s) = the plan' if now == was else
                                  f'plan changed: {len(now - was)} new, {len(was - now)} gone')


# ---------------------------------------------------------------------------------------------------------------------
# row_quarantine_manifest
# ---------------------------------------------------------------------------------------------------------------------

@content_test('row_quarantine_manifest')
def rows_moved_with_manifest(path, fresh=False, **ctx):
    import csv
    K = _K()
    sn = K.sec('archive_snapshot')
    qdir = os.path.dirname(os.path.abspath(path))
    cols, rows = K.read_csv(path)
    need = ('file', 'rows_before', 'quarantined', 'rows_after', 'sha256_before', 'sha256_after')
    if not set(need) <= set(cols):
        return False, None, f'header {cols} lacks {[c for c in need if c not in cols]}'
    problems = []
    shaf = os.path.join(qdir, 'MANIFEST.sha256')
    if not os.path.exists(shaf):
        problems.append('no MANIFEST.sha256')
    else:
        for line in open(shaf):
            if not line.strip():
                continue
            h, _, fn = line.strip().partition('  ')
            p = os.path.join(qdir, fn)
            if fn == 'MANIFEST.csv':
                continue                                            # appended by later runs; its own content is checked here
            if not os.path.exists(p) or K.checksum(p, 'sha256') != h:
                problems.append(f'MANIFEST.sha256: {fn} does not verify')
    for r in rows:
        nq = int(r['quarantined'] or 0)
        base = os.path.basename(r['file'])
        if nq:
            cp = os.path.join(qdir, base)
            if not os.path.exists(cp) or K.checksum(cp, 'sha256') != r['sha256_before']:
                problems.append(f'{base}: quarantined copy missing or sha256 != sha256_before')
            qr = cp + '.quarantined_rows.csv'
            if not os.path.exists(qr):
                problems.append(f'{base}: no quarantined_rows file')
            else:
                with open(qr, newline='') as f:
                    gone = list(csv.DictReader(f))
                if len(gone) != nq:
                    problems.append(f'{base}: {len(gone)} quarantined rows != {nq}')
                wrong = [g for g in gone if g.get('subband') not in sn.ROW_QUARANTINE_BANDS
                         or sn.ROW_QUARANTINE_MARK in (g.get('image') or '')]
                if wrong:
                    problems.append(f'{base}: {len(wrong)} quarantined row(s) are not unmarked wideband rows')
            if int(r['rows_before']) - nq != int(r['rows_after']):
                problems.append(f'{base}: rows_before - quarantined != rows_after')
        if fresh and os.path.exists(r['file']):
            with open(r['file'], newline='') as f:
                n = sum(1 for _ in csv.DictReader(f))
            if n != int(r['rows_after']):
                problems.append(f'{base}: live store has {n} rows, rows_after {r["rows_after"]}')
            if r.get('sha256_after') and K.checksum(r['file'], 'sha256') != r['sha256_after']:
                problems.append(f'{base}: live store sha256 != sha256_after')
    return _fmt(problems, len(rows), 'store rows')


# ---------------------------------------------------------------------------------------------------------------------
# restore_manifest
# ---------------------------------------------------------------------------------------------------------------------

def _later_quarantine(path, after, cell, stokes, qdir):
    """The moved_at of a quarantine-manifest row of ``path`` later than ``after`` (the restored file was quarantined
    again by a later search), or None."""
    K = _K()
    man = os.path.join(qdir, f'{cell}_{stokes}.csv')
    if not os.path.exists(man):
        return None
    _, rows = K.read_csv(man)
    later = [r['moved_at'] for r in rows if r.get('path') == path and (r.get('moved_at') or '') > after]
    return min(later) if later else None


@content_test('restore_manifest')
def rows_restored(path, quarantine_manifest_dir=None, checksum=True, **ctx):
    K = _K()
    cols, rows = K.read_csv(path)
    for c in ('cell', 'stokes', 'path', 'dest', 'md5', 'action'):
        if c not in cols:
            return False, None, f'header {cols} lacks {c!r}'
    qdir = quarantine_manifest_dir or K.expand(K.sec().FRAME_QUARANTINE_MANIFEST_DIR)
    problems, superseded = [], 0
    for r in rows:
        act = r['action']
        if act.startswith('would'):
            if not os.path.exists(r['dest']) or (checksum and K.checksum(r['dest'], 'md5') != r['md5']):
                problems.append(f"{r['dest']}: quarantined copy missing or md5 differs")
            elif act == 'would restore' and os.path.lexists(r['path']):
                problems.append(f"{r['path']}: occupied (planned plain restore)")
            continue
        when = r.get('restored_at') or ''
        if os.path.exists(r['path']) and (not checksum or K.checksum(r['path'], 'md5') == r['md5']):
            pass
        else:
            lq = _later_quarantine(r['path'], when, r['cell'], r['stokes'], qdir)
            if lq:
                superseded += 1
            else:
                problems.append(f"{r['path']}: not back with md5 {r['md5']} and no later quarantine explains it")
                continue
        if r.get('retired_to'):
            if not os.path.exists(r['retired_to']):
                problems.append(f"{r['retired_to']}: occupant copy missing")
            elif checksum and r.get('retired_md5') and K.checksum(r['retired_to'], 'md5') != r['retired_md5']:
                problems.append(f"{r['retired_to']}: occupant md5 differs")
    ok, v, d = _fmt(problems, len(rows), 'rows')
    return ok, v, d + (f'; {superseded} restored file(s) quarantined again later by a search (superseded rows)'
                       if superseded else '')


# ---------------------------------------------------------------------------------------------------------------------
# archive_plan / archive_mosaic_copy
# ---------------------------------------------------------------------------------------------------------------------

@content_test('archive_plan')
def plan_current(path, **ctx):
    from exopipe.phase3.core.record import fingerprint, same_fingerprint
    K = _K()
    rec = K.read_json(path)
    if 'action' not in rec:
        return False, None, 'plan record lacks action'
    changed = [p for p, fp in (rec.get('inputs') or {}).items() if not same_fingerprint(fp, fingerprint(p))]
    if changed:
        return False, len(changed), f'STALE: {len(changed)} input(s) changed since the plan, e.g. {changed[0]}'
    return True, len(rec.get('inputs') or {}), f"{rec['action']} plan; {len(rec.get('inputs') or {})} input(s) unchanged"


@content_test('archive_mosaic_copy')
def class_audit(path, **ctx):
    from exopipe.phase3.lib.archive.push import audit_mosaic
    p = audit_mosaic(path)
    return (not p), len(p), ('; '.join(p) if p else 'name, PBCORR, BUNIT, NSTACK, DATE-OBS/END, finite pixels agree')
