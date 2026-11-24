#!/usr/bin/env python
"""Two-week retention of the raw night data (`exopipe3 retention`, lib/retention/, config section retention; plan
CUTOVER_REMAINING_20261124.md row 40, ORCHESTRATION_DESIGN_20261117.md 5.4). Data free: a synthetic raw tree of small files in
the test's scratch directory (<subband>/<UT date>/<UT hour>/<YYYYMMDD_HHMMSS>_<subband>_averaged.ms.tar, dates relative to
the clock) and a synthetic fix ledger (CRLF). Every delete in this test is of that scratch tree (asserted before each one).
Asserts:
  - the table: production records and holds append one row each, read back; an empty range, a production before the range or
    in the future, no source, a name that is not a subband refuse (RT-RECORD / RT-HOLD); a damaged row or header refuses
    (RT-TABLE); a release ends a hold; a release of a hold that is not active refuses;
  - the tree is listed by name; names outside the layout are reported, never items (a date-shaped name that is no date
    among them: never a crash);
  - the status of each item: unprocessed (no record), held (a hold overlaps its hour, whatever its expiry), unrecognized (a
    subdirectory in it), empty, partial (a file no record covers), retained (expiry ahead), expired; the expiry is the
    latest covering production + RETENTION_DAYS (a re-processing restarts the clock);
  - the dry run: the plan table lists every item and every odd name; the manifest holds the oldest-expiring expired items
    up to the batch limit, each file's size, mtime and inode, an MS directory path by path (dir and file rows), per item an
    identity sha256; the sidecar's sha256 is the CSV's;
    no expired item -> no manifest; the content tests of retention_plan / retention_manifest pass and fail a tampered copy;
    the sidecar records the --out, table and holds digests the plan read; plan --dry-run lists and classifies only (no
    lstat, no checksum, nothing written);
  - the delete refuses, deleting nothing, with --out other than retention.ROOT (RT-ROOT), without --ruling-for
    retention.DELETE (RT-RULING), with an ID that is not a ledger row (config refusal), with a row that is not Gregg's (not
    R-GREGG-*, or authorized_by not Gregg, or a status not RULED), with Gregg's row that does not quote this manifest's
    sha256 (another manifest, the file name only, 11 hex digits, a longer hex string) (RT-RULING) - a row quoting the full
    sha256 passes; with a manifest planned from another --out (RT-MANIFEST); with the table rewritten since the plan
    (RT-RECORDS; appended rows are re-classified instead); with a file of an item changed (RT-MATCH; restoring the file's size and mtime makes it match again), a new
    file in an item or inside an MS directory (RT-MATCH), a hold made after the dry run (RT-HELD), a re-processing record (RT-NOT-EXPIRED), a manifest
    edited after its dry run (RT-MANIFEST), a manifest of another raw root (RT-MANIFEST); --dry-run deletes nothing;
  - then, with Gregg's row naming the manifest: only the listed paths are removed (an MS directory file by file, then its
    directories by rmdir), each written to the applied manifest (retention_applied passes), the emptied hour directories (and an emptied date directory) are removed, every other file
    of the tree is still there; the same manifest a second time refuses (RT-MATCH);
  - record / hold / release through the command; --ruling-for retention.DELETE on another action refuses (RT-ARGS);
  - the step is registered (cli.STEPS, config STEP_CONFIG_MODULES), has no space gate (SPACE_GATE = False: its writes free
    room) and RETENTION_DAYS is ruled by Gregg's 2026-11-17 row.
"""
import contextlib
import csv
import glob
import io
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exopipe.phase3 import cli  # noqa: E402
from exopipe.phase3 import config as C  # noqa: E402
from exopipe.phase3.config import CFG  # noqa: E402
from exopipe.phase3.core import validate  # noqa: E402
from exopipe.phase3.core.record import Refusal  # noqa: E402
from exopipe.phase3.lib.retention import plan as P  # noqa: E402
from exopipe.phase3.lib.retention import table as T  # noqa: E402

FAILS = []
DAY = 86400


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else '  -- %s' % (detail,)), flush=True)
    if not cond:
        FAILS.append(name)


class Patch:
    def __init__(self):
        self.saved = []

    def set(self, obj, name, value):
        self.saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, value in reversed(self.saved):
            setattr(obj, name, value)
        self.saved = []


def refuses(fn, code):
    try:
        fn()
    except Refusal as e:
        return str(e).startswith(code) or code in str(e).split(':')[0], str(e)
    return False, 'no refusal'


def day(t):
    return time.strftime('%Y-%m-%d', time.gmtime(t))


class World:
    """The synthetic raw tree under <d>/raw. Dates: OLD = 30 days ago (expired once recorded), NEW = 2 days ago (retained)."""

    def __init__(self, d):
        now = time.time()
        self.d, self.root, self.out = d, os.path.join(d, 'raw'), os.path.join(d, 'out')
        self.t_old = T.parse_utc(day(now - 30 * DAY))
        self.t_new = T.parse_utc(day(now - 2 * DAY))
        self.OLD, self.NEW, self.OLD2 = day(self.t_old), day(self.t_new), day(self.t_old + DAY)
        self.files = []
        self._f('73MHz', self.OLD, 2, (0, 600, 1200), (100, 200, 300))      # expired
        self._f('78MHz', self.OLD, 2, (0, 600), (400, 500))                 # expired
        self._f('73MHz', self.OLD, 3, (0, 600), (10, 20))                   # held
        self._f('78MHz', self.OLD, 4, (0,), (7,))                           # unrecognized (a subdirectory)
        os.makedirs(os.path.join(self.root, '78MHz', self.OLD, '04', 'junk'))
        os.makedirs(os.path.join(self.root, '73MHz', self.OLD2, '05'))       # empty
        self._f('78MHz', self.OLD, 6, (0, 1800), (5, 6))                    # partial
        self._f('73MHz', self.NEW, 7, (0,), (9,))                           # retained
        self._f('73MHz', self.NEW, 9, (0,), (3,))                           # unprocessed
        self._ms('78MHz', self.OLD2, 1)                                      # expired MS directory, alone in its date
        with open(os.path.join(self.root, 'README'), 'w') as f:            # odd names
            f.write('x')
        os.makedirs(os.path.join(self.root, 'notes'))
        os.makedirs(os.path.join(self.root, '73MHz', 'notadate'))
        os.makedirs(os.path.join(self.root, '73MHz', '2026-13-45', '02'))   # the date pattern, no date: odd, never a crash
        os.makedirs(os.path.join(self.root, '73MHz', self.OLD, 'xx'))

    def _f(self, sb, date, hh, offs, sizes):
        d = os.path.join(self.root, sb, date, '%02d' % hh)
        os.makedirs(d, exist_ok=True)
        t0 = T.parse_utc(date) + hh * 3600
        for o, n in zip(offs, sizes):
            p = os.path.join(d, '%s_%s_averaged.ms.tar' % (time.strftime('%Y%m%d_%H%M%S', time.gmtime(t0 + o)), sb))
            with open(p, 'wb') as f:
                f.write(b'v' * n)
            self.files.append(p)

    def _ms(self, sb, date, hh):
        """An MS directory (two files, a subdirectory with one file; 55 bytes) and its flagmeta.bin (10 bytes)."""
        d = os.path.join(self.root, sb, date, '%02d' % hh)
        stem = '%s_%s' % (time.strftime('%Y%m%d_%H%M%S', time.gmtime(T.parse_utc(date) + hh * 3600)), sb)
        self.ms = os.path.join(d, stem + '_averaged.ms')
        os.makedirs(os.path.join(self.ms, 'ANTENNA'))
        for rel, n in (('table.dat', 30), ('table.f1', 20), (os.path.join('ANTENNA', 'table.f0'), 5)):
            with open(os.path.join(self.ms, rel), 'wb') as f:
                f.write(b'm' * n)
            self.files.append(os.path.join(self.ms, rel))
        with open(os.path.join(d, stem + '_flagmeta.bin'), 'wb') as f:
            f.write(b'b' * 10)
        self.files.append(os.path.join(d, stem + '_flagmeta.bin'))

    def item(self, sb, date, hh):
        return '%s/%s/%02d' % (sb, date, hh)

    def records(self, out=None):
        out = out or self.out
        o, n = self.t_old, self.t_new
        T.record_production(out, o, o + 5 * 3600, '*', o + 12 * 3600, 'test chain old night')
        T.record_production(out, o + 6 * 3600, o + 6 * 3600 + 1200, '78MHz', o + 12 * 3600, 'test partial')
        T.record_production(out, n + 7 * 3600, n + 8 * 3600, ['73MHz'], n + 12 * 3600, 'test chain new night')
        T.record_production(out, o + DAY + 3600, o + DAY + 7200, '78MHz', o + DAY + 12 * 3600, 'test chain second night')
        return T.hold(out, o + 3 * 3600 + 600, o + 3 * 3600 + 1200, '73MHz', 'test candidate')

    def present(self):
        return sorted(p for p in self.files if os.path.exists(p))


def write_ledger(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\r\n')
        w.writerow(['date', 'fix_id', 'description', 'authorized_by', 'status'])
        for r in rows:
            w.writerow(r)


def test_table(d):
    out = os.path.join(d, 'tab')
    now = time.time()
    rid = T.record_production(out, now - 3 * DAY, now - 2 * DAY, '73MHz,78MHz', now - DAY, 'chain x')
    recs = T.read_records(out)
    check('record: one row read back (subbands, times)', len(recs) == 1 and recs[0]['record_id'] == rid
          and recs[0]['subbands'] == {'73MHz', '78MHz'} and abs(recs[0]['produced'] - (now - DAY)) < 1, recs)
    with open(T.table_path(out)) as f:
        head = f.readline().strip()
    check('the table\'s header is written with the file', head == ','.join(T.TABLE_COLUMNS), head)
    for what, call, code in (
            ('an empty range', lambda: T.record_production(out, now, now, '*', now, 's'), 'RT-RECORD'),
            ('a production before the range', lambda: T.record_production(out, now - DAY, now, '*', now - 2 * DAY, 's'),
             'RT-RECORD'),
            ('a production in the future', lambda: T.record_production(out, now - DAY, now, '*', now + DAY, 's'), 'RT-RECORD'),
            ('no source', lambda: T.record_production(out, now - DAY, now, '*', now, ' '), 'RT-RECORD'),
            ('a name that is not a subband', lambda: T.record_production(out, now - DAY, now, 'Full', now, 's'), 'RT-RECORD'),
            ('a hold without a reason', lambda: T.hold(out, now - DAY, now, '*', ''), 'RT-HOLD'),
            ('a release of a hold that is not active', lambda: T.release(out, 'H-nope', 'r'), 'RT-HOLD')):
        ok, why = refuses(call, code)
        check('refused (%s): %s' % (code, what), ok, why)
    check('the refused records appended nothing', len(T.read_records(out)) == 1)
    hid = T.hold(out, now - DAY, now, '*', 'candidate X')
    check('hold: active', [h['hold_id'] for h in T.read_holds(out)] == [hid])
    T.release(out, hid, 'extracted')
    check('release: no active hold', T.read_holds(out) == [])
    with open(T.table_path(out), 'a') as f:
        f.write('P-bad,not-a-time,2026-11-01T00:00:00Z,*,2026-11-01T00:00:00Z,s,x,y\n')
    ok, why = refuses(lambda: T.read_records(out), 'RT-TABLE')
    check('a damaged row refuses (RT-TABLE), never read as fewer records', ok, why)
    bad = os.path.join(d, 'tab_bad')
    os.makedirs(bad)
    with open(T.holds_path(bad), 'w') as f:
        f.write('a,b,c\n')
    ok, why = refuses(lambda: T.read_holds(bad), 'RT-TABLE')
    check('a holds file with another header refuses (RT-TABLE)', ok, why)
    ok, why = refuses(lambda: T.hold(bad, now - DAY, now, '*', 'r'), 'RT-TABLE')
    check('... and is never appended to', ok and open(T.holds_path(bad)).read() == 'a,b,c\n', why)


def test_classify(W):
    items, odd = P.list_tree(W.root)
    names = {it['item'] for it in items}
    check('the tree by name: 9 items', len(items) == 9, sorted(names))
    want_odd = {os.path.join(W.root, 'README'), os.path.join(W.root, 'notes') + '/',
                os.path.join(W.root, '73MHz', 'notadate') + '/', os.path.join(W.root, '73MHz', W.OLD, 'xx') + '/',
                os.path.join(W.root, '73MHz', '2026-13-45') + '/'}
    check('names outside the layout reported, never items', set(odd) == want_odd, odd)
    pl = P.make_plan(W.root, W.out, max_items=10)
    st = {it['item']: cls['status'] for it, cls in pl.items}
    want = {W.item('73MHz', W.OLD, 2): 'expired', W.item('78MHz', W.OLD, 2): 'expired',
            W.item('73MHz', W.OLD, 3): 'held', W.item('78MHz', W.OLD, 4): 'unrecognized',
            W.item('73MHz', W.OLD2, 5): 'unprocessed', W.item('78MHz', W.OLD, 6): 'partial',
            W.item('73MHz', W.NEW, 7): 'retained', W.item('73MHz', W.NEW, 9): 'unprocessed',
            W.item('78MHz', W.OLD2, 1): 'expired', W.item('78MHz', W.OLD, 3): None}
    got = {k: st.get(k) for k in want}
    check('statuses: expired x3, held, unrecognized, unprocessed x2, partial, retained', got == want, got)
    cls = dict((it['item'], c) for it, c in pl.items)[W.item('73MHz', W.OLD, 2)]
    check('expiry = production + 14 days', cls['expiry'] - cls['produced'] == 14 * DAY
          and cls['produced'] == W.t_old + 12 * 3600, cls)
    empty_it = dict(item='73MHz/%s/05' % W.OLD2, subband='73MHz', date=W.OLD2, hour='05',
                    path=os.path.join(W.root, '73MHz', W.OLD2, '05'), t0=W.t_old + DAY + 5 * 3600)
    empty_it['t1'] = empty_it['t0'] + 3600
    rec = [dict(record_id='P-x', start=W.t_old, end=W.t_old + 3 * DAY, subbands=None, produced=W.t_old + DAY, source='s')]
    check('empty: an hour directory without files (recorded)', P.classify(empty_it, rec, [], time.time())['status'] == 'empty')
    first = dict((it['item'], it) for it, c in pl.items)[W.item('73MHz', W.OLD, 2)]
    c3 = P.classify(first, T.read_records(W.out) + [dict(record_id='P-z', start=W.t_old, end=W.t_old + 3 * 3600,
                                                          subbands=None, produced=time.time() - 60, source='s')],
                    [], time.time())
    check('... an expired hour re-processed now is retained again', c3['status'] == 'retained', c3)
    c4 = P.classify(first, T.read_records(W.out), [dict(hold_id='H-x', start=0, end=2e9, subbands={'78MHz'}, reason='r')],
                    time.time())
    check('a hold of another subband does not hold it', c4['status'] == 'expired', c4)
    pl1 = P.make_plan(W.root, W.out, max_items=1)
    check('the batch limit: the oldest-expiring expired item first (73MHz before 78MHz at the same expiry)',
          [it['item'] for it, c, f in pl1.batch] == [W.item('73MHz', W.OLD, 2)] and pl1.n_expired == 3,
          [it['item'] for it, c, f in pl1.batch])
    pl0 = P.make_plan(W.root, W.out, now=W.t_old + 13 * DAY)
    p0, m0, j0 = P.write_plan(pl0, os.path.join(W.d, 'out_what_if'))
    check('no expired item (a what-if 13 days after the night): a plan table, no manifest', os.path.exists(p0)
          and m0 is None and j0 is None and pl0.counts().get('expired') is None, pl0.counts())
    ok, why = refuses(lambda: P.make_plan(W.root, W.out, checksum='md5'), 'RT-CHECKSUM')
    check('an unknown checksum kind refuses', ok, why)
    ok, why = refuses(lambda: P.list_tree(os.path.join(W.d, 'no_such_root')), 'RT-ROOT')
    check('a raw root that is not a directory refuses', ok, why)


def go(argv):
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            rc = cli.main(['retention'] + argv + ['--cores', 'none'])
    finally:
        CFG.reset()
    return rc, err.getvalue()


def test_step(W):
    check('the step is registered (cli.STEPS, config STEP_CONFIG_MODULES)', 'retention' in [s[0] for s in cli.STEPS]
          and 'retention' in C.STEP_CONFIG_MODULES)
    from exopipe.phase3.steps import retention as step
    check('the step has no space gate (SPACE_GATE = False: its delete frees room; cli.main skips the gate)',
          step.SPACE_GATE is False and step.WRITES is True)
    check('RETENTION_DAYS 14, ruled by R-GREGG-20261117-ORCHESTRATION-DESIGN-COMMENTS',
          CFG.retention.RETENTION_DAYS == 14
          and CFG.retention.RULED.get('RETENTION_DAYS') == 'R-GREGG-20261117-ORCHESTRATION-DESIGN-COMMENTS')
    before = W.present()
    rc, err = go(['--out', W.out])
    plans = sorted(glob.glob(os.path.join(W.out, 'plans', 'RETENTION_TABLE_*.csv')))
    mans = sorted(glob.glob(os.path.join(W.out, 'manifests', 'DRYRUN_RETENTION_*.csv')))
    check('plan (the default action): exit 0, a plan table and a manifest, nothing deleted', rc == 0 and len(plans) == 1
          and len(mans) == 1 and W.present() == before, (rc, err[-800:]))
    man = mans[0]
    side = json.load(open(os.path.splitext(man)[0] + '.json'))
    Pd = Patch()
    try:
        def no_stat(*a, **k):
            raise AssertionError('stat_entries called in a dry run')
        Pd.set(P, 'stat_entries', no_stat)
        Pd.set(P, 'sha256_file', no_stat)
        rc, err = go(['plan', '--out', W.out, '--dry-run'])
    finally:
        Pd.undo()
    check('plan --dry-run: exit 0, the counts printed (3 expired, 3 would be in the manifest), no lstat or checksum of an '
          'entry, nothing written', rc == 0 and 'no stat' in err and '3 expired, 3 would be in the manifest' in err
          and sorted(glob.glob(os.path.join(W.out, 'plans', '*'))) == plans
          and sorted(glob.glob(os.path.join(W.out, 'manifests', '*'))) == mans + [os.path.splitext(man)[0] + '.json'],
          (rc, err[-600:]))
    check('the sidecar: 3 items, 9 files, 2 directories (an MS and its subdirectory), 1565 bytes, identity checksum, the raw '
          'root, 14 days', side['n_items'] == 3 and side['n_files'] == 9 and side['n_dirs'] == 2 and side['bytes'] == 1565 and side['checksum'] == 'identity'
          and side['raw_root'] == os.path.realpath(W.root) and side['retention_days'] == 14
          and side['csv_sha256'] == P.sha256_file(man), side)
    rows = list(csv.DictReader(open(man)))
    dirs = sorted(r['path'] for r in rows if r['kind'] == 'dir')
    check('the MS directory is listed path by path: the MS and its subdirectory as dir rows, its 3 files as file rows',
          dirs == [W.ms, os.path.join(W.ms, 'ANTENNA')] and sum(r['kind'] == 'file' and r['path'].startswith(W.ms + '/')
                                                               for r in rows) == 3, dirs)
    f0 = [r for r in rows if r['kind'] == 'file'][0]
    st = os.lstat(f0['path'])
    check('a file row: its size, mtime (ns) and inode by lstat', (int(f0['bytes']), int(f0['mtime_ns']), int(f0['inode']))
          == (st.st_size, st.st_mtime_ns, st.st_ino), f0)
    for cls, path in (('retention_plan', plans[0]), ('retention_manifest', man), ('retention_records', T.table_path(W.out)),
                      ('retention_holds', T.holds_path(W.out))):
        res = validate.run(cls, path)
        check('content test %s passes on what the step wrote' % cls, res and all(r['passed'] for r in res), res)
    bad = os.path.join(W.d, 'tampered.csv')
    with open(man) as f, open(bad, 'w') as g:
        g.write(f.read().replace(',100,', ',101,', 1))
    side_bad = dict(side, csv=os.path.basename(bad), csv_sha256=P.sha256_file(bad))
    json.dump(side_bad, open(os.path.splitext(bad)[0] + '.json', 'w'))
    res = validate.run('retention_manifest', bad)
    check('content test retention_manifest fails a manifest whose file size disagrees with its item row',
          res and not all(r['passed'] for r in res), res)

    # a plan of the same tree from a copy of the records (another --out): a manifest a delete must not take
    out2 = os.path.join(W.d, 'out_copy')
    os.makedirs(out2)
    for f in (T.table_path(W.out), T.holds_path(W.out)):
        shutil.copy2(f, out2)
    rc, err = go(['plan', '--out', out2])
    man2 = (sorted(glob.glob(os.path.join(out2, 'manifests', 'DRYRUN_RETENTION_*.csv'))) or [None])[0]
    check('a plan from a copy of the records (another --out) writes its own manifest', rc == 0 and man2, err[-400:])
    side2 = json.load(open(os.path.splitext(man2)[0] + '.json'))
    check('the sidecar records what the plan read: the real --out, the table and holds sizes and sha256s',
          side['records']['root'] == os.path.realpath(W.out) and side['records']['table'] == list(
              P.prefix_digest(T.table_path(W.out))) and side['records']['holds'] == list(P.prefix_digest(T.holds_path(W.out)))
          and side2['records']['root'] == os.path.realpath(out2), (side.get('records'), side2.get('records')))

    led = os.path.join(W.d, 'fix_ledger.csv')
    sha = side['csv_sha256']
    sha12 = sha[:12]
    write_ledger(led, [
        ['2026-11-24', 'R-GREGG-TEST-GO', 'Gregg: go to delete the raw data of the manifest with sha256 %s' % sha12,
         'Gregg 2026-11-24', 'RULED - test'],
        ['2026-11-24', 'R-GREGG-TEST-FULL', 'Gregg: go for sha256 %s (%s)' % (sha.upper(), os.path.basename(man)),
         'Gregg Hallinan (terminal)', 'ruled'],
        ['2026-11-24', 'R-GREGG-TEST-NAME', 'Gregg: go to delete the raw data of manifest %s' % os.path.basename(man),
         'Gregg', 'RULED'],
        ['2026-11-24', 'R-GREGG-TEST-SHORT', 'Gregg: go for the manifest whose sha256 starts %s' % sha[:11], 'Gregg', 'RULED'],
        ['2026-11-24', 'R-GREGG-TEST-INSIDE', 'Gregg: go for 0%s' % sha12, 'Gregg', 'RULED'],
        ['2026-11-24', 'R-GREGG-TEST-NOTRULED', 'Gregg asked about manifest %s' % sha12, 'Gregg', 'RECEIVED, NOT RULED'],
        ['2026-11-24', 'R-GREGG-TEST-NOTGREGG', 'a go for manifest %s' % sha12, 'refactor chat', 'RULED'],
        ['2026-11-24', 'R-GREGG-TEST-OTHER', 'Gregg: go for DRYRUN_RETENTION_20990101T000000Z.csv, sha256 0123456789abcdef',
         'Gregg', 'RULED'],
        ['2026-11-24', 'R-GREGG-TEST-M2', 'Gregg: go for the manifest with sha256 %s' % side2['csv_sha256'][:16], 'Gregg',
         'RULED'],
        ['2026-11-24', 'FIX-TEST-NOT-GREGG', 'a fix naming %s' % sha12, 'refactor chat', 'DONE']])
    Pt = Patch()
    try:
        Pt.set(CFG.paths, 'FIX_LEDGER', led)
        root_real = os.path.realpath(W.root)
        assert root_real.startswith(os.path.realpath(W.d) + os.sep), 'the raw root of a delete test is the scratch tree'
        assert os.path.realpath(CFG.retention.RAW_ROOT) == root_real
        dele = ['delete', '--out', W.out, '--manifest', man]
        go_ = ['--ruling-for', 'retention.DELETE=R-GREGG-TEST-GO']
        rc, err = go(dele + go_ + ['--dry-run'])
        check('delete with --out other than retention.ROOT: RT-ROOT (the production records only), nothing deleted',
              rc == 2 and 'RT-ROOT' in err and W.present() == before, (rc, err[-600:]))
        Pt.set(CFG.retention, 'ROOT', W.out)
        assert os.path.realpath(CFG.retention.ROOT).startswith(os.path.realpath(W.d) + os.sep)
        for what, extra, code in (
                ('no --ruling-for', [], 'RT-RULING'),
                ('an ID that is not a ledger row', ['--ruling-for', 'retention.DELETE=R-GREGG-NO-SUCH-ROW'], 'no row'),
                ('a row that is not Gregg\'s', ['--ruling-for', 'retention.DELETE=FIX-TEST-NOT-GREGG'], 'RT-RULING'),
                ('Gregg\'s row naming another manifest', ['--ruling-for', 'retention.DELETE=R-GREGG-TEST-OTHER'], 'RT-RULING'),
                ('Gregg\'s row naming the manifest by FILE NAME only', ['--ruling-for', 'retention.DELETE=R-GREGG-TEST-NAME'],
                 'RT-RULING'),
                ('a row quoting 11 hex digits of the sha256', ['--ruling-for', 'retention.DELETE=R-GREGG-TEST-SHORT'],
                 'RT-RULING'),
                ('a row whose hex string only contains the sha256 prefix', ['--ruling-for',
                                                                           'retention.DELETE=R-GREGG-TEST-INSIDE'],
                 'RT-RULING'),
                ('an R-GREGG row whose status is not RULED', ['--ruling-for', 'retention.DELETE=R-GREGG-TEST-NOTRULED'],
                 'RT-RULING'),
                ('an R-GREGG row not authorized by Gregg', ['--ruling-for', 'retention.DELETE=R-GREGG-TEST-NOTGREGG'],
                 'RT-RULING')):
            rc, err = go(dele + extra)
            check('delete refused (%s): exit 2, %s, nothing deleted' % (what, code), rc == 2 and code in err
                  and W.present() == before, (rc, err[-600:]))
        rc, err = go(dele + ['--ruling-for', 'retention.DELETE=R-GREGG-TEST-FULL', '--dry-run'])
        check('a row quoting the full sha256 (any case), authorized_by "Gregg Hallinan (terminal)", status "ruled": accepted '
              '(--dry-run: exit 0, nothing deleted)', rc == 0 and 'the manifest holds' in err and W.present() == before,
              err[-600:])
        rc, err = go(['delete', '--out', W.out, '--manifest', man2, '--ruling-for', 'retention.DELETE=R-GREGG-TEST-M2'])
        check('a manifest planned from another --out (a copy of the records), with Gregg\'s row for it: RT-MANIFEST, '
              'nothing deleted', rc == 2 and 'RT-MANIFEST' in err and 'planned from the records' in err
              and W.present() == before, err[-600:])
        tab = T.table_path(W.out)
        tab0 = open(tab, 'rb').read()
        with open(tab, 'wb') as f:
            f.write(tab0.replace(b'test chain new night', b'Test chain new night'))
        rc, err = go(dele + go_ + ['--dry-run'])
        check('the retention table REWRITTEN after the dry run (a row edited in place): RT-RECORDS, nothing deleted',
              rc == 2 and 'RT-RECORDS' in err and W.present() == before, err[-600:])
        with open(tab, 'wb') as f:
            f.write(tab0)
        rc, err = go(dele + go_ + ['--dry-run'])
        check('... restored byte for byte: the preflight holds', rc == 0, err[-600:])
        # a file changed, then restored to the same size and mtime
        p = rows[[r['kind'] for r in rows].index('file')]['path']
        st0 = os.lstat(p)
        with open(p, 'ab') as f:
            f.write(b'+')
        rc, err = go(dele + go_)
        check('a file changed after the dry run: RT-MATCH, nothing deleted', rc == 2 and 'RT-MATCH' in err
              and W.present() == before, err[-600:])
        with open(p, 'r+b') as f:
            f.truncate(st0.st_size)
        os.utime(p, ns=(st0.st_atime_ns, st0.st_mtime_ns))
        rc, err = go(dele + go_ + ['--dry-run'])
        check('... restored (same size, mtime, inode): the preflight holds again (--dry-run: exit 0, nothing deleted)',
              rc == 0 and 'the manifest holds' in err and W.present() == before, err[-600:])
        extra_f = os.path.join(os.path.dirname(p), os.path.basename(p).replace('_averaged', '_late'))
        with open(extra_f, 'w') as f:
            f.write('n')
        rc, err = go(dele + go_)
        check('a new file in an item: RT-MATCH, nothing deleted', rc == 2 and 'RT-MATCH' in err
              and W.present() == before, err[-600:])
        os.unlink(extra_f)
        sub = os.path.join(W.ms, 'ANTENNA')
        st_sub = os.lstat(sub)
        inner = os.path.join(sub, 'table.f9')
        with open(inner, 'w') as f:
            f.write('n')
        rc, err = go(dele + go_)
        check('a new file inside an MS directory: RT-MATCH, nothing deleted', rc == 2 and 'RT-MATCH' in err
              and W.present() == before and os.path.exists(inner), err[-600:])
        os.unlink(inner)
        os.utime(sub, ns=(st_sub.st_atime_ns, st_sub.st_mtime_ns))
        rc, err = go(dele + go_ + ['--dry-run'])
        check('... removed again (the subdirectory\'s mtime restored): the preflight holds', rc == 0, err[-600:])
        o = W.t_old
        hid = T.hold(W.out, o + 2 * 3600 + 100, o + 2 * 3600 + 200, '78MHz', 'test hold after the dry run')
        rc, err = go(dele + go_)
        check('a hold made after the dry run: RT-HELD, nothing deleted', rc == 2 and 'RT-HELD' in err
              and W.present() == before, err[-600:])
        T.release(W.out, hid, 'test release')
        rc, err = go(dele + go_ + ['--dry-run'])
        check('... released (the holds file grew by two rows, its planned bytes unchanged): the preflight holds', rc == 0,
              err[-600:])
        tab0 = open(tab, 'rb').read()
        T.record_production(W.out, o + 2 * 3600, o + 3 * 3600, '73MHz', time.time() - 60, 'test re-processing')
        rc, err = go(dele + go_)
        check('a re-processing record appended after the dry run: RT-NOT-EXPIRED (not RT-RECORDS: the table only grew), '
              'nothing deleted', rc == 2 and 'RT-NOT-EXPIRED' in err and 'RT-RECORDS' not in err
              and W.present() == before, err[-600:])
        with open(tab, 'wb') as f:          # the test's own scratch table back to the bytes the plan read plus nothing
            f.write(tab0)
        ed = os.path.join(W.out, 'manifests', 'DRYRUN_RETENTION_19990101T000000Z.csv')
        with open(man) as f, open(ed, 'w') as g:
            g.write(f.read() + 'file,x,y,1,1,1,,,,,,\n')
        json.dump(dict(side, csv=os.path.basename(ed)), open(os.path.splitext(ed)[0] + '.json', 'w'))
        write_ledger(led, [['2026-11-24', 'R-GREGG-TEST-GO', 'go for the manifest of sha256 %s, and its copy' % sha12,
                            'Gregg', 'RULED']])
        rc, err = go(['delete', '--out', W.out, '--manifest', ed] + go_)
        check('a manifest edited after its dry run: RT-MANIFEST, nothing deleted', rc == 2 and 'RT-MANIFEST' in err
              and W.present() == before, err[-600:])
        other = os.path.join(W.d, 'raw_other')
        os.makedirs(other)
        Pt.set(CFG.retention, 'RAW_ROOT', other)
        rc, err = go(dele + go_)
        check('a manifest of another raw root: RT-MANIFEST, nothing deleted', rc == 2 and 'RT-MANIFEST' in err
              and W.present() == before, err[-600:])
        Pt.saved.pop()
        CFG.retention.RAW_ROOT = W.root
        rc, err = go(['plan', '--out', W.out] + go_)
        check('--ruling-for retention.DELETE on another action: RT-ARGS', rc == 2 and 'RT-ARGS' in err, err[-400:])
        # the delete
        listed = sorted(r['path'] for r in rows if r['kind'] == 'file')
        assert all(os.path.realpath(x).startswith(root_real + os.sep) for x in listed)
        rc, err = go(dele + go_)
        after = W.present()
        check('delete with Gregg\'s row naming the manifest: exit 0, exactly the 9 listed files gone, every other file kept',
              rc == 0 and after == sorted(set(before) - set(listed)) and len(before) - len(after) == 9, (rc, err[-800:]))
        check('the MS directory is gone (its subdirectory removed first, both by rmdir)', not os.path.lexists(W.ms))
        gone_dirs = [os.path.join(W.root, '73MHz', W.OLD, '02'), os.path.join(W.root, '78MHz', W.OLD, '02')]
        check('the emptied hour directories are removed, and the emptied date directory; the date directories that hold '
              'other hours are kept', not any(os.path.exists(x) for x in gone_dirs + [os.path.join(W.root, '78MHz', W.OLD2)])
              and os.path.isdir(os.path.join(W.root, '73MHz', W.OLD))
              and os.path.isdir(os.path.join(W.root, '78MHz', W.OLD)))
        check('the held, partial, unrecognized, retained and unprocessed items are untouched',
              all(os.path.isdir(os.path.join(W.root, *W.item(*x).split('/'))) for x in (
                  ('73MHz', W.OLD, 3), ('78MHz', W.OLD, 6), ('78MHz', W.OLD, 4), ('73MHz', W.NEW, 7), ('73MHz', W.NEW, 9))))
        app = glob.glob(os.path.join(W.out, 'manifests', 'APPLIED_RETENTION_*.csv'))
        res = validate.run('retention_applied', app[0]) if len(app) == 1 else []
        arows = list(csv.DictReader(open(app[0]))) if len(app) == 1 else []
        check('the applied manifest lists the 9 deleted files and the 2 removed directories; retention_applied passes',
              len(arows) == 11 and sorted(r['path'] for r in arows if r['kind'] == 'file') == listed
              and sorted(r['path'] for r in arows if r['kind'] == 'dir') == dirs and res and all(r['passed'] for r in res),
              (app, res))
        recs = glob.glob(os.path.join(W.out, '_records', 'retention', 'delete_*.json'))
        check('one unit record per item, done, naming the ruling', len(recs) == 3 and all(
            json.load(open(r))['status'] == 'done' and json.load(open(r))['meta'].get('ruling') == 'R-GREGG-TEST-GO'
            for r in recs), recs)
        rc, err = go(dele + go_)
        check('the same manifest a second time: RT-MATCH (the directories are gone)', rc == 2 and 'RT-MATCH' in err,
              err[-400:])
        # record / hold / release through the command
        rc, err = go(['record', '--out', W.out, '--start', '%sT10:00:00Z' % W.NEW, '--end', '%sT11:00:00Z' % W.NEW,
                      '--produced', '%sT12:00:00Z' % W.NEW, '--scopes', '73MHz', '--source', 'test cli'])
        check('record through the command: exit 0, one more record', rc == 0 and len(T.read_records(W.out)) == 5,
              (rc, err[-400:]))
        rc, err = go(['hold', '--out', W.out, '--start', '%sT10:00:00Z' % W.NEW, '--end', '%sT10:30:00Z' % W.NEW,
                      '--reason', 'test cli hold'])
        hid = err.split('hold ', 1)[1].split()[0] if 'hold H-' in err else ''
        check('hold through the command: exit 0, active', rc == 0 and hid in [h['hold_id'] for h in T.read_holds(W.out)],
              (rc, err[-400:]))
        rc, err = go(['release', '--out', W.out, '--hold-id', hid, '--reason', 'test cli release'])
        check('release through the command: exit 0, released', rc == 0 and hid not in [h['hold_id'] for h in
                                                                                       T.read_holds(W.out)], err[-400:])
        rc, err = go(['record', '--out', W.out, '--start', '%sT10:00:00Z' % W.NEW, '--end', '%sT09:00:00Z' % W.NEW,
                      '--source', 'x'])
        check('record of an empty range through the command: exit 2 RT-RECORD', rc == 2 and 'RT-RECORD' in err, err[-400:])
    finally:
        Pt.undo()


def main():
    base = os.environ.get('EXOPIPE3_TEST_SCRATCH') or None
    d = tempfile.mkdtemp(prefix='test_retention_', dir=base)
    Pt = Patch()
    try:
        test_table(d)
        W = World(os.path.join(d, 'w'))
        Pt.set(CFG.retention, 'RAW_ROOT', W.root)
        W.records()
        test_classify(W)
        test_step(W)
    finally:
        Pt.undo()
        shutil.rmtree(d, ignore_errors=True)
    print('%d check(s) failed: %s' % (len(FAILS), FAILS) if FAILS else 'all checks passed', flush=True)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
