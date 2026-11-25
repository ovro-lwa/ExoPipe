#!/usr/bin/env python
"""The do-not-delete gate of the raw-data retention (lib/retention/protect.py; core/exclusions.do_not_delete_cells;
core/cells.epoch_utc). Gregg marks cells "do not delete" in data_decisions.csv (e.g. "ionospheric vortex event; do not delete
(Gregg 2026-06-28)"); the October 2026 sunlit deletion, which did not read them, removed the cell LST 19h of 2025-05-04.
Data free: a synthetic raw tree of small files (date 2025-05-04, UT hours 11-14, subbands 13, 18, 46 and 90 MHz), synthetic
data_decisions.csv files and a synthetic fix ledger (CRLF) in the test's scratch directory; every delete is of that tree
(asserted). On the cluster it also reads the real data_decisions.csv (read only).
Asserts:
  - core/exclusions: the do-not-delete rows are read whatever their verdict, scope tag or subband ('Do-Not-Delete' matches;
    'deleted' and 'do delete' do not); a do-not-delete row that cannot be read (an LST, date or subband that is not one, more
    fields than the header) refuses the do-not-delete read (fail closed) but not the exclusion read of the co-adds; a
    missing file refuses;
  - core/cells.epoch_utc: LST 19h of 2025-05-04 is 12:02:37-13:02:27 UTC on 2025-05-04 (Phase 1's conversion), and the
    import of Phase 1's controller leaves this process's root logger as it was; a malformed LST or date is a ValueError;
  - protect.cells_over: an item's data run one integration past its hour (a cell starting 5 s after the hour is over it, one
    starting 11 s after is not);
  - the plan: the hours 12 and 13 of 2025-05-04 are 'protected' in EVERY subband - 13 MHz and 90 MHz included, which no row
    names - never listed, never in the manifest; hours 11 and 14 are expired and planned normally; the sidecar records the
    cells it honored; the content tests pass;
  - a PLANTED manifest (planned while the list had no do-not-delete row) that holds the protected hours: the delete refuses
    the whole manifest (RT-DO-NOT-DELETE), with Gregg's go for that manifest and in --dry-run, naming the cell and the items;
    nothing is deleted; preflight refuses it; the content test retention_manifest fails it;
  - the clean manifest (hours 11 and 14) is deleted normally; the protected hours stay;
  - delete_item re-checks: a cell added after the preflight refuses the item about to be deleted, nothing of it deleted;
  - fail closed: an unreadable do-not-delete row or a missing data_decisions.csv refuses the plan and the delete;
  - on the cluster: the real data_decisions.csv (read only) has exactly 17 do-not-delete cells (255 rows): 03h-12h of
    2025-01-04 and 13h-19h of 2025-05-04.
"""
import contextlib
import csv
import glob
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exopipe.phase3 import cli  # noqa: E402
from exopipe.phase3.config import CFG  # noqa: E402
from exopipe.phase3.core import cells as C  # noqa: E402
from exopipe.phase3.core import exclusions as E  # noqa: E402
from exopipe.phase3.core import validate  # noqa: E402
from exopipe.phase3.core.record import Refusal  # noqa: E402
from exopipe.phase3.lib.retention import plan as P  # noqa: E402
from exopipe.phase3.lib.retention import protect as D  # noqa: E402
from exopipe.phase3.lib.retention import table as T  # noqa: E402

FAILS = []
REAL_DD = '/lustre/gh/main/clv15/data_decisions.csv'
HEADER = ['lst', 'subband', 'date', 'verdict', 'reason', 'score_thermal_noise', 'score_flux_scale', 'score_ionosphere']
SB15 = ['18MHz', '23MHz', '27MHz', '32MHz', '36MHz', '41MHz', '46MHz', '50MHz', '55MHz', '59MHz', '64MHz', '69MHz', '73MHz',
        '78MHz', '82MHz']
DATE = '2025-05-04'
SUBBANDS = ('13MHz', '18MHz', '46MHz', '90MHz')      # 13 MHz: no row ever names it; 90 MHz: a subband without a row
HOURS = (11, 12, 13, 14)
VORTEX = 'ionospheric vortex event; do not delete (Gregg 2026-06-28)'


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
        return code in str(e), str(e)
    return False, 'no refusal'


def write_dd(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(HEADER)
        for r in rows:
            w.writerow(list(r) + [''] * (len(HEADER) - len(r)))
    return path


def vortex_rows(lst, date):
    """One do-not-delete row per imaged subband (the real list's pattern: no 13 MHz row)."""
    return [(lst, sb, date, 'exclude', VORTEX) for sb in SB15]


def go(argv):
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            rc = cli.main(['retention'] + argv + ['--cores', 'none'])
    finally:
        CFG.reset()
    return rc, err.getvalue()


def utc(text):
    return T.parse_utc(text)


def test_epoch_utc():
    root = logging.getLogger()
    before = (list(root.handlers), root.level)
    fresh = 'exopipe.phase1.pipeline_controller' not in sys.modules
    t0, t1 = C.epoch_utc('19h', DATE)
    check('epoch_utc: LST 19h of 2025-05-04 is 12:02:37-13:02:27 UTC on 2025-05-04 (Phase 1\'s conversion; within 1 s)',
          abs(t0 - utc('2025-05-04T12:02:37Z')) < 1 and abs(t1 - utc('2025-05-04T13:02:27Z')) < 1, (T.iso(t0), T.iso(t1)))
    check('epoch_utc: the import of Phase 1\'s controller leaves the root logger as it was',
          (list(root.handlers), root.level) == before, (fresh, root.handlers, root.level))
    for bad in (('24h', DATE), ('7h', DATE), ('19h', '2025-5-4'), ('19h', '2025-13-04')):
        try:
            C.epoch_utc(*bad)
            ok = False
        except ValueError:
            ok = True
        check('epoch_utc: %s %s is a ValueError' % bad, ok)


def test_exclusions(d):
    p = write_dd(os.path.join(d, 'dd_read.csv'), vortex_rows('19h', DATE) + [
        ('03h', '18MHz', '2025-01-04', 'keep', 'readmitted; Do-Not-Delete'),               # any verdict; case and hyphens
        ('05h', 'Red', '2025-01-04', 'exclude', '[PERBAND] vortex; DO NOT  DELETE'),       # a band label; a tag
        ('06h', '82MHz', '2025-01-04', 'exclude', '[SNAPSHOT-TIERS-ONLY] do not delete'),  # skipped by the default scope
        ('07h', '46MHz', '2025-01-04', 'exclude', 'frames deleted by the sunlit run'),     # not do not delete
        ('08h', '46MHz', '2025-01-04', 'exclude', 'do delete these'),
        ('99x', '46MHz', '2025-01-04', 'exclude', 'an ordinary exclusion row; its LST is never read here')])
    cells, fp = E.do_not_delete_cells(p)
    want = {('19h', DATE), ('03h', '2025-01-04'), ('05h', '2025-01-04'), ('06h', '2025-01-04')}
    check('do_not_delete_cells: every do-not-delete row, any verdict, tag, scope or subband label; nothing else',
          set(cells) == want and len(cells[('19h', DATE)]['lines']) == 15 and fp['n_do_not_delete_cells'] == 4,
          sorted(cells))
    check('... the cell keeps its rows\' subbands, reasons and file lines',
          cells[('19h', DATE)]['subbands'] == set(SB15) and cells[('19h', DATE)]['reasons'] == {VORTEX}
          and cells[('19h', DATE)]['lines'] == list(range(2, 17)), cells[('19h', DATE)])
    for what, row in (('an LST that is not 00h..23h', ('24h', '46MHz', DATE, 'exclude', 'do not delete')),
                      ('an LST without its zero', ('7h', '46MHz', DATE, 'exclude', 'do not delete')),
                      ('a date that is not YYYY-MM-DD', ('19h', '46MHz', '2025/05/04', 'exclude', 'do not delete')),
                      ('a date that is no date', ('19h', '46MHz', '2025-02-30', 'exclude', 'do not delete')),
                      ('a subband that is not a subband or band', ('19h', '99MHz', DATE, 'exclude', 'do not delete')),
                      ('an empty subband', ('19h', '', DATE, 'exclude', 'do not delete')),
                      ('the do-not-delete text in another column', ('19h', '46MHz', DATE, 'exclude', 'vortex',
                                                                     'do not delete'))):
        bad = write_dd(os.path.join(d, 'dd_bad.csv'), vortex_rows('19h', DATE) + [row])
        ok, why = refuses(lambda: E.do_not_delete_cells(bad), 'cannot be read')
        check('a do-not-delete row with %s refuses the do-not-delete read (fail closed), naming its line' % what,
              ok and 'line 17' in why, why)
        ex = E.Exclusions(bad)
        check('... and not the exclusion read of the co-adds (%d exclude rows)' % len(ex.rows), len(ex.rows) >= 15)
    bad = os.path.join(d, 'dd_extra.csv')
    with open(bad, 'w') as f:
        f.write(','.join(HEADER) + '\n19h,46MHz,%s,exclude,ionospheric vortex event, do not delete,,,\n' % DATE)
    ok, why = refuses(lambda: E.do_not_delete_cells(bad), 'more fields than the header')
    check('a do-not-delete row with more fields than the header (an unquoted comma) refuses', ok, why)
    ok, why = refuses(lambda: E.do_not_delete_cells(os.path.join(d, 'no_such.csv')), 'not found')
    check('a missing data_decisions.csv refuses the do-not-delete read (ruling C-09)', ok, why)


def test_cells_over():
    t0 = utc('2025-05-04T11:00:00Z')
    near = [dict(lst='x', date=DATE, start=t0 + 3600 + 5, end=t0 + 7200, subbands=[], reasons=[], lines=[2])]
    far = [dict(near[0], start=t0 + 3600 + 11)]
    check('cells_over: a cell starting 5 s after the hour is over it (the last 10-s set runs past the hour)',
          D.cells_over(t0, t0 + 3600, near) == near)
    check('cells_over: a cell starting 11 s after the hour is not (one integration, %.3f s)' % CFG.instrument.INTEGRATION_S,
          D.cells_over(t0, t0 + 3600, far) == [])


class World:
    def __init__(self, d):
        self.d, self.root, self.out = d, os.path.join(d, 'raw'), os.path.join(d, 'out')
        self.files = []
        for sb in SUBBANDS:
            for hh in HOURS:
                h = os.path.join(self.root, sb, DATE, '%02d' % hh)
                os.makedirs(h)
                for off in (0, 1800):
                    t = utc('%sT%02d:00:00Z' % (DATE, hh)) + off
                    p = os.path.join(h, '%s_%s_averaged.ms.tar' % (time.strftime('%Y%m%d_%H%M%S', time.gmtime(t)), sb))
                    with open(p, 'wb') as f:
                        f.write(b'v' * (10 + hh))
                    self.files.append(p)
        T.record_production(self.out, utc(DATE), utc(DATE) + 86400, '*', utc(DATE) + 15 * 3600, 'test night')

    def present(self):
        return sorted(p for p in self.files if os.path.exists(p))

    def item(self, sb, hh):
        return '%s/%s/%02d' % (sb, DATE, hh)


def write_ledger(path, sha):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\r\n')
        w.writerow(['date', 'fix_id', 'description', 'authorized_by', 'status'])
        for i, s in enumerate(sha):
            w.writerow(['2026-11-25', 'R-GREGG-TEST-GO-%d' % i, 'Gregg: go to delete the manifest with sha256 %s' % s[:16],
                        'Gregg', 'RULED - test'])


def test_plan_and_delete(d, Pt):
    W = World(os.path.join(d, 'w'))
    Pt.set(CFG.retention, 'RAW_ROOT', W.root)
    Pt.set(CFG.retention, 'ROOT', W.out)
    dd_dnd = write_dd(os.path.join(d, 'dd_dnd.csv'), vortex_rows('19h', DATE) + [
        ('17h', '46MHz', DATE, 'exclude', 'flux-scale anomaly')])                     # an exclusion, not do not delete
    dd_none = write_dd(os.path.join(d, 'dd_none.csv'), [('17h', '46MHz', DATE, 'exclude', 'flux-scale anomaly')])
    Pt.set(CFG.paths, 'DATA_DECISIONS', dd_dnd)
    before = W.present()

    listed = []
    real_list_item = P.list_item

    def spy(path):
        listed.append(path)
        return real_list_item(path)
    Pt.set(P, 'list_item', spy)
    pl = P.make_plan(W.root, W.out)
    Pt.saved.pop()
    P.list_item = real_list_item
    st = {it['item']: (cls['status'], cls['note']) for it, cls in pl.items}
    prot = sorted(k for k, v in st.items() if v[0] == 'protected')
    want = sorted(W.item(sb, hh) for sb in SUBBANDS for hh in (12, 13))
    check('plan: the hours 12 and 13 of 2025-05-04 (19h = 12:02:37-13:02:27 UTC) are protected in EVERY subband',
          prot == want, prot)
    check('the 13 MHz rule: 13 MHz (no row names it) and 90 MHz (a subband without a row) are protected with the rest',
          all(st[W.item(sb, hh)][0] == 'protected' for sb in ('13MHz', '90MHz') for hh in (12, 13)), st)
    check('a protected item names its cell (19h 2025-05-04, the UTC range, the rows\' lines and reason)',
          all(st[k][1].startswith('do not delete: 19h 2025-05-04 (2025-05-04T12:02:3') and 'lines 2-16' in st[k][1]
              and VORTEX in st[k][1] for k in prot), st[prot[0]][1])
    check('a protected item is never listed (no scandir of its directory)',
          not any(os.path.basename(p) in ('12', '13') for p in listed) and len(listed) == 8, listed)
    exp = sorted(k for k, v in st.items() if v[0] == 'expired')
    check('hours 11 and 14 are expired in every subband; the batch holds exactly them',
          exp == sorted(W.item(sb, hh) for sb in SUBBANDS for hh in (11, 14))
          and sorted(it['item'] for it, _c, _f in pl.batch) == exp, (exp, [it['item'] for it, _c, _f in pl.batch]))
    pcsv, man, mjson = P.write_plan(pl, W.out)
    side = json.load(open(mjson))
    dnd = side.get('do_not_delete') or {}
    check('the sidecar records the cells the plan honored and the data_decisions read (path, sha1)',
          [(c['lst'], c['date'], c['start_utc'][:16]) for c in dnd.get('cells', [])] == [('19h', DATE, '2025-05-04T12:02')]
          and dnd.get('data_decisions', {}).get('path') == dd_dnd and len(dnd['data_decisions'].get('sha1', '')) == 40, dnd)
    rows = list(csv.DictReader(open(pcsv)))
    check('the plan table lists the protected items with their status, not in the manifest',
          sorted(r['item'] for r in rows if r['status'] == 'protected') == want
          and not any(r['in_manifest'] for r in rows if r['status'] == 'protected'))
    for cls, path in (('retention_plan', pcsv), ('retention_manifest', man)):
        res = validate.run(cls, path)
        check('content test %s passes on the plan with protected items' % cls, res and all(r['passed'] for r in res), res)

    # a PLANTED manifest: planned while the list had no do-not-delete row, so it holds the hours 12 and 13
    Pt.set(CFG.paths, 'DATA_DECISIONS', dd_none)
    pl2 = P.make_plan(W.root, W.out)
    _p2, man2, _j2 = P.write_plan(pl2, W.out, when=time.time() - 3600)    # another stamp than the first manifest's
    Pt.saved.pop()
    CFG.paths.DATA_DECISIONS = dd_dnd
    planted = sorted(it['item'] for it, _c, _f in pl2.batch)
    assert set(want) <= set(planted) and man2 != man, planted
    sha2 = P.sha256_file(man2)
    led = os.path.join(d, 'fix_ledger.csv')
    write_ledger(led, [sha2, P.sha256_file(man)])
    Pt.set(CFG.paths, 'FIX_LEDGER', led)
    root_real = os.path.realpath(W.root)
    assert os.path.realpath(CFG.retention.RAW_ROOT) == root_real and root_real.startswith(os.path.realpath(d) + os.sep)
    check('the planted manifest holds the protected hours (%d items, 8 of them protected)' % len(planted),
          len(planted) == 16)
    for what, extra in (('with Gregg\'s go for that manifest', ['--ruling-for', 'retention.DELETE=R-GREGG-TEST-GO-0']),
                        ('with Gregg\'s go, --dry-run', ['--ruling-for', 'retention.DELETE=R-GREGG-TEST-GO-0', '--dry-run']),
                        ('without a ruling', [])):
        rc, err = go(['delete', '--out', W.out, '--manifest', man2] + extra)
        check('delete of the planted manifest %s: exit 2, RT-DO-NOT-DELETE naming the cell and the items, NOTHING deleted'
              % what, rc == 2 and 'RT-DO-NOT-DELETE' in err and '19h 2025-05-04' in err and '13MHz/2025-05-04/12' in err
              and 'holds 8 item(s)' in err and W.present() == before, (rc, err[-900:]))
    ok, why = refuses(lambda: P.preflight(man2, W.root, W.out), 'RT-DO-NOT-DELETE')
    check('preflight of the planted manifest: RT-DO-NOT-DELETE', ok, why)
    res = validate.run('retention_manifest', man2)
    check('content test retention_manifest fails the planted manifest', res and not all(r['passed'] for r in res), res)
    Pt.set(CFG.paths, 'DATA_DECISIONS', dd_none)
    rc, err = go(['delete', '--out', W.out, '--manifest', man2, '--ruling-for', 'retention.DELETE=R-GREGG-TEST-GO-0',
                  '--dry-run'])
    Pt.saved.pop()
    CFG.paths.DATA_DECISIONS = dd_dnd
    check('... the same delete with a list without the do-not-delete rows passes the preflight (--dry-run): the cells decide',
          rc == 0 and 'the manifest holds' in err, (rc, err[-600:]))

    # fail closed
    bad = write_dd(os.path.join(d, 'dd_badrow.csv'), vortex_rows('19h', DATE) + [('19h', '', DATE, 'exclude',
                                                                                    'do not delete')])
    for what, path in (('an unreadable do-not-delete row', bad), ('no data_decisions.csv', os.path.join(d, 'none.csv'))):
        Pt.set(CFG.paths, 'DATA_DECISIONS', path)
        ok, why = refuses(lambda: P.make_plan(W.root, W.out), 'RT-DO-NOT-DELETE')
        rc, err = go(['delete', '--out', W.out, '--manifest', man, '--ruling-for', 'retention.DELETE=R-GREGG-TEST-GO-1'])
        Pt.saved.pop()
        CFG.paths.DATA_DECISIONS = dd_dnd
        check('fail closed (%s): the plan refuses RT-DO-NOT-DELETE' % what, ok, why)
        check('fail closed (%s): the delete refuses RT-DO-NOT-DELETE, nothing deleted' % what,
              rc == 2 and 'RT-DO-NOT-DELETE' in err and W.present() == before, (rc, err[-600:]))

    # delete_item re-checks each item: a cell added after the preflight (18h = about 11:03-12:03 UTC covers hour 11)
    sd, items = P.preflight(man, W.root, W.out)
    dd_late = write_dd(os.path.join(d, 'dd_late.csv'), vortex_rows('19h', DATE) + vortex_rows('18h', DATE))
    Pt.set(CFG.paths, 'DATA_DECISIONS', dd_late)
    key = W.item('46MHz', 11)
    ok, why = refuses(lambda: P.delete_item(items[key], os.path.join(d, 'applied_late.csv')), 'RT-DO-NOT-DELETE')
    Pt.saved.pop()
    CFG.paths.DATA_DECISIONS = dd_dnd
    check('delete_item: a do-not-delete cell added after the preflight refuses the item about to be deleted, nothing of it '
          'deleted, no applied row', ok and W.present() == before and not os.path.exists(os.path.join(d, 'applied_late.csv')),
          why)

    # the clean manifest is deleted normally; the protected hours stay
    rc, err = go(['delete', '--out', W.out, '--manifest', man, '--ruling-for', 'retention.DELETE=R-GREGG-TEST-GO-1'])
    gone = sorted(set(before) - set(W.present()))
    want_gone = sorted(p for p in before if os.path.basename(os.path.dirname(p)) in ('11', '14'))
    check('the clean manifest (hours 11 and 14): exit 0, exactly its 16 files deleted, the protected hours 12 and 13 kept in '
          'every subband', rc == 0 and gone == want_gone and len(gone) == 16
          and all(os.path.isdir(os.path.join(W.root, sb, DATE, hh)) for sb in SUBBANDS for hh in ('12', '13')),
          (rc, err[-600:], len(gone)))
    app = glob.glob(os.path.join(W.out, 'manifests', 'APPLIED_RETENTION_*.csv'))
    res = validate.run('retention_applied', app[0]) if len(app) == 1 else []
    check('retention_applied passes on the clean delete', res and all(r['passed'] for r in res), (app, res))


def test_real_list():
    if not os.path.exists(REAL_DD):
        print('SKIP the real data_decisions.csv is not here (%s): the 17-cell check runs on the cluster' % REAL_DD, flush=True)
        return
    cells, fp = E.do_not_delete_cells(REAL_DD)
    want = {('%02dh' % h, '2025-01-04') for h in range(3, 13)} | {('%02dh' % h, '2025-05-04') for h in range(13, 20)}
    n_rows = sum(len(c['lines']) for c in cells.values())
    check('the real data_decisions.csv (read only): exactly 17 do-not-delete cells, 03h-12h of 2025-01-04 and 13h-19h of '
          '2025-05-04, %d rows' % n_rows, set(cells) == want and len(cells) == 17 and fp['n_do_not_delete_cells'] == 17,
          sorted(cells))
    check('... 255 do-not-delete rows, 15 per cell (18-82 MHz; no 13 MHz row)', n_rows == 255
          and all(len(c['lines']) == 15 and '13MHz' not in c['subbands'] for c in cells.values()), n_rows)
    rng = {(c['lst'], c['date']): c for c in D.ranges(REAL_DD)[0]}
    c = rng[('19h', '2025-05-04')]
    check('... 19h of 2025-05-04 is 12:02:37-13:02:27 UTC on 2025-05-04 (inside the October sunlit deletion from 11:54 UTC)',
          T.iso(c['start'])[:19] in ('2025-05-04T12:02:36', '2025-05-04T12:02:37')
          and T.iso(c['end'])[:16] == '2025-05-04T13:02', (T.iso(c['start']), T.iso(c['end'])))


def main():
    base = os.environ.get('EXOPIPE3_TEST_SCRATCH') or None
    d = tempfile.mkdtemp(prefix='test_retention_dnd_', dir=base)
    Pt = Patch()
    try:
        test_epoch_utc()
        test_exclusions(d)
        test_cells_over()
        test_plan_and_delete(d, Pt)
        test_real_list()
    finally:
        Pt.undo()
        CFG.reset()
        shutil.rmtree(d, ignore_errors=True)
    print('%d check(s) failed: %s' % (len(FAILS), FAILS) if FAILS else 'all checks passed', flush=True)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
