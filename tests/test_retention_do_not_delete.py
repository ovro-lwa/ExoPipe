#!/usr/bin/env python
"""The do-not-delete gate of the raw-data retention (lib/retention/protect.py; core/exclusions.do_not_delete_cells;
core/sidereal.lst_hour_ranges). Gregg marks cells "do not delete" in data_decisions.csv (e.g. "ionospheric vortex event; do
not delete (Gregg 2026-06-28)"); the October 2026 sunlit deletion, which did not read them, removed the cell LST 19h of
2025-05-04. Data free: a synthetic raw tree of small files (date 2025-05-04, UT hours 10-14, subbands 13, 18, 46 and 90 MHz),
synthetic data_decisions.csv files and a synthetic fix ledger (CRLF) in the test's scratch directory; every delete is of
that tree (asserted). On the cluster it also reads the real data_decisions.csv (read only).
Asserts:
  - core/sidereal: LST 19h of 2025-05-04 is 12:02:37-13:02:27 UTC; the 17 real do-not-delete cells match Phase 1's own
    conversion (LSTScheduler.parse_range, computed here when it runs, and the values it gave on lwacalim07 on 2026-11-25)
    within 2 s before the pad; an LST hour that recurs on one date (23h of 2025-01-03: 00:01:38 and 23:57:42 UTC) and one
    under way at 00:00 UTC give both ranges; the pad widens both sides; malformed input is a ValueError; the module imports
    neither astropy's time nor IERS code nor Phase 1, and leaves the root logger as it was;
  - core/exclusions: the do-not-delete rows are read whatever their verdict, scope tag or subband ('Do-Not-Delete' matches;
    plain 'keep' is not deletion wording, nor is 'keep' + 'raw' across two fields); a do-not-delete row whose LST, date or
    subband cannot be read refuses the do-not-delete read (fail closed) but not the exclusion read of the co-adds; deletion
    wording without "do not delete" (delet, keep raw, retain, preserv, don't) refuses, naming its line; a do-not-delete row
    with more fields than the header or the text outside the reason column protects its cell and is reported, unless its
    LST or date cannot be read (refuses); the reported lines are the file's physical lines (a blank line and multi-line
    quoted reasons above, LF and CRLF); a missing file refuses;
  - protect.cells_over: an item's data run one integration past its hour; any of a cell's ranges counts;
  - the plan: the hours 11 (by the 300-s pad), 12 and 13 of 2025-05-04 are 'protected' in EVERY subband - 13 MHz and 90 MHz
    included, which no row names - never listed, never in the manifest; hours 10 and 14 are expired and planned normally;
    the sidecar records the cells and their protected ranges; the content tests pass;
  - both occurrences of a recurring LST hour are protected (classify), with the pad reaching into the day before and the
    second occurrence into the day after;
  - a PLANTED manifest (planned while the list had no do-not-delete row) that holds the protected hours: the delete refuses
    the whole manifest (RT-DO-NOT-DELETE), with Gregg's go for that manifest and in --dry-run, naming the cell and the items;
    nothing is deleted; preflight refuses it;
  - the content test retention_manifest checks a manifest against the cells ITS OWN SIDECAR records: the planted manifest
    passes it, a sidecar recording a cell over an item fails it, a sidecar without cells fails it, and a row added to the
    live file later does not change its verdict;
  - delete_item re-checks with the preflight's read: no re-read while the file is unchanged (sha1), a row added in place
    after the preflight refuses the item about to be deleted (one re-read), nothing of it deleted; the whole clean delete
    reads the cells once;
  - fail closed: an unreadable do-not-delete row, deletion wording without "do not delete" or a missing data_decisions.csv
    refuses the plan and the delete;
  - the clean manifest (hours 10 and 14) is deleted normally; the protected hours stay;
  - on the cluster: the real data_decisions.csv (read only) has exactly 17 do-not-delete cells (255 rows, none irregular):
    03h-12h of 2025-01-04 and 13h-19h of 2025-05-04, one range each.
"""
import contextlib
import csv
import glob
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from exopipe.phase3 import cli  # noqa: E402
from exopipe.phase3.config import CFG  # noqa: E402
from exopipe.phase3.core import exclusions as E  # noqa: E402
from exopipe.phase3.core import sidereal as S  # noqa: E402
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
HOURS = (10, 11, 12, 13, 14)
VORTEX = 'ionospheric vortex event; do not delete (Gregg 2026-06-28)'
# Phase 1's own UTC range of each real do-not-delete cell (LSTScheduler.parse_range('<h>h-<h+1>h', date), seconds since 1970;
# computed on lwacalim07 on 2026-11-25 with astropy 5.2.2 and its IERS tables): the cross-check where Phase 1 cannot run
PHASE1_17 = {
    ('03h', '2025-01-04'): (1735963023.05, 1735966613.22), ('04h', '2025-01-04'): (1735966613.22, 1735970203.39),
    ('05h', '2025-01-04'): (1735970203.39, 1735973793.57), ('06h', '2025-01-04'): (1735973793.57, 1735977383.74),
    ('07h', '2025-01-04'): (1735977383.74, 1735980973.91), ('08h', '2025-01-04'): (1735980973.91, 1735984564.08),
    ('09h', '2025-01-04'): (1735984564.08, 1735988154.25), ('10h', '2025-01-04'): (1735988154.25, 1735991744.42),
    ('11h', '2025-01-04'): (1735991744.42, 1735995334.59), ('12h', '2025-01-04'): (1735995334.59, 1735998924.76),
    ('13h', '2025-05-04'): (1746338615.64, 1746342205.81), ('14h', '2025-05-04'): (1746342205.81, 1746345795.98),
    ('15h', '2025-05-04'): (1746345795.98, 1746349386.15), ('16h', '2025-05-04'): (1746349386.15, 1746352976.32),
    ('17h', '2025-05-04'): (1746352976.32, 1746356566.49), ('18h', '2025-05-04'): (1746356566.49, 1746360156.66),
    ('19h', '2025-05-04'): (1746360156.66, 1746363746.83)}
TWICE = ('23h', '2025-01-03')       # LST 23h starts 00:01:38 and 23:57:42 UTC on 2025-01-03


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


def day0(date):
    return float(utc(date))


def phase1():
    """Phase 1's conversion as a function (lst, date) -> (start, end), or (None, why) when it cannot run here. The root logger
    is restored after the import of Phase 1's controller (its import-time logging.basicConfig)."""
    import importlib
    root = logging.getLogger()
    before = (list(root.handlers), root.level)
    try:
        pc = importlib.import_module('exopipe.phase1.pipeline_controller')
        sch = pc.LSTScheduler()
    except Exception as e:
        return None, 'import: %s' % e
    finally:
        root.handlers[:] = before[0]
        root.setLevel(before[1])

    def conv(lst, date):
        h = int(lst[:2])
        try:
            t0, t1 = sch.parse_range('%dh-%dh' % (h, h + 1), date)
        except SystemExit:              # parse_range logs and exits on what it cannot compute (a command line's behavior)
            raise RuntimeError('Phase 1 cannot convert %s %s' % (lst, date))
        return float(t0.unix), float(t1.unix)
    try:
        conv('19h', DATE)
    except Exception as e:             # e.g. no IERS tables and no network here
        return None, str(e)
    return conv, ''


def test_sidereal():
    lon = CFG.instrument.SITE_LON_DEG
    rr = S.lst_hour_ranges(DATE, '19h')
    check('lst_hour_ranges: LST 19h of 2025-05-04 is 12:02:37-13:02:27 UTC (one range of one sidereal hour, 3590.17 s)',
          len(rr) == 1 and abs(rr[0][0] - utc('2025-05-04T12:02:37Z')) < 1 and abs(rr[0][1] - utc('2025-05-04T13:02:27Z')) < 1
          and abs(rr[0][1] - rr[0][0] - 3590.17) < 0.01, [(T.iso(a), T.iso(b)) for a, b in rr])
    check('lst_hour_ranges: the site longitude is the observatory\'s (CFG.instrument.SITE_LON_DEG = OVRO_LOC, -118.2817 deg)',
          abs(lon - -118.2816667) < 1e-6, lon)
    worst = max(min(max(abs(a - p0), abs(b - p1)) for a, b in S.lst_hour_ranges(date, lst, lon))
                for (lst, date), (p0, p1) in PHASE1_17.items())
    check('the 17 real cells: within 2 s of Phase 1\'s ranges recorded on lwacalim07 (worst %.3f s; UTC taken as UT1)' % worst,
          worst < 2.0, worst)
    conv, why = phase1()
    if conv is None:
        print('NOTE Phase 1\'s conversion cannot run here (%s): the recorded ranges above are the cross-check' % why, flush=True)
    else:
        live = max(min(max(abs(a - p0), abs(b - p1)) for a, b in S.lst_hour_ranges(date, lst, lon))
                   for (lst, date) in PHASE1_17 for p0, p1 in [conv(lst, date)])
        check('the 17 real cells: within 2 s of Phase 1\'s conversion computed now (worst %.3f s)' % live, live < 2.0, live)
    padded = S.lst_hour_ranges(DATE, '19h', pad_s=300)
    check('the pad widens both sides', padded == [(rr[0][0] - 300, rr[0][1] + 300)], padded)

    lst, date = TWICE
    t = S.lst_hour_ranges(date, lst, lon)
    d0 = day0(date)
    check('an LST hour that recurs on one date (23h of 2025-01-03): both ranges, about 00:02 and 23:58 UTC, each one '
          'sidereal hour, LMST 23h at each start',
          len(t) == 2 and 0 <= t[0][0] - d0 < 240 and 0 < d0 + 86400 - t[1][0] < 240
          and all(abs(b - a - S.SIDEREAL_HOUR_S) < 1e-6 for a, b in t)
          and all(abs(S.lmst_hours(a, lon) - 23) < 1e-6 for a, _b in t), [(T.iso(a), T.iso(b)) for a, b in t])
    if conv is not None:
        p0 = conv(lst, date)
        p1 = conv(lst, '2025-01-04')
        check('... the first is Phase 1\'s range of 23h 2025-01-03, the second Phase 1\'s range of 23h 2025-01-04 less one '
              'sidereal day (within 2 s)', max(abs(t[0][0] - p0[0]), abs(t[0][1] - p0[1])) < 2
              and abs(t[1][0] - (p1[0] - S.SIDEREAL_DAY_S)) < 2, (p0, p1, t))
    h = int(S.lmst_hours(d0, lon))
    u = S.lst_hour_ranges(date, h, lon)
    check('an LST hour under way at 00:00 UTC (%02dh of 2025-01-03): the range from the day before and the late one' % h,
          len(u) == 2 and u[0][0] < d0 < u[0][1] and d0 + 80000 < u[1][0] < d0 + 86400, [(T.iso(a), T.iso(b)) for a, b in u])
    if conv is not None:
        p = conv('%02dh' % h, date)
        check('... Phase 1\'s range of that cell is the late one (within 2 s)',
              max(abs(u[1][0] - p[0]), abs(u[1][1] - p[1])) < 2, (p, u))
    for bad in ((DATE, '24h'), (DATE, '7h'), (DATE, 24), (DATE, True), ('2025-5-4', '19h'), ('2025-13-04', '19h'),
                ('2025-02-30', '19h')):
        try:
            S.lst_hour_ranges(*bad, lon_deg=lon)
            ok = False
        except ValueError:
            ok = True
        check('lst_hour_ranges: date %r, LST %r is a ValueError' % bad, ok)
    try:
        S.lst_hour_ranges(DATE, '19h', lon, pad_s=-1)
        ok = False
    except ValueError:
        ok = True
    check('lst_hour_ranges: a negative pad is a ValueError', ok)
    code = ('import json, logging, sys\n'
            'from exopipe.phase3.core import sidereal as S\n'
            'r = S.lst_hour_ranges("2025-05-04", "19h", -118.2816667, pad_s=300)\n'
            'print(json.dumps(dict(r=r, mods=sorted(m for m in sys.modules if m.startswith(("astropy.time", '
            '"astropy.utils.iers", "astropy.coordinates", "exopipe.phase1"))), handlers=len(logging.getLogger().handlers))))\n')
    p = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
    try:
        got = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        got = {}
    check('core/sidereal imports no astropy time, coordinates or IERS code and no Phase 1 module, and configures no logging',
          got.get('mods') == [] and got.get('handlers') == 0 and len(got.get('r') or []) == 1, (got, p.stderr[-400:]))


def test_exclusions(d):
    p = write_dd(os.path.join(d, 'dd_read.csv'), vortex_rows('19h', DATE) + [
        ('03h', '18MHz', '2025-01-04', 'keep', 'readmitted; Do-Not-Delete'),               # any verdict; case and hyphens
        ('05h', 'Red', '2025-01-04', 'exclude', '[PERBAND] vortex; DO NOT  DELETE'),       # a band label; a tag
        ('06h', '82MHz', '2025-01-04', 'exclude', '[SNAPSHOT-TIERS-ONLY] do not delete'),  # skipped by the default scope
        ('09h', '46MHz', '2025-01-04', 'keep', 'readmitted; keep the night'),              # plain keep: neither
        ('10h', '46MHz', '2025-01-04', 'keep', 'raw sky looks fine'),                      # 'keep' + 'raw' in two fields
        ('99x', '46MHz', '2025-01-04', 'exclude', 'an ordinary exclusion row; its LST is never read here')])
    cells, fp = E.do_not_delete_cells(p)
    want = {('19h', DATE), ('03h', '2025-01-04'), ('05h', '2025-01-04'), ('06h', '2025-01-04')}
    check('do_not_delete_cells: every do-not-delete row, any verdict, tag, scope or subband label; plain "keep" is no '
          'deletion wording', set(cells) == want and len(cells[('19h', DATE)]['lines']) == 15
          and fp['n_do_not_delete_cells'] == 4 and fp['n_do_not_delete_irregular_rows'] == 0, sorted(cells))
    check('... the cell keeps its rows\' subbands, reasons and file lines',
          cells[('19h', DATE)]['subbands'] == set(SB15) and cells[('19h', DATE)]['reasons'] == {VORTEX}
          and cells[('19h', DATE)]['lines'] == list(range(2, 17)) and cells[('19h', DATE)]['irregular'] == [],
          cells[('19h', DATE)])
    for what, row in (('an LST that is not 00h..23h', ('24h', '46MHz', DATE, 'exclude', 'do not delete')),
                      ('an LST without its zero', ('7h', '46MHz', DATE, 'exclude', 'do not delete')),
                      ('a date that is not YYYY-MM-DD', ('19h', '46MHz', '2025/05/04', 'exclude', 'do not delete')),
                      ('a date that is no date', ('19h', '46MHz', '2025-02-30', 'exclude', 'do not delete')),
                      ('a subband that is not a subband or band', ('19h', '99MHz', DATE, 'exclude', 'do not delete')),
                      ('an empty subband', ('19h', '', DATE, 'exclude', 'do not delete')),
                      ('the text outside the reason column and a date that cannot be read',
                       ('19h', '46MHz', '2025/05/04', 'exclude', 'vortex', 'do not delete'))):
        bad = write_dd(os.path.join(d, 'dd_bad.csv'), vortex_rows('19h', DATE) + [row])
        ok, why = refuses(lambda: E.do_not_delete_cells(bad), 'cannot be read')
        check('a do-not-delete row with %s refuses the do-not-delete read (fail closed), naming its line' % what,
              ok and 'line 17:' in why, why)
        ex = E.Exclusions(bad)
        check('... and not the exclusion read of the co-adds (%d exclude rows)' % len(ex.rows), len(ex.rows) >= 15)
    for text in ('frames deleted by the sunlit run', 'please retain the raw data', 'Keep-Raw visibilities',
                 'preserve this night', "don't remove", 'don’t remove', 'deletion pending'):
        bad = write_dd(os.path.join(d, 'dd_word.csv'), vortex_rows('19h', DATE) + [('18h', '46MHz', DATE, 'exclude', text)])
        ok, why = refuses(lambda: E.do_not_delete_cells(bad), 'without the text "do not delete"')
        check('deletion wording without "do not delete" (%r) refuses the do-not-delete read, naming its line' % text,
              ok and 'line 17:' in why, why)
    bad = write_dd(os.path.join(d, 'dd_word.csv'), vortex_rows('19h', DATE) + [('18h', '46MHz', DATE, 'exclude', 'vortex',
                                                                                  'retain')])
    ok, why = refuses(lambda: E.do_not_delete_cells(bad), 'without the text')
    check('... in any column (a score column says "retain")', ok and 'line 17:' in why, why)

    # irregular rows whose LST, date and subband read: they protect their cells and are reported
    irr = os.path.join(d, 'dd_irregular.csv')
    with open(irr, 'w') as f:
        f.write(','.join(HEADER) + '\n19h,46MHz,%s,exclude,ionospheric vortex event, do not delete,,,\n' % DATE)
        csv.writer(f, lineterminator='\n').writerow(['18h', '46MHz', DATE, 'exclude', 'vortex', 'do not delete', '', ''])
    try:
        cells, fp = E.do_not_delete_cells(irr)
        got = {k: c['irregular'] for k, c in cells.items()}
    except Refusal as e:
        cells, fp, got = {}, {}, str(e)
    check('a do-not-delete row with more fields than the header (an unquoted comma) and one with the text outside the reason '
          'column, LST, date and subband readable: their cells are protected and the rows reported with their lines',
          got == {('19h', DATE): [(2, 'more fields than the header (an unquoted comma?)'),
                                  (2, 'the do-not-delete text is outside the reason column')],
                  ('18h', DATE): [(3, 'the do-not-delete text is outside the reason column')]}
          and fp.get('n_do_not_delete_irregular_rows') == 2, got)
    note = D.irregular_note(D.ranges(irr)[0]) if cells else ''
    check('... irregular_note names both rows', 'line 2 (19h %s)' % DATE in note and 'line 3 (18h %s)' % DATE in note, note)
    with open(irr, 'w') as f:
        f.write(','.join(HEADER) + '\n7h,46MHz,%s,exclude,ionospheric vortex event, do not delete,,,\n' % DATE)
    ok, why = refuses(lambda: E.do_not_delete_cells(irr), 'line 2: LST')
    check('a do-not-delete row with more fields than the header and an LST that cannot be read refuses', ok, why)

    # the reported line is the file's physical line: a multi-line quoted reason and blank lines above (LF and CRLF)
    for eol in ('\n', '\r\n'):
        body = eol.join([','.join(HEADER),
                         '04h,46MHz,2025-01-04,exclude,"flux scale', 'second line of the reason",,,',    # lines 2-3
                         '',                                                                          # line 4
                         '19h,46MHz,%s,exclude,do not delete,,,' % DATE,                              # line 5
                         '',                                                                          # line 6
                         '20h,46MHz,%s,exclude,"vortex;' % DATE, 'do not delete",,,']) + eol          # lines 7-8
        ml = os.path.join(d, 'dd_lines.csv')
        with open(ml, 'w', newline='') as f:
            f.write(body)
        cells, _fp = E.do_not_delete_cells(ml)
        ex = E.Exclusions(ml)
        check('file lines (%s): a do-not-delete row after a multi-line quoted reason and a blank line is line 5, one that '
              'starts a multi-line reason after another blank line is line 7' % ('CRLF' if eol == '\r\n' else 'LF'),
              cells[('19h', DATE)]['lines'] == [5] and cells[('20h', DATE)]['lines'] == [7]
              and ('04h', '46', '2025-01-04') in ex.rows, {k: c['lines'] for k, c in cells.items()})
        with open(ml, 'w', newline='') as f:
            f.write(body + '24h,46MHz,%s,exclude,do not delete,,,' % DATE + eol)                 # line 9
        ok, why = refuses(lambda: E.do_not_delete_cells(ml), 'line 9: LST')
        check('... a refused row below them is named as line 9 (%s)' % ('CRLF' if eol == '\r\n' else 'LF'), ok, why)
    ok, why = refuses(lambda: E.do_not_delete_cells(os.path.join(d, 'no_such.csv')), 'not found')
    check('a missing data_decisions.csv refuses the do-not-delete read (ruling C-09)', ok, why)


def test_cells_over():
    t0 = utc('2025-05-04T11:00:00Z')
    near = [dict(lst='x', date=DATE, ranges=[(t0 + 3600 + 5, t0 + 7200)], subbands=[], reasons=[], lines=[2])]
    far = [dict(near[0], ranges=[(t0 + 3600 + 11, t0 + 7200)])]
    two = [dict(near[0], ranges=[(t0 - 7200, t0 - 3600), (t0 + 1800, t0 + 1900)])]
    check('cells_over: a cell starting 5 s after the hour is over it (the last 10-s set runs past the hour)',
          D.cells_over(t0, t0 + 3600, near) == near)
    check('cells_over: a cell starting 11 s after the hour is not (one integration, %.3f s)' % CFG.instrument.INTEGRATION_S,
          D.cells_over(t0, t0 + 3600, far) == [])
    check('cells_over: any range of a cell counts (the second of two)', D.cells_over(t0, t0 + 3600, two) == two)


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


class Counter:
    """Counts the reads of the do-not-delete cells (protect.ranges)."""

    def __init__(self, Pt):
        self.n, self.real = 0, D.ranges
        Pt.set(D, 'ranges', self)

    def __call__(self, *a, **k):
        self.n += 1
        return self.real(*a, **k)


def doctored(man, name, edit):
    """A copy of a manifest under another name whose sidecar ``edit`` changed (the CSV, and so its sha256, unchanged)."""
    side = json.load(open(os.path.splitext(man)[0] + '.json'))
    new = os.path.join(os.path.dirname(man), name)
    shutil.copyfile(man, new)
    side['csv'] = os.path.basename(new)
    edit(side)
    with open(os.path.splitext(new)[0] + '.json', 'w') as f:
        json.dump(side, f)
    return new


def test_plan_and_delete(d, Pt):
    W = World(os.path.join(d, 'w'))
    Pt.set(CFG.retention, 'RAW_ROOT', W.root)
    Pt.set(CFG.retention, 'ROOT', W.out)
    dnd_rows = vortex_rows('19h', DATE) + [('17h', '46MHz', DATE, 'exclude', 'flux-scale anomaly')]
    dd_dnd = write_dd(os.path.join(d, 'dd_dnd.csv'), dnd_rows)                          # 17h: an exclusion only
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
    want = sorted(W.item(sb, hh) for sb in SUBBANDS for hh in (11, 12, 13))
    check('plan: the hours 11, 12 and 13 of 2025-05-04 (19h = 12:02:37-13:02:27 UTC, protected 11:57:37-13:07:27 with the '
          '300-s pad) are protected in EVERY subband', prot == want, prot)
    check('the 13 MHz rule: 13 MHz (no row names it) and 90 MHz (a subband without a row) are protected with the rest',
          all(st[W.item(sb, hh)][0] == 'protected' for sb in ('13MHz', '90MHz') for hh in (11, 12, 13)), st)
    check('a protected item names its cell (19h 2025-05-04, the protected UTC range, the rows\' lines and reason)',
          all(st[k][1].startswith('do not delete: 19h 2025-05-04 (protected 2025-05-04T11:57:3') and '-13:07:2' in st[k][1]
              and 'lines 2-16' in st[k][1] and VORTEX in st[k][1] for k in prot), st[prot[0]][1])
    check('a protected item is never listed (no scandir of its directory)',
          not any(os.path.basename(p) in ('11', '12', '13') for p in listed) and len(listed) == 8, listed)
    exp = sorted(k for k, v in st.items() if v[0] == 'expired')
    check('hours 10 and 14 are expired in every subband; the batch holds exactly them',
          exp == sorted(W.item(sb, hh) for sb in SUBBANDS for hh in (10, 14))
          and sorted(it['item'] for it, _c, _f in pl.batch) == exp, (exp, [it['item'] for it, _c, _f in pl.batch]))
    pcsv, man, mjson = P.write_plan(pl, W.out)
    side = json.load(open(mjson))
    dnd = side.get('do_not_delete') or {}
    rec = [(c['lst'], c['date'], [(r['start_utc'][:19], r['end_utc'][:16]) for r in c['ranges']], c['lines'])
           for c in dnd.get('cells', [])]
    check('the sidecar records the cells the plan honored (protected ranges, lines) and the data_decisions read (path, sha1, '
          'pad)', len(rec) == 1 and rec[0][:2] == ('19h', DATE) and rec[0][2][0][0][:18] == '2025-05-04T11:57:3'
          and rec[0][2][0][1] == '2025-05-04T13:07' and len(rec[0][2]) == 1 and rec[0][3] == list(range(2, 17))
          and dnd.get('data_decisions', {}).get('path') == dd_dnd and len(dnd['data_decisions'].get('sha1', '')) == 40
          and dnd['data_decisions'].get('pad_s') == 300.0, dnd)
    rows = list(csv.DictReader(open(pcsv)))
    check('the plan table lists the protected items with their status, not in the manifest',
          sorted(r['item'] for r in rows if r['status'] == 'protected') == want
          and not any(r['in_manifest'] for r in rows if r['status'] == 'protected'))
    for cls, path in (('retention_plan', pcsv), ('retention_manifest', man)):
        res = validate.run(cls, path)
        check('content test %s passes on the plan with protected items' % cls, res and all(r['passed'] for r in res), res)

    # a PLANTED manifest: planned while the list had no do-not-delete row, so it holds the hours 11, 12 and 13
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
    check('the planted manifest holds the protected hours (%d items, 12 of them protected)' % len(planted),
          len(planted) == 20)
    for what, extra in (('with Gregg\'s go for that manifest', ['--ruling-for', 'retention.DELETE=R-GREGG-TEST-GO-0']),
                        ('with Gregg\'s go, --dry-run', ['--ruling-for', 'retention.DELETE=R-GREGG-TEST-GO-0', '--dry-run'])):
        rc, err = go(['delete', '--out', W.out, '--manifest', man2] + extra)
        check('delete of the planted manifest %s: exit 2, RT-DO-NOT-DELETE naming the cell and the items, NOTHING deleted'
              % what, rc == 2 and 'RT-DO-NOT-DELETE' in err and '19h 2025-05-04' in err and '13MHz/2025-05-04/11' in err
              and 'holds 12 item(s)' in err and W.present() == before, (rc, err[-900:]))
    rc, err = go(['delete', '--out', W.out, '--manifest', man2])
    check('... without a ruling: refused (RT-RULING), nothing deleted', rc == 2 and 'RT-RULING' in err and W.present() == before,
          (rc, err[-400:]))
    ok, why = refuses(lambda: P.preflight(man2, W.root, W.out), 'RT-DO-NOT-DELETE')
    check('preflight of the planted manifest: RT-DO-NOT-DELETE', ok, why)

    # the content test checks a manifest against the cells ITS OWN SIDECAR records
    res = validate.run('retention_manifest', man2)
    check('content test retention_manifest passes the planted manifest: its sidecar records no cell (planned without the '
          'rows); the delete\'s gate, which reads the file at the delete, refuses it (above)',
          res and all(r['passed'] for r in res), res)
    cell = dnd['cells'][0]
    manx = doctored(man2, 'DRYRUN_RETENTION_19990101T000000Z.csv',
                    lambda s: s['do_not_delete']['cells'].append(cell))
    res = validate.run('retention_manifest', manx)
    check('... a sidecar that records a cell over its items fails it (RT-DO-NOT-DELETE)',
          res and not all(r['passed'] for r in res) and 'RT-DO-NOT-DELETE' in json.dumps(res), res)
    many = doctored(man2, 'DRYRUN_RETENTION_19990101T000001Z.csv', lambda s: s.pop('do_not_delete'))
    res = validate.run('retention_manifest', many)
    check('... a sidecar without do-not-delete cells (planned before the gate) fails it',
          res and not all(r['passed'] for r in res) and 'records no readable do-not-delete cells' in json.dumps(res), res)
    dd_late = write_dd(os.path.join(d, 'dd_late.csv'), dnd_rows + vortex_rows('20h', DATE))
    Pt.set(CFG.paths, 'DATA_DECISIONS', dd_late)
    res = validate.run('retention_manifest', man)
    Pt.saved.pop()
    CFG.paths.DATA_DECISIONS = dd_dnd
    check('... a row added to the live file later (20h covers hour 14) does not change its verdict on the clean manifest',
          res and all(r['passed'] for r in res), res)
    for m in (manx, many):
        os.remove(m)
        os.remove(os.path.splitext(m)[0] + '.json')

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
    word = write_dd(os.path.join(d, 'dd_wording.csv'), vortex_rows('19h', DATE) + [('20h', '46MHz', DATE, 'exclude',
                                                                                     'keep raw data for the vortex study')])
    for what, path in (('an unreadable do-not-delete row', bad), ('deletion wording without "do not delete"', word),
                       ('no data_decisions.csv', os.path.join(d, 'none.csv'))):
        Pt.set(CFG.paths, 'DATA_DECISIONS', path)
        ok, why = refuses(lambda: P.make_plan(W.root, W.out), 'RT-DO-NOT-DELETE')
        rc, err = go(['delete', '--out', W.out, '--manifest', man, '--ruling-for', 'retention.DELETE=R-GREGG-TEST-GO-1'])
        Pt.saved.pop()
        CFG.paths.DATA_DECISIONS = dd_dnd
        check('fail closed (%s): the plan refuses RT-DO-NOT-DELETE' % what, ok, why)
        check('fail closed (%s): the delete refuses RT-DO-NOT-DELETE, nothing deleted' % what,
              rc == 2 and 'RT-DO-NOT-DELETE' in err and W.present() == before, (rc, err[-600:]))

    # delete_item re-checks each item with the preflight's read: a re-read only when the file changed
    cnt = Counter(Pt)
    sd, items = P.preflight(man, W.root, W.out)
    key = W.item('46MHz', 14)
    g = items[key]['protect']
    n_pre = cnt.n
    g.current()
    g.current()
    check('the preflight reads the cells once; re-checks of the unchanged file (sha1) re-read nothing',
          n_pre == 1 and cnt.n == 1 and g.n_reads == 1, (n_pre, cnt.n, g.n_reads))
    write_dd(dd_dnd, dnd_rows + vortex_rows('20h', DATE))         # IN PLACE: 20h = 13:02-14:02 UTC covers hour 14
    ok, why = refuses(lambda: P.delete_item(items[key], os.path.join(d, 'applied_late.csv')), 'RT-DO-NOT-DELETE')
    check('delete_item: a do-not-delete row added in place after the preflight refuses the item about to be deleted (one '
          're-read), nothing of it deleted, no applied row', ok and '20h 2025-05-04' in why and g.n_reads == 2
          and cnt.n == 2 and W.present() == before and not os.path.exists(os.path.join(d, 'applied_late.csv')), (why, g.n_reads))
    write_dd(dd_dnd, dnd_rows)
    Pt.saved.pop()
    D.ranges = cnt.real

    # the clean manifest is deleted normally; the protected hours stay; the delete reads the cells once
    cnt = Counter(Pt)
    rc, err = go(['delete', '--out', W.out, '--manifest', man, '--ruling-for', 'retention.DELETE=R-GREGG-TEST-GO-1'])
    n_reads = cnt.n
    Pt.saved.pop()
    D.ranges = cnt.real
    gone = sorted(set(before) - set(W.present()))
    want_gone = sorted(p for p in before if os.path.basename(os.path.dirname(p)) in ('10', '14'))
    check('the clean manifest (hours 10 and 14): exit 0, exactly its 16 files deleted, the protected hours 11-13 kept in '
          'every subband', rc == 0 and gone == want_gone and len(gone) == 16
          and all(os.path.isdir(os.path.join(W.root, sb, DATE, hh)) for sb in SUBBANDS for hh in ('11', '12', '13')),
          (rc, err[-600:], len(gone)))
    check('... the delete of its 8 items read the cells once (per item: the sha1 of the file)', n_reads == 1, n_reads)
    app = glob.glob(os.path.join(W.out, 'manifests', 'APPLIED_RETENTION_*.csv'))
    res = validate.run('retention_applied', app[0]) if len(app) == 1 else []
    check('retention_applied passes on the clean delete', res and all(r['passed'] for r in res), (app, res))


def test_twice(d):
    """Both occurrences of an LST hour that recurs on its date are protected (classify)."""
    lst, date = TWICE
    path = write_dd(os.path.join(d, 'dd_twice.csv'), vortex_rows(lst, date))
    cells, _fp = D.ranges(path)
    st = {}
    for dt, hh in (('2025-01-02', 23), (date, 0), (date, 1), (date, 12), (date, 22), (date, 23), ('2025-01-04', 0),
                   ('2025-01-04', 2)):
        for sb in ('13MHz', '46MHz'):
            t0 = utc(dt) + hh * 3600
            it = dict(item='%s/%s/%02d' % (sb, dt, hh), subband=sb, date=dt, hour='%02d' % hh,
                      path=os.path.join(d, 'absent'), t0=t0, t1=t0 + 3600)
            st[(dt, hh, sb)] = P.classify(it, [], [], time.time(), protect=cells)['status']
    prot = sorted({(dt, hh) for (dt, hh, _sb), s in st.items() if s == 'protected'})
    check('23h of 2025-01-03 (00:01:38-01:01:28 and 23:57:42-00:57:32 UTC): both occurrences protected - hours 00, 01 and '
          '23 of the date, 00 of the next (the second runs past midnight) and 23 of the day before (the pad) - in 13 and 46 '
          'MHz; hours 12, 22 and 02 of the next day are not',
          prot == [('2025-01-02', 23), ('2025-01-03', 0), ('2025-01-03', 1), ('2025-01-03', 23), ('2025-01-04', 0)]
          and all(st[(dt, hh, '13MHz')] == st[(dt, hh, '46MHz')] for dt, hh, _sb in st), st)


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
    check('... 255 do-not-delete rows, 15 per cell (18-82 MHz; no 13 MHz row), none irregular', n_rows == 255
          and all(len(c['lines']) == 15 and '13MHz' not in c['subbands'] for c in cells.values())
          and fp['n_do_not_delete_irregular_rows'] == 0, n_rows)
    rng = {(c['lst'], c['date']): c for c in D.ranges(REAL_DD)[0]}
    c = rng[('19h', '2025-05-04')]
    check('... one range per cell; 19h of 2025-05-04 is protected 11:57:37-13:07:27 UTC (12:02:37-13:02:27 and the 300-s '
          'pad; inside the October sunlit deletion from 11:54 UTC)', all(len(x['ranges']) == 1 for x in rng.values())
          and T.iso(c['ranges'][0][0])[:19] in ('2025-05-04T11:57:36', '2025-05-04T11:57:37')
          and T.iso(c['ranges'][0][1])[:16] == '2025-05-04T13:07', [(T.iso(a), T.iso(b)) for a, b in c['ranges']])


def main():
    base = os.environ.get('EXOPIPE3_TEST_SCRATCH') or None
    d = tempfile.mkdtemp(prefix='test_retention_dnd_', dir=base)
    Pt = Patch()
    try:
        test_sidereal()
        test_exclusions(d)
        test_cells_over()
        test_plan_and_delete(d, Pt)
        test_twice(d)
        test_real_list()
    finally:
        Pt.undo()
        CFG.reset()
        shutil.rmtree(d, ignore_errors=True)
    print('%d check(s) failed: %s' % (len(FAILS), FAILS) if FAILS else 'all checks passed', flush=True)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
