#!/usr/bin/env python3
"""TESS orbit table refresh: a step of its own (plan/CUTOVER_READINESS.md, the TESS item (b); 2026-11-17).

Production's reader (exopipe_ops/tess/tess_coverage.py:57-64) fetched TESS_orbit_times.csv from tess.mit.edu and wrote it BESIDE
ITSELF whenever its copy was older than 7 days. That makes the reader write into a code tree (a frozen tree's data symlink would
become a file), and a failed fetch went unnoticed: the table stopped at 2026-12-03 after the certificate failures of 2026-11-13/15.
In v2 the table is a data input that this step refreshes; the per-hour driver schedules it daily (config driver.PERIODIC
'tess-orbits', core/driver.py). Readers then get --no-refresh once the driver's loop runs, at cutover (config
pages_deep.TESS_REFRESH_IN_READER).

A fetched table replaces the current one only when its content passes these checks (otherwise the current table stays):
  - it is UTF-8 CSV text; the header is exactly Sector,Orbit,Start of Orbit,End of Orbit; every line other than a blank or a
    '#' comment line (the trailing '# Created ...') is a sector, orbit, start, end row; there is at least one;
  - every row (a segment: from sector 56 on, each orbit is listed as two segments) starts before it ends; sectors and orbits
    do not decrease down the table and segments do not overlap;
  - every segment of the current table that has ENDED (end < now) is in the new table unchanged (sector, orbit, start, end):
    past segments do not change, so a change there is refused for a person to look at;
  - the new table's latest end is not earlier than the current table's.
A table that passes is installed through a temporary file beside the current one, placed (core/placement: the current table's
group, mode and ACL, or the directory's rule for a new one); a temporary file a killed earlier install left (not modified for an
hour) is removed first (placement.sweep). A write or rename that fails is recorded as 'install failed: ...'; a placement refusal
(the temporary file cannot be given those attributes) as 'refused: install not placed: ...'. Either way the temporary file is
removed and the current table stays.
Complete when, on content: the installed table's latest segment end is at least AHEAD_DAYS (14) days after now. That holds even
when the fetch failed or the install failed or was refused (the reason is in the record) and the current table still reaches
that far. Otherwise the step refuses, with exit status 2. Exit status 1 means the arguments or the current table are unusable.
'unchanged' and every 'refused: ...' (the checks', and a placement refusal's) set the table's modification time to now
("checked now"; content identity is its sha1, so nothing re-runs): while the reader still refreshes on its own
(pages_deep.TESS_REFRESH_IN_READER True, tess_coverage.py:57-64, older than 7 days), it then does not re-fetch and install, with
its weaker check, a table this step refused or could not place.
Once readers run with --no-refresh, tier_deep's TESS stage reads the table through stale_for_reader(): a table that does not reach
now (the refresh has stopped) is unusable for coverage, with the reason recorded.

Run: python -m exopipe.phase3.lib.pages.tess_orbits [--table T]     (default: <paths.OPS_ROOT>/tess/TESS_orbit_times.csv, the
table beside the reader tess_coverage.py in the ops tree)
It prints one JSON record (fetch result, checks, rows, latest orbit end, sha256 before and after) and writes nothing else.

THE READER'S CALL (reader_args, coverage_counts): one policy for every caller of tess_coverage.py - the deep pages' TESS stage
(lib/pages/tier_deep.stage_tess) and the serves' TESS steps (lib/serve/tess.py) - under one interpreter key
(pages_deep.TESS_PYTHON).
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
import re
import sys
import urllib.request

from exopipe.phase3.core import placement
from exopipe.phase3.config.steps.pages import pages_deep as _PD

HEADER = ['Sector', 'Orbit', 'Start of Orbit', 'End of Orbit']
TIME_FMT = '%Y-%m-%d %H:%M:%S'
URL = _PD.TESS_ORBIT_URL                 # config pages_deep (one source; tess_coverage.py's URL)
AHEAD_DAYS = _PD.TESS_ORBIT_AHEAD_DAYS   # days the installed table must reach past now (config pages_deep)
FETCH_TIMEOUT_S = 30     # s; as the reader's fetch (tess_coverage.py:61)


class TableError(ValueError):
    """The table's content is not a usable orbit table."""


def parse(data):
    """Orbit rows of a table's bytes: [(sector, orbit, start, end)] in file order. Blank lines and '#' comment lines (the
    trailing '# Created ...' line, which the reader skips, tess_coverage.py:66) are skipped; any other line that is not a
    sector, orbit, start, end row makes the table unusable (a dropped segment is never silent)."""
    try:
        rows = list(csv.reader(io.StringIO(data.decode('utf-8'))))
    except (UnicodeDecodeError, csv.Error) as e:
        raise TableError(f'not a CSV text table ({e.__class__.__name__}: {e})'[:300])
    if not rows or [c.strip() for c in rows[0]] != HEADER:
        raise TableError(f'header is {rows[0] if rows else "missing"}, not {HEADER}'[:300])
    out = []
    for i, r in enumerate(rows[1:], 2):
        if not ''.join(r).strip() or r[0].lstrip().startswith('#'):
            continue
        if len(r) < 4 or not r[0].strip().isdigit() or not r[1].strip().isdigit():
            raise TableError(f'line {i}: {r} is not sector, orbit, start, end'[:300])
        try:
            t0 = datetime.datetime.strptime(r[2].strip(), TIME_FMT)
            t1 = datetime.datetime.strptime(r[3].strip(), TIME_FMT)
        except ValueError as e:
            raise TableError(f'line {i}: {e}')
        out.append((int(r[0]), int(r[1]), t0, t1))
    if not out:
        raise TableError('no orbit rows')
    return out


def last_end(rows):
    """The latest segment end of parsed rows (by content, not file order)."""
    return max(r[3] for r in rows)


def stale_for_reader(table, now=None):
    """Why a reader run with --no-refresh may not use ``table`` (tier_deep's TESS stage): unreadable or unusable, or it ends
    before now - every epoch is before now, so a table that reaches now covers them all; None = usable."""
    now = now or datetime.datetime.utcnow()
    try:
        end = last_end(parse(open(table, 'rb').read()))
    except (OSError, TableError) as e:
        return f'unusable ({e})'
    return None if end >= now else f'ends {end.strftime(TIME_FMT)}, before now ({now.strftime(TIME_FMT)})'


def reader_args(table, parity=False):
    """(args, why) for a call of tess_coverage.py. args: --stokes pages_deep.TESS_STOKES unless the tool's default 'I,V' (the
    content test tess_rows reads the same key), and --no-refresh unless the reader refreshes its own table
    (pages_deep.TESS_REFRESH_IN_READER True, production) in a run that is not a parity run (a parity run never writes the shared
    orbit table beside the tool). With --no-refresh the table must reach now: ``why`` is stale_for_reader's reason, else None."""
    from exopipe.phase3.lib.pages import common
    D = common.CFG().pages_deep
    args = [] if D.TESS_STOKES == 'I,V' else ['--stokes', D.TESS_STOKES]
    if D.TESS_REFRESH_IN_READER and not parity:
        return args, None
    return args + ['--no-refresh'], stale_for_reader(table)


# tess_coverage.py:286-290, the line it prints after writing <run>/tess_coverage.csv
COVERAGE_SUMMARY = re.compile(r'^(?P<out>\S+): (?P<n>\d+) (?:detections|forced events) \(frame times .*; written (?P<written>\d+) '
                              r'\(left out, page says not computed: (?P<left_out>\d+)\)\s*$', re.M)


def own_output(res, max_bytes=8 << 20):
    """The output of this call alone: core.legacy.run appends every call to its log after a '===== <utc> <argv>' line, so the
    text after the LAST such line of this call's argv (read from the end of the log, at most ``max_bytes``); without a log
    (or a stand-in result without one), its tail. None when the log does not hold this call's line."""
    log, argv = getattr(res, 'log', None), getattr(res, 'argv', None)
    if not log or not argv:
        return getattr(res, 'tail', '') or ''
    try:
        with open(log, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - max_bytes))
            text = f.read().decode('utf-8', 'replace')
    except OSError:
        return None
    mark = re.compile(r'^===== \S+ ' + re.escape(' '.join(str(a) for a in argv)) + r'$', re.M)
    last = None
    for last in mark.finditer(text):
        pass
    return text[last.end():] if last else None


def coverage_counts(res, out):
    """(written, left_out) from the summary line tess_coverage.py printed for ``out`` (its <run>/tess_coverage.csv) anywhere
    in this call's own output (own_output; the last such line), else None."""
    text = own_output(res)
    if not text:
        return None
    want = os.path.normpath(out)
    hits = [m for m in COVERAGE_SUMMARY.finditer(text) if os.path.normpath(m['out']) == want]
    return (int(hits[-1]['written']), int(hits[-1]['left_out'])) if hits else None


def problems(old, new, now):
    """Reasons the table `new` may not replace `old` (both from parse); empty = it may."""
    bad = []
    for k, (s, o, t0, t1) in enumerate(new):
        if not t0 < t1:
            bad.append(f'sector {s} orbit {o}: start {t0} is not before end {t1}')
        if k and not (s >= new[k - 1][0] and o >= new[k - 1][1] and t0 >= new[k - 1][3]):
            bad.append(f'sector {s} orbit {o} from {t0}: not after the row before it (order or overlap)')
    rows = set(new)
    for r in old:
        if r[3] < now and r not in rows:
            bad.append(f'ended segment sector {r[0]} orbit {r[1]} {r[2]} to {r[3]} is not in the fetched table')
    if old and last_end(new) < last_end(old):
        bad.append(f'fetched table ends {last_end(new)}, before the current table ({last_end(old)})')
    return bad


def _fetch(url, timeout):
    return urllib.request.urlopen(url, timeout=timeout).read()


def refresh(table, url=URL, now=None, ahead_days=AHEAD_DAYS, fetch=None):
    """Fetch, check and install the orbit table at `table`; returns the record (see the module docstring)."""
    now = now or datetime.datetime.utcnow()
    fetch = fetch or (lambda: _fetch(url, FETCH_TIMEOUT_S))
    rec = {'table': table, 'url': url, 'now_utc': now.strftime(TIME_FMT), 'ahead_days': ahead_days}
    cur = open(table, 'rb').read() if os.path.exists(table) else b''
    rec['sha256_before'] = hashlib.sha256(cur).hexdigest() if cur else None
    old = parse(cur) if cur else []          # an unusable current table is an error (exit 1), not silently replaced
    try:
        data = fetch()
    except Exception as e:
        data = None
        rec['fetch'] = f'failed: {e}'[:300]
    if data is not None:
        try:
            new = parse(data)
        except TableError as e:
            new = None
            rec['fetch'] = f'refused: {e}'[:300]
        if new is not None:
            bad = problems(old, new, now)
            if bad:
                rec['fetch'] = 'refused: ' + '; '.join(bad)[:600]
            elif data == cur:
                rec['fetch'] = 'unchanged'
            else:
                # placed (core/placement.atomic: the replaced table's group, mode and ACL, or the directory's rule for a new
                # one); on any failure the temporary file is removed and the current table stays, judged below
                try:
                    swept = placement.sweep([table])            # a killed earlier install's temporary file
                    if swept:
                        rec['swept'] = swept
                    with placement.atomic(table) as part, open(part, 'wb') as f:
                        f.write(data)
                except placement.PlacementRefusal as e:   # not placed: a refusal (time touched below), not a crash of main
                    rec['fetch'] = f'refused: install not placed: {e}'[:600]
                except OSError as e:
                    rec['fetch'] = f'install failed: {e}'[:300]
                else:
                    rec['fetch'] = 'installed'
                    rec['rows_before'] = len(old)
                    old, cur = new, data
    if cur and (rec['fetch'] == 'unchanged' or rec['fetch'].startswith('refused')):
        try:
            os.utime(table)                   # "checked now" (module docstring): the reader's own refresh does not undo this
            rec['mtime_touched'] = True
        except OSError as e:
            rec['mtime_touched'] = f'failed: {e}'[:200]
    rec['sha256_after'] = hashlib.sha256(cur).hexdigest() if cur else None
    rec['rows'] = len(old)
    rec['last_orbit_end_utc'] = last_end(old).strftime(TIME_FMT) if old else None
    need = now + datetime.timedelta(days=ahead_days)
    rec['complete'] = bool(old) and last_end(old) >= need
    if not rec['complete']:
        rec['refusal'] = (f'REFUSING: the orbit table ends {rec["last_orbit_end_utc"]}, before {need.strftime(TIME_FMT)} '
                          f'(now + {ahead_days} d); fetch: {rec["fetch"]}')
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--table', help='the orbit table the readers use (it is replaced in place, atomically); default '
                    '<paths.OPS_ROOT>/tess/TESS_orbit_times.csv, beside the reader tess_coverage.py')
    ap.add_argument('--url', default=URL)
    ap.add_argument('--ahead-days', type=float, default=AHEAD_DAYS)
    a = ap.parse_args(argv)
    if not a.table:
        from exopipe.phase3.config import CFG
        a.table = os.path.join(CFG.paths.OPS_ROOT, 'tess', 'TESS_orbit_times.csv')
    try:
        rec = refresh(a.table, a.url, ahead_days=a.ahead_days)
    except (TableError, OSError, UnicodeDecodeError) as e:
        print(json.dumps({'table': a.table, 'error': f'current table unusable: {e}'}))
        return 1
    print(json.dumps(rec))
    return 0 if rec['complete'] else 2


if __name__ == '__main__':
    sys.exit(main())
