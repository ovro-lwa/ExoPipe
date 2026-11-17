#!/usr/bin/env python3
"""TESS orbit table refresh: a step of its own (plan/CUTOVER_READINESS.md, the TESS item (b); 2026-11-17).

Production's reader (exopipe_ops/tess/tess_coverage.py:57-64) fetched TESS_orbit_times.csv from tess.mit.edu and wrote it BESIDE
ITSELF whenever its copy was older than 7 days. That makes the reader write into a code tree (a frozen tree's data symlink would
become a file), and a failed fetch went unnoticed: the table stopped at 2026-12-03 after the certificate failures of 2026-11-13/15.
In v2 the table is a data input that this step refreshes. Readers then get --no-refresh once this step is scheduled daily
(config pages_deep.TESS_REFRESH_IN_READER).

A fetched table replaces the current one only when its content passes these checks (otherwise the current table stays):
  - the header is exactly Sector,Orbit,Start of Orbit,End of Orbit and there is at least one orbit row;
  - every row (a segment: from sector 56 on, each orbit is listed as two segments) starts before it ends; sectors and orbits
    do not decrease down the table and segments do not overlap;
  - every segment of the current table that has ENDED (end < now) is in the new table unchanged (sector, orbit, start, end):
    past segments do not change, so a change there is refused for a person to look at;
  - the new table's last end is not earlier than the current table's.
Complete when, on content: the installed table's last orbit end is at least AHEAD_DAYS (14) days after now. That holds even when
the fetch failed and the current table still reaches that far. Otherwise the step refuses, with exit status 2. Exit status 1 means
the arguments or the current table are unusable.

Run: python -m exopipe.phase3.lib.pages.tess_orbits --table /lustre/gh/main/exopipe_ops/tess/TESS_orbit_times.csv
It prints one JSON record (fetch result, checks, rows, last orbit end, sha256 before and after) and writes nothing else.
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
import sys
import urllib.request

HEADER = ['Sector', 'Orbit', 'Start of Orbit', 'End of Orbit']
TIME_FMT = '%Y-%m-%d %H:%M:%S'
URL = 'https://tess.mit.edu/public/files/TESS_orbit_times.csv'   # = config pages_deep.TESS_ORBIT_URL (tess_coverage.py URL)
AHEAD_DAYS = 14          # days the installed table must reach past now (config pages_deep.TESS_ORBIT_AHEAD_DAYS)
FETCH_TIMEOUT_S = 30     # s; as the reader's fetch (tess_coverage.py:61)


class TableError(ValueError):
    """The table's content is not a usable orbit table."""


def parse(data):
    """Orbit rows of a table's bytes: [(sector, orbit, start, end)] in file order. Rows whose Sector is not a number (the
    trailing '# Created ...' line) are skipped, as the reader skips them (tess_coverage.py:66)."""
    text = data.decode('utf-8')
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or [c.strip() for c in rows[0]] != HEADER:
        raise TableError(f'header is {rows[0] if rows else "missing"}, not {HEADER}')
    out = []
    for i, r in enumerate(rows[1:], 2):
        if not r or not r[0].strip().isdigit():
            continue
        if len(r) < 4 or not r[1].strip().isdigit():
            raise TableError(f'line {i}: {r} is not sector, orbit, start, end')
        try:
            t0 = datetime.datetime.strptime(r[2].strip(), TIME_FMT)
            t1 = datetime.datetime.strptime(r[3].strip(), TIME_FMT)
        except ValueError as e:
            raise TableError(f'line {i}: {e}')
        out.append((int(r[0]), int(r[1]), t0, t1))
    if not out:
        raise TableError('no orbit rows')
    return out


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
    if old and new[-1][3] < old[-1][3]:
        bad.append(f'fetched table ends {new[-1][3]}, before the current table ({old[-1][3]})')
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
                part = table + '.part'
                with open(part, 'wb') as f:
                    f.write(data)
                os.replace(part, table)
                rec['fetch'] = 'installed'
                rec['rows_before'] = len(old)
                old, cur = new, data
    rec['sha256_after'] = hashlib.sha256(cur).hexdigest() if cur else None
    rec['rows'] = len(old)
    rec['last_orbit_end_utc'] = old[-1][3].strftime(TIME_FMT) if old else None
    need = now + datetime.timedelta(days=ahead_days)
    rec['complete'] = bool(old) and old[-1][3] >= need
    if not rec['complete']:
        rec['refusal'] = (f'REFUSING: the orbit table ends {rec["last_orbit_end_utc"]}, before {need.strftime(TIME_FMT)} '
                          f'(now + {ahead_days} d); fetch: {rec["fetch"]}')
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--table', required=True, help='the orbit table the readers use (it is replaced in place, atomically)')
    ap.add_argument('--url', default=URL)
    ap.add_argument('--ahead-days', type=float, default=AHEAD_DAYS)
    a = ap.parse_args(argv)
    try:
        rec = refresh(a.table, a.url, ahead_days=a.ahead_days)
    except (TableError, OSError, UnicodeDecodeError) as e:
        print(json.dumps({'table': a.table, 'error': f'current table unusable: {e}'}))
        return 1
    print(json.dumps(rec))
    return 0 if rec['complete'] else 2


if __name__ == '__main__':
    sys.exit(main())
