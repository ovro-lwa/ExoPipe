#!/usr/bin/env python3
"""Tests of the TESS orbit table refresh step (exopipe/phase3/lib/pages/tess_orbits.py). Plain asserts; no network (the fetch is
injected). Scratch: a fresh tempfile directory (set TMPDIR to node-local scratch on the cluster), removed at the end.
    TMPDIR=/fast/gh/claude_tmp /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_tess_orbits.py
Checked: install of a longer table; 'unchanged'; refusals (changed ended segment, missing ended segment, shorter table, bad
header, overlap, no rows, an HTML page, bytes not UTF-8, a NUL byte, a future row with a blank sector) keep the current table
byte-identical; 'unchanged' and 'refused' set the mtime to now, a failed fetch does not; fetch failure complete only while the
current table reaches now + 14 days (exit 0 / 2), judged on the latest end by content; a failed install keeps the current table
and leaves no .part; an unusable current table is exit 1; the CLI record; the reader's check stale_for_reader; the two-segment
orbits of the real table (when the cluster copy is readable); the module takes URL and AHEAD_DAYS from config pages_deep;
tier_deep passes --no-refresh and checks the table only when TESS_REFRESH_IN_READER is False.
"""
import datetime
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from exopipe.phase3.lib.pages import tess_orbits as T   # noqa: E402

HEAD = 'Sector,Orbit,Start of Orbit,End of Orbit\n'
NOW = datetime.datetime(2026, 11, 17, 12, 0, 0)


def table(rows, trailer=True):
    s = HEAD + ''.join(f'{a},{b},{c},{d}\n' for a, b, c, d in rows)
    return (s + ('# Created 2026-11-16 by someone\n' if trailer else '')).encode()


R_OLD = [(97, 201, '2025-10-01 00:00:00', '2025-10-14 00:00:00'),
         (109, 229, '2026-11-01 00:00:00', '2026-11-08 00:00:00'),
         (109, 229, '2026-11-08 05:00:00', '2026-11-15 00:00:00'),
         (110, 230, '2026-11-15 05:00:00', '2026-11-22 00:00:00'),
         (110, 230, '2026-11-22 05:00:00', '2026-11-29 00:00:00')]
R_NEW = R_OLD[:3] + [(110, 230, '2026-11-15 05:00:00', '2026-11-22 03:00:00'),   # a future segment revised: allowed
                     (110, 230, '2026-11-22 05:00:00', '2026-11-29 00:00:00'),
                     (111, 231, '2026-11-29 05:00:00', '2026-12-06 00:00:00'),
                     (111, 231, '2026-12-06 05:00:00', '2026-12-13 00:00:00')]


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def run(d, cur, fetched, now=NOW):
    p = os.path.join(d, 'TESS_orbit_times.csv')
    if os.path.exists(p):
        os.remove(p)
    if cur is not None:
        open(p, 'wb').write(cur)
    before = sha(p) if cur is not None else None

    def fetch():
        if isinstance(fetched, Exception):
            raise fetched
        return fetched
    return T.refresh(p, now=now, fetch=fetch), p, before


def main():
    d = tempfile.mkdtemp(prefix='exopipe3_tests_tess_orbits_')
    try:
        old = table(R_OLD)
        # install: a longer table with an ended segment unchanged and a future one revised
        rec, p, _ = run(d, old, table(R_NEW))
        assert rec['fetch'] == 'installed' and rec['rows'] == 7 and rec['rows_before'] == 5, rec
        assert open(p, 'rb').read() == table(R_NEW) and not os.path.exists(p + '.part')
        assert rec['last_orbit_end_utc'] == '2026-12-13 00:00:00' and rec['complete'] is True, rec   # 12-13 >= 12-01
        # unchanged
        rec, p, b = run(d, table(R_NEW), table(R_NEW))
        assert rec['fetch'] == 'unchanged' and rec['complete'] and sha(p) == b, rec
        # refusals keep the current table byte-identical
        bad = {
            'changed ended segment': table([R_NEW[0], (109, 229, '2026-11-01 00:00:00', '2026-11-08 01:00:00')] + R_NEW[2:]),
            'missing ended segment': table(R_NEW[1:]),
            'shorter table': table(R_OLD[:4]),
            'bad header': b'Sector,Orbit,Start,End\n' + table(R_NEW)[len(HEAD):],
            'overlap': table(R_NEW[:5] + [(111, 231, '2026-11-28 23:00:00', '2026-12-06 00:00:00')] + R_NEW[6:]),
            'no rows': table([]),
            'html page': b'<html>Service unavailable</html>',
            'not UTF-8': b'\xff\xfe' + table(R_NEW),
            'NUL byte': table(R_NEW).replace(b'2026-12-06 05:00:00', b'2026-12-06 05:00\x0000'),
            'blank sector in a future row': table(R_NEW[:5] + [('', 231, '2026-11-29 05:00:00', '2026-12-06 00:00:00')]
                                                  + R_NEW[6:]),
        }
        for name, data in bad.items():
            rec, p, b = run(d, old, data)
            assert rec['fetch'].startswith('refused'), (name, rec)
            assert sha(p) == b and not os.path.exists(p + '.part'), name
            # the current table ends 11-29, before 12-01 = now + 14 d: the step refuses
            assert rec['complete'] is False and rec['refusal'].startswith('REFUSING'), (name, rec)
        # fetch failure: complete only while the current table reaches now + 14 d
        rec, p, b = run(d, table(R_NEW), OSError('certificate verify failed'))
        assert rec['fetch'].startswith('failed') and rec['complete'] is True and sha(p) == b, rec
        rec, p, b = run(d, old, OSError('certificate verify failed'))
        assert rec['complete'] is False and 'certificate' in rec['refusal'], rec
        # 'unchanged' and 'refused' set the table's mtime to now (the reader's own 7-day refresh does not re-fetch); a failed
        # fetch does not
        for fetched, touched in ((table(R_NEW), True), (table(R_OLD[:4]), True), (OSError('timed out'), False)):
            rec, p, b = run(d, table(R_NEW), fetched)
            os.utime(p, (1e9, 1e9))
            rec = T.refresh(p, now=NOW, fetch=(lambda f=fetched: (_ for _ in ()).throw(f)) if isinstance(fetched, Exception)
                            else (lambda f=fetched: f))
            assert (os.path.getmtime(p) > 1e9) is touched and rec.get('mtime_touched', False) is touched, (fetched, rec)
            assert sha(p) == b, rec
        # completeness is the latest end by content, not the last row in file order
        far = [(120, 250, '2030-01-01 00:00:00', '2030-01-10 00:00:00')]
        rec, p, b = run(d, table(R_OLD[:1] + far + R_OLD[1:]), OSError('certificate verify failed'))
        assert rec['complete'] is True and rec['last_orbit_end_utc'] == '2030-01-10 00:00:00', rec
        # an install that cannot write keeps the current table, leaves no .part, judges the current table (here: short, exit 2)
        rec, p, b = run(d, old, b'')
        os.chmod(d, 0o555)
        try:
            if not os.access(d, os.W_OK):            # (as root the directory stays writable: nothing to test)
                rec = T.refresh(p, now=NOW, fetch=lambda: table(R_NEW))
                assert rec['fetch'].startswith('install failed') and not os.path.exists(p + '.part'), rec
                assert sha(p) == b and rec['complete'] is False and rec['last_orbit_end_utc'] == '2026-11-29 00:00:00', rec
        finally:
            os.chmod(d, 0o755)
        # the reader's check (tier_deep, --no-refresh): the table must reach now
        open(p, 'wb').write(table(R_OLD))
        assert T.stale_for_reader(p, now=NOW) is None
        assert 'before now' in T.stale_for_reader(p, now=datetime.datetime(2026, 12, 1))
        assert T.stale_for_reader(os.path.join(d, 'missing.csv'), now=NOW).startswith('unusable')
        open(p, 'wb').write(b'garbage\n')
        assert T.stale_for_reader(p, now=NOW).startswith('unusable')
        # no current table: a good fetch is installed
        rec, p, _ = run(d, None, table(R_NEW))
        assert rec['fetch'] == 'installed' and rec['sha256_before'] is None and rec['complete'], rec
        # an unusable current table is an error, never silently replaced
        try:
            run(d, b'garbage\n', table(R_NEW))
            raise AssertionError('unusable current table accepted')
        except T.TableError:
            pass
        # CLI: exit status and one JSON record (the fetch goes to an unreachable URL: failure, current table judged)
        p = os.path.join(d, 'TESS_orbit_times.csv')
        far = [(120, 250, '2030-01-01 00:00:00', '2030-01-10 00:00:00')]
        for rows, want in ((R_OLD, 2), (R_OLD + far, 0)):
            open(p, 'wb').write(table(rows))
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = T.main(['--table', p, '--url', 'file:///nonexistent/TESS_orbit_times.csv'])
            out = json.loads(buf.getvalue())
            assert rc == want and out['fetch'].startswith('failed') and out['complete'] is (want == 0), (rc, out)
        open(p, 'wb').write(b'garbage\n')
        with redirect_stdout(io.StringIO()):
            assert T.main(['--table', p, '--url', 'file:///nonexistent']) == 1
        # the real table (two segments per orbit from sector 56): parses, passes against itself and a truncated copy
        real = '/lustre/gh/main/exopipe_ops/tess/TESS_orbit_times.csv'
        if os.path.exists(real):
            r = T.parse(open(real, 'rb').read())
            assert len(r) > 300 and T.problems(r, r, NOW) == [] and T.problems(r[:-3], r, NOW) == []
            assert T.problems(r, r[:-3], NOW), 'shorter real table accepted'
            print(f'real table: {len(r)} segments, last end {r[-1][3]}')
        # config mirrors the module; tier_deep's reader flag
        from exopipe.phase3.config.steps.pages import pages_deep
        assert pages_deep.TESS_ORBIT_URL == T.URL and pages_deep.TESS_ORBIT_AHEAD_DAYS == T.AHEAD_DAYS
        assert pages_deep.TESS_REFRESH_IN_READER is True   # production behavior until the refresh step is scheduled
        src = open(os.path.join(REPO, 'exopipe/phase3/lib/pages/tier_deep.py')).read()
        assert "+ ([] if D.TESS_REFRESH_IN_READER else ['--no-refresh'])" in src
        assert 'why = tess_orbits.stale_for_reader(table)' in src and 'if not D.TESS_REFRESH_IN_READER:' in src
        print('test_tess_orbits: PASS')
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    main()
