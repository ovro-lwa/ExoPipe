"""TESS COVERAGE OF A SERVED RUN, before its pages start (Gregg 2026-10-03: "The web interface should record for each transient if
TESS observing was simultaneous"; FINDING-20261115-TESS-COVERAGE-NOT-COMPUTED-ON-10MIN-SNAPSHOT-FORCED-PAGES, production's producers
FIX-20261115-TESS-COVERAGE-PRODUCER-SNAPSHOT-FORCED-10MIN-WIDEBAND). The tools run in place through core.legacy.run, as production's
serves run them:
  snapshot  snapshot_serve.sh:183-189: tess/tess_coverage.py --run DST --epoch-from-cand-id on the review copy (working directory
            serve.SNAPSHOT_STEP_CWD, log DST/tess_coverage.log), after the page context and before the bright-source filter
            (lib/serve/snapshot_copy.tess_stage). The port also passes --proto-root paths.SNAPSHOT_FLEET_OUT, the tool's default
            path (the search's noise tables, which give a 10-s or 1-min detection its integration).
  sublst    sublst_cumulative.sh:192-200: tess/frame_times_tenmin.py --run B --image-root <the 10-minute tree B> [--reuse
            U/frame_times.csv when that file is not empty] (site PYTHON 'default'), then tess/tess_coverage.py --run B, on the new
            build after its filters and before the carry and the swap; log B/incremental.log (lib/serve/sublst_build.tess_coverage).
tess_coverage.py runs under the site PYTHON[serve.TESS_PYTHON] (the TESS environment, tess-point).

The file annotates the pages; it hides and removes nothing. A failure REFUSES the unit, so nothing is served or swapped in
(production: exit 4 / exit 1, the served product and its servers untouched) and no page says "coverage not computed" for want of
the file. The deep pages' TESS stage differs on purpose (lib/pages/tier_deep.stage_tess: no-input, pages_deep.TESS_IS_GATE False,
p3s3_pass.sh:62-63).

The orbit table (TESS_orbit_times.csv beside the tool): pages_deep.TESS_REFRESH_IN_READER True = production (the tool fetches it
again when it is older than 7 days). False = the daily refresh step (lib/pages/tess_orbits.py) keeps it: the tool gets --no-refresh,
and a table that does not reach now refuses (tess_orbits.stale_for_reader).

Outputs, each with the tool's own counts from its summary line, the last line it prints (no such line refuses):
  <run>/tess_coverage.csv  [tess_coverage]  written = its rows, left_out = detections without an epoch time or a position, or after
                           the orbit table's last orbit: the tool leaves them out and the page says "not computed"
                           (tess_coverage.py, 2026-11-15 and 2026-11-16)
  <run>/frame_times.csv    [tenmin_frame_times]  (10-minute) windows = its rows, needed = the windows of master_catalog.csv; a
                           window without an image or without DATE-OBS / DATE-END is left out, and its detections then are too
"""
import os
import re

from exopipe.phase3.lib.serve import scripts

# tess_coverage.py:286-290: its summary, the last line it prints
TESS_SUMMARY = re.compile(r'; written (\d+) \(left out, page says not computed: (\d+)\)\s*$')
# frame_times_tenmin.py:101-102: its summary, the last line it prints
WINDOWS_SUMMARY = re.compile(r'frame_times\.csv: (\d+) of (\d+) windows \(')


def _CFG():
    from exopipe.phase3.config import CFG
    return CFG


def orbit_table():
    """The orbit table the tool reads: TESS_orbit_times.csv beside it (tess_coverage.py ORBITS)."""
    return os.path.join(os.path.dirname(scripts.path('tess_coverage')), 'TESS_orbit_times.csv')


def reader_args(u):
    """[] while the tool refreshes its own orbit table (pages_deep.TESS_REFRESH_IN_READER True, production); else ['--no-refresh'],
    once the table is shown to reach now (a table that does not refuses the unit: epochs after its end would get no coverage)."""
    if _CFG().pages_deep.TESS_REFRESH_IN_READER:
        return []
    from exopipe.phase3.lib.pages import tess_orbits
    table = orbit_table()
    why = tess_orbits.stale_for_reader(table)
    if why:
        u.refuse(f'TESS orbit table {table} {why} - the refresh step (python -m exopipe.phase3.lib.pages.tess_orbits) has not kept '
                 f'it current, and the reader runs with --no-refresh (pages_deep.TESS_REFRESH_IN_READER False)')
    u.check('serve.tess_orbit_table_reaches_now', True, None, f'{table} reaches now', path=table)
    return ['--no-refresh']


def summary(pattern, tail):
    """The match of ``pattern`` on the LAST line of the call's output, else None (the log is appended across calls, so an
    earlier line is never read as this call's summary)."""
    return pattern.search(scripts.last_line(tail))


def coverage(u, run_dir, args, log, cwd=None, after=''):
    """tess_coverage.py --run run_dir + args (+ reader_args) under the site PYTHON[serve.TESS_PYTHON]; its output appended to
    ``log``. A failure, or a summary line that cannot be read, refuses the unit (``after`` says what that leaves untouched).
    Records <run_dir>/tess_coverage.csv [tess_coverage] with written / left_out; returns (written, left_out)."""
    argv = ['--run', run_dir] + [str(a) for a in args] + reader_args(u)
    res = scripts.run('tess_coverage', argv, python=_CFG().serve.TESS_PYTHON, log=log, unit=u, cwd=cwd)
    if not res.ok:
        u.refuse(f'TESS coverage: {scripts.SCRIPTS["tess_coverage"][0]} failed on {run_dir} '
                 f'({res.reason or scripts.last_line(res.tail)}) - {after} (see {log})')
    m = summary(TESS_SUMMARY, res.tail)
    if m is None:
        u.refuse(f'TESS coverage: no summary line from {scripts.SCRIPTS["tess_coverage"][0]} on {run_dir} (its last line: '
                 f'{scripts.last_line(res.tail)[:200]!r}) - {after} (see {log})')
    written, left_out = int(m.group(1)), int(m.group(2))
    u.note(f'TESS coverage: {scripts.last_line(res.tail)[-300:]}')
    u.meta['tess_coverage'] = dict(written=written, left_out=left_out)
    u.add_output(os.path.join(run_dir, 'tess_coverage.csv'), 'tess_coverage', written=written, left_out=left_out)
    return written, left_out


def tenmin_frame_times(u, run_dir, image_root, reuse, log, after=''):
    """frame_times_tenmin.py --run run_dir --image-root image_root [--reuse reuse, when that file is not empty: production's
    `[ -s $U/frame_times.csv ]`] under the site PYTHON 'default'; its output appended to ``log``. A failure, or a summary line
    that cannot be read, refuses; so does a failed content test of the file, before the coverage runs. Records
    <run_dir>/frame_times.csv [tenmin_frame_times] with windows / needed."""
    argv = ['--run', run_dir, '--image-root', image_root]
    if reuse and os.path.isfile(reuse) and os.path.getsize(reuse) > 0:
        argv += ['--reuse', reuse]
    res = scripts.run('frame_times_tenmin', argv, log=log, unit=u)
    if not res.ok:
        u.refuse(f'TESS coverage: {scripts.SCRIPTS["frame_times_tenmin"][0]} failed on {run_dir} '
                 f'({res.reason or scripts.last_line(res.tail)}) - {after} (see {log})')
    m = summary(WINDOWS_SUMMARY, res.tail)
    if m is None:
        u.refuse(f'TESS coverage: no summary line from {scripts.SCRIPTS["frame_times_tenmin"][0]} on {run_dir} (its last line: '
                 f'{scripts.last_line(res.tail)[:200]!r}) - {after} (see {log})')
    windows, needed = int(m.group(1)), int(m.group(2))
    u.note(f'window times: {scripts.last_line(res.tail)[-300:]}')
    u.meta['frame_times'] = dict(windows=windows, needed=needed, reused='--reuse' in argv)
    out = u.add_output(os.path.join(run_dir, 'frame_times.csv'), 'tenmin_frame_times', windows=windows, needed=needed)
    bad = [t for t in u.failed_tests() if t.get('path') == out['path']]
    if bad:      # production's `&&`: the coverage never runs on window times that failed (here: their content test)
        u.refuse(f'TESS coverage: the window times {out["path"]} failed their content test ('
                 + '; '.join(f"{t['name']}: {t.get('detail') or t.get('value')}" for t in bad[:3]) + f') - {after}')
    return windows, needed
