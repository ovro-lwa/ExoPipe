"""TESS COVERAGE OF A SERVED RUN, before its pages start (Gregg 2026-10-03: "The web interface should record for each transient if
TESS observing was simultaneous"; FINDING-20261115-TESS-COVERAGE-NOT-COMPUTED-ON-10MIN-SNAPSHOT-FORCED-PAGES, production's producers
FIX-20261115-TESS-COVERAGE-PRODUCER-SNAPSHOT-FORCED-10MIN-WIDEBAND). The tools run in place through core.legacy.run, as production's
serves run them:
  snapshot  snapshot_serve.sh (exopipe_ops 1a9f9c8): tess/tess_coverage.py --run DST --epoch-from-cand-id on the review copy
            (working directory serve.SNAPSHOT_STEP_CWD, log DST/tess_coverage.log), after the page context and before the
            bright-source filter (lib/serve/snapshot_copy.tess_stage).
  sublst    sublst_cumulative.sh:192-200: tess/frame_times_tenmin.py --run B --image-root <the 10-minute tree B> [--reuse
            U/frame_times.csv when that file is not empty] (site PYTHON 'default'), then tess/tess_coverage.py --run B, on the new
            build after its filters and before the carry and the swap; log B/incremental.log (lib/serve/sublst_build.tess_coverage).
tess_coverage.py runs under the site PYTHON[pages_deep.TESS_PYTHON] (the TESS environment, tess-point), the key the deep pages'
TESS stage uses.

The file annotates the pages; it hides and removes nothing. A failure REFUSES the unit, so nothing is served or swapped in
(production: exit 4 / exit 1, the served product and its servers untouched) and no page says "coverage not computed" for want of
the file. The deep pages' TESS stage differs on purpose (lib/pages/tier_deep.stage_tess: no-input, pages_deep.TESS_IS_GATE False,
p3s3_pass.sh:62-63).

The reader's arguments are the deep stage's (lib/pages/tess_orbits.reader_args): --stokes pages_deep.TESS_STOKES unless the
tool's default 'I,V'; --no-refresh when the daily refresh step keeps the orbit table (pages_deep.TESS_REFRESH_IN_READER False)
and in every parity run (a parity run never writes the shared table beside the tool) - the table must then reach now, else the
unit refuses before the tool runs. With TESS_REFRESH_IN_READER True (production) a production run lets the tool refresh it.

Outputs, each with the tool's own counts from the summary line it printed for that file, found anywhere in this call's own
output (core.legacy.run appends calls to one log: only the text after this call's '=====' line is read; no such line refuses):
  <run>/tess_coverage.csv  [tess_coverage]  written = its rows, left_out = detections without an epoch time or a position, or after
                           the orbit table's last orbit: the tool leaves them out and the page says "not computed"
                           (tess_coverage.py, 2026-11-15 and 2026-11-16)
  <run>/frame_times.csv    [tenmin_frame_times]  (10-minute) windows = its rows, needed = the windows of master_catalog.csv; a
                           window without an image or without DATE-OBS / DATE-END is left out, and its detections then are too.
                           A failed content test of this file stops the step before the coverage (production's `&&`).
"""
import os
import re

from exopipe.phase3.lib.serve import scripts

# frame_times_tenmin.py:101-102, the line it prints after writing <run>/frame_times.csv
WINDOWS_SUMMARY = re.compile(r'^(?P<out>\S+): (?P<windows>\d+) of (?P<needed>\d+) windows \(', re.M)


def _CFG():
    from exopipe.phase3.config import CFG
    return CFG


def orbit_table():
    """The orbit table the tool reads: TESS_orbit_times.csv beside it (tess_coverage.py ORBITS)."""
    return os.path.join(os.path.dirname(scripts.path('tess_coverage')), 'TESS_orbit_times.csv')


def reader_args(u, parity=False):
    """The reader's extra arguments (lib/pages/tess_orbits.reader_args, the deep TESS stage's policy); a table that must reach
    now and does not refuses the unit (epochs after its end would get no coverage)."""
    from exopipe.phase3.lib.pages import tess_orbits
    table = orbit_table()
    args, why = tess_orbits.reader_args(table, parity=parity)
    if why:
        u.refuse(f'TESS orbit table {table} {why} - the reader runs with --no-refresh ('
                 + ('a parity run never writes the shared table' if parity and _CFG().pages_deep.TESS_REFRESH_IN_READER else
                    'pages_deep.TESS_REFRESH_IN_READER False')
                 + ') and the refresh step (python -m exopipe.phase3.lib.pages.tess_orbits) has not kept it current')
    if '--no-refresh' in args:
        u.check('serve.tess_orbit_table_reaches_now', True, None, f'{table} reaches now', path=table)
    return args


def _own_summary(pattern, res, out):
    """The last match of ``pattern`` naming ``out`` anywhere in this call's own output (tess_orbits.own_output), else None."""
    from exopipe.phase3.lib.pages import tess_orbits
    text = tess_orbits.own_output(res)
    want = os.path.normpath(out)
    hits = [m for m in pattern.finditer(text or '') if os.path.normpath(m['out']) == want]
    return hits[-1] if hits else None


def coverage(u, run_dir, args, log, cwd=None, after='', parity=False):
    """tess_coverage.py --run run_dir + args (+ reader_args) under the site PYTHON[pages_deep.TESS_PYTHON]; its output appended to
    ``log``. A failure, or no summary line for run_dir/tess_coverage.csv in the call's own output, refuses the unit (``after``
    says what that leaves untouched). Records it [tess_coverage] with written / left_out; returns (written, left_out)."""
    from exopipe.phase3.lib.pages import tess_orbits
    argv = ['--run', run_dir] + [str(a) for a in args] + reader_args(u, parity=parity)
    res = scripts.run('tess_coverage', argv, python=_CFG().pages_deep.TESS_PYTHON, log=log, unit=u, cwd=cwd)
    script = scripts.SCRIPTS['tess_coverage'][0]
    if not res.ok:
        u.refuse(f'TESS coverage: {script} failed on {run_dir} ({res.reason or scripts.last_line(res.tail)}) - {after} (see {log})')
    out = os.path.join(run_dir, 'tess_coverage.csv')
    counts = tess_orbits.coverage_counts(res, out)
    if counts is None:
        u.refuse(f'TESS coverage: no summary line for {out} in the output of this call of {script} (its last line: '
                 f'{scripts.last_line(res.tail)[:200]!r}) - {after} (see {log})')
    written, left_out = counts
    u.note(f'TESS coverage: written {written}, left out {left_out} (the page says "not computed" for those)')
    u.meta['tess_coverage'] = dict(written=written, left_out=left_out)
    u.add_output(out, 'tess_coverage', written=written, left_out=left_out)
    return written, left_out


def tenmin_frame_times(u, run_dir, image_root, reuse, log, after=''):
    """frame_times_tenmin.py --run run_dir --image-root image_root [--reuse reuse, when that file is not empty: production's
    `[ -s $U/frame_times.csv ]`] under the site PYTHON 'default'; its output appended to ``log``. A failure, or no summary line
    for run_dir/frame_times.csv in the call's own output, refuses; so does a failed content test of the file, before the
    coverage runs. Records <run_dir>/frame_times.csv [tenmin_frame_times] with windows / needed."""
    argv = ['--run', run_dir, '--image-root', image_root]
    if reuse and os.path.isfile(reuse) and os.path.getsize(reuse) > 0:
        argv += ['--reuse', reuse]
    res = scripts.run('frame_times_tenmin', argv, log=log, unit=u)
    script = scripts.SCRIPTS['frame_times_tenmin'][0]
    if not res.ok:
        u.refuse(f'TESS coverage: {script} failed on {run_dir} ({res.reason or scripts.last_line(res.tail)}) - {after} (see {log})')
    path = os.path.join(run_dir, 'frame_times.csv')
    m = _own_summary(WINDOWS_SUMMARY, res, path)
    if m is None:
        u.refuse(f'TESS coverage: no summary line for {path} in the output of this call of {script} (its last line: '
                 f'{scripts.last_line(res.tail)[:200]!r}) - {after} (see {log})')
    windows, needed = int(m['windows']), int(m['needed'])
    u.note(f'window times: {windows} of {needed} windows' + (' (the served run\'s file reused)' if '--reuse' in argv else ''))
    u.meta['frame_times'] = dict(windows=windows, needed=needed, reused='--reuse' in argv)
    out = u.add_output(path, 'tenmin_frame_times', windows=windows, needed=needed)
    bad = [t for t in u.failed_tests() if t.get('path') == out['path']]
    if bad:      # production's `&&`: the coverage never runs on window times that failed (here: their content test)
        u.refuse(f'TESS coverage: the window times {out["path"]} failed their content test ('
                 + '; '.join(f"{t['name']}: {t.get('detail') or t.get('value')}" for t in bad[:3]) + f') - {after}')
    return windows, needed
