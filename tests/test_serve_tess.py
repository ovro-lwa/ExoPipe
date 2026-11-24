#!/usr/bin/env python3
"""Data-free tests of the serve's TESS coverage steps (lib/serve/tess.py): the snapshot review copy's stage
(snapshot_copy.tess_stage, snapshot_serve.sh:183-189) and the 10-minute new build's step (sublst_build.tess_coverage,
sublst_cumulative.sh:192-200), with the content tests tess_coverage (with the tool's own counts) and tenmin_frame_times
(lib/pages/tests.py).

The tools (tess/tess_coverage.py, tess/frame_times_tenmin.py, production code run in place) are replaced by stand-ins that write
what the tools write and print their summary lines (one of them appending to the log as core.legacy.run does); the production fragments and the tools' presence are checked in
tests/test_serve.py (test_snapshot_steps, test_sublst_tess), which reads the cluster's production tree.

Run: python tests/test_serve_tess.py [test names]   (exit 0 = all passed, 1 = a failure, 2 = imports missing)
"""
import csv
import datetime
import inspect
import os
import shutil
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

_IMPORT_ERROR = None
try:
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core.record import Run
    from exopipe.phase3.lib.pages import common as pcommon
    from exopipe.phase3.lib.pages import tess_orbits
    from exopipe.phase3.lib.pages import tests as PT
    from exopipe.phase3.lib.serve import scripts, snapshot_copy, sublst_build, tess
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e
    if __name__ != '__main__':
        raise

TESS_COLS = ['cand_id', 'stokes', 'lst', 'date', 'band', 'ra', 'dec', 'epoch_start', 'epoch_end', 'status', 'sector', 'camera',
             'ccd', 'column', 'row']
FT_COLS = ['stokes', 'lst', 'band', 'date', 'date_obs', 'date_end', 'image', 'image_size', 'image_mtime']


class Patch:
    def __init__(self, *triples):
        self.triples, self.old = triples, []

    def __enter__(self):
        for obj, name, val in self.triples:
            self.old.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)
        return self

    def __exit__(self, *exc):
        for obj, name, val in reversed(self.old):
            setattr(obj, name, val)


class Tmp:
    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix='serve_tess_')
        return self.dir

    def __exit__(self, *exc):
        shutil.rmtree(self.dir, ignore_errors=True)


def wcsv(path, cols, rows, comment=None, crlf=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        if comment:
            f.write(comment + '\n')
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\r\n' if crlf else '\n')
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})


def snapshot_run(d):
    """A snapshot review copy: four detections (three Stokes V, one Stokes I) in master_catalog.csv."""
    DST = os.path.join(d, 'Run_SNAPTESS')
    ids = [f'05h_2025-01-02_23MHz_V_10s_20250102_03000{i}' for i in range(3)] + ['05h_2025-01-02_23MHz_I_1min_20250102_030000']
    wcsv(f'{DST}/master_catalog.csv', ['cand_id', 'lst', 'date', 'band', 'stokes'],
         [dict(cand_id=c, lst='05h', date='2025-01-02', band='23MHz', stokes=c.split('_')[3]) for c in ids])
    return DST, ids


def coverage_rows(ids, statuses):
    return [dict(cand_id=c, stokes=c.split('_')[3], lst='05h', date='2025-01-02', band='23MHz', ra='15.0', dec='1.0',
                 epoch_start='2025-01-02T03:00:00', epoch_end='2025-01-02T03:00:10', status=st,
                 **(dict(sector='87', camera='1', ccd='2', column='100', row='200') if st == 'simultaneous' else {}))
            for c, st in zip(ids, statuses)]


def tool(calls, tess_rows=None, summary=None, ok=True, ft_rows=None, ft_summary=None, ft_ok=True):
    """A stand-in for scripts.run: tess_coverage writes <run>/tess_coverage.csv (the tool's comment line, header, CRLF rows) and
    prints ``summary``; frame_times_tenmin writes <run>/frame_times.csv and prints ``ft_summary``."""
    def run(name, args=(), python='default', env=None, cwd=None, log=None, unit=None, timeout=None, **kw):
        calls.append(dict(name=name, args=list(args), python=python, cwd=cwd, log=log, env=env))
        run_dir = args[args.index('--run') + 1]
        if name == 'frame_times_tenmin':
            if not ft_ok:
                return types.SimpleNamespace(ok=False, rc=1, reason='exit 1 - error', tail=f'{run_dir}: no window - refused')
            wcsv(f'{run_dir}/frame_times.csv', FT_COLS, ft_rows or [], crlf=True)
            return types.SimpleNamespace(ok=True, rc=0, reason='', tail=ft_summary)
        if not ok:
            return types.SimpleNamespace(ok=False, rc=1, reason='exit 1 - error', tail='Traceback: KeyError cand_id')
        wcsv(f'{run_dir}/tess_coverage.csv', TESS_COLS, tess_rows or [], crlf=True,
             comment='# tess_coverage.py 2026-11-24T00:00:00Z; orbit table cached; tess-point; window = the radio epoch '
                     '(simultaneous only)')
        return types.SimpleNamespace(ok=True, rc=0, reason='', tail='a line before\n' + summary)
    return run


def summ(run_dir, written, left_out):
    return (f'{run_dir}/tess_coverage.csv: {written + left_out} detections (frame times the cand_id (own integration)); orbit '
            f'table cached; V not observed: 1; written {written} (left out, page says not computed: {left_out})')


def snapshot(d, parity=False, runner=None, table=None, **kw):
    """snapshot_copy.tess_stage on a synthetic review copy with a stand-in tool (the site's TESS interpreter stood in by this
    one); returns (unit, calls, DST, ids)."""
    DST, ids = snapshot_run(d)
    calls = []
    kw.setdefault('tess_rows', coverage_rows(ids[:3], ['simultaneous', 'not observed', 'not observed']))
    kw.setdefault('summary', summ(DST, 3, 1))
    run = Run('serve', ['test'], os.path.join(d, 'out'), CFG, config_sections=None)
    with Patch((scripts, 'run', runner(calls) if runner else tool(calls, **kw)),
               (pcommon, 'interpreter_or_none', lambda key: sys.executable),
               (tess, 'orbit_table', lambda: table or os.path.join(d, 'no_orbit_table.csv'))):
        u = snapshot_copy.tess_stage(run, types.SimpleNamespace(U=DST, run_name='Run_SNAPTESS', parity=parity))
    return u, calls, DST, ids


def logged(calls, outputs):
    """A stand-in for scripts.run that writes as core.legacy.run does: each call appends '===== <utc> <argv>' and its own
    output to the ONE log, and returns the log's last 40 lines as the tail, with log and argv. ``outputs``: one text per call;
    the tool still writes its file (3 rows of the review copy)."""
    texts = list(outputs)

    def run(name, args=(), python='default', env=None, cwd=None, log=None, unit=None, timeout=None, **kw):
        calls.append(dict(name=name, args=list(args), python=python, cwd=cwd, log=log, env=env))
        run_dir = args[args.index('--run') + 1]
        ids = [r['cand_id'] for r in csv.DictReader(open(f'{run_dir}/master_catalog.csv'))]
        wcsv(f'{run_dir}/tess_coverage.csv', TESS_COLS, coverage_rows(ids[:3], ['not observed'] * 3), crlf=True,
             comment='# tess_coverage.py x')
        argv = ['/py', '/ops/' + scripts.SCRIPTS[name][0]] + [str(a) for a in args]
        with open(log, 'a') as f:
            f.write(f'\n===== 2026-11-24T00:00:00Z {" ".join(argv)}\n' + texts.pop(0))
        tail = '\n'.join(open(log).read().splitlines()[-40:])
        return types.SimpleNamespace(ok=True, rc=0, reason='', tail=tail, log=log, argv=argv)
    return run


# =====================================================================================================================
# the snapshot review copy
# =====================================================================================================================

def test_the_snapshot_stage_runs_the_tool_as_production():
    with Tmp() as d:
        u, calls, DST, _ids = snapshot(d)
        assert u.name == 'snapshot_Run_SNAPTESS_tess' and u.status == 'done', (u.name, u.status, u.reason)
        assert calls == [dict(name='tess_coverage', args=['--run', DST, '--epoch-from-cand-id'],
                              python=CFG.pages_deep.TESS_PYTHON, cwd=CFG.serve.SNAPSHOT_STEP_CWD,
                              log=os.path.join(DST, 'tess_coverage.log'), env=None)], calls
        assert CFG.pages_deep.TESS_PYTHON == 'tess' and CFG.serve.SNAPSHOT_STEP_CWD == '/tmp'
        assert os.path.abspath(f'{DST}/master_catalog.csv') in u.inputs, u.inputs
        assert [o['class'] for o in u.outputs] == ['tess_coverage'] and u.outputs[0]['path'] == f'{DST}/tess_coverage.csv'
        assert u.meta['tess_coverage'] == {'written': 3, 'left_out': 1}, u.meta
        assert u.tests and all(t['passed'] for t in u.tests), u.tests
        assert snapshot_copy.STAGES.index('tess') == snapshot_copy.STAGES.index('page-context') + 1 \
            == snapshot_copy.STAGES.index('bright-filter') - 1, snapshot_copy.STAGES
        assert [s for s, _ in snapshot_copy.STEP_STAGES] == ['scintillation', 'page-context', 'tess', 'bright-filter',
                                                             'page-index'], snapshot_copy.STEP_STAGES


def test_a_failed_or_unreadable_tool_refuses():
    with Tmp() as d:
        u, _calls, DST, _ids = snapshot(d, ok=False)
        assert u.status == 'refused' and u.reason.startswith(f'TESS coverage: tess/tess_coverage.py failed on {DST} (exit 1') \
            and 'nothing served, servers untouched' in u.reason and not u.outputs, (u.status, u.reason)
    with Tmp() as d:
        u, _calls, _DST, _ids = snapshot(d, summary='done')
        assert u.status == 'refused' and 'no summary line' in u.reason and "'done'" in u.reason, (u.status, u.reason)
    with Tmp() as d:          # a summary line for another file is not this file's
        u, _calls, DST, _ids = snapshot(d, summary=summ('x', 3, 1) + '\nTraceback (most recent call last): boom')
        assert u.status == 'refused' and 'no summary line' in u.reason, (u.status, u.reason)
    with Tmp() as d:          # no master_catalog.csv: refused before the tool
        DST = os.path.join(d, 'Run_SNAPTESS')
        calls = []
        os.makedirs(DST)
        with Patch((scripts, 'run', tool(calls, summary='')), (pcommon, 'interpreter_or_none', lambda key: sys.executable),
                   (tess, 'orbit_table', lambda: os.path.join(d, 'no_orbit_table.csv'))):
            u = snapshot_copy.tess_stage(Run('serve', ['test'], os.path.join(d, 'out'), CFG, config_sections=None),
                                         types.SimpleNamespace(U=DST, run_name='Run_SNAPTESS'))
        assert u.status == 'refused' and 'master_catalog.csv' in u.reason and not calls, (u.status, u.reason)
    with Tmp() as d:          # the site has no TESS interpreter on this host: refused before the tool
        DST, _ids = snapshot_run(d)
        calls = []
        with Patch((scripts, 'run', tool(calls, summary='')), (pcommon, 'interpreter_or_none', lambda key: None)):
            u = snapshot_copy.tess_stage(Run('serve', ['test'], os.path.join(d, 'out'), CFG, config_sections=None),
                                         types.SimpleNamespace(U=DST, run_name='Run_SNAPTESS'))
        assert u.status == 'refused' and "PYTHON['tess'] interpreter is not on this host" in u.reason and not calls, u.reason


def test_the_summary_is_found_in_this_calls_own_output_only():
    """core.legacy.run appends every call to one log: the summary line is searched in the text after THIS call's '====='
    line, anywhere in it (not only on the last line); an earlier call's summary for the same file is never this call's."""
    with Tmp() as d:
        DST = os.path.join(d, 'Run_SNAPTESS')
        line = summ(DST, 3, 1)
        u, _c, DST, _ids = snapshot(d, runner=lambda calls: logged(calls, [f'{line}\nUserWarning: a line after it\n']))
        assert u.status == 'done' and u.meta['tess_coverage'] == {'written': 3, 'left_out': 1}, (u.status, u.reason)
    with Tmp() as d:
        DST = os.path.join(d, 'Run_SNAPTESS')
        os.makedirs(DST)
        with open(os.path.join(DST, 'tess_coverage.log'), 'a') as f:          # an earlier call, same file, same argv
            f.write(f'\n===== 2026-11-23T00:00:00Z /py /ops/{scripts.SCRIPTS["tess_coverage"][0]} --run {DST} '
                    f'--epoch-from-cand-id\n'
                    f'{summ(DST, 3, 1)}\n')
        u, _c, DST, _ids = snapshot(d, runner=lambda calls: logged(calls, ['Traceback (most recent call last): boom\n']))
        assert u.status == 'refused' and 'no summary line' in u.reason, (u.status, u.reason)
    with Tmp() as d:          # a stand-in result without a log: its tail
        res = types.SimpleNamespace(tail=f'x\n{summ("/r", 2, 0)}\ny')
        assert tess_orbits.coverage_counts(res, '/r/tess_coverage.csv') == (2, 0)
        assert tess_orbits.coverage_counts(res, '/r/./tess_coverage.csv') == (2, 0)
        assert tess_orbits.coverage_counts(res, '/s/tess_coverage.csv') is None
        res = types.SimpleNamespace(tail='', log=os.path.join(d, 'none.log'), argv=['a'])
        assert tess_orbits.own_output(res) is None and tess_orbits.coverage_counts(res, '/r/tess_coverage.csv') is None


def test_the_file_must_hold_exactly_what_the_tool_counted():
    with Tmp() as d:                                         # the tool counted 4 rows; the file holds 3
        u, _c, DST, _ids = snapshot(d, summary=summ(os.path.join(d, 'Run_SNAPTESS'), 4, 0))
        assert u.status == 'refused' and 'content test failed' in u.reason and '3 rows, the tool wrote 4' in u.reason, u.reason
    with Tmp() as d:                                         # one detection without a row, the tool said none left out
        u, _c, DST, _ids = snapshot(d, summary=summ(os.path.join(d, 'Run_SNAPTESS'), 3, 0))
        assert u.status == 'refused' and '1 master_catalog.csv detections without a row, the tool left out 0' in u.reason, \
            u.reason
    with Tmp() as d:                                         # a status the tool never writes
        DST, ids = snapshot_run(d)
        u, _c, DST, ids = snapshot(d, tess_rows=coverage_rows(ids[:3], ['simultaneous', 'not observed', 'no epoch time']))
        assert u.status == 'refused' and ids[2] in u.reason, u.reason
    with Tmp() as d:                                         # a simultaneous row without its sector / camera / CCD
        DST, ids = snapshot_run(d)
        rows = coverage_rows(ids[:3], ['simultaneous', 'not observed', 'not observed'])
        rows[0]['ccd'] = ''
        u, _c, DST, ids = snapshot(d, tess_rows=rows)
        assert u.status == 'refused' and ids[0] in u.reason, u.reason
    with Tmp() as d:                                         # a row of no detection of the run
        DST, ids = snapshot_run(d)
        rows = coverage_rows(ids[:2] + ['05h_2025-01-02_23MHz_V_10s_20250102_039999'], ['not observed'] * 3)
        u, _c, DST, ids = snapshot(d, tess_rows=rows, summary=summ(DST, 3, 1))
        assert u.status == 'refused' and '1 cand_ids not in master_catalog.csv' in u.reason, u.reason
    with Tmp() as d:                                         # every detection left out (no orbit reaches them): zero rows pass
        DST, ids = snapshot_run(d)
        u, _c, DST, ids = snapshot(d, tess_rows=[], summary=summ(DST, 0, 4))
        assert u.status == 'done' and u.meta['tess_coverage'] == {'written': 0, 'left_out': 4}, (u.status, u.reason)
    with Tmp() as d:                                         # without the counts (verify, the deep pages): exact equality
        DST, ids = snapshot_run(d)
        wcsv(f'{DST}/tess_coverage.csv', TESS_COLS, coverage_rows(ids, ['not observed'] * 4), comment='# tess_coverage.py x')
        assert PT.tess_rows(f'{DST}/tess_coverage.csv')[0]
        wcsv(f'{DST}/tess_coverage.csv', TESS_COLS, coverage_rows(ids[:3], ['not observed'] * 3), comment='# tess_coverage.py x')
        assert not PT.tess_rows(f'{DST}/tess_coverage.csv')[0]


def test_the_reader_uses_the_refreshed_orbit_table_when_the_refresh_step_owns_it():
    """pages_deep.TESS_REFRESH_IN_READER False: the tool gets --no-refresh once the orbit table reaches now; a table that ends
    before now refuses before the tool runs."""
    with Tmp() as d:
        table = os.path.join(d, 'TESS_orbit_times.csv')

        def orbits(end):
            with open(table, 'w') as f:
                f.write('Sector,Orbit,Start of Orbit,End of Orbit\n')
                f.write(f'100,200,2026-01-01 00:00:00,{end:%Y-%m-%d %H:%M:%S}\n')
        now = datetime.datetime.utcnow()
        orbits(now + datetime.timedelta(days=20))
        with Patch((CFG.pages_deep, 'TESS_REFRESH_IN_READER', False)):
            u, calls, DST, _ids = snapshot(d, table=table)
            assert u.status == 'done' and calls[0]['args'][-1] == '--no-refresh', (u.status, u.reason, calls)
            assert os.path.abspath(table) in u.inputs, u.inputs
            orbits(now - datetime.timedelta(days=2))
            u, calls, DST, _ids = snapshot(d, table=table)
            assert u.status == 'refused' and 'before now' in u.reason and not calls, (u.status, u.reason, calls)
        u, calls, DST, _ids = snapshot(d, table=table)       # True (production): the tool keeps its own table
        assert u.status == 'done' and '--no-refresh' not in calls[0]['args'], calls
        orbits(now + datetime.timedelta(days=20))            # ... except in a parity run, which never writes the shared table
        u, calls, DST, _ids = snapshot(d, parity=True, table=table)
        assert u.status == 'done' and calls[0]['args'][-1] == '--no-refresh', (u.status, u.reason, calls)
        orbits(now - datetime.timedelta(days=2))
        u, calls, DST, _ids = snapshot(d, parity=True, table=table)
        assert u.status == 'refused' and 'a parity run never writes the shared table' in u.reason and not calls, u.reason


def test_the_reader_gets_the_stokes_the_content_test_reads():
    """--stokes pages_deep.TESS_STOKES unless the tool's default 'I,V', the key tess_rows reads (the shared reader policy,
    lib/pages/tess_orbits.reader_args, as the deep pages)."""
    with Tmp() as d:
        u, calls, DST, ids = snapshot(d)
        assert '--stokes' not in calls[0]['args']
    with Tmp() as d, Patch((CFG.pages_deep, 'TESS_STOKES', 'V')):
        DST, ids = snapshot_run(d)
        u, calls, DST, ids = snapshot(d, summary=summ(DST, 3, 0))       # the three Stokes V detections, none left out
        assert u.status == 'done' and calls[0]['args'][-2:] == ['--stokes', 'V'], (u.status, u.reason, calls)
        assert tess_orbits.reader_args('/nonexistent', parity=False) == (['--stokes', 'V'], None)


# =====================================================================================================================
# the 10-minute new build
# =====================================================================================================================

WINDOWS = [('V', '05h', '23MHz', '2025-01-02T03'), ('V', '06h', '23MHz', '2025-01-04T11'), ('I', '06h', '27MHz', '2025-01-04T11')]


def tenmin_build(d):
    """A served 10-minute run U and its new build B: master_catalog.csv with one detection in each of three windows (two
    Stokes V, one Stokes I) and one deep row (its date is a night, not a window)."""
    U, B = os.path.join(d, 'Run_SLSTB'), os.path.join(d, 'Run_SLSTB.build')
    rows = [dict(cand_id=f'{S}1_{lst}_{w}_{b}_0001', stokes=S, lst=lst, band=b, date=w) for S, lst, b, w in WINDOWS]
    rows.append(dict(cand_id='V1_05h_2025-01-02_23MHz_0002', stokes='V', lst='05h', band='23MHz', date='2025-01-02'))
    wcsv(f'{B}/master_catalog.csv', ['cand_id', 'stokes', 'lst', 'band', 'date'], rows)
    os.makedirs(U)
    return U, B, [r['cand_id'] for r in rows]


def window_rows(windows=WINDOWS):
    return [dict(stokes=S, lst=lst, band=b, date=w, date_obs=f'{w}:00:01.0', date_end=f'{w}:09:59.0',
                 image=f'/x/{lst}_{b}/subtracted/{w}_{S}_sub_{b}.fits', image_size='100', image_mtime='1.000')
            for S, lst, b, w in windows]


def tenmin(d, reuse=None, **kw):
    U, B, ids = tenmin_build(d)
    if reuse is not None:
        with open(f'{U}/frame_times.csv', 'w') as f:
            f.write(reuse)
    calls = []
    kw.setdefault('ft_rows', window_rows())
    kw.setdefault('ft_summary', f'{B}/frame_times.csv: 3 of 3 windows (master_catalog.csv; images from /x; headers read 3, '
                                f'reused 0)')
    kw.setdefault('tess_rows', [dict(r, cand_id=c) for r, c in zip(coverage_rows(ids[:3], ['not observed'] * 3), ids[:3])])
    kw.setdefault('summary', summ(B, 3, 1))
    run = Run('serve', ['test'], os.path.join(d, 'out'), CFG, config_sections=None)
    with Patch((scripts, 'run', tool(calls, **kw)), (tess, 'orbit_table', lambda: os.path.join(d, 'no_orbit_table.csv'))):
        with run.unit('sublst_Run_SLSTB_views', stage='views', tier='sublst') as u:
            sublst_build.tess_coverage(u, types.SimpleNamespace(U=U, parity=False), B, f'{B}/incremental.log')
    return u, calls, U, B


def test_the_tenmin_step_runs_both_tools_as_production():
    cs = CFG.catalog_sublst
    tree = cs.TENMIN_TREE.format(root=CFG.paths.SUBLST_ROOT, tree=cs.ARM_TREES['B'])
    assert tree == '/lustre/gh/main/phase3/sublst/tenmin_root_x2', tree        # sublst_cumulative.sh:197
    with Tmp() as d:
        u, calls, U, B = tenmin(d)
        assert u.status == 'done', (u.status, u.reason)
        log = f'{B}/incremental.log'
        assert calls == [dict(name='frame_times_tenmin', args=['--run', B, '--image-root', tree], python='default', cwd=None,
                              log=log, env=None),
                         dict(name='tess_coverage', args=['--run', B], python='tess', cwd=None, log=log, env=None)], calls
        assert [o['class'] for o in u.outputs] == ['tenmin_frame_times', 'tess_coverage'], u.outputs
        assert u.meta['frame_times'] == {'windows': 3, 'needed': 3, 'reused': False} and \
            u.meta['tess_coverage'] == {'written': 3, 'left_out': 1}, u.meta
        assert u.tests and all(t['passed'] for t in u.tests), u.tests
    with Tmp() as d:                                         # the served run's window times, when that file is not empty
        u, calls, U, B = tenmin(d, reuse='stokes,lst\n')
        assert u.status == 'done' and calls[0]['args'][-2:] == ['--reuse', f'{U}/frame_times.csv'], calls
    with Tmp() as d:                                         # an empty file is not passed (production's `[ -s ]`)
        u, calls, U, B = tenmin(d, reuse='')
        assert u.status == 'done' and '--reuse' not in calls[0]['args'], calls


def test_the_tenmin_step_refuses_and_runs_no_coverage_after_a_failed_window_step():
    with Tmp() as d:
        u, calls, U, B = tenmin(d, ft_ok=False)
        assert u.status == 'refused' and [c['name'] for c in calls] == ['frame_times_tenmin'], (u.status, calls)
        assert u.reason.startswith(f'TESS coverage: tess/frame_times_tenmin.py failed on {B}') and 'nothing swapped in' in u.reason
    with Tmp() as d:
        u, calls, U, B = tenmin(d, ft_summary='finished')
        assert u.status == 'refused' and 'no summary line for' in u.reason and 'tess/frame_times_tenmin.py' in u.reason, u.reason
    with Tmp() as d:                                         # the tool counted 3 windows, the file holds 2
        u, calls, U, B = tenmin(d, ft_rows=window_rows(WINDOWS[:2]))
        assert u.status == 'refused' and 'failed their content test' in u.reason and '2 rows, the tool wrote 3' in u.reason \
            and [c['name'] for c in calls] == ['frame_times_tenmin'], (u.reason, calls)
    with Tmp() as d:
        u, calls, U, B = tenmin(d, ok=False)
        assert u.status == 'refused' and u.reason.startswith(f'TESS coverage: tess/tess_coverage.py failed on {B}'), u.reason


def test_the_window_times_content_test():
    with Tmp() as d:
        U, B, _ids = tenmin_build(d)
        p = f'{B}/frame_times.csv'
        wcsv(p, FT_COLS, window_rows())
        assert PT.tenmin_window_times(p, windows=3, needed=3)[0]
        assert PT.tenmin_window_times(p)[0]                                # without the counts
        neg = []
        for name, rows, kw in (
                ('a window twice', window_rows(WINDOWS + WINDOWS[:1]), dict(windows=4)),
                ('DATE-END before DATE-OBS', [dict(r, date_end=r['date_obs'][:13] + ':00:00.0') for r in window_rows()], {}),
                ('not a window name', window_rows([('V', '05h', '23MHz', '2025-01-02')]), {}),
                ('no window of master_catalog.csv', window_rows([('V', '07h', '23MHz', '2025-01-05T01')]), {}),
                ('rows != the tool\'s count', window_rows(), dict(windows=2)),
                ('master windows != the tool\'s needed', window_rows(), dict(windows=3, needed=4))):
            wcsv(p, FT_COLS, rows)
            r = PT.tenmin_window_times(p, **kw)
            neg.append((name, r))
        wcsv(p, FT_COLS[:-1], window_rows())
        neg.append(('columns', PT.tenmin_window_times(p)))
        wcsv(p, FT_COLS, [])
        neg.append(('no rows', PT.tenmin_window_times(p)))
        assert all(not r[0] for _, r in neg), [(n, r) for n, r in neg if r[0]]


def main(argv):
    if _IMPORT_ERROR is not None:
        print(f'imports missing: {type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}')
        return 2
    tests = [(n, f) for n, f in sorted(globals().items(), key=lambda kv: inspect.getsourcelines(kv[1])[1]
                                       if callable(kv[1]) and kv[0].startswith('test_') else 0)
             if callable(f) and n.startswith('test_') and (not argv or n in argv)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f'PASS {n}')
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f'FAIL {n}: {type(e).__name__}: {str(e)[:600]}')
    print(f'{len(tests) - failed} of {len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
