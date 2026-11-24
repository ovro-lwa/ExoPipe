#!/usr/bin/env python3
"""STANDING TEST - gate FE-10MIN-APPEND: the frames-only engine refuses a 10-minute append (data-free).

The search engine treats ANY 10-minute call with included nights as a per-hour append and then reads the frozen gate
populations of its base attempt (ledger FINDING-20261113-FRAMES-ONLY-ENGINE-TENMIN-APPEND-READS-FROZEN-POPULATIONS). A
frames-only attempt holds none, so the frames-only engine (lib/frames/deep_engine.py, deep_child.py) refuses that call
before any attempt directory, child or engine import exists:

  1. UnitSpec refuses cadence 10min with included nights (tier sublst, or an explicit cadence), with the gate code;
     a whole-cell 10-minute spec and a deep-cadence spec with included nights are built as before
  2. the engine child refuses the same spec (exit 2, result gate FE-10MIN-APPEND) without importing the engine module
  3. the engine itself (run_deep_cell with the frames_only keyword that lib/frames/deep_seam proposes for the real engine
     source) refuses such a call before any work, and its default call stays statement-for-statement unchanged

Run standalone:  python tests/test_frames_tenmin_append_gate.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')


def test_unitspec_refuses_a_tenmin_append():
    from exopipe.phase3.core.record import Refusal
    from exopipe.phase3.lib.frames import deep_engine as DE
    base = dict(lst='09h', scope='Red', stokes='I', label='X', attempt='/nonexistent/att')
    for kw in (dict(tier='sublst', included_dates=['2025-01-02']), dict(tier='sublst', cadence='10min',
                                                                         included_dates=['2025-01-02T09', '2025-01-02T10']),
               dict(tier='deep', cadence='10min', included_dates=['2025-01-02'])):
        try:
            DE.UnitSpec(**base, **kw)
        except Refusal as e:
            msg = str(e)
            assert msg.startswith('FE-10MIN-APPEND: 09h_Red: the frames-only engine refuses a 10-minute append'), msg
            assert 'gate_population.csv' in msg and 'tree-nonmember' in msg, msg
        else:
            raise AssertionError(f'UnitSpec({kw}) was not refused')
        assert not os.path.exists('/nonexistent/att')
    # not an append: a whole-cell 10-minute spec, a deep spec with included nights (the per-hour deep frames)
    s = DE.UnitSpec(**base, tier='sublst')
    assert s.cadence == '10min' and s.included_dates == []
    s = DE.UnitSpec(**base, tier='deep', included_dates=['2025-01-02'])
    assert s.cadence == 'deep' and s.included_dates == ['2025-01-02']
    assert DE.tenmin_append_refusal('09h_Red', '10min', []) is None
    assert DE.tenmin_append_refusal('09h_Red', 'deep', ['2025-01-02']) is None
    assert DE.tenmin_append_refusal('09h_Red', '10min', ['2025-01-02']).startswith(DE.TENMIN_APPEND_GATE + ': ')


def test_engine_child_refuses_before_the_engine_import():
    d = tempfile.mkdtemp(prefix='test_frames_tenmin_append_gate_')
    try:
        spec = {'tier': 'sublst', 'cadence': '10min', 'lst': '09h', 'scope': 'Red', 'stokes': 'I', 'label': 'X',
                'included_dates': ['2025-01-02'], 'attempt': os.path.join(d, 'att'),
                'result_path': os.path.join(d, 'result.json')}
        sp = os.path.join(d, 'spec.json')
        json.dump(spec, open(sp, 'w'))
        code = ('import sys; from exopipe.phase3.lib.frames import deep_child as c; rc = c.main(["deep_child", %r]); '
                'print("ENGINE-IMPORTED", "exopipe.phase3.lib.search.deep_search" in sys.modules); sys.exit(rc)' % sp)
        p = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True, timeout=300,
                           env=dict(os.environ, PYTHONPATH=REPO))
        assert p.returncode == 2, (p.returncode, p.stdout, p.stderr[-2000:])
        assert 'FRAMES-ENGINE-END refused FE-10MIN-APPEND' in p.stdout and 'ENGINE-IMPORTED False' in p.stdout, p.stdout
        res = json.load(open(spec['result_path']))
        assert res['status'] == 'refused' and res['gate'] == 'FE-10MIN-APPEND' and 'engine_file' not in res, res
        assert res['reason'].startswith('09h_Red: the frames-only engine refuses a 10-minute append'), res
        assert not os.path.exists(spec['attempt'])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_engine_refuses_with_the_frames_only_keyword():
    """The engine itself (run_deep_cell with the proposed frames_only keyword, lib/frames/deep_seam.propose on the real
    engine source) refuses a frames-only 10-minute call with included nights before any work: before the cadence is set
    in the environment, before any directory exists. The default call is unchanged (the neutrality proof)."""
    import ast
    from exopipe.phase3.lib.frames import deep_engine, deep_seam
    path = deep_engine.engine_source()
    src = open(path).read()
    if deep_seam.has_keyword(ast.parse(src)):
        patched = src                      # the search owner's commit landed: test the engine as it is
    else:
        patched = deep_seam.propose(src)
        ok, det = deep_seam.default_path_identical(patched, src)
        assert ok, det
    assert deep_seam.frames_only_path_ok(patched)[0]
    d = tempfile.mkdtemp(prefix='test_frames_tenmin_append_gate_engine_')
    try:
        pp = os.path.join(d, 'patched_engine.py')
        open(pp, 'w').write(patched)
        code = ('import os, sys, importlib.util as U\n'
                'os.environ.pop("P3_CADENCE", None)\n'
                f's = U.spec_from_file_location("patched_engine", {pp!r}); m = U.module_from_spec(s); s.loader.exec_module(m)\n'
                'for kw in (dict(cadence="10min", included_dates=["2025-01-02"]),\n'
                '           dict(cadence="10min", included_dates=["2025-01-02T09"], nonmember_nights={"2025-01-02": 1.0})):\n'
                '    try:\n'
                '        m.run_deep_cell("09h", "Red", label="X", stokes="I", frames_only=True, **kw)\n'
                '    except RuntimeError as e:\n'
                '        print("REFUSED", str(e))\n'
                '    else:\n'
                '        print("NOT REFUSED")\n'
                'print("CADENCE", os.environ.get("P3_CADENCE"))\n')
        p = subprocess.run([sys.executable, '-c', code], cwd=d, capture_output=True, text=True, timeout=600,
                           env=dict(os.environ, PYTHONPATH=REPO, P3_STAGING_ROOT=os.path.join(d, 'staging')))
        assert p.returncode == 0, (p.returncode, p.stdout[-2000:], p.stderr[-3000:])
        lines = [x for x in p.stdout.splitlines() if x.startswith(('REFUSED', 'NOT REFUSED', 'CADENCE'))]
        assert len(lines) == 3 and all(x.startswith('REFUSED FE-10MIN-APPEND: 09h_Red: the frames-only engine refuses')
                                       and x.endswith('- REFUSING') for x in lines[:2]), p.stdout[-3000:]
        assert lines[2] == 'CADENCE None', lines                     # refused before the engine set anything up
        left = sorted(x for x in os.listdir(d) if x != '__pycache__')              # the import's own byte code only
        assert left == ['patched_engine.py'], left                               # no staging root, no work directory
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    names = ('test_unitspec_refuses_a_tenmin_append', 'test_engine_child_refuses_before_the_engine_import',
             'test_engine_refuses_with_the_frames_only_keyword')
    for n in names:
        globals()[n]()
        print(f'PASS {n}', flush=True)
    print(f'ALL {len(names)} PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
