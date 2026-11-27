#!/usr/bin/env python3
"""Data-free tests: the deep catalog's harvest pulls each attempt's rules record (cutover row 64, design step 2;
plan/designs/ROW64_DEEP_PER_HOUR_CATALOG_DESIGN_20261126.md section 6).

The search step writes the removal rules an attempt was searched under into the attempt directory
(lib/search/removal_rules.RULES_RECORD, search_rules.json). The shell harvest it ports (ops/p3s3_harvest_published.sh:16-23)
pulled only the run tree and the log, so an attempt it pulled read as searched under 'production'
(removal_rules.attempt_rules) and the check (catalog_conform.MODE 'check', conform_check.rules_findings) refused it. Asserts:
  (i)    the record's rsync (deep_ported.harvest_rules_command): the same destination and host rule as the tree's
         (harvest_commands), the source <staging>/<cell>_<S>_<att>/search_rules.json, --ignore-missing-args only when the
         record is not required;
  (ii)   deep_ported.harvest_attempt with rsync on synthetic attempts: required and present - pulled, the destination reads
         'once-in-search' at RULES_VERSION; required and missing at the source - refused before the tree is pulled; not
         required and missing - the tree is pulled and the attempt reads 'production'; not required and present - the
         record is still copied; a tree missing at the source - refused;
  (iii)  the order of the commands, the pause after each remote rsync only, and a required record that is not at the
         destination after two successful exits (a runner that copies nothing) - refused;
  (iv)   deep_chain's harvest stage calls harvest_attempt with the record required exactly under catalog_conform.MODE
         'check'.

Scratch: a fresh directory under $EXOPIPE3_TEST_SCRATCH, else the system temp directory, removed at exit.
Run: python tests/test_catalog_harvest_rules.py [test names]   (exit 0 = all passed, 1 = a failure, 2 = imports missing)
"""
import inspect
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

_IMPORT_ERROR = None
try:
    from exopipe.phase3.lib.catalog import deep_chain as DC
    from exopipe.phase3.lib.catalog import deep_ported as DP
    from exopipe.phase3.lib.search import removal_rules as RR
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e
    if __name__ != '__main__':
        raise

LABEL = 'P3S5'
HOST = 'nodeA'
TMP = None


def _tmp():
    global TMP
    if TMP is None:
        TMP = tempfile.mkdtemp(prefix='catalog_harvest_rules_', dir=os.environ.get('EXOPIPE3_TEST_SCRATCH') or None)
    d = tempfile.mkdtemp(dir=TMP)
    return d


def _attempt(cell='00h_18MHz', st='V', att='20261127T153810', node=HOST):
    return dict(cell=cell, stokes=st, node=node, att=att)


def _staged(staging, a, rules='once-in-search', tree=True):
    """A synthetic staged attempt: Run_<label>/, transients/Run_<label>/v_deep_ledger.csv, <cell>.log and, unless ``rules`` is
    None, the search step's record (removal_rules.write_attempt_rules)."""
    d = os.path.join(staging, f"{a['cell']}_{a['stokes']}_{a['att']}")
    os.makedirs(os.path.join(d, f'Run_{LABEL}', 'vresid', a['cell']))
    with open(os.path.join(d, f'Run_{LABEL}', 'vresid', a['cell'], 'frame.txt'), 'w') as f:
        f.write('frame\n')
    if tree:
        os.makedirs(os.path.join(d, 'transients', f'Run_{LABEL}'))
        with open(os.path.join(d, 'transients', f'Run_{LABEL}', 'v_deep_ledger.csv'), 'w') as f:
            f.write('cand_id\n')
    with open(os.path.join(d, f"{a['cell']}.log"), 'w') as f:
        f.write('log\n')
    if rules is not None:
        RR.write_attempt_rules(d, rules)
    return d


def _rsync_run(calls):
    """run(argv, env additions) for harvest_attempt: real rsync for a local attempt, recording each argv."""
    def run(argv, extra):
        calls.append(list(argv))
        return subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL).returncode
    return run


def test_rules_command_shapes():
    for node in (HOST, 'nodeB'):
        a = _attempt(node=node)
        for req in (True, False):
            dest, argv, extra, remote = DP.harvest_rules_command(a, LABEL, HOST, '/dest', '/stg', req)
            tdest, targv, textra, tremote = DP.harvest_commands(a, LABEL, HOST, '/dest', '/stg')
            assert (dest, extra, remote) == (tdest, textra, tremote), (dest, extra, remote, tdest, textra, tremote)
            src = f"/stg/{a['cell']}_{a['stokes']}_{a['att']}/{RR.RULES_RECORD}"
            assert argv[-2] == (src if node == HOST else f'{node}:{src}'), argv
            assert argv[-1] == f'{dest}/', argv
            assert argv[0] == 'rsync' and '-a' in argv, argv
            assert ('--ignore-missing-args' in argv) == (not req), argv
            assert remote == (node != HOST) and (extra == ({'SSH_AUTH_SOCK': ''} if remote else {})), (remote, extra)
    assert RR.RULES_RECORD == 'search_rules.json'


def test_harvest_attempt_with_rsync():
    assert shutil.which('rsync'), 'rsync is not installed (the harvest runs it)'
    want = RR.rules_record('once-in-search')
    # required, present: pulled with the tree; the destination reads the in-search rules at the current version
    d = _tmp()
    a = _attempt()
    src = _staged(os.path.join(d, 'stg'), a)
    calls = []
    dest, why = DP.harvest_attempt(a, LABEL, HOST, os.path.join(d, 'dest'), os.path.join(d, 'stg'), True, _rsync_run(calls))
    assert why is None, why
    assert RR.attempt_rules(dest) == want == {'rules': 'once-in-search', 'version': RR.RULES_VERSION}, RR.attempt_rules(dest)
    assert open(os.path.join(dest, RR.RULES_RECORD)).read() == open(os.path.join(src, RR.RULES_RECORD)).read()
    for rel in (f'Run_{LABEL}/vresid/{a["cell"]}/frame.txt', f'transients/Run_{LABEL}/v_deep_ledger.csv', f'{a["cell"]}.log'):
        assert os.path.isfile(os.path.join(dest, rel)), rel
    assert len(calls) == 2 and calls[0][-2].endswith(RR.RULES_RECORD) and calls[1][-2].endswith(f'{a["cell"]}.log'), calls
    # required, missing at the source: refused, and the tree is NOT pulled into the collect root
    d = _tmp()
    _staged(os.path.join(d, 'stg'), a, rules=None)
    calls = []
    dest, why = DP.harvest_attempt(a, LABEL, HOST, os.path.join(d, 'dest'), os.path.join(d, 'stg'), True, _rsync_run(calls))
    assert why and RR.RULES_RECORD in why and "'check'" in why and 'not pulled' in why, why
    assert len(calls) == 1, calls
    assert not os.path.exists(os.path.join(dest, 'transients')) and not os.path.exists(os.path.join(dest, f'Run_{LABEL}')), \
        os.listdir(dest)
    # not required, missing: the tree is pulled, the attempt reads 'production' (as the shell harvest left it)
    d = _tmp()
    _staged(os.path.join(d, 'stg'), a, rules=None)
    dest, why = DP.harvest_attempt(a, LABEL, HOST, os.path.join(d, 'dest'), os.path.join(d, 'stg'), False, _rsync_run([]))
    assert why is None, why
    assert os.path.isfile(os.path.join(dest, 'transients', f'Run_{LABEL}', 'v_deep_ledger.csv'))
    assert RR.attempt_rules(dest) == {'rules': 'production', 'version': None}, RR.attempt_rules(dest)
    # not required, present: the record is still copied (an attempt keeps the rules it was searched under in every mode)
    d = _tmp()
    _staged(os.path.join(d, 'stg'), a)
    dest, why = DP.harvest_attempt(a, LABEL, HOST, os.path.join(d, 'dest'), os.path.join(d, 'stg'), False, _rsync_run([]))
    assert why is None and RR.attempt_rules(dest) == want, (why, RR.attempt_rules(dest))
    # the tree missing at the source (no transients/): refused, required or not
    for req in (True, False):
        d = _tmp()
        _staged(os.path.join(d, 'stg'), a, tree=False)
        dest, why = DP.harvest_attempt(a, LABEL, HOST, os.path.join(d, 'dest'), os.path.join(d, 'stg'), req, _rsync_run([]))
        assert why and 'run tree not pulled' in why, (req, why)


def test_order_pause_and_destination_check():
    a = _attempt(node='nodeB')
    for req in (True, False):
        d = _tmp()
        calls, pauses = [], []

        def run(argv, extra):
            calls.append((list(argv), dict(extra)))
            pauses.append('run')
            return 0
        dest, why = DP.harvest_attempt(a, LABEL, HOST, os.path.join(d, 'dest'), os.path.join(d, 'stg'), req, run,
                                       pause=lambda: pauses.append('pause'))
        assert pauses == ['run', 'pause', 'run', 'pause'], pauses
        assert [c[0][-2] for c in calls][0].startswith('nodeB:') and calls[0][0][-2].endswith(RR.RULES_RECORD), calls
        assert all(c[1] == {'SSH_AUTH_SOCK': ''} for c in calls), calls
        # a runner that exits 0 and copies nothing: under 'check' the record is not at the destination - refused
        if req:
            assert why and f'no {RR.RULES_RECORD} in {dest}' in why, why
        else:
            assert why is None, why
    # a local attempt: no pause
    d = _tmp()
    pauses = []
    DP.harvest_attempt(_attempt(), LABEL, HOST, os.path.join(d, 'dest'), os.path.join(d, 'stg'), False,
                       lambda argv, extra: 0, pause=lambda: pauses.append('pause'))
    assert pauses == [], pauses


def test_chain_requires_the_record_under_check():
    src = inspect.getsource(DC.DeepRun.st_harvest)
    assert 'deep_ported.harvest_attempt(' in src, 'the harvest stage does not pull through harvest_attempt'
    assert 'deep_ported.harvest_commands(' not in src, 'the harvest stage still pulls the tree without its rules record'
    assert "rules_required = _CC.conform_mode() == 'check'" in src, 'the record is not required exactly under check'


def main(argv):
    if _IMPORT_ERROR is not None:
        print(f'imports missing: {type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}')
        return 2
    tests = [(n, f) for n, f in sorted(globals().items(), key=lambda kv: inspect.getsourcelines(kv[1])[1]
                                       if callable(kv[1]) and kv[0].startswith('test_') else 0)
             if callable(f) and n.startswith('test_') and (not argv or n in argv)]
    failed = 0
    try:
        for n, f in tests:
            try:
                f()
                print(f'PASS {n}')
            except Exception as e:  # noqa: BLE001
                failed += 1
                import traceback
                traceback.print_exc()
                print(f'FAIL {n}: {type(e).__name__}: {str(e)[:600]}')
    finally:
        if TMP:
            shutil.rmtree(TMP, ignore_errors=True)
    print(f'{len(tests) - failed} of {len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
