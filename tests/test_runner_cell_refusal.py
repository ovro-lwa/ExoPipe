#!/usr/bin/env python3
"""STANDING TEST - per-cell refusals in a runner chain (core/runner.py; runner.CELL_REFUSAL_CHAINS), data-free.

The per-hour chain 'epoch' runs every cell of an hour through each step. Before this test's change, one refusing cell stopped
the whole hour at that step. Prerequisite (2) of the refactor chat's decision of 2026-11-14 (ledger
DECISION-20261114-PR21-ADMIT-STEPS-NOT-IN-EPOCH-CHAIN-YET): record a cell's refusal and continue the hour's other cells.

  1. refusal_code: the gate code a reason names (leading code or 'gate X'; v2 has no registry of gate codes, so no other
     token is taken; never a ledger ID)
  2. the step side: a real record.Run whose units refuse, complete or refuse before any unit -> runner.step_outcome ->
     runner.cell_refusals (a cell refusal only when the units' exit is the step's 2, every refused unit names its cell and
     nothing refused before or outside a unit; a unit's subband is kept with its cell)
  2b. the exclusion: a real record.Run handed refused (cell, subband) pairs (record.UPSTREAM_ENV): the guard
     run.upstream_refused records each unit of a refused cell (of that subband, or of the whole cell) refused, code
     UPSTREAM-REFUSED naming the step, without opening it; a refused cell's unit opened without the guard refuses the
     whole step; every per-cell unit of the chain steps' modules is opened behind the guard (a source check)
  3. the protocol through a real step process: `exopipe3 verify` writes its outcome to EXOPIPE3_STEP_OUTCOME_FD and does
     not leave the variable to its children
  4. a synthetic chain (fake step processes from a pinned --code repository, whose outcomes are the real step_outcome
     output of item 2): one step refuses a cell and the chain goes on through every later step; a second step refuses
     two cells; the item ends 'cells-refused' with "COMPLETE with 2 refused cells (...)", never done; the closing step runs
     with that outcome; the hour's email lists every refused cell with its code; `run status` prints the line
  5. what still stops the item: an error (no later step runs, the cells refused before it are kept), a refusal before any
     unit, a refused unit without a cell, an exit 2 without an outcome (a --code commit older than the protocol), and any
     refusal in a chain that is not in CELL_REFUSAL_CHAINS
  6. a retried cells-refused item resumes at the first step that refused a cell and runs every later step again; a
     'running' step of a dead worker resumes there; a later done entry is never dropped; a chain that is not in
     CELL_REFUSAL_CHAINS keeps the old resume (every done step skipped)
  7. --after with after_ok: an upstream item that ended cells-refused counts as finished well, and its refused cells carry
     over to the dependent job's steps through the same exclusion

Run standalone:  python tests/test_runner_cell_refusal.py [test_name ...]
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
PY = sys.executable


class Scratch:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix=f'test_runner_cell_refusal_{self.name}_')
        return self.path

    def __exit__(self, *exc):
        for dp, dn, fn in os.walk(self.path):
            for x in dn + fn:
                p = os.path.join(dp, x)
                if not os.path.islink(p):
                    os.chmod(p, 0o700)
        os.chmod(self.path, 0o700)
        shutil.rmtree(self.path, ignore_errors=True)


class runner_cfg:
    """Set CFG.runner attributes for the block (restored after)."""

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        from exopipe.phase3.config import CFG
        self.old = {k: getattr(CFG.runner, k) for k in self.kw}
        for k, v in self.kw.items():
            setattr(CFG.runner, k, v)
        return CFG

    def __exit__(self, *exc):
        from exopipe.phase3.config import CFG
        for k, v in self.old.items():
            setattr(CFG.runner, k, v)


def _git(repo, *args):
    r = subprocess.run(['git', '-C', repo, '-c', 'user.name=test', '-c', 'user.email=test@example.invalid'] + list(args),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


# ---------------------------------------------------------------------------------------------------------------------
# outcomes from real records
# ---------------------------------------------------------------------------------------------------------------------

def _outcome(out, units, run_reason=None, name='search', upstream=None):
    """The real step_outcome of a real record.Run (dry run: no record written) whose units end as listed:
    (unit name, cell or None, 'done' | 'no-input' | 'refused' | 'body', reason[, subband]); 'body' runs a body that would
    complete the unit (the unit framework refuses it first when ``upstream``, record.UPSTREAM_ENV's list, names its cell)."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import record, runner
    old = os.environ.pop(record.UPSTREAM_ENV, None)
    if upstream:
        os.environ[record.UPSTREAM_ENV] = json.dumps(upstream)
    try:
        run = record.Run(name, [f'exopipe3 {name}'], out, CFG, dry_run=True)
        _units(run, units)
    finally:
        os.environ.pop(record.UPSTREAM_ENV, None)
        if old is not None:
            os.environ[record.UPSTREAM_ENV] = old
    if run_reason:
        run.run_reason = run_reason
        rc = record.EXIT_REFUSED
    else:
        rc = run.exit_code()
    return runner.step_outcome(name, rc, run)


def _units(run, units):
    from exopipe.phase3.core import record
    for unit, cell, kind, reason, *sb in units:
        meta = {'cell': cell} if cell else {}
        if sb:
            meta['subband'] = sb[0]
        if kind == 'body':
            if run.upstream_refused(unit, **meta):         # the guard every unit loop calls before it opens a unit
                continue
            with run.unit(unit, **meta) as u:
                u.status = 'body-ran'
                raise record.NoInput('the body ran')
        elif kind == 'refused':
            with run.unit(unit, **meta):
                raise record.Refusal(reason)          # the unit's own context manager records it and swallows it
        elif kind == 'no-input':
            with run.unit(unit, **meta):
                raise record.NoInput(reason)
        else:
            u = run.unit(unit, **meta)
            u.status, u.reason = 'done', ''


def test_refusal_code():
    from exopipe.phase3.core.runner import refusal_code as rc
    assert rc('CP-GEN: photometry_campaign.GENERATION x') == 'CP-GEN'
    assert rc('G-K the window') == 'G-K'
    assert rc('FE-10MIN-APPEND: 09h_Red: the frames-only engine refuses') == 'FE-10MIN-APPEND'
    assert rc('04h_Red: the master violates the membership rulings (gate E3): x') == 'E3'
    assert rc('04h_Red: residuals_provisional.csv missing - gate E14 cannot run; refused (never skipped)') == 'E14'
    assert rc('UPSTREAM-REFUSED: 04h_Red was refused by search --tier deep [E3] earlier') == 'UPSTREAM-REFUSED'
    # a code-shaped token elsewhere is not taken (v2 has no registry of gate codes to check it against)
    assert rc('flow record /x status refused (r) - G-M05') is None
    assert rc('04h_Red: no deep I master at /x - inputs missing (gate E1)') == 'E1'
    # ledger IDs are never codes; a reason without a code has none
    assert rc('searched by a full re-search (R-SIGMA-STACK-INCLUDES-THE-RE-ADMITTED-EPOCH-BACKFILL-BY-RE-SEARCH)') is None
    assert rc('the ruling R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS (3) says') is None
    assert rc('every planned item refused (2): a.fits: below the elevation floor') is None
    assert rc('gate population missing; see the 2025-10-14 night') is None
    assert rc('') is None and rc(None) is None


def test_chain_outcome_lists_each_cell_once():
    from exopipe.phase3.core import runner
    argv = ['py', '-m', 'exopipe.phase3', 'search', '--tier', 'deep', '--epochs', '07h_2025-10-14']
    why = '07h_23MHz %s: 2025-10-14 already searched in 07h_23MHz_%s_20261111T102537 (an append would duplicate its rows)'
    steps = [{'step': 0, 'argv': argv, 'status': 'cells-refused', 'refused_cells': [
        {'cell': '07h_23MHz', 'unit': f'07h_23MHz_{s}_2025-10-14', 'code': None, 'reason': why % (s, s)} for s in 'VI']},
        {'step': 1, 'argv': argv[:3] + ['catalog'], 'status': 'done'}]
    assert runner.chain_outcome('cells-refused', steps, 2) == \
        'COMPLETE with 1 refused cell (07h_23MHz: search --tier deep [no code named])'
    assert len(runner.refused_cells_of(steps)) == 2                   # every refused unit stays in the record
    assert runner.chain_outcome('done', steps[1:], 2) == 'COMPLETE (2 steps)'
    assert runner.chain_outcome('no-input', [], 3) == 'NO INPUT (none of the 3 steps had anything to do)'
    stop = dict(steps[1], status='refused', rc=2)
    assert runner.chain_outcome('refused', [steps[0], stop], 5) == \
        'REFUSED at step 2 (catalog, exit 2); refused cells before it: 1 (07h_23MHz: search --tier deep [no code named])'


def test_step_outcome_and_cell_refusals():
    from exopipe.phase3.core import runner
    with Scratch('outcome') as d:
        # two cells done, one refused by a gate: a cell refusal
        o = _outcome(d, [('04h_Full_I', '04h_Full', 'done', ''),
                         ('04h_Red_I', '04h_Red', 'refused', '04h_Red: the master violates the membership rulings (gate E3): x'),
                         ('04h_Blue_I', '04h_Blue', 'no-input', 'excluded by data_decisions.csv')])
        assert o['protocol'] == runner.OUTCOME_PROTOCOL and o['rc'] == 2 and o['refusals_outside_units'] == 0, o
        assert [(u['unit'], u['status'], u['cell']) for u in o['units']] == [
            ('04h_Full_I', 'done', '04h_Full'), ('04h_Red_I', 'refused', '04h_Red'), ('04h_Blue_I', 'no-input', '04h_Blue')]
        assert o['unit_exit'] == 2 and [u['upstream'] for u in o['units']] == [False] * 3, o
        c = runner.cell_refusals(o, 2)
        assert c == [{'cell': '04h_Red', 'subband': None, 'unit': '04h_Red_I', 'code': 'E3',
                      'reason': '04h_Red: the master violates the membership rulings (gate E3): x'}], c
        # the exit the units give must be the step's 2 (a step that returned 2 itself is not a cell refusal)
        assert runner.cell_refusals(dict(o, unit_exit=0), 2) is None and runner.cell_refusals(dict(o, unit_exit=None), 2) is None
        # a unit with a subband: the refusal is that (cell, subband) pair
        o = _outcome(d, [('flow_x_41MHz', '04h_2025-10-14', 'refused', 'C-NOCUBE: x', '41MHz'),
                         ('flow_x_36MHz', '04h_2025-10-14', 'done', '', '36MHz')])
        assert [(x['cell'], x['subband'], x['code']) for x in runner.cell_refusals(o, 2)] == [
            ('04h_2025-10-14', '41MHz', 'C-NOCUBE')]
        assert runner.cell_refusals(o, 1) is None                 # the process exit decides: an error is never a cell refusal
        assert runner.cell_refusals(None, 2) is None              # no outcome handed back
        # every cell refused: still cell refusals (each recorded; the hour's other tiers go on)
        o = _outcome(d, [('a', '04h_Full', 'refused', 'G-M05 x'), ('b', '04h_Red', 'refused', 'G-M05 y')])
        assert [x['cell'] for x in runner.cell_refusals(o, 2)] == ['04h_Full', '04h_Red']
        # a refused unit that is not one cell's (a run-wide unit): not a cell refusal
        o = _outcome(d, [('04h_Full_I', '04h_Full', 'done', ''), ('incremental.associate', None, 'refused', 'G-C1: x')])
        assert o['rc'] == 2 and runner.cell_refusals(o, 2) is None
        # a refusal before any unit (the step's own): not a cell refusal
        o = _outcome(d, [], run_reason='refused: --epochs is refused for --tier deep')
        assert o['rc'] == 2 and runner.cell_refusals(o, 2) is None
        o = _outcome(d, [('04h_Red_I', '04h_Red', 'refused', 'E3')], run_reason='refused: config')
        assert runner.cell_refusals(o, 2) is None
        # a unit in an unknown state, refusals outside the units, another protocol: not a cell refusal
        bad = dict(o, run_reason=None, units=[dict(o['units'][0]), {'unit': 'x', 'status': 'running', 'cell': 'c'}])
        assert runner.cell_refusals(bad, 2) is None
        assert runner.cell_refusals(dict(bad, units=bad['units'][:1]), 2) is not None
        assert runner.cell_refusals(dict(bad, units=bad['units'][:1], refusals_outside_units=1), 2) is None
        assert runner.cell_refusals(dict(bad, units=bad['units'][:1], refusals_outside_units=None), 2) is None
        # reasons are kept to REASON_CHARS in the item record
        o = _outcome(d, [('a', '04h_Red', 'refused', 'E3 ' + 'x' * 5000)])
        assert len(runner.cell_refusals(o, 2)[0]['reason']) == runner.REASON_CHARS


def test_unit_framework_refuses_upstream_cells():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import record, runner
    up = [{'cell': '04h_Red', 'subband': None, 'step': 'search --tier deep', 'code': 'E3'},
          {'cell': '04h_2025-10-14', 'subband': '41MHz', 'step': 'flow --tier snapshot', 'code': None}]
    before = sys.gettrace()
    with Scratch('upstream') as d:
        o = _outcome(d, [('04h_Red_I', '04h_Red', 'body', ''), ('04h_Full_I', '04h_Full', 'done', ''),
                         ('s41', '04h_2025-10-14', 'body', '', '41MHz'), ('s36', '04h_2025-10-14', 'no-input', 'x', '36MHz'),
                         ('whole', '04h_2025-10-14', 'body', '')], upstream=up)
        st = {u['unit']: (u['status'], u['upstream'], u['reason']) for u in o['units']}
        assert st['04h_Full_I'][0] == 'done' and st['s36'] == ('no-input', False, 'x'), st
        # the bodies never ran: each unit recorded refused by the guard without being opened, naming the step
        for n, step in (('04h_Red_I', 'search --tier deep [E3]'), ('s41', 'flow --tier snapshot [no code named]'),
                        ('whole', 'flow --tier snapshot [no code named]')):
            assert st[n][:2] == ('refused', True) and st[n][2].startswith('UPSTREAM-REFUSED: ') and step in st[n][2], st[n]
        assert o['rc'] == o['unit_exit'] == 2 and runner.cell_refusals(o, 2) == [], o      # no NEW refused cell: goes on
        os.environ[record.UPSTREAM_ENV] = json.dumps(up)
        try:
            run = record.Run('x', ['exopipe3 x'], d, CFG, dry_run=True)
            # the fail-safe: a refused cell's unit opened WITHOUT the guard refuses the whole step, its body never run
            ran = []
            try:
                with run.unit('04h_Red_V', cell='04h_Red', stokes='V'):
                    ran.append(1)
                raise AssertionError('opened')
            except record.UpstreamRefused as e:
                assert not ran and 'opened without the guard' in str(e) and run.units[-1].status == 'refused', e
            # a unit of another cell opens as before; the guard says False and records nothing
            n = len(run.units)
            assert run.upstream_refused('04h_Full_V', cell='04h_Full') is False and len(run.units) == n
            with run.unit('04h_Full_V', cell='04h_Full') as u:
                u.status = 'done'
            # an unreadable list refuses the run (never a step without the list it was given)
            os.environ[record.UPSTREAM_ENV] = '{not json'
            run = record.Run('x', ['exopipe3 x'], d, CFG, dry_run=True)
            try:
                run.unit('a', cell='04h_Full')
                raise AssertionError('accepted')
            except record.Refusal as e:
                assert 'UPSTREAM-REFUSED' in str(e)
        finally:
            os.environ.pop(record.UPSTREAM_ENV, None)
    assert sys.gettrace() is before
    assert runner.upstream_of([{'cell': 'a', 'subband': None, 'name': 's', 'code': 'E3'},
                               {'cell': 'a', 'subband': None, 'name': 't', 'code': None},
                               {'cell': 'a', 'subband': '41MHz', 'name': 't', 'code': None}]) == [
        {'cell': 'a', 'subband': None, 'step': 's', 'code': 'E3'}, {'cell': 'a', 'subband': '41MHz', 'step': 't', 'code': None}]


# modules of the per-hour chain's steps (CELL_REFUSAL_CHAINS 'epoch'): every per-cell unit they open sits behind the guard
GUARDED = ('lib/search/deep_step.py', 'lib/frames/tier_snapshot.py', 'lib/flow/steps_snapshot.py',
           'lib/search/tier_snapshot.py', 'lib/search/tier_sublst.py', 'lib/masters/sublst_nonmember.py',
           'lib/masters/tier_sublst.py', 'steps/curves.py')
# modules with per-cell units in no CELL_REFUSAL_CHAIN today: the fail-safe applies (a refused cell refuses the step)
FAILSAFE_ONLY = ('lib/dewarp/steps_cell.py', 'lib/flow/steps_deep.py', 'lib/jupiter/measure.py',
                 'lib/jupiter/quarantine.py', 'lib/jupiter/v2.py', 'lib/masters/admit_stages.py', 'lib/masters/deep_stages.py',
                 'lib/photometry/campaign.py', 'lib/photometry/steps_cutouts.py', 'lib/photometry/steps_snapshot.py')


def test_every_chain_unit_of_a_cell_is_guarded():
    """Every `with ... .unit(... cell=...)` of a module in GUARDED has the guard (run.upstream_refused) in the lines
    before it, after the previous unit; a module with per-cell units in neither list fails (classify it)."""
    import re
    root = os.path.join(REPO, 'exopipe', 'phase3')
    bad, seen = [], set()
    for dp, _, fn in os.walk(root):
        for f in fn:
            if not f.endswith('.py'):
                continue
            path = os.path.join(dp, f)
            rel = os.path.relpath(path, root)
            lines = open(path).read().split('\n')
            for k, ln in enumerate(lines):
                if not re.search(r'\bwith\b.*\.unit\(', ln):
                    continue
                call = ' '.join(lines[k:k + 3])
                if not re.search(r'\bcell=', call):
                    continue
                seen.add(rel)
                if rel in FAILSAFE_ONLY:
                    continue
                if rel not in GUARDED:
                    bad.append(f'{rel}:{k + 1}: a per-cell unit in a module in neither GUARDED nor FAILSAFE_ONLY')
                    continue
                back = []                      # the lines since the previous unit opened or the function began
                for b in reversed(lines[max(0, k - 60):k]):
                    if re.search(r'\bwith\b.*\.unit\(', b) or re.match(r'\s*def ', b):
                        break
                    back.append(b)
                if not any('upstream_refused(' in b for b in back):
                    bad.append(f'{rel}:{k + 1}: opened without the guard run.upstream_refused')
    assert not bad, bad
    assert set(GUARDED) <= seen, sorted(set(GUARDED) - seen)


def test_step_process_hands_its_outcome():
    """A real step process (exopipe3 verify) writes its outcome to the inherited descriptor; without the variable it writes
    none; the variable is taken out of the step's environment (its children never see it)."""
    from exopipe.phase3.core import runner
    with Scratch('fd') as d:
        good = os.path.join(d, 'good.csv')
        open(good, 'w').write('a,b\n1,2\n')
        env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get('PYTHONPATH', ''))
        for args, want_rc in ((['verify', '--list', '--glob', good, '--cores', 'none'], 0),
                              (['verify', 'csv_table', os.path.join(d, 'missing.csv'), '--cores', 'none'], None)):
            with tempfile.TemporaryFile() as f:
                e = dict(env, **{runner.OUTCOME_FD_ENV: str(f.fileno())})
                p = subprocess.run([PY, '-m', 'exopipe.phase3'] + args, env=e, cwd=REPO, pass_fds=(f.fileno(),),
                                   capture_output=True, text=True, timeout=300)
                o = runner.read_outcome(f)
            assert o is not None, (p.returncode, p.stderr[-2000:])
            assert o['protocol'] == runner.OUTCOME_PROTOCOL and o['step'] == 'verify' and o['rc'] == p.returncode, o
            assert want_rc is None or p.returncode == want_rc, (p.returncode, p.stderr[-2000:])
            assert isinstance(o['units'], list) and o['refusals_outside_units'] == 0, o
        # no variable: nothing is written to an inherited descriptor
        with tempfile.TemporaryFile() as f:
            p = subprocess.run([PY, '-m', 'exopipe.phase3', 'verify', '--list', '--glob', good, '--cores', 'none'],
                               env=env, cwd=REPO, pass_fds=(f.fileno(),), capture_output=True, text=True, timeout=300)
            assert p.returncode == 0 and runner.read_outcome(f) is None
        # cli.main takes the variable out of its environment before the step runs
        code = ('import os, sys; from exopipe.phase3 import cli; os.environ["%s"] = "999"; rc = cli.main(["verify", "--list", '
                '"--glob", %r, "--cores", "none"]); print("LEFT", os.environ.get("%s")); sys.exit(0)'
                % (runner.OUTCOME_FD_ENV, good, runner.OUTCOME_FD_ENV))
        p = subprocess.run([PY, '-c', code], env=env, cwd=REPO, capture_output=True, text=True, timeout=300)
        assert 'LEFT None' in p.stdout and 'not handed to the runner' in p.stderr, (p.stdout, p.stderr[-2000:])


# ---------------------------------------------------------------------------------------------------------------------
# a synthetic chain of fake step processes
# ---------------------------------------------------------------------------------------------------------------------

FAKE_MAIN = r'''import os, sys
# fake exopipe3 step: <name> <outcome JSON file | -> <exit code> [anything]
a = sys.argv[1:]
name, path, rc = a[0], a[1], int(a[2])
fd = os.environ.get('EXOPIPE3_STEP_OUTCOME_FD')
print('FAKE-STEP', name, 'fd' if fd else 'nofd', ' '.join(a[3:]), flush=True)
print('FAKE-UPSTREAM', name, os.environ.get('EXOPIPE3_UPSTREAM_REFUSED') or '-', flush=True)
if fd and path != '-':
    data = open(path, 'rb').read()
    os.write(int(fd), data)
sys.exit(rc)
'''


def _fake_repo(d):
    repo = os.path.join(d, 'repo')
    os.makedirs(os.path.join(repo, 'exopipe', 'phase3'))
    open(os.path.join(repo, 'exopipe', '__init__.py'), 'w').write('')
    open(os.path.join(repo, 'exopipe', 'phase3', '__init__.py'), 'w').write('')
    open(os.path.join(repo, 'exopipe', 'phase3', '__main__.py'), 'w').write(FAKE_MAIN)
    subprocess.run(['git', 'init', '-q', repo], check=True, capture_output=True)
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '-m', 'fake steps')
    return repo, _git(repo, 'rev-parse', 'HEAD')


def _write(d, name, obj):
    p = os.path.join(d, name + '.json')
    with open(p, 'w') as f:
        json.dump(obj, f)
    return p


def test_chain_goes_on_past_cell_refusals():
    from exopipe.phase3.core import runner
    from exopipe.phase3.lib.notify import block
    with Scratch('chain') as d:
        repo, sha = _fake_repo(d)
        jobs = os.path.join(d, 'jobs')
        red = _write(d, 'red', _outcome(d, [
            ('04h_Full_I', '04h_Full', 'done', ''),
            ('04h_Red_I', '04h_Red', 'refused', '04h_Red: the master violates the membership rulings (gate E3): x')]))
        two = _write(d, 'two', _outcome(d, [
            ('admit-prep_04h_Full', '04h_Full', 'done', ''),
            ('admit-prep_04h_Red', '04h_Red', 'refused', 'admit-prep: 04h_Red holds an excluded night - G-M05'),
            ('admit-prep_04h_Blue', '04h_Blue', 'refused', 'admit-prep: no build statistics')], name='masters'))
        # photometry gets 04h_Red from search's refusal: the unit framework refuses its 04h_Red unit (UPSTREAM-REFUSED)
        upred = _write(d, 'upred', _outcome(d, [('04h_Full_I', '04h_Full', 'done', ''), ('04h_Red_I', '04h_Red', 'body', '')],
                                            name='photometry', upstream=[{'cell': '04h_Red', 'subband': None,
                                                                          'step': 'search --tier deep', 'code': 'E3'}]))
        runlevel = _write(d, 'runlevel', _outcome(d, [], run_reason='refused: --epochs is refused for --tier deep'))
        nocell = _write(d, 'nocell', _outcome(d, [('04h_Full_I', '04h_Full', 'done', ''),
                                                  ('incremental.associate', None, 'refused', 'G-C1: x')]))
        hour = '04h_2025-10-14'
        chains = {
            'tc': [f'search {red} 2 --tier deep --epochs {{item}}', f'photometry {upred} 2 --tier deep --epochs {{item}}',
                   f'masters {two} 2 --tier deep --stages admit-prep --epochs {{item}}', 'catalog - 3 --epochs {item}',
                   'pages - 0 --epochs {item}'],
            'te': [f'search {red} 2 --epochs {{item}}', 'photometry - 1 --epochs {item}', 'pages - 0 --epochs {item}'],
            'tw': [f'search {runlevel} 2 --epochs {{item}}', 'pages - 0 --epochs {item}'],
            'tn': [f'catalog {nocell} 2 --epochs {{item}}', 'pages - 0 --epochs {item}'],
            'tx': [f'search - 2 --epochs {{item}}', 'pages - 0 --epochs {item}'],          # exit 2, no outcome handed back
            'to': [f'search {red} 2 --epochs {{item}}', 'pages - 0 --epochs {item}'],      # not a cell-refusal chain
            'tu': ['pages - 0 --epochs {item}'],                                       # a job --after 'tc'
        }
        fin = {'tc': ['notify - 0 --chain-record {record}'], 'te': ['notify - 0 --chain-record {record}']}
        with runner_cfg(JOBS_ROOT=jobs, CODE_REPO=repo, CODE_WORKTREES_ROOT=os.path.join(d, '_code'), POLL_S=1,
                        CHAINS=chains, CHAIN_FINALLY=fin, CELL_REFUSAL_CHAINS=('tc', 'te', 'tw', 'tn', 'tx', 'tu')):
            host = socket.gethostname()

            def run(chain):
                job = runner.submit(None, [], [hour], [(host, 1)], start=False, chain=chain, code=sha)
                runner.worker(job)
                return job, runner.item_state(job, hour)

            # --- one step refuses a cell, a later one two cells: every step runs, the item is complete with 2 refused cells
            job, st = run('tc')
            assert runner.load_job(job)['cell_refusals'] is True
            assert [s['status'] for s in st['steps']] == ['cells-refused'] * 3 + ['no-input', 'done'], st
            assert st['status'] == 'cells-refused' and st['chain_status'] == 'cells-refused' and st['rc'] == 2, st
            assert st['steps'][0]['units'] == {'done': 1, 'refused': 1} and st['steps'][4].get('units') is None, st['steps']
            assert st['steps'][0]['refused_cells'] == [{'cell': '04h_Red', 'subband': None, 'unit': '04h_Red_I', 'code': 'E3',
                                                        'reason': '04h_Red: the master violates the membership rulings '
                                                        '(gate E3): x'}]
            # the later step's 04h_Red unit was refused by the unit framework: no new refusal, the cell is counted once
            assert st['steps'][1]['refused_cells'] == [] and st['steps'][1]['upstream_refused_units'] == 1, st['steps'][1]
            assert [x['cell'] for x in st['refused_by_cell']] == ['04h_Red', '04h_Blue'], st['refused_by_cell']
            assert [(c['step'], c['name'], c['cell'], c['code']) for c in st['refused_cells']] == [
                (0, 'search --tier deep', '04h_Red', 'E3'),
                (2, 'masters --tier deep --stages admit-prep', '04h_Red', None),
                (2, 'masters --tier deep --stages admit-prep', '04h_Blue', None)], st['refused_cells']
            assert st['summary'] == ('COMPLETE with 2 refused cells (04h_Red: search --tier deep [E3], masters --tier deep '
                                     '--stages admit-prep [no code named]; 04h_Blue: masters --tier deep --stages admit-prep '
                                     '[no code named])'), st['summary']
            assert [f['status'] for f in st['finally']] == ['done'], st['finally']
            log = open(st['log']).read()
            assert log.count('FAKE-STEP') == 6 and 'FAKE-STEP pages fd' in log and 'FAKE-STEP notify nofd' in log, log
            # every later step was handed the refused cells so far (record.UPSTREAM_ENV); the closing step is not
            ups = {ln.split()[1]: ln.split(' ', 2)[2] for ln in log.splitlines() if ln.startswith('FAKE-UPSTREAM')}
            assert ups['search'] == '-' and ups['notify'] == '-', ups
            assert [e['cell'] for e in json.loads(ups['photometry'])] == ['04h_Red'], ups
            assert [(e['cell'], e['step'], e['code']) for e in json.loads(ups['pages'])] == [
                ('04h_Red', 'search --tier deep', 'E3'), ('04h_Blue', 'masters --tier deep --stages admit-prep', None)], ups
            assert 'refused cell 04h_Red (unit 04h_Red_I, code E3)' in log and 'the chain goes on for the other cells' in log
            assert runner.job_state(job) == ('finished', {'cells-refused': 1})
            # the hour's email lists every refused cell with its step and code
            t, ok = block.chain_summary(os.path.join(runner.job_dir(job), 'items', f'{hour}.json'))
            assert ok is False and 'COMPLETE with 2 refused cells' in t and "other cells went on through every step" in t, t
            # one bullet per cell, from the record's per-cell list (the summary's)
            assert t.count('<li>') == 2 and '<code>search --tier deep [E3]</code>' in t, t
            assert ('04h_Red: refused by <code>search --tier deep [E3]</code>, <code>masters --tier deep --stages '
                    'admit-prep [no code named]</code>: 04h_Red: the master violates') in t, t
            assert '04h_Blue: refused by <code>masters --tier deep --stages admit-prep [no code named]</code>' in t, t
            # `run status` prints the line
            r = subprocess.run([PY, '-m', 'exopipe.phase3', 'run', 'status', job, '--cores', 'none', '--set',
                                f'runner.JOBS_ROOT={jobs}'], cwd=REPO, capture_output=True, text=True, timeout=120,
                               env=dict(os.environ, PYTHONPATH=REPO))
            assert r.returncode == 0 and 'COMPLETE with 2 refused cells (04h_Red' in r.stdout, (r.stdout, r.stderr)
            assert 'cells-refused ' in r.stdout and 'steps 5/5' in r.stdout, r.stdout     # every step finished

            # --- --after with after_ok: cells-refused is a good end, and its refused cells carry over to the dependent job
            import threading
            job2 = runner.submit(None, [], [hour], [], after=job, after_ok=True, start=False, chain='tu', code=sha)
            sup = threading.Thread(target=runner.supervise, args=(job2,), daemon=True)
            sup.start()
            for _ in range(60):
                if os.path.exists(os.path.join(runner.job_dir(job2), runner.UPSTREAM_FILE)):
                    break
                import time
                time.sleep(0.5)
            runner.worker(job2)
            sup.join(30)
            st3 = runner.item_state(job2, hour)
            assert not sup.is_alive() and st3['status'] == 'cells-refused', st3
            assert st3['summary'].startswith('COMPLETE with 2 refused cells (04h_Red: job ') and 'search --tier deep [E3]' \
                in st3['summary'], st3['summary']
            ups = [ln for ln in open(st3['log']).read().splitlines() if ln.startswith('FAKE-UPSTREAM pages')]
            assert len(ups) == 1 and [e['cell'] for e in json.loads(ups[0].split(' ', 2)[2])] == ['04h_Red', '04h_Blue'], ups

            # --- a retry resumes at the first step that refused a cell and runs every later step again
            first = st['steps'][0]['started']
            assert runner.retry(job, ('cells-refused',), start=False) == 1
            runner.worker(job)
            st2 = runner.item_state(job, hour)
            assert st2['attempts'] == 1 and len(st2['steps']) == 5 and st2['status'] == 'cells-refused', st2
            assert all(s['attempt'] == 1 for s in st2['steps']) and open(st2['log']).read().count('FAKE-STEP pages') == 2
            assert st2['steps'][0]['started'] >= first and len(st2['refused_cells']) == 3

            # --- a 'running' step (a dead worker) resumes there; earlier done steps are skipped, later ones run again
            ipath = os.path.join(runner.job_dir(job), 'items', f'{hour}.json')
            dead = dict(st2, status='failed')
            dead['steps'] = [dict(st2['steps'][0], status='done', refused_cells=[]), dict(st2['steps'][1], status='running')] \
                + st2['steps'][2:]
            runner._wjson(ipath, dead)
            n_search = open(st2['log']).read().count('FAKE-STEP search')
            assert runner.retry(job, ('failed',), start=False) == 1
            runner.worker(job)
            st4 = runner.item_state(job, hour)
            log = open(st4['log']).read()
            assert log.count('FAKE-STEP search') == n_search and log.count('FAKE-STEP pages') == 3, log
            assert [s['status'] for s in st4['steps']] == ['done', 'cells-refused', 'cells-refused', 'no-input', 'done'], st4

            # --- an error still stops the item; the cell refused before it is kept and said
            job, st = run('te')
            assert [s['status'] for s in st['steps']] == ['cells-refused', 'error'] and st['status'] == 'failed', st
            assert st['summary'] == ('FAILED at step 2 (photometry, exit 1); refused cells before it: 1 (04h_Red: search '
                                     '[E3])'), st['summary']
            assert 'FAKE-STEP pages' not in open(st['log']).read().split('closing step')[0]
            t, _ = block.chain_summary(os.path.join(runner.job_dir(job), 'items', f'{hour}.json'))
            assert 'FAILED at step 2' in t and 'Cells refused before it:' in t and '04h_Red' in t, t
            # a later done entry is never dropped: the item resumes at its failed step and stops there again; the done
            # entry of the earlier attempt after it stays in the record (its attempt says which)
            ipath = os.path.join(runner.job_dir(job), 'items', f'{hour}.json')
            later = {'step': 2, 'name': 'pages', 'argv': ['pages'], 'status': 'done', 'rc': 0, 'attempt': 1}
            runner._wjson(ipath, dict(st, steps=[dict(st['steps'][0], status='done', refused_cells=[]), st['steps'][1], later]))
            assert runner.retry(job, ('failed',), start=False) == 1
            runner.worker(job)
            st5 = runner.item_state(job, hour)
            assert [s['status'] for s in st5['steps']] == ['done', 'error', 'done'] and st5['steps'][2] == later, st5['steps']
            assert st5['summary'] == 'FAILED at step 2 (photometry, exit 1)', st5['summary']

            # --- what is not a cell refusal stops the item at that step, as before
            for chain in ('tw', 'tn', 'tx', 'to'):
                job, st = run(chain)
                assert st['status'] == 'refused' and [s['status'] for s in st['steps']] == ['refused'], (chain, st)
                assert st['refused_cells'] == [] and st['summary'].startswith('REFUSED at step 1'), (chain, st['summary'])
                assert 'FAKE-STEP pages' not in open(st['log']).read(), chain
            assert runner.load_job(job)['cell_refusals'] is False                # 'to' is not in CELL_REFUSAL_CHAINS
            # a chain that is not in CELL_REFUSAL_CHAINS keeps the old resume: every done step is skipped, wherever it is
            ipath = os.path.join(runner.job_dir(job), 'items', f'{hour}.json')
            runner._wjson(ipath, dict(st, steps=[st['steps'][0], dict(st['steps'][0], step=1, name='pages', status='done')]))
            assert runner.retry(job, ('refused',), start=False) == 1
            runner.worker(job)
            st6 = runner.item_state(job, hour)
            assert [s['status'] for s in st6['steps']] == ['refused', 'done'] and 'FAKE-STEP pages' not in open(
                st6['log']).read(), st6


def test_epoch_chain_is_a_cell_refusal_chain():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import runner
    assert 'epoch' in CFG.runner.CELL_REFUSAL_CHAINS and 'epoch' in CFG.runner.CHAINS
    assert set(CFG.runner.CELL_REFUSAL_CHAINS) <= set(CFG.runner.CHAINS)
    assert runner.CELLS_REFUSED in runner.TERMINAL and runner.CELLS_REFUSED in runner.AFTER_OK


def main(argv):
    names = argv or [n for n in sorted(globals()) if n.startswith('test_')]
    order = [n for n in ('test_refusal_code', 'test_chain_outcome_lists_each_cell_once', 'test_step_outcome_and_cell_refusals',
                         'test_unit_framework_refuses_upstream_cells', 'test_every_chain_unit_of_a_cell_is_guarded',
                         'test_step_process_hands_its_outcome',
                         'test_chain_goes_on_past_cell_refusals', 'test_epoch_chain_is_a_cell_refusal_chain') if n in names]
    order += [n for n in names if n not in order]
    for n in order:
        globals()[n]()
        print(f'PASS {n}', flush=True)
    print(f'ALL {len(order)} PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
