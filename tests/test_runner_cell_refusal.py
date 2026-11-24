#!/usr/bin/env python3
"""STANDING TEST - per-cell refusals in a runner chain (core/runner.py; runner.CELL_REFUSAL_CHAINS), data-free.

The per-hour chain 'epoch' runs every cell of an hour through each step. Before this test's change, one refusing cell stopped
the whole hour at that step. Prerequisite (2) of the refactor chat's decision of 2026-11-14 (ledger
DECISION-20261114-PR21-ADMIT-STEPS-NOT-IN-EPOCH-CHAIN-YET): record a cell's refusal and continue the hour's other cells.

  1. refusal_code: the gate code a reason names (leading code, 'gate X', a code-shaped token; never a ledger ID)
  2. the step side: a real record.Run whose units refuse, complete or refuse before any unit -> runner.step_outcome ->
     runner.cell_refusals (a cell refusal only when every refused unit names its cell and nothing refused outside a unit)
  3. the protocol through a real step process: `exopipe3 verify` writes its outcome to EXOPIPE3_STEP_OUTCOME_FD and does
     not leave the variable to its children
  4. a synthetic chain (fake step processes from a pinned --code repository, whose outcomes are the real step_outcome
     output of item 2): one step refuses a cell and the chain goes on through every later step; a second step refuses
     two cells; the item ends 'cells-refused' with "COMPLETE with 2 refused cells (...)", never done; the closing step runs
     with that outcome; the hour's email lists every refused cell with its code; `run status` prints the line
  5. what still stops the item: an error (no later step runs, the cells refused before it are kept), a refusal before any
     unit, a refused unit without a cell, an exit 2 without an outcome (a --code commit older than the protocol), and any
     refusal in a chain that is not in CELL_REFUSAL_CHAINS
  6. a retried cells-refused item resumes at the first step that refused a cell and runs every later step again

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

def _outcome(out, units, run_reason=None, name='search'):
    """The real step_outcome of a real record.Run (dry run: no record written) whose units end as listed:
    (unit name, cell or None, 'done' | 'no-input' | 'refused', reason)."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import record, runner
    run = record.Run(name, [f'exopipe3 {name}'], out, CFG, dry_run=True)
    for unit, cell, kind, reason in units:
        meta = {'cell': cell} if cell else {}
        if kind == 'refused':
            with run.unit(unit, **meta):
                raise record.Refusal(reason)          # the unit's own context manager records it and swallows it
        elif kind == 'no-input':
            with run.unit(unit, **meta):
                raise record.NoInput(reason)
        else:
            u = run.unit(unit, **meta)
            u.status, u.reason = 'done', ''
    if run_reason:
        run.run_reason = run_reason
        rc = record.EXIT_REFUSED
    else:
        rc = run.exit_code()
    return runner.step_outcome(name, rc, run)


def test_refusal_code():
    from exopipe.phase3.core.runner import refusal_code as rc
    assert rc('CP-GEN: photometry_campaign.GENERATION x') == 'CP-GEN'
    assert rc('G-K the window') == 'G-K'
    assert rc('FE-10MIN-APPEND: 09h_Red: the frames-only engine refuses') == 'FE-10MIN-APPEND'
    assert rc('04h_Red: the master violates the membership rulings (gate E3): x') == 'E3'
    assert rc('04h_Red: residuals_provisional.csv missing - gate E14 cannot run; refused (never skipped)') == 'E14'
    assert rc('flow record /x status refused (r) - G-M05') == 'G-M05'
    assert rc('04h_Red: no deep I master at /x - inputs missing (gate E1)') == 'E1'
    # ledger IDs are never codes; a reason without a code has none
    assert rc('searched by a full re-search (R-SIGMA-STACK-INCLUDES-THE-RE-ADMITTED-EPOCH-BACKFILL-BY-RE-SEARCH)') is None
    assert rc('the ruling R-GREGG-20261112-PER-HOUR-STEPS-SIX-RULINGS (3) says') is None
    assert rc('every planned item refused (2): a.fits: below the elevation floor') is None
    assert rc('gate population missing; see the 2025-10-14 night') is None
    assert rc('') is None and rc(None) is None


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
        c = runner.cell_refusals(o, 2)
        assert c == [{'cell': '04h_Red', 'unit': '04h_Red_I', 'code': 'E3',
                      'reason': '04h_Red: the master violates the membership rulings (gate E3): x'}], c
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
        runlevel = _write(d, 'runlevel', _outcome(d, [], run_reason='refused: --epochs is refused for --tier deep'))
        nocell = _write(d, 'nocell', _outcome(d, [('04h_Full_I', '04h_Full', 'done', ''),
                                                  ('incremental.associate', None, 'refused', 'G-C1: x')]))
        hour = '04h_2025-10-14'
        chains = {
            'tc': [f'search {red} 2 --tier deep --epochs {{item}}', 'photometry - 0 --tier deep --epochs {item}',
                   f'masters {two} 2 --tier deep --stages admit-prep --epochs {{item}}', 'catalog - 3 --epochs {item}',
                   'pages - 0 --epochs {item}'],
            'te': [f'search {red} 2 --epochs {{item}}', 'photometry - 1 --epochs {item}', 'pages - 0 --epochs {item}'],
            'tw': [f'search {runlevel} 2 --epochs {{item}}', 'pages - 0 --epochs {item}'],
            'tn': [f'catalog {nocell} 2 --epochs {{item}}', 'pages - 0 --epochs {item}'],
            'tx': [f'search - 2 --epochs {{item}}', 'pages - 0 --epochs {item}'],          # exit 2, no outcome handed back
            'to': [f'search {red} 2 --epochs {{item}}', 'pages - 0 --epochs {item}'],      # not a cell-refusal chain
        }
        fin = {'tc': ['notify - 0 --chain-record {record}'], 'te': ['notify - 0 --chain-record {record}']}
        with runner_cfg(JOBS_ROOT=jobs, CODE_REPO=repo, CODE_WORKTREES_ROOT=os.path.join(d, '_code'), POLL_S=1,
                        CHAINS=chains, CHAIN_FINALLY=fin, CELL_REFUSAL_CHAINS=('tc', 'te', 'tw', 'tn', 'tx')):
            host = socket.gethostname()

            def run(chain):
                job = runner.submit(None, [], [hour], [(host, 1)], start=False, chain=chain, code=sha)
                runner.worker(job)
                return job, runner.item_state(job, hour)

            # --- one step refuses a cell, a later one two cells: every step runs, the item is complete with 2 refused cells
            job, st = run('tc')
            assert runner.load_job(job)['cell_refusals'] is True
            assert [s['status'] for s in st['steps']] == ['cells-refused', 'done', 'cells-refused', 'no-input', 'done'], st
            assert st['status'] == 'cells-refused' and st['chain_status'] == 'cells-refused' and st['rc'] == 2, st
            assert st['steps'][0]['units'] == {'done': 1, 'refused': 1} and st['steps'][1].get('units') is None, st['steps']
            assert st['steps'][0]['refused_cells'] == [{'cell': '04h_Red', 'unit': '04h_Red_I', 'code': 'E3', 'reason':
                                                        '04h_Red: the master violates the membership rulings (gate E3): x'}]
            assert [(c['step'], c['name'], c['cell'], c['code']) for c in st['refused_cells']] == [
                (0, 'search --tier deep', '04h_Red', 'E3'),
                (2, 'masters --tier deep --stages admit-prep', '04h_Red', 'G-M05'),
                (2, 'masters --tier deep --stages admit-prep', '04h_Blue', None)], st['refused_cells']
            assert st['summary'] == ('COMPLETE with 2 refused cells (04h_Red: search --tier deep [E3], masters --tier deep '
                                     '--stages admit-prep [G-M05]; 04h_Blue: masters --tier deep --stages admit-prep '
                                     '[no code named])'), st['summary']
            assert [f['status'] for f in st['finally']] == ['done'], st['finally']
            log = open(st['log']).read()
            assert log.count('FAKE-STEP') == 6 and 'FAKE-STEP pages fd' in log and 'FAKE-STEP notify nofd' in log, log
            assert 'refused cell 04h_Red (unit 04h_Red_I, code E3)' in log and 'the chain goes on for the other cells' in log
            assert runner.job_state(job) == ('finished', {'cells-refused': 1})
            # the hour's email lists every refused cell with its step and code
            t, ok = block.chain_summary(os.path.join(runner.job_dir(job), 'items', f'{hour}.json'))
            assert ok is False and 'COMPLETE with 2 refused cells' in t and "other cells went on through every step" in t, t
            assert t.count('<li>') == 3 and 'refusal code E3' in t and 'refusal code none named' in t, t
            assert '<code>masters --tier deep --stages admit-prep</code>, refusal code G-M05' in t, t
            # `run status` prints the line
            r = subprocess.run([PY, '-m', 'exopipe.phase3', 'run', 'status', job, '--cores', 'none', '--set',
                                f'runner.JOBS_ROOT={jobs}'], cwd=REPO, capture_output=True, text=True, timeout=120,
                               env=dict(os.environ, PYTHONPATH=REPO))
            assert r.returncode == 0 and 'COMPLETE with 2 refused cells (04h_Red' in r.stdout, (r.stdout, r.stderr)

            # --- a retry resumes at the first step that refused a cell and runs every later step again
            first = st['steps'][0]['started']
            assert runner.retry(job, ('cells-refused',), start=False) == 1
            runner.worker(job)
            st2 = runner.item_state(job, hour)
            assert st2['attempts'] == 1 and len(st2['steps']) == 5 and st2['status'] == 'cells-refused', st2
            assert all(s['attempt'] == 1 for s in st2['steps']) and open(st2['log']).read().count('FAKE-STEP pages') == 2
            assert st2['steps'][0]['started'] >= first and len(st2['refused_cells']) == 3

            # --- an error still stops the item; the cell refused before it is kept and said
            job, st = run('te')
            assert [s['status'] for s in st['steps']] == ['cells-refused', 'error'] and st['status'] == 'failed', st
            assert st['summary'] == ('FAILED at step 2 (photometry, exit 1); refused cells before it: 1 (04h_Red: search '
                                     '[E3])'), st['summary']
            assert 'FAKE-STEP pages' not in open(st['log']).read().split('closing step')[0]
            t, _ = block.chain_summary(os.path.join(runner.job_dir(job), 'items', f'{hour}.json'))
            assert 'FAILED at step 2' in t and 'Cells refused before it:' in t and '04h_Red' in t, t

            # --- what is not a cell refusal stops the item at that step, as before
            for chain in ('tw', 'tn', 'tx', 'to'):
                job, st = run(chain)
                assert st['status'] == 'refused' and [s['status'] for s in st['steps']] == ['refused'], (chain, st)
                assert st['refused_cells'] == [] and st['summary'].startswith('REFUSED at step 1'), (chain, st['summary'])
                assert 'FAKE-STEP pages' not in open(st['log']).read(), chain
            assert runner.load_job(job)['cell_refusals'] is False                # 'to' is not in CELL_REFUSAL_CHAINS


def test_epoch_chain_is_a_cell_refusal_chain():
    from exopipe.phase3.config import CFG
    from exopipe.phase3.core import runner
    assert 'epoch' in CFG.runner.CELL_REFUSAL_CHAINS and 'epoch' in CFG.runner.CHAINS
    assert set(CFG.runner.CELL_REFUSAL_CHAINS) <= set(CFG.runner.CHAINS)
    assert runner.CELLS_REFUSED in runner.TERMINAL and runner.CELLS_REFUSED not in runner.AFTER_OK


def main(argv):
    names = argv or [n for n in sorted(globals()) if n.startswith('test_')]
    order = [n for n in ('test_refusal_code', 'test_step_outcome_and_cell_refusals', 'test_step_process_hands_its_outcome',
                         'test_chain_goes_on_past_cell_refusals', 'test_epoch_chain_is_a_cell_refusal_chain') if n in names]
    order += [n for n in names if n not in order]
    for n in order:
        globals()[n]()
        print(f'PASS {n}', flush=True)
    print(f'ALL {len(order)} PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
