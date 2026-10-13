#!/usr/bin/env python3
"""Refactor move checks (MOVE_PLAN.md section 8) - run after every move commit on the refactor branch.

1. NEW NAMES IMPORT. Every module of the package is imported in a fresh interpreter under its CURRENT name and the
   result is compared with the pre-move baseline of the module it came from (exopipe/phase3/core/_shim.LEGACY maps
   old -> new): a module that imported before the move must import now; one that failed before (a package missing
   from this environment) must fail the same way. A module new on the branch must import. A baseline module that is
   neither present nor in LEGACY is reported as LOST.
2. RUNNABLES. Every launched module (_shim.RUNNABLE: `python -m` from ops shell, or run by file path) is run with its
   probe argv (--help, or no argument where --help would not stop it) under its CURRENT name, and its exit code and
   output must equal the pre-move tree's under the OLD name (repository root, program name and module names
   normalised). Entries whose probe is None are never executed.
3. OLD NAMES - to be shimmed at cutover. SKIPPED unless EXOPIPE_SHIM_CHECK is set:
     EXOPIPE_SHIM_CHECK=finder  every LEGACY old name, resolved through _shim.install_legacy_finder(), is the new
                                module object (usable on the branch: no file at any old path);
     EXOPIPE_SHIM_CHECK=files   the same through the shim files at the old paths (the final merge commit), and every
                                RUNNABLE old name launches the new module with the baseline result.

Baseline: tests/refactor_shims_baseline.json, recorded from the pre-move tree (refactor branch e969032, a read-only
`git worktree add --detach`) with --rebaseline TREE. Subprocesses run with NUMPY_MADVISE_HUGEPAGE=0, every P3_* and
EXOPIPE3_SITE variable removed, PYTHONPATH = the tree under test, from a scratch directory that is deleted afterwards.

Run:  python tests/test_refactor_shims.py [--rebaseline TREE] [--legacy finder|files]
      (py38_orca_nkosogor: /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3; EXOPIPE_TEST_PYTHON overrides)
      pytest tests/test_refactor_shims.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from exopipe.phase3.core import _shim  # noqa: E402

BASELINE = os.path.join(REPO, 'tests', 'refactor_shims_baseline.json')
PY = os.environ.get('EXOPIPE_TEST_PYTHON', sys.executable)
WORKERS = int(os.environ.get('EXOPIPE_TEST_WORKERS', '8'))
TIMEOUT_S = 600

_IMPORT_CODE = (
    "import importlib, os, sys\n"
    "if os.environ.get('PROBE_FINDER'):\n"
    "    from exopipe.phase3.core import _shim\n"
    "    _shim.install_legacy_finder()\n"
    "try:\n"
    "    m = importlib.import_module(os.environ['PROBE_MODULE'])\n"
    "    if os.environ.get('PROBE_SAME_AS'):\n"
    "        n = importlib.import_module(os.environ['PROBE_SAME_AS'])\n"
    "        parent, _, leaf = os.environ['PROBE_MODULE'].rpartition('.')\n"
    "        via_from = getattr(importlib.import_module(parent), leaf, None)\n"
    "        if m is not n or via_from is not n:\n"
    "            print('IMPORT-FAIL NotSameModule', getattr(m, '__name__', m), 'is not', n.__name__)\n"
    "            sys.exit(0)\n"
    "    print('IMPORT-OK')\n"
    "except BaseException as e:\n"
    "    print('IMPORT-FAIL', type(e).__name__, (str(e).splitlines() or [''])[0])\n"
)


def _env(root, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith('P3_') and k != 'EXOPIPE3_SITE'}
    env.update(NUMPY_MADVISE_HUGEPAGE='0', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=root, COLUMNS='10000')
    env.update(extra)
    return env


def _nice():
    try:
        os.nice(10)
    except OSError:
        pass


def list_modules(root):
    out = []
    for dp, dn, fn in os.walk(os.path.join(root, 'exopipe')):
        dn[:] = sorted(d for d in dn if d != '__pycache__')
        for f in sorted(fn):
            if f.endswith('.py'):
                rel = os.path.relpath(os.path.join(dp, f), root)[:-3].replace(os.sep, '.')
                out.append(rel[:-len('.__init__')] if rel.endswith('.__init__') else rel)
    return sorted(out)


def _probe_one(root, cwd, mod, finder=False, same_as=None):
    extra = dict(PROBE_MODULE=mod)
    if finder:
        extra['PROBE_FINDER'] = '1'
    if same_as:
        extra['PROBE_SAME_AS'] = same_as
    try:
        r = subprocess.run([PY, '-c', _IMPORT_CODE], cwd=cwd, env=_env(root, **extra), capture_output=True,
                           text=True, timeout=TIMEOUT_S, preexec_fn=_nice)
    except subprocess.TimeoutExpired:
        return 'IMPORT-FAIL Timeout'
    lines = [x for x in r.stdout.splitlines() if x.startswith('IMPORT-')]
    if lines:
        return lines[-1]
    tail = (r.stderr.strip().splitlines() or [''])[-1]
    return f'IMPORT-FAIL NoResult rc={r.returncode} {tail}'


def probe_imports(root, modules, finder=False, same_as=None):
    cwd = tempfile.mkdtemp(prefix='refactor_shims_')
    try:
        with ThreadPoolExecutor(WORKERS) as ex:
            res = list(ex.map(lambda m: _probe_one(root, cwd, m, finder, (same_as or {}).get(m)), modules))
        return {m: normalise(r, root) for m, r in zip(modules, res)}
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


def _pairs():
    """(new, old) replacement pairs for normalising output: dotted names, repository paths and file basenames."""
    pairs = []
    for old, new in _shim.LEGACY.items():
        if old == new:
            continue
        pairs.append((new, old))
        pairs.append((_shim.module_path(new), _shim.module_path(old)))
        nb, ob = new.rpartition('.')[2] + '.py', old.rpartition('.')[2] + '.py'
        if nb != ob:
            pairs.append((nb, ob))
    return sorted(pairs, key=lambda p: -len(p[0]))


def normalise(text, root, pairs=()):
    text = text.replace(os.path.realpath(root), '<ROOT>').replace(root, '<ROOT>')
    for new, old in pairs:
        text = re.sub(r'(?<![\w.])' + re.escape(new) + r'(?![\w])', old, text)
    return text


def run_runnables(root, use_old_names=False, pairs=()):
    """{old dotted name: {'cmd', 'rc', 'out'}} for every RUNNABLE with a probe, under its old or current name."""
    cwd = tempfile.mkdtemp(prefix='refactor_shims_run_')

    def one(entry):
        kind, old, probe, _ = entry
        name = old if use_old_names else _shim.new_name(old)
        if kind == '-m':
            cmd = [PY, '-m', name] + list(probe)
        else:
            cmd = [PY, os.path.join(root, _shim.module_path(name))] + list(probe)
        try:
            r = subprocess.run(cmd, cwd=cwd, env=_env(root), capture_output=True, text=True, timeout=TIMEOUT_S,
                               preexec_fn=_nice)
            rc, out = r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            rc, out = 'timeout', ''
        shown = [normalise(c, root) for c in cmd[1:]]       # the command as run (current names)
        return old, dict(cmd=' '.join(['python'] + shown), rc=rc, out=normalise(out, root, pairs))

    try:
        with ThreadPoolExecutor(WORKERS) as ex:
            return dict(ex.map(one, [e for e in _shim.RUNNABLE if e[2] is not None]))
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


def rebaseline(tree):
    tree = os.path.abspath(tree)
    commit = subprocess.run(['git', '-C', tree, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    data = dict(tree=tree, commit=commit, python=PY,
                note='pre-move tree; module names and paths are the OLD ones; repository root shown as <ROOT>',
                imports=probe_imports(tree, list_modules(tree)),
                runnables=run_runnables(tree, use_old_names=True))
    with open(BASELINE, 'w') as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write('\n')
    return data


def _load_baseline():
    with open(BASELINE) as f:
        return json.load(f)


def check_new_names(verbose=True):
    """Check 1. Returns a list of failure strings."""
    base = _load_baseline()['imports']
    cur_mods = list_modules(REPO)
    cur = probe_imports(REPO, cur_mods)
    pairs = _pairs()
    inverse = {}
    for old, new in _shim.LEGACY.items():
        inverse.setdefault(new, []).append(old)
    fails, notes = [], []
    for m in cur_mods:
        olds = [o for o in inverse.get(m, []) if o in base] + ([m] if m in base else [])
        got = cur[m]
        if not olds:
            if got != 'IMPORT-OK':
                fails.append(f'NEW MODULE does not import: {m}: {got}')
            continue
        want = [base[o] for o in olds]
        if all(w == 'IMPORT-OK' for w in want):
            if got != 'IMPORT-OK':
                fails.append(f'REGRESSION {m} (was {", ".join(olds)}: IMPORT-OK): {got}')
        elif got == 'IMPORT-OK':
            notes.append(f'now imports (failed before): {m}')
        elif normalise(got, REPO, pairs) not in [w for w in want]:
            fails.append(f'CHANGED FAILURE {m}: {got} (baseline {want})')
    for b in base:
        if b not in cur and b not in _shim.LEGACY:
            fails.append(f'LOST module {b}: not in the tree and not in _shim.LEGACY')
    if verbose:
        n_ok = sum(v == 'IMPORT-OK' for v in cur.values())
        print(f'[1] new names: {len(cur_mods)} modules, {n_ok} import; {len(_shim.LEGACY)} LEGACY entries; '
              f'{len(fails)} failure(s)')
        for x in notes + fails:
            print('    ' + x)
    return fails


def check_runnables(verbose=True):
    """Check 2. Returns a list of failure strings."""
    base = _load_baseline()['runnables']
    cur = run_runnables(REPO, pairs=_pairs())
    fails = []
    for old, b in sorted(base.items()):
        c = cur.get(old)
        if c is None:
            fails.append(f'{old}: not run')
            continue
        first = (c['out'].strip().splitlines() or [''])[0][:110]
        same = c['rc'] == b['rc'] and c['out'] == b['out']
        if verbose:
            print(f'    {"OK  " if same else "DIFF"} rc={c["rc"]} (baseline {b["rc"]})  {c["cmd"]}  | {first}')
        if not same:
            fails.append(f'{old}: rc {c["rc"]} vs {b["rc"]}; output differs:\n' + _diff(b['out'], c['out']))
    if verbose:
        print(f'[2] runnables: {len(base)} compared with the pre-move tree; {len(fails)} failure(s)')
        for x in fails:
            print('    ' + x)
    return fails


def _diff(a, b):
    import difflib
    return ''.join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), 'pre-move', 'now', n=1))[:3000]


def check_old_names(mode, verbose=True):
    """Check 3 (to be shimmed at cutover). mode: 'finder' or 'files'. Returns a list of failure strings."""
    olds = sorted(_shim.LEGACY)
    res = probe_imports(REPO, olds, finder=(mode == 'finder'), same_as=dict(_shim.LEGACY))
    fails = [f'{o} -> {_shim.LEGACY[o]}: {r}' for o, r in res.items() if r != 'IMPORT-OK']
    if mode == 'files':
        base = _load_baseline()['runnables']
        cur = run_runnables(REPO, use_old_names=True)
        for old, b in sorted(base.items()):
            if cur[old]['rc'] != b['rc'] or cur[old]['out'] != b['out']:
                fails.append(f'launch {old}: rc {cur[old]["rc"]} vs {b["rc"]}\n' + _diff(b['out'], cur[old]['out']))
    if verbose:
        print(f'[3] old names ({mode}): {len(olds)} LEGACY names; {len(fails)} failure(s)')
        for x in fails:
            print('    ' + x)
    return fails


# ---- pytest entry points --------------------------------------------------------------------------------------------

def test_new_names_import():
    assert not check_new_names(verbose=False)


def test_runnables_match_premove():
    assert not check_runnables(verbose=False)


def test_old_names_to_be_shimmed_at_cutover():
    mode = os.environ.get('EXOPIPE_SHIM_CHECK', '')
    if mode not in ('finder', 'files'):
        import pytest
        pytest.skip('old-name check runs only with EXOPIPE_SHIM_CHECK=finder|files (to be shimmed at cutover)')
    assert not check_old_names(mode, verbose=False)


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--rebaseline', metavar='TREE', help='record the baseline from this pre-move tree and stop')
    ap.add_argument('--legacy', choices=('finder', 'files'), default=os.environ.get('EXOPIPE_SHIM_CHECK') or None,
                    help='also run the old-name check (default: skipped; EXOPIPE_SHIM_CHECK)')
    a = ap.parse_args(argv)
    if a.rebaseline:
        d = rebaseline(a.rebaseline)
        print(f'baseline: {len(d["imports"])} modules, {len(d["runnables"])} runnables from {d["tree"]} '
              f'({d["commit"][:12]}) -> {BASELINE}')
        return 0
    fails = check_new_names() + check_runnables()
    if a.legacy:
        fails += check_old_names(a.legacy)
    else:
        print('[3] old names: SKIPPED (to be shimmed at cutover; --legacy finder|files or EXOPIPE_SHIM_CHECK)')
    print('RESULT:', 'PASS' if not fails else f'FAIL ({len(fails)})')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
