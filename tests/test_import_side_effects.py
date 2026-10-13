#!/usr/bin/env python3
"""STANDING TEST — import-side-effect freedom, repo-wide.

Adopted per Gregg's directive (fix_ledger 2026-08-12, unguarded-writer-headline):
the structural closure for the 2026-08-12 unguarded-writer incident is a
strace-verified import test in the exopipe tree, standing, not a one-time
sweep. Every module in the exopipe package must be side-effect-free at
import; `python module.py` script behavior lives under a __main__ guard.

Two layers:

  1. AST layer (fast, always on): no module-level statement may call into
     the write/exec deny-list (file writes, renames, deletes, makedirs,
     subprocess, sys.exit, shutil ops) or read sys.argv mutably except the
     ratified save/clobber/restore import-protection pattern. Process-local
     configuration (matplotlib.use, warnings filters, logging.basicConfig,
     sys.stdout.reconfigure, os.environ tweaks, sys.path.insert) is allowed.

  2. strace layer (slow, opt-in: EXOPIPE_IMPORT_AUDIT=1): every module is
     imported in a fresh interpreter from a scratch CWD under
     `strace -f -e trace=openat,creat,rename*,unlink*,mkdir*`; FAIL on any
     write-mode open / rename / unlink / mkdir outside __pycache__, /tmp,
     and library config/cache zones. Modules whose imports need packages
     absent from the running env are reported UNTESTED-IMPORT (with reason),
     never silently skipped.

Run standalone:  python tests/test_import_side_effects.py [--strace]
Under pytest:    pytest tests/test_import_side_effects.py
                 (strace layer gated on EXOPIPE_IMPORT_AUDIT=1)
"""
import ast
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(REPO, 'exopipe')

# Calls that must never execute at module level (write/exec surface).
DENY_CALLS = {
    'makedirs', 'mkdir', 'chdir', 'remove', 'unlink', 'rename', 'renames',
    'rmtree', 'copy', 'copy2', 'copyfile', 'move', 'symlink',
    'run', 'call', 'check_call', 'check_output', 'Popen', 'system',
    'to_csv', 'to_pickle', 'savefig', 'writeto', 'exit', 'save_state',
    'write', 'writerow', 'writerows', 'dump', 'main',
}
# dotted forms that are unambiguous (leaf 'replace' alone is str.replace)
DENY_DOTTED = {'os.replace'}
# Process-local configuration ratified to run at import (guard sweep
# 2026-08-12): backend/warning/logging/env/path setup, no file writes.
ALLOW_CALLS = {
    'use', 'filterwarnings', 'simplefilter', 'basicConfig', 'reconfigure',
    'insert', 'setdefault', 'getLogger', 'set', 'register',
}
# (relpath, callname) exceptions reviewed and accepted with Gregg's sweep.
ACCEPTED = {
    ('exopipe/phase2/flow_cascade_dewarp.py', 'print'),   # GPU-DEBUG banner
    ('exopipe/common/phase2_config.py', '_resolve_gittag'),  # read-only git rev-parse
}

ALLOW_WRITE_PATH = re.compile(
    r'/__pycache__/|^/tmp/|^/dev/|^/proc/|^/sys/|/\.cache/|/\.astropy/'
    r'|/\.casa|/\.matplotlib|\.pyc$'
    # python-casacore opens its measures tables with an O_RDWR lock file
    # inside site-packages/casadata — library read-lock, not a pipeline write
    r'|/site-packages/casadata/.*/table\.lock$')
WRITE_FLAGS = re.compile(r'O_WRONLY|O_RDWR|O_CREAT|O_TRUNC|O_APPEND')


# Files whose PURPOSE is to act when executed or imported (not importable library modules):
# __main__.py runs `exopipe3` under `python -m exopipe.phase3`; the parity sitecustomize installs the write guard in
# Python subprocesses by design (core/parity_sitecustomize).
_ENTRY_FILES = ('__main__.py', os.path.join('parity_sitecustomize', 'sitecustomize.py'))


def _modules():
    out = []
    for dp, dn, fn in os.walk(PKG):
        dn[:] = [d for d in dn if d != '__pycache__']
        for f in sorted(fn):
            p = os.path.join(dp, f)
            if f.endswith('.py') and not any(p.endswith(os.sep + e) for e in _ENTRY_FILES):
                out.append(p)
    return out


def _is_main_guard(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == '__name__')


def _call_names(node):
    """Yield (leaf, dotted) for every Call in node."""
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            parts = []
            f = n.func
            while isinstance(f, ast.Attribute):
                parts.append(f.attr)
                f = f.value
            if isinstance(f, ast.Name):
                parts.append(f.id)
            if parts:
                yield parts[0], '.'.join(reversed(parts))


def ast_violations():
    """Return [(relpath, lineno, snippet)] of denied top-level calls."""
    bad = []
    for path in _modules():
        rel = os.path.relpath(path, REPO)
        src = open(path).read()
        tree = ast.parse(src, path)
        lines = src.splitlines()
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if _is_main_guard(node):
                continue
            for leaf, dotted in _call_names(node):
                if ((leaf in DENY_CALLS or dotted in DENY_DOTTED)
                        and (rel, leaf) not in ACCEPTED):
                    bad.append((rel, node.lineno,
                                lines[node.lineno - 1].strip()[:100], leaf))
    return bad


def test_ast_no_toplevel_side_effects():
    bad = ast_violations()
    assert not bad, 'module-level side-effect calls:\n' + '\n'.join(
        f'  {r}:{l} [{c}] {s}' for r, l, s, c in bad)


def strace_audit(python=None, scratch='/tmp/exopipe_import_audit'):
    """Import every module under strace; return dict mod -> (status, detail).

    status: PASS | WRITE-FAIL | IMPORT-ERROR | UNTESTED-IMPORT
    """
    python = python or os.environ.get('EXOPIPE_TEST_PYTHON', sys.executable)
    home = os.path.join(scratch, 'home')
    os.makedirs(home, exist_ok=True)
    results = {}
    for path in _modules():
        rel = os.path.relpath(path, REPO)
        mod = rel[:-3].replace('/', '.')
        if mod.endswith('.__init__'):
            mod = mod[:-9]
        tr = os.path.join(scratch, 'trace_' + mod + '.txt')
        env = dict(os.environ, PYTHONPATH=REPO, HOME=home,
                   MPLCONFIGDIR=os.path.join(home, 'mpl'),
                   XDG_CACHE_HOME=os.path.join(home, 'xdg'),
                   # user-site packages must stay importable under fake HOME
                   PYTHONUSERBASE=os.path.expanduser('~/.local'))
        p = subprocess.run(
            ['strace', '-f', '-e',
             'trace=openat,creat,rename,renameat,renameat2,unlink,unlinkat,'
             'mkdir,mkdirat', '-o', tr, '-s', '300',
             python, '-c', 'import ' + mod],
            cwd=scratch, env=env, capture_output=True, text=True, timeout=600)
        bad = []
        if os.path.exists(tr):
            for line in open(tr, errors='replace'):
                if ' = -1 ' in line and 'EEXIST' not in line:
                    continue
                m = re.search(r'(openat|creat|rename\w*|unlink\w*|mkdir\w*)'
                              r'\((.*)\)', line)
                if not m:
                    continue
                call, args = m.group(1), m.group(2)
                paths = re.findall(r'"([^"]+)"', args)
                if not paths:
                    continue
                if call in ('openat', 'creat'):
                    if call == 'openat' and not WRITE_FLAGS.search(args):
                        continue
                    if not ALLOW_WRITE_PATH.search(paths[-1]):
                        bad.append(line.strip()[:200])
                elif any(not ALLOW_WRITE_PATH.search(t) for t in paths):
                    bad.append(line.strip()[:200])
        if p.returncode != 0:
            err = (p.stderr.strip().splitlines() or ['?'])[-1][:200]
            kind = ('UNTESTED-IMPORT'
                    if ('ModuleNotFoundError' in p.stderr
                        or 'ImportError' in p.stderr) else 'IMPORT-ERROR')
            results[mod] = (kind, err)
        elif bad:
            results[mod] = ('WRITE-FAIL', ' | '.join(sorted(set(bad))[:5]))
        else:
            results[mod] = ('PASS', '')
    return results


def test_strace_import_audit():
    if os.environ.get('EXOPIPE_IMPORT_AUDIT') != '1':
        try:
            import pytest
            pytest.skip('slow strace layer: set EXOPIPE_IMPORT_AUDIT=1')
        except ImportError:
            return
    results = strace_audit()
    fails = {m: v for m, v in results.items()
             if v[0] in ('WRITE-FAIL', 'IMPORT-ERROR')}
    untested = {m: v for m, v in results.items() if v[0] == 'UNTESTED-IMPORT'}
    for m, (s, d) in sorted(untested.items()):
        print(f'UNTESTED-IMPORT {m}: {d}')
    assert not fails, '\n'.join(f'{m}: {v[0]} {v[1]}'
                                for m, v in sorted(fails.items()))


def main():
    bad = ast_violations()
    for r, l, s, c in bad:
        print(f'AST-FAIL {r}:{l} [{c}] {s}')
    print(f'AST layer: {"FAIL" if bad else "PASS"} '
          f'({len(_modules())} modules)')
    rc = 1 if bad else 0
    if '--strace' in sys.argv:
        results = strace_audit()
        counts = {}
        for m, (s, d) in sorted(results.items()):
            counts[s] = counts.get(s, 0) + 1
            if s != 'PASS':
                print(f'{s:16s} {m}  {d}')
        print('strace layer:', counts)
        if counts.get('WRITE-FAIL') or counts.get('IMPORT-ERROR'):
            rc = 1
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
