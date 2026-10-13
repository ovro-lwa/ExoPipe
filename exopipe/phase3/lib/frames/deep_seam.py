"""The frames/search seam of the deep engine: the proposed ``frames_only`` keyword of run_deep_cell and its neutrality proof.

Decision of the refactor chat (2026-10-13 ~04:30 UTC, _CROSSCHECK.md group 14): ``search detect`` stays the deep frame
producer during the refactor window; ``frames --tier deep`` is for frame-only rebuilds and is never chained before search;
the ``frames_only`` keyword goes into lib/search/deep_search.run_deep_cell as ONE reviewed commit by the search owner, and
it must be byte-neutral for the default call. This module (frames-owned) carries:

- ``propose(src)``: the keyword as a list of anchored edits of the engine source (each anchor must occur exactly once, so
  an edit can never land in the wrong place; the result is returned, never written into lib/search). With
  frames_only=True: one Stokes per call; everything up to the sigma stacks runs unchanged (setup, master and membership
  gates, flow cache, x2 flows, every admission gate, LOO selection, residual build, sigma stack); the search-only
  per-cell inputs (parent catalogue, I structure map, V sanity gate, extraction-refusal setup) are skipped; in the date
  loop only the frame blocks run (frame, sibling, archive copies) and the loop continues before the search; the call
  returns a dict {dates, frames{S}, siblings{S}, sigstack{S}, refusals, pbcorr_missing, policy_skip} instead of 0 / the
  candidate count; the P3-POLICY-SKIP returns give {'policy_skip': reason}.
- ``default_path_identical(patched, original)``: the proof of neutrality - the patched run_deep_cell with
  ``frames_only`` folded to False (``if frames_only`` -> its else branch, ``if not frames_only`` -> its body,
  ``X if frames_only else Y`` -> Y, the keyword and the docstring removed) has exactly the AST of the original
  function. Every statement of the default call is therefore the original statement in the original order: the default
  products cannot change (the runtime test frames.md:472 describes - unchanged main vs branch on one cell - is still owed
  by the search owner's commit).

    python -m exopipe.phase3.lib.frames.deep_seam --check            # propose on the worktree engine, prove neutrality
    python -m exopipe.phase3.lib.frames.deep_seam --write /tmp/x.py  # the patched engine source, for the search owner
    python -m exopipe.phase3.lib.frames.deep_seam --verify-engine    # engine already has frames_only: prove it against
                                                                     # the newest committed version without the keyword
"""
import ast
import os
import subprocess
import sys

KEYWORD = 'frames_only'
FUNC = 'run_deep_cell'

_HELPER = '''

def _frames_only_result(lst, scope, dates, fo, run_root, refdir, policy_skip=None):
    """Result of run_deep_cell(frames_only=True) (exopipe3 frames --tier deep reads it; lib/frames/deep_engine.py): the
    frames and pbcorr siblings written in this call, the cell sigma stacks present in the run root, the refusal records of
    the cell and the pbcorr-missing record. Never called by the default (frames_only=False) path."""
    out = {'dates': list(dates), 'policy_skip': policy_skip, 'frames': {}, 'siblings': {}, 'sigstack': {},
           'refusals': [], 'pbcorr_missing': None}
    for st in ('I', 'V'):
        fr = [p for p in fo.get(st, []) if os.path.exists(p)]
        out['frames'][st] = fr
        out['siblings'][st] = [p.replace('_redo_resid_', '_redo_resid_pbcorr_') for p in fr
                               if os.path.exists(p.replace('_redo_resid_', '_redo_resid_pbcorr_'))]
        if run_root:
            ss = f'{run_root}/sigstack/{lst}_{scope}_{st}_sigstack.fits'
            out['sigstack'][st] = ss if os.path.exists(ss) else None
    if refdir:
        out['refusals'] = sorted(glob.glob(f'{refdir}/{lst}_{scope}_*_excluded.csv'))
    pm = f"{os.environ.get('P3_STAGING_ROOT', '.')}/pbcorr_missing_{lst}_{scope}.csv"
    out['pbcorr_missing'] = pm if os.path.exists(pm) else None
    print(f'  {lst} {scope}: FRAMES-ONLY - ' + ', '.join(f'{st} {len(out["frames"][st])} frames' for st in ('I', 'V'))
          + (f' (policy skip: {policy_skip})' if policy_skip else ''), flush=True)
    return out

'''

_DOC_ADD = '''
    frames_only=True (exopipe3 frames --tier deep, frame-only rebuilds; refactor 2026-10-13, crosscheck group 14): one
    Stokes per call; every gate, the flows, the LOO selection, the residuals and the sigma stack run as in the default call;
    in the date loop only the frame blocks run (frame, pbcorr sibling, archive copies), the search is skipped, and the call
    returns _frames_only_result(...) (a dict) instead of the candidate count. The default call (frames_only=False) is
    statement-for-statement the call without the keyword (proof: lib/frames/deep_seam.default_path_identical)."""'''


def _edits():
    """(name, anchor, replacement): each anchor must occur exactly once in the engine source."""
    I_PUB = ("                _archive_frame_publish(ir_path.replace('_redo_resid_', '_redo_resid_pbcorr_'), lst, scope, d, "
             "'I', cadence, f'{lst} {scope}')\n")
    V_PUB = ("                _archive_frame_publish(vr_path.replace('_redo_resid_', '_redo_resid_pbcorr_'), lst, scope, d, "
             "'V', cadence, f'{lst} {scope}')\n")
    return [
        ('signature',
         "def run_deep_cell(lst, scope, label='PROD', included_dates=None, stokes='IV',\n                  cadence='deep'):\n",
         "def run_deep_cell(lst, scope, label='PROD', included_dates=None, stokes='IV',\n"
         "                  cadence='deep', frames_only=False):\n"),
        ('docstring',
         "    the frame source is the same chain (Gregg 2026-09-08: 'identical to deep').\"\"\"\n",
         "    the frame source is the same chain (Gregg 2026-09-08: 'identical to deep').\n" + _DOC_ADD + "\n"),
        ('one-stokes',
         "    t0 = time.time()\n    if cadence not in ('deep', '10min'):\n",
         "    t0 = time.time()\n    if frames_only and stokes not in ('I', 'V'):\n"
         "        raise ValueError(f'frames_only=True needs one Stokes (I or V), got {stokes!r}')\n"
         "    if cadence not in ('deep', '10min'):\n"),
        ('policy-skip-epochs',
         "P3-POLICY-SKIP only {len(dates)} epochs (minimum {_minE}, cadence {cadence})'); return 0\n",
         "P3-POLICY-SKIP only {len(dates)} epochs (minimum {_minE}, cadence {cadence})'); return "
         "(_frames_only_result(lst, scope, [], {'I': [], 'V': []}, run_root, _refdir, policy_skip=f'only {len(dates)} "
         "epochs (minimum {_minE}, cadence {cadence})') if frames_only else 0)\n"),
        ('policy-skip-flows',
         "P3-POLICY-SKIP only {len(dates)} flows (minimum {_minE}, cadence {cadence})'); return 0\n",
         "P3-POLICY-SKIP only {len(dates)} flows (minimum {_minE}, cadence {cadence})'); return "
         "(_frames_only_result(lst, scope, [], {'I': [], 'V': []}, run_root, _refdir, policy_skip=f'only {len(dates)} "
         "flows (minimum {_minE}, cadence {cadence})') if frames_only else 0)\n"),
        ('incremental-nothing',
         "print(f'  {lst} {scope}: INCREMENTAL - nothing admitted; done'); return 0\n",
         "print(f'  {lst} {scope}: INCREMENTAL - nothing admitted; done'); return "
         "(_frames_only_result(lst, scope, [], {'I': [], 'V': []}, run_root, _refdir) if frames_only else 0)\n"),
        ('i-frame-continue', I_PUB,
         I_PUB + "            if frames_only:\n                _fo['I'].append(ir_path)\n                continue\n"),
        ('i-chain-skip',
         "        if 'I' in stokes and d in resid_I and _xfI is None:\n",
         "        if not frames_only and 'I' in stokes and d in resid_I and _xfI is None:\n"),
        ('v-frame-continue', V_PUB,
         V_PUB + "            if frames_only:\n                _fo['V'].append(vr_path)\n                continue\n"),
        ('return-after-loop',
         "    # ── I 2-D BRIGHT-PARENT STAGE.",
         "    if frames_only:\n        return _frames_only_result(lst, scope, dates, _fo, run_root, _refdir)\n"
         "    # ── I 2-D BRIGHT-PARENT STAGE."),
        ('helper', "\n\ndef run_deep_cell(", _HELPER.rstrip('\n') + "\n\n\ndef run_deep_cell("),
    ]


# the search-only per-cell block (parent catalogue, I structure map, V sanity gate, extraction-refusal setup) between
# these two anchors is wrapped in `if not frames_only:` (re-indented; the else branch sets what frames_only needs)
_BLOCK_START = "    # ── R-ROUND2-RATIFIED shared per-cell inputs"
_BLOCK_END = "    for d in dates:\n        ot = _epoch_time(ebase, d, scope)\n"


def _once(src, anchor, name):
    n = src.count(anchor)
    if n != 1:
        raise ValueError(f'seam edit {name!r}: anchor found {n} times (must be exactly once) - the engine changed; '
                         f'refresh lib/frames/deep_seam.py')
    return src.index(anchor)


def propose(src):
    """The engine source with the frames_only keyword (see the module docstring). Raises ValueError when an anchor is
    not found exactly once or the source already has the keyword."""
    tree = ast.parse(src)
    if has_keyword(tree):
        raise ValueError(f'{FUNC} already takes {KEYWORD}')
    i0 = _once(src, _BLOCK_START, 'block-start')
    i1 = _once(src, _BLOCK_END, 'block-end')
    if not i0 < i1:
        raise ValueError('seam edit block: start after end')
    block = src[i0:i1]
    if '"""' in block or "'''" in block:
        raise ValueError('seam edit block: a triple-quoted string inside the block cannot be re-indented safely')
    ind = ''.join(('    ' + ln if ln.strip() else ln) for ln in block.splitlines(True))
    src = (src[:i0] + "    if not frames_only:\n" + ind
           + "    else:\n        skip_V_dates = set()      # frames only: no V search, no V sanity gate\n"
           + "    if frames_only:\n        _fo = {'I': [], 'V': []}\n" + src[i1:])
    for name, anchor, repl in _edits():
        j = _once(src, anchor, name)
        src = src[:j] + repl + src[j + len(anchor):]
    return src


# ---------------------------------------------------------------------------------------------------------------------
# proof of neutrality
# ---------------------------------------------------------------------------------------------------------------------

def has_keyword(tree):
    fn = _func(tree)
    return fn is not None and KEYWORD in [a.arg for a in fn.args.args + fn.args.kwonlyargs]


def _func(tree, name=FUNC):
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


def _kw(node):
    """A constant that came from the keyword (or was folded from one): only those decide a branch."""
    return isinstance(node, ast.Constant) and getattr(node, '_from_kw', False)


def _mk(value, like):
    c = ast.copy_location(ast.Constant(value=value, kind=None), like)
    c._from_kw = True
    return c


class _Fold(ast.NodeTransformer):
    """Fold the keyword to a constant and remove the branches it decides. Only constants derived from the keyword are
    folded (a literal such as the 0 of `x or 0` is never touched)."""

    def __init__(self, name, value):
        self.name, self.value = name, value

    def visit_Name(self, node):
        if node.id == self.name and isinstance(node.ctx, ast.Load):
            return _mk(self.value, node)
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and _kw(node.operand):
            return _mk(not node.operand.value, node)
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        vals = node.values
        is_and = isinstance(node.op, ast.And)
        # a keyword constant FIRST decides the whole expression when it short-circuits (False for and, True for or);
        # a non-short-circuiting keyword constant (True in and, False in or) is dropped
        if _kw(vals[0]) and (bool(vals[0].value) is not is_and):
            return vals[0]
        keep = [v for v in vals if not (_kw(v) and bool(v.value) is is_and)]
        if not keep:
            return _mk(is_and, node)
        if len(keep) == 1:
            return keep[0]
        node.values = keep
        return node

    def visit_IfExp(self, node):
        self.generic_visit(node)
        if _kw(node.test):
            return node.body if node.test.value else node.orelse
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if _kw(node.test):
            return node.body if node.test.value else node.orelse
        return node


def _strip(fn):
    """Remove the keyword argument (and its default) and the docstring of the function node."""
    a = fn.args
    names = [x.arg for x in a.args]
    if KEYWORD in names:
        i = names.index(KEYWORD)
        nd = len(a.defaults)
        first_default = len(a.args) - nd
        if i >= first_default:
            del a.defaults[i - first_default]
        del a.args[i]
    kn = [x.arg for x in a.kwonlyargs]
    if KEYWORD in kn:
        i = kn.index(KEYWORD)
        del a.kwonlyargs[i]
        del a.kw_defaults[i]
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) \
            and isinstance(fn.body[0].value.value, str):
        fn.body = fn.body[1:]
    return fn


def default_path_identical(patched_src, original_src):
    """(ok, detail): the patched run_deep_cell specialised at frames_only=False has the AST of the original function."""
    pf = _func(ast.parse(patched_src))
    of = _func(ast.parse(original_src))
    if pf is None or of is None:
        return False, f'{FUNC} not found'
    if not has_keyword(ast.parse(patched_src)):
        return False, f'the patched {FUNC} has no {KEYWORD} keyword'
    pf = _strip(_Fold(KEYWORD, False).visit(pf))
    of = _strip(of)
    a, b = ast.dump(pf), ast.dump(of)
    if a == b:
        return True, (f'{FUNC} with {KEYWORD}=False folded is AST-identical to the original ({len(pf.body)} top-level '
                      f'statements): the default call executes the original statements in the original order')
    # first differing top-level statement, for the report
    for k, (x, y) in enumerate(zip(pf.body, of.body)):
        if ast.dump(x) != ast.dump(y):
            return False, f'default path differs at top-level statement {k} (patched line {x.lineno}, original {y.lineno})'
    return False, f'default path differs in length ({len(pf.body)} vs {len(of.body)} statements) or in the signature'


_SEARCH_CALLS = ('run_identification_i', 'run_identification_v', 'process_image', '_v110_cell_verdicts',
                 '_read_gaul_parents', '_append_locked', 'apply_veto', '_size_test')
_TERMINAL = (ast.Return, ast.Raise, ast.Continue, ast.Break)


def _prune(stmts):
    """Statements of a block that can run: up to and including the first return / raise / continue / break."""
    out = []
    for st in stmts:
        out.append(st)
        if isinstance(st, _TERMINAL):
            break
    return out


def _reachable(stmts, calls, returns, top=True):
    """Collect the calls and (top-level function) returns reachable through ``stmts``, pruning dead tails."""
    for st in _prune(stmts):
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _reachable(st.body, calls, [], top=False)      # nested code: its calls count, its returns are its own
            continue
        if isinstance(st, ast.Return) and top:
            returns.append(st)
        for fld in ('body', 'orelse', 'finalbody'):
            sub = getattr(st, fld, None)
            if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                _reachable(sub, calls, returns, top)
        for h in getattr(st, 'handlers', []) or []:
            _reachable(h.body, calls, returns, top)
        # expressions of this statement (not its nested statement lists)
        for fld, val in ast.iter_fields(st):
            if fld in ('body', 'orelse', 'finalbody', 'handlers'):
                continue
            for n in (val if isinstance(val, list) else [val]):
                if isinstance(n, ast.AST):
                    for m in ast.walk(n):
                        if isinstance(m, ast.Call):
                            calls.add(m.func.id if isinstance(m.func, ast.Name) else getattr(m.func, 'attr', ''))


def frames_only_path_ok(patched_src):
    """(ok, detail): with frames_only folded to True no search stage is reachable (no detection, veto, parent catalogue,
    v110 stage or ledger append) and every return of the function returns the result dict builder."""
    pf = _strip(_Fold(KEYWORD, True).visit(_func(ast.parse(patched_src))))
    calls, rets = set(), []
    _reachable(pf.body, calls, rets)
    bad = sorted(c for c in calls if c in _SEARCH_CALLS)
    nondict = [r.lineno for r in rets if not (isinstance(r.value, ast.Call)
                                              and getattr(r.value.func, 'id', '') == '_frames_only_result')]
    if bad or nondict:
        return False, '; '.join(x for x in ((f'reachable search calls {bad}' if bad else ''),
                                            (f'returns without the result dict at lines {nondict}' if nondict else '')) if x)
    return True, f'{len(rets)} reachable returns, all _frames_only_result(...); no search stage reachable'


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def engine_path():
    from exopipe.phase3.lib.frames import deep_engine
    return deep_engine.engine_source()


def committed_without_keyword(path):
    """Source of the newest committed version of ``path`` whose run_deep_cell lacks the keyword (git, read only)."""
    repo = os.path.dirname(path)
    rel = subprocess.run(['git', '-C', repo, 'ls-files', '--full-name', path], capture_output=True, text=True).stdout.strip()
    revs = subprocess.run(['git', '-C', repo, 'log', '--format=%H', '-n', '50', '--', path], capture_output=True,
                          text=True).stdout.split()
    for r in revs:
        src = subprocess.run(['git', '-C', repo, 'show', f'{r}:{rel}'], capture_output=True, text=True).stdout
        if src and not has_keyword(ast.parse(src)):
            return r, src
    return None, None


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog='python -m exopipe.phase3.lib.frames.deep_seam', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--engine', help='engine source (default: the worktree lib/search/deep_search.py)')
    p.add_argument('--check', action='store_true', help='propose the keyword and prove neutrality (writes nothing)')
    p.add_argument('--write', metavar='PATH', help='write the patched engine source to PATH (never lib/search itself)')
    p.add_argument('--verify-engine', action='store_true',
                   help='the engine already has the keyword: prove it against the newest committed version without it')
    a = p.parse_args(argv)
    path = a.engine or engine_path()
    src = open(path).read()
    if a.verify_engine:
        rev, orig = committed_without_keyword(path)
        if orig is None:
            print('no committed version without the keyword found')
            return 1
        ok, det = default_path_identical(src, orig)
        ok2, det2 = frames_only_path_ok(src)
        print(f'{"PASS" if ok else "FAIL"} default path vs {rev[:10]}: {det}')
        print(f'{"PASS" if ok2 else "FAIL"} frames_only path: {det2}')
        return 0 if ok and ok2 else 2
    patched = propose(src)
    compile(patched, path, 'exec')
    ok, det = default_path_identical(patched, src)
    ok2, det2 = frames_only_path_ok(patched)
    print(f'{"PASS" if ok else "FAIL"} default path: {det}')
    print(f'{"PASS" if ok2 else "FAIL"} frames_only path: {det2}')
    if a.write:
        real = os.path.realpath(a.write)
        if real == os.path.realpath(path):
            print('refused: --write must not overwrite the engine (lib/search is the search owner\'s file)')
            return 2
        with open(a.write, 'w') as f:
            f.write(patched)
        print(f'patched source written to {a.write}')
    return 0 if ok and ok2 else 2


if __name__ == '__main__':
    sys.exit(main())
