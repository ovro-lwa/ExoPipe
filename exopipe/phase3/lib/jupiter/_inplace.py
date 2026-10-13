"""In-place driver of `exopipe3 jupiter` (refactor window, DESIGN 11.1): runs ONE function of a production module, unchanged,
in a child process started by core.legacy.run under the interpreter that module runs under today.

    <python> _inplace.py --module /lustre/gh/main/clv15/phase3_jupiter_measure.py --call stage_extract \
        --args '["05h_2024-12-29", "/out/products", "/work"]' [--globals '{"SL": "/res/sublst"}'] [--argv '[...]'] \
        [--test-stage] [--hash-roots '["/root", ...]'] --result /work/result.json

1. The module's own directory goes first on sys.path (the V2 modules import their siblings: jgeom, jsite_lib, jint_disclose),
   the module is imported from its path under its own name (so its `if __name__ == "__main__"` block does NOT run) and
   sys.argv is set to [module path] + --argv (for --call main).
2. --globals sets module globals AFTER the import: 'NAME' on the module, 'other.NAME' on an already imported module (e.g.
   'phase3_jupiter_measure.SL' for jupv2_masters, which copies it at import). An unknown name is an error (never a silent
   no-op). The step passes globals ONLY when a config value differs from the module literal (parity roots); a production
   run passes none.
3. Exit mapping without log parsing: SystemExit(str) starting 'REFUSED' -> exit 2 (reason = the text); SystemExit(1) with
   --test-stage (a content-test stage: stage_test, jupv2_masters test, jblue_test, test_cell, content_test, jsite_test) ->
   exit 4 (test failed); SystemExit(2) -> exit 2 (the backfill's copy-budget refusal); SystemExit(0/None) or a normal return
   -> exit 0; RuntimeError whose text starts 'REFUSING' (phase3_tier_rules) -> exit 2; anything else -> exit 1.
4. --result FILE (JSON): kind (ok, refused, test-failed, error), reason, the sha256 of every source file loaded from the
   --hash-roots (the V2 code tree and phase0 have no git commit: code identity per file, DESIGN 11.5), wall seconds.

This file imports nothing from exopipe (it runs under py310_orca too) and runs on Python 3.6+.
"""
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
import traceback


def _arg(argv, name, default=None):
    if name in argv:
        i = argv.index(name)
        return argv[i + 1]
    return default


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def _loaded_files(roots):
    roots = tuple(os.path.realpath(r).rstrip('/') + '/' for r in roots if r)
    out = {}
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None)
        if not f or not f.endswith('.py'):
            continue
        rf = os.path.realpath(f)
        if roots and rf.startswith(roots):
            try:
                out[rf] = _sha256(rf)
            except OSError:
                pass
    return out


def _set_globals(mod, gl):
    for key, value in gl.items():
        target, _, attr = key.rpartition('.')
        m = mod if not target else sys.modules.get(target)
        if m is None:
            m = importlib.import_module(target)
        if not hasattr(m, attr):
            raise KeyError('module global %s does not exist in %s - refusing to set it (a silent no-op would leave the '
                           'production value in force)' % (key, getattr(m, '__file__', target)))
        setattr(m, attr, value)


def main(argv):
    module = os.path.abspath(_arg(argv, '--module'))
    call = _arg(argv, '--call', 'main')
    args = json.loads(_arg(argv, '--args', '[]'))
    kwargs = json.loads(_arg(argv, '--kwargs', '{}'))
    child_argv = json.loads(_arg(argv, '--argv', '[]'))
    gl = json.loads(_arg(argv, '--globals', '{}'))
    roots = json.loads(_arg(argv, '--hash-roots', '[]'))
    result_path = _arg(argv, '--result')
    test_stage = '--test-stage' in argv
    t0 = time.time()
    res = {'module': module, 'call': call, 'kind': 'error', 'reason': '', 'globals_set': sorted(gl)}
    try:
        sys.path.insert(0, os.path.dirname(module))
        name = os.path.splitext(os.path.basename(module))[0]
        spec = importlib.util.spec_from_file_location(name, module)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        sys.argv = [module] + [str(a) for a in child_argv]
        spec.loader.exec_module(mod)
        _set_globals(mod, gl)
        fn = getattr(mod, call)
        ret = fn(*args, **kwargs)
        res.update(kind='ok', returned=repr(ret)[:2000] if ret is not None else None)
    except SystemExit as e:
        code = e.code
        if code is None or code == 0:
            res.update(kind='ok')
        elif isinstance(code, str):
            sys.stderr.write(code + '\n')
            res.update(kind='refused' if code.lstrip().upper().startswith('REFUS') else 'error', reason=code[:2000])
        elif code == 1 and test_stage:
            res.update(kind='test-failed', reason='content test failed (the stage exited 1)')
        elif code == 2:
            res.update(kind='refused', reason='exit 2 (the module refused; see the log)')
        else:
            res.update(kind='error', reason='exit %r' % (code,))
    except RuntimeError as e:
        txt = str(e)
        traceback.print_exc()
        res.update(kind='refused' if txt.startswith('REFUSING') else 'error', reason=txt[:2000])
    except BaseException as e:      # noqa: B902 - every outcome is recorded
        traceback.print_exc()
        res.update(kind='error', reason=('%s: %s' % (type(e).__name__, e))[:2000])
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    res['code_sha256'] = _loaded_files(roots + [os.path.dirname(module)])
    res['wall_s'] = round(time.time() - t0, 1)
    if result_path:
        tmp = result_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(res, f, indent=1, sort_keys=True)
        os.replace(tmp, result_path)
    return {'ok': 0, 'refused': 2, 'test-failed': 4}.get(res['kind'], 1)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
