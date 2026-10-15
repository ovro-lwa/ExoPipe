# exopipe3 PARITY WRITE GUARD for every Python process of a parity run (DESIGN 11.1; modelled on the sub-LST frozen-code
# guard, exopipe_ops/sublst_frozen_guard/sitecustomize.py). `exopipe3 <step> --parity` puts this directory first on
# PYTHONPATH and sets EXOPIPE3_PARITY_ROOT (the allowed roots, os.pathsep-separated), so every Python subprocess it starts -
# legacy scripts included - imports this file at start-up and installs core/guard.py: a write outside the parity roots
# raises a PermissionError naming the path instead of touching a production tree. With EXOPIPE3_PARITY_ROOT unset this
# file does nothing but chain to the next sitecustomize on the path. Python 3.6+.
import os
import sys

_ROOTS = os.environ.get('EXOPIPE3_PARITY_ROOT', '')
_HERE = os.path.dirname(os.path.abspath(__file__))

if _ROOTS:
    import importlib.util as _ilu
    _gpath = os.path.join(os.path.dirname(_HERE), 'guard.py')
    _spec = _ilu.spec_from_file_location('exopipe.phase3.core.guard', _gpath)
    _guard = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_guard)
    sys.modules.setdefault('exopipe.phase3.core.guard', _guard)
    _guard.install([r for r in _ROOTS.split(os.pathsep) if r], log=os.environ.get('EXOPIPE3_PARITY_LOG') or None)

# CODE FINGERPRINT of a legacy child (core/legacy.run): with EXOPIPE3_FINGERPRINT_OUT set, write at exit a JSON
# {module file: sha1[:12]} of every imported source file under the legacy roots (EXOPIPE3_FINGERPRINT_ROOTS,
# os.pathsep-separated) or in an 'exopipe' package, so the parent's run record names the code that produced its products.
# One file PER PROCESS (<path>.<pid>; legacy.run merges them) and none when this process imported no such file: a
# descendant that outlives the child - a multiprocessing forkserver exits after its parent and runs atexit - wrote '{}' over
# the child's fingerprint (a race legacy.run usually won; 2026-10-14, phase3_snapshot_refastro's forkserver pool).
_FP_OUT = os.environ.get('EXOPIPE3_FINGERPRINT_OUT', '')
if _FP_OUT:
    import atexit as _atexit
    import hashlib as _hashlib
    # Each module's source is hashed when its code is executed (an audit hook, Python 3.8+: its import, or a reload) and
    # the main script now, before it runs - not at exit: a file edited while the child runs would otherwise be recorded as
    # the code that ran (2026-10-15 review). Python 3.6 / 3.7 children have no audit hooks: there the modules are hashed
    # at exit, as before.
    _FP_RAN = {}

    def _fp_sha(f):
        with open(f, 'rb') as fo:
            return _hashlib.sha1(fo.read()).hexdigest()[:12]

    if hasattr(sys, 'addaudithook'):
        def _fp_audit(event, args):
            if event != 'exec':
                return
            try:
                co = args[0]
                f = getattr(co, 'co_filename', None)
                if f and f.endswith('.py') and getattr(co, 'co_name', None) == '<module>':
                    _FP_RAN[os.path.abspath(f)] = _fp_sha(f)
            except Exception:                  # an audit hook must never break the event it observes
                pass
        sys.addaudithook(_fp_audit)
    try:
        _m0 = sys.argv[0] if sys.argv else None
        if _m0 and _m0 not in ('-c', '-m') and os.path.isfile(_m0):
            _FP_RAN.setdefault(os.path.abspath(_m0), _fp_sha(_m0))
    except (OSError, AttributeError):
        pass

    def _dump_fp():
        import json
        roots = tuple(r.rstrip('/') + '/' for r in os.environ.get('EXOPIPE3_FINGERPRINT_ROOTS', '').split(os.pathsep) if r)
        out = {}
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, '__file__', None)
            if not f or not f.endswith('.py'):
                continue
            f = os.path.abspath(f)
            if name.startswith('exopipe') or (roots and f.startswith(roots)):
                try:
                    out[f] = _FP_RAN.get(f) or _fp_sha(f)
                except OSError:
                    pass
        main = getattr(sys.modules.get('__main__'), '__file__', None) or (sys.argv[0] if sys.argv else None)
        if main and main not in ('-c', '-m') and os.path.isfile(main):
            out[os.path.abspath(main)] = _FP_RAN.get(os.path.abspath(main)) or _fp_sha(main)
        if not out:
            return
        mine = f'{_FP_OUT}.{os.getpid()}'
        try:
            with open(mine + '.tmp', 'w') as fo:
                json.dump(out, fo, indent=0, sort_keys=True)
            os.replace(mine + '.tmp', mine)
        except OSError:
            pass
    _atexit.register(_dump_fp)

# B-30 FALLBACK STATE of a legacy child (core.legacy.run, core/fallback.py): with EXOPIPE3_FALLBACK_OUT set, write at exit
# the counts and refusals of the fallback sites the child reached - only when it imported core/fallback (a branch module;
# production modules never do), so the parent's unit records them.
_FB_OUT = os.environ.get('EXOPIPE3_FALLBACK_OUT', '')
if _FB_OUT:
    import atexit as _atexit_fb

    def _dump_fb():
        m = sys.modules.get('exopipe.phase3.core.fallback')
        if m is not None:
            try:
                m.dump(_FB_OUT)
            except OSError:
                pass
    _atexit_fb.register(_dump_fb)

# IMPORT WRAPS (core/wrap.py; B-32 first): with EXOPIPE3_WRAPS naming modules ('{}' = none), each named PRODUCTION module
# loads as always, then its branch patch file loads BY PATH (it imports nothing from the branch: a child's `exopipe` can be the
# production package), is registered as sys.modules['exopipe3_wrap.<module>'] (the code fingerprint above records it), checks
# its pin against the production source and applies with the config values the parent passed.
_WRAPS = os.environ.get('EXOPIPE3_WRAPS', '')
_WR = {}
if _WRAPS:
    import json as _json_w
    _WR = _json_w.loads(_WRAPS)
if _WR:
    import importlib.abc as _iabc_w
    import importlib.machinery as _imach_w
    import importlib.util as _iutil_w

    def _fallback_module():
        m = sys.modules.get('exopipe.phase3.core.fallback')
        if m is None:
            _fp = os.path.join(os.path.dirname(_HERE), 'fallback.py')
            _s = _iutil_w.spec_from_file_location('exopipe.phase3.core.fallback', _fp)
            m = _iutil_w.module_from_spec(_s)
            sys.modules['exopipe.phase3.core.fallback'] = m
            _s.loader.exec_module(m)
        return m

    class _WrapFinder(_iabc_w.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in _WR:
                return None
            spec = _imach_w.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return None
            _orig = spec.loader.exec_module

            def _exec(module, _orig=_orig, _name=name):
                _orig(module)
                w = _WR[_name]
                ps = _iutil_w.spec_from_file_location(f'exopipe3_wrap.{_name}', w['patch'])
                pm = _iutil_w.module_from_spec(ps)
                ps.loader.exec_module(pm)
                sys.modules[f'exopipe3_wrap.{_name}'] = pm      # named in the child's code fingerprint (review MINOR 5)
                pm.apply(module, w['cfg'], _fallback_module())
            spec.loader.exec_module = _exec
            return spec
    sys.meta_path.insert(0, _WrapFinder())

# chain to any other sitecustomize on the path (the environment's own, if it has one)
try:
    import importlib.machinery as _m
    _spec2 = _m.PathFinder.find_spec('sitecustomize', [p for p in sys.path if p and os.path.abspath(p) != _HERE])
    if _spec2 is not None and _spec2.origin and os.path.abspath(_spec2.origin) != os.path.abspath(__file__):
        with open(_spec2.origin) as _f:
            _code = compile(_f.read(), _spec2.origin, 'exec')
        exec(_code, {'__name__': 'sitecustomize', '__file__': _spec2.origin})
except SystemExit:
    raise
except Exception:
    pass
