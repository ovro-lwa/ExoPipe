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
_FP_OUT = os.environ.get('EXOPIPE3_FINGERPRINT_OUT', '')
if _FP_OUT:
    import atexit as _atexit

    def _dump_fp():
        import hashlib, json
        roots = tuple(r.rstrip('/') + '/' for r in os.environ.get('EXOPIPE3_FINGERPRINT_ROOTS', '').split(os.pathsep) if r)
        out = {}
        for name, mod in list(sys.modules.items()):
            f = getattr(mod, '__file__', None)
            if not f or not f.endswith('.py'):
                continue
            f = os.path.abspath(f)
            if name.startswith('exopipe') or (roots and f.startswith(roots)):
                try:
                    out[f] = hashlib.sha1(open(f, 'rb').read()).hexdigest()[:12]
                except OSError:
                    pass
        main = getattr(sys.modules.get('__main__'), '__file__', None) or (sys.argv[0] if sys.argv else None)
        if main and main not in ('-c', '-m') and os.path.isfile(main):
            out[os.path.abspath(main)] = hashlib.sha1(open(main, 'rb').read()).hexdigest()[:12]
        try:
            with open(_FP_OUT + '.tmp', 'w') as fo:
                json.dump(out, fo, indent=0, sort_keys=True)
            os.replace(_FP_OUT + '.tmp', _FP_OUT)
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

# IMPORT WRAPS (core/wrap.py; B-32 first): with EXOPIPE3_WRAPS set, each named PRODUCTION module loads as always, then its
# branch patch file loads BY PATH (it imports nothing from the branch: a child's `exopipe` can be the production package),
# checks its pins against the production source and applies with the config values the parent passed.
_WRAPS = os.environ.get('EXOPIPE3_WRAPS', '')
if _WRAPS:
    import json as _json_w
    import importlib.abc as _iabc_w
    import importlib.machinery as _imach_w
    import importlib.util as _iutil_w
    _WR = _json_w.loads(_WRAPS)

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
                ps = _iutil_w.spec_from_file_location(f'_exopipe3_wrap_{_name}', w['patch'])
                pm = _iutil_w.module_from_spec(ps)
                ps.loader.exec_module(pm)
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
