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
        main = getattr(sys.modules.get('__main__'), '__file__', None)
        if main and os.path.exists(main):
            out[os.path.abspath(main)] = hashlib.sha1(open(main, 'rb').read()).hexdigest()[:12]
        try:
            with open(_FP_OUT + '.tmp', 'w') as fo:
                json.dump(out, fo, indent=0, sort_keys=True)
            os.replace(_FP_OUT + '.tmp', _FP_OUT)
        except OSError:
            pass
    _atexit.register(_dump_fp)

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
