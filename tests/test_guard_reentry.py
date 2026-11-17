#!/usr/bin/env python3
"""STANDING TEST: the parity write guard's astropy hook and a finder that asks every finder (the sub-LST frozen-code guard).

core/guard.py's _AstropyHook patches astropy.io.fits writeto: on the import of astropy.io.fits it asks every other meta-path
finder for the module's spec and wraps that spec's loader. The sub-LST frozen-code guard (exopipe_ops/sublst_frozen_guard/
sitecustomize.py, FrozenCodeGuard) asks every other finder on every import. Without a re-entry flag in the hook the two
called each other without end: a parity (or injected) run under a frozen tree ended in RecursionError at
`from astropy.io import fits`, whichever of the two sitecustomize files came first on PYTHONPATH (found 2026-11-17).

Each check is a child process started as a parity child is (core/parity_sitecustomize on PYTHONPATH, EXOPIPE3_PARITY_ROOT
set) that imports astropy.io.fits and writes with it:
  alone          the hook without the frozen-code guard;
  frozen first   the frozen-code guard's sitecustomize first on PYTHONPATH: it chains to the parity one, whose hook then sits
                 AHEAD of the frozen-code finder in sys.meta_path;
  frozen second  the parity sitecustomize first: the frozen-code finder sits AHEAD of the hook.
Each imports astropy.io.fits (no RecursionError) with HDUList.writeto, fits.writeto and the convenience module's writeto
patched exactly once over astropy's own functions (as with the hook alone); fits.writeto and HDUList.writeto outside the
allowed roots are refused by the writeto wrappers (ParityWriteRefused naming them, nothing written); a write under the parity
root succeeds.
FALSIFIER: the same children with the hook's re-entry flag held off end in RecursionError in both orders, the cycle running
through the hook (core/guard.py) and the frozen-code finder - so the checks above exercise the flag.
The frozen-code guard is a stand-in with the real one's finder and chaining (always; CI), and the real file when it exists
(the cluster: exopipe_ops/sublst_frozen_guard, SUBLST_CODE_ROOT a scratch tree holding MANIFEST.sha256). Interpreter: this
one; on the cluster (the site file present) also the site's py36 and flow interpreters.

Run standalone: python tests/test_guard_reentry.py    Under pytest: pytest tests/test_guard_reentry.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
BOOT = os.path.join(REPO, 'exopipe', 'phase3', 'core', 'parity_sitecustomize')
SITE_FILE = '/lustre/gh/main/exopipe_ops/exopipe3_site.py'
FROZEN_GUARD = '/lustre/gh/main/exopipe_ops/sublst_frozen_guard'      # its sitecustomize.py: the real frozen-code guard
FROZEN_FINDERS = ('FrozenCodeGuard', 'FrozenCodeGuardStandIn')

# The stand-in: the real guard's finder (placed first, asks every other finder, no re-entry flag, returns the first spec) and
# its chaining to the next sitecustomize on the path; it refuses nothing.
STANDIN = r'''
import os
import sys


class FrozenCodeGuardStandIn(object):
    def find_spec(self, name, path=None, target=None):
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, 'find_spec'):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                return spec
        return None


sys.meta_path.insert(0, FrozenCodeGuardStandIn())
try:
    import importlib.machinery as _m
    _here = os.path.dirname(os.path.abspath(__file__))
    _spec = _m.PathFinder.find_spec('sitecustomize', [p for p in sys.path if p and os.path.abspath(p) != _here])
    if _spec is not None and _spec.origin and os.path.abspath(_spec.origin) != os.path.abspath(__file__):
        _code = compile(open(_spec.origin).read(), _spec.origin, 'exec')
        exec(_code, {'__name__': 'sitecustomize', '__file__': _spec.origin})
except SystemExit:
    raise
except Exception:
    pass
'''

# The child (Python 3.6+): argv = result file, mode ('flag-on' or 'flag-off': the hook's re-entry flag held off), the outside
# path prefix, the inside path. Imports astropy.io.fits, records the writeto patches and the originals they call, writes
# outside the allowed roots (fits.writeto, HDUList.writeto) and inside them.
CHILD = r'''
import builtins, json, os, sys, traceback
res, mode, outside, inside = sys.argv[1:5]
st = getattr(builtins, '__exopipe3_parity_guard__', {})
out = {'python': sys.version.split()[0], 'finders': [type(f).__name__ for f in sys.meta_path],
       'astropy_before': sorted(m for m in sys.modules if m.split('.')[0] == 'astropy')}
hook = st.get('hook')
if mode == 'flag-off' and hook is not None:
    type(hook)._busy = property(lambda self: False, lambda self, v: None)
try:
    from astropy.io import fits
except RecursionError as e:
    tb = traceback.extract_tb(e.__traceback__)
    out['import'] = 'RecursionError'
    out['frames'] = len(tb)
    out['cycle'] = sorted(set('%s/%s:%s' % (os.path.basename(os.path.dirname(f[0])), os.path.basename(f[0]), f[2])
                              for f in tb[-20:]))
else:
    import numpy as np
    out['import'] = 'ok'
    out['patched'] = sorted('%s.%s' % (getattr(o, '__name__', type(o).__name__), a)
                            for (o, a, k) in st.get('patched', []) if 'writeto' in k)
    out['wrappers'] = [fits.HDUList.writeto.__name__, fits.writeto.__name__]
    out['originals'] = sorted('%s=%s.%s' % (k, getattr(v, '__module__', ''), getattr(v, '__name__', ''))
                              for k, v in st.get('orig', {}).items() if 'writeto' in k)
    data = np.zeros((2, 2), dtype=np.float32)
    for what, call in (('fits.writeto', lambda p: fits.writeto(p, data)),
                       ('HDUList.writeto', lambda p: fits.HDUList([fits.PrimaryHDU(data)]).writeto(p))):
        try:
            call(outside + '.' + what + '.fits')
            out['outside ' + what] = 'written'
        except PermissionError as e:
            out['outside ' + what] = '%s: %s' % (type(e).__name__, e.strerror)
    fits.writeto(inside, data)
    out['inside'] = os.path.isfile(inside)
with open(res, 'w') as f:
    json.dump(out, f)
'''


class _Scratch(object):
    """A node-local scratch directory (TMPDIR): the child, the stand-in, a parity root and a frozen tree."""

    def __init__(self):
        self.d = tempfile.mkdtemp(prefix='exopipe3_guard_reentry_')
        self.child = os.path.join(self.d, 'reentry_child.py')
        with open(self.child, 'w') as f:
            f.write(CHILD)
        self.standin = os.path.join(self.d, 'frozen_guard_standin')
        os.makedirs(self.standin)
        with open(os.path.join(self.standin, 'sitecustomize.py'), 'w') as f:
            f.write(STANDIN)
        self.parity_root = os.path.join(self.d, 'parity_root')
        os.makedirs(self.parity_root)
        self.frozen_root = os.path.join(self.d, 'frozen_root')
        os.makedirs(self.frozen_root)
        open(os.path.join(self.frozen_root, 'MANIFEST.sha256'), 'w').close()
        # outside every allowed root (the parity root, /tmp, /dev/shm, $TMPDIR): next to this file
        self.outside = os.path.join(REPO, 'tests', '_guard_reentry_outside_%d' % os.getpid())
        allowed = [os.path.realpath(r) for r in (self.parity_root, '/tmp', '/dev/shm', os.environ.get('TMPDIR') or '/tmp')]
        o = os.path.realpath(os.path.dirname(self.outside))
        assert not any(o == r or o.startswith(r + os.sep) for r in allowed), (
            'run the test from a checkout outside the allowed roots (it is under one of %s)' % allowed)
        self.n = 0

    def close(self):
        shutil.rmtree(self.d, ignore_errors=True)


def _run(py, sc, pythonpath, mode='flag-on'):
    """Run the child with ``py`` and PYTHONPATH ``pythonpath``; returns its result, with what it wrote outside."""
    sc.n += 1
    res = os.path.join(sc.d, 'result_%d.json' % sc.n)
    inside = os.path.join(sc.parity_root, 'inside_%d.fits' % sc.n)
    env = {k: v for k, v in os.environ.items()
           if k != 'PYTHONPATH' and not k.startswith('EXOPIPE3_') and not k.startswith('SUBLST_')}
    env.update(PYTHONPATH=os.pathsep.join(pythonpath), EXOPIPE3_PARITY_ROOT=sc.parity_root, SUBLST_CODE_ROOT=sc.frozen_root,
               NUMPY_MADVISE_HUGEPAGE='0')
    p = subprocess.run([py, sc.child, res, mode, sc.outside, inside], env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=600)
    written = [sc.outside + '.' + w + '.fits' for w in ('fits.writeto', 'HDUList.writeto')]
    written = [q for q in written if os.path.exists(q)]
    for q in written:
        os.remove(q)
    assert p.returncode == 0 and os.path.isfile(res), (py, pythonpath, mode, p.returncode,
                                                       p.stdout.decode('utf-8', 'replace')[-3000:])
    with open(res) as f:
        r = json.load(f)
    r['outside_written'] = written
    return r


def _check_ok(r, what, alone=None):
    assert r['import'] == 'ok', (what, 'importing astropy.io.fits failed', r)
    assert r['astropy_before'] == [], (what, 'astropy imported before the child asked for it', r['astropy_before'])
    assert '_AstropyHook' in r['finders'], (what, 'the guard is not installed', r['finders'])
    p = r['patched']
    assert 'HDUList.writeto' in p and 'astropy.io.fits.writeto' in p and len(p) == len(set(p)), (
        what, 'writeto not patched exactly once', p)
    if alone is not None:
        assert p == alone['patched'], (what, 'patches differ from the hook alone', p, alone['patched'])
    assert r['wrappers'] == ['_w_writeto_method', '_w_fits_writeto'], (what, r['wrappers'])
    assert r['originals'] and all('=astropy.io.fits.' in o and '._w_' not in o for o in r['originals']), (
        what, 'a writeto wrapper wraps a wrapper', r['originals'])
    for w, label in (('fits.writeto', '(astropy.io.fits.writeto)'), ('HDUList.writeto', '(astropy.io.fits HDUList.writeto)')):
        got = r['outside ' + w]
        assert got.startswith('ParityWriteRefused: ') and got.endswith(label), (what, 'outside write', w, got)
    assert r['outside_written'] == [], (what, 'written outside the allowed roots', r['outside_written'])
    assert r['inside'] is True, (what, 'a write under the parity root failed', r)


def _order(r, first, what):
    """``first`` 'frozen' or 'hook': which of the two sits ahead in the child's sys.meta_path."""
    f = r['finders']
    fz = [i for i, n in enumerate(f) if n in FROZEN_FINDERS]
    assert len(fz) == 1 and f.count('_AstropyHook') == 1, (what, 'one frozen-code finder and one hook', f)
    hook_ahead = f.index('_AstropyHook') < fz[0]
    assert hook_ahead == (first == 'hook'), (what, 'sys.meta_path order', f)


def _check_falsifier(r, what):
    assert r['import'] == 'RecursionError', (
        what, 'with the hook\'s re-entry flag held off the import did not recurse - the check does not exercise the flag', r)
    frozen = ('sublst_frozen_guard/sitecustomize.py:find_spec', 'frozen_guard_standin/sitecustomize.py:find_spec')
    assert 'core/guard.py:find_spec' in r['cycle'] and any(c in r['cycle'] for c in frozen), (what, r['cycle'])


def _matrix(py, sc, frozen_dirs):
    """The hook alone, then each frozen-code guard in ``frozen_dirs`` first and second on PYTHONPATH (each with the flag on
    and held off). Returns the printed lines."""
    alone = _run(py, sc, [BOOT])
    _check_ok(alone, '%s hook alone' % py)
    assert not any(n in FROZEN_FINDERS for n in alone['finders']), alone['finders']
    lines = ['Python %s, hook alone: import ok, writeto patched once each (%s), outside writes refused, inside written'
             % (alone['python'], ', '.join(alone['patched']))]
    for fdir in frozen_dirs:
        name = 'stand-in' if fdir.startswith(sc.d) else fdir
        for path, ahead in (([fdir, BOOT], 'hook'), ([BOOT, fdir], 'frozen')):
            what = '%s, frozen-code guard %s %s on PYTHONPATH' % (py, name, 'first' if path[0] == fdir else 'second')
            r = _run(py, sc, path)
            _order(r, ahead, what)
            _check_ok(r, what, alone)
            f = _run(py, sc, path, 'flag-off')
            _order(f, ahead, what + ' (flag held off)')
            _check_falsifier(f, what + ' (flag held off)')
            lines.append('Python %s, frozen-code guard %s %s on PYTHONPATH (%s ahead in sys.meta_path): import ok, patched '
                         'once, refusals by the writeto wrappers; flag held off: RecursionError after %d frames, cycle %s'
                         % (r['python'], name, 'first' if path[0] == fdir else 'second', ahead, f['frames'],
                            ' + '.join(c for c in f['cycle'] if c.endswith(':find_spec') and 'parity_' not in c)))
    return lines


def test_hook_with_standin_frozen_guard():
    """Data-free (CI): this interpreter, the hook alone and with the stand-in frozen-code guard in both orders."""
    sc = _Scratch()
    try:
        for line in _matrix(sys.executable, sc, [sc.standin]):
            print(line)
    finally:
        sc.close()


def test_hook_with_real_frozen_guard_and_site_interpreters():
    """The cluster: the real frozen-code guard (this interpreter), then the site's py36 and flow interpreters with the
    stand-in and the real guard."""
    site = os.environ.get('EXOPIPE3_SITE') or SITE_FILE
    if not os.path.isfile(site) or not os.path.isfile(os.path.join(FROZEN_GUARD, 'sitecustomize.py')):
        print('      (skipped: no site file %s or no frozen-code guard %s on this node)' % (site, FROZEN_GUARD))
        return
    os.environ.setdefault('EXOPIPE3_SITE', site)
    from exopipe.phase3.core import legacy
    sc = _Scratch()
    try:
        lines = [x for x in _matrix(sys.executable, sc, [FROZEN_GUARD]) if 'hook alone' not in x]
        for key in ('py36', 'flow'):
            lines += _matrix(legacy.interpreter(key), sc, [sc.standin, FROZEN_GUARD])
        for line in lines:
            print(line)
    finally:
        sc.close()


if __name__ == '__main__':
    test_hook_with_standin_frozen_guard()
    test_hook_with_real_frozen_guard_and_site_interpreters()
    print('PASS: the parity guard\'s astropy hook composes with finders that ask every finder')
