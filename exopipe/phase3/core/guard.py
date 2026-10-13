"""Parity write guard: in a parity run, refuse every write outside the allowed roots (DESIGN 11.1).

The ``--out`` check of the CLI only covers the step's own product root; old modules and legacy scripts write through
environment variables, hard-coded paths and their own defaults. A parity run therefore executes both sides under this
guard (modelled on exopipe_ops/sublst_frozen_guard/sitecustomize.py, the sub-LST frozen-code import guard): every
write-mode call below, on a path not under an allowed root, raises ``ParityWriteRefused`` (a PermissionError naming
the path) BEFORE anything is written, and the refusal is printed to stderr (and appended to ``$EXOPIPE3_PARITY_LOG``
when set), so it stays visible even where legacy code swallows OSError.

Guarded: builtins.open / io.open (modes w, a, x, +), os.open (write flags), os.rename / os.replace / os.renames (both
paths), os.remove / os.unlink / os.rmdir, os.mkdir / os.makedirs (unless the directory exists), os.symlink / os.link
(the new name), os.truncate, shutil.copyfile / copy / copy2 / copytree (destination), shutil.move (both), shutil.rmtree,
pathlib's accessor (Path.mkdir/unlink/rename/...), and astropy.io.fits writeto (HDUList.writeto and fits.writeto,
patched when astropy.io.fits is imported, before or after the guard). Reads are never refused. Not guarded: non-Python
subprocesses (cp, rsync, shell redirections), C extensions writing through their own I/O, and modules that bound an
os function to a local name before the guard was installed.

Allowed: the roots given to ``install`` (a parity run passes --out and --work; the run records live under --out) plus
``DEFAULT_ALLOWED`` (/tmp, /dev/null) and $TMPDIR; /dev/stdout, /dev/stderr and /dev/tty are never refused. Paths are
compared after realpath, so a symlink cannot lead out of an allowed root.

Subprocesses: ``export_env(roots)`` sets EXOPIPE3_PARITY_ROOT (os.pathsep-joined roots) and puts
``parity_sitecustomize/`` first on PYTHONPATH; its sitecustomize.py installs this guard in every Python process
started with that environment (legacy scripts included).

This file imports nothing from exopipe and runs on Python 3.6+: the sitecustomize loads it by path.
"""
import builtins
import errno
import io
import os
import shutil
import sys

ENV_ROOTS = 'EXOPIPE3_PARITY_ROOT'
ENV_LOG = 'EXOPIPE3_PARITY_LOG'
DEFAULT_ALLOWED = ('/tmp', '/dev/null')
_DEVICES = ('/dev/null', '/dev/stdout', '/dev/stderr', '/dev/tty')     # never products; compared before realpath
SITECUSTOMIZE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parity_sitecustomize')
_WFLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
# State shared by every copy of this module in the process (the sitecustomize loads one by path; the package may import
# another): kept on builtins.
_STATE = builtins.__dict__.setdefault('__exopipe3_parity_guard__', {'roots': None, 'orig': {}, 'log': None,
                                                                     'refused': 0})


class ParityWriteRefused(PermissionError):
    """A write outside the parity roots (the path is the exception's filename)."""


def active():
    return _STATE['roots'] is not None


def allowed_roots():
    return tuple(_STATE['roots'] or ())


def _real(path):
    return os.path.realpath(os.path.abspath(path))


def _norm(path, dir_fd=None):
    if isinstance(path, int):
        return None                       # an open descriptor: its open was checked
    path = os.fsdecode(os.fspath(path))
    if dir_fd is not None and not os.path.isabs(path):
        try:
            path = os.path.join(os.readlink('/proc/self/fd/%d' % dir_fd), path)
        except OSError:
            pass
    return path


def is_allowed(path):
    """True when writing ``path`` is allowed (or no guard is installed)."""
    roots = _STATE['roots']
    if roots is None or path is None:
        return True
    if os.path.abspath(path) in _DEVICES:
        return True
    r = _real(path)
    for root in roots:
        if r == root or r.startswith(root.rstrip('/') + '/'):
            return True
    return False


def _refuse(path, what):
    _STATE['refused'] += 1
    msg = 'PARITY GUARD: %s refused - %s is outside the parity roots (%s)' % (what, path, ', '.join(_STATE['roots']))
    try:
        sys.stderr.write(msg + '\n')
        sys.stderr.flush()
    except Exception:
        pass
    log = _STATE.get('log')
    if log:
        try:
            if not os.path.isdir(os.path.dirname(log)):
                _STATE['orig'].get('os.makedirs', os.makedirs)(os.path.dirname(log), exist_ok=True)
            with _STATE['orig']['open'](log, 'a') as f:
                f.write(msg + '\n')
        except Exception:
            pass
    raise ParityWriteRefused(errno.EACCES, 'parity guard: write outside the parity roots refused (%s)' % what, path)


def check(path, what='write', dir_fd=None):
    """Raise ParityWriteRefused when ``path`` may not be written under the active guard."""
    p = _norm(path, dir_fd)
    if not is_allowed(p):
        _refuse(p, what)


def _writes(mode):
    return isinstance(mode, str) and any(c in mode for c in 'wax+')


# ---- wrappers (each calls the original function saved at install) --------------------------------------------------

def _w_open(file, mode='r', *a, **k):
    if _writes(mode):
        check(file, 'open(%r)' % mode)
    return _STATE['orig']['open'](file, mode, *a, **k)


def _w_os_open(path, flags, *a, **k):
    if flags & _WFLAGS:
        check(path, 'os.open', k.get('dir_fd'))
    return _STATE['orig']['os.open'](path, flags, *a, **k)


def _two(name):
    def w(src, dst, *a, **k):
        check(src, name + ' (source)', k.get('src_dir_fd'))
        check(dst, name + ' (destination)', k.get('dst_dir_fd'))
        return _STATE['orig'][name](src, dst, *a, **k)
    w.__name__ = name.split('.')[-1]
    return w


def _one(name):
    def w(path, *a, **k):
        check(path, name, k.get('dir_fd'))
        return _STATE['orig'][name](path, *a, **k)
    w.__name__ = name.split('.')[-1]
    return w


def _ancestor_of_root(p):
    r = _real(p).rstrip('/') + '/'
    return any(root.startswith(r) for root in (_STATE['roots'] or ()))


def _w_mkdir(name):
    """mkdir / makedirs: allowed for an existing directory (no write), a path under an allowed root, or an ancestor of
    an allowed root (creating the parents of --out)."""
    def w(path, *a, **k):
        p = _norm(path, k.get('dir_fd'))
        if p is not None and not os.path.isdir(p) and not is_allowed(p) and not _ancestor_of_root(p):
            _refuse(p, name)
        return _STATE['orig'][name](path, *a, **k)
    w.__name__ = name.split('.')[-1]
    return w


def _dst(name):
    def w(src, dst, *a, **k):
        check(dst, name, k.get('dir_fd'))
        return _STATE['orig'][name](src, dst, *a, **k)
    w.__name__ = name.split('.')[-1]
    return w


def _copy_target(src, dst):
    d = _norm(dst)
    if d is not None and os.path.isdir(d):
        return os.path.join(d, os.path.basename(_norm(src)))
    return d


def _w_copy(name):
    def w(src, dst, *a, **k):
        check(_copy_target(src, dst), name)
        return _STATE['orig'][name](src, dst, *a, **k)
    w.__name__ = name.split('.')[-1]
    return w


def _w_move(src, dst, *a, **k):
    check(src, 'shutil.move (source)')
    check(_copy_target(src, dst), 'shutil.move (destination)')
    return _STATE['orig']['shutil.move'](src, dst, *a, **k)


def _w_truncate(path, *a, **k):
    if not isinstance(path, int):
        check(path, 'os.truncate')
    return _STATE['orig']['os.truncate'](path, *a, **k)


def _w_writeto_method(self, fileobj, *a, **k):
    if isinstance(fileobj, (str, bytes)) or hasattr(fileobj, '__fspath__'):
        check(fileobj, 'astropy.io.fits HDUList.writeto')
    return _STATE['orig']['HDUList.writeto'](self, fileobj, *a, **k)


def _w_fits_writeto(filename, *a, **k):
    if isinstance(filename, (str, bytes)) or hasattr(filename, '__fspath__'):
        check(filename, 'astropy.io.fits.writeto')
    return _STATE['orig']['fits.writeto'](filename, *a, **k)


_OS_ONE = ('remove', 'unlink', 'rmdir')
_OS_DIR = ('mkdir', 'makedirs')
_OS_TWO = ('rename', 'replace')
_OS_DST = ('symlink', 'link')


def _patch(obj, attr, key, wrapper):
    if not hasattr(obj, attr):
        return
    if key not in _STATE['orig']:
        _STATE['orig'][key] = getattr(obj, attr)
    setattr(obj, attr, wrapper)
    _STATE.setdefault('patched', []).append((obj, attr, key))


def _patch_astropy(fits):
    try:
        _patch(fits.HDUList, 'writeto', 'HDUList.writeto', _w_writeto_method)
        _patch(fits, 'writeto', 'fits.writeto', _w_fits_writeto)
        conv = sys.modules.get('astropy.io.fits.convenience')
        if conv is not None:
            _patch(conv, 'writeto', 'fits.writeto', _w_fits_writeto)
    except Exception:
        pass


class _AstropyHook(object):
    """Meta-path hook: patches astropy.io.fits writeto when the module is first imported after the guard."""

    def find_spec(self, name, path=None, target=None):
        if name != 'astropy.io.fits' or not active():
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, 'find_spec'):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is None:
                continue
            loader = spec.loader
            orig_exec = getattr(loader, 'exec_module', None)
            if orig_exec is None:
                return spec

            def exec_module(module, _orig=orig_exec):
                _orig(module)
                _patch_astropy(module)
            loader.exec_module = exec_module
            return spec
        return None


def install(allowed_roots, defaults=DEFAULT_ALLOWED, log=None):
    """Install the guard (idempotent: a second call replaces the allowed roots). ``allowed_roots`` are directories
    under which writes are allowed; ``defaults`` (/tmp, /dev/null, ...) and $TMPDIR are added. Returns the roots."""
    roots = [r for r in list(allowed_roots) + list(defaults or ()) + [os.environ.get('TMPDIR') or ''] if r]
    _STATE['roots'] = tuple(dict.fromkeys(_real(r) for r in roots))
    _STATE['log'] = log
    if _STATE.get('installed'):
        return _STATE['roots']
    _patch(builtins, 'open', 'open', _w_open)
    if io.open is not builtins.open:
        _patch(io, 'open', 'open', _w_open)
    _patch(os, 'open', 'os.open', _w_os_open)
    for n in _OS_ONE:
        _patch(os, n, 'os.' + n, _one('os.' + n))
    for n in _OS_DIR:
        _patch(os, n, 'os.' + n, _w_mkdir('os.' + n))
    for n in _OS_TWO:
        _patch(os, n, 'os.' + n, _two('os.' + n))
    for n in _OS_DST:
        _patch(os, n, 'os.' + n, _dst('os.' + n))
    _patch(os, 'truncate', 'os.truncate', _w_truncate)
    for n in ('copyfile', 'copy', 'copy2'):
        _patch(shutil, n, 'shutil.' + n, _w_copy('shutil.' + n))
    _patch(shutil, 'copytree', 'shutil.copytree', _dst('shutil.copytree'))
    _patch(shutil, 'move', 'shutil.move', _w_move)
    _patch(shutil, 'rmtree', 'shutil.rmtree', _one('shutil.rmtree'))
    try:                                  # pathlib binds os functions in its accessor at import (Python < 3.11)
        import pathlib
        acc = getattr(pathlib, '_NormalAccessor', None)
        if acc is not None:
            # the accessor's open is os.open (3.8) or io.open (3.10): wrap it with the matching wrapper
            acc_open = getattr(acc, 'open', None)
            is_io = acc_open is _STATE['orig'].get('open') or acc_open is io.open
            for n, w, key in (('mkdir', _w_mkdir('pathlib.mkdir'), 'pathlib.mkdir'),
                              ('unlink', _one('pathlib.unlink'), 'pathlib.unlink'),
                              ('rmdir', _one('pathlib.rmdir'), 'pathlib.rmdir'),
                              ('rename', _two('pathlib.rename'), 'pathlib.rename'),
                              ('replace', _two('pathlib.replace'), 'pathlib.replace'),
                              ('symlink', _dst('pathlib.symlink'), 'pathlib.symlink'),
                              ('link_to', _dst('pathlib.link_to'), 'pathlib.link_to'),
                              ('open', _w_open if is_io else _w_os_open, 'open' if is_io else 'os.open')):
                if hasattr(acc, n):
                    if key not in _STATE['orig']:
                        _STATE['orig'][key] = getattr(acc, n)
                    setattr(acc, n, staticmethod(w))
                    _STATE.setdefault('patched', []).append((acc, n, key))
    except Exception:
        pass
    if 'astropy.io.fits' in sys.modules:
        _patch_astropy(sys.modules['astropy.io.fits'])
    hook = _AstropyHook()
    sys.meta_path.insert(0, hook)
    _STATE['hook'] = hook
    _STATE['installed'] = True
    return _STATE['roots']


def uninstall():
    """Remove the guard and restore every original function (tests)."""
    for obj, attr, key in reversed(_STATE.get('patched', [])):
        orig = _STATE['orig'].get(key)
        if orig is None:
            continue
        if isinstance(obj, type) and obj.__module__ == 'pathlib':
            setattr(obj, attr, staticmethod(orig) if not isinstance(orig, staticmethod) else orig)
        else:
            setattr(obj, attr, orig)
    hook = _STATE.pop('hook', None)
    if hook is not None and hook in sys.meta_path:
        sys.meta_path.remove(hook)
    _STATE.update(roots=None, log=None, patched=[], installed=False, orig={})


def export_env(roots, log=None, env=None):
    """Make every Python subprocess started with ``env`` (default os.environ) install the guard with ``roots``:
    sets EXOPIPE3_PARITY_ROOT and puts parity_sitecustomize/ first on PYTHONPATH. Returns the environment."""
    env = os.environ if env is None else env
    env[ENV_ROOTS] = os.pathsep.join(roots)
    if log:
        env[ENV_LOG] = log
    parts = [p for p in (env.get('PYTHONPATH') or '').split(os.pathsep) if p and p != SITECUSTOMIZE_DIR]
    env['PYTHONPATH'] = os.pathsep.join([SITECUSTOMIZE_DIR] + parts)
    return env
