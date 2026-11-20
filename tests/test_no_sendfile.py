#!/usr/bin/env python3
"""STANDING TEST: safe copies (exopipe/_safe_copy.py; 2026-11-20).

On Linux, Python 3.8+ shutil.copyfile (and copy, copy2, copytree) copies through os.sendfile and takes a return of 0 as the end
of the file, without an error. On lwacalim07 that silently truncated 14 of 560 Lustre-to-Lustre copies (0 to 85 percent of the
source), while ordinary reads and writes truncated 0 of 448 there.

1. Importing exopipe turns shutil's sendfile path off (and copy_file_range's, on Pythons that have it), leaves shutil's buffer
   size alone, and installs the short-copy gate on shutil.copyfile (once: a second install changes nothing).
2. THE FAILURE, REPRODUCED: os.sendfile is replaced by a stand-in that sends the first 4096 bytes and then returns 0 (what the
   lwacalim07 client did); os.copy_file_range, where it exists, by one that returns 0. FALSIFIERS (Linux, Python 3.8+): with
   the switch on, shutil's own copyfile (unwrapped) writes a 4096-byte copy without an error, and through the gate every copy
   form refuses (ShortCopyError; copytree reports it in shutil.Error) and leaves no copy. With the switch as importing exopipe
   sets it, no copy form calls either stand-in and every copy is byte-identical to its source (3 MiB + 123 bytes).
3. THE GATE, WHATEVER THE CAUSE: a read/write path that stops at half the file (shutil.copyfileobj replaced) is refused and the
   short copy removed. Not refused: a copy to os.devnull (not a regular file), a symlink copied as a link
   (follow_symlinks=False), an empty source.
4. An in-place child (it imports no exopipe) started with core/parity_sitecustomize first on PYTHONPATH has the switch off and the
   gate installed. FALSIFIER: the same child without it has neither (the switch on: Linux and Python 3.8+).
No Lustre file is used: everything is in a fresh scratch directory under TMPDIR.
"""
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
BOOT = os.path.join(REPO, 'exopipe', 'phase3', 'core', 'parity_sitecustomize')
DEFAULT_BUFSIZE = getattr(shutil, 'COPY_BUFSIZE', None)          # before exopipe is imported


def _scratch():
    return tempfile.mkdtemp(prefix='test_no_sendfile_', dir=os.environ.get('TMPDIR') or None)


def test_import_installs_safe_copies():
    import exopipe  # noqa: F401
    from exopipe import _safe_copy as S
    assert shutil._USE_CP_SENDFILE is False
    assert getattr(shutil, '_USE_CP_COPY_FILE_RANGE', False) is False
    assert getattr(shutil, 'COPY_BUFSIZE', None) == DEFAULT_BUFSIZE
    assert shutil.copyfile._exopipe_safe_copy is True
    gate = shutil.copyfile
    S.install()
    assert shutil.copyfile is gate                                  # idempotent: never a gate around a gate


def _copies(src, d, tag):
    """Every shutil copy form from ``src`` (alone in its directory) into ``d``: [(form, call, destination)]."""
    tree = os.path.join(d, f'{tag}_tree')
    return [(name, (lambda fn=fn, dst=os.path.join(d, f'{tag}_{name}.bin'): fn(src, dst)), os.path.join(d, f'{tag}_{name}.bin'))
            for name, fn in (('copy2', shutil.copy2), ('copyfile', shutil.copyfile), ('copy', shutil.copy))] + \
        [('copytree', lambda: shutil.copytree(os.path.dirname(src), tree), os.path.join(tree, os.path.basename(src)))]


def test_truncating_fast_paths():
    import exopipe  # noqa: F401
    from exopipe._safe_copy import ShortCopyError, sendfile_path_expected
    root = _scratch()
    real_sf, real_cfr = os.sendfile, getattr(os, 'copy_file_range', None)
    calls = []

    def sf_stand_in(out_fd, in_fd, offset, count):                 # the lwacalim07 client: one block, then 0 before the end
        calls.append(('sendfile', offset))
        return real_sf(out_fd, in_fd, offset, min(count, 4096)) if offset == 0 else 0

    def cfr_stand_in(*a, **k):
        calls.append(('copy_file_range',))
        return 0

    keep = shutil._USE_CP_SENDFILE
    try:
        sd = os.path.join(root, 'src')
        os.makedirs(sd)
        src = os.path.join(sd, 'master.fits')
        data = os.urandom((3 << 20) + 123)                         # not a block multiple
        with open(src, 'wb') as f:
            f.write(data)
        os.sendfile = sf_stand_in
        if real_cfr is not None:
            os.copy_file_range = cfr_stand_in
        if sendfile_path_expected():                               # FALSIFIERS: the switch on
            shutil._USE_CP_SENDFILE = True
            raw = os.path.join(root, 'raw.bin')
            shutil.copyfile.__wrapped__(src, raw)                  # shutil's own copyfile: silently short
            assert os.path.getsize(raw) == 4096, os.path.getsize(raw)
            for name, call, dst in _copies(src, root, 'on'):
                try:
                    call()
                except ShortCopyError as exc:
                    assert name != 'copytree' and 'SHORT-COPY' in str(exc) and '4096 of' in str(exc), (name, exc)
                except shutil.Error as exc:
                    assert name == 'copytree' and 'SHORT-COPY' in str(exc), (name, exc)
                else:
                    raise AssertionError(f'{name}: a short copy was not refused')
                assert not os.path.exists(dst), dst                # the short copy was removed
            assert ('sendfile', 0) in calls and shutil._USE_CP_SENDFILE is True
        shutil._USE_CP_SENDFILE = keep
        calls.clear()
        for name, call, dst in _copies(src, root, 'off'):
            call()
            with open(dst, 'rb') as f:
                assert f.read() == data, name
        assert calls == [], calls
    finally:
        os.sendfile = real_sf
        if real_cfr is not None:
            os.copy_file_range = real_cfr
        shutil._USE_CP_SENDFILE = keep
        shutil.rmtree(root, ignore_errors=True)


def test_gate_whatever_the_cause():
    import exopipe  # noqa: F401
    from exopipe._safe_copy import ShortCopyError
    root = _scratch()
    real = shutil.copyfileobj

    def half(fsrc, fdst, length=0):                                # a read path that stops early
        b = fsrc.read()
        fdst.write(b[:len(b) // 2])

    try:
        src = os.path.join(root, 'a.fits')
        with open(src, 'wb') as f:
            f.write(os.urandom(1 << 20))
        shutil.copyfileobj = half
        dst = os.path.join(root, 'b.fits')
        try:
            shutil.copy2(src, dst)
        except ShortCopyError as exc:
            assert f'{1 << 19} of {1 << 20} bytes' in str(exc), exc
        else:
            raise AssertionError('a half copy was not refused')
        assert not os.path.exists(dst)
        shutil.copyfileobj = real
        shutil.copyfile(src, os.devnull)                           # not a regular file: no size test
        link = os.path.join(root, 'link')
        os.symlink(src, link)
        shutil.copy2(link, os.path.join(root, 'link2'), follow_symlinks=False)
        assert os.path.islink(os.path.join(root, 'link2'))
        empty = os.path.join(root, 'empty')
        open(empty, 'wb').close()
        shutil.copy2(empty, os.path.join(root, 'empty2'))
        assert os.path.getsize(os.path.join(root, 'empty2')) == 0
    finally:
        shutil.copyfileobj = real
        shutil.rmtree(root, ignore_errors=True)


def _child(pythonpath):
    env = {k: v for k, v in os.environ.items()
           if k != 'PYTHONPATH' and not (k.startswith('EXOPIPE3_') and k != 'EXOPIPE3_SITE') and not k.startswith('SUBLST_')}
    if pythonpath:
        env['PYTHONPATH'] = pythonpath
    code = ("import shutil, sys; print(getattr(shutil, '_USE_CP_SENDFILE', None), "
            "getattr(shutil.copyfile, '_exopipe_safe_copy', False), 'exopipe' in sys.modules)")
    r = subprocess.run([sys.executable, '-c', code], env=env, cwd=tempfile.gettempdir(), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
    assert r.returncode == 0 and 'SAFE-COPY' not in r.stderr, r.stderr
    return r.stdout.split()


def test_inplace_child_safe_copies():
    from exopipe._safe_copy import sendfile_path_expected
    on = _child(BOOT)
    assert on == ['False', 'True', 'False'], on
    off = _child(None)
    assert off[1:] == ['False', 'False'], off
    if sendfile_path_expected():                                   # FALSIFIER: without the sitecustomize the fast path is on
        assert off[0] == 'True', off


if __name__ == '__main__':
    test_import_installs_safe_copies()
    test_truncating_fast_paths()
    test_gate_whatever_the_cause()
    test_inplace_child_safe_copies()
    print('PASS: safe copies')
