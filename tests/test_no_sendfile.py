#!/usr/bin/env python3
"""STANDING TEST: no sendfile copies (2026-11-20, FINDING-20261120-SHUTIL-SENDFILE-COPIES-TRUNCATE-ON-LWACALIM07).

On Linux, Python 3.8+ shutil.copyfile (and copy, copy2, copytree) copies through os.sendfile and takes a return of 0 as the end of
the file, without an error. On lwacalim07 that silently truncated 14 of 560 Lustre-to-Lustre copies (0 to 85 percent of the
source), while ordinary reads and writes truncated 0 of 448 there.

1. Importing exopipe turns shutil's sendfile path off (and copy_file_range's, on Pythons that have it) and sets 16 MiB blocks.
2. THE FAILURE, REPRODUCED: os.sendfile is replaced by a stand-in that sends the first 4096 bytes and then returns 0 (what the
   lwacalim07 client did). With the switch on (FALSIFIER, Linux and Python 3.8+ only), shutil.copy2 / copyfile / copy /
   copytree each write a truncated file, without an error, and call the stand-in. With the switch as importing exopipe sets it,
   none calls the stand-in and every copy is byte-identical to its source (a size that is not a block multiple).
3. An in-place child (it imports no exopipe) started with core/parity_sitecustomize first on PYTHONPATH has the switch off and
   16 MiB blocks. FALSIFIER: the same child without it has the switch on (Linux and Python 3.8+).
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
HAS_SENDFILE_PATH = hasattr(os, 'sendfile') and sys.platform.startswith('linux') and sys.version_info >= (3, 8)


def test_import_turns_sendfile_off():
    import exopipe  # noqa: F401
    assert shutil._USE_CP_SENDFILE is False
    assert getattr(shutil, '_USE_CP_COPY_FILE_RANGE', False) is False
    assert shutil.COPY_BUFSIZE == 16 << 20


def _copies(src, d, tag):
    """Every shutil copy form, from ``src`` (a file in its own directory) into ``d``; the copied file paths."""
    out = []
    for name, fn in (('copy2', shutil.copy2), ('copyfile', shutil.copyfile), ('copy', shutil.copy)):
        dst = os.path.join(d, f'{tag}_{name}.bin')
        fn(src, dst)
        out.append(dst)
    tree = os.path.join(d, f'{tag}_tree')
    shutil.copytree(os.path.dirname(src), tree)
    out.append(os.path.join(tree, os.path.basename(src)))
    return out


def test_truncating_sendfile_never_used():
    import exopipe  # noqa: F401
    root = tempfile.mkdtemp(prefix='test_no_sendfile_', dir=os.environ.get('TMPDIR') or None)
    real = os.sendfile
    calls = []

    def stand_in(out_fd, in_fd, offset, count):          # the lwacalim07 client: one block, then 0 before the end
        calls.append(offset)
        return real(out_fd, in_fd, offset, min(count, 4096)) if offset == 0 else 0

    keep = shutil._USE_CP_SENDFILE
    try:
        sd = os.path.join(root, 'src')
        os.makedirs(sd)
        src = os.path.join(sd, 'master.fits')
        data = os.urandom((3 << 20) + 123)                   # 3 MiB + 123 bytes: not a block multiple
        with open(src, 'wb') as f:
            f.write(data)
        os.sendfile = stand_in
        if HAS_SENDFILE_PATH:                                # FALSIFIER: the switch on reproduces the silent truncation
            shutil._USE_CP_SENDFILE = True
            for p in _copies(src, root, 'on'):
                assert os.path.getsize(p) == 4096, (p, os.path.getsize(p))
            assert calls == [0, 4096] * 4, calls             # two calls per copy: one block, then the early 0
            assert shutil._USE_CP_SENDFILE is True           # the stand-in raised nothing: shutil kept its fast path
        shutil._USE_CP_SENDFILE = keep
        calls.clear()
        for p in _copies(src, root, 'off'):
            with open(p, 'rb') as f:
                assert f.read() == data, p
        assert calls == [], calls
    finally:
        os.sendfile = real
        shutil._USE_CP_SENDFILE = keep
        shutil.rmtree(root, ignore_errors=True)


def _child(pythonpath):
    env = {k: v for k, v in os.environ.items()
           if k != 'PYTHONPATH' and not (k.startswith('EXOPIPE3_') and k != 'EXOPIPE3_SITE') and not k.startswith('SUBLST_')}
    if pythonpath:
        env['PYTHONPATH'] = pythonpath
    code = ("import shutil, sys; print(getattr(shutil, '_USE_CP_SENDFILE', None), shutil.COPY_BUFSIZE "
            "if hasattr(shutil, 'COPY_BUFSIZE') else None, 'exopipe' in sys.modules)")
    r = subprocess.run([sys.executable, '-c', code], env=env, cwd=tempfile.gettempdir(), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout.split()


def test_inplace_child_switch_off():
    on = _child(BOOT)
    assert on == ['False', str(16 << 20), 'False'], on
    off = _child(None)
    assert off[2] == 'False', off
    if HAS_SENDFILE_PATH:                                    # FALSIFIER: without the sitecustomize the fast path is on
        assert off[0] == 'True', off


if __name__ == '__main__':
    test_import_turns_sendfile_off()
    test_truncating_sendfile_never_used()
    test_inplace_child_switch_off()
    print('PASS: no sendfile copies')
