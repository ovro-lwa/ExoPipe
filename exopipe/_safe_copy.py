"""Safe file copies for every process that imports exopipe and for the in-place children (standard library only).

One file, installed by exopipe/__init__.py and loaded by path from exopipe/phase3/core/parity_sitecustomize/sitecustomize.py
(the in-place children import no exopipe).

Why: on Linux, Python 3.8+ shutil.copyfile (which shutil.copy, copy2, copytree and a move across filesystems call) copies
through os.sendfile (newer Pythons also through os.copy_file_range) and takes a return of 0 as the end of the file, without an
error. On lwacalim07 (2026-11-20) that silently truncated 14 of 560 Lustre-to-Lustre copies of 9-39 MB FITS files, to 0-85
percent of their size; shutil's ordinary read/write path truncated 0 of 448 copies of the same files there, and on lwacalim04
and lwacalim09 (same kernel and Lustre client) sendfile truncated none of 224 each. The 11-20 hybrid archive push refused 28
units at its checksum gate because of it.

install():
1. turns shutil's fast paths off, so shutil copies with ordinary reads and writes (shutil.copyfileobj, its own fallback; the
   buffer size is shutil's default). Where this Python should have the sendfile switch and does not, it refuses (raises):
   setting a renamed private name would silently do nothing.
2. replaces shutil.copyfile by a wrapper that refuses a SHORT copy whatever its cause: when the source and the copy are
   regular files and the copy is smaller than the source was when the copy began, the copy is removed and ShortCopyError
   (an OSError, errno EIO, text 'SHORT-COPY: ...') is raised. A source that grows during the copy (a log being written) is
   not refused. shutil.copy, copy2 and copytree call shutil.copyfile by its module name, so they are covered.
Idempotent per process (the wrapper is marked). tests/test_no_sendfile.py.
"""
import errno
import os
import shutil
import stat
import sys


class ShortCopyError(OSError):
    pass


def sendfile_path_expected():
    """True where shutil copies through os.sendfile unless switched off (Linux, Python 3.8+)."""
    return hasattr(os, 'sendfile') and sys.platform.startswith('linux') and sys.version_info >= (3, 8)


def install():
    if getattr(shutil.copyfile, '_exopipe_safe_copy', False):
        return
    if sendfile_path_expected() and not hasattr(shutil, '_USE_CP_SENDFILE'):
        raise RuntimeError('exopipe safe copies: this Python\'s shutil has no _USE_CP_SENDFILE switch, so its sendfile copies '
                           '(which truncated silently on lwacalim07, 2026-11-20) cannot be turned off; update exopipe/_safe_copy.py '
                           'for this Python before running exopipe with it')
    shutil._USE_CP_SENDFILE = False
    if hasattr(shutil, '_USE_CP_COPY_FILE_RANGE'):
        shutil._USE_CP_COPY_FILE_RANGE = False
    orig = shutil.copyfile

    def copyfile(src, dst, *, follow_symlinks=True):
        n0 = None
        if follow_symlinks or not os.path.islink(src):
            s0 = os.stat(src)
            if stat.S_ISREG(s0.st_mode):
                n0 = s0.st_size
        out = orig(src, dst, follow_symlinks=follow_symlinks)
        if n0:
            s1 = os.stat(out)
            if stat.S_ISREG(s1.st_mode) and s1.st_size < n0:
                try:
                    os.remove(out)
                except OSError:
                    pass
                raise ShortCopyError(errno.EIO, f'SHORT-COPY: {src} -> {dst}: {s1.st_size} of {n0} bytes copied; the short '
                                                f'copy was removed')
        return out

    copyfile._exopipe_safe_copy = True
    copyfile.__doc__ = orig.__doc__
    copyfile.__wrapped__ = orig
    shutil.copyfile = copyfile
