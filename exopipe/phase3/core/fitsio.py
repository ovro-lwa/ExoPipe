"""FITS and file I/O shared by every step: atomic writes, read-once caching, node-local scratch.

- ``write_fits(path, data, header)``: writes to ``<path>.tmp.<pid>`` in the same directory and renames, so a reader never
  sees a partial file and a crash never leaves a truncated product under the final name (CLAUDE.md 8: never rewrite
  FITS headers in place under a concurrent reader).
- ``read_image(path)``: (2-D float array, header, celestial WCS); results cached per process by (path, mtime, size), so a
  step that needs the same master for every epoch reads it once.
- ``scratch_dir(tag)``: a node-local scratch directory for small-file work (Lustre creates files ~180x slower than local
  disk; CLAUDE.md 7). Root from --work, else the site file's SCRATCH_ROOT for this host, else refuses: scratch never
  defaults to Lustre.
"""
import os
import shutil
import socket
import tempfile

_CACHE = {}
_CACHE_MAX = 8


def write_fits(path, data, header=None, overwrite=False):
    from astropy.io import fits
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f'{path} exists (pass overwrite=True deliberately)')
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f'.{os.path.basename(path)}.tmp.{os.getpid()}')
    fits.writeto(tmp, data, header, overwrite=True)
    os.replace(tmp, path)
    return path


def read_image(path, cache=True):
    """(data 2-D float32/64 as stored, header, celestial WCS) of a FITS image; leading axes of length 1 squeezed."""
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if cache and key in _CACHE:
        return _CACHE[key]
    with fits.open(path, memmap=False) as h:
        hdr = h[0].header.copy()
        data = np.squeeze(h[0].data)
    w = WCS(hdr).celestial
    out = (data, hdr, w)
    if cache:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = out
    return out


def scratch_root(work=None):
    from exopipe.phase3.config import CFG
    host = socket.gethostname()
    root = work or (CFG.site.SCRATCH_ROOT.get(host) if isinstance(CFG.site.SCRATCH_ROOT, dict) else None)
    if not root:
        from exopipe.phase3.core.record import Refusal
        raise Refusal(f'no node-local scratch root for {host}: pass --work /fast/... or set SCRATCH_ROOT in the site '
                      f'file (scratch never defaults to Lustre)')
    if root.startswith('/lustre'):
        from exopipe.phase3.core.record import Refusal
        raise Refusal(f'scratch root {root} is on Lustre - scratch must be node-local (/fast or /dataNN)')
    return root


def scratch_dir(tag, work=None):
    """A fresh node-local directory for this process; remove with cleanup(path) on success."""
    root = scratch_root(work)
    os.makedirs(root, exist_ok=True)
    return tempfile.mkdtemp(prefix=f'{tag}_', dir=root)


def cleanup(path):
    shutil.rmtree(path, ignore_errors=True)
