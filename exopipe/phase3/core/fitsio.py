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


def has_data(hdu):
    """True when the HDU's header declares a non-empty array (every NAXISn > 0), decided from the header alone: with memory
    maps off (R-GREGG-20261117-NO-MMAP-OF-LUSTRE-FILES-READ-OR-WRITE) ``hdu.data is not None`` reads the whole array."""
    h = hdu.header                       # a CompImageHDU's .header is already the image header (NAXIS, not ZNAXIS)
    n = int(h.get('NAXIS', 0))
    return n > 0 and all(int(h.get(f'NAXIS{i}', 0)) > 0 for i in range(1, n + 1))


def hdu_shape(hdu):
    """numpy-order shape of an image HDU from its header alone (() when it has no array; see has_data)."""
    return tuple(hdu.shape) if has_data(hdu) else ()


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


FITS_BLOCK = 2880                   # bytes, the FITS logical record


def fits_implied_size(path):
    """(bytes, HDUs) the headers of a FITS file imply: per HDU its header blocks plus its data (|BITPIX|/8 x GCOUNT x
    (PCOUNT + NAXIS1 x ... x NAXISn)) rounded up to 2880, the header blocks read with ordinary reads (never a memory map, never
    the data). The walk stops at the file's on-disk size, so a file cut short implies MORE bytes than it holds. Raises
    ValueError when a header is cut short, a required card is missing, or the bytes after an HDU do not start another HDU
    (bytes beyond the last HDU)."""
    size = os.path.getsize(path)
    off = n = 0
    with open(path, 'rb') as f:
        while off < size:
            f.seek(off)
            cards, nblk, end = {}, 0, False
            while not end:
                blk = f.read(FITS_BLOCK)
                if len(blk) < FITS_BLOCK:
                    raise ValueError(f'HDU {n}: the header is cut short at byte {off + nblk * FITS_BLOCK + len(blk)} of {size}')
                if nblk == 0:
                    first = blk[:8].decode('ascii', 'replace').strip()
                    if first != ('SIMPLE' if n == 0 else 'XTENSION'):
                        raise ValueError(f'{size - off} byte(s) at byte {off} after HDU {n - 1} do not start an HDU (first '
                                         f'keyword {first!r})' if n else f'not a FITS file (first keyword {first!r})')
                nblk += 1
                for i in range(0, FITS_BLOCK, 80):
                    card = blk[i:i + 80].decode('ascii', 'replace')
                    key = card[:8].strip()
                    if key == 'END':
                        end = True
                        break
                    if card[8:10] == '= ' and (key in ('BITPIX', 'PCOUNT', 'GCOUNT', 'GROUPS') or key.startswith('NAXIS')):
                        cards[key] = card[10:].split('/', 1)[0].strip()
            try:
                bitpix, naxis = abs(int(cards['BITPIX'])), int(cards['NAXIS'])
                dims = [int(cards[f'NAXIS{i}']) for i in range(1, naxis + 1)]
                pcount, gcount = int(cards.get('PCOUNT', 0)), int(cards.get('GCOUNT', 1))
            except (KeyError, ValueError) as e:
                raise ValueError(f'HDU {n}: header card {e} missing or not an integer') from None
            if n == 0 and cards.get('GROUPS') == 'T' and dims and dims[0] == 0:
                dims = dims[1:]                                     # random groups: NAXIS1 = 0 is not an axis
            nel = 1
            for d in dims:
                nel *= d
            data = bitpix // 8 * gcount * (pcount + nel) if naxis else 0
            off += nblk * FITS_BLOCK + -(-data // FITS_BLOCK) * FITS_BLOCK
            n += 1
    return off, n


def fsync_path(path):
    """fsync a written file by path (open read-only, fsync, close). OSError propagates: Linux reports a writeback error the
    writer never saw (pages a Lustre eviction discarded) to the next fsync of the file, a new descriptor's included."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def drop_cache(path):
    """Ask the kernel to drop a file's cached pages (POSIX_FADV_DONTNEED), so a later read is of what the disk holds, not of
    pages still in memory. Advice only: no outcome depends on it, a failure is ignored."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (OSError, AttributeError):
        pass
    finally:
        os.close(fd)


def settle_fits(path):
    """None when a freshly written FITS file is on disk whole: fsync it (an OSError is a failed write), then its on-disk size
    must equal the size its headers imply (fits_implied_size: a file cut short or with bytes beyond its last HDU fails), then
    its cached pages are dropped (drop_cache) so the caller's content tests read the disk. Else the reason (a string)."""
    try:
        fsync_path(path)
    except OSError as e:
        return f'fsync failed ({e})'
    try:
        size = os.path.getsize(path)
        implied, nhdu = fits_implied_size(path)
    except (OSError, ValueError) as e:
        return f'size check failed ({e})'
    if size != implied:
        return f'{size} bytes on disk != {implied} bytes its {nhdu} HDU header(s) imply ({size - implied:+d})'
    drop_cache(path)
    return None


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
    # absolute, so a caller that changes its working directory (followup investigate: os.chdir(scratch)) keeps the same
    # directory, and a relative --work under a Lustre working directory is judged where it really is
    root = os.path.abspath(root)
    if root.startswith('/lustre') or os.path.realpath(root).startswith('/lustre'):
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
