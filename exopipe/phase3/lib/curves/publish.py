"""Publishing light-curve tables (gate G12; replaces ``cp -n $LCOUT/*_lightcurves.fits $LCSHIP/`` at v9:372 and BF:74).

``cp -n`` silently kept an older table when a re-extraction differed (batch row B-49). Here a table is published only when
it is listed in a worker result and passed its content tests, by an atomic copy (temporary name in the destination
directory, then rename), and:
- destination absent                 -> 'published';
- destination byte-identical         -> 'skipped-valid' (nothing written);
- destination differs, no --supersede -> 'conflict' (gate G12: the pair is refused, the old table stays);
- destination differs, --supersede T -> the old table moves to <out>/_superseded/<T>/<cell>/ with a MANIFEST.csv row
                                        (utc, file, sha1, size, reason), then 'superseded' (new table published).
Every re-extraction differs at least in the EXTRACT card (wall-clock time of writing), so a re-run over published tables
needs --supersede.
"""
import csv
import hashlib
import os
import shutil
import time


def sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def same_bytes(a, b):
    return os.path.getsize(a) == os.path.getsize(b) and sha1(a) == sha1(b)


def _atomic_copy(src, dst):
    d = os.path.dirname(os.path.abspath(dst))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f'.{os.path.basename(dst)}.tmp.{os.getpid()}')
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _supersede(dst, out_root, tag, cell, reason):
    sdir = os.path.join(out_root, '_superseded', tag, cell)
    os.makedirs(sdir, exist_ok=True)
    moved = os.path.join(sdir, os.path.basename(dst))
    if os.path.exists(moved):
        raise FileExistsError(f'{moved} exists: tag {tag!r} already superseded this table - use a new --supersede TAG')
    h, size = sha1(dst), os.path.getsize(dst)
    man = os.path.join(sdir, 'MANIFEST.csv')
    new = not os.path.exists(man)
    with open(man, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['utc', 'file', 'sha1', 'size', 'reason'])
        w.writerow([time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), os.path.basename(dst), h, size, reason])
    os.replace(dst, moved)
    return moved


def publish(src, dst, out_root, cell, supersede=None, reason='re-extraction'):
    """(status, detail): 'published' | 'skipped-valid' | 'conflict' | 'superseded'."""
    if os.path.exists(dst):
        if same_bytes(src, dst):
            return 'skipped-valid', f'{dst} byte-identical'
        if not supersede:
            return 'conflict', (f'{dst} exists and differs from the new table (G12; pass --supersede TAG to move the '
                                f'old table to {os.path.join(out_root, "_superseded", "<TAG>", cell)})')
        moved = _supersede(dst, out_root, supersede, cell, reason)
        _atomic_copy(src, dst)
        return 'superseded', f'old table moved to {moved}'
    _atomic_copy(src, dst)
    return 'published', dst
