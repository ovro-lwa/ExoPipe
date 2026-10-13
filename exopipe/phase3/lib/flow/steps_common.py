"""Helpers shared by the three tiers of `exopipe3 flow` (steps/flow.py -> lib/flow/steps_{deep,snapshot,sublst}.py) and by
the content tests (lib/flow/tests.py). Light imports only: numpy/astropy are imported inside the functions that need them,
and no flow engine (JAX) is ever imported here - GPU work is always a child process started through core.legacy.run (the one
child runner, _CROSSCHECK decision 1) under the site's PYTHON['flow'] (or 'default' for the sub-LST refine module).
"""
import contextlib
import csv
import errno
import fcntl
import hashlib
import json
import os
import re
import socket
import sys
import time

from exopipe.phase3.core.record import Refusal

FLOW_SECTIONS = ('flow', 'flow_prep', 'flow_snapshot', 'flow_refine')


def cfg():
    from exopipe.phase3.config import CFG
    return CFG


def utcstamp():
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())


# ---------------------------------------------------------------------------------------------------------------------
# Snapshot cells in paths (the item grammar itself is core.cells.parse_night_cell / parse_epoch)
# ---------------------------------------------------------------------------------------------------------------------

_SNAP_CELL_IN_PATH = re.compile(r'^\d\dh_\d{4}-\d\d-\d\d$')


def parse_snapshot_item(item):
    """'18h_2025-04-11' -> ('18h', '2025-04-11', None); '18h_2025-04-11_36MHz' -> (..., '36MHz') - core.cells.parse_night_cell
    restricted to subbands (a band suffix refuses: flows are per subband)."""
    from exopipe.phase3.core import cells
    lst, date, sb = cells.parse_night_cell(item)
    if sb and sb not in cfg().instrument.ALL_SUBBANDS:
        raise ValueError(f'{item}: {sb} is not a subband - flows are per subband ({", ".join(cfg().instrument.ALL_SUBBANDS)})')
    return lst, date, sb


def snapshot_cell_of_path(path):
    """The snapshot cell (<LST>_<date>) named by a directory component of path, or None."""
    for part in reversed(os.path.abspath(path).split(os.sep)):
        if _SNAP_CELL_IN_PATH.match(part):
            return part
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Snapshot frame names (phase3_snapshot_flow.py:26-29; core/inputs.py docstring)
# ---------------------------------------------------------------------------------------------------------------------

FRAME_PAT = re.compile(r'(?P<sb>\d+MHz)-(?P<kind>Clean|Pilot)-Snapshot-(?P<ts>\d{8}_\d{6})(?:-(?P<st>[IV]))?-image\.fits$')
FIELD_PAT = re.compile(r'^flow_(?P<sb>\d+MHz)_(?P<ts>\d{8}_\d{6})\.npy$')


def frame_info(path):
    """(subband, ts, Stokes) of a staged frame or of its _flowed copy; None for anything else. Clean = Stokes I; Pilot
    without -I/-V = Stokes V (new era); Pilot -I-/-V- = pilot era."""
    name = os.path.basename(path)
    if name.endswith('_flowed.fits'):
        name = name[:-len('_flowed.fits')] + '.fits'
    m = FRAME_PAT.search(name)
    if not m:
        return None
    st = 'I' if m['kind'] == 'Clean' else (m['st'] or 'V')
    return m['sb'], m['ts'], st


def flowed_path(frame):
    return frame[:-5] + '_flowed.fits'


def raw_of_flowed(path):
    return path[:-len('_flowed.fits')] + '.fits'


def is_staged_frame(name):
    """A staged raw frame (not a flowed copy, not a leftover)."""
    return name.endswith('-image.fits') and frame_info(name) is not None


def snapshot_master(lst, sb, root=None):
    c = cfg().flow_snapshot
    return os.path.join(root or c.MASTERS_ROOT, c.MASTER_FILE.format(lst=lst, sb=sb))


def subband_mhz(sb):
    return float(str(sb).replace('MHz', ''))


# ---------------------------------------------------------------------------------------------------------------------
# FITS headers (raw, header-only, keeps comments) and blank-frame rule
# ---------------------------------------------------------------------------------------------------------------------

def header_cards(path, max_blocks=400):
    """{KEY: (value, comment)} of the primary header (first occurrence), 'HISTORY': [text, ...]. Header-only read."""
    from exopipe.phase3.core.record import _fits_value
    out, hist = {}, []
    with open(path, 'rb') as f:
        for n in range(max_blocks):
            blk = f.read(2880)
            if len(blk) < 2880 or (n == 0 and not blk.startswith(b'SIMPLE')):
                break
            for i in range(0, 2880, 80):
                card = blk[i:i + 80].decode('ascii', 'replace')
                key = card[:8].strip()
                if key == 'END':
                    out['HISTORY'] = hist
                    return out
                if key == 'HISTORY':
                    hist.append(card[8:].rstrip())
                    continue
                if card[8:10] != '= ' or key in out:
                    continue
                body = card[10:]
                # comment = text after the first '/' outside a quoted string
                q, cpos = False, None
                for j, ch in enumerate(body):
                    if ch == "'":
                        q = not q
                    elif ch == '/' and not q:
                        cpos = j
                        break
                comment = body[cpos + 1:].strip() if cpos is not None else ''
                out[key] = (_fits_value(body), comment)
    out['HISTORY'] = hist
    return out


def card(cards, key, default=None):
    v = cards.get(key)
    return v[0] if v is not None else default


def image_data(path):
    """2-D float32 data of a FITS image (leading axes of length 1 squeezed)."""
    import numpy as np
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        return np.asarray(np.squeeze(h[0].data), dtype=np.float32)


def is_blank_frame(path, frac=None):
    """(blank, finite fraction) by the flow module's own rule (phase3_snapshot_flow.py:95-100): finite fraction below
    BLANK_FINITE_FRACTION, or every finite pixel zero."""
    import numpy as np
    frac = cfg().flow_snapshot.BLANK_FINITE_FRACTION if frac is None else frac
    img = image_data(path)
    fin = np.isfinite(img)
    ff = float(fin.mean()) if img.size else 0.0
    return (ff < frac or not np.any(img[fin] != 0)), ff


def zero_visibility(path):
    """True when the frame's own header records WSCNVIS = 0 (v9:162-175). An unreadable header counts as 'has
    visibilities' (B-23 default (c), flow_snapshot.UNREADABLE_HEADER_HAS_VIS); returns (zero, readable)."""
    try:
        from astropy.io import fits
        v = fits.getheader(path).get('WSCNVIS', 1)
    except Exception:
        if cfg().flow_snapshot.UNREADABLE_HEADER_HAS_VIS:
            return False, False
        raise
    return (v is not None and float(v) == 0), True


# ---------------------------------------------------------------------------------------------------------------------
# Files, digests, listings
# ---------------------------------------------------------------------------------------------------------------------

def sha1_file(path, block=1 << 20):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


def same_bytes(a, b):
    try:
        if os.path.getsize(a) != os.path.getsize(b):
            return False
    except OSError:
        return False
    return sha1_file(a) == sha1_file(b)


def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


def listing(files):
    """[[path, size, mtime_ns], ...] sorted: the identity of a large set of inputs (staged frames, window stacks)."""
    out = []
    for p in sorted(set(files)):
        try:
            st = os.stat(p)
            out.append([os.path.abspath(p), st.st_size, st.st_mtime_ns])
        except FileNotFoundError:
            out.append([os.path.abspath(p), None, None])
    return out


def add_listing_input(u, json_path, files, role, dry_run=False):
    """Record a large input set as ONE small JSON listing (path, size, mtime_ns per file) added as an input of the unit, so
    --resume notices any change of the set or of a member (identity by size + mtime: hashing ~10,000 frames per cell is
    not affordable; recorded as such). In a dry run nothing is written."""
    lst = listing(files)
    if dry_run:
        u.meta.setdefault('listings', {})[role] = {'n': len(lst), 'path': json_path}
        return lst
    write_json_atomic(json_path, {'role': role, 'files': lst})
    u.add_input(json_path, extra={'role': f'listing:{role}', 'n': len(lst)})
    return lst


def add_code_inputs(u, paths):
    """Every script a unit shells out to (and the in-place modules it imports in a child) is an input with role 'code'
    (record.code_fingerprint sees only in-process modules; DESIGN 11.5)."""
    for p in paths:
        if p and os.path.exists(p):
            u.add_input(p, extra={'role': 'code'})


# ---------------------------------------------------------------------------------------------------------------------
# Site file: interpreters, legacy code roots, GPUs and their locks (spec 3.1 blocker B5)
# ---------------------------------------------------------------------------------------------------------------------

def host():
    return socket.gethostname()


def legacy_script(root_name, rel):
    """Absolute path of a clv15 / exopipe_ops file called in place (core.env.legacy_root; refuses when absent)."""
    from exopipe.phase3.core import env
    try:
        p = os.path.join(env.legacy_root(root_name), rel)
    except KeyError as e:
        raise Refusal(str(e))
    if not os.path.exists(p):
        raise Refusal(f'{p} not found (site LEGACY_CODE_ROOTS)')
    return p


def site_gpu_ids(h=None):
    ids = (getattr(cfg().site, 'GPU_IDS', None) or {}).get(h or host())
    return [str(x) for x in ids] if ids else []


def site_gpu_locks(gpu, h=None):
    """Lock files to hold around a GPU child on (host, gpu): site GPU_LOCKS {host: {gpu: [paths]} | [paths]}."""
    ent = (getattr(cfg().site, 'GPU_LOCKS', None) or {}).get(h or host())
    if ent is None:
        return []
    if isinstance(ent, dict):
        v = ent.get(gpu, ent.get(str(gpu), ent.get('*', [])))
    else:
        v = ent
    return [v] if isinstance(v, str) else list(v or [])


def parse_gpus(spec):
    return [g.strip() for g in str(spec).split(',') if g.strip()] if spec else []


@contextlib.contextmanager
def hold_locks(paths, wait_s, log=None):
    """Hold exclusive flock()s on every path (in order) for the duration; wait up to wait_s for each. An existing lock file
    is opened read-only (flock needs no write access, so a production lock can be shared without writing under a
    production root); a missing one is created only outside every production root, else refused."""
    held = []
    try:
        for p in paths:
            if os.path.exists(p):
                fd = os.open(p, os.O_RDONLY)
            else:
                if cfg().production_root_of(p) is not None and not cfg().paths.PRODUCTION_WRITES:
                    raise Refusal(f'GPU lock file {p} does not exist and lies under a production root - not created '
                                  f'(refactor window); create it or fix the site GPU_LOCKS')
                os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
                fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o664)
            t0 = time.time()
            if log:
                log(f'waiting for GPU lock {p}')
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as e:
                    if e.errno not in (errno.EAGAIN, errno.EACCES):
                        os.close(fd)
                        raise
                    if time.time() - t0 > wait_s:
                        os.close(fd)
                        raise Refusal(f'GPU lock {p} not obtained within {wait_s} s')
                    time.sleep(2.0)
            held.append((p, fd, round(time.time() - t0, 1)))
            if log:
                log(f'GPU lock acquired {p} after {held[-1][2]} s')
        yield [(p, w) for p, _, w in held]
    finally:
        for _, fd, _ in reversed(held):
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def tail(path, n=20):
    try:
        with open(path, errors='replace') as f:
            return f.read().splitlines()[-n:]
    except OSError:
        return []


class Tee:
    """Context manager: copy everything written to sys.stdout into a list (the in-place module's own lines are kept in the
    unit record) while still printing it."""

    def __init__(self, keep=400):
        self.lines, self.keep, self._buf = [], keep, ''

    def write(self, s):
        self._real.write(s)
        self._buf += s
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            self.lines.append(line)
            if len(self.lines) > 4 * self.keep:
                del self.lines[:-self.keep]

    def flush(self):
        self._real.flush()

    def __enter__(self):
        self._real = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *exc):
        if self._buf:
            self.lines.append(self._buf)
            self._buf = ''
        sys.stdout = self._real
        return False

    def last(self, pattern, default=None):
        rx = re.compile(pattern)
        for line in reversed(self.lines):
            if rx.search(line):
                return line.strip()
        return default


# ---------------------------------------------------------------------------------------------------------------------
# Supersede (never delete): --fresh --supersede TAG
# ---------------------------------------------------------------------------------------------------------------------

def supersede(paths, base, out_root, tag, what):
    """Move each path (file or directory) under base to <out_root>/_superseded/<tag>/<path relative to base>; append one
    row per move to <out_root>/_superseded/<tag>/MANIFEST.csv (src, dst, bytes, mtime_ns, utc, what). Returns the moves."""
    if not tag:
        raise Refusal('--fresh would replace earlier outputs: give --supersede TAG (they are moved aside, never deleted)')
    dest_root = os.path.join(out_root, '_superseded', tag)
    man = os.path.join(dest_root, 'MANIFEST.csv')
    moves = []
    for p in paths:
        if not os.path.lexists(p):
            continue
        rel = os.path.relpath(os.path.abspath(p), os.path.abspath(base))
        if rel.startswith('..'):
            rel = os.path.basename(p)
        dst = os.path.join(dest_root, rel)
        if os.path.lexists(dst):
            raise Refusal(f'supersede: {dst} exists already (tag {tag!r} used before) - choose another --supersede TAG')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        st = os.lstat(p)
        size = st.st_size if not os.path.isdir(p) else sum(
            os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
        os.rename(p, dst)
        moves.append((p, dst, size, st.st_mtime_ns))
    if moves:
        new = not os.path.exists(man)
        with open(man, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['src', 'dst', 'bytes', 'mtime_ns', 'utc', 'what'])
            for s, d, b, m in moves:
                w.writerow([s, d, b, m, utcstamp(), what])
    return moves


# ---------------------------------------------------------------------------------------------------------------------
# The drift guard (spec section 3). Protected keys are config.PROTECTED_KEYS of each section (config.py refuses them).
# ---------------------------------------------------------------------------------------------------------------------

def _base_key(dotted):
    return dotted.split('[', 1)[0].strip()


def overridden(dotted):
    return any(_base_key(o[0]) == dotted for o in cfg().overrides)


def drift_guard(pairs, note):
    """pairs: [(dotted config key, live module value, assign(value) or None)]. A config value that differs from the live
    in-place module value REFUSES (the owning chat changed production; the config must follow) unless the key was
    overridden explicitly: then the value is assigned into the module when possible (recorded), else refused because the
    override cannot reach a module that runs in a child process."""
    c = cfg()
    for dotted, live, assign in pairs:
        want = c.get(dotted)
        same = (want == live) or (isinstance(want, (tuple, list)) and isinstance(live, (tuple, list))
                                  and list(want) == list(live))
        if same:
            continue
        if not overridden(dotted):
            raise Refusal(f'drift guard: config {dotted} = {want!r} but the in-place production module has {live!r} - the '
                          f'owning chat changed production; update the flow config (or override the key explicitly)')
        if assign is None:
            raise Refusal(f'{dotted} was overridden to {want!r}, but the in-place module that uses it runs in a child '
                          f'process and cannot take the override during the refactor window')
        assign(want)
        note(f'drift guard: {dotted} overridden {live!r} -> {want!r}, assigned into the in-place module')
