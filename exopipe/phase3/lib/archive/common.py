"""Shared primitives of `exopipe3 archive` (spec archive.md sections 5 and 7.2 item 5).

A file of the archive is changed in exactly two ways here:
- ``quarantine_move``: rename on ONE filesystem into a quarantine Run, inode + size verified, moved back on mismatch -
  verbatim logic of ops/p3s5_quarantine_frames.py:58-67 (gates G3 same filesystem, G4 destination free, G5 move verified);
- ``verified_copy`` + ``commit_part``: copy to '<dst>.part', checksum against the source, then rename into place - the
  procedure of ops/push_mosaic_v2_20260929.py:68-72 and ops/p3s5_archive_sync.py:96-100 (gate G6; on a mismatch only the
  '.part' file this step just wrote is removed and nothing live is touched).
Nothing is ever deleted (archive.NEVER_DELETE).
"""
import csv
import hashlib
import os
import re
import time

from exopipe.phase3.core.record import Refusal

csv.field_size_limit(10 ** 9)


def cfg():
    from exopipe.phase3.config import CFG
    return CFG


def sec(name='archive'):
    return getattr(cfg(), name)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def stamp():
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())


def expand(template, **kw):
    """Expand a config template: {ops} paths.OPS_ROOT, {clv15} paths.CLV15_ROOT, {state} archive.STATE_DIR, {batch_root}
    archive_snapshot.BATCH_ROOT, plus the keywords given ({root}, {run}, {round}, {cell}, {stokes}, ...)."""
    if template is None:
        return None
    C = cfg()
    base = {'ops': C.paths.OPS_ROOT.rstrip('/'), 'clv15': C.paths.CLV15_ROOT.rstrip('/')}
    if '{state}' in template:
        base['state'] = C.archive.STATE_DIR.format(**base)
    if '{batch_root}' in template:
        base['batch_root'] = C.archive_snapshot.BATCH_ROOT.format(**base)
    base.update(kw)
    return template.format(**base)


def resolve_root(entry):
    """'paths.X' -> CFG.paths.X; any other string is a literal path."""
    if isinstance(entry, str) and entry.startswith('paths.'):
        return getattr(cfg().paths, entry.split('.', 1)[1])
    return entry


# ---------------------------------------------------------------------------------------------------------------------
# checksums
# ---------------------------------------------------------------------------------------------------------------------

def checksum(path, algo='md5'):
    """md5 or sha256 hex digest of a file, read in archive.CHECKSUM_BLOCK_BYTES blocks."""
    h = hashlib.new(algo)
    block = int(sec().CHECKSUM_BLOCK_BYTES)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# the run's roots and the window write guard (G2)
# ---------------------------------------------------------------------------------------------------------------------

def production_like(path):
    """The production tree that holds ``path`` (paths.PRODUCTION_ROOTS, plus paths.OPS_ROOT and paths.CLV15_ROOT, which hold
    today's manifests, state and the snapshot ship tree), or None."""
    C = cfg()
    root = C.production_root_of(path)
    if root:
        return root
    real = os.path.realpath(path)
    for r in (C.paths.OPS_ROOT, C.paths.CLV15_ROOT):
        rr = os.path.realpath(r)
        if real == rr or real.startswith(rr.rstrip('/') + '/'):
            return r
    return None


class Ctx:
    """Roots and switches of one `exopipe3 archive` invocation."""

    def __init__(self, args, run):
        C = cfg()
        self.args, self.run = args, run
        self.tier = getattr(args, 'tier', None) or 'deep'
        self.action = args.action
        self.archive_root = os.path.abspath(getattr(args, 'archive_root', None) or C.paths.PHASE3_ARCHIVE_ROOT)
        self.production_archive = os.path.realpath(self.archive_root) == os.path.realpath(C.paths.PHASE3_ARCHIVE_ROOT)
        self.gen_run = getattr(args, 'generation_run', None) or C.paths.ARCHIVE_RUN
        self.out = os.path.abspath(run.out_root)
        self.dry = bool(run.dry_run)
        self.apply = bool(getattr(args, 'apply', False))
        self.stamp = stamp()
        self.algo = C.archive.CHECKSUM.get(self.tier, 'md5')
        if not self.gen_run:
            raise Refusal('G18: no archive generation Run named (paths.ARCHIVE_RUN / --generation-run empty)')

    # ---- where manifests go ---------------------------------------------------------------------------------------
    def manifest_dir(self, kind):
        """Plans and window manifests: <out>/manifests/<kind>/ (never a production directory)."""
        return os.path.join(self.out, 'manifests', kind)

    def applied_manifest_dir(self, kind, production_template=None):
        """Applied manifests: where production keeps them when the move happens in the production archive (today's
        consumers read them there: rollback, sync, the fleet), else under <out>/manifests/<kind>/."""
        if self.production_archive and production_template:
            return expand(production_template)
        return self.manifest_dir(kind)

    def quarantine_run(self, name):
        return os.path.join(self.archive_root, name)

    # ---- G2 -------------------------------------------------------------------------------------------------------
    def guard_writes(self, paths, what):
        """G2: refuse an apply that would write or move anything under a production tree while paths.PRODUCTION_WRITES is
        False (refactor window; INCIDENT-DRY-TEST-WROTE-FOUR-FILES-INTO-THE-REAL-ARCHIVE-REMOVED, ledger 897)."""
        if cfg().paths.PRODUCTION_WRITES:
            return
        for p in paths:
            if p and production_like(p):
                raise Refusal(f'G2: {what} would write under the production tree {production_like(p)} ({p}) and '
                              f'paths.PRODUCTION_WRITES is False (refactor window: archive is plan-only; an apply runs '
                              f'only against a mirror, --archive-root DIR outside every production tree)')


# ---------------------------------------------------------------------------------------------------------------------
# moves and copies
# ---------------------------------------------------------------------------------------------------------------------

def _existing_ancestor(path):
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return p


def same_filesystem(src, dest):
    """G3: the source and the destination's nearest existing ancestor are on one filesystem (st_dev)."""
    return os.stat(src).st_dev == os.stat(_existing_ancestor(dest)).st_dev


def quarantine_move(src, dest, algo='md5', what='quarantine'):
    """Move ``src`` to ``dest`` by rename (p3s5_quarantine_frames.py:58-67): G4 destination free, G3 same filesystem,
    checksum of the source BEFORE the move, G5 inode + size equal after it (else moved back and refused).
    Returns (bytes, checksum, moved_at)."""
    if os.path.lexists(dest):
        raise Refusal(f'G4: {what} destination exists: {dest} - STOPPED')
    if sec().SAME_FILESYSTEM_ONLY and not same_filesystem(src, dest):
        raise Refusal(f'G3: {src} and {dest} are on different filesystems - a quarantine is a rename, never a copy '
                      f'(INCIDENT-QUARANTINE-CROSS-FS-MOVE-TRUNCATED-TWO-COPIES-RESTORED) - STOPPED')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    m = checksum(src, algo)
    s0 = os.stat(src)
    os.rename(src, dest)
    s1 = os.stat(dest)
    if (s1.st_ino, s1.st_size) != (s0.st_ino, s0.st_size):
        os.rename(dest, src)
        raise Refusal(f'G5: move of {src} not verified; moved back - STOPPED')
    return s0.st_size, m, utc()


def verified_copy(src, dst, algo='md5', want=None, preserve=True):
    """Copy ``src`` to '<dst>.part' (copy2 when ``preserve``, as p3s5_archive_sync.py:97; copyfile otherwise, as
    push_mosaic_v2:69) and check its checksum against ``want`` (default: the source's now). G6: on a mismatch the '.part'
    is removed and the copy refused; nothing live is touched. Returns (part path, checksum)."""
    import shutil
    m = want or checksum(src, algo)
    part = dst + '.part'
    if os.path.lexists(part):
        raise Refusal(f'G6: a stale {part} exists (an earlier transfer did not finish) - inspect it; never overwritten')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    (shutil.copy2 if preserve else shutil.copyfile)(src, part)
    got = checksum(part, algo)
    if got != m:
        os.remove(part)
        raise Refusal(f'G6: copy of {src} not verified ({algo} {got} != {m}) - STOPPED, nothing replaced')
    return part, m


def commit_part(part, dst, replace=False):
    """Rename '<dst>.part' into place. ``replace`` False refuses an existing destination (never overwrite an archive file;
    push_mosaic_v2:66-67); True uses os.replace (the old file was quarantined first; p3s5_archive_sync.py:100)."""
    if not replace and os.path.lexists(dst):
        raise Refusal(f'push: {dst} exists - never overwritten')
    os.replace(part, dst)


def free_space_ok(path, floor_gb):
    """G19: free space on the filesystem of ``path`` (nearest existing ancestor) >= floor_gb (1e9 bytes); 0 = no check."""
    if not floor_gb:
        return True, None
    st = os.statvfs(_existing_ancestor(path))
    free = st.f_bavail * st.f_frsize / 1e9
    return free >= float(floor_gb), round(free, 1)


# ---------------------------------------------------------------------------------------------------------------------
# rulings (G10), approved manifests (G11), writers (G12)
# ---------------------------------------------------------------------------------------------------------------------

_ID_TOKEN = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:+/-]{5,}')


def ruling_ids(texts):
    """Ledger IDs (rows of paths.FIX_LEDGER) found in the ruling texts."""
    from exopipe.phase3.core import ledger
    ids = ledger.ids(cfg().paths.FIX_LEDGER)
    found = []
    for t in texts or ():
        for tok in _ID_TOKEN.findall(str(t)):
            tok = tok.strip('.,;:')
            if tok in ids and tok not in found:
                found.append(tok)
    return found


def require_ruling(texts, what):
    """G10 (never waivable): an apply of a ruled action needs --apply-ruling "<Gregg's words, UTC, ledger ID>" whose ledger ID
    is a row of paths.FIX_LEDGER. Returns the IDs found."""
    if not texts:
        raise Refusal(f'G10: {what} needs Gregg\'s go: --apply-ruling "<Gregg\'s words, UTC, ledger ID>" (CLAUDE.md 8; '
                      f'archive.RULING_REQUIRED)')
    found = ruling_ids(texts)
    if not found:
        raise Refusal(f'G10: {what}: no ledger ID of {cfg().paths.FIX_LEDGER} found in --apply-ruling {texts!r}')
    return found


def compare_to_manifest(planned, manifest_rows, path_col, size_col=None, sum_col=None, algo='md5', sums=None):
    """G11: the plan re-derived now must equal the approved manifest: same path set, same sizes, and the same checksums
    where the manifest has them. ``planned``: {path: size}. ``sums``: {path: checksum} computed now (optional; computed
    here when the manifest has a checksum column). Returns a list of differences (empty = equal)."""
    diffs = []
    man = {r[path_col]: r for r in manifest_rows}
    extra, missing = sorted(set(planned) - set(man)), sorted(set(man) - set(planned))
    if extra:
        diffs.append(f'{len(extra)} file(s) planned now but not in the manifest, e.g. {extra[0]}')
    if missing:
        diffs.append(f'{len(missing)} manifest row(s) not planned now (moved or no longer excluded), e.g. {missing[0]}')
    for p in sorted(set(planned) & set(man)):
        r = man[p]
        if size_col and str(r.get(size_col, '')).strip() and int(r[size_col]) != int(planned[p]):
            diffs.append(f'{p}: size {planned[p]} != manifest {r[size_col]}')
        if sum_col and str(r.get(sum_col, '')).strip():
            got = (sums or {}).get(p) or checksum(p, algo)
            if got != r[sum_col]:
                diffs.append(f'{p}: {algo} {got} != manifest {r[sum_col]}')
    return diffs


def writer_claim(cell, stokes):
    """G12: (live, owner) for the search fleet's claim of (cell, Stokes) (p3s5_search_fleet.sh:261-266). A claim on this
    host is live when its pid runs; a claim on another host cannot be checked from here and counts as live."""
    p = expand(sec().WRITER_CLAIM, cell=cell, stokes=stokes)
    if not os.path.exists(p):
        return False, ''
    try:
        owner = open(p).read().strip()
    except OSError:
        return True, '(unreadable owner file)'
    host, _, pid = owner.partition(':')
    from exopipe.phase3.core import env
    if host == env.host() and pid.isdigit():
        try:
            os.kill(int(pid), 0)                 # signal 0: existence probe of that exact pid, nothing is sent
            return True, owner
        except ProcessLookupError:
            return False, owner + ' (dead)'
        except PermissionError:                  # the pid exists (another user's process): live
            return True, owner
    return True, owner


def refuse_live_writers(ctx, cells_stokes, what):
    """G12 on the PRODUCTION archive (the fleet's claims name production cells; a mirror has no concurrent writer): refuse
    when the search fleet holds a live claim of any (cell, Stokes)."""
    if not ctx.production_archive:
        return
    for cell, st in cells_stokes:
        live, owner = writer_claim(cell, st)
        if live:
            raise Refusal(f'G12: {what}: the search fleet holds a live claim of {cell} {st} ({owner}) - a concurrent writer')


def excluded_key(ex, path, root):
    """(lst, date, scope) when ``path`` (<root>/<LST>/<date>/<run>/<scope>/...) is a file of an EXCLUDED key under
    core.exclusions (the only reader of data_decisions.csv), else None. Exclusion is absolute (CLAUDE.md 2): no action
    may put such a file back into the generation Run (sync publish / replace, rollback - gate G20)."""
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root)).split(os.sep)
    if len(rel) < 5 or rel[0] == '..':
        return None
    if ex.is_excluded(rel[0], rel[3], rel[1]):
        return rel[0], rel[1], rel[3]
    return None


def intermediate(name):
    """True when a file name is an intermediate product never pushed (archive_push.INTERMEDIATE_PATTERN)."""
    return bool(re.search(sec('archive_push').INTERMEDIATE_PATTERN, os.path.basename(name)))


# ---------------------------------------------------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------------------------------------------------

def read_csv(path):
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        return list(rd.fieldnames or []), rows


def write_csv(path, columns, rows):
    """Write a CSV atomically (tmp + rename); extra keys in rows are ignored."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)
    return path


class CsvAppender:
    """Append rows to a manifest, one row per move, flushed and fsynced before the next move (a refusal mid-unit leaves
    every completed move recorded, so a rollback can undo a partial unit). Header written when the file is new; an existing
    file must carry the same header."""

    def __init__(self, path, columns):
        self.path, self.columns = path, list(columns)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        if not new:
            have, _ = read_csv(path)
            if have != self.columns:
                raise Refusal(f'manifest {path} has columns {have}, expected {self.columns} - not appended')
        self.f = open(path, 'a', newline='')
        self.w = csv.DictWriter(self.f, fieldnames=self.columns, extrasaction='ignore')
        if new:
            self.w.writeheader()
            self._sync()
        self.n = 0

    def _sync(self):
        self.f.flush()
        os.fsync(self.f.fileno())

    def write(self, row):
        self.w.writerow(row)
        self._sync()
        self.n += 1

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def write_json(path, obj):
    import json
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)
    return path


def read_json(path):
    import json
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------------------
# archive tree helpers
# ---------------------------------------------------------------------------------------------------------------------

def epoch_dir(root, run, lst, date, scope, stokes, cadence):
    return sec().EPOCH_TREE.format(root=root, run=run, lst=lst, date=date, scope=scope, stokes=stokes, cadence=cadence)


def deep_frames_glob(root, run, lst, scope, stokes):
    """<root>/<LST>/*/<run>/<scope>/<S>/deep/*.fits (p3s5_quarantine_frames.py:50; p3s5_archive_sync.py:60)."""
    return epoch_dir(root, run, lst, '*', scope, stokes, 'deep') + '/*.fits'


def rel_to(path, root):
    return os.path.relpath(path, root)


def stokes_list(spec, default=('V', 'I')):
    """--stokes I,V -> ('V', 'I') order of the search fleet (V before I) unless given otherwise."""
    if not spec:
        return tuple(default)
    out = [s.strip().upper() for s in str(spec).split(',') if s.strip()]
    bad = [s for s in out if s not in ('I', 'V')]
    if bad:
        raise ValueError(f'--stokes: {bad} (I or V)')
    return tuple(dict.fromkeys(out))
