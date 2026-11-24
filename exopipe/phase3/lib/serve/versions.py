"""Run versions and the served-run pointer (plan/CUTOVER_REMAINING_20261124.md row 8; the design item of 2026-11-15 in
plan/CUTOVER_READINESS.md:333-344; plan/ORCHESTRATION_DESIGN_20261117.md section 5.3 "Publishing": "Run versions plus an atomic
pointer").

A served run U - the path every registry entry names in --run, --queue, --labels-dir and --catalog-dir, and the path the
cross-tier links read (classify_transients.py builds /data07/gh/phase3/transients/<run>/source_catalogs_<S>) - is a POINTER: a
symbolic link to one complete, sealed VERSION of the run. A build writes a new version beside the served one; the pointer is
switched by one atomic rename (a new link written under a temporary name, os.replace over U), so a reader that opens U/<file>
gets the old version or the new one, never a run half written. The registry, its entries and the served URLs (/p/<id>/) do not
change: U keeps its path.

LAYOUT (name = basename(U); R = the VERSIONS ROOT, <dir>/<name><serve.RUN_VERSIONS_SUFFIX>, beside the run's real directory):
  R/<vid>/<name>/          the run of version <vid> (vid = the UTC time it was begun, %Y%m%dT%H%M%SZ, '-2', '-3' ... when two
                           begin in one second; an adopted legacy run: the time of its directory, '-adopted'). The run directory
                           keeps the run's NAME, so a reader that takes os.path.realpath(U) - classify_transients.py main (the
                           page's run_dir, whose basename keys every judgement row), build_detections_index.py,
                           page_context_tenmin.py - sees the name it always saw.
  R/<vid>/BUILDING.json    begin(): who builds it (pid, process start, host, UTC)
  R/<vid>/MANIFEST.tsv     seal(): every file and link of the run - kind (f / l), relative path, size in bytes, mtime in ns,
                           link target - sorted by path; but the REVIEW STATE the pages and the serve append to while the version
                           is served (serve.RUN_VERSIONS_MUTABLE, recorded in VERSION.json: the judgement journals, the run's
                           incremental.log, the page index's DIRTY marker, a publish marker), so a version stays verifiable
                           after it was served
  R/<vid>/VERSION.json     seal(), written LAST and atomically: THE COMPLETION RECORD (name, version, sealed UTC, file and byte
                           counts, the manifest's sha256, the required paths and the caller's content checks with their results)
  R/<vid>/holds/<id>.json  hold(): a reader of the version (a build that reads it); retire() never deletes a held version
  R/POINTER_LOG.jsonl      one line per switch (UTC, from, to, the switcher)
  R/RETIRE_LOG.jsonl       one line per retired version
  R/retire_plans/          the dry runs of retire()
  R/.lock                  flock: switch() and retire() exclusive
U -> <name><suffix>/<vid>/<name> (relative when R is beside U, else absolute).

COMPLETION. A version is complete only when seal() wrote its VERSION.json, and seal() writes it only when the run directory
holds at least one file, every required path exists (a file non-empty), every content check passed and no path holds a tab or
a newline. switch() REFUSES (Refusal: nothing switched, the served version untouched) a version without a completion record,
whose record or manifest does not parse or does not match (sha256), or whose run on disk differs from its manifest - a file
missing, added, of another size or modification time (verify(): one walk of the run). Names, counts or a marker alone never
make a version complete.

ADOPTION. The first switch of a run that is still a plain directory (today's Run_SLSTB) or a link to one (Run_P3S5 ->
/fast/gh/main/phase3/Run_P3S5) first moves that directory into R as version <its mtime>-adopted (a rename on its own file
system; R lies beside the real directory), then switches. Between the rename and the switch, about a millisecond once per
run, U does not exist; a page already built keeps its own realpath (the page code resolves --run once) and reads on.

RETENTION (retire). Kept: the version the pointer names (and any version another link in U's directory names), the newest
serve.RUN_VERSIONS_KEEP sealed or adopted versions, every version with a live hold or a live builder (a hold or builder on
another host counts as live: it cannot be checked from here). Everything else - older versions and abandoned unsealed builds -
is retired, ONLY through a dry run first: retire(U) writes the plan (R/retire_plans/RETIRE_PLAN_<UTC>.json: the versions and
their sizes) and deletes nothing; retire(U, plan=<that file>) re-derives the plan under the lock and refuses (nothing deleted)
unless every planned version is still to be retired; each one is renamed out of R's names (R/.retiring_<vid>) before it is
deleted, so a half-deleted version is never taken for a version.

HOLDS. A build that reads other runs resolves each pointer ONCE (hold(): the version it names, held until the build ends) and
reads that version's directory, so a switch during the build cannot mix two versions in it (CUTOVER_READINESS.md:333-344 (b),
(d)).
"""
import contextlib
import fcntl
import fnmatch
import hashlib
import json
import os
import re
import shutil
import socket
import time

from exopipe.phase3.core.record import Refusal

FORMAT = 1
_VID_RE = re.compile(r'^\d{8}T\d{6}Z(-\d+)?(-adopted)?$')
_BAD_NAME = re.compile(r'[\t\n\r]')


def _S():
    from exopipe.phase3.config import CFG
    return CFG.serve


def _guard(path, what):
    from exopipe.phase3.lib.serve import register
    register.guard(path, what)


def _utc(t=None):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() if t is None else t))


def _atomic_json(path, obj):
    from exopipe.phase3.core.record import atomic_write_json
    atomic_write_json(path, obj, durable=True)


def _append_jsonl(path, obj):
    with open(path, 'a') as f:
        f.write(json.dumps(obj, sort_keys=True) + '\n')


# ---- who is running -------------------------------------------------------------------------------------------------
def _proc_start(pid):
    """The process's start time in clock ticks since boot (/proc/<pid>/stat field 22), or None when it does not run."""
    try:
        with open(f'/proc/{int(pid)}/stat') as f:
            s = f.read()
        return int(s[s.rindex(')') + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def me(note=''):
    """The calling process as a holder or builder: pid, its start time, host, UTC, note."""
    pid = os.getpid()
    return {'pid': pid, 'start': _proc_start(pid), 'host': socket.gethostname(), 'utc': _utc(), 'note': str(note)[:300]}


def alive(who):
    """True when the holder/builder ``who`` (a me() dict) may still run: on this host, its pid runs with the same start time;
    on another host, always (it cannot be checked from here)."""
    if not who:
        return False
    if who.get('host') != socket.gethostname():
        return True
    st = _proc_start(who.get('pid', -1))
    return st is not None and (who.get('start') is None or st == who.get('start'))


# ---- where ----------------------------------------------------------------------------------------------------------
class Version:
    """One version of a run: root R, vid, dir R/<vid>, run R/<vid>/<name>."""

    def __init__(self, root, vid, name):
        self.root, self.vid, self.name = root, vid, name
        self.dir = os.path.join(root, vid)
        self.run = os.path.join(self.dir, name)

    def path(self, what):
        return os.path.join(self.dir, what)

    def record(self):
        """The completion record (VERSION.json) or None."""
        try:
            with open(self.path('VERSION.json')) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def builder(self):
        try:
            with open(self.path('BUILDING.json')) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def holds(self):
        """[(hold file, holder)] of the version."""
        d, out = self.path('holds'), []
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else ():
            try:
                with open(os.path.join(d, f)) as fh:
                    out.append((os.path.join(d, f), json.load(fh)))
            except (OSError, ValueError):
                out.append((os.path.join(d, f), {'unreadable': True, 'host': None}))
        return out

    def __repr__(self):
        return f'Version({self.run})'


def _suffix():
    return _S().RUN_VERSIONS_SUFFIX


def parse_target(U, target=None):
    """The Version a link target names when it has the pointer form <R>/<vid>/<name> with R = <dir>/<name><suffix>, else
    None. ``target`` defaults to os.readlink(U)."""
    name = os.path.basename(os.path.normpath(U))
    if target is None:
        if not os.path.islink(U):
            return None
        target = os.readlink(U)
    full = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(U)), target))
    vdir, run_name = os.path.split(full)
    root, vid = os.path.split(vdir)
    if run_name != name or os.path.basename(root) != name + _suffix() or not _VID_RE.match(vid):
        return None
    return Version(os.path.realpath(root), vid, name)


def is_pointer(U):
    return parse_target(U) is not None


def root_of(U):
    """The versions root R of U: the pointer's own when U is one; else beside U's real directory (a run that is a link to a
    directory elsewhere, like Run_P3S5 -> /fast/gh/main/phase3/Run_P3S5, keeps its versions on that file system)."""
    v = parse_target(U)
    if v is not None:
        return v.root
    name = os.path.basename(os.path.normpath(U))
    real = os.path.realpath(U) if os.path.lexists(U) else os.path.join(os.path.realpath(os.path.dirname(os.path.abspath(U))),
                                                                        name)
    return os.path.join(os.path.dirname(real), name + _suffix())


def served(U):
    """The Version U names now (None when U is not a pointer)."""
    return parse_target(U)


def all_versions(U):
    """Every Version under U's versions root, oldest first (vid order)."""
    R, name = root_of(U), os.path.basename(os.path.normpath(U))
    if not os.path.isdir(R):
        return []
    return [Version(R, v, name) for v in sorted(os.listdir(R)) if _VID_RE.match(v) and os.path.isdir(os.path.join(R, v))]


@contextlib.contextmanager
def _lock(R):
    os.makedirs(R, exist_ok=True)
    fd = os.open(os.path.join(R, '.lock'), os.O_RDWR | os.O_CREAT, 0o664)   # close-on-exec (Python's default)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ---- a new version ----------------------------------------------------------------------------------------------------
def _new_vid_dir(R, stamp):
    os.makedirs(R, exist_ok=True)
    for n in range(1, 1000):
        vid = stamp if n == 1 else f'{stamp}-{n}'
        try:
            os.mkdir(os.path.join(R, vid))
            return vid
        except FileExistsError:
            continue
    raise Refusal(f'no free version name under {R} for {stamp}')


def begin(U, note=''):
    """A new, empty, unsealed version of U: R/<vid>/<name>/ and its BUILDING.json (this process). Returns the Version."""
    _guard(U, 'a new version of')
    R, name = root_of(U), os.path.basename(os.path.normpath(U))
    vid = _new_vid_dir(R, time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    v = Version(R, vid, name)
    os.mkdir(v.run)
    _atomic_json(v.path('BUILDING.json'), dict(me(note), format=FORMAT, name=name, version=vid, pointer=os.path.abspath(U)))
    return v


def place(U, B, note=''):
    """A new unsealed version of U whose run is the finished build directory ``B``, moved in by one rename (B must lie on the
    versions root's file system; nothing is copied). Returns the Version."""
    _guard(U, 'a new version of')
    if not os.path.isdir(B) or os.path.islink(B):
        raise Refusal(f'no build directory {B} to place as a version of {U}')
    R, name = root_of(U), os.path.basename(os.path.normpath(U))
    os.makedirs(R, exist_ok=True)
    if os.stat(B).st_dev != os.stat(R).st_dev:
        raise Refusal(f'{B} is not on the file system of the versions root {R}: a version is placed by a rename, never a copy')
    vid = _new_vid_dir(R, time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    v = Version(R, vid, name)
    _atomic_json(v.path('BUILDING.json'), dict(me(note), format=FORMAT, name=name, version=vid, pointer=os.path.abspath(U),
                                               placed_from=os.path.abspath(B)))
    os.rename(B, v.run)
    return v


def unplace(v, B):
    """Undo place(): the unsealed version's run moved back to the build directory ``B`` and the version removed (a switch that
    was refused leaves the build where the next attempt looks for it). Refusal for a sealed version or an existing ``B``."""
    _guard(v.run, 'a placed version of')
    if v.record() is not None:
        raise Refusal(f'version {v.vid} is sealed: it stays a version')
    if os.path.lexists(B):
        raise Refusal(f'{B} exists: version {v.vid} not moved back')
    os.rename(v.run, B)
    shutil.rmtree(v.dir)


def placed_abandoned(U, B):
    """The newest unsealed version of U placed from ``B`` whose placing process is gone (a swap killed between its place and
    its switch), or None."""
    for v in reversed(all_versions(U)):
        b = v.builder() or {}
        if v.record() is None and b.get('placed_from') == os.path.abspath(B) and not alive(b) and os.path.isdir(v.run):
            return v
    return None


# ---- the manifest and the completion record ---------------------------------------------------------------------------
def walk(run, mutable=()):
    """[(kind, relative path, size, mtime_ns, link target)] of every file ('f') and link ('l') under ``run`` (links not
    followed) whose relative path matches no pattern of ``mutable`` (fnmatch), sorted by path. Refuses a path holding a tab or a
    newline (the manifest is tab-separated)."""
    out = []
    for d, dirs, files in os.walk(run):
        dirs.sort()
        for n in list(dirs):
            p = os.path.join(d, n)
            if os.path.islink(p):                       # a link to a directory: an entry, never walked into
                dirs.remove(n)
                files.append(n)
        for n in files:
            p = os.path.join(d, n)
            rel = os.path.relpath(p, run)
            if any(fnmatch.fnmatchcase(rel, m) for m in mutable):
                continue
            if _BAD_NAME.search(rel):
                raise Refusal(f'{p}: a tab or newline in a path - the manifest cannot record it')
            st = os.lstat(p)
            if os.path.islink(p):
                out.append(('l', rel, 0, 0, os.readlink(p)))
            else:
                out.append(('f', rel, st.st_size, st.st_mtime_ns, ''))
    out.sort(key=lambda e: e[1])
    return out


def _manifest_text(entries):
    return ''.join(f'{k}\t{rel}\t{size}\t{mt}\t{tgt}\n' for k, rel, size, mt, tgt in entries)


def _read_manifest(path):
    out = []
    with open(path, newline='\n') as f:
        for ln in f:
            k, rel, size, mt, tgt = ln.rstrip('\n').split('\t')
            out.append((k, rel, int(size), int(mt), tgt))
    return out


def seal(v, required=(), checks=(), note=''):
    """Seal version ``v``: its manifest, then its completion record, written only when the run is complete:
    - the run directory exists, is not a link, holds at least one file;
    - every path of ``required`` (absolute, or relative to the run) exists inside the run, a file non-empty;
    - every check of ``checks`` - (name, callable(run) -> (ok, detail)) - passes.
    Refusal (nothing written) otherwise, and for a version already sealed. Returns the record."""
    _guard(v.run, 'the seal of')
    if v.record() is not None:
        raise Refusal(f'version {v.vid} of {v.name} is already sealed ({v.path("VERSION.json")})')
    if not os.path.isdir(v.run) or os.path.islink(v.run):
        raise Refusal(f'version {v.vid}: no run directory {v.run}')
    bad, req = [], []
    for p in required:
        a = p if os.path.isabs(p) else os.path.join(v.run, p)
        rel = os.path.relpath(a, v.run)
        req.append(rel)
        if rel.startswith('..'):
            bad.append(f'{p} lies outside the run')
        elif not os.path.exists(a):
            bad.append(f'{rel} missing')
        elif os.path.isfile(a) and os.path.getsize(a) == 0:
            bad.append(f'{rel} empty')
    if bad:
        raise Refusal(f'version {v.vid} of {v.name} is not complete - not sealed: ' + '; '.join(bad)[:800])
    results = []
    for name, fn in checks:
        ok, detail = fn(v.run)
        results.append({'name': name, 'passed': bool(ok), 'detail': str(detail)[:500]})
        if not ok:
            raise Refusal(f'version {v.vid} of {v.name}: content check {name} failed ({str(detail)[:400]}) - not sealed')
    t0 = time.time()
    mutable = list(_S().RUN_VERSIONS_MUTABLE)
    entries = walk(v.run, mutable)
    if not any(e[0] == 'f' for e in entries):
        raise Refusal(f'version {v.vid} of {v.name}: the run directory holds no file - not sealed')
    text = _manifest_text(entries).encode()
    mpath = v.path('MANIFEST.tsv')
    with open(mpath + '.part', 'wb') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(mpath + '.part', mpath)
    rec = {'format': FORMAT, 'name': v.name, 'version': v.vid, 'state': 'sealed', 'sealed_utc': _utc(),
           'sealed_by': me(note), 'builder': v.builder(), 'n_files': sum(1 for e in entries if e[0] == 'f'),
           'n_links': sum(1 for e in entries if e[0] == 'l'), 'n_bytes': sum(e[2] for e in entries),
           'manifest': 'MANIFEST.tsv', 'manifest_sha256': hashlib.sha256(text).hexdigest(), 'mutable': mutable, 'required': req,
           'checks': results, 'walk_s': round(time.time() - t0, 1)}
    _atomic_json(v.path('VERSION.json'), rec)
    return rec


def verify(v):
    """(ok, detail): ``v`` holds a completion record of this format and name, its manifest's sha256 is the recorded one, and
    the run on disk is exactly its manifest (every file present with its size and mtime, every link with its target, nothing
    added; the record's mutable paths left out). One walk of the run."""
    rec = v.record()
    if rec is None:
        return False, f'version {v.vid}: no completion record {v.path("VERSION.json")} (not sealed: a build not finished)'
    if rec.get('format') != FORMAT or rec.get('name') != v.name or rec.get('version') != v.vid:
        return False, f'version {v.vid}: completion record of format {rec.get("format")}, name {rec.get("name")}, version ' \
                      f'{rec.get("version")}'
    mpath = v.path(rec.get('manifest') or 'MANIFEST.tsv')
    try:
        with open(mpath, 'rb') as f:
            text = f.read()
    except OSError as e:
        return False, f'version {v.vid}: manifest unreadable ({e})'
    if hashlib.sha256(text).hexdigest() != rec.get('manifest_sha256'):
        return False, f'version {v.vid}: manifest sha256 differs from its completion record'
    want = _read_manifest(mpath)
    try:
        have = walk(v.run, rec.get('mutable') or ())
    except (OSError, Refusal) as e:
        return False, f'version {v.vid}: run unreadable ({e})'
    if have == want:
        return True, f'version {v.vid}: {rec.get("n_files")} files, {rec.get("n_bytes")} bytes as sealed {rec.get("sealed_utc")}'
    w, h = {e[1]: e for e in want}, {e[1]: e for e in have}
    missing = sorted(set(w) - set(h))
    added = sorted(set(h) - set(w))
    changed = sorted(p for p in set(w) & set(h) if w[p] != h[p])
    return False, (f'version {v.vid}: the run differs from its manifest - {len(missing)} missing (e.g. {missing[:3]}), '
                   f'{len(added)} added (e.g. {added[:3]}), {len(changed)} changed (e.g. {changed[:3]})')


# ---- the switch -------------------------------------------------------------------------------------------------------
def _link_target(U, v):
    """The pointer's link text: relative when the versions root lies beside U, else absolute."""
    d = os.path.realpath(os.path.dirname(os.path.abspath(U)))
    return os.path.relpath(v.run, d) if os.path.dirname(v.root) == d else v.run


def _adopt(U, R):
    """Move the legacy run U (a directory, or a link to one) into R as an adopted version; returns (Version, the link U was or
    None). The caller holds R's lock and writes the pointer right after."""
    name = os.path.basename(os.path.normpath(U))
    was_link = os.readlink(U) if os.path.islink(U) else None
    real = os.path.realpath(U)
    if not os.path.isdir(real):
        raise Refusal(f'{U} is neither a pointer nor a run directory - nothing switched')
    if os.stat(real).st_dev != os.stat(R).st_dev:
        raise Refusal(f'{real} is not on the file system of the versions root {R} - nothing switched')
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(os.stat(real).st_mtime))
    for n in range(1, 1000):
        vid = f'{stamp}-adopted' if n == 1 else f'{stamp}-{n}-adopted'
        try:
            os.mkdir(os.path.join(R, vid))
            break
        except FileExistsError:
            continue
    v = Version(R, vid, name)
    _atomic_json(v.path('VERSION.json'), {'format': FORMAT, 'name': name, 'version': vid, 'state': 'adopted',
                                          'adopted_utc': _utc(), 'adopted_by': me(), 'adopted_from': real,
                                          'pointer_was_link': was_link, 'manifest': None})
    if was_link is not None:
        os.unlink(U)                                   # the old link first: the run keeps its directory until the rename
    os.rename(real, v.run)
    return v, was_link


def switch(U, v, note=''):
    """Point U at version ``v`` atomically: verify(v) (a refusal switches nothing), then a new link under a temporary name
    renamed over U. The first switch of a legacy run adopts it first (see ADOPTION). Returns {'previous': Version or None,
    'current': Version, 'adopted': Version or None, 'verify': detail}. Refusal when ``v`` is not a version of U's root."""
    _guard(U, 'the pointer')
    name = os.path.basename(os.path.normpath(U))
    if v.name != name:
        raise Refusal(f'version {v.run} is not a version of {U}')
    with _lock(v.root):
        if os.path.realpath(root_of(U)) != os.path.realpath(v.root):
            raise Refusal(f'version {v.run} is not under the versions root of {U} ({root_of(U)})')
        ok, detail = verify(v)
        if not ok:
            raise Refusal(f'not switched: {detail} - {U} still names {os.path.realpath(U) if os.path.lexists(U) else "nothing"}')
        prev, adopted = served(U), None
        if prev is None and os.path.lexists(U):
            adopted, _ = _adopt(U, v.root)
            prev = adopted
        if prev is not None and prev.vid == v.vid:
            raise Refusal(f'{U} already names version {v.vid}')
        tmp = os.path.join(os.path.dirname(os.path.abspath(U)), f'.{name}.pointer.{os.getpid()}')
        if os.path.lexists(tmp):
            os.unlink(tmp)
        os.symlink(_link_target(U, v), tmp)
        os.replace(tmp, U)
        _append_jsonl(os.path.join(v.root, 'POINTER_LOG.jsonl'),
                      {'utc': _utc(), 'pointer': os.path.abspath(U), 'from': prev.vid if prev else None, 'to': v.vid,
                       'adopted': adopted.vid if adopted else None, 'by': me(note)})
    return {'previous': prev, 'current': v, 'adopted': adopted, 'verify': detail}


def publish(U, v, required=(), checks=(), note=''):
    """seal(v) then switch(U, v): the version complete, then served. Returns switch()'s dict with the record added."""
    rec = seal(v, required=required, checks=checks, note=note)
    out = switch(U, v, note=note)
    out['record'] = rec
    return out


# ---- holds ------------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def hold(U, note=''):
    """Resolve U ONCE and hold the version it names for the block: yields the run directory to read (that version's; U itself
    when U is not a pointer - a legacy run, nothing to hold). retire() never deletes a held version."""
    v = served(U)
    if v is None:
        yield os.path.abspath(U)
        return
    with _lock(v.root):                                # retire() decides and deletes under the same lock: the hold is
        v = served(U)                                  # either seen by it or made after it (then on the version U names)
        if v is None or not os.path.isdir(v.dir):
            raise Refusal(f'{U} names no version to hold now ({os.readlink(U) if os.path.islink(U) else "not a pointer"})')
        d = v.path('holds')
        os.makedirs(d, exist_ok=True)
        who = me(note)
        path = os.path.join(d, f'{who["host"]}_{who["pid"]}_{time.time_ns()}.json')
        _atomic_json(path, who)
    try:
        yield v.run
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


# ---- retention --------------------------------------------------------------------------------------------------------
def _linked_vids(U, R):
    """The vids named by a link in U's directory or in R's parent (the pointer and any other link to a version)."""
    out = set()
    for d in {os.path.dirname(os.path.abspath(U)), os.path.dirname(R)}:
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for n in names:
            p = os.path.join(d, n)
            if os.path.islink(p):
                real = os.path.realpath(p)
                if real.startswith(R + os.sep):
                    out.add(os.path.relpath(real, R).split(os.sep)[0])
    return out


def _bytes(v):
    rec = v.record() or {}
    if rec.get('n_bytes') is not None:
        return int(rec['n_bytes'])
    n = 0
    for d, _, files in os.walk(v.dir):
        for f in files:
            with contextlib.suppress(OSError):
                n += os.lstat(os.path.join(d, f)).st_size
    return n


def retention(U, keep=None):
    """[(Version, 'keep' | 'retire', why)] for every version of U, oldest first (see RETENTION)."""
    keep = int(_S().RUN_VERSIONS_KEEP if keep is None else keep)
    if keep < 1:
        raise Refusal(f'keep {keep}: at least the newest version is kept')
    R = root_of(U)
    vs = all_versions(U)
    cur = served(U)
    linked = _linked_vids(U, R)
    done = [v for v in vs if (v.record() or {}).get('state') in ('sealed', 'adopted')]
    newest = {v.vid for v in done[-keep:]}
    out = []
    for v in vs:
        live_holds = [h for _, h in v.holds() if alive(h)]
        b = v.builder()
        state = (v.record() or {}).get('state') or 'unsealed'
        if cur is not None and v.vid == cur.vid:
            out.append((v, 'keep', 'the pointer names it'))
        elif v.vid in linked:
            out.append((v, 'keep', 'a link names it'))
        elif live_holds:
            out.append((v, 'keep', f'held by {len(live_holds)} reader(s), e.g. pid {live_holds[0].get("pid")} on '
                                   f'{live_holds[0].get("host")}'))
        elif state == 'unsealed' and alive(b):
            out.append((v, 'keep', f'being built by pid {b.get("pid")} on {b.get("host")}'))
        elif v.vid in newest:
            out.append((v, 'keep', f'one of the newest {keep} complete versions'))
        else:
            out.append((v, 'retire', 'an abandoned build (never sealed, its builder gone)' if state == 'unsealed' else
                        f'{state}, older than the newest {keep} complete versions'))
    return out


def retire(U, keep=None, plan=None, note=''):
    """THE DRY RUN FIRST. Without ``plan``: write R/retire_plans/RETIRE_PLAN_<UTC>.json (what would be retired, with sizes)
    and delete nothing; returns {'plan': path, 'retire': [...], 'keep': [...]}. With ``plan`` (that file): under R's lock the
    plan is derived again, and a planned version no longer to be retired - or a plan of another pointer or keep - refuses
    (nothing deleted; run the dry run again); else each planned version is renamed to R/.retiring_<vid>, deleted, and logged
    in R/RETIRE_LOG.jsonl. Returns {'retired': [...], 'bytes': n}."""
    _guard(U, 'the versions of')
    R = root_of(U)
    keep = int(_S().RUN_VERSIONS_KEEP if keep is None else keep)
    if plan is None:
        rows = retention(U, keep)
        doc = {'format': FORMAT, 'pointer': os.path.abspath(U), 'root': R, 'keep': keep, 'utc': _utc(), 'by': me(note),
               'serves': (served(U).vid if served(U) else None),
               'retire': [{'version': v.vid, 'why': why, 'bytes': _bytes(v)} for v, act, why in rows if act == 'retire'],
               'keep_versions': [{'version': v.vid, 'why': why} for v, act, why in rows if act == 'keep']}
        os.makedirs(os.path.join(R, 'retire_plans'), exist_ok=True)
        path = os.path.join(R, 'retire_plans', f'RETIRE_PLAN_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}.json')
        _atomic_json(path, doc)
        return {'plan': path, 'retire': doc['retire'], 'keep': doc['keep_versions']}
    with open(plan) as f:
        doc = json.load(f)
    if doc.get('format') != FORMAT or os.path.normpath(doc.get('pointer') or '') != os.path.normpath(os.path.abspath(U)) \
            or os.path.normpath(doc.get('root') or '') != os.path.normpath(R) or int(doc.get('keep', -1)) != keep:
        raise Refusal(f'retire plan {plan} is for pointer {doc.get("pointer")}, root {doc.get("root")}, keep {doc.get("keep")} - '
                      f'not for {U}, {R}, keep {keep}: nothing deleted')
    retired, freed = [], 0
    with _lock(R):
        now = {v.vid: (v, act, why) for v, act, why in retention(U, keep)}
        stale = [p['version'] for p in doc.get('retire', []) if p['version'] not in now or now[p['version']][1] != 'retire']
        if stale:
            why = ', '.join(f'{s}: ' + (now[s][2] if s in now else 'gone') for s in stale[:5])
            raise Refusal(f'retire plan {plan}: {len(stale)} planned version(s) are no longer to be retired ({why}) - nothing '
                          f'deleted; run the dry run again')
        for p in doc.get('retire', []):
            v = now[p['version']][0]
            gone = os.path.join(R, f'.retiring_{v.vid}')
            nb = int(p.get('bytes') or 0)
            os.rename(v.dir, gone)
            shutil.rmtree(gone)
            retired.append(v.vid)
            freed += nb
            _append_jsonl(os.path.join(R, 'RETIRE_LOG.jsonl'), {'utc': _utc(), 'pointer': os.path.abspath(U), 'version': v.vid,
                                                                'bytes': nb, 'plan': os.path.abspath(plan), 'by': me(note)})
    return {'retired': retired, 'bytes': freed}


def describe(U):
    """Dry-run lines: what U is (pointer, legacy directory, nothing) and its versions."""
    out = []
    v = served(U)
    if v is not None:
        out.append(f'  pointer {U} -> version {v.vid} ({(v.record() or {}).get("state", "unsealed")})')
    elif os.path.lexists(U):
        out.append(f'  {U} is a legacy run (no pointer yet): the first switch adopts it as a version under {root_of(U)}')
    else:
        out.append(f'  {U} does not exist: the first switch creates the pointer')
    vs = all_versions(U)
    out.append(f'  versions under {root_of(U)}: {len(vs)} ({", ".join(x.vid for x in vs[-5:])}{" ..." if len(vs) > 5 else ""}); '
               f'retention keeps the newest {_S().RUN_VERSIONS_KEEP} complete versions (serve --stages retire: a dry run first)')
    return out
