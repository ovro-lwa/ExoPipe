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
                           link target, the file's sha256 (CONTENT; empty when not hashed) - sorted by path; but the
                           REVIEW STATE the pages and the serve append to while the version is served
                           (serve.RUN_VERSIONS_MUTABLE, recorded in VERSION.json: the judgement journals, the run's
                           incremental.log, the page index's DIRTY marker, a publish marker), so a version stays verifiable
                           after it was served
  R/<vid>/VERSION.json     seal(), written LAST and atomically: THE COMPLETION RECORD (name, version, sealed UTC, file and byte
                           counts, the manifest's sha256, the required paths and the caller's content checks with their results)
  R/<vid>/holds/<id>.json  hold(): a reader of the version (a build that reads it); retire() never deletes a held version
  R/POINTER_LOG.jsonl      one line per switch (UTC, from, to, the switcher)
  R/CATCHUP_OWED.json      a switch whose catch-up of the old version's journal lines failed (owe_catchup): every later
                           begin/place/switch of U refuses and the old version is retained until catchup_done removes it
  R/RETIRE_LOG.jsonl       one line per retired version
  R/retire_plans/          the dry runs of retire()
  R/locks/journal_<J>.lock flock of the judgement journal directory <J> of every version (journal_locks; the two-port server's
                           POST lock of a labels directory in a version, two_port.journal_lock_path): one file across versions
  R/.lock                  flock: switch() and retire() exclusive
  Lock files are created group-writable (0o664 set explicitly; R/locks setgid, so they take its group: the serve and the hub
  may run as different users of one group); a lock that cannot be opened is a Refusal naming the file and its owner.
U -> <name><suffix>/<vid>/<name> (relative when R is beside U, else absolute).

COMPLETION. A version is complete only when seal() wrote its VERSION.json, and seal() writes it only when the run directory
holds at least one file, every required path exists (a file non-empty), every content check passed and no path holds a tab or
a newline. switch() REFUSES (Refusal: nothing switched, the served version untouched) a version without a completion record,
whose record or manifest does not parse or does not match (sha256), or whose run on disk differs from its manifest - a file
missing, added, of another size or modification time (verify(): one walk of the run; a caller that sealed the version itself
passes the record seal() returned, and its walk stands for the switch's: the record and manifest are checked, the run is not
walked again). Names, counts or a marker alone never make a version complete.

ADOPTION. The first switch of a run that is still a plain directory (today's Run_SLSTB) or a link to one (Run_P3S5 ->
/fast/gh/main/phase3/Run_P3S5) first moves that directory into R as version <its mtime>-adopted (a rename on its own file
system; R lies beside the real directory; an empty run directory is refused, nothing moved), then switches. Between the
rename and the switch, about a millisecond once per run, U does not exist; a page already built keeps its own realpath (the
page code resolves --run once) and reads on. Any failure after the rename and before U points at the new version moves the
run back to where it was (U as before) and removes the adopted version. The adopted version's record (state 'adopted') is
written only after the rename;
seal_adopted() then gives it a manifest (state 'sealed', the adoption kept in the record), so it can be switched back to like
any other version - publish() does so after the switch, outside the lock (one walk of the old run).

LEASES. A holder or builder record (me()) carries lease_until (UTC seconds, serve.RUN_VERSIONS_LEASE_S ahead) and lease_s. On
this host a record is alive while its pid runs with the same start time; on another host until TWICE its lease has passed
(lease_until + lease_s: a renewal delayed by a slow file system is not taken for a death), and a record without lease_until
(written before leases, or unreadable) counts as alive - it is never retired or moved back automatically. hold(), a placed or
begun version's BUILDING.json and the coalescing builder renew their lease in this process (a renewer thread, every quarter
lease: the new record written and fsynced outside the renewer's lock, only its rename under it) until they end, so a reader
or builder on another node is never taken for gone while it runs, and a dead one is not kept for ever. renew_stop(path) returns
only after a renewal of that path in flight has ended (its temporary file removed; the caller then removes or moves the
directory), waiting for it outside the lock: a slow fsync delays only the stop of its own path.

CONTENT. Names, sizes, modification times or markers never prove completion: the content test reads content. The seal
hashes (sha256) every file of a text type (serve.RUN_VERSIONS_HASH_TEXT: queues, catalogs, page indexes, pages, records) and
every other file up to serve.RUN_VERSIONS_HASH_SMALL_BYTES (16 MiB) not of a binary type - every file the seal's checks read
among them - and of the BINARIES (serve.RUN_VERSIONS_HASH_BINARY types - the panels - of any size, and any other file over
16 MiB) all (serve.RUN_VERSIONS_HASH_LARGE 'all') or their stable sample only ('sample'). The stable sample: the binaries whose
path's sha256 falls in the first serve.RUN_VERSIONS_SAMPLE_FRACTION (2%) of its range, at least serve.RUN_VERSIONS_SAMPLE_MIN
(20) of them (all when fewer). The rules are recorded in VERSION.json (hash_rules), so a later content test applies the seal's.
verify() without the seal's record (the content tests) walks the run and re-hashes every file that is not a binary, the
sample of the binaries and every file whose size or mtime differs from the manifest; a file whose content differs, or one
changed with no hash to compare, fails it, as does any file missing or added.

OWED CATCH-UP. A switch is followed by the catch-up of the journal lines written on the old version (sublst_build). If that
fails after the switch, owe_catchup() records it (R/CATCHUP_OWED.json: the old and new versions, the marks); until
catchup_done() removes it, begin(), place() and switch() of U refuse and retention keeps the old version. The sub-LST serve
retries the owed catch-up before it builds.

RETENTION (retire). Kept: the version the pointer names (and any version another link in U's directory names), the newest
serve.RUN_VERSIONS_KEEP sealed versions (an adopted version without its manifest cannot be switched back to: it is not one of
them, and it is kept until seal_adopted gives it one), every version with a live hold or a live builder (LEASES), the old
version of an owed catch-up. Everything else - older versions and abandoned unsealed
builds - is retired, ONLY through a dry run first: retire(U) writes the plan (R/retire_plans/RETIRE_PLAN_<UTC>.json: the versions and
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
import sys
import re
import shutil
import socket
import subprocess
import threading
import time

from exopipe.phase3.core.record import Refusal

FORMAT = 2                         # 2: MANIFEST.tsv carries each file's sha256 and VERSION.json its hash_rules (CONTENT)
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


def _lease_s():
    return float(_S().RUN_VERSIONS_LEASE_S)


def me(note=''):
    """The calling process as a holder or builder: pid, its start time, host, UTC, note, and its lease (lease_until, UTC
    seconds: serve.RUN_VERSIONS_LEASE_S ahead; renewed by renew() while the hold or build lasts)."""
    pid = os.getpid()
    return {'pid': pid, 'start': _proc_start(pid), 'host': socket.gethostname(), 'utc': _utc(), 'note': str(note)[:300],
            'lease_until': round(time.time() + _lease_s(), 3), 'lease_s': _lease_s()}


def alive(who):
    """True when the holder/builder ``who`` (a me() dict) may still run: on this host, its pid runs with the same start time;
    on another host, until twice its lease has passed (lease_until + lease_s); a record from another host without
    lease_until (from before leases, or unreadable) counts as alive - nothing removes it automatically (LEASES)."""
    if not who:
        return False
    if who.get('host') != socket.gethostname():
        if who.get('lease_until') is None:
            return True
        try:
            return time.time() < float(who['lease_until']) + float(who.get('lease_s') or _lease_s())
        except (TypeError, ValueError):
            return True
    st = _proc_start(who.get('pid', -1))
    return st is not None and (who.get('start') is None or st == who.get('start'))


# ---- lease renewal ----------------------------------------------------------------------------------------------------
_RENEW = {}                        # path of a holder/builder JSON -> stop test (callable -> True when the lease may end)
_RENEW_LOCK = threading.Lock()     # held while a renewed lease file is renamed into place (never during its write or fsync)
_RENEWER = [None]
_RENEW_WAKE = threading.Event()   # set by renew(): a new lease is renewed on the period of now, not of the last sleep
_INFLIGHT = {}                     # path -> Event set when the renewal of it in progress has ended (renew_stop waits for it)


def _renew_one(path, done):
    """One renewal of ``path``: read, written to a temporary file beside it and fsynced OUTSIDE _RENEW_LOCK (a slow file
    system never blocks renew_stop), renamed over ``path`` under the lock only while ``path`` is still renewed with this
    stop test (after renew_stop() the file is never rewritten). While it runs it is in _INFLIGHT, so renew_stop(path) waits
    until its temporary file is gone."""
    tmp = os.path.join(os.path.dirname(path), f'.{os.path.basename(path)}.lease.{os.getpid()}')
    with _RENEW_LOCK:
        if _RENEW.get(path) is not done:
            return                                    # stopped (or renewed anew) since the renewer's snapshot
        ev = _INFLIGHT[path] = threading.Event()
    try:
        if done() or not os.path.exists(path):
            with _RENEW_LOCK:
                if _RENEW.get(path) is done:
                    _RENEW.pop(path, None)
            return
        with open(path) as f:
            rec = json.load(f)
        rec['lease_until'] = round(time.time() + _lease_s(), 3)
        rec['lease_s'] = _lease_s()
        with open(tmp, 'w') as f:
            json.dump(rec, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        with _RENEW_LOCK:
            if _RENEW.get(path) is done and os.path.exists(path):
                os.replace(tmp, path)
    except (OSError, ValueError):
        with _RENEW_LOCK:
            if _RENEW.get(path) is done:
                _RENEW.pop(path, None)       # gone or unreadable: nothing left to renew
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        with _RENEW_LOCK:
            if _INFLIGHT.get(path) is ev:
                del _INFLIGHT[path]
        ev.set()


def _renew_loop():
    while True:
        _RENEW_WAKE.wait(max(_lease_s() / 4.0, 0.05))
        _RENEW_WAKE.clear()
        with _RENEW_LOCK:
            if not _RENEW:
                _RENEWER[0] = None
                return
            todo = list(_RENEW.items())
        for path, done in todo:
            _renew_one(path, done)


def renew(path, done=lambda: False):
    """Renew the lease of the holder/builder record ``path`` (written by this process) every quarter lease until ``done()``
    or the file is gone or renew_stop(path)."""
    with _RENEW_LOCK:
        _RENEW[path] = done
        if _RENEWER[0] is None or not _RENEWER[0].is_alive():
            _RENEWER[0] = threading.Thread(target=_renew_loop, name='versions-lease', daemon=True)
            _RENEWER[0].start()
    _RENEW_WAKE.set()


def renew_stop(path):
    """Stop renewing ``path``; returns after any renewal of it in progress has ended, its temporary file removed (a caller
    then removes or moves it, or its directory, safely). The wait is outside _RENEW_LOCK: other paths are never held up."""
    with _RENEW_LOCK:
        _RENEW.pop(path, None)
        ev = _INFLIGHT.get(path)
    if ev is not None and threading.current_thread() is not _RENEWER[0]:
        ev.wait()


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
        """[(hold file, holder)] of the version (a hold that cannot be read counts as alive: alive() of a record without a
        lease)."""
        d, out = self.path('holds'), []
        for f in sorted(n for n in os.listdir(d) if not n.startswith('.')) if os.path.isdir(d) else ():
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


def _open_lock(p):
    """two_port.open_lock (group-writable, close-on-exec), its permission failure a Refusal naming the file and owner."""
    from exopipe.phase3.lib.serve import two_port
    try:
        return two_port.open_lock(p)
    except two_port.LockPermissionError as e:
        raise Refusal(str(e)) from None


@contextlib.contextmanager
def _lock(R):
    os.makedirs(R, exist_ok=True)
    fd = _open_lock(os.path.join(R, '.lock'))
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
    refuse_owed(U, 'no new version begun')
    R, name = root_of(U), os.path.basename(os.path.normpath(U))
    vid = _new_vid_dir(R, time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    v = Version(R, vid, name)
    os.mkdir(v.run)
    _atomic_json(v.path('BUILDING.json'), dict(me(note), format=FORMAT, name=name, version=vid, pointer=os.path.abspath(U)))
    _renew_build(v)
    return v


def _renew_build(v):
    """Renew the lease of v's BUILDING.json while v is unsealed (LEASES)."""
    renew(v.path('BUILDING.json'), done=lambda: os.path.exists(v.path('VERSION.json')))


def place(U, B, note=''):
    """A new unsealed version of U whose run is the finished build directory ``B``, moved in by one rename (B must lie on the
    versions root's file system; nothing is copied). Returns the Version."""
    _guard(U, 'a new version of')
    if not os.path.isdir(B) or os.path.islink(B):
        raise Refusal(f'no build directory {B} to place as a version of {U}')
    refuse_owed(U, f'{B} not placed')
    R, name = root_of(U), os.path.basename(os.path.normpath(U))
    os.makedirs(R, exist_ok=True)
    if os.stat(B).st_dev != os.stat(R).st_dev:
        raise Refusal(f'{B} is not on the file system of the versions root {R}: a version is placed by a rename, never a copy')
    vid = _new_vid_dir(R, time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
    v = Version(R, vid, name)
    _atomic_json(v.path('BUILDING.json'), dict(me(note), format=FORMAT, name=name, version=vid, pointer=os.path.abspath(U),
                                               placed_from=os.path.abspath(B)))
    os.rename(B, v.run)
    _renew_build(v)
    return v


def ever_served(v):
    """True when R/POINTER_LOG.jsonl records a switch to ``v`` or a link names it now (a version readers may have seen). No
    log: no switch was ever made under R (the first switch writes it), so not served by a switch. A log that exists and
    cannot be read or parsed: Refusal (it cannot be decided)."""
    log = os.path.join(v.root, 'POINTER_LOG.jsonl')
    try:
        with open(log) as f:
            if any(json.loads(x).get('to') == v.vid for x in f if x.strip()):
                return True
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as e:
        raise Refusal(f'{log} cannot be read ({e}): whether version {v.vid} was served is not known') from None
    U = (v.builder() or {}).get('pointer') or os.path.join(os.path.dirname(v.root), v.name)
    return v.vid in _linked_vids(U, v.root)


def unplace(v, B, sealed=False):
    """Undo place(): the version's run moved back to the build directory ``B`` and the version removed (a switch that was
    refused leaves the build where the next attempt looks for it). A sealed version only with ``sealed`` (a seal that passed
    and a switch that then refused, or a swap killed between them) and only when it was never served (ever_served): its
    completion record is removed first, so it is never taken for complete again. Refusal otherwise, and for an existing
    ``B``."""
    _guard(v.run, 'a placed version of')
    if v.record() is not None:
        if not sealed:
            raise Refusal(f'version {v.vid} is sealed: it stays a version')
        if ever_served(v):
            raise Refusal(f'version {v.vid} was served (POINTER_LOG.jsonl or a link): it stays a version')
    if os.path.lexists(B):
        raise Refusal(f'{B} exists: version {v.vid} not moved back')
    renew_stop(v.path('BUILDING.json'))
    for f in ('VERSION.json', 'MANIFEST.tsv'):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(v.path(f))
    os.rename(v.run, B)
    shutil.rmtree(v.dir)


def placed_abandoned(U, B):
    """The newest version of U placed from ``B`` whose placing process is gone and that was never served - unsealed (a swap
    killed between its place and its seal) or sealed (killed between its seal and its switch, or a switch refused after the
    seal) - or None. unplace(v, B, sealed=True) moves it back."""
    for v in reversed(all_versions(U)):
        b = v.builder() or {}
        if b.get('placed_from') == os.path.abspath(B) and not alive(b) and os.path.isdir(v.run) \
                and (v.record() is None or ((v.record() or {}).get('state') == 'sealed' and not ever_served(v))):
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
    return ''.join(f'{k}\t{rel}\t{size}\t{mt}\t{tgt}\t{sha}\n' for k, rel, size, mt, tgt, sha in entries)


def _read_manifest(path):
    out = []
    with open(path, newline='\n') as f:
        for ln in f:
            k, rel, size, mt, tgt, sha = ln.rstrip('\n').split('\t')
            out.append((k, rel, int(size), int(mt), tgt, sha))
    return out


def hash_rules():
    """The CONTENT rules of a new seal (from the serve configuration), recorded in its VERSION.json."""
    S = _S()
    large = str(S.RUN_VERSIONS_HASH_LARGE)
    if large not in ('all', 'sample'):
        raise Refusal(f'serve.RUN_VERSIONS_HASH_LARGE = {large!r}: not all or sample')
    return {'small_bytes': int(S.RUN_VERSIONS_HASH_SMALL_BYTES), 'text': [str(x).lower() for x in S.RUN_VERSIONS_HASH_TEXT],
            'binary': [str(x).lower() for x in S.RUN_VERSIONS_HASH_BINARY], 'large': large,
            'sample_fraction': float(S.RUN_VERSIONS_SAMPLE_FRACTION), 'sample_min': int(S.RUN_VERSIONS_SAMPLE_MIN)}


def is_small(rel, size, rules):
    """A file every content test re-hashes (not a binary): one of the rules' text types, or up to its small_bytes and not of
    one of its binary types."""
    low = rel.lower()
    return low.endswith(tuple(rules['text'])) or (size <= rules['small_bytes'] and not low.endswith(tuple(rules['binary'])))


def large_sample(rels, rules):
    """The stable sample of the binaries ``rels`` (relative paths): those whose path's sha256 falls in the first
    sample_fraction of its range, topped up to sample_min by the smallest path hashes (all of them when fewer). It depends
    only on the paths, so the seal and every later content test choose the same files."""
    keyed = sorted((hashlib.sha256(r.encode()).hexdigest(), r) for r in rels)
    cut = int(rules['sample_fraction'] * 16 ** 16)
    n = max(sum(1 for k, _ in keyed if int(k[:16], 16) < cut), min(rules['sample_min'], len(keyed)))
    return {r for _, r in keyed[:n]}


def sha256_file(p):
    """sha256 of the file ``p``: ordinary 1 MiB reads, never a memory map."""
    d = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                return d.hexdigest()
            d.update(b)


def _manifest(run, mutable, rules, what):
    """(entries with their sha256, statistics): the walk of ``run`` and the sha256 of every file the rules hash (CONTENT).
    Refusal when a hashed file changed (size or mtime) or vanished while it was read."""
    t0 = time.time()
    entries = walk(run, mutable)
    walk_s = time.time() - t0
    files = [e for e in entries if e[0] == 'f']
    large = [e[1] for e in files if not is_small(e[1], e[2], rules)]
    todo = {e[1] for e in files if is_small(e[1], e[2], rules)}
    todo |= set(large) if rules['large'] == 'all' else large_sample(large, rules)
    out, n, nb = [], 0, 0
    t1 = time.time()
    for e in entries:
        sha = ''
        if e[0] == 'f' and e[1] in todo:
            p = os.path.join(run, e[1])
            try:
                sha = sha256_file(p)
                st = os.lstat(p)
            except FileNotFoundError:
                raise Refusal(f'{p} vanished while {what} read it - not sealed') from None
            if (st.st_size, st.st_mtime_ns) != (e[2], e[3]):
                raise Refusal(f'{p} changed while {what} read it (size or modification time) - not sealed')
            n, nb = n + 1, nb + e[2]
        out.append(tuple(e) + (sha,))
    return out, {'walk_s': round(walk_s, 1), 'hash_s': round(time.time() - t1, 1), 'n_hashed': n, 'hashed_bytes': nb,
                 'n_binaries': len(large), 'n_binaries_hashed': sum(1 for r in large if r in todo)}


def _write_manifest(v, entries):
    text = _manifest_text(entries).encode()
    mpath = v.path('MANIFEST.tsv')
    with open(mpath + '.part', 'wb') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(mpath + '.part', mpath)
    return hashlib.sha256(text).hexdigest()


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
    mutable, rules = list(_S().RUN_VERSIONS_MUTABLE), hash_rules()
    entries, stats = _manifest(v.run, mutable, rules, 'the seal')
    if not any(e[0] == 'f' for e in entries):
        raise Refusal(f'version {v.vid} of {v.name}: the run directory holds no file - not sealed')
    rec = {'format': FORMAT, 'name': v.name, 'version': v.vid, 'state': 'sealed', 'sealed_utc': _utc(),
           'sealed_by': me(note), 'builder': v.builder(), 'n_files': sum(1 for e in entries if e[0] == 'f'),
           'n_links': sum(1 for e in entries if e[0] == 'l'), 'n_bytes': sum(e[2] for e in entries),
           'manifest': 'MANIFEST.tsv', 'manifest_sha256': _write_manifest(v, entries), 'mutable': mutable, 'required': req,
           'checks': results, 'hash_rules': rules, **stats}
    renew_stop(v.path('BUILDING.json'))
    _atomic_json(v.path('VERSION.json'), rec)
    return rec


def verify(v, record=None, stats=None):
    """(ok, detail): ``v`` holds a completion record of this format and name, its manifest's sha256 is the recorded one, and
    the run on disk is its manifest IN CONTENT (CONTENT): every file and link present, nothing added (the record's mutable
    paths left out), every link's target, and the sha256 re-read of every file not a binary, of the stable sample of the
    binaries and of every file whose size or mtime differs from the manifest; a file changed without a hash to compare fails.
    One walk and those reads - unless ``record`` is the record seal() returned to the caller (its walk and reads made the
    manifest): then the record on disk must equal it and the manifest match it, and the run is not read again. ``stats``
    (a dict) receives the counts and the bytes read."""
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
    if record is not None:
        if record != rec:
            return False, f'version {v.vid}: the completion record on disk is not the one its seal returned'
        return True, (f'version {v.vid}: {rec.get("n_files")} files, {rec.get("n_bytes")} bytes, sealed {rec.get("sealed_utc")} '
                      f'by this process (its manifest walk of {rec.get("walk_s")} s and {rec.get("hashed_bytes")} bytes '
                      f'hashed; not read again)')
    rules = rec.get('hash_rules')
    if not isinstance(rules, dict):
        return False, f'version {v.vid}: its record holds no hash_rules (CONTENT) - its content cannot be tested'
    try:
        want = _read_manifest(mpath)
        have = walk(v.run, rec.get('mutable') or ())
    except (OSError, ValueError, Refusal) as e:
        return False, f'version {v.vid}: manifest or run unreadable ({e})'
    w, h = {e[1]: e for e in want}, {e[1]: e for e in have}
    missing = sorted(set(w) - set(h))
    added = sorted(set(h) - set(w))
    common = sorted(set(w) & set(h))
    links = [p for p in common if w[p][0] != h[p][0] or (w[p][0] == 'l' and w[p][4] != h[p][4])]
    files = [p for p in common if w[p][0] == 'f' and h[p][0] == 'f']
    meta = {p for p in files if w[p][2:4] != h[p][2:4]}
    small = {p for p in files if is_small(p, w[p][2], rules)}
    large = [p for p, e in w.items() if e[0] == 'f' and not is_small(p, e[2], rules)]
    sample = large_sample(large, rules) & set(files)
    todo = sorted(small | sample | meta)
    nohash = [p for p in todo if not w[p][5]]
    differ, nbytes, t0 = [], 0, time.time()
    for p in todo:
        if not w[p][5]:
            continue
        try:
            got = sha256_file(os.path.join(v.run, p))
        except OSError as e:
            return False, f'version {v.vid}: {p} unreadable ({e})'
        nbytes += h[p][2]
        if got != w[p][5]:
            differ.append(p)
    st = {'n_files': len(files), 'n_rehashed': len(todo) - len(nohash), 'bytes_read': nbytes, 'n_small': len(small),
          'n_binaries': len(large), 'n_sample': len(sample), 'n_meta_changed': len(meta), 'hash_s': round(time.time() - t0, 1)}
    if stats is not None:
        stats.update(st)
    if missing or added or links or differ or nohash:
        return False, (f'version {v.vid}: the run differs from its manifest - {len(missing)} missing (e.g. {missing[:3]}), '
                       f'{len(added)} added (e.g. {added[:3]}), {len(links)} links changed (e.g. {links[:3]}), '
                       f'{len(differ)} with other content (sha256; e.g. {differ[:3]}), {len(nohash)} changed in size or '
                       f'modification time with no sha256 to compare (e.g. {nohash[:3]})')
    return True, (f'version {v.vid}: {len(files)} files and the links as sealed {rec.get("sealed_utc")}; content re-read '
                  f'(sha256): {len(small)} files not binaries, {len(sample)} of {len(large)} binaries (the stable '
                  f'sample), {len(meta)} with a changed size or modification time (same content) - {st["n_rehashed"]} files, '
                  f'{nbytes} bytes')


# ---- the switch -------------------------------------------------------------------------------------------------------
def _link_target(U, v):
    """The pointer's link text: relative when the versions root lies beside U, else absolute."""
    d = os.path.realpath(os.path.dirname(os.path.abspath(U)))
    return os.path.relpath(v.run, d) if os.path.dirname(v.root) == d else v.run


def _has_file(d):
    """True when the directory tree ``d`` holds at least one file or link (stops at the first)."""
    for top, dirs, files in os.walk(d):
        if files or any(os.path.islink(os.path.join(top, n)) for n in dirs):
            return True
    return False


def _unadopt(v, real):
    """Undo _adopt: the adopted run moved back to ``real`` (U's directory, or the directory U links to) and the adopted
    version removed. Returns None, or the text of what could not be undone."""
    try:
        os.rename(v.run, real)
    except OSError as e:
        return (f'the legacy run could not be moved back: it is {v.run} (adopted version {v.vid}); move it back to {real} '
                f'by hand ({e})')
    with contextlib.suppress(OSError):
        os.unlink(v.path('VERSION.json'))
    with contextlib.suppress(OSError):
        os.rmdir(v.dir)
    return None


def _adopt(U, R):
    """Move the legacy run U (a directory, or a link to one) into R as an adopted version; returns (Version, the link U was or
    None, the directory it was moved from). The caller holds R's lock and writes the pointer right after (os.replace over U:
    over the link U was, now dangling, or where the directory was); a failure before U points at the new version calls
    _unadopt. Refusal, nothing moved: not a directory, another file system, an empty run, no free version name. The record
    (state 'adopted') is written only after the rename; a failed rename removes the empty version directory and refuses, a
    failed record moves the run back and refuses."""
    name = os.path.basename(os.path.normpath(U))
    was_link = os.readlink(U) if os.path.islink(U) else None
    real = os.path.realpath(U)
    if not os.path.isdir(real):
        raise Refusal(f'{U} is neither a pointer nor a run directory - nothing switched')
    if os.stat(real).st_dev != os.stat(R).st_dev:
        raise Refusal(f'{real} is not on the file system of the versions root {R} - nothing switched')
    if not _has_file(real):
        raise Refusal(f'{real} holds no file: an empty run is not adopted as a version - nothing switched')
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(os.stat(real).st_mtime))
    vid = None
    for n in range(1, 1000):
        cand = f'{stamp}-adopted' if n == 1 else f'{stamp}-{n}-adopted'
        try:
            os.mkdir(os.path.join(R, cand))
            vid = cand
            break
        except FileExistsError:
            continue
    if vid is None:
        raise Refusal(f'no free version name under {R} for {stamp}-adopted (999 taken) - nothing switched')
    v = Version(R, vid, name)
    try:
        os.rename(real, v.run)                         # U itself is left for the caller's os.replace
    except OSError as e:
        with contextlib.suppress(OSError):
            os.rmdir(v.dir)
        raise Refusal(f'{real} could not be moved into {v.run} ({e}) - nothing switched, {U} unchanged') from None
    try:
        _atomic_json(v.path('VERSION.json'), {'format': FORMAT, 'name': name, 'version': vid, 'state': 'adopted',
                                              'adopted_utc': _utc(), 'adopted_by': me(), 'adopted_from': real,
                                              'pointer_was_link': was_link, 'manifest': None})
    except BaseException as e:
        undo = _unadopt(v, real)
        if not isinstance(e, Exception):               # an interrupt stays an interrupt; what was not undone is printed
            if undo:
                print(f'versions: {undo}', file=sys.stderr, flush=True)
            raise
        raise Refusal(f'the adoption record of {v.vid} could not be written ({e}) - '
                      + (undo or f'the run moved back to {real}, nothing switched')) from None
    return v, was_link, real


def seal_adopted(v, note=''):
    """Give the adopted version ``v`` its manifest: a walk of its run and the sha256 of its files (CONTENT), MANIFEST.tsv,
    then its record rewritten (atomically) as state 'sealed' with the adoption kept (adopted_utc, adopted_by, adopted_from,
    pointer_was_link). Afterwards verify() and switch() treat it as any sealed version. The adoption required a run holding a
    file; one holding only review state (the mutable paths) gets an empty manifest. Refusal for a version not in state
    'adopted'. Returns the record."""
    _guard(v.run, 'the seal of')
    old = v.record() or {}
    if old.get('state') != 'adopted' or old.get('version') != v.vid:
        raise Refusal(f'version {v.vid}: state {old.get("state")!r}, not adopted - not sealed as adopted')
    mutable, rules = list(_S().RUN_VERSIONS_MUTABLE), hash_rules()
    entries, stats = _manifest(v.run, mutable, rules, 'the seal of the adopted run')
    rec = {k: old[k] for k in ('adopted_utc', 'adopted_by', 'adopted_from', 'pointer_was_link') if k in old}
    rec.update({'format': FORMAT, 'name': v.name, 'version': v.vid, 'state': 'sealed', 'adopted': True, 'sealed_utc': _utc(),
                'sealed_by': me(note), 'builder': None, 'n_files': sum(1 for e in entries if e[0] == 'f'),
                'n_links': sum(1 for e in entries if e[0] == 'l'), 'n_bytes': sum(e[2] for e in entries),
                'manifest': 'MANIFEST.tsv', 'manifest_sha256': _write_manifest(v, entries), 'mutable': mutable,
                'required': [], 'checks': [], 'hash_rules': rules, **stats})
    _atomic_json(v.path('VERSION.json'), rec)
    return rec


def switch(U, v, note='', record=None, to_work=False):
    """Point U at version ``v`` atomically: verify(v, record) (a refusal switches nothing; ``record``: the record seal()
    returned to this caller, whose walk stands for the switch's), then a new link under a temporary name renamed over U. The
    first switch of a legacy run adopts it first (see ADOPTION; the caller then calls seal_adopted, outside the lock) - or,
    with ``to_work`` (CONVERSION, WORK), moves it to U's work directory instead (``v`` is then its clone). Returns
    {'previous': Version or None, 'current': Version, 'adopted': Version or None, 'verify': detail, 'converted': the work
    directory or None}. Refusal when ``v`` is not a version of U's root."""
    _guard(U, 'the pointer')
    name = os.path.basename(os.path.normpath(U))
    if v.name != name:
        raise Refusal(f'version {v.run} is not a version of {U}')
    with _lock(v.root):
        if os.path.realpath(root_of(U)) != os.path.realpath(v.root):
            raise Refusal(f'version {v.run} is not under the versions root of {U} ({root_of(U)})')
        refuse_owed(U, 'not switched')
        ok, detail = verify(v, record)
        if not ok:
            raise Refusal(f'not switched: {detail} - {U} still names {os.path.realpath(U) if os.path.lexists(U) else "nothing"}')
        prev, adopted = served(U), None
        if prev is not None and prev.vid == v.vid:
            raise Refusal(f'{U} already names version {v.vid}')
        converted = None
        if prev is None and os.path.lexists(U):
            if to_work:                                # CONVERSION: the legacy run becomes U's work directory
                real = converted = _to_work(U, v.root)
            else:
                adopted, _, real = _adopt(U, v.root)
                prev = adopted
        tmp = os.path.join(os.path.dirname(os.path.abspath(U)), f'.{name}.pointer.{os.getpid()}')
        try:
            if os.path.lexists(tmp):
                os.unlink(tmp)
            os.symlink(_link_target(U, v), tmp)
            os.replace(tmp, U)
        except BaseException as e:                     # U does not point at v: an adopted run goes back where it was
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            undo = _unadopt(adopted, real) if adopted is not None else (_unwork(U, v.root, real) if converted else None)
            if not isinstance(e, Exception):           # an interrupt stays an interrupt; what was not undone is printed
                if undo:
                    print(f'versions: {undo}', file=sys.stderr, flush=True)
                raise
            if undo is not None or not isinstance(e, Refusal):
                raise Refusal(f'not switched: the pointer could not be written ({e})'
                              + (f' - AND {undo}' if undo else f' - {U} unchanged')) from e
            raise
        _append_jsonl(os.path.join(v.root, 'POINTER_LOG.jsonl'),
                      {'utc': _utc(), 'pointer': os.path.abspath(U), 'from': prev.vid if prev else None, 'to': v.vid,
                       'adopted': adopted.vid if adopted else None, 'by': me(note),
                       **({'converted_from': converted, 'work': work_path(U)} if converted else {})})
    return {'previous': prev, 'current': v, 'adopted': adopted, 'verify': detail,
            'converted': work_path(U) if converted else None}


def publish(U, v, required=(), checks=(), note=''):
    """seal(v) then switch(U, v, record): the version complete, then served (one walk: the seal's); an adopted legacy run
    then gets its manifest (seal_adopted; the switch is done by then, so a refusal of that seal is returned as
    'adopted_error' - the version stays 'adopted', kept by retention - not raised). Returns switch()'s dict with the record
    (and 'adopted_record' or 'adopted_error') added."""
    rec = seal(v, required=required, checks=checks, note=note)
    out = switch(U, v, note=note, record=rec)
    out['record'] = rec
    if out['adopted'] is not None:
        try:
            out['adopted_record'] = seal_adopted(out['adopted'], note=note)
        except (Refusal, OSError) as e:
            out['adopted_record'], out['adopted_error'] = None, str(e)
    return out


# ---- the work directory and its writers (WORK) ------------------------------------------------------------------------
WORK = 'work'                      # R/work/<name>: the work directory of a run whose steps write it over many invocations
_WRITER = {}                       # versions root -> descriptor of its writer lock held by this process until it ends


def work_path(U):
    """R/<WORK>/<name>, the work directory of U (whether or not it exists)."""
    return os.path.join(root_of(U), WORK, os.path.basename(os.path.normpath(U)))


def work_root(path):
    """The versions root R when ``path`` is the work directory R/<WORK>/<name> of a run (R = <dir>/<name><suffix>, an existing
    directory), else None."""
    p = os.path.normpath(os.path.abspath(path))
    d, name = os.path.split(p)
    R = os.path.dirname(d)
    if os.path.basename(d) == WORK and os.path.basename(R) == name + _suffix() and os.path.isdir(R):
        return R
    return None


def _writer_lock_file(R):
    d = os.path.join(R, 'locks')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'writer.lock')


def _take_writer(R, what):
    """flock the writer lock of R (waits; says so when it has to), returns the descriptor."""
    fd = _open_lock(_writer_lock_file(R))
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f'versions: {what} waits for the writer lock {_writer_lock_file(R)} (a writer step or a snapshot holds it)',
              file=sys.stderr, flush=True)
        fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def hold_writer(R):
    """THE WRITER LOCK of versions root R (R/locks/writer.lock, flock exclusive), taken once and kept until this process ends:
    a writer step holds it while it writes the work directory, so a snapshot never clones a directory a step is writing.
    Close-on-exec (two_port.open_lock): the step's child processes write while the step waits for them; none inherits it."""
    R = os.path.realpath(R)
    if R not in _WRITER:
        _WRITER[R] = _take_writer(R, f'pid {os.getpid()}')
    return _writer_lock_file(R)


@contextlib.contextmanager
def writer_lock(U):
    """U's writer lock for the block (a snapshot); a no-op when this process holds it already (hold_writer)."""
    R = os.path.realpath(root_of(U))
    if R in _WRITER:
        yield
        return
    fd = _take_writer(R, f'the snapshot of {U}')
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def writable(U, lock=True):
    """THE DIRECTORY A WRITER OF RUN U WRITES, and reads its own products from (WORK): when U is a pointer, its work directory
    (Refusal when it has none: a run its builder places - Run_SLSTB, the per-LST runs - is never written in place); else U
    itself (a legacy run, written in place as before; serve --tier deep --stages convert makes the deep run a versioned one).
    With ``lock`` the writer lock is taken and kept until this process ends (hold_writer): always for a work directory, and
    for a legacy run whose versions root exists (a conversion may be under way)."""
    if is_pointer(U):
        W = work_path(U)
        if not os.path.isdir(W):
            raise Refusal(f'{U} is a versioned run without a work directory {W}: its versions are placed whole by its builder '
                          f'and never written in place - nothing written')
        if lock:
            hold_writer(root_of(U))
        return W
    R = work_root(U)
    if R is None and lock and os.path.isdir(root_of(U)):
        R = root_of(U)
    if lock and R is not None:
        hold_writer(R)
    return os.path.abspath(U)


def reads(U):
    """The directory a step that writes run U reads U's own earlier products from: writable(U) without its lock."""
    return writable(U, lock=False)


def writer_out(out, lock=True):
    """A writer step's output directory (cli.main, before its record is made): a versioned run's pointer is written through
    its work directory (Refusal for a pointer without one), a work directory given as the output takes its writer lock
    (writable); any other directory is returned unchanged - with the lock taken when it is a legacy run whose versions root
    exists (a conversion may be under way). ``lock`` False (a dry run): the directory only."""
    if is_pointer(out) or work_root(out) or os.path.isdir(root_of(out)):
        return writable(out, lock=lock)
    return out


def _probe_reflink(d):
    """True when the file system of directory ``d`` makes copy-on-write clones (cp --reflink=always of a probe file)."""
    a = os.path.join(d, f'.reflink_probe_{os.getpid()}')
    b = a + '.clone'
    try:
        with open(a, 'wb') as f:
            f.write(b'\0' * 4096)
        r = subprocess.run(['cp', '--reflink=always', a, b], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0
    finally:
        for p in (a, b):
            with contextlib.suppress(OSError):
                os.unlink(p)


def clone(src, dst):
    """Copy the run directory ``src`` into the empty directory ``dst``: a COPY-ON-WRITE CLONE (cp -a --reflink=always: each
    file its own inode sharing src's data blocks until one of them is written, so a later write to either never reaches the
    other; no data is copied) where the file system makes them; elsewhere a full copy (cp -a) only of a run of at most
    serve.RUN_VERSIONS_COPY_MAX_BYTES, else Refusal (nothing copied: a large run is never duplicated in full). Modes,
    modification times and hard links within the run are kept. Returns {'method', 'seconds', 'bytes' (a full copy)}."""
    if not os.path.isdir(src) or os.listdir(dst):
        raise Refusal(f'clone {src} -> {dst}: no source directory, or the destination is not empty')
    t0, out = time.time(), {}
    if _probe_reflink(dst):
        cmd = ['cp', '-a', '--reflink=always', src.rstrip('/') + '/.', dst.rstrip('/') + '/']
        out['method'] = 'reflink'
    else:
        cap = int(_S().RUN_VERSIONS_COPY_MAX_BYTES)
        n = sum(e[2] for e in walk(src) if e[0] == 'f')
        if n > cap:
            raise Refusal(f'{dst} is on a file system without copy-on-write clones and {src} holds {n} bytes, over '
                          f'serve.RUN_VERSIONS_COPY_MAX_BYTES ({cap} bytes): it is not copied in full - nothing cloned')
        cmd = ['cp', '-a', src.rstrip('/') + '/.', dst.rstrip('/') + '/']
        out.update(method='copy', bytes=n)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if r.returncode != 0:
        raise Refusal(f'clone {src} -> {dst} failed ({" ".join(cmd[:3])}, rc {r.returncode}): {r.stdout[-400:]}')
    out['seconds'] = round(time.time() - t0, 1)
    return out


def same_content(src, run, exclude=(), stats=None):
    """(ok, detail): ``run`` holds exactly the files and links of ``src`` (relative paths matching a pattern of ``exclude``
    left out on both sides, and the mutable review state of RUN_VERSIONS_MUTABLE), every link with src's target, every file
    with src's size, and the same sha256 in both for every file the content rules re-hash (CONTENT): every file not a
    binary and the stable sample of the binaries. Reads both. ``stats`` receives the counts and the bytes read."""
    rules = hash_rules()
    skip = tuple(exclude) + tuple(_S().RUN_VERSIONS_MUTABLE)
    a, b = {e[1]: e for e in walk(src, skip)}, {e[1]: e for e in walk(run, skip)}
    missing, added = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    both = sorted(set(a) & set(b))
    kinds = [p for p in both if a[p][0] != b[p][0] or (a[p][0] == 'l' and a[p][4] != b[p][4])]
    files = [p for p in both if a[p][0] == b[p][0] == 'f']
    sizes = [p for p in files if a[p][2] != b[p][2]]
    small = {p for p in files if is_small(p, a[p][2], rules)}
    sample = large_sample([p for p in files if p not in small], rules)
    differ, nbytes = [], 0
    for p in sorted((small | sample) - set(sizes)):
        if sha256_file(os.path.join(src, p)) != sha256_file(os.path.join(run, p)):
            differ.append(p)
        nbytes += 2 * a[p][2]
    st = {'n_files': len(files), 'n_small': len(small), 'n_sample': len(sample), 'n_binaries': len(files) - len(small),
          'bytes_read': nbytes}
    if stats is not None:
        stats.update(st)
    if missing or added or kinds or sizes or differ:
        return False, (f'{run} is not {src}: {len(missing)} missing (e.g. {missing[:3]}), {len(added)} added (e.g. '
                       f'{added[:3]}), {len(kinds)} links or kinds differ (e.g. {kinds[:3]}), {len(sizes)} sizes differ (e.g. '
                       f'{sizes[:3]}), {len(differ)} with other content (sha256; e.g. {differ[:3]})')
    return True, (f'{run} holds the {len(files)} files and the links of {src}; sha256 equal for {len(small)} files not '
                  f'binaries and the stable sample of {len(sample)} of {len(files) - len(small)} binaries ({nbytes} bytes '
                  f'read)')


def snapshot(U, src, note=''):
    """A new unsealed version of U whose run is a clone of ``src`` (U's work directory, or the legacy run being converted),
    made under U's writer lock - no writer step writes ``src`` while it is cloned - and tested against it in content while the
    lock is still held (same_content). Returns (Version, {'clone': clone()'s dict, 'content': detail, 'stats': ...}). Any
    failure discards the version (nothing else changed)."""
    v = begin(U, note)
    try:
        with writer_lock(U):
            info = {'clone': clone(src, v.run)}
            st = {}
            ok, detail = same_content(src, v.run, stats=st)
        if not ok:
            raise Refusal(f'the snapshot of {src} is not its copy - {detail}')
        info.update(content=detail, stats=st)
    except BaseException:
        discard(v, note=f'{note}: the snapshot failed')
        raise
    return v, info


def discard(v, note=''):
    """Remove a version that was never served - a snapshot or placement whose build refused: unsealed, or sealed and never
    switched to (ever_served). It is renamed out of R's names first (R/.retiring_<vid>), then deleted, and logged in
    R/RETIRE_LOG.jsonl. Refusal for a version that was served."""
    _guard(v.run, 'a version of')
    if v.record() is not None and ever_served(v):
        raise Refusal(f'version {v.vid} was served (POINTER_LOG.jsonl or a link): it stays a version')
    renew_stop(v.path('BUILDING.json'))
    gone = os.path.join(v.root, f'.retiring_{v.vid}')
    os.rename(v.dir, gone)
    shutil.rmtree(gone)
    _append_jsonl(os.path.join(v.root, 'RETIRE_LOG.jsonl'), {'utc': _utc(), 'version': v.vid, 'discarded': True,
                                                             'by': me(note)})


def _to_work(U, R):
    """CONVERSION (switch with to_work): move the legacy run U (a directory, or a link to one) to its work directory
    R/<WORK>/<name>; returns the directory it was moved from. The caller holds R's lock and writes the pointer right after;
    _unwork undoes it. Refusal, nothing moved: not a directory, another file system, a work directory already there."""
    real = os.path.realpath(U)
    W = os.path.join(R, WORK, os.path.basename(os.path.normpath(U)))
    if not os.path.isdir(real):
        raise Refusal(f'{U} is neither a pointer nor a run directory - nothing switched')
    if os.path.lexists(W):
        raise Refusal(f'{W} exists already: {U} is not converted twice - nothing switched')
    os.makedirs(os.path.dirname(W), exist_ok=True)
    if os.stat(real).st_dev != os.stat(os.path.dirname(W)).st_dev:
        raise Refusal(f'{real} is not on the file system of the versions root {R} - nothing switched')
    try:
        os.rename(real, W)
    except OSError as e:
        raise Refusal(f'{real} could not be moved to {W} ({e}) - nothing switched, {U} unchanged') from None
    return real


def _unwork(U, R, real):
    """Undo _to_work: the work directory moved back to ``real``. Returns None, or the text of what could not be undone."""
    W = os.path.join(R, WORK, os.path.basename(os.path.normpath(U)))
    try:
        os.rename(W, real)
    except OSError as e:
        return f'the legacy run could not be moved back: it is {W}; move it back to {real} by hand ({e})'
    return None


def unshare_mutable(run):
    """Give every file of ``run`` matching serve.RUN_VERSIONS_MUTABLE (the review state appended in place while a version is
    served) its own inode when it is hard-linked (a placement's --link-dest): an append to it then never reaches another
    version. Returns the relative paths copied."""
    pats, out = tuple(_S().RUN_VERSIONS_MUTABLE), []
    for dp, _dn, fns in os.walk(run):
        for fn in fns:
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, run)
            if not any(fnmatch.fnmatchcase(rel, m) for m in pats) or os.path.islink(p) or os.lstat(p).st_nlink < 2:
                continue
            tmp = f'{p}.unshare.{os.getpid()}'
            shutil.copy2(p, tmp)
            os.replace(tmp, p)
            out.append(rel)
    return sorted(out)


def stamp(U):
    """What U is now, for a reader that reads U by its name over a while (a script given the path) and must know afterwards
    that it read one version: (link target or 'dir', inode, change time in ns) of U itself (lstat); every switch writes a new
    link (a new inode), so any switch in between - also away and back - changes it. None when U does not exist. A directory
    (a legacy run): ('dir', inode, None) - its change time moves whenever an entry is made in it (a log), which is no switch;
    its adoption (a link in its place) changes the stamp."""
    try:
        st = os.lstat(U)
    except FileNotFoundError:
        return None
    if not os.path.islink(U):
        return ('dir', st.st_ino, None)
    return (os.readlink(U), st.st_ino, st.st_ctime_ns)


@contextlib.contextmanager
def pinned(Us, note=''):
    """Hold the version each run of ``Us`` names (hold) for the block: yields {U: (run directory held, stamp)}. The stamp is
    taken after the hold and must name the held run (else the hold is taken again; Refusal after three tries: U is being
    switched without pause). With moved() the caller learns afterwards whether any U was switched in the block - a reader
    that read U by its name then read more than one version."""
    with contextlib.ExitStack() as st:
        out = {}
        for U in Us:
            for _ in range(3):
                held = contextlib.ExitStack()
                d = held.enter_context(hold(U, note=note))
                s = stamp(U)
                if os.path.realpath(U) == os.path.realpath(d):
                    st.push(held.pop_all())
                    out[U] = (d, s)
                    break
                held.close()
            else:
                raise Refusal(f'{U} was switched three times while its version was held - not pinned')
        yield out


def moved(pins):
    """[U] of ``pins`` (pinned()'s dict) whose pointer changed since it was pinned (stamp)."""
    return [U for U, (_d, s) in pins.items() if stamp(U) != s]


def journal_dirs(*runs):
    """The judgement journal directories (relative to the run) of ``runs``: every directory of the journal patterns of
    serve.RUN_VERSIONS_MUTABLE (those naming a judgements.jsonl) that holds one in any of them, sorted."""
    pats = [m for m in _S().RUN_VERSIONS_MUTABLE if m.endswith('/judgements.jsonl')]
    out = set()
    for r in runs:
        if not r or not os.path.isdir(r):
            continue
        for d in os.listdir(r):
            if os.path.isfile(os.path.join(r, d, 'judgements.jsonl')) and \
                    any(fnmatch.fnmatchcase(f'{d}/judgements.jsonl', m) for m in pats):
                out.add(d)
    return sorted(out)


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
    renew(path)                                        # LEASES: a holder on another node stays alive while it holds
    try:
        yield v.run
    finally:
        renew_stop(path)
        with contextlib.suppress(OSError):
            os.unlink(path)


# ---- the judgement journals across a switch ---------------------------------------------------------------------------
@contextlib.contextmanager
def journal_locks(U, journals):
    """Hold the journal lock of every journal directory ``journals`` (relative to the run) of U's versions: R/locks/
    journal_<J>.lock, the file the two-port server's POST to a labels directory of this run locks (two_port.journal_lock_path;
    the same file for every version, so it excludes across a switch). A switch and the catch-up of the lines written on the
    old version run inside it: no label write is in flight across the switch, and the old version's journals cannot grow
    after the catch-up read them."""
    from exopipe.phase3.lib.serve import two_port
    R = root_of(U)
    fds = []
    try:
        for J in sorted(set(journals)):
            fd = _open_lock(two_port.journal_lock_file(R, J))      # group-writable, close-on-exec
            fds.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        for fd in reversed(fds):
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


# ---- an owed catch-up -------------------------------------------------------------------------------------------------
OWED_FILE = 'CATCHUP_OWED.json'


def owed_catchup(U):
    """The owed catch-up record of U (R/CATCHUP_OWED.json) or None. A record that exists and cannot be read: Refusal."""
    p = os.path.join(root_of(U), OWED_FILE)
    try:
        with open(p) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        raise Refusal(f'{p} exists and cannot be read ({e}): an owed journal catch-up of {U} - resolve it first') from None


def refuse_owed(U, what):
    """Refusal when U owes a journal catch-up (OWED CATCH-UP)."""
    rec = owed_catchup(U)
    if rec is not None:
        raise Refusal(f'{what}: {U} owes the catch-up of the journal lines of version {rec.get("from_version")} into '
                      f'{rec.get("to_version")} (failed {rec.get("utc")}: {str(rec.get("error"))[:200]}; '
                      f'{os.path.join(root_of(U), OWED_FILE)}) - the next serve of the run (--tier sublst or deep) retries it first')


def owe_catchup(U, rec):
    """Record an owed catch-up (rec: from_version, from_run, to_version, the marks, the error). Returns its path."""
    p = os.path.join(root_of(U), OWED_FILE)
    _atomic_json(p, dict(rec, format=FORMAT, utc=_utc(), by=me(rec.get('note', ''))))
    return p


def catchup_done(U, rec):
    """Remove the owed catch-up record ``rec`` (it must still be the one on disk)."""
    p = os.path.join(root_of(U), OWED_FILE)
    if owed_catchup(U) != rec:
        raise Refusal(f'{p} changed since it was read - not removed')
    os.unlink(p)
    _append_jsonl(os.path.join(root_of(U), 'POINTER_LOG.jsonl'),
                  {'utc': _utc(), 'pointer': os.path.abspath(U), 'catchup_done': rec.get('from_version'),
                   'to': rec.get('to_version'), 'by': me()})


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
    owed = (owed_catchup(U) or {}).get('from_version')
    done = [v for v in vs if (v.record() or {}).get('state') == 'sealed']     # switchable: a manifest to verify against
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
        elif v.vid == owed:
            out.append((v, 'keep', f'its journal lines are owed to the served version ({OWED_FILE})'))
        elif live_holds:
            out.append((v, 'keep', f'held by {len(live_holds)} reader(s), e.g. pid {live_holds[0].get("pid")} on '
                                   f'{live_holds[0].get("host")}'))
        elif state == 'unsealed' and alive(b):
            out.append((v, 'keep', f'being built by pid {b.get("pid")} on {b.get("host")}'))
        elif v.vid in newest:
            out.append((v, 'keep', f'one of the newest {keep} complete versions'))
        elif state == 'adopted':                     # the legacy run: never deleted before it has a manifest
            out.append((v, 'keep', 'adopted without a manifest - not one of the newest complete versions (it cannot be '
                                   'switched back to); seal_adopted gives it one, then it is retained as any version'))
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
