"""Space gates: the room a writer needs before it writes. A gate refuses (record.Refusal, exit 2); it never warns.

Why. The free-space checks of the step preflights read os.statvfs / shutil.disk_usage, which do not see Lustre USER quotas.
ExoPipe will run as user pipeline (uid 1002), whose Lustre quota it shares with the telescope recorders
(FINDING-20261117-EXOPIPE-AS-PIPELINE-SHARES-RECORDER-QUOTA: 2.507 PB of 2.55 PB used on 2026-11-17): a product written as
pipeline takes the recorders' room, and a full quota stops the recorders' writes. This module generalizes the photometry
campaign's space gate (gen6_space_ok.sh; lib/photometry/campaign.space, which now calls it with its own floors) to every
writer step (cli.main before the step runs, record.Run.unit before each unit) and to any writer outside v2
(`python -m exopipe.phase3.core.space --path P`, exit 0 room / 2 refused).

What a check reads, once per file system among the roots it is given (a step's --out and --work):
  Lustre (the root's mount is of type lustre in /proc/self/mountinfo; the mount point is the file system argument of lfs):
    - the run user's quota, `lfs quota -q -u <user> <mount>`: used and limit, in KiB and in files. The limit is the smaller
      non-zero of the soft and hard limits; 0 means none is set, and then no floor applies to that quantity;
    - the project quota of ExoPipe's trees, `lfs quota -q -p <id> <mount>`, when space.PROJECT_ID is set (none exists yet:
      a project quota needs root on the Lustre servers, decision G2 of plan/CUTOVER_REMAINING_20261124.md);
    - the file system's free space, the available KiB of the filesystem_summary line of `lfs df <mount>`.
  Any other file system: the available bytes of os.statvfs (f_bavail x f_frsize); there is no user quota on node-local disks.
Floors (bytes and files; config section `space`, or the caller's own section - the photometry campaign passes
photometry_campaign): QUOTA_MIN_FREE_BYTES and QUOTA_MIN_FREE_FILES (user and project quota headroom), FS_MIN_FREE_BYTES
(Lustre free space), LOCAL_MIN_FREE_BYTES (any other file system). A caller about to write a known amount - the per-hour
driver's space booking (plan/ORCHESTRATION_DESIGN_20261117.md 5.3, "books the chain's /fast and destination bytes in a
reservation table and checks them against the disk's real free space minus the bookings") - passes need_bytes / need_files
and the bytes others have booked (booked_bytes): the test is then headroom - booked - need >= floor, for the quota and for the
free space alike.
Outcome codes (the first words of the refusal): SPACE-QUOTA (user quota bytes or files below the floor), SPACE-PROJECT
(project quota), SPACE-FS (Lustre free space), SPACE-LOCAL (another file system's free space), SPACE-UNREADABLE (lfs or statvfs
could not be read: refused, never assumed fine).

Run user: the effective user of this process (pwd.getpwuid(os.geteuid())), never a fixed name; `user=` reads another user's
quota (Lustre lets only root read another user's quota: `lfs quota -u pipeline` as gh answers "Permission denied", so that
read is SPACE-UNREADABLE unless lfs runs as root).
Reads only: lfs quota, lfs df, statvfs and /proc/self/mountinfo. Nothing is written.
"""
import os
import pwd
import subprocess
import sys
import threading
import time

CODES = ('SPACE-QUOTA', 'SPACE-PROJECT', 'SPACE-FS', 'SPACE-LOCAL', 'SPACE-UNREADABLE')
MOUNTINFO = '/proc/self/mountinfo'


def run_user():
    """The user this process runs as (the quota that its products count against)."""
    return pwd.getpwuid(os.geteuid()).pw_name


def lfs_runner(cmd='lfs', timeout=120):
    """argv -> (exit code, stdout, stderr) of `<cmd> <argv>`; cmd is one program or a command line ('sudo -n lfs')."""
    prog = cmd.split() if isinstance(cmd, str) else list(cmd)

    def run(argv):
        p = subprocess.run(prog + list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                           timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    return run


def _floor(floors, key, default=0):
    v = getattr(floors, key, default)
    return default if v is None else v


def _runner(floors, lfs):
    return lfs or lfs_runner(_floor(floors, 'LFS', 'lfs'), _floor(floors, 'LFS_TIMEOUT_S', 120))


# ---------------------------------------------------------------------------------------------------------------------
# lfs output
# ---------------------------------------------------------------------------------------------------------------------

def parse_quota(text, fs):
    """{'kbytes', 'kbytes_limit', 'files', 'files_limit'} of `lfs quota -q -u <user> <fs>` (or -p <project>; KiB; a limit
    None: none set - the smaller non-zero of the soft and hard limits otherwise). None when the line is not there."""
    toks = text.split()
    if fs not in toks:
        return None
    v = toks[toks.index(fs) + 1:toks.index(fs) + 9]
    if len(v) < 8:
        return None
    try:
        kb, kq, kl, fl, fq, fll = (int(x.rstrip('*')) for x in (v[0], v[1], v[2], v[4], v[5], v[6]))
    except ValueError:
        return None
    lim = lambda a, b: min([x for x in (a, b) if x > 0] or [0]) or None     # noqa: E731
    return dict(kbytes=kb, kbytes_limit=lim(kq, kl), files=fl, files_limit=lim(fq, fll))


def parse_df(text):
    """The available KiB of the filesystem_summary line of `lfs df <fs>` (its fourth field); None without that line."""
    for ln in text.splitlines():
        f = ln.split()
        if f and f[0] == 'filesystem_summary:' and len(f) >= 4:
            try:
                return int(f[3])
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------------------------------------------------
# one Lustre file system
# ---------------------------------------------------------------------------------------------------------------------

def _tib(b):
    return '%.2f TiB' % (b / 2 ** 40)


def lustre(fs, floors, lfs=None, user=None, project=None, need_bytes=0, need_files=0, booked_bytes=0):
    """(ok, text, values, codes) of the Lustre file system mounted at ``fs`` against ``floors``: the user's quota headroom
    (bytes and files), the project's when ``project`` (default floors.PROJECT_ID) is set, and the free space of the
    filesystem_summary line. ``lfs``: argv -> (rc, stdout, stderr) (tests; default `floors.LFS`). With need and booked 0 and
    no project, the text and values are those of the photometry campaign's gate (CP-SPACE) before this module existed."""
    lfs = _runner(floors, lfs)
    user = user or run_user()
    project = _floor(floors, 'PROJECT_ID', None) if project is None else project
    try:
        rq, oq, eq = lfs(['quota', '-q', '-u', user, fs])
        rd, od, ed = lfs(['df', fs])
        rp, op, ep = lfs(['quota', '-q', '-p', str(project), fs]) if project is not None else (0, '', '')
    except (OSError, subprocess.SubprocessError) as e:
        return False, 'lfs quota / lfs df could not run (%s: %s)' % (type(e).__name__, e), {}, ['SPACE-UNREADABLE']
    q = parse_quota(oq, fs) if rq == 0 else None
    av = parse_df(od) if rd == 0 else None
    if q is None or av is None:
        return False, 'lfs quota -u %s %s / lfs df %s unreadable (exit %s, %s; %s, %s)' % (
            user, fs, fs, rq, (eq or oq).strip()[:200], rd, (ed or 'no filesystem_summary line').strip()[:200]), {}, \
            ['SPACE-UNREADABLE']
    pq = parse_quota(op, fs) if project is not None and rp == 0 else None
    if project is not None and pq is None:
        return False, 'lfs quota -p %s %s unreadable (exit %s, %s)' % (project, fs, rp, (ep or op).strip()[:200]), {}, \
            ['SPACE-UNREADABLE']
    vals = dict(user=user, filesystem=fs, quota_free_bytes=None if q['kbytes_limit'] is None else
                (q['kbytes_limit'] - q['kbytes']) * 1024, quota_free_files=None if q['files_limit'] is None else
                q['files_limit'] - q['files'], fs_free_bytes=av * 1024)
    take = int(need_bytes or 0) + int(booked_bytes or 0)
    if take or need_files:
        vals.update(need_bytes=int(need_bytes or 0), need_files=int(need_files or 0), booked_bytes=int(booked_bytes or 0))
    after = ' after %s booked + %s needed' % (_tib(booked_bytes or 0), _tib(need_bytes or 0)) if take else ''
    low, codes = [], []
    qmin_b, qmin_f, fs_min = (_floor(floors, k) for k in ('QUOTA_MIN_FREE_BYTES', 'QUOTA_MIN_FREE_FILES', 'FS_MIN_FREE_BYTES'))
    if vals['quota_free_bytes'] is not None and vals['quota_free_bytes'] - take < qmin_b:
        low.append('quota headroom of %s %s%s < %s' % (user, _tib(vals['quota_free_bytes'] - take), after, _tib(qmin_b)))
        codes.append('SPACE-QUOTA')
    if vals['quota_free_files'] is not None and vals['quota_free_files'] - int(need_files or 0) < qmin_f:
        low.append('quota headroom of %s %d files%s < %d files' % (
            user, vals['quota_free_files'] - int(need_files or 0), ' after %d needed' % need_files if need_files else '',
            qmin_f))
        codes.append('SPACE-QUOTA')
    if vals['fs_free_bytes'] - take < fs_min:
        low.append('%s free %.1f TB%s < %.1f TB' % (fs, (vals['fs_free_bytes'] - take) / 1e12, after, fs_min / 1e12))
        codes.append('SPACE-FS')
    if pq is not None:
        vals.update(project=project, project_free_bytes=None if pq['kbytes_limit'] is None else
                    (pq['kbytes_limit'] - pq['kbytes']) * 1024, project_free_files=None if pq['files_limit'] is None else
                    pq['files_limit'] - pq['files'])
        if vals['project_free_bytes'] is not None and vals['project_free_bytes'] - take < qmin_b:
            low.append('project %s quota headroom %s%s < %s' % (project, _tib(vals['project_free_bytes'] - take), after,
                                                               _tib(qmin_b)))
            codes.append('SPACE-PROJECT')
        if vals['project_free_files'] is not None and vals['project_free_files'] - int(need_files or 0) < qmin_f:
            low.append('project %s quota headroom %d files < %d files' % (
                project, vals['project_free_files'] - int(need_files or 0), qmin_f))
            codes.append('SPACE-PROJECT')
    text = '%s quota headroom %s, %s files; %s free %.1f TB' % (
        user, 'unlimited' if vals['quota_free_bytes'] is None else _tib(vals['quota_free_bytes']),
        'unlimited' if vals['quota_free_files'] is None else vals['quota_free_files'], fs, vals['fs_free_bytes'] / 1e12)
    if pq is not None:
        text += '; project %s headroom %s' % (project, 'unlimited' if vals['project_free_bytes'] is None
                                              else _tib(vals['project_free_bytes']))
    return not low, ('; '.join(low) + ' (%s)' % text) if low else text, vals, list(dict.fromkeys(codes))


# ---------------------------------------------------------------------------------------------------------------------
# any root
# ---------------------------------------------------------------------------------------------------------------------

def existing_ancestor(path):
    """``path`` or its nearest existing ancestor, symlinks resolved (a root not made yet is on its parent's file system)."""
    p = os.path.realpath(os.path.abspath(path))
    while not os.path.exists(p) and os.path.dirname(p) != p:
        p = os.path.dirname(p)
    return p


def _unescape(field):
    """A mountinfo path field: octal escapes (\\040 = space) decoded."""
    out, i = [], 0
    while i < len(field):
        if field[i] == '\\' and field[i + 1:i + 4].isdigit():
            out.append(chr(int(field[i + 1:i + 4], 8)))
            i += 4
        else:
            out.append(field[i])
            i += 1
    return ''.join(out)


def mount_of(path, mountinfo=MOUNTINFO):
    """(mount point, file-system type) of the file system that holds ``path``: the mountinfo entry with the longest mount
    point above it, the LAST such entry on a tie (a mount over another one: /lustre is an autofs entry and then the lustre
    mount). None when mountinfo cannot be read."""
    p = existing_ancestor(path)
    try:
        with open(mountinfo) as f:
            rows = f.read().splitlines()
    except OSError:
        return None
    best = None
    for ln in rows:
        head, sep, tail = ln.partition(' - ')
        fields = head.split()
        if not sep or len(fields) < 5 or not tail.split():
            continue
        mp = _unescape(fields[4])
        if p == mp or p.startswith(mp.rstrip('/') + '/') or mp == '/':
            if best is None or len(mp) >= len(best[0]):
                best = (mp, tail.split()[0])
    return best


def local(path, floors, need_bytes=0, booked_bytes=0, statvfs=os.statvfs):
    """(ok, text, values, codes) of a file system without user quotas: statvfs's available bytes - booked - need against
    floors.LOCAL_MIN_FREE_BYTES."""
    p = existing_ancestor(path)
    try:
        st = statvfs(p)
    except OSError as e:
        return False, 'statvfs %s unreadable (%s)' % (p, e), {}, ['SPACE-UNREADABLE']
    free = st.f_bavail * st.f_frsize
    floor = _floor(floors, 'LOCAL_MIN_FREE_BYTES')
    take = int(need_bytes or 0) + int(booked_bytes or 0)
    vals = dict(path=p, free_bytes=free)
    if take:
        vals.update(need_bytes=int(need_bytes or 0), booked_bytes=int(booked_bytes or 0))
    text = '%s free %.1f GB' % (p, free / 1e9)
    if free - take < floor:
        return False, '%s free %.1f GB%s < %.1f GB (%s)' % (
            p, (free - take) / 1e9, ' after %.1f GB booked + %.1f GB needed' % ((booked_bytes or 0) / 1e9,
                                                                               (need_bytes or 0) / 1e9) if take else '',
            floor / 1e9, text), vals, ['SPACE-LOCAL']
    return True, text, vals, []


def check(roots, floors=None, lfs=None, user=None, need_bytes=0, need_files=0, booked_bytes=0, mountinfo=MOUNTINFO,
          statvfs=os.statvfs):
    """(ok, text, per-file-system values, codes) for every root in ``roots`` (a path or a list), each file system read once:
    Lustre through lustre(), any other through local(). An unreadable mount table is SPACE-UNREADABLE."""
    if floors is None:
        from exopipe.phase3.config import CFG
        floors = CFG.space
    roots = [roots] if isinstance(roots, str) else [r for r in roots if r]
    seen, texts, vals, codes, ok = set(), [], [], [], True
    for r in roots:
        m = mount_of(r, mountinfo)
        if m is None:
            ok = False
            texts.append('%s: the mount table %s is unreadable' % (r, mountinfo))
            codes.append('SPACE-UNREADABLE')
            continue
        key = m if m[1] == 'lustre' else ('local', existing_ancestor(r))
        if key in seen:
            continue
        seen.add(key)
        if m[1] == 'lustre':
            o, t, v, c = lustre(m[0], floors, lfs=lfs, user=user, need_bytes=need_bytes, need_files=need_files,
                                booked_bytes=booked_bytes)
        else:
            o, t, v, c = local(r, floors, need_bytes=need_bytes, booked_bytes=booked_bytes, statvfs=statvfs)
            v = dict(v, fstype=m[1])
        ok = ok and o
        texts.append(t)
        vals.append(dict(v, root=r, mount=m[0]))
        codes += c
    return ok, '; '.join(texts), vals, list(dict.fromkeys(codes))


def refusal_text(text, codes):
    """The refusal message: the codes first (SPACE-QUOTA, SPACE-FS, ...), then what was read."""
    return '%s: %s' % ('/'.join(codes) or 'SPACE', text)


def require(roots, floors=None, **kw):
    """check() or raise record.Refusal (exit 2) naming the codes and what was read. Returns the values read."""
    ok, text, vals, codes = check(roots, floors, **kw)
    if not ok:
        from exopipe.phase3.core.record import Refusal
        raise Refusal(refusal_text(text, codes))
    return vals


class Gate:
    """The space gate of one step run (cli.main makes it for every writer step that is not a dry run): ``require()`` before
    the step runs; ``before_unit()`` before each unit (record.Run.unit) reads again when the last read is older than
    floors.RECHECK_S seconds. Once a read is low or unreadable the gate stays shut for the rest of the run (sticky, as the
    photometry campaign's CP-SPACE): every later unit refuses with the same reason."""

    def __init__(self, roots, floors=None, recheck_s=None, **kw):
        if floors is None:
            from exopipe.phase3.config import CFG
            floors = CFG.space
        self.roots = [r for r in ([roots] if isinstance(roots, str) else roots) if r]
        self.floors, self.kw = floors, kw
        self.recheck_s = _floor(floors, 'RECHECK_S', 300) if recheck_s is None else recheck_s
        self.stopped, self.last, self.values, self.text = None, None, [], ''
        self._lock = threading.Lock()

    def _read(self):
        ok, text, vals, codes = check(self.roots, self.floors, **self.kw)
        self.last, self.values, self.text = time.monotonic(), vals, text
        if not ok:
            self.stopped = refusal_text(text, codes)
        return ok

    def _raise(self):
        from exopipe.phase3.core.record import Refusal
        raise Refusal(self.stopped)

    def require(self):
        with self._lock:
            if self.stopped or not self._read():
                self._raise()
        return self.values

    def before_unit(self):
        with self._lock:
            if self.stopped:
                self._raise()
            if self.last is None or time.monotonic() - self.last >= self.recheck_s:
                if not self._read():
                    self._raise()


def main(argv=None):
    """`python -m exopipe.phase3.core.space --path P [--path P2] [--need-bytes N] ...`: the gate for a writer outside v2 (a
    shell script). Prints the verdict; exit 0 room, 2 refused (any code, an unreadable read included)."""
    import argparse
    ap = argparse.ArgumentParser(prog='python -m exopipe.phase3.core.space', description=main.__doc__)
    ap.add_argument('--path', action='append', required=True, help='a root the writer will write under (repeatable)')
    ap.add_argument('--user', help='read this user\'s quota instead of the run user\'s (reading another user\'s quota needs '
                                   'root: --lfs "sudo -n lfs")')
    ap.add_argument('--lfs', help='lfs command line (default config space.LFS)')
    ap.add_argument('--need-bytes', type=int, default=0, help='bytes about to be written')
    ap.add_argument('--need-files', type=int, default=0, help='files about to be written')
    ap.add_argument('--booked-bytes', type=int, default=0, help='bytes other writers have booked on the same roots')
    ap.add_argument('--floors', default='space', help='config section holding the floors (default space)')
    a = ap.parse_args(argv)
    from exopipe.phase3.config import CFG
    floors = getattr(CFG, a.floors)
    lfs = lfs_runner(a.lfs, _floor(floors, 'LFS_TIMEOUT_S', 120)) if a.lfs else None
    ok, text, vals, codes = check(a.path, floors, lfs=lfs, user=a.user, need_bytes=a.need_bytes, need_files=a.need_files,
                                  booked_bytes=a.booked_bytes)
    print(('ROOM: ' + text) if ok else ('REFUSED ' + refusal_text(text, codes)))
    for v in vals:
        print('  ' + ', '.join('%s=%s' % kv for kv in sorted(v.items())))
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
