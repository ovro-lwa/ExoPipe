"""The coalescing builder (plan/CUTOVER_REMAINING_20261124.md row 8; plan/CUTOVER_READINESS.md:333-344 (c): "one coalescing
builder - a placement during a build schedules exactly one follow-up build"; plan/ORCHESTRATION_DESIGN_20261117.md section 5.3:
"while a build runs, further placements set one 'rebuild' flag, and ... exactly one follow-up build when the running one ends").

A SLOT is a directory (one per built product: for a served run, versions.root_of(U)/serve.COALESCE_SLOT_DIR):
  REQUESTED      the one request flag. A request writes it (a temporary file renamed over it), so any number of requests while
                 a build runs leave one flag: exactly one follow-up.
  lock           flock held by the builder for as long as it builds (close-on-exec: a build's child processes never hold it -
                 production's 2026-10-01 lesson, review servers that inherited the build lock's descriptor held it for days).
  BUILDER.json   who builds now (pid, process start, host, UTC).
  journal.jsonl  one line per request and per build (UTC, who, exit code, seconds).

submit(build): request, then drain. drain(build): take the lock without waiting - when another builder holds it, return
'coalesced' (that builder runs the build this request asked for: it takes the flag after its build ends; only a builder ever
holds the lock - status and builder() read BUILDER.json and never take it); else, while the flag exists, remove it and run
build() once. After the lock is released the flag is looked at once more: a request made between the
last look and the release would otherwise wait for the next request, so the builder takes the lock again and builds.
A build that fails (a non-zero exit, an exception) is recorded; a request made during it still gets its one follow-up (it asked
for a build that starts after it). A builder that dies releases the lock with its process; the request it took is lost with
the build, and the build's own record shows the failure (the per-hour driver resubmits from the stage records).

WIRED IN: serve --tier sublst coalesces by itself (sublst_build.run submits its build and start stages to the slot of U; a
serve that finds a build running exits 3 with a COALESCED no-input unit), and the sub-LST placement (pages --tier sublst,
stage place) requests a build of the cumulative run beside the runs it placed. So the per-hour chain (pages, then serve,
per item) needs no wrapper; another builder of a served product wraps its command with the CLI 'run' below.

CLI (for the per-hour driver; plan/ORCHESTRATION_DESIGN_20261117.md section 5.3):
  python -m exopipe.phase3.lib.serve.coalesce run --slot DIR [--who TEXT] -- CMD [ARG ...]   request, then build or coalesce
  python -m exopipe.phase3.lib.serve.coalesce request --slot DIR [--who TEXT]                 a request only
  python -m exopipe.phase3.lib.serve.coalesce status --slot DIR
'run' prints one JSON line and exits 0 when every build it ran exited 0, the first non-zero build exit code otherwise, and 3
(nothing done here) when the request was coalesced into a running build or already served by another builder's build.
"""
import argparse
import contextlib
import fcntl
import json
import os
import socket
import subprocess
import sys
import time

EXIT_COALESCED = 3


def _utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _who(note=''):
    from exopipe.phase3.lib.serve.versions import me
    return me(note)


class Slot:
    def __init__(self, d):
        self.d = os.path.abspath(d)
        self.flag = os.path.join(self.d, 'REQUESTED')
        self.lockfile = os.path.join(self.d, 'lock')
        self.builder_file = os.path.join(self.d, 'BUILDER.json')
        self.journal = os.path.join(self.d, 'journal.jsonl')

    def _log(self, **kv):
        os.makedirs(self.d, exist_ok=True)
        with open(self.journal, 'a') as f:
            f.write(json.dumps(dict(utc=_utc(), **kv), sort_keys=True) + '\n')

    def request(self, note=''):
        """Place a request: the flag written (a temporary file renamed over it - one flag however many requests)."""
        os.makedirs(self.d, exist_ok=True)
        who = _who(note)
        tmp = f'{self.flag}.{who["host"]}.{who["pid"]}.{time.time_ns()}'
        with open(tmp, 'w') as f:
            json.dump(who, f)
        os.replace(tmp, self.flag)
        self._log(event='request', who=who)

    def pending(self):
        return os.path.exists(self.flag)

    def _take(self):
        """Remove the flag; True when there was one (a build is owed)."""
        try:
            os.unlink(self.flag)
            return True
        except FileNotFoundError:
            return False

    def _try_lock(self):
        os.makedirs(self.d, exist_ok=True)
        fd = os.open(self.lockfile, os.O_RDWR | os.O_CREAT, 0o664)        # close-on-exec (Python's default)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            os.close(fd)
            return None

    def builder(self):
        """The builder BUILDER.json records when it may still run (versions.alive: its pid on this host, its lease on another),
        else None. Never takes the lock: a probe that held it, however briefly, made a concurrent drain() fail its try-lock and
        leave its request to no builder."""
        from exopipe.phase3.lib.serve.versions import alive
        try:
            with open(self.builder_file) as f:
                who = json.load(f)
        except (OSError, ValueError):
            return None
        return who if alive(who) else None

    def drain(self, build, note=''):
        """Run build() once per taken request while this process holds the lock (see the module text). Returns
        {'status': 'built' | 'coalesced' | 'covered', 'builds': [{'rc', 'seconds', 'error'}], 'builder': holder or None}:
        'coalesced' - another builder held the lock (it runs this request's build); 'covered' - no request was left (another
        builder's build, begun after the request, served it)."""
        builds = []
        while True:
            fd = self._try_lock()
            if fd is None:
                if builds:                               # this builder built; a newer builder took the slot meanwhile
                    return {'status': 'built', 'builds': builds, 'builder': None}
                return {'status': 'coalesced', 'builds': builds, 'builder': self.builder()}
            from exopipe.phase3.lib.serve import versions
            try:
                with open(self.builder_file + '.part', 'w') as f:
                    json.dump(_who(note), f)
                os.replace(self.builder_file + '.part', self.builder_file)
                versions.renew(self.builder_file)        # its lease, for status readers on other nodes
                while self._take():
                    t0, rc, err = time.time(), None, None
                    self._log(event='build-start', n=len(builds) + 1, who=_who(note))
                    try:
                        rc = build()
                        rc = 0 if rc is None else int(rc)
                    except Exception as e:               # recorded; a request made meanwhile still gets its build
                        rc, err = 1, f'{type(e).__name__}: {e}'[:500]
                    builds.append({'rc': rc, 'seconds': round(time.time() - t0, 3), 'error': err})
                    self._log(event='build-end', n=len(builds), rc=rc, seconds=builds[-1]['seconds'], error=err)
            finally:
                versions.renew_stop(self.builder_file)
                with contextlib.suppress(OSError):
                    os.unlink(self.builder_file)
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            if not self.pending():
                return {'status': 'built' if builds else 'covered', 'builds': builds, 'builder': None}

    def submit(self, build, note=''):
        """request() then drain(build)."""
        self.request(note)
        return self.drain(build, note)


def exit_code(res):
    if res['status'] in ('coalesced', 'covered'):
        return EXIT_COALESCED
    return next((b['rc'] for b in res['builds'] if b['rc']), 0)


def slot_for(U):
    """The slot of the served run U: versions.root_of(U)/serve.COALESCE_SLOT_DIR."""
    from exopipe.phase3.config import CFG
    from exopipe.phase3.lib.serve import versions
    return Slot(os.path.join(versions.root_of(U), CFG.serve.COALESCE_SLOT_DIR))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m exopipe.phase3.lib.serve.coalesce', description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('run', 'request', 'status'):
        p = sub.add_parser(name)
        p.add_argument('--slot', required=True, help='the slot directory')
        if name != 'status':
            p.add_argument('--who', default='', help='a note recorded with the request')
        if name == 'run':
            p.add_argument('command', nargs=argparse.REMAINDER, help='-- CMD [ARG ...]: the build')
    a = ap.parse_args(argv)
    slot = Slot(a.slot)
    if a.cmd == 'status':
        print(json.dumps({'slot': slot.d, 'pending': slot.pending(), 'builder': slot.builder(), 'host': socket.gethostname()}))
        return 0
    if a.cmd == 'request':
        slot.request(a.who)
        print(json.dumps({'slot': slot.d, 'requested': True, 'builder': slot.builder()}))
        return 0
    cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not cmd:
        ap.error('run: no build command after --')
    res = slot.submit(lambda: subprocess.call(cmd), a.who)
    print(json.dumps(dict(res, slot=slot.d, command=cmd)), flush=True)
    return exit_code(res)


if __name__ == '__main__':
    sys.exit(main())
