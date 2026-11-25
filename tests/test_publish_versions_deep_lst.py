#!/usr/bin/env python3
"""Tests of the deep and per-LST publishes on the versioned pointer (lib/serve/versions.py WORK section, lib/serve/deep.py
THE VERSIONED DEEP RUN, lib/pages/tier_sublst.py place_version, the G30 pins of lib/serve/sublst_build.py): data-free, plain
checks, runnable with python, under a minute. Scratch under $TMPDIR.

    NUMPY_MADVISE_HUGEPAGE=0 /opt/devel/pipeline/envs/py38_orca_nkosogor/bin/python3 tests/test_publish_versions_deep_lst.py

1. The work directory: work_path / work_root; writable(U) of a legacy run is U, of a pointer its work directory, of a pointer
   without one a refusal - the writer lock (beside the versions root, the same file before and after a conversion) taken
   shared and kept; writer_out takes it for a run that may be converted (its lock file exists, or it lies in the deep
   served-run root) and leaves any other directory unchanged.
2. clone and same_content: the clone holds the run (copy-on-write where the file system makes them, else a full copy under
   serve.RUN_VERSIONS_COPY_MAX_BYTES, a larger run refused with nothing copied); a write to the clone never reaches the source;
   same_content passes a copy and fails a changed content of equal size, an added file, a changed link; the review state is
   left out.
3. snapshot, seal and discard: the snapshot waits for a writer step's shared lock (G31 after serve.WRITER_LOCK_WAIT_S, no
   version left) and reads the source once under it (SOURCE_MANIFEST.tsv); a file or link naming the source names the public
   path in the version (the source untouched); the seal refuses a version naming the forbidden path (G32), a clone whose file
   of the source's size and time has other content, and - strict - any change since the snapshot; a failed snapshot is
   discarded, and a discard that fails too is added to the cause; a writer step does not snapshot its own run; discard
   removes an unsealed version (logged), refuses a served one.
4. Conversion (switch with to_work): the legacy run is the work directory, U a pointer to its sealed clone; a failure to write
   the pointer moves it back; a work directory already there refuses with nothing moved.
5. stamp / pinned / moved: a switch during the block (also away and back) is seen, a log written into a legacy run is not;
   the pinned versions are held; journal_dirs; unshare_mutable.
6. The per-LST placement as a version: the first placement adopts the legacy run, the next one is a new version with the
   unchanged files hard-linked from the served one and the review state on its own inodes; the previous version is unchanged;
   a placement that is not the page run, or whose page run fails its own content tests, is discarded before the switch, U
   unchanged; the seal's page-run tests are the unit's output tests when the version is unchanged since, else run again.
7. G30: the cumulative build refuses when an LST run was switched while the merge read it, not when nothing was switched (a
   log written in a legacy LST run included); the read versions are held during the merge and recorded.
8. The deep serve on a versioned run: a writer step started before the conversion blocks it (G31 after
   serve.WRITER_LOCK_WAIT_S, nothing converted; with the full wait the conversion follows the writer); a writer step that
   waited for the conversion writes the work directory; convert (the journal lines written on the legacy run during the
   conversion caught up), a publish (snapshot of the work directory, a log naming it made to name U, the served journals
   copied in, the lines written on the old version during the publish caught up, a journal begun on it during the publish -
   a first label - in the served version, the old version unchanged), a journal begun on the old version across the switch
   merged with its lines first, G32 refuses a version naming the work directory, G31 in a publish refuses and frees the
   coalescing slot, a refused stage discards the snapshot, a switch of U by another serve during the publish refuses, a
   failed catch-up is owed and completed by the next serve, a partial stage selection refuses, convert twice is no-input, a
   parity convert refuses, retire runs for deep.
9. cli: an injected run with --out U (a versioned run) is guarded on U's work directory and its writer lock, never on the
   version U names; nothing is written into the version.
"""
import contextlib
import fcntl
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
TMP = os.path.join(tempfile.gettempdir(), 'exopipe3_tests_publish_versions_deep_lst')

from exopipe.phase3.config import CFG                      # noqa: E402
from exopipe.phase3.core.record import Refusal, Run        # noqa: E402
from exopipe.phase3.lib.serve import versions              # noqa: E402

FAILS, N = [], [0]


def check(what, ok, detail=''):
    N[0] += 1
    if not ok:
        FAILS.append(f'{what}: {detail}')
        print('FAIL ', what, detail)


def refuses(fn, *a, **k):
    try:
        fn(*a, **k)
    except Refusal as e:
        return str(e)
    return None


def write(path, text, mode='w'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode) as f:
        f.write(text)


def read(path):
    with open(path) as f:
        return f.read()


def release_writers():
    """Drop the writer locks this process holds (a writer step holds its lock until it ends; the tests run many steps)."""
    for fd in versions._WRITER.values():
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    versions._WRITER.clear()


WRITER = """
import sys, time
sys.path.insert(0, sys.argv[1])
from exopipe.phase3.lib.serve import versions
p = versions.writable(sys.argv[2])
print('LOCKED', p, flush=True)
time.sleep(float(sys.argv[3]))
"""


def writer_step(U, hold):
    """A writer step in its own process: versions.writable(U) (the writer lock shared, kept), then ``hold`` s; prints its path."""
    return subprocess.Popen([sys.executable, '-c', WRITER, REPO, U, str(hold)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True,
                            env=dict(os.environ, NUMPY_MADVISE_HUGEPAGE='0'))


def line_with(stream, text, timeout=120):
    """Read ``stream`` until a line holding ``text`` (returned), its end or ``timeout`` s (None)."""
    end = time.time() + timeout
    while time.time() < end:
        r, _, _ = select.select([stream], [], [], max(0.0, end - time.time()))
        if not r:
            return None
        ln = stream.readline()
        if not ln:
            return None
        if text in ln:
            return ln
    return None


@contextlib.contextmanager
def patch(*items):
    saved = [(o, k, getattr(o, k)) for o, k, _ in items]
    for o, k, v in items:
        setattr(o, k, v)
    try:
        yield
    finally:
        for o, k, v in reversed(saved):
            setattr(o, k, v)


def legacy_run(U, tag='A'):
    write(f'{U}/a.txt', tag * 100)
    write(f'{U}/sub/b.csv', 'x,y\n1,2\n')
    write(f'{U}/candidates/I/J1/panel.png', 'P' * 3000)
    os.symlink('/nonexistent/v110', f'{U}/v110_link')
    return U


# ---- 1. the work directory --------------------------------------------------------------------------------------------------
def test_work():
    d = f'{TMP}/work'
    U = legacy_run(f'{d}/Run_W')
    R = versions.root_of(U)
    check('work_path is R/work/<name>', versions.work_path(U) == os.path.join(R, 'work', 'Run_W'), versions.work_path(U))
    lockf = versions.writer_lock_file(R)
    check('writable of a legacy run: U itself, its writer lock taken shared and kept - a file beside the versions root, which '
          'is not made', versions.writable(U) == os.path.abspath(U) and os.path.realpath(R) in versions._WRITER
          and os.path.isfile(lockf) and os.path.dirname(lockf) == os.path.realpath(d) and not os.path.exists(R),
          (versions._WRITER, lockf))
    release_writers()
    check('writer_out of a plain directory: unchanged, no lock', versions.writer_out(f'{d}/out') == f'{d}/out'
          and versions.writer_lock_of(f'{d}/out') is None and not versions._WRITER, versions._WRITER)
    check('writer_out of a legacy run whose writer lock file exists: U, with its lock (a conversion waits for it)',
          versions.writer_out(U) == os.path.abspath(U) and os.path.realpath(R) in versions._WRITER
          and versions.writer_lock_of(U) == lockf, versions._WRITER)
    release_writers()
    key = CFG.serve.RUN_ROOT_KEY['deep']
    with patch((CFG.paths, key, d)):
        fresh = f'{d}/Run_NEW'
        os.makedirs(fresh)
        check('a run in the deep served-run root may be converted: its writer steps lock it before any lock file exists',
              versions.convertible(fresh) and versions.writer_lock_of(fresh) == versions.writer_lock_file(versions.root_of(fresh))
              and not versions.convertible(f'{d}/out/Run_X'), versions.writer_lock_of(fresh))
    v = versions.begin(U, note='test')
    write(os.path.join(v.run, 'x.txt'), 'x')
    versions.publish(U, v)                              # adopts the legacy run; U a pointer without a work directory
    why = refuses(versions.writable, U)
    check('writable of a pointer without a work directory refuses (a placed run is never written in place)',
          why and 'without a work directory' in why, why)
    why = refuses(versions.writer_out, U)
    check('writer_out of such a pointer refuses too', why and 'without a work directory' in why, why)
    W = versions.work_path(U)
    os.makedirs(W)
    check('work_root of the work directory is R; of U and of a version None', versions.work_root(W) == os.path.realpath(R)
          and versions.work_root(U) is None and versions.work_root(v.run) is None, versions.work_root(W))
    check('writable of a pointer with a work directory: it, the writer lock taken and kept',
          versions.writable(U) == W and os.path.realpath(R) in versions._WRITER, versions._WRITER)
    check('reads(U): the work directory, no lock needed', versions.reads(U) == W)
    check('writer_out of the pointer: its work directory', versions.writer_out(U) == W)
    release_writers()
    check('writer_out of the work directory given as the output: it, with its lock', versions.writer_out(W) == W
          and os.path.realpath(R) in versions._WRITER, versions._WRITER)
    release_writers()


# ---- 2. clone and same_content ----------------------------------------------------------------------------------------------
def test_clone_content():
    d = f'{TMP}/clone'
    src = legacy_run(f'{d}/src')
    write(f'{src}/judgements_V/judgements.jsonl', '{"a": 1}\n')
    os.link(f'{src}/a.txt', f'{src}/a_hardlink.txt')
    dst = f'{d}/dst'
    os.makedirs(dst)
    info = versions.clone(src, dst)
    ok, detail = versions.same_content(src, dst)
    check('the clone holds the run (same_content passes)', ok and info['method'] in ('reflink', 'copy'), (info, detail))
    check('the clone keeps hard links within the run, and its files are not the source\'s inodes',
          os.stat(f'{dst}/a.txt').st_ino == os.stat(f'{dst}/a_hardlink.txt').st_ino
          and os.stat(f'{dst}/a.txt').st_ino != os.stat(f'{src}/a.txt').st_ino, info)
    write(f'{dst}/a.txt', 'B' * 100)
    check('a write to the clone never reaches the source', read(f'{src}/a.txt') == 'A' * 100)
    ok, detail = versions.same_content(src, dst)
    check('same_content fails a changed content of equal size (sha256; the file and its hard link)',
          not ok and '2 with other content' in detail, detail)
    write(f'{dst}/a.txt', 'A' * 100)
    write(f'{dst}/judgements_V/judgements.jsonl', '{"b": 2}\n', mode='a')
    ok, detail = versions.same_content(src, dst)
    check('the review state (a journal) is left out of the comparison', ok, detail)
    write(f'{dst}/new.txt', 'n')
    ok, detail = versions.same_content(src, dst)
    check('an added file fails', not ok and '1 added' in detail, detail)
    os.unlink(f'{dst}/new.txt')
    os.unlink(f'{dst}/v110_link')
    os.symlink('/nonexistent/other', f'{dst}/v110_link')
    ok, detail = versions.same_content(src, dst)
    check('a link with another target fails', not ok and '1 links or kinds differ' in detail, detail)
    ok, _ = versions.same_content(src, dst, exclude=('v110_link',))
    check('an excluded path is left out on both sides', ok)
    # no copy-on-write: a full copy only up to the cap
    dst2, dst3 = f'{d}/dst2', f'{d}/dst3'
    os.makedirs(dst2)
    os.makedirs(dst3)
    with patch((versions, '_probe_reflink', lambda dd: False), (CFG.serve, 'RUN_VERSIONS_COPY_MAX_BYTES', 10)):
        why = refuses(versions.clone, src, dst2)
    check('no copy-on-write and the run over RUN_VERSIONS_COPY_MAX_BYTES: refused, nothing copied',
          why and 'RUN_VERSIONS_COPY_MAX_BYTES' in why and not os.listdir(dst2), why)
    with patch((versions, '_probe_reflink', lambda dd: False)):
        info = versions.clone(src, dst3)
    check('no copy-on-write, a small run: a full copy, its bytes recorded', info['method'] == 'copy' and info['bytes'] > 3000
          and versions.same_content(src, dst3)[0], info)
    why = refuses(versions.clone, src, dst3)
    check('clone into a directory that is not empty refuses', why and 'not empty' in why, why)


# ---- 3. snapshot, seal and discard -----------------------------------------------------------------------------------------
def test_snapshot():
    d = f'{TMP}/snap'
    U = legacy_run(f'{d}/Run_S')
    write(f'{U}/logs/build.log', f'DONE -> {U}\n')     # a writer step names its output directory in a log
    os.symlink(f'{U}/a.txt', f'{U}/abs_link')           # and in an absolute link
    R = versions.root_of(U)
    os.makedirs(R)
    lockf = versions.writer_lock_file(R)
    got, release = threading.Event(), threading.Event()

    def holder():                                      # a writer step of another process: its own open file, SHARED
        fd = os.open(lockf, os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_SH)
        got.set()
        release.wait(10)
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def hold(seconds):
        got.clear()
        release.clear()
        th = threading.Thread(target=holder)
        th.start()
        got.wait(5)
        if seconds is not None:
            threading.Timer(seconds, release.set).start()
        return th
    th = hold(1.0)
    t0 = time.time()
    v, info = versions.snapshot(U, U, note='test', public=(U, '/public/Run_S'))
    waited = time.time() - t0
    th.join(5)
    check('the snapshot waits for a writer step holding the writer lock shared', waited >= 0.9 and info['lock_wait_s'] >= 0.9,
          (waited, info.get('lock_wait_s')))
    src = versions._read_manifest(v.path(versions.SOURCE_MANIFEST))
    check('the snapshot: an unsealed version holding the run, the source read once under the lock (its manifest, '
          'SOURCE_MANIFEST.tsv)', v.record() is None and versions.same_content(U, v.run, exclude=('logs/*', 'abs_link'))[0]
          and len(src) == 6 and info['source']['n_hashed'] >= 3 and 'lock_s' in info, (v.record(), info, len(src)))
    st_u, st_v = os.stat(f'{U}/logs/build.log'), os.stat(f'{v.run}/logs/build.log')
    check('a file and a link naming the source name the public path in the version (mode and modification time kept), the '
          'source untouched', read(f'{v.run}/logs/build.log') == 'DONE -> /public/Run_S\n'
          and read(f'{U}/logs/build.log') == f'DONE -> {U}\n' and st_v.st_mtime_ns == st_u.st_mtime_ns
          and st_v.st_mode == st_u.st_mode and st_v.st_ino != st_u.st_ino
          and os.readlink(f'{v.run}/abs_link') == '/public/Run_S/a.txt' and os.readlink(f'{U}/abs_link') == f'{U}/a.txt'
          and info['public'] == ['abs_link', 'logs/build.log'], (info['public'], read(f'{v.run}/logs/build.log')))
    rec = versions.seal(v, note='test', forbid=[U])
    sc = rec.get('source_check') or {}
    check('its seal: nothing names the source (G32), the clone is its source (the file of the same size and time: same '
          'sha256)', sc.get('differ') == 0 and sc.get('n_compared', 0) >= 3 and sc.get('n_source') == 6, sc)
    v2, _ = versions.snapshot(U, U, note='test')
    why = refuses(versions.seal, v2, note='test', forbid=[U])
    check('G32: a version whose file or link names the forbidden path (the work directory) is not sealed',
          why and why.startswith('G32 work path') and 'logs/build.log' in why and 'abs_link' in why and v2.record() is None,
          why)
    versions.discard(v2, note='test')
    v3, _ = versions.snapshot(U, U, note='test', public=(U, '/public/Run_S'))
    st = os.stat(f'{v3.run}/a.txt')
    write(f'{v3.run}/a.txt', 'B' * 100)
    os.utime(f'{v3.run}/a.txt', ns=(st.st_atime_ns, st.st_mtime_ns))
    why = refuses(versions.seal, v3, note='test')
    check('a clone whose file has the source\'s size and time but other content is not its source: not sealed',
          why and 'the clone is not its source' in why and '1 of' in why and v3.record() is None, why)
    versions.discard(v3, note='test')
    v4, _ = versions.snapshot(U, U, note='test', public=(U, '/public/Run_S'))
    write(f'{v4.run}/new.txt', 'n')
    why = refuses(versions.seal, v4, note='test', source_strict=True)
    check('strict (a conversion): a file added since the snapshot refuses the seal', why and '1 added' in why
          and 'none may differ' in why, why)
    versions.discard(v4, note='test')
    log = [json.loads(l) for l in open(os.path.join(R, 'RETIRE_LOG.jsonl'))]
    check('discard removes the unsealed version and logs it', not os.path.exists(v4.dir) and log[-1]['version'] == v4.vid
          and log[-1]['discarded'], log[-1])
    n = len(versions.all_versions(U))

    def bad_clone(a, b):
        raise Refusal('stand-in clone failure')
    with patch((versions, 'clone', bad_clone)):
        why = refuses(versions.snapshot, U, U, note='test')
    check('a failed snapshot is refused and discarded', why == 'stand-in clone failure'
          and len(versions.all_versions(U)) == n, (why, versions.all_versions(U)))
    real_discard = versions.discard

    def bad_discard(v, note=''):
        raise OSError(5, 'stand-in discard failure')
    with patch((versions, 'clone', bad_clone), (versions, 'discard', bad_discard)):
        why = refuses(versions.snapshot, U, U, note='test')
    check('a failed snapshot whose discard fails too: the cause raised, the discard failure added to it',
          why and why.startswith('stand-in clone failure') and 'could not be discarded' in why
          and 'stand-in discard failure' in why, why)
    for x in versions.all_versions(U):
        if x.record() is None:
            real_discard(x, note='test')
    th = hold(None)
    with patch((CFG.serve, 'WRITER_LOCK_WAIT_S', 0.3)):
        why = refuses(versions.snapshot, U, U, note='test')
    release.set()
    th.join(5)
    check('G31: a writer step holding the lock past serve.WRITER_LOCK_WAIT_S refuses the snapshot, no version left',
          why and why.startswith('G31 writer lock') and len(versions.all_versions(U)) == n, (why, versions.all_versions(U)))
    versions.hold_writer(R)
    why = refuses(versions.snapshot, U, U, note='test')
    release_writers()
    check('a writer step does not snapshot its own run', why and 'writer step' in why
          and len(versions.all_versions(U)) == n, why)
    v = versions.begin(U, note='test')
    write(os.path.join(v.run, 'a.txt'), 'x')
    versions.publish(U, v)
    why = refuses(versions.discard, v, note='test')
    check('discard of a served version refuses', why and 'was served' in why and os.path.isdir(v.run), why)


# ---- 4. conversion ----------------------------------------------------------------------------------------------------------
def test_convert():
    d = f'{TMP}/convert'
    os.makedirs(f'{d}/real')
    real = legacy_run(f'{d}/real/Run_C')
    U = f'{d}/served/Run_C'
    os.makedirs(os.path.dirname(U))
    os.symlink(real, U)                                # the deep run: a link on the review disk to the run elsewhere
    R = versions.root_of(U)
    check('the versions root of a link is beside the directory it names', R == f'{real}.versions'
          or os.path.realpath(R) == os.path.realpath(f'{real}.versions'), R)
    os.makedirs(R)
    v, _ = versions.snapshot(U, os.path.realpath(U), note='test')
    rec = versions.seal(v, note='test')
    real_symlink = os.symlink

    def broken(*a):
        raise OSError('stand-in: no pointer written')
    with patch((os, 'symlink', broken)):
        why = refuses(versions.switch, U, v, note='test', record=rec, to_work=True)
    check('a pointer that cannot be written: the run moved back, U unchanged, no work directory',
          why and 'stand-in' in why and os.readlink(U) == real and os.path.isfile(f'{real}/a.txt')
          and not os.path.exists(versions.work_path(U)) and os.symlink is real_symlink, why)
    sw = versions.switch(U, v, note='test', record=rec, to_work=True)
    W = versions.work_path(U)
    check('converted: U a pointer to the sealed clone, the legacy run is the work directory, nothing adopted',
          versions.served(U).vid == v.vid and sw['converted'] == W and sw['adopted'] is None and sw['previous'] is None
          and not os.path.exists(real) and read(f'{W}/a.txt') == 'A' * 100 and versions.verify(v)[0], sw)
    log = [json.loads(l) for l in open(os.path.join(R, 'POINTER_LOG.jsonl'))]
    check('the pointer log names the conversion', log[-1].get('converted_from') == real and log[-1].get('work') == W, log[-1])
    # a work directory already there: refused, nothing moved
    d2 = f'{TMP}/convert2'
    U2 = legacy_run(f'{d2}/Run_D')
    R2 = versions.root_of(U2)
    os.makedirs(os.path.join(R2, 'work', 'Run_D'))
    v2, _ = versions.snapshot(U2, U2, note='test')
    rec2 = versions.seal(v2, note='test')
    why = refuses(versions.switch, U2, v2, note='test', record=rec2, to_work=True)
    check('a work directory already there: refused, nothing moved', why and 'exists already' in why and os.path.isdir(U2)
          and not os.path.islink(U2) and os.path.isfile(f'{U2}/a.txt'), why)
    release_writers()


# ---- 5. stamp, pinned, moved, journal_dirs, unshare_mutable ----------------------------------------------------------------
def test_pins():
    d = f'{TMP}/pins'
    U = f'{d}/Run_P'
    os.makedirs(d)
    vs = []
    for tag in 'ABC':
        v = versions.begin(U, note='test')
        write(os.path.join(v.run, 'a.txt'), tag)
        vs.append(v)
    versions.publish(U, vs[0])
    with versions.pinned([U], note='test') as pins:
        held = vs[0].holds()
        check('pinned holds the version U names', pins[U][0] == vs[0].run and len(held) == 1, (pins, held))
        check('nothing switched: moved is empty', versions.moved(pins) == [])
        versions.publish(U, vs[1])
        check('a switch during the block: moved names U', versions.moved(pins) == [U])
    check('the hold is released after the block', vs[0].holds() == [], vs[0].holds())
    with versions.pinned([U], note='test') as pins:
        versions.publish(U, vs[2])
        versions.switch(U, vs[1], note='test back')
        check('a switch away and back during the block: moved names U', versions.moved(pins) == [U],
              (pins[U][1], versions.stamp(U)))
    L = legacy_run(f'{d}/Run_L')
    with versions.pinned([L], note='test') as pins:
        write(f'{L}/server.log', 'a log written into the legacy run')
        check('a log written into a legacy run: not a switch', versions.moved(pins) == [], (pins[L][1], versions.stamp(L)))
    # journal_dirs: the directories of the journal patterns holding a journal
    J = f'{d}/J'
    for jd in ('judgements', 'judgements_V', 'judgements_fpsample_I_20261022', 'notes'):
        write(f'{J}/{jd}/judgements.jsonl', '{}\n')
    os.makedirs(f'{J}/judgements_I')
    check('journal_dirs: every journal directory holding a judgements.jsonl, sorted',
          versions.journal_dirs(J, None, f'{d}/none') == ['judgements', 'judgements_V', 'judgements_fpsample_I_20261022'],
          versions.journal_dirs(J))
    # unshare_mutable
    a, b = f'{d}/ua', f'{d}/ub'
    write(f'{a}/judgements/judgements.jsonl', '{"x": 1}\n')
    write(f'{a}/data.csv', 'd\n')
    os.makedirs(f'{b}/judgements')
    os.link(f'{a}/judgements/judgements.jsonl', f'{b}/judgements/judgements.jsonl')
    os.link(f'{a}/data.csv', f'{b}/data.csv')
    got = versions.unshare_mutable(b)
    write(f'{b}/judgements/judgements.jsonl', '{"y": 2}\n', mode='a')
    check('unshare_mutable: the hard-linked journal gets its own inode (an append stays in its run), other files stay linked',
          got == ['judgements/judgements.jsonl'] and read(f'{a}/judgements/judgements.jsonl') == '{"x": 1}\n'
          and os.stat(f'{a}/data.csv').st_ino == os.stat(f'{b}/data.csv').st_ino, got)


# ---- 6. the per-LST placement as a version ----------------------------------------------------------------------------------
def test_place():
    from exopipe.phase3.lib.pages import night, tier_sublst
    d = f'{TMP}/place'
    lst = '10h'
    c = types.SimpleNamespace(out=f'{d}/pages_out', place_target=f'{d}/transients', sets=[])
    P = tier_sublst.P_of(c, lst)
    name = os.path.basename(P)
    U = f'{d}/transients/{name}'
    write(f'{U}/old.txt', 'the legacy placement')     # production's placed run, written in place
    write(f'{P}/candidates/V/x.txt', 'hour 1')
    write(f'{P}/loo/master.fits', 'M' * 5000)
    write(f'{P}/judgements/queue_p3s3_V_sb.txt', 'J1\n')
    write(f'{P}/judgements/judgements.jsonl', '{"j": 1}\n')
    write(f'{P}/{night.REDUCED_DIR}/sel.csv', 'reduced')

    page_ok = {'ok': True}
    from exopipe.phase3.core import validate
    real_validate, ran = validate.run, []

    def page_run_check(r):                             # the seal's check: its results kept as page_run_check keeps them
        tier_sublst._PAGE_RUN[os.path.realpath(r)] = [{'name': 'stand_in_page_run', 'passed': page_ok['ok'], 'value': None,
                                                       'detail': 'stand-in page-run test'}]
        return page_ok['ok'], f'stand-in page-run test on {r}'

    def counting(cls, path, **ctx):
        if cls == 'sublst_page_run':
            ran.append(path)
            return []
        return real_validate(cls, path, **ctx)

    def place(*extra):
        run = Run('pages', ['test'], f'{d}/records', CFG, config_sections=None)
        with patch((tier_sublst, 'page_run_check', page_run_check), (validate, 'run', counting), *extra):
            tier_sublst.stage_place(c, run, {}, lst)
        return run.units[-1]
    u = place()
    v1 = versions.served(U)
    adopted = [x for x in versions.all_versions(U) if x.vid.endswith('-adopted')]
    check('the first placement adopts the legacy run and switches U to a version holding exactly P (no reduced selections)',
          v1 is not None and len(adopted) == 1 and os.path.isfile(f'{adopted[0].run}/old.txt')
          and not os.path.exists(f'{U}/old.txt') and read(f'{U}/candidates/V/x.txt') == 'hour 1'
          and not os.path.exists(f'{U}/{night.REDUCED_DIR}') and versions.verify(v1)[0]
          and u.meta.get('version', {}).get('vid') == v1.vid, (u.status, u.reason, u.meta.get('version')))
    write(f'{P}/candidates/V/x.txt', 'hour 2')
    u = place()
    v2 = versions.served(U)
    check('the next placement: a new version, the changed file new, an unchanged one hard-linked from the served version',
          v2.vid != v1.vid and read(f'{U}/candidates/V/x.txt') == 'hour 2' and read(f'{v1.run}/candidates/V/x.txt') == 'hour 1'
          and os.stat(f'{v2.run}/loo/master.fits').st_ino == os.stat(f'{v1.run}/loo/master.fits').st_ino
          and u.meta['version']['linked_from'] == v1.run, (u.meta.get('version'), u.meta.get('rsync')))
    check('the review state of the new version is on its own inode', u.meta.get('unshared') == ['judgements/judgements.jsonl']
          and os.stat(f'{v2.run}/judgements/judgements.jsonl').st_ino != os.stat(f'{v1.run}/judgements/judgements.jsonl').st_ino,
          u.meta.get('unshared'))
    check('the previous version is unchanged (verify against its manifest)', versions.verify(v1)[0], versions.verify(v1))
    write(f'{P}/candidates/V/x.txt', 'hour 3')
    from exopipe.phase3.lib.serve import versions as V
    with patch((V, 'same_content', lambda a, b, **k: (False, 'stand-in: not the page run'))):
        u = place()
    check('a placement whose content test fails: refused, the version discarded, U unchanged',
          u.status == 'refused' and 'stand-in' in (u.reason or '') and versions.served(U).vid == v2.vid
          and len(versions.all_versions(U)) == 3, (u.status, u.reason, [x.vid for x in versions.all_versions(U)]))
    page_ok['ok'] = False
    u = place()
    page_ok['ok'] = True
    check('a page run whose own content tests fail (sublst_page_run): refused before the switch, discarded, U unchanged',
          u.status == 'refused' and 'sublst_page_run' in (u.reason or '') and versions.served(U).vid == v2.vid
          and len(versions.all_versions(U)) == 3, (u.status, u.reason))
    # the seal's page-run tests stand for the unit's output tests only while the version is the one sealed
    write(f'{P}/candidates/V/x.txt', 'hour 4')
    ran.clear()
    u = place()
    mine = [t for t in u.tests if t.get('cls') == 'sublst_page_run']
    check('the placed version unchanged since its seal: the seal\'s page-run tests (keyed by its manifest sha256) are the '
          'output\'s, not run again', u.status == 'done' and not ran and len(mine) == 1
          and mine[0]['name'] == 'stand_in_page_run' and 'the seal\'s test of version' in mine[0]['detail']
          and versions.served(U).record()['manifest_sha256'][:16] in mine[0]['detail'], (u.status, ran, mine))
    write(f'{P}/candidates/V/x.txt', 'hour 5')
    real_pv = tier_sublst.place_version

    def changed_after_seal(u, P_, U_, note):
        sw = real_pv(u, P_, U_, note)
        write(f'{sw["current"].run}/candidates/V/x.txt', 'written after the seal')
        return sw
    ran.clear()
    u = place((tier_sublst, 'place_version', changed_after_seal))
    check('a version changed since its seal: the page-run tests run again on the placed run', ran == [U]
          and any('run again' in n for n in u.notes), (ran, u.notes[-2:]))


# ---- 7. G30: the LST runs read by the cumulative build are pinned -----------------------------------------------------------
def test_g30():
    from exopipe.phase3.lib.catalog import deep_omission, grouping_stage
    from exopipe.phase3.lib.serve import scripts, sublst_build
    d = f'{TMP}/g30'
    tr = f'{d}/transients'
    os.makedirs(tr)
    lsts = ['05h', '06h']
    U5 = f'{tr}/Run_SLST05B'
    v5 = versions.begin(U5, note='test')
    for f in ('master_catalog.csv', 'pertie_detections.csv'):
        write(os.path.join(v5.run, f), 'source_id\nJ1\n')
    versions.publish(U5, v5)
    U6 = legacy_run(f'{tr}/Run_SLST06B')                # a legacy (unplaced as a version) LST run
    for f in ('master_catalog.csv', 'pertie_detections.csv'):
        write(f'{U6}/{f}', 'source_id\nJ1\n')
    seen = {}

    def merge(switch):
        def fake_run(name, args=(), env=None, **kw):
            if name == 'sublst_cumulative':
                seen['holds'] = len(v5.holds())
                write(f'{U6}/merge.log', 'a log in the legacy LST run')
                if switch:
                    v = versions.begin(U5, note='a placement during the build')
                    for f in ('master_catalog.csv', 'pertie_detections.csv'):
                        write(os.path.join(v.run, f), 'source_id\nJ2\n')
                    versions.publish(U5, v)
            return types.SimpleNamespace(ok=True, rc=0, tail='', reason='', refused=False)
        return fake_run
    plan = types.SimpleNamespace(run_name='Run_SLSTB', lsts=lsts, parity=True, U=f'{d}/out/sublst/Run_SLSTB',
                                 U_src=f'{d}/out/sublst/Run_SLSTB', out=f'{d}/out', tier='sublst')
    common = [(sublst_build, 'generation_mix', lambda l: []), (sublst_build, 'lst_run', lambda l: f'{tr}/Run_SLST{l[:2]}B'),
              (sublst_build, 'omission_table', lambda p, u: f'{d}/table.csv'),
              (grouping_stage, 'mode', lambda: types.SimpleNamespace(design_b=False)),
              (deep_omission, 'tenmin_rule_check', lambda l: None),
              (deep_omission, 'tenmin_size_gate', lambda: ('1.7', {}, 'production'))]
    with patch(*common, (scripts, 'run', merge(False))):
        u = sublst_build.build_stage(Run('serve', ['test'], f'{d}/rec', CFG, config_sections=None), plan)
    check('no switch during the merge (a log written into a legacy LST run): no G30 refusal (the next gate refuses this '
          'stand-in merge), the version read held during the merge and recorded',
          u.status == 'refused' and 'G30' not in (u.reason or '') and 'no deep omission' in (u.reason or '')
          and seen['holds'] == 1 and u.meta['lst_versions'] == {'05h': v5.run, '06h': os.path.abspath(U6)}
          and v5.holds() == [], (u.status, u.reason, seen, u.meta.get('lst_versions')))
    with patch(*common, (scripts, 'run', merge(True))):
        u = sublst_build.build_stage(Run('serve', ['test'], f'{d}/rec', CFG, config_sections=None), plan)
    check('an LST run switched while the merge read it: G30 refuses the build', u.status == 'refused'
          and 'G30' in (u.reason or '') and U5 in u.reason and v5.holds() == [] and os.path.isdir(v5.run),
          (u.status, u.reason))


# ---- 8. the deep serve on a versioned run -----------------------------------------------------------------------------------
def test_deep():
    from exopipe.phase3.lib.serve import deep, flow, sublst_build
    d = f'{TMP}/deep'
    os.makedirs(f'{d}/real')
    real = legacy_run(f'{d}/real/Run_P3SX')
    write(f'{real}/judgements_V/judgements.jsonl', '{"line": 0}\n')
    U = f'{d}/served/Run_P3SX'
    os.makedirs(os.path.dirname(U))
    os.symlink(real, U)
    rels = {'queue': 'judgements/queue_p3s3_V.txt', 'catalog': 'source_catalogs_V'}
    page = types.SimpleNamespace(key='V', kind='single', stokes='V',
                                 at=lambda f, root=None: os.path.join(root, rels[f]) if f in rels else None)
    plan = types.SimpleNamespace(U=U, U_src=U, run_name='Run_P3SX', parity=False, pages=[page], tier='deep', label='TEST',
                                 log_path=f'{U}/incremental.log', family='deep', out=f'{d}/out')
    args = types.SimpleNamespace(carry_from=None, no_hub=False, retire_plan=None)
    calls, hooks = [], {}

    def build_index(u, R, publisher=False):
        calls.append(('index', R))
        write(f'{R}/page_index/manifest.json', '{}')
        if hooks.get('index'):
            hooks.pop('index')()

    def views(run, p, state, marker=True):
        calls.append(('views', p.U, marker))
        if not hooks.get('no_queue'):
            write(f'{p.U}/judgements/queue_p3s3_V.txt', 'J1\n')
            write(f'{p.U}/source_catalogs_V/23MHz_sources_survivors.csv', 'source_id\nJ1\n')
        if hooks.get('views'):
            return hooks.pop('views')(run, p)
        return None

    def test_stage(run, p, state, name, root=None, probe=True):
        calls.append(('test', root))

    def start(run, p, state):
        calls.append(('start', os.path.realpath(p.U)))
    rec_dir = f'{d}/rec'

    def serve(stages):
        run = Run('serve', ['test'], rec_dir, CFG, config_sections=None)
        try:
            deep.run(args, run, plan, stages)
        finally:
            release_writers()
        return run
    stubs = [(flow, 'build_page_index', build_index), (flow, '_index_current', lambda R: '' if os.path.isfile(
                 f'{R}/page_index/manifest.json') else 'no index'), (deep, 'views_stage', views),
             (flow, 'test_stage', test_stage), (flow, 'start_stage', start),
             (flow, 'hub_stage', lambda run, p, s: calls.append(('hub',))),
             (flow, 'verify_stage', lambda run, p: calls.append(('verify',)))]
    default = [s for s in deep.STAGES if s not in deep.DEFAULT_OFF]
    with patch(*stubs):
        # a writer step started before the conversion (on the legacy run, no versions root yet) blocks it
        wp = writer_step(U, 4)
        ln = line_with(wp.stdout, 'LOCKED')
        with patch((CFG.serve, 'WRITER_LOCK_WAIT_S', 0.3)):
            run = serve(['convert'])
        u = run.units[-1]
        check('a writer step started before the conversion blocks it: G31 past serve.WRITER_LOCK_WAIT_S, nothing converted',
              ln and ln.split()[1] == os.path.abspath(U) and u.status == 'refused' and 'G31' in (u.reason or '')
              and os.readlink(U) == real and not versions.all_versions(U), (ln, u.status, u.reason))
        # convert: it waits for that writer; the legacy run's journal line written during the conversion (here: while the
        # page index is built) is caught up into the version; a writer step started during it waits, then writes the work
        # directory
        late = {}

        def during_convert():
            write(f'{real}/judgements_V/judgements.jsonl', '{"line": "during convert"}\n', mode='a')
            late['p'] = writer_step(U, 0)
            late['waits'] = line_with(late['p'].stderr, 'waits up to')
        hooks['index'] = during_convert
        t0 = time.time()
        run = serve(['convert'])
        waited = time.time() - t0
        wp.wait(30)
        out2, _ = late['p'].communicate(timeout=120)
        u = run.units[-1]
        v1 = versions.served(U)
        W = versions.work_path(U)
        check('the conversion waits for the writer step started before it (shared lock), then converts', waited >= 1.0
              and u.status == 'done' and wp.returncode == 0, (waited, u.status, u.reason))
        check('a writer step started during the conversion waits for it, then writes the work directory, never the version',
              late.get('waits') and out2.split() == ['LOCKED', W], (late.get('waits'), out2))
        check('convert: U a pointer to the sealed clone, the legacy run the work directory, the journal line written during the '
              'conversion caught up', u.name == 'deep_Run_P3SX_convert' and u.status == 'done' and v1 is not None
              and os.path.isdir(W) and not os.path.exists(real) and versions.verify(v1)[0]
              and read(f'{U}/judgements_V/judgements.jsonl').splitlines() == ['{"line": 0}', '{"line": "during convert"}']
              and u.meta.get('journals') == ['judgements_V'], (u.status, u.reason, u.meta.get('caught_up')))
        run = serve(['convert'])
        check('convert twice: no-input, nothing changed', run.units[-1].status == 'no-input'
              and versions.served(U).vid == v1.vid, (run.units[-1].status, run.units[-1].reason))
        # a publish: a writer step wrote the work directory; a label is posted on the served version during the publish
        write(f'{W}/a.txt', 'C' * 100)
        write(f'{W}/new_product.csv', 'n\n')
        write(f'{W}/logs/build.log', f'DONE -> {W}\n')     # the pages builder names its output directory in its log
        calls.clear()

        def during_publish(run, p):
            write(f'{v1.run}/judgements_V/judgements.jsonl', '{"line": "during publish"}\n', mode='a')
            write(f'{v1.run}/judgements_I/judgements.jsonl', '{"first": "label on page I"}\n')   # a journal begun
        hooks['views'] = during_publish
        run = serve(default)
        v2 = versions.served(U)
        names = [x.name for x in run.units]
        check('a publish: the units in order, views and tests on the snapshot without a publish marker, start on U after the '
              'switch', names[0] == 'deep_Run_P3SX_snapshot' and 'deep_Run_P3SX_switch' in names
              and [c[0] for c in calls] == ['views', 'test', 'index', 'start', 'hub', 'verify']
              and calls[0][1] == v2.run and calls[0][2] is False and calls[1][1] == v2.run and calls[3][1] == v2.run,
              (names, calls))
        check('the new version holds the work directory\'s products; the old version is unchanged',
              v2.vid != v1.vid and read(f'{U}/a.txt') == 'C' * 100 and os.path.isfile(f'{U}/new_product.csv')
              and read(f'{v1.run}/a.txt') == 'A' * 100 and versions.verify(v1)[0] and versions.verify(v2)[0],
              (v2.vid, versions.verify(v1)))
        check('the served journal copied in, the line posted on the old version during the publish caught up',
              read(f'{U}/judgements_V/judgements.jsonl').splitlines() == ['{"line": 0}', '{"line": "during convert"}',
                                                                          '{"line": "during publish"}'],
              read(f'{U}/judgements_V/judgements.jsonl'))
        sw = [x for x in run.units if x.name == 'deep_Run_P3SX_switch'][0]
        check('the switch unit is done, its journal recorded with the old version\'s as its source', sw.status == 'done'
              and any(o.get('cls') == 'judgement_journal' or o.get('class') == 'judgement_journal' for o in sw.outputs),
              (sw.status, sw.reason, sw.outputs))
        check('the first label on a page during the publish (a journal begun on the old version after the snapshot) reaches '
              'the served version', read(f'{U}/judgements_I/judgements.jsonl') == '{"first": "label on page I"}\n'
              and sw.meta.get('journals') == ['judgements_I', 'judgements_V'], sw.meta.get('journals'))
        sn = [x for x in run.units if x.name == 'deep_Run_P3SX_snapshot'][0]
        check('a log naming the work directory names U in the version (the work directory\'s own log unchanged)',
              read(f'{U}/logs/build.log') == f'DONE -> {os.path.abspath(U)}\n' and read(f'{W}/logs/build.log') ==
              f'DONE -> {W}\n' and sn.meta.get('public') == ['logs/build.log'], (sn.meta.get('public'),
                                                                                read(f'{U}/logs/build.log')))
        # a journal begun on the old version across the switch (its POST held that journal's lock, which the switch did
        # not hold), and a label on the new version's page of it after the switch: merged, the old line first
        real_cu = sublst_build.catch_up

        def cu_then_late(prev_run, *a, **k):
            got = real_cu(prev_run, *a, **k)
            write(f'{prev_run}/judgements_Q/judgements.jsonl', '{"q": "old page, across the switch"}\n')
            write(f'{U}/judgements_Q/judgements.jsonl', '{"q": "new page, after the switch"}\n')
            return got
        with patch((sublst_build, 'catch_up', cu_then_late)):
            run = serve(default)
        sw = [x for x in run.units if x.name == 'deep_Run_P3SX_switch'][0]
        check('a journal begun on the old version across the switch: merged into the served version, the old line first',
              sw.status == 'done' and read(f'{U}/judgements_Q/judgements.jsonl').splitlines() == [
                  '{"q": "old page, across the switch"}', '{"q": "new page, after the switch"}']
              and sw.meta.get('late_journals') == {'judgements_Q': 1}, (sw.status, sw.reason, sw.meta.get('late_journals')))
        v2 = versions.served(U)
        # G32: a file naming the work directory written after the snapshot (by the views stage) refuses the seal
        n, cur = len(versions.all_versions(U)), versions.served(U).vid
        hooks['views'] = lambda run, p: write(f'{p.U}/late.log', f'written by the views stage into {W}\n')
        run = serve(default)
        check('G32 in a publish: a version naming the work directory is not sealed, the snapshot discarded, U unchanged',
              any(x.status == 'refused' and 'G32' in (x.reason or '') for x in run.units)
              and versions.served(U).vid == cur and len(versions.all_versions(U)) == n,
              [(x.name, x.status, (x.reason or '')[:120]) for x in run.units])
        # G31 in a publish: a writer step holds the lock past the wait; the snapshot refuses, the coalescing slot is free
        from exopipe.phase3.lib.serve import coalesce
        wp = writer_step(U, 3)
        ln = line_with(wp.stdout, 'LOCKED')
        with patch((CFG.serve, 'WRITER_LOCK_WAIT_S', 0.3)):
            run = serve(default)
        wp.wait(30)
        slot = coalesce.slot_for(U)
        fd = slot._try_lock()
        if fd is not None:
            os.close(fd)
        check('G31 in a publish: the snapshot refuses, nothing published, the coalescing slot free (no build pending)',
              ln and ln.split()[1] == W and any(x.name == 'deep_Run_P3SX_snapshot' and x.status == 'refused'
                                                and 'G31' in (x.reason or '') for x in run.units)
              and versions.served(U).vid == cur and len(versions.all_versions(U)) == n and fd is not None
              and not slot.pending(), (ln, [(x.name, x.status, (x.reason or '')[:120]) for x in run.units]))
        # a refused stage before the switch discards the snapshot
        n = len(versions.all_versions(U))
        hooks['no_queue'] = True
        run = serve(default)
        hooks.pop('no_queue')
        check('a version without a page\'s queue: the seal refuses, the snapshot discarded, U unchanged',
              any(x.status == 'refused' and 'switch' in x.name for x in run.units) and versions.served(U).vid == v2.vid
              and len(versions.all_versions(U)) == n, ([(x.name, x.status, x.reason) for x in run.units]))
        # another serve switches U during this publish: refused, discarded
        other = versions.begin(U, note='another serve')
        write(os.path.join(other.run, 'a.txt'), 'other')
        orec = versions.seal(other, note='another serve')

        def switch_away(run, p):
            versions.switch(U, other, note='another serve', record=orec)
        hooks['views'] = switch_away
        run = serve(default)
        check('U switched by another serve during the publish: the switch refuses, the snapshot discarded',
              any(x.status == 'refused' and 'whose journals the snapshot holds' in (x.reason or '') for x in run.units)
              and versions.served(U).vid == other.vid and len(versions.all_versions(U)) == n + 1,
              [(x.name, x.status, (x.reason or '')[:120]) for x in run.units])
        versions.switch(U, v2, note='back')
        # a failed catch-up after the switch is owed; the next serve completes it first
        real_cu = sublst_build.catch_up
        state = {'n': 0}

        def failing(*a, **k):
            state['n'] += 1
            raise OSError('stand-in: the catch-up failed')
        hooks['views'] = lambda run, p: write(f'{v2.run}/judgements_V/judgements.jsonl', '{"line": "owed"}\n', mode='a')
        with patch((sublst_build, 'catch_up', failing)):
            run = serve(default)
        v3 = versions.served(U)
        owed = versions.owed_catchup(U)
        check('a failed catch-up after the switch: refused, OWED recorded with the deep journals, U on the new version',
              v3.vid != v2.vid and owed and owed['from_version'] == v2.vid and 'judgements_V' in owed['journals']
              and any('OWED' in (x.reason or '') for x in run.units) and 'start' not in [c[0] for c in calls[-1:]],
              (owed, [(x.name, x.status) for x in run.units]))
        run = serve(default)
        cu = [x for x in run.units if x.name == 'deep_Run_P3SX_catchup']
        check('the next serve completes the owed catch-up first, then publishes', cu and cu[0].status == 'done'
              and versions.owed_catchup(U) is None and '{"line": "owed"}' in read(f'{U}/judgements_V/judgements.jsonl')
              and sublst_build.catch_up is real_cu, [(x.name, x.status, (x.reason or '')[:100]) for x in run.units])
        # a partial stage selection on a versioned run refuses
        why = refuses(serve, ['views'])
        check('views without test and start on a versioned run: refused', why and 'not selected' in why, why)
        # retire for deep
        run = serve(['retire'])
        r = run.units[-1]
        check('retire runs for the deep run (a dry run: its plan written)', r.name == 'deep_Run_P3SX_retire'
              and r.status == 'done' and os.path.isfile(r.meta.get('plan', '')), (r.name, r.status, r.reason))
    # parity convert refuses
    pplan = types.SimpleNamespace(**dict(vars(plan), parity=True, U=f'{d}/parity/Run_P3SX'))
    os.makedirs(pplan.U)
    run = Run('serve', ['test'], rec_dir, CFG, config_sections=None)
    deep.run(args, run, pplan, ['convert'])
    release_writers()
    check('convert of a parity serve: refused', run.units[-1].status == 'refused' and 'parity' in run.units[-1].reason,
          (run.units[-1].status, run.units[-1].reason))


# ---- 9. cli: an injected run with --out U ---------------------------------------------------------------------------------
def test_cli_injected():
    d = f'{TMP}/cli'
    U = f'{d}/served/Run_INJ'
    v = versions.begin(U, note='test')
    write(os.path.join(v.run, 'a.txt'), 'x')
    versions.publish(U, v)
    W = versions.work_path(U)
    os.makedirs(W)
    audit = f'{d}/audit.csv'
    write(audit, 'a\n1\n')
    env = {k: x for k, x in os.environ.items() if k not in ('EXOPIPE3_INJECTED', 'P3_SNAP_INJECTED', 'EXOPIPE3_PARITY_ROOT',
                                                            'EXOPIPE3_PARITY_LOG')}
    env.update(PYTHONPATH=REPO, NUMPY_MADVISE_HUGEPAGE='0')
    r = subprocess.run([sys.executable, '-m', 'exopipe.phase3', 'status', 'register', audit, '--out', U, '--injected',
                        '--cores', 'none'], cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, timeout=300)
    g = [x for x in r.stderr.splitlines() if x.startswith('INJECTED-RUN GUARD (')]
    lockf = versions.writer_lock_file(versions.root_of(U))
    check('an injected run with --out U: its guard allows U\'s work directory and writer lock, never the version U names',
          g and g[0].split('writes allowed only under ', 1)[1].split(', ')[:2] == [W, lockf] and v.run not in g[0]
          and 'os.open refused' not in r.stderr, (r.returncode, g, r.stderr[-600:]))
    check('... its records written in the work directory, nothing into the version', os.path.isdir(f'{W}/_records')
          and versions.verify(v)[0] and sorted(os.listdir(v.run)) == ['a.txt'], (os.listdir(W), versions.verify(v)))


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    try:
        test_work()
        test_clone_content()
        test_snapshot()
        test_convert()
        test_pins()
        test_place()
        test_g30()
        test_deep()
        test_cli_injected()
    finally:
        release_writers()
        shutil.rmtree(TMP, ignore_errors=True)
    print(f'{N[0] - len(FAILS)} of {N[0]} checks passed')
    return 1 if FAILS else 0


def test_all():
    assert main() == 0


if __name__ == '__main__':
    sys.exit(main())
